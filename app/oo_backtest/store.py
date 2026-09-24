"""Saved strategies for the OO/Mesosim Backtest page (main database).

A saved strategy is the ORIGINAL UPLOADED FILE -- gzipped, byte for byte --
plus a name, optional notes, and a few facts read from parsing it (source
format, trade count, date range) so the dropdown can label it without
re-parsing.

WHY THE FILE AND NOT THE JOINED TRADES. Everything market-derived (VIX
levels, gaps, ratios) is recomputed from main.index_ohlc on every load, the
same as a fresh upload. Storing joined rows would freeze each log at the
market data of the day it was saved: the zero-filled-weekend fix of
2026-09-14 changed every Monday gap, and a stored copy would have kept the
wrong ones with nothing to say so. The raw file also keeps what the parser
retains but the page does not show (Mesosim entry_var_*), and makes each
record self-contained for the later portfolio page.

The facts stored alongside are computed by the SERVER from its own parse,
never taken from the client.

The table is created lazily and idempotently, like ticker_analysis_layouts.

capital_per_position is a DISPLAY input (the page's Avg Annual Return %,
Avg P/L % and Deployment dollars), stored so a reloaded log keeps it. It does
not touch the file or anything parsed from it; NULL means the page default.
Added after the table existed, hence the ADD COLUMN IF NOT EXISTS.
"""
from __future__ import annotations

import gzip
import hashlib
from datetime import date

TABLE = "oo_backtest_strategies"
NAME_MAX = 200
NOTES_MAX = 4000

CREATE_SQL = f"""
CREATE TABLE IF NOT EXISTS {TABLE} (
    id           SERIAL PRIMARY KEY,
    name         TEXT NOT NULL UNIQUE,
    notes        TEXT NOT NULL DEFAULT '',
    source       TEXT NOT NULL,              -- 'oo_csv' | 'mesosim_json'
    filename     TEXT NOT NULL,
    file_gz      BYTEA NOT NULL,
    file_sha256  TEXT NOT NULL,
    file_bytes   INTEGER NOT NULL,
    trade_count  INTEGER NOT NULL,
    date_min     DATE,
    date_max     DATE,
    capital_per_position DOUBLE PRECISION,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at   TIMESTAMPTZ NOT NULL DEFAULT now()
)
"""

_LIST_COLS = ("id, name, notes, source, filename, file_sha256, file_bytes, trade_count, "
              "date_min, date_max, capital_per_position, created_at, updated_at")


class NameTaken(Exception):
    """A strategy with this name exists and replace was not requested."""

    def __init__(self, name: str, existing_id: int):
        super().__init__(f'A saved strategy named "{name}" already exists.')
        self.name = name
        self.existing_id = existing_id


def clean_name(name: str | None) -> str:
    n = " ".join((name or "").split())
    if not n:
        raise ValueError("A name is required.")
    if len(n) > NAME_MAX:
        raise ValueError(f"Name is longer than {NAME_MAX} characters.")
    return n


def clean_notes(notes: str | None) -> str:
    n = (notes or "").strip()
    if len(n) > NOTES_MAX:
        raise ValueError(f"Notes are longer than {NOTES_MAX} characters.")
    return n


ADD_CAPITAL_SQL = f"ALTER TABLE {TABLE} ADD COLUMN IF NOT EXISTS capital_per_position DOUBLE PRECISION"
CAPITAL_MAX = 1e9

# The parsed frame, cached beside the file it came from. DERIVED DATA, never a
# source of truth: `parsed_for_sha` and `parsed_fingerprint` say which file and
# which parser produced it, and a miss on either re-parses. See
# app/oo_backtest/parsed_cache.py for why it is the parsed frame and not the
# joined one.
ADD_PARSED_SQL = [
    f"ALTER TABLE {TABLE} ADD COLUMN IF NOT EXISTS parsed_gz BYTEA",
    f"ALTER TABLE {TABLE} ADD COLUMN IF NOT EXISTS parsed_for_sha TEXT",
    f"ALTER TABLE {TABLE} ADD COLUMN IF NOT EXISTS parsed_fingerprint TEXT",
    f"ALTER TABLE {TABLE} ADD COLUMN IF NOT EXISTS parsed_rows INTEGER",
    f"ALTER TABLE {TABLE} ADD COLUMN IF NOT EXISTS parsed_at TIMESTAMPTZ",
]


def clean_capital(v) -> float | None:
    """None (use the page default) or a positive dollar amount."""
    if v is None or (isinstance(v, str) and not v.strip()):
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        raise ValueError(f"Capital per position must be a number, not {v!r}.")
    if not (0 < f <= CAPITAL_MAX):
        raise ValueError("Capital per position must be greater than 0.")
    return f


async def ensure_table(conn) -> None:
    await conn.execute(CREATE_SQL)
    await conn.execute(ADD_CAPITAL_SQL)
    for sql in ADD_PARSED_SQL:
        await conn.execute(sql)


async def read_parsed(pool, strategy_id: int, *, sha: str,
                      fingerprint: str) -> bytes | None:
    """The cached parse for this file and this parser, or None.

    BOTH KEYS ARE CHECKED IN SQL rather than read back and compared here, so
    a stale row simply does not match and the caller parses. There is no path
    that returns a parse made by a different parser or from a different file.
    """
    async with pool.acquire() as conn:
        await ensure_table(conn)
        return await conn.fetchval(
            f"SELECT parsed_gz FROM {TABLE} WHERE id = $1 "
            f"AND parsed_for_sha = $2 AND parsed_fingerprint = $3 "
            f"AND parsed_gz IS NOT NULL",
            strategy_id, sha, fingerprint)


async def write_parsed(pool, strategy_id: int, *, sha: str, fingerprint: str,
                       blob: bytes, rows: int) -> None:
    """Store a parse. Never touches `updated_at` — the strategy did not change.

    Guarded on the file's sha still matching: a save that replaced the file
    while this parse was running must not have a parse of the OLD file filed
    against it.
    """
    async with pool.acquire() as conn:
        await ensure_table(conn)
        await conn.execute(
            f"UPDATE {TABLE} SET parsed_gz = $2, parsed_for_sha = $3, "
            f"parsed_fingerprint = $4, parsed_rows = $5, parsed_at = now() "
            f"WHERE id = $1 AND file_sha256 = $3",
            strategy_id, blob, sha, fingerprint, int(rows))


def _row(r) -> dict:
    d = dict(r)
    for k in ("date_min", "date_max"):
        d[k] = d[k].isoformat() if d[k] else None
    for k in ("created_at", "updated_at"):
        d[k] = d[k].isoformat() if d[k] else None
    return d


async def list_strategies(pool) -> list[dict]:
    """Newest first (by last save), as the source app listed them."""
    async with pool.acquire() as conn:
        await ensure_table(conn)
        rows = await conn.fetch(f"SELECT {_LIST_COLS} FROM {TABLE} ORDER BY updated_at DESC, id DESC")
    return [_row(r) for r in rows]


async def save_strategy(pool, *, name: str, notes: str, source: str, filename: str, content: bytes,
                        trade_count: int, date_min: date | None, date_max: date | None,
                        replace: bool, capital_per_position=None) -> dict:
    """Insert, or overwrite an existing name when `replace` is set.

    Returns the saved row (without the file) plus `same_file_as`: the other
    saved names holding byte-identical content, so a duplicate is visible.
    """
    name, notes, capital = clean_name(name), clean_notes(notes), clean_capital(capital_per_position)
    sha = hashlib.sha256(content).hexdigest()
    gz = gzip.compress(content, compresslevel=6)
    async with pool.acquire() as conn:
        await ensure_table(conn)
        async with conn.transaction():
            existing = await conn.fetchrow(f"SELECT id FROM {TABLE} WHERE name = $1 FOR UPDATE", name)
            if existing and not replace:
                raise NameTaken(name, existing["id"])
            if existing:
                row = await conn.fetchrow(
                    f"""UPDATE {TABLE} SET notes=$2, source=$3, filename=$4, file_gz=$5, file_sha256=$6,
                            file_bytes=$7, trade_count=$8, date_min=$9, date_max=$10, capital_per_position=$11,
                            updated_at=now()
                        WHERE id=$1 RETURNING {_LIST_COLS}""",
                    existing["id"], notes, source, filename, gz, sha, len(content), trade_count, date_min, date_max,
                    capital)
            else:
                row = await conn.fetchrow(
                    f"""INSERT INTO {TABLE} (name, notes, source, filename, file_gz, file_sha256, file_bytes,
                            trade_count, date_min, date_max, capital_per_position)
                        VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11) RETURNING {_LIST_COLS}""",
                    name, notes, source, filename, gz, sha, len(content), trade_count, date_min, date_max, capital)
            same = await conn.fetch(f"SELECT name FROM {TABLE} WHERE file_sha256 = $1 AND id <> $2 ORDER BY name",
                                    sha, row["id"])
    out = _row(row)
    out["same_file_as"] = [r["name"] for r in same]
    return out


async def load_strategy_file(pool, strategy_id: int) -> tuple[dict, bytes] | None:
    """(metadata, original file bytes), or None if there is no such id.

    The stored hash is checked against the decompressed bytes: a record that
    no longer matches what was saved is an error, not a log to analyse.
    """
    async with pool.acquire() as conn:
        await ensure_table(conn)
        r = await conn.fetchrow(f"SELECT {_LIST_COLS}, file_gz FROM {TABLE} WHERE id = $1", strategy_id)
    if r is None:
        return None
    content = gzip.decompress(r["file_gz"])
    if hashlib.sha256(content).hexdigest() != r["file_sha256"]:
        raise ValueError(f'Saved strategy "{r["name"]}" is corrupt: its file no longer matches the stored hash.')
    meta = _row({k: r[k] for k in r.keys() if k != "file_gz"})
    return meta, content


async def set_capital(pool, strategy_id: int, capital_per_position) -> dict | None:
    """Update only the capital. updated_at is left alone: the list is ordered
    by last SAVE, and nudging a display input is not a save."""
    capital = clean_capital(capital_per_position)
    async with pool.acquire() as conn:
        await ensure_table(conn)
        r = await conn.fetchrow(f"UPDATE {TABLE} SET capital_per_position = $2 WHERE id = $1 RETURNING {_LIST_COLS}",
                                strategy_id, capital)
    return _row(r) if r else None


async def delete_strategy(pool, strategy_id: int) -> bool:
    """True if a row was deleted."""
    async with pool.acquire() as conn:
        await ensure_table(conn)
        status = await conn.execute(f"DELETE FROM {TABLE} WHERE id = $1", strategy_id)
    return status.endswith(" 1")
