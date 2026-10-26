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

A PARAMETER SWEEP (2026-10-18) is ONE saved strategy holding several files:
the row carries `sweep` (JSONB: the parameters and the variant list as
resolved when it was saved) and its files live in oo_backtest_strategy_files,
each gzipped with its own hash, in upload order. The row's own file_gz is an
empty gzip and its file_sha256 a hash of the files' names and hashes, so
`same_file_as` still finds the same sweep saved twice. A load re-parses the
files exactly as an upload does (sweep.combine); the recorded variant list is
only compared against, never trusted over the files. Deleting the row deletes
its files (ON DELETE CASCADE); saving a single file over a sweep's name
removes them.
"""
from __future__ import annotations

import gzip
import hashlib
import json
from datetime import date

TABLE = "oo_backtest_strategies"
NAME_MAX = 200
NOTES_MAX = 4000

CREATE_SQL = f"""
CREATE TABLE IF NOT EXISTS {TABLE} (
    id           SERIAL PRIMARY KEY,
    name         TEXT NOT NULL UNIQUE,
    notes        TEXT NOT NULL DEFAULT '',
    source       TEXT NOT NULL,              -- 'oo_csv' | 'oo_mcp_csv' | 'mesosim_json'
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
              "date_min, date_max, capital_per_position, split_on, split_cutoff, created_at, updated_at, "
              # A sweep's variant count; NULL for a single log.
              "CASE WHEN sweep IS NULL THEN NULL ELSE jsonb_array_length(sweep->'variants') END AS variants")

FILES_TABLE = "oo_backtest_strategy_files"
CREATE_FILES_SQL = f"""
CREATE TABLE IF NOT EXISTS {FILES_TABLE} (
    strategy_id  INTEGER NOT NULL REFERENCES {TABLE}(id) ON DELETE CASCADE,
    ord          INTEGER NOT NULL,           -- upload order
    filename     TEXT NOT NULL,
    file_gz      BYTEA NOT NULL,
    file_sha256  TEXT NOT NULL,
    file_bytes   INTEGER NOT NULL,
    PRIMARY KEY (strategy_id, ord)
)
"""
ADD_SWEEP_SQL = f"ALTER TABLE {TABLE} ADD COLUMN IF NOT EXISTS sweep JSONB"


class SweepStrategy(Exception):
    """load_strategy_file was asked for a sweep, which has no single file.
    Callers that can load a sweep use load_sweep_files; the rest refuse."""

    def __init__(self, meta: dict):
        super().__init__(f'"{meta["name"]}" is a parameter sweep of {meta.get("variants")} variants.')
        self.meta = meta


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

# The train/test split (P4, 2026-10-25): whether it is ON, and the CUTOFF --
# the out-of-sample boundary. NULL cutoff = the page computes its default
# (the entry date nearest 70% of the log's span), which is also what a
# strategy saved before these columns existed gets. The VIEW (Train / Full)
# is deliberately NOT saved: a loaded strategy always opens in Train, so test
# figures are never on screen before they are asked for. Written only by an
# explicit save (the upload's Save, or set_split from the page's Save
# button) -- never as the cutoff is moved, because moving the cutoff after
# seeing results is the way this feature is defeated.
ADD_SPLIT_SQL = [
    f"ALTER TABLE {TABLE} ADD COLUMN IF NOT EXISTS split_on BOOLEAN NOT NULL DEFAULT FALSE",
    f"ALTER TABLE {TABLE} ADD COLUMN IF NOT EXISTS split_cutoff DATE",
]


def clean_split(split_on, split_cutoff) -> tuple[bool, date | None]:
    """(on, cutoff): on is a real boolean (or the form's 'true'/'false');
    cutoff is None/'' (use the default) or an ISO date."""
    if isinstance(split_on, str):
        if split_on.strip().lower() not in ("true", "false", ""):
            raise ValueError(f"split_on must be true or false, not {split_on!r}.")
        split_on = split_on.strip().lower() == "true"
    if split_on is None:
        split_on = False
    if not isinstance(split_on, bool):
        raise ValueError(f"split_on must be true or false, not {split_on!r}.")
    if split_cutoff is None or (isinstance(split_cutoff, str) and not split_cutoff.strip()):
        return split_on, None
    if not isinstance(split_cutoff, str):
        raise ValueError(f"The cutoff must be a date (YYYY-MM-DD), not {split_cutoff!r}.")
    try:
        return split_on, date.fromisoformat(split_cutoff.strip())
    except ValueError:
        raise ValueError(f"The cutoff must be a date (YYYY-MM-DD), not {split_cutoff!r}.")

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
    await conn.execute(ADD_SWEEP_SQL)
    await conn.execute(CREATE_FILES_SQL)
    for sql in ADD_SPLIT_SQL:
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
    for k in ("date_min", "date_max", "split_cutoff"):
        if k in d:
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
                        replace: bool, capital_per_position=None, split_on=False, split_cutoff=None) -> dict:
    """Insert, or overwrite an existing name when `replace` is set.

    Returns the saved row (without the file) plus `same_file_as`: the other
    saved names holding byte-identical content, so a duplicate is visible.
    """
    name, notes, capital = clean_name(name), clean_notes(notes), clean_capital(capital_per_position)
    split_on, split_cutoff = clean_split(split_on, split_cutoff)
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
                            split_on=$12, split_cutoff=$13, sweep=NULL, updated_at=now()
                        WHERE id=$1 RETURNING {_LIST_COLS}""",
                    existing["id"], notes, source, filename, gz, sha, len(content), trade_count, date_min, date_max,
                    capital, split_on, split_cutoff)
                # A single file saved over a sweep's name: its files go.
                await conn.execute(f"DELETE FROM {FILES_TABLE} WHERE strategy_id = $1", existing["id"])
            else:
                row = await conn.fetchrow(
                    f"""INSERT INTO {TABLE} (name, notes, source, filename, file_gz, file_sha256, file_bytes,
                            trade_count, date_min, date_max, capital_per_position, split_on, split_cutoff)
                        VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13) RETURNING {_LIST_COLS}""",
                    name, notes, source, filename, gz, sha, len(content), trade_count, date_min, date_max, capital,
                    split_on, split_cutoff)
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
    if r["variants"] is not None:
        raise SweepStrategy(_row({k: r[k] for k in r.keys() if k != "file_gz"}))
    content = gzip.decompress(r["file_gz"])
    if hashlib.sha256(content).hexdigest() != r["file_sha256"]:
        raise ValueError(f'Saved strategy "{r["name"]}" is corrupt: its file no longer matches the stored hash.')
    meta = _row({k: r[k] for k in r.keys() if k != "file_gz"})
    return meta, content


def sweep_manifest(files: list[tuple[str, bytes]]) -> str:
    """The sweep's identity: its files' names and contents, in order."""
    lines = [f"{name}\t{hashlib.sha256(content).hexdigest()}" for name, content in files]
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()


async def save_sweep(pool, *, name: str, notes: str, source: str, files: list[tuple[str, bytes]], sweep: dict,
                     trade_count: int, date_min: date | None, date_max: date | None,
                     replace: bool, capital_per_position=None, split_on=False, split_cutoff=None) -> dict:
    """One saved strategy holding every file of a sweep (see the module
    docstring). Insert, or overwrite an existing name when `replace` is set;
    either way the files are written in the same transaction as the row."""
    name, notes, capital = clean_name(name), clean_notes(notes), clean_capital(capital_per_position)
    split_on, split_cutoff = clean_split(split_on, split_cutoff)
    if len(files) < 2:
        raise ValueError("A sweep is two or more files; save a single file as a normal strategy.")
    sha = sweep_manifest(files)
    total = sum(len(c) for _, c in files)
    empty_gz = gzip.compress(b"")
    meta = json.dumps(sweep)
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
                            sweep=$12::jsonb, parsed_gz=NULL, parsed_for_sha=NULL, split_on=$13, split_cutoff=$14,
                            updated_at=now()
                        WHERE id=$1 RETURNING {_LIST_COLS}""",
                    existing["id"], notes, source, f"{len(files)} files", empty_gz, sha, total, trade_count,
                    date_min, date_max, capital, meta, split_on, split_cutoff)
                await conn.execute(f"DELETE FROM {FILES_TABLE} WHERE strategy_id = $1", existing["id"])
            else:
                row = await conn.fetchrow(
                    f"""INSERT INTO {TABLE} (name, notes, source, filename, file_gz, file_sha256, file_bytes,
                            trade_count, date_min, date_max, capital_per_position, sweep, split_on, split_cutoff)
                        VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12::jsonb,$13,$14) RETURNING {_LIST_COLS}""",
                    name, notes, source, f"{len(files)} files", empty_gz, sha, total, trade_count,
                    date_min, date_max, capital, meta, split_on, split_cutoff)
            await conn.executemany(
                f"""INSERT INTO {FILES_TABLE} (strategy_id, ord, filename, file_gz, file_sha256, file_bytes)
                    VALUES ($1, $2, $3, $4, $5, $6)""",
                [(row["id"], k, fname, gzip.compress(content, compresslevel=6),
                  hashlib.sha256(content).hexdigest(), len(content)) for k, (fname, content) in enumerate(files)])
            same = await conn.fetch(f"SELECT name FROM {TABLE} WHERE file_sha256 = $1 AND id <> $2 ORDER BY name",
                                    sha, row["id"])
    out = _row(row)
    out["same_file_as"] = [r["name"] for r in same]
    return out


async def load_sweep_files(pool, strategy_id: int) -> tuple[dict, list[tuple[str, bytes]], dict] | None:
    """(metadata, [(filename, bytes)] in upload order, the sweep as recorded
    when saved), or None if there is no such id. Every file's hash is checked,
    and the set against the row's manifest: a sweep missing a file, or holding
    one that changed, is an error, not a smaller sweep to analyse."""
    async with pool.acquire() as conn:
        await ensure_table(conn)
        r = await conn.fetchrow(f"SELECT {_LIST_COLS}, sweep FROM {TABLE} WHERE id = $1", strategy_id)
        if r is None:
            return None
        rows = await conn.fetch(f"SELECT ord, filename, file_gz, file_sha256 FROM {FILES_TABLE} "
                                f"WHERE strategy_id = $1 ORDER BY ord", strategy_id)
    meta = _row({k: r[k] for k in r.keys() if k != "sweep"})
    if r["sweep"] is None:
        raise ValueError(f'Saved strategy "{meta["name"]}" is not a sweep.')
    files = []
    for f in rows:
        content = gzip.decompress(f["file_gz"])
        if hashlib.sha256(content).hexdigest() != f["file_sha256"]:
            raise ValueError(f'Saved sweep "{meta["name"]}" is corrupt: {f["filename"]} no longer matches its hash.')
        files.append((f["filename"], content))
    if sweep_manifest(files) != meta["file_sha256"]:
        raise ValueError(f'Saved sweep "{meta["name"]}" is corrupt: its files are not the ones it was saved with.')
    recorded = r["sweep"] if isinstance(r["sweep"], dict) else json.loads(r["sweep"])
    return meta, files, recorded


async def set_capital(pool, strategy_id: int, capital_per_position) -> dict | None:
    """Update only the capital. updated_at is left alone: the list is ordered
    by last SAVE, and nudging a display input is not a save."""
    capital = clean_capital(capital_per_position)
    async with pool.acquire() as conn:
        await ensure_table(conn)
        r = await conn.fetchrow(f"UPDATE {TABLE} SET capital_per_position = $2 WHERE id = $1 RETURNING {_LIST_COLS}",
                                strategy_id, capital)
    return _row(r) if r else None


async def set_split(pool, strategy_id: int, split_on, split_cutoff) -> dict | None:
    """The split, saved by the page's explicit Save on a loaded strategy.
    updated_at is left alone, as for the capital and the notes."""
    split_on, split_cutoff = clean_split(split_on, split_cutoff)
    async with pool.acquire() as conn:
        await ensure_table(conn)
        r = await conn.fetchrow(f"UPDATE {TABLE} SET split_on = $2, split_cutoff = $3 WHERE id = $1 "
                                f"RETURNING {_LIST_COLS}", strategy_id, split_on, split_cutoff)
    return _row(r) if r else None


async def set_notes(pool, strategy_id: int, notes) -> dict | None:
    """Update only the notes. updated_at is left alone, as for the capital:
    the list is ordered by when the FILE was last saved, and re-wording a
    note does not change what was saved."""
    if notes is not None and not isinstance(notes, str):
        raise ValueError("Notes must be text.")
    notes = clean_notes(notes)
    async with pool.acquire() as conn:
        await ensure_table(conn)
        r = await conn.fetchrow(f"UPDATE {TABLE} SET notes = $2 WHERE id = $1 RETURNING {_LIST_COLS}",
                                strategy_id, notes)
    return _row(r) if r else None


async def delete_strategy(pool, strategy_id: int) -> bool:
    """True if a row was deleted."""
    async with pool.acquire() as conn:
        await ensure_table(conn)
        status = await conn.execute(f"DELETE FROM {TABLE} WHERE id = $1", strategy_id)
    return status.endswith(" 1")
