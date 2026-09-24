"""The parsed trade log, kept so a portfolio does not re-parse five files.

WHY. A 2,100-trade MesoSim log is ~53 MB of JSON and takes ~3.0 s to parse on
the development machine; the VPS has measured about half that speed on
comparable work. The Backtest Portfolio page loads several saved strategies at
once, so five of them is 15 s here and nearer 30 s there -- on a page whose
whole point is adjusting allocations and looking again.

WHAT IS CACHED, AND WHAT IS NOT. The PARSED FRAME ONLY, before the market-data
join. The join is cheap (it reads a rollup this process already caches) and it
depends on `index_ohlc`, which is maintained by something else and is
backfilled: freezing a joined frame would pin a trade's VIX to whatever the
table held the day it was first loaded, and nothing would ever say so. So the
cache holds what the PARSER produced from the FILE, which is a pure function
of the two, and everything downstream still runs every time.

THE KEY IS THE FILE PLUS THE PARSER. `file_sha256` is already stored, and the
parser is identified by a fingerprint of its own source (see FINGERPRINT).
Either changing invalidates the entry. The fingerprint is derived rather than
a number to bump by hand: the failure it prevents -- a parser change that
silently keeps serving the old parse -- is one nobody would notice, because
the stale numbers are plausible. The price is a re-parse after any edit to
data_loader.py, comments included, which happens once per strategy.

THE FORMAT IS GZIPPED JSON, columnar, with the datetime columns named. Not
parquet (no pyarrow on either box) and not pickle: a pickle in a database is a
format nobody can read without this code, tied to the pandas that wrote it.
JSON round-trips through a version change, and the gate asserts the property
that actually matters -- that the PAYLOAD built from a decoded frame is
identical to the payload built from a freshly parsed one.
"""
from __future__ import annotations

import asyncio
import gzip
import hashlib
import json
import logging
import math
import time
from pathlib import Path

import pandas as pd

from app.oo_backtest import store

log = logging.getLogger(__name__)

FORMAT = 1

# Columns the parser produces as datetimes. Named rather than sniffed: a
# column that arrives as a string and leaves as a Timestamp (or the reverse)
# changes how the market join and every date filter behave, and sniffing
# would make that depend on the data rather than on the schema.
DATETIME_COLUMNS = ("date_opened", "date_closed")


def _fingerprint() -> str:
    """The parser's identity: a hash of its source.

    Derived, not declared, because a constant to bump by hand is a constant
    someone forgets -- and the symptom is a portfolio built from a parse the
    current code would no longer produce.
    """
    src = (Path(__file__).resolve().parent / "data_loader.py").read_bytes()
    return hashlib.sha256(src).hexdigest()[:16]


FINGERPRINT = _fingerprint()


def _cell(v):
    """One value, JSON-safe, with NaN and NaT flattened to null."""
    if v is None:
        return None
    if isinstance(v, float):
        return None if math.isnan(v) else v
    if isinstance(v, (pd.Timestamp,)):
        return None if pd.isna(v) else v.isoformat()
    # numpy scalars and the like
    if hasattr(v, "item"):
        try:
            v = v.item()
        except (ValueError, AttributeError):
            return str(v)
        return _cell(v)
    if isinstance(v, (str, int, bool)):
        return v
    if v is pd.NaT:
        return None
    try:
        if pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass
    return str(v)


def _json_safe(v):
    """A frame attribute, made storable. Containers are walked."""
    if isinstance(v, dict):
        return {str(k): _json_safe(x) for k, x in v.items()}
    if isinstance(v, (list, tuple, set)):
        return [_json_safe(x) for x in v]
    return _cell(v)


def encode(df: pd.DataFrame) -> bytes:
    """A parsed frame -> the bytes stored beside the file.

    `attrs` TRAVELS WITH THE COLUMNS. The parser hangs its notes there — the
    positions still open at the end of the backtest, the BacktestName the
    page suggests as a strategy name, which field the P/L was taken from, and
    every data-quality flag it raised — and the payload builder reads them
    into the `notes` block the page shows. A cache that carried only the
    columns dropped all of it silently: the same file loaded warm would have
    reported no open positions and no name. Caught by comparing PAYLOADS
    rather than frames, which is why the gate compares payloads.
    """
    cols = {}
    dt = []
    for name in df.columns:
        s = df[name]
        if pd.api.types.is_datetime64_any_dtype(s):
            dt.append(name)
            cols[name] = [None if pd.isna(v) else pd.Timestamp(v).isoformat()
                          for v in s]
        else:
            cols[name] = [_cell(v) for v in s]
    blob = {"v": FORMAT, "n": int(len(df)), "order": list(df.columns),
            "datetimes": dt, "columns": cols,
            "attrs": _json_safe(dict(df.attrs))}
    return gzip.compress(json.dumps(blob, allow_nan=False).encode("utf-8"), 6)


def decode(raw: bytes) -> pd.DataFrame:
    """The stored bytes -> the frame the parser produced.

    COLUMN ORDER IS RESTORED, not left to the dict. The payload builder reads
    a whitelist by name so order does not reach the browser, but a frame whose
    columns arrive in a different order than the parser emits is a frame that
    is not the same object, and the difference would surface somewhere else
    later.
    """
    blob = json.loads(gzip.decompress(raw).decode("utf-8"))
    if blob.get("v") != FORMAT:
        raise ValueError(f"parsed cache format {blob.get('v')!r}, expected {FORMAT}")
    df = pd.DataFrame(blob["columns"])
    for name in blob.get("datetimes", ()):
        if name in df.columns:
            df[name] = pd.to_datetime(df[name], errors="coerce")
    order = [c for c in blob.get("order", []) if c in df.columns]
    if order:
        df = df[order]
    # Restored AFTER the column selection: a DataFrame slice does not carry
    # attrs, so setting them earlier would drop them again.
    df.attrs.update(blob.get("attrs") or {})
    return df


async def load(pool, *, strategy_id: int, sha: str, content: bytes,
               filename: str, parse) -> tuple[pd.DataFrame, dict]:
    """The parsed frame for a saved strategy, from cache where possible.

    `parse` is the caller's own (content, filename) -> DataFrame, so this
    module decides WHEN to parse and the router still decides WHAT parsing
    means -- the same function a fresh upload goes through, rather than a
    second copy that could drift from it.

    Returns (frame, note). The note says which path ran and how long it took,
    and it is carried to the page: "loaded in 14 s" and "loaded in 0.3 s" are
    the difference between a cache that is working and one that is quietly
    missing every time, and nothing else would show it.
    """
    t0 = time.perf_counter()
    if pool is not None:
        try:
            raw = await store.read_parsed(pool, strategy_id, sha=sha,
                                          fingerprint=FINGERPRINT)
        except Exception as exc:                          # noqa: BLE001
            # A cache that cannot be read is a slow load, never a failed one.
            log.warning("parsed cache unreadable for strategy %s: %s",
                        strategy_id, exc)
            raw = None
        if raw:
            try:
                df = await asyncio.to_thread(decode, raw)
                return df, {"source": "cache", "seconds": time.perf_counter() - t0,
                            "rows": int(len(df))}
            except Exception as exc:                      # noqa: BLE001
                # A cache entry that will not decode is discarded and re-made;
                # it is derived data, so there is nothing to recover.
                log.warning("parsed cache for strategy %s did not decode: %s",
                            strategy_id, exc)

    df = await asyncio.to_thread(parse, content, filename)
    parsed_s = time.perf_counter() - t0
    if pool is not None:
        try:
            blob = await asyncio.to_thread(encode, df)
            await store.write_parsed(pool, strategy_id, sha=sha,
                                     fingerprint=FINGERPRINT, blob=blob,
                                     rows=len(df))
        except Exception as exc:                          # noqa: BLE001
            # Failing to STORE a parse costs the next load three seconds; it
            # does not cost this one anything, so it is logged and dropped.
            log.warning("parsed cache not written for strategy %s: %s",
                        strategy_id, exc)
    return df, {"source": "parsed", "seconds": parsed_s, "rows": int(len(df))}
