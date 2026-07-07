"""
Ticker Analysis — option-chain section (brief §5.4).

Reads the raw per-(ticker, year) parquet stores via DuckDB, applies the
universal split adjustment (mirroring build_features' chain_adj/oi views),
aggregates to JSON, and caches the result. Mounted under the same
/api/ticker-analysis prefix as ticker_analysis.py.

As-of conventions (kept consistent with daily_features / the metric layer):
  • OI (oi_raw) labeled trade_date T is the prior session's EOD position —
    the daily_features row for T reads OI at trade_date == T. So the OI
    views here filter oi_raw on trade_date == the selected date.
  • Chain (chain_eod, vol+IV) routes via feature_date = next_trading_day
    (trade_date); those views (later phase) filter on feature_date.

Split adjustment (per row, before any aggregation):
    adjusted_strike = raw_strike * COALESCE(adj_factor, 1.0)
    adjusted_count  = raw_count  / COALESCE(adj_factor, 1.0)
Applied via a LEFT JOIN on a `split_factors(trade_date, adj_factor)` relation
built from app.split_factors (vendored from build_features). See that module
for the bisect_left boundary convention.

Performance: parquet reads run in a worker thread (asyncio.to_thread) so they
don't block the event loop, and every result is cached in
ticker_analysis_chain_cache keyed by the query params. `force=1` bypasses the
cache; POST /chain/invalidate clears it.
"""
from __future__ import annotations

import asyncio
import json
import os
import re
from datetime import datetime
from typing import Optional

try:
    import duckdb
except ImportError:
    duckdb = None
import pandas as pd

from fastapi import APIRouter, Depends, Query

from app.db import get_oi_pool
from app.split_factors import make_split_factors, make_split_factor_map

router = APIRouter()

# Raw parquet stores. Absolute /data/... defaults match the VPS pipeline
# layout ({DIR}/{TICKER}/{YEAR}.parquet); override via env for other hosts.
OI_RAW_DIR = os.getenv("OI_RAW_DIR", "/data/oi_raw")
CHAIN_EOD_DIR = os.getenv("CHAIN_EOD_DIR", "/data/chain_eod")

# Bump to invalidate all cached chain payloads on the next deploy.
CHAIN_SCHEMA_VERSION = 1

_TICKER_RE = re.compile(r"^[A-Za-z0-9.\-^]{1,15}$")
_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _oi_year_path(ticker: str, year: int) -> str:
    return os.path.join(OI_RAW_DIR, ticker.upper(), f"{year}.parquet")


# ── Cache ─────────────────────────────────────────────────────────────────

async def _ensure_chain_cache_table(conn) -> None:
    await conn.execute(
        """
        CREATE TABLE IF NOT EXISTS ticker_analysis_chain_cache (
            cache_key  TEXT PRIMARY KEY,
            payload    JSONB NOT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """
    )


async def _cache_get(pool, key: str):
    async with pool.acquire() as conn:
        await _ensure_chain_cache_table(conn)
        row = await conn.fetchrow(
            "SELECT payload FROM ticker_analysis_chain_cache WHERE cache_key = $1",
            key,
        )
    if not row:
        return None
    p = row["payload"]
    return json.loads(p) if isinstance(p, str) else p


async def _cache_put(pool, key: str, payload: dict) -> None:
    async with pool.acquire() as conn:
        await _ensure_chain_cache_table(conn)
        await conn.execute(
            "INSERT INTO ticker_analysis_chain_cache (cache_key, payload) "
            "VALUES ($1, $2::jsonb) "
            "ON CONFLICT (cache_key) DO UPDATE "
            "  SET payload = EXCLUDED.payload, created_at = now()",
            key, json.dumps(payload),
        )


# ── Split factors + spot (from Postgres) ──────────────────────────────────

async def _load_splits_df(pool, ticker: str) -> pd.DataFrame:
    """underlying_ohlc.splits for one ticker as a DataFrame(trade_date, splits)
    — the same source build_features' load_splits reads."""
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT trade_date, splits FROM underlying_ohlc "
            "WHERE ticker = $1 AND splits IS NOT NULL AND splits != 0 "
            "ORDER BY trade_date",
            ticker,
        )
    return pd.DataFrame({
        "trade_date": [r["trade_date"] for r in rows],
        "splits":     [float(r["splits"]) for r in rows],
    })


async def _raw_close(pool, ticker: str, d) -> Optional[float]:
    async with pool.acquire() as conn:
        v = await conn.fetchval(
            "SELECT close FROM underlying_ohlc WHERE ticker = $1 AND trade_date = $2",
            ticker, d,
        )
    return float(v) if v is not None else None


# ── DuckDB compute (runs in a worker thread) ──────────────────────────────

def _compute_oi_profile(path: str, date_str: str, dte_min: int, dte_max: int,
                        moneyness: Optional[float], sf_df: pd.DataFrame,
                        spot_adj: Optional[float]) -> dict:
    """Adjusted OI-by-strike for one snapshot. Returns {strikes:[{strike,
    call_oi, put_oi}], ...}. Empty (not an error) when the year file or the
    snapshot date is absent."""
    if duckdb is None:
        return {"error": "duckdb not installed on this server"}
    if not os.path.isfile(path):
        return {"strikes": [], "n": 0, "empty": True}

    con = duckdb.connect()
    try:
        con.register("split_factors", sf_df)
        money_clause = ""
        if moneyness is not None and spot_adj:
            money_clause = f"WHERE ABS(strike / {spot_adj} - 1) <= {moneyness}"
        sql = f"""
            WITH adj AS (
                SELECT raw.strike * COALESCE(sf.adj_factor, 1.0)        AS strike,
                       raw.option_type                                 AS ot,
                       raw.open_interest / COALESCE(sf.adj_factor, 1.0) AS oi
                FROM (SELECT * FROM read_parquet('{path}')
                       WHERE trade_date = DATE '{date_str}') raw
                LEFT JOIN split_factors sf ON raw.trade_date = sf.trade_date::DATE
                WHERE (raw.expiration - raw.trade_date) BETWEEN {dte_min} AND {dte_max}
            )
            SELECT strike, ot, SUM(oi) AS oi
            FROM adj
            {money_clause}
            GROUP BY strike, ot
            ORDER BY strike
        """
        rows = con.execute(sql).fetchall()
    finally:
        con.close()

    by_strike: dict = {}
    for strike, ot, oi in rows:
        e = by_strike.setdefault(round(float(strike), 4), {"call_oi": 0.0, "put_oi": 0.0})
        if str(ot).upper().startswith("C"):
            e["call_oi"] += float(oi)
        else:
            e["put_oi"] += float(oi)
    strikes = [
        {"strike": k, "call_oi": round(v["call_oi"]), "put_oi": round(v["put_oi"])}
        for k, v in sorted(by_strike.items())
    ]
    return {"strikes": strikes, "n": len(strikes), "empty": len(strikes) == 0}


# ── Endpoints ─────────────────────────────────────────────────────────────

@router.get("/chain/dates")
async def chain_dates(ticker: str = Query(...), pool=Depends(get_oi_pool)):
    """Snapshot dates for the slider — daily_features rows that carry OI
    (total_oi not null), which is where oi_raw has data."""
    if not pool:
        return {"ticker": ticker, "dates": []}
    ticker = ticker.upper()
    if not _TICKER_RE.match(ticker):
        return {"error": "invalid ticker", "dates": []}
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT trade_date FROM daily_features "
            "WHERE ticker = $1 AND total_oi IS NOT NULL ORDER BY trade_date",
            ticker,
        )
    return {"ticker": ticker, "dates": [str(r["trade_date"]) for r in rows]}


@router.get("/chain/oi-profile")
async def chain_oi_profile(
    ticker: str = Query(...),
    date: str = Query(...),
    dte_min: int = Query(0),
    dte_max: int = Query(3650),
    moneyness: Optional[float] = Query(None),
    force: int = Query(0),
    pool=Depends(get_oi_pool),
):
    """Split-adjusted OI-by-strike (puts vs calls) for one snapshot."""
    if not pool:
        return {"error": "OI database not configured"}
    if duckdb is None:
        return {"error": "duckdb not installed on this server"}
    ticker = ticker.upper()
    if not _TICKER_RE.match(ticker):
        return {"error": "invalid ticker"}
    if not _DATE_RE.match(date):
        return {"error": "invalid date"}

    key = (f"oiprofile:v{CHAIN_SCHEMA_VERSION}:{ticker}:{date}:"
           f"{dte_min}:{dte_max}:{moneyness}")
    if not force:
        cached = await _cache_get(pool, key)
        if cached is not None:
            return cached

    d = datetime.strptime(date, "%Y-%m-%d").date()
    sf_df = await _load_splits_df(pool, ticker)
    factor = make_split_factor_map(sf_df, [d]).get(d, 1.0)
    raw_close = await _raw_close(pool, ticker, d)
    # Spot in the SAME adjusted units as the strikes (both scaled by adj_factor)
    # so the spot line lands among the adjusted strike cloud on historical dates.
    spot_adj = round(raw_close * factor, 4) if raw_close is not None else None

    sf_for_date = make_split_factors(sf_df, [d])   # DF(trade_date, adj_factor)
    result = await asyncio.to_thread(
        _compute_oi_profile, _oi_year_path(ticker, d.year), date,
        dte_min, dte_max, moneyness, sf_for_date, spot_adj,
    )
    result.update({
        "ticker": ticker, "date": date, "spot": spot_adj,
        "adj_factor": round(factor, 6),
        "dte_min": dte_min, "dte_max": dte_max, "moneyness": moneyness,
    })
    if "error" not in result:
        await _cache_put(pool, key, result)
    return result


@router.post("/chain/invalidate")
async def chain_invalidate(ticker: Optional[str] = Query(None), pool=Depends(get_oi_pool)):
    """Clear the chain cache — all, or just one ticker's rows."""
    if not pool:
        return {"error": "OI database not configured"}
    async with pool.acquire() as conn:
        await _ensure_chain_cache_table(conn)
        if ticker:
            await conn.execute(
                "DELETE FROM ticker_analysis_chain_cache WHERE cache_key LIKE $1",
                f"%:{ticker.upper()}:%",
            )
        else:
            await conn.execute("DELETE FROM ticker_analysis_chain_cache")
    return {"ok": True}
