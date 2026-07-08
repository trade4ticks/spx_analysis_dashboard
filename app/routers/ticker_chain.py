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
from datetime import datetime, timedelta
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
# v2: spot no longer double-adjusted by adj_factor.
# v3: spot = daily_features.spot_pc (prior-session close C_{T-1}), matching the
#     metric layer — replaces the wrong "close ≤ T from underlying_ohlc".
CHAIN_SCHEMA_VERSION = 3

_TICKER_RE = re.compile(r"^[A-Za-z0-9.\-^]{1,15}$")
_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _fill_forward(vals: list) -> list:
    """Carry the last non-null value forward (for the spot path over sessions
    where underlying_ohlc is missing). Leading nulls stay null."""
    out, last = [], None
    for v in vals:
        if v is not None:
            last = v
        out.append(last)
    return out


def _dte_clause(bands: Optional[str], dte_min: int, dte_max: int) -> str:
    """SQL boolean on (raw.expiration - raw.trade_date) for the DTE filter.

    `bands` (when provided) is a comma-separated 'lo-hi' list — the selected
    DTE buckets — turned into an OR of BETWEENs (multi-select). An empty /
    all-invalid list yields 'false' (the "None" selection → no contracts).
    When `bands` is absent, falls back to a single BETWEEN dte_min..dte_max.
    Only integer ranges are interpolated, so this is injection-safe."""
    col = "(raw.expiration - raw.trade_date)"
    if bands is not None:
        parts = []
        for seg in str(bands).split(","):
            m = re.match(r"^\s*(\d+)-(\d+)\s*$", seg)
            if m:
                parts.append(f"{col} BETWEEN {int(m.group(1))} AND {int(m.group(2))}")
        return "(" + " OR ".join(parts) + ")" if parts else "false"
    return f"{col} BETWEEN {int(dte_min)} AND {int(dte_max)}"


def _oi_year_path(ticker: str, year: int) -> str:
    return os.path.join(OI_RAW_DIR, ticker.upper(), f"{year}.parquet")


def _chain_year_path(ticker: str, year: int) -> str:
    return os.path.join(CHAIN_EOD_DIR, ticker.upper(), f"{year}.parquet")


# DTE buckets (rows) for the strike×DTE heatmap.
DTE_BUCKETS = [
    (0, 7, "0-7"), (8, 14, "8-14"), (15, 30, "15-30"), (31, 60, "31-60"),
    (61, 90, "61-90"), (91, 120, "91-120"), (121, 180, "121-180"),
    (181, 270, "181-270"), (271, 365, "271-365"), (366, 730, "1-2y"),
    (731, 10_000_000, "2y+"),
]


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


async def _spot_pc(pool, ticker: str, d) -> Optional[float]:
    """Spot = daily_features.spot_pc for (ticker, d) = the PRIOR session's
    close C_{T-1}, already split-adjusted (derived from underlying_ohlc.close).

    A chain row labeled trade_date T describes the T-1 session (OI/vol as-of
    T-1's close, knowable at T), and the whole project standardizes on prior
    close as the analysis spot — the metric layer uses spot_pc too. Reading it
    directly guarantees the chain views and the metrics share the identical
    spot by construction. NO split factor is applied (spot_pc is already
    adjusted). One spot serves both OI and volume views (same T-1 session).

    (The earlier "most recent close ≤ T from underlying_ohlc" was wrong: on a
    historical date it resolves to T's OWN 4pm close — a lookahead.)"""
    async with pool.acquire() as conn:
        v = await conn.fetchval(
            "SELECT spot_pc FROM daily_features WHERE ticker = $1 AND trade_date = $2",
            ticker, d,
        )
    return float(v) if v is not None else None


# ── DuckDB compute (runs in a worker thread) ──────────────────────────────

def _compute_oi_profile(path: str, date_str: str, dte_clause: str,
                        moneyness: Optional[float], sf_df: pd.DataFrame,
                        spot: Optional[float]) -> dict:
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
        # Moneyness compares ADJUSTED strike vs the already-adjusted spot.
        money_clause = ""
        if moneyness is not None and spot:
            money_clause = f"WHERE ABS(strike / {spot} - 1) <= {moneyness}"
        sql = f"""
            WITH adj AS (
                SELECT raw.strike * COALESCE(sf.adj_factor, 1.0)        AS strike,
                       raw.option_type                                 AS ot,
                       raw.open_interest / COALESCE(sf.adj_factor, 1.0) AS oi
                FROM (SELECT * FROM read_parquet('{path}')
                       WHERE trade_date = DATE '{date_str}') raw
                LEFT JOIN split_factors sf ON raw.trade_date = sf.trade_date::DATE
                WHERE {dte_clause}
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
    dte_bands: Optional[str] = Query(None),   # multi-select DTE buckets "lo-hi,lo-hi"
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

    dte_clause = _dte_clause(dte_bands, dte_min, dte_max)
    dtk = dte_bands if dte_bands is not None else f"{dte_min}-{dte_max}"
    key = (f"oiprofile:v{CHAIN_SCHEMA_VERSION}:{ticker}:{date}:"
           f"{dtk}:{moneyness}")
    if not force:
        cached = await _cache_get(pool, key)
        if cached is not None:
            return cached

    d = datetime.strptime(date, "%Y-%m-%d").date()
    sf_df = await _load_splits_df(pool, ticker)
    factor = make_split_factor_map(sf_df, [d]).get(d, 1.0)
    # Spot = daily_features.spot_pc (prior close, already split-adjusted) —
    # apply NO factor. adj_factor is returned only as a diagnostic.
    spot = await _spot_pc(pool, ticker, d)
    spot = round(spot, 4) if spot is not None else None

    sf_for_date = make_split_factors(sf_df, [d])   # DF(trade_date, adj_factor)
    result = await asyncio.to_thread(
        _compute_oi_profile, _oi_year_path(ticker, d.year), date,
        dte_clause, moneyness, sf_for_date, spot,
    )
    result.update({
        "ticker": ticker, "date": date, "spot": spot,
        "adj_factor": round(factor, 6),
        "dte_min": dte_min, "dte_max": dte_max, "moneyness": moneyness,
    })
    if "error" not in result:
        await _cache_put(pool, key, result)
    return result


def _compute_strike_dte(paths: list, date_str: str, date_field: str,
                        count_field: str, moneyness: Optional[float],
                        sf_df: pd.DataFrame, spot: Optional[float]) -> dict:
    """Strike × DTE-bucket grid for one snapshot. `date_field`/`count_field`
    switch between OI (trade_date / open_interest) and Vol (feature_date /
    volume). Returns a bucket-major matrix aligned to a sorted `strikes` list.
    Split factor joins on raw.trade_date (the actual session) for both."""
    if duckdb is None:
        return {"error": "duckdb not installed on this server"}
    labels = [lbl for *_, lbl in DTE_BUCKETS]
    existing = [p for p in paths if os.path.isfile(p)]
    if not existing:
        return {"dte_buckets": labels, "strikes": [], "rows": [], "max": 0, "empty": True}

    con = duckdb.connect()
    try:
        con.register("split_factors", sf_df)
        path_list = ", ".join("'" + p + "'" for p in existing)
        money = ""
        if moneyness is not None and spot:
            money = f"WHERE ABS(strike / {spot} - 1) <= {moneyness}"
        sql = f"""
            WITH adj AS (
                SELECT raw.strike * COALESCE(sf.adj_factor, 1.0)         AS strike,
                       (raw.expiration - raw.trade_date)                 AS dte,
                       raw.{count_field} / COALESCE(sf.adj_factor, 1.0)  AS cnt
                FROM (SELECT * FROM read_parquet([{path_list}])
                       WHERE {date_field} = DATE '{date_str}') raw
                LEFT JOIN split_factors sf ON raw.trade_date = sf.trade_date::DATE
            )
            SELECT strike, dte, SUM(cnt) AS v
            FROM adj
            {money}
            GROUP BY strike, dte
        """
        rows = con.execute(sql).fetchall()
    finally:
        con.close()

    def _bucket(dte: int):
        for i, (lo, hi, _) in enumerate(DTE_BUCKETS):
            if lo <= dte <= hi:
                return i
        return None

    agg: dict = {}
    strikes_set: set = set()
    for strike, dte, v in rows:
        bi = _bucket(int(dte))
        if bi is None:
            continue
        s = round(float(strike), 4)
        strikes_set.add(s)
        agg[(bi, s)] = agg.get((bi, s), 0.0) + float(v)

    strikes = sorted(strikes_set)
    sidx = {s: i for i, s in enumerate(strikes)}
    matrix = [[0 for _ in strikes] for _ in DTE_BUCKETS]
    maxv = 0.0
    for (bi, s), v in agg.items():
        matrix[bi][sidx[s]] = round(v)
        if v > maxv:
            maxv = v
    return {"dte_buckets": labels, "strikes": strikes, "rows": matrix,
            "max": round(maxv), "empty": len(strikes) == 0}


@router.get("/chain/strike-dte")
async def chain_strike_dte(
    ticker: str = Query(...),
    date: str = Query(...),
    metric: str = Query("oi"),          # 'oi' | 'vol'
    moneyness: Optional[float] = Query(None),
    force: int = Query(0),
    pool=Depends(get_oi_pool),
):
    """Split-adjusted strike × DTE-bucket heatmap for one snapshot."""
    if not pool:
        return {"error": "OI database not configured"}
    if duckdb is None:
        return {"error": "duckdb not installed on this server"}
    ticker = ticker.upper()
    if not _TICKER_RE.match(ticker):
        return {"error": "invalid ticker"}
    if not _DATE_RE.match(date):
        return {"error": "invalid date"}
    metric = "vol" if metric == "vol" else "oi"

    key = f"strikedte:v{CHAIN_SCHEMA_VERSION}:{ticker}:{date}:{metric}:{moneyness}"
    if not force:
        cached = await _cache_get(pool, key)
        if cached is not None:
            return cached

    d = datetime.strptime(date, "%Y-%m-%d").date()
    sf_df = await _load_splits_df(pool, ticker)
    factor = make_split_factor_map(sf_df, [d]).get(d, 1.0)
    spot = await _spot_pc(pool, ticker, d)
    spot = round(spot, 4) if spot is not None else None

    # Factor relation over a small back-window so the Vol path (chain rows are
    # the prior session, routed via feature_date == date) still matches its
    # raw.trade_date. OI (trade_date == date) matches d directly.
    sf_rel = make_split_factors(sf_df, [d - timedelta(days=i) for i in range(7)])

    if metric == "vol":
        paths = [_chain_year_path(ticker, d.year), _chain_year_path(ticker, d.year - 1)]
        date_field, count_field = "feature_date", "volume"
    else:
        paths = [_oi_year_path(ticker, d.year)]
        date_field, count_field = "trade_date", "open_interest"

    result = await asyncio.to_thread(
        _compute_strike_dte, paths, date, date_field, count_field,
        moneyness, sf_rel, spot,
    )
    result.update({
        "ticker": ticker, "date": date, "spot": spot, "metric": metric,
        "adj_factor": round(factor, 6), "moneyness": moneyness,
    })
    if "error" not in result:
        await _cache_put(pool, key, result)
    return result


def _compute_flow(oi_paths: list, chain_paths: list, window: list, start_s: str,
                  end_s: str, mode: str, n: int, dte_clause: str,
                  moneyness: Optional[float], sf_df: pd.DataFrame,
                  end_spot: Optional[float], need_oi: bool, need_vol: bool) -> dict:
    """Strike × time (flow map). `window` is the ordered list of session date
    strings = the columns. Rows = adjusted strikes (high→low), filtered by
    moneyness vs the end-of-window spot. Modes: oi / vol (level, sequential),
    voloi (turnover, level), doi / dvol (Δ over n sessions, signed)."""
    if duckdb is None:
        return {"error": "duckdb not installed on this server"}

    con = duckdb.connect()
    oi_map: dict = {}
    vol_map: dict = {}
    try:
        con.register("split_factors", sf_df)

        def _read(paths: list, date_field: str, count_field: str) -> dict:
            existing = [p for p in paths if os.path.isfile(p)]
            if not existing:
                return {}
            pl = ", ".join("'" + p + "'" for p in existing)
            sql = f"""
                WITH adj AS (
                    SELECT raw.{date_field}                                  AS d,
                           round(raw.strike * COALESCE(sf.adj_factor, 1.0), 2) AS strike,
                           raw.{count_field} / COALESCE(sf.adj_factor, 1.0)  AS cnt
                    FROM (SELECT * FROM read_parquet([{pl}])
                           WHERE {date_field} BETWEEN DATE '{start_s}' AND DATE '{end_s}') raw
                    LEFT JOIN split_factors sf ON raw.trade_date = sf.trade_date::DATE
                    WHERE {dte_clause}
                )
                SELECT d, strike, SUM(cnt) AS v FROM adj GROUP BY d, strike
            """
            out: dict = {}
            for d, strike, v in con.execute(sql).fetchall():
                out[(str(d), round(float(strike), 2))] = float(v)
            return out

        if need_oi:
            oi_map = _read(oi_paths, "trade_date", "open_interest")
        if need_vol:
            vol_map = _read(chain_paths, "feature_date", "volume")
    finally:
        con.close()

    strikes_set: set = set()
    if need_oi:
        strikes_set.update(s for _, s in oi_map)
    if need_vol:
        strikes_set.update(s for _, s in vol_map)
    strikes = sorted(strikes_set, reverse=True)   # high strikes on top
    if moneyness is not None and end_spot:
        strikes = [s for s in strikes if abs(s / end_spot - 1) <= moneyness]
    if not strikes:
        return {"dates": window, "strikes": [], "matrix": [], "signed": False,
                "max": 0, "empty": True}

    ncol, nrow = len(window), len(strikes)
    ridx = {s: i for i, s in enumerate(strikes)}
    cidx = {d: i for i, d in enumerate(window)}

    def _level(m: dict) -> list:
        M = [[None] * ncol for _ in range(nrow)]
        for (d, s), v in m.items():
            if d in cidx and s in ridx:
                M[ridx[s]][cidx[d]] = v
        return M

    signed = False
    if mode == "oi":
        M = _level(oi_map)
    elif mode == "vol":
        M = _level(vol_map)
    elif mode == "voloi":
        oiM, volM = _level(oi_map), _level(vol_map)
        M = [[None] * ncol for _ in range(nrow)]
        for r in range(nrow):
            for c in range(ncol):
                o, v = oiM[r][c], volM[r][c]
                M[r][c] = (v / o) if (o and o > 0 and v is not None) else None
    else:   # doi / dvol — Δ over n sessions
        signed = True
        base = _level(oi_map if mode == "doi" else vol_map)
        M = [[None] * ncol for _ in range(nrow)]
        for r in range(nrow):
            for c in range(n, ncol):
                cur, prev = base[r][c], base[r][c - n]
                if cur is not None or prev is not None:
                    M[r][c] = (cur or 0) - (prev or 0)

    if signed:
        scale = max((abs(v) for row in M for v in row if v is not None), default=0.0)
        scale = round(scale)
    else:
        scale = max((v for row in M for v in row if v is not None), default=0.0)
        scale = round(scale, 4) if mode == "voloi" else round(scale)

    def _rnd(v):
        if v is None:
            return None
        return round(v, 4) if mode == "voloi" else round(v)

    matrix = [[_rnd(v) for v in row] for row in M]
    return {"dates": window, "strikes": strikes, "matrix": matrix,
            "signed": signed, "max": scale, "empty": False}


@router.get("/chain/flow")
async def chain_flow(
    ticker: str = Query(...),
    date: str = Query(...),                 # end of the window (slider date)
    lookback: int = Query(252),             # sessions back from `date`
    mode: str = Query("oi"),                # oi | vol | voloi | doi | dvol
    n: int = Query(5),                      # sessions for the Δ modes
    dte_min: int = Query(0),
    dte_max: int = Query(3650),
    dte_bands: Optional[str] = Query(None),   # multi-select DTE buckets "lo-hi,lo-hi"
    moneyness: Optional[float] = Query(None),
    force: int = Query(0),
    pool=Depends(get_oi_pool),
):
    """Split-adjusted strike × time flow map."""
    if not pool:
        return {"error": "OI database not configured"}
    if duckdb is None:
        return {"error": "duckdb not installed on this server"}
    ticker = ticker.upper()
    if not _TICKER_RE.match(ticker):
        return {"error": "invalid ticker"}
    if not _DATE_RE.match(date):
        return {"error": "invalid date"}
    mode = mode if mode in ("oi", "vol", "voloi", "doi", "dvol") else "oi"
    lookback = max(5, min(1000, lookback))
    n = max(1, min(60, n))

    dte_clause = _dte_clause(dte_bands, dte_min, dte_max)
    dtk = dte_bands if dte_bands is not None else f"{dte_min}-{dte_max}"
    key = (f"flow:v{CHAIN_SCHEMA_VERSION}:{ticker}:{date}:{lookback}:{mode}:"
           f"{n}:{dtk}:{moneyness}")
    if not force:
        cached = await _cache_get(pool, key)
        if cached is not None:
            return cached

    d_end = datetime.strptime(date, "%Y-%m-%d").date()
    async with pool.acquire() as conn:
        drows = await conn.fetch(
            "SELECT trade_date FROM daily_features "
            "WHERE ticker = $1 AND total_oi IS NOT NULL AND trade_date <= $2 "
            "ORDER BY trade_date",
            ticker, d_end,
        )
    window = [str(r["trade_date"]) for r in drows][-lookback:]
    if not window:
        return {"ticker": ticker, "date": date, "dates": [], "strikes": [],
                "matrix": [], "empty": True, "mode": mode}

    d_start = datetime.strptime(window[0], "%Y-%m-%d").date()
    async with pool.acquire() as conn:
        srows = await conn.fetch(
            "SELECT trade_date, spot_pc FROM daily_features "
            "WHERE ticker = $1 AND trade_date BETWEEN $2 AND $3",
            ticker, d_start, d_end,
        )
    spot_by = {str(r["trade_date"]): (float(r["spot_pc"]) if r["spot_pc"] is not None else None)
               for r in srows}
    # End-of-window spot = daily_features.spot_pc at d_end (prior close).
    end_spot = await _spot_pc(pool, ticker, d_end)

    sf_df = await _load_splits_df(pool, ticker)
    # Factor relation over the whole window (+5-day back buffer so the Vol
    # path's prior-session rows still match). Factors are a step function, so
    # covering every calendar day is cheap and exact.
    span = (d_end - d_start).days + 6
    sf_rel = make_split_factors(sf_df, [d_start - timedelta(days=5) + timedelta(days=i)
                                        for i in range(span)])

    need_oi = mode in ("oi", "doi", "voloi")
    need_vol = mode in ("vol", "dvol", "voloi")
    years = list(range(d_start.year, d_end.year + 1))
    oi_paths = [_oi_year_path(ticker, y) for y in years]
    chain_paths = [_chain_year_path(ticker, y) for y in [years[0] - 1] + years]

    result = await asyncio.to_thread(
        _compute_flow, oi_paths, chain_paths, window, window[0], window[-1],
        mode, n, dte_clause, moneyness, sf_rel, end_spot, need_oi, need_vol,
    )
    result["spots"] = _fill_forward([spot_by.get(dd) for dd in result.get("dates", window)])
    result.update({
        "ticker": ticker, "date": date, "mode": mode, "n": n,
        "lookback": lookback, "moneyness": moneyness, "spot": end_spot,
    })
    if "error" not in result:
        await _cache_put(pool, key, result)
    return result


def _compute_surface(paths: list, window: list, start_s: str, end_s: str,
                     date_field: str, count_field: str, dte_clause: str,
                     moneyness: Optional[float], sf_df: pd.DataFrame,
                     end_spot: Optional[float]) -> dict:
    """Strike × time surface with SIGNED height: net = calls − puts (adjusted).
    Rows = adjusted strikes (moneyness-filtered), cols = window sessions. One
    metric (OI or Vol), aggregated across the DTE band."""
    if duckdb is None:
        return {"error": "duckdb not installed on this server"}

    net: dict = {}
    con = duckdb.connect()
    try:
        con.register("split_factors", sf_df)
        existing = [p for p in paths if os.path.isfile(p)]
        if existing:
            pl = ", ".join("'" + p + "'" for p in existing)
            sql = f"""
                WITH adj AS (
                    SELECT raw.{date_field}                                  AS d,
                           round(raw.strike * COALESCE(sf.adj_factor, 1.0), 2) AS strike,
                           raw.option_type                                  AS ot,
                           raw.{count_field} / COALESCE(sf.adj_factor, 1.0) AS cnt
                    FROM (SELECT * FROM read_parquet([{pl}])
                           WHERE {date_field} BETWEEN DATE '{start_s}' AND DATE '{end_s}') raw
                    LEFT JOIN split_factors sf ON raw.trade_date = sf.trade_date::DATE
                    WHERE {dte_clause}
                )
                SELECT d, strike, ot, SUM(cnt) AS v FROM adj GROUP BY d, strike, ot
            """
            for d, strike, ot, v in con.execute(sql).fetchall():
                k = (str(d), round(float(strike), 2))
                signed = float(v) if str(ot).upper().startswith("C") else -float(v)
                net[k] = net.get(k, 0.0) + signed
    finally:
        con.close()

    strikes = sorted({s for _, s in net}, reverse=True)
    if moneyness is not None and end_spot:
        strikes = [s for s in strikes if abs(s / end_spot - 1) <= moneyness]
    if not strikes:
        return {"dates": window, "strikes": [], "matrix": [], "max": 0,
                "empty": True, "signed": True}

    ncol, nrow = len(window), len(strikes)
    ridx = {s: i for i, s in enumerate(strikes)}
    cidx = {d: i for i, d in enumerate(window)}
    M = [[None] * ncol for _ in range(nrow)]
    for (d, s), v in net.items():
        if d in cidx and s in ridx:
            M[ridx[s]][cidx[d]] = v

    absmax = max((abs(v) for row in M for v in row if v is not None), default=0.0)
    matrix = [[(round(v) if v is not None else None) for v in row] for row in M]
    return {"dates": window, "strikes": strikes, "matrix": matrix,
            "max": round(absmax), "signed": True, "empty": False}


@router.get("/chain/surface")
async def chain_surface(
    ticker: str = Query(...),
    date: str = Query(...),
    lookback: int = Query(126),
    metric: str = Query("oi"),              # oi | vol
    dte_min: int = Query(0),
    dte_max: int = Query(3650),
    dte_bands: Optional[str] = Query(None),   # multi-select DTE buckets "lo-hi,lo-hi"
    moneyness: Optional[float] = Query(None),
    force: int = Query(0),
    pool=Depends(get_oi_pool),
):
    """Split-adjusted signed (calls − puts) strike × time surface."""
    if not pool:
        return {"error": "OI database not configured"}
    if duckdb is None:
        return {"error": "duckdb not installed on this server"}
    ticker = ticker.upper()
    if not _TICKER_RE.match(ticker):
        return {"error": "invalid ticker"}
    if not _DATE_RE.match(date):
        return {"error": "invalid date"}
    metric = "vol" if metric == "vol" else "oi"
    lookback = max(5, min(1000, lookback))

    dte_clause = _dte_clause(dte_bands, dte_min, dte_max)
    dtk = dte_bands if dte_bands is not None else f"{dte_min}-{dte_max}"
    key = (f"surface:v{CHAIN_SCHEMA_VERSION}:{ticker}:{date}:{lookback}:{metric}:"
           f"{dtk}:{moneyness}")
    if not force:
        cached = await _cache_get(pool, key)
        if cached is not None:
            return cached

    d_end = datetime.strptime(date, "%Y-%m-%d").date()
    async with pool.acquire() as conn:
        drows = await conn.fetch(
            "SELECT trade_date FROM daily_features "
            "WHERE ticker = $1 AND total_oi IS NOT NULL AND trade_date <= $2 "
            "ORDER BY trade_date",
            ticker, d_end,
        )
    window = [str(r["trade_date"]) for r in drows][-lookback:]
    if not window:
        return {"ticker": ticker, "date": date, "dates": [], "strikes": [],
                "matrix": [], "empty": True, "metric": metric}

    d_start = datetime.strptime(window[0], "%Y-%m-%d").date()
    async with pool.acquire() as conn:
        srows = await conn.fetch(
            "SELECT trade_date, spot_pc FROM daily_features "
            "WHERE ticker = $1 AND trade_date BETWEEN $2 AND $3",
            ticker, d_start, d_end,
        )
    spot_by = {str(r["trade_date"]): (float(r["spot_pc"]) if r["spot_pc"] is not None else None)
               for r in srows}
    end_spot = await _spot_pc(pool, ticker, d_end)

    sf_df = await _load_splits_df(pool, ticker)
    span = (d_end - d_start).days + 6
    sf_rel = make_split_factors(sf_df, [d_start - timedelta(days=5) + timedelta(days=i)
                                        for i in range(span)])

    years = list(range(d_start.year, d_end.year + 1))
    if metric == "vol":
        paths = [_chain_year_path(ticker, y) for y in [years[0] - 1] + years]
        date_field, count_field = "feature_date", "volume"
    else:
        paths = [_oi_year_path(ticker, y) for y in years]
        date_field, count_field = "trade_date", "open_interest"

    result = await asyncio.to_thread(
        _compute_surface, paths, window, window[0], window[-1],
        date_field, count_field, dte_clause, moneyness, sf_rel, end_spot,
    )
    result["spots"] = _fill_forward([spot_by.get(dd) for dd in result.get("dates", window)])
    result.update({
        "ticker": ticker, "date": date, "metric": metric,
        "lookback": lookback, "moneyness": moneyness, "spot": end_spot,
    })
    if "error" not in result:
        await _cache_put(pool, key, result)
    return result


def _compute_doi_profile(paths: list, date_now: str, date_prev: str,
                         dte_clause: str, moneyness: Optional[float],
                         sf_df: pd.DataFrame, spot: Optional[float]) -> dict:
    """Signed ΔOI per strike over an N-session (overnight) window:
    OI[now] − OI[prev], adjusted, aggregated across the DTE band."""
    if duckdb is None:
        return {"error": "duckdb not installed on this server"}
    existing = [p for p in paths if os.path.isfile(p)]
    if not existing:
        return {"strikes": [], "empty": True}
    con = duckdb.connect()
    try:
        con.register("split_factors", sf_df)
        pl = ", ".join("'" + p + "'" for p in existing)
        sql = f"""
            WITH adj AS (
                SELECT raw.trade_date                                    AS d,
                       round(raw.strike * COALESCE(sf.adj_factor, 1.0), 2) AS strike,
                       raw.open_interest / COALESCE(sf.adj_factor, 1.0)  AS oi
                FROM (SELECT * FROM read_parquet([{pl}])
                       WHERE trade_date IN (DATE '{date_now}', DATE '{date_prev}')) raw
                LEFT JOIN split_factors sf ON raw.trade_date = sf.trade_date::DATE
                WHERE {dte_clause}
            )
            SELECT d, strike, SUM(oi) AS oi FROM adj GROUP BY d, strike
        """
        rows = con.execute(sql).fetchall()
    finally:
        con.close()
    now: dict = {}
    prev: dict = {}
    for d, strike, oi in rows:
        s = round(float(strike), 2)
        (now if str(d) == date_now else prev)[s] = float(oi)
    strikes = sorted(set(now) | set(prev))
    if moneyness is not None and spot:
        strikes = [s for s in strikes if abs(s / spot - 1) <= moneyness]
    out = [{"strike": s, "doi": round(now.get(s, 0.0) - prev.get(s, 0.0))} for s in strikes]
    return {"strikes": out, "empty": len(out) == 0}


def _compute_vol_oi(oi_paths: list, chain_paths: list, date: str, dte_clause: str, moneyness: Optional[float], sf_df: pd.DataFrame,
                    spot: Optional[float]) -> dict:
    """Per-strike OI (oi_raw @ trade_date) and today's volume (chain_eod @
    feature_date), adjusted, aggregated across the DTE band — for the
    vol-vs-OI scatter."""
    if duckdb is None:
        return {"error": "duckdb not installed on this server"}
    con = duckdb.connect()
    oi_map: dict = {}
    vol_map: dict = {}
    try:
        con.register("split_factors", sf_df)

        def _rd(paths, date_field, count_field):
            ex = [p for p in paths if os.path.isfile(p)]
            if not ex:
                return {}
            pl = ", ".join("'" + p + "'" for p in ex)
            sql = f"""
                WITH adj AS (
                    SELECT round(raw.strike * COALESCE(sf.adj_factor, 1.0), 2) AS strike,
                           raw.{count_field} / COALESCE(sf.adj_factor, 1.0)  AS cnt
                    FROM (SELECT * FROM read_parquet([{pl}])
                           WHERE {date_field} = DATE '{date}') raw
                    LEFT JOIN split_factors sf ON raw.trade_date = sf.trade_date::DATE
                    WHERE {dte_clause}
                )
                SELECT strike, SUM(cnt) AS v FROM adj GROUP BY strike
            """
            return {round(float(s), 2): float(v) for s, v in con.execute(sql).fetchall()}

        oi_map = _rd(oi_paths, "trade_date", "open_interest")
        vol_map = _rd(chain_paths, "feature_date", "volume")
    finally:
        con.close()
    strikes = sorted(set(oi_map) | set(vol_map))
    if moneyness is not None and spot:
        strikes = [s for s in strikes if abs(s / spot - 1) <= moneyness]
    pts = [{"strike": s, "oi": round(oi_map.get(s, 0.0)), "vol": round(vol_map.get(s, 0.0))}
           for s in strikes]
    return {"points": pts, "empty": len(pts) == 0}


@router.get("/chain/doi-profile")
async def chain_doi_profile(
    ticker: str = Query(...),
    date: str = Query(...),
    n: int = Query(5),
    dte_min: int = Query(0),
    dte_max: int = Query(3650),
    dte_bands: Optional[str] = Query(None),   # multi-select DTE buckets "lo-hi,lo-hi"
    moneyness: Optional[float] = Query(None),
    force: int = Query(0),
    pool=Depends(get_oi_pool),
):
    """Signed ΔOI-by-strike over an N-session overnight window."""
    if not pool:
        return {"error": "OI database not configured"}
    if duckdb is None:
        return {"error": "duckdb not installed on this server"}
    ticker = ticker.upper()
    if not _TICKER_RE.match(ticker) or not _DATE_RE.match(date):
        return {"error": "invalid ticker/date"}
    n = max(1, min(60, n))

    dte_clause = _dte_clause(dte_bands, dte_min, dte_max)
    dtk = dte_bands if dte_bands is not None else f"{dte_min}-{dte_max}"
    key = f"doiprofile:v{CHAIN_SCHEMA_VERSION}:{ticker}:{date}:{n}:{dtk}:{moneyness}"
    if not force:
        cached = await _cache_get(pool, key)
        if cached is not None:
            return cached

    d_end = datetime.strptime(date, "%Y-%m-%d").date()
    async with pool.acquire() as conn:
        drows = await conn.fetch(
            "SELECT trade_date FROM daily_features "
            "WHERE ticker = $1 AND total_oi IS NOT NULL AND trade_date <= $2 "
            "ORDER BY trade_date",
            ticker, d_end,
        )
    dates = [str(r["trade_date"]) for r in drows]
    if len(dates) < n + 1:
        return {"ticker": ticker, "date": date, "strikes": [], "empty": True, "n": n}
    date_prev = dates[-(n + 1)]

    spot = await _spot_pc(pool, ticker, d_end)
    spot = round(spot, 4) if spot is not None else None
    d_prev = datetime.strptime(date_prev, "%Y-%m-%d").date()
    sf_df = await _load_splits_df(pool, ticker)
    sf_rel = make_split_factors(sf_df, [d_prev, d_end])

    years = sorted({d_prev.year, d_end.year})
    paths = [_oi_year_path(ticker, y) for y in years]
    result = await asyncio.to_thread(
        _compute_doi_profile, paths, date, date_prev, dte_clause,
        moneyness, sf_rel, spot,
    )
    result.update({"ticker": ticker, "date": date, "date_prev": date_prev,
                   "n": n, "spot": spot, "moneyness": moneyness})
    if "error" not in result:
        await _cache_put(pool, key, result)
    return result


@router.get("/chain/vol-oi")
async def chain_vol_oi(
    ticker: str = Query(...),
    date: str = Query(...),
    dte_min: int = Query(0),
    dte_max: int = Query(3650),
    dte_bands: Optional[str] = Query(None),   # multi-select DTE buckets "lo-hi,lo-hi"
    moneyness: Optional[float] = Query(None),
    force: int = Query(0),
    pool=Depends(get_oi_pool),
):
    """Per-strike volume-vs-OI (for the fresh-activity scatter)."""
    if not pool:
        return {"error": "OI database not configured"}
    if duckdb is None:
        return {"error": "duckdb not installed on this server"}
    ticker = ticker.upper()
    if not _TICKER_RE.match(ticker) or not _DATE_RE.match(date):
        return {"error": "invalid ticker/date"}

    dte_clause = _dte_clause(dte_bands, dte_min, dte_max)
    dtk = dte_bands if dte_bands is not None else f"{dte_min}-{dte_max}"
    key = f"voloi:v{CHAIN_SCHEMA_VERSION}:{ticker}:{date}:{dtk}:{moneyness}"
    if not force:
        cached = await _cache_get(pool, key)
        if cached is not None:
            return cached

    d = datetime.strptime(date, "%Y-%m-%d").date()
    spot = await _spot_pc(pool, ticker, d)
    spot = round(spot, 4) if spot is not None else None
    sf_df = await _load_splits_df(pool, ticker)
    sf_rel = make_split_factors(sf_df, [d - timedelta(days=i) for i in range(7)])
    oi_paths = [_oi_year_path(ticker, d.year)]
    chain_paths = [_chain_year_path(ticker, d.year), _chain_year_path(ticker, d.year - 1)]

    result = await asyncio.to_thread(
        _compute_vol_oi, oi_paths, chain_paths, date, dte_clause,
        moneyness, sf_rel, spot,
    )
    result.update({"ticker": ticker, "date": date, "spot": spot, "moneyness": moneyness})
    if "error" not in result:
        await _cache_put(pool, key, result)
    return result


def _compute_iv_smile(chain_paths: list, date: str, dte_clause: str,
                      moneyness: Optional[float], sf_df: pd.DataFrame,
                      spot: Optional[float]) -> dict:
    """IV vs (adjusted) strike at one snapshot — avg implied_vol per
    (strike, option_type) over the DTE band. call_iv / put_iv per strike."""
    if duckdb is None:
        return {"error": "duckdb not installed on this server"}
    ex = [p for p in chain_paths if os.path.isfile(p)]
    if not ex:
        return {"strikes": [], "empty": True}
    con = duckdb.connect()
    try:
        con.register("split_factors", sf_df)
        pl = ", ".join("'" + p + "'" for p in ex)
        sql = f"""
            WITH adj AS (
                SELECT round(raw.strike * COALESCE(sf.adj_factor, 1.0), 2) AS strike,
                       raw.option_type AS ot, raw.implied_vol AS iv
                FROM (SELECT * FROM read_parquet([{pl}])
                       WHERE feature_date = DATE '{date}' AND implied_vol IS NOT NULL) raw
                LEFT JOIN split_factors sf ON raw.trade_date = sf.trade_date::DATE
                WHERE {dte_clause}
            )
            SELECT strike, ot, AVG(iv) AS iv FROM adj GROUP BY strike, ot
        """
        rows = con.execute(sql).fetchall()
    finally:
        con.close()
    by: dict = {}
    for strike, ot, iv in rows:
        s = round(float(strike), 2)
        e = by.setdefault(s, {})
        if str(ot).upper().startswith("C"):
            e["call_iv"] = round(float(iv), 4)
        else:
            e["put_iv"] = round(float(iv), 4)
    strikes = sorted(by)
    if moneyness is not None and spot:
        strikes = [s for s in strikes if abs(s / spot - 1) <= moneyness]
    out = [{"strike": s, "call_iv": by[s].get("call_iv"), "put_iv": by[s].get("put_iv")}
           for s in strikes]
    return {"strikes": out, "empty": len(out) == 0}


_IV_TERM_TENORS = [7, 30, 90]
_IV_TERM_BANDS = {7: (3, 14), 30: (20, 45), 90: (60, 120)}


def _compute_iv_term(chain_paths: list, start_s: str, end_s: str, window: list) -> dict:
    """ATM IV (nearest |delta|→0.5 within each tenor's DTE band) per session,
    for the 7/30/90-day tenors. No split adjustment needed (IV is a scalar
    selected by delta/DTE, not by strike)."""
    if duckdb is None:
        return {"error": "duckdb not installed on this server"}
    ex = [p for p in chain_paths if os.path.isfile(p)]
    if not ex:
        return {"dates": window, "tenors": _IV_TERM_TENORS,
                "series": {str(t): [] for t in _IV_TERM_TENORS}, "empty": True}
    con = duckdb.connect()
    series: dict = {t: {} for t in _IV_TERM_TENORS}
    try:
        pl = ", ".join("'" + p + "'" for p in ex)
        for t in _IV_TERM_TENORS:
            lo, hi = _IV_TERM_BANDS[t]
            sql = f"""
                WITH r AS (
                    SELECT feature_date AS d, implied_vol AS iv,
                           row_number() OVER (PARTITION BY feature_date
                                              ORDER BY ABS(ABS(delta) - 0.5)) AS rn
                    FROM (SELECT * FROM read_parquet([{pl}])
                           WHERE feature_date BETWEEN DATE '{start_s}' AND DATE '{end_s}'
                             AND implied_vol IS NOT NULL AND delta IS NOT NULL
                             AND (expiration - trade_date) BETWEEN {lo} AND {hi}
                             AND ABS(delta) BETWEEN 0.3 AND 0.7) sub
                )
                SELECT d, iv FROM r WHERE rn = 1
            """
            for d, iv in con.execute(sql).fetchall():
                series[t][str(d)] = round(float(iv), 4)
    finally:
        con.close()
    out = {str(t): [series[t].get(dd) for dd in window] for t in _IV_TERM_TENORS}
    empty = all(all(v is None for v in vals) for vals in out.values())
    return {"dates": window, "tenors": _IV_TERM_TENORS, "series": out, "empty": empty}


@router.get("/chain/iv-smile")
async def chain_iv_smile(
    ticker: str = Query(...),
    date: str = Query(...),
    dte_min: int = Query(0),
    dte_max: int = Query(3650),
    dte_bands: Optional[str] = Query(None),   # multi-select DTE buckets "lo-hi,lo-hi"
    moneyness: Optional[float] = Query(None),
    force: int = Query(0),
    pool=Depends(get_oi_pool),
):
    """IV smile (per snapshot)."""
    if not pool:
        return {"error": "OI database not configured"}
    if duckdb is None:
        return {"error": "duckdb not installed on this server"}
    ticker = ticker.upper()
    if not _TICKER_RE.match(ticker) or not _DATE_RE.match(date):
        return {"error": "invalid ticker/date"}

    dte_clause = _dte_clause(dte_bands, dte_min, dte_max)
    dtk = dte_bands if dte_bands is not None else f"{dte_min}-{dte_max}"
    key = f"ivsmile:v{CHAIN_SCHEMA_VERSION}:{ticker}:{date}:{dtk}:{moneyness}"
    if not force:
        cached = await _cache_get(pool, key)
        if cached is not None:
            return cached

    d = datetime.strptime(date, "%Y-%m-%d").date()
    spot = await _spot_pc(pool, ticker, d)
    spot = round(spot, 4) if spot is not None else None
    sf_df = await _load_splits_df(pool, ticker)
    sf_rel = make_split_factors(sf_df, [d - timedelta(days=i) for i in range(7)])
    chain_paths = [_chain_year_path(ticker, d.year), _chain_year_path(ticker, d.year - 1)]

    result = await asyncio.to_thread(
        _compute_iv_smile, chain_paths, date, dte_clause, moneyness, sf_rel, spot,
    )
    result.update({"ticker": ticker, "date": date, "spot": spot, "moneyness": moneyness})
    if "error" not in result:
        await _cache_put(pool, key, result)
    return result


@router.get("/chain/iv-term")
async def chain_iv_term(
    ticker: str = Query(...),
    date: str = Query(...),
    lookback: int = Query(252),
    force: int = Query(0),
    pool=Depends(get_oi_pool),
):
    """IV term structure over time — ATM IV at the 7/30/90-day tenors."""
    if not pool:
        return {"error": "OI database not configured"}
    if duckdb is None:
        return {"error": "duckdb not installed on this server"}
    ticker = ticker.upper()
    if not _TICKER_RE.match(ticker) or not _DATE_RE.match(date):
        return {"error": "invalid ticker/date"}
    lookback = max(5, min(1000, lookback))

    key = f"ivterm:v{CHAIN_SCHEMA_VERSION}:{ticker}:{date}:{lookback}"
    if not force:
        cached = await _cache_get(pool, key)
        if cached is not None:
            return cached

    d_end = datetime.strptime(date, "%Y-%m-%d").date()
    async with pool.acquire() as conn:
        drows = await conn.fetch(
            "SELECT trade_date FROM daily_features "
            "WHERE ticker = $1 AND total_oi IS NOT NULL AND trade_date <= $2 "
            "ORDER BY trade_date",
            ticker, d_end,
        )
    window = [str(r["trade_date"]) for r in drows][-lookback:]
    if not window:
        return {"ticker": ticker, "date": date, "dates": [], "series": {}, "empty": True}
    d_start = datetime.strptime(window[0], "%Y-%m-%d").date()
    years = list(range(d_start.year, d_end.year + 1))
    chain_paths = [_chain_year_path(ticker, y) for y in [years[0] - 1] + years]

    result = await asyncio.to_thread(
        _compute_iv_term, chain_paths, window[0], window[-1], window,
    )
    result.update({"ticker": ticker, "date": date, "lookback": lookback})
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
