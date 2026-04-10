"""
Raw (non-interpolated) data endpoints.

Reads parquet files directly via DuckDB from the options data folder.
Folder structure: {PARQUET_BASE}/{trade_date_YYYYMMDD}/{expiration_YYYYMMDD}/{settlement}.parquet

GET /api/raw/expirations   — available expirations for a trade date
GET /api/raw/skew          — IV vs strike for one or more expirations (OTM puts + calls)
GET /api/raw/term          — IV vs expiration for selected strikes (auto-OTM)
GET /api/raw/historical    — IV time-series across trade dates for a fixed expiration + strikes
"""
import glob as globmod
import os
import re
from datetime import date as date_type

try:
    import duckdb
except ImportError:
    duckdb = None

from fastapi import APIRouter, Query, HTTPException

router = APIRouter(tags=["raw"])

PARQUET_BASE = os.getenv("PARQUET_BASE", "/data/spx_options")


def _require_duckdb():
    if duckdb is None:
        raise HTTPException(501, "duckdb not installed on this server")


def _validate_date_folder(d: str) -> str:
    """Accept YYYY-MM-DD or YYYYMMDD, return YYYYMMDD for folder lookup."""
    clean = d.replace("-", "")
    if not re.match(r"^\d{8}$", clean):
        raise HTTPException(400, f"Invalid date: {d}")
    return clean


def _to_iso(d: str) -> str:
    """YYYYMMDD -> YYYY-MM-DD."""
    return f"{d[:4]}-{d[4:6]}-{d[6:8]}"


def _to_date(d: str) -> date_type:
    """YYYYMMDD -> date."""
    return date_type(int(d[:4]), int(d[4:6]), int(d[6:8]))


def _validate_settlement(s: str) -> str:
    s = s.upper()
    if s not in ("AM", "PM"):
        raise HTTPException(400, "settlement must be AM or PM")
    return s


def _validate_time(t: str) -> str:
    if not re.match(r"^\d{2}:\d{2}(:\d{2})?$", t):
        raise HTTPException(400, f"Invalid time: {t}")
    return t if len(t) > 5 else t + ":00"


def _parse_strikes(s: str) -> list[float]:
    try:
        out = [float(x.strip()) for x in s.split(",") if x.strip()]
    except ValueError:
        raise HTTPException(400, "strikes must be numeric")
    if not out:
        raise HTTPException(400, "strikes required")
    return out


def _duckdb_query(sql: str):
    """Run a DuckDB query, return rows. Translates errors to 500 with detail."""
    con = duckdb.connect()
    try:
        return con.execute(sql).fetchall()
    except Exception as e:
        raise HTTPException(500, f"DuckDB query error: {e}")
    finally:
        con.close()


# ── Expirations ──────────────────────────────────────────────────────────────

@router.get("/expirations")
async def get_expirations(
    date: str = Query(..., description="Trade date YYYY-MM-DD"),
) -> dict:
    """List available expirations, settlements, and underlying price."""
    _require_duckdb()
    date_folder = _validate_date_folder(date)
    base = os.path.join(PARQUET_BASE, date_folder)

    if not os.path.isdir(base):
        return {"date": date, "expirations": [], "underlying": None}

    trade_d = _to_date(date_folder)
    results = []
    first_file = None

    for name in sorted(os.listdir(base)):
        path = os.path.join(base, name)
        if not os.path.isdir(path) or not re.match(r"^\d{8}$", name):
            continue
        settlements = sorted(
            f.replace(".parquet", "")
            for f in os.listdir(path)
            if f.endswith(".parquet")
        )
        if not settlements:
            continue
        exp_d = _to_date(name)
        dte = (exp_d - trade_d).days
        if dte < 0:
            continue
        results.append({
            "expiration": _to_iso(name),
            "dte": dte,
            "settlements": settlements,
        })
        if first_file is None:
            for s in ("PM", "AM"):
                candidate = os.path.join(path, f"{s}.parquet")
                if os.path.isfile(candidate):
                    first_file = candidate
                    break

    underlying = None
    if first_file:
        try:
            rows = _duckdb_query(
                f"SELECT underlying_price "
                f"FROM read_parquet('{first_file}') LIMIT 1"
            )
            if rows:
                underlying = rows[0][0]
        except Exception:
            pass  # non-critical — just means ATM button won't pre-fill

    return {"date": date, "expirations": results, "underlying": underlying}


# ── Raw skew ─────────────────────────────────────────────────────────────────

@router.get("/skew")
async def get_raw_skew(
    date:         str  = Query(...),
    expirations:  str  = Query(..., description="Comma-separated YYYY-MM-DD"),
    time:         str  = Query("15:45"),
    settlement:   str  = Query("PM"),
    filter_flags: bool = Query(True),
    x_axis:       str  = Query("strike", description="strike | moneyness"),
) -> dict:
    """IV vs strike (OTM puts + calls) for selected expirations."""
    _require_duckdb()
    date_folder = _validate_date_folder(date)
    settlement  = _validate_settlement(settlement)
    time_str    = _validate_time(time)
    exp_list    = [e.strip() for e in expirations.split(",") if e.strip()]
    if not exp_list:
        raise HTTPException(400, "expirations required")

    trade_d = _to_date(date_folder)
    series = []

    for exp_str in exp_list:
        exp_folder = _validate_date_folder(exp_str)
        pq = os.path.join(PARQUET_BASE, date_folder, exp_folder,
                          f"{settlement}.parquet")
        if not os.path.isfile(pq):
            continue

        flag_sql = "AND flag_any = false" if filter_flags else ""

        rows = _duckdb_query(f"""
            WITH snap AS (
                SELECT quote_time
                FROM (SELECT DISTINCT CAST(quote_time AS TIME) AS quote_time
                      FROM read_parquet('{pq}'))
                ORDER BY ABS(EXTRACT(EPOCH FROM
                             (quote_time - CAST('{time_str}' AS TIME))))
                LIMIT 1
            )
            SELECT strike, "right", implied_vol, delta, mid_price,
                   theta, vega, gamma, underlying_price,
                   moneyness, log_moneyness, bid, ask, spread_pct,
                   CAST(quote_time AS VARCHAR) AS qt
            FROM read_parquet('{pq}')
            WHERE CAST(quote_time AS TIME) = (SELECT quote_time FROM snap)
              AND implied_vol IS NOT NULL
              AND implied_vol > 0
              {flag_sql}
            ORDER BY strike
        """)

        if not rows:
            continue

        underlying   = rows[0][8]
        matched_time = str(rows[0][14])[:5]

        # OTM: puts below spot, calls at/above
        otm = [r for r in rows
               if (r[1] == "P" and r[0] < underlying)
               or (r[1] == "C" and r[0] >= underlying)]

        exp_d = _to_date(exp_folder)
        dte = (exp_d - trade_d).days

        series.append({
            "expiration":     exp_str,
            "dte":            dte,
            "label":          f"{exp_str} ({dte}D)",
            "underlying":     underlying,
            "matched_time":   matched_time,
            "strikes":        [r[0] for r in otm],
            "moneyness":      [round(r[9], 6) if r[9] else None for r in otm],
            "log_moneyness":  [round(r[10], 6) if r[10] else None for r in otm],
            "values":         [r[2] for r in otm],
            "rights":         [r[1] for r in otm],
            "metrics": [{
                "iv": r[2], "delta": r[3], "mid_price": r[4],
                "theta": r[5], "vega": r[6], "gamma": r[7],
                "bid": r[11], "ask": r[12], "spread_pct": r[13],
            } for r in otm],
        })

    return {
        "mode":       "raw_skew",
        "date":       date,
        "time":       time,
        "settlement": settlement,
        "x_axis":     x_axis,
        "series":     series,
    }


# ── Raw term structure ───────────────────────────────────────────────────────

@router.get("/term")
async def get_raw_term(
    date:         str  = Query(...),
    strikes:      str  = Query(..., description="Comma-separated strike prices"),
    time:         str  = Query("15:45"),
    settlement:   str  = Query("PM"),
    filter_flags: bool = Query(True),
) -> dict:
    """IV vs expiration for selected strikes (auto-OTM)."""
    _require_duckdb()
    date_folder = _validate_date_folder(date)
    settlement  = _validate_settlement(settlement)
    time_str    = _validate_time(time)
    strike_list = _parse_strikes(strikes)

    # Check glob has files before querying
    glob_pat = os.path.join(PARQUET_BASE, date_folder, "*",
                            f"{settlement}.parquet")
    if not globmod.glob(glob_pat):
        return {"mode": "raw_term", "series": [], "expirations": [],
                "underlying": None}

    strike_csv = ", ".join(str(s) for s in strike_list)
    flag_sql   = "AND flag_any = false" if filter_flags else ""
    trade_d    = _to_date(date_folder)

    rows = _duckdb_query(f"""
        SELECT
            regexp_extract(
                filename, '(\\d{{8}})/[^/]+\\.parquet$', 1
            ) AS exp_folder,
            strike, "right", implied_vol, delta, mid_price,
            theta, vega, gamma, underlying_price, bid, ask
        FROM read_parquet('{glob_pat}', filename=true)
        WHERE CAST(quote_time AS TIME) = CAST('{time_str}' AS TIME)
          AND strike IN ({strike_csv})
          AND implied_vol IS NOT NULL
          AND implied_vol > 0
          AND (("right" = 'P' AND strike < underlying_price)
            OR ("right" = 'C' AND strike >= underlying_price))
          {flag_sql}
        ORDER BY exp_folder, strike
    """)

    if not rows:
        return {"mode": "raw_term", "series": [], "expirations": [],
                "underlying": None}

    underlying = rows[0][9]

    bucket: dict[float, dict] = {s: {} for s in strike_list}
    for r in rows:
        exp_f, strike = r[0], r[1]
        if strike not in bucket:
            continue
        exp_iso = _to_iso(exp_f)
        exp_d = _to_date(exp_f)
        dte = (exp_d - trade_d).days
        if dte < 0:
            continue
        bucket[strike][exp_iso] = {
            "dte": dte, "iv": r[3],
            "delta": r[4], "mid_price": r[5],
            "theta": r[6], "vega": r[7], "gamma": r[8],
            "bid": r[10], "ask": r[11],
        }

    all_exps = sorted({e for sd in bucket.values() for e in sd})

    series = []
    for strike in strike_list:
        entries = bucket[strike]
        if not entries:
            continue
        label = str(int(strike)) if strike == int(strike) else str(strike)
        series.append({
            "label":       label,
            "strike":      strike,
            "expirations": all_exps,
            "dtes":   [entries.get(e, {}).get("dte") for e in all_exps],
            "values": [entries.get(e, {}).get("iv") for e in all_exps],
            "metrics": [entries[e] if e in entries else None
                        for e in all_exps],
        })

    return {
        "mode":        "raw_term",
        "date":        date,
        "time":        time,
        "settlement":  settlement,
        "underlying":  underlying,
        "expirations": all_exps,
        "series":      series,
    }


# ── Raw historical (IV time-series per strike across trade dates) ────────────

@router.get("/historical")
async def get_raw_historical(
    expiration:   str  = Query(..., description="Expiration date YYYY-MM-DD"),
    strikes:      str  = Query(..., description="Comma-separated strike prices"),
    start:        str  = Query(..., description="YYYY-MM-DD"),
    end:          str  = Query(..., description="YYYY-MM-DD"),
    time:         str  = Query("15:45"),
    settlement:   str  = Query("PM"),
    filter_flags: bool = Query(True),
) -> dict:
    """IV time-series: one line per strike across trade dates for a fixed expiration."""
    _require_duckdb()
    exp_folder  = _validate_date_folder(expiration)
    settlement  = _validate_settlement(settlement)
    time_str    = _validate_time(time)
    strike_list = _parse_strikes(strikes)
    start_f     = _validate_date_folder(start)
    end_f       = _validate_date_folder(end)

    # Glob across all trade dates for this expiration
    glob_pat = os.path.join(PARQUET_BASE, "*", exp_folder,
                            f"{settlement}.parquet")
    if not globmod.glob(glob_pat):
        return {"mode": "raw_historical", "series": [], "underlying": None}

    strike_csv = ", ".join(str(s) for s in strike_list)
    flag_sql   = "AND flag_any = false" if filter_flags else ""

    rows = _duckdb_query(f"""
        WITH base AS (
            SELECT
                regexp_extract(
                    filename, '(\\d{{8}})/\\d{{8}}/[^/]+\\.parquet$', 1
                ) AS td,
                CAST(quote_time AS TIME) AS qt,
                strike, "right", implied_vol, delta, mid_price,
                theta, vega, gamma, underlying_price, bid, ask
            FROM read_parquet('{glob_pat}', filename=true)
            WHERE strike IN ({strike_csv})
              AND implied_vol IS NOT NULL
              AND implied_vol > 0
              AND (("right" = 'P' AND strike < underlying_price)
                OR ("right" = 'C' AND strike >= underlying_price))
              {flag_sql}
        ),
        ranged AS (
            SELECT *, ROW_NUMBER() OVER (
                PARTITION BY td, strike
                ORDER BY ABS(EXTRACT(EPOCH FROM
                             (qt - CAST('{time_str}' AS TIME))))
            ) AS rn
            FROM base
            WHERE td BETWEEN '{start_f}' AND '{end_f}'
        )
        SELECT td, strike, "right", implied_vol, delta, mid_price,
               theta, vega, gamma, underlying_price, bid, ask
        FROM ranged WHERE rn = 1
        ORDER BY td, strike
    """)

    if not rows:
        return {"mode": "raw_historical", "series": [], "underlying": None}

    underlying = rows[0][9]

    # Bucket by strike -> { trade_date_iso: {iv, metrics…} }
    bucket: dict[float, dict] = {s: {} for s in strike_list}
    for r in rows:
        td_f, strike = r[0], r[1]
        if strike not in bucket:
            continue
        td_iso = _to_iso(td_f)
        bucket[strike][td_iso] = {
            "iv": r[3], "delta": r[4], "mid_price": r[5],
            "theta": r[6], "vega": r[7], "gamma": r[8],
            "bid": r[10], "ask": r[11],
        }

    all_dates = sorted({d for sd in bucket.values() for d in sd})

    series = []
    for strike in strike_list:
        entries = bucket[strike]
        if not entries:
            continue
        label = str(int(strike)) if strike == int(strike) else str(strike)
        series.append({
            "label":   label,
            "strike":  strike,
            "labels":  all_dates,
            "values":  [entries.get(d, {}).get("iv") for d in all_dates],
            "metrics": [entries[d] if d in entries else None
                        for d in all_dates],
        })

    return {
        "mode":        "raw_historical",
        "expiration":  expiration,
        "settlement":  settlement,
        "underlying":  underlying,
        "series":      series,
    }
