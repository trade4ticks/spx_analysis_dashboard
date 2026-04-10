"""
Raw (non-interpolated) data endpoints.

Reads parquet files directly via DuckDB from the options data folder.
Folder structure: {PARQUET_BASE}/{trade_date_YYYYMMDD}/{expiration_YYYYMMDD}/{settlement}.parquet

GET /api/raw/expirations  — available expirations for a trade date
GET /api/raw/skew         — IV vs strike for one or more expirations (OTM puts + calls)
GET /api/raw/term         — IV vs expiration for selected strikes (auto-OTM)
"""
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


def _validate_date_folder(d: str) -> str:
    """Accept YYYY-MM-DD or YYYYMMDD, return YYYYMMDD for folder lookup."""
    clean = d.replace("-", "")
    if not re.match(r"^\d{8}$", clean):
        raise HTTPException(400, f"Invalid date: {d}")
    return clean


def _to_iso(d: str) -> str:
    """YYYYMMDD -> YYYY-MM-DD."""
    return f"{d[:4]}-{d[4:6]}-{d[6:8]}"


def _validate_settlement(s: str) -> str:
    s = s.upper()
    if s not in ("AM", "PM"):
        raise HTTPException(400, "settlement must be AM or PM")
    return s


def _validate_time(t: str) -> str:
    if not re.match(r"^\d{2}:\d{2}(:\d{2})?$", t):
        raise HTTPException(400, f"Invalid time: {t}")
    return t if len(t) > 5 else t + ":00"


# ── Expirations ──────────────────────────────────────────────────────────────

@router.get("/expirations")
async def get_expirations(
    date: str = Query(..., description="Trade date YYYY-MM-DD"),
) -> dict:
    """List available expirations, settlements, and underlying price."""
    if duckdb is None:
        raise HTTPException(501, "duckdb not installed on this server")
    date_folder = _validate_date_folder(date)
    base = os.path.join(PARQUET_BASE, date_folder)

    if not os.path.isdir(base):
        return {"date": date, "expirations": [], "underlying": None}

    trade_d = date_type(int(date_folder[:4]), int(date_folder[4:6]),
                        int(date_folder[6:8]))
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
        exp_d = date_type(int(name[:4]), int(name[4:6]), int(name[6:8]))
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
        con = duckdb.connect()
        try:
            row = con.execute(
                f"SELECT underlying_price "
                f"FROM read_parquet('{first_file}') LIMIT 1"
            ).fetchone()
            if row:
                underlying = row[0]
        finally:
            con.close()

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
    if duckdb is None:
        raise HTTPException(501, "duckdb not installed on this server")
    date_folder = _validate_date_folder(date)
    settlement  = _validate_settlement(settlement)
    time_str    = _validate_time(time)
    exp_list    = [e.strip() for e in expirations.split(",") if e.strip()]
    if not exp_list:
        raise HTTPException(400, "expirations required")

    trade_d = date_type(int(date_folder[:4]), int(date_folder[4:6]),
                        int(date_folder[6:8]))
    series = []

    for exp_str in exp_list:
        exp_folder = _validate_date_folder(exp_str)
        pq = os.path.join(PARQUET_BASE, date_folder, exp_folder,
                          f"{settlement}.parquet")
        if not os.path.isfile(pq):
            continue

        flag_sql = "AND flag_any = false" if filter_flags else ""

        con = duckdb.connect()
        try:
            rows = con.execute(f"""
                WITH snap AS (
                    SELECT DISTINCT quote_time
                    FROM read_parquet('{pq}')
                    ORDER BY ABS(EXTRACT(EPOCH FROM
                                 (quote_time - '{time_str}'::TIME)))
                    LIMIT 1
                )
                SELECT strike, right, implied_vol, delta, mid_price,
                       theta, vega, gamma, underlying_price,
                       moneyness, log_moneyness, bid, ask, spread_pct,
                       quote_time
                FROM read_parquet('{pq}')
                WHERE quote_time = (SELECT quote_time FROM snap)
                  AND implied_vol IS NOT NULL
                  AND implied_vol > 0
                  {flag_sql}
                ORDER BY strike
            """).fetchall()
        finally:
            con.close()

        if not rows:
            continue

        underlying   = rows[0][8]
        matched_time = str(rows[0][14])[:5]

        # OTM: puts below spot, calls at/above
        otm = [r for r in rows
               if (r[1] == "P" and r[0] < underlying)
               or (r[1] == "C" and r[0] >= underlying)]

        exp_d = date_type(int(exp_folder[:4]), int(exp_folder[4:6]),
                          int(exp_folder[6:8]))
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
    """IV vs expiration for selected strikes (auto-OTM: put below spot, call above)."""
    if duckdb is None:
        raise HTTPException(501, "duckdb not installed on this server")
    date_folder = _validate_date_folder(date)
    settlement  = _validate_settlement(settlement)
    time_str    = _validate_time(time)

    try:
        strike_list = [float(s.strip()) for s in strikes.split(",") if s.strip()]
    except ValueError:
        raise HTTPException(400, "strikes must be numeric")
    if not strike_list:
        raise HTTPException(400, "strikes required")

    base = os.path.join(PARQUET_BASE, date_folder)
    if not os.path.isdir(base):
        return {"mode": "raw_term", "series": [], "expirations": [],
                "underlying": None}

    # Use DuckDB glob to read all expiration files in one query
    glob = os.path.join(PARQUET_BASE, date_folder, "*",
                        f"{settlement}.parquet")
    strike_csv = ", ".join(str(s) for s in strike_list)
    flag_sql   = "AND flag_any = false" if filter_flags else ""

    trade_d = date_type(int(date_folder[:4]), int(date_folder[4:6]),
                        int(date_folder[6:8]))

    con = duckdb.connect()
    try:
        rows = con.execute(f"""
            SELECT
                regexp_extract(
                    filename, '(\\d{{8}})/[^/]+\\.parquet$', 1
                ) AS exp_folder,
                strike, right, implied_vol, delta, mid_price,
                theta, vega, gamma, underlying_price, bid, ask
            FROM read_parquet('{glob}', filename=true)
            WHERE quote_time = '{time_str}'::TIME
              AND strike IN ({strike_csv})
              AND implied_vol IS NOT NULL
              AND implied_vol > 0
              AND ((right = 'P' AND strike < underlying_price)
                OR (right = 'C' AND strike >= underlying_price))
              {flag_sql}
            ORDER BY exp_folder, strike
        """).fetchall()
    finally:
        con.close()

    if not rows:
        return {"mode": "raw_term", "series": [], "expirations": [],
                "underlying": None}

    underlying = rows[0][9]

    # Bucket by strike -> { exp_iso: {dte, iv, metrics…} }
    bucket: dict[float, dict] = {s: {} for s in strike_list}
    for r in rows:
        exp_f, strike = r[0], r[1]
        if strike not in bucket:
            continue
        exp_iso = _to_iso(exp_f)
        exp_d = date_type(int(exp_f[:4]), int(exp_f[4:6]), int(exp_f[6:8]))
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
