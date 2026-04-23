"""
Raw (non-interpolated) data endpoints.

Reads parquet files directly via DuckDB from the options data folder.
Folder structure: {PARQUET_BASE}/{trade_date_YYYYMMDD}/{expiration_YYYYMMDD}/{settlement}.parquet

GET /api/raw/expirations  — available expirations for a trade date
GET /api/raw/skew         — IV vs strike for (date, expiration) combos
GET /api/raw/term         — IV vs DTE for (date, strike) combos
GET /api/raw/historical   — IV time-series for (expiration, strike) combos
"""
import glob as globmod
import math
import os
import re
from datetime import date as date_type
from typing import Optional

try:
    import duckdb
except ImportError:
    duckdb = None

from fastapi import APIRouter, Query, HTTPException

router = APIRouter(tags=["raw"])

def _init_parquet_bases() -> list[str]:
    env = os.getenv("PARQUET_BASES", "")
    if env:
        return [p.strip() for p in env.split(",") if p.strip()]
    single = os.getenv("PARQUET_BASE", "")
    if single:
        return [single]
    return ["/mnt/volume1/spx_options", "/mnt/volume2/spx_options"]

PARQUET_BASES = _init_parquet_bases()


def _require_duckdb():
    if duckdb is None:
        raise HTTPException(501, "duckdb not installed on this server")


def _validate_date_folder(d: str) -> str:
    """Accept YYYY-MM-DD or YYYYMMDD, return YYYYMMDD."""
    clean = d.replace("-", "")
    if not re.match(r"^\d{8}$", clean):
        raise HTTPException(400, f"Invalid date: {d}")
    return clean


def _to_iso(d: str) -> str:
    return f"{d[:4]}-{d[4:6]}-{d[6:8]}"


def _to_date(d: str) -> date_type:
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


def _split_csv(s: str) -> list[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def _strike_label(s: float) -> str:
    return str(int(s)) if s == int(s) else str(s)


VALID_METRICS = {"iv", "price", "theta", "vega", "gamma"}


def _validate_metric(m: str) -> str:
    if m not in VALID_METRICS:
        raise HTTPException(400, f"metric must be one of {VALID_METRICS}")
    return m


def _duckdb_query(sql: str):
    """Run a DuckDB query, return rows. Errors → HTTP 500 with detail."""
    con = duckdb.connect()
    try:
        return con.execute(sql).fetchall()
    except Exception as e:
        raise HTTPException(500, f"DuckDB query error: {e}")
    finally:
        con.close()


def _find_parquet(*rel_parts: str) -> str | None:
    """Return the full path of the first existing parquet file across all bases."""
    for base in PARQUET_BASES:
        path = os.path.join(base, *rel_parts)
        if os.path.isfile(path):
            return path
    return None


def _find_dir(*rel_parts: str) -> str | None:
    """Return the full path of the first existing directory across all bases."""
    for base in PARQUET_BASES:
        path = os.path.join(base, *rel_parts)
        if os.path.isdir(path):
            return path
    return None


def _listdir_merged(*rel_parts: str) -> list[str]:
    """Return sorted unique directory entries merged across all bases."""
    seen: set[str] = set()
    for base in PARQUET_BASES:
        d = os.path.join(base, *rel_parts)
        if os.path.isdir(d):
            seen.update(os.listdir(d))
    return sorted(seen)


def _glob_multi(rel_pattern: str) -> list[str]:
    """Glob a relative pattern across all bases, returning sorted unique paths."""
    results = []
    for base in PARQUET_BASES:
        results.extend(globmod.glob(os.path.join(base, rel_pattern)))
    return sorted(results)


# ── Expirations ──────────────────────────────────────────────────────────────

@router.get("/expirations")
async def get_expirations(
    date: str = Query(..., description="Trade date YYYY-MM-DD"),
) -> dict:
    """List available expirations, settlements, and underlying price."""
    _require_duckdb()
    date_folder = _validate_date_folder(date)

    trade_d = _to_date(date_folder)
    results = []
    first_file = None

    for name in _listdir_merged(date_folder):
        if not re.match(r"^\d{8}$", name):
            continue
        path = _find_dir(date_folder, name)
        if path is None:
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
            pass

    return {"date": date, "expirations": results, "underlying": underlying}


# ── Skew helpers ─────────────────────────────────────────────────────────────


def _otm_series(rows, dte, midx, label, exp_str, date_str):
    """Compute implied forward, OTM-filter rows, and build one series entry."""
    if not rows:
        return None
    underlying = rows[0][8]
    puts_by_k  = {r[0]: r for r in rows if r[1] == "P"}
    calls_by_k = {r[0]: r for r in rows if r[1] == "C"}
    both = sorted(set(puts_by_k) & set(calls_by_k))
    forward = underlying
    if both:
        k_atm = min(both, key=lambda k: abs(k - underlying))
        p_mid, c_mid = puts_by_k[k_atm][4], calls_by_k[k_atm][4]
        if p_mid is not None and c_mid is not None:
            T = max(dte, 1) / 365.0
            forward = k_atm + (c_mid - p_mid) * math.exp(0.045 * T)
    otm = [r for r in rows
           if (r[1] == "P" and r[0] < forward)
           or (r[1] == "C" and r[0] >= forward)]
    if not otm:
        return None
    return {
        "label": label, "expiration": exp_str, "date": date_str,
        "dte": dte, "underlying": underlying, "forward": forward,
        "strikes":   [r[0] for r in otm],
        "moneyness": [round(r[9], 6) if r[9] else None for r in otm],
        "values":    [r[midx] for r in otm],
        "rights":    [r[1] for r in otm],
        "metrics": [{"iv": r[2], "delta": r[3], "mid_price": r[4],
                     "theta": r[5], "vega": r[6], "gamma": r[7],
                     "bid": r[11], "ask": r[12], "spread_pct": r[13]}
                    for r in otm],
    }


# ── Raw skew ─────────────────────────────────────────────────────────────────

@router.get("/skew")
async def get_raw_skew(
    dates:        str  = Query(..., description="Comma-separated trade dates"),
    expirations:  str  = Query(..., description="Comma-separated expirations"),
    time:         str  = Query("15:45"),
    times:        Optional[str] = Query(None, description="Comma-separated HH:MM for intraday"),
    settlement:   str  = Query("PM"),
    filter_flags: bool = Query(True),
    x_axis:       str  = Query("strike", description="strike | moneyness"),
    metric:       str  = Query("iv"),
) -> dict:
    """OTM puts + calls vs strike for each (date, expiration[, time]) combo."""
    _require_duckdb()
    settlement = _validate_settlement(settlement)
    metric     = _validate_metric(metric)
    midx = {"iv": 2, "price": 4, "theta": 5, "vega": 6, "gamma": 7}[metric]
    date_list  = _split_csv(dates)
    exp_list   = _split_csv(expirations)
    if not date_list or not exp_list:
        raise HTTPException(400, "dates and expirations required")

    # Build time list: multiple → intraday, single → closest-match
    if times:
        time_list = [_validate_time(t.strip()) for t in times.split(",") if t.strip()]
    else:
        time_list = [_validate_time(time)]
    multi_time = len(time_list) > 1
    multi_date = len(date_list) > 1
    multi_exp  = len(exp_list) > 1
    flag_sql   = "AND flag_any = false" if filter_flags else ""

    series = []
    for date_str in date_list:
        date_folder = _validate_date_folder(date_str)
        trade_d = _to_date(date_folder)
        for exp_str in exp_list:
            exp_folder = _validate_date_folder(exp_str)
            pq = _find_parquet(date_folder, exp_folder, f"{settlement}.parquet")
            if pq is None:
                continue
            exp_d = _to_date(exp_folder)
            dte = (exp_d - trade_d).days

            if multi_time:
                # ── Intraday: all requested times in one query, group by time
                times_sql = ", ".join(f"CAST('{t}' AS TIME)" for t in time_list)
                rows = _duckdb_query(f"""
                    SELECT strike, "right", implied_vol, delta, mid_price,
                           theta, vega, gamma, underlying_price,
                           moneyness, log_moneyness, bid, ask, spread_pct,
                           CAST(quote_time AS VARCHAR) AS qt
                    FROM read_parquet('{pq}')
                    WHERE CAST(quote_time AS TIME) IN ({times_sql})
                      AND implied_vol IS NOT NULL AND implied_vol > 0
                      {flag_sql}
                    ORDER BY qt, strike
                """)
                if not rows:
                    continue
                # Group by time
                groups: dict[str, list] = {}
                for r in rows:
                    groups.setdefault(str(r[14])[:5], []).append(r)
                for t_key in sorted(groups):
                    s = _otm_series(groups[t_key], dte, midx, t_key,
                                    exp_str, date_str)
                    if s:
                        series.append(s)
            else:
                # ── Single time: closest-match (existing behavior)
                time_str = time_list[0]
                rows = _duckdb_query(f"""
                    WITH snap AS (
                        SELECT quote_time
                        FROM (SELECT DISTINCT CAST(quote_time AS TIME) AS quote_time
                              FROM read_parquet('{pq}'))
                        ORDER BY ABS(EXTRACT(EPOCH FROM quote_time)
                                   - EXTRACT(EPOCH FROM CAST('{time_str}' AS TIME)))
                        LIMIT 1
                    )
                    SELECT strike, "right", implied_vol, delta, mid_price,
                           theta, vega, gamma, underlying_price,
                           moneyness, log_moneyness, bid, ask, spread_pct
                    FROM read_parquet('{pq}')
                    WHERE CAST(quote_time AS TIME) = (SELECT quote_time FROM snap)
                      AND implied_vol IS NOT NULL AND implied_vol > 0
                      {flag_sql}
                    ORDER BY strike
                """)
                if not rows:
                    continue
                if multi_date and multi_exp:
                    label = f"{date_str} · {exp_str}"
                elif multi_date:
                    label = date_str
                else:
                    label = f"{exp_str} ({dte}D)"
                s = _otm_series(rows, dte, midx, label, exp_str, date_str)
                if s:
                    series.append(s)

    return {
        "mode":   "raw_skew",
        "x_axis": x_axis,
        "metric": metric,
        "series": series,
    }


# ── Raw term structure ───────────────────────────────────────────────────────

@router.get("/term")
async def get_raw_term(
    dates:        str  = Query(..., description="Comma-separated trade dates"),
    strikes:      str  = Query(..., description="Comma-separated strike prices"),
    time:         str  = Query("15:45"),
    times:        Optional[str] = Query(None, description="Comma-separated HH:MM for intraday"),
    settlement:   str  = Query("PM"),
    filter_flags: bool = Query(True),
    metric:       str  = Query("iv"),
) -> dict:
    """Selected metric vs DTE for each (date, strike[, time]) combo (auto-OTM)."""
    _require_duckdb()
    settlement  = _validate_settlement(settlement)
    metric      = _validate_metric(metric)
    date_list   = _split_csv(dates)
    strike_list = _parse_strikes(strikes)
    if not date_list:
        raise HTTPException(400, "dates required")
    metric_key = {"iv": "iv", "price": "mid_price",
                  "theta": "theta", "vega": "vega", "gamma": "gamma"}[metric]

    if times:
        time_list = [_validate_time(t.strip()) for t in times.split(",") if t.strip()]
    else:
        time_list = [_validate_time(time)]
    multi_time   = len(time_list) > 1
    multi_date   = len(date_list) > 1
    multi_strike = len(strike_list) > 1
    flag_sql     = "AND flag_any = false" if filter_flags else ""
    strike_csv   = ", ".join(str(s) for s in strike_list)

    series = []
    for date_str in date_list:
        date_folder = _validate_date_folder(date_str)
        trade_d = _to_date(date_folder)

        all_files = _glob_multi(
            os.path.join(date_folder, "*", f"{settlement}.parquet"))
        if not all_files:
            continue

        valid = []
        for f in all_files:
            parts = f.replace("\\", "/").split("/")
            if len(parts) < 2:
                continue
            exp_f = parts[-2]
            if re.match(r"^\d{8}$", exp_f):
                valid.append((f, exp_f))
        if not valid:
            continue

        # Build per-time queries
        for t_str in time_list:
            if multi_time:
                # Intraday: exact time match
                time_filter = f"CAST(quote_time AS TIME) = CAST('{t_str}' AS TIME)"
            else:
                time_filter = f"CAST(quote_time AS TIME) = CAST('{t_str}' AS TIME)"

            union_parts = []
            for f, exp_f in valid:
                union_parts.append(f"""
                    SELECT '{exp_f}' AS exp_folder,
                           strike, "right", implied_vol, delta, mid_price,
                           theta, vega, gamma, underlying_price, bid, ask
                    FROM read_parquet('{f}')
                    WHERE {time_filter}
                      AND strike IN ({strike_csv})
                      AND implied_vol IS NOT NULL AND implied_vol > 0
                      AND (("right" = 'P' AND strike < underlying_price)
                        OR ("right" = 'C' AND strike >= underlying_price))
                      {flag_sql}
                """)
            union_sql = " UNION ALL ".join(union_parts)
            rows = _duckdb_query(union_sql + " ORDER BY exp_folder, strike")
            if not rows:
                continue

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

            for strike in strike_list:
                entries = bucket[strike]
                if not entries:
                    continue
                klabel = _strike_label(strike)
                t_label = t_str[:5]
                if multi_time:
                    label = t_label if not multi_strike else f"{t_label} · K{klabel}"
                elif multi_date and multi_strike:
                    label = f"{date_str} · K{klabel}"
                elif multi_date:
                    label = date_str
                else:
                    label = klabel

                series.append({
                    "label":       label,
                    "strike":      strike,
                    "date":        date_str,
                    "expirations": all_exps,
                    "dtes":   [entries.get(e, {}).get("dte") for e in all_exps],
                    "values": [entries.get(e, {}).get(metric_key) for e in all_exps],
                    "metrics": [entries[e] if e in entries else None
                                for e in all_exps],
            })

    return {"mode": "raw_term", "metric": metric, "series": series}


# ── Raw historical (IV time-series per (exp, strike) across trade dates) ────

@router.get("/historical")
async def get_raw_historical(
    expirations:  str  = Query(..., description="Comma-separated expirations"),
    strikes:      str  = Query(..., description="Comma-separated strike prices"),
    start:        str  = Query(...),
    end:          str  = Query(...),
    time:         str  = Query("15:45"),
    settlement:   str  = Query("PM"),
    filter_flags: bool = Query(True),
    metric:       str  = Query("iv"),
) -> dict:
    """Selected metric time-series for each (expiration, strike) combo."""
    _require_duckdb()
    settlement  = _validate_settlement(settlement)
    time_str    = _validate_time(time)
    metric      = _validate_metric(metric)
    exp_list    = _split_csv(expirations)
    strike_list = _parse_strikes(strikes)
    start_f     = _validate_date_folder(start)
    end_f       = _validate_date_folder(end)
    if not exp_list:
        raise HTTPException(400, "expirations required")
    metric_key = {"iv": "iv", "price": "mid_price",
                  "theta": "theta", "vega": "vega", "gamma": "gamma"}[metric]

    multi_exp    = len(exp_list) > 1
    multi_strike = len(strike_list) > 1
    flag_sql     = "AND flag_any = false" if filter_flags else ""
    strike_csv   = ", ".join(str(s) for s in strike_list)

    # Collect all data: (exp_str, strike) -> {date_iso: metrics}
    all_data: dict[tuple, dict] = {}
    underlying = None

    for exp_str in exp_list:
        exp_folder = _validate_date_folder(exp_str)
        all_files = _glob_multi(
            os.path.join("*", exp_folder, f"{settlement}.parquet"))
        if not all_files:
            continue

        # Parse trade_date from each file path in Python (skip regex_extract
        # which has DuckDB version quirks). Path: .../YYYYMMDD/EXP/PM.parquet
        valid = []
        for f in all_files:
            parts = f.replace("\\", "/").split("/")
            if len(parts) < 3:
                continue
            td_folder = parts[-3]
            if not re.match(r"^\d{8}$", td_folder):
                continue
            if not (start_f <= td_folder <= end_f):
                continue
            valid.append((f, td_folder))

        if not valid:
            continue

        # Build a UNION ALL query with the trade_date hardcoded per file
        union_parts = []
        for f, td_folder in valid:
            union_parts.append(f"""
                SELECT '{td_folder}' AS td,
                       CAST(quote_time AS TIME) AS qt,
                       strike, "right", implied_vol, delta, mid_price,
                       theta, vega, gamma, underlying_price, bid, ask
                FROM read_parquet('{f}')
                WHERE strike IN ({strike_csv})
                  AND implied_vol IS NOT NULL
                  AND implied_vol > 0
                  AND (("right" = 'P' AND strike < underlying_price)
                    OR ("right" = 'C' AND strike >= underlying_price))
                  {flag_sql}
            """)
        union_sql = " UNION ALL ".join(union_parts)

        rows = _duckdb_query(f"""
            WITH base AS ({union_sql}),
            ranked AS (
                SELECT *, ROW_NUMBER() OVER (
                    PARTITION BY td, strike
                    ORDER BY ABS(EXTRACT(EPOCH FROM qt)
                               - EXTRACT(EPOCH FROM CAST('{time_str}' AS TIME)))
                ) AS rn
                FROM base
            )
            SELECT td, strike, "right", implied_vol, delta, mid_price,
                   theta, vega, gamma, underlying_price, bid, ask
            FROM ranked WHERE rn = 1
            ORDER BY td, strike
        """)

        if not rows:
            continue
        if underlying is None:
            underlying = rows[0][9]

        for r in rows:
            td_str, strike = r[0], r[1]
            if strike not in strike_list:
                continue
            td_iso = _to_iso(td_str)
            key = (exp_str, strike)
            all_data.setdefault(key, {})[td_iso] = {
                "iv": r[3], "delta": r[4], "mid_price": r[5],
                "theta": r[6], "vega": r[7], "gamma": r[8],
                "bid": r[10], "ask": r[11],
            }

    if not all_data:
        return {"mode": "raw_historical", "series": [], "underlying": None}

    # Unified date axis across all series
    all_dates = sorted({d for v in all_data.values() for d in v})

    series = []
    for exp_str in exp_list:
        for strike in strike_list:
            entries = all_data.get((exp_str, strike))
            if not entries:
                continue
            klabel = _strike_label(strike)
            if multi_exp and multi_strike:
                label = f"{exp_str} · K{klabel}"
            elif multi_exp:
                label = exp_str
            else:
                label = klabel

            series.append({
                "label":      label,
                "strike":     strike,
                "expiration": exp_str,
                "labels":     all_dates,
                "values":     [entries.get(d, {}).get(metric_key) for d in all_dates],
                "metrics":    [entries[d] if d in entries else None
                               for d in all_dates],
            })

    return {
        "mode":       "raw_historical",
        "metric":     metric,
        "underlying": underlying,
        "series":     series,
    }
