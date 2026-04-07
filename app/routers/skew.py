"""
Skew endpoints.

Three modes:
  GET /api/skew/by_dte     — single date+time, multiple DTEs (default skew view)
  GET /api/skew/by_date    — single DTE, multiple date+time pairs
  GET /api/skew/intraday   — single date, single DTE, multiple times within that day

All modes accept ?metric=iv|price|theta|vega|gamma
by_dte and by_date can include a history percentile band via ?band_days=N
"""
from datetime import date as date_type, time as time_type, timedelta
from typing import Optional

from fastapi import APIRouter, Depends, Query, HTTPException
from app.db import get_pool

router = APIRouter(tags=["skew"])

VALID_METRICS = {"iv", "price", "theta", "vega", "gamma"}

# Map integer put_delta to a human-readable label
def delta_label(pd: int) -> str:
    if pd == 50:
        return "ATM"
    if pd < 50:
        return f"{pd}Δp"
    return f"{100 - pd}Δc"


def _validate_metric(metric: str) -> str:
    if metric not in VALID_METRICS:
        raise HTTPException(400, f"metric must be one of {VALID_METRICS}")
    return metric


def _parse_ints(s: str) -> list[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


# ── by_dte ──────────────────────────────────────────────────────────────────

@router.get("/by_dte")
async def skew_by_dte(
    date:      str = Query(...),
    time:      str = Query(..., description="HH:MM"),
    dtes:      str = Query("7,30,90", description="Comma-separated DTEs"),
    metric:    str = Query("iv"),
    band_days: int = Query(0,  description="Lookback days for history band; 0 = off"),
    band_time: Optional[str] = Query(None, description="Target time for band (defaults to 'time')"),
    pool=Depends(get_pool),
) -> dict:
    """IV skew by put_delta for several DTEs on one snapshot."""
    metric = _validate_metric(metric)
    dte_list = _parse_ints(dtes)
    if not dte_list:
        raise HTTPException(400, "dtes must not be empty")

    async with pool.acquire() as conn:
        rows = await conn.fetch(
            f"""
            SELECT dte, put_delta,
                   {metric} AS value,
                   iv, price, theta, vega, gamma
            FROM spx_surface
            WHERE trade_date = $1
              AND quote_time  = $2::time
              AND dte         = ANY($3)
            ORDER BY dte, put_delta
            """,
            date_type.fromisoformat(date), time_type.fromisoformat(time), dte_list,
        )

        band = None
        if band_days > 0:
            ref_time = band_time or time
            end_d    = date_type.fromisoformat(date)
            start_d  = end_d - timedelta(days=band_days)
            band_rows = await conn.fetch(
                f"""
                WITH daily AS (
                    SELECT DISTINCT ON (trade_date, put_delta)
                        trade_date, put_delta, {metric} AS value
                    FROM spx_surface
                    WHERE dte         = $1
                      AND trade_date  BETWEEN $2 AND $3
                    ORDER BY trade_date, put_delta,
                             ABS(EXTRACT(EPOCH FROM (quote_time - $4::time)))
                )
                SELECT
                    put_delta,
                    PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY value) AS p25,
                    PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY value) AS p50,
                    PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY value) AS p75
                FROM daily
                GROUP BY put_delta
                ORDER BY put_delta
                """,
                dte_list[0], start_d, end_d, time_type.fromisoformat(ref_time),
            )
            band = {
                "put_deltas": [r["put_delta"] for r in band_rows],
                "p25": [r["p25"] for r in band_rows],
                "p50": [r["p50"] for r in band_rows],
                "p75": [r["p75"] for r in band_rows],
            }

    # Group rows by DTE
    by_dte: dict[int, list] = {d: [] for d in dte_list}
    metrics_by_dte: dict[int, list] = {d: [] for d in dte_list}
    for r in rows:
        by_dte[r["dte"]].append(r["value"])
        metrics_by_dte[r["dte"]].append({
            "iv":    r["iv"],
            "price": r["price"],
            "theta": r["theta"],
            "vega":  r["vega"],
            "gamma": r["gamma"],
        })

    # Collect unique put_deltas (same for every DTE)
    put_deltas = sorted({r["put_delta"] for r in rows})

    series = [
        {
            "label":      f"{d}D",
            "dte":        d,
            "put_deltas": put_deltas,
            "values":     by_dte[d],
            "metrics":    metrics_by_dte[d],
        }
        for d in dte_list
        if by_dte[d]
    ]

    return {
        "mode":   "by_dte",
        "date":   date,
        "time":   time,
        "metric": metric,
        "series": series,
        "band":   band,
    }


# ── by_date ──────────────────────────────────────────────────────────────────

@router.get("/by_date")
async def skew_by_date(
    dates:     str = Query(..., description="Comma-separated YYYY-MM-DD"),
    times:     str = Query(..., description="Comma-separated HH:MM (one per date, or single for all)"),
    dte:       int = Query(30),
    metric:    str = Query("iv"),
    band_days: int = Query(0),
    pool=Depends(get_pool),
) -> dict:
    """IV skew by put_delta for one DTE across several trade dates."""
    metric    = _validate_metric(metric)
    date_list = [d.strip() for d in dates.split(",") if d.strip()]
    time_list = [t.strip() for t in times.split(",") if t.strip()]

    # Broadcast single time to all dates
    if len(time_list) == 1:
        time_list = time_list * len(date_list)
    if len(time_list) != len(date_list):
        raise HTTPException(400, "times must have 1 entry or one per date")

    snapshots = list(zip(date_list, time_list))

    async with pool.acquire() as conn:
        rows = await conn.fetch(
            f"""
            SELECT trade_date, quote_time, put_delta, {metric} AS value
            FROM spx_surface
            WHERE dte = $1
              AND (trade_date::text, quote_time::text) = ANY(
                  SELECT x.d, x.t
                  FROM unnest($2::text[], $3::text[]) AS x(d, t)
              )
            ORDER BY trade_date, quote_time, put_delta
            """,
            dte,
            [s[0] for s in snapshots],
            [s[1] + ":00" for s in snapshots],
        )

    # Group by (trade_date, quote_time)
    grouped: dict[tuple, list] = {}
    for r in rows:
        key = (str(r["trade_date"]), str(r["quote_time"])[:5])
        grouped.setdefault(key, []).append(r["value"])

    put_deltas = sorted({r["put_delta"] for r in rows})

    series = [
        {
            "label":      f"{d} {t}",
            "date":       d,
            "time":       t,
            "put_deltas": put_deltas,
            "values":     grouped.get((d, t), []),
        }
        for d, t in snapshots
        if grouped.get((d, t))
    ]

    return {
        "mode":   "by_date",
        "dte":    dte,
        "metric": metric,
        "series": series,
        "band":   None,
    }


# ── intraday ─────────────────────────────────────────────────────────────────

@router.get("/intraday")
async def skew_intraday(
    date:   str = Query(...),
    dte:    int = Query(30),
    times:  str = Query("", description="Comma-separated HH:MM; empty = all available"),
    metric: str = Query("iv"),
    pool=Depends(get_pool),
) -> dict:
    """IV skew by put_delta for multiple intraday snapshots on a single date."""
    metric = _validate_metric(metric)
    time_list = [t.strip() for t in times.split(",") if t.strip()] if times else []

    async with pool.acquire() as conn:
        if time_list:
            rows = await conn.fetch(
                f"""
                SELECT quote_time, put_delta, {metric} AS value
                FROM spx_surface
                WHERE trade_date = $1
                  AND dte        = $2
                  AND quote_time = ANY($3::time[])
                ORDER BY quote_time, put_delta
                """,
                date_type.fromisoformat(date), dte,
                [time_type.fromisoformat(t) for t in time_list],
            )
        else:
            rows = await conn.fetch(
                f"""
                SELECT quote_time, put_delta, {metric} AS value
                FROM spx_surface
                WHERE trade_date = $1
                  AND dte        = $2
                ORDER BY quote_time, put_delta
                """,
                date_type.fromisoformat(date), dte,
            )

    grouped: dict[str, list] = {}
    for r in rows:
        key = str(r["quote_time"])[:5]
        grouped.setdefault(key, []).append(r["value"])

    put_deltas = sorted({r["put_delta"] for r in rows})
    ordered_times = sorted(grouped.keys())

    series = [
        {
            "label":      t,
            "time":       t,
            "put_deltas": put_deltas,
            "values":     grouped[t],
        }
        for t in ordered_times
    ]

    return {
        "mode":   "intraday",
        "date":   date,
        "dte":    dte,
        "metric": metric,
        "series": series,
    }
