"""
Term structure endpoints.

Two modes:
  GET /api/term/by_delta  — single date+time, multiple put_deltas, IV by DTE
  GET /api/term/by_date   — multiple dates, single put_delta, IV by DTE

Optionally include a history percentile band.
Also supports intraday: single date, multiple times, single put_delta.
"""
from datetime import date as date_type, time as time_type, timedelta
from typing import Optional

from fastapi import APIRouter, Depends, Query, HTTPException
from app.db import get_pool

router = APIRouter(tags=["term"])

VALID_METRICS = {"iv", "price", "theta", "vega", "gamma"}


def _validate_metric(metric: str) -> str:
    if metric not in VALID_METRICS:
        raise HTTPException(400, f"metric must be one of {VALID_METRICS}")
    return metric


def _parse_ints(s: str) -> list[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def _delta_label(pd: int) -> str:
    if pd == 50:
        return "ATM"
    if pd < 50:
        return f"{pd}Δp"
    return f"{100 - pd}Δc"


# ── by_delta ─────────────────────────────────────────────────────────────────

@router.get("/by_delta")
async def term_by_delta(
    date:      str = Query(...),
    time:      str = Query(...),
    deltas:    str = Query("25,50,75", description="Comma-separated put_delta integers"),
    dte_min:   int = Query(0),
    dte_max:   int = Query(360),
    metric:    str = Query("iv"),
    band_days: int = Query(0),
    band_time: Optional[str] = Query(None),
    pool=Depends(get_pool),
) -> dict:
    """Term structure by DTE for multiple delta levels on one snapshot."""
    metric     = _validate_metric(metric)
    delta_list = _parse_ints(deltas)
    if not delta_list:
        raise HTTPException(400, "deltas must not be empty")

    # When delta=50 is requested and metric is iv, use true forward ATM IV from spx_atm
    # instead of the 50-delta row from spx_surface.
    iv_expr    = "CASE WHEN s.put_delta = 50 THEN COALESCE(a.atm_iv, s.iv) ELSE s.iv END"
    value_expr = iv_expr if metric == "iv" else f"s.{metric}"

    async with pool.acquire() as conn:
        rows = await conn.fetch(
            f"""
            SELECT s.dte, s.put_delta,
                   {value_expr} AS value,
                   {iv_expr}    AS iv,
                   s.price, s.theta, s.vega, s.gamma
            FROM spx_surface s
            LEFT JOIN spx_atm a
                   ON a.trade_date = s.trade_date AND a.quote_time = s.quote_time
            WHERE s.trade_date = $1
              AND s.quote_time  = $2::time
              AND s.put_delta   = ANY($3)
              AND s.dte BETWEEN $4 AND $5
            ORDER BY s.put_delta, s.dte
            """,
            date_type.fromisoformat(date), time_type.fromisoformat(time), delta_list, dte_min, dte_max,
        )

        band = None
        if band_days > 0:
            ref_time = band_time or time
            end_d    = date_type.fromisoformat(date)
            start_d  = end_d - timedelta(days=band_days)
            # Band is built for each delta independently; return the ATM (pd=50)
            # band or the first delta if 50 not selected
            band_delta = 50 if 50 in delta_list else delta_list[0]
            if metric == "iv" and band_delta == 50:
                band_rows = await conn.fetch(
                    """
                    WITH daily_raw AS (
                        SELECT DISTINCT ON (trade_date, dte)
                            trade_date, dte, iv AS raw_iv
                        FROM spx_surface
                        WHERE put_delta    = $1
                          AND trade_date   BETWEEN $2 AND $3
                          AND dte          BETWEEN $4 AND $5
                        ORDER BY trade_date, dte,
                                 ABS(EXTRACT(EPOCH FROM (quote_time - $6::time)))
                    ),
                    atm AS (
                        SELECT DISTINCT ON (trade_date)
                            trade_date, atm_iv
                        FROM spx_atm
                        WHERE trade_date BETWEEN $2 AND $3
                        ORDER BY trade_date,
                                 ABS(EXTRACT(EPOCH FROM (quote_time - $6::time)))
                    ),
                    daily AS (
                        SELECT d.trade_date, d.dte,
                               COALESCE(a.atm_iv, d.raw_iv) AS value
                        FROM daily_raw d
                        LEFT JOIN atm a ON a.trade_date = d.trade_date
                    )
                    SELECT
                        dte,
                        MIN(value) AS pmin,
                        PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY value) AS p25,
                        PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY value) AS p50,
                        PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY value) AS p75,
                        MAX(value) AS pmax
                    FROM daily
                    GROUP BY dte
                    ORDER BY dte
                    """,
                    band_delta, start_d, end_d, dte_min, dte_max, time_type.fromisoformat(ref_time),
                )
            else:
                band_rows = await conn.fetch(
                    f"""
                    WITH daily AS (
                        SELECT DISTINCT ON (trade_date, dte)
                            trade_date, dte, {metric} AS value
                        FROM spx_surface
                        WHERE put_delta    = $1
                          AND trade_date   BETWEEN $2 AND $3
                          AND dte          BETWEEN $4 AND $5
                        ORDER BY trade_date, dte,
                                 ABS(EXTRACT(EPOCH FROM (quote_time - $6::time)))
                    )
                    SELECT
                        dte,
                        MIN(value) AS pmin,
                        PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY value) AS p25,
                        PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY value) AS p50,
                        PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY value) AS p75,
                        MAX(value) AS pmax
                    FROM daily
                    GROUP BY dte
                    ORDER BY dte
                    """,
                    band_delta, start_d, end_d, dte_min, dte_max, time_type.fromisoformat(ref_time),
                )
            band = {
                "delta":  band_delta,
                "label":  _delta_label(band_delta),
                "dtes":   [r["dte"] for r in band_rows],
                "pmin":   [r["pmin"] for r in band_rows],
                "p25":    [r["p25"]  for r in band_rows],
                "p50":    [r["p50"]  for r in band_rows],
                "p75":    [r["p75"]  for r in band_rows],
                "pmax":   [r["pmax"] for r in band_rows],
            }

    # Group by put_delta
    by_delta: dict[int, dict[int, float]] = {d: {} for d in delta_list}
    metrics_by_delta: dict[int, dict[int, dict]] = {d: {} for d in delta_list}
    for r in rows:
        by_delta[r["put_delta"]][r["dte"]] = r["value"]
        metrics_by_delta[r["put_delta"]][r["dte"]] = {
            "iv": r["iv"], "price": r["price"], "theta": r["theta"],
            "vega": r["vega"], "gamma": r["gamma"],
        }

    dtes = sorted({r["dte"] for r in rows})

    series = [
        {
            "label":   _delta_label(d),
            "delta":   d,
            "dtes":    dtes,
            "values":  [by_delta[d].get(dte) for dte in dtes],
            "metrics": [metrics_by_delta[d].get(dte) for dte in dtes],
        }
        for d in delta_list
        if by_delta[d]
    ]

    # Summary stats
    atm_vals = by_delta.get(50, {})
    stats = None
    if atm_vals:
        sorted_dtes = sorted(atm_vals.keys())
        short_iv = atm_vals[sorted_dtes[0]]
        long_iv  = atm_vals[sorted_dtes[-1]]
        stats = {
            "slope":     round(long_iv - short_iv, 4),
            "structure": "Contango" if long_iv > short_iv else "Backwardation",
            "short_dte": sorted_dtes[0],
            "long_dte":  sorted_dtes[-1],
            "short_iv":  round(short_iv, 4),
            "long_iv":   round(long_iv, 4),
        }

    return {
        "mode":   "by_delta",
        "date":   date,
        "time":   time,
        "metric": metric,
        "series": series,
        "band":   band,
        "stats":  stats,
    }


# ── by_date ───────────────────────────────────────────────────────────────────

@router.get("/by_date")
async def term_by_date(
    dates:   str = Query(..., description="Comma-separated YYYY-MM-DD"),
    times:   str = Query(..., description="Comma-separated HH:MM; one per date or single"),
    delta:   int = Query(50),
    dte_min: int = Query(0),
    dte_max: int = Query(360),
    metric:  str = Query("iv"),
    pool=Depends(get_pool),
) -> dict:
    """Term structure for a single delta across multiple trade dates."""
    metric    = _validate_metric(metric)
    date_list = [d.strip() for d in dates.split(",") if d.strip()]
    time_list = [t.strip() for t in times.split(",") if t.strip()]
    if len(time_list) == 1:
        time_list = time_list * len(date_list)
    if len(time_list) != len(date_list):
        raise HTTPException(400, "times must have 1 entry or one per date")

    value_expr = (
        "CASE WHEN s.put_delta = 50 THEN COALESCE(a.atm_iv, s.iv) ELSE s.iv END"
        if metric == "iv" else f"s.{metric}"
    )

    async with pool.acquire() as conn:
        rows = await conn.fetch(
            f"""
            SELECT s.trade_date, s.quote_time, s.dte, {value_expr} AS value
            FROM spx_surface s
            LEFT JOIN spx_atm a
                   ON a.trade_date = s.trade_date AND a.quote_time = s.quote_time
            WHERE s.put_delta = $1
              AND s.dte BETWEEN $2 AND $3
              AND (s.trade_date::text, s.quote_time::text) = ANY(
                  SELECT x.d, x.t
                  FROM unnest($4::text[], $5::text[]) AS x(d, t)
              )
            ORDER BY s.trade_date, s.dte
            """,
            delta, dte_min, dte_max,
            date_list,
            [t + ":00" for t in time_list],
        )

    by_snap: dict[tuple, dict[int, float]] = {}
    for r in rows:
        key = (str(r["trade_date"]), str(r["quote_time"])[:5])
        by_snap.setdefault(key, {})[r["dte"]] = r["value"]

    dtes = sorted({r["dte"] for r in rows})

    series = [
        {
            "label":  f"{d}",
            "date":   d,
            "time":   t,
            "dtes":   dtes,
            "values": [by_snap.get((d, t), {}).get(dte) for dte in dtes],
        }
        for d, t in zip(date_list, time_list)
        if by_snap.get((d, t))
    ]

    return {
        "mode":   "by_date",
        "delta":  delta,
        "metric": metric,
        "series": series,
    }


# ── intraday ──────────────────────────────────────────────────────────────────

@router.get("/intraday")
async def term_intraday(
    date:    str = Query(...),
    times:   str = Query("", description="Comma-separated HH:MM; empty = all"),
    delta:   int = Query(50),
    dte_min: int = Query(0),
    dte_max: int = Query(360),
    metric:  str = Query("iv"),
    pool=Depends(get_pool),
) -> dict:
    """Term structure for a single delta at multiple intraday times on one date."""
    metric    = _validate_metric(metric)
    time_list = [t.strip() for t in times.split(",") if t.strip()] if times else []

    value_expr = (
        "CASE WHEN s.put_delta = 50 THEN COALESCE(a.atm_iv, s.iv) ELSE s.iv END"
        if metric == "iv" else f"s.{metric}"
    )

    async with pool.acquire() as conn:
        if time_list:
            rows = await conn.fetch(
                f"""
                SELECT s.quote_time, s.dte, {value_expr} AS value
                FROM spx_surface s
                LEFT JOIN spx_atm a
                       ON a.trade_date = s.trade_date AND a.quote_time = s.quote_time
                WHERE s.trade_date = $1
                  AND s.put_delta  = $2
                  AND s.dte BETWEEN $3 AND $4
                  AND s.quote_time = ANY($5::time[])
                ORDER BY s.quote_time, s.dte
                """,
                date_type.fromisoformat(date), delta, dte_min, dte_max,
                [time_type.fromisoformat(t) for t in time_list],
            )
        else:
            rows = await conn.fetch(
                f"""
                SELECT s.quote_time, s.dte, {value_expr} AS value
                FROM spx_surface s
                LEFT JOIN spx_atm a
                       ON a.trade_date = s.trade_date AND a.quote_time = s.quote_time
                WHERE s.trade_date = $1
                  AND s.put_delta  = $2
                  AND s.dte BETWEEN $3 AND $4
                ORDER BY s.quote_time, s.dte
                """,
                date_type.fromisoformat(date), delta, dte_min, dte_max,
            )

    by_time: dict[str, dict[int, float]] = {}
    for r in rows:
        key = str(r["quote_time"])[:5]
        by_time.setdefault(key, {})[r["dte"]] = r["value"]

    dtes = sorted({r["dte"] for r in rows})

    series = [
        {
            "label":  t,
            "time":   t,
            "dtes":   dtes,
            "values": [by_time[t].get(dte) for dte in dtes],
        }
        for t in sorted(by_time)
    ]

    return {
        "mode":   "intraday",
        "date":   date,
        "delta":  delta,
        "metric": metric,
        "series": series,
    }
