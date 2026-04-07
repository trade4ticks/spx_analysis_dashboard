"""
Historical time-series endpoints.

GET /api/historical
  — daily mode:    one snapshot per trade_date (closest to target_time)
  — intraday mode: all snapshots for a date range (max 90 days), with optional
                   window shift via start/end

Returns IV (or other metric) over time for one DTE and one or more put_deltas.
"""
from datetime import date as date_type, timedelta
from typing import Optional

from fastapi import APIRouter, Depends, Query, HTTPException
from app.db import get_pool

router = APIRouter(tags=["historical"])

VALID_METRICS = {"iv", "price", "theta", "vega", "gamma"}
INTRADAY_MAX_DAYS = 90


def _validate_metric(m: str) -> str:
    if m not in VALID_METRICS:
        raise HTTPException(400, f"metric must be one of {VALID_METRICS}")
    return m


def _parse_ints(s: str) -> list[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def _delta_label(pd: int) -> str:
    if pd == 50:
        return "ATM"
    if pd < 50:
        return f"{pd}Δp"
    return f"{100 - pd}Δc"


@router.get("")
async def get_historical(
    dte:         int = Query(30),
    deltas:      str = Query("25,50,75", description="Comma-separated put_delta integers"),
    start:       str = Query(..., description="YYYY-MM-DD"),
    end:         str = Query(..., description="YYYY-MM-DD"),
    target_time: str = Query("15:45", description="HH:MM snapshot for daily mode"),
    metric:      str = Query("iv"),
    freq:        str = Query("daily", description="daily | intraday"),
    pool=Depends(get_pool),
) -> dict:
    """
    Time series of a surface metric for a fixed DTE across multiple deltas.

    daily    → one row per trade_date (closest snapshot to target_time).
    intraday → every snapshot within [start, end]; capped at 90 calendar days.
    """
    metric     = _validate_metric(metric)
    delta_list = _parse_ints(deltas)
    if not delta_list:
        raise HTTPException(400, "deltas must not be empty")

    start_d = date_type.fromisoformat(start)
    end_d   = date_type.fromisoformat(end)

    if freq == "intraday":
        if (end_d - start_d).days > INTRADAY_MAX_DAYS:
            raise HTTPException(
                400,
                f"Intraday mode is limited to {INTRADAY_MAX_DAYS} days. "
                f"Requested {(end_d - start_d).days} days."
            )

    async with pool.acquire() as conn:
        if freq == "daily":
            rows = await conn.fetch(
                f"""
                WITH closest AS (
                    SELECT DISTINCT ON (trade_date, put_delta)
                        trade_date,
                        quote_time,
                        put_delta,
                        {metric} AS value
                    FROM spx_surface
                    WHERE dte        = $1
                      AND put_delta  = ANY($2)
                      AND trade_date BETWEEN $3 AND $4
                    ORDER BY trade_date, put_delta,
                             ABS(EXTRACT(EPOCH FROM (quote_time - $5::time)))
                )
                SELECT trade_date, put_delta, value
                FROM closest
                ORDER BY trade_date, put_delta
                """,
                dte, delta_list, start_d, end_d, target_time,
            )
            # Labels are dates
            dates = sorted({str(r["trade_date"]) for r in rows})
            labels = dates

            by_delta: dict[int, dict[str, float]] = {d: {} for d in delta_list}
            for r in rows:
                by_delta[r["put_delta"]][str(r["trade_date"])] = r["value"]

            series = [
                {
                    "label":  _delta_label(d),
                    "delta":  d,
                    "dte":    dte,
                    "labels": labels,
                    "values": [by_delta[d].get(lbl) for lbl in labels],
                }
                for d in delta_list
            ]

        else:  # intraday
            rows = await conn.fetch(
                f"""
                SELECT trade_date, quote_time, put_delta, {metric} AS value
                FROM spx_surface
                WHERE dte        = $1
                  AND put_delta  = ANY($2)
                  AND trade_date BETWEEN $3 AND $4
                ORDER BY trade_date, quote_time, put_delta
                """,
                dte, delta_list, start_d, end_d,
            )
            # Labels are "YYYY-MM-DD HH:MM"
            timestamps = sorted({
                f"{r['trade_date']} {str(r['quote_time'])[:5]}"
                for r in rows
            })
            labels = timestamps

            by_delta = {d: {} for d in delta_list}
            for r in rows:
                ts  = f"{r['trade_date']} {str(r['quote_time'])[:5]}"
                by_delta[r["put_delta"]][ts] = r["value"]

            series = [
                {
                    "label":  _delta_label(d),
                    "delta":  d,
                    "dte":    dte,
                    "labels": labels,
                    "values": [by_delta[d].get(lbl) for lbl in labels],
                }
                for d in delta_list
            ]

    # Summary stats for each series
    for s in series:
        vals = [v for v in s["values"] if v is not None]
        if vals:
            s["stats"] = {
                "mean":    round(sum(vals) / len(vals), 4),
                "minimum": round(min(vals), 4),
                "maximum": round(max(vals), 4),
                "current": round(vals[-1], 4),
            }

    # Top-level stats list (one entry per series) for the panel footer
    stats = [
        {
            "label":   s["label"],
            "delta":   s["delta"],
            "current": s.get("stats", {}).get("current"),
            "mean":    s.get("stats", {}).get("mean"),
        }
        for s in series
    ]

    return {
        "freq":   freq,
        "dte":    dte,
        "metric": metric,
        "start":  start,
        "end":    end,
        "series": series,
        "stats":  stats,
    }
