"""
Historical time-series endpoints.

GET /api/historical
  — daily mode:    one snapshot per trade_date (closest to target_time)
  — intraday mode: all snapshots for a date range (max 90 days), with optional
                   window shift via start/end

Returns IV (or other metric) over time for one DTE and one or more put_deltas.
"""
from datetime import date as date_type, time as time_type, timedelta
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
    dte:         Optional[int] = Query(None),
    dtes:        Optional[str] = Query(None, description="Comma-separated DTEs (multi-DTE mode)"),
    delta:       Optional[int] = Query(None),
    deltas:      Optional[str] = Query(None, description="Comma-separated put_deltas (multi-delta mode)"),
    start:       str = Query(..., description="YYYY-MM-DD"),
    end:         str = Query(..., description="YYYY-MM-DD"),
    target_time: str = Query("15:45", description="HH:MM snapshot for daily mode"),
    metric:      str = Query("iv"),
    freq:        str = Query("daily", description="daily | intraday"),
    pool=Depends(get_pool),
) -> dict:
    """
    Time series of a surface metric. Either:
      - Multi-Delta:  one DTE, several put_deltas (one line per delta)
      - Multi-DTE:    one put_delta, several DTEs (one line per DTE)
    """
    metric = _validate_metric(metric)

    # Resolve which dimension is the multi-dim
    if deltas and dte is not None:
        dimension  = "delta"
        delta_list = _parse_ints(deltas)
        dte_list   = [dte]
        fixed_dte  = dte
        fixed_delta = None
    elif dtes and delta is not None:
        dimension  = "dte"
        dte_list   = _parse_ints(dtes)
        delta_list = [delta]
        fixed_dte  = None
        fixed_delta = delta
    else:
        raise HTTPException(400, "Provide either (dte + deltas) or (delta + dtes)")

    if not delta_list or not dte_list:
        raise HTTPException(400, "delta/dte lists must not be empty")

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
                """
                WITH closest_raw AS (
                    SELECT DISTINCT ON (trade_date, dte, put_delta)
                        trade_date, quote_time, dte, put_delta,
                        iv, price, theta, vega, gamma
                    FROM spx_surface
                    WHERE dte        = ANY($1)
                      AND put_delta  = ANY($2)
                      AND trade_date BETWEEN $3 AND $4
                    ORDER BY trade_date, dte, put_delta,
                             ABS(EXTRACT(EPOCH FROM (quote_time - $5::time)))
                ),
                atm AS (
                    SELECT DISTINCT ON (trade_date)
                        trade_date, atm_iv
                    FROM spx_atm
                    WHERE trade_date BETWEEN $3 AND $4
                    ORDER BY trade_date,
                             ABS(EXTRACT(EPOCH FROM (quote_time - $5::time)))
                ),
                closest AS (
                    SELECT c.trade_date, c.dte, c.put_delta,
                           CASE WHEN c.put_delta = 50
                                THEN COALESCE(a.atm_iv, c.iv)
                                ELSE c.iv END AS iv,
                           c.price, c.theta, c.vega, c.gamma
                    FROM closest_raw c
                    LEFT JOIN atm a ON a.trade_date = c.trade_date
                )
                SELECT trade_date, dte, put_delta, iv, price, theta, vega, gamma
                FROM closest
                ORDER BY trade_date, dte, put_delta
                """,
                dte_list, delta_list, start_d, end_d, time_type.fromisoformat(target_time),
            )
            label_keys = sorted({str(r["trade_date"]) for r in rows})
            row_key = lambda r: str(r["trade_date"])
        else:
            rows = await conn.fetch(
                """
                SELECT s.trade_date, s.quote_time, s.dte, s.put_delta,
                       CASE WHEN s.put_delta = 50
                            THEN COALESCE(a.atm_iv, s.iv)
                            ELSE s.iv END AS iv,
                       s.price, s.theta, s.vega, s.gamma
                FROM spx_surface s
                LEFT JOIN spx_atm a
                       ON a.trade_date = s.trade_date AND a.quote_time = s.quote_time
                WHERE s.dte        = ANY($1)
                  AND s.put_delta  = ANY($2)
                  AND s.trade_date BETWEEN $3 AND $4
                ORDER BY s.trade_date, s.quote_time, s.dte, s.put_delta
                """,
                dte_list, delta_list, start_d, end_d,
            )
            label_keys = sorted({
                f"{r['trade_date']} {str(r['quote_time'])[:5]}" for r in rows
            })
            row_key = lambda r: f"{r['trade_date']} {str(r['quote_time'])[:5]}"

    # Pull selected metric value out of each row and stash full metric set
    def _metrics(r):
        return {"iv": r["iv"], "price": r["price"], "theta": r["theta"],
                "vega": r["vega"], "gamma": r["gamma"]}
    metric_value = lambda r: r[metric]

    # Bucket rows into series — one per (dte, put_delta) combo, but we always
    # have len(dte_list)*len(delta_list) == len(dte_list)+len(delta_list)-1 == n_lines
    if dimension == "delta":
        # one line per delta
        bucket: dict[int, dict[str, dict]] = {d: {} for d in delta_list}
        for r in rows:
            bucket[r["put_delta"]][row_key(r)] = {"value": metric_value(r), **_metrics(r)}
        series = []
        for d in delta_list:
            entries = bucket[d]
            series.append({
                "label":   _delta_label(d),
                "delta":   d,
                "dte":     fixed_dte,
                "labels":  label_keys,
                "values":  [entries.get(k, {}).get("value") for k in label_keys],
                "metrics": [entries.get(k) for k in label_keys],
            })
    else:
        # one line per dte
        bucket = {d: {} for d in dte_list}
        for r in rows:
            bucket[r["dte"]][row_key(r)] = {"value": metric_value(r), **_metrics(r)}
        series = []
        for d in dte_list:
            entries = bucket[d]
            series.append({
                "label":   f"{d}D",
                "dte":     d,
                "delta":   fixed_delta,
                "labels":  label_keys,
                "values":  [entries.get(k, {}).get("value") for k in label_keys],
                "metrics": [entries.get(k) for k in label_keys],
            })

    return {
        "freq":      freq,
        "dimension": dimension,
        "metric":    metric,
        "start":     start,
        "end":       end,
        "series":    series,
    }
