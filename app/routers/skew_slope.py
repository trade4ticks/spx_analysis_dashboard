"""
Skew slope endpoint.

For a fixed DTE and two put_deltas (a < b), plots the IV slope between
those two nodes over time:

    slope_t = (IV_b(t) - IV_a(t)) / (b - a)

Supports daily and intraday frequencies and a multi-DTE mode (one
line per DTE) so the user can compare slope across maturities.

GET /api/skew_slope
  dte | dtes
  delta_a, delta_b      put_delta values (integers)
  start, end, target_time, freq
"""
from datetime import date as date_type, time as time_type
from typing import Optional

from fastapi import APIRouter, Depends, Query, HTTPException
from app.db import get_pool

router = APIRouter(tags=["skew_slope"])

INTRADAY_MAX_DAYS = 90


def _parse_ints(s: str) -> list[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


@router.get("")
async def get_skew_slope(
    dte:         Optional[int] = Query(None),
    dtes:        Optional[str] = Query(None),
    delta_a:     int = Query(25),
    delta_b:     int = Query(50),
    start:       str = Query(...),
    end:         str = Query(...),
    target_time: str = Query("15:45"),
    freq:        str = Query("daily"),
    pool=Depends(get_pool),
) -> dict:
    if dtes:
        dte_list = _parse_ints(dtes)
    elif dte is not None:
        dte_list = [dte]
    else:
        raise HTTPException(400, "Provide dte or dtes")

    if delta_a == delta_b:
        raise HTTPException(400, "delta_a and delta_b must differ")

    start_d = date_type.fromisoformat(start)
    end_d   = date_type.fromisoformat(end)
    if freq == "intraday" and (end_d - start_d).days > INTRADAY_MAX_DAYS:
        raise HTTPException(400, f"Intraday limited to {INTRADAY_MAX_DAYS} days")

    pair    = [delta_a, delta_b]
    divisor = float(delta_b - delta_a)

    async with pool.acquire() as conn:
        if freq == "daily":
            rows = await conn.fetch(
                """
                WITH closest AS (
                    SELECT DISTINCT ON (trade_date, dte, put_delta)
                        trade_date, dte, put_delta, iv
                    FROM spx_surface
                    WHERE dte        = ANY($1)
                      AND put_delta  = ANY($2)
                      AND trade_date BETWEEN $3 AND $4
                    ORDER BY trade_date, dte, put_delta,
                             ABS(EXTRACT(EPOCH FROM (quote_time - $5::time)))
                )
                SELECT trade_date::text AS label, dte,
                       a.iv AS iv_a, b.iv AS iv_b,
                       (b.iv - a.iv) / $8 AS slope
                FROM closest a
                JOIN closest b USING (trade_date, dte)
                WHERE a.put_delta = $6 AND b.put_delta = $7
                ORDER BY trade_date, dte
                """,
                dte_list, pair, start_d, end_d,
                time_type.fromisoformat(target_time),
                delta_a, delta_b, divisor,
            )
        else:
            rows = await conn.fetch(
                """
                WITH base AS (
                    SELECT trade_date, quote_time, dte, put_delta, iv
                    FROM spx_surface
                    WHERE dte        = ANY($1)
                      AND put_delta  = ANY($2)
                      AND trade_date BETWEEN $3 AND $4
                )
                SELECT (a.trade_date::text || ' ' || LEFT(a.quote_time::text, 5)) AS label,
                       a.dte, a.iv AS iv_a, b.iv AS iv_b,
                       (b.iv - a.iv) / $7 AS slope
                FROM base a
                JOIN base b USING (trade_date, quote_time, dte)
                WHERE a.put_delta = $5 AND b.put_delta = $6
                ORDER BY trade_date, quote_time, dte
                """,
                dte_list, pair, start_d, end_d, delta_a, delta_b, divisor,
            )

    seen = []
    seen_set = set()
    for r in rows:
        if r["label"] not in seen_set:
            seen.append(r["label"]); seen_set.add(r["label"])

    bucket: dict[int, dict[str, dict]] = {d: {} for d in dte_list}
    for r in rows:
        bucket[r["dte"]][r["label"]] = {
            "value": r["slope"], "iv_a": r["iv_a"], "iv_b": r["iv_b"],
        }

    series = []
    for d in dte_list:
        entries = bucket[d]
        series.append({
            "label":   f"{d}D",
            "dte":     d,
            "labels":  seen,
            "values":  [entries.get(k, {}).get("value") for k in seen],
            "metrics": [
                ({
                    "iv_a":  entries.get(k, {}).get("iv_a"),
                    "iv_b":  entries.get(k, {}).get("iv_b"),
                } if entries.get(k) else None)
                for k in seen
            ],
        })

    return {
        "freq":      freq,
        "dimension": "dte",
        "delta_a":   delta_a,
        "delta_b":   delta_b,
        "series":    series,
    }
