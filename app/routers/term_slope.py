"""
Term slope endpoint.

For a fixed put_delta and two DTEs (a < b), plots the IV slope along
the term structure between those two maturities over time:

    slope_t = (IV_b(t) - IV_a(t)) / (b - a)

Supports daily and intraday frequencies and a multi-delta mode (one
line per delta).

GET /api/term_slope
  delta | deltas
  dte_a, dte_b
  start, end, target_time, freq
"""
from datetime import date as date_type, time as time_type
from typing import Optional

from fastapi import APIRouter, Depends, Query, HTTPException
from app.db import get_pool

router = APIRouter(tags=["term_slope"])

INTRADAY_MAX_DAYS = 90


def _parse_ints(s: str) -> list[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def _delta_label(pd: int) -> str:
    if pd == 50:
        return "ATM"
    if pd < 50:
        return f"{pd}Δp"
    return f"{100 - pd}Δc"


@router.get("")
async def get_term_slope(
    delta:       Optional[int] = Query(None),
    deltas:      Optional[str] = Query(None),
    dte_a:       int = Query(7),
    dte_b:       int = Query(30),
    start:       str = Query(...),
    end:         str = Query(...),
    target_time: str = Query("15:45"),
    freq:        str = Query("daily"),
    pool=Depends(get_pool),
) -> dict:
    if deltas:
        delta_list = _parse_ints(deltas)
    elif delta is not None:
        delta_list = [delta]
    else:
        raise HTTPException(400, "Provide delta or deltas")

    if dte_a == dte_b:
        raise HTTPException(400, "dte_a and dte_b must differ")

    start_d = date_type.fromisoformat(start)
    end_d   = date_type.fromisoformat(end)
    if freq == "intraday" and (end_d - start_d).days > INTRADAY_MAX_DAYS:
        raise HTTPException(400, f"Intraday limited to {INTRADAY_MAX_DAYS} days")

    pair    = [dte_a, dte_b]
    divisor = float(dte_b - dte_a)

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
                SELECT trade_date::text AS label, put_delta,
                       a.iv AS iv_a, b.iv AS iv_b,
                       (b.iv - a.iv) / $8 AS slope
                FROM closest a
                JOIN closest b USING (trade_date, put_delta)
                WHERE a.dte = $6 AND b.dte = $7
                ORDER BY trade_date, put_delta
                """,
                pair, delta_list, start_d, end_d,
                time_type.fromisoformat(target_time),
                dte_a, dte_b, divisor,
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
                       a.put_delta, a.iv AS iv_a, b.iv AS iv_b,
                       (b.iv - a.iv) / $7 AS slope
                FROM base a
                JOIN base b USING (trade_date, quote_time, put_delta)
                WHERE a.dte = $5 AND b.dte = $6
                ORDER BY trade_date, quote_time, put_delta
                """,
                pair, delta_list, start_d, end_d, dte_a, dte_b, divisor,
            )

    seen = []
    seen_set = set()
    for r in rows:
        if r["label"] not in seen_set:
            seen.append(r["label"]); seen_set.add(r["label"])

    bucket: dict[int, dict[str, dict]] = {d: {} for d in delta_list}
    for r in rows:
        bucket[r["put_delta"]][r["label"]] = {
            "value": r["slope"], "iv_a": r["iv_a"], "iv_b": r["iv_b"],
        }

    series = []
    for d in delta_list:
        entries = bucket[d]
        series.append({
            "label":   _delta_label(d),
            "delta":   d,
            "labels":  seen,
            "values":  [entries.get(k, {}).get("value") for k in seen],
            "metrics": [
                ({
                    "iv_a": entries.get(k, {}).get("iv_a"),
                    "iv_b": entries.get(k, {}).get("iv_b"),
                } if entries.get(k) else None)
                for k in seen
            ],
        })

    return {
        "freq":      freq,
        "dimension": "delta",
        "dte_a":     dte_a,
        "dte_b":     dte_b,
        "series":    series,
    }
