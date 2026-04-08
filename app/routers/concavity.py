"""
Convexity (formerly concavity) endpoint.

Weighted curvature against the chord at the center delta:

    w_left  = (d_right  - d_center) / (d_right - d_left)
    w_right = (d_center - d_left  ) / (d_right - d_left)
    convexity = (w_left * IV_left + w_right * IV_right) - IV_center

This is the linear interpolation of left/right at the center delta
minus the actual center IV. Reduces to (IV_left+IV_right)/2 - IV_center
when wings are evenly spaced, but stays correct when they aren't.

Positive  -> wings interpolated above center (smile / convex up)
Negative  -> center above the chord (frown)

GET /api/convexity
  dte           single DTE      (mutually exclusive with dtes)
  dtes          comma DTEs      (multi-DTE mode)
  left_delta, center_delta, right_delta
  start, end, target_time
  freq          daily | intraday
"""
from datetime import date as date_type, time as time_type
from typing import Optional

from fastapi import APIRouter, Depends, Query, HTTPException
from app.db import get_pool

router = APIRouter(tags=["convexity"])

INTRADAY_MAX_DAYS = 90


def _parse_ints(s: str) -> list[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


@router.get("")
async def get_convexity(
    dte:          Optional[int] = Query(None),
    dtes:         Optional[str] = Query(None, description="Comma DTEs (multi-DTE mode)"),
    left_delta:   int = Query(25),
    center_delta: int = Query(50),
    right_delta:  int = Query(75),
    start:        str = Query(...),
    end:          str = Query(...),
    target_time:  str = Query("15:45"),
    freq:         str = Query("daily"),
    pool=Depends(get_pool),
) -> dict:
    if dtes:
        dte_list = _parse_ints(dtes)
    elif dte is not None:
        dte_list = [dte]
    else:
        raise HTTPException(400, "Provide dte or dtes")

    if not dte_list:
        raise HTTPException(400, "DTE list must not be empty")

    if not (left_delta < center_delta < right_delta):
        raise HTTPException(400, "Require left_delta < center_delta < right_delta")

    span    = right_delta - left_delta
    w_left  = (right_delta  - center_delta) / span
    w_right = (center_delta - left_delta)   / span

    start_d = date_type.fromisoformat(start)
    end_d   = date_type.fromisoformat(end)

    if freq == "intraday" and (end_d - start_d).days > INTRADAY_MAX_DAYS:
        raise HTTPException(400, f"Intraday limited to {INTRADAY_MAX_DAYS} days")

    needed_deltas = [left_delta, center_delta, right_delta]

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
                SELECT trade_date::text                          AS label,
                       dte,
                       l.iv                                      AS iv_left,
                       c.iv                                      AS iv_center,
                       r.iv                                      AS iv_right,
                       (l.iv * $9 + r.iv * $10) - c.iv           AS convexity
                FROM closest l
                JOIN closest c USING (trade_date, dte)
                JOIN closest r USING (trade_date, dte)
                WHERE l.put_delta = $6
                  AND c.put_delta = $7
                  AND r.put_delta = $8
                ORDER BY trade_date, dte
                """,
                dte_list, needed_deltas, start_d, end_d,
                time_type.fromisoformat(target_time),
                left_delta, center_delta, right_delta,
                w_left, w_right,
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
                SELECT (l.trade_date::text || ' ' || LEFT(l.quote_time::text, 5)) AS label,
                       l.dte,
                       l.iv  AS iv_left,
                       c.iv  AS iv_center,
                       r.iv  AS iv_right,
                       (l.iv * $8 + r.iv * $9) - c.iv AS convexity
                FROM base l
                JOIN base c USING (trade_date, quote_time, dte)
                JOIN base r USING (trade_date, quote_time, dte)
                WHERE l.put_delta = $5
                  AND c.put_delta = $6
                  AND r.put_delta = $7
                ORDER BY l.trade_date, l.quote_time, l.dte
                """,
                dte_list, needed_deltas, start_d, end_d,
                left_delta, center_delta, right_delta,
                w_left, w_right,
            )

    # Distinct labels in order seen
    seen = []
    seen_set = set()
    for r in rows:
        if r["label"] not in seen_set:
            seen.append(r["label"]); seen_set.add(r["label"])

    # Bucket per DTE
    bucket: dict[int, dict[str, dict]] = {d: {} for d in dte_list}
    for r in rows:
        bucket[r["dte"]][r["label"]] = {
            "value":  r["convexity"],
            "left":   r["iv_left"],
            "center": r["iv_center"],
            "right":  r["iv_right"],
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
                    "iv_left":   entries.get(k, {}).get("left"),
                    "iv_center": entries.get(k, {}).get("center"),
                    "iv_right":  entries.get(k, {}).get("right"),
                } if entries.get(k) else None)
                for k in seen
            ],
        })

    return {
        "freq":         freq,
        "dimension":    "dte",
        "left_delta":   left_delta,
        "center_delta": center_delta,
        "right_delta":  right_delta,
        "series":       series,
    }
