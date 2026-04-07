"""
Concavity / convexity endpoint.

Concavity = (IV_left + IV_right) / 2 − IV_center

Positive → smile (wings elevated vs center)
Negative → one-sided skew dominates

GET /api/concavity
  dte, left_delta, center_delta, right_delta
  start, end, target_time
  freq: daily | intraday
"""
from datetime import date as date_type
from fastapi import APIRouter, Depends, Query, HTTPException
from app.db import get_pool

router = APIRouter(tags=["concavity"])

INTRADAY_MAX_DAYS = 90


@router.get("")
async def get_concavity(
    dte:          int = Query(30),
    left_delta:   int = Query(25,  description="OTM put side put_delta"),
    center_delta: int = Query(50,  description="Center (ATM) put_delta"),
    right_delta:  int = Query(75,  description="OTM call side put_delta"),
    start:        str = Query(...),
    end:          str = Query(...),
    target_time:  str = Query("15:45"),
    freq:         str = Query("daily"),
    pool=Depends(get_pool),
) -> dict:
    """
    Time series of skew concavity for a fixed DTE.

    Queries the three delta levels (left, center, right) and returns
    concavity = (IV_left + IV_right) / 2 − IV_center at each timestamp.
    """
    start_d = date_type.fromisoformat(start)
    end_d   = date_type.fromisoformat(end)

    if freq == "intraday" and (end_d - start_d).days > INTRADAY_MAX_DAYS:
        raise HTTPException(
            400,
            f"Intraday mode limited to {INTRADAY_MAX_DAYS} days."
        )

    needed_deltas = [left_delta, center_delta, right_delta]

    async with pool.acquire() as conn:
        if freq == "daily":
            rows = await conn.fetch(
                """
                WITH closest AS (
                    SELECT DISTINCT ON (trade_date, put_delta)
                        trade_date, put_delta, iv
                    FROM spx_surface
                    WHERE dte        = $1
                      AND put_delta  = ANY($2)
                      AND trade_date BETWEEN $3 AND $4
                    ORDER BY trade_date, put_delta,
                             ABS(EXTRACT(EPOCH FROM (quote_time - $5::time)))
                )
                SELECT
                    l.trade_date                           AS label,
                    (l.iv + r.iv) / 2.0 - c.iv           AS concavity,
                    l.iv                                   AS iv_left,
                    c.iv                                   AS iv_center,
                    r.iv                                   AS iv_right
                FROM closest l
                JOIN closest c ON c.trade_date = l.trade_date AND c.put_delta = $6
                JOIN closest r ON r.trade_date = l.trade_date AND r.put_delta = $7
                WHERE l.put_delta = $8
                ORDER BY l.trade_date
                """,
                dte, needed_deltas, str(start_d), str(end_d), target_time,
                center_delta, right_delta, left_delta,
            )
            labels     = [str(r["label"]) for r in rows]

        else:  # intraday
            rows = await conn.fetch(
                """
                WITH base AS (
                    SELECT trade_date, quote_time, put_delta, iv
                    FROM spx_surface
                    WHERE dte        = $1
                      AND put_delta  = ANY($2)
                      AND trade_date BETWEEN $3 AND $4
                )
                SELECT
                    l.trade_date || ' ' || LEFT(l.quote_time::text, 5) AS label,
                    (l.iv + r.iv) / 2.0 - c.iv AS concavity,
                    l.iv  AS iv_left,
                    c.iv  AS iv_center,
                    r.iv  AS iv_right
                FROM base l
                JOIN base c ON c.trade_date = l.trade_date
                           AND c.quote_time = l.quote_time
                           AND c.put_delta  = $5
                JOIN base r ON r.trade_date = l.trade_date
                           AND r.quote_time = l.quote_time
                           AND r.put_delta  = $6
                WHERE l.put_delta = $7
                ORDER BY l.trade_date, l.quote_time
                """,
                dte, needed_deltas, str(start_d), str(end_d),
                center_delta, right_delta, left_delta,
            )
            labels = [r["label"] for r in rows]

    concavity_vals = [r["concavity"] for r in rows]

    # Stats
    valid = [v for v in concavity_vals if v is not None]
    stats = None
    if valid:
        stats = {
            "current": round(valid[-1], 4) if valid else None,
            "mean":    round(sum(valid) / len(valid), 4),
            "minimum": round(min(valid), 4),
            "maximum": round(max(valid), 4),
        }

    return {
        "freq":         freq,
        "dte":          dte,
        "left_delta":   left_delta,
        "center_delta": center_delta,
        "right_delta":  right_delta,
        "labels":       labels,
        "concavity":    concavity_vals,
        "iv_left":      [r["iv_left"]   for r in rows],
        "iv_center":    [r["iv_center"] for r in rows],
        "iv_right":     [r["iv_right"]  for r in rows],
        "stats":        stats,
    }
