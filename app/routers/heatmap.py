"""
Heatmap endpoints — full IV surface grid for a single snapshot.

GET /api/heatmap/iv          — raw IV per (dte, put_delta)
GET /api/heatmap/skew        — skew slope vs ATM forward per node
GET /api/heatmap/term        — forward vol anchored at 30 DTE per node
GET /api/heatmap/node_stats  — historical IV percentiles per node (for coloring)
"""
import math
from datetime import date as date_type, time as time_type

from fastapi import APIRouter, Depends, Query
from app.db import get_pool

router = APIRouter(tags=["heatmap"])


# ── IV ────────────────────────────────────────────────────────────────────────

@router.get("/iv")
async def heatmap_iv(
    date:      str = Query(...),
    time:      str = Query(...),
    prev_date: str = Query(None),
    prev_time: str = Query("15:45"),
    pool=Depends(get_pool),
) -> dict:
    async with pool.acquire() as conn:
        current = await _fetch_iv(conn, date, time)
        prev    = await _fetch_iv(conn, prev_date, prev_time) if prev_date else None
    return {"mode": "iv", "current": current, "prev": prev}


async def _fetch_iv(conn, date: str, time: str) -> list[dict]:
    rows = await conn.fetch(
        """
        SELECT dte, put_delta, iv
        FROM spx_surface
        WHERE trade_date = $1 AND quote_time = $2::time
        ORDER BY dte, put_delta
        """,
        date_type.fromisoformat(date),
        time_type.fromisoformat(time),
    )
    return [{"dte": r["dte"], "put_delta": r["put_delta"], "v": float(r["iv"])} for r in rows]


# ── Skew slope vs ATM forward ─────────────────────────────────────────────────

@router.get("/skew")
async def heatmap_skew(
    date:      str = Query(...),
    time:      str = Query(...),
    prev_date: str = Query(None),
    prev_time: str = Query("15:45"),
    pool=Depends(get_pool),
) -> dict:
    async with pool.acquire() as conn:
        current = await _fetch_skew(conn, date, time)
        prev    = await _fetch_skew(conn, prev_date, prev_time) if prev_date else None
    return {"mode": "skew", "current": current, "prev": prev}


async def _fetch_skew(conn, date: str, time: str) -> list[dict]:
    rows = await conn.fetch(
        """
        SELECT s.dte, s.put_delta, s.iv, s.strike,
               a.atm_iv, a.atm_strike
        FROM spx_surface s
        LEFT JOIN spx_atm a
               ON a.trade_date = s.trade_date
              AND a.quote_time = s.quote_time
              AND a.dte        = s.dte
        WHERE s.trade_date = $1
          AND s.quote_time  = $2::time
        ORDER BY s.dte, s.put_delta
        """,
        date_type.fromisoformat(date),
        time_type.fromisoformat(time),
    )

    result = []
    for r in rows:
        slope = None
        dte = r["dte"]
        if dte > 0 and r["atm_iv"] is not None and r["atm_strike"] and r["strike"]:
            try:
                T      = dte / 365.0
                ratio  = float(r["strike"]) / float(r["atm_strike"])
                if ratio > 0:
                    slope = round(
                        math.sqrt(T) * (float(r["iv"]) - float(r["atm_iv"])) / math.log(ratio),
                        4,
                    )
            except (ValueError, ZeroDivisionError):
                pass
        result.append({"dte": dte, "put_delta": r["put_delta"], "v": slope})
    return result


# ── Term slope anchored at 30 DTE ─────────────────────────────────────────────

@router.get("/term")
async def heatmap_term(
    date:      str = Query(...),
    time:      str = Query(...),
    prev_date: str = Query(None),
    prev_time: str = Query("15:45"),
    pool=Depends(get_pool),
) -> dict:
    async with pool.acquire() as conn:
        current = await _fetch_term(conn, date, time)
        prev    = await _fetch_term(conn, prev_date, prev_time) if prev_date else None
    return {"mode": "term", "current": current, "prev": prev}


async def _fetch_term(conn, date: str, time: str) -> list[dict]:
    rows = await conn.fetch(
        """
        WITH snap AS (
            SELECT s.dte, s.put_delta,
                   CASE WHEN s.put_delta = 50
                        THEN COALESCE(a.atm_iv, s.iv)
                        ELSE s.iv END AS iv
            FROM spx_surface s
            LEFT JOIN spx_atm a
                   ON a.trade_date = s.trade_date
                  AND a.quote_time = s.quote_time
                  AND a.dte        = s.dte
            WHERE s.trade_date = $1
              AND s.quote_time  = $2::time
        )
        SELECT s.dte, s.put_delta,
               s.iv                   AS iv_dte,
               r.iv                   AS iv_30
        FROM snap s
        LEFT JOIN snap r ON r.put_delta = s.put_delta AND r.dte = 30
        ORDER BY s.dte, s.put_delta
        """,
        date_type.fromisoformat(date),
        time_type.fromisoformat(time),
    )

    result = []
    for r in rows:
        dte    = r["dte"]
        iv_dte = r["iv_dte"]
        iv_30  = r["iv_30"]

        fwd_vol = None
        if dte != 30 and iv_dte is not None and iv_30 is not None:
            try:
                T_dte = float(dte) / 365.0
                T_30  = 30.0 / 365.0
                if dte < 30:
                    T_a, T_b = T_dte, T_30
                    iv_a, iv_b = float(iv_dte), float(iv_30)
                else:
                    T_a, T_b = T_30, T_dte
                    iv_a, iv_b = float(iv_30), float(iv_dte)
                dT = T_b - T_a
                if dT > 0:
                    fwd_var = (iv_b**2 * T_b - iv_a**2 * T_a) / dT
                    fwd_vol = round(math.sqrt(fwd_var), 4) if fwd_var > 0 else None
            except (ValueError, ZeroDivisionError):
                pass

        result.append({"dte": dte, "put_delta": r["put_delta"], "v": fwd_vol})
    return result


# ── Per-node IV statistics for colour scaling ─────────────────────────────────

@router.get("/node_stats")
async def heatmap_node_stats(pool=Depends(get_pool)) -> dict:
    """
    Historical IV percentiles (p05, p50, p95) per (dte, put_delta),
    using the daily closest-to-15:45 snapshot for each trade date.
    p50 is used as the grey midpoint; p05/p95 pin the colour extremes.
    """
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT dte, put_delta,
                   PERCENTILE_CONT(0.05) WITHIN GROUP (ORDER BY iv) AS p05,
                   PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY iv) AS p50,
                   PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY iv) AS p95
            FROM (
                SELECT DISTINCT ON (trade_date, dte, put_delta)
                    dte, put_delta, iv
                FROM spx_surface
                ORDER BY trade_date, dte, put_delta,
                         ABS(EXTRACT(EPOCH FROM (quote_time - '15:45'::time)))
            ) AS daily
            GROUP BY dte, put_delta
            ORDER BY dte, put_delta
            """
        )

    return {
        f"{r['dte']}_{r['put_delta']}": {
            "p05": float(r["p05"]) if r["p05"] is not None else None,
            "p50": float(r["p50"]) if r["p50"] is not None else None,
            "p95": float(r["p95"]) if r["p95"] is not None else None,
        }
        for r in rows
    }
