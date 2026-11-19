"""
Heatmap endpoints — full IV surface grid for a single snapshot.

GET /api/heatmap/iv          — raw IV per (dte, put_delta)
GET /api/heatmap/skew        — skew slope vs ATM forward per node
GET /api/heatmap/term        — forward vol anchored at 30 DTE per node
GET /api/heatmap/iv_zscore   — IV z-score per node vs its own 15:45 history,
                               ?lookback=90d|1y, window ending on the viewed date
"""
import logging
import math
from datetime import date as date_type, time as time_type, timedelta
from time import perf_counter

from fastapi import APIRouter, Depends, HTTPException, Query
from app.db import get_pool

router = APIRouter(tags=["heatmap"])

# uvicorn configures its own loggers but not the root logger, so INFO from a
# module-level logger would be silently dropped. Log through uvicorn's.
log = logging.getLogger("uvicorn.error")

# Z-score lookback, in calendar days ending on (and including) the trade_date
# being viewed, so scrolling back gives the z-score as it read that day. A
# closed set on purpose: anything else is a 400, never a fall-through to an
# unbounded scan of all history.
ZSCORE_LOOKBACK_DAYS = {"90d": 90, "1y": 365}

# Fewest 15:45 observations a node needs before its z-score is reported.
# stddev_samp is defined from 2, but a z built on a handful of days swings on
# noise. Below this the node's z is null; its n is still returned.
ZSCORE_MIN_DAYS = 20

# A stdev at or below this is float rounding over identical values, not
# dispersion (IV is ~0.05-1.0, so 1e-6 is a ten-thousandth of a vol point).
# Treated as zero: z is null rather than an enormous or infinite number.
ZSCORE_MIN_STDEV = 1e-6


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


# ── IV z-score vs each node's own history ─────────────────────────────────────

@router.get("/iv_zscore")
async def heatmap_iv_zscore(
    date:     str = Query(...),
    time:     str = Query(...),
    lookback: str = Query("90d"),
    pool=Depends(get_pool),
) -> dict:
    """
    Per (dte, put_delta): (iv at the viewed snapshot − mean) / stddev_samp,
    where mean and stdev come from that node's exact-15:45 IV on each trade
    date in the lookback window. The window ends on — and includes — the
    viewed date. Days without a 15:45 row (holidays, half-days) drop out; no
    fallback. z is null when n < ZSCORE_MIN_DAYS or the stdev is ~zero.

    Viewing a 15:45 snapshot, the viewed value is IN its own history (the
    window includes the viewed date), which pulls z slightly toward 0 --
    by design: the window is "as it read that day". The page colours z on a
    FIXED +/-3 scale, so a colour means the same on every date (heatmap.js
    header). Gates: check_heatmap (handler + page), check_heatmap_sql
    (real SQL; --live read-only on the VPS).
    """
    days = ZSCORE_LOOKBACK_DAYS.get(lookback)
    if days is None:
        raise HTTPException(
            status_code=400,
            detail=f"lookback must be one of {list(ZSCORE_LOOKBACK_DAYS)}; "
                   f"got {lookback!r}",
        )

    # A malformed date or time is the caller's error (400), not a 500.
    try:
        end_day   = date_type.fromisoformat(date)
        snap_time = time_type.fromisoformat(time)
    except ValueError:
        raise HTTPException(status_code=400,
                            detail=f"date must be YYYY-MM-DD and time HH:MM; got {date!r}, {time!r}")
    # Inclusive on both ends, so the window is exactly `days` calendar days.
    start_day = end_day - timedelta(days=days - 1)

    t0 = perf_counter()
    try:
        async with pool.acquire() as conn:
            t1 = perf_counter()
            rows = await conn.fetch(
                """
                WITH hist AS (
                    SELECT dte, put_delta,
                           count(iv)               AS n,
                           avg(iv::float8)         AS mean,
                           stddev_samp(iv::float8) AS sd
                    FROM spx_surface
                    WHERE trade_date BETWEEN $1 AND $2
                      AND quote_time = '15:45'
                    GROUP BY dte, put_delta
                )
                SELECT s.dte, s.put_delta, s.iv::float8 AS iv, h.n, h.mean, h.sd,
                       CASE WHEN h.n >= $4::int AND h.sd > $5::float8
                            THEN (s.iv::float8 - h.mean) / h.sd
                       END AS z
                FROM spx_surface s
                LEFT JOIN hist h ON h.dte = s.dte AND h.put_delta = s.put_delta
                WHERE s.trade_date = $2
                  AND s.quote_time = $3::time
                ORDER BY s.dte, s.put_delta
                """,
                start_day,
                end_day,
                snap_time,
                ZSCORE_MIN_DAYS,
                ZSCORE_MIN_STDEV,
            )
    except Exception as exc:
        # uvicorn logs the traceback itself; this adds what it can't know.
        log.error(
            "iv_zscore failed: lookback=%s window=%s..%s time=%s after %.0f ms — %s: %s",
            lookback, start_day, end_day, snap_time, (perf_counter() - t0) * 1000,
            type(exc).__name__, exc,
        )
        raise
    t2 = perf_counter()

    current = [
        {
            "dte":       r["dte"],
            "put_delta": r["put_delta"],
            "v":         r["z"],
            "iv":        r["iv"],
            "mean":      r["mean"],
            "sd":        r["sd"],
            "n":         r["n"] or 0,
        }
        for r in rows
    ]
    # Most 15:45 observations any node has: the number of days in the window
    # with a 15:45 snapshot, unless a day is missing every node. Cheaper than a
    # second pass for count(DISTINCT trade_date).
    max_n = max((c["n"] for c in current), default=0)
    null_z = sum(1 for c in current if c["v"] is None)

    log.info(
        "iv_zscore lookback=%s window=%s..%s time=%s nodes=%d days=%d null_z=%d "
        "acquire=%.0f ms query=%.0f ms",
        lookback, start_day, end_day, snap_time, len(current), max_n, null_z,
        (t1 - t0) * 1000, (t2 - t1) * 1000,
    )

    return {
        "mode":     "iv_zscore",
        "lookback": lookback,
        "window": {
            "start":    start_day.isoformat(),
            "end":      end_day.isoformat(),
            "days":     max_n,
            "min_days": ZSCORE_MIN_DAYS,
        },
        "current": current,
    }
