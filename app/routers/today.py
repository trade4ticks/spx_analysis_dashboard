"""
Today page endpoints — intraday IV grid keyed by actual strike.

GET /api/today/expirations?date=YYYY-MM-DD
    List of {dte, expiry, label} for all expirations available on that date.

GET /api/today/iv_grid?date=YYYY-MM-DD&dte=N
    Full intraday (quote_time, strike) → iv grid for the chosen expiration.
    Strike range is anchored to the ATM strike at the first quote_time of the day
    (floor to nearest 100, then 10 nodes below and 5 nodes above in steps of 100).
    Also returns prev-day 16:00 IV per strike for 5-min change at the 9:35 row.
"""
from datetime import date as date_type, timedelta

from fastapi import APIRouter, Depends, Query, HTTPException
from app.db import get_pool

router = APIRouter(tags=["today"])


@router.get("/expirations")
async def get_expirations(
    date: str = Query(...),
    pool=Depends(get_pool),
) -> list[dict]:
    trade_date = date_type.fromisoformat(date)
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT DISTINCT dte
            FROM spx_surface
            WHERE trade_date = $1
            ORDER BY dte
            """,
            trade_date,
        )
    result = []
    for r in rows:
        dte    = r["dte"]
        expiry = trade_date + timedelta(days=dte)
        label  = f"{expiry.strftime('%b %d')} ({dte} DTE)"
        result.append({"dte": dte, "expiry": str(expiry), "label": label})
    return result


@router.get("/iv_grid")
async def get_iv_grid(
    date: str = Query(...),
    dte:  int  = Query(...),
    pool=Depends(get_pool),
) -> dict:
    trade_date  = date_type.fromisoformat(date)
    expiry_date = trade_date + timedelta(days=dte)

    async with pool.acquire() as conn:
        # ── 1. Determine strike range from ATM at opening ────────────────────
        atm_row = await conn.fetchrow(
            """
            SELECT atm_strike
            FROM spx_atm
            WHERE trade_date = $1 AND dte = $2
            ORDER BY ABS(EXTRACT(EPOCH FROM (quote_time - '09:35'::time)))
            LIMIT 1
            """,
            trade_date, dte,
        )

        if atm_row and atm_row["atm_strike"] is not None:
            spot = float(atm_row["atm_strike"])
        else:
            # Fallback: delta=50 strike from surface
            fallback = await conn.fetchrow(
                """
                SELECT strike
                FROM spx_surface
                WHERE trade_date = $1 AND dte = $2 AND put_delta = 50
                ORDER BY ABS(EXTRACT(EPOCH FROM (quote_time - '09:35'::time)))
                LIMIT 1
                """,
                trade_date, dte,
            )
            if not fallback or fallback["strike"] is None:
                raise HTTPException(400, "Cannot determine opening spot for this date/dte")
            spot = float(fallback["strike"])

        ref     = int(spot / 100) * 100          # floor to nearest 100
        strikes = [ref - (9 - i) * 100 for i in range(10)] + \
                  [ref + j * 100          for j in range(1, 6)]
        # = [ref-900, ref-800, …, ref, ref+100, …, ref+500]  (15 strikes)

        # ── 2. Query intraday IV for those strikes ───────────────────────────
        rows = await conn.fetch(
            """
            SELECT quote_time, strike::int AS strike, iv
            FROM spx_surface
            WHERE trade_date = $1
              AND dte         = $2
              AND strike      = ANY($3::int[])
            ORDER BY quote_time, strike
            """,
            trade_date, dte, strikes,
        )

        # ── 3. Previous trading day 16:00 reference for 09:35 change ────────
        prev_date_row = await conn.fetchrow(
            "SELECT MAX(trade_date) AS dt FROM spx_surface WHERE trade_date < $1",
            trade_date,
        )
        prev_data: dict[str, float] = {}
        if prev_date_row and prev_date_row["dt"]:
            prev_date = prev_date_row["dt"]
            prev_dte  = (expiry_date - prev_date).days

            prev_rows = await conn.fetch(
                """
                SELECT DISTINCT ON (strike)
                    strike::int AS strike, iv
                FROM spx_surface
                WHERE trade_date = $1
                  AND dte         = $2
                  AND strike      = ANY($3::int[])
                ORDER BY strike,
                         ABS(EXTRACT(EPOCH FROM (quote_time - '16:00'::time)))
                """,
                prev_date, prev_dte, strikes,
            )
            prev_data = {
                str(r["strike"]): float(r["iv"])
                for r in prev_rows if r["iv"] is not None
            }

    # ── 4. Build time-keyed grid ─────────────────────────────────────────────
    time_map: dict[str, dict[str, float]] = {}
    for r in rows:
        t  = str(r["quote_time"])[:5]       # HH:MM
        sk = str(r["strike"])
        if t not in time_map:
            time_map[t] = {}
        if r["iv"] is not None:
            time_map[t][sk] = float(r["iv"])

    return {
        "strikes": strikes,
        "rows":    [{"time": t, "data": time_map[t]} for t in sorted(time_map)],
        "prev":    prev_data,
    }
