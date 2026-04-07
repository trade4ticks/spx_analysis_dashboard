"""
Meta endpoints — available dates, quote times, and surface grid constants.
Used by the frontend to populate dropdowns and chip selectors.
"""
from datetime import date as date_type

from fastapi import APIRouter, Depends, Query
from app.db import get_pool

router = APIRouter(tags=["meta"])


@router.get("/dates")
async def get_dates(pool=Depends(get_pool)) -> list[str]:
    """All distinct trade dates, newest first."""
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT DISTINCT trade_date FROM spx_surface ORDER BY trade_date DESC"
        )
    return [str(r["trade_date"]) for r in rows]


@router.get("/quote_times")
async def get_quote_times(
    date: str = Query(..., description="YYYY-MM-DD"),
    pool=Depends(get_pool),
) -> list[str]:
    """All distinct quote_times (HH:MM) available for a given trade_date."""
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT DISTINCT quote_time
            FROM spx_surface
            WHERE trade_date = $1
            ORDER BY quote_time
            """,
            date_type.fromisoformat(date),
        )
    # Return HH:MM strings
    return [str(r["quote_time"])[:5] for r in rows]


@router.get("/latest")
async def get_latest(pool=Depends(get_pool)) -> dict:
    """Most recent (trade_date, quote_time) snapshot in the database."""
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT trade_date, MAX(quote_time) AS quote_time
            FROM spx_surface
            WHERE trade_date = (SELECT MAX(trade_date) FROM spx_surface)
            GROUP BY trade_date
            """
        )
    if not row:
        return {"date": None, "time": None}
    return {"date": str(row["trade_date"]), "time": str(row["quote_time"])[:5]}


@router.get("/grid")
async def get_grid() -> dict:
    """Fixed surface grid constants (DTEs and put_deltas)."""
    return {
        "dtes":       [0,1,2,3,4,5,6,7,8,9,10,14,21,30,45,60,90,120,180,270,360],
        "put_deltas": list(range(5, 100, 5)),  # [5,10,...,95]
        # Curated subsets for UI chip defaults
        "common_dtes":    [7, 14, 30, 45, 60, 90, 120, 180],
        "common_deltas":  [10, 25, 35, 50, 65, 75, 90],
    }
