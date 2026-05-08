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


# Process-memory cache for the catalog. The table rarely changes;
# refetching on every page load is wasteful. Cleared by app restart.
_columns_catalog_cache: list[dict] | None = None


@router.get("/columns-catalog")
async def get_columns_catalog(pool=Depends(get_pool)) -> list[dict]:
    """Per-column semantic metadata for surface_metrics_core.
    Used by the frontend to group / describe / format columns in
    filter dropdowns, section metric pickers, and tooltips."""
    global _columns_catalog_cache
    if _columns_catalog_cache is not None:
        return _columns_catalog_cache
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """SELECT column_name, family, tenor, wing, form,
                      base_column, units, description, formula
               FROM surface_metrics_catalog
               ORDER BY family, tenor, wing, form, column_name"""
        )
    _columns_catalog_cache = [dict(r) for r in rows]
    return _columns_catalog_cache


@router.get("/value-trail")
async def get_value_trail(
    col:  str = Query(..., description="Column name from surface_metrics_catalog"),
    days: int = Query(5,   ge=1, le=30, description="Latest day plus prior days"),
    pool=Depends(get_pool),
) -> dict:
    """Trail of one column's latest-quote-of-day values for the last N
    trading days. Used by the Backtest IV heatmap to draw a path of
    recent positions, with today as the head."""
    catalog = await get_columns_catalog(pool)
    valid_cols = {r['column_name'] for r in catalog}
    if col not in valid_cols:
        return {'col': col, 'trail': []}
    # Column name is whitelisted, safe to interpolate.
    sql = (
        f'WITH latest_per_day AS ('
        f'  SELECT trade_date, MAX(quote_time) AS quote_time '
        f'  FROM surface_metrics_core '
        f'  WHERE "{col}" IS NOT NULL '
        f'  GROUP BY trade_date '
        f'  ORDER BY trade_date DESC '
        f'  LIMIT $1'
        f') '
        f'SELECT smc.trade_date, smc.quote_time, smc."{col}" AS value '
        f'FROM surface_metrics_core smc '
        f'JOIN latest_per_day lpd USING (trade_date, quote_time) '
        f'ORDER BY smc.trade_date'
    )
    async with pool.acquire() as conn:
        rows = await conn.fetch(sql, days)
    return {
        'col':   col,
        'trail': [
            {'date': str(r['trade_date']),
             'time': str(r['quote_time'])[:5],
             'value': float(r['value'])}
            for r in rows
        ],
    }


@router.get("/today-value")
async def get_today_value(
    col: str = Query(..., description="Column name from surface_metrics_catalog"),
    pool=Depends(get_pool),
) -> dict:
    """Latest value of a single surface_metrics_core column. Used by the
    Backtest IV page to mark "today's value" on decile charts and heatmaps.
    Trade-derived columns (entry_pos_*, portfolio_*, premium, etc.) are not
    in the catalog, so this returns null for them — the UI hides the marker
    in those cases.
    """
    catalog = await get_columns_catalog(pool)
    valid_cols = {r['column_name'] for r in catalog}
    if col not in valid_cols:
        return {'col': col, 'date': None, 'time': None, 'value': None,
                'reason': 'not in surface_metrics_catalog'}
    # Column name is now whitelisted — safe to interpolate.
    sql = (
        f'SELECT trade_date, quote_time, "{col}" AS value '
        f'FROM surface_metrics_core '
        f'WHERE trade_date = (SELECT MAX(trade_date) FROM surface_metrics_core) '
        f'  AND "{col}" IS NOT NULL '
        f'ORDER BY quote_time DESC LIMIT 1'
    )
    async with pool.acquire() as conn:
        row = await conn.fetchrow(sql)
    if not row or row['value'] is None:
        return {'col': col, 'date': None, 'time': None, 'value': None}
    return {
        'col':   col,
        'date':  str(row['trade_date']),
        'time':  str(row['quote_time'])[:5],
        'value': float(row['value']),
    }
