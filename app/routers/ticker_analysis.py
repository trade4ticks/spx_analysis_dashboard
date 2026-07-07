"""
Ticker Analysis page — single-ticker view.

Additive, self-contained router mounted at /api/ticker-analysis. Kept
separate from /api/factor-analysis so nothing on the universe-wide Factor
Analysis surface is disturbed (see ticker_analysis_build_brief.md §0).

Phase 1 (this file): scaffold only — a /tickers endpoint so the control
bar's ticker selector populates from the same OI universe the Factor
Analysis page uses. The metric-pane, stat-strip, saved-layout, and chain
endpoints land in later phases under this same prefix.
"""
from fastapi import APIRouter, Depends

from app.db import get_oi_pool

router = APIRouter()


@router.get("/tickers")
async def list_tickers(pool=Depends(get_oi_pool)):
    """Distinct ticker universe from daily_features (OI DB).

    Mirrors GET /api/factor-analysis/tickers so this page's selector
    offers the identical universe. Returns [] when the OI pool is not
    configured, matching the Factor Analysis degradation behavior.
    """
    if not pool:
        return []
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT DISTINCT ticker FROM daily_features ORDER BY ticker"
        )
    return [r["ticker"] for r in rows]
