"""Backtest IV Analysis API endpoints."""
import asyncio
import json
import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from app.db import get_pool
from research import iv_analysis as iv
from research.backtest import TRADE_FIELDS

log    = logging.getLogger(__name__)
router = APIRouter()

# In-memory trade cache: {upload_id: (trades, iv_columns)}
_cache: dict = {}
_MAX_CACHE = 6


async def _load_trades(upload_id: str, pool) -> tuple:
    if upload_id in _cache:
        return _cache[upload_id]

    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT data FROM research_backtest_uploads WHERE id = $1::uuid",
            upload_id,
        )
    if not row:
        raise HTTPException(404, f"Backtest upload '{upload_id}' not found")

    data   = row['data']
    trades = json.loads(data) if isinstance(data, str) else data
    if not trades:
        raise HTTPException(400, "Upload has no trade data")

    # IV columns = all keys not in TRADE_FIELDS
    iv_columns = sorted(
        k for k in trades[0].keys()
        if k not in TRADE_FIELDS and not k.startswith('_') and k != 'daily_path'
    )

    if len(_cache) >= _MAX_CACHE:
        del _cache[next(iter(_cache))]

    _cache[upload_id] = (trades, iv_columns)
    return trades, iv_columns


# ── Request models ─────────────────────────────────────────────────────────────

class HeatmapRequest(BaseModel):
    metric_a: str
    metric_b: str
    n_buckets: int = 5


class DeltaR2Request(BaseModel):
    metrics: list[str]
    target: str = 'pnl'


class DecileRequest(BaseModel):
    metric: str
    n_buckets: int = 10


class ConditionalSliceRequest(BaseModel):
    fix_metric:    str
    fix_bucket:    int
    fix_n_buckets: int = 5
    vary_metric:   str
    vary_n_buckets: int = 5


class DistributionRequest(BaseModel):
    metric:       Optional[str] = None
    bucket_index: Optional[int] = None
    n_buckets:    Optional[int] = None


class TimeStabilityRequest(BaseModel):
    metric:    str
    n_windows: int = 6


class FeatureCorrelationRequest(BaseModel):
    metrics: list[str]


# ── Endpoints ──────────────────────────────────────────────────────────────────

@router.get("/uploads")
async def list_uploads(pool=Depends(get_pool)):
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """SELECT id::text, name, source, trade_count, matched_count,
                      date_from, date_to, strategies, created_at
               FROM research_backtest_uploads
               ORDER BY created_at DESC LIMIT 20""",
        )
    result = []
    for r in rows:
        d = dict(r)
        s = d.get('strategies')
        d['strategies'] = (json.loads(s) if isinstance(s, str) else s) or []
        result.append(d)
    return result


@router.get("/{upload_id}/columns")
async def get_columns(upload_id: str, pool=Depends(get_pool)):
    trades, iv_columns = await _load_trades(upload_id, pool)
    return {'iv_columns': iv_columns, 'trade_count': len(trades)}


@router.post("/{upload_id}/heatmap")
async def heatmap(upload_id: str, req: HeatmapRequest, pool=Depends(get_pool)):
    if req.n_buckets not in (3, 5, 10):
        raise HTTPException(400, "n_buckets must be 3, 5, or 10")
    trades, _ = await _load_trades(upload_id, pool)
    return await asyncio.to_thread(
        iv.compute_heatmap, trades, req.metric_a, req.metric_b, req.n_buckets
    )


@router.post("/{upload_id}/delta-r2")
async def delta_r2(upload_id: str, req: DeltaR2Request, pool=Depends(get_pool)):
    if len(req.metrics) > 20:
        raise HTTPException(400, "Maximum 20 metrics for ΔR² grid")
    if len(req.metrics) < 2:
        raise HTTPException(400, "At least 2 metrics required")
    trades, _ = await _load_trades(upload_id, pool)
    return await asyncio.to_thread(
        iv.compute_delta_r2_grid, trades, req.metrics, req.target
    )


@router.post("/{upload_id}/decile")
async def decile(upload_id: str, req: DecileRequest, pool=Depends(get_pool)):
    trades, _ = await _load_trades(upload_id, pool)
    return iv.compute_decile_stats(trades, req.metric, req.n_buckets)


@router.post("/{upload_id}/conditional-slice")
async def conditional_slice(upload_id: str, req: ConditionalSliceRequest,
                             pool=Depends(get_pool)):
    trades, _ = await _load_trades(upload_id, pool)
    return await asyncio.to_thread(
        iv.compute_conditional_slice,
        trades, req.fix_metric, req.fix_bucket,
        req.fix_n_buckets, req.vary_metric, req.vary_n_buckets,
    )


@router.post("/{upload_id}/distribution")
async def distribution(upload_id: str, req: DistributionRequest, pool=Depends(get_pool)):
    trades, _ = await _load_trades(upload_id, pool)
    return iv.compute_distribution(trades, req.metric, req.bucket_index, req.n_buckets)


@router.post("/{upload_id}/time-stability")
async def time_stability(upload_id: str, req: TimeStabilityRequest,
                          pool=Depends(get_pool)):
    if not (2 <= req.n_windows <= 12):
        raise HTTPException(400, "n_windows must be 2–12")
    trades, _ = await _load_trades(upload_id, pool)
    return await asyncio.to_thread(
        iv.compute_time_stability, trades, req.metric, req.n_windows
    )


@router.post("/{upload_id}/feature-correlation")
async def feature_correlation(upload_id: str, req: FeatureCorrelationRequest,
                               pool=Depends(get_pool)):
    if len(req.metrics) > 25:
        raise HTTPException(400, "Maximum 25 metrics for redundancy matrix")
    trades, _ = await _load_trades(upload_id, pool)
    return await asyncio.to_thread(
        iv.compute_feature_correlation, trades, req.metrics
    )


@router.get("/{upload_id}/top-bottom")
async def top_bottom(upload_id: str, pool=Depends(get_pool)):
    trades, iv_columns = await _load_trades(upload_id, pool)
    return await asyncio.to_thread(
        iv.compute_top_bottom_regimes, trades, iv_columns
    )
