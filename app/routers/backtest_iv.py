"""Backtest IV Analysis API endpoints."""
import asyncio
import json
import logging
from typing import Literal, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
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


# ── Filtering ──────────────────────────────────────────────────────────────────

class FilterClause(BaseModel):
    """One metric filter clause. Operators:
      between  — uses min and/or max (either can be omitted for one-sided)
      gte/lte/lt/gt/eq — uses value
    Unknown columns or non-numeric trade values silently drop the trade.
    """
    col: str
    op:  Literal["between", "gte", "lte", "lt", "gt", "eq"]
    min:   Optional[float] = None
    max:   Optional[float] = None
    value: Optional[float] = None


def _filter_by_date(trades: list, date_from: Optional[str], date_to: Optional[str]) -> list:
    """Filter trades by date_opened (inclusive). Strings in YYYY-MM-DD compare lexically."""
    if not date_from and not date_to:
        return trades
    out = []
    for t in trades:
        d = t.get('date_opened')
        if not d:
            continue
        if date_from and d < date_from:
            continue
        if date_to and d > date_to:
            continue
        out.append(t)
    return out


def _build_matcher(f) -> Optional[tuple]:
    """Return (col, predicate) or None if the clause is malformed/unusable.
    Accepts either a FilterClause or a plain dict with the same keys."""
    if hasattr(f, 'model_dump'):
        f = f.model_dump()
    elif hasattr(f, 'dict'):
        f = f.dict()
    col = f.get('col')
    op  = f.get('op')
    if not col or not op:
        return None
    try:
        if op == 'between':
            lo_raw = f.get('min')
            hi_raw = f.get('max')
            lo = float(lo_raw) if lo_raw is not None and lo_raw != '' else None
            hi = float(hi_raw) if hi_raw is not None and hi_raw != '' else None
            if lo is None and hi is None:
                return None
            return (col, lambda v, lo=lo, hi=hi:
                    (lo is None or v >= lo) and (hi is None or v <= hi))
        if op in ('gte', 'lte', 'lt', 'gt', 'eq'):
            v_raw = f.get('value')
            if v_raw is None or v_raw == '':
                return None
            vf = float(v_raw)
            return {
                'gte': (col, lambda v, vf=vf: v >= vf),
                'lte': (col, lambda v, vf=vf: v <= vf),
                'gt':  (col, lambda v, vf=vf: v >  vf),
                'lt':  (col, lambda v, vf=vf: v <  vf),
                'eq':  (col, lambda v, vf=vf: v == vf),
            }[op]
    except (ValueError, TypeError):
        return None
    return None


def _apply_metric_filters(trades: list, filters: Optional[list]) -> list:
    """Apply metric-filter clauses. Trades missing a filter column or with a
    non-numeric value for that column are dropped."""
    if not filters:
        return trades
    matchers = [m for m in (_build_matcher(f) for f in filters) if m is not None]
    if not matchers:
        return trades
    out = []
    for t in trades:
        keep = True
        for col, fn in matchers:
            v = t.get(col)
            if v is None:
                keep = False
                break
            try:
                if not fn(float(v)):
                    keep = False
                    break
            except (ValueError, TypeError):
                keep = False
                break
        if keep:
            out.append(t)
    return out


def _apply_filters(
    trades: list,
    date_from: Optional[str],
    date_to: Optional[str],
    filters: Optional[list] = None,
) -> list:
    """Date filter + metric filters, in that order."""
    out = _filter_by_date(trades, date_from, date_to)
    if filters:
        out = _apply_metric_filters(out, filters)
    return out


def _parse_filter_param(s: str) -> Optional[dict]:
    """Parse one URL-encoded filter clause: 'col:op:arg1:arg2'.
      between → arg1=min (may be blank), arg2=max (may be blank)
      gte/lte/lt/gt/eq → arg1=value, arg2 ignored
    Returns a dict suitable for FilterClause / _build_matcher, or None."""
    if not s:
        return None
    parts = s.split(':', 3)
    if len(parts) < 3:
        return None
    col, op = parts[0], parts[1]
    arg1 = parts[2] if len(parts) > 2 else ''
    arg2 = parts[3] if len(parts) > 3 else ''
    if op == 'between':
        return {'col': col, 'op': 'between',
                'min': arg1 if arg1 != '' else None,
                'max': arg2 if arg2 != '' else None}
    if op in ('gte', 'lte', 'lt', 'gt', 'eq'):
        if arg1 == '':
            return None
        return {'col': col, 'op': op, 'value': arg1}
    return None


def _parse_filter_params(params: list[str]) -> list[dict]:
    """Parse repeated `f` query params into clause dicts. Bad clauses are skipped."""
    return [c for c in (_parse_filter_param(p) for p in (params or [])) if c]


# ── Request models ─────────────────────────────────────────────────────────────

class _DateFilterMixin(BaseModel):
    date_from: Optional[str] = None
    date_to:   Optional[str] = None
    # 'recompute' (default) — bin boundaries from filtered trades.
    # 'fixed'              — bin boundaries from the full upload.
    # Ignored by routes that don't bin (delta-r2, time-stability,
    # feature-correlation, correlation-overview).
    bin_mode:  Optional[str] = 'recompute'
    # Metric filters applied AFTER the date filter, BEFORE binning. With
    # bin_mode='fixed' the bin boundaries still come from the unfiltered
    # full upload, so D10 of vix_30d means the same threshold whether or
    # not filters are active.
    filters:   list[FilterClause] = []


class HeatmapRequest(_DateFilterMixin):
    metric_a: str
    metric_b: str
    n_buckets: int = 5


class DeltaR2Request(_DateFilterMixin):
    metrics: list[str]
    target: str = 'pnl'


class DecileRequest(_DateFilterMixin):
    metric: str
    n_buckets: int = 10


class ConditionalSliceRequest(_DateFilterMixin):
    fix_metric:    str
    fix_bucket:    int
    fix_n_buckets: int = 5
    vary_metric:   str
    vary_n_buckets: int = 5


class DistributionRequest(_DateFilterMixin):
    metric:       Optional[str] = None
    bucket_index: Optional[int] = None
    n_buckets:    Optional[int] = None


class TimeStabilityRequest(_DateFilterMixin):
    metric:    str
    n_windows: int = 6


class FeatureCorrelationRequest(_DateFilterMixin):
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
    trades_full, _ = await _load_trades(upload_id, pool)
    trades = _apply_filters(trades_full, req.date_from, req.date_to, req.filters)
    if not trades:
        raise HTTPException(400, "No trades after filters")
    boundary = trades_full if req.bin_mode == 'fixed' else None
    return await asyncio.to_thread(
        iv.compute_heatmap, trades, req.metric_a, req.metric_b, req.n_buckets,
        boundary,
    )


@router.post("/{upload_id}/delta-r2")
async def delta_r2(upload_id: str, req: DeltaR2Request, pool=Depends(get_pool)):
    if len(req.metrics) > 20:
        raise HTTPException(400, "Maximum 20 metrics for ΔR² grid")
    if len(req.metrics) < 2:
        raise HTTPException(400, "At least 2 metrics required")
    trades, _ = await _load_trades(upload_id, pool)
    trades = _apply_filters(trades, req.date_from, req.date_to, req.filters)
    if not trades:
        raise HTTPException(400, "No trades after filters")
    return await asyncio.to_thread(
        iv.compute_delta_r2_grid, trades, req.metrics, req.target
    )


@router.post("/{upload_id}/decile")
async def decile(upload_id: str, req: DecileRequest, pool=Depends(get_pool)):
    trades_full, _ = await _load_trades(upload_id, pool)
    trades = _apply_filters(trades_full, req.date_from, req.date_to, req.filters)
    if not trades:
        raise HTTPException(400, "No trades after filters")
    boundary = trades_full if req.bin_mode == 'fixed' else None
    return iv.compute_decile_stats(trades, req.metric, req.n_buckets, boundary)


@router.post("/{upload_id}/conditional-slice")
async def conditional_slice(upload_id: str, req: ConditionalSliceRequest,
                             pool=Depends(get_pool)):
    trades_full, _ = await _load_trades(upload_id, pool)
    trades = _apply_filters(trades_full, req.date_from, req.date_to, req.filters)
    if not trades:
        raise HTTPException(400, "No trades after filters")
    boundary = trades_full if req.bin_mode == 'fixed' else None
    return await asyncio.to_thread(
        iv.compute_conditional_slice,
        trades, req.fix_metric, req.fix_bucket,
        req.fix_n_buckets, req.vary_metric, req.vary_n_buckets,
        boundary,
    )


@router.post("/{upload_id}/distribution")
async def distribution(upload_id: str, req: DistributionRequest, pool=Depends(get_pool)):
    trades_full, _ = await _load_trades(upload_id, pool)
    trades = _apply_filters(trades_full, req.date_from, req.date_to, req.filters)
    if not trades:
        raise HTTPException(400, "No trades after filters")
    boundary = trades_full if req.bin_mode == 'fixed' else None
    return iv.compute_distribution(
        trades, req.metric, req.bucket_index, req.n_buckets, boundary,
    )


@router.post("/{upload_id}/time-stability")
async def time_stability(upload_id: str, req: TimeStabilityRequest,
                          pool=Depends(get_pool)):
    if not (2 <= req.n_windows <= 12):
        raise HTTPException(400, "n_windows must be 2–12")
    trades, _ = await _load_trades(upload_id, pool)
    trades = _apply_filters(trades, req.date_from, req.date_to, req.filters)
    if not trades:
        raise HTTPException(400, "No trades after filters")
    return await asyncio.to_thread(
        iv.compute_time_stability, trades, req.metric, req.n_windows
    )


@router.post("/{upload_id}/feature-correlation")
async def feature_correlation(upload_id: str, req: FeatureCorrelationRequest,
                               pool=Depends(get_pool)):
    if len(req.metrics) > 25:
        raise HTTPException(400, "Maximum 25 metrics for redundancy matrix")
    trades, _ = await _load_trades(upload_id, pool)
    trades = _apply_filters(trades, req.date_from, req.date_to, req.filters)
    if not trades:
        raise HTTPException(400, "No trades after filters")
    return await asyncio.to_thread(
        iv.compute_feature_correlation, trades, req.metrics
    )


@router.get("/{upload_id}/top-bottom")
async def top_bottom(
    upload_id: str,
    date_from: Optional[str] = None,
    date_to:   Optional[str] = None,
    bin_mode:  Optional[str] = 'recompute',
    f:         list[str] = Query(default_factory=list),
    pool=Depends(get_pool),
):
    trades_full, iv_columns = await _load_trades(upload_id, pool)
    trades = _apply_filters(trades_full, date_from, date_to, _parse_filter_params(f))
    if not trades:
        raise HTTPException(400, "No trades after filters")
    boundary = trades_full if bin_mode == 'fixed' else None
    return await asyncio.to_thread(
        iv.compute_top_bottom_regimes, trades, iv_columns, 8, 5, boundary,
    )


@router.get("/{upload_id}/correlation-overview")
async def correlation_overview(
    upload_id: str,
    target:    str = 'pnl',
    date_from: Optional[str] = None,
    date_to:   Optional[str] = None,
    f:         list[str] = Query(default_factory=list),
    pool=Depends(get_pool),
):
    if target not in ('pnl', 'is_win'):
        raise HTTPException(400, "target must be 'pnl' or 'is_win'")
    trades, iv_columns = await _load_trades(upload_id, pool)
    trades = _apply_filters(trades, date_from, date_to, _parse_filter_params(f))
    if not trades:
        raise HTTPException(400, "No trades after filters")
    return await asyncio.to_thread(
        iv.compute_correlation_overview, trades, iv_columns, target
    )
