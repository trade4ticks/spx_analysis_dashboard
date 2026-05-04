"""
Computation functions for Backtest IV Analysis dashboard.
All math uses stdlib only — no numpy/pandas.
"""
import math
import statistics
from typing import Optional


# ── Math helpers ───────────────────────────────────────────────────────────────

def _safe_float(v) -> Optional[float]:
    if v is None:
        return None
    try:
        f = float(v)
        return None if (math.isnan(f) or math.isinf(f)) else f
    except (TypeError, ValueError):
        return None


def _pearson_r(xs: list, ys: list) -> float:
    n = len(xs)
    if n < 3:
        return 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx  = math.sqrt(sum((x - mx) ** 2 for x in xs))
    dy  = math.sqrt(sum((y - my) ** 2 for y in ys))
    denom = dx * dy
    return num / denom if denom > 1e-12 else 0.0


def _ols_residuals(xs: list, ys: list) -> list:
    """OLS residuals of regressing ys on xs."""
    n   = len(xs)
    mx  = sum(xs) / n
    my  = sum(ys) / n
    vx  = sum((x - mx) ** 2 for x in xs)
    if vx < 1e-12:
        return [y - my for y in ys]
    b1  = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / vx
    b0  = my - b1 * mx
    return [y - (b0 + b1 * x) for x, y in zip(xs, ys)]


def _delta_r2(x1: list, x2: list, y: list) -> float:
    """Semi-partial R²: increment in R² from adding x2 to model with x1 already in."""
    x2_resid = _ols_residuals(x1, x2)
    r = _pearson_r(y, x2_resid)
    return r * r


def _percentile(sorted_vals: list, p: float) -> float:
    """Linear-interpolation percentile. p in [0, 100]."""
    if not sorted_vals:
        return 0.0
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    idx = p / 100.0 * (len(sorted_vals) - 1)
    lo  = int(idx)
    hi  = min(lo + 1, len(sorted_vals) - 1)
    return sorted_vals[lo] + (idx - lo) * (sorted_vals[hi] - sorted_vals[lo])


# ── Bucketing ──────────────────────────────────────────────────────────────────

def _quantile_boundaries(values: list, n: int) -> list:
    """Return n+1 boundary values for equal-count bucketing."""
    vals = sorted(v for v in values if v is not None)
    if not vals:
        return []
    return [_percentile(vals, i * 100.0 / n) for i in range(n + 1)]


def _assign_bucket(value: float, boundaries: list) -> Optional[int]:
    """Return 0-based bucket index, or None if out of range."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    n = len(boundaries) - 1
    for i in range(n):
        if i == n - 1:
            return i if value <= boundaries[n] else None
        if value < boundaries[i + 1]:
            return i
    return None


def _bucket_label(idx: int, boundaries: list) -> str:
    lo = boundaries[idx]
    hi = boundaries[idx + 1]
    return f"{lo:.3g}–{hi:.3g}"


def _bucket_stats(pnl_list: list) -> dict:
    if not pnl_list:
        return {'n': 0, 'mean_pnl': None, 'win_rate': None, 'pnl_std': None}
    n        = len(pnl_list)
    mean_pnl = sum(pnl_list) / n
    win_rate = sum(1 for p in pnl_list if p > 0) / n
    pnl_std  = statistics.stdev(pnl_list) if n > 1 else 0.0
    return {
        'n':        n,
        'mean_pnl': round(mean_pnl, 2),
        'win_rate': round(win_rate, 4),
        'pnl_std':  round(pnl_std, 2),
    }


def _valid_pairs(trades: list, metric: str, target: str = 'pnl') -> tuple:
    """Return parallel (xs, ys) lists of valid float pairs."""
    xs, ys = [], []
    for t in trades:
        x = _safe_float(t.get(metric))
        y = _safe_float(t.get(target))
        if x is not None and y is not None:
            xs.append(x)
            ys.append(y)
    return xs, ys


# ── Section 1: 2D Heatmap ──────────────────────────────────────────────────────

def compute_heatmap(trades: list, metric_a: str, metric_b: str, n_buckets: int) -> dict:
    """
    n_buckets × n_buckets heatmap of trade outcomes.
    Returns cells[row_b][col_a] with per-cell stats.
    """
    vals_a = [_safe_float(t.get(metric_a)) for t in trades]
    vals_b = [_safe_float(t.get(metric_b)) for t in trades]
    pnls   = [_safe_float(t.get('pnl'))    for t in trades]

    bounds_a = _quantile_boundaries([v for v in vals_a if v is not None], n_buckets)
    bounds_b = _quantile_boundaries([v for v in vals_b if v is not None], n_buckets)
    if not bounds_a or not bounds_b:
        return {'error': 'Insufficient data for bucketing'}

    cell_pnls: list = [[[] for _ in range(n_buckets)] for _ in range(n_buckets)]
    valid_count = 0
    for va, vb, pnl in zip(vals_a, vals_b, pnls):
        if va is None or vb is None or pnl is None:
            continue
        ia = _assign_bucket(va, bounds_a)
        ib = _assign_bucket(vb, bounds_b)
        if ia is None or ib is None:
            continue
        cell_pnls[ib][ia].append(pnl)
        valid_count += 1

    cells = [
        [_bucket_stats(cell_pnls[ib][ia]) for ia in range(n_buckets)]
        for ib in range(n_buckets)
    ]

    return {
        'metric_a':    metric_a,
        'metric_b':    metric_b,
        'n_buckets':   n_buckets,
        'labels_a':    [_bucket_label(i, bounds_a) for i in range(n_buckets)],
        'labels_b':    [_bucket_label(i, bounds_b) for i in range(n_buckets)],
        'cells':       cells,
        'valid_count': valid_count,
    }


# ── Section 2: Pairwise ΔR² Grid ──────────────────────────────────────────────

def compute_delta_r2_grid(trades: list, metrics: list, target: str = 'pnl') -> dict:
    """
    Symmetric ΔR² for each pair of metrics against target.
    matrix[i][j] = 0.5 * (ΔR²(fj|fi) + ΔR²(fi|fj)).
    Diagonal = individual R²(fi, target).
    """
    valid_metrics = []
    series: dict = {}

    for m in metrics:
        xs, ys = _valid_pairs(trades, m, target)
        if len(xs) >= 10:
            valid_metrics.append(m)
            series[m] = (xs, ys)

    r2_single = {
        m: round(_pearson_r(*series[m]) ** 2, 4)
        for m in valid_metrics
    }

    matrix = []
    for i, mi in enumerate(valid_metrics):
        row = []
        for j, mj in enumerate(valid_metrics):
            if i == j:
                row.append(r2_single[mi])
                continue
            # Gather rows where both metrics are valid
            xs_i, xs_j, ys = [], [], []
            for t in trades:
                xi = _safe_float(t.get(mi))
                xj = _safe_float(t.get(mj))
                y  = _safe_float(t.get(target))
                if xi is not None and xj is not None and y is not None:
                    xs_i.append(xi)
                    xs_j.append(xj)
                    ys.append(y)
            if len(xs_i) < 10:
                row.append(None)
                continue
            dr2_ji = _delta_r2(xs_i, xs_j, ys)   # marginal contribution of j given i
            dr2_ij = _delta_r2(xs_j, xs_i, ys)   # marginal contribution of i given j
            row.append(round(0.5 * (dr2_ji + dr2_ij), 4))
        matrix.append(row)

    return {
        'metrics':   valid_metrics,
        'matrix':    matrix,
        'r2_single': r2_single,
        'target':    target,
        'n_trades':  len(trades),
    }


# ── Section 3: Decile View ─────────────────────────────────────────────────────

def compute_decile_stats(trades: list, metric: str, n_buckets: int = 10) -> dict:
    """Equal-count buckets of metric vs trade outcomes."""
    xs, ys = _valid_pairs(trades, metric, 'pnl')
    if len(xs) < n_buckets:
        return {'error': f'Too few valid trades ({len(xs)}) for {n_buckets} buckets'}

    bounds      = _quantile_boundaries(xs, n_buckets)
    bucket_pnls = [[] for _ in range(n_buckets)]
    for x, pnl in zip(xs, ys):
        idx = _assign_bucket(x, bounds)
        if idx is not None:
            bucket_pnls[idx].append(pnl)

    buckets = []
    for i in range(n_buckets):
        stats = _bucket_stats(bucket_pnls[i])
        stats['label']      = _bucket_label(i, bounds)
        stats['x_min']      = round(bounds[i], 6)
        stats['x_max']      = round(bounds[i + 1], 6)
        stats['bucket_idx'] = i
        buckets.append(stats)

    return {
        'metric':      metric,
        'n_buckets':   n_buckets,
        'buckets':     buckets,
        'total_valid': len(xs),
        'pearson_r':   round(_pearson_r(xs, ys), 4),
    }


# ── Section 4: Conditional Slice ──────────────────────────────────────────────

def compute_conditional_slice(trades: list, fix_metric: str, fix_bucket: int,
                               fix_n_buckets: int, vary_metric: str,
                               vary_n_buckets: int) -> dict:
    """
    Within trades where fix_metric falls in fix_bucket, bucket vary_metric
    and compute trade outcomes per bucket.
    """
    fix_vals  = [_safe_float(t.get(fix_metric))  for t in trades]
    vary_vals = [_safe_float(t.get(vary_metric)) for t in trades]
    pnls      = [_safe_float(t.get('pnl'))       for t in trades]

    fix_bounds = _quantile_boundaries([v for v in fix_vals if v is not None], fix_n_buckets)
    if not fix_bounds:
        return {'error': 'Insufficient data for fix metric bucketing'}

    # Filter to trades in the fix bucket
    slice_vary, slice_pnl = [], []
    for fv, vv, pnl in zip(fix_vals, vary_vals, pnls):
        if fv is None or vv is None or pnl is None:
            continue
        if _assign_bucket(fv, fix_bounds) == fix_bucket:
            slice_vary.append(vv)
            slice_pnl.append(pnl)

    if len(slice_vary) < vary_n_buckets:
        return {'error': f'Too few trades in slice ({len(slice_vary)})'}

    vary_bounds  = _quantile_boundaries(slice_vary, vary_n_buckets)
    bucket_pnls  = [[] for _ in range(vary_n_buckets)]
    for vv, pnl in zip(slice_vary, slice_pnl):
        idx = _assign_bucket(vv, vary_bounds)
        if idx is not None:
            bucket_pnls[idx].append(pnl)

    buckets = []
    for i in range(vary_n_buckets):
        stats = _bucket_stats(bucket_pnls[i])
        stats['label'] = _bucket_label(i, vary_bounds)
        buckets.append(stats)

    return {
        'fix_metric':    fix_metric,
        'fix_bucket':    fix_bucket,
        'fix_label':     _bucket_label(fix_bucket, fix_bounds),
        'vary_metric':   vary_metric,
        'vary_n_buckets': vary_n_buckets,
        'buckets':       buckets,
        'slice_n':       len(slice_vary),
        'slice_pearson_r': round(_pearson_r(slice_vary, slice_pnl), 4),
    }


# ── Section 5: Distribution View ──────────────────────────────────────────────

def compute_distribution(trades: list, metric: Optional[str] = None,
                          bucket_index: Optional[int] = None,
                          n_buckets: Optional[int] = None) -> dict:
    """
    P&L distribution (full 0–100 percentile range) for all trades or a metric bucket.
    """
    if metric and bucket_index is not None and n_buckets:
        xs   = [_safe_float(t.get(metric)) for t in trades]
        pnls_raw = [_safe_float(t.get('pnl')) for t in trades]
        valid = [(x, p) for x, p in zip(xs, pnls_raw) if x is not None and p is not None]
        if not valid:
            return {'error': 'No valid data'}
        bounds = _quantile_boundaries([v[0] for v in valid], n_buckets)
        pnls   = [p for x, p in valid if _assign_bucket(x, bounds) == bucket_index]
        label  = _bucket_label(bucket_index, bounds) if bounds else '?'
    else:
        pnls  = [_safe_float(t.get('pnl')) for t in trades]
        pnls  = [p for p in pnls if p is not None]
        label = 'All trades'

    if not pnls:
        return {'error': 'No valid P&L data'}

    sv = sorted(pnls)
    n  = len(sv)
    return {
        'label':    label,
        'n':        n,
        'min':      round(sv[0], 2),
        'p5':       round(_percentile(sv, 5),  2),
        'p10':      round(_percentile(sv, 10), 2),
        'p25':      round(_percentile(sv, 25), 2),
        'p50':      round(_percentile(sv, 50), 2),
        'p75':      round(_percentile(sv, 75), 2),
        'p90':      round(_percentile(sv, 90), 2),
        'p95':      round(_percentile(sv, 95), 2),
        'max':      round(sv[-1], 2),
        'mean':     round(sum(pnls) / n, 2),
        'std':      round(statistics.stdev(pnls), 2) if n > 1 else 0.0,
        'win_rate': round(sum(1 for p in pnls if p > 0) / n, 4),
    }


# ── Section 6: Time Stability ──────────────────────────────────────────────────

def compute_time_stability(trades: list, metric: str, n_windows: int = 6) -> dict:
    """
    Chronological split into n_windows periods; compute metric↔P&L Pearson r per period.
    """
    valid = []
    for t in trades:
        x   = _safe_float(t.get(metric))
        pnl = _safe_float(t.get('pnl'))
        d   = t.get('date_opened', '')
        if x is not None and pnl is not None and d:
            valid.append((d, x, pnl))

    if len(valid) < n_windows * 5:
        return {'error': f'Too few trades ({len(valid)}) for {n_windows} windows'}

    valid.sort(key=lambda v: v[0])
    chunk = len(valid) // n_windows

    periods = []
    for i in range(n_windows):
        start = i * chunk
        end   = start + chunk if i < n_windows - 1 else len(valid)
        seg   = valid[start:end]
        dates = [c[0] for c in seg]
        xs    = [c[1] for c in seg]
        ys    = [c[2] for c in seg]
        r     = _pearson_r(xs, ys)
        periods.append({
            'label':    f"P{i+1}",
            'date_from': dates[0],
            'date_to':   dates[-1],
            'n':         len(seg),
            'r':         round(r, 4),
            'r2':        round(r * r, 4),
            'mean_pnl':  round(sum(ys) / len(ys), 2),
            'win_rate':  round(sum(1 for y in ys if y > 0) / len(ys), 4),
        })

    overall_r = _pearson_r([v[1] for v in valid], [v[2] for v in valid])
    sign_consistency = sum(
        1 for p in periods if (p['r'] >= 0) == (overall_r >= 0)
    ) / len(periods)

    return {
        'metric':            metric,
        'n_windows':         n_windows,
        'periods':           periods,
        'overall_r':         round(overall_r, 4),
        'sign_consistency':  round(sign_consistency, 2),
        'total_valid':       len(valid),
    }


# ── Section 7: Feature Redundancy Matrix ──────────────────────────────────────

def compute_feature_correlation(trades: list, metrics: list) -> dict:
    """Pairwise Pearson r between IV metrics (feature × feature)."""
    valid_metrics = []
    for m in metrics:
        xs = [_safe_float(t.get(m)) for t in trades if _safe_float(t.get(m)) is not None]
        if len(xs) >= 10:
            valid_metrics.append(m)

    matrix = []
    for mi in valid_metrics:
        row = []
        for mj in valid_metrics:
            if mi == mj:
                row.append(1.0)
                continue
            xi_list, xj_list = [], []
            for t in trades:
                xi = _safe_float(t.get(mi))
                xj = _safe_float(t.get(mj))
                if xi is not None and xj is not None:
                    xi_list.append(xi)
                    xj_list.append(xj)
            row.append(round(_pearson_r(xi_list, xj_list), 4) if len(xi_list) >= 5 else None)
        matrix.append(row)

    return {'metrics': valid_metrics, 'matrix': matrix}


# ── Section 8: Top/Bottom Regime Summary ──────────────────────────────────────

def compute_top_bottom_regimes(trades: list, iv_columns: list,
                                n_top: int = 8, n_buckets: int = 5) -> dict:
    """
    For each IV metric, find the best and worst equal-count bucket by mean P&L.
    Return the n_top best and worst across all metrics.
    """
    all_buckets = []
    for metric in iv_columns:
        xs, ys = _valid_pairs(trades, metric, 'pnl')
        if len(xs) < n_buckets * 5:
            continue
        bounds      = _quantile_boundaries(xs, n_buckets)
        bucket_pnls = [[] for _ in range(n_buckets)]
        for x, pnl in zip(xs, ys):
            idx = _assign_bucket(x, bounds)
            if idx is not None:
                bucket_pnls[idx].append(pnl)

        for i in range(n_buckets):
            stats = _bucket_stats(bucket_pnls[i])
            if stats['n'] < 8 or stats['mean_pnl'] is None:
                continue
            all_buckets.append({
                'metric':       metric,
                'bucket_label': _bucket_label(i, bounds),
                'bucket_idx':   i,
                'n_buckets':    n_buckets,
                **stats,
            })

    all_buckets.sort(key=lambda b: b['mean_pnl'])
    worst = all_buckets[:n_top]
    best  = list(reversed(all_buckets[-n_top:]))
    return {'best': best, 'worst': worst, 'total_buckets_evaluated': len(all_buckets)}
