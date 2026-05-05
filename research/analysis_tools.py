"""Analysis tool functions for the agentic P&L–IV correlation pipeline.

All functions are:
  - Pure in-memory (operate on list[dict] rows, no DB/LLM calls)
  - Deterministic
  - JSON-serializable return values
"""
import math
import statistics
from typing import Optional

import numpy as np
from scipy import stats as sp_stats

from research import scanner, blocks


# ── Helpers ───────────────────────────────────────────────────────────────────

def _float_pairs(rows: list[dict], col_a: str, col_b: str) -> list[tuple[float, float]]:
    pairs = []
    for r in rows:
        a, b = r.get(col_a), r.get(col_b)
        if a is None or b is None:
            continue
        try:
            af, bf = float(a), float(b)
            if not (math.isnan(af) or math.isnan(bf)):
                pairs.append((af, bf))
        except (ValueError, TypeError):
            pass
    return pairs


def _safe_float(v) -> Optional[float]:
    if v is None:
        return None
    try:
        f = float(v)
        return None if math.isnan(f) else f
    except (ValueError, TypeError):
        return None


def _col_means(subset: list[dict], cols: list[str]) -> dict[str, Optional[float]]:
    result = {}
    for col in cols:
        vals = [_safe_float(r.get(col)) for r in subset]
        vals = [v for v in vals if v is not None]
        result[col] = round(statistics.mean(vals), 4) if vals else None
    return result


# ── Tool functions ────────────────────────────────────────────────────────────

def run_correlation_scan(rows: list[dict], x_cols: list[str], y_col: str) -> list[dict]:
    """
    Scan each x_col vs y_col. Returns list sorted by |r| descending.
    """
    results = []
    for x_col in x_cols:
        if x_col == y_col:
            continue
        scan = scanner.scan_relationship(rows, x_col, y_col)
        if 'error' in scan:
            continue
        results.append({
            'x_col':   x_col,
            'y_col':   y_col,
            'r':       round(scan.get('pearson_r', 0), 4),
            'p_val':   round(scan.get('pearson_p', 1), 4),
            'pattern': scan.get('pattern', ''),
            'score':   round(scan.get('composite_score', 0), 1),
            'n':       scan.get('n', 0),
        })
    results.sort(key=lambda d: abs(d['r']), reverse=True)
    return results


def run_regression(rows: list[dict], x_cols: list[str], y_col: str) -> dict:
    """OLS regression of y_col on x_cols. Returns r2, coefficients, n."""
    result = blocks.regression_from_rows(rows, x_cols, y_col)
    if 'error' in result:
        return result
    coeffs = dict(result.get('coefficients', {}))
    intercept = coeffs.pop('intercept', 0)
    coef_list = [{'col': k, 'coef': v} for k, v in coeffs.items()]
    return {
        'r2':           result.get('r_squared', 0),
        'intercept':    round(float(intercept), 6),
        'coefficients': coef_list,
        'n':            result.get('n', 0),
        'y_col':        y_col,
        'x_cols':       x_cols,
    }


def run_lag_scan(rows: list[dict], x_col: str, y_col: str,
                 lags: list[int] = None) -> list[dict]:
    """
    Pearson r between x_col[t-lag] and y_col[t] for each lag.
    Rows are sorted by (trade_date, quote_time) before applying lags.
    """
    if lags is None:
        lags = [1, 5, 10, 30]
    sorted_rows = sorted(
        rows,
        key=lambda r: (r.get('trade_date', ''), r.get('quote_time', '')),
    )
    results = []
    for lag in lags:
        pairs = []
        for i in range(lag, len(sorted_rows)):
            xv = _safe_float(sorted_rows[i - lag].get(x_col))
            yv = _safe_float(sorted_rows[i].get(y_col))
            if xv is not None and yv is not None:
                pairs.append((xv, yv))
        entry = {'x_col': x_col, 'y_col': y_col, 'lag': lag, 'n': len(pairs)}
        if len(pairs) >= 20:
            xa = np.array([p[0] for p in pairs])
            ya = np.array([p[1] for p in pairs])
            r, p = sp_stats.pearsonr(xa, ya)
            entry['r']     = round(float(r), 4)
            entry['p_val'] = round(float(p), 4)
        else:
            entry['r']     = None
            entry['p_val'] = None
        results.append(entry)
    return results


def run_regime_split(rows: list[dict], x_col: str, y_col: str,
                     split_col: str, method: str = 'median') -> dict:
    """
    Split rows at median/tercile of split_col.
    Returns {high, low, [mid]} regime stats and r_difference.
    """
    valid = [r for r in rows
             if _safe_float(r.get(split_col)) is not None
             and _safe_float(r.get(y_col)) is not None]
    if len(valid) < 40:
        return {'error': 'insufficient data', 'n': len(valid)}
    vals = sorted(_safe_float(r[split_col]) for r in valid)
    n = len(vals)

    if method == 'tercile':
        low_thresh  = vals[n // 3]
        high_thresh = vals[2 * n // 3]
        high_rows = [r for r in valid if _safe_float(r[split_col]) >= high_thresh]
        low_rows  = [r for r in valid if _safe_float(r[split_col]) < low_thresh]
        mid_rows  = [r for r in valid
                     if low_thresh <= _safe_float(r[split_col]) < high_thresh]
    else:
        med = vals[n // 2]
        high_rows = [r for r in valid if _safe_float(r[split_col]) >= med]
        low_rows  = [r for r in valid if _safe_float(r[split_col]) < med]
        mid_rows  = []

    def _regime_stats(regime_rows):
        if len(regime_rows) < 20:
            return {'n': len(regime_rows), 'insufficient': True}
        scan = scanner.scan_relationship(regime_rows, x_col, y_col)
        if 'error' in scan:
            return {'n': len(regime_rows), 'error': scan['error']}
        y_vals = [_safe_float(r[y_col]) for r in regime_rows]
        y_vals = [v for v in y_vals if v is not None]
        return {
            'r':       round(scan.get('pearson_r', 0), 4),
            'p_val':   round(scan.get('pearson_p', 1), 4),
            'mean_y':  round(statistics.mean(y_vals), 4) if y_vals else None,
            'score':   round(scan.get('composite_score', 0), 1),
            'n':       len(regime_rows),
            'pattern': scan.get('pattern', ''),
        }

    result = {
        'split_col': split_col,
        'method':    method,
        'x_col':     x_col,
        'y_col':     y_col,
        'high':      _regime_stats(high_rows),
        'low':       _regime_stats(low_rows),
    }
    if mid_rows:
        result['mid'] = _regime_stats(mid_rows)
    h_r = result['high'].get('r') or 0
    l_r = result['low'].get('r') or 0
    result['r_difference'] = round(h_r - l_r, 4)
    return result


def run_rolling_correlation(rows: list[dict], x_col: str, y_col: str,
                             window: int = 30) -> list[dict]:
    """
    Rolling Pearson r over a window of bars.
    Returns [{date, r, n}] for each window endpoint.
    """
    sorted_rows = sorted(
        rows,
        key=lambda r: (r.get('trade_date', ''), r.get('quote_time', '')),
    )
    results = []
    for i in range(window - 1, len(sorted_rows)):
        win = sorted_rows[i - window + 1: i + 1]
        pairs = _float_pairs(win, x_col, y_col)
        if len(pairs) < 10:
            continue
        xa = np.array([p[0] for p in pairs])
        ya = np.array([p[1] for p in pairs])
        r_val, _ = sp_stats.pearsonr(xa, ya)
        td = str(sorted_rows[i].get('trade_date', ''))
        qt = str(sorted_rows[i].get('quote_time', ''))
        results.append({
            'date': f"{td} {qt}".strip(),
            'r':    round(float(r_val), 4),
            'n':    len(pairs),
        })
    return results


def run_tail_analysis(rows: list[dict], y_col: str, x_cols: list[str],
                      pct: int = 10) -> dict:
    """
    Compare IV metric means between top/bottom pct% of rows by y_col.
    Returns {top_pct, bottom_pct, difference} where difference is sorted by |diff|.
    """
    valid = []
    for r in rows:
        yv = _safe_float(r.get(y_col))
        if yv is not None:
            valid.append((yv, r))
    if not valid:
        return {'error': f'no valid values for {y_col}'}
    valid.sort(key=lambda x: x[0])
    n = len(valid)
    cutoff = max(1, int(n * pct / 100))
    top_rows = [r for _, r in valid[-cutoff:]]
    bot_rows = [r for _, r in valid[:cutoff]]

    top_stats = _col_means(top_rows, x_cols)
    bot_stats  = _col_means(bot_rows, x_cols)
    diff = {}
    for col in x_cols:
        t, b = top_stats.get(col), bot_stats.get(col)
        if t is not None and b is not None:
            diff[col] = round(t - b, 4)
    diff_sorted = dict(sorted(diff.items(), key=lambda kv: abs(kv[1]), reverse=True))

    return {
        'y_col':      y_col,
        'pct':        pct,
        'n_top':      len(top_rows),
        'n_bottom':   len(bot_rows),
        'top_pct':    top_stats,
        'bottom_pct': bot_stats,
        'difference': diff_sorted,
    }


def run_decile_profile(rows: list[dict], x_col: str, y_col: str) -> dict:
    """Full decile bucket profile for x_col → y_col, same as scanner output."""
    scan = scanner.scan_relationship(rows, x_col, y_col)
    if 'error' in scan:
        return scan
    bs = [b for b in (scan.get('bucket_stats') or []) if b is not None]
    return {
        'x_col':        x_col,
        'y_col':        y_col,
        'pearson_r':    round(scan.get('pearson_r', 0), 4),
        'spearman_r':   round(scan.get('spearman_r', 0), 4),
        'pattern':      scan.get('pattern', ''),
        'score':        round(scan.get('composite_score', 0), 1),
        'n':            scan.get('n', 0),
        'bucket_stats': bs,
    }


def run_greek_attribution(rows: list[dict], pnl_col: str = 'pnl') -> dict:
    """
    OLS regression of pnl on available greek columns.
    Returns r2, coefficients, and unexplained P&L residual stats.
    """
    greek_cols = [c for c in ('delta', 'theta', 'vega', 'gamma', 'wt_vega')
                  if rows and c in (rows[0] if rows else {})]
    if not greek_cols:
        return {'error': 'no greek columns found in data'}
    result = blocks.regression_from_rows(rows, greek_cols, pnl_col)
    if 'error' in result:
        return result

    coeffs = dict(result.get('coefficients', {}))
    intercept = float(coeffs.pop('intercept', 0))
    coef_list = [{'col': k, 'coef': v} for k, v in coeffs.items()]

    # Compute residuals
    resids = []
    for r in rows:
        yv = _safe_float(r.get(pnl_col))
        if yv is None:
            continue
        pred = intercept + sum(
            float(c['coef']) * (_safe_float(r.get(c['col'])) or 0.0)
            for c in coef_list
        )
        resids.append(yv - pred)

    unexplained = {}
    if resids:
        unexplained = {
            'mean': round(statistics.mean(resids), 4),
            'std':  round(statistics.stdev(resids), 4) if len(resids) > 1 else 0.0,
            'min':  round(min(resids), 4),
            'max':  round(max(resids), 4),
        }

    return {
        'r2':              result.get('r_squared', 0),
        'intercept':       round(intercept, 6),
        'coefficients':    coef_list,
        'greek_cols_used': greek_cols,
        'n':               result.get('n', 0),
        'unexplained_pnl': unexplained,
    }


def run_win_rate_analysis(rows: list[dict], x_col: str,
                          n_buckets: int = 5) -> dict:
    """
    Bucket rows by x_col into n_buckets equal-count groups.
    Per bucket: win_rate, mean_pnl, std_pnl, n, x_min, x_max.
    Useful for finding IV threshold effects on binary win/loss outcomes.
    Returns {x_col, n_buckets, buckets, win_rate_spread}.
    """
    valid = []
    for r in rows:
        xv  = _safe_float(r.get(x_col))
        pnl = _safe_float(r.get('pnl'))
        iw  = r.get('is_win')
        if xv is None or pnl is None:
            continue
        win = bool(iw) if iw is not None else (pnl > 0)
        valid.append((xv, pnl, win))

    min_rows = n_buckets * 5
    if len(valid) < min_rows:
        return {'error': f'insufficient data: need {min_rows} rows, have {len(valid)}', 'n': len(valid)}

    valid.sort(key=lambda t: t[0])
    n = len(valid)
    bucket_size = n // n_buckets

    buckets = []
    for i in range(n_buckets):
        start = i * bucket_size
        end   = (i + 1) * bucket_size if i < n_buckets - 1 else n
        chunk = valid[start:end]
        xvals  = [t[0] for t in chunk]
        pnls   = [t[1] for t in chunk]
        wins   = [t[2] for t in chunk]
        buckets.append({
            'bucket':   i + 1,
            'n':        len(chunk),
            'x_min':    round(min(xvals), 4),
            'x_max':    round(max(xvals), 4),
            'x_mean':   round(statistics.mean(xvals), 4),
            'win_rate': round(sum(wins) / len(wins), 4),
            'mean_pnl': round(statistics.mean(pnls), 2),
            'std_pnl':  round(statistics.stdev(pnls), 2) if len(pnls) > 1 else 0.0,
        })

    top_wr = buckets[-1]['win_rate']
    bot_wr = buckets[0]['win_rate']
    return {
        'x_col':           x_col,
        'n_buckets':       n_buckets,
        'n':               n,
        'buckets':         buckets,
        'win_rate_spread': round(top_wr - bot_wr, 4),
    }


def run_feature_redundancy_check(rows: list[dict], features: list[str],
                                  r_threshold: float = 0.7) -> dict:
    """
    Compute pairwise Pearson r among features and group redundant ones (|r| >= threshold).
    Returns {features_checked, r_threshold, n_groups, non_redundant, groups, pairwise_top}.
    Use this after discovery to avoid building a composite from correlated features.
    """
    valid_features = []
    for f in features:
        vals = [_safe_float(r.get(f)) for r in rows]
        if sum(v is not None for v in vals) >= 10:
            valid_features.append(f)

    if len(valid_features) < 2:
        return {'error': 'need ≥2 features with valid data', 'features_checked': len(features)}

    # Pairwise correlations
    pairs = []
    for i, f1 in enumerate(valid_features):
        for f2 in valid_features[i + 1:]:
            fp = _float_pairs(rows, f1, f2)
            if len(fp) < 10:
                continue
            xs, ys = [p[0] for p in fp], [p[1] for p in fp]
            try:
                r_val = float(np.corrcoef(xs, ys)[0, 1])
                if not math.isnan(r_val):
                    pairs.append({'f1': f1, 'f2': f2, 'r': round(r_val, 4)})
            except Exception:
                pass
    pairs.sort(key=lambda p: abs(p['r']), reverse=True)

    # Build r_map for quick lookup
    r_map: dict[tuple, float] = {}
    for p in pairs:
        r_map[(p['f1'], p['f2'])] = abs(p['r'])
        r_map[(p['f2'], p['f1'])] = abs(p['r'])

    # Greedy redundancy clustering
    assigned: set[str] = set()
    groups = []
    for f in valid_features:
        if f in assigned:
            continue
        group = [f]
        assigned.add(f)
        for f2 in valid_features:
            if f2 in assigned:
                continue
            max_r = max((r_map.get((gm, f2), 0.0) for gm in group), default=0.0)
            if max_r >= r_threshold:
                group.append(f2)
                assigned.add(f2)
        groups.append({'representative': f, 'members': group, 'size': len(group)})

    groups.sort(key=lambda g: -g['size'])

    return {
        'features_checked': len(valid_features),
        'r_threshold': r_threshold,
        'n_groups': len(groups),
        'non_redundant': [g['representative'] for g in groups],
        'groups': groups,
        'pairwise_top': pairs[:20],
    }


def run_two_factor_regime(rows: list[dict], factor1: str, factor2: str,
                           y_col: str, n_bins: int = 3) -> dict:
    """
    2-factor outcome grid: bin each factor into n_bins equal-count groups, compute
    mean_y and win_rate per cell. Returns main-effect marginals and interaction strength
    (max deviation of any cell from its additive prediction). Use this as the discovery
    layer for regime construction.
    """
    valid = []
    for r in rows:
        v1 = _safe_float(r.get(factor1))
        v2 = _safe_float(r.get(factor2))
        yv = _safe_float(r.get(y_col))
        if v1 is None or v2 is None or yv is None:
            continue
        iw = r.get('is_win')
        win = bool(iw) if iw is not None else (yv > 0)
        valid.append((v1, v2, yv, win))

    if len(valid) < n_bins * n_bins * 5:
        return {'error': f'insufficient data: need ~{n_bins*n_bins*5}, have {len(valid)}', 'n': len(valid)}

    def _bin_edges(vals):
        sv = sorted(vals)
        n = len(sv)
        return [sv[int(n * i / n_bins)] for i in range(1, n_bins)]

    def _assign_bin(v, edges):
        for i, e in enumerate(edges):
            if v <= e:
                return i
        return len(edges)

    f1_edges = _bin_edges([t[0] for t in valid])
    f2_edges = _bin_edges([t[1] for t in valid])
    bin_labels = ['low', 'mid', 'high'] if n_bins == 3 else [str(i + 1) for i in range(n_bins)]

    # Fill cells
    cells: dict[tuple, dict] = {}
    for v1, v2, yv, win in valid:
        key = (_assign_bin(v1, f1_edges), _assign_bin(v2, f2_edges))
        if key not in cells:
            cells[key] = {'y': [], 'wins': []}
        cells[key]['y'].append(yv)
        cells[key]['wins'].append(win)

    def _cell_stats(b1, b2):
        c = cells.get((b1, b2), {'y': [], 'wins': []})
        n = len(c['y'])
        return {
            'f1_bin': bin_labels[b1], 'f2_bin': bin_labels[b2],
            'n': n,
            'mean_y':   round(statistics.mean(c['y']), 2)          if n > 0 else None,
            'win_rate': round(sum(c['wins']) / n, 4)                if n > 0 else None,
            'std_y':    round(statistics.stdev(c['y']), 2)          if n > 1 else None,
        }

    grid = [[_cell_stats(b1, b2) for b2 in range(n_bins)] for b1 in range(n_bins)]
    cell_means = {(b1, b2): grid[b1][b2]['mean_y'] for b1 in range(n_bins) for b2 in range(n_bins)}

    # Marginals
    def _marginal(axis, b_idx):
        ys, wins = [], []
        for (b1, b2), c in cells.items():
            if (b1 if axis == 0 else b2) == b_idx:
                ys.extend(c['y']); wins.extend(c['wins'])
        n = len(ys)
        return {
            'bin': bin_labels[b_idx], 'n': n,
            'mean_y':   round(statistics.mean(ys), 2)     if n > 0 else None,
            'win_rate': round(sum(wins) / n, 4)           if n > 0 else None,
        }

    marginals_f1 = [_marginal(0, b) for b in range(n_bins)]
    marginals_f2 = [_marginal(1, b) for b in range(n_bins)]

    # Interaction strength: max deviation from additive prediction
    all_y = [t[2] for t in valid]
    global_mean = statistics.mean(all_y)
    max_interaction = 0.0
    best_cell = worst_cell = None
    best_mean = worst_mean = None
    for b1 in range(n_bins):
        for b2 in range(n_bins):
            actual = cell_means.get((b1, b2))
            if actual is None:
                continue
            m1 = marginals_f1[b1].get('mean_y')
            m2 = marginals_f2[b2].get('mean_y')
            if m1 is not None and m2 is not None:
                deviation = abs(actual - (m1 + m2 - global_mean))
                if deviation > max_interaction:
                    max_interaction = deviation
            if best_mean is None or actual > best_mean:
                best_mean = actual
                best_cell = {'f1_bin': bin_labels[b1], 'f2_bin': bin_labels[b2], 'mean_y': actual}
            if worst_mean is None or actual < worst_mean:
                worst_mean = actual
                worst_cell = {'f1_bin': bin_labels[b1], 'f2_bin': bin_labels[b2], 'mean_y': actual}

    return {
        'factor1': factor1, 'factor2': factor2, 'y_col': y_col,
        'n_bins': n_bins, 'n': len(valid),
        'global_mean_y': round(global_mean, 2),
        'grid': grid,
        'marginals_f1': marginals_f1,
        'marginals_f2': marginals_f2,
        'interaction_strength': round(max_interaction, 2),
        'best_cell': best_cell,
        'worst_cell': worst_cell,
    }


def run_composite_regime_score(
    rows: list[dict],
    components: list[dict],
    interactions: list[dict] | None = None,
    y_col: str = 'pnl',
    train_frac: float = 0.6,
) -> dict:
    """
    Build and validate a composite regime score from 2-5 component features.
    components: [{feature, direction}] where direction 1=higher-is-favorable, -1=lower-is-favorable.
    interactions: [{feature1, feature2, type}] where type is 'amplify' or 'conditional'.
      amplify:     bonus when both components point same way; penalty when opposite.
      conditional: feature1 gets +1 weight when feature2 is in top tercile, -1 when bottom tercile.
    Splits chronologically (train_frac train / remainder validate).
    Returns quintile profiles, train/validate r, and comparison vs best single component.
    """
    if not components:
        return {'error': 'no components specified'}
    interactions = interactions or []

    # Collect valid rows
    valid: list[dict] = []
    for r in rows:
        yv = _safe_float(r.get(y_col))
        if yv is None:
            continue
        comp_vals = {}
        ok = True
        for c in components:
            v = _safe_float(r.get(c['feature']))
            if v is None:
                ok = False
                break
            comp_vals[c['feature']] = v
        if not ok:
            continue
        iw = r.get('is_win')
        valid.append({'date': r.get('date_opened', ''), 'y': yv,
                      'win': bool(iw) if iw is not None else (yv > 0), **comp_vals})

    if len(valid) < 20:
        return {'error': f'insufficient data: {len(valid)} valid rows', 'n': len(valid)}

    valid.sort(key=lambda r: r['date'])
    split_idx = max(10, int(len(valid) * train_frac))
    train_rows = valid[:split_idx]
    val_rows   = valid[split_idx:]
    if len(val_rows) < 10:
        return {'error': f'validation set too small ({len(val_rows)} rows); reduce train_frac'}

    def _pct_rank(v: float, sorted_vals: list[float]) -> float:
        pos = int(np.searchsorted(sorted_vals, v, side='right'))
        return pos / len(sorted_vals)

    def _build_sorted(row_subset):
        return {c['feature']: sorted(r[c['feature']] for r in row_subset) for c in components}

    def _score_rows(row_subset, sorted_ref):
        scored = []
        for r in row_subset:
            comp_pcts: dict[str, float] = {}
            total = 0.0
            for c in components:
                f, d = c['feature'], int(c.get('direction', 1))
                pct = _pct_rank(r[f], sorted_ref[f])
                adj = pct if d >= 0 else (1.0 - pct)
                comp_pcts[f] = adj
                total += adj
            for ix in interactions:
                f1, f2 = ix.get('feature1'), ix.get('feature2')
                s1, s2 = comp_pcts.get(f1), comp_pcts.get(f2)
                if s1 is None or s2 is None:
                    continue
                if ix.get('type') == 'amplify':
                    total += 2.0 * (s1 - 0.5) * (s2 - 0.5)
                elif ix.get('type') == 'conditional':
                    total += s1 if s2 > 2/3 else (-s1 if s2 < 1/3 else 0.0)
            scored.append((total, r['y'], r['win']))
        return scored

    sorted_train = _build_sorted(train_rows)
    sorted_full  = _build_sorted(valid)

    train_scored = _score_rows(train_rows, sorted_train)
    val_scored   = _score_rows(val_rows,   sorted_train)   # use train distribution
    full_scored  = _score_rows(valid,      sorted_full)

    def _quintile_profile(scored):
        scored = sorted(scored, key=lambda x: x[0])
        n, q = len(scored), max(1, len(scored) // 5)
        out = []
        for qi in range(5):
            chunk = scored[qi * q : (qi + 1) * q if qi < 4 else n]
            ys = [t[1] for t in chunk]; wins = [t[2] for t in chunk]
            out.append({
                'quintile': qi + 1, 'n': len(chunk),
                'score_min': round(chunk[0][0], 3), 'score_max': round(chunk[-1][0], 3),
                'mean_y': round(statistics.mean(ys), 2),
                'win_rate': round(sum(wins) / len(wins), 4) if wins else 0.0,
                'std_y': round(statistics.stdev(ys), 2) if len(ys) > 1 else 0.0,
            })
        return out

    def _pearson_r(scored):
        xs = [t[0] for t in scored]; ys = [t[1] for t in scored]
        if len(xs) < 3:
            return 0.0
        try:
            r = float(np.corrcoef(xs, ys)[0, 1])
            return round(r, 4) if not math.isnan(r) else 0.0
        except Exception:
            return 0.0

    train_r = _pearson_r(train_scored)
    val_r   = _pearson_r(val_scored)
    full_r  = _pearson_r(full_scored)

    # Best single component on validation set (raw feature values vs y)
    best_single_name, best_single_val_r = None, 0.0
    for c in components:
        f = c['feature']
        pairs = [(r[f], r['y']) for r in val_rows]
        if len(pairs) < 5:
            continue
        xs, ys = [p[0] for p in pairs], [p[1] for p in pairs]
        try:
            r_val = float(np.corrcoef(xs, ys)[0, 1])
            if not math.isnan(r_val) and abs(r_val) > abs(best_single_val_r):
                best_single_val_r = round(r_val, 4)
                best_single_name = f
        except Exception:
            pass

    return {
        'y_col': y_col,
        'n': len(valid), 'n_train': len(train_rows), 'n_validate': len(val_rows),
        'components': components,
        'interactions': interactions,
        'full_quintiles':     _quintile_profile(full_scored),
        'train_quintiles':    _quintile_profile(train_scored),
        'validate_quintiles': _quintile_profile(val_scored),
        'full_r':   full_r,
        'train_r':  train_r,
        'validate_r': val_r,
        'best_single_component':  best_single_name,
        'best_single_validate_r': best_single_val_r,
        'composite_vs_single': round(abs(val_r) - abs(best_single_val_r), 4),
        'stable': abs(val_r) >= abs(train_r) * 0.55,
    }


def run_time_split_validation(rows: list[dict], y_col: str,
                              n_splits: int = 2) -> dict:
    """
    Split rows chronologically into n_splits periods by date_opened.
    Per period: top-5 IV correlations with y_col.
    consistency_score: fraction of period-1 top signals in all other periods' top-10.
    Returns {y_col, periods, consistency_score, stable_signals}.
    """
    min_per = 15
    sorted_rows = sorted(rows, key=lambda r: r.get('date_opened', ''))

    if len(sorted_rows) < min_per * n_splits:
        return {
            'error': f'insufficient data for {n_splits} splits (need ≥{min_per * n_splits} rows)',
            'n': len(rows),
        }

    chunk_size = len(sorted_rows) // n_splits
    periods_rows = []
    for i in range(n_splits):
        start = i * chunk_size
        end   = (i + 1) * chunk_size if i < n_splits - 1 else len(sorted_rows)
        periods_rows.append(sorted_rows[start:end])

    # IV columns: all keys present in first row except known trade/timestamp fields
    _non_iv = {
        'date_opened', 'date_closed', 'pnl', 'pnl_pct', 'strategy',
        'max_profit', 'max_loss', 'margin_req', 'legs', 'exit_reason',
        'premium', 'contracts', 'days_in_trade', 'is_win',
        'join_timestamp', 'trade_date', 'quote_time', 'id',
        'day_of_week', 'year', 'spx_open_price', 'spx_close_price',
        'time_opened', 'time_closed',
    }
    if sorted_rows:
        iv_cols = [k for k in sorted_rows[0].keys() if k not in _non_iv]
    else:
        iv_cols = []

    periods = []
    all_top5 = []
    for i, pr in enumerate(periods_rows):
        if len(pr) < min_per or not iv_cols:
            periods.append({
                'label': f'Period {i+1}',
                'n':     len(pr),
                'top_correlations': [],
            })
            all_top5.append(set())
            continue
        corr = run_correlation_scan(pr, iv_cols, y_col)
        top5  = [c['x_col'] for c in corr[:5]]
        top10 = {c['x_col'] for c in corr[:10]}
        periods.append({
            'label':            f'Period {i+1} ({(pr[0].get("date_opened","?"))[:7]} – {(pr[-1].get("date_opened","?"))[:7]})',
            'n':                len(pr),
            'top_correlations': [{'x_col': c['x_col'], 'r': c['r']} for c in corr[:5]],
        })
        all_top5.append((set(top5), top10))

    # Stable signals: in period-1 top-5 AND in every other period's top-10
    stable = []
    if all_top5 and isinstance(all_top5[0], tuple):
        first_top5 = all_top5[0][0]
        for sig in first_top5:
            if all(isinstance(p, tuple) and sig in p[1] for p in all_top5[1:]):
                stable.append(sig)

    score = round(len(stable) / max(len(all_top5[0][0]) if all_top5 and isinstance(all_top5[0], tuple) else 1, 1), 3)

    return {
        'y_col':             y_col,
        'n_splits':          n_splits,
        'periods':           periods,
        'stable_signals':    stable,
        'consistency_score': score,
    }


# ── Intratrade path tools ──────────────────────────────────────────────────────

def run_days_in_trade_profile(
    rows: list[dict],
    y_col: str,
    n_bins: int = 10,
    group_by_phase: bool = False,
) -> dict:
    """
    Profile y_col by time-in-trade.

    If group_by_phase=True: group by 'trade_phase' field (early/middle/late).
    Otherwise: bucket by 'dit' into n_bins equal-width bins.

    Returns per-bin stats and a trend label.
    """
    if group_by_phase:
        phases = ['early', 'middle', 'late']
        bins_out = []
        for phase in phases:
            subset = [r for r in rows if r.get('trade_phase') == phase]
            y_vals = [_safe_float(r.get(y_col)) for r in subset]
            y_vals = [v for v in y_vals if v is not None]
            bins_out.append({
                'phase':   phase,
                'n':       len(subset),
                'mean':    round(statistics.mean(y_vals), 4) if y_vals else None,
                'std':     round(statistics.stdev(y_vals), 4) if len(y_vals) > 1 else None,
                'dit_min': None,
                'dit_max': None,
            })
        means = [b['mean'] for b in bins_out if b['mean'] is not None]
        trend = _classify_trend(means)
        return {'y_col': y_col, 'group_by_phase': True, 'bins': bins_out, 'trend': trend}

    # DIT-bucket mode
    dit_vals = [_safe_float(r.get('dit')) for r in rows]
    valid = [(d, _safe_float(r.get(y_col))) for r, d in zip(rows, dit_vals)
             if d is not None and _safe_float(r.get(y_col)) is not None]

    if len(valid) < n_bins * 3:
        return {'error': f'Insufficient rows ({len(valid)}) for {n_bins} bins (need ≥{n_bins * 3})'}

    min_dit = min(d for d, _ in valid)
    max_dit = max(d for d, _ in valid)
    width   = (max_dit - min_dit) / n_bins if max_dit > min_dit else 1.0

    bins_out = []
    for i in range(n_bins):
        lo = min_dit + i * width
        hi = min_dit + (i + 1) * width
        subset_y = [y for d, y in valid if (lo <= d < hi) or (i == n_bins - 1 and d == max_dit)]
        bins_out.append({
            'bin':     i + 1,
            'dit_min': round(lo, 1),
            'dit_max': round(hi, 1),
            'n':       len(subset_y),
            'mean':    round(statistics.mean(subset_y), 4) if subset_y else None,
            'std':     round(statistics.stdev(subset_y), 4) if len(subset_y) > 1 else None,
        })

    means = [b['mean'] for b in bins_out if b['mean'] is not None]
    trend = _classify_trend(means)
    return {'y_col': y_col, 'n_bins': n_bins, 'bins': bins_out, 'trend': trend}


def _classify_trend(means: list) -> str:
    if len(means) < 2:
        return 'flat'
    first_half = means[:len(means) // 2]
    second_half = means[len(means) // 2:]
    avg_first = statistics.mean(v for v in first_half if v is not None) if any(v is not None for v in first_half) else 0
    avg_second = statistics.mean(v for v in second_half if v is not None) if any(v is not None for v in second_half) else 0
    diff = avg_second - avg_first
    scale = max(abs(avg_first), abs(avg_second), 1e-9)
    if abs(diff) / scale < 0.05:
        return 'flat'
    return 'improving' if diff > 0 else 'deteriorating'


def run_path_event_scan(
    rows: list[dict],
    event_col: str,
    threshold,
    direction: str,
    context_cols: list[str],
    window_days: int = 3,
    threshold_type: str = 'fixed',
) -> dict:
    """
    Find daily rows where event_col crosses a threshold.

    threshold_type='fixed': use threshold as a literal value.
    threshold_type='percentile': compute Nth percentile of event_col.
      direction='below_pct' → bottom N%,  direction='above_pct' → top N%.

    Returns mean context_col values window_days before and after each crossing.
    """
    vals = [_safe_float(r.get(event_col)) for r in rows]
    vals_clean = [v for v in vals if v is not None]
    if not vals_clean:
        return {'error': f'No valid values for {event_col}'}

    # Resolve threshold
    computed_threshold = float(threshold)
    if threshold_type == 'percentile':
        pct = float(threshold)
        sorted_vals = sorted(vals_clean)
        idx = max(0, int(len(sorted_vals) * pct / 100) - 1)
        computed_threshold = sorted_vals[idx]
        # Normalize direction
        if direction == 'below_pct':
            direction = 'below'
        elif direction == 'above_pct':
            direction = 'above'

    # Build lookup: (position_id, date) → row index
    pos_date_idx: dict = {}
    for i, r in enumerate(rows):
        pos_date_idx[(r.get('position_id'), r.get('date'))] = i

    from datetime import date as _dt, timedelta as _td

    def _add_days(date_str: str, n: int) -> str:
        try:
            return str((_dt.fromisoformat(date_str) + _td(days=n)))
        except (ValueError, TypeError):
            return ''

    # Find crossing events: rows where condition first becomes True
    event_rows = []
    prev_state = None
    for r, v in zip(rows, vals):
        if v is None:
            prev_state = None
            continue
        state = (v <= computed_threshold) if direction == 'below' else (v >= computed_threshold)
        if state and prev_state is not True:
            event_rows.append(r)
        prev_state = state

    if not event_rows:
        return {
            'event_col': event_col, 'threshold': threshold,
            'threshold_type': threshold_type,
            'computed_threshold_value': computed_threshold,
            'direction': direction, 'event_count': 0,
            'mean_context_before': {}, 'mean_context_after': {},
        }

    before_acc: dict[str, list] = {c: [] for c in context_cols}
    after_acc:  dict[str, list] = {c: [] for c in context_cols}

    for er in event_rows:
        pos_id   = er.get('position_id')
        ev_date  = er.get('date', '')
        for offset in range(1, window_days + 1):
            before_date = _add_days(ev_date, -offset)
            after_date  = _add_days(ev_date, offset)
            for col in context_cols:
                bi = pos_date_idx.get((pos_id, before_date))
                ai = pos_date_idx.get((pos_id, after_date))
                if bi is not None:
                    v = _safe_float(rows[bi].get(col))
                    if v is not None:
                        before_acc[col].append(v)
                if ai is not None:
                    v = _safe_float(rows[ai].get(col))
                    if v is not None:
                        after_acc[col].append(v)

    mean_before = {c: (round(statistics.mean(before_acc[c]), 4) if before_acc[c] else None)
                   for c in context_cols}
    mean_after  = {c: (round(statistics.mean(after_acc[c]), 4) if after_acc[c] else None)
                   for c in context_cols}

    return {
        'event_col':               event_col,
        'threshold':               threshold,
        'threshold_type':          threshold_type,
        'computed_threshold_value': computed_threshold,
        'direction':               direction,
        'event_count':             len(event_rows),
        'mean_context_before':     mean_before,
        'mean_context_after':      mean_after,
    }
