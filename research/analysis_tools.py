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
