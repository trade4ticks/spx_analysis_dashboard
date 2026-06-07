"""
Broad relationship scanner — multi-lens analysis + robustness ranking.

All functions operate on in-memory list[dict] rows (no DB, no LLM).
For each (feature, outcome) pair, scans for linear, monotonic, U-shaped,
threshold, isolated-region, and tail relationships, then scores robustness.
"""
from collections import defaultdict
from typing import Optional

import numpy as np
from scipy import stats as sp_stats


# ── Helpers ──────────────────────────────────────────────────────────────────

def _jsonify(obj):
    """Recursively convert numpy types to native Python for JSON serialization."""
    if isinstance(obj, dict):
        return {k: _jsonify(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_jsonify(v) for v in obj]
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

def _valid_pairs(rows: list[dict], x_col: str, y_col: str):
    """Extract (x, y, trade_date) tuples where both are non-None and non-NaN."""
    import math
    out = []
    for r in rows:
        xv, yv = r.get(x_col), r.get(y_col)
        if xv is None or yv is None:
            continue
        try:
            xf, yf = float(xv), float(yv)
        except (ValueError, TypeError):
            continue
        if math.isnan(xf) or math.isnan(yf):
            continue
        out.append((xf, yf, r.get("trade_date")))
    return out


def _bucket(pairs, n_buckets: int = 10):
    """Sort by x, split into n equal-ish buckets. Returns list of lists of y values."""
    if not pairs:
        return []
    sorted_pairs = sorted(pairs, key=lambda p: p[0])
    total = len(sorted_pairs)
    buckets = [[] for _ in range(n_buckets)]
    for i, (_, y, _) in enumerate(sorted_pairs):
        b = min(int(i / total * n_buckets), n_buckets - 1)
        buckets[b].append(y)
    return buckets


def _bucket_with_x(pairs, n_buckets: int = 10):
    """Same as _bucket but tracks x-values per bucket so callers can report
    feature-space edges. Returns list of (xs, ys) per bucket."""
    if not pairs:
        return []
    sorted_pairs = sorted(pairs, key=lambda p: p[0])
    total = len(sorted_pairs)
    buckets: list = [([], []) for _ in range(n_buckets)]
    for i, (x, y, _) in enumerate(sorted_pairs):
        b = min(int(i / total * n_buckets), n_buckets - 1)
        buckets[b][0].append(x)
        buckets[b][1].append(y)
    return buckets


def _sharpe(ys: list[float]) -> float:
    """Mean / std, annualization-agnostic. Returns 0 if std is 0."""
    if not ys:
        return 0.0
    a = np.array(ys)
    s = float(a.std())
    return float(a.mean() / s) if s > 0 else 0.0


def _year_from_date(d) -> Optional[int]:
    if d is None:
        return None
    if hasattr(d, "year"):
        return d.year
    try:
        return int(str(d)[:4])
    except Exception:
        return None


def _mutual_information(x: np.ndarray, y: np.ndarray, n_bins: int = 5) -> float:
    """Mutual information via equal-count (quantile) bins. Returns MI in nats."""
    n = len(x)
    if n < n_bins * 4:
        return 0.0
    q_x = np.quantile(x, np.linspace(0, 1, n_bins + 1))
    q_y = np.quantile(y, np.linspace(0, 1, n_bins + 1))
    bx = np.clip(np.searchsorted(q_x[1:-1], x), 0, n_bins - 1)
    by = np.clip(np.searchsorted(q_y[1:-1], y), 0, n_bins - 1)
    joint = np.zeros((n_bins, n_bins))
    for i in range(n):
        joint[bx[i], by[i]] += 1
    joint /= n
    px = joint.sum(axis=1, keepdims=True)
    py = joint.sum(axis=0, keepdims=True)
    mask = joint > 0
    mi = float(np.sum(joint[mask] * np.log(joint[mask] / (px * py)[mask])))
    return max(mi, 0.0)


# ── Multi-lens relationship profile ──────────────────────────────────────────

def scan_relationship(rows: list[dict], x_col: str, y_col: str,
                      ticker: Optional[str] = None,
                      n_buckets: int = 10) -> dict:
    """
    Comprehensive scan of the relationship between x_col and y_col.
    Returns a dict with correlation, bucket profile, pattern detection,
    and robustness diagnostics.
    """
    pairs = _valid_pairs(rows, x_col, y_col)
    n = len(pairs)

    if n < max(20, n_buckets * 3):
        return {"error": "insufficient data", "n": n,
                "x_col": x_col, "y_col": y_col, "ticker": ticker}

    xa = np.array([p[0] for p in pairs])
    ya = np.array([p[1] for p in pairs])

    # ── 1. Correlation ───────────────────────────────────────────────
    pr, pp = sp_stats.pearsonr(xa, ya)
    sr, sp_val = sp_stats.spearmanr(xa, ya)
    mi_val = _mutual_information(xa, ya)

    # ── 2. Bucket profile ────────────────────────────────────────────
    buckets_xy = _bucket_with_x(pairs, n_buckets)
    buckets    = [ys for (_, ys) in buckets_xy]  # back-compat for sections below
    bucket_stats = []
    for i, (xs, ys) in enumerate(buckets_xy):
        if not ys:
            bucket_stats.append(None)
            continue
        a = np.array(ys)
        xa_b = np.array(xs)
        bucket_stats.append({
            "bucket":    i + 1,
            "n":         len(ys),
            "avg_ret":   round(float(a.mean()), 6),
            "med_ret":   round(float(np.median(a)), 6),
            "win_rate":  round(float((a > 0).mean()), 4),
            "std_dev":   round(float(a.std()), 6),
            "sharpe":    round(_sharpe(ys), 4),
            "payoff":    round(float(a[a > 0].mean() / abs(a[a < 0].mean()))
                               if (a > 0).any() and (a < 0).any() else 0.0, 4),
            # Feature-space edges so callers can quote real thresholds:
            "x_min":     round(float(xa_b.min()),  6),
            "x_max":     round(float(xa_b.max()),  6),
            "x_mean":    round(float(xa_b.mean()), 6),
        })

    valid_buckets = [b for b in bucket_stats if b is not None]

    # ── 3. Monotonicity ──────────────────────────────────────────────
    avgs = [b["avg_ret"] for b in valid_buckets]
    if len(avgs) >= 2:
        transitions = sum(1 for i in range(len(avgs) - 1) if avgs[i + 1] > avgs[i])
        total_trans = len(avgs) - 1
        # 1.0 = perfectly increasing, 0.0 = perfectly decreasing, 0.5 = no trend
        mono_raw = transitions / total_trans
        # Convert to 0-1 scale where 1 = strongly monotonic (either direction)
        monotonicity = abs(mono_raw - 0.5) * 2
    else:
        monotonicity = 0.0

    # ── 4. Best single bucket & best adjacent pair ───────────────────
    best_single = max(valid_buckets, key=lambda b: abs(b["sharpe"])) if valid_buckets else None

    best_zone = None
    best_zone_sharpe = 0.0
    for i in range(len(buckets) - 1):
        pooled = buckets[i] + buckets[i + 1]
        if len(pooled) < 10:
            continue
        s = abs(_sharpe(pooled))
        if s > best_zone_sharpe:
            best_zone_sharpe = s
            best_zone = {
                "buckets":  [i + 1, i + 2],
                "n":        len(pooled),
                "avg_ret":  round(float(np.mean(pooled)), 6),
                "sharpe":   round(_sharpe(pooled), 4),
                "win_rate": round(float((np.array(pooled) > 0).mean()), 4),
            }

    # ── 5. U-shape / inverted-U detection ────────────────────────────
    if len(avgs) >= 4:
        wings = (avgs[0] + avgs[-1]) / 2
        center = np.mean(avgs[len(avgs) // 4: 3 * len(avgs) // 4])
        u_score = round(float(wings - center), 6)  # positive = U, negative = inverted-U
    else:
        u_score = 0.0

    # ── 6. Tail behavior (pooled top-2 vs bottom-2) ─────────────────
    if len(buckets) >= 4:
        bottom_pool = buckets[0] + buckets[1]
        top_pool = buckets[-2] + buckets[-1]
        tail_spread = (float(np.mean(top_pool)) - float(np.mean(bottom_pool))
                       if bottom_pool and top_pool else 0.0)
        tail_spread = round(tail_spread, 6)
    else:
        tail_spread = 0.0

    # ── 7. Pattern classification ────────────────────────────────────
    pattern = _classify_pattern(avgs, float(pr), float(sr), u_score, monotonicity)

    # ── 8. Robustness diagnostics ────────────────────────────────────
    robustness = _robustness_diagnostics(pairs, x_col, y_col, n_buckets)

    # ── 9. Composite score ───────────────────────────────────────────
    composite = _composite_score(
        rank_corr=abs(float(sr)),
        monotonicity=monotonicity,
        consistency_pct=robustness.get("yearly_consistency_pct") or 0,
        half_stability=robustness.get("half_sample_stable", False),
        concentration=robustness.get("concentration_risk") or 1.0,
        best_zone_sharpe=best_zone_sharpe,
        n=n,
        extreme_coverage=robustness.get("extreme_coverage", 1.0),
        mi=mi_val,
    )

    return _jsonify({
        "x_col":       x_col,
        "y_col":       y_col,
        "ticker":      ticker,
        "n":           n,

        # Correlation
        "pearson_r":   round(float(pr), 4),
        "pearson_p":   round(float(pp), 6),
        "spearman_r":  round(float(sr), 4),
        "spearman_p":  round(float(sp_val), 6),
        "mi":          round(mi_val, 6),
        "pearson_spearman_div": round(abs(float(pr)) - abs(float(sr)), 4),

        # Bucket profile
        "bucket_stats":   bucket_stats,
        "monotonicity":   round(monotonicity, 4),
        "u_score":        u_score,
        "tail_spread":    tail_spread,

        # Best zones
        "best_single_bucket": best_single,
        "best_adjacent_zone": best_zone,

        # Pattern
        "pattern": pattern,

        # Robustness
        "robustness":      robustness,
        "composite_score": composite,
    })


# ── Pattern classification ───────────────────────────────────────────────────

def _classify_pattern(avgs: list[float], pearson: float, spearman: float,
                      u_score: float, monotonicity: float) -> str:
    """Classify the dominant relationship pattern."""
    if len(avgs) < 3:
        return "insufficient_data"

    overall_range = max(avgs) - min(avgs) if avgs else 0
    if overall_range < 1e-8:
        return "flat"

    # Strong monotonic
    if monotonicity > 0.75 and abs(spearman) > 0.03:
        return "monotonic_positive" if spearman > 0 else "monotonic_negative"

    # Threshold / step first — a single extreme decile (e.g. only D10 elevated)
    # would otherwise be miscategorised as U-shape because the wing AVERAGE
    # gets pulled up by that one extreme even though the other wing is normal.
    diffs = [avgs[i + 1] - avgs[i] for i in range(len(avgs) - 1)]
    if diffs:
        max_diff = max(abs(d) for d in diffs)
        if max_diff > overall_range * 0.5:
            return "threshold"

    # U or inverted-U — require BOTH wings to be on the correct side of the
    # center, not just the average. Without this guard, a one-sided extreme
    # (e.g. D10 huge, D1 normal) fires u_shape because (D1+D10)/2 still
    # exceeds the middle.
    if len(avgs) >= 4 and abs(u_score) > overall_range * 0.3 and monotonicity < 0.5:
        center_mean = float(np.mean(avgs[len(avgs) // 4: 3 * len(avgs) // 4]))
        if u_score > 0:
            # U-shape: both extremes should sit above the middle.
            if min(avgs[0], avgs[-1]) > center_mean + overall_range * 0.15:
                return "u_shape"
        else:
            # Inverted-U: both extremes should sit below the middle.
            if max(avgs[0], avgs[-1]) < center_mean - overall_range * 0.15:
                return "inverted_u"

    # Moderate linear
    if abs(pearson) > 0.03:
        return "linear_weak_positive" if pearson > 0 else "linear_weak_negative"

    # Isolated region: best bucket stands out but no overall trend
    if avgs:
        best_idx = max(range(len(avgs)), key=lambda i: abs(avgs[i] - np.mean(avgs)))
        deviation = abs(avgs[best_idx] - np.mean(avgs))
        if deviation > overall_range * 0.4:
            return "isolated_region"

    return "no_clear_pattern"


# ── Robustness diagnostics ───────────────────────────────────────────────────

def _robustness_diagnostics(pairs, x_col: str, y_col: str,
                            n_buckets: int = 10) -> dict:
    """Temporal stability and concentration tests.

    Uses GLOBAL bucket assignments (computed once over the full corpus)
    for all subperiod comparisons.  Per-year or per-half sub-bucketing
    (re-ranking within the subperiod) is intentionally avoided: a row's
    decile must not change based on which time window we filter to.
    D10 = globally-highest-ranked rows; they remain D10 in every year
    and every half-period we slice to.
    """
    n = len(pairs)

    # ── Global bucket assignment (computed ONCE over full corpus) ────────
    # Pairs are (x, y, date).  Sort by x, assign bucket 0..n_buckets-1.
    # For IS mode x = raw metric value → equivalent to is_bins full-history
    # per-ticker rank.  For WF mode x = WF bin value (1..10) → same result
    # as the stored wf_bins assignment.
    sorted_idx = sorted(range(n), key=lambda i: pairs[i][0])
    global_bucket: list = [0] * n
    for rank, idx in enumerate(sorted_idx):
        global_bucket[idx] = min(int(rank / n * n_buckets), n_buckets - 1)

    # Top-2 and bottom-2 global-bucket thresholds
    top2_floor = n_buckets - 2    # indices {n_buckets-2, n_buckets-1}
    bot2_ceil  = 1                 # indices {0, 1}

    # Full-corpus direction: used to orient yearly / LOYO comparisons
    full_top_ys = [pairs[i][1] for i in range(n) if global_bucket[i] >= top2_floor]
    full_bot_ys = [pairs[i][1] for i in range(n) if global_bucket[i] <= bot2_ceil]
    full_buckets_valid = bool(full_top_ys and full_bot_ys)
    full_direction = 0
    full_top_avg = full_bot_avg = 0.0
    if full_buckets_valid:
        full_top_avg = float(np.mean(full_top_ys))
        full_bot_avg = float(np.mean(full_bot_ys))
        full_direction = 1 if full_top_avg > full_bot_avg else -1

    # ── Group by year (store index into pairs / global_bucket) ──────────
    by_year: dict[int, list] = defaultdict(list)
    for i, (x, y, d) in enumerate(pairs):
        yr = _year_from_date(d)
        if yr:
            by_year[yr].append(i)

    # ── Yearly consistency ───────────────────────────────────────────────
    # For each year: collect the returns of globally-defined top-2 and
    # bottom-2 bucket rows, compare directions.  No per-year re-ranking.
    years_checked = 0
    years_consistent = 0
    yearly_data: list = []
    yearly_spreads: dict = {}

    if full_buckets_valid:
        for yr, yr_idxs in sorted(by_year.items()):
            if len(yr_idxs) < n_buckets * 2:
                continue
            years_checked += 1
            yr_top_ys = [pairs[i][1] for i in yr_idxs if global_bucket[i] >= top2_floor]
            yr_bot_ys = [pairs[i][1] for i in yr_idxs if global_bucket[i] <= bot2_ceil]
            yr_top = float(np.mean(yr_top_ys)) if yr_top_ys else 0.0
            yr_bot = float(np.mean(yr_bot_ys)) if yr_bot_ys else 0.0
            yr_dir = 1 if yr_top > yr_bot else -1
            if yr_dir == full_direction:
                years_consistent += 1
            yearly_spreads[yr] = yr_top - yr_bot
            yearly_data.append({
                "year":      int(yr),
                "top_avg":   round(yr_top, 6),
                "bot_avg":   round(yr_bot, 6),
                "n":         len(yr_idxs),
                "top_beats": yr_top > yr_bot,
            })

    consistency_pct = round(years_consistent / years_checked * 100, 1) if years_checked else None

    # ── Concentration risk ───────────────────────────────────────────────
    total_abs_spread = sum(abs(v) for v in yearly_spreads.values())
    if total_abs_spread > 0 and yearly_spreads:
        concentration = round(max(abs(v) for v in yearly_spreads.values()) / total_abs_spread, 4)
    else:
        concentration = 1.0

    # ── Half-sample stability ────────────────────────────────────────────
    # Chronologically split; within each half use the global bucket label —
    # no re-ranking within the half.
    sorted_pairs_idx = sorted(range(n), key=lambda i: pairs[i][2] if pairs[i][2] else "")
    mid = n // 2
    h1_idxs = sorted_pairs_idx[:mid]
    h2_idxs = sorted_pairs_idx[mid:]
    half_stable = False

    if len(h1_idxs) >= n_buckets * 3 and len(h2_idxs) >= n_buckets * 3:
        h1_top = [pairs[i][1] for i in h1_idxs if global_bucket[i] >= top2_floor]
        h1_bot = [pairs[i][1] for i in h1_idxs if global_bucket[i] <= bot2_ceil]
        h2_top = [pairs[i][1] for i in h2_idxs if global_bucket[i] >= top2_floor]
        h2_bot = [pairs[i][1] for i in h2_idxs if global_bucket[i] <= bot2_ceil]
        if h1_top and h1_bot and h2_top and h2_bot:
            dir1 = float(np.mean(h1_top)) - float(np.mean(h1_bot))
            dir2 = float(np.mean(h2_top)) - float(np.mean(h2_bot))
            half_stable = (dir1 > 0 and dir2 > 0) or (dir1 < 0 and dir2 < 0)

    # ── Leave-one-year-out ───────────────────────────────────────────────
    # Ask: if we drop year Y, do the globally-defined top-2 / bottom-2
    # buckets still point the same direction?  Global buckets are frozen —
    # removing a year changes which rows are present but not their labels.
    loyo_fragile = False
    if full_buckets_valid and yearly_spreads:
        full_spread = full_top_avg - full_bot_avg
        for yr in yearly_spreads:
            rem_idxs = [i for i in range(n) if _year_from_date(pairs[i][2]) != yr]
            if len(rem_idxs) < n_buckets * 3:
                continue
            rt_ys  = [pairs[i][1] for i in rem_idxs if global_bucket[i] >= top2_floor]
            rb_ys  = [pairs[i][1] for i in rem_idxs if global_bucket[i] <= bot2_ceil]
            if not rt_ys or not rb_ys:
                continue
            rt   = float(np.mean(rt_ys))
            rbot = float(np.mean(rb_ys))
            if (full_spread > 0 and (rt - rbot) < 0) or (full_spread < 0 and (rt - rbot) > 0):
                loyo_fragile = True
                break

    # ── Min bucket size (over full-corpus global buckets) ────────────────
    bucket_y_by_gb: list = [[] for _ in range(n_buckets)]
    for i in range(n):
        bucket_y_by_gb[global_bucket[i]].append(pairs[i][1])
    bucket_sizes = [len(b) for b in bucket_y_by_gb if b]
    min_bucket_n = min(bucket_sizes) if bucket_sizes else 0

    # ── Extreme decile temporal coverage ────────────────────────────────
    # Fraction of qualifying years where globally-defined D1 and D10 have
    # at least one observation in that year.
    total_years = len(by_year)
    extreme_coverage = 1.0
    if total_years >= 3 and full_buckets_valid:
        d1_years = 0
        d10_years = 0
        for yr, yr_idxs in by_year.items():
            if len(yr_idxs) < n_buckets * 2:
                continue
            if any(global_bucket[i] == 0 for i in yr_idxs):
                d1_years += 1
            if any(global_bucket[i] == n_buckets - 1 for i in yr_idxs):
                d10_years += 1
        checked = max(years_checked, 1)
        d1_cov  = d1_years  / checked
        d10_cov = d10_years / checked
        extreme_coverage = min(d1_cov, d10_cov)

    return {
        "yearly_consistency_pct":  consistency_pct,
        "years_checked":           years_checked,
        "years_consistent":        years_consistent,
        "yearly_data":             yearly_data,
        "concentration_risk":      concentration,
        "half_sample_stable":      half_stable,
        "loyo_fragile":            loyo_fragile,
        "min_bucket_n":            min_bucket_n,
        "extreme_coverage":        round(extreme_coverage, 4),
    }


# ── Composite robustness score ───────────────────────────────────────────────

def _composite_score(rank_corr: float, monotonicity: float,
                     consistency_pct: float, half_stability: bool,
                     concentration: float, best_zone_sharpe: float,
                     n: int, extreme_coverage: float = 1.0,
                     mi: float = 0.0) -> float:
    """
    0-100 composite score. Higher = more robust signal.
    9 components, each normalized to 0-1, divided by max possible.
    """
    # 1. Rank correlation strength (0-1, capped at |r|=0.20)
    c_rank = min(rank_corr / 0.20, 1.0)

    # 2. Monotonicity (already 0-1)
    c_mono = monotonicity

    # 3. Yearly consistency (0-1)
    c_consistency = (consistency_pct / 100.0) if consistency_pct else 0.0

    # 4. Half-sample stability (binary 0 or 1)
    c_half = 1.0 if half_stability else 0.0

    # 5. Low concentration (1 = well-spread, 0 = all in one year)
    c_concentration = max(0.0, 1.0 - concentration)

    # 6. Best-zone Sharpe (0-1, capped at |sharpe|=0.5)
    c_sharpe = min(abs(best_zone_sharpe) / 0.5, 1.0)

    # 7. Sample size bonus (0-0.5, >500 gets full bonus)
    c_sample = min(n / 1000, 0.5)

    # 8. Extreme decile temporal coverage (0-1)
    # Penalizes signals where D1 or D10 only have data in a subset of years
    # (e.g., D1 fires only in 2020-2022 but not 2023-2025)
    c_coverage = extreme_coverage

    # 9. Mutual information (capped at 0.10 nats — strong for financial data)
    c_mi = min(mi / 0.10, 1.0)

    raw = (c_rank + c_mono + c_consistency + c_half + c_concentration
           + c_sharpe + c_sample + c_coverage + c_mi) / 8.5
    return round(raw * 100, 1)


# ── Multivariate interaction scanner ─────────────────────────────────────────

def _ols_r2(X: np.ndarray, y: np.ndarray) -> float:
    """Simple OLS R² for one or more predictors."""
    if len(y) < X.shape[1] + 5:
        return 0.0
    X_int = np.column_stack([np.ones(len(X)), X])
    coeffs, *_ = np.linalg.lstsq(X_int, y, rcond=None)
    y_pred = X_int @ coeffs
    ss_res = float(np.sum((y - y_pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    return round(1.0 - ss_res / ss_tot, 6) if ss_tot > 0 else 0.0


def _zone_stats(ys: list[float], label: str) -> dict:
    """Compute stats for a conditional zone (quadrant/octant)."""
    if not ys:
        return {"label": label, "n": 0, "avg_ret": None, "win_rate": None, "sharpe": 0.0}
    a = np.array(ys)
    return {
        "label":    label,
        "n":        len(ys),
        "avg_ret":  round(float(a.mean()), 6),
        "med_ret":  round(float(np.median(a)), 6),
        "win_rate": round(float((a > 0).mean()), 4),
        "std_dev":  round(float(a.std()), 6),
        "sharpe":   round(_sharpe(ys), 4),
    }


def scan_interaction_2f(rows: list[dict], feat_a: str, feat_b: str,
                        y_col: str, ticker: Optional[str] = None,
                        baseline_best_sharpe: float = 0.0) -> Optional[dict]:
    """
    Test whether combining feat_a and feat_b improves prediction of y_col.
    Splits each feature at median → 4 quadrants, computes stats per quadrant,
    measures interaction lift vs single-factor baseline.
    """
    valid = [r for r in rows
             if r.get(feat_a) is not None and r.get(feat_b) is not None
             and r.get(y_col) is not None]
    n = len(valid)
    if n < 60:
        return None

    xa = np.array([float(r[feat_a]) for r in valid])
    xb = np.array([float(r[feat_b]) for r in valid])
    ya = np.array([float(r[y_col]) for r in valid])

    med_a = float(np.median(xa))
    med_b = float(np.median(xb))

    # 4 quadrants
    zones = {
        f"{feat_a}_H+{feat_b}_H": [],
        f"{feat_a}_H+{feat_b}_L": [],
        f"{feat_a}_L+{feat_b}_H": [],
        f"{feat_a}_L+{feat_b}_L": [],
    }
    for i in range(n):
        a_high = xa[i] >= med_a
        b_high = xb[i] >= med_b
        key = f"{feat_a}_{'H' if a_high else 'L'}+{feat_b}_{'H' if b_high else 'L'}"
        zones[key].append(float(ya[i]))

    quadrants = [_zone_stats(ys, label) for label, ys in zones.items()]
    valid_quads = [q for q in quadrants if q["n"] >= 10]
    if not valid_quads:
        return None

    best_quad = max(valid_quads, key=lambda q: abs(q["sharpe"]))
    best_quad_sharpe = abs(best_quad["sharpe"])

    # Interaction lift: how much better is the best quadrant vs best single-factor?
    interaction_lift = round(best_quad_sharpe - abs(baseline_best_sharpe), 4)

    # OLS: 2-variable R² vs single-variable R²
    r2_combo = _ols_r2(np.column_stack([xa, xb]), ya)
    r2_a = _ols_r2(xa.reshape(-1, 1), ya)
    r2_b = _ols_r2(xb.reshape(-1, 1), ya)

    # Conditional monotonicity: within top half of A, does B still predict?
    a_high_mask = xa >= med_a
    a_low_mask = xa < med_a
    cond_mono = {}
    for label, mask in [("b_given_a_high", a_high_mask), ("b_given_a_low", a_low_mask)]:
        sub_xb = xb[mask]
        sub_ya = ya[mask]
        if len(sub_xb) >= 20:
            sr, _ = sp_stats.spearmanr(sub_xb, sub_ya)
            cond_mono[label] = round(float(sr), 4)
        else:
            cond_mono[label] = None

    b_high_mask = xb >= med_b
    b_low_mask = xb < med_b
    for label, mask in [("a_given_b_high", b_high_mask), ("a_given_b_low", b_low_mask)]:
        sub_xa = xa[mask]
        sub_ya = ya[mask]
        if len(sub_xa) >= 20:
            sr, _ = sp_stats.spearmanr(sub_xa, sub_ya)
            cond_mono[label] = round(float(sr), 4)
        else:
            cond_mono[label] = None

    # Composite interaction score
    c_lift = min(max(interaction_lift, 0) / 0.10, 1.0)
    c_r2_gain = min(max(r2_combo - max(r2_a, r2_b), 0) / 0.005, 1.0)
    c_quad_sharpe = min(best_quad_sharpe / 0.3, 1.0)
    c_sample = min(min(q["n"] for q in valid_quads) / 100, 1.0)
    composite = round((c_lift + c_r2_gain + c_quad_sharpe + c_sample) / 4 * 100, 1)

    return _jsonify({
        "combo":          [feat_a, feat_b],
        "y_col":          y_col,
        "ticker":         ticker,
        "n":              n,
        "quadrants":      quadrants,
        "best_quadrant":  best_quad,
        "baseline_best_single_sharpe": baseline_best_sharpe,
        "interaction_lift": interaction_lift,
        "ols_r2":         r2_combo,
        "single_r2_a":    r2_a,
        "single_r2_b":    r2_b,
        "r2_gain":        round(r2_combo - max(r2_a, r2_b), 6),
        "conditional_monotonicity": cond_mono,
        "composite_interaction_score": composite,
    })


def scan_interaction_3f(rows: list[dict], feat_a: str, feat_b: str, feat_c: str,
                        y_col: str, ticker: Optional[str] = None,
                        baseline_2f_sharpe: float = 0.0) -> Optional[dict]:
    """
    Test 3-feature combination. Splits each at median → 8 octants.
    Only run when sample size supports it (min 200 rows).
    """
    valid = [r for r in rows
             if r.get(feat_a) is not None and r.get(feat_b) is not None
             and r.get(feat_c) is not None and r.get(y_col) is not None]
    n = len(valid)
    if n < 200:
        return None

    xa = np.array([float(r[feat_a]) for r in valid])
    xb = np.array([float(r[feat_b]) for r in valid])
    xc = np.array([float(r[feat_c]) for r in valid])
    ya = np.array([float(r[y_col]) for r in valid])

    med_a, med_b, med_c = float(np.median(xa)), float(np.median(xb)), float(np.median(xc))

    # 8 octants
    zones: dict[str, list[float]] = {}
    for i in range(n):
        a_tag = "H" if xa[i] >= med_a else "L"
        b_tag = "H" if xb[i] >= med_b else "L"
        c_tag = "H" if xc[i] >= med_c else "L"
        key = f"{a_tag}{b_tag}{c_tag}"
        zones.setdefault(key, []).append(float(ya[i]))

    octants = [_zone_stats(ys, label) for label, ys in sorted(zones.items())]
    valid_octs = [o for o in octants if o["n"] >= 10]
    if not valid_octs:
        return None

    best_oct = max(valid_octs, key=lambda o: abs(o["sharpe"]))
    best_oct_sharpe = abs(best_oct["sharpe"])

    # 3-variable OLS R²
    r2_combo = _ols_r2(np.column_stack([xa, xb, xc]), ya)
    r2_ab = _ols_r2(np.column_stack([xa, xb]), ya)
    r2_ac = _ols_r2(np.column_stack([xa, xc]), ya)
    r2_bc = _ols_r2(np.column_stack([xb, xc]), ya)
    best_2f_r2 = max(r2_ab, r2_ac, r2_bc)

    interaction_lift = round(best_oct_sharpe - abs(baseline_2f_sharpe), 4)

    composite = round(min(abs(interaction_lift) / 0.05, 1.0) * 50
                       + min(max(r2_combo - best_2f_r2, 0) / 0.003, 1.0) * 25
                       + min(min(o["n"] for o in valid_octs) / 50, 1.0) * 25, 1)

    return _jsonify({
        "combo":           [feat_a, feat_b, feat_c],
        "y_col":           y_col,
        "ticker":          ticker,
        "n":               n,
        "octants":         octants,
        "best_octant":     best_oct,
        "baseline_2f_sharpe": baseline_2f_sharpe,
        "interaction_lift": interaction_lift,
        "ols_r2":          r2_combo,
        "best_2f_r2":      best_2f_r2,
        "r2_gain":         round(r2_combo - best_2f_r2, 6),
        "composite_interaction_score": composite,
    })


def scan_interaction_4f(rows: list[dict], feats: list[str],
                        y_col: str, ticker: Optional[str] = None,
                        baseline_sharpe: float = 0.0) -> Optional[dict]:
    """Test 4-feature combination. 16 zones from median splits. Min 300 rows."""
    valid = [r for r in rows
             if all(r.get(f) is not None for f in feats)
             and r.get(y_col) is not None]
    n = len(valid)
    if n < 300 or len(feats) != 4:
        return None

    arrays = [np.array([float(r[f]) for r in valid]) for f in feats]
    ya = np.array([float(r[y_col]) for r in valid])
    medians = [float(np.median(a)) for a in arrays]

    zones: dict[str, list[float]] = {}
    for i in range(n):
        tag = "".join("H" if arrays[j][i] >= medians[j] else "L" for j in range(4))
        zones.setdefault(tag, []).append(float(ya[i]))

    zone_stats_list = [_zone_stats(ys, label) for label, ys in sorted(zones.items())]
    valid_zones = [z for z in zone_stats_list if z["n"] >= 8]
    if not valid_zones:
        return None

    best = max(valid_zones, key=lambda z: abs(z["sharpe"]))
    best_sharpe = abs(best["sharpe"])
    lift = round(best_sharpe - abs(baseline_sharpe), 4)

    r2 = _ols_r2(np.column_stack(arrays), ya)

    composite = round(min(abs(lift) / 0.05, 1.0) * 40
                       + min(r2 / 0.01, 1.0) * 20
                       + min(best_sharpe / 0.3, 1.0) * 20
                       + min(min(z["n"] for z in valid_zones) / 30, 1.0) * 20, 1)

    return _jsonify({
        "combo":      feats,
        "y_col":      y_col,
        "ticker":     ticker,
        "n":          n,
        "zones":      zone_stats_list,
        "best_zone":  best,
        "baseline_sharpe": baseline_sharpe,
        "interaction_lift": lift,
        "ols_r2":     r2,
        "composite_interaction_score": composite,
    })


def combo_robustness(rows: list[dict], feats: list[str], y_col: str,
                     best_zone_label: str, ticker: Optional[str] = None) -> dict:
    """Robustness checks for a multi-factor combo's best zone."""
    valid = [r for r in rows
             if all(r.get(f) is not None for f in feats)
             and r.get(y_col) is not None]
    if len(valid) < 60:
        return _jsonify({"error": "insufficient data"})

    arrays = [np.array([float(r[f]) for r in valid]) for f in feats]
    ya = np.array([float(r[y_col]) for r in valid])
    medians = [float(np.median(a)) for a in arrays]

    zone_mask = np.ones(len(valid), dtype=bool)
    for j, c in enumerate(best_zone_label):
        if c == "H":
            zone_mask &= (arrays[j] >= medians[j])
        else:
            zone_mask &= (arrays[j] < medians[j])

    zone_returns = ya[zone_mask]
    n_zone = int(zone_mask.sum())
    if n_zone < 10:
        return _jsonify({"n_zone": n_zone, "warnings": ["too few in zone"]})

    # Yearly consistency
    by_year: dict[int, tuple] = {}
    for i, r in enumerate(valid):
        yr = _year_from_date(r.get("trade_date"))
        if yr is None:
            continue
        if yr not in by_year:
            by_year[yr] = ([], [])
        if zone_mask[i]:
            by_year[yr][0].append(float(ya[i]))
        else:
            by_year[yr][1].append(float(ya[i]))

    years_checked = 0
    years_consistent = 0
    yearly_zone_avgs = {}
    for yr in sorted(by_year):
        zone_ys, non_ys = by_year[yr]
        if len(zone_ys) < 5 or len(non_ys) < 5:
            continue
        years_checked += 1
        z_avg = float(np.mean(zone_ys))
        yearly_zone_avgs[yr] = z_avg
        if z_avg > float(np.mean(non_ys)):
            years_consistent += 1

    consistency_pct = round(years_consistent / years_checked * 100, 1) if years_checked else None
    total_abs = sum(abs(v) for v in yearly_zone_avgs.values()) if yearly_zone_avgs else 0
    concentration = round(max(abs(v) for v in yearly_zone_avgs.values()) / total_abs, 4) if total_abs > 0 else 1.0

    # Half-sample
    mid = len(valid) // 2
    h1 = ya[:mid][zone_mask[:mid]]
    h2 = ya[mid:][zone_mask[mid:]]
    half_stable = (len(h1) >= 5 and len(h2) >= 5
                   and (float(h1.mean()) > 0) == (float(h2.mean()) > 0))

    # Equity
    equity, peak, max_dd, wins = 1.0, 1.0, 0.0, 0
    for ret in zone_returns:
        equity *= (1.0 + ret)
        peak = max(peak, equity)
        max_dd = min(max_dd, (equity - peak) / peak)
        if ret > 0:
            wins += 1

    warnings = []
    if concentration > 0.5:
        warnings.append(f"concentrated: {concentration*100:.0f}% from single best year")
    if not half_stable:
        warnings.append("half-sample unstable")
    if consistency_pct is not None and consistency_pct < 50:
        warnings.append(f"yearly consistency only {consistency_pct}%")
    if n_zone < 50:
        warnings.append(f"small sample: only {n_zone} in zone")

    return _jsonify({
        "n_zone": n_zone,
        "avg_ret": round(float(zone_returns.mean()), 6),
        "med_ret": round(float(np.median(zone_returns)), 6),
        "win_rate": round(wins / n_zone, 4) if n_zone else None,
        "final_equity": round(equity, 4),
        "max_drawdown": round(max_dd, 4),
        "yearly_consistency_pct": consistency_pct,
        "concentration_risk": concentration,
        "half_sample_stable": half_stable,
        "warnings": warnings,
    })
