"""Information Coefficient (IC) computation layer.

Sibling module to `row_compute.py`. Where `row_compute` produces per-row
bin assignments under three method specs (in_sample, walk_forward,
train_test), this module produces date-indexed rolling-IC series. The
two shapes — per-row bin and per-date IC — don't unify cleanly, so they
live in parallel modules with the same generic-naming intent.

Three primitives:

  - `rolling_ic_single_ticker(rows, metric, outcome, window)` —
      time-series IC for one ticker. Rolling Spearman of
      (metric, forward_return) over a trailing window.

  - `rolling_ic_cross_sectional(rows, metric, outcome, window, min_tickers_per_day)` —
      cross-sectional daily IC across all tickers per day, then a
      trailing rolling mean of that daily series.

  - `sign_stability_from_rolling(rolling, reference_ic, epsilon)` —
      fraction of rolling windows whose IC sign agrees with a reference,
      with a noise-floor ε to classify "neutral" windows and a hard
      suppression rule when the reference itself is below noise.

Plus `noise_floor_epsilon(mode, window, horizon, k_tickers)` — derives ε
from the actual window length and forward-return horizon (overlapping
forward returns shrink the effective sample). ε is mode-aware: single-
ticker windows have ~2× the noise of cross-sectional rolled values.

# Step IC.1 scope
Pure functions, no endpoint, no DB, no caching. Subsequent steps wire
these into /analyze (IC.2/IC.3) and a new /ic-batch endpoint (IC.4),
plus frontend (IC.5).
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Literal, Optional

import numpy as np
from scipy import stats as sp_stats


# ── JSON-safe float helper ───────────────────────────────────────────────

def finite_or_none(v: float, ndigits: int = 6) -> Optional[float]:
    """Return round(v, ndigits) if v is finite, else None.

    Prevents json.dumps() from raising ValueError on inf/nan values that
    arise from degenerate IC windows. The canonical case: noise_floor_epsilon
    returns +inf when k_tickers < 2 (cross-sectional series with too few
    tickers in every rolling window). JSON null is the correct representation
    — the frontend already handles None epsilon as "n/a".

    Used by both _compute_ic_batch_sync (oi_analysis.py) and the offline
    precompute_ic_all.py script so the sanitization is shared.
    """
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return round(f, ndigits) if math.isfinite(f) else None


# ── Output contracts ─────────────────────────────────────────────────────

@dataclass(frozen=True)
class IcPoint:
    """One rolling-IC observation, aligned to the latest date in its window.

    `n` is the window length (single-ticker mode) or the median cross-section
    size in the rolling window (cross-sectional mode). Surfaced to the UI so
    the noise floor can be displayed alongside the value.

    `sign_class` is populated by `classified_rolling_ic` after the reference
    IC and ε are known. One of "same", "opposite", "neutral", or None when
    not yet classified. The rolling primitives leave it as None.
    """
    date: Any
    ic: float
    n: int
    sign_class: Optional[str] = None


@dataclass(frozen=True)
class SignStability:
    """Sign-stability summary for a rolling-IC series against a reference.

    `stability` is None when the result is suppressed (see below).
    `suppression_reason` ∈ {None, "reference_below_noise", "all_windows_neutral"}.

    Suppression rule: if |reference_ic| < epsilon, the reference itself is
    statistical noise and there is no meaningful sign for windows to be
    "stable" around. Returning a confident-looking percentage in that case
    would be exactly the false signal this tooling exists to prevent.
    """
    stability: Optional[float]
    reference_ic: float
    epsilon: float
    n_same: int
    n_opposite: int
    n_neutral: int
    n_total: int
    suppressed: bool
    suppression_reason: Optional[str]


# ── ε derivation (noise floor) ───────────────────────────────────────────

def _horizon_from_outcome(outcome_col: str) -> int:
    """Extract the forward-return horizon in trading days from an outcome
    column name like 'ret_5d_fwd_oc' → 5. Defaults to 1 if no digit found.
    """
    m = re.search(r"(\d+)d", outcome_col)
    return int(m.group(1)) if m else 1


def noise_floor_epsilon(
    mode: Literal["single_ticker", "cross_sectional"],
    window: int,
    horizon: int,
    k_tickers: Optional[int] = None,
) -> float:
    """Spearman-IC noise floor under H₀, mode-aware and horizon-corrected.

    The naïve formula `1/√(W-1)` understates noise when forward returns
    overlap. With a `horizon`-day forward return, adjacent observations
    share `horizon - 1` of `horizon` future days, so the effective number
    of independent observations in a `W`-day window is ≈ `W / horizon`.

      single_ticker:    ε = 1 / √(W/horizon - 1)
      cross_sectional:  ε = 1 / √((K - 1) · W/horizon)
        — daily cross-sectional IC has SE ≈ 1/√(K-1); rolling-mean over
          W/horizon effectively-independent days tightens to that ÷ √(W/h).

    For W=252, horizon=5, K=80:
        ε_single   ≈ 0.143
        ε_cross    ≈ 0.016
    Almost 10× different — sign-stability needs the mode-aware ε, otherwise
    the single-ticker leaderboard fills with false positives.

    Returns `+inf` for cross_sectional with K < 2 (degenerate, suppress
    everything).
    """
    if window < 2 or horizon < 1:
        raise ValueError(f"window={window}, horizon={horizon} must be ≥ 2 and ≥ 1")
    eff_n = max(window / float(horizon) - 1.0, 1.0)
    if mode == "single_ticker":
        return 1.0 / math.sqrt(eff_n)
    if mode == "cross_sectional":
        if not k_tickers or k_tickers < 2:
            return float("inf")
        return 1.0 / math.sqrt((k_tickers - 1) * (eff_n + 1.0))
    raise ValueError(f"unknown IC mode: {mode!r}")


# ── Rolling IC primitives ────────────────────────────────────────────────

def _coerce_pair(x: Any, y: Any) -> Optional[tuple[float, float]]:
    """Return (xf, yf) as floats, or None if either is missing/NaN/non-numeric."""
    if x is None or y is None:
        return None
    try:
        xf, yf = float(x), float(y)
    except (TypeError, ValueError):
        return None
    if math.isnan(xf) or math.isnan(yf):
        return None
    return xf, yf


def rolling_ic_single_ticker(
    rows_chrono: list[dict],
    metric: str,
    outcome: str,
    window: int = 252,
    stride: int = 1,
) -> list[IcPoint]:
    """Rolling Spearman IC of (metric, outcome) over trailing `window` rows.

    Input: one ticker's rows, chronologically sorted. Rows missing either
    field are excluded from window construction (they don't count toward the
    window size). Each emitted IcPoint is aligned to the latest input date
    in its window.

    Degenerate windows (zero-variance metric or zero-variance outcome) are
    skipped — they would produce undefined Spearman.

    `stride` controls how often a window is evaluated. stride=1 (default)
    evaluates every position; stride=3 evaluates every 3rd, reducing
    compute proportionally while leaving sign-stability counts accurate.
    The /analyze rolling-IC chart uses stride=1 (full resolution);
    /ic-batch uses stride=3 to stay under Cloudflare's 100s timeout.
    """
    if window < 2:
        raise ValueError(f"window must be ≥ 2, got {window}")
    if stride < 1:
        raise ValueError(f"stride must be ≥ 1, got {stride}")

    triples: list[tuple[Any, float, float]] = []
    for r in rows_chrono:
        d = r.get("trade_date")
        pair = _coerce_pair(r.get(metric), r.get(outcome))
        if d is None or pair is None:
            continue
        triples.append((d, pair[0], pair[1]))

    n = len(triples)
    if n < window:
        return []

    out: list[IcPoint] = []
    for end in range(window, n + 1, stride):
        win = triples[end - window:end]
        xs = np.fromiter((t[1] for t in win), dtype=np.float64, count=window)
        ys = np.fromiter((t[2] for t in win), dtype=np.float64, count=window)
        if xs.std() == 0 or ys.std() == 0:
            continue
        rho, _ = sp_stats.spearmanr(xs, ys)
        if math.isnan(rho):
            continue
        out.append(IcPoint(date=win[-1][0], ic=round(float(rho), 6), n=window))
    return out


def rolling_ic_cross_sectional(
    rows: list[dict],
    metric: str,
    outcome: str,
    window: int = 252,
    min_tickers_per_day: int = 5,
    stride: int = 1,
) -> list[IcPoint]:
    """Cross-sectional daily IC then trailing-mean over `window` days.

    For each trade_date, rank all tickers by `metric` and by `outcome`, then
    compute Spearman across them — one daily IC per day. The "rolling" view
    is a trailing mean of `window` consecutive daily ICs (the daily series
    is what the leaderboard and the colored line consume; the rolled value
    is what sign-stability is computed against).

    Days with fewer than `min_tickers_per_day` valid observations are
    skipped. Days with zero-variance metric or outcome (degenerate
    cross-section) are also skipped.

    Each emitted IcPoint's `n` field reports the *median* cross-section size
    over the window — useful for ε computation downstream.
    """
    if window < 1:
        raise ValueError(f"window must be ≥ 1, got {window}")

    by_date: dict[Any, list[tuple[float, float]]] = {}
    for r in rows:
        d = r.get("trade_date")
        if d is None:
            continue
        if r.get("ticker") is None:
            continue
        pair = _coerce_pair(r.get(metric), r.get(outcome))
        if pair is None:
            continue
        by_date.setdefault(d, []).append(pair)

    daily: list[tuple[Any, float, int]] = []  # (date, ic, n_tickers)
    for d in sorted(by_date.keys()):
        pairs = by_date[d]
        k = len(pairs)
        if k < min_tickers_per_day:
            continue
        xs = np.fromiter((p[0] for p in pairs), dtype=np.float64, count=k)
        ys = np.fromiter((p[1] for p in pairs), dtype=np.float64, count=k)
        if xs.std() == 0 or ys.std() == 0:
            continue
        rho, _ = sp_stats.spearmanr(xs, ys)
        if math.isnan(rho):
            continue
        daily.append((d, float(rho), k))

    n_days = len(daily)
    if n_days < window:
        return []

    out: list[IcPoint] = []
    for end in range(window, n_days + 1, stride):
        win = daily[end - window:end]
        ic_mean = float(np.mean([p[1] for p in win]))
        med_k = int(np.median([p[2] for p in win]))
        out.append(IcPoint(date=win[-1][0], ic=round(ic_mean, 6), n=med_k))
    return out


# ── Per-window classification ────────────────────────────────────────────

def classified_rolling_ic(
    rolling: list[IcPoint],
    reference_ic: float,
    epsilon: float,
) -> list[IcPoint]:
    """Return a new list of IcPoint with `sign_class` populated per window.

    Classification rule — kept consistent with `sign_stability_from_rolling`:
      - If |reference_ic| < epsilon (reference itself is noise), all windows
        get `sign_class="neutral"`. This matches the suppression semantics:
        when there is no meaningful reference sign, no window can be "stable"
        against it. The frontend renders a uniformly-grey line in this case.
      - Otherwise, classify per window:
          |ic| < epsilon                   → "neutral"
          sign(ic) == sign(reference_ic)   → "same"
          else                              → "opposite"

    This function is the single source of truth for sign_class. Endpoints
    that need per-window labels MUST go through this — duplicating the
    logic inline risks drift from `sign_stability_from_rolling`.
    """
    if abs(reference_ic) < epsilon:
        return [
            IcPoint(date=p.date, ic=p.ic, n=p.n, sign_class="neutral")
            for p in rolling
        ]
    ref_positive = reference_ic > 0
    out: list[IcPoint] = []
    for p in rolling:
        if abs(p.ic) < epsilon:
            cls = "neutral"
        elif (p.ic > 0) == ref_positive:
            cls = "same"
        else:
            cls = "opposite"
        out.append(IcPoint(date=p.date, ic=p.ic, n=p.n, sign_class=cls))
    return out


# ── Sign-stability ───────────────────────────────────────────────────────

def sign_stability_from_rolling(
    rolling: list[IcPoint],
    reference_ic: float,
    epsilon: float,
) -> SignStability:
    """Fraction of windows whose IC matches the reference's sign.

    Classification per window:
      - |ic| <  epsilon → neutral (excluded from numerator and denominator)
      - sign(ic) == sign(reference_ic) → same-sign
      - else → opposite-sign

    Sign-stability = n_same / (n_same + n_opposite).

    Suppression — hard rule:
      - If |reference_ic| < epsilon: the reference itself is below noise, so
        there is no meaningful sign for windows to be stable around. Return
        stability=None, suppressed=True. Surfaces as "—" / "n/a" in the UI.
      - If every window is neutral (n_same + n_opposite == 0): also
        suppressed; nothing decisive to report.
    """
    n_total = len(rolling)

    if abs(reference_ic) < epsilon:
        return SignStability(
            stability=None,
            reference_ic=reference_ic, epsilon=epsilon,
            n_same=0, n_opposite=0, n_neutral=n_total, n_total=n_total,
            suppressed=True, suppression_reason="reference_below_noise",
        )

    ref_positive = reference_ic > 0
    n_same = n_opp = n_neutral = 0
    for p in rolling:
        if abs(p.ic) < epsilon:
            n_neutral += 1
        elif (p.ic > 0) == ref_positive:
            n_same += 1
        else:
            n_opp += 1

    decisive = n_same + n_opp
    if decisive == 0:
        return SignStability(
            stability=None,
            reference_ic=reference_ic, epsilon=epsilon,
            n_same=0, n_opposite=0, n_neutral=n_neutral, n_total=n_total,
            suppressed=True, suppression_reason="all_windows_neutral",
        )

    return SignStability(
        stability=n_same / decisive,
        reference_ic=reference_ic, epsilon=epsilon,
        n_same=n_same, n_opposite=n_opp, n_neutral=n_neutral, n_total=n_total,
        suppressed=False, suppression_reason=None,
    )


# ── Per-ticker IC decomposition (IC.7) ──────────────────────────────────

def ic_decompose_cross_sectional(
    rows: list[dict],
    metric: str,
    outcome: str,
    min_tickers_per_day: int = 5,
    cutoff_date: Optional[str] = None,
) -> dict:
    """Per-ticker rank-product contribution to daily cross-sectional IC.

    For each trading day with at least `min_tickers_per_day` valid
    (metric, outcome) pairs:
      1. Rank all K tickers by `metric` and by `outcome` (midrank for ties)
      2. Center ranks: z_m = r_m − (K+1)/2, z_o = r_o − (K+1)/2
      3. Normalised contribution of ticker i to that day's Spearman IC:
             contrib[i,d] = z_m[i] · z_o[i] / (K·(K²−1)/12)
         Guarantees Σ_i contrib[i,d] = Spearman(metric, outcome) on that day.

    Per-ticker score = mean daily contribution over all days the ticker
    was present in the cross-section. Summing all scores weighted by
    (n_days[i] / n_total_days) recovers the reference IC — useful
    as a self-check for the caller.

    Reference IC convention matches /ic-batch and /analyze:
      • cutoff_date given → reference = mean of pre-cutoff daily ICs
      • cutoff_date None  → reference = mean of all daily ICs

    Returns dict with keys:
      tickers           [{ticker, score, n_days}] sorted by score descending
      reference_ic      float
      n_days            int   — total days with valid cross-sections
      n_tickers         int
      concentration_gini float | None  — Gini of |score|; 0 = equal,
                                         1 = perfectly concentrated
      effective_n        float | None  — (Σ|score|)² / Σ(score²); the
                                         effective number of contributing
                                         tickers (high = broad, low = concentrated)
      n_same_sign        int  — tickers whose score agrees with reference_ic sign
      n_opposite_sign    int
    """
    from scipy.stats import rankdata as _rankdata

    # Build date → {ticker: (metric_val, outcome_val)}
    by_date: dict = {}
    for r in rows:
        d = r.get("trade_date")
        tkr = r.get("ticker")
        if d is None or tkr is None:
            continue
        mv = r.get(metric)
        ov = r.get(outcome)
        if mv is None or ov is None:
            continue
        try:
            mf, of_ = float(mv), float(ov)
        except (TypeError, ValueError):
            continue
        if math.isnan(mf) or math.isnan(of_):
            continue
        by_date.setdefault(d, {})[tkr] = (mf, of_)

    # Accumulate per-ticker rank-product contributions across days
    contrib_sum: dict[str, float] = {}
    n_days_map:  dict[str, int]   = {}
    daily_ics:       list[float] = []
    pre_cutoff_ics:  list[float] = []

    # Per-ticker bubble-chart accumulators (populated in inner loop, used below)
    # Separated into top-half and bottom-half by metric rank so we can
    # direction-normalise after reference_ic is known (sign(reference_ic)
    # determines which half is the "predicted winner").
    n_pos_rp_map:    dict[str, int]   = {}  # days rank product > 0 (signal agreement)
    n_top_map:       dict[str, int]   = {}  # days ticker in top half by metric rank
    sum_ret_top_map: dict[str, float] = {}  # Σ outcomes when in top half
    n_pos_ret_top:   dict[str, int]   = {}  # positive-outcome count, top half
    n_bot_map:       dict[str, int]   = {}  # days ticker in bottom half
    sum_ret_bot_map: dict[str, float] = {}  # Σ outcomes when in bottom half
    n_pos_ret_bot:   dict[str, int]   = {}  # positive-outcome count, bottom half

    for d in sorted(by_date.keys()):
        tkr_vals = by_date[d]
        tickers  = list(tkr_vals.keys())
        K = len(tickers)
        if K < min_tickers_per_day:
            continue

        mv_arr = np.array([tkr_vals[t][0] for t in tickers], dtype=np.float64)
        ov_arr = np.array([tkr_vals[t][1] for t in tickers], dtype=np.float64)

        if mv_arr.std() == 0 or ov_arr.std() == 0:
            continue

        # Midrank for ties — matches scipy.stats.spearmanr internals
        r_m = _rankdata(mv_arr, method="average")
        r_o = _rankdata(ov_arr, method="average")

        mean_r = (K + 1) / 2.0
        z_m = r_m - mean_r
        z_o = r_o - mean_r

        # Normaliser: Σ_i z_m[i]·z_o[i] / norm = Spearman rho exactly
        norm = K * (K * K - 1) / 12.0
        if norm == 0:
            continue

        rho = float(np.dot(z_m, z_o) / norm)
        daily_ics.append(rho)

        d_str = str(d)
        if cutoff_date is None or d_str < cutoff_date:
            pre_cutoff_ics.append(rho)

        for i, tkr in enumerate(tickers):
            c = z_m[i] * z_o[i] / norm
            contrib_sum[tkr] = contrib_sum.get(tkr, 0.0) + c
            n_days_map[tkr]  = n_days_map.get(tkr, 0) + 1

            # Sign agreement: positive rank product = metric & outcome ranks aligned
            if z_m[i] * z_o[i] > 0:
                n_pos_rp_map[tkr] = n_pos_rp_map.get(tkr, 0) + 1

            # Conditional returns split by metric-rank half
            out_val = float(ov_arr[i])
            if r_m[i] > mean_r:        # top half by metric rank
                n_top_map[tkr]       = n_top_map.get(tkr, 0) + 1
                sum_ret_top_map[tkr] = sum_ret_top_map.get(tkr, 0.0) + out_val
                if out_val > 0:
                    n_pos_ret_top[tkr] = n_pos_ret_top.get(tkr, 0) + 1
            else:                      # bottom half (includes exact median — rare)
                n_bot_map[tkr]       = n_bot_map.get(tkr, 0) + 1
                sum_ret_bot_map[tkr] = sum_ret_bot_map.get(tkr, 0.0) + out_val
                if out_val > 0:
                    n_pos_ret_bot[tkr] = n_pos_ret_bot.get(tkr, 0) + 1

    n_days = len(daily_ics)
    if n_days == 0:
        return {
            "tickers": [], "reference_ic": 0.0,
            "n_days": 0, "n_tickers": 0,
            "concentration_gini": None, "effective_n": None,
            "n_same_sign": 0, "n_opposite_sign": 0,
        }

    reference_ic = (
        float(np.mean(pre_cutoff_ics)) if (cutoff_date and pre_cutoff_ics)
        else float(np.mean(daily_ics))
    )

    # Per-ticker mean daily contribution — this is the IC.7 score.
    # flag_top: True when reference_ic ≥ 0 (top half = predicted winner).
    # False when reference_ic < 0 (bottom half = predicted winner).
    # Y-axis and Size must use the same normalisation so the bubble chart
    # reads consistently regardless of the metric's sign.
    flag_top = reference_ic >= 0

    results = []
    for tkr, csum in contrib_sum.items():
        nd    = n_days_map[tkr]
        score = round(csum / nd, 8)

        # Sign agreement rate: fraction of days metric & outcome rank aligned
        n_pos_rp = n_pos_rp_map.get(tkr, 0)
        sign_agr = round(n_pos_rp / nd, 4) if nd > 0 else 0.5

        # Direction-normalised payoff: use predicted-winner half
        if flag_top:
            nf   = n_top_map.get(tkr, 0)
            sr   = sum_ret_top_map.get(tkr, 0.0)
            npos = n_pos_ret_top.get(tkr, 0)
        else:
            nf   = n_bot_map.get(tkr, 0)
            sr   = sum_ret_bot_map.get(tkr, 0.0)
            npos = n_pos_ret_bot.get(tkr, 0)

        avg_ret_flagged  = round(sr / nf, 6) if nf > 0 else None
        hit_rate_flagged = round(npos / nf, 4) if nf > 0 else None

        results.append({
            "ticker":              tkr,
            "score":               score,
            "n_days":              nd,
            "sign_agreement_rate": sign_agr,
            "avg_ret_flagged":     avg_ret_flagged,
            "hit_rate_flagged":    hit_rate_flagged,
            "n_flagged":           nf,
        })
    results.sort(key=lambda x: x["score"], reverse=True)

    # ── Concentration metrics ────────────────────────────────────────────
    scores_arr = np.array([r["score"] for r in results], dtype=np.float64)
    abs_arr    = np.abs(scores_arr)
    n_t        = len(abs_arr)

    if n_t > 1 and abs_arr.sum() > 0:
        # Gini of |score|: G = (2 Σ i·x_i)/(n·Σ x_i) − (n+1)/n  [x sorted asc, i=1..n]
        s   = np.sort(abs_arr)
        idx = np.arange(1, n_t + 1)
        gini = float((2.0 * np.dot(idx, s)) / (n_t * s.sum()) - (n_t + 1) / n_t)
        gini = round(max(0.0, min(1.0, gini)), 4)
    else:
        gini = None

    sum_abs     = float(abs_arr.sum())
    sum_sq      = float(np.dot(scores_arr, scores_arr))
    effective_n = round(sum_abs ** 2 / sum_sq, 2) if sum_sq > 0 else None

    ref_pos = reference_ic > 0
    n_same = sum(1 for r in results if r["score"] != 0 and (r["score"] > 0) == ref_pos)
    n_opp  = sum(1 for r in results if r["score"] != 0 and (r["score"] > 0) != ref_pos)

    return {
        "tickers":            results,
        "reference_ic":       round(reference_ic, 6),
        "n_days":             n_days,
        "n_tickers":          n_t,
        "concentration_gini": gini,
        "effective_n":        effective_n,
        "n_same_sign":        n_same,
        "n_opposite_sign":    n_opp,
    }
