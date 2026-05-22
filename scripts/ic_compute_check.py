"""IC.1 verification — hand-computed correctness tests for `ic_compute`.

Five tests plus property and ε-derivation checks. All assertions internal;
prints PASS lines for each. Runs without DB access.

  ε    — ε derivation: horizon extraction + mode-aware formula values
  A    — single-ticker monotone (x=i, y=i): every window ic=+1, stability=100%
  B    — single-ticker random (fixed seed): long-run ic ≈ 0 → suppressed
  C    — single-ticker regime flip (+ramp then −ramp vs monotone outcome):
           half the windows ic≈+1, half ic≈−1, mean ≈ 0 → suppressed
  D    — cross-sectional 5-day hand-rigged: daily IC matches by hand,
           trailing-mean window=2 matches by hand
  E    — sign-stability arithmetic on a hand-built rolling series

  prop — monotone+noise: high IC, 100% stability
  prop — sign-flip invariance: flipping x's sign flips reference but
           leaves sign-stability unchanged

Run:
    python scripts/ic_compute_check.py
"""
from __future__ import annotations

import io
import math
import sys
from pathlib import Path

# Allow running as a top-level script without installation.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# Windows console defaults to cp1252; PASS lines contain ε / √ / etc. Force
# UTF-8 stdout so the script runs cleanly on Windows + Linux alike.
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import numpy as np  # noqa: E402

from app.routers.ic_compute import (  # noqa: E402
    IcPoint,
    SignStability,
    noise_floor_epsilon,
    rolling_ic_cross_sectional,
    rolling_ic_single_ticker,
    sign_stability_from_rolling,
    _horizon_from_outcome,
)


# ──────────────────────────────────────────────────────────────────────────
# ε derivation
# ──────────────────────────────────────────────────────────────────────────

def test_epsilon_derivation():
    # Horizon extraction from outcome column name.
    assert _horizon_from_outcome("ret_5d_fwd_oc") == 5
    assert _horizon_from_outcome("ret_10d_fwd_oc") == 10
    assert _horizon_from_outcome("ret_1d_fwd_oc") == 1
    assert _horizon_from_outcome("no_horizon_here") == 1  # default

    # Single-ticker, W=252, horizon=1: ε = 1/√(251) ≈ 0.0631
    e = noise_floor_epsilon("single_ticker", window=252, horizon=1)
    assert abs(e - 1.0 / math.sqrt(251.0)) < 1e-9, e

    # Single-ticker, W=252, horizon=5: eff_n = 252/5 - 1 = 49.4 → ε = 1/√49.4 ≈ 0.1423
    e5 = noise_floor_epsilon("single_ticker", window=252, horizon=5)
    assert abs(e5 - 1.0 / math.sqrt(49.4)) < 1e-9, e5
    assert e5 > e, "5d horizon should give a larger ε than 1d"

    # Cross-sectional, W=252, h=5, K=80:
    #   ε = 1/√((K-1) · W/h) = 1/√(79 · 50.4) ≈ 0.01585
    e_cx = noise_floor_epsilon("cross_sectional", window=252, horizon=5, k_tickers=80)
    assert abs(e_cx - 1.0 / math.sqrt(79 * 50.4)) < 1e-9, e_cx
    assert e_cx < e5, "cross-sectional ε should be much tighter than single-ticker"

    # Degenerate K=1 → +inf (suppress everything).
    e_bad = noise_floor_epsilon("cross_sectional", window=252, horizon=5, k_tickers=1)
    assert math.isinf(e_bad)

    print(f"[ε] PASS: ε_single(W=252,h=1)={e:.4f}, ε_single(W=252,h=5)={e5:.4f}, "
          f"ε_cross(W=252,h=5,K=80)={e_cx:.4f}")


# ──────────────────────────────────────────────────────────────────────────
# A — single-ticker monotone
# ──────────────────────────────────────────────────────────────────────────

def test_a_monotone():
    rows = [{"trade_date": f"day_{i:04d}", "x": float(i), "y": float(i)}
            for i in range(300)]
    series = rolling_ic_single_ticker(rows, "x", "y", window=252)
    assert len(series) == 300 - 252 + 1, len(series)
    for p in series:
        assert abs(p.ic - 1.0) < 1e-9, f"expected ic=1.0, got {p.ic} on {p.date}"
        assert p.n == 252

    ref = float(np.mean([p.ic for p in series]))
    eps = noise_floor_epsilon("single_ticker", window=252, horizon=1)
    ss = sign_stability_from_rolling(series, reference_ic=ref, epsilon=eps)
    assert ss.suppressed is False
    assert ss.stability == 1.0, ss
    assert ss.n_same == len(series)
    assert ss.n_opposite == 0
    assert ss.n_neutral == 0
    print(f"[A] PASS monotone: {len(series)} windows ic=+1.0, stability=100%, neutral=0%")


# ──────────────────────────────────────────────────────────────────────────
# B — single-ticker random
# ──────────────────────────────────────────────────────────────────────────

def test_b_structurally_uncorrelated():
    """Sinusoid metric vs monotone outcome — no monotonic relationship → suppression.

    Within any window, x oscillates over multiple cycles while y ramps
    monotonically, so high y-values pair with the full range of x-values.
    Rank correlation is structurally near zero. Long-run IC < ε → suppress.

    (Random-data version would be flaky: 252-day windows share 251 days, so
    the long-run mean is essentially one single Spearman observation. Seed-
    dependent. A structural construction makes the property testable.)
    """
    rows = []
    for i in range(400):
        rows.append({"trade_date": f"day_{i:04d}",
                     "x": math.sin(i / 10.0),
                     "y": float(i)})
    series = rolling_ic_single_ticker(rows, "x", "y", window=252)
    assert len(series) > 0
    ref = float(np.mean([p.ic for p in series]))
    eps = noise_floor_epsilon("single_ticker", window=252, horizon=1)
    ss = sign_stability_from_rolling(series, reference_ic=ref, epsilon=eps)
    assert ss.suppressed is True, (
        f"sinusoid vs monotone should give |ref|<ε and suppress, "
        f"got ref={ref:.4f}, ε={eps:.4f}, ss={ss}")
    assert ss.suppression_reason == "reference_below_noise"
    print(f"[B] PASS structurally-uncorrelated (sin/monotone): "
          f"ref_ic={ref:+.4f}, ε={eps:.4f}, |ref|<ε → suppressed")


# ──────────────────────────────────────────────────────────────────────────
# C — single-ticker regime flip
# ──────────────────────────────────────────────────────────────────────────

def test_c_regime_flip():
    """Regime flip: pure +regime then pure -regime, long-run mean ≈ 0 → suppress.

    Need enough history that pure-regime windows dominate the small transition
    skew. Empirically, transition windows in this construction skew negative
    (rolling Spearman of a "ramp then drop" against a monotonic outcome
    produces strongly negative correlation, not the symmetric zero one might
    intuit). With n=2000 and window=100, pure-regime windows number ~1800 and
    transition windows number 100 — the transition skew is diluted enough that
    the mean falls below ε.
    """
    n = 2000
    rows = []
    for i in range(n):
        # Days 0..999: x and y both ramp up → pure +regime, IC=+1 inside.
        # Days 1000..1999: x ramps down (negatively), y ramps up → pure -regime, IC=-1.
        x = float(i) if i < 1000 else -float(i)
        y = float(i)
        rows.append({"trade_date": f"day_{i:04d}", "x": x, "y": y})

    series = rolling_ic_single_ticker(rows, "x", "y", window=100)
    assert len(series) == n - 100 + 1

    pos = sum(1 for p in series if p.ic > 0.95)
    neg = sum(1 for p in series if p.ic < -0.95)
    # ~900 windows of each regime expected.
    assert pos >= 800 and neg >= 800, (
        f"expected ~900 windows of each regime, got pos={pos}, neg={neg}")

    ref = float(np.mean([p.ic for p in series]))
    eps = noise_floor_epsilon("single_ticker", window=100, horizon=1)
    ss = sign_stability_from_rolling(series, reference_ic=ref, epsilon=eps)
    assert ss.suppressed is True, (
        f"regime flip with balanced pure-regime windows should suppress, "
        f"got ref={ref:.4f}, ε={eps:.4f}, ss={ss}")
    assert ss.suppression_reason == "reference_below_noise"
    print(f"[C] PASS regime flip: {pos} strongly-+, {neg} strongly-−, "
          f"ref={ref:+.4f} (|ref|<ε={eps:.4f}) → suppressed")


# ──────────────────────────────────────────────────────────────────────────
# D — cross-sectional 5-day hand-rigged
# ──────────────────────────────────────────────────────────────────────────

def test_d_cross_sectional():
    # 3 tickers × 5 days. Per-day Spearman computed by hand below.
    #
    # day 0  metric=[10,20,30]  outcome=[.01,.02,.03] → rank match → ρ = +1
    # day 1  metric=[30,20,10]  outcome=[.01,.02,.03] → rank reverse → ρ = -1
    # day 2  metric=[10,20,30]  outcome=[.03,.02,.01] → rank reverse → ρ = -1
    # day 3  metric=[10,30,20]  outcome=[.01,.02,.03]
    #        metric ranks=[1,3,2], outcome ranks=[1,2,3]
    #        d² = [0, 1, 1] → sum=2 → ρ = 1 - 6·2/(3·8) = 1 - 0.5 = +0.5
    # day 4  metric=[10,20,30]  outcome=[.01,.02,.03] → ρ = +1
    rows = []
    for day_offset, (m_pattern, o_pattern) in enumerate([
        ((10.0, 20.0, 30.0), (0.01, 0.02, 0.03)),  # +1
        ((30.0, 20.0, 10.0), (0.01, 0.02, 0.03)),  # -1
        ((10.0, 20.0, 30.0), (0.03, 0.02, 0.01)),  # -1
        ((10.0, 30.0, 20.0), (0.01, 0.02, 0.03)),  # +0.5
        ((10.0, 20.0, 30.0), (0.01, 0.02, 0.03)),  # +1
    ]):
        date = f"2020-01-{day_offset + 1:02d}"
        for tkr, x, y in zip(["A", "B", "C"], m_pattern, o_pattern):
            rows.append({"trade_date": date, "ticker": tkr, "x": x, "y": y})

    # window=2 over 5 days → 4 rolling points. Daily ICs: [+1, -1, -1, +0.5, +1].
    # Trailing-mean window=2:
    #   end day 1 (uses days 0..1): mean(+1, -1) = 0.0
    #   end day 2 (uses days 1..2): mean(-1, -1) = -1.0
    #   end day 3 (uses days 2..3): mean(-1, +0.5) = -0.25
    #   end day 4 (uses days 3..4): mean(+0.5, +1) = +0.75
    series = rolling_ic_cross_sectional(rows, "x", "y", window=2, min_tickers_per_day=3)
    expected = [0.0, -1.0, -0.25, 0.75]
    assert len(series) == 4, f"expected 4, got {len(series)}: {series}"
    for p, exp in zip(series, expected):
        assert abs(p.ic - exp) < 1e-6, f"on {p.date}: expected {exp}, got {p.ic}"
        assert p.n == 3, p.n
    print(f"[D] PASS cross-sectional 5-day: daily ICs (+1,-1,-1,+0.5,+1) "
          f"→ rolling means [0, -1, -0.25, +0.75] ✓")


# ──────────────────────────────────────────────────────────────────────────
# E — sign-stability arithmetic
# ──────────────────────────────────────────────────────────────────────────

def test_e_sign_stability():
    # 60 same-sign (ic=+0.20), 25 opposite (ic=-0.20), 15 neutral (ic=+0.01).
    # Reference = +0.20, epsilon = 0.05 → 0.01 is neutral, 0.20 is decisive.
    series = (
        [IcPoint(date=f"s{i}", ic=0.20, n=252) for i in range(60)]
        + [IcPoint(date=f"o{i}", ic=-0.20, n=252) for i in range(25)]
        + [IcPoint(date=f"n{i}", ic=0.01, n=252) for i in range(15)]
    )
    ss = sign_stability_from_rolling(series, reference_ic=0.20, epsilon=0.05)
    assert ss.suppressed is False
    assert ss.n_same == 60, ss.n_same
    assert ss.n_opposite == 25, ss.n_opposite
    assert ss.n_neutral == 15, ss.n_neutral
    assert ss.n_total == 100, ss.n_total
    expected = 60.0 / 85.0
    assert abs(ss.stability - expected) < 1e-12, ss.stability

    # Reference below noise → suppression.
    ss_sup = sign_stability_from_rolling(series, reference_ic=0.02, epsilon=0.05)
    assert ss_sup.suppressed is True
    assert ss_sup.stability is None
    assert ss_sup.suppression_reason == "reference_below_noise"

    # All-neutral case → suppression.
    all_neutral = [IcPoint(date=f"n{i}", ic=0.01, n=252) for i in range(20)]
    ss_nu = sign_stability_from_rolling(all_neutral, reference_ic=0.20, epsilon=0.05)
    assert ss_nu.suppressed is True
    assert ss_nu.stability is None
    assert ss_nu.suppression_reason == "all_windows_neutral"

    print(f"[E] PASS sign-stability arithmetic: "
          f"60+25+15 → stability=60/85={expected:.4f}; "
          f"suppression rules (ref<ε, all-neutral) fire correctly")


# ──────────────────────────────────────────────────────────────────────────
# Property tests
# ──────────────────────────────────────────────────────────────────────────

def test_property_monotone_with_noise():
    rng = np.random.default_rng(7)
    rows = []
    for i in range(300):
        rows.append({"trade_date": f"day_{i:04d}",
                     "x": float(i) + rng.normal() * 0.1,
                     "y": float(i)})
    series = rolling_ic_single_ticker(rows, "x", "y", window=252)
    ref = float(np.mean([p.ic for p in series]))
    eps = noise_floor_epsilon("single_ticker", window=252, horizon=1)
    ss = sign_stability_from_rolling(series, reference_ic=ref, epsilon=eps)
    assert ref > 0.9, f"expected high IC for monotone+noise, got {ref}"
    assert ss.suppressed is False
    assert ss.stability == 1.0
    print(f"[prop-monotone] PASS: ref={ref:.4f}, stability=100%")


def test_property_sign_flip_invariance():
    # Sign-flipping the metric should flip the reference's sign but leave
    # sign-stability unchanged — each window's IC flips along with the
    # reference, so same/opposite classifications are invariant.
    rng = np.random.default_rng(123)
    rows = []
    for i in range(400):
        rows.append({"trade_date": f"day_{i:04d}",
                     "x": float(i) + rng.normal() * 5.0,
                     "y": float(i) + rng.normal() * 3.0})
    series_pos = rolling_ic_single_ticker(rows, "x", "y", window=252)
    rows_flipped = [{**r, "x": -r["x"]} for r in rows]
    series_neg = rolling_ic_single_ticker(rows_flipped, "x", "y", window=252)

    ref_pos = float(np.mean([p.ic for p in series_pos]))
    ref_neg = float(np.mean([p.ic for p in series_neg]))
    assert abs(ref_pos + ref_neg) < 1e-6, f"ref should flip sign: {ref_pos} vs {ref_neg}"

    eps = noise_floor_epsilon("single_ticker", window=252, horizon=1)
    ss_pos = sign_stability_from_rolling(series_pos, reference_ic=ref_pos, epsilon=eps)
    ss_neg = sign_stability_from_rolling(series_neg, reference_ic=ref_neg, epsilon=eps)
    assert ss_pos.suppressed == ss_neg.suppressed
    if not ss_pos.suppressed:
        assert abs(ss_pos.stability - ss_neg.stability) < 1e-9, (
            f"stability should be invariant under sign flip: "
            f"{ss_pos.stability} vs {ss_neg.stability}")
    print(f"[prop-sign-flip] PASS: ref flips ({ref_pos:+.4f} → {ref_neg:+.4f}), "
          f"stability unchanged ({ss_pos.stability})")


# ──────────────────────────────────────────────────────────────────────────
# main
# ──────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_epsilon_derivation()
    test_a_monotone()
    test_b_structurally_uncorrelated()
    test_c_regime_flip()
    test_d_cross_sectional()
    test_e_sign_stability()
    test_property_monotone_with_noise()
    test_property_sign_flip_invariance()
    print()
    print("All IC.1 tests passed.")
