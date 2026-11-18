"""Filter gain written the slow, literal way -- per trade, every threshold
tried -- for the gates to hold surface_stats.filter_gain to. Shares nothing
with it but the rules (stated in surface_stats, the user's spec 2026-11-18).
"""
from __future__ import annotations

import math

MIN_N = 3
ONE_SIDED = (5, 10, 15, 20, 25, 30, 35, 40)
BOTH = (5, 10, 15, 20)


def reference(values: list, pnls: list) -> dict | None:
    """values[t], pnls[t] per TRADE (None / non-finite = not in the sample).
    Returns {score, side, n_removed, removed_mean, n_kept, kept_mean, lo, hi}
    or None below MIN_N."""
    tr = [(float(v), float(p)) for v, p in zip(values, pnls)
          if v is not None and p is not None and math.isfinite(float(v)) and math.isfinite(float(p))]
    n = len(tr)
    if n < MIN_N:
        return None
    zero = {"score": 0.0, "side": None}
    xs = sorted({v for v, _ in tr})
    if len(xs) < 2:
        return zero

    def left_x(pct):     # the deepest "skip v < X" removing at most pct % (X a value, so ties stay together)
        best = None
        for x in xs[1:]:
            if sum(1 for v, _ in tr if v < x) * 100 <= pct * n:
                best = x
        return best

    def right_x(pct):    # the deepest "skip v >= X"
        best = None
        for x in reversed(xs[1:]):
            if sum(1 for v, _ in tr if v >= x) * 100 <= pct * n:
                best = x
        return best

    def evaluate(lo, hi):
        rem = [p for v, p in tr if (lo is not None and v < lo) or (hi is not None and v >= hi)]
        kept = [p for v, p in tr if not ((lo is not None and v < lo) or (hi is not None and v >= hi))]
        rm, km = sum(rem) / len(rem), sum(kept) / len(kept)
        return {"n_removed": len(rem), "removed_mean": rm, "n_kept": len(kept), "kept_mean": km,
                "gap": rm - km, "score": len(rem) * (km - rm) / n, "lo": lo, "hi": hi}

    def run(cuts):
        best, seen = None, set()
        for lo, hi in cuts:
            if (lo, hi) in seen:
                continue
            seen.add((lo, hi))
            c = evaluate(lo, hi)
            if c["gap"] >= 0:
                break
            if best is None or c["score"] > best["score"]:
                best = c
        return best

    fams = {
        "left": [(left_x(p), None) for p in ONE_SIDED if left_x(p) is not None],
        "right": [(None, right_x(p)) for p in ONE_SIDED if right_x(p) is not None],
        "both": [(left_x(p), right_x(p)) for p in BOTH
                 if left_x(p) is not None and right_x(p) is not None and left_x(p) < right_x(p)],
    }
    best, side = None, None
    for name in ("left", "right", "both"):
        b = run(fams[name])
        if b is None:
            continue
        if best is None or b["score"] > best["score"] or (b["score"] == best["score"] and b["n_removed"] < best["n_removed"]):
            best, side = b, name
    if best is None:
        return zero
    return {"score": best["score"], "side": side, "n_removed": best["n_removed"], "removed_mean": best["removed_mean"],
            "n_kept": best["n_kept"], "kept_mean": best["kept_mean"], "lo": best["lo"], "hi": best["hi"]}


def same(fast: dict, ref: dict | None, tol: float = 1e-7) -> bool:
    """Does surface_stats.filter_gain's dict say what the reference says?"""
    if ref is None:
        return fast["filter_gain"] is None
    if ref["side"] is None:
        return fast["filter_gain"] == 0.0 and fast["fg_side"] is None

    def close(a, b):
        return a is None and b is None or (a is not None and b is not None and abs(a - b) <= tol * max(1.0, abs(b)))
    return (fast["fg_side"] == ref["side"] and fast["fg_n_removed"] == ref["n_removed"]
            and fast["fg_n_kept"] == ref["n_kept"] and close(fast["filter_gain"], ref["score"])
            and close(fast["fg_removed_mean"], ref["removed_mean"]) and close(fast["fg_kept_mean"], ref["kept_mean"])
            and fast["fg_lo"] == ref["lo"] and fast["fg_hi"] == ref["hi"])
