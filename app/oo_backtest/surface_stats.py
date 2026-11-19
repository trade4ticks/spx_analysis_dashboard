"""Correlation ranking of surface metrics against trade P/L. Pure: no I/O.

For each metric, over the trades that HAVE a value for it (pairwise -- a
metric whose coverage starts later is measured on fewer, later trades; the
page's "common coverage only" option is how that is controlled, by sending
only trades from a common start date):

  n            trades with a value
  bars         distinct entry bars among them -- trades entered on the same
               bar share one metric value, so this, not n, bounds how much
               independent information the correlation rests on. Reported,
               not corrected for.
  pearson      r and its two-sided p, as scipy.stats.pearsonr computes them
  spearman     rho and its two-sided p, as scipy.stats.spearmanr computes them
  *_p_bh       Benjamini-Hochberg adjusted p, over EVERY metric with a p for
               that method -- not over what a legend or form filter shows:
               hiding a family does not un-test it.

A metric with fewer than MIN_N values, or with every value identical (or
every P/L identical), has no correlation: r and p are None and it takes no
part in the BH adjustment.

VECTORISED BY NULL PATTERN. The first version called pearsonr and spearmanr
once per metric -- 2.5 s for 452 metrics x ~1,300 trades on the VPS. Metrics
that start on the same date are null on exactly the same trades, so columns
are grouped by their null mask (a handful of groups: one per coverage start,
plus the rows with no bar). Within a group the submatrix is dense, so:

  * pairwise dropping is exact by construction -- every metric in a group is
    measured on precisely the trades where it has a value;
  * P/L is ranked ONCE PER GROUP, over that group's trades only. Ranking P/L
    once over all trades would give every late-starting metric the wrong
    Spearman ranks -- the gate plants exactly that;
  * both correlations are one matrix operation per group, following scipy's
    own arithmetic (centre, scale by the max, norm, dot; ranks averaged over
    ties), and the p-values come from the same distributions scipy uses --
    the regularised incomplete beta for Pearson, Student's t for Spearman --
    evaluated on the whole vector at once.

check_oo_backtest compares every field against per-metric scipy calls.
"""
from __future__ import annotations

import math

import numpy as np
from scipy import special
from scipy.stats import rankdata

MIN_N = 3


def _none(a: np.ndarray) -> list:
    return [None if (v is None or not math.isfinite(v)) else float(v) for v in a.tolist()]


def _corr_dense(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Pearson r of each column of X (n x k, no NaN) with y (n,), in
    scipy.stats.pearsonr's order of operations."""
    xm = X - X.mean(axis=0)
    ym = y - y.mean()
    xmax = np.abs(xm).max(axis=0)
    ymax = np.abs(ym).max()
    with np.errstate(invalid="ignore", divide="ignore"):
        normx = xmax * np.linalg.norm(xm / xmax, axis=0)
        normy = ymax * np.linalg.norm(ym / ymax)
        r = (ym / normy) @ (xm / normx)
    return np.clip(r, -1.0, 1.0)


# At most this many trade x metric cells are materialised at once. A sweep is
# ~240,000 trades: the whole trades x 452 matrix is ~870 MB of float64 before
# the copies centring and ranking make, which is more than the VPS should
# spend on one request. A single log (~2,000 trades) is far below it, so it is
# computed in one block exactly as before.
BLOCK_CELLS = 4_000_000


def _ranks_by_count(values: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """rankdata's average ranks of the multiset where values[i] occurs
    counts[i] times, returned per DISTINCT row (gather to get per trade).

    The same numbers rankdata gives on the gathered column -- a value shared
    by c trades starting after s smaller ones ranks s + (c + 1) / 2, whether
    the ties come from one row or from several rows holding equal values --
    at the cost of sorting the distinct rows instead of every trade."""
    uniq, inv = np.unique(values, return_inverse=True)
    per = np.bincount(inv, weights=counts, minlength=len(uniq))
    before = np.concatenate(([0.0], np.cumsum(per)[:-1]))
    return (before + (per + 1.0) / 2.0)[inv]


def correlations(X: np.ndarray, pnl: np.ndarray, bar_ids: np.ndarray) -> dict[str, np.ndarray]:
    """Per column of X (trades x metrics, NaN where missing):
    n, bars, pearson, pearson_p, spearman, spearman_p -- arrays of length k,
    NaN where there is no correlation. pnl is (trades,), NaN where unusable;
    such a trade is in no metric's sample. bar_ids is (trades,) ints."""
    X = np.asarray(X, dtype=float)
    return correlations_gathered(X, np.arange(X.shape[0]), pnl, bar_ids)


def correlations_gathered(distinct: np.ndarray, gather: np.ndarray, pnl: np.ndarray,
                          bar_ids: np.ndarray) -> dict[str, np.ndarray]:
    """correlations() for X = distinct[gather], without building X.

    Trades entered on the same bar share that bar's metric values, so the
    matrix is held once per DISTINCT bar (a few thousand rows) and each trade
    points at its row. A sweep repeats every market day once per variant;
    this is what keeps 240,000 trades in memory the size of their bars."""
    distinct = np.asarray(distinct, dtype=float)
    gather = np.asarray(gather)
    pnl = np.asarray(pnl, dtype=float)
    keep = np.isfinite(pnl)
    gather, pnl, bar_ids = gather[keep], pnl[keep], np.asarray(bar_ids)[keep]
    k = distinct.shape[1]
    out = {name: np.full(k, np.nan) for name in ("pearson", "pearson_p", "spearman", "spearman_p")}
    out["n"] = np.zeros(k, dtype=int)
    out["bars"] = np.zeros(k, dtype=int)
    if not k or not len(pnl):
        return out

    # Two metrics are null on the same TRADES exactly when they are null on
    # the same distinct rows that some trade points at.
    valid_d = np.isfinite(distinct)                           # distinct rows x metrics
    used = np.unique(gather)
    packed = np.packbits(valid_d[used], axis=0)               # one column of bytes per metric
    groups: dict[bytes, list[int]] = {}
    for j in range(k):
        groups.setdefault(packed[:, j].tobytes(), []).append(j)

    for cols in groups.values():
        rows = np.flatnonzero(valid_d[gather, cols[0]])
        n = len(rows)
        out["n"][cols] = n
        out["bars"][cols] = len(np.unique(bar_ids[rows]))
        if n < MIN_N:
            continue
        y = pnl[rows]
        if np.all(y == y[0]):
            continue
        g = gather[rows]
        ry = rankdata(y)                                      # P/L ranked ONCE PER GROUP
        ab = n / 2.0 - 1.0
        dof = n - 2.0
        # Ranks from the distinct rows when trades repeat them (a sweep);
        # rankdata on the trades themselves otherwise, as it always was.
        hits = np.bincount(g, minlength=distinct.shape[0]).astype(float)
        on = np.flatnonzero(hits)
        by_count = len(on) < n
        width = max(1, BLOCK_CELLS // n)
        for a in range(0, len(cols), width):
            block = cols[a:a + width]
            Xg = distinct[:, block][g]
            live = ~np.all(Xg == Xg[0], axis=0)               # a constant metric has no correlation
            if not live.any():
                continue
            cols_live = [c for c, ok in zip(block, live) if ok]
            Xg = Xg[:, live]

            r = _corr_dense(Xg, y)
            out["pearson"][cols_live] = r
            out["pearson_p"][cols_live] = 2.0 * special.betaincc(ab, ab, (np.abs(r) + 1.0) / 2.0)

            if by_count:
                R = np.zeros((distinct.shape[0], len(cols_live)))
                for j, c in enumerate(cols_live):
                    R[on, j] = _ranks_by_count(distinct[on, c], hits[on])
                # Column-major, as rankdata(axis=0) returns it: numpy sums a
                # column in an order set by the layout, and the same layout
                # keeps rho bit-identical to the rankdata path.
                Xr = np.asfortranarray(R[g])
            else:
                Xr = rankdata(Xg, axis=0)
            rho = _corr_dense(Xr, ry)
            with np.errstate(divide="ignore", invalid="ignore"):
                t = rho * np.sqrt((dof / ((rho + 1.0) * (1.0 - rho))).clip(0))
            out["spearman"][cols_live] = rho
            out["spearman_p"][cols_live] = 2.0 * special.stdtr(dof, -np.abs(t))
    return out


def correlate(values: list, pnl: list, bars: list) -> dict:
    """One metric, for callers and checks that have lists: values[i] (None
    where missing), pnl[i] (None where unusable), bars[i] (hashable)."""
    ids = {b: i for i, b in enumerate(dict.fromkeys(bars))}
    res = correlations(np.array([[np.nan if v is None else v] for v in values], dtype=float),
                       np.array([np.nan if p is None else p for p in pnl], dtype=float),
                       np.array([ids[b] for b in bars]))
    return {"n": int(res["n"][0]), "bars": int(res["bars"][0]),
            **{f: _none(res[f])[0] for f in ("pearson", "pearson_p", "spearman", "spearman_p")}}


def benjamini_hochberg(pvalues: list) -> list:
    """Adjusted p-values aligned to the input; None stays None and is not
    counted in m. Sort ascending, p_(k) * m / k, then a running minimum from
    the largest rank down so the adjusted values stay monotone, capped at 1."""
    idx = [i for i, p in enumerate(pvalues) if p is not None]
    m = len(idx)
    out = [None] * len(pvalues)
    if not m:
        return out
    order = sorted(idx, key=lambda i: pvalues[i])
    running = 1.0
    for rank in range(m, 0, -1):
        i = order[rank - 1]
        running = min(running, pvalues[i] * m / rank)
        out[i] = min(running, 1.0)
    return out


def stat_rows(X: np.ndarray, pnl: list, bar_ids, gather=None) -> list[dict]:
    """Per column of X: n, bars, Pearson / Spearman r and p, and the filter
    gain fields (filter_gains) -- everything rank() reports but the BH
    adjustment, which belongs to whatever FAMILY of tests the caller
    defines (one per page section). X is trades x columns, NaN where
    missing, or with `gather` the DISTINCT rows each trade points at."""
    p = np.array([np.nan if v is None else v for v in pnl], dtype=float)
    X = np.asarray(X, dtype=float)
    g = np.arange(X.shape[0]) if gather is None else np.asarray(gather)
    res = (correlations(X, p, np.asarray(bar_ids)) if gather is None
           else correlations_gathered(X, g, p, np.asarray(bar_ids)))
    cols = {f: _none(res[f]) for f in ("pearson", "pearson_p", "spearman", "spearman_p")}
    fg = filter_gains(X, g, p)
    return [{"n": int(res["n"][j]), "bars": int(res["bars"][j]), **{f: cols[f][j] for f in cols}, **fg[j]}
            for j in range(X.shape[1])]


def with_bh(rows: list[dict]) -> list[dict]:
    """Benjamini-Hochberg over exactly these rows, per method, in place."""
    for method in ("pearson", "spearman"):
        adj = benjamini_hochberg([r[f"{method}_p"] for r in rows])
        for r, a in zip(rows, adj):
            r[f"{method}_p_bh"] = a
    return rows


def rank(metrics: list[dict], X: np.ndarray, pnl: list, bar_ids, gather=None) -> list[dict]:
    """One result per metric, in catalog order (the page sorts). X is
    trades x len(metrics), NaN where missing, columns in `metrics` order --
    or, with `gather`, the DISTINCT rows that each trade's gather points at.
    BH over every metric passed (the surface metrics are one family)."""
    stats = stat_rows(X, pnl, bar_ids, gather)
    out = [{"column": m["column_name"], "family": m["family"], "tenor": m["tenor"], "wing": m["wing"],
            "form": m["form"], **st} for m, st in zip(metrics, stats)]
    return with_bh(out)


# ── filter gain (2026-11-18, the user's spec) ───────────────────────────────
#
# Pearson and Spearman score the whole range; what the user acts on is a
# CUTTABLE EDGE -- a contiguous block at one or both ends of the metric whose
# trades are worse than the rest. Filter gain is how much the average P/L per
# trade rises if that block is not traded:
#
#   improvement = n_removed * (kept_mean - removed_mean) / n_total   ($/trade)
#
# over twenty candidate cuts: left and right 5..40 % (step 5), both-sides
# 5..20 % from EACH end.
#
# THE TWO RULES BELOW LOOK ARBITRARY AND ARE NOT. Both are the user's, both
# exist because the score without them prefers something that is not a
# filter, and check_filter_gain fails if either goes. Do not "improve" them.
#
#   THE 40 % CAP (FG_ONE_SIDED stops at 40, FG_BOTH at 20 from each end) is
#   the ONLY thing stopping the score preferring "remove 90 % of the trades":
#   removing nearly everything always maximises the remaining average, and
#   that is never a filter. It is not a tuning parameter -- raising it does
#   not make the ranking more sensitive, it makes the top of it the metrics
#   whose best trades happen to sit in one corner.
#
#   THE SIGN-FLIP WALK enforces "a contiguous block, starting at the edge".
#   Without it, a metric whose OUTERMOST bin is GOOD but whose next few are
#   bad scores well on a deeper cut, once the good bin is averaged away
#   inside the removed block. That shape is real (the user has an example)
#   and is not a trade the user would take: the rule "skip value < X" would
#   throw out the best trades with the bad ones. So each family is walked
#   from its shallowest cut and stops at the first cut whose removed block
#   is not worse than what is kept.
#
# Rules, each the user's:
#   * Pairwise: only trades with a value (and a usable P/L); percentages are
#     of THAT population. Below MIN_N there is no score (None).
#   * A cut is at a VALUE boundary -- tied trades fall on the same side, as
#     "metric < X" does. A nominal p % takes the deepest boundary removing AT
#     MOST p % of the trades (so the cap holds exactly); the actual share is
#     what is reported. Depths that land on the same cut are one cut; a cut
#     that removes nothing is not a cut. A both-sides cut needs a block at
#     EACH end.
#   * Sign flip: walk each family from its shallowest cut inward and keep
#     only the contiguous run where gap = removed_mean - kept_mean < 0;
#     stop at the first gap >= 0. If the shallowest is already >= 0 the
#     family scores 0. (A good outermost bin in front of bad ones is not
#     filterable, however good a deeper cut averages.)
#   * Family score = the largest improvement in its run; metric score = the
#     largest of left / right / both. Ties prefer the SHALLOWER cut (fewer
#     trades removed), then left, right, both.
#   * Dollars per trade: comparable within one backtest, not across them.
#
# Computed per DISTINCT entry bar: trades sharing a bar share every metric
# value, so the counts and P/L sums per bar are taken once per request and
# each metric sorts its few thousand bars, never a sweep's 300,000 trades.

FG_ONE_SIDED = (5, 10, 15, 20, 25, 30, 35, 40)
FG_BOTH = (5, 10, 15, 20)
FG_FIELDS = ("filter_gain", "fg_side", "fg_removed_pct", "fg_n_removed", "fg_removed_mean", "fg_n_kept",
             "fg_kept_mean", "fg_lo", "fg_hi")


def _fg_empty(score):
    return {"filter_gain": score, "fg_side": None, "fg_removed_pct": None, "fg_n_removed": None,
            "fg_removed_mean": None, "fg_n_kept": None, "fg_kept_mean": None, "fg_lo": None, "fg_hi": None}


def filter_gain(values: np.ndarray, counts: np.ndarray, sums: np.ndarray) -> dict:
    """Filter gain of one metric. values[i] is a metric value held by
    counts[i] trades whose P/L sums to sums[i] (values may repeat). Returns
    FG_FIELDS: filter_gain ($/trade; 0.0 when no cut qualifies, None below
    MIN_N trades), fg_side left|right|both, fg_removed_pct (ACTUAL % of the
    trades with a value), fg_n_removed / fg_removed_mean, fg_n_kept /
    fg_kept_mean, and the cut as a rule: skip a trade whose value is
    < fg_lo, or >= fg_hi (None where that side is not cut)."""
    counts = np.asarray(counts, dtype=np.int64)
    n = int(counts.sum())
    if n < MIN_N:
        return _fg_empty(None)
    u, inv = np.unique(np.asarray(values, dtype=float), return_inverse=True)
    c = np.bincount(inv, weights=counts).round().astype(np.int64)
    sm = np.bincount(inv, weights=np.asarray(sums, dtype=float))
    total = float(sm.sum())
    k = len(u)
    if k < 2:
        return _fg_empty(0.0)
    # Removing u[0..i] removes cl[i] trades with P/L sl[i]; removing the top
    # j+1 values removes cr[j] / sr[j]. Index -1 = nothing from that end; a
    # padded 0 at the front of each makes index -1 read as zero.
    cl, sl = np.concatenate(([0], np.cumsum(c))), np.concatenate(([0.0], np.cumsum(sm)))
    cr, sr = np.concatenate(([0], np.cumsum(c[::-1]))), np.concatenate(([0.0], np.cumsum(sm[::-1])))
    # The deepest boundary removing AT MOST p % (integer arithmetic: cl * 100
    # <= p * n), never every value: capped at k - 2.
    one = np.array(FG_ONE_SIDED)
    two = np.array(FG_BOTH)
    lo1 = np.minimum(np.searchsorted(cl[1:] * 100, one * n, side="right") - 1, k - 2)
    hi1 = np.minimum(np.searchsorted(cr[1:] * 100, one * n, side="right") - 1, k - 2)
    lo2 = np.minimum(np.searchsorted(cl[1:] * 100, two * n, side="right") - 1, k - 2)
    hi2 = np.minimum(np.searchsorted(cr[1:] * 100, two * n, side="right") - 1, k - 2)

    def family(i, j):
        """The family's distinct cuts in depth order -- i and j only grow
        with the depth, so a repeat is always the one before -- and the best
        of the contiguous run with gap < 0 from the shallowest (argmax: the
        first maximum, i.e. the shallower on a tie). None when no run."""
        if not len(i):
            return None
        keep = np.ones(len(i), dtype=bool)
        keep[1:] = (i[1:] != i[:-1]) | (j[1:] != j[:-1])
        i, j = i[keep], j[keep]
        nr = cl[i + 1] + cr[j + 1]
        s_r = sl[i + 1] + sr[j + 1]
        nk = n - nr
        rm, km = s_r / nr, (total - s_r) / nk
        gap = rm - km
        stop = np.flatnonzero(gap >= 0)
        run = stop[0] if len(stop) else len(gap)
        if run == 0:
            return None
        imp = nr[:run] * (km[:run] - rm[:run]) / n
        b = int(np.argmax(imp))
        return {"imp": float(imp[b]), "n_removed": int(nr[b]), "removed_mean": float(rm[b]), "n_kept": int(nk[b]),
                "kept_mean": float(km[b]), "lo": float(u[i[b] + 1]) if i[b] >= 0 else None,
                "hi": float(u[k - 1 - j[b]]) if j[b] >= 0 else None}

    m1 = lo1 >= 0
    m2 = hi1 >= 0
    mb = (lo2 >= 0) & (hi2 >= 0) & (lo2 + 1 < k - 1 - hi2)   # a block at each end, something kept between
    best, side = None, None
    for name, b in (("left", family(lo1[m1], np.full(int(m1.sum()), -1))),
                    ("right", family(np.full(int(m2.sum()), -1), hi1[m2])),
                    ("both", family(lo2[mb], hi2[mb]))):
        if b is None:
            continue
        if (best is None or b["imp"] > best["imp"]
                or (b["imp"] == best["imp"] and b["n_removed"] < best["n_removed"])):
            best, side = b, name
    if best is None:
        return _fg_empty(0.0)
    return {"filter_gain": best["imp"], "fg_side": side,
            "fg_removed_pct": 100.0 * best["n_removed"] / n, "fg_n_removed": best["n_removed"],
            "fg_removed_mean": best["removed_mean"], "fg_n_kept": best["n_kept"],
            "fg_kept_mean": best["kept_mean"], "fg_lo": best["lo"], "fg_hi": best["hi"]}


def filter_gains(distinct: np.ndarray, gather: np.ndarray, pnl: np.ndarray) -> list[dict]:
    """filter_gain per column of `distinct` (distinct rows x metrics), the
    trades being gather[t] -> row with P/L pnl[t] (NaN: unusable, in no
    sample). Counts and P/L sums per distinct row are taken once."""
    distinct = np.asarray(distinct, dtype=float)
    gather, pnl = np.asarray(gather), np.asarray(pnl, dtype=float)
    keep = np.isfinite(pnl)
    g, p = gather[keep], pnl[keep]
    hits = np.bincount(g, minlength=distinct.shape[0])
    psum = np.bincount(g, weights=p, minlength=distinct.shape[0])
    on = hits > 0
    out = []
    for j in range(distinct.shape[1]):
        col = distinct[:, j]
        r = on & np.isfinite(col)
        out.append(filter_gain(col[r], hits[r], psum[r]))
    return out
