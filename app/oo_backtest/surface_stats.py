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


def rank(metrics: list[dict], X: np.ndarray, pnl: list, bar_ids, gather=None) -> list[dict]:
    """One result per metric, in catalog order (the page sorts). X is
    trades x len(metrics), NaN where missing, columns in `metrics` order --
    or, with `gather`, the DISTINCT rows that each trade's gather points at."""
    p = np.array([np.nan if v is None else v for v in pnl], dtype=float)
    res = (correlations(X, p, np.asarray(bar_ids)) if gather is None
           else correlations_gathered(X, gather, p, np.asarray(bar_ids)))
    cols = {f: _none(res[f]) for f in ("pearson", "pearson_p", "spearman", "spearman_p")}
    out = [{"column": m["column_name"], "family": m["family"], "tenor": m["tenor"], "wing": m["wing"],
            "form": m["form"], "n": int(res["n"][j]), "bars": int(res["bars"][j]),
            **{f: cols[f][j] for f in cols}} for j, m in enumerate(metrics)]
    for method in ("pearson", "spearman"):
        adj = benjamini_hochberg([r[f"{method}_p"] for r in out])
        for r, a in zip(out, adj):
            r[f"{method}_p_bh"] = a
    return out
