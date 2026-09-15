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


def correlations(X: np.ndarray, pnl: np.ndarray, bar_ids: np.ndarray) -> dict[str, np.ndarray]:
    """Per column of X (trades x metrics, NaN where missing):
    n, bars, pearson, pearson_p, spearman, spearman_p -- arrays of length k,
    NaN where there is no correlation. pnl is (trades,), NaN where unusable;
    such a trade is in no metric's sample. bar_ids is (trades,) ints."""
    X = np.asarray(X, dtype=float)
    pnl = np.asarray(pnl, dtype=float)
    keep = np.isfinite(pnl)
    X, pnl, bar_ids = X[keep], pnl[keep], np.asarray(bar_ids)[keep]
    k = X.shape[1]
    out = {name: np.full(k, np.nan) for name in ("pearson", "pearson_p", "spearman", "spearman_p")}
    out["n"] = np.zeros(k, dtype=int)
    out["bars"] = np.zeros(k, dtype=int)
    if not k or not len(pnl):
        return out

    valid = np.isfinite(X)                                    # trades x metrics
    packed = np.packbits(valid, axis=0)                       # one column of bytes per metric
    groups: dict[bytes, list[int]] = {}
    for j in range(k):
        groups.setdefault(packed[:, j].tobytes(), []).append(j)

    for cols in groups.values():
        rows = np.flatnonzero(valid[:, cols[0]])
        n = len(rows)
        out["n"][cols] = n
        out["bars"][cols] = len(np.unique(bar_ids[rows]))
        if n < MIN_N:
            continue
        y = pnl[rows]
        if np.all(y == y[0]):
            continue
        Xg = X[np.ix_(rows, cols)]
        live = ~np.all(Xg == Xg[0], axis=0)                   # a constant metric has no correlation
        if not live.any():
            continue
        cols_live = [c for c, ok in zip(cols, live) if ok]
        Xg = Xg[:, live]

        r = _corr_dense(Xg, y)
        ab = n / 2.0 - 1.0
        out["pearson"][cols_live] = r
        out["pearson_p"][cols_live] = 2.0 * special.betaincc(ab, ab, (np.abs(r) + 1.0) / 2.0)

        rho = _corr_dense(rankdata(Xg, axis=0), rankdata(y))
        dof = n - 2.0
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


def rank(metrics: list[dict], X: np.ndarray, pnl: list, bar_ids) -> list[dict]:
    """One result per metric, in catalog order (the page sorts). X is
    trades x len(metrics), NaN where missing, columns in `metrics` order."""
    res = correlations(X, np.array([np.nan if p is None else p for p in pnl], dtype=float), np.asarray(bar_ids))
    cols = {f: _none(res[f]) for f in ("pearson", "pearson_p", "spearman", "spearman_p")}
    out = [{"column": m["column_name"], "family": m["family"], "tenor": m["tenor"], "wing": m["wing"],
            "form": m["form"], "n": int(res["n"][j]), "bars": int(res["bars"][j]),
            **{f: cols[f][j] for f in cols}} for j, m in enumerate(metrics)]
    for method in ("pearson", "spearman"):
        adj = benjamini_hochberg([r[f"{method}_p"] for r in out])
        for r, a in zip(out, adj):
            r[f"{method}_p_bh"] = a
    return out
