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
  pearson      r and its two-sided p (scipy.stats.pearsonr)
  spearman     rho and its two-sided p (scipy.stats.spearmanr)
  *_p_bh       Benjamini-Hochberg adjusted p, over EVERY metric with a p for
               that method -- not over what a legend or form filter shows:
               hiding a family does not un-test it.

A metric with fewer than MIN_N values, or with every value identical (or
every P/L identical), has no correlation: its r and p are None and it takes
no part in the BH adjustment.
"""
from __future__ import annotations

import math
import warnings

import numpy as np
from scipy import stats

MIN_N = 3


def _clean(x) -> float | None:
    if x is None:
        return None
    x = float(x)
    return None if math.isnan(x) or math.isinf(x) else x


def correlate(values: list, pnl: list, bars: list) -> dict:
    """values[i] (None where missing), pnl[i], bars[i] (a hashable entry-bar key)."""
    keep = [i for i, v in enumerate(values) if v is not None and pnl[i] is not None]
    n = len(keep)
    out = {"n": n, "bars": len({bars[i] for i in keep}),
           "pearson": None, "pearson_p": None, "spearman": None, "spearman_p": None}
    if n < MIN_N:
        return out
    x = np.fromiter((values[i] for i in keep), float, n)
    y = np.fromiter((pnl[i] for i in keep), float, n)
    if np.ptp(x) == 0 or np.ptp(y) == 0:
        return out
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")        # constant-input warnings are handled above
        pr = stats.pearsonr(x, y)
        sr = stats.spearmanr(x, y)
    out.update(pearson=_clean(pr.statistic), pearson_p=_clean(pr.pvalue),
               spearman=_clean(sr.statistic), spearman_p=_clean(sr.pvalue))
    return out


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


def rank(metrics: list[dict], rows: list[dict], pnl: list, bar_keys: list) -> list[dict]:
    """One result per metric (catalog order kept; the page sorts). `rows` are
    entry_values() output aligned with `pnl` and `bar_keys`."""
    out = []
    for m in metrics:
        c = m["column_name"]
        r = correlate([row[c] for row in rows], pnl, bar_keys)
        out.append({"column": c, "family": m["family"], "tenor": m["tenor"], "wing": m["wing"],
                    "form": m["form"], **r})
    for method in ("pearson", "spearman"):
        adj = benjamini_hochberg([r[f"{method}_p"] for r in out])
        for r, a in zip(out, adj):
            r[f"{method}_p_bh"] = a
    return out
