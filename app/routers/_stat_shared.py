"""Stat-bar quantities shared by the exits page and the portfolio page.

Both pages render the SAME stat bar, so the numbers behind it are defined
here once. A second implementation of a stat is a second definition of it,
and the two drift silently -- the bar keeps rendering, the label keeps
saying the same word, and only the value disagrees.

The client-side half of this contract lives in
static/js/factor_charts.js: FactorCharts.statRowValues(), which formats
these fields and derives the dollar figures. Keep the field NAMES here
identical to the ones it reads.
"""
from __future__ import annotations


def effective_tickers(counts) -> float:
    """Participation ratio: (sum n)^2 / sum(n^2).

    Reads ~N when N tickers contribute equally and ~k when k dominate, so it
    says at a glance whether an edge is broad or concentrated. Same form as
    an effective-N / inverse-Herfindahl.

    Moved here from factor_trades.py so the portfolio bar reports the same
    quantity rather than a lookalike.
    """
    c = [x for x in counts if x > 0]
    if not c:
        return 0.0
    tot = float(sum(c))
    return (tot * tot) / float(sum(x * x for x in c))


def time_in_market(trade_dates, trading_days, horizon: int):
    """Share of trading days on which at least one position was open.

    NOT a replacement for Avg DIT, and deliberately not called one. Avg DIT
    is how long the average trade is held -- on a page whose signals use
    fixed forward-return horizons that is a constant equal to the horizon,
    which restates the outcome name rather than measuring anything. This
    answers a different question: across the period this portfolio was
    active, how much of it had capital deployed.

    That is the question a portfolio is assembled to answer. It moves when a
    signal is added, and it moves for the right reason -- a signal that fires
    in regimes the others miss raises it, one that piles onto the same dates
    does not.

    Each trade occupies `horizon` trading days from its entry. Occupancy is a
    SET UNION over trading-day indices, so two signals firing the same day,
    or overlapping holds, count once -- the same dedup the equity curve
    applies. The denominator runs from the first entry to `horizon` days past
    the last, i.e. the span over which the portfolio could have held
    something.

    Returns a fraction in 0..1, or None when it cannot be computed (no
    trades, no calendar, or no trade date found on the calendar).
    """
    if not trade_dates or not trading_days or horizon <= 0:
        return None
    idx = {d: i for i, d in enumerate(trading_days)}
    entries = [idx[d] for d in trade_dates if d in idx]
    if not entries:
        return None

    occupied: set = set()
    for i in entries:
        occupied.update(range(i, min(i + horizon, len(trading_days))))

    first = min(entries)
    last = min(max(entries) + horizon, len(trading_days))
    span = last - first
    if span <= 0:
        return None
    return len(occupied) / float(span)


def canonical_stat_names(s: dict) -> dict:
    """Add the field names the shared client mapper reads.

    The two endpoints grew different vocabularies for the same quantities
    (std vs std_dev, n_winners vs n_win, ...). Rather than teach the mapper
    two vocabularies -- which is the same drift in a smaller box -- the
    canonical names are added here. The originals are left in place so
    existing consumers are untouched.
    """
    out = dict(s)
    for canon, legacy in (("std_dev",  "std"),
                          ("n_win",    "n_winners"),
                          ("avg_win",  "avg_winners"),
                          ("avg_loss", "avg_losers")):
        if canon not in out and legacy in out:
            out[canon] = out[legacy]
    return out
