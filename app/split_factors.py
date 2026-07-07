"""
Split-adjustment factors for raw option strikes/counts.

VENDORED from Open_Interest/lib/split_factors.py on 2026-07-07.
Source of truth lives there — keep in sync. `make_split_factors` and
`make_split_factor_map` are copied VERBATIM (they are DB-free). The source's
`load_splits` (a psycopg2 read_sql_df call) is NOT copied here; this project
loads the same underlying_ohlc.splits via asyncpg in the caller
(app/routers/ticker_chain.py) and passes the resulting DataFrame in.

Convention (must stay identical to build_features, which the chain parquet
stores are written for):
    adjusted_strike = raw_strike * adj_factor
    adjusted_count  = raw_count  / adj_factor      (count = OI or volume)

adj_factor for a session = product of (1/ratio) over all splits ON OR AFTER
that session. The bisect_left boundary means trade_date == split_date is
treated as PRE-split (adjusted) — load-bearing and tied to the ~1-day OI
publication lag these parquet stores carry; both projects must agree on it.
Do NOT switch to bisect_right.
"""
from __future__ import annotations

import bisect

import pandas as pd


def make_split_factors(splits_df: pd.DataFrame, dates: list) -> pd.DataFrame:
    """Return DataFrame(trade_date, adj_factor) for each date in `dates`.

    adj_factor = product of (1/ratio) for all splits with split_date >= date.
    Multiply raw strikes by adj_factor to get split-adjusted strikes that
    align with yfinance-adjusted spot prices.

    Boundary convention: trade_date <= split_date -> adjust (split affects this date);
                         trade_date >  split_date -> no adjustment (already past it).
    Handles forward (ratio>1) and reverse (ratio<1) splits uniformly.
    Tickers with no splits get adj_factor = 1.0 for every date (no-op).
    """
    if splits_df.empty:
        return pd.DataFrame({"trade_date": dates,
                             "adj_factor":  [1.0] * len(dates)})

    split_dates  = splits_df["trade_date"].tolist()   # sorted asc by query
    split_ratios = splits_df["splits"].tolist()

    # Suffix cumulative product: suffix_factors[i] = prod(1/ratio for splits[i:])
    n = len(split_dates)
    suffix_factors = [1.0] * (n + 1)
    for i in range(n - 1, -1, -1):
        suffix_factors[i] = suffix_factors[i + 1] / split_ratios[i]

    # bisect_left: split-date row falls in the PRE-split bucket (adjusted).
    factors = [suffix_factors[bisect.bisect_left(split_dates, td)] for td in dates]
    return pd.DataFrame({"trade_date": dates, "adj_factor": factors})


def make_split_factor_map(splits_df: pd.DataFrame, dates: list) -> dict:
    """Convenience: same as make_split_factors but returns {date: factor} dict."""
    df = make_split_factors(splits_df, dates)
    return dict(zip(df["trade_date"], df["adj_factor"]))
