"""Which symbols the scan holds, seeded from the pipeline's own universe.

ONE QUERY, ONE PLACE. It was two -- measure_scan_capacity and check_scan_live
each had their own `symbols_from_db`, and both were wrong in the same way
because there was nothing to fix once. That is the whole argument for this
file existing rather than a helper in each script.

--- The exclusion is READ, not re-derived -----------------------------------

`universe.spread_excluded` is the pipeline's decision that a name's spread is
too tight to scalp -- DEFAULT_FILTERS['min_spread_cents'] is 5, and a name
under it has nothing to capture however much it trades.

That column is TAKEN AS GIVEN. It is tempting to re-derive it here from
`spread_bps` and the close, and it would be wrong: on the 2026-09-06 universe
MSFT is excluded at 2.22 bps of $499 (11 cents) while GOOGL is NOT excluded at
1.06 bps of $338 (3.6 cents), so whatever the pipeline is thresholding, it is
not the product of those two columns. Reimplementing it here would produce a
second, quietly different universe -- which is the same class of mistake as a
second copy of quiet.py, arrived at through a filter instead of a function.

--- Why not order by dollar volume ------------------------------------------

Because that is what put SPY, QQQ and VYM at the top of a 600-symbol capacity
run: index ETFs at 0.3 bps that the spread floor already excludes, ordered
first for having the largest notional. Notional is not what the scan is
looking for and it is not what loads the socket either.

`trades` is the default instead. It is the count of prints, so it orders by
the thing that actually costs -- messages on the wire, and records in a ring --
and it does not reward a name for being expensive.

--- The filtered universe is smaller than the capacity run assumed ----------

On 2026-09-06: 740 symbols, 512 qualified, 639 not spread-excluded, and 434
both. So a scan holding "everything tradeable" is ~430 names, not 600. The
600-symbol capacity figures stand -- they included ~166 names the page would
never hold, among them SPY, QQQ, NVDA, AAPL and MSFT, which are some of the
heaviest on the tape -- but they are an OVERSTATEMENT of the load rather than
a measurement of it.
"""
from __future__ import annotations

import os
from urllib.parse import urlsplit, urlunsplit

# The brief's two seed thresholds, defaulted OFF. A threshold that filters by
# default is one nobody chose; these are inputs the page exposes, and zero
# means "every qualified, non-excluded name".
DEFAULT_MIN_RANGE_CENTS = 0.0
DEFAULT_MIN_DOLLAR_PER_MIN = 0.0

# The two metrics the brief names, at the pipeline's primary window.
RANGE_METRIC = "quiet_range_p10p90_cents_60s"
DOLLAR_METRIC = "quiet_dollar_vol_per_min_60s"

ORDERS = {
    # Prints, not notional. See the note above.
    "trades": "m_ord.value desc nulls last",
    # Widest capture first -- the page's own preference, once it has one.
    "range": "m_range.value desc nulls last",
    "dollars": "m_dollar.value desc nulls last",
}


def scalp_dsn() -> str:
    dsn = os.getenv("SCALP_DATABASE_URL")
    if dsn:
        return dsn
    parts = urlsplit(os.environ["DATABASE_URL"])
    return urlunsplit(parts._replace(path="/equities_scalp"))


_SQL = """
with u as (
    select symbol
    from universe
    where trade_date = (select max(trade_date) from universe)
      and ({qualified})
      and not spread_excluded
),
d as (select max(trade_date) as td from daily_metrics)
select u.symbol,
       m_range.value  as range_cents,
       m_dollar.value as dollar_per_min,
       m_ord.value    as trades
from u
join d on true
join daily_metrics m_range
     on m_range.symbol = u.symbol and m_range.trade_date = d.td
    and m_range.metric = $1
join daily_metrics m_dollar
     on m_dollar.symbol = u.symbol and m_dollar.trade_date = d.td
    and m_dollar.metric = $2
join daily_metrics m_ord
     on m_ord.symbol = u.symbol and m_ord.trade_date = d.td
    and m_ord.metric = 'trades'
where m_range.value >= $3 and m_dollar.value >= $4
order by {order}
limit $5
"""


async def seed(con, *, limit: int = 600,
               min_range_cents: float = DEFAULT_MIN_RANGE_CENTS,
               min_dollar_per_min: float = DEFAULT_MIN_DOLLAR_PER_MIN,
               order: str = "trades",
               qualified_only: bool = True) -> list[dict]:
    """Rows of {symbol, range_cents, dollar_per_min, trades}, best first.

    The metrics come back with the symbols rather than being looked up again
    later: the page shows why a name is on the list, and a seed that returns
    bare tickers forces a second query to say so.
    """
    if order not in ORDERS:
        raise ValueError(f"order must be one of {sorted(ORDERS)}, not {order!r}")
    sql = _SQL.format(order=ORDERS[order],
                      qualified="qualified" if qualified_only else "true")
    rows = await con.fetch(sql, RANGE_METRIC, DOLLAR_METRIC,
                           float(min_range_cents), float(min_dollar_per_min),
                           int(limit))
    return [dict(r) for r in rows]


async def counts(con) -> dict:
    """How many symbols each filter leaves, for a run to report.

    THE FILTER HAS TO BE VISIBLE IN THE OUTPUT. A dropped `not
    spread_excluded` does not fail anything -- it silently puts SPY and QQQ
    back at the top of the list, which is exactly what happened and was
    caught by a person reading symbol names rather than by anything here.
    Printing the counts every run makes the filter's effect impossible to
    miss.
    """
    q = """select count(*) filter (where true)                       as total,
                  count(*) filter (where qualified)                  as qualified,
                  count(*) filter (where not spread_excluded)        as not_excluded,
                  count(*) filter (where qualified
                                     and not spread_excluded)        as both
           from universe
           where trade_date = (select max(trade_date) from universe)"""
    return dict(await con.fetchrow(q))
