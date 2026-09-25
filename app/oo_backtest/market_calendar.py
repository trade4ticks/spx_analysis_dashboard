"""NYSE sessions, from the exchange calendar rather than from our own data.

THE CALENDAR DECIDES WHETHER A DAY WAS A SESSION. The data decides whether a
series has usable values on it. Those are two different questions and this
module answers only the first.

WHY THIS EXISTS, in the words of the day that produced it. 2026-04-08 was an
ordinary trading day. `index_ohlc` holds a full VIX session for it and zero
SPX bars, and the old rule -- a series has a session when it has at least 34
valid bars -- therefore called it "not an SPX session". That is not a
non-trading day, it is a MISSING DAY in the upstream pipeline, and reporting
it as the former hid it. The calendar is more reliable than the table it was
being inferred from, so the calendar is now the authority and the bar count
becomes a completeness check ON a day already known to be a session.

This reverses an earlier decision recorded in market.py and CLAUDE.md ("No
trading calendar, deliberately"), which was taken to avoid a calendar
disagreeing with the table on days like 2026-04-08. It disagreed because the
table was wrong. Do not restore the data-derived rule as a cleanup.

EXPECTED BARS COME FROM THE DAY'S OWN CLOSE, which is the other thing the
calendar buys: 09:30-16:00 is 78 five-minute bars and a 13:00 early close is
42, so a short half-day is now detectable instead of being waved through by a
single floor tuned for full days.

OFFLINE. pandas_market_calendars ships the exchange rules in the package;
nothing here touches the network. A missing package is a hard ImportError on
purpose -- falling back to the old rule would restore exactly the hiding this
removes.
"""
from __future__ import annotations

from datetime import date
from functools import lru_cache

import pandas_market_calendars as mcal

# SPX cash is calculated over NYSE hours and CBOE's index holidays match it on
# every scheduled closure; the two differ only on rare one-off events, which
# is not worth carrying a second calendar for (decided with the user,
# 2026-09-25).
MARKET = "NYSE"

# index_ohlc is five-minute bars labeled by START time, so a session running
# 09:30-16:00 has its last bar at 15:55 and holds 390/5 = 78 of them.
BAR_MINUTES = 5

# HOW MANY BARS A SERIES MAY BE SHORT and still count as complete for that
# day. Not a round number: the ingestion's own behaviour sets it. market.py
# records that VIX3M and VIX9D can arrive a bar or two behind, so a real
# session whose first bar has not landed yet must still keep its close --
# only its own gap goes null. Anything short by more than this is reported as
# a partial day rather than silently accepted, per series and with the count,
# so "SPX was 3 bars short on 12 sessions" is a thing you can read.
BAR_SHORTFALL_TOLERANCE = 2

_TZ = "America/New_York"


@lru_cache(maxsize=16)
def _sched(start: str, end: str) -> tuple[tuple[str, int, str, str], ...]:
    """(iso, expected_bars, open HH:MM, close HH:MM) per session, ascending.

    Cached because the page asks for the same span repeatedly and building a
    decade of schedule is milliseconds we need not spend twice. The cache
    holds tuples, not dicts, so a caller cannot mutate what the next caller
    receives.
    """
    cal = mcal.get_calendar(MARKET)
    sch = cal.schedule(start_date=start, end_date=end)
    out = []
    for day, row in sch.iterrows():
        opened, closed = row["market_open"], row["market_close"]
        minutes = int((closed - opened).total_seconds() // 60)
        out.append((
            day.strftime("%Y-%m-%d"),
            max(0, minutes // BAR_MINUTES),
            opened.tz_convert(_TZ).strftime("%H:%M"),
            closed.tz_convert(_TZ).strftime("%H:%M"),
        ))
    return tuple(out)


def _iso(d) -> str:
    return d if isinstance(d, str) else d.isoformat()


def sessions(start, end) -> list[str]:
    """ISO dates the exchange was open, inclusive, ascending."""
    return [r[0] for r in _sched(_iso(start), _iso(end))]


def day_table(start, end) -> dict[str, dict]:
    """{iso: {expected_bars, open, close, early}} for each session in range.

    A fresh dict every call: this is what the rollup joins against and it has
    no business sharing a structure with the cache.
    """
    out = {}
    for iso, bars, opened, closed in _sched(_iso(start), _iso(end)):
        out[iso] = {"expected_bars": bars, "open": opened, "close": closed,
                    "early": closed < "16:00"}
    return out


def expected_bars(start, end) -> dict[str, int]:
    """{iso: bars a COMPLETE series has that day}, by that day's own close."""
    return {r[0]: r[1] for r in _sched(_iso(start), _iso(end))}


def is_session(day) -> bool:
    iso = _iso(day)
    return bool(_sched(iso, iso))


def is_complete(actual_bars: int, expected: int) -> bool:
    """Whether a series' bar count is complete for a day KNOWN to be a
    session. Never asks whether the day was a session -- that is the
    calendar's answer, and mixing the two is what this module exists to
    stop."""
    if expected <= 0:
        return False
    return actual_bars >= expected - BAR_SHORTFALL_TOLERANCE


def shortfall(actual_bars: int, expected: int) -> int:
    """How many bars a series is short, floored at zero."""
    return max(0, expected - max(0, actual_bars))


def first_session_on_or_after(day) -> str | None:
    """The next session at or after `day`, within a year. Used where a trade
    lands on a non-trading day and has to be attributed to a real one."""
    iso = _iso(day)
    start = date.fromisoformat(iso)
    got = sessions(start, start.replace(year=start.year + 1))
    return got[0] if got else None
