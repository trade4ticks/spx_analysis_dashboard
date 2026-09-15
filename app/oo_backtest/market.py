"""Market data for the OO/Mesosim Backtest page, from main.index_ohlc.

index_ohlc is 5-minute bars, one row per (trade_date, quote_time), ET, LABELED
BY START TIME: 09:30:00 is the first regular bar and 15:55:00 spans
15:55-16:00, so its close is the session close. The 16:00:00 rows are partial
or NaN and are never read.

A VALUE IS VALID ONLY IF IT IS NOT NULL, NOT 'NaN', AND GREATER THAN ZERO.
Found on the VPS (2026-09-14): the table carries ZERO-FILLED rows -- 78 bars of
0.0 -- for every non-trading day (weekends and the NYSE holidays), and the
writer also pushes pandas NaN as 'NaN' rather than NULL. The first version of
this module nulled only 'NaN', so a zero-filled Sunday became a "session" with
a close of 0: every Monday's previous row was that Sunday, every Monday gap
came out null. _valid() applies at the BAR level, so a zero bar inside a
real session is missing too.

SESSIONS ARE PER SERIES and need MIN_SESSION_BARS valid closing bars (see
there). The next finding: VIX ingestion writes 11-25 valid bars on market
holidays, so "any valid bar" kept every holiday as a day, and the day after a
holiday took it as its previous session. Now a holiday with only VIX
artifacts is no session at all, and a day like 2026-04-08 (a full VIX
session, no SPX) is a VIX session and not an SPX one.

Postgres sorts NaN ABOVE every number, so max() over a raw column with one
NaN bar returns NaN. Values are made valid before any aggregate runs.

This module is the ONLY thing on the page that queries index_ohlc. It does so
two ways, both written here:

  DAILY_ROLLUP_SQL   one row per day on which any series has a SESSION: for
                     each series with a session that day, open (09:30 bar),
                     high, low, and close = the last valid bar at or before
                     15:55; plus, for SPX and VIX, the close of that series'
                     own previous session. Never a
                     close pinned to 15:55 -- early-close sessions end on the
                     12:55 bar and a fixed-time join drops them silently. How
                     often the fallback fires is reported (fallback_report).

  ENTRY_BAR_SQL      for each trade, the bar containing its entry time: the
                     last non-null bar on the entry DATE with quote_time <=
                     entry time. Never the previous session.

WHICH PRICE FROM THE ENTRY BAR. Its OPEN. A bar is labeled by its start, so a
15:30 entry lands on the 15:30 bar -- whose close is the price at 15:35,
five minutes after the trade was placed. Taking that close would reintroduce,
in miniature, exactly the lookahead this join exists to remove (the source
app gave every trade its date's daily close). The open of the bar that
contains the entry is the last price known at entry.

Everything else -- gaps, ratios, coverage, why a value is null -- is pandas
over those two results, so it is testable without a database.

THERE IS NO VENDOR GAP CROSS-CHECK. Option Omega's "Gap" column uses a
different definition from open-minus-prior-close (on one hand-checked day it
matched a post-16:00 straggler print, on another it did not), so it is not an
oracle for this calculation. It was built, found incomparable, and removed
(2026-09-14); 2018-05-07 was checked by hand against the OHLC and the computed
gap is exact.
"""
from __future__ import annotations

import asyncio
import logging
import math
import re
import time as _time
from datetime import date, datetime, time, timedelta

import pandas as pd

log = logging.getLogger(__name__)

SERIES = ("spx", "vix", "vix3m", "vix9d")
SESSION_OPEN = time(9, 30)
LAST_BAR = time(15, 55)
EARLY_CLOSE_LAST_BAR = time(13, 0)   # a session whose close bar is at/before this ended early


def _valid(col: str) -> str:
    """The column where it is a real price, else NULL.

    'NaN' must be tested explicitly: NaN > 0 is TRUE in Postgres.
    """
    return f"(CASE WHEN {col} = 'NaN'::float8 OR {col} <= 0 THEN NULL ELSE {col} END)"


OHLC_FIELDS = ("open", "high", "low", "close")


def _any_valid() -> str:
    """Non-null when ANY series has a valid value on the bar (for bars already
    passed through _valid)."""
    return "COALESCE(" + ", ".join(f"{s}_{f}" for s in SERIES for f in OHLC_FIELDS) + ")"


def _any_valid_raw() -> str:
    return "COALESCE(" + ", ".join(_valid(f"{s}_{f}") for s in SERIES for f in OHLC_FIELDS) + ")"


def _zero_bar(s: str) -> str:
    return "(" + " OR ".join(f"{s}_{f} <= 0" for f in OHLC_FIELDS) + ")"


def _nan_bar(s: str) -> str:
    # Separate from _zero_bar: 'NaN' <= 0 is FALSE in Postgres, so a zero test
    # alone under-counts invalid bars. NULL is neither -- before a series'
    # coverage every bar is NULL, and that is not a data fault.
    return "(" + " OR ".join(f"{s}_{f} = 'NaN'::float8" for f in OHLC_FIELDS) + ")"


# A SERIES HAS A SESSION on a day when it has at least MIN_SESSION_BARS valid
# closing bars in 09:30-15:55. Sessions are per series, not per day.
#
# Measured on the VPS (2026-09-14): VIX ingestion writes 11-25 valid bars on
# market holidays -- stale/partial artifacts, against 78 on a real session
# and 42 on an early close (09:30-12:55). The threshold sits between them. It
# is a count and not "a valid 09:30 open and a close" because the writer's own
# notes say VIX3M/VIX9D can arrive a bar or two behind: a real session whose
# first bar is missing must keep its close (the next day's prior close), and
# only its own gap goes null. The shortest session kept and the longest
# artifact rejected are reported per series (session_report), so a real
# early close falling under the line, or an artifact rising over it, shows.
#
# 2026-04-08 is the other shape: 78 valid VIX bars, zero SPX. That is a VIX
# session and not an SPX one -- a rollup row with SPX null.
MIN_SESSION_BARS = 34


def _agg_select() -> str:
    parts = []
    for s in SERIES:
        parts += [
            f"count({s}_close) AS {s}_bars",
            f"max({s}_open) FILTER (WHERE quote_time = TIME '09:30') AS {s}_open_x",
            f"max({s}_high) AS {s}_high_x",
            f"min({s}_low) AS {s}_low_x",
            f"(array_agg({s}_close ORDER BY quote_time DESC) FILTER (WHERE {s}_close IS NOT NULL))[1] AS {s}_close_x",
            f"(array_agg(quote_time ORDER BY quote_time DESC) FILTER (WHERE {s}_close IS NOT NULL))[1] AS {s}_close_time_x",
        ]
    return ",\n           ".join(parts)


def _session_select() -> str:
    parts = []
    for s in SERIES:
        ok = f"{s}_bars >= {MIN_SESSION_BARS}"
        parts += [f"{s}_bars", f"({ok}) AS {s}_session"]
        # Outside a session NOTHING of that series survives -- not its close,
        # high or low -- so an artifact can never become a price.
        parts += [f"CASE WHEN {ok} THEN {s}_{f}_x END AS {s}_{f}"
                  for f in ("open", "high", "low", "close", "close_time")]
    return ",\n           ".join(parts)


_BAR_COLS = ",\n           ".join(f"{_valid(f'{s}_{f}')} AS {s}_{f}" for s in SERIES
                                  for f in OHLC_FIELDS)

_PREV_SERIES = ("spx", "vix")   # the series with an overnight gap

# One row per day on which ANY series has a session. Everything downstream
# reads this, never the bars.
#
# The previous close is PER SERIES: the previous row among days where THAT
# series had a session. Not the previous row of the rollup (a VIX-only day
# like 2026-04-08 would be SPX's "previous session"), not trade_date - 1, and
# no calendar: the table's own validity is the definition of a session.
DAILY_ROLLUP_SQL = f"""
WITH bars AS (
    SELECT trade_date, quote_time,
           {_BAR_COLS}
    FROM index_ohlc
    WHERE quote_time BETWEEN TIME '09:30' AND TIME '15:55'
),
agg AS (
    SELECT trade_date,
           {_agg_select()}
    FROM bars
    GROUP BY trade_date
),
daily AS (
    SELECT trade_date,
           {_session_select()}
    FROM agg
    WHERE {" OR ".join(f"{s}_bars >= {MIN_SESSION_BARS}" for s in SERIES)}
),
{",".join(f'''
{s}_prev AS (
    SELECT trade_date,
           LAG(trade_date) OVER w AS {s}_prev_date,
           LAG({s}_close)  OVER w AS {s}_prev_close
    FROM daily
    WHERE {s}_session
    WINDOW w AS (ORDER BY trade_date)
)''' for s in _PREV_SERIES)}
SELECT d.*, {", ".join(f"{s}_prev.{s}_prev_date, {s}_prev.{s}_prev_close" for s in _PREV_SERIES)}
FROM daily d
{" ".join(f"LEFT JOIN {s}_prev USING (trade_date)" for s in _PREV_SERIES)}
ORDER BY d.trade_date
"""

_LEVEL_SERIES = ("vix", "vix3m", "vix9d")


def _entry_lateral(s: str) -> str:
    return f"""
LEFT JOIN LATERAL (
    SELECT quote_time AS {s}_bar_time, {_valid(f'{s}_open')} AS {s}_entry
    FROM index_ohlc
    WHERE trade_date = t.d
      AND quote_time >= TIME '09:30'
      AND quote_time <= LEAST(t.tm, TIME '15:55')
      AND {_valid(f'{s}_open')} IS NOT NULL
    ORDER BY quote_time DESC
    LIMIT 1
) {s}_b ON true"""


# $1 date[], $2 time[] -- one element per DISTINCT (entry date, entry time).
# Per series, because VIX3M/VIX9D can arrive a bar behind SPX: one series
# being NaN on a bar must not null the others. A bar found here on a day that
# series has NO session is discarded in apply_market().
ENTRY_BAR_SQL = (
    "SELECT t.d AS trade_date, t.tm AS entry_time, "
    + ", ".join(f"{s}_b.{s}_bar_time, {s}_b.{s}_entry" for s in _LEVEL_SERIES)
    + "\nFROM unnest($1::date[], $2::time[]) AS t(d, tm)"
    + "".join(_entry_lateral(s) for s in _LEVEL_SERIES)
)

# The cache key reads the RAW latest row, so any new row (zero-filled or not)
# rebuilds the rollup. What the page SHOWS is the latest VALID SPX bar in a
# session: a zero-filled weekend at the end of the table is not "data through
# Sunday".
CACHE_KEY_SQL = """
SELECT max(trade_date) AS raw_date,
       (SELECT max(quote_time) FROM index_ohlc
         WHERE trade_date = (SELECT max(trade_date) FROM index_ohlc)) AS raw_time
FROM index_ohlc
"""

FRESHNESS_SQL = f"""
SELECT trade_date AS latest_date, quote_time AS latest_time
FROM index_ohlc
WHERE quote_time BETWEEN TIME '09:30' AND TIME '15:55'
  AND {_valid('spx_close')} IS NOT NULL
ORDER BY trade_date DESC, quote_time DESC
LIMIT 1
"""

# How the table labels its bars, per year, over SPX SESSIONS only: a
# start-labeled session has a 09:30 bar and no valid 16:00 bar.
BAR_LABEL_SQL = f"""
WITH per_day AS (
    SELECT trade_date,
           count(*) FILTER (WHERE quote_time BETWEEN TIME '09:30' AND TIME '15:55'
                            AND {_valid('spx_close')} IS NOT NULL) AS spx_bars,
           bool_or(quote_time = TIME '09:30' AND {_valid('spx_open')} IS NOT NULL) AS has_0930,
           bool_or(quote_time = TIME '15:55' AND {_valid('spx_close')} IS NOT NULL) AS has_1555,
           bool_or(quote_time = TIME '16:00' AND {_valid('spx_close')} IS NOT NULL) AS valid_1600,
           bool_or(quote_time < TIME '09:30') AS premarket
    FROM index_ohlc
    GROUP BY trade_date
)
SELECT extract(year FROM trade_date)::int AS year,
       count(*) FILTER (WHERE spx_bars >= {MIN_SESSION_BARS}) AS days,
       count(*) FILTER (WHERE spx_bars >= {MIN_SESSION_BARS} AND has_0930) AS days_with_0930,
       count(*) FILTER (WHERE spx_bars >= {MIN_SESSION_BARS} AND has_1555) AS days_with_1555,
       count(*) FILTER (WHERE spx_bars >= {MIN_SESSION_BARS} AND valid_1600) AS days_with_valid_1600,
       count(*) FILTER (WHERE spx_bars >= {MIN_SESSION_BARS} AND premarket) AS days_with_premarket,
       count(*) FILTER (WHERE spx_bars < {MIN_SESSION_BARS}) AS non_spx_session_days
FROM per_day
GROUP BY 1 ORDER BY 1
"""

# Per day and series, over the session window: valid closing bars, zero bars
# and 'NaN' bars. One row per date in the table (a few thousand), from which
# session_report() classifies every day.
DAY_SERIES_SQL = f"""
SELECT trade_date, extract(isodow FROM trade_date)::int AS isodow,
       {", ".join(f"count(*) FILTER (WHERE {_valid(f'{s}_close')} IS NOT NULL) AS {s}_bars" for s in SERIES)},
       {", ".join(f"count(*) FILTER (WHERE {_zero_bar(s)}) AS {s}_zero_bars" for s in SERIES)},
       {", ".join(f"count(*) FILTER (WHERE {_nan_bar(s)}) AS {s}_nan_bars" for s in SERIES)}
FROM index_ohlc
WHERE quote_time BETWEEN TIME '09:30' AND TIME '15:55'
GROUP BY trade_date
ORDER BY trade_date
"""


# ── daily rollup cache ──────────────────────────────────────────────────────
#
# The rollup scans every bar (~200k) and changes only when the writer adds
# bars, so it is cached per process and rebuilt when the table's latest
# (trade_date, quote_time) moves. A freshness probe is two index lookups.

_CACHE: dict = {"key": None, "daily": None, "labels": None, "sessions": None,
                "built_at": None, "build_s": None}
_LOCK = asyncio.Lock()


async def get_daily(pool) -> tuple[pd.DataFrame, dict]:
    """(daily rollup DataFrame, freshness dict). Rebuilds when the table moved."""
    async with pool.acquire() as conn:
        k = await conn.fetchrow(CACHE_KEY_SQL)
        fr = await conn.fetchrow(FRESHNESS_SQL)
    key = (k["raw_date"], k["raw_time"])
    fresh = {"latest_date": fr["latest_date"].isoformat() if fr and fr["latest_date"] else None,
             "latest_time": fr["latest_time"].strftime("%H:%M:%S") if fr and fr["latest_time"] else None,
             "latest_raw_date": k["raw_date"].isoformat() if k["raw_date"] else None}
    async with _LOCK:
        if _CACHE["key"] != key or _CACHE["daily"] is None:
            t0 = _time.monotonic()
            async with pool.acquire() as conn:
                rows = await conn.fetch(DAILY_ROLLUP_SQL)
                labels = await conn.fetch(BAR_LABEL_SQL)
                per_day = await conn.fetch(DAY_SERIES_SQL)
            daily = pd.DataFrame([dict(r) for r in rows])
            sessions = session_report([dict(r) for r in per_day])
            _CACHE.update(key=key, daily=daily, labels=[dict(r) for r in labels], sessions=sessions,
                          built_at=datetime.now().isoformat(timespec="seconds"),
                          build_s=round(_time.monotonic() - t0, 2))
            rep = fallback_report(daily)
            log.info("oo-backtest daily rollup: %d days through %s in %.2fs; close fallback "
                     "early-close=%d full-session=%s", len(daily), key[0], _CACHE["build_s"],
                     len(rep["spx"]["early_close_days"]), {s: rep[s]["full_session_count"] for s in SERIES})
            log.info("oo-backtest index_ohlc sessions: %d zero-filled days, %d artifact-only days; "
                     "shortest kept %s; longest rejected %s", sessions["zero_filled_days"],
                     sessions["artifact_only_days"],
                     {s: v["shortest_kept"] for s, v in sessions["by_series"].items()},
                     {s: v["longest_rejected"] for s, v in sessions["by_series"].items()})
            for s in SERIES:
                if rep[s]["full_session_count"]:
                    log.warning("oo-backtest: %s close fell back on %d FULL sessions (e.g. %s) -- "
                                "expected only on early closes", s, rep[s]["full_session_count"],
                                rep[s]["full_session_sample"])
    fresh.update(built_at=_CACHE["built_at"], build_s=_CACHE["build_s"])
    return _CACHE["daily"], fresh


def bar_labels() -> list | None:
    return _CACHE["labels"]


def sessions() -> dict | None:
    return _CACHE["sessions"]


def session_report(rows: list[dict], min_bars: int = MIN_SESSION_BARS) -> dict:
    """Every date in the table, classified by the per-series session rule.

      zero-filled     no valid bar in any series (weekends, most holidays)
      artifact-only   some valid bars, but no series reaches a session
                      (the VIX holiday artifacts)
      session day     at least one series has a session; per series it may
                      still be MISSING (0 bars, e.g. SPX on 2026-04-08) or an
                      ARTIFACT (1..min-1 bars)

    Per series: the shortest session KEPT and the longest artifact REJECTED,
    with dates, so a threshold sitting too close to either side is visible.
    Invalid (zero or 'NaN') bars are counted inside sessions only.
    """
    iso = lambda d: d.isoformat() if hasattr(d, "isoformat") else str(d)   # noqa: E731
    zero_filled, artifact_only = [], []
    by = {s: {"sessions": 0, "shortest_kept": None, "longest_rejected": None,
              "missing_on_session_days": [], "artifact_on_session_days": [],
              "zero_bars_in_sessions": 0, "nan_bars_in_sessions": 0} for s in SERIES}
    for r in rows:
        counts = {s: r[f"{s}_bars"] for s in SERIES}
        has_session = [s for s in SERIES if counts[s] >= min_bars]
        if not any(counts.values()):
            zero_filled.append(r)
        elif not has_session:
            artifact_only.append({"date": iso(r["trade_date"]), "isodow": r["isodow"],
                                  **{s: counts[s] for s in SERIES}})
        for s in SERIES:
            n, v = counts[s], by[s]
            if n >= min_bars:
                v["sessions"] += 1
                v["zero_bars_in_sessions"] += r[f"{s}_zero_bars"]
                v["nan_bars_in_sessions"] += r[f"{s}_nan_bars"]
                if v["shortest_kept"] is None or n < v["shortest_kept"]["bars"]:
                    v["shortest_kept"] = {"date": iso(r["trade_date"]), "bars": n}
            elif n > 0:
                if v["longest_rejected"] is None or n > v["longest_rejected"]["bars"]:
                    v["longest_rejected"] = {"date": iso(r["trade_date"]), "bars": n}
                if has_session:
                    v["artifact_on_session_days"].append({"date": iso(r["trade_date"]), "bars": n})
            elif has_session:
                v["missing_on_session_days"].append(iso(r["trade_date"]))
    weekday_zero = [r for r in zero_filled if r["isodow"] <= 5]
    return {
        "min_session_bars": min_bars,
        "zero_filled_days": len(zero_filled),
        "zero_filled_weekdays": len(weekday_zero),
        "zero_filled_weekday_dates": [iso(r["trade_date"]) for r in weekday_zero],
        "artifact_only_days": len(artifact_only),
        "artifact_only": artifact_only,
        "by_series": by,
    }


# ── pure functions over the rollup (tested without a database) ─────────────

def _t(v) -> time | None:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return None
    if isinstance(v, time):
        return v
    if isinstance(v, pd.Timedelta):
        secs = int(v.total_seconds())
        return time(secs // 3600, (secs % 3600) // 60, secs % 60)
    return time.fromisoformat(str(v))


def fallback_report(daily: pd.DataFrame) -> dict:
    """Where a series' close did NOT come from the 15:55 bar.

    Split by whether the SESSION ended early (SPX's own close bar at or before
    13:00). Early closes are expected, one a year or so apiece. A fallback on a
    full session means a NaN 15:55 bar or a series lagging SPX -- worth seeing,
    and if common, a sign that something else is wrong.
    """
    out = {}
    if daily is None or daily.empty:
        return {s: {"early_close_days": [], "full_session_count": 0, "full_session_sample": [],
                    "no_close_days": 0} for s in SERIES}
    spx_t = daily["spx_close_time"].map(_t)
    early = spx_t.map(lambda x: x is not None and x <= EARLY_CLOSE_LAST_BAR)
    for s in SERIES:
        ct = daily[f"{s}_close_time"].map(_t)
        fell = ct.map(lambda x: x is not None and x < LAST_BAR)
        dates = daily["trade_date"].map(lambda d: d.isoformat() if hasattr(d, "isoformat") else str(d))
        full = dates[fell & ~early].tolist()
        out[s] = {
            "early_close_days": dates[fell & early].tolist(),
            "full_session_count": len(full),
            "full_session_sample": full[:10],
            "no_close_days": int(ct.isna().sum()),
        }
    return out


def coverage(daily: pd.DataFrame) -> dict:
    """First SESSION per series (real coverage, not the table's date range --
    VIX9D and VIX3M may have been backfilled later). Outside a session a
    series' close is null in the rollup, so an artifact cannot start coverage."""
    out = {}
    for s in SERIES:
        have = daily.loc[daily[f"{s}_close"].notna(), "trade_date"] if not daily.empty else []
        out[s] = min(have).isoformat() if len(have) else None
    return out


def session_days(daily: pd.DataFrame, df: pd.DataFrame, series: str = "spx") -> list[str]:
    """ISO dates `series` had a session on, from the log's first entry to its
    last exit: the x axis of the Deployment chart. Every session in the span,
    not only days with an entry or exit, so a stretch with nothing open shows
    as a run of zeros. SPX sessions, per the page's spec -- a day like
    2026-04-08 (VIX session, no SPX bars) is not in the list."""
    if daily.empty or df.empty:
        return []
    lo = pd.to_datetime(df["date_opened"]).min()
    hi = pd.to_datetime(df["date_closed"] if "date_closed" in df.columns else df["date_opened"]).max()
    if pd.isna(lo) or pd.isna(hi):
        return []
    days = pd.to_datetime(daily.loc[daily[f"{series}_session"].astype(bool), "trade_date"])
    return [d.date().isoformat() for d in days[(days >= lo.normalize()) & (days <= hi.normalize())]]


STALE_AFTER_DAYS = 5


def staleness(latest_date: str | None, today: date) -> dict:
    """Stale when the latest valid SPX bar is more than STALE_AFTER_DAYS
    calendar days old. No trading calendar, deliberately: the table's own data
    defines a session, and a calendar would disagree with it on days like
    2026-04-08. Five days covers a weekend plus a holiday; the cost is that a
    stalled writer is noticed up to five days late."""
    if latest_date is None:
        return {"stale": True, "age_days": None, "stale_after_days": STALE_AFTER_DAYS}
    age = (today - date.fromisoformat(latest_date)).days
    return {"stale": age > STALE_AFTER_DAYS, "age_days": age, "stale_after_days": STALE_AFTER_DAYS}


_TIME_FORMATS = ("%H:%M:%S", "%H:%M", "%I:%M:%S %p", "%I:%M %p", "%I:%M:%S%p", "%I:%M%p")


def normalize_entry_time(v) -> str | None:
    """'HH:MM:SS' from what a vendor writes ('15:30:00', '9:31', '3:30 PM'), or None."""
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return None
    if isinstance(v, time):
        return v.strftime("%H:%M:%S")
    s = str(v).strip()
    if not s or s.lower() in ("nan", "none", "nat"):
        return None
    for fmt in _TIME_FORMATS:
        try:
            return datetime.strptime(s.upper(), fmt).strftime("%H:%M:%S")
        except ValueError:
            continue
    m = re.search(r"(\d{1,2}):(\d{2})(?::(\d{2}))?", s)
    if m:
        h, mi, se = int(m.group(1)), int(m.group(2)), int(m.group(3) or 0)
        if h < 24 and mi < 60 and se < 60:
            return f"{h:02d}:{mi:02d}:{se:02d}"
    return None


def entry_bar_requests(df: pd.DataFrame) -> tuple[list[date], list[time], pd.Series, dict]:
    """The distinct (date, time) pairs to look up, each trade's key, and coverage.

    A trade with no parseable entry time is looked up at 09:30 (the session
    open) -- counted, because a silent 0% would turn every VIX metric into a
    daily-open proxy while the page looked normal.
    """
    raw = df["time_opened"] if "time_opened" in df.columns else pd.Series([None] * len(df), index=df.index)
    norm = raw.map(normalize_entry_time)
    found = int(norm.notna().sum())
    used = norm.fillna(SESSION_OPEN.strftime("%H:%M:%S"))
    dates = pd.to_datetime(df["date_opened"]).dt.date
    keys = pd.Series(list(zip(dates, used)), index=df.index)
    distinct = sorted(set(keys))
    tod = [time.fromisoformat(t) for t in norm.dropna()]
    info = {
        "entry_time_found": found,
        "trades": int(len(df)),
        "entry_time_fallback_0930": int(len(df) - found),
        "before_open": sum(1 for t in tod if t < SESSION_OPEN),
        "after_close": sum(1 for t in tod if t > time(16, 0)),
    }
    return [d for d, _ in distinct], [time.fromisoformat(t) for _, t in distinct], keys, info


def _pct(a, b):
    try:
        if a is None or b is None or pd.isna(a) or pd.isna(b) or b == 0:
            return None
        return (float(a) - float(b)) / float(b) * 100.0
    except (TypeError, ValueError):
        return None


def apply_market(df: pd.DataFrame, daily: pd.DataFrame, entry_rows: list[dict], keys: pd.Series) -> pd.DataFrame:
    """Join the rollup (by entry date) and the entry bars (by entry date+time)."""
    df = df.copy()
    dates = pd.to_datetime(df["date_opened"]).dt.date

    by_date = {r["trade_date"]: r for r in daily.to_dict("records")} if not daily.empty else {}
    get = lambda col: [by_date.get(d, {}).get(col) for d in dates]   # noqa: E731

    # Gaps are a property of the DAY: 09:30 open vs the previous row's close.
    df["gap"] = [_pct(o, p) for o, p in zip(get("spx_open"), get("spx_prev_close"))]
    df["vix_overnight_gap"] = [_pct(o, p) for o, p in zip(get("vix_open"), get("vix_prev_close"))]

    # Daily-close ratios: the source app's basis, kept for comparison only.
    vix_c, v3m_c, v9d_c = get("vix_close"), get("vix3m_close"), get("vix9d_close")
    df["vix3m_vix_ratio_close"] = [_ratio(a, b) for a, b in zip(v3m_c, vix_c)]
    df["vix_vix9d_ratio_close"] = [_ratio(a, b) for a, b in zip(vix_c, v9d_c)]

    by_key = {(r["trade_date"], r["entry_time"].strftime("%H:%M:%S")): r for r in entry_rows}
    ent = [by_key.get(k, {}) for k in keys]
    for s in _LEVEL_SERIES:
        # A bar on a day this series has no SESSION is an artifact (the VIX
        # holiday rows): the level is null, never that bar's price.
        in_session = [bool(by_date.get(d, {}).get(f"{s}_session")) for d in dates]
        df[f"{s}_level"] = [_float(r.get(f"{s}_entry")) if ok else None for r, ok in zip(ent, in_session)]
        df[f"{s}_bar_time"] = [r[f"{s}_bar_time"].strftime("%H:%M:%S") if ok and r.get(f"{s}_bar_time") else None
                               for r, ok in zip(ent, in_session)]
    df["vix3m_vix_ratio_entry"] = [_ratio(a, b) for a, b in zip(df["vix3m_level"], df["vix_level"])]
    df["vix_vix9d_ratio_entry"] = [_ratio(a, b) for a, b in zip(df["vix_level"], df["vix9d_level"])]
    return df


def _float(v):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(f) else f


def _ratio(a, b):
    a, b = _float(a), _float(b)
    return None if a is None or b in (None, 0) else a / b


def entry_bar_report(df: pd.DataFrame, keys: pd.Series) -> dict:
    """How the entry join landed: null levels, and bars earlier than the entry's
    own 5-minute bar (a NaN on the entry bar pushed it back)."""
    out = {}
    own = [(_floor5(t)) for _, t in keys]
    for s in _LEVEL_SERIES:
        bt = df[f"{s}_bar_time"].tolist()
        out[s] = {
            "null": int(df[f"{s}_level"].isna().sum()),
            # A missing bar arrives as NaN once it has been through a DataFrame.
            "earlier_than_entry_bar": sum(1 for b, o in zip(bt, own) if isinstance(b, str) and b < o),
        }
    return out


def _floor5(hms: str) -> str:
    t = time.fromisoformat(hms)
    m = t.hour * 60 + t.minute
    m -= m % 5
    m = max(min(m, 15 * 60 + 55), 9 * 60 + 30)
    return f"{m // 60:02d}:{m % 60:02d}:00"


# ── why a value is null ─────────────────────────────────────────────────────

def null_reasons(df: pd.DataFrame, daily: pd.DataFrame, keys: pd.Series, coverage: dict) -> dict:
    """For every null gap and every null VIX-family level: the reason, counted,
    with the trades listed (up to 25 per series)."""
    by_date = {r["trade_date"]: r for r in daily.to_dict("records")} if not daily.empty else {}
    dates = pd.to_datetime(df["date_opened"]).dt.date.tolist()
    out: dict = {}

    def no_session(r, s):
        return (f"no {s.upper()} session that day ({int(r.get(f'{s}_bars') or 0)} valid {s.upper()} bars; "
                f"{MIN_SESSION_BARS} required)")

    for name, col, s in (("spx_gap", "gap", "spx"), ("vix_gap", "vix_overnight_gap", "vix")):
        counts, sample = {}, []
        for i, (d, v) in enumerate(zip(dates, df[col])):
            if _float(v) is not None:
                continue
            r = by_date.get(d)
            if r is None:
                why = "entry date is not a session in index_ohlc (no series has a session that day)"
            elif not r.get(f"{s}_session"):
                why = no_session(r, s)
            elif _float(r.get(f"{s}_open")) is None:
                why = f"no valid 09:30 {s.upper()} open that day"
            elif r.get(f"{s}_prev_date") is None:
                why = f"first {s.upper()} session in the table (no earlier {s.upper()} session)"
            else:
                why = f"previous {s.upper()} session ({r[f'{s}_prev_date']}) has no valid close (should be impossible)"
            counts[why] = counts.get(why, 0) + 1
            if len(sample) < 25:
                sample.append({"row": i, "date": d.isoformat(), "reason": why})
        out[name] = {"null": sum(counts.values()), "reasons": counts, "trades": sample}

    for s in _LEVEL_SERIES:
        counts, sample = {}, []
        first = coverage.get(s)
        for i, (d, (kd, hms), v) in enumerate(zip(dates, keys, df[f"{s}_level"])):
            if _float(v) is not None:
                continue
            if time.fromisoformat(hms) < SESSION_OPEN:
                why = "entry before 09:30"
            elif first and d.isoformat() < first:
                why = f"entry date before {s.upper()} coverage ({first})"
            elif d not in by_date:
                why = "entry date is not a session in index_ohlc (no series has a session that day)"
            elif not by_date[d].get(f"{s}_session"):
                why = no_session(by_date[d], s)
            else:
                why = f"no valid {s.upper()} bar at or before the entry time that day"
            counts[why] = counts.get(why, 0) + 1
            if len(sample) < 25:
                sample.append({"row": i, "date": d.isoformat(), "time": hms, "reason": why})
        out[s] = {"null": sum(counts.values()), "reasons": counts, "trades": sample}
    return out


async def join_market(pool, df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Add levels, ratios and gaps to a parsed trade log. Returns (df, report)."""
    daily, fresh = await get_daily(pool)
    ds, ts, keys, entry_info = entry_bar_requests(df)
    async with pool.acquire() as conn:
        rows = await conn.fetch(ENTRY_BAR_SQL, ds, ts) if ds else []
    entry_rows = [dict(r) for r in rows]
    out = apply_market(df, daily, entry_rows, keys)
    known = set(daily["trade_date"]) if not daily.empty else set()

    # Diagnostics may not cost the log its market data: a failure is logged
    # with its traceback and reported by name.
    diag_errors: dict = {}
    try:
        reasons = null_reasons(out, daily, keys, coverage(daily))
    except Exception as exc:  # noqa: BLE001
        log.exception("oo-backtest null reasons failed")
        reasons, diag_errors["null_reasons"] = None, f"{type(exc).__name__}: {exc}"
    try:
        spx_sessions = session_days(daily, out)
    except Exception as exc:  # noqa: BLE001
        log.exception("oo-backtest session days failed")
        spx_sessions, diag_errors["spx_sessions"] = [], f"{type(exc).__name__}: {exc}"

    report = {
        "joined": True,
        "source": "main.index_ohlc",
        "freshness": fresh,
        "entry_time": entry_info,
        "entry_bars": entry_bar_report(out, keys),
        "null_reasons": reasons,
        "coverage": coverage(daily),
        "diagnostic_errors": diag_errors,
        "spx_sessions": spx_sessions,
        # How many trades got a gap at all. All-null here is what the zero-
        # filled weekends produced for a Monday-only log, silently.
        "gaps": {name: {"computed": int(out[col].notna().sum()), "null": int(out[col].isna().sum())}
                 for name, col in (("spx", "gap"), ("vix", "vix_overnight_gap"))},
        # Entry dates the table has no session for: before coverage, or a
        # date that is not a trading day at all (a timezone slip can do that).
        "trades_without_daily_row": int(sum(1 for d in pd.to_datetime(out["date_opened"]).dt.date
                                            if d not in known)),
    }
    return out, report
