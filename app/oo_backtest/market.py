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
came out null, and the Option Omega Gap cross-check crashed on it. _valid()
applies at the BAR level, so a zero bar inside a real session is missing too,
and a day with no valid bar at all is not a row of the rollup.

Postgres sorts NaN ABOVE every number, so max() over a raw column with one
NaN bar returns NaN. Values are made valid before any aggregate runs.

This module is the ONLY thing on the page that queries index_ohlc. It does so
two ways, both written here:

  DAILY_ROLLUP_SQL   one row per TRADING day (a date with at least one valid
                     bar): open (09:30 bar), high, low, and close = the last
                     valid bar at or before 15:55, for SPX, VIX, VIX3M, VIX9D;
                     plus the PREVIOUS ROW's close -- the previous trading day. Never a
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

Everything else -- gaps, ratios, the vendor Gap cross-check, coverage -- is
pandas over those two results, so it is testable without a database.
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


def _daily_select() -> str:
    parts = []
    for s in SERIES:
        parts += [
            f"max({s}_open) FILTER (WHERE quote_time = TIME '09:30') AS {s}_open",
            f"max({s}_high) AS {s}_high",
            f"min({s}_low) AS {s}_low",
            f"(array_agg({s}_close ORDER BY quote_time DESC) FILTER (WHERE {s}_close IS NOT NULL))[1] AS {s}_close",
            f"(array_agg(quote_time ORDER BY quote_time DESC) FILTER (WHERE {s}_close IS NOT NULL))[1] AS {s}_close_time",
        ]
    return ",\n           ".join(parts)


_BAR_COLS = ",\n           ".join(f"{_valid(f'{s}_{f}')} AS {s}_{f}" for s in SERIES
                                  for f in OHLC_FIELDS)

# One row per TRADING day. Everything downstream reads this, never the bars.
# prev_* is the PREVIOUS ROW of the rollup -- the prior trading day -- not
# trade_date - 1, which is wrong across every weekend and holiday. That is only
# true because the HAVING below drops the zero-filled non-trading days; without
# it the previous row of a Monday is a Sunday whose close is 0.
DAILY_ROLLUP_SQL = f"""
WITH bars AS (
    SELECT trade_date, quote_time,
           {_BAR_COLS}
    FROM index_ohlc
    WHERE quote_time BETWEEN TIME '09:30' AND TIME '15:55'
),
daily AS (
    SELECT trade_date,
           count({_any_valid()}) AS bar_count,
           {_daily_select()}
    FROM bars
    GROUP BY trade_date
    HAVING count({_any_valid()}) > 0
)
SELECT d.*,
       LAG(d.trade_date) OVER w AS prev_trade_date,
       LAG(d.spx_close)  OVER w AS spx_prev_close,
       LAG(d.vix_close)  OVER w AS vix_prev_close
FROM daily d
WINDOW w AS (ORDER BY d.trade_date)
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
# being NaN on a bar must not null the others.
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

# How the table labels its bars, per year: a start-labeled session has a
# 09:30 bar and no valid 16:00 bar. An end-labeled source (or a 5-minute
# shift between the backfill and the live writer) shows up here as years with
# no 09:30 bars or with valid 16:00 bars.
# Counted over TRADING days only (a valid bar in the session); zero-filled
# days are counted separately so they cannot inflate the others.
BAR_LABEL_SQL = f"""
WITH per_day AS (
    SELECT trade_date,
           count(*) FILTER (WHERE quote_time BETWEEN TIME '09:30' AND TIME '15:55'
                            AND {_any_valid_raw()} IS NOT NULL) AS valid_bars,
           bool_or(quote_time = TIME '09:30' AND {_valid('spx_open')} IS NOT NULL) AS has_0930,
           bool_or(quote_time = TIME '15:55' AND {_valid('spx_close')} IS NOT NULL) AS has_1555,
           bool_or(quote_time = TIME '16:00' AND {_valid('spx_close')} IS NOT NULL) AS valid_1600,
           bool_or(quote_time < TIME '09:30') AS premarket
    FROM index_ohlc
    GROUP BY trade_date
)
SELECT extract(year FROM trade_date)::int AS year,
       count(*) FILTER (WHERE valid_bars > 0) AS days,
       count(*) FILTER (WHERE valid_bars > 0 AND has_0930) AS days_with_0930,
       count(*) FILTER (WHERE valid_bars > 0 AND has_1555) AS days_with_1555,
       count(*) FILTER (WHERE valid_bars > 0 AND valid_1600) AS days_with_valid_1600,
       count(*) FILTER (WHERE valid_bars > 0 AND premarket) AS days_with_premarket,
       count(*) FILTER (WHERE valid_bars = 0) AS zero_filled_days
FROM per_day
GROUP BY 1 ORDER BY 1
"""

# Every session-window day that is either wholly without valid bars (zero-
# filled) or a trading day carrying some zero/negative or 'NaN' bars. Small:
# the first kind is ~1 row per weekend day and holiday, the second should be
# rare.
ZERO_DAYS_SQL = f"""
WITH per_day AS (
    SELECT trade_date,
           count(*) AS bars,
           count({_any_valid_raw()}) AS valid_bars,
           {", ".join(f"count(*) FILTER (WHERE {_zero_bar(s)}) AS {s}_zero_bars" for s in SERIES)},
           {", ".join(f"count(*) FILTER (WHERE {_nan_bar(s)}) AS {s}_nan_bars" for s in SERIES)}
    FROM index_ohlc
    WHERE quote_time BETWEEN TIME '09:30' AND TIME '15:55'
    GROUP BY trade_date
)
SELECT trade_date, extract(isodow FROM trade_date)::int AS isodow, bars, valid_bars,
       {", ".join(f"{s}_zero_bars, {s}_nan_bars" for s in SERIES)}
FROM per_day
WHERE valid_bars = 0 OR {" OR ".join(f"{s}_zero_bars > 0 OR {s}_nan_bars > 0" for s in SERIES)}
ORDER BY trade_date
"""


# ── daily rollup cache ──────────────────────────────────────────────────────
#
# The rollup scans every bar (~200k) and changes only when the writer adds
# bars, so it is cached per process and rebuilt when the table's latest
# (trade_date, quote_time) moves. A freshness probe is two index lookups.

_CACHE: dict = {"key": None, "daily": None, "labels": None, "zero_days": None,
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
                zeros = await conn.fetch(ZERO_DAYS_SQL)
            daily = pd.DataFrame([dict(r) for r in rows])
            zsum = zero_days_summary([dict(r) for r in zeros])
            _CACHE.update(key=key, daily=daily, labels=[dict(r) for r in labels], zero_days=zsum,
                          built_at=datetime.now().isoformat(timespec="seconds"),
                          build_s=round(_time.monotonic() - t0, 2))
            rep = fallback_report(daily)
            log.info("oo-backtest daily rollup: %d days through %s in %.2fs; close fallback "
                     "early-close=%d full-session=%s", len(daily), key[0], _CACHE["build_s"],
                     len(rep["spx"]["early_close_days"]), {s: rep[s]["full_session_count"] for s in SERIES})
            log.info("oo-backtest index_ohlc: %d zero-filled days excluded (%d weekdays); %d trading "
                     "days carry invalid bars, zero %s, NaN %s", zsum["zero_filled_days"],
                     zsum["zero_filled_weekdays"], zsum["partial_days"],
                     zsum["partial_zero_bars_by_series"], zsum["partial_nan_bars_by_series"])
            for s in SERIES:
                if rep[s]["full_session_count"]:
                    log.warning("oo-backtest: %s close fell back on %d FULL sessions (e.g. %s) -- "
                                "expected only on early closes", s, rep[s]["full_session_count"],
                                rep[s]["full_session_sample"])
    fresh.update(built_at=_CACHE["built_at"], build_s=_CACHE["build_s"])
    return _CACHE["daily"], fresh


def bar_labels() -> list | None:
    return _CACHE["labels"]


def zero_days() -> dict | None:
    return _CACHE["zero_days"]


def zero_days_summary(rows: list[dict]) -> dict:
    """Zero-filled (no valid bar) days, split weekend/weekday, and trading days
    that carry some zero bars, per series."""
    filled = [r for r in rows if r["valid_bars"] == 0]
    weekday = [r for r in filled if r["isodow"] <= 5]
    partial = [r for r in rows if r["valid_bars"] > 0]
    iso = lambda d: d.isoformat() if hasattr(d, "isoformat") else str(d)   # noqa: E731
    return {
        "zero_filled_days": len(filled),
        "zero_filled_weekend": len(filled) - len(weekday),
        "zero_filled_weekdays": len(weekday),
        # All of them: ~10 a year, and they should read as the NYSE holidays.
        "zero_filled_weekday_dates": [iso(r["trade_date"]) for r in weekday],
        # Trading days carrying some invalid bars, zero and 'NaN' counted apart.
        "partial_days": len(partial),
        "partial_zero_bars_by_series": {s: sum(r[f"{s}_zero_bars"] for r in partial) for s in SERIES},
        "partial_nan_bars_by_series": {s: sum(r[f"{s}_nan_bars"] for r in partial) for s in SERIES},
        "partial_sample": [{"date": iso(r["trade_date"]), "valid_bars": r["valid_bars"],
                            **{f"{s}_zero": r[f"{s}_zero_bars"] for s in SERIES},
                            **{f"{s}_nan": r[f"{s}_nan_bars"] for s in SERIES}} for r in partial[:25]],
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
    """First trade_date with a non-null close, per series (real coverage, not the
    table's date range -- VIX9D and VIX3M may have been backfilled later)."""
    out = {}
    for s in SERIES:
        have = daily.loc[daily[f"{s}_close"].notna(), "trade_date"] if not daily.empty else []
        out[s] = min(have).isoformat() if len(have) else None
    return out


def expected_last_session(now_et: datetime) -> date:
    """The most recent session that should be complete in the table."""
    try:
        import pandas_market_calendars as mcal   # lazy: optional on a dev box
        nyse = mcal.get_calendar("NYSE")
        sched = nyse.schedule(start_date=now_et.date() - timedelta(days=10), end_date=now_et.date())
        closes = [ts.tz_convert("America/New_York") for ts in sched["market_close"]]
        done = [c.date() for c in closes if c.replace(tzinfo=None) + timedelta(minutes=5) <= now_et]
        if done:
            return done[-1]
    except Exception as exc:  # noqa: BLE001 — fall back to weekdays, and say so
        log.info("oo-backtest: NYSE calendar unavailable (%s); staleness uses weekdays", exc)
    d = now_et.date()
    if now_et.time() < time(16, 5):
        d -= timedelta(days=1)
    while d.weekday() >= 5:
        d -= timedelta(days=1)
    return d


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
        df[f"{s}_level"] = [_float(r.get(f"{s}_entry")) for r in ent]
        df[f"{s}_bar_time"] = [r[f"{s}_bar_time"].strftime("%H:%M:%S") if r.get(f"{s}_bar_time") else None
                               for r in ent]
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


GAP_ALIGNMENTS = (("previous_row", "spx_prev_close"), ("two_rows_back", "spx_prev2_close"),
                  ("same_day_close", "spx_close"))
GAP_UNITS = ("pct", "points")
_MAG_EDGES = {"points": (2, 5, 10, 20, 40), "pct": (0.05, 0.1, 0.25, 0.5, 1.0)}


def _year_of(d) -> int:
    return d.year if hasattr(d, "year") else int(str(d)[:4])


def _mag_bucket(v: float, unit: str) -> str:
    edges = _MAG_EDGES[unit]
    a = abs(v)
    lo = 0
    for e in edges:
        if a < e:
            return f"{lo}–{e}"
        lo = e
    return f"≥{edges[-1]}"


def gap_crosscheck(df: pd.DataFrame, daily: pd.DataFrame, tol_pct: float = 0.02, tol_pts: float = 0.5) -> dict | None:
    """Option Omega's own Gap column vs the computed SPX gap.

    The vendor column is used as an oracle for the prior-row lookup. Its UNIT
    is undocumented, so every combination of unit (percent, index points) and
    prior-close alignment (previous row -- what the page uses --, two rows
    back, the same day's close) is scored and ALL SIX are returned, not just
    the winner: "two rows back" beating "previous row" would be a different
    diagnosis from "previous row wins at 63%".

    For the winning variant the disagreements are broken down by YEAR and by
    the vendor gap's MAGNITUDE, because a problem confined to older data points
    at the backfill, and one that grows with gap size points at the opening
    print rather than the prior close.
    """
    if "csv_gap" not in df.columns or df["csv_gap"].notna().sum() == 0 or daily.empty:
        return None
    d = daily.sort_values("trade_date").reset_index(drop=True)
    d["spx_prev2_close"] = d["spx_close"].shift(2)
    by_date = {r["trade_date"]: r for r in d.to_dict("records")}
    dates = pd.to_datetime(df["date_opened"]).dt.date
    vendor = df["csv_gap"].tolist()

    variants = {}
    for align, col in GAP_ALIGNMENTS:
        for unit in GAP_UNITS:
            diffs, skipped = [], 0
            for dt, v in zip(dates, vendor):
                r = by_date.get(dt)
                if r is None or v is None or pd.isna(v):
                    continue
                fo, fp = _float(r.get("spx_open")), _float(r.get(col))
                if fo is None or fp is None:
                    calc = None
                elif unit == "pct":
                    calc = _pct(fo, fp)          # None on a zero prior close
                else:
                    calc = fo - fp
                # A diagnostic never aborts the join: a pair that cannot be
                # computed (no open, no prior close, a zero close) is counted.
                if calc is None or (isinstance(calc, float) and math.isnan(calc)):
                    skipped += 1
                    continue
                diffs.append((dt, float(v), calc))
            tol = tol_pct if unit == "pct" else tol_pts
            agree = sum(1 for _, v, c in diffs if abs(v - c) <= tol)
            variants[(align, unit)] = {"agree": agree, "diffs": diffs, "skipped": skipped, "tol": tol}

    (align, unit), best = max(variants.items(),
                              key=lambda kv: (kv[1]["agree"], kv[0][0] == "previous_row"))
    tol = best["tol"]

    by_year: dict[int, dict] = {}
    by_mag: dict[str, dict] = {}
    for dt, v, c in best["diffs"]:
        ok = abs(v - c) <= tol
        y = by_year.setdefault(_year_of(dt), {"compared": 0, "agree": 0})
        m = by_mag.setdefault(_mag_bucket(v, unit), {"compared": 0, "agree": 0})
        for bucket in (y, m):
            bucket["compared"] += 1
            bucket["agree"] += ok
    edges = _MAG_EDGES[unit]
    mag_order = [f"{lo}–{hi}" for lo, hi in zip((0,) + edges[:-1], edges)] + [f"≥{edges[-1]}"]

    def rate(a, n):
        return round(a / n, 4) if n else None

    disagreements = [(dt, v, c) for dt, v, c in best["diffs"] if abs(v - c) > tol]
    worst = sorted(disagreements, key=lambda x: -abs(x[1] - x[2]))[:20]
    return {
        "compared": len(best["diffs"]),
        "best_alignment": align,
        "best_unit": unit,
        "agree": best["agree"],
        "disagree": len(best["diffs"]) - best["agree"],
        "rate": rate(best["agree"], len(best["diffs"])),
        "skipped_uncomputable": best["skipped"],
        "tolerance": tol,
        "variants": [{"alignment": a, "unit": u, "compared": len(v["diffs"]), "agree": v["agree"],
                      "rate": rate(v["agree"], len(v["diffs"])), "skipped": v["skipped"], "tolerance": v["tol"]}
                     for (a, u), v in variants.items()],
        "by_year": [{"year": y, "compared": v["compared"], "agree": v["agree"],
                     "disagree": v["compared"] - v["agree"], "rate": rate(v["agree"], v["compared"])}
                    for y, v in sorted(by_year.items())],
        "by_magnitude": [{"bucket": b, "compared": by_mag[b]["compared"], "agree": by_mag[b]["agree"],
                          "rate": rate(by_mag[b]["agree"], by_mag[b]["compared"])}
                         for b in mag_order if b in by_mag],
        "worst": [{"date": dt.isoformat(), "vendor": round(v, 4), "computed": round(c, 4),
                   "diff": round(v - c, 4)} for dt, v, c in worst],
        # Kept for gap_decomposition(); stripped before the payload.
        "_disagreements": disagreements,
    }


# ── what a disagreeing gap actually matches ────────────────────────────────

# The bars a vendor's opening print or prior close could plausibly have come
# from. A start-labeled 09:30 bar's open is our open; its close (= ~09:35) and
# the 09:35 bar are what a later print would match; a 09:25 row exists only if
# the table has pre-market bars.
OPEN_CANDIDATES = (("09:25", "open"), ("09:25", "close"), ("09:30", "open"), ("09:30", "close"),
                   ("09:35", "open"), ("09:35", "close"), ("09:40", "open"))
PREV_CLOSE_CANDIDATES = (("15:45", "close"), ("15:50", "close"), ("15:55", "open"), ("15:55", "close"),
                         ("16:00", "open"), ("16:00", "close"))

BARS_AT_SQL = f"""
SELECT trade_date, quote_time,
       {_valid('spx_open')} AS o, {_valid('spx_high')} AS h,
       {_valid('spx_low')} AS l, {_valid('spx_close')} AS c
FROM index_ohlc
WHERE trade_date = ANY($1::date[]) AND quote_time = ANY($2::time[])
"""


async def fetch_bars(pool, dates: set, times: set) -> dict:
    """{(date, 'HH:MM'): {o,h,l,c}} for the given dates x times (valid values only)."""
    if not dates or not times:
        return {}
    async with pool.acquire() as conn:
        rows = await conn.fetch(BARS_AT_SQL, sorted(dates), sorted(time.fromisoformat(t) for t in times))
    return {(r["trade_date"], r["quote_time"].strftime("%H:%M")): {k: _float(r[k]) for k in "ohlc"}
            for r in rows}


def gap_decomposition(crosscheck: dict, daily: pd.DataFrame, bars: dict, tol: float = 0.5) -> dict | None:
    """For each disagreeing day: which bar the vendor's gap is consistent with.

    Holding our prior close fixed, the vendor's IMPLIED OPEN is prior + gap;
    holding our open fixed, its IMPLIED PRIOR CLOSE is open - gap. Each is
    matched to the nearest candidate bar value within `tol` points. If most
    implied opens land on the 09:30 close / 09:35 bar, the vendor uses a later
    opening print (or our bars are shifted); if the implied prior closes land
    on a different close-of-day bar, the prior close is the problem. "none"
    means neither explains it.

    READ BOTH COLUMNS TOGETHER. When the market moves evenly, an open taken N
    bars late and a prior close taken N bars early produce the SAME vendor
    gap, and one disagreement can match in both columns. A clear diagnosis is
    one column concentrated on a single candidate while the other is mostly
    "none".
    """
    if not crosscheck or crosscheck["best_alignment"] != "previous_row":
        return None
    unit = crosscheck["best_unit"]
    by_date = {r["trade_date"]: r for r in daily.to_dict("records")}
    open_counts: dict[int, dict] = {}
    prev_counts: dict[int, dict] = {}
    for dt, v, _c in crosscheck["_disagreements"]:
        r = by_date.get(dt)
        if not r:
            continue
        o, p, pd_ = _float(r.get("spx_open")), _float(r.get("spx_prev_close")), r.get("prev_trade_date")
        if o is None or p is None:
            continue
        implied_open = p + v if unit == "points" else p * (1 + v / 100.0)
        implied_prev = o - v if unit == "points" else o / (1 + v / 100.0)
        y = _year_of(dt)
        for counts, implied, day, cands in ((open_counts, implied_open, dt, OPEN_CANDIDATES),
                                            (prev_counts, implied_prev, pd_, PREV_CLOSE_CANDIDATES)):
            best, dist = "none", None
            for hm, field in cands:
                val = (bars.get((day, hm)) or {}).get(field[0])
                if val is None:
                    continue
                dd = abs(val - implied)
                if dd <= tol and (dist is None or dd < dist):
                    best, dist = f"{hm} {field}", dd
            counts.setdefault(y, {})
            counts[y][best] = counts[y].get(best, 0) + 1
    fmt = lambda c: [{"year": y, "matches": dict(sorted(m.items(), key=lambda kv: -kv[1]))}   # noqa: E731
                     for y, m in sorted(c.items())]
    return {"tolerance_points": tol, "implied_open": fmt(open_counts), "implied_prev_close": fmt(prev_counts)}


def gap_decomposition_bars_needed(crosscheck: dict | None, daily: pd.DataFrame) -> tuple[set, set]:
    if not crosscheck or crosscheck["best_alignment"] != "previous_row":
        return set(), set()
    by_date = {r["trade_date"]: r for r in daily.to_dict("records")}
    dates = set()
    for dt, _v, _c in crosscheck["_disagreements"]:
        dates.add(dt)
        prev = (by_date.get(dt) or {}).get("prev_trade_date")
        if prev is not None:
            dates.add(prev)
    times = {hm for hm, _ in OPEN_CANDIDATES + PREV_CLOSE_CANDIDATES}
    return dates, times


# ── is the table start-labeled, year by year? ───────────────────────────────

def _shift5(hm: str, k: int) -> str:
    h, m = map(int, hm.split(":"))
    t = h * 60 + m + 5 * k
    return f"{t // 60:02d}:{t % 60:02d}"


def label_test_requests(df: pd.DataFrame) -> tuple[list, set, set]:
    """Trades usable for the intraday label test: an entry time and a vendor
    SPX price at that time (Option Omega's "Opening Price")."""
    if "spx_open_price" not in df.columns or "time_opened" not in df.columns:
        return [], set(), set()
    items, dates, times = [], set(), set()
    for d, t, px_ in zip(pd.to_datetime(df["date_opened"]).dt.date, df["time_opened"], df["spx_open_price"]):
        hms = normalize_entry_time(t)
        price = _float(px_)
        if hms is None or price is None:
            continue
        tt = time.fromisoformat(hms)
        if not (SESSION_OPEN <= tt <= LAST_BAR):
            continue
        own = _floor5(hms)[:5]
        items.append((d, hms, price, own))
        dates.add(d)
        times.update({_shift5(own, -1), own, _shift5(own, 1)})
    return items, dates, times


def label_test(items: list, bars: dict, pad: float = 0.05) -> dict | None:
    """Option Omega's SPX price at each entry time vs the bar that should hold it.

    If bars are labeled by START, the price at 10:32 lies inside the 10:30
    bar's [low, high]. If a source labels by END (or is shifted five minutes),
    it lies inside the bar labeled 10:35 instead -- or 10:25 for the opposite
    shift. Scored per YEAR, so a backfill labeled differently from the live
    writer shows up as the years where "start" stops winning. Entries exactly on
    a 5-minute boundary sit on the edge of two bars and are counted apart.
    """
    if not items:
        return None
    per_year: dict[int, dict] = {}
    for d, hms, price, own in items:
        y = per_year.setdefault(_year_of(d), {"trades": 0, "on_boundary": 0, "start": 0, "shift_plus5": 0,
                                               "shift_minus5": 0, "no_bar": 0})
        y["trades"] += 1
        if int(hms[3:5]) % 5 == 0 and hms[6:] == "00":
            y["on_boundary"] += 1
        for key, hm in (("start", own), ("shift_plus5", _shift5(own, 1)), ("shift_minus5", _shift5(own, -1))):
            b = bars.get((d, hm))
            if b and b["l"] is not None and b["h"] is not None and b["l"] - pad <= price <= b["h"] + pad:
                y[key] += 1
        if not bars.get((d, own)):
            y["no_bar"] += 1
    out = []
    for yr, v in sorted(per_year.items()):
        n = v["trades"]
        out.append({"year": yr, **v, **{f"{k}_rate": round(v[k] / n, 4) for k in ("start", "shift_plus5", "shift_minus5")}})
    return {"pad_points": pad, "by_year": out}


# ── why a value is null ─────────────────────────────────────────────────────

def null_reasons(df: pd.DataFrame, daily: pd.DataFrame, keys: pd.Series, coverage: dict) -> dict:
    """For every null gap and every null VIX-family level: the reason, counted,
    with the trades listed (up to 25 per series)."""
    by_date = {r["trade_date"]: r for r in daily.to_dict("records")} if not daily.empty else {}
    dates = pd.to_datetime(df["date_opened"]).dt.date.tolist()
    out: dict = {}

    for name, col, s in (("spx_gap", "gap", "spx"), ("vix_gap", "vix_overnight_gap", "vix")):
        counts, sample = {}, []
        for i, (d, v) in enumerate(zip(dates, df[col])):
            if _float(v) is not None:
                continue
            r = by_date.get(d)
            if r is None:
                why = "entry date is not a session in index_ohlc"
            elif _float(r.get(f"{s}_open")) is None:
                why = f"no valid 09:30 {s.upper()} open that day"
            elif r.get("prev_trade_date") is None:
                why = "first session in the table (no previous session)"
            elif _float(r.get(f"{s}_prev_close")) is None:
                why = f"previous session ({r['prev_trade_date']}) has no valid {s.upper()} close"
            else:
                why = "prior close is zero (should be impossible after validity filtering)"
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
                why = "entry date is not a session in index_ohlc"
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

    # Diagnostics. None of them may cost the log its market data: each failure
    # is logged with its traceback and reported by name.
    diag_errors: dict = {}
    crosscheck = decomposition = labels = None
    try:
        crosscheck = gap_crosscheck(out, daily)
    except Exception as exc:  # noqa: BLE001
        log.exception("oo-backtest gap cross-check failed")
        diag_errors["gap_crosscheck"] = f"{type(exc).__name__}: {exc}"
    try:
        dd, tt = gap_decomposition_bars_needed(crosscheck, daily)
        items, ld, lt = label_test_requests(out)
        bars = await fetch_bars(pool, dd | ld, tt | lt)
        decomposition = gap_decomposition(crosscheck, daily, bars)
        labels = label_test(items, bars)
    except Exception as exc:  # noqa: BLE001
        log.exception("oo-backtest gap decomposition / label test failed")
        diag_errors["gap_decomposition"] = f"{type(exc).__name__}: {exc}"
    try:
        reasons = null_reasons(out, daily, keys, coverage(daily))
    except Exception as exc:  # noqa: BLE001
        log.exception("oo-backtest null reasons failed")
        reasons, diag_errors["null_reasons"] = None, f"{type(exc).__name__}: {exc}"
    if crosscheck:
        crosscheck = {k: v for k, v in crosscheck.items() if not k.startswith("_")}

    report = {
        "joined": True,
        "source": "main.index_ohlc",
        "freshness": fresh,
        "entry_time": entry_info,
        "entry_bars": entry_bar_report(out, keys),
        "gap_crosscheck": crosscheck,
        "gap_decomposition": decomposition,
        "label_test": labels,
        "null_reasons": reasons,
        "coverage": coverage(daily),
        "diagnostic_errors": diag_errors,
        "gap_crosscheck_error": diag_errors.get("gap_crosscheck"),
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
