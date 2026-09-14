"""Market data for the OO/Mesosim Backtest page, from main.index_ohlc.

index_ohlc is 5-minute bars, one row per (trade_date, quote_time), ET, LABELED
BY START TIME: 09:30:00 is the first regular bar and 15:55:00 spans
15:55-16:00, so its close is the session close. The 16:00:00 rows are partial
or NaN and are never read. Missing values can be stored as 'NaN' rather than
NULL -- the writer pushes pandas frames -- so every read goes through
NULLIF(col, 'NaN') and "non-null" below means neither.

This module is the ONLY thing on the page that queries index_ohlc. It does so
two ways, both written here:

  DAILY_ROLLUP_SQL   one row per trade_date: open (09:30 bar), high, low, and
                     close = the last non-null bar at or before 15:55, for SPX,
                     VIX, VIX3M, VIX9D; plus the PREVIOUS ROW's close. Never a
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


def _nn(col: str) -> str:
    return f"NULLIF({col}, 'NaN'::float8)"


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


_BAR_COLS = ",\n           ".join(f"{_nn(f'{s}_{f}')} AS {s}_{f}" for s in SERIES
                                  for f in ("open", "high", "low", "close"))

# One row per trade_date. Everything downstream reads this, never the bars.
# prev_* is the PREVIOUS ROW of the rollup -- the prior trading day -- not
# trade_date - 1, which is wrong across every weekend and holiday.
DAILY_ROLLUP_SQL = f"""
WITH bars AS (
    SELECT trade_date, quote_time,
           {_BAR_COLS}
    FROM index_ohlc
    WHERE quote_time BETWEEN TIME '09:30' AND TIME '15:55'
),
daily AS (
    SELECT trade_date,
           count(*) AS bar_count,
           {_daily_select()}
    FROM bars
    GROUP BY trade_date
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
    SELECT quote_time AS {s}_bar_time, {_nn(f'{s}_open')} AS {s}_entry
    FROM index_ohlc
    WHERE trade_date = t.d
      AND quote_time >= TIME '09:30'
      AND quote_time <= LEAST(t.tm, TIME '15:55')
      AND {_nn(f'{s}_open')} IS NOT NULL
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

FRESHNESS_SQL = """
SELECT max(trade_date) AS latest_date,
       (SELECT max(quote_time) FROM index_ohlc
         WHERE trade_date = (SELECT max(trade_date) FROM index_ohlc)) AS latest_time
FROM index_ohlc
"""

# How the table labels its bars, per year: a start-labeled session has a
# 09:30 bar and no valid 16:00 bar. An end-labeled source (or a 5-minute
# shift between the backfill and the live writer) shows up here as years with
# no 09:30 bars or with valid 16:00 bars.
BAR_LABEL_SQL = """
SELECT extract(year FROM trade_date)::int AS year,
       count(DISTINCT trade_date) AS days,
       count(DISTINCT trade_date) FILTER (WHERE quote_time = TIME '09:30') AS days_with_0930,
       count(DISTINCT trade_date) FILTER (WHERE quote_time = TIME '15:55') AS days_with_1555,
       count(DISTINCT trade_date) FILTER (WHERE quote_time = TIME '16:00'
             AND NULLIF(spx_close, 'NaN'::float8) IS NOT NULL) AS days_with_valid_1600,
       count(DISTINCT trade_date) FILTER (WHERE quote_time < TIME '09:30') AS days_with_premarket
FROM index_ohlc
GROUP BY 1 ORDER BY 1
"""


# ── daily rollup cache ──────────────────────────────────────────────────────
#
# The rollup scans every bar (~200k) and changes only when the writer adds
# bars, so it is cached per process and rebuilt when the table's latest
# (trade_date, quote_time) moves. A freshness probe is two index lookups.

_CACHE: dict = {"key": None, "daily": None, "labels": None, "built_at": None, "build_s": None}
_LOCK = asyncio.Lock()


async def get_daily(pool) -> tuple[pd.DataFrame, dict]:
    """(daily rollup DataFrame, freshness dict). Rebuilds when the table moved."""
    async with pool.acquire() as conn:
        fr = await conn.fetchrow(FRESHNESS_SQL)
    key = (fr["latest_date"], fr["latest_time"])
    fresh = {"latest_date": fr["latest_date"].isoformat() if fr["latest_date"] else None,
             "latest_time": fr["latest_time"].strftime("%H:%M:%S") if fr["latest_time"] else None}
    async with _LOCK:
        if _CACHE["key"] != key or _CACHE["daily"] is None:
            t0 = _time.monotonic()
            async with pool.acquire() as conn:
                rows = await conn.fetch(DAILY_ROLLUP_SQL)
                labels = await conn.fetch(BAR_LABEL_SQL)
            daily = pd.DataFrame([dict(r) for r in rows])
            _CACHE.update(key=key, daily=daily, labels=[dict(r) for r in labels],
                          built_at=datetime.now().isoformat(timespec="seconds"),
                          build_s=round(_time.monotonic() - t0, 2))
            rep = fallback_report(daily)
            log.info("oo-backtest daily rollup: %d days through %s in %.2fs; close fallback "
                     "early-close=%d full-session=%s", len(daily), key[0], _CACHE["build_s"],
                     len(rep["spx"]["early_close_days"]), {s: rep[s]["full_session_count"] for s in SERIES})
            for s in SERIES:
                if rep[s]["full_session_count"]:
                    log.warning("oo-backtest: %s close fell back on %d FULL sessions (e.g. %s) -- "
                                "expected only on early closes", s, rep[s]["full_session_count"],
                                rep[s]["full_session_sample"])
    fresh.update(built_at=_CACHE["built_at"], build_s=_CACHE["build_s"])
    return _CACHE["daily"], fresh


def bar_labels() -> list | None:
    return _CACHE["labels"]


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


def gap_crosscheck(df: pd.DataFrame, daily: pd.DataFrame, tol_pct: float = 0.02, tol_pts: float = 0.5) -> dict | None:
    """Option Omega's own Gap column vs the computed SPX gap.

    An off-by-one in the prior-row lookup is this phase's likeliest bug and is
    otherwise invisible, so the vendor column is used as an oracle. Its UNIT is
    not documented, so both are tried -- percent and index points -- and each
    against three alignments of the prior close: the previous row (what the
    page uses), two rows back, and the same day's close. The alignment that
    agrees best is reported; if it is not "previous row", something is wrong.
    """
    if "csv_gap" not in df.columns or df["csv_gap"].notna().sum() == 0 or daily.empty:
        return None
    d = daily.sort_values("trade_date").reset_index(drop=True)
    d["spx_prev2_close"] = d["spx_close"].shift(2)
    by_date = {r["trade_date"]: r for r in d.to_dict("records")}
    dates = pd.to_datetime(df["date_opened"]).dt.date
    vendor = df["csv_gap"].tolist()

    variants = {}
    for align, col in (("previous_row", "spx_prev_close"), ("two_rows_back", "spx_prev2_close"),
                       ("same_day_close", "spx_close")):
        for unit in ("pct", "points"):
            diffs = []
            for dt, v in zip(dates, vendor):
                r = by_date.get(dt)
                if r is None or v is None or pd.isna(v):
                    continue
                o, p = r.get("spx_open"), r.get(col)
                if o is None or p is None or pd.isna(o) or pd.isna(p):
                    continue
                calc = _pct(o, p) if unit == "pct" else float(o) - float(p)
                diffs.append((dt, float(v), calc))
            tol = tol_pct if unit == "pct" else tol_pts
            agree = sum(1 for _, v, c in diffs if abs(v - c) <= tol)
            variants[(align, unit)] = (agree, diffs)

    (align, unit), (agree, diffs) = max(variants.items(), key=lambda kv: (kv[1][0], kv[0][0] == "previous_row"))
    worst = sorted(diffs, key=lambda x: -abs(x[1] - x[2]))[:5]
    prev_row = {u: variants[("previous_row", u)][0] for u in ("pct", "points")}
    return {
        "compared": len(diffs),
        "best_alignment": align,
        "best_unit": unit,
        "agree": agree,
        "disagree": len(diffs) - agree,
        "agree_previous_row": prev_row,
        "worst": [{"date": dt.isoformat(), "vendor": round(v, 4), "computed": round(c, 4)} for dt, v, c in worst],
        "tolerance": tol_pct if unit == "pct" else tol_pts,
    }


async def join_market(pool, df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Add levels, ratios and gaps to a parsed trade log. Returns (df, report)."""
    daily, fresh = await get_daily(pool)
    ds, ts, keys, entry_info = entry_bar_requests(df)
    async with pool.acquire() as conn:
        rows = await conn.fetch(ENTRY_BAR_SQL, ds, ts) if ds else []
    entry_rows = [dict(r) for r in rows]
    out = apply_market(df, daily, entry_rows, keys)
    known = set(daily["trade_date"]) if not daily.empty else set()
    report = {
        "joined": True,
        "source": "main.index_ohlc",
        "freshness": fresh,
        "entry_time": entry_info,
        "entry_bars": entry_bar_report(out, keys),
        "gap_crosscheck": gap_crosscheck(out, daily),
        # Entry dates the table has no session for: before coverage, or a
        # date that is not a trading day at all (a timezone slip can do that).
        "trades_without_daily_row": int(sum(1 for d in pd.to_datetime(out["date_opened"]).dt.date
                                            if d not in known)),
    }
    return out, report
