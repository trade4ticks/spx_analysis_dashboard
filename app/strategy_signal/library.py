"""The metric library for Strategy Signal: what a strategy can name, and how
each one becomes a time series.

A METRIC IS ONE OF THREE THINGS, deliberately no more:

  a SOURCE        "surface:<column>" -- any ranked column of
                  surface_metrics_core, straight from the catalog
                  app/oo_backtest/surface.py already builds (levels, daily and
                  weekly changes, z-scores: the existing derived metrics come
                  with it). Or "index:<series>" -- the close of SPX, VIX,
                  VIX9D or VIX3M from index_ohlc.
  A op B          two sources and one of + - * /. Not nested, not a formula.
  a TRANSFORM     optional, on either of the above. One entry today
                  (trailing 252-session percentile); TRANSFORMS is where the
                  next one goes.

Anything more sophisticated belongs here as a new source or transform, not
in the page: ADDING A METRIC is one entry in INDEX_SERIES or TRANSFORMS (or a
new column in surface_metrics_core, which appears on its own via the catalog).

VALUES ARE RAW. Surface columns are not rescaled to vol points the way the
OO page's added rows are: thresholds here are typed from research on the raw
columns (term_ratio_7d_30d > 0.8375, skew in decimals), and a threshold that
means something different on this page than in the research is the mistake
that costs money.

ONE CLOCK FOR BOTH TABLES. surface_metrics_core is point-in-time at its
quote_time (09:35..16:00). index_ohlc is labeled by bar START, so a bar's
close is the price at start + 5 minutes. An index point is therefore stamped
at its bar's END: the 09:30 bar's close is the 09:35 value, the 15:55 bar's
close the 16:00 value -- the same moments the surface uses, so VIX / iv_30d
divides two numbers from the same instant.

DAILY is the last observation of each session: the session close for a
finished day, the latest bar for the session in progress. The daily chart's
last point is therefore the value the decision uses. For index series a PAST
session with too few bars for its own close (market_calendar.is_complete) is
dropped rather than letting a mid-day hole pass for a close -- the same rule
the OO page's rollup applies. market.get_daily is not reused for this: it
leaves out the session in progress (it is not complete yet), and it rebuilds
its whole-table rollup whenever a new bar lands, which on a page that
refreshes every five minutes is every refresh.
"""
from __future__ import annotations

import asyncio
from datetime import date, datetime, time, timedelta

import numpy as np
import pandas as pd

from app.oo_backtest import market, market_calendar as cal, surface

TZ = "America/New_York"

# index_ohlc series a strategy can name. The key is the column prefix in
# index_ohlc; nothing else reaches the SQL.
INDEX_SERIES = {
    "spx": {"label": "SPX", "description": "S&P 500 index"},
    "vix": {"label": "VIX", "description": "CBOE 30-day volatility index"},
    "vix9d": {"label": "VIX9D", "description": "CBOE 9-day volatility index"},
    "vix3m": {"label": "VIX3M", "description": "CBOE 3-month volatility index"},
}

OPERATORS = {"+": np.add, "-": np.subtract, "*": np.multiply, "/": np.divide}

TRANSFORMS = {
    "pctile_252": {
        "label": "Percentile, trailing 252 sessions",
        "short": "pctl 252",
        "window": 252,
        "description": "Where the value sits among the previous 252 session closes, 0-100. "
                       "Today is compared with history, never with itself; a point with "
                       "fewer than 252 prior closes has no value.",
    },
}

# Lookback presets, in exchange SESSIONS (the calendar's, not index_ohlc's).
LOOKBACKS = {"5d": 5, "10d": 10, "1m": 21, "3m": 63, "6m": 126, "1y": 252, "2y": 504}
LOOKBACK_LABELS = {"5d": "5 days", "10d": "10 days", "1m": "1 month", "3m": "3 months",
                   "6m": "6 months", "1y": "1 year", "2y": "2 years"}
# 3 months of 5-minute bars is ~4,900 points; more than that is a daily chart.
INTRADAY_LOOKBACKS = ("5d", "10d", "1m", "3m")
RESOLUTIONS = ("intraday", "daily")

# The decision uses the latest observation within this many sessions. Older
# than that and the strategy reads NO DATA rather than deciding on a stale value.
DECISION_SESSIONS = 5

BAR = timedelta(minutes=5)


def _empty() -> pd.Series:
    """No values -- still with a DatetimeIndex, so slicing by time works on it."""
    return pd.Series(dtype=float, index=pd.DatetimeIndex([]))


# ── ids and labels ──────────────────────────────────────────────────────────

def split_source(src: str) -> tuple[str, str]:
    kind, _, name = (src or "").partition(":")
    return kind, name


def source_label(src: str) -> str:
    kind, name = split_source(src)
    if kind == "index" and name in INDEX_SERIES:
        return INDEX_SERIES[name]["label"]
    return name or src


def metric_label(m: dict) -> str:
    """What the page calls a metric: its own label if it has one."""
    if (m.get("label") or "").strip():
        return m["label"].strip()
    s = source_label(m["a"])
    if m.get("op"):
        s = f"{s} {m['op']} {source_label(m['b'])}"
    if m.get("transform"):
        s = f"{s} · {TRANSFORMS[m['transform']]['short']}"
    return s


async def catalog(pool) -> dict:
    """Every source a strategy can name, for the page's picker and for the
    store's validation. The surface part is surface.get_catalog's cached
    answer, so the first call after a table change costs its index walk."""
    cat = await surface.get_catalog(pool)
    groups = {f: g["label"] for g in cat["family_groups"] for f in g["families"]}
    surf = [{"id": f"surface:{r['column_name']}", "column": r["column_name"],
             "family": r["family"], "group": groups.get(r["family"], cat["other_group"]["label"]),
             "form": r["form"], "form_label": cat["form_labels"].get(r["form"], r["form"]),
             "units": r["units"], "description": r["description"], "min_date": r["min_date"]}
            for r in cat["metrics"]]
    idx = [{"id": f"index:{k}", "label": v["label"], "description": v["description"]}
           for k, v in INDEX_SERIES.items()]
    return {"index": idx, "surface": surf, "surface_last_date": cat["last_date"]}


def valid_sources(cat: dict) -> set[str]:
    return {s["id"] for s in cat["index"]} | {s["id"] for s in cat["surface"]}


# ── sessions ────────────────────────────────────────────────────────────────

def now_et() -> pd.Timestamp:
    return pd.Timestamp.now(tz=TZ)


def recent_sessions(n: int, today: date) -> list[str]:
    """The last n exchange sessions on or before today, ascending."""
    span = int(n * 1.5) + 14
    while True:
        s = cal.sessions(today - timedelta(days=span), today)
        if len(s) >= n or span > 40000:
            return s[-n:]
        span *= 2


# ── fetching ────────────────────────────────────────────────────────────────

INDEX_BARS_SQL = """
SELECT trade_date, quote_time, {v} AS v
FROM index_ohlc
WHERE trade_date = ANY($1::date[])
  AND quote_time BETWEEN TIME '09:30' AND TIME '15:55'
ORDER BY trade_date, quote_time"""

SURFACE_BARS_SQL = """
SELECT trade_date, quote_time, {col} AS v
FROM {table}
WHERE trade_date >= $1 AND {col} IS NOT NULL
ORDER BY trade_date, quote_time"""


async def fetch_bars(pool, src: str, sessions: list[str]) -> pd.Series:
    """Intraday values for one source over these sessions, indexed by the
    MOMENT they describe (see the module docstring), ascending, no nulls."""
    kind, name = split_source(src)
    if not sessions:
        return _empty()
    days = [date.fromisoformat(d) for d in sessions]
    async with pool.acquire() as conn:
        if kind == "index":
            if name not in INDEX_SERIES:
                raise ValueError(f"unknown index series {name!r}")
            rows = await conn.fetch(INDEX_BARS_SQL.format(v=market._valid(f"{name}_close")), days)
            shift = BAR
        elif kind == "surface":
            rows = await conn.fetch(SURFACE_BARS_SQL.format(col=surface.quote_ident(name), table=surface.CORE),
                                    days[0])
            shift = timedelta(0)
        else:
            raise ValueError(f"unknown source {src!r}")
    stamps, vals = [], []
    for r in rows:
        if r["v"] is None:
            continue
        stamps.append(datetime.combine(r["trade_date"], r["quote_time"]) + shift)
        vals.append(float(r["v"]))
    s = pd.Series(vals, index=pd.DatetimeIndex(stamps), dtype=float)
    if kind == "index":
        s = _drop_short_sessions(s)
    return s


def _drop_short_sessions(s: pd.Series) -> pd.Series:
    """Drop PAST sessions with too few valid bars for their own close. The
    last session in the data is kept whatever its count: it may be the one in
    progress, and its latest bar is the most recent observation there is."""
    if s.empty:
        return s
    days = s.index.normalize()
    counts = pd.Series(1, index=days).groupby(level=0).sum()
    expected = cal.expected_bars(counts.index[0].date(), counts.index[-1].date())
    last = counts.index[-1]
    keep = {d for d, n in counts.items()
            if d == last or cal.is_complete(int(n), expected.get(d.date().isoformat(), 0))}
    return s[[d in keep for d in days]]


def to_daily(bars: pd.Series) -> pd.Series:
    """Last observation per session, indexed by the session's date."""
    if bars.empty:
        return bars
    return bars.groupby(bars.index.normalize()).last()


def combine(a: pd.Series, op: str | None, b: pd.Series | None) -> pd.Series:
    """A op B on the moments both have. Division by zero is no value."""
    if not op:
        return a
    ja, jb = a.align(b, join="inner")
    with np.errstate(divide="ignore", invalid="ignore"):
        out = pd.Series(OPERATORS[op](ja.to_numpy(), jb.to_numpy()), index=ja.index)
    return out[np.isfinite(out.to_numpy())]


def pctile(points: pd.Series, daily: pd.Series, window: int) -> pd.Series:
    """Each point's percentile among the `window` session closes BEFORE its
    own session. (below + half the ties) / window x 100. NaN without a full
    window: a 60-session percentile labelled as a 252-session one is a
    different number wearing the same name."""
    if points.empty:
        return points
    hist = daily.to_numpy()
    hist_days = daily.index.to_numpy()
    out = np.full(len(points), np.nan)
    pdays = points.index.normalize()
    vals = points.to_numpy()
    for d in pd.unique(pdays):
        k = int(np.searchsorted(hist_days, np.datetime64(d), side="left"))   # closes strictly before d
        if k < window:
            continue
        w = np.sort(hist[k - window:k])
        sel = np.flatnonzero(pdays == d)
        v = vals[sel]
        lo = np.searchsorted(w, v, side="left")
        hi = np.searchsorted(w, v, side="right")
        out[sel] = (lo + 0.5 * (hi - lo)) / window * 100.0
    return pd.Series(out, index=points.index).dropna()


# ── one request's worth of series ───────────────────────────────────────────

class Board:
    """Every series a set of strategies needs, fetched ONCE per source.

    Two passes: need() records how far back each source must reach, load()
    fetches each source once for its deepest need, then series() and latest()
    are pure pandas over what was fetched."""

    def __init__(self, today: date):
        self.today = today
        self.depth: dict[str, int] = {}
        self.bars: dict[str, pd.Series] = {}
        self.errors: dict[str, str] = {}

    @staticmethod
    def _sources(m: dict) -> list[str]:
        return [m["a"]] + ([m["b"]] if m.get("op") else [])

    @staticmethod
    def _sessions_needed(m: dict, lookback_sessions: int) -> int:
        extra = TRANSFORMS[m["transform"]]["window"] + 10 if m.get("transform") else 0
        return lookback_sessions + extra

    def need(self, m: dict) -> None:
        n = max(DECISION_SESSIONS, LOOKBACKS[m["lookback"]] if m.get("chart") else 0)
        n = self._sessions_needed(m, n)
        for src in self._sources(m):
            self.depth[src] = max(self.depth.get(src, 0), n)

    async def load(self, pool) -> None:
        async def one(src, n):
            try:
                self.bars[src] = await fetch_bars(pool, src, recent_sessions(n, self.today))
            except Exception as exc:                          # noqa: BLE001 -- reported per metric
                self.errors[src] = f"{type(exc).__name__}: {exc}"
        await asyncio.gather(*(one(s, n) for s, n in self.depth.items()))

    def _base(self, m: dict, since: pd.Timestamp) -> pd.Series:
        for src in self._sources(m):
            if src in self.errors:
                raise RuntimeError(f"{source_label(src)}: {self.errors[src]}")
        a = self.bars[m["a"]] if len(self.bars[m["a"]]) else _empty()
        b = (self.bars[m["b"]] if len(self.bars[m["b"]]) else _empty()) if m.get("op") else None
        a = a[a.index >= since]
        if b is not None:
            b = b[b.index >= since]
        return combine(a, m.get("op"), b)

    def series(self, m: dict, resolution: str, sessions: int) -> pd.Series:
        """The metric at a resolution over the last `sessions` sessions."""
        days = recent_sessions(sessions, self.today)
        start = pd.Timestamp(days[0])
        if m.get("transform"):
            win = TRANSFORMS[m["transform"]]["window"]
            hist_days = recent_sessions(sessions + win + 10, self.today)
            daily = to_daily(self._base(m, pd.Timestamp(hist_days[0])))
            base = self._base(m, start)
            pts = base if resolution == "intraday" else to_daily(base)
            return pctile(pts, daily, win)
        base = self._base(m, start)
        return base if resolution == "intraday" else to_daily(base)

    def latest(self, m: dict) -> tuple[float | None, pd.Timestamp | None]:
        """The most recent observation, from the last DECISION_SESSIONS sessions."""
        s = self.series(m, "intraday", DECISION_SESSIONS)
        if s.empty:
            return None, None
        return float(s.iloc[-1]), s.index[-1]


def stamp(ts: pd.Timestamp, resolution: str) -> str:
    return ts.strftime("%Y-%m-%d" if resolution == "daily" else "%Y-%m-%d %H:%M")


def expected_latest_session(now: pd.Timestamp) -> str | None:
    """The session whose data should be the latest by now: today once it has
    had time to print a bar (09:40 ET), else the previous session."""
    today = now.date()
    if cal.is_session(today) and now.time() >= time(9, 40):
        return today.isoformat()
    prev = recent_sessions(1, today - timedelta(days=1))
    return prev[-1] if prev else None
