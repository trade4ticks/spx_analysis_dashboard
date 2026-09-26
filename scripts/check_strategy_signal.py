"""Gate: Strategy Signal's decision, metric series and config store. Offline.

No database and no network: the board's fetch is replaced with fabricated
bars, so this runs anywhere (the VPS included). What it pins:

  decide      weekday first; AND/OR; one threshold per level, first level
              that holds wins; a MISSING value is unknown, not false, and
              an unknown level reads NO DATA instead of falling through.
  pctile      strictly-prior window, ties counted half, no partial window.
  the clock   an index bar is stamped at its END, so VIX / surface divides
              values from the same instant; A op B is an inner join.
  daily       last observation per session; a short PAST index session is
              dropped, the last session is kept (it may be in progress).
  Board       each source fetched ONCE for its deepest need.
  store       clean_config refuses what the page could not use.
"""
from __future__ import annotations

import asyncio
import sys
from datetime import date, datetime, time, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.strategy_signal import evaluate as ev, library as lib, store  # noqa: E402

FAILS: list[str] = []


def check(ok: bool, msg: str) -> None:
    print(("  ok    " if ok else "  FAIL  ") + msg)
    if not ok:
        FAILS.append(msg)


def cfg(states, logic, metrics, days=(1, 2, 3, 4, 5)):
    return {"states": states, "logic": logic, "weekdays": list(days), "metrics": metrics}


def sig(mid, cmp, *th):
    return {"id": mid, "signal": True, "cmp": cmp, "thresholds": list(th)}


def check_decide():
    print("decide")
    b = cfg(["TRADE", "NO TRADE"], "and", [sig("r", "<", 1.14), sig("t", ">", 0.8375)])
    check(ev.decide(b, {"r": 1.0, "t": 0.9}, 2)["state"] == 0, "AND: both pass -> TRADE")
    check(ev.decide(b, {"r": 1.2, "t": 0.9}, 2)["state"] == 1, "AND: one fails -> NO TRADE")
    d = ev.decide(b, {"r": 1.2, "t": None}, 2)
    check(d["state"] == 1 and d["reason"] == "conditions", "AND: a false beside a missing is still false")
    d = ev.decide(b, {"r": 1.0, "t": None}, 2)
    check(d["state"] is None and d["reason"] == "no_data", "AND: a pass beside a missing is NO DATA")

    fri = cfg(["TRADE", "NO TRADE"], "and", [sig("r", "<", 1.14)], days=(5,))
    d = ev.decide(fri, {"r": 1.0}, 2)
    check(d["state"] == 1 and d["reason"] == "weekday", "Friday-only on a Tuesday -> NO TRADE though every condition passes")
    check(ev.decide(fri, {"r": 1.0}, 5)["state"] == 0, "Friday-only on a Friday -> TRADE")
    check(ev.decide(fri, {"r": None}, 6)["reason"] == "weekday", "a Saturday is not an entry day, data or not")

    o = cfg(["TRADE", "NO TRADE"], "or", [sig("r", "<", 1.14), sig("t", ">", 0.8375)])
    check(ev.decide(o, {"r": 1.2, "t": 0.9}, 1)["state"] == 0, "OR: one passes -> TRADE")
    check(ev.decide(o, {"r": 1.0, "t": None}, 1)["state"] == 0, "OR: a pass beside a missing is TRADE")
    check(ev.decide(o, {"r": 1.2, "t": None}, 1)["state"] is None, "OR: a fail beside a missing is NO DATA")
    check(ev.decide(o, {"r": 1.2, "t": 0.5}, 1)["state"] == 1, "OR: both fail -> NO TRADE")

    a = cfg(["FULL", "REDUCED", "MINIMAL", "NONE"], "and", [sig("p", "<", 30, 50, 70)])
    got = [ev.decide(a, {"p": v}, 3)["state"] for v in (10, 30, 45, 69.9, 70, 95)]
    check(got == [0, 1, 1, 2, 3, 3], f"tiers: pctl 10/30/45/69.9/70/95 -> FULL/RED/RED/MIN/NONE/NONE (got {got})")
    a2 = cfg(["FULL", "REDUCED", "NONE"], "and", [sig("p", "<", 30, 60), sig("q", ">", 1.0, 0.9)])
    check(ev.decide(a2, {"p": 20, "q": 0.95}, 3)["state"] == 1, "tiers with AND: level 1 fails on q, level 2 holds")
    check(ev.decide(cfg(["GO", "NO"], "and", []), {}, 3)["state"] == 0, "no conditions: an entry day is the first state")
    check(ev.compare(0.1 + 0.2, "=", 0.3) is True, "= tolerates float noise")


def check_pctile():
    print("pctile")
    days = pd.DatetimeIndex(pd.bdate_range("2024-01-01", periods=300))
    daily = pd.Series(np.arange(300, dtype=float), index=days)
    pts = pd.Series([1000.0, -1.0, 150.0], index=[days[260], days[260], days[299]])
    p = lib.pctile(pts, daily, 252)
    check(abs(p.iloc[0] - 100) < 1e-9 and abs(p.iloc[1]) < 1e-9, "above every prior close = 100, below = 0")
    # On days[299] the window is closes 47..298: 150 is above 47..149 (103) and ties itself (half).
    check(abs(p.iloc[2] - 103.5 / 252 * 100) < 1e-9, "the window is the 252 closes BEFORE the point's session")
    same_day = pd.Series([daily.iloc[280]], index=[days[280] + pd.Timedelta(hours=16)])
    check(lib.pctile(same_day, daily, 252).iloc[0] == 100.0, "a session's own close is not in its window")
    ties = pd.Series([5.0] * 300, index=days)
    check(lib.pctile(pd.Series([5.0], index=[days[299]]), ties, 252).iloc[0] == 50.0, "ties count half")
    check(lib.pctile(pd.Series([1.0], index=[days[100]]), daily, 252).empty, "fewer than 252 prior closes -> no value")


def check_clock_and_daily():
    print("clock, combine, daily")
    d = datetime(2026, 9, 25)
    # index bar 09:30 closes at 09:35; surface row 09:35 is the 09:35 value
    idx = pd.Series([20.0, 21.0], index=[d + timedelta(hours=9, minutes=35), d + timedelta(hours=16)])
    srf = pd.Series([10.0, 7.0], index=[d + timedelta(hours=9, minutes=35), d + timedelta(hours=16)])
    r = lib.combine(idx, "/", srf)
    check(list(r.round(6)) == [2.0, 3.0], "A / B pairs values from the same moment")
    r = lib.combine(idx, "/", srf.iloc[:1])
    check(len(r) == 1, "A op B keeps only moments both have")
    z = lib.combine(idx, "/", pd.Series([0.0, 7.0], index=srf.index))
    check(len(z) == 1, "division by zero is no value, not inf")
    check(lib.stamp(idx.index[1], "intraday") == "2026-09-25 16:00", "the 15:55 bar is stamped 16:00")

    # a full past session, a half-empty past session, and a partial last one
    s1, s2, s3 = date(2026, 9, 22), date(2026, 9, 23), date(2026, 9, 24)
    def bars(day, n):
        t0 = datetime.combine(day, time(9, 35))
        return [(t0 + lib.BAR * k, float(k)) for k in range(n)]
    rows = bars(s1, 78) + bars(s2, 30) + bars(s3, 12)
    ser = pd.Series([v for _, v in rows], index=pd.DatetimeIndex([t for t, _ in rows]))
    kept = lib._drop_short_sessions(ser)
    kd = sorted({t.date() for t in kept.index})
    check(kd == [s1, s3], f"short PAST session dropped, last session kept (got {kd})")
    daily = lib.to_daily(kept)
    check(list(daily) == [77.0, 11.0], "daily = last observation per session")


class FakeBoard(lib.Board):
    calls: list = []


def check_board():
    print("board")
    calls = []
    today = date(2026, 9, 25)
    sessions = lib.recent_sessions(600, today)

    async def fake_fetch(pool, src, sess):
        calls.append((src, len(sess)))
        idx, vals = [], []
        for i, d in enumerate(sess):
            base = datetime.fromisoformat(d)
            for k in range(3):
                idx.append(base + timedelta(hours=10, minutes=5 * k))
                vals.append(10.0 + i + k / 10 if src == "index:vix" else 5.0)
        return pd.Series(vals, index=pd.DatetimeIndex(idx))

    real = lib.fetch_bars
    lib.fetch_bars = fake_fetch
    try:
        b = lib.Board(today)
        m1 = {"id": "a", "a": "index:vix", "op": "/", "b": "index:vix9d", "transform": None,
              "chart": True, "resolution": "intraday", "lookback": "10d", "signal": True}
        m2 = {"id": "b", "a": "index:vix", "op": None, "b": None, "transform": "pctile_252",
              "chart": True, "resolution": "daily", "lookback": "1y", "signal": True}
        for m in (m1, m2, dict(m1, id="c")):
            b.need(m)
        asyncio.run(b.load(None))
        check(sorted(c[0] for c in calls) == ["index:vix", "index:vix9d"], "each source fetched once")
        check(dict(calls)["index:vix"] == 252 + 252 + 10, "VIX fetched for its deepest need (1y + 252 + 10)")
        v, ts = b.latest(m1)
        check(ts.date() == today and abs(v - (10.0 + 513 + 0.2) / 5.0) < 1e-9, "latest = last bar of the last session")
        s = b.series(m1, "intraday", 10)
        check(len(s) == 30 and len({t.date() for t in s.index}) == 10, "10 sessions of intraday points")
        p = b.series(m2, "daily", 252)
        check(len(p) == 252 and p.iloc[-1] == 100.0, "a rising series sits at the 100th percentile")
    finally:
        lib.fetch_bars = real


def check_store():
    print("store")
    src = {"index:vix", "index:vix9d", "surface:term_ratio_7d_30d"}
    ok = {"name": "  Short   term ", "notes": "why", "weekdays": [5, 1, 5], "states": ["TRADE", "NO TRADE"],
          "logic": "and", "manual": ["Premium must be >= $2.50", "  "],
          "metrics": [{"a": "index:vix", "op": "/", "b": "index:vix9d", "chart": True, "resolution": "intraday",
                       "lookback": "10d", "signal": True, "cmp": "<", "thresholds": [1.14]},
                      {"a": "surface:term_ratio_7d_30d", "chart": True, "signal": True, "cmp": ">",
                       "thresholds": ["0.8375"]}]}
    c = store.clean_config(ok, src)
    check(c["name"] == "Short term" and c["weekdays"] == [1, 5], "name squeezed, weekdays deduplicated and sorted")
    check(c["manual"] == ["Premium must be >= $2.50"], "blank manual requirements dropped")
    check(c["metrics"][1]["thresholds"] == [0.8375] and c["metrics"][1]["resolution"] == "daily",
          "thresholds become numbers; resolution defaults to daily")
    check(len({m["id"] for m in c["metrics"]}) == 2, "metrics get distinct ids")

    def refused(mut, why):
        import copy
        bad = copy.deepcopy(ok)
        mut(bad)
        try:
            store.clean_config(bad, src)
            check(False, f"refuses {why}")
        except ValueError:
            check(True, f"refuses {why}")
    refused(lambda x: x["metrics"][0].update(a="surface:not_a_column"), "an unknown metric")
    refused(lambda x: x["metrics"][0].update(op="^"), "an operator outside + - * /")
    refused(lambda x: x["metrics"][0].update(thresholds=[1, 2]), "the wrong number of thresholds")
    refused(lambda x: x["metrics"][0].update(lookback="1y"), "a 1-year INTRADAY chart")
    refused(lambda x: x["metrics"][0].update(chart=False, signal=False), "a metric that does nothing")
    refused(lambda x: x.update(weekdays=[]), "no entry days")
    refused(lambda x: x.update(weekdays=[6]), "a Saturday entry day")
    refused(lambda x: x.update(states=["A"]), "a single state")
    refused(lambda x: x.update(states=["GO", "go"]), "two states with one name")
    refused(lambda x: x["metrics"][1].update(thresholds=["abc"]), "a threshold that is not a number")


def main() -> int:
    check_decide()
    check_pctile()
    check_clock_and_daily()
    check_board()
    check_store()
    print()
    if FAILS:
        print(f"FAIL: {len(FAILS)} strategy-signal check(s) failed")
        return 1
    print("PASS: strategy signal — decision, percentile, clock, board fetch, config store")
    return 0


if __name__ == "__main__":
    sys.exit(main())
