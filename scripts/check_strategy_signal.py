"""Gate: Strategy Signal's decision, metric series and config store. Offline.

No database and no network: the board's fetch is replaced with fabricated
bars, so this runs anywhere (the VPS included). What it pins:

  decide      the SCHEDULE NEVER DECIDES (2026-10-16): a Wednesday strategy
              reads TRADE on a Thursday or a Saturday when its conditions
              pass; AND/OR; one threshold per level, first level that holds
              wins; a MISSING value is unknown, not false, and an unknown
              level reads NO DATA instead of falling through.
  pctile      strictly-prior window, ties counted half, no partial window.
  the clock   an index bar is stamped at its END, so VIX / surface divides
              values from the same instant; A op B is an inner join.
  daily       last observation per session; a short PAST index session is
              dropped, the last session is kept (it may be in progress).
  Board       each source fetched ONCE for its deepest need.
  store       clean_config refuses what the page could not use; entry
              times and display are validated; a row saved before they
              existed reads back with its days and the defaults.
  images      prepare() keeps the aspect ratio, never enlarges, refuses a
              file that is not an image whatever its name says.
"""
from __future__ import annotations

import asyncio
import copy
import io
import json
import sys
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.strategy_signal import evaluate as ev, images, library as lib, store  # noqa: E402

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

    # The schedule is information. A Wednesday strategy whose conditions
    # pass on Thursday reads TRADE on Thursday.
    wed = cfg(["TRADE", "NO TRADE"], "and", [sig("r", "<", 1.14)], days=(3,))
    d = ev.decide(wed, {"r": 1.2}, 3)
    check(d["state"] == 1 and d["scheduled_today"] is True,
          "Wednesday strategy on Wednesday, conditions fail -> NO TRADE (a scheduled day)")
    d = ev.decide(wed, {"r": 1.0}, 4)
    check(d["state"] == 0 and d["reason"] == "conditions" and d["scheduled_today"] is False,
          "Wednesday strategy on Thursday, conditions pass -> TRADE (not scheduled, not blocked)")
    check(ev.decide(wed, {"r": 1.0}, 5)["state"] == 0, "... and on Friday -> TRADE")
    check(ev.decide(wed, {"r": 1.0}, 6)["state"] == 0, "... and on a Saturday -> TRADE: the weekday is never a condition")
    check(ev.decide(wed, {"r": None}, 4)["reason"] == "no_data", "a missing value off-schedule is NO DATA, not NO TRADE")
    d = ev.decide(wed, {"r": 1.0}, 4)
    check(d["market_state"] == d["state"] and d["market_reason"] == d["reason"], "market_state is the decision itself")
    nod = cfg(["TRADE", "NO TRADE"], "and", [sig("r", "<", 1.14)], days=())
    d = ev.decide(nod, {"r": 1.0}, 2)
    check(d["state"] == 0 and d["scheduled_today"] is False, "no scheduled days at all -> decided by the conditions")
    check(ev.decide(wed, {"r": 1.0})["state"] == 0, "the weekday argument is optional")

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
    check(ev.decide(cfg(["GO", "NO"], "and", []), {}, 3)["state"] == 0, "no conditions: the first state")
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


def check_board():
    print("board")
    calls = []
    today = date(2026, 9, 25)

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


class Rec(dict):
    """An asyncpg Record's shape: [key] and .keys()."""


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
    check(c["entry_times"] == [] and c["display"] == "calendar", "no times and the calendar by default")

    t = copy.deepcopy(ok)
    t.update(weekdays=[5], display="card",
             entry_times=[{"day": 5, "time": "11:30"}, {"day": 2, "time": "9:30"}, {"day": 2, "time": "10:00"},
                          {"day": 2, "time": "10:00"}, {"day": 3, "time": "16:00"}])
    c = store.clean_config(t, src)
    check(c["entry_times"] == [{"day": 2, "time": "09:30"}, {"day": 2, "time": "10:00"},
                               {"day": 3, "time": "16:00"}, {"day": 5, "time": "11:30"}],
          "entry times: padded, deduplicated, sorted; several on one day; 09:30 and 16:00 allowed")
    check(c["weekdays"] == [2, 3, 5], "a day with a time is a scheduled day")
    check(c["display"] == "card", "standalone card kept")
    none = copy.deepcopy(ok)
    none["weekdays"] = []
    check(store.clean_config(none, src)["weekdays"] == [], "no scheduled days is allowed (a standalone strategy)")

    # A row saved BEFORE entry_times / display existed, as asyncpg returns it.
    legacy = {"name": "Old", "notes": "", "weekdays": [3], "states": ["TRADE", "NO TRADE"], "logic": "and",
              "metrics": [], "manual": []}
    r = store._row(Rec(id=7, position=2, name="Old", config=json.dumps(legacy),
                       updated_at=datetime(2026, 9, 26, tzinfo=timezone.utc)))
    check(r["weekdays"] == [3] and r["entry_times"] == [] and r["display"] == "calendar" and r["image"] is None,
          "a legacy row keeps its weekday as a scheduled day, gains no times, sits on the calendar")
    check(store.clean_config(r, src)["weekdays"] == [3], "a legacy row re-saves with its day")
    at = datetime(2026, 10, 1, tzinfo=timezone.utc)
    r = store._row(Rec(id=7, position=2, name="Old", config=legacy, updated_at=None, image_at=at,
                       image_w=800, image_h=500, image_tw=480, image_th=300))
    check(r["image"] == {"v": int(at.timestamp() * 1000), "w": 800, "h": 500, "tw": 480, "th": 300},
          "an image's meta rides on the strategy, versioned by its upload time")

    def refused(mut, why):
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
    refused(lambda x: x.update(weekdays=[6]), "a Saturday scheduled day")
    refused(lambda x: x.update(entry_times=[{"day": 2, "time": "09:00"}]), "an entry time before the open")
    refused(lambda x: x.update(entry_times=[{"day": 2, "time": "16:05"}]), "an entry time after the close")
    refused(lambda x: x.update(entry_times=[{"day": 2, "time": "10:61"}]), "a time that is not a time")
    refused(lambda x: x.update(entry_times=[{"day": 2, "time": "ten"}]), "a time that is a word")
    refused(lambda x: x.update(entry_times=[{"day": 6, "time": "10:00"}]), "an entry time on a Saturday")
    refused(lambda x: x.update(entry_times=[{"day": 1, "time": f"10:{i:02d}"} for i in range(21)]),
            "more than 20 entry times")
    refused(lambda x: x.update(display="sidebar"), "a display that is neither calendar nor card")
    refused(lambda x: x.update(states=["A"]), "a single state")
    refused(lambda x: x.update(states=["GO", "go"]), "two states with one name")
    refused(lambda x: x["metrics"][1].update(thresholds=["abc"]), "a threshold that is not a number")


T1_TRADE = {
    "v": 1, "entry_notes": "No Monday entry: the 6-DTE expiration is not listed.\nNo Friday entry: no 1-DTE.",
    "legs": [
        {"id": "sc", "side": "sell", "right": "call", "qty": 1, "dte": 1, "strike": {"method": "delta", "value": 30}},
        {"id": "sp", "side": "sell", "right": "put", "qty": 1, "dte": 1, "strike": {"method": "delta", "value": 30}},
        {"id": "lc", "side": "buy", "right": "call", "qty": 1, "dte": 6, "strike": {"method": "same", "ref": "sc"}},
        {"id": "lp", "side": "buy", "right": "put", "qty": 1, "dte": 6, "strike": {"method": "same", "ref": "sp"}}],
    "target": {"value": 25, "unit": "pct", "basis": ""},
    "stop": {"value": 10, "unit": "pct", "basis": ""},
    "exit": {"max_dit": None, "exit_dte": 0, "exit_time": "10:00", "rules": "the next trading morning"},
    "management": ""}


def check_trade():
    print("trade definition")
    src = {"index:vix"}
    base = {"name": "T-1", "notes": "", "weekdays": [2, 3, 4], "states": ["TRADE", "NO TRADE"], "logic": "and",
            "entry_times": [{"day": d, "time": "15:00"} for d in (2, 3, 4)],
            "metrics": [{"a": "index:vix", "chart": True, "signal": True, "cmp": "<", "thresholds": [30]}],
            "manual": ["Premium >= $2.00"]}
    c = store.clean_config({**base, "trade": T1_TRADE}, src)
    t = c["trade"]
    check(t is not None and t["v"] == 1 and [lg["id"] for lg in t["legs"]] == ["sc", "sp", "lc", "lp"],
          "the T-1 example saves: four legs, ids kept")
    check(t["legs"][2]["strike"] == {"method": "same", "value": None, "ref": "sc"}
          and t["legs"][0]["strike"] == {"method": "delta", "value": 30.0, "ref": None},
          "same-strike holds the leg id; delta holds its magnitude")
    check([lg["dte"] for lg in t["legs"]] == [1, 1, 6, 6], "each leg keeps its own DTE")
    check(t["target"] == {"value": 25.0, "unit": "pct", "basis": ""} and t["exit"]["exit_dte"] == 0
          and t["exit"]["exit_time"] == "10:00", "target, stop and exit kept; exit DTE 0 is a value, not empty")
    check(c["metrics"][0]["thresholds"] == [30.0] and c["manual"] == ["Premium >= $2.00"]
          and c["entry_times"][0] == {"day": 2, "time": "15:00"},
          "the definition rides beside metrics, manual checks and schedule, changing none")
    check(store.clean_config(json.loads(json.dumps(c)), src)["trade"] == t, "re-saving what was read gives the same definition")
    check(store.clean_config({**base, "trade": None}, src)["trade"] is None
          and store.clean_config(base, src)["trade"] is None, "no definition is valid")
    blank = {"entry_notes": "  ", "legs": [], "target": {"value": None, "unit": "pct", "basis": ""},
             "stop": {"value": "", "unit": "usd", "basis": ""},
             "exit": {"max_dit": "", "exit_dte": None, "exit_time": "", "rules": ""}, "management": ""}
    check(store.clean_config({**base, "trade": blank}, src)["trade"] is None, "an all-empty definition is stored as none")
    only_basis = store.clean_trade({"stop": {"value": None, "unit": "pct", "basis": "close if the shorts are breached"}})
    check(only_basis["stop"] == {"value": None, "unit": "pct", "basis": "close if the shorts are breached"},
          "a stop described in words alone is kept")
    check(store.clean_trade({"exit": {"exit_time": "17:30"}})["exit"]["exit_time"] == "17:30",
          "an exit time is any ET time (not only the entry session)")
    check(store.clean_trade({"legs": [{"side": "buy", "right": "put", "qty": "2", "dte": "0",
                                       "strike": {"method": "offset", "value": "-50"}}]})["legs"][0]
          == {"id": "L1", "side": "buy", "right": "put", "qty": 2, "dte": 0,
              "strike": {"method": "offset", "value": -50.0, "ref": None}, "note": ""},
          "numbers typed as text are accepted; a leg without an id gets one; 0 DTE allowed")
    atm = store.clean_trade({"legs": [{"id": "a", "side": "sell", "right": "call", "qty": 1, "dte": 7,
                                       "strike": {"method": "atm", "value": 99}}]})
    check(atm["legs"][0]["strike"] == {"method": "atm", "value": None, "ref": None}, "ATM keeps no stray value")
    r = store._row(Rec(id=1, position=1, name="Old", config={"name": "Old", "weekdays": [], "states": ["A", "B"],
                                                             "logic": "and", "metrics": [], "manual": []}, updated_at=None))
    check(r["trade"] is None, "a row saved before trade definitions reads back with none")

    def legs_with(mut):
        tr = copy.deepcopy(T1_TRADE)
        mut(tr["legs"])
        return tr

    def refused(tr, why):
        try:
            store.clean_trade(tr)
            check(False, f"refuses {why}")
        except ValueError:
            check(True, f"refuses {why}")
    refused(legs_with(lambda L: L[0].update(qty=0)), "a zero quantity")
    refused(legs_with(lambda L: L[0].update(qty=1.5)), "a fractional quantity")
    refused(legs_with(lambda L: L[0].update(dte=-1)), "a negative DTE")
    refused(legs_with(lambda L: L[0].update(dte=None)), "a leg without a DTE")
    refused(legs_with(lambda L: L[0].update(side="short")), "a side other than buy/sell")
    refused(legs_with(lambda L: L[0].update(right="straddle")), "a right other than call/put")
    refused(legs_with(lambda L: L[0]["strike"].update(value=0)), "a 0 delta")
    refused(legs_with(lambda L: L[0]["strike"].update(value=100)), "a 100 delta")
    refused(legs_with(lambda L: L[0]["strike"].update(value=None)), "a delta leg without its delta")
    refused(legs_with(lambda L: L[0].update(strike={"method": "offset"})), "a spot offset without its points")
    refused(legs_with(lambda L: L[0].update(strike={"method": "nearest"})), "an unknown strike method")
    refused(legs_with(lambda L: L[2].update(strike={"method": "same", "ref": None})), "same-strike with no leg chosen")
    refused(legs_with(lambda L: L.pop(0)), "same-strike pointing at a removed leg")
    refused(legs_with(lambda L: L[2].update(strike={"method": "same", "ref": "lc"})), "a leg sharing its own strike")
    refused(legs_with(lambda L: (L[0].update(strike={"method": "same", "ref": "lc"}))), "a circle of same-strike references")
    refused(legs_with(lambda L: L[1].update(id="sc")), "two legs with one id")
    refused(legs_with(lambda L: L.extend(copy.deepcopy(L[0]) | {"id": f"x{i}"} for i in range(5))), "more than 8 legs")
    refused({**T1_TRADE, "target": {"value": 25, "unit": "R"}}, "a target unit other than % / $ / points")
    refused({**T1_TRADE, "exit": {"exit_time": "25:00"}}, "an exit time that is not a time")
    refused({**T1_TRADE, "exit": {"max_dit": -3}}, "a negative maximum DIT")

    # Reordering is moving dicts: the reference is an id, so it follows.
    moved = copy.deepcopy(T1_TRADE)
    moved["legs"] = [moved["legs"][i] for i in (2, 0, 3, 1)]
    t2 = store.clean_trade(moved)
    check(t2["legs"][0]["strike"]["ref"] == "sc" and t2["legs"][1]["id"] == "sc",
          "reordered legs keep their references (the long call still shares the short call's strike)")

    # The decision never reads the definition, schedule or instructions.
    with_t = cfg(["TRADE", "NO TRADE"], "and", [sig("r", "<", 1.14)], days=(3,))
    for vals, wd in (({"r": 1.0}, 4), ({"r": 1.2}, 3), ({"r": None}, 2)):
        check(ev.decide({**with_t, "trade": T1_TRADE}, vals, wd) == ev.decide(with_t, vals, wd),
              f"decide is identical with and without a trade definition ({vals}, day {wd})")


def check_images():
    print("images")
    try:
        from PIL import Image
    except ImportError:
        check(False, "Pillow is importable (needed to test the risk-graph upload)")
        return

    def pic(w, h, fmt="PNG", mode="RGB"):
        b = io.BytesIO()
        Image.new(mode, (w, h), (52, 152, 219) if mode == "RGB" else (52, 152, 219, 128)).save(b, fmt)
        return b.getvalue()
    p = images.prepare(pic(1600, 600))
    check(p["content_type"] == "image/png" and (p["width"], p["height"]) == (1600, 600), "the original is kept as uploaded")
    check((p["thumb_width"], p["thumb_height"]) == (480, 180) and p["thumb_type"] == "image/webp",
          "the thumbnail fits 480x300 and keeps the aspect ratio (1600x600 -> 480x180)")
    check(Image.open(io.BytesIO(p["thumb"])).size == (480, 180), "the thumbnail bytes are that size")
    p = images.prepare(pic(200, 100, "JPEG"))
    check((p["thumb_width"], p["thumb_height"]) == (200, 100) and p["content_type"] == "image/jpeg",
          "a small image is never enlarged")
    p = images.prepare(pic(300, 900, "PNG", "RGBA"))
    check((p["thumb_width"], p["thumb_height"]) == (100, 300), "a tall image fits by its height")
    check(images.prepare(pic(10, 10, "GIF"))["content_type"] == "image/gif", "a GIF is accepted")
    for bad, why in ((b"", "an empty file"), (b"not an image at all", "text named .png"),
                     (b"\x89PNG\r\n\x1a\n" + b"0" * 40, "a truncated PNG"),
                     (b"x" * (images.MAX_BYTES + 1), "a file over 8 MB"),
                     (pic(4, 4, "BMP"), "a BMP (not on the list)")):
        try:
            images.prepare(bad)
            check(False, f"refuses {why}")
        except ValueError:
            check(True, f"refuses {why}")


def main() -> int:
    check_decide()
    check_pctile()
    check_clock_and_daily()
    check_board()
    check_store()
    check_trade()
    check_images()
    print()
    if FAILS:
        print(f"FAIL: {len(FAILS)} strategy-signal check(s) failed")
        return 1
    print("PASS: strategy signal — decision, percentile, clock, board fetch, config store, schedule fields, trade definitions, images")
    return 0


if __name__ == "__main__":
    sys.exit(main())
