"""The exchange calendar: the dates it calls sessions, and the bars it expects.

WHAT THIS PROTECTS. The calendar is now the authority on whether a day was a
session, so a wrong answer here is wrong everywhere downstream -- the rollup,
the deployment axis, coverage, the artifact report and the freshness check.
The cases below are the ones that decided the design: the day the old rule
got wrong, a day it got right for the wrong reason, and a half-day a single
bar floor cannot see.

No network and no database. Runs on the VPS.
"""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

FAILURES: list[str] = []


def check(cond, msg: str) -> None:
    if cond:
        print(f"  ok    {msg}")
    else:
        print(f"  FAIL  {msg}")
        FAILURES.append(msg)


def main() -> int:
    try:
        from app.oo_backtest import market_calendar as mc
    except ImportError as exc:
        print(f"  pandas_market_calendars is not installed: {exc}")
        print("  install it before pulling:  "
              "sudo /spx_analysis_dashboard/.venv/bin/pip install "
              "'pandas_market_calendars>=5.0'")
        return 1

    print("exchange calendar (NYSE)")
    check(mc.MARKET == "NYSE", "the market is NYSE, matching SPX cash hours")

    # ── THE DAY THAT PRODUCED ALL OF THIS ─────────────────────────────────
    # index_ohlc holds a full VIX session and zero SPX bars for 2026-04-08.
    # The old rule called it "not an SPX session"; it was a trading day with
    # missing data, and the whole point of the calendar is to tell them apart.
    check(mc.is_session("2026-04-08"),
          "2026-04-08 IS a session -- the day the bar-count rule got wrong")
    check("2026-04-08" in mc.sessions("2026-04-01", "2026-04-30"),
          "and it appears in a month's session list")

    # ── DAYS THAT ARE NOT SESSIONS ────────────────────────────────────────
    for day, what in [("2023-07-04", "Independence Day"),
                      ("2023-12-25", "Christmas"),
                      ("2024-03-29", "Good Friday"),
                      ("2024-11-28", "Thanksgiving"),
                      ("2026-01-01", "New Year's Day"),
                      ("2023-07-08", "a Saturday"),
                      ("2023-07-09", "a Sunday")]:
        check(not mc.is_session(day), f"{day} is not a session ({what})")

    # A holiday that MOVES, so the list is not a fixed set of month-days.
    check(not mc.is_session("2021-07-05"),
          "2021-07-05 is not a session (July 4th observed on the Monday)")
    check(mc.is_session("2021-07-06"), "but 2021-07-06 is")

    # ── EXPECTED BARS COME FROM THAT DAY'S OWN CLOSE ──────────────────────
    # This is what replaces MIN_SESSION_BARS. A single floor tuned for full
    # days cannot see a short half-day; the close time can.
    tbl = mc.day_table("2024-11-25", "2024-12-02")
    full = tbl["2024-11-26"]
    early = tbl["2024-11-29"]
    check(full["expected_bars"] == 78,
          "a full session expects 78 five-minute bars (09:30-16:00)")
    check(full["close"] == "16:00" and not full["early"],
          "and is not marked early")
    check(early["expected_bars"] == 42,
          "the half-day after Thanksgiving expects 42 (09:30-13:00)")
    check(early["early"] and early["close"] == "13:00",
          "and IS marked early, with its real close time")
    check(early["expected_bars"] < full["expected_bars"],
          "a half-day expects fewer bars than a full one -- the thing a "
          "fixed floor could not express")
    check(all(d["open"] == "09:30" for d in tbl.values()),
          "every session opens 09:30, which the rollup's window assumes")

    # ── COMPLETENESS IS A SEPARATE QUESTION ───────────────────────────────
    # The calendar says whether it was a session; the data says whether a
    # series has usable values on it. is_complete never asks the first.
    check(mc.BAR_SHORTFALL_TOLERANCE == 2,
          "the shortfall tolerance is 2 bars, set by VIX3M/VIX9D arriving "
          "a bar or two behind")
    check(mc.is_complete(78, 78), "78 of 78 is complete")
    check(mc.is_complete(76, 78), "76 of 78 is complete (inside tolerance)")
    check(not mc.is_complete(75, 78), "75 of 78 is NOT -- one past tolerance")
    check(not mc.is_complete(0, 78), "0 of 78 is not complete (the 04-08 shape)")
    check(mc.is_complete(42, 42), "42 of 42 is complete on a half-day")
    check(not mc.is_complete(42, 78),
          "42 bars on a FULL day is not complete, though it would have "
          "cleared the old floor of 34")
    check(not mc.is_complete(50, 0), "a non-session expects nothing and is never complete")
    check(mc.shortfall(75, 78) == 3 and mc.shortfall(80, 78) == 0,
          "shortfall counts the gap and floors at zero")

    # ── THE RANGE THE DEPLOYMENT AXIS NEEDS ───────────────────────────────
    # index_ohlc starts 2017-01-03. The calendar has to reach further back or
    # the capital-deployed chart still cannot draw a 2013 strategy.
    s13 = mc.sessions("2013-01-01", "2013-01-08")
    check(s13[0] == "2013-01-02",
          "2013 is covered and starts 2013-01-02 (the 1st is a holiday)")
    long = mc.sessions("2013-01-01", "2026-12-31")
    check(len(long) > 3000, f"a 14-year span returns {len(long)} sessions")
    check(long == sorted(set(long)),
          "sessions are ascending, unique and contiguous as a list")
    check(all(len(d) == 10 and d[4] == '-' for d in long[:50]),
          "and are ISO date strings, which every caller compares as strings")

    # ── PURE AND REPEATABLE ───────────────────────────────────────────────
    check(mc.sessions("2024-01-01", "2024-03-31")
          == mc.sessions("2024-01-01", "2024-03-31"),
          "two identical calls agree (the cache returns the same answer)")
    t1 = mc.day_table("2024-01-02", "2024-01-05")
    t1["2024-01-03"]["expected_bars"] = -1
    t2 = mc.day_table("2024-01-02", "2024-01-05")
    check(t2["2024-01-03"]["expected_bars"] == 78,
          "mutating a returned table does not corrupt the next caller's")

    # A far-future range still answers, which is what "the rules ship in the
    # package" means in practice -- nothing was fetched.
    check(len(mc.sessions("2030-01-01", "2030-12-31")) > 240,
          "a future year resolves offline, from bundled rules")

    # ── LANDING A DATE ON A REAL SESSION ──────────────────────────────────
    check(mc.first_session_on_or_after("2023-07-04") == "2023-07-05",
          "a holiday resolves forward to the next session")
    check(mc.first_session_on_or_after("2023-07-05") == "2023-07-05",
          "a session resolves to itself")
    check(mc.first_session_on_or_after("2023-07-08") == "2023-07-10",
          "a Saturday resolves to the Monday")

    print(f"\ncalendar cases: {len(FAILURES)} failed"
          if FAILURES else "\nPASS: exchange calendar")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    raise SystemExit(main())
