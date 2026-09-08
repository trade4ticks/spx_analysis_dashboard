"""Gate: the Replay data layer, against a fabricated session.

NO PARQUET, NO DATABASE, NO MARKET. The parts most likely to be wrong -- which
minute a trade lands in, where the print/candle switch falls, whether the NBBO
steps survive being thinned, whether the payload is even valid JSON -- are all
decidable from a synthetic session, and none of them can be checked by looking
at a chart and deciding it looks about right.

WHAT IS BEING PROTECTED, in order of how bad it would be:

  * THE SESSION ORIGIN. Every candle is attributed by minutes-since-09:30. Get
    the origin from the first trade instead of the date and a name that does
    not print until 09:31 has its whole session shifted one minute left --
    consistently, so it looks entirely correct.
  * THE MODE IS NAMED, NOT INFERRED. "few trades" and "too many to send" look
    identical at the edge, and that edge is two minutes on NVDA.
  * NaN IS NOT JSON. A minute with no trades must not take the whole chart
    down with it.
  * THE NBBO THINNING IS LOSSLESS AS A STEP. Only rows where the quote moved
    are sent; if the first row is dropped the line starts in the wrong place.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app import replay                                    # noqa: E402

FAILS: list[str] = []


def check(cond, msg):
    if not cond:
        FAILS.append(msg)
        print(f"  FAIL  {msg}")


SYM, DAY = "TESTX", "2026-08-27"


def seed(t, p, s, bid=None, ask=None):
    """Put a fabricated session straight into the cache.

    Seeding the cache rather than writing a parquet file keeps this gate free
    of pyarrow and of the disk, and it exercises exactly the arrays `load()`
    produces -- which is the boundary worth testing on either side of.
    """
    t = np.asarray(t, dtype="float64")
    n = t.size
    d = {
        "symbol": SYM, "date": DAY,
        "t": t,
        "p": np.asarray(p, dtype="float32"),
        "s": np.asarray(s, dtype="float32"),
        "bid": np.asarray(bid if bid is not None else np.asarray(p) - 0.01,
                          dtype="float32"),
        "ask": np.asarray(ask if ask is not None else np.asarray(p) + 0.01,
                          dtype="float32"),
    }
    replay._CACHE.clear()
    replay._CACHE[(SYM, DAY)] = d
    return d


def case_candles_land_in_the_right_minute():
    """A trade at 09:31:30 belongs to minute 1, not minute 0 or 2."""
    # One trade in minute 0, three in minute 1, none in minute 2, one in 3.
    t = [5.0, 90.0, 95.0, 119.9, 200.0]
    p = [10.0, 11.0, 13.0, 12.0, 20.0]
    s = [100, 1, 2, 3, 50]
    seed(t, p, s)
    c = replay.candles(SYM, DAY)

    check(c["minutes"] == 390, f"{c['minutes']} bars, want 390")
    check(c["n"][0] == 1 and c["n"][1] == 3 and c["n"][2] == 0
          and c["n"][3] == 1,
          f"trades per minute came out {c['n'][:4]}, want [1, 3, 0, 1] -- a "
          f"session origin taken from the first trade instead of the date "
          f"shifts every bar and still looks correct")
    check(c["o"][1] == 11.0 and c["c"][1] == 12.0,
          f"minute 1 opened {c['o'][1]} closed {c['c'][1]}, want 11.0 / 12.0")
    check(c["h"][1] == 13.0 and c["l"][1] == 11.0,
          f"minute 1 high/low {c['h'][1]}/{c['l'][1]}, want 13.0 / 11.0")
    check(c["v"][1] == 6, f"minute 1 volume {c['v'][1]}, want 6")

    # AN EMPTY MINUTE IS null, NOT ZERO. A zero open would draw a candle at
    # the bottom of the axis for a minute in which nothing happened.
    check(c["o"][2] is None and c["h"][2] is None,
          f"an empty minute came back {c['o'][2]} / {c['h'][2]}, want null")
    check(c["v"][2] == 0 and c["n"][2] == 0,
          "an empty minute should carry zero volume and zero trades")

    # AND THE WHOLE THING MUST BE JSON. NaN is not in the grammar; a browser
    # drops the entire frame rather than one bar.
    json.loads(json.dumps(c, allow_nan=False))


def case_the_mode_switches_on_count_not_time():
    """The threshold is trades in view, and it is NAMED in the response.

    A time-width rule would be wrong in both directions at once: NVDA at
    11,092 trades a minute passes 20,000 in under two minutes, and FDX at 118
    would not pass it in two hours.
    """
    # 60 seconds of a very busy tape: 5,000 trades in one minute.
    n = 5000
    t = np.linspace(0.0, 59.999, n)
    seed(t, np.full(n, 10.0), np.ones(n))

    w = replay.window(SYM, DAY, 0.0, 60.0, limit=20000)
    check(w["mode"] == "prints" and w["count"] == n,
          f"{n} trades under a 20,000 limit came back as {w['mode']} "
          f"with count {w['count']}")
    check(len(w["t"]) == n and len(w["p"]) == n and len(w["s"]) == n,
          "columnar arrays are not all the same length as the count")

    w2 = replay.window(SYM, DAY, 0.0, 60.0, limit=1000)
    check(w2["mode"] == "candles",
          f"5,000 trades over a 1,000 limit came back as {w2['mode']}")
    check("t" not in w2,
          "the over-limit response still shipped the trades it refused to "
          "draw -- slow AND useless, in that order")
    check(w2["count"] == n and w2["limit"] == 1000,
          f"the refusal does not say how many there were: {w2['count']} of "
          f"{w2['limit']} -- the page cannot explain the switch without it")

    # THE SAME TIME WIDTH, A DIFFERENT ANSWER. This is the whole point of
    # counting rather than measuring the window.
    seed(np.linspace(0.0, 59.999, 100), np.full(100, 10.0), np.ones(100))
    w3 = replay.window(SYM, DAY, 0.0, 60.0, limit=20000)
    check(w3["mode"] == "prints",
          "a quiet minute of the same width was refused prints, so the "
          "threshold is reading time and not trades")


def case_window_bounds_are_half_open():
    """[t0, t1) -- a trade exactly on the boundary belongs to one window."""
    t = [0.0, 10.0, 20.0, 30.0]
    seed(t, [1.0, 2.0, 3.0, 4.0], [1, 1, 1, 1])
    a = replay.window(SYM, DAY, 0.0, 20.0)
    b = replay.window(SYM, DAY, 20.0, 40.0)
    check(a["count"] == 2 and b["count"] == 2,
          f"a half-open split gave {a['count']} and {b['count']}, want 2 and "
          f"2 -- a trade on the boundary is being counted twice or dropped")
    check(a["p"][-1] == 2.0 and b["p"][0] == 3.0,
          f"the boundary trade landed in the wrong window: {a['p']} / {b['p']}")


def case_nbbo_is_thinned_but_still_a_step():
    """Only rows where the quote moved, and never without the first one."""
    t = [0.0, 1.0, 2.0, 3.0, 4.0]
    p = [10.0] * 5
    bid = [9.99, 9.99, 9.98, 9.98, 9.98]
    ask = [10.01, 10.01, 10.01, 10.02, 10.02]
    seed(t, p, [1] * 5, bid=bid, ask=ask)
    w = replay.window(SYM, DAY, 0.0, 5.0)
    nb = w["nbbo"]
    check(nb["t"] == [0.0, 2.0, 3.0],
          f"quote change points came out {nb['t']}, want [0.0, 2.0, 3.0]")
    check(nb["bid"][0] == 9.99 and nb["ask"][0] == 10.01,
          f"the first change point is wrong ({nb['bid'][0]}, {nb['ask'][0]}) "
          f"-- dropping it starts the step line at the wrong level")
    check(len(nb["t"]) == len(nb["bid"]) == len(nb["ask"]),
          "the NBBO arrays are not the same length")

    off = replay.window(SYM, DAY, 0.0, 5.0, nbbo=False)
    check("nbbo" not in off, "nbbo=False still sent the quote")


def case_stats_describe_and_do_not_judge():
    """A readout: counts, a span, a spread. No threshold, no verdict."""
    # 120 trades over two minutes, 100 shares each, price walking 10.00-10.20.
    n = 120
    t = np.linspace(0.0, 119.0, n)
    p = np.linspace(10.00, 10.20, n)
    seed(t, p, np.full(n, 100))
    w = replay.window(SYM, DAY, 0.0, 120.0)
    st = w["stats"]

    check(st["trades"] == n, f"{st['trades']} trades, want {n}")
    check(abs(st["trades_per_min"] - 60.0) < 1e-6,
          f"{st['trades_per_min']} trades/min over two minutes, want 60")
    check(abs(st["shares_per_min"] - 6000.0) < 1e-6,
          f"{st['shares_per_min']} shares/min, want 6000")
    # p10-p90 of an even walk from 10.00 to 10.20 spans 80% of 20 cents.
    check(abs(st["p10_p90_cents"] - 16.0) < 0.4,
          f"p10-p90 came out {st['p10_p90_cents']} cents, want ~16")
    # bid/ask default to +/- 1 cent, so the quoted spread is 2 cents.
    check(abs(st["spread_cents_tw"] - 2.0) < 0.05,
          f"time-weighted spread came out {st['spread_cents_tw']}, want 2.0")

    check(not any(k in st for k in ("score", "rank", "verdict", "pass",
                                    "grade", "ok")),
          f"the readout has grown a judgement: {sorted(st)} -- it describes "
          f"the window, it does not grade it, and every metric family so far "
          f"has failed by grading before anyone agreed what good looked like")

    empty = replay.window(SYM, DAY, 300.0, 360.0)
    check(empty["count"] == 0 and empty["stats"]["trades"] == 0,
          "an empty window did not report zero")
    check(empty["stats"]["p10_p90_cents"] is None,
          f"an empty window reported a span of "
          f"{empty['stats']['p10_p90_cents']} rather than null")
    json.loads(json.dumps(empty, allow_nan=False))


def case_a_missing_session_says_so():
    """No parquet is a sentence, not an empty chart.

    Drawing nothing for a date with no file is indistinguishable from drawing
    a session in which nothing traded, and telling those apart is the entire
    purpose of the tool.
    """
    import os
    import tempfile
    replay._CACHE.clear()

    def refusal(sym, day):
        try:
            replay.load(sym, day)
        except replay.ReplayError as exc:
            return str(exc)
        except Exception as exc:                          # noqa: BLE001
            check(False, f"raised {type(exc).__name__} rather than a "
                         f"ReplayError the page can print: {exc}")
            return None
        check(False, "a missing session did not raise at all -- the page "
                     "would draw an empty chart and call it a quiet day")
        return None

    # TWO WAYS TO HAVE NOTHING, and they need different answers. An unset
    # SCALP_DATA_DIR is a deployment fault; a missing file is an ordinary
    # gap in 45-day retention. Reporting the second as the first would send
    # someone to check the environment over a date that was simply pruned.
    prev = os.environ.pop("SCALP_DATA_DIR", None)
    try:
        msg = refusal("NOSUCHSYMBOL", "1999-01-04")
        if msg is not None:
            check("SCALP_DATA_DIR" in msg,
                  f"an unset data dir was not named as the cause: {msg}")

        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "raw").mkdir()
            os.environ["SCALP_DATA_DIR"] = d
            replay._CACHE.clear()
            msg = refusal("NOSUCHSYMBOL", "1999-01-04")
            if msg is not None:
                check("NOSUCHSYMBOL" in msg and "1999-01-04" in msg,
                      f"the refusal does not name what was asked for: {msg}")
                check("retention" in msg.lower(),
                      f"the refusal does not say a gap is ordinary: {msg}")
    finally:
        os.environ.pop("SCALP_DATA_DIR", None)
        if prev is not None:
            os.environ["SCALP_DATA_DIR"] = prev


CASES = [
    ("candles land right",      case_candles_land_in_the_right_minute),
    ("mode switches on count",  case_the_mode_switches_on_count_not_time),
    ("windows are half-open",   case_window_bounds_are_half_open),
    ("nbbo thins to steps",     case_nbbo_is_thinned_but_still_a_step),
    ("stats do not judge",      case_stats_describe_and_do_not_judge),
    ("missing session speaks",  case_a_missing_session_says_so),
]


def main() -> int:
    for name, fn in CASES:
        before = len(FAILS)
        try:
            fn()
        except Exception as exc:                          # noqa: BLE001
            FAILS.append(f"{name}: raised {type(exc).__name__}: {exc}")
            print(f"  FAIL {name}: raised {type(exc).__name__}: {exc}")
        del before
    print(f"\nreplay cases: {len(CASES)}, failures: {len(FAILS)}")
    return 1 if FAILS else 0


sys.exit(main())
