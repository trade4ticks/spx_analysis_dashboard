"""The arrival-rate comparison must be against the right quarter-hour.

WHY THIS IS CHECKED AT ALL. Every way this can be wrong is invisible on
screen. A bucket off by one puts 09:44 against the 09:45 history and the band
still looks like a band. A daily average instead of a clock-time one makes
every symbol look like it is deteriorating at noon, and it is — so the plot
agrees with itself and stays wrong. Today's own partial session leaking into
its own baseline tightens the band exactly when the live value is extreme.

None of that produces an error, an empty pane, or a number that looks odd. So
the boundaries, the date bind and the exclusion are asserted here.

Four things:

    the 15-minute bucket is right AT the boundaries, and absent
    outside the session;

    the date bound to the query is a `datetime.date` — asyncpg binds
    by type, and a stringified date raises at bind time, which is how
    /meta returned 500 once already;

    today is excluded from its own normal;

    a host with no timezone database SAYS SO instead of silently
    using local time, which would compare a symbol against a
    different part of its own day and look entirely plausible.
"""
from __future__ import annotations

import asyncio
import datetime as dt
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from live import norms                                       # noqa: E402


class FakePool:
    """Records what it was asked, and with what types."""

    def __init__(self, rows):
        self.rows = rows
        self.sql = None
        self.args = None

    async def fetch(self, sql, *args):
        self.sql = sql
        self.args = args
        return self.rows


def main() -> int:
    bad = 0

    # ── the boundaries ───────────────────────────────────────────────────
    day = dt.date(2026, 9, 2)

    def at(h, m, s=0):
        return norms.bucket_for(dt.datetime.combine(day, dt.time(h, m, s)))

    cases = [
        ((9, 29, 59), None,          "a second before the open is not a bucket"),
        ((9, 30, 0),  dt.time(9, 30), "the open is the first bucket"),
        ((9, 44, 59), dt.time(9, 30), "09:44:59 belongs to 09:30, not 09:45"),
        ((9, 45, 0),  dt.time(9, 45), "09:45:00 starts the next bucket"),
        ((12, 7, 30), dt.time(12, 0), "12:07 belongs to 12:00"),
        ((15, 59, 59), dt.time(15, 45), "the last bucket runs to the close"),
        ((16, 0, 0),  None,          "the close is past the last bucket"),
        ((8, 0, 0),   None,          "pre-market is outside the session"),
    ]
    for (h, m, s), want, why in cases:
        got = at(h, m, s)
        if got != want:
            bad += 1
            print(f"\n  {h:02d}:{m:02d}:{s:02d} gave {got}, expected {want}"
                  f"\n      {why}")

    # ── percentiles over a small sample ──────────────────────────────────
    # Eleven values, so the quartiles land between order statistics and the
    # interpolation is exercised rather than assumed.
    xs = [10, 12, 14, 15, 18, 20, 22, 25, 30, 40, 90]
    s = norms.summarise(list(reversed(xs)))          # order must not matter
    for key, want in (("n", 11), ("min", 10), ("max", 90), ("med", 20),
                      ("p25", 14.5), ("p75", 27.5)):
        if abs(s[key] - want) > 1e-9:
            bad += 1
            print(f"\n  summarise()[{key!r}] = {s[key]}, expected {want}")
    if norms.summarise([]) is not None:
        bad += 1
        print("\n  an empty sample summarised to something rather than None")
    # A single session is a legitimate answer, reported as n=1 rather than
    # dressed up as a distribution.
    one = norms.summarise([7.0])
    if not one or one["n"] != 1 or one["min"] != one["max"] != 7.0:
        bad += 1
        print(f"\n  a one-session sample summarised to {one}")

    # ── the query: date type, and today excluded ─────────────────────────
    if norms.tz_problem():
        # The tz-less path is itself under test below; the query test needs a
        # clock, so it supplies one rather than skipping.
        pass

    rows = [{"trade_date": dt.date(2026, 8, 17), "trades_per_min": 40.0,
             "shares_per_min": 4000.0},
            {"trade_date": dt.date(2026, 8, 18), "trades_per_min": 60.0,
             "shares_per_min": 9000.0}]
    fake = FakePool(rows)
    norms._pool = fake
    norms._CACHE.clear()
    now = dt.datetime(2026, 9, 2, 11, 20, 0)
    out = asyncio.run(norms.arrival_norm("FDX", now=now))
    norms._pool = None

    if not out.get("ok"):
        bad += 1
        print(f"\n  a populated history produced no reading: {out}")
    else:
        if out["bucket"] != "11:15":
            bad += 1
            print(f"\n  11:20 was compared against the {out['bucket']} bucket")
        if out["sessions"] != 2:
            bad += 1
            print(f"\n  two sessions reported as {out['sessions']} — the count "
                  f"is on screen and has to be the real one")

    if fake.args is None:
        bad += 1
        print("\n  no query was issued at all")
    else:
        sym, bucket, upto = fake.args
        # asyncpg BINDS BY TYPE. A DATE column handed a string raises at bind
        # time; a TIME column handed a string does too. This is the same class
        # of fault that returned 500 from /meta.
        if not isinstance(upto, dt.date) or isinstance(upto, dt.datetime):
            bad += 1
            print(f"\n  the date bound to the query is {type(upto).__name__}, "
                  f"not datetime.date — asyncpg binds by type and raises on a "
                  f"string")
        if not isinstance(bucket, dt.time):
            bad += 1
            print(f"\n  the bucket bound to the query is "
                  f"{type(bucket).__name__}, not datetime.time")
        # TODAY EXCLUDED. Its own partial session must not become part of the
        # normal it is measured against — that would tighten the band exactly
        # when the live value is extreme.
        if upto != now.date():
            bad += 1
            print(f"\n  the query bound {upto} as the upper bound against a "
                  f"clock of {now.date()}")
        if "trade_date < $3" not in " ".join(fake.sql.split()):
            bad += 1
            print(f"\n  the query does not exclude today: today's own partial "
                  f"session would become part of its own baseline")
        if "bucket_time = $2" not in " ".join(fake.sql.split()):
            bad += 1
            print("\n  the query does not pin the clock time — a daily average "
                  "makes every name look like it deteriorates at noon, because "
                  "every name does")

    # ── outside the session, and no history ──────────────────────────────
    norms._CACHE.clear()
    norms._pool = FakePool([])
    shut = asyncio.run(norms.arrival_norm(
        "FDX", now=dt.datetime(2026, 9, 2, 17, 0)))
    if shut.get("ok") or "outside the session" not in (shut.get("why") or ""):
        bad += 1
        print(f"\n  17:00 produced {shut} rather than saying it is outside "
              f"the session")
    empty = asyncio.run(norms.arrival_norm("ZZZZ", now=now))
    norms._pool = None
    if empty.get("ok"):
        bad += 1
        print("\n  a symbol with no stored history produced a reading anyway")
    elif "no stored history" not in (empty.get("why") or ""):
        bad += 1
        print(f"\n  a symbol with no history said {empty.get('why')!r} rather "
              f"than naming the reason")

    # ── no timezone database: SAY SO, never fall back to local ───────────
    real = norms.tz_problem
    norms.tz_problem = lambda: "no timezone database (simulated)"
    norms._CACHE.clear()
    blind = asyncio.run(norms.arrival_norm("FDX"))
    norms.tz_problem = real
    if blind.get("ok"):
        bad += 1
        print("\n  with no market clock available the pane still produced a "
              "comparison — local time would put the symbol against the wrong "
              "quarter of its own day and look entirely plausible doing it")
    elif "timezone" not in (blind.get("why") or "").lower():
        bad += 1
        print(f"\n  the missing clock was reported as {blind.get('why')!r}")

    here = norms.tz_problem()
    print(f"\nbucket boundaries: {len(cases)}, "
          f"market clock here: {'unavailable — ' + here[:40] if here else 'ok'}"
          f"; problems: {bad}")
    if not bad:
        print("  the bucket is right at its edges, the binds are typed, and "
              "today is excluded from its own normal")
    return 1 if bad else 0


sys.exit(main())
