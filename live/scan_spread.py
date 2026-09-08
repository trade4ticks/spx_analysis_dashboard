"""Live quoted spread: the bid-ask width, time-weighted, per symbol.

WHY THE SCAN NOW TAKES QUOTES AFTER ALL. The tier was built trades-only and
the saving was real -- measured at 200 symbols, quotes were 68% of records and
2.4x the bandwidth. But spread is the metric the user has screened on since
before this project existed, and there is no source for a CURRENT bid-ask
width other than the quote channel. INTC sits near the top of the grid on a
1-2 cent spread, and no amount of tuning the quiet ratio removes it.

So the trade is made deliberately: ~3.1x the records, ingest busy from ~6% to
~19%, CPU from ~35% to ~48% of one core at 430 symbols. Nothing saturates. The
tier's OTHER savings -- the ring shape instead of a deque of dicts, and six
minutes of retention instead of fifteen -- are untouched and are why it still
exists.

--- Nothing stores the quote tape ------------------------------------------

Spread over a five-minute window is an AGGREGATE, so there is no reason to
keep the quotes themselves. A per-minute accumulator -- the duration-weighted
sums and a count -- is ~240 bytes a symbol, about 100 KB across 430, against
~42 MB for a quote ring at 2.1 quotes per trade.

That choice also picks the statistic, and picks it the right way round. An
accumulator can produce a MEAN and a TIME-WEIGHTED mean but not a median. The
pipeline's own ranking reads `spread_bps_tw` (metrics.py, the ranking
section), so the time-weighted figure is both the cheap one and the one
already being ranked on. Matching it is not a compromise.

--- Time-weighted, because quotes do not arrive evenly ---------------------

A quote that stands for eight seconds and one that is replaced two
milliseconds later are one observation each to a simple mean, and a name whose
book flickers would report whatever the flicker was doing. What matters is
what was ACTUALLY QUOTED over the window, which is the duration-weighted
average: each quote is weighted by how long it stood before the next one
replaced it.

--- The definition is upstream's, and cannot be vendored -------------------

scalp/metrics.py::spread_metrics is the source of truth: drop crossed and
locked quotes (ask <= bid) as transient artefacts of a consolidated feed
rather than capturable spreads, bps = spread / mid * 10,000, weight by
duration. That function imports from `scalp` and takes a pandas DataFrame per
window, so it can be neither vendored verbatim nor called on a per-quote hot
path.

It is therefore REIMPLEMENTED here and FINGERPRINTED in
scripts/check_vendored.py against the upstream function, which is the same
arrangement app/trade_path_rules.py already uses. The fingerprint does not
prove this copy is correct; it proves the source has not moved since someone
last read both, which is the failure that actually happens.
"""
from __future__ import annotations

import numpy as np

# Columns of one minute bucket.
W, SP_W, BPS_W, N, CROSSED = 0, 1, 2, 3, 4
_COLS = 5

_CENTS = 100.0


class SpreadAccum:
    """Duration-weighted spread for one symbol, bucketed by minute.

    A RING OF MINUTES, not a list of quotes. Each bucket holds the weighted
    sums for the minute it covers, and the minute NUMBER is stored beside it so
    a bucket from an hour ago is recognised as stale rather than added in.
    """

    __slots__ = ("minute", "acc", "n_buckets", "cap_dwell_ms",
                 "last_t", "last_sp_c", "last_bps", "have_last")

    def __init__(self, minutes: int, cap_dwell_s: float):
        self.n_buckets = max(2, int(minutes))
        self.cap_dwell_ms = float(cap_dwell_s) * 1000.0
        self.minute = np.full(self.n_buckets, -1, dtype="int64")
        self.acc = np.zeros((self.n_buckets, _COLS), dtype="float64")
        # The quote currently standing. Its duration is not known until the
        # next one arrives, so it is carried rather than accumulated.
        self.last_t = 0.0
        self.last_sp_c = 0.0
        self.last_bps = 0.0
        self.have_last = False

    def _bucket(self, t_ms: float) -> int:
        """The row for this timestamp's minute, cleared if it is a new one."""
        minute = int(t_ms // 60000)
        i = minute % self.n_buckets
        if self.minute[i] != minute:
            self.minute[i] = minute
            self.acc[i] = 0.0
        return i

    def push(self, t_ms: float, bid: float, ask: float) -> None:
        """One quote. Closes the previous one's duration, then stands."""
        # THE PREVIOUS QUOTE STOOD UNTIL NOW. Its weight is settled here,
        # against the bucket of when it STARTED -- a quote is attributed to the
        # minute it was quoted in, not the minute it happened to be replaced
        # in, so a quote straddling a boundary lands in one bucket and is not
        # split across two.
        if self.have_last:
            dt = t_ms - self.last_t
            if dt > 0:
                # CAPPED. A quote standing across a halt, a feed gap or a
                # subscription that has just been restored would otherwise
                # dominate a five-minute window with a price nobody could have
                # traded. The cap makes a stale quote count for a plausible
                # dwell rather than for the whole outage.
                if dt > self.cap_dwell_ms:
                    dt = self.cap_dwell_ms
                i = self._bucket(self.last_t)
                self.acc[i, W] += dt
                self.acc[i, SP_W] += self.last_sp_c * dt
                self.acc[i, BPS_W] += self.last_bps * dt

        mid = (ask + bid) / 2.0
        usable = bid > 0 and ask > 0 and mid > 0
        if not usable:
            # Not counted at all, in either direction. A malformed quote is
            # not evidence about the book.
            return

        j = self._bucket(t_ms)
        self.acc[j, N] += 1.0
        spread = ask - bid
        if spread <= 0:
            # CROSSED OR LOCKED. Counted so the share is reportable, but it
            # does not stand and carries no weight -- a negative spread is not
            # a capturable one, and a zero spread is a locked book rather than
            # a free trade.
            self.acc[j, CROSSED] += 1.0
            self.have_last = False
            return

        self.last_t = t_ms
        self.last_sp_c = spread * _CENTS
        self.last_bps = spread / mid * 1e4
        self.have_last = True

    def window(self, now_ms: float, window_s: float) -> tuple:
        """(spread_cents_tw, spread_bps_tw, observations, crossed_share).

        The quote STANDING RIGHT NOW is included, weighted by how long it has
        stood so far. Leaving it out would make a name whose book has been
        still for the whole window report nothing at all -- which is exactly
        the quiet, wide-spread name the page is looking for.
        """
        first = int((now_ms - window_s * 1000.0) // 60000)
        last = int(now_ms // 60000)
        w = sp_w = bps_w = 0.0
        n = crossed = 0.0
        for minute in range(first, last + 1):
            i = minute % self.n_buckets
            if self.minute[i] != minute:
                continue
            row = self.acc[i]
            w += row[W]
            sp_w += row[SP_W]
            bps_w += row[BPS_W]
            n += row[N]
            crossed += row[CROSSED]

        if self.have_last:
            dt = min(now_ms - self.last_t, self.cap_dwell_ms)
            if dt > 0:
                w += dt
                sp_w += self.last_sp_c * dt
                bps_w += self.last_bps * dt

        if w <= 0:
            return (float("nan"), float("nan"), int(n),
                    float(crossed / n) if n else float("nan"))
        return (sp_w / w, bps_w / w, int(n),
                float(crossed / n) if n else float("nan"))

    def bytes_held(self) -> int:
        return self.acc.nbytes + self.minute.nbytes


# ── the self-test ───────────────────────────────────────────────────────────
#
# numpy only, no pandas and nothing from the hub, so this runs on a machine
# that cannot import the service. That is deliberate: the arithmetic is the
# part most likely to be wrong and the part least able to be checked against a
# live feed, so it is made checkable on its own.

def self_test() -> int:
    fails = []

    def check(cond, msg):
        if not cond:
            fails.append(msg)

    minute_ms = 60000.0
    base = 10 * minute_ms          # an arbitrary minute boundary

    # A CONSTANT 2-CENT SPREAD ON A $100 STOCK is 2.0 cents and 2.0 bps.
    # 0.02 / 100 * 10,000 = 2. If the bps scaling is wrong this is the check
    # that says so, because the two numbers happen to coincide only at $100.
    a = SpreadAccum(minutes=8, cap_dwell_s=30.0)
    for k in range(60):
        a.push(base + k * 1000.0, 99.99, 100.01)
    c, b, n, x = a.window(base + 60000.0, 300.0)
    check(abs(c - 2.0) < 1e-6, f"constant 2c spread came out {c} cents")
    check(abs(b - 2.0) < 1e-3, f"constant 2c spread on $100 came out {b} bps, "
                               f"want 2.0 -- the bps scaling is wrong")
    check(n == 60, f"{n} observations for 60 quotes")
    check(x == 0.0, f"crossed share {x} with no crossed quotes")

    # TIME WEIGHTING. A 1-cent spread standing nine seconds and a 10-cent
    # spread standing one second is 1.9 cents weighted, and 5.5 unweighted.
    # The two are far enough apart that a simple mean cannot pass this.
    t = SpreadAccum(minutes=8, cap_dwell_s=30.0)
    t.push(base, 99.995, 100.005)                 # 1c, stands 9s
    t.push(base + 9000.0, 99.95, 100.05)          # 10c, stands 1s
    c2, _, _, _ = t.window(base + 10000.0, 300.0)
    check(abs(c2 - 1.9) < 1e-3,
          f"time-weighted spread came out {c2} cents, want 1.9 (a simple mean "
          f"would give 5.5) -- durations are not being weighted")

    # CROSSED AND LOCKED ARE DROPPED, AND COUNTED. A locked book (ask == bid)
    # is not a free trade and a crossed one is a feed artefact; either landing
    # in the average would report a spread nobody could capture.
    cr = SpreadAccum(minutes=8, cap_dwell_s=30.0)
    cr.push(base, 99.99, 100.01)                  # good, 2c
    cr.push(base + 1000.0, 100.00, 100.00)        # locked
    cr.push(base + 2000.0, 100.02, 99.98)         # crossed
    cr.push(base + 3000.0, 99.99, 100.01)         # good again
    c3, _, n3, x3 = cr.window(base + 4000.0, 300.0)
    check(abs(c3 - 2.0) < 1e-6,
          f"a locked and a crossed quote moved the average to {c3}, want 2.0")
    check(n3 == 4, f"{n3} observations, want 4 -- crossed quotes are still "
                   f"observations even though they carry no weight")
    check(abs(x3 - 0.5) < 1e-9,
          f"crossed share {x3}, want 0.5 (two of four)")

    # A STALE QUOTE DOES NOT DOMINATE. One quote, then silence for ten
    # minutes: its weight is capped at the dwell ceiling rather than counting
    # for the whole gap, or a name that stopped quoting at the open would
    # report that quote as the whole window.
    st = SpreadAccum(minutes=8, cap_dwell_s=30.0)
    st.push(base, 99.90, 100.10)
    c4, _, _, _ = st.window(base + 600000.0, 300.0)
    check(abs(c4 - 20.0) < 1e-6,
          f"a single stale quote reported {c4} cents; the value should still "
          f"be its own spread")
    check(st.window(base + 600000.0, 300.0)[0] == c4,
          "window() is not idempotent -- it is mutating the accumulator")

    # THE WINDOW EXPIRES. A quote from twenty minutes ago must not be in a
    # five-minute window; the ring is only eight minutes long, so its bucket
    # has been reused and must read as stale rather than as data.
    ex = SpreadAccum(minutes=8, cap_dwell_s=30.0)
    ex.push(base, 99.00, 101.00)                  # 200c, twenty minutes back
    ex.have_last = False                          # it was replaced long ago
    for k in range(30):
        ex.push(base + 20 * minute_ms + k * 1000.0, 99.99, 100.01)
    c5, _, _, _ = ex.window(base + 20 * minute_ms + 30000.0, 300.0)
    check(abs(c5 - 2.0) < 1e-6,
          f"a quote from outside the window reported {c5} cents, want 2.0 -- "
          f"the minute ring is not recognising a reused bucket as stale")

    # NO QUOTES AT ALL IS NaN, NOT ZERO. A zero spread is a locked book, which
    # is a real and different thing; a name nobody is quoting must not screen
    # as the tightest on the page.
    empty = SpreadAccum(minutes=8, cap_dwell_s=30.0)
    ce, be, ne, _ = empty.window(base, 300.0)
    check(ce != ce and be != be,
          f"an unquoted symbol reported {ce} cents / {be} bps rather than NaN "
          f"-- it would screen as the tightest spread on the page")
    check(ne == 0, f"{ne} observations with no quotes")

    for f in fails:
        print(f"  FAIL  {f}")
    print(f"  spread self-test: {'PASS' if not fails else str(len(fails)) + ' FAILED'}")
    return 1 if fails else 0


if __name__ == "__main__":
    import sys
    sys.exit(self_test())
