"""The wall tier: a rolling spread window for each of up to 300 symbols.

WHAT THE WALL SHOWS (2026-11-06, windowed 2026-11-09). One row per symbol:
the ticker, a range bar and a number in cents. The page is swept for which
names get wide often enough to catch, so the tier subscribes the QUOTE
channel only (see Hub._wants) and keeps, per symbol, the spread over the
last `WALL_WINDOW_S` seconds.

WHY A WINDOW, AND WHY NOT A MEDIAN. The single latest quote made rows flash
(20c, 57c, 35c, four repaints a second). The page's trader rests limit orders
inside the spread and waits to be hit: a name sitting at 20c but gapping to
57c a few seconds in every thirty is a GOOD name, because the gap is the
fill. A central measure scores it ~35c and dims it under a 59c floor. So the
page is driven by the window's HIGH percentile (`WALL_PCT_HI`, p90) -- "how
wide does this get, often enough to catch" -- and shows p10..p90 with the
median as a tick. The tape wall's one-minute median was removed in d56f7b9
and is not coming back.

THE WINDOW IS TIME BUCKETS, NOT QUOTES. `WALL_BUCKETS` fixed slots of
`WALL_BUCKET_MS` in a ring. A quote is a STEP FUNCTION -- the NBBO stands
until replaced -- so each bucket holds the spread STANDING AT ITS END (the
last quote in it, or the one carried in from before), and buckets with no
quote are CARRIED FORWARD, never left empty. That makes the statistic
time-weighted: a name that requotes furiously while tight and sits still
while wide has its wide stretch counted at its full length, and a name that
quotes once and goes quiet is at that spread for every bucket after. A dead
feed carries its last spread forward too, which is correct: that quote is
still standing. Memory is bounded: 300 floats a symbol.

THE CLOCK. A quote goes in the bucket of its OWN timestamp, clamped to
[the last bucket written, now] -- a quote stamped ahead of this box's clock
(skew) lands now, and one older than a bucket already written updates the
latest, so buckets never go backwards. The window is rolled forward to the
wall clock when it is read, so a quiet name ages like a busy one.

Percentiles are NEAREST-RANK over the filled buckets: p90 is the smallest
spread that at least 90% of the buckets are at or under -- a spread that
actually stood, and the name was at least that wide for at least ~10% of the
window. Until the window has filled, it is over what has been seen.

--- Why a crossed or empty quote is dropped --------------------------------

A bid of 0 -- which arrives -- would read as a spread of the whole ask, and
on a list sorted or coloured by spread it would put that name at the top in
the wide colour, where it reads as a symbol worth trading. A quote that is
not a two-sided market is not taken (the previous spread goes on standing),
and is counted so the discarding is visible.
"""
from __future__ import annotations

import math

from live import config


def _cents(x: float):
    """Spread in cents, to a hundredth: float subtraction of two prices leaves
    7.000000000000028, and the runner compares values to what it last sent,
    so noise in the 14th digit would be a change. Whole cents go out as ints
    (the wire is a third shorter for the usual penny spread)."""
    v = round(x, 2)
    return int(v) if v == int(v) else v


def _rank(sorted_vals: list, pct: float):
    """Nearest-rank percentile: the value at ceil(p/100 * n), 1-based."""
    n = len(sorted_vals)
    k = max(1, math.ceil(pct / 100.0 * n - 1e-9))
    return sorted_vals[min(n, k) - 1]


class WallSym:
    """One symbol's wall state: its spread window and its last quote."""

    __slots__ = ("last_quote", "quotes_seen", "crossed", "ring", "b_last",
                 "filled", "cur")

    def __init__(self):
        # (t_ms, bid, ask) of the last two-sided quote.
        self.last_quote: tuple[float, float, float] | None = None
        self.quotes_seen = 0
        self.crossed = 0
        self.ring: list = [None] * config.WALL_BUCKETS
        self.b_last: int | None = None     # absolute index of the newest bucket
        self.filled = 0                    # buckets holding a value, <= WALL_BUCKETS
        self.cur: float | None = None      # the spread standing now, in cents

    # ── the ring ────────────────────────────────────────────────────────
    def _roll(self, b: int) -> None:
        """Carry the standing spread forward into every bucket up to b."""
        n = config.WALL_BUCKETS
        gap = b - self.b_last
        if gap <= 0:
            return
        for k in range(self.b_last + 1, self.b_last + 1 + min(gap, n)):
            self.ring[k % n] = self.cur
        self.filled = min(n, self.filled + gap)
        self.b_last = b

    def push_quote(self, t_ms: float, bid: float, ask: float,
                   now_ms: float | None = None) -> None:
        if not (bid > 0 and ask > 0 and ask >= bid):
            self.crossed += 1
            return
        self.last_quote = (t_ms, bid, ask)
        self.quotes_seen += 1
        sp = _cents((ask - bid) * 100.0)
        b = int((t_ms if now_ms is None else min(t_ms, now_ms))
                // config.WALL_BUCKET_MS)
        if self.b_last is None:
            self.b_last, self.filled = b, 1
        elif b > self.b_last:
            self._roll(b - 1)            # the old spread stood until this bucket
            self.b_last = b
            self.filled = min(config.WALL_BUCKETS, self.filled + 1)
        # b <= b_last: this bucket (or a later one already written) -- the
        # newest quote is what stands at the end of the newest bucket.
        self.ring[self.b_last % config.WALL_BUCKETS] = sp
        self.cur = sp

    def window(self, now_ms: float):
        """(p10, median, p90) in cents over the last WALL_WINDOW_S, rolled to
        now; None before the first quote."""
        if self.b_last is None:
            return None
        self._roll(int(now_ms // config.WALL_BUCKET_MS))
        n = config.WALL_BUCKETS
        if self.filled < n:
            vals = sorted(self.ring[(self.b_last - i) % n]
                          for i in range(self.filled))
        else:
            vals = sorted(self.ring)
        return (_rank(vals, config.WALL_PCT_LO), _rank(vals, 50.0),
                _rank(vals, config.WALL_PCT_HI))

    def spread_cents(self):
        """The current (instantaneous) spread in cents, or None before the
        first quote. Not what the page shows -- see window()."""
        return self.cur
