"""The wall tier: a short trade tape and a quote band, for many symbols at once.

WHAT THE WALL NEEDS THAT THE OTHER TWO TIERS DO NOT HOLD.

  a pane   keeps every field of every trade AND every quote for fifteen
           minutes, for eight symbols. Too much to run a hundred of.
  the scan keeps trades only, reduced to four numbers a minute, and
           deliberately keeps NO quote tape at all -- only a duration-weighted
           accumulator, because a quote tape at 430 symbols is ~42 MB.

The wall draws a picture: trade bubbles over the last couple of minutes with
the bid and the ask drawn under them. So it needs the quote OVER TIME, which
the scan does not keep, for far more symbols than a pane can hold -- but for
two minutes rather than fifteen, and with only the three numbers that get
drawn. That is its own store.

--- Why the quote is sampled, not taped -----------------------------------

A busy name quotes 50 times a second. At two minutes that is 6,000 records a
symbol to draw a band whose whole job is to show where it sat and when it
jumped, redrawn once a second. Sampling at WALL_QUOTE_SAMPLE_MS bounds the
ring at ~600 records and costs nothing visible: 200 ms is a fifth of one
redraw interval.

THE CURRENT QUOTE IS NOT SAMPLED. `last_quote` takes every message, because
the spread number on the pane is a fact about now and must not be up to
200 ms stale or, worse, missing entirely on a symbol that has not requoted
since the ring was made.

--- Why a crossed or empty quote is dropped --------------------------------

The pane's vertical scale is built from the spread, so a bid of 0 -- which
arrives -- would put the mid at half the ask and the band across the whole
pane, and it would do it to one pane in a wall of a hundred where it reads as
a symbol doing something interesting. A quote that is not a two-sided market
is not drawn, and is counted so the discarding is visible.
"""
from __future__ import annotations

import numpy as np

from live.scan import SymbolBuf


def tail(buf: SymbolBuf, k: int):
    """The last k records of a ring, oldest first, without rolling it.

    SymbolBuf.ordered() copies the WHOLE ring; this is called once a second
    per symbol per client, and the wall holds a hundred symbols, so the
    difference between "the last twelve records" and "a megabyte of copy" is
    worth the index arithmetic.
    """
    k = max(0, min(int(k), buf.n))
    if k == 0:
        z = np.zeros(0, dtype="float64")
        return z, z, z
    end = buf.head                      # one past the newest
    start = (end - k) % buf.cap
    if start < end:
        sl = slice(start, end)
        return buf.t[sl], buf.p[sl], buf.s[sl]
    return (np.concatenate((buf.t[start:], buf.t[:end])),
            np.concatenate((buf.p[start:], buf.p[:end])),
            np.concatenate((buf.s[start:], buf.s[:end])))


class WallSym:
    """One symbol's wall state: a trade ring, a quote ring, and the last quote.

    The quote ring is a SymbolBuf read as (t, bid, ask) rather than
    (t, price, size). A second three-float64-array ring implementation would
    be the same code with different field names, and this one is the one the
    capacity work measured.
    """

    def __init__(self, retain_s: float, ring_start: int, ring_max: int,
                 quote_ring_max: int, sample_ms: float):
        self.trades = SymbolBuf(retain_s, ring_start, ring_max)
        self.quotes = SymbolBuf(retain_s, min(256, quote_ring_max),
                                quote_ring_max)
        self.sample_ms = float(sample_ms)
        # Lifetime counters, never reset. A client's cursor is a count rather
        # than a timestamp because a busy symbol prints several trades in one
        # millisecond and "everything after t" would drop all but the first
        # of them -- silently, and only on the names that print fastest.
        self.trades_seen = 0
        self.quotes_seen = 0
        # (t_ms, bid, ask), every message, unsampled.
        self.last_quote: tuple[float, float, float] | None = None
        self.last_sample_ms = 0.0
        self.crossed = 0

    # ── ingest ──────────────────────────────────────────────────────────
    def push_trade(self, t_ms: float, price: float, size: float) -> None:
        self.trades.push(t_ms, price, size)
        self.trades_seen += 1

    def push_quote(self, t_ms: float, bid: float, ask: float) -> None:
        if not (bid > 0 and ask > 0 and ask >= bid):
            self.crossed += 1
            return
        self.last_quote = (t_ms, bid, ask)
        # `>=` against the SAMPLE time, not the ring's newest record: a quote
        # arriving out of order must not be able to re-open the gate.
        if t_ms - self.last_sample_ms >= self.sample_ms:
            self.quotes.push(t_ms, bid, ask)
            self.quotes_seen += 1
            self.last_sample_ms = t_ms

    # ── reading ─────────────────────────────────────────────────────────
    def new_trades(self, cursor: int, cutoff_ms: float) -> list[list]:
        t, p, s = tail(self.trades, self.trades_seen - int(cursor or 0))
        return [[float(t[i]), float(p[i]), float(s[i])]
                for i in range(t.size) if t[i] >= cutoff_ms]

    def new_quotes(self, cursor: int, cutoff_ms: float) -> list[list]:
        t, b, a = tail(self.quotes, self.quotes_seen - int(cursor or 0))
        return [[float(t[i]), float(b[i]), float(a[i])]
                for i in range(t.size) if t[i] >= cutoff_ms]

    def quote_at(self, cutoff_ms: float) -> list | None:
        """The band's state at the left edge of the window.

        The newest sample AT OR BEFORE the cutoff. Without it a pane whose
        symbol has not requoted inside the window draws no band at all --
        which is exactly the quiet name the wall exists to show -- and a pane
        whose first sample is thirty seconds in draws a band that starts in
        the middle of itself.
        """
        t, b, a = tail(self.quotes, self.quotes.n)
        idx = int(np.searchsorted(t, cutoff_ms, side="right")) - 1
        if idx < 0:
            return None
        return [float(t[idx]), float(b[idx]), float(a[idx])]

    def spreads(self, now_ms: float, window_ms: float) -> tuple:
        """(spread_now_cents, spread_typical_cents, mid) -- the scale's input.

        TYPICAL IS A MEDIAN, over the last minute of samples. The pane's
        height is a multiple of it, so a single wide print-through quote
        must not be able to rescale the pane and make every trade in it
        collapse to the centre line for the next second. A median of a minute
        moves when the spread really moves and ignores one bad tick.

        Falls back to the current quote when the window holds no samples,
        which is the quiet symbol that quoted once and then stopped: its
        spread is still a fact, it is just an old one.
        """
        if self.last_quote is None:
            nan = float("nan")
            return nan, nan, nan
        _, bid, ask = self.last_quote
        now_c = (ask - bid) * 100.0
        mid = (ask + bid) / 2.0
        t, b, a = tail(self.quotes, self.quotes.n)
        if t.size:
            keep = t >= (now_ms - window_ms)
            if keep.any():
                typ = float(np.median((a[keep] - b[keep]))) * 100.0
                return now_c, typ, mid
        return now_c, now_c, mid

    def bytes_held(self) -> int:
        return self.trades.bytes_held() + self.quotes.bytes_held()
