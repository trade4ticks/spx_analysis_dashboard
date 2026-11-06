"""The wall tier: the current quote of up to 300 symbols, and nothing else.

WHAT THE WALL SHOWS (2026-11-06). One row per symbol: the ticker, a bar and
the spread in cents. It used to draw a small tape per symbol -- two minutes
of trades and a sampled quote band -- and that is gone with the tape: the
page is swept for which names currently clear a minimum spread, and nobody
was reading the prints. So the tier subscribes the QUOTE channel only (see
Hub._wants) and keeps one record a symbol: the last two-sided quote.

EVERY QUOTE IS TAKEN. Nothing is sampled: a push is three assignments, and
the cost that matters is the upstream message rate, which the tier pays
whether or not it keeps the message. What the page sees is throttled by the
runner's tick, not here.

--- Why a crossed or empty quote is dropped --------------------------------

A bid of 0 -- which arrives -- would read as a spread of the whole ask, and
on a list sorted or coloured by spread it would put that name at the top in
the wide colour, where it reads as a symbol worth trading. A quote that is
not a two-sided market is not taken, and is counted so the discarding is
visible.
"""
from __future__ import annotations


class WallSym:
    """One symbol's wall state: its last two-sided quote."""

    __slots__ = ("last_quote", "quotes_seen", "crossed")

    def __init__(self):
        # (t_ms, bid, ask), every message.
        self.last_quote: tuple[float, float, float] | None = None
        self.quotes_seen = 0
        self.crossed = 0

    def push_quote(self, t_ms: float, bid: float, ask: float) -> None:
        if not (bid > 0 and ask > 0 and ask >= bid):
            self.crossed += 1
            return
        self.last_quote = (t_ms, bid, ask)
        self.quotes_seen += 1

    def spread_cents(self) -> float | None:
        """The current spread in cents, or None before the first quote.

        Rounded to a hundredth of a cent: float subtraction of two prices
        leaves 7.000000000000028, and the runner compares this value to what
        it last sent, so noise in the 14th digit would be a change.
        """
        if self.last_quote is None:
            return None
        _, bid, ask = self.last_quote
        return round((ask - bid) * 100.0, 2)
