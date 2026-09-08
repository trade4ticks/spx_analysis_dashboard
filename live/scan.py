"""The scan tier's trade store: many symbols, trades only, bounded memory.

WHY THIS IS NOT THE TAPE'S BUFFER. The tape holds eight symbols and needs
every field of every record -- venue, tape, condition codes -- because a pane
draws them. A deque of dicts is the right shape for that and it is not changed
here.

The scan holds up to six hundred and needs three numbers per trade. At the
open, 600 symbols print ~3,850 trades a second (measured); over a six-minute
retention that is 1.4M records, and 1.4M dicts of six fields is gigabytes on a
box that has been OOM-killed twice. The same records as three float64 arrays
are 33 MB. That is the whole reason this module exists.

QUOTES ARE NOT STORED AND NOT SUBSCRIBED. quiet.py is trades-only by
construction -- a 29-share bid pulled on a thin book moves the midpoint 19
cents while the stock does not move at all -- so the scan has no use for them.
Measured at 200 symbols, quotes were 68% of records and the whole subscription
was 2.4x the message volume of trades alone. Not asking for them is the single
largest saving available, and it costs nothing the scan wants.

--- The ring grows rather than truncating -----------------------------------

The capacity run sized every symbol's ring from one assumed rate -- 20
trades/sec -- and every step logged overflows: the busiest names averaged 53
a second at the open. Two things were wrong with that, in opposite directions.
The busy names silently lost the back of their five-minute range window, so
the range bar read short exactly where it mattered; and the ~550 quiet names
were each given a ring for a rate they never came close to, which is where the
measured 106 MB came from against 33 MB of actual data.

So a ring starts small and GROWS, and the trigger is the thing that was
actually wrong: an eviction that would drop a record still inside the
retention window. Overwriting a record older than the window is correct and
must not grow anything -- that is the ring working. Overwriting one the range
bar still needs is the fault, and it is the only case that allocates.

The effect is that each symbol converges on the size ITS OWN rate needs,
which is the fix for both directions at once and needs no per-symbol
configuration to guess at.
"""
from __future__ import annotations

import asyncio
import time

import numpy as np

from app import scalp_quiet as quiet


class SymbolBuf:
    """A growable ring of (time, price, size), oldest evicted first.

    Three parallel float64 arrays rather than a list of tuples or dicts: 24
    bytes a trade against ~120 for a 3-tuple and ~400 for a dict, and the
    rollup can hand slices straight to numpy without rebuilding an array on
    every pass.
    """

    __slots__ = ("t", "p", "s", "n", "head", "cap", "cap_max", "retain_s",
                 "grows", "evicted_live")

    def __init__(self, retain_s: float, cap0: int, cap_max: int):
        self.retain_s = float(retain_s)
        self.cap = max(8, int(cap0))
        self.cap_max = max(self.cap, int(cap_max))
        self.t = np.zeros(self.cap, dtype="float64")
        self.p = np.zeros(self.cap, dtype="float64")
        self.s = np.zeros(self.cap, dtype="float64")
        self.head = 0        # next write position; when full, the oldest
        self.n = 0
        self.grows = 0
        # Counted, and surfaced in status. A ring at cap_max that is still
        # evicting live records is a truncated window, and the page has to be
        # able to say so rather than quietly drawing a short bar.
        self.evicted_live = 0

    def push(self, t_ms: float, price: float, size: float) -> None:
        if self.n == self.cap:
            # The record about to be overwritten. If it is still inside the
            # retention window the range bar needs it, so grow instead --
            # subject to the ceiling, past which the loss is reported.
            oldest = self.t[self.head]
            if (t_ms - oldest) < self.retain_s * 1000.0:
                if self.cap < self.cap_max:
                    self._grow()
                else:
                    self.evicted_live += 1
        i = self.head
        self.t[i] = t_ms
        self.p[i] = price
        self.s[i] = size
        self.head = (i + 1) % self.cap
        if self.n < self.cap:
            self.n += 1

    def _grow(self) -> None:
        """Double, capped, re-laid out oldest-first so head returns to n."""
        new_cap = min(self.cap * 2, self.cap_max)
        t, p, s = self.ordered()
        self.t = np.zeros(new_cap, dtype="float64")
        self.p = np.zeros(new_cap, dtype="float64")
        self.s = np.zeros(new_cap, dtype="float64")
        self.t[:t.size] = t
        self.p[:p.size] = p
        self.s[:s.size] = s
        self.cap = new_cap
        self.head = t.size
        self.n = t.size
        self.grows += 1

    def ordered(self):
        """(t, p, s) oldest-first. One roll when the ring has wrapped."""
        if self.n < self.cap:
            return self.t[:self.n], self.p[:self.n], self.s[:self.n]
        k = -self.head
        return np.roll(self.t, k), np.roll(self.p, k), np.roll(self.s, k)

    def bytes_held(self) -> int:
        return self.t.nbytes + self.p.nbytes + self.s.nbytes


def rollup_one(buf: SymbolBuf, now_s: float, *, quiet_window_s: float,
               slow_window_s: float, min_trades: int) -> tuple:
    """One symbol's current state: (ratio, range_cents, dollar_per_min, n).

    THE TWO LOOKBACKS ARE DIFFERENT ON PURPOSE. Quiet uses a 60-second window
    because it is the thing that changes and the reason the page is open, so
    it has to be responsive. Range and dollar volume use five minutes because
    they answer "is this name worth anything at all", which is not a
    per-minute question -- and a bar that jumps while you glance at it for a
    tenth of a second is worse than no bar.

    The quiet ratio comes from the VENDORED window_series, at the window and
    step the pipeline uses, so the grid and the stored metric are the same
    arithmetic rather than two things that ought to agree. If they ever
    disagree it is because the name changed, which is the only reading that is
    any use.
    """
    t, p, s = buf.ordered()
    if t.size == 0:
        return (float("nan"), float("nan"), float("nan"), 0)
    t_sec = t / 1000.0

    # The ratio is a difference against the PREVIOUS window, so the span asked
    # for is a window plus a step, not a window. Asking for exactly one window
    # yields one measurement with nothing to difference it against, and a
    # ratio that is NaN forever.
    step_s = quiet.step_for(quiet_window_s)
    ser = quiet.window_series(t_sec, p, s, window_s=quiet_window_s,
                              start_s=now_s - (quiet_window_s + step_s),
                              end_s=now_s, step_s=step_s,
                              min_trades=min_trades)
    ratio = float("nan")
    if ser["ratio"].size:
        r = ser["ratio"][-1]                  # the last window is the current
        ratio = float(r) if np.isfinite(r) else float("nan")

    slow = t_sec >= (now_s - slow_window_s)
    n_slow = int(slow.sum())
    if n_slow >= min_trades:
        ps = p[slow]
        q10, q90 = np.percentile(ps, (10.0, 90.0))
        range_c = float((q90 - q10) * 100.0)
        dollars = float(np.dot(ps, s[slow])) / (slow_window_s / 60.0)
    else:
        range_c = float("nan")
        dollars = float("nan")
    return (ratio, range_c, dollars, n_slow)


async def rollup_all(bufs, now_s: float | None = None, *,
                     quiet_window_s: float, slow_window_s: float,
                     min_trades: int, slice_s: float) -> dict:
    """Every symbol's current state, WITHOUT holding the event loop.

    The pass costs ~0.85 ms a symbol, so 430 symbols is 367 ms -- and run as
    one uninterrupted loop that is 367 ms in which nothing else in this
    process runs. Not the upstream reader, so frames queue; and not
    Hub.pump(), which flushes to every browser every 100 ms, so the tape
    stalls for three and a half flush intervals every five seconds. The scan
    shares the tape's connection AND its process, so the cost lands on the
    page that is not even asking for it.

    So the loop yields whenever it has held the thread for `slice_s`. On a
    TIME rather than a symbol count, deliberately: a count tuned to today's
    per-symbol cost is a block that grows silently as the rollup gets more
    expensive or the box gets slower, and the property worth keeping is
    "never blocks longer than X", which only a clock can express.

    NOW IS FROZEN FOR THE WHOLE PASS, and that is not laziness about the
    clock. Every window ends at the same instant, so trades arriving mid-pass
    are excluded from every symbol equally and the rows stay comparable to
    each other -- which is the entire premise of a grid you read by scanning
    down it. Taking a fresh clock per chunk would make a row's quietness
    depend on where it happened to fall in the iteration order.
    """
    now_s = time.time() if now_s is None else now_s
    out = {}
    started = time.perf_counter()
    for sym, buf in list(bufs.items()):
        out[sym] = rollup_one(buf, now_s, quiet_window_s=quiet_window_s,
                              slow_window_s=slow_window_s,
                              min_trades=min_trades)
        if time.perf_counter() - started >= slice_s:
            # sleep(0) rather than a real delay: it reschedules this
            # coroutine behind whatever is already runnable, which is exactly
            # "let the reader and the pump have their turn" and nothing more.
            await asyncio.sleep(0)
            started = time.perf_counter()
    return out
