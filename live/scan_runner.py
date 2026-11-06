"""The scan's clock: the live column, the finished minute, and the flush.

Three cadences, and they are three because they answer different questions.

  every SCAN_TICK_S   the live "now" state for every symbol. This is what the
                      rightmost column and the sort order read, and it is the
                      thing the page is open for.
  every minute        the cell for the minute that just ENDED, computed with
                      `now` set to the boundary so the 60-second quiet window
                      covers exactly that minute rather than straddling two.
  every minute        the session file, so a deploy does not blank the grid.

WHY THE MINUTE CELL IS NOT JUST THE LAST TICK. A tick lands wherever the timer
fell -- 3 seconds before the boundary, or 4 after. Storing it as "the cell for
09:47" would make each cell a 60-second window ending at an arbitrary offset
into the next minute, so two adjacent cells could share 40 seconds of tape or
none. The grid's whole premise is that a column is a minute, so the boundary
rollup is computed at the boundary.

The extra cost is one pass a minute against twelve, which is under 10% more
work for the property that makes the axis mean anything.

HELD ONLY WHILE A PAGE IS OPEN (2026-11-06). The tier used to keep whatever
list it was last given until the next restart, page open or not -- up to 600
symbols of both channels on the one upstream socket, feeding nobody. Now the
list is REMEMBERED here (`wanted`) and HELD in the hub only while a scan page
is connected; SCAN_IDLE_GRACE_S after the last one closes it is released, and
the next page to connect re-holds it. The grace is what keeps a refresh from
unsubscribing and resubscribing a few hundred symbols. The price is stated:
minutes nobody was watching are not computed, so the grid has a gap for
them, and the first five minutes after a re-hold read off a partly filled
buffer -- exactly as they do after a restart.
"""
from __future__ import annotations

import asyncio
import logging
import time

from live import config
from live.scan_history import (CELL_WIDTH, ScanHistory, minute_index,
                               minute_epoch, session_date)

log = logging.getLogger("live.scan_runner")


class ScanRunner:
    """Owns the scan's periodic work. One instance, started with the app."""

    def __init__(self, hub, history: ScanHistory | None = None):
        self.hub = hub
        self.history = history or ScanHistory(config.SCAN_HISTORY_DIR)
        # symbol -> the 8-tuple from Hub.scan_state: the four trade
        # fields then the four quote ones. The live column.
        self.live: dict[str, tuple] = {}
        self.live_at: float | None = None
        self.ticks = 0
        self.minutes_written = 0
        self.last_minute: int | None = None
        self._stop = False
        self._subscribers: set = set()
        # The list the page last set, kept while the tier is released so the
        # next page to open gets it back. Not persisted: after a restart the
        # page starts empty, as it always has, and Seed refills it.
        self.wanted: list[str] = []
        self.released_at: float | None = None
        self.releases = 0
        self._idle_task: asyncio.Task | None = None

    # ── the loop ────────────────────────────────────────────────────────
    async def run(self) -> None:
        """Tick, close minutes, flush. One task, so the order is decided.

        A minute boundary is detected by the INDEX CHANGING rather than by
        arithmetic on the tick interval: the loop can be late, the box can
        sleep, and a service started at 09:47:58 must not decide it has two
        seconds of minute 09:47 to write. Comparing the index the tick landed
        in against the last one it saw is true regardless of how it got here.
        """
        self.last_minute = minute_index()
        while not self._stop:
            try:
                await self._tick()
            except asyncio.CancelledError:
                raise
            except Exception as exc:                      # noqa: BLE001
                # The scan must not be able to take the tape down. It shares
                # the process, and a bad minute is a gap in a grid where a
                # crash is a dead pane for everyone.
                log.warning("scan tick failed: %s: %s", type(exc).__name__, exc)
            await asyncio.sleep(config.SCAN_TICK_S)

    async def _tick(self) -> None:
        if not self.hub.scan:
            return
        now = time.time()
        state = await self.hub.scan_state(now)
        self.live = state
        self.live_at = now
        self.ticks += 1

        idx = minute_index(now)
        if idx is not None and self.last_minute is not None \
                and idx != self.last_minute:
            await self._close_minute(self.last_minute, idx)
        self.last_minute = idx
        await self._push(state)

    async def _close_minute(self, ended: int, now_idx: int) -> None:
        """Write the finished minute, then flush.

        The boundary time is derived from the minute that ENDED, not from
        `now`: a tick that arrives four seconds late must still write a cell
        whose window covers the minute, or the cell silently includes four
        seconds of the next one.
        """
        boundary = minute_epoch(self.history.date, ended + 1)
        cells = await self.hub.scan_state(boundary)
        for sym, cell in cells.items():
            # THE TRADE FIELDS ONLY. scan_state returns spread alongside them
            # and a cell is CELL_WIDTH wide; slicing at the call site rather
            # than truncating inside write() keeps the decision visible, since
            # "spread is live-only" is a choice and not an implementation
            # detail of the store.
            self.history.write(sym, ended, cell[:CELL_WIDTH])
        self.minutes_written += 1

        # THE SESSION DATE CAN CHANGE UNDER A LONG-RUNNING PROCESS. Rolling to
        # a new file here rather than at startup is what makes a service left
        # running overnight write tomorrow into tomorrow's file instead of
        # appending it to yesterday's.
        today = session_date()
        if today != self.history.date:
            await self.history.flush()
            log.info("scan history: session rolled %s -> %s",
                     self.history.date, today)
            self.history = ScanHistory(config.SCAN_HISTORY_DIR, today)
            self.history.load()
        await self.history.flush()

    # ── holding the list only while it is watched ───────────────────────
    async def set_symbols(self, symbols) -> tuple[list[str], list[str], list[str]]:
        """The page's list: remembered, and held in the hub.

        Held at once even with no page connected (a script, or a page whose
        socket is between reconnects), with the idle timer armed so it does
        not stay held for nobody.
        """
        added, dropped, refused = await self.hub.scan_set(symbols)
        self.wanted = sorted(self.hub.scan)
        self.released_at = None
        if self.hub.scan:
            self._resume_clock()
        if not self._subscribers:
            self._arm_idle()
        return added, dropped, refused

    async def subscribe(self, ws) -> None:
        """A page connected: cancel any pending release, re-hold the list."""
        self._subscribers.add(ws)
        if self._idle_task is not None:
            self._idle_task.cancel()
            self._idle_task = None
        if self.wanted and not self.hub.scan:
            await self.hub.scan_set(self.wanted)
            self._resume_clock()
            log.info("scan: re-held %d symbols for a connected page",
                     len(self.hub.scan))
        self.released_at = None

    def unsubscribe(self, ws) -> None:
        self._subscribers.discard(ws)
        if not self._subscribers:
            self._arm_idle()

    def _resume_clock(self) -> None:
        """Start minute-closing from NOW. While released the loop does not
        tick, so last_minute is stale; left alone, the first tick would close
        that old minute with an empty buffer and write it as a real cell."""
        self.last_minute = minute_index()

    def _arm_idle(self) -> None:
        if self._idle_task is not None or not self.hub.scan:
            return
        self._idle_task = asyncio.get_running_loop().create_task(
            self._release_when_idle())

    async def _release_when_idle(self) -> None:
        try:
            await asyncio.sleep(config.SCAN_IDLE_GRACE_S)
            if self._subscribers or not self.hub.scan:
                return
            self.wanted = sorted(self.hub.scan)
            # Write the minutes already closed to disk before the buffers go.
            # The minute in progress is not a whole minute and is not kept.
            try:
                await self.history.flush()
            except Exception as exc:                      # noqa: BLE001
                log.warning("scan flush before release failed: %s", exc)
            await self.hub.scan_set([])
            self.live = {}
            self.released_at = time.time()
            self.releases += 1
            log.info("scan: no page for %.0fs, released %d symbols "
                     "(kept as the list for the next page)",
                     config.SCAN_IDLE_GRACE_S, len(self.wanted))
        finally:
            if self._idle_task is asyncio.current_task():
                self._idle_task = None

    # ── fan-out ─────────────────────────────────────────────────────────

    async def _push(self, state: dict) -> None:
        """The live column to every open scan page.

        Only the live column. The 120 minutes behind it went out once on
        connect and do not change; re-sending them every five seconds would be
        a megabyte of JSON a minute for data the page already has.
        """
        if not self._subscribers:
            return
        from live.scan_history import _num
        bufs = self.hub.scan
        payload = {"ev": "tick", "at": self.live_at,
                   "minute": self.last_minute,
                   # price rides along on the tick only -- see
                   # SymbolBuf.last_price for why it is not a fifth cell field.
                   # ratio, range_c, $/min, trades, price, spread_c,
                   # spread_bps. Positional rather than named, because this
                   # frame goes out for every symbol on every tick and the
                   # keys would be most of the bytes.
                   "live": {s: [_num(v[0], 3), _num(v[1], 1), _num(v[2], 0),
                                _num(v[3], 0),
                                _num(bufs[s].last_price(), 2)
                                if s in bufs else None,
                                _num(v[4], 2), _num(v[5], 2)]
                            for s, v in state.items()}}
        dead = []
        for ws in list(self._subscribers):
            try:
                await ws.send_json(payload)
            except Exception:                             # noqa: BLE001
                dead.append(ws)
        for d in dead:
            self._subscribers.discard(d)

    def status(self) -> dict:
        return {
            "ticks": self.ticks,
            "tick_s": config.SCAN_TICK_S,
            "live_symbols": len(self.live),
            "live_age_s": (time.time() - self.live_at) if self.live_at else None,
            "minutes_written": self.minutes_written,
            "current_minute": self.last_minute,
            "subscribers": len(self._subscribers),
            "held": len(self.hub.scan),
            "wanted": len(self.wanted),
            "idle_grace_s": config.SCAN_IDLE_GRACE_S,
            "release_pending": self._idle_task is not None,
            "released_at": self.released_at,
            "releases": self.releases,
            "history": self.history.status(),
        }

    async def stop(self) -> None:
        self._stop = True
        if self._idle_task is not None:
            self._idle_task.cancel()
        # A LAST FLUSH ON THE WAY DOWN. A deploy is the common shutdown, and
        # losing the final minute to it is losing exactly the minutes someone
        # was looking at when they pushed.
        try:
            await self.history.flush()
        except Exception as exc:                          # noqa: BLE001
            log.warning("final scan flush failed: %s", exc)
