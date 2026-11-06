"""The wall's clock and fan-out: four frames a second, one socket a page.

ONE SOCKET A PAGE. The service caps browser connections at MAX_CLIENTS, so
the page opens one connection and this pushes every row down it.

WHAT GOES DOWN THE WIRE IS A DELTA, per socket: the symbols whose spread, in
cents, differs from what THAT socket was last sent. A quote that moves the
bid and the ask together changes nothing on the page and is not sent. A new
socket is sent every symbol once, a null for one that has not quoted yet.

THE FRAME STILL GOES OUT EVERY TICK, empty if it has to, because a page that
cannot tell "nothing changed" from "the feed stopped" is the failure this
project keeps designing out.
"""
from __future__ import annotations

import asyncio
import logging
import time

from live import config
from live.wall_store import WallStore

log = logging.getLogger("live.wall_runner")


_UNSENT = object()


class WallRunner:
    """Owns the wall's tick, its subscribers, and its watchlist."""

    def __init__(self, hub, store: WallStore | None = None):
        self.hub = hub
        self.store = store or WallStore()
        self.ticks = 0
        self.frames = 0
        self.last_at: float | None = None
        self._stop = False
        # socket -> {"sent": {sym: the spread last sent to it}}
        self._subs: dict = {}

    # ── the watchlist ───────────────────────────────────────────────────
    async def restore(self) -> int:
        """Load the saved list and hold it. Called once, at startup."""
        n = self.store.load()
        if self.store.symbols():
            await self.hub.wall_set(self.store.symbols())
        return n

    async def apply(self, entries, settings=None) -> dict:
        """Set the watchlist: clean it, hold it, save it, tell every page.

        ONE PATH, so the three cannot disagree. A list that reached the hub
        but not the file comes back changed after a deploy; a list that
        reached the file but not the hub draws panes with no data in them;
        and a second tab that never hears about either is editing a list that
        no longer exists.
        """
        refused = self.store.set(entries, settings)
        added, dropped, hub_refused = await self.hub.wall_set(
            self.store.symbols())
        # A symbol the hub refused is not in the tier, so it must not stay in
        # the saved list pretending to be watched.
        if hub_refused:
            held = set(self.hub.wall)
            self.store.entries = [e for e in self.store.entries
                                  if e["symbol"] in held]
            refused = refused + hub_refused
        self.store.save()
        for st in self._subs.values():
            for sym in dropped:
                st["sent"].pop(sym, None)
        await self.broadcast({"ev": "watchlist", **self.state()})
        return {"added": added, "dropped": dropped, "refused": refused,
                **self.state()}

    def state(self) -> dict:
        st = self.store.state()
        st["held"] = sorted(self.hub.wall)
        st["caps"] = {"symbols": config.WALL_MAX_SYMBOLS,
                      "tick_s": config.WALL_TICK_S,
                      "floor_min": config.WALL_FLOOR_MIN,
                      "floor_max": config.WALL_FLOOR_MAX,
                      "wide_min": config.WALL_WIDE_MIN,
                      "wide_max": config.WALL_WIDE_MAX,
                      "wide_step": config.WALL_WIDE_STEP}
        return st

    # ── the loop ────────────────────────────────────────────────────────
    async def run(self) -> None:
        while not self._stop:
            try:
                await self._tick()
            except asyncio.CancelledError:
                raise
            except Exception as exc:                      # noqa: BLE001
                # The wall must not be able to take the tape down: it shares
                # the process with the pane that has live orders on it.
                log.warning("wall tick failed: %s: %s",
                            type(exc).__name__, exc)
            await asyncio.sleep(config.WALL_TICK_S)

    async def _tick(self) -> None:
        if not self._subs:
            return
        now = time.time()
        self.ticks += 1
        self.last_at = now
        dead = []
        for ws, st in list(self._subs.items()):
            try:
                await ws.send_json(self.frame(st, now))
                self.frames += 1
            except Exception:                             # noqa: BLE001
                dead.append(ws)
        for d in dead:
            self._subs.pop(d, None)

    def frame(self, st: dict, now: float | None = None) -> dict:
        """One socket's delta: {sym: spread in cents, or null}.

        `sent` is advanced HERE, as the frame is built. There is no
        acknowledgement to wait for, and advancing it later would drop
        whatever changed in between.
        """
        now = time.time() if now is None else now
        sent = st["sent"]
        out = {}
        for sym, w in list(self.hub.wall.items()):
            sp = w.spread_cents()
            if sent.get(sym, _UNSENT) == sp:
                continue
            sent[sym] = sp
            out[sym] = sp
        for sym in [s for s in sent if s not in self.hub.wall]:
            sent.pop(sym, None)
        return {"ev": "tick", "at": now, "syms": out}

    # ── fan-out ─────────────────────────────────────────────────────────
    def subscribe(self, ws) -> dict:
        st = {"sent": {}}
        self._subs[ws] = st
        return st

    def unsubscribe(self, ws) -> None:
        self._subs.pop(ws, None)

    async def broadcast(self, payload: dict) -> None:
        dead = []
        for ws in list(self._subs):
            try:
                await ws.send_json(payload)
            except Exception:                             # noqa: BLE001
                dead.append(ws)
        for d in dead:
            self._subs.pop(d, None)

    def status(self) -> dict:
        return {
            "ticks": self.ticks,
            "frames": self.frames,
            "tick_s": config.WALL_TICK_S,
            "subscribers": len(self._subs),
            "age_s": (time.time() - self.last_at) if self.last_at else None,
            "store": self.store.status(),
        }

    async def stop(self) -> None:
        self._stop = True
