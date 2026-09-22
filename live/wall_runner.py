"""The wall's clock and fan-out: one frame a second, per socket.

ONE FRAME A SECOND, AND ONE SOCKET A PAGE. Both are requirements rather than
tuning. The service caps browser connections at MAX_CLIENTS, and a wall of a
hundred panes each holding its own would exceed that twelve times over, so the
page opens one connection and this pushes every pane's data down it. A hundred
panes redrawn at 30fps is a page that melts a laptop to show tape a person
reads in glances; a second is the cadence the eye needs to answer "is this one
moving", and pushing faster would only queue frames the browser discards.

WHAT GOES DOWN THE WIRE IS A DELTA, per socket, per symbol. A cursor is a
COUNT of records the socket has been sent, not a timestamp: a busy name prints
several trades inside one millisecond, and "everything after t" would drop all
but the first of them -- silently, and only on the names that print fastest,
which are the ones being watched.

A SYMBOL WITH NOTHING NEW IS NOT SENT. That is most of the wall most of the
time, and it is also what lets the page skip redrawing that pane: no data, no
work. The frame still goes out every tick, empty if it has to, because a page
that cannot tell "quiet" from "the feed stopped" is the failure this whole
project keeps designing out.
"""
from __future__ import annotations

import asyncio
import logging
import time

from live import config
from live.wall_store import WallStore

log = logging.getLogger("live.wall_runner")


def _num(v, places):
    """Round for the wire, and turn NaN into null rather than 'NaN'."""
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if f != f:
        return None
    return round(f, places)


class WallRunner:
    """Owns the wall's tick, its subscribers, and its watchlist."""

    def __init__(self, hub, store: WallStore | None = None):
        self.hub = hub
        self.store = store or WallStore()
        self.ticks = 0
        self.frames = 0
        self.last_at: float | None = None
        self._stop = False
        # socket -> {"cursors": {sym: [trades_sent, quotes_sent, last_q_ms]},
        #            "window_s": float}
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
                st["cursors"].pop(sym, None)
        await self.broadcast({"ev": "watchlist", **self.state()})
        return {"added": added, "dropped": dropped, "refused": refused,
                **self.state()}

    def state(self) -> dict:
        st = self.store.state()
        st["held"] = sorted(self.hub.wall)
        st["caps"] = {"symbols": config.WALL_MAX_SYMBOLS,
                      "retain_s": config.WALL_RETAIN_S,
                      "share_min": config.WALL_SHARE_MIN,
                      "share_max": config.WALL_SHARE_MAX,
                      "tick_s": config.WALL_TICK_S}
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
        """One socket's delta, and the cursor advance that goes with it.

        The cursor is advanced HERE, as the frame is built, from the counts
        the records were read at -- not from what the socket acknowledges.
        There is no acknowledgement to wait for, and a cursor advanced on a
        later read would re-send whatever arrived in between.
        """
        now = time.time() if now is None else now
        now_ms = now * 1000.0
        window_s = float(st.get("window_s") or config.WALL_WINDOW_S)
        cutoff = now_ms - window_s * 1000.0
        spread_ms = config.WALL_SPREAD_WINDOW_S * 1000.0
        cursors = st["cursors"]
        out = {}
        for sym, w in list(self.hub.wall.items()):
            cur = cursors.get(sym)
            fresh = cur is None
            if fresh:
                # Everything in the window, plus the band's state at its left
                # edge -- see WallSym.quote_at for why that one is needed.
                cur = [0, 0, 0.0]
            trades = w.new_trades(cur[0], cutoff)
            quotes = w.new_quotes(cur[1], cutoff)
            last_q = w.last_quote[0] if w.last_quote else 0.0
            cursors[sym] = [w.trades_seen, w.quotes_seen, last_q]
            if not fresh and not trades and not quotes and last_q == cur[2]:
                # NOTHING NEW. Not sent, which is what lets the page leave
                # that pane alone for a second.
                continue
            sp, tp, mid = w.spreads(now_ms, spread_ms)
            cell = {"t": [[round(r[0]), _num(r[1], 4), _num(r[2], 0)]
                          for r in trades],
                    "q": [[round(r[0]), _num(r[1], 4), _num(r[2], 4)]
                          for r in quotes],
                    "sp": _num(sp, 2), "tp": _num(tp, 2), "mid": _num(mid, 4)}
            if fresh:
                cell["full"] = True
                q0 = w.quote_at(cutoff)
                if q0:
                    cell["q0"] = [round(q0[0]), _num(q0[1], 4),
                                  _num(q0[2], 4)]
            out[sym] = cell
        for sym in [s for s in cursors if s not in self.hub.wall]:
            cursors.pop(sym, None)
        return {"ev": "tick", "at": now, "window_s": window_s, "syms": out}

    # ── fan-out ─────────────────────────────────────────────────────────
    def subscribe(self, ws, window_s: float | None = None) -> dict:
        st = {"cursors": {},
              "window_s": float(window_s or config.WALL_WINDOW_S)}
        self._subs[ws] = st
        return st

    def unsubscribe(self, ws) -> None:
        self._subs.pop(ws, None)

    def set_window(self, ws, window_s: float) -> None:
        """A change of window re-sends everything, deliberately.

        Clearing the cursors is what fills the new axis: widening it from one
        minute to three otherwise leaves the two extra minutes blank until
        they happen, which looks exactly like a symbol that stopped trading.
        """
        st = self._subs.get(ws)
        if st is None:
            return
        from live.wall_store import clean_window
        st["window_s"] = clean_window(window_s, st["window_s"])
        st["cursors"].clear()

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
