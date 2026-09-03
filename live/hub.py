"""One upstream socket, fanned out to every browser watching.

WHY ONE CONNECTION. Polygon multiplexes subscriptions over a single socket, so
a connection per pane would be several authentications and several reconnect
storms for the same data. Symbols are reference-counted: the last pane to drop
a symbol unsubscribes it, and nothing else does.

NOTHING IS AGGREGATED. Every trade is kept at its own timestamp with its own
price and size. Seven prints at .401 stay seven prints at .401 — that is a
single marketable order sweeping seven venues, and collapsing it is exactly
what made the other tools useless here. The only batching is on the WIRE:
records accumulate for FLUSH_MS and go out in one frame, which changes how
often the browser is spoken to and not what it is told.

CAPS ARE STRUCTURAL. Every buffer is bounded in both time and count, and the
symbol table is bounded too. The box has been OOM-killed twice this week.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from collections import deque

import websockets

from live import config

log = logging.getLogger("live.hub")


class Hub:
    def __init__(self):
        # symbol -> deque of records, newest last.
        self.trades: dict[str, deque] = {}
        self.quotes: dict[str, deque] = {}
        # symbol -> number of things holding it: panes, plus one for a pin.
        self.refs: dict[str, int] = {}
        # PINNED SYMBOLS STAY SUBSCRIBED with nothing watching them.
        #
        # Closing the browser used to drop the last reference and unsubscribe,
        # so the buffer went with it and the next pane opened on "buffering
        # 55s of 180s" — three minutes of axis over fifty seconds of tape,
        # every time. A pin is simply a reference that no pane owns and no
        # pane can release, which is why nothing in acquire/release needs a
        # special case for it.
        #
        # Memory is the whole cost and it is small: at the 15-minute ceiling a
        # busy name is ~5,100 records, and a record is six numbers.
        self.pinned: set[str] = set()
        # socket -> the symbols THAT socket asked for. A dict rather than a
        # set of pairs: the value is itself a set, which is unhashable, so the
        # obvious set-of-tuples does not survive contact with Python.
        self.clients: dict = {}

        self.connected = False
        self.authed = False
        self.last_error: str | None = None
        self.connected_at: float | None = None
        self.reconnects = 0
        # Counted rather than inferred: "is anything arriving" is the first
        # question when a plot looks wrong, and an empty window answers it
        # ambiguously.
        self.msgs_in = 0
        self.trades_in = 0
        self.quotes_in = 0
        self.dropped_cap = 0

        self._ws = None
        self._pending: dict[str, dict] = {}
        self._lock = asyncio.Lock()
        self._stop = False

    # ── subscriptions ───────────────────────────────────────────────────
    async def acquire(self, symbol: str) -> str | None:
        """Reference a symbol. Returns an error string, or None on success."""
        sym = (symbol or "").strip().upper()
        if not sym or not sym.isalnum():
            return f"{symbol!r} is not a symbol."
        async with self._lock:
            if sym not in self.refs and len(self.refs) >= config.MAX_SYMBOLS:
                # Refused, not silently ignored. A pane that quietly shows
                # nothing is indistinguishable from a quiet tape.
                return (f"at the {config.MAX_SYMBOLS}-symbol cap "
                        f"({', '.join(sorted(self.refs))}); close a pane first.")
            first = sym not in self.refs
            self.refs[sym] = self.refs.get(sym, 0) + 1
            self.trades.setdefault(sym, deque())
            self.quotes.setdefault(sym, deque())
        if first:
            await self._send({"action": "subscribe",
                              "params": f"T.{sym},Q.{sym}"})
        return None

    async def release(self, symbol: str) -> None:
        sym = (symbol or "").strip().upper()
        async with self._lock:
            n = self.refs.get(sym, 0) - 1
            if n > 0:
                self.refs[sym] = n
                return
            self.refs.pop(sym, None)
            self.trades.pop(sym, None)
            self.quotes.pop(sym, None)
        await self._send({"action": "unsubscribe", "params": f"T.{sym},Q.{sym}"})

    # ── pins ────────────────────────────────────────────────────────────
    async def pin(self, symbol: str) -> str | None:
        """Hold a symbol regardless of whether any pane wants it."""
        sym = (symbol or "").strip().upper()
        if sym in self.pinned:
            return None
        err = await self.acquire(sym)
        if err:
            return err
        self.pinned.add(sym)
        log.info("pinned %s (%d pinned, %d subscribed)",
                 sym, len(self.pinned), len(self.refs))
        return None

    async def unpin(self, symbol: str) -> None:
        sym = (symbol or "").strip().upper()
        if sym not in self.pinned:
            return
        self.pinned.discard(sym)
        # Releases the PIN's reference only. A pane still watching the symbol
        # keeps it alive, which is the same rule as every other holder.
        await self.release(sym)

    async def pin_all(self, symbols) -> list[str]:
        """Pin the configured set at startup. Returns the refusals."""
        out = []
        for s in symbols:
            err = await self.pin(s)
            if err:
                out.append(f"{s}: {err}")
        return out

    async def broadcast(self, payload: dict) -> None:
        """One message to every client. Used when the pin set changes.

        A second tab that pinned nothing still has to see the pin, or the two
        tabs disagree about what is being held and one of them is wrong.
        """
        for ws in list(self.clients):
            try:
                await ws.send_json(payload)
            except Exception:                             # noqa: BLE001
                self.clients.pop(ws, None)

    async def _send(self, payload: dict) -> None:
        ws = self._ws
        if ws is None:
            return
        try:
            await ws.send(json.dumps(payload))
        except Exception as exc:                          # noqa: BLE001
            log.warning("upstream send failed: %s", exc)

    # ── the upstream connection ─────────────────────────────────────────
    async def run(self) -> None:
        delay = config.RECONNECT_BASE_S
        while not self._stop:
            try:
                await self._session()
                delay = config.RECONNECT_BASE_S
            except asyncio.CancelledError:
                raise
            except Exception as exc:                      # noqa: BLE001
                self.last_error = f"{type(exc).__name__}: {exc}"
                log.warning("upstream session ended: %s", self.last_error)
            finally:
                self.connected = self.authed = False
                self._ws = None
            if self._stop:
                break
            self.reconnects += 1
            # Backoff, capped. A tight retry loop against a socket that is
            # refusing is how a service that "reconnects" becomes the reason
            # the box is unresponsive.
            await asyncio.sleep(delay)
            delay = min(delay * 2, config.RECONNECT_MAX_S)

    async def _session(self) -> None:
        url = config.feed_url()
        async with websockets.connect(url, ping_interval=20,
                                      ping_timeout=20,
                                      max_queue=4096) as ws:
            self._ws = ws
            self.connected = True
            self.connected_at = time.time()
            self.last_error = None
            await ws.send(json.dumps({"action": "auth",
                                      "params": config.API_KEY}))
            # Re-subscribe everything on every connect: after a drop the
            # server remembers nothing, and a reconnect that restores the
            # socket without the subscriptions is a live-looking dead plot.
            async with self._lock:
                syms = sorted(self.refs)
            if syms:
                params = ",".join(f"T.{s},Q.{s}" for s in syms)
                await ws.send(json.dumps({"action": "subscribe",
                                          "params": params}))
            async for raw in ws:
                self._ingest(raw)

    def _ingest(self, raw) -> None:
        try:
            msgs = json.loads(raw)
        except Exception:                                 # noqa: BLE001
            return
        if isinstance(msgs, dict):
            msgs = [msgs]
        now_ms = time.time() * 1000.0
        for m in msgs:
            self.msgs_in += 1
            ev = m.get("ev")
            if ev == "status":
                if m.get("status") in ("auth_success", "connected"):
                    self.authed = m.get("status") == "auth_success" or self.authed
                elif m.get("status") in ("auth_failed", "error"):
                    self.last_error = str(m.get("message") or m.get("status"))
                continue
            if ev == "T":
                sym = m.get("sym")
                if sym in self.trades:
                    rec = {"t": m.get("t"), "p": m.get("p"), "s": m.get("s"),
                           "x": m.get("x"), "z": m.get("z"),
                           "c": m.get("c") or []}
                    self.trades_in += 1
                    # Only forwarded if it SURVIVED the caps. Reading the
                    # buffer's tail back assumed the record just appended was
                    # still there; a record older than the window is evicted
                    # by the same call that added it, and the tail was then
                    # either someone else's record or an IndexError on an
                    # empty deque.
                    if self._push(self.trades[sym], rec,
                                  config.MAX_TRADES_PER_SYMBOL, now_ms):
                        self._pending.setdefault(
                            sym, {"t": [], "q": []})["t"].append(rec)
            elif ev == "Q":
                sym = m.get("sym")
                if sym in self.quotes:
                    rec = {"t": m.get("t"), "bp": m.get("bp"),
                           "ap": m.get("ap"), "bs": m.get("bs"),
                           "as": m.get("as")}
                    self.quotes_in += 1
                    if self._push(self.quotes[sym], rec,
                                  config.MAX_QUOTES_PER_SYMBOL, now_ms):
                        self._pending.setdefault(
                            sym, {"t": [], "q": []})["q"].append(rec)

    def _push(self, buf: deque, rec: dict, cap: int, now_ms: float) -> bool:
        """Append and trim. Returns whether the record survived.

        A record can be evicted by the very call that adds it — an out-of-
        window timestamp, or a burst that overruns the count cap — and the
        caller must not forward what is no longer held.
        """
        buf.append(rec)
        # BOTH bounds, whichever binds first. Time alone does not hold when a
        # halt reopens and a minute of tape arrives in a second; count alone
        # would keep an hour of a quiet name.
        horizon = now_ms - config.MAX_WINDOW_S * 1000.0
        while buf and (buf[0].get("t") or 0) < horizon:
            buf.popleft()
        while len(buf) > cap:
            buf.popleft()
            self.dropped_cap += 1
        return bool(buf) and buf[-1] is rec

    # ── fan-out ─────────────────────────────────────────────────────────
    async def pump(self) -> None:
        """Flush accumulated records to every client on a fixed cadence."""
        while not self._stop:
            await asyncio.sleep(config.FLUSH_MS / 1000.0)
            if not self.clients:
                self._pending.clear()
                continue
            batch, self._pending = self._pending, {}
            if not batch:
                continue
            dead = []
            for ws, wanted in list(self.clients.items()):
                payload = {s: batch[s] for s in batch if s in wanted}
                if not payload:
                    continue
                try:
                    await ws.send_json({"ev": "batch", "data": payload})
                except Exception:                         # noqa: BLE001
                    dead.append(ws)
            for d in dead:
                self.clients.pop(d, None)

    def snapshot(self, symbol: str, window_s: int) -> dict:
        """Everything inside the window, for a pane that has just opened."""
        sym = (symbol or "").upper()
        cutoff = time.time() * 1000.0 - min(window_s, config.MAX_WINDOW_S) * 1000
        tr = [r for r in self.trades.get(sym, ()) if (r.get("t") or 0) >= cutoff]
        qt = [r for r in self.quotes.get(sym, ()) if (r.get("t") or 0) >= cutoff]
        return {"symbol": sym, "trades": tr, "quotes": qt}

    def status(self) -> dict:
        return {
            "connected": self.connected,
            "authed": self.authed,
            "delayed": config.feed_is_delayed(),
            "feed": config.FEED,
            "url": config.feed_url(),
            "error": self.last_error,
            "reconnects": self.reconnects,
            "uptime_s": (time.time() - self.connected_at)
                        if self.connected_at and self.connected else 0,
            "symbols": sorted(self.refs),
            "pinned": sorted(self.pinned),
            "msgs_in": self.msgs_in,
            "trades_in": self.trades_in,
            "quotes_in": self.quotes_in,
            "dropped_cap": self.dropped_cap,
            "caps": {"symbols": config.MAX_SYMBOLS,
                     "window_s": config.MAX_WINDOW_S,
                     "trades": config.MAX_TRADES_PER_SYMBOL,
                     "quotes": config.MAX_QUOTES_PER_SYMBOL,
                     "clients": config.MAX_CLIENTS},
            "problems": config.problems(),
        }

    async def stop(self) -> None:
        self._stop = True
        if self._ws is not None:
            try:
                await self._ws.close()
            except Exception:                             # noqa: BLE001
                pass


HUB = Hub()
