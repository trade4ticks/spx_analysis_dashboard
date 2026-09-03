"""Equities Live — the scrolling tape.

A SEPARATE SERVICE ON A SEPARATE PORT. It shares no process, no database and
no router with the dashboards; the only thing borrowed is the stylesheet, read
off disk. If this crashes, nothing else notices.
"""
from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from live import config, norms
from live.hub import HUB

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("live")


@asynccontextmanager
async def lifespan(app: FastAPI):
    for p in config.problems():
        # Logged, and repeated on the page. A misconfigured feed that starts
        # anyway and shows an empty plot is the failure this project keeps
        # trying to design out.
        log.warning("configuration: %s", p)
    # Stated at startup rather than discovered when the pane is blank. The
    # comparison is read against the MARKET clock, and a host without a zone
    # database would otherwise compare a symbol to the wrong quarter of its
    # own day while looking entirely correct.
    tz = norms.tz_problem()
    if tz:
        log.warning("arrival norms: %s", tz)
    tasks = [asyncio.create_task(HUB.run()), asyncio.create_task(HUB.pump())]
    # Pinned BEFORE anything connects, so a symbol on the list is already
    # buffering by the time a pane asks for it — which is the entire point of
    # pinning rather than watching.
    if config.PINNED:
        refused = await HUB.pin_all(config.PINNED)
        log.info("pinned at startup: %s", ", ".join(sorted(HUB.pinned)) or "none")
        for r in refused:
            log.warning("pin refused: %s", r)
    try:
        yield
    finally:
        await HUB.stop()
        await norms.close()
        for t in tasks:
            t.cancel()


app = FastAPI(title="Equities Live", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=str(config.ROOT / "static")),
          name="static")
templates = Jinja2Templates(directory=str(config.ROOT / "templates"))
templates.env.keep_trailing_newline = True

# The same content-hash cache-buster the dashboards use, imported rather than
# reimplemented — a second copy would drift from the one the other service
# serves, and both read the same static directory.
from app.assets import asset                              # noqa: E402
templates.env.globals["asset"] = asset
templates.env.globals["live_port"] = config.PORT


@app.get("/", response_class=HTMLResponse)
async def page(request: Request):
    return templates.TemplateResponse(request, "equities_live.html")


@app.get("/status")
async def status():
    return HUB.status()


@app.get("/arrival-norm")
async def arrival_norm(symbol: str):
    """This symbol's own normal arrival rate for the current 15-minute bucket.

    THE ONE READ of equities_scalp, and read-only: no writes, no shared pool,
    no router. Everything else in this service still touches no database, and
    a failure here degrades one pane rather than the tape.
    """
    return await norms.arrival_norm(symbol)


@app.websocket("/ws")
async def ws(sock: WebSocket):
    """One browser pane. Sends {action: watch|unwatch, symbol}.

    A client's subscription set is its OWN, so two panes on different symbols
    do not receive each other's tape — the fan-out filters per socket rather
    than broadcasting everything and letting the page discard it.
    """
    await sock.accept()
    if len(HUB.clients) >= config.MAX_CLIENTS:
        await sock.send_json({"ev": "refused",
                              "why": f"at the {config.MAX_CLIENTS}-client cap"})
        await sock.close()
        return

    wanted: set = set()
    HUB.clients[sock] = wanted
    await sock.send_json({"ev": "status", "data": HUB.status()})
    try:
        while True:
            msg = await sock.receive_json()
            act = msg.get("action")
            if act == "watch":
                sym = (msg.get("symbol") or "").strip().upper()
                # IDEMPOTENT PER SOCKET.
                #
                # THE REPORTED FAULT — "CRS is not watched on this
                # connection". The client counted panes per symbol and sent
                # `watch` only on 0->1, `snapshot` after that. But `send()`
                # drops silently when the socket is not open yet, so the first
                # watch could vanish while the count still went to one; the
                # next pane then asked for a snapshot of a symbol the server
                # had never heard of and was refused. Nothing was at a cap,
                # which is why the next symbol worked.
                #
                # The count and the server's set could disagree at all, is the
                # actual bug. Now a repeat `watch` is a snapshot request and
                # nothing else — the client never has to know which case it is
                # in, so the two cannot drift apart.
                if sym not in wanted:
                    err = await HUB.acquire(sym)
                    if err:
                        await sock.send_json({"ev": "refused", "symbol": sym,
                                              "why": err})
                        continue
                    wanted.add(sym)
                # The window's worth of tape that already arrived, so a pane
                # opens onto a populated plot rather than filling in over the
                # next three minutes.
                await sock.send_json({
                    "ev": "snapshot",
                    "data": HUB.snapshot(sym, int(msg.get("window_s")
                                                  or config.DEFAULT_WINDOW_S)),
                })
            elif act == "unwatch":
                sym = (msg.get("symbol") or "").strip().upper()
                if sym in wanted:
                    wanted.discard(sym)
                    await HUB.release(sym)
            elif act == "pin":
                # A pin outlives this socket, so it is not added to `wanted`:
                # the disconnect handler releases everything in there, which
                # is exactly what a pin must survive.
                sym = (msg.get("symbol") or "").strip().upper()
                err = await HUB.pin(sym)
                if err:
                    await sock.send_json({"ev": "refused", "symbol": sym,
                                          "why": err})
                else:
                    await HUB.broadcast({"ev": "pinned",
                                         "data": sorted(HUB.pinned)})
            elif act == "unpin":
                await HUB.unpin((msg.get("symbol") or "").strip().upper())
                await HUB.broadcast({"ev": "pinned",
                                     "data": sorted(HUB.pinned)})
            elif act == "pinned":
                await sock.send_json({"ev": "pinned",
                                      "data": sorted(HUB.pinned)})
            elif act == "status":
                await sock.send_json({"ev": "status", "data": HUB.status()})
    except WebSocketDisconnect:
        pass
    except Exception as exc:                              # noqa: BLE001
        log.info("client socket ended: %s", exc)
    finally:
        HUB.clients.pop(sock, None)
        for sym in list(wanted):
            await HUB.release(sym)
