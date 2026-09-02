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
            elif act == "snapshot":
                # A SECOND PANE on a symbol this socket already holds. It must
                # not acquire again: the hub reference-counts acquisitions
                # while `wanted` is a set, so a second acquire would leave the
                # count at two against one set entry and the eventual unwatch
                # would strand the subscription. The backlog is all the new
                # pane needs.
                sym = (msg.get("symbol") or "").strip().upper()
                if sym not in wanted:
                    await sock.send_json({
                        "ev": "refused", "symbol": sym,
                        "why": f"{sym} is not watched on this connection"})
                    continue
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
