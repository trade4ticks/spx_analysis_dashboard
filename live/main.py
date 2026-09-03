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

from live import broker, config, norms
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
    # Stated at startup either way, because "why can I not place an order"
    # should be answerable from the journal and not only from the page.
    log.info("live trading: %s",
             "ENABLED" if config.TRADING_ENABLED
             else "disabled (set LIVE_TRADING_ENABLED=1 to allow orders)")
    for p in broker.problems():
        log.warning("broker: %s", p)
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
        await broker.aclose()
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


def _status() -> dict:
    """The hub's status plus whether an order can leave right now.

    Merged HERE rather than inside Hub.status(), so the hub keeps knowing
    nothing about the broker. Both the HTTP endpoint and the WebSocket's
    status frames go through this, or the page and any other caller would be
    reading two different answers to the same question.
    """
    st = HUB.status()
    st["trading"] = broker.trading_state()
    return st


@app.get("/status")
async def status():
    return _status()


# ── the runtime trading switch ──────────────────────────────────────────────
#
# WHY IT EXISTS. Flipping LIVE_TRADING_ENABLED means a restart, and a restart
# drops the upstream socket and every pane's buffer mid-session. Changing your
# mind about being armed should not cost the tape.
#
# The environment variable stays the OUTER gate: with it off, nothing here can
# turn trading on. With it on, this flag decides — and it is off again after
# every start, because a service that comes back armed from a crash nobody
# watched is not a thing to build.
@app.get("/broker/trading")
async def broker_trading_get():
    return broker.trading_state()


@app.post("/broker/trading")
async def broker_trading_set(req: Request):
    """{"enabled": bool, "token": str, "who": str}

    `token` is required to ENABLE and ignored when disabling; it may also be
    sent as the `X-Live-Token` header, which is the better place for it when
    another service is calling. `who` is free text recorded in the state and
    the journal, so "who armed this" is answerable afterwards.
    """
    try:
        b = await req.json()
    except Exception:                                       # noqa: BLE001
        b = {}
    if "enabled" not in b:
        return {"ok": False,
                "why": "the body needs {\"enabled\": true|false}",
                "trading": broker.trading_state()}
    try:
        st = broker.set_trading(
            bool(b.get("enabled")),
            token=(req.headers.get("X-Live-Token") or b.get("token")),
            who=(b.get("who")
                 or (req.client.host if req.client else None)))
        return {"ok": True, "trading": st}
    except Exception as exc:                                # noqa: BLE001
        # The CURRENT state travels with every refusal, so a caller that was
        # refused still learns where things actually stand rather than having
        # to ask again.
        return {"ok": False, "why": str(exc),
                "trading": broker.trading_state()}


@app.get("/arrival-norm")
async def arrival_norm(symbol: str):
    """This symbol's own normal arrival rate for the current 15-minute bucket.

    THE ONE READ of equities_scalp, and read-only: no writes, no shared pool,
    no router. Everything else in this service still touches no database, and
    a failure here degrades one pane rather than the tape.
    """
    return await norms.arrival_norm(symbol)


# ── trading ─────────────────────────────────────────────────────────────────
#
# REST, not the WebSocket, on purpose. An order is a request with a reply and
# a round-trip time the page displays: part of why this exists is to find out
# whether Schwab's path is quicker than the click it replaces, and a fire-and-
# forget socket message could not answer that.
#
# Every response carries `rt_ms` and every failure carries `why` in words the
# pane can show. Nothing here returns a bare 500 — an order-placing endpoint
# that fails opaquely is worse than one that refuses.
def _broker_fail(exc: Exception) -> dict:
    return {"ok": False, "why": str(exc)}


@app.get("/broker/health")
async def broker_health():
    h = broker.health()
    h["problems"] = broker.problems()
    return h


@app.get("/broker/state")
async def broker_state(symbols: str = ""):
    """Positions and working orders, with when they were last confirmed.

    THE PAGE IS NOT THE SOURCE OF TRUTH. `as_of` is the whole point of this
    response: the pane shows its age and says so loudly past
    config.STALE_AFTER_S, because a confident wrong list is the failure that
    costs money here.
    """
    syms = [s for s in (symbols or "").upper().split(",") if s]
    try:
        st = await broker.state(syms or None)
        st["stale_after_s"] = config.STALE_AFTER_S
        return st
    except Exception as exc:                                # noqa: BLE001
        log.warning("broker state failed: %s", exc)
        return _broker_fail(exc)


@app.post("/broker/order")
async def broker_order(req: Request):
    b = await req.json()
    try:
        return await broker.place(
            symbol=str(b.get("symbol") or "").upper(),
            side=str(b.get("side") or "").upper(),
            qty=int(b.get("qty") or 0),
            price=(float(b["price"]) if b.get("price") is not None else None),
            armed=bool(b.get("armed")),
            reference=(float(b["reference"]) if b.get("reference") else None),
            position_qty=float(b.get("position_qty") or 0))
    except Exception as exc:                                # noqa: BLE001
        return _broker_fail(exc)


@app.post("/broker/replace")
async def broker_replace(req: Request):
    b = await req.json()
    try:
        return await broker.replace(
            order_id=str(b.get("order_id") or ""),
            symbol=str(b.get("symbol") or "").upper(),
            side=str(b.get("side") or "").upper(),
            qty=int(b.get("qty") or 0),
            price=float(b.get("price")),
            armed=bool(b.get("armed")),
            reference=(float(b["reference"]) if b.get("reference") else None),
            position_qty=float(b.get("position_qty") or 0))
    except Exception as exc:                                # noqa: BLE001
        return _broker_fail(exc)


@app.post("/broker/cancel")
async def broker_cancel(req: Request):
    """Not gated on arming — see the note on broker.cancel()."""
    b = await req.json()
    try:
        return await broker.cancel(order_id=str(b.get("order_id") or ""))
    except Exception as exc:                                # noqa: BLE001
        return _broker_fail(exc)


@app.post("/broker/flatten")
async def broker_flatten(req: Request):
    b = await req.json()
    try:
        return await broker.flatten(
            symbol=str(b.get("symbol") or "").upper(),
            armed=bool(b.get("armed")))
    except Exception as exc:                                # noqa: BLE001
        return _broker_fail(exc)


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
    await sock.send_json({"ev": "status", "data": _status()})
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
                await sock.send_json({"ev": "status", "data": _status()})
    except WebSocketDisconnect:
        pass
    except Exception as exc:                              # noqa: BLE001
        log.info("client socket ended: %s", exc)
    finally:
        HUB.clients.pop(sock, None)
        for sym in list(wanted):
            await HUB.release(sym)
