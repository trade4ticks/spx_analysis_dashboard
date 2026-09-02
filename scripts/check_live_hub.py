"""The tape hub, against a fabricated upstream.

NO MARKET NEEDED, and that is the point: the parts most likely to be wrong —
reference counting, the memory caps, resubscribing after a drop, and the
promise that nothing is aggregated — are all decidable from a synthetic
message stream, and none of them can be checked by looking at a live plot and
deciding it looks about right.

WHAT IS BEING PROTECTED, in order of how bad it would be:

  * NOTHING IS AGGREGATED. Seven prints sharing a millisecond must survive as
    seven records. Collapsing them is what made every other tool useless for
    this, and it is the kind of thing an optimisation adds back silently.
  * THE CAPS HOLD. The box has been OOM-killed twice this week; a buffer that
    grows without bound because the time bound alone was trusted is how a
    third happens.
  * A RECONNECT RESTORES THE SUBSCRIPTIONS. A socket that comes back without
    them is a live-looking dead plot, which is worse than staying down.
"""
from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from live import config                                   # noqa: E402
from live.hub import Hub                                  # noqa: E402

FAILS: list[str] = []


def check(cond, msg):
    if not cond:
        FAILS.append(msg)


def trade(sym, t, p, s, x=4):
    return {"ev": "T", "sym": sym, "t": t, "p": p, "s": s, "x": x, "z": 1}


def quote(sym, t, bp, ap):
    return {"ev": "Q", "sym": sym, "t": t, "bp": bp, "ap": ap,
            "bs": 1, "as": 1}


async def case_no_aggregation():
    """Seven prints in one millisecond stay seven prints."""
    h = Hub()
    await h.acquire("FDX")
    now = time.time() * 1000
    burst = [trade("FDX", now, 334.40 + i * 0.01, 1 + i) for i in range(7)]
    h._ingest(json.dumps(burst))
    got = list(h.trades["FDX"])
    check(len(got) == 7,
          f"seven prints at one timestamp became {len(got)} records — the "
          f"clustering IS the information, and collapsing it is what made "
          f"the other tools useless here")
    check(len({(r["p"], r["s"]) for r in got}) == 7,
          "prints at one timestamp were merged into fewer distinct records")
    check([r["t"] for r in got] == [now] * 7,
          "timestamps were altered; every trade must plot at its own")


async def case_odd_lots_survive():
    """A 1-share print is not noise to be filtered."""
    h = Hub()
    await h.acquire("FDX")
    now = time.time() * 1000
    h._ingest(json.dumps([trade("FDX", now, 334.4, 1),
                          trade("FDX", now + 1, 334.4, 200)]))
    sizes = sorted(r["s"] for r in h.trades["FDX"])
    check(sizes == [1, 200],
          f"sizes {sizes} — 91% of this tape is under 40 shares and it is the "
          f"part being traded in; a minimum size anywhere is the bug every "
          f"other tool has")


async def case_count_cap():
    """The COUNT bound holds when the time bound cannot."""
    h = Hub()
    await h.acquire("FDX")
    cap = config.MAX_TRADES_PER_SYMBOL
    now = time.time() * 1000
    # All inside the window, so only the count bound can stop this.
    h._ingest(json.dumps([trade("FDX", now, 334.4, 1)
                          for _ in range(cap + 500)]))
    n = len(h.trades["FDX"])
    check(n <= cap,
          f"{n} trades held against a cap of {cap} — a halt reopening puts a "
          f"minute of tape into a second and the time bound alone does not "
          f"hold")
    check(h.dropped_cap > 0,
          "records were dropped at the cap without being counted, so the "
          "page cannot say the window is truncated")


async def case_time_cap():
    """Records older than the ceiling leave, whatever the count."""
    h = Hub()
    await h.acquire("FDX")
    now = time.time() * 1000
    old = now - (config.MAX_WINDOW_S + 120) * 1000
    h._ingest(json.dumps([trade("FDX", old, 334.4, 1)]))
    h._ingest(json.dumps([trade("FDX", now, 334.5, 1)]))
    ts = [r["t"] for r in h.trades["FDX"]]
    check(old not in ts,
          "a record older than the window ceiling was retained")
    check(now in ts, "the current record was dropped")


async def case_symbol_cap_refuses():
    """Over the cap is REFUSED with a reason, never silently ignored."""
    h = Hub()
    for i in range(config.MAX_SYMBOLS):
        err = await h.acquire(f"SYM{i}")
        check(err is None, f"acquiring symbol {i} failed: {err}")
    err = await h.acquire("ONEMORE")
    check(err is not None,
          "the symbol cap did not refuse — a pane that quietly shows nothing "
          "is indistinguishable from a quiet tape")
    check(err and "cap" in err.lower(), f"the refusal does not say why: {err}")
    check("ONEMORE" not in h.trades,
          "a refused symbol still allocated a buffer")


async def case_refcount():
    """The last pane to drop a symbol is the one that unsubscribes it."""
    h = Hub()
    await h.acquire("FDX")
    await h.acquire("FDX")
    await h.release("FDX")
    check("FDX" in h.refs,
          "a symbol was released while another pane still watched it")
    await h.release("FDX")
    check("FDX" not in h.refs, "the last release did not drop the symbol")
    check("FDX" not in h.trades, "the buffer outlived its subscription")


async def case_two_panes_one_symbol():
    """Two panes on one symbol must leave nothing subscribed behind them.

    THE FAULT THIS EXISTS FOR. The hub reference-counts acquisitions, but a
    socket's subscription set is a SET — so a second `watch` for a symbol the
    socket already holds raises the hub's count to two while the set stays at
    one. The client sends one `unwatch` when the last pane closes, the set
    entry goes, and the count drops to one and stays there: the symbol is
    subscribed forever with nobody receiving it, occupying one of four slots
    until the service restarts.

    The second pane asks for `snapshot` instead, which must NOT acquire.
    """
    import live.main as live_main

    class FakeSock:
        def __init__(self, script):
            self.script, self.sent, self.i = script, [], 0

        async def accept(self): pass
        async def send_json(self, o): self.sent.append(o)
        async def close(self): pass

        async def receive_json(self):
            if self.i >= len(self.script):
                from fastapi import WebSocketDisconnect
                raise WebSocketDisconnect(1000)
            self.i += 1
            return self.script[self.i - 1]

    h = live_main.HUB
    h.refs.clear(); h.trades.clear(); h.quotes.clear(); h.clients.clear()

    sock = FakeSock([
        {"action": "watch", "symbol": "FDX"},      # pane one
        {"action": "snapshot", "symbol": "FDX"},   # pane two, same symbol
        {"action": "unwatch", "symbol": "FDX"},    # the last pane closes
    ])
    await live_main.ws(sock)

    snaps = [m for m in sock.sent if m.get("ev") == "snapshot"]
    check(len(snaps) == 2,
          f"two panes on one symbol got {len(snaps)} snapshots — the second "
          f"opens onto an empty plot and fills in over three minutes")
    check("FDX" not in h.refs,
          f"FDX is still subscribed after every pane closed ({h.refs}) — "
          f"a stranded subscription holds one of "
          f"{config.MAX_SYMBOLS} slots until the service restarts")
    check(not h.clients, "the client was not removed on disconnect")

    # A snapshot for something this socket does not hold must be refused, not
    # quietly served: it would be a subscription the hub is not counting.
    h.refs.clear(); h.trades.clear(); h.quotes.clear(); h.clients.clear()
    sock2 = FakeSock([{"action": "snapshot", "symbol": "NVDA"}])
    await live_main.ws(sock2)
    refused = [m for m in sock2.sent if m.get("ev") == "refused"]
    check(len(refused) == 1,
          "a snapshot for an unwatched symbol was served rather than refused")
    check("NVDA" not in h.refs,
          "a bare snapshot acquired a symbol the hub is not counting")


async def case_resubscribe_on_reconnect():
    """A reconnect must restore every subscription."""
    sent = []

    class FakeWS:
        async def send(self, s): sent.append(json.loads(s))
        def __aiter__(self): return self
        async def __anext__(self): raise StopAsyncIteration
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return False

    h = Hub()
    await h.acquire("FDX")
    await h.acquire("EXPE")
    sent.clear()

    import live.hub as hubmod
    real = hubmod.websockets.connect
    hubmod.websockets.connect = lambda *a, **k: FakeWS()
    try:
        await h._session()
    finally:
        hubmod.websockets.connect = real

    subs = [m for m in sent if m.get("action") == "subscribe"]
    check(subs, "no subscribe was sent on connect — the socket would come "
                "back with no data and the plot would look merely quiet")
    params = ",".join(m["params"] for m in subs)
    for sym in ("FDX", "EXPE"):
        check(f"T.{sym}" in params, f"{sym} trades were not resubscribed")
        check(f"Q.{sym}" in params, f"{sym} quotes were not resubscribed")
    check(any(m.get("action") == "auth" for m in sent),
          "the session did not authenticate")


async def case_snapshot_window():
    """A pane opens onto the window, not onto everything held."""
    h = Hub()
    await h.acquire("FDX")
    now = time.time() * 1000
    h._ingest(json.dumps([trade("FDX", now - 600_000, 334.0, 1),
                          trade("FDX", now - 1_000, 334.5, 1)]))
    snap = h.snapshot("FDX", 180)
    check(len(snap["trades"]) == 1,
          f"snapshot returned {len(snap['trades'])} trades for a 180s window "
          f"that contains one")


async def case_status_states_the_feed():
    """Delayed must be announced, not inferred."""
    h = Hub()
    st = h.status()
    check("delayed" in st and "feed" in st and "url" in st,
          "status does not say which feed it is on — a 15-minute-old tape "
          "renders identically to a live one")
    check("caps" in st, "status does not report the caps, so a truncated "
                        "window cannot be explained")
    check(isinstance(st.get("problems"), list),
          "status does not carry configuration problems")


CASES = [
    ("no aggregation",          case_no_aggregation),
    ("odd lots survive",        case_odd_lots_survive),
    ("count cap holds",         case_count_cap),
    ("time cap holds",          case_time_cap),
    ("symbol cap refuses",      case_symbol_cap_refuses),
    ("subscription refcount",   case_refcount),
    ("two panes, one symbol",   case_two_panes_one_symbol),
    ("resubscribe on connect",  case_resubscribe_on_reconnect),
    ("snapshot honours window", case_snapshot_window),
    ("status names the feed",   case_status_states_the_feed),
]


async def main() -> int:
    for name, fn in CASES:
        before = len(FAILS)
        try:
            await fn()
        except Exception as exc:                          # noqa: BLE001
            FAILS.append(f"{name}: raised {type(exc).__name__}: {exc}")
        for m in FAILS[before:]:
            print(f"  FAIL {name}: {m}")
    print(f"\nhub cases: {len(CASES)}, failures: {len(FAILS)}")
    return 1 if FAILS else 0


sys.exit(asyncio.run(main()))
