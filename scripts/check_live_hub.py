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


class FakeSock:
    """A browser socket, scripted. Disconnects when the script runs out."""

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


def _fresh_hub():
    import live.main as live_main
    h = live_main.HUB
    h.refs.clear(); h.trades.clear(); h.quotes.clear()
    h.clients.clear(); h.pinned.clear()
    return live_main, h


async def case_repeat_watch_is_idempotent():
    """A repeat `watch` from one socket must snapshot, not acquire again.

    THE FAULT THIS EXISTS FOR — reported as "CRS is not watched on this
    connection", with nothing at any cap.

    The client counted panes per symbol and sent `watch` on 0->1 and a
    different verb after. But the browser drops a send when the socket is not
    open yet, so the first watch could vanish while the count still went to
    one; the next pane then asked about a symbol the server had never held.
    The count being part of the protocol at all was the bug.

    So the server now takes any number of watches per socket. Two properties
    have to hold together, and they pull in opposite directions:

      * the SECOND watch still returns a snapshot, or a second pane opens
        onto an empty plot and fills in over three minutes;

      * the second watch does NOT acquire, or the hub's count reaches two
        against one entry in the socket's set, the single unwatch drops it to
        one, and the symbol stays subscribed with nobody receiving it —
        holding one of the symbol slots until the service restarts.
    """
    live_main, h = _fresh_hub()

    sock = FakeSock([
        {"action": "watch", "symbol": "FDX"},      # pane one
        {"action": "watch", "symbol": "FDX"},      # pane two, same symbol
        {"action": "watch", "symbol": "FDX"},      # and a stray repeat
        {"action": "unwatch", "symbol": "FDX"},    # the last pane closes
    ])
    await live_main.ws(sock)

    snaps = [m for m in sock.sent if m.get("ev") == "snapshot"]
    check(len(snaps) == 3,
          f"three watches produced {len(snaps)} snapshots — a later pane on a "
          f"held symbol opens onto an empty plot")
    check(not [m for m in sock.sent if m.get("ev") == "refused"],
          "a repeat watch was refused; the client cannot know which case it "
          "is in, which is what produced 'CRS is not watched'")
    check("FDX" not in h.refs,
          f"FDX is still subscribed after every pane closed ({h.refs}) — a "
          f"stranded subscription holds one of {config.MAX_SYMBOLS} slots "
          f"until the service restarts")
    check(not h.clients, "the client was not removed on disconnect")


async def case_pins_outlive_the_socket():
    """A pinned symbol stays subscribed with nothing watching it.

    The reason pins exist: the last pane closing dropped the last reference,
    the buffer went with the subscription, and the next pane opened on
    "buffering 55s of 180s" — three minutes of axis over fifty seconds of
    tape, every time.

    A pin is a reference no pane owns, so what has to be checked is that the
    disconnect handler does not release it along with everything the socket
    did own.
    """
    live_main, h = _fresh_hub()

    sock = FakeSock([
        {"action": "pin", "symbol": "FDX"},
        {"action": "watch", "symbol": "FDX"},      # a pane on the pinned name
        {"action": "watch", "symbol": "NVDA"},     # and one that is not
    ])
    await live_main.ws(sock)

    check("FDX" in h.refs and "FDX" in h.pinned,
          f"the pin did not survive the socket closing: refs={h.refs} "
          f"pinned={h.pinned} — the buffer is gone and the next pane starts "
          f"from empty, which is the whole thing pinning is for")
    check("NVDA" not in h.refs,
          f"an unpinned symbol outlived its only pane ({h.refs})")
    check("FDX" in h.trades, "the pinned symbol kept no buffer")

    # Unpinning is the only thing that can drop it, and it drops only the
    # pin's own reference.
    await h.pin("NVDA")
    await h.acquire("NVDA")                        # a pane arrives too
    await h.unpin("NVDA")
    check("NVDA" in h.refs,
          "unpinning released a symbol a pane was still watching")
    await h.release("NVDA")
    check("NVDA" not in h.refs,
          "the pane's own release did not drop the unpinned symbol")

    # Pins are bounded by the same symbol cap; they are subscriptions.
    live_main, h = _fresh_hub()
    refused = await h.pin_all([f"SYM{i}" for i in range(config.MAX_SYMBOLS + 3)])
    check(len(h.pinned) <= config.MAX_SYMBOLS,
          f"{len(h.pinned)} symbols pinned against a cap of "
          f"{config.MAX_SYMBOLS} — pins are subscriptions and buffers")
    check(len(refused) == 3,
          f"pinning past the cap reported {len(refused)} refusals, expected 3")
    check(all("cap" in r.lower() for r in refused),
          f"a refusal past the cap does not say why: {refused}")


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


# -- the scan tier -----------------------------------------------------------
#
# The scan holds hundreds of symbols on the SAME socket as the tape, because
# the account permits exactly one -- measured: a second connection
# authenticates, is accepted for every subscription, and is then closed with
# 1008 while the incumbent reconnects and evicts it in turn.
#
# That makes the two tiers share a subscription set, and every case below is
# some way that sharing goes wrong QUIETLY. None of them raises; each one
# produces a page that looks like it is working.


class SpySock:
    """The upstream socket, recording what the hub sends it."""

    def __init__(self):
        self.sent = []

    async def send(self, s):
        self.sent.append(json.loads(s))

    def params(self, action=None):
        return [m["params"] for m in self.sent
                if action is None or m.get("action") == action]


def _spy_hub():
    h = Hub()
    h._ws = SpySock()
    return h, h._ws


async def case_scan_subscribes_trades_only():
    """A scan symbol takes trades and NOT quotes.

    The single largest saving on the socket, and free: measured at 200
    symbols, quotes were 68% of all records and the full subscription was 2.4x
    the message volume of trades alone. quiet.py is trades-only by
    construction -- a 29-share bid pulled on a thin book moves the midpoint 19
    cents while the stock does not move -- so a quote the scan subscribed to
    is bandwidth spent on a number it must not use.
    """
    h, ws = _spy_hub()
    added, _, refused = await h.scan_set(["AAPL", "MSFT"])
    check(sorted(added) == ["AAPL", "MSFT"], f"scan_set added {added}")
    check(not refused, f"ordinary symbols were refused: {refused}")
    params = ",".join(ws.params("subscribe"))
    check("T.AAPL" in params and "T.MSFT" in params,
          f"the scan did not subscribe to trades: {params}")
    check("Q." not in params,
          f"the scan subscribed to QUOTES: {params} -- 68% of the message "
          f"volume, for records quiet.py must not look at")
    check(len(ws.params("subscribe")) == 1,
          f"{len(ws.params('subscribe'))} subscribe frames for two symbols; "
          f"a 600-symbol set must go out as one message, not six hundred")


async def case_scan_does_not_allocate_pane_buffers():
    """A scan-only symbol gets a ring, not a fifteen-minute deque of dicts.

    THE MEMORY FAILURE, and the box has been OOM-killed twice. A pane's buffer
    keeps every field of every record because a pane draws them; at 600
    symbols and 3,850 trades a second that shape is gigabytes, where three
    float64 arrays are 33 MB. Nothing about a scan symbol landing in
    self.trades would look wrong until the box died.
    """
    h, _ = _spy_hub()
    await h.scan_set(["AAPL"])
    check("AAPL" in h.scan, "the scan symbol got no ring")
    check("AAPL" not in h.trades,
          "a scan-only symbol allocated a pane trade deque -- at 600 symbols "
          "that is the buffer shape that OOMs the box")
    check("AAPL" not in h.quotes,
          "a scan-only symbol allocated a quote deque, for quotes it never "
          "subscribed to")


async def case_scan_and_pane_share_a_symbol():
    """A pane opening on a scan symbol UPGRADES it; closing DOWNGRADES it.

    Both directions are silent when wrong, in opposite ways. Miss the upgrade
    and the pane draws a tape with no quotes and no spread, which looks like a
    thin book rather than a missing subscription. Unsubscribe both channels on
    the pane closing and the scan row stops updating -- and a symbol that has
    stopped printing is indistinguishable from one that has gone quiet, which
    is the single thing the grid exists to tell apart.
    """
    h, ws = _spy_hub()
    await h.scan_set(["AAPL"])
    ws.sent.clear()

    err = await h.acquire("AAPL")
    check(err is None, f"a pane could not open on a scan symbol: {err}")
    up = ",".join(ws.params("subscribe"))
    check("Q.AAPL" in up,
          f"the pane did not add quotes to a scan-held symbol: {up}")
    check("AAPL" in h.trades, "the pane got no trade buffer of its own")
    ws.sent.clear()

    await h.release("AAPL")
    down = ",".join(ws.params("unsubscribe"))
    check("Q.AAPL" in down, f"the pane closing did not drop quotes: {down}")
    check("T.AAPL" not in down,
          f"the pane closing unsubscribed TRADES on a symbol the scan still "
          f"holds ({down}) -- the row goes dead and reads as gone quiet")
    check("AAPL" in h.scan, "the scan lost its symbol when a pane closed")
    check("AAPL" not in h.trades, "the pane's deque outlived the pane")


async def case_scan_removal_spares_a_watched_symbol():
    """Dropping a symbol from the scan must not unsubscribe a pane's."""
    h, ws = _spy_hub()
    await h.scan_set(["AAPL"])
    await h.acquire("AAPL")
    ws.sent.clear()

    _, dropped, _ = await h.scan_set([])
    check(dropped == ["AAPL"], f"scan_set([]) dropped {dropped}")
    check("AAPL" not in h.scan, "the scan kept a symbol it was told to drop")
    check(not ws.params("unsubscribe"),
          f"the scan unsubscribed a symbol a pane is watching: "
          f"{ws.params('unsubscribe')}")
    check("AAPL" in h.refs, "the pane lost its reference")


async def case_scan_cap_refuses_with_a_reason():
    """Past the cap is refused and SAID, never silently truncated."""
    h, _ = _spy_hub()
    n = config.SCAN_MAX_SYMBOLS
    _, _, refused = await h.scan_set([f"S{i}" for i in range(n + 5)])
    check(len(h.scan) == n,
          f"{len(h.scan)} symbols held against a cap of {n}")
    check(len(refused) == 5,
          f"{len(refused)} refusals for 5 symbols past the cap")
    check(all("cap" in r.lower() for r in refused),
          f"a refusal past the cap does not say why: {refused[:2]}")


async def case_scan_reconnect_restores_both_tiers():
    """A reconnect restores the scan too, and with the right channels.

    The server remembers nothing after a drop. Restore only the panes and the
    grid freezes with every row looking quiet; restore the scan with quotes
    and the saving that justifies the tier is gone. Both are invisible from
    the page.
    """
    sent = []

    class FakeWS:
        async def send(self, s): sent.append(json.loads(s))
        def __aiter__(self): return self
        async def __anext__(self): raise StopAsyncIteration
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return False

    h = Hub()
    await h.scan_set(["AAPL", "MSFT"])
    await h.acquire("FDX")
    sent.clear()

    import live.hub as hubmod
    real = hubmod.websockets.connect
    hubmod.websockets.connect = lambda *a, **k: FakeWS()
    try:
        await h._session()
    finally:
        hubmod.websockets.connect = real

    params = ",".join(m["params"] for m in sent
                      if m.get("action") == "subscribe")
    for sym in ("AAPL", "MSFT"):
        check(f"T.{sym}" in params,
              f"the scan symbol {sym} was not resubscribed after a drop -- "
              f"the grid freezes and every row reads as quiet")
        check(f"Q.{sym}" not in params,
              f"the scan symbol {sym} came back WITH quotes: {params}")
    check("T.FDX" in params and "Q.FDX" in params,
          f"the pane symbol lost a channel on reconnect: {params}")


async def case_scan_ingest_routes_to_both_stores():
    """A trade reaches every tier holding its symbol, and only those."""
    h, _ = _spy_hub()
    await h.scan_set(["AAPL", "MSFT"])
    await h.acquire("AAPL")                    # held by both
    await h.acquire("FDX")                     # pane only
    now = time.time() * 1000
    h._ingest(json.dumps([trade("AAPL", now, 100.0, 5),
                          trade("MSFT", now, 200.0, 7),
                          trade("FDX", now, 300.0, 9)]))

    check(h.scan["AAPL"].n == 1 and len(h.trades["AAPL"]) == 1,
          f"a symbol held by both tiers reached {h.scan['AAPL'].n} rings and "
          f"{len(h.trades['AAPL'])} deques; both must see it")
    check(h.scan["MSFT"].n == 1,
          "a scan-only symbol's trade did not reach its ring")
    check("MSFT" not in h.trades,
          "a scan-only trade allocated a pane deque on arrival")
    check("FDX" not in h.scan and len(h.trades["FDX"]) == 1,
          "a pane-only trade leaked into the scan store")
    check(h.scan["AAPL"].p[0] == 100.0 and h.scan["AAPL"].s[0] == 5,
          f"the ring stored the wrong fields: p={h.scan['AAPL'].p[0]} "
          f"s={h.scan['AAPL'].s[0]}")


async def case_scan_state_computes_something():
    """scan_state returns a real ratio, not a column of NaN.

    THE FAILURE THIS EXISTS FOR has happened in this project before: a
    computation that quietly returns nothing is faster than one that works, so
    nothing downstream complains and the grid renders every cell the same
    colour -- which on a quietness grid reads as a market with nothing
    happening in it.

    So the tape is synthetic and the answer is known: a name whose level is
    still must score lower than one that shifted twenty cents.
    """
    h, _ = _spy_hub()
    await h.scan_set(["STILL", "MOVED"])
    now = time.time()
    msgs = []
    for k in range(400):
        age = 300.0 * (1.0 - k / 400.0)
        t_ms = (now - age) * 1000.0
        jitter = 0.01 * ((k % 7) - 3)
        msgs.append(trade("STILL", t_ms, 100.0 + jitter, 100))
        msgs.append(trade("MOVED", t_ms,
                          100.0 + jitter + (0.20 if age < 20.0 else 0.0), 100))
    h._ingest(json.dumps(msgs))

    st = await h.scan_state(now)
    check(set(st) == {"STILL", "MOVED"}, f"scan_state returned {sorted(st)}")
    for sym in ("STILL", "MOVED"):
        ratio, range_c, dollars, n = st[sym]
        check(ratio == ratio,
              f"{sym} produced a NaN quiet ratio -- the rollup is computing "
              f"nothing and every cell would render identically")
        check(range_c == range_c and range_c > 0,
              f"{sym} produced range {range_c}, want a positive span")
        check(dollars == dollars and dollars > 0,
              f"{sym} produced {dollars} dollars/min, want positive")
        check(n >= 10, f"{sym} saw {n} trades in the slow window")
    check(st["MOVED"][0] > st["STILL"][0],
          f"a tape that shifted 20 cents scored {st['MOVED'][0]:.3f}, no "
          f"louder than one that did not ({st['STILL'][0]:.3f}) -- the ratio "
          f"is not responding to the shift it exists to measure")
    check(st["STILL"][0] < 1.0,
          f"an unmoved tape scored {st['STILL'][0]:.3f}; a still name must "
          f"read quiet or the grid lights up on nothing")


async def case_scan_status_names_a_truncated_window():
    """A ring at its ceiling losing live records must be REPORTED.

    A short range bar is indistinguishable from a narrow one. This was the
    actual finding of the capacity run -- every step evicted live records
    against an assumed rate 2.6x too low -- and only a counter said so.
    """
    h, _ = _spy_hub()
    await h.scan_set(["AAPL"])
    buf = h.scan["AAPL"]
    buf.cap_max = buf.cap                      # already at its ceiling
    now = time.time() * 1000
    for i in range(buf.cap + 20):
        buf.push(now + i, 100.0, 1.0)          # all inside the window
    st = h.scan_status()
    check(st["truncated_count"] == 1 and st["truncated"] == ["AAPL"],
          f"a truncated symbol was not named: {st}")
    check(st["held"] == 1 and st["cap"] == config.SCAN_MAX_SYMBOLS,
          f"scan status does not report what it holds: {st}")
    check(st["buffer_mb"] > 0, "scan status reports no memory held")


async def case_rollup_does_not_block_the_loop():
    """The rollup must never hold the event loop for a visible stall.

    WHY THIS IS NOT A PERFORMANCE NICETY. The scan shares the tape's process
    as well as its socket, because the account permits one connection. A pass
    costs ~0.85 ms a symbol, so 430 symbols run as a plain loop is ~370 ms in
    which the upstream reader does not run and Hub.pump() does not flush --
    and pump flushes to every browser every 100 ms. The tape stalls for three
    and a half flush intervals every five seconds, on a page that never asked
    for the scan.

    MEASURED AS EVENT-LOOP LAG, not as pass duration. The chunked pass takes
    slightly LONGER end to end than the blocking one; what changes is that it
    lets go regularly. Timing the pass would show the fix as a small
    regression, which is why the thing being asserted is the gap between
    consecutive opportunities for another coroutine to run.

    The blocking version is run too, and required to stall. Without that half,
    a chunked pass that quietly did nothing would also report tiny gaps, and
    this case would pass while measuring an empty loop.
    """
    import asyncio as _aio
    from live.scan import SymbolBuf, rollup_all, rollup_one

    n_syms, n_trades = 430, 60
    now = time.time()
    bufs = {}
    for i in range(n_syms):
        b = SymbolBuf(360.0, 512, 72000)
        for k in range(n_trades):
            age = 300.0 * (1.0 - k / n_trades)
            b.push((now - age) * 1000.0, 100.0 + 0.01 * (k % 7), 100.0)
        bufs[f"S{i}"] = b

    async def heartbeat(stop, gaps):
        """Yield constantly and record how long each turn had to wait."""
        last = time.perf_counter()
        while not stop.is_set():
            await _aio.sleep(0)
            t = time.perf_counter()
            gaps.append(t - last)
            last = t

    async def worst_gap_during(coro):
        stop = _aio.Event()
        gaps = []
        hb = _aio.create_task(heartbeat(stop, gaps))
        await _aio.sleep(0)
        result = await coro
        stop.set()
        await hb
        return max(gaps) * 1000.0, result

    kw = dict(quiet_window_s=config.SCAN_QUIET_WINDOW_S,
              slow_window_s=config.SCAN_SLOW_WINDOW_S,
              min_trades=10)

    async def blocking():
        # What the hub used to do: one uninterrupted pass.
        return {s: rollup_one(b, now, **kw) for s, b in bufs.items()}

    block_ms, block_out = await worst_gap_during(blocking())
    slice_ms = config.SCAN_ROLLUP_SLICE_MS
    chunk_ms, chunk_out = await worst_gap_during(
        rollup_all(bufs, now, slice_s=slice_ms / 1000.0, **kw))

    print(f"    rollup over {n_syms} symbols: worst loop gap "
          f"{block_ms:.0f} ms blocking -> {chunk_ms:.0f} ms chunked "
          f"(slice {slice_ms:.0f} ms)")

    # THE INSTRUMENT WORKS. If the blocking pass does not stall, this box is
    # too fast for the symbol count and the comparison below proves nothing.
    check(block_ms > 50.0,
          f"the blocking pass held the loop only {block_ms:.0f} ms over "
          f"{n_syms} symbols — too fast to demonstrate a stall, so the "
          f"chunked result below is not evidence of anything")

    # A relative bound rather than only an absolute one: the absolute number
    # moves with the box, the ratio does not.
    check(chunk_ms < block_ms / 5.0,
          f"chunking barely helped: {chunk_ms:.0f} ms against "
          f"{block_ms:.0f} ms blocking")
    check(chunk_ms < 60.0,
          f"the chunked pass still held the loop {chunk_ms:.0f} ms, which is "
          f"more than half of Hub.pump()'s 100 ms flush interval — the tape "
          f"would visibly hitch")

    # CHUNKING MUST NOT CHANGE THE ANSWER. Same frozen `now`, so every window
    # covers the same interval and the two passes have to agree exactly; if
    # they do not, the yield is letting state move underneath the pass.
    check(set(block_out) == set(chunk_out),
          "the chunked pass returned a different symbol set")
    same = all(
        all((a != a and b != b) or a == b
            for a, b in zip(block_out[s], chunk_out[s]))
        for s in block_out)
    check(same,
          "the chunked pass produced different numbers from the blocking one "
          "at the same frozen `now` — the rows are no longer comparable to "
          "each other, which is the whole premise of the grid")


CASES = [
    ("no aggregation",          case_no_aggregation),
    ("odd lots survive",        case_odd_lots_survive),
    ("count cap holds",         case_count_cap),
    ("time cap holds",          case_time_cap),
    ("symbol cap refuses",      case_symbol_cap_refuses),
    ("subscription refcount",   case_refcount),
    ("repeat watch idempotent", case_repeat_watch_is_idempotent),
    ("pins outlive the socket", case_pins_outlive_the_socket),
    ("resubscribe on connect",  case_resubscribe_on_reconnect),
    ("snapshot honours window", case_snapshot_window),
    ("status names the feed",   case_status_states_the_feed),
    ("scan takes trades only",  case_scan_subscribes_trades_only),
    ("scan buffers are rings",  case_scan_does_not_allocate_pane_buffers),
    ("pane up/downgrades scan", case_scan_and_pane_share_a_symbol),
    ("scan drop spares a pane", case_scan_removal_spares_a_watched_symbol),
    ("scan cap refuses",        case_scan_cap_refuses_with_a_reason),
    ("reconnect: both tiers",   case_scan_reconnect_restores_both_tiers),
    ("ingest routes by tier",   case_scan_ingest_routes_to_both_stores),
    ("scan state computes",     case_scan_state_computes_something),
    ("scan names truncation",   case_scan_status_names_a_truncated_window),
    ("rollup yields the loop",  case_rollup_does_not_block_the_loop),
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
