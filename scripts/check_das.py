"""The DAS CMD API adapter, against a fabricated socket.

NO NETWORK, NO DAS, NO ORDERS. Everything here is decidable from a writer
that records lines and a dispatcher fed the replies by hand, which is the
point: this is the second module in the project where being wrong costs
money rather than time, and none of it can be checked by placing a real
order and deciding it looked about right.

WHAT IS BEING PROTECTED, in order of how bad it would be:

  * WHICH STATUSES MEAN "STILL LIVE". A status in WORKING_STATES draws on
    the ladder and can be cancelled and flattened; one outside it reaches
    nothing. So an unrecognised status must resolve to WORKING — the cost of
    the two mistakes is not symmetric, and the expensive one hides a live
    order from the screen and from the controls that get you out.

  * THE FIELD LAYOUT. The manual documents two shapes for %ORDER and its own
    examples use both. Every field after the id shifts between them, so a
    parser that guesses wrong does not produce a slightly wrong order: it
    reads the side as the symbol. %OrderAct's `notes` is optional with the
    token AFTER it, and %TRADE arrives with and without Liq/EcnFee/PL.

  * THE END MARKERS. `#POSEND` starts with `#POS`, `#OrderEnd` with
    `#Order`. Tested in the wrong order, the end of a snapshot starts a new
    one and the record is wiped exactly when it has just been filled.

  * TIMEOUT AND Send_Rej ARE NOT FAILURES. They are unknowns, and the
    difference is the whole point of BrokerIndeterminate: a retry is how a
    timeout becomes a double position. A REJECTED status, by contrast, is
    determinate and must not be reported as an unknown — that would leave
    the pane blocked on a placement that provably never rested.

  * THE ACKNOWLEDGEMENT IS CLAIMED BEFORE THE COMMAND IS SENT. Over a local
    socket the answer can arrive inside the await that sends the question. A
    case here pushes the reply synchronously from the write to prove the
    placement still resolves.

  * FLATTEN CANCELS FIRST. Closing at market while an order rests in the
    same name can reopen the position in the opposite direction the moment
    the flatten fills. The order of the two commands is the safety property
    and it is invisible from the outside.

  * THE TOKEN IS THE MATCH. reconcile must answer from the client order id,
    never from the shape — that is the thing Schwab's API cannot do and the
    reason its version documents an ambiguity.

  * NO MARKET DATA. The tape runs on Polygon; a second quote source drawn on
    the same chart would be two prices under one label. The source is
    scanned for subscription commands.
"""
from __future__ import annotations

import asyncio
import inspect
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from live import config                                     # noqa: E402
from live.brokers import base, das                          # noqa: E402
from live.brokers.base import BrokerError, BrokerIndeterminate  # noqa: E402

# CONFIGURED, so the messages under test are the ones a working box
# produces. Nothing here connects to anything: these values are read for
# what problems() and health() SAY, and the socket is always a fake.
config.DAS_HOST = config.DAS_HOST or "100.96.29.108"
config.DAS_TRADER = config.DAS_TRADER or "GATE"
config.DAS_PASSWORD = config.DAS_PASSWORD or "gate"
config.DAS_ACCOUNT = config.DAS_ACCOUNT or "GATEACCT"

FAILS: list[str] = []


def check(cond, msg):
    if not cond:
        FAILS.append(msg)


# ── a socket that never leaves this process ─────────────────────────────────
class FakeWriter:
    """Records the lines written, and answers them if told how.

    `sync=True` pushes the reply from INSIDE the write, which is the racy
    ordering a local socket really produces: the answer lands before the
    caller has awaited anything.
    """

    def __init__(self, link, reply=None, sync=False):
        self.link = link
        self.reply = reply
        self.sync = sync
        self.lines: list[str] = []
        self.closed = False

    def write(self, blob: bytes) -> None:
        line = blob.decode("utf-8").strip()
        self.lines.append(line)
        if not self.reply:
            return
        answers = self.reply(line) or []
        if self.sync:
            for a in answers:
                self.link._dispatch(a)
        else:
            loop = asyncio.get_event_loop()
            for a in answers:
                loop.call_soon(self.link._dispatch, a)

    async def drain(self) -> None:
        return None

    def close(self) -> None:
        self.closed = True

    def verbs(self) -> list[str]:
        return [ln.split()[0].upper() for ln in self.lines if ln.split()]


def live_link(reply=None, sync=False, *, record=True) -> das.DasLink:
    """A link that believes it is connected, with nothing behind it."""
    link = das.DasLink()
    link.writer = FakeWriter(link, reply=reply, sync=sync)
    link.connected = True
    link.logged_in = True
    link.confirmed_at = time.time()
    if record:
        link.order_snapshot = True
        link.pos_snapshot = True
    return link


def use(link: das.DasLink):
    das.LINK = link
    return link


ORDER_18 = ("%ORDER 1 102 MSFT B L 100 100 0 476.17 SMAT Accepted 09:49:27 "
            "0 730001 BIAN CMDAPI DAY+ N/A")
ORDER_15 = ("%ORDER 56 1 +MSFT^GCI400 B L 1 1 0 116.35 COMP Accepted "
            "20:14:11 56 TRBIAN BIAN")
ORDER_OLD = ("%ORDER 7 MSFT S L 100 100 476.17 ARCA Accepted 09:49:27 "
             "0 730001 BIAN")


# ── the statuses ────────────────────────────────────────────────────────────
def case_statuses():
    """Every status the manual lists is classified, and unknown means LIVE."""
    documented = {"CLOSED", "HOLD", "SENDING", "ACCEPTED", "CANCELED",
                  "REJECTED", "EXECUTED", "PARTIAL", "TRIGGERED"}
    unclassified = sorted(documented - das.WORKING_STATES
                          - das.TERMINAL_STATES)
    check(not unclassified,
          f"these documented statuses are in neither WORKING_STATES nor "
          f"TERMINAL_STATES: {unclassified}. An unclassified status is not a "
          f"silent problem here — working_from_status calls it live — but it "
          f"means nobody decided, and the next one added may be terminal.")

    overlap = sorted(das.WORKING_STATES & das.TERMINAL_STATES)
    check(not overlap, f"these statuses are both live and terminal: {overlap}")

    for st in ("Hold", "Sending", "Accepted", "Partial", "Triggered"):
        check(das.working_from_status(st) is True,
              f"{st!r} was not treated as a working order. It is one, and an "
              f"order wrongly called dead vanishes from the screen AND from "
              f"cancel and flatten.")
    for st in ("Closed", "Canceled", "Rejected", "Executed"):
        check(das.working_from_status(st) is False,
              f"{st!r} was treated as still working; it is terminal")

    # THE ASYMMETRIC ONE. A status nobody has heard of has to draw rather
    # than disappear.
    for st in ("Quantum", "", None, "PendingSomething"):
        check(das.working_from_status(st) is True,
              f"an unrecognised status {st!r} was treated as terminal — that "
              f"hides an order that may be live at DAS from the controls "
              f"that close it")


# ── the field layouts ───────────────────────────────────────────────────────
def case_order_layout():
    o = das.parse_order(ORDER_18, mine={102})
    check(o is not None, "the current 18-field %ORDER did not parse")
    if o:
        check(o["order_id"] == "1", f"order id {o['order_id']!r}")
        check(o["token"] == 102, f"token {o['token']!r}")
        check(o["symbol"] == "MSFT", f"symbol {o['symbol']!r}")
        check(o["side"] == "BUY",
              f"side {o['side']!r} — the pane tests startsWith('BUY'), so a "
              f"raw 'B' draws every buy as a sell")
        check(o["type"] == "LIMIT", f"type {o['type']!r}")
        check(o["qty"] == 100 and o["filled"] == 0,
              f"qty/filled {o['qty']}/{o['filled']}")
        check(o["price"] == 476.17, f"price {o['price']!r}")
        check(o["route"] == "SMAT", f"route {o['route']!r}")
        check(o["status"] == "Accepted" and o["working"] is True,
              f"status {o['status']!r} working={o['working']}")
        check(o["tag"] == "102",
              f"tag {o['tag']!r} — the token IS the per-order handle here")
        check(o["from_api"] is True, "a token this process minted did not "
                                     "mark the order as ours")
        check((o["entered"] or "").startswith("20") and "T09:49:27" in
              (o["entered"] or ""),
              f"entered {o['entered']!r} is not an ISO stamp; "
              f"base.entered_epoch cannot read it and `recent` sorts on it")
        check(base.entered_epoch(o["entered"]) is not None,
              f"base.entered_epoch could not parse {o['entered']!r}, so the "
              f"recent list would sort every DAS order to the same place")

    # FEWER FIELDS, SAME MEANING. The manual's own complex-order example.
    o15 = das.parse_order(ORDER_15)
    check(o15 is not None, "a 15-field %ORDER did not parse")
    if o15:
        check(o15["symbol"] == "+MSFT^GCI400" and o15["side"] == "BUY",
              f"the short line mis-parsed: {o15['symbol']!r} {o15['side']!r}")
        check(o15["tif"] is None and o15["pref"] is None,
              "fields that are not on the line came back as something")
        check(o15["from_api"] is False,
              "an order with an unknown token was claimed as ours")

    # THE OLD LAYOUT, where every field after the id is shifted by one.
    old = das.parse_order(ORDER_OLD)
    check(old is not None, "the older %ORDER layout did not parse")
    if old:
        check(old["symbol"] == "MSFT",
              f"the old layout read the symbol as {old['symbol']!r} — that is "
              f"the side field, and an order for a symbol called 'S' would be "
              f"drawn on whatever pane happened to be open")
        check(old["side"] == "SELL", f"old-layout side {old['side']!r}")
        check(old["price"] == 476.17, f"old-layout price {old['price']!r}")
        check(old["route"] == "ARCA", f"old-layout route {old['route']!r}")
        check(old["working"] is True, "old-layout order was not working")

    # A LINE TOO SHORT TO MEAN ANYTHING returns None rather than raising: one
    # bad line must not take the order list down with it.
    for junk in ("%ORDER", "%ORDER 1", "%ORDER 1 2 3"):
        check(das.parse_order(junk) is None,
              f"{junk!r} parsed into an order")


def case_order_act():
    """`notes` is optional and the token is LAST. The manual's own example."""
    a = das.parse_order_act(
        "%OrderAct 56 Accept Buy +MSFT^GCI400 1 116.35 COMP 20:14:10 1")
    check(a is not None, "the manual's own %OrderAct example did not parse")
    if a:
        check(a["order_id"] == "56" and a["action"] == "Accept",
              f"{a['order_id']!r} {a['action']!r}")
        check(a["token"] == 1,
              f"token {a['token']!r} — the trailing field is the token, not "
              f"a note, which is the whole reason it is read from the end")
        check(a["notes"] is None,
              f"a notes field was invented: {a['notes']!r}")

    b = das.parse_order_act(
        "%OrderAct 77 Send_Rej B FDX 100 318.50 ARCA 09:50:01 "
        "route rejected the order 4242")
    check(b is not None, "an %OrderAct with notes did not parse")
    if b:
        check(b["token"] == 4242, f"token with notes present: {b['token']!r}")
        check(b["notes"] == "route rejected the order",
              f"notes {b['notes']!r} — the reason DAS gives is the useful "
              f"half of a rejection")


def case_trade_and_pos():
    t = das.parse_trade("%TRADE 1 MSFT B 100 28.3 SMAT 18:00:31 3")
    check(t is not None, "the short %TRADE form did not parse")
    if t:
        check(t["order_id"] == "3" and t["price"] == 28.3,
              f"{t['order_id']!r} {t['price']!r}")
        check(t["liq"] is None and t["ecn_fee"] is None,
              "fields absent from the line came back as values")

    full = das.parse_trade(
        "%TRADE 9 FDX SS 50 318.51 ARCA 09:51:02 77 + -0.0012 12.5")
    check(full is not None, "the full %TRADE form did not parse")
    if full:
        check(full["liq"] == "+",
              f"liq {full['liq']!r} — the Cobra statement only aggregates ECN "
              f"fees per symbol-day, so this is the only per-fill record of "
              f"whether the fill added or removed liquidity")
        check(full["ecn_fee"] == -0.0012, f"ecn_fee {full['ecn_fee']!r}")
        check(full["side"] == "SELL_SHORT", f"side {full['side']!r}")

    p = das.parse_pos("%POS AAPL 3 100 117.34 0 0 -12.5 "
                      "2022/04/07-09:56:43 -245")
    check(p is not None, "%POS did not parse")
    if p:
        check(p["qty"] == -100,
              f"a type-3 (short) position came back as {p['qty']} — the "
              f"normalised shape is signed, and an unsigned short is a long "
              f"everywhere it is read")
        check(p["avg"] == 117.34, f"avg {p['avg']!r}")
        check(p["day_pl"] == -12.5,
              f"day_pl {p['day_pl']!r} — it is Realized. Unrealized (-245 "
              f"here) is a snapshot from when the position was sent and must "
              f"not be shown as a current number")

    long_p = das.parse_pos("%POS MSFT 2 100 210.39 0 0 0 "
                           "2022/04/07-09:56:43 110")
    check(long_p and long_p["qty"] == 100,
          f"a margin long was signed wrongly: {long_p}")


def case_shapes():
    """The normalised keys are the pane's contract, not a suggestion."""
    link = use(live_link())
    link._dispatch(ORDER_18)
    link._dispatch("%TRADE 5 MSFT B 40 476.10 SMAT 09:49:30 1 + -0.30 0")
    link._dispatch("%POS MSFT 2 100 476.00 0 0 0 2026/09/20-09:49:00 12")
    st = asyncio.run(das.read_orders(["MSFT"]))

    want_order = {"order_id", "symbol", "side", "qty", "filled", "price",
                  "type", "status", "working", "entered", "tag", "from_api",
                  "fills"}
    check(st["working"], "the pushed order did not reach the working list")
    if st["working"]:
        o = st["working"][0]
        missing = sorted(want_order - set(o))
        check(not missing, f"the normalised order is missing {missing}")
        check(o["fills"] and o["fills"][0]["price"] == 476.10,
              f"the fill did not reach the order: {o['fills']}")
        check(o["fills"][0].get("liq") == "+",
              "the liquidity flag was dropped on the way to the fill")

    for key in ("ok", "as_of", "rt_ms", "working", "recent", "limits",
                "stale_after_s"):
        check(key in st, f"read_orders did not carry {key}")

    pos = asyncio.run(das.read_positions(["MSFT"]))
    for key in ("ok", "as_of", "rt_ms", "positions", "account_type",
                "is_day_trader", "round_trips", "limits", "stale_after_s"):
        check(key in pos, f"read_positions did not carry {key}")
    check(pos["positions"] and set(pos["positions"][0]) ==
          {"symbol", "qty", "avg", "day_pl"},
          f"the normalised position is not the contract shape: "
          f"{pos['positions']}")

    # A CLOSED POSITION IS NOT A POSITION.
    link._dispatch("%POS MSFT 2 0 0 0 0 5 2026/09/20-10:00:00 0")
    pos = asyncio.run(das.read_positions(["MSFT"]))
    check(pos["positions"] == [],
          f"a flat position was still listed: {pos['positions']}")

    both = asyncio.run(das.state(["MSFT"]))
    check(both["as_of"] == st["as_of"] or both["as_of"] > 0,
          "state() lost its as_of")
    check("working" in both and "positions" in both,
          "state() dropped half its payload")


# ── the snapshot markers ────────────────────────────────────────────────────
def case_snapshot_markers():
    """#POSEND starts with #POS. Tested in the wrong order, it wipes.

    A snapshot REPLACES the record rather than merging into it, which is what
    makes an order cancelled elsewhere disappear. The end marker has to be
    recognised as an end, or every dump ends by clearing what it just built.
    """
    link = use(live_link(record=False))
    link._dispatch("#Order id token symb b/s mkt/lmt qty lvqty cxlqty price "
                   "route status time origoid account trader orderSrc TIF Pref")
    link._dispatch(ORDER_18)
    link._dispatch("#OrderEnd")
    check(len(link.orders) == 1,
          f"after a complete order dump the record holds {len(link.orders)} "
          f"orders — #OrderEnd was read as the start of a new snapshot, which "
          f"clears it")
    check(link.order_snapshot is True,
          "the order snapshot was not marked as received, so every read "
          "would refuse forever")

    link._dispatch("#POS symb type qty avgcost")
    link._dispatch("%POS FDX 2 300 318.00 0 0 0 2026/09/20-09:30:00 0")
    link._dispatch("#POSEND")
    check(len(link.positions) == 1 and link.pos_snapshot,
          f"the position dump did not survive its own end marker: "
          f"{link.positions}")

    # A SECOND DUMP REPLACES THE FIRST. An order that filled elsewhere has to
    # vanish; a merge is how a phantom stays on the screen.
    #
    # THE REAL HEADER ROW, not a bare "#Order": since 2026-09-21 a dump is
    # opened only by the field-name row the platform actually sends (see
    # LOGIN_BANNER, captured by telnet), because an `#Order` line carrying an
    # order is indistinguishable from a bare one by its head alone.
    link._dispatch(LOGIN_BANNER[3])
    link._dispatch("#OrderEnd")
    check(link.orders == {},
          f"a fresh order dump merged into the old one: {link.orders}")


def case_refuses_to_invent():
    """Before any record has arrived, a read refuses rather than saying none.

    THE NARROW CASE. A quiet or dropped link is answered from the cache
    (case_serves_cache); this is the one where nothing has ever been seen,
    and the alternative is not a stale answer but a fabricated one: an empty
    working list reads as "nothing is resting" and an empty position list
    reads as "flat".
    """
    real = config.DAS_SNAPSHOT_S
    config.DAS_SNAPSHOT_S = 0.1
    try:
        use(live_link(record=False))
        try:
            asyncio.run(das.read_orders(["FDX"]))
            check(False, "a read before any snapshot returned a list — an "
                         "empty one, which reads as 'nothing working'")
        except BrokerIndeterminate:
            check(False, "refusing before the first snapshot was reported as "
                         "an UNKNOWN; nothing was sent, so it is determinate")
        except BrokerError as exc:
            check("has not arrived" in str(exc),
                  f"the refusal does not say what is missing: {exc}")

        # Positions have their own snapshot: orders arriving does not license
        # an answer about what is held.
        link = use(live_link(record=False))
        link._dispatch("#Order")
        link._dispatch("#OrderEnd")
        try:
            asyncio.run(das.read_positions(["FDX"]))
            check(False, "positions were reported flat off the ORDER "
                         "snapshot, before any position list had arrived")
        except BrokerError:
            pass
    finally:
        config.DAS_SNAPSHOT_S = real


def case_serves_cache():
    """A dropped link still answers, and says how old the answer is.

    NOT A REFUSAL, deliberately (2026-09-20). DAS Trader is open on the same
    screen showing the same orders, so a divergence is seen immediately, and
    a pane that went blank every time the platform restarted would interrupt
    far more often than it would protect. What must NOT happen is the answer
    looking current: `as_of` is the last line DAS actually sent, so the age
    on it climbs from the moment the socket dies.
    """
    link = use(live_link())
    link._dispatch(ORDER_18)
    was = link.confirmed_at = time.time() - 42.0

    # The platform was closed: not connected, and the reconnect is in its
    # cooldown so no attempt is made on this read.
    link.connected = link.logged_in = False
    link._retry_at = time.time() + 30
    link.last_error_why = "DAS Trader Pro is not running"

    st = asyncio.run(das.read_orders(["MSFT"]))
    check(st["ok"] and len(st["working"]) == 1,
          f"a read with the link down did not serve the record it holds: {st}")
    check(abs(st["as_of"] - was) < 0.01,
          f"as_of moved to now ({st['as_of']}) although nothing has been "
          f"confirmed since {was} — that is the fresh-looking dead socket "
          f"this whole arrangement exists to prevent")
    sock = st.get("socket") or {}
    check(sock.get("connected") is False and sock.get("age_s", 0) > 40,
          f"the payload does not carry the link's real state: {sock}")

    probs = das.problems()
    check(any("not connected" in p for p in probs),
          f"problems() does not name the dropped link: {probs}")

    # AND AN ORDER STILL REFUSES, determinately: there is no socket to write
    # to, so nothing was sent anywhere.
    try:
        asyncio.run(das.place(symbol="MSFT", side="BUY", qty=10,
                              price=100.0, route=None))
        check(False, "an order was accepted with the socket down")
    except BrokerIndeterminate:
        check(False, "a placement with no socket was called UNKNOWN; nothing "
                     "was written, so it is determinate")
    except BrokerError:
        pass


def case_liveness():
    """`as_of` is the last PROOF, and the heartbeat is what produces one.

    On a pushed feed a quiet book and a dead socket look identical from the
    outside — nothing arrives in either case. ECHO is the difference.
    """
    link = use(live_link())
    link.confirmed_at = time.time() - 10.0
    st = link.state()
    check(st["age_s"] >= 10,
          f"the link age did not climb while nothing arrived: {st['age_s']}")
    check(st["quiet"] is True,
          f"ten seconds of silence at a {config.DAS_HEARTBEAT_S}s heartbeat "
          f"was not called quiet: {st}")
    check(config.DAS_HEARTBEAT_S * 3 < config.STALE_AFTER_S * 4,
          f"the heartbeat is {config.DAS_HEARTBEAT_S}s, which is slow enough "
          f"that a dead socket could look fresh for longer than the page's "
          f"own staleness threshold of {config.STALE_AFTER_S}s")

    # ANY line is proof. This is what makes a push double as a heartbeat.
    link._dispatch("%POS MSFT 2 100 210.39 0 0 0 2026/09/20-09:56:43 110")
    check(time.time() - link.confirmed_at < 0.1,
          "a pushed line did not count as proof the socket is alive")
    check(link.state()["quiet"] is False, "the link stayed marked quiet")

    # The heartbeat command itself must be the cheap one, not an order
    # command that spends the published quota.
    src = (ROOT / "live" / "brokers" / "das.py").read_text(encoding="utf-8")
    check("ECHO" in src, "no heartbeat command in the adapter at all")


# ── placing ─────────────────────────────────────────────────────────────────
def _accept(order_id="900"):
    def reply(line):
        if line.startswith("NEWORDER"):
            tok = line.split()[1]
            return [f"%ORDER {order_id} {tok} FDX B L 100 100 0 318.50 SMAT "
                    f"Accepted 09:50:00 0 730001 BIAN CMDAPI DAY+ N/A"]
        return []
    return reply


def _replaced(order_id="65377"):
    """DAS answers a REPLACE with the order, at its new price."""
    def reply(line):
        if line.startswith("REPLACE"):
            parts = line.split()
            return [f"%ORDER {parts[1]} 1749678729 LLY S L {parts[2]} "
                    f"{parts[2]} 0 {parts[3]} SMART Accepted 09:58:54 0 "
                    f"730001 BIAN CMDAPI DAY+ N/A"]
        return []
    return reply


def case_place():
    link = use(live_link(reply=_accept()))
    out = asyncio.run(das.place(symbol="FDX", side="BUY", qty=100,
                                price=318.5, route="ARCA"))
    check(out["ok"] and out["order_id"] == "900",
          f"the placement did not resolve to an order id: {out}")
    sent = link.writer.lines[0]
    check(sent.startswith("NEWORDER "), f"the first line sent was {sent!r}")
    parts = sent.split()
    check(parts[2] == "B" and parts[3] == "FDX" and parts[4] == "ARCA",
          f"the NEWORDER line is not the manual's shape: {sent!r}")
    check(parts[5] == "100" and parts[6] == "318.50",
          f"size and price on the wire: {sent!r}")
    check(das.MIN_INT <= int(parts[1]) <= das.MAX_INT,
          f"token {parts[1]} is outside the C int range the API accepts")
    check(out.get("token") == int(parts[1]),
          "the token sent and the token reported differ")

    # THE ROUTE IS THE POINT OF THIS BROKER. It rides on the order, and the
    # default only applies when none was named.
    link = use(live_link(reply=_accept("901")))
    asyncio.run(das.place(symbol="FDX", side="SELL", qty=5, price=1.23,
                          route=None))
    check(link.writer.lines[0].split()[4] == config.DAS_ROUTE,
          f"an order with no route named did not use the configured default "
          f"{config.DAS_ROUTE}: {link.writer.lines[0]!r}")


def case_ack_race():
    """The answer arrives INSIDE the write. It must still be heard.

    This is the ordering a local socket really produces, and the reason the
    waiter is registered before the command is sent. Registered after, this
    placement times out as UNKNOWN while the order rests happily on the book
    — the worst of both, since an unknown is never retried and the pane
    blocks until someone settles it by hand.
    """
    link = use(live_link(reply=_accept("902"), sync=True))
    out = asyncio.run(das.place(symbol="FDX", side="BUY", qty=100,
                                price=318.5, route="ARCA"))
    check(out["ok"] and out["order_id"] == "902",
          f"an answer that arrived before the await was lost: {out}")


def case_indeterminacy():
    """The three answers a placement can get, and they are not the same.

    TimeOut and Send_Rej are UNKNOWN: the order may be live. A Rejected
    status is a REFUSAL: DAS looked at it, gave it an id, and nothing is
    resting. Reporting a refusal as unknown blocks the pane on a placement
    that provably never happened; reporting an unknown as a refusal invites
    the retry that doubles the position.
    """
    real = config.DAS_ACK_S
    config.DAS_ACK_S = 0.2
    try:
        for act in ("TimeOut", "Send_Rej"):
            def reply(line, act=act):
                if line.startswith("NEWORDER"):
                    tok = line.split()[1]
                    return [f"%OrderAct 903 {act} B FDX 100 318.50 ARCA "
                            f"09:50:00 the route did not answer {tok}"]
                return []
            use(live_link(reply=reply))
            try:
                asyncio.run(das.place(symbol="FDX", side="BUY", qty=100,
                                      price=318.5, route="ARCA"))
                check(False, f"{act} was reported as a successful placement")
            except BrokerIndeterminate:
                pass
            except BrokerError as exc:
                check(False, f"{act} was reported as a determinate failure "
                             f"({exc}). It is not one — the order may be "
                             f"live, and a determinate failure may be retried")

        # NO ANSWER AT ALL is the same kind of unknown.
        use(live_link(reply=lambda line: []))
        try:
            asyncio.run(das.place(symbol="FDX", side="BUY", qty=100,
                                  price=318.5, route="ARCA"))
            check(False, "a placement nobody acknowledged was reported ok")
        except BrokerIndeterminate as exc:
            check("must not be retried" in str(exc),
                  f"the unknown does not say what not to do: {exc}")
        except BrokerError:
            check(False, "an unacknowledged placement was called a failure")

        # A REJECTION IS DETERMINATE.
        def rejected(line):
            if line.startswith("NEWORDER"):
                tok = line.split()[1]
                return [f"%ORDER 904 {tok} FDX B L 100 0 100 318.50 SMAT "
                        f"Rejected 09:50:00 0 730001 BIAN CMDAPI DAY+ N/A"]
            return []
        use(live_link(reply=rejected))
        try:
            asyncio.run(das.place(symbol="FDX", side="BUY", qty=100,
                                  price=318.5, route="ARCA"))
            check(False, "a rejected order was reported as placed")
        except BrokerIndeterminate:
            check(False, "an explicit Rejected status was reported as "
                         "UNKNOWN — it is a refusal, and the pane would sit "
                         "blocked on an order that never existed")
        except BrokerError:
            pass

        # CancelRej and ReplaceRej are refusals too: the order is unchanged.
        for cmd, act in (("CANCEL", "CancelRej"), ("REPLACE", "ReplaceRej")):
            def reply(line, cmd=cmd, act=act):
                if line.startswith(cmd):
                    return [f"%OrderAct 905 {act} B FDX 100 318.50 ARCA "
                            f"09:50:00 too late"]
                return []
            link = use(live_link(reply=reply))
            link._dispatch("%ORDER 905 1 FDX B L 100 100 0 318.50 ARCA "
                           "Accepted 09:49:00 0 730001 BIAN CMDAPI DAY+ N/A")
            try:
                if cmd == "CANCEL":
                    asyncio.run(das.cancel(order_id="905"))
                else:
                    asyncio.run(das.replace(order_id="905", symbol="FDX",
                                            side="BUY", qty=100, price=318.4))
                check(False, f"{act} was reported as success")
            except BrokerIndeterminate:
                check(False, f"{act} was reported as UNKNOWN; DAS refused it "
                             f"and the order is unchanged")
            except BrokerError:
                pass
    finally:
        config.DAS_ACK_S = real


def case_cancel_and_replace():
    def reply(line):
        if line.startswith("CANCEL"):
            return ["%OrderAct 910 Canceled B FDX 100 318.50 ARCA 09:51:00"]
        if line.startswith("REPLACE"):
            return ["%OrderAct 910 Replaced B FDX 100 318.40 ARCA 09:51:00"]
        return []
    link = use(live_link(reply=reply))
    link._dispatch("%ORDER 910 1 FDX B L 100 100 0 318.50 ARCA Accepted "
                   "09:49:00 0 730001 BIAN CMDAPI DAY+ N/A")

    out = asyncio.run(das.cancel(order_id="910"))
    check(out["ok"] and link.writer.lines[-1] == "CANCEL 910",
          f"cancel sent {link.writer.lines[-1]!r}")

    out = asyncio.run(das.replace(order_id="910", symbol="FDX", side="BUY",
                                  qty=100, price=318.40))
    check(out["ok"], f"the replace did not resolve: {out}")
    check(link.writer.lines[-1].startswith("REPLACE 910 100 318.40"),
          f"the replace line is not the manual's shape: "
          f"{link.writer.lines[-1]!r}")
    check("ROUTE" not in link.writer.lines[-1].upper(),
          "a route was put on a REPLACE, which has no route field")

    # A REPLACE THAT WOULD MOVE THE VENUE IS REFUSED, not silently ignored.
    # DAS reprices where the order already rests, so accepting the route
    # would put a venue on the screen the order is not at.
    try:
        asyncio.run(das.replace(order_id="910", symbol="FDX", side="BUY",
                                qty=100, price=318.30, route="NSDQ"))
        check(False, "a replace onto a different venue was accepted; DAS's "
                     "REPLACE carries no route and the order would have "
                     "stayed on ARCA")
    except BrokerIndeterminate:
        check(False, "refusing a venue change was reported as UNKNOWN")
    except BrokerError as exc:
        check("ARCA" in str(exc) and "NSDQ" in str(exc),
              f"the refusal does not name both venues: {exc}")


def case_flatten_order():
    """CANCEL FIRST, THEN CLOSE. The order of the two is the safety property.

    Closing at market while an order rests in the same name can leave that
    order to open a fresh position in the opposite direction the moment the
    flatten fills.
    """
    def reply(line):
        if line.startswith("CANCEL ALLSYMB"):
            return ["%OrderAct 920 Canceled B FDX 100 318.50 ARCA 09:52:00",
                    "%OrderAct 921 Canceled B FDX 100 318.40 ARCA 09:52:00"]
        if line.startswith("NEWORDER"):
            tok = line.split()[1]
            return [f"%ORDER 930 {tok} FDX S M 300 0 0 0 SMAT Executed "
                    f"09:52:01 0 730001 BIAN CMDAPI DAY N/A"]
        return []
    link = use(live_link(reply=reply))
    for oid, px in (("920", "318.50"), ("921", "318.40")):
        link._dispatch(f"%ORDER {oid} 1 FDX B L 100 100 0 {px} ARCA Accepted "
                       f"09:49:00 0 730001 BIAN CMDAPI DAY+ N/A")
    link._dispatch("%POS FDX 2 300 318.00 0 0 0 2026/09/20-09:30:00 0")

    out = asyncio.run(das.flatten(symbol="FDX"))
    verbs = link.writer.verbs()
    check(verbs[:2] == ["CANCEL", "NEWORDER"],
          f"flatten sent {verbs} — the cancel has to come FIRST, every time")
    check(link.writer.lines[0] == "CANCEL ALLSYMB FDX",
          f"the cancel was {link.writer.lines[0]!r}; one command cancels "
          f"every open order in the name, including any placed by hand")
    check(sorted(out["cancelled"]) == ["920", "921"],
          f"the cancelled ids were not reported: {out}")
    close = link.writer.lines[1].split()
    check(close[2] == "S" and close[5] == "300" and close[6] == "MKT",
          f"the closing order is not a 300-share market sell: "
          f"{link.writer.lines[1]!r}")
    check(out["ok"] and out["closed"] == {"side": "SELL", "qty": 300},
          f"flatten did not report what it closed: {out}")

    # A SHORT CLOSES BY BUYING.
    link = use(live_link(reply=reply))
    link._dispatch("%POS FDX 3 200 318.00 0 0 0 2026/09/20-09:30:00 0")
    asyncio.run(das.flatten(symbol="FDX"))
    check(link.writer.lines[-1].split()[2] == "B",
          f"a short position was flattened with a sell: "
          f"{link.writer.lines[-1]!r}")

    # FLAT IS FLAT: nothing is sent at all.
    link = use(live_link(reply=reply))
    out = asyncio.run(das.flatten(symbol="FDX"))
    check(out.get("flat") and link.writer.lines == [],
          f"a flatten with no position sent {link.writer.lines}")

    # A CANCEL THAT COULD NOT BE SENT STOPS THE CLOSE. The resting order
    # would otherwise be left to reopen the position.
    def refuse(line):
        return []
    link = use(live_link(reply=refuse))
    link._dispatch("%ORDER 940 1 FDX B L 100 100 0 318.50 ARCA Accepted "
                   "09:49:00 0 730001 BIAN CMDAPI DAY+ N/A")
    link._dispatch("%POS FDX 2 300 318.00 0 0 0 2026/09/20-09:30:00 0")
    for b in das.LIMITER.buckets.values():
        b.calls.clear()
    das.LIMITER.buckets["cancel"].calls.extend(
        [time.time()] * das.LIMITER.buckets["cancel"].limit)
    try:
        asyncio.run(das.flatten(symbol="FDX"))
        check(False, "a flatten closed at market although its cancel had "
                     "been refused")
    except BrokerError as exc:
        check("nothing was closed" in str(exc),
              f"the refusal does not say the position was left alone: {exc}")
    check("NEWORDER" not in link.writer.verbs(),
          "a closing order went out after the cancel was refused")
    for b in das.LIMITER.buckets.values():
        b.calls.clear()


def case_reconcile_by_token():
    """THE TOKEN, NOT THE SHAPE. This is what Schwab's API cannot do.

    Two identical placements seconds apart are permanently indistinguishable
    by shape — that is the ambiguity written out in schwab.py. Here each
    carries a client order id of ours, so the answer is exact and
    `ambiguous` is unreachable.
    """
    real = config.DAS_ACK_S
    config.DAS_ACK_S = 0.2
    try:
        link = use(live_link(reply=lambda line: []))
        # A placement whose answer never arrived: unknown, and never retried.
        try:
            asyncio.run(das.place(symbol="FDX", side="BUY", qty=100,
                                  price=318.5, route="ARCA"))
        except BrokerIndeterminate:
            pass
        tok = max(link.minted)

        # Nothing has landed: absent, and said so plainly.
        out = asyncio.run(das.reconcile(symbol="FDX", side="BUY", qty=100,
                                        price=318.5, sent_at=time.time()))
        check(out["state"] == "absent" and out["matched_on"] == "token",
              f"a placement that never landed was not reported absent: {out}")

        # Now it turns up, carrying our token.
        link._dispatch(f"%ORDER 950 {tok} FDX B L 100 100 0 318.50 ARCA "
                       f"Accepted 09:50:00 0 730001 BIAN CMDAPI DAY+ N/A")
        out = asyncio.run(das.reconcile(symbol="FDX", side="BUY", qty=100,
                                        price=318.5, sent_at=time.time()))
        check(out["state"] == "found" and out["order"]["order_id"] == "950",
              f"the placement was not found by its token: {out}")
        check(out["matched_on"] == "token",
              f"the match fell back to shape with a token available: {out}")

        # A DECOY OF THE SAME SHAPE, placed by hand in DAS seconds apart, is
        # exactly the case base.match_placement has to call ambiguous. Here
        # it is simply not ours.
        link._dispatch("%ORDER 951 77 FDX B L 100 100 0 318.50 ARCA Accepted "
                       "09:50:02 0 730001 BIAN Montage DAY+ N/A")
        out = asyncio.run(das.reconcile(symbol="FDX", side="BUY", qty=100,
                                        price=318.5, sent_at=time.time()))
        check(out["state"] != "ambiguous",
              f"an identical order placed elsewhere made the answer ambiguous "
              f"({out['state']}) — the token is what stops that")
        check(out.get("order", {}).get("order_id") == "950",
              f"the wrong order was matched: {out.get('order')}")

        # WITHOUT THE MINT MAP (this service restarted between the placement
        # and the reconcile) the token is unrecoverable and shape is all
        # there is. It is used, and the answer says which rule produced it.
        link.minted.clear()
        out = asyncio.run(das.reconcile(symbol="FDX", side="BUY", qty=100,
                                        price=318.5, sent_at=0))
        check(out["matched_on"] == "shape",
              f"with no minted token the fallback was not reported: {out}")
    finally:
        config.DAS_ACK_S = real


# ── the wire format ─────────────────────────────────────────────────────────
def case_commands():
    b = das.build_neworder(token=1, side="BUY", symbol="fdx", qty=100,
                           price=318.5200000000001, route="SMAT")
    check(" 318.52 " in b + " ",
          f"a float limit went out unrounded: {b!r}. A sub-penny limit on a "
          f"dollar-plus equity is refused")
    check(b.endswith("TIF=DAY+"),
          f"a resting limit did not default to DAY+: {b!r}")

    m = das.build_neworder(token=2, side="SELL", symbol="FDX", qty=100,
                           price=None, route="SMAT")
    check(" MKT " in m + " " and m.endswith("TIF=DAY"),
          f"a market order is not the manual's shape: {m!r}")

    # SUB-DOLLAR NAMES trade in hundredths of a penny (Reg NMS 612), and
    # rounding one to two decimals is a different price.
    p = das.build_neworder(token=3, side="BUY", symbol="ABC", qty=100,
                           price=0.85321, route="SMAT")
    check("0.8532" in p, f"a sub-dollar limit was rounded to pennies: {p!r}")

    # THE MONTAGE SUFFIX. The dropdown says ARCAL; the API takes ARCA.
    s = das.build_neworder(token=4, side="BUY", symbol="FDX", qty=1,
                           price=1.0, route="ARCAL",
                           known_routes={"ARCA", "SMAT"})
    check(" ARCA " in s, f"the L suffix was not stripped: {s!r}")
    # ...but only when the stripped name is one DAS knows. Turning INET into
    # INE would be a worse failure than passing a suffix through.
    keep = das.build_neworder(token=5, side="BUY", symbol="FDX", qty=1,
                              price=1.0, route="INET",
                              known_routes={"INET", "SMAT"})
    check(" INET " in keep, f"a real route ending in T was mangled: {keep!r}")
    unknown = das.build_neworder(token=6, side="BUY", symbol="FDX", qty=1,
                                 price=1.0, route="WEIRDL",
                                 known_routes={"SMAT"})
    check(" WEIRDL " in unknown,
          f"an unknown route was mangled on a guess: {unknown!r}")

    # LIMIT / MARKET / STOP are montage names, not routes: they require SMAT.
    for name in ("LIMIT", "MARKET", "STOP"):
        line = das.build_neworder(token=7, side="BUY", symbol="FDX", qty=1,
                                  price=1.0, route=name)
        check(" SMAT " in line,
              f"route {name} was sent as a route; the manual says those "
              f"order types require SMAT: {line!r}")

    # A TOKEN OUTSIDE THE C INT RANGE is refused here rather than by DAS.
    for bad in (das.MAX_INT + 1, das.MIN_INT - 1):
        try:
            das.build_neworder(token=bad, side="BUY", symbol="FDX", qty=1,
                               price=1.0, route="SMAT")
            check(False, f"token {bad} was accepted; Token is a C int")
        except BrokerError:
            pass

    # THE FLAGS THAT ARE NOT WIRED UP YET still have to be right when they
    # are: the protocol half is written so the later change is additive.
    flags = das.build_neworder(token=8, side="SELL_SHORT", symbol="FDX",
                               qty=100, price=318.5, route="ARCA",
                               tif="IOC", post_only=True, not_route_out=True,
                               display=0, minume="AON", pref="ARCA",
                               known_routes={"ARCA"})
    check(flags.split()[2] == "SS", f"a short did not go out as SS: {flags!r}")
    for want in ("TIF=IOC", "PostOnly", "NotRouteOut", "Display=0",
                 "Minume=AON", "Pref=ARCA"):
        check(want in flags, f"{want} is not on the line: {flags!r}")

    r = das.build_replace(order_id="5", qty=100, price=318.5)
    check(r == "REPLACE 5 100 318.50", f"the replace line is {r!r}")


def case_limiter():
    """The published limits, with a reserve only getting flat may spend."""
    lim = das.RateLimiter()
    lim.buckets["cancel"] = das.Bucket("CANCEL", 10, 60.0, 3)
    for i in range(7):
        check(lim.take("cancel") is None,
              f"ordinary cancel {i + 1} of 7 was refused")
    why = lim.take("cancel")
    check(why and "held back" in why,
          f"ordinary traffic spent into the reserve: {why!r}")
    for i in range(3):
        check(lim.take("cancel", priority=True) is None,
              f"priority cancel {i + 1} of 3 was refused inside the reserve")
    check(lim.take("cancel", priority=True) is not None,
          "priority spent past the published ceiling")

    # THE WINDOWS ARE DIFFERENT, which is why this is not one bucket: a
    # NEWORDER limit is per SECOND and a CANCEL limit per MINUTE.
    check(lim.buckets["new"].window_s == 1.0,
          "the NEWORDER bucket is not a per-second window")
    check(lim.buckets["replace"].window_s == 60.0,
          "the REPLACE bucket is not a per-minute window")
    check(lim.buckets["new"].limit == 50
          and lim.buckets["replace"].limit == 100,
          f"the published defaults are 50/s and 100/min: "
          f"{lim.buckets['new'].limit}, {lim.buckets['replace'].limit}")

    st = das.LIMITER.state()
    for key in ("per_min", "reserve", "used", "available", "refusals",
                "n_429", "label"):
        check(key in st, f"the limits payload is missing {key}, which the "
                         f"site bar reads for whichever broker is loaded")


# ── what this adapter must never do ─────────────────────────────────────────
def case_no_market_data():
    """Order entry only. The tape is Polygon's and there is one of it.

    A second quote source drawn on the same chart would be two prices under
    one label, and there is no depth of book in this API anyway.
    """
    src = (ROOT / "live" / "brokers" / "das.py").read_text(encoding="utf-8")
    code = "\n".join(
        # Comments and docstrings talk about these deliberately; what matters
        # is whether a command is ever BUILT.
        ln for ln in src.splitlines()
        if not ln.lstrip().startswith("#"))
    for banned in (r'"SB ', r"'SB ", r'"UNSB', r"'UNSB", "ReturnFullLv1",
                   '"SB Lv1"', "TOPLIST"):
        check(banned not in code,
              f"the adapter builds a market-data command ({banned}). The tape "
              f"and the scan run on Polygon; two quote sources on one chart "
              f"is two prices under one label.")

    # And the façade is still the only thing above it: the adapter must not
    # reach for the switches or the guards. check_broker scans every adapter
    # for this; it is asserted here too because this is the file that is
    # about to grow.
    for token in ("_armed_check(", "check_guards(", "trading_allowed(",
                  "_runtime_enabled", "set_trading(", "armed=", "armed:"):
        check(token not in src,
              f"das.py consults the trading policy ({token}). Arming and the "
              f"guards are checked once, in live/broker.py, above every "
              f"adapter.")



# ── the montage is the list ─────────────────────────────────────────────────
def case_route_list():
    """The dropdown offers the MONTAGE; RouteStatus only marks it.

    RouteStatus answers "what can this login see" -- options, short-locate,
    test and PRO routes included -- and none of those is somewhere to send an
    equity order. So the offered list is the configured montage, in its own
    order, and the broker's reply becomes a state on each entry. The three
    states are not interchangeable: DISABLED is "the broker says no for now"
    and UNCONFIRMED is "nobody has said", and collapsing the second into the
    first is how a venue that works looks broken.
    """
    link = use(live_link())
    try:
        # A REAL-SHAPED REPLY: some montage routes enabled, one disabled, one
        # montage route (PSMT) never mentioned, and three routes DAS can see
        # that the montage does not carry.
        for line in ("$RouteStatus SMAT Enabled",
                     "$RouteStatus ARCAE Enabled",
                     "$RouteStatus NSDQ Enabled",
                     "$RouteStatus BATS Disabled",
                     "$RouteStatus OPTX Enabled",       # options route
                     "$RouteStatus LOCATE Enabled",     # short locate
                     "$RouteStatus PROTEST Enabled"):   # a PRO/test route
            link._dispatch(line)

        r = das.health()["routing"]
        check(r["choices"] == list(config.DAS_ROUTES),
              "the offered routes are not the montage, in the montage's order")
        check(r.get("source") == "montage" and r["supported"] is True,
              f"routing does not say where the list came from: {r.get('source')!r}")

        # NOTHING DAS CAN SEE BUT THE MONTAGE DOES NOT CARRY IS OFFERED.
        for stray in ("OPTX", "LOCATE", "PROTEST"):
            check(stray not in r["choices"],
                  f"{stray} came from RouteStatus and is not in the montage, "
                  f"but the page would offer it")
            check(stray not in r["states"],
                  f"{stray} is marked in `states` although it is not offered")

        st = r["states"]
        check(st["SMAT"] == "enabled" and st["ARCAE"] == "enabled"
              and st["NSDQ"] == "enabled",
              f"an enabled montage route is not marked enabled: {st['SMAT']!r}")

        # DISABLED STAYS ON THE LIST. Dropping it would change the dropdown's
        # shape between pre-market and the session.
        check("BATS" in r["choices"] and st["BATS"] == "disabled",
              f"a disabled route was dropped from the list or mismarked: "
              f"{st.get('BATS')!r}")

        # PSMT: in the montage, absent from RouteStatus. Unknown, NOT off.
        check("PSMT" in r["choices"], "PSMT is in the montage but not offered")
        check(st["PSMT"] == "unconfirmed",
              f"PSMT is absent from RouteStatus and was marked {st['PSMT']!r} — "
              f"'disabled' would claim DAS said something it never said")

        # Every montage route is marked, one way or another.
        check(set(st) == set(config.DAS_ROUTES),
              "some montage routes carry no state at all")
        check(r["from_broker"] is True,
              "RouteStatus answered and from_broker says otherwise")

        # BEFORE ANY REPLY, everything is unconfirmed and the page is told
        # that the silence is the reason -- not that 45 routes are off.
        quiet = use(live_link())
        r2 = das.health()["routing"]
        check(set(r2["states"].values()) == {"unconfirmed"} and r2["from_broker"] is False,
              f"before RouteStatus answers, the marks are {set(r2['states'].values())} "
              f"and from_broker={r2['from_broker']}")
        check(r2["choices"] == list(config.DAS_ROUTES),
              "the list changes shape before the broker has answered")
        assert quiet is das.LINK
    finally:
        use(link)



# ── the login banner, and what comes after it ───────────────────────────────
#
# CAPTURED FROM THE REAL PLATFORM by telnet on 2026-09-21, pasted verbatim.
# Everything about this sequence had been guessed from the manual until then,
# and three of the guesses were candidates for why the pane never showed an
# order: the ack is SUCCESSED and not SUCCESS, an account with no orders sends
# only the header and the END marker with no %ORDER lines at all, and
# #SLOrder / #SLOrderEnd sit in the same dump. All three turned out to be
# handled. The fault was a line that arrives AFTER the banner.
LOGIN_BANNER = [
    "#LOGIN SUCCESSED",
    "#POS symb type qty avgcost initqty initprice Realized CreatTime Unrealized",
    "#POSEND",
    "#Order id token symb b/s mkt/lmt qty lvqty cxlqty price route status "
    "time origoid account trader orderSrc TIF Pref",
    "#OrderEnd",
    "#Trade id symb b/s qty price route time orderid Liq EcnFee PL",
    "#TradeEnd",
    "#SLOrder id symb shares openshares exeshares exeprice status route time "
    "lmtPrice token notes",
    "#SLOrderEnd",
]


def case_login_banner():
    """The real banner is recognised, and an EMPTY account still counts.

    The dump for an account holding nothing is headers and END markers with
    no rows between them. If arrival depended on seeing a row, it would never
    come -- and the snapshot guard would then refuse every read for the whole
    session, which is exactly what the pane showed.
    """
    link = use(live_link(record=False))
    check(not link.order_snapshot and not link.pos_snapshot,
          "the link began believing it had a record")
    for ln in LOGIN_BANNER:
        link._dispatch(ln)
    check(link.order_snapshot,
          "the order snapshot was not recognised from the real login banner — "
          "headers and an END marker with no rows IS the answer for an empty "
          "account")
    check(link.pos_snapshot, "the position snapshot was not recognised")
    check(link.orders == {} and link.positions == {},
          f"an empty dump invented a record: {link.orders} {link.positions}")

    # SUCCESSED, not SUCCESS: the ack settles the login wait either way.
    link2 = use(live_link(record=False))
    fut = asyncio.new_event_loop().create_future()
    link2._login_wait = fut
    link2._dispatch("#LOGIN SUCCESSED")
    check(fut.done() and fut.result() is True,
          "#LOGIN SUCCESSED did not settle the login wait")


def case_order_server_line():
    """`#OrderServer` IS NOT THE START OF AN ORDER SNAPSHOT.

    THE 2026-09-21 FAULT, in one line. Dispatch matched prefixes, so
    `#OrderServer Connected` -- a routine status push -- was read as the
    header of a fresh order dump. It cleared `order_snapshot`, so `_ready`
    refused every read for the rest of the session and the pane stayed empty,
    and it opened a staging buffer that no `#OrderEnd` ever closed, so every
    order pushed afterwards was filed where nothing reads. Orders reached DAS
    and worked; none of them ever appeared.
    """
    link = use(live_link(record=False))
    for ln in LOGIN_BANNER:
        link._dispatch(ln)

    for status in ("#OrderServer Connected", "#OrderServer: OK",
                   "#QuoteServer Connected"):
        link._dispatch(status)
        check(link.order_snapshot,
              f"{status!r} cleared the order snapshot — it is a status line, "
              f"not the header of a new dump, and the read guard never "
              f"recovers because a snapshot only arrives at login")
        check(link._staging_orders is None,
              f"{status!r} opened a staging buffer nothing will close; every "
              f"order pushed after it goes somewhere nobody reads")

    # AND THE ORDER ARRIVES. The end-to-end symptom was not a flag, it was an
    # order that worked at DAS and never drew, so the case ends where the pane
    # does: read_orders returning it.
    link._dispatch(ORDER_18)
    out = asyncio.run(das.read_orders())
    ids = [o["order_id"] for o in out["working"] + out["recent"]]
    check("1" in ids, f"the order did not reach the pane's read: {ids}")

    # Unknown heads are COUNTED, not guessed at. #SLOrder is the standing
    # example: it is in the banner, this adapter does not model short locates,
    # and it must not be mistaken for anything else.
    check(link.unhandled.get("#SLORDER") == 1
          and link.unhandled.get("#SLORDEREND") == 1,
          f"short-locate lines were not counted as unhandled: {link.unhandled}")
    check("#ORDER" not in link.unhandled and "#ORDERSERVER" not in link.unhandled,
          f"a line that WAS handled is also counted unhandled: {link.unhandled}")
    check("unhandled" in link.state(),
          "the link's state does not carry what arrived unhandled, which is "
          "how the next surprise gets diagnosed from the box")


def case_head_matching():
    """The head is matched WHOLE. A prefix test cannot be made safe by order.

    Ordering the checks fixed `#OrderEnd` vs `#Order` and nothing else: the
    next `#Order*` line DAS adds breaks it again, which is precisely what
    `#OrderServer` did.
    """
    cases = [
        # line                        starts a dump?   ends one?
        ("#Order id token symb b/s mkt/lmt qty", "start_orders", None),
        ("#OrderEnd",                 None,            "end_orders"),
        ("#OrderServer Connected",    None,            None),
        ("#OrderStatus whatever",     None,            None),   # invented
        ("#POS symb type qty avgcost initqty",   "start_pos",   None),
        ("#POSEND",                   None,            "end_pos"),
        ("#POSITIONLIMIT 3",          None,            None),   # invented
    ]
    for line, starts, ends in cases:
        link = use(live_link(record=False))
        link.order_snapshot = link.pos_snapshot = True
        link._dispatch(line)
        # A HEADER OPENS A DUMP AND LEAVES THE FLAG ALONE. Changed
        # 2026-09-21: clearing it on the header is what let one stray line
        # blind every read until a reconnect. What we were told stays true
        # until an END marker replaces it.
        if starts == "start_orders":
            check(link.order_snapshot and link._staging_orders == {},
                  f"{line!r} did not open an order dump, or cleared the flag")
        elif starts == "start_pos":
            check(link.pos_snapshot and link._staging_pos == {},
                  f"{line!r} did not open a position dump, or cleared the flag")
        else:
            check(link.order_snapshot and link.pos_snapshot,
                  f"{line!r} was mistaken for the start of a dump — it shares "
                  f"a prefix with one, which is not the same as being one")
            check(link._staging_orders is None and link._staging_pos is None,
                  f"{line!r} opened a staging buffer")



# ── only a real header opens a dump ─────────────────────────────────────────
#
# TWICE NOW a line from the #Order family has been read as the start of an
# order snapshot: #OrderServer (2026-09-17) and #OrderSending (2026-09-21).
# Exact-token matching killed the first and not the second, and neither fix
# would survive the next name DAS invents -- including the case no token test
# can catch, an #Order line carrying an ORDER rather than the field names.
#
# So the rule under test is not a list of exceptions. It is: a dump is opened
# ONLY by the documented header row, the snapshot flag is set ONLY by the END
# marker, and NOTHING else in the family touches either.
ORDER_FAMILY_NOISE = [
    "#OrderSending",
    "#OrderServer Connected",
    "#OrderServer: OK",
    # The one a head test cannot separate: same first token as the header,
    # carrying an order.
    "#Order 55 102 LLY B L 1 0 0 1158.60 SMAT Accepted 09:58:54 0 730001 BIAN",
    # Whatever is added next.
    "#OrderInvented 1",
    "#OrderQueue 3 pending",
    "#POSUpdate LLY 1",
    "#TradeSummary 4",
]


def case_only_a_header_opens_a_dump():
    link = use(live_link(record=False))
    for ln in LOGIN_BANNER:
        link._dispatch(ln)
    check(link.order_snapshot and link.pos_snapshot,
          "the banner did not establish the record")
    link._dispatch(ORDER_18)
    held = dict(link.orders)
    check(held, "the pushed order was not recorded")

    for ln in ORDER_FAMILY_NOISE:
        link._dispatch(ln)
        check(link.order_snapshot,
              f"{ln!r} cleared the order snapshot. Every read for the rest of "
              f"the session then fails with 'order list has not arrived yet', "
              f"and only a reconnect clears it")
        check(link.pos_snapshot, f"{ln!r} cleared the position snapshot")
        check(link._staging_orders is None and link._staging_pos is None,
              f"{ln!r} opened a staging buffer; every push after it is filed "
              f"where nothing reads")

    check(link.orders == held,
          f"the record changed while informational lines arrived: {link.orders}")

    # AND THE PANE STILL READS. The symptom was never a flag, it was cancel
    # and replace aiming at orders DAS had already closed because this process
    # had stopped being told anything.
    out = asyncio.run(das.read_orders())
    ids = [o["order_id"] for o in out["working"] + out["recent"]]
    check("1" in ids, f"the record is no longer readable: {ids}")

    # A SECOND REAL HEADER is still a dump -- and until its END arrives the
    # old record stays readable, because what we were told is true until
    # something replaces it.
    link._dispatch(LOGIN_BANNER[3])
    check(link.order_snapshot,
          "a genuine header cleared the snapshot flag; the pane goes blind "
          "for as long as the dump takes, and forever if its END never comes")
    check(link._staging_orders == {}, "a genuine header did not open a dump")
    link._dispatch("%ORDER 2 103 MSFT B L 5 5 0 400.00 SMAT Accepted "
                   "09:59:00 0 730001 BIAN CMDAPI DAY+ N/A")
    check(list(link.orders) == list(held),
          "a dump in progress was published before its END marker")
    link._dispatch("#OrderEnd")
    check(list(link.orders) == ["2"],
          f"the END marker did not replace the record: {list(link.orders)}")


def case_dump_never_ends():
    """A dump whose END never arrives is abandoned, not left swallowing pushes.

    The header is genuine here, so nothing above rejects it; what must not
    happen is the failure that followed both faults -- every subsequent order
    filed into a staging dict nobody reads.
    """
    link = use(live_link(record=False))
    for ln in LOGIN_BANNER:
        link._dispatch(ln)
    link._dispatch(LOGIN_BANNER[3])                 # opens a dump
    check(link._staging_orders == {}, "the dump did not open")

    # Still inside the window: pushes stage, as they should.
    link._dispatch(ORDER_18)
    check(link.orders == {} and list(link._staging_orders) == ["1"],
          "a push during a dump did not stage")

    # Past it: the dump is abandoned and the live record takes over.
    link._dump_open_at["orders"] = time.time() - config.DAS_DUMP_TIMEOUT_S - 1
    link._dispatch("%ORDER 9 104 NVDA B L 2 2 0 100.00 SMAT Accepted "
                   "10:00:00 0 730001 BIAN CMDAPI DAY+ N/A")
    check(link._staging_orders is None,
          "a dump that never ended is still open and still swallowing pushes")
    check("9" in link.orders,
          f"the push after the abandoned dump did not reach the record: "
          f"{list(link.orders)}")
    check(any("unended" in k for k in link.unhandled),
          f"the missing END marker was not recorded: {link.unhandled}")
    out = asyncio.run(das.read_orders())
    check("9" in [o["order_id"] for o in out["working"] + out["recent"]],
          "the pane cannot read an order that arrived after an abandoned dump")


def case_wire_price():
    """What goes on the wire, and what the journal says went on it.

    1158.6000000000001 reads as a sub-penny limit nobody could have placed;
    the wire carried 1158.60 and the log printed the float it came from.
    """
    check(das.fmt_price(1158.6000000000001) == "1158.60",
          f"the price on the wire is {das.fmt_price(1158.6000000000001)!r}")
    cmd = das.build_neworder(token=1, side="BUY", symbol="LLY", qty=1,
                             price=1158.6000000000001)
    check(cmd.split()[6] == "1158.60", f"NEWORDER carries {cmd!r}")

    # AND THE LOG LINE AGREES. Captured from the logger the adapter uses, so
    # a future edit that prints the raw float again fails here.
    import logging
    seen = []

    class Grab(logging.Handler):
        def emit(self, rec):
            seen.append(rec.getMessage())

    h = Grab()
    das.log.addHandler(h)
    was = das.log.level
    das.log.setLevel(logging.INFO)          # or the record is never made
    try:
        link = use(live_link(reply=_accept("902")))
        asyncio.run(das.place(symbol="LLY", side="BUY", qty=1,
                              price=1158.6000000000001, route="SMAT"))
        check(link.writer.lines[0].split()[6] == "1158.60",
              f"the order as sent: {link.writer.lines[0]!r}")
    finally:
        das.log.removeHandler(h)
        das.log.setLevel(was)
    placed = [m for m in seen if "DAS placed" in m]
    check(placed and "1158.60" in placed[0] and "1158.6000000000001" not in placed[0],
          f"the journal does not show the price as sent: {placed}")



# ── a replace carries the order's TOTAL ─────────────────────────────────────
def case_replace_sends_total():
    """DAS MODIFIES the order, so the share field is its total.

    Measured on 2026-09-22: order 65377 went 5 -> 4 -> 3 -> 2 across four
    nudges with NO fills. The pane sent the remaining, DAS read it as the new
    total, and the reply's smaller remaining became the next request's
    total -- an order shrinking itself one nudge at a time.
    """
    link = use(live_link(reply=_replaced()))
    # DAS's own record: five ordered, two gone, three resting.
    link._dispatch("%ORDER 65377 1749678729 LLY S L 5 3 0 1166.38 SMART Partial")
    asyncio.run(das.replace(order_id="65377", symbol="LLY", side="SELL",
                            qty=5, price=1166.40, filled=2))
    sent = link.writer.lines[-1]
    check(sent.split()[:3] == ["REPLACE", "65377", "5"],
          f"REPLACE did not carry the order's total: {sent!r}")

    # FOUR NUDGES, NO FILLS: the size must not move. This is the exact walk
    # that was observed, and it only stays flat if the total is the total.
    for px in (1166.42, 1166.44, 1166.46, 1166.48):
        asyncio.run(das.replace(order_id="65377", symbol="LLY", side="SELL",
                                qty=5, price=px, filled=2))
    sizes = [ln.split()[2] for ln in link.writer.lines if ln.startswith("REPLACE")]
    check(sizes == ["5"] * 5,
          f"the order shrank across nudges: {sizes} — each reply's remaining "
          f"became the next request's total, which is the 5/4/3/2 walk")

    # THE WALK ITSELF, driven the way it actually happened: a caller that
    # passes the REMAINING, as the pane did. The adapter takes the total from
    # DAS's own record, so the wire is right even when the caller's number is
    # the one that caused the bug -- and the order stops shrinking.
    link = use(live_link(reply=_replaced()))
    link._dispatch("%ORDER 65377 1749678729 LLY S L 5 3 0 1166.38 SMART Partial")
    for px in (1166.40, 1166.42, 1166.44, 1166.46):
        o = link.orders["65377"]
        remaining = int((o["qty"] or 0) - (o["filled"] or 0))
        asyncio.run(das.replace(order_id="65377", symbol="LLY", side="SELL",
                                qty=remaining, price=px, filled=o["filled"]))
    walk = [ln.split()[2] for ln in link.writer.lines if ln.startswith("REPLACE")]
    check(walk == ["5"] * 4,
          f"a caller passing the remaining still shrinks the order: {walk} — "
          f"this is the 5/4/3/2 walk, and DAS's own record is what prevents it")

    # AN ORDER THIS PROCESS HAS NEVER BEEN TOLD ABOUT: the caller's own two
    # numbers add back up to the total.
    link = use(live_link(reply=_replaced("777")))
    asyncio.run(das.replace(order_id="777", symbol="LLY", side="SELL",
                            qty=3, price=1166.40, filled=2))
    check(link.writer.lines[-1].split()[2] == "5",
          f"without a record, total is qty+filled: {link.writer.lines[-1]!r}")


# ── nothing left is not working ─────────────────────────────────────────────
GHOST = "%ORDER 65377 1749678729 LLY S L 2 0 1 1166.38 SMART Partial"


def case_zero_remaining_is_not_working():
    """`Partial` with zero left is a finished order wearing a live status.

    DAS sent exactly this after a cancel: two ordered, ZERO left, one
    cancelled, status still `Partial` because a share had filled. The status
    says what HAPPENED to the order, not whether it still rests -- so it
    outlives the order, and taking it as live left a dead order on the ladder
    that a nudge could not move and a cancel was refused for
    ("order not open").
    """
    link = use(live_link())
    link._dispatch(GHOST)
    o = link.orders["65377"]
    check(o["working"] is False,
          f"an order with nothing left is still marked working: qty={o['qty']} "
          f"filled={o['filled']} cancelled={o['cancelled_qty']} "
          f"status={o['status']!r}")

    out = asyncio.run(das.read_orders())
    check(not out["working"] and [x["order_id"] for x in out["recent"]] == ["65377"],
          f"the ghost is still on the ladder: working={out['working']}")

    # AND FLATTEN LEAVES IT ALONE. Cancelling an order DAS has already closed
    # is the "order not open" error, from the one path that must not waste
    # its attempts.
    link = use(live_link(reply=lambda ln: []))
    link._dispatch(GHOST)
    link._dispatch("%POS LLY 2 0 1166.00 0 0 0 2026/09/22-09:30:00 0")
    asyncio.run(das.flatten(symbol="LLY"))
    cancels = [ln for ln in link.writer.lines if ln.startswith("CANCEL")]
    check(cancels == [],
          f"flatten tried to cancel an order with nothing left: {cancels}")

    # A PARTIAL WITH A BALANCE IS STILL LIVE -- the half of the rule that
    # must not be lost while fixing the other half.
    link = use(live_link())
    link._dispatch("%ORDER 65380 1 LLY S L 5 3 0 1166.38 SMART Partial")
    check(link.orders["65380"]["working"] is True,
          "a partial with three shares still resting was called finished")

    # NOT YET AT THE EXCHANGE: zero resting says nothing, and hiding it would
    # hide an order about to be live -- and its cancel with it.
    for status in ("Sending", "Hold"):
        link = use(live_link())
        link._dispatch(f"%ORDER 65381 1 LLY S L 5 0 0 1166.38 SMART {status}")
        check(link.orders["65381"]["working"] is True,
              f"a {status} order with nothing resting yet was hidden; it is "
              f"about to be live and must stay cancellable")

    # An older layout carries no lvqty at all: fall back to the status.
    o = das.parse_order(ORDER_OLD)
    check(o["working"] is True,
          "a layout without lvqty was called finished on a missing field")


def case_interface():
    """It is a Broker, and it answers the façade's questions."""
    try:
        a = das.DasBroker()
    except TypeError as exc:
        check(False, f"DasBroker does not implement the interface: {exc}")
        return
    check(isinstance(a, base.Broker), "DasBroker is not a base.Broker")
    check(a.name == "das", f"the adapter calls itself {a.name!r}")

    for meth in ("place", "replace"):
        params = inspect.signature(getattr(a, meth)).parameters
        check("route" in params, f"{meth}() does not take a route")

    # RECONCILE IS OVERRIDDEN, and that is the whole point of this broker:
    # inheriting the default would put Schwab's shape heuristic, and its
    # documented ambiguity, on an API that has a real client order id.
    check(das.DasBroker.reconcile is not base.Broker.reconcile,
          "DasBroker inherits base.Broker.reconcile — the token makes an "
          "exact match possible, and the inherited one matches by shape and "
          "can answer `ambiguous`")

    h = a.health()
    check(h.get("broker") == "das", f"health() says {h.get('broker')!r}")
    r = h.get("routing") or {}
    check(r.get("supported") is True, "DAS reports that it does not route")
    check(r.get("default") == config.DAS_ROUTE,
          f"the default route in health() is {r.get('default')!r}")
    check("socket" in h, "health() does not carry the link's state, which is "
                         "the only outward sign a pushed feed gives")


CASES = [
    ("order statuses", case_statuses),
    ("the real login banner", case_login_banner),
    ("#OrderServer is not a snapshot", case_order_server_line),
    ("only a header opens a dump", case_only_a_header_opens_a_dump),
    ("a dump that never ends", case_dump_never_ends),
    ("the price on the wire", case_wire_price),
    ("a replace carries the total", case_replace_sends_total),
    ("nothing left is not working", case_zero_remaining_is_not_working),
    ("the head is matched whole", case_head_matching),
    ("the route list is the montage", case_route_list),
    ("the %ORDER layouts", case_order_layout),
    ("%OrderAct notes and token", case_order_act),
    ("%TRADE and %POS", case_trade_and_pos),
    ("the normalised shapes", case_shapes),
    ("the snapshot markers", case_snapshot_markers),
    ("it refuses to invent a record", case_refuses_to_invent),
    ("a dropped link still answers", case_serves_cache),
    ("liveness is proven, not assumed", case_liveness),
    ("placing", case_place),
    ("the answer can beat the await", case_ack_race),
    ("determinate vs unknown", case_indeterminacy),
    ("cancel and replace", case_cancel_and_replace),
    ("flatten cancels first", case_flatten_order),
    ("reconcile matches the token", case_reconcile_by_token),
    ("the wire format", case_commands),
    ("the rate limiter", case_limiter),
    ("no market data, no policy", case_no_market_data),
    ("the broker interface", case_interface),
]


def main() -> int:
    real_link = das.LINK
    try:
        for name, fn in CASES:
            before = len(FAILS)
            try:
                fn()
            except Exception as exc:                        # noqa: BLE001
                FAILS.append(f"{name} raised {type(exc).__name__}: {exc}")
            if len(FAILS) > before:
                for m in FAILS[before:]:
                    print(f"  FAIL {name}: {m}")
    finally:
        das.LINK = real_link

    print(f"\nDAS cases: {len(CASES)}, failures: {len(FAILS)}")
    if not FAILS:
        print("  an unknown status still draws; the token is the match; "
              "flatten cancels before it closes; a timeout is never a "
              "refusal; a dead socket cannot look fresh")
    return 1 if FAILS else 0


sys.exit(main())
