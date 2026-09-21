"""The order path, against a fabricated Schwab.

NO NETWORK AND NO ORDERS, which is the point: everything here is decidable
from a stubbed transport, and none of it can be checked by placing a real
order and deciding it looked about right. This is the one module in the
project where being wrong costs money rather than time.

WHAT IS BEING PROTECTED, in order of how bad it would be:

  * THE FOUR SWITCHES. LIVE_TRADING_ENABLED, the runtime flag, the pane's
    arm flag, and the guards. All four are checked SERVER-side; the browser
    is where a guard is easiest to bypass by accident — a stale page, a
    replayed request, a hand-typed fetch during debugging.

  * THE RUNTIME FLAG CANNOT ESCALATE past the environment gate, defaults to
    off on every start, needs the shared secret to turn ON, and needs
    nothing to turn OFF.

  * FLATTEN CANCELS FIRST. Closing at market while a working order rests on
    the same name can reopen the position in the opposite direction the
    moment the flatten fills. The order of those two calls is the whole
    safety property, and it is invisible from the outside.

  * THE GUARD ON ENDING POSITION, not current. A 400-share sell against a
    300-share long is a 100-share short.

  * A MISTYPED PRICE, which no share limit catches: 31.85 for 318.50 is a
    marketable order at a tenth of the price.

  * THE RESERVE. A cancel must never be the call refused for quota, which is
    what an undifferentiated bucket would eventually do.

  * CANCEL IS NOT GATED ON ARMING. A safety switch that traps an order is
    not a safety switch.

  * THE TOKEN REFRESH, both ways. It runs only when a refresh is REFUSED,
    which is why a requests-style `r.reason` on an httpx response survived
    from the first commit to 2026-09-17 and then took the pane down with an
    AttributeError where Schwab's own answer should have been. The case
    drives a fake httpx whose response has httpx's attributes AND NO OTHERS.

  * THE SWITCHES SIT ABOVE THE ADAPTER. Since the second broker made an
    interface necessary, arming and the guards are checked once in
    live/broker.py and an adapter is reached only after they pass. The cases
    below drive a FAKE adapter and assert it was NOT CALLED AT ALL when a
    switch or a guard refuses -- a broker that is never reached cannot send
    anything, and the next adapter inherits that without being trusted to.
"""
from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from live import broker, brokers, config                    # noqa: E402
from live.brokers import base, schwab                       # noqa: E402

# READ AT IMPORT, before any case assigns it. Asserting the default after a
# test has set it is asserting nothing — a module shipping the flag as True
# passed that version of this check.
RUNTIME_AT_IMPORT = broker._runtime_enabled

# THE REAL TRANSPORT, captured before any case stubs it out. Earlier cases
# replace schwab._acall with a recorder, so a case that means to exercise the
# real one has to hold its own reference — without this the classification
# test drove the recorder and every assertion passed vacuously.
REAL_ACALL = schwab._acall

FAILS: list[str] = []


def check(cond, msg):
    if not cond:
        FAILS.append(msg)


class Recorder:
    """Stands in for _acall. Records every call, in order."""

    def __init__(self, replies=None):
        self.calls = []
        self.replies = replies or {}

    async def __call__(self, method, path, *, params=None, body=None,
                       priority=False):
        self.calls.append({"method": method, "path": path, "body": body,
                           "priority": priority})
        # ENDSWITH, not `in`. "/accounts/H" is a prefix of
        # "/accounts/H/orders", so a substring match handed the account
        # payload back for the orders call and every working order vanished —
        # in the harness, which would have made the real check vacuous.
        if method == "GET" and path.endswith("/orders"):
            return self.replies.get("orders", []), 200, 12.0
        if method == "GET":
            return self.replies.get("account"), 200, 12.0
        if method == "POST":
            return {"order_id": "NEW1"}, 201, 20.0
        return None, 200, 8.0

    def verbs(self):
        return [(c["method"], c["path"].split("/")[-1] or "orders")
                for c in self.calls]


async def _fake_token():
    return "TOK"


def _acct(positions=(), orders=()):
    return {
        "account": {"securitiesAccount": {
            "type": "MARGIN", "isDayTrader": True, "roundTrips": 3,
            "positions": list(positions)}},
        "orders": list(orders),
    }


def pos(sym, long_q=0.0, short_q=0.0, avg=100.0):
    return {"instrument": {"symbol": sym}, "longQuantity": long_q,
            "shortQuantity": short_q, "averagePrice": avg}


def order(oid, sym, side, qty, price, status="WORKING", tag=None):
    o = {"orderId": oid, "quantity": qty, "filledQuantity": 0,
         "price": price, "orderType": "LIMIT", "status": status,
         "orderLegCollection": [{"instruction": side, "quantity": qty,
                                 "instrument": {"symbol": sym}}]}
    if tag is not None:
        o["tag"] = tag
    return o


# ── which application placed it ─────────────────────────────────────────────
def case_order_source():
    """`from_api` must follow Schwab's stamp, because the pane acts on it.

    Schwab stamps `tag` itself and a client cannot set it — a body carrying
    one is rejected outright (400, tested 2026-09-03). Orders placed through
    this API come back `TA_<account-derived>`; thinkorswim's come back
    `API_TOS:AT_LADDER_AS`. The tape's primaryOrder() fallback uses this to
    refuse to reprice an order it did not place, so a wrong answer here moves
    a stranger's order.
    """
    n = schwab._norm_order

    mine = n(order("A", "FDX", "BUY", 100, 318.5, tag="TA_examplestamp1774"))
    check(mine["from_api"] is True,
          f"an order carrying Schwab's TA_ stamp was not recognised as this "
          f"app's: {mine.get('tag')!r} -> {mine.get('from_api')!r}")

    tos = n(order("B", "FDX", "BUY", 100, 318.5, tag="API_TOS:AT_LADDER_AS"))
    check(tos["from_api"] is False,
          "a thinkorswim order was claimed as this app's — the pane would "
          "reprice an order placed by hand in another application")

    # NO TAG AT ALL is the case that decides which way the code fails. It
    # must read as "not ours": the fallback then declines rather than acting
    # on an order whose source is unknown.
    bare = n(order("C", "FDX", "BUY", 100, 318.5))
    check(bare["from_api"] is False,
          "an untagged order defaulted to being this app's. Unknown source "
          "must mean 'not ours' — the safe direction is to decline.")
    check(bare["tag"] is None,
          f"a missing tag did not come through as None: {bare['tag']!r}")

    # The prefix and nothing looser. `TA_` is a prefix test, not a substring
    # one, or a tag merely CONTAINING it would pass.
    sneaky = n(order("D", "FDX", "BUY", 100, 318.5, tag="XX_TA_not_ours"))
    check(sneaky["from_api"] is False,
          "the source test matched TA_ anywhere in the tag instead of at the "
          "start, so a foreign stamp could be read as ours")


# ── where my orders actually traded ─────────────────────────────────────────
def case_fills():
    """Executions come off the ORDER, never off the tape.

    The tape carries every print in the name and says nothing about whose it
    was. Matching our orders onto it by price and time would be the guess
    match_placement refuses to make — and drawn on a chart it would look like
    fact. Schwab keeps the executions on the order, so this reads them.

    The shape is documented rather than typed, which is why it is asserted
    here against a realistic payload: if a field name is wrong the fills
    simply stop appearing, with nothing raised and nothing on screen to say
    so. That is the failure this case exists to make loud.
    """
    n = schwab._norm_order

    # A single order filled in three pieces at two prices, which is the case
    # worth seeing against the tape.
    o = order("A", "FDX", "BUY", 300, 318.50)
    o["orderActivityCollection"] = [{
        "activityType": "EXECUTION",
        "executionType": "FILL",
        "quantity": 300,
        "executionLegs": [
            {"legId": 1, "quantity": 100, "price": 318.49,
             "time": "2026-09-04T14:31:02+0000"},
            {"legId": 1, "quantity": 100, "price": 318.49,
             "time": "2026-09-04T14:31:03+0000"},
            {"legId": 1, "quantity": 100, "price": 318.50,
             "time": "2026-09-04T14:31:09+0000"},
        ],
    }]
    f = n(o)["fills"]
    check(len(f) == 3,
          f"three execution legs produced {len(f)} fills. A partial fill at "
          f"several prices is exactly what is worth seeing on the tape, and "
          f"collapsing it to one loses where it traded.")
    check([x["price"] for x in f] == [318.49, 318.49, 318.50],
          f"the fill prices did not survive: {[x['price'] for x in f]}")
    check([x["qty"] for x in f] == [100, 100, 100],
          f"the fill quantities did not survive: {[x['qty'] for x in f]}")
    check(all(x["t"] for x in f),
          "a fill came through with no time, so it cannot be placed on the "
          "tape at all")

    # Not every activity is an execution.
    o2 = order("B", "FDX", "BUY", 100, 318.50)
    o2["orderActivityCollection"] = [
        {"activityType": "ORDER_ACTION", "executionLegs": [
            {"quantity": 100, "price": 999.0, "time": "2026-09-04T14:00:00+0000"}]},
    ]
    check(n(o2)["fills"] == [],
          f"a non-execution activity was read as a fill: {n(o2)['fills']}. "
          f"That would draw a mark on the chart where nothing traded.")

    # THE PATHS THAT ONLY RUN ON A BAD PAYLOAD. Each of these must produce no
    # fills rather than an exception: this is parsed inside the order read,
    # and raising here would take the whole order list down with it.
    for label, bad in (
            ("no activity collection at all", None),
            ("a null collection", {"orderActivityCollection": None}),
            ("a string where an activity should be",
             {"orderActivityCollection": ["nonsense"]}),
            ("an activity with no legs",
             {"orderActivityCollection": [{"activityType": "EXECUTION"}]}),
            ("a leg with no price",
             {"orderActivityCollection": [{"activityType": "EXECUTION",
              "executionLegs": [{"quantity": 5, "time": "x"}]}]}),
            ("a leg with no quantity",
             {"orderActivityCollection": [{"activityType": "EXECUTION",
              "executionLegs": [{"price": 1.0, "time": "x"}]}]}),
            ("a leg that is not an object",
             {"orderActivityCollection": [{"activityType": "EXECUTION",
              "executionLegs": [7]}]})):
        o3 = order("C", "FDX", "BUY", 100, 318.50)
        if bad:
            o3.update(bad)
        try:
            got = n(o3)["fills"]
        except Exception as exc:                            # noqa: BLE001
            check(False, f"{label} RAISED {type(exc).__name__}: {exc}. This "
                         f"is parsed inside the order read; an exception "
                         f"here loses every order, not one fill.")
            continue
        check(got == [], f"{label} produced fills: {got}")


# ── the recent list is the RECENT end of the day ────────────────────────────
def case_recent_is_newest():
    """`recent` must be the newest orders, not the twelve the API listed last.

    THIS SHIPPED WRONG. The slice was `recent[-12:]`, which is the newest
    twelve only if the payload is oldest-first. Verified against the live
    account on 2026-09-04: Schwab returns NEWEST-FIRST — 575 orders running
    from 15:20 down to the previous day — so the page was being served the
    twelve OLDEST orders in a day-long window. Fills from minutes earlier
    never arrived, and on a chart that is indistinguishable from a marker
    too small to see.

    Asserted from BOTH directions, because the point is not to depend on the
    order at all.
    """
    import time as _t

    def at(hhmm, oid):
        o = order(oid, "FDX", "BUY", 100, 318.50, status="FILLED")
        o["enteredTime"] = f"2026-09-04T{hhmm}:00+0000"
        return o

    # Twenty orders, 10:00 through 19:00-ish, built oldest-first.
    oldest_first = [at(f"{10 + i // 2:02d}:{(i % 2) * 30:02d}", f"O{i:02d}")
                    for i in range(20)]
    newest_first = list(reversed(oldest_first))

    for label, rows in (("newest-first (what Schwab actually sends)",
                         newest_first),
                        ("oldest-first", oldest_first)):
        rec = Recorder(_acct(orders=rows))
        schwab._acall = rec
        schwab._account_hash = "H"
        got = asyncio.run(broker.read_orders(["FDX"]))["recent"]
        check(len(got) == 12,
              f"{label}: {len(got)} recent orders, expected 12")
        ids = [o["order_id"] for o in got]
        # The newest twelve of twenty are O08..O19.
        want = {f"O{i:02d}" for i in range(8, 20)}
        check(set(ids) == want,
              f"{label}: got {sorted(ids)}, expected the NEWEST twelve "
              f"{sorted(want)}. Serving the oldest twelve of a day-long "
              f"window is why a fill from a minute ago never reached the "
              f"page.")
        check(ids[0] == "O19",
              f"{label}: the list does not start at the newest: {ids[:3]}")


# ── every order status is accounted for ─────────────────────────────────────
def case_order_states():
    """A status the pane has never heard of must not silently disappear.

    WHY THIS IS ASYMMETRIC. A status in WORKING_STATES puts the order in
    `working`: it draws, primaryOrder() can pick it up, it can be repriced
    and cancelled. A status outside it goes to `recent`, where it draws
    nothing and no control reaches it. Calling a dead order live leaves an
    inert marker; calling a LIVE order dead hides a working order from the
    screen meant to show it, and from cancel and flatten, which are how you
    get out.

    So the real failure is a status nobody classified. Schwab adds one, it
    matches neither set, and an order that is live at the broker is invisible
    here with nothing raised.
    """
    # Every status Schwab documents, plus the six this account has actually
    # produced (surveyed over 603 orders, 2026-09-04).
    documented = {
        "AWAITING_PARENT_ORDER", "AWAITING_CONDITION",
        "AWAITING_STOP_CONDITION", "AWAITING_MANUAL_REVIEW", "ACCEPTED",
        "AWAITING_UR_OUT", "PENDING_ACTIVATION", "QUEUED", "WORKING",
        "REJECTED", "PENDING_CANCEL", "CANCELED", "PENDING_REPLACE",
        "REPLACED", "FILLED", "EXPIRED", "NEW", "AWAITING_RELEASE_TIME",
        "PENDING_ACKNOWLEDGEMENT", "PENDING_RECALL", "UNKNOWN",
    }
    unclassified = sorted(
        documented - schwab.WORKING_STATES - schwab.TERMINAL_STATES)
    check(not unclassified,
          f"these documented statuses are in neither WORKING_STATES nor "
          f"TERMINAL_STATES: {unclassified}. An unclassified status falls "
          f"into `recent`, so an order that is live at Schwab draws nothing, "
          f"cannot be cancelled from the pane and does not block it — "
          f"silently.")

    overlap = sorted(schwab.WORKING_STATES & schwab.TERMINAL_STATES)
    check(not overlap,
          f"these statuses are both live and terminal: {overlap}")

    # THE FOUR ADDED AFTER THE SURVEY. Each is an order Schwab still has, and
    # each was landing in `recent` where nothing could reach it.
    for st in ("PENDING_ACKNOWLEDGEMENT", "AWAITING_RELEASE_TIME",
               "PENDING_RECALL", "UNKNOWN"):
        check(st in schwab.WORKING_STATES,
              f"{st} is not treated as live. It is an order Schwab still "
              f"has: it would draw no marker, be unreachable by cancel or "
              f"reprice, and leave the pane free to send another on top.")

    # And the terminal five stay terminal, or every filled order in the
    # window comes back as a working one.
    for st in ("FILLED", "CANCELED", "REJECTED", "EXPIRED", "REPLACED"):
        check(st not in schwab.WORKING_STATES,
              f"{st} is treated as live; it is over.")

    # The normaliser has to agree with the sets, not carry its own opinion.
    live = _norm_status("PENDING_ACKNOWLEDGEMENT")
    check(live["working"] is True,
          f"the normaliser disagrees with WORKING_STATES: {live}")
    dead = _norm_status("FILLED")
    check(dead["working"] is False,
          f"a filled order came back as working: {dead}")


def _norm_status(status):
    o = order("S1", "FDX", "BUY", 100, 318.50, status=status)
    return schwab._norm_order(o)


# ── the guards ──────────────────────────────────────────────────────────────
def case_guards():
    g = broker.check_guards
    ok = dict(symbol="FDX", side="BUY", qty=100, price=318.50,
              reference=318.40, position_qty=0)

    check(g(**ok) is None, f"an ordinary order was refused: {g(**ok)}")

    # Size.
    why = g(**{**ok, "qty": config.MAX_ORDER_SHARES + 1})
    check(why and "per-order" in why,
          f"an order over the share limit was allowed: {why!r}")
    for bad_qty in (0, -100, 10.5, None):
        check(g(**{**ok, "qty": bad_qty}) is not None,
              f"quantity {bad_qty!r} was accepted as a share count")

    # ENDING POSITION, not current. A sell through flat is a short.
    why = g(symbol="CHEAP", side="SELL", qty=400, price=12.00,
            reference=12.00, position_qty=300)
    check(why is None,
          f"a 400 sell against a 300 long was refused, and -100 is inside "
          f"the limit: {why!r}")
    # Adding to a SHORT past the limit. Kept under the per-order cap so this
    # tests the ending-position rule and not the size rule.
    why = g(symbol="CHEAP", side="SELL", qty=200, price=12.00,
            reference=12.00, position_qty=-(config.MAX_POSITION_SHARES - 50))
    check(why and "position limit" in why,
          f"a sell that drives an existing short past the limit was "
          f"allowed: {why!r}")
    why = g(symbol="CHEAP", side="BUY", qty=400, price=12.00, reference=12.00,
            position_qty=config.MAX_POSITION_SHARES)
    check(why and "position limit" in why,
          f"adding to a position already at the cap was allowed: {why!r}")

    # Notional — shares alone do not bound a $900 name.
    why = g(symbol="BRK", side="BUY", qty=100, price=900.0, reference=900.0,
            position_qty=0)
    check(why and "notional" in why,
          f"$90,000 of a $900 stock passed a 500-share limit: {why!r}")

    # A MISTYPED PRICE. This is the expensive fat finger.
    why = g(**{**ok, "price": 31.85, "reference": 318.50})
    check(why and "%" in why,
          f"a limit at a tenth of the price was allowed: {why!r}")
    why = g(**{**ok, "price": 3185.0, "reference": 318.50})
    check(why is not None, "a limit at ten times the price was allowed")
    # With no reference there is nothing to compare against, and refusing
    # every order because the tape is empty would be its own failure.
    check(g(**{**ok, "price": 31.85, "reference": None}) is None,
          "an order was refused for distance with no reference price to "
          "measure against")
    # A market order has no price to check, and must still be allowed.
    check(g(**{**ok, "price": None}) is None,
          "a market order was refused by the price guards")


# ── the three switches ──────────────────────────────────────────────────────
def _arm_runtime():
    """Both outer gates on, for the cases that are about something else."""
    config.TRADING_ENABLED = True
    config.CONTROL_TOKEN = "T"
    broker.set_trading(True, token="T", who="check")


def case_runtime_toggle():
    """The runtime switch cannot escalate past the environment gate.

    WHY IT EXISTS: flipping the environment variable means a restart, and a
    restart drops the upstream socket and every pane's buffer mid-session.
    Changing your mind about being armed should not cost the tape.

    WHAT MUST HOLD: the environment stays the outer gate, the flag is off on
    every start, enabling needs the shared secret, and DISABLING never needs
    anything — a control that fails closed at the worst moment is not a
    safety control.
    """
    config.TRADING_ENABLED = False
    config.CONTROL_TOKEN = "T"
    broker._runtime_enabled = False

    check(RUNTIME_AT_IMPORT is False,
          f"the runtime flag is {RUNTIME_AT_IMPORT!r} at import — it must be "
          f"off on every start, or a crash nobody watched brings the service "
          f"back able to trade")
    check(not broker.trading_allowed(),
          "trading was allowed with the environment gate off")
    try:
        broker.set_trading(True, token="T")
        FAILS.append("the runtime toggle enabled trading past the environment "
                     "gate — the environment is supposed to be the outer one")
    except broker.BrokerError as exc:
        check("LIVE_TRADING_ENABLED" in str(exc),
              f"the refusal does not name the outer gate: {exc}")
    check(not broker.trading_allowed(),
          "a refused enable still left trading allowed")

    # With the environment on, the flag decides.
    config.TRADING_ENABLED = True
    check(not broker.trading_allowed(),
          "the runtime flag defaulted to ON; it must be off on every start, "
          "or a crash nobody watched brings the service back able to trade")

    for bad in (None, "", "wrong", "t"):
        try:
            broker.set_trading(True, token=bad)
            FAILS.append(f"token {bad!r} was accepted")
        except broker.BrokerError:
            pass
    check(not broker.trading_allowed(), "a bad token still enabled trading")

    st = broker.set_trading(True, token="T", who="someone")
    check(st["allowed"] and st["runtime_enabled"],
          f"the right token did not enable trading: {st}")
    check(st["changed_by"] == "someone" and st["changed_at"],
          f"who and when were not recorded: {st}")

    # DISABLING NEEDS NOTHING. Same rule as cancel.
    st = broker.set_trading(False)
    check(not st["allowed"] and not st["runtime_enabled"],
          f"disabling without a token was refused or ignored: {st}")

    # With no token configured, enabling over HTTP is refused rather than
    # left open — this service is reachable from the internet.
    config.CONTROL_TOKEN = ""
    try:
        broker.set_trading(True, token="anything")
        FAILS.append("trading was enabled with no control token configured")
    except broker.BrokerError as exc:
        check("LIVE_CONTROL_TOKEN" in str(exc),
              f"the refusal does not say what to set: {exc}")
    # And disabling still works with no token configured at all.
    check(not broker.set_trading(False)["allowed"],
          "disabling failed when no token was configured")

    # The reported state has to agree with the decision. A page or another
    # service reading `allowed` must get the same answer an order would.
    for env in (False, True):
        for runtime in (False, True):
            config.TRADING_ENABLED = env
            broker._runtime_enabled = runtime
            st = broker.trading_state()
            check(st["allowed"] == (env and runtime),
                  f"env={env} runtime={runtime} reported allowed="
                  f"{st['allowed']}")
            check(bool(st["why"]), "no reason was given for the state")


def case_switches():
    rec = Recorder()
    schwab._acall = rec
    schwab._account_hash = "H"
    args = dict(symbol="FDX", side="BUY", qty=100, price=318.50,
                reference=318.50, position_qty=0)

    # 1. the server switch
    config.TRADING_ENABLED = False
    try:
        asyncio.run(broker.place(armed=True, **args))
        FAILS.append("an order was placed with LIVE_TRADING_ENABLED off")
    except broker.BrokerError as exc:
        check("LIVE_TRADING_ENABLED" in str(exc),
              f"the refusal does not name the switch to flip: {exc}")
    check(not rec.calls, "a refused order still reached the transport")

    # 2. the RUNTIME switch, between the environment and the pane. Exists so
    #    arming does not cost a restart, and a restart drops the upstream
    #    socket and every pane's buffer.
    config.TRADING_ENABLED = True
    broker._runtime_enabled = False
    try:
        asyncio.run(broker.place(armed=True, **args))
        FAILS.append("an order was placed with the runtime switch off")
    except broker.BrokerError as exc:
        check("runtime" in str(exc),
              f"the refusal does not mention the runtime switch: {exc}")
    check(not rec.calls, "an order refused at runtime reached the transport")

    # 3. the pane switch. The BODY carries it — the server does not trust
    #    the button, because the button is not what sends the request.
    _arm_runtime()
    try:
        asyncio.run(broker.place(armed=False, **args))
        FAILS.append("an order was placed by an unarmed pane")
    except broker.BrokerError as exc:
        check("armed" in str(exc).lower(),
              f"the refusal does not say the pane is unarmed: {exc}")
    check(not rec.calls, "an unarmed order still reached the transport")

    # 4. the guards, with every switch on
    try:
        asyncio.run(broker.place(armed=True, **{**args, "qty": 99999}))
        FAILS.append("the guards were skipped once the switches were on")
    except broker.BrokerError as exc:
        check("guards" in str(exc), f"a guard refusal is not labelled: {exc}")
    check(not rec.calls, "an order refused by the guards reached Schwab")

    # All three on: it goes.
    out = asyncio.run(broker.place(armed=True, **args))
    check(out.get("ok") and out.get("order_id") == "NEW1",
          f"a fully permitted order did not go through: {out}")
    check(len(rec.calls) == 1 and rec.calls[0]["method"] == "POST",
          f"expected one POST, got {rec.verbs()}")
    body = rec.calls[0]["body"]
    check(body["orderType"] == "LIMIT" and body["price"] == "318.50",
          f"the order body is wrong: {body}")
    check(out.get("rt_ms") is not None,
          "no round-trip time was reported — it is a number the page shows")

    # CANCEL IS NOT GATED. A disarmed pane must still be able to undo.
    rec.calls.clear()
    config.TRADING_ENABLED = False
    out = asyncio.run(broker.cancel(order_id="X1"))
    check(out.get("ok"),
          "a cancel was refused because trading was disabled — a safety "
          "switch that traps an order is not a safety switch")
    check(rec.calls and rec.calls[0]["priority"],
          "a cancel was sent as ordinary traffic; it must be able to spend "
          "the reserve")
    config.TRADING_ENABLED = True


# ── flatten cancels first ───────────────────────────────────────────────────
def case_flatten_order():
    rec = Recorder(_acct(positions=[pos("FDX", long_q=300, avg=318.0)],
                         orders=[order("O1", "FDX", "BUY", 100, 317.5),
                                 order("O2", "FDX", "BUY", 100, 317.0),
                                 order("O9", "NVDA", "BUY", 5, 900.0)]))
    schwab._acall = rec
    schwab._account_hash = "H"
    _arm_runtime()

    # FLATTEN OBEYS THE RUNTIME FLAG TOO. It sends a market order, so a
    # flatten that consulted the environment gate alone would trade while the
    # runtime switch said no.
    config.TRADING_ENABLED = True
    broker._runtime_enabled = False
    try:
        asyncio.run(broker.flatten(symbol="FDX", armed=True))
        FAILS.append("flatten sent a market order with the runtime switch off")
    except broker.BrokerError:
        pass
    check(not rec.calls, "a refused flatten still reached the transport")
    _arm_runtime()

    out = asyncio.run(broker.flatten(symbol="FDX", armed=True))
    seq = [c["method"] for c in rec.calls]
    check(out.get("ok"), f"flatten failed: {out}")

    # THE SAFETY PROPERTY. Every DELETE must precede the closing POST: a
    # resting order left working can reopen the position the moment the
    # flatten fills.
    if "POST" in seq:
        first_post = seq.index("POST")
        deletes = [i for i, m in enumerate(seq) if m == "DELETE"]
        check(deletes and all(i < first_post for i in deletes),
              f"the market close was sent before a cancel: {seq} — a resting "
              f"order can reopen the position the instant the flatten fills")
    else:
        FAILS.append(f"flatten never closed the position: {seq}")

    check(len(out.get("cancelled") or []) == 2,
          f"cancelled {out.get('cancelled')} — the two FDX orders should go "
          f"and the NVDA one should not")
    close = [c for c in rec.calls if c["method"] == "POST"][-1]["body"]
    check(close["orderType"] == "MARKET",
          f"the closing order is not a market order: {close}")
    check(close["orderLegCollection"][0]["instruction"] == "SELL"
          and close["orderLegCollection"][0]["quantity"] == 300,
          f"a 300-share long was not closed by selling 300: {close}")
    check(all(c["priority"] for c in rec.calls),
          "part of the flatten path was ordinary traffic; getting flat must "
          "be able to spend the reserve")

    # A short closes by buying to cover, and a flat name does nothing.
    rec2 = Recorder(_acct(positions=[pos("FDX", short_q=200, avg=318.0)]))
    schwab._acall = rec2
    asyncio.run(broker.flatten(symbol="FDX", armed=True))
    close = [c for c in rec2.calls if c["method"] == "POST"][-1]["body"]
    check(close["orderLegCollection"][0]["instruction"] == "BUY_TO_COVER",
          f"a short was not covered: {close}")

    rec3 = Recorder(_acct())
    schwab._acall = rec3
    out = asyncio.run(broker.flatten(symbol="FDX", armed=True))
    check(out.get("flat") and not [c for c in rec3.calls if c["method"] == "POST"],
          f"a flat name still sent a market order: {out}")


# ── reading the record ──────────────────────────────────────────────────────
def case_state():
    rec = Recorder(_acct(
        positions=[pos("FDX", long_q=300, avg=318.0), pos("NVDA", long_q=5)],
        orders=[order("O1", "FDX", "BUY", 100, 317.5),
                order("O2", "FDX", "BUY", 100, 317.0, status="FILLED"),
                order("O3", "FDX", "SELL", 100, 319.0, status="CANCELED"),
                order("O4", "FDX", "BUY", 100, 316.0, status="PENDING_ACTIVATION")]))
    schwab._acall = rec
    schwab._account_hash = "H"
    st = asyncio.run(broker.state(["FDX"]))

    ids = sorted(o["order_id"] for o in st["working"])
    check(ids == ["O1", "O4"],
          f"working orders came back as {ids} — a FILLED or CANCELED order "
          f"drawn as working is an order the position is not behind")
    check(len(st["positions"]) == 1
          and st["positions"][0]["symbol"] == "FDX"
          and st["positions"][0]["qty"] == 300,
          f"the position filter or sign is wrong: {st['positions']}")
    check(st["as_of"] and abs(st["as_of"] - time.time()) < 5,
          "the state carries no usable timestamp — the page cannot say how "
          "stale it is, which is the one thing it must be able to say")
    check(not any(c["priority"] for c in rec.calls),
          "the background state read spends the reserve; the reserve exists "
          "so a cancel is never the call refused for quota")

    # A short reads negative, so the guards and the display agree on sign.
    rec2 = Recorder(_acct(positions=[pos("FDX", short_q=200)]))
    schwab._acall = rec2
    st = asyncio.run(broker.state(["FDX"]))
    check(st["positions"][0]["qty"] == -200,
          f"a 200-share short read as {st['positions'][0]['qty']}")

    # Every non-terminal Schwab status has to survive as working, or an order
    # silently disappears from the screen while resting at the broker.
    for status in sorted(schwab.WORKING_STATES):
        rec3 = Recorder(_acct(orders=[order("O", "FDX", "BUY", 1, 1.0,
                                            status=status)]))
        schwab._acall = rec3
        st = asyncio.run(broker.state(["FDX"]))
        check(len(st["working"]) == 1,
              f"status {status} was dropped from the working list")


# ── the rate limiter ────────────────────────────────────────────────────────
def case_limiter():
    lim = schwab.RateLimiter(per_min=10, reserve=3)

    # Ordinary traffic stops at the reserve.
    for i in range(7):
        check(lim.take() is None, f"ordinary call {i + 1} of 7 was refused")
    why = lim.take()
    check(why and "held back" in why,
          f"ordinary traffic spent into the reserve: {why!r}")

    # Priority may spend it — this is the cancel that must always go.
    for i in range(3):
        check(lim.take(priority=True) is None,
              f"priority call {i + 1} of 3 was refused inside the reserve")
    check(lim.take(priority=True) is not None,
          "priority spent past the total ceiling")

    # 429 blocks everything, priority included, and says for how long.
    lim2 = schwab.RateLimiter(per_min=100, reserve=10)
    lim2.note_429("2")
    why = lim2.take(priority=True)
    check(why and "429" in why,
          f"a 429 did not hold off a subsequent call: {why!r}")
    check(lim2.state()["n_429"] == 1, "the 429 was not counted")
    # A missing or junk Retry-After must not crash or become forever.
    for hdr in (None, "", "soon", "99999"):
        lim3 = schwab.RateLimiter(100, 10)
        lim3.note_429(hdr)
        wait = lim3.state()["blocked_for_s"]
        check(0 < wait <= 60,
              f"Retry-After {hdr!r} produced a {wait}s hold-off")

    # The window rolls: a call from over a minute ago is not spent quota.
    lim4 = schwab.RateLimiter(2, 0)
    lim4.calls.append(time.time() - 61)
    lim4.calls.append(time.time() - 61)
    check(lim4.take() is None,
          "calls older than the minute still counted against the budget")


# ── the order body ──────────────────────────────────────────────────────────
def case_order_body():
    b = schwab._equity_order("buy", 100, "fdx", 318.5200000000001)
    check(b["price"] == "318.52",
          f"a float limit was sent unrounded as {b['price']!r} — Schwab "
          f"refuses a sub-penny equity limit")
    check(b["orderLegCollection"][0]["instrument"]["symbol"] == "FDX",
          "the symbol was not upper-cased")
    check(b["orderLegCollection"][0]["instruction"] == "BUY",
          "the instruction was not upper-cased")
    check(b["duration"] == "DAY" and b["session"] == "NORMAL",
          f"an unexpected duration or session: {b}")

    m = schwab._equity_order("SELL", 50, "FDX", None)
    check(m["orderType"] == "MARKET" and "price" not in m,
          f"a market order carries a price: {m}")

    # Round-trip through JSON, because that is how it actually leaves.
    check(json.loads(json.dumps(b))["price"] == "318.52",
          "the price does not survive serialisation as a 2-decimal string")


# ── determinate vs unknown ──────────────────────────────────────────────────
def case_indeterminacy():
    """A rejection and a timeout must not look alike to the caller.

    THE DANGEROUS CASE. A 4xx means Schwab looked at the request and refused
    it; nothing happened and a retry is safe. A timeout, a dropped connection
    or a 5xx means the order may be resting right now. Collapsing the two is
    how a timeout becomes a retry becomes a double position.
    """
    check(issubclass(broker.BrokerIndeterminate, broker.BrokerError),
          "BrokerIndeterminate is not a BrokerError, so existing handlers "
          "would stop catching it")
    check(broker.BrokerIndeterminate is not broker.BrokerError,
          "the two failure kinds are the same class, so nothing can tell a "
          "rejection from an unknown")

    # THE CLASSIFICATION ITSELF, in the transport where it is decided.
    #
    # Testing only the class hierarchy left the actual decision unchecked:
    # a version of _acall that raised a plain BrokerError on a timeout
    # passed, which is the exact bug this case exists to prevent.
    class FakeResp:
        def __init__(self, code):
            self.status_code, self.headers, self.content = code, {}, b"x"
            self.text = "boom"

        def json(self):
            return {}

    class FakeClient:
        def __init__(self, behaviour):
            self.behaviour = behaviour
            self.is_closed = False

        async def request(self, *a, **kw):
            if self.behaviour == "timeout":
                raise TimeoutError("read timed out")
            return FakeResp(self.behaviour)

    def drive(behaviour):
        schwab._conn = FakeClient(behaviour)
        schwab._httpx = lambda: type("H", (), {})()
        schwab._client = lambda _h: schwab._conn
        schwab._token = _fake_token
        schwab._account_hash = "H"
        try:
            asyncio.run(REAL_ACALL("POST", "/accounts/H/orders",
                                   body={"x": 1}, priority=True))
            return None
        except broker.BrokerError as exc:
            return exc
        finally:
            schwab._conn = None

    for behaviour, want_unknown, label in (
            ("timeout", True, "a timeout"),
            (503, True, "a 5xx"),
            (400, False, "a 4xx")):
        exc = drive(behaviour)
        check(exc is not None, f"{label} did not raise at all")
        if exc is None:
            continue
        got_unknown = isinstance(exc, broker.BrokerIndeterminate)
        check(got_unknown == want_unknown,
              f"{label} raised {type(exc).__name__}; expected "
              f"{'BrokerIndeterminate' if want_unknown else 'BrokerError'}. "
              f"A 4xx means Schwab refused and nothing happened; a timeout "
              f"or a 5xx means the order may be resting right now, and "
              f"collapsing the two is how a timeout becomes a retry becomes "
              f"a double position.")

    import live.main as live_main
    det = live_main._broker_fail(broker.BrokerError("guards refused"))
    ind = live_main._broker_fail(broker.BrokerIndeterminate("timed out"))
    check(det["indeterminate"] is False,
          f"a plain refusal was reported as indeterminate: {det}")
    check(ind["indeterminate"] is True,
          f"a timeout was reported as determinate: {ind} — the pane would "
          f"treat it as 'nothing happened' and allow another order on top")


# ── matching a placement nobody saw the answer to ───────────────────────────
def case_match_placement():
    """The reconciliation rule, including the case it refuses to guess at."""
    import time as _t
    sent = _t.time()
    after = "2026-09-03T20:00:10+0000"
    before = "2026-09-03T19:00:00+0000"

    def o(oid, sym="FDX", side="BUY", qty=100, price=318.50, entered=after):
        return {"order_id": oid, "symbol": sym, "side": side, "qty": qty,
                "price": price, "entered": entered, "status": "WORKING"}

    want = dict(symbol="FDX", side="BUY", qty=100, price=318.50,
                sent_at=0)          # epoch 0 so `entered` always passes

    r = broker.match_placement([], **want)
    check(r["state"] == "absent", f"an empty list matched: {r}")

    r = broker.match_placement([o("A")], **want)
    check(r["state"] == "found" and r["order"]["order_id"] == "A",
          f"the one matching order was not found: {r}")

    # THE AMBIGUITY. Two identical orders are indistinguishable, and the
    # rule reports that rather than picking one — a guess would attach the
    # pane to an order that might be the other, and the next nudge would
    # reprice a stranger's order.
    r = broker.match_placement([o("A"), o("B")], **want)
    check(r["state"] == "ambiguous" and len(r["orders"]) == 2,
          f"two identical orders produced {r['state']} instead of an "
          f"explicit ambiguity — this is the case that must never be guessed")

    # Every field has to discriminate, or the match is looser than it reads.
    for label, bad in (("symbol", o("A", sym="NVDA")),
                       ("side", o("A", side="SELL")),
                       ("quantity", o("A", qty=50)),
                       ("price", o("A", price=318.60))):
        r = broker.match_placement([bad], **want)
        check(r["state"] == "absent",
              f"an order differing only in {label} was matched: {r}")

    # A cent of tolerance, because the price went out as a 2-decimal string.
    r = broker.match_placement([o("A", price=318.5000001)], **want)
    check(r["state"] == "found", "float noise on the price broke the match")

    # ENTERED BEFORE WE SENT. An order already resting is not the one just
    # placed, and matching it would adopt somebody else's order.
    r = broker.match_placement([o("A", entered=before)],
                               **{**want, "sent_at": _t.mktime(
                                   _t.strptime("2026-09-03T19:59:00",
                                               "%Y-%m-%dT%H:%M:%S"))})
    check(r["state"] == "absent",
          f"an order entered before the placement was matched as it: {r}")

    # A market order has no price, and must still match on everything else.
    r = broker.match_placement([o("A", price=None)],
                               **{**want, "price": None})
    check(r["state"] == "found", "a market order could not be reconciled")

    # THE CAVEAT ITSELF. It is the thing most likely to be lost in a tidy-up,
    # and it is the reason the ambiguity above is acceptable rather than a bug.
    # It is no longer provisional: the tag probe ran on 2026-09-03 and the way
    # out is closed, so the words now have to say that too.
    # IN THE ADAPTER, because the caveat is Schwab's: its API gives a
    # placement no client-supplied handle, which is what forces matching by
    # shape. base.match_placement is the shared mechanism; this is the reason.
    src = (ROOT / "live" / "brokers" / "schwab.py").read_text(encoding="utf-8")
    body = src[src.index("async def reconcile"):src.index("def _equity_order")]
    for phrase in ("NOT GUARANTEED UNIQUE", "probe_schwab_tag",
                   "ADJACENT PRICES", "NOT SETTABLE"):
        check(phrase in body,
              f"the matching function no longer explains {phrase!r} — the "
              f"ambiguity is tested and permanent, and it belongs where the "
              f"matching happens")

    # AND THE ORDER BODY STAYS CLEAN. A tag there is not a label, it is a
    # rejected order: Schwab 400s the whole request over the field.
    order_src = src[src.index("def _equity_order"):src.index("async def place")]  # adapter
    check('"tag"' not in order_src and "'tag'" not in order_src,
          "the order body carries a `tag` again. Schwab REJECTS an order "
          "that carries one — 400 with the tagged body, 201 with the "
          "identical body without it, tested 2026-09-03. This does not "
          "mislabel orders, it stops them leaving.")




# ── the token refresh ───────────────────────────────────────────────────────
#
# THE PATH THAT TOOK THE PANE DOWN ON 2026-09-17, and the reason it reached
# production: it only runs when a refresh is actually REFUSED, and nothing
# here had ever driven it. `_do_refresh` read `r.reason` on an httpx response,
# which spells it `reason_phrase` and has no attribute of that name, so the
# 400 turned into an AttributeError; the handler upstream caught it and
# reported it as the reason the refresh failed. Schwab's own answer — the only
# thing that says whether the token needs re-authorising — never reached the
# screen.
#
# So the case drives BOTH outcomes through a FAKE httpx whose response carries
# httpx's attributes AND NO OTHERS. Reading anything else raises, which is
# what reproduces the fault rather than asserting around it.
class _StrictResponse:
    """An httpx.Response as far as this code is concerned. Nothing more.

    `__getattr__` refuses every other name, so a requests-ism (`.reason`,
    `.ok`, `.raise_for_status()`) fails here exactly as it failed on the box.
    """

    _ALLOWED = {"status_code", "text", "content", "headers", "reason_phrase", "json"}

    def __init__(self, status, payload, phrase="Bad Request"):
        object.__setattr__(self, "status_code", status)
        object.__setattr__(self, "text", payload if isinstance(payload, str) else json.dumps(payload))
        object.__setattr__(self, "content", self.text.encode())
        object.__setattr__(self, "headers", {})
        object.__setattr__(self, "reason_phrase", phrase)
        object.__setattr__(self, "_payload", payload)

    def json(self):
        if isinstance(self._payload, str):
            raise ValueError("not json")
        return self._payload

    def __getattr__(self, name):
        if name in self._ALLOWED:
            raise AttributeError(name)
        raise AttributeError(
            f"{type(self).__name__!r} object has no attribute {name!r} — httpx "
            f"does not have it either. This is the 2026-09-17 fault: the error "
            f"path read a requests-style attribute off an httpx response.")


class _FakeHttpx:
    """Stands in for the httpx module; records the refresh request."""

    def __init__(self, response):
        self.response = response
        self.posts = []

    def post(self, url, **kw):
        self.posts.append({"url": url, **kw})
        return self.response


def _token_env(tmp, tokens):
    """Point the adapter at a throwaway token file holding `tokens`."""
    Path(tmp).write_text(json.dumps(tokens), encoding="utf-8")
    config.SCHWAB_TOKEN_FILE = str(tmp)
    config.SCHWAB_API_KEY = config.SCHWAB_API_KEY or "KEY"
    config.SCHWAB_API_SECRET = config.SCHWAB_API_SECRET or "SECRET"


def case_token_refresh():
    import tempfile
    real_httpx, real_file = schwab._httpx, config.SCHWAB_TOKEN_FILE
    real_key, real_secret = config.SCHWAB_API_KEY, config.SCHWAB_API_SECRET
    tmpdir = tempfile.mkdtemp(prefix="obtok")
    tmp = Path(tmpdir) / "schwab_tokens.json"
    expired = {"access_token": "OLD", "refresh_token": "R1",
               "expires_at": time.time() - 1}
    try:
        # ── 1. A REFUSED REFRESH SURFACES AS A MESSAGE, not an AttributeError.
        _token_env(tmp, expired)
        body = {"error": "unsupported_token_type",
                "error_description": "400 Bad Request: refresh token invalid"}
        fake = _FakeHttpx(_StrictResponse(400, body))
        schwab._httpx = lambda: fake
        try:
            schwab._token_sync()
            check(False, "a refused refresh returned a token")
        except broker.BrokerError as exc:
            msg = str(exc)
            check("attribute" not in msg.lower(),
                  f"the refusal surfaced as a crash in the error path, not as "
                  f"Schwab's answer: {msg}")
            check("400" in msg and "refresh token invalid" in msg,
                  f"the message does not carry Schwab's own answer, which is the "
                  f"only thing that says whether the token needs re-authorising: {msg}")
        except Exception as exc:                            # noqa: BLE001
            check(False, f"a refused refresh raised {type(exc).__name__}: {exc} — "
                         f"the pane can only show a BrokerError")

        # The refresh really was attempted, with the grant the API wants.
        check(len(fake.posts) == 1
              and fake.posts[0]["data"]["grant_type"] == "refresh_token"
              and fake.posts[0]["data"]["refresh_token"] == "R1",
              f"the refresh request was not the one Schwab expects: {fake.posts}")

        # ── 2. AND IT REACHES THE PANE. health() carries last_error, which is
        #      the string the pane puts under "ORDER STATE MAY BE STALE".
        h = broker.health()
        check("token refresh failed" in (h.get("last_error") or ""),
              f"the refusal did not reach health().last_error: {h.get('last_error')}")

        # ── 3. THE SUCCESS PATH still works and is atomic: the file holds the
        #      NEW pair, because the portfolio dashboard reads this same file
        #      and the old refresh token is dead the moment this one lands.
        _token_env(tmp, expired)
        fresh = {"access_token": "NEW", "refresh_token": "R2", "expires_in": 1800}
        schwab._httpx = lambda: _FakeHttpx(_StrictResponse(200, fresh, "OK"))
        got = schwab._token_sync()
        on_disk = json.loads(tmp.read_text(encoding="utf-8"))
        check(got == "NEW", f"the refreshed access token was not returned: {got!r}")
        check(on_disk["access_token"] == "NEW" and on_disk["refresh_token"] == "R2",
              f"the rotated pair was not written back for the other process: {on_disk}")
        check(on_disk.get("expires_at", 0) > time.time() + 1000,
              "expires_at was not stamped, so every call would refresh again")

        # ── 4. A LIVE TOKEN IS NOT REFRESHED. Two processes share this file and
        #      Schwab rotates the refresh token on every refresh, so a needless
        #      one is how the other process ends up holding a dead token.
        _token_env(tmp, {"access_token": "LIVE", "refresh_token": "R9",
                         "expires_at": time.time() + 3600})
        spy = _FakeHttpx(_StrictResponse(500, "should not be called"))
        schwab._httpx = lambda: spy
        check(schwab._token_sync() == "LIVE" and spy.posts == [],
              "a token with an hour left was refreshed anyway")

        # ── 5. NO TOKEN FILE AT ALL: a message naming the file and what to do,
        #      not a stack trace.
        tmp.unlink()
        try:
            schwab._token_sync()
            check(False, "a missing token file returned a token")
        except broker.BrokerError as exc:
            check("authorise" in str(exc).lower(),
                  f"the missing-token message does not say how to fix it: {exc}")
    finally:
        schwab._httpx = real_httpx
        config.SCHWAB_TOKEN_FILE = real_file
        config.SCHWAB_API_KEY, config.SCHWAB_API_SECRET = real_key, real_secret
        schwab._last_error = None
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)


# ── the switches sit ABOVE the adapter ──────────────────────────────────────
#
# THE SAFETY PROPERTY OF THE REFACTOR, and the reason a second broker is safe
# to add. Every check runs in live/broker.py before any adapter is reached, so
# a new adapter cannot trade while disarmed or past the guards by forgetting
# to ask. These cases drive a FAKE adapter and assert on WHETHER IT WAS
# CALLED AT ALL -- an adapter that is never reached cannot send anything,
# which is a stronger statement than "the call raised".
class FakeBroker(base.Broker):
    """Records what the façade asks of it. Sends nothing anywhere."""

    name = "fake"

    def __init__(self):
        self.calls = []

    async def read_orders(self, symbols=None, priority=False):
        self.calls.append(("read_orders", symbols, priority))
        return {"ok": True, "as_of": time.time(), "rt_ms": 1.0,
                "working": [], "recent": [], "limits": None,
                "stale_after_s": config.STALE_AFTER_S}

    async def read_positions(self, symbols=None, priority=False):
        self.calls.append(("read_positions", symbols, priority))
        return {"ok": True, "as_of": time.time(), "rt_ms": 1.0,
                "positions": [], "account_type": "MARGIN", "is_day_trader": False,
                "round_trips": 0, "limits": None,
                "stale_after_s": config.STALE_AFTER_S}

    async def place(self, *, symbol, side, qty, price, route=None):
        self.calls.append(("place", symbol, side, qty, price, route))
        return {"ok": True, "order_id": "F1", "status": 201, "rt_ms": 1.0}

    async def replace(self, *, order_id, symbol, side, qty, price, route=None,
                      filled=0.0):
        self.calls.append(("replace", order_id, symbol, side, qty, price,
                           route, filled))
        return {"ok": True, "order_id": order_id, "status": 200, "rt_ms": 1.0}

    async def cancel(self, *, order_id):
        self.calls.append(("cancel", order_id))
        return {"ok": True, "order_id": order_id, "status": 200, "rt_ms": 1.0}

    async def flatten(self, *, symbol):
        self.calls.append(("flatten", symbol))
        return {"ok": True, "cancelled": [], "rt_ms": 1.0, "flat": True}

    def health(self):
        return {"broker": "fake", "its_own_fact": 42}

    def problems(self):
        return ["fake broker"]


def _with_fake():
    """Install the fake adapter and return it (call _restore_real after)."""
    fake = FakeBroker()
    brokers._active = fake
    return fake


def _restore_real():
    brokers.reset()


def case_policy_above_adapter():
    fake = _with_fake()
    try:
        ok = dict(symbol="AAPL", side="BUY", qty=10, price=100.0,
                  reference=100.0, position_qty=0.0)

        # 1. DISARMED: not "the adapter refused", but never asked.
        config.TRADING_ENABLED = True
        broker.set_trading(False)
        try:
            asyncio.run(broker.place(armed=True, **ok))
            check(False, "a placement left while the runtime flag was off")
        except broker.BrokerError:
            pass
        check(fake.calls == [],
              "the adapter was called with trading switched off — the switches "
              "must be checked ABOVE it, or every new broker has to remember to")

        # 2. ARMED runtime, but the pane did not carry its arm flag.
        _arm_runtime()
        try:
            asyncio.run(broker.place(armed=False, **ok))
            check(False, "a placement left without the pane's arm flag")
        except broker.BrokerError:
            pass
        check(fake.calls == [], "the adapter was called for an unarmed pane")

        # 3a. THE ROUTE REACHES THE ADAPTER UNCHANGED. It is not policy --
        #     which venue an order goes to is a trading choice, and the
        #     guards bound size, ending position, notional and distance,
        #     none of which move with the venue. What matters is that the
        #     façade does not quietly drop it.
        fake.calls.clear()
        asyncio.run(broker.place(armed=True, route="ARCA", **ok))
        check(fake.calls and fake.calls[-1][-1] == "ARCA",
              f"the route did not reach the adapter: {fake.calls[-1:]!r}")
        fake.calls.clear()
        asyncio.run(broker.replace(order_id="X", armed=True, route="NSDQ",
                                   **ok))
        # The replace tuple is (…, price, route, filled): name the position
        # rather than counting from the end, which a new argument moves.
        check(fake.calls and fake.calls[-1][6] == "NSDQ",
              f"the route did not reach the adapter on a replace: "
              f"{fake.calls[-1:]!r}")

        # 3b. AND A ROUTE DOES NOT BUY A WAY PAST THE GUARDS. Naming a venue
        #     is not an escalation: the switches and the limits are checked
        #     before the adapter either way.
        fake.calls.clear()
        try:
            asyncio.run(broker.place(armed=False, route="ARCA", **ok))
            check(False, "an unarmed placement left because it named a route")
        except broker.BrokerError:
            pass
        check(fake.calls == [],
              "the adapter was called for an unarmed pane that named a route")

        # 3. ARMED, but the guards refuse it (over the share limit).
        try:
            asyncio.run(broker.place(armed=True,
                                     **{**ok, "qty": config.MAX_ORDER_SHARES + 1}))
            check(False, "a placement over the share limit left")
        except broker.BrokerError:
            pass
        check(fake.calls == [], "the adapter was called for an order the guards refused")

        # 4. ARMED and within the guards: NOW it is reached, and with the
        #    order alone — no arming or guard inputs leak into the adapter.
        r = asyncio.run(broker.place(armed=True, **ok))
        check(r["ok"] and fake.calls == [("place", "AAPL", "BUY", 10, 100.0,
                                          None)],
              f"an armed, guarded placement did not reach the adapter cleanly: {fake.calls}")

        # 5. REPLACE takes the same two checks — a nudge is an order, and a
        #    fat-fingered price arrives on the reprice.
        fake.calls.clear()
        try:
            asyncio.run(broker.replace(order_id="X", armed=True,
                                       **{**ok, "price": 10.0}))
            check(False, "a reprice 90% from the last print left")
        except broker.BrokerError:
            pass
        check(fake.calls == [], "the adapter was called for a reprice the guards refused")
        asyncio.run(broker.replace(order_id="X", armed=True, **ok))
        check(fake.calls == [("replace", "X", "AAPL", "BUY", 10, 100.0, None, 0.0)],
              "an armed, guarded reprice did not reach the adapter")

        # 6. CANCEL is NOT gated on arming — deliberately, and the fake proves
        #    it reaches the broker while disarmed.
        fake.calls.clear()
        broker.set_trading(False)
        r = asyncio.run(broker.cancel(order_id="Z9"))
        check(r["ok"] and fake.calls == [("cancel", "Z9")],
              "cancel did not reach the adapter while disarmed — a safety switch "
              "that traps an order is not a safety switch")

        # 7. FLATTEN needs trading allowed (it sends a market order), and is
        #    refused above the adapter when it is not.
        fake.calls.clear()
        try:
            asyncio.run(broker.flatten(symbol="AAPL", armed=True))
            check(False, "a flatten left while trading was switched off")
        except broker.BrokerError:
            pass
        check(fake.calls == [], "the adapter was called to flatten while disarmed")
        _arm_runtime()
        asyncio.run(broker.flatten(symbol="AAPL", armed=True))
        check(fake.calls == [("flatten", "AAPL")],
              "an allowed flatten did not reach the adapter")
    finally:
        broker.set_trading(False)
        _restore_real()


# ── the façade is the only thing above the adapters ─────────────────────────
def case_replace_quantities():
    """`qty` is the order's TOTAL and `filled` rides with it.

    The two brokers want different numbers on the wire -- DAS modifies the
    resting order and wants the total, Schwab cancels and places a new order
    and wants the remaining -- so the caller sends the order's own figures and
    the adapter does the arithmetic. Sending the remaining to DAS shrank a
    partially filled order on every nudge (5/4/3/2, 2026-09-22).
    """
    fake = _with_fake()
    try:
        config.TRADING_ENABLED = True
        _arm_runtime()
        asyncio.run(broker.replace(order_id="65377", symbol="LLY", side="SELL",
                                   qty=5, price=1166.40, filled=2, armed=True,
                                   reference=1166.00, position_qty=0.0))
        check(fake.calls == [("replace", "65377", "LLY", "SELL", 5, 1166.40, None, 2)],
              f"the façade did not pass the order's total and filled through: "
              f"{fake.calls}")
    finally:
        broker.set_trading(False)
        _restore_real()

    # SCHWAB'S WIRE: the remaining, because its replace is a new order and
    # the total would buy the executed part a second time.
    rec = Recorder({"account": _acct()})
    schwab._acall, schwab._account_hash = rec, "H"
    asyncio.run(schwab.replace(order_id="65377", symbol="LLY", side="SELL",
                               qty=5, price=1166.40, filled=2))
    body = rec.calls[-1]["body"]
    check(body["orderLegCollection"][0]["quantity"] == 3,
          f"Schwab was sent {body['orderLegCollection'][0]['quantity']} shares "
          f"for an order of 5 with 2 filled; its replace is a cancel and a new "
          f"order, so the new one is for what is LEFT")

    # And nothing left to reprice is refused rather than sent as zero.
    try:
        asyncio.run(schwab.replace(order_id="65377", symbol="LLY", side="SELL",
                                   qty=2, price=1166.40, filled=2))
        check(False, "a replace with nothing remaining was sent")
    except broker.BrokerError as exc:
        check("cancel" in str(exc).lower(),
              f"the refusal does not say what to do instead: {exc}")


def case_interface():
    # Schwab implements every method the interface declares: an abstract
    # method left out would raise here rather than at 09:31 on the morning it
    # is called.
    try:
        adapter = schwab.SchwabBroker()
        check(isinstance(adapter, base.Broker), "SchwabBroker is not a base.Broker")
    except TypeError as exc:
        check(False, f"SchwabBroker does not implement the interface: {exc}")
        return

    for name in ("read_orders", "read_positions", "state", "place", "replace",
                 "cancel", "flatten", "reconcile", "health", "problems", "aclose"):
        check(callable(getattr(adapter, name, None)),
              f"the interface is missing {name}, which the façade calls")

    # EVERY ADAPTER, not just the one this box is pointed at. An adapter
    # that cannot be constructed, or that is missing a method, must fail
    # here rather than at 09:31 on the morning it is selected. Constructing
    # one must also not connect to anything -- that is why this can run on a
    # laptop with no DAS and no network.
    import importlib
    for key in sorted(brokers._ADAPTERS):
        try:
            a = brokers._ADAPTERS[key]()
        except Exception as exc:                            # noqa: BLE001
            check(False, f"the {key} adapter could not be constructed: "
                         f"{type(exc).__name__}: {exc}")
            continue
        check(isinstance(a, base.Broker), f"{key} is not a base.Broker")
        check(a.name == key,
              f"the {key} adapter calls itself {a.name!r}; the key in "
              f"_ADAPTERS is what LIVE_BROKER is set to and what health() "
              f"reports, so they have to agree")
        # ROUTE IS ON THE INTERFACE, so every adapter has to accept it --
        # one that does not would raise TypeError the first time a venue is
        # picked on the page, which is a click on a live trading control.
        import inspect
        for meth in ("place", "replace"):
            params = inspect.signature(getattr(a, meth)).parameters
            check("route" in params,
                  f"{key}.{meth}() does not take `route`. The façade passes "
                  f"it for every broker; an adapter with no venue selection "
                  f"accepts it, ignores it, and says so in health().")
        # And every adapter states whether it routes at all, so the page can
        # ask one question rather than knowing which broker it is talking to.
        r = a.health().get("routing")
        check(isinstance(r, dict) and "supported" in r,
              f"{key}.health() does not carry `routing`, so the page cannot "
              f"tell whether to offer a venue control")

    # NO ADAPTER MAY CARRY POLICY. The switches and the guards are the
    # façade's; an adapter that consulted them would make the real gate
    # ambiguous, and one that re-implemented them would drift from it. This
    # scans EVERY adapter, so the DAS one is held to it the day it lands.
    import glob
    for path in glob.glob(str(ROOT / "live" / "brokers" / "*.py")):
        if path.endswith("base.py") or path.endswith("__init__.py"):
            continue
        src = Path(path).read_text(encoding="utf-8")
        for banned in ("_armed_check(", "check_guards(", "trading_allowed(",
                       "_runtime_enabled", "set_trading(", "armed=", "armed:"):
            check(banned not in src,
                  f"{Path(path).name} consults the trading policy ({banned}). "
                  f"Arming and the guards are checked once, in live/broker.py, "
                  f"above every adapter.")

    # The endpoints and the pane must reach a broker only through the façade.
    main_src = (ROOT / "live" / "main.py").read_text(encoding="utf-8")
    check("brokers.schwab" not in main_src and "from live.brokers" not in main_src,
          "live/main.py imports an adapter directly — the endpoints talk to "
          "live/broker.py, which is what makes the broker swappable")

    # An unknown LIVE_BROKER raises rather than silently falling back: a box
    # pointed at another broker must not keep trading through this one.
    was, brokers._active = brokers._active, None
    old = config.BROKER
    try:
        config.BROKER = "not_a_broker"
        try:
            brokers.active()
            check(False, "an unknown LIVE_BROKER fell back to a working broker")
        except broker.BrokerError as exc:
            check("not_a_broker" in str(exc), "the refusal does not name the value")
    finally:
        config.BROKER = old
        brokers._active = was

    # health() is the union: the adapter's own facts AND the switches, which
    # an adapter never reports on.
    fake = _with_fake()
    try:
        h = broker.health()
        check(h.get("its_own_fact") == 42, "the adapter's facts are missing from health()")
        check(h.get("broker") == "fake", "health() does not say which broker it is")
        check("trading" in h and "trading_enabled" in h and "guards" in h,
              "health() lost the switches or the guards, which the pane reads")
        check(broker.problems() == ["fake broker"], "problems() does not reach the adapter")
    finally:
        _restore_real()


# ── the two reads are separate ──────────────────────────────────────────────
def case_split_reads():
    """Orders and positions are polled at different rates, so ONE call each.

    Orders in this account live one to six seconds; positions move only when
    one of them fills. Pairing the reads made the cheap one wait on the
    expensive one and the expensive one run no more often than the cheap one
    needed.
    """
    rec = Recorder(_acct(positions=[pos("FDX", long_q=300)],
                         orders=[order("O1", "FDX", "BUY", 100, 317.5)]))
    schwab._acall = rec
    schwab._account_hash = "H"

    asyncio.run(broker.read_orders(["FDX"]))
    check(len(rec.calls) == 1 and rec.calls[0]["path"].endswith("/orders"),
          f"the orders read made {rec.verbs()} — it must not drag the "
          f"positions call along at 2-second intervals")

    rec.calls.clear()
    asyncio.run(broker.read_positions(["FDX"]))
    check(len(rec.calls) == 1 and not rec.calls[0]["path"].endswith("/orders"),
          f"the positions read made {rec.verbs()}")

    rec.calls.clear()
    st = asyncio.run(broker.state(["FDX"]))
    check(len(rec.calls) == 2,
          f"the combined read made {len(rec.calls)} calls, expected 2")
    check(st["working"] and st["positions"],
          f"the combined read lost half its payload: {st}")
    check(st["stale_after_s"] == config.STALE_AFTER_S,
          "the staleness threshold is not carried to the page")
    check(config.STALE_AFTER_S <= 6,
          f"the staleness threshold is {config.STALE_AFTER_S}s against orders "
          f"that live 1-6s — it would call a list current that had already "
          f"missed an order's whole life")


CASES = [
    ("the guards", case_guards),
    ("the runtime toggle", case_runtime_toggle),
    ("the four switches", case_switches),
    ("flatten cancels first", case_flatten_order),
    ("reading the record", case_state),
    ("split reads", case_split_reads),
    ("determinate vs unknown", case_indeterminacy),
    ("which application placed it", case_order_source),
    ("where my orders actually traded", case_fills),
    ("the recent list is the recent end", case_recent_is_newest),
    ("every order status is accounted for", case_order_states),
    ("matching a placement", case_match_placement),
    ("the rate limiter", case_limiter),
    ("the order body", case_order_body),
    ("the switches sit above the adapter", case_policy_above_adapter),
    ("replace quantities", case_replace_quantities),
    ("the broker interface", case_interface),
    ("the token refresh", case_token_refresh),
]


def main() -> int:
    real_acall, real_hash = schwab._acall, schwab._account_hash
    real_enabled = config.TRADING_ENABLED
    real_token, real_runtime = config.CONTROL_TOKEN, broker._runtime_enabled
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
        schwab._acall, schwab._account_hash = real_acall, real_hash
        config.TRADING_ENABLED = real_enabled
        config.CONTROL_TOKEN, broker._runtime_enabled = real_token, real_runtime

    print(f"\nbroker cases: {len(CASES)}, failures: {len(FAILS)}")
    if not FAILS:
        print("  four switches hold and the runtime one cannot escalate past "
              "the environment; flatten cancels before it closes; the guards "
              "bound the ENDING position; the reserve is intact")
    return 1 if FAILS else 0


sys.exit(main())
