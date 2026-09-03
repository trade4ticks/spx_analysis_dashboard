"""The order path, against a fabricated Schwab.

NO NETWORK AND NO ORDERS, which is the point: everything here is decidable
from a stubbed transport, and none of it can be checked by placing a real
order and deciding it looked about right. This is the one module in the
project where being wrong costs money rather than time.

WHAT IS BEING PROTECTED, in order of how bad it would be:

  * THE THREE SWITCHES. LIVE_TRADING_ENABLED, the pane's arm flag, and the
    guards. All three are checked SERVER-side; the browser is where a guard
    is easiest to bypass by accident — a stale page, a replayed request, a
    hand-typed fetch during debugging.

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
"""
from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from live import broker, config                             # noqa: E402

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


def order(oid, sym, side, qty, price, status="WORKING"):
    return {"orderId": oid, "quantity": qty, "filledQuantity": 0,
            "price": price, "orderType": "LIMIT", "status": status,
            "orderLegCollection": [{"instruction": side, "quantity": qty,
                                    "instrument": {"symbol": sym}}]}


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
def case_switches():
    rec = Recorder()
    broker._acall = rec
    broker._account_hash = "H"
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

    # 2. the pane switch. The BODY carries it — the server does not trust
    #    the button, because the button is not what sends the request.
    config.TRADING_ENABLED = True
    try:
        asyncio.run(broker.place(armed=False, **args))
        FAILS.append("an order was placed by an unarmed pane")
    except broker.BrokerError as exc:
        check("armed" in str(exc).lower(),
              f"the refusal does not say the pane is unarmed: {exc}")
    check(not rec.calls, "an unarmed order still reached the transport")

    # 3. the guards, with both switches on
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
    broker._acall = rec
    broker._account_hash = "H"
    config.TRADING_ENABLED = True

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
    broker._acall = rec2
    asyncio.run(broker.flatten(symbol="FDX", armed=True))
    close = [c for c in rec2.calls if c["method"] == "POST"][-1]["body"]
    check(close["orderLegCollection"][0]["instruction"] == "BUY_TO_COVER",
          f"a short was not covered: {close}")

    rec3 = Recorder(_acct())
    broker._acall = rec3
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
    broker._acall = rec
    broker._account_hash = "H"
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
    broker._acall = rec2
    st = asyncio.run(broker.state(["FDX"]))
    check(st["positions"][0]["qty"] == -200,
          f"a 200-share short read as {st['positions'][0]['qty']}")

    # Every non-terminal Schwab status has to survive as working, or an order
    # silently disappears from the screen while resting at the broker.
    for status in sorted(broker.WORKING_STATES):
        rec3 = Recorder(_acct(orders=[order("O", "FDX", "BUY", 1, 1.0,
                                            status=status)]))
        broker._acall = rec3
        st = asyncio.run(broker.state(["FDX"]))
        check(len(st["working"]) == 1,
              f"status {status} was dropped from the working list")


# ── the rate limiter ────────────────────────────────────────────────────────
def case_limiter():
    lim = broker.RateLimiter(per_min=10, reserve=3)

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
    lim2 = broker.RateLimiter(per_min=100, reserve=10)
    lim2.note_429("2")
    why = lim2.take(priority=True)
    check(why and "429" in why,
          f"a 429 did not hold off a subsequent call: {why!r}")
    check(lim2.state()["n_429"] == 1, "the 429 was not counted")
    # A missing or junk Retry-After must not crash or become forever.
    for hdr in (None, "", "soon", "99999"):
        lim3 = broker.RateLimiter(100, 10)
        lim3.note_429(hdr)
        wait = lim3.state()["blocked_for_s"]
        check(0 < wait <= 60,
              f"Retry-After {hdr!r} produced a {wait}s hold-off")

    # The window rolls: a call from over a minute ago is not spent quota.
    lim4 = broker.RateLimiter(2, 0)
    lim4.calls.append(time.time() - 61)
    lim4.calls.append(time.time() - 61)
    check(lim4.take() is None,
          "calls older than the minute still counted against the budget")


# ── the order body ──────────────────────────────────────────────────────────
def case_order_body():
    b = broker._equity_order("buy", 100, "fdx", 318.5200000000001)
    check(b["price"] == "318.52",
          f"a float limit was sent unrounded as {b['price']!r} — Schwab "
          f"refuses a sub-penny equity limit")
    check(b["orderLegCollection"][0]["instrument"]["symbol"] == "FDX",
          "the symbol was not upper-cased")
    check(b["orderLegCollection"][0]["instruction"] == "BUY",
          "the instruction was not upper-cased")
    check(b["duration"] == "DAY" and b["session"] == "NORMAL",
          f"an unexpected duration or session: {b}")

    m = broker._equity_order("SELL", 50, "FDX", None)
    check(m["orderType"] == "MARKET" and "price" not in m,
          f"a market order carries a price: {m}")

    # Round-trip through JSON, because that is how it actually leaves.
    check(json.loads(json.dumps(b))["price"] == "318.52",
          "the price does not survive serialisation as a 2-decimal string")


CASES = [
    ("the guards", case_guards),
    ("the three switches", case_switches),
    ("flatten cancels first", case_flatten_order),
    ("reading the record", case_state),
    ("the rate limiter", case_limiter),
    ("the order body", case_order_body),
]


def main() -> int:
    real_acall, real_hash = broker._acall, broker._account_hash
    real_enabled = config.TRADING_ENABLED
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
        broker._acall, broker._account_hash = real_acall, real_hash
        config.TRADING_ENABLED = real_enabled

    print(f"\nbroker cases: {len(CASES)}, failures: {len(FAILS)}")
    if not FAILS:
        print("  three switches hold, flatten cancels before it closes, the "
              "guards bound the ENDING position, and the reserve is intact")
    return 1 if FAILS else 0


sys.exit(main())
