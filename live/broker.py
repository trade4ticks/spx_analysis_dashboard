"""The broker façade: the switches, the guards, and one adapter behind them.

WHAT THE PANE AND THE ENDPOINTS TALK TO. Nothing above this module names a
broker: `live/main.py` calls the functions here, and here calls whichever
adapter `LIVE_BROKER` selects (live/brokers/__init__.py). Schwab's protocol
lives in live/brokers/schwab.py and nowhere else.

THIS DASHBOARD IS NOT THE SOURCE OF TRUTH, and the whole subsystem is
arranged around that. If the WebSocket drops or the page closes, the orders
are still live at the broker. So every piece of state carries when it was
last confirmed, the page shows that age, and nothing caches an order's
existence — the broker's own list is re-read and replaces whatever was held.

The dangerous version of this code is the one that shows a confident, wrong
list of working orders. A stale list that says it is stale is safe; a
fresh-looking list that is thirty seconds old is not.

FOUR INDEPENDENT SWITCHES have to be on before a single order can leave, and
ALL FOUR ARE CHECKED HERE — above the adapter, once, for every broker:

    LIVE_TRADING_ENABLED    in the environment, default OFF, and the
                            OUTER gate. New order-placing code should
                            not be able to trade because a page element
                            was clicked, and nothing below can escalate
                            past this one.
    the runtime flag        flipped over HTTP without a restart, because
                            a restart drops the upstream socket and every
                            pane's buffer mid-session. Off on every
                            start; needs the shared secret to turn on and
                            nothing at all to turn off.
    the pane's arm toggle   off by default, per pane, and the request
                            carries it — the server refuses a body
                            without it rather than trusting the UI.
    the guards              size, position, notional and distance from
                            the last print, all checked HERE and not
                            only in the browser.

THAT PLACEMENT IS THE POINT OF THE REFACTOR. An adapter is reached only after
the four have passed, so a new broker cannot trade while disarmed or past the
limits by forgetting to ask. The one deliberate exception is `cancel`, which
is never gated on arming: a safety switch that traps an order is not a safety
switch. `flatten` needs trading to be allowed, cancels first, and is ordered
here rather than in the adapter because "cancel before you close" is a
trading rule, not a Schwab one.

RATE LIMITS belong to the adapter — they are a fact about a broker's API, and
DAS's CMD API is a local socket with different ones. What stays here is the
rule they serve: getting flat must never be the call refused for quota, which
is why `cancel`, `flatten` and `reconcile` pass `priority`.
"""
from __future__ import annotations

import hmac
import logging
import time

from live import config
from live import brokers
from live.brokers.base import BrokerError, BrokerIndeterminate, match_placement

log = logging.getLogger("live.broker")

# Re-exported so callers (and checks) keep importing them from one place.
__all__ = [
    "BrokerError", "BrokerIndeterminate", "match_placement",
    "read_orders", "read_positions", "state", "place", "replace", "cancel",
    "flatten", "reconcile", "health", "problems", "aclose",
    "check_guards", "trading_allowed", "trading_state", "set_trading",
]


def _broker():
    return brokers.active()


def _broker_name() -> str:
    """The adapter actually in use, not the configured string.

    The two agree in production. They differ exactly when something has
    swapped the adapter in-process (the checks do), and then the one that
    would send the order is the honest answer.
    """
    try:
        return _broker().name
    except BrokerError:
        return brokers.name()


# ── the guards ──────────────────────────────────────────────────────────────
def check_guards(*, symbol: str, side: str, qty: float, price: float | None,
                 reference: float | None, position_qty: float) -> str | None:
    """Refuse before the call, not after. Returns why, or None.

    ENFORCED HERE and not only in the browser, because the browser is where a
    guard is easiest to bypass by accident — a stale page, a replayed
    request, a hand-typed fetch during debugging. And enforced ABOVE the
    adapter, so every broker is behind the same limits.
    """
    g = config
    if qty is None or qty <= 0 or qty != int(qty):
        return f"quantity {qty!r} is not a positive whole number of shares"
    qty = int(qty)
    if qty > g.MAX_ORDER_SHARES:
        return (f"{qty} shares is over the {g.MAX_ORDER_SHARES}-share "
                f"per-order limit")

    # Where the position ENDS UP, not where it is. A 400-share sell against a
    # 300-share long is a 100-share short, and the limit applies to that too.
    delta = qty if side.upper().startswith("BUY") else -qty
    ending = position_qty + delta
    if abs(ending) > g.MAX_POSITION_SHARES:
        return (f"this would leave {ending:+.0f} shares of {symbol}, over the "
                f"{g.MAX_POSITION_SHARES}-share position limit")

    if price is not None:
        if price <= 0:
            return f"limit price {price} is not a price"
        if qty * price > g.MAX_NOTIONAL:
            return (f"${qty * price:,.0f} is over the ${g.MAX_NOTIONAL:,.0f} "
                    f"notional limit — shares alone do not bound a $900 name")
        # A MISTYPED PRICE is the expensive fat finger, not a mistyped size:
        # 31.85 for 318.50 is a marketable order at a tenth of the price.
        if reference:
            off = abs(price - reference) / reference * 100.0
            if off > g.MAX_LIMIT_DISTANCE_PCT:
                return (f"{price} is {off:.1f}% from the last print "
                        f"({reference}), over the "
                        f"{g.MAX_LIMIT_DISTANCE_PCT}% limit")
    return None


# ── the runtime gate ────────────────────────────────────────────────────────
#
# FOUR SWITCHES NOW, and the new one sits between the other two.
#
#   LIVE_TRADING_ENABLED   the OUTER gate, from the environment. If it is
#                          off, nothing below can turn trading on — the
#                          runtime toggle cannot escalate past it.
#   the runtime flag       this. Flipped over HTTP without a restart,
#                          because restarting drops the upstream socket
#                          and every pane's buffer mid-session, which is
#                          a real cost to pay for changing your mind
#                          about arming.
#   the pane's arm toggle  per pane, in the request body.
#   the guards             size, ending position, notional, distance.
#
# OFF ON EVERY START, unconditionally. A flag that survived a restart would
# be a service that comes back able to trade after a crash nobody watched,
# and "it was on before" is not a reason to be on now.
_runtime_enabled = False
_runtime_changed_at: float | None = None
_runtime_changed_by: str | None = None


def trading_allowed() -> bool:
    """The env gate AND the runtime flag. Both, always."""
    return bool(config.TRADING_ENABLED and _runtime_enabled)


def trading_state() -> dict:
    return {
        # What actually decides whether an order can leave.
        "allowed": trading_allowed(),
        # The outer gate: environment, needs a restart to change.
        "env_enabled": config.TRADING_ENABLED,
        # The runtime flag: this is what the endpoints move.
        "runtime_enabled": _runtime_enabled,
        "changed_at": _runtime_changed_at,
        "changed_by": _runtime_changed_by,
        # Whether ENABLING is possible at all right now, and why not.
        "can_enable": config.TRADING_ENABLED,
        "control_token_set": bool(config.CONTROL_TOKEN),
        "why": _trading_why(),
        # Which broker an armed order would leave through.
        "broker": _broker_name(),
    }


def _trading_why() -> str:
    if not config.TRADING_ENABLED:
        return ("LIVE_TRADING_ENABLED is 0, so trading cannot be turned on "
                "at runtime. It is the outer gate: set it in "
                "/spx_analysis_dashboard/.env and restart spx-live.")
    if not _runtime_enabled:
        return ("trading is allowed by the environment but switched off at "
                "runtime. POST /broker/trading {\"enabled\": true} to arm it; "
                "it is off again after any restart.")
    return "trading is on: the environment allows it and the runtime flag is set."


def set_trading(enabled: bool, *, token: str | None = None,
                who: str | None = None) -> dict:
    """Flip the runtime flag. Returns the new state, or raises.

    DISABLING IS NEVER REFUSED — not for a missing token, not for anything.
    It is the same rule as cancel: a control that can only be reached with
    the right credentials is a control that fails closed at the worst moment,
    and switching trading OFF has no failure mode worth guarding against.

    ENABLING NEEDS THE TOKEN, because this service is reachable from the
    internet through the tunnel and an unauthenticated POST that arms live
    trading is not a thing to leave lying around. Behind cloudflared every
    request appears to come from localhost, so filtering by address would
    prove nothing.
    """
    global _runtime_enabled, _runtime_changed_at, _runtime_changed_by

    if not enabled:
        _runtime_enabled = False
        _runtime_changed_at = time.time()
        _runtime_changed_by = who or "unnamed"
        log.warning("live trading DISABLED at runtime by %s",
                    _runtime_changed_by)
        return trading_state()

    if not config.TRADING_ENABLED:
        raise BrokerError(
            "LIVE_TRADING_ENABLED is 0. The runtime toggle cannot enable "
            "trading past the environment gate — set it in "
            "/spx_analysis_dashboard/.env and restart spx-live.")
    if not config.CONTROL_TOKEN:
        raise BrokerError(
            "LIVE_CONTROL_TOKEN is not set, so there is no way to tell an "
            "authorised caller from any other. Set it in the environment "
            "before enabling trading over HTTP. Disabling never needs it.")
    if not token or not hmac.compare_digest(token, config.CONTROL_TOKEN):
        raise BrokerError("the control token is missing or wrong.")

    _runtime_enabled = True
    _runtime_changed_at = time.time()
    _runtime_changed_by = who or "unnamed"
    log.warning("live trading ENABLED at runtime by %s", _runtime_changed_by)
    return trading_state()


def _armed_check(armed: bool) -> str | None:
    if not config.TRADING_ENABLED or not _runtime_enabled:
        return _trading_why()
    if not armed:
        return ("this pane is not armed — the request did not carry it. "
                "Arming is per pane and off by default.")
    return None


# ── reading the record ──────────────────────────────────────────────────────
async def read_orders(symbols: list[str] | None = None,
                      priority: bool = False) -> dict:
    """Working and recent orders, as the adapter normalises them."""
    return await _broker().read_orders(symbols, priority)


async def read_positions(symbols: list[str] | None = None,
                         priority: bool = False) -> dict:
    """Positions and whatever account facts the broker exposes."""
    return await _broker().read_positions(symbols, priority)


async def state(symbols: list[str] | None = None,
                priority: bool = False) -> dict:
    """Both reads as one consistent answer, `as_of` the older of the two."""
    return await _broker().state(symbols, priority)


async def reconcile(*, symbol: str, side: str, qty: float,
                    price: float | None, sent_at: float) -> dict:
    """Did a placement whose reply never arrived actually land?

    PRIORITY inside the adapter: this is a caller that does not know its own
    position, and being refused for quota is the one thing that must not
    happen to it.
    """
    return await _broker().reconcile(symbol=symbol, side=side, qty=qty,
                                     price=price, sent_at=sent_at)


# ── orders ──────────────────────────────────────────────────────────────────
async def place(*, symbol: str, side: str, qty: int, price: float | None,
                armed: bool, reference: float | None,
                position_qty: float, route: str | None = None) -> dict:
    """Arm, guard, then send. In that order, for every broker.

    `route` PASSES STRAIGHT THROUGH and is not policy. Which venue an order
    goes to is a trading choice made per order on the page, not a safety
    limit — the guards bound size, ending position, notional and distance,
    and none of those change with the venue. The adapter decides what an
    unknown route means, because only it knows which ones exist.
    """
    why = _armed_check(armed)
    if why:
        raise BrokerError(why)
    why = check_guards(symbol=symbol, side=side, qty=qty, price=price,
                       reference=reference, position_qty=position_qty)
    if why:
        raise BrokerError(f"refused by the guards: {why}")
    return await _broker().place(symbol=symbol, side=side, qty=qty,
                                 price=price, route=route)


async def replace(*, order_id: str, symbol: str, side: str, qty: int,
                  price: float, armed: bool, reference: float | None,
                  position_qty: float, route: str | None = None) -> dict:
    """Reprice. Same two checks as a placement — a nudge is an order.

    The guards run on the REPRICED order, not the original: repricing is
    exactly where a fat-fingered price arrives, and the distance guard is the
    one that catches it.
    """
    why = _armed_check(armed)
    if why:
        raise BrokerError(why)
    why = check_guards(symbol=symbol, side=side, qty=qty, price=price,
                       reference=reference, position_qty=position_qty)
    if why:
        raise BrokerError(f"refused by the guards: {why}")
    return await _broker().replace(order_id=order_id, symbol=symbol, side=side,
                                   qty=qty, price=price, route=route)


async def cancel(*, order_id: str) -> dict:
    """NOT gated on arming, deliberately.

    Cancelling is how a mistake is undone. A disarmed pane that cannot cancel
    would be a pane whose safety switch traps an order, and a cancel refused
    for quota is the specific failure an adapter's reserve exists to stop.
    """
    return await _broker().cancel(order_id=order_id)


async def flatten(*, symbol: str, armed: bool) -> dict:
    """Cancel everything working in the name, then close at market.

    GATED ON TRADING BEING ALLOWED but not on the pane's arm toggle: the
    original behaviour, kept. A flatten sends a market order, so the
    environment gate and the runtime flag still apply.

    ORDER MATTERS, and it is checked here rather than trusted to each
    adapter: closing while a working order rests on the same symbol can leave
    that order to open a fresh position in the opposite direction the moment
    the flatten fills.
    """
    if not trading_allowed():
        raise BrokerError(_trading_why())
    return await _broker().flatten(symbol=symbol)


# ── health ──────────────────────────────────────────────────────────────────
def health() -> dict:
    """The adapter's own facts, plus the switches, which are ours.

    `trading_enabled` stays the flat field the page already reads: what it
    means is "can an order leave right now", which is the combined answer and
    not the env gate alone.
    """
    h = dict(_broker().health())
    h["trading"] = trading_state()
    h["trading_enabled"] = trading_allowed()
    h["broker"] = _broker_name()
    h["guards"] = {
        "max_order_shares": config.MAX_ORDER_SHARES,
        "max_position_shares": config.MAX_POSITION_SHARES,
        "max_notional": config.MAX_NOTIONAL,
        "max_limit_distance_pct": config.MAX_LIMIT_DISTANCE_PCT,
    }
    # THE SHAPE IS GUARANTEED HERE so the page can ask one question —
    # "does this broker route?" — of whichever adapter is loaded. An
    # adapter that says nothing is one that does not route, which is the
    # answer that offers no control rather than a control that does nothing.
    h.setdefault("routing", {
        "supported": False, "default": None, "choices": [],
        "why": "this broker's API has no venue selection.",
    })
    return h


def problems() -> list[str]:
    """Why an order could not leave right now, in words a person can act on."""
    try:
        return list(_broker().problems())
    except BrokerError as exc:
        # An unresolvable LIVE_BROKER is itself the problem worth reporting,
        # and health() must not raise on the way to saying so.
        return [str(exc)]


async def aclose() -> None:
    try:
        await _broker().aclose()
    except BrokerError:
        return
