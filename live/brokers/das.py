"""DAS Trader CMD API adapter: one socket, pushed state, a token per order.

THE SECOND BROKER, written against live/brokers/base.Broker. Everything in
this file is about speaking to DAS. The things that are decisions about
TRADING rather than about DAS -- the four switches, the guards, that a
flatten needs trading allowed -- live in the façade and are checked before
any method here is reached. This file must not consult them, and
`check_broker.py` scans it to be sure.

IT IS A SOCKET, NOT AN API. That is the whole structural difference from
Schwab and it shapes the rest:

  * ONE PERSISTENT TCP CONNECTION to DAS Trader Pro, plain-text lines
    terminated CRLF, reached over Tailscale from this box. DAS MUST BE
    RUNNING AND LOGGED IN for the socket to exist -- closing the platform
    drops it -- so the link's own state is a first-class thing this file
    reports rather than something that shows up as a mysterious failure.
  * STATE IS PUSHED. %ORDER, %OrderAct, %TRADE and %POS arrive as they
    happen. There is no poll, so nothing here measures a round trip to
    decide how fresh it is; `as_of` is WHEN THE SOCKET WAS LAST PROVEN
    ALIVE, which is what the heartbeat below exists to establish.
  * THE TOKEN IS A REAL CLIENT ORDER ID. We choose it on NEWORDER and it
    comes back on %ORDER and %OrderAct. So `reconcile` matches on it
    EXACTLY and overrides base.match_placement's shape heuristic with its
    documented ambiguity. Schwab rejects a client tag outright (see the long
    block in schwab.py); this is the thing that API could not do.

WHAT THIS ADAPTER DELIBERATELY DOES NOT DO:

  * NO MARKET DATA. No SB, no Level 1, Level 2, time & sales or charts. The
    tape and the scan run on Polygon, there is no depth of book in this API,
    and a second quote source drawn on the same chart would be two prices
    under one label. `check_das.py` fails if a subscription command appears
    in this file.
  * NO UNREALISED P&L FROM %POS. The manual is explicit that Unrealized is a
    snapshot from when the position was sent, not a live number. The pane
    computes open P&L from the last print on the tape, which is both current
    and already on screen, so the stale field is dropped rather than shown as
    if it were now.
  * NO REFUSING TO ANSWER WHEN THE LINK IS QUIET. Once the record has been
    seen, a read is served from the pushed cache even if the socket has gone
    silent, with `as_of` frozen at the last confirmation and the link's state
    in health(). DAS Trader is on screen beside this page showing the same
    orders, so a divergence is visible immediately, and blocking the pane on
    a quiet socket would cost more than it protects. What is NOT done is
    inventing a record: before the first snapshot has arrived this process
    has never seen the account, and a read then REFUSES rather than
    reporting an empty order list that would read as "nothing working".

RATE LIMITS are the published defaults (manual, "Limitations"), adjustable
by arrangement with DAS: NEWORDER 50/second, CANCEL 100/minute, REPLACE
100/minute. One person clicking comes nowhere near them. They are here for
the same reason Schwab's are: so the call that must never be refused for
quota -- the one that gets flat -- has a reserve nothing else may spend.
"""
from __future__ import annotations

import asyncio
import logging
import random
import time
from collections import deque
from datetime import datetime, timedelta

from live import config
from live.brokers import base
from live.brokers.base import BrokerError, BrokerIndeterminate

log = logging.getLogger("live.broker")

LINE_END = "\r\n"

# ── which statuses mean "this order is still live at DAS" ───────────────────
#
# THE ASYMMETRY IS THE SAME ONE AS SCHWAB'S, and so is the conclusion. A
# status in here puts the order in `working`: it draws on the ladder,
# primaryOrder() can pick it up, it can be repriced, cancelled and flattened.
# A status outside it goes to `recent`, where nothing reaches it.
#
# So calling a dead order live leaves an inert marker on the chart, and
# calling a LIVE order dead hides an order that is working at the broker from
# the screen meant to show it -- and from cancel and flatten, which are how
# you get out. Anything that might still be live belongs in here, and
# `working_from_status` resolves an UNRECOGNISED status to True for exactly
# that reason.
#
# Read off the %ORDER status list in the manual (CMD_API_Manual.pdf, %ORDER):
#   Closed    this order is closed                   terminal
#   Hold      open, but NOT sent to the exchange     LIVE: DAS is holding it
#                                                    and it can be cancelled
#   Sending   sending to the exchange                LIVE
#   Accepted  accepted by the exchange               LIVE
#   Canceled  has been canceled                      terminal
#   Rejected  is rejected                            terminal
#   Executed  fully executed                         terminal
#   Partial   partial filled                         LIVE: the balance works
#   Triggered an auto-route order, fully OR          LIVE: "or partially" is
#             partially executed                     the reason it is here
WORKING_STATES = {"HOLD", "SENDING", "ACCEPTED", "PARTIAL", "TRIGGERED"}

# Terminal: over, nothing can be done to it. Kept explicitly so the two sets
# can be checked against each other -- a status in neither is one nobody has
# thought about, and check_das fails on it.
#
# CANCELLED with two Ls is not in the manual. It is here because one spelling
# of one status is not a thing to bet the cancel path on.
TERMINAL_STATES = {"CLOSED", "CANCELED", "CANCELLED", "REJECTED", "EXECUTED"}


def working_from_status(status: str | None) -> bool:
    """Is this order still live at DAS? An unknown answer is YES. See above."""
    s = (status or "").strip().upper()
    if not s:
        return True
    return s not in TERMINAL_STATES


# ── %OrderAct action types ──────────────────────────────────────────────────
#
# THE TWO THAT ARE NOT FAILURES. TimeOut and Send_Rej say the order's fate is
# UNKNOWN to us, and the difference between that and a refusal is the whole
# point of BrokerIndeterminate: a determinate failure may be retried and this
# one must never be, because a retry is how a timeout becomes a double
# position.
#
# Send_Rej is grouped with TimeOut deliberately, and it is the safe
# direction: the send was rejected somewhere between here and the exchange,
# the notes that come with it are free text, and treating it as "nothing
# happened" would licence an automatic retry. Being unresolved costs a read
# of the record; the other mistake costs a position.
ACT_INDETERMINATE = {"TIMEOUT", "SEND_REJ"}

# DETERMINATE refusals of a cancel or a replace: the order is untouched and
# the pane can say so plainly. Never applied to a placement -- see above.
ACT_REFUSED = {"CANCELREJ", "REPLACEREJ"}

# Acknowledgements that settle each kind of command.
ACT_PLACED = {"SENDING", "ACCEPT", "ACCEPTED", "EXECUTE", "TRIGGERED"}
ACT_CANCELLED = {"CANCELING", "CANCELLING", "CANCELED", "CANCELLED", "CLOSE"}
ACT_REPLACED = {"REPLACED", "REPLACING"}

# Montage route names rather than API routes. The manual: "If you want to use
# LIMIT, MARKET or STOP, route needs to be set as SMAT."
MONTAGE_ROUTES = {"LIMIT", "MARKET", "STOP"}

# The C int range the manual gives for Token.
MIN_INT, MAX_INT = -2147483648, 2147483647


# ── the rate limiter ────────────────────────────────────────────────────────
class Bucket:
    """One published limit, with a reserve that only getting flat may spend."""

    def __init__(self, name: str, limit: int, window_s: float, reserve: int):
        self.name = name
        self.limit = limit
        self.window_s = window_s
        self.reserve = max(0, min(reserve, limit))
        self.calls: deque[float] = deque()
        self.refusals = 0

    def _prune(self, now: float) -> None:
        while self.calls and self.calls[0] < now - self.window_s:
            self.calls.popleft()

    def take(self, priority: bool = False) -> str | None:
        """Spend one command. Returns why it was refused, or None."""
        now = time.time()
        self._prune(now)
        ceiling = self.limit if priority else self.limit - self.reserve
        if len(self.calls) >= ceiling:
            self.refusals += 1
            wait = max(0.0, self.window_s - (now - self.calls[0]))
            per = "second" if self.window_s <= 1.0 else "minute"
            return (f"at {len(self.calls)}/{self.limit} {self.name} commands "
                    f"this {per}"
                    + ("" if priority else
                       f" ({self.reserve} held back so a cancel or a flatten "
                       f"always has quota)")
                    + f"; {wait:.1f}s until one frees")
        self.calls.append(now)
        return None

    def state(self) -> dict:
        now = time.time()
        self._prune(now)
        return {"limit": self.limit, "window_s": self.window_s,
                "reserve": self.reserve, "used": len(self.calls),
                "refusals": self.refusals}


class RateLimiter:
    """DAS's three order limits, each in its own window.

    NOT ONE SHARED BUDGET like Schwab's 120 a minute: the CMD API counts
    NEWORDER per SECOND and CANCEL and REPLACE per MINUTE, so a single bucket
    would either throttle placements that are fine or let cancels through a
    limit that is real.
    """

    def __init__(self):
        self.buckets = {
            "new": Bucket("NEWORDER", config.DAS_ORDERS_PER_SEC, 1.0,
                          config.DAS_ORDER_RESERVE),
            "cancel": Bucket("CANCEL", config.DAS_CANCELS_PER_MIN, 60.0,
                             config.DAS_CANCEL_RESERVE),
            "replace": Bucket("REPLACE", config.DAS_REPLACES_PER_MIN, 60.0,
                              config.DAS_REPLACE_RESERVE),
        }

    def take(self, kind: str, priority: bool = False) -> str | None:
        b = self.buckets.get(kind)
        return b.take(priority=priority) if b else None

    def state(self) -> dict:
        """The shape the site bar reads, with every bucket underneath it.

        `per_min` and `used` are the CANCEL bucket's: it is the tightest
        per-minute limit and the only one a person could actually reach.
        `label` is what the readout shows, because "50/s new · 100/min
        cancel" is the honest summary and no single number is.
        """
        detail = {k: b.state() for k, b in self.buckets.items()}
        cancel = self.buckets["cancel"]
        return {
            "label": (f"{self.buckets['new'].limit}/s new · "
                      f"{cancel.limit}/min cancel"),
            "per_min": cancel.limit,
            "reserve": cancel.reserve,
            "used": detail["cancel"]["used"],
            "available": max(0, cancel.limit - cancel.reserve
                             - detail["cancel"]["used"]),
            "blocked_for_s": 0.0,
            "refusals": sum(d["refusals"] for d in detail.values()),
            # No 429 equivalent on a socket. Kept so one readout serves both
            # brokers rather than branching on which one is loaded.
            "n_429": 0,
            "buckets": detail,
        }


LIMITER = RateLimiter()


# ── parsing ─────────────────────────────────────────────────────────────────
#
# EVERY PARSER HERE IS TOLERANT OF LENGTH, and that is not defensiveness for
# its own sake. The manual documents two shapes for %ORDER -- an older one
# without token, cxlqty, orderSrc, TIF and Pref, and the current 18-field one
# -- and its own complex-order example shows a 15-field line. %TRADE is
# documented with Liq, EcnFee and PL and its example arrives without them. A
# parser that indexed field 17 unconditionally would take the whole order
# list down on a line that is perfectly legal.
def _at(parts: list[str], i: int) -> str | None:
    return parts[i] if 0 <= i < len(parts) else None


def _f(x) -> float | None:
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _i(x) -> int | None:
    v = _f(x)
    return int(v) if v is not None else None


# DAS sides are B / S / SS (and the options forms BO/BC/SO/SC). The pane
# tests `side.startsWith('BUY')` in several places and Schwab hands it
# BUY/SELL, so the normalised side has to speak that language -- a raw "B"
# here would draw every buy as a sell.
SIDE_IN = {"B": "BUY", "S": "SELL", "SS": "SELL_SHORT",
           "BO": "BUY_TO_OPEN", "BC": "BUY_TO_CLOSE",
           "SO": "SELL_TO_OPEN", "SC": "SELL_TO_CLOSE",
           "BUY": "BUY", "SELL": "SELL", "SHRT": "SELL_SHORT",
           "SHORT": "SELL_SHORT"}


def norm_side(side: str | None) -> str | None:
    if not side:
        return None
    return SIDE_IN.get(side.strip().upper(), side.strip().upper())


def das_side(side: str | None) -> str:
    """The wire form. BUY -> B, SELL -> S, a short -> SS.

    The order server corrects S against SS by checking the position (the
    manual says so), so a plain S is safe for a sell that turns out to be a
    short. Sending SS for something that is not is the mistake worth avoiding.
    """
    s = (side or "").strip().upper()
    if s.startswith("BUY"):
        return "B"
    if s in ("SELL_SHORT", "SHORT", "SS"):
        return "SS"
    return "S"


TYPE_IN = {"L": "LIMIT", "LMT": "LIMIT", "M": "MARKET", "MKT": "MARKET"}


def norm_type(t: str | None) -> str | None:
    if not t:
        return None
    u = t.strip().upper()
    return TYPE_IN.get(u, u)


def iso_stamp(hhmmss: str | None, *, now: float | None = None) -> str | None:
    """DAS's HH:MM:SS as an ISO stamp with an offset.

    THE ZONE HAS TO COME FROM SOMEWHERE. DAS stamps orders and trades with a
    bare time in the PLATFORM's local zone -- the Windows machine's, not this
    box's, which runs UTC. `base.entered_epoch` parses ISO-with-offset and
    nothing else, and `recent` is sorted by it, so a naive stamp would sort
    the recent list by a number that is hours out.

    The date is today's IN THAT ZONE, with one correction: a stamp that lands
    more than an hour in the future is yesterday's (read at 00:01, an order
    entered 23:59 is not tomorrow).
    """
    if not hhmmss:
        return None
    txt = hhmmss.strip()
    # %POS carries a full YYYY/MM/DD-HH:MM:SS; orders and trades do not.
    if "-" in txt and "/" in txt:
        try:
            d = datetime.strptime(txt, "%Y/%m/%d-%H:%M:%S")
        except ValueError:
            return None
        return _with_zone(d).isoformat()
    try:
        t = datetime.strptime(txt, "%H:%M:%S")
    except ValueError:
        return None
    ref = _now_local(now)
    d = ref.replace(hour=t.hour, minute=t.minute, second=t.second,
                    microsecond=0, tzinfo=None)
    if (d - ref.replace(tzinfo=None)).total_seconds() > 3600:
        d -= timedelta(days=1)
    return _with_zone(d).isoformat()


def _zone():
    try:
        from zoneinfo import ZoneInfo
        return ZoneInfo(config.DAS_TZ)
    except Exception:                                       # noqa: BLE001
        # A box without tzdata must not lose the order list over a timestamp.
        return None


def _now_local(now: float | None = None):
    ts = now if now is not None else time.time()
    z = _zone()
    if z is None:
        return datetime.fromtimestamp(ts).astimezone()
    return datetime.fromtimestamp(ts, z)


def _with_zone(d: datetime) -> datetime:
    z = _zone()
    if z is not None:
        return d.replace(tzinfo=z)
    return d.astimezone()


def parse_order(line: str, *, mine=None) -> dict | None:
    """`%ORDER id token symb b/s type qty lvqty cxlqty price route status
    time origoid account trader orderSrc TIF Pref` -> the normalised order.

    `mine` is the set of tokens this process minted, which is what makes
    `from_api` a PER-ORDER fact here. Schwab's equivalent is an
    account-level stamp that can only say "not thinkorswim"; a token says
    "this exact order came from this page".
    """
    parts = line.split()
    if len(parts) < 11:
        return None

    # WHICH LAYOUT IS THIS. The manual carries two: the current one with
    # `token` second and `cxlqty` after `lvqty`, and an older one with
    # neither. Every field after the id shifts between them, so guessing
    # wrong does not produce a slightly wrong order -- it reads the SIDE as
    # the symbol and the symbol as an order id. The side is what tells them
    # apart: it is B/S/SS in a fixed place in each, and a token never is.
    idx = dict(token=2, symbol=3, side=4, type=5, qty=6, lvqty=7, cxl=8,
               price=9, route=10, status=11, time=12, orig=13, account=14,
               trader=15, src=16, tif=17, pref=18)
    if (_at(parts, 4) or "").upper() not in SIDE_IN \
            and (_at(parts, 3) or "").upper() in SIDE_IN:
        idx = dict(token=None, symbol=2, side=3, type=4, qty=5, lvqty=6,
                   cxl=None, price=7, route=8, status=9, time=10, orig=11,
                   account=12, trader=13, src=None, tif=None, pref=None)

    def at(key):
        i = idx.get(key)
        return _at(parts, i) if i is not None else None

    token = _i(at("token"))
    qty = _f(at("qty"))
    lvqty = _f(at("lvqty"))
    cxlqty = _f(at("cxl"))
    status = at("status")
    src = (at("src") or "").upper()
    filled = None
    if qty is not None and lvqty is not None:
        filled = qty - lvqty - (cxlqty or 0.0)
        if filled < 0:
            filled = None
    return {
        "order_id": str(_at(parts, 1) or ""),
        "token": token,
        "symbol": (at("symbol") or "").upper() or None,
        "side": norm_side(at("side")),
        "type": norm_type(at("type")),
        "qty": qty,
        "filled": filled,
        "cancelled_qty": cxlqty,
        "price": _f(at("price")),
        "route": at("route"),
        "status": status,
        "working": working_from_status(status),
        "entered": iso_stamp(at("time")),
        "orig_order_id": at("orig"),
        "account": at("account"),
        "trader": at("trader"),
        "order_src": at("src"),
        "tif": at("tif"),
        "pref": at("pref"),
        # WHERE THE ORDER CAME FROM. The token is ours and exact; orderSrc
        # CMDAPI is the fallback that survives a restart of this service,
        # when the minted set is empty but the orders it placed are not gone.
        "tag": str(token) if token is not None else None,
        "from_api": bool((mine and token in mine) or src == "CMDAPI"),
        "fills": [],
    }


def parse_order_act(line: str) -> dict | None:
    """`%OrderAct id ActionType B/S symbol qty price route time notes token`.

    THE TOKEN IS LAST AND `notes` IS OPTIONAL, which the manual's own example
    shows: `%OrderAct 56 Accept Buy +MSFT^GCI400 1 116.35 COMP 20:14:10 1`
    has nine fields after the verb and that trailing 1 is the token, not a
    note. So the token is taken from the END, and only when it parses as an
    integer -- a note that happens to be a bare number is the one case this
    cannot tell apart, and `place` resolves on the order id anyway.
    """
    parts = line.split()
    if len(parts) < 3:
        return None
    token = None
    tail = _at(parts, len(parts) - 1)
    if len(parts) >= 10 and tail is not None:
        token = _i(tail)
        if token is not None and not (MIN_INT <= token <= MAX_INT):
            token = None
    notes = None
    if len(parts) >= 11:
        notes = " ".join(parts[9:len(parts) - 1])
    return {
        "order_id": str(_at(parts, 1) or ""),
        "action": (_at(parts, 2) or "").strip(),
        "side": norm_side(_at(parts, 3)),
        "symbol": (_at(parts, 4) or "").upper() or None,
        "qty": _f(_at(parts, 5)),
        "price": _f(_at(parts, 6)),
        "route": _at(parts, 7),
        "time": _at(parts, 8),
        "notes": notes,
        "token": token,
    }


def parse_trade(line: str) -> dict | None:
    """`%TRADE id symb b/s qty price route time orderid Liq EcnFee PL`.

    Liq IS WORTH KEEPING: `+` added liquidity, `-` removed it, `R` routed
    out, and the Cobra statement only aggregates ECN fees per symbol-day, so
    this is the only per-fill record of which it was. It rides along on the
    normalised fill, where the pane can ignore it and the log cannot lose it.
    """
    parts = line.split()
    if len(parts) < 9:
        return None
    return {
        "trade_id": str(_at(parts, 1) or ""),
        "symbol": (_at(parts, 2) or "").upper() or None,
        "side": norm_side(_at(parts, 3)),
        "qty": _f(_at(parts, 4)),
        "price": _f(_at(parts, 5)),
        "route": _at(parts, 6),
        "t": iso_stamp(_at(parts, 7)),
        "order_id": str(_at(parts, 8) or ""),
        "liq": _at(parts, 9),
        "ecn_fee": _f(_at(parts, 10)),
        "pl": _f(_at(parts, 11)),
    }


def parse_pos(line: str) -> dict | None:
    """`%POS Symbol Type Quantity AvgCost InitQuantity InitPrice Realized
    CreateTime Unrealized` -> the normalised position.

    TYPE 3 IS A SHORT and the quantity that comes with it is stated as a
    size, not a signed number (the manual's example is `%POS AAPL 3 100
    ...`). The normalised shape is signed -- negative is short, everywhere
    else in this system -- so a type-3 row is negated. `-abs()` rather than
    `-`, so a build that already signs it is not turned back into a long.

    `day_pl` IS REALIZED, and it is not the same number Schwab puts there
    (currentDayProfitLoss includes the open leg). Unrealized is deliberately
    not used: the manual says it is a snapshot from when the position was
    sent, and a stale P&L shown as a current one is the kind of number that
    gets acted on.
    """
    parts = line.split()
    if len(parts) < 4:
        return None
    ptype = _i(_at(parts, 2))
    qty = _f(_at(parts, 3))
    if qty is not None and ptype == 3:
        qty = -abs(qty)
    return {
        "symbol": (_at(parts, 1) or "").upper() or None,
        "qty": qty,
        "avg": _f(_at(parts, 4)),
        "day_pl": _f(_at(parts, 7)),
        "pos_type": ptype,
        "created": iso_stamp(_at(parts, 8)),
    }


# ── building the commands ───────────────────────────────────────────────────
def norm_route(route: str | None, *, known: set[str] | None = None) -> str:
    """The route as the API wants it: a base name, no montage suffix.

    TWO RULES FROM THE MANUAL, both of which silently reject an order if
    they are got wrong:

      * the montage's dropdown carries ARCAL and ARCAM -- L for limit, M for
        market -- and the API takes the BASE name, ARCA. The suffix is
        stripped only when the stripped name is one DAS has told us about
        (GET RouteStatus, or the configured fallback), because plenty of
        real route names end in L or M and turning INET into INE would be
        a worse failure than passing a suffix through.
      * LIMIT, MARKET and STOP are not routes an order may name; those order
        types require SMAT, which is DAS's smart router.
    """
    r = (route or "").strip().upper() or config.DAS_ROUTE
    if r in MONTAGE_ROUTES:
        log.info("route %s is a montage name; sending SMAT, which is what "
                 "that order type requires", r)
        return "SMAT"
    if len(r) > 2 and r[-1] in ("L", "M"):
        pool = known if known else set(config.DAS_ROUTES)
        if r[:-1] in pool and r not in pool:
            return r[:-1]
    return r


def fmt_price(price: float) -> str:
    """Two decimals, or four under a dollar.

    Reg NMS 612: a quote at or above $1.00 moves in pennies, below it in
    hundredths of a penny. A float that stringifies to 318.52000000000004 is
    a sub-penny limit and gets refused; so is 0.8532 rounded to 0.85 in a
    name where the increment is 0.0001, in the other direction.
    """
    return f"{price:.4f}" if abs(price) < 1.0 else f"{price:.2f}"


def build_neworder(*, token: int, side: str, symbol: str, qty: int,
                   price: float | None, route: str | None = None,
                   tif: str | None = None, post_only: bool = False,
                   not_route_out: bool = False, display: int | None = None,
                   minume: str | None = None, pref: str | None = None,
                   known_routes: set[str] | None = None) -> str:
    """NEWORDER, exactly as the manual spells it.

        NEWORDER <token> <B/S/SS> <symbol> <route> <shares> <price> TIF=DAY+
        NEWORDER <token> <B/S/SS> <symbol> <route> <shares> MKT TIF=DAY

    MORE PARAMETERS ARE BUILT HERE THAN THE INTERFACE CAN REACH, on purpose.
    PostOnly and NotRouteOut are the reason for moving to a routing broker in
    the first place and TIF is a day-to-day choice; all three are one string
    each on this line, and the expensive part of adding them later is the
    wiring above -- interface, façade, endpoint, control. Writing and
    gate-covering the protocol half now means that later change is additive
    and cannot be got wrong quietly. They are not reachable yet: nothing
    above passes them, and `check_das` asserts what each one puts on the wire.
    """
    if not (MIN_INT <= int(token) <= MAX_INT):
        raise BrokerError(
            f"token {token} is outside the C int range the CMD API accepts "
            f"[{MIN_INT}, {MAX_INT}]")
    if qty is None or int(qty) <= 0:
        raise BrokerError(f"{qty!r} is not a number of shares")
    r = norm_route(route, known=known_routes)
    px = "MKT" if price is None else fmt_price(float(price))
    out = [f"NEWORDER {int(token)} {das_side(side)} {symbol.upper()} {r} "
           f"{int(qty)} {px}"]
    # The manual's own default. DAY+ includes the pre/post session, which is
    # what a resting limit here wants; a market order is a DAY order.
    out.append(f"TIF={tif}" if tif
               else ("TIF=DAY" if price is None else "TIF=DAY+"))
    if display is not None:
        out.append(f"Display={int(display)}")
    if minume:
        out.append(f"Minume={minume}")
    if pref:
        out.append(f"Pref={pref.strip().upper()}")
    if post_only:
        out.append("PostOnly")
    if not_route_out:
        out.append("NotRouteOut")
    return " ".join(out)


def build_replace(*, order_id: str, qty: int, price: float | None,
                  tif: str | None = None) -> str:
    """REPLACE orderid share price -- a genuine modify, in one call.

    NOT cancel-then-place: that loses the queue position and leaves a window
    with no order resting at all, which is the whole reason the interface
    asks for a native replace.

    NO ROUTE FIELD EXISTS on REPLACE. That is why `replace` below refuses a
    route that differs from the one the order is already resting on, rather
    than accepting it and repricing the order where it already was.
    """
    px = "MKT" if price is None else fmt_price(float(price))
    out = [f"REPLACE {order_id} {int(qty)} {px}"]
    if tif:
        out.append(f"TIF={tif}")
    return " ".join(out)


# ── the link ────────────────────────────────────────────────────────────────
class DasLink:
    """The socket, the login, the pushed cache, and the proof it is alive.

    ONE OBJECT OWNS THE CONNECTION so that everything which depends on its
    state -- what the cache holds, when it was last confirmed, which commands
    are still waiting for an acknowledgement -- moves together. It reconnects
    on its own, because DAS being closed and reopened is a normal Tuesday and
    not an error the page should have to be clicked through.

    WHAT `confirmed_at` MEANS, and it is the answer to "how do I know this
    isn't a dead socket showing a confident list": it is the moment the last
    line arrived from DAS. Every push updates it, and so does the reply to
    the ECHO heartbeat, which is sent every few seconds precisely so a quiet
    name cannot look the same as a dead link. `as_of` on every read is this
    number, so the age the page shows is the age of the last PROOF, never the
    age of the last request.
    """

    def __init__(self):
        self.reader = None
        self.writer = None
        self.task: asyncio.Task | None = None
        self.beat: asyncio.Task | None = None
        self.lock = asyncio.Lock()
        self.connected = False
        self.logged_in = False
        self.connected_at: float | None = None
        self.confirmed_at: float = 0.0
        self.last_error: str | None = None
        self.last_error_why: str | None = None
        self.n_connects = 0
        self.n_drops = 0
        self._retry_at = 0.0
        self._backoff = 1.0

        # The record, as DAS has pushed it.
        self.orders: dict[str, dict] = {}
        self.positions: dict[str, dict] = {}
        self.fills: dict[str, list[dict]] = {}
        self.order_snapshot = False
        self.pos_snapshot = False
        self._staging_orders: dict[str, dict] | None = None
        self._staging_pos: dict[str, dict] | None = None
        self._staging_fills: dict[str, list[dict]] | None = None

        # What DAS says about itself.
        self.routes: dict[str, bool] = {}
        self.order_server: str | None = None

        # Tokens this process minted, and what each was for. This is what
        # makes reconcile exact; see `reconcile`.
        self.minted: dict[int, dict] = {}
        self._token = random.randint(1_000_000_000, 1_900_000_000)

        # Commands waiting for their acknowledgement.
        self._by_token: dict[int, asyncio.Future] = {}
        self._by_order: dict[str, asyncio.Future] = {}
        self._login_wait: asyncio.Future | None = None

    # ── connecting ──────────────────────────────────────────────────────
    async def ensure(self) -> None:
        """Connected and logged in, or raise saying why not.

        LAZY, not at startup: a box running the tape with DAS configured but
        the platform closed should not spend its life retrying a connection
        nobody asked for. The first read or order opens it.

        AND NOT ON EVERY CALL WHEN IT IS DOWN. The pane reads twice a second
        across its panes; with DAS closed, a connect attempt per read would
        queue five-second timeouts behind each other and make the page feel
        broken for a reason that has nothing to do with the page. Failed
        attempts back off to half a minute, and a read inside the cooldown
        is answered from the cache (see `_ready`) rather than made to wait.
        """
        if self.connected and self.logged_in:
            return
        now = time.time()
        if now < self._retry_at:
            raise BrokerError(
                self.last_error_why
                or f"not connected to the DAS CMD API at {config.DAS_HOST}:"
                   f"{config.DAS_PORT}.")
        async with self.lock:
            if self.connected and self.logged_in:
                return
            if time.time() < self._retry_at:
                raise BrokerError(self.last_error_why or "not connected.")
            try:
                await self._connect()
            except BrokerError as exc:
                self._backoff = min(30.0, max(1.0, self._backoff * 2))
                self._retry_at = time.time() + self._backoff
                self.last_error_why = str(exc)
                raise
            self._backoff = 1.0
            self._retry_at = 0.0
            self.last_error_why = None

    async def _connect(self) -> None:
        if not config.DAS_HOST:
            raise BrokerError(
                "LIVE_DAS_HOST is not set, so there is nowhere to connect. "
                "It is the Tailscale address of the machine running DAS "
                "Trader Pro.")
        if not (config.DAS_TRADER and config.DAS_PASSWORD
                and config.DAS_ACCOUNT):
            raise BrokerError(
                "LIVE_DAS_TRADER / LIVE_DAS_PASSWORD / LIVE_DAS_ACCOUNT are "
                "not all set, so the CMD API cannot be logged in to.")
        await self._teardown()
        try:
            self.reader, self.writer = await asyncio.wait_for(
                asyncio.open_connection(config.DAS_HOST, config.DAS_PORT),
                timeout=config.DAS_CONNECT_TIMEOUT_S)
        except Exception as exc:                            # noqa: BLE001
            self.last_error = f"{type(exc).__name__}: {exc}"
            # DETERMINATE: nothing was sent anywhere. The message names the
            # one cause that is nearly always it.
            raise BrokerError(
                f"cannot reach the DAS CMD API at {config.DAS_HOST}:"
                f"{config.DAS_PORT} ({self.last_error}). DAS Trader Pro has "
                f"to be running and logged in on that machine for the socket "
                f"to exist at all.") from exc

        self.connected = True
        self.connected_at = time.time()
        self.confirmed_at = time.time()
        self.n_connects += 1
        self.order_snapshot = self.pos_snapshot = False
        self.task = asyncio.create_task(self._read_loop())
        loop = asyncio.get_running_loop()
        self._login_wait = loop.create_future()
        await self.write_line(f"LOGIN {config.DAS_TRADER} {config.DAS_PASSWORD} "
                          f"{config.DAS_ACCOUNT} 0")
        try:
            await asyncio.wait_for(self._login_wait,
                                   timeout=config.DAS_CONNECT_TIMEOUT_S)
        except asyncio.TimeoutError as exc:
            await self._teardown()
            raise BrokerError(
                f"DAS accepted the connection but did not answer the login "
                f"within {config.DAS_CONNECT_TIMEOUT_S}s. Check that the "
                f"trader, password and account in LIVE_DAS_* are the ones "
                f"that platform is logged in with.") from exc
        except BrokerError:
            await self._teardown()
            raise
        self.logged_in = True
        self.beat = asyncio.create_task(self._heartbeat())
        # WHAT DAS ITSELF OFFERS, rather than a venue list written down here
        # that goes stale the day the entitlements change.
        await self.write_line("GET RouteStatus")
        log.info("DAS CMD API: logged in as %s on %s:%s",
                 config.DAS_TRADER, config.DAS_HOST, config.DAS_PORT)

    async def _teardown(self) -> None:
        for t in (self.task, self.beat):
            if t is not None:
                t.cancel()
        self.task = self.beat = None
        if self.writer is not None:
            try:
                self.writer.close()
            except Exception:                               # noqa: BLE001
                pass
        self.reader = self.writer = None
        self.connected = self.logged_in = False

    async def aclose(self) -> None:
        if self.writer is not None:
            try:
                await self.write_line("QUIT")
            except Exception:                               # noqa: BLE001
                pass
        await self._teardown()

    # ── writing ─────────────────────────────────────────────────────────
    async def write_line(self, line: str) -> None:
        if self.writer is None:
            raise BrokerError("the DAS socket is not open")
        self.writer.write((line + LINE_END).encode("ascii", "replace"))
        await self.writer.drain()

    # ── reading ─────────────────────────────────────────────────────────
    async def _read_loop(self) -> None:
        """Every line DAS sends, forever, until the socket dies.

        A DROP IS NOT AN ERROR HERE. It is recorded, the waiting commands are
        failed as UNKNOWN (an order sent a moment before the socket died may
        well have landed), and `confirmed_at` simply stops advancing -- which
        is what makes the age on the page start climbing.
        """
        try:
            while True:
                raw = await self.reader.readline()
                if not raw:
                    raise ConnectionError("DAS closed the connection")
                self.confirmed_at = time.time()
                line = raw.decode("utf-8", "replace").strip()
                if line:
                    try:
                        self._dispatch(line)
                    except Exception:                       # noqa: BLE001
                        # ONE BAD LINE MUST NOT TAKE THE LINK DOWN. The whole
                        # record would go with it.
                        log.exception("DAS: could not handle %r", line[:200])
        except asyncio.CancelledError:
            raise
        except Exception as exc:                            # noqa: BLE001
            self.n_drops += 1
            self.last_error = f"{type(exc).__name__}: {exc}"
            log.warning("DAS link lost: %s", self.last_error)
            self.connected = self.logged_in = False
            self._fail_waiters(
                f"the DAS link dropped ({self.last_error}). Whether a command "
                f"sent just before it reached the order server is UNKNOWN.")

    def _fail_waiters(self, why: str) -> None:
        for fut in list(self._by_token.values()) + list(self._by_order.values()):
            if not fut.done():
                fut.set_exception(BrokerIndeterminate(why))
        self._by_token.clear()
        self._by_order.clear()
        if self._login_wait is not None and not self._login_wait.done():
            self._login_wait.set_exception(BrokerError(why))

    async def _heartbeat(self) -> None:
        """ECHO, so silence and death are different things.

        THE PROBLEM THIS SOLVES. Schwab's freshness comes free: every read is
        a request, so a read that returned proves the link. Here the state is
        pushed, and a name nobody is trading pushes nothing for hours -- so
        without this, a socket that died at 10:04 and a quiet book look
        exactly alike, and the page would show a confidently fresh order list
        drawn from a cache nobody can update. ECHO is answered immediately,
        is not one of the rate-limited order commands, and costs one line
        every few seconds.
        """
        try:
            while True:
                await asyncio.sleep(config.DAS_HEARTBEAT_S)
                try:
                    await self.write_line("ECHO")
                except Exception as exc:                    # noqa: BLE001
                    log.warning("DAS heartbeat failed: %s", exc)
                    return
        except asyncio.CancelledError:
            raise

    # ── dispatch ────────────────────────────────────────────────────────
    def _dispatch(self, line: str) -> None:
        """One line from DAS.

        EVERY LINE IS PROOF OF LIFE, which is why the stamp is here rather
        than only in the read loop: a push and a heartbeat reply are the same
        evidence, and `as_of` is built on this one number.

        THE END MARKERS ARE CHECKED FIRST, and that is not a style choice:
        `#POSEND` starts with `#POS`, `#OrderEnd` with `#Order` and
        `#TradeEnd` with `#Trade`. Testing the prefixes in the other order
        would treat the end of every snapshot as the start of a new one, and
        the record would be wiped exactly when it had just been filled.
        """
        self.confirmed_at = time.time()
        up = line.upper()

        if up.startswith("#POSEND"):
            if self._staging_pos is not None:
                self.positions = self._staging_pos
                self._staging_pos = None
            self.pos_snapshot = True
            return
        if up.startswith("#ORDEREND"):
            if self._staging_orders is not None:
                self.orders = self._staging_orders
                self._staging_orders = None
            self.order_snapshot = True
            return
        if up.startswith("#TRADEEND"):
            if self._staging_fills is not None:
                self.fills = self._staging_fills
                self._staging_fills = None
            return

        # A snapshot REPLACES what is held rather than merging into it. An
        # order that filled or was cancelled elsewhere has to disappear, and
        # a merge is how a phantom order stays on the screen.
        if up.startswith("#POS"):
            self._staging_pos = {}
            self.pos_snapshot = False
            return
        if up.startswith("#ORDER"):
            self._staging_orders = {}
            self.order_snapshot = False
            return
        if up.startswith("#TRADE"):
            self._staging_fills = {}
            return

        if up.startswith("%POS"):
            p = parse_pos(line)
            if p and p["symbol"]:
                (self._staging_pos if self._staging_pos is not None
                 else self.positions)[p["symbol"]] = p
            return
        if up.startswith("%ORDERACT"):
            self._on_order_act(line)
            return
        if up.startswith("%ORDER"):
            self._on_order(line)
            return
        if up.startswith("%TRADE"):
            self._on_trade(line)
            return

        if up.startswith("$ROUTESTATUS"):
            parts = line.split()
            if len(parts) >= 3:
                self.routes[parts[1].upper()] = \
                    parts[2].strip().upper() == "ENABLED"
            return

        if up.startswith("#ORDERSERVER") or up.startswith("#QUOTESERVER"):
            # The socket can be perfectly alive while the order server behind
            # it is not, which is a different failure and worth naming.
            self.order_server = line
            if "LOST" in up or "MISSING" in up or "FAILED" in up:
                log.warning("DAS: %s", line)
            return

        if up.startswith("#LOGIN") or "LOGIN" in up[:20]:
            self._on_login(line, up)
            return

    def _on_login(self, line: str, up: str) -> None:
        """Settle the login wait.

        THE ACK STRING IS NOT IN THE MANUAL -- it documents the command and
        the position/order/trade dump that follows a success, and not what
        the server says in between. So both paths are handled: an explicit
        #LOGIN line either way, and the arrival of the dump itself (in
        `_snapshot_started`) as proof that the login took.
        """
        if self._login_wait is None or self._login_wait.done():
            return
        if "SUCCESS" in up or "SUCCESSED" in up or "OK" in up:
            self._login_wait.set_result(True)
        elif "FAIL" in up or "REJECT" in up or "INVALID" in up or "ERROR" in up:
            self._login_wait.set_exception(BrokerError(
                f"DAS refused the login: {line}"))

    def _snapshot_started(self) -> None:
        if self._login_wait is not None and not self._login_wait.done():
            self._login_wait.set_result(True)

    def _on_order(self, line: str) -> None:
        o = parse_order(line, mine=set(self.minted))
        if not o or not o["order_id"]:
            return
        self._snapshot_started()
        target = (self._staging_orders if self._staging_orders is not None
                  else self.orders)
        target[o["order_id"]] = o
        tok = o.get("token")
        if tok is not None and tok in self.minted:
            self.minted[tok]["order_id"] = o["order_id"]
            self.minted[tok]["seen"] = True
        self._settle_place(o["order_id"], tok, o)
        self._trim()

    def _on_order_act(self, line: str) -> None:
        a = parse_order_act(line)
        if not a:
            return
        self._snapshot_started()
        act = a["action"].strip().upper()
        oid, tok = a["order_id"], a["token"]
        if tok is not None and tok in self.minted:
            self.minted[tok]["order_id"] = oid
            self.minted[tok]["seen"] = True

        fut = self._waiter(oid, tok)
        if fut is None:
            return
        note = f": {a['notes']}" if a.get("notes") else ""
        if act in ACT_INDETERMINATE:
            self._resolve(oid, tok, exc=BrokerIndeterminate(
                f"DAS answered {a['action']} for this order{note}. Whether it "
                f"reached the exchange is UNKNOWN — it must not be retried, "
                f"and only the record can settle it."))
        elif act in ACT_REFUSED:
            self._resolve(oid, tok, exc=BrokerError(
                f"DAS refused it: {a['action']}{note}. The order is "
                f"unchanged."))
        elif act in ACT_PLACED or act in ACT_CANCELLED or act in ACT_REPLACED:
            self._resolve(oid, tok, value={"order_id": oid,
                                           "status": a["action"],
                                           "notes": a.get("notes")})

    def _on_trade(self, line: str) -> None:
        t = parse_trade(line)
        if not t or not t["order_id"]:
            return
        self._snapshot_started()
        target = (self._staging_fills if self._staging_fills is not None
                  else self.fills)
        rows = target.setdefault(t["order_id"], [])
        if any(r.get("trade_id") == t["trade_id"] for r in rows):
            return
        rows.append({"t": t["t"], "price": t["price"], "qty": t["qty"],
                     "trade_id": t["trade_id"], "liq": t["liq"],
                     "ecn_fee": t["ecn_fee"], "route": t["route"]})

    def _trim(self) -> None:
        """Keep every working order and a tail of the finished ones."""
        if len(self.orders) <= 400:
            return
        done = [o for o in self.orders.values() if not o["working"]]
        done.sort(key=lambda o: base.entered_epoch(o.get("entered")) or 0.0)
        for o in done[:len(done) - 200]:
            self.orders.pop(o["order_id"], None)
            self.fills.pop(o["order_id"], None)

    # ── waiting for an acknowledgement ──────────────────────────────────
    def drop_waiter(self, *, token: int | None = None,
                    order_id: str | None = None) -> None:
        """Give up a claim that will never be answered (the send failed)."""
        for key, table in ((token, self._by_token), (order_id, self._by_order)):
            if key is not None and key in table:
                fut = table.pop(key)
                if not fut.done():
                    fut.cancel()

    def _waiter(self, order_id: str | None, token: int | None):
        if token is not None and token in self._by_token:
            return self._by_token[token]
        if order_id and order_id in self._by_order:
            return self._by_order[order_id]
        return None

    def _resolve(self, order_id: str | None, token: int | None, *,
                 value=None, exc=None) -> None:
        for key, table in ((token, self._by_token), (order_id, self._by_order)):
            if key is None or key not in table:
                continue
            fut = table.pop(key)
            if not fut.done():
                if exc is not None:
                    fut.set_exception(exc)
                else:
                    fut.set_result(value)

    def _settle_place(self, order_id: str, token: int | None,
                      o: dict) -> None:
        """A %ORDER carrying our token is the placement's answer.

        A REJECTION HERE IS DETERMINATE, unlike Send_Rej: DAS has looked at
        the order, given it an id and a status of Rejected, and nothing is
        resting. That is a refusal the pane can state plainly rather than a
        placement whose fate needs reading back.
        """
        fut = self._waiter(order_id, token)
        if fut is None:
            return
        if (o.get("status") or "").strip().upper() == "REJECTED":
            self._resolve(order_id, token, exc=BrokerError(
                f"DAS rejected the order (status {o['status']})."))
        else:
            self._resolve(order_id, token,
                          value={"order_id": order_id, "status": o["status"]})

    def register(self, *, token: int | None = None,
                 order_id: str | None = None) -> asyncio.Future:
        """Claim the acknowledgement BEFORE the command is written.

        THE RACE THIS CLOSES IS NOT THEORETICAL. Writing first and then
        awaiting means an `await` between the two, and the reader task runs
        in that gap — over a local socket the %ORDER can and does arrive
        first. The answer would then find no waiter, be dropped as an
        unsolicited push, and the placement would time out as UNKNOWN while
        the order sat happily on the book. Registering first makes the
        answer impossible to miss whichever order they arrive in.
        """
        fut = asyncio.get_running_loop().create_future()
        if token is not None:
            self._by_token[token] = fut
        if order_id:
            self._by_order[order_id] = fut
        return fut

    async def wait_ack(self, fut: asyncio.Future, *,
                       token: int | None = None,
                       order_id: str | None = None,
                       what: str = "the command") -> dict:
        """Wait for the push that settles a command already sent."""
        try:
            return await asyncio.wait_for(fut, timeout=config.DAS_ACK_S)
        except asyncio.TimeoutError as exc:
            raise BrokerIndeterminate(
                f"DAS did not acknowledge {what} within "
                f"{config.DAS_ACK_S:.0f}s. Whether it reached the order "
                f"server is UNKNOWN — it must not be retried, and only a "
                f"read of the record can settle it.") from exc
        finally:
            if token is not None:
                self._by_token.pop(token, None)
            if order_id:
                self._by_order.pop(order_id, None)

    # ── tokens ──────────────────────────────────────────────────────────
    def mint(self, *, symbol: str, side: str, qty: float,
             price: float | None, route: str | None) -> int:
        """The next client order id, and what it was minted for.

        WHY THE RECORD IS KEPT. `reconcile` is called for a placement whose
        answer never arrived, and it is given the SHAPE of that placement
        (symbol, side, quantity, price, when it was sent) because that is all
        Schwab could ever be matched on. Here the token is exact -- but only
        if this process remembers which token belonged to which placement,
        which is what this map is. An unresolved entry is a placement nobody
        has an answer for, and there is at most one of those at a time
        because the pane refuses to send again until it is settled.

        THE RANGE IS THE MANUAL'S: Token is a C int. Started at a random
        point high in the positive half and stepped by one, so two runs of
        this service on the same account do not hand out the same numbers
        and neither collides with the small tokens the montage uses.
        """
        self._token += 1
        if self._token >= MAX_INT:
            self._token = 1_000_000_000
        tok = self._token
        self.minted[tok] = {"symbol": symbol.upper(), "side": side.upper(),
                            "qty": float(qty), "price": price,
                            "route": route, "sent_at": time.time(),
                            "resolved": False, "seen": False,
                            "order_id": None}
        if len(self.minted) > 500:
            for k in sorted(self.minted,
                            key=lambda k: self.minted[k]["sent_at"])[:200]:
                self.minted.pop(k, None)
        return tok

    # ── what the link itself is doing ───────────────────────────────────
    def state(self) -> dict:
        age = (time.time() - self.confirmed_at) if self.confirmed_at else None
        return {
            "host": f"{config.DAS_HOST}:{config.DAS_PORT}"
                    if config.DAS_HOST else None,
            "connected": self.connected,
            "logged_in": self.logged_in,
            "confirmed_at": self.confirmed_at or None,
            # THE NUMBER THE PAGE SHOWS UNOBTRUSIVELY. It climbs the moment
            # the heartbeat stops being answered, which is the only outward
            # sign a pushed feed gives that it has died.
            "age_s": round(age, 1) if age is not None else None,
            "heartbeat_s": config.DAS_HEARTBEAT_S,
            "quiet": bool(age is not None
                          and age > config.DAS_HEARTBEAT_S * 3),
            "have_record": self.order_snapshot,
            "orders_held": len(self.orders),
            "positions_held": len(self.positions),
            "connects": self.n_connects,
            "drops": self.n_drops,
            "order_server": self.order_server,
            "last_error": self.last_error,
        }


LINK = DasLink()


# ── reading the record ──────────────────────────────────────────────────────
def _order_out(o: dict, fills: dict) -> dict:
    """The normalised order the pane draws, and only the keys it draws."""
    return {
        "order_id": o["order_id"],
        "symbol": o["symbol"],
        "side": o["side"],
        "qty": o["qty"],
        "filled": o["filled"],
        "price": o["price"],
        "type": o["type"],
        "status": o["status"],
        "working": o["working"],
        "entered": o["entered"],
        "tag": o["tag"],
        "from_api": o["from_api"],
        "fills": fills.get(o["order_id"], []),
        # Beyond the contract, and harmless: the venue this order is actually
        # resting on is the reason this broker was chosen.
        "route": o.get("route"),
    }


async def _ready(*, positions: bool = False) -> None:
    """Connected, logged in, and the record actually seen at least once.

    THE ONE REFUSAL HERE IS NARROW AND DELIBERATE, and it is not "the socket
    has gone quiet".

    A QUIET OR DROPPED LINK IS ANSWERED FROM THE CACHE. DAS Trader is open on
    the same screen as this page showing the same orders and positions, so a
    divergence is seen immediately, and a pane that refused to draw every
    time the platform was restarted would interrupt far more often than it
    would protect. The link's own state rides along in `socket` and is shown
    unobtrusively; `as_of` simply stops advancing, which is the truthful
    thing for it to do.

    WHAT IS REFUSED is a read taken before this process has EVER been told
    what the account holds. The alternative there is not a stale answer, it
    is a fabricated one: an empty working list reads as "nothing is resting"
    and an empty position list reads as "flat", and both are statements.
    """
    have = LINK.order_snapshot and (LINK.pos_snapshot or not positions)
    try:
        await LINK.ensure()
    except BrokerError:
        if have:
            return          # the link is down; the record is still the record
        raise
    if have or (LINK.order_snapshot
                and (LINK.pos_snapshot or not positions)):
        return
    deadline = time.time() + config.DAS_SNAPSHOT_S
    while time.time() < deadline:
        if LINK.order_snapshot and (LINK.pos_snapshot or not positions):
            return
        await asyncio.sleep(0.05)
    what = "position" if positions else "order"
    raise BrokerError(
        f"connected to DAS, but its {what} list has not arrived yet "
        f"({config.DAS_SNAPSHOT_S:.0f}s). Nothing is shown rather than an "
        f"empty list, which would read as a statement about the account.")


async def read_orders(symbols: list[str] | None = None,
                      priority: bool = False) -> dict:
    """Working and recent orders, from what DAS has pushed. Microseconds.

    NO CALL GOES OUT. The record here is not fetched on demand the way
    Schwab's is -- it arrives, and this reads what arrived. `as_of` is
    therefore the socket's last confirmation rather than the time of a
    request, and `rt_ms` is the honest near-zero cost of serving it, which is
    the whole difference from an 850ms poll.
    """
    t0 = time.perf_counter()
    await _ready()
    want = {s.upper() for s in symbols} if symbols else None
    working, recent = [], []
    for o in LINK.orders.values():
        if want and (o["symbol"] or "").upper() not in want:
            continue
        (working if o["working"] else recent).append(_order_out(o, LINK.fills))
    recent.sort(key=lambda o: base.entered_epoch(o.get("entered")) or 0.0,
                reverse=True)
    return {"ok": True, "as_of": LINK.confirmed_at,
            "rt_ms": round((time.perf_counter() - t0) * 1000.0, 1),
            "working": working, "recent": recent[:12],
            "limits": LIMITER.state(), "socket": LINK.state(),
            "stale_after_s": config.STALE_AFTER_S}


async def read_positions(symbols: list[str] | None = None,
                         priority: bool = False) -> dict:
    """Positions, from what DAS has pushed.

    A CLOSED POSITION IS NOT A POSITION. DAS pushes %POS with a quantity of
    zero when one is closed out; those are dropped rather than listed as a
    flat line the pane would have to know to ignore.

    `account_type`, `is_day_trader` and `round_trips` are Schwab facts with
    no DAS equivalent. They are None rather than invented — the pane does not
    draw them, and a made-up account type is worse than a missing one.
    """
    t0 = time.perf_counter()
    await _ready(positions=True)
    want = {s.upper() for s in symbols} if symbols else None
    positions = [
        {"symbol": p["symbol"], "qty": p["qty"], "avg": p["avg"],
         "day_pl": p["day_pl"]}
        for p in LINK.positions.values()
        if p["qty"] and abs(p["qty"]) > 1e-9
        and not (want and (p["symbol"] or "").upper() not in want)
    ]
    return {"ok": True, "as_of": LINK.confirmed_at,
            "rt_ms": round((time.perf_counter() - t0) * 1000.0, 1),
            "account_type": None, "is_day_trader": None, "round_trips": None,
            "positions": positions, "limits": LIMITER.state(),
            "socket": LINK.state(), "stale_after_s": config.STALE_AFTER_S}


async def state(symbols: list[str] | None = None,
                priority: bool = False) -> dict:
    """Both halves of one cache, so they are the same moment by construction.

    Schwab's version runs two HTTP reads concurrently and takes the OLDER
    `as_of` because they genuinely differ. Here there is one socket and one
    confirmation time, so the two halves cannot disagree.
    """
    o = await read_orders(symbols, priority)
    p = await read_positions(symbols, priority)
    return {**p, **o, "positions": p["positions"],
            "as_of": min(o["as_of"], p["as_of"]),
            "rt_ms": round(max(o["rt_ms"], p["rt_ms"]), 1),
            "account_type": p["account_type"],
            "is_day_trader": p["is_day_trader"],
            "round_trips": p["round_trips"]}


# ── sending ─────────────────────────────────────────────────────────────────
async def _send(line: str, *, kind: str, priority: bool = False) -> None:
    """One command down the socket, past the limiter.

    THE TWO FAILURES ARE DIFFERENT AND ARE RAISED DIFFERENTLY. A limiter
    refusal, or a socket that is not open, means nothing was written:
    determinate. A write that fails midway means it may have been: unknown.
    """
    refused = LIMITER.take(kind, priority=priority)
    if refused:
        raise BrokerError(f"rate limit: {refused}")
    try:
        await LINK.write_line(line)
    except BrokerError:
        raise
    except Exception as exc:                                # noqa: BLE001
        raise BrokerIndeterminate(
            f"the DAS socket failed while sending ({type(exc).__name__}: "
            f"{exc}). Whether the command reached the order server is "
            f"UNKNOWN — it must not be retried.") from exc


async def place(*, symbol: str, side: str, qty: int, price: float | None,
                route: str | None = None) -> dict:
    """Send one order. Arming and the guards have already passed upstream.

    THE ROUTE IS PER ORDER, which is the point of this broker: it is passed
    down from the pane, and `None` means the configured default rather than
    "no route" -- the CMD API requires one on every order.
    """
    t0 = time.perf_counter()
    await LINK.ensure()
    tok = LINK.mint(symbol=symbol, side=side, qty=qty, price=price,
                    route=route)
    cmd = build_neworder(token=tok, side=side, symbol=symbol, qty=int(qty),
                         price=price, route=route,
                         known_routes=set(LINK.routes))
    # CLAIMED BEFORE IT IS SENT. See DasLink.register: over a local socket
    # the answer can arrive inside the await that sends the question.
    fut = LINK.register(token=tok)
    await _send(cmd, kind="new")
    # A rejection comes back determinate and a timeout unknown, both from
    # here, and neither is caught: the distinction is the caller's to act on.
    # The token is already recorded either way, which is what lets an
    # unknown one be settled exactly later.
    ack = await LINK.wait_ack(fut, token=tok, what="the order")
    LINK.minted[tok]["resolved"] = True
    ms = (time.perf_counter() - t0) * 1000.0
    log.info("DAS placed %s %s %s @ %s via %s -> %s in %.0fms",
             side, qty, symbol, price, route or config.DAS_ROUTE,
             ack.get("order_id"), ms)
    return {"ok": True, "order_id": ack.get("order_id"),
            "status": ack.get("status"), "token": tok,
            "route": norm_route(route, known=set(LINK.routes)),
            "rt_ms": round(ms, 1)}


async def replace(*, order_id: str, symbol: str, side: str, qty: int,
                  price: float, route: str | None = None) -> dict:
    """Reprice a working order in ONE call. This is the ladder nudge.

    A ROUTE THAT WOULD MOVE THE ORDER IS REFUSED, and refused rather than
    ignored. REPLACE takes an order id, a size and a price and has no route
    field at all, so an order resting on ARCA stays on ARCA however the
    request is phrased. Accepting the route and repricing it where it already
    was would put a venue on the screen that the order is not at; the way to
    move an order between venues is to cancel it and place a new one, and
    saying so is more useful than a silent no-op.
    """
    t0 = time.perf_counter()
    await LINK.ensure()
    if route:
        resting = LINK.orders.get(str(order_id)) or {}
        at = (resting.get("route") or "").strip().upper()
        want = norm_route(route, known=set(LINK.routes))
        if at and want != at:
            raise BrokerError(
                f"this order is resting on {at} and DAS's REPLACE cannot "
                f"move it to {want} — the command carries no route. Cancel "
                f"it and place a new order on {want}.")
    cmd = build_replace(order_id=str(order_id), qty=int(qty), price=price)
    fut = LINK.register(order_id=str(order_id))
    await _send(cmd, kind="replace")
    ack = await LINK.wait_ack(fut, order_id=str(order_id),
                              what="the replace")
    ms = (time.perf_counter() - t0) * 1000.0
    return {"ok": True, "order_id": ack.get("order_id") or str(order_id),
            "status": ack.get("status"), "rt_ms": round(ms, 1)}


async def cancel(*, order_id: str) -> dict:
    """PRIORITY, and not gated on arming.

    Cancelling is how a mistake is undone. A disarmed pane that cannot cancel
    would be a pane whose safety switch traps an order, and a cancel refused
    for quota is the specific failure the limiter's reserve exists to stop.
    """
    t0 = time.perf_counter()
    await LINK.ensure()
    fut = LINK.register(order_id=str(order_id))
    await _send(f"CANCEL {order_id}", kind="cancel", priority=True)
    ack = await LINK.wait_ack(fut, order_id=str(order_id),
                              what="the cancel")
    ms = (time.perf_counter() - t0) * 1000.0
    log.info("DAS cancelled %s in %.0fms", order_id, ms)
    return {"ok": True, "order_id": str(order_id),
            "status": ack.get("status"), "rt_ms": round(ms, 1)}


async def flatten(*, symbol: str) -> dict:
    """Cancel everything working in the name, THEN close at market.

    ORDER MATTERS and it is the whole safety property. Closing while an order
    rests on the same symbol can leave that order to open a fresh position in
    the opposite direction the moment the flatten fills.

    ONE CANCEL, NOT N. `CANCEL ALLSYMB <ticker>` cancels every open order in
    the name in a single command -- including any placed by hand in DAS
    itself, which is correct: "everything working in this name" is what the
    interface promises and what makes the close safe. Each id is then waited
    for individually, and one that does not confirm is REPORTED, never
    swallowed -- the close below may reopen against it.
    """
    t0 = time.perf_counter()
    sym = symbol.upper()
    await _ready(positions=True)
    out: dict = {"cancelled": [], "rt_ms": 0.0}

    live = [o for o in LINK.orders.values()
            if o["working"] and (o["symbol"] or "").upper() == sym
            and o["order_id"]]
    if live:
        # Every id claimed BEFORE the cancel goes out, for the reason in
        # DasLink.register — and here it matters twice over, because one
        # command answers for several orders at once.
        futs = [LINK.register(order_id=o["order_id"]) for o in live]
        try:
            await _send(f"CANCEL ALLSYMB {sym}", kind="cancel", priority=True)
        except BrokerError as exc:
            for o in live:
                LINK.drop_waiter(order_id=o["order_id"])
            raise BrokerError(
                f"could not cancel the working orders in {sym}, so nothing "
                f"was closed: {exc}. Closing while an order rests can reopen "
                f"the position the moment the flatten fills.") from exc
        done = await asyncio.gather(*[
            LINK.wait_ack(f, order_id=o["order_id"], what="the cancel")
            for f, o in zip(futs, live)], return_exceptions=True)
        for o, res in zip(live, done):
            if isinstance(res, BaseException):
                out.setdefault("problems", []).append(
                    f"could not confirm the cancel of {o['order_id']}: {res}")
            else:
                out["cancelled"].append(o["order_id"])

    pos = LINK.positions.get(sym)
    qty = float(pos["qty"]) if pos and pos["qty"] else 0.0
    if abs(qty) < 1e-9:
        out["ok"] = True
        out["flat"] = True
        out["note"] = f"no {sym} position to close"
        out["rt_ms"] = round((time.perf_counter() - t0) * 1000.0, 1)
        return out

    side = "SELL" if qty > 0 else "BUY"
    tok = LINK.mint(symbol=sym, side=side, qty=abs(qty), price=None,
                    route=config.DAS_ROUTE)
    cmd = build_neworder(token=tok, side=side, symbol=sym, qty=int(abs(qty)),
                         price=None, route=config.DAS_ROUTE,
                         known_routes=set(LINK.routes))
    fut = LINK.register(token=tok)
    await _send(cmd, kind="new", priority=True)
    try:
        ack = await LINK.wait_ack(fut, token=tok, what="the closing order")
        out["status"] = ack.get("status")
    except BrokerIndeterminate as exc:
        # The cancels are done and the close is in the air. Said plainly,
        # because the position may or may not be closing and the next thing
        # to do is look, not send another one.
        out["ok"] = False
        out["indeterminate"] = True
        out["closed"] = {"side": side, "qty": int(abs(qty))}
        out.setdefault("problems", []).append(str(exc))
        out["rt_ms"] = round((time.perf_counter() - t0) * 1000.0, 1)
        return out
    out["ok"] = True
    out["closed"] = {"side": side, "qty": int(abs(qty))}
    out["rt_ms"] = round((time.perf_counter() - t0) * 1000.0, 1)
    log.info("DAS flattened %s: %s %d in %.0fms",
             sym, side, abs(qty), out["rt_ms"])
    return out


async def reconcile(*, symbol: str, side: str, qty: float,
                    price: float | None, sent_at: float) -> dict:
    """Did a placement whose reply never arrived actually land?

    ############# MATCHED ON THE TOKEN, NOT ON THE SHAPE #############
    #
    # This is the method base.match_placement exists to be overridden by, and
    # the reason the caveat in schwab.py does not apply here.
    #
    # Schwab gives a placement no identifier of ours: the only handle is
    # symbol, side, quantity, price and "entered after we sent", so two
    # identical orders seconds apart are permanently indistinguishable and
    # the honest answer is `ambiguous`. DAS takes a CLIENT-SUPPLIED token on
    # NEWORDER and returns it on %ORDER and %OrderAct, so the order that
    # placement created is the one carrying THAT NUMBER and no other.
    #
    # The map from token to placement is this process's (`LINK.minted`), and
    # an entry stays unresolved exactly while nobody has an answer for it --
    # which is the placement this call is asking about. The pane refuses to
    # send again while one is outstanding, so there is at most one.
    #
    # `sent_at` IS NOT USED AS THE KEY, deliberately. It comes from the
    # browser's clock and this map is stamped with the server's; a few
    # seconds of skew is normal and would throw away the right answer. The
    # shape still has to agree, so a reconcile for a different order cannot
    # attach to this one.
    #
    # THE ONE CASE THAT FALLS BACK TO SHAPE is this service having restarted
    # between the placement and the reconcile: the map is then empty, the
    # token is unrecoverable, and base.match_placement is all there is. It
    # is used, and the answer says so in `matched_on`, because a caller that
    # is told `ambiguous` should know which of the two rules produced it.
    #
    #################################################################
    """
    st = await read_orders([symbol], priority=True)
    pool = (st.get("working") or []) + (st.get("recent") or [])
    want_side = (side or "").upper()

    mine = [(tok, m) for tok, m in LINK.minted.items()
            if not m["resolved"]
            and m["symbol"] == symbol.upper()
            and m["side"] == want_side
            and abs(m["qty"] - float(qty)) < 1e-9
            and (price is None or m["price"] is None
                 or abs(float(m["price"]) - float(price)) < 0.005)]
    if mine:
        mine.sort(key=lambda kv: kv[1]["sent_at"])
        tok = mine[-1][0]
        hits = [o for o in pool if str(o.get("tag") or "") == str(tok)]
        out = {"state": "found" if hits else "absent", "orders": hits,
               "matched_on": "token", "token": tok}
        if hits:
            out["order"] = hits[0]
            # THE MINT IS NOT SPENT HERE. Marking it resolved would make a
            # second reconcile of the SAME placement fall back to matching by
            # shape -- and the pane may well ask twice. `place` marks it when
            # it gets its own answer; an entry that never got one stays
            # unresolved, and `mine` above prefers the most recent, so an old
            # one cannot be preferred to the placement being asked about.
            LINK.minted[tok]["order_id"] = hits[0].get("order_id")
    else:
        out = base.match_placement(pool, symbol=symbol, side=side, qty=qty,
                                   price=price, sent_at=sent_at)
        out["matched_on"] = "shape"
        out["why_shape"] = ("this service has no record of minting a token "
                            "for that placement — it has restarted since — "
                            "so the match is by shape and can be ambiguous.")
    out["ok"] = True
    out["as_of"] = st["as_of"]
    out["searched"] = len(pool)
    return out


# ── health ──────────────────────────────────────────────────────────────────
def health() -> dict:
    """DAS's own facts. The trading switches are the façade's to report."""
    link = LINK.state()
    return {
        "broker": "das",
        "host": link["host"],
        "account": config.DAS_ACCOUNT or None,
        "trader": config.DAS_TRADER or None,
        "have_credentials": bool(config.DAS_TRADER and config.DAS_PASSWORD
                                 and config.DAS_ACCOUNT),
        "socket": link,
        "last_error": link["last_error"],
        "limits": LIMITER.state(),
        # THE MONTAGE IS THE LIST; RouteStatus only marks it.
        #
        # RouteStatus answers "what can this login see", which includes
        # options, short-locate, test and PRO routes Cobra does not expose in
        # the montage — not one of them is somewhere to send an equity order.
        # So the offered list is config.DAS_ROUTES (the montage, in its own
        # order) and the broker's answer becomes a STATE on each entry:
        #
        #   enabled      RouteStatus says ENABLED
        #   disabled     RouteStatus says anything else. STILL OFFERED, and
        #                the page greys it: hiding it would make the dropdown
        #                change shape between pre-market and the session,
        #                which is how a venue you meant to use disappears
        #                without saying so.
        #   unconfirmed  RouteStatus never mentioned it (PSMT today), or has
        #                not answered yet. NOT "disabled": we do not know, and
        #                the montage says it exists. Shown, flagged, and
        #                selectable — DAS is the authority on the order, and a
        #                stale snapshot here must not block a live venue.
        "routing": {
            "supported": True,
            "default": config.DAS_ROUTE,
            "choices": list(config.DAS_ROUTES),
            "states": {r: ("enabled" if LINK.routes.get(r) else
                           "disabled" if r in LINK.routes else "unconfirmed")
                       for r in config.DAS_ROUTES},
            # Whether RouteStatus has answered AT ALL. False means every entry
            # is unconfirmed because nothing has been heard yet, which is a
            # different thing from a route being missing from a reply.
            "from_broker": bool(LINK.routes),
            "source": "montage",
            "why": ("DAS routes every order explicitly; SMAT is its smart "
                    "router and the only route that accepts every order "
                    "type. The list is your montage; DAS's RouteStatus marks "
                    "each one enabled or disabled."),
        },
    }


def problems() -> list[str]:
    """Why an order could not leave right now, in words that say what to do."""
    out = []
    if not config.DAS_HOST:
        out.append("LIVE_DAS_HOST is not set — it is the Tailscale address "
                   "of the machine running DAS Trader Pro.")
    if not (config.DAS_TRADER and config.DAS_PASSWORD and config.DAS_ACCOUNT):
        out.append("LIVE_DAS_TRADER / LIVE_DAS_PASSWORD / LIVE_DAS_ACCOUNT "
                   "are not all set, so the CMD API cannot be logged in to.")
    if out:
        return out
    link = LINK.state()
    if not link["connected"]:
        out.append(f"not connected to the DAS CMD API at {link['host']} — "
                   f"DAS Trader Pro has to be running and logged in on that "
                   f"machine."
                   + (f" Last: {link['last_error']}"
                      if link["last_error"] else ""))
    elif not link["logged_in"]:
        out.append("connected to DAS but not logged in.")
    elif link["quiet"]:
        out.append(f"the DAS link has been silent for {link['age_s']}s "
                   f"(heartbeat every {link['heartbeat_s']}s) — the orders "
                   f"below are the last ones it pushed.")
    if link["order_server"] and ("LOST" in link["order_server"].upper()
                                 or "MISSING" in link["order_server"].upper()):
        out.append(f"DAS reports {link['order_server']} — the socket is up "
                   f"but the order server behind it is not.")
    return out


async def aclose() -> None:
    await LINK.aclose()


# ── the adapter ─────────────────────────────────────────────────────────────
#
# METHODS DELEGATE TO THE MODULE FUNCTIONS ABOVE, by name, at call time —
# the same arrangement as schwab.py, and for the same reason: the functions
# are what the checks drive and monkeypatch, and binding them here rather
# than at import keeps a patched one visible to the class.
class DasBroker(base.Broker):
    """DAS Trader, over the CMD API. One socket, one account, equities."""

    name = "das"

    async def read_orders(self, symbols=None, priority=False):
        return await read_orders(symbols, priority)

    async def read_positions(self, symbols=None, priority=False):
        return await read_positions(symbols, priority)

    async def state(self, symbols=None, priority=False):
        return await state(symbols, priority)

    async def place(self, *, symbol, side, qty, price, route=None):
        return await place(symbol=symbol, side=side, qty=qty, price=price,
                           route=route)

    async def replace(self, *, order_id, symbol, side, qty, price, route=None):
        return await replace(order_id=order_id, symbol=symbol, side=side,
                             qty=qty, price=price, route=route)

    async def cancel(self, *, order_id):
        return await cancel(order_id=order_id)

    async def flatten(self, *, symbol):
        return await flatten(symbol=symbol)

    async def reconcile(self, *, symbol, side, qty, price, sent_at):
        return await reconcile(symbol=symbol, side=side, qty=qty,
                               price=price, sent_at=sent_at)

    def health(self):
        return health()

    def problems(self):
        return problems()

    async def aclose(self):
        await aclose()
