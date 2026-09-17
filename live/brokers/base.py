"""The broker interface: what an adapter must implement, and what it may assume.

WHY THIS EXISTS. The pane and the arming logic must not know which broker is
behind them. `live/broker.py` is the façade they talk to; it owns everything
that is a POLICY decision -- the four switches, the guards, the order in which
a flatten does its work -- and calls an adapter for the parts that are a
PROTOCOL decision: how to speak to this particular broker.

THE SPLIT IS A SAFETY PROPERTY, not tidiness. Arming and the guards are
checked in the façade, ONCE, before any adapter is called. A new adapter
therefore cannot trade while disarmed, cannot exceed the size, position,
notional or distance limits, and cannot flatten without cancelling first --
not because its author remembered to check, but because it is never reached
until the façade has. An adapter that re-checked them would be harmless; one
that was trusted to check them would put the switches in every new file.

WHAT AN ADAPTER MUST NOT DO:
  * consult the arming state, the runtime flag or the guards -- by the time
    it is called those have passed, and a second opinion here would make the
    real gate ambiguous;
  * retry a call whose outcome it does not know (see BrokerIndeterminate);
  * cache the existence of an order. The broker's own record is the truth
    and is re-read; a cached list that looks fresh is the failure this whole
    subsystem is arranged against.

WHAT AN ADAPTER MUST DO:
  * return the NORMALISED shapes below, whatever the wire format is. The
    pane draws these keys and nothing else;
  * stamp every read with `as_of` (epoch seconds, when the broker confirmed
    it) and `rt_ms`, because the page shows the age of what it is drawing;
  * raise BrokerError for a determinate failure and BrokerIndeterminate for
    an unknown one. Getting that distinction wrong is how a timeout becomes
    a double position.

THE NORMALISED SHAPES (the pane's contract; `check_broker.py` asserts them):

  order    {order_id: str, symbol: str, side: str, qty: float,
            filled: float|None, price: float|None, type: str|None,
            status: str, working: bool, entered: str|None (ISO),
            tag: str|None, from_api: bool, fills: [fill]}
           `working` is the adapter's own judgement of "still live at the
           broker", and it must resolve an unknown status to True: an order
           wrongly called dead disappears from the screen AND from cancel
           and flatten, which is how you get out.
  fill     {t: str|None (ISO), price: float, qty: float}
  position {symbol: str, qty: float (net; negative is short),
            avg: float|None, day_pl: float|None}

  read_orders()    -> {ok, as_of, rt_ms, working: [order], recent: [order],
                       limits: dict|None, stale_after_s: float}
                     `recent` is newest-first and may be truncated.
  read_positions() -> {ok, as_of, rt_ms, positions: [position],
                       account_type, is_day_trader, round_trips,
                       limits: dict|None, stale_after_s: float}
  place/replace    -> {ok: True, order_id: str|None, status: int|str,
                       rt_ms: float}
  cancel           -> {ok: True, order_id: str, status: int|str, rt_ms: float}
  flatten          -> {ok: True, cancelled: [order_id], rt_ms: float,
                       flat?: bool, note?: str, closed?: {side, qty},
                       problems?: [str]}
  health()         -> dict of this broker's own facts (credentials, session,
                      limits). The façade adds the trading switches; an
                      adapter never reports on them.
  problems()       -> [str] reasons this broker could not trade right now,
                      in words a person can act on ("authorise in ...").
"""
from __future__ import annotations

from abc import ABC, abstractmethod


class BrokerError(Exception):
    """Anything that stopped a call reaching the broker, or that it refused.

    DETERMINATE. Raising this says nothing was done: a guard refused, the
    limiter refused, or the broker replied with a rejection. The caller may
    safely assume the account is unchanged.
    """


class BrokerIndeterminate(BrokerError):
    """WE DO NOT KNOW WHETHER THE ORDER LANDED.

    A timeout, a dropped connection, or a 5xx after the request was already
    on the wire. The distinction from BrokerError is the whole point and it
    is not cosmetic: a determinate failure can be retried, and this one must
    never be -- a retry is how a timeout becomes a double position.

    Everything that raises this puts the pane into an unresolved state that
    only a read of the broker's own record can clear.
    """


def match_placement(orders: list[dict], *, symbol: str, side: str,
                    qty: float, price: float | None,
                    sent_at: float) -> dict:
    """Find the order a timed-out placement may have created, by SHAPE.

    Broker-agnostic and deliberately conservative: symbol, side, quantity,
    price and "entered after we sent". More than one match returns
    `ambiguous` rather than a guess -- attaching the pane to the wrong order
    means the next nudge reprices a stranger's order.

    An adapter whose API gives placements a CLIENT-SUPPLIED id does not need
    this: it should override `reconcile` and match on that id instead, which
    is exact where this is not. Schwab's does not (see schwab.py), which is
    why this exists.
    """
    want_side = (side or "").upper()
    hits = []
    for o in orders:
        if (o.get("symbol") or "").upper() != symbol.upper():
            continue
        if (o.get("side") or "").upper() != want_side:
            continue
        if abs(float(o.get("qty") or 0) - float(qty)) > 1e-9:
            continue
        if price is not None:
            op = o.get("price")
            if op is None or abs(float(op) - float(price)) > 0.005:
                continue
        # Entered AFTER we sent, with a second of slack for clock skew
        # between this box and the broker's stamp.
        ts = entered_epoch(o.get("entered"))
        if ts is not None and ts < sent_at - 1.0:
            continue
        hits.append(o)

    if not hits:
        return {"state": "absent", "orders": []}
    if len(hits) > 1:
        return {"state": "ambiguous", "orders": hits}
    return {"state": "found", "orders": hits, "order": hits[0]}


def entered_epoch(stamp: str | None) -> float | None:
    if not stamp:
        return None
    from datetime import datetime
    for fmt in ("%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dT%H:%M:%S.%f%z"):
        try:
            return datetime.strptime(stamp, fmt).timestamp()
        except ValueError:
            continue
    return None


class Broker(ABC):
    """One venue. Transport only -- the switches and the guards are upstream.

    `priority` marks a call that is getting flat (a cancel, a flatten, a
    reconcile that does not know its own position). An adapter with a rate
    limiter must keep a reserve such a call may spend and ordinary traffic
    may not: the call that must never be refused for quota is the one that
    closes a position.
    """

    #: Short, stable key used in config, logs and the health payload.
    name: str = "broker"

    # ── reading the record ──────────────────────────────────────────────
    @abstractmethod
    async def read_orders(self, symbols: list[str] | None = None,
                          priority: bool = False) -> dict:
        """Working and recent orders, normalised. See the shapes above."""

    @abstractmethod
    async def read_positions(self, symbols: list[str] | None = None,
                             priority: bool = False) -> dict:
        """Positions and whatever account facts this broker exposes."""

    async def state(self, symbols: list[str] | None = None,
                    priority: bool = False) -> dict:
        """Both reads as ONE consistent answer, concurrently where possible.

        The default is correct but serial; an adapter that can overlap the
        two reads should override it (Schwab's does -- ~850ms and ~370ms
        together cost the slower one). `as_of` must be the OLDER of the two,
        never the newer: the pane shows the age of the whole snapshot.
        """
        o = await self.read_orders(symbols, priority)
        pos = await self.read_positions(symbols, priority)
        return {**pos, **o,
                "positions": pos["positions"],
                "as_of": min(o["as_of"], pos["as_of"]),
                "rt_ms": round(max(o["rt_ms"], pos["rt_ms"]), 1),
                "account_type": pos.get("account_type"),
                "is_day_trader": pos.get("is_day_trader"),
                "round_trips": pos.get("round_trips")}

    # ── changing the record ─────────────────────────────────────────────
    @abstractmethod
    async def place(self, *, symbol: str, side: str, qty: int,
                    price: float | None) -> dict:
        """Send one order. `price` None means market.

        Arming and the guards have already passed. Raise BrokerError if the
        broker refuses it, BrokerIndeterminate if the outcome is unknown.
        """

    @abstractmethod
    async def replace(self, *, order_id: str, symbol: str, side: str,
                      qty: int, price: float) -> dict:
        """Reprice a working order, in ONE call where the API allows it.

        This is the ladder nudge: a cancel-then-place round trip loses the
        queue position and leaves a window with no order at all, so an
        adapter should use a native replace and say so here if it cannot.
        """

    @abstractmethod
    async def cancel(self, *, order_id: str) -> dict:
        """Cancel one order. PRIORITY, and never gated on arming.

        Cancelling is how a mistake is undone: a disarmed pane that cannot
        cancel is a safety switch that traps an order.
        """

    @abstractmethod
    async def flatten(self, *, symbol: str) -> dict:
        """Cancel everything working in the name, THEN close at market.

        ORDER MATTERS and it is not optional: closing while an order rests
        on the same symbol can leave that order to open a fresh position in
        the opposite direction the moment the flatten fills. A cancel that
        fails is reported in `problems`, never swallowed -- the flatten may
        then reopen against it.
        """

    # ── reconciling a placement nobody saw the answer to ────────────────
    async def reconcile(self, *, symbol: str, side: str, qty: float,
                        price: float | None, sent_at: float) -> dict:
        """Did a placement whose reply never arrived actually land?

        The default reads the record and matches by shape (match_placement),
        which can answer `ambiguous`. Override where the API has a
        client-supplied order id to match on exactly.
        """
        st = await self.read_orders([symbol], priority=True)
        pool = (st.get("working") or []) + (st.get("recent") or [])
        out = match_placement(pool, symbol=symbol, side=side, qty=qty,
                              price=price, sent_at=sent_at)
        out["ok"] = True
        out["as_of"] = st["as_of"]
        out["searched"] = len(pool)
        return out

    # ── housekeeping ────────────────────────────────────────────────────
    @abstractmethod
    def health(self) -> dict:
        """This broker's own facts. Never the trading switches."""

    def problems(self) -> list[str]:
        """Why this broker could not trade right now, in actionable words."""
        return []

    async def aclose(self) -> None:
        """Release any connection held. Called on service shutdown."""
        return None
