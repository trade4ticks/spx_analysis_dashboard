"""Schwab Trader API — place, replace, cancel, flatten, and read the record.

THIS DASHBOARD IS NOT THE SOURCE OF TRUTH, and the whole module is arranged
around that. If the WebSocket drops or the page closes, the orders are still
live at Schwab. So every piece of state carries when it was last confirmed,
the page shows that age, and nothing here caches an order's existence — the
broker's own list is re-read and replaces whatever was held.

The dangerous version of this module is the one that shows a confident,
wrong list of working orders. A stale list that says it is stale is safe; a
fresh-looking list that is thirty seconds old is not.

THREE INDEPENDENT SWITCHES have to be on before a single order can leave:

    LIVE_TRADING_ENABLED    in the environment, default OFF. New
                            order-placing code should not be able to
                            trade because a page element was clicked.
    the pane's arm toggle   off by default, per pane, and the request
                            carries it — the server refuses a body
                            without it rather than trusting the UI.
    the guards              size, position, notional and distance from
                            the last print, all checked HERE and not
                            only in the browser.

RATE: 120 CALLS PER MINUTE, TOTAL, across every pane. A round trip with four
repricings on entry and three on exit is eight calls, which is nothing on
average and can burst when three panes reprice together. So the limiter
keeps a RESERVE that only getting-flat may spend: a cancel or a flatten must
never be the call that gets refused for quota. 429 is handled explicitly and
surfaced — an order that silently failed to move is the worst outcome here,
because the position is then not where the screen says it is.

TOKENS ARE SHARED WITH THE PORTFOLIO DASHBOARD, which is the one piece of
genuinely awkward state. See _token().
"""
from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import time
from collections import deque
from pathlib import Path

from live import config

# httpx IS IMPORTED LAZILY, inside the two functions that speak HTTP.
#
# Everything worth checking here — the guards, the rate limiter, the order
# body, the normalisers — is pure, and a module-level import would make all
# of it unreachable on a machine without the library. The checks are the
# reason: they run on a laptop, the orders leave from the box.
def _httpx():
    try:
        import httpx
        return httpx
    except ImportError as exc:                              # pragma: no cover
        raise BrokerError(
            "httpx is not installed in this environment, so no Schwab call "
            "can be made. It is in requirements.txt.") from exc

log = logging.getLogger("live.broker")

_TOKEN_URL = "https://api.schwabapi.com/v1/oauth/token"
_API = "https://api.schwabapi.com/trader/v1"

# Schwab's own status strings for an order that is still live. Anything not in
# here is terminal and must not be drawn as a working order.
WORKING_STATES = {
    "AWAITING_PARENT_ORDER", "AWAITING_CONDITION", "AWAITING_STOP_CONDITION",
    "AWAITING_MANUAL_REVIEW", "ACCEPTED", "AWAITING_UR_OUT",
    "PENDING_ACTIVATION", "QUEUED", "WORKING", "NEW",
    "PENDING_REPLACE", "PENDING_CANCEL",
}

_account_hash: str | None = None
_last_error: str | None = None


# ── the rate limiter ────────────────────────────────────────────────────────
class RateLimiter:
    """120 calls a minute, shared, with a reserve for getting flat.

    A plain bucket would let four panes repricing on a fast tape spend the
    whole minute's quota, and the next call — the cancel — would be the one
    refused. Ordinary traffic may therefore spend only down to the reserve;
    cancel and flatten may spend all of it.
    """

    def __init__(self, per_min: int, reserve: int):
        self.per_min = per_min
        self.reserve = reserve
        self.calls: deque[float] = deque()
        self.blocked_until = 0.0
        self.refusals = 0
        self.n_429 = 0

    def _prune(self, now: float) -> None:
        while self.calls and self.calls[0] < now - 60.0:
            self.calls.popleft()

    def take(self, priority: bool = False) -> str | None:
        """Spend one call. Returns why it was refused, or None."""
        now = time.time()
        self._prune(now)
        if now < self.blocked_until:
            self.refusals += 1
            return (f"Schwab returned 429; holding off for "
                    f"{self.blocked_until - now:.1f}s more")
        ceiling = self.per_min if priority else self.per_min - self.reserve
        if len(self.calls) >= ceiling:
            self.refusals += 1
            oldest = self.calls[0]
            wait = max(0.0, 60.0 - (now - oldest))
            return (f"at {len(self.calls)}/{self.per_min} calls this minute"
                    + ("" if priority else
                       f" ({self.reserve} held back so a cancel or flatten "
                       f"always has quota)")
                    + f"; {wait:.1f}s until one frees")
        self.calls.append(now)
        return None

    def note_429(self, retry_after: str | None) -> None:
        """Explicit, never swallowed."""
        self.n_429 += 1
        try:
            wait = float(retry_after) if retry_after else 5.0
        except (TypeError, ValueError):
            wait = 5.0
        held = max(1.0, min(wait, 60.0))
        self.blocked_until = time.time() + held
        # The CLAMPED value, because that is the one in force. A log that
        # says 99999s while the limiter waits 60 is a log that has to be
        # reconciled against the code before it can be believed.
        log.warning("Schwab 429 — holding off %.1fs (Retry-After %r)",
                    held, retry_after)

    def state(self) -> dict:
        now = time.time()
        self._prune(now)
        return {
            "per_min": self.per_min,
            "reserve": self.reserve,
            "used": len(self.calls),
            "available": max(0, self.per_min - self.reserve - len(self.calls)),
            "blocked_for_s": round(max(0.0, self.blocked_until - now), 1),
            "refusals": self.refusals,
            "n_429": self.n_429,
        }


LIMITER = RateLimiter(config.SCHWAB_CALLS_PER_MIN, config.SCHWAB_CALL_RESERVE)


# ── tokens ──────────────────────────────────────────────────────────────────
def _token_file() -> Path:
    return Path(config.SCHWAB_TOKEN_FILE)


def _read_tokens() -> dict | None:
    try:
        return json.loads(_token_file().read_text())
    except Exception:                                       # noqa: BLE001
        return None


def _basic() -> str:
    return "Basic " + base64.b64encode(
        f"{config.SCHWAB_API_KEY}:{config.SCHWAB_API_SECRET}".encode()).decode()


def _token_sync() -> str:
    """A valid access token, refreshing only when it has to.

    THE SHARED-STATE PROBLEM, stated because it is real and not fully fixable
    from this side. The token file belongs to the portfolio dashboard, and
    Schwab rotates the refresh token on every refresh. Two processes that
    refresh at the same moment send the same refresh token and one of them
    ends up holding a dead one — which is a re-authorisation, not a retry.

    What this does about it:

      * refreshes only inside the last REFRESH_MARGIN_S of the access
        token's life, so the two processes are rarely both due;
      * takes an exclusive lock around the refresh, and RE-READS the file
        after acquiring it — if the other process refreshed while this one
        waited, its token is used and no second refresh happens;
      * on a refused refresh, re-reads once before giving up, because the
        most likely cause is exactly that rotation.

    The lock cannot bind the portfolio dashboard, which does not take it. The
    proper fix is one owner for the refresh; until then this is a narrow
    window rather than a closed one, and it is written down.
    """
    global _last_error
    tokens = _read_tokens()
    if tokens is None:
        raise BrokerError(
            f"no Schwab tokens at {_token_file()} — authorise in the "
            f"portfolio dashboard first (/schwab/auth)")
    if time.time() < tokens.get("expires_at", 0) - config.REFRESH_MARGIN_S:
        return tokens["access_token"]

    lock = _token_file().with_suffix(".lock")
    with _FileLock(lock):
        fresh = _read_tokens() or tokens
        if time.time() < fresh.get("expires_at", 0) - config.REFRESH_MARGIN_S:
            # Somebody else refreshed while this call waited for the lock.
            return fresh["access_token"]
        try:
            return _do_refresh(fresh)
        except Exception as exc:                            # noqa: BLE001
            again = _read_tokens()
            if again and again.get("access_token") != fresh.get("access_token"):
                log.info("refresh failed but the file moved on; using it")
                return again["access_token"]
            _last_error = f"token refresh failed: {exc}"
            raise BrokerError(_last_error) from exc


async def _token() -> str:
    """`_token_sync` in a thread: it can take a file lock and wait on HTTP.

    Both would stall the event loop, and this service's loop is also the tape
    — the flush pump, every client's fan-out, and the frame the browser is
    waiting on.
    """
    return await asyncio.to_thread(_token_sync)


def _do_refresh(tokens: dict) -> str:
    r = _httpx().post(_TOKEN_URL, timeout=15,
                      headers={"Authorization": _basic(),
                               "Content-Type":
                                   "application/x-www-form-urlencoded"},
                      data={"grant_type": "refresh_token",
                            "refresh_token": tokens["refresh_token"]})
    if r.status_code >= 400:
        raise BrokerError(f"{r.status_code} {r.reason}: {r.text[:180]}")
    new = r.json()
    new["expires_at"] = time.time() + int(new.get("expires_in", 1800))
    # Written back to the SHARED file, because the portfolio dashboard reads
    # this same file on every call and would otherwise keep using the token
    # this refresh has just invalidated.
    tmp = _token_file().with_suffix(".tmp")
    tmp.write_text(json.dumps(new))
    tmp.replace(_token_file())
    log.info("refreshed the Schwab access token")
    return new["access_token"]


class _FileLock:
    """An exclusive lock, where the platform has one.

    flock is POSIX; the box is Linux and that is where refreshes happen. On a
    development machine without it the lock is a no-op, which is honest: it
    is not silently pretending to serialise anything.
    """

    def __init__(self, path: Path):
        self.path = path
        self.fh = None

    def __enter__(self):
        try:
            import fcntl
            self.fh = open(self.path, "w")
            fcntl.flock(self.fh, fcntl.LOCK_EX)
        except Exception:                                   # noqa: BLE001
            self.fh = None
        return self

    def __exit__(self, *a):
        if self.fh is not None:
            try:
                import fcntl
                fcntl.flock(self.fh, fcntl.LOCK_UN)
            finally:
                self.fh.close()
        return False


class BrokerError(Exception):
    """Anything that stopped a call reaching Schwab, or that Schwab refused."""


# ── the HTTP layer ──────────────────────────────────────────────────────────
async def _acall(method: str, path: str, *, params=None, body=None,
                 priority: bool = False) -> tuple[object, int, float]:
    """One Schwab call. Returns (parsed, status, milliseconds).

    The elapsed time is measured around the request and nothing else, because
    it is a number the page displays and reasons about: part of the point of
    trading from here is finding out whether this path is quicker than the
    click it replaces. Token acquisition and rate limiting are deliberately
    outside the measurement — they are this service's overhead, not Schwab's.
    """
    global _last_error
    refused = LIMITER.take(priority=priority)
    if refused:
        raise BrokerError(f"rate limit: {refused}")
    tok = await _token()
    httpx = _httpx()
    # CONTENT-TYPE ONLY WHERE THERE IS CONTENT.
    #
    # Schwab answers a bodyless GET carrying `Content-Type: application/json`
    # with 400 wrapping an internal 500 — no mention of the header, and the
    # same request without it returns 200. Sent unconditionally this made
    # every read fail, starting with the account lookup, and the message gave
    # no way to tell that from an expired token or a scope problem.
    headers = {"Authorization": f"Bearer {tok}"}
    if body is not None:
        headers["Content-Type"] = "application/json"
    t0 = time.perf_counter()
    async with httpx.AsyncClient(timeout=15) as client:
        r = await client.request(
            method, _API + path, headers=headers,
            params=params or None,
            content=json.dumps(body) if body is not None else None)
    ms = (time.perf_counter() - t0) * 1000.0

    if r.status_code == 429:
        # EXPLICIT. An order that silently failed to move is the worst
        # outcome available here, because the position is then not where the
        # screen says it is.
        LIMITER.note_429(r.headers.get("Retry-After"))
        raise BrokerError("Schwab returned 429 (too many requests). The order "
                          "did NOT change. Check the working list before "
                          "retrying.")
    if r.status_code >= 400:
        _last_error = f"{r.status_code}: {r.text[:200]}"
        raise BrokerError(_last_error)
    _last_error = None

    parsed: object = None
    if r.content:
        try:
            parsed = r.json()
        except ValueError:
            parsed = r.text
    # A placement replies 201 with no body; the id is in Location.
    if r.status_code == 201 and "location" in {k.lower() for k in r.headers}:
        loc = r.headers.get("Location") or r.headers.get("location") or ""
        parsed = {"order_id": loc.rstrip("/").split("/")[-1]}
    return parsed, r.status_code, ms


async def account_hash() -> str:
    global _account_hash
    if _account_hash:
        return _account_hash
    if config.SCHWAB_ACCOUNT_HASH:
        _account_hash = config.SCHWAB_ACCOUNT_HASH
        return _account_hash
    data, _, _ = await _acall("GET", "/accounts/accountNumbers", priority=True)
    rows = data if isinstance(data, list) else [data]
    if not rows:
        raise BrokerError("Schwab returned no accounts")
    _account_hash = rows[0]["hashValue"]
    return _account_hash


# ── reading the record ──────────────────────────────────────────────────────
def _leg(order: dict) -> dict:
    legs = order.get("orderLegCollection") or [{}]
    return legs[0]


def _norm_order(o: dict) -> dict:
    leg = _leg(o)
    inst = leg.get("instrument") or {}
    return {
        "order_id": str(o.get("orderId") or ""),
        "symbol": inst.get("symbol"),
        "side": leg.get("instruction"),
        "qty": o.get("quantity"),
        "filled": o.get("filledQuantity"),
        "price": o.get("price"),
        "type": o.get("orderType"),
        "status": o.get("status"),
        "working": o.get("status") in WORKING_STATES,
        "entered": o.get("enteredTime"),
    }


def _norm_position(p: dict) -> dict:
    inst = p.get("instrument") or {}
    long_q = float(p.get("longQuantity") or 0)
    short_q = float(p.get("shortQuantity") or 0)
    net = long_q - short_q
    return {
        "symbol": inst.get("symbol"),
        "qty": net,
        "avg": p.get("averagePrice"),
        "day_pl": p.get("currentDayProfitLoss"),
    }


async def state(symbols: list[str] | None = None,
                priority: bool = False) -> dict:
    """Positions and working orders, as the broker has them right now.

    TWO CALLS, and ORDINARY quota by default. The page polls this every four
    seconds, which is thirty calls a minute; letting a background poll spend
    the reserve would empty the very quota the reserve exists to keep for a
    cancel. Priority is for the reads that are part of getting flat, where
    being refused is the failure being designed out.

    A refused poll is safe precisely because of `as_of`: the page keeps the
    older state and keeps saying how old it is.
    """
    hv = await account_hash()
    now = time.time()
    acct, _, ms1 = await _acall("GET", f"/accounts/{hv}",
                                params={"fields": "positions"},
                                priority=priority)
    # A day is the right window: an order entered this morning can still be
    # working, and nothing older can be.
    from datetime import datetime, timedelta, timezone
    fmt = lambda d: d.strftime("%Y-%m-%dT%H:%M:%S.000Z")            # noqa: E731
    utc = datetime.now(timezone.utc)
    orders, _, ms2 = await _acall(
        "GET", f"/accounts/{hv}/orders",
        params={"fromEnteredTime": fmt(utc - timedelta(days=1)),
                "toEnteredTime": fmt(utc + timedelta(minutes=1))},
        priority=priority)

    sec = ((acct or {}).get("securitiesAccount") or {}) \
        if isinstance(acct, dict) else {}
    positions = [_norm_position(p) for p in (sec.get("positions") or [])]
    all_orders = [_norm_order(o) for o in (orders or [])
                  if isinstance(o, dict)]
    working = [o for o in all_orders if o["working"]]

    if symbols:
        want = {s.upper() for s in symbols}
        positions = [p for p in positions if (p["symbol"] or "").upper() in want]
        working = [o for o in working if (o["symbol"] or "").upper() in want]

    return {
        "ok": True,
        "as_of": now,
        "rt_ms": round(ms1 + ms2, 1),
        "account_type": sec.get("type"),
        "is_day_trader": bool(sec.get("isDayTrader")),
        "round_trips": sec.get("roundTrips"),
        "positions": positions,
        "working": working,
        # Recent terminal orders, so a fill is visible rather than an order
        # just vanishing from the working list.
        "recent": [o for o in all_orders if not o["working"]][-12:],
        "limits": LIMITER.state(),
    }


# ── the guards ──────────────────────────────────────────────────────────────
def check_guards(*, symbol: str, side: str, qty: float, price: float | None,
                 reference: float | None, position_qty: float) -> str | None:
    """Refuse before the call, not after. Returns why, or None.

    ENFORCED HERE and not only in the browser, because the browser is where a
    guard is easiest to bypass by accident — a stale page, a replayed
    request, a hand-typed fetch during debugging.
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


def _armed_check(armed: bool) -> str | None:
    if not config.TRADING_ENABLED:
        return ("live trading is disabled on the server: set "
                "LIVE_TRADING_ENABLED=1 in /spx_analysis_dashboard/.env and "
                "restart spx-live. Deliberately off by default.")
    if not armed:
        return ("this pane is not armed — the request did not carry it. "
                "Arming is per pane and off by default.")
    return None


# ── orders ──────────────────────────────────────────────────────────────────
def _equity_order(side: str, qty: int, symbol: str,
                  price: float | None) -> dict:
    body = {
        "session": "NORMAL",
        "duration": "DAY",
        "orderStrategyType": "SINGLE",
        "orderType": "LIMIT" if price is not None else "MARKET",
        "orderLegCollection": [{
            "instruction": side.upper(),
            "quantity": int(qty),
            "instrument": {"symbol": symbol.upper(), "assetType": "EQUITY"},
        }],
    }
    if price is not None:
        # Two decimals. Schwab refuses a sub-penny limit on an equity, and a
        # float that stringifies to 318.52000000000004 is a sub-penny limit.
        body["price"] = f"{price:.2f}"
    return body


async def place(*, symbol: str, side: str, qty: int, price: float | None,
                armed: bool, reference: float | None,
                position_qty: float) -> dict:
    why = _armed_check(armed)
    if why:
        raise BrokerError(why)
    why = check_guards(symbol=symbol, side=side, qty=qty, price=price,
                       reference=reference, position_qty=position_qty)
    if why:
        raise BrokerError(f"refused by the guards: {why}")
    hv = await account_hash()
    data, status, ms = await _acall("POST", f"/accounts/{hv}/orders",
                                    body=_equity_order(side, qty, symbol,
                                                       price))
    oid = (data or {}).get("order_id") if isinstance(data, dict) else None
    log.info("placed %s %s %s @ %s -> %s in %.0fms",
             side, qty, symbol, price, oid, ms)
    return {"ok": True, "order_id": oid, "status": status,
            "rt_ms": round(ms, 1)}


async def replace(*, order_id: str, symbol: str, side: str, qty: int,
                  price: float, armed: bool, reference: float | None,
                  position_qty: float) -> dict:
    """Reprice. A replace is one call, and it is how the ladder nudge moves."""
    why = _armed_check(armed)
    if why:
        raise BrokerError(why)
    why = check_guards(symbol=symbol, side=side, qty=qty, price=price,
                       reference=reference, position_qty=position_qty)
    if why:
        raise BrokerError(f"refused by the guards: {why}")
    hv = await account_hash()
    data, status, ms = await _acall(
        "PUT", f"/accounts/{hv}/orders/{order_id}",
        body=_equity_order(side, qty, symbol, price))
    oid = (data or {}).get("order_id") if isinstance(data, dict) else None
    return {"ok": True, "order_id": oid or order_id, "status": status,
            "rt_ms": round(ms, 1)}


async def cancel(*, order_id: str) -> dict:
    """PRIORITY, and not gated on arming.

    Cancelling is how a mistake is undone. A disarmed pane that cannot cancel
    would be a pane whose safety switch traps an order, and a cancel refused
    for quota is the specific failure the limiter's reserve exists to stop.
    """
    hv = await account_hash()
    _, status, ms = await _acall("DELETE", f"/accounts/{hv}/orders/{order_id}",
                                 priority=True)
    log.info("cancelled %s in %.0fms", order_id, ms)
    return {"ok": True, "order_id": order_id, "status": status,
            "rt_ms": round(ms, 1)}


async def flatten(*, symbol: str, armed: bool) -> dict:
    """Cancel everything working in the name, then close at market.

    ORDER MATTERS. Closing while a working order rests on the same symbol can
    leave the resting order to open a fresh position in the opposite
    direction the moment the flatten fills. Cancels first, every time.

    Priority throughout: this is the getting-flat path.
    """
    if not config.TRADING_ENABLED:
        raise BrokerError(_armed_check(True))
    sym = symbol.upper()
    st = await state([sym], priority=True)
    out = {"cancelled": [], "rt_ms": 0.0}
    for o in st["working"]:
        if (o["symbol"] or "").upper() != sym or not o["order_id"]:
            continue
        try:
            r = await cancel(order_id=o["order_id"])
            out["cancelled"].append(o["order_id"])
            out["rt_ms"] += r["rt_ms"]
        except BrokerError as exc:
            # Reported, not swallowed: a cancel that failed means the resting
            # order is still there and the flatten below may reopen against it.
            out.setdefault("problems", []).append(
                f"could not cancel {o['order_id']}: {exc}")

    pos = next((p for p in st["positions"]
                if (p["symbol"] or "").upper() == sym), None)
    qty = float(pos["qty"]) if pos else 0.0
    if abs(qty) < 1e-9:
        out["ok"] = True
        out["flat"] = True
        out["note"] = f"no {sym} position to close"
        return out

    side = "SELL" if qty > 0 else "BUY_TO_COVER"
    hv = await account_hash()
    _, status, ms = await _acall(
        "POST", f"/accounts/{hv}/orders",
        body=_equity_order(side, int(abs(qty)), sym, None), priority=True)
    out["ok"] = True
    out["closed"] = {"side": side, "qty": int(abs(qty))}
    out["status"] = status
    out["rt_ms"] = round(out["rt_ms"] + ms, 1)
    log.info("flattened %s: %s %d in %.0fms", sym, side, abs(qty), ms)
    return out


# ── health ──────────────────────────────────────────────────────────────────
def health() -> dict:
    tokens = _read_tokens()
    return {
        "trading_enabled": config.TRADING_ENABLED,
        "token_file": str(_token_file()),
        "have_tokens": tokens is not None,
        "token_expires_in_s": round((tokens or {}).get("expires_at", 0)
                                    - time.time(), 0) if tokens else None,
        "have_credentials": bool(config.SCHWAB_API_KEY
                                 and config.SCHWAB_API_SECRET),
        "last_error": _last_error,
        "limits": LIMITER.state(),
        "guards": {
            "max_order_shares": config.MAX_ORDER_SHARES,
            "max_position_shares": config.MAX_POSITION_SHARES,
            "max_notional": config.MAX_NOTIONAL,
            "max_limit_distance_pct": config.MAX_LIMIT_DISTANCE_PCT,
        },
    }


def problems() -> list[str]:
    out = []
    if not config.SCHWAB_API_KEY or not config.SCHWAB_API_SECRET:
        out.append("SCHWAB_API_KEY / SCHWAB_API_SECRET are not set, so no "
                   "order can be placed. They live in the portfolio "
                   "dashboard's .env; point SCHWAB_ENV_FILE at it.")
    elif not _token_file().is_file():
        out.append(f"no Schwab tokens at {_token_file()} — authorise in the "
                   f"portfolio dashboard, which owns that file.")
    return out
