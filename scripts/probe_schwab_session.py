"""Which `session` values does Schwab accept, and does SEAMLESS change routing?

TWO QUESTIONS, NEITHER ANSWERABLE BY READING. The order body sends
`"session": "NORMAL"` and that is known to work because orders have gone out
with it. Whether SEAMLESS, AM and PM are accepted — and whether any of them
needs something else on the body — cannot be settled from here: a bad body
comes back as 400 "A validation error occurred while processing the request."
naming no field, exactly as the `tag` probe found. So each value is sent, on
a body identical in every other respect, and the answers compared.

THE ROUTING QUESTION MATTERS MORE. Extended-hours orders typically route to a
specific ECN rather than to the usual destination. If a SEAMLESS order routes
that way DURING the regular session, then turning the toggle on would quietly
worsen every ordinary fill — which would make it a switch to leave off and
label, not a convenience.

That is not visible in the request. It is visible in the record Schwab keeps:
an order carries `requestedDestination` and `destinationLinkName`, and this
places a NORMAL order and a SEAMLESS one, one after the other, on the same
symbol at the same price, and prints every field on which the two differ. If
routing changes, it shows up there. If the two records are identical apart
from `session`, that is the answer too — as far as the record reports it,
which is the limit of what this can establish.

    python scripts/probe_schwab_session.py --symbol FDX --price 250.00 \\
        --reference 318.40 --yes

THIS PLACES REAL ORDERS — one share each, as LIMITs, at a price you supply,
cancelled in a `finally`. Same refusals as the other probes: a price within
3% of `--reference` is refused because it could fill. Run it DURING THE
REGULAR SESSION: the routing question is specifically about what a seamless
order does while the normal market is open.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from live import broker                                     # noqa: E402

# Schwab documents these four. All are sent; the ones that are refused are
# the finding, and so is any that needs a field this body does not carry.
SESSIONS = ["NORMAL", "SEAMLESS", "AM", "PM"]

# Fields worth diffing between a NORMAL and a SEAMLESS order. Anything else
# that differs is printed too — the list is what gets looked at FIRST, not a
# filter.
ROUTING_FIELDS = ["requestedDestination", "destinationLinkName", "session",
                  "duration", "orderType", "status", "complexOrderStrategyType"]


async def place(hv: str, symbol: str, side: str, price: float,
                session: str) -> tuple[str | None, dict | None, str | None]:
    """One order with this session. Returns (order_id, raw record, error)."""
    body = broker._equity_order(side, 1, symbol, price)
    body["session"] = session
    try:
        data, status, ms = await broker._acall(
            "POST", f"/accounts/{hv}/orders", body=body, priority=True)
    except broker.BrokerIndeterminate:
        print(f"  {session:9} TIMED OUT — whether it reached Schwab is "
              f"UNKNOWN. Check the working orders before running this again.")
        raise
    except broker.BrokerError as exc:
        print(f"  {session:9} REJECTED: {exc}")
        return None, None, str(exc)
    oid = (data or {}).get("order_id") if isinstance(data, dict) else None
    print(f"  {session:9} accepted -> {status} in {ms:.0f}ms, id {oid}")
    if not oid:
        return None, None, "accepted but no order id in the Location header"

    # The RAW record: the normaliser keeps none of the routing fields, and
    # they are the whole point of the second question.
    raw = None
    for _ in range(12):
        await asyncio.sleep(0.5)
        rows, _, _ = await broker._acall(
            "GET", f"/accounts/{hv}/orders",
            params={"fromEnteredTime": _fmt(time.time() - 300),
                    "toEnteredTime": _fmt(time.time() + 60)}, priority=True)
        raw = next((o for o in (rows or [])
                    if str(o.get("orderId")) == str(oid)), None)
        if raw:
            break
    if raw is None:
        print(f"  {session:9} accepted but could not be read back in 6s")
    return oid, raw, None


async def run(symbol: str, price: float, side: str) -> int:
    hv = await broker.account_hash()
    placed: list[str] = []
    records: dict[str, dict] = {}
    errors: dict[str, str] = {}
    try:
        print("Sending one order per session value, identical otherwise:\n")
        for i, sess in enumerate(SESSIONS):
            # A different price per value so the read-back cannot confuse two
            # of them, and so none of them sits on top of another in the book.
            px = round(price - i * 0.01, 2)
            oid, raw, err = await place(hv, symbol, side, px, sess)
            if oid:
                placed.append(oid)
            if raw:
                records[sess] = raw
            if err:
                errors[sess] = err
            # Cancel as we go: fewer live orders at any moment is fewer to
            # get wrong if this dies halfway.
            if oid:
                try:
                    await broker.cancel(order_id=oid)
                    placed.remove(oid)
                except Exception as exc:                    # noqa: BLE001
                    print(f"    *** could not cancel {oid}: {exc}")

        print("\n" + "=" * 68)
        print("ACCEPTED:", ", ".join(s for s in SESSIONS if s in records)
              or "(none)")
        print("REJECTED:", ", ".join(f"{s}" for s in errors) or "(none)")
        for s, e in errors.items():
            print(f"  {s}: {e}")

        # ── what Schwab actually stored ──────────────────────────────────
        print("\nWhat came back, per accepted value:")
        for s, raw in records.items():
            got = raw.get("session")
            flag = "" if got == s else f"   <-- REWRITTEN, sent {s!r}"
            print(f"  {s:9} session={got!r}{flag}")
            for f in ROUTING_FIELDS:
                if f == "session":
                    continue
                if f in raw:
                    print(f"            {f} = {raw.get(f)!r}")

        # ── the routing question ─────────────────────────────────────────
        print("\n" + "=" * 68)
        if "NORMAL" in records and "SEAMLESS" in records:
            a, b = records["NORMAL"], records["SEAMLESS"]
            diffs = _diff(a, b)
            routing = [k for k in diffs
                       if k in ("requestedDestination", "destinationLinkName")]
            if routing:
                print("ROUTING DIFFERS between NORMAL and SEAMLESS:")
                for k in routing:
                    print(f"  {k}: NORMAL={a.get(k)!r}  SEAMLESS={b.get(k)!r}")
                print()
                print("  So a seamless order does NOT route like a normal one "
                      "during the regular session. The toggle should default "
                      "off, say this on its label, and not persist across a "
                      "reload — leaving it on would quietly change where "
                      "every ordinary order goes.")
            else:
                print("ROUTING IS THE SAME on both, as far as the record "
                      "reports it:")
                for k in ("requestedDestination", "destinationLinkName"):
                    print(f"  {k}: {a.get(k)!r}")
                print()
                print("  Nothing here says a seamless order is handled "
                      "differently while the regular market is open. That is "
                      "the record's answer, not the venue's — it does not "
                      "prove the FILLS are identical, only that Schwab "
                      "stored the same destination.")
            other = [k for k in diffs if k not in routing and k not in (
                "orderId", "enteredTime", "price", "closeTime", "tag",
                "orderLegCollection", "cancelTime", "statusDescription")]
            if other:
                print("\n  Other fields that differ (worth an eye):")
                for k in other:
                    print(f"    {k}: NORMAL={a.get(k)!r}  SEAMLESS={b.get(k)!r}")
        else:
            print("Cannot compare routing: need BOTH a NORMAL and a SEAMLESS "
                  "order to have been accepted, and they were not.")
        return 0 if "SEAMLESS" in records else 1
    finally:
        for oid in list(placed):
            try:
                r = await broker.cancel(order_id=oid)
                print(f"\ncancelled {oid} in {r['rt_ms']:.0f}ms")
            except Exception as exc:                        # noqa: BLE001
                print(f"\n  *** COULD NOT CANCEL {oid}: {exc}")
                print("  *** Cancel it in thinkorswim or at schwab.com NOW.")
        await broker.aclose()


def _diff(a: dict, b: dict) -> list[str]:
    keys = set(a) | set(b)
    return sorted(k for k in keys
                  if json.dumps(a.get(k), default=str, sort_keys=True)
                  != json.dumps(b.get(k), default=str, sort_keys=True))


def _fmt(ts: float) -> str:
    from datetime import datetime, timezone
    return datetime.fromtimestamp(ts, timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%S.000Z")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", required=True)
    ap.add_argument("--price", type=float, required=True,
                    help="A LIMIT far from the market so it rests rather than "
                         "fills. Each value is sent a cent below the last.")
    ap.add_argument("--side", default="BUY", choices=["BUY", "SELL"])
    ap.add_argument("--reference", type=float, default=None,
                    help="The current price, so the refusal below has "
                         "something to check against.")
    ap.add_argument("--yes", action="store_true",
                    help="Required. This places real orders.")
    a = ap.parse_args()

    if not a.yes:
        print(f"This places {len(SESSIONS)} REAL one-share orders, one per "
              f"session value, cancelling each as it goes. Re-run with --yes.")
        return 2
    if a.reference:
        if a.side == "BUY" and a.price > a.reference * 0.97:
            print(f"refusing: a buy at {a.price} against a reference of "
                  f"{a.reference} is within 3% and could fill. Go lower.")
            return 2
        if a.side == "SELL" and a.price < a.reference * 1.03:
            print(f"refusing: a sell at {a.price} against a reference of "
                  f"{a.reference} is within 3% and could fill. Go higher.")
            return 2
    return asyncio.run(run(a.symbol.upper(), a.price, a.side))


sys.exit(main())
