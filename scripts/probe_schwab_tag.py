"""Does Schwab echo a client-supplied `tag` on an order? Place one and look.

WHY THIS MATTERS. When a placement times out, the caller does not know
whether the order landed. Without an identifier of our own the only way to
find out is to match on (symbol, side, quantity, price, entered after we
sent) — which is near-unique in practice and NOT guaranteed, because two
identical orders seconds apart are indistinguishable. Your own history has
orders four seconds apart at adjacent prices, so that is not a hypothetical.

If Schwab accepts and echoes a `tag`, it becomes an exact reconciliation key
and the ambiguity disappears. Every order in your account currently carries
`tag: "API_TOS:AT_LADDER_AS"`, which thinkorswim's ladder set — so the field
is real and a client populates it. Whether the REST API lets a caller set it
is the open question, and it cannot be answered by reading. It has to be
placed.

THIS PLACES A REAL ORDER. One share, as a LIMIT, at a price you supply, and
it cancels in a `finally` so it goes away even if this script raises. Choose
a price far below the market for a BUY so it cannot fill — the script refuses
to send anything it thinks could execute, but the price is yours to pick and
the responsibility with it.

    python scripts/probe_schwab_tag.py --symbol FDX --price 250.00 --yes

Deliberately NOT routed through broker.place(): that path is gated on
LIVE_TRADING_ENABLED, on the runtime flag and on the pane's arm flag, all of
which exist to stop an order leaving by accident. Running this script IS the
deliberate act, and weakening those gates so a probe could use them would be
the wrong trade. The order body is built here, in the open, where you can
read exactly what will be sent.
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

TAG = "PBL_LIVE_PROBE"


async def run(symbol: str, price: float, side: str) -> int:
    hv = await broker.account_hash()
    body = {
        "session": "NORMAL",
        "duration": "DAY",
        "orderStrategyType": "SINGLE",
        "orderType": "LIMIT",
        "price": f"{price:.2f}",
        # THE WHOLE QUESTION. Everything else here is the smallest possible
        # order that can carry it.
        "tag": TAG,
        "orderLegCollection": [{
            "instruction": side,
            "quantity": 1,
            "instrument": {"symbol": symbol, "assetType": "EQUITY"},
        }],
    }
    print("about to send:\n" + json.dumps(body, indent=2))
    print()

    sent_at = time.time()
    order_id = None
    try:
        data, status, ms = await broker._acall(
            "POST", f"/accounts/{hv}/orders", body=body, priority=True)
        order_id = (data or {}).get("order_id") if isinstance(data, dict) else None
        print(f"POST  -> {status} in {ms:.0f}ms, order id {order_id}")
        if not order_id:
            print("\n  no order id came back in the Location header. That is "
                  "worth knowing on its own: the placement path depends on "
                  "it, and without it a timeout cannot be reconciled by id "
                  "even in the success case.")

        # Schwab needs a moment before an order appears in the list. Poll,
        # rather than sleeping a guessed amount and concluding from one look.
        found = None
        for attempt in range(10):
            await asyncio.sleep(1.0)
            st = await broker.state([symbol], priority=True)
            pool = st["working"] + st["recent"]
            found = next((o for o in pool if o["order_id"] == order_id), None)
            if found:
                print(f"  appeared in the order list after "
                      f"{attempt + 1}s")
                break
        else:
            print("  did NOT appear within 10s — that is itself a finding, "
                  "and it sets the floor for how long a timeout "
                  "reconciliation has to keep looking.")

        # The raw record, because the normaliser drops `tag` and this is the
        # one field being asked about.
        raw, _, _ = await broker._acall(
            "GET", f"/accounts/{hv}/orders",
            params={"fromEnteredTime": _fmt(sent_at - 60),
                    "toEnteredTime": _fmt(time.time() + 60)},
            priority=True)
        mine = next((o for o in (raw or [])
                     if str(o.get("orderId")) == str(order_id)), None)

        print("\n" + "=" * 64)
        if mine is None:
            print("INCONCLUSIVE — the order was not found to read back.")
            return 2
        echoed = mine.get("tag")
        print(f"tag sent     : {TAG!r}")
        print(f"tag returned : {echoed!r}")
        print()
        if echoed == TAG:
            print("SETTABLE. Reconciliation after a timeout can key on an "
                  "identifier we chose, and the two-identical-orders "
                  "ambiguity goes away. Put a per-placement unique tag in "
                  "the order body and match on it.")
            rc = 0
        elif echoed:
            print(f"REWRITTEN. Schwab replaced the tag with {echoed!r}, so it "
                  f"is not ours to use as a key. Reconciliation stays "
                  f"heuristic — but if that value is stable it still "
                  f"distinguishes orders from THIS app from ones placed in "
                  f"thinkorswim, which is worth having.")
            rc = 1
        else:
            print("DROPPED. The tag did not survive. Reconciliation after a "
                  "timeout has to match on (symbol, side, quantity, price, "
                  "entered after we sent), and the ambiguity of two "
                  "identical orders seconds apart is real and permanent. "
                  "The caveat belongs at that matching code.")
            rc = 1

        print("\nother fields that came back, for the record:")
        for k in ("status", "cancelable", "editable", "enteredTime",
                  "orderId", "quantity", "price"):
            print(f"  {k:12} {mine.get(k)!r}")
        return rc
    finally:
        # ALWAYS. Even if the read-back raised, even on Ctrl-C.
        if order_id:
            try:
                r = await broker.cancel(order_id=order_id)
                print(f"\ncancelled {order_id} in {r['rt_ms']:.0f}ms")
            except Exception as exc:                        # noqa: BLE001
                print(f"\n  *** COULD NOT CANCEL {order_id}: {exc}")
                print("  *** Cancel it in thinkorswim or at schwab.com NOW.")
        await broker.aclose()


def _fmt(ts: float) -> str:
    from datetime import datetime, timezone
    return datetime.fromtimestamp(ts, timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%S.000Z")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", required=True)
    ap.add_argument("--price", type=float, required=True,
                    help="A LIMIT far from the market so it rests rather "
                         "than fills. For a buy, well below.")
    ap.add_argument("--side", default="BUY", choices=["BUY", "SELL"])
    ap.add_argument("--reference", type=float, default=None,
                    help="The current price, if you want the refusal check "
                         "below to have something to check against.")
    ap.add_argument("--yes", action="store_true",
                    help="Required. This places a real order.")
    a = ap.parse_args()

    if not a.yes:
        print("This places a REAL one-share order. Re-run with --yes.")
        return 2
    # A cheap guard on the obvious mistake: a buy ABOVE the reference, or a
    # sell below it, is marketable and would fill.
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
