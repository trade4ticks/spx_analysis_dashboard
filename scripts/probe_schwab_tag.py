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

WHEN THE TAGGED ORDER IS REJECTED. Schwab answers a bad order body with a
flat 400 "A validation error occurred while processing the request." and
names no field, so a rejection on its own does not say the tag caused it.
`--no-tag` sends the byte-identical body with the `tag` key removed and
nothing else changed, which is the control:

    python scripts/probe_schwab_tag.py --symbol FDX --price 250.00 --yes --no-tag

Accepted without the tag and rejected with it means the tag is not settable,
and reconciliation stays heuristic for good. Rejected both ways means the tag
is exonerated and the fault is elsewhere in the body — session, duration,
price format, assetType — and the next probe changes one of those instead.

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


async def run(symbol: str, price: float, side: str, use_tag: bool) -> int:
    hv = await broker.account_hash()
    body = {
        "session": "NORMAL",
        "duration": "DAY",
        "orderStrategyType": "SINGLE",
        "orderType": "LIMIT",
        "price": f"{price:.2f}",
        "orderLegCollection": [{
            "instruction": side,
            "quantity": 1,
            "instrument": {"symbol": symbol, "assetType": "EQUITY"},
        }],
    }
    if use_tag:
        # THE WHOLE QUESTION. Everything else here is the smallest possible
        # order that can carry it, and --no-tag sends exactly this body
        # without the key so the two runs differ in one thing only.
        body["tag"] = TAG

    print(f"mode: {'WITH tag' if use_tag else 'WITHOUT tag (control run)'}")
    print("about to send:\n" + json.dumps(body, indent=2))
    print()

    sent_at = time.time()
    order_id = None
    try:
        try:
            data, status, ms = await broker._acall(
                "POST", f"/accounts/{hv}/orders", body=body, priority=True)
        except broker.BrokerIndeterminate:
            # MUST COME FIRST: it subclasses BrokerError, and reporting a
            # timeout as a rejection would be the reverse of the truth. A
            # rejection leaves nothing behind; this may have left a live
            # order that this script has no id to cancel.
            print("\n*** NOT A REJECTION — the request timed out or the "
                  "connection failed. Whether the order reached Schwab is "
                  "UNKNOWN, and there is no id here to cancel it with.")
            print("*** CHECK THE WORKING ORDERS IN THINKORSWIM NOW.")
            raise
        except broker.BrokerError as exc:
            # A 4xx is Schwab reading the body and refusing it, and the whole
            # reason --no-tag exists is that its message names no field. The
            # rejection IS the result here, so it is reported as one rather
            # than raised as a traceback that says nothing more.
            return _rejected(str(exc), use_tag)
        order_id = (data or {}).get("order_id") if isinstance(data, dict) else None
        print(f"POST  -> {status} in {ms:.0f}ms, order id {order_id}")
        if not use_tag:
            print("\n  ACCEPTED WITHOUT THE TAG. Hold that thought until the "
                  "read-back below; the verdict needs the tagged run too.")
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
        print(f"tag sent     : {repr(TAG) if use_tag else '(none — control run)'}")
        print(f"tag returned : {echoed!r}")
        print()
        if not use_tag:
            # The control run cannot answer "is the tag settable"; it answers
            # "is the rest of the body valid", and that is the whole point of
            # running it. What Schwab stamps on an untagged order is recorded
            # because it is free to look at and bears on whether a stamped
            # value could distinguish this app's orders from thinkorswim's.
            print("THE REST OF THE BODY IS VALID. This exact order minus the "
                  "`tag` key was accepted, so session, duration, price format "
                  "and the leg are all fine as written.")
            print()
            if echoed:
                print(f"Schwab stamped {echoed!r} of its own accord on an "
                      f"order that carried no tag.")
            else:
                print("Schwab stamped no tag of its own on it.")
            print()
            print("If the tagged run 400s against this same body, the tag is "
                  "the cause and it is NOT settable: reconciliation after a "
                  "timeout stays a match on (symbol, side, quantity, price, "
                  "entered after we sent), and match_placement keeps its "
                  "caveat permanently.")
            rc = 0
        elif echoed == TAG:
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


def _rejected(msg: str, use_tag: bool) -> int:
    """Schwab refused the body. Say what that does and does not establish."""
    print(f"POST  -> REJECTED: {msg}")
    print("\n" + "=" * 64)
    if use_tag:
        print("REJECTED WITH THE TAG. On its own this proves nothing about "
              "the tag: Schwab's validation message names no field, so the "
              "fault could be anywhere in the body.")
        print()
        print("Run the control — the same command with --no-tag, which sends "
              "this body with the `tag` key removed and nothing else changed:")
        print("  * accepted without it  -> the tag is the cause and is not "
              "settable")
        print("  * rejected without it  -> the tag is exonerated and the "
              "fault is elsewhere in the body")
        return 3
    print("REJECTED WITHOUT THE TAG. The tag is EXONERATED — this body has no "
          "tag in it and Schwab still refused it, so whatever it objects to "
          "is one of the other fields: session, duration, orderStrategyType, "
          "the price format (a string here), or the leg's assetType.")
    print()
    print("Change ONE of those and re-run this control before touching the "
          "tag question again; the tagged run cannot mean anything while the "
          "untagged one fails.")
    return 4


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
    ap.add_argument("--no-tag", action="store_true",
                    help="The control run: send this identical body with the "
                         "`tag` key removed. If it is accepted and the tagged "
                         "one was not, the tag is what Schwab rejected.")
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
    return asyncio.run(run(a.symbol.upper(), a.price, a.side, not a.no_tag))


sys.exit(main())
