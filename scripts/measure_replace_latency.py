"""Where does the time go when a resting order is repriced? Measure it.

THE QUESTION. Moving an order feels slow while `rt` on the page reads ~68ms.
Those are both true, because they are not measuring the same thing. A move is
two legs and the screen only ever shows the first one before it is
overwritten:

    PUT /accounts/{hash}/orders/{id}     the replace itself
    GET /accounts/{hash}/orders          the read-back the marker waits for

and there is a THIRD term that neither leg reports: how long Schwab takes to
show the new price in the order list at all. The read-back can return
promptly and still carry the OLD price, in which case the marker does not
move on the read-back either — it waits for a later poll. That term is
invisible from inside the app and it is the one worth knowing before adding
optimistic rendering to hide it.

This measures all three against a real order:

    1. place a resting limit far from the market
    2. PUT a new price and time the call
    3. poll the order list every 150ms until it reports the NEW price,
       timing every read
    4. cancel, in a `finally`

    python scripts/measure_replace_latency.py --symbol FDX --price 250.00 \\
        --reference 318.40 --yes

THIS PLACES REAL ORDERS — one share, as a LIMIT, at a price you supply, and
it refuses a price within 3% of `--reference` because that could fill. Same
shape and same refusals as scripts/probe_schwab_tag.py, and for the same
reason: the deliberate act is running it.
"""
from __future__ import annotations

import argparse
import asyncio
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from live import broker                                     # noqa: E402

POLL_S = 0.15
GIVE_UP_S = 15.0


async def run(symbol: str, price: float, side: str, moves: int) -> int:
    hv = await broker.account_hash()
    order_id = None
    try:
        # ── 1. a resting order to move ───────────────────────────────────
        body = broker._equity_order(side, 1, symbol, price)
        t0 = time.perf_counter()
        data, status, ms = await broker._acall(
            "POST", f"/accounts/{hv}/orders", body=body, priority=True)
        order_id = (data or {}).get("order_id") if isinstance(data, dict) else None
        print(f"placed {symbol} {side} 1 @ {price:.2f} -> {status} in "
              f"{ms:.0f}ms, id {order_id}")
        if not order_id:
            print("  no order id in the Location header; cannot move it.")
            return 2

        # It has to be visible before it can be moved.
        appeared = await _wait_for(hv, order_id, price, "the placement")
        if appeared is None:
            return 2
        print()

        puts, waits, reads = [], [], []
        px = price
        for i in range(moves):
            px = round(px + 0.01, 2)
            t_send = time.perf_counter()
            _, st, put_ms = await broker._acall(
                "PUT", f"/accounts/{hv}/orders/{order_id}",
                body=broker._equity_order(side, 1, symbol, px), priority=True)
            puts.append(put_ms)
            # A REPLACE MAKES A NEW ORDER. Schwab gives no link from the old
            # id to the new one, so the poll below matches on the PRICE and
            # picks up whatever id now carries it.
            got = await _wait_for(hv, None, px, f"move {i + 1} -> {px:.2f}",
                                  since=t_send, put_ms=put_ms)
            if got is None:
                return 2
            new_id, total_s, read_ms = got
            order_id = new_id
            waits.append(total_s * 1000.0)
            reads.extend(read_ms)

        print()
        print("=" * 66)
        print(f"{'':22}{'median':>10}{'min':>10}{'max':>10}")
        _row("PUT (the replace)", puts)
        _row("GET (one read-back)", reads)
        _row("click -> new price", waits)
        print()
        med_put = statistics.median(puts)
        med_read = statistics.median(reads)
        med_wait = statistics.median(waits)
        # The part that is NOT either call: Schwab's own propagation.
        propagation = med_wait - med_put - med_read
        print(f"So of the {med_wait:.0f}ms before the marker can move:")
        print(f"  {med_put:6.0f}ms  the PUT — what `rt` shows, then loses")
        print(f"  {med_read:6.0f}ms  ONE read-back")
        print(f"  {propagation:6.0f}ms  Schwab, between accepting the replace "
              f"and reporting it")
        if propagation > med_read:
            print()
            print("  THE PROPAGATION DOMINATES. That time cannot be polled "
                  "away: reading sooner just reads the old price again, and "
                  "a faster read-back would not move the marker any earlier. "
                  "Optimistic rendering is the only thing that hides it.")
        elif propagation < 0:
            print()
            print("  The new price was already there on the first read, so "
                  "the wait IS the read-back and nothing else.")
        return 0
    finally:
        if order_id:
            try:
                r = await broker.cancel(order_id=order_id)
                print(f"\ncancelled {order_id} in {r['rt_ms']:.0f}ms")
            except Exception as exc:                        # noqa: BLE001
                print(f"\n  *** COULD NOT CANCEL {order_id}: {exc}")
                print("  *** Cancel it in thinkorswim or at schwab.com NOW.")
        await broker.aclose()


def _row(label: str, xs: list[float]) -> None:
    print(f"{label:22}{statistics.median(xs):9.0f}ms{min(xs):9.0f}ms"
          f"{max(xs):9.0f}ms")


async def _wait_for(hv: str, order_id: str | None, price: float, what: str,
                    since: float | None = None, put_ms: float = 0.0):
    """Poll until the order list reports `price`. Returns (id, secs, reads)."""
    t0 = since if since is not None else time.perf_counter()
    reads: list[float] = []
    while time.perf_counter() - t0 < GIVE_UP_S:
        st = await broker.read_orders([_SYM], priority=True)
        reads.append(st["rt_ms"])
        for o in st["working"]:
            if order_id is not None and str(o["order_id"]) != str(order_id):
                continue
            if o["price"] is None or abs(float(o["price"]) - price) > 0.005:
                continue
            secs = time.perf_counter() - t0
            print(f"  {what}: visible at {price:.2f} after {secs * 1000:.0f}ms"
                  f" (PUT {put_ms:.0f}ms, {len(reads)} read"
                  f"{'s' if len(reads) != 1 else ''})")
            return str(o["order_id"]), secs, reads
        await asyncio.sleep(POLL_S)
    print(f"  {what}: NOT visible within {GIVE_UP_S:.0f}s — that is itself "
          f"the answer, and it is worse than the one this was looking for.")
    return None


_SYM = ""


def main() -> int:
    global _SYM
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", required=True)
    ap.add_argument("--price", type=float, required=True,
                    help="A LIMIT far from the market so it rests. For a buy, "
                         "well below. Each move walks it a cent further.")
    ap.add_argument("--side", default="BUY", choices=["BUY", "SELL"])
    ap.add_argument("--moves", type=int, default=5,
                    help="How many replaces to time. The median of five is "
                         "worth more than one sample.")
    ap.add_argument("--reference", type=float, default=None,
                    help="The current price, so the refusal below has "
                         "something to check against.")
    ap.add_argument("--yes", action="store_true")
    a = ap.parse_args()

    if not a.yes:
        print("This places a REAL one-share order and moves it. Re-run with "
              "--yes.")
        return 2
    if a.moves < 1 or a.moves > 20:
        print("--moves outside 1..20; twenty replaces is already 20 of the "
              "minute's order quota.")
        return 2
    # A buy ABOVE the reference, or a sell below it, is marketable. The walk
    # moves the price UP a cent per move, so the ceiling is checked against
    # where it ENDS, not where it starts.
    if a.reference:
        end = a.price + 0.01 * a.moves
        if a.side == "BUY" and end > a.reference * 0.97:
            print(f"refusing: a buy walked to {end:.2f} against a reference "
                  f"of {a.reference} is within 3% and could fill. Go lower.")
            return 2
        if a.side == "SELL" and a.price < a.reference * 1.03:
            print(f"refusing: a sell at {a.price} against a reference of "
                  f"{a.reference} is within 3% and could fill. Go higher.")
            return 2
    _SYM = a.symbol.upper()
    return asyncio.run(run(_SYM, a.price, a.side, a.moves))


sys.exit(main())
