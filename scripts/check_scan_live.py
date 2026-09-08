"""The scan tier against the real feed, using the real Hub.

WHY THIS IS NOT THE HUB GATE. check_live_hub drives a Hub with a fabricated
message stream, which is the right way to check reference counting and the
caps -- no market needed, and every branch reachable. What it cannot check is
whether the upstream agrees: that a `T.`-only subscription is honoured, that
adding `Q.` to a symbol already subscribed for trades actually starts quotes
rather than being ignored as a duplicate, and that unsubscribing `Q.` alone
leaves the trades flowing.

Every one of those is a server-side behaviour, every one of them is invisible
from a synthetic stream, and the third is the expensive one: if unsubscribing
quotes silently drops the trade channel too, the scan row for that symbol goes
dead -- and a symbol that has stopped printing is indistinguishable from one
that has gone quiet, which is the single thing the grid exists to tell apart.

WHY IT USES live.hub.Hub RATHER THAN ITS OWN CLIENT. The point is to exercise
the shipped code path. A bespoke socket in this file would prove that the FEED
behaves, not that the hub does.

    systemctl stop spx-live
    python scripts/check_scan_live.py --symbols 200 --seconds 120
    systemctl start spx-live

It needs the connection to itself -- the account permits one. See the
reconnect-storm check below, which is what running it alongside spx-live
actually looks like from in here.
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from live import config                                   # noqa: E402
from live.hub import Hub                                  # noqa: E402

FAILS: list[str] = []


def check(cond, msg):
    if not cond:
        FAILS.append(msg)
        print(f"  FAIL  {msg}")
    return bool(cond)


async def seed_symbols(limit: int) -> list[str]:
    """The scan's seed, checked on the way past.

    THE FILTER IS VERIFIED, not trusted. Dropping `not spread_excluded` does
    not fail anything -- it silently puts SPY and QQQ back at the top of the
    list, which is what happened, and it was caught by a person reading
    symbol names rather than by any check. So the unfiltered query is run
    too, and the two are compared: if the filter removes nothing, it is not
    doing its job and this says so before the socket opens.
    """
    import asyncpg
    from live import scan_universe

    con = await asyncpg.connect(scan_universe.scalp_dsn())
    try:
        c = await scan_universe.counts(con)
        print(f"universe: {c['total']} symbols, {c['qualified']} qualified, "
              f"{c['not_excluded']} not spread-excluded, {c['both']} both")
        check(c["both"] < c["total"],
              f"the spread-exclusion filter removed nothing ({c['both']} of "
              f"{c['total']}) — index ETFs the pipeline already excludes "
              f"would be back on the list")
        rows = await scan_universe.seed(con, limit=limit)
        unfiltered = await con.fetch(
            """select symbol from universe
               where trade_date = (select max(trade_date) from universe)
                 and spread_excluded
               order by dollar_volume desc nulls last limit 40""")
    finally:
        await con.close()

    syms = [r["symbol"] for r in rows]
    excluded = {r["symbol"] for r in unfiltered}
    leaked = sorted(excluded.intersection(syms))
    check(not leaked,
          f"spread-excluded names reached the seed list: {leaked[:8]} — the "
          f"scan would hold names with nothing to capture, and they are the "
          f"heaviest on the tape")
    return syms


async def settle(hub: Hub, seconds: float, label: str) -> None:
    """Wait, reporting progress, and fail loudly on a reconnect storm.

    RUNNING THIS WHILE spx-live HOLDS THE CONNECTION does not raise and does
    not return an error. Both processes authenticate, both are accepted, and
    each one's connect evicts the other -- so what it looks like from in here
    is a hub that reconnects every few seconds while collecting a fraction of
    the tape. Left undetected, every number below would be quietly measured
    against a feed this process keeps being thrown off.
    """
    start = hub.reconnects
    until = time.time() + seconds
    while time.time() < until:
        await asyncio.sleep(2.0)
        if hub.reconnects - start > 2:
            print("")
            print("  " + "=" * 70)
            print("  RECONNECT STORM: something else holds the connection")
            print("  " + "=" * 70)
            print(f"    {hub.reconnects - start} reconnects in "
                  f"{seconds - (until - time.time()):.0f}s.")
            print("    The account permits ONE concurrent websocket. Stop the")
            print("    holder for the length of this check:")
            print("      systemctl stop spx-live")
            print("      " + " ".join(sys.argv))
            print("      systemctl start spx-live")
            raise SystemExit(3)
        left = until - time.time()
        st = hub.scan_status()
        print(f"\r  {label}: {left:5.0f}s left, {st['trades_held']:>7} trades "
              f"held across {st['held']} symbols, {st['buffer_mb']:.1f} MB, "
              f"{st['rings_grown']} grows   ", end="", flush=True)
    print("")


async def main() -> int:
    ap = argparse.ArgumentParser(
        description="Exercise the scan tier against the real upstream.")
    ap.add_argument("--symbols", type=int, default=200)
    ap.add_argument("--seconds", type=float, default=120.0,
                    help="accumulation before the rollup is believed; must "
                         "exceed the quiet window plus its step")
    ap.add_argument("--cross-seconds", type=float, default=20.0,
                    help="dwell either side of the pane open/close")
    args = ap.parse_args()

    for p in config.problems():
        print(f"configuration: {p}")
        return 2
    need = config.SCAN_QUIET_WINDOW_S + config.SCAN_QUIET_WINDOW_S / 3.0
    if args.seconds < need:
        print(f"--seconds {args.seconds:.0f} is below the {need:.0f}s the "
              f"quiet ratio needs (a {config.SCAN_QUIET_WINDOW_S:.0f}s window "
              f"plus the step it is differenced against); every ratio would "
              f"be NaN and this would report it as a failure of the code.")
        return 2

    syms = await seed_symbols(args.symbols)
    print(f"feed:  {config.FEED} -> {config.feed_url()}")
    print(f"scan:  {len(syms)} symbols, busiest first: "
          f"{', '.join(syms[:6])} ...")

    hub = Hub()
    runner = asyncio.create_task(hub.run())
    try:
        for _ in range(100):                    # up to ~10s to authenticate
            if hub.connected and hub.authed:
                break
            await asyncio.sleep(0.1)
        if not check(hub.connected and hub.authed,
                     f"the hub did not authenticate "
                     f"(connected={hub.connected} authed={hub.authed} "
                     f"error={hub.last_error})"):
            return 1

        added, _, refused = await hub.scan_set(syms)
        check(len(added) == len(syms),
              f"scan_set took {len(added)} of {len(syms)} symbols: "
              f"{refused[:3]}")

        await settle(hub, args.seconds, "accumulating")

        # ── the tier holds data at all ──────────────────────────────────
        st = hub.scan_status()
        print(f"\n  status: {st['held']} held, {st['trades_held']} trades, "
              f"{st['buffer_mb']:.1f} MB, {st['rings_grown']} grows, "
              f"{st['truncated_count']} truncated")
        check(st["trades_held"] > 0,
              "no trades reached the scan store at all — the tier subscribed "
              "and received nothing, which on the page is a grid of empty rows")

        state = hub.scan_state()
        live_syms = [s for s, v in state.items() if v[3] > 0]
        finite = [s for s, v in state.items() if v[0] == v[0]]
        print(f"  rollup: {len(live_syms)}/{len(state)} symbols printed, "
              f"{len(finite)} produced a finite quiet ratio")
        check(len(live_syms) >= 0.5 * len(state),
              f"only {len(live_syms)} of {len(state)} symbols printed — the "
              f"market may be closed, and none of this is a live check")
        check(len(finite) >= 0.25 * len(state),
              f"only {len(finite)} of {len(state)} produced a finite quiet "
              f"ratio; a grid that is mostly blank is not a grid")

        ratios = np.array([v[0] for v in state.values() if v[0] == v[0]])
        if ratios.size:
            print(f"  ratios: p10 {np.percentile(ratios, 10):.2f}  "
                  f"p50 {np.percentile(ratios, 50):.2f}  "
                  f"p90 {np.percentile(ratios, 90):.2f}  "
                  f"max {ratios.max():.2f}")
            # A COLUMN OF ONE VALUE is the failure that looks like data. If
            # every name scores the same the grid is one colour, which reads
            # as a market with nothing happening in it rather than as a broken
            # rollup.
            check(float(np.std(ratios)) > 0.01,
                  f"every symbol scored the same quiet ratio "
                  f"(sd={np.std(ratios):.4f}) — the grid would render as a "
                  f"single flat colour")

        # ── the tier crossing, which is the point ───────────────────────
        target = max(live_syms, key=lambda s: state[s][3]) if live_syms \
            else syms[0]
        print(f"\n  crossing tiers on {target} "
              f"(busiest of the printing symbols)")

        before_ring = hub.scan[target].n
        err = await hub.acquire(target)
        check(err is None, f"a pane could not open on a scan symbol: {err}")
        await asyncio.sleep(args.cross_seconds)

        quotes = len(hub.quotes.get(target, ()))
        pane_trades = len(hub.trades.get(target, ()))
        ring_during = hub.scan[target].n
        print(f"  pane open:  {quotes} quotes, {pane_trades} pane trades, "
              f"ring +{ring_during - before_ring}")
        check(quotes > 0,
              f"opening a pane on a scan-held symbol produced NO quotes — the "
              f"upstream ignored the Q-only upgrade, so the pane draws a tape "
              f"with no NBBO and it reads as a thin book")
        check(ring_during > before_ring,
              "the scan ring stopped filling while a pane was open")

        await hub.release(target)
        ring_at_release = hub.scan[target].n
        await asyncio.sleep(args.cross_seconds)
        ring_after = hub.scan[target].n
        quotes_after = len(hub.quotes.get(target, ()))
        print(f"  pane closed: ring +{ring_after - ring_at_release} since "
              f"release, quote buffer {quotes_after}")
        # THE EXPENSIVE ONE. If the Q-only unsubscribe took the trade channel
        # with it, this is where it shows, and nowhere else.
        check(ring_after > ring_at_release,
              f"the scan ring STOPPED filling after the pane closed "
              f"({ring_at_release} -> {ring_after}) — unsubscribing quotes "
              f"took the trade channel with it, and the row now looks quiet "
              f"rather than dead")
        check(target in hub.scan,
              "the scan lost the symbol when the pane closed")
    finally:
        await hub.stop()
        runner.cancel()
        try:
            await runner
        except asyncio.CancelledError:
            pass

    print(f"\n  scan-live checks: {'PASS' if not FAILS else str(len(FAILS)) + ' FAILED'}")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
