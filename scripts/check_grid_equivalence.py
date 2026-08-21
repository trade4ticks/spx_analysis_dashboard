#!/usr/bin/env python3
"""Gate: does the vectorised grid agree with build_combine_sql, exactly?

Why this exists
---------------
/api/factor-trades/grid used to issue one SQL query per combination. It now
issues ONE query for the union of exit columns the sweep touches and resolves
every combination in numpy. That is a SECOND IMPLEMENTATION of the combine
convention -- LEAST over exit bars, side-priority tie-break, structural
horizon backstop -- and a second implementation can drift from the first
without anything failing.

`build_combine_sql` (vendored from Open_Interest) is the oracle. It stays.
This runs both paths over a real zone and asserts they agree per trade on
(exit_bar, exit_return, exit_rule).

BIT-IDENTICAL, NOT APPROXIMATELY EQUAL. Both paths read the same stored
columns, so there is no legitimate source of small differences and no
tolerance is applied. A tolerance here would hide exactly the class of bug
this exists to catch: a mis-transcribed tie-break shows up as the wrong
rule's return, which is a plausible number, not a nearby one.

Usage
-----
    python scripts/check_grid_equivalence.py --sweep fixed_stop,fixed_target
    python scripts/check_grid_equivalence.py --sweep atr_stop,atr_target,max_days \
        --cells "[[10,10],[11,10]]" --verify 12

Needs DATABASE_URL and OI_DATABASE_URL. Exit 0 = every checked combination
matched. Exit 1 = a mismatch, printed per trade with both values.

The same check is reachable from the UI by posting verify=N to /grid, which
is the cheap way to run it on whatever zone is on screen.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


async def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sweep", required=True,
                    help="comma-separated families, e.g. fixed_stop,fixed_target")
    ap.add_argument("--primary", default=None, help="primary metric (default: first available)")
    ap.add_argument("--secondary", default=None)
    ap.add_argument("--cells", default="[[10,0]]", help="JSON [[bp,bs],...]")
    ap.add_argument("--anchor", default="open")
    ap.add_argument("--max-strike", type=float, default=1000.0)
    ap.add_argument("--verify", type=int, default=0,
                    help="combinations to diff (0 = all of them)")
    a = ap.parse_args()

    if not (os.getenv("DATABASE_URL") and os.getenv("OI_DATABASE_URL")):
        print("SKIPPED — needs both DATABASE_URL and OI_DATABASE_URL.")
        print("  Without them the pool is stubbed and /grid answers "
              "'OI database not configured', which would pass vacuously.")
        return 0

    from app.db import init_pool, close_pool, get_oi_pool
    from app.routers.factor_trades import grid, GridReq

    await init_pool()
    try:
        pool = await get_oi_pool()
        primary = a.primary
        if not primary:
            async with pool.acquire() as conn:
                r = await conn.fetchrow(
                    "SELECT column_name FROM information_schema.columns "
                    "WHERE table_name='tt_bins' AND column_name LIKE 'bin20_%' "
                    "ORDER BY column_name LIMIT 1")
            if not r:
                print("FAIL — tt_bins has no bin20_ columns to sweep against.")
                return 1
            primary = r["column_name"][len("bin20_"):]
            print(f"  primary metric (auto): {primary}")

        req = GridReq(
            primary_metric=primary,
            secondary_metric=a.secondary,
            entry_anchor=a.anchor,
            rule_keys=[],
            n_bins=20,
            max_strike=a.max_strike,
            window="train",
            cells=json.loads(a.cells),
            sweep_families=[x.strip() for x in a.sweep.split(",") if x.strip()],
            # 0 means "all of them" here, unlike the endpoint's default.
            verify=a.verify or 10_000,
        )
        out = await grid(req, pool=pool)
    finally:
        await close_pool()

    if out.get("error") and not out.get("verify"):
        print(f"FAIL — {out['error']}")
        return 1

    v = out.get("verify") or {}
    print(f"  combinations checked : {v.get('combinations_checked')} "
          f"of {v.get('of')}")
    print(f"  trades compared      : {v.get('trades_compared'):,}")
    print(f"  columns fetched      : {out.get('n_columns')} (1 query)")
    print(f"  mismatches           : {v.get('n_mismatches', 0)}")

    if v.get("mismatches"):
        print()
        print("MISMATCHES — the vectorised path disagrees with the oracle:")
        for m in v["mismatches"]:
            print(f"    combo {m.get('combo')} {m.get('trade','')} "
                  f"{m.get('field')}: sql={m.get('sql')!r} "
                  f"numpy={m.get('numpy')!r}")
        print()
        print("These are NOT rounding. Both paths read the same stored "
              "columns; a difference is a mis-transcription of LEAST or of "
              "the side-priority tie-break.")
        return 1

    print()
    print("vectorised grid == build_combine_sql on every checked combination")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
