#!/usr/bin/env python3
"""Recompute stored stats for saved signals against the current is_bins.

Why this is needed
------------------
Signal stats are written once at save time and never again unless someone
asks. is_bins is IN-SAMPLE: its edges are full-history quantiles, so the
daily pipeline extending it does not merely add dates -- it re-derives the
boundaries and RE-LABELS every historical row. Stored stats therefore drift
from the moment they are written, and the signals list shows numbers that
are wrong today and further wrong tomorrow.

Relationship to the existing endpoint
-------------------------------------
POST /signals/refresh already does this, and the Saved Signals table has a
button wired to it (refreshSelectedSignals, oi_analysis.js:5724). But it
only refreshes the ids you have ticked, which makes "keep everything
current" a manual chore that has to be remembered. This script is the
all-signals form, runnable from cron.

It uses the SAME formula as _compute_signal_stats (oi_analysis.py:8945) --
same cell-index arithmetic, same n-weighted aggregate, same overnight_gap
special case -- so a signal refreshed here and one refreshed through the
endpoint land on identical numbers.

Usage (VPS, project root, venv active):
    python scripts/refresh_signal_stats.py --dry-run     # show the drift
    python scripts/refresh_signal_stats.py               # write it
    python scripts/refresh_signal_stats.py --ids 3 7 21  # a subset

--dry-run reports what WOULD change and writes nothing. Run it first: the
drift itself is information about how much a saved zone has moved.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(_ROOT / ".env")

import asyncpg  # noqa: E402

_SAFE = set("abcdefghijklmnopqrstuvwxyz_0123456789")
_GAP = "overnight_gap"


def _safe(*names: str) -> bool:
    return all(n and all(c in _SAFE for c in n) for n in names)


async def _stats(conn, prim, sec, outcome, n_bins, cells):
    """Byte-for-byte the same computation as _compute_signal_stats()."""
    if not cells or n_bins not in (3, 5, 10, 20):
        return None
    if not _safe(prim, sec) or (outcome != _GAP and not _safe(outcome)):
        return None
    xs = [int(c[0]) for c in cells]
    ys = [int(c[1]) for c in cells]
    if outcome == _GAP:
        expr = "AVG(df.ret_1d_fwd_cc - df.ret_1d_fwd_oc)"
        filt = "df.ret_1d_fwd_cc IS NOT NULL AND df.ret_1d_fwd_oc IS NOT NULL"
    else:
        expr = f"AVG(df.{outcome})"
        filt = f"df.{outcome} IS NOT NULL"
    sql = f"""
        SELECT ((ib.bin20_{prim} - 1) * $1::int) / 20 AS ix,
               ((ib.bin20_{sec}  - 1) * $1::int) / 20 AS iy,
               {expr}::float8 AS avg_ret,
               COUNT(*) AS n
        FROM daily_features df
        JOIN is_bins ib USING (ticker, trade_date)
        WHERE ib.bin20_{prim} > 0 AND ib.bin20_{sec} > 0
          AND {filt}
          AND (((ib.bin20_{prim} - 1) * $1::int) / 20,
               ((ib.bin20_{sec}  - 1) * $1::int) / 20)
              IN (SELECT * FROM unnest($2::int[], $3::int[]))
        GROUP BY ix, iy
    """
    try:
        rows = await conn.fetch(sql, n_bins, xs, ys)
    except Exception:
        return None
    per_cell, total, wsum = [], 0, 0.0
    for r in rows:
        n = int(r["n"])
        avg = float(r["avg_ret"]) if r["avg_ret"] is not None else 0.0
        per_cell.append({"ix": int(r["ix"]), "iy": int(r["iy"]),
                         "avg_ret": round(avg, 6), "n": n})
        total += n
        wsum += avg * n
    return {"agg_avg_ret": round(wsum / total, 6) if total else None,
            "agg_n": total, "per_cell_stats": per_cell}


def _pct(new: int, old: int) -> str:
    if not old:
        return "  n/a"
    return f"{100.0 * (new - old) / old:+5.1f}%"


async def run(args) -> None:
    dsn = os.getenv("OI_DATABASE_URL")
    if not dsn:
        print("ERROR: OI_DATABASE_URL not set.")
        sys.exit(1)
    conn = await asyncpg.connect(dsn=dsn)
    try:
        where, params = "", []
        if args.ids:
            where, params = "WHERE id = ANY($1)", [args.ids]
        sigs = await conn.fetch(
            f"""SELECT id, name, primary_metric, secondary_metric, outcome,
                       n_bins, cell_set, agg_avg_ret, agg_n, stats_updated_at
                FROM signals {where} ORDER BY id""", *params)
        if not sigs:
            print("No signals matched.")
            return

        mode = "DRY RUN — nothing will be written" if args.dry_run else "WRITING"
        print(f"{mode}\nsignals: {len(sigs)}\n")
        print(f"{'signal':<34} {'stored n':>9} {'fresh n':>9} {'delta':>8} "
              f"{'stored avg':>11} {'fresh avg':>11}")
        print("-" * 90)

        changed = unchanged = failed = 0
        up, down = 0, 0
        for s in sigs:
            cells_raw = s["cell_set"] or "[]"
            cells = json.loads(cells_raw) if isinstance(cells_raw, str) else cells_raw
            st = await _stats(conn, s["primary_metric"], s["secondary_metric"],
                              s["outcome"], int(s["n_bins"]), cells)
            label = f"#{s['id']} {s['name'][:28]}"
            if st is None:
                print(f"{label:<34} {'—':>9} {'FAILED — unsafe name or bad cells':>50}")
                failed += 1
                continue
            old_n = int(s["agg_n"] or 0)
            old_a = float(s["agg_avg_ret"]) if s["agg_avg_ret"] is not None else None
            new_n, new_a = st["agg_n"], st["agg_avg_ret"]
            same = (old_n == new_n
                    and ((old_a is None and new_a is None)
                         or (old_a is not None and new_a is not None
                             and abs(old_a - new_a) < 2e-6)))
            print(f"{label:<34} {old_n:>9,} {new_n:>9,} {_pct(new_n, old_n):>8} "
                  f"{(f'{old_a*100:.3f}%' if old_a is not None else '—'):>11} "
                  f"{(f'{new_a*100:.3f}%' if new_a is not None else '—'):>11}")
            if same:
                unchanged += 1
                continue
            changed += 1
            if new_n > old_n:
                up += 1
            elif new_n < old_n:
                down += 1
            if not args.dry_run:
                await conn.execute(
                    """UPDATE signals
                       SET agg_avg_ret      = $2,
                           agg_n            = $3,
                           per_cell_stats   = $4::jsonb,
                           stats_updated_at = NOW()
                       WHERE id = $1""",
                    s["id"], new_a, new_n, json.dumps(st["per_cell_stats"]))

        print("\n" + "=" * 90)
        print(f"  would change : {changed}" if args.dry_run
              else f"  refreshed    : {changed}")
        print(f"  already current: {unchanged}")
        print(f"  failed         : {failed}")
        print(f"  n went UP      : {up}      n went DOWN: {down}")
        if down:
            print("\n  A signal whose n went DOWN is worth a look. is_bins is not")
            print("  append-only -- a rebuild re-derives full-history quantile")
            print("  edges, so rows can move OUT of a saved zone as well as in.")
            print("  Run: python scripts/audit_signal_provenance.py --explain-drift")
            print("  to split the delta into new dates vs re-labelled history.")
        if args.dry_run and changed:
            print("\n  Re-run without --dry-run to write these.")
    finally:
        await conn.close()


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true",
                    help="Report the drift; write nothing")
    ap.add_argument("--ids", type=int, nargs="+",
                    help="Only these signal ids (default: all)")
    asyncio.run(run(ap.parse_args()))


if __name__ == "__main__":
    main()
