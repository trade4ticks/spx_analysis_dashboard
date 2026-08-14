#!/usr/bin/env python3
"""Test-window bin20 occupancy per metric — a level-drift detector for tt_bins.

tt_bins edges are TRAIN quantiles, frozen at the cutoff and applied
unchanged to test rows.  That only sorts test data correctly if the metric
is stationary in LEVEL.  If a metric drifts up, its test rows pile into the
high bins and vacate the low ones; the "top decile" stops meaning what it
meant in train, and any corner built on that extreme is comparing two
different populations across the split.

WHY THE NAIVE 5% BASELINE IS NOT ENOUGH
---------------------------------------
"Each bin should hold ~5% of test rows" is true only for a CONTINUOUS
metric.  Bin edges are quantiles, so a metric with heavy ties — integer
counts, a pile of exact zeros, a floor or cap — cannot be split into 20
equal groups even in train.  Its TRAIN occupancy is already lumpy, and
measuring test against a flat 5% would flag that discreteness as drift.

So this reports both, and sorts on the one that isolates drift:

    test vs 5%     — what a flat expectation gives you (asked for; shown)
    test vs TRAIN  — the drift measure, since train occupancy is the
                     realised baseline the edges actually produced

Both as total variation distance, 0.5 * sum |p_test(b) - p_ref(b)| over the
20 bins: 0.00 = identical, 1.00 = disjoint.  Reading TVD_vs_train: ~0.05 is
noise, 0.15 is a visible shift, 0.30+ means the extremes have substantially
changed membership.

Also reported per metric:
    b1 / b20 occupancy, train vs test, and the test/train ratio for each —
    the extremes are what the corner scan actually uses
    drift  — signed: (test high5 - test low5) - (train high5 - train low5),
             where high5 = bins 16..20 and low5 = bins 1..5.
             positive = level drifted UP, negative = DOWN

RAW vs SELF-NORMALISING PAIRS
-----------------------------
Section 3 pairs metrics whose names are related by containment (e.g. `x`
and `x_zscore`) and shows their drift side by side.  A z-scored metric
re-centres itself, so it stays near 5% per bin while its raw counterpart
migrates.  That divergence is exactly what produces "raw extreme one way,
z-score extreme the other way" corners with an empty train joint.

Usage (VPS, project root, venv active):
    python scripts/tt_bin_occupancy.py
    python scripts/tt_bin_occupancy.py --top 25 --histograms 10
    python scripts/tt_bin_occupancy.py --eligible-only
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(_ROOT / ".env")

import asyncpg  # noqa: E402

# ASCII on purpose: block-drawing glyphs raise UnicodeEncodeError on a
# cp1252 console, and a diagnostic must not die while printing its result.
_BARS = " .:-=+*#%@"


def _tvd(p: list[float], q: list[float]) -> float:
    return 0.5 * sum(abs(a - b) for a, b in zip(p, q))


def _spark(shares: list[float], ceiling: float = 0.15) -> str:
    """20-char sparkline; 5% renders mid-scale so drift reads at a glance."""
    out = []
    for v in shares:
        idx = min(len(_BARS) - 1, int(round((v / ceiling) * (len(_BARS) - 1))))
        out.append(_BARS[idx])
    return "".join(out)


async def run(args) -> None:
    dsn = os.getenv("OI_DATABASE_URL")
    if not dsn:
        print("ERROR: OI_DATABASE_URL not set.")
        sys.exit(1)

    conn = await asyncpg.connect(dsn=dsn)
    try:
        cutoff = await conn.fetchval("SELECT MAX(cutoff_date) FROM tt_bins")
        if cutoff is None:
            print("ERROR: tt_bins has no cutoff_date.")
            sys.exit(1)

        cols = await conn.fetch(
            """SELECT column_name FROM information_schema.columns
               WHERE table_name = 'tt_bins' AND table_schema = 'public'
                 AND column_name LIKE 'bin20_%'
               ORDER BY column_name""")
        metrics = [r["column_name"][6:] for r in cols]

        if args.eligible_only:
            elig = {r["metric"] for r in await conn.fetch(
                "SELECT metric FROM metric_classification "
                "WHERE eligible_as_metric = true")}
            metrics = [m for m in metrics if m in elig]

        span = await conn.fetchrow(
            """SELECT COUNT(*) FILTER (WHERE trade_date <  $1) AS n_train,
                      COUNT(*) FILTER (WHERE trade_date >= $1) AS n_test,
                      MIN(trade_date) AS d0, MAX(trade_date) AS d1
               FROM tt_bins""", cutoff)

        print(f"Frozen TT cutoff : {cutoff}")
        print(f"tt_bins span     : {span['d0']} .. {span['d1']}")
        print(f"rows             : train {int(span['n_train']):,}"
              f"   test {int(span['n_test']):,}")
        print(f"metrics          : {len(metrics)}\n")

        # ── Per-metric occupancy ─────────────────────────────────────────────
        results = []
        for i, m in enumerate(metrics, 1):
            rows = await conn.fetch(
                f'''SELECT bt."bin20_{m}" AS b,
                           COUNT(*) FILTER (WHERE bt.trade_date <  $1) AS tr,
                           COUNT(*) FILTER (WHERE bt.trade_date >= $1) AS te
                    FROM tt_bins bt
                    WHERE bt."bin20_{m}" > 0
                    GROUP BY 1''', cutoff)
            tr_c = [0] * 20
            te_c = [0] * 20
            for r in rows:
                b = int(r["b"])
                if 1 <= b <= 20:
                    tr_c[b - 1] = int(r["tr"])
                    te_c[b - 1] = int(r["te"])
            n_tr, n_te = sum(tr_c), sum(te_c)
            if n_tr == 0 or n_te == 0:
                results.append(dict(metric=m, n_train=n_tr, n_test=n_te,
                                    skip=True))
                continue
            tr_p = [c / n_tr for c in tr_c]
            te_p = [c / n_te for c in te_c]
            unif = [0.05] * 20
            drift = ((sum(te_p[15:]) - sum(te_p[:5]))
                     - (sum(tr_p[15:]) - sum(tr_p[:5])))
            results.append(dict(
                metric=m, n_train=n_tr, n_test=n_te, skip=False,
                tr_p=tr_p, te_p=te_p,
                tvd_train=_tvd(te_p, tr_p),
                tvd_unif=_tvd(te_p, unif),
                b1_tr=tr_p[0], b1_te=te_p[0],
                b20_tr=tr_p[19], b20_te=te_p[19],
                drift=drift))
            if args.progress and i % 20 == 0:
                print(f"  … {i}/{len(metrics)} metrics scanned",
                      file=sys.stderr)

        usable = [r for r in results if not r["skip"]]
        skipped = [r for r in results if r["skip"]]
        usable.sort(key=lambda r: r["tvd_train"], reverse=True)

        # ── 1. Summary table ─────────────────────────────────────────────────
        print("═" * 118)
        print("1. TEST-WINDOW OCCUPANCY SKEW — worst first (sorted by "
              "TVD vs train)")
        print("═" * 118)
        print(f"{'metric':<40} {'test n':>8} "
              f"{'b1 tr%':>7} {'b1 te%':>7} {'x':>5} "
              f"{'b20 tr%':>8} {'b20 te%':>8} {'x':>5} "
              f"{'TVDtr':>6} {'TVD5%':>6} {'drift':>7}")
        print("-" * 118)
        for r in usable[:args.top]:
            b1_ratio = (r["b1_te"] / r["b1_tr"]) if r["b1_tr"] else float("inf")
            b20_ratio = (r["b20_te"] / r["b20_tr"]) if r["b20_tr"] else float("inf")
            print(f"{r['metric']:<40} {r['n_test']:>8,} "
                  f"{100*r['b1_tr']:>7.2f} {100*r['b1_te']:>7.2f} "
                  f"{b1_ratio:>5.1f} "
                  f"{100*r['b20_tr']:>8.2f} {100*r['b20_te']:>8.2f} "
                  f"{b20_ratio:>5.1f} "
                  f"{r['tvd_train']:>6.3f} {r['tvd_unif']:>6.3f} "
                  f"{100*r['drift']:>+6.1f}%")
        if len(usable) > args.top:
            print(f"... {len(usable) - args.top} more metrics "
                  f"(use --top {len(usable)} for all)")

        # ── 2. Full 20-bin histograms for the worst offenders ────────────────
        print("\n" + "═" * 118)
        print(f"2. FULL 20-BIN TEST OCCUPANCY — worst {args.histograms}")
        print("═" * 118)
        print("   bar scale: each glyph is one bin20, full height = 15% of "
              "rows, flat 5% ≈ mid-height\n")
        for r in usable[:args.histograms]:
            print(f"  {r['metric']}   "
                  f"TVD vs train {r['tvd_train']:.3f}   "
                  f"drift {100*r['drift']:+.1f}%")
            print(f"    train |{_spark(r['tr_p'])}|")
            print(f"    test  |{_spark(r['te_p'])}|")
            print("    test %: " + " ".join(
                f"{100*v:.1f}" for v in r["te_p"]))
            print()

        # ── 3. Raw vs self-normalising pairs ─────────────────────────────────
        print("═" * 118)
        print("3. RELATED-NAME PAIRS  (raw vs self-normalising)")
        print("═" * 118)
        by_name = {r["metric"]: r for r in usable}
        pairs = []
        for a in by_name:
            for b in by_name:
                if a != b and a in b:          # e.g. 'x' inside 'x_zscore'
                    pairs.append((a, b))
        if not pairs:
            print("  no name-containment pairs found — if your raw/z-score")
            print("  naming does not nest, read section 1 directly.")
        else:
            pairs.sort(key=lambda t: abs(by_name[t[0]]["drift"]), reverse=True)
            print(f"{'shorter (raw?)':<38} {'drift':>7} {'TVDtr':>6}   "
                  f"{'longer (derived?)':<38} {'drift':>7} {'TVDtr':>6}")
            print("-" * 118)
            for a, b in pairs[:args.top]:
                ra, rb = by_name[a], by_name[b]
                print(f"{a:<38} {100*ra['drift']:>+6.1f}% {ra['tvd_train']:>6.3f}   "
                      f"{b:<38} {100*rb['drift']:>+6.1f}% {rb['tvd_train']:>6.3f}")
            print("\n  A raw metric with large |drift| beside a derived one")
            print("  near zero is the signature that produces 'raw extreme")
            print("  one way, normalised extreme the other' corners with an")
            print("  empty train joint.")

        # ── 4. Headline counts ───────────────────────────────────────────────
        print("\n" + "═" * 118)
        print("4. SUMMARY")
        print("═" * 118)
        for thr in (0.10, 0.20, 0.30):
            n = sum(1 for r in usable if r["tvd_train"] >= thr)
            print(f"  metrics with TVD vs train >= {thr:.2f} : {n:>4} "
                  f"of {len(usable)}")
        vacated = sum(1 for r in usable
                      if r["b1_tr"] and r["b1_te"] / r["b1_tr"] < 0.5)
        crowded = sum(1 for r in usable
                      if r["b20_tr"] and r["b20_te"] / r["b20_tr"] > 2.0)
        print(f"  metrics whose bin 1 lost  >50% of its share : {vacated:>4}")
        print(f"  metrics whose bin 20 gained >2x its share   : {crowded:>4}")
        up = sum(1 for r in usable if r["drift"] > 0.05)
        dn = sum(1 for r in usable if r["drift"] < -0.05)
        print(f"  metrics drifted UP   (>+5pp) : {up:>4}")
        print(f"  metrics drifted DOWN (<-5pp) : {dn:>4}")
        if skipped:
            print(f"\n  {len(skipped)} metric(s) had no usable rows in one "
                  f"window and were skipped:")
            for r in skipped[:10]:
                print(f"    {r['metric']:<44} train={r['n_train']:,} "
                      f"test={r['n_test']:,}")
        print("\n  Frozen edges only sort test data correctly for a")
        print("  stationary metric. A high-TVD metric's D1/D10 in test is not")
        print("  the same population as in train, so its corners compare")
        print("  different things across the split — the test number is still")
        print("  a real out-of-sample result, but it is not measuring the")
        print("  same conditioning event.")
    finally:
        await conn.close()


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--top", type=int, default=30,
                    help="Rows in the summary table (default 30)")
    ap.add_argument("--histograms", type=int, default=8,
                    help="Metrics to show full 20-bin detail for (default 8)")
    ap.add_argument("--eligible-only", action="store_true",
                    help="Restrict to metric_classification.eligible_as_metric")
    ap.add_argument("--progress", action="store_true",
                    help="Print scan progress to stderr")
    asyncio.run(run(ap.parse_args()))


if __name__ == "__main__":
    main()
