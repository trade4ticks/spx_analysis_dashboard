#!/usr/bin/env python3
"""Characterise the train_test rows that have only ONE window populated.

A corner with zero qualifying rows in a window is not automatically a bug.
With bin edges frozen at the cutoff and a test period shorter than the
train period, a thin corner can genuinely have no test observations, and
that is correct behaviour rather than something to "fix".

But three OTHER things produce the same symptom, and they are real
problems.  This script separates them:

  (a) THIN TAIL — expected.
      Small n on the populated side, spread across many metric pairs, and
      skewed toward long forward horizons (a 20-day forward return is
      undefined for the last ~20 trading days, so the test window loses
      its tail).  Nothing to do but let the check allow it.

  (b) STALE METRIC — a data problem.
      corner_scan.py's usable-bin guard drops a metric only when it has
      ZERO usable rows in a window.  A metric with, say, 5 usable train
      rows passes that guard and then produces train-thin corners against
      every partner.  Symptom: the one-window rows CONCENTRATE on one or
      two metrics instead of spreading out.

  (c) COMPUTATION ARTIFACT — a code problem.
      Substantial n on the populated side with a genuinely empty other
      side is not a thin corner; it means rows are being lost.  Symptom:
      large n on the populated side, and the independent SQL recount
      disagreeing with the stored zero.

Reports, in order:
  1. direction split (train-only vs test-only)
  2. n distribution on the populated side
  3. concentration by metric        -> discriminates (b)
  4. concentration by outcome       -> confirms (a)'s expected skew
  5. independent SQL recount of the EMPTY side on a sample -> rules out (c)

Usage (VPS, project root, venv active):
    python scripts/corner_scan_tt_onewindow.py
    python scripts/corner_scan_tt_onewindow.py --sample 40 --substantial 100

Exit status is non-zero if anything looks like (b) or (c).
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


def _outcome_expr(outcome: str) -> str:
    if outcome == "overnight_gap":
        return "(df.spot_co / NULLIF(df.spot_pc, 0) - 1.0)"
    return f'df."{outcome}"'


async def _sql_count(conn, p, s, direction, outcome, n_bins, window, cutoff):
    """Independent recount of one corner in one window, straight from SQL."""
    p_lbl, s_lbl = direction.split("-")
    p_edge = 1 if p_lbl == "low" else n_bins
    s_edge = 1 if s_lbl == "low" else n_bins
    conds = [
        f'bt."bin20_{p}" > 0',
        f'bt."bin20_{s}" > 0',
        f'((bt."bin20_{p}" - 1) * {n_bins}) / 20 + 1 = {p_edge}',
        f'((bt."bin20_{s}" - 1) * {n_bins}) / 20 + 1 = {s_edge}',
    ]
    params: list = [cutoff]
    conds.append("bt.trade_date < $1" if window == "train"
                 else "bt.trade_date >= $1")
    expr = _outcome_expr(outcome)
    sql = (
        f"SELECT COUNT({expr}) AS n FROM tt_bins bt "
        f"JOIN daily_features df "
        f"  ON bt.ticker = df.ticker AND bt.trade_date = df.trade_date "
        f"WHERE {' AND '.join(conds)}"
    )
    return int(await conn.fetchval(sql, *params) or 0)


async def run(args) -> None:
    dsn = os.getenv("OI_DATABASE_URL")
    if not dsn:
        print("ERROR: OI_DATABASE_URL not set.")
        sys.exit(1)

    conn = await asyncpg.connect(dsn=dsn)
    try:
        cutoff = await conn.fetchval("SELECT MAX(cutoff_date) FROM tt_bins")
        print(f"Frozen TT cutoff: {cutoff}")

        total = await conn.fetchval(
            "SELECT COUNT(*) FROM corner_scan_2f WHERE mode = 'train_test'")
        one_win = await conn.fetchval(
            """SELECT COUNT(*) FROM corner_scan_2f
               WHERE mode = 'train_test'
                 AND (d_train_n IS NULL) <> (d_test_n IS NULL)""")
        print(f"TT rows total: {total:,}")
        print(f"One-window rows: {one_win:,} "
              f"({100.0 * one_win / total:.3f}% of the table)\n")
        if one_win == 0:
            print("Nothing to characterise.")
            return

        # ── 1. Direction split ───────────────────────────────────────────────
        print("═" * 74)
        print("1. DIRECTION SPLIT")
        print("═" * 74)
        row = await conn.fetchrow(
            """SELECT
                 COUNT(*) FILTER (WHERE d_train_n IS NOT NULL
                                    AND d_test_n IS NULL) AS train_only,
                 COUNT(*) FILTER (WHERE d_test_n IS NOT NULL
                                    AND d_train_n IS NULL) AS test_only
               FROM corner_scan_2f
               WHERE mode = 'train_test'
                 AND (d_train_n IS NULL) <> (d_test_n IS NULL)""")
        tr_only, te_only = int(row["train_only"]), int(row["test_only"])
        print(f"  train populated, test EMPTY : {tr_only:>7,}"
              f"   ({100.0 * tr_only / one_win:5.1f}%)")
        print(f"  test populated, train EMPTY : {te_only:>7,}"
              f"   ({100.0 * te_only / one_win:5.1f}%)")
        print("\n  Expected: overwhelmingly train-populated/test-empty. The")
        print("  test window is shorter, so it is the side that runs out of")
        print("  observations first. A large test-only share would instead")
        print("  suggest metrics whose bins only exist late in the sample.")

        # ── 2. n distribution on the populated side ──────────────────────────
        print("\n" + "═" * 74)
        print("2. n DISTRIBUTION ON THE POPULATED SIDE")
        print("═" * 74)
        dist = await conn.fetchrow(
            f"""SELECT COUNT(*) AS n_rows,
                       MIN(pop_n) AS min_n,
                       PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY pop_n) AS p25,
                       PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY pop_n) AS p50,
                       PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY pop_n) AS p75,
                       PERCENTILE_CONT(0.90) WITHIN GROUP (ORDER BY pop_n) AS p90,
                       PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY pop_n) AS p99,
                       MAX(pop_n) AS max_n,
                       COUNT(*) FILTER (WHERE pop_n >= 30)  AS ge_30,
                       COUNT(*) FILTER (WHERE pop_n >= 100) AS ge_100,
                       COUNT(*) FILTER (WHERE pop_n >= $1)  AS ge_sub
                FROM (
                    SELECT COALESCE(d_train_n, d_test_n) AS pop_n
                    FROM corner_scan_2f
                    WHERE mode = 'train_test'
                      AND (d_train_n IS NULL) <> (d_test_n IS NULL)
                ) t""",
            args.substantial)
        for label, key in (("min", "min_n"), ("p25", "p25"), ("median", "p50"),
                           ("p75", "p75"), ("p90", "p90"), ("p99", "p99"),
                           ("max", "max_n")):
            v = dist[key]
            print(f"  {label:>6}: {float(v):>10,.0f}" if v is not None
                  else f"  {label:>6}:          —")
        print(f"\n  n >= 30           : {int(dist['ge_30']):>7,}")
        print(f"  n >= 100          : {int(dist['ge_100']):>7,}")
        print(f"  n >= {args.substantial:<13}: {int(dist['ge_sub']):>7,}"
              f"   <- 'substantial' threshold")
        substantial = int(dist["ge_sub"])

        # ── 3. Concentration by metric ───────────────────────────────────────
        print("\n" + "═" * 74)
        print("3. CONCENTRATION BY METRIC  (stale-metric detector)")
        print("═" * 74)
        conc = await conn.fetch(
            """SELECT m, COUNT(*) AS n FROM (
                   SELECT primary_metric AS m FROM corner_scan_2f
                   WHERE mode='train_test'
                     AND (d_train_n IS NULL) <> (d_test_n IS NULL)
                   UNION ALL
                   SELECT secondary_metric AS m FROM corner_scan_2f
                   WHERE mode='train_test'
                     AND (d_train_n IS NULL) <> (d_test_n IS NULL)
               ) t GROUP BY m ORDER BY n DESC LIMIT 12""")
        n_metrics_involved = await conn.fetchval(
            """SELECT COUNT(DISTINCT m) FROM (
                   SELECT primary_metric AS m FROM corner_scan_2f
                   WHERE mode='train_test'
                     AND (d_train_n IS NULL) <> (d_test_n IS NULL)
                   UNION ALL
                   SELECT secondary_metric AS m FROM corner_scan_2f
                   WHERE mode='train_test'
                     AND (d_train_n IS NULL) <> (d_test_n IS NULL)
               ) t""")
        top_share = (100.0 * int(conc[0]["n"]) / (2 * one_win)) if conc else 0.0
        print(f"  distinct metrics involved: {n_metrics_involved}")
        print(f"  {'metric':<46} {'appearances':>12}")
        print(f"  {'-'*46} {'-'*12}")
        for r in conc:
            print(f"  {r['m']:<46} {int(r['n']):>12,}")
        print(f"\n  top metric accounts for {top_share:.1f}% of all slots.")
        print("  Spread across many metrics -> thin tail (expected).")
        print("  Concentrated on one or two -> that metric's bins are stale")
        print("  in one window; rebuild tt_bins upstream.")

        # ── 4. Concentration by outcome ──────────────────────────────────────
        print("\n" + "═" * 74)
        print("4. CONCENTRATION BY OUTCOME  (expected-skew confirmation)")
        print("═" * 74)
        by_out = await conn.fetch(
            """SELECT outcome, COUNT(*) AS n FROM corner_scan_2f
               WHERE mode='train_test'
                 AND (d_train_n IS NULL) <> (d_test_n IS NULL)
               GROUP BY outcome ORDER BY n DESC""")
        for r in by_out:
            print(f"  {r['outcome']:<20} {int(r['n']):>8,}")
        print("\n  Expected skew: LONG horizons (10d, 20d) dominate. A forward")
        print("  return is undefined for the last N trading days, so the test")
        print("  window loses its tail and long-horizon corners empty out")
        print("  first. Flat across horizons would be less consistent with a")
        print("  pure thin-tail explanation.")

        # ── 5. Independent SQL recount of the EMPTY side ─────────────────────
        print("\n" + "═" * 74)
        print(f"5. SQL RECOUNT OF THE EMPTY SIDE  (sample of {args.sample},")
        print("   largest populated-n first — the most suspicious ones)")
        print("═" * 74)
        sample = await conn.fetch(
            """SELECT primary_metric, secondary_metric, corner_direction,
                      outcome, d_train_n, d_test_n
               FROM corner_scan_2f
               WHERE mode='train_test'
                 AND (d_train_n IS NULL) <> (d_test_n IS NULL)
               ORDER BY COALESCE(d_train_n, d_test_n) DESC
               LIMIT $1""",
            args.sample)
        mismatches = 0
        print(f"  {'corner':<58} {'pop n':>8} {'empty side (SQL)':>17}")
        print(f"  {'-'*58} {'-'*8} {'-'*17}")
        for r in sample:
            empty_win = "test" if r["d_test_n"] is None else "train"
            pop_n = r["d_train_n"] if r["d_test_n"] is None else r["d_test_n"]
            sql_n = await _sql_count(
                conn, r["primary_metric"], r["secondary_metric"],
                r["corner_direction"], r["outcome"], 10, empty_win, cutoff)
            flag = "" if sql_n == 0 else f"  <- MISMATCH (stored 0)"
            if sql_n != 0:
                mismatches += 1
            name = (f"{r['primary_metric']}x{r['secondary_metric']} "
                    f"[{r['corner_direction']}] {r['outcome']}")
            print(f"  {name[:58]:<58} {int(pop_n):>8,} "
                  f"{empty_win}={sql_n:<12}{flag}")

        # ── Verdict ──────────────────────────────────────────────────────────
        print("\n" + "═" * 74)
        print("VERDICT")
        print("═" * 74)
        problems = []
        if mismatches:
            problems.append(
                f"(c) {mismatches} sampled row(s) have a NON-ZERO SQL count on "
                f"the side stored as empty — rows are being lost.")
        if substantial:
            problems.append(
                f"(c?) {substantial} row(s) have n >= {args.substantial} on the "
                f"populated side with the other side empty — verify those.")
        if top_share > 25.0:
            problems.append(
                f"(b) one metric accounts for {top_share:.1f}% of slots — "
                f"looks like a stale metric, not a thin tail.")
        if problems:
            for p in problems:
                print("  ✗ " + p)
            print("\n  Do NOT relax the batch self-check yet.")
            sys.exit(1)
        print("  ✓ Spread across metrics, small n on the populated side, and")
        print("    every sampled empty side independently confirms as zero.")
        print("    Consistent with a genuine thin tail under frozen edges.")
        print("    Safe to let the self-check allow one-window rows.")
    finally:
        await conn.close()


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sample", type=int, default=25,
                    help="How many corners to SQL-recount (default 25)")
    ap.add_argument("--substantial", type=int, default=300,
                    help="n on the populated side that counts as "
                         "'substantial' and warrants suspicion (default 300)")
    asyncio.run(run(ap.parse_args()))


if __name__ == "__main__":
    main()
