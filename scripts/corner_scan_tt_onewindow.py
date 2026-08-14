#!/usr/bin/env python3
"""Characterise the train_test corners that have only ONE window populated.

THE UNIT IS THE CORNER, NOT THE ROW.  Every corner (P, S, direction) emits
13 rows, one per outcome, and corner membership depends on BINS, not on the
outcome — so when a corner is empty in a window it is empty for all 13
outcomes at once.  Counting rows therefore multiplies every finding by ~13
and makes any per-outcome breakdown mechanically flat.  Everything below
counts distinct corners.

Two corner identities are reported:
  (P, S, direction)   — as stored; each logical corner appears twice
  unordered           — (A, B, dir) and (B, A, reversed dir) collapsed,
                        since they name the same set of trade-dates

A corner with zero qualifying rows in a window is not automatically a bug.
With edges frozen at the cutoff and a shorter test period, a thin corner
can genuinely have no test observations.  Three things share the symptom:

  (a) THIN TAIL — expected.  Small n, spread across many pairs, and
      TRAIN-populated / TEST-empty: the test window is shorter, so it is
      the side that runs out.

  (b) THIN TRAIN COVERAGE — a data problem.  corner_scan.py's usable-bin
      guard drops a metric only at ZERO usable rows in a window; a metric
      with a handful of usable train rows clears it and then produces
      train-empty corners against every partner.

  (c) ROWS BEING LOST — a code problem.  Substantial n on the populated
      side with a genuinely empty other side is not a thin corner.

TEST-ONLY corners are the direction that needs explaining.  Bin edges are
quantiles OF THE TRAIN WINDOW, so by construction each bin20 holds ~5% of
each ticker's train rows and every extreme has a healthy train marginal.
For the JOINT to be empty in train while populated in test, one of these
must hold — section 6 measures which:

    * genuine anti-correlation: both train marginals healthy, joint 0,
      and the pair only co-occurs at the extremes after some regime shift
    * thin train coverage:      a train marginal is tiny (case b)
    * late-universe tickers:    the test rows come from tickers that have
      no train history at all, so they could not contribute a train row

Usage (VPS, project root, venv active):
    python scripts/corner_scan_tt_onewindow.py
    python scripts/corner_scan_tt_onewindow.py --sample 15 --substantial 300

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


# One-window rows, tagged with both corner identities.  Reused by every
# section below so the corner definition can never drift between them.
_OW_CTE = """
WITH ow AS (
    SELECT primary_metric, secondary_metric, corner_direction, outcome,
           d_train_n, d_test_n,
           COALESCE(d_train_n, d_test_n) AS pop_n,
           CASE WHEN d_test_n IS NULL THEN 'train_only'
                ELSE 'test_only' END     AS side,
           CASE WHEN primary_metric <= secondary_metric
                THEN primary_metric ELSE secondary_metric END AS m1,
           CASE WHEN primary_metric <= secondary_metric
                THEN secondary_metric ELSE primary_metric END AS m2,
           CASE WHEN primary_metric <= secondary_metric
                THEN corner_direction
                ELSE CASE corner_direction
                       WHEN 'low-high' THEN 'high-low'
                       WHEN 'high-low' THEN 'low-high'
                       ELSE corner_direction END END          AS udir
    FROM corner_scan_2f
    WHERE mode = 'train_test'
      AND (d_train_n IS NULL) <> (d_test_n IS NULL)
)
"""


def _outcome_expr(outcome: str) -> str:
    if outcome == "overnight_gap":
        return "(df.spot_co / NULLIF(df.spot_pc, 0) - 1.0)"
    return f'df."{outcome}"'


def _edge_vals(direction: str, n_bins: int) -> tuple[int, int]:
    p_lbl, s_lbl = direction.split("-")
    return (1 if p_lbl == "low" else n_bins,
            1 if s_lbl == "low" else n_bins)


async def _joint_count(conn, p, s, direction, outcome, n_bins, window, cutoff):
    """Independent SQL recount of one corner in one window."""
    p_edge, s_edge = _edge_vals(direction, n_bins)
    conds = [
        f'bt."bin20_{p}" > 0',
        f'bt."bin20_{s}" > 0',
        f'((bt."bin20_{p}" - 1) * {n_bins}) / 20 + 1 = {p_edge}',
        f'((bt."bin20_{s}" - 1) * {n_bins}) / 20 + 1 = {s_edge}',
        "bt.trade_date < $1" if window == "train" else "bt.trade_date >= $1",
    ]
    expr = _outcome_expr(outcome)
    return int(await conn.fetchval(
        f"SELECT COUNT({expr}) AS n FROM tt_bins bt "
        f"JOIN daily_features df "
        f"  ON bt.ticker = df.ticker AND bt.trade_date = df.trade_date "
        f"WHERE {' AND '.join(conds)}",
        cutoff) or 0)


async def _anatomy(conn, p, s, direction, cutoff):
    """Marginals + ticker coverage for a corner, to explain a test-only case."""
    p_edge, s_edge = _edge_vals(direction, 10)

    def bin_expr(m, edge):
        return (f'(bt."bin20_{m}" > 0 AND '
                f'((bt."bin20_{m}" - 1) * 10) / 20 + 1 = {edge})')

    row = await conn.fetchrow(
        f"""SELECT
              COUNT(*) FILTER (WHERE bt.trade_date <  $1
                                 AND bt."bin20_{p}" > 0) AS p_usable_train,
              COUNT(*) FILTER (WHERE bt.trade_date <  $1
                                 AND bt."bin20_{s}" > 0) AS s_usable_train,
              COUNT(*) FILTER (WHERE bt.trade_date <  $1
                                 AND {bin_expr(p, p_edge)}) AS p_edge_train,
              COUNT(*) FILTER (WHERE bt.trade_date <  $1
                                 AND {bin_expr(s, s_edge)}) AS s_edge_train,
              COUNT(*) FILTER (WHERE bt.trade_date <  $1
                                 AND {bin_expr(p, p_edge)}
                                 AND {bin_expr(s, s_edge)}) AS joint_train,
              COUNT(*) FILTER (WHERE bt.trade_date >= $1
                                 AND {bin_expr(p, p_edge)}
                                 AND {bin_expr(s, s_edge)}) AS joint_test
            FROM tt_bins bt""",
        cutoff)

    # Which tickers supply the test-side rows, and do they have ANY train
    # history?  A ticker added to the universe after the cutoff cannot
    # contribute a train row no matter how common the corner is.
    tk = await conn.fetchrow(
        f"""WITH test_tk AS (
                SELECT DISTINCT bt.ticker FROM tt_bins bt
                WHERE bt.trade_date >= $1
                  AND {bin_expr(p, p_edge)} AND {bin_expr(s, s_edge)}
            ),
            train_tk AS (
                SELECT DISTINCT ticker FROM tt_bins WHERE trade_date < $1
            )
            SELECT (SELECT COUNT(*) FROM test_tk) AS n_test_tk,
                   (SELECT COUNT(*) FROM test_tk
                      WHERE ticker NOT IN (SELECT ticker FROM train_tk))
                       AS n_late_tk""",
        cutoff)
    return dict(row), dict(tk)


async def run(args) -> None:
    dsn = os.getenv("OI_DATABASE_URL")
    if not dsn:
        print("ERROR: OI_DATABASE_URL not set.")
        sys.exit(1)

    conn = await asyncpg.connect(dsn=dsn)
    try:
        cutoff = await conn.fetchval("SELECT MAX(cutoff_date) FROM tt_bins")
        print(f"Frozen TT cutoff: {cutoff}")

        totals = await conn.fetchrow(f"""
            {_OW_CTE}
            SELECT (SELECT COUNT(*) FROM corner_scan_2f
                     WHERE mode='train_test')                  AS all_rows,
                   (SELECT COUNT(DISTINCT (primary_metric, secondary_metric,
                                           corner_direction))
                      FROM corner_scan_2f WHERE mode='train_test')
                                                               AS all_corners,
                   (SELECT COUNT(*) FROM ow)                   AS ow_rows,
                   (SELECT COUNT(DISTINCT (primary_metric, secondary_metric,
                                           corner_direction)) FROM ow)
                                                               AS ow_corners,
                   (SELECT COUNT(DISTINCT (m1, m2, udir)) FROM ow)
                                                               AS ow_unordered
        """)
        ow_corners = int(totals["ow_corners"])
        print(f"TT rows total          : {int(totals['all_rows']):>9,}")
        print(f"TT corners total       : {int(totals['all_corners']):>9,}")
        print(f"one-window rows        : {int(totals['ow_rows']):>9,}")
        print(f"one-window CORNERS     : {ow_corners:>9,}"
              f"   ({100.0 * ow_corners / max(int(totals['all_corners']), 1):.3f}%"
              f" of all corners)")
        print(f"  unordered (dedup'd)  : {int(totals['ow_unordered']):>9,}"
              f"   <- logical corners; each appears in both orientations")
        if ow_corners == 0:
            print("\nNothing to characterise.")
            return

        # ── 1. Direction split, BY CORNER ────────────────────────────────────
        print("\n" + "═" * 74)
        print("1. DIRECTION SPLIT  (distinct corners)")
        print("═" * 74)
        d = await conn.fetchrow(f"""
            {_OW_CTE}
            SELECT COUNT(DISTINCT (primary_metric, secondary_metric,
                                   corner_direction))
                     FILTER (WHERE side='train_only') AS train_only,
                   COUNT(DISTINCT (primary_metric, secondary_metric,
                                   corner_direction))
                     FILTER (WHERE side='test_only')  AS test_only,
                   COUNT(DISTINCT (m1, m2, udir))
                     FILTER (WHERE side='train_only') AS u_train_only,
                   COUNT(DISTINCT (m1, m2, udir))
                     FILTER (WHERE side='test_only')  AS u_test_only
            FROM ow
        """)
        tr_o, te_o = int(d["train_only"]), int(d["test_only"])
        tot = max(tr_o + te_o, 1)
        print(f"  train populated, test EMPTY : {tr_o:>6,} corners"
              f"  ({100.0 * tr_o / tot:5.1f}%)"
              f"   unordered {int(d['u_train_only']):>5,}")
        print(f"  test populated, train EMPTY : {te_o:>6,} corners"
              f"  ({100.0 * te_o / tot:5.1f}%)"
              f"   unordered {int(d['u_test_only']):>5,}")
        print("\n  train-only is the EXPECTED direction: the test window is")
        print("  shorter, so it runs out of observations first.")
        print("  test-only is the ODD direction. Bin edges are quantiles of")
        print("  the TRAIN window, so every extreme has a healthy train")
        print("  marginal by construction — an empty train JOINT needs an")
        print("  explanation. Section 6 measures which one applies.")

        # ── 2. n distribution, BY CORNER ─────────────────────────────────────
        print("\n" + "═" * 74)
        print("2. n DISTRIBUTION ON THE POPULATED SIDE  (per corner)")
        print("═" * 74)
        print("  A corner's n is MAX(n) across its 13 outcomes — the largest")
        print("  outcome coverage, i.e. closest to raw bin membership.\n")
        dist = await conn.fetchrow(f"""
            {_OW_CTE},
            per_corner AS (
                SELECT primary_metric, secondary_metric, corner_direction,
                       MAX(pop_n) AS corner_n
                FROM ow
                GROUP BY primary_metric, secondary_metric, corner_direction
            )
            SELECT COUNT(*) AS n_corners,
                   MIN(corner_n) AS min_n,
                   PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY corner_n) AS p25,
                   PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY corner_n) AS p50,
                   PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY corner_n) AS p75,
                   PERCENTILE_CONT(0.90) WITHIN GROUP (ORDER BY corner_n) AS p90,
                   MAX(corner_n) AS max_n,
                   COUNT(*) FILTER (WHERE corner_n >= 30)  AS ge_30,
                   COUNT(*) FILTER (WHERE corner_n >= 100) AS ge_100,
                   COUNT(*) FILTER (WHERE corner_n >= $1)  AS ge_sub
            FROM per_corner
        """, args.substantial)
        for label, key in (("min", "min_n"), ("p25", "p25"), ("median", "p50"),
                           ("p75", "p75"), ("p90", "p90"), ("max", "max_n")):
            v = dist[key]
            print(f"  {label:>6}: {float(v):>9,.0f}" if v is not None
                  else f"  {label:>6}:         —")
        substantial = int(dist["ge_sub"])
        print(f"\n  corners with n >= 30        : {int(dist['ge_30']):>5,}")
        print(f"  corners with n >= 100       : {int(dist['ge_100']):>5,}")
        print(f"  corners with n >= {args.substantial:<9}: {substantial:>5,}"
              f"   <- 'substantial'")

        # ── 3. Metric concentration, BY CORNER ───────────────────────────────
        print("\n" + "═" * 74)
        print("3. METRIC CONCENTRATION  (distinct corners per metric)")
        print("═" * 74)
        conc = await conn.fetch(f"""
            {_OW_CTE},
            corners AS (
                SELECT DISTINCT primary_metric, secondary_metric,
                       corner_direction, side
                FROM ow
            ),
            slots AS (
                SELECT primary_metric AS m, side FROM corners
                UNION ALL
                SELECT secondary_metric AS m, side FROM corners
            )
            SELECT m,
                   COUNT(*) AS n,
                   COUNT(*) FILTER (WHERE side='test_only') AS n_test_only
            FROM slots GROUP BY m ORDER BY n DESC LIMIT 12
        """)
        n_metrics = await conn.fetchval(f"""
            {_OW_CTE},
            corners AS (SELECT DISTINCT primary_metric, secondary_metric,
                               corner_direction FROM ow)
            SELECT COUNT(DISTINCT m) FROM (
                SELECT primary_metric AS m FROM corners
                UNION ALL SELECT secondary_metric AS m FROM corners) t
        """)
        total_slots = 2 * ow_corners
        top_share = (100.0 * int(conc[0]["n"]) / total_slots) if conc else 0.0
        print(f"  distinct metrics involved: {int(n_metrics)}")
        print(f"  {'metric':<44} {'corners':>8} {'test-only':>10}")
        print(f"  {'-'*44} {'-'*8} {'-'*10}")
        for r in conc:
            print(f"  {r['m']:<44} {int(r['n']):>8,} {int(r['n_test_only']):>10,}")
        print(f"\n  top metric holds {top_share:.1f}% of corner slots "
              f"({total_slots:,} total).")
        print("  Spread out -> thin tail. Concentrated -> that metric's bins")
        print("  are thin in one window; check it upstream.")

        # ── 4. Corners per outcome (interpretable form) ──────────────────────
        print("\n" + "═" * 74)
        print("4. CORNERS PER OUTCOME")
        print("═" * 74)
        print("  Corner membership is bin-driven, so a corner empty in a")
        print("  window is empty for ALL its outcomes. Counts here should be")
        print("  near-identical across outcomes; the informative number is")
        print("  the split below, not the per-outcome list.\n")
        full_split = await conn.fetchrow(f"""
            {_OW_CTE},
            per_corner AS (
                SELECT primary_metric, secondary_metric, corner_direction,
                       COUNT(*) AS n_outcomes
                FROM ow
                GROUP BY primary_metric, secondary_metric, corner_direction
            )
            SELECT COUNT(*) FILTER (WHERE n_outcomes = 13) AS all13,
                   COUNT(*) FILTER (WHERE n_outcomes <  13) AS partial,
                   MIN(n_outcomes) AS min_o
            FROM per_corner
        """)
        print(f"  one-window for ALL 13 outcomes : "
              f"{int(full_split['all13']):>5,}"
              f"   <- pure bin emptiness (expected)")
        print(f"  one-window for SOME outcomes   : "
              f"{int(full_split['partial']):>5,}"
              f"   <- outcome-specific; see note")
        print(f"  fewest outcomes on any corner  : {int(full_split['min_o']):>5,}")
        print("\n  A PARTIAL corner is not necessarily wrong: CC outcomes are")
        print("  excluded when a MORNING metric is on either axis, and a long")
        print("  forward return is undefined for the last N trading days, so")
        print("  a corner whose few test rows sit at the very end can empty")
        print("  out for 20d while surviving for 1d.")

        # ── 5. SQL recount of the empty side (per corner, deduped) ───────────
        print("\n" + "═" * 74)
        print(f"5. SQL RECOUNT OF THE EMPTY SIDE  ({args.sample} corners,")
        print("   largest populated n first)")
        print("═" * 74)
        sample = await conn.fetch(f"""
            {_OW_CTE},
            per_corner AS (
                -- side is paired with top_outcome via the same ordering, so
                -- the recount always asks about the window that is actually
                -- empty FOR THAT OUTCOME. A corner can in principle be
                -- train_only for one outcome and test_only for another when
                -- an outcome is NULL across a whole window.
                SELECT primary_metric, secondary_metric, corner_direction,
                       (ARRAY_AGG(side    ORDER BY pop_n DESC))[1] AS side,
                       MAX(pop_n) AS corner_n,
                       (ARRAY_AGG(outcome ORDER BY pop_n DESC))[1] AS top_outcome
                FROM ow
                GROUP BY primary_metric, secondary_metric, corner_direction
            )
            SELECT * FROM per_corner ORDER BY corner_n DESC LIMIT $1
        """, args.sample)
        mismatches = 0
        print(f"  {'corner':<52} {'pop n':>7} {'empty side':>16}")
        print(f"  {'-'*52} {'-'*7} {'-'*16}")
        for r in sample:
            empty_win = "test" if r["side"] == "train_only" else "train"
            sql_n = await _joint_count(
                conn, r["primary_metric"], r["secondary_metric"],
                r["corner_direction"], r["top_outcome"], 10, empty_win, cutoff)
            if sql_n != 0:
                mismatches += 1
            flag = "" if sql_n == 0 else "  <- MISMATCH"
            name = (f"{r['primary_metric']}x{r['secondary_metric']} "
                    f"[{r['corner_direction']}]")
            print(f"  {name[:52]:<52} {int(r['corner_n']):>7,} "
                  f"{empty_win}={sql_n:<9}{flag}")

        # ── 6. Anatomy of TEST-ONLY corners ─────────────────────────────────
        print("\n" + "═" * 74)
        print("6. ANATOMY OF TEST-ONLY CORNERS  (the odd direction)")
        print("═" * 74)
        test_only = await conn.fetch(f"""
            {_OW_CTE},
            per_corner AS (
                SELECT primary_metric, secondary_metric, corner_direction,
                       MAX(pop_n) AS corner_n
                FROM ow WHERE side = 'test_only'
                GROUP BY primary_metric, secondary_metric, corner_direction
            )
            SELECT * FROM per_corner ORDER BY corner_n DESC LIMIT $1
        """, args.sample)
        thin_train = late_tickers = anti_corr = 0
        if not test_only:
            print("  none — every one-window corner is train-populated,")
            print("  which is the expected direction. Nothing to explain.")
        else:
            print("  train marginals are ~5% of train rows per bin20 BY")
            print("  CONSTRUCTION, so a healthy marginal with a zero joint is")
            print("  genuine anti-correlation; a tiny marginal is thin data.\n")
            for r in test_only:
                a, tk = await _anatomy(
                    conn, r["primary_metric"], r["secondary_metric"],
                    r["corner_direction"], cutoff)
                pm, sm = int(a["p_edge_train"]), int(a["s_edge_train"])
                if tk["n_late_tk"] and tk["n_late_tk"] == tk["n_test_tk"]:
                    verdict = "LATE-UNIVERSE TICKERS"
                    late_tickers += 1
                elif min(pm, sm) < args.thin_marginal:
                    verdict = "THIN TRAIN MARGINAL"
                    thin_train += 1
                else:
                    verdict = "anti-correlated (ok)"
                    anti_corr += 1
                print(f"  {r['primary_metric']}x{r['secondary_metric']} "
                      f"[{r['corner_direction']}]  test n={int(r['corner_n']):,}")
                print(f"      train marginals: P-edge={pm:,}  S-edge={sm:,}"
                      f"   joint_train={int(a['joint_train']):,}"
                      f"  joint_test={int(a['joint_test']):,}")
                print(f"      test tickers={int(tk['n_test_tk'])}"
                      f"  of which no train history={int(tk['n_late_tk'])}"
                      f"   -> {verdict}")

        # ── Verdict ──────────────────────────────────────────────────────────
        print("\n" + "═" * 74)
        print("VERDICT  (corner units)")
        print("═" * 74)
        problems = []
        if mismatches:
            problems.append(
                f"(c) {mismatches} sampled corner(s) have a NON-ZERO SQL count "
                f"on the side stored as empty — rows are being lost.")
        if substantial:
            problems.append(
                f"(c?) {substantial} corner(s) have n >= {args.substantial} on "
                f"the populated side with the other side empty.")
        if top_share > 25.0:
            problems.append(
                f"(b) one metric holds {top_share:.1f}% of corner slots — "
                f"looks like thin bin coverage, not a thin tail.")
        if thin_train:
            problems.append(
                f"(b) {thin_train} test-only corner(s) have a train marginal "
                f"below {args.thin_marginal} — thin train data, not "
                f"anti-correlation.")
        if late_tickers:
            problems.append(
                f"(info) {late_tickers} test-only corner(s) are driven purely "
                f"by tickers with no train history.")
        if problems:
            for p in problems:
                print("  ✗ " + p)
            print("\n  Do NOT relax the batch self-check yet.")
            sys.exit(1)
        print("  ✓ Spread across metrics, small n per corner, every sampled")
        print("    empty side independently confirms as zero, and any")
        print("    test-only corners are explained. Consistent with a genuine")
        print("    thin tail under frozen edges — safe to allow one-window")
        print("    corners in the self-check.")
    finally:
        await conn.close()


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sample", type=int, default=15,
                    help="Corners to SQL-recount / dissect (default 15)")
    ap.add_argument("--substantial", type=int, default=300,
                    help="Populated-side n per CORNER that warrants "
                         "suspicion (default 300)")
    ap.add_argument("--thin-marginal", type=int, default=200,
                    help="Train-window extreme-bin marginal below which a "
                         "test-only corner is called thin data rather than "
                         "anti-correlation (default 200)")
    asyncio.run(run(ap.parse_args()))


if __name__ == "__main__":
    main()
