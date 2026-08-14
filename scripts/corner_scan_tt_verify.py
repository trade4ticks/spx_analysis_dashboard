#!/usr/bin/env python3
"""Verify the train_test corner-scan rows before trusting the table.

Two independent checks, both run against the DB — neither shares code with
the numpy aggregation in corner_scan.py, so agreement is real evidence
rather than the same bug reflected twice.

  CHECK 1 — SPLIT CONSERVATION (exact; this one must pass)
      For each sampled corner, recompute train and test straight from
      tt_bins + daily_features in SQL and confirm:
          stored train == SQL train,  stored test == SQL test
          train_n + test_n == the full-window n over the same bins
          the n-weighted pooling of train and test == the full-window mean
      Any failure here means the split or the bin source is wrong.

  CHECK 2 — TT-TRAIN vs IS RECONCILIATION (approximate by design)
      Compare the TT train stat against the IS stat for the same corner
      over the same date range.

      These are CLOSE but NOT identical, and that is correct.  The corner
      definition is the same (P and S binned independently over the full
      population, as the 2D heatmap bins them) and the date range is the
      same, but the BIN SOURCE differs:

          is_bins — edges from full history, including post-cutoff data
          tt_bins — edges frozen at the cutoff, pre-cutoff data only

      Different edges put different rows in D1/D10, so the two disagree by
      however much the post-cutoff data would have moved the thresholds.
      The script quantifies exactly that: it reports the row-level overlap
      between the two bin assignments alongside the stat gap, so a
      discrepancy can be attributed rather than guessed at.

      A large stat gap WITH high bin overlap is a red flag.
      A large stat gap WITH low bin overlap is just the frozen edges doing
      their job.

Usage (VPS, project root, venv active):
    python scripts/corner_scan_tt_verify.py                  # top 15 by test n
    python scripts/corner_scan_tt_verify.py --limit 40
    python scripts/corner_scan_tt_verify.py \
        --primary <P> --secondary <S> --direction high-high --outcome ret_5d_fwd_oc

Exit status is non-zero if CHECK 1 fails for any sampled corner.
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

_TOL = 1e-9


def _edges(direction: str) -> tuple[str, str]:
    """'low-high' -> ('low', 'high')"""
    p_lbl, s_lbl = direction.split("-")
    return p_lbl, s_lbl


def _outcome_expr(outcome: str) -> str:
    """SQL expression for an outcome column, incl. the derived overnight gap."""
    if outcome == "overnight_gap":
        return "(df.spot_co / NULLIF(df.spot_pc, 0) - 1.0)"
    return f'df."{outcome}"'


async def _corner_stat(
    conn, bin_table: str, p: str, s: str, direction: str, outcome: str,
    n_bins: int, window: str, cutoff,
) -> tuple[int, float | None]:
    """Recompute ONE corner directly in SQL.

    window: 'train' (trade_date < cutoff) | 'test' (>= cutoff) | 'all'
    n_bins: 10 for the decile corner, 5 for the quintile corner

    Bin collapse is written out in SQL rather than reused from Python so
    this is a genuinely independent path:
        bin = ((bin20 - 1) * n_bins) / 20 + 1     (integer division)
    which is the same partition corner_scan.py computes in numpy and the
    same one _split_bins_by_cutoff() computes 0-indexed.
    """
    p_lbl, s_lbl = _edges(direction)
    p_edge = 1 if p_lbl == "low" else n_bins
    s_edge = 1 if s_lbl == "low" else n_bins

    conds = [
        f'bt."bin20_{p}" > 0',
        f'bt."bin20_{s}" > 0',
        f'((bt."bin20_{p}" - 1) * {n_bins}) / 20 + 1 = {p_edge}',
        f'((bt."bin20_{s}" - 1) * {n_bins}) / 20 + 1 = {s_edge}',
    ]
    params: list = []
    if window == "train":
        params.append(cutoff)
        conds.append(f"bt.trade_date < ${len(params)}")
    elif window == "test":
        params.append(cutoff)
        conds.append(f"bt.trade_date >= ${len(params)}")

    expr = _outcome_expr(outcome)
    sql = (
        f"SELECT COUNT({expr}) AS n, AVG({expr}) AS avg_ret "
        f"FROM {bin_table} bt "
        f"JOIN daily_features df "
        f"  ON bt.ticker = df.ticker AND bt.trade_date = df.trade_date "
        f"WHERE {' AND '.join(conds)}"
    )
    row = await conn.fetchrow(sql, *params)
    n = int(row["n"] or 0)
    avg = float(row["avg_ret"]) if row["avg_ret"] is not None else None
    return n, avg


async def _bin_overlap(conn, p: str, s: str, direction: str, cutoff) -> dict:
    """How much do is_bins and tt_bins agree on WHICH rows are in the corner?

    Restricted to the train window, where both tables describe the same
    dates.  Returns counts of the corner membership under each table and
    of their intersection.
    """
    p_lbl, s_lbl = _edges(direction)
    p_edge = 1 if p_lbl == "low" else 10
    s_edge = 1 if s_lbl == "low" else 10

    def member(tbl_alias: str) -> str:
        return (
            f'{tbl_alias}."bin20_{p}" > 0 AND {tbl_alias}."bin20_{s}" > 0 '
            f'AND (({tbl_alias}."bin20_{p}" - 1) * 10) / 20 + 1 = {p_edge} '
            f'AND (({tbl_alias}."bin20_{s}" - 1) * 10) / 20 + 1 = {s_edge}'
        )

    sql = f"""
        WITH tt AS (
            SELECT ticker, trade_date FROM tt_bins bt
            WHERE bt.trade_date < $1 AND {member('bt')}
        ),
        isb AS (
            SELECT ticker, trade_date FROM is_bins ib
            WHERE ib.trade_date < $1 AND {member('ib')}
        )
        SELECT (SELECT COUNT(*) FROM tt)  AS tt_n,
               (SELECT COUNT(*) FROM isb) AS is_n,
               (SELECT COUNT(*) FROM tt JOIN isb USING (ticker, trade_date))
                   AS both_n
    """
    row = await conn.fetchrow(sql, cutoff)
    return {"tt_n": int(row["tt_n"]), "is_n": int(row["is_n"]),
            "both_n": int(row["both_n"])}


def _pct(v: float | None) -> str:
    return "     —  " if v is None else f"{v * 100:8.4f}%"


async def run(args) -> None:
    dsn = os.getenv("OI_DATABASE_URL")
    if not dsn:
        print("ERROR: OI_DATABASE_URL not set (check .env or environment).")
        sys.exit(1)

    conn = await asyncpg.connect(dsn=dsn)
    try:
        cutoff = await conn.fetchval("SELECT MAX(cutoff_date) FROM tt_bins")
        if cutoff is None:
            print("ERROR: tt_bins has no cutoff_date — rebuild it upstream.")
            sys.exit(1)
        print(f"Frozen TT cutoff (from tt_bins): {cutoff}\n")

        # ── Pick the corners to verify ───────────────────────────────────────
        if args.primary and args.secondary and args.direction and args.outcome:
            rows = await conn.fetch(
                """SELECT primary_metric, secondary_metric, corner_direction,
                          outcome, d_train_avg_ret, d_train_n,
                          d_test_avg_ret, d_test_n,
                          q_train_avg_ret, q_train_n,
                          q_test_avg_ret, q_test_n, cutoff_date
                   FROM corner_scan_2f
                   WHERE mode = 'train_test'
                     AND primary_metric = $1 AND secondary_metric = $2
                     AND corner_direction = $3 AND outcome = $4""",
                args.primary, args.secondary, args.direction, args.outcome,
            )
            if not rows:
                print("No stored train_test row for that corner.")
                sys.exit(1)
        else:
            # Corners that exist in BOTH modes, ordered by test n — the ones
            # with enough sample to be worth reconciling.
            rows = await conn.fetch(
                """SELECT tt.primary_metric, tt.secondary_metric,
                          tt.corner_direction, tt.outcome,
                          tt.d_train_avg_ret, tt.d_train_n,
                          tt.d_test_avg_ret,  tt.d_test_n,
                          tt.q_train_avg_ret, tt.q_train_n,
                          tt.q_test_avg_ret,  tt.q_test_n, tt.cutoff_date
                   FROM corner_scan_2f tt
                   JOIN corner_scan_2f isr
                     ON  isr.mode             = 'in_sample'
                     AND isr.primary_metric   = tt.primary_metric
                     AND isr.secondary_metric = tt.secondary_metric
                     AND isr.corner_direction = tt.corner_direction
                     AND isr.outcome          = tt.outcome
                   WHERE tt.mode = 'train_test' AND tt.d_test_n IS NOT NULL
                   ORDER BY tt.d_test_n DESC
                   LIMIT $1""",
                args.limit,
            )
            if not rows:
                print("No train_test rows found that also exist in in_sample.")
                print("Run: python scripts/corner_scan.py --mode train_test")
                sys.exit(1)

        failures = 0

        for r in rows:
            p, s = r["primary_metric"], r["secondary_metric"]
            d, o = r["corner_direction"], r["outcome"]
            print("═" * 78)
            print(f"{p} × {s}   [{d}]   {o}")
            print("═" * 78)

            # ── CHECK 1: split conservation, decile + quintile ───────────────
            print("  CHECK 1 — split conservation (tt_bins, SQL recompute)")
            for res, n_bins, tr_a, tr_n, te_a, te_n in (
                ("D", 10, r["d_train_avg_ret"], r["d_train_n"],
                          r["d_test_avg_ret"],  r["d_test_n"]),
                ("Q",  5, r["q_train_avg_ret"], r["q_train_n"],
                          r["q_test_avg_ret"],  r["q_test_n"]),
            ):
                sql_tr_n, sql_tr_a = await _corner_stat(
                    conn, "tt_bins", p, s, d, o, n_bins, "train", cutoff)
                sql_te_n, sql_te_a = await _corner_stat(
                    conn, "tt_bins", p, s, d, o, n_bins, "test", cutoff)
                sql_all_n, sql_all_a = await _corner_stat(
                    conn, "tt_bins", p, s, d, o, n_bins, "all", cutoff)

                stored_tr_n = int(tr_n or 0)
                stored_te_n = int(te_n or 0)

                ok_tr = (stored_tr_n == sql_tr_n and (
                    (tr_a is None and sql_tr_a is None)
                    or (tr_a is not None and sql_tr_a is not None
                        and abs(tr_a - sql_tr_a) < 1e-9)))
                ok_te = (stored_te_n == sql_te_n and (
                    (te_a is None and sql_te_a is None)
                    or (te_a is not None and sql_te_a is not None
                        and abs(te_a - sql_te_a) < 1e-9)))
                ok_cons = (sql_tr_n + sql_te_n == sql_all_n)

                # n-weighted pooling of the two windows == the full window
                if sql_tr_a is not None and sql_te_a is not None and sql_all_n:
                    pooled = (sql_tr_a * sql_tr_n + sql_te_a * sql_te_n) / sql_all_n
                    ok_pool = sql_all_a is not None and abs(pooled - sql_all_a) < 1e-9
                else:
                    ok_pool = True

                mark = "✓" if (ok_tr and ok_te and ok_cons and ok_pool) else "✗ FAIL"
                if mark != "✓":
                    failures += 1
                print(f"    {mark}  {res}: stored train {_pct(tr_a)} n={stored_tr_n:>7,}"
                      f"   SQL {_pct(sql_tr_a)} n={sql_tr_n:>7,}")
                print(f"        {' ':2} stored test  {_pct(te_a)} n={stored_te_n:>7,}"
                      f"   SQL {_pct(sql_te_a)} n={sql_te_n:>7,}")
                print(f"        {' ':2} conservation: {sql_tr_n:,} + {sql_te_n:,} "
                      f"= {sql_tr_n + sql_te_n:,} vs full {sql_all_n:,} "
                      f"{'✓' if ok_cons else '✗'}"
                      f"   pooled-mean {'✓' if ok_pool else '✗'}")

            # ── CHECK 2: TT train vs IS, same date range ─────────────────────
            print("\n  CHECK 2 — TT train vs IS over the same range "
                  f"(< {cutoff})")
            is_n, is_a = await _corner_stat(
                conn, "is_bins", p, s, d, o, 10, "train", cutoff)
            tt_tr_a, tt_tr_n = r["d_train_avg_ret"], int(r["d_train_n"] or 0)
            ov = await _bin_overlap(conn, p, s, d, cutoff)

            gap = (None if (tt_tr_a is None or is_a is None)
                   else abs(tt_tr_a - is_a))
            denom = max(ov["tt_n"], ov["is_n"]) or 1
            overlap_pct = 100.0 * ov["both_n"] / denom

            print(f"    TT train (tt_bins): {_pct(tt_tr_a)}  n={tt_tr_n:>7,}")
            print(f"    IS       (is_bins): {_pct(is_a)}  n={is_n:>7,}")
            if gap is not None:
                print(f"    gap: {gap * 100:.4f} pp")
            print(f"    corner membership overlap (train window): "
                  f"{ov['both_n']:,} of {denom:,} rows = {overlap_pct:.1f}%")
            if overlap_pct >= 90 and gap is not None and gap > 0.002:
                print("    ⚠ large stat gap despite high bin overlap — "
                      "investigate before trusting this row.")
            else:
                print("    → difference attributable to frozen vs full-history "
                      "bin edges (expected).")
            print()

        print("═" * 78)
        print(f"Corners verified: {len(rows)}")
        print(f"CHECK 1 failures: {failures}")
        print("RESULT:", "PASS ✓" if failures == 0 else "FAIL ✗")
        if failures:
            sys.exit(1)
    finally:
        await conn.close()


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--limit", type=int, default=15,
                    help="How many corners to sample (default 15)")
    ap.add_argument("--primary")
    ap.add_argument("--secondary")
    ap.add_argument("--direction", help="e.g. high-high")
    ap.add_argument("--outcome",   help="e.g. ret_5d_fwd_oc")
    asyncio.run(run(ap.parse_args()))


if __name__ == "__main__":
    main()
