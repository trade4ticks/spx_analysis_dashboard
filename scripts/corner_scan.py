#!/usr/bin/env python3
"""Offline corner-scan job — sweeps all eligible metric pairs for hot corners.

A 'corner' is a metric-pair (P × S) where both metrics are simultaneously in
an extreme bin (D1/D10 decile or Q1/Q5 quintile) on the same trade-date.

Bin assignments are read directly from the stored bin tables (wf_bins or
is_bins) — NOT recomputed locally.  This guarantees every corner row reflects
the SAME bin assignment every other dashboard view uses for the same
(ticker, trade_date, metric) triple.  "Algorithmically equivalent" local
recomputation was the old approach; this is the correct approach.

Output tables (OI DB):
  corner_scan_2f  — metric-pair corners  (PK: P, S, direction, outcome, mode)
  corner_scan_1f  — single-metric extremes (PK: metric, extreme, outcome, mode)

Both tables are mode-partitioned.  WF, IS and TT rows coexist; running one
mode never touches another mode's rows.

TRAIN-TEST MODE
---------------
`--mode train_test` reads tt_bins and splits the SAME corners into a train
window and a test window.  The corner definition is unchanged from IS/WF:
P and S are binned independently over the full population, exactly as the
2D heatmap bins them, so a scan row still points at the heatmap cell you
would open to define a zone.  Only the window differs.

Bin edges are frozen upstream.  tt_bins is built by build_bin_tables.py
--build-tt-bins in the Open_Interest data project, with edges fixed at
TT_CUTOFF_DATE and never re-derived — each rebuild only classifies rows
against those fixed thresholds.  This script READS tt_bins; it never
computes or refreshes an edge.  The cutoff is read back from the table
(MAX(cutoff_date)) and is NOT a command-line parameter, so there is no way
to evaluate against a split the bins were not built for.

Uneven bin counts between train and test are EXPECTED and correct.  Even
counts would require hindsight — the edges were fixed before the test
window existed.  Do not "fix" this.

TT writes corner_scan_2f only.  Phase 5 (1F) is skipped: the 1F pane's TT
toggle is still disabled, and writing rows behind a toggle that refuses to
load them is worse than writing nothing.

Run monthly on the VPS (project root, venv active):
    python scripts/corner_scan.py --mode walk_forward [--force] [--dry-run]
    python scripts/corner_scan.py --mode in_sample    [--force] [--dry-run]
    python scripts/corner_scan.py --mode train_test   [--force] [--dry-run]

    --mode     walk_forward reads wf_bins; in_sample reads is_bins;
               train_test reads tt_bins                             (required)
    --force    re-run even if this mode's rows were already written today
    --dry-run  compute and print row counts; do not write to DB

Exit status: non-zero if any metric was skipped for having no usable bins,
so a wrapper can notice a stale tt_bins column without reading the log.
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time
from datetime import date as _date, datetime, timezone
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(_ROOT / ".env")

import asyncpg  # noqa: E402

# ── Constants ─────────────────────────────────────────────────────────────────

_FWDRET_OUTCOMES: list[str] = [
    "ret_1d_fwd_oc",  "ret_3d_fwd_oc",  "ret_5d_fwd_oc",
    "ret_7d_fwd_oc",  "ret_10d_fwd_oc", "ret_20d_fwd_oc",
    "ret_1d_fwd_cc",  "ret_3d_fwd_cc",  "ret_5d_fwd_cc",
    "ret_7d_fwd_cc",  "ret_10d_fwd_cc", "ret_20d_fwd_cc",
]

# 13 total outcomes (12 fwd-returns + overnight gap)
_OUTCOME_LIST: list[str] = _FWDRET_OUTCOMES + ["overnight_gap"]
_O = len(_OUTCOME_LIST)  # 13

# Holding days for ret_per_day; None = overnight_gap (not per-day-comparable)
_HOLDING_DAYS: dict[str, int | None] = {
    "ret_1d_fwd_oc":  1,  "ret_3d_fwd_oc":  3,  "ret_5d_fwd_oc":  5,
    "ret_7d_fwd_oc":  7,  "ret_10d_fwd_oc": 10, "ret_20d_fwd_oc": 20,
    "ret_1d_fwd_cc":  1,  "ret_3d_fwd_cc":  3,  "ret_5d_fwd_cc":  5,
    "ret_7d_fwd_cc":  7,  "ret_10d_fwd_cc": 10, "ret_20d_fwd_cc": 20,
    "overnight_gap":  None,
}

# CC + gap outcomes: excluded when MORNING metric is on either P or S axis
_CC_OUTCOMES: frozenset[str] = frozenset(
    o for o in _OUTCOME_LIST if o.endswith("_cc") or o == "overnight_gap"
)

# ── DDL ───────────────────────────────────────────────────────────────────────
# Schema matches the live tables (created/migrated by _ensure_corner_scan_tables
# in oi_analysis.py).  Kept in sync here so the script is self-sufficient on
# a clean DB without depending on the endpoint running first.

_DDL_2F = """
CREATE TABLE IF NOT EXISTS corner_scan_2f (
    primary_metric    TEXT NOT NULL,
    secondary_metric  TEXT NOT NULL,
    corner_direction  TEXT NOT NULL,
    outcome           TEXT NOT NULL,
    d_avg_ret         DOUBLE PRECISION,
    d_ret_per_day     DOUBLE PRECISION,
    d_n               INTEGER,
    q_avg_ret         DOUBLE PRECISION,
    q_ret_per_day     DOUBLE PRECISION,
    q_n               INTEGER,
    d_train_avg_ret   DOUBLE PRECISION,
    d_train_n         INTEGER,
    d_test_avg_ret    DOUBLE PRECISION,
    d_test_n          INTEGER,
    q_train_avg_ret   DOUBLE PRECISION,
    q_train_n         INTEGER,
    q_test_avg_ret    DOUBLE PRECISION,
    q_test_n          INTEGER,
    cutoff_date       DATE,
    as_of             DATE        NOT NULL,
    mode              TEXT        NOT NULL DEFAULT 'walk_forward',
    scanned_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (primary_metric, secondary_metric, corner_direction, outcome, mode)
);
"""

# Migration for tables created before the train-test columns existed.
# Mirrors _DDL_CORNER_2F_MIGRATE_TT in app/routers/oi_analysis.py — keep
# the two in sync.  Idempotent.
_DDL_2F_MIGRATE_TT = """
ALTER TABLE corner_scan_2f
    ADD COLUMN IF NOT EXISTS d_train_avg_ret DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS d_train_n       INTEGER,
    ADD COLUMN IF NOT EXISTS d_test_avg_ret  DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS d_test_n        INTEGER,
    ADD COLUMN IF NOT EXISTS q_train_avg_ret DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS q_train_n       INTEGER,
    ADD COLUMN IF NOT EXISTS q_test_avg_ret  DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS q_test_n        INTEGER,
    ADD COLUMN IF NOT EXISTS cutoff_date     DATE;
"""

_DDL_1F = """
CREATE TABLE IF NOT EXISTS corner_scan_1f (
    metric        TEXT NOT NULL,
    extreme       TEXT NOT NULL,
    outcome       TEXT NOT NULL,
    d_avg_ret     DOUBLE PRECISION,
    d_ret_per_day DOUBLE PRECISION,
    d_n           INTEGER,
    q_avg_ret     DOUBLE PRECISION,
    q_ret_per_day DOUBLE PRECISION,
    q_n           INTEGER,
    as_of         DATE        NOT NULL,
    mode          TEXT        NOT NULL DEFAULT 'walk_forward',
    scanned_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (metric, extreme, outcome, mode)
);
"""

# ── Helpers ───────────────────────────────────────────────────────────────────

def _avg_or_none(total: float, n: int) -> float | None:
    return float(total / n) if n > 0 else None


def _rpd(avg: float | None, days: int | None) -> float | None:
    """Per-day average; None if avg is None or outcome is not per-day-comparable."""
    if avg is None or days is None:
        return None
    return float(avg / days)


# How many one-window corners the batch re-counts in SQL before trusting
# them.  Ordered by populated-side n, so the sample is the corners where a
# lost-rows bug would show up most clearly rather than a random draw.
_RECOUNT_SAMPLE = 25


def _cs_outcome_expr(outcome: str) -> str:
    """SQL for an outcome column, including the derived overnight gap."""
    if outcome == "overnight_gap":
        return "(df.spot_co / NULLIF(df.spot_pc, 0) - 1.0)"
    return f'df."{outcome}"'


async def _recount_empty_sides(conn, cutoff, limit: int) -> list[str]:
    """Re-count, straight from tt_bins, the window each corner stored as empty.

    A one-window corner is legitimate only if the empty side really is empty.
    This recomputes it in SQL — a path that shares no code with the numpy
    aggregation — and returns a description for every corner where the two
    disagree.  Empty list means every sampled corner checks out.
    """
    rows = await conn.fetch(
        """SELECT primary_metric, secondary_metric, corner_direction,
                  (ARRAY_AGG(side    ORDER BY pop_n DESC))[1] AS side,
                  (ARRAY_AGG(outcome ORDER BY pop_n DESC))[1] AS outcome,
                  MAX(pop_n) AS corner_n
           FROM (
               SELECT primary_metric, secondary_metric, corner_direction,
                      outcome,
                      COALESCE(d_train_n, d_test_n) AS pop_n,
                      CASE WHEN d_test_n IS NULL THEN 'train_only'
                           ELSE 'test_only' END     AS side
               FROM corner_scan_2f
               WHERE mode = 'train_test'
                 AND (d_train_n IS NULL) <> (d_test_n IS NULL)
           ) t
           GROUP BY primary_metric, secondary_metric, corner_direction
           ORDER BY MAX(pop_n) DESC
           LIMIT $1""",
        limit,
    )
    bad: list[str] = []
    for r in rows:
        p, s = r["primary_metric"], r["secondary_metric"]
        p_lbl, s_lbl = r["corner_direction"].split("-")
        p_edge = 1 if p_lbl == "low" else 10
        s_edge = 1 if s_lbl == "low" else 10
        empty_win = "test" if r["side"] == "train_only" else "train"
        date_cond = ("bt.trade_date < $1" if empty_win == "train"
                     else "bt.trade_date >= $1")
        expr = _cs_outcome_expr(r["outcome"])
        n = await conn.fetchval(
            f"""SELECT COUNT({expr}) FROM tt_bins bt
                JOIN daily_features df
                  ON bt.ticker = df.ticker AND bt.trade_date = df.trade_date
                WHERE bt."bin20_{p}" > 0 AND bt."bin20_{s}" > 0
                  AND ((bt."bin20_{p}" - 1) * 10) / 20 + 1 = {p_edge}
                  AND ((bt."bin20_{s}" - 1) * 10) / 20 + 1 = {s_edge}
                  AND {date_cond}""",
            cutoff,
        )
        if int(n or 0) != 0:
            bad.append(
                f"{p} x {s} [{r['corner_direction']}] {r['outcome']}: "
                f"stored {empty_win} empty, SQL says {int(n):,}"
            )
    return bad


def _print_skipped_summary(
    skipped: list, bin_table: str, is_tt: bool, cutoff_iso: str
) -> None:
    """Restate the skipped-metric list at the very END of the run.

    Phase 0 has scrolled far off screen by the time a batch finishes, so the
    list is printed twice on purpose: once where it is discovered, once here
    where it is actually read.  Callers exit non-zero when this is non-empty.
    """
    print("\n── Skipped metrics ─────────────────────────────────────────────")
    if not skipped:
        print(f"  none — every eligible metric had usable bins in {bin_table}.")
        return
    print(
        f"  ⚠ {len(skipped)} metric(s) had NO usable bins in {bin_table} and "
        f"were excluded\n"
        f"    from the scan entirely (no rows written for any pair involving "
        f"them).\n"
        f"    Most likely cause: the column was added to {bin_table} after the "
        f"last full\n"
        f"    build, so bin20_<metric> exists but every row is NULL. Rebuild "
        f"{bin_table}\n"
        f"    upstream, then re-run this scan."
    )
    if is_tt:
        print(f"    Window split at the frozen cutoff {cutoff_iso}.")
    print()
    hdr = ("train rows / test rows" if is_tt else "usable rows")
    print(f"    {'metric':<48} {hdr}")
    print(f"    {'-' * 48} {'-' * len(hdr)}")
    for m_name, tr, te in skipped:
        detail = f"{tr:,} / {te:,}" if is_tt else f"{tr:,}"
        print(f"    {m_name:<48} {detail}")
    print("\n  Exit status will be non-zero because of the above.")


# ── Main ──────────────────────────────────────────────────────────────────────

async def run(mode: str, dry_run: bool, force: bool) -> None:
    oi_dsn = os.getenv("OI_DATABASE_URL")
    if not oi_dsn:
        print("ERROR: OI_DATABASE_URL not set (check .env or environment).")
        sys.exit(1)

    bin_table = {
        "walk_forward": "wf_bins",
        "in_sample":    "is_bins",
        "train_test":   "tt_bins",
    }[mode]
    is_tt = mode == "train_test"

    t_total = time.perf_counter()
    conn = await asyncpg.connect(dsn=oi_dsn)

    try:
        await conn.execute(_DDL_2F)
        await conn.execute(_DDL_2F_MIGRATE_TT)
        await conn.execute(_DDL_1F)

        # ── TT: read the frozen cutoff from tt_bins ───────────────────────────
        # Same source as _get_tt_cutoff() in the router.  Never a CLI param:
        # the bins were built against exactly one split, and evaluating them
        # against any other split would silently mix train rows into test.
        cutoff_date: "_date | None" = None
        cutoff_iso  = ""
        if is_tt:
            cutoff_date = await conn.fetchval(
                "SELECT MAX(cutoff_date) FROM tt_bins"
            )
            if cutoff_date is None:
                print(
                    "ERROR: tt_bins has no cutoff_date. Rebuild it with\n"
                    "       build_bin_tables.py --build-tt-bins in the "
                    "Open_Interest project first."
                )
                sys.exit(1)
            cutoff_iso = cutoff_date.isoformat()
            print(f"TT frozen cutoff (from tt_bins): {cutoff_iso}")

        # Guard: skip if already run today FOR THIS MODE, unless --force.
        # Mode-aware: a WF run today does not block an IS run today.
        if not force and not dry_run:
            today_str    = str(_date.today())
            existing_aso = await conn.fetchval(
                "SELECT as_of FROM corner_scan_2f WHERE mode = $1 LIMIT 1",
                mode,
            )
            if existing_aso is not None and str(existing_aso) == today_str:
                print(
                    f"corner_scan_2f already populated for {today_str} "
                    f"(mode={mode}). Use --force to overwrite."
                )
                return

        # ── Phase 0: Setup ───────────────────────────────────────────────────
        print(f"Phase 0: Loading metric classification…  (mode={mode}, bin_table={bin_table})")
        t0 = time.perf_counter()

        class_rows = await conn.fetch(
            """SELECT metric, tier
               FROM metric_classification
               WHERE eligible_as_metric = true
               ORDER BY metric"""
        )
        if not class_rows:
            print(
                "ERROR: metric_classification is empty. "
                "Run load_metric_classification.py first."
            )
            sys.exit(1)

        eligible_by_name: dict[str, str] = {
            r["metric"]: r["tier"] for r in class_rows
        }

        # Discover which bin20 columns actually exist in the stored bin table.
        # This is the authoritative source: metrics absent here have no stored
        # bin (null-by-design or not yet precomputed).  Do NOT fall back to
        # daily_features column presence — that would reintroduce the
        # "independent recomputation" problem we're eliminating.
        bin_col_rows = await conn.fetch(
            """SELECT column_name FROM information_schema.columns
               WHERE table_name = $1 AND table_schema = 'public'
                 AND column_name LIKE 'bin20_%'""",
            bin_table,
        )
        available_bin20: set[str] = {
            r["column_name"][6:]  # strip 'bin20_' prefix → metric name
            for r in bin_col_rows
        }

        # Verify outcome columns exist in daily_features (sanity check).
        df_cols = {
            r["column_name"]
            for r in await conn.fetch(
                """SELECT column_name FROM information_schema.columns
                   WHERE table_name = 'daily_features' AND table_schema = 'public'"""
            )
        }
        missing_outcomes = [o for o in _FWDRET_OUTCOMES if o not in df_cols]
        if missing_outcomes:
            print(
                f"ERROR: outcome columns missing from daily_features: "
                f"{missing_outcomes}"
            )
            sys.exit(1)

        # Eligible metrics = in metric_classification AND have a bin20 column
        # in the stored bin table for this mode.
        eligible_metrics: list[str] = sorted(
            m for m in eligible_by_name if m in available_bin20
        )
        F = len(eligible_metrics)
        metric_tier: dict[str, str] = {
            m: eligible_by_name[m] for m in eligible_metrics
        }
        morning_set: frozenset[str] = frozenset(
            m for m, t in metric_tier.items() if t == "MORNING"
        )

        print(f"  Eligible metrics in {bin_table}: {F}")
        print(
            f"    MORNING: {sum(1 for t in metric_tier.values() if t == 'MORNING')}"
        )
        print(
            f"    EVENING: {sum(1 for t in metric_tier.values() if t == 'EVENING')}"
        )
        print(f"  Phase 0: {time.perf_counter() - t0:.1f}s")

        def excluded(p: str, s: str, outcome: str) -> bool:
            """True when this (P, S, outcome) trio should be skipped."""
            return outcome in _CC_OUTCOMES and (
                p in morning_set or s in morning_set
            )

        # ── Phase 1: Load stored bins + outcomes via JOIN ─────────────────────
        # Single JOIN query: bin20 columns from the stored bin table + outcome
        # columns from daily_features.  INNER JOIN ensures alignment — only
        # rows present in both tables are included.
        print(f"\nPhase 1: Loading {bin_table} bins + daily_features outcomes via JOIN…")
        t1 = time.perf_counter()

        bin_cols_sql = ", ".join(f'bt."bin20_{m}"' for m in eligible_metrics)
        out_cols_sql = ", ".join(f'df."{o}"'       for o in _FWDRET_OUTCOMES)

        db_rows = await conn.fetch(
            f"""SELECT bt.ticker, bt.trade_date,
                       {bin_cols_sql},
                       df.spot_co, df.spot_pc,
                       {out_cols_sql}
                FROM {bin_table} bt
                INNER JOIN daily_features df
                    ON bt.ticker     = df.ticker
                   AND bt.trade_date = df.trade_date
                ORDER BY bt.ticker, bt.trade_date""",
            timeout=300,
        )
        N_total = len(db_rows)
        print(f"  {N_total:,} rows in {time.perf_counter() - t1:.1f}s")

        # ── Phase 2: Build bin and outcome arrays from stored values ──────────
        print("\nPhase 2: Building arrays from stored bins…")
        t2 = time.perf_counter()

        # raw_bin20_full: (N, F) int32 — stored bin20 values, 0 = sentinel.
        # Populated column-by-column (better cache locality for the numpy write).
        raw_bin20_full = np.zeros((N_total, F),  dtype=np.int32)
        out_full       = np.full ((N_total, _O), np.nan, dtype=np.float64)
        vld_full       = np.zeros((N_total, _O), dtype=bool)

        for f_idx, feat in enumerate(eligible_metrics):
            col_name = f"bin20_{feat}"
            raw_bin20_full[:, f_idx] = [
                (v if v is not None else 0) for v in (row[col_name] for row in db_rows)
            ]

        # ── SENTINEL FILTER (critical): apply bin20 > 0 BEFORE converting ────
        # bin20 = 0 is the sentinel for warmup / NaN / null-by-design.
        # The low extreme (D1 / Q1) maps to bin20 = 1..2 / 1..4.
        # Without this guard a sentinel row (bin20 = 0) could propagate to
        # bin_d = 1 via ((0-1)*10)//20+1 = ... — we avoid the formula
        # entirely for sentinel rows by zeroing bins_d/q at those positions.
        mask_valid = (raw_bin20_full > 0)          # (N, F) bool

        bins_d_full = np.zeros((N_total, F), dtype=np.int32)
        bins_q_full = np.zeros((N_total, F), dtype=np.int32)
        b20v = raw_bin20_full[mask_valid]           # 1-D slice of valid values
        bins_d_full[mask_valid] = ((b20v - 1) * 10) // 20 + 1  # → 1..10
        bins_q_full[mask_valid] = ((b20v - 1) * 5)  // 20 + 1  # → 1..5
        # Rows where mask_valid is False stay 0 — excluded from extremes
        # naturally (0 ≠ 1 and 0 ≠ 10/5 in Phase 5/6 comparison masks).

        del raw_bin20_full  # free memory; no longer needed

        # ── Train/test row mask ──────────────────────────────────────────────
        # Same rule as _split_bins_by_cutoff() in app/routers/oi_analysis.py:
        #   train = trade_date <  cutoff,  test = everything else.
        # That helper is the canonical splitter for every per-bin train-vs-test
        # surface, but it is row-dict shaped and handles one metric × one
        # outcome per call; this scan is an (N,F)×(N,O) matmul over every pair,
        # so we port the rule rather than the function.  The bin collapse
        # already agrees: its  bn = min(((b20-1)*n)//20, n-1)  is this file's
        # bins_d/bins_q minus one (0-indexed vs 1-indexed), so its D1/D10 are
        # our 1/10 exactly.
        if is_tt:
            is_train_full = np.fromiter(
                (r["trade_date"] < cutoff_date for r in db_rows),
                dtype=bool, count=N_total,
            )
        else:
            # Non-TT modes have no split; a single "all rows" window keeps
            # the Phase 6 loop uniform across modes.
            is_train_full = np.zeros(N_total, dtype=bool)

        # ── Usable-bin guard (question B) ────────────────────────────────────
        # A metric added to the bin table after the last full build has its
        # bin20_<m> column present but every row NULL.  Phase 0 discovers
        # eligibility from information_schema, so such a metric PASSES —
        # then maps to all-zero sentinels, matches no extreme, and emits a
        # full set of corners with NULL stats and NULL n.  Those rows are
        # then invisible in the UI (NULL >= min_n is never true), so the
        # batch looks successful and the pane looks empty for no stated
        # reason.  Skip and report instead: drop the metric before the pair
        # loop (which also shrinks F), list it in the FINAL summary, and
        # exit non-zero so a wrapper notices.
        usable_all = mask_valid.sum(axis=0)                      # (F,)
        if is_tt:
            usable_train = mask_valid[is_train_full].sum(axis=0)
            usable_test  = mask_valid[~is_train_full].sum(axis=0)
        else:
            usable_train = usable_all
            usable_test  = usable_all

        skipped_metrics: list[tuple[str, int, int]] = []
        keep_idx: list[int] = []
        for f_idx, m_name in enumerate(eligible_metrics):
            tr, te = int(usable_train[f_idx]), int(usable_test[f_idx])
            # TT needs bins in BOTH windows — a metric with train rows but no
            # test rows can produce a train number with nothing to check it
            # against, which is exactly the misleading half-result we are
            # avoiding.
            if tr == 0 or te == 0:
                skipped_metrics.append((m_name, tr, te))
            else:
                keep_idx.append(f_idx)

        if skipped_metrics:
            print(
                f"\n  ⚠ SKIPPED {len(skipped_metrics)} metric(s) with no usable "
                f"bins in {bin_table} — excluded from the scan:"
            )
            for m_name, tr, te in skipped_metrics:
                detail = (f"train={tr:,} test={te:,}" if is_tt
                          else f"usable rows={tr:,}")
                print(f"      {m_name:<48} {detail}")

        if not keep_idx:
            print(
                f"\nERROR: no metric in {bin_table} has usable bins. "
                f"The bin table is empty or was never populated."
            )
            sys.exit(1)

        if len(keep_idx) != F:
            keep_arr        = np.array(keep_idx, dtype=np.intp)
            bins_d_full     = bins_d_full[:, keep_arr]
            bins_q_full     = bins_q_full[:, keep_arr]
            eligible_metrics = [eligible_metrics[i] for i in keep_idx]
            F               = len(eligible_metrics)
            metric_tier     = {m: metric_tier[m] for m in eligible_metrics}
            morning_set     = frozenset(
                m for m, t in metric_tier.items() if t == "MORNING"
            )
            print(f"  Metrics after usable-bin filter: {F}")

        del mask_valid

        # Outcome matrix: 12 fwd-return cols + overnight gap.
        for o_idx, o_name in enumerate(_FWDRET_OUTCOMES):
            for i, row in enumerate(db_rows):
                v = row[o_name]
                if v is not None:
                    out_full[i, o_idx] = float(v)
                    vld_full[i, o_idx] = True

        # Overnight gap (col 12): O_T / C_{T-1} − 1
        spot_co = np.array(
            [float(r["spot_co"]) if r["spot_co"] is not None else np.nan
             for r in db_rows]
        )
        spot_pc = np.array(
            [float(r["spot_pc"]) if r["spot_pc"] is not None else np.nan
             for r in db_rows]
        )
        with np.errstate(invalid="ignore", divide="ignore"):
            gap = spot_co / spot_pc - 1.0
        out_full[:, 12] = gap
        vld_full[:, 12] = ~np.isnan(gap)

        # Build ticker slice map (rows are ORDER BY ticker, trade_date).
        tickers_order: list[str] = []
        ticker_slices: dict[str, tuple[int, int]] = {}
        prev_tkr:  str | None = None
        prev_start: int       = 0
        for i, row in enumerate(db_rows):
            tkr = row["ticker"]
            if tkr != prev_tkr:
                if prev_tkr is not None:
                    ticker_slices[prev_tkr] = (prev_start, i)
                tickers_order.append(tkr)
                prev_start = i
                prev_tkr   = tkr
        if prev_tkr is not None:
            ticker_slices[prev_tkr] = (prev_start, N_total)

        del db_rows  # release raw row dicts; arrays have all we need

        print(f"  {len(tickers_order)} tickers; arrays built in "
              f"{time.perf_counter() - t2:.1f}s")

        # ── Phase 5: 1-factor scan ───────────────────────────────────────────
        # Skipped for train_test: the 1F pane's TT toggle is still disabled,
        # so rows written here would sit behind a control that refuses to
        # load them.  corner_scan_1f is left completely untouched in TT mode
        # (Phase 7's mode-scoped DELETE is skipped too).
        as_of       = _date.today()
        scanned_now = datetime.now(tz=timezone.utc)
        rows_1f: list[tuple] = []

        if is_tt:
            print("\nPhase 5: 1-factor scan… SKIPPED (train_test writes 2F only)")
        else:
            print("\nPhase 5: 1-factor scan…")
            t5 = time.perf_counter()

            for f_idx, m_name in enumerate(eligible_metrics):
                for extreme, d_edge, q_edge in [("low", 1, 1), ("high", 10, 5)]:
                    d_mask = bins_d_full[:, f_idx] == d_edge  # (N_total,) bool
                    q_mask = bins_q_full[:, f_idx] == q_edge

                    for o_idx, o_name in enumerate(_OUTCOME_LIST):
                        d_vals = out_full[d_mask, o_idx]
                        d_vals = d_vals[~np.isnan(d_vals)]
                        q_vals = out_full[q_mask, o_idx]
                        q_vals = q_vals[~np.isnan(q_vals)]

                        d_n   = len(d_vals)
                        q_n   = len(q_vals)
                        d_avg = float(np.mean(d_vals)) if d_n else None
                        q_avg = float(np.mean(q_vals)) if q_n else None
                        days  = _HOLDING_DAYS[o_name]

                        rows_1f.append((
                            m_name, extreme, o_name,
                            d_avg,  _rpd(d_avg, days), d_n or None,
                            q_avg,  _rpd(q_avg, days), q_n or None,
                            as_of, mode, scanned_now,
                        ))

            print(f"  {len(rows_1f):,} rows in {time.perf_counter() - t5:.1f}s")

        # ── Phase 6: 2-factor scan ───────────────────────────────────────────
        print(f"\nPhase 6: 2-factor scan ({F} P-metric outer loop)…")
        t6      = time.perf_counter()
        rows_2f: list[tuple] = []

        # Window axis.  Non-TT modes run a single window ("all rows") so the
        # loop below is identical across modes; TT runs two.  The corner
        # DEFINITION is the same in both cases — P and S are binned
        # independently over the full population, exactly as the 2D heatmap
        # bins them, so a scan row still points at the heatmap cell you would
        # open to define a zone.  Only which rows are averaged differs.
        #
        # W_TRAIN / W_TEST index the leading axis of the accumulators.
        W_TRAIN, W_TEST = 0, 1
        n_win = 2 if is_tt else 1

        # Per-ticker split BOUNDARY, computed once.
        # Phase 1 selects ORDER BY ticker, trade_date, so within a ticker's
        # slice the dates ascend and the train window (trade_date < cutoff)
        # is a contiguous PREFIX.  That means the split is a single index and
        # both windows stay plain slice views — no boolean indexing, no
        # per-(P, ticker) copies.  Keeps the TT path's memory profile the
        # same as the single-window path it extends.
        split_at: dict[str, int] = {}
        if is_tt:
            for tkr in tickers_order:
                s, e = ticker_slices[tkr]
                # Counting the Trues gives the boundary directly, PROVIDED
                # the prefix property actually holds.  Verify it rather than
                # trust it: if the ORDER BY in Phase 1 ever changed, this
                # would misassign rows between train and test silently, which
                # is the worst possible failure for this table.
                w = is_train_full[s:e]
                cut = s + int(w.sum())
                if w.size and not (w[:cut - s].all() and not w[cut - s:].any()):
                    print(
                        f"ERROR: train rows are not a contiguous prefix for "
                        f"{tkr}. Phase 1 must SELECT ... ORDER BY ticker, "
                        f"trade_date for the train/test split to be valid."
                    )
                    sys.exit(1)
                split_at[tkr] = cut

        for p_idx, p_name in enumerate(eligible_metrics):
            # Per-P accumulators:
            # axes: [window, p_edge (0=low,1=high), s_edge (0=low,1=high),
            #        S_idx, outcome]
            # d_ = decile resolution,  q_ = quintile resolution
            d_sums = np.zeros((n_win, 2, 2, F, _O), dtype=np.float64)
            d_cnts = np.zeros((n_win, 2, 2, F, _O), dtype=np.int32)
            q_sums = np.zeros((n_win, 2, 2, F, _O), dtype=np.float64)
            q_cnts = np.zeros((n_win, 2, 2, F, _O), dtype=np.int32)

            for tkr in tickers_order:
                s, e = ticker_slices[tkr]

                # Windows for this ticker as (accumulator index, lo, hi) row
                # ranges.  TT: train = [s, split), test = [split, e) — the
                # same rule _split_bins_by_cutoff() applies, expressed as a
                # boundary because the rows are date-sorted.  Uneven
                # train/test bin counts are expected: the edges were frozen
                # before the test window existed, so matching counts would
                # require hindsight.
                if is_tt:
                    cut     = split_at[tkr]
                    windows = ((W_TRAIN, s, cut), (W_TEST, cut, e))
                else:
                    windows = ((W_TRAIN, s, e),)

                for w_idx, lo, hi in windows:
                    # Skip only an EMPTY window. A 1-row window is real data
                    # and must be counted — same reasoning as the n_pe guard
                    # below, which this used to mask.
                    if hi <= lo:
                        continue
                    bd_t  = bins_d_full[lo:hi, :]  # (N_w, F) stored deciles
                    bq_t  = bins_q_full[lo:hi, :]  # (N_w, F) stored quintiles
                    out_t = out_full[lo:hi, :]     # (N_w, O)
                    vld_t = vld_full[lo:hi, :]     # (N_w, O) bool

                    # Four P-edge configurations: D-low, D-high, Q-low, Q-high.
                    # S-bin matrix for D configs is bd_t (decile 1..10);
                    # for Q configs it's bq_t (quintile 1..5).
                    # bin_d/q = 0 rows (sentinel) are excluded naturally —
                    # 0 ≠ 1 and 0 ≠ 10/5, so they never enter a P or S edge mask.
                    pe_configs = [
                        (bd_t[:, p_idx],  1, bd_t, 10, d_sums, d_cnts, 0),  # D-low
                        (bd_t[:, p_idx], 10, bd_t, 10, d_sums, d_cnts, 1),  # D-high
                        (bq_t[:, p_idx],  1, bq_t,  5, q_sums, q_cnts, 0),  # Q-low
                        (bq_t[:, p_idx],  5, bq_t,  5, q_sums, q_cnts, 1),  # Q-high
                    ]

                    for p_col, p_edge_val, S_bin_mat, n_bins_S, sums_ref, cnts_ref, p_ei in pe_configs:
                        pe_rows = np.where(p_col == p_edge_val)[0]
                        n_pe    = len(pe_rows)
                        # Skip only a GENUINELY EMPTY P-edge. This used to be
                        # `n_pe < 2`, which silently discarded a ticker's whole
                        # contribution to every corner whenever that ticker had
                        # exactly ONE row at the P extreme in this window.
                        #
                        # That made the scan orientation-dependent: the guard
                        # keys on P, so (A×B) and (B×A) — the same symmetric
                        # corner — dropped different tickers and returned
                        # different counts. The loss accumulates across
                        # tickers (it is NOT bounded at one row) and grows as
                        # the window shrinks, which is why it showed up in the
                        # TT test window while train matched SQL exactly.
                        #
                        # There is no numerical reason for the old threshold:
                        # an (F,1)@(1,O) matmul is well-defined, and sums and
                        # counts are linear. Empty is skipped purely as an
                        # optimisation.
                        if n_pe == 0:
                            continue

                        S_bins  = S_bin_mat[pe_rows, :]    # (n_pe, F) stored bins
                        out_pe  = out_t[pe_rows, :]         # (n_pe, O)
                        vld_pe  = vld_t[pe_rows, :]         # (n_pe, O) bool

                        # NaN outcomes → 0 so matmul sums correctly; vld_pe tracks counts.
                        out_cln = np.where(vld_pe, out_pe, 0.0)  # (n_pe, O)

                        S_low_f  = (S_bins == 1       ).astype(np.float64)  # (n_pe, F)
                        S_high_f = (S_bins == n_bins_S).astype(np.float64)  # (n_pe, F)
                        vld_f    = vld_pe.astype(np.float64)                  # (n_pe, O)

                        # Vectorised aggregation over all (S_metric, outcome) at once.
                        # S_low_f.T: (F, n_pe) @ (n_pe, O) → (F, O)
                        sums_ref[w_idx, p_ei, 0] += S_low_f.T  @ out_cln
                        cnts_ref[w_idx, p_ei, 0] += (S_low_f.T  @ vld_f).astype(np.int32)
                        sums_ref[w_idx, p_ei, 1] += S_high_f.T @ out_cln
                        cnts_ref[w_idx, p_ei, 1] += (S_high_f.T @ vld_f).astype(np.int32)

            # Emit rows for all (S, direction, outcome) combos for this P.
            for s_idx, s_name in enumerate(eligible_metrics):
                if s_idx == p_idx:
                    continue  # skip self-pair
                for p_ei, p_lbl in [(0, "low"), (1, "high")]:
                    for s_ej, s_lbl in [(0, "low"), (1, "high")]:
                        direction = f"{p_lbl}-{s_lbl}"
                        for o_idx, o_name in enumerate(_OUTCOME_LIST):
                            if excluded(p_name, s_name, o_name):
                                continue

                            if is_tt:
                                # TT row: train + test stats side by side, the
                                # single-window columns left NULL.  Both n's
                                # are stored even though the pane shows test n
                                # only — it costs nothing and turns a future
                                # "show me train n too" into a display change
                                # rather than a re-scan.
                                d_tr_n = int(d_cnts[W_TRAIN, p_ei, s_ej, s_idx, o_idx])
                                d_te_n = int(d_cnts[W_TEST,  p_ei, s_ej, s_idx, o_idx])
                                q_tr_n = int(q_cnts[W_TRAIN, p_ei, s_ej, s_idx, o_idx])
                                q_te_n = int(q_cnts[W_TEST,  p_ei, s_ej, s_idx, o_idx])
                                rows_2f.append((
                                    p_name, s_name, direction, o_name,
                                    None, None, None,          # d_avg / d_rpd / d_n
                                    None, None, None,          # q_avg / q_rpd / q_n
                                    _avg_or_none(float(d_sums[W_TRAIN, p_ei, s_ej, s_idx, o_idx]), d_tr_n),
                                    d_tr_n or None,
                                    _avg_or_none(float(d_sums[W_TEST,  p_ei, s_ej, s_idx, o_idx]), d_te_n),
                                    d_te_n or None,
                                    _avg_or_none(float(q_sums[W_TRAIN, p_ei, s_ej, s_idx, o_idx]), q_tr_n),
                                    q_tr_n or None,
                                    _avg_or_none(float(q_sums[W_TEST,  p_ei, s_ej, s_idx, o_idx]), q_te_n),
                                    q_te_n or None,
                                    cutoff_date,
                                    as_of, mode, scanned_now,
                                ))
                            else:
                                d_n   = int(d_cnts[W_TRAIN, p_ei, s_ej, s_idx, o_idx])
                                q_n   = int(q_cnts[W_TRAIN, p_ei, s_ej, s_idx, o_idx])
                                d_avg = _avg_or_none(
                                    float(d_sums[W_TRAIN, p_ei, s_ej, s_idx, o_idx]), d_n
                                )
                                q_avg = _avg_or_none(
                                    float(q_sums[W_TRAIN, p_ei, s_ej, s_idx, o_idx]), q_n
                                )
                                days  = _HOLDING_DAYS[o_name]
                                rows_2f.append((
                                    p_name, s_name, direction, o_name,
                                    d_avg,  _rpd(d_avg, days), d_n or None,
                                    q_avg,  _rpd(q_avg, days), q_n or None,
                                    None, None, None, None,    # d_train/test
                                    None, None, None, None,    # q_train/test
                                    None,                      # cutoff_date
                                    as_of, mode, scanned_now,
                                ))

            # Progress print every 10 P-metrics.
            if (p_idx + 1) % 10 == 0 or p_idx == F - 1:
                elapsed = time.perf_counter() - t6
                rate    = (p_idx + 1) / elapsed
                eta     = (F - p_idx - 1) / rate if rate > 0 else 0.0
                print(
                    f"  P {p_idx + 1:>3}/{F}  {p_name:<42} "
                    f"elapsed={elapsed:.0f}s  ETA={eta:.0f}s"
                )

        print(
            f"  Phase 6 done: {len(rows_2f):,} rows in "
            f"{time.perf_counter() - t6:.0f}s"
        )

        # ── Phase 7: Write to DB (or dry-run summary) ────────────────────────
        phase6_secs = time.perf_counter() - t6

        if dry_run:
            print(f"\n── DRY RUN — no DB changes ──────────────────────────────")
            if is_tt:
                print(f"  corner_scan_1f would write: 0 rows  (skipped in train_test)")
            else:
                print(f"  corner_scan_1f would write: {len(rows_1f):,} rows  (mode={mode})")
            print(f"  corner_scan_2f would write: {len(rows_2f):,} rows  (mode={mode})")
            print(f"  Phase 6 runtime: {phase6_secs:.0f}s")
            print(f"  Total elapsed:   {time.perf_counter() - t_total:.0f}s")
            _print_skipped_summary(skipped_metrics, bin_table, is_tt, cutoff_iso)
            if skipped_metrics:
                sys.exit(1)
            return

        print(f"\nPhase 7: Writing to DB (mode={mode})…")
        t7 = time.perf_counter()

        async with conn.transaction():
            # Mode-scoped DELETE: remove only rows for the current mode.
            # Other modes' rows (e.g. IS rows when running WF) are preserved.
            # TT skips corner_scan_1f entirely — no DELETE, no INSERT — so a
            # TT run cannot disturb the WF/IS 1F rows the pane does serve.
            if is_tt:
                print("  corner_scan_1f: skipped (train_test writes 2F only).")
            else:
                await conn.execute(
                    "DELETE FROM corner_scan_1f WHERE mode = $1", mode
                )
                await conn.executemany(
                    """INSERT INTO corner_scan_1f
                       (metric, extreme, outcome,
                        d_avg_ret, d_ret_per_day, d_n,
                        q_avg_ret, q_ret_per_day, q_n,
                        as_of, mode, scanned_at)
                       VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12)""",
                    rows_1f,
                )
                print(f"  corner_scan_1f: {len(rows_1f):,} rows written.")

            await conn.execute(
                "DELETE FROM corner_scan_2f WHERE mode = $1", mode
            )
            await conn.executemany(
                """INSERT INTO corner_scan_2f
                   (primary_metric, secondary_metric, corner_direction, outcome,
                    d_avg_ret, d_ret_per_day, d_n,
                    q_avg_ret, q_ret_per_day, q_n,
                    d_train_avg_ret, d_train_n, d_test_avg_ret, d_test_n,
                    q_train_avg_ret, q_train_n, q_test_avg_ret, q_test_n,
                    cutoff_date,
                    as_of, mode, scanned_at)
                   VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,
                           $11,$12,$13,$14,$15,$16,$17,$18,$19,
                           $20,$21,$22)""",
                rows_2f,
            )
            print(f"  corner_scan_2f: {len(rows_2f):,} rows written.")

        print(f"  DB write done in {time.perf_counter() - t7:.1f}s")

        # ── Self-checks ───────────────────────────────────────────────────────
        print("\n── Self-checks ─────────────────────────────────────────────────")

        cc_bad = await conn.fetchval(
            """SELECT COUNT(*) FROM corner_scan_2f
               WHERE mode = $1
                 AND (outcome LIKE '%_cc' OR outcome = 'overnight_gap')
                 AND primary_metric IN (
                     SELECT metric FROM metric_classification
                     WHERE tier = 'MORNING')""",
            mode,
        )
        mark = "✓" if cc_bad == 0 else "✗ FAIL"
        print(f"  {mark}  CC/gap with MORNING primary [{mode}]: {cc_bad} (must be 0)")

        inelig_bad = await conn.fetchval(
            """SELECT COUNT(*) FROM corner_scan_2f
               WHERE mode = $1
                 AND primary_metric IN (
                     SELECT metric FROM metric_classification
                     WHERE eligible_as_metric = false)""",
            mode,
        )
        mark = "✓" if inelig_bad == 0 else "✗ FAIL"
        print(f"  {mark}  Rows with ineligible primary [{mode}]: {inelig_bad} (must be 0)")

        totals = await conn.fetchrow(
            """SELECT
                 (SELECT COUNT(*)                       FROM corner_scan_2f WHERE mode=$1) AS n_2f,
                 (SELECT COUNT(*)                       FROM corner_scan_1f WHERE mode=$1) AS n_1f,
                 (SELECT COUNT(DISTINCT primary_metric) FROM corner_scan_2f WHERE mode=$1) AS n_p,
                 (SELECT as_of                          FROM corner_scan_2f WHERE mode=$1 LIMIT 1) AS as_of""",
            mode,
        )
        # Orientation symmetry (all modes).  (A, B, "low-high") and
        # (B, A, "high-low") name the SAME set of trade-dates, so their
        # counts must be identical.  They diverged when a per-ticker guard
        # keyed on P dropped tickers asymmetrically; this check exists so
        # that class of bug can never return silently.
        n_col = "d_test_n" if is_tt else "d_n"
        asym = await conn.fetchrow(
            f"""SELECT COUNT(*) AS n_bad,
                       COALESCE(MAX(ABS(a.{n_col} - b.{n_col})), 0) AS max_diff
                FROM corner_scan_2f a
                JOIN corner_scan_2f b
                  ON  b.mode             = a.mode
                  AND b.primary_metric   = a.secondary_metric
                  AND b.secondary_metric = a.primary_metric
                  AND b.outcome          = a.outcome
                  AND b.corner_direction = CASE a.corner_direction
                        WHEN 'low-high' THEN 'high-low'
                        WHEN 'high-low' THEN 'low-high'
                        ELSE a.corner_direction END
                WHERE a.mode = $1
                  AND a.{n_col} IS DISTINCT FROM b.{n_col}""",
            mode,
        )
        mark = "✓" if asym["n_bad"] == 0 else "✗ FAIL"
        print(f"  {mark}  Orientation-asymmetric corners [{mode}]: "
              f"{asym['n_bad']} (must be 0; max row diff {asym['max_diff']})")

        # TT-only checks.
        if is_tt:
            # (i) One-window corners are ALLOWED and merely reported.
            #
            # This used to fail the batch.  It shouldn't: with edges frozen at
            # the cutoff and a shorter test period, a thin corner can genuinely
            # have zero qualifying rows in one window, and a corner that is
            # structurally impossible in the test era is correct behaviour, not
            # a defect.  (The confirmed example: near the 52-week low AND the
            # 52-week high at once needs a compressed 52-week range — common
            # 2019-2023, absent in a 2024-2026 melt-up.)  Such a corner should
            # render as train stats plus a NULL test, so it is counted here in
            # CORNER units — a corner emits 13 rows, one per outcome, and
            # emptiness is bin-driven, so it empties for all of them at once.
            ow = await conn.fetchrow(
                """SELECT COUNT(*) AS n_rows,
                          COUNT(DISTINCT (primary_metric, secondary_metric,
                                          corner_direction)) AS n_corners
                   FROM corner_scan_2f
                   WHERE mode = 'train_test'
                     AND (d_train_n IS NULL) <> (d_test_n IS NULL)""",
            )
            print(f"  ·  TT one-window corners: {int(ow['n_corners']):,} "
                  f"({int(ow['n_rows']):,} rows) — allowed; inspect with "
                  f"scripts/corner_scan_tt_onewindow.py")

            # (ii) FAILS: a window that is internally inconsistent.  Within a
            # window the average and the count must both be present or both be
            # NULL.  One without the other means the aggregation lost track of
            # a window, which is the real defect the old check was groping at.
            incons = await conn.fetchval(
                """SELECT COUNT(*) FROM corner_scan_2f
                   WHERE mode = 'train_test'
                     AND ( (d_train_avg_ret IS NULL) <> (d_train_n IS NULL)
                        OR (d_test_avg_ret  IS NULL) <> (d_test_n  IS NULL)
                        OR (q_train_avg_ret IS NULL) <> (q_train_n IS NULL)
                        OR (q_test_avg_ret  IS NULL) <> (q_test_n  IS NULL) )""",
            )
            mark = "✓" if incons == 0 else "✗ FAIL"
            print(f"  {mark}  TT rows with avg/n disagreeing within a window: "
                  f"{incons} (must be 0)")

            # (iii) FAILS: an "empty" side that SQL says is not empty.  Bounded
            # to the corners with the largest populated-side n — the ones where
            # a lost-rows bug would be most visible — so this stays cheap.
            recount_bad = await _recount_empty_sides(
                conn, cutoff_date, limit=_RECOUNT_SAMPLE
            )
            mark = "✓" if not recount_bad else "✗ FAIL"
            print(f"  {mark}  TT empty sides contradicted by SQL "
                  f"(top {_RECOUNT_SAMPLE} by n): {len(recount_bad)} (must be 0)")
            for desc in recount_bad[:5]:
                print(f"        {desc}")

            # (iv) FAILS: every TT row must carry the cutoff actually frozen
            # in tt_bins, so a row can never be read against the wrong split.
            stale_cutoff = await conn.fetchval(
                """SELECT COUNT(*) FROM corner_scan_2f
                   WHERE mode = 'train_test' AND cutoff_date IS DISTINCT FROM $1""",
                cutoff_date,
            )
            mark = "✓" if stale_cutoff == 0 else "✗ FAIL"
            print(f"  {mark}  TT rows not stamped with the frozen cutoff "
                  f"{cutoff_iso}: {stale_cutoff} (must be 0)")
        else:
            incons       = 0
            recount_bad  = []
            stale_cutoff = 0

        print(f"\n  corner_scan_2f [{mode}]:  {totals['n_2f']:>10,} rows")
        print(f"  corner_scan_1f [{mode}]:  {totals['n_1f']:>10,} rows")
        print(f"  Distinct primaries:      {totals['n_p']}")
        print(f"  as_of:                   {totals['as_of']}")
        if is_tt:
            print(f"  frozen cutoff:           {cutoff_iso}")
        print(f"  Phase 6 runtime:         {phase6_secs:.0f}s")

        all_ok = (cc_bad == 0 and inelig_bad == 0
                  and incons == 0 and not recount_bad
                  and stale_cutoff == 0
                  and asym["n_bad"] == 0)
        print(
            "\n── "
            + ("All self-checks PASSED ✓" if all_ok
               else "FAILURES detected ✗ — review above")
            + " ──"
        )

        # Skipped-metric report lives HERE, in the final summary — Phase 0 is
        # long gone by the time anyone reads the tail of this log.
        _print_skipped_summary(skipped_metrics, bin_table, is_tt, cutoff_iso)

        print(f"\nTotal elapsed: {time.perf_counter() - t_total:.0f}s")

        # Non-zero on a hard failure OR on any skipped metric, so a wrapper
        # can notice a stale bin column without parsing the log.  A skip is a
        # warning, not a stop: one stale column must not block the batch, and
        # the rows that were computed are correct and already committed.
        if not all_ok:
            sys.exit(1)
        if skipped_metrics:
            sys.exit(1)

    finally:
        await conn.close()


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--mode",
        choices=["walk_forward", "in_sample", "train_test"],
        required=True,
        help=(
            "Bin mode to scan: 'walk_forward' reads wf_bins, "
            "'in_sample' reads is_bins, 'train_test' reads tt_bins and "
            "splits on the cutoff frozen in that table. There is "
            "deliberately no --cutoff flag: the split is whatever the "
            "upstream build froze."
        ),
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Re-run even if this mode's rows were already written today",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute and print row counts; do not write to DB",
    )
    args = p.parse_args()

    asyncio.run(run(mode=args.mode, dry_run=args.dry_run, force=args.force))


if __name__ == "__main__":
    main()
