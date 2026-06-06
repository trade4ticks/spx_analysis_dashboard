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

Both tables are mode-partitioned.  WF rows and IS rows coexist; running one
mode never touches the other mode's rows.

Run monthly on the VPS (project root, venv active):
    python scripts/corner_scan.py --mode walk_forward [--force] [--dry-run]
    python scripts/corner_scan.py --mode in_sample    [--force] [--dry-run]

    --mode     walk_forward reads wf_bins; in_sample reads is_bins  (required)
    --force    re-run even if this mode's rows were already written today
    --dry-run  compute and print row counts; do not write to DB
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
    as_of             DATE        NOT NULL,
    mode              TEXT        NOT NULL DEFAULT 'walk_forward',
    scanned_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (primary_metric, secondary_metric, corner_direction, outcome, mode)
);
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


# ── Main ──────────────────────────────────────────────────────────────────────

async def run(mode: str, dry_run: bool, force: bool) -> None:
    oi_dsn = os.getenv("OI_DATABASE_URL")
    if not oi_dsn:
        print("ERROR: OI_DATABASE_URL not set (check .env or environment).")
        sys.exit(1)

    bin_table = "wf_bins" if mode == "walk_forward" else "is_bins"

    t_total = time.perf_counter()
    conn = await asyncpg.connect(dsn=oi_dsn)

    try:
        await conn.execute(_DDL_2F)
        await conn.execute(_DDL_1F)

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
        print("\nPhase 5: 1-factor scan…")
        t5          = time.perf_counter()
        as_of       = _date.today()
        scanned_now = datetime.now(tz=timezone.utc)
        rows_1f: list[tuple] = []

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

        for p_idx, p_name in enumerate(eligible_metrics):
            # Per-P accumulators:
            # axes: [p_edge (0=low,1=high), s_edge (0=low,1=high), S_idx, outcome]
            # d_ = decile resolution,  q_ = quintile resolution
            d_sums = np.zeros((2, 2, F, _O), dtype=np.float64)
            d_cnts = np.zeros((2, 2, F, _O), dtype=np.int32)
            q_sums = np.zeros((2, 2, F, _O), dtype=np.float64)
            q_cnts = np.zeros((2, 2, F, _O), dtype=np.int32)

            for tkr in tickers_order:
                s, e    = ticker_slices[tkr]
                bd_t    = bins_d_full[s:e, :]   # (N_t, F) stored-bin deciles
                bq_t    = bins_q_full[s:e, :]   # (N_t, F) stored-bin quintiles
                out_t   = out_full[s:e, :]      # (N_t, O)
                vld_t   = vld_full[s:e, :]      # (N_t, O) bool

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
                    if n_pe < 2:
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
                    sums_ref[p_ei, 0] += S_low_f.T  @ out_cln
                    cnts_ref[p_ei, 0] += (S_low_f.T  @ vld_f).astype(np.int32)
                    sums_ref[p_ei, 1] += S_high_f.T @ out_cln
                    cnts_ref[p_ei, 1] += (S_high_f.T @ vld_f).astype(np.int32)

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

                            d_n   = int(d_cnts[p_ei, s_ej, s_idx, o_idx])
                            q_n   = int(q_cnts[p_ei, s_ej, s_idx, o_idx])
                            d_avg = _avg_or_none(
                                float(d_sums[p_ei, s_ej, s_idx, o_idx]), d_n
                            )
                            q_avg = _avg_or_none(
                                float(q_sums[p_ei, s_ej, s_idx, o_idx]), q_n
                            )
                            days  = _HOLDING_DAYS[o_name]
                            rows_2f.append((
                                p_name, s_name, direction, o_name,
                                d_avg,  _rpd(d_avg, days), d_n or None,
                                q_avg,  _rpd(q_avg, days), q_n or None,
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
        if dry_run:
            print(f"\n── DRY RUN — no DB changes ──────────────────────────────")
            print(f"  corner_scan_1f would write: {len(rows_1f):,} rows  (mode={mode})")
            print(f"  corner_scan_2f would write: {len(rows_2f):,} rows  (mode={mode})")
            print(f"  Total elapsed: {time.perf_counter() - t_total:.0f}s")
            return

        print(f"\nPhase 7: Writing to DB (mode={mode})…")
        t7 = time.perf_counter()

        async with conn.transaction():
            # Mode-scoped DELETE: remove only rows for the current mode.
            # Other modes' rows (e.g. IS rows when running WF) are preserved.
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
                    as_of, mode, scanned_at)
                   VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13)""",
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
        print(f"\n  corner_scan_2f [{mode}]:  {totals['n_2f']:>10,} rows")
        print(f"  corner_scan_1f [{mode}]:  {totals['n_1f']:>10,} rows")
        print(f"  Distinct primaries:      {totals['n_p']}")
        print(f"  as_of:                   {totals['as_of']}")

        all_ok = cc_bad == 0 and inelig_bad == 0
        print(
            "\n── "
            + ("All self-checks PASSED ✓" if all_ok
               else "FAILURES detected ✗ — review above")
            + " ──"
        )
        if not all_ok:
            sys.exit(1)

        print(f"\nTotal elapsed: {time.perf_counter() - t_total:.0f}s")

    finally:
        await conn.close()


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--mode",
        choices=["walk_forward", "in_sample"],
        required=True,
        help=(
            "Bin mode to scan: 'walk_forward' reads from wf_bins, "
            "'in_sample' reads from is_bins"
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
