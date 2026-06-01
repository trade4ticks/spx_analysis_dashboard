#!/usr/bin/env python3
"""Offline corner-scan job — sweeps all eligible metric pairs for hot corners.

A 'corner' is a metric-pair (P × S) where both metrics are simultaneously in
an extreme bin (D1/D10 decile or Q1/Q5 quintile) on the same trade-date.
Both P and S bins are the full-universe walk-forward decile/quintile computed
in Phase 2+3 — each trade's bin reflects only the history available at that
date for that ticker, with no lookahead and no cross-sectional contamination.
The corner is the intersection: P in its full-universe WF extreme AND S in its
OWN full-universe WF extreme.  Consistent with the heatmap Y-axis assignment.

Output tables (OI DB):
  corner_scan_2f  — metric-pair corners  (PK: P, S, direction, outcome)
  corner_scan_1f  — single-metric extremes (PK: metric, extreme, outcome)

Run monthly on the VPS (project root, venv active):
    python scripts/corner_scan.py [--force] [--dry-run]

    --force    re-run even if corner_scan_2f was already populated today
    --dry-run  compute and print row counts; do not write to DB
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time
from datetime import date as _date
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

# Minimum valid prior values before a WF bin is trusted
_WARMUP = 30

# ── DDL ───────────────────────────────────────────────────────────────────────

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
    as_of             DATE NOT NULL,
    PRIMARY KEY (primary_metric, secondary_metric, corner_direction, outcome)
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
    as_of         DATE NOT NULL,
    PRIMARY KEY (metric, extreme, outcome)
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


def _wf_bins(X_t: np.ndarray, warmup: int) -> tuple[np.ndarray, np.ndarray]:
    """Walk-forward decile + quintile bins for one ticker.

    Args:
        X_t:    (N_t, F) float64; NaN for missing values.
        warmup: minimum prior valid observations before a bin is trusted.

    Returns:
        bins_d: (N_t, F) int32, deciles 1..10; 0 = warmup or NaN row.
        bins_q: (N_t, F) int32, quintiles 1..5; 0 = warmup or NaN row.
    """
    N_t, F = X_t.shape

    # wf_rank[i, f] = #{j < i : X[j, f] < X[i, f]}
    # NaN comparisons return False → NaN rows accumulate rank 0 (correct: lowest).
    wf_rank = np.zeros((N_t, F), dtype=np.int32)
    for j in range(N_t - 1):
        wf_rank[j + 1:] += X_t[j + 1:] > X_t[j]

    nan_mask  = np.isnan(X_t)
    valid_cum = np.cumsum(~nan_mask, axis=0, dtype=np.int32)
    safe_n    = np.where(valid_cum > 0, valid_cum, 1).astype(np.float64)

    bins_d = (
        np.minimum(
            (wf_rank.astype(np.float64) / safe_n * 10).astype(np.int32),
            9,
        ) + 1  # 1..10
    )
    bins_d[nan_mask | (valid_cum < warmup)] = 0

    bins_q = (
        np.minimum(
            (wf_rank.astype(np.float64) / safe_n * 5).astype(np.int32),
            4,
        ) + 1  # 1..5
    )
    bins_q[nan_mask | (valid_cum < warmup)] = 0

    return bins_d, bins_q


# ── Main ──────────────────────────────────────────────────────────────────────

async def run(dry_run: bool, force: bool) -> None:
    oi_dsn = os.getenv("OI_DATABASE_URL")
    if not oi_dsn:
        print("ERROR: OI_DATABASE_URL not set (check .env or environment).")
        sys.exit(1)

    t_total = time.perf_counter()
    conn = await asyncpg.connect(dsn=oi_dsn)

    try:
        await conn.execute(_DDL_2F)
        await conn.execute(_DDL_1F)

        # Guard: skip if already run today, unless --force.
        if not force and not dry_run:
            today_str    = str(_date.today())
            existing_aso = await conn.fetchval(
                "SELECT as_of FROM corner_scan_2f LIMIT 1"
            )
            if existing_aso is not None and str(existing_aso) == today_str:
                print(
                    f"corner_scan_2f already populated for {today_str}. "
                    "Use --force to overwrite."
                )
                return

        # ── Phase 0: Setup ───────────────────────────────────────────────────
        print("Phase 0: Loading metric classification…")
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

        # Columns actually present in daily_features.
        actual_cols: set[str] = {
            r["column_name"]
            for r in await conn.fetch(
                """SELECT column_name FROM information_schema.columns
                   WHERE table_name = 'daily_features'
                   AND table_schema = 'public'"""
            )
        }

        missing_outcomes = [o for o in _FWDRET_OUTCOMES if o not in actual_cols]
        if missing_outcomes:
            print(
                f"ERROR: outcome columns missing from daily_features: "
                f"{missing_outcomes}"
            )
            sys.exit(1)

        # Eligible metrics that are also present in the table.
        eligible_metrics: list[str] = sorted(
            m for m in eligible_by_name if m in actual_cols
        )
        F = len(eligible_metrics)
        metric_tier: dict[str, str] = {
            m: eligible_by_name[m] for m in eligible_metrics
        }
        morning_set: frozenset[str] = frozenset(
            m for m, t in metric_tier.items() if t == "MORNING"
        )

        print(f"  Eligible metrics in daily_features: {F}")
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

        # ── Phase 1: Data load (one query) ───────────────────────────────────
        print("\nPhase 1: Loading daily_features (one query)…")
        t1 = time.perf_counter()

        cols_sql = ", ".join(f'"{m}"' for m in eligible_metrics)
        out_sql  = ", ".join(f'"{o}"' for o in _FWDRET_OUTCOMES)
        db_rows  = await conn.fetch(
            f"SELECT ticker, trade_date, spot_co, spot_pc, {cols_sql}, {out_sql} "
            f"FROM daily_features ORDER BY ticker, trade_date"
        )
        N_total = len(db_rows)
        print(f"  {N_total:,} rows in {time.perf_counter() - t1:.1f}s")

        # ── Phase 2+3: Build arrays + WF bins ────────────────────────────────
        print("\nPhase 2+3: Building arrays + walk-forward binning…")
        t23 = time.perf_counter()

        # Preallocate full arrays — all tickers concatenated in ticker,date order.
        X_full      = np.empty((N_total, F),   dtype=np.float64)
        bins_d_full = np.zeros((N_total, F),   dtype=np.int32)
        bins_q_full = np.zeros((N_total, F),   dtype=np.int32)
        out_full    = np.full((N_total, _O),   np.nan, dtype=np.float64)
        vld_full    = np.zeros((N_total, _O),  dtype=bool)

        # Feature matrix — column by column to minimise Python loop iterations.
        for f_idx, feat in enumerate(eligible_metrics):
            vals = [row[feat] for row in db_rows]
            X_full[:, f_idx] = [
                float(v) if v is not None else np.nan for v in vals
            ]

        # Outcome matrix (first 12 cols = fwd returns).
        for o_idx, o_name in enumerate(_FWDRET_OUTCOMES):
            for i, row in enumerate(db_rows):
                v = row[o_name]
                if v is not None:
                    out_full[i, o_idx] = float(v)
                    vld_full[i, o_idx] = True

        # Overnight gap col 12: O_T / C_{T-1} - 1.
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

        # Build ticker slice map (data is ORDER BY ticker, trade_date).
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

        # WF binning per ticker (in-place into full arrays via views).
        for tkr in tickers_order:
            s, e = ticker_slices[tkr]
            X_t = X_full[s:e, :]       # view — no copy
            bd, bq = _wf_bins(X_t, _WARMUP)
            bins_d_full[s:e, :] = bd
            bins_q_full[s:e, :] = bq

        print(f"  {len(tickers_order)} tickers; WF binning done in "
              f"{time.perf_counter() - t23:.1f}s")

        # ── Phase 5: 1-factor scan ───────────────────────────────────────────
        print("\nPhase 5: 1-factor scan…")
        t5      = time.perf_counter()
        as_of   = _date.today()
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
                        as_of,
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
                bd_t    = bins_d_full[s:e, :]   # (N_t, F) full-universe decile bins
                bq_t    = bins_q_full[s:e, :]   # (N_t, F) full-universe quintile bins
                out_t   = out_full[s:e, :]      # (N_t, O)
                vld_t   = vld_full[s:e, :]      # (N_t, O) bool

                # Four P-edge configurations: D-low, D-high, Q-low, Q-high.
                # Each tuple: (P-bin column, P edge value, S-bin matrix,
                #              extreme S value, sums accumulator, cnts accumulator,
                #              p_ei index)
                # S-bin matrix is the full-universe WF bin array for each config:
                #   D configs use bd_t (decile 1..10); Q configs use bq_t (quintile 1..5).
                # Bin 0 means warmup or NaN — excluded naturally since 0 ≠ 1 and 0 ≠ 10/5.
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

                    # Full-universe WF bins for all S metrics at the P-edge rows.
                    # No recomputation needed — bins were already assigned in Phase 2+3.
                    # Bin 0 (warmup/NaN) is never equal to 1 or n_bins_S, so those
                    # rows are excluded from both S-low and S-high masks automatically.
                    S_bins  = S_bin_mat[pe_rows, :]   # (n_pe, F)
                    out_pe  = out_t[pe_rows, :]        # (n_pe, O)
                    vld_pe  = vld_t[pe_rows, :]        # (n_pe, O) bool

                    # NaN outcomes → 0 so matmul sums correctly; vld_pe tracks counts.
                    out_cln = np.where(vld_pe, out_pe, 0.0)  # (n_pe, O)

                    # Cast to float64 for BLAS matmul.
                    S_low_f  = (S_bins == 1       ).astype(np.float64)  # (n_pe, F)
                    S_high_f = (S_bins == n_bins_S).astype(np.float64)  # (n_pe, F)
                    vld_f    = vld_pe.astype(np.float64)                 # (n_pe, O)

                    # Vectorised aggregation over all (S_metric, outcome) simultaneously.
                    # S_low_f.T:  (F, n_pe) @ (n_pe, O) → (F, O)
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
                                as_of,
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
            print(f"  corner_scan_1f would write: {len(rows_1f):,} rows")
            print(f"  corner_scan_2f would write: {len(rows_2f):,} rows")
            print(f"  Total elapsed: {time.perf_counter() - t_total:.0f}s")
            return

        print("\nPhase 7: Writing to DB…")
        t7 = time.perf_counter()

        async with conn.transaction():
            await conn.execute("DELETE FROM corner_scan_1f")
            await conn.executemany(
                """INSERT INTO corner_scan_1f
                   (metric, extreme, outcome,
                    d_avg_ret, d_ret_per_day, d_n,
                    q_avg_ret, q_ret_per_day, q_n, as_of)
                   VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10)""",
                rows_1f,
            )
            print(f"  corner_scan_1f: {len(rows_1f):,} rows written.")

            await conn.execute("DELETE FROM corner_scan_2f")
            await conn.executemany(
                """INSERT INTO corner_scan_2f
                   (primary_metric, secondary_metric, corner_direction, outcome,
                    d_avg_ret, d_ret_per_day, d_n,
                    q_avg_ret, q_ret_per_day, q_n, as_of)
                   VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11)""",
                rows_2f,
            )
            print(f"  corner_scan_2f: {len(rows_2f):,} rows written.")

        print(f"  DB write done in {time.perf_counter() - t7:.1f}s")

        # ── Self-checks ───────────────────────────────────────────────────────
        print("\n── Self-checks ─────────────────────────────────────────────────")

        cc_bad = await conn.fetchval(
            """SELECT COUNT(*) FROM corner_scan_2f
               WHERE (outcome LIKE '%_cc' OR outcome = 'overnight_gap')
                 AND primary_metric IN (
                     SELECT metric FROM metric_classification
                     WHERE tier = 'MORNING')"""
        )
        mark = "✓" if cc_bad == 0 else "✗ FAIL"
        print(f"  {mark}  CC/gap with MORNING primary: {cc_bad} (must be 0)")

        inelig_bad = await conn.fetchval(
            """SELECT COUNT(*) FROM corner_scan_2f
               WHERE primary_metric IN (
                   SELECT metric FROM metric_classification
                   WHERE eligible_as_metric = false)"""
        )
        mark = "✓" if inelig_bad == 0 else "✗ FAIL"
        print(f"  {mark}  Rows with ineligible primary: {inelig_bad} (must be 0)")

        totals = await conn.fetchrow(
            """SELECT
                 (SELECT COUNT(*)                      FROM corner_scan_2f)    AS n_2f,
                 (SELECT COUNT(*)                      FROM corner_scan_1f)    AS n_1f,
                 (SELECT COUNT(DISTINCT primary_metric) FROM corner_scan_2f)   AS n_p,
                 (SELECT as_of                         FROM corner_scan_2f LIMIT 1) AS as_of"""
        )
        print(f"\n  corner_scan_2f:     {totals['n_2f']:>10,} rows")
        print(f"  corner_scan_1f:     {totals['n_1f']:>10,} rows")
        print(f"  Distinct primaries: {totals['n_p']}")
        print(f"  as_of:              {totals['as_of']}")

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
        "--force",
        action="store_true",
        help="Re-run even if corner_scan_2f was already populated today",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute and print row counts; do not write to DB",
    )
    args = p.parse_args()

    asyncio.run(run(dry_run=args.dry_run, force=args.force))


if __name__ == "__main__":
    main()
