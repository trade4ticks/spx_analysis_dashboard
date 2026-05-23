#!/usr/bin/env python3
"""Offline pre-computation of ALL-mode IC batch — avoids web-worker OOM.

The /ic-batch?ticker=ALL endpoint needs cross-sectional Spearman IC across
~123 metrics × ~141K rows. Fetching all 125 columns at once in a uvicorn
multiprocessing worker balloons to 4–5 GB and triggers the kernel OOM killer
(confirmed via dmesg on the VPS: 6 kills, python at 3.6–5 GB).

This script computes the same result **one metric at a time**, keeping peak
RAM at ~4 MB per metric (4 cols × 141K rows) instead of ~800 MB total.
It writes the result to ic_batch_cache using the same cache key the endpoint
reads, so subsequent GET /ic-batch?ticker=ALL requests are instant cache hits.

Run once on the VPS (from the project root, with the venv active):

    python scripts/precompute_ic_all.py

Options:
    --outcome     Outcome column name (default: ret_5d_fwd_oc)
    --window      Rolling-IC window in days (default: 252)
    --stride      Date stride for cross-sectional loop (default: 3)
    --cutoff-date ISO date string for train/test split (default: none)
    --force       Overwrite an existing cache entry (default: skip if present)

The cache key format matches the endpoint exactly:
    ic_batch:{ticker}:{outcome}:{window}:{mode_tag}:s{stride}

where ticker="ALL" and mode_tag="default" or "tt:{cutoff_date}".
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import re
import sys
from datetime import date as _date
from pathlib import Path

import numpy as np

# Allow running as a top-level script without installation.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(_ROOT / ".env")

import asyncpg  # noqa: E402

from app.routers.ic_compute import (  # noqa: E402
    finite_or_none,
    noise_floor_epsilon,
    rolling_ic_cross_sectional,
    sign_stability_from_rolling,
)

# ── DDL (same as oi_analysis.py) ─────────────────────────────────────────────

_IC_BATCH_TABLE_DDL = """
CREATE TABLE IF NOT EXISTS ic_batch_cache (
    cache_key   TEXT PRIMARY KEY,
    ticker      TEXT NOT NULL,
    outcome     TEXT NOT NULL,
    window_size INT  NOT NULL,
    cutoff_date DATE,
    payload     JSONB NOT NULL,
    cached_at   TIMESTAMPTZ DEFAULT NOW()
);
"""


# ── Helpers ───────────────────────────────────────────────────────────────────

def _parse_horizon(col_name: str) -> int:
    m = re.search(r"(\d+)d", col_name)
    return int(m.group(1)) if m else 1


async def _fetch_feature_cols(conn) -> list:
    """Mirror _fetch_ic_feature_columns in oi_analysis.py."""
    rows = await conn.fetch(
        """SELECT column_name FROM information_schema.columns
           WHERE table_name = 'daily_features' AND table_schema = 'public'
           AND data_type IN ('double precision','numeric','real',
                             'integer','bigint','smallint')
           AND column_name NOT IN ('id','ticker','trade_date',
                                   'created_at','updated_at')
           ORDER BY ordinal_position"""
    )
    all_cols = [r["column_name"] for r in rows]
    outcomes = {c for c in all_cols if "ret_" in c and "fwd" in c}
    return [c for c in all_cols if c not in outcomes and not c.endswith("_pc")]


# ── Main ──────────────────────────────────────────────────────────────────────

async def run(
    outcome: str,
    window: int,
    stride: int,
    cutoff_date: str | None,
    force: bool,
) -> None:
    oi_dsn = os.getenv("OI_DATABASE_URL")
    if not oi_dsn:
        print("ERROR: OI_DATABASE_URL not set (check .env or environment).")
        sys.exit(1)

    print(f"Connecting to OI database...")
    conn = await asyncpg.connect(dsn=oi_dsn)

    try:
        await conn.execute(_IC_BATCH_TABLE_DDL)

        # Build cache key — must match the endpoint exactly.
        mode_tag  = f"tt:{cutoff_date}" if cutoff_date else "default"
        cache_key = f"ic_batch:ALL:{outcome}:{window}:{mode_tag}:s{stride}"

        # Check for existing entry.
        if not force:
            existing = await conn.fetchval(
                "SELECT cached_at FROM ic_batch_cache WHERE cache_key = $1",
                cache_key,
            )
            if existing is not None:
                print(f"Cache entry already exists (cached_at={existing}).")
                print(f"  key: {cache_key}")
                print(f"Use --force to recompute and overwrite.")
                return

        feature_cols = await _fetch_feature_cols(conn)
        metrics      = [c for c in feature_cols if c != outcome]
        horizon      = _parse_horizon(outcome)

        print(f"Computing ALL-mode IC batch:")
        print(f"  outcome={outcome}, window={window}, stride={stride}, "
              f"cutoff_date={cutoff_date or 'none'}")
        print(f"  metrics to compute: {len(metrics)}")
        print(f"  cache key: {cache_key}")
        print()

        nonfinite_metrics = []  # track metrics that produced inf/nan epsilon
        results = []
        for i, metric in enumerate(metrics, 1):
            print(f"  [{i:>3}/{len(metrics)}] {metric:<40}", end="", flush=True)

            # Fetch only this metric + outcome — 4 cols, ~4 MB peak, no OOM.
            db_rows = await conn.fetch(
                f'SELECT ticker, trade_date, "{metric}", "{outcome}" '
                f'FROM daily_features '
                f'WHERE "{outcome}" IS NOT NULL '
                f'ORDER BY trade_date, ticker'
            )
            rows = [dict(r) for r in db_rows]

            # Cross-sectional IC: one Spearman per day, trailing-mean over window.
            series = rolling_ic_cross_sectional(
                rows, metric, outcome,
                window=window,
                stride=stride,
            )

            del rows  # release memory before next iteration

            if series:
                if cutoff_date:
                    pre = [p.ic for p in series if str(p.date) < cutoff_date]
                    reference_ic = float(np.mean(pre)) if pre else 0.0
                else:
                    reference_ic = float(np.mean([p.ic for p in series]))
                median_k = int(np.median([p.n for p in series]))
            else:
                reference_ic = 0.0
                median_k     = 0

            epsilon   = noise_floor_epsilon(
                "cross_sectional", window=window,
                horizon=horizon, k_tickers=median_k,
            )
            stability = sign_stability_from_rolling(series, reference_ic, epsilon)
            n_total   = stability.n_total

            epsilon_nonfinite = not math.isfinite(epsilon)
            if epsilon_nonfinite:
                nonfinite_metrics.append(metric)

            results.append({
                "name":            metric,
                "long_run_ic":     finite_or_none(reference_ic),
                "long_run_ic_abs": finite_or_none(abs(reference_ic)),
                "epsilon":         finite_or_none(epsilon),
                "n_windows":       n_total,
                "sign_stability":  (
                    finite_or_none(stability.stability, 4)
                    if stability.stability is not None else None
                ),
                "n_same":          stability.n_same,
                "n_opposite":      stability.n_opposite,
                "n_neutral":       stability.n_neutral,
                "neutral_pct":     (
                    round(100.0 * stability.n_neutral / n_total, 2)
                    if n_total else 0.0
                ),
                "suppressed":      stability.suppressed,
                "suppression_reason": stability.suppression_reason,
            })

            tag = f"IC={reference_ic:+.4f}"
            if stability.stability is not None:
                tag += f"  stab={stability.stability:.0%}"
            else:
                tag += f"  stab=— ({stability.suppression_reason})"
            if epsilon_nonfinite:
                tag += "  [ε=∞→null]"
            print(tag)

        # Report any metrics that produced non-finite epsilon.
        if nonfinite_metrics:
            print(f"\n  WARNING: {len(nonfinite_metrics)} metric(s) produced ε=∞ "
                  f"(k_tickers < 2 in every rolling window) — stored as null:")
            for m in nonfinite_metrics:
                print(f"    {m}")
        else:
            print(f"\n  All epsilons finite.")

        # Write to cache.
        print(f"\nWriting {len(results)} metrics to ic_batch_cache...")
        payload_json = json.dumps({"metrics": results})
        cutoff_obj   = _date.fromisoformat(cutoff_date) if cutoff_date else None

        await conn.execute(
            """INSERT INTO ic_batch_cache
               (cache_key, ticker, outcome, window_size, cutoff_date, payload, cached_at)
               VALUES ($1, $2, $3, $4, $5, $6::jsonb, NOW())
               ON CONFLICT (cache_key) DO UPDATE
               SET payload    = EXCLUDED.payload,
                   cached_at  = NOW()""",
            cache_key, "ALL", outcome, window, cutoff_obj, payload_json,
        )
        print(f"Done. Cache key: {cache_key}")

    finally:
        await conn.close()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--outcome",     default="ret_5d_fwd_oc",
                   help="Outcome column (default: ret_5d_fwd_oc)")
    p.add_argument("--window",      type=int, default=252,
                   help="Rolling-IC window in trading days (default: 252)")
    p.add_argument("--stride",      type=int, default=3,
                   help="Date stride for cross-sectional loop (default: 3)")
    p.add_argument("--cutoff-date", default=None,
                   help="ISO date for train/test split (default: none)")
    p.add_argument("--force",       action="store_true",
                   help="Overwrite existing cache entry")
    args = p.parse_args()

    asyncio.run(run(
        outcome=args.outcome,
        window=args.window,
        stride=args.stride,
        cutoff_date=args.cutoff_date,
        force=args.force,
    ))


if __name__ == "__main__":
    main()
