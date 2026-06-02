#!/usr/bin/env python3
"""Measure Analyze-load substep timings end-to-end. Pure measurement.

Prints a stage-by-stage breakdown to stdout. NO log files, NO loggers —
everything goes directly to the terminal where you run it.

Covers two paths:

  - BUNDLE (12-outcome) — calls _compute_analyze_bundle_sync directly
    with _measure=True so each substep prints. Includes the rolling-IC
    split (metric-rank vs outcome-rank vs correlation) on the FIRST
    outcome to keep noise low; subsequent outcomes use the production
    code path so their totals are honest.

  - SINGLE-OUTCOME — re-uses the same fetched rows, but builds a bundle
    of only ONE outcome with _measure=True. This isolates the per-outcome
    cost without the overhead of the legacy /analyze endpoint (which adds
    yearly_consistency, equity_by_decile, half_sample, and other extras
    not in the bundle path). If you want the live /analyze 15s number too,
    hit it directly via curl after this script runs.

Run on the VPS from the project root with the venv active:

    cd /spx_analysis_dashboard
    .venv/bin/python scripts/measure_analyze.py \\
        --ticker ALL --metric concavity_30d --mode in_sample

By default it runs both the 12-outcome bundle AND a single-outcome bundle
for comparison. Flags:

  --bundle-only          skip the single-outcome run
  --single-only          skip the 12-outcome run (still does the SQL fetch)
  --outcome <name>       which outcome to use for --single (default: ret_5d_fwd_oc)
  --cutoff-date <iso>    required when --mode train_test
  --json-write           run json.dumps on the bundle dict so you see
                         serialization cost too (default: skipped)

The script connects to the same DB the app uses (reads DATABASE_URL and
OI_DATABASE_URL from .env via python-dotenv). It does NOT write to the
analyze_cache table or any cache; it just runs the compute.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

# Allow running as a top-level script without installation.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from dotenv import load_dotenv  # noqa: E402
load_dotenv()

import asyncpg  # noqa: E402

from app.routers.oi_analysis import (  # noqa: E402
    _compute_analyze_bundle_sync,
    _discover_analyze_bundle_outcomes,
    _fetch_analyze_bundle_rows,
)


def _hr(title: str) -> None:
    """Section divider — visible in plain terminal."""
    bar = "=" * 78
    print(f"\n{bar}\n  {title}\n{bar}", flush=True)


def _sub(title: str) -> None:
    print(f"\n--- {title}", flush=True)


async def _connect() -> asyncpg.Pool:
    """Create the same OI pool the app uses. Errors loudly if missing."""
    dsn = os.getenv("OI_DATABASE_URL")
    if not dsn:
        print("ERROR: OI_DATABASE_URL not set in environment / .env", file=sys.stderr)
        sys.exit(2)
    return await asyncpg.create_pool(
        dsn=dsn, min_size=1, max_size=2, command_timeout=300,
    )


async def main(args: argparse.Namespace) -> None:
    _hr(f"measure_analyze — ticker={args.ticker} metric={args.metric} mode={args.mode}")
    print(f"  python={sys.version.split()[0]}  cwd={Path.cwd()}", flush=True)
    if args.mode == "train_test" and not args.cutoff_date:
        print("ERROR: --mode train_test requires --cutoff-date", file=sys.stderr)
        sys.exit(2)

    # ── Connect ──────────────────────────────────────────────────────────
    t = time.perf_counter()
    pool = await _connect()
    print(f"  pool_connect={time.perf_counter() - t:.3f}s", flush=True)

    try:
        # ── Outcome discovery ────────────────────────────────────────────
        _sub("outcome discovery")
        t = time.perf_counter()
        outcomes = await _discover_analyze_bundle_outcomes(pool)
        print(f"  discover_outcomes={time.perf_counter() - t:.3f}s  "
              f"(found {len(outcomes)}: {outcomes})", flush=True)
        if not outcomes:
            print("ERROR: no forward-return outcome columns discovered", file=sys.stderr)
            sys.exit(2)
        if args.outcome not in outcomes:
            print(f"WARNING: --outcome {args.outcome!r} not in discovered list; "
                  f"single-outcome run will use {outcomes[0]!r}", file=sys.stderr)
            single_outcome = outcomes[0]
        else:
            single_outcome = args.outcome

        # ── Fetch (one query, all outcomes) ──────────────────────────────
        _sub("SQL fetch (one query, all 12 outcome columns)")
        t = time.perf_counter()
        rows = await _fetch_analyze_bundle_rows(
            pool, args.ticker, args.metric, outcomes,
        )
        fetch_secs = time.perf_counter() - t
        print(f"  fetch={fetch_secs:.3f}s  rows={len(rows)}", flush=True)
        if not rows:
            print(f"ERROR: no rows returned for ticker={args.ticker} "
                  f"metric={args.metric!r}. Check the metric name.",
                  file=sys.stderr)
            sys.exit(2)

        # ── BUNDLE: 12 outcomes ──────────────────────────────────────────
        bundle_total = None
        if not args.single_only:
            _hr(f"BUNDLE COMPUTE — {len(outcomes)} outcomes")
            t = time.perf_counter()
            bundle = _compute_analyze_bundle_sync(
                rows, args.metric, args.ticker, args.mode,
                args.cutoff_date if args.mode == "train_test" else None,
                outcomes, n_bins=20, _measure=True,
            )
            bundle_total = time.perf_counter() - t
            print(f"\n[TOTAL] _compute_analyze_bundle_sync (12 outcomes) "
                  f"= {bundle_total:.3f}s", flush=True)
            print(f"[TOTAL] fetch + bundle = {fetch_secs + bundle_total:.3f}s",
                  flush=True)

            if args.json_write:
                _sub("json.dumps the bundle (serialization cost)")
                t = time.perf_counter()
                payload = json.dumps(bundle)
                print(f"  json_dumps={time.perf_counter() - t:.3f}s  "
                      f"payload_bytes={len(payload):,}", flush=True)

        # ── SINGLE-OUTCOME: 1 outcome via the same bundle path ───────────
        if not args.bundle_only:
            _hr(f"SINGLE-OUTCOME COMPUTE — outcome={single_outcome}")
            t = time.perf_counter()
            bundle1 = _compute_analyze_bundle_sync(
                rows, args.metric, args.ticker, args.mode,
                args.cutoff_date if args.mode == "train_test" else None,
                [single_outcome], n_bins=20, _measure=True,
            )
            single_total = time.perf_counter() - t
            print(f"\n[TOTAL] _compute_analyze_bundle_sync (1 outcome) "
                  f"= {single_total:.3f}s", flush=True)
            print(f"[TOTAL] fetch + single = {fetch_secs + single_total:.3f}s",
                  flush=True)
            if bundle_total is not None:
                _extras_12 = bundle_total - single_total
                print(f"[DELTA] 12-outcome minus 1-outcome = "
                      f"{_extras_12:.3f}s ⇒ "
                      f"~{_extras_12/11:.3f}s per additional outcome",
                      flush=True)

        # ── Note about the live /analyze endpoint ────────────────────────
        _hr("Live /analyze endpoint (for the 15s number you observed)")
        print("  The /analyze endpoint has additional work not in the bundle:", flush=True)
        print("    - decile_stats with per-trade `returns` array", flush=True)
        print("    - equity_by_decile (10 deciles × 2 modes, precomputed)", flush=True)
        print("    - yearly_consistency (per-year re-binning)", flush=True)
        print("    - half_sample + composite scoring", flush=True)
        print("    - monthly_stats, dow_stats, activity_by_date", flush=True)
        print("  The existing _tlog in /analyze writes to /tmp/analyze_timing.log.", flush=True)
        print("  To see its wall-clock without using log files, hit it directly:", flush=True)
        print(f"    curl -s -o /dev/null -w '%{{time_total}}s\\n' \\", flush=True)
        print(f"      'http://localhost:8000/api/factor-analysis/analyze"
              f"?ticker={args.ticker}&metric={args.metric}"
              f"&outcome={single_outcome}&force=true'", flush=True)
        print("  The `force=true` bypasses both in-memory and DB cache.", flush=True)

    finally:
        await pool.close()
        print("\n[done] pool closed", flush=True)


def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ticker",  default="ALL",
                   help="Ticker filter; ALL for cross-sectional (default).")
    p.add_argument("--metric",  required=True,
                   help="Metric column in daily_features (e.g. concavity_30d).")
    p.add_argument("--mode",    default="in_sample",
                   choices=["in_sample", "walk_forward", "train_test"],
                   help="Binning mode (default: in_sample).")
    p.add_argument("--cutoff-date", default=None,
                   help="ISO date for --mode train_test.")
    p.add_argument("--outcome", default="ret_5d_fwd_oc",
                   help="Outcome for the single-outcome run (default: ret_5d_fwd_oc).")
    p.add_argument("--bundle-only", action="store_true",
                   help="Skip the single-outcome run.")
    p.add_argument("--single-only", action="store_true",
                   help="Skip the 12-outcome bundle (still fetches once).")
    p.add_argument("--json-write", action="store_true",
                   help="Also time json.dumps on the bundle dict.")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse()
    asyncio.run(main(args))
