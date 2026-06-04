#!/usr/bin/env python3
"""Measure filtered wide-read on is_bins for the Group-4 migration decision.

Reports MEASURED timing and EXPLAIN ANALYZE — gates whether
/secondary-corr-bins can ride the same migration as /secondary-detail
and /secondary-correlation, or whether it has to be split out.

Two filter sizes (the whole point is that cost scales with the filtered
set, not the universe):
  Narrow ≈ one primary bin selected  (e.g. bin20 = 1, ~5% of universe)
  Wide   ≈ five primary bins         (bin20 in 1..5, ~25% of universe)

Decision rule (from the migration plan):
  - If narrow AND wide both come back in the ~1-3s range
        → migrate all three Group-4 endpoints as one shot
  - If wide balloons toward 15s (the /global-metric-bins shape)
        → split: ship /secondary-detail + /secondary-correlation,
          handle /secondary-corr-bins separately

Run from the VPS (the only host with DB access):

    cd /spx_analysis_dashboard
    .venv/bin/python scripts/measure_secondary_corr_bins.py

Or via the project's run-python wrapper if one exists.
"""
import asyncio
import os
import statistics
import time

import asyncpg


# Standard test case — same metric/outcome we've been using for tie-outs.
PRIMARY_METRIC = "oi_weighted_all_div_spot_co"
OUTCOME        = "ret_5d_fwd_oc"


async def main() -> None:
    dsn = (os.environ.get("OI_DATABASE_URL")
           or os.environ.get("DATABASE_URL"))
    if not dsn:
        # Fall back to the app's own .env if running from project root.
        try:
            from dotenv import load_dotenv  # type: ignore
            load_dotenv()
            dsn = (os.environ.get("OI_DATABASE_URL")
                   or os.environ.get("DATABASE_URL"))
        except Exception:
            pass
    if not dsn:
        raise SystemExit(
            "Set OI_DATABASE_URL or DATABASE_URL (or run from project root "
            "with a .env that defines one).")

    conn = await asyncpg.connect(dsn)
    try:
        await _run(conn)
    finally:
        await conn.close()


async def _run(conn) -> None:
    # ── 0. Sanity: indexes on is_bins ─────────────────────────────────────
    print("=" * 72)
    print("Indexes on is_bins (relevant for the planner's choice):")
    idx_rows = await conn.fetch(
        "SELECT indexname, indexdef FROM pg_indexes "
        "WHERE schemaname = 'public' AND tablename = 'is_bins'")
    if not idx_rows:
        print("  (none — table has no indexes!)")
    for r in idx_rows:
        print(f"  {r['indexname']}: {r['indexdef']}")
    print()

    # ── 1. Discover the feature column set the way _ensure_sec_cache does ─
    col_rows = await conn.fetch("""
        SELECT column_name FROM information_schema.columns
        WHERE table_name = 'daily_features' AND table_schema = 'public'
          AND data_type IN ('double precision','numeric','real',
                            'integer','bigint','smallint')
          AND column_name NOT IN ('id','ticker','trade_date',
                                  'created_at','updated_at')
        ORDER BY ordinal_position""")
    all_num_cols = [r["column_name"] for r in col_rows]
    outcome_cols = [c for c in all_num_cols if "ret_" in c and "fwd" in c]
    feature_cols = [c for c in all_num_cols
                    if c not in outcome_cols and c != PRIMARY_METRIC
                    and not c.startswith("spot")
                    and not c.endswith("_pc")]

    is_bins_cols_rows = await conn.fetch("""
        SELECT column_name FROM information_schema.columns
        WHERE table_name = 'is_bins' AND table_schema = 'public'
          AND column_name LIKE 'bin20_%'""")
    bin20_cols = {r["column_name"] for r in is_bins_cols_rows}
    eligible = [f for f in feature_cols if f"bin20_{f}" in bin20_cols]
    missing  = [f for f in feature_cols if f"bin20_{f}" not in bin20_cols]
    print("=" * 72)
    print("Column inventory:")
    print(f"  daily_features numeric cols:          {len(all_num_cols)}")
    print(f"    outcome cols (ret_*_fwd_*):         {len(outcome_cols)}")
    print(f"    feature cols (the corr-bins loop):  {len(feature_cols)}")
    print(f"  is_bins bin20_* cols:                 {len(bin20_cols)}")
    print(f"  features eligible (bin20_<feat> in is_bins): {len(eligible)}")
    print(f"  features missing from is_bins:        {len(missing)}")
    if missing:
        print(f"    examples: {missing[:5]}")
    print()

    # ── 2. Build the two primary-filter sets ──────────────────────────────
    print("=" * 72)
    print("Building primary-filter sets (in-sample style, outcome NOT NULL):")
    narrow_rows = await conn.fetch(f"""
        SELECT df.ticker, df.trade_date
        FROM is_bins ib
        JOIN daily_features df USING (ticker, trade_date)
        WHERE ib.bin20_{PRIMARY_METRIC} = 1
          AND df.{OUTCOME} IS NOT NULL""")
    wide_rows = await conn.fetch(f"""
        SELECT df.ticker, df.trade_date
        FROM is_bins ib
        JOIN daily_features df USING (ticker, trade_date)
        WHERE ib.bin20_{PRIMARY_METRIC} BETWEEN 1 AND 5
          AND df.{OUTCOME} IS NOT NULL""")
    narrow_tkrs  = [r["ticker"]     for r in narrow_rows]
    narrow_dates = [r["trade_date"] for r in narrow_rows]
    wide_tkrs    = [r["ticker"]     for r in wide_rows]
    wide_dates   = [r["trade_date"] for r in wide_rows]
    print(f"  narrow  (bin20={PRIMARY_METRIC} = 1):      {len(narrow_tkrs):>7,} rows")
    print(f"  wide    (bin20={PRIMARY_METRIC} in 1..5):  {len(wide_tkrs):>7,} rows")
    print()

    # ── 3. The query we're benchmarking — the filtered wide bin20 read ────
    bin20_select = ",\n               ".join(f"ib.bin20_{f}" for f in eligible)
    sql_wide_read = f"""
        SELECT ib.ticker, ib.trade_date,
               {bin20_select}
        FROM is_bins ib
        JOIN unnest($1::text[], $2::date[])
             AS f(ticker, trade_date)
          ON ib.ticker = f.ticker AND ib.trade_date = f.trade_date
    """

    # ── 4. Timed trials (1 warmup + 3 measured per filter size) ───────────
    print("=" * 72)
    print(f"Wide bin20 read ({len(eligible)} bin20_* cols + ticker,date):")
    print("(1 warmup pass, 3 timed trials)")
    print()
    for label, tkrs, dates in [
            ("narrow (~20K)", narrow_tkrs, narrow_dates),
            ("wide  (~100K)", wide_tkrs,   wide_dates)]:
        # Warmup
        await conn.fetch(sql_wide_read, tkrs, dates)
        # Timed
        times = []
        last_rowcount = 0
        for _ in range(3):
            t0 = time.perf_counter()
            rows = await conn.fetch(sql_wide_read, tkrs, dates)
            times.append(time.perf_counter() - t0)
            last_rowcount = len(rows)
        print(f"  {label}  filter={len(tkrs):>7,}  returned={last_rowcount:>7,}")
        print(f"    trials: {[f'{t:.2f}s' for t in times]}")
        print(f"    median: {statistics.median(times):.2f}s   "
              f"min: {min(times):.2f}s   max: {max(times):.2f}s")
        print()

    # ── 5. EXPLAIN ANALYZE on both sizes (so we see the plan transition) ──
    for label, tkrs, dates in [("NARROW", narrow_tkrs, narrow_dates),
                                ("WIDE",   wide_tkrs,   wide_dates)]:
        print("=" * 72)
        print(f"EXPLAIN (ANALYZE, BUFFERS) — {label} filter "
              f"({len(tkrs):,} rows)")
        print()
        plan = await conn.fetch(
            f"EXPLAIN (ANALYZE, BUFFERS, TIMING) {sql_wide_read}",
            tkrs, dates)
        for r in plan:
            print(r["QUERY PLAN"])
        print()


if __name__ == "__main__":
    asyncio.run(main())
