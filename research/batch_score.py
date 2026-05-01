"""
Exhaustive batch scoring: every ticker × metric × forward return.

Calls scanner.scan_relationship() on each combo, stores results
in the oi_score_matrix table. Run via API endpoint or CLI:

    python -m research.batch_score
"""
import asyncio
import math
import json
from datetime import datetime

from research import scanner

_EXCLUDE_COLS = {"id", "ticker", "trade_date", "created_at", "updated_at"}
_EXCLUDE_FEATURES = {"spot_co", "spot_close", "spot_co_pc", "spot_close_pc"}

_TABLE_DDL = """\
CREATE TABLE IF NOT EXISTS oi_score_matrix (
    id              SERIAL PRIMARY KEY,
    ticker          TEXT NOT NULL,
    metric          TEXT NOT NULL,
    fwd_ret         TEXT NOT NULL,
    composite_score REAL,
    pattern         TEXT,
    spearman_r      REAL,
    monotonicity    REAL,
    yearly_pct      REAL,
    concentration   REAL,
    tail_spread     REAL,
    n_obs           INTEGER,
    d10_avg         REAL,
    d1_avg          REAL,
    d10_wr          REAL,
    d1_wr           REAL,
    best_sharpe     REAL,
    scanned_at      TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (ticker, metric, fwd_ret)
);
CREATE INDEX IF NOT EXISTS idx_oi_score_ticker ON oi_score_matrix(ticker);
CREATE INDEX IF NOT EXISTS idx_oi_score_metric ON oi_score_matrix(metric);
CREATE INDEX IF NOT EXISTS idx_oi_score_score ON oi_score_matrix(composite_score DESC);
"""

_UPSERT = """\
INSERT INTO oi_score_matrix
    (ticker, metric, fwd_ret, composite_score, pattern, spearman_r,
     monotonicity, yearly_pct, concentration, tail_spread, n_obs,
     d10_avg, d1_avg, d10_wr, d1_wr, best_sharpe, scanned_at)
VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15,$16,NOW())
ON CONFLICT (ticker, metric, fwd_ret) DO UPDATE SET
    composite_score=EXCLUDED.composite_score, pattern=EXCLUDED.pattern,
    spearman_r=EXCLUDED.spearman_r, monotonicity=EXCLUDED.monotonicity,
    yearly_pct=EXCLUDED.yearly_pct, concentration=EXCLUDED.concentration,
    tail_spread=EXCLUDED.tail_spread, n_obs=EXCLUDED.n_obs,
    d10_avg=EXCLUDED.d10_avg, d1_avg=EXCLUDED.d1_avg,
    d10_wr=EXCLUDED.d10_wr, d1_wr=EXCLUDED.d1_wr,
    best_sharpe=EXCLUDED.best_sharpe, scanned_at=NOW()
"""

# Shared progress state for API polling
_progress = {"running": False, "message": "", "last_run": None}


def get_progress() -> dict:
    return dict(_progress)


async def ensure_table(pool):
    async with pool.acquire() as conn:
        await conn.execute(_TABLE_DDL)


async def run_batch_score(oi_pool, main_pool, log=print):
    """Run the full batch scan. oi_pool=open_interest DB, main_pool=spx_interpolated DB."""
    _progress["running"] = True
    _progress["message"] = "Starting..."

    try:
        # Ensure table exists (in main DB where research tables live)
        await ensure_table(main_pool)

        # 1. Get tickers
        async with oi_pool.acquire() as conn:
            ticker_rows = await conn.fetch(
                "SELECT DISTINCT ticker FROM daily_features ORDER BY ticker")
        tickers = [r["ticker"] for r in ticker_rows]
        log(f"Found {len(tickers)} tickers")

        # 2. Get columns
        async with oi_pool.acquire() as conn:
            col_rows = await conn.fetch("""
                SELECT column_name FROM information_schema.columns
                WHERE table_name = 'daily_features' AND table_schema = 'public'
                  AND data_type IN ('double precision','numeric','real','integer','bigint','smallint')
                  AND column_name NOT IN ('id','ticker','trade_date','created_at','updated_at')
            """)
        all_cols = [r["column_name"] for r in col_rows]

        features = [c for c in all_cols
                     if not (c.startswith("ret_") and "fwd" in c)
                     and c not in _EXCLUDE_FEATURES]
        outcomes = [c for c in all_cols if c.startswith("ret_") and "fwd" in c]

        total_combos = len(tickers) * len(features) * len(outcomes)
        log(f"Scanning: {len(tickers)} tickers × {len(features)} features × {len(outcomes)} outcomes = {total_combos} combos")

        total_scored = 0
        total_skipped = 0

        for ti, ticker in enumerate(tickers):
            _progress["message"] = f"Ticker {ti+1}/{len(tickers)}: {ticker}"
            log(f"[{ti+1}/{len(tickers)}] {ticker}...")

            # Fetch all data for this ticker once
            async with oi_pool.acquire() as conn:
                rows = await conn.fetch(
                    "SELECT * FROM daily_features WHERE ticker = $1 ORDER BY trade_date",
                    ticker)
            rows = [dict(r) for r in rows]

            if len(rows) < 30:
                log(f"  Skipping {ticker} — only {len(rows)} rows")
                total_skipped += len(features) * len(outcomes)
                continue

            avail = set(rows[0].keys())
            batch_params = []

            for feature in features:
                if feature not in avail:
                    continue
                for outcome in outcomes:
                    if outcome not in avail or feature == outcome:
                        continue
                    try:
                        result = scanner.scan_relationship(rows, feature, outcome, ticker)
                    except Exception:
                        total_skipped += 1
                        continue

                    if "error" in result or result.get("n", 0) < 30:
                        total_skipped += 1
                        continue

                    rob = result.get("robustness") or {}
                    bs = result.get("bucket_stats") or []
                    valid_bs = [b for b in bs if b is not None]
                    d10 = next((b for b in valid_bs if b.get("bucket") == 10), {})
                    d1 = next((b for b in valid_bs if b.get("bucket") == 1), {})
                    best = result.get("best_single_bucket") or {}

                    batch_params.append((
                        ticker, feature, outcome,
                        result.get("composite_score"),
                        result.get("pattern"),
                        result.get("spearman_r"),
                        result.get("monotonicity"),
                        rob.get("yearly_consistency_pct"),
                        rob.get("concentration_risk"),
                        result.get("tail_spread"),
                        result.get("n"),
                        d10.get("avg_ret"),
                        d1.get("avg_ret"),
                        d10.get("win_rate"),
                        d1.get("win_rate"),
                        best.get("sharpe"),
                    ))
                    total_scored += 1

            # Batch upsert for this ticker
            if batch_params:
                async with main_pool.acquire() as conn:
                    await conn.executemany(_UPSERT, batch_params)
                log(f"  {len(batch_params)} scores saved")

        _progress["message"] = f"Complete: {total_scored} scored, {total_skipped} skipped"
        _progress["last_run"] = datetime.utcnow().isoformat()
        log(f"Done. {total_scored} scored, {total_skipped} skipped.")

    except Exception as exc:
        _progress["message"] = f"Error: {exc}"
        log(f"BATCH ERROR: {exc}")
        raise
    finally:
        _progress["running"] = False


# CLI entry point
if __name__ == "__main__":
    import asyncpg
    import os
    from dotenv import load_dotenv
    load_dotenv()

    async def main():
        oi_url = os.environ.get("OI_DATABASE_URL")
        main_url = os.environ.get("DATABASE_URL")
        if not oi_url or not main_url:
            print("Set OI_DATABASE_URL and DATABASE_URL env vars")
            return
        oi_pool = await asyncpg.create_pool(oi_url, min_size=2, max_size=5)
        main_pool = await asyncpg.create_pool(main_url, min_size=2, max_size=5)
        try:
            await run_batch_score(oi_pool, main_pool)
        finally:
            await oi_pool.close()
            await main_pool.close()

    asyncio.run(main())
