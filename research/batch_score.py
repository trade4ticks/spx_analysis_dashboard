"""
Exhaustive batch scoring: every ticker × metric × forward return.

Calls scanner.scan_relationship() on each combo, stores results
in the oi_score_matrix table. Run via API endpoint or CLI:

    python -m research.batch_score
"""
import asyncio
import math
from datetime import datetime

import numpy as np

from research import scanner

_EXCLUDE_COLS = {"id", "ticker", "trade_date", "created_at", "updated_at"}
_EXCLUDE_FEATURES = {"spot_co", "spot_close", "spot_co_pc", "spot_close_pc"}

_TABLE_DDL = """\
CREATE TABLE IF NOT EXISTS oi_score_matrix (
    id              SERIAL PRIMARY KEY,
    ticker          TEXT NOT NULL,
    metric          TEXT NOT NULL,
    fwd_ret         TEXT NOT NULL,
    mode            TEXT NOT NULL DEFAULT 'in_sample',
    cutoff_date     DATE,
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
    mi              REAL,
    pearson_r       REAL,
    loyo_fragile    BOOLEAN,
    scanned_at      TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_oi_score_ticker ON oi_score_matrix(ticker);
CREATE INDEX IF NOT EXISTS idx_oi_score_metric ON oi_score_matrix(metric);
CREATE INDEX IF NOT EXISTS idx_oi_score_score  ON oi_score_matrix(composite_score DESC);
CREATE INDEX IF NOT EXISTS idx_oi_score_mode   ON oi_score_matrix(mode);
"""

# mode is $4, cutoff_date is $5; total 21 positional params (was 20).
# Conflict target is (ticker, metric, fwd_ret, mode, cutoff_date) — uses
# NULLS NOT DISTINCT so in_sample / walk_forward rows (cutoff_date IS NULL)
# behave the same as before. Train-test runs with different cutoffs get
# separate rows; the most-recent run of a given cutoff updates in place.
_UPSERT = """\
INSERT INTO oi_score_matrix
    (ticker, metric, fwd_ret, mode, cutoff_date,
     composite_score, pattern, spearman_r,
     monotonicity, yearly_pct, concentration, tail_spread, n_obs,
     d10_avg, d1_avg, d10_wr, d1_wr, best_sharpe, mi, pearson_r, loyo_fragile, scanned_at)
VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15,$16,$17,$18,$19,$20,$21,NOW())
ON CONFLICT (ticker, metric, fwd_ret, mode, cutoff_date) DO UPDATE SET
    composite_score=EXCLUDED.composite_score, pattern=EXCLUDED.pattern,
    spearman_r=EXCLUDED.spearman_r, monotonicity=EXCLUDED.monotonicity,
    yearly_pct=EXCLUDED.yearly_pct, concentration=EXCLUDED.concentration,
    tail_spread=EXCLUDED.tail_spread, n_obs=EXCLUDED.n_obs,
    d10_avg=EXCLUDED.d10_avg, d1_avg=EXCLUDED.d1_avg,
    d10_wr=EXCLUDED.d10_wr, d1_wr=EXCLUDED.d1_wr,
    best_sharpe=EXCLUDED.best_sharpe,
    mi=EXCLUDED.mi, pearson_r=EXCLUDED.pearson_r, loyo_fragile=EXCLUDED.loyo_fragile,
    scanned_at=NOW()
"""

# Shared progress state for API polling
_progress = {"running": False, "message": "", "last_run": None, "walk_forward": False, "cutoff_date": None}


def get_progress() -> dict:
    return dict(_progress)


async def ensure_table(pool):
    async with pool.acquire() as conn:
        await conn.execute(_TABLE_DDL)
        # Column migrations for rows created before the current DDL.
        # Use try/except instead of IF NOT EXISTS — some PG versions/wrappers silently
        # ignore IF NOT EXISTS on ADD COLUMN and don't actually add the column.
        for col, typ in [('mi', 'REAL'), ('pearson_r', 'REAL'), ('loyo_fragile', 'BOOLEAN'),
                         ('mode', "TEXT NOT NULL DEFAULT 'in_sample'"),
                         ('cutoff_date', 'DATE')]:
            try:
                await conn.execute(
                    f'ALTER TABLE oi_score_matrix ADD COLUMN {col} {typ}')
            except Exception:
                pass  # column already exists
        # Unique index now includes cutoff_date so train-test runs with
        # different cutoffs persist independently. NULLS NOT DISTINCT
        # (PG 15+) treats NULL == NULL — so in_sample / walk_forward rows
        # (cutoff_date IS NULL) keep their single-row-per-(ticker, metric,
        # fwd_ret, mode) behaviour unchanged.
        #
        # The previous mode-only index is dropped — it would over-constrain
        # train-test rows (only one cutoff could ever be stored per combo).
        try:
            await conn.execute("DROP INDEX IF EXISTS oi_score_matrix_uq_mode")
        except Exception:
            pass
        await conn.execute("""
            CREATE UNIQUE INDEX IF NOT EXISTS oi_score_matrix_uq_mode_cutoff
            ON oi_score_matrix (ticker, metric, fwd_ret, mode, cutoff_date)
            NULLS NOT DISTINCT
        """)
        # Drop the old 3-column constraint if it still exists (non-fatal if already gone).
        try:
            await conn.execute(
                "ALTER TABLE oi_score_matrix "
                "DROP CONSTRAINT IF EXISTS oi_score_matrix_ticker_metric_fwd_ret_key")
        except Exception:
            pass


def _safe_float(v):
    if v is None:
        return float("nan")
    try:
        f = float(v)
        return f if not math.isnan(f) else float("nan")
    except (TypeError, ValueError):
        return float("nan")


def _wf_bin_matrix(rows_chrono: list, feature_list: list, n_bins: int = 10, warmup: int = 252):
    """Compute walk-forward bin assignments for all features simultaneously.

    Returns (bin_mat, valid_cum, nan_mask):
      bin_mat   — int32 (N, F), 1-indexed bin for each (row, feature); 0 = warmup/NaN
      valid_cum — int32 (N, F), cumulative non-NaN count up to and including row i
      nan_mask  — bool  (N, F)
    """
    N = len(rows_chrono)
    F = len(feature_list)
    X = np.array(
        [[_safe_float(r.get(feat)) for feat in feature_list] for r in rows_chrono],
        dtype=np.float64,
    )
    nan_mask = np.isnan(X)
    # Set NaN to 0 for comparison (nan > x == False in numpy, but safer to be explicit)
    X_cmp = np.where(nan_mask, np.finfo(np.float64).min, X)

    wf_rank = np.zeros((N, F), dtype=np.int32)
    for j in range(N - 1):
        valid_j = ~nan_mask[j]                # features with a real value at row j
        gt = (X_cmp[j + 1:] > X_cmp[j]) & (~nan_mask[j + 1:]) & valid_j
        wf_rank[j + 1:] += gt.astype(np.int32)

    valid_cum = np.cumsum(~nan_mask, axis=0, dtype=np.int32)
    safe_n = np.where(valid_cum > 0, valid_cum, 1).astype(np.float64)
    raw_bin = (wf_rank.astype(np.float64) / safe_n * n_bins).astype(np.int32)
    bin_mat = np.minimum(raw_bin, n_bins - 1) + 1   # 1-indexed

    # Zero out warmup rows and NaN cells so callers can skip them
    past_warm = valid_cum >= warmup
    bin_mat = np.where(past_warm & ~nan_mask, bin_mat, 0)
    return bin_mat, valid_cum, nan_mask


def _tt_bin_matrix(rows_chrono: list, feature_list: list, cutoff_date_str: str, n_bins: int = 10):
    """Compute train-test bin assignments for all features simultaneously.

    Training rows (trade_date < cutoff) build per-feature sorted thresholds
    via np.searchsorted. Test rows (trade_date >= cutoff) are assigned bins
    against the frozen training distribution.

    Returns (test_rows, bin_mat) where bin_mat is int32 (N_test, F), 1-indexed;
    0 = insufficient training history for that feature or NaN value.
    """
    train_rows = [r for r in rows_chrono if r.get("trade_date", "") < cutoff_date_str]
    test_rows  = [r for r in rows_chrono if r.get("trade_date", "") >= cutoff_date_str]

    F      = len(feature_list)
    N_test = len(test_rows)
    bin_mat = np.zeros((N_test, F), dtype=np.int32)

    for fi, feat in enumerate(feature_list):
        train_vals = []
        for r in train_rows:
            v = _safe_float(r.get(feat))
            if not math.isnan(v):
                train_vals.append(v)
        if len(train_vals) < n_bins:
            continue  # leave column as 0 (insufficient training)
        train_sorted = np.sort(np.array(train_vals, dtype=np.float64))
        n_train = len(train_sorted)
        for ti, r in enumerate(test_rows):
            v = _safe_float(r.get(feat))
            if math.isnan(v):
                continue
            rank = int(np.searchsorted(train_sorted, v, side="right"))
            b = min(int(rank / n_train * n_bins), n_bins - 1) + 1  # 1-indexed
            bin_mat[ti, fi] = b

    return test_rows, bin_mat


def _score_one_ticker(
    ticker: str, rows: list, features: list, outcomes: list,
    mode_label: str, walk_forward: bool, cutoff_date: str,
) -> tuple[list, int, int]:
    """Pure-sync per-ticker scoring. Runs on a worker thread via
    `asyncio.to_thread` so the FastAPI event loop stays responsive
    while this numpy-heavy work executes.

    Returns (batch_params, scored, skipped):
      batch_params — list of _UPSERT param tuples ready for executemany
      scored       — number of (feature, outcome) pairs scored
      skipped      — number skipped (insufficient data, scanner error, etc.)

    No asyncpg access here — the caller fetches `rows` on the event
    loop, passes them in, and writes `batch_params` after this returns.
    """
    # Parse cutoff_date string into a date object once. None for
    # in_sample / walk_forward; a `datetime.date` for train_test.
    # asyncpg expects a date object for the DATE column in _UPSERT.
    from datetime import date as _date_cls
    cutoff_date_obj = _date_cls.fromisoformat(cutoff_date) if cutoff_date else None

    avail = set(rows[0].keys())
    batch_params: list = []
    scored = 0
    skipped = 0

    if cutoff_date:
        # ── Train-test mode ──────────────────────────────────────────
        # Bins frozen from training rows (trade_date < cutoff); proxy
        # rows built from test rows only.
        avail_features = [f for f in features if f in avail]
        if not avail_features:
            return batch_params, 0, len(features) * len(outcomes)

        test_rows, bin_mat = _tt_bin_matrix(rows, avail_features, cutoff_date, n_bins=10)

        if len(test_rows) < 30:
            return batch_params, 0, len(avail_features) * len(outcomes)

        for fi, feature in enumerate(avail_features):
            for outcome in outcomes:
                if outcome not in avail or feature == outcome:
                    continue
                proxy = []
                for ri, r in enumerate(test_rows):
                    b = bin_mat[ri, fi]
                    if b == 0:
                        continue  # insufficient training history or NaN
                    o = _safe_float(r.get(outcome))
                    if math.isnan(o):
                        continue
                    proxy.append({
                        feature:      float(b),
                        outcome:      o,
                        "trade_date": r.get("trade_date"),
                    })
                if len(proxy) < 30:
                    skipped += 1
                    continue
                try:
                    result = scanner.scan_relationship(proxy, feature, outcome, ticker)
                except Exception:
                    skipped += 1
                    continue
                if "error" in result or result.get("n", 0) < 30:
                    skipped += 1
                    continue
                _append_params(batch_params, ticker, feature, outcome,
                               mode_label, cutoff_date_obj, result)
                scored += 1

    elif walk_forward:
        # ── Walk-forward mode ────────────────────────────────────────
        avail_features = [f for f in features if f in avail]
        if not avail_features:
            return batch_params, 0, len(features) * len(outcomes)

        bin_mat, valid_cum, nan_mask = _wf_bin_matrix(
            rows, avail_features, n_bins=10, warmup=252)

        for fi, feature in enumerate(avail_features):
            for outcome in outcomes:
                if outcome not in avail or feature == outcome:
                    continue
                proxy = []
                for i, r in enumerate(rows):
                    b = bin_mat[i, fi]
                    if b == 0:
                        continue
                    o = _safe_float(r.get(outcome))
                    if math.isnan(o):
                        continue
                    proxy.append({
                        feature:      float(b),
                        outcome:      o,
                        "trade_date": r.get("trade_date"),
                    })
                if len(proxy) < 30:
                    skipped += 1
                    continue
                try:
                    result = scanner.scan_relationship(proxy, feature, outcome, ticker)
                except Exception:
                    skipped += 1
                    continue
                if "error" in result or result.get("n", 0) < 30:
                    skipped += 1
                    continue
                _append_params(batch_params, ticker, feature, outcome,
                               mode_label, cutoff_date_obj, result)
                scored += 1

    else:
        # ── In-sample mode ───────────────────────────────────────────
        for feature in features:
            if feature not in avail:
                continue
            for outcome in outcomes:
                if outcome not in avail or feature == outcome:
                    continue
                try:
                    result = scanner.scan_relationship(rows, feature, outcome, ticker)
                except Exception:
                    skipped += 1
                    continue
                if "error" in result or result.get("n", 0) < 30:
                    skipped += 1
                    continue
                _append_params(batch_params, ticker, feature, outcome,
                               mode_label, cutoff_date_obj, result)
                scored += 1

    return batch_params, scored, skipped


async def run_batch_score(oi_pool, main_pool, walk_forward: bool = False,
                          cutoff_date: str = "", log=print):
    """Run the full batch scan. oi_pool=open_interest DB, main_pool=spx_interpolated DB."""
    _progress["running"]      = True
    _progress["walk_forward"] = walk_forward
    _progress["cutoff_date"]  = cutoff_date or None
    _progress["message"]      = "Starting…"
    if cutoff_date:
        mode_label = "train_test"
    elif walk_forward:
        mode_label = "walk_forward"
    else:
        mode_label = "in_sample"

    try:
        await ensure_table(main_pool)

        # 1. Get tickers
        async with oi_pool.acquire() as conn:
            ticker_rows = await conn.fetch(
                "SELECT DISTINCT ticker FROM daily_features ORDER BY ticker")
        tickers = [r["ticker"] for r in ticker_rows]
        log(f"Found {len(tickers)} tickers  [mode={mode_label}]")

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
                    and c not in _EXCLUDE_FEATURES
                    and not c.endswith("_pc")]
        outcomes = [c for c in all_cols if c.startswith("ret_") and "fwd" in c]

        total_combos = len(tickers) * len(features) * len(outcomes)
        log(f"Scanning: {len(tickers)} tickers × {len(features)} features × "
            f"{len(outcomes)} outcomes = {total_combos} combos")

        total_scored = 0
        total_skipped = 0

        for ti, ticker in enumerate(tickers):
            _progress["message"] = f"Ticker {ti+1}/{len(tickers)}: {ticker}"
            log(f"[{ti+1}/{len(tickers)}] {ticker}…")

            async with oi_pool.acquire() as conn:
                rows = await conn.fetch(
                    "SELECT * FROM daily_features "
                    "WHERE ticker = $1 AND trade_date >= '2020-01-01' "
                    "ORDER BY trade_date",
                    ticker)
            rows = [dict(r) for r in rows]
            for r in rows:
                if hasattr(r.get("trade_date"), "isoformat"):
                    r["trade_date"] = r["trade_date"].isoformat()
                else:
                    r["trade_date"] = str(r["trade_date"])

            if len(rows) < 30:
                log(f"  Skipping {ticker} — only {len(rows)} rows")
                total_skipped += len(features) * len(outcomes)
                continue

            # Score this ticker on a worker thread so the FastAPI event
            # loop stays free to serve other dashboards. The scoring is
            # numpy-heavy (np.searchsorted, scanner.scan_relationship),
            # and numpy releases the GIL during big array ops — so the
            # thread genuinely parallelizes with the main loop.
            batch_params, scored, skipped = await asyncio.to_thread(
                _score_one_ticker,
                ticker, rows, features, outcomes, mode_label,
                walk_forward, cutoff_date,
            )
            total_scored  += scored
            total_skipped += skipped

            if batch_params:
                async with main_pool.acquire() as conn:
                    await conn.executemany(_UPSERT, batch_params)
                log(f"  {len(batch_params)} scores saved")

        _progress["message"]  = f"Complete: {total_scored} scored, {total_skipped} skipped"
        _progress["last_run"] = datetime.utcnow().isoformat()
        log(f"Done. {total_scored} scored, {total_skipped} skipped.")

    except Exception as exc:
        _progress["message"] = f"Error: {exc}"
        log(f"BATCH ERROR: {exc}")
        raise
    finally:
        _progress["running"] = False


def _nz(v):
    """Convert NaN floats to None so they store as SQL NULL instead of
    PostgreSQL's float NaN. NaN-poisoned rows break downstream AVG()
    aggregates (AVG over a set containing NaN returns NaN, which the
    JSON encoder rejects with `Out of range float values are not JSON
    compliant`)."""
    if isinstance(v, float) and math.isnan(v):
        return None
    return v


def _append_params(batch_params: list, ticker: str, feature: str, outcome: str,
                   mode: str, cutoff_date_obj, result: dict) -> None:
    """Extract scanner result fields and append an _UPSERT param tuple.

    `cutoff_date_obj` is a `datetime.date` for train_test mode, None for
    in_sample / walk_forward. Stored as the 5th positional param (after
    mode) — matches the `_UPSERT` column order.

    Every float field is run through `_nz` so NaN becomes SQL NULL —
    NaN values in REAL columns poison AVG/MAX aggregates downstream.
    """
    rob  = result.get("robustness") or {}
    bs   = result.get("bucket_stats") or []
    valid_bs = [b for b in bs if b is not None]
    d10  = next((b for b in valid_bs if b.get("bucket") == 10), {})
    d1   = next((b for b in valid_bs if b.get("bucket") == 1),  {})
    best = result.get("best_single_bucket") or {}
    batch_params.append((
        ticker, feature, outcome, mode, cutoff_date_obj,
        _nz(result.get("composite_score")),
        result.get("pattern"),
        _nz(result.get("spearman_r")),
        _nz(result.get("monotonicity")),
        _nz(rob.get("yearly_consistency_pct")),
        _nz(rob.get("concentration_risk")),
        _nz(result.get("tail_spread")),
        result.get("n"),
        _nz(d10.get("avg_ret")),
        _nz(d1.get("avg_ret")),
        _nz(d10.get("win_rate")),
        _nz(d1.get("win_rate")),
        _nz(best.get("sharpe")),
        _nz(result.get("mi")),
        _nz(result.get("pearson_r")),
        rob.get("loyo_fragile"),
    ))


# CLI entry point
if __name__ == "__main__":
    import asyncpg
    import os
    import argparse
    from dotenv import load_dotenv
    load_dotenv()

    parser = argparse.ArgumentParser()
    parser.add_argument("--walk-forward", action="store_true",
                        help="Compute walk-forward scores instead of in-sample")
    args = parser.parse_args()

    async def main():
        oi_url   = os.environ.get("OI_DATABASE_URL")
        main_url = os.environ.get("DATABASE_URL")
        if not oi_url or not main_url:
            print("Set OI_DATABASE_URL and DATABASE_URL env vars")
            return
        oi_pool   = await asyncpg.create_pool(oi_url,   min_size=2, max_size=5)
        main_pool = await asyncpg.create_pool(main_url, min_size=2, max_size=5)
        try:
            await run_batch_score(oi_pool, main_pool, walk_forward=args.walk_forward)
        finally:
            await oi_pool.close()
            await main_pool.close()

    asyncio.run(main())
