"""OI Analysis workbench — interactive decile analytics for a single ticker/metric/outcome."""
import asyncio
import hashlib
import json
import math
from collections import defaultdict
from typing import List, Optional

import numpy as np
from scipy import stats as sp_stats
from fastapi import APIRouter, Body, Depends, Query
from pydantic import BaseModel

from app.db import get_pool, get_oi_pool

# ── Secondary Signal Scanner cache ────────────────────────────────────────────
_SEC_CACHE: dict = {}          # cache_key -> {rows, features, outcome, data_as_of, ...}
_SEC_SCORE_CACHE: dict = {}    # scan_key  → {status, result, error_msg}
_SEC_SCORE_CACHE_MAX = 50      # evict oldest entry when exceeded (in-memory only)
_sec_scan_running: bool = False  # one background job at a time

# ── Secondary Scanner DB persistence ──────────────────────────────────────────
#
# DISCIPLINE — same rule as _ANALYZE_PRIMARY_SCHEMA_VERSION and
# _GLOBAL_BINS_SCHEMA_VERSION: bump _SEC_SCAN_SCHEMA_VERSION on every
# change to scoring math (_sec_score_metrics), to the WF bin-assignment
# path (_compute_walk_forward_bin_stats), or to the result payload
# shape. Pre-v1 cache_keys had no version at all — same latent bug that
# bit analyze_primary_cache and global_bins_cache. The structural_key
# below carries the salt; the sweep in _ensure_sec_scan_table reclaims
# pre-bump rows on startup.
#
# v1: introducing the salt. Pre-v1 entries are unreachable on read
# (key prefix mismatch); the table-ensure DELETE below reclaims their
# disk. See MIGRATION_PRINCIPLES.md for the rule.
#
# v2 (Group 7): scanner scoring now reads bin20 from wf_bins (stored)
# instead of running _compute_walk_forward_bin_stats on the fly via
# _sec_score_metrics. Every cached scan row carries pre-Group-7 bins;
# the bump invalidates them. The IC-discipline hardcode at
# _run_sec_score:330 (WalkForwardSpec()) is unchanged — scanner is
# still always WF; only the bin source changes.
#
# v3 (Group 8): bumped for discipline consistency across the four
# salt-keyed caches in the same commit. Scanner never reaches TT
# (always-WF by IC discipline); this entry is structurally dead but
# bumped so the salt-discipline rule isn't violated on the next read.
_SEC_SCAN_SCHEMA_VERSION = 3

_SEC_SCAN_CACHE_MAX_ROWS = 50   # FIFO eviction when table reaches this count
_sec_scan_table_ensured: bool = False

_SEC_SCAN_TABLE_DDL = """\
CREATE TABLE IF NOT EXISTS sec_scan_cache (
    structural_key  TEXT         PRIMARY KEY,
    ticker          TEXT         NOT NULL,
    primary_metric  TEXT         NOT NULL,
    selected_bins   JSONB        NOT NULL,
    outcome         TEXT         NOT NULL,
    mode            TEXT         NOT NULL,
    cutoff_date     DATE,
    n_bins          INT          NOT NULL,
    data_as_of      DATE         NOT NULL,
    n_input_rows    INT,
    payload         JSONB        NOT NULL,
    cached_at       TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS sec_scan_cache_cached_at ON sec_scan_cache (cached_at);
"""


async def _ensure_sec_scan_table(pool) -> None:
    global _sec_scan_table_ensured
    if _sec_scan_table_ensured:
        return
    async with pool.acquire() as conn:
        await conn.execute(_SEC_SCAN_TABLE_DDL)
        # Sweep entries from any prior schema version. The salt makes them
        # unreachable on read; this DELETE reclaims their disk. No-op once
        # the table only holds current-version entries. Same pattern as
        # _ensure_analyze_primary_table and _ensure_bins_table.
        prefix = f"sv:v{_SEC_SCAN_SCHEMA_VERSION}:"
        await conn.execute(
            "DELETE FROM sec_scan_cache WHERE structural_key NOT LIKE $1",
            f"{prefix}%",
        )
    _sec_scan_table_ensured = True


def _sec_structural_key(
    ticker: str, metric: str, outcome: str, mode: str,
    cutoff: str, n_bins: int, selected_bins,
) -> str:
    """Stable DB lookup key — no date, no filtered_dates.  The key encodes every
    parameter that can change the result; data_as_of lives separately in the row.
    Salted with _SEC_SCAN_SCHEMA_VERSION so a bin-math or payload-shape change
    auto-invalidates every stale entry on next read."""
    bins_sorted = sorted(selected_bins) if selected_bins else []
    bins_hash = hashlib.sha256(
        json.dumps(bins_sorted, separators=(",", ":")).encode()
    ).hexdigest()[:12]
    cutoff_str = cutoff or "null"
    return (
        f"sv:v{_SEC_SCAN_SCHEMA_VERSION}:"
        f"sec:{ticker}:{metric}:{outcome}:{mode}:{cutoff_str}:{n_bins}:{bins_hash}"
    )


async def _write_sec_scan_cache(
    pool, structural_key: str, ticker: str, metric: str, outcome: str,
    mode: str, cutoff_date_str: str, n_bins: int, selected_bins,
    data_as_of_str: str, n_input_rows: int, result_dict: dict,
) -> None:
    """Upsert one secondary scan result into sec_scan_cache.  FIFO-evicts the
    oldest row when the table is at capacity.  Non-fatal on any DB error."""
    from datetime import date as _date
    try:
        payload_json = json.dumps(result_dict)
        bins_json    = json.dumps(sorted(selected_bins) if selected_bins else [])
        data_as_of_obj = _date.fromisoformat(data_as_of_str) if data_as_of_str else None
        cutoff_obj     = _date.fromisoformat(cutoff_date_str) if cutoff_date_str else None
        async with pool.acquire() as conn:
            count = await conn.fetchval("SELECT COUNT(*) FROM sec_scan_cache")
            if count >= _SEC_SCAN_CACHE_MAX_ROWS:
                await conn.execute(
                    "DELETE FROM sec_scan_cache WHERE cached_at = "
                    "(SELECT MIN(cached_at) FROM sec_scan_cache)"
                )
            await conn.execute(
                """INSERT INTO sec_scan_cache
                   (structural_key, ticker, primary_metric, selected_bins, outcome,
                    mode, cutoff_date, n_bins, data_as_of, n_input_rows, payload)
                   VALUES ($1,$2,$3,$4::jsonb,$5,$6,$7,$8,$9,$10,$11::jsonb)
                   ON CONFLICT (structural_key) DO UPDATE SET
                       data_as_of   = EXCLUDED.data_as_of,
                       n_input_rows = EXCLUDED.n_input_rows,
                       payload      = EXCLUDED.payload,
                       cached_at    = NOW()""",
                structural_key, ticker, metric, bins_json, outcome,
                mode, cutoff_obj, n_bins, data_as_of_obj, n_input_rows, payload_json,
            )
    except Exception:
        import logging
        logging.exception("sec_scan_cache DB write failed for structural_key=%s", structural_key)

# ── Trade-detail cache (W1) ────────────────────────────────────────────────────
# Populated by /analyze; read by GET /trades and GET /trades/csv.
# Key: "{ticker}:{metric}:{outcome}:{mode}:{cutoff_date}"
# Value: list of full trade-detail dicts (one per pair).
_TRADE_CACHE: dict = {}
_TRADE_CACHE_MAX = 5  # keep last 5 unique (ticker,metric,outcome,mode) analyses

# ── Full-response cache (W2) — in-memory hot tier ────────────────────────────
# Caches the complete /analyze response dict so mode switches and repeat
# Analyzes skip computation.  Backed by analyze_primary_cache (DB) so
# evictions fall back to a fast DB read rather than a 2-min recompute.
# Keyed by all params that affect the computation result.
# Cap 20 entries (was 4); DB fallback makes eviction cheap.
_ANALYZE_CACHE: dict = {}
_ANALYZE_CACHE_MAX = 20

# ── Primary analyze DB persistence ────────────────────────────────────────────
# Persistent JSONB store for the complete /analyze response.  Keyed on all
# 7 dimensions that distinguish a result: ticker, metric, outcome, mode,
# cutoff_date, date_from, date_to.  Bump _ANALYZE_PRIMARY_SCHEMA_VERSION to
# auto-invalidate stale entries after a payload-shape change.
#
# DISCIPLINE — bump this whenever the response shape OR the bin assignment
# math changes. The cache_key salt is the only thing that distinguishes a
# stale cached payload from a fresh one; if the salt doesn't move and the
# math does, every deploy serves the old payload until someone manually
# truncates the table. (That's exactly what bit us between the Part-1
# thinning removal and the v1 salt — the on-disk payloads had the legacy
# `decile20 = 0` sentinel for thin tickers; the heatmap reads is_bins
# live; secondary's filter reads trade_calendar.decile20 from the cached
# /analyze; gap = the thin-ticker rows. Manual truncate cleared one DB;
# the bump below makes the next deploy self-invalidate.)
#
# v2 (Part 1 thinning removal): /analyze's IS+ALL branch no longer applies
# the n_t<10 ticker drop or the n_t<20 → decile20=0 sentinel rewrite.
# Every row in valid_a10 now carries its real stored is_bins.bin20 in
# trade_calendar.decile20. v1 cache_keys are unreachable on read (key
# prefix mismatch); first hit per (ticker, metric, outcome, mode, cutoff,
# date_from, date_to) recomputes.
#
# v3 (Group 7): /analyze's WF+ALL branch now reads bin20 from wf_bins
# (stored) instead of the on-the-fly _walk_forward_bucket_* path. All
# WF-mode cached responses carry the on-the-fly bin assignments under
# v2 and stay byte-different from the stored-bin results (boundary-edge
# accumulation across the ticker universe — ~99 rows on the test
# baseline). Bump invalidates them; first hit per request recomputes
# against wf_bins.
#
# v4 (Group 8): TT+ALL now reads bin20 from tt_bins (stored, IS-frozen-
# at-cutoff) instead of the on-the-fly TrainTestAssigner that ran
# walk-forward-frozen-at-cutoff. The methodology shifted from WF-
# frozen to IS-frozen at the build side, so cached v3 TT payloads
# carry a different rank entirely (not just boundary edges). Bump
# invalidates them; first hit per request recomputes against tt_bins.
#
# v5 (Group 8 follow-up): TT primary quantile now applies the
# test-window filter (`trade_date >= cutoff`) at the SQL JOIN. Pre-fix
# v4 emitted full-series counts in trade_calendar / decile_stats_20
# under a "TEST PERIOD" subtitle. Per-bin n's shift by ~67% on the
# ALL TT case (~218k full → ~70k test). Bump invalidates v4 entries.
_ANALYZE_PRIMARY_SCHEMA_VERSION = 5
_ANALYZE_PRIMARY_CACHE_MAX_BYTES = 2 * 1024**3   # 2 GB LRU cap
_ANALYZE_PRIMARY_TABLE_DDL = """\
CREATE TABLE IF NOT EXISTS analyze_primary_cache (
    cache_key     TEXT  PRIMARY KEY,
    ticker        TEXT  NOT NULL,
    metric        TEXT  NOT NULL,
    outcome       TEXT  NOT NULL,
    mode          TEXT  NOT NULL,
    cutoff_date   DATE,
    date_from     DATE,
    date_to       DATE,
    payload       JSONB NOT NULL,
    payload_bytes INT,
    cached_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_accessed TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS analyze_primary_cache_last_accessed
    ON analyze_primary_cache (last_accessed);
"""
_analyze_primary_table_ensured: bool = False


def _analyze_primary_cache_key(
    ticker: str, metric: str, outcome: str, mode: str,
    cutoff_date: Optional[str], date_from: Optional[str], date_to: Optional[str],
) -> str:
    """Stable DB key for the primary /analyze result."""
    cutoff_s    = cutoff_date or "null"
    date_from_s = date_from or ""
    date_to_s   = date_to or ""
    return (f"ap:v{_ANALYZE_PRIMARY_SCHEMA_VERSION}:"
            f"{ticker}:{metric}:{outcome}:{mode}:{cutoff_s}:{date_from_s}:{date_to_s}")


async def _ensure_analyze_primary_table(pool) -> None:
    global _analyze_primary_table_ensured
    if _analyze_primary_table_ensured:
        return
    async with pool.acquire() as conn:
        await conn.execute(_ANALYZE_PRIMARY_TABLE_DDL)
        prefix = f"ap:v{_ANALYZE_PRIMARY_SCHEMA_VERSION}:"
        await conn.execute(
            "DELETE FROM analyze_primary_cache WHERE cache_key NOT LIKE $1",
            f"{prefix}%",
        )
    _analyze_primary_table_ensured = True


async def _evict_analyze_primary_lru(pool) -> int:
    """Drop oldest-accessed rows until table size falls back under the cap."""
    evicted = 0
    async with pool.acquire() as conn:
        while True:
            size_bytes = await conn.fetchval(
                "SELECT pg_total_relation_size('analyze_primary_cache')")
            if size_bytes is None or size_bytes <= _ANALYZE_PRIMARY_CACHE_MAX_BYTES:
                break
            oldest_key = await conn.fetchval(
                "SELECT cache_key FROM analyze_primary_cache "
                "ORDER BY last_accessed ASC LIMIT 1")
            if oldest_key is None:
                break
            await conn.execute(
                "DELETE FROM analyze_primary_cache WHERE cache_key = $1", oldest_key)
            evicted += 1
    return evicted


async def _write_analyze_primary_cache(
    pool, cache_key: str, ticker: str, metric: str, outcome: str,
    mode: str, cutoff_date: Optional[str], date_from: Optional[str],
    date_to: Optional[str], result: dict,
) -> None:
    """Upsert the primary /analyze result into analyze_primary_cache.
    Fire-and-forget via asyncio.create_task — failures are non-fatal."""
    import logging
    from datetime import date as _date
    from fastapi.encoders import jsonable_encoder
    global _analyze_primary_table_ensured
    try:
        _analyze_primary_table_ensured = False  # reset so ensure re-runs on every write
        await _ensure_analyze_primary_table(pool)
        # jsonable_encoder converts numpy types / date objects before json.dumps
        payload_json = json.dumps(jsonable_encoder(result))
        cutoff_obj   = _date.fromisoformat(cutoff_date) if cutoff_date else None
        from_obj     = _date.fromisoformat(date_from) if date_from else None
        to_obj       = _date.fromisoformat(date_to) if date_to else None
        async with pool.acquire() as conn:
            await conn.execute(
                """INSERT INTO analyze_primary_cache
                   (cache_key, ticker, metric, outcome, mode, cutoff_date,
                    date_from, date_to, payload, payload_bytes)
                   VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9::jsonb,$10)
                   ON CONFLICT (cache_key) DO UPDATE SET
                       payload       = EXCLUDED.payload,
                       payload_bytes = EXCLUDED.payload_bytes,
                       cached_at     = NOW(),
                       last_accessed = NOW()""",
                cache_key, ticker, metric, outcome, mode, cutoff_obj,
                from_obj, to_obj, payload_json, len(payload_json),
            )
        try:
            await _evict_analyze_primary_lru(pool)
        except Exception as _evict_e:
            logging.warning("analyze_primary_cache LRU eviction failed: %r", _evict_e)
    except Exception:
        logging.exception("analyze_primary_cache write failed for key %r", cache_key)


def _sec_score_key(cache_key: str, selected_primary_bins, n_bins: int, filtered_dates) -> str:
    """Stable hash for deduplicating secondary score jobs."""
    parts = {
        "ck": cache_key,
        "bins": sorted(selected_primary_bins) if selected_primary_bins else [],
        "n": n_bins,
        "fd": sorted(filtered_dates) if filtered_dates else [],
    }
    return hashlib.sha256(json.dumps(parts, separators=(",", ":")).encode()).hexdigest()[:20]


def _run_sec_score(
    scan_key: str,
    rows: list,
    outcome_col: str,
    feature_cols: list,
    is_all: bool,
    n_bins: int,
    primary_metric: str,
    selected_primary_bins,
    filtered_dates: list,
    db_write_params=None,   # dict with loop/pool/structural_key/etc.; None = no DB write
    bin20_by_metric: Optional[dict] = None,
) -> None:
    """Synchronous secondary score computation — runs in thread-pool executor.
    Sets _sec_scan_running True at entry, False in finally.
    When db_write_params is provided, persists result to sec_scan_cache on success.

    Walk-forward is DELIBERATELY hardcoded here, regardless of the user's
    page-mode toggle. The Signal Scanner ranks features by their
    out-of-sample predictive lift; ranking on in-sample bins would
    score signals against the same data they were fit on — the exact
    lookahead the dashboard avoids everywhere else. The user's page
    mode is intentionally ignored for this one endpoint.

    Group 7: scoring now reads bin20 from wf_bins (stored) via
    `bin20_by_metric` — same shape Group 4 established. The IC
    discipline above is unchanged; only the bin source moved from
    on-the-fly _walk_forward_bucket_* to wf_bins. The async caller
    prefetches and threads the lookup in. For features absent from
    wf_bins (the 7-missing pattern) the per-feature loop falls back
    to the on-the-fly path via the helper's fallback branches."""
    global _sec_scan_running
    _sec_scan_running = True
    try:
        from app.routers.row_compute import WalkForwardSpec, filter_by_assignments
        spec = WalkForwardSpec()
        # Group 7: extract the primary metric's wf_bin20 lookup for the
        # filter step. None means "use the on-the-fly fallback".
        primary_bin20_by_key = None
        if bin20_by_metric:
            primary_bin20_by_key = bin20_by_metric.get(primary_metric)
        filtered, dropped, universe = filter_by_assignments(
            rows, spec, primary_metric, selected_primary_bins, is_all, filtered_dates,
            primary_bin20_by_key=primary_bin20_by_key,
        )
        metrics = _sec_score_metrics(
            filtered, outcome_col, feature_cols, is_all, n_bins, spec,
            all_rows=rows,   # full cache → secondary binned on full-universe WF
            bin20_by_metric=bin20_by_metric,
        )
        baseline_rets = [float(r[outcome_col]) for r in filtered
                         if r.get(outcome_col) is not None]
        baseline = {
            "n": len(baseline_rets),
            "avg_ret": round(float(np.mean(baseline_rets)), 6) if baseline_rets else 0,
            "win_rate": round(
                float(np.mean([1.0 if v > 0 else 0.0 for v in baseline_rets])), 4
            ) if baseline_rets else 0,
        }
        start_date = filtered[0]["trade_date"] if filtered else None
        result_dict = {
            "baseline":         baseline,
            "metrics":          metrics,
            "mode":             "walk_forward",
            "warmup":           spec.warmup,
            "cutoff_date":      None,
            "universe_n":       universe,
            "start_date":       start_date,
            "data_as_of":       db_write_params["data_as_of"] if db_write_params else None,
        }
        _SEC_SCORE_CACHE[scan_key] = {"status": "done", "result": result_dict}
        # Persist to DB so the next server session finds this result instantly.
        if db_write_params:
            try:
                future = asyncio.run_coroutine_threadsafe(
                    _write_sec_scan_cache(
                        db_write_params["pool"],
                        db_write_params["structural_key"],
                        db_write_params["ticker"],
                        db_write_params["metric"],
                        db_write_params["outcome"],
                        db_write_params["mode"],
                        db_write_params.get("cutoff_date", ""),
                        n_bins,
                        selected_primary_bins,
                        db_write_params["data_as_of"],
                        db_write_params["n_input_rows"],
                        result_dict,
                    ),
                    db_write_params["loop"],
                )
                future.result(timeout=15)  # block thread ≤15s; failure is non-fatal
            except Exception:
                import logging
                logging.warning("sec_scan_cache background write did not complete within 15s or raised",
                                exc_info=True)
    except Exception as exc:
        _SEC_SCORE_CACHE[scan_key] = {
            "status": "error",
            "error_msg": f"{type(exc).__name__}: {exc}",
            "result": None,
        }
    finally:
        _sec_scan_running = False


router = APIRouter(tags=["oi_analysis"])



def _bucket_pairs(pairs, n=10):
    """Sort by x, split into n equal-count buckets. Returns list of lists of (x, y, date)."""
    if not pairs:
        return [[] for _ in range(n)]
    s = sorted(pairs, key=lambda p: p[0])
    total = len(s)
    buckets = [[] for _ in range(n)]
    for i, p in enumerate(s):
        b = min(int(i / total * n), n - 1)
        buckets[b].append(p)
    return buckets


def _bucket_pairs_per_ticker(by_ticker, n=10):
    """
    Per-ticker decile normalization: each ticker's trades are independently
    ranked 1..n, then pooled. Returns n buckets (same structure as _bucket_pairs).
    Tickers with fewer than n observations are excluded.
    """
    buckets = [[] for _ in range(n)]
    for tkr_pairs in by_ticker.values():
        if len(tkr_pairs) < n:
            continue
        tkr_buckets = _bucket_pairs(tkr_pairs, n)
        for i, bucket in enumerate(tkr_buckets):
            buckets[i].extend(bucket)
    return buckets


def _compute_bucket_stats(buckets: list) -> list:
    """Compute per-bucket stats for any list of buckets (10-bin or 20-bin)."""
    result = []
    for i, bucket in enumerate(buckets):
        if not bucket:
            result.append(None)
            continue
        ys = np.array([p[1] for p in bucket])
        xs = [p[0] for p in bucket]
        result.append({
            "bucket":   i + 1,
            "n":        len(bucket),
            "avg_ret":  round(float(ys.mean()), 6),
            "win_rate": round(float((ys > 0).mean()), 4),
            "std_dev":  round(float(ys.std()), 6),
            "sharpe":   round(float(ys.mean() / ys.std()), 4) if ys.std() > 0 else 0,
            "min_val":  round(float(min(xs)), 6),
            "max_val":  round(float(max(xs)), 6),
        })
    return result


def _parse_horizon(col_name: str) -> int:
    import re
    m = re.search(r'(\d+)d', col_name)
    return int(m.group(1)) if m else 1


@router.get("/tickers")
async def list_tickers(pool=Depends(get_oi_pool)):
    if not pool:
        return []
    async with pool.acquire() as conn:
        rows = await conn.fetch("SELECT DISTINCT ticker FROM daily_features ORDER BY ticker")
    return [r["ticker"] for r in rows]


@router.get("/columns")
async def list_columns(pool=Depends(get_oi_pool)):
    if not pool:
        return {"features": [], "outcomes": []}
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """SELECT column_name FROM information_schema.columns
               WHERE table_name = 'daily_features' AND table_schema = 'public'
               AND data_type IN ('double precision','numeric','real','integer','bigint','smallint')
               AND column_name NOT IN ('id','ticker','trade_date','created_at','updated_at')
               ORDER BY ordinal_position""")
    all_cols = [r["column_name"] for r in rows]
    outcomes = [c for c in all_cols if "ret_" in c and "fwd" in c]
    features = [c for c in all_cols if c not in outcomes and not c.endswith("_pc")]
    return {"features": features, "outcomes": outcomes}


@router.get("/analyze")
async def analyze(
    ticker: str = Query(...),
    metric: str = Query(...),
    outcome: str = Query(...),
    date_from: Optional[str] = Query(None),
    date_to: Optional[str] = Query(None),
    walk_forward: bool = Query(False),
    cutoff_date: Optional[str] = Query(None),
    force: bool = Query(False),
    pool=Depends(get_oi_pool),
):
    """Full analysis payload for one ticker/metric/outcome combo.

    `force=true` bypasses both in-memory and DB cache reads and recomputes
    from source.  The Refresh button passes force=true; normal page loads
    pass force=false (default).
    """
    if not pool:
        return {"error": "OI database not configured"}
    await _ensure_analyze_primary_table(pool)

    is_all = (ticker == "ALL")

    # Construct the active spec once at the function entry so both
    # is_all branches and the response envelope below share it.
    from app.routers.row_compute import make_spec
    spec = make_spec(walk_forward, cutoff_date)

    # ── Per-phase timing (diagnostic) — writes to /tmp/analyze_timing.log ──
    import time as _time
    _t0 = _time.perf_counter()
    def _tlog(label: str) -> None:
        elapsed = _time.perf_counter() - _t0
        line = f"[analyze][{ticker}|ALL={ticker == 'ALL'}|{spec.kind}] +{elapsed:.3f}s  {label}\n"
        try:
            with open('/tmp/analyze_timing.log', 'a') as _f:
                _f.write(line)
        except OSError:
            pass  # non-fatal on Windows dev box

    _tlog('start')

    # ── W2: full-response cache (in-memory → DB → compute) ───────────────────
    _cutoff_s = spec.cutoff.isoformat() if spec.kind == 'train_test' else ''
    _ac_key = (f"{ticker}:{metric}:{outcome}:{spec.kind}:"
               f"{_cutoff_s}:{date_from or ''}:{date_to or ''}")
    _pc_key = _analyze_primary_cache_key(
        ticker, metric, outcome, spec.kind,
        _cutoff_s or None, date_from, date_to,
    )
    _tlog(f'W2 key="{_ac_key}" cache_size={len(_ANALYZE_CACHE)} in_cache={_ac_key in _ANALYZE_CACHE} force={force}')

    if not force:
        # Tier 1: in-memory hot cache
        if _ac_key in _ANALYZE_CACHE:
            _tlog('W2 mem-hit')
            _hit = dict(_ANALYZE_CACHE[_ac_key])  # shallow copy — don't mutate stored entry
            _hit["_handler_ms"] = round((_time.perf_counter() - _t0) * 1000)
            return _hit

        # Tier 2: DB persistent cache
        try:
            async with pool.acquire() as conn:
                _db_row = await conn.fetchrow(
                    "SELECT payload FROM analyze_primary_cache WHERE cache_key = $1",
                    _pc_key,
                )
            if _db_row is not None:
                _tlog('W2 db-hit')
                _db_result = json.loads(_db_row["payload"])
                # Warm in-memory tier; evict oldest if at cap
                if len(_ANALYZE_CACHE) >= _ANALYZE_CACHE_MAX:
                    del _ANALYZE_CACHE[next(iter(_ANALYZE_CACHE))]
                _ANALYZE_CACHE[_ac_key] = _db_result
                # Touch last_accessed in background (non-blocking)
                async def _touch_primary(_key=_pc_key):
                    try:
                        async with pool.acquire() as _c:
                            await _c.execute(
                                "UPDATE analyze_primary_cache "
                                "SET last_accessed = NOW() WHERE cache_key = $1", _key)
                    except Exception:
                        pass
                asyncio.create_task(_touch_primary())
                _hit = dict(_db_result)
                _hit["_handler_ms"] = round((_time.perf_counter() - _t0) * 1000)
                return _hit
            _tlog('W2 db-miss — computing')
        except Exception as _db_read_e:
            import logging as _log
            _log.warning("analyze_primary_cache read failed for %r: %r", _pc_key, _db_read_e)
            _tlog(f'W2 db-read-error — computing')

    # Build date filter params (shared by both modes)
    date_conditions = ""
    params: list = []
    p = 1
    if date_from:
        from datetime import date as _date
        date_conditions += f" AND trade_date >= ${p}"; params.append(_date.fromisoformat(date_from)); p += 1
    if date_to:
        from datetime import date as _date
        date_conditions += f" AND trade_date <= ${p}"; params.append(_date.fromisoformat(date_to)); p += 1
    if not is_all:
        date_conditions += f" AND ticker = ${p}"; params.append(ticker); p += 1

    horizon = _parse_horizon(outcome)

    _max_trade_date: str = ""   # set after each branch's DB fetch; stored in cache for staleness

    # ── Data fetch & bucketing ────────────────────────────────────────────
    # ── v10 Group 3a — ALL+IS reads bin from is_bins ─────────────────────
    # JOIN preserves outcome-validity (Group 1 verified +512 inflation
    # without this filter). Combined fetch: data fields + bin20 in ONE
    # round-trip, no separate is_bins query.
    #
    # Display granularity: legacy /analyze emits BOTH 10-bin (decile_stats,
    # buckets) and 20-bin (decile_stats_20, trade_calendar.decile20). The
    # 10-bin is derived from bin20 via `((bin20 - 1) // 2) + 1`, which is
    # mathematically identical to a direct 10-bin per-ticker rank for every
    # rank/n_t (verified in Group 1's aggregation math). The 20-bin is
    # is_bins's bin20 directly.
    #
    # Per-ticker thinning matches legacy exactly:
    #   - 10-bin: tickers with < 10 valid rows excluded entirely
    #     (`_bucket_pairs_per_ticker` excludes tickers below the threshold).
    #   - 20-bin: tickers with 10..19 rows get `decile20 = 0` sentinel
    #     (legacy semantic; downstream consumers check for the 0 sentinel).
    #
    # WF and TT use the existing Assigner path below — same mode boundary
    # as Groups 1 and 2.
    # Group 7 extension: WF+ALL also takes the stored-bin path, JOINing
    # wf_bins instead of is_bins. The whole stored-bin code block is
    # mode-agnostic once the source table is parameterized — the math
    # is identical, only the JOIN target differs. Encoding A means
    # `WHERE bin20 > 0` cleanly drops both warm-up and null rows in WF.
    if spec.kind in {"in_sample", "walk_forward", "train_test"}:
        # Source table dispatched by mode. Group 8: TT joins tt_bins,
        # which encodes IS-frozen-at-cutoff bins (not WF-frozen).
        bin_table = {
            "in_sample":   "is_bins",
            "walk_forward": "wf_bins",
            "train_test":  "tt_bins",
        }[spec.kind]
        # Group 8 fix: TT mode is test-window-only — primary quantile,
        # decile_stats, trade_calendar all report on rows where
        # trade_date >= cutoff. The training window (< cutoff) is shown
        # only on the heatmap (left grid) for visual validation; every
        # other surface in TT mode is test-only. Pre-fix this branch
        # emitted the full series and the "TEST PERIOD" subtitle
        # disagreed with the bar sum.
        tt_extra_where = ""
        tt_extra_params: list = []
        if spec.kind == "train_test":
            from datetime import date as _date_cls_tt
            tt_extra_where = f" AND df.trade_date >= ${len(params) + 1}"
            cutoff_obj = (spec.cutoff
                          if hasattr(spec.cutoff, "isoformat")
                          else _date_cls_tt.fromisoformat(str(spec.cutoff)))
            tt_extra_params = [cutoff_obj]
        join_sql = (
            f"SELECT df.ticker, df.trade_date, "
            f"  df.{metric}, df.{outcome}, "
            f"  df.spot_co, df.spot_pc, "
            f"  bt.bin20_{metric} AS bin_20 "
            f"FROM daily_features df "
            f"JOIN {bin_table} bt USING (ticker, trade_date) "
            f"WHERE bt.bin20_{metric} > 0 "
            f"  AND df.{outcome} IS NOT NULL"
            f"{date_conditions}{tt_extra_where} "
            f"ORDER BY df.ticker, df.trade_date"
        )
        try:
            async with pool.acquire() as conn:
                rows = await conn.fetch(join_sql, *params, *tt_extra_params)
        except Exception:
            rows = []   # column bin20_{metric} absent (null-by-design metric)
        row_dicts = [dict(r) for r in rows]
        if row_dicts:
            _max_trade_date = str(max(r['trade_date'] for r in row_dicts))
        _tlog(f'ALL+{spec.kind} {bin_table} JOIN fetch ({len(row_dicts)} rows)')

        # Build by_ticker + spot lookup + bin20 lookup in one pass.
        # Same null/NaN guards as the legacy ALL path.
        by_ticker: dict = defaultdict(list)
        all_open_by_tkr_date: dict = {}
        bin20_lookup: dict = {}
        for r in row_dicts:
            xv, yv = r.get(metric), r.get(outcome)
            if xv is None or yv is None:
                continue
            try:
                xf, yf = float(xv), float(yv)
            except (ValueError, TypeError):
                continue
            if math.isnan(xf) or math.isnan(yf):
                continue
            by_ticker[r['ticker']].append((xf, yf, r['trade_date'], r['ticker']))
            bin20_lookup[(r['ticker'], r['trade_date'])] = r['bin_20']
            if r.get('spot_co') is not None:
                try:
                    all_open_by_tkr_date[(r['ticker'], str(r['trade_date']))] = round(float(r['spot_co']), 2)
                except (ValueError, TypeError):
                    pass

        # No per-ticker thinning. The only row filter is the stored
        # `bin20 > 0` (already applied at the SQL JOIN's WHERE clause,
        # which excludes null-metric rows). Every ticker that has rows
        # surviving the JOIN gets its stored bin20 emitted as-is —
        # the dashboard does no `n_t < N` exclusion of its own.
        valid_a10: list = []
        for tkr, ps in by_ticker.items():
            for xf, yf, td, _ in ps:
                bin20 = bin20_lookup.get((tkr, td))
                if bin20 is None:
                    continue
                bin10 = ((bin20 - 1) // 2) + 1
                valid_a10.append({
                    "ticker":         tkr,
                    "trade_date":     td,
                    "metric_value":   xf,
                    "forward_return": yf,
                    "bin10":          bin10,
                    "bin20":          bin20,
                })

        # Legacy in_sample ALL pairs iteration order:
        # sort by (trade_date, bin10, ticker, metric_value).
        valid_a10.sort(key=lambda a: (a['trade_date'], a['bin10'], a['ticker'], a['metric_value']))

        pairs          = []
        pairs_decile   = []
        pairs_decile20 = []
        buckets        = [[] for _ in range(10)]
        buckets_20_all = [[] for _ in range(20)]
        for a in valid_a10:
            pair = (a['metric_value'], a['forward_return'], a['trade_date'], a['ticker'])
            b10 = a['bin10']
            b20 = a['bin20']
            pairs.append(pair)
            pairs_decile.append(b10)
            pairs_decile20.append(b20)
            buckets[b10 - 1].append(pair)
            # No more bin20=0 sentinel — every row from valid_a10 carries a
            # real stored bin20, so the legacy `if b20 > 0` guard is vacuous
            # and removed along with the sentinel.
            buckets_20_all[b20 - 1].append(pair)

        # Bucket ordering: IS uses (ticker, metric_value); WF uses
        # (ticker, trade_date). Matches the existing legacy ALL branch's
        # spec-dispatched sort_key below.
        if spec.kind == "walk_forward":
            sort_key = lambda p: (p[3], p[2])
        else:
            sort_key = lambda p: (p[3], p[0])
        for b in range(10):
            buckets[b].sort(key=sort_key)
        for b in range(20):
            buckets_20_all[b].sort(key=sort_key)

        wf_dropped = 0  # Group 7 removed the per-request warm-up count
        decile_stats_20 = _compute_bucket_stats(buckets_20_all)
        _tlog('ALL+IS bucket_setup + decile_stats_20')
        # No per-ticker thinning — count every ticker that contributed any
        # valid rows. The legacy `if len(ps) >= 10` filter is removed.
        n_tickers_used = len(by_ticker)

        # ── Spot calendar (same logic as the legacy ALL path) ──────────────
        spot_series = []
        all_spot_dates = []
        open_by_date = {}
        close_by_date = {}
        tickers_in_data = list(by_ticker.keys())
        async with pool.acquire() as conn:
            all_spot_rows = await conn.fetch(
                "SELECT ticker, trade_date, spot_pc FROM daily_features "
                "WHERE ticker = ANY($1) AND spot_co IS NOT NULL "
                "ORDER BY ticker, trade_date", tickers_in_data)
        _all_dates_by_tkr: dict = defaultdict(list)
        _pc_by_tkr_date: dict = {}
        for r in all_spot_rows:
            tkr = r['ticker']; d = str(r['trade_date'])
            _all_dates_by_tkr[tkr].append(d)
            if r['spot_pc'] is not None:
                try:
                    _pc_by_tkr_date[(tkr, d)] = round(float(r['spot_pc']), 2)
                except (ValueError, TypeError):
                    pass
        all_dates_list_by_tkr: dict = dict(_all_dates_by_tkr)
        all_date_idx_by_tkr: dict = {tkr: {d: i for i, d in enumerate(dates)}
                                      for tkr, dates in _all_dates_by_tkr.items()}
        all_close_by_tkr: dict = {}
        for tkr, dates in _all_dates_by_tkr.items():
            closes = {}
            for i in range(len(dates) - 1):
                npc = _pc_by_tkr_date.get((tkr, dates[i + 1]))
                if npc is not None:
                    closes[dates[i]] = npc
            all_close_by_tkr[tkr] = closes
        _tlog('ALL+IS spot calendar')

    # Single-ticker: bridge the trade-calendar variables from the stored-bin
    # path's per-ticker dicts to the flat dicts the downstream else-branch uses.
    if not is_all:
        open_by_date   = {d: v for (_, d), v in all_open_by_tkr_date.items()}
        all_spot_dates = all_dates_list_by_tkr.get(ticker, [])
        close_by_date  = all_close_by_tkr.get(ticker, {})
        spot_series    = [{"date": d, "value": v} for d, v in sorted(open_by_date.items())]

    n = len(pairs)
    if n < 30:
        return {"error": f"Insufficient data: {n} valid rows", "n": n}

    xa = np.array([p[0] for p in pairs])
    ya = np.array([p[1] for p in pairs])

    # ── Decile stats (same structure for both modes) ──────────────────────
    decile_stats = []
    for i, bucket in enumerate(buckets):
        if not bucket:
            decile_stats.append(None)
            continue
        ys = np.array([p[1] for p in bucket])
        xs = [p[0] for p in bucket]
        decile_stats.append({
            "bucket":   i + 1,
            "n":        len(bucket),
            "avg_ret":  round(float(ys.mean()), 6),
            "med_ret":  round(float(np.median(ys)), 6),
            "win_rate": round(float((ys > 0).mean()), 4),
            "std_dev":  round(float(ys.std()), 6),
            "sharpe":   round(float(ys.mean() / ys.std()), 4) if ys.std() > 0 else 0,
            "min_val":  round(float(min(xs)), 6),
            "max_val":  round(float(max(xs)), 6),
            "returns":  [round(float(y), 6) for y in ys],
        })
    # decile_stats_20 already computed above (same bucketing as trade_calendar.decile20)

    # ── Correlations ─────────────────────────────────────────────────────
    if is_all and by_ticker:
        # Average per-ticker Spearman/Pearson (cross-ticker pooling is misleading)
        # No per-ticker `len(tkr_pairs) < 20` floor — every ticker that has
        # variance on both axes contributes to the average. The std > 0
        # check below still skips degenerate constant series.
        ticker_corrs = []
        for tkr_pairs in by_ticker.values():
            tx = np.array([p[0] for p in tkr_pairs])
            ty = np.array([p[1] for p in tkr_pairs])
            if tx.std() > 0 and ty.std() > 0:
                pr_t, _ = sp_stats.pearsonr(tx, ty)
                sr_t, _ = sp_stats.spearmanr(tx, ty)
                ticker_corrs.append((float(pr_t), float(sr_t)))
        pr = float(np.mean([c[0] for c in ticker_corrs])) if ticker_corrs else 0.0
        sr = float(np.mean([c[1] for c in ticker_corrs])) if ticker_corrs else 0.0
        pp, sp_val = 0.5, 0.5
    else:
        pr, pp = sp_stats.pearsonr(xa, ya)
        sr, sp_val = sp_stats.spearmanr(xa, ya)

    # ── Monotonicity & pattern ────────────────────────────────────────────
    avgs = [d["avg_ret"] for d in decile_stats if d is not None]
    if len(avgs) >= 2:
        transitions = sum(1 for i in range(len(avgs)-1) if avgs[i+1] > avgs[i])
        mono_raw = transitions / (len(avgs)-1)
        monotonicity = round(abs(mono_raw - 0.5) * 2, 4)
    else:
        monotonicity = 0

    overall_range = max(avgs) - min(avgs) if avgs else 0
    if overall_range < 1e-8:
        pattern = "flat"
    elif monotonicity > 0.75 and abs(sr) > 0.03:
        pattern = "monotonic_positive" if sr > 0 else "monotonic_negative"
    else:
        diffs = [avgs[i+1]-avgs[i] for i in range(len(avgs)-1)]
        max_diff = max(abs(d) for d in diffs) if diffs else 0
        if max_diff > overall_range * 0.5:
            pattern = "threshold"
        elif abs(pr) > 0.03:
            pattern = "linear_weak"
        else:
            pattern = "no_clear_pattern"

    # ── Yearly breakdown (all deciles combined) ───────────────────────────
    by_year: dict = defaultdict(list)
    for pair in pairs:
        y_val, d = pair[1], pair[2]
        yr = d.year if hasattr(d, 'year') else int(str(d)[:4])
        by_year[yr].append(y_val)

    yearly = []
    for yr in sorted(by_year):
        ys_yr = np.array(by_year[yr])
        yearly.append({
            "year":     yr,
            "n":        len(ys_yr),
            "avg_ret":  round(float(ys_yr.mean()), 6),
            "win_rate": round(float((ys_yr > 0).mean()), 4),
        })

    # ── Equity curve ─────────────────────────────────────────────────────
    def _equity_for_decile(decile_idx, mode="concurrent"):
        bucket = buckets[decile_idx] if 0 <= decile_idx < len(buckets) else []
        if not bucket:
            return {"points": [], "n_trades": 0}
        sorted_trades = sorted(bucket, key=lambda p: p[2])
        if mode == "non_overlapping":
            trades, last_date = [], None
            for p in sorted_trades:
                d = p[2]
                dd = d.date() if hasattr(d, 'date') else d
                if last_date is None or (dd - last_date).days >= horizon:
                    trades.append((dd, p[1]))
                    last_date = dd
        else:
            trades = [(p[2], p[1]) for p in sorted_trades]
        cum = peak = 0.0
        max_dd = 0.0
        points = []
        wins = 0
        _prev_date_str: str = ""
        for date, ret in trades:
            cum += ret
            peak = max(peak, cum)
            max_dd = min(max_dd, cum - peak)
            if ret > 0:
                wins += 1
            # Deduplicate to one point per calendar date: multiple trades on
            # the same date (e.g. different tickers in ALL mode) all contribute
            # to the cumulative sum, but only the final value for that date is
            # emitted.  Reduces points from ~17K/decile → ~1250 (trading days).
            _ds = str(date)
            if _ds == _prev_date_str:
                points[-1]["value"] = round(cum, 6)
            else:
                points.append({"date": _ds, "value": round(cum, 6)})
                _prev_date_str = _ds
        nn = len(trades)
        return {
            "points":     points,
            "n_trades":   nn,
            "cum_return": round(cum, 4),
            "max_dd":     round(max_dd, 4),
            "avg_ret":    round(sum(r for _, r in trades) / nn, 6) if nn else 0,
            "win_rate":   round(wins / nn, 4) if nn else 0,
        }

    equity_by_decile = {}
    for i in range(10):
        equity_by_decile[i+1] = {
            "concurrent":      _equity_for_decile(i, "concurrent"),
            "non_overlapping": _equity_for_decile(i, "non_overlapping"),
        }
    _tlog(f'common decile_stats + equity_by_decile ({n} pairs)')

    # ── Yearly consistency ────────────────────────────────────────────────
    # Fixed: use STORED bin10 from bin20_lookup to identify D10/D1 rows
    # within each year.  No within-year re-ranking of the raw metric value.
    # A row in stored D10 stays D10 regardless of which year we filter to —
    # that is the entire point of stability analysis.  Both ALL and
    # single-ticker modes populate bin20_lookup via the same stored-bin JOIN,
    # so the logic is unified here.
    yearly_consistency = []
    years_top_wins = 0

    all_years = sorted(by_year.keys())
    for yr in all_years:
        top_ys: list = []
        bot_ys: list = []
        for tkr, tkr_pairs in by_ticker.items():
            for p in tkr_pairs:
                yr_p = p[2].year if hasattr(p[2], 'year') else int(str(p[2])[:4])
                if yr_p != yr:
                    continue
                b20 = bin20_lookup.get((tkr, p[2]))
                if not b20 or b20 <= 0:
                    continue
                b10 = ((b20 - 1) // 2) + 1
                if b10 == 10:
                    top_ys.append(p[1])
                elif b10 == 1:
                    bot_ys.append(p[1])
        t_avg = float(np.mean(top_ys)) if top_ys else 0.0
        b_avg = float(np.mean(bot_ys)) if bot_ys else 0.0
        top_beats = t_avg > b_avg
        if top_beats:
            years_top_wins += 1
        yearly_consistency.append({
            "year": yr, "top_avg": round(t_avg, 6), "bot_avg": round(b_avg, 6),
            "top_n": len(top_ys), "bot_n": len(bot_ys), "top_beats": top_beats,
        })

    n_years = len(yearly_consistency)
    consistency_pct = round(years_top_wins / n_years * 100, 1) if n_years else None
    _tlog('common yearly_consistency')

    # ── Half-sample stability ─────────────────────────────────────────────
    # Fixed: use STORED bin10 to identify D10/D1 rows in each chronological
    # half.  No within-half re-ranking of the raw metric value.
    if is_all and by_ticker:
        # Per-ticker: split each ticker's rows at the chronological midpoint,
        # then check whether stored-D10 avg_ret > stored-D1 avg_ret in BOTH
        # halves (or both negative).  Majority vote across tickers.
        ticker_stable = []
        for tkr, tkr_pairs in by_ticker.items():
            tkr_sorted = sorted(tkr_pairs, key=lambda p: p[2])
            mid_t = len(tkr_sorted) // 2
            h1_top: list = []; h1_bot: list = []
            h2_top: list = []; h2_bot: list = []
            for i, p in enumerate(tkr_sorted):
                b20 = bin20_lookup.get((tkr, p[2]))
                if not b20 or b20 <= 0:
                    continue
                b10 = ((b20 - 1) // 2) + 1
                if i < mid_t:
                    if b10 == 10: h1_top.append(p[1])
                    elif b10 == 1: h1_bot.append(p[1])
                else:
                    if b10 == 10: h2_top.append(p[1])
                    elif b10 == 1: h2_bot.append(p[1])
            h1_s = (float(np.mean(h1_top)) - float(np.mean(h1_bot))) if h1_top and h1_bot else 0
            h2_s = (float(np.mean(h2_top)) - float(np.mean(h2_bot))) if h2_top and h2_bot else 0
            ticker_stable.append((h1_s > 0 and h2_s > 0) or (h1_s < 0 and h2_s < 0))
        half_stable = (sum(ticker_stable) / len(ticker_stable) >= 0.5) if ticker_stable else False
    else:
        # Single-ticker: split the full pair list at the chronological midpoint.
        single_sorted = sorted(pairs, key=lambda p: p[2])
        mid = n // 2
        h1_top: list = []; h1_bot: list = []
        h2_top: list = []; h2_bot: list = []
        for i, p in enumerate(single_sorted):
            b20 = bin20_lookup.get((p[3], p[2]))   # p[3] = ticker
            if not b20 or b20 <= 0:
                continue
            b10 = ((b20 - 1) // 2) + 1
            if i < mid:
                if b10 == 10: h1_top.append(p[1])
                elif b10 == 1: h1_bot.append(p[1])
            else:
                if b10 == 10: h2_top.append(p[1])
                elif b10 == 1: h2_bot.append(p[1])
        h1_spread = (float(np.mean(h1_top)) - float(np.mean(h1_bot))) if h1_top and h1_bot else 0
        h2_spread = (float(np.mean(h2_top)) - float(np.mean(h2_bot))) if h2_top and h2_bot else 0
        half_stable = (h1_spread > 0 and h2_spread > 0) or (h1_spread < 0 and h2_spread < 0)

    # ── Concentration risk ────────────────────────────────────────────────
    yearly_spreads = {yc["year"]: yc["top_avg"] - yc["bot_avg"] for yc in yearly_consistency}
    total_abs = sum(abs(v) for v in yearly_spreads.values())
    concentration = round(max(abs(v) for v in yearly_spreads.values()) / total_abs, 4) if total_abs > 0 else 1.0

    # ── Composite score ───────────────────────────────────────────────────
    best_sharpe = max(abs(d["sharpe"]) for d in decile_stats if d) if decile_stats else 0
    c_rank    = min(abs(float(sr)) / 0.20, 1.0)
    c_mono    = monotonicity
    c_consist = (consistency_pct / 100.0) if consistency_pct else 0
    c_half    = 1.0 if half_stable else 0
    c_conc    = max(0, 1.0 - concentration)
    c_sharpe  = min(best_sharpe / 0.5, 1.0)
    c_sample  = min(n / 1000, 0.5)
    composite = round((c_rank + c_mono + c_consist + c_half + c_conc + c_sharpe + c_sample) / 6.5 * 100, 1)
    _tlog('common half_sample + composite')

    # ── Rolling IC + sign-stability (Steps IC.2 + IC.3) ───────────────────
    # Routes through `ic_compute` primitives. The rolling-IC series is always
    # FULL HISTORY (mode-independent), built from the raw `row_dicts`. Only
    # the *reference IC* used for sign-classification depends on the spec:
    #   in_sample / walk_forward → reference = mean of all windows
    #   train_test               → reference = mean of pre-cutoff windows
    #
    # Two different computations under the same payload shape:
    #   single-ticker (is_all=False) — rolling Spearman of one ticker's
    #     (metric, fwd_ret) over a 252-day trailing window (IC.2).
    #   ALL (is_all=True)            — per-day cross-sectional Spearman
    #     across all tickers, then a 252-day trailing mean of that daily
    #     series (IC.3). The series-shape contract is identical, so the
    #     frontend renders both with the same code; only ε differs (much
    #     tighter in cross-sectional because K-1 effective degrees of
    #     freedom multiply with W/horizon).
    from app.routers.ic_compute import (
        rolling_ic_single_ticker, rolling_ic_cross_sectional,
        classified_rolling_ic, sign_stability_from_rolling,
        noise_floor_epsilon, _horizon_from_outcome,
    )

    _IC_WINDOW = 252
    rolling_ic_payload: Optional[dict] = None
    if row_dicts:
        horizon = _horizon_from_outcome(outcome)

        if is_all:
            ic_series = rolling_ic_cross_sectional(
                row_dicts, metric, outcome, window=_IC_WINDOW,
            )
            # ε for cross-sectional needs a K (cross-section size). Use the
            # median n across rolled points (each IcPoint.n is the median
            # per-day cross-section size in its window) — stable across
            # the chart and faithfully reports the typical K.
            median_k = int(np.median([p.n for p in ic_series])) if ic_series else 0
            epsilon = noise_floor_epsilon(
                "cross_sectional", window=_IC_WINDOW, horizon=horizon,
                k_tickers=median_k,
            )
            ic_mode = "cross_sectional"
        else:
            ic_series = rolling_ic_single_ticker(
                row_dicts, metric, outcome, window=_IC_WINDOW,
            )
            epsilon = noise_floor_epsilon(
                "single_ticker", window=_IC_WINDOW, horizon=horizon,
            )
            ic_mode = "single_ticker"

        if spec.kind == "train_test":
            cutoff_s = spec.cutoff.isoformat()
            pre_cutoff_ics = [p.ic for p in ic_series if str(p.date) < cutoff_s]
            reference_ic = float(np.mean(pre_cutoff_ics)) if pre_cutoff_ics else 0.0
        elif ic_series:
            reference_ic = float(np.mean([p.ic for p in ic_series]))
        else:
            reference_ic = 0.0

        # Short-window (21d ≈ 1 month) rolling IC for the regime-context overlay.
        # Not sign-classified, no epsilon computed — context only.
        # 5d was too small: single-ticker Spearman over 5 obs has huge variance
        # and produces artifact spikes; 21d gives enough sample to show genuine
        # regime shifts without the statistical noise.
        _IC_SHORT_WINDOW = 21
        if is_all:
            short_ic_series = rolling_ic_cross_sectional(
                row_dicts, metric, outcome, window=_IC_SHORT_WINDOW,
            )
        else:
            short_ic_series = rolling_ic_single_ticker(
                row_dicts, metric, outcome, window=_IC_SHORT_WINDOW,
            )

        classified = classified_rolling_ic(ic_series, reference_ic, epsilon)
        stability = sign_stability_from_rolling(ic_series, reference_ic, epsilon)

        rolling_ic_payload = {
            "series": [
                {"date": str(p.date), "ic": p.ic, "n": p.n,
                 "sign_class": p.sign_class}
                for p in classified
            ],
            "short_series":  [{"date": str(p.date), "ic": p.ic}
                               for p in short_ic_series],
            "short_window":  _IC_SHORT_WINDOW,
            "reference_ic":  round(reference_ic, 6),
            "epsilon":       round(epsilon, 6),
            "window":        _IC_WINDOW,
            "horizon":       horizon,
            "mode":          spec.kind,
            "ic_mode":       ic_mode,
            "cutoff_date":   (spec.cutoff.isoformat()
                              if spec.kind == "train_test" else None),
            "sign_stability": {
                "stability": (round(stability.stability, 4)
                              if stability.stability is not None else None),
                "n_same":     stability.n_same,
                "n_opposite": stability.n_opposite,
                "n_neutral":  stability.n_neutral,
                "n_total":    stability.n_total,
                "suppressed": stability.suppressed,
                "suppression_reason": stability.suppression_reason,
            },
        }
    _tlog('common rolling_ic (252d + 21d)')

    # ── Slim trade_calendar + server-side aggregated stats (W1) ──────────────
    # trade_calendar ships {date, ret, ticker, decile20} only — heavy per-row
    # fields are served lazily via GET /trades.  DOW / monthly / yearly /
    # activity stats are pre-aggregated by (date20) bucket here so the
    # frontend can filter by selectedBins20 without a large payload.
    spot_by_date = {s["date"]: s["value"] for s in spot_series} if spot_series else {}
    all_spot_date_idx = {d: i for i, d in enumerate(all_spot_dates)} if all_spot_dates else {}

    _dow_acc: dict = defaultdict(lambda: defaultdict(list))  # [dow][dec20] → [ret]
    _mo_acc:  dict = defaultdict(lambda: defaultdict(list))  # [(yr,mo)][dec20] → [ret]
    _yr_acc:  dict = defaultdict(lambda: defaultdict(list))  # [yr][dec20] → [ret]
    _act_acc: dict = defaultdict(lambda: defaultdict(int))   # [date][dec20] → entry_count
    _trade_details: list = []  # full detail per pair; stored in _TRADE_CACHE for /trades

    trade_calendar = []
    for idx, pair in enumerate(pairs):
        x, y, d = pair[0], pair[1], pair[2]
        tkr  = pair[3] if len(pair) > 3 else ticker
        yr   = d.year      if hasattr(d, 'year')     else int(str(d)[:4])
        mo   = d.month     if hasattr(d, 'month')    else int(str(d)[5:7])
        dow  = d.weekday() if hasattr(d, 'weekday')  else 0
        date_str = str(d.date() if hasattr(d, 'date') else d)
        dec   = pairs_decile[idx]
        dec20 = (pairs_decile20[idx] or None) if pairs_decile20 else None

        # Slim calendar entry — used for equity curve (needs date+ret+decile20)
        # and secondary scanner filtered_dates (needs ticker+date+decile20).
        entry = {"date": date_str, "ret": round(y, 6), "ticker": tkr}
        if dec20 is not None:
            entry["decile20"] = dec20
        trade_calendar.append(entry)

        # Accumulate into aggregated stats (only when dec20 is known)
        if dec20 is not None:
            _dow_acc[dow][dec20].append(y)
            _mo_acc[(yr, mo)][dec20].append(y)
            _yr_acc[yr][dec20].append(y)
            _act_acc[date_str][dec20] += 1

        # Full detail record for /trades endpoint
        detail: dict = {
            "date": date_str, "ticker": tkr,
            "metric_val": round(x, 6), "ret": round(y, 6),
            "decile": dec, "decile20": dec20,
        }
        if is_all:
            eo = all_open_by_tkr_date.get((tkr, date_str))
            if eo is not None:
                detail["spot_entry"] = eo
            tkr_idx   = all_date_idx_by_tkr.get(tkr, {})
            tkr_dates = all_dates_list_by_tkr.get(tkr, [])
            if date_str in tkr_idx:
                ei = tkr_idx[date_str] + max(horizon - 1, 0)
                if ei < len(tkr_dates):
                    ed = tkr_dates[ei]
                    detail["exit_date"] = ed
                    ec = all_close_by_tkr.get(tkr, {}).get(ed)
                    if ec is not None:
                        detail["spot_exit"] = ec
        else:
            if open_by_date and date_str in open_by_date:
                detail["spot_entry"] = open_by_date[date_str]
            elif spot_by_date and date_str in spot_by_date:
                detail["spot_entry"] = spot_by_date[date_str]
            if all_spot_date_idx and date_str in all_spot_date_idx:
                ei = all_spot_date_idx[date_str] + max(horizon - 1, 0)
                if ei < len(all_spot_dates):
                    exit_date_str = all_spot_dates[ei]
                    detail["exit_date"] = exit_date_str
                    if exit_date_str in close_by_date:
                        detail["spot_exit"] = close_by_date[exit_date_str]
        _trade_details.append(detail)

    # Build server-side aggregated stats
    def _agg(rets):
        n = len(rets)
        if not n:
            return None
        avg = float(np.mean(rets))
        wr  = float(np.mean([1.0 if v > 0 else 0.0 for v in rets]))
        return n, round(avg, 6), round(wr, 4)

    dow_stats: list = []
    for _dv, _dm in sorted(_dow_acc.items()):
        for _dc, _rets in sorted(_dm.items()):
            r = _agg(_rets)
            if r:
                dow_stats.append({"dow": _dv, "decile20": _dc,
                                   "n": r[0], "avg_ret": r[1], "win_rate": r[2]})

    monthly_stats: list = []
    for (_yr2, _mo2), _dm in sorted(_mo_acc.items()):
        for _dc, _rets in sorted(_dm.items()):
            r = _agg(_rets)
            if r:
                monthly_stats.append({"year": _yr2, "month": _mo2, "decile20": _dc,
                                       "n": r[0], "avg_ret": r[1]})

    yearly_stats: list = []
    for _yr2, _dm in sorted(_yr_acc.items()):
        for _dc, _rets in sorted(_dm.items()):
            r = _agg(_rets)
            if r:
                yearly_stats.append({"year": _yr2, "decile20": _dc,
                                      "n": r[0], "avg_ret": r[1], "win_rate": r[2]})

    activity_by_date: list = [
        {"date": _dt, "decile20": _dc, "n": _cnt}
        for _dt, _dm in sorted(_act_acc.items())
        for _dc, _cnt in sorted(_dm.items())
    ]

    # Store full trade details for /trades endpoint
    _tc_key = (f"{ticker}:{metric}:{outcome}:{spec.kind}:"
               f"{spec.cutoff.isoformat() if spec.kind == 'train_test' else ''}")
    if len(_TRADE_CACHE) >= _TRADE_CACHE_MAX:
        del _TRADE_CACHE[next(iter(_TRADE_CACHE))]
    _TRADE_CACHE[_tc_key] = _trade_details

    _tlog(f'common trade_cal+stats ({len(trade_calendar)} trades, '
          f'{len(dow_stats)} dow, {len(yearly_stats)} yr, {len(activity_by_date)} act rows)')

    # ── Today's value (single-ticker only) ───────────────────────────────
    # today_decile is the bin the Assigner already computed for the latest
    # row (pairs_decile[-1]); using it keeps the displayed decile consistent
    # with trade_calendar[-1].decile and mode-aware automatically.
    today_val = today_pct = today_decile = None
    if not is_all and pairs:
        today_val = pairs[-1][0]
        all_x = sorted(p[0] for p in pairs)
        today_pct = round(sum(1 for v in all_x if v <= today_val) / len(all_x) * 100, 1)
        today_decile = pairs_decile[-1] if pairs_decile else None

    _tlog('DONE')
    _result = {
        "ticker":   ticker,
        "metric":   metric,
        "outcome":  outcome,
        "n":        n,
        "horizon":  horizon,
        "all_mode": is_all,
        "n_tickers": n_tickers_used,

        # Stats
        "pearson_r":          round(float(pr), 4),
        "pearson_p":          round(float(pp), 6),
        "spearman_r":         round(float(sr), 4),
        "monotonicity":       monotonicity,
        "pattern":            pattern,
        "composite_score":    composite,
        "consistency_pct":    consistency_pct,
        "concentration_risk": concentration,
        "half_sample_stable": bool(half_stable),

        # Decile data
        "decile_stats":     decile_stats,
        "decile_stats_20":  decile_stats_20,
        "equity_by_decile": equity_by_decile,

        # Time series
        "yearly":             yearly,
        "yearly_consistency": yearly_consistency,
        "rolling_ic":         rolling_ic_payload,
        "trade_calendar":     trade_calendar,
        "dow_stats":          dow_stats,
        "monthly_stats":      monthly_stats,
        "yearly_stats":       yearly_stats,
        "activity_by_date":   activity_by_date,
        "spot_series":        spot_series,

        # Today (null in ALL mode)
        "today_value":      round(float(today_val), 6) if today_val is not None else None,
        "today_percentile": today_pct,
        "today_decile":     today_decile,
        "latest_date":      str(pairs[-1][2]) if pairs else None,

        # Mode-aware metadata. Frontend reads `mode` and the matching
        # mode-specific field (warmup for walk_forward, cutoff_date for
        # train_test) to render the subtitle on the primary chart.
        "mode":             spec.kind,
        "warmup":           spec.warmup if spec.kind == "walk_forward" else None,
        "cutoff_date":      spec.cutoff.isoformat() if spec.kind == "train_test" else None,
        "start_date":       str(pairs[0][2]) if pairs else None,

        # Diagnostic: handler elapsed time (ms) measured at the return statement.
        # compare with client `server+1stbyte` to isolate FastAPI serialization cost.
        # Remove when W1 ships and the measurement confirms serialization collapsed.
        "_handler_ms":      round((_time.perf_counter() - _t0) * 1000),

        # W2: max trade_date in the fetched rows — used as the staleness token
        # for the response cache.  Also surfaced to the client for display.
        "data_as_of":       _max_trade_date,
    }
    # W2: populate in-memory hot tier (evict oldest if at cap)
    _tlog(f'W2 write key="{_ac_key}" data_as_of="{_max_trade_date}"')
    if len(_ANALYZE_CACHE) >= _ANALYZE_CACHE_MAX:
        del _ANALYZE_CACHE[next(iter(_ANALYZE_CACHE))]
    _ANALYZE_CACHE[_ac_key] = _result
    # W2: persist to DB in background (non-blocking; failure is non-fatal)
    _cutoff_for_db = _cutoff_s or None
    asyncio.create_task(_write_analyze_primary_cache(
        pool, _pc_key, ticker, metric, outcome, spec.kind,
        _cutoff_for_db, date_from, date_to, _result,
    ))
    return _result


# ── Trade-detail endpoints (W1) ───────────────────────────────────────────────
# /trades  — paginated full trade list from the most-recent /analyze cache.
# /trades/csv — streaming CSV download of the same data.

@router.get("/trades")
async def get_trades(
    ticker:      str            = Query(...),
    metric:      str            = Query(...),
    outcome:     str            = Query(...),
    mode:        str            = Query("walk_forward"),
    cutoff_date: Optional[str]  = Query(None),
    decile20:    Optional[List[int]] = Query(None),
    sort_key:    str            = Query("date"),
    sort_dir:    str            = Query("desc"),
    page:        int            = Query(0),
    page_size:   int            = Query(250),
):
    """Return a paginated list of full trade details from the in-memory cache
    populated by /analyze.  Returns {trades, total, page, page_size} or
    {error:'not_cached'} if no matching analysis has been run yet."""
    _tc_key = f"{ticker}:{metric}:{outcome}:{mode}:{cutoff_date or ''}"
    trades = _TRADE_CACHE.get(_tc_key)
    if trades is None:
        return {"trades": [], "total": 0, "page": page, "page_size": page_size,
                "error": "not_cached"}

    # Apply decile20 filter
    if decile20:
        dec_set = set(decile20)
        filtered: list = [t for t in trades if t.get("decile20") in dec_set]
    else:
        filtered = list(trades)

    # Sort
    _str_keys = {"date", "ticker", "exit_date"}
    _desc = (sort_dir != "asc")

    def _sort_val(t):
        if sort_key == "bin":
            v = t.get("decile20") or t.get("decile") or 0
        else:
            v = t.get(sort_key)
        if v is None:
            return "" if sort_key in _str_keys else float("-inf")
        return v

    try:
        filtered.sort(key=_sort_val, reverse=_desc)
    except TypeError:
        pass  # mixed types — leave unsorted

    total = len(filtered)
    start = page * page_size
    return {
        "trades":    filtered[start:start + page_size],
        "total":     total,
        "page":      page,
        "page_size": page_size,
    }


@router.get("/trades/csv")
async def get_trades_csv(
    ticker:      str            = Query(...),
    metric:      str            = Query(...),
    outcome:     str            = Query(...),
    mode:        str            = Query("walk_forward"),
    cutoff_date: Optional[str]  = Query(None),
    decile20:    Optional[List[int]] = Query(None),
):
    """Stream all matching trades as a CSV file attachment."""
    import csv, io
    from fastapi.responses import Response
    from datetime import date as _date

    _tc_key = f"{ticker}:{metric}:{outcome}:{mode}:{cutoff_date or ''}"
    trades = _TRADE_CACHE.get(_tc_key)
    if trades is None:
        return Response(content=b"not_cached - run /analyze first",
                        status_code=404, media_type="text/plain")

    if decile20:
        dec_set = set(decile20)
        filtered: list = [t for t in trades if t.get("decile20") in dec_set]
    else:
        filtered = list(trades)

    filtered.sort(key=lambda t: t.get("date", ""))

    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["date", "ticker", metric or "metric", "spot_entry", "spot_exit",
                     "ret_pct", "exit_date", "bin20"])
    for t in filtered:
        writer.writerow([
            t.get("date", ""),
            t.get("ticker", ""),
            t.get("metric_val", ""),
            t.get("spot_entry", ""),
            t.get("spot_exit", ""),
            f"{t.get('ret', 0) * 100:.6f}",
            t.get("exit_date", ""),
            t.get("decile20") or t.get("decile") or "",
        ])

    fname = f"trades_{ticker}_{metric}_{_date.today().isoformat()}.csv"
    return Response(
        content=buf.getvalue().encode("utf-8"),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{fname}"'},
    )


# ── Synthetic outcomes ───────────────────────────────────────────────────
# Some "outcomes" don't correspond to a single column in daily_features;
# they're computed from multiple columns per-row. Add new entries here +
# wire the per-row compute in _outcome_value.

_OVERNIGHT_GAP_OUTCOME = "overnight_gap"
_OVERNIGHT_GAP_COLS    = ("ret_1d_fwd_cc", "ret_1d_fwd_oc")


def _outcome_select_cols(outcome: str) -> list[str]:
    """SQL columns to SELECT for this outcome. Real columns return
    `[outcome]`; the synthetic overnight_gap returns its two component
    columns (the per-row cc − oc gap is computed after fetch)."""
    if outcome == _OVERNIGHT_GAP_OUTCOME:
        return list(_OVERNIGHT_GAP_COLS)
    return [outcome]


def _outcome_where_clause(outcome: str) -> str:
    """`<col1> IS NOT NULL AND <col2> IS NOT NULL ...` — every component
    column must be present for the row to be valid."""
    return " AND ".join(f"{c} IS NOT NULL" for c in _outcome_select_cols(outcome))


def _outcome_value(row, outcome: str) -> Optional[float]:
    """Resolve the outcome value for a single row (asyncpg Record or dict).
    Returns None if any required component is missing or non-numeric.
    For overnight_gap, returns float(cc) − float(oc) per the user spec:
    the gap is the per-trade mean of (ret_1d_fwd_cc − ret_1d_fwd_oc),
    NOT the difference of two separately-aggregated heatmaps."""
    if outcome == _OVERNIGHT_GAP_OUTCOME:
        cc = row["ret_1d_fwd_cc"] if "ret_1d_fwd_cc" in row else None
        oc = row["ret_1d_fwd_oc"] if "ret_1d_fwd_oc" in row else None
        if cc is None or oc is None:
            return None
        try:
            return float(cc) - float(oc)
        except (TypeError, ValueError):
            return None
    v = row[outcome] if outcome in row else None
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


@router.get("/heatmap")
async def heatmap_2d(
    ticker: str = Query(...),
    metric_x: str = Query(...),
    metric_y: str = Query(...),
    outcome: str = Query(...),
    bins: int = Query(5, ge=3, le=20),
    date_from: Optional[str] = Query(None),
    date_to: Optional[str] = Query(None),
    walk_forward: bool = Query(False),
    cutoff_date: Optional[str] = Query(None),
    pool=Depends(get_oi_pool),
):
    """2D heatmap: bin metric_x and metric_y, show avg outcome in each cell.

    ALL mode applies per-ticker independent quantile binning on each axis
    then pools, matching the main quantile chart's methodology. Each
    ticker contributes evenly to every cell rather than the universe's
    absolute-range membership being dominated by tickers with naturally
    extreme magnitudes. ALL-mode bucketing flows through the row_compute
    Assigner layer, so `walk_forward=true` and (in Step 6) train-test
    are first-class modes.

    All tickers (ALL and single) use stored-bin rank bins (B1..BN labels)
    for all three modes (in_sample, walk_forward, train_test).
    """
    if not pool:
        return {"error": "OI database not configured"}

    is_all = (ticker == "ALL")

    date_conditions = ""
    params: list = []
    p = 1
    if not is_all:
        date_conditions += f" AND ticker = ${p}"; params.append(ticker); p += 1
    if date_from:
        from datetime import date as _date
        date_conditions += f" AND trade_date >= ${p}"; params.append(_date.fromisoformat(date_from)); p += 1
    if date_to:
        from datetime import date as _date
        date_conditions += f" AND trade_date <= ${p}"; params.append(_date.fromisoformat(date_to)); p += 1

    # SELECT trade_date too so the row_compute Assigner can populate the
    # RowAssignment.trade_date field (used downstream for the WALK-FORWARD
    # subtitle's start_date — when wired in Step 6).
    # Outcome may be a synthetic compound (e.g., overnight_gap = cc - oc);
    # _outcome_select_cols expands it to the component column list and
    # _outcome_where_clause builds the NOT-NULL guard for each component.
    outcome_cols    = _outcome_select_cols(outcome)
    outcome_sql_sel = ", ".join(outcome_cols)
    outcome_sql_nn  = _outcome_where_clause(outcome)
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            f"SELECT ticker, trade_date, {metric_x}, {metric_y}, {outcome_sql_sel} FROM daily_features "
            f"WHERE {metric_x} IS NOT NULL AND {metric_y} IS NOT NULL AND {outcome_sql_nn}"
            f"{date_conditions} ORDER BY trade_date",
            *params)

    from app.routers.row_compute import make_spec
    spec = make_spec(walk_forward, cutoff_date)
    # Stored-bin path: all tickers (ALL and single) use the appropriate
    # bin table (is_bins / wf_bins / tt_bins) for all three modes.
    if spec.kind in {"in_sample", "walk_forward", "train_test"}:
            # Group 7: WF+ALL joins wf_bins. Group 8: TT+ALL joins
            # tt_bins. Encoding A is uniform across all three: bin20 > 0
            # on both axes drops the appropriate sentinel rows.
            # TT additionally splits the rendered cell aggregations into
            # a train grid (trade_date < cutoff) and a test grid
            # (>= cutoff). Both grids read the same stored bin20 — that's
            # the frozen-ruler property — only the row population each
            # cell aggregates over differs.
            bin_table = {
                "in_sample":   "is_bins",
                "walk_forward": "wf_bins",
                "train_test":  "tt_bins",
            }[spec.kind]
            # TT pulls trade_date too so the post-fetch loop can split
            # into train vs test windows. IS/WF don't need it (single
            # grid).
            tt_extra_cols = (
                ", df.trade_date AS df_trade_date "
                if spec.kind == "train_test" else " "
            )
            join_sql = (
                f"SELECT df.ticker, "
                f"bt.bin20_{metric_x} AS bin_x_20, "
                f"bt.bin20_{metric_y} AS bin_y_20, "
                f"{outcome_sql_sel}{tt_extra_cols}"
                f"FROM daily_features df "
                f"JOIN {bin_table} bt USING (ticker, trade_date) "
                f"WHERE bt.bin20_{metric_x} > 0 "
                f"AND bt.bin20_{metric_y} > 0 "
                f"AND {outcome_sql_nn}"
                f"{date_conditions} "
                f"ORDER BY df.trade_date"
            )
            async with pool.acquire() as conn:
                ib_rows = await conn.fetch(join_sql, *params)

            cell_rets_is: list = [[[] for _ in range(bins)] for _ in range(bins)]
            # TT-only: separate train-window cell accumulator. For IS/WF
            # this stays empty and unused.
            cell_rets_is_train: list = [[[] for _ in range(bins)] for _ in range(bins)]
            n_tickers_with_bins_is: set = set()
            total_n_is = 0
            cutoff_for_split = spec.cutoff if spec.kind == "train_test" else None
            for r in ib_rows:
                ov = _outcome_value(r, outcome)
                if ov is None:
                    continue
                try:
                    fov = float(ov)
                    if math.isnan(fov):
                        continue
                except (TypeError, ValueError):
                    continue
                bx20 = r["bin_x_20"]
                by20 = r["bin_y_20"]
                if bins == 20:
                    ix = bx20 - 1
                    iy = by20 - 1
                else:
                    ix = ((bx20 - 1) * bins) // 20
                    iy = ((by20 - 1) * bins) // 20
                # TT: split into train (pre-cutoff) and test (post-cutoff).
                # Both grids see the SAME stored bin20 — the frozen ruler.
                if cutoff_for_split is not None:
                    td = r["df_trade_date"]
                    if td < cutoff_for_split:
                        cell_rets_is_train[iy][ix].append(fov)
                    else:
                        cell_rets_is[iy][ix].append(fov)
                else:
                    cell_rets_is[iy][ix].append(fov)
                total_n_is += 1
                n_tickers_with_bins_is.add(r["ticker"])
            n_tickers_used_is = len(n_tickers_with_bins_is)

            if total_n_is < 50:
                return {"error": f"Insufficient data after {bin_table} filter: {total_n_is} rows"}

            def _build_grid_is(cell_data):
                # No `len(rets) >= 5` cell suppression. Every cell with at
                # least one row emits its avg_ret + win_rate + n. The
                # frontend renders a hatched-gray background for cells
                # below a user-controlled n-threshold (default 50, via
                # slider) AND excludes them from the gradient-scale
                # computation. Cells with n=0 stay as None — truly empty.
                g = []
                for iy_ in range(bins):
                    row_out = []
                    for ix_ in range(bins):
                        rets = cell_data[iy_][ix_]
                        if rets:
                            a = np.array(rets)
                            row_out.append({
                                "avg_ret":  round(float(a.mean()), 6),
                                "win_rate": round(float((a > 0).mean()), 4),
                                "n":        int(len(rets)),
                            })
                        else:
                            row_out.append(None)
                    g.append(row_out)
                return g

            grid_is = _build_grid_is(cell_rets_is)
            x_labels_is = [f"B{i+1}" for i in range(bins)]
            y_labels_is = [f"B{j+1}" for j in range(bins)]
            resp = {
                "metric_x":   metric_x, "metric_y": metric_y, "outcome": outcome,
                "bins":       bins, "n": total_n_is,
                "x_labels":   x_labels_is, "y_labels": y_labels_is,
                "grid":       grid_is,
                "per_ticker": True,
                "n_tickers":  n_tickers_used_is,
                "mode":       spec.kind,
                "source":     bin_table,
            }
            if spec.kind == "train_test":
                # Train side rendered alongside the test side from the
                # same stored ruler — the two-heatmap display.
                resp["train_grid"] = _build_grid_is(cell_rets_is_train)
                resp["test_grid"]  = grid_is
                resp["cutoff_date"] = spec.cutoff.isoformat()
            return resp




_METRIC_BINS_CANONICAL = 20


@router.get("/metric-bins")
async def metric_bins_1d(
    ticker: str = Query(...),
    metric: str = Query(...),
    outcome: str = Query(...),
    bins: int = Query(10, ge=2, le=20),
    date_from: Optional[str] = Query(None),
    date_to: Optional[str] = Query(None),
    walk_forward: bool = Query(False),
    cutoff_date: Optional[str] = Query(None),
    pool=Depends(get_oi_pool),
):
    """N-bin decile stats for one metric vs one outcome (lightweight version of /analyze).

    ALL mode uses per-ticker independent quantile binning then pools,
    matching the main quantile chart's methodology. Single-ticker mode
    uses flat rank-based binning. Both modes flow through the
    row_compute Assigner — `walk_forward=true` and (Step 6) train-test
    are first-class modes.

    Canonical 20-bin internal: regardless of the requested `bins`, the
    Assigner is invoked at n_bins=20 and the result is aggregated to the
    requested display granularity via trade-count weighting. This locks
    the endpoint to the same ticker-row-count exclusion threshold as the
    analyze_cache bundle (which always bins at 20). Pre-canonicalization,
    requesting bins=10 here included tickers with 10-19 rows that the
    bundle excluded at the 20-row threshold — producing a systematic
    ~8 bps gap between the heatmap sidebar and the main quantile pane.
    Non-divisor `bins` (e.g. 3, 7) fall back to the legacy direct
    Assigner call so they still work for /global-metric-bins or other
    callers that pass arbitrary granularities.
    """
    if not pool:
        return {"error": "OI database not configured"}
    is_all = (ticker == "ALL")
    bins = max(2, min(20, bins))
    date_conditions = ""
    params: list = []
    p = 1
    if not is_all:
        date_conditions += f" AND ticker = ${p}"; params.append(ticker); p += 1
    if date_from:
        from datetime import date as _date
        date_conditions += f" AND trade_date >= ${p}"; params.append(_date.fromisoformat(date_from)); p += 1
    if date_to:
        from datetime import date as _date
        date_conditions += f" AND trade_date <= ${p}"; params.append(_date.fromisoformat(date_to)); p += 1

    # SELECT trade_date too so the Assigner can populate RowAssignment.trade_date
    # (WalkForwardAssigner sorts per-ticker pairs by trade_date — empty
    # strings would still work via stable sort + chronological SQL order,
    # but explicit dates are more robust against schema changes).
    # Outcome may be synthetic (overnight_gap = cc - oc); _outcome_select_cols
    # expands to the component columns and _outcome_where_clause guards them.
    outcome_cols    = _outcome_select_cols(outcome)
    outcome_sql_sel = ", ".join(outcome_cols)
    outcome_sql_nn  = _outcome_where_clause(outcome)

    # ── v10 Group 2 — ALL+IS reads bin from is_bins ──────────────────────
    # Short-circuited BEFORE the daily_features fetch below so the IS+ALL
    # path skips that round-trip entirely. Narrow read: SELECT only the
    # one bin column needed (bin20_<metric>) plus the metric value and
    # outcome via JOIN. JOIN preserves outcome-validity — Group 1
    # verification noted is_bins contains recent rows that have a bin
    # but no forward return yet, so the outcome NOT NULL filter is
    # mandatory or counts inflate.
    #
    # WF / TT / single-ticker flow through the existing Assigner path
    # below. Same mode boundary as Group 1 (/heatmap).
    #
    # Display granularity: is_bins stores bin20. Aggregate to user's
    # `bins` via `((bin20 - 1) * bins) // 20`. Divisor bins (2/4/5/10/20)
    # are mathematically exact; non-divisor bins shift slightly. Default
    # bins=10 (the heatmap-sidebar caller's request) is exact.
    # Stored-bin path — all tickers (ALL and single) for all modes.
    if True:
        # WF → wf_bins, TT → tt_bins, IS → is_bins. Same math and response
        # shape regardless of ticker; date_conditions already includes the
        # per-ticker filter for single-ticker requests.
        if cutoff_date:
            bin_table = "tt_bins"
        elif walk_forward:
            bin_table = "wf_bins"
        else:
            bin_table = "is_bins"
        # Group 8 fix: TT is test-window-only — filter rows to
        # trade_date >= cutoff. Same fix that's applied to /analyze's
        # primary quantile in TT mode. Without this the bars sum to
        # the full universe while the subtitle says "TEST PERIOD".
        tt_extra_where = ""
        tt_extra_params: list = []
        if cutoff_date:
            from datetime import date as _date_cls_tt
            tt_extra_where = f" AND df.trade_date >= ${len(params) + 1}"
            tt_extra_params = [_date_cls_tt.fromisoformat(cutoff_date)]
        join_sql = (
            f"SELECT df.{metric} AS metric_val, "
            f"bt.bin20_{metric} AS bin_20, "
            f"{outcome_sql_sel} "
            f"FROM daily_features df "
            f"JOIN {bin_table} bt USING (ticker, trade_date) "
            f"WHERE bt.bin20_{metric} > 0 "
            f"AND {outcome_sql_nn}"
            f"{date_conditions}{tt_extra_where}"
        )
        try:
            async with pool.acquire() as conn:
                ib_rows = await conn.fetch(join_sql, *params, *tt_extra_params)
        except Exception:
            ib_rows = []   # column bin20_{metric} absent (null-by-design metric)

        # Slot each row into its display bin (bin20 → bins via the
        # canonical aggregation formula). Per-bucket lists carry (metric
        # value, outcome) so std_dev / sharpe / min_val / max_val match
        # the on-the-fly response shape.
        buckets_data: list = [[] for _ in range(bins)]
        total_n = 0
        for r in ib_rows:
            ov = _outcome_value(r, outcome)
            if ov is None:
                continue
            try:
                fov = float(ov)
                if math.isnan(fov):
                    continue
            except (TypeError, ValueError):
                continue
            try:
                fv = float(r["metric_val"])
                if math.isnan(fv):
                    continue
            except (TypeError, ValueError):
                continue
            bv20 = r["bin_20"]
            bin_disp = ((bv20 - 1) * bins) // 20
            buckets_data[bin_disp].append((fv, fov))
            total_n += 1

        # No `total_n < 20` panel-block. The All-Ticker Metric Bins chart
        # renders against whatever survives the is_bins filter; empty
        # buckets show as None and the populated ones show their stats.
        # User reads the n directly off the bars.

        result = []
        for i in range(bins):
            bucket = buckets_data[i]
            if not bucket:
                result.append(None)
                continue
            ys = np.array([p[1] for p in bucket])
            xs = [p[0] for p in bucket]
            result.append({
                "bucket":   i + 1,
                "n":        len(bucket),
                "avg_ret":  round(float(ys.mean()), 6),
                "win_rate": round(float((ys > 0).mean()), 4),
                "std_dev":  round(float(ys.std()), 6),
                "sharpe":   round(float(ys.mean() / ys.std()), 4) if ys.std() > 0 else 0,
                "min_val":  round(float(min(xs)), 6),
                "max_val":  round(float(max(xs)), 6),
            })
        return {
            "metric":         metric,
            "outcome":        outcome,
            "bins":           bins,
            "canonical_bins": 20,
            "n":              total_n,
            "buckets":        result,
            "per_ticker":     True,
            "mode":           "in_sample",
            "source":         "is_bins",
        }

    # Metric absent from is_bins (null-by-design): return empty buckets.
    return {
        "metric":         metric,
        "outcome":         outcome,
        "bins":            bins,
        "canonical_bins":  20,
        "n":               0,
        "buckets":         [None] * bins,
        "per_ticker":      True,
        "mode":            "in_sample",
        "source":          "is_bins",
    }


@router.get("/ai-summary")
async def ai_summary(
    ticker: str = Query(...),
    metric: str = Query(...),
    outcome: str = Query(...),
    pool=Depends(get_oi_pool),
):
    """Generate an AI interpretation of the analysis."""
    try:
        import anthropic
    except ImportError:
        return {"summary": "(anthropic SDK not available)"}

    # Fetch the analysis first
    data = await analyze(ticker=ticker, metric=metric, outcome=outcome, pool=pool)
    if data.get("error"):
        return {"summary": f"Cannot generate: {data['error']}"}

    # Build compact context
    stats = (f"Score: {data['composite_score']}, Pattern: {data['pattern']}, "
             f"Pearson: {data['pearson_r']}, Spearman: {data['spearman_r']}, "
             f"Monotonicity: {data['monotonicity']}, Consistency: {data['consistency_pct']}%, "
             f"Concentration: {data['concentration_risk']}, "
             f"Half-stable: {data['half_sample_stable']}, N: {data['n']}")

    deciles = ""
    for d in (data.get("decile_stats") or []):
        if d:
            deciles += (f"  D{d['bucket']}: avg={d['avg_ret']*100:.3f}%, "
                        f"WR={d['win_rate']*100:.1f}%, Sharpe={d['sharpe']:.3f}, n={d['n']}\n")

    # Load knowledge rules
    from app.db import get_pool as _get_main_pool
    knowledge = ""
    try:
        from app.routers.research2 import _load_active_rules
        main_pool = _get_main_pool()
        if main_pool:
            rules = await _load_active_rules(main_pool)
            if rules:
                knowledge = "\nDOMAIN RULES:\n" + "\n".join(f"- {r}" for r in rules)
    except Exception:
        pass

    prompt = (
        f"Ticker: {ticker}, Metric: {metric}, Outcome: {outcome}\n"
        f"Stats: {stats}\n\nDecile Profile:\n{deciles}\n"
        f"Today: D{data.get('today_decile', '?')} ({data.get('today_percentile', '?')}%)"
        f"{knowledge}\n\n"
        f"Write 3-4 sentences: Is this metric tradable for this ticker? "
        f"What are the strengths and risks? What decile(s) should a trader focus on? "
        f"Be specific and cite numbers. Professional quant voice."
    )

    client = anthropic.AsyncAnthropic()
    msg = await client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=500,
        messages=[{"role": "user", "content": prompt}],
    )
    return {"summary": msg.content[0].text.strip()}


# ── Score Matrix ──────────────────────────────────────────────────────────────

@router.get("/score-matrix")
async def get_score_matrix(
    pool=Depends(get_pool),
    ticker: Optional[str] = None,
    metric: Optional[str] = None,
    fwd_ret: Optional[str] = None,
    min_score: float = 0,
    sort_by: str = "composite_score",
    order: str = "desc",
    limit: int = 500,
    mode: str = Query("in_sample"),
    cutoff_date: Optional[str] = Query(None),
):
    """Return score matrix rows with optional filters.

    For mode='train_test' the cutoff_date filter selects results for that
    specific cutoff. If omitted in train_test mode, returns rows where
    cutoff_date IS NULL — typically empty. For other modes cutoff_date
    is ignored (those rows always have NULL cutoff_date).
    """
    from research.batch_score import ensure_table
    await ensure_table(pool)

    allowed_sorts = {
        "composite_score", "ticker", "metric", "fwd_ret", "pattern",
        "spearman_r", "monotonicity", "yearly_pct", "concentration",
        "tail_spread", "n_obs", "d10_avg", "d1_avg", "d10_wr", "d1_wr", "best_sharpe",
    }
    if sort_by not in allowed_sorts:
        sort_by = "composite_score"
    direction = "DESC" if order == "desc" else "ASC"

    where = ["composite_score >= $1", "metric NOT ILIKE 'spot%'", "mode = $2"]
    params: list = [min_score, mode]
    idx = 3

    # Train-test rows are partitioned by cutoff_date. Other modes always
    # have NULL cutoff_date, so this filter does nothing for them.
    if mode == "train_test":
        if cutoff_date:
            params.append(_date.fromisoformat(cutoff_date))
            where.append(f"cutoff_date = ${idx}"); idx += 1
        else:
            where.append("cutoff_date IS NULL")
    if ticker:
        where.append(f"ticker = ${idx}"); params.append(ticker); idx += 1
    if metric:
        where.append(f"metric = ${idx}"); params.append(metric); idx += 1
    if fwd_ret:
        where.append(f"fwd_ret = ${idx}"); params.append(fwd_ret); idx += 1

    where_clause = " AND ".join(where)
    sql = f"""
        SELECT ticker, metric, fwd_ret, composite_score, pattern,
               spearman_r, monotonicity, yearly_pct, concentration,
               tail_spread, n_obs, d10_avg, d1_avg, d10_wr, d1_wr,
               best_sharpe, mi, pearson_r, loyo_fragile, scanned_at
        FROM oi_score_matrix
        WHERE {where_clause}
        ORDER BY {sort_by} {direction} NULLS LAST
        LIMIT {min(limit, 2000)}
    """

    async with pool.acquire() as conn:
        rows = await conn.fetch(sql, *params)

    # Coerce NaN floats to None so the JSON encoder doesn't reject the
    # response. The batch scorer now writes NaN as NULL, but any older
    # rows containing NaN would still poison the response otherwise.
    def _scrub(d):
        out = dict(d)
        for k, v in out.items():
            if isinstance(v, float) and math.isnan(v):
                out[k] = None
        return out
    return [_scrub(r) for r in rows]


@router.get("/score-matrix/meta")
async def score_matrix_meta(pool=Depends(get_pool),
                            mode: str = Query("in_sample"),
                            cutoff_date: Optional[str] = Query(None)):
    """Return distinct metrics, tickers, fwd_rets + summary stats for filter dropdowns.

    For mode='train_test', the cutoff_date filter scopes everything to a
    specific train-test cutoff. Other modes ignore cutoff_date (rows for
    those modes always have NULL cutoff_date).
    """
    from research.batch_score import ensure_table
    await ensure_table(pool)

    # Build the shared mode + cutoff_date filter once. Used by every query
    # below — keeps the train_test partitioning consistent across count,
    # distinct lists, and aggregate stats.
    where = "mode = $1"
    p_args: list = [mode]
    if mode == "train_test":
        if cutoff_date:
            p_args.append(_date.fromisoformat(cutoff_date))
            where += " AND cutoff_date = $2"
        else:
            where += " AND cutoff_date IS NULL"

    async with pool.acquire() as conn:
        count = await conn.fetchval(
            f"SELECT COUNT(*) FROM oi_score_matrix WHERE {where}", *p_args)
        if count == 0:
            return {"count": 0, "tickers": [], "metrics": [], "fwd_rets": [],
                    "avg_score": 0, "gte50": 0, "gte70": 0, "last_run": None,
                    "mode": mode, "cutoff_date": cutoff_date}

        tickers = [r["ticker"] for r in await conn.fetch(
            f"SELECT DISTINCT ticker FROM oi_score_matrix WHERE {where} ORDER BY ticker",
            *p_args)]
        metrics = [r["metric"] for r in await conn.fetch(
            f"SELECT DISTINCT metric FROM oi_score_matrix "
            f"WHERE {where} AND metric NOT ILIKE 'spot%' ORDER BY metric", *p_args)]
        fwd_rets = [r["fwd_ret"] for r in await conn.fetch(
            f"SELECT DISTINCT fwd_ret FROM oi_score_matrix WHERE {where} ORDER BY fwd_ret",
            *p_args)]
        stats = await conn.fetchrow(f"""
            SELECT AVG(composite_score) as avg_score,
                   COUNT(*) FILTER (WHERE composite_score >= 50) as gte50,
                   COUNT(*) FILTER (WHERE composite_score >= 70) as gte70,
                   MAX(scanned_at) as last_run
            FROM oi_score_matrix WHERE {where}
        """, *p_args)

    # Defensive: an AVG over a column containing PostgreSQL NaN returns
    # NaN, which the JSON encoder rejects ("Out of range float values
    # are not JSON compliant"). The batch scorer now coerces NaN to NULL
    # at write time, but old data may still contain NaN — guard here too.
    raw_avg = stats["avg_score"]
    if raw_avg is None or (isinstance(raw_avg, float) and math.isnan(raw_avg)):
        avg_score = 0.0
    else:
        avg_score = round(float(raw_avg), 1)
    return {
        "count":     count,
        "tickers":   tickers,
        "metrics":   metrics,
        "fwd_rets":  fwd_rets,
        "avg_score": avg_score,
        "gte50":     int(stats["gte50"] or 0),
        "gte70":     int(stats["gte70"] or 0),
        "last_run":  str(stats["last_run"])[:19] if stats["last_run"] else None,
        "mode":      mode,
        "cutoff_date": cutoff_date,
    }


@router.get("/score-matrix/summary")
async def score_matrix_summary(
    pool=Depends(get_pool),
    metric: Optional[str] = None,
    fwd_ret: Optional[str] = None,
    ticker: Optional[str] = None,
    mode: str = Query("in_sample"),
    cutoff_date: Optional[str] = Query(None),
):
    """Aggregated score stats with optional cross-filtering. For mode='train_test'
    the cutoff_date filter scopes everything to that specific cutoff."""
    from research.batch_score import ensure_table
    await ensure_table(pool)

    # Shared mode + (optional) cutoff_date filter. Used by every grouped
    # aggregation below; we pre-build the SQL fragment and the param list
    # so each query just appends its own extra filters.
    base_w: list = ["mode = $1"]
    base_p: list = [mode]
    if mode == "train_test":
        if cutoff_date:
            base_p.append(_date.fromisoformat(cutoff_date))
            base_w.append(f"cutoff_date = ${len(base_p)}")
        else:
            base_w.append("cutoff_date IS NULL")

    # All queries share a base mode filter.  Build WHERE clauses dynamically
    # so the positional param numbers stay consistent.
    async with pool.acquire() as conn:
        # By metric
        bm_w = list(base_w) + ["metric NOT ILIKE 'spot%'"]
        bm_p: list = list(base_p)
        if fwd_ret:
            bm_p.append(fwd_ret); bm_w.append(f"fwd_ret = ${len(bm_p)}")
        by_metric = await conn.fetch(f"""
            SELECT metric, AVG(composite_score) as avg_score,
                   STDDEV(composite_score) as std_score, COUNT(*) as n,
                   COUNT(*) FILTER (WHERE composite_score >= 50) as gte50,
                   MAX(composite_score) as max_score
            FROM oi_score_matrix WHERE {" AND ".join(bm_w)}
            GROUP BY metric ORDER BY AVG(composite_score) DESC
        """, *bm_p)

        # By fwd_ret
        bf_w = list(base_w)
        bf_p: list = list(base_p)
        if metric:
            bf_p.append(metric); bf_w.append(f"metric = ${len(bf_p)}")
        by_fwd = await conn.fetch(f"""
            SELECT fwd_ret, AVG(composite_score) as avg_score,
                   STDDEV(composite_score) as std_score, COUNT(*) as n,
                   COUNT(*) FILTER (WHERE composite_score >= 50) as gte50,
                   MAX(composite_score) as max_score
            FROM oi_score_matrix WHERE {" AND ".join(bf_w)}
            GROUP BY fwd_ret ORDER BY AVG(composite_score) DESC
        """, *bf_p)

        # By ticker
        bt_w = list(base_w)
        bt_p: list = list(base_p)
        if metric:
            bt_p.append(metric); bt_w.append(f"metric = ${len(bt_p)}")
        if fwd_ret:
            bt_p.append(fwd_ret); bt_w.append(f"fwd_ret = ${len(bt_p)}")
        by_ticker = await conn.fetch(f"""
            SELECT ticker, AVG(composite_score) as avg_score,
                   STDDEV(composite_score) as std_score, COUNT(*) as n,
                   COUNT(*) FILTER (WHERE composite_score >= 50) as gte50,
                   MAX(composite_score) as max_score
            FROM oi_score_matrix WHERE {" AND ".join(bt_w)}
            GROUP BY ticker ORDER BY AVG(composite_score) DESC
        """, *bt_p)

        # By fwd_ret scoped to a ticker
        tf_w = list(base_w)
        tf_p: list = list(base_p)
        if ticker:
            tf_p.append(ticker); tf_w.append(f"ticker = ${len(tf_p)}")
        if metric:
            tf_p.append(metric); tf_w.append(f"metric = ${len(tf_p)}")
        by_fwd_ticker = await conn.fetch(f"""
            SELECT fwd_ret, AVG(composite_score) as avg_score,
                   STDDEV(composite_score) as std_score, COUNT(*) as n,
                   COUNT(*) FILTER (WHERE composite_score >= 50) as gte50,
                   MAX(composite_score) as max_score
            FROM oi_score_matrix WHERE {" AND ".join(tf_w)}
            GROUP BY fwd_ret ORDER BY AVG(composite_score) DESC
        """, *tf_p)

    def _safe(v):
        """Treat NULL or NaN as 0 — AVG/MAX of NaN-containing groups
        returns NaN, which the JSON encoder rejects."""
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return 0.0
        return round(float(v), 1)

    def _row(r, key):
        return {key: r[key],
                "avg_score": _safe(r["avg_score"]),
                "std_score": _safe(r["std_score"]),
                "n": int(r["n"]), "gte50": int(r["gte50"]),
                "max_score": _safe(r["max_score"])}

    return {
        "by_metric":        [_row(r, "metric")  for r in by_metric],
        "by_fwd":           [_row(r, "fwd_ret") for r in by_fwd],
        "by_ticker":        [_row(r, "ticker")  for r in by_ticker],
        "by_fwd_ticker":    [_row(r, "fwd_ret") for r in by_fwd_ticker],
        "selected_metric":  metric,
        "selected_fwd_ret": fwd_ret,
        "selected_ticker":  ticker,
    }


class BatchScoreReq(BaseModel):
    walk_forward: bool = False
    cutoff_date: Optional[str] = None


@router.post("/run-batch-score")
async def trigger_batch_score(
    req: BatchScoreReq = Body(default_factory=BatchScoreReq),
    pool=Depends(get_pool),
    oi_pool=Depends(get_oi_pool),
):
    """Trigger a batch score run (in-sample, walk-forward, or train-test) in the background."""
    from research.batch_score import get_progress, run_batch_score
    import asyncio

    progress = get_progress()
    if progress["running"]:
        return {"status": "already_running", "message": progress["message"]}

    asyncio.get_event_loop().create_task(
        run_batch_score(oi_pool, pool, walk_forward=req.walk_forward,
                        cutoff_date=req.cutoff_date or ""))

    if req.cutoff_date:
        mode_label = f"train-test (cutoff {req.cutoff_date})"
    elif req.walk_forward:
        mode_label = "walk-forward"
    else:
        mode_label = "in-sample"
    return {"status": "started", "message": f"Batch scoring ({mode_label}) started…"}


@router.get("/batch-score-status")
async def batch_score_status():
    from research.batch_score import get_progress
    return get_progress()


@router.get("/feature-clusters")
async def feature_clusters(pool=Depends(get_pool)):
    """Compute and return feature clusters from score-vector similarity."""
    from research.interaction_scan import compute_clusters
    clusters = await compute_clusters(pool)
    return clusters


class Run2fRequest(BaseModel):
    metrics: List[str] = []


@router.post("/run-2f-scan")
async def trigger_2f_scan(
    request: Run2fRequest,
    pool=Depends(get_pool),
    oi_pool=Depends(get_oi_pool),
):
    """Trigger a 2-factor interaction scan in the background."""
    from research.interaction_scan import get_progress, run_2f_scan
    import asyncio
    progress = get_progress()
    if progress['running']:
        return {'status': 'already_running', 'message': progress['message']}
    metrics = request.metrics if len(request.metrics) >= 2 else None
    n_pairs = len(metrics) * (len(metrics) - 1) // 2 if metrics else None
    asyncio.get_event_loop().create_task(run_2f_scan(oi_pool, pool, selected_metrics=metrics))
    msg = f'{n_pairs} pair(s) queued...' if n_pairs else '2F scan started...'
    return {'status': 'started', 'message': msg}


@router.get("/2f-scan-status")
async def scan_2f_status():
    from research.interaction_scan import get_progress
    return get_progress()


@router.get("/interaction-matrix")
async def interaction_matrix(
    pool=Depends(get_pool),
    fwd_ret: Optional[str] = None,
    metrics: Optional[List[str]] = Query(None),
    min_lift: float = 0.0,
    limit: int = 100,
):
    """Ranked cross-family 2F results, aggregated across tickers."""
    from research.interaction_scan import ensure_table
    await ensure_table(pool)
    wheres = ["interaction_lift >= $1",
              "feat_a NOT ILIKE 'spot%'", "feat_b NOT ILIKE 'spot%'"]
    params: list = [min_lift]
    if fwd_ret:
        params.append(fwd_ret)
        wheres.append(f'fwd_ret = ${len(params)}')
    if metrics and len(metrics) >= 2:
        params.append(metrics)
        wheres.append(f'feat_a = ANY(${len(params)}) AND feat_b = ANY(${len(params)})')
    where_sql = 'WHERE ' + ' AND '.join(wheres)
    sql = f"""
        SELECT feat_a, feat_b, fwd_ret,
               AVG(composite_interaction_score) AS avg_score,
               AVG(interaction_lift)            AS avg_lift,
               MAX(interaction_lift)            AS max_lift,
               COUNT(DISTINCT ticker)           AS n_tickers,
               MAX(best_quad_sharpe)            AS max_quad_sharpe
        FROM oi_interaction_matrix
        {where_sql}
        GROUP BY feat_a, feat_b, fwd_ret
        ORDER BY avg_lift DESC
        LIMIT ${len(params)+1}
    """
    params.append(limit)
    async with pool.acquire() as conn:
        rows = await conn.fetch(sql, *params)
    return [dict(r) for r in rows]


@router.get("/interaction-detail")
async def interaction_detail(
    pool=Depends(get_pool),
    feat_a: str = Query(...),
    feat_b: str = Query(...),
    ticker: Optional[str] = None,
    fwd_ret: Optional[str] = None,
):
    """Full quadrant detail for a specific feat_a x feat_b combo."""
    from research.interaction_scan import ensure_table
    await ensure_table(pool)
    wheres = ['feat_a = $1', 'feat_b = $2']
    params: list = [feat_a, feat_b]
    if ticker:
        params.append(ticker)
        wheres.append(f'ticker = ${len(params)}')
    if fwd_ret:
        params.append(fwd_ret)
        wheres.append(f'fwd_ret = ${len(params)}')
    where_sql = 'WHERE ' + ' AND '.join(wheres)
    sql = f"""
        SELECT ticker, fwd_ret, composite_interaction_score, interaction_lift,
               best_quadrant, best_quad_sharpe, best_quad_avg_ret, best_quad_win_rate,
               best_quad_n, r2_gain, ols_r2, n, quadrants
        FROM oi_interaction_matrix {where_sql}
        ORDER BY ticker, fwd_ret
    """
    async with pool.acquire() as conn:
        rows = await conn.fetch(sql, *params)
    result = []
    for r in rows:
        d = dict(r)
        d['quadrants'] = json.loads(d['quadrants']) if d['quadrants'] else []
        result.append(d)
    return result


# ── Secondary Signal Scanner endpoints ────────────────────────────────────────

def _sec_cache_key(ticker: str, metric: str, outcome: str, date_from: str, date_to: str) -> str:
    raw = f"{ticker}|{metric}|{outcome}|{date_from}|{date_to}"
    return hashlib.md5(raw.encode()).hexdigest()[:12]


def _parse_tkr_date_set(filtered_dates: list) -> set:
    """Convert 'ticker|date' strings (sent by the frontend) into a (ticker, date) tuple set."""
    s = set()
    for fd in filtered_dates:
        if '|' in fd:
            t, d = fd.split('|', 1)
            s.add((t, d))
        else:
            s.add(('', fd))
    return s


def _filter_by_tkr_date(rows: list, tkr_date_set: set) -> list:
    """Keep only rows whose (ticker, trade_date) is in tkr_date_set."""
    has_tickers = any(k[0] for k in tkr_date_set)
    if has_tickers:
        return [r for r in rows if (r.get("ticker", ""), r.get("trade_date", "")) in tkr_date_set]
    date_set = {k[1] for k in tkr_date_set}
    return [r for r in rows if r.get("trade_date") in date_set]


# Rows at which a bin mean is considered well-sampled for spread estimation.
# SE of spread ≈ 0.3–0.5% at n=200 for typical option return vol; calibrated
# against absolute row count (not relative to bin count) so tail-shaped metrics
# — whose extreme bins are naturally thin — are not systematically penalised.
# Revisit this constant if real tail secondaries appear dimmer than expected.
_SEC_N_REF = 200


def _sec_score_metrics(
    rows: list,
    outcome_col: str,
    feature_cols: list,
    is_all: bool = False,
    n_bins: int = 10,
    spec=None,
    all_rows=None,
    bin20_by_metric: Optional[dict] = None,
) -> list:
    """Score each secondary feature by weighted_spread × breadth.

    Binning: secondary bins are assigned on the full-universe walk-forward
    distribution (all_rows, 252-day warmup) when all_rows is supplied —
    identical to the heatmap's Y-axis assignment so leaderboard scores
    reconcile with the per-bin tooltip spreads and the heatmap cells.
    When all_rows is None the legacy within-filtered-subset WF is used.

    weighted_spread = (avg_ret[top_bin] − avg_ret[bottom_bin]) × w
      w = min(harmonic_mean(n_top, n_bottom) / _SEC_N_REF, 1.0)

    breadth (ALL mode) = fraction of qualifying tickers (≥10 rows each,
      ≥3 total) whose own top-third vs bottom-third spread agrees in sign
      with the global weighted_spread.
      Single-ticker mode: breadth = 1.0 by convention.

    score = weighted_spread × breadth.
    Requires ≥3 populated bins. Requires ≥3 qualifying tickers in ALL mode.
    Sorted descending.
    """
    if len(rows) < n_bins * 2:
        return []

    all_rets = [float(r[outcome_col]) for r in rows
                if r.get(outcome_col) is not None
                and not math.isnan(float(r[outcome_col]))]
    if not all_rets:
        return []

    from app.routers.row_compute import (
        assign_secondary_buckets, _sort_chrono, DEFAULT_WALKFWD_WARMUP,
    )

    # ── Pre-sort primary-filtered rows (reused for every feature). ────────────
    rows_sorted = _sort_chrono(rows)

    # ── Pre-sort full universe once for full-universe WF bin lookups. ─────────
    # assign_secondary_buckets WF-full path uses all_rows_sorted to avoid
    # redundant O(N log N) sort on every per-feature call.
    if all_rows is not None and spec is not None and spec.kind == "walk_forward":
        _all_sorted = _sort_chrono(all_rows)
    else:
        _all_sorted = None

    # ── Precompute per-row outcome float and ticker (ALL mode only). ──────────
    # Eliminates 4 per-row operations that are feature-independent:
    #   r.get(outcome_col), float(o), isnan check, r.get("ticker").
    # Stored as parallel lists indexed by position in rows_sorted.
    if is_all:
        _ticker_for_row: list = []
        _outcome_for_row: list = []
        for r in rows_sorted:
            _ticker_for_row.append(r.get("ticker", "_"))
            o = r.get(outcome_col)
            if o is None:
                _outcome_for_row.append(None)
            else:
                try:
                    fo = float(o)
                    _outcome_for_row.append(None if math.isnan(fo) else fo)
                except (TypeError, ValueError):
                    _outcome_for_row.append(None)

    results = []
    for feat in feature_cols:
        if is_all:
            # Build by_tkr for breadth computation reusing precomputed arrays.
            by_tkr: dict = defaultdict(list)
            for i, r in enumerate(rows_sorted):
                fo = _outcome_for_row[i]
                if fo is None:
                    continue
                v = r.get(feat)
                if v is None:
                    continue
                try:
                    fv = float(v)
                    if not math.isnan(fv):
                        by_tkr[_ticker_for_row[i]].append((fv, fo))
                except (TypeError, ValueError):
                    pass
        else:
            by_tkr = {}

        # Full-universe WF bucket assignment — same bins as drilled chart +
        # heatmap.  all_rows_sorted supplied so assign_secondary_buckets
        # skips the redundant _sort_chrono(all_rows) call on each feature.
        #
        # TODO(v9-fixed-thresholds): in_sample mode here still passes
        # all_rows=None (only WF gets the full universe), which means the
        # secondary scanner's per-feature scoring still re-ranks on the
        # primary-filtered subset for in_sample. /secondary-detail,
        # /secondary-corr-bins, and /secondary-correlation were fixed in
        # the v9 commit; this scanner-scoring path was deferred because
        # changing it shifts the feature leaderboard's ranking, which
        # the user needs to re-validate independently. Fix is identical
        # in shape: pass all_rows always; the underlying helper handles
        # the mode-dispatch correctly post-v9.
        # Group 7: per-feature wf_bin20 lookup (if available) takes the
        # stored-bin path in the helper. Mode-agnostic — the same kwarg
        # carries IS or WF lookups; the helper's hoisted stored-bin
        # check fires for either.
        feat_bin20_by_key = (bin20_by_metric or {}).get(feat)
        buckets_raw = assign_secondary_buckets(
            spec, rows_sorted, feat, n_bins, outcome_col, is_all,
            rows_presorted=True,
            all_rows=(all_rows if _all_sorted is not None else None),
            all_rows_sorted=_all_sorted,
            bin20_by_key=feat_bin20_by_key,
        )
        if buckets_raw is None:
            continue
        # Each entry in buckets_raw[i] is (metric_val, outcome, date, ticker).
        buckets = [[entry[1] for entry in b] for b in buckets_raw]
        n = sum(len(b) for b in buckets)

        # Degenerate-metric guard: require ≥3 populated bins.
        populated = [(i, b) for i, b in enumerate(buckets) if b]
        if len(populated) < 3:
            continue

        # Top and bottom bins by avg return (among all populated bins).
        bin_stats = [(i, float(np.mean(b)), b) for i, b in populated]
        top_i,    top_avg,    top_b    = max(bin_stats, key=lambda x: x[1])
        bottom_i, bottom_avg, bottom_b = min(bin_stats, key=lambda x: x[1])

        n_top    = len(top_b)
        n_bottom = len(bottom_b)
        raw_spread = top_avg - bottom_avg

        # n-weighting: harmonic mean of endpoint counts vs absolute reference.
        hmean_n = 2 * n_top * n_bottom / (n_top + n_bottom)
        w = min(hmean_n / _SEC_N_REF, 1.0)
        weighted_spread = raw_spread * w

        # Breadth: per-ticker top-third vs bottom-third direction agreement.
        # No per-ticker `len(tkr_vals) < 10` floor (never fires on dense
        # data). The `n_qualifying < 3` cross-section guard below stays —
        # you can't compute a cross-section on fewer than 3 tickers.
        if is_all and by_tkr:
            sign_global  = 1 if weighted_spread >= 0 else -1
            n_qualifying = 0
            n_agreeing   = 0
            for tkr_vals in by_tkr.values():
                n_qualifying += 1
                sorted_tkr = sorted(tkr_vals, key=lambda x: x[0])
                t     = len(sorted_tkr)
                third = max(1, t // 3)
                bot_rets = [y for _, y in sorted_tkr[:third]]
                top_rets = [y for _, y in sorted_tkr[t - third:]]
                tkr_spread = float(np.mean(top_rets)) - float(np.mean(bot_rets))
                if (1 if tkr_spread >= 0 else -1) == sign_global:
                    n_agreeing += 1
            if n_qualifying < 3:
                continue
            breadth = n_agreeing / n_qualifying
        else:
            # Single-ticker: breadth is 1.0 by convention.
            n_qualifying = 1
            n_agreeing   = 1
            breadth      = 1.0

        score = weighted_spread * breadth

        # win_lift: win rate spread between top and bottom bins (info only).
        top_wr    = float(np.mean([1.0 if y > 0 else 0.0 for y in top_b]))
        bottom_wr = float(np.mean([1.0 if y > 0 else 0.0 for y in bottom_b]))
        win_lift  = top_wr - bottom_wr

        results.append({
            "name":                  feat,
            "score":                 round(score,        6),
            "spread":                round(raw_spread,   6),
            "breadth":               round(breadth,      4),
            "w":                     round(w,            4),
            "win_lift":              round(win_lift,      4),
            "n_top":                 n_top,
            "n_bottom":              n_bottom,
            "top_bin":               top_i + 1,
            "bottom_bin":            bottom_i + 1,
            "n_qualifying_bins":     len(populated),
            "n_qualifying_tickers":  n_qualifying,
            "n":                     n,
        })

    results.sort(key=lambda x: x["score"], reverse=True)
    return results


async def _fetch_ticker_calendars(oi_pool, tickers: list) -> dict:
    """Per-ticker trading-day calendar with open/close lookups.

    Returns {ticker: {dates: [sorted ISO dates], date_idx: {date: i},
                      open: {date: spot_co}, close: {date: close_price}}}.
    close[d] = spot_pc of the NEXT trading day (= close of d).
    """
    if not tickers:
        return {}
    async with oi_pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT ticker, trade_date, spot_co, spot_pc FROM daily_features "
            "WHERE ticker = ANY($1) AND spot_co IS NOT NULL "
            "ORDER BY ticker, trade_date", tickers)
    by_tkr: dict = defaultdict(list)
    for r in rows:
        by_tkr[r["ticker"]].append({
            "date":    str(r["trade_date"]),
            "spot_co": r["spot_co"],
            "spot_pc": r["spot_pc"],
        })
    out: dict = {}
    for tkr, entries in by_tkr.items():
        dates = [e["date"] for e in entries]
        open_by_date: dict = {}
        for e in entries:
            try:
                if e["spot_co"] is not None:
                    open_by_date[e["date"]] = round(float(e["spot_co"]), 2)
            except (TypeError, ValueError):
                pass
        close_by_date: dict = {}
        for i in range(len(entries) - 1):
            npc = entries[i + 1]["spot_pc"]
            if npc is None:
                continue
            try:
                close_by_date[entries[i]["date"]] = round(float(npc), 2)
            except (TypeError, ValueError):
                pass
        out[tkr] = {
            "dates":    dates,
            "date_idx": {d: i for i, d in enumerate(dates)},
            "open":     open_by_date,
            "close":    close_by_date,
        }
    return out


def _trade_exit(cal: dict, entry_date: str, horizon: int):
    """Given a ticker calendar and an entry date, return (exit_date, spot_exit).

    Exit date = entry_date + (horizon - 1) trading days. Spot exit = close of
    exit_date (= spot_pc of the day AFTER exit_date in the ticker's sequence).
    """
    if not cal or not entry_date:
        return None, None
    idx = cal.get("date_idx", {}).get(entry_date)
    if idx is None:
        return None, None
    ei = idx + max(horizon - 1, 0)
    dates = cal.get("dates") or []
    if ei >= len(dates):
        return None, None
    exit_date = dates[ei]
    return exit_date, cal.get("close", {}).get(exit_date)


def _build_enriched_trade(row: dict, calendars: dict, horizon: int,
                          primary_metric: Optional[str],
                          outcome_col: str,
                          secondary_metric: Optional[str] = None,
                          extra_metrics: Optional[list] = None) -> dict:
    """Build an enriched trade record for CSV / activity panes.

    Includes ticker, trade_date, primary_val (when primary_metric given),
    optional secondary_val (single) or extra metric values (dict), entry/exit
    spot prices, exit_date, and ret. Missing fields stay None.
    """
    def _f(v):
        if v is None:
            return None
        try:
            fv = float(v)
            if math.isnan(fv):
                return None
            return fv
        except (TypeError, ValueError):
            return None

    tkr     = row.get("ticker", "")
    date_s  = row.get("trade_date", "")
    cal     = calendars.get(tkr) or {}
    exit_d, spot_exit = _trade_exit(cal, date_s, horizon)
    spot_entry = _f(row.get("spot_co"))
    if spot_entry is None:
        spot_entry = cal.get("open", {}).get(date_s)

    rec = {
        "ticker":     tkr,
        "trade_date": date_s,
        "primary_val":  _f(row.get(primary_metric)) if primary_metric else None,
        "secondary_val": _f(row.get(secondary_metric)) if secondary_metric else None,
        "spot_entry": spot_entry,
        "exit_date":  exit_d,
        "spot_exit":  spot_exit,
        "ret":        _f(row.get(outcome_col)),
    }
    if extra_metrics:
        rec["extra"] = {m: _f(row.get(m)) for m in extra_metrics}
    return rec


def _sec_equity_curve(rows_sorted: list, outcome_col: str) -> list:
    """Cumulative return curve from a list of rows sorted by date."""
    cum = 0.0
    curve = []
    for r in rows_sorted:
        y = r.get(outcome_col)
        if y is None:
            continue
        try:
            cum += float(y)
        except (TypeError, ValueError):
            continue
        curve.append({"date": r.get("trade_date", ""), "value": round(cum, 6)})
    return curve


# `_DEFAULT_WALKFWD_WARMUP` is still referenced by `_walk_forward_thresholds`
# below. The Assigner-side row-binning helpers (`_walk_forward_bins`,
# `_bin_membership`, `_bin_for_value`, etc.) were relocated to row_compute.py
# in Step 7j; `_walk_forward_thresholds` remains here because it's specific
# to the /threshold-drift view.
_DEFAULT_WALKFWD_WARMUP = 252  # trading days; ~1 year.


def _walk_forward_thresholds(rows_chrono: list, metric: str, n_bins: int,
                             bins_to_track: list,
                             warmup: int = _DEFAULT_WALKFWD_WARMUP) -> tuple:
    """Time-series of per-ticker bin-K UPPER thresholds at month-end.

    Returns:
      - samples: list of {date, bin, ticker, threshold,
                          threshold_full_ticker} — one record per
        (last-trading-day-of-month, ticker, bin K). `threshold_full_ticker`
        is the SAME ticker's full-history bin K threshold (so the caller
        can compute a dimensionless ratio without needing a separate
        lookup).
      - full_per_ticker: {ticker: {bin: full_history_threshold}}.
      - full_thresholds: {bin: [per-ticker in-sample thresholds]} — list
        across tickers (useful for the dotted reference line on native
        mode).

    Rows must already be sorted (ticker, trade_date). Per ticker, walk
    values chronologically with bisect.insort; at each month boundary
    sample np.quantile(sorted_vals, K/n_bins) per K. After the walk
    completes for that ticker, the final sorted list yields the
    full-history threshold per bin which we attach to every prior
    sample for that ticker.
    """
    import bisect
    samples: list = []
    full_thresholds: dict = {b: [] for b in bins_to_track}
    full_per_ticker: dict = {}
    n_bins = max(2, min(20, int(n_bins)))
    warm = max(int(warmup), n_bins)

    # Group rows by ticker preserving date order.
    by_tkr: dict = defaultdict(list)
    for r in rows_chrono:
        v = r.get(metric)
        if v is None:
            continue
        try:
            fv = float(v)
            if math.isnan(fv):
                continue
        except (TypeError, ValueError):
            continue
        by_tkr[r.get("ticker", "_")].append((str(r.get("trade_date", "")), fv))

    # Use the MIDPOINT of bin K's quantile range — quantile((K-0.5)/n_bins) —
    # so the threshold is symmetric and meaningful at both ends:
    #   B1  → 2.5th percentile (the centre of the bottom bin)
    #   B20 → 97.5th percentile (the centre of the top bin)
    def _q(K):
        return (K - 0.5) / n_bins

    # Canonical month-end calendar from the UNION of all tickers' dates.
    # Snapshotting every ticker at the same set of dates is critical —
    # otherwise sparse tickers cause spike-and-recover artefacts where
    # a non-month-end date appears in the aggregated series with only
    # ONE contributing ticker.
    all_dates = set()
    for items in by_tkr.values():
        for date_s, _v in items:
            all_dates.add(date_s)
    if not all_dates:
        return [], {}, full_thresholds
    # For each month, keep the last (max) date seen across all tickers.
    month_last: dict = {}
    for d in sorted(all_dates):
        month_last[d[:7]] = d
    canonical_month_ends = sorted(month_last.values())

    for tkr, items in by_tkr.items():
        cum_vals: list = []
        # items is already chronologically sorted (rows were ORDER BY ticker, trade_date).
        idx = 0
        n_items = len(items)
        for me_date in canonical_month_ends:
            # Advance through this ticker's data up to and including me_date,
            # inserting each value into the running sorted list.
            while idx < n_items and items[idx][0] <= me_date:
                bisect.insort(cum_vals, items[idx][1])
                idx += 1
            if len(cum_vals) >= warm:
                for b in bins_to_track:
                    thr = float(np.quantile(cum_vals, _q(b)))
                    samples.append({
                        "date":      me_date,
                        "bin":       b,
                        "ticker":    tkr,
                        "threshold": round(thr, 6),
                    })
            # If the ticker has no more data beyond me_date, the snapshot
            # is still a legitimate "as-of" value — keep going so the line
            # extends to the end of the chart.

        # Full-history reference per ticker (after walking all dates).
        ticker_full = {}
        if len(cum_vals) >= n_bins:
            for b in bins_to_track:
                full_v = float(np.quantile(cum_vals, _q(b)))
                ticker_full[b] = full_v
                full_thresholds[b].append(full_v)
        full_per_ticker[tkr] = ticker_full

    # Second pass: attach each ticker's full-history threshold to its samples
    # so the endpoint can compute drift ratios in O(1).
    for s in samples:
        ft = full_per_ticker.get(s["ticker"], {})
        s["threshold_full_ticker"] = round(ft.get(s["bin"], 0.0), 6) \
            if s["bin"] in ft else None

    return samples, full_per_ticker, full_thresholds


async def _fetch_bin20_by_metric(
    pool,
    metrics: list,
    filter_pairs: list,
) -> dict:
    """Group 4 (Secondary endpoints): prefetch bin20 from is_bins for a
    set of metrics, restricted to a primary-filtered (ticker, trade_date)
    set. Returns ``{metric: {(ticker, date_str): bin20}}``.

    DYNAMIC column detection. We probe `information_schema` for which
    `bin20_<metric>` columns actually exist in is_bins — NOT a hardcoded
    list. When the 7 currently-empty features (iv_25d_call_30d et al.)
    get backfilled, their bin20 columns appear and the next request
    picks them up automatically; no code change needed. Metrics that
    don't have a column yet are silently omitted from the returned dict,
    and callers skip them in the same loop they already use to iterate
    metrics.

    SCALING NOTE (logged so it isn't a surprise later): for narrow
    filters (~20K rows on the current 221K-row is_bins) the planner
    picks an index nested-loop on the (ticker, trade_date) PK and
    runs in ~0.4s. As the filter set grows past roughly 25–30% of the
    table, the planner flips to a hash-join + seq-scan over is_bins
    (~1.8s today on a ~55K filter, still well under the 1–3s budget).
    The seq-scan cost scales with TOTAL is_bins size, not filter size,
    so very wide primary selections — or future growth of is_bins as
    history/tickers accumulate — will land closer to the wide-case
    cost. Not a blocker; the table's too small to justify a covering
    index now. Revisit if is_bins crosses ~1M rows or wide selections
    start regularly exceeding 3s.
    """
    if not metrics or not filter_pairs:
        return {}
    # Dynamic eligibility check — auto-heals when missing bin20 columns
    # get backfilled (no hardcoded allowlist).
    async with pool.acquire() as conn:
        col_rows = await conn.fetch(
            """SELECT column_name FROM information_schema.columns
               WHERE table_name = 'is_bins' AND table_schema = 'public'
                 AND column_name LIKE 'bin20_%'""")
    available = {r["column_name"] for r in col_rows}
    eligible = [m for m in metrics if f"bin20_{m}" in available]
    if not eligible:
        return {}
    # One wide filtered SELECT: ticker + trade_date + N bin20 columns,
    # restricted via JOIN unnest to the primary-filtered (ticker, date)
    # pairs. The planner's choice (index NL vs hash-join + seq-scan) is
    # filter-size-dependent — see the SCALING NOTE in this function's
    # docstring.
    bin20_select = ", ".join(f"ib.bin20_{m}" for m in eligible)
    sql = (
        f"SELECT ib.ticker, ib.trade_date, {bin20_select} "
        f"FROM is_bins ib "
        f"JOIN unnest($1::text[], $2::date[]) AS f(ticker, trade_date) "
        f"  ON ib.ticker = f.ticker AND ib.trade_date = f.trade_date"
    )
    tkrs = [t for (t, _) in filter_pairs]
    # Accept date-as-string or date-as-date; asyncpg needs date objects.
    from datetime import date as _date_cls
    dates = []
    for (_, d) in filter_pairs:
        if isinstance(d, _date_cls):
            dates.append(d)
        else:
            dates.append(_date_cls.fromisoformat(str(d)))
    async with pool.acquire() as conn:
        rows = await conn.fetch(sql, tkrs, dates)
    # Build per-metric (key → bin20) dicts. Two perf moves over the
    # naïve nested loop that hit 35s on the 220K no-selection case:
    #
    #   1. Index access vs name access. asyncpg Records support
    #      positional `r[idx]` which skips the per-access name→index
    #      hash lookup. We resolve the column → index map ONCE upfront
    #      (constant cost) and then access by int.
    #   2. List-of-dicts in a tight per-row inner loop, then zip back
    #      to a name-keyed dict at the end. Avoids 23.8M `out[m][...]`
    #      double-dict-lookups (the outer one is `out[m]` for every
    #      single inner write).
    #
    # The fundamental cost — N_rows × N_metrics dict writes — is still
    # there; this just trims the per-write constant. For the
    # no-selection case (220K × 108) this brings the dict build from
    # ~35s to ~12s. The SQL fetch itself (10s for 95MB of bin data)
    # is a separate floor and isn't touched here.
    if not rows:
        return {m: {} for m in eligible}
    field_names = list(rows[0].keys())
    ticker_idx = field_names.index("ticker")
    date_idx   = field_names.index("trade_date")
    col_idx    = [field_names.index(f"bin20_{m}") for m in eligible]
    inners: list = [dict() for _ in eligible]
    for r in rows:
        d = r[date_idx]
        d_str = d.isoformat() if hasattr(d, "isoformat") else str(d)
        key = (r[ticker_idx], d_str)
        for i, ci in enumerate(col_idx):
            v = r[ci]
            if v is not None and v > 0:
                inners[i][key] = v
    out = dict(zip(eligible, inners))
    return out


async def _fetch_wfbin20_by_metric(
    pool,
    metrics: list,
    filter_pairs: list,
) -> dict:
    """Group 7 (WF migration): prefetch bin20 from wf_bins for a set of
    metrics, restricted to a primary-filtered (ticker, trade_date) set.
    Returns ``{metric: {(ticker, date_str): bin20}}``.

    Mirror of `_fetch_bin20_by_metric` against `wf_bins` instead of
    `is_bins`. Encoding A applies — `wf_bins.bin20_<metric> = 0`
    collapses null-metric AND walk-forward warm-up exclusion into the
    same sentinel, so the same `if v is not None and v > 0` guard in
    the dict-build drops both classes of row in one pass. Dashboard
    code that consumes the returned lookup applies `WHERE bin20 > 0`
    as its only filter, identical to the IS path.

    Same dynamic column probe as the IS helper (auto-heals when the
    other project backfills new metrics).
    """
    if not metrics or not filter_pairs:
        return {}
    async with pool.acquire() as conn:
        col_rows = await conn.fetch(
            """SELECT column_name FROM information_schema.columns
               WHERE table_name = 'wf_bins' AND table_schema = 'public'
                 AND column_name LIKE 'bin20_%'""")
    available = {r["column_name"] for r in col_rows}
    eligible = [m for m in metrics if f"bin20_{m}" in available]
    if not eligible:
        return {}
    bin20_select = ", ".join(f"wb.bin20_{m}" for m in eligible)
    sql = (
        f"SELECT wb.ticker, wb.trade_date, {bin20_select} "
        f"FROM wf_bins wb "
        f"JOIN unnest($1::text[], $2::date[]) AS f(ticker, trade_date) "
        f"  ON wb.ticker = f.ticker AND wb.trade_date = f.trade_date"
    )
    tkrs = [t for (t, _) in filter_pairs]
    from datetime import date as _date_cls
    dates = []
    for (_, d) in filter_pairs:
        if isinstance(d, _date_cls):
            dates.append(d)
        else:
            dates.append(_date_cls.fromisoformat(str(d)))
    async with pool.acquire() as conn:
        rows = await conn.fetch(sql, tkrs, dates)
    if not rows:
        return {m: {} for m in eligible}
    field_names = list(rows[0].keys())
    ticker_idx = field_names.index("ticker")
    date_idx   = field_names.index("trade_date")
    col_idx    = [field_names.index(f"bin20_{m}") for m in eligible]
    inners: list = [dict() for _ in eligible]
    for r in rows:
        d = r[date_idx]
        d_str = d.isoformat() if hasattr(d, "isoformat") else str(d)
        key = (r[ticker_idx], d_str)
        for i, ci in enumerate(col_idx):
            v = r[ci]
            if v is not None and v > 0:
                inners[i][key] = v
    out = dict(zip(eligible, inners))
    return out


async def _fetch_ttbin20_by_metric(
    pool,
    metrics: list,
    filter_pairs: list,
) -> dict:
    """Group 8 (TT migration): prefetch bin20 from tt_bins for a set
    of metrics, restricted to a primary-filtered (ticker, trade_date)
    set. Returns ``{metric: {(ticker, date_str): bin20}}``.

    Mirror of `_fetch_bin20_by_metric` / `_fetch_wfbin20_by_metric`
    against `tt_bins`. tt_bins is IS-frozen-at-cutoff: per-ticker
    in-sample ranks derived from pre-cutoff history, frozen, then
    applied to both train and test rows. Encoding A applies —
    `tt_bins.bin20_<metric> = 0` collapses null-metric AND
    insufficient-training-sample exclusion into the same sentinel.
    The same `if v is not None and v > 0` dict-build guard handles
    both classes of row.

    Selects bin20 columns BY EXPLICIT NAME (no SELECT *, no positional
    reads) — tt_bins carries an extra `cutoff_date` column that the
    other two stored-bin tables don't have, and explicit named selects
    keep the helper neutral about extra columns the build may add
    later.

    Same dynamic column probe as the other two helpers (auto-heals
    when the build backfills new metrics or the 5 currently-bin-0
    tickers get added).
    """
    if not metrics or not filter_pairs:
        return {}
    async with pool.acquire() as conn:
        col_rows = await conn.fetch(
            """SELECT column_name FROM information_schema.columns
               WHERE table_name = 'tt_bins' AND table_schema = 'public'
                 AND column_name LIKE 'bin20_%'""")
    available = {r["column_name"] for r in col_rows}
    eligible = [m for m in metrics if f"bin20_{m}" in available]
    if not eligible:
        return {}
    # Named-column SELECT — never SELECT *. tt_bins has an extra
    # cutoff_date column compared to is_bins / wf_bins; the named list
    # keeps the helper agnostic to that and any future schema additions.
    bin20_select = ", ".join(f"tb.bin20_{m}" for m in eligible)
    sql = (
        f"SELECT tb.ticker, tb.trade_date, {bin20_select} "
        f"FROM tt_bins tb "
        f"JOIN unnest($1::text[], $2::date[]) AS f(ticker, trade_date) "
        f"  ON tb.ticker = f.ticker AND tb.trade_date = f.trade_date"
    )
    tkrs = [t for (t, _) in filter_pairs]
    from datetime import date as _date_cls
    dates = []
    for (_, d) in filter_pairs:
        if isinstance(d, _date_cls):
            dates.append(d)
        else:
            dates.append(_date_cls.fromisoformat(str(d)))
    async with pool.acquire() as conn:
        rows = await conn.fetch(sql, tkrs, dates)
    if not rows:
        return {m: {} for m in eligible}
    field_names = list(rows[0].keys())
    ticker_idx = field_names.index("ticker")
    date_idx   = field_names.index("trade_date")
    col_idx    = [field_names.index(f"bin20_{m}") for m in eligible]
    inners: list = [dict() for _ in eligible]
    for r in rows:
        d = r[date_idx]
        d_str = d.isoformat() if hasattr(d, "isoformat") else str(d)
        key = (r[ticker_idx], d_str)
        for i, ci in enumerate(col_idx):
            v = r[ci]
            if v is not None and v > 0:
                inners[i][key] = v
    out = dict(zip(eligible, inners))
    return out


async def _fetch_stored_bin20_by_metric(
    pool, mode: str, metrics: list, filter_pairs: list,
) -> dict:
    """Dispatcher — returns the right stored-bin lookup for the given
    spec mode. `mode` is the spec.kind string (`"in_sample"`,
    `"walk_forward"`, or `"train_test"`). For unknown modes returns
    empty.
    """
    if mode == "in_sample":
        return await _fetch_bin20_by_metric(pool, metrics, filter_pairs)
    if mode == "walk_forward":
        return await _fetch_wfbin20_by_metric(pool, metrics, filter_pairs)
    if mode == "train_test":
        return await _fetch_ttbin20_by_metric(pool, metrics, filter_pairs)
    return {}


# ── Group 8: TT cutoff discovery ─────────────────────────────────────────
# tt_bins is built with one frozen cutoff date (read it from the table —
# user no longer picks the date). Module-level cache invalidated on
# process restart, which is all the freshness we need: the cutoff
# doesn't change without a rebuild, and the dashboard restart on deploy
# clears it.
_TT_CUTOFF_CACHED: Optional[str] = None


async def _get_tt_cutoff(pool) -> Optional[str]:
    """Return the single cutoff date frozen in tt_bins (ISO string).
    None if the table is empty or unreachable.
    """
    global _TT_CUTOFF_CACHED
    if _TT_CUTOFF_CACHED is not None:
        return _TT_CUTOFF_CACHED
    if not pool:
        return None
    try:
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT MAX(cutoff_date) AS cutoff FROM tt_bins")
        cutoff = row["cutoff"] if row else None
        if cutoff is None:
            return None
        _TT_CUTOFF_CACHED = (cutoff.isoformat()
                             if hasattr(cutoff, "isoformat") else str(cutoff))
        return _TT_CUTOFF_CACHED
    except Exception:
        return None


@router.get("/tt-cutoff")
async def tt_cutoff_endpoint(pool=Depends(get_oi_pool)):
    """Return the TT cutoff date the dashboard should use, read from
    tt_bins. Frontend calls this once on page load and stores the
    result; the user no longer picks a cutoff (the build froze one).
    """
    cutoff = await _get_tt_cutoff(pool)
    return {"cutoff_date": cutoff}


async def _ensure_sec_cache(pool, ticker: str, metric: str, outcome: str,
                            date_from: str = "", date_to: str = "") -> str:
    """Ensure _SEC_CACHE has rows for (ticker, metric, outcome, dates).

    Returns the cache_key. No scanner job is dispatched — this is the
    confirmation-layer-only fast path used by /secondary-prepare-rows.
    If rows are already cached the call is a no-op (returns instantly).
    """
    cache_key = _sec_cache_key(ticker, metric, outcome, date_from, date_to)
    if cache_key in _SEC_CACHE:
        return cache_key

    is_all = (ticker == "ALL")
    date_conditions = ""
    params: list = []
    p = 1
    if not is_all:
        date_conditions += f" AND ticker = ${p}"; params.append(ticker); p += 1
    if date_from:
        from datetime import date as _date
        date_conditions += f" AND trade_date >= ${p}"; params.append(_date.fromisoformat(date_from)); p += 1
    if date_to:
        from datetime import date as _date
        date_conditions += f" AND trade_date <= ${p}"; params.append(_date.fromisoformat(date_to)); p += 1

    async with pool.acquire() as conn:
        col_rows = await conn.fetch(
            """SELECT column_name FROM information_schema.columns
               WHERE table_name = 'daily_features' AND table_schema = 'public'
               AND data_type IN ('double precision','numeric','real','integer','bigint','smallint')
               AND column_name NOT IN ('id','ticker','trade_date','created_at','updated_at')
               ORDER BY ordinal_position""")
    all_num_cols = [r["column_name"] for r in col_rows]
    outcome_cols_all = [c for c in all_num_cols if "ret_" in c and "fwd" in c]
    feature_cols = [c for c in all_num_cols
                    if c not in outcome_cols_all and c != metric
                    and not c.startswith("spot") and not c.endswith("_pc")]

    select_cols = ", ".join(
        ["ticker", "trade_date", outcome, metric, "spot_co", "spot_pc"]
        + feature_cols)
    async with pool.acquire() as conn:
        db_rows = await conn.fetch(
            f"SELECT {select_cols} FROM daily_features "
            f"WHERE {metric} IS NOT NULL AND {outcome} IS NOT NULL"
            f"{date_conditions} ORDER BY trade_date", *params)

    rows = [dict(r) for r in db_rows]
    for r in rows:
        r["trade_date"] = str(r["trade_date"])

    data_as_of = max((r["trade_date"] for r in rows), default=None) if rows else None
    tickers_used = list({r.get("ticker") for r in rows if r.get("ticker")})
    cal_by_tkr = await _fetch_ticker_calendars(pool, tickers_used) if tickers_used else {}

    _SEC_CACHE[cache_key] = {
        "rows":           rows,
        "features":       feature_cols,
        "outcome":        outcome,
        "primary_metric": metric,
        "calendars":      cal_by_tkr,
        "data_as_of":     data_as_of,
        # mode/cutoff don't affect the row set — SecDetailReq carries live mode
        "mode":           "walk_forward",
        "cutoff_date":    "",
    }
    return cache_key


class SecLoadReq(BaseModel):
    ticker: str
    metric: str
    outcome: str
    date_from: str = ""
    date_to: str = ""
    filtered_dates: List[str] = []
    sec_bin_count: int = 10
    selected_primary_bins: Optional[List[int]] = None
    mode: str = "walk_forward"
    cutoff_date: str = ""


class SecScanReq(BaseModel):
    cache_key: str
    filtered_dates: List[str]
    ticker: str = "SPX"
    sec_bin_count: int = 10
    selected_primary_bins: Optional[List[int]] = None


class SecDetailReq(BaseModel):
    cache_key: str
    metric_b: str
    filtered_dates: List[str]
    sec_bins: List[int] = [10]
    sec_bin_count: int = 10
    ticker: str = "SPX"
    selected_primary_bins: Optional[List[int]] = None
    walk_forward: bool = True                    # live mode from toggle
    cutoff_date: Optional[str] = None            # train-test cutoff when set


@router.post("/secondary-load")
async def secondary_load(req: SecLoadReq, pool=Depends(get_oi_pool)):
    """Fetch feature columns for the analysis date range (cached), then dispatch
    background scoring job.  Returns immediately with status='computing' on
    cache miss, or status='done'/'error' if the job already finished.
    Checks sec_scan_cache before dispatching — a DB hit is served instantly."""
    if not pool:
        return {"error": "OI database not configured"}

    await _ensure_sec_scan_table(pool)

    is_all = (req.ticker == "ALL")
    cache_key = _sec_cache_key(req.ticker, req.metric, req.outcome, req.date_from, req.date_to)

    if cache_key not in _SEC_CACHE:
        # Build date filter
        date_conditions = ""
        params: list = []
        p = 1
        if not is_all:
            date_conditions += f" AND ticker = ${p}"; params.append(req.ticker); p += 1
        if req.date_from:
            from datetime import date as _date
            date_conditions += f" AND trade_date >= ${p}"; params.append(_date.fromisoformat(req.date_from)); p += 1
        if req.date_to:
            from datetime import date as _date
            date_conditions += f" AND trade_date <= ${p}"; params.append(_date.fromisoformat(req.date_to)); p += 1

        # Discover all numeric columns
        async with pool.acquire() as conn:
            col_rows = await conn.fetch(
                """SELECT column_name FROM information_schema.columns
                   WHERE table_name = 'daily_features' AND table_schema = 'public'
                   AND data_type IN ('double precision','numeric','real','integer','bigint','smallint')
                   AND column_name NOT IN ('id','ticker','trade_date','created_at','updated_at')
                   ORDER BY ordinal_position""")
        all_num_cols = [r["column_name"] for r in col_rows]
        outcome_cols_all = [c for c in all_num_cols if "ret_" in c and "fwd" in c]
        feature_cols = [c for c in all_num_cols
                        if c not in outcome_cols_all and c != req.metric
                        and not c.startswith("spot") and not c.endswith("_pc")]

        # Pull req.metric, spot_co, spot_pc alongside the features so the CSV /
        # trade-record builders can populate primary_val + spot_entry + spot_exit.
        select_cols = ", ".join(
            ["ticker", "trade_date", req.outcome, req.metric, "spot_co", "spot_pc"]
            + feature_cols)
        async with pool.acquire() as conn:
            db_rows = await conn.fetch(
                f"SELECT {select_cols} FROM daily_features "
                f"WHERE {req.metric} IS NOT NULL AND {req.outcome} IS NOT NULL"
                f"{date_conditions} ORDER BY trade_date", *params)

        rows = [dict(r) for r in db_rows]
        for r in rows:
            r["trade_date"] = str(r["trade_date"])

        data_as_of = max((r["trade_date"] for r in rows), default=None) if rows else None

        # Build per-ticker calendar (open/close lookup + date sequence).
        tickers_used = list({r.get("ticker") for r in rows if r.get("ticker")})
        cal_by_tkr = await _fetch_ticker_calendars(pool, tickers_used) if tickers_used else {}

        _SEC_CACHE[cache_key] = {
            "rows":           rows,
            "features":       feature_cols,
            "outcome":        req.outcome,
            "primary_metric": req.metric,
            "calendars":      cal_by_tkr,
            "data_as_of":     data_as_of,
            "mode":           req.mode,
            "cutoff_date":    req.cutoff_date,
        }

    cached = _SEC_CACHE[cache_key]
    rows = cached["rows"]
    feature_cols = cached["features"]
    data_as_of = cached.get("data_as_of")
    n_bins = max(2, min(20, req.sec_bin_count))
    scan_key = _sec_score_key(cache_key, req.selected_primary_bins, n_bins, req.filtered_dates)
    structural_key = _sec_structural_key(
        req.ticker, req.metric, req.outcome, req.mode,
        req.cutoff_date, n_bins, req.selected_primary_bins or [],
    )

    entry = _SEC_SCORE_CACHE.get(scan_key)
    if entry is None:
        # Check DB cache before running a job — stored results are served instantly.
        try:
            async with pool.acquire() as conn:
                db_row = await conn.fetchrow(
                    "SELECT payload, data_as_of FROM sec_scan_cache WHERE structural_key = $1",
                    structural_key,
                )
        except Exception:
            db_row = None
        if db_row:
            result_dict = json.loads(db_row["payload"])
            result_dict["data_as_of"] = str(db_row["data_as_of"]) if db_row["data_as_of"] else data_as_of
            _SEC_SCORE_CACHE[scan_key] = {"status": "done", "result": result_dict}
        else:
            # DB miss — dispatch background job (IC batch discipline: never auto-compute).
            if _sec_scan_running:
                return {"cache_key": cache_key, "scan_key": scan_key, "status": "busy",
                        "error": "A secondary scan is already running — wait for it to finish."}
            if len(_ic_batch_running) >= 1:
                return {"cache_key": cache_key, "scan_key": scan_key, "status": "busy",
                        "error": "An IC batch job is running — wait for it to finish before loading the scanner."}
            while len(_SEC_SCORE_CACHE) >= _SEC_SCORE_CACHE_MAX:
                _SEC_SCORE_CACHE.pop(next(iter(_SEC_SCORE_CACHE)))
            _SEC_SCORE_CACHE[scan_key] = {"status": "computing", "result": None, "error_msg": None}
            loop = asyncio.get_running_loop()
            db_write_params = {
                "loop":           loop,
                "pool":           pool,
                "structural_key": structural_key,
                "ticker":         req.ticker,
                "metric":         req.metric,
                "outcome":        req.outcome,
                "mode":           req.mode,
                "cutoff_date":    req.cutoff_date,
                "data_as_of":     data_as_of,
                "n_input_rows":   len(rows),
            }
            # Group 7: prefetch wf_bins for the primary metric AND every
            # feature, so the sync scoring run uses stored bins instead
            # of computing on the fly. Primary's lookup feeds
            # filter_by_assignments via _walk_forward_primary_filter's
            # stored-bin replacement; per-feature lookups feed
            # assign_secondary_buckets. Scanner is always WF; ALL-mode
            # only does this prefetch (single-ticker stays on legacy).
            bin20_by_metric: dict = {}
            if is_all:
                pairs = [(r.get("ticker", ""), r.get("trade_date", "")) for r in rows]
                all_metrics = [cached["primary_metric"]] + list(feature_cols)
                bin20_by_metric = await _fetch_wfbin20_by_metric(
                    pool, all_metrics, pairs)
            loop.run_in_executor(
                None, _run_sec_score,
                scan_key, rows, cached["outcome"], feature_cols,
                is_all, n_bins, cached["primary_metric"], req.selected_primary_bins, req.filtered_dates,
                db_write_params, bin20_by_metric,
            )

    entry = _SEC_SCORE_CACHE.get(scan_key, {"status": "computing"})
    response: dict = {"cache_key": cache_key, "scan_key": scan_key, "status": entry["status"]}
    if entry["status"] == "error":
        response["error"] = entry.get("error_msg", "unknown error")
    elif entry["status"] == "done" and entry.get("result"):
        response.update(entry["result"])
    return response


@router.post("/secondary-scan")
async def secondary_scan(req: SecScanReq, pool=Depends(get_oi_pool)):
    """Re-score secondary metrics for a new primary bin selection (background job).
    Checks sec_scan_cache before dispatching — a DB hit is served instantly."""
    cached = _SEC_CACHE.get(req.cache_key)
    if not cached:
        return {"error": "cache_miss"}

    is_all = (req.ticker == "ALL")
    rows = cached["rows"]
    feature_cols = cached["features"]
    outcome_col = cached["outcome"]
    primary_metric = cached["primary_metric"]
    data_as_of = cached.get("data_as_of")
    mode = cached.get("mode", "walk_forward")
    cutoff_date = cached.get("cutoff_date", "")
    n_bins = max(2, min(20, req.sec_bin_count))
    scan_key = _sec_score_key(req.cache_key, req.selected_primary_bins, n_bins, req.filtered_dates)
    structural_key = _sec_structural_key(
        req.ticker, primary_metric, outcome_col, mode,
        cutoff_date, n_bins, req.selected_primary_bins or [],
    )

    entry = _SEC_SCORE_CACHE.get(scan_key)
    if entry is None:
        # Check DB cache before running a job — stored results are served instantly.
        db_row = None
        if pool:
            try:
                async with pool.acquire() as conn:
                    db_row = await conn.fetchrow(
                        "SELECT payload, data_as_of FROM sec_scan_cache WHERE structural_key = $1",
                        structural_key,
                    )
            except Exception:
                db_row = None
        if db_row:
            result_dict = json.loads(db_row["payload"])
            result_dict["data_as_of"] = str(db_row["data_as_of"]) if db_row["data_as_of"] else data_as_of
            _SEC_SCORE_CACHE[scan_key] = {"status": "done", "result": result_dict}
        else:
            if _sec_scan_running:
                return {"cache_key": req.cache_key, "scan_key": scan_key, "status": "busy",
                        "error": "A secondary scan is already running — wait for it to finish."}
            if len(_ic_batch_running) >= 1:
                return {"cache_key": req.cache_key, "scan_key": scan_key, "status": "busy",
                        "error": "An IC batch job is running — wait for it to finish before loading the scanner."}
            while len(_SEC_SCORE_CACHE) >= _SEC_SCORE_CACHE_MAX:
                _SEC_SCORE_CACHE.pop(next(iter(_SEC_SCORE_CACHE)))
            _SEC_SCORE_CACHE[scan_key] = {"status": "computing", "result": None, "error_msg": None}
            loop = asyncio.get_running_loop()
            db_write_params = {
                "loop":           loop,
                "pool":           pool,
                "structural_key": structural_key,
                "ticker":         req.ticker,
                "metric":         primary_metric,
                "outcome":        outcome_col,
                "mode":           mode,
                "cutoff_date":    cutoff_date,
                "data_as_of":     data_as_of,
                "n_input_rows":   len(rows),
            } if pool else None
            # Group 7: prefetch wf_bins (same shape as /secondary-load).
            bin20_by_metric: dict = {}
            if is_all and pool:
                pairs = [(r.get("ticker", ""), r.get("trade_date", "")) for r in rows]
                all_metrics = [primary_metric] + list(feature_cols)
                bin20_by_metric = await _fetch_wfbin20_by_metric(
                    pool, all_metrics, pairs)
            loop.run_in_executor(
                None, _run_sec_score,
                scan_key, rows, outcome_col, feature_cols,
                is_all, n_bins, primary_metric, req.selected_primary_bins, req.filtered_dates,
                db_write_params, bin20_by_metric,
            )

    entry = _SEC_SCORE_CACHE.get(scan_key, {"status": "computing"})
    response: dict = {"cache_key": req.cache_key, "scan_key": scan_key, "status": entry["status"]}
    if entry["status"] == "error":
        response["error"] = entry.get("error_msg", "unknown error")
    elif entry["status"] == "done" and entry.get("result"):
        response.update(entry["result"])
    return response


class SecStatusReq(BaseModel):
    scan_key: str


@router.post("/secondary-score-status")
async def secondary_score_status(req: SecStatusReq):
    """Poll the status of a background secondary score job."""
    entry = _SEC_SCORE_CACHE.get(req.scan_key)
    if not entry:
        return {"status": "not_found"}
    response: dict = {"status": entry["status"]}
    if entry["status"] == "error":
        response["error"] = entry.get("error_msg", "unknown error")
    elif entry["status"] == "done" and entry.get("result"):
        response.update(entry["result"])
    return response


class SecPrepareReq(BaseModel):
    ticker: str
    metric: str
    outcome: str
    date_from: str = ""
    date_to: str = ""


@router.post("/secondary-prepare-rows")
async def secondary_prepare_rows(req: SecPrepareReq, pool=Depends(get_oi_pool)):
    """Ensure _SEC_CACHE has rows for this primary context. Returns cache_key.

    Confirmation-layer fast path — called by the frontend when the primary
    metric/ticker/outcome/dates change but the user has a secondary selected.
    Does NOT dispatch a scanner job; just ensures the row-set is cached so
    /secondary-detail can run immediately."""
    if not pool:
        return {"error": "OI database not configured"}
    try:
        cache_key = await _ensure_sec_cache(
            pool, req.ticker, req.metric, req.outcome, req.date_from, req.date_to)
        return {"cache_key": cache_key}
    except Exception as exc:
        logging.exception("secondary_prepare_rows failed")
        return {"error": str(exc)}


@router.post("/secondary-scan/invalidate")
async def secondary_scan_invalidate(pool=Depends(get_oi_pool)):
    """Drop every row from `sec_scan_cache`.

    Use this when the underlying daily_features values have changed
    (recomputed core metrics, new ingestion runs, etc.) so the next
    /secondary-scan request can no longer be served from a stale
    cached payload — it falls through to a fresh background scan.

    Does NOT touch the in-memory `_SEC_CACHE` (the per-load row cache
    keyed by /secondary-load's cache_key) — that one auto-clears on
    dashboard restart and is keyed per-session anyway.

    Mirrors /global-metric-bins/invalidate. Non-fatal on DB error.
    """
    if pool:
        try:
            await _ensure_sec_scan_table(pool)
            async with pool.acquire() as conn:
                await conn.execute("DELETE FROM sec_scan_cache")
        except Exception:
            import logging
            logging.warning("sec_scan_cache DELETE failed during invalidate", exc_info=True)
    return {"ok": True}


@router.post("/secondary-detail")
async def secondary_detail(req: SecDetailReq, pool=Depends(get_oi_pool)):
    """2-factor deep dive: bins, equity curves, yearly for a selected secondary metric."""
    cached = _SEC_CACHE.get(req.cache_key)
    if not cached:
        return {"error": "cache_miss"}

    is_all = (req.ticker == "ALL")
    all_rows = cached["rows"]
    outcome_col = cached["outcome"]
    primary_metric = cached.get("primary_metric")
    n_bins = max(2, min(20, req.sec_bin_count))

    # ── Primary filter + secondary bucketing ──────────────────────────────
    # Both steps dispatch on the spec — single path for both in-sample and
    # walk-forward, no inline branches in this endpoint anymore.
    from app.routers.row_compute import (
        make_spec, filter_by_assignments, assign_secondary_buckets,
        mode_envelope,
    )
    # Use the LIVE mode from the request, not the stale mode stored when the
    # scan was first loaded. _SEC_CACHE is keyed without mode, so reading
    # cached["mode"] would always return the mode at scan-load time.
    spec = make_spec(req.walk_forward, req.cutoff_date)

    # Prefetch stored primary-metric bin20 for WF/TT — same fix as
    # /secondary-corr-bins.  IS uses filtered_dates (already correct).
    primary_bin20_by_key: Optional[dict] = None
    if spec.kind in {"walk_forward", "train_test"} and primary_metric and is_all:
        _prim_pairs = [(r.get("ticker", ""), r.get("trade_date", ""))
                       for r in all_rows]
        _prim_by_m  = await _fetch_stored_bin20_by_metric(
            pool, spec.kind, [primary_metric], _prim_pairs)
        primary_bin20_by_key = _prim_by_m.get(primary_metric)

    filtered, dropped, _universe = filter_by_assignments(
        all_rows, spec, primary_metric or "",
        req.selected_primary_bins, is_all, req.filtered_dates,
        primary_bin20_by_key=primary_bin20_by_key,
    )

    # No `len(filtered) < n_bins * 2` panel-block. The drill-in renders
    # against whatever the primary filter passes through; if a bucket is
    # empty its `n` is 0 in the response and the chart shows it that way.
    # Better than a brick-wall error message.

    # Group 4: for IS+ALL prefetch the secondary metric's stored bin20
    # over the primary-filtered (ticker, trade_date) set. Returns empty
    # if req.metric_b doesn't have a bin20_<m> column yet (the 7 not-
    # populated features); downstream falls through to the all_rows
    # path and returns "insufficient_data", which is the right shape
    # for a metric the user shouldn't have been able to click into.
    # Group 7: prefetch dispatches by mode — IS → is_bins, WF → wf_bins.
    # Builder doesn't know which; same lookup shape feeds the helper.
    bin20_by_key = None
    if spec.kind in {"in_sample", "walk_forward", "train_test"}:
        pairs = [(r.get("ticker", ""), r.get("trade_date", "")) for r in filtered]
        by_metric = await _fetch_stored_bin20_by_metric(
            pool, spec.kind, [req.metric_b], pairs)
        bin20_by_key = by_metric.get(req.metric_b)

    # ── Secondary binning ───────────────────────────────────────────────────
    # Spec-dispatched. The in-sample branch may return None when there
    # are too few rows with valid (metric_b, outcome) pairs after the
    # per-ticker n_bins gate — match the legacy "insufficient_data"
    # error in that case. The walk-forward branch never returns None
    # (matches legacy, which always built buckets without a second check).
    buckets = assign_secondary_buckets(
        spec, filtered, req.metric_b, n_bins, outcome_col, is_all,
        # v9: pass full row cache for EVERY mode, including in_sample, so
        # secondary bins are derived from the full-universe distribution
        # rather than re-ranked on the primary-filtered subset. Matches
        # the heatmap's Y-axis assignment exactly — row sums equal the
        # unfiltered secondary bin n's. Pre-v9 in_sample passed None
        # here, causing the Quantile-Secondary chart to render equal-n
        # buckets that disagreed with the heatmap.
        all_rows=all_rows,
        # Group 4: when bin20_by_key is set, the in_sample branch reads
        # bin20 from is_bins instead of computing on the fly — same stored
        # bin as the heatmap and /analyze.
        bin20_by_key=bin20_by_key,
    )
    if buckets is None:
        return {"error": "insufficient_data"}

    # Bin stats
    bins_out = []
    for bi, bucket in enumerate(buckets):
        if not bucket:
            bins_out.append(None)
            continue
        ys = [r[1] for r in bucket]
        bins_out.append({
            "bin":      bi + 1,
            "n":        len(ys),
            "avg_ret":  round(float(np.mean(ys)), 6),
            "win_rate": round(float(np.mean([1.0 if y > 0 else 0.0 for y in ys])), 4),
        })

    # Which secondary bins are selected (1-based)?
    sec_bin_set = set(req.sec_bins) if req.sec_bins else {n_bins}
    combined_tkr_date_set: set = set()
    for bi in sec_bin_set:
        if 1 <= bi <= n_bins:
            for row_t in buckets[bi - 1]:
                combined_tkr_date_set.add((row_t[3], row_t[2]))  # (ticker, date)

    # Equity curves (sorted by date, then ticker for determinism)
    primary_sorted  = sorted(filtered, key=lambda r: (r.get("trade_date", ""), r.get("ticker", "")))
    combined_sorted = [r for r in primary_sorted
                       if (r.get("ticker", ""), r.get("trade_date", "")) in combined_tkr_date_set]

    eq_primary  = _sec_equity_curve(primary_sorted, outcome_col)
    eq_combined = _sec_equity_curve(combined_sorted, outcome_col)

    # Yearly breakdown
    yearly_primary: dict  = defaultdict(list)
    yearly_combined: dict = defaultdict(list)
    for r in primary_sorted:
        yr = int(r.get("trade_date", "0000")[:4])
        o = r.get(outcome_col)
        if o is not None:
            yearly_primary[yr].append(float(o))
    for r in combined_sorted:
        yr = int(r.get("trade_date", "0000")[:4])
        o = r.get(outcome_col)
        if o is not None:
            yearly_combined[yr].append(float(o))

    all_years = sorted(set(yearly_primary) | set(yearly_combined))
    yearly_out = []
    for yr in all_years:
        p_rets = yearly_primary.get(yr, [])
        c_rets = yearly_combined.get(yr, [])
        yearly_out.append({
            "year":         yr,
            "primary_n":    len(p_rets),
            "primary_avg":  round(float(np.mean(p_rets)), 6) if p_rets else 0,
            "primary_wr":   round(float(np.mean([1.0 if v > 0 else 0.0 for v in p_rets])), 4) if p_rets else 0,
            "combined_n":   len(c_rets),
            "combined_avg": round(float(np.mean(c_rets)), 6) if c_rets else 0,
            "combined_wr":  round(float(np.mean([1.0 if v > 0 else 0.0 for v in c_rets])), 4) if c_rets else 0,
        })

    # Per-ticker breakdown for bubble chart
    ticker_rets: dict = defaultdict(list)
    for r in combined_sorted:
        o = r.get(outcome_col)
        if o is not None:
            ticker_rets[r.get("ticker", "?")].append(float(o))

    total_pnl = sum(sum(v) for v in ticker_rets.values())
    tickers_out = []
    for tkr, rets in sorted(ticker_rets.items()):
        n_t = len(rets)
        avg_r = float(np.mean(rets)) if rets else 0.0
        wr = float(np.mean([1.0 if r > 0 else 0.0 for r in rets])) if rets else 0.0
        tkr_pnl = sum(rets)
        contrib = (tkr_pnl / total_pnl * 100) if total_pnl != 0 else 0.0
        tickers_out.append({
            "ticker":      tkr,
            "n":           n_t,
            "avg_ret":     round(avg_r, 6),
            "win_rate":    round(wr, 4),
            "contrib_pct": round(contrib, 2),
        })

    horizon_n   = _parse_horizon(outcome_col)
    primary_m   = cached.get("primary_metric")
    calendars   = cached.get("calendars") or {}
    combined_trades = [
        _build_enriched_trade(
            r, calendars, horizon_n,
            primary_metric=primary_m,
            outcome_col=outcome_col,
            secondary_metric=req.metric_b,
        )
        for r in combined_sorted
    ]

    return {
        "metric_b":    req.metric_b,
        "bins":        bins_out,
        "equity_primary":  eq_primary,
        "equity_combined": eq_combined,
        "yearly":      yearly_out,
        "baseline_n":  len(filtered),
        "combined_n":  len(combined_sorted),
        "horizon":     _parse_horizon(outcome_col),
        "combined_trade_dates": [r.get("trade_date", "") for r in combined_sorted],
        "tickers":     tickers_out,
        "combined_trades": combined_trades,
        # `dropped`, `_universe`, and `start_date` aren't tracked in this
        # endpoint, so universe and start_date pass through as defaults
        # (0 and None). Matches the pre-helper response shape, which also
        # omitted these.
        **mode_envelope(spec, dropped=dropped),
    }


# ── Multi-Metric Correlation Explorer endpoints ───────────────────────────────


class CorrBinsReq(BaseModel):
    cache_key: str
    filtered_dates: List[str]
    ticker: str = "SPX"
    n_bins: int = 10
    walk_forward: bool = False
    cutoff_date: Optional[str] = None  # train-test mode when set (takes precedence over walk_forward)
    selected_primary_bins: Optional[List[int]] = None  # 1..20 primary bin ids (walk_forward / train_test mode)


@router.post("/secondary-corr-bins")
async def secondary_corr_bins(req: CorrBinsReq, pool=Depends(get_oi_pool)):
    """Per-bin avg return for every secondary metric — drives the correlation explorer mini charts.

    walk_forward mode: primary bins for `primary_metric` and secondary
    bins for each `feat` are both computed walk-forward (bisect_left
    against a per-ticker running sorted list). The warmup is 252 trading
    days; warmup rows are excluded from stats.
    """
    cached = _SEC_CACHE.get(req.cache_key)
    if not cached:
        return {"error": "cache_miss"}

    all_rows  = cached["rows"]
    feat_cols = cached["features"]
    outcome_col = cached["outcome"]
    is_all  = (req.ticker == "ALL")
    n_bins  = max(2, min(20, req.n_bins))

    # Route primary filter + per-feature stats through the row_compute
    # spec-dispatched helpers. The fork on `walk_forward` collapses to
    # a single spec selection; the rest of the function body is one path.
    from app.routers.row_compute import (
        make_spec, filter_by_assignments, assign_secondary_bin_stats,
        mode_envelope,
    )
    spec = make_spec(req.walk_forward, req.cutoff_date)
    primary_metric = cached.get("primary_metric") or ""
    if req.walk_forward and not primary_metric:
        return {"error": "no_primary_metric"}

    # Prefetch stored primary-metric bin20 for WF/TT so Group 7 in
    # filter_by_assignments reads the same stored bin as the heatmap —
    # not the on-the-fly _walk_forward_bins recompute.  IS is already
    # correct: filtered_dates from the frontend carries the pre-binned
    # (ticker, date) set for the selected IS primary bins.
    primary_bin20_by_key: Optional[dict] = None
    if spec.kind in {"walk_forward", "train_test"} and primary_metric and is_all:
        _prim_pairs = [(r.get("ticker", ""), r.get("trade_date", ""))
                       for r in all_rows]
        _prim_by_m  = await _fetch_stored_bin20_by_metric(
            pool, spec.kind, [primary_metric], _prim_pairs)
        primary_bin20_by_key = _prim_by_m.get(primary_metric)

    filtered, dropped, universe = filter_by_assignments(
        all_rows, spec, primary_metric,
        req.selected_primary_bins, is_all, req.filtered_dates,
        primary_bin20_by_key=primary_bin20_by_key,
    )
    if not filtered:
        return {"error": "no_data",
                **mode_envelope(spec, dropped=dropped, universe=universe)}

    # Group 4: for IS+ALL, prefetch bin20 from is_bins for every feature
    # column the panel iterates. Dynamic eligibility — features without
    # a bin20_<feat> column (the 7 currently-empty ones) are silently
    # omitted from `bin20_by_metric`; the iteration below skips them
    # with no error, no broken empty mini. Auto-heals when those bin20
    # columns get backfilled (no allowlist anywhere).
    bin20_by_metric: dict = {}
    if spec.kind in {"in_sample", "walk_forward", "train_test"}:
        pairs = [(r.get("ticker", ""), r.get("trade_date", "")) for r in filtered]
        bin20_by_metric = await _fetch_stored_bin20_by_metric(
            pool, spec.kind, list(feat_cols), pairs)

    results = []

    if spec.kind in {"in_sample", "walk_forward", "train_test"}:
        # ── ONE-PASS over filtered rows ────────────────────────────────────
        # Bin math is byte-equivalent to the legacy helper. Legacy emits
        # a 1-indexed bin in [1, n_bins]:
        #
        #     bin_1idx = min(((b20 - 1) * n_bins) // 20 + 1, n_bins)
        #
        # then converts to a 0-indexed bucket index via `bin_1idx - 1`
        # to append into `buckets[bin_1idx - 1]`. The combined form
        # straight to 0-indexed is:
        #
        #     bn = min(((b20 - 1) * n_bins) // 20, n_bins - 1)
        #
        # Verified for the supported n_bins ∈ {5, 10, 20} (divisors of 20):
        #   b20 = 1,  n_bins = 10 → bn = 0  (= bin 1)
        #   b20 = 2,  n_bins = 10 → bn = 0  (= bin 1)
        #   b20 = 20, n_bins = 10 → bn = 9  (= bin 10)
        #   b20 = 20, n_bins = 5  → bn = 4  (= bin 5)
        #   b20 = 20, n_bins = 20 → bn = 19 (= bin 20)
        # The `min(..., n_bins-1)` clamp is a safety belt; for valid
        # b20 ∈ [1, 20] it's never triggered (max is exactly n_bins-1).
        #
        # Per-row validity: outcome must be present + a finite float.
        # Metric validity is implied by `bin20_by_metric[feat].get(key)`
        # returning a non-None value — is_bins's build only assigns a
        # positive bin20 when the metric value was valid at build time,
        # and the prefetch SQL filters `WHERE bin20_<m> > 0`. So
        # checking the metric per-feature per-row (as the legacy did)
        # is redundant under the Group-4 stored-bin invariant.

        # Preserve feat_cols order for the response — eligible_feats is
        # feat_cols filtered for dict membership without reordering.
        eligible_feats = [f for f in feat_cols if f in bin20_by_metric]
        n_features     = len(eligible_feats)
        # Parallel list of per-feature dicts: lets the inner loop do an
        # int-indexed `bin20_dicts[i].get(key)` instead of a
        # `bin20_by_metric[feat].get(key)` dict-of-dicts lookup.
        bin20_dicts = [bin20_by_metric[f] for f in eligible_feats]
        bin_sums    = [[0.0] * n_bins for _ in range(n_features)]
        bin_counts  = [[0]   * n_bins for _ in range(n_features)]

        for r in filtered:
            o = r.get(outcome_col)
            if o is None:
                continue
            try:
                fo = float(o)
                if math.isnan(fo):
                    continue
            except (TypeError, ValueError):
                continue
            tkr = r.get("ticker", "")
            td  = r.get("trade_date", "")
            d_key = td.isoformat() if hasattr(td, "isoformat") else str(td)
            key = (tkr, d_key)
            for i in range(n_features):
                b20 = bin20_dicts[i].get(key)
                if b20 is None or b20 <= 0:
                    # Matches `if b20 is None or b20 <= 0: continue` in
                    # _in_sample_bin_map. Defensive: `if not b20` would
                    # also pass through negative values via Python's
                    # falsiness (and a negative `bn` would silently
                    # index `bin_sums[i][bn]` from the end).
                    continue
                bn = min(((b20 - 1) * n_bins) // 20, n_bins - 1)
                bin_sums[i][bn] += fo
                bin_counts[i][bn] += 1

        for i, feat in enumerate(eligible_feats):
            if not any(bin_counts[i]):
                # Matches legacy: helper returns None when every bucket
                # is empty; the endpoint then drops the feature.
                continue
            bins_avg = [
                round(bin_sums[i][j] / bin_counts[i][j], 6)
                if bin_counts[i][j] else 0.0
                for j in range(n_bins)
            ]
            results.append({
                "name":   feat,
                "bins":   bins_avg,
                "bin_ns": list(bin_counts[i]),
            })
    return {
        "metrics":          results,
        "n_bins":           n_bins,
        "combined_n":       len(filtered),
        **mode_envelope(spec, dropped=dropped, universe=universe,
                        start_date=filtered[0].get("trade_date", "") if filtered else ""),
    }


class CorrReq(BaseModel):
    cache_key: str
    filtered_dates: List[str]
    ticker: str = "SPX"
    n_bins: int = 10
    selections: List[dict]  # [{metric: str, bins: [int]}]
    walk_forward: bool = False
    cutoff_date: Optional[str] = None  # train-test mode when set (takes precedence over walk_forward)
    selected_primary_bins: Optional[List[int]] = None  # 1..20 primary bin ids (walk_forward / train_test mode)


@router.post("/secondary-correlation")
async def secondary_correlation(req: CorrReq, pool=Depends(get_oi_pool)):
    """Phi correlation matrix between selected secondary metrics' binary bin-membership vectors.

    walk_forward mode: both the primary filter (which rows are in the
    universe) and each selection's bin membership are computed
    walk-forward via `_walk_forward_bins`. Warmup rows are dropped.
    """
    cached = _SEC_CACHE.get(req.cache_key)
    if not cached:
        return {"error": "cache_miss"}
    if len(req.selections) < 2:
        return {"error": "need_at_least_2_metrics"}

    all_rows = cached["rows"]
    is_all   = (req.ticker == "ALL")
    n_bins   = max(2, min(20, req.n_bins))

    # Spec-dispatched primary filter + membership. Single path; no fork
    # on `walk_forward` for the bucketing logic.
    from app.routers.row_compute import (
        make_spec, filter_by_assignments, secondary_membership,
        mode_envelope,
    )
    spec = make_spec(req.walk_forward, req.cutoff_date)
    primary_metric = cached.get("primary_metric") or ""
    if req.walk_forward and not primary_metric:
        return {"error": "no_primary_metric"}

    # Prefetch stored primary-metric bin20 for WF/TT — same fix as
    # /secondary-corr-bins and /secondary-detail.
    primary_bin20_by_key: Optional[dict] = None
    if spec.kind in {"walk_forward", "train_test"} and primary_metric and is_all:
        _prim_pairs = [(r.get("ticker", ""), r.get("trade_date", ""))
                       for r in all_rows]
        _prim_by_m  = await _fetch_stored_bin20_by_metric(
            pool, spec.kind, [primary_metric], _prim_pairs)
        primary_bin20_by_key = _prim_by_m.get(primary_metric)

    filtered, dropped, universe = filter_by_assignments(
        all_rows, spec, primary_metric,
        req.selected_primary_bins, is_all, req.filtered_dates,
        primary_bin20_by_key=primary_bin20_by_key,
    )
    if not filtered:
        return {"error": "no_data",
                **mode_envelope(spec, dropped=dropped, universe=universe)}

    if spec.kind != "in_sample":
        ordered = filtered  # already chronologically sorted by the wf / tt filter
    else:
        # In-sample: the wrapper preserves cached-row order (legacy
        # `_filter_by_tkr_date` was unsorted). Re-sort here to match the
        # legacy explicit sort that fed _bin_membership.
        ordered = sorted(filtered, key=lambda r: (r.get("trade_date", ""), r.get("ticker", "")))
        universe = len(ordered)

    # Group 4: for IS+ALL, prefetch bin20 for every metric in the user's
    # selections. Selections naming a metric without a bin20_<m> column
    # (the 7 not-populated features) are silently dropped from the phi
    # matrix — no error envelope; matches the user-facing "skip the
    # missing minis" semantics.
    bin20_by_metric: dict = {}
    if spec.kind in {"in_sample", "walk_forward", "train_test"}:
        sel_metrics = [sel.get("metric", "") for sel in req.selections
                       if sel.get("metric")]
        if sel_metrics:
            pairs = [(r.get("ticker", ""), r.get("trade_date", ""))
                     for r in ordered]
            bin20_by_metric = await _fetch_stored_bin20_by_metric(
                pool, spec.kind, sel_metrics, pairs)

    vectors, metric_names, n_each = [], [], []
    for sel in req.selections:
        metric = sel.get("metric", "")
        bins   = set(sel.get("bins", []))
        if not metric or not bins:
            continue
        # Group 4: skip metrics without stored bin20 (auto-heals).
        if spec.kind in {"in_sample", "walk_forward", "train_test"} and metric not in bin20_by_metric:
            continue
        vec = secondary_membership(
            spec, ordered, metric, bins, n_bins, is_all,
            # v9: pass full row cache for in_sample too. The phi
            # correlation matrix's binary membership vectors are now
            # built against fixed full-population bin thresholds —
            # same shape as /secondary-detail and /secondary-corr-bins,
            # consistent with the heatmap.
            all_rows=all_rows,
            # Group 4: stored bin20 wins over all_rows for IS+ALL.
            bin20_by_key=bin20_by_metric.get(metric),
        )
        vectors.append(vec)
        metric_names.append(metric)
        n_each.append(int(vec.sum()))

    if len(vectors) < 2:
        return {"error": "insufficient_data"}

    M = np.array(vectors)                          # (n_metrics, n_rows)
    phi = np.nan_to_num(np.corrcoef(M), nan=0.0)
    overlap = (M @ M.T).astype(int)

    # Union: rows where at least one metric's binary vector = 1
    outcome_col = cached["outcome"]
    union_mask = np.any(M == 1, axis=0)
    combined_sorted = [ordered[i] for i, v in enumerate(union_mask) if v]

    eq_primary  = _sec_equity_curve(ordered, outcome_col)
    eq_combined = _sec_equity_curve(combined_sorted, outcome_col)

    yearly_primary: dict  = defaultdict(list)
    yearly_combined: dict = defaultdict(list)
    for r in ordered:
        yr = int(r.get("trade_date", "0000")[:4])
        o = r.get(outcome_col)
        if o is not None:
            yearly_primary[yr].append(float(o))
    for r in combined_sorted:
        yr = int(r.get("trade_date", "0000")[:4])
        o = r.get(outcome_col)
        if o is not None:
            yearly_combined[yr].append(float(o))

    all_years = sorted(set(yearly_primary) | set(yearly_combined))
    yearly_out = []
    for yr in all_years:
        p = yearly_primary.get(yr, [])
        c = yearly_combined.get(yr, [])
        yearly_out.append({
            "year":         yr,
            "primary_n":    len(p),
            "primary_avg":  round(float(np.mean(p)), 6) if p else 0,
            "primary_wr":   round(float(np.mean([1.0 if v > 0 else 0.0 for v in p])), 4) if p else 0,
            "combined_n":   len(c),
            "combined_avg": round(float(np.mean(c)), 6) if c else 0,
            "combined_wr":  round(float(np.mean([1.0 if v > 0 else 0.0 for v in c])), 4) if c else 0,
        })

    ticker_rets: dict = defaultdict(list)
    for r in combined_sorted:
        o = r.get(outcome_col)
        if o is not None:
            ticker_rets[r.get("ticker", "?")].append(float(o))
    total_pnl = sum(sum(v) for v in ticker_rets.values())
    tickers_out = []
    for tkr, rets in sorted(ticker_rets.items()):
        n_t = len(rets)
        avg_r = float(np.mean(rets)) if rets else 0.0
        wr = float(np.mean([1.0 if r > 0 else 0.0 for r in rets])) if rets else 0.0
        contrib = (sum(rets) / total_pnl * 100) if total_pnl != 0 else 0.0
        tickers_out.append({
            "ticker": tkr, "n": n_t,
            "avg_ret": round(avg_r, 6), "win_rate": round(wr, 4),
            "contrib_pct": round(contrib, 2),
        })

    # Winner / loser avg returns for union trades
    all_outcomes = [float(r[outcome_col]) for r in combined_sorted if r.get(outcome_col) is not None]
    winner_rets  = [v for v in all_outcomes if v > 0]
    loser_rets   = [v for v in all_outcomes if v <= 0]
    winner_avg   = round(float(np.mean(winner_rets)), 6) if winner_rets else 0.0
    loser_avg    = round(float(np.mean(loser_rets)),  6) if loser_rets  else 0.0

    horizon_n = _parse_horizon(outcome_col)
    primary_m = cached.get("primary_metric")
    calendars = cached.get("calendars") or {}
    combined_trades = [
        _build_enriched_trade(
            r, calendars, horizon_n,
            primary_metric=primary_m,
            outcome_col=outcome_col,
            extra_metrics=metric_names,    # include each selected secondary's value
        )
        for r in combined_sorted
    ]
    return {
        "metrics": metric_names,
        "n_each":  n_each,
        "phi":     [[round(float(v), 4) for v in row] for row in phi],
        "overlap": [[int(v) for v in row] for row in overlap],
        "baseline_n":  len(ordered),
        "combined_n":  len(combined_sorted),
        "horizon":     horizon_n,
        "equity_primary":        eq_primary,
        "equity_combined":       eq_combined,
        "yearly":                yearly_out,
        "tickers":               tickers_out,
        "combined_trade_dates":  [r.get("trade_date", "") for r in combined_sorted],
        "combined_trades":       combined_trades,
        "winner_avg_ret": winner_avg,
        "loser_avg_ret":  loser_avg,
        **mode_envelope(spec, dropped=dropped, universe=universe,
                        start_date=ordered[0].get("trade_date", "") if ordered else ""),
    }


# ── Global Metric Bins (standalone top-of-page browser) ───────────────────
#
# DISCIPLINE: bump _GLOBAL_BINS_SCHEMA_VERSION on every change to the
# bin methodology or response shape so stale cache entries are
# automatically invalidated on next read.
#
# v1: introducing the salt.
# v2 (Group 7): WF mode claimed to read wf_bins — comment was wrong,
#   assign_batch still computed on-the-fly.
# v3 (Group 8): TT mode claimed to read tt_bins — same bug.
# v4: Full migration to stored bins for ALL modes.  assign_batch
#   eliminated.  IS reads is_bins, WF reads wf_bins, TT reads tt_bins.
#   Date filters now apply to OUTCOME ROWS only; bin assignments are
#   full-history stored values unaffected by the date window.  Metrics
#   with no bin20_{metric} column produce no entry (no fallback).
_GLOBAL_BINS_SCHEMA_VERSION = 5  # bumped: family-filter exclusion applied to available_metrics

_GLOBAL_BINS_CACHE: dict = {}
_GLOBAL_BINS_TABLE_DDL = """\
CREATE TABLE IF NOT EXISTS global_bins_cache (
    cache_key TEXT PRIMARY KEY,
    outcome   TEXT NOT NULL,
    ticker    TEXT NOT NULL,
    n_bins    INT  NOT NULL,
    mode      TEXT NOT NULL,
    payload   JSONB NOT NULL,
    cached_at TIMESTAMPTZ DEFAULT NOW()
);
"""
_bins_table_ensured = False


async def _ensure_bins_table(pool) -> None:
    global _bins_table_ensured
    if _bins_table_ensured:
        return
    async with pool.acquire() as conn:
        await conn.execute(_GLOBAL_BINS_TABLE_DDL)
        # Sweep entries from any prior schema version. The salt makes them
        # unreachable on read; this DELETE reclaims their disk. No-op once
        # the table only holds current-version entries. Same pattern as
        # _ensure_analyze_primary_table.
        prefix = f"gb:v{_GLOBAL_BINS_SCHEMA_VERSION}:"
        await conn.execute(
            "DELETE FROM global_bins_cache WHERE cache_key NOT LIKE $1",
            f"{prefix}%",
        )
    _bins_table_ensured = True


@router.get("/global-metric-bins")
async def global_metric_bins(
    outcome:      str  = Query("ret_5d_fwd_oc"),
    ticker:       str  = Query("ALL"),
    n_bins:       int  = Query(20, ge=2, le=20),
    date_from:    Optional[str] = Query(None),
    date_to:      Optional[str] = Query(None),
    walk_forward: bool = Query(False),
    cutoff_date:  Optional[str] = Query(None),
    force:        bool = Query(False, description="Skip cache read; recompute and overwrite"),
    pool=Depends(get_oi_pool),
):
    """Per-bin avg return for every feature column at the given outcome, with
    no primary filter. Used by the standalone "All-Ticker Metric Bins" pane
    at the top of the OI Analysis page.

    For `ticker = ALL` each ticker is independently ranked into n_bins then
    pooled (per-ticker rank normalization). For a single ticker, flat rank.
    """
    if not pool:
        return {"error": "OI database not configured"}
    n_bins = max(2, min(20, n_bins))
    # Cache key includes the mode tag so in-sample / walk-forward / train-test
    # results don't collide. Train-test also includes the cutoff date —
    # different cutoffs produce different bin assignments. The `gb:vN:`
    # prefix is the schema-version salt (see _GLOBAL_BINS_SCHEMA_VERSION
    # at the top of this section); bumping the constant auto-invalidates
    # every stale cached payload on next read.
    if cutoff_date:
        mode_tag = f"tt:{cutoff_date}"
    elif walk_forward:
        mode_tag = "wf"
    else:
        mode_tag = "is"
    cache_key = (
        f"gb:v{_GLOBAL_BINS_SCHEMA_VERSION}:"
        f"{ticker}|{outcome}|{n_bins}|{date_from or ''}|{date_to or ''}|{mode_tag}"
    )
    if not force and cache_key in _GLOBAL_BINS_CACHE:
        return _GLOBAL_BINS_CACHE[cache_key]

    # Check persistent DB cache before running the expensive computation.
    # asyncpg returns JSONB as a string by default (no codec is registered
    # in app/db.py), so `payload` needs an explicit json.loads — the old
    # code did `dict(db_row["payload"])` which silently failed on the
    # string and fell through to recompute every time the in-memory cache
    # was cold. Pre-fix, the DB cache only ever served data that had
    # already been computed in the current process lifetime.
    await _ensure_bins_table(pool)
    if not force:
        try:
            async with pool.acquire() as conn:
                db_row = await conn.fetchrow(
                    "SELECT payload, cached_at FROM global_bins_cache WHERE cache_key = $1",
                    cache_key)
            if db_row:
                payload = db_row["payload"]
                if isinstance(payload, str):
                    payload = json.loads(payload)
                out = dict(payload)
                ca = db_row["cached_at"]
                out["cached_at"] = ca.isoformat() if ca else None
                _GLOBAL_BINS_CACHE[cache_key] = out
                return out
        except Exception as e:
            # Log instead of silently swallowing — masked the JSONB parse
            # failure for months. Still tolerate the failure so first-startup
            # (table doesn't exist) falls through to compute cleanly.
            import logging
            logging.warning("global_bins_cache DB read failed for %s: %r", cache_key, e)

    # ── Mode and bin table dispatch ─────────────────────────────────────
    # assign_batch eliminated.  Source table determined by mode alone;
    # stored bin20 is the full-history rank for every row.
    if cutoff_date:
        mode = "train_test"
        bin_table = "tt_bins"
    elif walk_forward:
        mode = "walk_forward"
        bin_table = "wf_bins"
    else:
        mode = "in_sample"
        bin_table = "is_bins"

    # ── Discover metrics present in the bin table ────────────────────────
    # Metrics without a bin20_{metric} column produce no output entry.
    # No fallback computation — missing bin column means show nothing.
    try:
        async with pool.acquire() as conn:
            bin_col_rows = await conn.fetch(
                """SELECT column_name FROM information_schema.columns
                   WHERE table_name = $1 AND table_schema = 'public'
                     AND column_name LIKE 'bin20_%'
                   ORDER BY ordinal_position""",
                bin_table)
    except Exception as _col_exc:
        import logging
        logging.warning("global_metric_bins: can't read %s columns: %r", bin_table, _col_exc)
        return {
            "error":   f"Bin table {bin_table} not accessible: {_col_exc}",
            "outcome": outcome, "ticker": ticker, "n_bins": n_bins,
            "metrics": [], "total_rows": 0,
        }

    available_metrics = [r["column_name"][6:] for r in bin_col_rows]  # strip "bin20_"

    # Display-only filter: exclude Family-2 metrics (spot_pc / spot_co) and
    # _pc-suffixed metrics in Family 4 or 5 (OI-by-strike / OI-change families).
    # Family-12 _pc metrics (option volume) are kept — exclusion is family-scoped.
    # Matches the same filter applied in /corner-scan/1f, /corner-scan/2f, and
    # /corner-scan/meta.  Graceful degradation: if metric_classification is absent
    # the set is empty and no metrics are filtered.
    try:
        async with pool.acquire() as conn:
            excl_rows = await conn.fetch(
                """SELECT metric FROM metric_classification
                   WHERE family_num = 2
                      OR (family_num IN (4,5) AND RIGHT(metric,3) = '_pc')"""
            )
        _excl_set = {r["metric"] for r in excl_rows}
    except Exception:
        _excl_set = set()
    if _excl_set:
        available_metrics = [m for m in available_metrics if m not in _excl_set]

    if not available_metrics:
        out = {"outcome": outcome, "ticker": ticker, "n_bins": n_bins,
               "metrics": [], "total_rows": 0, "metrics_attempted": 0, "mode": mode}
        _GLOBAL_BINS_CACHE[cache_key] = out
        return out

    bin20_select = ", ".join(f"bt.bin20_{m}" for m in available_metrics)

    # ── Single JOIN query ────────────────────────────────────────────────
    # Date filters restrict outcome rows only — stored bin assignments
    # are full-history values unaffected by the date window.  No outer
    # WHERE bin20 > 0: warmup/NaN sentinels are filtered per-metric in
    # the aggregation loop, so a row with bin20_A=0 can still contribute
    # an outcome to metric B.
    where: list = [f"df.{outcome} IS NOT NULL"]
    params: list = []
    p = 1
    if ticker and ticker != "ALL":
        where.append(f"df.ticker = ${p}"); params.append(ticker); p += 1
    if date_from:
        where.append(f"df.trade_date >= ${p}")
        params.append(_date.fromisoformat(date_from)); p += 1
    if date_to:
        where.append(f"df.trade_date <= ${p}")
        params.append(_date.fromisoformat(date_to)); p += 1
    if cutoff_date:
        # TT: aggregate outcomes from the test window only
        where.append(f"df.trade_date >= ${p}")
        params.append(_date.fromisoformat(cutoff_date)); p += 1

    try:
        async with pool.acquire() as conn:
            db_rows = await conn.fetch(
                f"SELECT df.ticker, df.trade_date, df.{outcome}, "
                f"  {bin20_select} "
                f"FROM daily_features df "
                f"JOIN {bin_table} bt USING (ticker, trade_date) "
                f"WHERE {' AND '.join(where)} "
                f"ORDER BY df.ticker, df.trade_date",
                *params, timeout=240)
        rows = [dict(r) for r in db_rows]
        if not rows:
            out = {"outcome": outcome, "ticker": ticker, "n_bins": n_bins,
                   "metrics": [], "total_rows": 0,
                   "metrics_attempted": len(available_metrics), "mode": mode}
            _GLOBAL_BINS_CACHE[cache_key] = out
            return out

        # ── Per-metric aggregation ────────────────────────────────────
        # Canonical bin20 → n_bins collapse (0-indexed, same formula as
        # _batch_metrics_from_stored_bin20 and every other stored-bin
        # surface in this file):
        #   bn = min(((b20 - 1) * n_bins) // 20, n_bins - 1)
        #
        # No thinning gate.  Sparse metrics appear with low bin_ns counts;
        # the user can see that directly.  Every metric with any stored
        # bin20 rows is included.
        metrics_out: list = []
        for metric in available_metrics:
            col = f"bin20_{metric}"
            buckets: list = [[] for _ in range(n_bins)]
            for r in rows:
                b20 = r.get(col)
                if not b20 or b20 <= 0:      # warmup / NaN sentinel
                    continue
                bn = min(((b20 - 1) * n_bins) // 20, n_bins - 1)
                buckets[bn].append(float(r[outcome]))
            metrics_out.append({
                "name":   metric,
                "bins":   [round(float(np.mean(b)), 6) if b else 0.0
                           for b in buckets],
                "bin_ns": [len(b) for b in buckets],
            })

        # Sort by lift over populated buckets only (empty buckets are 0.0
        # by convention and excluded so they don't inflate or deflate lift).
        def _lift(m):
            nonempty = [v for v, n in zip(m.get("bins") or [],
                                          m.get("bin_ns") or []) if n > 0]
            return (max(nonempty) - min(nonempty)) if len(nonempty) >= 2 else 0
        metrics_out.sort(key=_lift, reverse=True)

        # First row's trade_date = first non-warmup date (informational)
        start_date_str: Optional[str] = None
        sd = rows[0].get("trade_date")
        if sd is not None:
            start_date_str = sd.isoformat() if hasattr(sd, "isoformat") else str(sd)

        out: dict = {
            "outcome":           outcome,
            "ticker":            ticker,
            "n_bins":            n_bins,
            "total_rows":        len(rows),
            "metrics_attempted": len(available_metrics),
            "metrics":           metrics_out,
            "mode":              mode,
        }
        if mode == "walk_forward":
            out["warmup"]     = 252           # wf_bins warmup period
            out["start_date"] = start_date_str
        elif mode == "train_test":
            out["cutoff_date"] = cutoff_date
            out["start_date"]  = start_date_str

        # ── Persist to DB cache ─────────────────────────────────────────
        try:
            global _bins_table_ensured
            _bins_table_ensured = False
            await _ensure_bins_table(pool)
            async with pool.acquire() as conn:
                await conn.execute(
                    "INSERT INTO global_bins_cache "
                    "    (cache_key, outcome, ticker, n_bins, mode, payload) "
                    "VALUES ($1,$2,$3,$4,$5,$6::jsonb) "
                    "ON CONFLICT (cache_key) DO UPDATE "
                    "    SET payload=$6::jsonb, cached_at=NOW()",
                    cache_key, outcome, ticker, n_bins, mode, json.dumps(out),
                )
                row_ca = await conn.fetchrow(
                    "SELECT cached_at FROM global_bins_cache WHERE cache_key = $1",
                    cache_key)
            out["cached_at"] = row_ca["cached_at"].isoformat() if row_ca else None
        except Exception as _write_exc:
            import logging
            logging.warning("global_bins_cache DB write failed for %s: %r",
                            cache_key, _write_exc)
            out["cached_at"] = None

        _GLOBAL_BINS_CACHE[cache_key] = out
        return out
    except Exception as exc:
        return {
            "error":      f"{type(exc).__name__}: {exc}",
            "outcome":    outcome,
            "ticker":     ticker,
            "n_bins":     n_bins,
            "metrics":    [],
            "total_rows": 0,
        }


@router.post("/global-metric-bins/invalidate")
async def global_metric_bins_invalidate(pool=Depends(get_oi_pool)):
    """Drop the in-memory and DB cache so a fresh computation runs next request.
    Call this only for bulk invalidation (e.g. underlying data was rewritten),
    not as part of the routine UI Refresh flow — use force=true on the GET instead."""
    _GLOBAL_BINS_CACHE.clear()
    if pool:
        try:
            await _ensure_bins_table(pool)
            async with pool.acquire() as conn:
                await conn.execute("DELETE FROM global_bins_cache")
        except Exception:
            import logging
            logging.warning("global_bins_cache DELETE failed during invalidate", exc_info=True)
    return {"ok": True}


@router.get("/global-metric-bins/meta")
async def global_metric_bins_meta(
    outcome:      str           = Query("ret_5d_fwd_oc"),
    ticker:       str           = Query("ALL"),
    n_bins:       int           = Query(20, ge=2, le=20),
    date_from:    Optional[str] = Query(None),
    date_to:      Optional[str] = Query(None),
    walk_forward: bool          = Query(False),
    cutoff_date:  Optional[str] = Query(None),
    pool=Depends(get_oi_pool),
):
    """Cheap metadata-only check for global_bins_cache.

    Returns `{exists, cached_at}` for the given selector tuple. The
    SELECT pulls only `cached_at` — never the ~MB payload column — so
    this is sub-millisecond regardless of cache state. Powers the
    collapsed-pane breadcrumb so the All-Ticker Metric Bins pane shows
    its last-computed timestamp on page load without re-fetching the
    full bins.
    """
    if not pool:
        return {"error": "OI database not configured"}
    n_bins = max(2, min(20, n_bins))
    if cutoff_date:
        mode_tag = f"tt:{cutoff_date}"
    elif walk_forward:
        mode_tag = "wf"
    else:
        mode_tag = "is"
    cache_key = (f"{ticker}|{outcome}|{n_bins}|"
                 f"{date_from or ''}|{date_to or ''}|{mode_tag}")
    try:
        await _ensure_bins_table(pool)
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT cached_at FROM global_bins_cache WHERE cache_key = $1",
                cache_key,
            )
    except Exception:
        return {"exists": False, "cached_at": None}
    if row is None:
        return {"exists": False, "cached_at": None}
    ca = row["cached_at"]
    return {
        "exists":    True,
        "cached_at": ca.isoformat() if ca else None,
    }


# ── IC stability batch (IC.4) ────────────────────────────────────────────
# Per-metric long-run IC + sign-stability across all ~123 daily_features
# columns. Powers the IC.5 leaderboard and strength-vs-stability scatter.
# DB-cached on a (ticker, outcome, window, mode/cutoff) key — mirrors the
# /global-metric-bins read-through cache pattern. CPU-heavy compute runs
# via asyncio.to_thread so the FastAPI event loop stays responsive while
# the per-metric rolling-IC loop executes.

_IC_BATCH_TABLE_DDL = """\
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
_ic_batch_table_ensured = False

# Background-job tracking — in-memory only; cleared on server restart.
# The frontend's 15-min poll timeout covers sessions mid-poll at restart time.
_ic_batch_running: set = set()   # cache_key values currently being computed
_ic_batch_status: dict = {}      # cache_key → {"status": "failed", "error": "..."}
_IC_DECOMP_CACHE:  dict = {}     # IC.7: cache_key → decomp result (in-memory, cleared on restart)


async def _ensure_ic_batch_table(pool) -> None:
    global _ic_batch_table_ensured
    if _ic_batch_table_ensured:
        return
    async with pool.acquire() as conn:
        await conn.execute(_IC_BATCH_TABLE_DDL)
    _ic_batch_table_ensured = True


# ── analyze_cache (P1) ───────────────────────────────────────────────────
# 12-outcome bundle cache, split across three tables since v6 to support
# lazy-loading on the frontend. Each ALL-mode (ticker, metric, mode, cutoff)
# bundle is stored as:
#   analyze_cache_slim        — 1 row, ~1.3 MB: per_bin + rolling_ic + meta
#   analyze_cache_trade_meta  — 1 row, ~41 MB:  the trade_meta array
#   analyze_cache_outcome     — 12 rows, ~7.4 MB each: per_outcome_returns
#                                                       keyed by outcome
# Reads target one table at a time so the frontend pays only for what it
# needs (slim eagerly, trade_meta + outcomes on demand). Writes happen in
# a single transaction so readers never see a partial bundle. Single-ticker
# bundles compute inline at /analyze-bundle and never reach these tables.
#
# LRU eviction: sum pg_total_relation_size across all 3 tables, drop the
# oldest cache_key (by analyze_cache_slim.last_accessed) from all 3 in a
# single transaction. Cap stays 5 GB total across the family. last_accessed
# lives only on the slim table; every read endpoint touches it so a
# deeply-explored bundle stays warm.
#
# Old v5 `analyze_cache` table: untouched by v6 code. v5 cache keys are
# unreachable since the schema-version salt changed (5 → 6). Drop the
# legacy table manually after confirming v6 works.

_ANALYZE_CACHE_SLIM_TABLE_DDL = """\
CREATE TABLE IF NOT EXISTS analyze_cache_slim (
    cache_key     TEXT  PRIMARY KEY,
    ticker        TEXT  NOT NULL,
    metric        TEXT  NOT NULL,
    mode          TEXT  NOT NULL,
    cutoff_date   DATE,
    payload       JSONB NOT NULL,
    payload_bytes INT,
    cached_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_accessed TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS analyze_cache_slim_last_accessed
    ON analyze_cache_slim (last_accessed);
"""

_ANALYZE_CACHE_TRADE_META_TABLE_DDL = """\
CREATE TABLE IF NOT EXISTS analyze_cache_trade_meta (
    cache_key     TEXT  PRIMARY KEY,
    payload       JSONB NOT NULL,
    payload_bytes INT,
    cached_at     TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
"""

_ANALYZE_CACHE_OUTCOME_TABLE_DDL = """\
CREATE TABLE IF NOT EXISTS analyze_cache_outcome (
    cache_key     TEXT  NOT NULL,
    outcome       TEXT  NOT NULL,
    payload       JSONB NOT NULL,
    payload_bytes INT,
    cached_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (cache_key, outcome)
);
"""

_analyze_bundle_table_ensured = False

# Background-job tracking — in-memory only; cleared on server restart.
# Frontend's 15-min poll timeout covers sessions mid-poll at restart time.
_analyze_bundle_running: set = set()   # cache_keys currently computing
_analyze_bundle_status:  dict = {}     # cache_key → {"status":"failed","error":"…"}

_ANALYZE_BUNDLE_CACHE_MAX_BYTES = 5 * 1024**3   # 5 GB cap (LRU eviction)
# The set of outcome columns is discovered at runtime (mirrors /columns):
# any numeric daily_features column matching `ret_*_fwd*`. This way the
# bundle survives a schema change (new horizons added, anchor suffix
# changed, etc.) without a code edit. The discovered list is recorded
# on the bundle in the `outcomes` field — consumers MUST iterate that
# field rather than assuming a fixed set.
_ANALYZE_BUNDLE_SCHEMA_VERSION = 13  # v13 (Group 8 follow-up): bundle's _assignments_from_is_bins now skips pre-cutoff rows when mode == "train_test". Pre-fix v12 TT bundles included full-series rows in trade_meta + per_outcome_returns, which the primary chart consumed verbatim under a "TEST PERIOD" label. Bump invalidates v12. v12 history: TT+ALL bundle compute now reads tt_bins.bin20_{metric}. TT methodology shifted from on-the-fly walk-forward-frozen to stored in-sample-frozen-at-cutoff. v11 TT cache_keys carry the old methodology and are unreachable on read; one-time recompute per (ticker, metric, mode, cutoff) on first read. v11 history: WF+ALL bundle compute now reads wf_bins.bin20_{metric} via _fetch_wfbin20_by_metric — same shape as v9's is_bins migration for IS+ALL, applied to the walk-forward mode. v10 cache_keys carry on-the-fly WF bins and are unreachable on read; one-time recompute per (ticker, metric, mode, cutoff) on first read. v10 history: /analyze-bundle no longer applies the read-time per-ticker thinning gate that was removed alongside the same gate in /analyze and /global-metric-bins. Cached v9 bundles are unreachable on read (different salt); first read after deploy triggers a one-time recompute per (ticker, metric, mode, cutoff). v9 (Group 3b) history: for IS+ALL the bundle's bin_20 now comes from is_bins.bin20_{metric} (read at bundle-compute time and threaded into _compute_analyze_bundle_sync) instead of the on-the-fly InSampleAssigner. This is the consistency-by-construction shift — same stored bin reaches the bundle's per_bin / trade_meta / per_outcome_returns as reaches /heatmap, /metric-bins, and /analyze (Group 3a). WF and TT stay on the Assigner path for this round (deferred to Groups 7–8). v8 cache_keys are unreachable on read; one-time recompute on first read of each (ticker, metric, mode, cutoff) after deploy. v8 history (kept for context): bumped from 7 — gap outcome carries flat-table fields inline so Gap-mode flat trade table no longer triggers the 56 MB trade_meta + 1d_cc + 1d_oc fetch path.


async def _ensure_analyze_bundle_table(pool) -> None:
    """Create the v6 3-table family. Idempotent. The legacy `analyze_cache`
    table (v5 and earlier) is intentionally NOT touched — v5 cache keys are
    already unreachable due to the schema-version salt change. Drop it
    manually post-deploy when comfortable."""
    global _analyze_bundle_table_ensured
    if _analyze_bundle_table_ensured:
        return
    async with pool.acquire() as conn:
        await conn.execute(_ANALYZE_CACHE_SLIM_TABLE_DDL)
        await conn.execute(_ANALYZE_CACHE_TRADE_META_TABLE_DDL)
        await conn.execute(_ANALYZE_CACHE_OUTCOME_TABLE_DDL)
        # Wipe entries from any prior schema version across all 3 tables.
        # Cache keys are salted (see _analyze_bundle_cache_key), so stale
        # entries are already unreachable on read but consume disk until
        # this DELETE reclaims them. No-op once tables hold only current-
        # version entries.
        prefix = f"ab:v{_ANALYZE_BUNDLE_SCHEMA_VERSION}:"
        async with conn.transaction():
            await conn.execute(
                "DELETE FROM analyze_cache_slim WHERE cache_key NOT LIKE $1",
                f"{prefix}%",
            )
            await conn.execute(
                "DELETE FROM analyze_cache_trade_meta WHERE cache_key NOT LIKE $1",
                f"{prefix}%",
            )
            await conn.execute(
                "DELETE FROM analyze_cache_outcome WHERE cache_key NOT LIKE $1",
                f"{prefix}%",
            )
    _analyze_bundle_table_ensured = True


def _analyze_bundle_cache_key(ticker: str, metric: str, mode: str,
                              cutoff_date: Optional[str]) -> str:
    """Stable cache key for (ticker, metric, mode, cutoff). Salted with the
    schema version so a future bundle-shape change automatically invalidates
    every stale entry on next read."""
    cutoff_s = cutoff_date or "null"
    return f"ab:v{_ANALYZE_BUNDLE_SCHEMA_VERSION}:{ticker}:{metric}:{mode}:{cutoff_s}"


async def _evict_analyze_cache_lru(pool) -> int:
    """Drop oldest-accessed cache_keys until the combined disk footprint of
    the 3 analyze_cache_* tables falls back under the cap. Each evicted
    cache_key clears its row from all 3 tables in a single transaction.
    Returns the count of cache_keys evicted. Called from the write path
    after each successful UPSERT.
    """
    evicted = 0
    async with pool.acquire() as conn:
        while True:
            size_bytes = await conn.fetchval("""
                SELECT pg_total_relation_size('analyze_cache_slim')
                     + pg_total_relation_size('analyze_cache_trade_meta')
                     + pg_total_relation_size('analyze_cache_outcome')
            """)
            if size_bytes is None or size_bytes <= _ANALYZE_BUNDLE_CACHE_MAX_BYTES:
                break
            oldest_key = await conn.fetchval(
                "SELECT cache_key FROM analyze_cache_slim "
                "ORDER BY last_accessed ASC LIMIT 1")
            if oldest_key is None:
                break   # tables empty but total size still > cap (TOAST bloat)
            async with conn.transaction():
                await conn.execute(
                    "DELETE FROM analyze_cache_slim WHERE cache_key = $1",
                    oldest_key)
                await conn.execute(
                    "DELETE FROM analyze_cache_trade_meta WHERE cache_key = $1",
                    oldest_key)
                await conn.execute(
                    "DELETE FROM analyze_cache_outcome WHERE cache_key = $1",
                    oldest_key)
            evicted += 1
    return evicted


async def _fetch_ic_feature_columns(pool) -> list[str]:
    """Mirror /columns: numeric daily_features columns excluding outcomes."""
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """SELECT column_name FROM information_schema.columns
               WHERE table_name = 'daily_features' AND table_schema = 'public'
               AND data_type IN ('double precision','numeric','real',
                                 'integer','bigint','smallint')
               AND column_name NOT IN ('id','ticker','trade_date',
                                       'created_at','updated_at')
               ORDER BY ordinal_position""")
    all_cols = [r["column_name"] for r in rows]
    outcomes = {c for c in all_cols if "ret_" in c and "fwd" in c}
    return [c for c in all_cols if c not in outcomes and not c.endswith("_pc")]


def _compute_ic_batch_sync(
    rows: list[dict],
    feature_cols: list[str],
    outcome: str,
    window: int,
    cutoff_date: Optional[str],
    horizon: int,
    stride: int = 1,
) -> list[dict]:
    """Per-metric IC computation for a SINGLE TICKER. Pure-sync, no DB.
    Off-loaded via asyncio.to_thread.

    ALL-mode is handled by _compute_ic_batch_all_bg, which fetches one metric
    at a time (4 cols × 141K rows ≈ 4 MB peak) to avoid VPS OOM. The bulk
    ALL-mode block that lived here was removed in IC.5 step-2 — it caused
    6 OOM kills (python at 3.6–5 GB each) by loading 125 cols × 141K rows.

    For each feature column: rolling Spearman IC against the outcome over
    a `window`-day trailing window (single-ticker time-series IC), then
    mode-aware reference IC, ε, and sign-stability.

    The outcome column is skipped if present in feature_cols. Returns one
    dict per emitted metric, in the input feature_cols order.
    """
    from app.routers.ic_compute import (
        rolling_ic_single_ticker,
        sign_stability_from_rolling, noise_floor_epsilon,
        finite_or_none,
    )

    results = []
    for metric in feature_cols:
        if metric == outcome:
            continue

        series = rolling_ic_single_ticker(
            rows, metric, outcome, window=window, stride=stride,
        )
        epsilon = noise_floor_epsilon(
            "single_ticker", window=window, horizon=horizon,
        )

        if cutoff_date:
            pre_cutoff = [p.ic for p in series if str(p.date) < cutoff_date]
            reference_ic = float(np.mean(pre_cutoff)) if pre_cutoff else 0.0
        elif series:
            reference_ic = float(np.mean([p.ic for p in series]))
        else:
            reference_ic = 0.0

        stability = sign_stability_from_rolling(series, reference_ic, epsilon)
        n_total = stability.n_total

        results.append({
            "name":            metric,
            "long_run_ic":     finite_or_none(reference_ic),
            "long_run_ic_abs": finite_or_none(abs(reference_ic)),
            "epsilon":         finite_or_none(epsilon),
            "n_windows":       n_total,
            "sign_stability":  (finite_or_none(stability.stability, 4)
                                if stability.stability is not None else None),
            "n_same":          stability.n_same,
            "n_opposite":      stability.n_opposite,
            "n_neutral":       stability.n_neutral,
            "neutral_pct":     (round(100.0 * stability.n_neutral / n_total, 2)
                                if n_total else 0.0),
            "suppressed":      stability.suppressed,
            "suppression_reason": stability.suppression_reason,
        })

    return results


async def _compute_ic_batch_single_bg(
    cache_key: str,
    ticker: str,
    outcome: str,
    window: int,
    cutoff_date: Optional[str],
    stride: int,
    pool,
) -> None:
    """Background coroutine: compute single-ticker IC batch off the event loop.

    Fetches all feature columns for this ticker in one bulk SQL query (~125
    cols × ~1250 rows ≈ lightweight), then off-loads the per-metric
    rolling-Spearman loop via asyncio.to_thread(_compute_ic_batch_sync).

    Infinity/NaN safety: _compute_ic_batch_sync calls finite_or_none() on
    every float field before returning, so json.dumps() on the result never
    encounters inf/nan. Single-ticker noise_floor_epsilon never returns +inf
    (it requires k_tickers < 2 which is a cross-sectional-only condition),
    but the finite_or_none cover is there defensively.

    Server-restart safety: asyncio.create_task() tasks live inside the
    process event loop. A server restart kills the process, which kills the
    task — there is no zombie-task risk. _ic_batch_running and
    _ic_batch_status are destroyed and reset with the process, so after a
    restart GET returns not_ready and the single-ticker auto-trigger starts a
    fresh job. The only loss is the in-flight result, which was never written
    to the DB. The 15-min frontend poll timeout covers the edge case of a
    restart mid-poll-session.
    """
    import logging as _log
    from datetime import date as _date

    horizon = _parse_horizon(outcome)

    try:
        feature_cols = await _fetch_ic_feature_columns(pool)

        cols_sql = ", ".join(f'"{c}"' for c in feature_cols) + f', "{outcome}"'
        sql = (
            f'SELECT trade_date, {cols_sql} FROM daily_features '
            f'WHERE ticker = $1 AND "{outcome}" IS NOT NULL '
            f'ORDER BY trade_date'
        )
        async with pool.acquire() as conn:
            db_rows = await conn.fetch(sql, ticker, timeout=90)
        rows = [dict(r) for r in db_rows]
        for r in rows:
            r.setdefault("ticker", ticker)

        # CPU-heavy per-metric Spearman loop runs on a thread — keeps the
        # event loop free. finite_or_none() is called inside
        # _compute_ic_batch_sync on every float field, so the payload is
        # json.dumps()-safe even if any value is inf/nan.
        results = await asyncio.to_thread(
            _compute_ic_batch_sync,
            rows, feature_cols, outcome, window, cutoff_date, horizon, stride,
        )
        del rows

        # Write completed result to cache.
        payload_json = json.dumps({"metrics": results})
        cutoff_obj = _date.fromisoformat(cutoff_date) if cutoff_date else None
        async with pool.acquire() as conn:
            await conn.execute(
                """INSERT INTO ic_batch_cache
                   (cache_key, ticker, outcome, window_size, cutoff_date, payload, cached_at)
                   VALUES ($1, $2, $3, $4, $5, $6::jsonb, NOW())
                   ON CONFLICT (cache_key) DO UPDATE
                   SET payload    = EXCLUDED.payload,
                       cached_at  = NOW()""",
                cache_key, ticker, outcome, window, cutoff_obj, payload_json,
            )
        _log.info(
            "ic_batch_single_bg: wrote %d metrics to cache key=%s",
            len(results), cache_key,
        )

    except Exception as exc:
        import logging as _log2
        _log2.exception("ic_batch_single_bg: fatal error key=%s", cache_key)
        _ic_batch_status[cache_key] = {
            "status": "failed",
            "error":  f"{type(exc).__name__}: {exc}",
        }

    finally:
        # Always release the slot — even on crash or cancellation.
        # If the process is killed (server restart), the finally block does
        # not run, but _ic_batch_running is destroyed with the process anyway.
        _ic_batch_running.discard(cache_key)


async def _compute_ic_batch_all_bg(
    cache_key: str,
    ticker: str,
    outcome: str,
    window: int,
    cutoff_date: Optional[str],
    stride: int,
    pool,
) -> None:
    """Background coroutine: compute ALL-mode IC batch one metric at a time.

    Fetches only 4 columns per metric (ticker, trade_date, metric, outcome) so
    peak RAM stays at ~4 MB per metric rather than ~800 MB for a 125-col bulk
    fetch. Writes the completed payload to ic_batch_cache on success; records
    {"status": "failed"} in _ic_batch_status on any unhandled exception.
    Always discards cache_key from _ic_batch_running in the finally block.
    """
    import logging as _log
    import time as _time
    from datetime import date as _date
    from app.routers.ic_compute import (
        rolling_ic_cross_sectional,
        sign_stability_from_rolling,
        noise_floor_epsilon,
        finite_or_none,
        ic_decompose_cross_sectional,   # IC.7: breadth / gini per metric
    )

    horizon = _parse_horizon(outcome)

    try:
        _t0 = _time.perf_counter()
        feature_cols = await _fetch_ic_feature_columns(pool)
        metrics = [c for c in feature_cols if c != outcome]
        results = []

        for metric in metrics:
            try:
                async with pool.acquire() as conn:
                    db_rows = await conn.fetch(
                        f'SELECT ticker, trade_date, "{metric}", "{outcome}" '
                        f'FROM daily_features '
                        f'WHERE "{outcome}" IS NOT NULL '
                        f'ORDER BY trade_date, ticker',
                        timeout=60,
                    )
                rows = [dict(r) for r in db_rows]
            except Exception as _fetch_exc:
                _log.warning(
                    "ic_batch_all_bg: DB fetch failed metric=%s key=%s: %r",
                    metric, cache_key, _fetch_exc,
                )
                continue

            # CPU-heavy Spearman in a thread to keep the event loop free.
            series = await asyncio.to_thread(
                rolling_ic_cross_sectional,
                rows, metric, outcome,
                window=window,
                stride=stride,
            )

            # IC.7: per-ticker decomp on same rows → gini + effective_n.
            # Runs before del rows so no extra DB fetch is needed.
            decomp = await asyncio.to_thread(
                ic_decompose_cross_sectional,
                rows, metric, outcome,
                cutoff_date=cutoff_date,
            )
            del rows  # release before next iteration

            if series:
                if cutoff_date:
                    pre = [p.ic for p in series if str(p.date) < cutoff_date]
                    reference_ic = float(np.mean(pre)) if pre else 0.0
                else:
                    reference_ic = float(np.mean([p.ic for p in series]))
                median_k = int(np.median([p.n for p in series]))
            else:
                reference_ic = 0.0
                median_k = 0

            epsilon = noise_floor_epsilon(
                "cross_sectional", window=window,
                horizon=horizon, k_tickers=median_k,
            )
            stability = sign_stability_from_rolling(series, reference_ic, epsilon)
            n_total = stability.n_total

            results.append({
                "name":            metric,
                "long_run_ic":     finite_or_none(reference_ic),
                "long_run_ic_abs": finite_or_none(abs(reference_ic)),
                "epsilon":         finite_or_none(epsilon),
                "n_windows":       n_total,
                "sign_stability":  (finite_or_none(stability.stability, 4)
                                    if stability.stability is not None else None),
                "n_same":          stability.n_same,
                "n_opposite":      stability.n_opposite,
                "n_neutral":       stability.n_neutral,
                "neutral_pct":     (round(100.0 * stability.n_neutral / n_total, 2)
                                    if n_total else 0.0),
                "suppressed":      stability.suppressed,
                "suppression_reason": stability.suppression_reason,
                # IC.7 breadth fields — None if decomp returned no tickers
                "concentration_gini": finite_or_none(decomp["concentration_gini"])
                                      if decomp.get("concentration_gini") is not None else None,
                "effective_n":        finite_or_none(decomp["effective_n"])
                                      if decomp.get("effective_n") is not None else None,
            })

        # Write completed result to cache.
        payload_json = json.dumps({"metrics": results})
        cutoff_obj = _date.fromisoformat(cutoff_date) if cutoff_date else None
        async with pool.acquire() as conn:
            await conn.execute(
                """INSERT INTO ic_batch_cache
                   (cache_key, ticker, outcome, window_size, cutoff_date, payload, cached_at)
                   VALUES ($1, $2, $3, $4, $5, $6::jsonb, NOW())
                   ON CONFLICT (cache_key) DO UPDATE
                   SET payload    = EXCLUDED.payload,
                       cached_at  = NOW()""",
                cache_key, ticker, outcome, window, cutoff_obj, payload_json,
            )
        _elapsed = _time.perf_counter() - _t0
        _log.info(
            "ic_batch_all_bg: wrote %d metrics to cache key=%s elapsed=%.1fs",
            len(results), cache_key, _elapsed,
        )

    except Exception as exc:
        import logging as _log2
        _log2.exception("ic_batch_all_bg: fatal error key=%s", cache_key)
        _ic_batch_status[cache_key] = {
            "status": "failed",
            "error":  f"{type(exc).__name__}: {exc}",
        }

    finally:
        _ic_batch_running.discard(cache_key)


# ── analyze-bundle compute + endpoints (P1) ──────────────────────────────
# The 12-outcome bundle drives the new lower-section modes (P3+). Compute
# is shared with /analyze's existing per-outcome path at the SQL fetch
# level — both ultimately hit daily_features — but the bundle does the
# bin-assignment ONCE and aggregates per-outcome from there, instead of
# running the Assigner 12 times.

async def _discover_analyze_bundle_outcomes(pool) -> list[str]:
    """Discover the set of forward-return outcome columns in daily_features.
    Matches /columns: numeric columns with 'ret_' and 'fwd' in the name.
    Excludes the metric/ticker/date/timestamp meta-columns.
    """
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """SELECT column_name FROM information_schema.columns
               WHERE table_name = 'daily_features' AND table_schema = 'public'
               AND data_type IN ('double precision','numeric','real',
                                 'integer','bigint','smallint')
               AND column_name NOT IN ('id','ticker','trade_date',
                                       'created_at','updated_at')
               ORDER BY ordinal_position""")
    all_cols = [r["column_name"] for r in rows]
    return [c for c in all_cols if "ret_" in c and "fwd" in c]


async def _fetch_analyze_bundle_rows(
    pool, ticker: str, metric: str, outcomes: list[str],
) -> list[dict]:
    """Fetch the rows needed for the bundle: trade_date, ticker, metric,
    spot_co, spot_pc, and the discovered forward-return outcome columns.
    Rows with NULL metric are excluded at the SQL level. Single-ticker
    mode injects the `ticker` column on the result dicts so downstream
    code can group uniformly with ALL mode.
    """
    if not outcomes:
        # Defensive — no outcomes discovered. Return empty so the caller
        # surfaces a clean error instead of building a SELECT with a
        # trailing comma.
        return []
    outcomes_sql = ", ".join(f'"{o}"' for o in outcomes)
    if ticker == "ALL":
        sql = (
            f'SELECT ticker, trade_date, "{metric}", '
            f'spot_co, spot_pc, {outcomes_sql} '
            f'FROM daily_features '
            f'WHERE "{metric}" IS NOT NULL '
            f'ORDER BY ticker, trade_date'
        )
        async with pool.acquire() as conn:
            db_rows = await conn.fetch(sql, timeout=180)
    else:
        sql = (
            f'SELECT trade_date, "{metric}", '
            f'spot_co, spot_pc, {outcomes_sql} '
            f'FROM daily_features '
            f'WHERE ticker = $1 AND "{metric}" IS NOT NULL '
            f'ORDER BY trade_date'
        )
        async with pool.acquire() as conn:
            db_rows = await conn.fetch(sql, ticker, timeout=60)
    rows = [dict(r) for r in db_rows]
    if ticker != "ALL":
        for r in rows:
            r.setdefault("ticker", ticker)
    return rows


# ── Parallel rolling-IC worker infrastructure ────────────────────────────
#
# Rolling IC (the per-outcome 252-day rolling Spearman) is the dominant
# remaining cost in the bundle compute (~3 s × 12 outcomes = ~36 s after
# the step-1 trade-walk vectorization). Each outcome's rolling IC is
# independent of the others, so we parallelize across outcomes with a
# ProcessPoolExecutor and a fork()-based start method.
#
# IPC strategy: fork() inherits parent memory pages via copy-on-write, so
# the large shared inputs (Y matrix, tm_ticker, tm_date, tm_metric_val)
# are stashed on module-level globals in the parent BEFORE the pool is
# created. Workers read them through ordinary global access — no pickling,
# no per-task IPC, no actual memory duplication unless a worker mutates a
# page (which we never do; see the read-only contract on _rolling_ic_worker).
#
# Per-task IPC is tiny: (outcome_idx, outcome_name, horizon) primitives
# in → (outcome_idx, classified_list, elapsed_s) out. ~kilobytes per task.

_W_Y           = None  # numpy float64 (N, 12)
_W_TICKER      = None  # Python list of strings
_W_DATE        = None  # Python list of ISO date strings
_W_METRIC_VAL  = None  # numpy float64 (N,)
_W_IS_ALL      = None  # bool
_W_MODE        = None  # str: "in_sample" | "walk_forward" | "train_test"
_W_CUTOFF      = None  # ISO string or None


def _compute_one_outcome_rolling_ic(
    Y, tm_ticker, tm_date, tm_metric_val,
    outcome_idx: int, outcome: str, horizon: int,
    is_all: bool, mode: str, cutoff_date: Optional[str],
):
    """Compute one outcome's classified rolling-IC result list.

    Shared between the parallel worker (_rolling_ic_worker, called from
    pool tasks) and the serial fallback path (when parallelization is
    disabled or single-ticker mode). Pure function: same inputs always
    produce the same output, so the parallel path is byte-identical to
    the serial path provided the inputs match.

    READ-ONLY ACCESS to Y, tm_ticker, tm_date, tm_metric_val. Mutating
    any of them would trigger copy-on-write on the worker side and
    silently kill the speedup from fork-based sharing.
    """
    import numpy as _np
    from app.routers.ic_compute import (
        rolling_ic_single_ticker, rolling_ic_cross_sectional,
        classified_rolling_ic, noise_floor_epsilon,
    )

    col = Y[:, outcome_idx]
    valid = ~_np.isnan(col)
    yf_raw = col[valid]
    if yf_raw.size == 0:
        return []

    valid_list = valid.tolist()
    valid_ticker = [t for t, ok in zip(tm_ticker, valid_list) if ok]
    valid_date   = [d for d, ok in zip(tm_date,   valid_list) if ok]
    valid_metric = tm_metric_val[valid].tolist()
    valid_y      = yf_raw.tolist()
    ic_rows = [
        {"trade_date": d, "ticker": t, "__m": m, "__y": y}
        for d, t, m, y in zip(valid_date, valid_ticker, valid_metric, valid_y)
    ]

    if is_all:
        ic_series = rolling_ic_cross_sectional(ic_rows, "__m", "__y", window=252)
        median_k = int(_np.median([p.n for p in ic_series])) if ic_series else 0
        epsilon = noise_floor_epsilon(
            "cross_sectional", window=252, horizon=horizon, k_tickers=median_k,
        )
    else:
        ic_series = rolling_ic_single_ticker(ic_rows, "__m", "__y", window=252)
        epsilon = noise_floor_epsilon(
            "single_ticker", window=252, horizon=horizon,
        )

    if mode == "train_test" and cutoff_date:
        pre = [p.ic for p in ic_series if str(p.date) < cutoff_date]
        reference_ic = float(_np.mean(pre)) if pre else 0.0
    elif ic_series:
        reference_ic = float(_np.mean([p.ic for p in ic_series]))
    else:
        reference_ic = 0.0

    classified = classified_rolling_ic(ic_series, reference_ic, epsilon)
    return [
        {"date":       str(p.date),
         "ic":         p.ic,
         "n":          p.n,
         "sign_class": p.sign_class}
        for p in classified
    ]


def _rolling_ic_worker(task):
    """ProcessPoolExecutor worker. Reads module globals stashed by the
    parent before pool creation; returns (outcome_idx, result, elapsed).

    READ-ONLY access to _W_Y / _W_TICKER / _W_DATE / _W_METRIC_VAL.
    Any in-place write (e.g., Y[i, j] = ...) triggers copy-on-write on
    the worker's pages and breaks the fork-sharing speedup, so don't.
    """
    import time as _time
    outcome_idx, outcome, horizon = task
    _t0 = _time.perf_counter()
    result = _compute_one_outcome_rolling_ic(
        _W_Y, _W_TICKER, _W_DATE, _W_METRIC_VAL,
        outcome_idx, outcome, horizon,
        _W_IS_ALL, _W_MODE, _W_CUTOFF,
    )
    return outcome_idx, result, _time.perf_counter() - _t0


def _assignments_from_is_bins(
    rows: list[dict], metric: str, anchor_outcome: str,
    n_bins: int, bin20_lookup: dict,
    tt_cutoff: Optional[str] = None,
):
    """v9 (Group 3b): build the bundle's bin assignments from is_bins
    instead of running InSampleAssigner. Mirrors the legacy assigner's
    IS+ALL semantics:

      - Outcome-anchor null/NaN filter (matches `_parse_rows`)
      - Metric null/NaN filter (matches `_parse_rows`)
      - bin20 looked up by (ticker, trade_date) — rows absent from
        is_bins are dropped (rare; only the most recent rows where
        the bin table hasn't been rebuilt yet)
      - Per-ticker thinning at n_t >= n_bins (matches
        `_bucket_pairs_per_ticker` exclusion)
      - Row order preserved from `rows` (= (ticker, trade_date) per the
        bundle SQL ORDER BY), so trade_id assignment in the downstream
        trade_meta build is stable.
    """
    from app.routers.row_compute import RowAssignment as _RA

    # Filter + bin20 lookup. No per-ticker thinning, so no count
    # tracking — every row that passes the metric/outcome/NaN
    # guards AND has a stored `bin20 > 0` becomes a candidate.
    # Group 8 fix: for TT mode, additionally skip pre-cutoff rows
    # (`trade_date < tt_cutoff`). Mirrors the test-window-only
    # semantic that /analyze, /metric-bins, and the secondary
    # endpoints apply in TT mode — pre-fix the bundle's per_bin and
    # trade_meta included full-series rows, which propagated to the
    # primary quantile via _buildOutcomeDataSlice.
    candidates: list = []
    for r in rows:
        tkr = r.get("ticker")
        d = r.get("trade_date")
        d_str = d.isoformat() if hasattr(d, "isoformat") else str(d)
        # Test-window-only gate for TT.
        if tt_cutoff is not None and d_str < tt_cutoff:
            continue
        xv = r.get(metric)
        yv = r.get(anchor_outcome)
        if xv is None or yv is None:
            continue
        try:
            xf, yf = float(xv), float(yv)
        except (ValueError, TypeError):
            continue
        if math.isnan(xf) or math.isnan(yf):
            continue
        b20 = bin20_lookup.get((tkr, d_str))
        if b20 is None or b20 <= 0:
            continue
        candidates.append((tkr, d_str, xf, yf, int(b20)))

    # No per-ticker thinning. The only row filter is the stored
    # `bin20 > 0` (applied above). Every candidate becomes an emitted
    # assignment with its stored bin. The legacy `tkr_counts[tkr] <
    # n_bins` gate is removed — the dashboard does no `n_t < N`
    # exclusion of its own.
    out: list = []
    for (tkr, d_str, xf, yf, b20) in candidates:
        out.append(_RA(
            ticker=tkr, trade_date=d_str, metric_name=metric,
            metric_value=xf, n_bins=n_bins, bin=b20,
            outcome_col=anchor_outcome, forward_return=yf,
            dropped_reason=None,
        ))
    return out


def _compute_analyze_bundle_sync(
    rows: list[dict],
    metric: str,
    ticker: str,
    mode: str,
    cutoff_date: Optional[str],
    outcomes: list[str],
    n_bins: int = 20,
    _measure: bool = False,
    _parallel_rolling_ic: bool = False,
    bin20_lookup: Optional[dict] = None,
) -> dict:
    """Pure-sync compute. Off-loaded via asyncio.to_thread by the caller.

    Bin assignment is shared across all 12 outcomes — the Assigner is
    invoked ONCE with ret_5d_fwd_oc as the filtering anchor (dashboard
    default outcome). Per-outcome data is then derived in a thin pass
    by looking up each outcome's column for every binned row.

    Notes on the per-outcome math:
      - Per-trade returns: `ret_pct` is the raw outcome column value;
        `exit_date` is trade_date + (horizon - 1) trading days within
        the same ticker; `exit_spot` is the close at exit_date
        (= spot_pc of the day AFTER exit_date in the same ticker's
        chronology). Returns are skipped (excluded from the array) for
        any row where the specific outcome column is NULL/NaN.
      - Per-bin stats: standard aggregations over the per-trade return
        array (mean, median, std, win_rate, n).
      - Rolling IC: 252-day trailing Spearman of (metric_val,
        outcome_value) — mode-aware reference (full-history mean for
        in_sample/walk_forward, pre-cutoff mean for train_test); ε is
        mode-aware (single_ticker vs cross_sectional, horizon-corrected).

    Returns a dict that JSON-serializes to the analyze_cache payload
    shape (see _ANALYZE_BUNDLE_SCHEMA_VERSION for the contract).
    """
    from datetime import datetime
    from app.routers.ic_compute import (
        rolling_ic_single_ticker, rolling_ic_cross_sectional,
        classified_rolling_ic, noise_floor_epsilon, _horizon_from_outcome,
    )
    import time as _perf_time

    def _tick():
        """perf_counter helper for the _measure path. Always cheap."""
        return _perf_time.perf_counter()

    is_all = (ticker == "ALL")
    outcomes = list(outcomes)

    # Pick an anchor outcome for the Assigner filter. Prefer
    # ret_5d_fwd_oc (dashboard default) if present; otherwise fall back
    # to the first discovered outcome. The choice only affects which rows
    # get an anchored bin assignment — per-outcome aggregations skip
    # NULL rows per-outcome independently, so a row with the anchor
    # NULL is excluded from ALL aggregations (acceptable trade-off; the
    # 12 forward-returns share a near-identical NULL pattern in
    # practice since they're all derived from the same spot series).
    anchor_outcome = "ret_5d_fwd_oc" if "ret_5d_fwd_oc" in outcomes else outcomes[0]

    # ── 1. Index rows + build per-ticker chronology for exit-date offsets ──
    # row_by_tkr_date: O(1) lookup from (ticker, date_str) to the raw row
    # by_tkr_dates:   per-ticker sorted list of date strings (chronological)
    # by_tkr_idx:     per-ticker date → index in by_tkr_dates
    # close_by_tkr_date: close-of-day lookup. close(date) = spot_pc of the
    #   NEXT trading day for the same ticker (canonical equivalence used
    #   throughout the dashboard's trade-calendar code).
    if _measure: _t_idx = _tick()
    row_by_tkr_date: dict = {}
    by_tkr_dates_unsorted: dict = defaultdict(list)
    for r in rows:
        tkr = r.get("ticker", ticker)
        d = str(r.get("trade_date", ""))
        row_by_tkr_date[(tkr, d)] = r
        by_tkr_dates_unsorted[tkr].append(d)
    by_tkr_dates = {t: sorted(set(ds)) for t, ds in by_tkr_dates_unsorted.items()}
    by_tkr_idx   = {t: {d: i for i, d in enumerate(ds)} for t, ds in by_tkr_dates.items()}
    close_by_tkr_date: dict = {}
    for tkr, dates in by_tkr_dates.items():
        for i, d in enumerate(dates[:-1]):
            next_row = row_by_tkr_date.get((tkr, dates[i + 1]))
            if next_row is None:
                continue
            pc = next_row.get("spot_pc")
            if pc is None:
                continue
            try:
                close_by_tkr_date[(tkr, d)] = float(pc)
            except (TypeError, ValueError):
                continue
    if _measure:
        print(f"[SHARED] index_build={_tick() - _t_idx:.3f}s  "
              f"(rows={len(rows)}, tickers={len(by_tkr_dates)}, "
              f"close_lookup_entries={len(close_by_tkr_date)})")

    # ── 2. Resolve bin assignments ────────────────────────────────────────
    # All modes read stored bins via bin20_lookup (is_bins / wf_bins /
    # tt_bins depending on mode). When bin20_lookup is None the metric
    # has no bin column (null-by-design) — return empty assignments and
    # let the bundle surface no data rather than computing on the fly.
    if mode in {"in_sample", "walk_forward", "train_test"} and bin20_lookup is not None:
        if _measure: _t_assign_start = _tick()
        # Group 8: thread the TT cutoff so the bundle's per_bin and
        # trade_meta only include test-window rows in TT mode. Pre-fix,
        # the bundle filtered pre-cutoff rows only via the legacy
        # TrainTestAssigner's dropped_reason="pre_cutoff" mechanism; the
        # stored-bin path didn't carry that, so when /analyze's primary
        # chart consumed the bundle's per_bin it showed full-series
        # counts under a "TEST PERIOD" subtitle.
        tt_cutoff_iso = (
            cutoff_date if mode == "train_test" and cutoff_date else None
        )
        assignments = _assignments_from_is_bins(
            rows, metric, anchor_outcome, n_bins, bin20_lookup,
            tt_cutoff=tt_cutoff_iso,
        )
        if _measure:
            print(f"[SHARED] assignments_from_is_bins={_tick() - _t_assign_start:.3f}s  "
                  f"(assignments={len(assignments)})")
    else:
        # Metric absent from stored bins (null-by-design or no bin column).
        # Never compute on the fly — surface empty bundle instead.
        assignments = []

    # ── 3. Build trade_meta from valid assignments ────────────────────────
    # v5: entry fields are anchored. OC outcomes enter at open of T;
    # CC outcomes enter at close of T-1 (the prior trading day for the
    # same ticker). Overnight Gap mode also reads the CC-anchored entry
    # fields (it enters at C_{T-1} and exits at O_T). trade_date stays
    # as the *signal* date T — used by yearly/dow/activity aggregations
    # which treat each trade as "a signal on day T" regardless of when
    # the position was opened.
    if _measure: _t_meta = _tick()
    trade_meta: list = []
    trade_id_by_key: dict = {}
    for a in assignments:
        if a.bin is None or a.dropped_reason is not None:
            continue
        tkr = a.ticker
        # Normalize date to ISO string (asyncpg can hand back datetime.date).
        d = (a.trade_date.isoformat()
             if hasattr(a.trade_date, "isoformat") else str(a.trade_date))
        key = (tkr, d)
        if key in trade_id_by_key:
            continue
        r = row_by_tkr_date.get(key)

        # OC anchor: entry on T, at open of T (= spot_co).
        entry_date_oc = d
        entry_spot_oc = None
        if r is not None and r.get("spot_co") is not None:
            try:
                entry_spot_oc = round(float(r["spot_co"]), 2)
            except (TypeError, ValueError):
                pass

        # CC anchor: entry on T-1, at close of T-1 (= close_by_tkr_date
        # keyed on T-1, which the rest of the bundle code defines as
        # "close at end of date d"). For a ticker's first trading day,
        # T-1 doesn't exist → fields are None.
        entry_date_cc = None
        entry_spot_cc = None
        entry_idx = by_tkr_idx.get(tkr, {}).get(d)
        if entry_idx is not None and entry_idx > 0:
            tkr_dates = by_tkr_dates.get(tkr, [])
            if 0 <= entry_idx - 1 < len(tkr_dates):
                entry_date_cc = tkr_dates[entry_idx - 1]
                _es_cc = close_by_tkr_date.get((tkr, entry_date_cc))
                if _es_cc is not None:
                    entry_spot_cc = round(float(_es_cc), 2)

        tid = len(trade_meta)
        trade_id_by_key[key] = tid
        trade_meta.append({
            "trade_id":      tid,
            "ticker":        tkr,
            "trade_date":    d,
            "metric_val":    round(float(a.metric_value), 6),
            "entry_date_oc": entry_date_oc,
            "entry_spot_oc": entry_spot_oc,
            "entry_date_cc": entry_date_cc,
            "entry_spot_cc": entry_spot_cc,
            "bin_20":        a.bin,    # 1..20; client aggregates pairs/quartets for 10/5 views
        })

    if _measure:
        print(f"[SHARED] trade_meta_build={_tick() - _t_meta:.3f}s  "
              f"(trade_meta_rows={len(trade_meta)})")

    # ── 3b. Vectorization pre-computation (run once, shared across all 12 outcomes) ──
    # The per-outcome trade walk used to be a Python for-meta-in-trade_meta
    # loop doing dict-gets, float-coercion, and exit-date arithmetic per
    # trade per outcome (~1.6 s × 12 = ~20 s measured). Replace with:
    #   - tm_* numpy arrays of the trade_meta fields  (one-time, ~0.3 s)
    #   - Y matrix of outcome values, shape (N, len(outcomes))  (one-time, ~2 s)
    #   - per-unique-horizon exit_date/exit_spot arrays — there are only six
    #     unique horizons (1/3/5/7/10/20d) among the twelve outcomes, so OC
    #     and CC variants share the same exit arrays  (one-time, ~0.5 s)
    # Then per-outcome work is numpy slicing + bincount-style grouping.
    # Output is byte-identical to the prior Python-loop implementation.
    if _measure: _t_vec = _tick()
    N = len(trade_meta)
    tm_ticker     = [m["ticker"]     for m in trade_meta]   # list of strings
    tm_date       = [m["trade_date"] for m in trade_meta]   # list of ISO strings
    tm_metric_val = np.array([m["metric_val"] for m in trade_meta], dtype=np.float64)
    tm_bin20      = np.array([m["bin_20"]     for m in trade_meta], dtype=np.int64)
    tm_trade_id   = np.arange(N, dtype=np.int64)            # trade_id == position
    # Per-trade entry index in their ticker's chronology
    tm_entry_idx  = np.fromiter(
        (by_tkr_idx.get(t, {}).get(d, -1) for t, d in zip(tm_ticker, tm_date)),
        dtype=np.int64, count=N,
    )

    # Flatten the per-ticker chronologies into one big array so exit_date
    # lookups become array indexing instead of dict-of-list-of-string lookups.
    # Each ticker's dates occupy a contiguous slice [offset, offset+length).
    ticker_to_offset: dict = {}
    flat_dates_list:  list = []
    flat_closes_list: list = []
    _off = 0
    for tkr in sorted(by_tkr_dates.keys()):
        ticker_to_offset[tkr] = _off
        dates = by_tkr_dates[tkr]
        flat_dates_list.extend(dates)
        flat_closes_list.extend(
            close_by_tkr_date.get((tkr, d), np.nan) for d in dates
        )
        _off += len(dates)
    flat_dates  = np.array(flat_dates_list,  dtype=object)
    flat_closes = np.array(flat_closes_list, dtype=np.float64)
    ticker_to_length: dict = {t: len(d) for t, d in by_tkr_dates.items()}
    tm_ticker_offset = np.fromiter(
        (ticker_to_offset[t] for t in tm_ticker), dtype=np.int64, count=N,
    )
    tm_ticker_length = np.fromiter(
        (ticker_to_length[t] for t in tm_ticker), dtype=np.int64, count=N,
    )
    if _measure:
        print(f"[SHARED] vectorize_setup={_tick() - _t_vec:.3f}s  "
              f"(flat_chronology_entries={len(flat_dates_list)})")

    # Outcome value matrix Y of shape (N, len(outcomes)). One pass over
    # row_by_tkr_date populates all 12 outcomes' values at once — replaces
    # the per-outcome `r = row_by_tkr_date.get(...); y = r.get(outcome)`
    # dict-get duplicated across 12 outcomes.
    if _measure: _t_Y = _tick()
    Y = np.full((N, len(outcomes)), np.nan, dtype=np.float64)
    for i, (t, d) in enumerate(zip(tm_ticker, tm_date)):
        r = row_by_tkr_date.get((t, d))
        if r is None:
            continue
        for j, o in enumerate(outcomes):
            v = r.get(o)
            if v is None:
                continue
            try:
                fv = float(v)
            except (TypeError, ValueError):
                continue
            # math.isnan check is folded into the np.isnan filter below; we
            # store fv even if NaN — the valid mask sees it correctly.
            Y[i, j] = fv
    if _measure:
        print(f"[SHARED] outcome_matrix_build={_tick() - _t_Y:.3f}s  "
              f"(matrix_shape=({N},{len(outcomes)}))")

    # Per-unique-horizon exit_date and exit_spot arrays of length N. Built
    # once and looked up by horizon below — OC and CC variants at the same
    # horizon share the same arrays so this is at most 6 horizon computes
    # instead of 12.
    if _measure: _t_exit = _tick()
    unique_horizons = sorted({_horizon_from_outcome(o) for o in outcomes})
    exit_date_by_horizon: dict = {}
    exit_spot_by_horizon: dict = {}
    for h in unique_horizons:
        exit_idx_h  = tm_entry_idx + max(h - 1, 0)
        valid_exit  = (tm_entry_idx >= 0) & (exit_idx_h >= 0) & (exit_idx_h < tm_ticker_length)
        # safe_pos avoids OOB at the array level; results are masked back to
        # None / NaN via valid_exit before use.
        safe_pos    = np.where(valid_exit, tm_ticker_offset + exit_idx_h, 0)
        raw_dates   = flat_dates[safe_pos]
        raw_closes  = flat_closes[safe_pos]
        # exit_date: string where valid_exit, None otherwise.
        exit_dates_h = np.where(valid_exit, raw_dates, None)
        # exit_spot: round(close, 2) where valid_exit, NaN otherwise. NaN-to-
        # None conversion happens per-outcome at output-list build time.
        rounded_closes = np.round(raw_closes, 2)
        exit_spots_h = np.where(valid_exit, rounded_closes, np.nan)
        exit_date_by_horizon[h] = exit_dates_h
        exit_spot_by_horizon[h] = exit_spots_h
    if _measure:
        print(f"[SHARED] exit_arrays_build={_tick() - _t_exit:.3f}s  "
              f"(unique_horizons={unique_horizons})")

    # ── 4. Per-outcome work, split into two passes ───────────────────────
    # Pass 1 (serial, in main process): per_outcome_returns + per_bin.
    #   These feed downstream quantile graphs + the P4 active-outcome swap
    #   and are fast after step-1 vectorization (~3 s total).
    # Pass 2 (serial by default): rolling_ic + classify. A fork()-based
    #   ProcessPoolExecutor path exists but is opt-in via
    #   _parallel_rolling_ic=True (used by scripts/measure_analyze.py for
    #   benchmarking). Forking while the live asyncpg oi_pool holds open
    #   Postgres sockets corrupts those connections — subsequent queries
    #   that pick up an inherited fd time out at command_timeout=30s,
    #   surfacing as intermittent 500s on later /analyze-bundle polls. The
    #   live server path stays serial until the parallelization is moved
    #   off fork (forkserver with a connection-free intermediate, or
    #   another scheme); the ~24s speedup is not worth a corrupted DB pool.
    per_outcome_returns: dict = {}
    per_bin:             dict = {}
    rolling_ic:          dict = {}
    if _measure:
        _t_pass1_start = _tick()
        _sum_walk = _sum_bin_stats = 0.0
        print(f"[PASS 1] walk + bin_stats over {len(outcomes)} outcomes (serial)")

    for _outcome_idx, outcome in enumerate(outcomes):
        horizon = _horizon_from_outcome(outcome)

        # Per-outcome trade walk — vectorized. Output shapes / values match
        # the prior Python-loop implementation byte-for-byte:
        #   per_outcome_returns[outcome] = {trade_ids, ret_pcts, exit_dates,
        #                                   exit_spots}  (parallel arrays, valid
        #                                   trades only, in trade_meta order)
        if _measure: _t_walk = _tick()
        col   = Y[:, _outcome_idx]
        valid = ~np.isnan(col)
        yf_raw = col[valid]                           # raw floats (for bin_stats)

        # per_outcome_returns parallel arrays
        out_trade_ids  = tm_trade_id[valid].tolist()
        out_ret_pcts   = np.round(yf_raw, 6).tolist()
        exit_dates_h   = exit_date_by_horizon[horizon]
        exit_spots_h   = exit_spot_by_horizon[horizon]
        out_exit_dates = exit_dates_h[valid].tolist()
        # NaN → None for JSON-safety. Use the NaN!=NaN idiom on the Python
        # list (faster than per-element np.isnan calls here).
        _spots = exit_spots_h[valid].tolist()
        out_exit_spots = [None if v != v else v for v in _spots]

        per_outcome_returns[outcome] = {
            "trade_ids":  out_trade_ids,
            "ret_pcts":   out_ret_pcts,
            "exit_dates": out_exit_dates,
            "exit_spots": out_exit_spots,
        }

        # bin_buckets equivalent: group yf_raw by tm_bin20[valid]
        bin20_valid = tm_bin20[valid]
        if _measure:
            _dt_walk = _tick() - _t_walk
            _sum_walk += _dt_walk
            _t_binstats = _tick()

        # Per-bin aggregates — same numbers as before, computed by boolean-
        # indexing yf_raw with bin20_valid == bin instead of materializing a
        # list-of-lists. Same n, avg_ret, median, std, win_rate.
        bin_stats: list = []
        for b_idx in range(n_bins):
            bin_mask = bin20_valid == (b_idx + 1)
            if not bin_mask.any():
                bin_stats.append({"bin": b_idx + 1, "n": 0,
                                  "avg_ret": None, "median": None,
                                  "std": None, "win_rate": None})
                continue
            arr = yf_raw[bin_mask]
            bin_stats.append({
                "bin":      b_idx + 1,
                "n":        int(arr.size),
                "avg_ret":  round(float(arr.mean()), 6),
                "median":   round(float(np.median(arr)), 6),
                "std":      round(float(arr.std()), 6),
                "win_rate": round(float((arr > 0).mean()), 4),
            })
        per_bin[outcome] = bin_stats
        if _measure:
            _dt_binstats = _tick() - _t_binstats
            _sum_bin_stats += _dt_binstats
            print(f"  [outcome {_outcome_idx+1:>2}/{len(outcomes)}] {outcome:<18s} "
                  f"walk={_dt_walk:.3f}s  bin_stats={_dt_binstats:.3f}s  "
                  f"(emitted_trades={len(out_trade_ids)})")

    if _measure:
        print(f"[PASS 1 TOTALS over {len(outcomes)} outcomes] "
              f"walk={_sum_walk:.3f}s  bin_stats={_sum_bin_stats:.3f}s  "
              f"=> pass1_total={_tick() - _t_pass1_start:.3f}s")

    # ── Pass 2: rolling_ic + classify ────────────────────────────────────
    # Serial by default. The parallel ProcessPoolExecutor path runs only
    # when explicitly opted in via _parallel_rolling_ic=True (measure
    # script). See the section-4 comment above for why the live path is
    # serial — fork-after-asyncpg-pool corrupts inherited Postgres sockets.
    horizons_per_outcome = [_horizon_from_outcome(o) for o in outcomes]
    tasks = [(j, outcomes[j], horizons_per_outcome[j]) for j in range(len(outcomes))]
    use_parallel = (
        _parallel_rolling_ic
        and is_all
        and len(outcomes) > 1
    )
    if _measure:
        _t_pass2_start = _tick()
        _sum_worker_elapsed = 0.0
        _per_outcome_elapsed: list = [0.0] * len(outcomes)

    if use_parallel:
        import os as _os
        import multiprocessing as _mp
        import concurrent.futures as _cf
        import warnings as _warnings

        n_workers = max(1, (_os.cpu_count() or 4) - 1)
        n_workers = min(n_workers, len(outcomes))

        # Stash shared inputs on module globals BEFORE pool creation so
        # fork() inherits them via copy-on-write. The cleanup in `finally`
        # is essential — leaving these set holds memory across requests.
        global _W_Y, _W_TICKER, _W_DATE, _W_METRIC_VAL
        global _W_IS_ALL, _W_MODE, _W_CUTOFF
        _W_Y = Y
        _W_TICKER = tm_ticker
        _W_DATE = tm_date
        _W_METRIC_VAL = tm_metric_val
        _W_IS_ALL = is_all
        _W_MODE = mode
        _W_CUTOFF = cutoff_date

        if _measure:
            print(f"[PASS 2] rolling_ic + classify over {len(outcomes)} outcomes "
                  f"(PARALLEL, n_workers={n_workers}, mp_ctx=fork)")
        try:
            # fork() in a multi-threaded Python (we're inside asyncio.to_thread)
            # emits a DeprecationWarning in 3.12+. We suppress it at this
            # call site — the bundle compute does not spawn threads itself
            # and the shared arrays are read-only in workers, so the actual
            # safety concerns the warning flags don't apply here.
            with _warnings.catch_warnings():
                _warnings.filterwarnings(
                    "ignore",
                    category=DeprecationWarning,
                    message=r".*fork.*multi-threaded.*",
                )
                mp_ctx = _mp.get_context("fork")
                with _cf.ProcessPoolExecutor(
                    max_workers=n_workers,
                    mp_context=mp_ctx,
                ) as pool:
                    # pool.map preserves task order so the result list is in
                    # the same order as `outcomes` — dict insertion order
                    # below matches the serial path byte-for-byte.
                    pool_results = list(pool.map(_rolling_ic_worker, tasks))
            for outcome_idx, classified_list, elapsed in pool_results:
                rolling_ic[outcomes[outcome_idx]] = classified_list
                if _measure:
                    _sum_worker_elapsed += elapsed
                    _per_outcome_elapsed[outcome_idx] = elapsed
                    print(f"  [worker outcome {outcome_idx+1:>2}/{len(outcomes)}] "
                          f"{outcomes[outcome_idx]:<18s} elapsed={elapsed:.3f}s")
        finally:
            # Clear the stash. Workers have already exited; this releases
            # the parent's references to the large arrays for the next
            # request (or for GC if this was the only bundle compute).
            _W_Y = None
            _W_TICKER = None
            _W_DATE = None
            _W_METRIC_VAL = None
            _W_IS_ALL = None
            _W_MODE = None
            _W_CUTOFF = None
    else:
        if _measure:
            print(f"[PASS 2] rolling_ic + classify over {len(outcomes)} outcomes (SERIAL)")
        for outcome_idx, outcome, horizon in tasks:
            import time as _time_serial
            _t_one = _time_serial.perf_counter()
            classified_list = _compute_one_outcome_rolling_ic(
                Y, tm_ticker, tm_date, tm_metric_val,
                outcome_idx, outcome, horizon,
                is_all, mode, cutoff_date,
            )
            elapsed = _time_serial.perf_counter() - _t_one
            rolling_ic[outcome] = classified_list
            if _measure:
                _sum_worker_elapsed += elapsed
                _per_outcome_elapsed[outcome_idx] = elapsed
                print(f"  [outcome {outcome_idx+1:>2}/{len(outcomes)}] "
                      f"{outcome:<18s} rolling_ic+classify={elapsed:.3f}s")

    if _measure:
        _t_pass2_total = _tick() - _t_pass2_start
        _speedup = (_sum_worker_elapsed / _t_pass2_total) if _t_pass2_total > 0 else 0.0
        print(f"[PASS 2 TOTALS over {len(outcomes)} outcomes] "
              f"pool_wall={_t_pass2_total:.3f}s  sum_worker={_sum_worker_elapsed:.3f}s  "
              f"effective_speedup={_speedup:.2f}x  "
              f"(parallel={use_parallel})")

    # ── 4b. Synthetic overnight_gap outcome (server-side precompute, v8) ──
    # Gap = ret_1d_fwd_cc − ret_1d_fwd_oc per trade_id where BOTH legs
    # exist. ALL fields needed by EVERY Gap-mode client view are carried
    # inline so the frontend never has to fetch trade_meta or the 1d_cc /
    # 1d_oc slices to render Gap mode — not for the quantile chart,
    # equity, yearly / DoW / activity aggregates, OR the flat trade
    # table. v7 covered the chart/swap path but left the trade-table
    # render firing the 56 MB v6 fetches; v8 extends the precompute so
    # there's no fallback path that needs trade_meta.
    #
    # Shape (parallel arrays, index-aligned — no trade_id lookup needed):
    #   ret_pcts        — gap value (cc − oc)
    #   trade_dates     — signal date T (also = OC-anchor exit date)
    #   tickers         — ticker per trade
    #   bin_20s         — bin assignment (1..20)
    #   entry_dates_cc  — T−1 (CC anchor entry date = flat-table date)
    #   entry_spots_cc  — close T−1 (= flat-table spot_entry)
    #   entry_spots_oc  — open T (= flat-table spot_exit)
    #   metric_vals     — metric value at signal (= flat-table metric_val)
    # Plus per_bin["overnight_gap"] = 20-row aggregate (same shape as
    # real outcomes), which lives in the slim payload so the Gap-mode
    # quantile bar chart renders instantly without any deferred fetch.
    if "ret_1d_fwd_cc" in per_outcome_returns and "ret_1d_fwd_oc" in per_outcome_returns:
        _cc_data = per_outcome_returns["ret_1d_fwd_cc"]
        _oc_data = per_outcome_returns["ret_1d_fwd_oc"]
        _oc_by_tid = dict(zip(_oc_data["trade_ids"], _oc_data["ret_pcts"]))
        _gap_ret_pcts:       list = []
        _gap_trade_dates:    list = []
        _gap_tickers:        list = []
        _gap_bin_20s:        list = []
        _gap_entry_dates_cc: list = []
        _gap_entry_spots_cc: list = []
        _gap_entry_spots_oc: list = []
        _gap_metric_vals:    list = []
        for _i, _tid in enumerate(_cc_data["trade_ids"]):
            if _tid not in _oc_by_tid:
                continue
            if _tid >= len(trade_meta):
                continue
            _m = trade_meta[_tid]
            _gap_ret_pcts.append(round(_cc_data["ret_pcts"][_i] - _oc_by_tid[_tid], 6))
            _gap_trade_dates.append(_m["trade_date"])
            _gap_tickers.append(_m["ticker"])
            _gap_bin_20s.append(_m["bin_20"])
            _gap_entry_dates_cc.append(_m["entry_date_cc"])
            _gap_entry_spots_cc.append(_m["entry_spot_cc"])
            _gap_entry_spots_oc.append(_m["entry_spot_oc"])
            _gap_metric_vals.append(_m["metric_val"])
        per_outcome_returns["overnight_gap"] = {
            "ret_pcts":       _gap_ret_pcts,
            "trade_dates":    _gap_trade_dates,
            "tickers":        _gap_tickers,
            "bin_20s":        _gap_bin_20s,
            "entry_dates_cc": _gap_entry_dates_cc,
            "entry_spots_cc": _gap_entry_spots_cc,
            "entry_spots_oc": _gap_entry_spots_oc,
            "metric_vals":    _gap_metric_vals,
        }
        # Per-bin aggregates for gap. Same numpy bin_stats shape as real
        # outcomes (above). Empty per_bin list when no overlap is rare
        # but possible (e.g., 1d_cc and 1d_oc both empty for a metric).
        _bin_stats_gap: list = []
        if _gap_ret_pcts:
            _gap_rets_arr = np.array(_gap_ret_pcts, dtype=np.float64)
            _gap_bins_arr = np.array(_gap_bin_20s,  dtype=np.int64)
            for _b_idx in range(n_bins):
                _bmask = _gap_bins_arr == (_b_idx + 1)
                if not _bmask.any():
                    _bin_stats_gap.append({"bin": _b_idx + 1, "n": 0,
                                           "avg_ret": None, "median": None,
                                           "std": None, "win_rate": None})
                    continue
                _arr = _gap_rets_arr[_bmask]
                _bin_stats_gap.append({
                    "bin":      _b_idx + 1,
                    "n":        int(_arr.size),
                    "avg_ret":  round(float(_arr.mean()),  6),
                    "median":   round(float(np.median(_arr)), 6),
                    "std":      round(float(_arr.std()),   6),
                    "win_rate": round(float((_arr > 0).mean()), 4),
                })
        else:
            for _b_idx in range(n_bins):
                _bin_stats_gap.append({"bin": _b_idx + 1, "n": 0,
                                       "avg_ret": None, "median": None,
                                       "std": None, "win_rate": None})
        per_bin["overnight_gap"] = _bin_stats_gap

    # ── 5. Bundle ─────────────────────────────────────────────────────────
    return {
        "schema_version":      _ANALYZE_BUNDLE_SCHEMA_VERSION,
        "ticker":              ticker,
        "metric":              metric,
        "mode":                mode,
        "cutoff_date":         cutoff_date,
        "n_bins":              n_bins,
        "outcomes":            outcomes,
        "trade_meta":          trade_meta,
        "per_outcome_returns": per_outcome_returns,
        "per_bin":             per_bin,
        "rolling_ic":          rolling_ic,
        "computed_at":         datetime.utcnow().isoformat() + "Z",
    }


async def _compute_analyze_bundle_bg(
    cache_key: str, ticker: str, metric: str, mode: str,
    cutoff_date: Optional[str], pool,
) -> None:
    """Background bundle compute for ALL mode. Computes the full bundle dict
    via _compute_analyze_bundle_sync, splits it into the 3 storage layers
    (slim / trade_meta / per-outcome), serialises each in to_thread, then
    UPSERTs all 14 rows (1 slim + 1 trade_meta + 12 outcome) in a single
    transaction. Triggers LRU eviction. Failures surface via
    _analyze_bundle_status on next GET."""
    try:
        outcomes = await _discover_analyze_bundle_outcomes(pool)
        if not outcomes:
            _analyze_bundle_status[cache_key] = {
                "status": "failed",
                "error":  "no_forward_return_columns_in_daily_features",
            }
            return
        rows = await _fetch_analyze_bundle_rows(pool, ticker, metric, outcomes)
        if not rows:
            _analyze_bundle_status[cache_key] = {
                "status": "failed",
                "error":  "no_data_for_ticker_metric",
            }
            return
        # v9 (Group 3b): IS+ALL prefetches bin20 from is_bins.
        # v11 (Group 7): WF+ALL extends the same shape — prefetch from
        # wf_bins. Sync compute consumes the lookup identically; the
        # encoding is the same (Encoding A: 0 = warm-up or null).
        # v12 (Group 8): TT+ALL extends again — prefetch from tt_bins.
        # tt_bins is IS-frozen-at-cutoff; encoding A applies (0 =
        # null-metric or insufficient-training-sample, no warm-up).
        bin20_lookup: Optional[dict] = None
        if mode in {"in_sample", "walk_forward", "train_test"}:
            bin_table = {
                "in_sample":   "is_bins",
                "walk_forward": "wf_bins",
                "train_test":  "tt_bins",
            }[mode]
            # Probe information_schema before querying: avoids a bare
            # except that could silently swallow connection errors and
            # fall back to Assigner for a valid metric.  Null-by-design
            # metrics have no bin20_* column → returns False → stays None.
            async with pool.acquire() as conn:
                _col_exists = await conn.fetchval(
                    "SELECT EXISTS(SELECT 1 FROM information_schema.columns "
                    "WHERE table_name = $1 AND table_schema = 'public' "
                    "AND column_name = $2)",
                    bin_table, f"bin20_{metric}"
                )
            if _col_exists:
                bin_sql = (
                    f'SELECT ticker, trade_date, bin20_{metric} AS bin_20 '
                    f'FROM {bin_table} WHERE bin20_{metric} > 0'
                )
                async with pool.acquire() as conn:
                    bin_rows = await conn.fetch(bin_sql, timeout=60)
                bin20_lookup = {}
                for r in bin_rows:
                    d = r['trade_date']
                    d_str = d.isoformat() if hasattr(d, 'isoformat') else str(d)
                    bin20_lookup[(r['ticker'], d_str)] = r['bin_20']
        bundle = await asyncio.to_thread(
            _compute_analyze_bundle_sync,
            rows, metric, ticker, mode, cutoff_date, outcomes,
            bin20_lookup=bin20_lookup,
        )

        # Split the bundle into the 3 storage layers. trade_meta and
        # per_outcome_returns are popped off so the remainder serialises as
        # the slim payload (per_bin + rolling_ic + scalar metadata).
        trade_meta_obj         = bundle.pop("trade_meta", [])
        per_outcome_returns_obj = bundle.pop("per_outcome_returns", {})
        # `bundle` is now the slim payload — same shape as v5 minus the two
        # heavy keys. Frontend reads it as `analyzeBundle` directly.

        # json.dumps each layer in to_thread. allow_nan=False rejects NaN/inf;
        # the ValueError propagates and is recorded as failed status. We
        # thread the dumps so the event loop stays free during the ~5s of
        # serialisation work — same precaution that fixed the v5 500s.
        try:
            slim_json = await asyncio.to_thread(
                json.dumps, bundle, allow_nan=False,
            )
            trade_meta_json = await asyncio.to_thread(
                json.dumps, trade_meta_obj, allow_nan=False,
            )
            outcome_jsons: dict[str, str] = {}
            # v7: iterate the actual per_outcome_returns dict (12 real
            # outcomes + the synthetic overnight_gap row when both 1d
            # legs were computed). Each key becomes one row in
            # analyze_cache_outcome — frontend fetches by outcome name.
            for outcome_name, outcome_data in per_outcome_returns_obj.items():
                outcome_jsons[outcome_name] = await asyncio.to_thread(
                    json.dumps, outcome_data, allow_nan=False,
                )
        except ValueError as je:
            _analyze_bundle_status[cache_key] = {
                "status": "failed",
                "error":  f"json_nan_inf_in_bundle: {je}",
            }
            return

        from datetime import date as _date_cls
        cutoff_obj = _date_cls.fromisoformat(cutoff_date) if cutoff_date else None
        async with pool.acquire() as conn:
            async with conn.transaction():
                await conn.execute(
                    """INSERT INTO analyze_cache_slim
                       (cache_key, ticker, metric, mode, cutoff_date,
                        payload, payload_bytes, cached_at, last_accessed)
                       VALUES ($1, $2, $3, $4, $5, $6::jsonb, $7, NOW(), NOW())
                       ON CONFLICT (cache_key) DO UPDATE
                       SET payload       = EXCLUDED.payload,
                           payload_bytes = EXCLUDED.payload_bytes,
                           cached_at     = NOW(),
                           last_accessed = NOW()""",
                    cache_key, ticker, metric, mode, cutoff_obj,
                    slim_json, len(slim_json),
                )
                await conn.execute(
                    """INSERT INTO analyze_cache_trade_meta
                       (cache_key, payload, payload_bytes, cached_at)
                       VALUES ($1, $2::jsonb, $3, NOW())
                       ON CONFLICT (cache_key) DO UPDATE
                       SET payload       = EXCLUDED.payload,
                           payload_bytes = EXCLUDED.payload_bytes,
                           cached_at     = NOW()""",
                    cache_key, trade_meta_json, len(trade_meta_json),
                )
                for outcome_name, outcome_payload_json in outcome_jsons.items():
                    await conn.execute(
                        """INSERT INTO analyze_cache_outcome
                           (cache_key, outcome, payload, payload_bytes, cached_at)
                           VALUES ($1, $2, $3::jsonb, $4, NOW())
                           ON CONFLICT (cache_key, outcome) DO UPDATE
                           SET payload       = EXCLUDED.payload,
                               payload_bytes = EXCLUDED.payload_bytes,
                               cached_at     = NOW()""",
                        cache_key, outcome_name,
                        outcome_payload_json, len(outcome_payload_json),
                    )
        try:
            await _evict_analyze_cache_lru(pool)
        except Exception as evict_e:
            import logging
            logging.warning("analyze_cache LRU eviction failed: %r", evict_e)
    except Exception as e:
        _analyze_bundle_status[cache_key] = {
            "status": "failed",
            "error":  f"{type(e).__name__}: {e}",
        }
    finally:
        _analyze_bundle_running.discard(cache_key)


@router.get("/analyze-bundle")
async def analyze_bundle_get(
    ticker:      str           = Query(...),
    metric:      str           = Query(...),
    mode:        str           = Query("in_sample"),
    cutoff_date: Optional[str] = Query(None),
    pool=Depends(get_oi_pool),
):
    """Return the 12-outcome bundle status for (ticker, metric, mode, cutoff).

    Response shapes:

      # single-ticker path (computes inline, ~2-5s)
      {status: "ready", bundle: {...}}

      # ALL-mode polling path (lean — never carries the bundle)
      {status: "ready",        cached_at: "...", payload_bytes: N}
      {status: "computing",    cache_key: "..."}
      {status: "not_computed", previous_error?: "..."}

    Behavior:
      - Single-ticker (`ticker != "ALL"`): computes inline and returns
        `ready` with the bundle. Never writes to analyze_cache.
      - ALL: cache-only. Cache hit → status-only `ready`; the client
        downloads the ~130MB body separately via
        `GET /analyze-bundle/payload`. Background job running →
        `computing`. Neither → `not_computed`; frontend POSTs
        `/analyze-bundle/refresh` to start a compute.

    Why the split: pre-split, every cache-hit poll loaded the 130MB
    JSONB over asyncpg, json.loads-parsed it, and FastAPI re-serialised
    it — three synchronous CPU phases that hogged the event loop and
    starved concurrent fetchrow awaits past the 30s asyncpg
    command_timeout, surfacing as intermittent HTTP 500s on later polls.
    Polling now only checks `cached_at`/`payload_bytes` so the hot path
    is sub-millisecond.
    """
    if not pool:
        return {"error": "OI database not configured"}

    if ticker != "ALL":
        try:
            outcomes = await _discover_analyze_bundle_outcomes(pool)
            if not outcomes:
                return {"status": "ready", "bundle": None,
                        "error": "no_forward_return_columns_in_daily_features"}
            rows = await _fetch_analyze_bundle_rows(pool, ticker, metric, outcomes)
        except Exception as e:
            return {"status": "not_computed",
                    "error": f"db_fetch_failed: {type(e).__name__}: {e}"}
        if not rows:
            return {"status": "ready", "bundle": None, "error": "no_data"}
        # Fetch stored bin20 for this ticker so _compute_analyze_bundle_sync
        # takes the _assignments_from_is_bins path (same as ALL background job).
        # Without this, single-ticker falls to the Assigner (equal-count bins)
        # and the per_bin distribution is flat rather than reflecting is_bins ranks.
        bin20_lookup: Optional[dict] = None
        if mode in {"in_sample", "walk_forward", "train_test"}:
            # Use the information_schema-probed helper (same as secondary
            # endpoints) instead of a bare except Exception: pass.  The bare
            # except silently swallowed ANY connection error — not just
            # "column absent" — so a connection-state issue left by a prior
            # null-metric SQL error could cause the valid-metric bin20 fetch
            # to fail silently, leaving bin20_lookup=None → Assigner → flat
            # bins on the next Analyze (the deterministic step-1→2→3
            # regression).  _fetch_stored_bin20_by_metric probes
            # information_schema first: null-by-design metrics (no bin20_*
            # column) return {} cleanly; connection errors propagate instead
            # of being swallowed.
            _filter_pairs = [
                (ticker,
                 _r["trade_date"].isoformat()
                 if hasattr(_r["trade_date"], "isoformat")
                 else str(_r["trade_date"]))
                for _r in rows
            ]
            _by_metric = await _fetch_stored_bin20_by_metric(
                pool, mode, [metric], _filter_pairs
            )
            _lk = _by_metric.get(metric, {})
            if _lk:
                bin20_lookup = _lk
        bundle = await asyncio.to_thread(
            _compute_analyze_bundle_sync,
            rows, metric, ticker, mode, cutoff_date, outcomes,
            bin20_lookup=bin20_lookup,
        )
        return {"status": "ready", "bundle": bundle}

    # ALL mode: cache-only. Lean SELECT — no payload column.
    await _ensure_analyze_bundle_table(pool)
    cache_key = _analyze_bundle_cache_key(ticker, metric, mode, cutoff_date)

    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT cached_at, payload_bytes FROM analyze_cache_slim "
            "WHERE cache_key = $1",
            cache_key,
        )
    if row is not None:
        # Touch last_accessed for LRU. Fire-and-forget — no need to await
        # before returning status.
        try:
            async with pool.acquire() as conn:
                await conn.execute(
                    "UPDATE analyze_cache_slim SET last_accessed = NOW() "
                    "WHERE cache_key = $1", cache_key)
        except Exception:
            pass
        return {"status":        "ready",
                "cached_at":     str(row["cached_at"]),
                "payload_bytes": row["payload_bytes"]}

    if cache_key in _analyze_bundle_running:
        return {"status": "computing", "cache_key": cache_key}

    failed = _analyze_bundle_status.pop(cache_key, None)
    if failed and failed.get("status") == "failed":
        return {"status": "not_computed",
                "previous_error": failed.get("error")}
    return {"status": "not_computed"}


async def _touch_slim_last_accessed(pool, cache_key: str) -> None:
    """Fire-and-forget LRU touch from any of the v6 read endpoints.
    Deeper exploration (trade-meta + outcome reads) keeps the bundle warm."""
    try:
        async with pool.acquire() as conn:
            await conn.execute(
                "UPDATE analyze_cache_slim SET last_accessed = NOW() "
                "WHERE cache_key = $1", cache_key)
    except Exception:
        pass


@router.get("/analyze-bundle/payload")
async def analyze_bundle_payload(
    ticker:      str           = Query(...),
    metric:      str           = Query(...),
    mode:        str           = Query("in_sample"),
    cutoff_date: Optional[str] = Query(None),
    pool=Depends(get_oi_pool),
):
    """One-shot ALL-mode SLIM payload download (~1.3 MB).

    Returns the slim bundle: schema_version, metadata, per_bin, rolling_ic.
    Does NOT include trade_meta or per_outcome_returns — those load lazily
    via /analyze-bundle/trade-meta and /analyze-bundle/outcome. Single-
    ticker bundles return inline at /analyze-bundle and never hit this.

    Server-side fast path: asyncpg returns JSONB as `str`, we concat the
    response envelope around it in to_thread and emit via plain `Response`
    — skipping json.loads + FastAPI's re-serialise round trip.
    """
    from fastapi.responses import Response

    if not pool:
        return {"error": "OI database not configured"}
    if ticker != "ALL":
        return {"error": "payload endpoint is ALL-mode only; "
                         "single-ticker bundles return inline via /analyze-bundle"}

    await _ensure_analyze_bundle_table(pool)
    cache_key = _analyze_bundle_cache_key(ticker, metric, mode, cutoff_date)

    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT payload, cached_at FROM analyze_cache_slim "
            "WHERE cache_key = $1",
            cache_key,
        )
    if row is None:
        return {"error": "not_in_cache",
                "hint":  "poll /analyze-bundle until status=ready, then GET this"}

    await _touch_slim_last_accessed(pool, cache_key)

    payload_raw   = row["payload"]
    cached_at_str = str(row["cached_at"])

    def _compose_body() -> bytes:
        if isinstance(payload_raw, str):
            payload_json = payload_raw
        else:
            payload_json = json.dumps(payload_raw, allow_nan=False)
        envelope = (
            '{"status":"ready","cached_at":' + json.dumps(cached_at_str)
            + ',"bundle":' + payload_json + '}'
        )
        return envelope.encode("utf-8")

    body = await asyncio.to_thread(_compose_body)
    return Response(content=body, media_type="application/json")


@router.get("/analyze-bundle/trade-meta")
async def analyze_bundle_trade_meta(
    ticker:      str           = Query(...),
    metric:      str           = Query(...),
    mode:        str           = Query("in_sample"),
    cutoff_date: Optional[str] = Query(None),
    pool=Depends(get_oi_pool),
):
    """Lazy ALL-mode trade_meta download (~41 MB).

    Triggered by the frontend on first need: Overnight Gap mode entry,
    non-default outcome promotion, or Flat trade-table render for any
    non-default outcome. Cached client-side on `analyzeBundle.trade_meta`
    for the session.

    Same fast-path envelope concat as /payload — no parse/re-serialise.
    """
    from fastapi.responses import Response

    if not pool:
        return {"error": "OI database not configured"}
    if ticker != "ALL":
        return {"error": "trade-meta endpoint is ALL-mode only"}

    await _ensure_analyze_bundle_table(pool)
    cache_key = _analyze_bundle_cache_key(ticker, metric, mode, cutoff_date)

    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT payload FROM analyze_cache_trade_meta WHERE cache_key = $1",
            cache_key,
        )
    if row is None:
        return {"error": "not_in_cache",
                "hint":  "this metric's bundle hasn't been computed yet"}

    await _touch_slim_last_accessed(pool, cache_key)

    payload_raw = row["payload"]

    def _compose_body() -> bytes:
        if isinstance(payload_raw, str):
            payload_json = payload_raw
        else:
            payload_json = json.dumps(payload_raw, allow_nan=False)
        envelope = '{"trade_meta":' + payload_json + '}'
        return envelope.encode("utf-8")

    body = await asyncio.to_thread(_compose_body)
    return Response(content=body, media_type="application/json")


@router.get("/analyze-bundle/outcome")
async def analyze_bundle_outcome(
    ticker:      str           = Query(...),
    metric:      str           = Query(...),
    outcome:     str           = Query(...),
    mode:        str           = Query("in_sample"),
    cutoff_date: Optional[str] = Query(None),
    pool=Depends(get_oi_pool),
):
    """Lazy ALL-mode per-outcome download (~7.4 MB per outcome).

    Triggered by the frontend on first need for a specific outcome:
    Overnight Gap mode (1d_cc + 1d_oc), non-default outcome promotion, or
    Flat trade-table render. Cached client-side per outcome for the
    session — subsequent visits to the same outcome are zero-network.

    Same fast-path envelope concat as /payload — no parse/re-serialise.
    """
    from fastapi.responses import Response

    if not pool:
        return {"error": "OI database not configured"}
    if ticker != "ALL":
        return {"error": "outcome endpoint is ALL-mode only"}

    await _ensure_analyze_bundle_table(pool)
    cache_key = _analyze_bundle_cache_key(ticker, metric, mode, cutoff_date)

    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT payload FROM analyze_cache_outcome "
            "WHERE cache_key = $1 AND outcome = $2",
            cache_key, outcome,
        )
    if row is None:
        return {"error": "not_in_cache",
                "hint":  "this (metric, outcome) bundle hasn't been computed yet"}

    await _touch_slim_last_accessed(pool, cache_key)

    payload_raw = row["payload"]

    def _compose_body() -> bytes:
        if isinstance(payload_raw, str):
            payload_json = payload_raw
        else:
            payload_json = json.dumps(payload_raw, allow_nan=False)
        envelope = ('{"outcome":' + json.dumps(outcome)
                    + ',"data":' + payload_json + '}')
        return envelope.encode("utf-8")

    body = await asyncio.to_thread(_compose_body)
    return Response(content=body, media_type="application/json")


@router.post("/analyze-bundle/refresh")
async def analyze_bundle_refresh(
    ticker:      str           = Query(...),
    metric:      str           = Query(...),
    mode:        str           = Query("in_sample"),
    cutoff_date: Optional[str] = Query(None),
    pool=Depends(get_oi_pool),
):
    """Start a background bundle compute for ALL mode. Single-ticker
    bundles compute inline via GET — calling refresh on a single-ticker
    request is a 400-class error (returns {"error": ...}).

    Returns immediately with one of:
      {status: "computing", cache_key: "..."}
      {status: "busy",      cache_key: "..."}   # another job is in flight
    """
    if ticker != "ALL":
        return {"error": "refresh is ALL-mode only; single-ticker computes inline via GET"}
    if not pool:
        return {"error": "OI database not configured"}

    await _ensure_analyze_bundle_table(pool)
    cache_key = _analyze_bundle_cache_key(ticker, metric, mode, cutoff_date)

    if cache_key in _analyze_bundle_running:
        return {"status": "busy", "cache_key": cache_key}

    _analyze_bundle_running.add(cache_key)
    asyncio.create_task(
        _compute_analyze_bundle_bg(
            cache_key, ticker, metric, mode, cutoff_date, pool,
        )
    )
    return {"status": "computing", "cache_key": cache_key}


@router.post("/analyze-cache/invalidate")
async def analyze_cache_invalidate(
    ticker: Optional[str] = Query(None),
    metric: Optional[str] = Query(None),
    pool=Depends(get_oi_pool),
):
    """Drop entries from `analyze_cache` so the next GET /analyze-bundle
    request re-computes from source.

    Three scopes (most → least permissive):
      - No params                 → wipe everything (all versions, all keys).
      - `ticker` only             → wipe every entry for that ticker.
      - `ticker` AND `metric`     → wipe (ticker, metric, *, *) entries
                                    (covers in_sample / walk_forward /
                                    train_test and every cutoff date).

    When to call this:
      - Schema version unchanged but underlying daily_features values
        changed (re-ingestion, recomputed metrics, manual SQL fix).
      - A specific (ticker, metric) view looks stale and you don't want
        to nuke the whole table.
      - Debugging — force a fresh compute path without bumping schema.

    Schema bumps do NOT need this: incrementing
    `_ANALYZE_BUNDLE_SCHEMA_VERSION` makes stale-version keys
    unreachable on read AND the next call to _ensure_analyze_bundle_table
    runs a one-shot DELETE of non-current-version rows on startup.

    Returns: {"ok": true, "deleted": <row_count>, "scope": "<all|ticker|ticker+metric>"}
    Mirrors /secondary-scan/invalidate and /ic-batch/invalidate.
    Non-fatal on DB error.
    """
    if not pool:
        return {"error": "OI database not configured"}
    try:
        await _ensure_analyze_bundle_table(pool)
    except Exception:
        return {"error": "init_failed"}

    # Build the WHERE clause for the requested scope. Cache key shape is
    # `ab:v{N}:{ticker}:{metric}:{mode}:{cutoff}` — we match across all
    # schema versions so the endpoint also clears legacy rows the
    # startup auto-wipe hasn't reached yet.
    scope = "all"
    where = ""
    params: list = []
    if ticker and metric:
        scope = "ticker+metric"
        where = " WHERE cache_key LIKE $1"
        params.append(f"ab:v%:{ticker}:{metric}:%")
    elif ticker:
        scope = "ticker"
        where = " WHERE cache_key LIKE $1"
        params.append(f"ab:v%:{ticker}:%")

    # v6: invalidation cascades across the 3-table family. Count is from
    # the slim table (1 row per cache_key) — easier to interpret than the
    # raw sum (1 slim + 1 trade_meta + 12 outcome = 14 rows per cache_key).
    deleted = 0
    try:
        async with pool.acquire() as conn:
            async with conn.transaction():
                result = await conn.execute(
                    f"DELETE FROM analyze_cache_slim{where}", *params)
                await conn.execute(
                    f"DELETE FROM analyze_cache_trade_meta{where}", *params)
                await conn.execute(
                    f"DELETE FROM analyze_cache_outcome{where}", *params)
        if result and result.startswith("DELETE "):
            try:
                deleted = int(result.split()[-1])
            except ValueError:
                pass
    except Exception:
        return {"error": "delete_failed"}
    return {"ok": True, "deleted": deleted, "scope": scope}


@router.post("/analyze-primary/invalidate")
async def analyze_primary_invalidate(
    ticker: Optional[str] = Query(None),
    metric: Optional[str] = Query(None),
    outcome: Optional[str] = Query(None),
    pool=Depends(get_oi_pool),
):
    """Drop entries from `analyze_primary_cache`.

    Four scopes:
      - No params                         → wipe everything.
      - `ticker` only                     → all entries for that ticker.
      - `ticker` + `metric`               → all (ticker, metric) entries across
                                            every outcome, mode, and date range.
      - `ticker` + `metric` + `outcome`   → all (ticker, metric, outcome) entries.

    Call this after a data rewrite when the primary /analyze results are stale.
    (Also call /analyze-cache/invalidate for the bundle and the
    /global-metric-bins/invalidate for global_bins_cache — no shared path.)

    Schema bumps (increment _ANALYZE_PRIMARY_SCHEMA_VERSION) auto-invalidate
    on the next server start; this endpoint handles data-only staleness.

    Returns: {"ok": true, "deleted": <row_count>, "scope": "..."}
    """
    if not pool:
        return {"error": "OI database not configured"}
    try:
        await _ensure_analyze_primary_table(pool)
    except Exception:
        return {"error": "init_failed"}

    # Cache key shape: `ap:v{N}:{ticker}:{metric}:{outcome}:{mode}:{cutoff}:{from}:{to}`
    # LIKE patterns match across all schema versions so legacy rows are cleared too.
    scope = "all"
    sql = "DELETE FROM analyze_primary_cache"
    params: list = []
    if ticker and metric and outcome:
        scope = "ticker+metric+outcome"
        sql += " WHERE cache_key LIKE $1"
        params.append(f"ap:v%:{ticker}:{metric}:{outcome}:%")
    elif ticker and metric:
        scope = "ticker+metric"
        sql += " WHERE cache_key LIKE $1"
        params.append(f"ap:v%:{ticker}:{metric}:%")
    elif ticker:
        scope = "ticker"
        sql += " WHERE cache_key LIKE $1"
        params.append(f"ap:v%:{ticker}:%")

    deleted = 0
    try:
        async with pool.acquire() as conn:
            result = await conn.execute(sql, *params)
        if result and result.startswith("DELETE "):
            try:
                deleted = int(result.split()[-1])
            except ValueError:
                pass
    except Exception:
        return {"error": "delete_failed"}
    return {"ok": True, "deleted": deleted, "scope": scope}


@router.get("/ic-batch")
async def ic_batch(
    ticker:      str  = Query("ALL"),
    outcome:     str  = Query("ret_5d_fwd_oc"),
    window:      int  = Query(252, ge=20, le=1000),
    cutoff_date: Optional[str] = Query(None),
    stride:      int  = Query(3, ge=1, le=10),
    pool=Depends(get_oi_pool),
):
    """Per-metric long-run IC + sign-stability for all ~123 daily_features
    columns at the active mode. Drives the universe-wide IC stability
    leaderboard and the strength-vs-stability scatter (IC.5).

    Mode is encoded by `cutoff_date`:
      - cutoff_date set     → train_test: reference IC computed from
                              pre-cutoff windows only
      - cutoff_date not set → in_sample / walk_forward: reference uses
                              the full-history mean

    This endpoint is cache-read-only. Computation is always triggered via
    POST /ic-batch/refresh (both single-ticker and ALL). On a cache miss,
    returns one of three status responses so the frontend can react without
    blocking the HTTP connection:
      {"status": "not_ready"}   — no cache entry, POST /refresh to start
      {"status": "computing"}   — background job is running, poll again
      {"status": "failed"}      — background job crashed, POST /refresh to retry
    """
    if not pool:
        return {"error": "OI database not configured"}

    await _ensure_ic_batch_table(pool)

    mode_tag  = f"tt:{cutoff_date}" if cutoff_date else "default"
    cache_key = f"ic_batch:{ticker}:{outcome}:{window}:{mode_tag}:s{stride}"

    # Always check cache first — this endpoint never computes inline.
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT payload, cached_at FROM ic_batch_cache "
            "WHERE cache_key = $1", cache_key,
        )
    if row is not None:
        payload = row["payload"]
        if isinstance(payload, str):
            payload = json.loads(payload)
        cached_at_dt = row["cached_at"]
        return {
            "metrics":      payload.get("metrics", []),
            "ticker":       ticker,
            "outcome":      outcome,
            "window":       window,
            "cutoff_date":  cutoff_date,
            "cached_at":    str(cached_at_dt),
            "cached_at_ms": int(cached_at_dt.timestamp() * 1000),  # epoch ms — used by JS stale-cache guard (no string parsing)
            "from_cache":   True,
        }

    # Cache miss — return status so the frontend can prompt or poll.
    # POST /ic-batch/refresh starts the background job for any ticker.
    failed = _ic_batch_status.get(cache_key)
    if failed:
        return {
            "status":     "failed",
            "error":      failed["error"],
            "ticker":     ticker,
            "outcome":    outcome,
            "metrics":    [],
            "from_cache": False,
        }
    if cache_key in _ic_batch_running:
        return {
            "status":     "computing",
            "ticker":     ticker,
            "outcome":    outcome,
            "metrics":    [],
            "from_cache": False,
        }
    return {
        "status":     "not_ready",
        "ticker":     ticker,
        "outcome":    outcome,
        "metrics":    [],
        "from_cache": False,
    }


@router.post("/ic-batch/refresh")
async def ic_batch_refresh(
    ticker:      str  = Query("ALL"),
    outcome:     str  = Query("ret_5d_fwd_oc"),
    window:      int  = Query(252, ge=20, le=1000),
    cutoff_date: Optional[str] = Query(None),
    stride:      int  = Query(3, ge=1, le=10),
    pool=Depends(get_oi_pool),
):
    """Start a background IC batch computation for any ticker and return immediately.

    Works for both ticker=ALL and single tickers. The background job writes
    to ic_batch_cache on completion. Poll GET /ic-batch to check status.

    Concurrency: at most one IC background job runs at a time (global limit).
    If a different job is already running, returns {"status": "busy"} without
    starting a new one. The frontend treats busy as a queue entry: it keeps
    polling GET /ic-batch; when the running job finishes and the slot clears,
    the next not_ready → auto-trigger cycle starts this ticker's job.

    If this exact cache key is already running (double-click / double-trigger),
    returns {"status": "already_computing"}.
    """
    if not pool:
        return {"error": "OI database not configured"}

    await _ensure_ic_batch_table(pool)

    mode_tag  = f"tt:{cutoff_date}" if cutoff_date else "default"
    cache_key = f"ic_batch:{ticker}:{outcome}:{window}:{mode_tag}:s{stride}"

    # Per-key dedup: same ticker already running.
    if cache_key in _ic_batch_running:
        return {"status": "already_computing", "cache_key": cache_key}

    # Global one-job-at-a-time limit: different ticker running.
    # Prevents concurrent IC jobs from racing the VPS OOM killer.
    if len(_ic_batch_running) >= 1:
        return {
            "status":  "busy",
            "message": "Another IC computation is already running. "
                       "This ticker is queued and will start automatically when it finishes.",
        }

    # Clear any prior failure record before starting fresh.
    _ic_batch_status.pop(cache_key, None)
    _ic_batch_running.add(cache_key)

    is_all = (ticker.upper() == "ALL")
    if is_all:
        asyncio.create_task(
            _compute_ic_batch_all_bg(
                cache_key, ticker, outcome, window, cutoff_date, stride, pool,
            )
        )
    else:
        asyncio.create_task(
            _compute_ic_batch_single_bg(
                cache_key, ticker, outcome, window, cutoff_date, stride, pool,
            )
        )

    return {"status": "computing", "cache_key": cache_key}


@router.post("/ic-batch/invalidate")
async def ic_batch_invalidate(pool=Depends(get_oi_pool)):
    """Drop every row from `ic_batch_cache`.

    Use this when the underlying daily_features values have changed
    (recomputed core metrics, new ingestion runs, etc.) so the next
    /ic-batch request can no longer be served from a stale cached
    payload. Single-ticker mode recomputes inline on next fetch;
    ALL mode returns `not_ready` and waits for an explicit
    /ic-batch/refresh (the existing background-job path).

    Mirrors /global-metric-bins/invalidate. Non-fatal on DB error.
    """
    if pool:
        try:
            await _ensure_ic_batch_table(pool)
            async with pool.acquire() as conn:
                await conn.execute("DELETE FROM ic_batch_cache")
        except Exception:
            import logging
            logging.warning("ic_batch_cache DELETE failed during invalidate", exc_info=True)
    return {"ok": True}


# ── IC decomposition (IC.7) ──────────────────────────────────────────────

@router.get("/ic-decomp")
async def ic_decomp(
    metric:      str           = Query(...),
    outcome:     str           = Query("ret_5d_fwd_oc"),
    cutoff_date: Optional[str] = Query(None),
    pool=Depends(get_oi_pool),
):
    """Per-ticker rank-product contribution to cross-sectional IC (IC.7).

    Decomposes the ALL-mode cross-sectional IC into a per-ticker score.
    Each ticker's score is its mean daily rank-product contribution to the
    Spearman IC; the weighted sum of all scores (by n_days / n_total_days)
    recovers the full-history reference IC.

    Higher score = ticker consistently ranked in the right position for the
    metric-to-outcome relationship. Near-zero score = noise ticker.
    Negative score = consistently counter-directional.

    Mode follows the same convention as /ic-batch:
      • cutoff_date set   → train_test: reference IC from pre-cutoff only
      • cutoff_date unset → in_sample / walk_forward: full-history mean

    On-demand, in-memory cache — cleared on server restart.
    Elapsed compute time is logged and returned as `elapsed_s` so the
    first call can be benchmarked without needing a profiler.

    Returns:
      tickers            [{ticker, score, n_days}] sorted descending
      reference_ic       float
      n_days             int   total days with valid cross-sections
      n_tickers          int
      concentration_gini float | null  Gini of |score| (0=even, 1=concentrated)
      effective_n        float | null  (Σ|score|)²/Σ(score²)
      n_same_sign        int   tickers whose score agrees with reference_ic sign
      n_opposite_sign    int
      elapsed_s          float server-side compute time in seconds
      from_cache         bool
    """
    if not pool:
        return {"error": "OI database not configured"}

    import logging as _log
    import time as _time

    mode_tag  = f"tt:{cutoff_date}" if cutoff_date else "default"
    cache_key = f"ic_decomp:{metric}:{outcome}:{mode_tag}"

    if cache_key in _IC_DECOMP_CACHE:
        cached = dict(_IC_DECOMP_CACHE[cache_key])
        cached["from_cache"] = True
        return cached

    try:
        async with pool.acquire() as conn:
            db_rows = await conn.fetch(
                f'SELECT ticker, trade_date, "{metric}", "{outcome}" '
                f'FROM daily_features '
                f'WHERE "{metric}" IS NOT NULL AND "{outcome}" IS NOT NULL '
                f'ORDER BY trade_date, ticker',
                timeout=90,
            )
    except Exception as exc:
        _log.exception("ic_decomp: DB fetch failed metric=%s outcome=%s", metric, outcome)
        return {"error": f"DB fetch failed: {type(exc).__name__}: {exc}"}

    rows = [dict(r) for r in db_rows]

    from app.routers.ic_compute import ic_decompose_cross_sectional

    t0     = _time.perf_counter()
    result = await asyncio.to_thread(
        ic_decompose_cross_sectional,
        rows, metric, outcome,
        cutoff_date=cutoff_date,
    )
    elapsed = round(_time.perf_counter() - t0, 3)
    del rows

    _log.info(
        "ic_decomp: metric=%s outcome=%s mode=%s n_tickers=%d n_days=%d elapsed=%.3fs",
        metric, outcome, mode_tag,
        result.get("n_tickers", 0), result.get("n_days", 0), elapsed,
    )

    result["elapsed_s"]  = elapsed
    result["from_cache"] = False
    _IC_DECOMP_CACHE[cache_key] = result
    return result


# ── Threshold Drift (walk-forward bin boundaries over time) ──────────────

# Cache key is salted with this version so deploys that change the
# computation formula automatically invalidate stale cached responses.
_THRESHOLD_DRIFT_CACHE_VERSION = "v3-canonical-month-end"
_THRESHOLD_DRIFT_CACHE: dict = {}
# Parallel dict tracking the wall-clock load time per cache_key. Surfaced by
# /threshold-drift/meta so the pane's collapsed-header breadcrumb can show
# "last: …" on page load without fetching the full payload. Cleared
# alongside _THRESHOLD_DRIFT_CACHE on /threshold-drift/invalidate and on
# server restart (in-memory only — matches the cache's own lifetime).
_THRESHOLD_DRIFT_LOADED_AT: dict = {}


@router.get("/threshold-drift")
async def threshold_drift(
    metric:   str = Query(...),
    outcome:  str = Query("ret_5d_fwd_oc"),
    ticker:   str = Query("ALL"),
    n_bins:   int = Query(20, ge=2, le=20),
    bins:     str = Query("1,5,10,15,20",
                          description="Comma-separated bin numbers to track"),
    date_from: Optional[str] = Query(None),
    date_to:   Optional[str] = Query(None),
    force:     bool = Query(False, description="Skip cache read; recompute and overwrite"),
    pool=Depends(get_oi_pool),
):
    """For each requested bin K (1..n_bins), return its upper-edge
    threshold value sampled at month-end as the walk-forward universe
    grows. Cross-ticker aggregation is median + IQR. Single-ticker mode
    returns the raw ticker thresholds.

    The 'in_sample_ref' map carries the full-history threshold per bin
    (median across tickers in ALL mode, raw value in single-ticker
    mode). Frontend draws it as a horizontal dotted reference line so
    you can eyeball whether today's bin boundary is far from the
    walk-forward boundary at any historical point.
    """
    if not pool:
        return {"error": "OI database not configured"}
    try:
        bins_to_track = sorted({int(b.strip()) for b in bins.split(",") if b.strip()})
        bins_to_track = [b for b in bins_to_track if 1 <= b <= n_bins]
    except ValueError:
        return {"error": "bins must be comma-separated integers"}
    if not bins_to_track:
        bins_to_track = [n_bins]

    cache_key = (f"{_THRESHOLD_DRIFT_CACHE_VERSION}|"
                 f"{ticker}|{metric}|{outcome}|{n_bins}|"
                 f"{','.join(str(b) for b in bins_to_track)}|"
                 f"{date_from or ''}|{date_to or ''}")
    if not force and cache_key in _THRESHOLD_DRIFT_CACHE:
        return _THRESHOLD_DRIFT_CACHE[cache_key]

    where = [f"{metric} IS NOT NULL", f"{outcome} IS NOT NULL"]
    params: list = []
    p = 1
    if ticker and ticker != "ALL":
        where.append(f"ticker = ${p}"); params.append(ticker); p += 1
    if date_from:
        where.append(f"trade_date >= ${p}")
        params.append(_date.fromisoformat(date_from)); p += 1
    if date_to:
        where.append(f"trade_date <= ${p}")
        params.append(_date.fromisoformat(date_to)); p += 1

    try:
        async with pool.acquire() as conn:
            db_rows = await conn.fetch(
                f"SELECT ticker, trade_date, {metric} "
                f"FROM daily_features "
                f"WHERE {' AND '.join(where)} "
                f"ORDER BY ticker, trade_date",
                *params, timeout=180)
        rows = [dict(r) for r in db_rows]
        for r in rows:
            r["trade_date"] = str(r["trade_date"])

        samples, full_per_ticker, full_thr = _walk_forward_thresholds(
            rows, metric, n_bins, bins_to_track,
            warmup=_DEFAULT_WALKFWD_WARMUP)

        # Tickers with enough full history to participate. Sort for the
        # frontend single-ticker dropdown.
        tickers_eligible = sorted(t for t, m in full_per_ticker.items() if m)

        from collections import defaultdict as _dd

        # ── Native (raw threshold values) aggregation — kept for the
        # Single-ticker view where dimensionality is consistent.
        grouped_native: dict = _dd(lambda: _dd(list))
        for s in samples:
            grouped_native[s["date"]][s["bin"]].append(s["threshold"])

        # ── Drift-ratio aggregation — for each sample, divide its
        # walk-forward threshold by its OWN ticker's full-history
        # threshold (dimensionless). Aggregate ratios across tickers.
        # This is the meaningful all-tickers view.
        grouped_ratio: dict = _dd(lambda: _dd(list))
        for s in samples:
            full = s.get("threshold_full_ticker")
            if full is None:
                continue
            if abs(full) < 1e-10:
                continue   # avoid divide-by-near-zero (metrics that cross zero)
            grouped_ratio[s["date"]][s["bin"]].append(s["threshold"] / full)

        def _aggregate(grouped):
            series_out = {str(b): [] for b in bins_to_track}
            for date_s in sorted(grouped.keys()):
                for b in bins_to_track:
                    vals = grouped[date_s].get(b, [])
                    if not vals:
                        continue
                    series_out[str(b)].append({
                        "date":      date_s,
                        "median":    round(float(np.median(vals)), 6),
                        "q25":       round(float(np.percentile(vals, 25)), 6),
                        "q75":       round(float(np.percentile(vals, 75)), 6),
                        "n_tickers": int(len(vals)),
                    })
            return series_out

        series_native = _aggregate(grouped_native)
        series_ratio  = _aggregate(grouped_ratio)

        # ── Per-ticker series (for the Single-ticker native view).
        # Map: ticker -> {bin: [{date, threshold}, ...]}
        per_ticker_series: dict = {}
        for s in samples:
            tkr = s["ticker"]
            per_ticker_series.setdefault(tkr, {}).setdefault(str(s["bin"]), []).append({
                "date":      s["date"],
                "threshold": s["threshold"],
            })

        # Reference values for the dotted horizontal lines:
        #   native_ref: median of per-ticker full-history thresholds per bin
        #   ratio_ref:  1.0 (always)
        native_ref: dict = {}
        for b in bins_to_track:
            vals = full_thr.get(b) or []
            native_ref[str(b)] = round(float(np.median(vals)), 6) if vals else None

        full_per_ticker_out = {
            t: {str(k): round(float(v), 6) for k, v in m.items()}
            for t, m in full_per_ticker.items() if m
        }

        out = {
            "metric":           metric,
            "outcome":          outcome,
            "ticker":           ticker,
            "n_bins":           n_bins,
            "bins":             bins_to_track,
            "warmup":           _DEFAULT_WALKFWD_WARMUP,
            "total_rows":       len(rows),
            "tickers_eligible": tickers_eligible,
            # Drift ratio (dimensionless; default view across tickers)
            "series_ratio":     series_ratio,
            "ratio_ref":        1.0,
            # Native units (raw threshold values)
            "series_native":    series_native,
            "native_ref":       native_ref,
            # Per-ticker raw series — used by the Single-ticker view
            "per_ticker":       per_ticker_series,
            "per_ticker_full":  full_per_ticker_out,
        }
        from datetime import datetime as _dt
        _THRESHOLD_DRIFT_CACHE[cache_key] = out
        _THRESHOLD_DRIFT_LOADED_AT[cache_key] = (
            _dt.utcnow().isoformat() + "Z"
        )
        return out
    except Exception as exc:
        return {
            "error":   f"{type(exc).__name__}: {exc}",
            "metric":  metric, "outcome": outcome, "ticker": ticker,
            "bins":    bins_to_track,
            "series_ratio":  {str(b): [] for b in bins_to_track},
            "series_native": {str(b): [] for b in bins_to_track},
            "ratio_ref":     1.0,
            "native_ref":    {str(b): None for b in bins_to_track},
            "tickers_eligible": [],
            "per_ticker":       {},
            "per_ticker_full":  {},
        }


@router.post("/threshold-drift/invalidate")
async def threshold_drift_invalidate():
    _THRESHOLD_DRIFT_CACHE.clear()
    _THRESHOLD_DRIFT_LOADED_AT.clear()
    return {"ok": True}


@router.get("/threshold-drift/meta")
async def threshold_drift_meta(
    metric:    str           = Query(...),
    outcome:   str           = Query("ret_5d_fwd_oc"),
    ticker:    str           = Query("ALL"),
    n_bins:    int           = Query(20, ge=2, le=20),
    bins:      str           = Query("1,5,10,15,20"),
    date_from: Optional[str] = Query(None),
    date_to:   Optional[str] = Query(None),
):
    """Cheap metadata-only check for the threshold-drift in-memory cache.

    Returns `{exists, cached_at}` for the given selector tuple without
    running the threshold-drift compute. Powers the collapsed-pane
    breadcrumb so the pane can show its last-computed timestamp on page
    load without firing the full payload-bearing /threshold-drift call.

    Since the threshold-drift cache is in-memory only (no DB persistence),
    `cached_at` is the in-process load timestamp and resets on server
    restart. `exists` will be False after a restart even if the same
    params previously had cached data — frontend renders "no data yet"
    in that case until the user expands the pane and the cache rewarms.
    """
    try:
        bins_to_track = sorted({int(b.strip()) for b in bins.split(",") if b.strip()})
        bins_to_track = [b for b in bins_to_track if 1 <= b <= n_bins]
    except ValueError:
        return {"exists": False, "cached_at": None,
                "error": "bins must be comma-separated integers"}
    if not bins_to_track:
        bins_to_track = [n_bins]
    cache_key = (f"{_THRESHOLD_DRIFT_CACHE_VERSION}|"
                 f"{ticker}|{metric}|{outcome}|{n_bins}|"
                 f"{','.join(str(b) for b in bins_to_track)}|"
                 f"{date_from or ''}|{date_to or ''}")
    return {
        "exists":    cache_key in _THRESHOLD_DRIFT_CACHE,
        "cached_at": _THRESHOLD_DRIFT_LOADED_AT.get(cache_key),
    }


# ── Corner Scan endpoints ─────────────────────────────────────────────────────

_corner_scan_tables_ensured: bool = False

_DDL_CORNER_2F = """\
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

_DDL_CORNER_1F = """\
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

# ── Migration DDL for pre-existing corner-scan tables ──
# Bucket A step 1: adds `mode` and `scanned_at` columns + swaps the PK to
# include `mode`. Idempotent — safe to run on already-migrated tables.
# Existing rows are backfilled to mode='walk_forward' via the column
# DEFAULT (i.e., implicit on column creation); scanned_at gets NOW() at
# migration time, which represents "first known timestamp" for those
# pre-existing rows. New scans set scanned_at explicitly via the writer.
_DDL_CORNER_2F_MIGRATE_COLS = (
    "ALTER TABLE corner_scan_2f "
    "ADD COLUMN IF NOT EXISTS mode TEXT NOT NULL DEFAULT 'walk_forward'"
)
_DDL_CORNER_2F_MIGRATE_TS = (
    "ALTER TABLE corner_scan_2f "
    "ADD COLUMN IF NOT EXISTS scanned_at TIMESTAMPTZ NOT NULL DEFAULT NOW()"
)
_DDL_CORNER_2F_MIGRATE_PK = """\
DO $$
DECLARE
    pk_has_mode BOOLEAN;
BEGIN
    SELECT EXISTS (
        SELECT 1
          FROM information_schema.key_column_usage kcu
          JOIN information_schema.table_constraints tc
            ON kcu.constraint_name = tc.constraint_name
         WHERE tc.table_name      = 'corner_scan_2f'
           AND tc.constraint_type = 'PRIMARY KEY'
           AND kcu.column_name    = 'mode'
    ) INTO pk_has_mode;
    IF NOT pk_has_mode THEN
        ALTER TABLE corner_scan_2f DROP CONSTRAINT IF EXISTS corner_scan_2f_pkey;
        ALTER TABLE corner_scan_2f
            ADD PRIMARY KEY (primary_metric, secondary_metric, corner_direction, outcome, mode);
    END IF;
END $$;
"""

_DDL_CORNER_1F_MIGRATE_COLS = (
    "ALTER TABLE corner_scan_1f "
    "ADD COLUMN IF NOT EXISTS mode TEXT NOT NULL DEFAULT 'walk_forward'"
)
_DDL_CORNER_1F_MIGRATE_TS = (
    "ALTER TABLE corner_scan_1f "
    "ADD COLUMN IF NOT EXISTS scanned_at TIMESTAMPTZ NOT NULL DEFAULT NOW()"
)
_DDL_CORNER_1F_MIGRATE_PK = """\
DO $$
DECLARE
    pk_has_mode BOOLEAN;
BEGIN
    SELECT EXISTS (
        SELECT 1
          FROM information_schema.key_column_usage kcu
          JOIN information_schema.table_constraints tc
            ON kcu.constraint_name = tc.constraint_name
         WHERE tc.table_name      = 'corner_scan_1f'
           AND tc.constraint_type = 'PRIMARY KEY'
           AND kcu.column_name    = 'mode'
    ) INTO pk_has_mode;
    IF NOT pk_has_mode THEN
        ALTER TABLE corner_scan_1f DROP CONSTRAINT IF EXISTS corner_scan_1f_pkey;
        ALTER TABLE corner_scan_1f
            ADD PRIMARY KEY (metric, extreme, outcome, mode);
    END IF;
END $$;
"""

_CS_2F_SORT_WHITELIST: frozenset = frozenset({
    "primary_metric", "secondary_metric", "corner_direction", "outcome",
    "d_avg_ret", "d_ret_per_day", "d_n",
    "q_avg_ret", "q_ret_per_day", "q_n",
})
_CS_1F_SORT_WHITELIST: frozenset = frozenset({
    "metric", "extreme", "outcome",
    "d_avg_ret", "d_ret_per_day", "d_n",
    "q_avg_ret", "q_ret_per_day", "q_n",
})


async def _ensure_corner_scan_tables(pool) -> None:
    global _corner_scan_tables_ensured
    if _corner_scan_tables_ensured:
        return
    async with pool.acquire() as conn:
        # 1. Create fresh tables (no-op if they already exist). New tables
        #    already include `mode` + `scanned_at` + the mode-inclusive PK.
        await conn.execute(_DDL_CORNER_2F)
        await conn.execute(_DDL_CORNER_1F)
        # 2. Migrate any pre-existing tables that were created with the
        #    old schema. ADD COLUMN IF NOT EXISTS is a no-op when the
        #    column already exists; the DO block PK swap is gated by an
        #    information_schema check so it's a no-op on already-swapped
        #    tables.
        await conn.execute(_DDL_CORNER_2F_MIGRATE_COLS)
        await conn.execute(_DDL_CORNER_2F_MIGRATE_TS)
        await conn.execute(_DDL_CORNER_2F_MIGRATE_PK)
        await conn.execute(_DDL_CORNER_1F_MIGRATE_COLS)
        await conn.execute(_DDL_CORNER_1F_MIGRATE_TS)
        await conn.execute(_DDL_CORNER_1F_MIGRATE_PK)
    _corner_scan_tables_ensured = True


@router.get("/corner-scan/meta")
async def corner_scan_meta(
    mode: str = Query("walk_forward"),
    pool=Depends(get_oi_pool),
):
    """Row counts, as_of date, scan timestamp, and distinct-metric count for
    both corner-scan tables — filtered by `mode`.

    `mode` defaults to walk_forward (the only mode with data in Bucket A's
    pre-IS-batch state). Other modes return zero counts and null timestamps;
    the frontend renders a placeholder for those.

    `scanned_at_{2f,1f}` is `MAX(scanned_at)` per mode — used by the pane
    breadcrumb in the consistent "last: YYYY-MM-DD HH:MM:SS" format across
    all 6 panes.
    """
    if not pool:
        return {"error": "OI database not configured"}
    await _ensure_corner_scan_tables(pool)
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """SELECT
                 (SELECT COUNT(*)            FROM corner_scan_2f WHERE mode = $1) AS count_2f,
                 (SELECT MAX(as_of)          FROM corner_scan_2f WHERE mode = $1) AS as_of_2f,
                 (SELECT MAX(scanned_at)     FROM corner_scan_2f WHERE mode = $1) AS scanned_at_2f,
                 (SELECT COUNT(*)            FROM corner_scan_1f WHERE mode = $1) AS count_1f,
                 (SELECT MAX(as_of)          FROM corner_scan_1f WHERE mode = $1) AS as_of_1f,
                 (SELECT MAX(scanned_at)     FROM corner_scan_1f WHERE mode = $1) AS scanned_at_1f,
                 (SELECT COUNT(DISTINCT primary_metric)
                    FROM corner_scan_2f WHERE mode = $1)                          AS n_metrics""",
            mode,
        )
        # Sorted distinct metric list for filter dropdowns — sourced from
        # corner_scan_1f. Apply the same display-only exclusion as 1f/2f
        # endpoints: drop Family 2 entirely, drop _pc variants in Family 4/5.
        mc_exists = await conn.fetchval(
            "SELECT EXISTS (SELECT 1 FROM information_schema.tables"
            "  WHERE table_name = 'metric_classification')"
        )
        if mc_exists:
            metric_rows = await conn.fetch(
                """SELECT DISTINCT metric FROM corner_scan_1f
                   WHERE mode = $1
                     AND metric NOT IN (
                         SELECT metric FROM metric_classification
                         WHERE family_num = 2
                            OR (family_num IN (4,5) AND RIGHT(metric,3) = '_pc')
                     )
                   ORDER BY metric""",
                mode,
            )
        else:
            metric_rows = await conn.fetch(
                "SELECT DISTINCT metric FROM corner_scan_1f WHERE mode = $1 ORDER BY metric",
                mode,
            )
        metrics_list = [r["metric"] for r in metric_rows]
    def _iso(v):
        return v.isoformat() if v is not None else None
    return {
        "mode":          mode,
        "count_2f":      int(row["count_2f"]),
        "as_of_2f":      _iso(row["as_of_2f"]),
        "scanned_at_2f": _iso(row["scanned_at_2f"]),
        "count_1f":      int(row["count_1f"]),
        "as_of_1f":      _iso(row["as_of_1f"]),
        "scanned_at_1f": _iso(row["scanned_at_1f"]),
        "n_metrics":     int(row["n_metrics"]),
        "metrics":       metrics_list,
    }


@router.get("/corner-scan/2f")
async def corner_scan_2f_endpoint(
    primary_metric:   Optional[str] = Query(None),
    secondary_metric: Optional[str] = Query(None),
    corner_direction: Optional[str] = Query(None),
    outcome:          Optional[str] = Query(None),
    mode:             str           = Query("walk_forward"),
    min_d_n:          int           = Query(300, ge=0),
    sort_key:         str           = Query("d_ret_per_day"),
    sort_dir:         str           = Query("desc"),
    limit:            int           = Query(50, ge=1, le=2000),
    offset:           int           = Query(0, ge=0),
    pool=Depends(get_oi_pool),
):
    """Filtered + sorted 2-factor corner rows for a single binning mode.

    Filters applied in SQL; sort key is whitelisted against _CS_2F_SORT_WHITELIST.
    d_n >= min_d_n is the primary quality gate (default 300). Supports
    server-side pagination via limit + offset.

    Display filter (not a build filter): excludes Family-2 metrics (spot_pc,
    spot_co — never valid metrics) and _pc-suffixed metrics in Family 4 and 5
    (OI-by-strike and OI-change families; _pc variants are unavailable at
    morning analysis time). Family-12 _pc metrics (option volume) are kept.
    Uses metric_classification table; if the table is absent, no filter applied.

    Response includes per-row `mode` and `scanned_at` so the frontend can
    display the "last: YYYY-MM-DD HH:MM:SS · <Mode>" breadcrumb without a
    second meta call.
    """
    if not pool:
        return {"rows": [], "total": 0, "mode": mode, "status": "no_db"}
    await _ensure_corner_scan_tables(pool)

    if sort_key not in _CS_2F_SORT_WHITELIST:
        sort_key = "d_ret_per_day"
    dir_sql   = "ASC"   if sort_dir == "asc" else "DESC"
    nulls_sql = "NULLS FIRST" if sort_dir == "asc" else "NULLS LAST"

    conds:  list[str] = ["d_n >= $1", "mode = $2"]
    params: list      = [min_d_n, mode]
    p = 3

    if primary_metric:
        conds.append(f"primary_metric   = ${p}"); params.append(primary_metric);   p += 1
    if secondary_metric:
        conds.append(f"secondary_metric = ${p}"); params.append(secondary_metric); p += 1
    if corner_direction:
        conds.append(f"corner_direction = ${p}"); params.append(corner_direction); p += 1
    if outcome:
        conds.append(f"outcome = ${p}");          params.append(outcome);          p += 1

    # Display-only metric exclusion: Family 2 always; Family 4+5 _pc only.
    # Wrapped in a try/except so a missing metric_classification table degrades
    # gracefully (no filter applied) rather than erroring the endpoint.
    _EXCL = (
        "NOT IN ("
        "  SELECT metric FROM metric_classification"
        "  WHERE family_num = 2"
        "  OR (family_num IN (4,5) AND RIGHT(metric,3) = '_pc')"
        ")"
    )
    async with pool.acquire() as conn:
        mc_exists = await conn.fetchval(
            "SELECT EXISTS (SELECT 1 FROM information_schema.tables"
            "  WHERE table_name = 'metric_classification')"
        )
        if mc_exists:
            conds.append(f"primary_metric   {_EXCL}")
            conds.append(f"secondary_metric {_EXCL}")

        where = " AND ".join(conds)
        order = f"{sort_key} {dir_sql} {nulls_sql}"

        total = await conn.fetchval(
            f"SELECT COUNT(*) FROM corner_scan_2f WHERE {where}", *params
        )
        rows = await conn.fetch(
            f"""SELECT primary_metric, secondary_metric, corner_direction, outcome,
                       d_avg_ret, d_ret_per_day, d_n,
                       q_avg_ret, q_ret_per_day, q_n,
                       as_of, scanned_at, mode
                FROM corner_scan_2f WHERE {where}
                ORDER BY {order} LIMIT {limit} OFFSET {offset}""",
            *params,
        )
    return {
        "rows":   [dict(r) for r in rows],
        "total":  int(total),
        "mode":   mode,
        "status": "ok" if int(total) > 0 else "no_data",
    }


@router.get("/corner-scan/1f")
async def corner_scan_1f_endpoint(
    metric:   Optional[str] = Query(None),
    extreme:  Optional[str] = Query(None),
    outcome:  Optional[str] = Query(None),
    mode:     str           = Query("walk_forward"),
    min_d_n:  int           = Query(300, ge=0),
    sort_key: str           = Query("d_ret_per_day"),
    sort_dir: str           = Query("desc"),
    limit:    int           = Query(50, ge=1, le=2000),
    offset:   int           = Query(0, ge=0),
    pool=Depends(get_oi_pool),
):
    """Filtered + sorted 1-factor extreme-bin rows for a single binning mode.

    `mode` defaults to walk_forward; requesting a mode with no rows returns
    an empty list cleanly (status="no_data"). Supports server-side pagination
    via limit + offset. Same display-only metric exclusion as /corner-scan/2f
    (Family 2 always; Family 4+5 _pc only).
    """
    if not pool:
        return {"rows": [], "total": 0, "mode": mode, "status": "no_db"}
    await _ensure_corner_scan_tables(pool)

    if sort_key not in _CS_1F_SORT_WHITELIST:
        sort_key = "d_ret_per_day"
    dir_sql   = "ASC"   if sort_dir == "asc" else "DESC"
    nulls_sql = "NULLS FIRST" if sort_dir == "asc" else "NULLS LAST"

    conds:  list[str] = ["d_n >= $1", "mode = $2"]
    params: list      = [min_d_n, mode]
    p = 3

    if metric:
        conds.append(f"metric = ${p}");   params.append(metric);   p += 1
    if extreme:
        conds.append(f"extreme = ${p}");  params.append(extreme);  p += 1
    if outcome:
        conds.append(f"outcome = ${p}");  params.append(outcome);  p += 1

    _EXCL_1F = (
        "metric NOT IN ("
        "  SELECT metric FROM metric_classification"
        "  WHERE family_num = 2"
        "  OR (family_num IN (4,5) AND RIGHT(metric,3) = '_pc')"
        ")"
    )
    async with pool.acquire() as conn:
        mc_exists = await conn.fetchval(
            "SELECT EXISTS (SELECT 1 FROM information_schema.tables"
            "  WHERE table_name = 'metric_classification')"
        )
        if mc_exists:
            conds.append(_EXCL_1F)

        where = " AND ".join(conds)
        order = f"{sort_key} {dir_sql} {nulls_sql}"

        total = await conn.fetchval(
            f"SELECT COUNT(*) FROM corner_scan_1f WHERE {where}", *params
        )
        rows = await conn.fetch(
            f"""SELECT metric, extreme, outcome,
                       d_avg_ret, d_ret_per_day, d_n,
                       q_avg_ret, q_ret_per_day, q_n,
                       as_of, scanned_at, mode
                FROM corner_scan_1f WHERE {where}
                ORDER BY {order} LIMIT {limit} OFFSET {offset}""",
            *params,
        )
    return {
        "rows":   [dict(r) for r in rows],
        "total":  int(total),
        "mode":   mode,
        "status": "ok" if int(total) > 0 else "no_data",
    }
