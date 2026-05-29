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
    _sec_scan_table_ensured = True


def _sec_structural_key(
    ticker: str, metric: str, outcome: str, mode: str,
    cutoff: str, n_bins: int, selected_bins,
) -> str:
    """Stable DB lookup key — no date, no filtered_dates.  The key encodes every
    parameter that can change the result; data_as_of lives separately in the row."""
    bins_sorted = sorted(selected_bins) if selected_bins else []
    bins_hash = hashlib.sha256(
        json.dumps(bins_sorted, separators=(",", ":")).encode()
    ).hexdigest()[:12]
    cutoff_str = cutoff or "null"
    return f"sec:{ticker}:{metric}:{outcome}:{mode}:{cutoff_str}:{n_bins}:{bins_hash}"


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
        pass  # non-fatal — in-memory _SEC_SCORE_CACHE still works

# ── Trade-detail cache (W1) ────────────────────────────────────────────────────
# Populated by /analyze; read by GET /trades and GET /trades/csv.
# Key: "{ticker}:{metric}:{outcome}:{mode}:{cutoff_date}"
# Value: list of full trade-detail dicts (one per pair).
_TRADE_CACHE: dict = {}
_TRADE_CACHE_MAX = 5  # keep last 5 unique (ticker,metric,outcome,mode) analyses

# ── Full-response cache (W2) ──────────────────────────────────────────────────
# Caches the complete /analyze response dict so mode switches and repeat
# Analyzes skip the full 25s computation.  Staleness is checked on every
# hit via one cheap SELECT MAX(trade_date).  Keyed by all params that
# affect the computation result.
# Max 4 entries × ~15MB each ≈ 60MB ceiling.
_ANALYZE_CACHE: dict = {}
_ANALYZE_CACHE_MAX = 4


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
) -> None:
    """Synchronous secondary score computation — runs in thread-pool executor.
    Sets _sec_scan_running True at entry, False in finally.
    When db_write_params is provided, persists result to sec_scan_cache on success."""
    global _sec_scan_running
    _sec_scan_running = True
    try:
        from app.routers.row_compute import WalkForwardSpec, filter_by_assignments
        spec = WalkForwardSpec()
        filtered, dropped, universe = filter_by_assignments(
            rows, spec, primary_metric, selected_primary_bins, is_all, filtered_dates,
        )
        metrics = _sec_score_metrics(filtered, outcome_col, feature_cols, is_all, n_bins, spec)
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
            "dropped_warmup_n": dropped,
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
                pass
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
    pool=Depends(get_oi_pool),
):
    """Full analysis payload for one ticker/metric/outcome combo."""
    if not pool:
        return {"error": "OI database not configured"}

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

    # ── W2: full-response cache (staleness-checked) ───────────────────────────
    _ac_key = (f"{ticker}:{metric}:{outcome}:{spec.kind}:"
               f"{spec.cutoff.isoformat() if spec.kind == 'train_test' else ''}:"
               f"{date_from or ''}:{date_to or ''}")
    _tlog(f'W2 key="{_ac_key}" cache_size={len(_ANALYZE_CACHE)} in_cache={_ac_key in _ANALYZE_CACHE}')

    if _ac_key in _ANALYZE_CACHE:
        _tlog('W2 cache hit')
        _hit = dict(_ANALYZE_CACHE[_ac_key])  # shallow copy — don't mutate the stored entry
        _hit["_handler_ms"] = round((_time.perf_counter() - _t0) * 1000)
        return _hit

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

    horizon = _parse_horizon(outcome)

    _max_trade_date: str = ""   # set after each branch's DB fetch; stored in cache for staleness

    # ── Data fetch & bucketing ────────────────────────────────────────────
    if is_all:
        # Per-ticker normalization: fetch all tickers, decile each independently
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                f"SELECT ticker, trade_date, {metric}, {outcome}, spot_co, spot_pc "
                f"FROM daily_features "
                f"WHERE {metric} IS NOT NULL AND {outcome} IS NOT NULL {date_conditions} "
                f"ORDER BY ticker, trade_date", *params)
        row_dicts = [dict(r) for r in rows]
        if row_dicts:
            _max_trade_date = str(max(r['trade_date'] for r in row_dicts))
        _tlog(f'ALL db_fetch1 ({len(row_dicts)} rows)')

        by_ticker: dict = defaultdict(list)
        all_open_by_tkr_date: dict = {}
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
            if r.get('spot_co') is not None:
                try:
                    all_open_by_tkr_date[(r['ticker'], str(r['trade_date']))] = round(float(r['spot_co']), 2)
                except (ValueError, TypeError):
                    pass

        # Route bucketing through the row_compute layer. Both in-sample
        # and walk-forward branches now flow through one path; the
        # method choice picks the Assigner. Downstream of this block
        # (decile_stats_20, equity_by_decile, trade_calendar, etc.) is
        # unchanged.
        from app.routers.row_compute import ASSIGNERS, _validate_assignments
        # `spec` was constructed at the top of the function.
        assigner = ASSIGNERS[spec.kind](spec)
        state10 = assigner.fit(row_dicts, metric, 10, True)
        a10 = assigner.assign(row_dicts, metric, 10, True, state10, outcome)
        _validate_assignments(a10, 10)
        state20 = assigner.fit(row_dicts, metric, 20, True)
        a20 = assigner.assign(row_dicts, metric, 20, True, state20, outcome)
        _validate_assignments(a20, 20)
        _tlog('ALL bin_assign (10-bin + 20-bin)')

        # Cross-reference 10-bin and 20-bin assignments by (ticker, date).
        # A ticker with 10..19 rows clears the 10-bin threshold but not
        # the 20-bin one (`_bucket_pairs_per_ticker` excludes tickers
        # with < n rows). Legacy emits those rows with `decile20 = 0`
        # and does NOT add them to any 20-bin bucket; preserve that
        # exact semantics here for bit-equivalence on the regression
        # check.
        # `dropped_reason is None` excludes train-test pre_cutoff rows
        # (where bin IS set but the row is training-window — included on
        # the assignment for a future side-by-side view, excluded from
        # the standard test-period aggregation). For in_sample and
        # walk_forward this is a no-op since dropped_reason is always
        # None when bin is not None.
        key_b20: dict = {(a.ticker, a.trade_date): a.bin
                         for a in a20
                         if a.bin is not None and a.dropped_reason is None}
        valid_a10 = [a for a in a10
                     if a.bin is not None and a.dropped_reason is None]
        # Chronological across tickers, with legacy within-date tie-break:
        #   in-sample: legacy iterates buckets sequentially then stable-sorts
        #     by date, so same-date rows come out in (bin, ticker, x) order.
        #   walk-forward: legacy has no bucket-iteration step; same-date rows
        #     come out in alphabetical-ticker order (matches mine without
        #     extra tie-break).
        # Pairs iteration order is chronological (walk_forward) or
        # (date, bin, ticker, x) (in_sample/train_test) — matching legacy ordering.
        if walk_forward:
            valid_a10.sort(key=lambda a: a.trade_date)
        else:
            valid_a10.sort(key=lambda a: (a.trade_date, a.bin, a.ticker, a.metric_value))

        pairs          = []
        pairs_decile   = []
        pairs_decile20 = []
        buckets        = [[] for _ in range(10)]
        buckets_20_all = [[] for _ in range(20)]
        for a in valid_a10:
            pair = (a.metric_value, a.forward_return, a.trade_date, a.ticker)
            b10 = a.bin
            b20 = key_b20.get((a.ticker, a.trade_date), 0)
            pairs.append(pair)
            pairs_decile.append(b10)
            pairs_decile20.append(b20)
            buckets[b10 - 1].append(pair)
            if b20 > 0:
                buckets_20_all[b20 - 1].append(pair)

        # Match legacy bucket ordering so `decile_stats[i].returns` is
        # bit-equivalent. Legacy populated each bucket by iterating
        # tickers in by_ticker order (alphabetical from SQL) and then:
        #   in-sample:    pairs sorted by x within each ticker
        #   walk-forward: pairs in per-ticker chronological order
        # My loop above filled buckets in cross-ticker chronological
        # order; sort each bucket here to restore legacy iteration order.
        # Walk-forward populates buckets in per-ticker-chrono order (legacy);
        # in-sample and train-test (rank-based math against frozen training
        # history) both populate in (ticker, sorted-by-x) order.
        sort_key = (lambda p: (p[3], p[2])) if spec.kind == "walk_forward" else (lambda p: (p[3], p[0]))
        for b in range(10):
            buckets[b].sort(key=sort_key)
        for b in range(20):
            buckets_20_all[b].sort(key=sort_key)

        # Mode-aware "dropped" count for the response subtitle:
        #   in_sample    -> 0 (no method-specific gate)
        #   walk_forward -> rows that didn't clear the 252-day warmup
        #   train_test   -> rows whose ticker had < n_bins training samples
        from app.routers.row_compute import dropped_count_for_mode
        wf_dropped = dropped_count_for_mode(spec, a10)

        decile_stats_20 = _compute_bucket_stats(buckets_20_all)
        _tlog('ALL bucket_setup + decile_stats_20')
        n_tickers_used = sum(1 for ps in by_ticker.values() if len(ps) >= 10)
        spot_series = []
        all_spot_dates = []
        open_by_date = {}
        close_by_date = {}

        # Fetch complete per-ticker date lists for accurate exit_date/exit_close
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
        all_dates_list_by_tkr: dict = dict(_all_dates_by_tkr)  # tkr → [sorted dates]
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
        _tlog('ALL db_fetch2 + close_calendar')

    else:
        # Single-ticker mode
        single_ticker_cond = f" AND ticker = ${p}"
        params_single = params + [ticker]
        # spot_co = current open (entry price); spot_pc = prior close (not needed)
        spot_col = "spot_co"
        spot_select = f", {spot_col}"

        async with pool.acquire() as conn:
            rows = await conn.fetch(
                f"SELECT trade_date, {metric}, {outcome}{spot_select} FROM daily_features "
                f"WHERE {metric} IS NOT NULL AND {outcome} IS NOT NULL {single_ticker_cond}"
                f"{date_conditions} ORDER BY trade_date", *params_single)
        row_dicts = [dict(r) for r in rows]
        if row_dicts:
            _max_trade_date = str(max(r['trade_date'] for r in row_dicts))
        _tlog(f'single db_fetch1 ({len(row_dicts)} rows)')
        # Single-ticker rows don't carry a "ticker" column in the SQL
        # SELECT; inject it so the row_compute Assigners can group
        # consistently (they look up r["ticker"]).
        for r in row_dicts:
            r.setdefault("ticker", ticker)

        # Route bucketing through the row_compute layer. Both in-sample
        # and walk-forward branches flow through one path.
        from app.routers.row_compute import ASSIGNERS, _validate_assignments
        # `spec` was constructed at the top of the function.
        assigner = ASSIGNERS[spec.kind](spec)
        state10 = assigner.fit(row_dicts, metric, 10, False)
        a10 = assigner.assign(row_dicts, metric, 10, False, state10, outcome)
        _validate_assignments(a10, 10)
        state20 = assigner.fit(row_dicts, metric, 20, False)
        a20 = assigner.assign(row_dicts, metric, 20, False, state20, outcome)
        _validate_assignments(a20, 20)
        _tlog('single bin_assign (10-bin + 20-bin)')

        key_b20: dict = {(a.ticker, a.trade_date): a.bin
                         for a in a20 if a.bin is not None}
        valid_a10 = [a for a in a10
                     if a.bin is not None
                     and (a.ticker, a.trade_date) in key_b20]
        # Single-ticker mode: assignments already chronological per
        # input row order (SQL ORDER BY trade_date); sort again to be
        # explicit and tolerate any reordering the Assigner might do.
        valid_a10.sort(key=lambda a: a.trade_date)

        pairs          = []
        pairs_decile   = []
        pairs_decile20 = []
        buckets        = [[] for _ in range(10)]
        buckets_20     = [[] for _ in range(20)]
        for a in valid_a10:
            # Single-ticker `pairs` are 3-tuples (no ticker column —
            # legacy shape; downstream `_clean_pairs` consumers also
            # treat them as 3-tuples).
            pair = (a.metric_value, a.forward_return, a.trade_date)
            b10 = a.bin
            b20 = key_b20[(a.ticker, a.trade_date)]
            pairs.append(pair)
            pairs_decile.append(b10)
            pairs_decile20.append(b20)
            buckets[b10 - 1].append(pair)
            buckets_20[b20 - 1].append(pair)

        # Match legacy bucket ordering. Single-ticker in-sample legacy
        # used `_bucket_pairs(pairs, n)` which sorts by x then
        # distributes — bucket[i] is in sorted-by-x order. Walk-forward
        # legacy bucket[i] is already in chronological order (only one
        # ticker, so per-ticker-chrono == cross-ticker-chrono), which
        # matches my output; no sort needed.
        # Single-ticker: legacy in-sample sorted each bucket by x. Train-test
        # uses the same rank-based math as in-sample (just against a frozen
        # training history) so its bucket ordering is also sort-by-x. Only
        # walk-forward leaves buckets in chronological order.
        if spec.kind != "walk_forward":
            for b in range(10):
                buckets[b].sort(key=lambda p: p[0])
            for b in range(20):
                buckets_20[b].sort(key=lambda p: p[0])

        from app.routers.row_compute import dropped_count_for_mode
        wf_dropped = dropped_count_for_mode(spec, a10)

        decile_stats_20 = _compute_bucket_stats(buckets_20)
        _tlog('single bucket_setup + decile_stats_20')
        by_ticker = None  # not needed in single-ticker mode
        n_tickers_used = 1
        all_open_by_tkr_date = {}
        all_date_idx_by_tkr = {}
        all_dates_list_by_tkr = {}
        all_close_by_tkr = {}

        # spot_co = current open; used for entry_spot and chart overlay
        spot_series = []
        all_spot_dates: list = []
        open_by_date: dict = {}
        for r in row_dicts:
            date_s = str(r["trade_date"])
            sv = r.get(spot_col)
            if sv is not None:
                try:
                    fv = round(float(sv), 2)
                    open_by_date[date_s] = fv
                    spot_series.append({"date": date_s, "value": fv})
                except (ValueError, TypeError):
                    pass
        # Complete date list + spot_pc for exit close lookup (unfiltered by metric/outcome nulls)
        # close of day T = spot_pc of day T+1 in the trading day sequence
        async with pool.acquire() as conn:
            all_dates_rows = await conn.fetch(
                f"SELECT trade_date, spot_pc FROM daily_features "
                f"WHERE ticker = $1 AND {spot_col} IS NOT NULL "
                f"ORDER BY trade_date", ticker)
        all_spot_dates = [str(r["trade_date"]) for r in all_dates_rows]
        # close_by_date[d] = spot_pc of the next trading day = close price of day d
        _pc_map = {str(r["trade_date"]): r["spot_pc"] for r in all_dates_rows}
        close_by_date: dict = {}
        for i in range(len(all_spot_dates) - 1):
            next_pc = _pc_map.get(all_spot_dates[i + 1])
            if next_pc is not None:
                try:
                    close_by_date[all_spot_dates[i]] = round(float(next_pc), 2)
                except (ValueError, TypeError):
                    pass
        _tlog('single db_fetch2 + close_calendar')

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
        ticker_corrs = []
        for tkr_pairs in by_ticker.values():
            if len(tkr_pairs) < 20:
                continue
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
    yearly_consistency = []
    years_top_wins = 0

    if is_all and by_ticker:
        all_years = sorted(by_year.keys())
        for yr in all_years:
            yr_by_ticker: dict = defaultdict(list)
            for tkr, tkr_pairs in by_ticker.items():
                yr_ps = [p for p in tkr_pairs
                         if ((p[2].year if hasattr(p[2], 'year') else int(str(p[2])[:4])) == yr)]
                if yr_ps:
                    yr_by_ticker[tkr] = yr_ps
            yr_buckets = _bucket_pairs_per_ticker(yr_by_ticker, 10)
            top_ys = [p[1] for p in yr_buckets[9]]
            bot_ys = [p[1] for p in yr_buckets[0]]
            if len(top_ys) + len(bot_ys) < 20:
                continue
            t_avg = float(np.mean(top_ys)) if top_ys else 0.0
            b_avg = float(np.mean(bot_ys)) if bot_ys else 0.0
            top_beats = t_avg > b_avg
            if top_beats:
                years_top_wins += 1
            yearly_consistency.append({
                "year": yr, "top_avg": round(t_avg, 6), "bot_avg": round(b_avg, 6),
                "top_n": len(top_ys), "bot_n": len(bot_ys), "top_beats": top_beats,
            })
    else:
        for yr in sorted(by_year):
            yr_pairs = [(x, y, d) for x, y, d in pairs
                        if (d.year if hasattr(d, 'year') else int(str(d)[:4])) == yr]
            if len(yr_pairs) < 30:
                continue
            yr_buckets = _bucket_pairs(yr_pairs, 10)
            top_ys = [p[1] for p in yr_buckets[9]] if yr_buckets[9] else []
            bot_ys = [p[1] for p in yr_buckets[0]] if yr_buckets[0] else []
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
    if is_all and by_ticker:
        ticker_stable = []
        for tkr_pairs in by_ticker.values():
            if len(tkr_pairs) < 20:
                continue
            tkr_sorted = sorted(tkr_pairs, key=lambda p: p[2])
            mid_t = len(tkr_sorted) // 2
            h1 = _bucket_pairs(tkr_sorted[:mid_t], 10)
            h2 = _bucket_pairs(tkr_sorted[mid_t:], 10)
            h1_s = (np.mean([p[1] for p in h1[9]]) - np.mean([p[1] for p in h1[0]])) if h1[0] and h1[9] else 0
            h2_s = (np.mean([p[1] for p in h2[9]]) - np.mean([p[1] for p in h2[0]])) if h2[0] and h2[9] else 0
            ticker_stable.append((h1_s > 0 and h2_s > 0) or (h1_s < 0 and h2_s < 0))
        half_stable = (sum(ticker_stable) / len(ticker_stable) >= 0.5) if ticker_stable else False
    else:
        mid = n // 2
        h1 = _bucket_pairs(sorted(pairs, key=lambda p: p[2])[:mid], 10)
        h2 = _bucket_pairs(sorted(pairs, key=lambda p: p[2])[mid:], 10)
        h1_spread = (np.mean([p[1] for p in h1[9]]) - np.mean([p[1] for p in h1[0]])) if h1[0] and h1[9] else 0
        h2_spread = (np.mean([p[1] for p in h2[9]]) - np.mean([p[1] for p in h2[0]])) if h2[0] and h2[9] else 0
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
        "dropped_warmup_n": wf_dropped,
        "start_date":       str(pairs[0][2]) if pairs else None,

        # Diagnostic: handler elapsed time (ms) measured at the return statement.
        # compare with client `server+1stbyte` to isolate FastAPI serialization cost.
        # Remove when W1 ships and the measurement confirms serialization collapsed.
        "_handler_ms":      round((_time.perf_counter() - _t0) * 1000),

        # W2: max trade_date in the fetched rows — used as the staleness token
        # for the response cache.  Also surfaced to the client for display.
        "data_as_of":       _max_trade_date,
    }
    # W2: populate response cache (evict oldest if at capacity)
    _tlog(f'W2 write key="{_ac_key}" data_as_of="{_max_trade_date}"')
    if len(_ANALYZE_CACHE) >= _ANALYZE_CACHE_MAX:
        del _ANALYZE_CACHE[next(iter(_ANALYZE_CACHE))]
    _ANALYZE_CACHE[_ac_key] = _result
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

    Single-ticker mode uses absolute `np.percentile` bin edges (a
    deliberately different algorithm — the response carries the literal
    edge values as labels, e.g. "0.05–0.10", which the frontend renders).
    `walk_forward` is rejected for single-ticker because the absolute-edge
    semantic doesn't translate to a running-history view; if/when
    single-ticker /heatmap needs walk-forward, it gets its own design
    (rank-based bucketing with B1..BN labels, separate UI path).
    """
    if not pool:
        return {"error": "OI database not configured"}

    is_all = (ticker == "ALL")
    # Walk-forward and train-test heatmap modes both require ALL — single-
    # ticker /heatmap uses absolute np.percentile edges, not rank bins.
    if (walk_forward or cutoff_date) and not is_all:
        return {"error": "walk_forward / train_test not supported for "
                         "single-ticker /heatmap (uses absolute-percentile "
                         "edges, not rank bins)"}

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
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            f"SELECT ticker, trade_date, {metric_x}, {metric_y}, {outcome} FROM daily_features "
            f"WHERE {metric_x} IS NOT NULL AND {metric_y} IS NOT NULL AND {outcome} IS NOT NULL"
            f"{date_conditions} ORDER BY trade_date",
            *params)

    if is_all:
        # ALL mode: route x-axis and y-axis bucketing through the
        # row_compute Assigner. Two parallel calls — same per-ticker rank
        # math as the legacy inline loop, but the dispatch now picks
        # in-sample / walk-forward / train-test.
        from app.routers.row_compute import ASSIGNERS, make_spec
        spec = make_spec(walk_forward, cutoff_date)
        assigner = ASSIGNERS[spec.kind](spec)

        # Pre-filter to rows with all three fields valid + numeric. This
        # mirrors the legacy validity gate exactly so the per-ticker
        # row counts the Assigner sees match the legacy by_ticker.values()
        # iteration.
        rows_clean: list = []
        for r in rows:
            try:
                xv, yv, ov = float(r[metric_x]), float(r[metric_y]), float(r[outcome])
            except (TypeError, ValueError):
                continue
            if math.isnan(xv) or math.isnan(yv) or math.isnan(ov):
                continue
            rows_clean.append({
                "ticker":     r["ticker"],
                "trade_date": r.get("trade_date"),
                metric_x:     xv,
                metric_y:     yv,
                outcome:      ov,
            })

        # Two parallel Assigner calls — one per axis. Each row appears
        # in both result lists in the same input order.
        state_x = assigner.fit(rows_clean, metric_x, bins, True)
        a_x = assigner.assign(rows_clean, metric_x, bins, True, state_x, outcome)
        state_y = assigner.fit(rows_clean, metric_y, bins, True)
        a_y = assigner.assign(rows_clean, metric_y, bins, True, state_y, outcome)

        # cell_rets[y_bin][x_bin] — outer index is y so the returned grid
        # matches the frontend's grid[iy][ix] convention (rows=y, cols=x).
        cell_rets: list = [[[] for _ in range(bins)] for _ in range(bins)]
        n_tickers_with_bins: set = set()
        total_n = 0
        for ax, ay, row in zip(a_x, a_y, rows_clean):
            # Rows from tickers with < bins observations have bin=None
            # under both Assigners (`_bucket_pairs_per_ticker` excludes
            # tickers below the threshold). Skip them — matches legacy
            # `if len(items) < bins: continue`.
            # In train-test mode, pre-cutoff rows carry a real bin (frozen
            # training thresholds applied to training rows) but must be
            # excluded from the test-period aggregation — same as /analyze.
            if ax.bin is None or ay.bin is None:
                continue
            if ax.dropped_reason == "pre_cutoff" or ay.dropped_reason == "pre_cutoff":
                continue
            # 1-indexed → 0-indexed for Python list indexing.
            cell_rets[ay.bin - 1][ax.bin - 1].append(row[outcome])
            total_n += 1
            n_tickers_with_bins.add(ax.ticker)
        n_tickers_used = len(n_tickers_with_bins)

        if total_n < 50:
            return {"error": f"Insufficient data after per-ticker filter: {total_n} rows"}

        grid = []
        for iy in range(bins):
            row = []
            for ix in range(bins):
                rets = cell_rets[iy][ix]
                if len(rets) >= 5:
                    a = np.array(rets)
                    row.append({
                        "avg_ret":  round(float(a.mean()), 6),
                        "win_rate": round(float((a > 0).mean()), 4),
                        "n":        int(len(rets)),
                    })
                else:
                    row.append(None)
            grid.append(row)
        # Bin labels — absolute ranges don't make sense in ALL mode since
        # each ticker has its own boundaries. Use B1..BN.
        x_labels = [f"B{i+1}" for i in range(bins)]
        y_labels = [f"B{j+1}" for j in range(bins)]
        out: dict = {
            "metric_x":  metric_x, "metric_y": metric_y, "outcome": outcome,
            "bins":      bins, "n": total_n,
            "x_labels":  x_labels, "y_labels": y_labels,
            "grid":      grid,
            "per_ticker": True,
            "n_tickers": n_tickers_used,
            "mode":      spec.kind,
        }
        from app.routers.row_compute import dropped_count_for_mode
        if spec.kind == "walk_forward":
            out["warmup"]           = spec.warmup
            out["dropped_warmup_n"] = dropped_count_for_mode(spec, a_x)
        elif spec.kind == "train_test":
            out["cutoff_date"]      = spec.cutoff.isoformat()
            out["dropped_warmup_n"] = dropped_count_for_mode(spec, a_x)
        return out

    # Single-ticker mode — original absolute-percentile binning.
    valid = []
    for r in rows:
        try:
            xv, yv, ov = float(r[metric_x]), float(r[metric_y]), float(r[outcome])
            if not (math.isnan(xv) or math.isnan(yv) or math.isnan(ov)):
                valid.append((xv, yv, ov))
        except (ValueError, TypeError):
            continue

    if len(valid) < 50:
        return {"error": f"Insufficient data: {len(valid)} rows"}

    xs = np.array([v[0] for v in valid])
    ys = np.array([v[1] for v in valid])
    os_ = np.array([v[2] for v in valid])

    x_edges = np.percentile(xs, np.linspace(0, 100, bins + 1))
    y_edges = np.percentile(ys, np.linspace(0, 100, bins + 1))
    x_edges[-1] += 1e-9
    y_edges[-1] += 1e-9

    # grid[iy][ix] so rows match the y-axis (matches frontend convention).
    grid = []
    for iy in range(bins):
        row = []
        for ix in range(bins):
            mask = ((xs >= x_edges[ix]) & (xs < x_edges[ix+1]) &
                    (ys >= y_edges[iy]) & (ys < y_edges[iy+1]))
            crets = os_[mask]
            if len(crets) >= 5:
                row.append({
                    "avg_ret":  round(float(crets.mean()), 6),
                    "win_rate": round(float((crets > 0).mean()), 4),
                    "n":        int(len(crets)),
                })
            else:
                row.append(None)
        grid.append(row)

    x_labels = [f"{x_edges[i]:.2f}–{x_edges[i+1]:.2f}" for i in range(bins)]
    y_labels = [f"{y_edges[j]:.2f}–{y_edges[j+1]:.2f}" for j in range(bins)]

    return {
        "metric_x":  metric_x, "metric_y": metric_y, "outcome": outcome,
        "bins":      bins, "n": len(valid),
        "x_labels":  x_labels, "y_labels": y_labels,
        "grid":      grid,
        "per_ticker": False,
    }


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
    are first-class modes. This endpoint backs the heatmap's side bin
    charts; pre-Step-5.5-continuation it was on inline math with no
    walk-forward support, producing in-sample hindsight-monotone shapes
    that didn't match the page's other binning panes.
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
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            f"SELECT ticker, trade_date, {metric}, {outcome} FROM daily_features "
            f"WHERE {metric} IS NOT NULL AND {outcome} IS NOT NULL {date_conditions} "
            f"ORDER BY trade_date", *params)

    # Spec-dispatched binning. Both ALL and single-ticker flow through
    # the Assigner with the same shape — the legacy fork on `is_all`
    # collapses to a single path inside the Assigner.
    from app.routers.row_compute import ASSIGNERS, make_spec
    spec = make_spec(walk_forward, cutoff_date)
    assigner = ASSIGNERS[spec.kind](spec)

    rows_for_assigner = [dict(r) for r in rows]
    if not is_all:
        # Single-ticker rows don't carry a "ticker" column from the SQL
        # SELECT — inject it so the Assigner can group consistently.
        for r in rows_for_assigner:
            r.setdefault("ticker", ticker)

    state = assigner.fit(rows_for_assigner, metric, bins, is_all)
    assignments = assigner.assign(rows_for_assigner, metric, bins, is_all, state, outcome)

    # Build buckets_data: list of (xf, yf) tuples per bin (1-indexed →
    # 0-indexed list position). Drops bin=None rows (warmup, missing_value,
    # insufficient_history) AND train-test pre_cutoff rows (training-window
    # rows that have bin set but are excluded from test-period aggregation).
    # For in_sample/walk_forward the dropped_reason check is a no-op.
    buckets_data: list = [[] for _ in range(bins)]
    total_n = 0
    for a in assignments:
        if a.bin is None or a.dropped_reason is not None:
            continue
        buckets_data[a.bin - 1].append((a.metric_value, a.forward_return))
        total_n += 1

    if total_n < 20:
        if is_all:
            return {"error": f"Insufficient data after per-ticker filter: {total_n} rows"}
        return {"error": f"Insufficient data: {total_n} rows"}

    result = []
    for i, bucket in enumerate(buckets_data):
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
            # In ALL mode these are the cross-ticker min/max of observed
            # values that landed in this bin — informational only, since
            # each ticker's bin boundary is independent.
            "min_val":  round(float(min(xs)), 6),
            "max_val":  round(float(max(xs)), 6),
        })
    out: dict = {
        "metric":     metric,
        "outcome":    outcome,
        "bins":       bins,
        "n":          total_n,
        "buckets":    result,
        "per_ticker": is_all,
        "mode":       spec.kind,
    }
    from app.routers.row_compute import dropped_count_for_mode
    if spec.kind == "walk_forward":
        out["warmup"]           = spec.warmup
        out["dropped_warmup_n"] = dropped_count_for_mode(spec, assignments)
    elif spec.kind == "train_test":
        out["cutoff_date"]      = spec.cutoff.isoformat()
        out["dropped_warmup_n"] = dropped_count_for_mode(spec, assignments)
    return out


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
) -> list:
    """Score each secondary feature by weighted_spread × breadth.

    Binning: uses walk-forward assign_secondary_buckets, identical to the
    drilled secondary_detail chart — leaderboard scores reconcile with
    the per-bin tooltip spreads.

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

    from app.routers.row_compute import assign_secondary_buckets, _sort_chrono

    # ── Pre-sort rows chronologically once (reused by every feature). ─────────
    # assign_secondary_buckets (WF mode) calls _sort_chrono internally; passing
    # rows_presorted=True skips that redundant O(N log N) sort per feature.
    rows_sorted = _sort_chrono(rows)

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

        # WF-consistent bucket assignment — same path as drilled chart.
        # rows_presorted=True skips the internal _sort_chrono call.
        buckets_raw = assign_secondary_buckets(
            spec, rows_sorted, feat, n_bins, outcome_col, is_all,
            rows_presorted=True,
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
        if is_all and by_tkr:
            sign_global  = 1 if weighted_spread >= 0 else -1
            n_qualifying = 0
            n_agreeing   = 0
            for tkr_vals in by_tkr.values():
                if len(tkr_vals) < 10:
                    continue
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
            loop.run_in_executor(
                None, _run_sec_score,
                scan_key, rows, cached["outcome"], feature_cols,
                is_all, n_bins, cached["primary_metric"], req.selected_primary_bins, req.filtered_dates,
                db_write_params,
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
            loop.run_in_executor(
                None, _run_sec_score,
                scan_key, rows, outcome_col, feature_cols,
                is_all, n_bins, primary_metric, req.selected_primary_bins, req.filtered_dates,
                db_write_params,
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
            pass
    return {"ok": True}


@router.post("/secondary-detail")
async def secondary_detail(req: SecDetailReq):
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
        WalkForwardSpec, filter_by_assignments, assign_secondary_buckets,
        mode_envelope,
    )
    spec = WalkForwardSpec()
    filtered, dropped, _universe = filter_by_assignments(
        all_rows, spec, primary_metric or "",
        req.selected_primary_bins, is_all, req.filtered_dates,
    )

    if len(filtered) < n_bins * 2:
        return {"error": "insufficient_data"}

    # ── Secondary binning ───────────────────────────────────────────────────
    # Spec-dispatched. The in-sample branch may return None when there
    # are too few rows with valid (metric_b, outcome) pairs after the
    # per-ticker n_bins gate — match the legacy "insufficient_data"
    # error in that case. The walk-forward branch never returns None
    # (matches legacy, which always built buckets without a second check).
    buckets = assign_secondary_buckets(
        spec, filtered, req.metric_b, n_bins, outcome_col, is_all,
        all_rows=(all_rows if spec.kind == "train_test" else None),
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
async def secondary_corr_bins(req: CorrBinsReq):
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

    filtered, dropped, universe = filter_by_assignments(
        all_rows, spec, primary_metric,
        req.selected_primary_bins, is_all, req.filtered_dates,
    )
    if not filtered:
        return {"error": "no_data",
                **mode_envelope(spec, dropped=dropped, universe=universe)}

    results = []
    for feat in feat_cols:
        r = assign_secondary_bin_stats(
            spec, filtered, feat, n_bins, outcome_col, is_all,
            all_rows=(all_rows if spec.kind == "train_test" else None),
        )
        if r:
            results.append(r)

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
async def secondary_correlation(req: CorrReq):
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

    filtered, dropped, universe = filter_by_assignments(
        all_rows, spec, primary_metric,
        req.selected_primary_bins, is_all, req.filtered_dates,
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

    vectors, metric_names, n_each = [], [], []
    for sel in req.selections:
        metric = sel.get("metric", "")
        bins   = set(sel.get("bins", []))
        if not metric or not bins:
            continue
        vec = secondary_membership(
            spec, ordered, metric, bins, n_bins, is_all,
            all_rows=(all_rows if spec.kind == "train_test" else None),
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
    # different cutoffs produce different bin assignments.
    if cutoff_date:
        mode_tag = f"tt:{cutoff_date}"
    elif walk_forward:
        mode_tag = "wf"
    else:
        mode_tag = "is"
    cache_key = f"{ticker}|{outcome}|{n_bins}|{date_from or ''}|{date_to or ''}|{mode_tag}"
    if cache_key in _GLOBAL_BINS_CACHE:
        return _GLOBAL_BINS_CACHE[cache_key]

    # Check persistent DB cache before running the expensive computation.
    # asyncpg returns JSONB as a string by default (no codec is registered
    # in app/db.py), so `payload` needs an explicit json.loads — the old
    # code did `dict(db_row["payload"])` which silently failed on the
    # string and fell through to recompute every time the in-memory cache
    # was cold. Pre-fix, the DB cache only ever served data that had
    # already been computed in the current process lifetime.
    await _ensure_bins_table(pool)
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

    # Build date filter
    where = [f"{outcome} IS NOT NULL"]
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
            col_rows = await conn.fetch(
                """SELECT column_name FROM information_schema.columns
                   WHERE table_name = 'daily_features' AND table_schema = 'public'
                   AND data_type IN ('double precision','numeric','real','integer','bigint','smallint')
                   AND column_name NOT IN ('id','ticker','trade_date','created_at','updated_at')
                   ORDER BY ordinal_position""")
        all_num_cols = [r["column_name"] for r in col_rows]
        outcome_cols_all = [c for c in all_num_cols if "ret_" in c and "fwd" in c]
        feature_cols = [c for c in all_num_cols
                        if c not in outcome_cols_all
                        and not c.startswith("spot") and not c.endswith("_pc")]

        select_cols = ", ".join(["ticker", "trade_date", outcome] + feature_cols)
        async with pool.acquire() as conn:
            # Override the pool's 30 s command_timeout — pulling every feature
            # column for every row across 80+ tickers can easily exceed it.
            db_rows = await conn.fetch(
                f"SELECT {select_cols} FROM daily_features "
                f"WHERE {' AND '.join(where)} ORDER BY ticker, trade_date",
                *params, timeout=240)
        rows = [dict(r) for r in db_rows]
        if not rows:
            out = {"outcome": outcome, "ticker": ticker, "n_bins": n_bins,
                   "metrics": [], "total_rows": 0,
                   "metrics_attempted": 0}
            _GLOBAL_BINS_CACHE[cache_key] = out
            return out

        is_all = (ticker == "ALL")

        # Route through the row_compute layer. Both branches delegate to
        # the same numpy-vectorized helpers as before; the if/else here
        # is now a one-line spec selection instead of duplicated call
        # sites with diverging return shapes.
        from app.routers.row_compute import ASSIGNERS, make_spec
        spec = make_spec(walk_forward, cutoff_date)
        assigner = ASSIGNERS[spec.kind](spec)
        metrics_out, dropped_n, wf_start = assigner.assign_batch(
            rows, feature_cols, outcome, n_bins, is_all,
        )

        # Sort by lift (max - min avg ret) so most-interesting metrics appear first.
        def _lift(m):
            bs = m.get("bins") or []
            return (max(bs) - min(bs)) if bs else 0
        metrics_out.sort(key=_lift, reverse=True)

        out: dict = {
            "outcome":           outcome,
            "ticker":            ticker,
            "n_bins":            n_bins,
            "total_rows":        len(rows),
            "metrics_attempted": len(feature_cols),
            "metrics":           metrics_out,
            "mode":              spec.kind,
        }
        if spec.kind == "walk_forward":
            out["warmup"]           = spec.warmup
            out["dropped_warmup_n"] = dropped_n
            out["start_date"]       = wf_start
        elif spec.kind == "train_test":
            out["cutoff_date"]      = spec.cutoff.isoformat()
            out["dropped_warmup_n"] = dropped_n
            out["start_date"]       = wf_start

        # Persist to DB so next server restart loads instantly.
        try:
            async with pool.acquire() as conn:
                await conn.execute(
                    "INSERT INTO global_bins_cache "
                    "    (cache_key, outcome, ticker, n_bins, mode, payload) "
                    "VALUES ($1,$2,$3,$4,$5,$6::jsonb) "
                    "ON CONFLICT (cache_key) DO UPDATE "
                    "    SET payload=$6::jsonb, cached_at=NOW()",
                    cache_key, outcome, ticker, n_bins,
                    spec.kind,
                    json.dumps(out),
                )
                row_ca = await conn.fetchrow(
                    "SELECT cached_at FROM global_bins_cache WHERE cache_key = $1",
                    cache_key)
            out["cached_at"] = row_ca["cached_at"].isoformat() if row_ca else None
        except Exception:
            out["cached_at"] = None

        _GLOBAL_BINS_CACHE[cache_key] = out
        return out
    except Exception as exc:
        # Surface failures to the frontend instead of returning a generic 500
        # that the UI swallows into "no data". Most likely cause is a query
        # timeout when daily_features has grown a lot.
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
    """Drop the in-memory and DB cache so a fresh computation runs next request."""
    _GLOBAL_BINS_CACHE.clear()
    if pool:
        try:
            await _ensure_bins_table(pool)
            async with pool.acquire() as conn:
                await conn.execute("DELETE FROM global_bins_cache")
        except Exception:
            pass
    return {"ok": True}


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
# 12-outcome bundle cache. Each entry: one (ticker, metric, mode, cutoff)
# tuple's full bundle covering all forward-return outcomes. Only ALL-mode
# entries persist; single-ticker bundles compute inline (~2-5s) and never
# reach this table.
#
# LRU eviction by aggregate size: on each write, while
# pg_total_relation_size('analyze_cache') > 5 GB, drop the row with the
# oldest `last_accessed`. The 5 GB cap protects the disk; in practice
# ~25-30 ALL-mode entries fit before eviction kicks in.

_ANALYZE_BUNDLE_TABLE_DDL = """\
CREATE TABLE IF NOT EXISTS analyze_cache (
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
CREATE INDEX IF NOT EXISTS analyze_cache_last_accessed
    ON analyze_cache (last_accessed);
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
_ANALYZE_BUNDLE_SCHEMA_VERSION = 2  # bumped from 1: dynamic outcomes (v1 caches re-compute)


async def _ensure_analyze_bundle_table(pool) -> None:
    global _analyze_bundle_table_ensured
    if _analyze_bundle_table_ensured:
        return
    async with pool.acquire() as conn:
        await conn.execute(_ANALYZE_BUNDLE_TABLE_DDL)
    _analyze_bundle_table_ensured = True


def _analyze_bundle_cache_key(ticker: str, metric: str, mode: str,
                              cutoff_date: Optional[str]) -> str:
    """Stable cache key for (ticker, metric, mode, cutoff). Salted with the
    schema version so a future bundle-shape change automatically invalidates
    every stale entry on next read."""
    cutoff_s = cutoff_date or "null"
    return f"ab:v{_ANALYZE_BUNDLE_SCHEMA_VERSION}:{ticker}:{metric}:{mode}:{cutoff_s}"


async def _evict_analyze_cache_lru(pool) -> int:
    """Drop oldest-accessed rows until pg_total_relation_size('analyze_cache')
    falls back under the cap. Returns the count of rows evicted. Called from
    the write path after each successful insert/update.
    """
    evicted = 0
    async with pool.acquire() as conn:
        while True:
            size_bytes = await conn.fetchval(
                "SELECT pg_total_relation_size('analyze_cache')")
            if size_bytes is None or size_bytes <= _ANALYZE_BUNDLE_CACHE_MAX_BYTES:
                break
            oldest_key = await conn.fetchval(
                "SELECT cache_key FROM analyze_cache "
                "ORDER BY last_accessed ASC LIMIT 1")
            if oldest_key is None:
                break   # table empty but pg_total_relation_size still > cap (bloat)
            await conn.execute(
                "DELETE FROM analyze_cache WHERE cache_key = $1", oldest_key)
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


def _compute_analyze_bundle_sync(
    rows: list[dict],
    metric: str,
    ticker: str,
    mode: str,
    cutoff_date: Optional[str],
    outcomes: list[str],
    n_bins: int = 10,
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
    from app.routers.row_compute import ASSIGNERS, make_spec
    from app.routers.ic_compute import (
        rolling_ic_single_ticker, rolling_ic_cross_sectional,
        classified_rolling_ic, noise_floor_epsilon, _horizon_from_outcome,
    )

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

    # ── 2. Run Assigner once (anchor selected above) ──────────────────────
    spec = make_spec(
        walk_forward=(mode == "walk_forward"),
        cutoff_date=cutoff_date if mode == "train_test" else None,
    )
    assigner = ASSIGNERS[spec.kind](spec)
    state = assigner.fit(rows, metric, n_bins, is_all)
    assignments = assigner.assign(rows, metric, n_bins, is_all, state, anchor_outcome)

    # ── 3. Build trade_meta from valid assignments ────────────────────────
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
        entry_spot = None
        if r is not None and r.get("spot_co") is not None:
            try:
                entry_spot = round(float(r["spot_co"]), 2)
            except (TypeError, ValueError):
                pass
        tid = len(trade_meta)
        trade_id_by_key[key] = tid
        trade_meta.append({
            "trade_id":   tid,
            "ticker":     tkr,
            "trade_date": d,
            "metric_val": round(float(a.metric_value), 6),
            "entry_spot": entry_spot,
            "bin":        a.bin,
        })

    # ── 4. Per-outcome: returns + per-bin stats + rolling IC ──────────────
    per_outcome_returns: dict = {}
    per_bin:             dict = {}
    rolling_ic:          dict = {}

    for outcome in outcomes:
        horizon = _horizon_from_outcome(outcome)

        out_records: list = []
        bin_buckets: list = [[] for _ in range(n_bins)]
        ic_rows:     list = []   # input for rolling_ic_* — needs trade_date + ticker keys

        for meta in trade_meta:
            tkr = meta["ticker"]
            d   = meta["trade_date"]
            r   = row_by_tkr_date.get((tkr, d))
            if r is None:
                continue
            y = r.get(outcome)
            if y is None:
                continue
            try:
                yf = float(y)
            except (TypeError, ValueError):
                continue
            if math.isnan(yf):
                continue

            # exit_date = trade_date + (horizon - 1) trading days for the
            # same ticker; exit_spot = close (= spot_pc of the day after
            # exit_date) for the same ticker.
            exit_date = None
            exit_spot = None
            entry_idx = by_tkr_idx.get(tkr, {}).get(d)
            if entry_idx is not None:
                exit_idx  = entry_idx + max(horizon - 1, 0)
                tkr_dates = by_tkr_dates.get(tkr, [])
                if 0 <= exit_idx < len(tkr_dates):
                    exit_date = tkr_dates[exit_idx]
                    es = close_by_tkr_date.get((tkr, exit_date))
                    if es is not None:
                        exit_spot = round(es, 2)

            out_records.append({
                "trade_id":  meta["trade_id"],
                "ret_pct":   round(yf, 6),
                "exit_date": exit_date,
                "exit_spot": exit_spot,
            })
            bin_buckets[meta["bin"] - 1].append(yf)
            ic_rows.append({"trade_date": d, "ticker": tkr,
                            "__m": meta["metric_val"], "__y": yf})

        per_outcome_returns[outcome] = out_records

        # Per-bin aggregates
        bin_stats: list = []
        for b_idx in range(n_bins):
            vals = bin_buckets[b_idx]
            if not vals:
                bin_stats.append({"bin": b_idx + 1, "n": 0,
                                  "avg_ret": None, "median": None,
                                  "std": None, "win_rate": None})
                continue
            arr = np.array(vals, dtype=np.float64)
            bin_stats.append({
                "bin":      b_idx + 1,
                "n":        int(len(arr)),
                "avg_ret":  round(float(arr.mean()), 6),
                "median":   round(float(np.median(arr)), 6),
                "std":      round(float(arr.std()), 6),
                "win_rate": round(float((arr > 0).mean()), 4),
            })
        per_bin[outcome] = bin_stats

        # Rolling IC: same primitives /analyze uses for its rolling_ic field
        if is_all:
            ic_series = rolling_ic_cross_sectional(ic_rows, "__m", "__y", window=252)
            median_k = int(np.median([p.n for p in ic_series])) if ic_series else 0
            epsilon = noise_floor_epsilon(
                "cross_sectional", window=252, horizon=horizon, k_tickers=median_k)
        else:
            ic_series = rolling_ic_single_ticker(ic_rows, "__m", "__y", window=252)
            epsilon = noise_floor_epsilon(
                "single_ticker", window=252, horizon=horizon)

        if mode == "train_test" and cutoff_date:
            pre = [p.ic for p in ic_series if str(p.date) < cutoff_date]
            reference_ic = float(np.mean(pre)) if pre else 0.0
        elif ic_series:
            reference_ic = float(np.mean([p.ic for p in ic_series]))
        else:
            reference_ic = 0.0

        classified = classified_rolling_ic(ic_series, reference_ic, epsilon)
        rolling_ic[outcome] = [
            {"date":       str(p.date),
             "ic":         p.ic,
             "n":          p.n,
             "sign_class": p.sign_class}
            for p in classified
        ]

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
    """Background bundle compute for ALL mode. Writes result to
    analyze_cache, triggers LRU eviction. Failures surface via
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
        bundle = await asyncio.to_thread(
            _compute_analyze_bundle_sync,
            rows, metric, ticker, mode, cutoff_date, outcomes,
        )
        payload_json = json.dumps(bundle)
        from datetime import date as _date_cls
        cutoff_obj = _date_cls.fromisoformat(cutoff_date) if cutoff_date else None
        async with pool.acquire() as conn:
            await conn.execute(
                """INSERT INTO analyze_cache
                   (cache_key, ticker, metric, mode, cutoff_date,
                    payload, payload_bytes, cached_at, last_accessed)
                   VALUES ($1, $2, $3, $4, $5, $6::jsonb, $7, NOW(), NOW())
                   ON CONFLICT (cache_key) DO UPDATE
                   SET payload       = EXCLUDED.payload,
                       payload_bytes = EXCLUDED.payload_bytes,
                       cached_at     = NOW(),
                       last_accessed = NOW()""",
                cache_key, ticker, metric, mode, cutoff_obj,
                payload_json, len(payload_json),
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
    """Return the 12-outcome bundle for (ticker, metric, mode, cutoff).

    Three response shapes:

      {status: "ready",        bundle: {...}, cached_at?: "..."}
      {status: "computing",    cache_key: "..."}                     # ALL only
      {status: "not_computed", previous_error?: "..."}               # ALL only

    Behavior:
      - Single-ticker (`ticker != "ALL"`): computes inline (~2-5s) and
        returns `ready` with the bundle. Never writes to analyze_cache.
      - ALL: cache-only. Cache hit → `ready` + bundle. Background job
        running → `computing`. Neither → `not_computed`. The frontend
        triggers POST /analyze-bundle/refresh to start a compute.
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
        bundle = await asyncio.to_thread(
            _compute_analyze_bundle_sync,
            rows, metric, ticker, mode, cutoff_date, outcomes,
        )
        return {"status": "ready", "bundle": bundle}

    # ALL mode: cache-only
    await _ensure_analyze_bundle_table(pool)
    cache_key = _analyze_bundle_cache_key(ticker, metric, mode, cutoff_date)

    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT payload, cached_at FROM analyze_cache WHERE cache_key = $1",
            cache_key,
        )
    if row is not None:
        # Touch last_accessed for LRU. Fire-and-forget — no need to await
        # before returning the bundle.
        try:
            async with pool.acquire() as conn:
                await conn.execute(
                    "UPDATE analyze_cache SET last_accessed = NOW() "
                    "WHERE cache_key = $1", cache_key)
        except Exception:
            pass
        payload = row["payload"]
        if isinstance(payload, str):
            payload = json.loads(payload)
        return {"status": "ready", "bundle": payload,
                "cached_at": str(row["cached_at"])}

    if cache_key in _analyze_bundle_running:
        return {"status": "computing", "cache_key": cache_key}

    failed = _analyze_bundle_status.pop(cache_key, None)
    if failed and failed.get("status") == "failed":
        return {"status": "not_computed",
                "previous_error": failed.get("error")}
    return {"status": "not_computed"}


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
            pass
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
    if cache_key in _THRESHOLD_DRIFT_CACHE:
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
        _THRESHOLD_DRIFT_CACHE[cache_key] = out
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
    return {"ok": True}
