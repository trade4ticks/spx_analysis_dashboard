"""OI Analysis workbench — interactive decile analytics for a single ticker/metric/outcome."""
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
_SEC_CACHE: dict = {}  # cache_key -> {rows, features, outcome}

router = APIRouter(tags=["oi_analysis"])


def _clean_pairs(rows, x_col, y_col):
    """Extract (x, y, date) tuples, filtering None and NaN."""
    out = []
    for r in rows:
        xv, yv = r.get(x_col), r.get(y_col)
        if xv is None or yv is None:
            continue
        try:
            xf, yf = float(xv), float(yv)
        except (ValueError, TypeError):
            continue
        if math.isnan(xf) or math.isnan(yf):
            continue
        out.append((xf, yf, r.get("trade_date")))
    return out


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


def _walk_forward_bucket_pairs(pairs, n_bins_list, warmup):
    """Walk-forward equivalent of `_bucket_pairs` (single-ticker) for one or
    more bin counts simultaneously.

    Each pair is (x, y, date[, ticker]). Pairs MUST be in chronological order
    at the call site (we don't re-sort here — for ALL mode the caller iterates
    per-ticker chronologically).

    For each pair, the running per-history rank is computed via bisect_left
    against a sorted list of prior x values. After the per-history count
    reaches `max(warmup, max(n_bins_list))`, the bin is emitted as
    `min(int(rank / n_after * n_bins) + 1, n_bins)`. Pairs in warmup are
    skipped.

    Returns:
      assignments: list of (pair, {n_bins: bin_int}) for pairs that cleared
                   warmup, in the chronological order they were provided.
      dropped_warmup_n: count of pairs that didn't clear warmup.
    """
    import bisect
    max_bins = max(n_bins_list)
    warm = max(int(warmup), int(max_bins))
    assignments: list = []
    dropped = 0
    sorted_vals: list = []
    for pair in pairs:
        val = pair[0]
        rank = bisect.bisect_left(sorted_vals, val)
        bisect.insort(sorted_vals, val)
        n_after = len(sorted_vals)
        if n_after < warm:
            dropped += 1
            continue
        bins_for_pair = {nb: min(int(rank / n_after * nb) + 1, nb) for nb in n_bins_list}
        assignments.append((pair, bins_for_pair))
    return assignments, dropped


def _walk_forward_bucket_per_ticker(by_ticker, n_bins_list,
                                    warmup=None) -> tuple:
    """Walk-forward equivalent of `_bucket_pairs_per_ticker`.

    For each ticker, walks the pairs chronologically (sorted by trade_date)
    and emits walk-forward bin assignments at every requested granularity in
    a single bisect_left pass. Pairs whose per-ticker history hasn't reached
    `max(warmup, max(n_bins_list))` are dropped.

    Returns:
      buckets_per_granularity: dict {n_bins: list of n_bins buckets, each a
                                     list of pair tuples assigned to that bin}
      assignments_chrono: list of (pair, {n_bins: bin_int}) sorted
                          chronologically across all tickers — convenient for
                          building `pairs_with_d`-style structures downstream.
      dropped_warmup_n: total count of pairs dropped to warmup (across all
                        tickers).
    """
    if warmup is None:
        warmup = _DEFAULT_WALKFWD_WARMUP
    buckets_by_n: dict = {nb: [[] for _ in range(nb)] for nb in n_bins_list}
    all_assignments: list = []
    dropped = 0
    for tkr_pairs in by_ticker.values():
        chrono = sorted(tkr_pairs, key=lambda p: p[2])
        a_t, d_t = _walk_forward_bucket_pairs(chrono, n_bins_list, warmup)
        dropped += d_t
        all_assignments.extend(a_t)
        for pair, bins_for_pair in a_t:
            for nb, b in bins_for_pair.items():
                buckets_by_n[nb][b - 1].append(pair)
    # Sort chronologically across tickers (pair[2] is the date)
    all_assignments.sort(key=lambda ab: ab[0][2])
    return buckets_by_n, all_assignments, dropped


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
    pool=Depends(get_oi_pool),
):
    """Full analysis payload for one ticker/metric/outcome combo."""
    if not pool:
        return {"error": "OI database not configured"}

    is_all = (ticker == "ALL")

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

        if walk_forward:
            # Walk-forward bin assignment per ticker. Both 10-bin and 20-bin
            # bucketing computed in one bisect pass per ticker. Warmup rows
            # (per-ticker history < 252) are dropped.
            buckets_by_n, wf_assignments, wf_dropped = _walk_forward_bucket_per_ticker(
                by_ticker, [10, 20]
            )
            buckets        = buckets_by_n[10]
            buckets_20_all = buckets_by_n[20]
            pairs          = [a[0]        for a in wf_assignments]
            pairs_decile   = [a[1][10]    for a in wf_assignments]
            pairs_decile20 = [a[1][20]    for a in wf_assignments]
        else:
            buckets = _bucket_pairs_per_ticker(by_ticker, 10)
            buckets_20_all = _bucket_pairs_per_ticker(by_ticker, 20)
            wf_dropped = 0

            # pairs + parallel decile lists from pre-assigned buckets
            pairs_with_d = sorted(
                [(p, i + 1) for i, bucket in enumerate(buckets) for p in bucket],
                key=lambda x: x[0][2]  # chronological
            )
            pairs = [pd[0] for pd in pairs_with_d]
            pairs_decile = [pd[1] for pd in pairs_with_d]

            # 20-bin per-ticker normalization (same pattern, 20 buckets)
            pair_to_dec20: dict = {}
            for bin_idx, bucket in enumerate(buckets_20_all):
                for p in bucket:
                    pair_to_dec20[id(p)] = bin_idx + 1
            pairs_decile20 = [pair_to_dec20.get(id(pd[0]), 0) for pd in pairs_with_d]

        decile_stats_20 = _compute_bucket_stats(buckets_20_all)
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
        pairs = _clean_pairs(row_dicts, metric, outcome)

        if walk_forward:
            # pairs is already chronological (SELECT ORDER BY trade_date).
            wf_assignments, wf_dropped = _walk_forward_bucket_pairs(
                pairs, [10, 20], _DEFAULT_WALKFWD_WARMUP
            )
            buckets    = [[] for _ in range(10)]
            buckets_20 = [[] for _ in range(20)]
            for pair, bins_for_pair in wf_assignments:
                buckets[bins_for_pair[10] - 1].append(pair)
                buckets_20[bins_for_pair[20] - 1].append(pair)
            pairs          = [a[0]      for a in wf_assignments]
            pairs_decile   = [a[1][10]  for a in wf_assignments]
            pairs_decile20 = [a[1][20]  for a in wf_assignments]
        else:
            buckets = _bucket_pairs(pairs, 10)
            buckets_20 = _bucket_pairs(pairs, 20)
            wf_dropped = 0

            # Build decile assignments from flat rank order
            sorted_by_x = sorted(range(len(pairs)), key=lambda i: pairs[i][0])
            dm: dict = {}
            dm20: dict = {}
            for rank, idx in enumerate(sorted_by_x):
                dm[idx]   = min(int(rank / len(pairs) * 10) + 1, 10)
                dm20[idx] = min(int(rank / len(pairs) * 20) + 1, 20)
            pairs_decile   = [dm[i]   for i in range(len(pairs))]
            pairs_decile20 = [dm20[i] for i in range(len(pairs))]

        decile_stats_20 = _compute_bucket_stats(buckets_20)
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
        for date, ret in trades:
            cum += ret
            peak = max(peak, cum)
            max_dd = min(max_dd, cum - peak)
            if ret > 0:
                wins += 1
            points.append({"date": str(date), "value": round(cum, 6)})
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

    # ── Rolling correlation (252-day window; skipped in ALL mode) ─────────
    rolling_corr = []
    if not is_all:
        sorted_by_date = sorted(pairs, key=lambda p: p[2])
        rolling_window = 252
        if len(sorted_by_date) > rolling_window:
            for end in range(rolling_window, len(sorted_by_date)):
                window = sorted_by_date[end - rolling_window:end]
                wx = np.array([p[0] for p in window])
                wy = np.array([p[1] for p in window])
                if wx.std() > 0 and wy.std() > 0:
                    rc, _ = sp_stats.spearmanr(wx, wy)
                    rolling_corr.append({"date": str(window[-1][2]), "spearman": round(float(rc), 4)})

    # ── Trade calendar & day-of-week (uses per-ticker decile assignments) ─
    spot_by_date = {s["date"]: s["value"] for s in spot_series} if spot_series else {}
    # Index into the complete (unfiltered) trading-day list for correct exit_date offsets
    all_spot_date_idx = {d: i for i, d in enumerate(all_spot_dates)} if all_spot_dates else {}

    trade_calendar = []
    dow_data = []
    for idx, pair in enumerate(pairs):
        x, y, d = pair[0], pair[1], pair[2]
        tkr = pair[3] if len(pair) > 3 else ticker
        yr  = d.year     if hasattr(d, 'year')    else int(str(d)[:4])
        mo  = d.month    if hasattr(d, 'month')   else int(str(d)[5:7])
        dow = d.weekday() if hasattr(d, 'weekday') else 0
        date_str = str(d.date() if hasattr(d, 'date') else d)
        dec  = pairs_decile[idx]
        dec20 = (pairs_decile20[idx] or None) if pairs_decile20 else None
        entry = {
            "ticker": tkr, "metric_val": round(x, 6),
            "year": yr, "month": mo, "date": date_str,
            "ret": round(y, 6), "decile": dec,
        }
        if dec20 is not None:
            entry["decile20"] = dec20
        # entry = open of trade_date; exit = close of trade_date + (N-1) trading days
        if is_all:
            eo = all_open_by_tkr_date.get((tkr, date_str))
            if eo is not None:
                entry["spot_entry"] = eo
            tkr_idx = all_date_idx_by_tkr.get(tkr, {})
            tkr_dates = all_dates_list_by_tkr.get(tkr, [])
            if date_str in tkr_idx:
                ei = tkr_idx[date_str] + max(horizon - 1, 0)
                if ei < len(tkr_dates):
                    ed = tkr_dates[ei]
                    entry["exit_date"] = ed
                    ec = all_close_by_tkr.get(tkr, {}).get(ed)
                    if ec is not None:
                        entry["spot_exit"] = ec
        else:
            if open_by_date and date_str in open_by_date:
                entry["spot_entry"] = open_by_date[date_str]
            elif spot_by_date and date_str in spot_by_date:
                entry["spot_entry"] = spot_by_date[date_str]
            if all_spot_date_idx and date_str in all_spot_date_idx:
                ei = all_spot_date_idx[date_str] + max(horizon - 1, 0)
                if ei < len(all_spot_dates):
                    exit_date_str = all_spot_dates[ei]
                    entry["exit_date"] = exit_date_str
                    if exit_date_str in close_by_date:
                        entry["spot_exit"] = close_by_date[exit_date_str]
        trade_calendar.append(entry)
        dow_entry = {"dow": dow, "ret": round(y, 6), "decile": dec}
        if dec20 is not None:
            dow_entry["decile20"] = dec20
        dow_data.append(dow_entry)


    # ── Today's value (single-ticker only) ───────────────────────────────
    today_val = today_pct = today_decile = None
    if not is_all and pairs:
        today_val = pairs[-1][0]
        all_x = sorted(p[0] for p in pairs)
        today_pct = round(sum(1 for v in all_x if v <= today_val) / len(all_x) * 100, 1)
        today_decile = min(int(today_pct / 10) + 1, 10)

    return {
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
        "rolling_corr":       rolling_corr,
        "trade_calendar":     trade_calendar,
        "dow_data":           dow_data,
        "spot_series":        spot_series,

        # Today (null in ALL mode)
        "today_value":      round(float(today_val), 6) if today_val is not None else None,
        "today_percentile": today_pct,
        "today_decile":     today_decile,
        "latest_date":      str(pairs[-1][2]) if pairs else None,

        # Walk-forward metadata (in-sample mode → mode='in_sample',
        # warmup=None, dropped_warmup_n=0; consumers use it to render the
        # WALK-FORWARD subtitle on the primary chart).
        "mode":             "walk_forward" if walk_forward else "in_sample",
        "warmup":           _DEFAULT_WALKFWD_WARMUP if walk_forward else None,
        "dropped_warmup_n": wf_dropped,
        "start_date":       str(pairs[0][2]) if pairs else None,
    }


@router.get("/heatmap")
async def heatmap_2d(
    ticker: str = Query(...),
    metric_x: str = Query(...),
    metric_y: str = Query(...),
    outcome: str = Query(...),
    bins: int = Query(5, ge=3, le=10),
    date_from: Optional[str] = Query(None),
    date_to: Optional[str] = Query(None),
    pool=Depends(get_oi_pool),
):
    """2D heatmap: bin metric_x and metric_y, show avg outcome in each cell.

    ALL mode applies per-ticker independent quantile binning on each axis
    then pools, matching the main quantile chart's methodology. Each
    ticker contributes evenly to every cell rather than the universe's
    absolute-range membership being dominated by tickers with naturally
    extreme magnitudes.
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

    # Always select ticker so ALL mode can group per-ticker.
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            f"SELECT ticker, {metric_x}, {metric_y}, {outcome} FROM daily_features "
            f"WHERE {metric_x} IS NOT NULL AND {metric_y} IS NOT NULL AND {outcome} IS NOT NULL"
            f"{date_conditions} ORDER BY trade_date",
            *params)

    if is_all:
        # Group by ticker, independently bin each axis per ticker, then pool
        # the joint membership into the global grid.
        by_ticker: dict = defaultdict(list)
        for r in rows:
            try:
                xv, yv, ov = float(r[metric_x]), float(r[metric_y]), float(r[outcome])
            except (TypeError, ValueError):
                continue
            if math.isnan(xv) or math.isnan(yv) or math.isnan(ov):
                continue
            by_ticker[r['ticker']].append((xv, yv, ov))

        # cell_rets[y_bin][x_bin] — outer index is y so the returned grid
        # matches the frontend's grid[iy][ix] convention (rows=y, cols=x).
        cell_rets: list = [[[] for _ in range(bins)] for _ in range(bins)]
        n_tickers_used = 0
        total_n = 0
        for items in by_ticker.values():
            if len(items) < bins:
                continue
            n_tickers_used += 1
            n_t = len(items)
            # Assign x-bin via rank, independently for this ticker.
            order_x = sorted(range(n_t), key=lambda k: items[k][0])
            x_bin = [0] * n_t
            for rank, k in enumerate(order_x):
                x_bin[k] = min(int(rank / n_t * bins), bins - 1)
            # Assign y-bin via rank, independently.
            order_y = sorted(range(n_t), key=lambda k: items[k][1])
            y_bin = [0] * n_t
            for rank, k in enumerate(order_y):
                y_bin[k] = min(int(rank / n_t * bins), bins - 1)
            # Pool outcomes into joint cells. Index order: [y][x] so rows
            # represent the y-axis bins as the template expects.
            for k in range(n_t):
                cell_rets[y_bin[k]][x_bin[k]].append(items[k][2])
                total_n += 1

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
        return {
            "metric_x":  metric_x, "metric_y": metric_y, "outcome": outcome,
            "bins":      bins, "n": total_n,
            "x_labels":  x_labels, "y_labels": y_labels,
            "grid":      grid,
            "per_ticker": True,
            "n_tickers": n_tickers_used,
        }

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
    pool=Depends(get_oi_pool),
):
    """N-bin decile stats for one metric vs one outcome (lightweight version of /analyze).

    ALL mode uses per-ticker independent quantile binning then pools, matching
    the main quantile chart's methodology. Single-ticker mode uses ordinary
    percentile binning.
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

    # Always select ticker so ALL mode can group; ignored in single-ticker.
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            f"SELECT ticker, {metric}, {outcome} FROM daily_features "
            f"WHERE {metric} IS NOT NULL AND {outcome} IS NOT NULL {date_conditions} "
            f"ORDER BY trade_date", *params)

    if is_all:
        # Per-ticker independent quantile, then pool. Tickers with < `bins`
        # observations are excluded (same rule as _bucket_pairs_per_ticker).
        by_ticker: dict = defaultdict(list)
        for r in rows:
            try:
                xf, yf = float(r[metric]), float(r[outcome])
            except (TypeError, ValueError):
                continue
            if math.isnan(xf) or math.isnan(yf):
                continue
            by_ticker[r['ticker']].append((xf, yf))
        buckets_data = _bucket_pairs_per_ticker(by_ticker, bins)
        total_n = sum(len(b) for b in buckets_data)
        if total_n < 20:
            return {"error": f"Insufficient data after per-ticker filter: {total_n} rows"}
    else:
        row_dicts = [dict(r) for r in rows]
        pairs = _clean_pairs(row_dicts, metric, outcome)
        if len(pairs) < 20:
            return {"error": f"Insufficient data: {len(pairs)} rows"}
        buckets_data = _bucket_pairs(pairs, bins)
        total_n = len(pairs)

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
    return {
        "metric":     metric,
        "outcome":    outcome,
        "bins":       bins,
        "n":          total_n,
        "buckets":    result,
        "per_ticker": is_all,
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
):
    """Return score matrix rows with optional filters."""
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

    return [dict(r) for r in rows]


@router.get("/score-matrix/meta")
async def score_matrix_meta(pool=Depends(get_pool),
                            mode: str = Query("in_sample")):
    """Return distinct metrics, tickers, fwd_rets + summary stats for filter dropdowns."""
    from research.batch_score import ensure_table
    await ensure_table(pool)

    async with pool.acquire() as conn:
        count = await conn.fetchval(
            "SELECT COUNT(*) FROM oi_score_matrix WHERE mode = $1", mode)
        if count == 0:
            return {"count": 0, "tickers": [], "metrics": [], "fwd_rets": [],
                    "avg_score": 0, "gte50": 0, "gte70": 0, "last_run": None,
                    "mode": mode}

        tickers = [r["ticker"] for r in await conn.fetch(
            "SELECT DISTINCT ticker FROM oi_score_matrix WHERE mode = $1 ORDER BY ticker",
            mode)]
        metrics = [r["metric"] for r in await conn.fetch(
            "SELECT DISTINCT metric FROM oi_score_matrix "
            "WHERE mode = $1 AND metric NOT ILIKE 'spot%' ORDER BY metric", mode)]
        fwd_rets = [r["fwd_ret"] for r in await conn.fetch(
            "SELECT DISTINCT fwd_ret FROM oi_score_matrix WHERE mode = $1 ORDER BY fwd_ret",
            mode)]
        stats = await conn.fetchrow("""
            SELECT AVG(composite_score) as avg_score,
                   COUNT(*) FILTER (WHERE composite_score >= 50) as gte50,
                   COUNT(*) FILTER (WHERE composite_score >= 70) as gte70,
                   MAX(scanned_at) as last_run
            FROM oi_score_matrix WHERE mode = $1
        """, mode)

    return {
        "count":     count,
        "tickers":   tickers,
        "metrics":   metrics,
        "fwd_rets":  fwd_rets,
        "avg_score": round(float(stats["avg_score"] or 0), 1),
        "gte50":     int(stats["gte50"] or 0),
        "gte70":     int(stats["gte70"] or 0),
        "last_run":  str(stats["last_run"])[:19] if stats["last_run"] else None,
        "mode":      mode,
    }


@router.get("/score-matrix/summary")
async def score_matrix_summary(
    pool=Depends(get_pool),
    metric: Optional[str] = None,
    fwd_ret: Optional[str] = None,
    ticker: Optional[str] = None,
    mode: str = Query("in_sample"),
):
    """Aggregated score stats with optional cross-filtering."""
    from research.batch_score import ensure_table
    await ensure_table(pool)

    # All queries share a base mode filter.  Build WHERE clauses dynamically
    # so the positional param numbers stay consistent.
    async with pool.acquire() as conn:
        # By metric
        bm_w = ["mode = $1", "metric NOT ILIKE 'spot%'"]
        bm_p: list = [mode]
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
        bf_w = ["mode = $1"]
        bf_p: list = [mode]
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
        bt_w = ["mode = $1"]
        bt_p: list = [mode]
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
        tf_w = ["mode = $1"]
        tf_p: list = [mode]
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

    def _row(r, key):
        return {key: r[key],
                "avg_score": round(float(r["avg_score"] or 0), 1),
                "std_score": round(float(r["std_score"] or 0), 1),
                "n": int(r["n"]), "gte50": int(r["gte50"]),
                "max_score": round(float(r["max_score"] or 0), 1)}

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


@router.post("/run-batch-score")
async def trigger_batch_score(
    req: BatchScoreReq = Body(default_factory=BatchScoreReq),
    pool=Depends(get_pool),
    oi_pool=Depends(get_oi_pool),
):
    """Trigger a batch score run (in-sample or walk-forward) in the background."""
    from research.batch_score import get_progress, run_batch_score
    import asyncio

    progress = get_progress()
    if progress["running"]:
        return {"status": "already_running", "message": progress["message"]}

    asyncio.get_event_loop().create_task(
        run_batch_score(oi_pool, pool, walk_forward=req.walk_forward))

    mode_label = "walk-forward" if req.walk_forward else "in-sample"
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


def _sec_score_metrics(
    all_rows: list,
    filtered_dates: list,
    outcome_col: str,
    feature_cols: list,
    is_all: bool = False,
    n_bins: int = 5,
) -> list:
    """Score each secondary feature by its lift over baseline in the filtered date subset."""
    if not filtered_dates:
        filtered = all_rows
    else:
        filtered = _filter_by_tkr_date(all_rows, _parse_tkr_date_set(filtered_dates))
    if len(filtered) < n_bins * 2:
        return []

    all_rets = [float(r[outcome_col]) for r in filtered
                if r.get(outcome_col) is not None and not math.isnan(float(r[outcome_col]))]
    if not all_rets:
        return []
    baseline_avg = float(np.mean(all_rets))
    baseline_wr  = float(np.mean([1.0 if r > 0 else 0.0 for r in all_rets]))

    results = []
    for feat in feature_cols:
        if is_all:
            # Per-ticker rank-normalize within filtered subset
            by_tkr: dict = defaultdict(list)
            for r in filtered:
                v = r.get(feat)
                o = r.get(outcome_col)
                if v is None or o is None:
                    continue
                try:
                    fv, fo = float(v), float(o)
                    if not (math.isnan(fv) or math.isnan(fo)):
                        by_tkr[r.get("ticker", "_")].append((fv, fo))
                except (TypeError, ValueError):
                    pass
            # Rank-normalize each ticker, pool into global list
            norm_vals = []
            for tkr_vals in by_tkr.values():
                if len(tkr_vals) < n_bins:
                    continue
                sorted_t = sorted(tkr_vals, key=lambda x: x[0])
                n_t = len(sorted_t)
                for rank, (_, y) in enumerate(sorted_t):
                    norm_vals.append((rank / n_t, y))
        else:
            norm_vals = []
            for r in filtered:
                v = r.get(feat)
                o = r.get(outcome_col)
                if v is None or o is None:
                    continue
                try:
                    fv, fo = float(v), float(o)
                    if not (math.isnan(fv) or math.isnan(fo)):
                        norm_vals.append((fv, fo))
                except (TypeError, ValueError):
                    pass

        if len(norm_vals) < n_bins * 2:
            continue

        sorted_vals = sorted(norm_vals, key=lambda x: x[0])
        n = len(sorted_vals)
        buckets: list = [[] for _ in range(n_bins)]
        for i, (_, y) in enumerate(sorted_vals):
            b = min(int(i / n * n_bins), n_bins - 1)
            buckets[b].append(y)

        bucket_avgs = [float(np.mean(b)) if b else None for b in buckets]
        valid = [(i, a) for i, a in enumerate(bucket_avgs) if a is not None]
        if not valid:
            continue
        best_i, best_avg = max(valid, key=lambda x: x[1])
        lift = best_avg - baseline_avg
        best_wr = float(np.mean([1.0 if y > 0 else 0.0 for y in buckets[best_i]])) if buckets[best_i] else 0.0
        win_lift = best_wr - baseline_wr

        results.append({
            "name":     feat,
            "lift":     round(lift, 6),
            "win_lift": round(win_lift, 4),
            "top_bin":  best_i + 1,
            "n_top":    len(buckets[best_i]),
            "n":        n,
        })

    results.sort(key=lambda x: x["lift"], reverse=True)
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


def _bin_membership(ordered: list, metric: str, selected_bins: set,
                    n_bins: int, is_all: bool) -> np.ndarray:
    """Binary 0/1 vector across `ordered` rows: 1 where metric's bin is in selected_bins.

    ALL mode applies per-ticker rank normalization (each ticker ranked
    independently then pooled), matching the OI Analysis primary chart and
    the secondary correlation explorer. Single-ticker mode uses a flat rank.
    Tickers (ALL mode) with fewer than n_bins observations are excluded —
    their rows stay 0.
    """
    n_rows = len(ordered)
    assignments: list = [None] * n_rows
    n_bins = max(2, min(20, n_bins))
    if is_all:
        by_tkr: dict = defaultdict(list)
        for idx, r in enumerate(ordered):
            v = r.get(metric)
            if v is None:
                continue
            try:
                fv = float(v)
                if not math.isnan(fv):
                    by_tkr[r.get("ticker", "_")].append((fv, idx))
            except (TypeError, ValueError):
                pass
        for tkr_vals in by_tkr.values():
            if len(tkr_vals) < n_bins:
                continue
            sorted_t = sorted(tkr_vals, key=lambda x: x[0])
            n_t = len(sorted_t)
            for rank, (_, orig_idx) in enumerate(sorted_t):
                assignments[orig_idx] = min(int(rank / n_t * n_bins) + 1, n_bins)
    else:
        pairs = []
        for idx, r in enumerate(ordered):
            v = r.get(metric)
            if v is None:
                continue
            try:
                fv = float(v)
                if not math.isnan(fv):
                    pairs.append((fv, idx))
            except (TypeError, ValueError):
                pass
        sorted_p = sorted(pairs, key=lambda x: x[0])
        n_t = len(sorted_p)
        for rank, (_, idx) in enumerate(sorted_p):
            assignments[idx] = min(int(rank / n_t * n_bins) + 1, n_bins)
    return np.array([1.0 if assignments[i] in selected_bins else 0.0
                     for i in range(n_rows)])


def _bin_for_value(value, history_values: list, n_bins: int):
    """Return the 1..n_bins bin that `value` would occupy in `history_values`.

    Mirrors _bin_membership's ranking math: bin = min(int(rank / n * n_bins) + 1, n_bins)
    where rank is the position in the sorted ascending list (number of
    values strictly less than `value`).

    Returns None if value is None/NaN or history is too short (< n_bins).
    """
    if value is None:
        return None
    try:
        v = float(value)
        if math.isnan(v):
            return None
    except (TypeError, ValueError):
        return None
    if not history_values or len(history_values) < n_bins:
        return None
    from bisect import bisect_left
    rank = bisect_left(history_values, v)
    n = len(history_values)
    return min(int(rank / n * n_bins) + 1, n_bins)


# ── Walk-forward bin primitives ──────────────────────────────────────────
# These are the foundation for the three walk-forward features:
#   1) /threshold-drift — bin thresholds over time
#   2) /secondary-corr-bins + /secondary-correlation walk_forward=true mode
#   3) /portfolios/{pid}/aggregate walk_forward=true mode
# All three reuse the same "rank a value against prior history per ticker"
# math via bisect_left on a running sorted list.

_DEFAULT_WALKFWD_WARMUP = 252  # trading days; ~1 year. ~2019 worth of data.


def _walk_forward_bins(rows_chrono: list, metric: str, n_bins: int,
                       is_all: bool, warmup: int = _DEFAULT_WALKFWD_WARMUP) -> dict:
    """Walk-forward bin assignment per row.

    Returns {row_index_in_input: bin_or_None}. For each row, the bin is
    computed using only data from prior dates AT THAT ROW'S TICKER
    (ALL mode) or all prior rows (single ticker). Rows whose group has
    < max(warmup, n_bins) prior observations get None and should be
    excluded from any downstream stats.

    Implementation: per group, iterate chronologically and maintain a
    sorted insertion list of prior metric values. For each new value,
    bisect_left gives the count strictly less than it (= its rank in
    [0, n_so_far)). After inserting, compute
    `min(int(rank / n_after_insert * n_bins) + 1, n_bins)` so the bin
    formula matches the in-sample _bin_membership exactly.

    Time complexity: O(N log N) per group (bisect_left is log; list
    insertion is O(N), but Python's list.insert is implemented in C so
    constants are small enough for our data sizes).
    """
    import bisect
    out: dict = {}
    n_bins = max(2, min(20, int(n_bins)))
    warm = max(int(warmup), n_bins)

    def _vals(items):
        return items

    if is_all:
        groups: dict = {}
        for i, r in enumerate(rows_chrono):
            v = r.get(metric)
            if v is None:
                continue
            try:
                fv = float(v)
                if math.isnan(fv):
                    continue
            except (TypeError, ValueError):
                continue
            tkr = r.get("ticker", "_")
            groups.setdefault(tkr, []).append((i, fv))
    else:
        flat = []
        for i, r in enumerate(rows_chrono):
            v = r.get(metric)
            if v is None:
                continue
            try:
                fv = float(v)
                if math.isnan(fv):
                    continue
            except (TypeError, ValueError):
                continue
            flat.append((i, fv))
        groups = {"_": flat}

    for _tkr, items in groups.items():
        sorted_vals: list = []
        for orig_idx, value in items:
            rank = bisect.bisect_left(sorted_vals, value)
            bisect.insort(sorted_vals, value)
            n_after = len(sorted_vals)
            if n_after < warm:
                out[orig_idx] = None
            else:
                out[orig_idx] = min(int(rank / n_after * n_bins) + 1, n_bins)
    return out


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


class SecScanReq(BaseModel):
    cache_key: str
    filtered_dates: List[str]
    ticker: str = "SPX"
    walk_forward: bool = False
    selected_primary_bins: Optional[List[int]] = None


class SecDetailReq(BaseModel):
    cache_key: str
    metric_b: str
    filtered_dates: List[str]
    sec_bins: List[int] = [10]
    sec_bin_count: int = 10
    ticker: str = "SPX"
    walk_forward: bool = False
    selected_primary_bins: Optional[List[int]] = None


@router.post("/secondary-load")
async def secondary_load(req: SecLoadReq, pool=Depends(get_oi_pool)):
    """Fetch all feature columns for the analysis date range and cache; compute initial secondary scores."""
    if not pool:
        return {"error": "OI database not configured"}

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

        # Build per-ticker calendar (open/close lookup + date sequence) so
        # combined_trades can compute exit_date + spot_exit (close of
        # entry_date + horizon-1 trading days).
        tickers_used = list({r.get("ticker") for r in rows if r.get("ticker")})
        cal_by_tkr = await _fetch_ticker_calendars(pool, tickers_used) if tickers_used else {}

        _SEC_CACHE[cache_key] = {
            "rows":           rows,
            "features":       feature_cols,
            "outcome":        req.outcome,
            "primary_metric": req.metric,
            "calendars":      cal_by_tkr,
        }

    cached = _SEC_CACHE[cache_key]
    rows = cached["rows"]
    feature_cols = cached["features"]

    metrics_result = _sec_score_metrics(rows, req.filtered_dates, req.outcome, feature_cols, is_all)

    if req.filtered_dates:
        baseline_subset = _filter_by_tkr_date(rows, _parse_tkr_date_set(req.filtered_dates))
    else:
        baseline_subset = rows
    baseline_rets = [float(r[req.outcome]) for r in baseline_subset if r.get(req.outcome) is not None]
    baseline = {
        "n": len(baseline_rets),
        "avg_ret": round(float(np.mean(baseline_rets)), 6) if baseline_rets else 0,
        "win_rate": round(float(np.mean([1.0 if v > 0 else 0.0 for v in baseline_rets])), 4) if baseline_rets else 0,
    }

    return {"cache_key": cache_key, "baseline": baseline, "metrics": metrics_result}


@router.post("/secondary-scan")
async def secondary_scan(req: SecScanReq):
    """Re-score secondary metrics for a new bin selection (in-memory from cache)."""
    cached = _SEC_CACHE.get(req.cache_key)
    if not cached:
        return {"error": "cache_miss"}

    is_all = (req.ticker == "ALL")
    rows = cached["rows"]
    feature_cols = cached["features"]
    outcome_col = cached["outcome"]
    primary_metric = cached["primary_metric"]

    if req.walk_forward:
        sel = set(int(b) for b in (req.selected_primary_bins or []))
        filtered, dropped, universe = _walk_forward_primary_filter(
            rows, primary_metric, sel, is_all)
        metrics_result = _sec_score_metrics(filtered, [], outcome_col, feature_cols, is_all)
        baseline_subset = filtered
        resp_mode = "walk_forward"
        start_date = filtered[0]["trade_date"] if filtered else None
    else:
        metrics_result = _sec_score_metrics(rows, req.filtered_dates, outcome_col, feature_cols, is_all)
        baseline_subset = (_filter_by_tkr_date(rows, _parse_tkr_date_set(req.filtered_dates))
                           if req.filtered_dates else rows)
        dropped, universe = 0, len(baseline_subset)
        resp_mode = "in_sample"
        start_date = rows[0]["trade_date"] if rows else None

    baseline_rets = [float(r[outcome_col]) for r in baseline_subset if r.get(outcome_col) is not None]
    baseline = {
        "n": len(baseline_rets),
        "avg_ret": round(float(np.mean(baseline_rets)), 6) if baseline_rets else 0,
        "win_rate": round(float(np.mean([1.0 if v > 0 else 0.0 for v in baseline_rets])), 4) if baseline_rets else 0,
    }

    return {
        "baseline":         baseline,
        "metrics":          metrics_result,
        "mode":             resp_mode,
        "dropped_warmup_n": dropped,
        "universe_n":       universe,
        "start_date":       start_date,
    }


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

    # ── Primary filter ──────────────────────────────────────────────────────
    if req.walk_forward:
        sel = set(int(b) for b in (req.selected_primary_bins or []))
        filtered, dropped, _universe = _walk_forward_primary_filter(
            all_rows, primary_metric, sel, is_all)
        resp_mode = "walk_forward"
    else:
        filtered = _filter_by_tkr_date(all_rows, _parse_tkr_date_set(req.filtered_dates))
        dropped = 0
        resp_mode = "in_sample"

    if len(filtered) < n_bins * 2:
        return {"error": "insufficient_data"}

    # ── Secondary binning ───────────────────────────────────────────────────
    if req.walk_forward:
        # Walk-forward bins within the primary-filtered chronological subset.
        # warmup=n_bins: tiny warmup so even small filtered sets work.
        filtered_chrono = _sort_chrono(filtered)
        wf_sec = _walk_forward_bins(filtered_chrono, req.metric_b, n_bins, is_all, warmup=n_bins)
        buckets: list = [[] for _ in range(n_bins)]
        for i, r in enumerate(filtered_chrono):
            b = wf_sec.get(i)
            if b is None:
                continue
            v = r.get(req.metric_b)
            o = r.get(outcome_col)
            if v is None or o is None:
                continue
            try:
                fv, fo = float(v), float(o)
                if not (math.isnan(fv) or math.isnan(fo)):
                    buckets[b - 1].append((fv, fo, r.get("trade_date", ""), r.get("ticker", "")))
            except (TypeError, ValueError):
                pass
    else:
        # In-sample: per-ticker rank-normalize for ALL, global rank for single.
        if is_all:
            by_tkr: dict = defaultdict(list)
            for r in filtered:
                v = r.get(req.metric_b)
                o = r.get(outcome_col)
                if v is None or o is None:
                    continue
                try:
                    fv, fo = float(v), float(o)
                    if not (math.isnan(fv) or math.isnan(fo)):
                        by_tkr[r.get("ticker", "_")].append(
                            (fv, fo, r.get("trade_date", ""), r.get("ticker", ""))
                        )
                except (TypeError, ValueError):
                    pass
            norm_rows = []
            for tkr_vals in by_tkr.values():
                if len(tkr_vals) < n_bins:
                    continue
                sorted_t = sorted(tkr_vals, key=lambda x: x[0])
                n_t = len(sorted_t)
                for rank, (_, y, d, tkr) in enumerate(sorted_t):
                    norm_rows.append((rank / n_t, y, d, tkr))
        else:
            norm_rows = []
            for r in filtered:
                v = r.get(req.metric_b)
                o = r.get(outcome_col)
                if v is None or o is None:
                    continue
                try:
                    fv, fo = float(v), float(o)
                    if not (math.isnan(fv) or math.isnan(fo)):
                        norm_rows.append((fv, fo, r.get("trade_date", ""), r.get("ticker", "")))
                except (TypeError, ValueError):
                    pass

        if len(norm_rows) < n_bins * 2:
            return {"error": "insufficient_data"}

        sorted_norm = sorted(norm_rows, key=lambda x: x[0])
        n = len(sorted_norm)
        buckets: list = [[] for _ in range(n_bins)]
        for i, row_t in enumerate(sorted_norm):
            b = min(int(i / n * n_bins), n_bins - 1)
            buckets[b].append(row_t)

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
        "mode":            resp_mode,
        "dropped_warmup_n": dropped,
    }


# ── Multi-Metric Correlation Explorer endpoints ───────────────────────────────

def _compute_bins_for_metric(filtered: list, feat: str, outcome_col: str,
                              n_bins: int, is_all: bool) -> dict | None:
    """Return per-bin avg_ret (and n) for one feature over a filtered row set."""
    if is_all:
        by_tkr: dict = defaultdict(list)
        for r in filtered:
            v, o = r.get(feat), r.get(outcome_col)
            if v is None or o is None:
                continue
            try:
                fv, fo = float(v), float(o)
                if not (math.isnan(fv) or math.isnan(fo)):
                    by_tkr[r.get("ticker", "_")].append((fv, fo))
            except (TypeError, ValueError):
                pass
        norm_vals = []
        for tkr_vals in by_tkr.values():
            if len(tkr_vals) < n_bins:
                continue
            sorted_t = sorted(tkr_vals, key=lambda x: x[0])
            n_t = len(sorted_t)
            for rank, (_, y) in enumerate(sorted_t):
                norm_vals.append((rank / n_t, y))
    else:
        norm_vals = []
        for r in filtered:
            v, o = r.get(feat), r.get(outcome_col)
            if v is None or o is None:
                continue
            try:
                fv, fo = float(v), float(o)
                if not (math.isnan(fv) or math.isnan(fo)):
                    norm_vals.append((fv, fo))
            except (TypeError, ValueError):
                pass

    if len(norm_vals) < n_bins * 2:
        return None

    sorted_vals = sorted(norm_vals, key=lambda x: x[0])
    n = len(sorted_vals)
    buckets: list = [[] for _ in range(n_bins)]
    for i, (_, y) in enumerate(sorted_vals):
        b = min(int(i / n * n_bins), n_bins - 1)
        buckets[b].append(y)

    return {
        "name":    feat,
        "bins":    [round(float(np.mean(b)), 6) if b else 0.0 for b in buckets],
        "bin_ns":  [len(b) for b in buckets],
    }


class CorrBinsReq(BaseModel):
    cache_key: str
    filtered_dates: List[str]
    ticker: str = "SPX"
    n_bins: int = 10
    walk_forward: bool = False
    selected_primary_bins: Optional[List[int]] = None  # 1..20 primary bin ids (walk_forward mode)


def _sort_chrono(rows: list) -> list:
    """Stable chronological sort by (trade_date, ticker)."""
    return sorted(rows, key=lambda r: (r.get("trade_date", ""), r.get("ticker", "")))


def _walk_forward_primary_filter(rows: list, primary_metric: str,
                                  selected_primary_bins: set,
                                  is_all: bool) -> tuple:
    """Walk-forward equivalent of `_filter_by_tkr_date` for the corr-explorer.

    Sorts `rows` chronologically, computes walk-forward bins for
    `primary_metric` (20-bin universe, matching the OI Analysis primary
    chart), then keeps rows whose walk-forward bin is in
    `selected_primary_bins`. Rows in warmup (bin=None) are excluded.

    Returns (filtered_chrono_rows, dropped_warmup_n, universe_n).
      filtered_chrono_rows — rows matching selected_primary_bins
      dropped_warmup_n     — rows excluded because their primary wf bin
                             is None (warmup not yet satisfied, or
                             missing primary metric value)
      universe_n           — total rows with a defined wf primary bin
                             (= len(rows) - dropped_warmup_n). Useful so
                             the UI can show the post-warmup universe
                             separate from the primary-filtered subset.
    """
    ordered = _sort_chrono(rows)
    wf = _walk_forward_bins(ordered, primary_metric, 20, is_all)
    kept: list = []
    dropped = 0
    universe = 0
    sel = set(int(b) for b in (selected_primary_bins or []))
    for i, r in enumerate(ordered):
        b = wf.get(i)
        if b is None:
            dropped += 1
            continue
        universe += 1
        if sel and b not in sel:
            continue
        kept.append(r)
    return kept, dropped, universe


def _compute_walk_forward_bin_stats(rows_chrono: list, feat: str, outcome_col: str,
                                    n_bins: int, is_all: bool) -> dict | None:
    """Walk-forward equivalent of `_compute_bins_for_metric` for one feature.

    Within the (already walk-forward-primary-filtered) chronological row
    set, compute walk-forward bins for `feat` and aggregate per-bin avg
    outcome. Uses a small warmup (= n_bins) inside the subset since the
    primary filter has already enforced the macro warmup gate.
    """
    n_bins = max(2, min(20, int(n_bins)))
    # Within an already-walk-forward subset we just need enough samples per
    # group to make `min(rank/n*n_bins)+1` meaningful; n_bins is enough.
    wf = _walk_forward_bins(rows_chrono, feat, n_bins, is_all, warmup=n_bins)
    buckets: list = [[] for _ in range(n_bins)]
    for i, r in enumerate(rows_chrono):
        b = wf.get(i)
        if b is None:
            continue
        o = r.get(outcome_col)
        if o is None:
            continue
        try:
            ov = float(o)
            if math.isnan(ov):
                continue
        except (TypeError, ValueError):
            continue
        buckets[b - 1].append(ov)
    if all(len(bk) == 0 for bk in buckets):
        return None
    return {
        "name":   feat,
        "bins":   [round(float(np.mean(bk)), 6) if bk else 0.0 for bk in buckets],
        "bin_ns": [len(bk) for bk in buckets],
    }


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

    if req.walk_forward:
        primary_metric = cached.get("primary_metric") or ""
        if not primary_metric:
            return {"error": "no_primary_metric"}
        filtered, dropped, universe = _walk_forward_primary_filter(
            all_rows, primary_metric, set(req.selected_primary_bins or []), is_all
        )
        if not filtered:
            return {"error": "no_data", "mode": "walk_forward",
                    "warmup": _DEFAULT_WALKFWD_WARMUP, "dropped_warmup_n": dropped,
                    "universe_n": universe}
        results = []
        for feat in feat_cols:
            r = _compute_walk_forward_bin_stats(filtered, feat, outcome_col, n_bins, is_all)
            if r:
                results.append(r)
        return {
            "metrics": results, "n_bins": n_bins,
            "mode": "walk_forward",
            "warmup": _DEFAULT_WALKFWD_WARMUP,
            "dropped_warmup_n": dropped,
            "universe_n":       universe,
            "combined_n":       len(filtered),
            "start_date":       filtered[0].get("trade_date", "") if filtered else "",
        }

    filtered = _filter_by_tkr_date(all_rows, _parse_tkr_date_set(req.filtered_dates))
    if not filtered:
        return {"error": "no_data"}

    results = []
    for feat in feat_cols:
        r = _compute_bins_for_metric(filtered, feat, outcome_col, n_bins, is_all)
        if r:
            results.append(r)

    return {"metrics": results, "n_bins": n_bins, "mode": "in_sample"}


class CorrReq(BaseModel):
    cache_key: str
    filtered_dates: List[str]
    ticker: str = "SPX"
    n_bins: int = 10
    selections: List[dict]  # [{metric: str, bins: [int]}]
    walk_forward: bool = False
    selected_primary_bins: Optional[List[int]] = None  # 1..20 primary bin ids (walk_forward mode)


def _walk_forward_membership(rows_chrono: list, metric: str, selected_bins: set,
                              n_bins: int, is_all: bool) -> np.ndarray:
    """Walk-forward equivalent of `_bin_membership`.

    Computes walk-forward bins for `metric` over the already-filtered
    chronological row set, returning a 0/1 vector where 1 = the row's
    walk-forward bin is in `selected_bins`. Warmup rows (bin=None) stay
    0. Uses a small warmup (= n_bins) since the primary filter has
    already enforced the macro warmup gate.
    """
    n_rows = len(rows_chrono)
    out = np.zeros(n_rows, dtype=np.float64)
    if not selected_bins:
        return out
    wf = _walk_forward_bins(rows_chrono, metric, n_bins, is_all, warmup=n_bins)
    for i in range(n_rows):
        b = wf.get(i)
        if b is not None and b in selected_bins:
            out[i] = 1.0
    return out


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

    if req.walk_forward:
        primary_metric = cached.get("primary_metric") or ""
        if not primary_metric:
            return {"error": "no_primary_metric"}
        filtered, dropped, universe = _walk_forward_primary_filter(
            all_rows, primary_metric, set(req.selected_primary_bins or []), is_all
        )
        if not filtered:
            return {"error": "no_data", "mode": "walk_forward",
                    "warmup": _DEFAULT_WALKFWD_WARMUP, "dropped_warmup_n": dropped,
                    "universe_n": universe}
        ordered = filtered  # already chronologically sorted by _walk_forward_primary_filter
        mode_out = "walk_forward"
    else:
        filtered = _filter_by_tkr_date(all_rows, _parse_tkr_date_set(req.filtered_dates))
        if not filtered:
            return {"error": "no_data"}
        ordered = sorted(filtered, key=lambda r: (r.get("trade_date", ""), r.get("ticker", "")))
        dropped = 0
        universe = len(ordered)
        mode_out = "in_sample"

    vectors, metric_names, n_each = [], [], []
    for sel in req.selections:
        metric = sel.get("metric", "")
        bins   = set(sel.get("bins", []))
        if not metric or not bins:
            continue
        if req.walk_forward:
            vec = _walk_forward_membership(ordered, metric, bins, n_bins, is_all)
        else:
            vec = _bin_membership(ordered, metric, bins, n_bins, is_all)
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
        "mode":             mode_out,
        "warmup":           _DEFAULT_WALKFWD_WARMUP if req.walk_forward else None,
        "dropped_warmup_n": dropped,
        "universe_n":       universe,
        "start_date":       ordered[0].get("trade_date", "") if ordered else "",
    }


# ── Global Metric Bins (standalone top-of-page browser) ───────────────────

_GLOBAL_BINS_CACHE: dict = {}


def _compute_all_bins_fast(rows: list, feature_cols: list, outcome_col: str,
                           n_bins: int, is_all: bool) -> list:
    """Numpy-vectorized batch bin computation for many features at once.

    Replaces the per-metric Python loop in `_compute_bins_for_metric` with
    a single-pass NaN-aware numpy pipeline. ~10x faster on >100k-row
    daily_features fetches with 80+ feature columns — keeps the All-Ticker
    Metric Bins endpoint under Cloudflare's ~100s upstream timeout.
    """
    n = len(rows)
    if n < n_bins * 2 or not feature_cols:
        return []

    # Ticker → int id (single pass).
    tickers_str = np.array([r.get("ticker", "_") for r in rows], dtype=object)
    unique_tkrs, ticker_id = np.unique(tickers_str, return_inverse=True)
    n_tkrs = len(unique_tkrs)
    ticker_indices = [np.where(ticker_id == t)[0] for t in range(n_tkrs)]

    # Outcome column as numpy float (NaN for missing/bad).
    def _to_float_arr(seq):
        arr = np.empty(len(seq), dtype=np.float64)
        for i, v in enumerate(seq):
            if v is None:
                arr[i] = np.nan
                continue
            try:
                fv = float(v)
                arr[i] = fv if not math.isnan(fv) else np.nan
            except (TypeError, ValueError):
                arr[i] = np.nan
        return arr

    outcomes = _to_float_arr([r.get(outcome_col) for r in rows])

    results = []
    for feat in feature_cols:
        col = _to_float_arr([r.get(feat) for r in rows])

        # Per-ticker fractional ranks (ALL mode) or one big bucket (single).
        all_ranks: list = []
        all_outs:  list = []
        if is_all:
            for idxs in ticker_indices:
                vals = col[idxs]
                outs = outcomes[idxs]
                m = ~np.isnan(vals) & ~np.isnan(outs)
                if m.sum() < n_bins:
                    continue
                v_clean = vals[m]
                o_clean = outs[m]
                order = np.argsort(v_clean)
                n_t = len(v_clean)
                all_ranks.append(np.arange(n_t) / n_t)
                all_outs.append(o_clean[order])
        else:
            m = ~np.isnan(col) & ~np.isnan(outcomes)
            if m.sum() < n_bins:
                continue
            v_clean = col[m]
            o_clean = outcomes[m]
            order = np.argsort(v_clean)
            n_t = len(v_clean)
            all_ranks.append(np.arange(n_t) / n_t)
            all_outs.append(o_clean[order])

        if not all_ranks:
            continue
        ranks_flat = np.concatenate(all_ranks)
        outs_flat  = np.concatenate(all_outs)
        if len(ranks_flat) < n_bins * 2:
            continue

        order2 = np.argsort(ranks_flat)
        outs_flat = outs_flat[order2]
        n2 = len(ranks_flat)
        bin_idx = np.minimum((np.arange(n2) / n2 * n_bins).astype(int), n_bins - 1)

        bins_avg = np.zeros(n_bins)
        bins_n   = np.zeros(n_bins, dtype=int)
        for b in range(n_bins):
            mm = bin_idx == b
            bins_n[b] = int(mm.sum())
            if bins_n[b] > 0:
                bins_avg[b] = float(outs_flat[mm].mean())

        results.append({
            "name":   feat,
            "bins":   [round(float(v), 6) for v in bins_avg],
            "bin_ns": [int(v) for v in bins_n],
        })

    return results


def _compute_all_bins_walk_forward(rows: list, feature_cols: list,
                                   outcome_col: str, n_bins: int, is_all: bool,
                                   warmup: int = _DEFAULT_WALKFWD_WARMUP) -> tuple:
    """Walk-forward batch bin computation for global-metric-bins.

    Per-ticker j-loop: for each row j, adds 1 to wf_rank[i, f] for all i > j
    where X[i, f] > X[j, f]. This is equivalent to bisect_left rank counting
    but operates on a (N-j, F) slice per step instead of one (N, N) matrix.
    Peak memory: 2 × (N, F) arrays per ticker (~1.6 MB for N=1 300, F=80).

    Returns:
        (results, dropped_warmup_n, start_date)
        results: list of {"name", "bins", "bin_ns"} — same format as
                 _compute_all_bins_fast.
        dropped_warmup_n: rows excluded to warmup (outcome-column level).
        start_date: earliest trade_date string that cleared warmup (or None).
    """
    n_bins = max(2, min(20, n_bins))
    warm   = max(int(warmup), n_bins)
    F      = len(feature_cols)

    def _to_f64(seq):
        a = np.empty(len(seq), dtype=np.float64)
        for i, v in enumerate(seq):
            if v is None:
                a[i] = np.nan; continue
            try:
                fv = float(v)
                a[i] = fv if not math.isnan(fv) else np.nan
            except (TypeError, ValueError):
                a[i] = np.nan
        return a

    # Group by ticker and sort chronologically.
    by_ticker: dict = {}
    for r in rows:
        tkr = r.get("ticker", "_") if is_all else "_"
        by_ticker.setdefault(tkr, []).append(r)
    for tkr in by_ticker:
        by_ticker[tkr].sort(key=lambda r: r.get("trade_date", ""))

    feat_bins: list = [[[] for _ in range(n_bins)] for _ in range(F)]
    dropped_total = 0
    start_date    = None

    for tkr, tkr_rows in by_ticker.items():
        N = len(tkr_rows)

        # Extract (N, F) feature matrix and outcome vector.
        outcomes = _to_f64([r.get(outcome_col) for r in tkr_rows])
        X = np.empty((N, F), dtype=np.float64)
        for f_idx, feat in enumerate(feature_cols):
            X[:, f_idx] = _to_f64([r.get(feat) for r in tkr_rows])

        outcome_valid = ~np.isnan(outcomes)

        # Dropped / start_date at outcome-column level.
        n_valid_outcome = int(outcome_valid.sum())
        dropped_total += min(warm, n_valid_outcome)
        if n_valid_outcome > warm:
            first_past = int(np.where(outcome_valid)[0][warm])
            d = tkr_rows[first_past].get("trade_date")
            if d is not None:
                ds = str(d)
                if start_date is None or ds < start_date:
                    start_date = ds

        # wf_rank[i, f] = #{j < i : X[j, f] < X[i, f]}
        # j-loop: each step contributes to all later rows simultaneously.
        # NaN comparisons return False so NaN rows auto-contribute 0.
        wf_rank = np.zeros((N, F), dtype=np.int32)
        for j in range(N - 1):
            wf_rank[j + 1:] += X[j + 1:] > X[j]   # bool adds as 0/1

        # Per-feature cumulative non-NaN count → n_after denominator.
        nan_mask    = np.isnan(X)
        valid_cum   = np.cumsum(~nan_mask, axis=0, dtype=np.int32)   # (N, F)
        safe_n      = np.where(valid_cum > 0, valid_cum, 1).astype(np.float64)
        bin_mat     = np.minimum(
            (wf_rank.astype(np.float64) / safe_n * n_bins).astype(np.int32),
            n_bins - 1,
        )   # (N, F)

        # use_mask: valid feature value, valid outcome, past warmup.
        past_warm_mat = (~nan_mask) & (valid_cum >= warm)           # (N, F)
        use_mask      = past_warm_mat & outcome_valid[:, np.newaxis] # (N, F)

        # Accumulate outcomes per feature per bin.
        for f_idx in range(F):
            col_mask = use_mask[:, f_idx]
            if not col_mask.any():
                continue
            o_sel = outcomes[col_mask]
            b_sel = bin_mat[col_mask, f_idx]
            for b in range(n_bins):
                hits = o_sel[b_sel == b]
                if hits.size:
                    feat_bins[f_idx][b].extend(hits.tolist())

    # Build results in the same format as _compute_all_bins_fast.
    results = []
    for f_idx, feat in enumerate(feature_cols):
        total = sum(len(b) for b in feat_bins[f_idx])
        if total < n_bins * 2:
            continue
        results.append({
            "name":   feat,
            "bins":   [round(float(np.mean(b)), 6) if b else 0.0
                       for b in feat_bins[f_idx]],
            "bin_ns": [len(b) for b in feat_bins[f_idx]],
        })

    return results, dropped_total, start_date


@router.get("/global-metric-bins")
async def global_metric_bins(
    outcome:      str  = Query("ret_5d_fwd_oc"),
    ticker:       str  = Query("ALL"),
    n_bins:       int  = Query(20, ge=2, le=20),
    date_from:    Optional[str] = Query(None),
    date_to:      Optional[str] = Query(None),
    walk_forward: bool = Query(False),
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
    mode_tag = "wf" if walk_forward else "is"
    cache_key = f"{ticker}|{outcome}|{n_bins}|{date_from or ''}|{date_to or ''}|{mode_tag}"
    if cache_key in _GLOBAL_BINS_CACHE:
        return _GLOBAL_BINS_CACHE[cache_key]

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

        if walk_forward:
            metrics_out, dropped_n, wf_start = _compute_all_bins_walk_forward(
                rows, feature_cols, outcome, n_bins, is_all,
                warmup=_DEFAULT_WALKFWD_WARMUP,
            )
        else:
            # Vectorised batch helper — ~10x faster than the per-metric Python
            # loop for large daily_features fetches.
            metrics_out = _compute_all_bins_fast(rows, feature_cols, outcome, n_bins, is_all)
            dropped_n = None
            wf_start = None

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
            "mode":              "walk_forward" if walk_forward else "in_sample",
        }
        if walk_forward:
            out["warmup"]           = _DEFAULT_WALKFWD_WARMUP
            out["dropped_warmup_n"] = dropped_n
            out["start_date"]       = wf_start

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
async def global_metric_bins_invalidate():
    """Drop the in-memory cache so a fresh fetch is computed (e.g. after the
    user adds new metric columns to daily_features)."""
    _GLOBAL_BINS_CACHE.clear()
    return {"ok": True}


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
