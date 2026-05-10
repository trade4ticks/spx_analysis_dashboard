"""OI Analysis workbench — interactive decile analytics for a single ticker/metric/outcome."""
import json
import math
from collections import defaultdict
from typing import List, Optional

import numpy as np
from scipy import stats as sp_stats
from fastapi import APIRouter, Body, Depends, Query
from pydantic import BaseModel

from app.db import get_pool, get_oi_pool

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
    features = [c for c in all_cols if c not in outcomes]
    return {"features": features, "outcomes": outcomes}


@router.get("/analyze")
async def analyze(
    ticker: str = Query(...),
    metric: str = Query(...),
    outcome: str = Query(...),
    date_from: Optional[str] = Query(None),
    date_to: Optional[str] = Query(None),
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
                f"SELECT ticker, trade_date, {metric}, {outcome} FROM daily_features "
                f"WHERE {metric} IS NOT NULL AND {outcome} IS NOT NULL {date_conditions} "
                f"ORDER BY ticker, trade_date", *params)
        row_dicts = [dict(r) for r in rows]

        by_ticker: dict = defaultdict(list)
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

        buckets = _bucket_pairs_per_ticker(by_ticker, 10)
        buckets_20_all = _bucket_pairs_per_ticker(by_ticker, 20)

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

    else:
        # Single-ticker mode
        single_ticker_cond = f" AND ticker = ${p}"
        params_single = params + [ticker]
        async with pool.acquire() as conn:
            col_check = await conn.fetch(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_name = 'daily_features' AND column_name IN "
                "('spot_op','spot_open','spot_co','spot_close','open','close')")
        spot_cols_found = {r["column_name"] for r in col_check}
        # Entry = open of trade_date; Exit = close of trade_date + (N-1) trading days
        spot_open_col  = next((c for c in ('spot_op','spot_open','open')  if c in spot_cols_found), None)
        spot_close_col = next((c for c in ('spot_co','spot_close','close') if c in spot_cols_found), None)
        spot_select = ""
        if spot_open_col:
            spot_select += f", {spot_open_col}"
        if spot_close_col and spot_close_col != spot_open_col:
            spot_select += f", {spot_close_col}"

        async with pool.acquire() as conn:
            rows = await conn.fetch(
                f"SELECT trade_date, {metric}, {outcome}{spot_select} FROM daily_features "
                f"WHERE {metric} IS NOT NULL AND {outcome} IS NOT NULL {single_ticker_cond}"
                f"{date_conditions} ORDER BY trade_date", *params_single)
        row_dicts = [dict(r) for r in rows]
        pairs = _clean_pairs(row_dicts, metric, outcome)

        buckets = _bucket_pairs(pairs, 10)
        buckets_20 = _bucket_pairs(pairs, 20)

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

        # spot_series for chart display uses close; entry_spot uses open
        spot_col = spot_close_col or spot_open_col  # fallback for spot_series chart
        spot_series = []
        all_spot_dates: list = []   # complete ordered trading-day list (unfiltered by metric/outcome)
        open_by_date: dict = {}     # trade_date → open price (for entry_spot)
        for r in row_dicts:
            date_s = str(r["trade_date"])
            if spot_open_col:
                ov = r.get(spot_open_col)
                if ov is not None:
                    try:
                        open_by_date[date_s] = round(float(ov), 2)
                    except (ValueError, TypeError):
                        pass
            if spot_col:
                sv = r.get(spot_col)
                if sv is not None:
                    try:
                        spot_series.append({"date": date_s, "value": round(float(sv), 2)})
                    except (ValueError, TypeError):
                        pass
        if spot_col:
            # Fetch complete date list so exit_date offset is always correct
            async with pool.acquire() as conn:
                all_dates_rows = await conn.fetch(
                    f"SELECT trade_date FROM daily_features "
                    f"WHERE ticker = $1 AND {spot_col} IS NOT NULL "
                    f"ORDER BY trade_date", ticker)
            all_spot_dates = [str(r["trade_date"]) for r in all_dates_rows]

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
        # entry_spot = open of trade_date; exit_date = trade_date + (N-1) trading days
        if open_by_date and date_str in open_by_date:
            entry["spot_entry"] = open_by_date[date_str]
        elif spot_by_date and date_str in spot_by_date:
            entry["spot_entry"] = spot_by_date[date_str]  # fallback to close if no open col
        if all_spot_date_idx and date_str in all_spot_date_idx:
            ei = all_spot_date_idx[date_str] + max(horizon - 1, 0)
            if ei < len(all_spot_dates):
                entry["exit_date"] = all_spot_dates[ei]
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
    }


@router.get("/heatmap")
async def heatmap_2d(
    ticker: str = Query(...),
    metric_x: str = Query(...),
    metric_y: str = Query(...),
    outcome: str = Query(...),
    bins: int = Query(5, ge=3, le=10),
    pool=Depends(get_oi_pool),
):
    """2D heatmap: bin metric_x and metric_y, show avg outcome in each cell."""
    if not pool:
        return {"error": "OI database not configured"}

    async with pool.acquire() as conn:
        if ticker == "ALL":
            rows = await conn.fetch(
                f"SELECT {metric_x}, {metric_y}, {outcome} FROM daily_features "
                f"WHERE {metric_x} IS NOT NULL AND {metric_y} IS NOT NULL AND {outcome} IS NOT NULL "
                f"ORDER BY trade_date")
        else:
            rows = await conn.fetch(
                f"SELECT {metric_x}, {metric_y}, {outcome} FROM daily_features "
                f"WHERE ticker = $1 AND {metric_x} IS NOT NULL AND {metric_y} IS NOT NULL "
                f"AND {outcome} IS NOT NULL ORDER BY trade_date", ticker)

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

    grid = []
    for i in range(bins):
        row = []
        for j in range(bins):
            mask = ((xs >= x_edges[i]) & (xs < x_edges[i+1]) &
                    (ys >= y_edges[j]) & (ys < y_edges[j+1]))
            cell_rets = os_[mask]
            if len(cell_rets) >= 5:
                row.append({
                    "avg_ret": round(float(cell_rets.mean()), 6),
                    "win_rate": round(float((cell_rets > 0).mean()), 4),
                    "n": int(len(cell_rets)),
                })
            else:
                row.append(None)
        grid.append(row)

    x_labels = [f"{x_edges[i]:.2f}–{x_edges[i+1]:.2f}" for i in range(bins)]
    y_labels = [f"{y_edges[j]:.2f}–{y_edges[j+1]:.2f}" for j in range(bins)]

    return {
        "metric_x": metric_x, "metric_y": metric_y, "outcome": outcome,
        "bins": bins, "n": len(valid),
        "x_labels": x_labels, "y_labels": y_labels,
        "grid": grid,
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
    """N-bin decile stats for one metric vs one outcome (lightweight version of /analyze)."""
    if not pool:
        return {"error": "OI database not configured"}
    bins = max(2, min(20, bins))
    date_conditions = ""
    params: list = []
    p = 1
    if ticker != "ALL":
        date_conditions += f" AND ticker = ${p}"; params.append(ticker); p += 1
    if date_from:
        from datetime import date as _date
        date_conditions += f" AND trade_date >= ${p}"; params.append(_date.fromisoformat(date_from)); p += 1
    if date_to:
        from datetime import date as _date
        date_conditions += f" AND trade_date <= ${p}"; params.append(_date.fromisoformat(date_to)); p += 1
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            f"SELECT {metric}, {outcome} FROM daily_features "
            f"WHERE {metric} IS NOT NULL AND {outcome} IS NOT NULL {date_conditions} "
            f"ORDER BY trade_date", *params)
    row_dicts = [dict(r) for r in rows]
    pairs = _clean_pairs(row_dicts, metric, outcome)
    if len(pairs) < 20:
        return {"error": f"Insufficient data: {len(pairs)} rows"}
    buckets_data = _bucket_pairs(pairs, bins)
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
            "min_val":  round(float(min(xs)), 6),
            "max_val":  round(float(max(xs)), 6),
        })
    return {"metric": metric, "outcome": outcome, "bins": bins, "n": len(pairs), "buckets": result}


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

    where = ["composite_score >= $1", "metric NOT ILIKE 'spot%'"]
    params = [min_score]
    idx = 2

    if ticker:
        where.append(f"ticker = ${idx}")
        params.append(ticker)
        idx += 1
    if metric:
        where.append(f"metric = ${idx}")
        params.append(metric)
        idx += 1
    if fwd_ret:
        where.append(f"fwd_ret = ${idx}")
        params.append(fwd_ret)
        idx += 1

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
async def score_matrix_meta(pool=Depends(get_pool)):
    """Return distinct metrics, tickers, fwd_rets + summary stats for filter dropdowns."""
    from research.batch_score import ensure_table
    await ensure_table(pool)

    async with pool.acquire() as conn:
        count = await conn.fetchval("SELECT COUNT(*) FROM oi_score_matrix")
        if count == 0:
            return {"count": 0, "tickers": [], "metrics": [], "fwd_rets": [],
                    "avg_score": 0, "gte50": 0, "gte70": 0, "last_run": None}

        tickers = [r["ticker"] for r in await conn.fetch(
            "SELECT DISTINCT ticker FROM oi_score_matrix ORDER BY ticker")]
        metrics = [r["metric"] for r in await conn.fetch(
            "SELECT DISTINCT metric FROM oi_score_matrix WHERE metric NOT ILIKE 'spot%' ORDER BY metric")]
        fwd_rets = [r["fwd_ret"] for r in await conn.fetch(
            "SELECT DISTINCT fwd_ret FROM oi_score_matrix ORDER BY fwd_ret")]
        stats = await conn.fetchrow("""
            SELECT AVG(composite_score) as avg_score,
                   COUNT(*) FILTER (WHERE composite_score >= 50) as gte50,
                   COUNT(*) FILTER (WHERE composite_score >= 70) as gte70,
                   MAX(scanned_at) as last_run
            FROM oi_score_matrix
        """)

    return {
        "count": count,
        "tickers": tickers,
        "metrics": metrics,
        "fwd_rets": fwd_rets,
        "avg_score": round(float(stats["avg_score"] or 0), 1),
        "gte50": int(stats["gte50"] or 0),
        "gte70": int(stats["gte70"] or 0),
        "last_run": str(stats["last_run"])[:19] if stats["last_run"] else None,
    }


@router.get("/score-matrix/summary")
async def score_matrix_summary(
    pool=Depends(get_pool),
    metric: Optional[str] = None,
    fwd_ret: Optional[str] = None,
    ticker: Optional[str] = None,
):
    """Aggregated score stats with optional cross-filtering."""
    from research.batch_score import ensure_table
    await ensure_table(pool)

    async with pool.acquire() as conn:
        # By metric — filtered by fwd_ret if selected
        if fwd_ret:
            by_metric = await conn.fetch("""
                SELECT metric, AVG(composite_score) as avg_score,
                       STDDEV(composite_score) as std_score, COUNT(*) as n,
                       COUNT(*) FILTER (WHERE composite_score >= 50) as gte50,
                       MAX(composite_score) as max_score
                FROM oi_score_matrix WHERE fwd_ret = $1 AND metric NOT ILIKE 'spot%'
                GROUP BY metric ORDER BY AVG(composite_score) DESC
            """, fwd_ret)
        else:
            by_metric = await conn.fetch("""
                SELECT metric, AVG(composite_score) as avg_score,
                       STDDEV(composite_score) as std_score, COUNT(*) as n,
                       COUNT(*) FILTER (WHERE composite_score >= 50) as gte50,
                       MAX(composite_score) as max_score
                FROM oi_score_matrix WHERE metric NOT ILIKE 'spot%'
                GROUP BY metric ORDER BY AVG(composite_score) DESC
            """)

        # By fwd_ret — filtered by metric if selected
        if metric:
            by_fwd = await conn.fetch("""
                SELECT fwd_ret, AVG(composite_score) as avg_score,
                       STDDEV(composite_score) as std_score, COUNT(*) as n,
                       COUNT(*) FILTER (WHERE composite_score >= 50) as gte50,
                       MAX(composite_score) as max_score
                FROM oi_score_matrix WHERE metric = $1
                GROUP BY fwd_ret ORDER BY AVG(composite_score) DESC
            """, metric)
        else:
            by_fwd = await conn.fetch("""
                SELECT fwd_ret, AVG(composite_score) as avg_score,
                       STDDEV(composite_score) as std_score, COUNT(*) as n,
                       COUNT(*) FILTER (WHERE composite_score >= 50) as gte50,
                       MAX(composite_score) as max_score
                FROM oi_score_matrix
                GROUP BY fwd_ret ORDER BY AVG(composite_score) DESC
            """)

        # By ticker — filtered by metric and fwd_ret if selected
        ticker_wheres, ticker_params = [], []
        if metric:
            ticker_params.append(metric)
            ticker_wheres.append(f"metric = ${len(ticker_params)}")
        if fwd_ret:
            ticker_params.append(fwd_ret)
            ticker_wheres.append(f"fwd_ret = ${len(ticker_params)}")
        ticker_where_sql = ("WHERE " + " AND ".join(ticker_wheres)) if ticker_wheres else ""
        by_ticker = await conn.fetch(f"""
            SELECT ticker, AVG(composite_score) as avg_score,
                   STDDEV(composite_score) as std_score, COUNT(*) as n,
                   COUNT(*) FILTER (WHERE composite_score >= 50) as gte50,
                   MAX(composite_score) as max_score
            FROM oi_score_matrix {ticker_where_sql}
            GROUP BY ticker ORDER BY AVG(composite_score) DESC
        """, *ticker_params)

        # By fwd_ret scoped to a ticker — filtered by ticker (and metric if selected)
        tfwd_wheres, tfwd_params = [], []
        if ticker:
            tfwd_params.append(ticker)
            tfwd_wheres.append(f"ticker = ${len(tfwd_params)}")
        if metric:
            tfwd_params.append(metric)
            tfwd_wheres.append(f"metric = ${len(tfwd_params)}")
        tfwd_where_sql = ("WHERE " + " AND ".join(tfwd_wheres)) if tfwd_wheres else ""
        by_fwd_ticker = await conn.fetch(f"""
            SELECT fwd_ret, AVG(composite_score) as avg_score,
                   STDDEV(composite_score) as std_score, COUNT(*) as n,
                   COUNT(*) FILTER (WHERE composite_score >= 50) as gte50,
                   MAX(composite_score) as max_score
            FROM oi_score_matrix {tfwd_where_sql}
            GROUP BY fwd_ret ORDER BY AVG(composite_score) DESC
        """, *tfwd_params)

    def _row(r, key):
        return {key: r[key],
                "avg_score": round(float(r["avg_score"] or 0), 1),
                "std_score": round(float(r["std_score"] or 0), 1),
                "n": int(r["n"]), "gte50": int(r["gte50"]),
                "max_score": round(float(r["max_score"] or 0), 1)}

    return {
        "by_metric":     [_row(r, "metric")  for r in by_metric],
        "by_fwd":        [_row(r, "fwd_ret") for r in by_fwd],
        "by_ticker":     [_row(r, "ticker")  for r in by_ticker],
        "by_fwd_ticker": [_row(r, "fwd_ret") for r in by_fwd_ticker],
        "selected_metric":  metric,
        "selected_fwd_ret": fwd_ret,
        "selected_ticker":  ticker,
    }


@router.post("/run-batch-score")
async def trigger_batch_score(
    pool=Depends(get_pool),
    oi_pool=Depends(get_oi_pool),
):
    """Trigger a batch score run in the background."""
    from research.batch_score import get_progress, run_batch_score
    import asyncio

    progress = get_progress()
    if progress["running"]:
        return {"status": "already_running", "message": progress["message"]}

    asyncio.get_event_loop().create_task(run_batch_score(oi_pool, pool))

    return {"status": "started", "message": "Batch scoring started..."}


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
