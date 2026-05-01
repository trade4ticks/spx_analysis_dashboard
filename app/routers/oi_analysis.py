"""OI Analysis workbench — interactive decile analytics for a single ticker/metric/outcome."""
import math
from collections import defaultdict
from typing import Optional

import numpy as np
from scipy import stats as sp_stats
from fastapi import APIRouter, Depends, Query

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

    # Fetch data with optional date range
    date_conditions = ""
    params = []
    p = 1
    if ticker != "ALL":
        date_conditions += f" AND ticker = ${p}"; params.append(ticker); p += 1
    if date_from:
        from datetime import date as _date
        date_conditions += f" AND trade_date >= ${p}"; params.append(_date.fromisoformat(date_from)); p += 1
    if date_to:
        from datetime import date as _date
        date_conditions += f" AND trade_date <= ${p}"; params.append(_date.fromisoformat(date_to)); p += 1

    # Check for spot price column (prefer spot_co, fallback to spot_close)
    async with pool.acquire() as conn:
        col_check = await conn.fetch(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_name = 'daily_features' AND column_name IN ('spot_co', 'spot_close')")
    spot_cols_found = {r["column_name"] for r in col_check}
    spot_col = "spot_co" if "spot_co" in spot_cols_found else ("spot_close" if "spot_close" in spot_cols_found else None)
    has_spot = spot_col is not None
    spot_select = f", {spot_col}" if has_spot else ""

    async with pool.acquire() as conn:
        rows = await conn.fetch(
            f"SELECT trade_date, {metric}, {outcome}{spot_select} FROM daily_features "
            f"WHERE {metric} IS NOT NULL AND {outcome} IS NOT NULL {date_conditions} "
            f"ORDER BY trade_date", *params)

    row_dicts = [dict(r) for r in rows]
    pairs = _clean_pairs(row_dicts, metric, outcome)
    n = len(pairs)
    if n < 30:
        return {"error": f"Insufficient data: {n} valid rows", "n": n}

    xa = np.array([p[0] for p in pairs])
    ya = np.array([p[1] for p in pairs])
    horizon = _parse_horizon(outcome)

    # Spot price time series (for equity overlay)
    spot_series = []
    if has_spot:
        for r in row_dicts:
            sv = r.get(spot_col)
            if sv is not None:
                try:
                    spot_series.append({"date": str(r["trade_date"]), "value": round(float(sv), 2)})
                except (ValueError, TypeError):
                    pass

    # ── Decile stats ─────────────────────────────────────────────────────
    buckets = _bucket_pairs(pairs, 10)
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
            "returns":  [round(float(y), 6) for y in ys],  # raw returns for boxplot
        })

    # ── Correlations ─────────────────────────────────────────────────────
    pr, pp = sp_stats.pearsonr(xa, ya)
    sr, sp_val = sp_stats.spearmanr(xa, ya)

    # ── Monotonicity ─────────────────────────────────────────────────────
    avgs = [d["avg_ret"] for d in decile_stats if d is not None]
    if len(avgs) >= 2:
        transitions = sum(1 for i in range(len(avgs)-1) if avgs[i+1] > avgs[i])
        mono_raw = transitions / (len(avgs)-1)
        monotonicity = round(abs(mono_raw - 0.5) * 2, 4)
    else:
        monotonicity = 0

    # ── Pattern classification ───────────────────────────────────────────
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

    # ── Yearly breakdown ─────────────────────────────────────────────────
    by_year = defaultdict(list)
    for x, y, d in pairs:
        yr = d.year if hasattr(d, 'year') else int(str(d)[:4])
        by_year[yr].append(y)

    yearly = []
    for yr in sorted(by_year):
        ys = np.array(by_year[yr])
        yearly.append({
            "year":     yr,
            "n":        len(ys),
            "avg_ret":  round(float(ys.mean()), 6),
            "win_rate": round(float((ys > 0).mean()), 4),
        })

    # ── Equity curve (both modes) ────────────────────────────────────────
    def _equity_for_decile(decile_idx, mode="concurrent"):
        bucket = buckets[decile_idx] if 0 <= decile_idx < len(buckets) else []
        if not bucket:
            return {"points": [], "n_trades": 0}
        sorted_trades = sorted(bucket, key=lambda p: p[2])

        if mode == "non_overlapping":
            trades, last_date = [], None
            for x, y, d in sorted_trades:
                dd = d.date() if hasattr(d, 'date') else d
                if last_date is None or (dd - last_date).days >= horizon:
                    trades.append((dd, y))
                    last_date = dd
        else:
            trades = [(p[2], p[1]) for p in sorted_trades]

        cum = 0.0
        peak = 0.0
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
            "concurrent":     _equity_for_decile(i, "concurrent"),
            "non_overlapping": _equity_for_decile(i, "non_overlapping"),
        }

    # ── Yearly consistency (top vs bottom) ───────────────────────────────
    yearly_consistency = []
    years_top_wins = 0
    for yr in sorted(by_year):
        yr_pairs = [(x, y, d) for x, y, d in pairs
                    if (d.year if hasattr(d, 'year') else int(str(d)[:4])) == yr]
        if len(yr_pairs) < 30:
            continue
        yr_buckets = _bucket_pairs(yr_pairs, 10)
        top_ys = [p[1] for p in yr_buckets[9]] if yr_buckets[9] else []
        bot_ys = [p[1] for p in yr_buckets[0]] if yr_buckets[0] else []
        t_avg = float(np.mean(top_ys)) if top_ys else 0
        b_avg = float(np.mean(bot_ys)) if bot_ys else 0
        top_beats = t_avg > b_avg
        if top_beats:
            years_top_wins += 1
        yearly_consistency.append({
            "year": yr, "top_avg": round(t_avg, 6), "bot_avg": round(b_avg, 6),
            "top_n": len(top_ys), "bot_n": len(bot_ys), "top_beats": top_beats,
        })

    n_years = len(yearly_consistency)
    consistency_pct = round(years_top_wins / n_years * 100, 1) if n_years else None

    # ── Half-sample stability ────────────────────────────────────────────
    mid = n // 2
    h1 = _bucket_pairs(sorted(pairs, key=lambda p: p[2])[:mid], 10)
    h2 = _bucket_pairs(sorted(pairs, key=lambda p: p[2])[mid:], 10)
    h1_spread = (np.mean([p[1] for p in h1[9]]) - np.mean([p[1] for p in h1[0]])) if h1[0] and h1[9] else 0
    h2_spread = (np.mean([p[1] for p in h2[9]]) - np.mean([p[1] for p in h2[0]])) if h2[0] and h2[9] else 0
    half_stable = (h1_spread > 0 and h2_spread > 0) or (h1_spread < 0 and h2_spread < 0)

    # ── Concentration risk ───────────────────────────────────────────────
    yearly_spreads = {}
    for yc in yearly_consistency:
        yearly_spreads[yc["year"]] = yc["top_avg"] - yc["bot_avg"]
    total_abs = sum(abs(v) for v in yearly_spreads.values())
    concentration = round(max(abs(v) for v in yearly_spreads.values()) / total_abs, 4) if total_abs > 0 else 1.0

    # ── Composite score ──────────────────────────────────────────────────
    best_sharpe = max(abs(d["sharpe"]) for d in decile_stats if d) if decile_stats else 0
    c_rank = min(abs(float(sr)) / 0.20, 1.0)
    c_mono = monotonicity
    c_consist = (consistency_pct / 100.0) if consistency_pct else 0
    c_half = 1.0 if half_stable else 0
    c_conc = max(0, 1.0 - concentration)
    c_sharpe = min(best_sharpe / 0.5, 1.0)
    c_sample = min(n / 1000, 0.5)
    composite = round((c_rank + c_mono + c_consist + c_half + c_conc + c_sharpe + c_sample) / 6.5 * 100, 1)

    # ── Rolling correlation (252-day window) ───────────────────────────
    sorted_by_date = sorted(pairs, key=lambda p: p[2])
    rolling_window = 252
    rolling_corr = []
    if len(sorted_by_date) > rolling_window:
        for end in range(rolling_window, len(sorted_by_date)):
            window = sorted_by_date[end - rolling_window:end]
            wx = np.array([p[0] for p in window])
            wy = np.array([p[1] for p in window])
            if wx.std() > 0 and wy.std() > 0:
                rc, _ = sp_stats.spearmanr(wx, wy)
                rolling_corr.append({
                    "date": str(window[-1][2]),
                    "spearman": round(float(rc), 4),
                })

    # ── Trade calendar (month × year avg return per decile) ──────────────
    # Pre-assign decile to each pair efficiently
    sorted_by_x = sorted(range(len(pairs)), key=lambda i: pairs[i][0])
    decile_map = {}
    for rank, idx in enumerate(sorted_by_x):
        decile_map[idx] = min(int(rank / len(pairs) * 10) + 1, 10)

    trade_calendar = []
    dow_data = []  # day-of-week returns
    for idx, (x, y, d) in enumerate(pairs):
        yr = d.year if hasattr(d, 'year') else int(str(d)[:4])
        mo = d.month if hasattr(d, 'month') else int(str(d)[5:7])
        # Day of week: 0=Mon ... 4=Fri
        dow = d.weekday() if hasattr(d, 'weekday') else 0
        trade_calendar.append({"year": yr, "month": mo, "ret": round(y, 6), "decile": decile_map[idx]})
        dow_data.append({"dow": dow, "ret": round(y, 6), "decile": decile_map[idx]})

    # ── Today's value ────────────────────────────────────────────────────
    today_val = pairs[-1][0] if pairs else None
    today_pct = None
    today_decile = None
    if today_val is not None:
        all_x = sorted(p[0] for p in pairs)
        today_pct = round(sum(1 for v in all_x if v <= today_val) / len(all_x) * 100, 1)
        today_decile = min(int(today_pct / 10) + 1, 10)

    return {
        "ticker":      ticker,
        "metric":      metric,
        "outcome":     outcome,
        "n":           n,
        "horizon":     horizon,

        # Stats
        "pearson_r":    round(float(pr), 4),
        "pearson_p":    round(float(pp), 6),
        "spearman_r":   round(float(sr), 4),
        "monotonicity": monotonicity,
        "pattern":      pattern,
        "composite_score": composite,
        "consistency_pct": consistency_pct,
        "concentration_risk": concentration,
        "half_sample_stable": bool(half_stable),

        # Decile data
        "decile_stats":    decile_stats,
        "equity_by_decile": equity_by_decile,

        # Time series
        "yearly":              yearly,
        "yearly_consistency":  yearly_consistency,
        "rolling_corr":        rolling_corr,
        "trade_calendar":      trade_calendar,
        "dow_data":            dow_data,
        "spot_series":         spot_series,

        # Today
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

    where = ["composite_score >= $1"]
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
               best_sharpe, scanned_at
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
            "SELECT DISTINCT metric FROM oi_score_matrix ORDER BY metric")]
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
):
    """Aggregated score stats: avg/stddev by metric, and by fwd_ret for a selected metric."""
    from research.batch_score import ensure_table
    await ensure_table(pool)

    async with pool.acquire() as conn:
        # Avg score by metric (all metrics)
        by_metric = await conn.fetch("""
            SELECT metric,
                   AVG(composite_score) as avg_score,
                   STDDEV(composite_score) as std_score,
                   COUNT(*) as n,
                   COUNT(*) FILTER (WHERE composite_score >= 50) as gte50,
                   MAX(composite_score) as max_score
            FROM oi_score_matrix
            GROUP BY metric
            ORDER BY AVG(composite_score) DESC
        """)

        # By fwd_ret for selected metric (or all if none selected)
        if metric:
            by_fwd = await conn.fetch("""
                SELECT fwd_ret,
                       AVG(composite_score) as avg_score,
                       STDDEV(composite_score) as std_score,
                       COUNT(*) as n,
                       COUNT(*) FILTER (WHERE composite_score >= 50) as gte50,
                       MAX(composite_score) as max_score
                FROM oi_score_matrix
                WHERE metric = $1
                GROUP BY fwd_ret
                ORDER BY AVG(composite_score) DESC
            """, metric)
        else:
            by_fwd = await conn.fetch("""
                SELECT fwd_ret,
                       AVG(composite_score) as avg_score,
                       STDDEV(composite_score) as std_score,
                       COUNT(*) as n,
                       COUNT(*) FILTER (WHERE composite_score >= 50) as gte50,
                       MAX(composite_score) as max_score
                FROM oi_score_matrix
                GROUP BY fwd_ret
                ORDER BY AVG(composite_score) DESC
            """)

    return {
        "by_metric": [
            {"metric": r["metric"],
             "avg_score": round(float(r["avg_score"] or 0), 1),
             "std_score": round(float(r["std_score"] or 0), 1),
             "n": int(r["n"]),
             "gte50": int(r["gte50"]),
             "max_score": round(float(r["max_score"] or 0), 1)}
            for r in by_metric
        ],
        "by_fwd": [
            {"fwd_ret": r["fwd_ret"],
             "avg_score": round(float(r["avg_score"] or 0), 1),
             "std_score": round(float(r["std_score"] or 0), 1),
             "n": int(r["n"]),
             "gte50": int(r["gte50"]),
             "max_score": round(float(r["max_score"] or 0), 1)}
            for r in by_fwd
        ],
        "selected_metric": metric,
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
