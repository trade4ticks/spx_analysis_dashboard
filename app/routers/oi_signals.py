"""OI Signals 'Today' page — daily signal dashboard for all tickers."""
import math
from collections import defaultdict

import numpy as np
from fastapi import APIRouter, Depends, Query

from app.db import get_oi_pool

router = APIRouter(tags=["oi_signals"])

_METRICS = [
    "zscore_oi_above_below_ratio_3m",
    "zscore_oi_weighted_strike_all_div_spot_3m",
    "d1_oi_weighted_strike_all_div_spot_change",
]

_OUTCOME = "ret_1d_fwd_oc"


@router.get("/data")
async def get_signal_data(pool=Depends(get_oi_pool)):
    """
    For every ticker: compute historical decile avg returns per metric,
    and where today's value falls (as percentile + decile bucket).
    Returns one payload for the entire page.
    """
    if pool is None:
        return {"error": "OI database not configured", "tickers": []}

    cols = list(dict.fromkeys(["trade_date"] + _METRICS + [_OUTCOME]))
    col_sql = ", ".join(cols)

    async with pool.acquire() as conn:
        rows = await conn.fetch(
            f"SELECT ticker, {col_sql} FROM daily_features ORDER BY ticker, trade_date"
        )

    # Group by ticker
    by_ticker: dict[str, list] = defaultdict(list)
    for r in rows:
        by_ticker[r["ticker"]].append(dict(r))

    result_tickers = []

    for ticker in sorted(by_ticker.keys()):
        t_rows = by_ticker[ticker]
        if len(t_rows) < 30:
            continue

        ticker_data = {"ticker": ticker, "n_days": len(t_rows), "metrics": {}}
        latest = t_rows[-1]

        for metric in _METRICS:
            # Get valid (metric, outcome) pairs
            valid = []
            for r in t_rows:
                mv = r.get(metric)
                ov = r.get(_OUTCOME)
                if mv is None or ov is None:
                    continue
                try:
                    mf, of = float(mv), float(ov)
                except (ValueError, TypeError):
                    continue
                if math.isnan(mf) or math.isnan(of):
                    continue
                valid.append((mf, of))

            if len(valid) < 30:
                ticker_data["metrics"][metric] = {"error": "insufficient data", "n": len(valid)}
                continue

            # Sort by metric value, bucket into deciles
            sorted_pairs = sorted(valid, key=lambda p: p[0])
            n = len(sorted_pairs)
            n_buckets = 10
            buckets = [[] for _ in range(n_buckets)]
            bucket_ranges = [[] for _ in range(n_buckets)]
            for i, (mv, ov) in enumerate(sorted_pairs):
                b = min(int(i / n * n_buckets), n_buckets - 1)
                buckets[b].append(ov)
                bucket_ranges[b].append(mv)

            decile_stats = []
            for i, (rets, vals) in enumerate(zip(buckets, bucket_ranges)):
                if not rets:
                    decile_stats.append(None)
                    continue
                a = np.array(rets)
                decile_stats.append({
                    "bucket": i + 1,
                    "n": len(rets),
                    "avg_ret": round(float(a.mean()) * 100, 3),  # as percentage
                    "win_rate": round(float((a > 0).mean()) * 100, 1),
                    "min_val": round(float(min(vals)), 6),
                    "max_val": round(float(max(vals)), 6),
                })

            # Today's value and percentile
            today_val = latest.get(metric)
            today_pct = None
            today_decile = None
            if today_val is not None:
                try:
                    tv = float(today_val)
                    if not math.isnan(tv):
                        all_vals = [p[0] for p in sorted_pairs]
                        today_pct = round(sum(1 for v in all_vals if v <= tv) / len(all_vals) * 100, 1)
                        today_decile = min(int(today_pct / 10) + 1, 10)
                except (ValueError, TypeError):
                    pass

            ticker_data["metrics"][metric] = {
                "n": n,
                "deciles": decile_stats,
                "today_value": round(float(today_val), 6) if today_val is not None else None,
                "today_percentile": today_pct,
                "today_decile": today_decile,
            }

        ticker_data["latest_date"] = str(latest.get("trade_date", ""))
        result_tickers.append(ticker_data)

    return {
        "metrics": _METRICS,
        "outcome": _OUTCOME,
        "tickers": result_tickers,
    }


@router.get("/cooccurrence")
async def get_cooccurrence(
    metric: str = Query(_METRICS[0]),
    decile: int = Query(1, ge=1, le=10),
    pool=Depends(get_oi_pool),
):
    """
    Co-occurrence matrix: for each pair of tickers, what % of days where
    Ticker A is in the target decile does Ticker B ALSO land in the same decile?
    High overlap = correlated signals (redundant). Low overlap = independent.
    """
    if pool is None:
        return {"error": "OI database not configured"}

    async with pool.acquire() as conn:
        rows = await conn.fetch(
            f"SELECT ticker, trade_date, {metric} FROM daily_features "
            f"WHERE {metric} IS NOT NULL ORDER BY ticker, trade_date"
        )

    # Group by ticker, compute decile thresholds per ticker
    by_ticker: dict[str, list] = defaultdict(list)
    for r in rows:
        try:
            v = float(r[metric])
            if not math.isnan(v):
                by_ticker[r["ticker"]].append((r["trade_date"], v))
        except (ValueError, TypeError):
            continue

    # For each ticker, find which dates are in the target decile
    ticker_decile_dates: dict[str, set] = {}
    tickers = sorted(by_ticker.keys())

    for ticker in tickers:
        vals = by_ticker[ticker]
        if len(vals) < 30:
            continue
        sorted_vals = sorted(vals, key=lambda x: x[1])
        n = len(sorted_vals)
        # Get dates in the target decile
        dates_in_decile = set()
        for i, (d, v) in enumerate(sorted_vals):
            b = min(int(i / n * 10) + 1, 10)
            if b == decile:
                dates_in_decile.add(d)
        if dates_in_decile:
            ticker_decile_dates[ticker] = dates_in_decile

    # Build co-occurrence matrix
    active_tickers = sorted(ticker_decile_dates.keys())
    matrix = {}
    for ta in active_tickers:
        dates_a = ticker_decile_dates[ta]
        row = {}
        for tb in active_tickers:
            if ta == tb:
                row[tb] = 100.0
                continue
            dates_b = ticker_decile_dates[tb]
            overlap = len(dates_a & dates_b)
            pct = round(overlap / len(dates_a) * 100, 1) if dates_a else 0
            row[tb] = pct
        matrix[ta] = row

    return {
        "metric": metric,
        "decile": decile,
        "tickers": active_tickers,
        "matrix": matrix,
    }
