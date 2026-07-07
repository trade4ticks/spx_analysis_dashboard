"""
Ticker Analysis page — single-ticker view.

Additive, self-contained router mounted at /api/ticker-analysis. Kept
separate from /api/factor-analysis so nothing on the universe-wide Factor
Analysis surface is disturbed (see ticker_analysis_build_brief.md §0).

Endpoints:
  GET /tickers  — ticker universe (same as Factor Analysis).
  GET /price    — full-history close + split markers for the price chart.
  GET /metric   — per-(ticker, metric, horizon) bundle driving one metric
                  pane: 20-bin IS stats + per-date value/return/bin series
                  + today's in-bin position. One payload so the stat strip,
                  bin-highlight, and value-over-time all recompute client-
                  side without further round-trips (brief §4, §7).

Bins are per-ticker, IN-SAMPLE (is_bins.bin20_{metric}) — decided with the
user: "extreme for this name," reusing the stored bins verbatim (no
re-binning). Horizon is any ret_*_fwd_* column.
"""
from collections import defaultdict

from fastapi import APIRouter, Depends, Query

from app.db import get_oi_pool

router = APIRouter()


async def _table_columns(conn, table: str) -> set:
    """Column-name set for a public table — used to whitelist identifiers
    before they are interpolated into SQL (asyncpg cannot parameterize
    column names, so this validation is the injection guard)."""
    rows = await conn.fetch(
        "SELECT column_name FROM information_schema.columns "
        "WHERE table_name = $1 AND table_schema = 'public'",
        table,
    )
    return {r["column_name"] for r in rows}


@router.get("/tickers")
async def list_tickers(pool=Depends(get_oi_pool)):
    """Distinct ticker universe from daily_features (OI DB).

    Mirrors GET /api/factor-analysis/tickers so this page's selector
    offers the identical universe. Returns [] when the OI pool is not
    configured, matching the Factor Analysis degradation behavior.
    """
    if not pool:
        return []
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT DISTINCT ticker FROM daily_features ORDER BY ticker"
        )
    return [r["ticker"] for r in rows]


@router.get("/price")
async def price(
    ticker: str = Query(...),
    pool=Depends(get_oi_pool),
):
    """Full-history close line + split markers (brief §5.1).

    Uses underlying_ohlc.close (unadjusted) and flags rows where
    `splits` is a real ratio (not 0 / 1) so the chart can mark them.
    """
    if not pool:
        return {"error": "OI database not configured"}
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT trade_date, close, splits FROM underlying_ohlc "
            "WHERE ticker = $1 AND close IS NOT NULL ORDER BY trade_date",
            ticker,
        )
    series = [
        {"date": str(r["trade_date"]), "close": round(float(r["close"]), 4)}
        for r in rows
    ]
    splits = [
        {"date": str(r["trade_date"]), "ratio": round(float(r["splits"]), 4)}
        for r in rows
        if r["splits"] is not None and float(r["splits"]) not in (0.0, 1.0)
    ]
    return {"ticker": ticker, "series": series, "splits": splits}


@router.get("/metric")
async def metric_panel(
    ticker: str = Query(...),
    metric: str = Query(...),
    horizon: str = Query(...),
    pool=Depends(get_oi_pool),
):
    """Per-metric bundle for one pane.

    Returns:
      bins:   [{bin, lo, hi, avg_ret, win_rate, n} | null] × 20  (IS)
              lo/hi = observed metric-value range of the bin (drives the
              today-marker frac and the value-over-time selected band).
      series: [{date, val, ret|null, bin}]  per date, chronological.
              `ret` is null for the most recent `horizon` sessions (no
              forward data yet) — kept so value/bin still render.
      today:  {date, value, bin, frac, percentile}  in-bin lean (§5.3).
    """
    if not pool:
        return {"error": "OI database not configured"}

    async with pool.acquire() as conn:
        df_cols = await _table_columns(conn, "daily_features")
        ib_cols = await _table_columns(conn, "is_bins")

        if metric not in df_cols:
            return {"error": f"Unknown metric column: {metric}"}
        if horizon not in df_cols:
            return {"error": f"Unknown horizon column: {horizon}"}
        bin_col = f"bin20_{metric}"
        if bin_col not in ib_cols:
            # Some metrics are null-by-design and carry no stored bins.
            return {"error": f"No stored IS bins for metric: {metric}",
                    "no_bins": True}

        # Identifiers below are whitelisted against the column sets above,
        # so the f-string interpolation is injection-safe.
        rows = await conn.fetch(
            f'SELECT df.trade_date AS d, df."{metric}" AS val, '
            f'       df."{horizon}" AS ret, ib."{bin_col}" AS bin '
            f'FROM daily_features df '
            f'JOIN is_bins ib '
            f'  ON ib.ticker = df.ticker AND ib.trade_date = df.trade_date '
            f'WHERE df.ticker = $1 AND ib."{bin_col}" > 0 '
            f'  AND df."{metric}" IS NOT NULL '
            f'ORDER BY df.trade_date',
            ticker,
        )

    if not rows:
        return {"error": f"No data for {ticker} / {metric}", "n": 0}

    series = []
    vals_by_bin: dict = defaultdict(list)   # bin -> [val]     (for lo/hi)
    rets_by_bin: dict = defaultdict(list)   # bin -> [ret]     (for stats)
    all_vals: list = []

    for r in rows:
        b = int(r["bin"])
        v = float(r["val"])
        ret = None if r["ret"] is None else float(r["ret"])
        series.append({
            "date": str(r["d"]),
            "val":  round(v, 6),
            "ret":  (round(ret, 6) if ret is not None else None),
            "bin":  b,
        })
        vals_by_bin[b].append(v)
        all_vals.append(v)
        if ret is not None:
            rets_by_bin[b].append(ret)

    bins: list = []
    for b in range(1, 21):
        vals = vals_by_bin.get(b)
        if not vals:
            bins.append(None)
            continue
        rets = rets_by_bin.get(b, [])
        if rets:
            avg = round(sum(rets) / len(rets), 6)
            wr = round(sum(1 for x in rets if x > 0) / len(rets), 4)
        else:
            avg = None
            wr = None
        bins.append({
            "bin": b,
            "lo": round(min(vals), 6),
            "hi": round(max(vals), 6),
            "avg_ret": avg,
            "win_rate": wr,
            "n": len(rets),
        })

    # Today — in-bin lean (§5.3)
    last = series[-1]
    tbin = last["bin"]
    tval = last["val"]
    bmeta = bins[tbin - 1]
    if bmeta and bmeta["hi"] > bmeta["lo"]:
        frac = (tval - bmeta["lo"]) / (bmeta["hi"] - bmeta["lo"])
        frac = max(0.0, min(1.0, frac))
    else:
        frac = 0.5
    pct = round(sum(1 for v in all_vals if v <= tval) / len(all_vals) * 100, 1)
    today = {
        "date": last["date"],
        "value": round(tval, 6),
        "bin": tbin,
        "frac": round(frac, 4),
        "percentile": pct,
    }

    return {
        "ticker": ticker,
        "metric": metric,
        "horizon": horizon,
        "n": len(series),
        "bins": bins,
        "series": series,
        "today": today,
    }
