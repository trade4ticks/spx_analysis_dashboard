"""
Analysis building blocks. Each function takes an asyncpg connection,
queries the source table, computes statistics in Python, and returns a dict.
These are database-agnostic — pass the appropriate pool connection.
"""
import re
from typing import Optional
import numpy as np
from scipy import stats
import asyncpg


# ── Helpers ───────────────────────────────────────────────────────────────────

def _parse_horizon(col_name: str) -> int:
    """Extract holding period in days from a column name like ret_5d_fwd."""
    m = re.search(r'(\d+)d', col_name)
    return int(m.group(1)) if m else 1


def _build_where(ticker, date_from, date_to, extra_not_null=None, start_param=1):
    """Build a WHERE clause and params list. Returns (clause_str, params, next_param_idx)."""
    conditions = []
    params = []
    p = start_param

    if extra_not_null:
        conditions.extend(f"{c} IS NOT NULL" for c in extra_not_null)

    if ticker:
        conditions.append(f"ticker = ${p}")
        params.append(ticker)
        p += 1
    if date_from:
        conditions.append(f"trade_date >= ${p}")
        params.append(date_from)
        p += 1
    if date_to:
        conditions.append(f"trade_date <= ${p}")
        params.append(date_to)
        p += 1

    clause = ("WHERE " + " AND ".join(conditions)) if conditions else ""
    return clause, params, p


# ── Building blocks ───────────────────────────────────────────────────────────

async def compute_correlation(
    conn: asyncpg.Connection,
    table: str,
    x_col: str,
    y_col: str,
    ticker: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
) -> dict:
    """Pearson and Spearman correlation with p-values."""
    where, params, _ = _build_where(ticker, date_from, date_to, extra_not_null=[x_col, y_col])
    sql = f"SELECT {x_col}, {y_col} FROM {table} {where} ORDER BY trade_date"
    rows = await conn.fetch(sql, *params)

    if len(rows) < 10:
        return {"error": "insufficient data", "n": len(rows),
                "x_col": x_col, "y_col": y_col, "ticker": ticker}

    xa = np.array([float(r[0]) for r in rows])
    ya = np.array([float(r[1]) for r in rows])
    pr, pp = stats.pearsonr(xa, ya)
    sr, sp = stats.spearmanr(xa, ya)

    return {
        "pearson_r":  round(float(pr), 4),
        "pearson_p":  round(float(pp), 6),
        "spearman_r": round(float(sr), 4),
        "spearman_p": round(float(sp), 6),
        "n":          len(rows),
        "x_col":      x_col,
        "y_col":      y_col,
        "ticker":     ticker,
    }


async def compute_decile_analysis(
    conn: asyncpg.Connection,
    table: str,
    feature_col: str,
    outcome_col: str,
    ticker: Optional[str] = None,
    n_deciles: int = 10,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
) -> dict:
    """Global decile stats: avg/median return, win rate, sample count, std dev per decile."""
    where, params, _ = _build_where(
        ticker, date_from, date_to, extra_not_null=[feature_col, outcome_col]
    )
    sql = f"""
    WITH ranked AS (
        SELECT {outcome_col},
               NTILE({n_deciles}) OVER (ORDER BY {feature_col}) AS decile
        FROM {table} {where}
    )
    SELECT
        decile,
        COUNT(*)                                                              AS n,
        ROUND(AVG({outcome_col})::numeric, 6)                                AS avg_ret,
        ROUND(PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY {outcome_col})::numeric, 6) AS med_ret,
        ROUND((SUM(CASE WHEN {outcome_col} > 0 THEN 1 ELSE 0 END)::float
               / NULLIF(COUNT(*),0))::numeric, 4)                            AS win_rate,
        ROUND(STDDEV({outcome_col})::numeric, 6)                             AS std_dev
    FROM ranked
    GROUP BY decile
    ORDER BY decile
    """
    rows = await conn.fetch(sql, *params)
    if not rows:
        return {"error": "no data", "deciles": [],
                "feature_col": feature_col, "outcome_col": outcome_col, "ticker": ticker}

    deciles = [
        {
            "decile":        int(r["decile"]),
            "n":             int(r["n"]),
            "avg_ret":       float(r["avg_ret"])  if r["avg_ret"]  is not None else None,
            "med_ret":       float(r["med_ret"])  if r["med_ret"]  is not None else None,
            "win_rate":      float(r["win_rate"]) if r["win_rate"] is not None else None,
            "std_dev":       float(r["std_dev"])  if r["std_dev"]  is not None else None,
        }
        for r in rows
    ]

    top_avg = deciles[-1]["avg_ret"]
    bot_avg = deciles[0]["avg_ret"]
    spread = round(top_avg - bot_avg, 6) if top_avg is not None and bot_avg is not None else None

    return {
        "deciles":           deciles,
        "top_bottom_spread": spread,
        "n_deciles":         n_deciles,
        "feature_col":       feature_col,
        "outcome_col":       outcome_col,
        "ticker":            ticker,
    }


async def compute_yearly_consistency(
    conn: asyncpg.Connection,
    table: str,
    feature_col: str,
    outcome_col: str,
    ticker: Optional[str] = None,
    n_deciles: int = 10,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
) -> dict:
    """Year-by-year: does the top decile beat the bottom decile?"""
    where, params, _ = _build_where(
        ticker, date_from, date_to, extra_not_null=[feature_col, outcome_col]
    )
    sql = f"""
    WITH ranked AS (
        SELECT EXTRACT(YEAR FROM trade_date)::int AS yr,
               {outcome_col},
               NTILE({n_deciles}) OVER (PARTITION BY EXTRACT(YEAR FROM trade_date)
                                        ORDER BY {feature_col}) AS decile
        FROM {table} {where}
    )
    SELECT yr,
           AVG(CASE WHEN decile = {n_deciles} THEN {outcome_col} END) AS top_avg,
           AVG(CASE WHEN decile = 1           THEN {outcome_col} END) AS bot_avg,
           COUNT(CASE WHEN decile = {n_deciles} THEN 1 END)           AS top_n,
           COUNT(CASE WHEN decile = 1           THEN 1 END)           AS bot_n
    FROM ranked
    GROUP BY yr ORDER BY yr
    """
    rows = await conn.fetch(sql, *params)
    years = []
    wins = 0
    for r in rows:
        t = float(r["top_avg"]) if r["top_avg"] is not None else None
        b = float(r["bot_avg"]) if r["bot_avg"] is not None else None
        beats = t is not None and b is not None and t > b
        if beats:
            wins += 1
        years.append({
            "year":      r["yr"],
            "top_avg":   round(t, 6) if t is not None else None,
            "bot_avg":   round(b, 6) if b is not None else None,
            "top_n":     int(r["top_n"]),
            "bot_n":     int(r["bot_n"]),
            "top_beats": beats,
        })

    return {
        "years":           years,
        "consistency_pct": round(wins / len(years) * 100, 1) if years else None,
        "wins":            wins,
        "total_years":     len(years),
        "feature_col":     feature_col,
        "outcome_col":     outcome_col,
        "ticker":          ticker,
    }


async def compute_equity_curve(
    conn: asyncpg.Connection,
    table: str,
    feature_col: str,
    outcome_col: str,
    ticker: Optional[str] = None,
    which: str = "top",
    n_deciles: int = 10,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
) -> dict:
    """
    Non-overlapping equity curve for top or bottom decile.
    Skips entries within `horizon` calendar days of the previous trade.
    """
    horizon = _parse_horizon(outcome_col)
    where, params, _ = _build_where(
        ticker, date_from, date_to, extra_not_null=[feature_col, outcome_col]
    )
    target_decile = n_deciles if which == "top" else 1

    sql = f"""
    WITH ranked AS (
        SELECT trade_date, {outcome_col} AS ret,
               NTILE({n_deciles}) OVER (ORDER BY {feature_col}) AS decile
        FROM {table} {where}
    )
    SELECT trade_date, ret FROM ranked
    WHERE decile = {target_decile}
    ORDER BY trade_date
    """
    rows = await conn.fetch(sql, *params)
    if not rows:
        return {"error": "no data", "points": [],
                "feature_col": feature_col, "outcome_col": outcome_col,
                "ticker": ticker, "which": which}

    # Non-overlapping: enforce minimum gap of `horizon` calendar days between entries
    trades = []
    last_date = None
    for r in rows:
        d = r["trade_date"]
        if last_date is None or (d - last_date).days >= horizon:
            trades.append((d, float(r["ret"])))
            last_date = d

    equity = 1.0
    peak = 1.0
    max_dd = 0.0
    points = []
    for date, ret in trades:
        equity *= (1.0 + ret)
        peak = max(peak, equity)
        max_dd = min(max_dd, (equity - peak) / peak)
        points.append({"date": str(date), "value": round(equity, 6)})

    n = len(trades)
    avg_ret = sum(r for _, r in trades) / n if n else None
    win_rate = sum(1 for _, r in trades if r > 0) / n if n else None

    return {
        "points":       points,
        "n_trades":     n,
        "final_equity": round(equity, 4),
        "max_drawdown": round(max_dd, 4),
        "avg_ret":      round(avg_ret, 6) if avg_ret is not None else None,
        "win_rate":     round(win_rate, 4) if win_rate is not None else None,
        "horizon":      horizon,
        "which":        which,
        "feature_col":  feature_col,
        "outcome_col":  outcome_col,
        "ticker":       ticker,
    }


async def compute_scatter_sample(
    conn: asyncpg.Connection,
    table: str,
    x_col: str,
    y_col: str,
    ticker: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    limit: int = 2000,
) -> dict:
    """Returns a (date, x, y) sample for scatter visualization."""
    where, params, p = _build_where(
        ticker, date_from, date_to, extra_not_null=[x_col, y_col]
    )
    params.append(limit)
    sql = f"""
    SELECT trade_date, {x_col}, {y_col}
    FROM {table} {where}
    ORDER BY trade_date
    LIMIT ${p}
    """
    rows = await conn.fetch(sql, *params)
    return {
        "points": [{"date": str(r[0]), "x": float(r[1]), "y": float(r[2])} for r in rows],
        "n":      len(rows),
        "x_col":  x_col,
        "y_col":  y_col,
        "ticker": ticker,
    }


async def compute_regression(
    conn: asyncpg.Connection,
    table: str,
    x_cols: list[str],
    y_col: str,
    ticker: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
) -> dict:
    """OLS regression of y_col on x_cols. Returns coefficients and R²."""
    where, params, _ = _build_where(
        ticker, date_from, date_to, extra_not_null=x_cols + [y_col]
    )
    col_list = ", ".join(x_cols + [y_col])
    sql = f"SELECT {col_list} FROM {table} {where} ORDER BY trade_date"
    rows = await conn.fetch(sql, *params)

    if len(rows) < len(x_cols) + 5:
        return {"error": "insufficient data", "n": len(rows), "x_cols": x_cols, "y_col": y_col}

    X = np.array([[float(r[c]) for c in x_cols] for r in rows])
    y = np.array([float(r[y_col]) for r in rows])
    X_int = np.column_stack([np.ones(len(X)), X])
    coeffs, *_ = np.linalg.lstsq(X_int, y, rcond=None)

    y_pred = X_int @ coeffs
    ss_res = float(np.sum((y - y_pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    coeff_dict = {"intercept": round(float(coeffs[0]), 6)}
    for i, col in enumerate(x_cols):
        coeff_dict[col] = round(float(coeffs[i + 1]), 6)

    return {
        "coefficients": coeff_dict,
        "r_squared":    round(r2, 4),
        "n":            len(rows),
        "y_col":        y_col,
        "x_cols":       x_cols,
        "ticker":       ticker,
    }
