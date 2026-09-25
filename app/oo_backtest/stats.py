"""Statistics calculations for the dashboard.

Copied from Options-Backtest-Dashboard/utils/stats.py (211198c), with ONE
deliberate change since: max drawdown is evaluated at day end rather than
per trade (see below). The same change was made in the source app, so the
two still agree -- but this is no longer a verbatim copy.
"""

import numpy as np
import pandas as pd


def calculate_stats(df: pd.DataFrame) -> dict:
    """
    Calculate summary statistics for the trade data.

    Returns dict with:
        - num_trades: total number of trades
        - win_pct: win percentage
        - avg_pnl: average P/L per trade
        - total_pnl: total P/L
        - avg_days_in_trade: average holding period
        - max_drawdown: maximum drawdown
        - avg_win_pnl: average winning trade P/L
        - avg_loss_pnl: average losing trade P/L
        - max_winner: largest winning trade
        - max_loser: largest losing trade
    """
    if df.empty:
        return {
            "num_trades": 0,
            "win_pct": 0.0,
            "avg_pnl": 0.0,
            "total_pnl": 0.0,
            "avg_days_in_trade": 0.0,
            "max_drawdown": 0.0,
            "max_drawdown_pct": 0.0,
            "avg_win_pnl": 0.0,
            "avg_loss_pnl": 0.0,
            "max_winner": 0.0,
            "max_loser": 0.0,
        }

    num_trades = len(df)
    wins = df[df["pnl"] > 0]
    losses = df[df["pnl"] <= 0]

    win_pct = (len(wins) / num_trades) * 100 if num_trades > 0 else 0

    avg_pnl = df["pnl"].mean()
    total_pnl = df["pnl"].sum()

    # Average days in trade
    if "days_in_trade" in df.columns:
        avg_days_in_trade = df["days_in_trade"].mean()
    else:
        avg_days_in_trade = 0.0

    # Calculate max drawdown, AT DAY END ONLY.
    #
    # Two trades closing on one day at -5,000 and +5,000 are not a 5,000
    # drawdown -- the other position was open and offsetting, and only the
    # day's net was ever at risk. Summing per close date before the cumsum
    # also removes the tie-break: a day's net does not depend on the order
    # its trades are summed in, so this no longer needs a stable sort.
    daily_pnl = df.groupby("date_closed")["pnl"].sum().sort_index()
    cumulative = daily_pnl.cumsum()
    peak = cumulative.cummax()
    drawdown = cumulative - peak
    max_drawdown = drawdown.min()

    # Max drawdown percentage
    if peak.max() > 0:
        # Find the drawdown at each point relative to the peak at that point
        drawdown_pct = np.where(peak != 0, (drawdown / peak) * 100, 0)
        max_drawdown_pct = np.min(drawdown_pct)
    else:
        max_drawdown_pct = 0.0

    # Win/loss averages
    avg_win_pnl = wins["pnl"].mean() if len(wins) > 0 else 0.0
    avg_loss_pnl = losses["pnl"].mean() if len(losses) > 0 else 0.0

    # Max winner/loser
    max_winner = df["pnl"].max()
    max_loser = df["pnl"].min()

    return {
        "num_trades": num_trades,
        "win_pct": win_pct,
        "avg_pnl": avg_pnl,
        "total_pnl": total_pnl,
        "avg_days_in_trade": avg_days_in_trade,
        "max_drawdown": max_drawdown,
        "max_drawdown_pct": max_drawdown_pct,
        "avg_win_pnl": avg_win_pnl,
        "avg_loss_pnl": avg_loss_pnl,
        "max_winner": max_winner,
        "max_loser": max_loser,
    }


def format_currency(value: float) -> str:
    """Format a number as currency."""
    if value >= 0:
        return f"${value:,.2f}"
    else:
        return f"-${abs(value):,.2f}"


def format_percent(value: float) -> str:
    """Format a number as percentage."""
    return f"{value:.1f}%"


def format_number(value: float, decimals: int = 1) -> str:
    """Format a number with specified decimal places."""
    return f"{value:,.{decimals}f}"
