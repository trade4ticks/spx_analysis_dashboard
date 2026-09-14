"""Data loader for backtest trade data files.

COPIED from Options-Backtest-Dashboard/utils/data_loader.py (211198c), the
Dash app this page replaces. The parsers encode vendor edge cases -- tab vs
comma CSV, StrategyName vs BacktestName, pos_pnl vs pos_realized_pnl -- so the
body is kept as it was. Two deliberate removals, nothing else:

  * `from utils.db import is_postgres` -- the dual Postgres/SQLite layer is
    not ported; this app has one connection layer (app/db.py).
  * join_sharpetwo_data_wrapper() -- SharpTwo is dropped scope. Leaving the
    join in place would feed all-NaN vrp/iv_rv/vov columns into the binning
    and produce plausible garbage rather than an error.

join_market_data() stays here, where the source put it.
"""

import io
import json
from collections import defaultdict
from datetime import datetime

import pandas as pd


# Column mapping for Option Omega CSV format
COLUMN_MAPPING = {
    "Date Opened": "date_opened",
    "Time Opened": "time_opened",
    "Opening Price": "spx_open_price",
    "Legs": "legs",
    "Premium": "premium",
    "Closing Price": "spx_close_price",
    "Date Closed": "date_closed",
    "Time Closed": "time_closed",
    "Avg. Closing Cost": "avg_closing_cost",
    "Reason For Close": "exit_reason",
    "P/L": "pnl",
    "P/L %": "pnl_pct",
    "No. of Contracts": "contracts",
    "Funds at Close": "funds_at_close",
    "Margin Req.": "margin_req",
    "Strategy": "strategy",
    "Opening Commissions + Fees": "open_fees",
    "Closing Commissions + Fees": "close_fees",
    "Opening Short/Long Ratio": "open_sl_ratio",
    "Closing Short/Long Ratio": "close_sl_ratio",
    "Gap": "csv_gap",  # Keep original CSV gap as csv_gap, we'll calculate our own
    "Movement": "movement",
    "Max Profit": "max_profit",
    "Max Loss": "max_loss",
}

# Exit reason mapping for Mesosim ExitSignal messages
MESOSIM_EXIT_REASONS = {
    "profit target": "Profit Target",
    "stop loss": "Stop Loss",
    "max time in trade": "Max DIT",
    "adjustment count": "Max Adjustments",
}


def _decode_upload_contents(contents: str | bytes) -> str:
    """Decode base64-encoded Dash upload contents to string."""
    if isinstance(contents, str) and "," in contents and contents.startswith("data:"):
        import base64
        content_type, content_string = contents.split(",")
        contents = base64.b64decode(content_string)

    if isinstance(contents, bytes):
        contents = contents.decode("utf-8")

    return contents


def parse_csv(contents: str | bytes, filename: str = "") -> pd.DataFrame:
    """
    Parse an Option Omega CSV file.

    Args:
        contents: CSV file contents (string or bytes)
        filename: Original filename (for error messages)

    Returns:
        DataFrame with standardized column names and parsed dates
    """
    contents = _decode_upload_contents(contents)

    # Read CSV
    df = pd.read_csv(io.StringIO(contents), sep="\t")

    # If tab-separated didn't work, try comma
    if len(df.columns) <= 1:
        df = pd.read_csv(io.StringIO(contents), sep=",")

    # Rename columns
    df = df.rename(columns=COLUMN_MAPPING)

    # Parse dates - use flexible parsing to handle various formats
    df["date_opened"] = pd.to_datetime(df["date_opened"], format="mixed", dayfirst=False)
    df["date_closed"] = pd.to_datetime(df["date_closed"], format="mixed", dayfirst=False)

    # Calculate derived fields
    df["day_of_week"] = df["date_opened"].dt.dayofweek
    df["day_name"] = df["date_opened"].dt.day_name()
    df["days_in_trade"] = (df["date_closed"] - df["date_opened"]).dt.days
    df["year"] = df["date_opened"].dt.year

    # Win/loss flag
    df["is_win"] = df["pnl"] > 0

    # Ensure numeric columns are numeric
    numeric_cols = [
        "spx_open_price", "premium", "spx_close_price", "avg_closing_cost",
        "pnl", "pnl_pct", "contracts", "funds_at_close", "margin_req",
        "open_fees", "close_fees", "open_sl_ratio", "close_sl_ratio",
        "csv_gap", "movement", "max_profit", "max_loss"
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Use CSV gap as default 'gap' column (will be overwritten if market data is joined)
    if "csv_gap" in df.columns:
        df["gap"] = df["csv_gap"]

    return df


def _simplify_exit_reason(message: str) -> str:
    """Convert Mesosim ExitSignal message to a simplified exit reason."""
    msg_lower = message.lower()
    for pattern, label in MESOSIM_EXIT_REASONS.items():
        if pattern in msg_lower:
            return label
    return message


def parse_mesosim_json(contents: str | bytes, filename: str = "") -> pd.DataFrame:
    """
    Parse a DeltaRay Mesosim events JSON file.

    Extracts per-position trade data by grouping events by PositionId
    and extracting entry/exit information.

    Args:
        contents: JSON file contents (string or bytes)
        filename: Original filename (for error messages)

    Returns:
        DataFrame with standardized column names and parsed dates
    """
    contents = _decode_upload_contents(contents)
    events = json.loads(contents)

    # Group events by PositionId
    positions = defaultdict(list)
    strategy = None

    for event in events:
        # Extract strategy name from Start event
        if event.get("EventType") == "Start" and strategy is None:
            msg = event.get("Message", "")
            # Parse: "... StrategyName: allantis MesoSimVersion: ..."
            # Newer Mesosim versions use "BacktestName:" instead of "StrategyName:"
            for key in ("StrategyName:", "BacktestName:"):
                if key in msg:
                    parts = msg.split(key)
                    strategy = parts[1].split()[0].strip() if len(parts) > 1 else None
                    break

        pos_id = event.get("PositionId")
        if pos_id is not None:
            positions[pos_id].append(event)

    # Extract trade data from each position
    trades = []
    for pos_id, pos_events in positions.items():
        enter_event = None
        exit_event = None
        exit_signal = None
        entry_trades = []

        for event in pos_events:
            event_type = event.get("EventType")
            if event_type == "EnterPosition":
                enter_event = event
            elif event_type == "ExitPosition":
                exit_event = event
            elif event_type == "ExitSignal":
                exit_signal = event
            elif event_type == "EntryTrade":
                entry_trades.append(event)

        # Skip positions without both entry and exit
        if not enter_event or not exit_event:
            continue

        # Extract dates (date portion of SimTime)
        date_opened = pd.to_datetime(enter_event["SimTime"]).normalize()
        date_closed = pd.to_datetime(exit_event["SimTime"]).normalize()

        # PnL from exit position vars
        # Field name varies by Mesosim version: "pos_pnl" (newer) or "pos_realized_pnl" (older)
        exit_vars = exit_event.get("Vars") or {}
        pnl = exit_vars.get("pos_pnl", exit_vars.get("pos_realized_pnl", 0))

        # Margin from enter position vars
        # Field name varies: "pos_margin" (older) or fall back to "stop_loss" as proxy
        enter_vars = enter_event.get("Vars") or {}
        margin_req = enter_vars.get("pos_margin", enter_vars.get("stop_loss", 0))

        # Premium: sum of Price * Qty * Multiplier from initial EntryTrade events
        # Initial entry trades share the same SimTime as EnterPosition
        enter_time = enter_event["SimTime"]
        initial_entry_trades = [
            t for t in entry_trades if t["SimTime"] == enter_time
        ]
        premium = 0
        for trade in initial_entry_trades:
            te = trade.get("TradeEvent") or {}
            price = te.get("Price", 0)
            qty = te.get("Qty", 0)
            multiplier = (te.get("Contract") or {}).get("Multiplier", 100)
            premium += price * qty * multiplier

        # Exit reason from ExitSignal message
        exit_reason = "Unknown"
        if exit_signal:
            exit_reason = _simplify_exit_reason(exit_signal.get("Message", "Unknown"))

        # Leg count from initial entry trades
        legs = len(initial_entry_trades)

        trades.append({
            "date_opened": date_opened,
            "date_closed": date_closed,
            "pnl": pnl,
            "premium": premium,
            "exit_reason": exit_reason,
            "margin_req": margin_req,
            "strategy": strategy,
            "legs": legs,
        })

    if not trades:
        raise ValueError(f"No complete positions found in {filename}")

    df = pd.DataFrame(trades)

    # Calculate derived fields (same as CSV path)
    df["days_in_trade"] = (df["date_closed"] - df["date_opened"]).dt.days
    df["day_of_week"] = df["date_opened"].dt.dayofweek
    df["day_name"] = df["date_opened"].dt.day_name()
    df["year"] = df["date_opened"].dt.year
    df["is_win"] = df["pnl"] > 0

    return df


def parse_upload(contents: str | bytes, filename: str = "") -> pd.DataFrame:
    """
    Auto-detect file type and parse accordingly.

    Args:
        contents: File contents (string or bytes)
        filename: Original filename

    Returns:
        DataFrame with standardized column names and parsed dates
    """
    if filename.lower().endswith(".json"):
        return parse_mesosim_json(contents, filename)
    elif filename.lower().endswith(".csv"):
        return parse_csv(contents, filename)
    else:
        raise ValueError(f"Unsupported file type: {filename}. Please upload a CSV or JSON file.")


def get_date_range(df: pd.DataFrame) -> tuple[datetime, datetime]:
    """Get the min and max dates from the trade data."""
    min_date = df["date_opened"].min()
    max_date = df["date_closed"].max()
    return min_date.to_pydatetime(), max_date.to_pydatetime()


def join_market_data(trades_df: pd.DataFrame, market_df: pd.DataFrame) -> pd.DataFrame:
    """
    Join trade data with market data based on trade open date.

    Adds VIX, VIX3M, VIX9D levels at trade entry, overnight gaps, and ratios.
    """
    if market_df.empty:
        return trades_df

    # Create a date column for joining (just the date part)
    trades_df = trades_df.copy()
    trades_df["join_date"] = trades_df["date_opened"].dt.date

    market_df = market_df.copy()
    market_df["join_date"] = market_df["date"].dt.date

    # Sort market data by date for shift calculations
    market_df = market_df.sort_values("date").reset_index(drop=True)

    # Calculate SPX overnight gap (from market data, not CSV)
    if "SPX_open" in market_df.columns and "SPX_close" in market_df.columns:
        market_df["spx_prev_close"] = market_df["SPX_close"].shift(1)
        market_df["spx_overnight_gap"] = (
            (market_df["SPX_open"] - market_df["spx_prev_close"])
            / market_df["spx_prev_close"] * 100
        )

    # Calculate VIX overnight gap
    if "VIX_open" in market_df.columns and "VIX_close" in market_df.columns:
        market_df["vix_prev_close"] = market_df["VIX_close"].shift(1)
        market_df["vix_overnight_gap"] = (
            (market_df["VIX_open"] - market_df["vix_prev_close"])
            / market_df["vix_prev_close"] * 100
        )

    # Select relevant market data columns
    market_cols = ["join_date"]
    for prefix in ["VIX", "VIX3M", "VIX9D", "SPX"]:
        for suffix in ["open", "high", "low", "close"]:
            col = f"{prefix}_{suffix}"
            if col in market_df.columns:
                market_cols.append(col)

    # Add calculated gap columns
    if "spx_overnight_gap" in market_df.columns:
        market_cols.append("spx_overnight_gap")
    if "vix_overnight_gap" in market_df.columns:
        market_cols.append("vix_overnight_gap")

    market_subset = market_df[market_cols].drop_duplicates(subset=["join_date"])

    # Merge
    merged = trades_df.merge(market_subset, on="join_date", how="left")
    merged = merged.drop(columns=["join_date"])

    # Calculate additional metrics
    if "VIX_close" in merged.columns:
        merged["vix_level"] = merged["VIX_close"]
    if "VIX3M_close" in merged.columns:
        merged["vix3m_level"] = merged["VIX3M_close"]
    if "VIX9D_close" in merged.columns:
        merged["vix9d_level"] = merged["VIX9D_close"]

    # VIX3M/VIX ratio (inverted from original VIX/VIX3M)
    # Values > 1 indicate contango, < 1 indicate backwardation
    if "VIX_close" in merged.columns and "VIX3M_close" in merged.columns:
        merged["vix3m_vix_ratio"] = merged["VIX3M_close"] / merged["VIX_close"]

    # VIX/VIX9D ratio
    if "VIX_close" in merged.columns and "VIX9D_close" in merged.columns:
        merged["vix_vix9d_ratio"] = merged["VIX_close"] / merged["VIX9D_close"]

    # Use calculated SPX gap as the main "gap" column for filtering
    # Fall back to original CSV gap if no market data
    if "spx_overnight_gap" in merged.columns:
        merged["gap"] = merged["spx_overnight_gap"]
    elif "csv_gap" in merged.columns:
        merged["gap"] = merged["csv_gap"]

    return merged


def validate_data(df: pd.DataFrame) -> tuple[bool, str]:
    """
    Validate that the uploaded data has required columns.

    Returns (is_valid, error_message)
    """
    required_cols = ["date_opened", "date_closed", "pnl"]
    missing = [col for col in required_cols if col not in df.columns]

    if missing:
        return False, f"Missing required columns: {', '.join(missing)}"

    if df.empty:
        return False, "File contains no trade data"

    return True, ""

