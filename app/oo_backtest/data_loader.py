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

join_market_data() is kept as the source had it but is NOT CALLED: it gives
every trade its entry date's daily CLOSE, which is lookahead. The page joins
market data through app/oo_backtest/market.py instead.

parse_mesosim_json() IS REWRITTEN (2026-09-14) against a field spec for the
MesoSim 3.1 event stream, checked on the real `allantis - v2` export and on
six MesoSim 2.13 exports. The source already reduced by PositionId, so the
trade count was right; what it got wrong, and what changed:

  * exit reason     free text with thresholds embedded ("Reached profit
                    target: 1291.521"); unknown messages passed through raw,
                    one category per threshold. Now: known patterns -> fixed
                    labels, anything else cut at the first colon and stripped
                    of parenthesised numbers. Where several signals fire on
                    the exit bar, EXIT_PRECEDENCE decides (price > adjustments
                    > time), and a case where that disagrees with file order
                    is reported.
  * P/L             Vars.pos_realized_pnl, falling back to pos_pnl only where a
                    version has no realized field (2.13 has none). The source
                    preferred pos_pnl. Disagreements are reported.
  * premium         Vars.entry_net_premium where present; the leg-fill sum
                    only as the fallback (2.13 has no entry_net_premium).
  * margin          Vars.pos_margin, else null. The source substituted
                    stop_loss, which is not a margin.
  * times           time_opened/time_closed kept. SimTime is exchange-local
                    Eastern -- EndOfDay fires at 13:00 on the early-close
                    days. The source normalised the times away.
  * strategy name   parsed by key, not first word: the source turned
                    "allantis - weekly entry - Mon" into "allantis".
  * open positions  still excluded, now REPORTED (df.attrs["parse_notes"]).
  * entry Vars      every numeric EnterPosition Var kept as entry_var_*,
                    since which ones matter later is not known yet.
  * MissingData     flagged where it lands on a position's entry or exit bar,
                    rather than trusting that fill silently.
"""

import io
import json
import re
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


# Exit reason labels for Mesosim ExitSignal messages. Matched as substrings of
# the lower-cased message; first match wins.
MESOSIM_EXIT_REASONS = [
    ("profit target", "Profit Target"),
    ("stop loss", "Stop Loss"),
    ("max time in trade", "Max Time in Trade"),
    ("adjustment count", "Max Adjustments"),
]

# Start event keys, in any order: "BacktestName: <x> StrategyName: <y>
# MesoSimVersion: <z>" (3.1) or "... TemplateName: <y> ..." (2.13).
_START_KEYS = ("BacktestName", "StrategyName", "TemplateName", "MesoSimVersion")
_START_KEY_RE = re.compile(r"\b(" + "|".join(_START_KEYS) + r"):\s*")


def _simplify_exit_reason(message: str) -> str:
    """A stable label for an ExitSignal message.

    Known reasons map to fixed labels. Anything else is cut at the first colon
    and stripped of parenthesised numbers, so a threshold in the text cannot
    make every trade its own category.
    """
    msg = (message or "").strip()
    low = msg.lower()
    for pattern, label in MESOSIM_EXIT_REASONS:
        if pattern in low:
            return label
    msg = msg.split(":", 1)[0]
    msg = re.sub(r"\([^)]*\d[^)]*\)", "", msg)
    msg = re.sub(r"\s+", " ", msg).strip(" ,.;")
    return msg or "Unknown"


# When more than one exit signal fires on a position's exit bar, a PRICE exit
# outranks an adjustment-count exit, which outranks the TIME limit. The
# order MesoSim writes them in is an implementation detail -- across 3.1 and
# 2.13 it happens to put time first and price second, and the P/L agreed with
# the price signal in all nine cases -- but two versions are already in play,
# so the rule is stated rather than inferred from file order. Lower wins.
EXIT_PRECEDENCE = {"Profit Target": 0, "Stop Loss": 0, "Max Adjustments": 1, "Max Time in Trade": 2}
_UNKNOWN_PRECEDENCE = 1


def _resolve_exit_reason(signals: list, exit_idx: int, exit_event: dict) -> tuple[str, dict]:
    """The exit reason for one position, and how it was decided.

    Candidates are the signals written before the ExitPosition on the exit's
    own bar; if none share that bar, the last signal before the exit. Among
    candidates the lowest EXIT_PRECEDENCE wins, ties to the later in file.
    `precedence_overrode_order` is True where that differs from simply taking
    the last signal -- the case that means MesoSim's ordering has changed.
    """
    prior = [(i, e) for i, e in signals if i < exit_idx]
    if not prior:
        return "Unknown", {"candidates": 0, "precedence_overrode_order": False, "last_in_file": None}
    same_bar = [(i, e) for i, e in prior if e.get("SimTime") == exit_event.get("SimTime")]
    candidates = same_bar or prior[-1:]
    labelled = [(i, _simplify_exit_reason(e.get("Message", ""))) for i, e in candidates]
    chosen = min(labelled, key=lambda x: (EXIT_PRECEDENCE.get(x[1], _UNKNOWN_PRECEDENCE), -x[0]))[1]
    last = _simplify_exit_reason(prior[-1][1].get("Message", ""))
    return chosen, {"candidates": len(candidates), "precedence_overrode_order": chosen != last,
                    "last_in_file": last}


def _leg_fill_premium(fills: list) -> float:
    """Sum of Price * Qty * Multiplier over the entry-bar fills (short legs carry
    a negative Qty). Rounded to cents; the float sum is not (10364.999999999993)."""
    total = 0.0
    for trade in fills:
        te = trade.get("TradeEvent") or {}
        multiplier = (te.get("Contract") or {}).get("Multiplier", 100)
        total += te.get("Price", 0) * te.get("Qty", 0) * multiplier
    return round(total, 2)


def _parse_start_message(msg: str) -> dict:
    """{"BacktestName": ..., "StrategyName": ...} from the Start event text."""
    out = {}
    matches = list(_START_KEY_RE.finditer(msg or ""))
    for i, m in enumerate(matches):
        stop = matches[i + 1].start() if i + 1 < len(matches) else len(msg)
        out[m.group(1)] = msg[m.end():stop].strip()
    return out


def parse_mesosim_json(contents: str | bytes, filename: str = "") -> pd.DataFrame:
    """
    Parse a DeltaRay Mesosim events JSON file.

    The file is an EVENT STREAM (EndOfDay, EntryTrade/ExitTrade leg fills,
    adjustments, ...), not a trade log. One trade = one PositionId with both an
    EnterPosition and an ExitPosition. Leg fills are read only as the premium
    fallback; counting them as trades would inflate the count ~16x.

    Returns a DataFrame with standardized column names. Parse diagnostics are
    in df.attrs["parse_notes"].
    """
    contents = _decode_upload_contents(contents)
    events = json.loads(contents)
    if not isinstance(events, list):
        raise ValueError(f"{filename}: expected a Mesosim events array")

    start = next((e for e in events if e.get("EventType") == "Start"), None)
    names = _parse_start_message(start.get("Message", "") if start else "")
    strategy = names.get("StrategyName") or names.get("TemplateName") or None
    backtest_name = names.get("BacktestName") or None

    # Index the events that matter by PositionId, remembering file position.
    enter, exit_ = {}, {}
    signals, entry_trades, missing_times = defaultdict(list), defaultdict(list), defaultdict(set)
    for idx, event in enumerate(events):
        et = event.get("EventType")
        pid = event.get("PositionId")
        if pid is None:
            continue
        if et == "EnterPosition":
            if pid in enter:
                raise ValueError(f"{filename}: PositionId {pid} is entered twice")
            enter[pid] = (idx, event)
        elif et == "ExitPosition":
            exit_[pid] = (idx, event)
        elif et == "ExitSignal":
            signals[pid].append((idx, event))
        elif et == "EntryTrade":
            entry_trades[pid].append(event)
        elif et == "MissingData":
            missing_times[pid].add(event.get("SimTime"))

    open_ids = sorted(pid for pid in enter if pid not in exit_)
    notes = {
        "open_positions": len(open_ids),
        "open_position_ids": open_ids,
        "multi_signal_positions": 0,
        "precedence_vs_order": [],
        "pnl_contradicts_reason": [],
        "premium_leg_mismatch": [],
        "pnl_mismatch": [],
        "missing_data_at_fill": [],
        "pnl_field": None,
        "premium_field": None,
        "backtest_name": backtest_name,
    }
    pnl_fields, premium_fields = set(), set()

    trades = []
    for pid, (_enter_idx, enter_event) in enter.items():
        if pid not in exit_:
            continue
        exit_idx, exit_event = exit_[pid]

        opened = pd.to_datetime(enter_event["SimTime"])
        closed = pd.to_datetime(exit_event["SimTime"])

        exit_vars = exit_event.get("Vars") or {}
        realized, pos_pnl = exit_vars.get("pos_realized_pnl"), exit_vars.get("pos_pnl")
        if realized is not None:
            pnl = realized
            pnl_fields.add("pos_realized_pnl")
            if pos_pnl is not None and abs(float(realized) - float(pos_pnl)) > 0.01:
                notes["pnl_mismatch"].append({"position_id": pid, "pos_realized_pnl": realized,
                                              "pos_pnl": pos_pnl})
        elif pos_pnl is not None:
            pnl = pos_pnl
            pnl_fields.add("pos_pnl")
        else:
            raise ValueError(f"{filename}: ExitPosition for PositionId {pid} carries no P/L")

        enter_vars = enter_event.get("Vars") or {}
        initial = [t for t in entry_trades[pid] if t.get("SimTime") == enter_event["SimTime"]]
        leg_sum = _leg_fill_premium(initial)
        if enter_vars.get("entry_net_premium") is not None:
            premium = enter_vars["entry_net_premium"]
            premium_fields.add("entry_net_premium")
            # 3.1 carries both, which validates the 2.13 fallback -- sign
            # included. A flipped premium reads as a different strategy, not
            # as a parse error, so any disagreement is reported.
            if initial and abs(leg_sum - float(premium)) > 0.01:
                notes["premium_leg_mismatch"].append(
                    {"position_id": pid, "entry_net_premium": premium, "leg_sum": leg_sum})
        else:
            premium = leg_sum
            premium_fields.add("leg fills")

        exit_reason, reason_notes = _resolve_exit_reason(signals.get(pid, []), exit_idx, exit_event)
        if reason_notes["candidates"] > 1:
            notes["multi_signal_positions"] += 1
        if reason_notes["precedence_overrode_order"]:
            notes["precedence_vs_order"].append({"position_id": pid, "chosen": exit_reason,
                                                 "last_in_file": reason_notes["last_in_file"]})
        if (exit_reason == "Profit Target" and pnl <= 0) or (exit_reason == "Stop Loss" and pnl >= 0):
            notes["pnl_contradicts_reason"].append({"position_id": pid, "reason": exit_reason, "pnl": pnl})

        at_fill = [k for k, ev in (("entry", enter_event), ("exit", exit_event))
                   if ev.get("SimTime") in missing_times.get(pid, ())]
        if at_fill:
            notes["missing_data_at_fill"].append({"position_id": pid, "at": at_fill})

        row = {
            "position_id": pid,
            "date_opened": opened.normalize(),
            "time_opened": opened.strftime("%H:%M:%S"),
            "date_closed": closed.normalize(),
            "time_closed": closed.strftime("%H:%M:%S"),
            "pnl": pnl,
            "premium": premium,
            "exit_reason": exit_reason,
            "margin_req": enter_vars.get("pos_margin"),
            "strategy": strategy,
            "legs": len(initial),
            "missing_data_at_fill": ",".join(at_fill) or None,
        }
        for k, v in enter_vars.items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                row[f"entry_var_{k}"] = v
        trades.append(row)

    if not trades:
        raise ValueError(f"No complete positions found in {filename}")

    df = pd.DataFrame(trades)
    df["margin_req"] = pd.to_numeric(df["margin_req"], errors="coerce")

    # Calculate derived fields (same as CSV path)
    df["days_in_trade"] = (df["date_closed"] - df["date_opened"]).dt.days
    df["day_of_week"] = df["date_opened"].dt.dayofweek
    df["day_name"] = df["date_opened"].dt.day_name()
    df["year"] = df["date_opened"].dt.year
    df["is_win"] = df["pnl"] > 0

    notes["pnl_field"] = " + ".join(sorted(pnl_fields))
    notes["premium_field"] = " + ".join(sorted(premium_fields))
    df.attrs["parse_notes"] = notes
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
    NOT USED -- lookahead (daily close assigned at entry). See market.py.

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

