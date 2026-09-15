"""Trade DataFrame -> the JSON the page holds in the browser.

The whole trade log goes to the client once; every filter and re-bin after
that happens in JavaScript. Two properties are deliberate:

DATES ARE ISO STRINGS, "YYYY-MM-DD", made here. The source app round-tripped
datetimes through dcc.Store, they came back as strings, and filter_dataframe()
had to convert them back -- a documented bug. A fixed-width ISO date compares
correctly as a plain string, so the browser never parses a date to filter one.

COLUMNS ARE A WHITELIST. Anything not listed here does not reach the page,
which is how the dropped SharpTwo and skew columns (vrp_calc, iv_rv_ratio,
vov, realized_vol, spot_vol_correlation, regime, *_skew) are removed rather
than merely hidden: an all-NaN column that reaches the binning produces a
plausible empty section instead of an error.
"""
from __future__ import annotations

import math

import pandas as pd

# Columnar: {name: [values]}. A few thousand rows of ~20 columns is small
# either way; columns also filter faster in JS than an array of objects.
TRADE_COLUMNS = [
    "date_opened", "time_opened", "date_closed", "time_closed", "pnl", "premium", "exit_reason",
    "margin_req", "legs", "days_in_trade", "day_of_week", "year", "is_win",
    "vix_level", "vix3m_level", "vix9d_level",
    "vix_bar_time", "vix3m_bar_time", "vix9d_bar_time",
    "gap", "vix_overnight_gap",
    # Both ratio bases until one is chosen (registry `basis`).
    "vix3m_vix_ratio_entry", "vix3m_vix_ratio_close", "vix_vix9d_ratio_entry", "vix_vix9d_ratio_close",
    # CSV-only extras (null for Mesosim)
    "spx_open_price", "spx_close_price", "contracts", "pnl_pct",
    "max_profit", "max_loss", "strategy",
    # Mesosim-only (null for CSV)
    "position_id", "missing_data_at_fill",
]

# The parser keeps every Mesosim EnterPosition Var as entry_var_* on the
# DataFrame, but none reach the page: ~20 auto-included columns would undo the
# explicit whitelist above. Add one here by name when it is wanted.


def allowed_column(name: str) -> bool:
    return name in TRADE_COLUMNS

DATE_COLUMNS = ("date_opened", "date_closed")
TEXT_COLUMNS = ("exit_reason", "legs", "strategy", "time_opened", "time_closed", "missing_data_at_fill",
                "vix_bar_time", "vix3m_bar_time", "vix9d_bar_time")
INT_COLUMNS = ("days_in_trade", "day_of_week", "year", "position_id")


def _num(v):
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(f) or math.isinf(f) else f


def trades_to_payload(df: pd.DataFrame) -> dict:
    cols: dict[str, list] = {}
    for c in [c for c in TRADE_COLUMNS if c in df.columns]:
        s = df[c]
        if c in DATE_COLUMNS:
            d = pd.to_datetime(s, errors="coerce")
            cols[c] = [None if pd.isna(x) else x.strftime("%Y-%m-%d") for x in d]
        elif c in TEXT_COLUMNS:
            cols[c] = [None if (x is None or (isinstance(x, float) and math.isnan(x))) else str(x)
                       for x in s]
        elif c == "is_win":
            cols[c] = [bool(x) for x in s]
        elif c in INT_COLUMNS:
            cols[c] = [None if _num(x) is None else int(x) for x in s]
        else:
            cols[c] = [_num(x) for x in s]

    # Order by open date and time so cumulative P/L and drawdown read left to
    # right without the client re-sorting. Stable, so ties keep file order.
    n = len(df)
    times = cols.get("time_opened") or [None] * n
    order = sorted(range(n), key=lambda i: (cols["date_opened"][i] or "", times[i] or "", i))
    if order != list(range(n)):
        cols = {k: [v[i] for i in order] for k, v in cols.items()}

    opened = [d for d in cols["date_opened"] if d]
    closed = [d for d in cols["date_closed"] if d]
    return {
        "n": n,
        "columns": cols,
        "date_min": min(opened) if opened else None,
        "date_max": max(closed) if closed else None,
        "notes": _jsonable(df.attrs.get("parse_notes") or {}),
    }


def _jsonable(obj):
    """parse_notes may hold numpy scalars; make them plain JSON."""
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if hasattr(obj, "item"):
        return obj.item()
    return obj


def assert_no_dropped_columns(payload: dict) -> None:
    extra = {c for c in payload["columns"] if not allowed_column(c)}
    if extra:
        raise AssertionError(f"payload carries non-whitelisted columns: {sorted(extra)}")

