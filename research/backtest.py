"""Backtest upload utilities: CSV/JSON parsing, surface alignment, trade summary.

Supports:
  - Option Omega tab/comma-delimited CSV (one row per trade)
  - DeltaRay Mesosim events JSON (flat array, grouped by PositionId)

All functions are pandas-free (stdlib only). Date normalization reuses pnl.normalize_date.
"""
import csv
import decimal
import io
import json
import math
import statistics
from collections import defaultdict
from datetime import date as _date, datetime as _datetime
from typing import Optional

from research.pnl import normalize_date


# ── Column mapping for Option Omega CSV ──────────────────────────────────────

_COLUMN_MAPPING = {
    "Date Opened":              "date_opened",
    "Date Closed":              "date_closed",
    "P/L":                      "pnl",
    "P/L %":                    "pnl_pct",
    "Strategy":                 "strategy",
    "Max Profit":               "max_profit",
    "Max Loss":                 "max_loss",
    "Margin Req.":              "margin_req",
    "Legs":                     "legs",
    "Reason For Close":         "exit_reason",
    "Premium":                  "premium",
    "No. of Contracts":         "contracts",
    "Opening Price":            "spx_open_price",
    "Closing Price":            "spx_close_price",
    "Time Opened":              "time_opened",
    "Time Closed":              "time_closed",
}

_MESOSIM_EXIT_REASONS = {
    "profit target":  "Profit Target",
    "stop loss":      "Stop Loss",
    "max time in trade": "Max DIT",
    "adjustment count":  "Max Adjustments",
}

# Trade-level fields that are NOT IV surface metrics (used to separate IV cols from trade cols)
TRADE_FIELDS = {
    'date_opened', 'date_closed', 'pnl', 'pnl_pct', 'strategy',
    'max_profit', 'max_loss', 'margin_req', 'legs', 'exit_reason',
    'premium', 'contracts', 'spx_open_price', 'spx_close_price',
    'time_opened', 'time_closed', 'days_in_trade', 'is_win',
    'join_timestamp', 'trade_date', 'quote_time', 'id',
    'day_of_week', 'year',
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _safe_float(v) -> Optional[float]:
    if v is None or v == '':
        return None
    try:
        f = float(v)
        return None if math.isnan(f) else f
    except (ValueError, TypeError):
        return None


def _parse_date(s: str) -> str:
    """Parse a date string in various formats → YYYY-MM-DD."""
    s = s.strip()
    if not s:
        raise ValueError("Empty date string")
    # Try ISO datetime first (Mesosim SimTime may be passed directly)
    if 'T' in s or ' ' in s:
        try:
            return str(_datetime.fromisoformat(s.split('T')[0].split(' ')[0]).date())
        except ValueError:
            pass
    return normalize_date(s)


def _days_between(d1: str, d2: str) -> Optional[int]:
    try:
        return (_date.fromisoformat(d2) - _date.fromisoformat(d1)).days
    except (ValueError, TypeError):
        return None


def _simplify_exit_reason(msg: str) -> str:
    msg_lower = (msg or '').lower()
    for pattern, label in _MESOSIM_EXIT_REASONS.items():
        if pattern in msg_lower:
            return label
    return msg or 'Unknown'


def _infer_column_type(name: str) -> str:
    n = name.lower()
    if n in ('date_opened', 'date_closed'):
        return 'date'
    if n in ('pnl', 'pnl_pct'):
        return 'pnl'
    if n in ('strategy', 'exit_reason'):
        return 'text'
    return 'numeric'


# ── CSV Parser (Option Omega) ─────────────────────────────────────────────────

def parse_backtest_csv(content: str) -> tuple[list[dict], dict]:
    """
    Parse Option Omega tab/comma-delimited CSV.
    Returns (trades, meta) where:
      trades: list of normalized trade dicts
      meta: {source, trade_count, date_from, date_to, strategies, columns}
    """
    # Try tab first, fall back to comma
    sample = io.StringIO(content)
    first_line = sample.readline()
    delimiter = '\t' if first_line.count('\t') > first_line.count(',') else ','
    sample.seek(0)

    reader = csv.DictReader(sample, delimiter=delimiter)
    raw_rows = list(reader)
    if not raw_rows:
        raise ValueError("CSV file is empty or has no data rows")

    trades = []
    for raw in raw_rows:
        row: dict = {}
        for orig_col, val in raw.items():
            if orig_col is None:
                continue
            mapped = _COLUMN_MAPPING.get(orig_col.strip(), orig_col.strip())
            val = (val or '').strip()
            if not val or val in ('None', 'null', 'NA', 'N/A', '-'):
                row[mapped] = None
            elif mapped in ('date_opened', 'date_closed'):
                try:
                    row[mapped] = _parse_date(val)
                except ValueError:
                    row[mapped] = val
            elif mapped in ('strategy', 'exit_reason', 'time_opened', 'time_closed'):
                row[mapped] = val
            else:
                row[mapped] = _safe_float(val) if _safe_float(val) is not None else val

        # Derived fields
        d_open = row.get('date_opened')
        d_close = row.get('date_closed')
        if d_open and d_close:
            row['days_in_trade'] = _days_between(d_open, d_close)
        pnl = _safe_float(row.get('pnl'))
        row['is_win'] = bool(pnl is not None and pnl > 0)

        trades.append(row)

    if not trades:
        raise ValueError("No trades parsed from CSV")

    dates = sorted(t['date_opened'] for t in trades if t.get('date_opened'))
    strategies = sorted({t.get('strategy') for t in trades if t.get('strategy')})

    # Build columns list from keys of first trade
    columns = [
        {'name': k, 'type': _infer_column_type(k)}
        for k in trades[0].keys()
        if k not in ('time_opened', 'time_closed', 'spx_open_price', 'spx_close_price')
    ]

    return trades, {
        'source':      'csv',
        'trade_count': len(trades),
        'date_from':   dates[0] if dates else None,
        'date_to':     dates[-1] if dates else None,
        'strategies':  strategies,
        'columns':     columns,
    }


# ── JSON Parser (DeltaRay Mesosim) ────────────────────────────────────────────

def parse_backtest_json(content: str) -> tuple[list[dict], dict]:
    """
    Parse DeltaRay Mesosim events JSON (flat array grouped by PositionId).
    Returns (trades, meta).
    """
    try:
        events = json.loads(content)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON: {e}") from e

    if not isinstance(events, list):
        raise ValueError("Expected a JSON array of events")

    positions: dict = defaultdict(list)
    strategy: Optional[str] = None

    for event in events:
        # Extract strategy name from Start event
        if event.get('EventType') == 'Start' and strategy is None:
            msg = event.get('Message', '')
            for key in ('StrategyName:', 'BacktestName:'):
                if key in msg:
                    parts = msg.split(key)
                    if len(parts) > 1:
                        strategy = parts[1].split()[0].strip()
                    break

        pos_id = event.get('PositionId')
        if pos_id is not None:
            positions[pos_id].append(event)

    trades = []
    for pos_id, pos_events in positions.items():
        enter_event = exit_event = exit_signal = None
        entry_trades = []

        for ev in pos_events:
            et = ev.get('EventType')
            if et == 'EnterPosition':
                enter_event = ev
            elif et == 'ExitPosition':
                exit_event = ev
            elif et == 'ExitSignal':
                exit_signal = ev
            elif et == 'EntryTrade':
                entry_trades.append(ev)

        if not enter_event or not exit_event:
            continue

        # Dates from SimTime (ISO datetime → date portion)
        try:
            date_opened = enter_event['SimTime'][:10]
            date_closed = exit_event['SimTime'][:10]
            # Validate
            _date.fromisoformat(date_opened)
            _date.fromisoformat(date_closed)
        except (KeyError, ValueError):
            continue

        exit_vars  = exit_event.get('Vars') or {}
        enter_vars = enter_event.get('Vars') or {}

        pnl       = _safe_float(exit_vars.get('pos_pnl', exit_vars.get('pos_realized_pnl', 0)))
        margin_req = _safe_float(enter_vars.get('pos_margin', enter_vars.get('stop_loss', 0)))

        # Premium from initial EntryTrade events (same SimTime as EnterPosition)
        enter_time = enter_event.get('SimTime', '')
        initial_entries = [t for t in entry_trades if t.get('SimTime') == enter_time]
        premium = 0.0
        for t in initial_entries:
            te = t.get('TradeEvent') or {}
            price = float(te.get('Price', 0) or 0)
            qty   = float(te.get('Qty', 0) or 0)
            mult  = float((te.get('Contract') or {}).get('Multiplier', 100) or 100)
            premium += price * qty * mult

        exit_reason = _simplify_exit_reason(
            exit_signal.get('Message', 'Unknown') if exit_signal else 'Unknown'
        )

        pnl_val = pnl if pnl is not None else 0.0
        trades.append({
            'date_opened':   date_opened,
            'date_closed':   date_closed,
            'pnl':           round(pnl_val, 4),
            'strategy':      strategy or 'Unknown',
            'premium':       round(premium, 4),
            'margin_req':    margin_req,
            'legs':          len(initial_entries),
            'exit_reason':   exit_reason,
            'days_in_trade': _days_between(date_opened, date_closed),
            'is_win':        pnl_val > 0,
        })

    if not trades:
        raise ValueError("No complete positions (EnterPosition + ExitPosition) found in JSON")

    dates = sorted(t['date_opened'] for t in trades)
    strategies = sorted({t.get('strategy') for t in trades if t.get('strategy')})

    columns = [
        {'name': k, 'type': _infer_column_type(k)}
        for k in trades[0].keys()
    ]

    return trades, {
        'source':      'json',
        'trade_count': len(trades),
        'date_from':   dates[0] if dates else None,
        'date_to':     dates[-1] if dates else None,
        'strategies':  strategies,
        'columns':     columns,
    }


# ── Auto-detect parser ────────────────────────────────────────────────────────

def parse_backtest_upload(content: bytes, filename: str) -> tuple[list[dict], dict]:
    """Auto-detect format by filename extension. Returns (trades, meta)."""
    for enc in ('utf-8-sig', 'utf-8', 'latin-1'):
        try:
            text = content.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise ValueError("Could not decode file — unsupported encoding")

    fname = filename.lower()
    if fname.endswith('.json'):
        return parse_backtest_json(text)
    elif fname.endswith('.csv'):
        return parse_backtest_csv(text)
    else:
        raise ValueError(f"Unsupported file type '{filename}' — upload a .csv or .json file")


# ── Surface alignment ─────────────────────────────────────────────────────────

async def align_trades_to_surface(
    trades: list[dict],
    pool,
    date_from: str,
    date_to: str,
) -> tuple[list[dict], dict]:
    """
    LEFT JOIN trades to surface_metrics_core at 09:35 on date_opened.
    All trades are kept; unmatched trades have None IV fields.
    Returns (enriched_trades, stats).
    """
    def _as_date(v):
        if isinstance(v, _date):
            return v
        return _date.fromisoformat(str(v))

    async with pool.acquire() as conn:
        surface_rows = await conn.fetch(
            """SELECT * FROM surface_metrics_core
               WHERE trade_date BETWEEN $1 AND $2
                 AND quote_time = '09:35:00'::time
               ORDER BY trade_date""",
            _as_date(date_from), _as_date(date_to),
        )

    def _json_safe(v):
        if isinstance(v, (_date, _datetime)):
            return str(v)
        if isinstance(v, decimal.Decimal):
            f = float(v)
            return None if (math.isnan(f) or math.isinf(f)) else f
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            return None
        return v

    # Build lookup keyed by trade_date string; sanitize for JSON storage
    surface_lookup: dict[str, dict] = {}
    for row in surface_rows:
        surface_lookup[str(row['trade_date'])] = {k: _json_safe(v) for k, v in dict(row).items()}

    enriched = []
    matched = 0
    for trade in trades:
        d = trade.get('date_opened', '')
        surf = surface_lookup.get(d)
        if surf is not None:
            merged = {**surf, **trade}
            merged['join_timestamp'] = f"{d} 09:35:00"
            matched += 1
        else:
            merged = dict(trade)
            merged['join_timestamp'] = None
        enriched.append(merged)

    total = len(trades)
    unmatched = total - matched
    match_rate = round(matched / max(total, 1), 3)

    warnings = []
    if total < 30:
        warnings.append(f"Small sample: only {total} trades. Results may be unstable.")
    if match_rate < 0.80:
        warnings.append(
            f"Low IV match rate ({match_rate:.0%}). "
            "Check that trade entry dates overlap with surface_metrics_core data."
        )
    years = {str(t.get('date_opened', ''))[:4] for t in trades if t.get('date_opened')}
    if len(years) <= 1:
        warnings.append(
            "All trades are from a single year. "
            "Time-split stability validation will be limited."
        )

    stats = {
        'total':      total,
        'matched':    matched,
        'unmatched':  unmatched,
        'match_rate': match_rate,
        'warnings':   warnings,
    }
    return enriched, stats


# ── Trade summary ─────────────────────────────────────────────────────────────

def compute_trade_summary(trades: list[dict]) -> dict:
    """High-level summary of enriched trade list for agentic context."""
    pnl_vals = [_safe_float(t.get('pnl')) for t in trades]
    pnl_vals = [v for v in pnl_vals if v is not None]

    wins = [t for t in trades if t.get('is_win')]
    strategies = sorted({t.get('strategy') for t in trades if t.get('strategy')})
    matched = sum(1 for t in trades if t.get('join_timestamp'))
    dates = sorted(t.get('date_opened', '') for t in trades if t.get('date_opened'))

    return {
        'n':            len(trades),
        'date_from':    dates[0] if dates else None,
        'date_to':      dates[-1] if dates else None,
        'strategies':   strategies,
        'win_rate':     round(len(wins) / max(len(trades), 1), 4),
        'mean_pnl':     round(statistics.mean(pnl_vals), 2) if pnl_vals else None,
        'std_pnl':      round(statistics.stdev(pnl_vals), 2) if len(pnl_vals) > 1 else None,
        'matched_count': matched,
        'match_rate':   round(matched / max(len(trades), 1), 3),
    }
