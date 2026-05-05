"""Backtest upload utilities: CSV/JSON parsing, surface alignment, trade summary.

Supports:
  - Option Omega tab/comma-delimited CSV (one row per trade)
  - DeltaRay Mesosim events JSON (flat array, grouped by PositionId)

All functions are pandas-free (stdlib only). Date normalization reuses pnl.normalize_date.
"""
import asyncio
import csv
import decimal
import io
import json
import math
import statistics
from collections import defaultdict
from datetime import date as _date, datetime as _datetime, time as _time
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
    'day_of_week', 'year', 'daily_path', 'position_id',
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
        row['daily_path'] = []

        trades.append(row)

    if not trades:
        raise ValueError("No trades parsed from CSV")

    dates = sorted(t['date_opened'] for t in trades if t.get('date_opened'))
    strategies = sorted({t.get('strategy') for t in trades if t.get('strategy')})

    # Build columns list from keys of first trade
    columns = [
        {'name': k, 'type': _infer_column_type(k)}
        for k in trades[0].keys()
        if k not in ('time_opened', 'time_closed', 'spx_open_price', 'spx_close_price', 'daily_path')
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

def parse_backtest_json(content: str, skip_daily_path: bool = True) -> tuple[list[dict], dict]:
    """
    Parse DeltaRay Mesosim events JSON (flat array grouped by PositionId).
    Returns (trades, meta).

    When skip_daily_path=True (default), EndOfDay events are ignored
    for much faster parsing on large files.
    """
    try:
        events = json.loads(content)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON: {e}") from e

    if not isinstance(events, list):
        raise ValueError("Expected a JSON array of events")

    # Only keep events we need — skip EndOfDay (bulk of events) when not building daily_path
    _KEEP_TYPES = {'EnterPosition', 'ExitPosition', 'ExitSignal', 'EntryTrade', 'Start'}
    if not skip_daily_path:
        _KEEP_TYPES.add('EndOfDay')

    positions: dict = defaultdict(list)
    strategy: Optional[str] = None

    for event in events:
        et = event.get('EventType')

        # Extract strategy name from Start event
        if et == 'Start' and strategy is None:
            msg = event.get('Message', '')
            for key in ('StrategyName:', 'BacktestName:'):
                if key in msg:
                    parts = msg.split(key)
                    if len(parts) > 1:
                        strategy = parts[1].split()[0].strip()
                    break
            continue

        if et not in _KEEP_TYPES:
            continue

        pos_id = event.get('PositionId')
        if pos_id is not None:
            positions[pos_id].append(event)

    trades = []
    for pos_id, pos_events in positions.items():
        enter_event = exit_event = exit_signal = None
        entry_trades: list = []
        eod_events:   list = []

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
            elif et == 'EndOfDay':
                eod_events.append(ev)

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

        # Build daily path from EndOfDay events (only if not skipped)
        daily_path = []
        if not skip_daily_path and eod_events:
            eod_events.sort(key=lambda e: e.get('SimTime', ''))
            for eod in eod_events:
                sim_time = eod.get('SimTime', '')
                eod_date = sim_time[:10] if sim_time else None
                if not eod_date:
                    continue
                msg = eod.get('Message', '')
                try:
                    dit = int(msg.split('=')[1].strip()) if '=' in msg else None
                except (IndexError, ValueError):
                    dit = None
                vars_ = eod.get('Vars') or {}
                daily_path.append({
                    'date':      eod_date,
                    'dit':       dit,
                    'pos_pnl':   _safe_float(vars_.get('pos_pnl')),
                    'pos_delta': _safe_float(vars_.get('pos_delta')),
                    'pos_gamma': _safe_float(vars_.get('pos_gamma')),
                    'pos_theta': _safe_float(vars_.get('pos_theta')),
                    'pos_vega':  _safe_float(vars_.get('pos_vega')),
                    'pos_wvega': _safe_float(vars_.get('pos_wvega')),
                })

        pnl_val = pnl if pnl is not None else 0.0
        trades.append({
            'position_id':   str(pos_id),
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
            'daily_path':    daily_path,
        })

    if not trades:
        raise ValueError("No complete positions (EnterPosition + ExitPosition) found in JSON")

    dates = sorted(t['date_opened'] for t in trades)
    strategies = sorted({t.get('strategy') for t in trades if t.get('strategy')})

    columns = [
        {'name': k, 'type': _infer_column_type(k)}
        for k in trades[0].keys()
        if k != 'daily_path'
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
        if isinstance(v, (_date, _datetime, _time)):
            return str(v)
        if isinstance(v, decimal.Decimal):
            f = float(v)
            return None if (math.isnan(f) or math.isinf(f)) else f
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            return None
        return v

    # CPU-bound dict building — run in thread so event loop stays free
    def _build(surface_rows, trades):
        surface_lookup: dict[str, dict] = {}
        for row in surface_rows:
            surface_lookup[str(row['trade_date'])] = {
                k: _json_safe(v) for k, v in dict(row).items()
            }

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

    return await asyncio.to_thread(_build, surface_rows, trades)


# ── Trade summary ─────────────────────────────────────────────────────────────

def extract_daily_paths(content: str) -> list[dict]:
    """
    Extract EndOfDay snapshots from a Mesosim JSON file.
    Returns a flat list of daily rows sorted by (position_id, date).
    Returns [] for non-Mesosim content (no EndOfDay events).
    """
    try:
        events = json.loads(content)
    except (json.JSONDecodeError, ValueError):
        return []
    if not isinstance(events, list):
        return []

    positions: dict = defaultdict(list)
    for event in events:
        if event.get('EventType') == 'EndOfDay':
            pos_id = event.get('PositionId')
            if pos_id is not None:
                positions[pos_id].append(event)

    if not positions:
        return []

    rows = []
    for pos_id, eod_events in positions.items():
        eod_events.sort(key=lambda e: e.get('SimTime', ''))
        for idx, eod in enumerate(eod_events):
            sim_time = eod.get('SimTime', '')
            eod_date = sim_time[:10] if len(sim_time) >= 10 else None
            if not eod_date:
                continue
            msg = eod.get('Message', '')
            # DIT from "DIT: N" or "DIT=N" pattern in Message
            dit = None
            for sep in ('DIT: ', 'DIT=', 'DIT:'):
                if sep in msg:
                    try:
                        dit = int(msg.split(sep)[1].split()[0].strip())
                    except (IndexError, ValueError):
                        pass
                    break
            if dit is None:
                dit = idx + 1  # fallback: count events

            vars_ = eod.get('Vars') or {}
            rows.append({
                'position_id': str(pos_id),
                'date':        eod_date,
                'dit':         dit,
                'pos_pnl':     _safe_float(vars_.get('pos_pnl')),
                'pos_delta':   _safe_float(vars_.get('pos_delta')),
                'pos_gamma':   _safe_float(vars_.get('pos_gamma')),
                'pos_theta':   _safe_float(vars_.get('pos_theta')),
                'pos_vega':    _safe_float(vars_.get('pos_vega')),
                'pos_wvega':   _safe_float(vars_.get('pos_wvega')),
            })

    rows.sort(key=lambda r: (r['position_id'], r['date']))
    return rows


async def align_daily_paths_to_surface(
    daily_rows: list[dict],
    pool,
    date_from: str,
    date_to: str,
) -> tuple[list[dict], dict]:
    """
    LEFT JOIN daily path rows to surface_metrics_core at 09:35 on each row's date.
    Returns (enriched_rows, stats).
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

    import decimal

    def _json_safe(v):
        if isinstance(v, (_date, _datetime, _time)):
            return str(v)
        if isinstance(v, decimal.Decimal):
            f = float(v)
            return None if (math.isnan(f) or math.isinf(f)) else f
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            return None
        return v

    def _build(surface_rows, daily_rows):
        surface_lookup: dict[str, dict] = {}
        for row in surface_rows:
            surface_lookup[str(row['trade_date'])] = {
                k: _json_safe(v) for k, v in dict(row).items()
            }

        enriched = []
        matched = 0
        for dr in daily_rows:
            d = dr.get('date', '')
            surf = surface_lookup.get(d)
            if surf is not None:
                merged = {**surf, **dr}
                merged['join_timestamp'] = f"{d} 09:35:00"
                matched += 1
            else:
                merged = dict(dr)
                merged['join_timestamp'] = None
            enriched.append(merged)

        total = len(daily_rows)
        return enriched, {
            'total':      total,
            'matched':    matched,
            'unmatched':  total - matched,
            'match_rate': round(matched / max(total, 1), 3),
        }

    return await asyncio.to_thread(_build, surface_rows, daily_rows)


def _compute_path_derived_fields(daily_rows: list[dict], enriched_trades: list[dict]) -> list[dict]:
    """
    Compute derived per-row fields after surface alignment:
      - daily_pnl_change, max_pnl_to_date, drawdown_from_peak, pct_of_peak_retained
      - trade_phase (early/middle/late based on % of total trade duration)
      - {iv_col}_since_entry for each IV metric present in both daily row and parent trade

    Requires enriched_trades to have 'position_id' linking to daily_rows.
    """
    from collections import defaultdict

    trades_lookup = {t.get('position_id'): t for t in enriched_trades if t.get('position_id')}

    # Group daily rows by position_id, preserving order
    by_position: dict[str, list] = defaultdict(list)
    for dr in daily_rows:
        by_position[dr.get('position_id', '')].append(dr)

    # Identify IV columns from a representative trade (those not in TRADE_FIELDS)
    iv_col_set: set = set()
    if enriched_trades:
        iv_col_set = {k for k in enriched_trades[0].keys()
                      if k not in TRADE_FIELDS and not k.startswith('_')
                      and k not in ('trade_date', 'quote_time')}

    result = []
    for pos_id, series in by_position.items():
        trade = trades_lookup.get(pos_id) or {}
        total_days = trade.get('days_in_trade') or 0
        if not total_days:
            # Fall back to max dit in series
            dits = [r.get('dit') for r in series if r.get('dit') is not None]
            total_days = max(dits) if dits else 1

        max_pnl = None
        prev_pnl = None
        for dr in series:
            row = dict(dr)
            pnl = _safe_float(row.get('pos_pnl'))

            # Running max
            if pnl is not None:
                max_pnl = pnl if max_pnl is None else max(max_pnl, pnl)

            row['max_pnl_to_date'] = max_pnl
            row['daily_pnl_change'] = (
                round(pnl - prev_pnl, 4)
                if (pnl is not None and prev_pnl is not None)
                else None
            )
            row['drawdown_from_peak'] = (
                round(pnl - max_pnl, 4)
                if (pnl is not None and max_pnl is not None)
                else None
            )
            row['pct_of_peak_retained'] = (
                round(pnl / max_pnl, 4)
                if (pnl is not None and max_pnl is not None and max_pnl > 0)
                else None
            )

            # Trade phase
            dit = row.get('dit')
            if dit is not None and total_days > 0:
                pct = dit / total_days
                row['trade_phase'] = 'early' if pct <= 0.2 else ('late' if pct > 0.7 else 'middle')
            else:
                row['trade_phase'] = None

            # Since-entry IV deltas
            for iv_col in iv_col_set:
                entry_val = _safe_float(trade.get(iv_col))
                daily_val = _safe_float(row.get(iv_col))
                if entry_val is not None and daily_val is not None:
                    row[f'{iv_col}_since_entry'] = round(daily_val - entry_val, 6)

            prev_pnl = pnl
            result.append(row)

    return result


def compute_portfolio_context_features(
    daily_paths: list[dict],
    enriched_trades: list[dict],
    windows: tuple = (1, 3, 5, 10, 15),
    min_history_for_pctl: int = 60,
) -> dict:
    """
    Per-trade portfolio-context features answering: when existing positions
    had unusually large recent P&L moves, does entering a new trade now
    correlate with future P&L?

    For each trade T entered on date D:
      portfolio_pnl_chg_{N}d_at_entry      cumulative average per-trade daily
                                           P&L change over the last N trading
                                           days ending at D's 3:30pm slice
      portfolio_pnl_chg_{N}d_at_entry_pctl 0-100 expanding-window percentile
                                           rank of that value within the prior
                                           history of the same metric

    The "average per-trade" roll-up strips out concurrent-position-count bias
    (50 positions vs 20 won't artificially shift the magnitude). Calendar is
    the union of dates appearing in any daily_path row — weekends/holidays
    skipped automatically. T excludes itself: on D, T has no D-1 P&L so its
    daily_change is None and is not part of the avg.

    Look-ahead note: 3:30pm of D is the reference. For trades entered before
    3:30pm on D, the 1d feature uses information not yet observable at entry.
    Documented trade-off for time-axis consistency with Mesosim's daily slice.
    """
    by_pos: dict = defaultdict(list)
    for dp in daily_paths:
        pid = dp.get('position_id')
        if pid:
            by_pos[pid].append(dp)
    for rows in by_pos.values():
        rows.sort(key=lambda r: r.get('date', ''))

    daily_changes: dict = defaultdict(dict)
    for pid, rows in by_pos.items():
        prev_pnl = None
        for r in rows:
            pnl = _safe_float(r.get('pos_pnl'))
            d = r.get('date')
            if pnl is not None and prev_pnl is not None and d:
                daily_changes[d][pid] = pnl - prev_pnl
            prev_pnl = pnl

    if not daily_changes:
        return {}

    calendar = sorted(daily_changes.keys())

    avg_pnl_chg: dict = {}
    for d, pos_changes in daily_changes.items():
        vals = list(pos_changes.values())
        avg_pnl_chg[d] = sum(vals) / len(vals) if vals else None

    series_by_window: dict = {}
    for N in windows:
        s: dict = {}
        for i, d in enumerate(calendar):
            if i < N - 1:
                s[d] = None
                continue
            window_dates = calendar[i - N + 1:i + 1]
            vals = [avg_pnl_chg.get(wd) for wd in window_dates]
            s[d] = round(sum(vals), 4) if all(v is not None for v in vals) else None
        series_by_window[N] = s

    # Expanding-window percentile rank: for each calendar date d, rank its
    # series value within {series[d'] for d' in calendar, d' < d, value not None}.
    pctl_by_window: dict = {}
    for N, series in series_by_window.items():
        pctls: dict = {}
        prior_vals: list = []
        for d in calendar:
            v = series[d]
            if v is None or len(prior_vals) < min_history_for_pctl:
                pctls[d] = None
            else:
                n_below = sum(1 for x in prior_vals if x < v)
                n_equal = sum(1 for x in prior_vals if x == v)
                pctls[d] = round((n_below + n_equal / 2) / len(prior_vals) * 100, 1)
            if v is not None:
                prior_vals.append(v)
        pctl_by_window[N] = pctls

    out: dict = {}
    for trade in enriched_trades:
        pid = trade.get('position_id')
        date_opened = trade.get('date_opened')
        if not pid or not date_opened:
            continue
        d_ref = None
        for d in reversed(calendar):
            if d <= date_opened:
                d_ref = d
                break
        if d_ref is None:
            continue
        feat: dict = {}
        for N in windows:
            feat[f'portfolio_pnl_chg_{N}d_at_entry'] = series_by_window[N].get(d_ref)
            feat[f'portfolio_pnl_chg_{N}d_at_entry_pctl'] = pctl_by_window[N].get(d_ref)
        out[pid] = feat

    return out


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
