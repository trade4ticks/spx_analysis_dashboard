"""P&L upload utilities: CSV parsing, date normalization, surface alignment, summary stats."""
import csv
import io
import math
import re
import statistics
from typing import Optional


def normalize_date(s: str) -> str:
    """Convert M/D/YYYY, MM/DD/YYYY, or YYYY-MM-DD → canonical YYYY-MM-DD."""
    s = s.strip()
    if re.match(r'^\d{4}-\d{2}-\d{2}$', s):
        return s
    m = re.match(r'^(\d{1,2})/(\d{1,2})/(\d{4})$', s)
    if m:
        month, day, year = m.group(1), m.group(2), m.group(3)
        return f"{year}-{month.zfill(2)}-{day.zfill(2)}"
    raise ValueError(f"Unrecognized date format: {s!r}")


def normalize_time(s: str) -> str:
    """Convert H:MM:SS or HH:MM:SS → canonical HH:MM:SS."""
    s = s.strip()
    parts = s.split(':')
    if len(parts) == 3:
        h, m, sec = parts
        return f"{h.zfill(2)}:{m}:{sec}"
    raise ValueError(f"Unrecognized time format: {s!r}")


def _detect_column_roles(header: list[str]) -> dict[str, str]:
    """Return {col_name: role} where role is 'date'|'time'|'pnl'|'greek'|'other'."""
    roles = {}
    for col in header:
        c = col.strip().lower()
        if c in ('trade_date', 'date'):
            roles[col] = 'date'
        elif c in ('quote_time', 'time'):
            roles[col] = 'time'
        elif c in ('pnl', 'p_l', 'pl', 'profit_loss', 'p&l'):
            roles[col] = 'pnl'
        elif c in ('delta', 'theta', 'vega', 'gamma', 'wt_vega'):
            roles[col] = 'greek'
        else:
            roles[col] = 'other'
    return roles


def parse_pnl_csv(content: str) -> tuple[list[dict], dict]:
    """
    Parse P&L CSV string. Returns (rows, meta) where:
    - rows: normalized row dicts with YYYY-MM-DD dates and HH:MM:SS times
    - meta: {columns: [{name, type}], row_count, date_from, date_to}
    """
    reader = csv.DictReader(io.StringIO(content))
    header = list(reader.fieldnames or [])
    roles = _detect_column_roles(header)

    rows = []
    for raw in reader:
        row = {}
        for col, val in raw.items():
            val = (val or '').strip()
            role = roles.get(col, 'other')
            if role == 'date' and val:
                try:
                    row['trade_date'] = normalize_date(val)
                except ValueError:
                    row['trade_date'] = val
            elif role == 'time' and val:
                try:
                    row['quote_time'] = normalize_time(val)
                except ValueError:
                    row['quote_time'] = val
            else:
                if val in ('', 'None', 'null', 'NA', 'N/A'):
                    row[col] = None
                else:
                    try:
                        row[col] = float(val)
                    except (ValueError, TypeError):
                        row[col] = val
        rows.append(row)

    dates = sorted({r.get('trade_date', '') for r in rows if r.get('trade_date')})
    columns = []
    for c in header:
        role = roles.get(c, 'other')
        canonical = 'trade_date' if role == 'date' else ('quote_time' if role == 'time' else c)
        columns.append({'name': canonical, 'type': role})

    return rows, {
        'columns':   columns,
        'row_count': len(rows),
        'date_from': dates[0] if dates else None,
        'date_to':   dates[-1] if dates else None,
    }


async def align_pnl_to_surface(
    pnl_rows: list[dict],
    pool,
    date_from: str,
    date_to: str,
) -> tuple[list[dict], dict]:
    """
    Inner-join P&L rows with surface_metrics_core on (trade_date, quote_time).
    Returns (merged_rows, stats).
    """
    async with pool.acquire() as conn:
        surface_rows = await conn.fetch(
            """SELECT * FROM surface_metrics_core
               WHERE trade_date BETWEEN $1::date AND $2::date
               ORDER BY trade_date, quote_time""",
            date_from, date_to,
        )

    surface_lookup: dict[tuple, dict] = {}
    for row in surface_rows:
        key = (str(row['trade_date']), str(row['quote_time']))
        surface_lookup[key] = dict(row)

    merged = []
    for pnl_row in pnl_rows:
        td = pnl_row.get('trade_date', '')
        qt = pnl_row.get('quote_time', '')
        surf = surface_lookup.get((td, qt))
        if surf is None:
            continue
        merged_row = {**surf, **pnl_row}
        merged_row['trade_date'] = td
        merged_row['quote_time'] = qt
        merged.append(merged_row)

    stats = {
        'matched':      len(merged),
        'pnl_rows':     len(pnl_rows),
        'surface_rows': len(surface_rows),
        'match_rate':   round(len(merged) / max(len(pnl_rows), 1), 3),
        'date_range':   f"{date_from} → {date_to}",
    }
    return merged, stats


def compute_summary_stats(rows: list[dict], cols: list[str]) -> dict:
    """Return {col: {mean, std, min, max, p25, p75, n}} for numeric columns."""
    result = {}
    for col in cols:
        vals = []
        for r in rows:
            v = r.get(col)
            if v is None:
                continue
            try:
                f = float(v)
                if not math.isnan(f):
                    vals.append(f)
            except (ValueError, TypeError):
                pass
        if not vals:
            continue
        vs = sorted(vals)
        n = len(vs)
        result[col] = {
            'mean': round(statistics.mean(vals), 4),
            'std':  round(statistics.stdev(vals), 4) if n > 1 else 0.0,
            'min':  round(vs[0], 4),
            'max':  round(vs[-1], 4),
            'p25':  round(vs[int(n * 0.25)], 4),
            'p75':  round(vs[int(n * 0.75)], 4),
            'n':    n,
        }
    return result
