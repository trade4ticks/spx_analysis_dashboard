"""
Fast-path routing to surface_metrics_core for skew_slope / term_slope / convexity.

The /api/skew_slope, /api/term_slope, and /api/convexity endpoints normally
query spx_surface with a DISTINCT ON + ABS(quote_time - target_time) ORDER BY,
which scales linearly with the date range and times out past ~1300 days.

surface_metrics_core stores the same slope/convexity values (and their
underlying IV legs) for every 5-minute time slice, keyed on (trade_date,
quote_time). When the dashboard controls happen to match a precomputed
column, we can serve from this table with a simple range scan.

This module exposes a try_*_fast() coroutine per endpoint. Each one:
  - Maps dashboard params (deltas, DTEs) to a canonical column name.
  - Returns None if any required column is missing — caller falls back to the
    slow path unchanged.
  - Returns a payload in the same JSON shape as the slow path, plus
    "source": "precomputed" so the frontend can render its ⚡ badge.
"""
from __future__ import annotations

import asyncio
from datetime import date as date_type, time as time_type
from typing import Optional


# ── Catalog cache ───────────────────────────────────────────────────────────
# Loaded once on first call, never invalidated. Restart the app if the column
# set of surface_metrics_core changes.

_columns_cache: Optional[set[str]] = None
_columns_lock = asyncio.Lock()


async def _load_columns(pool) -> set[str]:
    global _columns_cache
    if _columns_cache is not None:
        return _columns_cache
    async with _columns_lock:
        if _columns_cache is not None:
            return _columns_cache
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_schema = 'public' "
                "  AND table_name = 'surface_metrics_core'"
            )
        _columns_cache = {r["column_name"] for r in rows}
    return _columns_cache


# ── Wing / time helpers ─────────────────────────────────────────────────────

_WING_BY_DELTA = {10: "10p", 25: "25p", 50: "atm", 75: "25c", 90: "10c"}


def wing_for_delta(d: int) -> Optional[str]:
    """Map a put_delta to its surface_metrics_core wing label.
    Returns None for anything not in the precomputed set."""
    return _WING_BY_DELTA.get(int(d))


def round_to_5min(t: time_type) -> time_type:
    """Snap a target_time to the nearest 5-minute boundary (the table's grain)."""
    total = t.hour * 60 + t.minute
    rounded = round(total / 5) * 5
    rounded = max(0, min(rounded, 24 * 60 - 1))
    return time_type(rounded // 60, rounded % 60)


# ── Shared response-building helpers ────────────────────────────────────────

def _format_label(trade_date, quote_time, freq: str) -> str:
    if freq == "daily":
        return trade_date.isoformat()
    return f"{trade_date.isoformat()} {quote_time.strftime('%H:%M')}"


def _ordered_unique(items):
    seen = []
    seen_set = set()
    for x in items:
        if x not in seen_set:
            seen.append(x); seen_set.add(x)
    return seen


async def _fetch_core(pool, select_cols: list[str], start_d, end_d,
                     freq: str, target_time_rounded: time_type):
    """Run the actual surface_metrics_core query.

    Daily mode filters to a single quote_time per day (snapped to 5-min grid).
    Intraday mode returns every 5-min slice in the date range.
    """
    cols_sql = ", ".join(["trade_date", "quote_time"] + select_cols)
    if freq == "daily":
        sql = (
            f"SELECT {cols_sql} FROM surface_metrics_core "
            f"WHERE trade_date BETWEEN $1 AND $2 AND quote_time = $3 "
            f"ORDER BY trade_date, quote_time"
        )
        params = (start_d, end_d, target_time_rounded)
    else:
        sql = (
            f"SELECT {cols_sql} FROM surface_metrics_core "
            f"WHERE trade_date BETWEEN $1 AND $2 "
            f"ORDER BY trade_date, quote_time"
        )
        params = (start_d, end_d)
    async with pool.acquire() as conn:
        return await conn.fetch(sql, *params)


# ── skew_slope ──────────────────────────────────────────────────────────────

async def try_skew_slope_fast(
    pool, *,
    dte_list: list[int],
    delta_a: int,
    delta_b: int,
    start_d: date_type,
    end_d: date_type,
    target_time_str: str,
    freq: str,
) -> Optional[dict]:
    wing_a = wing_for_delta(delta_a)
    wing_b = wing_for_delta(delta_b)
    if wing_a is None or wing_b is None:
        return None

    # Canonical column order: ascending delta (lower-K wing first)
    lo, hi = sorted((delta_a, delta_b))
    wing_lo, wing_hi = wing_for_delta(lo), wing_for_delta(hi)

    columns = await _load_columns(pool)

    needed: list[tuple[int, str, str, str]] = []
    for d in dte_list:
        slope_col = f"skew_{d}d_{wing_lo}_{wing_hi}"
        iv_a_col  = f"iv_{d}d_{wing_a}"
        iv_b_col  = f"iv_{d}d_{wing_b}"
        if not all(c in columns for c in (slope_col, iv_a_col, iv_b_col)):
            return None
        needed.append((d, slope_col, iv_a_col, iv_b_col))

    select_cols = []
    for _, sc, ia, ib in needed:
        for c in (sc, ia, ib):
            if c not in select_cols:
                select_cols.append(c)

    target_time = round_to_5min(time_type.fromisoformat(target_time_str))
    rows = await _fetch_core(pool, select_cols, start_d, end_d, freq, target_time)

    labels = _ordered_unique(_format_label(r["trade_date"], r["quote_time"], freq)
                             for r in rows)
    by_dte: dict[int, dict[str, dict]] = {d: {} for d in dte_list}
    for r in rows:
        lbl = _format_label(r["trade_date"], r["quote_time"], freq)
        for d, sc, ia, ib in needed:
            by_dte[d][lbl] = {
                "value": r[sc],
                "iv_a":  r[ia],
                "iv_b":  r[ib],
            }

    series = []
    for d in dte_list:
        entries = by_dte[d]
        series.append({
            "label":   f"{d}D",
            "dte":     d,
            "labels":  labels,
            "values":  [entries.get(lbl, {}).get("value") for lbl in labels],
            "metrics": [
                ({
                    "iv_a": entries.get(lbl, {}).get("iv_a"),
                    "iv_b": entries.get(lbl, {}).get("iv_b"),
                    # Strikes aren't stored in surface_metrics_core — tooltip
                    # just won't show k_a / k_b in fast mode.
                    "k_a":  None,
                    "k_b":  None,
                } if entries.get(lbl) else None)
                for lbl in labels
            ],
        })

    return {
        "freq":      freq,
        "dimension": "dte",
        "delta_a":   delta_a,
        "delta_b":   delta_b,
        "series":    series,
        "source":    "precomputed",
    }


# ── term_slope ──────────────────────────────────────────────────────────────

# term_slope columns exist only for these tenor pairs and these wings.
_TERM_SLOPE_PAIRS = {(1, 7), (7, 30), (30, 90)}
_TERM_SLOPE_WINGS = {"25p", "atm", "25c"}


def _delta_label_term(pd: int) -> str:
    if pd == 50:
        return "ATM"
    if pd < 50:
        return f"{pd}Δp"
    return f"{100 - pd}Δc"


async def try_term_slope_fast(
    pool, *,
    delta_list: list[int],
    dte_a: int,
    dte_b: int,
    start_d: date_type,
    end_d: date_type,
    target_time_str: str,
    freq: str,
) -> Optional[dict]:
    dte_lo, dte_hi = sorted((dte_a, dte_b))
    if (dte_lo, dte_hi) not in _TERM_SLOPE_PAIRS:
        return None

    wings_by_delta: dict[int, str] = {}
    for d in delta_list:
        w = wing_for_delta(d)
        if w is None or w not in _TERM_SLOPE_WINGS:
            return None
        wings_by_delta[d] = w

    columns = await _load_columns(pool)

    # One slope column per requested delta; iv legs for tooltip
    needed: list[tuple[int, str, str, str, str]] = []  # delta, slope, iv_a, iv_b, wing
    for d, wing in wings_by_delta.items():
        slope_col = f"term_slope_{dte_lo}_{dte_hi}_{wing}"
        iv_a_col  = f"iv_{dte_a}d_{wing}"
        iv_b_col  = f"iv_{dte_b}d_{wing}"
        if not all(c in columns for c in (slope_col, iv_a_col, iv_b_col)):
            return None
        needed.append((d, slope_col, iv_a_col, iv_b_col, wing))

    select_cols = []
    for _, sc, ia, ib, _w in needed:
        for c in (sc, ia, ib):
            if c not in select_cols:
                select_cols.append(c)

    target_time = round_to_5min(time_type.fromisoformat(target_time_str))
    rows = await _fetch_core(pool, select_cols, start_d, end_d, freq, target_time)

    labels = _ordered_unique(_format_label(r["trade_date"], r["quote_time"], freq)
                             for r in rows)

    T_a = dte_a / 365.0
    T_b = dte_b / 365.0
    dT  = T_b - T_a  # may be negative if user picked dte_a > dte_b — sign-stable for fwd_var

    by_delta: dict[int, dict[str, dict]] = {d: {} for d in delta_list}
    for r in rows:
        lbl = _format_label(r["trade_date"], r["quote_time"], freq)
        for d, sc, ia, ib, _w in needed:
            iv_a = r[ia]
            iv_b = r[ib]
            fwd_var = None
            if iv_a is not None and iv_b is not None and dT != 0:
                fwd_var = (iv_b * iv_b * T_b - iv_a * iv_a * T_a) / dT
            by_delta[d][lbl] = {
                "value":   r[sc],
                "fwd_var": fwd_var,
                "iv_a":    iv_a,
                "iv_b":    iv_b,
            }

    series = []
    for d in delta_list:
        entries = by_delta[d]
        series.append({
            "label":   _delta_label_term(d),
            "delta":   d,
            "labels":  labels,
            "values":  [entries.get(lbl, {}).get("value") for lbl in labels],
            "metrics": [
                ({
                    "iv_a":    entries.get(lbl, {}).get("iv_a"),
                    "iv_b":    entries.get(lbl, {}).get("iv_b"),
                    "fwd_var": entries.get(lbl, {}).get("fwd_var"),
                } if entries.get(lbl) else None)
                for lbl in labels
            ],
        })

    return {
        "freq":      freq,
        "dimension": "delta",
        "dte_a":     dte_a,
        "dte_b":     dte_b,
        "series":    series,
        "source":    "precomputed",
    }


# ── convexity ───────────────────────────────────────────────────────────────

# Only these (left_delta, center_delta, right_delta) triples have convex columns.
_CONVEX_TRIPLES = {(10, 25, 50), (10, 50, 90), (25, 50, 75), (50, 75, 90)}


async def try_convexity_fast(
    pool, *,
    dte_list: list[int],
    left_delta: int,
    center_delta: int,
    right_delta: int,
    start_d: date_type,
    end_d: date_type,
    target_time_str: str,
    freq: str,
) -> Optional[dict]:
    if (left_delta, center_delta, right_delta) not in _CONVEX_TRIPLES:
        return None
    wing_l = wing_for_delta(left_delta)
    wing_c = wing_for_delta(center_delta)
    wing_r = wing_for_delta(right_delta)
    if not all((wing_l, wing_c, wing_r)):
        return None

    columns = await _load_columns(pool)

    needed: list[tuple[int, str, str, str, str]] = []  # dte, conv, iv_l, iv_c, iv_r
    for d in dte_list:
        conv_col = f"convex_{d}d_{wing_l}_{wing_c}_{wing_r}"
        iv_l_col = f"iv_{d}d_{wing_l}"
        iv_c_col = f"iv_{d}d_{wing_c}"
        iv_r_col = f"iv_{d}d_{wing_r}"
        if not all(c in columns for c in (conv_col, iv_l_col, iv_c_col, iv_r_col)):
            return None
        needed.append((d, conv_col, iv_l_col, iv_c_col, iv_r_col))

    select_cols = []
    for _, cc, il, ic, ir in needed:
        for c in (cc, il, ic, ir):
            if c not in select_cols:
                select_cols.append(c)

    target_time = round_to_5min(time_type.fromisoformat(target_time_str))
    rows = await _fetch_core(pool, select_cols, start_d, end_d, freq, target_time)

    labels = _ordered_unique(_format_label(r["trade_date"], r["quote_time"], freq)
                             for r in rows)

    by_dte: dict[int, dict[str, dict]] = {d: {} for d in dte_list}
    for r in rows:
        lbl = _format_label(r["trade_date"], r["quote_time"], freq)
        for d, cc, il, ic, ir in needed:
            by_dte[d][lbl] = {
                "value":    r[cc],
                "iv_left":  r[il],
                "iv_center": r[ic],
                "iv_right": r[ir],
            }

    series = []
    for d in dte_list:
        entries = by_dte[d]
        series.append({
            "label":   f"{d}D",
            "dte":     d,
            "labels":  labels,
            "values":  [entries.get(lbl, {}).get("value") for lbl in labels],
            "metrics": [
                ({
                    "iv_left":   entries.get(lbl, {}).get("iv_left"),
                    "iv_center": entries.get(lbl, {}).get("iv_center"),
                    "iv_right":  entries.get(lbl, {}).get("iv_right"),
                } if entries.get(lbl) else None)
                for lbl in labels
            ],
        })

    return {
        "freq":         freq,
        "dimension":    "dte",
        "left_delta":   left_delta,
        "center_delta": center_delta,
        "right_delta":  right_delta,
        "series":       series,
        "source":       "precomputed",
    }
