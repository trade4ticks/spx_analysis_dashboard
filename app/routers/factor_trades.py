"""Factor Trades — 20x20 heatmap and stats recomputed under an exit policy.

Reads `trade_paths` (one row per ticker/trade_date/entry_anchor, with a
precomputed exit bar + exit return for each of ~59 rule-parameter
combinations) joined to `tt_bins` on (ticker, trade_date). The precompute is
signal-agnostic — it resolves exits for every ticker-day — so this slices
into it by bin.

Two things are deliberately NOT done here:

  * Binning. Bins are read from the stored `tt_bins` columns, never derived.
  * The exit combine. `app.trade_path_rules.build_combine_sql` (vendored from
    Open_Interest) owns it, including the horizon backstop that makes an
    unbounded exit unreachable. Do not hand-roll a LEAST() here.

The train/test split uses the same frozen cutoff as every other TT surface,
read from tt_bins via oi_analysis._get_tt_cutoff. There is no free-form
cutoff.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from datetime import date as _date
from typing import Any, Optional

import numpy as np

from fastapi import APIRouter, Body, Depends, Query
from pydantic import BaseModel

from app.db import get_oi_pool
from app.trade_path_rules import (
    CombineError,
    HORIZON_RULE_KEY,
    # The tie-break order the vectorised combine transcribes. Imported from
    # the vendored module rather than restated here, so the numpy path and
    # the SQL cannot come to disagree about which rule wins a same-bar tie.
    SIDE_PRIORITY,
    build_combine_sql,
    by_key_from_rows,
)

router = APIRouter(tags=["factor_trades"])

# trade_paths.exit_bar counts BARS, not days. The paths are built on 1-minute
# bars over the regular US session, so one session is 390 bars. Displaying the
# raw count reads as "558.3d" for what is really ~1.4 sessions, which is why
# this conversion belongs server-side rather than in each consumer.
BARS_PER_SESSION = 390.0

# side -> display group. Four groups, not five: `trail` and `breakeven` live
# under 'stop' in the registry, and that is honest — a trailing stop is a
# stop. Family is the sub-grouping inside each.
SIDE_GROUPS = [
    ("stop",   "Stops"),
    ("target", "Targets"),
    ("time",   "Time"),
    ("trend",  "Trend"),
]


def _coerce_params(params: Any) -> dict:
    """Catalog `params` as a dict, from either JSONB shape."""
    if isinstance(params, str):
        try:
            params = json.loads(params)
        except (ValueError, TypeError):
            return {}
    return params if isinstance(params, dict) else {}


def _param_unit(name: str) -> Optional[str]:
    """Unit implied by a PARAMETER NAME, or None if the catalog does not say.

    Deliberately keyed on the parameter name and never on the family: a family
    check is a list this file has to be told about every time the catalog
    grows, and being told late is exactly how a new family renders wrong.

    These are the same conventions the rail applies client-side
    (factor_trades.js `_dimUnit`), so a value cannot read as 2% in the split
    dropdown and 2.0 in the sweep chip for the same rule.

    An unrecognised name returns None and the caller shows the raw value with
    its parameter name attached -- the name is then the only claim being made,
    rather than a unit being guessed onto it.
    """
    d = (name or "").lower()
    if "pct" in d or "percent" in d:
        return "pct"
    if d == "k" or "atr" in d:
        return "atr"
    if d == "n" or d in ("days", "bars") or "day" in d:
        return "day"
    return None


def _fmt_param(name: str, value: Any, alone: bool) -> str:
    """One parameter, with its unit if the name declares one.

    The name is dropped ONLY when this is the family's single parameter AND
    the name declares a unit -- the unit is then what carries the meaning
    ("2%", "2x ATR"). A unitless parameter keeps its name even when alone,
    because a bare "1" for breakeven's activation says neither what it is nor
    what it is measured in. That also leaves every existing single-parameter
    label exactly as it was.
    """
    unit = _param_unit(name)
    try:
        num = float(value)
    except (TypeError, ValueError):
        return f"{name}={value}"
    if unit == "pct":
        shown = f"{num:g}%"
    elif unit == "atr":
        shown = f"{num:g}x ATR"
    elif unit == "day":
        # Bare number, no "d". The family name beside the control already
        # says these are days, and the suffix made the values look like
        # strings to sort as strings.
        shown = f"{num:g}"
    else:
        shown = f"{num:g}"
    return shown if (alone and unit) else f"{name}={shown}"


def _rule_label(family: str, params: Any) -> str:
    """Human label for a rule option, derived from its params JSON.

    Never parses the rule_key — the key's encoding (2p5 for 2.5) is a column
    naming artefact, not a display format.

    EVERY parameter is rendered, which is the whole point. The previous
    version returned on the first recognised parameter it found, so a
    two-parameter family sharing that parameter collapsed: {k, activation}
    rendered as "1.5x ATR" for every activation of a given k, and the rail,
    the sweep chips and the grid axis all showed several different rules under
    one identical label. A label that cannot distinguish two rules is worse
    than a raw params dict, because it looks correct.

    Single-parameter families whose parameter declares a unit still render the
    value alone, exactly as before — "2%", "2x ATR", "20".
    """
    params = _coerce_params(params)
    if not params:
        return family
    alone = len(params) == 1
    return ", ".join(_fmt_param(a, params[a], alone) for a in sorted(params))


def _rule_sort_key(params: Any) -> tuple:
    """Sort key for a family's options, over ALL of its parameters.

    The catalog is read ORDER BY rule_key, which is a STRING sort: max_days
    came out 1, 10, 15, 20, 3, 5, 7. The ordering never came from the label,
    so the options are sorted here on the numeric parameters instead.

    ALL parameters, in sorted-name order — not the first recognised one. A
    single parameter is unaffected; a two-parameter family previously tied on
    every rule and fell back to whatever order the catalog query happened to
    emit. That was already live: all 16 `trail` rules shared one key, as did
    every `breakeven`, `no_progress` and `ma_close_below` rule.

    Sorted-name order is the SAME order the sweep panel groups by
    (factor_trades.js `famDims`), so the chip groups and the axis ordering
    agree by construction rather than by coincidence.

    Each value is tagged (0, float) when numeric and (1, str) otherwise, so a
    non-numeric parameter sorts after the numeric ones without making the
    tuples incomparable.
    """
    params = _coerce_params(params)
    if not params:
        return (1, ())
    out = []
    for name in sorted(params):
        v = params[name]
        try:
            out.append((0, float(v), ""))
        except (TypeError, ValueError):
            out.append((1, 0.0, str(v)))
    return (0, tuple(out))


# Participation ratio. The implementation moved to _stat_shared so the
# portfolio bar reports the SAME quantity; this alias keeps every existing
# call site here unchanged.
from app.routers._stat_shared import effective_tickers as _effective_tickers


PRICE_BUCKETS = [(0, 25), (25, 50), (50, 100), (100, 150), (150, 200),
                 (200, 300), (300, 400), (400, 500), (500, 750),
                 (750, 1000), (1000, float("inf"))]


def _price_bins(trades: list) -> list:
    """Avg return by entry-price bucket, over ONE population.

    Server-side precisely because it is a single-window aggregate: computed
    in the client from a trade list, it would silently pick up whichever
    population that list happened to hold. Buckets are FIXED rather than
    quantile-derived so the axis means the same thing across runs.
    """
    acc = [[0, 0.0] for _ in PRICE_BUCKETS]
    for t in trades:
        px = t.get("entry_price")
        if px is None:
            continue
        for i, (lo, hi) in enumerate(PRICE_BUCKETS):
            if lo <= px < hi:
                acc[i][0] += 1
                acc[i][1] += t.get("ret") or 0.0
                break
    out = []
    for (lo, hi), (n, tot) in zip(PRICE_BUCKETS, acc):
        out.append({
            "label":   f"${lo}+" if hi == float("inf") else f"${lo}-{hi}",
            "n":       n,
            # None, not 0.0 -- an empty bucket is "no trades here", which must
            # not render as a flat zero-return bar.
            "avg_ret": (tot / n) if n else None,
        })
    return out


PNL_BUCKETS = [
    (-1e9, -0.20), (-0.2, -0.15), (-0.15, -0.1),
    (-0.1, -0.07), (-0.07, -0.05), (-0.05, -0.03),
    (-0.03, -0.02), (-0.02, -0.01), (-0.01, 0.0),
    (0.0, 0.01), (0.01, 0.02), (0.02, 0.03),
    (0.03, 0.05), (0.05, 0.07), (0.07, 0.1),
    (0.1, 0.15), (0.15, 0.2), (0.20, 1e9),
]


def _pnl_dist(trades: list) -> list:
    """Trade counts by P&L bucket, over ONE population.

    Server-side for the same reason the price bins are: computed in the
    client from a trade list it would pick up whichever population that list
    held. Fixed buckets so the axis is stable across runs.
    """
    counts = [0] * len(PNL_BUCKETS)
    for t in trades:
        r = t.get("ret")
        if r is None:
            continue
        for i, (lo, hi) in enumerate(PNL_BUCKETS):
            if lo <= r < hi:
                counts[i] += 1
                break

    def _lab(lo, hi):
        # Symmetric edges so the two tails can be compared by eye -- a stop
        # should truncate the left tail and leave the right one alone, and
        # that is only readable if the buckets mirror.
        if lo <= -1e8:
            return "< -20%"
        if hi >= 1e8:
            return "> 20%"
        return f"{lo * 100:g} to {hi * 100:g}%"

    return [{"label": _lab(lo, hi), "n": n, "lo": lo}
            for (lo, hi), n in zip(PNL_BUCKETS, counts)]


def _window_stats(rets: list, holds: list, tickers, span_days: float) -> dict:
    """Full stat set for one window. Matches Recall's box set plus the three
    this page needs (Calmar, Max DD, Avg Hold).

    max_dd is peak-to-trough on the ADDITIVE cumulative return curve, the same
    convention the equity panes use; Calmar is total return over |max_dd|.
    Hold is converted bars -> sessions here so no consumer has to know the
    bar size.
    """
    n = len(rets)
    if not n:
        return {"n": 0, "n_tickers": len(tickers),
                "eff_tickers": 0.0, "avg_ret": 0.0, "median": 0.0,
                "std_dev": 0.0, "p5": 0.0, "p95": 0.0, "win_rate": 0.0, "n_win": 0,
                "avg_win": 0.0, "avg_loss": 0.0, "trades_per_year": 0.0,
                "calmar": None, "max_dd": 0.0, "avg_hold": 0.0}
    a = np.asarray(rets, dtype=np.float64)
    wins, losses = a[a > 0], a[a <= 0]
    cum = np.cumsum(a)
    peak = np.maximum.accumulate(cum)
    dd = cum - peak
    max_dd = float(dd.min()) if dd.size else 0.0
    total = float(cum[-1])
    years = max(span_days / 365.25, 1e-9)
    return {
        "n": n,
        "n_tickers": (len(tickers) if not hasattr(tickers, "values") else len(tickers)),
        # tickers is a Counter of per-ticker trade counts; a plain set
        # degrades to equal weighting, which is the right fallback.
        "eff_tickers": _effective_tickers(
            tickers.values() if hasattr(tickers, "values") else [1] * len(tickers)),
        "avg_ret":  float(a.mean()),
        "median":   float(np.median(a)),
        "std_dev":  float(a.std()),
        "p5":       float(np.percentile(a, 5)),
        "p95":      float(np.percentile(a, 95)),
        "win_rate": float((a > 0).mean()),
        "n_win":    int((a > 0).sum()),
        "avg_win":  float(wins.mean()) if wins.size else 0.0,
        "avg_loss": float(losses.mean()) if losses.size else 0.0,
        "trades_per_year": n / years,
        # Additive total and its annualisation, in RETURN units. Both were
        # already computed here for Calmar; returning them lets the batch
        # runner report Total / Avg Annual without a second derivation.
        "total_ret":  total,
        "avg_annual": total / years,
        # Calmar is ANNUALISED return over max drawdown, so the numerator is
        # total/years, not total. Undefined without a drawdown; None renders
        # as "—" rather than as a spurious infinity.
        "calmar":   ((total / years) / abs(max_dd)) if max_dd < 0 else None,
        "max_dd":   max_dd,
        "avg_hold": (float(np.mean(holds)) / BARS_PER_SESSION) if holds else 0.0,
    }


async def _load_rules(conn) -> list[dict]:
    rows = await conn.fetch(
        "SELECT rule_key, family, side, fill_mode, params, "
        "       exit_bar_col, exit_return_col, is_horizon "
        "FROM trade_path_rules ORDER BY side, family, rule_key")
    return [dict(r) for r in rows]


@router.get("/rules")
async def list_rules(pool=Depends(get_oi_pool)):
    """Exit-rule catalog for the left rail, grouped side -> family.

    Every option's column names come from the table; nothing is constructed.
    `max_days` is exposed as ordinary selectable policy including the value
    that doubles as the backstop — the backstop is a floor underneath
    whatever the user picks, not a substitute for picking. `is_horizon` is
    passed through so the UI can annotate that, not to filter it out.
    """
    if not pool:
        return {"error": "OI database not configured"}
    async with pool.acquire() as conn:
        rules = await _load_rules(conn)
    if not rules:
        return {"groups": [], "horizon_rule": HORIZON_RULE_KEY,
                "error": "trade_path_rules is empty"}

    by_side: dict[str, dict[str, list]] = {}
    for r in rules:
        fam = by_side.setdefault(r["side"], {}).setdefault(r["family"], [])
        _pr = r["params"]
        if isinstance(_pr, str):
            try:
                _pr = json.loads(_pr)
            except (ValueError, TypeError):
                _pr = {}
        fam.append({
            "rule_key":   r["rule_key"],
            "label":      _rule_label(r["family"], r["params"]),
            "is_horizon": bool(r["is_horizon"]),
            # Raw params so the rail can split a family whose rules are
            # PAIRS (trail: activation x trail distance) into one dropdown
            # per dimension. A single combined list of every pair is hard to
            # scan and makes holding one side constant awkward.
            "params":     _pr or {},
        })

    # Numeric order within every family, for every family -- not special-
    # cased to max_days, because any family whose options are numbers has
    # the same problem the moment it grows a two-digit value.
    for fams in by_side.values():
        for rs in fams.values():
            rs.sort(key=lambda r: _rule_sort_key(r["params"]))

    groups = []
    for side, title in SIDE_GROUPS:
        fams = by_side.pop(side, None)
        if not fams:
            continue
        groups.append({
            "side": side, "title": title,
            "families": [{"family": f, "rules": rs} for f, rs in sorted(fams.items())],
        })
    # Any side the registry grows that this UI does not know about still
    # renders, rather than silently vanishing from the rail.
    for side, fams in sorted(by_side.items()):
        groups.append({
            "side": side, "title": side.title(),
            "families": [{"family": f, "rules": rs} for f, rs in sorted(fams.items())],
        })

    return {"groups": groups, "horizon_rule": HORIZON_RULE_KEY,
            "n_rules": len(rules)}


async def _bin_columns(conn) -> set[str]:
    rows = await conn.fetch(
        "SELECT column_name FROM information_schema.columns "
        "WHERE table_name = 'tt_bins' AND table_schema = 'public' "
        "  AND column_name LIKE 'bin20_%'")
    return {r["column_name"] for r in rows}


# ── Portfolio mode ────────────────────────────────────────────────────────
#
# A portfolio is the deduped union of several saved signals, traded under ONE
# exit policy. The argument for it is that exit policy is a portfolio-level
# decision: when three signals put the same ticker on today's list you trade
# it once, so a per-signal exit is not a thing that could be traded.
#
# The union is expressed as an OR of per-signal cell predicates against the
# SAME tt_bins row, which is what makes dedup structural rather than a pass
# someone has to remember: one trade_paths row joins one tt_bins row, so an OR
# over that row cannot emit it twice. There is no dedup step to forget.
#
# Signals carry no exit information and none is invented for them here.

# JS bitwise operators are 32-bit and signed, so bit 31 would go negative in
# the client. 31 signals is the hard ceiling; the practical use is 8-10.
MAX_PORTFOLIO_SIGNALS = 31

# The only entry the exit paths can honour. daily_features forward returns are
# either _oc (enter at T's 9:30 open) or _cc (enter at T-1's close); trade_paths
# has anchors 'open' and 't1000' and nothing at the prior close. A _cc signal
# would silently be entered a half-session later than it was selected under, so
# it is refused rather than quietly re-anchored.
_OC_OUTCOME_SUFFIX = "_oc"

# Metric names are interpolated into SQL (they are column names, which cannot
# be parameters), so they are whitelisted rather than trusted. Same character
# set the Factor Analysis side validates against.
_SAFE_METRIC_NAME = set("abcdefghijklmnopqrstuvwxyz_0123456789")

# per_cell_stats / status / color_slot drive the SAME thumbnail and swatch the
# Factor Analysis Saved Signals pane draws (window.SignalThumb), so the picker
# shows the record the user already knows how to read rather than a summary of
# it. Selected here, not on the hot path: this endpoint runs once on page load.
_SIGNAL_COLUMNS = """id, name, primary_metric, secondary_metric, outcome,
                     n_bins, cell_set, agg_n, agg_avg_ret, per_cell_stats,
                     stats_updated_at, status, color_slot, corner,
                     selection_mode, selection_cutoff"""


def _collapse_expr(col: str, n_bins: int) -> str:
    """bin20 column -> 0-based cell index at n_bins resolution.

    THE definition, in one place. It is the exact integer formula the Factor
    Analysis heatmap renderer and /secondary-zone-analyze use, so a zone drawn
    there addresses the same cells here. LEAST is a no-op for bin20 in 1..20
    and exists only to make an out-of-range stored bin fail closed.
    """
    return f"LEAST(((bt.{col} - 1) * {n_bins}) / 20, {n_bins} - 1)"


def ft_parse_jsonb(v):
    """JSONB comes back as a parsed list or as a string depending on the
    asyncpg codec in play. Both shapes, one reader."""
    if v is None:
        return None
    if isinstance(v, (list, dict)):
        return v
    try:
        return json.loads(v)
    except (TypeError, ValueError):
        return None


def _signal_cells(sig) -> list:
    """cell_set as a list of [ix, iy], tolerating asyncpg's two JSONB shapes."""
    raw = sig["cell_set"]
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except (TypeError, ValueError):
            return []
    return raw or []


def _signal_ineligible(sig, tt_cutoff: str, bin_cols: set) -> Optional[str]:
    """Why this saved signal cannot be traded on this page, or None if it can.

    ONE definition of eligibility, server-side, so the checkbox the user sees
    disabled and the request the server would refuse cannot disagree. The
    reason is returned as prose because it is shown next to the signal — an
    older signal missing from the list is otherwise indistinguishable from a
    signal that was deleted.
    """
    mode = sig["selection_mode"] or "in_sample"
    if mode != "train_test":
        # is_bins edges are full-history quantiles re-derived on every pipeline
        # run; tt_bins edges are frozen at the cutoff. Same cell coordinates,
        # different boundaries, so the zone would resolve to a different set of
        # entries here than on the page it was saved from. Silent, and wrong.
        return (f"selected on {mode} bins — only train/test signals resolve to "
                f"the same trades on this page")

    cut = sig["selection_cutoff"]
    cut = cut.isoformat() if hasattr(cut, "isoformat") else (str(cut) if cut else None)
    if not cut:
        return "train/test signal with no stored split — cannot be verified against tt_bins"
    if cut != tt_cutoff:
        # The stored cutoff and the live one agree today. They stop agreeing
        # the moment the pipeline re-freezes tt_bins, and then the cells this
        # signal was drawn on are not the cells that would be resolved.
        return (f"saved against split {cut}, but tt_bins is now frozen at "
                f"{tt_cutoff} — the bins have been rebuilt since it was saved")

    outcome = sig["outcome"] or ""
    if not outcome.endswith(_OC_OUTCOME_SUFFIX):
        return (f"selected on {outcome}, which does not enter at the open — "
                f"this page can only enter at the open of trade_date")

    n_bins = int(sig["n_bins"] or 0)
    if n_bins not in (3, 5, 10, 20):
        return f"unsupported bin resolution {n_bins}"

    prim, sec = sig["primary_metric"], sig["secondary_metric"]
    if not sec:
        return "single-metric signal — this page's portfolio mode expects a metric pair"
    for m in (prim, sec):
        if any(c not in _SAFE_METRIC_NAME for c in (m or "")):
            return f"unsafe metric name {m!r}"
        if f"bin20_{m}" not in bin_cols:
            return f"no stored bins for {m!r} in tt_bins"

    cells = _signal_cells(sig)
    if not cells:
        return "empty cell set"
    for c in cells:
        if (not c or len(c) < 2 or not (0 <= int(c[0]) < n_bins)
                or not (0 <= int(c[1]) < n_bins)):
            return f"cell {c} out of range for n_bins={n_bins}"
    return None


def _portfolio_selection(sigs: list, base: int) -> tuple:
    """(where_bins, cell_pred, mask_sql, params) for a set of saved signals.

    `base` is how many placeholders the caller has already used ($1 anchor,
    $2 cutoff), so this numbers its own from base+1 rather than assuming.

    THE SAME PREDICATE STRING is used twice: once in the OR chain that selects
    rows, once in the CASE arm that tags them. That is deliberate and is the
    whole reason the mask can be trusted — there is one definition of "signal i
    fired here", so the tag cannot come to disagree with the filter.

    The tag is a BITMASK, not one column per signal. Signal i owns bit i, so
    the mask carries three things the client would otherwise need three
    queries for: membership (mask & 1<<i), how many signals fired
    (popcount), and WHICH combination fired (the mask value itself is the
    combination's identity). One small int per row instead of N columns.
    """
    preds, arms, params = [], [], []
    p = base
    for bit, sig in enumerate(sigs):
        n = int(sig["n_bins"])
        prim, sec = sig["primary_metric"], sig["secondary_metric"]
        cells = _signal_cells(sig)
        xs = [int(c[0]) for c in cells]
        ys = [int(c[1]) for c in cells]
        params.append(xs); params.append(ys)
        ix, iy = p + 1, p + 2
        p += 2
        pred = (f"(bt.bin20_{prim} > 0 AND bt.bin20_{sec} > 0"
                f" AND ({_collapse_expr(f'bin20_{prim}', n)},"
                f" {_collapse_expr(f'bin20_{sec}', n)})"
                f" IN (SELECT * FROM unnest(${ix}::int[], ${iy}::int[])))")
        preds.append(pred)
        arms.append(f"CASE WHEN {pred} THEN {1 << bit} ELSE 0 END")

    # The per-signal bin-present guards live inside each disjunct, because in a
    # portfolio they differ per signal. where_bins is therefore already
    # satisfied and stays TRUE rather than being dropped, so every call site
    # that interpolates it is untouched.
    return "TRUE", "(" + " OR ".join(preds) + ")", "(" + " + ".join(arms) + ")", params


async def _load_signals(conn, ids: list) -> dict:
    rows = await conn.fetch(
        f"SELECT {_SIGNAL_COLUMNS} FROM signals WHERE id = ANY($1::int[])", ids)
    return {r["id"]: r for r in rows}


@router.get("/signals")
async def list_portfolio_signals(pool=Depends(get_oi_pool)):
    """Saved signals, annotated with whether this page can trade them.

    EVERY signal is returned, including the ones that cannot be selected, each
    carrying the reason. A signal that simply vanished from the list is
    indistinguishable from one that was deleted, and the difference matters:
    the first is a property of this page, the second is a thing the user did.

    Eligibility is decided HERE and not in the client, so the disabled checkbox
    and the request the server would refuse cannot come to disagree.
    """
    if not pool:
        return {"error": "OI database not configured", "signals": []}

    from app.routers.oi_analysis import _get_tt_cutoff

    cutoff = await _get_tt_cutoff(pool)
    if not cutoff:
        return {"error": "tt_bins has no cutoff_date", "signals": []}

    async with pool.acquire() as conn:
        bin_cols = await _bin_columns(conn)
        rows = await conn.fetch(
            f"SELECT {_SIGNAL_COLUMNS} FROM signals ORDER BY created_at DESC")

    out = []
    for r in rows:
        reason = _signal_ineligible(r, cutoff, bin_cols)
        cut = r["selection_cutoff"]
        out.append({
            "id":               r["id"],
            "name":             r["name"],
            "primary_metric":   r["primary_metric"],
            "secondary_metric": r["secondary_metric"],
            "outcome":          r["outcome"],
            "n_bins":           r["n_bins"],
            "n_cells":          len(_signal_cells(r)),
            # The Factor Analysis n: daily_features universe, train window.
            # NOT this page's n, which is the trade_paths subset after
            # max_strike. Named so it reads as a picking aid and cannot be
            # mistaken for a promise about what the stat bar will say.
            "agg_n":            int(r["agg_n"]) if r["agg_n"] is not None else None,
            "agg_avg_ret":      float(r["agg_avg_ret"]) if r["agg_avg_ret"] is not None else None,
            # Read by window.SignalThumb.thumbnailSVG -- the same function and
            # the same fields the Factor Analysis pane passes it.
            "per_cell_stats":   ft_parse_jsonb(r["per_cell_stats"]) or [],
            "stats_updated_at": str(r["stats_updated_at"])[:19] if r["stats_updated_at"] else None,
            "status":           r["status"] or "Test",
            "color_slot":       r["color_slot"],
            "corner":           r["corner"],
            "selection_mode":   r["selection_mode"] or "in_sample",
            "selection_cutoff": (cut.isoformat() if hasattr(cut, "isoformat")
                                 else (str(cut) if cut else None)),
            "eligible":         reason is None,
            "reason":           reason,
        })
    return {"signals": out, "cutoff_date": cutoff,
            "max_selectable": MAX_PORTFOLIO_SIGNALS}


async def _resolve_selection(conn, req, bin_cols, base: int, cutoff: str) -> tuple:
    """(where_bins, cell_pred, mask_sql, params, provenance) for either mode.

    ONE resolver for /zone, /grid and /suite, so the three cannot come to
    disagree about what the selected population is. Returns a dict as the
    fifth element on success, or a string as the sole element of a 1-tuple on
    failure -- callers check `len(...) == 1`.

    mask_sql is None in signal mode: there is only one signal, so there is
    nothing to attribute and no column is emitted.
    """
    if req.signal_ids:
        ids = list(dict.fromkeys(int(i) for i in req.signal_ids))
        if len(ids) > MAX_PORTFOLIO_SIGNALS:
            return (f"{len(ids)} signals selected; the maximum is "
                    f"{MAX_PORTFOLIO_SIGNALS}",)

        by_id = await _load_signals(conn, ids)
        missing = [i for i in ids if i not in by_id]
        if missing:
            return (f"saved signal(s) {missing} no longer exist",)

        # Ordered by the REQUEST, not by the table, because bit i of the mask
        # is signal i of this list and the client reads it positionally.
        sigs = [by_id[i] for i in ids]
        for sig in sigs:
            reason = _signal_ineligible(sig, cutoff, bin_cols)
            if reason:
                return (f"signal {sig['name']!r} cannot be traded here: {reason}",)

        where_bins, cell_pred, mask_sql, params = _portfolio_selection(sigs, base)
        prov = {
            "mode": "portfolio",
            "signals": [{"id": s["id"], "name": s["name"], "bit": b,
                         "primary_metric": s["primary_metric"],
                         "secondary_metric": s["secondary_metric"],
                         "n_bins": s["n_bins"], "n_cells": len(_signal_cells(s))}
                        for b, s in enumerate(sigs)],
        }
        return where_bins, cell_pred, mask_sql, params, prov

    # ── Signal mode: exactly what existed before ──────────────────────────
    #
    # EVERY "nothing is selected" message below names the control the user
    # would actually use, and which mode it thinks it is in. The generic
    # "no cells selected" that used to guard these endpoints fired in
    # PORTFOLIO mode too, where there are no cells and no heatmap to click --
    # it pointed at a control that is not on screen.
    if not req.primary_metric:
        return ("no selection: pick a metric and click heatmap cells for a "
                "single zone, or switch to Portfolio and tick saved signals",)
    p_col = f"bin20_{req.primary_metric}"
    if p_col not in bin_cols:
        return (f"no stored bins for {req.primary_metric!r} in tt_bins",)
    two_factor = bool(req.secondary_metric)
    s_col = f"bin20_{req.secondary_metric}" if two_factor else None
    if two_factor and s_col not in bin_cols:
        return (f"no stored bins for {req.secondary_metric!r} in tt_bins",)

    n_bins = max(2, min(20, int(req.n_bins)))
    bps, bss = [], []
    for c in req.cells:
        if not c:
            continue
        bp = int(c[0])
        bs = int(c[1]) if (two_factor and len(c) > 1) else 0
        if not (0 <= bp < n_bins) or not (0 <= bs < n_bins):
            return (f"cell out of range for n_bins={n_bins}: {c}",)
        bps.append(bp); bss.append(bs)
    if not bps:
        return ("no cells selected — click cells on the heatmap to define a "
                "zone (signal mode); portfolio mode takes its cells from the "
                "saved signals instead",)

    if two_factor:
        cell_pred = (f"({_collapse_expr(p_col, n_bins)}, "
                     f"{_collapse_expr(s_col, n_bins)}) "
                     f"IN (SELECT * FROM unnest(${base+1}::int[], ${base+2}::int[]))")
        params = [bps, bss]
    else:
        cell_pred = f"{_collapse_expr(p_col, n_bins)} = ANY(${base+1}::int[])"
        params = [bps]
    where_bins = f"bt.{p_col} > 0" + (f" AND bt.{s_col} > 0" if two_factor else "")
    return where_bins, cell_pred, None, params, {
        "mode": "2f" if two_factor else "1f", "signals": []}


class RunReq(BaseModel):
    # Optional because portfolio mode selects by signal instead. Exactly one of
    # (primary_metric, signal_ids) is expected; the handlers say which they got
    # rather than defaulting, so a request that supplies neither is an error
    # with a name and not an empty grid.
    primary_metric:   Optional[str] = None
    secondary_metric: Optional[str] = None      # None => single-metric mode
    # PORTFOLIO MODE. Non-empty => the trade set is the deduped union of these
    # saved signals and the metric pickers are ignored. The signals are the
    # ones saved on the Factor Analysis page; there is no second save
    # mechanism here, so there is one store and nothing to drift.
    signal_ids:       list[int] = []
    entry_anchor:     str = "open"
    rule_keys:        list[str] = []
    n_bins:           int = 20
    # Filters the TRADE POPULATION, so it must be applied here rather than in
    # the client's sizing pass. A trade above this price is not taken, and
    # nothing downstream -- grid, stats, exit reasons, ticker breakdown --
    # should ever see it. Applying it client-side made the stat bar report N
    # over the full set while Total Ret covered a filtered subset, two numbers
    # that cannot be reconciled and no way to tell which population a given
    # box covered.
    max_strike:       Optional[float] = None
    # Page-level window. Everything except the heatmap reports on it: the
    # stat bar, exit reasons, ticker breakdown, price bins, and the
    # time-series panes, which simply stop at the cutoff in train mode
    # because no test trade is returned at all.
    window:           str = "train"
    label:            Optional[str] = None


@router.post("/run")
async def run(req: RunReq = Body(...), pool=Depends(get_oi_pool)):
    """Recompute the grid and stats under one exit policy.

    Returns the TRAIN-window grid (selection happens on train; test is the
    verdict and surfaces in the stat rows), the exit-reason breakdown, and
    the run's provenance.

    `horizon_auto_added` and `excludes_unresolved` are returned because both
    silently change what the numbers mean: the first means a policy that
    never fires is really the 20-day horizon, the second means unresolved
    paths were dropped and the trade count is not the raw universe.
    """
    if not pool:
        return {"error": "OI database not configured"}
    # /run exists to build the 20x20 grid, and a grid is a property of ONE
    # metric pair. A portfolio spans several pairs at several resolutions, so
    # there is no single grid to draw and portfolio mode goes straight to
    # /zone instead. Refused by name rather than served an empty heatmap.
    if req.signal_ids:
        return {"error": "portfolio mode has no heatmap — call /zone with "
                         "signal_ids instead of /run"}
    if not req.primary_metric:
        return {"error": "primary_metric is required"}

    from app.routers.oi_analysis import _get_tt_cutoff

    async with pool.acquire() as conn:
        rules = await _load_rules(conn)
        if not rules:
            return {"error": "trade_path_rules is empty"}
        by_key = by_key_from_rows(rules)
        meta_by_key = {r["rule_key"]: r for r in rules}

        bin_cols = await _bin_columns(conn)
        p_col = f"bin20_{req.primary_metric}"
        if p_col not in bin_cols:
            return {"error": f"no stored bins for primary metric "
                             f"{req.primary_metric!r} in tt_bins"}
        two_factor = bool(req.secondary_metric)
        s_col = f"bin20_{req.secondary_metric}" if two_factor else None
        if two_factor and s_col not in bin_cols:
            return {"error": f"no stored bins for secondary metric "
                             f"{req.secondary_metric!r} in tt_bins"}

        try:
            combine_sql, combine_meta = build_combine_sql(
                req.rule_keys, by_key, include_exit_rule=True)
        except CombineError as e:
            return {"error": str(e)}

        # _get_tt_cutoff returns an ISO STRING by contract. asyncpg does not
        # coerce str -> date for a DATE parameter; it raises DataError
        # ("'str' object has no attribute 'toordinal'") at query time. Keep
        # the two forms explicitly named: *_iso for JSON and cache keys,
        # *_d for anything handed to conn.fetch.
        cutoff_iso = await _get_tt_cutoff(pool)
        if not cutoff_iso:
            return {"error": "tt_bins has no cutoff_date — cannot split train/test"}
        try:
            cutoff_d = _date.fromisoformat(cutoff_iso)
        except (TypeError, ValueError):
            return {"error": f"tt_bins cutoff_date is not an ISO date: {cutoff_iso!r}"}

        # Applied to entry_price, which trade_paths stores AS-TRADED -- the
        # price a fill would actually have happened at.
        strike_pred, strike_args = "", []
        if req.max_strike and req.max_strike > 0:
            strike_pred = " AND tp.entry_price <= $3"
            strike_args = [float(req.max_strike)]

        n_bins = max(2, min(20, int(req.n_bins)))
        # Canonical bin20 collapse, identical to every other stored-bin
        # surface in this codebase -- now literally the same function.
        def _collapse(col: str) -> str:
            return _collapse_expr(col, n_bins)

        sel_bins = f"{_collapse(p_col)} AS bp"
        grp = "bp"
        where_bins = f"bt.{p_col} > 0"
        if two_factor:
            sel_bins += f", {_collapse(s_col)} AS bs"
            grp = "bp, bs"
            where_bins += f" AND bt.{s_col} > 0"

        # One pass: per-cell aggregates split into train/test, plus the
        # exit-rule attribution. $1 = entry_anchor, $2 = cutoff.
        sql = f"""
        WITH c AS (
{combine_sql}
        )
        SELECT {sel_bins},
               c.exit_rule,
               (c.trade_date < $2::date) AS is_train,
               COUNT(*)            AS n,
               AVG(c.exit_return)  AS avg_ret,
               AVG(c.exit_bar)     AS avg_hold
        FROM c
        JOIN tt_bins bt USING (ticker, trade_date)
        JOIN trade_paths tp USING (ticker, trade_date, entry_anchor)
        WHERE c.entry_anchor = $1 AND {where_bins}{strike_pred}
        GROUP BY {grp}, c.exit_rule, is_train
        """
        rows = await conn.fetch(sql, req.entry_anchor, cutoff_d, *strike_args)

        # Median / percentiles / max-DD cannot be derived from the grouped
        # aggregate above, so the overall stat set comes from a second pass
        # that returns per-trade returns without the 400-way bin grouping.
        stat_sql = f"""
        WITH c AS (
{combine_sql}
        )
        SELECT c.ticker, c.trade_date, c.exit_return, c.exit_bar,
               (c.trade_date < $2::date) AS is_train
        FROM c
        JOIN tt_bins bt USING (ticker, trade_date)
        JOIN trade_paths tp USING (ticker, trade_date, entry_anchor)
        WHERE c.entry_anchor = $1 AND {where_bins}{strike_pred}
        ORDER BY c.trade_date
        """
        srows = await conn.fetch(stat_sql, req.entry_anchor, cutoff_d, *strike_args)

    # ── Fold into grid + breakdown ────────────────────────────────────────
    grid = [[None] * n_bins for _ in range(n_bins if two_factor else 1)]
    acc: dict = {}
    reasons: dict = {}
    tot = {"train": [0, 0.0, 0.0], "test": [0, 0.0, 0.0]}   # n, sum_ret, sum_hold
    active = "train" if (req.window or "train") == "train" else "test"
    # TWO POPULATIONS, named by purpose so they cannot be confused:
    #
    #   window_trades  strictly the active window. Every single-window number
    #                  is computed FROM IT SERVER-SIDE and shipped as a
    #                  finished aggregate -- stats, exit reasons, ticker
    #                  breakdown, effective tickers, price bins. No client
    #                  pane ever sees a list it could widen.
    #   series_trades  what the three time-series panes draw: train-only in
    #                  TRAIN, the FULL history in TEST so the whole record and
    #                  the cutoff are both visible.
    #
    # In TRAIN the two are the same population, which is the invariant worth
    # asserting downstream: the equity endpoint must equal Total Ret there.
    for r in rows:
        win = "train" if r["is_train"] else "test"
        bp, bs = int(r["bp"]), (int(r["bs"]) if two_factor else 0)
        n, avg, hold = int(r["n"]), float(r["avg_ret"] or 0), float(r["avg_hold"] or 0)
        cell = acc.setdefault((bs, bp), {"train": [0, 0.0], "test": [0, 0.0]})
        cell[win][0] += n
        cell[win][1] += avg * n
        t = tot[win]
        t[0] += n; t[1] += avg * n; t[2] += hold * n
        if win == active:
            rk = r["exit_rule"]
            reasons[rk] = reasons.get(rk, 0) + n

    for (bs, bp), c in acc.items():
        if 0 <= bs < len(grid) and 0 <= bp < n_bins:
            tr_n, tr_s = c["train"]
            te_n, te_s = c["test"]
            grid[bs][bp] = {
                "n":          tr_n,
                "avg_ret":    (tr_s / tr_n) if tr_n else 0.0,
                "test_n":     te_n,
                "test_avg":   (te_s / te_n) if te_n else 0.0,
            }

    from collections import Counter as _Ctr2
    acc2 = {"train": ([], [], _Ctr2(), []), "test": ([], [], _Ctr2(), [])}
    for r in srows:
        w = "train" if r["is_train"] else "test"
        rets, holds, tks, dates = acc2[w]
        rets.append(float(r["exit_return"] or 0.0))
        holds.append(float(r["exit_bar"] or 0.0))
        tks[r["ticker"]] += 1; dates.append(r["trade_date"])

    def _stats(win: str) -> dict:
        rets, holds, tks, dates = acc2[win]
        span = ((max(dates) - min(dates)).days if len(dates) > 1 else 1)
        return _window_stats(rets, holds, tks, span)

    # Exit-reason breakdown. A user-selected max_days and the auto-appended
    # backstop are the SAME column with opposite meanings, so the backstop is
    # labelled as such ONLY when it was auto-added.
    # Same page-level window as /zone.
    active = "train" if (req.window or "train") == "train" else "test"
    tot_active = tot[active][0] or 1
    breakdown = []
    for rk, n in sorted(reasons.items(), key=lambda kv: -kv[1]):
        m = meta_by_key.get(rk, {})
        is_backstop = (rk == HORIZON_RULE_KEY and combine_meta["horizon_auto_added"])
        breakdown.append({
            "rule_key":    rk,
            "family":      m.get("family", rk),
            "side":        m.get("side", ""),
            "label":       ("backstop — no selected rule fired" if is_backstop
                            else f"{m.get('family', rk)} {_rule_label(m.get('family',''), m.get('params'))}"),
            "is_backstop": is_backstop,
            "n":           n,
            "frac":        n / tot_active,
        })

    return {
        "mode":             "2f" if two_factor else "1f",
        "primary_metric":   req.primary_metric,
        "secondary_metric": req.secondary_metric,
        "entry_anchor":     req.entry_anchor,
        "max_strike":       req.max_strike,
        # Echoed so a locked run carries its own window: comparing a TRAIN
        # lock against a TEST run would cross populations silently.
        "window":           active,
        "n_bins":           n_bins,
        "cutoff_date":      cutoff_iso,
        "grid":             grid,
        "train":            _stats("train"),
        "test":             _stats("test"),
        "exit_reasons":     breakdown,
        # Provenance — both of these change what the numbers mean and belong
        # on the run card, not buried here.
        "rules":               combine_meta["rules"],
        "horizon_rule":        combine_meta["horizon_rule"],
        "horizon_auto_added":  combine_meta["horizon_auto_added"],
        "excludes_unresolved": combine_meta["excludes_unresolved"],
        "tie_break_order":     combine_meta["tie_break_order"],
        "label":               req.label,
    }


# Fixed by default so a baseline does not move under the user while they
# iterate on exit rules. See the seed discussion in _random_zone_sql.
DEFAULT_BASELINE_SEED = 20240701


class ZoneReq(RunReq):
    """A run plus the selected cells. Cells are [bp, bs] pairs at the run's
    n_bins resolution (bs is ignored in single-metric mode)."""
    cells: list[list[int]] = []
    # Random-entry baseline: run this exact exit policy against randomly
    # chosen entries instead of the zone's, matched on count and on date
    # distribution. Answers "is this policy signal-specific, or would any
    # long position under these exits look like this".
    randomize: bool = False
    seed: Optional[int] = None
    # "entry"      random ticker, real exit policy  — does the metric pick
    #              better names than chance on the same days?
    # "entry_exit" random ticker AND a holding period drawn from the policy's
    #              own distribution — do the exit rules add anything over
    #              being in the market for a similar duration?
    baseline_kind: str = "entry"


def _maxdays_rules(rules: list[dict]) -> list[tuple[int, str]]:
    """[(n_sessions, exit_return_col)] for the max_days family, ascending.

    Read from the catalog, never constructed: the available holding periods
    are whatever was precomputed, and assuming 1..20 would silently produce
    a CASE arm referencing a column that does not exist.

    max_days is in TRADING DAYS -- N exits at the close of the (N-1)th
    session after entry, so max_days=1 is a one-session hold.
    """
    out: list[tuple[int, str]] = []
    for r in rules:
        if r.get("family") != "max_days":
            continue
        pr = r.get("params")
        if isinstance(pr, str):
            try:
                pr = json.loads(pr)
            except (ValueError, TypeError):
                pr = {}
        pr = pr or {}
        n = pr.get("n") or pr.get("days") or pr.get("bars")
        col = r.get("exit_return_col")
        if n is None or not col:
            continue
        out.append((int(n), col))
    out.sort()
    return out


def _hold_cdf(sess_counts: dict[int, int],
              avail: list[int]) -> tuple[list[int], list[float], list[float]]:
    """Empirical holding-period distribution, snapped to available max_days.

    Returns (n_days, lo, hi) as parallel arrays forming a half-open inverse
    CDF over [0, 1), so a uniform draw picks a holding period with the same
    frequency the real policy produced it.

    Snapping is to the NEAREST available N rather than the floor: with a
    coarse catalog, flooring would bias every hold downward and the baseline
    would systematically under-hold relative to the policy it is standing in
    for -- which is the one thing this baseline must not do, since holding
    period is exactly what it is controlling for.
    """
    if not avail:
        return [], [], []
    weights: dict[int, int] = {}
    for sess, cnt in sess_counts.items():
        near = min(avail, key=lambda a: (abs(a - sess), a))
        weights[near] = weights.get(near, 0) + cnt
    total = float(sum(weights.values()))
    if total <= 0:
        return [], [], []
    ns, los, his = [], [], []
    acc = 0.0
    for n in sorted(weights):
        lo = acc / total
        acc += weights[n]
        ns.append(n); los.append(lo); his.append(acc / total)
    # The top bucket is closed at just over 1.0 so a draw of exactly 1.0
    # cannot fall through every half-open interval and lose its trade.
    his[-1] = 1.0000001
    return ns, los, his


# Seed suffixes. The ENTRY ordering hash is deliberately identical to the
# entry-only baseline's, so both baselines pick the SAME tickers on the same
# dates from the same seed. Their difference is then purely the exit rule,
# which is what decomposes signal selection from exit policy -- the whole
# reason this second baseline exists. The exit draw takes its own suffix so
# it is independent of the entry draw rather than a function of it.
_EXIT_SALT = "|exit"


def _u01(expr: str) -> str:
    """A uniform [0,1) draw from a text expression, deterministic per seed.

    md5 -> first 8 hex -> bit(32) -> bigint, masked to 31 bits because the
    signed cast makes the top bit negative and a negative u would fall
    outside every CDF interval.
    """
    return (f"((('x' || substr(md5({expr}), 1, 8))::bit(32)::bigint)"
            f" & 2147483647)::float8 / 2147483647.0")


def _random_exit_zone_sql(combine_sql: str, ret_case: str, strike_pred: str) -> str:
    """Random entry AND random exit: the layer beneath the entry baseline.

    The entry-only baseline holds timing and exit policy constant and
    randomises ticker, so it isolates "does the metric pick better names
    than chance on the same days". This one additionally replaces the exit
    rules with a holding period drawn from the distribution the real policy
    produces, so the difference between the two baselines isolates whether
    the exit rules add anything over simply being in the market for a
    similar duration. Those two effects are confounded without it.

    exit_bar is written as n_days * BARS_PER_SESSION so avg DIT, the
    activity pane and the capital-tied series all stay in the same units as
    a real run rather than needing a special case downstream.
    """
    return f"""
    WITH c AS (
{combine_sql}
    ),
    want AS (
        SELECT * FROM unnest($3::date[], $4::int[]) AS t(trade_date, k)
    ),
    cdf AS (
        SELECT * FROM unnest($6::int[], $7::float8[], $8::float8[])
                     AS t(n_days, lo, hi)
    ),
    elig AS (
        SELECT c.ticker, c.trade_date, tp.entry_price,
               row_number() OVER (
                   PARTITION BY c.trade_date
                   ORDER BY md5(c.ticker || '|' || c.trade_date::text
                                || '|' || $5::text)
               ) AS rn,
               {_u01("c.ticker || '|' || c.trade_date::text || '|' || $5::text"
                     " || '" + _EXIT_SALT + "'")} AS u
        FROM c
        JOIN trade_paths tp USING (ticker, trade_date, entry_anchor)
        JOIN want w ON w.trade_date = c.trade_date
        WHERE c.entry_anchor = $1{strike_pred}
    ),
    picked AS (
        SELECT e.* FROM elig e
        JOIN want w ON w.trade_date = e.trade_date
        WHERE e.rn <= w.k
    )
    SELECT p.ticker, p.trade_date,
           (d.n_days * {BARS_PER_SESSION})::float8 AS exit_bar,
           {ret_case} AS exit_return,
           'max_days__' || d.n_days::text AS exit_rule,
           p.entry_price,
           (p.trade_date < $2::date) AS is_train
    FROM picked p
    JOIN cdf d ON p.u >= d.lo AND p.u < d.hi
    -- Rejoined for the return columns: the sampled holding period is not
    -- known until the cdf join, so the CASE cannot be evaluated earlier.
    JOIN trade_paths t2 ON t2.ticker = p.ticker
                       AND t2.trade_date = p.trade_date
                       AND t2.entry_anchor = $1
    ORDER BY p.trade_date, p.ticker
    """


def _random_exit_only_sql(combine_sql: str, where_bins: str, cell_pred: str,
                          strike_pred: str, ret_case: str,
                          p_seed: int, p_ns: int, p_lo: int, p_hi: int) -> str:
    """The signal's REAL entries with a randomly drawn holding period.

    Completes the 2x2. With policy = signal entries + rule exits:

        policy   vs  exit-only    -> do my exit rules add value ON MY TRADES
        policy   vs  entry-only   -> does my selection beat chance
        both     vs  entry+exit   -> the combined effect over the floor

    Exit-only is the more direct question about exit rules than entry+exit
    is, because it asks it against the trades actually being taken rather
    than against random ones.

    Uses the SAME exit salt as _random_exit_zone_sql, so a given
    (ticker, date) draws the same holding period in both and the two
    random-exit runs differ only by entry selection -- the same principle as
    sharing the entry draw between the two random-entry runs. Note what that
    does and does not claim: the two runs hold different TRADES, so their
    drawn periods are not trade-for-trade identical; what is identical is the
    drawing rule and the distribution, so no part of the gap between them is
    a different exit draw.

    Placeholder positions are passed in because this query reuses the zone
    query's own cell and strike predicates, whose numbers are computed
    against that arg list. The new parameters are APPENDED after it rather
    than renumbering those fragments.
    """
    u = _u01(f"c.ticker || '|' || c.trade_date::text || '|' || ${p_seed}::text"
             f" || '{_EXIT_SALT}'")
    return f"""
    WITH c AS (
{combine_sql}
    ),
    cdf AS (
        SELECT * FROM unnest(${p_ns}::int[], ${p_lo}::float8[], ${p_hi}::float8[])
                     AS t(n_days, lo, hi)
    ),
    base AS (
        SELECT c.ticker, c.trade_date, tp.entry_price, {u} AS u
        FROM c
        JOIN tt_bins bt USING (ticker, trade_date)
        JOIN trade_paths tp USING (ticker, trade_date, entry_anchor)
        WHERE c.entry_anchor = $1 AND {where_bins} AND {cell_pred}{strike_pred}
    )
    SELECT b.ticker, b.trade_date,
           (d.n_days * {BARS_PER_SESSION})::float8 AS exit_bar,
           {ret_case} AS exit_return,
           'max_days__' || d.n_days::text AS exit_rule,
           b.entry_price,
           (b.trade_date < $2::date) AS is_train
    FROM base b
    JOIN cdf d ON b.u >= d.lo AND b.u < d.hi
    JOIN trade_paths t2 ON t2.ticker = b.ticker
                       AND t2.trade_date = b.trade_date
                       AND t2.entry_anchor = $1
    ORDER BY b.trade_date, b.ticker
    """


def _zone_count_sql(combine_sql: str, where_bins: str,
                    cell_pred: str, strike_pred: str) -> str:
    """Per-date trade counts for the real zone -- the shape a baseline matches.

    Takes the ZONE QUERY'S OWN arg list unchanged, because cell_pred and
    strike_pred are that query's fragments and their placeholder numbers are
    computed against it. Renumbering them for a shorter list is exactly how
    the two populations would drift apart, so the list stays identical and
    this query simply does not need $2.

    A parameter that appears NOWHERE in a statement has no typed context at
    all, and Postgres raises IndeterminateDatatypeError ("could not determine
    data type of parameter $2") rather than ignoring it. The zone query gets
    $2's type from `(c.trade_date < $2::date) AS is_train` in its SELECT; that
    expression is gone here, so the cast has to be stated explicitly.
    """
    return f"""
    WITH c AS (
{combine_sql}
    )
    SELECT c.trade_date, COUNT(*) AS k
    FROM c
    JOIN tt_bins bt USING (ticker, trade_date)
    JOIN trade_paths tp USING (ticker, trade_date, entry_anchor)
    WHERE c.entry_anchor = $1 AND $2::date IS NOT NULL
          AND {where_bins} AND {cell_pred}{strike_pred}
    GROUP BY c.trade_date
    """


def _random_zone_sql(combine_sql: str, strike_pred: str) -> str:
    """Sample the SAME NUMBER of trades per date as the real zone, at random.

    Matching on the date distribution is the whole point. Sampling uniformly
    across the window would draw a different market-exposure profile -- a
    zone that fires 40 times in March 2020 and 3 times in July 2021 has to be
    compared against a random set with that same shape, or the comparison is
    between two different markets rather than between two entry choices.

    Ticker is deliberately NOT matched: it is the thing being randomised.

    THE COUNT IS EXACT, NOT BEST-EFFORT. The eligible universe on a date is
    (combine JOIN trade_paths); the real zone is that same set intersected
    with tt_bins bins-present and the selected cells. The zone is therefore a
    SUBSET of the universe on every date it appears on, so k picks always
    exist. The caller still verifies the total rather than trusting the
    argument.

    Ordering is md5(ticker | date | seed) rather than random(): random()
    would need setseed() on a pooled connection, which is per-session state
    that another request can clobber. The hash is deterministic, needs no
    session state, and has a useful property -- changing k re-uses the same
    ordering, so a bigger zone's sample is a superset of a smaller one's
    rather than an unrelated draw.
    """
    return f"""
    WITH c AS (
{combine_sql}
    ),
    want AS (
        SELECT * FROM unnest($3::date[], $4::int[]) AS t(trade_date, k)
    ),
    elig AS (
        SELECT c.ticker, c.trade_date, c.exit_bar, c.exit_return, c.exit_rule,
               tp.entry_price,
               row_number() OVER (
                   PARTITION BY c.trade_date
                   ORDER BY md5(c.ticker || '|' || c.trade_date::text
                                || '|' || $5::text)
               ) AS rn
        FROM c
        JOIN trade_paths tp USING (ticker, trade_date, entry_anchor)
        JOIN want w ON w.trade_date = c.trade_date
        WHERE c.entry_anchor = $1{strike_pred}
    )
    SELECT e.ticker, e.trade_date, e.exit_bar, e.exit_return, e.exit_rule,
           e.entry_price, (e.trade_date < $2::date) AS is_train
    FROM elig e
    JOIN want w ON w.trade_date = e.trade_date
    WHERE e.rn <= w.k
    ORDER BY e.trade_date, e.ticker
    """


@router.post("/zone")
async def zone(req: ZoneReq = Body(...), pool=Depends(get_oi_pool)):
    """Per-trade payloads for the selected cells, under the run's exit policy.

    Separate from /run on purpose. These series only mean anything once a
    zone is chosen, and computing them for all 400 cells on every parameter
    tweak is exactly the cost that would make iteration unusable.

    The payload deliberately matches the contracts the Recall charts already
    consume, so Factor Trades renders through the SAME functions rather than
    a second implementation:
      equity_primary / equity_combined / combined_trades -> _renderSecEquity
      combined_trades                                    -> _renderZoneYearly
      combined_trades + combined_trade_dates + horizon
        + trading_days                                   -> _renderSecActivity
      tickers                                            -> _renderSecBubble
    """
    if not pool:
        return {"error": "OI database not configured"}

    from app.routers.oi_analysis import _get_tt_cutoff, _sec_equity_curve

    async with pool.acquire() as conn:
        rules = await _load_rules(conn)
        if not rules:
            return {"error": "trade_path_rules is empty"}
        by_key = by_key_from_rows(rules)
        meta_by_key = {r["rule_key"]: r for r in rules}

        bin_cols = await _bin_columns(conn)

        try:
            combine_sql, combine_meta = build_combine_sql(
                req.rule_keys, by_key, include_exit_rule=True)
        except CombineError as e:
            return {"error": str(e)}

        cutoff_iso = await _get_tt_cutoff(pool)
        if not cutoff_iso:
            return {"error": "tt_bins has no cutoff_date"}
        try:
            cutoff_d = _date.fromisoformat(cutoff_iso)
        except (TypeError, ValueError):
            return {"error": f"tt_bins cutoff_date is not an ISO date: {cutoff_iso!r}"}

        # $1 anchor, $2 cutoff, then the selection's own placeholders.
        sel = await _resolve_selection(conn, req, bin_cols, 2, cutoff_iso)
        if len(sel) == 1:
            return {"error": sel[0]}
        where_bins, cell_pred, mask_sql, sel_params, prov = sel
        args = [req.entry_anchor, cutoff_d, *sel_params]

        # Same population filter as /run. The placeholder index depends on how
        # many args the cell predicate already consumed, so it is computed
        # rather than hardcoded.
        strike_pred = ""
        if req.max_strike and req.max_strike > 0:
            strike_pred = f" AND tp.entry_price <= ${len(args) + 1}"
            args = args + [float(req.max_strike)]

        # Portfolio mode only. Rides along in the row that already exists, so
        # it adds no rows, no joins and no second query -- see
        # _portfolio_selection for why it is a bitmask rather than one column
        # per signal.
        # A baseline replaces the entries, so only a real portfolio run carries
        # attribution. Resolved once here rather than probed per row.
        emits_mask = bool(mask_sql) and not req.randomize
        mask_sel = f",\n               {mask_sql} AS sig_mask" if emits_mask else ""

        sql = f"""
        WITH c AS (
{combine_sql}
        )
        SELECT c.ticker, c.trade_date, c.exit_bar, c.exit_return, c.exit_rule,
               tp.entry_price,
               (c.trade_date < $2::date) AS is_train{mask_sel}
        FROM c
        JOIN tt_bins bt USING (ticker, trade_date)
        -- entry_price is not exposed by build_combine_sql's outer SELECT, and
        -- the vendored function is not the place to add it. Rejoining
        -- trade_paths on its primary key is cheap and leaves the combine
        -- untouched.
        JOIN trade_paths tp USING (ticker, trade_date, entry_anchor)
        WHERE c.entry_anchor = $1 AND {where_bins} AND {cell_pred}{strike_pred}
        ORDER BY c.trade_date, c.ticker
        """
        baseline = None
        if not req.randomize:
            rows = await conn.fetch(sql, *args)
        else:
            # Step 1: the SHAPE to match -- how many trades the real zone
            # fires on each date. Counted with the identical predicate, so
            # the target is the real zone's own distribution and not a
            # re-derivation of it. Both windows are counted; the active
            # window filter runs downstream on the random rows exactly as it
            # does on real ones, so TRAIN and TEST both stay matched.
            cnt_rows = await conn.fetch(
                _zone_count_sql(combine_sql, where_bins, cell_pred, strike_pred),
                *args)
            if not cnt_rows:
                return {"error": "the selected zone has no trades — nothing to "
                                 "match a random baseline against"}
            want_dates = [r["trade_date"] for r in cnt_rows]
            want_ks    = [int(r["k"]) for r in cnt_rows]
            seed = DEFAULT_BASELINE_SEED if req.seed is None else int(req.seed)

            # Placeholders are fixed at $1..$5 in _random_zone_sql, so the
            # optional strike filter takes $6 -- it cannot reuse the zone
            # query's computed index, which counted a different arg list.
            kind = (req.baseline_kind or "entry").strip()
            if kind not in ("entry", "exit", "entry_exit"):
                return {"error": f"unknown baseline_kind {req.baseline_kind!r}"}

            if kind == "entry":
                r_strike, r_args = "", []
                if req.max_strike and req.max_strike > 0:
                    r_strike = " AND tp.entry_price <= $6"
                    r_args = [float(req.max_strike)]
                rows = await conn.fetch(
                    _random_zone_sql(combine_sql, r_strike),
                    req.entry_anchor, cutoff_d, want_dates, want_ks, str(seed), *r_args)
                hold_note = None
            else:
                # Both random-exit kinds. One CDF, one CASE, built once and
                # used by whichever sampler runs -- two constructions is how
                # "the same holding-period distribution" would stop being
                # true without anything failing.
                avail = _maxdays_rules(rules)
                if not avail:
                    return {"error": "no max_days rules in trade_path_rules — a "
                                     "random-exit baseline has no holding "
                                     "periods to draw from"}
                # Step 2: the holding-period distribution to match, measured
                # off the REAL zone under the REAL policy. Sessions, so it is
                # in the same unit as max_days and as Avg DIT.
                h_rows = await conn.fetch(f"""
                WITH c AS (
{combine_sql}
                )
                SELECT GREATEST(1, CEIL(c.exit_bar / {BARS_PER_SESSION}))::int AS sess,
                       COUNT(*) AS n
                FROM c
                JOIN tt_bins bt USING (ticker, trade_date)
                JOIN trade_paths tp USING (ticker, trade_date, entry_anchor)
                WHERE c.entry_anchor = $1 AND $2::date IS NOT NULL
                      AND {where_bins} AND {cell_pred}{strike_pred}
                GROUP BY 1
                """, *args)
                sess_counts = {int(r["sess"]): int(r["n"]) for r in h_rows}
                ns, los, his = _hold_cdf(sess_counts, [n for n, _ in avail])
                if not ns:
                    return {"error": "could not build a holding-period "
                                     "distribution for the selected zone"}
                col_by_n = dict(avail)
                # CASE arms come from the catalog's own column names. Only the
                # periods the CDF can actually draw get an arm, so a missing
                # arm is impossible by construction rather than by review.
                ret_case = ("CASE d.n_days\n"
                            + "\n".join(f"               WHEN {n} THEN t2.{col_by_n[n]}"
                                        for n in ns)
                            + "\n           END")
                if kind == "entry_exit":
                    r_strike, r_args = "", []
                    if req.max_strike and req.max_strike > 0:
                        r_strike = " AND tp.entry_price <= $9"
                        r_args = [float(req.max_strike)]
                    rows = await conn.fetch(
                        _random_exit_zone_sql(combine_sql, ret_case, r_strike),
                        req.entry_anchor, cutoff_d, want_dates, want_ks, str(seed),
                        ns, los, his, *r_args)
                else:
                    # exit-only: the REAL entries, so it reuses the zone
                    # query's own predicates and arg list, and appends its
                    # four new parameters after them rather than renumbering
                    # the shared fragments.
                    base_n = len(args)
                    rows = await conn.fetch(
                        _random_exit_only_sql(
                            combine_sql, where_bins, cell_pred, strike_pred,
                            ret_case, base_n + 1, base_n + 2, base_n + 3, base_n + 4),
                        *args, str(seed), ns, los, his)
                # A resolved path has every max_days column populated, so a
                # NULL return here means that assumption is wrong. Reported
                # rather than silently averaged as zero, which would drag the
                # baseline toward flat and look like a real finding.
                nulls = sum(1 for r in rows if r["exit_return"] is None)
                if nulls:
                    return {"error": f"random-exit baseline: {nulls} of "
                                     f"{len(rows)} sampled trades have no "
                                     f"return at their drawn holding period"}
                hold_note = {
                    "periods": ns,
                    "weights": [round(h - l, 6) for l, h in zip(los, his)],
                    "avg_sessions": round(
                        sum(n * (h - l) for n, l, h in zip(ns, los, his)), 2),
                }

            # The count is a guarantee, not a hope (see _random_zone_sql), so
            # a mismatch is a bug in that reasoning and is reported as one
            # rather than being smoothed over into a quietly smaller baseline.
            got = len(rows)
            wanted = sum(want_ks)
            baseline = {
                "kind": kind, "seed": seed, "dates": len(want_dates),
                "requested": wanted, "delivered": got,
                # None for an entry-only baseline; the drawn holding-period
                # distribution for a random-exit one.
                "hold": hold_note,
            }
            if got != wanted:
                short = {}
                by_d: dict = {}
                for r in rows:
                    by_d[r["trade_date"]] = by_d.get(r["trade_date"], 0) + 1
                for d_, k_ in zip(want_dates, want_ks):
                    if by_d.get(d_, 0) < k_:
                        short[d_.isoformat()] = [by_d.get(d_, 0), k_]
                return {"error": f"random baseline could not be matched: got "
                                 f"{got} of {wanted} trades. The eligible "
                                 f"universe was smaller than the zone on "
                                 f"{len(short)} date(s).",
                        "baseline_shortfall": dict(list(short.items())[:20])}

        td_rows = await conn.fetch(
            "SELECT DISTINCT trade_date FROM tt_bins ORDER BY trade_date")
        trading_days = [r["trade_date"].isoformat() for r in td_rows]

    # ── Fold into the Recall chart contracts ──────────────────────────────
    trades, dates, reasons = [], [], {}
    hold_by_rule: dict = {}
    series_trades: list = []
    from collections import Counter as _Ctr
    zacc = {"train": ([], [], _Ctr(), []), "test": ([], [], _Ctr(), [])}
    by_ticker: dict[str, list] = {}
    tot = {"train": [0, 0.0, 0.0], "test": [0, 0.0, 0.0]}
    active = "train" if (req.window or "train") == "train" else "test"
    for r in rows:
        win = "train" if r["is_train"] else "test"
        # Trades outside the active window are not returned at all. That is
        # what makes the time-series panes stop at the cutoff in train mode
        # rather than drawing test data the user has chosen not to look at.
        # TRAIN never draws test data at all; TEST draws the full history, so
        # only the train-mode skip happens before _rec is built. The
        # single-window filter is applied AFTER series_trades has taken its
        # copy -- doing it here is what made TEST equity start at the cutoff.
        if active == "train" and win != "train":
            continue
        ret = float(r["exit_return"] or 0.0)
        hold = float(r["exit_bar"] or 0.0)
        d = r["trade_date"].isoformat()
        t = tot[win]
        t[0] += 1; t[1] += ret; t[2] += hold
        # Exit reasons and the ticker breakdown are TEST-window surfaces:
        # they answer "what happened out of sample", so counting train trades
        # in them would dilute exactly the thing being judged.
        if win == active:
            reasons[r["exit_rule"]] = reasons.get(r["exit_rule"], 0) + 1
            hold_by_rule.setdefault(r["exit_rule"], []).append(hold)
        if win == active:
            za = zacc[win]
            za[0].append(ret); za[1].append(hold)
            za[2][r["ticker"]] += 1; za[3].append(r["trade_date"])
        # trade_paths.entry_price is stored AS-TRADED (the store is
        # adjusted=false), so it is already the price a fill would have
        # happened at. That makes it exactly what the sizing path wants:
        # _computeDollarSeries divides capital by spot_entry_raw and only
        # falls back to the adjusted spot_entry when raw is missing.
        #
        # spot_entry (the split-adjusted basis) is deliberately NOT emitted.
        # Deriving it would mean re-applying split factors here, and
        # build_trade_paths already owns that via make_split_factors with
        # inclusive=False -- a second derivation is how the two drift apart.
        # Nothing on this page needs the adjusted basis: returns come from
        # exit_return, and sizing and the max-strike filter both want the
        # as-traded price.
        _px = r["entry_price"]
        _rec = {
            "ticker": r["ticker"], "trade_date": d,
            "ret": ret, "exit_bar": hold, "exit_rule": r["exit_rule"],
            "window": win,
            "entry_price":    float(_px) if _px is not None else None,
            "spot_entry_raw": float(_px) if _px is not None else None,
        }
        # Portfolio mode only, and deliberately absent on a BASELINE run: those
        # rows are randomly drawn tickers, so no signal claimed them and a mask
        # of 0 would read as "all signals declined" rather than "not asked".
        if emits_mask:
            _rec["sig_mask"] = int(r["sig_mask"] or 0)
        series_trades.append(_rec)
        if win != active:
            continue          # everything below is single-window only
        trades.append(_rec)
        if win == active:
            dates.append(d)
        if win == active:
            by_ticker.setdefault(r["ticker"], []).append(ret)

    eq = _sec_equity_curve(trades, "ret")

    total_pnl = sum(sum(v) for v in by_ticker.values())
    tickers_out = []
    for tkr, rets in by_ticker.items():
        n = len(rets)
        s = sum(rets)
        tickers_out.append({
            "ticker": tkr, "n": n,
            "avg_ret":  round(s / n, 6) if n else 0.0,
            "win_rate": round(sum(1 for x in rets if x > 0) / n, 4) if n else 0.0,
            "contrib_pct": round(s / total_pnl * 100, 2) if total_pnl else 0.0,
        })
    tickers_out.sort(key=lambda x: -x["n"])

    def _stats(win: str) -> dict:
        w = zacc[win]
        span = ((max(w[3]) - min(w[3])).days if len(w[3]) > 1 else 1)
        return _window_stats(w[0], w[1], w[2], span)

    tot_active = tot[active][0] or 1
    breakdown = []
    for rk, n in sorted(reasons.items(), key=lambda kv: -kv[1]):
        m = meta_by_key.get(rk, {})
        is_backstop = (rk == HORIZON_RULE_KEY and combine_meta["horizon_auto_added"])
        _hb = hold_by_rule.get(rk) or []
        breakdown.append({
            # Sessions, same conversion as Avg DIT, so "a stop fires at 1.2
            # sessions and the backstop at 20" reads directly.
            "avg_hold": (sum(_hb) / len(_hb) / BARS_PER_SESSION) if _hb else 0.0,
            "rule_key": rk, "family": m.get("family", rk), "side": m.get("side", ""),
            "label": ("backstop — no selected rule fired" if is_backstop
                      else f"{m.get('family', rk)} {_rule_label(m.get('family',''), m.get('params'))}"),
            "is_backstop": is_backstop, "n": n, "frac": n / tot_active,
        })

    return {
        "cells": req.cells,
        "window": active,
        # Present only on a baseline run, so the client can label the card
        # with the draw it came from and refetch the SAME draw on lock.
        "randomize": req.randomize,
        "baseline": baseline,
        "entry_anchor": req.entry_anchor,
        "cutoff_date": cutoff_iso,
        # Provenance echoed so PORTFOLIO MODE can drive the run card straight
        # off this payload. There is no /run in portfolio mode -- /run exists
        # to build the 20x20 grid, and a portfolio has no single grid to build
        # -- so these are the fields the card would otherwise have taken from
        # a run that never happens. Harmless in signal mode, where they simply
        # echo the request.
        "mode":             prov["mode"],
        "signals":          prov["signals"],
        "primary_metric":   req.primary_metric,
        "secondary_metric": req.secondary_metric,
        "max_strike":       req.max_strike,
        "n_bins":           max(2, min(20, int(req.n_bins))),
        "baseline_kind":    req.baseline_kind if req.randomize else None,
        "seed":             (DEFAULT_BASELINE_SEED if req.seed is None else int(req.seed)) if req.randomize else None,
        # True when every row carries sig_mask. The client must not infer
        # attribution from mode alone: a portfolio BASELINE is mode=portfolio
        # with no mask.
        "has_sig_mask":     emits_mask,
        # Recall chart contracts — same field names, same shapes.
        # window_trades: the single-window population. series_trades: what
        # the three time-series panes draw. combined_trades is kept as an
        # alias of window_trades so nothing that still reads it silently gets
        # the wider set.
        "window_trades":  trades,
        "series_trades":  series_trades,
        "combined_trades": trades,
        "price_bins":     _price_bins(trades),
        "pnl_dist":       _pnl_dist(trades),
        "combined_trade_dates": dates,
        "equity_primary": eq,
        "equity_combined": eq,
        "tickers": tickers_out,
        "trading_days": trading_days,
        "horizon": 1,
        # Factor Trades additions.
        "train": _stats("train"),
        "test": _stats("test"),
        "exit_reasons": breakdown,
        "rules": combine_meta["rules"],
        "horizon_rule": combine_meta["horizon_rule"],
        "horizon_auto_added": combine_meta["horizon_auto_added"],
        "excludes_unresolved": combine_meta["excludes_unresolved"],
        "label": req.label,
    }


# ── Shared batch runner ───────────────────────────────────────────────────
#
# ONE batch path, not two. The baseline suite and the parameter grid are the
# same tool: run the same query N times varying ONE input, aggregate, and
# report a matrix instead of a point. The suite varies the SEED and holds the
# exit parameters fixed; the grid varies the EXIT PARAMETERS and holds the
# seed fixed. Everything else -- the zone, the anchor, the max strike, the
# windows, the concurrency, the fold into stats -- is identical, so it lives
# here and both features are thin layers on top.
#
# A variant is a dict:
#   key            identity in the result matrix
#   group          which row of the matrix it aggregates into
#   rule_keys      exit policy            (the GRID varies this)
#   baseline_kind  None = the policy itself (real entries, real exits),
#                  else "entry" | "exit" | "entry_exit"
#   seed           draw identity          (the SUITE varies this)

# The OI pool is min_size=2, max_size=10. Five leaves headroom, so a batch
# cannot starve the page's own requests -- a suite that makes the UI it is
# rendering into unresponsive is not an improvement on doing it by hand.
BATCH_MAX_CONCURRENCY = 5


def _row_seq_hash(rows) -> str:
    """Digest of a result's (ticker, trade_date) sequence.

    Lets the row-order invariant be checked without keeping every
    combination's rows resident. A digest mismatch says the same thing a
    full comparison would -- this combination did not see the same trades --
    it just cannot name the first divergent row, which is why the skeleton
    combination is compared in full and the rest by digest.
    """
    h = hashlib.blake2b(digest_size=16)
    for r in rows:
        h.update(r["ticker"].encode("utf-8"))
        h.update(b"|")
        h.update(str(r["trade_date"].toordinal()).encode("ascii"))
        h.update(b";")
    return h.hexdigest()


def _stats_by_window(rows) -> dict:
    """Train AND test stats from one variant's rows.

    Deliberately not /zone's fold. That one is window-scoped by design: it
    accumulates only the active window because every surface it feeds is
    single-window. A batch reports both windows for every variant in one
    pass, so this is a different aggregation rather than a second copy of
    the same one.
    """
    from collections import Counter as _C
    acc = {"train": ([], [], _C(), []), "test": ([], [], _C(), [])}
    nulls = {"train": 0, "test": 0}
    for r in rows:
        w = "train" if r["is_train"] else "test"
        rets, holds, tks, dates = acc[w]
        ret = r["exit_return"]
        if ret is None:
            # Skipped, not zero-filled: a missing return is not a flat
            # trade, and averaging it in as one drags the row toward zero.
            # Counted and surfaced, because on the policy path the combine
            # should always produce a value -- if these appear, that is news
            # rather than housekeeping.
            nulls[w] += 1
            continue
        rets.append(float(ret))
        holds.append(float(r["exit_bar"] or 0.0))
        tks[r["ticker"]] += 1
        dates.append(r["trade_date"])
    out = {}
    for w, (rets, holds, tks, dates) in acc.items():
        span = ((max(dates) - min(dates)).days if len(dates) > 1 else 1)
        out[w] = _window_stats(rets, holds, tks, span)
        out[w]["null_returns"] = nulls[w]
    return out


class BatchCtx:
    """Everything a batch shares across its variants, resolved once.

    Built from the base request, so a variant supplies only what it varies.
    The per-date counts and the holding-period CDF resolve lazily: the grid
    never randomises, and making it pay for two queries it will not read
    would be a real cost at grid sizes.
    """

    def __init__(self, req, rules, by_key, cutoff_d, cutoff_iso, args,
                 where_bins, cell_pred, strike_pred):
        self.req = req
        self.rules = rules
        self.by_key = by_key
        self.cutoff_d = cutoff_d
        self.cutoff_iso = cutoff_iso
        self.args = args
        self.where_bins = where_bins
        self.cell_pred = cell_pred
        self.strike_pred = strike_pred
        self.want_dates = None
        self.want_ks = None
        self.cdf = None          # (ns, los, his)
        self.ret_case = None
        self.hold = None
        # Resolved ONCE, under a lock. Without it the guard in ensure_random
        # is a race with teeth: want_dates is assigned before cdf, so a
        # second variant could see the shape already resolved, return
        # immediately, and unpack cdf while it was still None. It fires only
        # under concurrency -- which is the only way this is ever called --
        # and would read as a data problem rather than a lock problem.
        self._lock = asyncio.Lock()
        self._resolved = False
        self._resolve_err = None

    def combine(self, rule_keys):
        return build_combine_sql(rule_keys, self.by_key, include_exit_rule=True)

    async def ensure_random(self, conn):
        """Resolve the entry shape and the holding-period distribution.

        Both are measured off the BASE policy under the BASE rules, once.
        Every baseline in a suite matches the same real zone, so deriving
        them per variant would be the same two queries thirty times -- and
        worse, would let two variants disagree about what they are matching.

        Returns an error string or None. Not raising, because a batch
        reports per-variant failures rather than dying.
        """
        async with self._lock:
            if self._resolved:
                return self._resolve_err
            self._resolve_err = await self._resolve(conn)
            self._resolved = True
            return self._resolve_err

    async def _resolve(self, conn):
        combine_sql, _ = self.combine(self.req.rule_keys)
        cnt = await conn.fetch(
            _zone_count_sql(combine_sql, self.where_bins,
                            self.cell_pred, self.strike_pred), *self.args)
        if not cnt:
            return "the selected zone has no trades — nothing to match against"
        self.want_dates = [r["trade_date"] for r in cnt]
        self.want_ks = [int(r["k"]) for r in cnt]

        avail = _maxdays_rules(self.rules)
        if not avail:
            return "no max_days rules in trade_path_rules"
        h = await conn.fetch(f"""
        WITH c AS (
{combine_sql}
        )
        SELECT GREATEST(1, CEIL(c.exit_bar / {BARS_PER_SESSION}))::int AS sess,
               COUNT(*) AS n
        FROM c
        JOIN tt_bins bt USING (ticker, trade_date)
        JOIN trade_paths tp USING (ticker, trade_date, entry_anchor)
        WHERE c.entry_anchor = $1 AND $2::date IS NOT NULL
              AND {self.where_bins} AND {self.cell_pred}{self.strike_pred}
        GROUP BY 1
        """, *self.args)
        ns, los, his = _hold_cdf({int(r["sess"]): int(r["n"]) for r in h},
                                 [n for n, _ in avail])
        if not ns:
            return "could not build a holding-period distribution"
        col_by_n = dict(avail)
        self.cdf = (ns, los, his)
        self.ret_case = ("CASE d.n_days\n"
                         + "\n".join(f"               WHEN {n} THEN t2.{col_by_n[n]}"
                                     for n in ns)
                         + "\n           END")
        self.hold = {"periods": ns,
                     "avg_sessions": round(
                         sum(n * (b - a) for n, a, b in zip(ns, los, his)), 2)}
        return None


def _policy_sql(combine_sql: str, where_bins: str,
                cell_pred: str, strike_pred: str) -> str:
    """The zone's own trades under one exit policy -- no randomisation.

    $2 is cast for the same reason _zone_count_sql casts it: it is the only
    typed context the parameter gets in this statement.
    """
    return f"""
    WITH c AS (
{combine_sql}
    )
    SELECT c.ticker, c.trade_date, c.exit_bar, c.exit_return, c.exit_rule,
           tp.entry_price,
           (c.trade_date < $2::date) AS is_train
    FROM c
    JOIN tt_bins bt USING (ticker, trade_date)
    JOIN trade_paths tp USING (ticker, trade_date, entry_anchor)
    WHERE c.entry_anchor = $1 AND {where_bins} AND {cell_pred}{strike_pred}
    -- NOT optional. max_dd is a running peak-to-trough over np.cumsum, so
    -- it depends on the order rows arrive in; without this the policy row's
    -- Max DD and Calmar would be computed over whatever order the planner
    -- happened to emit, and would not match the stat bar. The three random
    -- samplers already order by the same key.
    ORDER BY c.trade_date, c.ticker
    """


async def _run_variant(pool, ctx, v: dict) -> dict:
    """One variant, on its own pooled connection."""
    rule_keys = v.get("rule_keys") or ctx.req.rule_keys
    kind = v.get("baseline_kind")
    seed = v.get("seed")
    out = {"key": v["key"], "group": v.get("group"), "seed": seed, "kind": kind}
    async with pool.acquire() as conn:
        try:
            combine_sql, meta = ctx.combine(rule_keys)
        except CombineError as e:
            return {**out, "error": str(e)}

        ns = los = his = None
        if kind:
            err = await ctx.ensure_random(conn)
            if err:
                return {**out, "error": err}
            ns, los, his = ctx.cdf

        if not kind:
            rows = await conn.fetch(
                _policy_sql(combine_sql, ctx.where_bins, ctx.cell_pred,
                            ctx.strike_pred), *ctx.args)
        elif kind == "entry":
            r_strike, r_args = "", []
            if ctx.req.max_strike and ctx.req.max_strike > 0:
                r_strike = " AND tp.entry_price <= $6"
                r_args = [float(ctx.req.max_strike)]
            rows = await conn.fetch(
                _random_zone_sql(combine_sql, r_strike),
                ctx.req.entry_anchor, ctx.cutoff_d, ctx.want_dates, ctx.want_ks,
                str(seed), *r_args)
        elif kind == "entry_exit":
            r_strike, r_args = "", []
            if ctx.req.max_strike and ctx.req.max_strike > 0:
                r_strike = " AND tp.entry_price <= $9"
                r_args = [float(ctx.req.max_strike)]
            rows = await conn.fetch(
                _random_exit_zone_sql(combine_sql, ctx.ret_case, r_strike),
                ctx.req.entry_anchor, ctx.cutoff_d, ctx.want_dates, ctx.want_ks,
                str(seed), ns, los, his, *r_args)
        elif kind == "exit":
            base_n = len(ctx.args)
            rows = await conn.fetch(
                _random_exit_only_sql(combine_sql, ctx.where_bins, ctx.cell_pred,
                                      ctx.strike_pred, ctx.ret_case,
                                      base_n + 1, base_n + 2, base_n + 3, base_n + 4),
                *ctx.args, str(seed), ns, los, his)
        else:
            return {**out, "error": f"unknown baseline_kind {kind!r}"}

    st = _stats_by_window(rows)
    res = {**out, "rule_keys": list(rule_keys),
           "train": st["train"], "test": st["test"],
           "horizon_auto_added": meta["horizon_auto_added"]}
    if v.get("compact"):
        # THE MEMORY FIX. Holding asyncpg Records for every variant meant
        # the server carried combinations x trades of them at once --
        # 183 x 5,409 is ~990k Record objects, which is what actually died
        # on a larger sweep. A compact variant keeps only the returns it
        # contributes plus a digest of its (ticker, date) sequence, so the
        # row-order invariant is still checked, in O(1) per combination
        # instead of O(n), against a skeleton built once.
        res["r"] = [None if r["exit_return"] is None
                    else round(float(r["exit_return"]), 8) for r in rows]
        res["seq_hash"] = _row_seq_hash(rows)
        res["n_rows"] = len(rows)
        res["reasons"] = _grid_reasons(rows)
    else:
        res["rows"] = rows
    return res


async def run_batch(pool, ctx, variants: list,
                    concurrency: int = BATCH_MAX_CONCURRENCY) -> list:
    """Run every variant concurrently against the same zone.

    A semaphore rather than an unbounded gather: sixty simultaneous acquires
    on a ten-connection pool would queue anyway, but they would ALSO hold
    every other request on the page behind them.

    A variant that raises returns an error entry instead of taking the batch
    down. One bad parameter set in a grid should cost that cell, not the
    whole surface -- and at 60+ variants, losing the batch to the last one
    is losing 90 seconds.
    """
    sem = asyncio.Semaphore(max(1, min(int(concurrency), BATCH_MAX_CONCURRENCY)))

    async def one(v):
        async with sem:
            try:
                return await _run_variant(pool, ctx, v)
            except Exception as e:
                logging.exception("factor-trades batch variant failed: %s", v.get("key"))
                return {"key": v["key"], "group": v.get("group"),
                        "error": f"{type(e).__name__}: {e}"}

    return await asyncio.gather(*[one(v) for v in variants])


# ── Baseline suite ────────────────────────────────────────────────────────

# Metrics reported per run type. All are RETURN-unit (additive), which is
# what _window_stats computes -- NOT the dollar-sized figures the stat bar
# shows for Total / Avg Annual / Max DD / Calmar.
#
# That difference is deliberate and has to be labelled in the UI, because
# two definitions of "Calmar" on one page is exactly the confusion that was
# fixed once already. The dollar figures are derived client-side by
# _computeDollarSeries from per-trade rows; reproducing that server-side
# would be a second implementation of sizing, and shipping ~100k trades to
# the browser so the existing one could be reused would cost more than the
# queries do. Return units are also the right unit for THIS table: they are
# sizing-independent, so a seed comparison is not partly a comparison of
# how capital happened to be deployed.
#
# The labels carry the unit. "Calmar" alone on a page whose stat bar also
# says "Calmar" is two definitions of one word to be reconciled by hand
# across sixty numbers -- the same failure as a percent Calmar sitting
# beside a blank dollar Max DD, at forty times the scale.
SUITE_METRICS = [
    ("avg_ret",    "Avg Ret",        "pct"),
    ("total_ret",  "Total Ret",      "usd"),
    ("avg_annual", "Avg Annual Ret", "usd"),
    ("max_dd",     "Max DD",         "usd"),
    ("calmar",     "Calmar",         "ratio"),
]

# Stated on the panel and repeated in the CSV, not left to the column
# labels alone.
SUITE_UNITS_NOTE = (
    "Total Ret, Avg Annual, Max DD and Calmar are DOLLARS, sized by the "
    "rail's $/trade and daily cap and derived in the browser by the same "
    "function that fills the stat bar — so the policy row equals the stat "
    "bar exactly. Avg Ret stays a per-trade percentage; the daily cap "
    "cannot affect a per-trade mean."
)

SUITE_GROUPS = [
    ("policy",     "Policy (signal entries + rule exits)", None),
    ("entry",      "Random entries + rule exits",          "entry"),
    ("exit",       "Signal entries + random exits",        "exit"),
    ("entry_exit", "Random entries + random exits",        "entry_exit"),
]


def _split_pack(rows, pack):
    """Pack one variant's rows per window.

    Windows are packed SEPARATELY because they are sized separately: the
    stat bar's dollar figures come from the active window's trades alone,
    with equity starting at zero in that window. Packing them together and
    splitting client-side would invite a caller to size the pair as one
    continuous account, which is not what any box on this page shows.
    """
    tr = [r for r in rows if r["is_train"]]
    te = [r for r in rows if not r["is_train"]]
    return {"train": pack(tr), "test": pack(te)}


def _suite_seeds(n: int, base: int) -> list[int]:
    """Deterministic seed list, so a suite is reproducible.

    Derived from the base seed rather than drawn randomly: re-running a
    suite on the same zone must give the same table, or "beats 0 of 10"
    cannot be checked. Re-sampling the suite means passing a different base.
    """
    return [base + 1000003 * i for i in range(max(1, n))]


class SuiteReq(ZoneReq):
    n_draws: int = 10
    concurrency: int = BATCH_MAX_CONCURRENCY


@router.post("/suite")
async def suite(req: SuiteReq = Body(...), pool=Depends(get_oi_pool)):
    """Policy + three baseline types x N seeds, train and test, in one batch.

    A thin layer on run_batch. The parameter grid will be another one: it
    varies rule_keys where this varies seed, and reads the same matrix.

    The headline number is BEATS: how many of the N draws the policy beat on
    each metric. "Signal beats 0 of 10" answers the question directly, where
    two means require the reader to hold a distribution in their head.
    """
    if not pool:
        return {"error": "OI database not configured"}

    from app.routers.oi_analysis import _get_tt_cutoff

    n = max(1, min(50, int(req.n_draws)))

    async with pool.acquire() as conn:
        rules = await _load_rules(conn)
        if not rules:
            return {"error": "trade_path_rules is empty"}
        by_key = by_key_from_rows(rules)

        bin_cols = await _bin_columns(conn)

        cutoff_iso = await _get_tt_cutoff(pool)
        if not cutoff_iso:
            return {"error": "tt_bins has no cutoff_date"}
        try:
            cutoff_d = _date.fromisoformat(cutoff_iso)
        except (TypeError, ValueError):
            return {"error": f"tt_bins cutoff_date is not an ISO date: {cutoff_iso!r}"}

        # Same resolver /zone uses, so a suite or a grid cannot come to
        # disagree with the zone it is sweeping around. The mask is discarded:
        # these paths aggregate to stats and never emit a trade row, so there
        # is nothing to attribute.
        sel = await _resolve_selection(conn, req, bin_cols, 2, cutoff_iso)
        if len(sel) == 1:
            return {"error": sel[0]}
        where_bins, cell_pred, _mask_sql, sel_params, _prov = sel
        args = [req.entry_anchor, cutoff_d, *sel_params]

        strike_pred = ""
        if req.max_strike and req.max_strike > 0:
            strike_pred = f" AND tp.entry_price <= ${len(args) + 1}"
            args = args + [float(req.max_strike)]

    ctx = BatchCtx(req, rules, by_key, cutoff_d, cutoff_iso, args,
                   where_bins, cell_pred, strike_pred)

    base_seed = DEFAULT_BASELINE_SEED if req.seed is None else int(req.seed)
    seeds = _suite_seeds(n, base_seed)

    # The policy row is generated by the SAME code path as the baselines --
    # one variant among the rest, folded by the same _stats_by_window. A row
    # transcribed from a previous run is a row that can silently be from a
    # different population.
    variants = [{"key": "policy", "group": "policy", "baseline_kind": None}]
    for gkey, _label, kind in SUITE_GROUPS:
        if kind is None:
            continue
        for s in seeds:
            variants.append({"key": f"{gkey}:{s}", "group": gkey,
                             "baseline_kind": kind, "seed": s})

    results = await run_batch(pool, ctx, variants, req.concurrency)

    # ── Columnar packing ─────────────────────────────────────────────────
    #
    # The four portfolio metrics -- Total Ret, Avg Annual, Max DD, Calmar --
    # are NOT computed here. They depend on $/trade sizing and the daily
    # cap, and a return-unit version of them describes a system with
    # unlimited capital that nobody trades: with a $10k daily cap, a day
    # firing 40 names cannot take them all, and drawdown accrues against
    # deployed capital rather than a sum of per-trade returns. The earlier
    # claim that sizing is "constant across draws" was wrong -- the cap
    # interacts with the trade distribution, so a draw holding different
    # durations or different prices meets a different cap.
    #
    # So this ships the per-trade rows and the CLIENT derives dollars
    # through the SAME _computeDollarSeries that produces the stat bar.
    # That is not merely one implementation instead of two: it makes
    # agreement definitional rather than something a gate has to keep
    # proving on whichever zones it happens to be run against.
    #
    # Columnar with shared ticker and date tables, because the naive
    # array-of-objects form is ~3x the bytes for the same information and
    # every baseline shares the policy's dates by construction.
    tick_ix: dict = {}
    date_ix: dict = {}
    tickers: list = []
    dates: list = []

    def _ix(v, table, out):
        i = table.get(v)
        if i is None:
            i = len(out)
            table[v] = i
            out.append(v)
        return i

    def _pack(rows):
        t, d, r, px = [], [], [], []
        for row in rows:
            ret = row["exit_return"]
            if ret is None:
                continue
            t.append(_ix(row["ticker"], tick_ix, tickers))
            d.append(_ix(row["trade_date"].isoformat(), date_ix, dates))
            r.append(round(float(ret), 8))
            # trade_paths.entry_price is stored AS-TRADED, which is exactly
            # what sizing wants: _computeDollarSeries divides the allocation
            # by spot_entry_raw. Shipped under that name so the client can
            # hand these rows to the existing function untouched.
            _p = row["entry_price"]
            px.append(round(float(_p), 4) if _p is not None else None)
        return {"t": t, "d": d, "r": r, "p": px}

    by_group: dict = {}
    errors = []
    policy = None
    total_rows = 0
    for r in results:
        if r.get("error"):
            errors.append({"key": r["key"], "error": r["error"]})
            continue
        total_rows += len(r.get("rows") or [])
        if r["group"] == "policy":
            policy = r
        else:
            by_group.setdefault(r["group"], []).append(r)

    if policy is None:
        return {"error": "the policy run failed, so nothing can be compared "
                         "against it", "errors": errors}

    # An honest ceiling rather than a browser that dies silently. ~400k
    # trades is ~10MB packed; past that the answer is a smaller N, and
    # saying so beats shipping it and hoping.
    if total_rows > 400_000:
        return {"error": f"this suite would ship {total_rows:,} trades "
                         f"({len(variants)} runs). Lower the draw count — "
                         f"dollar figures are derived per-trade in the "
                         f"browser, so the payload scales with N."}

    out_rows = []
    for gkey, label, kind in SUITE_GROUPS:
        src = [policy] if kind is None else by_group.get(gkey, [])
        out_rows.append({
            "key": gkey, "label": label, "kind": kind, "draws": len(src),
            "runs": [{
                "seed": g.get("seed"),
                # Sizing-independent, so it stays server-side: a per-trade
                # mean is not a portfolio quantity and the daily cap cannot
                # touch it.
                "avg_ret": {w: g[w].get("avg_ret") for w in ("train", "test")},
                "n":       {w: g[w].get("n") for w in ("train", "test")},
                "null_returns": {w: g[w].get("null_returns", 0)
                                 for w in ("train", "test")},
                "trades":  _split_pack(g.get("rows") or [], _pack),
            } for g in src],
        })

    return {
        "n_draws": n, "base_seed": base_seed, "seeds": seeds,
        "cutoff_date": cutoff_iso,
        "cells": req.cells,
        "rule_keys": list(req.rule_keys),
        "entry_anchor": req.entry_anchor,
        "max_strike": req.max_strike,
        "metrics": [{"key": k, "label": l, "unit": u} for k, l, u in SUITE_METRICS],
        "rows": out_rows,
        "tickers": tickers,
        "dates": dates,
        "hold": ctx.hold,
        "n_variants": len(variants),
        "n_trades": total_rows,
        "concurrency": max(1, min(int(req.concurrency), BATCH_MAX_CONCURRENCY)),
        # Surfaced, never swallowed: a suite missing draws is a different
        # table from one where every draw ran.
        "errors": errors,
        "units_note": SUITE_UNITS_NOTE,
    }


# ── Parameter response grid ───────────────────────────────────────────────
#
# The second layer on run_batch. The suite varies the SEED and holds
# parameters fixed; this varies the PARAMETERS and holds the seed fixed.
# Same fan-out, same folding, same concurrency cap — only the variant list
# differs.
#
# The question is whether a parameter's effect is smooth and consistent, NOT
# which combination scores best. That is why the primary view is marginal
# (each parameter averaged across every setting of the others) with a spread
# band: a one-at-a-time sweep only describes the slice it was run on, and if
# the stop's effect depends on the target -- it does -- that slice does not
# generalise.

# 600 combinations is ~2.5 minutes of queries and ~12MB of returns. Past
# that the honest answer is fewer families, not a batch that will not finish
# well. Four families at seven values each is 2,401.
GRID_MAX_COMBOS = 600

# The combination cap alone was the WRONG GUARD. It says nothing about how
# many trades each combination carries, so a 5,409-trade zone sailed through
# a 183-combination sweep at ~990k returns and took the server down instead
# of answering. What scales is combinations x trades, so that is what is
# capped. 1.2M returns is ~11MB of JSON and ~40MB resident.
GRID_MAX_RETURNS = 1_200_000

GRID_METRICS = [
    ("calmar",     "Calmar",     "ratio"),
    ("avg_ret",    "Avg Ret",    "pct"),
    ("total_ret",  "Total Ret",  "usd"),
    ("max_dd",     "Max DD",     "usd"),
    ("avg_hold",   "Avg DIT",    "sess"),
    ("exit_share", "Exit share", "pct"),
]


class GridReq(ZoneReq):
    # Families to sweep. Every catalog value of each is used, so the caller
    # picks families rather than values -- the point is the whole response
    # curve, and a hand-picked subset of it is the thing this replaces.
    sweep_families: list[str] = []
    # {family: [rule_key, ...]} restricting which catalog values are swept.
    # Absent family = all of its values. Filtering is what makes a fourth
    # family affordable, since the payload cap binds on combinations x trades.
    sweep_values: dict[str, list[str]] = {}
    concurrency: int = BATCH_MAX_CONCURRENCY
    # Diff N combinations against build_combine_sql before returning. 0 = off.
    # On demand rather than always, because verifying costs exactly the
    # per-combination queries this change removes.
    verify: int = 0


@router.post("/grid")
async def grid(req: GridReq = Body(...), pool=Depends(get_oi_pool)):
    """Full cross product over the swept families' catalog values.

    TRAIN and TEST both come back from every combination at no extra cost:
    is_train is a column in the SELECT, not a filter, so the rank scatter is
    free rather than a doubling.

    Returns per-trade RETURNS per combination against ONE shared skeleton.
    This path's row filter is `path_status = 'ok'`, which does not depend on
    which rules are selected, so every combination sees the same trades in the
    same order -- only the exit changes. Shipping the skeleton once instead of
    per combination is a ~30x reduction, and the invariant is asserted per
    combination rather than trusted.

    KNOWN DIVERGENCE FROM THE ORACLE, not yet resolved. Since source 48aeb5b
    build_combine_sql filters on the SHORTEST SELECTED max_days rather than on
    path_status, so its population IS now selection-dependent and this scan's
    is not. Consequences, both in the direction of this path being too tight:

      - today, a swept or held max_days shorter than 20 resolves more rows on
        the zone/stat-bar path than the grid will show for the same rules;
      - after a rebuild at MAX_HORIZON_SESSIONS = 40, `path_status = 'ok'`
        means 40 resolvable sessions while the oracle asks only for the
        selection's own, so the gap widens to nearly every selection.

    The exits themselves still agree -- see _ordered, which tracks the oracle's
    rule set exactly -- so this is a population difference, not a wrong exit.
    Fixing it means making the shared scan the LOOSEST population across the
    sweep and applying each combination's own resolution mask in numpy, where
    every backstop column is already in `bars`. Deliberately not done blind:
    scripts/check_grid_equivalence.py is the gate for this path and it needs a
    database.
    """
    if not pool:
        return {"error": "OI database not configured"}
    if not req.sweep_families:
        return {"error": "pick at least one family to sweep"}

    from app.routers.oi_analysis import _get_tt_cutoff

    async with pool.acquire() as conn:
        rules = await _load_rules(conn)
        if not rules:
            return {"error": "trade_path_rules is empty"}
        by_key = by_key_from_rows(rules)

        bin_cols = await _bin_columns(conn)

        cutoff_iso = await _get_tt_cutoff(pool)
        if not cutoff_iso:
            return {"error": "tt_bins has no cutoff_date"}
        try:
            cutoff_d = _date.fromisoformat(cutoff_iso)
        except (TypeError, ValueError):
            return {"error": f"tt_bins cutoff_date is not an ISO date: {cutoff_iso!r}"}

        # Same resolver /zone uses, so a suite or a grid cannot come to
        # disagree with the zone it is sweeping around. The mask is discarded:
        # these paths aggregate to stats and never emit a trade row, so there
        # is nothing to attribute.
        sel = await _resolve_selection(conn, req, bin_cols, 2, cutoff_iso)
        if len(sel) == 1:
            return {"error": sel[0]}
        where_bins, cell_pred, _mask_sql, sel_params, _prov = sel
        args = [req.entry_anchor, cutoff_d, *sel_params]

        strike_pred = ""
        if req.max_strike and req.max_strike > 0:
            strike_pred = f" AND tp.entry_price <= ${len(args) + 1}"
            args = args + [float(req.max_strike)]

    # ── Cross product ────────────────────────────────────────────────────
    fam_rules: dict[str, list] = {}
    for r in rules:
        fam_rules.setdefault(r["family"], []).append(r)
    for rs in fam_rules.values():
        rs.sort(key=lambda r: _rule_sort_key(r["params"]))

    swept = []
    for fam in req.sweep_families:
        rs = fam_rules.get(fam)
        if not rs:
            return {"error": f"no such family in trade_path_rules: {fam!r}"}
        keep = req.sweep_values.get(fam)
        if keep:
            wanted = set(keep)
            rs = [r for r in rs if r["rule_key"] in wanted]
            if not rs:
                return {"error": f"every value of {fam!r} was filtered out; "
                                 f"a swept family needs at least one value"}
        swept.append({
            "family": fam,
            "values": [{"rule_key": r["rule_key"],
                        "label": _rule_label(fam, r["params"]),
                        "sort": _rule_sort_key(r["params"])[1]} for r in rs],
        })

    total = 1
    for f in swept:
        total *= len(f["values"])
    if total > GRID_MAX_COMBOS:
        dims = " x ".join(f"{len(f['values'])} {f['family']}" for f in swept)
        return {"error": f"that sweep is {dims} = {total:,} combinations, over "
                         f"the {GRID_MAX_COMBOS} limit. At ~1.2s each with "
                         f"{BATCH_MAX_CONCURRENCY}-way concurrency that is "
                         f"~{int(total * 1.2 / BATCH_MAX_CONCURRENCY / 60)} "
                         f"minutes and roughly "
                         f"{int(total * 20 / 1000)}MB of returns. Sweep fewer "
                         f"families."}

    # Families NOT being swept keep whatever the rail has selected, so a grid
    # is a slice through the CURRENT policy rather than through a bare one.
    fam_of = {r["rule_key"]: r["family"] for r in rules}
    swept_names = {f["family"] for f in swept}
    held = [k for k in (req.rule_keys or []) if fam_of.get(k) not in swept_names]

    import itertools
    combos_meta = []
    for combo in itertools.product(*[f["values"] for f in swept]):
        ix = len(combos_meta)
        combos_meta.append({
            "i": ix,
            "params": {f["family"]: c["rule_key"]
                       for f, c in zip(swept, combo)},
            "labels": {f["family"]: c["label"] for f, c in zip(swept, combo)},
            "is_null": False,
            # Carried EXPLICITLY rather than reconstructed downstream as
            # held + params.values(). Reference combinations below are not
            # expressible that way, and a second reconstruction is where the
            # resolve path and the oracle could come to disagree about what
            # a combination even is.
            "rule_keys": held + [c["rule_key"] for c in combo],
        })

    # The null combination is a PERMANENT reference, not something to
    # remember to add: no stop, no target, no held rules -- an empty rule
    # list, which build_combine_sql resolves to the horizon backstop alone.
    # "Do nothing" has to be on every grid because a policy that cannot beat
    # it is not a policy.
    null_ix = len(combos_meta)
    combos_meta.append({"i": null_ix, "params": {}, "labels": {},
                        "is_null": True, "rule_keys": [HORIZON_RULE_KEY]})

    # ── Reference combinations ───────────────────────────────────────────
    #
    # "Do nothing" is not one number when max_days is swept: horizon-only at
    # 5 days and horizon-only at 20 days are different baselines, and the
    # max_days panel needs the CURVE, not a flat line through one of them.
    # The other panels marginalise over the swept horizons, so their
    # reference is the mean of these, computed client-side from the same
    # values rather than from a separate notion of null.
    #
    # These are nearly free under the single-query design: the max_days
    # columns are already in the fetch, so each costs one more argmin.
    ref_ix: dict[str, int] = {}
    hz_fam = next((f for f in swept if f["family"] == "max_days"), None)
    if hz_fam:
        for v in hz_fam["values"]:
            i = len(combos_meta)
            combos_meta.append({
                "i": i, "params": {}, "labels": {"max_days": v["label"]},
                "is_null": False, "is_ref": True,
                "ref_family": "max_days", "ref_value": v["rule_key"],
                # Horizon-only AT this horizon: no stop, no target, nothing
                # held. build_combine_sql resolves a lone max_days rule
                # without auto-appending the backstop, since it IS one.
                "rule_keys": [v["rule_key"]],
            })
            ref_ix[v["rule_key"]] = i
    # NOT appended to `variants`: it runs first and alone, because the
    # skeleton and the reference digest have to exist before any other
    # combination can be checked against them.

    # ── Pre-flight: how big is this actually going to be? ────────────────
    #
    # The combination cap alone was the wrong guard. It counts combinations
    # and says nothing about trades, so a zone with 5,409 trades passed a
    # 183-combination sweep straight through at ~990k returns -- and the
    # server, holding every variant's rows at once, died rather than
    # answering. The guard has to fire BEFORE the fan-out and has to measure
    # the thing that actually scales: combinations x trades.
    #
    # One cheap COUNT buys that. It deliberately does NOT go through
    # build_combine_sql: the population is `path_status = 'ok'` intersected
    # with the zone predicate, and none of that depends on which rules are
    # selected -- the same fact that makes the sweep comparable cell to cell.
    #
    # That last clause stopped being true of the ORACLE at source 48aeb5b,
    # which filters on the shortest selected max_days. This count still
    # measures this path's own population, so it remains the right guard for
    # the fan-out it is protecting; see the divergence note on _grid_scan_sql.
    #
    # Routing it through a combine also made an empty rule list fatal.
    # build_combine_sql rightly refuses `[]` (a combine with no rules has no
    # exit), so ANY caller holding nothing fixed -- which is what a sweep of
    # every family means, and what check_grid_equivalence.py passes -- died
    # in the pre-flight with "no rules selected" before a single combination
    # was built. Counting rows needs no exit at all.
    cnt_sql = f"""
    SELECT COUNT(*) AS k
    FROM tt_bins bt
    JOIN trade_paths tp USING (ticker, trade_date)
    WHERE tp.entry_anchor = $1 AND tp.path_status = 'ok'
          AND $2::date IS NOT NULL
          AND {where_bins} AND {cell_pred}{strike_pred}
    """
    async with pool.acquire() as conn:
        trades_each = int((await conn.fetchrow(cnt_sql, *args))["k"] or 0)
    if not trades_each:
        return {"error": "the selected zone has no trades"}

    # + the null, + one horizon-only reference per swept max_days value.
    # The references ship returns like any other combination, so the cap has
    # to count them or the guard understates the payload it is guarding.
    n_ref = len(next((f["values"] for f in swept if f["family"] == "max_days"), []))
    n_combos = total + 1 + n_ref
    returns = n_combos * trades_each
    if returns > GRID_MAX_RETURNS:
        dims = " x ".join(f"{len(f['values'])} {f['family']}" for f in swept)
        max_combos = max(1, GRID_MAX_RETURNS // trades_each)
        return {"error": f"that sweep is {dims} = {n_combos:,} combinations "
                         f"against {trades_each:,} trades each = "
                         f"{returns:,} returns, over the "
                         f"{GRID_MAX_RETURNS:,} limit (~"
                         f"{int(returns * 9 / 1e6)}MB). This zone supports "
                         f"about {max_combos:,} combinations. Sweep fewer "
                         f"families, or narrow the zone.",
                 "n_combos": n_combos, "trades_each": trades_each,
                 "limit": GRID_MAX_RETURNS}

    # ── ONE query, then arithmetic ───────────────────────────────────────
    #
    # The grid used to issue one query per combination -- 295 round trips,
    # each rescanning the same trade_paths x tt_bins rows to LEAST a
    # different three or four columns. But an exit is not simulated here: it
    # is a PRECOMPUTED COLUMN, so every combination is arithmetic over stored
    # values that a single fetch can supply in full.
    #
    # So: fetch the union of bar/return columns the sweep touches, once, and
    # resolve every combination in numpy. Cost becomes linear in swept VALUES
    # (columns) rather than multiplicative in combinations, and the query
    # count is bounded by the catalog at 118 exit columns no matter how many
    # families are swept.
    #
    # build_combine_sql REMAINS THE ORACLE. This is a second implementation
    # of the same convention and it is gated against the SQL, permanently,
    # not once at merge -- see verify= below and
    # scripts/check_grid_equivalence.py.
    union_keys: list[str] = []
    for k in held:
        if k not in union_keys:
            union_keys.append(k)
    for f in swept:
        for v in f["values"]:
            if v["rule_key"] not in union_keys:
                union_keys.append(v["rule_key"])
    if HORIZON_RULE_KEY not in union_keys:
        union_keys.append(HORIZON_RULE_KEY)

    sel = []
    for k in union_keys:
        m = by_key[k]
        sel.append(f"tp.{m.bar_col}")
        sel.append(f"tp.{m.ret_col}")

    col_sql = f"""
    SELECT tp.ticker, tp.trade_date, tp.entry_price,
           (tp.trade_date < $2::date) AS is_train,
           {", ".join(sel)}
    FROM tt_bins bt
    JOIN trade_paths tp USING (ticker, trade_date)
    WHERE tp.entry_anchor = $1 AND tp.path_status = 'ok'
          AND {where_bins} AND {cell_pred}{strike_pred}
    -- Row order is part of the contract: max_dd is a running peak-to-trough
    -- over np.cumsum, so every combination must fold the same trades in the
    -- same sequence. Matches the ordering build_combine_sql's consumers use.
    ORDER BY tp.trade_date, tp.ticker
    """
    async with pool.acquire() as conn:
        prows = await conn.fetch(col_sql, *args)

    if not prows:
        return {"error": "the selected zone has no trades"}
    n_tr = len(prows)

    # bar -> float64 with NULL as +inf, DELIBERATELY not NaN. A NULL bar means
    # "this rule never fired", which is what LEAST's NULL-skipping encodes;
    # NaN would propagate through min() and produce a plausible-looking
    # number instead of losing to every real bar.
    bars = np.empty((len(union_keys), n_tr), dtype=np.float64)
    rets = np.empty((len(union_keys), n_tr), dtype=np.float64)
    for ki, k in enumerate(union_keys):
        m = by_key[k]
        bcol, rcol = m.bar_col, m.ret_col
        for i, row in enumerate(prows):
            b = row[bcol]
            bars[ki, i] = np.inf if b is None else float(b)
            r = row[rcol]
            rets[ki, i] = np.nan if r is None else float(r)

    key_ix = {k: i for i, k in enumerate(union_keys)}
    is_train = np.fromiter((1 if r["is_train"] else 0 for r in prows),
                           dtype=np.int8, count=n_tr).astype(bool)
    holds_all = None      # filled per combination from the winning bar

    # The Q1 invariant the backstop depends on: path_status='ok' must imply
    # the horizon bar exists, or LEAST could still be NULL and a trade would
    # never exit. Asserted rather than inherited.
    hz = bars[key_ix[HORIZON_RULE_KEY]]
    n_unbounded = int(np.isinf(hz).sum())
    if n_unbounded:
        return {"error": f"{n_unbounded} of {n_tr} trades have "
                         f"path_status='ok' but no {HORIZON_RULE_KEY} exit "
                         f"bar. The horizon backstop cannot guarantee an "
                         f"exit for them, so no combination is trustworthy. "
                         f"This is an upstream trade_paths problem.",
                "invariant": "backstop_null"}

    def _ordered(rule_keys: list[str]) -> list[str]:
        """Exactly build_combine_sql's ordering, transcribed.

        dedup preserving first occurrence -> append the backstop ONLY when the
        selection contains no max_days at all -> STABLE sort by side priority.
        Stability is load-bearing: two rules of the SAME side tying on a bar
        resolve by the caller's order, which a non-stable sort would silently
        reassign.

        The append condition tracks the oracle, which stopped appending a fixed
        backstop when the selection already caps the hold (source 48aeb5b).
        Appending it unconditionally is not cosmetic once the catalog runs past
        20 sessions: for a swept max_days__40 it puts a 20-session bar into the
        LEAST, and argmin then returns the 20-session exit for a rule that was
        asked for 40 -- a plausible number, not a missing one, and the SQL would
        not agree with it.
        """
        keys = list(dict.fromkeys(rule_keys))
        if not any(by_key[k].family == "max_days" for k in keys):
            keys.append(HORIZON_RULE_KEY)
        return sorted(keys, key=lambda k: SIDE_PRIORITY[by_key[k].side])

    def _resolve(rule_keys: list[str]):
        """LEAST + the winning rule's return, vectorised over all trades.

        np.argmin returns the FIRST occurrence of the minimum, which is
        precisely the SQL CASE's top-down "first WHEN that matches" over the
        same `ordered` list. The tie-break therefore falls out of array
        ordering rather than needing separate logic that could disagree.
        """
        order = _ordered(rule_keys)
        ix = [key_ix[k] for k in order]
        B = bars[ix]                       # (K, n_tr)
        w = np.argmin(B, axis=0)           # first minimum == CASE order
        eb = B[w, np.arange(n_tr)]
        er = rets[ix][w, np.arange(n_tr)]
        return order, w, eb, er

    combos_out = []
    tickers: list = []
    dates: list = []
    tick_ix: dict = {}
    date_ix: dict = {}

    def _ix(v, table, out):
        i = table.get(v)
        if i is None:
            i = len(out); table[v] = i; out.append(v)
        return i

    skel_t, skel_d, skel_p, skel_w = [], [], [], []
    for row in prows:
        skel_t.append(_ix(row["ticker"], tick_ix, tickers))
        skel_d.append(_ix(row["trade_date"].isoformat(), date_ix, dates))
        _p = row["entry_price"]
        skel_p.append(round(float(_p), 4) if _p is not None else None)
        skel_w.append(1 if row["is_train"] else 0)

    tr_mask, te_mask = is_train, ~is_train
    for m in combos_meta:
        order, w, eb, er = _resolve(m["rule_keys"])
        rule_of = np.array(order, dtype=object)[w]

        stats = {}
        reasons = {"train": {}, "test": {}}
        for wn, mask in (("train", tr_mask), ("test", te_mask)):
            n = int(mask.sum())
            if not n:
                stats[wn] = {"n": 0, "avg_ret": None, "avg_hold": None}
                continue
            sub_r = er[mask]
            ok = ~np.isnan(sub_r)
            stats[wn] = {
                "n": int(ok.sum()),
                "avg_ret": float(sub_r[ok].mean()) if ok.any() else None,
                "avg_hold": float(eb[mask][ok].mean() / BARS_PER_SESSION)
                            if ok.any() else None,
            }
            ru, rc = np.unique(rule_of[mask], return_counts=True)
            tot = int(rc.sum()) or 1
            reasons[wn] = {str(a): int(b) / tot for a, b in zip(ru, rc)}

        combos_out.append({
            **m,
            "r": [None if np.isnan(x) else round(float(x), 8) for x in er],
            "train": stats["train"], "test": stats["test"],
            "reasons": reasons,
        })

    # ── Oracle gate ──────────────────────────────────────────────────────
    # Runs the SQL path for a sample of combinations and diffs it against the
    # numpy path, per trade, on (exit_bar, exit_return, exit_rule). Off by
    # default because it reintroduces the per-combination queries this change
    # exists to remove -- on demand, it is the whole point.
    verify_report = None
    if req.verify:
        verify_report = await _grid_verify(
            pool, combos_meta, held, by_key, prows, key_ix, bars, rets,
            _ordered, where_bins, cell_pred, strike_pred, args,
            max(1, min(int(req.verify), len(combos_meta))))
        if verify_report.get("mismatches"):
            return {"error": "VERIFY FAILED — the vectorised path disagrees "
                             "with build_combine_sql. The grid has not been "
                             "rendered.", "verify": verify_report}

    return {
        "sweep": swept,
        "held": held,
        "cutoff_date": cutoff_iso,
        "cells": req.cells,
        "entry_anchor": req.entry_anchor,
        "max_strike": req.max_strike,
        "metrics": [{"key": k, "label": l, "unit": u} for k, l, u in GRID_METRICS],
        "tickers": tickers,
        "dates": dates,
        "skeleton": {"t": skel_t, "d": skel_d, "p": skel_p, "w": skel_w},
        "combos": combos_out,
        "null_index": null_ix,
        # rule_key -> combination index for horizon-only at that horizon.
        # Empty when max_days is not swept, in which case the single null is
        # the whole reference.
        "ref_index": ref_ix,
        "n_combos": len(combos_meta),
        "n_trades": n_tr,
        "n_returns": returns,
        "n_columns": 2 * len(union_keys),
        "queries": 1,
        "concurrency": 1,
        "errors": [],
        "verify": verify_report,
        "units_note": SUITE_UNITS_NOTE,
    }


async def _grid_verify(pool, combos_meta, held, by_key, prows, key_ix,
                       bars, rets, ordered_fn, where_bins, cell_pred,
                       strike_pred, args, sample: int) -> dict:
    """Diff the vectorised path against build_combine_sql, per trade.

    Bit-identical or it fails. There are no legitimate small differences
    here: both paths read the SAME stored columns, so any disagreement is a
    bug in the transcription of LEAST + side-priority, not rounding.

    Samples evenly across the combination list rather than taking the first
    N, so a sweep whose late combinations differ in shape is still covered.
    """
    n = len(combos_meta)
    step = max(1, n // sample)
    picks = list(range(0, n, step))[:sample]
    if n - 1 not in picks:
        picks.append(n - 1)          # the null combination is always last

    checked, mismatches = 0, []
    key_of = [(r["ticker"], r["trade_date"]) for r in prows]
    async with pool.acquire() as conn:
        for pi in picks:
            m = combos_meta[pi]
            rk = m["rule_keys"]
            combine_sql, _meta = build_combine_sql(rk, by_key,
                                                   include_exit_rule=True)
            rows = await conn.fetch(f"""
            WITH c AS (
{combine_sql}
            )
            SELECT c.ticker, c.trade_date, c.exit_bar, c.exit_return, c.exit_rule
            FROM c
            JOIN tt_bins bt USING (ticker, trade_date)
            JOIN trade_paths tp USING (ticker, trade_date, entry_anchor)
            WHERE c.entry_anchor = $1 AND $2::date IS NOT NULL
                  AND {where_bins} AND {cell_pred}{strike_pred}
            ORDER BY c.trade_date, c.ticker
            """, *args)

            order = ordered_fn(rk)
            ix = [key_ix[k] for k in order]
            B = bars[ix]
            w = np.argmin(B, axis=0)
            ar = np.arange(B.shape[1])
            eb = B[w, ar]
            er = rets[ix][w, ar]
            rule_of = np.array(order, dtype=object)[w]

            if len(rows) != len(key_of):
                mismatches.append({"combo": pi, "field": "row_count",
                                   "sql": len(rows), "numpy": len(key_of)})
                continue
            for j, row in enumerate(rows):
                checked += 1
                if (row["ticker"], row["trade_date"]) != key_of[j]:
                    mismatches.append({"combo": pi, "row": j, "field": "identity",
                                       "sql": f"{row['ticker']} {row['trade_date']}",
                                       "numpy": f"{key_of[j][0]} {key_of[j][1]}"})
                    break
                sb = row["exit_bar"]
                if sb is None or float(sb) != float(eb[j]):
                    mismatches.append({"combo": pi, "row": j, "field": "exit_bar",
                                       "sql": sb, "numpy": float(eb[j]),
                                       "trade": f"{key_of[j][0]} {key_of[j][1]}"})
                sr = row["exit_return"]
                nr = None if np.isnan(er[j]) else float(er[j])
                # Exact equality, not a tolerance. Both sides read the same
                # stored float; a tolerance here would hide the very class of
                # bug this exists to catch.
                if (sr is None) != (nr is None) or (
                        sr is not None and float(sr) != nr):
                    mismatches.append({"combo": pi, "row": j, "field": "exit_return",
                                       "sql": sr, "numpy": nr,
                                       "trade": f"{key_of[j][0]} {key_of[j][1]}"})
                if row["exit_rule"] != rule_of[j]:
                    mismatches.append({"combo": pi, "row": j, "field": "exit_rule",
                                       "sql": row["exit_rule"], "numpy": str(rule_of[j]),
                                       "trade": f"{key_of[j][0]} {key_of[j][1]}"})
                if len(mismatches) >= 20:
                    break
            if len(mismatches) >= 20:
                break

    return {"combinations_checked": len(picks), "of": n,
            "trades_compared": checked,
            "mismatches": mismatches[:20],
            "n_mismatches": len(mismatches)}


def _grid_reasons(rows) -> dict:
    """Exit-rule share per window for one combination.

    Server-side because it is a count, not a portfolio quantity -- sizing
    cannot touch it, so shipping it derived costs nothing and shipping the
    rule per trade would cost a third array.
    """
    out = {"train": {}, "test": {}}
    tot = {"train": 0, "test": 0}
    for row in rows:
        w = "train" if row["is_train"] else "test"
        rk = row["exit_rule"]
        if rk is None:
            continue
        out[w][rk] = out[w].get(rk, 0) + 1
        tot[w] += 1
    for w in ("train", "test"):
        n = tot[w] or 1
        out[w] = {k: v / n for k, v in out[w].items()}
    return out
