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

import json
from datetime import date as _date
from typing import Any, Optional

import numpy as np

from fastapi import APIRouter, Body, Depends, Query
from pydantic import BaseModel

from app.db import get_oi_pool
from app.trade_path_rules import (
    CombineError,
    HORIZON_RULE_KEY,
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


def _rule_label(family: str, params: Any) -> str:
    """Human label for a rule option, derived from its params JSON.

    Never parses the rule_key — the key's encoding (2p5 for 2.5) is a column
    naming artefact, not a display format. An unrecognised family falls back
    to rendering the params dict, which degrades to something readable rather
    than raising on a rule this UI has not seen.
    """
    if isinstance(params, str):
        try:
            params = json.loads(params)
        except (ValueError, TypeError):
            params = {}
    params = params or {}
    pct = params.get("pct")
    k = params.get("k")
    n = params.get("n") or params.get("days") or params.get("bars")
    if pct is not None:
        return f"{float(pct):g}%"
    if k is not None:
        return f"{float(k):g}x ATR"
    if n is not None:
        return f"{int(n)}d"
    if not params:
        return family
    return ", ".join(f"{a}={b}" for a, b in sorted(params.items()))


def _window_stats(rets: list, holds: list, tickers: set, span_days: float) -> dict:
    """Full stat set for one window. Matches Recall's box set plus the three
    this page needs (Calmar, Max DD, Avg Hold).

    max_dd is peak-to-trough on the ADDITIVE cumulative return curve, the same
    convention the equity panes use; Calmar is total return over |max_dd|.
    Hold is converted bars -> sessions here so no consumer has to know the
    bar size.
    """
    n = len(rets)
    if not n:
        return {"n": 0, "n_tickers": len(tickers), "avg_ret": 0.0, "median": 0.0,
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
        "n_tickers": len(tickers),
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
        fam.append({
            "rule_key":   r["rule_key"],
            "label":      _rule_label(r["family"], r["params"]),
            "is_horizon": bool(r["is_horizon"]),
        })

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


class RunReq(BaseModel):
    primary_metric:   str
    secondary_metric: Optional[str] = None      # None => single-metric mode
    entry_anchor:     str = "open"
    rule_keys:        list[str] = []
    n_bins:           int = 20
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

        n_bins = max(2, min(20, int(req.n_bins)))
        # Canonical bin20 collapse, identical to every other stored-bin
        # surface in this codebase.
        def _collapse(col: str) -> str:
            return f"LEAST(((bt.{col} - 1) * {n_bins}) / 20, {n_bins} - 1)"

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
        WHERE c.entry_anchor = $1 AND {where_bins}
        GROUP BY {grp}, c.exit_rule, is_train
        """
        rows = await conn.fetch(sql, req.entry_anchor, cutoff_d)

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
        WHERE c.entry_anchor = $1 AND {where_bins}
        ORDER BY c.trade_date
        """
        srows = await conn.fetch(stat_sql, req.entry_anchor, cutoff_d)

    # ── Fold into grid + breakdown ────────────────────────────────────────
    grid = [[None] * n_bins for _ in range(n_bins if two_factor else 1)]
    acc: dict = {}
    reasons: dict = {}
    tot = {"train": [0, 0.0, 0.0], "test": [0, 0.0, 0.0]}   # n, sum_ret, sum_hold
    for r in rows:
        win = "train" if r["is_train"] else "test"
        bp, bs = int(r["bp"]), (int(r["bs"]) if two_factor else 0)
        n, avg, hold = int(r["n"]), float(r["avg_ret"] or 0), float(r["avg_hold"] or 0)
        cell = acc.setdefault((bs, bp), {"train": [0, 0.0], "test": [0, 0.0]})
        cell[win][0] += n
        cell[win][1] += avg * n
        t = tot[win]
        t[0] += n; t[1] += avg * n; t[2] += hold * n
        if win == "train":
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

    acc2 = {"train": ([], [], set(), []), "test": ([], [], set(), [])}
    for r in srows:
        w = "train" if r["is_train"] else "test"
        rets, holds, tks, dates = acc2[w]
        rets.append(float(r["exit_return"] or 0.0))
        holds.append(float(r["exit_bar"] or 0.0))
        tks.add(r["ticker"]); dates.append(r["trade_date"])

    def _stats(win: str) -> dict:
        rets, holds, tks, dates = acc2[win]
        span = ((max(dates) - min(dates)).days if len(dates) > 1 else 1)
        return _window_stats(rets, holds, tks, span)

    # Exit-reason breakdown. A user-selected max_days and the auto-appended
    # backstop are the SAME column with opposite meanings, so the backstop is
    # labelled as such ONLY when it was auto-added.
    tot_train = tot["train"][0] or 1
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
            "frac":        n / tot_train,
        })

    return {
        "mode":             "2f" if two_factor else "1f",
        "primary_metric":   req.primary_metric,
        "secondary_metric": req.secondary_metric,
        "entry_anchor":     req.entry_anchor,
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


class ZoneReq(RunReq):
    """A run plus the selected cells. Cells are [bp, bs] pairs at the run's
    n_bins resolution (bs is ignored in single-metric mode)."""
    cells: list[list[int]] = []


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
    if not req.cells:
        return {"error": "no cells selected"}

    from app.routers.oi_analysis import _get_tt_cutoff, _sec_equity_curve

    async with pool.acquire() as conn:
        rules = await _load_rules(conn)
        if not rules:
            return {"error": "trade_path_rules is empty"}
        by_key = by_key_from_rows(rules)
        meta_by_key = {r["rule_key"]: r for r in rules}

        bin_cols = await _bin_columns(conn)
        p_col = f"bin20_{req.primary_metric}"
        if p_col not in bin_cols:
            return {"error": f"no stored bins for {req.primary_metric!r} in tt_bins"}
        two_factor = bool(req.secondary_metric)
        s_col = f"bin20_{req.secondary_metric}" if two_factor else None
        if two_factor and s_col not in bin_cols:
            return {"error": f"no stored bins for {req.secondary_metric!r} in tt_bins"}

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

        n_bins = max(2, min(20, int(req.n_bins)))

        def _collapse(col: str) -> str:
            return f"LEAST(((bt.{col} - 1) * {n_bins}) / 20, {n_bins} - 1)"

        # Cells arrive as ints from the client but are interpolated into SQL,
        # so they are coerced and range-checked here rather than trusted.
        bps, bss = [], []
        for c in req.cells:
            if not c:
                continue
            bp = int(c[0])
            bs = int(c[1]) if (two_factor and len(c) > 1) else 0
            if not (0 <= bp < n_bins) or not (0 <= bs < n_bins):
                return {"error": f"cell out of range for n_bins={n_bins}: {c}"}
            bps.append(bp); bss.append(bs)
        if not bps:
            return {"error": "no valid cells"}

        if two_factor:
            cell_pred = ("(" + _collapse(p_col) + ", " + _collapse(s_col) +
                         ") IN (SELECT * FROM unnest($3::int[], $4::int[]))")
            args = [req.entry_anchor, cutoff_d, bps, bss]
        else:
            cell_pred = _collapse(p_col) + " = ANY($3::int[])"
            args = [req.entry_anchor, cutoff_d, bps]

        where_bins = f"bt.{p_col} > 0" + (f" AND bt.{s_col} > 0" if two_factor else "")

        sql = f"""
        WITH c AS (
{combine_sql}
        )
        SELECT c.ticker, c.trade_date, c.exit_bar, c.exit_return, c.exit_rule,
               tp.entry_price,
               (c.trade_date < $2::date) AS is_train
        FROM c
        JOIN tt_bins bt USING (ticker, trade_date)
        -- entry_price is not exposed by build_combine_sql's outer SELECT, and
        -- the vendored function is not the place to add it. Rejoining
        -- trade_paths on its primary key is cheap and leaves the combine
        -- untouched.
        JOIN trade_paths tp USING (ticker, trade_date, entry_anchor)
        WHERE c.entry_anchor = $1 AND {where_bins} AND {cell_pred}
        ORDER BY c.trade_date, c.ticker
        """
        rows = await conn.fetch(sql, *args)

        td_rows = await conn.fetch(
            "SELECT DISTINCT trade_date FROM tt_bins ORDER BY trade_date")
        trading_days = [r["trade_date"].isoformat() for r in td_rows]

    # ── Fold into the Recall chart contracts ──────────────────────────────
    trades, dates, reasons = [], [], {}
    zacc = {"train": ([], [], set(), []), "test": ([], [], set(), [])}
    by_ticker: dict[str, list] = {}
    tot = {"train": [0, 0.0, 0.0], "test": [0, 0.0, 0.0]}
    for r in rows:
        win = "train" if r["is_train"] else "test"
        ret = float(r["exit_return"] or 0.0)
        hold = float(r["exit_bar"] or 0.0)
        d = r["trade_date"].isoformat()
        t = tot[win]
        t[0] += 1; t[1] += ret; t[2] += hold
        if win == "train":
            reasons[r["exit_rule"]] = reasons.get(r["exit_rule"], 0) + 1
        za = zacc[win]
        za[0].append(ret); za[1].append(hold); za[2].add(r["ticker"]); za[3].append(r["trade_date"])
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
        trades.append({
            "ticker": r["ticker"], "trade_date": d,
            "ret": ret, "exit_bar": hold, "exit_rule": r["exit_rule"],
            "window": win,
            "entry_price":    float(_px) if _px is not None else None,
            "spot_entry_raw": float(_px) if _px is not None else None,
        })
        dates.append(d)
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

    tot_train = tot["train"][0] or 1
    breakdown = []
    for rk, n in sorted(reasons.items(), key=lambda kv: -kv[1]):
        m = meta_by_key.get(rk, {})
        is_backstop = (rk == HORIZON_RULE_KEY and combine_meta["horizon_auto_added"])
        breakdown.append({
            "rule_key": rk, "family": m.get("family", rk), "side": m.get("side", ""),
            "label": ("backstop — no selected rule fired" if is_backstop
                      else f"{m.get('family', rk)} {_rule_label(m.get('family',''), m.get('params'))}"),
            "is_backstop": is_backstop, "n": n, "frac": n / tot_train,
        })

    return {
        "cells": req.cells,
        "entry_anchor": req.entry_anchor,
        "cutoff_date": cutoff_iso,
        # Recall chart contracts — same field names, same shapes.
        "combined_trades": trades,
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
