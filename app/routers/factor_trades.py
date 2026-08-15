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

    def _stats(win: str) -> dict:
        n, s, h = tot[win]
        return {"n": n,
                "avg_ret":  (s / n) if n else 0.0,
                "avg_hold": (h / n) if n else 0.0}

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
