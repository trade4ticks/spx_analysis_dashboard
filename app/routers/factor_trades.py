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


def _effective_tickers(counts) -> float:
    """Participation ratio: (sum n)^2 / sum(n^2).

    Reads ~N when N tickers contribute equally and ~k when k dominate, so it
    says at a glance whether a zone's edge is broad or concentrated. Same
    form as an effective-N / inverse-Herfindahl.
    """
    c = [x for x in counts if x > 0]
    if not c:
        return 0.0
    tot = float(sum(c))
    return (tot * tot) / float(sum(x * x for x in c))


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

        # Same population filter as /run. The placeholder index depends on how
        # many args the cell predicate already consumed, so it is computed
        # rather than hardcoded.
        strike_pred = ""
        if req.max_strike and req.max_strike > 0:
            strike_pred = f" AND tp.entry_price <= ${len(args) + 1}"
            args = args + [float(req.max_strike)]

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
    SELECT c.ticker, c.trade_date, c.exit_bar, c.exit_return,
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
    return {**out, "rule_keys": list(rule_keys),
            "train": st["train"], "test": st["test"],
            "rows": rows,
            "horizon_auto_added": meta["horizon_auto_added"]}


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
    if not req.cells:
        return {"error": "no cells selected"}

    from app.routers.oi_analysis import _get_tt_cutoff

    n = max(1, min(50, int(req.n_draws)))

    async with pool.acquire() as conn:
        rules = await _load_rules(conn)
        if not rules:
            return {"error": "trade_path_rules is empty"}
        by_key = by_key_from_rows(rules)

        bin_cols = await _bin_columns(conn)
        p_col = f"bin20_{req.primary_metric}"
        if p_col not in bin_cols:
            return {"error": f"no stored bins for {req.primary_metric!r} in tt_bins"}
        two_factor = bool(req.secondary_metric)
        s_col = f"bin20_{req.secondary_metric}" if two_factor else None
        if two_factor and s_col not in bin_cols:
            return {"error": f"no stored bins for {req.secondary_metric!r} in tt_bins"}

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
