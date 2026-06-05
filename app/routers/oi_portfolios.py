"""System Portfolio Builder — third analysis tier on the OI Analysis page.

A *Portfolio* is a saved research artifact anchored to a single
(ticker, outcome, date_from, date_to) tuple. It contains one or more
*Systems*, each of which is a primary-metric+bins selection combined with
a set of secondary-metric+bins filters. The portfolio's trades are the
union of every enabled system's trades.

CRUD lives in the main app DB (via get_pool). The aggregate endpoint
queries daily_features (via get_oi_pool) to compute the union stats
and two phi-correlation heatmaps (system-level and pair-level).
"""
from collections import defaultdict
from datetime import date as _date
from typing import List, Optional

import numpy as np
from fastapi import APIRouter, Body, Depends, HTTPException
from pydantic import BaseModel

from app.db import get_pool, get_oi_pool
from app.routers.oi_analysis import (
    _DEFAULT_WALKFWD_WARMUP,
    _sec_equity_curve,
    _parse_horizon,
    _fetch_ticker_calendars,
    _build_enriched_trade,
)
# _bin_membership and _walk_forward_bins were used directly in the
# per-system loop pre-Step-5; PortfolioVectorBuilder in row_compute.py
# now owns both calls and imports them lazily.

router = APIRouter(tags=["oi_portfolios"])


_DDL = """
CREATE TABLE IF NOT EXISTS oi_research_portfolios (
    id          SERIAL PRIMARY KEY,
    name        TEXT NOT NULL,
    description TEXT,
    ticker      TEXT NOT NULL,
    outcome     TEXT NOT NULL,
    date_from   TEXT,
    date_to     TEXT,
    created_at  TIMESTAMPTZ DEFAULT NOW(),
    updated_at  TIMESTAMPTZ DEFAULT NOW()
);
-- Per-portfolio "watch this on the Signals page" flag. Default off so
-- existing portfolios don't auto-fire signals until the user opts in.
ALTER TABLE oi_research_portfolios
    ADD COLUMN IF NOT EXISTS monitored BOOLEAN DEFAULT FALSE;
CREATE TABLE IF NOT EXISTS oi_research_systems (
    id                SERIAL PRIMARY KEY,
    portfolio_id      INTEGER REFERENCES oi_research_portfolios(id) ON DELETE CASCADE,
    name              TEXT NOT NULL,
    enabled           BOOLEAN DEFAULT TRUE,
    position          INTEGER DEFAULT 0,
    primary_metric    TEXT NOT NULL,
    primary_bins      INTEGER[] NOT NULL,
    primary_bin_count INTEGER NOT NULL DEFAULT 20,
    secondaries       JSONB NOT NULL,
    created_at        TIMESTAMPTZ DEFAULT NOW(),
    updated_at        TIMESTAMPTZ DEFAULT NOW()
);
-- System library: anchor-agnostic templates that can be added to any
-- portfolio. Stores the system definition (primary metric + bins +
-- secondaries) without binding to any specific ticker/outcome/dates.
CREATE TABLE IF NOT EXISTS oi_research_system_library (
    id                SERIAL PRIMARY KEY,
    name              TEXT NOT NULL,
    description       TEXT,
    primary_metric    TEXT NOT NULL,
    primary_bins      INTEGER[] NOT NULL,
    primary_bin_count INTEGER NOT NULL DEFAULT 20,
    secondaries       JSONB NOT NULL,
    created_at        TIMESTAMPTZ DEFAULT NOW(),
    updated_at        TIMESTAMPTZ DEFAULT NOW()
);
ALTER TABLE oi_research_systems        ADD COLUMN IF NOT EXISTS is_short BOOLEAN DEFAULT FALSE;
ALTER TABLE oi_research_system_library ADD COLUMN IF NOT EXISTS is_short BOOLEAN DEFAULT FALSE;
"""


async def _ensure_tables(pool):
    async with pool.acquire() as conn:
        await conn.execute(_DDL)


# ── Pydantic bodies ─────────────────────────────────────────────────────────


class PortfolioIn(BaseModel):
    name: str
    description: Optional[str] = None
    ticker: str
    outcome: str
    date_from: Optional[str] = None
    date_to: Optional[str] = None


class PortfolioUpdate(BaseModel):
    name:        Optional[str]  = None
    description: Optional[str]  = None
    monitored:   Optional[bool] = None


class SecondarySpec(BaseModel):
    metric: str
    bins: List[int]
    bin_count: int = 10


class LibraryItemIn(BaseModel):
    name: str
    description: Optional[str] = None
    primary_metric: str
    primary_bins: List[int]
    primary_bin_count: int = 20
    secondaries: List[SecondarySpec] = []
    is_short: bool = False


class LibraryItemUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    primary_metric: Optional[str] = None
    primary_bins: Optional[List[int]] = None
    primary_bin_count: Optional[int] = None
    secondaries: Optional[List[SecondarySpec]] = None
    is_short: Optional[bool] = None


class SystemIn(BaseModel):
    name: Optional[str] = None
    primary_metric: str
    primary_bins: List[int]
    primary_bin_count: int = 20
    secondaries: List[SecondarySpec] = []
    is_short: bool = False


class SystemUpdate(BaseModel):
    name: Optional[str] = None
    enabled: Optional[bool] = None
    position: Optional[int] = None
    primary_metric: Optional[str] = None
    primary_bins: Optional[List[int]] = None
    primary_bin_count: Optional[int] = None
    secondaries: Optional[List[SecondarySpec]] = None
    is_short: Optional[bool] = None


# ── Helpers ─────────────────────────────────────────────────────────────────


def _portfolio_row_to_dict(r) -> dict:
    d = dict(r)
    for k in ("created_at", "updated_at"):
        if d.get(k) is not None:
            d[k] = str(d[k])[:19]
    return d


def _system_row_to_dict(r) -> dict:
    d = dict(r)
    # asyncpg returns JSONB as Python obj; arrays as list. Normalise dates.
    for k in ("created_at", "updated_at"):
        if d.get(k) is not None:
            d[k] = str(d[k])[:19]
    if isinstance(d.get("secondaries"), str):
        import json as _json
        d["secondaries"] = _json.loads(d["secondaries"])
    return d


# ── Portfolio CRUD ──────────────────────────────────────────────────────────


@router.get("/portfolios")
async def list_portfolios(pool=Depends(get_pool)):
    await _ensure_tables(pool)
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """SELECT p.id, p.name, p.description, p.ticker, p.outcome,
                      p.date_from, p.date_to, p.monitored,
                      p.created_at, p.updated_at,
                      (SELECT COUNT(*) FROM oi_research_systems s
                        WHERE s.portfolio_id = p.id) AS system_count
               FROM oi_research_portfolios p
               ORDER BY p.updated_at DESC""")
    return [_portfolio_row_to_dict(r) for r in rows]


@router.post("/portfolios")
async def create_portfolio(body: PortfolioIn, pool=Depends(get_pool)):
    await _ensure_tables(pool)
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """INSERT INTO oi_research_portfolios
                 (name, description, ticker, outcome, date_from, date_to)
               VALUES ($1, $2, $3, $4, $5, $6)
               RETURNING id, name, description, ticker, outcome,
                         date_from, date_to, monitored, created_at, updated_at""",
            body.name, body.description, body.ticker, body.outcome,
            body.date_from, body.date_to)
    return _portfolio_row_to_dict(row)


@router.put("/portfolios/{pid}")
async def update_portfolio(pid: int, body: PortfolioUpdate,
                           pool=Depends(get_pool)):
    sets, params, p = [], [], 1
    if body.name is not None:
        sets.append(f"name = ${p}"); params.append(body.name); p += 1
    if body.description is not None:
        sets.append(f"description = ${p}"); params.append(body.description); p += 1
    if body.monitored is not None:
        sets.append(f"monitored = ${p}"); params.append(body.monitored); p += 1
    if not sets:
        raise HTTPException(400, "no fields to update")
    sets.append("updated_at = NOW()")
    params.append(pid)
    sql = (f"UPDATE oi_research_portfolios SET {', '.join(sets)} "
           f"WHERE id = ${p} RETURNING id, name, description, ticker, outcome, "
           f"date_from, date_to, monitored, created_at, updated_at")
    async with pool.acquire() as conn:
        row = await conn.fetchrow(sql, *params)
    if not row:
        raise HTTPException(404, "portfolio not found")
    return _portfolio_row_to_dict(row)


@router.delete("/portfolios/{pid}")
async def delete_portfolio(pid: int, pool=Depends(get_pool)):
    async with pool.acquire() as conn:
        await conn.execute("DELETE FROM oi_research_portfolios WHERE id = $1", pid)
    return {"ok": True}


@router.get("/portfolios/{pid}")
async def get_portfolio(pid: int, pool=Depends(get_pool)):
    await _ensure_tables(pool)
    async with pool.acquire() as conn:
        prow = await conn.fetchrow(
            """SELECT id, name, description, ticker, outcome,
                      date_from, date_to, monitored, created_at, updated_at
               FROM oi_research_portfolios WHERE id = $1""", pid)
        if not prow:
            raise HTTPException(404, "portfolio not found")
        srows = await conn.fetch(
            """SELECT id, portfolio_id, name, enabled, position,
                      primary_metric, primary_bins, primary_bin_count, secondaries,
                      is_short, created_at, updated_at
               FROM oi_research_systems
               WHERE portfolio_id = $1
               ORDER BY position, id""", pid)
    return {
        "portfolio": _portfolio_row_to_dict(prow),
        "systems":   [_system_row_to_dict(r) for r in srows],
    }


# ── System CRUD ─────────────────────────────────────────────────────────────


@router.post("/portfolios/{pid}/systems")
async def add_system(pid: int, body: SystemIn, pool=Depends(get_pool)):
    await _ensure_tables(pool)
    import json as _json
    async with pool.acquire() as conn:
        port = await conn.fetchrow(
            "SELECT id FROM oi_research_portfolios WHERE id = $1", pid)
        if not port:
            raise HTTPException(404, "portfolio not found")
        # Default name + position
        cnt = await conn.fetchval(
            "SELECT COUNT(*) FROM oi_research_systems WHERE portfolio_id = $1",
            pid)
        name = body.name or f"System {cnt + 1}"
        secondaries_json = _json.dumps([s.dict() for s in body.secondaries])
        row = await conn.fetchrow(
            """INSERT INTO oi_research_systems
                 (portfolio_id, name, position, primary_metric, primary_bins,
                  primary_bin_count, secondaries, is_short)
               VALUES ($1, $2, $3, $4, $5, $6, $7::jsonb, $8)
               RETURNING id, portfolio_id, name, enabled, position,
                         primary_metric, primary_bins, primary_bin_count,
                         secondaries, is_short, created_at, updated_at""",
            pid, name, cnt, body.primary_metric, body.primary_bins,
            body.primary_bin_count, secondaries_json, body.is_short)
        await conn.execute(
            "UPDATE oi_research_portfolios SET updated_at = NOW() WHERE id = $1",
            pid)
    return _system_row_to_dict(row)


@router.put("/portfolios/{pid}/systems/{sid}")
async def update_system(pid: int, sid: int, body: SystemUpdate,
                        pool=Depends(get_pool)):
    import json as _json
    sets, params, p = [], [], 1
    if body.name is not None:
        sets.append(f"name = ${p}"); params.append(body.name); p += 1
    if body.enabled is not None:
        sets.append(f"enabled = ${p}"); params.append(body.enabled); p += 1
    if body.position is not None:
        sets.append(f"position = ${p}"); params.append(body.position); p += 1
    if body.primary_metric is not None:
        sets.append(f"primary_metric = ${p}"); params.append(body.primary_metric); p += 1
    if body.primary_bins is not None:
        sets.append(f"primary_bins = ${p}"); params.append(body.primary_bins); p += 1
    if body.primary_bin_count is not None:
        sets.append(f"primary_bin_count = ${p}"); params.append(body.primary_bin_count); p += 1
    if body.secondaries is not None:
        sets.append(f"secondaries = ${p}::jsonb")
        params.append(_json.dumps([s.dict() for s in body.secondaries])); p += 1
    if body.is_short is not None:
        sets.append(f"is_short = ${p}"); params.append(body.is_short); p += 1
    if not sets:
        raise HTTPException(400, "no fields to update")
    sets.append("updated_at = NOW()")
    params.extend([pid, sid])
    sql = (f"UPDATE oi_research_systems SET {', '.join(sets)} "
           f"WHERE portfolio_id = ${p} AND id = ${p + 1} "
           f"RETURNING id, portfolio_id, name, enabled, position, "
           f"primary_metric, primary_bins, primary_bin_count, secondaries, "
           f"is_short, created_at, updated_at")
    async with pool.acquire() as conn:
        row = await conn.fetchrow(sql, *params)
        if not row:
            raise HTTPException(404, "system not found")
        await conn.execute(
            "UPDATE oi_research_portfolios SET updated_at = NOW() WHERE id = $1",
            pid)
    return _system_row_to_dict(row)


@router.delete("/portfolios/{pid}/systems/{sid}")
async def delete_system(pid: int, sid: int, pool=Depends(get_pool)):
    async with pool.acquire() as conn:
        await conn.execute(
            "DELETE FROM oi_research_systems WHERE portfolio_id = $1 AND id = $2",
            pid, sid)
        await conn.execute(
            "UPDATE oi_research_portfolios SET updated_at = NOW() WHERE id = $1",
            pid)
    return {"ok": True}


# ── Aggregate ───────────────────────────────────────────────────────────────


async def _fetch_anchor_rows(oi_pool, ticker: str, outcome: str,
                             date_from: Optional[str], date_to: Optional[str],
                             metrics: list) -> list:
    """Fetch daily_features rows for the portfolio anchor.

    Returns rows sorted by (trade_date, ticker) so binary vectors align.
    Only the columns referenced by any system are selected (plus
    trade_date / ticker / outcome).
    """
    cols = ["trade_date", "ticker", outcome] + sorted(set(metrics))
    # Deduplicate while preserving order
    seen, ordered_cols = set(), []
    for c in cols:
        if c not in seen:
            ordered_cols.append(c); seen.add(c)
    col_sql = ", ".join(ordered_cols)
    where = [f"{outcome} IS NOT NULL"]
    params: list = []
    p = 1
    if ticker and ticker != "ALL":
        where.append(f"ticker = ${p}"); params.append(ticker); p += 1
    if date_from:
        where.append(f"trade_date >= ${p}")
        params.append(_date.fromisoformat(date_from)); p += 1
    if date_to:
        where.append(f"trade_date <= ${p}")
        params.append(_date.fromisoformat(date_to)); p += 1
    where_sql = " AND ".join(where)
    sql = (f"SELECT {col_sql} FROM daily_features "
           f"WHERE {where_sql} ORDER BY trade_date, ticker")
    async with oi_pool.acquire() as conn:
        rows = await conn.fetch(sql, *params)
    # Convert to plain dicts so .get() works, normalise trade_date to str
    out = []
    for r in rows:
        d = dict(r)
        td = d.get("trade_date")
        if td is not None:
            d["trade_date"] = td.isoformat() if hasattr(td, "isoformat") else str(td)
        out.append(d)
    return out


def _stats_from_trades(trades: list, outcome_col: str) -> dict:
    """Aggregate stats (n, win_rate, avg_ret, contrib_pct) from a list of trade rows."""
    rets = []
    for r in trades:
        v = r.get(outcome_col)
        if v is None:
            continue
        try:
            fv = float(v)
            if not np.isnan(fv):
                rets.append(fv)
        except (TypeError, ValueError):
            continue
    n = len(rets)
    if n == 0:
        return {"n": 0, "win_rate": 0.0, "avg_ret": 0.0, "sum_ret": 0.0}
    arr = np.array(rets)
    return {
        "n":        n,
        "win_rate": round(float((arr > 0).mean()), 4),
        "avg_ret":  round(float(arr.mean()), 6),
        "sum_ret":  float(arr.sum()),
    }


def _signed_row(row: dict, sign: int, outcome_col: str) -> dict:
    """Return row unchanged if sign==+1, else a copy with outcome negated."""
    if sign == 1:
        return row
    r = dict(row)
    v = r.get(outcome_col)
    if v is not None:
        try:
            r[outcome_col] = -float(v)
        except (TypeError, ValueError):
            pass
    return r


class AggregateRequest(BaseModel):
    walk_forward: bool = False
    cutoff_date: Optional[str] = None  # train-test mode when set (takes precedence over walk_forward)


@router.post("/portfolios/{pid}/aggregate")
async def portfolio_aggregate(pid: int,
                              req: AggregateRequest = Body(default_factory=AggregateRequest),
                              pool=Depends(get_pool),
                              oi_pool=Depends(get_oi_pool)):
    """Compute union stats + system/pair heatmaps for the portfolio."""
    if not oi_pool:
        return {"error": "OI database not configured"}
    await _ensure_tables(pool)
    async with pool.acquire() as conn:
        prow = await conn.fetchrow(
            """SELECT id, name, description, ticker, outcome,
                      date_from, date_to
               FROM oi_research_portfolios WHERE id = $1""", pid)
        if not prow:
            raise HTTPException(404, "portfolio not found")
        srows = await conn.fetch(
            """SELECT id, name, enabled, position,
                      primary_metric, primary_bins, primary_bin_count, secondaries, is_short
               FROM oi_research_systems
               WHERE portfolio_id = $1
               ORDER BY position, id""", pid)

    portfolio = _portfolio_row_to_dict(prow)
    systems_all = [_system_row_to_dict(r) for r in srows]
    enabled_systems = [s for s in systems_all if s["enabled"]]

    if not enabled_systems:
        return {
            "portfolio": portfolio,
            "systems": [{"id": s["id"], "name": s["name"], "enabled": s["enabled"],
                         "n_trades": 0, "win_rate": 0.0, "avg_ret": 0.0,
                         "contrib_pct": 0.0} for s in systems_all],
            "baseline_n": 0, "combined_n": 0, "horizon": _parse_horizon(portfolio["outcome"]),
            "equity_primary": [], "equity_combined": [], "yearly": [], "tickers": [],
            "combined_trade_dates": [],
            "winner_avg_ret": 0.0, "loser_avg_ret": 0.0,
            "n_each": [], "utilisation": 100.0,
            "phi_systems": [], "overlap_systems": [], "system_labels": [],
            "phi_pairs": [], "overlap_pairs": [], "pair_labels": [],
            "system_boundaries": [],
        }

    # Collect every metric referenced (primary + secondaries) so the fetch is minimal.
    needed_metrics = set()
    for s in enabled_systems:
        needed_metrics.add(s["primary_metric"])
        for sec in (s.get("secondaries") or []):
            needed_metrics.add(sec["metric"])

    ticker  = portfolio["ticker"]
    outcome = portfolio["outcome"]
    rows = await _fetch_anchor_rows(
        oi_pool, ticker, outcome,
        portfolio.get("date_from"), portfolio.get("date_to"),
        sorted(needed_metrics))
    if not rows:
        return {"error": "no rows match the portfolio anchor"}

    is_all = (ticker == "ALL")

    # Build per-system and per-pair binary vectors
    system_vectors = []
    system_labels = []
    pair_vectors = []
    pair_labels = []
    system_boundaries = []   # cumulative pair count after each system
    per_system_pair_n = []
    primary_vectors = []     # per-system V_primary (for the trade-eligible universe)

    # Spec-dispatched per-system loop via PortfolioVectorBuilder. The
    # builder encapsulates the spec-by-spec path selection plus the
    # per-(metric, n_bins) caching that the legacy `wf_cache` provided.
    # Single path; no `if req.walk_forward` inside this loop.
    from app.routers.row_compute import make_spec, PortfolioVectorBuilder
    spec = make_spec(req.walk_forward, req.cutoff_date)

    # Group 6: for IS+ALL, prefetch bin20 from is_bins for every metric
    # referenced by any system (primary + secondaries). The builder uses
    # this lookup to derive both primary and secondary bins from the
    # stored is_bins.bin20 instead of computing per-ticker ranks live —
    # closes the re-rank-on-filtered-subset asymmetry that the legacy
    # IS path had on secondaries, and matches the bin assignments every
    # other on-screen view shows (heatmap, /analyze, corr explorer).
    # WF/TT and single-ticker paths get an empty lookup and fall
    # through to the existing on-the-fly bin maps.
    bin20_by_metric: dict = {}
    if spec.kind == "in_sample" and is_all:
        from app.routers.oi_analysis import _fetch_bin20_by_metric
        filter_pairs = [(r["ticker"], r["trade_date"]) for r in rows]
        bin20_by_metric = await _fetch_bin20_by_metric(
            oi_pool, sorted(needed_metrics), filter_pairs)

    builder = PortfolioVectorBuilder(
        spec, rows, is_all, bin20_by_metric=bin20_by_metric)
    cleared_any: set = set()  # rows cleared by ANY system's primary metric

    for s in enabled_systems:
        prim_bins  = set(int(b) for b in (s.get("primary_bins") or []))
        prim_count = int(s.get("primary_bin_count") or 20)

        V_p = builder.primary_vector(s["primary_metric"], prim_bins, prim_count)
        # primary_cleared_indices returns empty set in in-sample mode,
        # which is exactly legacy's behaviour (wf_dropped is hardcoded to
        # 0 in the response metadata branch for in_sample).
        cleared_any.update(builder.primary_cleared_indices(s["primary_metric"], prim_count))

        primary_vectors.append(V_p)
        primary_indices = [i for i, v in enumerate(V_p) if v == 1.0]
        primary_rows = [rows[i] for i in primary_indices]

        secs = s.get("secondaries") or []
        if not secs or not prim_bins or not primary_rows:
            zero = np.zeros(len(rows))
            system_vectors.append(zero)
            system_labels.append(s["name"])
            per_system_pair_n.append(0)
            system_boundaries.append(len(pair_vectors))
            continue

        expanded_sec_vecs = []
        for sec in secs:
            sec_bins  = set(int(b) for b in (sec.get("bins") or []))
            sec_count = int(sec.get("bin_count") or 10)

            V_si_full = builder.secondary_vector(
                sec["metric"], sec_bins, sec_count, primary_indices,
            )

            expanded_sec_vecs.append(V_si_full)
            pair_vectors.append(V_si_full)
            pair_labels.append(f"{s['name']}: {sec['metric']}")

        V_S = np.max(np.stack(expanded_sec_vecs), axis=0)
        system_vectors.append(V_S)
        system_labels.append(s["name"])
        per_system_pair_n.append(len(expanded_sec_vecs))
        system_boundaries.append(len(pair_vectors))

    # Portfolio union = OR across enabled systems
    M_sys = np.stack(system_vectors) if system_vectors else np.zeros((1, len(rows)))
    V_port = (M_sys.sum(axis=0) > 0).astype(float)
    union_rows = [rows[i] for i, v in enumerate(V_port) if v == 1.0]

    # Per-row direction: +1 if any long system fires, -1 if only short systems fire.
    system_is_short = [bool(s.get("is_short", False)) for s in enabled_systems]
    union_sign_map: dict = {}
    for r_idx, vp in enumerate(V_port):
        if vp != 1.0:
            continue
        has_long = any(
            (not system_is_short[k]) and (system_vectors[k][r_idx] == 1.0)
            for k in range(len(system_vectors))
        )
        union_sign_map[r_idx] = +1 if has_long else -1
    union_rows_signed = [
        _signed_row(rows[i], union_sign_map.get(i, +1), outcome)
        for i, v in enumerate(V_port) if v == 1.0
    ]

    # Trade-eligible universe: union of all enabled systems' primary filters.
    # phi correlations restrict to this universe so values match the corr
    # explorer (which only sees primary-filtered rows). The "primary" equity
    # curve / yearly baseline use this universe too.
    if primary_vectors:
        M_prim = np.stack(primary_vectors)
        universe_mask = (M_prim.sum(axis=0) > 0)
    else:
        universe_mask = np.zeros(len(rows), dtype=bool)
    universe_idx = np.where(universe_mask)[0]
    universe_rows = [rows[i] for i in universe_idx.tolist()]

    # Per-system summary
    sys_stats = []
    total_sum_ret_union = 0.0
    for r in union_rows_signed:
        v = r.get(outcome)
        if v is not None:
            try:
                total_sum_ret_union += float(v)
            except (TypeError, ValueError):
                pass
    for s, vec, is_short_sys in zip(enabled_systems, system_vectors, system_is_short):
        sys_rows = [rows[i] for i, v in enumerate(vec) if v == 1.0]
        if is_short_sys:
            sys_rows = [_signed_row(r, -1, outcome) for r in sys_rows]
        st = _stats_from_trades(sys_rows, outcome)
        sys_stats.append({
            "id": s["id"], "name": s["name"], "enabled": True,
            "is_short": s.get("is_short", False),
            "n_trades": st["n"], "win_rate": st["win_rate"],
            "avg_ret": st["avg_ret"],
            "contrib_pct": round(st["sum_ret"] / total_sum_ret_union * 100, 2)
                            if total_sum_ret_union else 0.0,
        })
    # Include disabled systems in the output so the UI can show them too
    enabled_ids = {s["id"] for s in enabled_systems}
    for s in systems_all:
        if s["id"] not in enabled_ids:
            sys_stats.append({
                "id": s["id"], "name": s["name"], "enabled": False,
                "is_short": s.get("is_short", False),
                "n_trades": 0, "win_rate": 0.0, "avg_ret": 0.0,
                "contrib_pct": 0.0,
            })
    # Re-order sys_stats to match systems_all order (position)
    by_id = {x["id"]: x for x in sys_stats}
    sys_stats = [by_id[s["id"]] for s in systems_all if s["id"] in by_id]

    # Equity curves — primary universe (blue) and union (pink) — match the
    # corr explorer's two-line plot exactly.
    equity_primary  = _sec_equity_curve(universe_rows, outcome)
    equity_combined = _sec_equity_curve(union_rows_signed, outcome)

    # Yearly breakdown with both primary baseline + combined union, same
    # field names the corr explorer's yearly chart expects.
    yp: dict = defaultdict(list)
    yc: dict = defaultdict(list)
    for r in universe_rows:
        yr = int(str(r.get("trade_date", "0000"))[:4])
        v = r.get(outcome)
        if v is None:
            continue
        try:
            yp[yr].append(float(v))
        except (TypeError, ValueError):
            pass
    for r in union_rows_signed:
        yr = int(str(r.get("trade_date", "0000"))[:4])
        v = r.get(outcome)
        if v is None:
            continue
        try:
            yc[yr].append(float(v))
        except (TypeError, ValueError):
            pass
    yearly_out = []
    for yr in sorted(set(yp) | set(yc)):
        p = yp.get(yr, [])
        c = yc.get(yr, [])
        yearly_out.append({
            "year":         yr,
            "primary_n":    len(p),
            "primary_avg":  round(float(np.mean(p)), 6) if p else 0.0,
            "primary_wr":   round(float(np.mean([1.0 if v > 0 else 0.0 for v in p])), 4) if p else 0.0,
            "combined_n":   len(c),
            "combined_avg": round(float(np.mean(c)), 6) if c else 0.0,
            "combined_wr":  round(float(np.mean([1.0 if v > 0 else 0.0 for v in c])), 4) if c else 0.0,
        })
    horizon = _parse_horizon(outcome)

    # Ticker breakdown (use signed returns so short systems contribute positively)
    ticker_rets: dict = defaultdict(list)
    for r in union_rows_signed:
        v = r.get(outcome)
        if v is None:
            continue
        try:
            fv = float(v)
            ticker_rets[r.get("ticker", "?")].append(fv)
        except (TypeError, ValueError):
            continue
    total_pnl = sum(sum(v) for v in ticker_rets.values())
    tickers_out = []
    for tkr, rets in sorted(ticker_rets.items()):
        n_t = len(rets)
        avg_r = float(np.mean(rets)) if rets else 0.0
        wr = float(np.mean([1.0 if r > 0 else 0.0 for r in rets])) if rets else 0.0
        contrib = (sum(rets) / total_pnl * 100) if total_pnl != 0 else 0.0
        tickers_out.append({
            "ticker": tkr, "n": n_t,
            "avg_ret": round(avg_r, 6), "win_rate": round(wr, 4),
            "contrib_pct": round(contrib, 2),
        })

    # Winner / loser averages (signed)
    union_outcomes = [float(r[outcome]) for r in union_rows_signed
                      if r.get(outcome) is not None]
    winners = [v for v in union_outcomes if v > 0]
    losers  = [v for v in union_outcomes if v <= 0]
    winner_avg = round(float(np.mean(winners)), 6) if winners else 0.0
    loser_avg  = round(float(np.mean(losers)),  6) if losers  else 0.0

    # Utilisation matches the corr explorer's normalised-exclusivity formula
    # computed over PAIR firing counts (each "leg" of any system). Single-pair
    # case returns 100% by convention.
    n_each_pair = [int(v.sum()) for v in pair_vectors]
    union_n = int(V_port.sum())
    sum_n = sum(n_each_pair)
    min_n = min(n_each_pair) if n_each_pair else 0
    if sum_n > min_n:
        utilisation = (union_n - min_n) / (sum_n - min_n) * 100
    else:
        utilisation = 100.0
    utilisation = round(float(utilisation), 1)

    # Aggregate trade summary stats
    union_stats = _stats_from_trades(union_rows, outcome)

    # Enriched per-trade records (for CSV + activity panes). Each row in
    # the union of systems' trade sets gets entry/exit prices, exit_date,
    # ret, and a list of system names that fired for it.
    # (horizon is already computed earlier in the function — reuse it.)
    tickers_in_union = sorted({r.get("ticker", "") for r in union_rows
                                if r.get("ticker")})
    calendars_for_trades = await _fetch_ticker_calendars(oi_pool, tickers_in_union)
    combined_trades = []
    for r_idx, vp in enumerate(V_port):
        if vp != 1.0:
            continue
        fired = [enabled_systems[k]["name"]
                 for k, vS in enumerate(system_vectors)
                 if vS[r_idx] == 1.0]
        rec = _build_enriched_trade(
            rows[r_idx], calendars_for_trades, horizon,
            primary_metric=None,     # multi-system → no single primary
            outcome_col=outcome,
        )
        rec["fired_systems"] = fired
        sign = union_sign_map.get(r_idx, +1)
        rec["direction"] = "short" if sign == -1 else "long"
        if sign == -1 and rec.get("ret") is not None:
            rec["ret"] = -rec["ret"]
        combined_trades.append(rec)

    # phi correlations restricted to the trade-eligible universe so the
    # denominator matches the corr explorer's filtered subset.
    def _phi_restricted(vectors: list) -> tuple:
        if not vectors:
            return [], []
        if len(universe_idx) == 0:
            n = len(vectors)
            return ([[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)],
                    [[0 for _ in range(n)] for _ in range(n)])
        M = np.stack([v[universe_idx] for v in vectors])
        if M.shape[0] == 1:
            phi = [[1.0]]
            ov  = [[int(M[0].sum())]]
        else:
            phi_arr = np.nan_to_num(np.corrcoef(M), nan=0.0)
            phi = [[round(float(v), 4) for v in row] for row in phi_arr]
            ov_arr = (M @ M.T).astype(int)
            ov  = [[int(v) for v in row] for row in ov_arr]
        return phi, ov

    phi_sys,   overlap_sys   = _phi_restricted(system_vectors)
    phi_pairs, overlap_pairs = _phi_restricted(pair_vectors)

    # `cleared_any` is empty in in_sample mode (PortfolioVectorBuilder
    # returns an empty set there to keep wf_dropped=0 / wf_start=first
    # row) and populated in walk_forward / train_test mode.
    if spec.kind == "in_sample":
        wf_dropped = 0
        wf_start   = rows[0]["trade_date"] if rows else None
    else:
        wf_dropped = len(rows) - len(cleared_any)
        wf_start   = (min(rows[i]["trade_date"] for i in cleared_any)
                      if cleared_any else None)
    resp_mode = spec.kind

    # Fields named to mirror the corr explorer's /secondary-correlation
    # response — the frontend reuses corrStats() and the corr render
    # functions verbatim.
    return {
        "portfolio": portfolio,
        "systems":   sys_stats,
        "horizon":   horizon,
        "mode":              resp_mode,
        "warmup":            spec.warmup if spec.kind == "walk_forward" else None,
        "cutoff_date":       spec.cutoff.isoformat() if spec.kind == "train_test" else None,
        "dropped_warmup_n":  wf_dropped,
        "start_date":        wf_start,
        "baseline_n": int(universe_mask.sum()),
        "combined_n": int(union_n),
        "equity_primary":  equity_primary,
        "equity_combined": equity_combined,
        "yearly":          yearly_out,
        "tickers":         tickers_out,
        "combined_trade_dates": [r.get("trade_date", "") for r in union_rows],
        "combined_trades":      combined_trades,
        "winner_avg_ret":  winner_avg,
        "loser_avg_ret":   loser_avg,
        "n_each":          n_each_pair,
        "utilisation":     utilisation,
        # Heatmaps
        "phi_systems":       phi_sys,
        "overlap_systems":   overlap_sys,
        "system_labels":     system_labels,
        "phi_pairs":         phi_pairs,
        "overlap_pairs":     overlap_pairs,
        "pair_labels":       pair_labels,
        "system_boundaries": system_boundaries,
    }


# ── System Library (anchor-agnostic template store) ───────────────────────


def _library_row_to_dict(r) -> dict:
    d = dict(r)
    for k in ("created_at", "updated_at"):
        if d.get(k) is not None:
            d[k] = str(d[k])[:19]
    if isinstance(d.get("secondaries"), str):
        import json as _json
        d["secondaries"] = _json.loads(d["secondaries"])
    return d


@router.get("/library/systems")
async def list_library_systems(pool=Depends(get_pool)):
    await _ensure_tables(pool)
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """SELECT id, name, description, primary_metric, primary_bins,
                      primary_bin_count, secondaries, is_short, created_at, updated_at
               FROM oi_research_system_library
               ORDER BY updated_at DESC""")
    return [_library_row_to_dict(r) for r in rows]


@router.post("/library/systems")
async def add_library_system(body: LibraryItemIn, pool=Depends(get_pool)):
    await _ensure_tables(pool)
    import json as _json
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """INSERT INTO oi_research_system_library
                 (name, description, primary_metric, primary_bins,
                  primary_bin_count, secondaries, is_short)
               VALUES ($1, $2, $3, $4, $5, $6::jsonb, $7)
               RETURNING id, name, description, primary_metric, primary_bins,
                         primary_bin_count, secondaries, is_short, created_at, updated_at""",
            body.name, body.description, body.primary_metric, body.primary_bins,
            body.primary_bin_count,
            _json.dumps([s.dict() for s in body.secondaries]),
            body.is_short)
    return _library_row_to_dict(row)


@router.put("/library/systems/{lid}")
async def update_library_system(lid: int, body: LibraryItemUpdate,
                                pool=Depends(get_pool)):
    import json as _json
    sets, params, p = [], [], 1
    if body.name is not None:
        sets.append(f"name = ${p}"); params.append(body.name); p += 1
    if body.description is not None:
        sets.append(f"description = ${p}"); params.append(body.description); p += 1
    if body.primary_metric is not None:
        sets.append(f"primary_metric = ${p}"); params.append(body.primary_metric); p += 1
    if body.primary_bins is not None:
        sets.append(f"primary_bins = ${p}"); params.append(body.primary_bins); p += 1
    if body.primary_bin_count is not None:
        sets.append(f"primary_bin_count = ${p}"); params.append(body.primary_bin_count); p += 1
    if body.secondaries is not None:
        sets.append(f"secondaries = ${p}::jsonb")
        params.append(_json.dumps([s.dict() for s in body.secondaries])); p += 1
    if body.is_short is not None:
        sets.append(f"is_short = ${p}"); params.append(body.is_short); p += 1
    if not sets:
        raise HTTPException(400, "no fields to update")
    sets.append("updated_at = NOW()")
    params.append(lid)
    sql = (f"UPDATE oi_research_system_library SET {', '.join(sets)} "
           f"WHERE id = ${p} RETURNING id, name, description, primary_metric, "
           f"primary_bins, primary_bin_count, secondaries, is_short, created_at, updated_at")
    async with pool.acquire() as conn:
        row = await conn.fetchrow(sql, *params)
    if not row:
        raise HTTPException(404, "library system not found")
    return _library_row_to_dict(row)


@router.delete("/library/systems/{lid}")
async def delete_library_system(lid: int, pool=Depends(get_pool)):
    async with pool.acquire() as conn:
        await conn.execute(
            "DELETE FROM oi_research_system_library WHERE id = $1", lid)
    return {"ok": True}


@router.post("/portfolios/{pid}/systems/from-library/{lid}")
async def add_system_from_library(pid: int, lid: int, pool=Depends(get_pool)):
    """Copy a library system into the portfolio as a new system."""
    await _ensure_tables(pool)
    import json as _json
    async with pool.acquire() as conn:
        port = await conn.fetchrow(
            "SELECT id FROM oi_research_portfolios WHERE id = $1", pid)
        if not port:
            raise HTTPException(404, "portfolio not found")
        lib = await conn.fetchrow(
            """SELECT name, primary_metric, primary_bins, primary_bin_count, secondaries, is_short
               FROM oi_research_system_library WHERE id = $1""", lid)
        if not lib:
            raise HTTPException(404, "library system not found")
        cnt = await conn.fetchval(
            "SELECT COUNT(*) FROM oi_research_systems WHERE portfolio_id = $1", pid)
        # Build a default name. If the library name collides with an existing
        # system in this portfolio, suffix with "(copy)" or "(N)".
        base_name = lib["name"]
        exists = {r["name"] for r in await conn.fetch(
            "SELECT name FROM oi_research_systems WHERE portfolio_id = $1", pid)}
        name = base_name
        if name in exists:
            for k in range(2, 100):
                cand = f"{base_name} ({k})"
                if cand not in exists:
                    name = cand
                    break
        secs = lib["secondaries"]
        if isinstance(secs, str):
            secs = _json.loads(secs)
        row = await conn.fetchrow(
            """INSERT INTO oi_research_systems
                 (portfolio_id, name, position, primary_metric, primary_bins,
                  primary_bin_count, secondaries, is_short)
               VALUES ($1, $2, $3, $4, $5, $6, $7::jsonb, $8)
               RETURNING id, portfolio_id, name, enabled, position,
                         primary_metric, primary_bins, primary_bin_count,
                         secondaries, is_short, created_at, updated_at""",
            pid, name, cnt, lib["primary_metric"], lib["primary_bins"],
            lib["primary_bin_count"], _json.dumps(secs), lib.get("is_short", False))
        await conn.execute(
            "UPDATE oi_research_portfolios SET updated_at = NOW() WHERE id = $1",
            pid)
    return _system_row_to_dict(row)
