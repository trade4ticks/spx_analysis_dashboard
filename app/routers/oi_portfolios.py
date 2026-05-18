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
    _bin_membership,
    _sec_equity_curve,
    _parse_horizon,
)

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
    name: Optional[str] = None
    description: Optional[str] = None


class SecondarySpec(BaseModel):
    metric: str
    bins: List[int]
    bin_count: int = 10


class SystemIn(BaseModel):
    name: Optional[str] = None
    primary_metric: str
    primary_bins: List[int]
    primary_bin_count: int = 20
    secondaries: List[SecondarySpec] = []


class SystemUpdate(BaseModel):
    name: Optional[str] = None
    enabled: Optional[bool] = None
    position: Optional[int] = None
    primary_metric: Optional[str] = None
    primary_bins: Optional[List[int]] = None
    primary_bin_count: Optional[int] = None
    secondaries: Optional[List[SecondarySpec]] = None


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
                      p.date_from, p.date_to, p.created_at, p.updated_at,
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
                         date_from, date_to, created_at, updated_at""",
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
    if not sets:
        raise HTTPException(400, "no fields to update")
    sets.append("updated_at = NOW()")
    params.append(pid)
    sql = (f"UPDATE oi_research_portfolios SET {', '.join(sets)} "
           f"WHERE id = ${p} RETURNING id, name, description, ticker, outcome, "
           f"date_from, date_to, created_at, updated_at")
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
                      date_from, date_to, created_at, updated_at
               FROM oi_research_portfolios WHERE id = $1""", pid)
        if not prow:
            raise HTTPException(404, "portfolio not found")
        srows = await conn.fetch(
            """SELECT id, portfolio_id, name, enabled, position,
                      primary_metric, primary_bins, primary_bin_count, secondaries,
                      created_at, updated_at
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
                  primary_bin_count, secondaries)
               VALUES ($1, $2, $3, $4, $5, $6, $7::jsonb)
               RETURNING id, portfolio_id, name, enabled, position,
                         primary_metric, primary_bins, primary_bin_count,
                         secondaries, created_at, updated_at""",
            pid, name, cnt, body.primary_metric, body.primary_bins,
            body.primary_bin_count, secondaries_json)
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
    if not sets:
        raise HTTPException(400, "no fields to update")
    sets.append("updated_at = NOW()")
    params.extend([pid, sid])
    sql = (f"UPDATE oi_research_systems SET {', '.join(sets)} "
           f"WHERE portfolio_id = ${p} AND id = ${p + 1} "
           f"RETURNING id, portfolio_id, name, enabled, position, "
           f"primary_metric, primary_bins, primary_bin_count, secondaries, "
           f"created_at, updated_at")
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


@router.post("/portfolios/{pid}/aggregate")
async def portfolio_aggregate(pid: int,
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
                      primary_metric, primary_bins, primary_bin_count, secondaries
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
            "n_trades": 0, "win_rate": 0.0, "avg_ret": 0.0,
            "equity": [], "yearly": [], "activity": [], "tickers": [],
            "winner_avg_ret": 0.0, "loser_avg_ret": 0.0, "utilisation": 0.0,
            "phi_systems": [], "overlap_systems": [], "system_labels": [],
            "phi_pairs": [], "overlap_pairs": [], "pair_labels": [],
            "system_boundaries": [],
            "horizon": _parse_horizon(portfolio["outcome"]),
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

    for s in enabled_systems:
        prim_bins = set(int(b) for b in (s.get("primary_bins") or []))
        prim_count = int(s.get("primary_bin_count") or 20)
        V_p = _bin_membership(rows, s["primary_metric"], prim_bins,
                              prim_count, is_all)
        secs = s.get("secondaries") or []
        if not secs or not prim_bins:
            # Degenerate system — no secondaries OR no primary bins.
            # Skip but keep slot so labels stay aligned with systems list.
            zero = np.zeros(len(rows))
            system_vectors.append(zero)
            system_labels.append(s["name"])
            per_system_pair_n.append(0)
            system_boundaries.append(len(pair_vectors))
            continue

        sec_vecs = []
        for sec in secs:
            sec_bins = set(int(b) for b in (sec.get("bins") or []))
            sec_count = int(sec.get("bin_count") or 10)
            V_si = _bin_membership(rows, sec["metric"], sec_bins,
                                   sec_count, is_all)
            sec_vecs.append(V_si)
            # Pair vector = primary AND this single secondary
            V_pair = np.minimum(V_p, V_si)
            pair_vectors.append(V_pair)
            pair_labels.append(f"{s['name']}: {sec['metric']}")
        V_sec_union = np.max(np.stack(sec_vecs), axis=0) if sec_vecs else np.zeros(len(rows))
        V_S = np.minimum(V_p, V_sec_union)
        system_vectors.append(V_S)
        system_labels.append(s["name"])
        per_system_pair_n.append(len(sec_vecs))
        system_boundaries.append(len(pair_vectors))

    # Portfolio union = OR across enabled systems
    M_sys = np.stack(system_vectors) if system_vectors else np.zeros((1, len(rows)))
    V_port = (M_sys.sum(axis=0) > 0).astype(float)
    union_rows = [rows[i] for i, v in enumerate(V_port) if v == 1.0]

    # Per-system summary
    sys_stats = []
    total_sum_ret_union = 0.0
    for r in union_rows:
        v = r.get(outcome)
        if v is not None:
            try:
                total_sum_ret_union += float(v)
            except (TypeError, ValueError):
                pass
    for s, vec in zip(enabled_systems, system_vectors):
        sys_rows = [rows[i] for i, v in enumerate(vec) if v == 1.0]
        st = _stats_from_trades(sys_rows, outcome)
        sys_stats.append({
            "id": s["id"], "name": s["name"], "enabled": True,
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
                "n_trades": 0, "win_rate": 0.0, "avg_ret": 0.0,
                "contrib_pct": 0.0,
            })
    # Re-order sys_stats to match systems_all order (position)
    by_id = {x["id"]: x for x in sys_stats}
    sys_stats = [by_id[s["id"]] for s in systems_all if s["id"] in by_id]

    # Equity curve
    equity = _sec_equity_curve(union_rows, outcome)

    # Yearly breakdown
    yearly_buckets: dict = defaultdict(list)
    for r in union_rows:
        td = r.get("trade_date", "0000")
        yr = int(str(td)[:4])
        v = r.get(outcome)
        if v is not None:
            try:
                yearly_buckets[yr].append(float(v))
            except (TypeError, ValueError):
                pass
    yearly = []
    for yr in sorted(yearly_buckets):
        rets = yearly_buckets[yr]
        a = np.array(rets) if rets else np.array([0.0])
        yearly.append({
            "year": yr, "n": len(rets),
            "avg_ret":  round(float(a.mean()), 6) if rets else 0.0,
            "win_rate": round(float((a > 0).mean()), 4) if rets else 0.0,
        })

    # Trade activity (open positions per day, horizon-based)
    horizon = _parse_horizon(outcome)
    activity_by_date: dict = defaultdict(int)
    # Each union trade contributes +1 from entry_date through entry_date+horizon-1 (calendar days proxy)
    # We don't have a trading-day calendar here; the existing corr explorer uses entry-date density.
    # Match its pattern: count entries per date, then forward-fill horizon days via the row order.
    entry_dates = [r.get("trade_date", "") for r in union_rows]
    # Build a sorted unique list of dates from anchor rows for the spine
    all_dates_sorted = sorted({r.get("trade_date", "") for r in rows})
    date_idx = {d: i for i, d in enumerate(all_dates_sorted)}
    open_count = [0] * len(all_dates_sorted)
    for ed in entry_dates:
        i = date_idx.get(ed)
        if i is None:
            continue
        end = min(i + horizon, len(open_count))
        for j in range(i, end):
            open_count[j] += 1
    activity = [{"date": all_dates_sorted[i], "n_open": open_count[i]}
                for i in range(len(all_dates_sorted))]

    # Ticker breakdown
    ticker_rets: dict = defaultdict(list)
    for r in union_rows:
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

    # Winner / loser averages
    union_outcomes = [float(r[outcome]) for r in union_rows
                      if r.get(outcome) is not None]
    winners = [v for v in union_outcomes if v > 0]
    losers  = [v for v in union_outcomes if v <= 0]
    winner_avg = round(float(np.mean(winners)), 6) if winners else 0.0
    loser_avg  = round(float(np.mean(losers)),  6) if losers  else 0.0

    # Utilisation: normalised exclusivity across enabled systems
    n_each_sys = [int(v.sum()) for v in system_vectors]
    union_n = int(V_port.sum())
    sum_n = sum(n_each_sys)
    min_n = min(n_each_sys) if n_each_sys else 0
    if sum_n > min_n:
        utilisation = (union_n - min_n) / (sum_n - min_n) * 100
    else:
        utilisation = 0.0
    utilisation = round(float(utilisation), 1)

    # Aggregate trade summary stats
    union_stats = _stats_from_trades(union_rows, outcome)

    # System × System heatmap
    if len(system_vectors) >= 1:
        Msys = np.stack(system_vectors)
        if len(system_vectors) == 1:
            phi_sys = [[1.0]]
            overlap_sys = [[int(Msys[0].sum())]]
        else:
            phi_sys = np.nan_to_num(np.corrcoef(Msys), nan=0.0)
            phi_sys = [[round(float(v), 4) for v in row] for row in phi_sys]
            overlap_sys = (Msys @ Msys.T).astype(int)
            overlap_sys = [[int(v) for v in row] for row in overlap_sys]
    else:
        phi_sys = []
        overlap_sys = []

    # Pair × Pair heatmap
    if len(pair_vectors) >= 1:
        Mpairs = np.stack(pair_vectors)
        if len(pair_vectors) == 1:
            phi_pairs = [[1.0]]
            overlap_pairs = [[int(Mpairs[0].sum())]]
        else:
            phi_pairs = np.nan_to_num(np.corrcoef(Mpairs), nan=0.0)
            phi_pairs = [[round(float(v), 4) for v in row] for row in phi_pairs]
            overlap_pairs = (Mpairs @ Mpairs.T).astype(int)
            overlap_pairs = [[int(v) for v in row] for row in overlap_pairs]
    else:
        phi_pairs = []
        overlap_pairs = []

    return {
        "portfolio": portfolio,
        "systems":   sys_stats,
        "n_trades":  union_stats["n"],
        "win_rate":  union_stats["win_rate"],
        "avg_ret":   union_stats["avg_ret"],
        "horizon":   horizon,
        "equity":    equity,
        "yearly":    yearly,
        "activity":  activity,
        "tickers":   tickers_out,
        "winner_avg_ret": winner_avg,
        "loser_avg_ret":  loser_avg,
        "utilisation":    utilisation,
        "phi_systems":      phi_sys,
        "overlap_systems":  overlap_sys,
        "system_labels":    system_labels,
        "phi_pairs":        phi_pairs,
        "overlap_pairs":    overlap_pairs,
        "pair_labels":      pair_labels,
        "system_boundaries": system_boundaries,
        "union_trade_dates": [r.get("trade_date", "") for r in union_rows],
    }
