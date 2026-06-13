"""Portfolio Builder — third analysis tier on the OI Analysis page.

A Portfolio is a saved research artifact. It contains one or more
Signals (heatmap cell-set definitions from the `signals` table in the
OI DB). The portfolio's trades are the deduped union of every enabled
signal's trades: a (ticker, trade_date) pair that fires in multiple
signals is counted ONCE in the portfolio stats and equity curve.

Portfolio metadata (CRUD) lives in the main app DB (via get_pool).
Signal definitions live in the OI DB (via get_oi_pool).
The aggregate endpoint queries is_bins + daily_features (via get_oi_pool).
"""
from collections import defaultdict
from datetime import date as _date, timedelta
from typing import List, Optional
import json
import math

try:
    from scipy import stats as _sp_stats
except ImportError:
    _sp_stats = None

import numpy as np
from fastapi import APIRouter, Body, Depends, HTTPException
from pydantic import BaseModel

from app.db import get_pool, get_oi_pool
from app.routers.oi_analysis import (
    _sec_equity_curve,
    _parse_horizon,
    _fetch_ticker_calendars,
    _build_enriched_trade,
    _outcome_value,
    _outcome_select_cols,
)

router = APIRouter(tags=["oi_portfolios"])


_DDL = """
CREATE TABLE IF NOT EXISTS oi_research_portfolios (
    id          SERIAL PRIMARY KEY,
    name        TEXT NOT NULL,
    description TEXT,
    ticker      TEXT,
    outcome     TEXT,
    date_from   TEXT,
    date_to     TEXT,
    created_at  TIMESTAMPTZ DEFAULT NOW(),
    updated_at  TIMESTAMPTZ DEFAULT NOW()
);
ALTER TABLE oi_research_portfolios
    ADD COLUMN IF NOT EXISTS monitored BOOLEAN DEFAULT FALSE;
-- Make ticker/outcome nullable for the signal-derived outcome model
-- (idempotent: DROP NOT NULL on a nullable column is a no-op in PG).
ALTER TABLE oi_research_portfolios ALTER COLUMN ticker  DROP NOT NULL;
ALTER TABLE oi_research_portfolios ALTER COLUMN outcome DROP NOT NULL;

-- portfolio_signals: join table linking portfolios → cell-set signals.
-- signal_id is a plain INTEGER (no FK constraint — signals lives in the
-- OI DB, portfolios in the app DB; cross-DB FKs are not supported).
-- The application layer enforces referential integrity on add.
CREATE TABLE IF NOT EXISTS portfolio_signals (
    id           SERIAL PRIMARY KEY,
    portfolio_id INTEGER NOT NULL REFERENCES oi_research_portfolios(id) ON DELETE CASCADE,
    signal_id    INTEGER NOT NULL,
    position     INTEGER DEFAULT 0,
    enabled      BOOLEAN DEFAULT TRUE,
    created_at   TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(portfolio_id, signal_id)
);
"""


async def _ensure_tables(pool):
    async with pool.acquire() as conn:
        await conn.execute(_DDL)


# ── Pydantic models ─────────────────────────────────────────────────────────


class PortfolioIn(BaseModel):
    name: str
    description: Optional[str] = None
    ticker: Optional[str] = "ALL"     # retained for optional trade filtering
    outcome: Optional[str] = None     # NULL until first signal is added
    date_from: Optional[str] = None
    date_to: Optional[str] = None


class PortfolioUpdate(BaseModel):
    name:        Optional[str]  = None
    description: Optional[str]  = None
    monitored:   Optional[bool] = None


class SignalLinkIn(BaseModel):
    signal_id: int


class SignalLinkUpdate(BaseModel):
    enabled:  Optional[bool] = None
    position: Optional[int]  = None


class AggregateRequest(BaseModel):
    # Outlier filter: exclude trades whose outcome > this threshold
    # (one-sided high cutoff; losses always kept). Applied in the
    # per-signal trade-row assembly so both the per-signal
    # contribution stats AND the deduped union_rows reconcile.
    outlier_max_ret: Optional[float] = None


class LossAnalysisRequest(BaseModel):
    """Request body for POST /portfolios/{pid}/loss-analysis.

    date_from / date_to override (or narrow) the portfolio's own date range.
    per_trade / daily_cap mirror _computeDollarSeries JS defaults.
    """
    date_from:       Optional[str]   = None
    date_to:         Optional[str]   = None
    per_trade:       float           = 2000.0
    daily_cap:       float           = 10000.0
    outlier_max_ret: Optional[float] = None


# ── Helpers ──────────────────────────────────────────────────────────────────


def _portfolio_row_to_dict(r) -> dict:
    d = dict(r)
    for k in ("created_at", "updated_at"):
        if d.get(k) is not None:
            d[k] = str(d[k])[:19]
    return d


def _stats_from_trades(trades: list, outcome_col: str) -> dict:
    """Aggregate stats (n, win_rate, avg_ret, sum_ret) from a list of trade dicts."""
    rets = []
    for r in trades:
        v = r.get(outcome_col)
        if v is None:
            continue
        try:
            fv = float(v)
            if not math.isnan(fv):
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


def _empty_aggregate(portfolio: dict, outcome: str, all_ps: list,
                     outlier_excluded: int = 0) -> dict:
    """Return a zero-state aggregate response when there are no enabled signals."""
    signals_out = [
        {"id": ps["ps_id"], "signal_id": ps["signal_id"],
         "name": f"Signal {ps['signal_id']}", "enabled": ps["enabled"],
         "n_trades": 0, "win_rate": 0.0, "avg_ret": 0.0, "contrib_pct": 0.0}
        for ps in all_ps
    ]
    return {
        "portfolio": portfolio, "signals": signals_out,
        "horizon": _parse_horizon(outcome),
        "n": 0, "avg_ret": None, "win_rate": None, "std": None,
        "p5": None, "p95": None, "median": None, "n_tickers": 0,
        "n_winners": 0, "avg_winners": None, "avg_losers": None,
        "trades_per_year": None,
        "combined_n": 0,
        "equity_primary": [], "equity_combined": [],
        "yearly": [], "tickers": [],
        "combined_trades": [], "combined_trade_dates": [],
        "winner_avg_ret": 0.0, "loser_avg_ret": 0.0,
        "n_each": [], "utilisation": 100.0,
        "phi_systems": [], "overlap_systems": [], "system_labels": [],
        "phi_pairs": [], "overlap_pairs": [], "pair_labels": [],
        "system_boundaries": [],
        "outlier_excluded": outlier_excluded,
        "trading_days": [],
    }


# ── Loss-correlation helpers ─────────────────────────────────────────────────


def _iso_week_key(d: _date) -> str:
    """Return 'YYYY-Www' ISO week key for date d."""
    iso = d.isocalendar()
    return f"{iso[0]:04d}-W{iso[1]:02d}"


def _week_monday(wk_key: str) -> str:
    """Return ISO-8601 date string for the Monday of 'YYYY-Www'."""
    yr, wk = int(wk_key[:4]), int(wk_key[6:])
    return _date.fromisocalendar(yr, wk, 1).isoformat()


def _all_iso_weeks(d_min: _date, d_max: _date) -> list:
    """Sorted list of all ISO week keys from d_min's week through d_max's week."""
    weeks = []
    cur = d_min - timedelta(days=d_min.weekday())   # Monday of d_min's week
    end = d_max - timedelta(days=d_max.weekday())   # Monday of d_max's week
    while cur <= end:
        weeks.append(_iso_week_key(cur))
        cur += timedelta(weeks=1)
    return weeks


def _dollar_pnl_by_week(
    trade_rows: list,
    outcome: str,
    per_trade: float,
    daily_cap: float,
    all_weeks: list,
) -> dict:
    """Compute weekly dollar P&L with dollar-capped position sizing.

    Sizing mirrors _computeDollarSeries in oi_analysis.js:
      per_ticker_alloc = min(per_trade, daily_cap / N_day)
      shares = max(1, floor(alloc / spot_co))
      dollar_pnl = shares × spot_co × ret

    Returns {week_key: total_dollar_pnl} zero-filled for all_weeks.
    """
    # Group by trade_date to compute N per day (distinct tickers per signal day).
    by_date: dict = defaultdict(list)
    for r in trade_rows:
        by_date[r["trade_date"]].append(r)

    week_sums: dict = {w: 0.0 for w in all_weeks}
    for td, rows in by_date.items():
        n_day = len(rows)
        alloc = min(per_trade, daily_cap / n_day) if n_day > 0 else per_trade
        wk = _iso_week_key(_date.fromisoformat(td))
        for r in rows:
            spot = float(r.get("spot_co") or 0)
            ret  = float(r.get(outcome)   or 0)
            shares = max(1, int(alloc / spot)) if spot > 0 else 1
            dollar_pnl = shares * spot * ret
            if wk in week_sums:
                week_sums[wk] += dollar_pnl
    return week_sums


# ── Portfolio CRUD ───────────────────────────────────────────────────────────


@router.get("/portfolios")
async def list_portfolios(pool=Depends(get_pool)):
    await _ensure_tables(pool)
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """SELECT p.id, p.name, p.description, p.ticker, p.outcome,
                      p.date_from, p.date_to, p.monitored,
                      p.created_at, p.updated_at,
                      (SELECT COUNT(*) FROM portfolio_signals ps
                        WHERE ps.portfolio_id = p.id) AS signal_count
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
async def get_portfolio(pid: int,
                        pool=Depends(get_pool),
                        oi_pool=Depends(get_oi_pool)):
    """Return portfolio metadata + attached signal definitions."""
    await _ensure_tables(pool)
    async with pool.acquire() as conn:
        prow = await conn.fetchrow(
            """SELECT id, name, description, ticker, outcome,
                      date_from, date_to, monitored, created_at, updated_at
               FROM oi_research_portfolios WHERE id = $1""", pid)
        if not prow:
            raise HTTPException(404, "portfolio not found")
        ps_rows = await conn.fetch(
            """SELECT id AS ps_id, portfolio_id, signal_id, position, enabled, created_at
               FROM portfolio_signals
               WHERE portfolio_id = $1
               ORDER BY position, id""", pid)

    # Fetch signal definitions from OI DB so the UI can display them
    signals_out = []
    if ps_rows and oi_pool:
        all_sids = [r["signal_id"] for r in ps_rows]
        async with oi_pool.acquire() as conn:
            sig_rows = await conn.fetch(
                """SELECT id, name, primary_metric, secondary_metric,
                          outcome, n_bins, cell_set
                   FROM signals WHERE id = ANY($1)""", all_sids)
        sig_by_id = {r["id"]: dict(r) for r in sig_rows}
        for ps in ps_rows:
            sig = sig_by_id.get(ps["signal_id"], {})
            signals_out.append({
                "id":               ps["ps_id"],
                "portfolio_id":     ps["portfolio_id"],
                "signal_id":        ps["signal_id"],
                "position":         ps["position"],
                "enabled":          ps["enabled"],
                "created_at":       str(ps["created_at"])[:19],
                "name":             sig.get("name", f"Signal {ps['signal_id']}"),
                "primary_metric":   sig.get("primary_metric"),
                "secondary_metric": sig.get("secondary_metric"),
                "outcome":          sig.get("outcome"),
                "n_bins":           sig.get("n_bins"),
                "n_cells":          len(sig.get("cell_set") or []),
            })

    return {
        "portfolio": _portfolio_row_to_dict(prow),
        "signals":   signals_out,
    }


# ── Portfolio Signals CRUD ───────────────────────────────────────────────────


@router.post("/portfolios/{pid}/signals")
async def add_portfolio_signal(pid: int, body: SignalLinkIn,
                               pool=Depends(get_pool),
                               oi_pool=Depends(get_oi_pool)):
    """Attach a saved signal to this portfolio.

    Rejects the signal if its outcome does not match the portfolio's
    outcome — this is what makes the Phase-3 'first-wins' dedup correct
    (same (ticker, date) row has the same outcome value regardless of
    which signal fires it).
    """
    await _ensure_tables(pool)
    if not oi_pool:
        raise HTTPException(503, "OI database not configured")

    # Load portfolio (main DB)
    async with pool.acquire() as conn:
        port = await conn.fetchrow(
            "SELECT id, outcome FROM oi_research_portfolios WHERE id = $1", pid)
    if not port:
        raise HTTPException(404, "portfolio not found")

    # Verify signal exists (OI DB)
    async with oi_pool.acquire() as conn:
        sig = await conn.fetchrow(
            "SELECT id, name, outcome FROM signals WHERE id = $1", body.signal_id)
    if not sig:
        raise HTTPException(404, f"Signal {body.signal_id} not found")

    # Outcome handling:
    # - First signal (portfolio.outcome is NULL) → auto-set portfolio outcome
    # - Subsequent signals → must match existing portfolio outcome
    if port["outcome"] is None:
        async with pool.acquire() as conn:
            await conn.execute(
                "UPDATE oi_research_portfolios SET outcome=$1, updated_at=NOW() WHERE id=$2",
                sig["outcome"], pid)
    elif sig["outcome"] != port["outcome"]:
        raise HTTPException(400,
            f"Signal outcome '{sig['outcome']}' does not match "
            f"portfolio outcome '{port['outcome']}'. "
            "All signals in a portfolio must share the same outcome.")

    # Insert link (UPSERT — silently no-ops if already present)
    async with pool.acquire() as conn:
        cnt = await conn.fetchval(
            "SELECT COUNT(*) FROM portfolio_signals WHERE portfolio_id = $1", pid)
        row = await conn.fetchrow(
            """INSERT INTO portfolio_signals (portfolio_id, signal_id, position)
               VALUES ($1, $2, $3)
               ON CONFLICT (portfolio_id, signal_id) DO NOTHING
               RETURNING id AS ps_id, portfolio_id, signal_id, position, enabled""",
            pid, body.signal_id, int(cnt))
        await conn.execute(
            "UPDATE oi_research_portfolios SET updated_at=NOW() WHERE id=$1", pid)

    if not row:
        return {"ok": True, "message": "signal already attached to this portfolio"}
    d = dict(row)
    d["name"] = sig["name"]
    return d


@router.delete("/portfolios/{pid}/signals/{ps_id}")
async def remove_portfolio_signal(pid: int, ps_id: int, pool=Depends(get_pool)):
    async with pool.acquire() as conn:
        await conn.execute(
            "DELETE FROM portfolio_signals WHERE id=$1 AND portfolio_id=$2",
            ps_id, pid)
        await conn.execute(
            "UPDATE oi_research_portfolios SET updated_at=NOW() WHERE id=$1", pid)
    return {"ok": True}


@router.put("/portfolios/{pid}/signals/{ps_id}")
async def update_portfolio_signal(pid: int, ps_id: int, body: SignalLinkUpdate,
                                  pool=Depends(get_pool)):
    sets, params, p = [], [], 1
    if body.enabled is not None:
        sets.append(f"enabled = ${p}"); params.append(body.enabled); p += 1
    if body.position is not None:
        sets.append(f"position = ${p}"); params.append(body.position); p += 1
    if not sets:
        raise HTTPException(400, "nothing to update")
    params.extend([ps_id, pid])
    sql = (f"UPDATE portfolio_signals SET {', '.join(sets)} "
           f"WHERE id = ${p} AND portfolio_id = ${p+1} "
           f"RETURNING id AS ps_id, portfolio_id, signal_id, position, enabled")
    async with pool.acquire() as conn:
        row = await conn.fetchrow(sql, *params)
        if not row:
            raise HTTPException(404, "portfolio signal not found")
        await conn.execute(
            "UPDATE oi_research_portfolios SET updated_at=NOW() WHERE id=$1", pid)
    return dict(row)


# ── Aggregate ────────────────────────────────────────────────────────────────

_SAFE_METRIC = set("abcdefghijklmnopqrstuvwxyz_0123456789")


@router.post("/portfolios/{pid}/aggregate")
async def portfolio_aggregate(
    pid:      int,
    req:      AggregateRequest = Body(default_factory=AggregateRequest),
    pool      = Depends(get_pool),
    oi_pool   = Depends(get_oi_pool),
):
    """Compute deduped union stats + equity + signal correlation heatmap.

    Algorithm
    ---------
    Phase 1  Load portfolio + portfolio_signals (main DB) and signal
             definitions (OI DB).
    Phase 2  Per-signal IS trade query: same SQL as /secondary-zone-analyze,
             filtering daily_features via is_bins cell-set membership.
    Phase 3  Union dedup via seen-dict: a (ticker, trade_date) satisfying
             multiple signals is counted ONCE. Structurally identical to the
             old OR-of-binary-vectors (V_port = M_sys.sum > 0).
    Phase 4  Equity via _sec_equity_curve (cumulative SUM — same function,
             same call, only the input list differs from the zone endpoint).
    Phase 5  Per-signal contribution stats (from each signal's OWN rows,
             before dedup — so per-signal n's can exceed the portfolio total).
    Phase 6  Signal-vs-signal phi correlation matrix over union_rows universe.
    """
    if not oi_pool:
        return {"error": "OI database not configured"}
    await _ensure_tables(pool)

    # ── Phase 1a: load portfolio + portfolio_signals (main DB) ──────────────
    async with pool.acquire() as conn:
        prow = await conn.fetchrow(
            """SELECT id, name, description, ticker, outcome, date_from, date_to
               FROM oi_research_portfolios WHERE id = $1""", pid)
        if not prow:
            raise HTTPException(404, "portfolio not found")
        ps_rows = await conn.fetch(
            """SELECT id AS ps_id, signal_id, position, enabled
               FROM portfolio_signals
               WHERE portfolio_id = $1
               ORDER BY position, id""", pid)

    portfolio  = _portfolio_row_to_dict(prow)
    outcome    = portfolio["outcome"]
    ticker     = portfolio["ticker"]
    all_ps     = [dict(r) for r in ps_rows]
    enabled_ps = [p for p in all_ps if p["enabled"]]

    if not enabled_ps:
        return _empty_aggregate(portfolio, outcome, all_ps)

    # ── Phase 1b: load signal definitions from OI DB ────────────────────────
    all_sids = list({ps["signal_id"] for ps in all_ps})
    async with oi_pool.acquire() as conn:
        sig_rows = await conn.fetch(
            """SELECT id, name, primary_metric, secondary_metric,
                      outcome, n_bins, cell_set
               FROM signals WHERE id = ANY($1)""", all_sids)
    sig_by_id = {r["id"]: dict(r) for r in sig_rows}

    # Build ordered (ps, sig) pairs for enabled signals only,
    # skipping any whose outcome doesn't match the portfolio outcome.
    enabled_signals = []
    for ps in enabled_ps:
        sig = sig_by_id.get(ps["signal_id"])
        if not sig:
            continue   # signal deleted from OI DB
        if sig["outcome"] != outcome:
            continue   # outcome mismatch — should not happen if add endpoint is used
        enabled_signals.append({"ps": ps, "sig": sig})

    if not enabled_signals:
        return _empty_aggregate(portfolio, outcome, all_ps)

    # Safety: validate metric names (same check as zone-analyze)
    for entry in enabled_signals:
        sig = entry["sig"]
        for m in (sig["primary_metric"], sig["secondary_metric"]):
            if not all(c in _SAFE_METRIC for c in (m or "")):
                return {"error": f"Unsafe metric name in signal '{sig['name']}': {m}"}

    # ── Phase 2: per-signal IS trade queries via is_bins (OI DB) ────────────
    # Build shared date / ticker filter clauses (same across all signals).
    # Signal-specific params are: $1=n_bins, $2=xs, $3=ys.
    # Date/ticker params start at $4 and are appended here.
    date_params: list = []
    date_sql    = ""
    p_idx       = 4
    if portfolio.get("date_from"):
        date_sql += f" AND df.trade_date >= ${p_idx}"
        date_params.append(_date.fromisoformat(portfolio["date_from"])); p_idx += 1
    if portfolio.get("date_to"):
        date_sql += f" AND df.trade_date <= ${p_idx}"
        date_params.append(_date.fromisoformat(portfolio["date_to"])); p_idx += 1
    ticker_sql = ""
    if ticker != "ALL":
        ticker_sql = f" AND df.ticker = ${p_idx}"
        date_params.append(ticker); p_idx += 1

    outcome_cols = _outcome_select_cols(outcome)
    out_sel = ", ".join(f"df.{c}" for c in outcome_cols)
    out_nn  = " AND ".join(f"df.{c} IS NOT NULL" for c in outcome_cols)

    per_signal_rows: list = []   # list[list[dict]]
    # Outlier exclusion telemetry — track unique (ticker, trade_date)
    # rows dropped by the filter so the tagline reports the de-duped
    # count that matches the union-level `n`. A row excluded by one
    # signal is excluded by every signal (same outcome value) so the
    # set naturally dedupes across signals.
    outlier_excluded_keys: set = set()
    async with oi_pool.acquire() as conn:
        for entry in enabled_signals:
            sig    = entry["sig"]
            n_bins = int(sig["n_bins"])
            raw_cs = sig["cell_set"] or "[]"
            cells  = json.loads(raw_cs) if isinstance(raw_cs, str) else raw_cs
            xs = [int(c[0]) for c in cells]
            ys = [int(c[1]) for c in cells]
            prim   = sig["primary_metric"]
            sec    = sig["secondary_metric"]

            query = f"""
                SELECT df.ticker, df.trade_date::text AS trade_date,
                       df.spot_co, {out_sel}
                FROM daily_features df
                JOIN is_bins ib USING (ticker, trade_date)
                WHERE ib.bin20_{prim} > 0
                  AND ib.bin20_{sec}  > 0
                  AND {out_nn}
                  AND (
                    ((ib.bin20_{prim} - 1) * $1::int) / 20,
                    ((ib.bin20_{sec}  - 1) * $1::int) / 20
                  ) IN (SELECT * FROM unnest($2::int[], $3::int[]))
                  {date_sql}{ticker_sql}
                ORDER BY df.trade_date, df.ticker
            """
            try:
                raw = await conn.fetch(query, n_bins, xs, ys, *date_params)
            except Exception as exc:
                per_signal_rows.append([])
                continue

            sig_trade_rows = []
            for r in raw:
                ov = _outcome_value(r, outcome)
                if ov is None:
                    continue
                try:
                    fov = float(ov)
                    if math.isnan(fov):
                        continue
                except (TypeError, ValueError):
                    continue
                # OUTLIER FILTER — applied at the per-signal trade-row
                # assembly so both per-signal contribution stats AND
                # the deduped union_rows reconcile. One-sided high.
                if req.outlier_max_ret is not None and fov > req.outlier_max_ret:
                    outlier_excluded_keys.add((r["ticker"], str(r["trade_date"])))
                    continue
                sig_trade_rows.append({
                    "ticker":     r["ticker"],
                    "trade_date": str(r["trade_date"]),
                    "spot_co":    r.get("spot_co"),
                    outcome:      fov,
                })
            per_signal_rows.append(sig_trade_rows)

    # ── Phase 3: Union dedup ─────────────────────────────────────────────────
    # A (ticker, trade_date) satisfying multiple signals is counted ONCE.
    # Identical semantics to the old OR-of-binary-vectors:
    #   V_port = (M_sys.sum(axis=0) > 0).astype(float)
    # "first wins" is exact because all signals share the same outcome column
    # — the same (ticker, date) row has the same outcome value regardless of
    # which signal fires it first.
    seen: dict = {}
    for sig_trade_rows in per_signal_rows:
        for row in sig_trade_rows:
            key = (row["ticker"], row["trade_date"])
            if key not in seen:
                seen[key] = row
    union_rows = sorted(seen.values(), key=lambda r: (r["trade_date"], r["ticker"]))

    n = len(union_rows)
    if n == 0:
        return _empty_aggregate(portfolio, outcome, all_ps,
                                outlier_excluded=len(outlier_excluded_keys))

    # ── Stats ────────────────────────────────────────────────────────────────
    arr    = np.array([r[outcome] for r in union_rows], dtype=np.float64)
    wins   = arr[arr > 0]
    losses = arr[arr < 0]
    avg_ret     = float(np.mean(arr))
    win_rate    = float(len(wins) / n)
    std         = float(np.std(arr))
    p5          = float(np.percentile(arr, 5))
    p95         = float(np.percentile(arr, 95))
    median      = float(np.median(arr))
    n_winners   = int(len(wins))
    avg_winners = float(np.mean(wins))   if n_winners    else 0.0
    avg_losers  = float(np.mean(losses)) if len(losses)  else 0.0
    n_tickers   = len({r["ticker"] for r in union_rows})

    # ── Trading-day calendar ride-along ────────────────────────────────────────
    # Dense NYSE-style trading-day list spanning the union trade set's
    # date range. Drives the Trade Activity pane's x-axis on initial
    # page load (no primary Analyze required) so its rolling H-day
    # window indexes actual trading days, not fired-trade dates.
    # Range matches the equity curve (eq_port spans min→max of
    # union_rows.trade_date), so the two panes line up exactly.
    trading_days: list = []
    try:
        td_dates = [r["trade_date"] for r in union_rows]
        d0 = _date.fromisoformat(min(td_dates))
        d1 = _date.fromisoformat(max(td_dates))
        async with oi_pool.acquire() as conn:
            td_rows = await conn.fetch(
                "SELECT DISTINCT trade_date FROM daily_features "
                "WHERE trade_date BETWEEN $1 AND $2 "
                "ORDER BY trade_date",
                d0, d1,
            )
        trading_days = [r["trade_date"].isoformat() for r in td_rows]
    except Exception:
        trading_days = sorted({r["trade_date"] for r in union_rows})

    winner_avg = round(avg_winners, 6)
    loser_avg  = round(avg_losers,  6)

    dates_all = [r["trade_date"] for r in union_rows]
    trd_yr: float = float(n)
    if dates_all:
        try:
            d0 = _date.fromisoformat(min(dates_all))
            d1 = _date.fromisoformat(max(dates_all))
            years = (d1 - d0).days / 365.25
            trd_yr = round(n / years, 1) if years > 0.1 else float(n)
        except Exception:
            pass

    # ── Phase 4: Equity ──────────────────────────────────────────────────────
    # Reuse _sec_equity_curve verbatim — cumulative SUM, same formula used by
    # zone-analyze and the secondary-detail component.  Only the input list
    # differs. equity_primary = equity_combined (single curve — no primary
    # universe split; the singleSeries=true path in _renderSecEquity handles
    # this on the frontend).
    eq_port = _sec_equity_curve(union_rows, outcome)

    # ── Yearly breakdown (single series) ────────────────────────────────────
    by_year: dict = defaultdict(list)
    for r in union_rows:
        yr = int(r["trade_date"][:4])
        by_year[yr].append(r[outcome])
    yearly_out = []
    for yr in sorted(by_year):
        yv = np.array(by_year[yr])
        yearly_out.append({
            "year":     yr,
            "n":        len(yv),
            "avg_ret":  round(float(np.mean(yv)), 6),
            "win_rate": round(float(np.mean(yv > 0)), 4),
        })

    # ── Ticker breakdown ─────────────────────────────────────────────────────
    by_ticker: dict = defaultdict(list)
    for r in union_rows:
        by_ticker[r["ticker"]].append(r[outcome])
    total_pnl = sum(sum(v) for v in by_ticker.values())
    tickers_out = []
    for tkr, tv in by_ticker.items():
        ta = np.array(tv)
        tickers_out.append({
            "ticker":      tkr,
            "n":           len(ta),
            "avg_ret":     round(float(np.mean(ta)), 6),
            "win_rate":    round(float(np.mean(ta > 0)), 4),
            "contrib_pct": round(float(np.sum(ta)) / total_pnl * 100, 2)
                           if total_pnl != 0 else 0.0,
        })
    tickers_out.sort(key=lambda x: -x["n"])

    # ── Enriched combined_trades (for activity chart + CSV) ──────────────────
    horizon = _parse_horizon(outcome)
    tickers_in_union = sorted({r["ticker"] for r in union_rows if r.get("ticker")})
    calendars        = await _fetch_ticker_calendars(oi_pool, tickers_in_union)
    combined_trades  = []
    for r in union_rows:
        rec = _build_enriched_trade(r, calendars, horizon,
                                    primary_metric=None, outcome_col=outcome)
        combined_trades.append(rec)

    # ── Phase 5: per-signal contribution stats ───────────────────────────────
    # Computed over each signal's OWN rows (before cross-signal dedup).
    # Per-signal n's will sum to MORE than portfolio n when signals overlap
    # — that is by design and proves the dedup is working.
    total_sum_ret = float(arr.sum())
    signals_out = []
    for entry, sig_trade_rows in zip(enabled_signals, per_signal_rows):
        st = _stats_from_trades(sig_trade_rows, outcome)
        signals_out.append({
            "id":          entry["ps"]["ps_id"],
            "signal_id":   entry["sig"]["id"],
            "name":        entry["sig"]["name"],
            "enabled":     True,
            "n_trades":    st["n"],
            "win_rate":    st["win_rate"],
            "avg_ret":     st["avg_ret"],
            "contrib_pct": round(st["sum_ret"] / total_sum_ret * 100, 2)
                           if total_sum_ret else 0.0,
        })
    # Append disabled signals (zero stats, for display in the UI)
    enabled_ps_ids = {e["ps"]["ps_id"] for e in enabled_signals}
    for ps in all_ps:
        if ps["ps_id"] in enabled_ps_ids:
            continue
        name = sig_by_id.get(ps["signal_id"], {}).get("name", f"Signal {ps['signal_id']}")
        signals_out.append({
            "id": ps["ps_id"], "signal_id": ps["signal_id"], "name": name,
            "enabled": False, "n_trades": 0, "win_rate": 0.0,
            "avg_ret": 0.0, "contrib_pct": 0.0,
        })

    # ── Phase 6: signal-vs-signal phi correlation ────────────────────────────
    # Each signal's binary vector: 1 for (ticker, date) in that signal's
    # trade set, 0 otherwise. Universe = union_rows (all deduped portfolio
    # trades). Same np.corrcoef + M@M.T formula as the old system heatmap.
    phi_out, overlap_out, signal_labels = [], [], []
    if enabled_signals:
        union_pairs = [(r["ticker"], r["trade_date"]) for r in union_rows]
        signal_vecs = []
        for sig_trade_rows in per_signal_rows:
            sig_set = {(r["ticker"], r["trade_date"]) for r in sig_trade_rows}
            vec = np.array([1.0 if p in sig_set else 0.0 for p in union_pairs])
            signal_vecs.append(vec)
        signal_labels = [e["sig"]["name"] for e in enabled_signals]

        if len(signal_vecs) == 1:
            phi_out     = [[1.0]]
            overlap_out = [[int(signal_vecs[0].sum())]]
        else:
            M           = np.stack(signal_vecs)
            phi_arr     = np.nan_to_num(np.corrcoef(M), nan=0.0)
            overlap_arr = (M @ M.T).astype(int)
            phi_out     = [[round(float(v), 4) for v in row] for row in phi_arr]
            overlap_out = [[int(v)             for v in row] for row in overlap_arr]

    # ── Utilisation (same formula, over per-signal raw n's) ─────────────────
    n_each  = [len(sig_trade_rows) for sig_trade_rows in per_signal_rows]
    sum_n   = sum(n_each)
    min_n   = min(n_each) if n_each else 0
    utilisation = round((n - min_n) / (sum_n - min_n) * 100, 1) \
                  if sum_n > min_n else 100.0

    return {
        "portfolio": portfolio,
        "signals":   signals_out,
        "horizon":   horizon,
        # Full 12-field stats bar — same field names as zone/sec-detail
        "n":               n,
        "avg_ret":         round(avg_ret,  6),
        "win_rate":        round(win_rate, 4),
        "std":             round(std,      6),
        "p5":              round(p5,       6),
        "p95":             round(p95,      6),
        "median":          round(median,   6),
        "n_tickers":       n_tickers,
        "n_winners":       n_winners,
        "avg_winners":     round(avg_winners, 6),
        "avg_losers":      round(avg_losers,  6),
        "trades_per_year": trd_yr,
        # Chart data
        "combined_n":           n,
        "equity_primary":       eq_port,  # single curve; equity_primary = equity_combined
        "equity_combined":      eq_port,  # singleSeries=true on frontend
        "yearly":               yearly_out,
        "tickers":              tickers_out,
        "combined_trades":      combined_trades,
        "combined_trade_dates": [r.get("trade_date", "") for r in union_rows],
        "winner_avg_ret":       winner_avg,
        "loser_avg_ret":        loser_avg,
        "n_each":               n_each,
        "utilisation":          utilisation,
        # Signal-vs-signal correlation heatmap
        "phi_systems":       phi_out,
        "overlap_systems":   overlap_out,
        "system_labels":     signal_labels,
        # Pair-level heatmap removed (not applicable to cell-set signals)
        "phi_pairs":         [],
        "overlap_pairs":     [],
        "pair_labels":       [],
        "system_boundaries": [],
        # Outlier filter telemetry — de-duped count of (ticker,date) rows
        # dropped because outcome > req.outlier_max_ret. 0 when off.
        "outlier_excluded":  len(outlier_excluded_keys),
        # Dense trading-day calendar spanning the union trade set's
        # date range; drives the Trade Activity pane's x-axis on load.
        "trading_days":      trading_days,
    }


# ── Loss-Correlation Analysis ────────────────────────────────────────────────


@router.post("/portfolios/{pid}/loss-analysis")
async def portfolio_loss_analysis(
    pid:    int,
    req:    LossAnalysisRequest = Body(default_factory=LossAnalysisRequest),
    pool    = Depends(get_pool),
    oi_pool = Depends(get_oi_pool),
):
    """Weekly dollar-P&L series per signal + downside Spearman correlation.

    Algorithm
    ---------
    Phases 1-2  Same as /aggregate: load portfolio + signals, per-signal
                IS trade queries via is_bins.
    Phase 3     Union dedup (identical to /aggregate).
    Phase 4     Dollar-cap sizing per signal + deduped union → weekly sums.
                Sizing mirrors JS _computeDollarSeries:
                  alloc = min(per_trade, daily_cap / N_day)
                  shares = max(1, floor(alloc / spot_co))
                  dollar_pnl = shares × spot_co × ret
    Phase 5     Downside Spearman correlation: weeks where any signal < 0.
    Phase 6     Top-50 worst weeks by net (deduped) P&L.

    Returns
    -------
    {signal_names, weeks, week_mondays, signal_weekly, net_weekly,
     corr_matrix, corr_n_weeks, worst_weeks, n_weeks}
    """
    if not oi_pool:
        return {"error": "OI database not configured"}
    await _ensure_tables(pool)

    # ── Phase 1a: load portfolio + signals ──────────────────────────────────
    async with pool.acquire() as conn:
        prow = await conn.fetchrow(
            """SELECT id, name, description, ticker, outcome, date_from, date_to
               FROM oi_research_portfolios WHERE id = $1""", pid)
        if not prow:
            raise HTTPException(404, "portfolio not found")
        ps_rows = await conn.fetch(
            """SELECT id AS ps_id, signal_id, position, enabled
               FROM portfolio_signals
               WHERE portfolio_id = $1
               ORDER BY position, id""", pid)

    portfolio  = _portfolio_row_to_dict(prow)
    outcome    = portfolio["outcome"]
    ticker     = portfolio["ticker"]
    all_ps     = [dict(r) for r in ps_rows]
    enabled_ps = [p for p in all_ps if p["enabled"]]

    if not enabled_ps:
        return {"error": "no enabled signals"}

    # ── Phase 1b: load signal definitions ───────────────────────────────────
    all_sids = list({ps["signal_id"] for ps in all_ps})
    async with oi_pool.acquire() as conn:
        sig_rows = await conn.fetch(
            """SELECT id, name, primary_metric, secondary_metric,
                      outcome, n_bins, cell_set
               FROM signals WHERE id = ANY($1)""", all_sids)
    sig_by_id = {r["id"]: dict(r) for r in sig_rows}

    enabled_signals = []
    for ps in enabled_ps:
        sig = sig_by_id.get(ps["signal_id"])
        if not sig:
            continue
        if sig["outcome"] != outcome:
            continue
        enabled_signals.append({"ps": ps, "sig": sig})

    if not enabled_signals:
        return {"error": "no enabled signals with matching outcome"}

    for entry in enabled_signals:
        sig = entry["sig"]
        for m in (sig["primary_metric"], sig["secondary_metric"]):
            if not all(c in _SAFE_METRIC for c in (m or "")):
                return {"error": f"unsafe metric: {m}"}

    # ── Phase 2: per-signal IS trade queries ─────────────────────────────────
    # req dates override (narrow) portfolio dates.
    date_params: list = []
    date_sql    = ""
    p_idx       = 4
    eff_from = req.date_from or portfolio.get("date_from")
    eff_to   = req.date_to   or portfolio.get("date_to")
    if eff_from:
        date_sql += f" AND df.trade_date >= ${p_idx}"
        date_params.append(_date.fromisoformat(eff_from)); p_idx += 1
    if eff_to:
        date_sql += f" AND df.trade_date <= ${p_idx}"
        date_params.append(_date.fromisoformat(eff_to)); p_idx += 1
    ticker_sql = ""
    if ticker != "ALL":
        ticker_sql = f" AND df.ticker = ${p_idx}"
        date_params.append(ticker); p_idx += 1

    outcome_cols = _outcome_select_cols(outcome)
    out_sel = ", ".join(f"df.{c}" for c in outcome_cols)
    out_nn  = " AND ".join(f"df.{c} IS NOT NULL" for c in outcome_cols)

    per_signal_rows: list = []
    async with oi_pool.acquire() as conn:
        for entry in enabled_signals:
            sig    = entry["sig"]
            n_bins = int(sig["n_bins"])
            raw_cs = sig["cell_set"] or "[]"
            cells  = json.loads(raw_cs) if isinstance(raw_cs, str) else raw_cs
            xs = [int(c[0]) for c in cells]
            ys = [int(c[1]) for c in cells]
            prim = sig["primary_metric"]
            sec  = sig["secondary_metric"]

            query = f"""
                SELECT df.ticker, df.trade_date::text AS trade_date,
                       df.spot_co, {out_sel}
                FROM daily_features df
                JOIN is_bins ib USING (ticker, trade_date)
                WHERE ib.bin20_{prim} > 0
                  AND ib.bin20_{sec}  > 0
                  AND {out_nn}
                  AND (
                    ((ib.bin20_{prim} - 1) * $1::int) / 20,
                    ((ib.bin20_{sec}  - 1) * $1::int) / 20
                  ) IN (SELECT * FROM unnest($2::int[], $3::int[]))
                  {date_sql}{ticker_sql}
                ORDER BY df.trade_date, df.ticker
            """
            try:
                raw = await conn.fetch(query, n_bins, xs, ys, *date_params)
            except Exception:
                per_signal_rows.append([])
                continue

            sig_trade_rows = []
            for r in raw:
                ov = _outcome_value(r, outcome)
                if ov is None:
                    continue
                try:
                    fov = float(ov)
                    if math.isnan(fov):
                        continue
                except (TypeError, ValueError):
                    continue
                if req.outlier_max_ret is not None and fov > req.outlier_max_ret:
                    continue
                sig_trade_rows.append({
                    "ticker":     r["ticker"],
                    "trade_date": str(r["trade_date"]),
                    "spot_co":    r.get("spot_co"),
                    outcome:      fov,
                })
            per_signal_rows.append(sig_trade_rows)

    # ── Phase 3: Union dedup ─────────────────────────────────────────────────
    seen: dict = {}
    for sig_trade_rows in per_signal_rows:
        for row in sig_trade_rows:
            key = (row["ticker"], row["trade_date"])
            if key not in seen:
                seen[key] = row
    union_rows = sorted(seen.values(), key=lambda r: (r["trade_date"], r["ticker"]))

    if not union_rows:
        return {"error": "no trades in the selected date range"}

    # ── Build ISO-week grid ──────────────────────────────────────────────────
    all_dates = [r["trade_date"] for r in union_rows]
    d_min = _date.fromisoformat(min(all_dates))
    d_max = _date.fromisoformat(max(all_dates))
    all_weeks    = _all_iso_weeks(d_min, d_max)
    week_mondays = [_week_monday(w) for w in all_weeks]

    # ── Phase 4: Dollar-capped weekly P&L per signal + net ───────────────────
    per_trade = req.per_trade
    daily_cap = req.daily_cap

    signal_week_maps = [
        _dollar_pnl_by_week(srows, outcome, per_trade, daily_cap, all_weeks)
        for srows in per_signal_rows
    ]
    net_week_map = _dollar_pnl_by_week(union_rows, outcome, per_trade, daily_cap, all_weeks)

    signal_weekly = [[wm[w] for w in all_weeks] for wm in signal_week_maps]
    net_weekly    = [net_week_map[w] for w in all_weeks]

    # ── Phase 5: Downside Spearman correlation ────────────────────────────────
    n_sigs       = len(enabled_signals)
    signal_names = [e["sig"]["name"] for e in enabled_signals]

    corr_matrix  = [[1.0 if i == j else None
                     for j in range(n_sigs)]
                    for i in range(n_sigs)]
    corr_n_weeks = 0

    if n_sigs > 1 and _sp_stats is not None and all_weeks:
        sig_arr   = np.array(signal_weekly, dtype=np.float64)   # (n_sigs, n_weeks)
        down_mask = (sig_arr < 0).any(axis=0)                   # weeks where any < 0
        n_down    = int(down_mask.sum())
        corr_n_weeks = n_down
        if n_down >= 3:
            sig_down = sig_arr[:, down_mask].T   # (n_down, n_sigs)
            result   = _sp_stats.spearmanr(sig_down)
            # result[0] is scalar when n_sigs==2, ndarray otherwise
            corr_val = result[0]
            if n_sigs == 2:
                rv = round(float(corr_val), 4)
                corr_matrix = [[1.0, rv], [rv, 1.0]]
            else:
                corr_matrix = [[round(float(v), 4) for v in row]
                               for row in np.array(corr_val)]

    # ── Phase 6: Top-50 worst weeks by net P&L ───────────────────────────────
    worst_weeks = []
    for i, wk in enumerate(all_weeks):
        net = net_weekly[i]
        sig_pnls = [
            {"name": signal_names[j], "pnl": round(signal_weekly[j][i], 2)}
            for j in range(n_sigs)
        ]
        worst_weeks.append({
            "week":        wk,
            "monday":      week_mondays[i],
            "net_pnl":     round(net, 2),
            "signal_pnls": sig_pnls,
        })
    worst_weeks.sort(key=lambda x: x["net_pnl"])
    worst_weeks = worst_weeks[:50]

    return {
        "signal_names":  signal_names,
        "weeks":         all_weeks,
        "week_mondays":  week_mondays,
        "signal_weekly": [[round(v, 2) for v in row] for row in signal_weekly],
        "net_weekly":    [round(v, 2) for v in net_weekly],
        "corr_matrix":   corr_matrix,
        "corr_n_weeks":  corr_n_weeks,
        "worst_weeks":   worst_weeks,
        "n_weeks":       len(all_weeks),
    }
