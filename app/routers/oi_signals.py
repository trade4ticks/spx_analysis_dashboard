"""OI Signals — tracked watchlist + firing engine + Open Positions Calendar.

A signal fires for a ticker when the ticker's stored bin assignments (in
is_bins) for the signal's primary AND secondary metrics fall inside the
signal's cell-set. Firing is evaluated at MAX(trade_date) in is_bins —
the page is anchored to data, not to the calendar clock.

Tracked signals are persisted across sessions in tracked_signals. Two
views over the same set:
  - GET /firing — ticker-centric, one row per ticker with >=1 firing signal
                  on the as_of date. SCOPE A (deduped union across firings
                  for the ticker) + SCOPE A stability (positive years etc.)
                  + SCOPE B (ticker slice) + per-signal expanded breakdown.
  - GET /roster — every tracked signal with overall stats + stability,
                  regardless of today's firing. Watchlist-with-performance.

Performance discipline: every tracked signal's history is fetched exactly
ONCE per request (signal_trade_cache) and every per-ticker aggregation
reads from the cache. A signal firing on N tickers does not produce N
history queries.

Stage 3 adds the Open Positions Calendar back, keyed on the new
(signal_id, ticker, entry_date) shape. The Gantt's +add source now lives
on the per-signal rows in the expanded ticker view (frontend). exit_date
is computed server-side by walking forward `horizon` trading days in the
ticker's daily_features calendar.
"""
import json
import math
import re
from collections import defaultdict
from datetime import date as _date, timedelta
from typing import Optional

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from app.db import get_pool, get_oi_pool

router = APIRouter(tags=["oi_signals"])

# Metric-name allowlist (lowercase ascii + digits + underscore). Used to
# refuse injection via f-string-built SQL on the cell-set query.
_SAFE_METRIC = set("abcdefghijklmnopqrstuvwxyz_0123456789")

# CV(yearly_avg_ret) becomes undefined when |mean| approaches zero. Below
# this threshold (10 bps) the metric explodes to a number that LOOKS real
# but isn't. The endpoint emits None and the frontend renders 'n/a'. Not
# a gate — every other stat still appears for the signal.
_CV_MEAN_EPSILON = 0.001


_DDL = """
CREATE TABLE IF NOT EXISTS oi_signal_calendar (
    id          SERIAL PRIMARY KEY,
    entry_date  DATE NOT NULL,
    added_at    TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS tracked_signals (
    signal_id   INTEGER PRIMARY KEY,
    created_at  TIMESTAMPTZ DEFAULT NOW()
);

-- Stage 1 legacy-schema teardown. The DO block is idempotent: it only
-- runs when the pre-Stage-1 column trigger_id still exists (i.e. on an
-- existing install). After Stage 1 has deployed once, the column is gone
-- and the block is a no-op on every subsequent startup. Fresh installs
-- skip the block entirely.
DO $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'oi_signal_calendar'
          AND column_name = 'trigger_id'
    ) THEN
        TRUNCATE oi_signal_calendar;
        ALTER TABLE oi_signal_calendar
            DROP COLUMN IF EXISTS trigger_id,
            DROP COLUMN IF EXISTS system_id,
            DROP COLUMN IF EXISTS portfolio_id;
    END IF;
END $$;

ALTER TABLE oi_signal_calendar
    ADD COLUMN IF NOT EXISTS ticker  TEXT,
    ADD COLUMN IF NOT EXISTS outcome TEXT;

-- Stage 3.1 migration: calendar entry identity changes from
-- (signal_id, ticker, entry_date) to (ticker, outcome, entry_date).
-- The calendar is signal-agnostic now — a lightweight "what's on"
-- visual. If the same ticker is on a 5d horizon AND a 3d horizon
-- on the same day, that's two separate entries (different exit
-- dates). Old signal-keyed entries get TRUNCATEd; the user re-adds
-- whatever they want via the new + Cal flow on the firing rows.
DO $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'oi_signal_calendar'
          AND column_name = 'signal_id'
    ) THEN
        TRUNCATE oi_signal_calendar;
        ALTER TABLE oi_signal_calendar DROP COLUMN IF EXISTS signal_id;
    END IF;
END $$;

DROP INDEX IF EXISTS oi_signal_calendar_trigger_uniq;
DROP INDEX IF EXISTS oi_signal_calendar_system_uniq;
DROP INDEX IF EXISTS oi_signal_calendar_signal_uniq;
CREATE UNIQUE INDEX IF NOT EXISTS oi_signal_calendar_outcome_uniq
    ON oi_signal_calendar (ticker, outcome, entry_date)
    WHERE outcome IS NOT NULL;

DROP TABLE IF EXISTS oi_signal_triggers         CASCADE;
DROP TABLE IF EXISTS oi_research_systems        CASCADE;
DROP TABLE IF EXISTS oi_research_system_library CASCADE;
"""

_ensured = False


async def _ensure_tables(pool):
    global _ensured
    if _ensured:
        return
    async with pool.acquire() as conn:
        await conn.execute(_DDL)
    _ensured = True


# ── Pydantic models ─────────────────────────────────────────────────────────


class TrackedSignalIn(BaseModel):
    signal_id: int


class TrackedFromPortfolioIn(BaseModel):
    portfolio_id: int


class CalendarAddIn(BaseModel):
    ticker:     str
    outcome:    str
    entry_date: str   # ISO YYYY-MM-DD


# ── Tracked-signals CRUD ────────────────────────────────────────────────────


@router.get("/tracked")
async def list_tracked(pool=Depends(get_pool), oi_pool=Depends(get_oi_pool)):
    """List every tracked signal, enriched with metadata from `signals`."""
    await _ensure_tables(pool)
    async with pool.acquire() as conn:
        tracked_rows = await conn.fetch(
            "SELECT signal_id, created_at FROM tracked_signals "
            "ORDER BY created_at")
    if not tracked_rows:
        return {"tracked": []}
    sids = [r["signal_id"] for r in tracked_rows]
    tracked_at_by_id = {r["signal_id"]: str(r["created_at"])[:19]
                        for r in tracked_rows}

    sig_by_id: dict = {}
    if oi_pool:
        async with oi_pool.acquire() as conn:
            sig_rows = await conn.fetch(
                """SELECT id, name, primary_metric, secondary_metric, outcome,
                          n_bins, cell_set, created_at
                   FROM signals WHERE id = ANY($1)""", sids)
        sig_by_id = {r["id"]: r for r in sig_rows}

    out = []
    for sid in sids:
        s = sig_by_id.get(sid)
        if not s:
            out.append({
                "signal_id":  sid,
                "tracked_at": tracked_at_by_id[sid],
                "missing":    True,
            })
            continue
        cell_set = s["cell_set"]
        if isinstance(cell_set, str):
            cell_set = json.loads(cell_set)
        out.append({
            "signal_id":        s["id"],
            "name":             s["name"],
            "primary_metric":   s["primary_metric"],
            "secondary_metric": s["secondary_metric"],
            "outcome":          s["outcome"],
            "n_bins":           s["n_bins"],
            "cell_set":         cell_set,
            "tracked_at":       tracked_at_by_id[sid],
            "created_at":       str(s["created_at"])[:19],
        })
    return {"tracked": out}


@router.post("/tracked")
async def add_tracked(body: TrackedSignalIn,
                      pool=Depends(get_pool),
                      oi_pool=Depends(get_oi_pool)):
    """Add one signal to the tracked watchlist. Idempotent — already-tracked
    signal returns ok without inserting a duplicate."""
    await _ensure_tables(pool)
    if oi_pool:
        async with oi_pool.acquire() as conn:
            exists = await conn.fetchval(
                "SELECT 1 FROM signals WHERE id = $1", body.signal_id)
        if not exists:
            raise HTTPException(404, f"signal_id {body.signal_id} not found")
    async with pool.acquire() as conn:
        await conn.execute(
            "INSERT INTO tracked_signals (signal_id) VALUES ($1) "
            "ON CONFLICT (signal_id) DO NOTHING",
            body.signal_id)
    return {"ok": True, "signal_id": body.signal_id}


@router.post("/tracked/from-portfolio")
async def add_tracked_from_portfolio(body: TrackedFromPortfolioIn,
                                     pool=Depends(get_pool)):
    """Bulk-add: flatten a portfolio into its constituent signals and add
    each to tracked_signals. Portfolio identity does NOT persist — only the
    signal_ids land in tracked_signals. Dedupe by signal_id (a signal
    already tracked is counted as 'already_present', not 're-added')."""
    await _ensure_tables(pool)
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT signal_id FROM portfolio_signals "
            "WHERE portfolio_id = $1", body.portfolio_id)
        if not rows:
            raise HTTPException(
                404, f"portfolio_id {body.portfolio_id} not found or has no signals")
        sids = [r["signal_id"] for r in rows]
        already_rows = await conn.fetch(
            "SELECT signal_id FROM tracked_signals WHERE signal_id = ANY($1)",
            sids)
        already = {r["signal_id"] for r in already_rows}
        new_sids = [s for s in sids if s not in already]
        if new_sids:
            await conn.executemany(
                "INSERT INTO tracked_signals (signal_id) VALUES ($1) "
                "ON CONFLICT (signal_id) DO NOTHING",
                [(s,) for s in new_sids])
    return {
        "total_in_portfolio": len(sids),
        "added":              len(new_sids),
        "already_present":    len(sids) - len(new_sids),
    }


@router.delete("/tracked/{signal_id}")
async def remove_tracked(signal_id: int, pool=Depends(get_pool)):
    """Untrack a signal. Idempotent — returns ok=False if not tracked."""
    await _ensure_tables(pool)
    async with pool.acquire() as conn:
        deleted = await conn.fetchval(
            "DELETE FROM tracked_signals WHERE signal_id = $1 "
            "RETURNING signal_id", signal_id)
    return {"ok": deleted is not None}


# ── Stats helpers (reuse the portfolio aggregate stats shape) ───────────────


def _full_stats(rets: list) -> dict:
    """SCOPE A primary stats over a list of realized return floats. No
    gates: an empty list returns the schema with zeros. The 9-field block
    matches what the stats bar on the Factor Analysis page already
    renders."""
    n = len(rets)
    if n == 0:
        return {"n": 0, "avg_ret": 0.0, "median": 0.0, "std_dev": 0.0,
                "p5": 0.0, "p95": 0.0, "win_rate": 0.0,
                "avg_win": 0.0, "avg_loss": 0.0}
    arr    = np.array(rets, dtype=np.float64)
    wins   = arr[arr > 0]
    losses = arr[arr < 0]
    return {
        "n":        n,
        "avg_ret":  round(float(arr.mean()), 6),
        "median":   round(float(np.median(arr)), 6),
        "std_dev":  round(float(arr.std()), 6),
        "p5":       round(float(np.percentile(arr, 5)), 6),
        "p95":      round(float(np.percentile(arr, 95)), 6),
        "win_rate": round(float((arr > 0).mean()), 4),
        "avg_win":  round(float(wins.mean()),   6) if len(wins)   else 0.0,
        "avg_loss": round(float(losses.mean()), 6) if len(losses) else 0.0,
    }


def _compact_stats(rets: list) -> dict:
    """SCOPE B — three fields. Same no-gate rule."""
    n = len(rets)
    if n == 0:
        return {"n": 0, "avg_ret": 0.0, "win_rate": 0.0}
    arr = np.array(rets, dtype=np.float64)
    return {
        "n":        n,
        "avg_ret":  round(float(arr.mean()), 6),
        "win_rate": round(float((arr > 0).mean()), 4),
    }


def _stability_stats(trades: list) -> dict:
    """Group fixed-trade-set by calendar year. No within-year re-ranking —
    the trades are already fixed by cell-set membership. Returns:
        positive_years      : count of years where the year's avg_ret > 0
        total_years         : distinct years with realized trades
        cv_yearly_avg_ret   : CV of per-year avg_ret. None when |mean| is
                              within epsilon of zero (the zero-boundary
                              blowup case — see _CV_MEAN_EPSILON).
        dispersion_yearly_n : CV of per-year trade count.
        yearly              : [{year, n, avg_ret}, ...] raw breakdown."""
    by_year: dict = defaultdict(list)
    for t in trades:
        ov = t.get("outcome_val")
        if ov is None:
            continue
        try:
            yr = int(t["trade_date"][:4])
        except (TypeError, ValueError, IndexError, KeyError):
            continue
        by_year[yr].append(float(ov))
    if not by_year:
        return {"positive_years": 0, "total_years": 0,
                "cv_yearly_avg_ret": None, "dispersion_yearly_n": None,
                "yearly": []}
    years = sorted(by_year.keys())
    yearly = []
    avgs   = []
    ns     = []
    positive = 0
    for yr in years:
        vals = by_year[yr]
        a = np.array(vals, dtype=np.float64)
        yr_avg = float(a.mean())
        yearly.append({"year": yr, "n": len(vals),
                       "avg_ret": round(yr_avg, 6)})
        avgs.append(yr_avg)
        ns.append(len(vals))
        if yr_avg > 0:
            positive += 1
    avg_arr = np.array(avgs, dtype=np.float64)
    m       = float(avg_arr.mean())
    cv_avg  = (None if abs(m) < _CV_MEAN_EPSILON
               else round(float(avg_arr.std()) / abs(m), 4))
    n_arr   = np.array(ns, dtype=np.float64)
    mn      = float(n_arr.mean())
    disp_n  = round(float(n_arr.std()) / mn, 4) if mn > 0 else None
    return {
        "positive_years":      positive,
        "total_years":         len(years),
        "cv_yearly_avg_ret":   cv_avg,
        "dispersion_yearly_n": disp_n,
        "yearly":              yearly,
    }


# ── Per-signal trade cache ──────────────────────────────────────────────────


async def _fetch_signal_trade_cache(oi_pool, signals: list) -> dict:
    """Per-signal trade history. Run ONCE per request and reuse for every
    per-ticker aggregation and the roster — a signal firing on N tickers
    is still queried only once.

    Each cache entry is a list of {ticker, trade_date (str), outcome_val
    (float|None)}. outcome_val is None on rows where the forward return
    hasn't realized yet (the as_of row itself); these rows still appear
    so firing detection at as_of works. Stats code filters None.
    """
    cache: dict = {}
    if not signals:
        return cache
    async with oi_pool.acquire() as conn:
        for sig in signals:
            sid    = sig["id"]
            prim   = sig["primary_metric"]
            sec    = sig["secondary_metric"]
            out    = sig["outcome"]
            n_bins = int(sig["n_bins"])
            cell_set = sig["cell_set"]
            if isinstance(cell_set, str):
                cell_set = json.loads(cell_set)
            if not cell_set:
                cache[sid] = []
                continue
            if (any(c not in _SAFE_METRIC for c in (prim or "")) or
                any(c not in _SAFE_METRIC for c in (sec  or "")) or
                any(c not in _SAFE_METRIC for c in (out  or ""))):
                cache[sid] = []
                continue
            xs = [int(c[0]) for c in cell_set]
            ys = [int(c[1]) for c in cell_set]
            query = f"""
                SELECT df.ticker,
                       df.trade_date::text AS trade_date,
                       df.{out} AS outcome_val
                FROM daily_features df
                JOIN is_bins ib USING (ticker, trade_date)
                WHERE ib.bin20_{prim} > 0
                  AND ib.bin20_{sec}  > 0
                  AND (
                    ((ib.bin20_{prim} - 1) * $1::int) / 20,
                    ((ib.bin20_{sec}  - 1) * $1::int) / 20
                  ) IN (SELECT * FROM unnest($2::int[], $3::int[]))
                ORDER BY df.trade_date, df.ticker
            """
            try:
                rows = await conn.fetch(query, n_bins, xs, ys)
            except Exception:
                cache[sid] = []
                continue
            trades = []
            for r in rows:
                raw = r["outcome_val"]
                fov: Optional[float] = None
                if raw is not None:
                    try:
                        f = float(raw)
                        if not math.isnan(f):
                            fov = f
                    except (TypeError, ValueError):
                        pass
                trades.append({
                    "ticker":      r["ticker"],
                    "trade_date":  str(r["trade_date"]),
                    "outcome_val": fov,
                })
            cache[sid] = trades
    return cache


# ── Firing engine ──────────────────────────────────────────────────────────


@router.get("/firing")
async def get_firing(
    date: Optional[str] = Query(None),
    pool=Depends(get_pool),
    oi_pool=Depends(get_oi_pool),
):
    """Firing engine — anchored to MAX(trade_date) in is_bins by default.

    Per-ticker aggregation reuses the portfolio aggregate dedup pattern:
    a (ticker, trade_date, outcome) appearing in multiple firing signals
    counts ONCE in SCOPE A union stats. Stats are computed without gates
    — even small samples surface; n is reported so the user can judge.
    """
    await _ensure_tables(pool)
    if not oi_pool:
        return {"error": "OI database not configured",
                "as_of": None, "rows": []}

    async with oi_pool.acquire() as conn:
        if date:
            try:
                as_of = _date.fromisoformat(date)
            except ValueError:
                raise HTTPException(400, "date must be ISO YYYY-MM-DD")
        else:
            as_of = await conn.fetchval(
                "SELECT MAX(trade_date) FROM is_bins")
        if as_of is None:
            return {"error": "is_bins is empty",
                    "as_of": None, "rows": []}
    as_of_str = (as_of.isoformat() if hasattr(as_of, "isoformat")
                 else str(as_of))

    async with pool.acquire() as conn:
        tracked_rows = await conn.fetch(
            "SELECT signal_id FROM tracked_signals")
    if not tracked_rows:
        return {"as_of": as_of_str, "rows": [],
                "n_tracked": 0, "n_firing_rows": 0, "n_firing_tickers": 0}

    sids = [r["signal_id"] for r in tracked_rows]
    async with oi_pool.acquire() as conn:
        sig_rows = await conn.fetch(
            """SELECT id, name, primary_metric, secondary_metric, outcome,
                      n_bins, cell_set
               FROM signals WHERE id = ANY($1)""", sids)
    signals   = [dict(r) for r in sig_rows]
    sig_by_id = {s["id"]: s for s in signals}

    cache = await _fetch_signal_trade_cache(oi_pool, signals)

    # Detect firings, grouped by (ticker, outcome) — NOT by ticker alone.
    # avg_ret is only comparable within ONE outcome horizon (a 1d-fwd
    # return and a 20d-fwd return live on totally different scales —
    # averaging them produces nonsense). So a ticker firing a 5d signal
    # AND a 20d signal renders as TWO separate rows. Each row is
    # single-outcome by construction and every downstream stat is
    # comparable.
    firings_by_group: dict = defaultdict(list)   # (ticker, outcome) -> [sid,...]
    for sid, trades in cache.items():
        outcome = sig_by_id[sid]["outcome"]
        for t in trades:
            if t["trade_date"] == as_of_str:
                firings_by_group[(t["ticker"], outcome)].append(sid)
                break

    if not firings_by_group:
        return {"as_of": as_of_str, "rows": [],
                "n_tracked": len(sids), "n_firing_rows": 0,
                "n_firing_tickers": 0}

    out_rows = []
    for ticker, outcome in sorted(firings_by_group.keys()):
        firing_sids = firings_by_group[(ticker, outcome)]

        # SCOPE A union. All signals in this group share the outcome
        # (the grouping IS by outcome), so dedup key drops to
        # (ticker, trade_date) — same first-wins semantic as portfolio
        # aggregate. avg_ret over this union is comparable because every
        # trade is on the same horizon.
        seen: dict = {}
        for sid in firing_sids:
            for t in cache.get(sid, []):
                if t["outcome_val"] is None:
                    continue
                key = (t["ticker"], t["trade_date"])
                if key not in seen:
                    seen[key] = t
        union_trades = list(seen.values())

        scope_a       = _full_stats([t["outcome_val"] for t in union_trades])
        scope_a_stab  = _stability_stats(union_trades)
        ticker_trades = [t for t in union_trades if t["ticker"] == ticker]
        scope_b       = _compact_stats([t["outcome_val"] for t in ticker_trades])

        # Per-signal detail. Naturally filtered to this row's outcome
        # since firing_sids only contains sids matching it — no extra
        # filtering needed.
        signals_out = []
        for sid in firing_sids:
            sig        = sig_by_id[sid]
            sig_trades = cache.get(sid, [])
            overall_rets = [t["outcome_val"] for t in sig_trades
                            if t["outcome_val"] is not None]
            ticker_rets  = [t["outcome_val"] for t in sig_trades
                            if t["ticker"] == ticker
                            and t["outcome_val"] is not None]
            signals_out.append({
                "signal_id":        sid,
                "name":             sig["name"],
                "primary_metric":   sig["primary_metric"],
                "secondary_metric": sig["secondary_metric"],
                "outcome":          sig["outcome"],
                "n_bins":           sig["n_bins"],
                "overall":          _full_stats(overall_rets),
                "ticker_slice":     _compact_stats(ticker_rets),
            })

        out_rows.append({
            "ticker":            ticker,
            "outcome":           outcome,
            "n_signals_firing":  len(firing_sids),
            "scope_a":           scope_a,
            "scope_a_stability": scope_a_stab,
            "scope_b":           scope_b,
            "signals_firing":    signals_out,
        })

    return {
        "as_of":             as_of_str,
        "n_tracked":         len(sids),
        "n_firing_rows":     len(out_rows),
        # Distinct tickers across all rows (one ticker firing on N
        # outcomes still counts as one ticker for this metric).
        "n_firing_tickers":  len({r["ticker"] for r in out_rows}),
        "rows":              out_rows,
    }


# ── Roster ──────────────────────────────────────────────────────────────────


@router.get("/roster")
async def get_roster(pool=Depends(get_pool), oi_pool=Depends(get_oi_pool)):
    """Roster — every tracked signal with overall stats + stability stats,
    independent of today's firing. Watchlist-with-performance lens. Shares
    the per-signal trade cache strategy with /firing."""
    await _ensure_tables(pool)
    if not oi_pool:
        return {"tracked": [], "n_tracked": 0}
    async with pool.acquire() as conn:
        tracked_rows = await conn.fetch(
            "SELECT signal_id, created_at FROM tracked_signals "
            "ORDER BY created_at")
    if not tracked_rows:
        return {"tracked": [], "n_tracked": 0}
    sids = [r["signal_id"] for r in tracked_rows]
    tracked_at_by_id = {r["signal_id"]: str(r["created_at"])[:19]
                        for r in tracked_rows}
    async with oi_pool.acquire() as conn:
        sig_rows = await conn.fetch(
            """SELECT id, name, primary_metric, secondary_metric, outcome,
                      n_bins, cell_set, created_at
               FROM signals WHERE id = ANY($1)""", sids)
    signals   = [dict(r) for r in sig_rows]
    sig_by_id = {s["id"]: s for s in signals}

    cache = await _fetch_signal_trade_cache(oi_pool, signals)

    out = []
    for sid in sids:
        s = sig_by_id.get(sid)
        if not s:
            out.append({
                "signal_id":  sid,
                "tracked_at": tracked_at_by_id[sid],
                "missing":    True,
            })
            continue
        trades = cache.get(sid, [])
        rets   = [t["outcome_val"] for t in trades
                  if t["outcome_val"] is not None]
        cell_set = s["cell_set"]
        if isinstance(cell_set, str):
            cell_set = json.loads(cell_set)
        out.append({
            "signal_id":        s["id"],
            "name":             s["name"],
            "primary_metric":   s["primary_metric"],
            "secondary_metric": s["secondary_metric"],
            "outcome":          s["outcome"],
            "n_bins":           s["n_bins"],
            "cell_set":         cell_set,
            "tracked_at":       tracked_at_by_id[sid],
            "created_at":       str(s["created_at"])[:19],
            "overall":          _full_stats(rets),
            "stability":        _stability_stats(trades),
        })
    return {"tracked": out, "n_tracked": len(out)}


# ── Open Positions Calendar ─────────────────────────────────────────────────


def _parse_horizon(outcome: str) -> int:
    """Extract the integer day count from an outcome column name like
    ret_5d_fwd_oc → 5. Falls back to 1 on unparseable names."""
    m = re.search(r"(\d+)d", outcome or "")
    return int(m.group(1)) if m else 1


async def _ticker_calendars(oi_pool, tickers: list) -> dict:
    """{ticker: [sorted distinct trade_date,...]} for exit_date walks.
    One query covers every ticker referenced by the calendar entries."""
    if not tickers or not oi_pool:
        return {}
    async with oi_pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT ticker, trade_date FROM daily_features "
            "WHERE ticker = ANY($1) ORDER BY ticker, trade_date",
            tickers)
    by_t: dict = defaultdict(list)
    for r in rows:
        by_t[r["ticker"]].append(r["trade_date"])
    return by_t


def _exit_date_for(td_list: list, entry: _date, horizon: int) -> Optional[str]:
    """Walk horizon trading days forward from entry_date in the ticker's
    sorted calendar. Returns ISO date string or None on no-data."""
    if not td_list or horizon <= 0:
        return None
    idx = next((i for i, x in enumerate(td_list) if x >= entry), None)
    if idx is None:
        return None
    exit_idx = idx + max(horizon - 1, 0)
    if exit_idx < len(td_list):
        return td_list[exit_idx].isoformat()
    # Past the calendar's end (rare — only for very recent entries before
    # the next OHLC ingest). Extrapolate at ~1.4 cal days per trading day.
    extra = exit_idx - len(td_list) + 1
    return (td_list[-1] + timedelta(days=int(extra * 1.4))).isoformat()


@router.get("/calendar")
async def list_calendar(pool=Depends(get_pool), oi_pool=Depends(get_oi_pool)):
    """Calendar entries with computed exit_date. Identity is
    (ticker, outcome, entry_date) — signal-agnostic. exit_date is
    derived from the row's OWN outcome horizon (no signal lookup);
    color is derived frontend-side from a hash of the ticker symbol."""
    await _ensure_tables(pool)
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """SELECT id, ticker, outcome, entry_date, added_at
               FROM oi_signal_calendar
               WHERE outcome IS NOT NULL
               ORDER BY entry_date""")
    if not rows:
        return []
    tickers   = list({r["ticker"] for r in rows if r["ticker"]})
    calendars = await _ticker_calendars(oi_pool, tickers)
    out = []
    for r in rows:
        horizon = _parse_horizon(r["outcome"])
        td      = calendars.get(r["ticker"], [])
        exit_d  = _exit_date_for(td, r["entry_date"], horizon)
        out.append({
            "id":         r["id"],
            "ticker":     r["ticker"],
            "outcome":    r["outcome"],
            "entry_date": r["entry_date"].isoformat(),
            "exit_date":  exit_d,
        })
    return out


@router.post("/calendar")
async def add_calendar(body: CalendarAddIn, pool=Depends(get_pool)):
    """Add a position to the Gantt. Keyed on (ticker, outcome, entry_date).
    Duplicate adds (same ticker + outcome + day) are no-ops via the
    partial unique index — adding the same firing twice on the same day
    doesn't create a second bar."""
    await _ensure_tables(pool)
    try:
        entry = _date.fromisoformat(body.entry_date)
    except ValueError:
        raise HTTPException(400, "entry_date must be ISO YYYY-MM-DD")
    if not body.ticker or not body.outcome:
        raise HTTPException(400, "ticker and outcome are required")
    async with pool.acquire() as conn:
        await conn.execute(
            """INSERT INTO oi_signal_calendar (ticker, outcome, entry_date)
               VALUES ($1, $2, $3)
               ON CONFLICT (ticker, outcome, entry_date) DO NOTHING""",
            body.ticker, body.outcome, entry)
        row = await conn.fetchrow(
            """SELECT id FROM oi_signal_calendar
               WHERE ticker = $1 AND outcome = $2 AND entry_date = $3""",
            body.ticker, body.outcome, entry)
    return {"id": row["id"], "ticker": body.ticker,
            "outcome": body.outcome, "entry_date": body.entry_date}


@router.delete("/calendar/{cid}")
async def delete_calendar(cid: int, pool=Depends(get_pool)):
    """Remove one calendar entry by id."""
    await _ensure_tables(pool)
    async with pool.acquire() as conn:
        deleted = await conn.fetchval(
            "DELETE FROM oi_signal_calendar WHERE id = $1 RETURNING id", cid)
    return {"ok": deleted is not None}
