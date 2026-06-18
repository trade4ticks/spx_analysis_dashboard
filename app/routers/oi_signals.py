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


def _parse_signal_jsonb(v):
    """JSONB columns may come back as parsed list/dict (asyncpg JSONB codec)
    or as a JSON string depending on environment. Mirrors the helper at
    oi_analysis.py:_parse_jsonb so per_cell_stats round-trips reliably."""
    if v is None:
        return None
    if isinstance(v, (list, dict)):
        return v
    try:
        return json.loads(v)
    except (TypeError, ValueError):
        return None

router = APIRouter(tags=["oi_signals"])

# Metric-name allowlist (lowercase ascii + digits + underscore). Used to
# refuse injection via f-string-built SQL on the cell-set query.
_SAFE_METRIC = set("abcdefghijklmnopqrstuvwxyz_0123456789")

# CV(yearly_avg_ret) becomes undefined when |mean| approaches zero. Below
# this threshold (10 bps) the metric explodes to a number that LOOKS real
# but isn't. The endpoint emits None and the frontend renders 'n/a'. Not
# a gate — every other stat still appears for the signal.
_CV_MEAN_EPSILON = 0.001


# Schema migration broken into independent, idempotent steps. Each step
# is executed in its own conn.execute() call so a stumble on one statement
# can't silently skip the rest (which the prior single-mega-block
# approach did — the partial unique index never got created on at least
# one VPS instance, leaving the table without the constraint that the
# INSERT's ON CONFLICT clause infers against).
#
# Each step's own internal guard (IF EXISTS / IF NOT EXISTS / DO block
# check) decides what to do given the CURRENT real schema state, so a
# half-migrated table heals on the next restart without any one outer
# gate skipping the whole repair.
_DDL_STEPS = [
    # 1. Make sure the tables exist.
    """CREATE TABLE IF NOT EXISTS oi_signal_calendar (
        id          SERIAL PRIMARY KEY,
        entry_date  DATE NOT NULL,
        added_at    TIMESTAMPTZ DEFAULT NOW()
    )""",
    """CREATE TABLE IF NOT EXISTS tracked_signals (
        signal_id   INTEGER PRIMARY KEY,
        created_at  TIMESTAMPTZ DEFAULT NOW()
    )""",

    # 2. Add the columns we use today. IF NOT EXISTS so re-runs are no-ops.
    "ALTER TABLE oi_signal_calendar ADD COLUMN IF NOT EXISTS ticker  TEXT",
    "ALTER TABLE oi_signal_calendar ADD COLUMN IF NOT EXISTS outcome TEXT",

    # 3. Drop any legacy columns one at a time. Each check is independent
    #    so a half-migrated state (some legacy cols dropped, others not)
    #    heals cleanly. TRUNCATE wipes legacy rows whose identity is
    #    being removed — the user re-adds via the new flow.
    """DO $$ BEGIN
        IF EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_name='oi_signal_calendar' AND column_name='trigger_id') THEN
            TRUNCATE oi_signal_calendar;
            ALTER TABLE oi_signal_calendar DROP COLUMN trigger_id;
        END IF;
    END $$""",
    """DO $$ BEGIN
        IF EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_name='oi_signal_calendar' AND column_name='system_id') THEN
            TRUNCATE oi_signal_calendar;
            ALTER TABLE oi_signal_calendar DROP COLUMN system_id;
        END IF;
    END $$""",
    """DO $$ BEGIN
        IF EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_name='oi_signal_calendar' AND column_name='portfolio_id') THEN
            TRUNCATE oi_signal_calendar;
            ALTER TABLE oi_signal_calendar DROP COLUMN portfolio_id;
        END IF;
    END $$""",
    """DO $$ BEGIN
        IF EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_name='oi_signal_calendar' AND column_name='signal_id') THEN
            TRUNCATE oi_signal_calendar;
            ALTER TABLE oi_signal_calendar DROP COLUMN signal_id;
        END IF;
    END $$""",

    # 4. Drop legacy indexes — independent IF EXISTS so each is a clean
    #    no-op when missing.
    "DROP INDEX IF EXISTS oi_signal_calendar_trigger_uniq",
    "DROP INDEX IF EXISTS oi_signal_calendar_system_uniq",
    "DROP INDEX IF EXISTS oi_signal_calendar_signal_uniq",

    # 5. Create the partial unique index that the INSERT's ON CONFLICT
    #    (ticker, outcome, entry_date) WHERE outcome IS NOT NULL clause
    #    infers against. IF NOT EXISTS so it's safe on every restart.
    #    This is the step that was previously silently skipped on at
    #    least one DB; isolating it in its own execute() guarantees it
    #    runs independent of any other step's success.
    """CREATE UNIQUE INDEX IF NOT EXISTS oi_signal_calendar_outcome_uniq
        ON oi_signal_calendar (ticker, outcome, entry_date)
        WHERE outcome IS NOT NULL""",

    # 6. Drop legacy tables — independent and idempotent.
    "DROP TABLE IF EXISTS oi_signal_triggers         CASCADE",
    "DROP TABLE IF EXISTS oi_research_systems        CASCADE",
    "DROP TABLE IF EXISTS oi_research_system_library CASCADE",
]

_ensured = False


async def _ensure_tables(pool):
    global _ensured
    if _ensured:
        return
    async with pool.acquire() as conn:
        for sql in _DDL_STEPS:
            await conn.execute(sql)
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
                "n_tracked": 0, "n_firing_rows": 0, "n_firing_tickers": 0,
                "all_signals": []}

    sids = [r["signal_id"] for r in tracked_rows]
    async with oi_pool.acquire() as conn:
        # per_cell_stats + agg_avg_ret + agg_n added so the response can
        # carry the column thumbnail data (per_cell_stats drives the
        # SignalThumb.thumbnailSVG render) and the lifetime stats used
        # by the Signals page's fixed-column firing grid in ALL mode.
        sig_rows = await conn.fetch(
            """SELECT id, name, primary_metric, secondary_metric, outcome,
                      n_bins, cell_set,
                      per_cell_stats, agg_avg_ret, agg_n,
                      status, color_slot, corner
               FROM signals WHERE id = ANY($1)""", sids)
    signals   = [dict(r) for r in sig_rows]
    sig_by_id = {s["id"]: s for s in signals}

    # Build all_signals[] up front so every return path below can include
    # it — when the firing set is empty, the grid still renders its
    # fixed-position column headers (just with no row data), which is
    # what gives the layout its day-to-day visual consistency.
    all_signals_out = [
        {
            "signal_id":        sig["id"],
            "name":             sig["name"],
            "primary_metric":   sig["primary_metric"],
            "secondary_metric": sig["secondary_metric"],
            "outcome":          sig["outcome"],
            "n_bins":           sig["n_bins"],
            "per_cell_stats":   _parse_signal_jsonb(sig.get("per_cell_stats")) or [],
            "agg_avg_ret":      sig.get("agg_avg_ret"),
            "agg_n":            sig.get("agg_n") or 0,
            # Identity color + tier — drives the column-header color bar
            # in the firing grid via SignalThumb.colorForSlot (single
            # source shared with Saved Signals / Lab / Corner Scan /
            # Portfolio cards). Test rows degrade to neutral gray.
            "status":           sig.get("status") or "Test",
            "color_slot":       sig.get("color_slot"),
            "corner":           sig.get("corner"),
        }
        for sig in sorted(signals, key=lambda s: s["id"])
    ]

    cache = await _fetch_signal_trade_cache(oi_pool, signals)

    # Detect firings, grouped by (ticker, outcome) — NOT by ticker alone.
    # avg_ret is only comparable within ONE outcome horizon (a 1d-fwd
    # return and a 20d-fwd return live on totally different scales —
    # averaging them produces nonsense). So a ticker firing a 5d signal
    # AND a 20d signal renders as TWO separate rows. Each row is
    # single-outcome by construction and every downstream stat is
    # comparable.
    # The Signals page enumerates EVERY (ticker, trade_date == as_of)
    # firing — it's a ticker-discovery tool, with per-ticker signal
    # attribution as a subset view. Previously this loop broke after
    # the first match per signal, which captured only the alphabetically-
    # first ticker that signal fired on for the date (cache rows are
    # ORDER BY trade_date, ticker). That collapsed multi-ticker firings
    # to one per signal AND corrupted per-ticker attribution (a ticker
    # driven by N signals would list only those whose first-alpha
    # firing landed there). Collect every match, dedupe via set in
    # case a signal yields duplicate (ticker, as_of) rows defensively.
    firings_by_group: dict = defaultdict(set)    # (ticker, outcome) -> {sid, ...}
    for sid, trades in cache.items():
        outcome = sig_by_id[sid]["outcome"]
        for t in trades:
            if t["trade_date"] == as_of_str:
                firings_by_group[(t["ticker"], outcome)].add(sid)

    if not firings_by_group:
        return {"as_of": as_of_str, "rows": [],
                "n_tracked": len(sids), "n_firing_rows": 0,
                "n_firing_tickers": 0,
                "all_signals": all_signals_out}

    out_rows = []
    for ticker, outcome in sorted(firings_by_group.keys()):
        # Sort for deterministic per-row signals_firing output now that
        # the source is a set rather than an append-ordered list.
        firing_sids = sorted(firings_by_group[(ticker, outcome)])

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
        "all_signals":       all_signals_out,
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


_NYSE_VALID_DAYS_CACHE: dict = {"date": None, "valid_days": None}


def _get_nyse_valid_days() -> Optional[set]:
    """Forward NYSE trading-day set from pandas_market_calendars.

    Used by _exit_date_for's forward-projection branch when the
    entry is at or past daily_features's last trade_date. The
    library knows floating holidays (MLK 3rd Mon Jan, Memorial
    last Mon May, Labor 1st Mon Sep, Thanksgiving 4th Thu Nov,
    Good Friday by Easter) — the (mm, dd) heuristic this replaces
    couldn't.

    Cached daily — the underlying NYSE calendar is static so one
    fetch covers every calendar request that day. Returns None
    when pandas_market_calendars isn't installed; callers fall back
    to weekend-only skipping and log nothing (the in-range branch
    via daily_features remains the source of truth where data
    exists, so the bleeding-edge forward case is the only thing
    that degrades).

    Range: 30 days back to 400 days forward of today. The back
    overlap exists ONLY for the seam-verification logic the user
    asked about — the most recent known holiday in daily_features
    should also be a holiday in this library, otherwise the
    handoff between branches drifts off-by-one near the data
    boundary."""
    today = _date.today()
    if _NYSE_VALID_DAYS_CACHE["date"] == today:
        return _NYSE_VALID_DAYS_CACHE["valid_days"]
    try:
        import pandas_market_calendars as mcal   # type: ignore
    except ImportError:
        _NYSE_VALID_DAYS_CACHE["date"] = today
        _NYSE_VALID_DAYS_CACHE["valid_days"] = None
        return None
    try:
        nyse  = mcal.get_calendar("NYSE")
        start = today - timedelta(days=30)
        end   = today + timedelta(days=400)
        sched = nyse.valid_days(start_date=start.isoformat(),
                                  end_date=end.isoformat())
        valid = {ts.date() for ts in sched}
    except Exception:
        valid = None
    _NYSE_VALID_DAYS_CACHE["date"] = today
    _NYSE_VALID_DAYS_CACHE["valid_days"] = valid
    return valid


def _next_business_day(d: _date, valid_days: Optional[set] = None) -> _date:
    """Add one calendar day, then advance until reaching the next
    valid trading day. valid_days is the NYSE set from
    _get_nyse_valid_days. When None (library unavailable), falls
    back to weekend-only skipping — the documented limitation."""
    d = d + timedelta(days=1)
    if valid_days:
        # Safety cap so a malformed cache can't hang the request.
        guard = 0
        while d not in valid_days and guard < 60:
            d = d + timedelta(days=1)
            guard += 1
        return d
    # Library unavailable: weekend-only fallback (off for holidays).
    while d.weekday() >= 5:
        d = d + timedelta(days=1)
    return d


def _exit_date_for(td_list: list, entry: _date, horizon: int,
                    valid_days: Optional[set] = None) -> Optional[str]:
    """Exit date = the Nth trade_date INCLUSIVE OF entry, where N = horizon.
    Entry day is day 1. A 1d outcome (held open-to-close same session)
    exits on the entry day itself; a 5d outcome entered Mon exits Fri
    of the same week (assuming no holiday).

    The two branches operate on disjoint date ranges so they never
    disagree on the same day — clean handoff at the data boundary:

    In-range branch (entry within td_list, exit within td_list):
    walks the actual trade_date list from daily_features. Source of
    truth where data exists; weekends + holidays skipped exactly via
    the list. This branch's correctness is independent of
    valid_days — preserved as-is.

    Forward-projection branch (entry or exit past td_list end):
    walks the NYSE valid-days set from pandas_market_calendars,
    which knows floating holidays (MLK, Memorial, Labor,
    Thanksgiving, Good Friday). Falls back to weekend-only skipping
    if the library is unavailable."""
    if horizon <= 0:
        return None
    # 1d (entry day only) always returns the entry date itself, even if
    # entry isn't yet in daily_features (today before ingest).
    if horizon == 1:
        return entry.isoformat()

    if not td_list:
        d = entry
        for _ in range(horizon - 1):
            d = _next_business_day(d, valid_days)
        return d.isoformat()

    # Find the first index in td_list at or after entry. If entry is
    # past the end of the calendar, walk from entry directly.
    idx = next((i for i, x in enumerate(td_list) if x >= entry), None)
    if idx is None:
        d = entry
        for _ in range(horizon - 1):
            d = _next_business_day(d, valid_days)
        return d.isoformat()

    # Step forward (horizon - 1) trading days from idx (entry counts as
    # day 1, so we need horizon - 1 more steps).
    exit_idx = idx + (horizon - 1)
    if exit_idx < len(td_list):
        return td_list[exit_idx].isoformat()
    # Past the end: walk forward via the NYSE library.
    d = td_list[-1]
    steps_past_end = exit_idx - (len(td_list) - 1)
    for _ in range(steps_past_end):
        d = _next_business_day(d, valid_days)
    return d.isoformat()


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
    # NYSE valid-days from pandas_market_calendars for the forward-
    # projection branch (entry/exit past daily_features's end).
    # Library knows floating holidays + Good Friday + early closes.
    # In-range branch uses daily_features directly — unchanged.
    valid_days = _get_nyse_valid_days()
    out = []
    for r in rows:
        horizon = _parse_horizon(r["outcome"])
        td      = calendars.get(r["ticker"], [])
        exit_d  = _exit_date_for(td, r["entry_date"], horizon, valid_days)
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
        # ON CONFLICT must specify the partial index's WHERE predicate
        # — postgres only infers a partial unique index when the
        # conflict_target's predicate matches the index's predicate.
        # Without the WHERE clause here, postgres won't find an arbiter
        # even when the index exists, and throws InvalidColumnReferenceError.
        await conn.execute(
            """INSERT INTO oi_signal_calendar (ticker, outcome, entry_date)
               VALUES ($1, $2, $3)
               ON CONFLICT (ticker, outcome, entry_date)
               WHERE outcome IS NOT NULL DO NOTHING""",
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
