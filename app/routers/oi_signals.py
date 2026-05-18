"""OI Signals — trigger definitions, firing status, and position calendar."""
import math
import re
from datetime import date as _date, timedelta
from typing import Optional

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from app.db import get_pool, get_oi_pool

router = APIRouter(tags=["oi_signals"])

_DDL = """
CREATE TABLE IF NOT EXISTS oi_signal_triggers (
    id          SERIAL PRIMARY KEY,
    name        TEXT NOT NULL,
    ticker      TEXT NOT NULL,
    metric      TEXT NOT NULL,
    outcome     TEXT NOT NULL,
    min_val     DOUBLE PRECISION,
    max_val     DOUBLE PRECISION,
    color       TEXT DEFAULT '#3498db',
    created_at  TIMESTAMPTZ DEFAULT NOW()
);
-- Migrate REAL → DOUBLE PRECISION for any pre-existing tables. REAL stores
-- only ~7 decimal digits, so values like 1.08 get round-tripped to
-- 1.0800000429153442 on display. Widening the type is idempotent and
-- preserves existing values; rows already affected by REAL precision loss
-- need to be re-entered to fully recover, but new writes will be exact.
ALTER TABLE oi_signal_triggers
    ALTER COLUMN min_val TYPE DOUBLE PRECISION,
    ALTER COLUMN max_val TYPE DOUBLE PRECISION;
CREATE TABLE IF NOT EXISTS oi_signal_calendar (
    id          SERIAL PRIMARY KEY,
    trigger_id  INTEGER REFERENCES oi_signal_triggers(id) ON DELETE CASCADE,
    entry_date  DATE NOT NULL,
    added_at    TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(trigger_id, entry_date)
);
"""


def _parse_horizon(outcome: str) -> int:
    m = re.search(r'(\d+)', outcome)
    return int(m.group(1)) if m else 1


async def _ensure_tables(pool):
    async with pool.acquire() as conn:
        await conn.execute(_DDL)


class TriggerIn(BaseModel):
    name: str
    ticker: str
    metric: str
    outcome: str
    min_val: Optional[float] = None
    max_val: Optional[float] = None
    color: str = '#3498db'


class CalendarIn(BaseModel):
    trigger_id: int
    entry_date: str


@router.get("/triggers")
async def list_triggers(pool=Depends(get_pool)):
    await _ensure_tables(pool)
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT id, name, ticker, metric, outcome, min_val, max_val, color "
            "FROM oi_signal_triggers ORDER BY created_at")
    return [dict(r) for r in rows]


@router.post("/triggers")
async def create_trigger(body: TriggerIn, pool=Depends(get_pool)):
    await _ensure_tables(pool)
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "INSERT INTO oi_signal_triggers (name, ticker, metric, outcome, min_val, max_val, color) "
            "VALUES ($1,$2,$3,$4,$5,$6,$7) "
            "RETURNING id, name, ticker, metric, outcome, min_val, max_val, color",
            body.name, body.ticker, body.metric, body.outcome,
            body.min_val, body.max_val, body.color)
    return dict(row)


@router.put("/triggers/{tid}")
async def update_trigger(tid: int, body: TriggerIn, pool=Depends(get_pool)):
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "UPDATE oi_signal_triggers SET name=$1, ticker=$2, metric=$3, outcome=$4, "
            "min_val=$5, max_val=$6, color=$7 WHERE id=$8 "
            "RETURNING id, name, ticker, metric, outcome, min_val, max_val, color",
            body.name, body.ticker, body.metric, body.outcome,
            body.min_val, body.max_val, body.color, tid)
    if not row:
        raise HTTPException(status_code=404, detail="Trigger not found")
    return dict(row)


@router.delete("/triggers/{tid}")
async def delete_trigger(tid: int, pool=Depends(get_pool)):
    async with pool.acquire() as conn:
        await conn.execute("DELETE FROM oi_signal_triggers WHERE id=$1", tid)
    return {"ok": True}


@router.get("/firing")
async def get_firing(
    date: Optional[str] = Query(None),
    pool=Depends(get_pool),
    oi_pool=Depends(get_oi_pool),
):
    """
    For each trigger: check if it fires on the given date (defaults to latest available).
    Returns 20-bin distribution + today's bin for mini chart.
    """
    await _ensure_tables(pool)
    if not oi_pool:
        return {"error": "OI database not configured", "results": []}

    async with pool.acquire() as conn:
        triggers = await conn.fetch(
            "SELECT id, name, ticker, metric, outcome, min_val, max_val, color "
            "FROM oi_signal_triggers ORDER BY created_at")

    if not triggers:
        return {"date": date, "results": []}

    n_bins = 20
    results = []

    for t in triggers:
        ticker = t["ticker"]
        metric = t["metric"]
        outcome = t["outcome"]
        min_val = t["min_val"]
        max_val = t["max_val"]

        try:
            async with oi_pool.acquire() as conn:
                if date:
                    d = _date.fromisoformat(date)
                    rows = await conn.fetch(
                        f"SELECT {metric}, {outcome} FROM daily_features "
                        f"WHERE ticker=$1 AND {metric} IS NOT NULL AND {outcome} IS NOT NULL "
                        f"AND trade_date <= $2 ORDER BY trade_date",
                        ticker, d)
                    cur = await conn.fetchrow(
                        f"SELECT trade_date, {metric} FROM daily_features "
                        f"WHERE ticker=$1 AND {metric} IS NOT NULL AND trade_date <= $2 "
                        f"ORDER BY trade_date DESC LIMIT 1",
                        ticker, d)
                else:
                    rows = await conn.fetch(
                        f"SELECT {metric}, {outcome} FROM daily_features "
                        f"WHERE ticker=$1 AND {metric} IS NOT NULL AND {outcome} IS NOT NULL "
                        f"ORDER BY trade_date",
                        ticker)
                    cur = await conn.fetchrow(
                        f"SELECT trade_date, {metric} FROM daily_features "
                        f"WHERE ticker=$1 AND {metric} IS NOT NULL "
                        f"ORDER BY trade_date DESC LIMIT 1",
                        ticker)
        except Exception as e:
            results.append({"trigger": dict(t), "error": str(e), "firing": False})
            continue

        valid = []
        for r in rows:
            try:
                mv, ov = float(r[metric]), float(r[outcome])
                if not (math.isnan(mv) or math.isnan(ov)):
                    valid.append((mv, ov))
            except (ValueError, TypeError):
                continue

        if len(valid) < 20:
            results.append({
                "trigger": dict(t),
                "error": f"insufficient data (n={len(valid)})",
                "firing": False,
                "bins": [],
            })
            continue

        sorted_pairs = sorted(valid, key=lambda p: p[0])
        n = len(sorted_pairs)
        bins_data = [[] for _ in range(n_bins)]
        bin_mvals = [[] for _ in range(n_bins)]
        for i, (mv, ov) in enumerate(sorted_pairs):
            b = min(int(i / n * n_bins), n_bins - 1)
            bins_data[b].append(ov)
            bin_mvals[b].append(mv)

        bin_stats = []
        for i, (rets, vals) in enumerate(zip(bins_data, bin_mvals)):
            if rets:
                a = np.array(rets)
                bin_stats.append({
                    "bin": i + 1,
                    "n": len(rets),
                    "avg_ret": round(float(a.mean()) * 100, 3),
                    "win_rate": round(float((a > 0).mean()) * 100, 1),
                    "min_val": round(float(min(vals)), 6),
                    "max_val": round(float(max(vals)), 6),
                })
            else:
                bin_stats.append(None)

        current_val = None
        current_date = None
        today_bin = None
        if cur:
            try:
                cv = float(cur[metric])
                if not math.isnan(cv):
                    current_val = round(cv, 6)
                    current_date = str(cur["trade_date"])
                    all_sorted_vals = [p[0] for p in sorted_pairs]
                    rank = sum(1 for v in all_sorted_vals if v < current_val)
                    today_bin = min(int(rank / n * n_bins) + 1, n_bins)
            except (ValueError, TypeError):
                pass

        firing = False
        if current_val is not None:
            above_min = (min_val is None) or (current_val >= min_val)
            below_max = (max_val is None) or (current_val <= max_val)
            firing = above_min and below_max

        results.append({
            "trigger": dict(t),
            "current_val": current_val,
            "current_date": current_date,
            "today_bin": today_bin,
            "firing": firing,
            "n": n,
            "bins": bin_stats,
        })

    return {"date": date, "results": results}


@router.get("/calendar")
async def list_calendar(pool=Depends(get_pool), oi_pool=Depends(get_oi_pool)):
    """List calendar entries with computed exit dates."""
    await _ensure_tables(pool)
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """SELECT c.id, c.trigger_id, c.entry_date,
                      t.name, t.ticker, t.outcome, t.color
               FROM oi_signal_calendar c
               JOIN oi_signal_triggers t ON t.id = c.trigger_id
               ORDER BY c.entry_date""")

    if not rows:
        return []

    if not oi_pool:
        return [{**dict(r), "entry_date": str(r["entry_date"]), "exit_date": None} for r in rows]

    tickers = list({r["ticker"] for r in rows})
    td_by_ticker: dict = {}
    for ticker in tickers:
        async with oi_pool.acquire() as conn:
            td_rows = await conn.fetch(
                "SELECT DISTINCT trade_date FROM daily_features WHERE ticker=$1 "
                "ORDER BY trade_date", ticker)
        td_by_ticker[ticker] = [r["trade_date"] for r in td_rows]

    result = []
    for r in rows:
        ticker = r["ticker"]
        outcome = r["outcome"]
        entry_date = r["entry_date"]
        horizon = _parse_horizon(outcome)
        td = td_by_ticker.get(ticker, [])

        exit_date = None
        if td and horizon > 0:
            try:
                idx = next((i for i, x in enumerate(td) if x >= entry_date), None)
                if idx is not None:
                    exit_idx = idx + max(horizon - 1, 0)
                    if exit_idx < len(td):
                        exit_date = str(td[exit_idx])
                    else:
                        extra = exit_idx - len(td) + 1
                        exit_date = str(td[-1] + timedelta(days=int(extra * 1.4)))
            except Exception:
                pass

        result.append({**dict(r), "entry_date": str(entry_date), "exit_date": exit_date})

    return result


@router.post("/calendar")
async def add_calendar(body: CalendarIn, pool=Depends(get_pool)):
    await _ensure_tables(pool)
    entry = _date.fromisoformat(body.entry_date)
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "INSERT INTO oi_signal_calendar (trigger_id, entry_date) VALUES ($1,$2) "
            "ON CONFLICT (trigger_id, entry_date) DO NOTHING "
            "RETURNING id, trigger_id, entry_date",
            body.trigger_id, entry)
        if not row:
            row = await conn.fetchrow(
                "SELECT id, trigger_id, entry_date FROM oi_signal_calendar "
                "WHERE trigger_id=$1 AND entry_date=$2",
                body.trigger_id, entry)
    return {"id": row["id"], "trigger_id": row["trigger_id"], "entry_date": str(row["entry_date"])}


@router.delete("/calendar/{cid}")
async def delete_calendar(cid: int, pool=Depends(get_pool)):
    async with pool.acquire() as conn:
        await conn.execute("DELETE FROM oi_signal_calendar WHERE id=$1", cid)
    return {"ok": True}


# ── Monitored portfolio firings ─────────────────────────────────────────────


def _verdict_for(ticker_stats: dict) -> str:
    """Tiny heuristic so the firing card can flag obvious skips."""
    n  = ticker_stats.get("n", 0)
    wr = ticker_stats.get("win_rate", 0.0)
    ar = ticker_stats.get("avg_ret", 0.0)
    if n >= 100 and wr >= 0.55 and ar > 0:
        return "strong"
    if n < 30 or wr < 0.45 or ar < 0:
        return "weak"
    return "mixed"


@router.get("/firing-portfolios")
async def firing_portfolios(
    date: Optional[str] = Query(None),
    pool=Depends(get_pool),
    oi_pool=Depends(get_oi_pool),
):
    """For every portfolio with monitored=true, check each enabled system on
    each ticker against the most recent fully-resolved row (≤ `date`).
    Returns a flat list of firings with portfolio-context vs per-ticker stats."""
    if not oi_pool:
        return {"date": date, "results": []}

    # Late imports to avoid circular and keep this endpoint self-contained.
    from app.routers.oi_portfolios import _ensure_tables as _ensure_port_tables, _fetch_anchor_rows
    from app.routers.oi_analysis import _bin_for_value
    await _ensure_port_tables(pool)

    # 1) Load monitored portfolios + their enabled systems.
    async with pool.acquire() as conn:
        ports = await conn.fetch(
            """SELECT id, name, ticker, outcome, date_from, date_to
               FROM oi_research_portfolios
               WHERE monitored = TRUE
               ORDER BY name""")
        if not ports:
            return {"date": date, "results": []}
        sys_rows = await conn.fetch(
            """SELECT id, portfolio_id, name, enabled, position,
                      primary_metric, primary_bins, primary_bin_count, secondaries
               FROM oi_research_systems
               WHERE portfolio_id = ANY($1) AND enabled = TRUE
               ORDER BY portfolio_id, position, id""",
            [p["id"] for p in ports])

    # Group systems by portfolio.
    systems_by_pid: dict = {}
    import json as _json
    for r in sys_rows:
        d = dict(r)
        if isinstance(d.get("secondaries"), str):
            d["secondaries"] = _json.loads(d["secondaries"])
        systems_by_pid.setdefault(d["portfolio_id"], []).append(d)

    date_param = _date.fromisoformat(date) if date else None
    results = []

    for p in ports:
        pid     = p["id"]
        pname   = p["name"]
        ticker  = p["ticker"]
        outcome = p["outcome"]
        sys_list = systems_by_pid.get(pid, [])
        if not sys_list:
            continue

        # All metrics referenced anywhere in this portfolio's systems.
        needed = set()
        for s in sys_list:
            needed.add(s["primary_metric"])
            for sec in (s.get("secondaries") or []):
                needed.add(sec["metric"])

        # Historical rows for stats + binning (outcome NOT NULL). When the
        # user picks a past date, truncate the historical universe at that
        # date so today's bin is computed against the distribution that
        # existed AS OF that date — not against the future.
        effective_date_to = p["date_to"]
        if date_param:
            d_str = date_param.isoformat()
            if not effective_date_to or d_str < effective_date_to:
                effective_date_to = d_str
        anchor_rows = await _fetch_anchor_rows(
            oi_pool, ticker, outcome, p["date_from"], effective_date_to, sorted(needed))
        if not anchor_rows:
            continue

        # Group historical rows by ticker so per-ticker primary distributions
        # only see their own data (matches per-ticker rank normalisation).
        rows_by_tkr: dict = {}
        for r in anchor_rows:
            rows_by_tkr.setdefault(r.get("ticker", "_"), []).append(r)

        tickers_to_check = list(rows_by_tkr.keys()) if ticker == "ALL" else [ticker]

        # Latest "today" rows per ticker — most recent row with the primary
        # metric not null, on/before date_param. Independent of outcome
        # availability so we can fire even when 5-day forward isn't in yet.
        async with oi_pool.acquire() as conn:
            cols_today = ["ticker", "trade_date"] + sorted(needed)
            params, p_idx = [], 1
            where = [f"ticker = ANY(${p_idx})"]; params.append(tickers_to_check); p_idx += 1
            if date_param:
                where.append(f"trade_date <= ${p_idx}"); params.append(date_param); p_idx += 1
            today_sql = (
                f"SELECT DISTINCT ON (ticker) {', '.join(cols_today)} "
                f"FROM daily_features "
                f"WHERE {' AND '.join(where)} "
                f"ORDER BY ticker, trade_date DESC")
            today_rows = await conn.fetch(today_sql, *params)
        today_by_tkr = {r["ticker"]: dict(r) for r in today_rows}

        for s in sys_list:
            prim_metric = s["primary_metric"]
            prim_bins   = set(int(b) for b in (s["primary_bins"] or []))
            prim_count  = int(s["primary_bin_count"] or 20)
            secs        = s.get("secondaries") or []
            if not prim_bins or not secs:
                continue

            for tkr in tickers_to_check:
                hist = rows_by_tkr.get(tkr) or []
                if len(hist) < prim_count:
                    continue
                today = today_by_tkr.get(tkr)
                if not today:
                    continue

                # Today's primary bin against ticker history.
                prim_hist_sorted = sorted(
                    float(r[prim_metric]) for r in hist
                    if r.get(prim_metric) is not None)
                today_prim_val = today.get(prim_metric)
                today_prim_bin = _bin_for_value(today_prim_val, prim_hist_sorted, prim_count)
                if today_prim_bin is None or today_prim_bin not in prim_bins:
                    continue

                # Ticker's primary-filtered subset for secondary binning.
                hist_prim_subset = []
                for r in hist:
                    v = r.get(prim_metric)
                    if v is None:
                        continue
                    try:
                        fv = float(v)
                        if math.isnan(fv):
                            continue
                    except (TypeError, ValueError):
                        continue
                    b = _bin_for_value(fv, prim_hist_sorted, prim_count)
                    if b in prim_bins:
                        hist_prim_subset.append(r)
                if not hist_prim_subset:
                    continue

                # Walk each secondary; first one that fires marks the system
                # as firing for this ticker. Record values + bin info for all
                # so the card can show them.
                today_secs = []
                fires = False
                for sec in secs:
                    sm   = sec["metric"]
                    sbins = set(int(b) for b in (sec.get("bins") or []))
                    scnt  = int(sec.get("bin_count") or 10)
                    sec_hist_sorted = sorted(
                        float(r[sm]) for r in hist_prim_subset
                        if r.get(sm) is not None)
                    today_sec_val = today.get(sm)
                    today_sec_bin = _bin_for_value(today_sec_val, sec_hist_sorted, scnt)
                    in_selected = today_sec_bin is not None and today_sec_bin in sbins
                    if in_selected:
                        fires = True
                    today_secs.append({
                        "metric":      sm,
                        "value":       round(float(today_sec_val), 6) if today_sec_val is not None else None,
                        "bin":         today_sec_bin,
                        "bin_count":   scnt,
                        "selected_bins": sorted(list(sbins)),
                        "in_selected": in_selected,
                    })

                if not fires:
                    continue

                # ── Historical stats: this ticker, this system ────────────
                # Build the system's trade set restricted to this ticker:
                # primary in primary_bins AND any selected secondary bin
                # (computed within ticker's primary-filtered subset).
                rets_ticker = []
                # Pre-compute each secondary's per-subset bins for hist_prim_subset rows
                sec_hist_bins = []
                for sec in secs:
                    sm    = sec["metric"]
                    scnt  = int(sec.get("bin_count") or 10)
                    sbins = set(int(b) for b in (sec.get("bins") or []))
                    vals_for_bin = sorted(
                        float(r[sm]) for r in hist_prim_subset
                        if r.get(sm) is not None)
                    sec_hist_bins.append({
                        "metric": sm, "selected": sbins, "count": scnt,
                        "sorted": vals_for_bin,
                    })
                for r in hist_prim_subset:
                    # Check at least one secondary fires for this row.
                    any_fires = False
                    for shb in sec_hist_bins:
                        v = r.get(shb["metric"])
                        if v is None:
                            continue
                        b = _bin_for_value(v, shb["sorted"], shb["count"])
                        if b is not None and b in shb["selected"]:
                            any_fires = True
                            break
                    if not any_fires:
                        continue
                    yv = r.get(outcome)
                    if yv is None:
                        continue
                    try:
                        rets_ticker.append(float(yv))
                    except (TypeError, ValueError):
                        continue
                if not rets_ticker:
                    continue
                arr = np.array(rets_ticker)
                ticker_stats = {
                    "n":        int(len(arr)),
                    "win_rate": round(float((arr > 0).mean()), 4),
                    "avg_ret":  round(float(arr.mean()), 6),
                    "cum_ret":  round(float(arr.sum()), 6),
                }

                # ── ALL stats: same system across every ticker ─────────────
                rets_all = []
                for tkr_other, hist_other in rows_by_tkr.items():
                    if len(hist_other) < prim_count:
                        continue
                    prim_sorted_o = sorted(
                        float(rr[prim_metric]) for rr in hist_other
                        if rr.get(prim_metric) is not None)
                    prim_subset_o = []
                    for rr in hist_other:
                        v = rr.get(prim_metric)
                        if v is None: continue
                        try:
                            fv = float(v);
                            if math.isnan(fv): continue
                        except (TypeError, ValueError):
                            continue
                        b = _bin_for_value(fv, prim_sorted_o, prim_count)
                        if b in prim_bins:
                            prim_subset_o.append(rr)
                    # secondary bin within this ticker's primary subset
                    sec_o_bins = []
                    for sec in secs:
                        sm = sec["metric"]
                        scnt = int(sec.get("bin_count") or 10)
                        sbins = set(int(b) for b in (sec.get("bins") or []))
                        vals = sorted(
                            float(rr[sm]) for rr in prim_subset_o
                            if rr.get(sm) is not None)
                        sec_o_bins.append({"metric": sm, "selected": sbins,
                                           "count": scnt, "sorted": vals})
                    for rr in prim_subset_o:
                        ok = False
                        for shb in sec_o_bins:
                            v = rr.get(shb["metric"])
                            if v is None: continue
                            b = _bin_for_value(v, shb["sorted"], shb["count"])
                            if b is not None and b in shb["selected"]:
                                ok = True
                                break
                        if not ok:
                            continue
                        yv = rr.get(outcome)
                        if yv is None: continue
                        try:
                            rets_all.append(float(yv))
                        except (TypeError, ValueError):
                            continue
                if rets_all:
                    arr_all = np.array(rets_all)
                    all_stats = {
                        "n":        int(len(arr_all)),
                        "win_rate": round(float((arr_all > 0).mean()), 4),
                        "avg_ret":  round(float(arr_all.mean()), 6),
                        "cum_ret":  round(float(arr_all.sum()), 6),
                    }
                else:
                    all_stats = {"n": 0, "win_rate": 0.0, "avg_ret": 0.0, "cum_ret": 0.0}

                results.append({
                    "type":         "system",
                    "portfolio_id":   pid,
                    "portfolio_name": pname,
                    "system_id":      s["id"],
                    "system_name":    s["name"],
                    "ticker":         tkr,
                    "outcome":        outcome,
                    "firing":         True,
                    "today_date":     str(today.get("trade_date") or ""),
                    "today_primary": {
                        "metric":        prim_metric,
                        "value":         round(float(today_prim_val), 6) if today_prim_val is not None else None,
                        "bin":           today_prim_bin,
                        "bin_count":     prim_count,
                        "selected_bins": sorted(list(prim_bins)),
                    },
                    "today_secondaries": today_secs,
                    "all_stats":         all_stats,
                    "ticker_stats":      ticker_stats,
                    "verdict":           _verdict_for(ticker_stats),
                })

    return {"date": date, "results": results}
