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
