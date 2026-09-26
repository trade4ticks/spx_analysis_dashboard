"""Strategy Signal: /api/strategy-signal.

GET  /sources               what a strategy can name (index series, surface
                            columns, operators, transforms, lookbacks)
GET  /board[?as_of=DATE]    every strategy, its decision and its charts --
                            the one call the page makes, and remakes every
                            five minutes. With as_of, the same logic run at
                            the CLOSE of that day: its weekday, its last
                            observations, charts ending there.
POST /strategies            create; PUT /strategies/{id} replace;
DELETE /strategies/{id};    POST /strategies/{id}/move {"direction": -1|1}

The board fetches each source ONCE for the deepest lookback any strategy
asks of it (library.Board), so ten strategies reading VIX cost one query.
A failure on one metric is reported on that metric and the strategy reads
NO DATA if it needed it; it never takes the page down.
"""
from __future__ import annotations

import logging
import math
from datetime import date

import pandas as pd

from fastapi import APIRouter, Body, Depends, HTTPException

from app.db import get_pool
from app.oo_backtest import market_calendar as cal
from app.strategy_signal import evaluate as ev, library as lib, store

log = logging.getLogger(__name__)

router = APIRouter(prefix="/api/strategy-signal", tags=["strategy-signal"])


@router.get("/sources")
async def sources(pool=Depends(get_pool)):
    cat = await lib.catalog(pool)
    return {**cat,
            "operators": list(lib.OPERATORS),
            "transforms": [{"id": k, "label": v["label"], "description": v["description"]}
                           for k, v in lib.TRANSFORMS.items()],
            "lookbacks": [{"id": k, "label": lib.LOOKBACK_LABELS[k], "sessions": n,
                           "intraday": k in lib.INTRADAY_LOOKBACKS} for k, n in lib.LOOKBACKS.items()],
            "cmps": list(ev.CMPS)}


def _num(v):
    return None if v is None or not math.isfinite(v) else round(float(v), 8)


WEEKDAY_FULL = {1: "Monday", 2: "Tuesday", 3: "Wednesday", 4: "Thursday", 5: "Friday",
                6: "Saturday", 7: "Sunday"}


def _when(as_of: str | None):
    """(the moment evaluated, live?). A past day is evaluated after its
    close, so its last bar is the session close and its data is not stale."""
    real = lib.now_et()
    if not as_of:
        return real, True
    try:
        d = date.fromisoformat(as_of)
    except ValueError:
        raise HTTPException(400, f"as_of must be a date (YYYY-MM-DD), not {as_of!r}.")
    if d > real.date():
        raise HTTPException(400, f"{as_of} is in the future.")
    if d == real.date():
        return real, True
    return pd.Timestamp(f"{d.isoformat()} 16:05", tz=lib.TZ), False


@router.get("/board")
async def board(as_of: str | None = None, pool=Depends(get_pool)):
    now, live = _when(as_of)
    strategies = await store.list_strategies(pool)
    b = lib.Board(now.date())
    for s in strategies:
        for m in s["metrics"]:
            b.need(m)
    await b.load(pool)
    expected = lib.expected_latest_session(now)

    out = []
    for s in strategies:
        values, metrics, charts = {}, [], {}
        for m in s["metrics"]:
            info = {"id": m["id"], "label": lib.metric_label(m), "value": None, "as_of": None,
                    "stale": False, "error": None}
            try:
                v, ts = b.latest(m)
                info["value"] = _num(v)
                if ts is not None:
                    info["as_of"] = lib.stamp(ts, "intraday")
                    info["stale"] = bool(expected and ts.date().isoformat() < expected)
                if m.get("chart"):
                    ser = b.series(m, m["resolution"], lib.LOOKBACKS[m["lookback"]])
                    charts[m["id"]] = {"t": [lib.stamp(t, m["resolution"]) for t in ser.index],
                                       "v": [_num(x) for x in ser.to_numpy()]}
            except Exception as exc:                          # noqa: BLE001 -- one metric, not the page
                log.warning("strategy-signal %r metric %s: %s", s["name"], m["id"], exc)
                info["error"] = str(exc)
                info["value"] = None
            values[m["id"]] = info["value"]
            metrics.append(info)
        decision = ev.decide(s, values, now.isoweekday())
        used = {m["id"] for m in s["metrics"] if m.get("signal")}
        out.append({"strategy": s, "decision": decision, "metrics": metrics, "charts": charts,
                    "stale": any(x["stale"] for x in metrics if x["id"] in used)})
    return {"now": now.strftime("%Y-%m-%d %H:%M"), "date": now.date().isoformat(), "live": live,
            "weekday": ev.WEEKDAY_NAMES[now.isoweekday()], "weekday_full": WEEKDAY_FULL[now.isoweekday()],
            "weekday_names": WEEKDAY_FULL, "is_session": cal.is_session(now.date()),
            "expected_session": expected, "strategies": out}


async def _clean(pool, body) -> dict:
    cat = await lib.catalog(pool)
    try:
        return store.clean_config(body, lib.valid_sources(cat))
    except ValueError as exc:
        raise HTTPException(400, str(exc))


@router.post("/strategies")
async def create(body: dict = Body(...), pool=Depends(get_pool)):
    cfg = await _clean(pool, body)
    try:
        return {"strategy": await store.create(pool, cfg)}
    except ValueError as exc:
        raise HTTPException(400, str(exc))


@router.put("/strategies/{sid}")
async def update(sid: int, body: dict = Body(...), pool=Depends(get_pool)):
    cfg = await _clean(pool, body)
    try:
        saved = await store.update(pool, sid, cfg)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    if saved is None:
        raise HTTPException(404, f"No strategy with id {sid}.")
    return {"strategy": saved}


@router.post("/strategies/{sid}/move")
async def move(sid: int, body: dict = Body(...), pool=Depends(get_pool)):
    if not await store.move(pool, sid, int(body.get("direction", 0) or 0)):
        raise HTTPException(404, f"No strategy with id {sid}.")
    return {"moved": sid}


@router.delete("/strategies/{sid}")
async def delete(sid: int, pool=Depends(get_pool)):
    if not await store.delete(pool, sid):
        raise HTTPException(404, f"No strategy with id {sid}.")
    return {"deleted": sid}
