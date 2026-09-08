"""Replay endpoints: read-only over the trade_quote parquet.

No database, no writes, no connection to the live service or the scan grid.
Every handler either returns data or returns `error` in words -- a missing
session must say so rather than draw an empty chart, because an empty chart is
indistinguishable from a session where nothing traded and telling those apart
is what the tool is for.
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, Query

from app import replay

log = logging.getLogger(__name__)

router = APIRouter(prefix="/api/replay", tags=["replay"])


def _fail(exc: Exception) -> dict:
    """A refusal the page can print.

    ReplayError carries a sentence written for a person; anything else is a
    fault and is logged with its traceback rather than shown raw.
    """
    if isinstance(exc, replay.ReplayError):
        return {"error": str(exc)}
    log.exception("replay failed")
    return {"error": f"{type(exc).__name__}: {exc}"}


@router.get("/sessions")
async def sessions():
    """Session dates, newest first, each with its symbol count.

    The count is not decoration: 2026-08-14 holds 96 symbols where the rest
    hold 634-739, and a bare date list would offer a day that looks identical
    and is mostly empty.
    """
    try:
        return {"sessions": replay.sessions(),
                "print_limit": replay.PRINT_LIMIT,
                "session_minutes": replay.SESSION_MINUTES}
    except Exception as exc:                              # noqa: BLE001
        return _fail(exc)


@router.get("/symbols")
async def symbols(date: str = Query(...)):
    try:
        return {"date": date, "symbols": replay.symbols_for(date)}
    except Exception as exc:                              # noqa: BLE001
        return _fail(exc)


@router.get("/candles")
async def candles(symbol: str = Query(...), date: str = Query(...)):
    """390 one-minute bars for the whole session. The overview."""
    try:
        return replay.candles(symbol, date)
    except Exception as exc:                              # noqa: BLE001
        return _fail(exc)


@router.get("/window")
async def window(symbol: str = Query(...), date: str = Query(...),
                 t0: float = Query(...), t1: float = Query(...),
                 nbbo: bool = Query(True)):
    """Every trade in [t0, t1) -- or a refusal, named.

    The MODE is decided server-side and returned explicitly. The client must
    not infer it from how much came back: "few trades" and "too many to send"
    look the same at the edge, and that edge is two minutes on NVDA.
    """
    try:
        if t1 <= t0:
            return {"error": f"t1 ({t1}) must be after t0 ({t0})"}
        return replay.window(symbol, date, t0, t1, nbbo=nbbo)
    except Exception as exc:                              # noqa: BLE001
        return _fail(exc)


@router.get("/status")
async def status():
    """What is cached, so a slow first zoom can be told from a slow box."""
    try:
        return {"cache": replay.cache_status(),
                "print_limit": replay.PRINT_LIMIT}
    except Exception as exc:                              # noqa: BLE001
        return _fail(exc)
