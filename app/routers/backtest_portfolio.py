"""Backtest Portfolio — several saved strategies, combined.

P1: LOADING ONLY. This phase proves the data path — pick saved strategies,
load them, hold their trades in the browser — and deliberately computes
nothing. The stats, the curves and the correlation work land in P2-P4, on the
definitions the single-backtest page already uses.

WHAT IT SHARES WITH /oo-backtest, rather than reimplements:

  * the PARSE, its validation, the market join and the payload whitelist —
    `oo_backtest._analyze`, entered with a cached frame. A second copy of
    that path would be a second set of trade numbers for the same file.
  * the saved-strategy store, including `capital_per_position`, which seeds
    each strategy's row here.
  * the metric registry with coverage, which will drive the per-strategy
    filters in P2 — the old Dash app kept five parallel dicts of metrics and
    this page will not.

WHAT IS ITS OWN: how several strategies are combined. Decided with the brief
and settled:

  * qty SCALES P/L linearly (2x qty = 2x P/L). Applied in the browser, where
    every other recomputation on these pages happens; nothing server-side
    multiplies a P/L.
  * capital per strategy seeds from the saved value and can be overridden
    here; portfolio capital is the SUM of the per-strategy planned capital.
  * the portfolio's date range defaults to the UNION of the loaded spans,
    with intersection available.
  * P/L is dated by CLOSE, exactly as the single-backtest page does it, so
    the same strategy reads the same on both pages.
"""
from __future__ import annotations

import asyncio
import logging
import time

from fastapi import APIRouter, Body, Depends, HTTPException

from app.db import get_pool
from app.oo_backtest import market, store
from app.oo_backtest.registry import registry_with_coverage
from app.routers.oo_backtest import _analyze, load_parsed

log = logging.getLogger(__name__)

router = APIRouter(prefix="/api/backtest-portfolio", tags=["backtest-portfolio"])

# How many strategies one load may carry. Not a storage limit -- it is the
# point past which the page is no longer a portfolio anyone is reading, and a
# request that would take a minute should be refused with a reason rather
# than served slowly.
MAX_STRATEGIES = 12

# The strategy colours, in order. Eight because the old app had eight and the
# palette is the page's, not the data's; a ninth strategy reuses the first
# colour rather than generating one, because a generated hue is not
# distinguishable from the others by eye anyway.
COLORS = ["#3498db", "#e84393", "#2ecc71", "#f39c12",
          "#9b59b6", "#1abc9c", "#e67e22", "#7f8c8d"]


@router.get("/registry")
async def get_registry(pool=Depends(get_pool)):
    """The same registry the single-backtest page filters on."""
    try:
        daily, _ = await market.get_daily(pool)
        return {"metrics": registry_with_coverage(market.coverage(daily)),
                "coverage_error": None}
    except Exception as exc:                              # noqa: BLE001
        log.warning("portfolio registry coverage unavailable: %s", exc)
        return {"metrics": registry_with_coverage(None),
                "coverage_error": f"{type(exc).__name__}: {exc}"}


@router.get("/strategies")
async def list_saved(pool=Depends(get_pool)):
    """Everything saved on the single-backtest page, with its capital."""
    return {"strategies": await store.list_strategies(pool),
            "colors": COLORS, "max": MAX_STRATEGIES}


@router.post("/load")
async def load_portfolio(body: dict = Body(...), pool=Depends(get_pool)):
    """{"ids": [1, 2, 3]} -> each strategy's trades, ready for the browser.

    SEQUENTIAL, not gathered. A cache hit is a few hundred milliseconds and a
    miss is seconds of CPU in a thread; firing five misses at once would put
    five parses on the event loop's thread pool and make every other page on
    this process wait behind them. The page reports what the load cost, so a
    slow one is visible rather than mysterious.
    """
    ids = body.get("ids")
    if not isinstance(ids, list) or not ids:
        raise HTTPException(400, "Pass a list of saved strategy ids.")
    try:
        ids = [int(i) for i in ids]
    except (TypeError, ValueError):
        raise HTTPException(400, "Strategy ids must be integers.")
    if len(set(ids)) != len(ids):
        raise HTTPException(400, "The same strategy was listed twice.")
    if len(ids) > MAX_STRATEGIES:
        raise HTTPException(
            400, f"{len(ids)} strategies asked for; {MAX_STRATEGIES} is the "
                 f"most this page loads at once.")

    t0 = time.perf_counter()
    out = []
    for n, sid in enumerate(ids):
        try:
            found = await store.load_strategy_file(pool, sid)
        except ValueError as exc:                         # sha mismatch
            log.error("portfolio: saved strategy %s unreadable: %s", sid, exc)
            raise HTTPException(500, str(exc))
        if found is None:
            raise HTTPException(404, f"No saved strategy with id {sid}.")
        meta, content = found
        df, note = await load_parsed(pool, meta, content)
        payload = await _analyze(content, meta["filename"], pool, df=df)
        payload["saved"] = meta
        payload["parse"] = note
        payload["color"] = COLORS[n % len(COLORS)]
        # The saved capital seeds the row; the page may override it, and the
        # override is the page's until it is saved with a profile (P5).
        payload["capital_per_position"] = meta.get("capital_per_position")
        payload["qty"] = 1
        # A parser change since the save moves the count. Said out loud here
        # for the same reason the single page says it: two different trade
        # counts for one file is not something to discover later.
        if payload["n"] != meta["trade_count"]:
            payload["saved_count_changed"] = {"when_saved": meta["trade_count"],
                                              "now": payload["n"]}
        out.append(payload)

    total = time.perf_counter() - t0
    cached = sum(1 for p in out if p["parse"]["source"] == "cache")
    log.info("portfolio load: %d strategies in %.1fs (%d from cache)",
             len(out), total, cached)
    return {
        "strategies": out,
        "load": {"seconds": total, "n": len(out), "from_cache": cached,
                 "parsed": len(out) - cached},
    }
