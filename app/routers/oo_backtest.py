"""OO/Mesosim Backtest page — API, mounted at /api/oo-backtest.

Replaces the Plotly Dash app in Options-Backtest-Dashboard. The server's job
is narrow: parse a trade log, join market data, store and return saved logs.
It does NOT filter and does NOT build charts -- the whole log goes to the
browser once and every filter change is answered there (see payload.py).

Endpoints:
  GET  /registry       the metric registry, minDate filled from real coverage
  POST /parse          multipart trade log -> columnar trades payload, joined
                       to market data (app/oo_backtest/market.py)
  GET  /market-status  READ-ONLY freshness of main.index_ohlc: latest bar,
                       staleness, per-series coverage, close-fallback counts,
                       bar-label diagnostics
  GET    /strategies           saved strategies (no file content)
  POST   /strategies           multipart file + name + notes [+ replace]; 409
                               when the name exists and replace is not set
  GET    /strategies/{id}/load the saved file re-parsed and re-joined -- the
                               same payload as /parse, plus `saved`
  PUT    /strategies/{id}/capital  {"capital_per_position": number|null} --
                               the display input only; no re-parse
  DELETE /strategies/{id}
  GET    /surface/catalog      ranked surface metrics: catalog fields + min_date
                               (first non-null date, read from the table)
  POST   /surface/rank         {"trades": [[date, time, pnl], ...]} -> per metric
                               n, bars, Pearson/Spearman r + p + BH-adjusted p
  POST   /surface/values       {"column": name, "trades": [[date, time], ...]} ->
                               that metric at each trade's entry bar, in order

A saved strategy stores the ORIGINAL FILE, not joined trades, so a load gets
today's market data; see app/oo_backtest/store.py.

There is no write path to market data. index_ohlc is maintained by the
Thetadata_Raw_SPX pipeline on the VPS; a fetch here would be a second writer.
"""
from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime
from pathlib import PurePath
from zoneinfo import ZoneInfo

from fastapi import APIRouter, Body, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from app.db import get_pool
from app.oo_backtest import (data_loader, market, parsed_cache, store, surface,
                             surface_stats)
from app.oo_backtest.payload import assert_no_dropped_columns, trades_to_payload
from app.oo_backtest.registry import registry_with_coverage

log = logging.getLogger(__name__)
router = APIRouter()

_VENDOR_NAMES = {v: k for k, v in data_loader.COLUMN_MAPPING.items()}


@router.get("/registry")
async def get_registry(pool=Depends(get_pool)):
    try:
        daily, _ = await market.get_daily(pool)
        return {"metrics": registry_with_coverage(market.coverage(daily)), "coverage_error": None}
    except Exception as exc:  # noqa: BLE001 — the registry still serves, minDate unknown, and says why
        log.warning("oo-backtest registry coverage unavailable: %s", exc)
        return {"metrics": registry_with_coverage(None),
                "coverage_error": f"{type(exc).__name__}: {exc}"}


def _parse_df(content: bytes, filename: str):
    df = data_loader.parse_upload(content, filename)
    ok, msg = data_loader.validate_data(df)
    if not ok:
        raise ValueError(msg)
    return df


def _payload(df, filename: str, market_report: dict) -> dict:
    payload = trades_to_payload(df)
    assert_no_dropped_columns(payload)
    strategies = payload["columns"].get("strategy") or []
    named = next((s for s in strategies if s), None)
    # Mesosim's BacktestName ("allantis - v2: 5+4+1, PT SPX*.35, 60DIT, mon,
    # 2021-2026") says which variant this is; StrategyName ("allantis") does
    # not, and five variants of one strategy would all save under one name.
    named = payload["notes"].get("backtest_name") or named
    payload.update({
        "filename": filename,
        "source": "mesosim_json" if filename.lower().endswith(".json") else "oo_csv",
        "suggested_name": named or PurePath(filename).stem,
        "market": market_report,
    })
    return payload


def _parse(content: bytes, filename: str) -> dict:
    """Parse WITHOUT market data (used by checks that have no database)."""
    return _payload(_parse_df(content, filename), filename, {"joined": False, "error": "not requested"})


async def load_parsed(pool, meta: dict, content: bytes):
    """A saved strategy's parsed frame, from the cache when it is current.

    Shared with the Backtest Portfolio page, which loads several strategies
    at once: at ~3 s a parse, five of them is the difference between a page
    you adjust and one you wait for. A cache MISS still produces the right
    answer, just slowly, so nothing here raises on its own account -- the
    parse failure path below is the only one that refuses.
    """
    def _parse(c, n):
        try:
            return _parse_df(c, n)
        except Exception:
            log.exception("oo-backtest parse failed for saved %r", n)
            raise
    return await parsed_cache.load(
        pool, strategy_id=meta["id"], sha=meta["file_sha256"],
        content=content, filename=meta["filename"], parse=_parse)


async def _parse_or_400(content: bytes, name: str):
    """Parse, turning a bad file into a 400 that names the problem."""
    if not content:
        raise HTTPException(400, "The file is empty.")
    try:
        return await asyncio.to_thread(_parse_df, content, name)
    except (ValueError, KeyError, UnicodeDecodeError) as exc:
        # KeyError is what a CSV without "Date Opened" produces -- the loader
        # indexes the renamed column directly. Name the column as it appears
        # in the vendor's header, not as the loader renamed it.
        if isinstance(exc, KeyError):
            col = exc.args[0] if exc.args else ""
            vendor = _VENDOR_NAMES.get(col, col)
            detail = f'Missing column "{vendor}" — is this an Option Omega trade log?'
        elif isinstance(exc, json.JSONDecodeError):
            detail = f"{name} is not valid JSON ({exc.msg}, line {exc.lineno})."
        else:
            detail = str(exc)
        log.info("oo-backtest parse rejected %r: %s", name, detail)
        raise HTTPException(400, detail)
    except Exception as exc:  # noqa: BLE001 — logged with traceback, then surfaced
        log.exception("oo-backtest parse failed for %r", name)
        raise HTTPException(422, f"Could not parse {name}: {type(exc).__name__}: {exc}")


async def _analyze(content: bytes, name: str, pool, df=None) -> dict:
    """Parse + market join + payload: the ONE path both a fresh upload and a
    saved strategy's load go through, so the two cannot drift apart.

    `df` lets a caller supply a frame it already has -- a saved strategy's
    cached parse -- WITHOUT skipping the join or the payload build. Only the
    parse is ever served from cache; the market join runs every time, because
    `index_ohlc` is backfilled and a frozen join would pin a trade's VIX to
    whatever the table held the first time it was loaded.
    """
    if df is None:
        df = await _parse_or_400(content, name)

    # The trades are good even when market data is not. A failed join is
    # reported on the page -- the VIX and gap sections then show as skipped
    # with the reason -- rather than refusing a log that parsed.
    try:
        df, report = await market.join_market(pool, df)
    except Exception as exc:  # noqa: BLE001 — logged with traceback, surfaced in the payload
        log.exception("oo-backtest market join failed for %r", name)
        report = {"joined": False, "error": f"{type(exc).__name__}: {exc}"}
    try:
        return await asyncio.to_thread(_payload, df, name, report)
    except Exception as exc:  # noqa: BLE001
        log.exception("oo-backtest payload build failed for %r", name)
        raise HTTPException(422, f"Could not build the trade payload for {name}: {type(exc).__name__}: {exc}")


@router.post("/parse")
async def parse_trade_log(file: UploadFile = File(...), pool=Depends(get_pool)):
    return await _analyze(await file.read(), file.filename or "", pool)


# ── saved strategies ────────────────────────────────────────────────────────

def _source_of(filename: str) -> str:
    return "mesosim_json" if filename.lower().endswith(".json") else "oo_csv"


@router.get("/strategies")
async def list_saved(pool=Depends(get_pool)):
    return {"strategies": await store.list_strategies(pool)}


@router.post("/strategies")
async def save_saved(file: UploadFile = File(...), name: str = Form(...), notes: str = Form(""),
                     replace: bool = Form(False), capital_per_position: str = Form(""),
                     pool=Depends(get_pool)):
    filename = file.filename or ""
    content = await file.read()
    # The count and date range stored with the file come from the SERVER's
    # parse of these exact bytes, not from anything the page sends.
    df = await _parse_or_400(content, filename)
    dates_open = df["date_opened"].dropna()
    dates_close = df["date_closed"].dropna()
    try:
        saved = await store.save_strategy(
            pool, name=name, notes=notes, source=_source_of(filename), filename=filename, content=content,
            trade_count=int(len(df)),
            date_min=dates_open.min().date() if len(dates_open) else None,
            date_max=dates_close.max().date() if len(dates_close) else None,
            replace=replace, capital_per_position=capital_per_position)
    except store.NameTaken as exc:
        return JSONResponse(status_code=409, content={"detail": str(exc), "existing_id": exc.existing_id})
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    log.info("oo-backtest saved strategy %r (id %s, %d trades, replace=%s)",
             saved["name"], saved["id"], saved["trade_count"], replace)
    return {"strategy": saved}


@router.get("/strategies/{strategy_id}/load")
async def load_saved(strategy_id: int, pool=Depends(get_pool)):
    try:
        found = await store.load_strategy_file(pool, strategy_id)
    except ValueError as exc:
        log.error("oo-backtest saved strategy %s unreadable: %s", strategy_id, exc)
        raise HTTPException(500, str(exc))
    if found is None:
        raise HTTPException(404, f"No saved strategy with id {strategy_id}.")
    meta, content = found
    df, cache_note = await load_parsed(pool, meta, content)
    payload = await _analyze(content, meta["filename"], pool, df=df)
    payload["saved"] = meta
    payload["parse"] = cache_note
    # The stored facts were computed when it was saved; a parser change since
    # can move them. Say so rather than show two different trade counts.
    if payload["n"] != meta["trade_count"]:
        payload["saved_count_changed"] = {"when_saved": meta["trade_count"], "now": payload["n"]}
    return payload


@router.put("/strategies/{strategy_id}/capital")
async def set_saved_capital(strategy_id: int, body: dict = Body(...), pool=Depends(get_pool)):
    try:
        saved = await store.set_capital(pool, strategy_id, body.get("capital_per_position"))
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    if saved is None:
        raise HTTPException(404, f"No saved strategy with id {strategy_id}.")
    return {"strategy": saved}


@router.delete("/strategies/{strategy_id}")
async def delete_saved(strategy_id: int, pool=Depends(get_pool)):
    if not await store.delete_strategy(pool, strategy_id):
        raise HTTPException(404, f"No saved strategy with id {strategy_id}.")
    log.info("oo-backtest deleted saved strategy id %s", strategy_id)
    return {"ok": True}


@router.get("/market-status")
async def market_status(pool=Depends(get_pool)):
    try:
        daily, fresh = await market.get_daily(pool)
    except Exception as exc:  # noqa: BLE001
        log.warning("oo-backtest market-status failed: %s", exc)
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    today_et = datetime.now(ZoneInfo("America/New_York")).date()
    return {
        "ok": True,
        "source": "main.index_ohlc",
        **fresh,
        **market.staleness(fresh["latest_date"], today_et),
        "days": int(len(daily)),
        "coverage": market.coverage(daily),
        "close_fallback": market.fallback_report(daily),
        "bar_labels": market.bar_labels(),
        "sessions": market.sessions(),
    }


# ── surface metrics ─────────────────────────────────────────────────────────

def _check_bar_rule(body: dict) -> None:
    """The entry bar is fixed (surface.BAR_RULE; lookahead confirmed absent).
    A request naming another rule -- the removed previous_bar -- is refused
    rather than silently answered with the only rule there is."""
    rule = body.get("bar_rule")
    if rule is not None and rule != surface.BAR_RULE:
        raise HTTPException(400, f"bar_rule {rule!r} is not supported; the entry bar is {surface.BAR_RULE}.")


async def _surface_catalog(pool) -> dict:
    try:
        return await surface.get_catalog(pool)
    except Exception as exc:  # noqa: BLE001 — logged with traceback, surfaced by name
        log.exception("oo-backtest surface catalog failed")
        raise HTTPException(503, f"Surface metrics unavailable: {type(exc).__name__}: {exc}")


@router.get("/surface/catalog")
async def surface_catalog(pool=Depends(get_pool)):
    return await _surface_catalog(pool)


@router.post("/surface/rank")
async def surface_rank(body: dict = Body(...), pool=Depends(get_pool)):
    import time as _t
    _check_bar_rule(body)
    try:
        trades, parse_report = surface.parse_trades(body.get("trades"), with_pnl=True)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    cat = await _surface_catalog(pool)
    cols = [m["column_name"] for m in cat["metrics"]]
    bar_times, X, join_report = await surface.entry_matrix(pool, trades, cols)
    t0 = _t.monotonic()
    ids: dict = {}
    bar_ids = [ids.setdefault((t[0], bt), len(ids)) for t, bt in zip(trades, bar_times)]
    result = await asyncio.to_thread(surface_stats.rank, cat["metrics"], X, [t[2] for t in trades], bar_ids)
    compute_s = round(_t.monotonic() - t0, 3)
    log.info("oo-backtest surface rank: %d trades (%d with a bar), %d distinct entries, %d metrics; "
             "query %.2fs, stats %.3fs", len(trades), join_report["with_bar"], join_report["distinct_entries"],
             len(cols), join_report["query_s"], compute_s)
    return {"rows": result, "report": {**parse_report, **join_report, "metrics": len(cols), "compute_s": compute_s},
            "bar_rule": surface.BAR_RULE, "lookahead_confirmed": surface.LOOKAHEAD_CONFIRMED,
            "catalog_built_at": cat["built_at"]}


@router.post("/surface/values")
async def surface_values(body: dict = Body(...), pool=Depends(get_pool)):
    _check_bar_rule(body)
    column = body.get("column")
    cat = await _surface_catalog(pool)
    if column not in {m["column_name"] for m in cat["metrics"]}:
        raise HTTPException(400, f"Unknown or unranked surface metric: {column!r}.")
    try:
        trades, parse_report = surface.parse_trades(body.get("trades"), with_pnl=False)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    rows, join_report = await surface.entry_values(pool, trades, [column])
    return {"column": column, "values": [r[column] for r in rows],
            "bar_times": [r["bar_time"].strftime("%H:%M:%S") if r["bar_time"] else None for r in rows],
            "report": {**parse_report, **join_report}, "bar_rule": surface.BAR_RULE,
            "lookahead_confirmed": surface.LOOKAHEAD_CONFIRMED}
