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

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from app.db import get_pool
from app.oo_backtest import data_loader, market
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


@router.post("/parse")
async def parse_trade_log(file: UploadFile = File(...), pool=Depends(get_pool)):
    name = file.filename or ""
    content = await file.read()
    if not content:
        raise HTTPException(400, "The file is empty.")
    try:
        df = await asyncio.to_thread(_parse_df, content, name)
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


@router.get("/market-status")
async def market_status(pool=Depends(get_pool)):
    try:
        daily, fresh = await market.get_daily(pool)
    except Exception as exc:  # noqa: BLE001
        log.warning("oo-backtest market-status failed: %s", exc)
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    now_et = datetime.now(ZoneInfo("America/New_York")).replace(tzinfo=None)
    expected = market.expected_last_session(now_et)
    latest = fresh["latest_date"]
    return {
        "ok": True,
        "source": "main.index_ohlc",
        **fresh,
        "expected_session": expected.isoformat(),
        "stale": bool(latest is None or latest < expected.isoformat()),
        "days": int(len(daily)),
        "coverage": market.coverage(daily),
        "close_fallback": market.fallback_report(daily),
        "bar_labels": market.bar_labels(),
    }
