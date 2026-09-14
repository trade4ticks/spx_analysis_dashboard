"""OO/Mesosim Backtest page — API, mounted at /api/oo-backtest.

Replaces the Plotly Dash app in Options-Backtest-Dashboard. The server's job
is narrow: parse a trade log, join market data, store and return saved logs.
It does NOT filter and does NOT build charts -- the whole log goes to the
browser once and every filter change is answered there (see payload.py).

Endpoints (Phase 1):
  GET  /registry       the metric registry (app/oo_backtest/registry.py)
  POST /parse          multipart trade log -> columnar trades payload
  GET  /market-status  where SPX/VIX daily OHLC lives, and how fresh it is

/market-status is a PROBE in this phase. The brief says the market data is
already on the VPS but does not say which table, and the source app's table
(bt_market_data: date, ticker, open/high/low/close) is not one this dashboard
reads today. Rather than guess, the endpoint reports every candidate it can
see, and Phase 2 wires the join to whichever one it names.
"""
from __future__ import annotations

import asyncio
import json
import logging
from pathlib import PurePath

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from app.db import get_oi_pool, get_pool
from app.oo_backtest import data_loader
from app.oo_backtest.payload import assert_no_dropped_columns, trades_to_payload
from app.oo_backtest.registry import REGISTRY

log = logging.getLogger(__name__)
router = APIRouter()

# The old app's table first. Its ticker names were the config keys, not the
# yfinance symbols, but both spellings are probed so a differently-loaded
# copy is still found.
_CANDIDATE_TABLES = ("bt_market_data", "market_data", "index_ohlc", "underlying_ohlc")
_TICKERS = ["SPX", "^GSPC", "$SPX", "SPY", "VIX", "^VIX", "VIX3M", "^VIX3M", "VIX9D", "^VIX9D"]

_VENDOR_NAMES = {v: k for k, v in data_loader.COLUMN_MAPPING.items()}


@router.get("/registry")
async def get_registry():
    return {"metrics": REGISTRY}


def _parse(content: bytes, filename: str) -> dict:
    df = data_loader.parse_upload(content, filename)
    ok, msg = data_loader.validate_data(df)
    if not ok:
        raise ValueError(msg)
    payload = trades_to_payload(df)
    assert_no_dropped_columns(payload)
    strategies = payload["columns"].get("strategy") or []
    named = next((s for s in strategies if s), None)
    payload.update({
        "filename": filename,
        "source": "mesosim_json" if filename.lower().endswith(".json") else "oo_csv",
        "suggested_name": named or PurePath(filename).stem,
        "market_joined": False,
    })
    return payload


@router.post("/parse")
async def parse_trade_log(file: UploadFile = File(...)):
    name = file.filename or ""
    content = await file.read()
    if not content:
        raise HTTPException(400, "The file is empty.")
    try:
        return await asyncio.to_thread(_parse, content, name)
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


async def _probe_pool(pool, db_label: str) -> list[dict]:
    out: list[dict] = []
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT table_name, column_name, data_type FROM information_schema.columns "
            "WHERE table_schema = 'public' AND table_name = ANY($1::text[]) "
            "ORDER BY table_name, ordinal_position",
            list(_CANDIDATE_TABLES),
        )
        tables: dict[str, list] = {}
        for r in rows:
            tables.setdefault(r["table_name"], []).append(f'{r["column_name"]}:{r["data_type"]}')

        for table, columns in tables.items():
            entry = {"db": db_label, "table": table, "columns": columns, "coverage": None, "error": None}
            names = {c.split(":")[0] for c in columns}
            date_col = "date" if "date" in names else ("trade_date" if "trade_date" in names else None)
            try:
                if "ticker" in names and date_col:
                    # Identifiers are from information_schema above, never user input.
                    cov = await conn.fetch(
                        f"SELECT ticker, COUNT(*) AS n, MIN({date_col})::text AS first, "
                        f"MAX({date_col})::text AS last FROM {table} "
                        f"WHERE ticker = ANY($1::text[]) GROUP BY ticker ORDER BY ticker",
                        _TICKERS,
                    )
                    entry["coverage"] = [dict(r) for r in cov]
                elif date_col:
                    cov = await conn.fetchrow(
                        f"SELECT MIN({date_col})::text AS first, MAX({date_col})::text AS last FROM {table}")
                    entry["coverage"] = [dict(cov)] if cov else []
            except Exception as exc:  # noqa: BLE001 — reported in the payload and logged
                log.warning("oo-backtest market probe %s.%s failed: %s", db_label, table, exc)
                entry["error"] = f"{type(exc).__name__}: {exc}"
            out.append(entry)
    return out


@router.get("/market-status")
async def market_status(pool=Depends(get_pool), oi_pool=Depends(get_oi_pool)):
    result = {"phase": 1, "joined": False, "candidates": [], "databases": [], "errors": []}
    try:
        async with pool.acquire() as conn:
            dbs = await conn.fetch("SELECT datname FROM pg_database WHERE NOT datistemplate ORDER BY 1")
            result["databases"] = [r["datname"] for r in dbs]
    except Exception as exc:  # noqa: BLE001
        log.warning("oo-backtest pg_database probe failed: %s", exc)
        result["errors"].append(f"pg_database: {type(exc).__name__}: {exc}")

    for label, p in (("main", pool), ("open_interest", oi_pool)):
        if p is None:
            result["errors"].append(f"{label}: pool not configured")
            continue
        try:
            result["candidates"].extend(await _probe_pool(p, label))
        except Exception as exc:  # noqa: BLE001
            log.warning("oo-backtest probe of %s failed: %s", label, exc)
            result["errors"].append(f"{label}: {type(exc).__name__}: {exc}")
    return result
