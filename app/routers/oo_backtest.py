"""OO/Mesosim Backtest page — API, mounted at /api/oo-backtest.

Replaces the Plotly Dash app in Options-Backtest-Dashboard. The server's job
is narrow: parse a trade log, join market data, store and return saved logs.
It does NOT filter and does NOT build charts -- the whole log goes to the
browser once and every filter change is answered there (see payload.py).

Endpoints:
  GET  /registry       the metric registry, minDate filled from real coverage
  POST /parse          multipart trade log -> columnar trades payload, joined
                       to market data (app/oo_backtest/market.py)
  POST /parse-batch    multipart `files` (each may be gzipped by the page) ->
                       ONE payload for a parameter sweep: every trade tagged
                       with its variant, plus `sweep` (app/oo_backtest/sweep.py).
                       Refuses the whole batch, naming the file, when the
                       files do not form one sweep. /parse is this with one file.
  GET  /market-status  READ-ONLY freshness of main.index_ohlc: latest bar,
                       staleness, per-series coverage, close-fallback counts,
                       bar-label diagnostics
  GET    /strategies           saved strategies (no file content)
  POST   /strategies           multipart file + name + notes [+ replace]; 409
                               when the name exists and replace is not set
  POST   /strategies/sweep     multipart `files` (gzipped parts, as /parse-batch)
                               + name/notes/replace/capital: a parameter sweep
                               saved as ONE strategy holding every file; refused
                               as /parse-batch refuses; 409 on a taken name
  GET    /strategies/{id}/load the saved file re-parsed and re-joined -- the
                               same payload as /parse, plus `saved`. A sweep:
                               every file through /parse-batch's path, ALL
                               selected, and `saved_variants_changed` when the
                               files now resolve to other variants than when
                               it was saved
  PUT    /strategies/{id}/capital  {"capital_per_position": number|null} --
                               the display input only; no re-parse
  PUT    /strategies/{id}/notes    {"notes": text} -- the notes only; the
                               file, its facts and the list order are untouched
  DELETE /strategies/{id}
  GET    /surface/catalog      ranked surface metrics: catalog fields + min_date
                               (first non-null date, read from the table)
  POST   /surface/rank         {"bars": [[date, time], ...] (distinct), "bar_of":
                               [bar index per trade], "pnl": [per trade]} (or the
                               old {"trades": [[date, time, pnl], ...]}) -> per
                               metric n, bars, Pearson/Spearman r + p + BH p
  POST   /surface/values       {"column": name, "bars": [[date, time], ...]} ->
                               that metric at each bar, in order ("trades" is
                               accepted as the same list). Caps: distinct bars,
                               and trades for rank -- see surface.MAX_*

A saved strategy stores the ORIGINAL FILE, not joined trades, so a load gets
today's market data; see app/oo_backtest/store.py.

There is no write path to market data. index_ohlc is maintained by the
Thetadata_Raw_SPX pipeline on the VPS; a fetch here would be a second writer.
"""
from __future__ import annotations

import asyncio
import gzip
import json
import logging
import time as _time
import zlib
from datetime import date, datetime, time
from pathlib import PurePath
from zoneinfo import ZoneInfo

from fastapi import APIRouter, Body, Depends, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse, Response

from app.db import get_pool
from app.oo_backtest import (data_loader, market, parsed_cache, store, surface,
                             surface_stats, sweep)
from app.oo_backtest.payload import assert_no_dropped_columns, trades_to_payload
from app.oo_backtest.registry import registry_with_coverage

log = logging.getLogger(__name__)
router = APIRouter()

GZIP_MAGIC = bytes((0x1F, 0x8B))

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
        "source": _source_of(filename, df),
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


def _json_default(o):
    # What jsonable_encoder would have made of the types a payload holds (the
    # market report's dates); anything else is a bug, not something to stringify.
    if isinstance(o, (date, datetime, time)):
        return o.isoformat()
    if hasattr(o, "item"):
        return o.item()
    if isinstance(o, (set, frozenset)):
        return list(o)
    raise TypeError(f"{type(o).__name__} is not JSON serialisable")


try:
    import orjson as _orjson
except ImportError:          # speed only: the stdlib below gives the same JSON, ~10x slower
    _orjson = None
_ORJSON_WARNED = False


def encode_json(obj) -> bytes:
    """What Starlette's JSONResponse would send, without jsonable_encoder.

    orjson when installed: a 75-variant sweep's 85 MB encodes in 0.24 s
    against 2.3 s for the stdlib here (and the VPS is slower), on a request
    the tunnel ends at 100 s. Same JSON -- compact, UTF-8, dates as ISO,
    non-string keys as strings; check_oo_sweep compares it with JSONResponse.
    One difference: the stdlib REFUSES a NaN (allow_nan=False) where orjson
    writes null. The payload builder never emits one (payload._num), and the
    parser gates still encode with the stdlib's refusal."""
    global _ORJSON_WARNED
    if _orjson is not None:
        return _orjson.dumps(obj, default=_json_default,
                             option=_orjson.OPT_NON_STR_KEYS | _orjson.OPT_SERIALIZE_NUMPY)
    if not _ORJSON_WARNED:
        _ORJSON_WARNED = True
        log.warning("orjson is not installed: large OO payloads encode ~10x slower "
                    "(pip install 'orjson>=3.9')")
    return json.dumps(obj, ensure_ascii=False, allow_nan=False, separators=(",", ":"),
                      default=_json_default).encode("utf-8")


def json_gzip_response(obj, request: Request | None, *, what: str = "") -> Response:
    """A large payload, encoded and compressed HERE.

    A sweep's payload is ~90 MB of JSON. Returned as a dict, FastAPI walks
    every value through jsonable_encoder (5.8 s for 240,000 trades on the dev
    box) and GZipMiddleware compresses at level 9 (9.2 s), against a tunnel
    that ends the request at 100 s and a VPS about half this machine's speed.
    Level 5 is 0.9 s and ~4 MB. The middleware leaves a response that already
    has a Content-Encoding alone.
    """
    t0 = _time.monotonic()
    body = encode_json(obj)
    t1 = _time.monotonic()
    accepts = request is None or "gzip" in (request.headers.get("accept-encoding") or "").lower()
    if not accepts:
        return Response(body, media_type="application/json")
    z = gzip.compress(body, compresslevel=5)
    log.info("oo-backtest %s response: %.1f MB json in %.2fs, %.1f MB gzipped in %.2fs",
             what or "payload", len(body) / 1e6, t1 - t0, len(z) / 1e6, _time.monotonic() - t1)
    return Response(z, media_type="application/json", headers={"Content-Encoding": "gzip", "Vary": "Accept-Encoding"})


async def _read_upload(f: UploadFile, budget: int) -> tuple[str, bytes]:
    """A part's bytes, un-gzipped when the page compressed it (it does, so 75
    CSVs cross the tunnel as ~4 MB instead of ~40). The filename is the
    original's. Never decompresses past `budget`."""
    name = f.filename or ""
    raw = await f.read()
    if raw[:2] != GZIP_MAGIC:
        content = raw
    else:
        d = zlib.decompressobj(16 + zlib.MAX_WBITS)
        try:
            content = d.decompress(raw, budget + 1)
        except zlib.error as exc:
            raise HTTPException(400, f"{name}: could not be decompressed ({exc}).")
        if len(content) > budget or d.unconsumed_tail:
            raise _TooLarge(name)
    if len(content) > budget:
        raise _TooLarge(name)
    return name, content


class _TooLarge(Exception):
    pass


def _too_large(name: str) -> HTTPException:
    return HTTPException(400, f"The upload is larger than {sweep.MAX_BATCH_BYTES // 2**20} MB uncompressed "
                              f"(reached at {name}) -- it would not finish inside the tunnel's 100 s. "
                              f"Load fewer files at once.")


async def parse_batch(files: list[tuple[str, bytes]]):
    """(parsed, df, sweep): every file parsed, then combined into one sweep or
    refused with a 400 naming the file. The ONE place an upload, a sweep save
    and a saved sweep's load decide what the files are."""
    multi = len(files) > 1
    parsed = []
    for name, content in files:
        try:
            parsed.append((name, await _parse_or_400(content, name)))
        except HTTPException as exc:
            if multi and name not in str(exc.detail):
                exc.detail = f"{name}: {exc.detail}"
            if multi:
                exc.detail = f"{exc.detail} Nothing was loaded."
            raise
    try:
        df, sw = sweep.combine(parsed)
    except sweep.SweepError as exc:
        log.info("oo-backtest sweep refused: %s", exc)
        raise HTTPException(400, str(exc))
    return parsed, df, sw


async def analyze_batch(files: list[tuple[str, bytes]], pool) -> dict:
    """parse_batch, then the one market join and payload build every load goes
    through (_analyze)."""
    t0 = _time.monotonic()
    multi = len(files) > 1
    parsed, df, sw = await parse_batch(files)
    t1 = _time.monotonic()
    name0 = files[0][0]
    payload = await _analyze(files[0][1] if not multi else b"", name0, pool, df=df)
    t2 = _time.monotonic()
    if sw is not None:
        sources = {_source_of(n, f) for n, f in parsed}
        payload.update({
            "filename": name0 if not multi else f"{len(files)} files",
            "source": sources.pop() if len(sources) == 1 else "mixed",
            "suggested_name": sw["suggested_name"],
        })
        sw["timing"] = {"parse_s": round(t1 - t0, 2), "join_payload_s": round(t2 - t1, 2)}
    payload["sweep"] = sw
    log.info("oo-backtest batch: %d file(s), %d trades, %s variant(s); parse %.2fs, join+payload %.2fs",
             len(files), payload["n"], len(sw["variants"]) if sw else 1, t1 - t0, t2 - t1)
    return payload


@router.post("/parse")
async def parse_trade_log(request: Request, file: UploadFile = File(...), pool=Depends(get_pool)):
    try:
        name, content = await _read_upload(file, sweep.MAX_BATCH_BYTES)
    except _TooLarge as exc:
        raise _too_large(exc.args[0])
    return json_gzip_response(await analyze_batch([(name, content)], pool), request, what="parse")


@router.post("/parse-batch")
async def parse_batch_endpoint(request: Request, files: list[UploadFile] = File(...), pool=Depends(get_pool)):
    read = await _read_parts(files)
    return json_gzip_response(await analyze_batch(read, pool), request, what="batch")


# ── saved strategies ────────────────────────────────────────────────────────

def _source_of(filename: str, df=None) -> str:
    """'oo_csv' | 'oo_mcp_csv' | 'mesosim_json'. The parser says which (both
    Option Omega formats are .csv); a frame cached before it did falls back
    to the extension."""
    fmt = getattr(df, "attrs", {}).get("source_format") if df is not None else None
    return fmt or ("mesosim_json" if filename.lower().endswith(".json") else "oo_csv")


@router.get("/strategies")
async def list_saved(pool=Depends(get_pool)):
    return {"strategies": await store.list_strategies(pool)}


@router.post("/strategies")
async def save_saved(file: UploadFile = File(...), name: str = Form(...), notes: str = Form(""),
                     replace: bool = Form(False), capital_per_position: str = Form(""),
                     split_on: str = Form("false"), split_cutoff: str = Form(""),
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
            pool, name=name, notes=notes, source=_source_of(filename, df), filename=filename, content=content,
            trade_count=int(len(df)),
            date_min=dates_open.min().date() if len(dates_open) else None,
            date_max=dates_close.max().date() if len(dates_close) else None,
            replace=replace, capital_per_position=capital_per_position,
            split_on=split_on, split_cutoff=split_cutoff)
    except store.NameTaken as exc:
        return JSONResponse(status_code=409, content={"detail": str(exc), "existing_id": exc.existing_id})
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    log.info("oo-backtest saved strategy %r (id %s, %d trades, replace=%s)",
             saved["name"], saved["id"], saved["trade_count"], replace)
    return {"strategy": saved}


async def _read_parts(files: list[UploadFile]) -> list[tuple[str, bytes]]:
    """Every part, un-gzipped, against one budget for the whole upload."""
    if len(files) > sweep.MAX_FILES:
        raise HTTPException(400, f"{len(files)} files is more than the {sweep.MAX_FILES} one batch takes.")
    read, total = [], 0
    for f in files:
        try:
            name, content = await _read_upload(f, sweep.MAX_BATCH_BYTES - total)
        except _TooLarge as exc:
            raise _too_large(exc.args[0])
        total += len(content)
        read.append((name, content))
    return read


def recorded_sweep(sw: dict) -> dict:
    """What a saved sweep remembers about itself: the parameters, the variant
    list and each file's reading -- not the per-request timing."""
    return {k: v for k, v in sw.items() if k != "timing"}


@router.post("/strategies/sweep")
async def save_saved_sweep(files: list[UploadFile] = File(...), name: str = Form(...), notes: str = Form(""),
                           replace: bool = Form(False), capital_per_position: str = Form(""),
                           split_on: str = Form("false"), split_cutoff: str = Form(""),
                           pool=Depends(get_pool)):
    read = await _read_parts(files)
    if len(read) < 2:
        raise HTTPException(400, "A sweep is two or more files; save a single file as a normal strategy.")
    # The SERVER's reading of these bytes decides the variants and the facts.
    parsed, df, sw = await parse_batch(read)
    dates_open, dates_close = df["date_opened"].dropna(), df["date_closed"].dropna()
    sources = {_source_of(n, f) for n, f in parsed}
    try:
        saved = await store.save_sweep(
            pool, name=name, notes=notes, source=sources.pop() if len(sources) == 1 else "mixed", files=read,
            sweep=recorded_sweep(sw), trade_count=int(len(df)),
            date_min=dates_open.min().date() if len(dates_open) else None,
            date_max=dates_close.max().date() if len(dates_close) else None,
            replace=replace, capital_per_position=capital_per_position,
            split_on=split_on, split_cutoff=split_cutoff)
    except store.NameTaken as exc:
        return JSONResponse(status_code=409, content={"detail": str(exc), "existing_id": exc.existing_id})
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    log.info("oo-backtest saved sweep %r (id %s, %d files, %d variants, %d trades, replace=%s)",
             saved["name"], saved["id"], len(read), len(sw["variants"]), saved["trade_count"], replace)
    return {"strategy": saved}


async def _load_saved_sweep(pool, strategy_id: int) -> dict:
    try:
        found = await store.load_sweep_files(pool, strategy_id)
    except ValueError as exc:
        log.error("oo-backtest saved sweep %s unreadable: %s", strategy_id, exc)
        raise HTTPException(500, str(exc))
    if found is None:
        raise HTTPException(404, f"No saved strategy with id {strategy_id}.")
    meta, files, recorded = found
    payload = await analyze_batch(files, pool)
    payload["saved"] = meta
    payload["parse"] = {"source": "parse", "files": len(files)}
    payload["suggested_name"] = meta["name"]
    if payload["n"] != meta["trade_count"]:
        payload["saved_count_changed"] = {"when_saved": meta["trade_count"], "now": payload["n"]}
    # The files are re-read every load; the recorded list is only compared.
    # A parser change since the save can resolve them differently -- said, so
    # the page never quietly shows another set of variants under the old name.
    now = [(v["label"], v["n"]) for v in (payload.get("sweep") or {}).get("variants", [])]
    then = [(v["label"], v["n"]) for v in recorded.get("variants", [])]
    if now != then:
        payload["saved_variants_changed"] = {"when_saved": [f"{lab} ({n})" for lab, n in then],
                                             "now": [f"{lab} ({n})" for lab, n in now]}
    return payload


@router.get("/strategies/{strategy_id}/load")
async def load_saved(strategy_id: int, request: Request, pool=Depends(get_pool)):
    try:
        found = await store.load_strategy_file(pool, strategy_id)
    except store.SweepStrategy:
        return json_gzip_response(await _load_saved_sweep(pool, strategy_id), request, what="saved sweep")
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


@router.put("/strategies/{strategy_id}/split")
async def set_saved_split(strategy_id: int, body: dict = Body(...), pool=Depends(get_pool)):
    """The train/test split of a loaded strategy -- only from the page's
    explicit Save, never as the cutoff is moved."""
    try:
        saved = await store.set_split(pool, strategy_id, body.get("split_on"), body.get("split_cutoff"))
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    if saved is None:
        raise HTTPException(404, f"No saved strategy with id {strategy_id}.")
    return {"strategy": saved}


@router.put("/strategies/{strategy_id}/notes")
async def set_saved_notes(strategy_id: int, body: dict = Body(...), pool=Depends(get_pool)):
    try:
        saved = await store.set_notes(pool, strategy_id, body.get("notes"))
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
        trades, parse_report = surface.parse_rank_request(body)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    cat = await _surface_catalog(pool)
    cols = [m["column_name"] for m in cat["metrics"]]
    bar_times, distinct, gather, join_report = await surface.entry_matrix_gathered(pool, trades, cols)
    t0 = _t.monotonic()
    ids: dict = {}
    bar_ids = [ids.setdefault((t[0], bt), len(ids)) for t, bt in zip(trades, bar_times)]
    result = await asyncio.to_thread(surface_stats.rank, cat["metrics"], distinct, [t[2] for t in trades], bar_ids,
                                     gather)
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
        # Distinct entry bars from the page; "trades" (the old field) is the
        # same [[date, time], ...] list, and is held to the same bar cap.
        trades, parse_report = surface.parse_bars(body.get("bars", body.get("trades")), surface.MAX_VALUE_BARS)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    rows, join_report = await surface.entry_values(pool, trades, [column])
    return {"column": column, "values": [r[column] for r in rows],
            "bar_times": [r["bar_time"].strftime("%H:%M:%S") if r["bar_time"] else None for r in rows],
            "report": {**parse_report, **join_report}, "bar_rule": surface.BAR_RULE,
            "lookahead_confirmed": surface.LOOKAHEAD_CONFIRMED}
