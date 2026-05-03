"""Research 2 API endpoints — orchestration-driven research."""
import asyncio
import json
import logging
import traceback
from datetime import date as _date
from typing import List, Optional

log = logging.getLogger(__name__)

import anthropic
from fastapi import APIRouter, Depends, BackgroundTasks, HTTPException, UploadFile, File, Form
from fastapi.responses import Response
from pydantic import BaseModel

from app.db import get_pool, get_oi_pool
from research import db as rdb
from research import orchestrator
from research import export as rexport
from research.engine import format_corr_table
from research.pnl import parse_pnl_csv
from research import backtest as rbacktest

router = APIRouter()

_OI_TABLES = {"daily_features", "option_oi_surface", "underlying_ohlc"}
_NUMERIC_TYPES = {
    "double precision", "numeric", "real", "integer",
    "bigint", "smallint", "decimal",
}
_EXCLUDE_COLS = {"id", "ticker", "trade_date", "created_at", "updated_at"}


def _extract_prose(ai_summary: str | None) -> str:
    """Extract markdown prose from ai_summary (which may be JSON sections or plain text)."""
    if not ai_summary:
        return "(none)"
    try:
        sections = json.loads(ai_summary)
        if isinstance(sections, list):
            return "\n\n".join(
                s.get("content", "") for s in sections if s.get("type") == "markdown"
            )
    except (json.JSONDecodeError, TypeError):
        pass
    return ai_summary


# ── Request model ─────────────────────────────────────────────────────────────

class RunRequest(BaseModel):
    name: str
    question: str
    table: str = "daily_features"
    tickers: list[str] = []
    date_from: Optional[str] = None
    date_to: Optional[str] = None
    model: str = "claude-sonnet-4-6"
    pnl_upload_id:      Optional[str] = None
    backtest_upload_id: Optional[str] = None


# ── Helpers ───────────────────────────────────────────────────────────────────

async def _get_columns(table: str, pool, oi_pool) -> list[dict]:
    use_pool = oi_pool if table in _OI_TABLES else pool
    if use_pool is None:
        return []
    async with use_pool.acquire() as conn:
        rows = await conn.fetch(
            """SELECT column_name, data_type
               FROM information_schema.columns
               WHERE table_name = $1 AND table_schema = 'public'
               ORDER BY ordinal_position""",
            table,
        )
    return [
        {"name": r["column_name"], "type": r["data_type"]}
        for r in rows
        if r["data_type"] in _NUMERIC_TYPES and r["column_name"] not in _EXCLUDE_COLS
    ]


async def _load_all_tickers(oi_pool, table: str) -> list[str]:
    """Load all distinct tickers from the OI database."""
    if not oi_pool or table not in _OI_TABLES:
        return []
    try:
        async with oi_pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT DISTINCT ticker FROM daily_features ORDER BY ticker")
        return [r["ticker"] for r in rows]
    except Exception:
        return []


# ── Background execution ──────────────────────────────────────────────────────

async def _execute_run(main_pool, oi_pool, run_id: str, question: str,
                       plan: dict, config: dict, model: str):
    logs: list[str] = []

    def _log(msg: str):
        logs.append(msg)

    # Load knowledge rules for prompt injection
    knowledge_rules = await _load_active_rules(main_pool)

    try:
        summary = await orchestrator.execute_v2_pipeline(
            main_pool=main_pool,
            oi_pool=oi_pool,
            run_id=run_id,
            question=question,
            plan=plan,
            config=config,
            model=model,
            knowledge_rules=knowledge_rules,
            log=_log,
        )
        # Always save pipeline log for diagnostics
        log_text = "\n".join(logs) if logs else None
        async with main_pool.acquire() as conn:
            await rdb.update_run(conn, run_id, status="complete",
                                 ai_summary=summary, error_msg=log_text)
    except Exception as exc:
        log_tail = "\n".join(logs[-30:]) if logs else ""
        err = f"{exc}"
        if log_tail:
            err = f"{err}\n\n--- Execution log ---\n{log_tail}"
        async with main_pool.acquire() as conn:
            await rdb.update_run(conn, run_id, status="error", error_msg=err)


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.get("/runs")
async def list_runs(pool=Depends(get_pool)):
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """SELECT id::text, name, question, status, created_at, completed_at, config
               FROM research_runs
               WHERE config->>'engine' = 'v2'
               ORDER BY created_at DESC LIMIT 50""",
        )
    result = []
    for r in rows:
        d = dict(r)
        cfg = d.get("config")
        d["config"] = json.loads(cfg) if isinstance(cfg, str) else cfg
        result.append(d)
    return result


@router.get("/run/{run_id}")
async def get_run(run_id: str, pool=Depends(get_pool)):
    async with pool.acquire() as conn:
        run = await rdb.load_run(conn, run_id)
        if not run:
            raise HTTPException(404, "Run not found")
        chart_count = await conn.fetchval(
            "SELECT COUNT(*) FROM research_charts WHERE run_id = $1::uuid", run_id)
    cfg = run.get("config")
    if isinstance(cfg, str):
        run = {**run, "config": json.loads(cfg)}
    return {**run, "chart_count": int(chart_count)}


@router.get("/run/{run_id}/charts")
async def get_chart_list(run_id: str, pool=Depends(get_pool)):
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """SELECT id::text, ticker, x_col, y_col, chart_type, title, created_at
               FROM research_charts WHERE run_id = $1::uuid
               ORDER BY created_at""",
            run_id,
        )
    return [dict(r) for r in rows]


@router.delete("/run/{run_id}")
async def delete_run(run_id: str, pool=Depends(get_pool)):
    async with pool.acquire() as conn:
        run = await rdb.load_run(conn, run_id)
        if not run:
            raise HTTPException(404, "Run not found")
        await conn.execute("DELETE FROM research_runs WHERE id = $1::uuid", run_id)
    return {"deleted": run_id}


@router.get("/run/{run_id}/pdf")
async def download_pdf(run_id: str, pool=Depends(get_pool)):
    async with pool.acquire() as conn:
        run = await rdb.load_run(conn, run_id)
        if not run:
            raise HTTPException(404, "Run not found")
        results = await rdb.load_results(conn, run_id)
        # Research 2 embeds charts in ai_summary; research_charts table is not used
        charts = []

    for r in results:
        if isinstance(r.get("result"), str):
            r["result"] = json.loads(r["result"])

    try:
        pdf_bytes = rexport.build_pdf_bytes(run, results, charts)
    except Exception as exc:
        log.error("PDF build failed for run %s:\n%s", run_id, traceback.format_exc())
        raise HTTPException(500, f"PDF generation failed: {exc}") from exc
    safe_name = run["name"].replace(" ", "_").replace("/", "-")
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{safe_name}.pdf"'},
    )


@router.post("/run")
async def start_run(req: RunRequest, background_tasks: BackgroundTasks,
                    pool=Depends(get_pool), oi_pool=Depends(get_oi_pool)):
    # ── P&L agentic mode: bypass column lookup + classification ──────────
    if req.pnl_upload_id:
        await _ensure_pnl_table(pool)
        async with pool.acquire() as conn:
            upload = await conn.fetchrow(
                "SELECT id::text, date_from, date_to FROM research_pnl_uploads "
                "WHERE id = $1::uuid", req.pnl_upload_id)
        if upload is None:
            raise HTTPException(400, f"P&L upload '{req.pnl_upload_id}' not found")

        date_from = req.date_from or (str(upload["date_from"]) if upload["date_from"] else None)
        date_to   = req.date_to   or (str(upload["date_to"])   if upload["date_to"]   else None)
        if not date_from or not date_to:
            raise HTTPException(400, "Could not determine date range from upload — specify date_from/date_to")

        plan = {
            "task_type":      "pnl-iv-correlation",
            "task_reasoning": "Agentic P&L–IV analysis: Claude will explore surface_metrics_core columns vs uploaded P&L data.",
            "hypotheses":     [],
            "feature_columns": [],
            "outcome_columns": ["pnl"],
            "scan_focus":     "IV metrics correlated with P&L",
            "report_guidance": "Report on IV–P&L correlations. Cite specific r values and patterns.",
        }
        config = {
            "engine":        "v2",
            "pnl_upload_id": req.pnl_upload_id,
            "date_from":     date_from,
            "date_to":       date_to,
            "model":         req.model,
            "workflow_plan": plan,
        }
        async with pool.acquire() as conn:
            run_id = await rdb.create_run(conn, req.name, req.question, config)
        background_tasks.add_task(
            _execute_run, pool, oi_pool, run_id, req.question, plan, config, req.model,
        )
        return {"run_id": run_id, "plan": plan}

    # ── Backtest agentic mode ─────────────────────────────────────────────
    if req.backtest_upload_id:
        await _ensure_backtest_table(pool)
        async with pool.acquire() as conn:
            upload = await conn.fetchrow(
                "SELECT id::text, date_from, date_to, trade_count, matched_count "
                "FROM research_backtest_uploads WHERE id = $1::uuid",
                req.backtest_upload_id,
            )
        if upload is None:
            raise HTTPException(400, f"Backtest upload '{req.backtest_upload_id}' not found")

        plan = {
            "task_type":       "backtest-regime-analysis",
            "task_reasoning":  "Agentic backtest analysis: Claude will explore which entry-time IV conditions predict trade outcomes.",
            "hypotheses":      [],
            "feature_columns": [],
            "outcome_columns": ["pnl", "is_win"],
            "scan_focus":      "Entry-morning IV conditions predictive of trade P&L and win rate",
            "report_guidance": "Report on which IV conditions at trade entry (09:35) predict better outcomes. Cite win rates, mean P&L by bucket, and time-split stability.",
        }
        config = {
            "engine":             "v2",
            "backtest_upload_id": req.backtest_upload_id,
            "date_from":          str(upload["date_from"]) if upload["date_from"] else None,
            "date_to":            str(upload["date_to"])   if upload["date_to"]   else None,
            "model":              req.model,
            "workflow_plan":      plan,
        }
        async with pool.acquire() as conn:
            run_id = await rdb.create_run(conn, req.name, req.question, config)
        background_tasks.add_task(
            _execute_run, pool, oi_pool, run_id, req.question, plan, config, req.model,
        )
        return {"run_id": run_id, "plan": plan}

    # Fetch available columns for the requested table
    available_cols = await _get_columns(req.table, pool, oi_pool)
    if not available_cols:
        raise HTTPException(400, f"No numeric columns found in table '{req.table}'")

    # Classify and plan — synchronous, happens before background task so the
    # client gets the plan immediately in the response
    knowledge_rules = await _load_active_rules(pool)

    try:
        plan = await orchestrator.classify_and_plan(
            question=req.question,
            available_columns=available_cols,
            tickers=req.tickers,
            table=req.table,
            date_from=req.date_from,
            date_to=req.date_to,
            model=req.model,
            knowledge_rules=knowledge_rules,
        )
    except Exception as exc:
        raise HTTPException(500, f"Workflow planning failed: {exc}")

    if not plan.get("feature_columns"):
        raise HTTPException(
            400,
            "Could not identify relevant feature columns from the question. "
            "Try being more specific about which metrics to analyze.",
        )
    if not plan.get("outcome_columns"):
        raise HTTPException(
            400,
            "Could not identify outcome columns. "
            "Specify what you want to predict (e.g., forward returns).",
        )

    # Resolve tickers: user-selected > planner-selected > auto-detect from question
    tickers = req.tickers
    if not tickers:
        # Check if planner explicitly selected tickers
        if plan.get("tickers_mode") == "all_individual":
            tickers = await _load_all_tickers(oi_pool, req.table)
        elif plan.get("tickers_override"):
            tickers = plan["tickers_override"]
        else:
            # Auto-detect: if the question mentions per-ticker analysis, load all
            q_lower = req.question.lower()
            ticker_keywords = [
                "per-ticker", "per ticker", "ticker-level", "ticker level",
                "specific tickers", "which tickers", "across tickers",
                "explore tickers", "individual ticker", "each ticker",
                "ticker by ticker", "compare tickers", "best tickers",
                "show promise", "which symbols", "which stocks",
            ]
            if any(kw in q_lower for kw in ticker_keywords):
                tickers = await _load_all_tickers(oi_pool, req.table)

    config = {
        "engine":        "v2",
        "table":         req.table,
        "tickers":       tickers,
        "date_from":     req.date_from,
        "date_to":       req.date_to,
        "model":         req.model,
        "workflow_plan": plan,
    }

    async with pool.acquire() as conn:
        run_id = await rdb.create_run(conn, req.name, req.question, config)

    background_tasks.add_task(
        _execute_run, pool, oi_pool, run_id,
        req.question, plan, config, req.model,
    )

    return {"run_id": run_id, "plan": plan}


@router.get("/tickers")
async def available_tickers(pool=Depends(get_oi_pool)):
    if pool is None:
        return []
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT DISTINCT ticker FROM daily_features ORDER BY ticker")
    return [r["ticker"] for r in rows]


@router.get("/columns")
async def available_columns_endpoint(
    table: str = "daily_features",
    pool=Depends(get_pool),
    oi_pool=Depends(get_oi_pool),
):
    return await _get_columns(table, pool, oi_pool)


# ── Follow-up questions ──────────────────────────────────────────────────────

class FollowupRequest(BaseModel):
    question: str


class KnowledgeRule(BaseModel):
    category: str = "policy"      # terminology, assumption, caveat, policy
    rule: str


class KnowledgeUpdate(BaseModel):
    category: Optional[str] = None
    rule: Optional[str] = None
    active: Optional[bool] = None


# ── P&L upload table ────────────────────────────────────────────────────────

_pnl_table_ready = False


async def _ensure_pnl_table(pool):
    global _pnl_table_ready
    if _pnl_table_ready:
        return
    try:
        async with pool.acquire() as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS research_pnl_uploads (
                    id         UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    name       TEXT NOT NULL,
                    row_count  INTEGER,
                    date_from  DATE,
                    date_to    DATE,
                    columns    JSONB,
                    data       JSONB,
                    created_at TIMESTAMPTZ DEFAULT NOW()
                )
            """)
        _pnl_table_ready = True
    except Exception:
        pass


@router.post("/upload-pnl")
async def upload_pnl(
    file: UploadFile = File(...),
    name: str = Form(...),
    pool=Depends(get_pool),
):
    """Upload a P&L CSV and store it. Returns upload_id, preview, column types."""
    await _ensure_pnl_table(pool)
    content_bytes = await file.read()
    try:
        content = content_bytes.decode("utf-8-sig")
    except UnicodeDecodeError:
        content = content_bytes.decode("latin-1")

    try:
        rows, meta = parse_pnl_csv(content)
    except Exception as exc:
        raise HTTPException(400, f"CSV parse error: {exc}")

    if not rows:
        raise HTTPException(400, "CSV appears empty")

    def _to_date(s):
        return _date.fromisoformat(s) if s else None

    date_from = _to_date(meta.get("date_from"))
    date_to   = _to_date(meta.get("date_to"))

    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """INSERT INTO research_pnl_uploads
               (name, row_count, date_from, date_to, columns, data)
               VALUES ($1, $2, $3, $4, $5::jsonb, $6::jsonb)
               RETURNING id::text, name, row_count, date_from, date_to, created_at""",
            name,
            meta["row_count"],
            date_from,
            date_to,
            json.dumps(meta["columns"]),
            json.dumps(rows),
        )

    return {
        "upload_id": row["id"],
        "name":      row["name"],
        "row_count": row["row_count"],
        "date_from": str(row["date_from"]) if row["date_from"] else None,
        "date_to":   str(row["date_to"])   if row["date_to"]   else None,
        "columns":   meta["columns"],
        "preview":   rows[:5],
    }


@router.get("/uploads")
async def list_uploads(pool=Depends(get_pool)):
    """List recent P&L uploads."""
    await _ensure_pnl_table(pool)
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """SELECT id::text, name, row_count, date_from, date_to, created_at
               FROM research_pnl_uploads
               ORDER BY created_at DESC LIMIT 20""",
        )
    return [dict(r) for r in rows]


# ── Backtest upload table ────────────────────────────────────────────────────

_backtest_table_ready = False


async def _ensure_backtest_table(pool):
    global _backtest_table_ready
    if _backtest_table_ready:
        return
    try:
        async with pool.acquire() as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS research_backtest_uploads (
                    id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    name          TEXT NOT NULL,
                    source        TEXT,
                    trade_count   INTEGER,
                    matched_count INTEGER,
                    date_from     DATE,
                    date_to       DATE,
                    strategies    JSONB,
                    columns       JSONB,
                    sources       JSONB,
                    data          JSONB,
                    created_at    TIMESTAMPTZ DEFAULT NOW()
                )
            """)
            await conn.execute(
                "ALTER TABLE research_backtest_uploads "
                "ADD COLUMN IF NOT EXISTS sources JSONB"
            )
        _backtest_table_ready = True
    except Exception:
        pass


@router.post("/upload-backtest")
async def upload_backtest(
    files: List[UploadFile] = File(...),
    name: str = Form(...),
    pool=Depends(get_pool),
):
    """Upload one or more backtest files (CSV or JSON), aggregate, align to IV surface, store."""
    await _ensure_backtest_table(pool)

    all_trades: list = []
    all_metas:  list = []
    sources:    list = []

    # Read all files first, then parse in parallel threads
    file_data = []
    for file in files:
        content_bytes = await file.read()
        file_data.append((content_bytes, file.filename))
        sources.append(file.filename)

    log.info("Backtest upload: parsing %d files (%s) in parallel...",
             len(file_data), ", ".join(sources))

    parse_tasks = [
        asyncio.to_thread(rbacktest.parse_backtest_upload, content, fname)
        for content, fname in file_data
    ]
    results = await asyncio.gather(*parse_tasks, return_exceptions=True)

    for i, result in enumerate(results):
        if isinstance(result, Exception):
            raise HTTPException(400, f"Parse error in '{sources[i]}': {result}")
        trades, meta = result
        # Drop daily_path immediately — massive and not needed for regime analysis
        for t in trades:
            t.pop('daily_path', None)
        all_trades.extend(trades)
        all_metas.append(meta)

    log.info("Parsed %d trades from %d files", len(all_trades), len(file_data))

    if not all_trades:
        raise HTTPException(400, "No trades found in uploaded files")

    # Sort combined trades chronologically
    all_trades.sort(key=lambda t: t.get('date_opened', ''))

    # Merge meta across files
    date_froms = [m['date_from'] for m in all_metas if m.get('date_from')]
    date_tos   = [m['date_to']   for m in all_metas if m.get('date_to')]
    date_from  = min(date_froms) if date_froms else None
    date_to    = max(date_tos)   if date_tos   else None
    if not date_from or not date_to:
        raise HTTPException(400, "Could not determine date range from files")

    strategies = sorted({s for m in all_metas for s in (m.get('strategies') or [])})
    columns    = all_metas[0].get('columns') or []
    source     = all_metas[0].get('source', 'unknown')

    try:
        enriched, stats = await rbacktest.align_trades_to_surface(
            all_trades, pool, date_from, date_to
        )
    except Exception as exc:
        raise HTTPException(500, f"Surface alignment failed: {exc}")

    def _to_date(s):
        return _date.fromisoformat(s) if s else None

    # Strip daily_path from ALL trades — it bloats the payload massively
    # (thousands of trades × 30-45 daily entries each). Regime analysis only
    # needs entry date, exit date, P&L, and IV surface fields at entry.
    for t in enriched:
        t.pop('daily_path', None)

    def _strip_path(t):
        return {k: v for k, v in t.items() if k != 'daily_path'}

    try:
        enriched_json = await asyncio.to_thread(json.dumps, enriched)
    except (TypeError, ValueError) as exc:
        raise HTTPException(500, f"Serialization failed: {exc}") from exc

    try:
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                """INSERT INTO research_backtest_uploads
                   (name, source, trade_count, matched_count, date_from, date_to,
                    strategies, columns, sources, data)
                   VALUES ($1,$2,$3,$4,$5,$6,$7::jsonb,$8::jsonb,$9::jsonb,$10::jsonb)
                   RETURNING id::text, name, source, trade_count, matched_count,
                             date_from, date_to, created_at""",
                name,
                source,
                len(all_trades),
                stats["matched"],
                _to_date(date_from),
                _to_date(date_to),
                json.dumps(strategies),
                json.dumps(columns),
                json.dumps(sources),
                enriched_json,
            )
    except Exception as exc:
        log.error("Backtest INSERT failed: %s", exc)
        raise HTTPException(500, f"Database insert failed: {exc}") from exc

    return {
        "upload_id":     row["id"],
        "name":          row["name"],
        "source":        row["source"],
        "trade_count":   row["trade_count"],
        "matched_count": row["matched_count"],
        "match_rate":    stats["match_rate"],
        "date_from":     str(row["date_from"]) if row["date_from"] else None,
        "date_to":       str(row["date_to"])   if row["date_to"]   else None,
        "strategies":    strategies,
        "columns":       columns,
        "sources":       sources,
        "validation":    {"stats": stats, "warnings": stats.get("warnings", [])},
        "preview":       [_strip_path(t) for t in enriched[:5]],
    }


@router.get("/backtest-uploads")
async def list_backtest_uploads(pool=Depends(get_pool)):
    """List recent backtest uploads."""
    await _ensure_backtest_table(pool)
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """SELECT id::text, name, source, trade_count, matched_count,
                      date_from, date_to, strategies, created_at
               FROM research_backtest_uploads
               ORDER BY created_at DESC LIMIT 20""",
        )
    result = []
    for r in rows:
        d = dict(r)
        d["strategies"] = d["strategies"] if isinstance(d["strategies"], list) else json.loads(d["strategies"] or "[]")
        result.append(d)
    return result


# ── Knowledge library ────────────────────────────────────────────────────────

_knowledge_table_ready = False


async def _ensure_knowledge_table(pool):
    global _knowledge_table_ready
    if _knowledge_table_ready:
        return
    try:
        async with pool.acquire() as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS research_knowledge (
                    id         SERIAL PRIMARY KEY,
                    category   TEXT NOT NULL DEFAULT 'policy',
                    rule       TEXT NOT NULL,
                    active     BOOLEAN DEFAULT true,
                    created_at TIMESTAMPTZ DEFAULT NOW()
                )
            """)
        _knowledge_table_ready = True
    except Exception:
        pass


async def _load_active_rules(pool) -> list[str]:
    """Load active knowledge rules as formatted strings for prompt injection."""
    try:
        await _ensure_knowledge_table(pool)
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT category, rule FROM research_knowledge WHERE active = true ORDER BY id")
        return [f"[{r['category']}] {r['rule']}" for r in rows]
    except Exception:
        return []


@router.get("/run/{run_id}/results")
async def get_results(run_id: str, pool=Depends(get_pool)):
    async with pool.acquire() as conn:
        results = await rdb.load_results(conn, run_id)
    for r in results:
        if isinstance(r.get("result"), str):
            r["result"] = json.loads(r["result"])
    return results


@router.get("/run/{run_id}/followups")
async def get_followups(run_id: str, pool=Depends(get_pool)):
    async with pool.acquire() as conn:
        return await rdb.load_followups(conn, run_id)


@router.post("/run/{run_id}/followup")
async def ask_followup(run_id: str, req: FollowupRequest, pool=Depends(get_pool)):
    """Follow-up question about a completed Research 2 run."""
    try:
        return await _do_followup(run_id, req, pool)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(500, f"Followup error: {type(exc).__name__}: {exc}")


async def _do_followup(run_id: str, req: FollowupRequest, pool):
    async with pool.acquire() as conn:
        run = await rdb.load_run(conn, run_id)
        if not run:
            raise HTTPException(404, "Run not found")
        results   = await rdb.load_results(conn, run_id)
        prev_fups = await rdb.load_followups(conn, run_id)

    for r in results:
        if isinstance(r.get("result"), str):
            r["result"] = json.loads(r["result"])

    # Build scan context
    scan_rows = []
    for r in results:
        rd = r.get("result") or {}
        if r.get("analysis_type") == "scan" and "error" not in rd:
            scan_rows.append({**rd, "ticker": r.get("ticker"),
                              "x_col": r.get("x_col"), "y_col": r.get("y_col")})
    corr_table = format_corr_table(scan_rows) if scan_rows else "(no scans)"

    # Interaction context
    int_lines = []
    for r in results:
        if r.get("analysis_type") in ("interaction", "interaction_3f"):
            rd = r.get("result") or {}
            combo = "+".join(rd.get("combo", []))
            int_lines.append(
                f"  {r.get('ticker') or 'all'} | {combo} → {r.get('y_col')}: "
                f"score={rd.get('composite_interaction_score', 0)}, "
                f"lift={rd.get('interaction_lift', 0)}")
    int_summary = "\n".join(int_lines) if int_lines else "(none)"

    cfg = run.get("config") or {}
    if isinstance(cfg, str):
        cfg = json.loads(cfg)
    plan = cfg.get("workflow_plan") or {}

    system = (
        "You are a quantitative research analyst reviewing completed research. "
        "Answer follow-up questions concisely, citing numbers from the analysis. "
        "Be honest about what the data shows and does not show."
    )

    context_msg = (
        f"Research question: {run.get('question')}\n"
        f"Task type: {plan.get('task_type', '?')}\n"
        f"Table: {cfg.get('table')}  |  "
        f"Tickers: {', '.join(cfg.get('tickers') or ['all'])}\n\n"
        f"=== Scan Results ===\n{corr_table}\n\n"
        f"=== Interactions ===\n{int_summary}\n\n"
        f"=== AI Summary ===\n{_extract_prose(run.get('ai_summary'))}"
    )

    messages = [
        {"role": "user", "content": context_msg},
        {"role": "assistant", "content": "I have the full research context. What's your question?"},
    ]
    for fup in prev_fups:
        messages.append({"role": "user", "content": fup["question"]})
        messages.append({"role": "assistant", "content": fup.get("answer") or ""})
    messages.append({"role": "user", "content": req.question})

    client = anthropic.AsyncAnthropic()
    resp = await client.messages.create(
        model=cfg.get("model") or "claude-sonnet-4-6",
        max_tokens=2048,
        system=system,
        messages=messages,
    )
    answer = resp.content[0].text if resp.content else "No response."

    async with pool.acquire() as conn:
        saved = await rdb.save_followup(conn, run_id, req.question, answer)
    return saved


# ── Knowledge library endpoints ──────────────────────────────────────────────

@router.get("/knowledge")
async def list_knowledge(pool=Depends(get_pool)):
    await _ensure_knowledge_table(pool)
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT id, category, rule, active, created_at "
            "FROM research_knowledge ORDER BY id")
    return [dict(r) for r in rows]


@router.post("/knowledge")
async def add_knowledge(req: KnowledgeRule, pool=Depends(get_pool)):
    await _ensure_knowledge_table(pool)
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "INSERT INTO research_knowledge (category, rule) VALUES ($1, $2) "
            "RETURNING id, category, rule, active, created_at",
            req.category, req.rule)
    return dict(row)


@router.patch("/knowledge/{rule_id}")
async def update_knowledge(rule_id: int, req: KnowledgeUpdate, pool=Depends(get_pool)):
    await _ensure_knowledge_table(pool)
    async with pool.acquire() as conn:
        existing = await conn.fetchrow(
            "SELECT id FROM research_knowledge WHERE id = $1", rule_id)
        if not existing:
            raise HTTPException(404, "Rule not found")
        if req.rule is not None:
            await conn.execute(
                "UPDATE research_knowledge SET rule = $2 WHERE id = $1", rule_id, req.rule)
        if req.category is not None:
            await conn.execute(
                "UPDATE research_knowledge SET category = $2 WHERE id = $1", rule_id, req.category)
        if req.active is not None:
            await conn.execute(
                "UPDATE research_knowledge SET active = $2 WHERE id = $1", rule_id, req.active)
        row = await conn.fetchrow(
            "SELECT id, category, rule, active, created_at "
            "FROM research_knowledge WHERE id = $1", rule_id)
    return dict(row)


@router.delete("/knowledge/{rule_id}")
async def delete_knowledge(rule_id: int, pool=Depends(get_pool)):
    await _ensure_knowledge_table(pool)
    async with pool.acquire() as conn:
        await conn.execute("DELETE FROM research_knowledge WHERE id = $1", rule_id)
    return {"deleted": rule_id}
