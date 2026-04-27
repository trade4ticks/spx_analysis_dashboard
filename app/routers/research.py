"""Research runner web endpoints."""
import json
import asyncio
from typing import Optional

import anthropic
from fastapi import APIRouter, Depends, BackgroundTasks, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel

from app.db import get_pool, get_oi_pool
import asyncpg

from research import db as rdb, engine as rengine, charts as rcharts, export as rexport

router = APIRouter()


# ── Background execution ──────────────────────────────────────────────────────

async def _execute_run(main_pool, oi_pool, run_id: str, question: str,
                       config: dict, model: str):
    pipeline_logs: list[str] = []

    def _log(msg: str):
        pipeline_logs.append(msg)

    try:
        summary = await rengine.run_pipeline(
            main_pool=main_pool,
            oi_pool=oi_pool,
            run_id=run_id,
            question=question,
            config=config,
            model=model,
            signal_threshold=config.get("signal_threshold", 0.03),
            max_signals=config.get("max_signals", 30),
            analysis_types=set(config.get("analysis_types") or [
                "correlation", "decile", "yearly_consistency",
                "equity_curve", "regression"]),
            log=_log,
        )
        # Always save pipeline log for diagnostic visibility
        log_text = "\n".join(pipeline_logs) if pipeline_logs else None
        async with main_pool.acquire() as conn:
            await rdb.update_run(conn, run_id, status="complete",
                                 ai_summary=summary or None,
                                 error_msg=log_text)
    except Exception as exc:
        log_text = "\n".join(pipeline_logs[-50:]) if pipeline_logs else None
        err = str(exc)
        if log_text:
            err = f"{err}\n\n--- Pipeline log ---\n{log_text}"
        async with main_pool.acquire() as conn:
            await rdb.update_run(conn, run_id, status="error", error_msg=err)


# ── Request models ────────────────────────────────────────────────────────────

class RunRequest(BaseModel):
    name: str
    question: str
    table: str = "daily_features"
    tickers: list[str] = []
    buckets: dict[str, list[str]] = {}   # {"bucket_name": ["col1", "col2"], ...}
    x_columns: list[str] = []            # backward compat: treated as single bucket
    y_columns: list[str] = []
    date_from: Optional[str] = None
    date_to: Optional[str] = None
    model: str = "claude-sonnet-4-6"
    signal_threshold: float = 0.03
    max_signals: int = 30
    analysis_types: list[str] = ["scan", "combo", "equity_curve", "regression"]


class FollowupRequest(BaseModel):
    question: str


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.get("/runs")
async def list_runs(pool=Depends(get_pool)):
    async with pool.acquire() as conn:
        runs = await rdb.list_runs(conn, limit=50)
    return runs


@router.get("/run/{run_id}")
async def get_run(run_id: str, pool=Depends(get_pool)):
    async with pool.acquire() as conn:
        run = await rdb.load_run(conn, run_id)
        if not run:
            raise HTTPException(404, "Run not found")
        result_count = await conn.fetchval(
            "SELECT COUNT(*) FROM research_results WHERE run_id = $1::uuid", run_id)
        chart_count = await conn.fetchval(
            "SELECT COUNT(*) FROM research_charts WHERE run_id = $1::uuid", run_id)
    cfg = run.get("config")
    if isinstance(cfg, str):
        run = {**run, "config": json.loads(cfg)}
    return {**run, "result_count": int(result_count), "chart_count": int(chart_count)}


@router.get("/run/{run_id}/results")
async def get_results(run_id: str, pool=Depends(get_pool)):
    async with pool.acquire() as conn:
        results = await rdb.load_results(conn, run_id)
    # Parse JSONB strings
    for r in results:
        if isinstance(r.get("result"), str):
            r["result"] = json.loads(r["result"])
    return results


@router.get("/run/{run_id}/charts")
async def get_chart_list(run_id: str, pool=Depends(get_pool)):
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """SELECT id::text, ticker, x_col, y_col, chart_type, title, created_at
               FROM research_charts WHERE run_id = $1::uuid
               ORDER BY ticker NULLS FIRST, chart_type""",
            run_id,
        )
    return [dict(r) for r in rows]


@router.get("/chart/{chart_id}.png")
async def get_chart_png(chart_id: str, pool=Depends(get_pool)):
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT png_data FROM research_charts WHERE id = $1::uuid", chart_id)
    if not row or not row["png_data"]:
        raise HTTPException(404, "Chart not found")
    return Response(content=bytes(row["png_data"]), media_type="image/png")


@router.post("/run")
async def start_run(req: RunRequest, background_tasks: BackgroundTasks,
                    pool=Depends(get_pool), oi_pool=Depends(get_oi_pool)):
    if oi_pool is None and req.table in {"daily_features", "option_oi_surface", "underlying_ohlc"}:
        raise HTTPException(400, "OI_DATABASE_URL not configured")

    # Normalize: if buckets provided, derive x_columns from them; else make one bucket
    buckets = req.buckets
    if not buckets and req.x_columns:
        buckets = {"features": req.x_columns}
    all_x = list(dict.fromkeys(col for cols in buckets.values() for col in cols))

    config = {
        "name":              req.name,
        "table":             req.table,
        "tickers":           req.tickers,
        "buckets":           buckets,
        "x_columns":         all_x,
        "y_columns":         req.y_columns,
        "date_from":         req.date_from,
        "date_to":           req.date_to,
        "model":             req.model,
        "signal_threshold":  req.signal_threshold,
        "max_signals":       req.max_signals,
        "analysis_types":    req.analysis_types,
    }

    async with pool.acquire() as conn:
        run_id = await rdb.create_run(conn, req.name, req.question, config)

    background_tasks.add_task(
        _execute_run, pool, oi_pool, run_id,
        req.question, config, req.model,
    )

    return {"run_id": run_id, "name": req.name}


@router.delete("/run/{run_id}")
async def delete_run(run_id: str, pool=Depends(get_pool)):
    async with pool.acquire() as conn:
        run = await rdb.load_run(conn, run_id)
        if not run:
            raise HTTPException(404, "Run not found")
        await conn.execute(
            "DELETE FROM research_runs WHERE id = $1::uuid", run_id)
    return {"deleted": run_id}


@router.post("/run/{run_id}/retry")
async def retry_run(run_id: str, req: RunRequest, background_tasks: BackgroundTasks,
                    pool=Depends(get_pool), oi_pool=Depends(get_oi_pool)):
    """Re-run a failed run with updated config. Clears partial results first."""
    async with pool.acquire() as conn:
        run = await rdb.load_run(conn, run_id)
        if not run:
            raise HTTPException(404, "Run not found")
        if run["status"] not in ("error", "running"):
            raise HTTPException(400, "Only failed runs can be retried")

        retry_buckets = req.buckets
        if not retry_buckets and req.x_columns:
            retry_buckets = {"features": req.x_columns}
        retry_all_x = list(dict.fromkeys(col for cols in retry_buckets.values() for col in cols))

        config = {
            "name":              req.name,
            "table":             req.table,
            "tickers":           req.tickers,
            "buckets":           retry_buckets,
            "x_columns":         retry_all_x,
            "y_columns":         req.y_columns,
            "date_from":         req.date_from,
            "date_to":           req.date_to,
            "model":             req.model,
            "signal_threshold":  req.signal_threshold,
            "max_signals":       req.max_signals,
            "analysis_types":    req.analysis_types,
        }
        # Clear any partial results from the failed run
        await conn.execute("DELETE FROM research_results WHERE run_id = $1::uuid", run_id)
        await conn.execute("DELETE FROM research_charts  WHERE run_id = $1::uuid", run_id)
        await conn.execute("DELETE FROM research_series  WHERE run_id = $1::uuid", run_id)
        await conn.execute(
            """UPDATE research_runs SET
                name = $2, question = $3, config = $4::jsonb,
                status = 'running', error_msg = NULL, ai_summary = NULL, completed_at = NULL
               WHERE id = $1::uuid""",
            run_id, req.name, req.question, json.dumps(config),
        )

    background_tasks.add_task(
        _execute_run, pool, oi_pool, run_id,
        req.question, config, req.model,
    )
    return {"run_id": run_id, "status": "running"}


@router.get("/run/{run_id}/followups")
async def get_followups(run_id: str, pool=Depends(get_pool)):
    async with pool.acquire() as conn:
        return await rdb.load_followups(conn, run_id)


@router.post("/run/{run_id}/followup")
async def ask_followup(run_id: str, req: FollowupRequest, pool=Depends(get_pool)):
    """Send a follow-up question about a completed run. Claude responds with full context."""
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

    # Parse result JSONB
    for r in results:
        if isinstance(r.get("result"), str):
            r["result"] = json.loads(r["result"])

    # Build correlation table context — support both scan (new) and correlation (legacy)
    corr_rows = []
    for r in results:
        rd = r.get("result") or {}
        if isinstance(rd, str):
            rd = json.loads(rd)
        if r.get("analysis_type") == "scan" and "error" not in rd:
            corr_rows.append({**rd, "ticker": r.get("ticker"),
                               "x_col": r.get("x_col"), "y_col": r.get("y_col")})
        elif r.get("analysis_type") == "correlation" and "error" not in rd:
            corr_rows.append({**rd, "ticker": r.get("ticker"),
                               "x_col": r.get("x_col"), "y_col": r.get("y_col")})
    corr_table = rengine.format_corr_table(corr_rows) if corr_rows else "(no correlations saved)"

    # Build scan/decile summary
    decile_lines = []
    for r in results:
        rd = r.get("result") or {}
        if isinstance(rd, str):
            rd = json.loads(rd)
        if r.get("analysis_type") == "scan" and "error" not in rd:
            spread = rd.get("tail_spread")
            score = rd.get("composite_score")
            pattern = rd.get("pattern")
            if spread is not None:
                decile_lines.append(
                    f"  [{score or 0:.0f}] {r.get('ticker') or 'all'} | {r.get('x_col')} → {r.get('y_col')}: "
                    f"tail_spread={spread*100:.3f}%, pattern={pattern}")
        elif r.get("analysis_type") == "decile":
            spread = rd.get("top_bottom_spread")
            if spread is not None:
                decile_lines.append(
                    f"  {r.get('ticker') or 'all'} | {r.get('x_col')} → {r.get('y_col')}: "
                    f"D10–D1 spread={spread*100:.3f}%")
    decile_summary = "\n".join(decile_lines) if decile_lines else "(none)"

    cfg = run.get("config") or {}
    if isinstance(cfg, str):
        cfg = json.loads(cfg)

    system = (
        "You are a quantitative research analyst reviewing completed statistical research. "
        "Answer the user's follow-up question specifically and concisely. "
        "Cite numbers from the analysis when relevant. "
        "Be honest about what the data does and does not show. "
        "If the question asks you to run new analysis you cannot do here, say so clearly."
    )

    # First message: full run context
    context_msg = (
        f"Research question: {run.get('question')}\n"
        f"Table: {cfg.get('table')}  |  "
        f"Tickers: {', '.join(cfg.get('tickers') or ['all'])}  |  "
        f"Features: {', '.join(cfg.get('x_columns') or [])}  |  "
        f"Outcomes: {', '.join(cfg.get('y_columns') or [])}\n\n"
        f"=== Correlation Matrix ===\n{corr_table}\n\n"
        f"=== Decile Spreads ===\n{decile_summary}\n\n"
        f"=== AI Summary ===\n{run.get('ai_summary') or '(no summary generated)'}"
    )

    # Build multi-turn message history
    messages = [{"role": "user", "content": context_msg},
                {"role": "assistant", "content": "Understood. I have the full research context. What's your follow-up question?"}]
    for fup in prev_fups:
        messages.append({"role": "user",      "content": fup["question"]})
        messages.append({"role": "assistant", "content": fup.get("answer") or ""})
    messages.append({"role": "user", "content": req.question})

    try:
        client = anthropic.AsyncAnthropic()
        resp = await client.messages.create(
            model=cfg.get("model") or "claude-sonnet-4-6",
            max_tokens=2048,
            system=system,
            messages=messages,
        )
        answer = resp.content[0].text if resp.content else "No response."
    except Exception as exc:
        answer = f"Error generating response: {type(exc).__name__}: {exc}"

    async with pool.acquire() as conn:
        saved = await rdb.save_followup(conn, run_id, req.question, answer)

    return saved


@router.get("/run/{run_id}/pdf")
async def download_pdf(run_id: str, pool=Depends(get_pool)):
    async with pool.acquire() as conn:
        run = await rdb.load_run(conn, run_id)
        if not run:
            raise HTTPException(404, "Run not found")
        results = await rdb.load_results(conn, run_id)
        charts  = await rdb.load_charts(conn, run_id)

    for r in results:
        if isinstance(r.get("result"), str):
            r["result"] = json.loads(r["result"])

    pdf_bytes = rexport.build_pdf_bytes(run, results, charts)
    safe_name = run["name"].replace(" ", "_").replace("/", "-")
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{safe_name}.pdf"'},
    )


@router.get("/tickers")
async def available_tickers(pool=Depends(get_oi_pool)):
    if pool is None:
        return []
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT DISTINCT ticker FROM daily_features ORDER BY ticker")
    return [r["ticker"] for r in rows]


_OI_TABLES = {"daily_features", "option_oi_surface", "underlying_ohlc"}

_NUMERIC_TYPES = {
    "double precision", "numeric", "real", "integer",
    "bigint", "smallint", "decimal",
}
_EXCLUDE_COLS = {"id", "ticker", "trade_date", "created_at", "updated_at"}


@router.get("/columns")
async def available_columns(table: str = "daily_features",
                            pool=Depends(get_pool), oi_pool=Depends(get_oi_pool)):
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
        if r["data_type"] in _NUMERIC_TYPES
        and r["column_name"] not in _EXCLUDE_COLS
    ]
