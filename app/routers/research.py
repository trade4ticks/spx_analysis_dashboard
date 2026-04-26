"""Research runner web endpoints."""
import json
import asyncio
from typing import Optional

from fastapi import APIRouter, Depends, BackgroundTasks, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel

from app.db import get_pool, get_oi_pool
import asyncpg

from research import db as rdb, agent as ragent, charts as rcharts, export as rexport

router = APIRouter()


# ── Background execution ──────────────────────────────────────────────────────

async def _execute_run(main_pool, oi_pool, run_id: str, question: str,
                       config: dict, model: str, max_tool_calls: int):
    try:
        summary = await ragent.run_agent(
            main_pool=main_pool,
            oi_pool=oi_pool,
            run_id=run_id,
            question=question,
            config=config,
            model=model,
            max_tool_calls=max_tool_calls,
            log=lambda msg: None,  # silent in web mode
        )
        # Generate correlation heatmap if multiple tickers
        tickers = config.get("tickers") or []
        x_cols  = config.get("x_columns") or []
        y_cols  = config.get("y_columns") or []
        if len(tickers) > 1 and x_cols and y_cols:
            async with main_pool.acquire() as conn:
                all_results = await rdb.load_results(conn, run_id)
            flat = []
            for r in all_results:
                if r.get("analysis_type") == "correlation":
                    rd = r.get("result") or {}
                    if isinstance(rd, str):
                        rd = json.loads(rd)
                    flat.append({**rd, "ticker": r.get("ticker"),
                                 "x_col": r.get("x_col"), "y_col": r.get("y_col")})
            if flat:
                png = rcharts.correlation_heatmap(flat, tickers, x_cols, y_cols)
                if png:
                    async with main_pool.acquire() as conn:
                        await rdb.save_chart(conn, run_id, "correlation_heatmap",
                                             "Pearson Correlation Heatmap", png)
        async with main_pool.acquire() as conn:
            await rdb.update_run(conn, run_id, status="complete", ai_summary=summary)
    except Exception as exc:
        async with main_pool.acquire() as conn:
            await rdb.update_run(conn, run_id, status="error", error_msg=str(exc))


# ── Request models ────────────────────────────────────────────────────────────

class RunRequest(BaseModel):
    name: str
    question: str
    table: str = "daily_features"
    tickers: list[str] = []
    x_columns: list[str] = []
    y_columns: list[str] = []
    date_from: Optional[str] = None
    date_to: Optional[str] = None
    model: str = "claude-sonnet-4-6"
    max_tool_calls: int = 60


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

    config = {
        "name":         req.name,
        "table":        req.table,
        "tickers":      req.tickers,
        "x_columns":    req.x_columns,
        "y_columns":    req.y_columns,
        "date_from":    req.date_from,
        "date_to":      req.date_to,
        "model":        req.model,
        "max_tool_calls": req.max_tool_calls,
    }

    async with pool.acquire() as conn:
        run_id = await rdb.create_run(conn, req.name, req.question, config)

    background_tasks.add_task(
        _execute_run, pool, oi_pool, run_id,
        req.question, config, req.model, req.max_tool_calls,
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
