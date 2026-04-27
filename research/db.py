"""DB helpers for research tables (main database)."""
import json
import asyncpg
from typing import Optional


async def create_run(conn: asyncpg.Connection, name: str, question: str, config: dict) -> str:
    row = await conn.fetchrow(
        """
        INSERT INTO research_runs (name, question, config, status)
        VALUES ($1, $2, $3::jsonb, 'running')
        RETURNING id::text
        """,
        name, question, json.dumps(config),
    )
    return row["id"]


async def update_run(conn: asyncpg.Connection, run_id: str, *,
                     status: Optional[str] = None,
                     ai_summary: Optional[str] = None,
                     error_msg: Optional[str] = None):
    await conn.execute(
        """
        UPDATE research_runs SET
            status       = COALESCE($2, status),
            completed_at = CASE WHEN $2 IN ('complete','error') THEN NOW() ELSE completed_at END,
            ai_summary   = COALESCE($3, ai_summary),
            error_msg    = COALESCE($4, error_msg)
        WHERE id = $1::uuid
        """,
        run_id, status, ai_summary, error_msg,
    )


async def save_result(conn: asyncpg.Connection, run_id: str, analysis_type: str,
                      x_col: str, y_col: str, result: dict,
                      ticker: Optional[str] = None):
    await conn.execute(
        """
        INSERT INTO research_results (run_id, ticker, x_col, y_col, analysis_type, result)
        VALUES ($1::uuid, $2, $3, $4, $5, $6::jsonb)
        """,
        run_id, ticker, x_col, y_col, analysis_type, json.dumps(result),
    )


async def save_series(conn: asyncpg.Connection, run_id: str, x_col: str, series_name: str,
                      data: list, ticker: Optional[str] = None, y_col: Optional[str] = None):
    await conn.execute(
        """
        INSERT INTO research_series (run_id, ticker, x_col, y_col, series_name, data)
        VALUES ($1::uuid, $2, $3, $4, $5, $6::jsonb)
        """,
        run_id, ticker, x_col, y_col, series_name, json.dumps(data),
    )


async def save_chart(conn: asyncpg.Connection, run_id: str, chart_type: str,
                     title: str, png_bytes: bytes,
                     ticker: Optional[str] = None,
                     x_col: Optional[str] = None,
                     y_col: Optional[str] = None):
    await conn.execute(
        """
        INSERT INTO research_charts (run_id, ticker, x_col, y_col, chart_type, title, png_data)
        VALUES ($1::uuid, $2, $3, $4, $5, $6, $7)
        """,
        run_id, ticker, x_col, y_col, chart_type, title, png_bytes,
    )


async def load_run(conn: asyncpg.Connection, run_id_or_name: str) -> Optional[dict]:
    row = await conn.fetchrow(
        """
        SELECT id::text, name, question, config, status,
               created_at, completed_at, ai_summary, error_msg
        FROM research_runs
        WHERE id::text = $1 OR name = $1
        ORDER BY created_at DESC LIMIT 1
        """,
        run_id_or_name,
    )
    return dict(row) if row else None


async def load_results(conn: asyncpg.Connection, run_id: str) -> list[dict]:
    rows = await conn.fetch(
        """
        SELECT id::text, run_id::text, ticker, x_col, y_col, analysis_type, result, created_at
        FROM research_results
        WHERE run_id = $1::uuid
        ORDER BY ticker NULLS LAST, x_col, y_col, analysis_type
        """,
        run_id,
    )
    return [dict(r) for r in rows]


async def load_series(conn: asyncpg.Connection, run_id: str) -> list[dict]:
    rows = await conn.fetch(
        """
        SELECT id::text, ticker, x_col, y_col, series_name, data
        FROM research_series
        WHERE run_id = $1::uuid
        ORDER BY ticker NULLS LAST, series_name
        """,
        run_id,
    )
    return [dict(r) for r in rows]


async def load_charts(conn: asyncpg.Connection, run_id: str) -> list[dict]:
    rows = await conn.fetch(
        """
        SELECT id::text, ticker, x_col, y_col, chart_type, title, png_data, created_at
        FROM research_charts
        WHERE run_id = $1::uuid
        ORDER BY ticker NULLS LAST, chart_type
        """,
        run_id,
    )
    return [dict(r) for r in rows]


async def save_followup(conn: asyncpg.Connection, run_id: str,
                        question: str, answer: str) -> dict:
    row = await conn.fetchrow(
        """
        INSERT INTO research_followups (run_id, question, answer)
        VALUES ($1::uuid, $2, $3)
        RETURNING id::text, created_at
        """,
        run_id, question, answer,
    )
    return {"id": row["id"], "question": question, "answer": answer,
            "created_at": row["created_at"]}


async def load_followups(conn: asyncpg.Connection, run_id: str) -> list[dict]:
    rows = await conn.fetch(
        """
        SELECT id::text, question, answer, created_at
        FROM research_followups
        WHERE run_id = $1::uuid
        ORDER BY created_at
        """,
        run_id,
    )
    return [dict(r) for r in rows]


async def list_runs(conn: asyncpg.Connection, limit: int = 20) -> list[dict]:
    rows = await conn.fetch(
        """
        SELECT id::text, name, question, status, created_at, completed_at
        FROM research_runs
        ORDER BY created_at DESC
        LIMIT $1
        """,
        limit,
    )
    return [dict(r) for r in rows]
