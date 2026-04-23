"""
AI Surface Explorer endpoint.

POST /api/ai-explorer/query
  {"question": "..."}

Flow:
  1. Question + schema → Claude Sonnet → SQL
  2. Validate SQL (SELECT-only, block dangerous keywords)
  3. Enforce 300-row cap
  4. Execute against surface_metrics_core in a read-only transaction
  5. Claude Haiku summarizes result in plain English
  6. Return {sql, columns, rows, summary, error}
"""
import os
import re
import json

import asyncpg
from anthropic import AsyncAnthropic
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from app.db import get_pool

router = APIRouter(tags=["ai_explorer"])

# ── Full column list (all 86 columns from surface_metrics_core) ───────────────

_ALL_COLUMNS = """
  trade_date, quote_time, day_of_week, days_to_monthly_opex, spot,
  forward_1d, forward_7d, forward_30d, forward_90d, forward_180d,
  iv_1d_25p, iv_1d_atm, iv_1d_25c,
  iv_7d_25p, iv_7d_atm, iv_7d_25c,
  iv_30d_25p, iv_30d_atm, iv_30d_25c,
  iv_90d_25p, iv_90d_atm, iv_90d_25c,
  iv_180d_25p, iv_180d_atm, iv_180d_25c,
  vix_1d, vix_7d, vix_30d, vix_90d, vix_180d,
  term_ratio_1d_7d, term_ratio_7d_30d, term_ratio_30d_90d,
  skew_1d_10p_25p, skew_1d_25p_atm, skew_1d_10p_atm, skew_1d_atm_25c, skew_1d_atm_10c, skew_1d_25p_25c,
  skew_7d_10p_25p, skew_7d_25p_atm, skew_7d_10p_atm, skew_7d_atm_25c, skew_7d_atm_10c, skew_7d_25p_25c,
  skew_30d_10p_25p, skew_30d_25p_atm, skew_30d_10p_atm, skew_30d_atm_25c, skew_30d_atm_10c, skew_30d_25p_25c,
  skew_90d_10p_25p, skew_90d_25p_atm, skew_90d_10p_atm, skew_90d_atm_25c, skew_90d_atm_10c, skew_90d_25p_25c,
  skew_180d_10p_25p, skew_180d_25p_atm, skew_180d_10p_atm, skew_180d_atm_25c, skew_180d_atm_10c, skew_180d_25p_25c,
  term_slope_1_7_25p, term_slope_1_7_atm, term_slope_1_7_25c,
  term_slope_7_30_25p, term_slope_7_30_atm, term_slope_7_30_25c,
  term_slope_30_90_25p, term_slope_30_90_atm, term_slope_30_90_25c,
  convex_1d_10p_25p_atm, convex_1d_atm_25c_10c, convex_1d_25p_atm_25c,
  convex_7d_10p_25p_atm, convex_7d_atm_25c_10c, convex_7d_25p_atm_25c,
  convex_30d_10p_25p_atm, convex_30d_atm_25c_10c, convex_30d_25p_atm_25c,
  convex_90d_10p_25p_atm, convex_90d_atm_25c_10c, convex_90d_25p_atm_25c,
  convex_180d_10p_25p_atm, convex_180d_atm_25c_10c, convex_180d_25p_atm_25c
"""

# ── System prompt (cached across requests) ────────────────────────────────────

_SCHEMA_TEXT = f"""You are a PostgreSQL assistant for an SPX implied volatility dashboard.
Generate a single valid SELECT query for the table described below.
Output ONLY the raw SQL — no markdown fences, no explanation, no comments.
Allowed: SELECT, WHERE, GROUP BY, ORDER BY, LIMIT, aggregates, CTEs, subqueries.
Forbidden: INSERT, UPDATE, DELETE, DROP, ALTER, TRUNCATE, CREATE, GRANT, REVOKE, EXECUTE, COPY.
Always include LIMIT (max 300).

TABLE: surface_metrics_core
Primary key: (trade_date, quote_time)
One row per intraday snapshot — every 5 min from 09:35 to 16:00 on each trading day.
ONLY query surface_metrics_core — do not reference any other tables.

COLUMNS (all lowercase):
  trade_date              DATE     — trading date (YYYY-MM-DD)
  quote_time              TIME     — snapshot time (HH:MM:SS)
  day_of_week             INTEGER  — 0=Mon … 4=Fri
  days_to_monthly_opex    INTEGER  — calendar days to next monthly SPX expiry
  spot                    FLOAT    — SPX spot price
  forward_1d/7d/30d/90d/180d  FLOAT — SPX forward prices by tenor

  -- Implied vol (decimal; 0.15 = 15%) by tenor × delta:
  iv_{{tenor}}_25p / iv_{{tenor}}_atm / iv_{{tenor}}_25c
  Available tenors: 1d, 7d, 30d, 90d, 180d

  -- VIX-equivalent ATM IV scaled to VIX units:
  vix_1d, vix_7d, vix_30d, vix_90d, vix_180d

  -- Term ratios (short/long ATM IV; >1 = backwardation):
  term_ratio_1d_7d, term_ratio_7d_30d, term_ratio_30d_90d

  -- Skew = IV differences, named skew_{{tenor}}_{{delta_a}}_{{delta_b}}
  -- (delta_a IV minus delta_b IV, e.g. skew_30d_25p_atm = iv_30d_25p − iv_30d_atm)
  -- Available for tenors 1d/7d/30d/90d/180d × pairs: 10p_25p, 25p_atm, 10p_atm, atm_25c, atm_10c, 25p_25c

  -- Term slope (fwd vol between adjacent tenors) for deltas 25p/atm/25c:
  term_slope_1_7_25p, term_slope_1_7_atm, term_slope_1_7_25c
  term_slope_7_30_25p, term_slope_7_30_atm, term_slope_7_30_25c
  term_slope_30_90_25p, term_slope_30_90_atm, term_slope_30_90_25c

  -- Convexity (wing premium above center) for tenors 1d/7d/30d/90d/180d:
  convex_{{tenor}}_10p_25p_atm, convex_{{tenor}}_atm_25c_10c, convex_{{tenor}}_25p_atm_25c

FULL COLUMN LIST:{_ALL_COLUMNS}

QUERY GUIDANCE:
- Default behavior: return ALL intraday rows (include quote_time in SELECT, no GROUP BY date).
- For "daily", "per day", "end of day", "closing" questions:
    DISTINCT ON (trade_date) ORDER BY trade_date, quote_time DESC
  or GROUP BY trade_date with appropriate aggregates.
- IV values are decimals — mention this in context but do NOT auto-multiply in SQL unless asked.
- Always include LIMIT (max 300). Put LIMIT at the very end of the query.

CHART GUIDANCE (follow when the user requests a chart, graph, or visualization):
- Never return a wide single-row pivot (many metric columns, one row). Charts cannot render that shape.
- For comparisons across tenors or metrics: use UNION ALL to produce one row per item with a
  string label column (e.g. 'metric') plus one or more numeric value columns.
  Example shape: (metric TEXT, min FLOAT, max FLOAT, mean FLOAT, p25 FLOAT, p75 FLOAT)
- For time series: return (trade_date, quote_time if intraday) + one or more numeric columns,
  ordered by time — the chart detector maps date/time + numeric → line chart automatically.
- For a single distribution stat (e.g. "just show me the mean"): returning a two-column result
  (label, value) is still better than a one-column scalar.
"""

_SYSTEM_SQL = [
    {
        "type": "text",
        "text": _SCHEMA_TEXT,
        "cache_control": {"type": "ephemeral"},
    }
]

# ── Safety ────────────────────────────────────────────────────────────────────

_BLOCKED_RE = re.compile(
    r'\b(INSERT|UPDATE|DELETE|DROP|ALTER|TRUNCATE|CREATE|GRANT|REVOKE|EXECUTE|EXEC|CALL|COPY)\b',
    re.IGNORECASE,
)
ROW_CAP = 300


def _validate_sql(sql: str) -> str:
    s = sql.strip()
    if not re.match(r'^\s*(?:WITH\b|SELECT\b)', s, re.IGNORECASE):
        raise ValueError("Only SELECT queries (including CTEs) are allowed")
    if _BLOCKED_RE.search(s):
        raise ValueError("Query contains a disallowed keyword")
    return s


_FETCH_RE = re.compile(
    r'\bFETCH\s+(?:FIRST|NEXT)\s+(\d+)\s+ROWS?\s+(?:ONLY|WITH\s+TIES)\b',
    re.IGNORECASE,
)

def _enforce_limit(sql: str) -> str:
    # Handle standard LIMIT n syntax
    limit_re = re.compile(r'\bLIMIT\s+(\d+)', re.IGNORECASE)
    m = limit_re.search(sql)
    if m:
        if int(m.group(1)) > ROW_CAP:
            sql = limit_re.sub(f'LIMIT {ROW_CAP}', sql)
        return sql

    # Handle SQL-standard FETCH FIRST n ROWS ONLY — replace with LIMIT so
    # we don't end up appending LIMIT after it (syntax error in Postgres).
    mf = _FETCH_RE.search(sql)
    if mf:
        cap = min(int(mf.group(1)), ROW_CAP)
        sql = _FETCH_RE.sub(f'LIMIT {cap}', sql)
        return sql

    # No row cap found — append one
    sql = sql.rstrip().rstrip(';') + f'\nLIMIT {ROW_CAP}'
    return sql


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_client() -> AsyncAnthropic:
    key = os.getenv('ANTHROPIC_API_KEY')
    if not key:
        raise HTTPException(500, "ANTHROPIC_API_KEY is not set — add it to your .env file")
    return AsyncAnthropic(api_key=key)


def _serialize(pg_rows) -> tuple[list[str], list[dict]]:
    if not pg_rows:
        return [], []
    columns = list(pg_rows[0].keys())
    rows = []
    for r in pg_rows:
        row = {}
        for k in columns:
            v = r[k]
            row[k] = v.isoformat() if hasattr(v, 'isoformat') else v
        rows.append(row)
    return columns, rows


def _strip_fences(text: str) -> str:
    text = re.sub(r'^```[\w]*\s*', '', text.strip(), flags=re.IGNORECASE)
    text = re.sub(r'\s*```\s*$', '', text)
    return text.strip()


# ── Request models ────────────────────────────────────────────────────────────

class HistoryTurn(BaseModel):
    question: str
    sql:      Optional[str] = None
    summary:  Optional[str] = None
    error:    Optional[str] = None


class QueryRequest(BaseModel):
    question: str
    history:  list[HistoryTurn] = []


# ── Endpoint ──────────────────────────────────────────────────────────────────

@router.post("/query")
async def ai_query(req: QueryRequest, pool=Depends(get_pool)) -> dict:
    question = req.question.strip()
    if not question:
        raise HTTPException(400, "question must not be empty")

    client = _get_client()
    sql = None  # kept in scope so catch-all can return it

    try:
        # 1. Build conversation messages (last 6 turns for context)
        messages = []
        for turn in req.history[-6:]:
            messages.append({"role": "user", "content": turn.question})
            if turn.error:
                asst = f"Error: {turn.error}"
            elif turn.sql and turn.summary:
                asst = f"Generated SQL:\n```sql\n{turn.sql}\n```\n\nResult: {turn.summary}"
            elif turn.sql:
                asst = f"Generated SQL:\n```sql\n{turn.sql}\n```"
            else:
                asst = "No result."
            messages.append({"role": "assistant", "content": asst})
        messages.append({"role": "user", "content": question})

        # 2. Generate SQL via Claude Sonnet (system prompt is cached)
        sql_msg = await client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1024,
            system=_SYSTEM_SQL,
            messages=messages,
        )
        raw_sql = _strip_fences(sql_msg.content[0].text)

        # 2. Validate + enforce row cap
        try:
            sql = _validate_sql(raw_sql)
            sql = _enforce_limit(sql)
        except ValueError as e:
            return {"sql": raw_sql, "error": str(e), "columns": [], "rows": [], "summary": None}

        # 3. Execute in a read-only transaction
        try:
            async with pool.acquire() as conn:
                async with conn.transaction(readonly=True):
                    pg_rows = await conn.fetch(sql)
        except asyncpg.PostgresError as e:
            return {"sql": sql, "error": f"Database error: {e}", "columns": [], "rows": [], "summary": None}

        columns, rows = _serialize(pg_rows)

        if not rows:
            return {"sql": sql, "error": None, "columns": columns, "rows": [], "summary": "The query returned no results."}

        # 4. Summarize via Claude Haiku
        preview = json.dumps(rows[:15], default=str)
        summary_msg = await client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=256,
            messages=[{
                "role": "user",
                "content": (
                    f"User asked: {question!r}\n\n"
                    f"The SQL returned {len(rows)} row(s). First rows (JSON):\n{preview}\n\n"
                    "Write a concise 1–3 sentence plain English summary of the key insight. "
                    "Use financial terminology. Do not describe the data format or JSON structure."
                ),
            }],
        )
        summary = summary_msg.content[0].text.strip()

        return {"sql": sql, "error": None, "columns": columns, "rows": rows, "summary": summary}

    except Exception as e:
        # Catch-all: always return JSON so the frontend can display the real error.
        # Common causes: Anthropic API auth/network error, table doesn't exist yet.
        return {
            "sql": sql,
            "error": f"{type(e).__name__}: {e}",
            "columns": [],
            "rows": [],
            "summary": None,
        }
