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
from typing import Optional

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

Output format — one of two modes:

MODE A — Data query (most requests):
  Line 1: chart_type hint — one of: line, bar, scatter, none
  Line 2+: the raw SQL query (no markdown fences)
  Allowed SQL: SELECT, WHERE, GROUP BY, ORDER BY, LIMIT, aggregates, CTEs, subqueries.
  Forbidden: INSERT, UPDATE, DELETE, DROP, ALTER, TRUNCATE, CREATE, GRANT, REVOKE, EXECUTE, COPY.
  Always include LIMIT (max 300).

MODE B — Text response (for follow-up questions, clarifications, explanations):
  If the user asks a question that doesn't need a data query (e.g. "why did you...",
  "can you explain...", "what does X mean"), respond with plain text. Do NOT wrap it
  in SQL. Start your response with "TEXT:" on the first line, then your explanation.

Example Mode A:
scatter
SELECT col_a, col_b FROM surface_metrics_core LIMIT 300

Example Mode B:
TEXT:
The previous query used 30-day metrics because they are the most liquid tenor...

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
- For a single distribution stat: returning (label, value) is better than a one-column scalar.
- If the question does not need a chart, use chart_type hint: none
- For scatterplots: include date/time columns if the user wants them in the tooltip/table,
  the chart config controls what gets plotted (not the SQL columns).

CHART CONFIG (optional — append after SQL to customize rendering):
After the SQL, you may append a ---CHART--- section with a JSON Chart.js config.
Our renderer will populate data from query results using column references.

Special dataset keys (resolved by renderer):
  xColumn / yColumn    — column names for scatter x,y
  labelsColumn         — column name for x-axis labels (line/bar)
  colorColumn          — scatter color gradient column (red=negative, green=positive)
  regression: true     — compute & overlay regression line (set xSource, ySource)

All standard Chart.js dataset properties are supported (borderColor, borderDash,
borderWidth, pointRadius, tension, fill, type, label, order, backgroundColor, etc.).
Set axis labels under options.scales.x.title / y.title.

Example — scatter with regression and color-by:
---CHART---
{{"type":"scatter","datasets":[
  {{"xColumn":"iv_30d_atm","yColumn":"skew_30d_25p_atm","colorColumn":"spot_return",
    "pointRadius":4,"borderWidth":0}},
  {{"regression":true,"xSource":"iv_30d_atm","ySource":"skew_30d_25p_atm",
    "borderColor":"rgba(231,76,60,0.5)","borderDash":[6,4],"borderWidth":1.5,
    "pointRadius":0,"label":"Regression","type":"line"}}
],"options":{{"scales":{{"x":{{"title":{{"display":true,"text":"30D ATM IV"}}}},
"y":{{"title":{{"display":true,"text":"Put Skew"}}}}}}}}}}

Example — multi-line time series:
---CHART---
{{"type":"line","datasets":[
  {{"labelsColumn":"trade_date","yColumn":"iv_30d_atm","label":"30D ATM",
    "borderColor":"#3498db","borderWidth":1.5,"pointRadius":0}},
  {{"labelsColumn":"trade_date","yColumn":"iv_7d_atm","label":"7D ATM",
    "borderColor":"#2ecc71","borderWidth":1.5,"pointRadius":0}}
],"options":{{"scales":{{"y":{{"title":{{"display":true,"text":"IV"}}}}}}}}}}

Include ---CHART--- whenever the user asks for specific styling, regression,
custom axis labels, or any visual customization. Omit it for basic auto-rendered charts.
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


# ── Request classifier ────────────────────────────────────────────────────────

_CLASSIFIER_PROMPT = """You classify user questions about SPX implied volatility data.

Respond with exactly one word:
- "analysis" — if the user asks about the current vol regime, market conditions,
  regime summary, overall state of volatility, or a broad "what's happening" question
  that needs multiple charts/perspectives to answer properly.
- "direct" — if the user asks a specific data query, wants a particular chart, or
  requests something concrete (show me X, what was Y, plot Z, compare A vs B).

Examples:
  "What's the current vol regime?" → analysis
  "Summarize the IV surface" → analysis
  "Describe market conditions" → analysis
  "What stands out right now?" → analysis
  "Take a look at the metrics and tell me what's going on" → analysis
  "Show 30D ATM IV for last 2 weeks" → direct
  "What was the peak VIX reading this month?" → direct
  "Create a scatterplot of skew vs IV" → direct
  "Compare 7D vs 30D term slopes" → direct
"""


async def _classify_request(client, question: str) -> str:
    """Return 'analysis' or 'direct'. Falls back to 'direct' on any issue."""
    try:
        msg = await client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=16,
            messages=[{"role": "user", "content": question}],
            system=[{"type": "text", "text": _CLASSIFIER_PROMPT}],
        )
        result = msg.content[0].text.strip().lower()
        return result if result in ("analysis", "direct") else "direct"
    except Exception:
        return "direct"


# ── Regime analysis pipeline ─────────────────────────────────────────────────

_REGIME_METRICS = [
    ("iv_7d_atm",               "7D ATM IV"),
    ("iv_30d_atm",              "30D ATM IV"),
    ("iv_90d_atm",              "90D ATM IV"),
    ("iv_180d_atm",             "180D ATM IV"),
    ("skew_7d_25p_atm",         "7D Put Skew"),
    ("skew_30d_25p_atm",        "30D Put Skew"),
    ("skew_90d_25p_atm",        "90D Put Skew"),
    ("term_slope_7_30_atm",     "7-30D Term Slope"),
    ("term_slope_30_90_atm",    "30-90D Term Slope"),
    ("term_ratio_7d_30d",       "Term Ratio 7/30"),
    ("convex_30d_25p_atm_25c",  "30D Butterfly"),
    ("spot",                    "SPX Spot"),
]

_REGIME_COLS = ", ".join(col for col, _ in _REGIME_METRICS)


def _regime_ts_sql(lookback: int = 90) -> str:
    return f"""
        SELECT trade_date, {_REGIME_COLS}
        FROM (
            SELECT DISTINCT ON (trade_date) trade_date, {_REGIME_COLS}
            FROM surface_metrics_core
            WHERE trade_date >= CURRENT_DATE - INTERVAL '{lookback} days'
            ORDER BY trade_date, quote_time DESC
        ) daily ORDER BY trade_date
    """


def _regime_full_sql() -> str:
    return f"""
        SELECT trade_date, {_REGIME_COLS}
        FROM (
            SELECT DISTINCT ON (trade_date) trade_date, {_REGIME_COLS}
            FROM surface_metrics_core
            ORDER BY trade_date, quote_time DESC
        ) daily ORDER BY trade_date
    """


def _val_n_back(rows, col, n):
    idx = max(0, len(rows) - 1 - n)
    return rows[idx].get(col)


def _compute_regime_stats(lookback_rows, full_rows):
    import statistics as st
    if not lookback_rows:
        return []
    # Use the most recent row that has IV data populated (today's row might
    # have spot but no surface metrics yet if the pipeline hasn't run).
    cur = lookback_rows[-1]
    iv_check_col = "iv_30d_atm"
    if cur.get(iv_check_col) is None:
        for r in reversed(lookback_rows):
            if r.get(iv_check_col) is not None:
                cur = r
                break
    out = []
    for col, label in _REGIME_METRICS:
        cv = cur.get(col)
        if cv is None:
            continue
        v1w = _val_n_back(lookback_rows, col, 5)
        v1m = _val_n_back(lookback_rows, col, 21)
        c1w = (cv - v1w) if v1w is not None else None
        c1m = (cv - v1m) if v1m is not None else None
        vals = [r[col] for r in full_rows if r.get(col) is not None]
        if vals:
            pct = sum(1 for v in vals if v <= cv) / len(vals)
            mean = st.mean(vals)
            sd = st.pstdev(vals)
            z = (cv - mean) / sd if sd > 0 else 0.0
        else:
            pct, z = None, None
        out.append({
            "metric": col, "label": label,
            "current":    round(cv, 6),
            "change_1w":  round(c1w, 6) if c1w is not None else None,
            "change_1m":  round(c1m, 6) if c1m is not None else None,
            "percentile": round(pct * 100, 1) if pct is not None else None,
            "z_score":    round(z, 2) if z is not None else None,
        })
    return out


def _pct_color(pct):
    if pct >= 80 or pct <= 20:
        return "rgba(231,76,60,0.7)"
    if pct >= 65 or pct <= 35:
        return "rgba(243,156,18,0.7)"
    return "rgba(52,152,219,0.7)"


def _build_regime_charts(lookback_rows, stats):
    ts_cols = ["trade_date"] + [c for c, _ in _REGIME_METRICS]
    charts = []

    # Chart 1: ATM IV across tenors
    charts.append({
        "title": "ATM Implied Volatility",
        "rows": lookback_rows, "columns": ts_cols,
        "chart_config": {
            "type": "line",
            "datasets": [
                {"label": "7D", "labelsColumn": "trade_date", "yColumn": "iv_7d_atm",
                 "borderColor": "#2ecc71", "borderWidth": 1.5, "pointRadius": 0, "tension": 0.1},
                {"label": "30D", "labelsColumn": "trade_date", "yColumn": "iv_30d_atm",
                 "borderColor": "#3498db", "borderWidth": 1.5, "pointRadius": 0, "tension": 0.1},
                {"label": "90D", "labelsColumn": "trade_date", "yColumn": "iv_90d_atm",
                 "borderColor": "#9b59b6", "borderWidth": 1.2, "pointRadius": 0, "tension": 0.1,
                 "borderDash": [4, 3]},
                {"label": "180D", "labelsColumn": "trade_date", "yColumn": "iv_180d_atm",
                 "borderColor": "#e67e22", "borderWidth": 1.2, "pointRadius": 0, "tension": 0.1,
                 "borderDash": [4, 3]},
            ],
            "options": {"scales": {"y": {"title": {"display": True, "text": "IV (decimal)"}}}},
        },
    })

    # Chart 2: Put Skew across tenors
    charts.append({
        "title": "Put Skew (25P - ATM)",
        "rows": lookback_rows, "columns": ts_cols,
        "chart_config": {
            "type": "line",
            "datasets": [
                {"label": "7D", "labelsColumn": "trade_date", "yColumn": "skew_7d_25p_atm",
                 "borderColor": "#f39c12", "borderWidth": 1.5, "pointRadius": 0, "tension": 0.1},
                {"label": "30D", "labelsColumn": "trade_date", "yColumn": "skew_30d_25p_atm",
                 "borderColor": "#e74c3c", "borderWidth": 1.5, "pointRadius": 0, "tension": 0.1},
                {"label": "90D", "labelsColumn": "trade_date", "yColumn": "skew_90d_25p_atm",
                 "borderColor": "#9b59b6", "borderWidth": 1.2, "pointRadius": 0, "tension": 0.1,
                 "borderDash": [4, 3]},
            ],
            "options": {"scales": {"y": {"title": {"display": True, "text": "Skew (IV diff)"}}}},
        },
    })

    # Chart 3: Term Structure (two slope pairs + ratio)
    charts.append({
        "title": "Term Structure",
        "rows": lookback_rows, "columns": ts_cols,
        "chart_config": {
            "type": "line",
            "datasets": [
                {"label": "7-30D Slope", "labelsColumn": "trade_date",
                 "yColumn": "term_slope_7_30_atm",
                 "borderColor": "#9b59b6", "borderWidth": 1.5, "pointRadius": 0, "tension": 0.1,
                 "yAxisID": "y"},
                {"label": "30-90D Slope", "labelsColumn": "trade_date",
                 "yColumn": "term_slope_30_90_atm",
                 "borderColor": "#e74c3c", "borderWidth": 1.2, "pointRadius": 0, "tension": 0.1,
                 "borderDash": [4, 3], "yAxisID": "y"},
                {"label": "Ratio 7/30", "labelsColumn": "trade_date",
                 "yColumn": "term_ratio_7d_30d",
                 "borderColor": "#1abc9c", "borderWidth": 1.5, "pointRadius": 0, "tension": 0.1,
                 "yAxisID": "y1"},
            ],
            "options": {"scales": {
                "y":  {"title": {"display": True, "text": "Term Slope"}, "position": "left"},
                "y1": {"title": {"display": True, "text": "Term Ratio"}, "position": "right",
                       "grid": {"drawOnChartArea": False}},
            }},
        },
    })

    # Chart 4: Percentile bar chart
    pct_data = [
        {"metric": s["label"], "percentile": s["percentile"]}
        for s in stats if s["percentile"] is not None and s["metric"] != "spot"
    ]
    charts.append({
        "title": "Current Percentile Rank (vs Full History)",
        "rows": pct_data,
        "columns": ["metric", "percentile"],
        "chart_config": {
            "type": "bar",
            "datasets": [{
                "label": "Percentile",
                "labelsColumn": "metric", "yColumn": "percentile",
                "backgroundColor": [
                    _pct_color(s["percentile"]) for s in stats
                    if s["percentile"] is not None and s["metric"] != "spot"
                ],
                "borderWidth": 0,
            }],
            "options": {
                "scales": {
                    "y": {"title": {"display": True, "text": "Percentile"}, "min": 0, "max": 100},
                },
                "indexAxis": "y",
            },
        },
    })

    return charts


async def _regime_narrative(client, stats, question, stats_date=""):
    stats_text = json.dumps(stats, indent=2)
    msg = await client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=512,
        messages=[{
            "role": "user",
            "content": (
                f"User asked: {question!r}\n\n"
                f"SPX IV surface regime statistics as of {stats_date} (daily close, changes vs "
                f"1 week / 1 month ago, historical percentile rank, z-score):\n\n"
                f"{stats_text}\n\n"
                "Provide a concise 3-5 sentence regime summary. Cover:\n"
                "1. Overall vol level (high/low/normal) with percentile context\n"
                "2. Skew dynamics (steep/flat, tightening/widening)\n"
                "3. Term structure shape (contango/backwardation, steepening/flattening)\n"
                "4. Any notable moves in the past week\n"
                "Use professional vol trading language. Interpret — do not list raw numbers."
            ),
        }],
    )
    return msg.content[0].text.strip()


async def _handle_regime_analysis(client, question, pool) -> dict:
    async with pool.acquire() as conn:
        async with conn.transaction(readonly=True):
            lb_pg   = await conn.fetch(_regime_ts_sql(90))
            full_pg = await conn.fetch(_regime_full_sql())

    _, lb_rows   = _serialize(lb_pg)
    _, full_rows = _serialize(full_pg)

    if not lb_rows:
        return {"response_type": "analysis", "error": "No data in surface_metrics_core",
                "summary": None, "stats": [], "charts": []}

    stats      = _compute_regime_stats(lb_rows, full_rows)
    # Include the snapshot date in the stats payload for Claude
    stats_date = lb_rows[-1].get("trade_date", "unknown")
    # Find actual stats date (may differ if latest row lacks IV)
    iv_check = "iv_30d_atm"
    for r in reversed(lb_rows):
        if r.get(iv_check) is not None:
            stats_date = r.get("trade_date", stats_date)
            break
    narrative  = await _regime_narrative(client, stats, question, stats_date)
    charts     = _build_regime_charts(lb_rows, stats)

    return {
        "response_type": "analysis",
        "error":   None,
        "summary": narrative,
        "stats":   stats,
        "charts":  charts,
    }


# ── Request models ────────────────────────────────────────────────────────────

class HistoryTurn(BaseModel):
    question: str
    sql:      Optional[str] = None
    summary:  Optional[str] = None
    error:    Optional[str] = None


class QueryRequest(BaseModel):
    question: str
    history:  list[HistoryTurn] = []


# ── Query log ────────────────────────────────────────────────────────────────

_log_table_ready = False


async def _ensure_log_table(pool):
    global _log_table_ready
    if _log_table_ready:
        return
    try:
        async with pool.acquire() as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS ai_explorer_log (
                    id          SERIAL PRIMARY KEY,
                    ts          TIMESTAMPTZ DEFAULT NOW(),
                    question    TEXT NOT NULL,
                    response_type TEXT,
                    sql         TEXT,
                    summary     TEXT,
                    error       TEXT
                )
            """)
        _log_table_ready = True
    except Exception:
        pass


async def _log_query(pool, question, response):
    try:
        async with pool.acquire() as conn:
            await conn.execute(
                """INSERT INTO ai_explorer_log
                   (question, response_type, sql, summary, error)
                   VALUES ($1, $2, $3, $4, $5)""",
                question,
                response.get("response_type"),
                response.get("sql"),
                response.get("summary"),
                response.get("error"),
            )
    except Exception:
        pass  # logging must never break the response


# ── Endpoint ──────────────────────────────────────────────────────────────────

@router.post("/query")
async def ai_query(req: QueryRequest, pool=Depends(get_pool)) -> dict:
    question = req.question.strip()
    if not question:
        raise HTTPException(400, "question must not be empty")

    client = _get_client()
    sql = None  # kept in scope so catch-all can return it

    try:
        await _ensure_log_table(pool)

        # 0. Classify: broad analysis vs direct SQL query
        req_type = await _classify_request(client, question)
        if req_type == "analysis":
            result = await _handle_regime_analysis(client, question, pool)
            await _log_query(pool, question, result)
            return result

        # ── Direct path (existing, unchanged) ────────────────────────

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
            elif turn.summary:
                asst = f"Analysis result: {turn.summary}"
            else:
                asst = "No result."
            messages.append({"role": "assistant", "content": asst})
        messages.append({"role": "user", "content": question})

        # 2. Generate SQL via Claude Sonnet (system prompt is cached)
        sql_msg = await client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=2048,
            system=_SYSTEM_SQL,
            messages=messages,
        )
        raw_text = _strip_fences(sql_msg.content[0].text)

        # Check for text-only response (MODE B: follow-ups, explanations)
        if raw_text.strip().upper().startswith('TEXT:'):
            text_body = raw_text.strip()[5:].strip()
            return {"response_type": "direct", "sql": None, "error": None,
                    "columns": [], "rows": [], "summary": text_body,
                    "chart_hint": None, "chart_config": None}

        # Parse: line 1 = chart hint, then SQL, then optional ---CHART--- config
        chart_hint = None
        chart_config = None
        lines = raw_text.strip().split('\n', 1)
        if len(lines) == 2 and lines[0].strip().lower() in ('line', 'bar', 'scatter', 'none'):
            chart_hint = lines[0].strip().lower()
            if chart_hint == 'none':
                chart_hint = None
            rest = lines[1]
        else:
            rest = raw_text.strip()

        if '---CHART---' in rest:
            parts = rest.split('---CHART---', 1)
            raw_sql = parts[0].strip()
            config_text = parts[1].strip()
            config_text = re.sub(r'^```(?:json)?\n?', '', config_text)
            config_text = re.sub(r'\n?```\s*$', '', config_text)
            try:
                chart_config = json.loads(config_text)
            except json.JSONDecodeError:
                pass
        else:
            raw_sql = rest.strip()

        # 2. Validate + enforce row cap
        # If Claude produced text instead of SQL (without TEXT: prefix), return as summary
        try:
            sql = _validate_sql(raw_sql)
            sql = _enforce_limit(sql)
        except ValueError:
            if not re.match(r'^\s*(?:WITH\b|SELECT\b)', raw_sql, re.IGNORECASE):
                # Not SQL at all — return Claude's response as text
                return {"response_type": "direct", "sql": None, "error": None,
                        "columns": [], "rows": [],
                        "summary": raw_text.strip(),
                        "chart_hint": None, "chart_config": None}
            return {"response_type": "direct", "sql": raw_sql, "error": "Query validation failed",
                    "columns": [], "rows": [], "summary": None}

        # 3. Execute in a read-only transaction
        try:
            async with pool.acquire() as conn:
                async with conn.transaction(readonly=True):
                    pg_rows = await conn.fetch(sql)
        except asyncpg.PostgresError as e:
            return {"response_type": "direct", "sql": sql, "error": f"Database error: {e}", "columns": [], "rows": [], "summary": None}

        columns, rows = _serialize(pg_rows)

        if not rows:
            return {"response_type": "direct", "sql": sql, "error": None, "columns": columns, "rows": [], "summary": "The query returned no results."}

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

        result = {"response_type": "direct", "sql": sql, "error": None,
                  "columns": columns, "rows": rows, "summary": summary,
                  "chart_hint": chart_hint, "chart_config": chart_config}
        await _log_query(pool, question, result)
        return result

    except Exception as e:
        result = {
            "response_type": "direct",
            "sql": sql,
            "error": f"{type(e).__name__}: {e}",
            "columns": [], "rows": [], "summary": None,
        }
        await _log_query(pool, question, result)
        return result
