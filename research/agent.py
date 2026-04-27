"""
Two-phase research agent.

Phase 1 (no AI tokens): fetch all data into memory, run full correlation
matrix for every ticker × x_col × y_col, save results + scatter charts.

Phase 2 (Claude agent): Claude receives the full correlation summary and
uses tool calls only for depth — decile analysis, yearly consistency,
equity curves, regression — focused on the most promising signals.
"""
import json
from datetime import date as _date
import asyncpg
import anthropic

from research import blocks, charts, db as rdb

_OI_TABLES = {"daily_features", "option_oi_surface", "underlying_ohlc"}

# ── Phase 1 helpers ───────────────────────────────────────────────────────────

async def _fetch_cache(pool: asyncpg.Pool, table: str,
                       tickers: list[str], x_cols: list[str], y_cols: list[str],
                       date_from: str | None, date_to: str | None,
                       log) -> tuple[dict[str, list[dict]], list[str]]:
    """Fetch all required columns once per ticker. Returns (cache, errors)."""
    needed = list(dict.fromkeys(["trade_date"] + x_cols + y_cols))
    col_list = ", ".join(needed)
    cache, errors = {}, []
    keys = tickers if tickers else [None]

    async with pool.acquire() as conn:
        for ticker in keys:
            conditions, params, p = [], [], 1
            if ticker:
                conditions.append(f"ticker = ${p}"); params.append(ticker); p += 1
            if date_from:
                df_val = _date.fromisoformat(date_from) if isinstance(date_from, str) else date_from
                conditions.append(f"trade_date >= ${p}"); params.append(df_val); p += 1
            if date_to:
                dt_val = _date.fromisoformat(date_to) if isinstance(date_to, str) else date_to
                conditions.append(f"trade_date <= ${p}"); params.append(dt_val); p += 1

            where = ("WHERE " + " AND ".join(conditions)) if conditions else ""
            sql = f"SELECT {col_list} FROM {table} {where} ORDER BY trade_date"
            try:
                fetched = await conn.fetch(sql, *params)
            except Exception as exc:
                msg = f"ticker={ticker or 'all'}: {exc}"
                errors.append(msg)
                log(f"  FETCH ERROR {msg}")
                continue

            if fetched:
                row_list = [dict(r) for r in fetched]
                cache[ticker or "_all"] = row_list
                log(f"  Cached {len(row_list):,} rows for {ticker or 'all'}")
            else:
                msg = f"ticker={ticker or 'all'}: query returned 0 rows (table={table})"
                errors.append(msg)
                log(f"  No data found for {ticker or 'all'}")

    return cache, errors


def _build_fallback_summary(corr_results: list[dict]) -> str:
    """Fallback when finish() is called with an empty summary."""
    if not corr_results:
        return "Analysis complete. No correlations were computed."
    valid = [r for r in corr_results if "error" not in r]
    strong   = sorted([r for r in valid if abs(r.get("pearson_r") or 0) > 0.06],
                      key=lambda x: abs(x.get("pearson_r") or 0), reverse=True)
    moderate = sorted([r for r in valid if 0.03 < abs(r.get("pearson_r") or 0) <= 0.06],
                      key=lambda x: abs(x.get("pearson_r") or 0), reverse=True)
    lines = [f"Analysis complete. {len(valid)} correlations computed.\n"]
    if strong:
        lines.append("Strong signals (|r| > 0.06):")
        for r in strong[:8]:
            lines.append(f"  {r.get('ticker') or 'all'} | {r.get('x_col')} → {r.get('y_col')}: "
                         f"r={r.get('pearson_r', 0):.4f}, n={r.get('n', 0)}")
    if moderate:
        lines.append("\nModerate signals (0.03 < |r| ≤ 0.06):")
        for r in moderate[:5]:
            lines.append(f"  {r.get('ticker') or 'all'} | {r.get('x_col')} → {r.get('y_col')}: "
                         f"r={r.get('pearson_r', 0):.4f}, n={r.get('n', 0)}")
    if not strong and not moderate:
        lines.append("No signals above the |r| > 0.03 threshold found.")
    return "\n".join(lines)


def _format_corr_table(corr_results: list[dict]) -> str:
    """Format correlation matrix as a readable text table for Claude."""
    header = f"{'Ticker':<8} {'Feature':<42} {'Outcome':<16} {'Pearson r':>9} {'Spearman r':>10} {'N':>6}  Signal"
    sep = "-" * len(header)
    lines = [header, sep]
    for r in sorted(corr_results, key=lambda x: abs(x.get("pearson_r") or 0), reverse=True):
        if "error" in r:
            continue
        pr = r.get("pearson_r") or 0
        note = "***" if abs(pr) > 0.12 else "**" if abs(pr) > 0.06 else "*" if abs(pr) > 0.03 else ""
        lines.append(
            f"{(r.get('ticker') or 'all'):<8} {r.get('x_col',''):<42} {r.get('y_col',''):<16} "
            f"{pr:>9.4f} {(r.get('spearman_r') or 0):>10.4f} {r.get('n',0):>6}  {note}"
        )
    lines.append(sep)
    lines.append("*** |r|>0.12  ** |r|>0.06  * |r|>0.03")
    return "\n".join(lines)


async def _phase1(main_pool: asyncpg.Pool, data_pool: asyncpg.Pool,
                  run_id: str, table: str,
                  tickers: list[str], x_cols: list[str], y_cols: list[str],
                  date_from, date_to, log) -> tuple[dict, list[dict], list[str]]:
    """
    Fetch data, run all correlations, save scatter charts.
    Returns (data_cache, corr_results, fetch_errors).
    """
    log("Phase 1: fetching data…")
    cache, fetch_errors = await _fetch_cache(data_pool, table, tickers, x_cols, y_cols,
                                             date_from, date_to, log)
    if not cache:
        return {}, [], fetch_errors

    log(f"Phase 1: running correlation matrix ({len(cache)} ticker(s) × {len(x_cols)} features × {len(y_cols)} outcomes)…")
    corr_results = []

    async with main_pool.acquire() as conn:
        for cache_key, rows in cache.items():
            ticker = None if cache_key == "_all" else cache_key
            cols_available = set(rows[0].keys()) if rows else set()
            for x_col in x_cols:
                if x_col not in cols_available:
                    log(f"  SKIP {x_col} — not in data for {cache_key}")
                    continue
                for y_col in y_cols:
                    if y_col not in cols_available:
                        continue
                    result = blocks.correlation_from_rows(rows, x_col, y_col, ticker)
                    await rdb.save_result(conn, run_id, "correlation",
                                         x_col, y_col, result, ticker)
                    corr_results.append(result)

                    # Scatter chart saved once per (ticker, x, y)
                    scatter = blocks.scatter_sample_from_rows(rows, x_col, y_col, ticker=ticker)
                    if scatter.get("points"):
                        png = charts.scatter_chart(scatter)
                        if png:
                            await rdb.save_chart(
                                conn, run_id, "scatter",
                                f"Scatter {ticker or 'all'} | {x_col} → {y_col}",
                                png, ticker, x_col, y_col,
                            )

    log(f"Phase 1 complete — {len(corr_results)} correlations computed.")
    return cache, corr_results, []


# ── Phase 2: Claude agent tools ───────────────────────────────────────────────

_SYSTEM = """\
You are a quantitative research agent. Phase 1 (data fetch + correlation screening) is already done.
You are given the full correlation matrix. Your job is to drill into the promising signals.

Guidelines:
- Investigate combinations with |pearson_r| > 0.03 or any interesting pattern
- For each: run decile_analysis, yearly_consistency, then equity_curve if spread looks tradeable
- Use regression to test multi-factor combinations if you find multiple correlated features
- Skip combinations with |r| < 0.02 unless something else makes them interesting
- When done, call finish() with a thorough narrative

finish() summary must include:
1. Strongest signals (spread, consistency %, final equity, max drawdown)
2. Weak/null results
3. Year-over-year stability
4. Key caveats (sample size, regime concentration)
5. Suggested next steps
"""

_TOOLS = [
    {
        "name": "run_decile_analysis",
        "description": "Split feature into deciles, compute avg/median return, win rate, sample count per decile. Returns top-bottom spread.",
        "input_schema": {
            "type": "object",
            "properties": {
                "feature_col": {"type": "string"},
                "outcome_col": {"type": "string"},
                "ticker":      {"type": "string", "description": "Ticker (must be in the dataset)"},
                "n_deciles":   {"type": "integer", "description": "Number of buckets, default 10"},
            },
            "required": ["feature_col", "outcome_col"],
        },
    },
    {
        "name": "run_yearly_consistency",
        "description": "Check whether top decile beats bottom decile year-by-year. High % = stable signal.",
        "input_schema": {
            "type": "object",
            "properties": {
                "feature_col": {"type": "string"},
                "outcome_col": {"type": "string"},
                "ticker":      {"type": "string"},
            },
            "required": ["feature_col", "outcome_col"],
        },
    },
    {
        "name": "run_equity_curve",
        "description": "Non-overlapping equity curve for top or bottom decile. Returns cumulative return, max drawdown, win rate.",
        "input_schema": {
            "type": "object",
            "properties": {
                "feature_col": {"type": "string"},
                "outcome_col": {"type": "string"},
                "ticker":      {"type": "string"},
                "which":       {"type": "string", "enum": ["top", "bottom"]},
            },
            "required": ["feature_col", "outcome_col"],
        },
    },
    {
        "name": "run_regression",
        "description": "OLS regression of outcome on multiple features. Returns R² and coefficients.",
        "input_schema": {
            "type": "object",
            "properties": {
                "x_cols": {"type": "array", "items": {"type": "string"}},
                "y_col":  {"type": "string"},
                "ticker": {"type": "string"},
            },
            "required": ["x_cols", "y_col"],
        },
    },
    {
        "name": "finish",
        "description": "Call when research is complete. Write a comprehensive findings narrative.",
        "input_schema": {
            "type": "object",
            "properties": {
                "summary": {"type": "string", "description": "Full findings narrative (500-1000 words)"},
            },
            "required": ["summary"],
        },
    },
]


async def _dispatch(tool_name: str, inputs: dict,
                    main_conn: asyncpg.Connection,
                    data_cache: dict[str, list[dict]],
                    run_id: str, log) -> dict:
    """Execute one tool call using in-memory row lists. Saves results to DB."""
    ticker = inputs.get("ticker") or None
    cache_key = ticker if ticker else "_all"
    rows = data_cache.get(cache_key)

    if rows is None:
        available = [k for k in data_cache if k != "_all"]
        return {"error": f"No cached data for ticker='{ticker}'. "
                         f"Available: {available or ['_all (no ticker filter)']}"}

    if tool_name == "run_decile_analysis":
        n = inputs.get("n_deciles", 10)
        result = blocks.decile_analysis_from_rows(
            rows, inputs["feature_col"], inputs["outcome_col"], n, ticker)
        await rdb.save_result(main_conn, run_id, "decile",
                              inputs["feature_col"], inputs["outcome_col"], result, ticker)
        png = charts.decile_bar_chart(result)
        if png:
            await rdb.save_chart(
                main_conn, run_id, "decile_bars",
                f"Decile {ticker or 'all'} | {inputs['feature_col']} → {inputs['outcome_col']}",
                png, ticker, inputs["feature_col"], inputs["outcome_col"],
            )
        spread = result.get("top_bottom_spread")
        log(f"  decile {ticker or 'all'} {inputs['feature_col']}→{inputs['outcome_col']}: spread={spread}")
        return {
            "top_bottom_spread": spread,
            "top_decile": result["deciles"][-1] if result.get("deciles") else None,
            "bot_decile": result["deciles"][0]  if result.get("deciles") else None,
            "n_total":    sum(d["n"] for d in result.get("deciles", [])),
        }

    elif tool_name == "run_yearly_consistency":
        result = blocks.yearly_consistency_from_rows(
            rows, inputs["feature_col"], inputs["outcome_col"], ticker=ticker)
        await rdb.save_result(main_conn, run_id, "yearly_consistency",
                              inputs["feature_col"], inputs["outcome_col"], result, ticker)
        png = charts.yearly_consistency_chart(result)
        if png:
            await rdb.save_chart(
                main_conn, run_id, "yearly_consistency",
                f"Yearly {ticker or 'all'} | {inputs['feature_col']} → {inputs['outcome_col']}",
                png, ticker, inputs["feature_col"], inputs["outcome_col"],
            )
        log(f"  yearly {ticker or 'all'}: {result.get('consistency_pct')}% "
            f"({result.get('wins')}/{result.get('total_years')} years)")
        return {
            "consistency_pct": result.get("consistency_pct"),
            "wins":            result.get("wins"),
            "total_years":     result.get("total_years"),
            "by_year":         [{"year": y["year"], "top_avg": y["top_avg"], "beats": y["top_beats"]}
                                for y in result.get("years", [])],
        }

    elif tool_name == "run_equity_curve":
        which = inputs.get("which", "top")
        result = blocks.equity_curve_from_rows(
            rows, inputs["feature_col"], inputs["outcome_col"], which, ticker=ticker)
        await rdb.save_result(main_conn, run_id, f"equity_curve_{which}",
                              inputs["feature_col"], inputs["outcome_col"], result, ticker)
        await rdb.save_series(main_conn, run_id, inputs["feature_col"],
                              f"equity_curve_{which}", result.get("points", []),
                              ticker=ticker, y_col=inputs["outcome_col"])

        # Always generate both top+bottom for the chart
        if which == "top":
            bot = blocks.equity_curve_from_rows(
                rows, inputs["feature_col"], inputs["outcome_col"], "bottom", ticker=ticker)
            png = charts.equity_curve_chart(result, bot)
        else:
            png = charts.equity_curve_chart(result)

        if png:
            await rdb.save_chart(
                main_conn, run_id, "equity_curve",
                f"Equity {ticker or 'all'} | {inputs['feature_col']} → {inputs['outcome_col']}",
                png, ticker, inputs["feature_col"], inputs["outcome_col"],
            )
        log(f"  equity {which} {ticker or 'all'}: final={result.get('final_equity')}, "
            f"maxDD={result.get('max_drawdown')}, n={result.get('n_trades')}")
        return {
            "n_trades":     result.get("n_trades"),
            "final_equity": result.get("final_equity"),
            "max_drawdown": result.get("max_drawdown"),
            "avg_ret":      result.get("avg_ret"),
            "win_rate":     result.get("win_rate"),
        }

    elif tool_name == "run_regression":
        result = blocks.regression_from_rows(rows, inputs["x_cols"], inputs["y_col"], ticker)
        await rdb.save_result(main_conn, run_id, "regression",
                              "+".join(inputs["x_cols"]), inputs["y_col"], result, ticker)
        log(f"  regression {ticker or 'all'} → {inputs['y_col']}: R²={result.get('r_squared')}")
        return result

    return {"error": f"unknown tool: {tool_name}"}


# ── Main entry point ──────────────────────────────────────────────────────────

async def run_agent(
    main_pool: asyncpg.Pool,
    oi_pool:   asyncpg.Pool,
    run_id:    str,
    question:  str,
    config:    dict,
    model:     str = "claude-sonnet-4-6",
    max_tool_calls: int = 60,
    log=print,
) -> str:
    client = anthropic.AsyncAnthropic()

    table    = config.get("table", "daily_features")
    tickers  = config.get("tickers")  or []
    x_cols   = config.get("x_columns") or []
    y_cols   = config.get("y_columns") or []
    date_from = config.get("date_from") or None
    date_to   = config.get("date_to")   or None

    data_pool = oi_pool if table in _OI_TABLES else main_pool

    # ── Phase 1 ───────────────────────────────────────────────────────────────
    data_cache, corr_results, fetch_errors = await _phase1(
        main_pool, data_pool, run_id, table,
        tickers, x_cols, y_cols, date_from, date_to, log,
    )

    if not data_cache:
        detail = "\n".join(fetch_errors) if fetch_errors else "query returned 0 rows"
        raise ValueError(f"Phase 1 failed — no data loaded.\n{detail}")

    # Generate correlation heatmap if multiple tickers
    if len(data_cache) > 1 and x_cols and y_cols:
        png = charts.correlation_heatmap(corr_results, tickers or list(data_cache.keys()),
                                         x_cols, y_cols)
        if png:
            async with main_pool.acquire() as conn:
                await rdb.save_chart(conn, run_id, "correlation_heatmap",
                                     "Pearson Correlation Heatmap", png)

    corr_summary = _format_corr_table(corr_results)

    # ── Phase 2 ───────────────────────────────────────────────────────────────
    df_from = config.get("date_from") or "earliest available"
    df_to   = config.get("date_to")   or "today"

    user_msg = (
        f"Research question: {question}\n\n"
        f"Data loaded into memory:\n"
        f"  Table:    {table}\n"
        f"  Tickers:  {', '.join(tickers) if tickers else 'all'}\n"
        f"  Features: {', '.join(x_cols)}\n"
        f"  Outcomes: {', '.join(y_cols)}\n"
        f"  Period:   {df_from} → {df_to}\n\n"
        f"Phase 1 complete. Full correlation matrix (sorted by |Pearson r|):\n\n"
        f"{corr_summary}\n\n"
        f"Now drill into promising signals. Do NOT re-run correlations — they are already saved. "
        f"Use run_decile_analysis, run_yearly_consistency, run_equity_curve, and run_regression. "
        f"Call finish() when done."
    )

    messages = [{"role": "user", "content": user_msg}]
    tool_calls_made = 0

    async with main_pool.acquire() as main_conn:
        while tool_calls_made < max_tool_calls:
            resp = await client.messages.create(
                model=model,
                max_tokens=4096,
                system=_SYSTEM,
                tools=_TOOLS,
                messages=messages,
            )

            assistant_content = []
            finish_summary = None

            for block in resp.content:
                assistant_content.append(block)
                if block.type != "tool_use":
                    continue

                tool_calls_made += 1
                inp = block.input
                log(f"[{tool_calls_made}/{max_tool_calls}] {block.name}("
                    f"{json.dumps({k: v for k, v in inp.items()}, default=str)[:120]})")

                if block.name == "finish":
                    finish_summary = inp.get("summary", "") or ""
                    if not finish_summary.strip():
                        log("  finish() called with empty summary — building fallback")
                        finish_summary = _build_fallback_summary(corr_results)
                    break

                try:
                    result = await _dispatch(block.name, inp, main_conn,
                                            data_cache, run_id, log)
                except Exception as exc:
                    result = {"error": str(exc)}
                    log(f"  ERROR: {exc}")

                messages.append({"role": "assistant", "content": assistant_content})
                messages.append({
                    "role": "user",
                    "content": [{
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": json.dumps(result, default=str),
                    }],
                })
                assistant_content = []
                break  # restart loop

            if finish_summary is not None:
                log("Agent called finish()")
                return finish_summary

            if resp.stop_reason == "end_turn" and not any(
                b.type == "tool_use" for b in resp.content
            ):
                text = " ".join(b.text for b in resp.content if hasattr(b, "text") and b.text)
                return text or "Agent completed without a summary."

    return (f"Agent reached the {max_tool_calls} tool-call limit. "
            f"Partial results are saved. Try increasing max_tool_calls or reducing scope.")
