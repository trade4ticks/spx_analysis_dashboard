"""
Claude tool-calling agent loop.
The agent is given a research question + data config, then autonomously
calls analysis building blocks until it calls finish().
All results are saved to DB as they are produced.
"""
import json
import asyncpg
import anthropic

from research import blocks, charts, db as rdb

_OI_TABLES = {"daily_features", "option_oi_surface", "underlying_ohlc"}

_SYSTEM = """\
You are a quantitative research agent analyzing financial market data.

Your job: systematically investigate the research question using the tools provided.
All results are automatically saved to the database as you go.

Work methodically:
1. Run correlations first — quick signal strength check across all ticker × x × y combinations.
2. For combinations with |pearson_r| > 0.05 or any notable pattern, run decile analysis.
3. For the strongest decile results (top-bottom spread > 0.003 absolute return), run yearly
   consistency and equity curve.
4. Use scatter only to confirm relationship shape when something looks unusual.
5. Use regression only if you want to test multi-factor predictiveness.
6. When done, call finish() with a thorough narrative.

Be honest: report null results. A genuine null finding is valuable.
Report in your finish() summary:
- Strongest signals (quantitative: spread, consistency %, equity final value, max drawdown)
- Tickers / horizons where signals are absent or fragile
- Yearly consistency — any regime-dependence
- Key caveats (sample size, overlap with other factors)
- Suggested next steps
"""

_TOOLS = [
    {
        "name": "run_correlation",
        "description": (
            "Pearson + Spearman correlation between a feature and an outcome column. "
            "Run this first for every ticker × x_col × y_col combination."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "table":     {"type": "string", "description": "Source table name"},
                "x_col":     {"type": "string", "description": "Feature / predictor column"},
                "y_col":     {"type": "string", "description": "Outcome column"},
                "ticker":    {"type": "string", "description": "Ticker filter (omit = all tickers)"},
                "date_from": {"type": "string", "description": "YYYY-MM-DD start (optional)"},
                "date_to":   {"type": "string", "description": "YYYY-MM-DD end (optional)"},
            },
            "required": ["table", "x_col", "y_col"],
        },
    },
    {
        "name": "run_decile_analysis",
        "description": (
            "Split feature into deciles, compute avg/median return, win rate, "
            "sample count, std dev per decile. Returns top-bottom spread."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "table":       {"type": "string"},
                "feature_col": {"type": "string"},
                "outcome_col": {"type": "string"},
                "ticker":      {"type": "string"},
                "n_deciles":   {"type": "integer", "description": "Number of quantile buckets (default 10)"},
                "date_from":   {"type": "string"},
                "date_to":     {"type": "string"},
            },
            "required": ["table", "feature_col", "outcome_col"],
        },
    },
    {
        "name": "run_yearly_consistency",
        "description": (
            "Check whether top decile outperforms bottom decile year-by-year. "
            "High consistency % = robust signal."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "table":       {"type": "string"},
                "feature_col": {"type": "string"},
                "outcome_col": {"type": "string"},
                "ticker":      {"type": "string"},
                "date_from":   {"type": "string"},
                "date_to":     {"type": "string"},
            },
            "required": ["table", "feature_col", "outcome_col"],
        },
    },
    {
        "name": "run_equity_curve",
        "description": (
            "Non-overlapping equity curve for top or bottom decile. "
            "Returns cumulative return series, max drawdown, win rate."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "table":       {"type": "string"},
                "feature_col": {"type": "string"},
                "outcome_col": {"type": "string"},
                "ticker":      {"type": "string"},
                "which":       {"type": "string", "enum": ["top", "bottom"],
                                "description": "Which extreme decile to simulate"},
                "date_from":   {"type": "string"},
                "date_to":     {"type": "string"},
            },
            "required": ["table", "feature_col", "outcome_col"],
        },
    },
    {
        "name": "run_scatter",
        "description": "Sample (date, x, y) pairs for scatter visualization.",
        "input_schema": {
            "type": "object",
            "properties": {
                "table":     {"type": "string"},
                "x_col":     {"type": "string"},
                "y_col":     {"type": "string"},
                "ticker":    {"type": "string"},
                "date_from": {"type": "string"},
                "date_to":   {"type": "string"},
            },
            "required": ["table", "x_col", "y_col"],
        },
    },
    {
        "name": "run_regression",
        "description": "OLS regression of outcome on one or more features. Returns R² and coefficients.",
        "input_schema": {
            "type": "object",
            "properties": {
                "table":     {"type": "string"},
                "x_cols":    {"type": "array", "items": {"type": "string"}},
                "y_col":     {"type": "string"},
                "ticker":    {"type": "string"},
                "date_from": {"type": "string"},
                "date_to":   {"type": "string"},
            },
            "required": ["table", "x_cols", "y_col"],
        },
    },
    {
        "name": "finish",
        "description": "Call when research is complete. Provide comprehensive findings narrative.",
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
                    oi_conn: asyncpg.Connection,
                    run_id: str,
                    log) -> dict:
    """Execute one tool call, save result to DB, return condensed result for Claude."""
    table = inputs.get("table", "")
    conn = oi_conn if table in _OI_TABLES else main_conn
    ticker = inputs.get("ticker") or None
    date_from = inputs.get("date_from") or None
    date_to = inputs.get("date_to") or None

    if tool_name == "run_correlation":
        result = await blocks.compute_correlation(
            conn, table, inputs["x_col"], inputs["y_col"],
            ticker=ticker, date_from=date_from, date_to=date_to,
        )
        await rdb.save_result(main_conn, run_id, "correlation",
                              inputs["x_col"], inputs["y_col"], result, ticker)
        # Generate scatter chart to accompany this combination
        scatter = await blocks.compute_scatter_sample(
            conn, table, inputs["x_col"], inputs["y_col"],
            ticker=ticker, date_from=date_from, date_to=date_to,
        )
        if scatter.get("points"):
            png = charts.scatter_chart(scatter)
            if png:
                await rdb.save_chart(main_conn, run_id, "scatter",
                                     f"Scatter {ticker or 'all'} {inputs['x_col']} → {inputs['y_col']}",
                                     png, ticker, inputs["x_col"], inputs["y_col"])
        log(f"  correlation {ticker or 'all'} {inputs['x_col']}→{inputs['y_col']}: "
            f"r={result.get('pearson_r')}, n={result.get('n')}")
        return result

    elif tool_name == "run_decile_analysis":
        n = inputs.get("n_deciles", 10)
        result = await blocks.compute_decile_analysis(
            conn, table, inputs["feature_col"], inputs["outcome_col"],
            ticker=ticker, n_deciles=n, date_from=date_from, date_to=date_to,
        )
        await rdb.save_result(main_conn, run_id, "decile",
                              inputs["feature_col"], inputs["outcome_col"], result, ticker)
        png = charts.decile_bar_chart(result)
        if png:
            await rdb.save_chart(main_conn, run_id, "decile_bars",
                                 f"Decile {ticker or 'all'} {inputs['feature_col']} → {inputs['outcome_col']}",
                                 png, ticker, inputs["feature_col"], inputs["outcome_col"])
        spread = result.get("top_bottom_spread")
        log(f"  decile {ticker or 'all'} {inputs['feature_col']}→{inputs['outcome_col']}: "
            f"spread={spread}")
        # Return compact summary to Claude (not full decile rows)
        return {
            "top_bottom_spread": spread,
            "top_decile":  result["deciles"][-1] if result.get("deciles") else None,
            "bot_decile":  result["deciles"][0]  if result.get("deciles") else None,
            "n_total":     sum(d["n"] for d in result.get("deciles", [])),
        }

    elif tool_name == "run_yearly_consistency":
        result = await blocks.compute_yearly_consistency(
            conn, table, inputs["feature_col"], inputs["outcome_col"],
            ticker=ticker, date_from=date_from, date_to=date_to,
        )
        await rdb.save_result(main_conn, run_id, "yearly_consistency",
                              inputs["feature_col"], inputs["outcome_col"], result, ticker)
        png = charts.yearly_consistency_chart(result)
        if png:
            await rdb.save_chart(main_conn, run_id, "yearly_consistency",
                                 f"Yearly {ticker or 'all'} {inputs['feature_col']} → {inputs['outcome_col']}",
                                 png, ticker, inputs["feature_col"], inputs["outcome_col"])
        log(f"  yearly {ticker or 'all'}: consistency={result.get('consistency_pct')}% "
            f"({result.get('wins')}/{result.get('total_years')} years)")
        return {
            "consistency_pct": result.get("consistency_pct"),
            "wins":            result.get("wins"),
            "total_years":     result.get("total_years"),
            "years_summary":   [
                {"year": y["year"], "top_avg": y["top_avg"], "top_beats": y["top_beats"]}
                for y in result.get("years", [])
            ],
        }

    elif tool_name == "run_equity_curve":
        which = inputs.get("which", "top")
        result = await blocks.compute_equity_curve(
            conn, table, inputs["feature_col"], inputs["outcome_col"],
            ticker=ticker, which=which, date_from=date_from, date_to=date_to,
        )
        await rdb.save_result(main_conn, run_id, f"equity_curve_{which}",
                              inputs["feature_col"], inputs["outcome_col"], result, ticker)
        await rdb.save_series(main_conn, run_id, inputs["feature_col"],
                              f"equity_curve_{which}", result.get("points", []),
                              ticker=ticker, y_col=inputs["outcome_col"])
        # Also try to get bottom curve for combined chart if this is top
        if which == "top":
            bot = await blocks.compute_equity_curve(
                conn, table, inputs["feature_col"], inputs["outcome_col"],
                ticker=ticker, which="bottom", date_from=date_from, date_to=date_to,
            )
            png = charts.equity_curve_chart(result, bot)
        else:
            png = charts.equity_curve_chart(result)
        if png:
            await rdb.save_chart(main_conn, run_id, "equity_curve",
                                 f"Equity {ticker or 'all'} {inputs['feature_col']} → {inputs['outcome_col']}",
                                 png, ticker, inputs["feature_col"], inputs["outcome_col"])
        log(f"  equity {which} {ticker or 'all'}: "
            f"final={result.get('final_equity')}, maxDD={result.get('max_drawdown')}, "
            f"n={result.get('n_trades')}")
        return {
            "n_trades":     result.get("n_trades"),
            "final_equity": result.get("final_equity"),
            "max_drawdown": result.get("max_drawdown"),
            "avg_ret":      result.get("avg_ret"),
            "win_rate":     result.get("win_rate"),
        }

    elif tool_name == "run_scatter":
        result = await blocks.compute_scatter_sample(
            conn, table, inputs["x_col"], inputs["y_col"],
            ticker=ticker, date_from=date_from, date_to=date_to,
        )
        await rdb.save_result(main_conn, run_id, "scatter",
                              inputs["x_col"], inputs["y_col"], {"n": result["n"]}, ticker)
        png = charts.scatter_chart(result)
        if png:
            await rdb.save_chart(main_conn, run_id, "scatter",
                                 f"Scatter {ticker or 'all'} {inputs['x_col']} → {inputs['y_col']}",
                                 png, ticker, inputs["x_col"], inputs["y_col"])
        log(f"  scatter {ticker or 'all'} {inputs['x_col']}→{inputs['y_col']}: n={result['n']}")
        return {"n": result["n"], "saved": True}

    elif tool_name == "run_regression":
        result = await blocks.compute_regression(
            conn, table, inputs["x_cols"], inputs["y_col"],
            ticker=ticker, date_from=date_from, date_to=date_to,
        )
        await rdb.save_result(main_conn, run_id, "regression",
                              "+".join(inputs["x_cols"]), inputs["y_col"], result, ticker)
        log(f"  regression {ticker or 'all'} → {inputs['y_col']}: R²={result.get('r_squared')}")
        return result

    return {"error": f"unknown tool: {tool_name}"}


async def run_agent(
    main_pool: asyncpg.Pool,
    oi_pool: asyncpg.Pool,
    run_id: str,
    question: str,
    config: dict,
    model: str = "claude-sonnet-4-6",
    max_tool_calls: int = 60,
    log=print,
) -> str:
    """
    Run the research agent loop. Returns the final AI summary string.
    Saves all intermediate results to DB via main_pool.
    """
    client = anthropic.AsyncAnthropic()

    # Build the initial user message from config
    tickers = config.get("tickers", [])
    x_cols  = config.get("x_columns", [])
    y_cols  = config.get("y_columns", [])
    table   = config.get("table", "daily_features")
    df      = config.get("date_from") or "earliest available"
    dt      = config.get("date_to")   or "today"

    user_msg = (
        f"Research question: {question}\n\n"
        f"Available data:\n"
        f"  Table:    {table}\n"
        f"  Tickers:  {', '.join(tickers) if tickers else 'all'}\n"
        f"  Features (x): {', '.join(x_cols)}\n"
        f"  Outcomes (y): {', '.join(y_cols)}\n"
        f"  Date range:   {df} → {dt}\n\n"
        "Please analyze systematically and call finish() when done."
    )

    messages = [{"role": "user", "content": user_msg}]
    tool_calls_made = 0

    async with main_pool.acquire() as main_conn, oi_pool.acquire() as oi_conn:
        while tool_calls_made < max_tool_calls:
            resp = await client.messages.create(
                model=model,
                max_tokens=4096,
                system=_SYSTEM,
                tools=_TOOLS,
                messages=messages,
            )

            # Collect assistant message
            assistant_content = []
            finish_summary = None

            for block in resp.content:
                assistant_content.append(block)
                if block.type == "tool_use":
                    tool_calls_made += 1
                    inputs = block.input
                    log(f"[{tool_calls_made}/{max_tool_calls}] {block.name}({json.dumps({k: v for k, v in inputs.items() if k != 'table'}, default=str)[:120]})")

                    if block.name == "finish":
                        finish_summary = inputs.get("summary", "")
                        break

                    try:
                        result = await _dispatch(
                            block.name, inputs, main_conn, oi_conn, run_id, log
                        )
                    except Exception as exc:
                        result = {"error": str(exc)}
                        log(f"  ERROR: {exc}")

                    # Append assistant turn + tool result before next iteration
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
                    break  # restart loop with updated messages

            if finish_summary is not None:
                log("Agent called finish()")
                return finish_summary

            if resp.stop_reason == "end_turn" and not any(
                b.type == "tool_use" for b in resp.content
            ):
                # Claude stopped without calling finish — extract any text
                text = " ".join(
                    b.text for b in resp.content if hasattr(b, "text") and b.text
                )
                return text or "Agent completed without producing a summary."

    return f"Agent hit max_tool_calls limit ({max_tool_calls}). Partial results saved to DB."
