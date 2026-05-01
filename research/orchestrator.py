"""
Research 2 orchestration layer — Phase 1.

Sits above the deterministic tools and decides:
  1. What kind of analysis the question requires (classify)
  2. Which columns / tools to run (plan)
  3. Which specific visuals are worth generating (direct)
  4. How to frame the final narrative (report)

The deterministic tools (scanner, blocks, charts) are unchanged building blocks.
"""
import json
import re
from typing import Optional

try:
    import anthropic as _anthropic
except ImportError:
    _anthropic = None

from research import scanner, blocks, db as rdb

_OI_TABLES = {"daily_features", "option_oi_surface", "underlying_ohlc"}
_EXCLUDE_COLS = {"id", "ticker", "trade_date", "created_at", "updated_at"}

# ── Structured output tools (guaranteed valid JSON via tool_use) ──────────────

_PLAN_TOOL = {
    "name": "produce_plan",
    "description": "Output the workflow plan as structured JSON.",
    "input_schema": {
        "type": "object",
        "properties": {
            "task_type":                 {"type": "string"},
            "task_reasoning":            {"type": "string"},
            "hypotheses":                {"type": "array", "items": {"type": "string"}},
            "feature_columns":           {"type": "array", "items": {"type": "string"}},
            "outcome_columns":           {"type": "array", "items": {"type": "string"}},
            "depth":                     {"type": "string", "enum": ["broad", "deep", "targeted"]},
            "scan_focus":                {"type": "string"},
            "key_questions":             {"type": "array", "items": {"type": "string"}},
            "report_guidance":           {"type": "string"},
            "column_selection_reasoning": {"type": "string"},
            "tickers": {"type": "array", "items": {"type": "string"},
                        "description": "Tickers to analyze individually. Use ['ALL'] if the question asks for per-ticker or ticker-level analysis across the full universe."},
        },
        "required": ["task_type", "feature_columns", "outcome_columns"],
    },
}

_REPORT_TOOL = {
    "name": "produce_report",
    "description": "Output the research report as structured sections with chart references.",
    "input_schema": {
        "type": "object",
        "properties": {
            "sections": {
                "type": "array",
                "description": "Interleaved markdown prose and chart reference blocks.",
                "items": {
                    "type": "object",
                    "properties": {
                        "type": {
                            "type": "string",
                            "enum": ["markdown", "chart"],
                        },
                        "content": {
                            "type": "string",
                            "description": "Markdown text (only for type=markdown).",
                        },
                        "chart_id": {
                            "type": "string",
                            "description": "Chart reference ID (only for type=chart). Must be one of the available chart IDs listed in the prompt.",
                        },
                    },
                    "required": ["type"],
                },
            },
        },
        "required": ["sections"],
    },
}

TASK_TYPES = [
    "single-factor-scan",
    "multi-factor-interaction",
    "backtest-pnl-attribution",
    "event-study",
    "regime-analysis",
    "microstructure-investigation",
    "strategy-entry-condition",
    "anomaly-investigation",
]

# ── Prompts ───────────────────────────────────────────────────────────────────

_PLAN_SYSTEM = (
    "You are a quantitative research orchestrator. "
    "Read a research question and available data, then produce a precise workflow plan. "
    "Output ONLY valid JSON — no prose, no markdown fences."
)

_PLAN_USER = """\
Research question:
\"\"\"{question}\"\"\"

Data:
  Table:  {table}
  Tickers: {tickers}
  Period:  {date_range}

Available numeric columns:
{columns}

Return a JSON object with exactly these keys:

{{
  "task_type": "<one of: {task_types}>",
  "task_reasoning": "<1-2 sentences: why this classification>",
  "hypotheses": ["<testable hypothesis>", ...],
  "feature_columns": ["<col_name>", ...],
  "outcome_columns": ["<col_name>", ...],
  "depth": "<broad | deep | targeted>",
  "scan_focus": "<what relationship patterns to look for>",
  "key_questions": ["<specific question the analysis must answer>", ...],
  "report_guidance": "<instruction for the final narrative>",
  "column_selection_reasoning": "<why these columns>"
}}

Column selection rules:
- feature_columns: predictor/signal columns (X), max 15
- outcome_columns: what to predict (Y), prefer forward return columns (ret_Nd_fwd), max 4
- Use ONLY column names from the available list above
- Never include: id, ticker, trade_date, created_at, updated_at
- CRITICAL: If the question explicitly names or describes specific columns/metrics, those columns
  MUST be included in feature_columns or outcome_columns. Do not substitute similar columns.
  The user's named metrics are the primary focus — add related columns only as supplements.
- broad depth: include all plausibly relevant features
- targeted depth: 3-7 most directly relevant features

Ticker selection rules:
- If the question mentions specific tickers (e.g. "SPY", "AAPL"), include them in tickers array
- If the question asks for "per-ticker", "ticker-level", "specific tickers", "which tickers",
  "across tickers", or "explore tickers" — set tickers to ["ALL"] to analyze each ticker separately
- If the question is about a general concept without mentioning tickers, omit the tickers field
  (data will be pooled across all tickers)
- The tickers field in the available data is: {tickers}
"""


_REPORT_USER = """\
RESEARCH QUESTION (primary directive — address this directly):
\"\"\"{question}\"\"\"

TASK TYPE: {task_type}
FOCUS: {scan_focus}
REPORT GUIDANCE: {report_guidance}

TOP SINGLE-FACTOR SIGNALS:
{signal_table}

FULL BUCKET PROFILES (top 5):
{bucket_profiles}

MULTI-FACTOR INTERACTIONS:
{interaction_summary}

EQUITY CURVES:
{equity_summary}

AVAILABLE CHARTS (reference by chart_id in your report):
{available_charts}

Produce a research report using the produce_report tool. The report is an array of sections,
each either markdown prose or a chart reference. Interleave text and charts naturally —
introduce a finding in prose, then immediately reference the chart that supports it.

For chart sections, set type="chart" and chart_id to one of the IDs listed in AVAILABLE CHARTS.
You should reference most or all of the available charts, placing each where it best supports the narrative.

PROSE GUIDELINES (for markdown sections):
- Answer the research question directly in the first section
- Describe the full bucket profile shape — not just top/bottom
- If multi-factor combos were tested, describe as usable trading rules
- Distinguish tradable from spurious. Cite specific numbers.
- State what did NOT work
- End with actionable conclusion
- Professional quant voice, no fluff, 400-700 words total prose
"""


# ── Phase 1: Classify and plan ────────────────────────────────────────────────

async def classify_and_plan(
    question: str,
    available_columns: list[dict],
    tickers: list[str],
    table: str,
    date_from: Optional[str],
    date_to: Optional[str],
    model: str = "claude-sonnet-4-6",
    knowledge_rules: list[str] = None,
) -> dict:
    """
    Call Claude to classify the research question and produce a workflow plan.
    Uses tool_use for guaranteed valid JSON output.
    """
    if _anthropic is None:
        raise RuntimeError("anthropic SDK not installed")

    col_lines = "\n".join(f"  {c['name']}" for c in available_columns)
    tickers_str = ", ".join(tickers) if tickers else "all (no ticker filter)"
    date_range = f"{date_from or 'earliest'} → {date_to or 'latest'}"

    prompt = _PLAN_USER.format(
        question=question,
        table=table,
        tickers=tickers_str,
        date_range=date_range,
        columns=col_lines,
        task_types=" | ".join(TASK_TYPES),
    )

    # Inject knowledge rules if available
    if knowledge_rules:
        rules_text = "\n".join(f"- {r}" for r in knowledge_rules)
        prompt += f"\n\nDOMAIN KNOWLEDGE (follow these strictly):\n{rules_text}"

    client = _anthropic.AsyncAnthropic()
    resp = await client.messages.create(
        model=model,
        max_tokens=1500,
        system=[{"type": "text", "text": _PLAN_SYSTEM,
                 "cache_control": {"type": "ephemeral"}}],
        tools=[_PLAN_TOOL],
        tool_choice={"type": "tool", "name": "produce_plan"},
        messages=[{"role": "user", "content": prompt}],
    )

    # Extract structured output from tool_use block
    plan = None
    for block in resp.content:
        if block.type == "tool_use" and block.name == "produce_plan":
            plan = block.input
            break

    if plan is None:
        # Fallback: try parsing text response as JSON
        raw = resp.content[0].text if resp.content and hasattr(resp.content[0], "text") else ""
        raw = re.sub(r"^```(?:json)?\s*", "", raw.strip())
        raw = re.sub(r"\s*```$", "", raw).strip()
        plan = json.loads(raw)

    # Validate: only keep columns that actually exist in the table
    valid_cols = {c["name"] for c in available_columns}
    plan["feature_columns"] = [c for c in plan.get("feature_columns") or [] if c in valid_cols]
    plan["outcome_columns"] = [c for c in plan.get("outcome_columns") or [] if c in valid_cols]
    plan.setdefault("task_type", "single-factor-scan")
    plan.setdefault("hypotheses", [])
    plan.setdefault("key_questions", [])

    # Force-include columns the user explicitly mentioned in the question
    # Match against available columns (case-insensitive substring match)
    q_lower = question.lower()
    for col_name in valid_cols:
        if col_name.lower() in q_lower and col_name not in _EXCLUDE_COLS:
            # Determine if it's a feature or outcome based on name
            if col_name.startswith("ret_") or col_name.endswith("_fwd") or "_fwd_" in col_name:
                if col_name not in plan["outcome_columns"]:
                    plan["outcome_columns"].append(col_name)
            else:
                if col_name not in plan["feature_columns"]:
                    plan["feature_columns"].append(col_name)

    # Handle ticker selection from the plan
    plan_tickers = plan.get("tickers") or []
    if plan_tickers == ["ALL"] or "ALL" in plan_tickers:
        plan["tickers_mode"] = "all_individual"
    elif plan_tickers:
        plan["tickers_override"] = plan_tickers

    return plan


# ── Compose final report with inline charts ─────────────────────────────────────

def _build_chart_configs(
    ranked_scans: list[dict],
    equity_results: list[dict],
    ticker_scoreboard: dict | None,
) -> dict[str, dict]:
    """Build Chart.js configs deterministically from analysis data. Returns {chart_id: {title, config}}."""
    charts = {}
    valid = [s for s in ranked_scans if "error" not in s]
    single = [s for s in valid if s.get("x_col") and not s.get("combo")]
    colors_pos = '#3498db'
    colors_neg = '#e84393'
    dark_scales = {
        'x': {'ticks': {'color': '#888', 'font': {'size': 9}, 'maxRotation': 45},
               'grid': {'color': 'rgba(255,255,255,0.05)'}, 'border': {'color': 'transparent'}},
        'y': {'ticks': {'color': '#888', 'font': {'size': 9}},
               'grid': {'color': 'rgba(255,255,255,0.05)'}, 'border': {'color': 'transparent'}},
    }
    base_opts = {
        'responsive': True, 'maintainAspectRatio': False, 'animation': False,
        'plugins': {'legend': {'labels': {'color': '#aaa', 'font': {'size': 10}}},
                    'tooltip': {'backgroundColor': 'rgba(20,20,20,0.95)',
                                'borderColor': '#444', 'borderWidth': 1}},
        'scales': dark_scales,
    }

    # 1. Ticker scoreboard — horizontal bar of best scores
    if ticker_scoreboard and len(ticker_scoreboard) > 1:
        sorted_tickers = sorted(ticker_scoreboard.items(), key=lambda x: x[1]['score'], reverse=True)
        charts['ticker_scoreboard'] = {
            'title': 'Best Composite Score by Ticker',
            'config': {
                'type': 'bar',
                'data': {
                    'labels': [tk for tk, _ in sorted_tickers],
                    'datasets': [{
                        'label': 'Composite Score',
                        'data': [round(info['score'], 1) for _, info in sorted_tickers],
                        'backgroundColor': [colors_pos if info['score'] >= 40 else
                                            '#95a5a6' if info['score'] >= 25 else
                                            colors_neg for _, info in sorted_tickers],
                        'borderWidth': 0,
                    }],
                },
                'options': {**base_opts, 'indexAxis': 'y',
                            'plugins': {**base_opts['plugins'],
                                        'legend': {'display': False}}},
            },
        }

    # 2. Bucket profiles for top signals (up to 8)
    seen_profiles = set()
    for s in single[:12]:
        tk = s.get('ticker') or 'all'
        x_col = s['x_col']
        y_col = s['y_col']
        key = f"{tk}_{x_col}_{y_col}"
        if key in seen_profiles:
            continue
        seen_profiles.add(key)
        bs = [b for b in (s.get('bucket_stats') or []) if b is not None]
        if len(bs) < 3:
            continue
        avgs = [round(b['avg_ret'] * 100, 4) for b in bs]
        cid = f"bucket_{tk}_{x_col}_{y_col}"
        charts[cid] = {
            'title': f'Decile Profile: {tk} | {x_col} → {y_col} (score={s.get("composite_score", 0):.0f})',
            'config': {
                'type': 'bar',
                'data': {
                    'labels': [f'D{b["bucket"]}' for b in bs],
                    'datasets': [{
                        'label': 'Avg Return %',
                        'data': avgs,
                        'backgroundColor': [colors_pos if v >= 0 else colors_neg for v in avgs],
                        'borderWidth': 0,
                    }],
                },
                'options': {**base_opts,
                            'plugins': {**base_opts['plugins'], 'legend': {'display': False}}},
            },
        }
        if len(charts) - (1 if 'ticker_scoreboard' in charts else 0) >= 8:
            break

    # 3. Equity curves (top 3 pairs)
    eq_by_key = {}
    for r in equity_results:
        tk = r.get('ticker') or 'all'
        key = f"{tk}_{r.get('feature_col')}_{r.get('outcome_col')}"
        eq_by_key.setdefault(key, {})[r.get('which', '')] = r

    eq_count = 0
    for key, sides in eq_by_key.items():
        if eq_count >= 3:
            break
        top_r = sides.get('top') or sides.get('top2')
        bot_r = sides.get('bottom') or sides.get('bottom2')
        if not top_r or not top_r.get('points'):
            continue
        pts = top_r['points']
        step = max(1, len(pts) // 60)
        sampled_pts = pts[::step]
        datasets = [{
            'label': f'Top ({top_r.get("which", "D10")})',
            'data': [round(p['value'] * 100, 2) for p in sampled_pts],
            'borderColor': colors_pos, 'backgroundColor': 'transparent',
            'borderWidth': 2, 'pointRadius': 0, 'tension': 0.1,
        }]
        if bot_r and bot_r.get('points'):
            bot_pts = bot_r['points']
            bot_step = max(1, len(bot_pts) // 60)
            bot_sampled = bot_pts[::bot_step]
            datasets.append({
                'label': f'Bottom ({bot_r.get("which", "D1")})',
                'data': [round(p['value'] * 100, 2) for p in bot_sampled],
                'borderColor': colors_neg, 'backgroundColor': 'transparent',
                'borderWidth': 2, 'pointRadius': 0, 'tension': 0.1,
            })
        tk = top_r.get('ticker') or 'all'
        fc = top_r.get('feature_col', '?')
        oc = top_r.get('outcome_col', '?')
        cid = f"equity_{tk}_{fc}_{oc}"
        charts[cid] = {
            'title': f'Equity Curve: {tk} | {fc} → {oc}',
            'config': {
                'type': 'line',
                'data': {
                    'labels': [p['date'][:7] for p in sampled_pts],
                    'datasets': datasets,
                },
                'options': {**base_opts,
                            'scales': {**dark_scales,
                                       'x': {**dark_scales['x'],
                                              'ticks': {**dark_scales['x']['ticks'], 'maxTicksLimit': 12}}}},
            },
        }
        eq_count += 1

    # 4. Yearly consistency for the most consistent signal
    most_consistent = sorted(
        [s for s in single if (s.get('robustness') or {}).get('yearly_data')],
        key=lambda s: (s.get('robustness') or {}).get('yearly_consistency_pct', 0), reverse=True)
    if most_consistent:
        s = most_consistent[0]
        yd = s['robustness']['yearly_data']
        tk = s.get('ticker') or 'all'
        charts[f"yearly_{tk}_{s['x_col']}_{s['y_col']}"] = {
            'title': f'Yearly Consistency: {tk} | {s["x_col"]} → {s["y_col"]}',
            'config': {
                'type': 'bar',
                'data': {
                    'labels': [str(y.get('year', '?')) for y in yd],
                    'datasets': [
                        {'label': 'Top Decile', 'data': [round(y.get('top_avg', 0) * 100, 3) for y in yd],
                         'backgroundColor': colors_pos, 'borderWidth': 0},
                        {'label': 'Bottom Decile', 'data': [round(y.get('bot_avg', 0) * 100, 3) for y in yd],
                         'backgroundColor': colors_neg, 'borderWidth': 0},
                    ],
                },
                'options': base_opts,
            },
        }

    # 5. Win rate comparison across tickers (if multi-ticker)
    if ticker_scoreboard and len(ticker_scoreboard) > 2:
        # Collect best signal's D10 win rate per ticker
        wr_by_ticker = {}
        for s in single:
            tk = s.get('ticker') or 'all'
            if tk == 'all' or tk in wr_by_ticker:
                continue
            bs = [b for b in (s.get('bucket_stats') or []) if b is not None]
            if bs:
                d10 = next((b for b in bs if b['bucket'] == 10), None)
                d1 = next((b for b in bs if b['bucket'] == 1), None)
                if d10 and d1:
                    wr_by_ticker[tk] = {
                        'd10_wr': round(d10.get('win_rate', 0.5) * 100, 1),
                        'd1_wr': round(d1.get('win_rate', 0.5) * 100, 1),
                    }
        if len(wr_by_ticker) > 2:
            sorted_tk = sorted(wr_by_ticker.keys())
            charts['winrate_comparison'] = {
                'title': 'D10 vs D1 Win Rate by Ticker (best signal per ticker)',
                'config': {
                    'type': 'bar',
                    'data': {
                        'labels': sorted_tk,
                        'datasets': [
                            {'label': 'D10 Win Rate %', 'data': [wr_by_ticker[t]['d10_wr'] for t in sorted_tk],
                             'backgroundColor': colors_pos, 'borderWidth': 0},
                            {'label': 'D1 Win Rate %', 'data': [wr_by_ticker[t]['d1_wr'] for t in sorted_tk],
                             'backgroundColor': colors_neg, 'borderWidth': 0},
                        ],
                    },
                    'options': base_opts,
                },
            }

    # 6. Spread comparison — D10-D1 avg return spread by ticker
    if ticker_scoreboard and len(ticker_scoreboard) > 2:
        spread_data = {}
        for s in single:
            tk = s.get('ticker') or 'all'
            if tk == 'all' or tk in spread_data:
                continue
            bs = [b for b in (s.get('bucket_stats') or []) if b is not None]
            if bs:
                d10 = next((b for b in bs if b['bucket'] == 10), None)
                d1 = next((b for b in bs if b['bucket'] == 1), None)
                if d10 and d1:
                    spread_data[tk] = round((d10['avg_ret'] - d1['avg_ret']) * 100, 4)
        if len(spread_data) > 2:
            sorted_tk = sorted(spread_data, key=lambda t: spread_data[t], reverse=True)
            charts['spread_comparison'] = {
                'title': 'D10 minus D1 Return Spread by Ticker (%)',
                'config': {
                    'type': 'bar',
                    'data': {
                        'labels': sorted_tk,
                        'datasets': [{
                            'label': 'Spread %',
                            'data': [spread_data[t] for t in sorted_tk],
                            'backgroundColor': [colors_pos if spread_data[t] >= 0 else colors_neg for t in sorted_tk],
                            'borderWidth': 0,
                        }],
                    },
                    'options': {**base_opts, 'plugins': {**base_opts['plugins'], 'legend': {'display': False}}},
                },
            }

    return charts


async def _compose_report(
    question: str,
    plan: dict,
    ranked_scans: list[dict],
    equity_results: list[dict],
    model: str,
    knowledge_rules: list[str] = None,
    extra_context: str = "",
    ticker_scoreboard: dict = None,
) -> list[dict]:
    """Compose report: LLM writes prose + chart references, charts built deterministically."""
    # Build all charts from data
    chart_configs = _build_chart_configs(ranked_scans, equity_results, ticker_scoreboard)

    if _anthropic is None:
        sections = [{"type": "markdown", "content": "(anthropic SDK not available)"}]
        for cid, chart in chart_configs.items():
            sections.append({"type": "chart", "chart_id": cid, "title": chart["title"], "config": chart["config"]})
        return sections

    valid = [s for s in ranked_scans if "error" not in s]
    top = valid[:20]

    # Signal table
    sig_lines = [
        f"{'Score':>5}  {'Signal':<55} {'Pattern':<22} {'Consist':>8} {'Conc':>5}",
        "-" * 100,
    ]
    for s in top:
        rob = s.get("robustness", {})
        sig = f"{s.get('ticker') or 'all'} | {s.get('x_col')} → {s.get('y_col')}"
        sig_lines.append(
            f"{s.get('composite_score', 0):>5.0f}  {sig:<55} "
            f"{s.get('pattern', ''):<22} "
            f"{str(rob.get('yearly_consistency_pct', '—'))+'%':>8} "
            f"{rob.get('concentration_risk', 1.0):>5.2f}"
        )

    # Bucket profiles for top 5
    bp_lines = []
    for s in top[:5]:
        bs = [b for b in (s.get("bucket_stats") or []) if b is not None]
        if bs:
            bp_lines.append(
                f"\n{s.get('ticker') or 'all'} | {s.get('x_col')} → {s.get('y_col')} "
                f"(score={s.get('composite_score', 0):.0f}, pattern={s.get('pattern')}):"
            )
            bp_lines.append(f"  {'Bucket':>6} {'N':>5} {'AvgRet':>9} {'WinRate':>8} {'Sharpe':>7}")
            for b in bs:
                bp_lines.append(
                    f"  {b['bucket']:>6} {b['n']:>5} "
                    f"{b['avg_ret']*100:>8.3f}% {b['win_rate']*100:>7.1f}% "
                    f"{b.get('sharpe', 0):>7.3f}"
                )

    # Interaction highlights
    interactions = [s for s in valid if s.get("composite_interaction_score") is not None]
    int_lines = []
    if interactions:
        top_int = sorted(interactions,
                         key=lambda s: s.get("composite_interaction_score", 0), reverse=True)[:5]
        for s in top_int:
            combo = "+".join(s.get("combo", []))
            bz = s.get("best_quadrant") or s.get("best_octant") or s.get("best_zone") or {}
            int_lines.append(
                f"  {s.get('ticker') or 'all'} | {combo} → {s.get('y_col')}: "
                f"score={s.get('composite_interaction_score', 0):.0f}, "
                f"lift={s.get('interaction_lift', 0):+.4f}, "
                f"zone={bz.get('label', '?')} "
                f"(avg={bz.get('avg_ret', 0):.4f}, WR={bz.get('win_rate', 0):.0%})")

    # Equity summary
    eq_lines = []
    for r in sorted(equity_results,
                    key=lambda x: abs(x.get("cumulative_return") or x.get("final_equity") or 0),
                    reverse=True)[:8]:
        cum = r.get("cumulative_return") or r.get("final_equity")
        eq_lines.append(
            f"  {r.get('ticker') or 'all'} | {r.get('feature_col')} → {r.get('outcome_col')} "
            f"({r.get('which')}): cumReturn={cum*100:.1f}%, "
            f"maxDD={r.get('max_drawdown')}, n={r.get('n_trades')}, "
            f"winRate={r.get('win_rate')}"
        )

    # Available charts listing for the LLM
    chart_listing = []
    for cid, chart in chart_configs.items():
        chart_listing.append(f"  {cid}: {chart['title']}")

    def _safe(s: str) -> str:
        return str(s).replace("{", "{{").replace("}", "}}")

    # Knowledge rules injection
    report_system = (
        "You are a quantitative research analyst writing focused research reports. "
        "Professional quant voice. No fluff."
    )
    if knowledge_rules:
        rules_text = "\n".join(f"- {r}" for r in knowledge_rules)
        report_system += (
            f"\n\nDOMAIN KNOWLEDGE & POLICIES (follow these strictly):\n{rules_text}"
        )

    user_content = _REPORT_USER.format(
        question=_safe(question),
        task_type=_safe(plan.get("task_type", "")),
        scan_focus=_safe(plan.get("scan_focus", "")),
        report_guidance=_safe(plan.get("report_guidance", "")),
        signal_table="\n".join(sig_lines),
        bucket_profiles="\n".join(bp_lines) if bp_lines else "(none)",
        interaction_summary="\n".join(int_lines) if int_lines else "(none tested)",
        equity_summary="\n".join(eq_lines) if eq_lines else "(none)",
        available_charts="\n".join(chart_listing) if chart_listing else "(none)",
    )
    if extra_context:
        user_content += f"\n\nTASK-SPECIFIC ANALYSIS:\n{extra_context}"

    client = _anthropic.AsyncAnthropic()
    msg = await client.messages.create(
        model=model,
        max_tokens=3000,
        system=[{"type": "text", "text": report_system,
                 "cache_control": {"type": "ephemeral"}}],
        tools=[_REPORT_TOOL],
        tool_choice={"type": "tool", "name": "produce_report"},
        messages=[{"role": "user", "content": user_content}],
    )

    # Extract structured output — LLM returns chart references, we inject configs
    sections = []
    llm_sections = None
    for block in msg.content:
        if block.type == "tool_use" and block.name == "produce_report":
            llm_sections = block.input.get("sections", [])
            break

    if llm_sections is None:
        # Fallback: wrap raw text, attach all charts at the end
        raw = msg.content[0].text if msg.content and hasattr(msg.content[0], "text") else ""
        sections.append({"type": "markdown", "content": raw.strip() or "(Report generation failed)"})
        for cid, chart in chart_configs.items():
            sections.append({"type": "chart", "chart_id": cid, "title": chart["title"], "config": chart["config"]})
        return sections

    # Merge LLM prose + chart references with actual configs
    referenced = set()
    for sec in llm_sections:
        if sec.get("type") == "markdown":
            sections.append(sec)
        elif sec.get("type") == "chart":
            cid = sec.get("chart_id", "")
            if cid in chart_configs:
                sections.append({
                    "type": "chart",
                    "chart_id": cid,
                    "title": chart_configs[cid]["title"],
                    "config": chart_configs[cid]["config"],
                })
                referenced.add(cid)

    # Append any charts the LLM didn't reference (so nothing is lost)
    unreferenced = [cid for cid in chart_configs if cid not in referenced]
    if unreferenced:
        for cid in unreferenced:
            sections.append({
                "type": "chart",
                "chart_id": cid,
                "title": chart_configs[cid]["title"],
                "config": chart_configs[cid]["config"],
            })

    return sections


# ── Phase 3: Skeptic / validator pass ────────────────────────────────────────

_SKEPTIC_SYSTEM = (
    "You are a skeptical quantitative reviewer. Your job is to challenge research "
    "findings and identify weaknesses, overfit risks, and misleading conclusions. "
    "Be specific and cite numbers. Do not repeat the findings — only challenge them."
)

_SKEPTIC_USER = """\
RESEARCH REPORT TO CHALLENGE:
\"\"\"
{report}
\"\"\"

KEY DIAGNOSTICS:
{diagnostics}

Write a concise skeptic review (150-300 words). Challenge:

1. SAMPLE SIZE: Are bucket sizes large enough? Is N sufficient for the claimed precision?
   Flag any claim based on fewer than 100 observations per bucket.

2. CONSISTENCY: Does "100% yearly consistency" on 3-4 years mean anything statistically?
   How many years were actually tested?

3. MONOTONICITY vs TAILS: Is the claimed monotonic relationship truly smooth, or is
   most of the edge concentrated in extreme deciles while the middle is flat?

4. DRAWDOWN: If max drawdown exceeds 30%, is the signal really tradeable?
   Would the claimed edge survive transaction costs and slippage?

5. MULTI-FACTOR OVERFIT: Do the multi-factor combos add genuine lift, or are they
   slicing the sample thin enough that noise looks like signal?

6. REGIME CONCENTRATION: Does the signal work across market regimes, or is it
   dominated by one unusual period?

7. LOOK-AHEAD: Could any feature construction embed look-ahead bias?

Only raise issues that are actually supported by the diagnostics.
Do not manufacture problems that don't exist. Be fair but rigorous.
"""


async def _skeptic_review(
    report: str,
    ranked_scans: list[dict],
    equity_results: list[dict],
    model: str,
    knowledge_rules: list[str] = None,
) -> str:
    """ONE Claude call to challenge the report findings."""
    if _anthropic is None:
        return ""

    # Build compact diagnostics for the skeptic
    diag_lines = []
    valid = [s for s in ranked_scans if "error" not in s]

    for s in valid[:8]:
        rob = s.get("robustness", {})
        bs = [b for b in (s.get("bucket_stats") or []) if b is not None]
        min_n = min((b["n"] for b in bs), default=0) if bs else 0
        mid_range = ""
        if len(bs) >= 6:
            mid_avgs = [b["avg_ret"] for b in bs[3:7]]
            mid_range = f"mid-decile range={max(mid_avgs)-min(mid_avgs):.4f}"

        diag_lines.append(
            f"  {s.get('ticker') or 'all'} | {s.get('x_col')} → {s.get('y_col')}: "
            f"score={s.get('composite_score', 0):.0f}, n={s.get('n', 0)}, "
            f"min_bucket_n={min_n}, "
            f"consistency={rob.get('yearly_consistency_pct', '?')}% "
            f"({rob.get('years_checked', '?')} years), "
            f"concentration={rob.get('concentration_risk', 1):.2f}, "
            f"half_stable={'Y' if rob.get('half_sample_stable') else 'N'}, "
            f"loyo_fragile={'Y' if rob.get('loyo_fragile') else 'N'}, "
            f"{mid_range}"
        )

    for r in equity_results[:6]:
        cum = r.get("cumulative_return") or r.get("final_equity", 0)
        diag_lines.append(
            f"  Equity {r.get('ticker') or 'all'} | {r.get('feature_col')} → {r.get('outcome_col')} "
            f"({r.get('which')}): cumReturn={cum*100:.1f}%, "
            f"maxDD={r.get('max_drawdown', 0)*100:.1f}%, "
            f"n_trades={r.get('n_trades', 0)}, winRate={r.get('win_rate', 0)*100:.0f}%"
        )

    diagnostics = "\n".join(diag_lines) if diag_lines else "(no diagnostics available)"

    skeptic_system = _SKEPTIC_SYSTEM
    if knowledge_rules:
        rules_text = "\n".join(f"- {r}" for r in knowledge_rules)
        skeptic_system += f"\n\nDOMAIN RULES (apply these):\n{rules_text}"

    client = _anthropic.AsyncAnthropic()
    msg = await client.messages.create(
        model=model,
        max_tokens=800,
        system=[{"type": "text", "text": skeptic_system,
                 "cache_control": {"type": "ephemeral"}}],
        messages=[{
            "role": "user",
            "content": _SKEPTIC_USER.format(
                report=report[:3000],
                diagnostics=diagnostics,
            ),
        }],
    )
    return msg.content[0].text.strip()


# ── Task-type-specific analysis steps ────────────────────────────────────────

async def _run_regime_analysis(cache, ranked, all_scans, x_cols, y_cols,
                                main_pool, run_id, log):
    """Split data by regime indicator, scan within each, compare."""
    regime_results = []
    # Use the median of the best-performing feature as the regime split
    # Or if rv_20d is available, use realized vol as regime
    for cache_key, rows in cache.items():
        ticker = None if cache_key == "_all" else cache_key
        avail = set(rows[0].keys()) if rows else set()

        # Find regime column: prefer rv_20d, then spot_close trend, else use median of first x
        regime_col = None
        for candidate in ["rv_20d", "rv_5d"]:
            if candidate in avail:
                regime_col = candidate
                break

        if not regime_col:
            log(f"  No regime column (rv_20d/rv_5d) for {ticker or 'all'} — skipping regime split")
            continue

        # Split at median of regime column
        import math
        vals = [float(r[regime_col]) for r in rows
                if r.get(regime_col) is not None and not math.isnan(float(r[regime_col]))]
        if len(vals) < 100:
            continue
        import numpy as np
        median_val = float(np.median(vals))

        high_regime = [r for r in rows if r.get(regime_col) is not None
                       and float(r[regime_col]) >= median_val]
        low_regime = [r for r in rows if r.get(regime_col) is not None
                      and float(r[regime_col]) < median_val]

        log(f"  Regime split on {regime_col} (median={median_val:.4f}): "
            f"high={len(high_regime)}, low={len(low_regime)}")

        # Scan the top features in each regime separately
        top_features = list(dict.fromkeys(s["x_col"] for s in ranked[:10] if s.get("x_col")))[:5]

        async with main_pool.acquire() as conn:
            for x_col in top_features:
                if x_col not in avail:
                    continue
                for y_col in y_cols:
                    if y_col not in avail or x_col == y_col:
                        continue
                    for regime_label, regime_rows in [("high_vol", high_regime), ("low_vol", low_regime)]:
                        if len(regime_rows) < 60:
                            continue
                        try:
                            scan = scanner.scan_relationship(regime_rows, x_col, y_col, ticker)
                            scan["regime"] = regime_label
                            scan["regime_col"] = regime_col
                            scan["regime_n"] = len(regime_rows)
                            await rdb.save_result(conn, run_id, f"scan_regime_{regime_label}",
                                                  x_col, y_col, scan, ticker)
                            regime_results.append(scan)
                        except Exception as exc:
                            log(f"    REGIME ERROR {regime_label} {x_col}→{y_col}: {exc}")

    log(f"  {len(regime_results)} regime-conditional scans completed.")
    return regime_results


async def _run_event_study(cache, x_cols, y_cols, main_pool, run_id, log):
    """Analyze signals around specific event windows (monthly OpEx, etc.)."""
    event_results = []

    for cache_key, rows in cache.items():
        ticker = None if cache_key == "_all" else cache_key
        avail = set(rows[0].keys()) if rows else set()

        # Use days_to_monthly_opex if available, else use dte-based proxy
        opex_col = None
        for candidate in ["days_to_monthly_opex", "pct_oi_in_front_expiry"]:
            if candidate in avail:
                opex_col = candidate
                break

        if not opex_col:
            log(f"  No event column for {ticker or 'all'} — skipping event study")
            continue

        import math
        # Split into "near event" (within 5 days of OpEx) vs "far from event"
        if opex_col == "days_to_monthly_opex":
            near = [r for r in rows if r.get(opex_col) is not None
                    and 0 <= float(r[opex_col]) <= 5]
            far = [r for r in rows if r.get(opex_col) is not None
                   and float(r[opex_col]) > 5]
        else:
            # pct_oi_in_front_expiry: high = near expiry
            import numpy as np
            vals = [float(r[opex_col]) for r in rows
                    if r.get(opex_col) is not None and not math.isnan(float(r[opex_col]))]
            if not vals:
                continue
            p75 = float(np.percentile(vals, 75))
            near = [r for r in rows if r.get(opex_col) is not None
                    and float(r[opex_col]) >= p75]
            far = [r for r in rows if r.get(opex_col) is not None
                   and float(r[opex_col]) < p75]

        log(f"  Event split on {opex_col}: near={len(near)}, far={len(far)}")

        async with main_pool.acquire() as conn:
            for x_col in x_cols[:8]:
                if x_col not in avail or x_col == opex_col:
                    continue
                for y_col in y_cols:
                    if y_col not in avail or x_col == y_col:
                        continue
                    for event_label, event_rows in [("near_event", near), ("far_event", far)]:
                        if len(event_rows) < 40:
                            continue
                        try:
                            scan = scanner.scan_relationship(event_rows, x_col, y_col, ticker)
                            scan["event_window"] = event_label
                            scan["event_col"] = opex_col
                            scan["event_n"] = len(event_rows)
                            await rdb.save_result(conn, run_id, f"scan_event_{event_label}",
                                                  x_col, y_col, scan, ticker)
                            event_results.append(scan)
                        except Exception as exc:
                            log(f"    EVENT ERROR {event_label} {x_col}→{y_col}: {exc}")

    log(f"  {len(event_results)} event-conditional scans completed.")
    return event_results


async def _run_strategy_entry(cache, ranked, x_cols, y_cols,
                               main_pool, run_id, log):
    """Generate detailed signal cards for the best combo zones."""
    strategy_results = []

    # Find the best single-factor signals and best interactions
    single_top = [s for s in ranked if s.get("x_col") and not s.get("combo")][:3]
    combo_top = [s for s in ranked if s.get("combo")][:3]
    candidates = single_top + combo_top

    if not candidates:
        log("  No candidates for strategy entry conditions.")
        return []

    async with main_pool.acquire() as conn:
        for s in candidates:
            ticker = s.get("ticker")
            cache_key = ticker if ticker else "_all"
            rows = cache.get(cache_key, [])
            if not rows:
                continue

            y_col = s.get("y_col")
            combo = s.get("combo")

            if combo:
                # Multi-factor: compute robustness for the best zone
                bz = s.get("best_quadrant") or s.get("best_octant") or s.get("best_zone") or {}
                zone_label = bz.get("label", "")
                if zone_label:
                    try:
                        rob = scanner.combo_robustness(rows, combo, y_col, zone_label, ticker)
                        entry_card = {
                            "type": "multi_factor_entry",
                            "combo": combo,
                            "y_col": y_col,
                            "ticker": ticker,
                            "zone": zone_label,
                            "zone_stats": bz,
                            "robustness": rob,
                            "rule": f"Enter when {' AND '.join(f'{f} is {z}' for f, z in zip(combo, zone_label))} "
                                    f"→ expect {y_col}",
                        }
                        await rdb.save_result(conn, run_id, "strategy_entry",
                                              "+".join(combo), y_col, entry_card, ticker)
                        strategy_results.append(entry_card)
                        log(f"  ✓ Strategy card: {'+'.join(combo)} zone={zone_label}")
                    except Exception as exc:
                        log(f"    STRATEGY ERROR {combo}: {exc}")
            else:
                # Single factor: generate entry rule from best bucket
                x_col = s["x_col"]
                best_b = s.get("best_single_bucket") or {}
                bucket_num = best_b.get("bucket", "?")
                bs = s.get("bucket_stats") or []
                valid_bs = [b for b in bs if b is not None]

                # Walk-forward: split 70/30 and check if signal holds out-of-sample
                split_idx = int(len(rows) * 0.7)
                in_sample = rows[:split_idx]
                out_sample = rows[split_idx:]

                is_scan = oos_scan = None
                if len(in_sample) >= 60 and len(out_sample) >= 30:
                    try:
                        is_scan = scanner.scan_relationship(in_sample, x_col, y_col, ticker)
                        oos_scan = scanner.scan_relationship(out_sample, x_col, y_col, ticker)
                    except Exception:
                        pass

                entry_card = {
                    "type": "single_factor_entry",
                    "feature": x_col,
                    "y_col": y_col,
                    "ticker": ticker,
                    "best_bucket": bucket_num,
                    "bucket_stats": best_b,
                    "full_sample_score": s.get("composite_score"),
                    "pattern": s.get("pattern"),
                    "in_sample_score": is_scan.get("composite_score") if is_scan and "error" not in is_scan else None,
                    "out_sample_score": oos_scan.get("composite_score") if oos_scan and "error" not in oos_scan else None,
                    "rule": f"Enter when {x_col} is in bucket {bucket_num} → expect {y_col}",
                }
                await rdb.save_result(conn, run_id, "strategy_entry",
                                      x_col, y_col, entry_card, ticker)
                strategy_results.append(entry_card)

                oos_label = f"{entry_card.get('out_sample_score', '?')}" if entry_card.get("out_sample_score") else "N/A"
                log(f"  ✓ Strategy card: {x_col} bucket={bucket_num} "
                    f"IS_score={entry_card.get('in_sample_score', '?')} "
                    f"OOS_score={oos_label}")

    log(f"  {len(strategy_results)} strategy entry cards generated.")
    return strategy_results


# ── Phase 1: Main execution pipeline ─────────────────────────────────────────

async def execute_v2_pipeline(
    main_pool,
    oi_pool,
    run_id: str,
    question: str,
    plan: dict,
    config: dict,
    model: str = "claude-sonnet-4-6",
    knowledge_rules: list[str] = None,
    log=print,
) -> str:
    """
    Phase 1 execution:
      1. Fetch data for AI-selected columns
      2. Run scans
      3. Ask Claude which visuals to generate (3-6 charts)
      4. Generate only those visuals
      5. Compose focused narrative report
    """
    from research.engine import _fetch_cache  # reuse fetch logic

    table = config.get("table", "daily_features")
    tickers = config.get("tickers") or []
    date_from = config.get("date_from")
    date_to = config.get("date_to")

    x_cols = plan.get("feature_columns") or []
    y_cols = plan.get("outcome_columns") or []

    if not x_cols or not y_cols:
        raise ValueError("Workflow plan has no feature or outcome columns to scan.")

    data_pool = oi_pool if table in _OI_TABLES else main_pool

    # ── 1. Fetch ──────────────────────────────────────────────────────────
    log(f"[1/4] Fetching data — {len(x_cols)} features, {len(y_cols)} outcomes...")
    async with main_pool.acquire() as conn:
        await rdb.update_run(conn, run_id,
                             ai_summary=f"[RUNNING] Step 1/4: Fetching data…")

    cache, fetch_errors = await _fetch_cache(
        data_pool, table, tickers, x_cols, y_cols, date_from, date_to, log)
    if not cache:
        raise ValueError(f"No data loaded. {'; '.join(fetch_errors)}")

    # ── 2. Scan ───────────────────────────────────────────────────────────
    n_pairs = sum(
        1 for rows in cache.values()
        for x in x_cols for y in y_cols
        if x != y and x in (rows[0].keys() if rows else {}) and y in (rows[0].keys() if rows else {})
    )
    log(f"[2/4] Scanning {n_pairs} pairs…")
    async with main_pool.acquire() as conn:
        await rdb.update_run(conn, run_id,
                             ai_summary=f"[RUNNING] Step 2/4: Scanning {n_pairs} pairs…")

    all_scans = []
    async with main_pool.acquire() as conn:
        for cache_key, rows in cache.items():
            ticker = None if cache_key == "_all" else cache_key
            avail = set(rows[0].keys()) if rows else set()
            for x_col in x_cols:
                if x_col not in avail:
                    continue
                for y_col in y_cols:
                    if y_col not in avail or x_col == y_col:
                        continue
                    try:
                        scan = scanner.scan_relationship(rows, x_col, y_col, ticker)
                        n_pts = scan.get("n", 0)
                        if n_pts < len(rows) * 0.9:
                            # Count why rows were lost
                            n_x_none = sum(1 for r in rows if r.get(x_col) is None)
                            n_y_none = sum(1 for r in rows if r.get(y_col) is None)
                            import math
                            n_x_nan = sum(1 for r in rows if r.get(x_col) is not None
                                          and isinstance(r.get(x_col), float)
                                          and math.isnan(r.get(x_col)))
                            n_y_nan = sum(1 for r in rows if r.get(y_col) is not None
                                          and isinstance(r.get(y_col), float)
                                          and math.isnan(r.get(y_col)))
                            log(f"    {x_col}→{y_col}: {n_pts}/{len(rows)} pairs "
                                f"(x_none={n_x_none}, x_nan={n_x_nan}, "
                                f"y_none={n_y_none}, y_nan={n_y_nan})")
                        await rdb.save_result(conn, run_id, "scan", x_col, y_col, scan, ticker)
                        all_scans.append(scan)
                    except Exception as exc:
                        log(f"  SCAN ERROR {ticker or 'all'} {x_col}→{y_col}: {exc}")

    valid = [s for s in all_scans if "error" not in s]
    ranked = sorted(valid, key=lambda s: s.get("composite_score", 0), reverse=True)
    log(f"  {len(valid)} pairs scanned. "
        f"Top: {ranked[0].get('x_col')} → {ranked[0].get('y_col')} "
        f"score={ranked[0].get('composite_score', 0):.0f}" if ranked else "  No valid results.")

    # ── 2b. Adaptive branching ─────────────────────────────────────────────
    task_type = plan.get("task_type", "single-factor-scan")
    strong = [s for s in ranked if s.get("composite_score", 0) >= 25]
    weak_run = len(strong) == 0

    if weak_run and len(x_cols) < 8:
        log(f"  Adaptive: no signals ≥25 — skipping interaction scan.")
    elif task_type in ("multi-factor-interaction", "strategy-entry-condition") or len(strong) >= 2:
        # Run interaction analysis on top features
        log(f"[2b] Adaptive: {len(strong)} strong signals — running interaction scan…")
        async with main_pool.acquire() as conn:
            await rdb.update_run(conn, run_id,
                                 ai_summary=f"[RUNNING] Step 2b: Interaction scan ({len(strong)} strong signals)…")

        from itertools import combinations
        # Get top features per (ticker, y_col)
        feat_by_ty: dict[tuple, list[str]] = {}
        for s in ranked[:15]:
            key = (s.get("ticker"), s["y_col"])
            if s["x_col"] not in feat_by_ty.setdefault(key, []):
                feat_by_ty[key].append(s["x_col"])

        n_interactions = 0
        async with main_pool.acquire() as conn:
            for (ticker, y_col), feats in feat_by_ty.items():
                cache_key = ticker if ticker else "_all"
                rows = cache.get(cache_key, [])
                if not rows or len(feats) < 2:
                    continue

                # Best single-factor baseline for lift
                baseline = max(
                    (abs((s.get("best_single_bucket") or {}).get("sharpe", 0))
                     for s in ranked
                     if s.get("ticker") == ticker and s["y_col"] == y_col),
                    default=0,
                )

                top_feats = feats[:5]
                for fa, fb in combinations(top_feats, 2):
                    try:
                        result = scanner.scan_interaction_2f(
                            rows, fa, fb, y_col, ticker, baseline)
                    except Exception as exc:
                        log(f"    2F ERROR {ticker or 'all'} {fa}+{fb}→{y_col}: {exc}")
                        continue
                    if result is None:
                        continue
                    await rdb.save_result(conn, run_id, "interaction",
                                          f"{fa}+{fb}", y_col, result, ticker)
                    all_scans.append(result)
                    n_interactions += 1

                # 3-factor combos from top 2-factor features
                if len(top_feats) >= 3:
                    for fa, fb, fc in combinations(top_feats[:4], 3):
                        try:
                            result = scanner.scan_interaction_3f(
                                rows, fa, fb, fc, y_col, ticker, baseline)
                        except Exception as exc:
                            log(f"    3F ERROR {fa}+{fb}+{fc}: {exc}")
                            continue
                        if result is None:
                            continue
                        await rdb.save_result(conn, run_id, "interaction_3f",
                                              f"{fa}+{fb}+{fc}", y_col, result, ticker)
                        all_scans.append(result)
                        n_interactions += 1

        log(f"  {n_interactions} interaction combos tested.")
        # Re-rank with interactions included
        valid = [s for s in all_scans if "error" not in s]
        ranked = sorted(valid, key=lambda s: (
            s.get("composite_interaction_score") or s.get("composite_score") or 0
        ), reverse=True)

    # ── 2c. Task-type-specific analysis ────────────────────────────────────
    task_type = plan.get("task_type", "single-factor-scan")
    extra_context_lines = []

    if task_type == "regime-analysis":
        log("[2c] Running regime-conditional analysis…")
        async with main_pool.acquire() as conn:
            await rdb.update_run(conn, run_id,
                                 ai_summary="[RUNNING] Step 2c: Regime analysis…")
        regime_results = await _run_regime_analysis(
            cache, ranked, all_scans, x_cols, y_cols, main_pool, run_id, log)
        if regime_results:
            extra_context_lines.append("\n=== Regime-Conditional Results ===")
            for s in sorted(regime_results,
                            key=lambda x: x.get("composite_score", 0), reverse=True)[:10]:
                extra_context_lines.append(
                    f"  [{s.get('composite_score', 0):.0f}] {s.get('regime', '?')} | "
                    f"{s.get('x_col')} → {s.get('y_col')}: pattern={s.get('pattern')}")

    elif task_type == "event-study":
        log("[2c] Running event-study analysis…")
        async with main_pool.acquire() as conn:
            await rdb.update_run(conn, run_id,
                                 ai_summary="[RUNNING] Step 2c: Event study…")
        event_results = await _run_event_study(
            cache, x_cols, y_cols, main_pool, run_id, log)
        if event_results:
            extra_context_lines.append("\n=== Event-Conditional Results ===")
            for s in sorted(event_results,
                            key=lambda x: x.get("composite_score", 0), reverse=True)[:10]:
                extra_context_lines.append(
                    f"  [{s.get('composite_score', 0):.0f}] {s.get('event_window', '?')} | "
                    f"{s.get('x_col')} → {s.get('y_col')}: pattern={s.get('pattern')}")

    elif task_type == "strategy-entry-condition":
        log("[2c] Generating strategy entry conditions…")
        async with main_pool.acquire() as conn:
            await rdb.update_run(conn, run_id,
                                 ai_summary="[RUNNING] Step 2c: Strategy entry cards…")
        strategy_results = await _run_strategy_entry(
            cache, ranked, x_cols, y_cols, main_pool, run_id, log)
        if strategy_results:
            extra_context_lines.append("\n=== Strategy Entry Cards ===")
            for s in strategy_results:
                rule = s.get("rule", "?")
                oos = s.get("out_sample_score")
                extra_context_lines.append(
                    f"  Rule: {rule}\n"
                    f"    IS_score={s.get('in_sample_score', '?')}, "
                    f"OOS_score={oos if oos is not None else 'N/A'}")

    # ── 3. Equity curve data (for report context, no PNGs) ─────────────────
    log("[3/5] Computing equity curves for top signals…")
    async with main_pool.acquire() as conn:
        await rdb.update_run(conn, run_id,
                             ai_summary="[RUNNING] Step 3/5: Computing equity curves…")

    single_ranked = [s for s in ranked if s.get("x_col") and not s.get("combo")]
    equity_results = []
    async with main_pool.acquire() as conn:
        for s in single_ranked[:3]:
            x_col = s["x_col"]
            y_col = s["y_col"]
            ticker = s.get("ticker") or None
            cache_key = ticker if ticker else "_all"
            rows = cache.get(cache_key)
            if not rows:
                continue
            avail = set(rows[0].keys())
            if x_col not in avail or y_col not in avail:
                continue

            rob = s.get("robustness", {})
            conc = rob.get("concentration_risk", 1.0)
            min_bn = rob.get("min_bucket_n", 0)
            expanded = conc > 0.60 or min_bn < 20
            top_w = "top2" if expanded else "top"
            bot_w = "bottom2" if expanded else "bottom"

            try:
                top_eq = blocks.equity_curve_from_rows(rows, x_col, y_col, top_w, 10, ticker)
                bot_eq = blocks.equity_curve_from_rows(rows, x_col, y_col, bot_w, 10, ticker)
                await rdb.save_result(conn, run_id, "equity_curve_top", x_col, y_col, top_eq, ticker)
                await rdb.save_result(conn, run_id, "equity_curve_bottom", x_col, y_col, bot_eq, ticker)
                equity_results.extend([top_eq, bot_eq])
                log(f"  ✓ Equity: {ticker or 'all'} | {x_col} → {y_col}")
            except Exception as exc:
                log(f"  EQUITY ERROR {x_col}→{y_col}: {exc}")

    # ── 3b. Build ticker scoreboard ─────────────────────────────────────
    ticker_scoreboard = {}
    for s in valid:
        tk = s.get("ticker") or "all"
        score = s.get("composite_score", 0)
        if tk not in ticker_scoreboard or score > ticker_scoreboard[tk]["score"]:
            ticker_scoreboard[tk] = {
                "score": score,
                "x_col": s.get("x_col"),
                "y_col": s.get("y_col"),
                "pattern": s.get("pattern"),
                "consistency": (s.get("robustness") or {}).get("yearly_consistency_pct"),
            }
    if ticker_scoreboard:
        log(f"  Ticker scoreboard: {len(ticker_scoreboard)} tickers scored.")

    # ── 4. Compose report with inline charts ─────────────────────────────
    log("[4/5] Composing report with charts…")
    async with main_pool.acquire() as conn:
        await rdb.update_run(conn, run_id, ai_summary="[RUNNING] Step 4/5: Composing report…")

    extra_ctx = "\n".join(extra_context_lines) if extra_context_lines else ""
    sections = await _compose_report(question, plan, ranked, equity_results, model,
                                     knowledge_rules=knowledge_rules,
                                     extra_context=extra_ctx,
                                     ticker_scoreboard=ticker_scoreboard)
    log(f"  Report complete — {len(sections)} sections.")

    # ── 5. Skeptic review ────────────────────────────────────────────────
    log("[5/5] Running skeptic review…")
    async with main_pool.acquire() as conn:
        await rdb.update_run(conn, run_id, ai_summary="[RUNNING] Step 5/5: Skeptic review…")

    # Extract prose from sections for the skeptic to review
    report_text = "\n\n".join(
        s.get("content", "") for s in sections if s.get("type") == "markdown"
    )

    try:
        skeptic = await _skeptic_review(report_text, ranked, equity_results, model,
                                        knowledge_rules=knowledge_rules)
        if skeptic:
            log("  Skeptic review complete.")
            sections.append({"type": "markdown", "content": f"---\n\n## Caveats & Challenges\n\n{skeptic}"})
    except Exception as exc:
        log(f"  Skeptic review failed: {exc}. Using report without caveats.")

    return json.dumps(sections)
