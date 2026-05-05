"""
Research 2 orchestration layer — Phase 1.

Sits above the deterministic tools and decides:
  1. What kind of analysis the question requires (classify)
  2. Which columns / tools to run (plan)
  3. Which specific visuals are worth generating (direct)
  4. How to frame the final narrative (report)

The deterministic tools (scanner, blocks, charts) are unchanged building blocks.
"""
import asyncio
import json
import re
from typing import Optional

try:
    import anthropic as _anthropic
except ImportError:
    _anthropic = None

from research import scanner, blocks, db as rdb
from research import backtest as rbacktest

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
    "description": "Output the research report with prose narrative and chart placement.",
    "input_schema": {
        "type": "object",
        "properties": {
            "executive_summary": {
                "type": "string",
                "description": "2-3 paragraph executive summary answering the research question directly. Cite specific numbers.",
            },
            "body": {
                "type": "string",
                "description": "Main body of the report (300-500 words). Describe findings in detail, bucket profile shapes, what worked and what didn't. Cite scores, win rates, spreads.",
            },
            "conclusions": {
                "type": "string",
                "description": "Actionable conclusions. Specific trading rules or recommendations with numbers.",
            },
            "chart_sequence": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Ordered list of chart_ids to display between the body paragraphs. Use IDs from the AVAILABLE CHARTS list.",
            },
        },
        "required": ["executive_summary", "body", "conclusions", "chart_sequence"],
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
- IMPORTANT: forward_Nd columns (forward_1d, forward_7d, forward_30d, forward_90d, forward_180d)
  are FORWARD IMPLIED VOLATILITY levels, NOT forward price returns. They must be FEATURE columns,
  never outcome columns. Outcome columns for return analysis should be ret_*_fwd_* columns only.
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

Produce a research report using the produce_report tool. The tool has these fields:

- executive_summary: 2-3 paragraphs answering the research question directly. Cite numbers.
- body: 300-500 words of detailed findings. Describe bucket profiles, patterns, what worked/didn't.
- conclusions: actionable trading rules or recommendations with specific numbers.
- chart_sequence: ordered list of chart_ids (from AVAILABLE CHARTS) to display in the report.

ALL THREE TEXT FIELDS (executive_summary, body, conclusions) are REQUIRED and must contain
substantial prose. The text IS the report — charts illustrate it.

WRITING GUIDELINES:
- Answer the research question directly in executive_summary
- Describe the full bucket profile shape — not just top/bottom
- If multi-factor combos were tested, describe as usable trading rules
- Distinguish tradable from spurious. Cite specific numbers (scores, spreads, win rates)
- State what did NOT work
- Professional quant voice, no fluff
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
    user_requested_features = []
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
                user_requested_features.append(col_name)
    plan["user_requested_features"] = user_requested_features

    # forward_* columns are forward IV levels, not returns — always features, never outcomes
    moved_fwd = [c for c in plan["outcome_columns"] if c.startswith("forward_")]
    if moved_fwd:
        plan["outcome_columns"] = [c for c in plan["outcome_columns"] if not c.startswith("forward_")]
        for c in moved_fwd:
            if c not in plan["feature_columns"]:
                plan["feature_columns"].append(c)

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
    user_requested_features: list[str] = None,
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

    # 2. Bucket profiles — prioritize user-requested features, then top by score
    uf = set(user_requested_features or [])
    # Sort: user-requested features first, then by score
    profile_candidates = sorted(single, key=lambda s: (
        0 if s.get('x_col') in uf else 1,
        -(s.get('composite_score') or 0),
    ))
    seen_profiles = set()
    for s in profile_candidates:
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
        n_bucket_charts = sum(1 for k in charts if k.startswith('bucket_'))
        if n_bucket_charts >= 8:
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
    chart_configs = _build_chart_configs(ranked_scans, equity_results, ticker_scoreboard,
                                        user_requested_features=plan.get("user_requested_features"))

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

    # Bucket profiles — prioritize user-requested features, then top by score
    user_feats = set(plan.get("user_requested_features") or [])
    bp_shown = set()
    bp_candidates = []
    # User-requested signals first
    if user_feats:
        for s in valid:
            if s.get("x_col") in user_feats and s.get("x_col") not in bp_shown:
                bp_candidates.append(s)
                bp_shown.add(s.get("x_col"))
    # Fill remaining with top by score
    for s in top:
        key = f"{s.get('ticker')}_{s.get('x_col')}_{s.get('y_col')}"
        if key not in bp_shown:
            bp_candidates.append(s)
            bp_shown.add(key)
        if len(bp_candidates) >= 8:
            break

    bp_lines = []
    for s in bp_candidates[:8]:
        bs = [b for b in (s.get("bucket_stats") or []) if b is not None]
        if bs:
            marker = " ★USER-REQUESTED" if s.get("x_col") in user_feats else ""
            bp_lines.append(
                f"\n{s.get('ticker') or 'all'} | {s.get('x_col')} → {s.get('y_col')} "
                f"(score={s.get('composite_score', 0):.0f}, pattern={s.get('pattern')}){marker}:"
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

    # Tell the LLM which features were explicitly requested
    if user_feats:
        user_feat_note = (
            f"\n\nUSER-REQUESTED FEATURES (the user specifically asked about these — "
            f"they MUST be the primary focus of your analysis narrative, "
            f"even if other features scored higher):\n"
            + ", ".join(sorted(user_feats))
        )
    else:
        user_feat_note = ""

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
    if user_feat_note:
        user_content += user_feat_note
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

    # Extract structured output — separate prose fields + chart sequence
    report_data = None
    for block in msg.content:
        if block.type == "tool_use" and block.name == "produce_report":
            report_data = block.input
            break

    sections = []

    if report_data is None:
        # Fallback: raw text + all charts
        raw = msg.content[0].text if msg.content and hasattr(msg.content[0], "text") else ""
        sections.append({"type": "markdown", "content": raw.strip() or "(Report generation failed)"})
        for cid, chart in chart_configs.items():
            sections.append({"type": "chart", "chart_id": cid, "title": chart["title"], "config": chart["config"]})
        return sections

    # Build sections from the structured fields
    exec_summary = (report_data.get("executive_summary") or "").strip()
    body = (report_data.get("body") or "").strip()
    conclusions = (report_data.get("conclusions") or "").strip()
    chart_seq = report_data.get("chart_sequence") or []

    # If ALL prose fields are empty, make a fallback prose-only call
    if not exec_summary and not body and not conclusions:
        fallback_msg = await client.messages.create(
            model=model,
            max_tokens=2000,
            system=[{"type": "text", "text": report_system,
                     "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content":
                user_content.split("AVAILABLE CHARTS")[0] +
                "\n\nWrite a focused research report (400-700 words) answering the research question. "
                "Professional quant voice. No fluff. Cite specific numbers."
            }],
        )
        prose = fallback_msg.content[0].text.strip() if fallback_msg.content else "(Report generation failed)"
        sections.append({"type": "markdown", "content": prose})
        for cid, chart in chart_configs.items():
            sections.append({"type": "chart", "chart_id": cid, "title": chart["title"], "config": chart["config"]})
        return sections

    # Executive summary first
    if exec_summary:
        sections.append({"type": "markdown", "content": exec_summary})

    # Interleave body paragraphs with charts
    # Split body into paragraphs, distribute charts between them
    body_paras = [p.strip() for p in body.split("\n\n") if p.strip()] if body else []
    referenced = set()

    if body_paras and chart_seq:
        # Distribute charts evenly among paragraphs
        charts_per_gap = max(1, len(chart_seq) // max(len(body_paras), 1))
        chart_idx = 0
        for i, para in enumerate(body_paras):
            sections.append({"type": "markdown", "content": para})
            # Insert charts after this paragraph
            for _ in range(charts_per_gap):
                if chart_idx < len(chart_seq):
                    cid = chart_seq[chart_idx]
                    if cid in chart_configs:
                        sections.append({
                            "type": "chart", "chart_id": cid,
                            "title": chart_configs[cid]["title"],
                            "config": chart_configs[cid]["config"],
                        })
                        referenced.add(cid)
                    chart_idx += 1
        # Any remaining charts
        while chart_idx < len(chart_seq):
            cid = chart_seq[chart_idx]
            if cid in chart_configs:
                sections.append({
                    "type": "chart", "chart_id": cid,
                    "title": chart_configs[cid]["title"],
                    "config": chart_configs[cid]["config"],
                })
                referenced.add(cid)
            chart_idx += 1
    elif body_paras:
        for para in body_paras:
            sections.append({"type": "markdown", "content": para})
    elif chart_seq:
        for cid in chart_seq:
            if cid in chart_configs:
                sections.append({
                    "type": "chart", "chart_id": cid,
                    "title": chart_configs[cid]["title"],
                    "config": chart_configs[cid]["config"],
                })
                referenced.add(cid)

    # Conclusions
    if conclusions:
        sections.append({"type": "markdown", "content": f"## Conclusions\n\n{conclusions}"})

    # Append any charts not referenced in chart_sequence
    for cid, chart in chart_configs.items():
        if cid not in referenced:
            sections.append({
                "type": "chart", "chart_id": cid,
                "title": chart["title"], "config": chart["config"],
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


# ── Agentic P&L pipeline ─────────────────────────────────────────────────────

_AGENTIC_TOOLS = [
    {
        "name": "run_correlation_scan",
        "description": "Scan multiple IV metric columns vs the P&L column. Returns ranked correlation table.",
        "input_schema": {
            "type": "object",
            "properties": {
                "x_cols": {"type": "array", "items": {"type": "string"},
                           "description": "IV metric column names to scan"},
                "y_col":  {"type": "string", "description": "Outcome column (usually 'pnl')"},
            },
            "required": ["x_cols", "y_col"],
        },
    },
    {
        "name": "run_regression",
        "description": "OLS regression of y_col on x_cols. Returns r2, coefficients.",
        "input_schema": {
            "type": "object",
            "properties": {
                "x_cols": {"type": "array", "items": {"type": "string"}},
                "y_col":  {"type": "string"},
            },
            "required": ["x_cols", "y_col"],
        },
    },
    {
        "name": "run_lag_scan",
        "description": "Pearson r between x_col[t-lag] and y_col[t] at specified lags (in 5-min bars).",
        "input_schema": {
            "type": "object",
            "properties": {
                "x_col": {"type": "string"},
                "y_col": {"type": "string"},
                "lags":  {"type": "array", "items": {"type": "integer"},
                          "description": "Lag values in bars, e.g. [1, 5, 10, 30]"},
            },
            "required": ["x_col", "y_col"],
        },
    },
    {
        "name": "run_regime_split",
        "description": "Split data at median or tercile of split_col and compare x→y correlation in each regime.",
        "input_schema": {
            "type": "object",
            "properties": {
                "x_col":     {"type": "string"},
                "y_col":     {"type": "string"},
                "split_col": {"type": "string", "description": "Column used to define regimes"},
                "method":    {"type": "string", "enum": ["median", "tercile"],
                              "description": "Split method"},
            },
            "required": ["x_col", "y_col", "split_col"],
        },
    },
    {
        "name": "run_rolling_correlation",
        "description": "Rolling window Pearson r between x_col and y_col over time.",
        "input_schema": {
            "type": "object",
            "properties": {
                "x_col":  {"type": "string"},
                "y_col":  {"type": "string"},
                "window": {"type": "integer", "description": "Window size in bars"},
            },
            "required": ["x_col", "y_col"],
        },
    },
    {
        "name": "run_tail_analysis",
        "description": "Compare IV metric means between top/bottom pct% P&L rows.",
        "input_schema": {
            "type": "object",
            "properties": {
                "y_col":  {"type": "string"},
                "x_cols": {"type": "array", "items": {"type": "string"}},
                "pct":    {"type": "integer", "description": "Percentile cutoff, e.g. 10 for top/bottom 10%"},
            },
            "required": ["y_col", "x_cols"],
        },
    },
    {
        "name": "run_decile_profile",
        "description": "Decile bucket profile showing how y_col varies across 10 buckets of x_col.",
        "input_schema": {
            "type": "object",
            "properties": {
                "x_col": {"type": "string"},
                "y_col": {"type": "string"},
            },
            "required": ["x_col", "y_col"],
        },
    },
    {
        "name": "run_greek_attribution",
        "description": "Regression of P&L on position greeks (delta, theta, vega, gamma). Reveals how much P&L is explained by greeks vs unexplained IV changes.",
        "input_schema": {
            "type": "object",
            "properties": {
                "pnl_col": {"type": "string", "description": "P&L column name (usually 'pnl')"},
            },
            "required": ["pnl_col"],
        },
    },
    {
        "name": "write_report",
        "description": "Write the final research report. Call this when you have enough findings.",
        "input_schema": {
            "type": "object",
            "properties": {
                "executive_summary": {
                    "type": "string",
                    "description": "2-3 paragraph executive summary answering the research question.",
                },
                "body": {
                    "type": "string",
                    "description": "Detailed findings (300-500 words). Cite specific numbers.",
                },
                "conclusions": {
                    "type": "string",
                    "description": "Actionable conclusions with specific numbers.",
                },
                "chart_sequence": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Ordered chart IDs to display (from the AVAILABLE CHARTS list).",
                },
            },
            "required": ["executive_summary", "body", "conclusions", "chart_sequence"],
        },
    },
]

_AGENTIC_SYSTEM = """\
You are a quantitative analyst exploring how SPX implied-volatility metrics relate to \
a nondirectional options strategy P&L. You have a merged dataset of 5-minute P&L slices \
and IV surface metrics.

Use the analysis tools iteratively to explore the data. Call 4-8 tools before writing \
your report. Build up a picture of:
  - Which IV metrics correlate most with P&L
  - Whether relationships hold across different IV regimes
  - Whether there are lead/lag dynamics
  - What the greek attribution reveals about unexplained P&L

When you have enough findings, call write_report to produce the final report.\
"""

# ── Backtest agentic tools/system ─────────────────────────────────────────────

_BACKTEST_AGENTIC_TOOLS = [
    t for t in _AGENTIC_TOOLS
    if t["name"] not in ("run_lag_scan", "run_greek_attribution", "write_report")
] + [
    {
        "name": "run_win_rate_analysis",
        "description": (
            "Bucket trades by an IV metric column and compute win rate + mean P&L per bucket. "
            "Use this to find non-linear IV thresholds that predict binary trade outcomes. "
            "More informative than Pearson r for win/loss analysis."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "x_col":     {"type": "string",  "description": "IV metric column to bucket by"},
                "n_buckets": {"type": "integer", "description": "Number of equal-count buckets (default 5)"},
            },
            "required": ["x_col"],
        },
    },
    {
        "name": "run_time_split_validation",
        "description": (
            "Split trades chronologically and check if top IV correlations are stable across time periods. "
            "Use this on any promising signal. Default n_splits=3 when n≥600, else 2."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "y_col":    {"type": "string",  "description": "Outcome column (pnl or is_win)"},
                "n_splits": {"type": "integer", "description": "Number of chronological periods (3 recommended for large samples)"},
            },
            "required": ["y_col"],
        },
    },
    {
        "name": "run_feature_redundancy_check",
        "description": (
            "Compute pairwise Pearson r among candidate features and group redundant ones (|r| ≥ threshold). "
            "Call this after discovery to get one non-redundant representative per feature family before "
            "building grids or composites."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "features":    {"type": "array", "items": {"type": "string"},
                                "description": "List of IV metric column names to check"},
                "r_threshold": {"type": "number",
                                "description": "Redundancy threshold — features with |r| ≥ this are grouped (default 0.7)"},
            },
            "required": ["features"],
        },
    },
    {
        "name": "run_two_factor_regime",
        "description": (
            "Build a 2-factor outcome grid (n_bins × n_bins cells) showing mean_y and win_rate per cell. "
            "Returns marginals (main effects) and interaction_strength (max deviation from additive prediction). "
            "Use as the discovery layer — run on several non-redundant feature pairs to find interaction patterns."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "factor1": {"type": "string", "description": "First IV metric column"},
                "factor2": {"type": "string", "description": "Second IV metric column"},
                "y_col":   {"type": "string", "description": "Outcome column (pnl or is_win)"},
                "n_bins":  {"type": "integer", "description": "Bins per factor (default 3 → 3×3 grid)"},
            },
            "required": ["factor1", "factor2", "y_col"],
        },
    },
    {
        "name": "run_composite_regime_score",
        "description": (
            "Build a composite regime score from 2–5 component features, apply optional interactions, "
            "and validate out-of-sample on a chronological holdout. "
            "Returns quintile outcome profiles for full/train/validate splits, train_r, validate_r, "
            "and comparison vs the best single component. "
            "Only call this after finding ≥2 stable, non-redundant features via grids and time-split validation."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "components": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "feature":   {"type": "string"},
                            "direction": {"type": "integer",
                                          "description": "1 = higher is favorable, -1 = lower is favorable"},
                        },
                        "required": ["feature", "direction"],
                    },
                    "description": "2–5 feature components with their outcome direction",
                },
                "interactions": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "feature1": {"type": "string"},
                            "feature2": {"type": "string"},
                            "type": {"type": "string", "enum": ["amplify", "conditional"],
                                     "description": "amplify: bonus when both point same way. conditional: feature1 double-weighted when feature2 is extreme."},
                        },
                        "required": ["feature1", "feature2", "type"],
                    },
                    "description": "Optional interaction adjustments (max 4). Only add if grid scan showed evidence of interaction.",
                },
                "y_col":       {"type": "string",  "description": "Outcome column (default pnl)"},
                "train_frac":  {"type": "number",  "description": "Fraction for training split (default 0.6)"},
            },
            "required": ["components"],
        },
    },
    {
        "name": "write_report",
        "description": "Write the final research report. Call this when you have enough findings.",
        "input_schema": {
            "type": "object",
            "properties": {
                "executive_summary": {
                    "type": "string",
                    "description": "2-3 paragraph executive summary answering the research question.",
                },
                "body": {
                    "type": "string",
                    "description": "Detailed findings (300-500 words). Cite specific numbers.",
                },
                "conclusions": {
                    "type": "string",
                    "description": "Actionable conclusions with specific numbers.",
                },
                "chart_sequence": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Ordered chart IDs to display (from the AVAILABLE CHARTS list).",
                },
            },
            "required": ["executive_summary", "body", "conclusions", "chart_sequence"],
        },
    },
]

_BACKTEST_AGENTIC_SYSTEM = """\
You are a quantitative analyst evaluating a completed options backtest. Each row is a \
completed trade with entry date, exit date, P&L, and SPX implied-volatility surface \
metrics captured at trade entry (09:35 on the entry date — entry-morning conditions).

Your goal: find which entry-time IV conditions PREDICT better or worse trade outcomes, \
and build a compact, validated regime description if the data supports it.

Key outcome columns: pnl (continuous), is_win (binary 0/1), days_in_trade.
IV metrics are surface_metrics_core columns (skew, term structure, convexity, level, etc.).

ANALYSIS WORKFLOW — follow this sequence:

STEP 1 — DISCOVERY: Run run_correlation_scan (both pnl and is_win) and run_win_rate_analysis \
on the strongest candidates to identify the top 10–15 predictive features.

STEP 2 — REDUNDANCY: Run run_feature_redundancy_check on those top features. Use the returned \
non_redundant list for all subsequent steps. Avoid building composites from correlated features.

STEP 3 — GRID SCAN: Run run_two_factor_regime on 3–5 promising non-redundant pairs (spanning \
different conceptual domains where possible). For each grid, look for:
  - Main effects: consistent marginal differences across factor bins.
  - Interaction effects: cells with mean_y far from additive prediction (high interaction_strength).
  - Conditional effects: one factor only matters when another is favorable/unfavorable.

STEP 4 — STABILITY: Run run_time_split_validation on the best 2–3 features. Use n_splits=3 \
if you have ≥600 trades. Only features that appear stable across periods are candidates for the composite.

STEP 5 — COMPOSITE (only if ≥2 stable non-redundant features found): Run run_composite_regime_score:
  - components: 2–4 features with their directions (based on correlation sign from step 1).
  - interactions: add amplify or conditional ONLY if the grid scan showed clear evidence. Max 4 interactions.
  - Check validate_r vs best_single_validate_r. Report whether the composite improves over the \
    best single metric. If composite_vs_single ≤ 0 or stable=false, say so explicitly.

STEP 6 — REPORT: Describe favorable and unfavorable regime conditions in market terms \
(e.g. "high term-structure slope + low skew → mean P&L $X, win rate Y%"). \
Cite actual numbers from validate_quintiles, not train. Flag unstable or low-sample findings. \
Recommend whether this is useful for filtering, sizing, or exploratory only.

Skip step 5 if fewer than 2 stable features are found — report that no composite is warranted.
Call 6–12 tools before write_report.\
"""


# ── Intratrade path tools/system ─────────────────────────────────────────────

_INTRATRADE_AGENTIC_TOOLS = [
    t for t in _AGENTIC_TOOLS
    if t["name"] not in ("run_regression", "run_lag_scan", "run_greek_attribution",
                          "write_report")
] + [
    {
        "name": "run_days_in_trade_profile",
        "description": (
            "Profile how y_col evolves over trade duration. "
            "group_by_phase=true groups by 'early'/'middle'/'late' phase labels (normalized for trade length). "
            "group_by_phase=false bins by absolute DIT value into n_bins equal-width bins. "
            "Use this to find which phase of the trade drives most damage or recovery."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "y_col":           {"type": "string", "description": "Column to profile (e.g. daily_pnl_change, drawdown_from_peak)"},
                "n_bins":          {"type": "integer", "description": "Number of DIT bins when group_by_phase=false (default 10)"},
                "group_by_phase":  {"type": "boolean", "description": "If true, group by early/middle/late phase instead of DIT bins"},
            },
            "required": ["y_col"],
        },
    },
    {
        "name": "run_path_event_scan",
        "description": (
            "Find daily rows where event_col crosses a threshold, then report mean context_col values "
            "window_days before and after each crossing. "
            "threshold_type='percentile' with direction='below_pct' triggers on the worst N% of event_col values. "
            "Use this to answer: 'What IV conditions exist just before losses accelerate?'"
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "event_col":      {"type": "string", "description": "Column to watch for crossings (e.g. drawdown_from_peak, daily_pnl_change)"},
                "threshold":      {"type": "number", "description": "Fixed value OR percentile N (0-100) depending on threshold_type"},
                "direction":      {"type": "string", "enum": ["above", "below", "above_pct", "below_pct"],
                                   "description": "above/below for fixed threshold; above_pct/below_pct for percentile"},
                "context_cols":   {"type": "array", "items": {"type": "string"},
                                   "description": "IV or greek columns to measure before/after the event"},
                "window_days":    {"type": "integer", "description": "Days before and after to include in context (default 3)"},
                "threshold_type": {"type": "string", "enum": ["fixed", "percentile"],
                                   "description": "fixed = literal value; percentile = derive threshold from Nth pct of distribution"},
            },
            "required": ["event_col", "threshold", "direction", "context_cols"],
        },
    },
    {
        "name": "write_report",
        "description": "Write the final research report. Call this when you have enough findings.",
        "input_schema": {
            "type": "object",
            "properties": {
                "executive_summary": {
                    "type": "string",
                    "description": "2-3 paragraph executive summary answering the research question.",
                },
                "body": {
                    "type": "string",
                    "description": "Detailed findings (300-500 words). Cite specific numbers.",
                },
                "conclusions": {
                    "type": "string",
                    "description": "Actionable conclusions with specific numbers.",
                },
                "chart_sequence": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Ordered chart IDs to display (from the AVAILABLE CHARTS list).",
                },
            },
            "required": ["executive_summary", "body", "conclusions", "chart_sequence"],
        },
    },
]

_INTRATRADE_AGENTIC_SYSTEM = """\
You are a quantitative analyst exploring how options trades evolve day-by-day after entry. \
Each row in the dataset is one trade on one calendar day (NOT a completed trade). \
The dataset contains daily P&L snapshots and IV surface conditions across all days each position was open.

KEY COLUMNS:
  pos_pnl        — cumulative P&L since entry at this point in time (not final outcome)
  daily_pnl_change — day-over-day change in pos_pnl (PREFER THIS over pos_pnl for correlations)
  drawdown_from_peak — pos_pnl minus the running maximum P&L for this trade (0 or negative)
  max_pnl_to_date — highest pos_pnl this trade has seen up to this day
  trade_phase    — early/middle/late based on % of total trade duration
  dit            — absolute days in trade at this snapshot
  *_since_entry  — change in an IV metric since trade entry (PREFER THESE over level columns for correlations)

ANALYSIS GUIDANCE:
  - Correlate IV against daily_pnl_change or drawdown_from_peak, NOT raw pos_pnl. \
    Raw pos_pnl drifts with DIT and will produce spurious correlations with any time-correlated metric.
  - IV *_since_entry columns measure what changed in the market since entry — more informative than \
    absolute IV levels for understanding why trades deteriorate.
  - Use run_days_in_trade_profile with group_by_phase=true to compare early vs middle vs late path behavior.
  - Use run_path_event_scan with threshold_type='percentile', direction='below_pct' to find IV conditions \
    surrounding the worst damage days.
  - Use run_regime_split with split_col='trade_phase' to compare IV signal strength across trade phases.

Call 4-8 tools before writing your report. Focus on actionable patterns: which IV conditions predict \
daily P&L damage, when in the trade life do losses typically accelerate, and which IV changes since \
entry are most associated with drawdowns.\
"""


def _build_intratrade_chart_configs(tool_results: list[dict]) -> dict[str, dict]:
    """Build Chart.js configs from accumulated intratrade agentic tool results."""
    charts = {}
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

    for tr in tool_results:
        tool_name = tr.get('tool')
        result    = tr.get('result') or {}

        if tool_name == 'run_correlation_scan' and isinstance(result, list) and result:
            top = result[:15]
            labels = [d['x_col'] for d in top]
            values = [d['r'] for d in top]
            y_col  = top[0].get('y_col', 'outcome')
            cid = f"corr_scan_{len(charts)}"
            charts[cid] = {
                'title': f"IV Correlation with {y_col} (top {len(top)})",
                'config': {
                    'type': 'bar',
                    'data': {
                        'labels': labels,
                        'datasets': [{'label': 'Pearson r', 'data': values,
                                      'backgroundColor': [colors_pos if v >= 0 else colors_neg for v in values],
                                      'borderWidth': 0}],
                    },
                    'options': {**base_opts, 'plugins': {**base_opts['plugins'], 'legend': {'display': False}}},
                },
            }

        elif tool_name == 'run_days_in_trade_profile':
            bins = result.get('bins') or []
            y_col = result.get('y_col', '?')
            by_phase = result.get('group_by_phase', False)
            if len(bins) >= 2:
                if by_phase:
                    labels = [b.get('phase', '?') for b in bins]
                else:
                    labels = [f"DIT {b.get('dit_min', '?'):.0f}–{b.get('dit_max', '?'):.0f}" for b in bins]
                means = [b.get('mean') for b in bins]
                ns    = [b.get('n', 0) for b in bins]
                cid   = f"dit_profile_{y_col}_{len(charts)}"
                charts[cid] = {
                    'title': f"{'Phase' if by_phase else 'DIT'} Profile: {y_col} (trend: {result.get('trend','?')})",
                    'config': {
                        'type': 'bar',
                        'data': {
                            'labels': labels,
                            'datasets': [{
                                'label': f'Mean {y_col}',
                                'data': means,
                                'backgroundColor': [colors_pos if (v or 0) >= 0 else colors_neg for v in means],
                                'borderWidth': 0,
                            }],
                        },
                        'options': {**base_opts,
                                    'plugins': {**base_opts['plugins'], 'legend': {'display': False}}},
                    },
                }

        elif tool_name == 'run_path_event_scan':
            before = result.get('mean_context_before') or {}
            after  = result.get('mean_context_after') or {}
            cols   = [c for c in before if c in after and before[c] is not None and after[c] is not None]
            if cols:
                ev_col    = result.get('event_col', '?')
                threshold = result.get('computed_threshold_value', result.get('threshold', '?'))
                cid = f"path_event_{ev_col}_{len(charts)}"
                charts[cid] = {
                    'title': f"Context around {ev_col} crossings (n={result.get('event_count', 0)})",
                    'config': {
                        'type': 'bar',
                        'data': {
                            'labels': cols,
                            'datasets': [
                                {'label': 'Before crossing', 'data': [before[c] for c in cols],
                                 'backgroundColor': colors_pos, 'borderWidth': 0},
                                {'label': 'After crossing', 'data': [after[c] for c in cols],
                                 'backgroundColor': colors_neg, 'borderWidth': 0},
                            ],
                        },
                        'options': base_opts,
                    },
                }

        elif tool_name == 'run_decile_profile':
            bs = result.get('bucket_stats') or []
            bs = [b for b in bs if b is not None]
            if len(bs) >= 3:
                x_col = result.get('x_col', '?')
                y_col = result.get('y_col', '?')
                avgs  = [b.get('avg_y', b.get('avg_ret', 0)) for b in bs]
                cid = f"decile_{x_col}_{y_col}"
                charts[cid] = {
                    'title': f"Decile Profile: {x_col} → {y_col} (r={result.get('pearson_r', 0):.3f})",
                    'config': {
                        'type': 'bar',
                        'data': {
                            'labels': [f"D{b['bucket']}" for b in bs],
                            'datasets': [{'label': y_col, 'data': avgs,
                                          'backgroundColor': [colors_pos if v >= 0 else colors_neg for v in avgs],
                                          'borderWidth': 0}],
                        },
                        'options': {**base_opts,
                                    'plugins': {**base_opts['plugins'], 'legend': {'display': False}}},
                    },
                }

        elif tool_name == 'run_tail_analysis':
            diff  = result.get('difference') or {}
            if diff:
                top_cols = list(diff.keys())[:10]
                vals  = [diff[c] for c in top_cols]
                y_col = result.get('y_col', 'outcome')
                cid   = f"tail_{y_col}_{len(charts)}"
                charts[cid] = {
                    'title': f"IV Diff: Best vs Worst {result.get('pct', 10)}% {y_col}",
                    'config': {
                        'type': 'bar',
                        'data': {
                            'labels': top_cols,
                            'datasets': [{'label': 'Top% minus Bottom% IV mean', 'data': vals,
                                          'backgroundColor': [colors_pos if v >= 0 else colors_neg for v in vals],
                                          'borderWidth': 0}],
                        },
                        'options': {**base_opts,
                                    'plugins': {**base_opts['plugins'], 'legend': {'display': False}}},
                    },
                }

        if len(charts) >= 8:
            break

    return charts


def _build_agentic_chart_configs(tool_results: list[dict]) -> dict[str, dict]:
    """Build Chart.js configs from accumulated agentic tool results."""
    charts = {}
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

    for tr in tool_results:
        tool_name = tr.get('tool')
        result    = tr.get('result') or {}

        if tool_name == 'run_correlation_scan' and isinstance(result, list) and result:
            top = result[:15]
            labels = [d['x_col'] for d in top]
            values = [d['r'] for d in top]
            cid = f"corr_scan_{len(charts)}"
            charts[cid] = {
                'title': f"Correlation with {top[0]['y_col']} (top {len(top)})",
                'config': {
                    'type': 'bar',
                    'data': {
                        'labels': labels,
                        'datasets': [{
                            'label': 'Pearson r',
                            'data': values,
                            'backgroundColor': [colors_pos if v >= 0 else colors_neg for v in values],
                            'borderWidth': 0,
                        }],
                    },
                    'options': {**base_opts,
                                'plugins': {**base_opts['plugins'], 'legend': {'display': False}}},
                },
            }

        elif tool_name == 'run_decile_profile':
            bs = result.get('bucket_stats') or []
            bs = [b for b in bs if b is not None]
            if len(bs) >= 3:
                x_col = result.get('x_col', '?')
                y_col = result.get('y_col', '?')
                avgs = [round(b['avg_ret'] * 100, 4) if 'avg_ret' in b else b.get('avg_y', 0) for b in bs]
                cid = f"decile_{x_col}_{y_col}"
                charts[cid] = {
                    'title': f"Decile Profile: {x_col} → {y_col} (r={result.get('pearson_r', 0):.3f})",
                    'config': {
                        'type': 'bar',
                        'data': {
                            'labels': [f"D{b['bucket']}" for b in bs],
                            'datasets': [{
                                'label': y_col,
                                'data': avgs,
                                'backgroundColor': [colors_pos if v >= 0 else colors_neg for v in avgs],
                                'borderWidth': 0,
                            }],
                        },
                        'options': {**base_opts,
                                    'plugins': {**base_opts['plugins'], 'legend': {'display': False}}},
                    },
                }

        elif tool_name == 'run_rolling_correlation' and isinstance(result, list) and len(result) > 5:
            x_col = tr.get('inputs', {}).get('x_col', '?')
            y_col = tr.get('inputs', {}).get('y_col', '?')
            step = max(1, len(result) // 80)
            sampled = result[::step]
            cid = f"rolling_corr_{x_col}_{y_col}"
            charts[cid] = {
                'title': f"Rolling Correlation: {x_col} → {y_col}",
                'config': {
                    'type': 'line',
                    'data': {
                        'labels': [p['date'][:16] for p in sampled],
                        'datasets': [{
                            'label': 'Rolling r',
                            'data': [p['r'] for p in sampled],
                            'borderColor': colors_pos, 'backgroundColor': 'transparent',
                            'borderWidth': 2, 'pointRadius': 0, 'tension': 0.1,
                        }],
                    },
                    'options': {**base_opts,
                                'scales': {**dark_scales,
                                           'x': {**dark_scales['x'],
                                                  'ticks': {**dark_scales['x']['ticks'],
                                                             'maxTicksLimit': 10}}}},
                },
            }

        elif tool_name == 'run_tail_analysis':
            diff = result.get('difference') or {}
            if diff:
                top_items = list(diff.items())[:12]
                labels = [k for k, _ in top_items]
                values = [v for _, v in top_items]
                cid = f"tail_{result.get('y_col', 'pnl')}"
                charts[cid] = {
                    'title': f"IV Metrics: Top vs Bottom {result.get('pct', 10)}% P&L",
                    'config': {
                        'type': 'bar',
                        'data': {
                            'labels': labels,
                            'datasets': [{
                                'label': 'Difference (top − bottom)',
                                'data': values,
                                'backgroundColor': [colors_pos if v >= 0 else colors_neg for v in values],
                                'borderWidth': 0,
                            }],
                        },
                        'options': {**base_opts,
                                    'plugins': {**base_opts['plugins'], 'legend': {'display': False}}},
                    },
                }

        elif tool_name == 'run_regime_split':
            high = result.get('high') or {}
            low  = result.get('low') or {}
            if high.get('r') is not None and low.get('r') is not None:
                x_col = result.get('x_col', '?')
                sc    = result.get('split_col', '?')
                cid = f"regime_{sc}_{x_col}"
                charts[cid] = {
                    'title': f"Regime Split: {x_col}→P&L by {sc}",
                    'config': {
                        'type': 'bar',
                        'data': {
                            'labels': ['High regime', 'Low regime'],
                            'datasets': [
                                {
                                    'label': 'Pearson r',
                                    'data': [round(high['r'], 4), round(low['r'], 4)],
                                    'backgroundColor': [colors_pos, colors_neg],
                                    'borderWidth': 0,
                                },
                                {
                                    'label': 'Mean P&L',
                                    'data': [high.get('mean_y', 0), low.get('mean_y', 0)],
                                    'backgroundColor': ['rgba(52,152,219,0.35)', 'rgba(232,67,147,0.35)'],
                                    'borderWidth': 0,
                                },
                            ],
                        },
                        'options': base_opts,
                    },
                }

        if len(charts) >= 8:
            break

    return charts


def _build_backtest_chart_configs(tool_results: list[dict]) -> dict[str, dict]:
    """Build Chart.js configs from accumulated backtest agentic tool results."""
    charts = {}
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

    for tr in tool_results:
        tool_name = tr.get('tool')
        result    = tr.get('result') or {}

        if tool_name == 'run_correlation_scan' and isinstance(result, list) and result:
            top = result[:15]
            labels = [d['x_col'] for d in top]
            values = [d['r'] for d in top]
            y_col  = top[0].get('y_col', 'outcome')
            cid = f"corr_scan_{len(charts)}"
            charts[cid] = {
                'title': f"IV Correlation with {y_col} (top {len(top)})",
                'config': {
                    'type': 'bar',
                    'data': {
                        'labels': labels,
                        'datasets': [{
                            'label': 'Pearson r',
                            'data': values,
                            'backgroundColor': [colors_pos if v >= 0 else colors_neg for v in values],
                            'borderWidth': 0,
                        }],
                    },
                    'options': {**base_opts,
                                'plugins': {**base_opts['plugins'], 'legend': {'display': False}}},
                },
            }

        elif tool_name == 'run_win_rate_analysis':
            buckets = result.get('buckets') or []
            if len(buckets) >= 3:
                x_col = result.get('x_col', '?')
                labels   = [f"B{b['bucket']}" for b in buckets]
                wr_vals  = [round(b['win_rate'] * 100, 1) for b in buckets]
                pnl_vals = [b.get('mean_pnl', 0) for b in buckets]
                cid = f"winrate_{x_col}"
                charts[cid] = {
                    'title': f"Win Rate & Mean P&L by {x_col} Bucket",
                    'config': {
                        'type': 'bar',
                        'data': {
                            'labels': labels,
                            'datasets': [
                                {
                                    'label': 'Win Rate %',
                                    'data': wr_vals,
                                    'backgroundColor': colors_pos,
                                    'borderWidth': 0,
                                    'yAxisID': 'y',
                                },
                                {
                                    'label': 'Mean P&L',
                                    'data': pnl_vals,
                                    'backgroundColor': colors_neg,
                                    'borderWidth': 0,
                                    'yAxisID': 'y1',
                                },
                            ],
                        },
                        'options': {
                            **base_opts,
                            'scales': {
                                **dark_scales,
                                'y1': {
                                    'position': 'right',
                                    'ticks': {'color': '#e84393', 'font': {'size': 9}},
                                    'grid': {'drawOnChartArea': False},
                                },
                            },
                        },
                    },
                }

        elif tool_name == 'run_decile_profile':
            bs = result.get('bucket_stats') or []
            bs = [b for b in bs if b is not None]
            if len(bs) >= 3:
                x_col = result.get('x_col', '?')
                y_col = result.get('y_col', '?')
                avgs  = [b.get('avg_y', b.get('avg_ret', 0)) for b in bs]
                cid = f"decile_{x_col}_{y_col}"
                charts[cid] = {
                    'title': f"Decile Profile: {x_col} → {y_col} (r={result.get('pearson_r', 0):.3f})",
                    'config': {
                        'type': 'bar',
                        'data': {
                            'labels': [f"D{b['bucket']}" for b in bs],
                            'datasets': [{
                                'label': y_col,
                                'data': avgs,
                                'backgroundColor': [colors_pos if v >= 0 else colors_neg for v in avgs],
                                'borderWidth': 0,
                            }],
                        },
                        'options': {**base_opts,
                                    'plugins': {**base_opts['plugins'], 'legend': {'display': False}}},
                    },
                }

        elif tool_name == 'run_regime_split':
            high = result.get('high') or {}
            low  = result.get('low') or {}
            sc   = result.get('split_col', '?')
            x_col = result.get('x_col', '?')
            if high.get('r') is not None and low.get('r') is not None:
                cid = f"regime_{sc}_{x_col}"
                charts[cid] = {
                    'title': f"Regime Split: {x_col} by {sc}",
                    'config': {
                        'type': 'bar',
                        'data': {
                            'labels': ['High regime', 'Low regime'],
                            'datasets': [
                                {
                                    'label': 'Pearson r',
                                    'data': [round(high['r'], 4), round(low['r'], 4)],
                                    'backgroundColor': [colors_pos, colors_neg],
                                    'borderWidth': 0,
                                },
                                {
                                    'label': 'Mean P&L',
                                    'data': [high.get('mean_y', 0), low.get('mean_y', 0)],
                                    'backgroundColor': [colors_pos, colors_neg],
                                    'borderWidth': 0,
                                },
                            ],
                        },
                        'options': base_opts,
                    },
                }

        elif tool_name == 'run_tail_analysis':
            diff = result.get('difference') or {}
            if diff:
                top_cols = list(diff.keys())[:10]
                vals = [diff[c] for c in top_cols]
                y_col = result.get('y_col', 'outcome')
                cid = f"tail_{y_col}_{len(charts)}"
                charts[cid] = {
                    'title': f"IV Difference: Best vs Worst {result.get('pct', 10)}% {y_col}",
                    'config': {
                        'type': 'bar',
                        'data': {
                            'labels': top_cols,
                            'datasets': [{
                                'label': 'Top% minus Bottom% IV mean',
                                'data': vals,
                                'backgroundColor': [colors_pos if v >= 0 else colors_neg for v in vals],
                                'borderWidth': 0,
                            }],
                        },
                        'options': {**base_opts,
                                    'plugins': {**base_opts['plugins'], 'legend': {'display': False}}},
                    },
                }

        elif tool_name == 'run_time_split_validation':
            periods = result.get('periods') or []
            score   = result.get('consistency_score', 0)
            if len(periods) >= 2:
                # Show top-3 correlations per period as grouped bars
                all_cols = []
                for p in periods:
                    for c in (p.get('top_correlations') or [])[:3]:
                        if c['x_col'] not in all_cols:
                            all_cols.append(c['x_col'])
                all_cols = all_cols[:8]
                if all_cols:
                    ds_colors = [colors_pos, colors_neg, '#2ecc71', '#e67e22']
                    datasets = []
                    for pi, p in enumerate(periods):
                        r_map = {c['x_col']: c['r'] for c in (p.get('top_correlations') or [])}
                        datasets.append({
                            'label': p.get('label', f'Period {pi+1}'),
                            'data': [r_map.get(c, 0) for c in all_cols],
                            'backgroundColor': ds_colors[pi % len(ds_colors)],
                            'borderWidth': 0,
                        })
                    cid = f"timesplit_{result.get('y_col', 'outcome')}"
                    charts[cid] = {
                        'title': f"Correlation Stability Across Periods (consistency={score:.0%})",
                        'config': {
                            'type': 'bar',
                            'data': {'labels': all_cols, 'datasets': datasets},
                            'options': base_opts,
                        },
                    }

        elif tool_name == 'run_feature_redundancy_check':
            pairs = result.get('pairwise_top') or []
            if len(pairs) >= 2:
                threshold = result.get('r_threshold', 0.7)
                top = pairs[:15]
                labels = [f"{p['f1'][:18]}×{p['f2'][:18]}" for p in top]
                vals   = [abs(p['r']) for p in top]
                cid = f"redundancy_{len(charts)}"
                charts[cid] = {
                    'title': f"Feature Pairwise |r| (redundancy threshold={threshold})",
                    'config': {
                        'type': 'bar',
                        'data': {
                            'labels': labels,
                            'datasets': [{
                                'label': '|r|',
                                'data': vals,
                                'backgroundColor': [colors_neg if v >= threshold else colors_pos for v in vals],
                                'borderWidth': 0,
                            }],
                        },
                        'options': {**base_opts,
                                    'plugins': {**base_opts['plugins'], 'legend': {'display': False}}},
                    },
                }

        elif tool_name == 'run_two_factor_regime':
            grid = result.get('grid') or []
            f1, f2, y_col_r = result.get('factor1', '?'), result.get('factor2', '?'), result.get('y_col', '?')
            if grid and len(grid) >= 2:
                n_bins = len(grid)
                f2_labels = [f"f2={grid[0][b2]['f2_bin']}" for b2 in range(n_bins)]
                ds_colors = [colors_pos, '#2ecc71', '#e67e22']
                datasets = []
                for b1 in range(n_bins):
                    f1_bin = grid[b1][0]['f1_bin']
                    datasets.append({
                        'label': f"f1={f1_bin}",
                        'data': [grid[b1][b2].get('mean_y', 0) for b2 in range(n_bins)],
                        'backgroundColor': ds_colors[b1 % len(ds_colors)],
                        'borderWidth': 0,
                    })
                ix = round(result.get('interaction_strength', 0), 2)
                cid = f"grid_{f1[:12]}_{f2[:12]}"
                charts[cid] = {
                    'title': f"2-Factor Grid: {f1[:20]} × {f2[:20]} → {y_col_r} (interaction={ix})",
                    'config': {
                        'type': 'bar',
                        'data': {'labels': f2_labels, 'datasets': datasets},
                        'options': base_opts,
                    },
                }

        elif tool_name == 'run_composite_regime_score':
            vq = result.get('validate_quintiles') or []
            if len(vq) >= 4:
                labels   = [f"Q{q['quintile']}" for q in vq]
                pnl_vals = [q.get('mean_y', 0) for q in vq]
                wr_vals  = [round(q.get('win_rate', 0) * 100, 1) for q in vq]
                val_r    = result.get('validate_r', 0)
                vs_single = result.get('composite_vs_single', 0)
                stable    = result.get('stable', False)
                cid = f"composite_{len(charts)}"
                charts[cid] = {
                    'title': (
                        f"Composite Regime (validate r={val_r:+.3f}, vs single {vs_single:+.3f}, "
                        f"{'✓stable' if stable else '⚠unstable'})"
                    ),
                    'config': {
                        'type': 'bar',
                        'data': {
                            'labels': labels,
                            'datasets': [
                                {
                                    'label': 'Mean P&L (validate)',
                                    'data': pnl_vals,
                                    'backgroundColor': [colors_pos if v >= 0 else colors_neg for v in pnl_vals],
                                    'borderWidth': 0,
                                    'yAxisID': 'y',
                                },
                                {
                                    'label': 'Win Rate %',
                                    'data': wr_vals,
                                    'backgroundColor': 'rgba(46,204,113,0.4)',
                                    'borderWidth': 0,
                                    'yAxisID': 'y1',
                                },
                            ],
                        },
                        'options': {
                            **base_opts,
                            'scales': {
                                **dark_scales,
                                'y1': {
                                    'position': 'right',
                                    'ticks': {'color': '#2ecc71', 'font': {'size': 9}},
                                    'grid': {'drawOnChartArea': False},
                                },
                            },
                        },
                    },
                }

        if len(charts) >= 10:
            break

    return charts


async def execute_agentic_pipeline(
    main_pool,
    run_id: str,
    question: str,
    merged_rows: list[dict],
    pnl_col: str,
    available_cols: list[str],
    model: str,
    knowledge_rules: list[str],
    log,
) -> str:
    """
    Multi-step Claude tool-use loop for P&L–IV correlation analysis.
    Claude calls analysis tools iteratively and terminates with write_report.
    Returns JSON sections string.
    """
    from research import analysis_tools as at

    if _anthropic is None:
        raise RuntimeError("anthropic SDK not installed")

    # ── 1. Quick correlation table (pre-loop context) ─────────────────────
    log("[RUNNING] Computing initial correlation table…")
    from research.pnl import compute_summary_stats
    iv_cols = [c for c in available_cols if c not in (pnl_col, 'trade_date', 'quote_time')]
    quick_corr = at.run_correlation_scan(merged_rows, iv_cols[:40], pnl_col)
    top15 = quick_corr[:15]

    pnl_stats = compute_summary_stats(merged_rows, [pnl_col] + iv_cols[:20])

    # Format quick correlation table
    corr_lines = [f"{'x_col':<45} {'r':>7} {'p':>8} {'pattern':<20} {'score':>6}",
                  "-" * 90]
    for d in top15:
        corr_lines.append(
            f"{d['x_col']:<45} {d['r']:>7.4f} {d['p_val']:>8.4f} "
            f"{d['pattern']:<20} {d['score']:>6.1f}"
        )

    pnl_s = pnl_stats.get(pnl_col) or {}
    summary_text = (
        f"DATASET SUMMARY:\n"
        f"  Total rows (matched P&L ∩ surface): {len(merged_rows)}\n"
        f"  P&L stats: mean={pnl_s.get('mean', '?')}, std={pnl_s.get('std', '?')}, "
        f"min={pnl_s.get('min', '?')}, max={pnl_s.get('max', '?')}\n"
        f"  Available IV metric columns: {len(iv_cols)}\n"
        f"  Date range: {merged_rows[0].get('trade_date', '?')} → "
        f"{merged_rows[-1].get('trade_date', '?')}\n\n"
        f"TOP 15 CORRELATIONS (full scan, sorted by |r|):\n"
        + "\n".join(corr_lines)
    )

    # ── 2. Build agentic system prompt ────────────────────────────────────
    system = _AGENTIC_SYSTEM
    if knowledge_rules:
        rules_text = "\n".join(f"- {r}" for r in knowledge_rules)
        system += f"\n\nDOMAIN KNOWLEDGE (follow these strictly):\n{rules_text}"

    first_user = (
        f"RESEARCH QUESTION: {question}\n\n"
        f"{summary_text}\n\n"
        f"IV COLUMNS AVAILABLE (all {len(iv_cols)}):\n"
        + ", ".join(iv_cols)
        + "\n\nGreek columns available: "
        + ", ".join(c for c in ('delta', 'theta', 'vega', 'gamma', 'wt_vega')
                    if merged_rows and c in merged_rows[0])
        + "\n\nBegin your analysis. Call tools to explore the data."
    )

    messages = [{"role": "user", "content": first_user}]
    tool_results_acc: list[dict] = []
    client = _anthropic.AsyncAnthropic()
    report_data = None
    max_steps = 8

    # ── 3. Agentic loop ───────────────────────────────────────────────────
    for step in range(max_steps):
        log(f"[RUNNING] Agentic analysis step {step + 1}/{max_steps}…")
        async with main_pool.acquire() as conn:
            await rdb.update_run(
                conn, run_id,
                ai_summary=f"[RUNNING] Agentic analysis step {step + 1}/{max_steps}…",
            )

        resp = await client.messages.create(
            model=model,
            max_tokens=4000,
            system=[{"type": "text", "text": system,
                     "cache_control": {"type": "ephemeral"}}],
            tools=_AGENTIC_TOOLS,
            messages=messages,
        )

        # Collect assistant message content
        messages.append({"role": "assistant", "content": resp.content})

        # Check stop reason
        if resp.stop_reason == "end_turn":
            log("  Claude finished without calling write_report — extracting text.")
            text = " ".join(b.text for b in resp.content if hasattr(b, 'text') and b.text)
            if text:
                sections = [{"type": "markdown", "content": text}]
                return json.dumps(sections)
            break

        # Collect and dispatch tool calls
        tool_use_blocks = [b for b in resp.content if b.type == "tool_use"]
        if not tool_use_blocks:
            break

        tool_result_messages = []
        for block in tool_use_blocks:
            tool_name = block.name
            inputs = block.input or {}

            if tool_name == "write_report":
                report_data = inputs
                break  # exit loop, will build report below

            # Dispatch to analysis_tools
            log(f"    → {tool_name}({list(inputs.keys())})")
            try:
                if tool_name == "run_correlation_scan":
                    result = at.run_correlation_scan(merged_rows, inputs["x_cols"], inputs["y_col"])
                elif tool_name == "run_regression":
                    result = at.run_regression(merged_rows, inputs["x_cols"], inputs["y_col"])
                elif tool_name == "run_lag_scan":
                    result = at.run_lag_scan(
                        merged_rows, inputs["x_col"], inputs["y_col"],
                        inputs.get("lags", [1, 5, 10, 30]),
                    )
                elif tool_name == "run_regime_split":
                    result = at.run_regime_split(
                        merged_rows, inputs["x_col"], inputs["y_col"],
                        inputs["split_col"], inputs.get("method", "median"),
                    )
                elif tool_name == "run_rolling_correlation":
                    result = at.run_rolling_correlation(
                        merged_rows, inputs["x_col"], inputs["y_col"],
                        inputs.get("window", 30),
                    )
                elif tool_name == "run_tail_analysis":
                    result = at.run_tail_analysis(
                        merged_rows, inputs["y_col"], inputs["x_cols"],
                        inputs.get("pct", 10),
                    )
                elif tool_name == "run_decile_profile":
                    result = at.run_decile_profile(merged_rows, inputs["x_col"], inputs["y_col"])
                elif tool_name == "run_greek_attribution":
                    result = at.run_greek_attribution(merged_rows, inputs.get("pnl_col", pnl_col))
                else:
                    result = {"error": f"Unknown tool: {tool_name}"}
            except Exception as exc:
                result = {"error": str(exc)}

            tool_results_acc.append({"tool": tool_name, "inputs": inputs, "result": result})
            result_str = json.dumps(result)
            if len(result_str) > 8000:
                result_str = result_str[:8000] + "… (truncated)"
            tool_result_messages.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": result_str,
            })

        if report_data is not None:
            break

        if tool_result_messages:
            messages.append({"role": "user", "content": tool_result_messages})
        else:
            break

    # ── 4. Force a write_report if we exited without one ──────────────────
    if report_data is None:
        log("  [RUNNING] Composing report (forced)…")
        force_msg = (
            "You have completed your analysis. Now call write_report with your findings. "
            "Be thorough — cite specific numbers from your tool results."
        )
        messages.append({"role": "user", "content": force_msg})
        resp2 = await client.messages.create(
            model=model,
            max_tokens=4000,
            system=[{"type": "text", "text": system}],
            tools=_AGENTIC_TOOLS,
            tool_choice={"type": "tool", "name": "write_report"},
            messages=messages,
        )
        for block in resp2.content:
            if block.type == "tool_use" and block.name == "write_report":
                report_data = block.input
                break

    # ── 5. Build chart configs and assemble sections ──────────────────────
    log("[RUNNING] Composing final report…")
    chart_configs = _build_agentic_chart_configs(tool_results_acc)

    if report_data is None:
        sections = [{"type": "markdown", "content": "(Report generation failed — no tool call returned)"}]
        for cid, chart in chart_configs.items():
            sections.append({"type": "chart", "chart_id": cid,
                              "title": chart["title"], "config": chart["config"]})
        return json.dumps(sections)

    exec_summary = (report_data.get("executive_summary") or "").strip()
    body = (report_data.get("body") or "").strip()
    conclusions = (report_data.get("conclusions") or "").strip()
    chart_seq = report_data.get("chart_sequence") or []

    sections = []
    if exec_summary:
        sections.append({"type": "markdown", "content": exec_summary})

    body_paras = [p.strip() for p in body.split("\n\n") if p.strip()]
    referenced = set()

    if body_paras and chart_seq:
        charts_per_gap = max(1, len(chart_seq) // max(len(body_paras), 1))
        chart_idx = 0
        for para in body_paras:
            sections.append({"type": "markdown", "content": para})
            for _ in range(charts_per_gap):
                if chart_idx < len(chart_seq):
                    cid = chart_seq[chart_idx]
                    if cid in chart_configs:
                        sections.append({
                            "type": "chart", "chart_id": cid,
                            "title": chart_configs[cid]["title"],
                            "config": chart_configs[cid]["config"],
                        })
                        referenced.add(cid)
                    chart_idx += 1
        while chart_idx < len(chart_seq):
            cid = chart_seq[chart_idx]
            if cid in chart_configs:
                sections.append({"type": "chart", "chart_id": cid,
                                  "title": chart_configs[cid]["title"],
                                  "config": chart_configs[cid]["config"]})
                referenced.add(cid)
            chart_idx += 1
    else:
        for para in body_paras:
            sections.append({"type": "markdown", "content": para})

    if conclusions:
        sections.append({"type": "markdown", "content": f"## Conclusions\n\n{conclusions}"})

    for cid, chart in chart_configs.items():
        if cid not in referenced:
            sections.append({"type": "chart", "chart_id": cid,
                              "title": chart["title"], "config": chart["config"]})

    return json.dumps(sections)


async def _execute_pnl_pipeline(
    main_pool,
    run_id: str,
    question: str,
    config: dict,
    model: str,
    knowledge_rules: list[str],
    log,
) -> str:
    """Load P&L upload, align to surface_metrics_core, run agentic pipeline."""
    from research.pnl import align_pnl_to_surface, compute_summary_stats

    upload_id = config["pnl_upload_id"]
    date_from = config.get("date_from")
    date_to   = config.get("date_to")

    # Load upload record
    log("[RUNNING] Loading P&L upload…")
    async with main_pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT data, date_from, date_to FROM research_pnl_uploads WHERE id = $1::uuid",
            upload_id,
        )
    if row is None:
        raise ValueError(f"P&L upload {upload_id} not found")

    pnl_rows = json.loads(row["data"]) if isinstance(row["data"], str) else row["data"]
    if not date_from:
        date_from = str(row["date_from"])
    if not date_to:
        date_to = str(row["date_to"])

    # Align to surface
    async with main_pool.acquire() as conn:
        await rdb.update_run(conn, run_id,
                             ai_summary=f"[RUNNING] Aligning P&L data to surface metrics…")
    merged, stats = await align_pnl_to_surface(pnl_rows, main_pool, date_from, date_to)
    log(f"[RUNNING] Aligned {stats['matched']}/{stats['pnl_rows']} P&L rows "
        f"(match_rate={stats['match_rate']:.1%}, surface_rows={stats['surface_rows']})")

    if not merged:
        raise ValueError(
            f"No rows matched after P&L–surface JOIN. "
            f"P&L rows: {stats['pnl_rows']}, surface rows for {date_from}–{date_to}: {stats['surface_rows']}. "
            "Check that dates/times in the CSV overlap with surface_metrics_core."
        )

    available_cols = list(merged[0].keys()) if merged else []

    # Separate IV metric columns from P&L / greek / timestamp columns
    _non_iv = {'trade_date', 'quote_time', 'pnl', 'delta', 'theta',
               'vega', 'gamma', 'wt_vega', 'id'}
    iv_cols   = [c for c in available_cols if c not in _non_iv]
    greek_cols = [c for c in ('delta', 'theta', 'vega', 'gamma', 'wt_vega')
                  if c in available_cols]

    # Enrich the workflow plan with actual column info and match stats
    async with main_pool.acquire() as conn:
        await conn.execute(
            """UPDATE research_runs
               SET config = jsonb_set(
                   jsonb_set(
                       jsonb_set(config,
                           '{workflow_plan,feature_columns}', $2::jsonb),
                       '{workflow_plan,task_reasoning}', $3::jsonb),
                   '{workflow_plan,hypotheses}', $4::jsonb)
               WHERE id = $1::uuid""",
            run_id,
            json.dumps(iv_cols),
            json.dumps(
                f"P&L–IV agentic analysis: {stats['matched']} matched 5-min bars "
                f"({stats['match_rate']:.0%} match rate) across "
                f"{date_from} – {date_to}. "
                f"Claude will explore {len(iv_cols)} IV metric columns vs P&L."
            ),
            json.dumps([
                f"Some IV metrics correlate meaningfully with intraday P&L",
                f"IV regime (level of skew/vol) conditions the P&L relationship",
                f"Greek attribution leaves unexplained P&L that IV metrics can explain",
            ]),
        )

    return await execute_agentic_pipeline(
        main_pool=main_pool,
        run_id=run_id,
        question=question,
        merged_rows=merged,
        pnl_col="pnl",
        available_cols=available_cols,
        model=model,
        knowledge_rules=knowledge_rules,
        log=log,
    )


def _iv_family(col: str) -> str:
    """Classify an IV column into a feature family for redundancy-aware pairing."""
    c = col.lower()
    if c.startswith("skew_"):
        tenor = c.split("_")[1] if "_" in c else ""
        return f"skew_{tenor}"
    if c.startswith("convex_"):
        tenor = c.split("_")[1] if "_" in c else ""
        return f"convex_{tenor}"
    if c.startswith("term_slope_"):
        return "term_slope"
    if c.startswith("term_ratio_"):
        return "term_ratio"
    if c.startswith("vix_") or c.startswith("iv_"):
        return "iv_level"
    if c.startswith("forward_"):
        return "forward"
    if c == "spot":
        return "spot"
    if "opex" in c:
        return "opex"
    return "other"


async def execute_backtest_pipeline(
    main_pool,
    run_id: str,
    question: str,
    enriched_trades: list[dict],
    trade_summary: dict,
    model: str,
    knowledge_rules: list[str],
    log,
) -> str:
    """
    Deterministic backtest analysis pipeline.
    Builds a complete evidence packet, then hands it to 1 LLM call for narrative.
    Returns JSON sections string.
    """
    from research import analysis_tools as at
    from research.backtest import TRADE_FIELDS
    from itertools import combinations

    trades = enriched_trades
    n_trades = len(trades)
    all_cols = list(trades[0].keys()) if trades else []
    iv_cols = [c for c in all_cols if c not in TRADE_FIELDS]

    # Cast is_win to float
    for t in trades:
        if 'is_win' in t:
            t['is_win'] = float(1.0 if t['is_win'] else 0.0)

    evidence = {"trade_summary": trade_summary, "warnings": []}
    tool_results_acc = []  # for chart builder

    # ── Step 1: Single-factor discovery ──────────────────────────────────
    log("[1/9] Correlation scan…")
    async with main_pool.acquire() as conn:
        await rdb.update_run(conn, run_id, ai_summary="[RUNNING] Step 1/9: Correlation scan…")

    corr_pnl = at.run_correlation_scan(trades, iv_cols, 'pnl')[:15]
    corr_win = at.run_correlation_scan(trades, iv_cols, 'is_win')[:15]
    evidence["corr_pnl"] = corr_pnl
    evidence["corr_win"] = corr_win
    tool_results_acc.append({"tool": "run_correlation_scan", "inputs": {"y_col": "pnl"}, "result": corr_pnl})
    tool_results_acc.append({"tool": "run_correlation_scan", "inputs": {"y_col": "is_win"}, "result": corr_win})

    # Merge top candidates
    seen = set()
    candidates = []
    for r in corr_pnl + corr_win:
        if r["x_col"] not in seen:
            candidates.append(r["x_col"])
            seen.add(r["x_col"])
    candidates = candidates[:20]
    log(f"  {len(candidates)} candidate features identified")

    # ── Step 2: Redundancy filtering ─────────────────────────────────────
    log("[2/9] Redundancy check…")
    async with main_pool.acquire() as conn:
        await rdb.update_run(conn, run_id, ai_summary="[RUNNING] Step 2/9: Redundancy filtering…")

    redundancy = at.run_feature_redundancy_check(trades, candidates, r_threshold=0.7)
    non_redundant = redundancy.get("non_redundant", candidates[:8])
    evidence["redundancy"] = redundancy
    evidence["non_redundant"] = non_redundant
    log(f"  {len(non_redundant)} non-redundant features: {', '.join(non_redundant[:6])}")

    # ── Step 3: Profiles for non-redundant features ──────────────────────
    log("[3/9] Decile profiles + win rate analysis…")
    async with main_pool.acquire() as conn:
        await rdb.update_run(conn, run_id, ai_summary="[RUNNING] Step 3/9: Profiles…")

    profiles = {}
    win_rates = {}
    for feat in non_redundant[:8]:
        try:
            prof = at.run_decile_profile(trades, feat, 'pnl')
            profiles[feat] = prof
            tool_results_acc.append({"tool": "run_decile_profile", "inputs": {"x_col": feat, "y_col": "pnl"}, "result": prof})
        except Exception as exc:
            log(f"    Profile error {feat}: {exc}")
        try:
            wr = at.run_win_rate_analysis(trades, feat, n_buckets=5)
            win_rates[feat] = wr
            tool_results_acc.append({"tool": "run_win_rate_analysis", "inputs": {"x_col": feat}, "result": wr})
        except Exception as exc:
            log(f"    Win rate error {feat}: {exc}")
    evidence["profiles"] = profiles
    evidence["win_rates"] = win_rates

    # ── Step 4: Time-split validation ────────────────────────────────────
    log("[4/9] Time-split validation…")
    async with main_pool.acquire() as conn:
        await rdb.update_run(conn, run_id, ai_summary="[RUNNING] Step 4/9: Time-split validation…")

    n_splits = 3 if n_trades >= 600 else 2
    time_splits = at.run_time_split_validation(trades, 'pnl', n_splits=n_splits)
    evidence["time_splits"] = time_splits
    tool_results_acc.append({"tool": "run_time_split_validation", "inputs": {"y_col": "pnl"}, "result": time_splits})

    # Classify features as stable/unstable
    stable_features = []
    unstable_features = []
    period_corrs = time_splits.get("feature_stability", {})
    for feat in non_redundant:
        feat_periods = period_corrs.get(feat, {}).get("periods", [])
        if not feat_periods:
            unstable_features.append(feat)
            continue
        # Stable if correlation sign is consistent across majority of periods
        signs = [1 if p.get("r", 0) > 0 else -1 for p in feat_periods if p.get("r") is not None]
        if signs and abs(sum(signs)) >= len(signs) * 0.6:
            stable_features.append(feat)
        else:
            unstable_features.append(feat)
    evidence["stable_features"] = stable_features
    evidence["unstable_features"] = unstable_features
    log(f"  Stable: {stable_features[:5]}, Unstable: {unstable_features[:5]}")

    # ── Step 5: Tail analysis ────────────────────────────────────────────
    log("[5/9] Tail analysis…")
    async with main_pool.acquire() as conn:
        await rdb.update_run(conn, run_id, ai_summary="[RUNNING] Step 5/9: Tail analysis…")

    tail = at.run_tail_analysis(trades, 'pnl', non_redundant[:10], pct=15)
    evidence["tail_analysis"] = tail
    tool_results_acc.append({"tool": "run_tail_analysis", "inputs": {"y_col": "pnl", "x_cols": non_redundant[:10]}, "result": tail})

    # ── Step 6: Rolling correlation for top 3 stable features ────────────
    log("[6/9] Rolling correlation…")
    async with main_pool.acquire() as conn:
        await rdb.update_run(conn, run_id, ai_summary="[RUNNING] Step 6/9: Rolling correlation…")

    rolling_feats = (stable_features or non_redundant)[:3]
    window = max(30, n_trades // 10)
    rolling_corrs = {}
    for feat in rolling_feats:
        try:
            rc = at.run_rolling_correlation(trades, feat, 'pnl', window=window)
            rolling_corrs[feat] = rc
            tool_results_acc.append({"tool": "run_rolling_correlation", "inputs": {"x_col": feat, "y_col": "pnl"}, "result": rc})
        except Exception as exc:
            log(f"    Rolling corr error {feat}: {exc}")
    evidence["rolling_corrs"] = rolling_corrs

    # ── Step 7: Two-factor regime grids (different families) ─────────────
    log("[7/9] Two-factor regime grids…")
    async with main_pool.acquire() as conn:
        await rdb.update_run(conn, run_id, ai_summary="[RUNNING] Step 7/9: 2-factor grids…")

    # Pick pairs from different families
    feat_families = {}
    for feat in non_redundant:
        fam = _iv_family(feat)
        feat_families.setdefault(fam, []).append(feat)

    # Best feature per family
    best_per_family = {}
    for fam, feats in feat_families.items():
        best = max(feats, key=lambda f: abs(next((r["r"] for r in corr_pnl if r["x_col"] == f), 0)))
        best_per_family[fam] = best

    family_reps = list(best_per_family.values())
    two_factor_grids = []
    grid_count = 0
    for f1, f2 in combinations(family_reps, 2):
        if grid_count >= 3:
            break
        if _iv_family(f1) == _iv_family(f2):
            continue
        try:
            grid_pnl = at.run_two_factor_regime(trades, f1, f2, 'pnl', n_bins=3)
            grid_win = at.run_two_factor_regime(trades, f1, f2, 'is_win', n_bins=3)
            two_factor_grids.append({"f1": f1, "f2": f2, "pnl": grid_pnl, "is_win": grid_win})
            tool_results_acc.append({"tool": "run_two_factor_regime", "inputs": {"factor1": f1, "factor2": f2, "y_col": "pnl"}, "result": grid_pnl})
            tool_results_acc.append({"tool": "run_two_factor_regime", "inputs": {"factor1": f1, "factor2": f2, "y_col": "is_win"}, "result": grid_win})
            grid_count += 1
        except Exception as exc:
            log(f"    2-factor grid error {f1}×{f2}: {exc}")
    evidence["two_factor_grids"] = two_factor_grids

    # ── Step 8: Regime split by VIX ──────────────────────────────────────
    log("[8/9] VIX regime split…")
    async with main_pool.acquire() as conn:
        await rdb.update_run(conn, run_id, ai_summary="[RUNNING] Step 8/9: VIX regime split…")

    best_feat = non_redundant[0] if non_redundant else None
    regime_split = None
    for vix_col in ['vix_30d', 'vix_90d', 'vix_180d']:
        if vix_col in iv_cols and best_feat:
            try:
                regime_split = at.run_regime_split(trades, best_feat, 'pnl', vix_col, method='median')
                tool_results_acc.append({"tool": "run_regime_split", "inputs": {"x_col": best_feat, "y_col": "pnl", "split_col": vix_col}, "result": regime_split})
                break
            except Exception:
                pass
    evidence["regime_split"] = regime_split

    # ── Step 9: Composite + model comparison ─────────────────────────────
    log("[9/9] Composite score + model comparison…")
    async with main_pool.acquire() as conn:
        await rdb.update_run(conn, run_id, ai_summary="[RUNNING] Step 9/9: Model comparison…")

    # Best single feature
    best_single = None
    best_single_r = 0
    for feat in non_redundant:
        r_val = abs(next((r["r"] for r in corr_pnl if r["x_col"] == feat), 0))
        if r_val > best_single_r:
            best_single_r = r_val
            best_single = feat
    best_single_corr = next((r for r in corr_pnl if r["x_col"] == best_single), {}) if best_single else {}

    # Best single win-rate spread
    best_single_wr_spread = 0
    if best_single and best_single in win_rates:
        wr_data = win_rates[best_single]
        buckets = wr_data.get("buckets", [])
        if len(buckets) >= 2:
            wrs = [b.get("win_rate", 0.5) for b in buckets]
            best_single_wr_spread = max(wrs) - min(wrs)

    # Best 2-factor
    best_2f = None
    best_2f_wr_spread = 0
    for grid in two_factor_grids:
        cells = grid["is_win"].get("cells", [])
        if cells:
            wrs = [c.get("mean_y", 0.5) for c in cells if c.get("n", 0) >= 10]
            if wrs:
                spread = max(wrs) - min(wrs)
                if spread > best_2f_wr_spread:
                    best_2f_wr_spread = spread
                    best_2f = grid

    improvement_2f = best_2f_wr_spread - best_single_wr_spread
    meaningful_2f = improvement_2f > 0.05  # >5pp improvement

    # Composite (only if 2+ stable features)
    composite = None
    if len(stable_features) >= 2:
        components = []
        for feat in stable_features[:3]:
            direction = next((r["r"] for r in corr_pnl if r["x_col"] == feat), 0)
            components.append({"feature": feat, "direction": "positive" if direction > 0 else "negative"})

        # Only add interactions if 2-factor grid showed clear interaction
        interactions = None
        if best_2f and best_2f.get("pnl", {}).get("interaction_strength", 0) > 0.1:
            interactions = [{"type": "amplify", "feature1": best_2f["f1"], "feature2": best_2f["f2"]}]

        try:
            composite = at.run_composite_regime_score(
                trades, components, interactions, 'pnl', train_frac=0.6)
            tool_results_acc.append({"tool": "run_composite_regime_score", "inputs": {"components": components}, "result": composite})
        except Exception as exc:
            log(f"    Composite error: {exc}")
            composite = None

    # Model comparison verdict
    composite_recommended = False
    if composite:
        validate_r = composite.get("validate_r", 0)
        best_single_validate_r = composite.get("best_single_validate_r", 0)
        composite_recommended = (
            validate_r > best_single_validate_r and
            composite.get("stable", False) and
            composite.get("composite_vs_single", 0) > 0
        )

    if composite_recommended:
        verdict = "composite"
        verdict_reason = (f"Composite (validate_r={composite.get('validate_r', 0):.3f}) improves over "
                         f"best single ({best_single}, validate_r={composite.get('best_single_validate_r', 0):.3f}) "
                         f"and is stable across time splits.")
    elif meaningful_2f:
        verdict = "two_factor"
        verdict_reason = (f"2-factor ({best_2f['f1']} × {best_2f['f2']}) adds {improvement_2f:.1%} win-rate spread "
                         f"over best single feature ({best_single}).")
    else:
        verdict = "single_feature"
        verdict_reason = (f"Best usable signal is {best_single} alone "
                         f"(r={best_single_corr.get('r', 0):+.3f}, WR spread={best_single_wr_spread:.1%}). "
                         f"2-factor adds only {improvement_2f:.1%} and composite does not improve OOS.")

    evidence["model_comparison"] = {
        "best_single": {"feature": best_single, "r": best_single_corr.get("r"), "wr_spread": best_single_wr_spread},
        "best_2factor": {"f1": best_2f["f1"], "f2": best_2f["f2"], "wr_spread": best_2f_wr_spread,
                         "improvement": improvement_2f, "meaningful": meaningful_2f} if best_2f else None,
        "composite": {"validate_r": composite.get("validate_r"), "recommended": composite_recommended} if composite else None,
        "verdict": verdict,
        "verdict_reason": verdict_reason,
    }
    evidence["composite"] = composite
    log(f"  Model verdict: {verdict} — {verdict_reason[:100]}")

    # ── Build charts ─────────────────────────────────────────────────────
    chart_configs = _build_backtest_chart_configs(tool_results_acc)

    # ── Compose report (1 LLM call) ──────────────────────────────────────
    log("[RUNNING] Composing backtest report…")
    async with main_pool.acquire() as conn:
        await rdb.update_run(conn, run_id, ai_summary="[RUNNING] Composing report…")

    evidence_text = _format_evidence_packet(evidence, question)
    chart_listing = "\n".join(f"  {cid}: {chart['title']}" for cid, chart in chart_configs.items())

    report_system = (
        "You are a quantitative analyst writing a research report on backtest trade outcomes. "
        "Professional quant voice. No fluff. Cite specific numbers."
    )
    if knowledge_rules:
        rules_text = "\n".join(f"  - {r}" for r in knowledge_rules)
        report_system += f"\n\nKNOWLEDGE BASE RULES:\n{rules_text}"

    report_prompt = (
        f"{evidence_text}\n\n"
        f"AVAILABLE CHARTS:\n{chart_listing}\n\n"
        f"Write a report using the produce_report tool.\n"
        f"- executive_summary: 2-3 paragraphs answering the research question. Cite r values, win rates, P&L.\n"
        f"- body: 400-700 words. Cover: what mattered, why it likely mattered (economic interpretation), "
        f"how strong the evidence was, whether it held over time (cite rolling correlation and time-split), "
        f"whether 2-factor/composite added value, what didn't work.\n"
        f"- conclusions: Explicitly state 'Best usable signal is: [single / 2-factor / composite]' and justify. "
        f"Recommend filter, sizing input, or exploratory-only.\n"
        f"- chart_sequence: ordered chart IDs to display.\n\n"
        f"Do NOT recommend a more complex model unless it clearly improves performance AND stability OOS."
    )

    client = _anthropic.AsyncAnthropic()
    msg = await client.messages.create(
        model=model,
        max_tokens=4000,
        system=[{"type": "text", "text": report_system,
                 "cache_control": {"type": "ephemeral"}}],
        tools=[_BACKTEST_AGENTIC_TOOLS[-1]],  # write_report tool only
        tool_choice={"type": "tool", "name": "write_report"},
        messages=[{"role": "user", "content": report_prompt}],
    )

    report_data = None
    for block in msg.content:
        if block.type == "tool_use" and block.name == "write_report":
            report_data = block.input
            break

    # ── Skeptic review (1 LLM call) ──────────────────────────────────────
    log("[RUNNING] Skeptic review…")
    async with main_pool.acquire() as conn:
        await rdb.update_run(conn, run_id, ai_summary="[RUNNING] Skeptic review…")

    report_prose = ""
    if report_data:
        report_prose = "\n\n".join(filter(None, [
            report_data.get("executive_summary"),
            report_data.get("body"),
            report_data.get("conclusions"),
        ]))

    skeptic_text = await _skeptic_review(
        report_prose,
        [],  # no ranked_scans for backtest
        [],  # no equity_results
        model,
        knowledge_rules=knowledge_rules,
    ) if report_prose else ""

    # ── Assemble sections ────────────────────────────────────────────────
    sections: list[dict] = []

    if report_data:
        exec_summary = (report_data.get("executive_summary") or "").strip()
        body = (report_data.get("body") or "").strip()
        conclusions = (report_data.get("conclusions") or "").strip()
        chart_seq = report_data.get("chart_sequence") or []

        if exec_summary:
            sections.append({"type": "markdown", "content": exec_summary})

        body_paras = [p.strip() for p in body.split("\n\n") if p.strip()]
        referenced: set[str] = set()

        if body_paras and chart_seq:
            charts_per_gap = max(1, len(chart_seq) // max(len(body_paras), 1))
            chart_idx = 0
            for para in body_paras:
                sections.append({"type": "markdown", "content": para})
                for _ in range(charts_per_gap):
                    if chart_idx < len(chart_seq):
                        cid = chart_seq[chart_idx]
                        if cid in chart_configs:
                            sections.append({"type": "chart", "chart_id": cid,
                                              "title": chart_configs[cid]["title"],
                                              "config": chart_configs[cid]["config"]})
                            referenced.add(cid)
                        chart_idx += 1
            while chart_idx < len(chart_seq):
                cid = chart_seq[chart_idx]
                if cid in chart_configs and cid not in referenced:
                    sections.append({"type": "chart", "chart_id": cid,
                                      "title": chart_configs[cid]["title"],
                                      "config": chart_configs[cid]["config"]})
                    referenced.add(cid)
                chart_idx += 1
        else:
            for para in body_paras:
                sections.append({"type": "markdown", "content": para})

        if conclusions:
            sections.append({"type": "markdown", "content": f"## Conclusions\n\n{conclusions}"})

        for cid, chart in chart_configs.items():
            if cid not in referenced:
                sections.append({"type": "chart", "chart_id": cid,
                                  "title": chart["title"], "config": chart["config"]})
    else:
        sections.append({"type": "markdown", "content": "(Report generation failed)"})
        for cid, chart in chart_configs.items():
            sections.append({"type": "chart", "chart_id": cid,
                              "title": chart["title"], "config": chart["config"]})

    if skeptic_text:
        sections.append({"type": "markdown", "content": f"---\n\n## Caveats & Challenges\n\n{skeptic_text}"})

    return json.dumps(sections)


def _format_evidence_packet(evidence: dict, question: str) -> str:
    """Format the evidence packet as structured text for the LLM."""
    ts = evidence.get("trade_summary", {})

    # Detect portfolio-context features — they need a glossary so the writer
    # interprets them as "recent open-position P&L behavior at entry," not IV.
    all_cited_cols = set()
    for r in (evidence.get("corr_pnl") or []) + (evidence.get("corr_win") or []):
        all_cited_cols.add(r.get("x_col", ""))
    for c in evidence.get("non_redundant") or []:
        all_cited_cols.add(c)
    has_portfolio_cols = any(c.startswith("portfolio_pnl_chg_") for c in all_cited_cols)

    lines = [
        f"RESEARCH QUESTION: {question}",
        "",
    ]
    if has_portfolio_cols:
        lines += [
            "COLUMN GLOSSARY (portfolio-context features):",
            "  portfolio_pnl_chg_Nd_at_entry — cumulative average per-trade daily P&L",
            "    change over the last N trading days ending at the entry date's 3:30pm",
            "    slice. 'Per-trade average' (not portfolio total) so concurrent-position",
            "    count doesn't drive magnitude. The new trade itself is excluded.",
            "  portfolio_pnl_chg_Nd_at_entry_pctl — 0–100 expanding-window historical",
            "    percentile rank of the above value vs the prior history of the same",
            "    metric. p≥90 ≈ 'twice/month' rare; p≥98 ≈ 'few times/year' rare.",
            "  Note: 1d feature uses 3:30pm-of-entry-day slice → can include partial",
            "    same-day P&L for trades entered before 3:30pm. 5d+ windows are less",
            "    sensitive to that.",
            "",
        ]
    lines += [
        "TRADE SUMMARY:",
        f"  Total trades: {ts.get('n', 0)}",
        f"  IV-matched: {ts.get('matched_count', 0)} ({ts.get('match_rate', 0):.0%})",
        f"  Date range: {ts.get('date_from')} → {ts.get('date_to')}",
        f"  Strategies: {', '.join(ts.get('strategies') or ['unknown'])}",
        f"  Win rate: {ts.get('win_rate', 0):.1%}",
        f"  Mean P&L: ${ts.get('mean_pnl', 0):.0f}",
        f"  Std P&L: ${ts.get('std_pnl', 0):.0f}",
        "",
        "TOP CORRELATIONS WITH pnl:",
    ]
    for r in evidence.get("corr_pnl", [])[:10]:
        lines.append(f"  {r['x_col']:<40} r={r['r']:+.4f}  p={r.get('p_val', 0):.4f}")

    lines.append("\nTOP CORRELATIONS WITH is_win:")
    for r in evidence.get("corr_win", [])[:10]:
        lines.append(f"  {r['x_col']:<40} r={r['r']:+.4f}  p={r.get('p_val', 0):.4f}")

    lines.append(f"\nNON-REDUNDANT FEATURES ({len(evidence.get('non_redundant', []))}):")
    lines.append(f"  {', '.join(evidence.get('non_redundant', []))}")

    dropped = evidence.get("redundancy", {}).get("dropped_pairs", [])
    if dropped:
        lines.append(f"\n  Dropped for redundancy ({len(dropped)} pairs with |r|>0.7)")

    # Profiles
    lines.append("\nDECILE PROFILES (top non-redundant → pnl):")
    for feat, prof in list(evidence.get("profiles", {}).items())[:5]:
        bs = prof.get("bucket_stats") or []
        bs = [b for b in bs if b is not None]
        if bs:
            d1 = bs[0] if bs else {}
            d10 = bs[-1] if bs else {}
            lines.append(f"  {feat}: r={prof.get('pearson_r', 0):+.3f}, "
                        f"D1 avg=${d1.get('avg_y', d1.get('avg_ret', 0)):,.0f} WR={d1.get('win_rate', 0):.0%}, "
                        f"D10 avg=${d10.get('avg_y', d10.get('avg_ret', 0)):,.0f} WR={d10.get('win_rate', 0):.0%}")

    # Win rates
    lines.append("\nWIN RATE ANALYSIS (5-bucket):")
    for feat, wr in list(evidence.get("win_rates", {}).items())[:5]:
        buckets = wr.get("buckets", [])
        if buckets:
            wrs = [b.get("win_rate", 0.5) for b in buckets]
            lines.append(f"  {feat}: WR range {min(wrs):.0%}–{max(wrs):.0%}, spread={max(wrs)-min(wrs):.1%}")

    # Time stability
    lines.append(f"\nSTABLE FEATURES: {', '.join(evidence.get('stable_features', []))}")
    lines.append(f"UNSTABLE FEATURES: {', '.join(evidence.get('unstable_features', []))}")

    # Tail analysis
    tail = evidence.get("tail_analysis", {})
    if tail.get("differences"):
        lines.append("\nTAIL ANALYSIS (top 15% vs bottom 15% trades, IV differences):")
        for d in tail["differences"][:8]:
            lines.append(f"  {d['feature']:<40} top_mean={d['top_mean']:+.4f}  bot_mean={d['bot_mean']:+.4f}  "
                        f"diff={d['diff']:+.4f}")

    # Rolling correlation
    lines.append("\nROLLING CORRELATION SUMMARY:")
    for feat, rc in evidence.get("rolling_corrs", {}).items():
        if rc:
            r_vals = [p.get("r", 0) for p in rc if p.get("r") is not None]
            if r_vals:
                lines.append(f"  {feat}: range [{min(r_vals):+.3f}, {max(r_vals):+.3f}], "
                            f"latest={r_vals[-1]:+.3f}")

    # 2-factor grids
    lines.append("\nTWO-FACTOR REGIME GRIDS:")
    for grid in evidence.get("two_factor_grids", []):
        pnl_grid = grid.get("pnl", {})
        win_grid = grid.get("is_win", {})
        cells = pnl_grid.get("cells", [])
        if cells:
            pnls = [c.get("mean_y", 0) for c in cells if c.get("n", 0) >= 10]
            best_cell = max(cells, key=lambda c: c.get("mean_y", 0)) if cells else {}
            worst_cell = min(cells, key=lambda c: c.get("mean_y", 0)) if cells else {}
            lines.append(f"  {grid['f1']} × {grid['f2']}: "
                        f"interaction={pnl_grid.get('interaction_strength', 0):.3f}, "
                        f"best zone={best_cell.get('label', '?')} (P&L=${best_cell.get('mean_y', 0):,.0f}), "
                        f"worst zone={worst_cell.get('label', '?')} (P&L=${worst_cell.get('mean_y', 0):,.0f})")

    # Regime split
    rs = evidence.get("regime_split")
    if rs:
        lines.append(f"\nVIX REGIME SPLIT ({rs.get('split_col', '?')} median):")
        for regime, data in rs.get("regimes", {}).items():
            lines.append(f"  {regime}: r={data.get('r', 0):+.3f}, mean_pnl=${data.get('mean_y', 0):,.0f}, n={data.get('n', 0)}")

    # Model comparison
    mc = evidence.get("model_comparison", {})
    lines.append(f"\nMODEL COMPARISON VERDICT: {mc.get('verdict', '?')}")
    lines.append(f"  Reason: {mc.get('verdict_reason', '')}")
    bs = mc.get("best_single", {})
    if bs:
        lines.append(f"  Best single: {bs.get('feature')} r={bs.get('r', 0):+.3f} WR_spread={bs.get('wr_spread', 0):.1%}")
    b2f = mc.get("best_2factor")
    if b2f:
        lines.append(f"  Best 2-factor: {b2f.get('f1')}×{b2f.get('f2')} WR_spread={b2f.get('wr_spread', 0):.1%} "
                    f"improvement={b2f.get('improvement', 0):+.1%} meaningful={b2f.get('meaningful')}")
    comp = mc.get("composite")
    if comp:
        lines.append(f"  Composite: validate_r={comp.get('validate_r', 0):.3f} recommended={comp.get('recommended')}")

    # Warnings
    for w in evidence.get("warnings", []):
        lines.append(f"\nWARNING: {w}")

    return "\n".join(lines)


async def _execute_backtest_run(
    main_pool,
    run_id: str,
    question: str,
    config: dict,
    model: str,
    knowledge_rules: list[str],
    log,
) -> str:
    """Load backtest upload (already enriched with IV metrics), run backtest pipeline."""
    from research.backtest import (
        compute_trade_summary,
        compute_portfolio_context_features,
        TRADE_FIELDS,
    )

    upload_id = config["backtest_upload_id"]

    log("[RUNNING] Loading backtest upload…")
    async with main_pool.acquire() as conn:
        await rdb.update_run(conn, run_id,
                             ai_summary="[RUNNING] Loading backtest trade data…")
        row = await conn.fetchrow(
            "SELECT data, trade_count, matched_count, daily_paths, has_daily_paths "
            "FROM research_backtest_uploads WHERE id = $1::uuid",
            upload_id,
        )
    if row is None:
        raise ValueError(f"Backtest upload {upload_id} not found")

    enriched_trades = json.loads(row["data"]) if isinstance(row["data"], str) else list(row["data"])
    log(f"[RUNNING] Loaded {len(enriched_trades)} trades "
        f"({row['matched_count'] or 0} matched to IV surface)")

    # Portfolio-context features (1d/3d/5d/10d/15d avg per-trade P&L change at entry,
    # plus expanding-window historical percentile rank). Requires Mesosim daily paths.
    n_portfolio_features = 0
    if row["has_daily_paths"]:
        daily_raw = row["daily_paths"]
        daily_paths = (json.loads(daily_raw) if isinstance(daily_raw, str)
                       else list(daily_raw or []))
        log(f"[RUNNING] Computing portfolio-context features from {len(daily_paths)} daily rows…")
        portfolio_features = await asyncio.to_thread(
            compute_portfolio_context_features, daily_paths, enriched_trades
        )
        for trade in enriched_trades:
            pid = trade.get("position_id")
            if pid and pid in portfolio_features:
                trade.update(portfolio_features[pid])
        if portfolio_features:
            sample = next(iter(portfolio_features.values()))
            n_portfolio_features = len(sample)
            log(f"[RUNNING] Attached {n_portfolio_features} portfolio-context columns "
                f"to {len(portfolio_features)} trades")
    else:
        log("[RUNNING] No daily paths — skipping portfolio-context features (CSV upload)")

    # Separate IV columns
    iv_cols = [c for c in (enriched_trades[0].keys() if enriched_trades else [])
               if c not in TRADE_FIELDS]

    # Compute summary
    trade_summary = compute_trade_summary(enriched_trades)

    # Enrich workflow plan in DB
    async with main_pool.acquire() as conn:
        await conn.execute(
            """UPDATE research_runs
               SET config = jsonb_set(
                   jsonb_set(
                       jsonb_set(config,
                           '{workflow_plan,feature_columns}', $2::jsonb),
                       '{workflow_plan,task_reasoning}', $3::jsonb),
                   '{workflow_plan,hypotheses}', $4::jsonb)
               WHERE id = $1::uuid""",
            run_id,
            json.dumps(iv_cols),
            json.dumps(
                f"Backtest IV analysis: {trade_summary['n']} trades, "
                f"{trade_summary['matched_count']} matched to entry-morning IV "
                f"({trade_summary['match_rate']:.0%} match rate). "
                f"Claude will explore {len(iv_cols)} feature columns vs trade outcomes"
                + (f" (incl. {n_portfolio_features} portfolio-context columns: "
                   f"avg per-trade P&L change over 1/3/5/10/15 trading days "
                   f"ending at entry-day 3:30pm, with historical percentile rank)."
                   if n_portfolio_features else ".")
            ),
            json.dumps([
                "Some entry-time IV metrics predict final trade P&L",
                "IV regime at entry conditions win/loss probability",
                "The IV predictive signal is stable across different time periods",
            ]),
        )

    return await execute_backtest_pipeline(
        main_pool=main_pool,
        run_id=run_id,
        question=question,
        enriched_trades=enriched_trades,
        trade_summary=trade_summary,
        model=model,
        knowledge_rules=knowledge_rules,
        log=log,
    )


async def execute_intratrade_pipeline(
    main_pool,
    run_id: str,
    question: str,
    daily_rows: list[dict],
    path_summary: dict,
    model: str,
    knowledge_rules: list[str],
    log,
) -> str:
    """
    True agentic Claude loop for intratrade path analysis.
    Claude receives pre-computed context and iteratively calls generic tools.
    Returns JSON sections string.
    """
    from research import analysis_tools as at

    if _anthropic is None:
        raise RuntimeError("anthropic SDK not installed")

    # ── 1. Pre-compute initial context ────────────────────────────────────
    log("[RUNNING] Computing intratrade initial context…")

    iv_cols = path_summary.get('iv_level_fields', [])
    since_entry_cols = path_summary.get('since_entry_fields', [])
    all_feature_cols = iv_cols + since_entry_cols

    # Top correlations against daily_pnl_change and drawdown_from_peak
    quick_corr_change = at.run_correlation_scan(
        daily_rows, all_feature_cols[:40], 'daily_pnl_change'
    ) if all_feature_cols else []
    quick_corr_dd = at.run_correlation_scan(
        daily_rows, all_feature_cols[:40], 'drawdown_from_peak'
    ) if all_feature_cols else []

    # DIT profile of daily_pnl_change
    dit_profile = at.run_days_in_trade_profile(daily_rows, 'daily_pnl_change', n_bins=8)

    def _fmt_corr(corr_list, n=12):
        if not corr_list:
            return '  (none)'
        lines = [f"  {'x_col':<45} {'r':>7}"]
        for d in corr_list[:n]:
            lines.append(f"  {d['x_col']:<45} {d['r']:>7.4f}")
        return '\n'.join(lines)

    def _fmt_dit(profile):
        if 'error' in (profile or {}):
            return f"  ({profile['error']})"
        bins = profile.get('bins', [])
        lines = []
        for b in bins:
            label = b.get('phase') or f"DIT {b.get('dit_min','?'):.0f}–{b.get('dit_max','?'):.0f}"
            mean  = b.get('mean')
            n     = b.get('n', 0)
            lines.append(f"  {label:<22} mean={mean:>8.1f}  n={n}" if mean is not None
                         else f"  {label:<22} (no data)")
        return '\n'.join(lines) or '  (no data)'

    ps = path_summary
    summary_text = (
        f"DATASET SUMMARY:\n"
        f"  Total daily rows: {ps.get('total_rows', len(daily_rows))}\n"
        f"  Trades covered: {ps.get('total_trades', '?')}\n"
        f"  DIT range: min={ps.get('dit_min', '?')}, median={ps.get('dit_p50', '?')}, "
        f"max={ps.get('dit_max', '?')}\n"
        f"  Phase breakdown: early={ps.get('n_early', '?')} rows, "
        f"middle={ps.get('n_middle', '?')}, late={ps.get('n_late', '?')}\n"
        f"  IV level columns: {len(iv_cols)}\n"
        f"  Since-entry delta columns: {len(since_entry_cols)}\n\n"
        f"TOP 12 CORRELATIONS: IV vs daily_pnl_change:\n{_fmt_corr(quick_corr_change)}\n\n"
        f"TOP 12 CORRELATIONS: IV vs drawdown_from_peak:\n{_fmt_corr(quick_corr_dd)}\n\n"
        f"DIT PROFILE of daily_pnl_change (trend: {dit_profile.get('trend', '?')}):\n"
        + _fmt_dit(dit_profile)
    )

    # ── 2. Build system prompt ────────────────────────────────────────────
    system = _INTRATRADE_AGENTIC_SYSTEM
    if knowledge_rules:
        rules_text = '\n'.join(f'- {r}' for r in knowledge_rules)
        system += f'\n\nDOMAIN KNOWLEDGE (follow these strictly):\n{rules_text}'

    greek_cols = [c for c in ('pos_pnl', 'pos_delta', 'pos_gamma', 'pos_theta', 'pos_vega', 'pos_wvega')
                  if daily_rows and c in daily_rows[0]]

    first_user = (
        f"RESEARCH QUESTION: {question}\n\n"
        f"{summary_text}\n\n"
        f"GREEK/PATH COLUMNS: {', '.join(greek_cols)}\n"
        f"DERIVED PATH COLUMNS: daily_pnl_change, drawdown_from_peak, max_pnl_to_date, "
        f"pct_of_peak_retained, trade_phase, dit\n"
        f"IV LEVEL COLUMNS ({len(iv_cols)}): {', '.join(iv_cols[:30])}"
        + (f"… (+{len(iv_cols) - 30} more)" if len(iv_cols) > 30 else '') + '\n'
        f"SINCE-ENTRY DELTA COLUMNS ({len(since_entry_cols)}): {', '.join(since_entry_cols[:30])}"
        + (f"… (+{len(since_entry_cols) - 30} more)" if len(since_entry_cols) > 30 else '') + '\n\n'
        f"Begin your analysis. Call tools to explore the data."
    )

    messages = [{"role": "user", "content": first_user}]
    tool_results_acc: list[dict] = []
    client = _anthropic.AsyncAnthropic()
    report_data = None
    max_steps = 10

    # ── 3. Agentic loop ───────────────────────────────────────────────────
    for step in range(max_steps):
        log(f"[RUNNING] Intratrade analysis step {step + 1}/{max_steps}…")
        async with main_pool.acquire() as conn:
            await rdb.update_run(
                conn, run_id,
                ai_summary=f"[RUNNING] Intratrade path analysis step {step + 1}/{max_steps}…",
            )

        resp = await client.messages.create(
            model=model,
            max_tokens=4000,
            system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            tools=_INTRATRADE_AGENTIC_TOOLS,
            messages=messages,
        )

        messages.append({"role": "assistant", "content": resp.content})

        if resp.stop_reason == "end_turn":
            log("  Claude finished without write_report — extracting text.")
            text = " ".join(b.text for b in resp.content if hasattr(b, 'text') and b.text)
            if text:
                return json.dumps([{"type": "markdown", "content": text}])
            break

        tool_use_blocks = [b for b in resp.content if b.type == "tool_use"]
        if not tool_use_blocks:
            break

        tool_result_messages = []
        for block in tool_use_blocks:
            tool_name = block.name
            inputs    = block.input or {}

            if tool_name == "write_report":
                report_data = inputs
                break

            log(f"    → {tool_name}({list(inputs.keys())})")
            try:
                if tool_name == "run_correlation_scan":
                    result = at.run_correlation_scan(daily_rows, inputs["x_cols"], inputs["y_col"])
                elif tool_name == "run_regime_split":
                    result = at.run_regime_split(
                        daily_rows, inputs["x_col"], inputs["y_col"],
                        inputs["split_col"], inputs.get("method", "median"),
                    )
                elif tool_name == "run_rolling_correlation":
                    result = at.run_rolling_correlation(
                        daily_rows, inputs["x_col"], inputs["y_col"],
                        inputs.get("window", 20),
                    )
                elif tool_name == "run_tail_analysis":
                    result = at.run_tail_analysis(
                        daily_rows, inputs["y_col"], inputs["x_cols"],
                        inputs.get("pct", 10),
                    )
                elif tool_name == "run_decile_profile":
                    result = at.run_decile_profile(daily_rows, inputs["x_col"], inputs["y_col"])
                elif tool_name == "run_days_in_trade_profile":
                    result = at.run_days_in_trade_profile(
                        daily_rows, inputs["y_col"],
                        inputs.get("n_bins", 10),
                        inputs.get("group_by_phase", False),
                    )
                elif tool_name == "run_path_event_scan":
                    result = at.run_path_event_scan(
                        daily_rows,
                        inputs["event_col"],
                        inputs["threshold"],
                        inputs["direction"],
                        inputs["context_cols"],
                        inputs.get("window_days", 3),
                        inputs.get("threshold_type", "fixed"),
                    )
                else:
                    result = {"error": f"Unknown tool: {tool_name}"}
            except Exception as exc:
                result = {"error": str(exc)}

            tool_results_acc.append({"tool": tool_name, "inputs": inputs, "result": result})
            result_str = json.dumps(result)
            if len(result_str) > 8000:
                result_str = result_str[:8000] + "… (truncated)"
            tool_result_messages.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": result_str,
            })

        if report_data is not None:
            break

        if tool_result_messages:
            messages.append({"role": "user", "content": tool_result_messages})
        else:
            break

    # ── 4. Force write_report if not called ──────────────────────────────
    if report_data is None:
        log("  [RUNNING] Composing intratrade report (forced)…")
        messages.append({"role": "user", "content":
            "You have completed your analysis. Now call write_report with your findings. "
            "Be thorough — cite specific numbers from your tool results."})
        resp2 = await client.messages.create(
            model=model,
            max_tokens=4000,
            system=[{"type": "text", "text": system}],
            tools=_INTRATRADE_AGENTIC_TOOLS,
            tool_choice={"type": "tool", "name": "write_report"},
            messages=messages,
        )
        for block in resp2.content:
            if block.type == "tool_use" and block.name == "write_report":
                report_data = block.input
                break

    # ── 5. Build chart configs and assemble sections ──────────────────────
    log("[RUNNING] Composing intratrade report…")
    chart_configs = _build_intratrade_chart_configs(tool_results_acc)

    if report_data is None:
        sections = [{"type": "markdown",
                     "content": "(Report generation failed — no tool call returned)"}]
        for cid, chart in chart_configs.items():
            sections.append({"type": "chart", "chart_id": cid,
                              "title": chart["title"], "config": chart["config"]})
        return json.dumps(sections)

    exec_summary = (report_data.get("executive_summary") or "").strip()
    body         = (report_data.get("body") or "").strip()
    conclusions  = (report_data.get("conclusions") or "").strip()
    chart_seq    = report_data.get("chart_sequence") or []

    sections = []
    if exec_summary:
        sections.append({"type": "markdown", "content": exec_summary})

    body_paras = [p.strip() for p in body.split("\n\n") if p.strip()]
    referenced = set()
    if body_paras and chart_seq:
        charts_per_gap = max(1, len(chart_seq) // max(len(body_paras), 1))
        chart_idx = 0
        for para in body_paras:
            sections.append({"type": "markdown", "content": para})
            for _ in range(charts_per_gap):
                if chart_idx < len(chart_seq):
                    cid = chart_seq[chart_idx]
                    if cid in chart_configs:
                        sections.append({"type": "chart", "chart_id": cid,
                                          "title": chart_configs[cid]["title"],
                                          "config": chart_configs[cid]["config"]})
                        referenced.add(cid)
                    chart_idx += 1
        while chart_idx < len(chart_seq):
            cid = chart_seq[chart_idx]
            if cid in chart_configs:
                sections.append({"type": "chart", "chart_id": cid,
                                  "title": chart_configs[cid]["title"],
                                  "config": chart_configs[cid]["config"]})
                referenced.add(cid)
            chart_idx += 1
    else:
        for para in body_paras:
            sections.append({"type": "markdown", "content": para})

    if conclusions:
        sections.append({"type": "markdown",
                          "content": f"## Conclusions\n\n{conclusions}"})

    for cid, chart in chart_configs.items():
        if cid not in referenced:
            sections.append({"type": "chart", "chart_id": cid,
                              "title": chart["title"], "config": chart["config"]})

    return json.dumps(sections)


async def _execute_intratrade_run(
    main_pool,
    run_id: str,
    question: str,
    config: dict,
    model: str,
    knowledge_rules: list[str],
    log,
) -> str:
    """Load daily paths from a backtest upload and run the intratrade agentic pipeline."""
    upload_id = config["backtest_upload_id"]

    log("[RUNNING] Loading intratrade daily paths…")
    async with main_pool.acquire() as conn:
        await rdb.update_run(conn, run_id,
                             ai_summary="[RUNNING] Loading intratrade daily path data…")
        row = await conn.fetchrow(
            "SELECT daily_paths, has_daily_paths, path_count, date_from, date_to, data "
            "FROM research_backtest_uploads WHERE id = $1::uuid",
            upload_id,
        )
    if row is None:
        raise ValueError(f"Backtest upload {upload_id} not found")
    if not row["has_daily_paths"]:
        raise ValueError(
            "This upload has no daily paths. "
            "Only Mesosim JSON files with EndOfDay events support intratrade analysis."
        )

    daily_rows_raw = row["daily_paths"]
    daily_rows = (json.loads(daily_rows_raw) if isinstance(daily_rows_raw, str)
                  else list(daily_rows_raw))
    log(f"[RUNNING] Loaded {len(daily_rows)} compact daily rows — aligning to IV surface…")

    # Align to surface and compute derived fields here (background task — no timeout constraint).
    # Daily paths are stored compact (~9 cols) to stay under PostgreSQL's 256 MB JSONB limit;
    # full enrichment (100+ surface cols + since_entry deltas) happens at query time.
    date_from_str = str(row["date_from"]) if row["date_from"] else None
    date_to_str   = str(row["date_to"])   if row["date_to"]   else None
    data_raw      = row["data"]
    enriched_trades = (json.loads(data_raw) if isinstance(data_raw, str) else list(data_raw or []))

    try:
        aligned_daily, path_stats = await rbacktest.align_daily_paths_to_surface(
            daily_rows, main_pool, date_from_str, date_to_str
        )
        daily_rows = await asyncio.to_thread(
            rbacktest._compute_path_derived_fields, aligned_daily, enriched_trades
        )
        log(f"[RUNNING] Enriched {len(daily_rows)} daily rows "
            f"(surface match_rate={path_stats.get('match_rate', 0):.2f})")
    except Exception as exc:
        log(f"[WARNING] Surface alignment failed ({exc}) — proceeding with compact rows")

    log(f"[RUNNING] {len(daily_rows)} daily rows ready for analysis")

    # Categorize columns
    _PATH_FIELDS = {'position_id', 'date', 'dit', 'trade_phase', 'join_timestamp',
                    'daily_pnl_change', 'drawdown_from_peak', 'max_pnl_to_date', 'pct_of_peak_retained'}
    _GREEK_FIELDS = {'pos_pnl', 'pos_delta', 'pos_gamma', 'pos_theta', 'pos_vega', 'pos_wvega'}
    _SURFACE_META = {'trade_date', 'quote_time', 'id'}

    sample = daily_rows[0] if daily_rows else {}
    since_entry_fields = sorted(k for k in sample if k.endswith('_since_entry'))
    iv_level_fields = sorted(
        k for k in sample
        if k not in _PATH_FIELDS and k not in _GREEK_FIELDS and k not in _SURFACE_META
        and not k.startswith('_') and not k.endswith('_since_entry')
    )

    # Path-level summary stats
    dits  = [r.get('dit') for r in daily_rows if r.get('dit') is not None]
    phase_counts = {}
    for r in daily_rows:
        ph = r.get('trade_phase')
        if ph:
            phase_counts[ph] = phase_counts.get(ph, 0) + 1
    pos_ids = {r.get('position_id') for r in daily_rows if r.get('position_id')}

    import statistics as _stats
    path_summary = {
        'total_rows':        len(daily_rows),
        'total_trades':      len(pos_ids),
        'dit_min':           min(dits) if dits else None,
        'dit_p50':           round(_stats.median(dits), 1) if dits else None,
        'dit_max':           max(dits) if dits else None,
        'n_early':           phase_counts.get('early', 0),
        'n_middle':          phase_counts.get('middle', 0),
        'n_late':            phase_counts.get('late', 0),
        'iv_level_fields':   iv_level_fields,
        'since_entry_fields': since_entry_fields,
    }

    # Enrich workflow plan
    async with main_pool.acquire() as conn:
        await conn.execute(
            """UPDATE research_runs
               SET config = jsonb_set(
                   jsonb_set(config,
                       '{workflow_plan,feature_columns}', $2::jsonb),
                   '{workflow_plan,task_reasoning}', $3::jsonb)
               WHERE id = $1::uuid""",
            run_id,
            json.dumps(iv_level_fields + since_entry_fields),
            json.dumps(
                f"Intratrade path analysis: {len(daily_rows)} daily rows across "
                f"{len(pos_ids)} trades. "
                f"{len(iv_level_fields)} IV level fields + {len(since_entry_fields)} "
                f"since-entry delta fields available."
            ),
        )

    return await execute_intratrade_pipeline(
        main_pool=main_pool,
        run_id=run_id,
        question=question,
        daily_rows=daily_rows,
        path_summary=path_summary,
        model=model,
        knowledge_rules=knowledge_rules,
        log=log,
    )


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
    # ── Intratrade path analysis mode ───────────────────────────────────
    if config.get("intratrade_mode") and config.get("backtest_upload_id"):
        return await _execute_intratrade_run(
            main_pool=main_pool,
            run_id=run_id,
            question=question,
            config=config,
            model=model,
            knowledge_rules=knowledge_rules,
            log=log,
        )

    # ── Backtest agentic mode ────────────────────────────────────────────
    if config.get("backtest_upload_id"):
        return await _execute_backtest_run(
            main_pool=main_pool,
            run_id=run_id,
            question=question,
            config=config,
            model=model,
            knowledge_rules=knowledge_rules,
            log=log,
        )

    # ── P&L agentic mode ────────────────────────────────────────────────
    if config.get("pnl_upload_id"):
        return await _execute_pnl_pipeline(
            main_pool=main_pool,
            run_id=run_id,
            question=question,
            config=config,
            model=model,
            knowledge_rules=knowledge_rules,
            log=log,
        )

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
