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

from research import scanner, blocks, charts, db as rdb

_OI_TABLES = {"daily_features", "option_oi_surface", "underlying_ohlc"}

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
        },
        "required": ["task_type", "feature_columns", "outcome_columns"],
    },
}

_VIZ_TOOL = {
    "name": "select_charts",
    "description": "Select which charts to generate.",
    "input_schema": {
        "type": "object",
        "properties": {
            "charts": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "chart_type": {"type": "string", "enum": [
                            "bucket_profile", "equity_curve", "scatter", "yearly_consistency"]},
                        "ticker":     {"type": "string"},
                        "x_col":      {"type": "string"},
                        "y_col":      {"type": "string"},
                        "reason":     {"type": "string"},
                    },
                    "required": ["chart_type", "x_col", "y_col"],
                },
            },
        },
        "required": ["charts"],
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
- broad depth: include all plausibly relevant features
- targeted depth: 3-7 most directly relevant features
"""

_VIZ_SYSTEM = (
    "You are a data visualization selector for quantitative research. "
    "Choose the minimum set of charts that best answer the research question. "
    "Output ONLY valid JSON."
)

_VIZ_USER = """\
Research question: {question}

Top scan results (sorted by composite score):
{scan_summary}

Choose 3 to 6 charts. Available chart types:
- bucket_profile  : avg return + win rate across all decile buckets (shows full relationship shape)
- equity_curve    : simulated P&L for top/bottom decile over time (shows tradability)
- scatter         : raw scatter plot with regression line (shows distribution and outliers)
- yearly_consistency : per-year returns top vs bottom (shows temporal stability)

Return a JSON array (each element has these keys):
[
  {{
    "chart_type": "<bucket_profile|equity_curve|scatter|yearly_consistency>",
    "ticker": "<ticker or null>",
    "x_col": "<feature column>",
    "y_col": "<outcome column>",
    "reason": "<one sentence: why this chart answers the question>"
  }}
]
"""

_REPORT_USER = """\
RESEARCH QUESTION (primary directive — address this directly):
\"\"\"{question}\"\"\"

TASK TYPE: {task_type}
FOCUS: {scan_focus}
REPORT GUIDANCE: {report_guidance}

TOP SINGLE-FACTOR SIGNALS:
{signal_table}

FULL BUCKET PROFILES (top 3):
{bucket_profiles}

MULTI-FACTOR INTERACTIONS:
{interaction_summary}

EQUITY CURVES:
{equity_summary}

Write a focused research report (400-700 words):
1. Answer the research question directly in the first paragraph
2. Describe the full bucket profile shape — not just top/bottom
3. If multi-factor combos were tested, explain which improved on singles
   and describe as usable trading rules (e.g., "when X is high AND Y is rising")
4. Distinguish what is tradable from what looks spurious
5. Cite specific numbers (scores, spreads, consistency %)
6. State clearly what was tested and what did NOT work
7. End with a one-sentence actionable conclusion

Professional quant voice. No fluff.
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

    return plan


# ── Phase 1: AI-directed visualization selection ──────────────────────────────

async def select_visualizations(
    question: str,
    scan_results: list[dict],
    model: str = "claude-sonnet-4-6",
) -> list[dict]:
    """
    After scan results are available, ask Claude which charts to generate.
    Uses tool_use for guaranteed valid JSON output.
    """
    if _anthropic is None or not scan_results:
        return []

    ranked = sorted(
        [s for s in scan_results if "error" not in s],
        key=lambda s: (s.get("composite_interaction_score") or s.get("composite_score") or 0),
        reverse=True,
    )[:15]

    if not ranked:
        return []

    lines = []
    for s in ranked:
        rob = s.get("robustness", {})
        combo = s.get("combo")
        label = "+".join(combo) if combo else s.get("x_col", "?")
        lines.append(
            f"  [{s.get('composite_interaction_score') or s.get('composite_score', 0):.0f}] "
            f"{s.get('ticker') or 'all'} | {label} → {s.get('y_col')}: "
            f"pattern={s.get('pattern', '?')}, "
            f"consistency={rob.get('yearly_consistency_pct', '?')}%"
        )

    prompt = _VIZ_USER.format(question=question, scan_summary="\n".join(lines))

    client = _anthropic.AsyncAnthropic()
    resp = await client.messages.create(
        model=model,
        max_tokens=800,
        system=[{"type": "text", "text": _VIZ_SYSTEM,
                 "cache_control": {"type": "ephemeral"}}],
        tools=[_VIZ_TOOL],
        tool_choice={"type": "tool", "name": "select_charts"},
        messages=[{"role": "user", "content": prompt}],
    )

    # Extract from tool_use block
    for block in resp.content:
        if block.type == "tool_use" and block.name == "select_charts":
            charts = block.input.get("charts", [])
            return charts[:6]

    # Fallback: try parsing text
    raw = resp.content[0].text if resp.content and hasattr(resp.content[0], "text") else "[]"
    raw = re.sub(r"^```(?:json)?\s*", "", raw.strip())
    raw = re.sub(r"\s*```$", "", raw).strip()
    try:
        specs = json.loads(raw)
        return specs[:6] if isinstance(specs, list) else []
    except Exception:
        return []


# ── Phase 1: Compose final report ─────────────────────────────────────────────

async def _compose_report(
    question: str,
    plan: dict,
    ranked_scans: list[dict],
    equity_results: list[dict],
    model: str,
    knowledge_rules: list[str] = None,
) -> str:
    if _anthropic is None:
        return "(anthropic SDK not available)"

    valid = [s for s in ranked_scans if "error" not in s]
    top = valid[:10]

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

    # Bucket profiles for top 3
    bp_lines = []
    for s in top[:3]:
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
    for r in sorted(equity_results, key=lambda x: x.get("final_equity") or 0, reverse=True)[:8]:
        eq_lines.append(
            f"  {r.get('ticker') or 'all'} | {r.get('feature_col')} → {r.get('outcome_col')} "
            f"({r.get('which')}): final={r.get('final_equity')}, "
            f"maxDD={r.get('max_drawdown')}, n={r.get('n_trades')}"
        )

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

    client = _anthropic.AsyncAnthropic()
    msg = await client.messages.create(
        model=model,
        max_tokens=1500,
        system=[{"type": "text", "text": report_system,
                 "cache_control": {"type": "ephemeral"}}],
        messages=[{
            "role": "user",
            "content": _REPORT_USER.format(
                question=_safe(question),
                task_type=_safe(plan.get("task_type", "")),
                scan_focus=_safe(plan.get("scan_focus", "")),
                report_guidance=_safe(plan.get("report_guidance", "")),
                signal_table="\n".join(sig_lines),
                bucket_profiles="\n".join(bp_lines) if bp_lines else "(none)",
                interaction_summary="\n".join(int_lines) if int_lines else "(none tested)",
                equity_summary="\n".join(eq_lines) if eq_lines else "(none)",
            ),
        }],
    )
    return msg.content[0].text.strip()


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

    # ── 3. Deterministic chart generation (no LLM call) ────────────────────
    log("[3/5] Generating charts deterministically from top signals…")
    async with main_pool.acquire() as conn:
        await rdb.update_run(conn, run_id,
                             ai_summary="[RUNNING] Step 3/4: Generating charts…")

    # Build chart specs deterministically: bucket profiles for top 5, equity for top 3,
    # scatter for strongest, yearly consistency for most consistent
    single_ranked = [s for s in ranked if s.get("x_col") and not s.get("combo")]
    viz_specs = []

    # Bucket profiles for top 5 single-factor signals
    for s in single_ranked[:5]:
        viz_specs.append({"chart_type": "bucket_profile",
                          "ticker": s.get("ticker"),
                          "x_col": s["x_col"], "y_col": s["y_col"]})

    # Equity curves for top 3
    for s in single_ranked[:3]:
        viz_specs.append({"chart_type": "equity_curve",
                          "ticker": s.get("ticker"),
                          "x_col": s["x_col"], "y_col": s["y_col"]})

    # Scatter for the strongest signal
    if single_ranked:
        s = single_ranked[0]
        viz_specs.append({"chart_type": "scatter",
                          "ticker": s.get("ticker"),
                          "x_col": s["x_col"], "y_col": s["y_col"]})

    # Yearly consistency for the most consistent signal
    most_consistent = sorted(
        [s for s in single_ranked if (s.get("robustness") or {}).get("yearly_consistency_pct") is not None],
        key=lambda s: s["robustness"]["yearly_consistency_pct"], reverse=True)
    if most_consistent:
        s = most_consistent[0]
        viz_specs.append({"chart_type": "yearly_consistency",
                          "ticker": s.get("ticker"),
                          "x_col": s["x_col"], "y_col": s["y_col"]})

    log(f"  {len(viz_specs)} charts to generate from top {len(single_ranked)} signals")

    # Generate charts
    equity_results = []
    async with main_pool.acquire() as conn:
        for spec in viz_specs:
            c_type = spec.get("chart_type")
            x_col  = spec.get("x_col")
            y_col  = spec.get("y_col")
            ticker = spec.get("ticker") or None
            cache_key = ticker if ticker else "_all"
            rows = cache.get(cache_key)
            if not rows or not x_col or not y_col:
                continue

            avail = set(rows[0].keys())
            if x_col not in avail or y_col not in avail:
                log(f"  SKIP {c_type}: {x_col}/{y_col} not in fetched data")
                continue

            scan_r = next(
                (s for s in valid
                 if s.get("x_col") == x_col and s.get("y_col") == y_col
                 and (s.get("ticker") or None) == ticker),
                {},
            )
            title_base = f"{ticker or 'all'} | {x_col} → {y_col}"

            try:
                if c_type == "bucket_profile" and scan_r:
                    png = charts.bucket_profile_chart(scan_r)
                    if png:
                        await rdb.save_chart(conn, run_id, "bucket_profile",
                                             f"Profile {title_base}",
                                             png, ticker, x_col, y_col)

                elif c_type == "equity_curve":
                    rob = scan_r.get("robustness", {}) if scan_r else {}
                    conc = rob.get("concentration_risk", 1.0)
                    min_bn = rob.get("min_bucket_n", 0)
                    expanded = conc > 0.60 or min_bn < 20
                    top_w = "top2" if expanded else "top"
                    bot_w = "bottom2" if expanded else "bottom"
                    note = None
                    if expanded:
                        note = (f"Expanded to D9-D10/D1-D2 "
                                f"(concentration={conc:.2f}, min_n={min_bn})")

                    top_eq = blocks.equity_curve_from_rows(rows, x_col, y_col, top_w, 10, ticker)
                    bot_eq = blocks.equity_curve_from_rows(rows, x_col, y_col, bot_w, 10, ticker)
                    if note:
                        top_eq["concentration_note"] = note
                        bot_eq["concentration_note"] = note

                    await rdb.save_result(conn, run_id, "equity_curve_top",  x_col, y_col, top_eq, ticker)
                    await rdb.save_result(conn, run_id, "equity_curve_bottom", x_col, y_col, bot_eq, ticker)
                    equity_results.extend([top_eq, bot_eq])

                    png = charts.equity_curve_chart(top_eq, bot_eq)
                    if png:
                        await rdb.save_chart(conn, run_id, "equity_curve",
                                             f"Equity {title_base}",
                                             png, ticker, x_col, y_col)

                elif c_type == "scatter":
                    scat = blocks.scatter_sample_from_rows(rows, x_col, y_col, ticker=ticker)
                    if scat.get("points"):
                        png = charts.scatter_chart(scat)
                        if png:
                            await rdb.save_chart(conn, run_id, "scatter",
                                                 f"Scatter {title_base}",
                                                 png, ticker, x_col, y_col)

                elif c_type == "yearly_consistency":
                    ydata = (scan_r.get("robustness") or {}).get("yearly_data") if scan_r else None
                    if ydata:
                        yr_result = {
                            "years": ydata,
                            "consistency_pct": (scan_r.get("robustness") or {}).get(
                                "yearly_consistency_pct"),
                            "ticker": ticker,
                            "feature_col": x_col,
                            "outcome_col": y_col,
                        }
                        png = charts.yearly_consistency_chart(yr_result)
                        if png:
                            await rdb.save_chart(conn, run_id, "yearly_consistency",
                                                 f"Yearly {title_base}",
                                                 png, ticker, x_col, y_col)

                log(f"  ✓ {c_type}: {title_base}")
            except Exception as exc:
                log(f"  CHART ERROR {c_type} {title_base}: {exc}")

    # ── 4. Compose report ─────────────────────────────────────────────────
    log("[4/4] Composing report…")
    async with main_pool.acquire() as conn:
        await rdb.update_run(conn, run_id, ai_summary="[RUNNING] Step 4/4: Composing report…")

    summary = await _compose_report(question, plan, ranked, equity_results, model,
                                    knowledge_rules=knowledge_rules)
    log("  Report complete.")
    return summary
