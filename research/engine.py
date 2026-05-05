"""
Deterministic research engine — replaces the LLM agent loop.

Pipeline:
  1. Fetch data into memory (one query per ticker)
  2. Correlation matrix for all (ticker, x_col, y_col) combos
  3. Signal selection: filter by |pearson_r| > threshold
  4. For each selected signal: decile analysis, yearly consistency, equity curve
  5. Regression for tickers with multiple significant features on the same outcome
  6. Correlation heatmap (if multi-ticker)
  7. ONE LLM call to summarize all findings

No agent loop. No tool calls. No repeated SQL generation.
"""
import json
import os
from collections import defaultdict
from datetime import date as _date
from typing import Optional

import asyncpg

try:
    import anthropic
except ImportError:
    anthropic = None

from research import blocks, charts, scanner, db as rdb

_OI_TABLES = {"daily_features", "option_oi_surface", "underlying_ohlc"}


# ── Data fetch (reused from agent.py) ────────────────────────────────────────

async def _fetch_cache(pool: asyncpg.Pool, table: str,
                       tickers: list[str], x_cols: list[str], y_cols: list[str],
                       date_from: Optional[str], date_to: Optional[str],
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


# ── Formatting helpers ───────────────────────────────────────────────────────

def format_corr_table(corr_results: list[dict]) -> str:
    """Format correlation matrix as a readable text table."""
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


def _build_fallback_summary(ranked_scans: list[dict],
                            equity_results: list[dict]) -> str:
    """Build a summary when the LLM call fails."""
    valid = [s for s in ranked_scans if "error" not in s]
    top = [s for s in valid if s.get("composite_score", 0) >= 30]

    lines = [f"Analysis complete. {len(valid)} pairs scanned, "
             f"{len(equity_results)} equity curves generated.\n"]

    if top:
        lines.append(f"Top signals (robustness score >= 30):")
        for s in top[:10]:
            rob = s.get("robustness", {})
            lines.append(
                f"  [{s.get('composite_score', 0):.0f}] "
                f"{s.get('ticker') or 'all'} | {s.get('x_col')} -> {s.get('y_col')}: "
                f"pattern={s.get('pattern')}, "
                f"consistency={rob.get('yearly_consistency_pct', '?')}%, "
                f"half_stable={'Y' if rob.get('half_sample_stable') else 'N'}")
    else:
        lines.append("No signals scored above 30 (no robust relationships detected).")

    fragile = [s for s in valid[:20] if s.get("robustness", {}).get("loyo_fragile")]
    if fragile:
        lines.append(f"\n{len(fragile)} signals flagged as fragile (leave-one-year-out flips sign).")

    return "\n".join(lines)


# ── LLM summary ──────────────────────────────────────────────────────────────

async def _generate_summary(
    question: str,
    config: dict,
    ranked_scans: list[dict],
    equity_results: list[dict],
    interaction_results: list[dict] = None,
    model: str = "claude-sonnet-4-6",
) -> str:
    """ONE LLM call with ranked findings → narrative summary."""
    if anthropic is None:
        return "(anthropic SDK not available — no AI summary)"

    lines = [
        f"Table: {config.get('table')}",
        f"Tickers: {', '.join(config.get('tickers') or ['all'])}",
        f"Features: {', '.join(config.get('x_columns') or [])}",
        f"Outcomes: {', '.join(config.get('y_columns') or [])}",
        f"Period: {config.get('date_from') or 'earliest'} to {config.get('date_to') or 'latest'}",
        f"Total pairs scanned: {len(ranked_scans)}",
        "",
        "=== Top Signals by Robustness Score ===",
        f"{'Score':>5}  {'Ticker':<8} {'Feature':<35} {'Outcome':<16} {'Pattern':<22} "
        f"{'Pr':>6} {'Sr':>6} {'Mono':>5} {'Consist':>7} {'HalfOK':>6} {'Conc':>5}",
        "-" * 130,
    ]

    for s in ranked_scans[:20]:
        rob = s.get("robustness", {})
        lines.append(
            f"{s.get('composite_score', 0):>5.0f}  "
            f"{(s.get('ticker') or 'all'):<8} "
            f"{s.get('x_col', ''):<35} "
            f"{s.get('y_col', ''):<16} "
            f"{s.get('pattern', ''):<22} "
            f"{s.get('pearson_r', 0):>6.3f} "
            f"{s.get('spearman_r', 0):>6.3f} "
            f"{s.get('monotonicity', 0):>5.2f} "
            f"{str(rob.get('yearly_consistency_pct', '—')) + '%':>7} "
            f"{'Y' if rob.get('half_sample_stable') else 'N':>6} "
            f"{rob.get('concentration_risk', 1.0):>5.2f}"
        )

    # Full bucket profiles for top 5 signals (all deciles visible to Claude)
    lines.append("\n=== Full Bucket Profiles (top signals) ===")
    for s in ranked_scans[:5]:
        bs = s.get("bucket_stats") or []
        valid_bs = [b for b in bs if b is not None]
        if not valid_bs:
            continue
        lines.append(f"\n{s.get('ticker') or 'all'} | {s.get('x_col')} -> {s.get('y_col')} "
                     f"(pattern={s.get('pattern')}, score={s.get('composite_score', 0):.0f}):")
        lines.append(f"  {'Bucket':>6} {'N':>5} {'AvgRet':>9} {'MedRet':>9} {'WinRate':>8} {'Sharpe':>7}")
        for b in valid_bs:
            lines.append(
                f"  {b['bucket']:>6} {b['n']:>5} "
                f"{b['avg_ret']*100:>8.3f}% {b['med_ret']*100:>8.3f}% "
                f"{b['win_rate']*100:>7.1f}% {b['sharpe']:>7.3f}")

    # Flag fragile signals
    fragile = [s for s in ranked_scans[:20] if s.get("robustness", {}).get("loyo_fragile")]
    if fragile:
        lines.append("\nFRAGILE (leave-one-year-out flips sign):")
        for s in fragile:
            lines.append(f"  {s.get('ticker') or 'all'} | {s.get('x_col')} -> {s.get('y_col')}")

    # Equity curve highlights
    if equity_results:
        lines.append("\n=== Equity Curves ===")
        for r in sorted(equity_results,
                        key=lambda x: abs(x.get("cumulative_return") or x.get("final_equity") or 0),
                        reverse=True)[:15]:
            cum = r.get("cumulative_return") or r.get("final_equity")
            lines.append(
                f"  {r.get('ticker') or 'all'} | {r.get('feature_col')} -> {r.get('outcome_col')} "
                f"({r.get('which')}): cumReturn={cum*100:.1f}%, "
                f"maxDD={r.get('max_drawdown')}, trades={r.get('n_trades')}, "
                f"winRate={r.get('win_rate')}")

    # Multi-factor combo highlights
    if interaction_results:
        top_combos = sorted(
            [r for r in interaction_results if r.get("composite_interaction_score", 0) > 0],
            key=lambda r: r.get("composite_interaction_score", 0), reverse=True)
        if top_combos:
            lines.append("\n=== Multi-Factor Combos ===")
            for r in top_combos[:10]:
                combo = "+".join(r.get("combo", []))
                bz = r.get("best_quadrant") or r.get("best_octant") or r.get("best_zone") or {}
                rob = r.get("robustness") or {}
                lines.append(
                    f"  {r.get('ticker') or 'all'} | {combo} -> {r.get('y_col')}: "
                    f"score={r.get('composite_interaction_score', 0):.0f}, "
                    f"lift={r.get('interaction_lift', 0):+.4f}, "
                    f"zone={bz.get('label', '?')} "
                    f"(avg={bz.get('avg_ret', 0):.4f}, WR={bz.get('win_rate', 0):.0%}), "
                    f"R²={r.get('ols_r2', 0):.4f}"
                    + (f", warnings={rob.get('warnings')}" if rob.get("warnings") else ""))

    # Null results
    null_count = sum(1 for s in ranked_scans if s.get("composite_score", 0) < 10)
    if null_count:
        lines.append(f"\n{null_count} pairs scored below 10 (no meaningful relationship).")

    findings_text = "\n".join(lines)

    # Build the prompt with the user's question as a PRIORITY DIRECTIVE
    client = anthropic.AsyncAnthropic()
    _ENGINE_WRITER_SYSTEM = (
        "You are a professional quantitative researcher writing research summaries. "
        "Address the user's specific question as your primary directive. If they asked "
        "about specific deciles, patterns, comparisons, or conditions — answer those "
        "directly using the bucket profile data provided.\n\n"
        "Always cover:\n"
        "1. Full bucket profile interpretation: don't just report top/bottom — describe "
        "the shape across ALL deciles. Flag adjacent deciles with similar performance "
        "(e.g., 'deciles 8-10 all show positive returns, not just the top').\n"
        "2. Multi-factor combos: which improved on singles? Describe as usable rules.\n"
        "3. Non-linear patterns: U-shaped, threshold, isolated regions.\n"
        "4. Equity curve viability.\n"
        "5. Fragile/overfit warnings.\n"
        "6. Suggested next steps.\n\n"
        "Use professional quant research language. Translate findings into potential "
        "trading rules. Do NOT just list numbers — interpret patterns. Write 500-800 words."
    )
    msg = await client.messages.create(
        model=model,
        max_tokens=2000,
        system=[{"type": "text", "text": _ENGINE_WRITER_SYSTEM,
                 "cache_control": {"type": "ephemeral"}}],
        messages=[{
            "role": "user",
            "content": (
                f"USER'S RESEARCH QUESTION:\n"
                f'"""\n{question}\n"""\n\n'
                f"ANALYSIS RESULTS:\n{findings_text}"
            ),
        }],
    )
    return msg.content[0].text.strip()


# ── Main pipeline ────────────────────────────────────────────────────────────

async def run_pipeline(
    main_pool: asyncpg.Pool,
    oi_pool: asyncpg.Pool,
    run_id: str,
    question: str,
    config: dict,
    model: str = "claude-sonnet-4-6",
    signal_threshold: float = 0.03,
    max_signals: int = 30,
    analysis_types: Optional[set[str]] = None,
    log=print,
) -> str:
    """
    Deterministic research pipeline with broad relationship scanning.
    Returns the AI summary string. All results saved to the database.
    """
    if analysis_types is None:
        analysis_types = {"scan", "combo", "equity_curve", "regression"}

    table    = config.get("table", "daily_features")
    tickers  = config.get("tickers") or []
    buckets  = config.get("buckets") or {}
    x_cols   = config.get("x_columns") or []
    y_cols   = config.get("y_columns") or []
    date_from = config.get("date_from")
    date_to   = config.get("date_to")

    # Normalize: if no buckets, make one from x_cols
    if not buckets and x_cols:
        buckets = {"features": x_cols}
    if not x_cols:
        x_cols = list(dict.fromkeys(col for cols in buckets.values() for col in cols))

    data_pool = oi_pool if table in _OI_TABLES else main_pool

    # ── Step 1: Fetch data ───────────────────────────────────────────────
    log("Step 1/5: Fetching data...")
    cache, fetch_errors = await _fetch_cache(
        data_pool, table, tickers, x_cols, y_cols, date_from, date_to, log)

    if not cache:
        detail = "\n".join(fetch_errors) if fetch_errors else "query returned 0 rows"
        raise ValueError(f"No data loaded.\n{detail}")

    # ── Step 2: Broad relationship scan ──────────────────────────────────
    total_combos = sum(1 for _ in cache for _ in x_cols for _ in y_cols)
    log(f"Step 2/5: Scanning {total_combos} (ticker × feature × outcome) pairs...")
    all_scans = []

    scan_errors = 0
    async with main_pool.acquire() as conn:
        for cache_key, rows in cache.items():
            ticker = None if cache_key == "_all" else cache_key
            cols_available = set(rows[0].keys()) if rows else set()
            log(f"  Ticker={ticker or 'all'}: {len(rows)} rows, "
                f"cols available: {sorted(cols_available - {'trade_date'})}")
            for x_col in x_cols:
                if x_col not in cols_available:
                    log(f"    SKIP x_col={x_col} — not in data")
                    continue
                for y_col in y_cols:
                    if y_col not in cols_available:
                        log(f"    SKIP y_col={y_col} — not in data")
                        continue
                    if x_col == y_col:
                        continue
                    try:
                        scan_result = scanner.scan_relationship(rows, x_col, y_col, ticker)
                    except Exception as exc:
                        log(f"    SCAN ERROR {ticker or 'all'} {x_col}->{y_col}: {exc}")
                        scan_errors += 1
                        continue

                    if "error" in scan_result:
                        log(f"    INSUF DATA {ticker or 'all'} {x_col}->{y_col}: "
                            f"n={scan_result.get('n', 0)}")
                        scan_errors += 1

                    await rdb.save_result(conn, run_id, "scan",
                                          x_col, y_col, scan_result, ticker)
                    all_scans.append(scan_result)

                    # Scatter chart for every pair
                    scatter = blocks.scatter_sample_from_rows(rows, x_col, y_col, ticker=ticker)
                    if scatter.get("points"):
                        png = charts.scatter_chart(scatter)
                        if png:
                            await rdb.save_chart(
                                conn, run_id, "scatter",
                                f"Scatter {ticker or 'all'} | {x_col} -> {y_col}",
                                png, ticker, x_col, y_col)

    valid_scans = [s for s in all_scans if "error" not in s]
    error_scans = [s for s in all_scans if "error" in s]
    log(f"  {len(valid_scans)} pairs scanned successfully, "
        f"{len(error_scans)} insufficient data, {scan_errors} errors.")

    # ── Step 3: Single-factor baseline ranking ────────────────────────────
    ranked = sorted(valid_scans,
                    key=lambda s: s.get("composite_score", 0), reverse=True)
    top_signals = ranked[:max_signals]
    log(f"Step 3/6: Single-factor baseline — top {len(top_signals)} signals.")

    # Bucket profile charts for top single-factor signals
    async with main_pool.acquire() as conn:
        for s in top_signals[:10]:  # cap charts at 10
            png = charts.bucket_profile_chart(s)
            if png:
                await rdb.save_chart(
                    conn, run_id, "bucket_profile",
                    f"Profile {s.get('ticker') or 'all'} | {s['x_col']} -> {s['y_col']}",
                    png, s.get("ticker"), s["x_col"], s["y_col"])

    # ── Step 4: Multi-factor combo scan (PRIMARY OUTPUT) ─────────────────
    from itertools import combinations, product
    combo_results = []
    bucket_names = list(buckets.keys())
    n_buckets_available = len(bucket_names)

    if "combo" in analysis_types and n_buckets_available >= 2:
        log(f"Step 4/6: Multi-factor combo scan across {n_buckets_available} buckets...")

        # Build best single-factor Sharpe per (ticker, y_col) for lift calculation
        best_single_sharpe: dict[tuple, float] = {}
        best_single_feature: dict[tuple, str] = {}
        for s in valid_scans:
            key = (s.get("ticker"), s["y_col"])
            bsb = s.get("best_single_bucket") or {}
            baz = s.get("best_adjacent_zone") or {}
            sharpe = max(abs(bsb.get("sharpe", 0)), abs(baz.get("sharpe", 0)))
            if sharpe > best_single_sharpe.get(key, 0):
                best_single_sharpe[key] = sharpe
                best_single_feature[key] = s["x_col"]

        # Cap within-bucket features to top 3 by single-factor score
        feat_scores = {(s.get("ticker"), s["x_col"], s["y_col"]): s.get("composite_score", 0)
                       for s in valid_scans}

        n_combos = 0
        async with main_pool.acquire() as conn:
            for cache_key, rows in cache.items():
                ticker = None if cache_key == "_all" else cache_key
                for y_col in y_cols:
                    baseline = best_single_sharpe.get((ticker, y_col), 0)
                    baseline_feat = best_single_feature.get((ticker, y_col), "?")

                    # Get top 3 per bucket for this ticker/y_col
                    bucket_tops = {}
                    for bname, bcols in buckets.items():
                        scored = [(c, feat_scores.get((ticker, c, y_col), 0)) for c in bcols
                                  if c in (set(rows[0].keys()) if rows else set())]
                        scored.sort(key=lambda x: x[1], reverse=True)
                        bucket_tops[bname] = [c for c, _ in scored[:3]]

                    # Generate cross-bucket combos for 2, 3, 4 buckets
                    for k in range(2, min(n_buckets_available + 1, 5)):
                        for bucket_combo in combinations(bucket_names, k):
                            feature_lists = [bucket_tops.get(b, []) for b in bucket_combo]
                            if any(not fl for fl in feature_lists):
                                continue
                            for feat_combo in product(*feature_lists):
                                feat_list = list(feat_combo)
                                try:
                                    if len(feat_list) == 2:
                                        result = scanner.scan_interaction_2f(
                                            rows, feat_list[0], feat_list[1],
                                            y_col, ticker, baseline)
                                    elif len(feat_list) == 3:
                                        result = scanner.scan_interaction_3f(
                                            rows, feat_list[0], feat_list[1], feat_list[2],
                                            y_col, ticker, baseline)
                                    elif len(feat_list) == 4:
                                        result = scanner.scan_interaction_4f(
                                            rows, feat_list, y_col, ticker, baseline)
                                    else:
                                        continue
                                except Exception as exc:
                                    log(f"    COMBO ERROR {ticker or 'all'} "
                                        f"{'+'.join(feat_list)}->{y_col}: {exc}")
                                    continue
                                if result is None:
                                    continue

                                # Add bucket info and baseline reference
                                result["buckets_used"] = list(bucket_combo)
                                result["baseline_best_single"] = {
                                    "feature": baseline_feat, "sharpe": baseline}

                                # Robustness check for best zone
                                bz = result.get("best_quadrant") or result.get("best_octant") or result.get("best_zone") or {}
                                if bz.get("label"):
                                    try:
                                        rob = scanner.combo_robustness(
                                            rows, feat_list, y_col, bz["label"], ticker)
                                        result["robustness"] = rob
                                        result["overfit_warnings"] = rob.get("warnings", [])
                                    except Exception:
                                        pass

                                x_label = "+".join(feat_list)
                                await rdb.save_result(conn, run_id, "combo",
                                                      x_label, y_col, result, ticker)
                                combo_results.append(result)
                                n_combos += 1

                                # Chart for 2-factor combos
                                if len(feat_list) == 2:
                                    png = charts.quadrant_chart(result)
                                    if png:
                                        await rdb.save_chart(
                                            conn, run_id, "combo_quadrant",
                                            f"Combo {ticker or 'all'} | {x_label} -> {y_col}",
                                            png, ticker, x_label, y_col)

        log(f"  {n_combos} multi-factor combos tested.")
    elif "combo" in analysis_types:
        log("Step 4/6: Combo scan — skipped (need 2+ buckets).")

    # ── Step 5: Equity curves + regression for top signals ───────────────
    equity_results = []
    if "equity_curve" in analysis_types and top_signals:
        log(f"Step 5/6: Equity curves for top {len(top_signals)} signals...")
        async with main_pool.acquire() as conn:
            for s in top_signals:
                ticker = s.get("ticker")
                x_col, y_col = s["x_col"], s["y_col"]
                cache_key = ticker if ticker else "_all"
                rows = cache.get(cache_key, [])

                rob = s.get("robustness", {})
                concentration = rob.get("concentration_risk", 1.0)
                min_bucket_n = rob.get("min_bucket_n", 0)

                # Expand to D9-D10 / D1-D2 when extreme decile is too concentrated or too thin
                use_expanded = concentration > 0.60 or min_bucket_n < 20
                top_which = "top2" if use_expanded else "top"
                bot_which = "bottom2" if use_expanded else "bottom"

                concentration_note = None
                if use_expanded:
                    if concentration > 0.60:
                        concentration_note = (
                            f"D10 concentration: {concentration*100:.0f}% of signal from one year"
                            f" — expanded to D9-D10 / D1-D2"
                        )
                    else:
                        concentration_note = (
                            f"Extreme decile thin (min n={min_bucket_n} per bucket)"
                            f" — expanded to D9-D10 / D1-D2"
                        )
                    log(f"    {ticker or 'all'} {x_col}->{y_col}: {concentration_note}")

                top = blocks.equity_curve_from_rows(rows, x_col, y_col, top_which, 10, ticker)
                bot = blocks.equity_curve_from_rows(rows, x_col, y_col, bot_which, 10, ticker)

                if concentration_note:
                    top["concentration_note"] = concentration_note
                    bot["concentration_note"] = concentration_note

                await rdb.save_result(conn, run_id, "equity_curve_top",
                                      x_col, y_col, top, ticker)
                await rdb.save_result(conn, run_id, "equity_curve_bottom",
                                      x_col, y_col, bot, ticker)
                equity_results.extend([top, bot])

                if top.get("points"):
                    await rdb.save_series(conn, run_id, x_col, "equity_curve_top",
                                          top["points"], ticker=ticker, y_col=y_col)
                if bot.get("points"):
                    await rdb.save_series(conn, run_id, x_col, "equity_curve_bottom",
                                          bot["points"], ticker=ticker, y_col=y_col)

                png = charts.equity_curve_chart(top, bot)
                if png:
                    await rdb.save_chart(
                        conn, run_id, "equity_curve",
                        f"Equity {ticker or 'all'} | {x_col} -> {y_col}",
                        png, ticker, x_col, y_col)

                # When already expanded, skip the adjacency-conditional extras.
                # When using strict single decile, still offer D9-D10/D1-D2 if adjacent
                # decile is directionally consistent with D10/D1.
                if not use_expanded:
                    bs = s.get("bucket_stats") or []
                    valid_bs = [b for b in bs if b is not None]
                    if len(valid_bs) >= 9:
                        d9_avg = valid_bs[8].get("avg_ret", 0) if len(valid_bs) > 8 else 0
                        d10_avg = valid_bs[9].get("avg_ret", 0) if len(valid_bs) > 9 else 0
                        if d9_avg > 0 and d10_avg > 0:
                            top2 = blocks.equity_curve_from_rows(rows, x_col, y_col, "top2", 10, ticker)
                            if top2.get("points"):
                                await rdb.save_result(conn, run_id, "equity_curve_top2",
                                                      x_col, y_col, top2, ticker)
                                await rdb.save_series(conn, run_id, x_col, "equity_curve_top2",
                                                      top2["points"], ticker=ticker, y_col=y_col)
                                equity_results.append(top2)

                        d1_avg = valid_bs[0].get("avg_ret", 0)
                        d2_avg = valid_bs[1].get("avg_ret", 0) if len(valid_bs) > 1 else 0
                        if d1_avg < 0 and d2_avg < 0:
                            bot2 = blocks.equity_curve_from_rows(rows, x_col, y_col, "bottom2", 10, ticker)
                            if bot2.get("points"):
                                await rdb.save_result(conn, run_id, "equity_curve_bottom2",
                                                      x_col, y_col, bot2, ticker)
                                await rdb.save_series(conn, run_id, x_col, "equity_curve_bottom2",
                                                      bot2["points"], ticker=ticker, y_col=y_col)
                                equity_results.append(bot2)

        log(f"  {len(equity_results)} equity curves done.")

    # Regression for multi-feature combos
    regression_results = []
    if "regression" in analysis_types and top_signals:
        groups: dict[tuple, list[str]] = defaultdict(list)
        for s in top_signals:
            key = (s.get("ticker"), s["y_col"])
            if s["x_col"] not in groups[key]:
                groups[key].append(s["x_col"])

        multi_groups = {k: v for k, v in groups.items() if len(v) >= 2}
        if multi_groups:
            log(f"  Regression: {len(multi_groups)} multi-feature groups...")
            async with main_pool.acquire() as conn:
                for (ticker, y_col), x_list in multi_groups.items():
                    cache_key = ticker if ticker else "_all"
                    rows = cache.get(cache_key, [])
                    result = blocks.regression_from_rows(rows, x_list, y_col, ticker)
                    await rdb.save_result(conn, run_id, "regression",
                                          "+".join(x_list), y_col, result, ticker)
                    regression_results.append(result)

    # Correlation heatmap
    if len(cache) > 1 and x_cols and y_cols:
        log("  Generating correlation heatmap...")
        corr_for_heatmap = [
            {"ticker": s.get("ticker"), "x_col": s["x_col"], "y_col": s["y_col"],
             "pearson_r": s.get("pearson_r")}
            for s in valid_scans if s.get("pearson_r") is not None
        ]
        png = charts.correlation_heatmap(
            corr_for_heatmap, tickers or list(cache.keys()), x_cols, y_cols)
        if png:
            async with main_pool.acquire() as conn:
                await rdb.save_chart(conn, run_id, "correlation_heatmap",
                                     "Pearson Correlation Heatmap", png)

    # ── Step 6: LLM summary (ONE call) ──────────────────────────────────
    log("Step 6/6: Generating AI summary...")
    try:
        summary = await _generate_summary(
            question, config, ranked, equity_results,
            interaction_results=combo_results, model=model)
        log("  Summary generated.")
    except Exception as exc:
        log(f"  LLM summary failed: {exc}. Using fallback.")
        summary = _build_fallback_summary(ranked, equity_results)

    return summary
