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
        f"Research question: {question}",
        f"Table: {config.get('table')}",
        f"Tickers: {', '.join(config.get('tickers') or ['all'])}",
        f"Features: {', '.join(config.get('x_columns') or [])}",
        f"Outcomes: {', '.join(config.get('y_columns') or [])}",
        f"Period: {config.get('date_from') or 'earliest'} to {config.get('date_to') or 'latest'}",
        f"\nTotal pairs scanned: {len(ranked_scans)}",
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

    # Flag fragile signals
    fragile = [s for s in ranked_scans[:20] if s.get("robustness", {}).get("loyo_fragile")]
    if fragile:
        lines.append("\nFRAGILE (leave-one-year-out flips sign):")
        for s in fragile:
            lines.append(f"  {s.get('ticker') or 'all'} | {s.get('x_col')} -> {s.get('y_col')}")

    # Equity curve highlights for top signals
    if equity_results:
        lines.append("\n=== Equity Curves (top signals) ===")
        for r in sorted(equity_results,
                        key=lambda x: x.get("final_equity") or 0, reverse=True)[:10]:
            lines.append(
                f"  {r.get('ticker') or 'all'} | {r.get('feature_col')} -> {r.get('outcome_col')} "
                f"({r.get('which')}): final={r.get('final_equity')}, "
                f"maxDD={r.get('max_drawdown')}, trades={r.get('n_trades')}, "
                f"winRate={r.get('win_rate')}")

    # Interaction highlights
    if interaction_results:
        top_interactions = sorted(
            [r for r in interaction_results if r.get("composite_interaction_score", 0) > 0],
            key=lambda r: r.get("composite_interaction_score", 0), reverse=True)
        if top_interactions:
            lines.append("\n=== Multi-Factor Interactions ===")
            for r in top_interactions[:10]:
                combo = "+".join(r.get("combo", []))
                bq = r.get("best_quadrant") or r.get("best_octant") or {}
                lines.append(
                    f"  {r.get('ticker') or 'all'} | {combo} -> {r.get('y_col')}: "
                    f"score={r.get('composite_interaction_score', 0):.0f}, "
                    f"lift={r.get('interaction_lift', 0):+.4f}, "
                    f"best zone={bq.get('label', '?')} "
                    f"(avg={bq.get('avg_ret', 0):.4f}, WR={bq.get('win_rate', 0):.0%}), "
                    f"R²={r.get('ols_r2', 0):.4f}")

    # Null results summary
    null_count = sum(1 for s in ranked_scans if s.get("composite_score", 0) < 10)
    if null_count:
        lines.append(f"\n{null_count} pairs scored below 10 (no meaningful relationship detected).")

    findings_text = "\n".join(lines)

    client = anthropic.AsyncAnthropic()
    msg = await client.messages.create(
        model=model,
        max_tokens=1500,
        messages=[{
            "role": "user",
            "content": (
                f"{findings_text}\n\n"
                "Write a concise research summary (400-600 words). Cover:\n"
                "1. Strongest single-factor signals: pattern, robustness score, consistency\n"
                "2. Multi-factor interactions: which combos improved on singles? Describe as usable rules\n"
                "3. Non-linear patterns: any U-shaped, threshold, or isolated-region findings\n"
                "4. Equity curve viability for top signals\n"
                "5. Fragile or overfit-looking results and why\n"
                "6. Null results worth noting (features that do NOT predict)\n"
                "7. Suggested next steps — which combos deserve further testing?\n\n"
                "Use professional quant research language. Translate findings into potential trading rules."
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
        analysis_types = {"scan", "interaction", "equity_curve", "regression"}

    table    = config.get("table", "daily_features")
    tickers  = config.get("tickers") or []
    x_cols   = config.get("x_columns") or []
    y_cols   = config.get("y_columns") or []
    date_from = config.get("date_from")
    date_to   = config.get("date_to")

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

    # ── Step 3: Rank by robustness & deep-dive top signals ───────────────
    ranked = sorted(valid_scans,
                    key=lambda s: s.get("composite_score", 0), reverse=True)
    top_signals = ranked[:max_signals]
    log(f"Step 3/6: Top {len(top_signals)} signals selected for deep analysis.")

    # Bucket profile charts for top signals
    async with main_pool.acquire() as conn:
        for s in top_signals:
            png = charts.bucket_profile_chart(s)
            if png:
                await rdb.save_chart(
                    conn, run_id, "bucket_profile",
                    f"Profile {s.get('ticker') or 'all'} | {s['x_col']} -> {s['y_col']}",
                    png, s.get("ticker"), s["x_col"], s["y_col"])

    # ── Step 3b: Multivariate interaction scan ──────────────────────────
    interaction_results = []
    if "interaction" in analysis_types and len(x_cols) >= 2:
        from itertools import combinations
        log(f"Step 3b: Interaction scan (2-factor and 3-factor combos)...")

        # Build best single-factor Sharpe lookup per (ticker, y_col)
        best_single_sharpe: dict[tuple, float] = {}
        for s in valid_scans:
            key = (s.get("ticker"), s["y_col"])
            bsb = s.get("best_single_bucket") or {}
            sharpe = abs(bsb.get("sharpe", 0))
            baz = s.get("best_adjacent_zone") or {}
            sharpe = max(sharpe, abs(baz.get("sharpe", 0)))
            best_single_sharpe[key] = max(best_single_sharpe.get(key, 0), sharpe)

        # For each (ticker, y_col), get usable features (had valid scans)
        usable_features: dict[tuple, list[str]] = defaultdict(list)
        for s in valid_scans:
            key = (s.get("ticker"), s["y_col"])
            if s["x_col"] not in usable_features[key]:
                usable_features[key].append(s["x_col"])

        n_2f = 0
        n_3f = 0
        async with main_pool.acquire() as conn:
            for (ticker, y_col), feats in usable_features.items():
                cache_key = ticker if ticker else "_all"
                rows = cache.get(cache_key, [])
                baseline = best_single_sharpe.get((ticker, y_col), 0)

                # Cap features to top 6 by single-factor score for this y_col
                feat_scores = {}
                for s in valid_scans:
                    if s.get("ticker") == ticker and s["y_col"] == y_col:
                        feat_scores[s["x_col"]] = s.get("composite_score", 0)
                top_feats = sorted(feats, key=lambda f: feat_scores.get(f, 0),
                                   reverse=True)[:6]

                # 2-factor combos
                best_2f_sharpe: dict[tuple, float] = {}
                for fa, fb in combinations(top_feats, 2):
                    try:
                        result = scanner.scan_interaction_2f(
                            rows, fa, fb, y_col, ticker, baseline)
                    except Exception as exc:
                        log(f"    2F ERROR {ticker or 'all'} {fa}+{fb}->{y_col}: {exc}")
                        continue
                    if result is None:
                        continue
                    await rdb.save_result(conn, run_id, "interaction",
                                          f"{fa}+{fb}", y_col, result, ticker)
                    interaction_results.append(result)
                    n_2f += 1
                    # Track best 2f Sharpe for 3f baseline
                    bq = result.get("best_quadrant") or {}
                    best_2f_sharpe[(fa, fb)] = abs(bq.get("sharpe", 0))

                    png = charts.quadrant_chart(result)
                    if png:
                        await rdb.save_chart(
                            conn, run_id, "interaction_quadrant",
                            f"Interaction {ticker or 'all'} | {fa}+{fb} -> {y_col}",
                            png, ticker, f"{fa}+{fb}", y_col)

                # 3-factor combos (only features from top 2f results)
                if len(top_feats) >= 3:
                    top_2f_feats = set()
                    top_2f_results = sorted(
                        [r for r in interaction_results
                         if r.get("ticker") == ticker and r["y_col"] == y_col],
                        key=lambda r: r.get("composite_interaction_score", 0),
                        reverse=True)[:5]
                    for r in top_2f_results:
                        top_2f_feats.update(r.get("combo", []))

                    best_2f_overall = max(best_2f_sharpe.values()) if best_2f_sharpe else baseline
                    for fa, fb, fc in combinations(sorted(top_2f_feats), 3):
                        try:
                            result = scanner.scan_interaction_3f(
                                rows, fa, fb, fc, y_col, ticker, best_2f_overall)
                        except Exception as exc:
                            log(f"    3F ERROR {ticker or 'all'} {fa}+{fb}+{fc}->{y_col}: {exc}")
                            continue
                        if result is None:
                            continue
                        await rdb.save_result(conn, run_id, "interaction_3f",
                                              f"{fa}+{fb}+{fc}", y_col, result, ticker)
                        interaction_results.append(result)
                        n_3f += 1

        log(f"  {n_2f} two-factor + {n_3f} three-factor interactions tested.")
    else:
        log("Step 3b: Interaction scan — skipped (need 2+ features).")

    # ── Step 4: Equity curves + regression for top signals ───────────────
    equity_results = []
    if "equity_curve" in analysis_types and top_signals:
        log(f"Step 4/5: Equity curves for top {len(top_signals)} signals...")
        async with main_pool.acquire() as conn:
            for s in top_signals:
                ticker = s.get("ticker")
                x_col, y_col = s["x_col"], s["y_col"]
                cache_key = ticker if ticker else "_all"
                rows = cache.get(cache_key, [])

                top = blocks.equity_curve_from_rows(rows, x_col, y_col, "top", 10, ticker)
                bot = blocks.equity_curve_from_rows(rows, x_col, y_col, "bottom", 10, ticker)

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
            interaction_results=interaction_results, model=model)
        log("  Summary generated.")
    except Exception as exc:
        log(f"  LLM summary failed: {exc}. Using fallback.")
        summary = _build_fallback_summary(ranked, equity_results)

    return summary
