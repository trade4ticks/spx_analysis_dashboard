#!/usr/bin/env python
"""
Research runner CLI.

Usage:
    python run_research.py --config research_config.yaml
    python run_research.py --config research_config.yaml --name "My run"
    python run_research.py --list
"""
import argparse
import asyncio
import os
import sys
from datetime import datetime

import asyncpg
import yaml
from dotenv import load_dotenv

load_dotenv()

from research import agent as research_agent
from research import db as rdb
from research import charts as rcharts


# ── Pool helpers ──────────────────────────────────────────────────────────────

async def _make_pool(dsn: str, min_size=2, max_size=8) -> asyncpg.Pool:
    return await asyncpg.create_pool(dsn=dsn, min_size=min_size, max_size=max_size,
                                     command_timeout=60)


# ── Commands ──────────────────────────────────────────────────────────────────

async def cmd_run(args):
    with open(args.config) as f:
        config = yaml.safe_load(f)

    run_name = args.name or config.get("name") or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    question = config.get("question", "")
    model    = config.get("model", "claude-sonnet-4-6")
    max_calls = config.get("max_tool_calls", 60)

    main_dsn = os.environ.get("DATABASE_URL")
    oi_dsn   = os.environ.get("OI_DATABASE_URL")

    if not main_dsn:
        sys.exit("DATABASE_URL not set in .env")
    if not oi_dsn:
        sys.exit("OI_DATABASE_URL not set in .env")

    print(f"Connecting to databases…")
    main_pool = await _make_pool(main_dsn)
    oi_pool   = await _make_pool(oi_dsn)

    async with main_pool.acquire() as conn:
        run_id = await rdb.create_run(conn, run_name, question, config)

    print(f"Run ID: {run_id}")
    print(f"Name:   {run_name}")
    print(f"Model:  {model} | max_tool_calls: {max_calls}")
    print(f"Q: {question[:120]}")
    print("-" * 60)

    try:
        summary = await research_agent.run_agent(
            main_pool=main_pool,
            oi_pool=oi_pool,
            run_id=run_id,
            question=question,
            config=config,
            model=model,
            max_tool_calls=max_calls,
            log=print,
        )

        # Generate correlation heatmap if multiple tickers / combos
        tickers = config.get("tickers") or []
        x_cols  = config.get("x_columns") or []
        y_cols  = config.get("y_columns") or []
        if len(tickers) > 1 and x_cols and y_cols:
            async with main_pool.acquire() as conn:
                all_results = await rdb.load_results(conn, run_id)
            corr_results = [
                dict(r, **r.get("result", {}))
                for r in all_results
                if r.get("analysis_type") == "correlation"
            ]
            # Load result JSONB into flat dict for the heatmap function
            flat = []
            for r in all_results:
                if r.get("analysis_type") == "correlation":
                    rd = r.get("result") or {}
                    import json
                    if isinstance(rd, str):
                        rd = json.loads(rd)
                    flat.append({**rd, "ticker": r.get("ticker"), "x_col": r.get("x_col"), "y_col": r.get("y_col")})
            if flat:
                png = rcharts.correlation_heatmap(flat, tickers, x_cols, y_cols)
                if png:
                    async with main_pool.acquire() as conn:
                        await rdb.save_chart(conn, run_id, "correlation_heatmap",
                                             "Pearson Correlation Heatmap", png)

        async with main_pool.acquire() as conn:
            await rdb.update_run(conn, run_id, status="complete", ai_summary=summary)

        print("-" * 60)
        print("FINDINGS SUMMARY:")
        print(summary)
        print("-" * 60)
        print(f"Run complete. ID: {run_id}")
        print(f"Export PDF: python export_report.py --run \"{run_name}\"")

    except Exception as exc:
        async with main_pool.acquire() as conn:
            await rdb.update_run(conn, run_id, status="error", error_msg=str(exc))
        print(f"ERROR: {exc}")
        raise
    finally:
        await main_pool.close()
        await oi_pool.close()


async def cmd_list(args):
    main_dsn = os.environ.get("DATABASE_URL")
    if not main_dsn:
        sys.exit("DATABASE_URL not set in .env")
    pool = await _make_pool(main_dsn)
    async with pool.acquire() as conn:
        runs = await rdb.list_runs(conn, limit=20)
    await pool.close()

    if not runs:
        print("No research runs found.")
        return
    print(f"{'ID[:8]':<10} {'Status':<10} {'Name':<35} {'Created':<20}")
    print("-" * 80)
    for r in runs:
        print(f"{str(r['id'])[:8]:<10} {r['status']:<10} {r['name'][:34]:<35} {str(r['created_at'])[:16]}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="SPX Research Runner")
    sub = parser.add_subparsers(dest="cmd")

    run_p = sub.add_parser("run", help="Start a research run")
    run_p.add_argument("--config", required=True, help="Path to YAML config file")
    run_p.add_argument("--name", help="Override run name from config")

    sub.add_parser("list", help="List recent research runs")

    # Allow positional shorthand: python run_research.py --config foo.yaml
    parser.add_argument("--config", help="Config file (shorthand for 'run' command)")
    parser.add_argument("--name",   help="Override run name")
    parser.add_argument("--list",   action="store_true", help="List recent runs")

    args = parser.parse_args()

    if args.cmd == "list" or getattr(args, "list", False):
        asyncio.run(cmd_list(args))
    elif args.cmd == "run" or getattr(args, "config", None):
        asyncio.run(cmd_run(args))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
