#!/usr/bin/env python
"""
Export a completed research run as a PDF.

Usage:
    python export_report.py --run "OI_study_Apr26"
    python export_report.py --run <uuid>
    python export_report.py --run "OI_study_Apr26" --out reports/my_report.pdf
"""
import argparse
import asyncio
import os
import sys

import asyncpg
from dotenv import load_dotenv

load_dotenv()

from research import db as rdb, export as rexport


async def main_async(args):
    main_dsn = os.environ.get("DATABASE_URL")
    if not main_dsn:
        sys.exit("DATABASE_URL not set in .env")

    pool = await asyncpg.create_pool(dsn=main_dsn, min_size=1, max_size=4, command_timeout=30)

    async with pool.acquire() as conn:
        run = await rdb.load_run(conn, args.run)
        if not run:
            await pool.close()
            sys.exit(f"Run not found: {args.run}")

        results = await rdb.load_results(conn, run["id"])
        charts  = await rdb.load_charts(conn, run["id"])

    await pool.close()

    status = run.get("status", "")
    if status != "complete":
        print(f"Warning: run status is '{status}' (not 'complete'). Exporting anyway.")

    out_path = args.out or f"reports/{run['name'].replace(' ', '_')}.pdf"
    os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else ".", exist_ok=True)

    print(f"Building PDF for run: {run['name']}")
    print(f"  Results: {len(results)} | Charts: {len(charts)}")
    print(f"  Output:  {out_path}")

    rexport.build_pdf(run, results, charts, out_path)
    print(f"Done: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Export research run to PDF")
    parser.add_argument("--run",  required=True, help="Run name or UUID")
    parser.add_argument("--out",  help="Output PDF path (default: reports/<name>.pdf)")
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
