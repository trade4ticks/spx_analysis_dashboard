"""How long does a parameter-sweep load take on THIS machine and database?

STRICTLY READ-ONLY. The connection is opened with
default_transaction_read_only; the only statements are the market join's
SELECTs on index_ohlc. Nothing is saved and no service is touched.

The batch endpoint has to finish inside the Cloudflare tunnel's 100 s:
un-gzip every part, parse every file, combine them into one sweep, join
market data, build the payload, encode and gzip it. This runs exactly that
code (app.routers.oo_backtest.analyze_batch + json_gzip_response) over
synthetic Option Omega CSVs and prints each stage.

  --files N --trades M   the batch (default 75 x 3,200 = 240,000 trades)
  --entries shared       every variant enters on the same days at the same
                         time -- a sweep of exit parameters (default)
  --entries random       every variant its own entry days and times: the
                         worst case for the entry-bar lookup
  --format mcp           Option Omega MCP exports -- camelCase header,
                         param_delta / param_pt columns, ~2% isIgnored: the
                         files a real sweep arrives as (default)
  --format site          the website CSV export, variants in the filenames
  --budget S             the server-side total this load must stay under
                         (default 20 s); over it, the script ends "OVER" and
                         exits 1, so a regression is a result, not a mood

DATA GAPS ARE NOT TIMINGS. The market join logs what index_ohlc is missing
(e.g. SPX on 2026-04-08, VIX for most of May-June 2026). Those warnings are
collected and printed once, in their own block, AFTER the timings -- they
are an ingestion matter, and the verdict line below them is about speed.

    sudo -u gates .venv/bin/python scripts/measure_sweep_parse.py
    sudo -u gates .venv/bin/python scripts/measure_sweep_parse.py --entries random
"""
from __future__ import annotations

import argparse
import asyncio
import gzip
import io
import os
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

HEADER = ["Date Opened", "Time Opened", "Opening Price", "Legs", "Premium", "Closing Price",
          "Date Closed", "Time Closed", "Avg. Closing Cost", "Reason For Close", "P/L", "P/L %",
          "No. of Contracts", "Funds at Close", "Margin Req.", "Strategy", "Gap", "Movement"]


MCP_HEADER = ("param_delta,param_pt,dateOpened,timeOpened,dateClosed,timeClosed,daysInTrade,underlying,legs,"
              "numberOfContracts,premiumPerContract,averageClosingCostPerContract,openingUnderlyingPrice,"
              "closingUnderlyingPrice,reasonClosed,profit,profitPercentage,fundsAtClose,buyingPowerRequired,"
              "openingFees,closingFees,shortLongRatio,closingShortLongRatio,gap,intradayMovement,"
              "highestProfitPercentage,highestLossPercentage,isIgnored")


def make_csv(seed: int, n: int, sessions: list, shared: bool, mcp: bool) -> bytes:
    entry = random.Random(0 if shared else seed)
    other = random.Random(seed)
    out = io.StringIO()
    out.write((MCP_HEADER if mcp else ",".join(HEADER)) + "\n")
    delta, pt = 5 + seed % 15, 25 + seed // 15 * 5
    for _ in range(n):
        i = entry.randrange(len(sessions) - 10)
        m = 9 * 60 + 35 + 5 * entry.randrange(0, 70)
        d0, d1 = sessions[i], sessions[i + other.randrange(0, 8)]
        pnl = round(other.gauss(30, 400), 2)
        if mcp:
            # ~2% isIgnored, as the MCP export keeps its wide-bid-ask trades.
            out.write(f"{delta},{pt},{d0},{m // 60:02d}:{m % 60:02d}:00,{d1},15:59:00,2,SPX,"
                      f"\"1 Mar 8 P 5000 STO | 1 Mar 8 P 4950 BTO\",1,12.5,0.0,4500.12,4510.3,Profit Target,"
                      f"{pnl},12.3,101200,5000,1.3,1.3,0.5,0.5,0.35,20,{round(other.uniform(0, 100), 1)},"
                      f"{round(-other.uniform(0, 200), 1)},{'true' if other.random() < 0.02 else 'false'}\n")
            continue
        out.write(f"{d0},{m // 60:02d}:{m % 60:02d}:00,4500.12,\"1 Mar 8 P 5000 STO | 1 Mar 8 P 4950 BTO\","
                  f"12.5,4510.3,{d1},15:59:00,0.0,Profit Target,{pnl},12.3,1,"
                  f"101200,5000,IC 45,0.35,20\n")
    return out.getvalue().encode()


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", type=int, default=75)
    ap.add_argument("--trades", type=int, default=3200)
    ap.add_argument("--entries", choices=("shared", "random"), default="shared")
    ap.add_argument("--format", choices=("mcp", "site"), default="mcp")
    ap.add_argument("--budget", type=float, default=20.0)
    args = ap.parse_args()

    # The join's data-gap warnings, held back from the timings (see above).
    import logging
    gaps: list[str] = []

    class _Gaps(logging.Handler):
        def emit(self, record):
            gaps.append(record.getMessage())
    for name in ("app.oo_backtest.market", "app.oo_backtest"):
        lg = logging.getLogger(name)
        lg.addHandler(_Gaps(level=logging.WARNING))
        lg.propagate = False

    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
    dsn = os.environ.get("DATABASE_URL")
    if not dsn:
        print("DATABASE_URL is not set")
        return 2
    import asyncpg

    from app.oo_backtest import market, market_calendar
    from app.routers import oo_backtest as R

    sessions = market_calendar.sessions("2017-01-03", "2026-09-30")

    t = time.monotonic()
    names = [f"sweep_dte{30 + i % 5 * 5}_pt{25 + (i // 5) % 5 * 25}_sl{100 + i // 25 * 100}.csv"
             for i in range(args.files)]
    if len(set(names)) < len(names):
        names = [f"sweep_v{i}.csv" for i in range(args.files)]
    mcp = args.format == "mcp"
    if mcp:     # the variants are in the param_* columns, not the names
        names = [f"mcp_export_{i}.csv" for i in range(args.files)]
    files = [(n, make_csv(i, args.trades, sessions, args.entries == "shared", mcp)) for i, n in enumerate(names)]
    parts = [(n, gzip.compress(c, 6)) for n, c in files]
    print(f"{args.files} {args.format} files x {args.trades:,} trades, entries {args.entries}: "
          f"{sum(len(c) for _, c in files) / 1e6:.1f} MB raw, {sum(len(z) for _, z in parts) / 1e6:.1f} MB "
          f"as the page uploads it (built in {time.monotonic() - t:.1f}s, not part of the request)")

    pool = await asyncpg.create_pool(dsn, min_size=1, max_size=2,
                                     server_settings={"default_transaction_read_only": "on"})
    try:
        t0 = time.monotonic()
        await market.get_daily(pool)
        t1 = time.monotonic()
        print(f"daily rollup, cold          {t1 - t0:6.2f} s  (once per process, then cached)")
        totals = []
        for run in ("first", "second"):
            t0 = time.monotonic()
            read = [(n, gzip.decompress(z)) for n, z in parts]
            t1 = time.monotonic()
            payload = await R.analyze_batch(read, pool)
            t2 = time.monotonic()
            resp = R.json_gzip_response(payload, None)
            t3 = time.monotonic()
            tm = payload["sweep"]["timing"]
            print(f"[{run} run]")
            print(f"  un-gzip parts             {t1 - t0:6.2f} s")
            print(f"  parse {args.files} files            {tm['parse_s']:6.2f} s")
            print(f"  join + payload            {tm['join_payload_s']:6.2f} s  "
                  f"(market joined: {payload['market'].get('joined')})")
            print(f"  encode + gzip             {t3 - t2:6.2f} s  ({len(resp.body) / 1e6:.1f} MB response)")
            totals.append(t3 - t0)
            print(f"  SERVER TOTAL              {t3 - t0:6.2f} s  of the tunnel's 100 s "
                  f"({payload['n']:,} trades, {len(payload['sweep']['variants'])} variants)")
    finally:
        await pool.close()
    print("Upload and download time over your connection come on top: the request is the size above, "
          "the response the size printed.")
    if gaps:
        print()
        print("Data gaps the market join reported (index_ohlc ingestion -- NOT part of the timing):")
        for g in dict.fromkeys(gaps):
            print("  - " + g.replace("oo-backtest: ", "")[:240])
    best = min(totals)
    print()
    ok = best <= args.budget
    print(f"{'PASS' if ok else 'OVER'}: best server total {best:.2f} s against a budget of {args.budget:.0f} s "
          f"({args.files} {args.format} files, entries {args.entries})")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
