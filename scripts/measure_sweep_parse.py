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


def make_csv(seed: int, n: int, sessions: list, shared: bool) -> bytes:
    entry = random.Random(0 if shared else seed)
    other = random.Random(seed)
    out = io.StringIO()
    out.write(",".join(HEADER) + "\n")
    for _ in range(n):
        i = entry.randrange(len(sessions) - 10)
        m = 9 * 60 + 35 + 5 * entry.randrange(0, 70)
        d0, d1 = sessions[i], sessions[i + other.randrange(0, 8)]
        out.write(f"{d0},{m // 60:02d}:{m % 60:02d}:00,4500.12,\"1 Mar 8 P 5000 STO | 1 Mar 8 P 4950 BTO\","
                  f"12.5,4510.3,{d1},15:59:00,0.0,Profit Target,{round(other.gauss(30, 400), 2)},12.3,1,"
                  f"101200,5000,IC 45,0.35,20\n")
    return out.getvalue().encode()


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", type=int, default=75)
    ap.add_argument("--trades", type=int, default=3200)
    ap.add_argument("--entries", choices=("shared", "random"), default="shared")
    args = ap.parse_args()

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
    files = [(n, make_csv(i, args.trades, sessions, args.entries == "shared")) for i, n in enumerate(names)]
    parts = [(n, gzip.compress(c, 6)) for n, c in files]
    print(f"{args.files} files x {args.trades:,} trades, entries {args.entries}: "
          f"{sum(len(c) for _, c in files) / 1e6:.1f} MB raw, {sum(len(z) for _, z in parts) / 1e6:.1f} MB "
          f"as the page uploads it (built in {time.monotonic() - t:.1f}s, not part of the request)")

    pool = await asyncpg.create_pool(dsn, min_size=1, max_size=2,
                                     server_settings={"default_transaction_read_only": "on"})
    try:
        t0 = time.monotonic()
        await market.get_daily(pool)
        t1 = time.monotonic()
        print(f"daily rollup, cold          {t1 - t0:6.2f} s  (once per process, then cached)")
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
            print(f"  SERVER TOTAL              {t3 - t0:6.2f} s  of the tunnel's 100 s "
                  f"({payload['n']:,} trades, {len(payload['sweep']['variants'])} variants)")
    finally:
        await pool.close()
    print("Upload and download time over your connection come on top: the request is the size above, "
          "the response the size printed.")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
