"""How long does a surface-metric ranking take on the real table?

STRICTLY READ-ONLY. SELECTs and EXPLAIN ANALYZE of SELECTs against
DATABASE_URL; it writes nothing and touches no service.

The ranking reads ~450 DOUBLE PRECISION columns (545 MB, 110k rows) at each
trade's entry bar. Row count is small, but the rows are wide, so the numbers
this prints decide whether the page can rank on demand as designed or needs
fewer columns per request / a covering index.

It times each stage the endpoint runs, twice (cold, then warm cache):

  catalog     surface_metrics_catalog + information_schema
  coverage    first non-null date per metric -- the index-walk form the app
              uses; add --scan to also time the one-pass full-scan form
  join        the entry-bar LATERAL join for every ranked column, over the
              trades' distinct (date, time) pairs; EXPLAIN (ANALYZE, BUFFERS)
              once, for the plan's own timing and buffer reads
  stats       Pearson + Spearman + BH over the joined values (in process)
  payload     JSON size of a /surface/rank response and a /surface/values one

Trades: --log an Option Omega CSV or Mesosim JSON (parsed by the app's own
parser, so the entry dates and times are the real ones), else a synthetic
2,097 trades on weekdays from 2018-01-02 at random times 09:30-15:55 -- the
shape of the OO log, including trades before any metric coverage.

    .venv/bin/python scripts/measure_surface_rank.py
    .venv/bin/python scripts/measure_surface_rank.py --log /path/to/trades.csv --scan
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def synthetic_trades(n: int = 2097) -> list[tuple]:
    rng = random.Random(1)
    days, d = [], date(2018, 1, 2)
    while d <= date.today():
        if d.weekday() < 5:
            days.append(d)
        d += timedelta(days=1)
    out = []
    for _ in range(n):
        m = rng.randrange(9 * 60 + 30, 15 * 60 + 56)
        out.append((rng.choice(days), f"{m // 60:02d}:{m % 60:02d}:00", round(rng.gauss(20, 400), 2)))
    return sorted(out)


def log_trades(path: str) -> list[tuple]:
    from app.routers.oo_backtest import _parse_df
    df = _parse_df(Path(path).read_bytes(), Path(path).name)
    times = df["time_opened"] if "time_opened" in df.columns else [None] * len(df)
    return [(d.date(), t, float(p)) for d, t, p in zip(df["date_opened"], times, df["pnl"])]


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", help="trade log to take entry dates/times and P/L from")
    ap.add_argument("--scan", action="store_true", help="also time the full-scan coverage form (reads the whole table)")
    args = ap.parse_args()

    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
    dsn = os.environ.get("DATABASE_URL")
    if not dsn:
        print("DATABASE_URL is not set")
        return 2
    import asyncpg
    from app.oo_backtest import surface, surface_stats

    raw = log_trades(args.log) if args.log else synthetic_trades()
    trades, prep = surface.parse_trades([[d.isoformat(), t, p] for d, t, p in raw], with_pnl=True)
    print(f"trades: {len(trades)} ({'log ' + args.log if args.log else 'synthetic, from 2018-01-02'}); parse {prep}")

    # Every session read-only at the server: a write here is refused, not trusted to be absent.
    pool = await asyncpg.create_pool(dsn, min_size=1, max_size=2,
                                     server_settings={"default_transaction_read_only": "on"})
    timings: dict[str, list[float]] = {}

    def lap(name, t0):
        timings.setdefault(name, []).append(round(time.monotonic() - t0, 3))

    try:
        async with pool.acquire() as conn:
            for rnd in (1, 2):
                t0 = time.monotonic()
                catalog = [dict(r) for r in await conn.fetch(surface.CATALOG_SQL)]
                table_cols = {r["column_name"]: r["data_type"]
                              for r in await conn.fetch(surface.TABLE_COLUMNS_SQL, surface.CORE)}
                ranked, rep = surface.metric_set(catalog, table_cols)
                cols = [r["column_name"] for r in ranked]
                lap("catalog", t0)

                t0 = time.monotonic()
                cov = dict(await conn.fetchrow(surface.coverage_sql(cols)))
                lap("coverage (index walk)", t0)
                if args.scan:
                    t0 = time.monotonic()
                    cov2 = dict(await conn.fetchrow(surface.coverage_scan_sql(cols)))
                    lap("coverage (full scan)", t0)
                    if rnd == 1 and cov2 != cov:
                        diff = {c: (cov[c], cov2[c]) for c in cov if cov[c] != cov2[c]}
                        print(f"  !! coverage forms disagree on {len(diff)} columns, e.g. {list(diff.items())[:3]}")
        print(f"metric set: {rep['ranked']} ranked of {rep['catalog_rows']}; excluded {rep['excluded']}; "
              f"missing {rep['missing_from_table']}; wrong type {rep['wrong_type']}; uncatalogued {rep['uncatalogued']}")
        firsts = sorted({v for v in cov.values() if v})
        print(f"coverage first dates seen: {[d.isoformat() for d in firsts]}"
              f"{'; columns with no data: ' + str([c for c, v in cov.items() if v is None]) if None in cov.values() else ''}")

        for rnd in (1, 2):
            t0 = time.monotonic()
            rows, jrep = await surface.entry_values(pool, trades, cols)
            lap("join (all ranked columns)", t0)
            t0 = time.monotonic()
            bar_keys = [(t[0], r["bar_time"]) for t, r in zip(trades, rows)]
            result = surface_stats.rank(ranked, rows, [t[2] for t in trades], bar_keys)
            lap("stats (pearson+spearman+BH)", t0)
        print(f"join: {jrep['distinct_entries']} distinct entries, {jrep['no_bar']} trades with no bar")

        keys = sorted({(t[0], t[1]) for t in trades if t[0] and t[1]})
        async with pool.acquire() as conn:
            plan = await conn.fetch("EXPLAIN (ANALYZE, BUFFERS) " + surface.entry_sql(cols),
                                    [k[0] for k in keys], [k[1] for k in keys])
        lines = [r[0] for r in plan]
        print("EXPLAIN (ANALYZE, BUFFERS), entry join -- first lines and totals:")
        for ln in lines[:8] + [ln for ln in lines if ln.startswith(("Planning", "Execution"))]:
            print("   ", ln)

        ns = sorted(r["n"] for r in result)
        rank_json = json.dumps({"rows": result}, default=str)
        one = cols[0]
        values_json = json.dumps({"values": [r[one] for r in rows]}, default=str)
        print(f"rank n per metric: min {ns[0]}, median {ns[len(ns) // 2]}, max {ns[-1]} (of {len(trades)} trades)")
        print(f"payload: rank {len(rank_json) / 1024:.0f} KB; values for one metric {len(values_json) / 1024:.0f} KB; "
              f"request body ~{len(json.dumps([[d.isoformat(), t, p] for d, t, p in raw])) / 1024:.0f} KB")
    finally:
        await pool.close()

    print("\ntimings (s), round 1 (cold) / round 2 (warm):")
    for name, v in timings.items():
        print(f"  {name:<30} {' / '.join(f'{x:.3f}' for x in v)}")
    print(f"\nat {datetime.now().isoformat(timespec='seconds')}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
