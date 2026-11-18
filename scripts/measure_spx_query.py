"""Where does the OI / volume / price ranking's query time go -- and the
surface (IV) join's, on the same entries, for comparison?

STRICTLY READ-ONLY: SELECTs, EXPLAIN ANALYZE of SELECTs and COPY ... TO a
buffer in this process, every session opened with
default_transaction_read_only=on. Writes nothing, touches no service.
Measures; changes nothing in the app.

/spx-metrics/rank's `query_s` is spx_metrics.bars_matrix: one statement per
source table (four round trips), each returning every served column of that
table at every distinct entry, then a Python loop filling a numpy matrix.
That wall time is several things at once. This splits it, per source and
for the surface join (surface.bars_matrix, 452 columns, main database):

  ping        SELECT 1 -- the round trip itself
  server      EXPLAIN (ANALYZE, BUFFERS): the plan's own Execution Time and
              buffer hits/reads -- the work inside Postgres only
  keys only   the same join returning only the key columns: join cost plus
              a small transfer, no metric values
  copy        COPY (the full statement) TO a buffer, binary: server + the
              bytes on the wire, with NO per-value Python decoding
  fetch       conn.fetch of the full statement: server + wire + asyncpg
              turning every value into a Python object
  fill        the bars_matrix loop over those rows into numpy (Python only)
  endpoint    bars_matrix itself, as the route calls it, end to end

so  fetch - copy  ~ decoding,  copy - server ~ transfer,  and
server vs keys only says whether the join or the width costs the time.

For a DAILY table it also times the plain date lookup -- the distinct DATES,
`WHERE trade_date = ANY($1)` -- against the shipped shape (every distinct
(date, time) pair joined), to answer whether the daily tables do more than a
date lookup needs.

Each timing runs 3 times; printed as first / best of the other two (the
first may be cold).

Entries: by default 1,500 random bars of spx_oi_5min (the shape of
check_spx_metrics_sql --live's run, which printed query 1.169 s); --log a
trade log (Option Omega CSV / MesoSim JSON) for real entries; --n to resize.

    .venv/bin/python scripts/measure_spx_query.py
    .venv/bin/python scripts/measure_spx_query.py --log /path/to/trades.csv
"""
from __future__ import annotations

import argparse
import asyncio
import io
import os
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

RUNS = 3


def fmt(ts: list[float]) -> str:
    rest = ts[1:] or ts
    return f"{ts[0] * 1000:8.0f} ms first, {min(rest) * 1000:8.0f} ms best"


async def timed(fn, runs=RUNS):
    out, last = [], None
    for _ in range(runs):
        t0 = time.perf_counter()
        last = await fn()
        out.append(time.perf_counter() - t0)
    return out, last


def plan_summary(lines: list[str]) -> str:
    exe = next((ln for ln in lines if ln.startswith("Execution Time")), "?")
    plan = next((ln for ln in lines if ln.startswith("Planning Time")), "?")
    bufs = [ln.strip() for ln in lines if "Buffers:" in ln][:1]
    return f"{exe}; {plan}; top {bufs[0] if bufs else 'no buffer line'}"


def fill(rows, ncols: int, nkeys: int, pos: dict, first: int):
    """bars_matrix's per-row work, on already-fetched rows."""
    import numpy as np
    X = np.full((nkeys + 1, ncols), np.nan)
    idx, vals = [], []
    for r in rows:
        i = pos[(r["entry_date"], r["entry_time"])]
        if first == 4 and not r["present"]:
            continue
        if first == 3 and r["bar_time"] is None:
            continue
        idx.append(i)
        vals.append(tuple(r)[first:])
    if idx:
        v = np.array(vals, dtype=object)
        v[v == None] = np.nan                       # noqa: E711
        X[idx] = v.astype(float)
    return X


async def measure(conn, label: str, sql: str, keys_sql: str, args: tuple, ncols: int, nkeys: int, pos: dict,
                  first: int) -> dict:
    print(f"\n{label}")
    pings, _ = await timed(lambda: conn.fetchval("SELECT 1"), runs=5)
    print(f"  ping        {statistics.median(pings) * 1000:8.1f} ms median of 5")
    plan = [r[0] for r in await conn.fetch("EXPLAIN (ANALYZE, BUFFERS) " + sql, *args)]
    print(f"  server      {plan_summary(plan)}")
    t_keys, krows = await timed(lambda: conn.fetch(keys_sql, *args))
    print(f"  keys only   {fmt(t_keys)}   ({len(krows):,} rows)")

    async def copy():
        buf = io.BytesIO()
        await conn.copy_from_query(sql, *args, output=buf, format="binary")
        return buf.tell()
    t_copy, nbytes = await timed(copy)
    print(f"  copy        {fmt(t_copy)}   ({nbytes / 1e6:,.1f} MB binary)")
    t_fetch, rows = await timed(lambda: conn.fetch(sql, *args))
    print(f"  fetch       {fmt(t_fetch)}   ({len(rows):,} rows x {ncols} metric columns)")
    t0 = time.perf_counter()
    fill(rows, ncols, nkeys, pos, first)
    t_fill = time.perf_counter() - t0
    print(f"  fill        {t_fill * 1000:8.0f} ms")
    best = lambda ts: min(ts[1:] or ts)   # noqa: E731
    print(f"  => decoding ~{(best(t_fetch) - best(t_copy)) * 1000:,.0f} ms, transfer+server ~{best(t_copy) * 1000:,.0f} ms, "
          f"join alone ~{best(t_keys) * 1000:,.0f} ms")
    return {"fetch": best(t_fetch), "copy": best(t_copy), "keys": best(t_keys), "fill": t_fill, "plan": plan}


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", help="trade log for real entry (date, time) pairs")
    ap.add_argument("--n", type=int, default=1500, help="random spx_oi_5min bars when no --log (default 1500)")
    ap.add_argument("--plans", action="store_true", help="print each full EXPLAIN plan")
    a = ap.parse_args()
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
    oi_dsn, main_dsn = os.environ.get("OI_DATABASE_URL"), os.environ.get("DATABASE_URL")
    if not oi_dsn:
        print("OI_DATABASE_URL is not set")
        return 2
    import asyncpg

    from app.oo_backtest import spx_metrics as sm
    from app.oo_backtest import surface
    ro = {"default_transaction_read_only": "on", "statement_timeout": "300000"}
    oi = await asyncpg.create_pool(oi_dsn, min_size=1, max_size=2, server_settings=ro)
    main_pool = await asyncpg.create_pool(main_dsn, min_size=1, max_size=2, server_settings=ro) if main_dsn else None
    try:
        async with oi.acquire() as conn:
            if await conn.fetchval("SHOW default_transaction_read_only") != "on":
                print("refusing: the session is not read-only")
                return 2
            if a.log:
                from scripts.measure_surface_rank import log_trades
                parsed, _ = surface.parse_trades([[d.isoformat(), t, p] for d, t, p in log_trades(a.log)],
                                                 with_pnl=True, max_rows=None)
                keys = sorted({(d, t) for d, t, _ in parsed if d and t})
                src_label = f"log {a.log}"
            else:
                rows = await conn.fetch(f"SELECT trade_date, quote_time FROM spx_oi_5min ORDER BY random() LIMIT {a.n}")
                keys = sorted({(r["trade_date"], r["quote_time"]) for r in rows})
                src_label = f"{a.n} random spx_oi_5min bars"
        dates = sorted({k[0] for k in keys})
        print(f"entries: {len(keys):,} distinct (date, time) on {len(dates):,} distinct dates ({src_label})")
        cat = await sm.get_catalog(oi)
        metrics = cat["metrics"]
        pos = {k: i for i, k in enumerate(keys, start=1)}
        args = ([k[0] for k in keys], [k[1] for k in keys])

        print("\n== the endpoint's whole query step (spx_metrics.bars_matrix, four statements) ==")
        t_end, _ = await timed(lambda: sm.bars_matrix(oi, keys, metrics))
        print(f"  endpoint    {fmt(t_end)}   ({len(metrics)} metrics)")

        res = {}
        async with oi.acquire() as conn:
            for src, s in sm.SOURCES.items():
                cols = [m["metric"] for m in metrics if m["source"] == src]
                if not cols:
                    continue
                sql = sm.matrix_sql(s["table"], s["join"], cols)
                keys_sql = sm.matrix_sql(s["table"], s["join"], cols[:1])
                res[src] = await measure(conn, f"== {s['table']} ({s['join']} join, {len(cols)} columns) ==", sql,
                                         keys_sql, args, len(cols), len(keys), pos, 4)
                if a.plans:
                    print("\n".join("    " + ln for ln in res[src]["plan"]))
                if s["join"] == "date":
                    t = surface.quote_ident(s["table"])
                    dsql = (f"SELECT trade_date, {', '.join(sm._valid(c) for c in cols)} FROM {t} "
                            f"WHERE trade_date = ANY($1::date[])")
                    plan = [r[0] for r in await conn.fetch("EXPLAIN (ANALYZE, BUFFERS) " + dsql, dates)]
                    t_d, drows = await timed(lambda: conn.fetch(dsql, dates))
                    print(f"  date lookup {fmt(t_d)}   ({len(drows):,} rows for {len(dates):,} distinct dates; "
                          f"{plan_summary(plan)})")

        total = {k: sum(r[k] for r in res.values()) for k in ("fetch", "copy", "keys", "fill")}
        print(f"\nOI sum over the four: fetch {total['fetch'] * 1000:,.0f} ms = copy (server+wire) "
              f"{total['copy'] * 1000:,.0f} ms + decode ~{(total['fetch'] - total['copy']) * 1000:,.0f} ms; "
              f"fill {total['fill'] * 1000:,.0f} ms")

        if main_pool is None:
            print("\nDATABASE_URL is not set -- no surface (IV) comparison")
            return 0
        icat = await surface.get_catalog(main_pool)
        icols = [m["column_name"] for m in icat["metrics"]]
        print(f"\n== surface (IV) join on the same entries: surface_metrics_core, {len(icols)} columns ==")
        t_iv, _ = await timed(lambda: surface.bars_matrix(main_pool, keys, icols))
        print(f"  endpoint    {fmt(t_iv)}   (surface.bars_matrix, as /surface/rank calls it)")
        async with main_pool.acquire() as conn:
            r_iv = await measure(conn, "== surface_metrics_core, broken down ==", surface.entry_sql(icols),
                                 surface.entry_sql(icols[:1]), args, len(icols), len(keys), pos, 3)
            if a.plans:
                print("\n".join("    " + ln for ln in r_iv["plan"]))
        print("\nRead: 'server' is Postgres's own work; copy - server is the wire; fetch - copy is asyncpg making a "
              "Python float of every value; fill is the numpy loop. CLAUDE.md's '26 ms' for the IV join does not say "
              "which of these it was -- compare it with both the IV 'server' and 'endpoint' lines.")
    finally:
        await oi.close()
        if main_pool is not None:
            await main_pool.close()
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
