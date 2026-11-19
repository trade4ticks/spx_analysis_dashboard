"""Gate: the Heatmap's IV z-score SQL (GET /api/heatmap/iv_zscore) against a
REAL Postgres, compared with a plain Python reference. Needs initdb.

A temporary cluster (check_oo_market_sql's Cluster) holding a fabricated
spx_surface with every case the query has to get right -- nothing touches
the real database:

  * z = (iv at the viewed snapshot - mean) / sample stdev of that node's
    15:45 IV over the window, equal to statistics.mean / statistics.stdev
  * history is the 15:45 snapshot ONLY (wild rows at 15:40 are ignored);
    the window is exactly 90 / 365 calendar days ending ON the viewed date
    (a wild row the day before it starts is ignored, the day it starts
    counts); a NULL iv day does not count toward n
  * BLANK z, each for its own reason: 19 days of history (n below 20), a
    constant IV (stdev 0), a jitter of 1e-9 (stdev > 0 but <= 1e-6), no
    history at all (n = 0), no IV at the viewed snapshot
  * a mid-session view: the current IV is that snapshot's, the history the
    same 15:45 one; an unknown lookback / bad date is a 400

--live (the VPS): the same questions against the EXISTING main database
(DATABASE_URL, or --dsn), READ-ONLY -- default_transaction_read_only=on, a
write proven refused first. On the latest 15:45 snapshot, both windows:
every node's n, mean, sd and z equal a plain Python recomputation from the
window's raw 15:45 rows (one SELECT, no SQL aggregates); every |z| <=
(n-1)/sqrt(n) -- EXACT, not a judgement: at 15:45 the viewed value is in
its own sample, and no point of a sample sits further than that from its
mean in sample-stdev units; window.days against the distinct 15:45 dates.
The BLANK cases on real data: the 10th 15:45 date of the table's history
(every node under 20 days -> every z blank, the line says "thin"), and any
node with 20+ days and a ~zero deviation (reported if the real data has
none -- the temporary cluster holds that case). A mid-session snapshot
reads its own IV against the same history.
  .venv/bin/python scripts/check_heatmap_sql.py --live
"""
from __future__ import annotations

import asyncio
import math
import os
import random
import statistics
import sys
import time as _time
from datetime import date, time, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from check_oo_market_sql import Cluster, EXIT_SKIPPED, find_bin  # noqa: E402

FAILS: list[str] = []
REL = 1e-9


def check(ok: bool, msg: str) -> None:
    print(("  ok    " if ok else "  FAIL  ") + msg)
    if not ok:
        FAILS.append(msg)


def close(a, b) -> bool:
    if a is None or b is None:
        return a is None and b is None
    return abs(a - b) <= REL * max(1.0, abs(a), abs(b))


def reference(hist: dict, current: dict, min_days: int, min_sd: float) -> dict:
    """Per node, from raw values: hist {(dte, pd): [iv at 15:45 per day in
    the window, None allowed]}, current {(dte, pd): iv at the viewed
    snapshot}."""
    out = {}
    for key, cur in current.items():
        xs = [x for x in hist.get(key, []) if x is not None]
        n = len(xs)
        mean = statistics.fmean(xs) if n else None
        sd = statistics.stdev(xs) if n >= 2 else None
        z = None
        if cur is not None and n >= min_days and sd is not None and sd > min_sd:
            z = (cur - mean) / sd
        out[key] = {"n": n, "mean": mean, "sd": sd, "z": z, "iv": cur}
    return out


def compare(body: dict, ref: dict, label: str) -> None:
    got = {(c["dte"], c["put_delta"]): c for c in body["current"]}
    check(set(got) == set(ref), f"{label}: one row per node of the viewed snapshot ({len(got)} vs {len(ref)})")
    bad = []
    for k, r in ref.items():
        g = got.get(k)
        if g is None:
            continue
        if not (g["n"] == r["n"] and close(g["mean"], r["mean"]) and close(g["v"], r["z"])
                and (r["n"] < 2 or close(g["sd"], r["sd"]))):
            bad.append((k, {"n": g["n"], "z": g["v"], "sd": g["sd"]}, {"n": r["n"], "z": r["z"], "sd": r["sd"]}))
    check(not bad, f"{label}: every node's n, mean, sd and z equal the Python reference "
                   f"({len(ref) - len(bad)}/{len(ref)}){'; first: ' + str(bad[:2]) if bad else ''}")


# ── the temporary cluster ────────────────────────────────────────────────────

SCHEMA = """
CREATE TABLE spx_surface (
    trade_date date NOT NULL, quote_time time NOT NULL,
    dte integer NOT NULL, put_delta integer NOT NULL,
    iv double precision, strike double precision,
    PRIMARY KEY (trade_date, quote_time, dte, put_delta));
"""
VIEW = date(2026, 3, 31)
T1545, T1540, T1000 = time(15, 45), time(15, 40), time(10, 0)


async def run(dsn: str) -> None:
    import asyncpg
    from fastapi import HTTPException

    from app.routers import heatmap as hm

    rng = random.Random(7)
    start90 = VIEW - timedelta(days=89)
    days = [VIEW - timedelta(days=i) for i in range(400)]
    sessions = sorted(d for d in days if d.weekday() < 5)
    rows = []          # (date, time, dte, pd, iv)
    hist90, hist1y = {}, {}

    def add(d, t, key, iv):
        rows.append((d, t, key[0], key[1], iv))
        if t == T1545:
            for h, first in ((hist90, start90), (hist1y, VIEW - timedelta(days=364))):
                if first <= d <= VIEW:
                    h.setdefault(key, []).append(iv)

    VARIED, THIN, FLAT, JITTER, GAPPY, NOHIST, NOIV = (7, 25), (7, 50), (30, 25), (30, 50), (60, 25), (90, 25), (7, 10)
    in90 = [d for d in sessions if d >= start90]
    for d in sessions:
        add(d, T1545, VARIED, 0.15 + rng.random() * 0.1)
        add(d, T1540, VARIED, 9.0)                     # not 15:45: must be ignored
        add(d, T1545, FLAT, 0.2)
        add(d, T1545, JITTER, 0.2 + (1e-9 if d.toordinal() % 2 else 0.0))
        add(d, T1545, GAPPY, None if d.day % 5 == 0 else 0.3 + rng.random() * 0.05)
        add(d, T1545, NOIV, 0.4 + rng.random() * 0.05)
    for d in in90[-19:]:                               # exactly 19 days in the 90d window
        add(d, T1545, THIN, 0.25 + rng.random() * 0.02)
    # The window's edges: the day before it starts is ignored; its first day counts.
    add(start90 - timedelta(days=1), T1545, (120, 25), 50.0)
    add(start90, T1545, (120, 25), 0.22)
    for d in in90[-30:]:
        add(d, T1545, (120, 25), 0.2 + rng.random() * 0.03)
    # The viewed snapshot: VIEW 15:45 rows exist already except for these.
    rows[:] = [r for r in rows if not (r[0] == VIEW and r[1] == T1545 and (r[2], r[3]) == NOIV)]
    for h in (hist90, hist1y):           # NOIV's viewed-day value was removed from the table
        if NOIV in h:
            h[NOIV] = h[NOIV][:-1]
    add(VIEW, T1545, NOIV, None)
    add(VIEW, T1545, NOHIST, 0.33)
    # A mid-session snapshot on the viewed date.
    cur10 = {}
    for key in (VARIED, THIN, FLAT, JITTER, GAPPY, NOHIST, NOIV, (120, 25)):
        v = 0.5 + rng.random() * 0.1
        rows.append((VIEW, T1000, key[0], key[1], v))
        cur10[key] = v

    pool = await asyncpg.create_pool(dsn, min_size=1, max_size=2)
    try:
        async with pool.acquire() as conn:
            await conn.execute(SCHEMA)
            await conn.copy_records_to_table("spx_surface", records=[(*r, None) for r in rows],
                                             columns=["trade_date", "quote_time", "dte", "put_delta", "iv", "strike"])
        cur = {(r[2], r[3]): r[4] for r in rows if r[0] == VIEW and r[1] == T1545}

        print("the z-score SQL vs a Python reference")
        for lb, h in (("90d", hist90), ("1y", hist1y)):
            body = await hm.heatmap_iv_zscore(date=VIEW.isoformat(), time="15:45", lookback=lb, pool=pool)
            ref = reference(h, cur, hm.ZSCORE_MIN_DAYS, hm.ZSCORE_MIN_STDEV)
            compare(body, ref, lb)
            got = {(c["dte"], c["put_delta"]): c for c in body["current"]}
            if lb == "90d":
                check(got[VARIED]["v"] is not None and got[VARIED]["n"] == len(in90),
                      f"a node with every day: z reported, n = the window's {len(in90)} sessions (wild 15:40 rows ignored)")
                check(got[(120, 25)]["n"] == 31 and abs(got[(120, 25)]["mean"] - 50) > 40,
                      f"the window starts exactly 89 days before the viewed date: that day counts, the day before "
                      f"does not (n {got[(120, 25)]['n']})")
                check(got[THIN]["n"] == 19 and got[THIN]["v"] is None,
                      f"19 days of history: z blank, n kept ({got[THIN]['n']})")
                check(body["window"]["days"] == len(in90), f"window.days = the sessions with a 15:45 row ({body['window']})")
            check(got[FLAT]["v"] is None and got[FLAT]["n"] >= 20 and (got[FLAT]["sd"] or 0) <= 1e-6,
                  f"{lb}: a constant IV: z blank (sd {got[FLAT]['sd']})")
            check(got[JITTER]["v"] is None and 0 < (got[JITTER]["sd"] or 0) <= 1e-6,
                  f"{lb}: a 1e-9 jitter -- sd above 0 but under 1e-6: z blank, not enormous (sd {got[JITTER]['sd']:.2e})")
            check(got[GAPPY]["n"] == sum(x is not None for x in h[GAPPY]) and got[GAPPY]["v"] is not None,
                  f"{lb}: NULL iv days do not count toward n ({got[GAPPY]['n']})")
            check(got[NOHIST]["n"] == 0 and got[NOHIST]["v"] is None and got[NOHIST]["mean"] is None,
                  f"{lb}: a node with no 15:45 history: n 0, z blank")
            check(got[NOIV]["v"] is None and got[NOIV]["n"] >= 20 and got[NOIV]["iv"] is None,
                  f"{lb}: no IV at the viewed snapshot: z blank, its history still counted (n {got[NOIV]['n']})")

        print("a mid-session snapshot")
        body = await hm.heatmap_iv_zscore(date=VIEW.isoformat(), time="10:00", lookback="90d", pool=pool)
        compare(body, reference(hist90, cur10, hm.ZSCORE_MIN_DAYS, hm.ZSCORE_MIN_STDEV), "10:00 vs 90d")
        got = {(c["dte"], c["put_delta"]): c for c in body["current"]}
        check(got[VARIED]["iv"] == cur10[VARIED] and got[VARIED]["n"] == len(in90),
              "it reads the 10:00 IV against the same 15:45 history")

        for kw in ({"lookback": "6m"}, {"date": "2026-13-01"}):
            args = {"date": VIEW.isoformat(), "time": "15:45", "lookback": "90d", **kw}
            try:
                await hm.heatmap_iv_zscore(pool=pool, **args)
                check(False, f"{kw} was accepted")
            except HTTPException as e:
                check(e.status_code == 400, f"{kw} is a 400 ({e.status_code})")
    finally:
        await pool.close()


# ── --live: read-only against the real main database ────────────────────────

async def run_live(dsn: str) -> None:
    import asyncpg

    from app.routers import heatmap as hm

    pool = await asyncpg.create_pool(
        dsn, min_size=1, max_size=2,
        server_settings={"default_transaction_read_only": "on", "statement_timeout": "120000"})
    try:
        async with pool.acquire() as conn:
            ro = await conn.fetchval("SHOW default_transaction_read_only")
            check(ro == "on", f"every connection is read-only (default_transaction_read_only={ro})")
            if ro != "on":
                return
            try:
                await conn.execute("CREATE TEMP TABLE heatmap_gate_probe (x int)")
                wrote = True
            except asyncpg.exceptions.ReadOnlySQLTransactionError:
                wrote = False
            check(not wrote, "and a write is refused by the server")
            if wrote:
                return
            types = await conn.fetch(
                "SELECT column_name, data_type FROM information_schema.columns "
                "WHERE table_name = 'spx_surface' AND column_name IN ('trade_date','quote_time','dte','put_delta','iv') "
                "ORDER BY ordinal_position")
            print("  info  spx_surface: " + ", ".join(f"{r['column_name']} {r['data_type']}" for r in types))
            last = await conn.fetchval("SELECT max(trade_date) FROM spx_surface WHERE quote_time = '15:45'")
            early = await conn.fetch("SELECT trade_date FROM spx_surface WHERE quote_time = '15:45' "
                                     "GROUP BY trade_date ORDER BY trade_date LIMIT 10")
            mid = await conn.fetchval("SELECT min(quote_time) FROM spx_surface WHERE trade_date = $1", last)
        check(last is not None, f"the latest 15:45 snapshot: {last}")
        if last is None:
            return

        async def raw(first, end, t):
            async with pool.acquire() as conn:
                hist = await conn.fetch(
                    "SELECT trade_date, dte, put_delta, iv::float8 AS iv FROM spx_surface "
                    "WHERE trade_date BETWEEN $1 AND $2 AND quote_time = '15:45'", first, end)
                cur = await conn.fetch(
                    "SELECT dte, put_delta, iv::float8 AS iv FROM spx_surface "
                    "WHERE trade_date = $1 AND quote_time = $2", end, t)
            h = {}
            for r in hist:
                h.setdefault((r["dte"], r["put_delta"]), []).append(r["iv"])
            return h, {(r["dte"], r["put_delta"]): r["iv"] for r in cur}, len({r["trade_date"] for r in hist})

        async def endpoint(d, t, lb):
            t0 = _time.perf_counter()
            body = await hm.heatmap_iv_zscore(date=d.isoformat(), time=t.strftime("%H:%M"), lookback=lb, pool=pool)
            return body, _time.perf_counter() - t0

        flat_seen = 0
        print(f"the latest snapshot, {last} 15:45")
        for lb, ndays in hm.ZSCORE_LOOKBACK_DAYS.items():
            body, secs = await endpoint(last, time(15, 45), lb)
            first = last - timedelta(days=ndays - 1)
            h, cur, distinct = await raw(first, last, time(15, 45))
            ref = reference(h, cur, hm.ZSCORE_MIN_DAYS, hm.ZSCORE_MIN_STDEV)
            w = body["window"]
            print(f"  info  {lb}: {len(body['current'])} nodes, window {w['start']}..{w['end']}, {w['days']} days, "
                  f"{secs:.2f}s")
            check(w["start"] == first.isoformat() and w["end"] == last.isoformat(),
                  f"{lb}: the window is {ndays} calendar days ending on the viewed date")
            compare(body, ref, lb)
            check(w["days"] <= distinct and w["days"] >= 0.9 * distinct,
                  f"{lb}: window.days ({w['days']}) is the fullest node's count, against {distinct} dates with a "
                  f"15:45 row")
            zs = [c for c in body["current"] if c["v"] is not None]
            over = [c for c in zs if abs(c["v"]) > (c["n"] - 1) / math.sqrt(c["n"]) + 1e-9]
            check(not over, f"{lb}: every |z| <= (n-1)/sqrt(n) -- the viewed value is in its own sample "
                            f"({len(zs)} z values, largest |z| {max((abs(c['v']) for c in zs), default=0):.2f})"
                            f"{'; over: ' + str(over[:2]) if over else ''}")
            check(len(zs) >= 0.5 * len(body["current"]),
                  f"{lb}: most nodes have a z ({len(zs)} of {len(body['current'])})")
            share3 = sum(abs(c["v"]) > 3 for c in zs) / max(1, len(zs))
            print(f"  info  {lb}: {share3:.1%} of z beyond +/-3 (drawn clamped), "
                  f"blank: {len(body['current']) - len(zs)}")
            flat = [c for c in body["current"] if c["v"] is None and c["iv"] is not None
                    and c["n"] >= hm.ZSCORE_MIN_DAYS]
            for c in flat:
                r = ref[(c["dte"], c["put_delta"])]
                check(r["sd"] is not None and r["sd"] <= hm.ZSCORE_MIN_STDEV,
                      f"{lb}: node {c['dte']}/{c['put_delta']} is blank with {c['n']} days because its deviation "
                      f"is ~0 ({r['sd']})")
            flat_seen += len(flat)
        if not flat_seen:
            print("  info  no node with 20+ days and a ~zero deviation in the real data here -- that case is held "
                  "by the temporary-cluster mode, not by this run")

        if mid is not None and mid != time(15, 45):
            print(f"a mid-session snapshot, {last} {mid}")
            body, _ = await endpoint(last, mid, "90d")
            h, cur, _ = await raw(last - timedelta(days=89), last, mid)
            compare(body, reference(h, cur, hm.ZSCORE_MIN_DAYS, hm.ZSCORE_MIN_STDEV), f"{mid} vs 90d")

        if len(early) == 10:
            d10 = early[9]["trade_date"]
            print(f"the table's 10th 15:45 date, {d10}: too little history for any z")
            body, _ = await endpoint(d10, time(15, 45), "90d")
            h, cur, distinct = await raw(d10 - timedelta(days=89), d10, time(15, 45))
            compare(body, reference(h, cur, hm.ZSCORE_MIN_DAYS, hm.ZSCORE_MIN_STDEV), "10th date")
            check(body["current"] and all(c["v"] is None and c["n"] <= 10 for c in body["current"]),
                  f"every node is blank with at most 10 days ({len(body['current'])} nodes, window.days "
                  f"{body['window']['days']})")
            check(body["window"]["days"] < hm.ZSCORE_MIN_DAYS,
                  "and the window says it is under 20 days (the page prints 'thin')")
    finally:
        await pool.close()


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--live", action="store_true",
                    help="read-only against the existing main database instead of a temporary cluster "
                         "(DATABASE_URL from the environment or .env, or --dsn)")
    ap.add_argument("--dsn", help="with --live: the database to read (default DATABASE_URL)")
    a = ap.parse_args()
    if a.live:
        dsn = a.dsn
        if not dsn:
            try:
                from dotenv import load_dotenv
                load_dotenv(ROOT / ".env")
            except ImportError:
                pass
            dsn = os.getenv("DATABASE_URL")
        if not dsn:
            print("FAIL: --live needs DATABASE_URL (environment or .env) or --dsn")
            return 1
        asyncio.run(run_live(dsn))
        print()
        if FAILS:
            print(f"FAIL: {len(FAILS)} heatmap z-score check(s) failed (live, read-only)")
            return 1
        print("PASS: heatmap IV z-score on the live main database (read-only)")
        return 0
    if os.name == "posix" and os.geteuid() == 0:
        print("SKIP: running as root — initdb refuses to create a cluster as root; use --live for a read-only run "
              "against the existing database")
        return EXIT_SKIPPED
    bindir = find_bin()
    if bindir is None:
        print("SKIP: no Postgres server binaries (initdb, pg_ctl) found — set PG_BIN to run this check")
        return EXIT_SKIPPED
    try:
        import asyncpg  # noqa: F401
    except ImportError as exc:
        print(f"SKIP: {exc.name} not installed")
        return EXIT_SKIPPED
    with Cluster(bindir) as c:
        asyncio.run(run(c.dsn))
    print()
    if FAILS:
        print(f"FAIL: {len(FAILS)} heatmap z-score SQL check(s) failed")
        return 1
    print("PASS: heatmap IV z-score SQL on a real Postgres")
    return 0


if __name__ == "__main__":
    sys.exit(main())
