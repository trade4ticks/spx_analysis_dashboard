"""Gate: the OI / volume / price metrics (app/oo_backtest/spx_metrics.py,
/api/oo-backtest/spx-metrics/*) against a REAL Postgres. Needs initdb.

A temporary cluster (check_oo_market_sql's Cluster) with the four tables and
spx_metric_catalog fabricated to hold every case the code has to handle --
nothing touches the real database. What it pins:

  catalog   surfaced honoured PER ROW; a row whose table is not one of the
            four, whose column is missing or not numeric, is reported and
            not served; the same name in two tables (`spot`) stays two
            ids; a `cross` metric is in OI AND Volume; a family in no
            section is served and reported.
  coverage  the first date with a value per metric -- NaN is no value; a
            gap after it (the spot guard) does not move it.
  values    through the router's own handlers: a 5-minute table gives the
            last bar at or before the entry time on the entry DATE (never
            the previous session); a daily table the entry date's row; NaN
            and NULL read null; `present` tells "no bar / no row" from "a
            bar / row with no value"; an unknown or unsurfaced id is a 400,
            no pool a 503.
  cache     an edit to the catalog, and a new row in a table, each rebuild it.

--live (the VPS: root, no initdb): the same questions against the EXISTING
open_interest database (OI_DATABASE_URL, or --dsn), READ-ONLY --
default_transaction_read_only=on, a write proven refused first. It also
holds the catalog to the counts agreed on 2026-11-17 (155 served, 168
displayed: OI 70, Volume 53, Price 45; 25 not surfaced), checks every
metric's coverage date against the table (a value on it, none before), and
compares sampled real entries with a plain per-key lookup.
  .venv/bin/python scripts/check_spx_metrics_sql.py --live
"""
from __future__ import annotations

import asyncio
import os
import random
import sys
from datetime import date, time, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from check_oo_market_sql import Cluster, EXIT_SKIPPED, find_bin  # noqa: E402

FAILS: list[str] = []


def check(ok: bool, msg: str) -> None:
    print(("  ok    " if ok else "  FAIL  ") + msg)
    if not ok:
        FAILS.append(msg)


D1, D2, D3, D4, D5 = date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4), date(2024, 1, 5), date(2024, 1, 8)
SAT = date(2024, 1, 6)

SCHEMA = f"""
CREATE TABLE spx_metric_catalog (table_name text, metric text, description text, family text,
    surfaced boolean, built_at timestamptz, form text, base_column text, units text, formula text,
    PRIMARY KEY (table_name, metric));
CREATE TABLE spx_oi_daily (trade_date date PRIMARY KEY, pct_front double precision, n_strikes integer,
    lvl_hidden double precision, ratio_x numeric, note text);
CREATE TABLE spx_oi_5min (trade_date date, quote_time time, spot double precision,
    pct_atm double precision, z_atm double precision, PRIMARY KEY (trade_date, quote_time));
CREATE TABLE spx_price_daily (trade_date date PRIMARY KEY, rsi double precision);
CREATE TABLE spx_price_5min (trade_date date, quote_time time, spot double precision, ret double precision,
    odd real, PRIMARY KEY (trade_date, quote_time));

INSERT INTO spx_metric_catalog VALUES
  ('spx_oi_daily',   'pct_front',  'front share',   'oi',             true,  now(), 'level', NULL, 'pct',   ''),
  ('spx_oi_daily',   'n_strikes',  'strikes',       'oi',             true,  now(), 'level', NULL, 'int',   ''),
  ('spx_oi_daily',   'lvl_hidden', 'drifts',        'oi',             false, now(), 'level', NULL, 'pct',   ''),
  ('spx_oi_daily',   'ratio_x',    'vol / oi',      'cross',          true,  now(), 'level', NULL, 'ratio', ''),
  ('spx_oi_daily',   'note',       'text column',   'oi',             true,  now(), 'level', NULL, 'int',   ''),
  ('spx_oi_daily',   'ghost',      'not built',     'oi',             true,  now(), 'level', NULL, 'int',   ''),
  ('spx_other',      'x',          'other table',   'oi',             true,  now(), 'level', NULL, 'int',   ''),
  ('spx_oi_5min',    'spot',       'ingredient',    'ingredient',     false, now(), 'level', NULL, 'price', ''),
  ('spx_oi_5min',    'pct_atm',    'atm share',     'oi_spot',        true,  now(), 'level', NULL, 'pct',   ''),
  ('spx_oi_5min',    'z_atm',      'z of atm',      'oi_spot',        true,  now(), 'z', 'pct_atm', 'z_score', ''),
  ('spx_price_daily','rsi',        'rsi',           'price_momentum', true,  now(), 'level', NULL, 'index_0_100', ''),
  ('spx_price_5min', 'spot',       'ingredient',    'ingredient',     false, now(), 'level', NULL, 'price', ''),
  ('spx_price_5min', 'ret',        'return',        'price_trend',    true,  now(), 'level', NULL, 'ratio', ''),
  ('spx_price_5min', 'odd',        'no section',    'mystery',        true,  now(), 'level', NULL, 'ratio', '');

-- pct_front: NULL, then NaN, then values, then a NULL AFTER coverage (D5).
INSERT INTO spx_oi_daily VALUES
  ('{D1}', NULL,  10, 1, 0.5,  'a'),
  ('{D2}', 'NaN', 11, 1, 0.6,  'b'),
  ('{D3}', 0.2,   12, 1, 0.7,  'c'),
  ('{D4}', 0.3,   13, 1, 0.8,  'd'),
  ('{D5}', NULL,  14, 1, 0.9,  'e');
-- 5-minute: D1 all NULL (before coverage); D2 09:30 value, 09:35 NULL (the
-- spot guard), 09:40 value, 15:55 NaN. No rows on D3.
INSERT INTO spx_oi_5min VALUES
  ('{D1}', '09:30', 4700, NULL,  NULL),
  ('{D1}', '09:35', 4701, NULL,  NULL),
  ('{D2}', '09:30', 4710, 0.10,  NULL),
  ('{D2}', '09:35', NULL, NULL,  NULL),
  ('{D2}', '09:40', 4712, 0.12,  NULL),
  ('{D2}', '15:55', 4715, 'NaN', 1.5);
INSERT INTO spx_price_daily VALUES ('{D2}', 55.5), ('{D3}', 61.25);
INSERT INTO spx_price_5min VALUES ('{D2}', '09:30', 4710, 0.001, 1), ('{D2}', '10:00', 4720, -0.002, 2);
"""


async def run(dsn: str) -> None:
    import asyncpg
    from fastapi import HTTPException

    from app.oo_backtest import spx_metrics as sm
    from app.routers import oo_backtest as r

    pool = await asyncpg.create_pool(dsn, min_size=1, max_size=2)
    try:
        async with pool.acquire() as conn:
            await conn.execute(SCHEMA)
        print("catalog")
        cat = await r.spx_metrics_catalog(pool=pool)
        ids = sorted(m["id"] for m in cat["metrics"])
        check(ids == ["oi_5min.pct_atm", "oi_5min.z_atm", "oi_daily.n_strikes", "oi_daily.pct_front",
                      "oi_daily.ratio_x", "price_5min.odd", "price_5min.ret", "price_daily.rsi"],
              f"served: surfaced rows of the four tables with a numeric column ({ids})")
        rep = cat["report"]
        check(rep["not_surfaced"] == 3 and rep["unknown_table"] == ["spx_other.x"]
              and rep["missing_from_table"] == ["spx_oi_daily.ghost"] and rep["wrong_type"] == ["spx_oi_daily.note (text)"]
              and rep["no_section"] == ["spx_price_5min.odd"],
              f"each refusal reported by name; surfaced = false per row, not per family ({rep})")
        check(rep["served"] == 8 and rep["unique_names"] == 8 and rep["displayed"] == 9,
              f"8 served, 9 displayed: the cross metric counts in OI and Volume ({rep['served']}, {rep['displayed']})")
        by = {m["id"]: m for m in cat["metrics"]}
        check(by["oi_daily.ratio_x"]["sections"] == ["oi", "volume"] and by["oi_5min.pct_atm"]["sections"] == ["oi"]
              and by["price_daily.rsi"]["sections"] == ["price"] and by["price_5min.odd"]["sections"] == [],
              "sections from the family: cross in both OI and Volume, a family in none has none")
        check(by["oi_daily.pct_front"]["join"] == "date" and by["oi_5min.pct_atm"]["join"] == "bar"
              and by["oi_5min.z_atm"]["base_column"] == "pct_atm" and by["oi_daily.pct_front"]["units"] == "pct",
              "each metric carries its join, base_column and units")
        cov = {k: m["min_date"] for k, m in by.items()}
        check(cov["oi_daily.pct_front"] == D3.isoformat() and cov["oi_daily.n_strikes"] == D1.isoformat()
              and cov["oi_5min.pct_atm"] == D2.isoformat() and cov["oi_5min.z_atm"] == D2.isoformat()
              and cov["price_daily.rsi"] == D2.isoformat(),
              f"coverage: the first date with a value -- NaN is no value, a NULL after it does not move it ({cov})")

        print("values")

        async def values(mid, bars):
            return await r.spx_metrics_values(body={"id": mid, "bars": bars}, pool=pool)

        bars = [[D2.isoformat(), "09:30:00"], [D2.isoformat(), "09:37"], [D2.isoformat(), "09:29:00"],
                [D2.isoformat(), "16:00:00"], [D3.isoformat(), "10:00:00"], [D1.isoformat(), "09:40:00"],
                [D2.isoformat(), "09:30:00"], ["not a date", "10:00:00"], [D2.isoformat(), "09:42"]]
        v = await values("oi_5min.pct_atm", bars)
        check(v["values"] == [0.1, None, None, None, None, None, 0.1, None, 0.12],
              f"5-minute: the last bar at or before the entry time; NULL and NaN read null ({v['values']})")
        check(v["present"] == [True, True, False, True, False, True, True, False, True]
              and v["bar_times"] == ["09:30:00", "09:35:00", None, "15:55:00", None, "09:35:00", "09:30:00", None, "09:40:00"],
              "5-minute: present = a bar was found -- before 09:30 none, never the previous session (D3 has no "
              "bars), a 16:00 entry reads 15:55")
        check(v["report"]["not_present"] == 3 and v["report"]["present_no_value"] == 3 and v["report"]["with_value"] == 3
              and v["join"] == "bar",
              f"the report counts no bar / a bar with no value / a value apart ({v['report']})")
        d = await values("oi_daily.pct_front", [[D3.isoformat(), "14:45:00"], [D5.isoformat(), "09:30:00"],
                                                [SAT.isoformat(), "10:00:00"], [D2.isoformat(), "09:31:00"],
                                                [D4.isoformat(), "23:59:00"]])
        check(d["values"] == [0.2, None, None, None, 0.3] and d["present"] == [True, True, False, True, True]
              and d["bar_times"] is None and d["join"] == "date",
              f"daily: the entry DATE's row whatever the time; no row on a weekend; NaN reads null ({d['values']}, "
              f"{d['present']})")
        i = await values("oi_daily.n_strikes", [[D1.isoformat(), "10:00"], [D4.isoformat(), "10:00"]])
        check(i["values"] == [10.0, 13.0], f"an integer column reads as a number ({i['values']})")
        x = await values("oi_daily.ratio_x", [[D3.isoformat(), "10:00"]])
        check(x["values"] == [0.7], f"a numeric column reads as a float ({x['values']})")

        async def status(coro):
            try:
                await coro
                return 200
            except HTTPException as e:
                return e.status_code
        check(await status(values("oi_daily.lvl_hidden", bars)) == 400, "an unsurfaced metric's id is refused (400)")
        check(await status(values("pct_atm", bars)) == 400, "a bare name is not an id (400)")
        check(await status(values("oi_daily.pct_front", "nope")) == 400, "bars that are not a list are refused (400)")
        check(await status(r.spx_metrics_catalog(pool=None)) == 503
              and await status(r.spx_metrics_values(body={"id": "oi_daily.pct_front", "bars": bars}, pool=None)) == 503,
              "no open_interest pool configured: 503 from both endpoints, never a crash")

        print("cache")
        built = cat["built_at"]
        again = await sm.get_catalog(pool)
        check(again["metrics"] is cat["metrics"], "a second read with nothing changed is the cached catalog")
        async with pool.acquire() as conn:
            await conn.execute("UPDATE spx_metric_catalog SET description = 'edited' WHERE metric = 'rsi'")
        ed = await sm.get_catalog(pool)
        check(next(m for m in ed["metrics"] if m["id"] == "price_daily.rsi")["description"] == "edited",
              "an edit to the catalog rebuilds it (its content is in the cache key)")
        async with pool.acquire() as conn:
            await conn.execute(f"UPDATE spx_metric_catalog SET surfaced = false WHERE metric = 'ret'")
        ed2 = await sm.get_catalog(pool)
        check("price_5min.ret" not in {m["id"] for m in ed2["metrics"]}, "un-surfacing a row removes it on the next read")
        async with pool.acquire() as conn:
            await conn.execute(f"INSERT INTO spx_oi_5min VALUES ('{D1 - timedelta(days=1)}', '15:55', 4600, 0.05, NULL)")
        ed3 = await sm.get_catalog(pool)
        check(next(m for m in ed3["metrics"] if m["id"] == "oi_5min.pct_atm")["min_date"] == (D1 - timedelta(days=1)).isoformat(),
              "a backfilled row in a table rebuilds coverage (dates and row count are in the key)")
        _ = built
    finally:
        await pool.close()


# ── --live: read-only against the real open_interest database ───────────────

# Agreed with the user, 2026-11-17. A rebuilt catalog that changes them fails
# here on purpose: say so, and move them.
LIVE_EXPECT = {"served": 155, "displayed": 168, "not_surfaced": 25, "sections": {"oi": 70, "volume": 53, "price": 45}}
LIVE_SAMPLE = 40


async def run_live(dsn: str) -> None:
    import asyncpg

    from app.oo_backtest import spx_metrics as sm
    from app.oo_backtest.surface import quote_ident

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
                await conn.execute("CREATE TEMP TABLE spx_gate_probe (x int)")
                wrote = True
            except asyncpg.exceptions.ReadOnlySQLTransactionError:
                wrote = False
            check(not wrote, "and a write is refused by the server")
            if wrote:
                return

        print("catalog")
        cat = await sm.get_catalog(pool)
        rep = cat["report"]
        print(f"  info  built in {cat['build_s']}s; tables {cat['tables']}")
        check(not rep["unknown_table"] and not rep["missing_from_table"] and not rep["wrong_type"] and not rep["no_section"],
              f"every surfaced catalog row is served: none in an unknown table, missing, non-numeric or without a "
              f"section ({ {k: rep[k] for k in ('unknown_table', 'missing_from_table', 'wrong_type', 'no_section')} })")
        per = {s["key"]: sum(1 for m in cat["metrics"] if s["key"] in m["sections"]) for s in sm.SECTIONS}
        got = {"served": rep["served"], "displayed": rep["displayed"], "not_surfaced": rep["not_surfaced"], "sections": per}
        check(got == LIVE_EXPECT, f"the catalog matches the agreed counts ({got}; expected {LIVE_EXPECT})")
        check(rep["unique_names"] == rep["served"], f"no served name repeats across tables ({rep['unique_names']})")
        units = sorted({m["units"] for m in cat["metrics"]})
        from app.oo_backtest.surface import UNIT_FORMATS
        check(set(units) <= set(UNIT_FORMATS), f"every served unit has a display format ({units})")
        no_cov = [m["id"] for m in cat["metrics"] if not m["min_date"]]
        check(not no_cov, f"every served metric has a coverage date ({no_cov[:10]})")

        print("coverage, every metric")
        bad = []
        async with pool.acquire() as conn:
            for m in cat["metrics"]:
                if not m["min_date"]:
                    continue
                t = quote_ident(sm.SOURCES[m["source"]]["table"])
                col = f"NULLIF({quote_ident(m['metric'])}::float8, 'NaN'::float8)"
                md = date.fromisoformat(m["min_date"])
                on = await conn.fetchval(f"SELECT count(*) FROM {t} WHERE trade_date = $1 AND {col} IS NOT NULL", md)
                before = await conn.fetchval(f"SELECT count(*) FROM {t} WHERE trade_date < $1 AND {col} IS NOT NULL", md)
                if not on or before:
                    bad.append((m["id"], m["min_date"], on, before))
        check(not bad, f"each coverage date has a value on it and none before it, checked in the table ({bad[:5]})")

        print("values, sampled real entries")
        rng = random.Random(17)
        async with pool.acquire() as conn:
            for src, s in sm.SOURCES.items():
                ms = [m for m in cat["metrics"] if m["source"] == src]
                if not ms:
                    check(False, f"{src}: no served metric to sample")
                    continue
                picks = [ms[0], ms[-1]] + ([m for m in ms if m["form"] == "z"][:1])
                t = quote_ident(s["table"])
                if s["join"] == "bar":
                    rows = await conn.fetch(f"SELECT trade_date, quote_time FROM {t} ORDER BY random() LIMIT {LIVE_SAMPLE}")
                    keys = [(r["trade_date"], r["quote_time"]) for r in rows]
                    # Between bars, before the first, at 16:00, on a weekend.
                    keys += [(d, time(h, mi)) for d, _ in keys[:6] for h, mi in ((9, 29), (10, 2), (16, 0))]
                else:
                    rows = await conn.fetch(f"SELECT trade_date FROM {t} ORDER BY random() LIMIT {LIVE_SAMPLE}")
                    keys = [(r["trade_date"], time(rng.randint(9, 15), rng.choice((0, 30, 45)))) for r in rows]
                keys += [(k[0] + timedelta(days=(5 - k[0].weekday()) % 7 or 7), time(10, 0)) for k in keys[:2]]
                for m in picks:
                    vals, present, bar_times, _ = await sm.entry_values(pool, m, keys)
                    mismatches = []
                    col = f"NULLIF({quote_ident(m['metric'])}::float8, 'NaN'::float8)"
                    for k, v, p, bt in zip(keys, vals, present, bar_times):
                        if s["join"] == "bar":
                            r = await conn.fetchrow(f"SELECT quote_time, {col} AS v FROM {t} WHERE trade_date = $1 "
                                                    f"AND quote_time <= $2 ORDER BY quote_time DESC LIMIT 1", k[0], k[1])
                            want = (r is not None, r["quote_time"].strftime("%H:%M:%S") if r else None,
                                    None if r is None or r["v"] is None else float(r["v"]))
                        else:
                            r = await conn.fetchrow(f"SELECT {col} AS v FROM {t} WHERE trade_date = $1", k[0])
                            want = (r is not None, None, None if r is None or r["v"] is None else float(r["v"]))
                        if (p, bt, v) != want:
                            mismatches.append((str(k), (p, bt, v), want))
                    check(not mismatches, f"{m['id']}: {len(keys)} entries equal a plain per-key lookup "
                                          f"({sum(present)} present, {sum(1 for x in vals if x is not None)} with a value)"
                                          + (f" -- first mismatch {mismatches[0]}" if mismatches else ""))
    finally:
        await pool.close()


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--live", action="store_true",
                    help="read-only against the existing open_interest database instead of a temporary cluster "
                         "(OI_DATABASE_URL from the environment or .env, or --dsn)")
    ap.add_argument("--dsn", help="with --live: the database to read (default OI_DATABASE_URL)")
    a = ap.parse_args()
    if a.live:
        dsn = a.dsn
        if not dsn:
            try:
                from dotenv import load_dotenv
                load_dotenv(ROOT / ".env")
            except ImportError:
                pass
            dsn = os.getenv("OI_DATABASE_URL")
        if not dsn:
            print("FAIL: --live needs OI_DATABASE_URL (environment or .env) or --dsn")
            return 1
        asyncio.run(run_live(dsn))
        print()
        if FAILS:
            print(f"FAIL: {len(FAILS)} OI / volume / price metric check(s) failed (live, read-only)")
            return 1
        print("PASS: OI / volume / price metrics on the live open_interest database (read-only)")
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
        print(f"FAIL: {len(FAILS)} OI / volume / price metric SQL check(s) failed")
        return 1
    print("PASS: OI / volume / price metrics on a real Postgres")
    return 0


if __name__ == "__main__":
    sys.exit(main())
