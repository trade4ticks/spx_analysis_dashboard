"""Gate: Factor Trades' export values and min strike against a REAL Postgres.
Needs initdb.

A temporary cluster (check_oo_market_sql's Cluster) -- nothing touches the
real database. What it pins:

  values      /trade-values through the router's own handler: raw values in
              KEY order (duplicates and all), NaN and a missing row as null,
              bins at the run's n_bins through _collapse_expr (a stored 0 is
              no bin), an unknown metric, a forward return and a metric
              with no bin column each reported in `missing`, never queried.
  min strike  Population's fragment, run as Postgres runs it: entry_price
              >= min, NaN excluded (NaN >= x is TRUE in Postgres), numeric
              and float8 alike, with max strike and a metric filter beside
              it, every placeholder bound to its own value.

PG_BIN points at initdb/pg_ctl if they are somewhere unusual. No binaries:
SKIP, not PASS.

--live (for a host that cannot run initdb, e.g. as root): the same questions
against the EXISTING database (OI_DATABASE_URL, or --dsn), READ-ONLY --
every connection opens with default_transaction_read_only=on and the gate
proves a write is refused before it reads anything. Real rows are sampled
from tt_bins / daily_features / one ticker of trade_paths and each value
compared with a plain per-key lookup; the NaN cases are VALUES lists.
  .venv/bin/python scripts/check_ft_values_sql.py --live
"""
from __future__ import annotations

import asyncio
import math
import os
import sys
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


SCHEMA = """
CREATE TABLE daily_features (ticker text, trade_date date, a double precision,
                             b numeric, ret_5d double precision, ret_5d_fwd_oc double precision,
                             nobin double precision);
CREATE TABLE tt_bins (ticker text, trade_date date, bin20_a int, bin20_b int);
CREATE TABLE trade_paths (ticker text, trade_date date, entry_price numeric, entry_f8 double precision);
INSERT INTO daily_features VALUES
  ('AAA', '2024-01-02', 1.5,   2.25, 0.03,  0.9, 7),
  ('AAA', '2024-01-03', 'NaN', 3.5,  -0.01, 0.9, 8),
  ('BBB', '2024-01-02', -4.0,  'NaN', NULL, 0.9, 9);
INSERT INTO tt_bins VALUES
  ('AAA', '2024-01-02', 13, 20),
  ('AAA', '2024-01-03', 1,  0),
  ('BBB', '2024-01-02', 20, 5);
INSERT INTO trade_paths VALUES
  ('P1', '2024-01-02', 5,     5),
  ('P2', '2024-01-02', 20,    20),
  ('P3', '2024-01-02', 300,   300),
  ('P4', '2024-01-02', 'NaN', 'NaN'),
  ('P5', '2024-01-02', NULL,  NULL);
"""


async def run(dsn: str) -> None:
    import asyncpg
    from app.routers import factor_trades as ft

    pool = await asyncpg.create_pool(dsn, min_size=1, max_size=2)
    try:
        async with pool.acquire() as conn:
            await conn.execute(SCHEMA)

        # ── /trade-values ──────────────────────────────────────────────
        keys = [("AAA", "2024-01-03"), ("BBB", "2024-01-02"), ("CCC", "2024-01-02"),
                ("AAA", "2024-01-02"), ("AAA", "2024-01-03")]
        req = ft.TradeValuesReq(primary_metric="a", secondary_metric="b", n_bins=5,
                                metrics=["ret_5d", "a", "nope", "ret_5d_fwd_oc", "nobin"],
                                tickers=[k[0] for k in keys], dates=[k[1] for k in keys])
        d = await ft.trade_values(req=req, pool=pool)
        check("error" not in d and d["n"] == len(keys), f"answers every key, in order ({d.get('n')} of {len(keys)})")
        check(d["values"].get("a") == [None, -4.0, None, 1.5, None],
              f"primary raw values by key; NaN and no row are null: {d['values'].get('a')}")
        check(d["values"].get("b") == [3.5, None, None, 2.25, 3.5],
              f"secondary (a numeric column) comes back as a number: {d['values'].get('b')}")
        check(d["values"].get("ret_5d") == [-0.01, None, None, 0.03, -0.01], "filter metric raw values")
        check(d["values"].get("nobin") == [8.0, 9.0, None, 7.0, 8.0], "a raw-only metric with no bin column")
        # n_bins 5: LEAST(((bin20 - 1) * 5) / 20, 4); bin20 0 = no bin.
        exp_a = [((1 - 1) * 5) // 20, ((20 - 1) * 5) // 20, None, ((13 - 1) * 5) // 20, 0]
        check(d["bins"].get("a") == exp_a, f"primary bins at n_bins 5: {d['bins'].get('a')} vs {exp_a}")
        exp_b = [None, ((5 - 1) * 5) // 20, None, 4, None]
        check(d["bins"].get("b") == exp_b, f"secondary bins; a stored 0 is no bin: {d['bins'].get('b')} vs {exp_b}")
        miss = set(d["missing"])
        check(miss == {"nope (raw value)", "ret_5d_fwd_oc (raw value)"},
              f"an unknown metric and a forward return are reported, not exported: {sorted(miss)}")
        check("ret_5d_fwd_oc" not in d["values"] and "nope" not in d["values"], "and never appear as columns")
        r2 = await ft.trade_values(req=ft.TradeValuesReq(primary_metric="nobin", n_bins=20,
                                                         tickers=["AAA"], dates=["2024-01-02"]), pool=pool)
        check(r2["missing"] == ["nobin (bin)"] and r2["values"]["nobin"] == [7.0],
              f"a primary with no tt_bins column: value kept, bin reported missing ({r2['missing']})")
        r3 = await ft.trade_values(req=ft.TradeValuesReq(tickers=["AAA"], dates=["2024-1-2x"]), pool=pool)
        check("error" in r3, "a malformed date is refused")

        # ── min strike, as Postgres runs it ────────────────────────────
        async def picked(**kw):
            pop = ft.Population(kw.get("hi"), kw.get("filters", []), kw.get("lo"))
            ps = pop.sql(0)
            sql = (f"SELECT tp.ticker FROM trade_paths tp{ps.join} WHERE true{ps.where}"
                   " ORDER BY tp.ticker")
            async with pool.acquire() as conn:
                return [r["ticker"] for r in await conn.fetch(sql, *ps.args)]
        check(await picked(lo=20.0) == ["P2", "P3"], "min strike keeps entry_price >= 20; NaN and NULL excluded")
        check(await picked(lo=20.0, hi=250.0) == ["P2"], "min and max together bound both ends")
        check(await picked() == ["P1", "P2", "P3", "P4", "P5"], "neither: every row (min strike adds nothing when off)")
        bare = ("SELECT ticker FROM trade_paths WHERE entry_price >= 20 ORDER BY ticker")
        async with pool.acquire() as conn:
            naive = [r["ticker"] for r in await conn.fetch(bare)]
        check("P4" in naive, "the trap is real: a bare >= keeps the NaN row (why min strike uses the predicate)")
    finally:
        await pool.close()


# ── --live: the same questions against an EXISTING database, read-only ────
#
# For a host that cannot run initdb (the VPS runs as root). Nothing is
# created, written or dropped: every connection is opened with
# default_transaction_read_only=on, so Postgres itself refuses a write, and
# the gate checks that setting took before it asks anything. Rows the
# fixture planted (NaN, a stored 0 bin) are found in the real tables where
# they exist, or built in a VALUES list where they cannot be (the NaN
# entry_price). Expected values come from a DIFFERENT path than the
# endpoint's: one plain SELECT per key, and the bin collapse in Python.

LIVE_KEYS = 40
LIVE_NBINS = 7          # not 20 and not a divisor of it: the collapse must work


def _collapse_py(b, n):
    return None if b is None or b <= 0 else min(((int(b) - 1) * n) // 20, n - 1)


async def run_live(dsn: str) -> None:
    import asyncpg
    from app.metric_filter import DAILY_FEATURES_NUMERIC_SQL
    from app.routers import factor_trades as ft

    pool = await asyncpg.create_pool(
        dsn, min_size=1, max_size=2,
        server_settings={"default_transaction_read_only": "on", "statement_timeout": "60000"})
    try:
        async with pool.acquire() as conn:
            ro = await conn.fetchval("SHOW default_transaction_read_only")
            check(ro == "on", f"every connection is read-only (default_transaction_read_only={ro})")
            if ro != "on":
                return
            try:
                await conn.execute("CREATE TEMP TABLE ft_gate_probe (x int)")
                wrote = True
            except asyncpg.exceptions.ReadOnlySQLTransactionError:
                wrote = False
            check(not wrote, "and a write is refused by the server")

            have = [r["column_name"] for r in await conn.fetch(DAILY_FEATURES_NUMERIC_SQL)]
            bins = await ft._bin_columns(conn)
            both = [m for m in have if f"bin20_{m}" in bins and not ft.is_outcome(m)]
            outcome = next((m for m in have if ft.is_outcome(m)), None)
            raw_only = next((m for m in have if f"bin20_{m}" not in bins and not ft.is_outcome(m)), None)
            check(len(both) >= 2, f"daily_features and tt_bins share binned metrics ({len(both)})")
            if len(both) < 2:
                return
            p, s = both[0], both[1]
            pc = ft._qi("bin20_" + p)
            keys = [(r["ticker"], r["trade_date"]) for r in await conn.fetch(
                f"SELECT ticker, trade_date FROM tt_bins WHERE {pc} > 0 LIMIT {LIVE_KEYS}")]
            n_pos = len(keys)
            # A stored 0 / NULL bin, where the table has one.
            keys += [(r["ticker"], r["trade_date"]) for r in await conn.fetch(
                f"SELECT ticker, trade_date FROM tt_bins WHERE {pc} IS NULL OR {pc} <= 0 LIMIT 3")]
            check(len(keys) >= 10, f"sampled {len(keys)} real trade days for {p} x {s}")
            keys = keys + [("ZZ_GATE_NO_SUCH_TICKER", keys[0][1]), keys[0]]   # no row; a duplicate

            # Expected: one plain lookup per key.
            async def one(table, col, t, d):
                return await conn.fetchval(
                    f"SELECT {col} FROM {table} WHERE ticker = $1 AND trade_date = $2 LIMIT 1", t, d)
            exp = {m: [] for m in (p, s)}
            expb = {m: [] for m in (p, s)}
            for t, d in keys:
                for m in (p, s):
                    v = await one("daily_features", f"{ft._qi(m)}::float8", t, d)
                    exp[m].append(None if v is None or not math.isfinite(v) else float(v))
                    expb[m].append(_collapse_py(await one("tt_bins", ft._qi("bin20_" + m), t, d), LIVE_NBINS))

        metrics = [p] + ([raw_only] if raw_only else []) + ["ft_gate_no_such_metric"] \
            + ([outcome] if outcome else [])
        req = ft.TradeValuesReq(primary_metric=p, secondary_metric=s, n_bins=LIVE_NBINS, metrics=metrics,
                                tickers=[k[0] for k in keys], dates=[k[1].isoformat() for k in keys])
        d = await ft.trade_values(req=req, pool=pool)
        check("error" not in d and d.get("n") == len(keys),
              f"/trade-values answers every key, in order ({d.get('n')} of {len(keys)}) {d.get('error') or ''}")
        if "error" in d:
            return

        def same(a, b):
            return len(a) == len(b) and all((x is None and y is None) or
                                            (x is not None and y is not None and abs(x - y) <= 1e-12 * max(1, abs(y)))
                                            for x, y in zip(a, b))
        for m, role in ((p, "primary"), (s, "secondary")):
            check(same(d["values"].get(m, []), exp[m]), f"{role} {m}: every raw value equals a direct lookup of its key")
            check(d["bins"].get(m) == expb[m], f"{role} {m}: every bin equals the stored bin20 collapsed to {LIVE_NBINS}")
        check(d["values"][p][-1] == d["values"][p][0] and d["values"][p][-2] is None,
              "a duplicate key repeats its value; a key with no row is null")
        zero = d["bins"][p][n_pos:-2]
        if zero:
            check(all(b is None for b in zero), f"a stored 0 / NULL bin exports as no bin ({len(zero)} found)")
        if raw_only:
            check(raw_only in d["values"] and raw_only not in d["bins"],
                  f"a filter metric with no bin column ({raw_only}): raw value only")
        miss = set(d["missing"])
        want = {"ft_gate_no_such_metric (raw value)"} | ({f"{outcome} (raw value)"} if outcome else set())
        check(miss == want, f"an unknown metric{' and a forward return (' + outcome + ')' if outcome else ''} "
                            f"are reported missing, not exported: {sorted(miss)}")

        # ── min strike ─────────────────────────────────────────────────
        async with pool.acquire() as conn:
            for cast in ("numeric", "double precision"):
                pop = ft.Population(250.0, [], 20.0)
                ps = pop.sql(0)
                rows = await conn.fetch(
                    f"SELECT tp.ticker FROM (VALUES ('P1', 5::{cast}), ('P2', 20::{cast}), ('P3', 300::{cast}),"
                    f" ('P4', 'NaN'::{cast}), ('P5', NULL::{cast})) AS tp(ticker, entry_price)"
                    f" WHERE true{ps.where} ORDER BY 1", *ps.args)
                check([r["ticker"] for r in rows] == ["P2"],
                      f"min 20 / max 250 on {cast}: keeps 20 only; NaN and NULL out ({[r['ticker'] for r in rows]})")
            naive = await conn.fetch("SELECT 1 FROM (VALUES ('NaN'::float8)) v(x) WHERE x >= 20")
            check(len(naive) == 1, "the trap is real here too: a bare >= keeps NaN")
            # On the real trade_paths, one ticker (its primary-key prefix):
            # the fragment's count equals Python's count of the same rows.
            t0 = keys[0][0]
            px = [r["entry_price"] for r in await conn.fetch(
                "SELECT entry_price FROM trade_paths WHERE ticker = $1", t0)]
            fin = sorted(float(x) for x in px if x is not None and math.isfinite(float(x)))
            if fin:
                lo = fin[len(fin) // 2]
                ps = ft.Population(None, [], lo).sql(1)
                n = await conn.fetchval(f"SELECT count(*) FROM trade_paths tp WHERE tp.ticker = $1{ps.where}",
                                        t0, *ps.args)
                check(n == sum(1 for x in fin if x >= lo),
                      f"real trade_paths ({t0}, {len(px)} rows): min strike {lo:g} keeps {n}, as counted directly")
            else:
                check(False, f"real trade_paths has no priced rows for {t0}")
    finally:
        await pool.close()


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--live", action="store_true",
                    help="read-only against an existing database instead of a temporary cluster "
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
            print(f"FAIL: {len(FAILS)} factor-trades values check(s) failed (live, read-only)")
            return 1
        print("PASS: factor-trades export values and min strike on the live database (read-only)")
        return 0
    if os.name == "posix" and os.geteuid() == 0:
        print("SKIP: running as root — initdb refuses to create a cluster as root; use --live for a read-only run against the existing database")
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
        print(f"FAIL: {len(FAILS)} factor-trades values SQL check(s) failed")
        return 1
    print("PASS: factor-trades export values and min strike on a real Postgres")
    return 0


if __name__ == "__main__":
    sys.exit(main())
