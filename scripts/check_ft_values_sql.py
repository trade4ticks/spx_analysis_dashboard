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


def main() -> int:
    if os.name == "posix" and os.geteuid() == 0:
        print("SKIP: running as root — initdb refuses to create a cluster as root")
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
