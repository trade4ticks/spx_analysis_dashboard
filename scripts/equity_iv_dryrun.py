"""Dry-run the new equity_iv endpoints against a fake connection.

No database is reachable from this environment, so this does two things a
parse check cannot:

  1. Runs every SQL string the new endpoints build through sqlglot's postgres
     parser. A malformed clause -- a dangling AND from an f-string branch, a
     missing space at a concatenation seam, a $N that outran its args list --
     fails here instead of at first click.
  2. Executes the Python paths end to end, so a NameError or a bad key in a
     branch that only fires under one toggle is caught too.

It does NOT check that the queries return the right rows. That needs the real
tables.
"""
import asyncio, datetime, json, os, sys, io, re
import sqlglot

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import app.routers.equity_iv as eiv

SEEN, BAD = [], []


def check_sql(sql, args):
    SEEN.append(sql)
    try:
        sqlglot.parse_one(sql, dialect="postgres")
    except Exception as exc:
        BAD.append(("PARSE", sql, str(exc)))
        return
    # placeholder/arg agreement -- an off-by-one here is a runtime 500
    used = {int(n) for n in re.findall(r"\$(\d+)", sql)}
    if used and (max(used) != len(args) or used != set(range(1, len(args) + 1))):
        BAD.append(("ARGS", sql, f"placeholders {sorted(used)} vs {len(args)} args"))


class Row(dict):
    def __getitem__(self, k):
        if k in self:
            return dict.__getitem__(self, k)
        if k in ("trade_date",):
            return datetime.date(2026, 8, 24)
        if k == "snapshot":
            return "1545"
        if k == "captured_at":
            return datetime.datetime(2026, 8, 24, 15, 45)
        if k == "source":
            return "live"
        if k == "n":
            return 250
        if k == "ex" or re.fullmatch(r"e\d+", k):
            return False
        if "extrap" in k and "rate" not in k:
            return False
        return 1.25


class Conn:
    async def fetch(self, sql, *args):
        check_sql(sql, args)
        if "generate_series" in sql or "snapshot" in sql:
            return [Row(trade_date=datetime.date(2026, 8, d), snapshot=s)
                    for d in (20, 21, 24) for s in ("0945", "1545")]
        return [Row() for _ in range(30)]

    async def fetchrow(self, sql, *args):
        check_sql(sql, args)
        return Row()

    async def fetchval(self, sql, *args):
        check_sql(sql, args)
        if "max(trade_date)" in sql:
            return datetime.date(2026, 8, 24)
        if "max(snapshot)" in sql:
            return "1545"
        return 0.62


class Acq:
    async def __aenter__(self): return Conn()
    async def __aexit__(self, *a): return False


class Pool:
    def acquire(self): return Acq()


def seed_catalog():
    """A synthetic catalog covering every shape the endpoints branch on."""
    by_col = {}

    def add(col, form, base=None, wing=None, tenor=None, fam="skew"):
        by_col[col] = {"column_name": col, "family": fam, "tenor": tenor,
                       "wing": wing, "form": form, "base_column": base,
                       "units": "vol_pts", "description": col, "formula": None,
                       "extrap_flags": []}

    for col, wing, tenor in (("skew_30d_25p_atm", "25p_atm", 30),
                             ("iv_30d_atm", "atm", 30),
                             ("rr_30d_25d", "25d", 30),
                             ("term_ratio_30d_90d", None, 30),
                             ("rv_21d", None, None)):
        add(col, "base", wing=wing, tenor=tenor)
        by_col[col]["extrap_flags"] = (
            [f"extrap_{tenor}d_{w}" for w in (wing or "").split("_") if w]
            if wing else [])
        for zw in (63, 252):
            add(f"{col}_z_{zw}", f"z_{zw}", base=col, wing=wing, tenor=tenor)

    eiv._catalog_cache = {
        "by_col": by_col,
        "extrap_cols": {"extrap_30d_25p", "extrap_30d_atm"},
        "live_metric_cols": {"spot", "extrap_rate_short", "source",
                             "captured_at", "median_n_strikes_clean",
                             "iv_30d_atm", "rv_21d", "term_ratio_30d_90d",
                             "px_vs_50dma", "spotvol_beta_1m", "spotvol_r2_1m",
                             *[c for c, e in by_col.items() if e["form"] == "base"]},
    }


async def main():
    seed_catalog()
    pool = Pool()
    cases = []

    cases.append(("ticker-header",
                  eiv.ticker_header(ticker="AAPL", date=None, snapshot=None, pool=pool)))

    for zw in (63, 252):
        for ex in (True, False):
            cases.append((f"unusual z={zw} excl={ex}",
                          eiv.unusual(ticker="AAPL", date=None, snapshot=None,
                                      z_window=zw, window="1y", limit=5,
                                      families=None, exclude_extrapolated=ex,
                                      pool=pool)))
    cases.append(("unusual family filter",
                  eiv.unusual(ticker="AAPL", date=None, snapshot=None, z_window=63,
                              window="all", limit=5, families="skew",
                              exclude_extrapolated=True, pool=pool)))

    for win in ("3m", "1y", "all"):
        for ex in (True, False):
            cases.append((f"rails window={win} excl={ex}",
                          eiv.rails(ticker="AAPL", metrics=None, date=None,
                                    snapshot=None, window=win, z_window=63,
                                    exclude_extrapolated=ex, pool=pool)))
    cases.append(("rails explicit metrics",
                  eiv.rails(ticker="AAPL", metrics="iv_30d_atm,rv_21d", date=None,
                            snapshot=None, window="1y", z_window=252,
                            exclude_extrapolated=True, pool=pool)))

    for mode in ("daily", "intraday", "candle"):
        for win in ("3m", "all"):
            for envon in (True, False):
                cases.append((f"series {mode} {win} env={envon}",
                              eiv.series(ticker="AAPL",
                                         metrics="skew_30d_25p_atm,iv_30d_atm",
                                         mode=mode, snapshot=None, window=win,
                                         z_window=63, envelope=envon,
                                         env_window=63, env_lo=0.10, env_hi=0.90,
                                         exclude_extrapolated=True, pool=pool)))
    cases.append(("series z-form metric",
                  eiv.series(ticker="AAPL", metrics="skew_30d_25p_atm_z_63",
                             mode="daily", snapshot="0945", window="1y",
                             z_window=63, envelope=True, env_window=20,
                             env_lo=0.05, env_hi=0.95,
                             exclude_extrapolated=False, pool=pool)))

    failed = 0
    for name, coro in cases:
        try:
            res = await coro
        except Exception as exc:
            failed += 1
            print(f"  RAISED  {name}: {type(exc).__name__}: {exc}")
            continue
        # FastAPI serialises the return value; a date or datetime that was
        # never passed through _jsonable() becomes a 500 at response time,
        # which no amount of SQL checking would have surfaced.
        try:
            json.dumps(res)
        except TypeError as exc:
            failed += 1
            print(f"  UNSERIALISABLE  {name}: {exc}")
    print(f"\ncases: {len(cases)}, raised: {failed}")
    print(f"sql statements checked: {len(SEEN)} ({len(set(SEEN))} distinct)")
    for kind, sql, msg in BAD[:12]:
        print(f"\n  {kind}: {msg}\n    {sql[:300]}")
    print(f"sql problems: {len(BAD)}")
    return 1 if (failed or BAD) else 0


sys.exit(asyncio.run(main()))
