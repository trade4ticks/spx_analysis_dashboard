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
from fastapi import HTTPException

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import app.routers.equity_iv as eiv
import app.routers.equity_iv_surface as eivs
import app.routers.equity_structures as eivp
from app.metrics_config import BASE_COLUMNS as _MC_BASE, Z_COLUMNS as _MC_Z

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
    _check_contamination(sql)


# The baseline must never contain the date it is scoring. These are the two
# ways the file has actually got that wrong, so they are asserted rather than
# left to review:
#
#   1. `trade_date <= $n` inside a baseline or distribution scan. The first
#      version of _daily_baseline shipped with exactly that, and it escaped
#      notice because mid-session the current day's 1545 row does not exist
#      yet -- the window was clean by accident and dirty after the close.
#   2. a rolling window frame ending at CURRENT ROW instead of 1 PRECEDING.
#
# The z windows these were written for are gone -- equity_metrics_z stores the
# score now. They still guard the PERCENTILE and BAND windows, which this page
# does still compute and which have the same requirement, and they still stand
# watch over the derivation being reintroduced.
def _check_contamination(sql):
    flat = " ".join(sql.split())
    is_baseline = ("bl_days AS" in flat or "stddev_samp" in flat
                   or "percentile_cont" in flat)
    if is_baseline and re.search(r"trade_date <= \$\d+", flat):
        BAD.append(("CONTAMINATED", flat,
                    "a scoring window uses trade_date <= (includes today)"))
    if "ROWS BETWEEN" in flat and "AND CURRENT ROW" in flat:
        BAD.append(("CONTAMINATED", flat,
                    "rolling frame ends at CURRENT ROW (scores a date "
                    "against itself); want 1 PRECEDING"))
    if "bl_days AS" in flat and not re.search(r"trade_date < \$\d+", flat):
        BAD.append(("CONTAMINATED", flat,
                    "baseline CTE has no strict trade_date < bound"))


class Row(dict):
    def __getitem__(self, k):
        if k in self:
            return dict.__getitem__(self, k)
        if k in ("trade_date",):
            return datetime.date(2026, 8, 24)
        # earnings_calendar is DATE NOT NULL + TEXT; the catch-all float below
        # made the header's day arithmetic fail on a fixture value rather than
        # on anything real.
        if k == "earnings_date":
            return datetime.date(2026, 9, 5)
        if k == "earnings_session":
            return "amc"
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


ANCHOR = datetime.date(2026, 8, 24)


class Conn:
    async def fetch(self, sql, *args):
        check_sql(sql, args)
        if "ORDER BY m.trade_date, m.snapshot" in sql:
            # The /series data query, fixtured as a real mid-session state:
            # four settled days, then today with intraday buckets present and
            # NO 1545 close. That is the only shape that exercises all three
            # modes honestly -- the daily line has nothing for today and must
            # append, while intraday and candle already have today's buckets.
            rows = [Row(trade_date=datetime.date(2026, 8, d), snapshot=s)
                    for d in (18, 19, 20, 21) for s in ("0945", "1200", "1545")]
            rows += [Row(trade_date=ANCHOR, snapshot=s) for s in ("0945", "1200")]
            # This fake ignores WHERE clauses in general, which is fine for SQL
            # checking and wrong for behaviour: without the snapshot filter,
            # daily mode would see today's intraday rows, conclude the session
            # was settled, and skip the append the case exists to test. So the
            # two filters that change WHICH ROWS COME BACK are applied.
            one = re.search(r"m\.snapshot = \$(\d+)", sql)
            if one:
                want = args[int(one.group(1)) - 1]
                rows = [r for r in rows if r["snapshot"] == want]
            cap = re.search(r"m\.snapshot <= \$(\d+)", sql)
            if cap:
                upto = args[int(cap.group(1)) - 1]
                rows = [r for r in rows
                        if r["trade_date"] < ANCHOR or r["snapshot"] <= upto]
            return rows
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

    # The scatter's default `size` is a catalogued metric in production, so
    # the synthetic catalog has to carry it or every cross-section case 400s
    # on the whitelist rather than on anything real.
    add("median_n_strikes_clean", "base", fam="liquidity")
    by_col["median_n_strikes_clean"]["units"] = "count"

    # term_ratio is a PAIR family: family "term_ratio" and tenor None, not a
    # single-tenor column. The seed filed it under the default family with a
    # tenor, which made a pair-only endpoint reject it and a pair member it
    # maps to absent entirely -- a fixture disagreeing with the real catalog
    # about the exact property under test.
    for col in ("term_ratio_7d_14d", "term_ratio_14d_30d",
                "term_ratio_30d_90d", "term_ratio_7d_30d"):
        add(col, "base", fam="term_ratio")
        by_col[col]["wing"] = "atm"

    for col, wing, tenor in (("skew_30d_25p_atm", "25p_atm", 30),
                             ("iv_30d_atm", "atm", 30),
                             ("rr_30d_25", "25d", 30),
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


def self_test():
    """Prove the contamination checks still fire before trusting a clean run.

    A check that cannot fail reads exactly like a check that passes. These
    assertions are the only thing standing between a future edit and the bug
    this file was written for, so they get tested themselves.
    """
    must_fire = [
        ("baseline scan with <=",
         "WITH bl_days AS (SELECT ticker, trade_date FROM equity_metrics "
         "WHERE snapshot = $1 AND trade_date <= $2 AND trade_date >= $3) SELECT 1"),
        ("rolling frame ending at CURRENT ROW",
         "SELECT avg(v) OVER (PARTITION BY ticker ORDER BY trade_date "
         "ROWS BETWEEN $1 PRECEDING AND CURRENT ROW), stddev_samp(v) FROM v"),
        ("baseline CTE with no strict upper bound",
         "WITH bl_days AS (SELECT ticker FROM equity_metrics "
         "WHERE snapshot = $1) SELECT 1"),
    ]
    must_not = [
        ("clean baseline",
         "WITH bl_days AS (SELECT ticker, trade_date FROM equity_metrics "
         "WHERE snapshot = $1 AND trade_date < $2 AND trade_date >= $3) "
         "SELECT stddev_samp(x) FROM t"),
        ("clean rolling frame",
         "SELECT avg(v) OVER (PARTITION BY ticker ORDER BY trade_date "
         "ROWS BETWEEN $1 PRECEDING AND 1 PRECEDING), stddev_samp(v) FROM v"),
    ]
    # The live partial point may be SCORED by the envelope but must never
    # enter it. This is a pure function, so it is tested directly rather than
    # inferred from a chart.
    env = eiv._rolling_pct_envelope
    settled   = [1.0] * 40
    with_live = settled + [99.0]            # an absurd live reading
    admit     = [True] * 40 + [False]

    # hi_q = 1.0 so the band's top IS the window maximum. At 0.9 a lone
    # outlier among 20 values never reaches the band, and the assertion would
    # pass whether or not the point leaked -- which is how the first version
    # of this test managed to prove nothing.
    a = env(with_live, 20, 0.0, 1.0, admit=admit)
    b = env(settled,   20, 0.0, 1.0)
    if a[:40] != b:
        print("  SELF-TEST FAILED: live point changed an earlier band")
        return 1
    if a[40] == (None, None):
        print("  SELF-TEST FAILED: live point got no band at all")
        return 1
    if a[40][1] >= 99.0:
        print("  SELF-TEST FAILED: live point leaked into its own band")
        return 1

    # ...and it MUST leak without the admit list, or none of the above is
    # evidence of anything.
    tail_leak = env(with_live + [1.0], 20, 0.0, 1.0)[41][1]
    tail_keep = env(with_live + [1.0], 20, 0.0, 1.0, admit=admit + [True])[41][1]
    if tail_leak != 99.0:
        print("  SELF-TEST FAILED: canary — outlier absent from an unfiltered band")
        return 1
    if tail_keep == tail_leak:
        print("  SELF-TEST FAILED: admit list has no effect on later bands")
        return 1

    bad = 0
    for name, sql in must_fire:
        n = len(BAD)
        _check_contamination(sql)
        if len(BAD) == n:
            print(f"  SELF-TEST FAILED: no finding for {name!r}")
            bad += 1
        del BAD[n:]
    for name, sql in must_not:
        n = len(BAD)
        _check_contamination(sql)
        if len(BAD) != n:
            print(f"  SELF-TEST FAILED: false positive on {name!r}")
            bad += 1
        del BAD[n:]
    return bad


async def main():
    if self_test():
        print("contamination checks are broken; not running the rest")
        return 1
    seed_catalog()
    pool = Pool()
    cases = []
    must_400 = []

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
                          eiv.rails(ticker="AAPL", metrics=None, z_windows=None, date=None,
                                    snapshot=None, window=win, z_window=63,
                                    exclude_extrapolated=ex, pool=pool)))
    # A per-rail window, mixed: the whole point is that two rails can sit in
    # one panel on different windows.
    cases.append(("rails mixed z windows",
                  eiv.rails(ticker="AAPL", metrics="iv_30d_atm,rv_21d,skew_30d_25p_atm",
                            z_windows="63,252,", date=None, snapshot=None,
                            window="1y", z_window=63,
                            exclude_extrapolated=True, pool=pool)))
    must_400.append(("rails rejects an unknown z window",
                     eiv.rails(ticker="AAPL", metrics="iv_30d_atm",
                               z_windows="99", date=None, snapshot=None,
                               window="1y", z_window=63,
                               exclude_extrapolated=True, pool=pool)))

    cases.append(("rails explicit metrics",
                  eiv.rails(ticker="AAPL", metrics="iv_30d_atm,rv_21d", z_windows=None, date=None,
                            snapshot=None, window="1y", z_window=252,
                            exclude_extrapolated=True, pool=pool)))

    for mode in ("daily", "intraday", "candle"):
        for win in ("3m", "all"):
            for envon in (True, False):
                cases.append((f"series {mode} {win} env={envon}",
                              eiv.series(ticker="AAPL",
                                         metrics="skew_30d_25p_atm,iv_30d_atm",
                                         mode=mode, snapshot=None, date=None,
                                         live_snapshot=None, include_today=True,
                                         window=win,
                                         z_window=63, envelope=envon,
                                         env_window=63, env_lo=0.10, env_hi=0.90,
                                         exclude_extrapolated=True, spot=True,
                                         pool=pool)))
    # The live-point paths: anchored date + selected bucket, in every mode,
    # and with the append suppressed.
    for mode in ("daily", "intraday", "candle"):
        for live in ("1200", "1545"):
            cases.append((f"series live {mode} @{live}",
                          eiv.series(ticker="AAPL", metrics="iv_30d_atm",
                                     mode=mode, snapshot=None,
                                     date="2026-08-24", live_snapshot=live,
                                     include_today=True, window="1y",
                                     z_window=63, envelope=True, env_window=63,
                                     env_lo=0.10, env_hi=0.90,
                                     exclude_extrapolated=True, spot=True,
                                     pool=pool)))
    cases.append(("series live suppressed",
                  eiv.series(ticker="AAPL", metrics="iv_30d_atm", mode="daily",
                             snapshot=None, date=None, live_snapshot="1200",
                             include_today=False, window="all", z_window=63,
                             envelope=False, env_window=63, env_lo=0.10,
                             env_hi=0.90, exclude_extrapolated=True, spot=True,
                             pool=pool)))

    cases.append(("series alt snapshot, no z_stored",
                  eiv.series(ticker="AAPL", metrics="skew_30d_25p_atm",
                             mode="daily", snapshot="0945", date=None,
                             live_snapshot=None, include_today=True, window="1y",
                             z_window=63, envelope=True, env_window=20,
                             env_lo=0.05, env_hi=0.95,
                             exclude_extrapolated=False, spot=True, pool=pool)))

    # These three were must-refuse while the page DERIVED its z: a stored z
    # column could not be re-derived per point, and two z windows in one
    # request needed two baselines. Reading the stored column removes both
    # constraints -- a z column is now just a column -- so they are ordinary
    # requests and are asserted to SUCCEED.
    cases.append(("series accepts a stored z column",
                  eiv.series(ticker="AAPL", metrics="skew_30d_25p_atm_z_63",
                             mode="daily", snapshot=None, date=None,
                             live_snapshot=None, include_today=True, window="1y",
                             z_window=63, envelope=True, env_window=20,
                             env_lo=0.05, env_hi=0.95,
                             exclude_extrapolated=False, spot=True, pool=pool)))
    cases.append(("rails accepts a stored z column",
                  eiv.rails(ticker="AAPL", metrics="skew_30d_25p_atm_z_63",
                            z_windows=None, date=None, snapshot=None, window="1y",
                            z_window=63, exclude_extrapolated=True,
                            pool=pool)))
    cases.append(("scanner accepts mixed z windows",
                  eiv.scanner(columns="skew_30d_25p_atm_z_63,iv_30d_atm_z_252",
                              date=None, snapshot=None, filter=[], sort=None,
                              dir="desc", limit=10,
                              exclude_extrapolated=True, pool=pool)))

    # ── the universe half, which reads z the same way ────────────────────
    for zc in ("skew_30d_25p_atm_z_63", "skew_30d_25p_atm"):
        for ex in (True, False):
            cases.append((f"cross-section x={zc} excl={ex}",
                          eiv.cross_section(x=zc, y="iv_30d_atm", date=None,
                                            snapshot=None,
                                            size="median_n_strikes_clean",
                                            color=None, exclude_extrapolated=ex,
                                            pool=pool)))
    cases.append(("cross-section coloured by z",
                  eiv.cross_section(x="skew_30d_25p_atm_z_252",
                                    y="iv_30d_atm_z_252", date=None,
                                    snapshot=None, size=None,
                                    color="rv_21d_z_252",
                                    exclude_extrapolated=True, pool=pool)))

    for m in ("term_ratio_30d_90d", "term_ratio_7d_30d"):
        for ex in (True, False):
            cases.append((f"universe-term-state {m} excl={ex}",
                          eiv.universe_term_state(metric=m, date=None,
                                                  snapshot=None, sessions=20,
                                                  exclude_extrapolated=ex,
                                                  pool=pool)))
    # Contango is a property of a tenor PAIR. Counting it on a single-tenor
    # column would silently compare a vol level against 1.0.
    must_400.append(("term-state rejects a non-pair metric",
                     eiv.universe_term_state(metric="skew_30d_25p_atm", date=None,
                                             snapshot=None, sessions=20,
                                             exclude_extrapolated=True,
                                             pool=pool)))

    for metric in ("skew_30d_25p_atm_z_63", "iv_30d_atm"):
        for win in ("3m", "all"):
            cases.append((f"universe-stats {metric} {win}",
                          eiv.universe_stats(metric=metric, date=None,
                                             snapshot=None, window=win, hot=1.5,
                                             exclude_extrapolated=True,
                                             pool=pool)))

    cases.append(("scanner z cols + z filter + z sort",
                  eiv.scanner(columns="skew_30d_25p_atm_z_63,iv_30d_atm,rv_21d",
                              date=None, snapshot=None,
                              filter=["skew_30d_25p_atm_z_63:gt:1.5",
                                      "term_ratio_30d_90d:lt:1.0"],
                              sort="skew_30d_25p_atm_z_63", dir="desc",
                              limit=300, exclude_extrapolated=True, pool=pool)))
    cases.append(("scanner base only, no baseline needed",
                  eiv.scanner(columns="iv_30d_atm,rv_21d", date=None,
                              snapshot=None, filter=[], sort="iv_30d_atm",
                              dir="asc", limit=50, exclude_extrapolated=True,
                              pool=pool)))
    cases.append(("scanner abs + nullok filters on z",
                  eiv.scanner(columns="skew_30d_25p_atm_z_252", date=None,
                              snapshot=None,
                              filter=["skew_30d_25p_atm_z_252:absgt:2",
                                      "iv_30d_atm:nullorgt:0.1"],
                              sort=None, dir="desc", limit=10,
                              exclude_extrapolated=False, pool=pool)))

    # ── rows 6-8: the surface panels ─────────────────────────────────────
    for kind in ("skew", "term", "skew_term"):
        for ex in (True, False):
            cases.append((f"curve-band {kind} excl={ex}",
                          eivs.curve_band(ticker="AAPL", kind=kind, dte=30,
                                          wing=25, deltas="25,75", date=None,
                                          snapshot=None, window="1y",
                                          exclude_extrapolated=ex, pool=pool)))
    cases.append(("curve-band term, all window",
                  eivs.curve_band(ticker="AAPL", kind="term", dte=30, wing=25,
                                  deltas="10,25,75,90", date=None, snapshot=None,
                                  window="all", exclude_extrapolated=True,
                                  pool=pool)))

    for win in ("3m", "all"):
        cases.append((f"tent {win}",
                      eivs.tent(ticker="AAPL", dte=30, long_delta=25, date=None,
                                snapshot=None, window=win,
                                exclude_extrapolated=True, pool=pool)))

    for ex in (True, False):
        cases.append((f"sticky-strike excl={ex}",
                      eivs.sticky_strike(ticker="AAPL", dte=30, date=None,
                                         snapshot=None, prev_date=None,
                                         exclude_extrapolated=ex, pool=pool)))
    cases.append(("sticky-strike explicit prev",
                  eivs.sticky_strike(ticker="AAPL", dte=30, date="2026-08-24",
                                     snapshot=None, prev_date="2026-08-21",
                                     exclude_extrapolated=True, pool=pool)))

    for view in eivs.GRID_VIEWS:
        for win in ("1y", "all"):
            cases.append((f"surface-grid {view} {win}",
                          eivs.surface_grid(ticker="AAPL", view=view, date=None,
                                            snapshot=None, window=win,
                                            exclude_extrapolated=True,
                                            pool=pool)))

    for xa, ya in (("skew_30d_25p_atm_z_63", "iv_30d_atm_z_63"),
                   ("skew_30d_25p_atm", "iv_30d_atm"),
                   ("skew_30d_25p_atm_z_63", "iv_30d_atm")):
        for win in ("3m", "all"):
            cases.append((f"time-scatter {xa}/{ya} {win}",
                          eivs.time_scatter(ticker="AAPL", x=xa, y=ya, date=None,
                                            snapshot=None, window=win,
                                            exclude_extrapolated=True,
                                            pool=pool)))

    for win in ("1y", "all"):
        cases.append((f"spot-vol {win}",
                      eivs.spot_vol(ticker="AAPL", dte=30, date=None,
                                    snapshot=None, window=win, pool=pool)))

    must_400.append(("curve-band rejects an unknown kind",
                     eivs.curve_band(ticker="AAPL", kind="nope", dte=30, wing=25,
                                     deltas="25", date=None, snapshot=None,
                                     window="1y", exclude_extrapolated=True,
                                     pool=pool)))
    # NOT "z_iv" -- that was this case's sentinel until z_iv became a real
    # view, at which point the assertion started testing that a valid view is
    # rejected. A sentinel has to be a name that cannot become meaningful.
    must_400.append(("surface-grid rejects an unknown view",
                     eivs.surface_grid(ticker="AAPL", view="__not_a_view__",
                                       date=None, snapshot=None, window="1y",
                                       exclude_extrapolated=True, pool=pool)))

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


    # ── the live point actually behaves ──────────────────────────────────
    # Running the endpoint is not the same as checking what it returned.
    async def one(**kw):
        base = dict(ticker="AAPL", metrics="iv_30d_atm", mode="daily",
                    snapshot=None, date="2026-08-24", live_snapshot="1200",
                    include_today=True, window="1y", z_window=63,
                    envelope=True, env_window=63, env_lo=0.10, env_hi=0.90,
                    exclude_extrapolated=True, spot=True, pool=pool)
        base.update(kw)
        return await eiv.series(**base)

    # ── structure presets ────────────────────────────────────────────────
    # Every built-in must resolve every column it names at every tenor. A
    # preset one rail short looks exactly like one that worked.
    from app.equity_presets import BUILTIN_STRUCTURES, resolve_preset
    real_cat = {c.name: {"column_name": c.name, "family": c.family,
                         "tenor": c.tenor, "wing": c.wing, "form": c.form}
                for c in list(_MC_BASE) + list(_MC_Z)}
    for p in BUILTIN_STRUCTURES:
        for t in (7, 14, 21, 30, 60, 90):
            r = resolve_preset(real_cat, p, t)
            if r["unresolved"]:
                failed += 1
                names = ", ".join(u["column"] for u in r["unresolved"])
                print(f"  PRESET  {p['name']} @ {t}d cannot resolve: {names}")
            if r["tenor"] != t:
                failed += 1
                print(f"  PRESET  {p['name']} asked for {t}d, resolved {r['tenor']}d")
    # A preset's TENOR is part of the preset. Listing without an override
    # must return each preset's own; the override is for a caller asking to
    # read one at a horizon it was not written for.
    #
    # This was live: the page passed its current tenor on every list, which
    # stamped it onto every preset and silently discarded the horizon a
    # saved preset carried. It hid behind the built-in, whose own tenor is
    # 21 -- so at 21 the override was a no-op and the feature looked fine.
    _saved_at_7 = {
        "key": "user:1", "id": 1, "name": "probe", "note": None,
        "tenor": 7, "builtin": False,
        "rails": [{"b": "zc_width_sigma_7d", "w": 63}],
        "scanner_columns": [], "scanner_filters": [],
        "scanner_sort": None, "scatter_x": None, "scatter_y": None,
    }
    kept = resolve_preset(real_cat, _saved_at_7, None)
    if kept["tenor"] != 7:
        failed += 1
        print(f"  PRESET  a preset saved at 7d listed as {kept['tenor']}d "
              f"with no override -- its own tenor was discarded")
    if kept["rails"][0]["b"] != "zc_width_sigma_7d":
        failed += 1
        print(f"  PRESET  and its columns moved with it: "
              f"{kept['rails'][0]['b']}")
    forced = resolve_preset(real_cat, _saved_at_7, 30)
    if forced["tenor"] != 30 or forced["rails"][0]["b"] != "zc_width_sigma_30d":
        failed += 1
        print(f"  PRESET  an explicit override did not take: "
              f"{forced['tenor']}d / {forced['rails'][0]['b']}")

    # The earnings filter must let NULL through: an ETF has no earnings date,
    # and a plain "> 20" on NULL is false in SQL, so a filter meant to exclude
    # events would exclude the entire ETF universe.
    for p in BUILTIN_STRUCTURES:
        for f in p.get("scanner_filters", []):
            if f["b"] == "days_to_earnings" and not f["op"].startswith("nullor"):
                failed += 1
                print(f"  PRESET  {p['name']} filters days_to_earnings with "
                      f"{f['op']!r}, which drops every ETF")

    # ── the header's earnings pill ───────────────────────────────────────
    hj = await eiv.ticker_header(ticker="AAPL", date="2026-08-24",
                                 snapshot=None, pool=pool)
    if hj.get("earnings_session") != "amc":
        failed += 1
        print(f"  HEADER  earnings_session came back {hj.get('earnings_session')!r}")
    if hj.get("earnings_days_calc") != 12:
        failed += 1
        print(f"  HEADER  earnings_days_calc {hj.get('earnings_days_calc')!r}, want 12")

    # ── the spot reference on /series ────────────────────────────────────
    for md in ("daily", "intraday", "candle"):
        sj = await eiv.series(ticker="AAPL", metrics="iv_30d_atm", mode=md,
                              snapshot=None, date="2026-08-24",
                              live_snapshot="1200", include_today=True,
                              window="1y", z_window=63, envelope=True,
                              env_window=63, env_lo=0.10, env_hi=0.90,
                              exclude_extrapolated=True, spot=True, pool=pool)
        sp = sj.get("spot") or {}
        if not sp.get("on"):
            failed += 1
            print(f"  SPOT  {md}: not on by default")
        elif not sp.get("points"):
            failed += 1
            print(f"  SPOT  {md}: no points returned")
        else:
            # It has to share the metric points' label space, or the client
            # places it in slots of its own and the reference drifts.
            def lab(p):
                return f"{p['t']} {p['snapshot']}" if p.get("snapshot") else p["t"]
            mlabels = {lab(p) for p in sj["series"][0]["points"]}
            slabels = {lab(p) for p in sp["points"]}
            if not (slabels & mlabels):
                failed += 1
                print(f"  SPOT  {md}: labels share nothing with the metric "
                      f"points -- {sorted(slabels)[:3]} vs {sorted(mlabels)[:3]}")
    off = await eiv.series(ticker="AAPL", metrics="iv_30d_atm", mode="daily",
                           snapshot=None, date=None, live_snapshot=None,
                           include_today=True, window="1y", z_window=63,
                           envelope=False, env_window=63, env_lo=0.10,
                           env_hi=0.90, exclude_extrapolated=True, spot=False,
                           pool=pool)
    if (off.get("spot") or {}).get("points"):
        failed += 1
        print("  SPOT  spot=False still returned points")

    r = await one()
    pts = r["series"][0]["points"]
    if not r["live_point"]["appended"]:
        failed += 1
        print("  LIVE  daily mid-session did not append today's point")
    elif not pts[-1].get("partial"):
        failed += 1
        print("  LIVE  appended point is not flagged partial")
    elif pts[-1]["t"] != "2026-08-24":
        failed += 1
        print(f"  LIVE  appended point dated {pts[-1]['t']}, want the anchor")
    elif pts[-1].get("snapshot") != "1200":
        failed += 1
        print(f"  LIVE  appended point at {pts[-1].get('snapshot')}, want 1200")
    elif any(p.get("partial") for p in pts[:-1]):
        failed += 1
        print("  LIVE  a settled close was flagged partial")

    # The selected bucket drives it, so it advances with the session.
    r2 = await one(live_snapshot="1330")
    if r2["series"][0]["points"][-1].get("snapshot") != "1330":
        failed += 1
        print("  LIVE  point did not follow the selected snapshot")

    # At the close there is nothing to append -- the settled row IS the point.
    r3 = await one(live_snapshot="1545")
    if r3["live_point"]["appended"]:
        failed += 1
        print("  LIVE  appended a partial point at the close bucket")

    r4 = await one(include_today=False)
    if r4["live_point"]["appended"]:
        failed += 1
        print("  LIVE  include_today=False still appended")

    # The intraday view marks NOTHING. Every point there is a snapshot and
    # none is a close, so flagging today's would mark a whole run for a
    # property they all share -- and with a few weeks of intraday history it
    # would be the entire chart. The daily and candle views keep the mark,
    # where a settled session really is a different kind of observation.
    r5 = await one(mode="intraday")
    n_intraday_partial = sum(1 for p in r5["series"][0]["points"] if p.get("partial"))
    if n_intraday_partial:
        failed += 1
        print(f"  LIVE  intraday flagged {n_intraday_partial} point(s) partial; "
              f"the daily/partial distinction does not apply there")
    if not r5["series"][0]["points"]:
        failed += 1
        print("  LIVE  intraday returned no points, so the check above is vacuous")

    r6 = await one(mode="candle")
    if not any(p.get("partial") for p in r6["series"][0]["points"]):
        failed += 1
        print("  LIVE  candle mode lost the partial bar; today's bar is an "
              "unfinished session and must stay distinguishable")

    # Endpoints that must REFUSE rather than fall back to the stored z. A
    # silent fallback here is indistinguishable from a correct answer, which
    # is exactly why it gets an assertion instead of a comment.
    for name, coro in must_400:
        try:
            await coro
        except HTTPException as exc:
            if exc.status_code != 400:
                failed += 1
                print(f"  WRONG STATUS  {name}: got {exc.status_code}")
        except Exception as exc:
            failed += 1
            print(f"  WRONG ERROR   {name}: {type(exc).__name__}: {exc}")
        else:
            failed += 1
            print(f"  NOT REFUSED   {name}: returned instead of raising 400")

    print(f"\nself-test: contamination checks fire correctly")
    print(f"cases: {len(cases)} + {len(must_400)} must-refuse, failures: {failed}")
    print(f"sql statements checked: {len(SEEN)} ({len(set(SEEN))} distinct)")
    for kind, sql, msg in BAD[:12]:
        print(f"\n  {kind}: {msg}\n    {sql[:300]}")
    print(f"sql problems: {len(BAD)}")
    return 1 if (failed or BAD) else 0


sys.exit(asyncio.run(main()))
