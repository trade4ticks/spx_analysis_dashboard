"""Exercise the Equities Scalp endpoints without a database.

The scalp database lives on the VPS, so nothing here can reach it from a
development machine — which is exactly the situation in which an endpoint
ships broken. This runs each one against a fake connection that returns
realistically-shaped rows, and asserts the things that are true regardless of
what the data says.

WHAT IT CHECKS, and why each one has a reason to exist:

  * the three states are distinguishable. "No pool", "pool but no rows" and
    "rows" are different answers needing different actions from whoever is
    reading the page at 9am, and the failure is that they collapse into one
    empty table.
  * the catalog is a pure function of what the fake returns. If a metric the
    fake does not emit shows up in the response, something in the router
    declared it, which is the one thing this page must never do.
  * the default noise variant is never the median statistic. It reads exactly
    0.0 on sparse-quote names — when more than half of consecutive buckets
    carry an identical midpoint, the median change is zero by construction —
    which makes the ratio infinite and sorts the least tradeable names to the
    top of the ranking.
  * every SQL statement parses, and its placeholders match its arguments. An
    unused $1 cannot have its type inferred and fails to prepare at all.
  * every bound argument has the PYTHON TYPE its column requires. asyncpg
    binds by type and does not coerce, unlike psycopg2, so a date passed as a
    string raises at bind time:

        invalid input for query argument $1: '2026-08-28'
        ('str' object has no attribute 'toordinal')

    That shipped. The first version of this harness passed while the endpoint
    500'd on its first real request, because a fake that accepts anything
    cannot catch a type error -- it checked that the SQL was well formed and
    never that the arguments could be bound to it.

RUN IT LIVE WHERE THERE IS A DATABASE. `--live` calls the same endpoints
against the real one and is the only thing that proves a query runs rather
than merely parses. It SKIPS cleanly where no database is reachable, which is
every development machine here, so the fake path above is what has to carry
the type checking.
"""
from __future__ import annotations

import asyncio
import datetime
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.routers import equities_scalp as sc            # noqa: E402

BAD: list[str] = []

# A metric set shaped like the pipeline's: both generated families across
# several variants, horizons and statistics, plus fixed flow/spread names.
FAKE_METRICS = [
    "spread_bps_tw", "spread_cents_tw", "trades_per_min", "trade_size_median",
    "two_sided_balance", "off_exchange_share", "quote_bucket_coverage_10s",
    "noise_bps_tw_mid_10s_median", "noise_bps_tw_mid_10s_rms",
    "noise_bps_tw_mid_30s_rms", "noise_bps_last_mid_10s_p75",
    "noise_bps_bid_side_5s_mean",
    "ratio_tw_mid_10s", "ratio_last_mid_30s",
    "move_rate_tw_mid_10s", "move_bps_tw_mid_10s",
]
FAKE_DATES = [datetime.date(2026, 8, 28), datetime.date(2026, 8, 27)]


# ── column types, from the pipeline's DDL in scalp/db.py ────────────────────
#
# The point of this table is that asyncpg's error names neither the column nor
# the endpoint, so knowing which parameter was wrong means reading the SQL by
# eye. Here it is mechanical: the placeholder is matched to the column it is
# compared against, and the argument's type is checked against what that column
# accepts.
DATE_COLUMNS = {"trade_date", "first_entered", "sticky_until"}
TIMESTAMP_COLUMNS = {"bucket_start"}
TIMESTAMPTZ_COLUMNS = {"run_ts"}
TEXT_COLUMNS = {"symbol", "metric", "item", "variant"}

# `column <op> $n`, and the ANY() form a symbol list uses.
_BIND_RE = re.compile(
    r"\b([a-z_]+)\s*(?:=|<|>|<=|>=|<>|!=)\s*(?:ANY\s*\()?\$(\d+)")


def _type_problem(column: str, value):
    """None if `value` can bind to `column`, else why not."""
    if value is None:
        return None                      # NULL binds to anything
    if column in DATE_COLUMNS:
        # datetime is a SUBCLASS of date, so it would pass a naive isinstance
        # and then silently drop its time component. A DATE column wants a
        # date.
        if isinstance(value, datetime.datetime):
            return ("a datetime, not a date — it binds, and silently discards "
                    "the time")
        if not isinstance(value, datetime.date):
            return f"{type(value).__name__}, but this is a DATE column"
    elif column in TIMESTAMP_COLUMNS | TIMESTAMPTZ_COLUMNS:
        if not isinstance(value, datetime.datetime):
            return f"{type(value).__name__}, but this is a TIMESTAMP column"
    elif column in TEXT_COLUMNS:
        if isinstance(value, (list, tuple)):
            bad = [v for v in value if not isinstance(v, str)]
            if bad:
                return f"a list containing {type(bad[0]).__name__}, not str"
        elif not isinstance(value, str):
            return f"{type(value).__name__}, but this is a TEXT column"
    return None


def check_sql(sql: str, args: tuple) -> None:
    flat = " ".join(sql.split())
    used = {int(n) for n in re.findall(r"\$(\d+)", flat)}
    if used and max(used) != len(args):
        BAD.append(f"placeholder/arg mismatch: highest ${max(used)} with "
                   f"{len(args)} args — {flat[:110]}")
    # A gap is worse than a mismatch: Postgres cannot infer an unused
    # parameter's type and refuses to prepare the statement at all.
    missing = set(range(1, max(used) + 1)) - used if used else set()
    if missing:
        BAD.append(f"unused placeholders {sorted(missing)} — {flat[:110]}")

    # THE BIND CHECK. Every placeholder compared against a known column has to
    # carry a value that column will accept.
    for column, n in _BIND_RE.findall(flat):
        idx = int(n) - 1
        if idx >= len(args):
            continue
        problem = _type_problem(column, args[idx])
        if problem:
            BAD.append(
                f"${n} binds to {column} as {problem}. asyncpg binds by type "
                f"and will not coerce — {flat[:90]}")

    try:
        import sqlglot
        sqlglot.parse_one(re.sub(r"\$\d+", "'x'", flat), read="postgres")
    except ImportError:
        pass
    except Exception as exc:
        BAD.append(f"unparseable SQL: {exc} — {flat[:110]}")


def self_test_binds() -> int:
    """Prove the bind check fires. It was added because it was absent.

    A harness that passes while the endpoint 500s is worse than no harness,
    and the only defence is to make the check demonstrate itself.
    """
    bad = 0
    cases = [
        ("trade_date", "2026-08-28", True,  "a date as a string"),
        ("trade_date", datetime.date(2026, 8, 28), False, "a real date"),
        ("trade_date", datetime.datetime(2026, 8, 28, 9), True,
         "a datetime where a date is wanted"),
        ("trade_date", None, False, "NULL"),
        ("symbol", "FDX", False, "a ticker"),
        ("symbol", 3, True, "a number as a ticker"),
        ("symbol", ["FDX", "LLY"], False, "a ticker list"),
        ("bucket_start", "2026-08-28 09:45", True, "a timestamp as a string"),
        ("bucket_start", datetime.datetime(2026, 8, 28, 9, 45), False,
         "a real timestamp"),
        ("value", "anything", False, "an unmapped column, left alone"),
    ]
    for column, value, should_flag, what in cases:
        flagged = _type_problem(column, value) is not None
        if flagged != should_flag:
            verb = "was not flagged" if should_flag else "was flagged"
            print(f"  SELF-TEST: {what} on {column} {verb}")
            bad += 1
    if not bad:
        print("self-test: the bind check rejects a string date and accepts a "
              "real one")
    return bad


class Conn:
    def __init__(self, empty: bool = False):
        self.empty = empty

    async def fetch(self, sql, *args):
        check_sql(sql, args)
        if self.empty:
            return []
        if "DISTINCT trade_date" in sql:
            return [{"trade_date": d} for d in FAKE_DATES]
        if "GROUP BY metric" in sql:
            return [{"metric": m, "n": 587,
                     # One deliberately all-null metric: an all-null column and
                     # an absent one look identical in a pivot and only one of
                     # them means the pipeline broke.
                     "n_value": 0 if m == "noise_bps_bid_side_5s_mean" else 561}
                    for m in sorted(FAKE_METRICS)]
        return []

    async def fetchval(self, sql, *args):
        check_sql(sql, args)
        return 0 if self.empty else 587

    async def fetchrow(self, sql, *args):
        check_sql(sql, args)
        return None


class Acq:
    def __init__(self, empty): self.empty = empty
    async def __aenter__(self): return Conn(self.empty)
    async def __aexit__(self, *a): return False


class Pool:
    def __init__(self, empty=False): self.empty = empty
    def acquire(self): return Acq(self.empty)


async def run() -> int:
    fails = self_test_binds()

    # ── state 1: no pool at all ──────────────────────────────────────────
    j = await sc.meta(date=None, pool=None)
    if j.get("connected") is not False or not j.get("error"):
        print("  no-pool state does not report itself as disconnected")
        fails += 1
    if j.get("metrics") or j.get("dates"):
        print("  no-pool state returns data it cannot have")
        fails += 1

    # ── state 2: pool, but the pipeline has not written a session ────────
    j = await sc.meta(date=None, pool=Pool(empty=True))
    if not j.get("connected"):
        print("  empty-database state reports as disconnected — it is not, and")
        print("    the two need different actions from the reader")
        fails += 1
    if not j.get("note"):
        print("  empty-database state says nothing about why it is empty")
        fails += 1
    if not j.get("filters", {}).get("defaults"):
        print("  empty-database state drops the filter defaults, so the page")
        print("    cannot draw its sliders until data exists")
        fails += 1

    # ── state 3: real rows ───────────────────────────────────────────────
    j = await sc.meta(date=None, pool=Pool())
    if not j.get("connected") or not j.get("date"):
        print("  populated state did not resolve a date")
        fails += 1

    got = {m["metric"] for m in j["metrics"]}
    if got != set(FAKE_METRICS):
        extra = got - set(FAKE_METRICS)
        missing = set(FAKE_METRICS) - got
        print(f"  the catalog is not a pure function of the database: "
              f"extra={sorted(extra)} missing={sorted(missing)}")
        fails += 1

    # The one metric the fake returns as all-null must survive as a row, with
    # its emptiness legible rather than silently dropped.
    nulls = [m for m in j["metrics"] if m["n_value"] == 0]
    if len(nulls) != 1:
        print(f"  the all-null metric is not distinguishable: {len(nulls)} found")
        fails += 1

    # Every generated name parsed; every fixed name did not pretend to.
    for m in j["metrics"]:
        parsed = m["variant"]
        looks_generated = re.search(r"_\d+s(_[a-z0-9]+)?$", m["metric"]) \
            and m["metric"].startswith(("noise_bps_", "ratio_"))
        if bool(parsed) != bool(looks_generated):
            print(f"  {m['metric']}: parsed={parsed} but generated="
                  f"{bool(looks_generated)}")
            fails += 1

    # ── the default variant ──────────────────────────────────────────────
    dn = j.get("default_noise")
    if dn is None:
        print("  no default noise variant chosen")
        fails += 1
    elif dn.endswith("_median"):
        print(f"  the default noise variant is a MEDIAN ({dn}). It reads 0.0")
        print("    on sparse-quote names, which sorts the least tradeable")
        print("    names to the top of the ranking.")
        fails += 1
    elif dn not in FAKE_METRICS:
        print(f"  the default noise variant {dn!r} is not in the database")
        fails += 1

    # It must also degrade rather than invent when nothing preferred exists.
    only_median = sc._default_noise(["noise_bps_tw_mid_10s_median"])
    if only_median != "noise_bps_tw_mid_10s_median":
        print("  with only a median available the default is not it — the")
        print("    preference has become a requirement")
        fails += 1
    if sc._default_noise([]) is not None:
        print("  an empty metric set still yields a default")
        fails += 1

    # ── a pinned date is honoured, an unknown one falls back ─────────────
    j2 = await sc.meta(date=str(FAKE_DATES[1]), pool=Pool())
    if j2["date"] != str(FAKE_DATES[1]):
        print(f"  a pinned date was ignored: asked {FAKE_DATES[1]}, got {j2['date']}")
        fails += 1
    j3 = await sc.meta(date="1999-01-01", pool=Pool())
    if j3["date"] != str(FAKE_DATES[0]):
        print("  an unknown date did not fall back to the latest")
        fails += 1

    for b in BAD:
        print(f"  {b}")
    fails += len(BAD)

    print(f"\nstates checked: 3, dates: 2, metrics: {len(FAKE_METRICS)}, "
          f"failures: {fails}")
    return 1 if fails else 0


# ── against the real database ────────────────────────────────────────────────
#
# The fake above can only check what it knows to check. This calls the same
# endpoints against the actual database, which is the only thing that proves a
# query RUNS rather than merely parses -- the difference the string-date bug
# lived in.
#
# It SKIPS rather than fails where no database is reachable. That is every
# development machine here, and a check that cannot run is not a check that
# failed; making it fail would only teach everyone to ignore its output. The
# consequence is that the fake path has to carry the type checking, which is
# why the bind check above exists at all.

async def run_live() -> int:
    from app import db

    if db._scalp_dsn() is None:
        print("SKIP --live: no DSN. It derives from DATABASE_URL; set that or "
              "SCALP_DATABASE_URL.")
        return 0

    await db.init_pool()
    pool = await db.get_scalp_pool()
    if pool is None:
        st = db.pool_status().get("equities_scalp", {})
        print(f"SKIP --live: no pool — {st.get('error') or 'not configured'}")
        await db.close_pool()
        return 0

    fails = 0
    try:
        j = await sc.meta(date=None, pool=pool)
        if not j.get("connected"):
            print(f"  live /meta reports not connected: {j.get('error')}")
            return 1
        if not j.get("date"):
            print("  live /meta resolved no date — daily_metrics is empty.")
            print("    Not a failure of this code; the pipeline has not run.")
            return 0

        print(f"  /meta            {j['date']}  {j['symbols']} symbols  "
              f"{len(j['metrics'])} metrics")
        print(f"  default variant  {j.get('default_noise')}")
        print(f"  dates available  {len(j['dates'])}")
        if j.get("undocumented"):
            print(f"  undocumented     {len(j['undocumented'])}: "
                  f"{', '.join(j['undocumented'][:6])}")

        # THE REGRESSION, run for real. A pinned date arrives as a string from
        # the query layer and has to reach the query as a date. Nothing but a
        # live bind can prove it.
        pinned = j["dates"][min(1, len(j["dates"]) - 1)]
        j2 = await sc.meta(date=pinned, pool=pool)
        if j2.get("date") != pinned:
            print(f"  a pinned date did not round-trip: asked {pinned}, got "
                  f"{j2.get('date')}")
            fails += 1
        else:
            print(f"  pinned date      {pinned} bound and returned")

        # And an unparseable one must be a 400 rather than a 500.
        try:
            await sc.meta(date="not-a-date", pool=pool)
            print("  a malformed date was accepted rather than refused")
            fails += 1
        except Exception as exc:
            if getattr(exc, "status_code", None) != 400:
                print(f"  a malformed date raised {type(exc).__name__} rather "
                      f"than a 400")
                fails += 1
            else:
                print("  malformed date   refused with 400")

        if (j.get("default_noise") or "").endswith("_median"):
            print("  the live default noise variant is a MEDIAN — it reads 0.0")
            print("    on sparse-quote names and sorts them to the top")
            fails += 1
    finally:
        await db.close_pool()

    print(f"\nlive endpoints checked: 1, failures: {fails}")
    return 1 if fails else 0


if "--live" in sys.argv:
    sys.exit(asyncio.run(run_live()))
sys.exit(asyncio.run(run()))
