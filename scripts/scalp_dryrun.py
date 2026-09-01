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
from app import scalp_columns, scalp_config             # noqa: E402

BAD: list[str] = []

# A metric set shaped like the pipeline's: both generated families across
# several variants, horizons and statistics, plus fixed flow/spread names.
FAKE_METRICS = [
    "spread_bps_tw", "spread_cents_tw", "trades_per_min", "trade_size_median",
    "two_sided_balance", "off_exchange_share", "unidentified_exchange_share",
    "shares_per_min",
    # Not in any default column set, deliberately — the case that matters is a
    # metric nobody chose in advance still being screenable, which is the
    # whole principle.
    "off_mid_bps",
    # The trade-price basis, so the roles that hold it fixed resolve and the
    # comparison column can be exercised rather than merely declared.
    "noise_bps_trade_price_5s_rms", "ratio_trade_price_5s_rms",
    "quote_bucket_coverage_10s",
    # The pinned 5s family, complete. Adding only the noise column made the
    # coverage filter go INERT — the coverage role resolves at the selected
    # horizon, so a 5s variant needs 5s coverage — and the harness caught it.
    # A partial family in a fixture silently disables whatever depends on the
    # missing half.
    "noise_bps_tw_mid_5s_rms", "ratio_tw_mid_5s_rms",
    "quote_bucket_coverage_5s", "move_rate_tw_mid_5s", "move_bps_tw_mid_5s",
    # THE MEDIAN IS THE UNSUFFIXED FORM. The fixture carried
    # "noise_bps_tw_mid_10s_median" — a name the pipeline never emits — so
    # every median assertion here was testing a shape that does not exist.
    "noise_bps_tw_mid_10s", "noise_bps_tw_mid_10s_rms",
    "noise_bps_tw_mid_30s_rms", "noise_bps_last_mid_10s_p75",
    "noise_bps_bid_side_5s_mean",
    "ratio_tw_mid_10s", "ratio_tw_mid_10s_rms", "ratio_last_mid_30s",
    "move_rate_tw_mid_10s", "move_bps_tw_mid_10s",
    "reference_price",
    # Undocumented ON PURPOSE. The catalog must carry it, unlinked, rather
    # than dropping it: a metric that exists and is not shown is worse than
    # one shown without a definition, since only the second is visible as a
    # gap. metric_docs has no entry for this shape.
    "some_future_metric_nobody_documented",
]
UNDOCUMENTED = "some_future_metric_nobody_documented"
FAKE_DATES = [datetime.date(2026, 8, 28), datetime.date(2026, 8, 27)]

# A longer run for the health panel, newest last. 2026-08-24 is the BROKEN
# session: arrivals 37% below trailing and eleven symbols short, which is the
# shape of the failure the panel exists for. Without a bad date in the fixture
# the panel would be asserted to work having never flagged anything.
HEALTH_DATES = [datetime.date(2026, 8, d) for d in
                (10, 11, 12, 13, 14, 17, 18, 19, 20, 21, 24)]
BROKEN_DATE = datetime.date(2026, 8, 24)
# A NORMAL date missing eight symbols: index ETFs that returned quotes with no
# trades and were correctly refused at fetch. 579/587 is 98.6% and must be
# VISIBLE without being FLAGGED — the system working is not an incident, and a
# panel that reddens on it is one nobody reads.
REFUSED_DATE = datetime.date(2026, 8, 20)
UNIVERSE_N = 587
MISSING = {BROKEN_DATE: 60, REFUSED_DATE: 8}


def _health_median(day, metric):
    """The per-date median the fake reports for a watched metric."""
    if metric == "trades_per_min":
        return 24.0 * (0.63 if day == BROKEN_DATE else 1.0)
    if metric == "off_exchange_share":
        return 0.38
    if metric == "unidentified_exchange_share":
        return 0.02
    return 1.0

# Enough symbols to sort, and deliberately not all alike: one passes every
# filter, one is too tight to trade, one is sparsely quoted, and one has a
# missing ratio so the null-ordering assertion has something to order.
FAKE_SYMBOLS = ["AAAA", "BBBB", "CCCC", "DDDD", "EEEE"]

# The correlation and stability panels need more than five names to say
# anything -- both refuse a sample under five, and a fixture that trips the
# refusal tests the refusal rather than the estimator.
_CORR_SYMBOLS = [f"S{i:02d}" for i in range(24)]


def _pivot_row(symbol: str, i: int, names: list[str]) -> dict:
    """One pivoted row, with values shaped like the metric each column holds.

    A fake that answers 1.25 to everything cannot exercise a filter: every row
    passes or every row fails, and the thresholds are never the thing under
    test.
    """
    row = {"symbol": symbol}
    for j, name in enumerate(names):
        if name.startswith("noise_bps"):
            # EEEE is quiet enough to trip min_noise_bps; CCCC is too noisy
            # for max_noise_bps.
            v = {0: 2.0, 1: 3.0, 2: 9.9, 3: 1.4, 4: 0.05}[i]
        elif name.startswith("ratio"):
            v = None if symbol == "DDDD" else {0: 4.0, 1: 2.5, 2: 0.4, 3: 0.0,
                                               4: 40.0}[i]
        elif name.startswith("quote_bucket_coverage"):
            v = {0: 0.99, 1: 0.95, 2: 0.90, 3: 0.33, 4: 0.97}[i]
        elif name.startswith("spread_cents"):
            v = {0: 8.0, 1: 7.5, 2: 4.0, 3: 6.0, 4: 9.0}[i]
        elif name.startswith("spread_bps"):
            v = 6.0 + i
        elif name.startswith("move_rate"):
            v = {0: 0.62, 1: 0.55, 2: 0.71, 3: 0.09, 4: 0.40}[i]
        elif name.startswith("move_bps"):
            v = 1.0 + i * 0.3
        elif name == "shares_per_min":
            v = {0: 9000.0, 1: 4400.0, 2: 12000.0, 3: 300.0, 4: 3600.0}[i]
        elif name == "trades_per_min":
            v = {0: 45.0, 1: 22.0, 2: 60.0, 3: 3.0, 4: 18.0}[i]
        elif name == "trade_size_median":
            v = 40.0 + i * 5
        elif name == "two_sided_balance":
            v = 0.9 - i * 0.1
        elif name == "off_exchange_share":
            v = 0.35 + i * 0.02
        elif name == "reference_price":
            v = 100.0 + i * 50
        else:
            v = 1.25
        row[f"v{j}"] = v
    return row


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
    """None if `value` can bind to `column`, else why not.

    A list is checked ELEMENT-WISE, not rejected. `column = ANY($n)` is the
    normal way to pass a set of dates or symbols, and the element type is what
    asyncpg binds — so a list of dates on a DATE column is right and a list of
    strings on one is the same defect as a bare string.
    """
    if value is None:
        return None                      # NULL binds to anything
    if isinstance(value, (list, tuple)):
        for element in value:
            problem = _type_problem(column, element)
            if problem:
                return f"a list containing {problem}"
        return None
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
        if not isinstance(value, str):
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


def self_test_fixture_is_real() -> int:
    """Every fake metric must be a name the pipeline could actually emit.

    The fixture carried "noise_bps_tw_mid_10s_median" for weeks. No such
    metric exists — the median is the UNSUFFIXED form — so every median
    assertion in this file was passing against a shape production never
    produces, and the by-statistic grouping bug it should have caught went
    straight through.

    metric_docs is the authority for what can exist, the same one
    check_scalp_metrics uses. The deliberately-undocumented entry is exempt
    because being undocumented is the property it exists to test.
    """
    from app import scalp_metric_docs as docs
    bad = 0
    for m in FAKE_METRICS:
        if m == UNDOCUMENTED:
            continue
        if docs.describe(m) is None:
            print(f"  SELF-TEST: fixture metric {m!r} is not a name the "
                  f"pipeline can emit — metric_docs does not recognise it, so "
                  f"every assertion about it is about fiction")
            bad += 1
    if not bad:
        print(f"self-test: all {len(FAKE_METRICS) - 1} fixture metrics are "
              f"shapes the pipeline can emit")
    return bad


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
        ("symbol", ["FDX", 7], True, "a ticker list with a number in it"),
        ("trade_date", [datetime.date(2026, 8, 28)], False, "a date list"),
        ("trade_date", ["2026-08-28"], True, "a list of date strings"),
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
        # INTRADAY FIRST, because its queries share prefixes with several
        # below — "DISTINCT trade_date" in particular. The looser branch
        # swallowed the intraday date list and handed back the daily fixture's
        # two dates, so the repeatability matrix keyed on dates it had no rows
        # for and came out entirely null. Third time branch ordering in this
        # fake has produced a silently empty result.
        if "FROM intraday_monthly" in sql:
            import datetime as _dt
            out = []
            for mo in (_dt.date(2026, 8, 1), _dt.date(2026, 7, 1)):
                for b in range(26):
                    row = {"bucket_key": mo,
                           "t": _dt.time(9 + (30 + b * 15) // 60,
                                         (30 + b * 15) % 60),
                           # 10 and 11 on purpose: the partial-month count the
                           # panel has to surface.
                           "sessions": 10 + (b % 2),
                           "trades": 4000 + b}
                    for k in args[0:0] or []:
                        pass
                    # EVERY ROLE KEY, derived. A hardcoded list here broke
                    # the moment a role was added — the endpoint selects the
                    # roles that resolve, so the fake has to answer for all of
                    # them or a new column is a KeyError in the harness rather
                    # than a finding about the code.
                    for k in scalp_columns.BY_KEY:
                        row[k] = 1.0 + b * 0.2 + (0.5 if mo.month == 7 else 0)
                    out.append(row)
            return out
        if "FROM intraday_metrics" in sql and "DISTINCT trade_date" in sql:
            return [{"trade_date": d} for d in HEALTH_DATES]
        if "FROM intraday_metrics" in sql and "count(*) AS sessions" in sql:
            import datetime as _dt
            out = []
            for b in range(26):
                row = {"t": _dt.time(9 + (30 + b * 15) // 60, (30 + b * 15) % 60),
                       "sessions": 11}
                for k in scalp_columns.BY_KEY:
                    row[k] = 1.0 + b * 0.3
                out.append(row)
            return out
        if "FROM intraday_metrics" in sql:
            import datetime as _dt
            return [{"trade_date": d,
                     "t": _dt.time(9 + (30 + b * 15) // 60, (30 + b * 15) % 60),
                     "v": 1.0 + b * 0.3 + i * 0.05}
                    for i, d in enumerate(HEALTH_DATES) for b in range(26)]
        if "min(entry_ts)" in sql:
            import datetime as _dt
            return [{"trade_date": HEALTH_DATES[0],
                     "first_entry": _dt.time(9, 40),
                     "last_exit": _dt.time(13, 30), "trips": 12}]
        if "FROM provenance" in sql:
            return [{"item": "trade_retained_share", "value": 0.94},
                    {"item": "rows_raw", "value": 812345.0}]
        # MOST SPECIFIC FIRST. /meta and /candidates take the newest dates with
        # a literal LIMIT; /health takes a parameterised one because its span
        # depends on the trailing window. Testing the looser pattern first
        # swallowed the health query and handed it two dates, which is not
        # enough to have a trailing baseline at all — the panel then looked
        # empty rather than wrong.
        # MOST SPECIFIC FIRST. Three different queries read fills_daily and
        # the loosest pattern would swallow all of them -- which it did, and
        # the traded-marker aggregate came back shaped like a daily row.
        if "GROUP BY symbol" in sql and "fills_daily" in sql and "days" in sql:
            # Two of the five symbols have been traded, one profitably and one
            # not. A fixture where everything is traded cannot show that the
            # marker distinguishes, and one where nothing is cannot show it
            # appears at all.
            return [
                {"symbol": "AAAA", "days": 3, "trips": 42, "net_pnl": 128.4,
                 "pnl_per_min": 1.42},
                {"symbol": "CCCC", "days": 2, "trips": 18, "net_pnl": -31.0,
                 "pnl_per_min": -0.55},
            ]
        if "sum(net_pnl)" in sql and "GROUP BY symbol" in sql:
            return [{"symbol": s, "pnl": 10.0} for s in ("AAAA", "CCCC")]
        if "FROM fills_daily f" in sql or "JOIN fills_daily" in sql:
            # The calibration join. Deliberately shaped so one metric ranks
            # PERFECTLY with the target and the rest do not — a correlation
            # harness whose data has no signal cannot tell a working estimator
            # from one that returns zero.
            out = []
            for i, sym in enumerate(_CORR_SYMBOLS):
                for m in ("ratio_tw_mid_10s_rms", "two_sided_balance",
                          "trades_per_min", "spread_bps_tw",
                          "quote_bucket_coverage_5s"):
                    x = {
                        # Ranks WITH the target: the metric that works.
                        "ratio_tw_mid_10s_rms": float(i),
                        # Two metrics that rank AGAINST the direction their
                        # roles declare. Both are needed, because one
                        # contradiction and two are different objects and only
                        # the second exercises the grouping.
                        "two_sided_balance": float(-i),
                        "trades_per_min": float(-i) + (i % 2) * 0.1,
                        "spread_bps_tw": float(i % 3),
                        # CONSTANT on purpose: a metric with one value has no
                        # ordering to correlate, and must come back as None
                        # rather than 0.0 — a zero sorts among the real
                        # answers and reads as "no relationship found".
                        "quote_bucket_coverage_5s": 0.97,
                    }[m]
                    out.append({"metric": m, "x": x, "y": float(i),
                                "trade_date": FAKE_DATES[0], "symbol": sym})
            return out
        if "FROM fills_daily" in sql:
            return [{"trade_date": FAKE_DATES[0], "symbol": s, "trips": 4,
                     "net_pnl": 12.5, "win_rate": 0.75,
                     "attention_minutes": 30.0, "dollars_per_min": 0.42,
                     "trips_per_min": 0.13, "median_hold_s": 8.0,
                     "shares": 200.0} for s in FAKE_SYMBOLS]
        if "DISTINCT trade_date" in sql and "LIMIT $1" in sql:
            return [{"trade_date": x} for x in reversed(HEALTH_DATES)]
        if "DISTINCT trade_date" in sql:
            return [{"trade_date": d} for d in FAKE_DATES]
        if "GROUP BY trade_date, metric" in sql:
            return [{"trade_date": day, "metric": m,
                     "med": _health_median(day, m), "n": UNIVERSE_N}
                    for day in HEALTH_DATES for m in args[1]]
        if "count(DISTINCT symbol)" in sql and "GROUP BY trade_date" in sql:
            return [{"trade_date": day,
                     "n_symbols": UNIVERSE_N - MISSING.get(day, 0),
                     "n_metrics": len(FAKE_METRICS)} for day in HEALTH_DATES]
        if "FROM universe u" in sql:
            return [{"trade_date": day, "symbol": f"X{i:03d}"}
                    for day, n in MISSING.items() for i in range(n)]
        if "FROM universe" in sql:
            return [{"trade_date": day, "n": UNIVERSE_N} for day in HEALTH_DATES]
        if "SELECT symbol, metric, value" in sql and "metric = ANY($2)" in sql:
            # The coherence probe: the contradicting metrics across the whole
            # universe. Shaped so they rank together, which is the case the
            # grouped callout exists for.
            out = []
            for i, s in enumerate(_CORR_SYMBOLS):
                for m in args[1]:
                    out.append({"symbol": s, "metric": m,
                                "value": float(i) + (0.01 if "trades" in m else 0)})
            return out
        if "SELECT symbol, metric, value" in sql:
            # For the correlation matrix. Two metrics are deliberately made
            # IDENTICAL up to a scale factor so the redundancy detector has
            # something real to find, and one is constant so the zero-variance
            # branch is exercised rather than assumed.
            out = []
            for i, s in enumerate(_CORR_SYMBOLS):
                for m in FAKE_METRICS:
                    if m == "ratio_tw_mid_10s":
                        v = float(i)
                    elif m == "ratio_tw_mid_10s_rms":
                        v = float(i) * 3.0 + 1.0        # perfectly co-ranked
                    elif m == "spread_bps_tw":
                        v = 7.0                          # constant
                    else:
                        v = float((i * 7 + len(m)) % 23)
                    out.append({"symbol": s, "metric": m, "value": v})
            return out
        if "symbol = ANY($2)" in sql and "metric = ANY($3)" in sql:
            # /series: the metrics and symbols the caller actually asked for,
            # so a response carrying something it was not asked for is a bug
            # the harness can see.
            out = []
            for day in HEALTH_DATES:
                for s in args[1]:
                    for m in args[2]:
                        out.append({"trade_date": day, "symbol": s,
                                    "metric": m,
                                    "value": 2.0 + len(m) % 5 + len(s)})
            return out
        if "SELECT trade_date, symbol, metric, value" in sql:
            # For rank stability. One metric ranks identically every session,
            # one reverses completely -- so an estimator that returns a
            # constant cannot pass.
            out = []
            for di, day in enumerate(HEALTH_DATES):
                for i, s in enumerate(_CORR_SYMBOLS):
                    for m in ("ratio_tw_mid_10s", "ratio_last_mid_30s"):
                        if m == "ratio_tw_mid_10s":
                            v = float(i)                     # stable
                        else:
                            v = float(i if di % 2 == 0 else -i)   # flips
                        out.append({"trade_date": day, "symbol": s,
                                    "metric": m, "value": v})
            return out
        if "DISTINCT metric" in sql:
            return [{"metric": m} for m in FAKE_METRICS]
        if "GROUP BY m.symbol" in sql:
            # The pivot. Aliases are read OUT OF THE SQL rather than assumed
            # from the bind list: the statement now also selects derived
            # columns as `d_<key>` expressions, which are not columns and do
            # not appear in the metric list. Assuming the shape is what made
            # the first derived column a KeyError here rather than a finding.
            aliases = re.findall(r"AS (v\d+|d_\w+)", sql)
            names = [a for a in args[1:-1]]
            out = []
            for i, sym in enumerate(FAKE_SYMBOLS):
                row = _pivot_row(sym, i, names)
                for a in aliases:
                    if a.startswith("d_"):
                        # A product of two of the bound metrics. The exact
                        # value does not matter; that it is a NUMBER and
                        # varies by symbol does, or nothing can screen on it.
                        row[a] = (1000.0 + i * 250.0) * (10.0 + i)
                    row.setdefault(a, None)
                out.append(row)
            return out
        if "trade_date = ANY(" in sql:
            # The sparkline history.
            out = []
            for sym in FAKE_SYMBOLS:
                for dt in FAKE_DATES:
                    out.append({"trade_date": dt, "symbol": sym,
                                "value": 1.5 + len(sym) * 0.1})
            return out
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
        if self.empty:
            return None if "max(" in sql else 0
        # A scalar's TYPE follows its column. Returning a number for
        # max(trade_date) makes the fake hand a date column an int, which
        # is a bind error in production and was one here — the bind check
        # caught it as soon as an endpoint started using it.
        if "max(trade_date)" in sql:
            return FAKE_DATES[0]
        return 587

    async def execute(self, sql, *args):
        # DDL carries no placeholders; check_sql's arity test would read
        # CREATE TABLE as a statement with zero args and be right about it.
        if "$" in sql:
            check_sql(sql, args)
        return "OK"

    async def executemany(self, sql, args_list):
        for a in args_list:
            check_sql(sql, tuple(a))
        return "OK"

    def transaction(self):
        return _Txn()

    async def fetchrow(self, sql, *args):
        check_sql(sql, args)
        if self.empty:
            return None
        # A bare aggregate ALWAYS returns a row in Postgres, even over an
        # empty table — count() gives 0 and sum() gives NULL. A fake that
        # answers None to one is not modelling the database, it is modelling
        # a table with no rows, and the difference is a TypeError in the
        # endpoint that would never happen in production.
        if "count(" in sql.lower():
            return {"n": 20, "pnl": 152.5, "symbols": 5, "sessions": 4}
        return None


class _Txn:
    async def __aenter__(self): return None
    async def __aexit__(self, *a): return False


class Acq:
    def __init__(self, empty): self.empty = empty
    async def __aenter__(self): return Conn(self.empty)
    async def __aexit__(self, *a): return False


class Pool:
    def __init__(self, empty=False): self.empty = empty
    def acquire(self): return Acq(self.empty)


async def run() -> int:
    fails = self_test_binds() + self_test_fixture_is_real()

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

    # ── nothing is filtered out of the catalog ───────────────────────────
    if UNDOCUMENTED not in got:
        print(f"  {UNDOCUMENTED!r} is in the database and not in the catalog.")
        print("    An undocumented metric must appear unlinked, not vanish —")
        print("    a metric that exists and is not shown is worse than one")
        print("    shown without a definition.")
        fails += 1
    else:
        entry = next(m for m in j["metrics"] if m["metric"] == UNDOCUMENTED)
        if entry["href"] is not None or entry["tooltip"] is not None:
            print("  an undocumented metric was given a link or a definition")
            fails += 1
        if UNDOCUMENTED not in (j.get("undocumented") or []):
            print("  an undocumented metric is not reported as undocumented,")
            print("    so nothing on the page marks it as a gap")
            fails += 1

    # ── the default variant ──────────────────────────────────────────────
    dn = j.get("default_noise")
    if dn is None:
        print("  no default noise variant chosen")
        fails += 1
    elif (sc._parse_variant(dn) or {}).get("statistic") == "median":
        print(f"  the default noise variant is a MEDIAN ({dn}). It reads 0.0")
        print("    on sparse-quote names, which sorts the least tradeable")
        print("    names to the top of the ranking.")
        fails += 1
    elif dn not in FAKE_METRICS:
        print(f"  the default noise variant {dn!r} is not in the database")
        fails += 1

    # THE PIN IS SINGLE-SOURCE. The pipeline builds intraday_metrics around
    # config.INTRADAY_NOISE_COLUMN, and the ranked table must open on the same
    # variant — two panels disagreeing about which noise definition they mean
    # is the defect two copies of a baseline produced on the IV page.
    pin = getattr(scalp_config, "INTRADAY_NOISE_COLUMN", None)
    if pin:
        if pin not in FAKE_METRICS:
            print(f"  the pipeline pins {pin!r} and the fixture does not carry")
            print("    it, so this check cannot see whether the page follows")
            fails += 1
        elif dn != pin:
            print(f"  the pipeline pins {pin!r} and the page defaults to {dn!r}")
            print("    — the ranked table and the intraday view would be about")
            print("    different noise definitions")
            fails += 1

    # It must also degrade rather than invent when nothing preferred exists.
    only_median = sc._default_noise(["noise_bps_tw_mid_10s"])
    if only_median != "noise_bps_tw_mid_10s":
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

    fails += await check_candidates()
    fails += await check_health()
    fails += await check_calibration()
    fails += await check_narrowing()
    fails += await check_geometry_and_series()
    fails += await check_ticker_detail()

    for b in BAD:
        print(f"  {b}")
    fails += len(BAD)

    print(f"\nstates checked: 3, dates: 2, metrics: {len(FAKE_METRICS)}, "
          f"failures: {fails}")
    return 1 if fails else 0


async def _cand(**over):
    """/candidates with every parameter defaulted.

    Called with explicit kwargs at eight sites before this existed, so adding
    a query parameter meant editing all eight — and missing one surfaced as a
    FastAPI Query object reaching float() rather than as anything legible.
    """
    args = dict(date=None, noise=None, columns=None, extra=None, sort=None,
                desc=True, limit=600, spark_sessions=10,
                min_spread_cents=None, min_trades_per_min=None,
                max_noise_bps=None, min_noise_bps=None,
                min_quote_bucket_coverage=None,
                min_spread_bps=None, min_shares_per_min=None,
                filters=None,
                pool=Pool())
    args.update(over)
    return await sc.candidates(**args)


async def check_ticker_detail() -> int:
    """P3, in BOTH scopes — the monthly one shipped rendering blank cells."""
    fails = 0

    for scope in ("sessions", "months"):
        j = await sc.ticker_detail(symbol="FDX", date=None, sessions=10,
                                   scope=scope, pool=Pool())
        where = f"scope={scope}"

        # THE BUG. The heat range was derived on the client from `repeat`,
        # which is empty in monthly scope, so every cell rendered blank while
        # the counts beside them were right and nothing errored. The scale now
        # comes from the server, and it has to be present and usable in BOTH
        # scopes or the chart is invisible again.
        hr = j.get("heat_range")
        if not hr:
            print(f"  {where}: no heat_range — the heatmap has no colour "
                  f"scale and every cell renders blank")
            fails += 1
        elif hr["hi"] <= hr["lo"]:
            print(f"  {where}: heat_range is degenerate ({hr}) — every cell "
                  f"would take the same colour")
            fails += 1

        # missing_roles in BOTH scopes: present in one and absent in the other
        # is how a reader concludes the monthly view reads columns it does not.
        if "missing_roles" not in j:
            print(f"  {where}: no missing_roles, so the header cannot say "
                  f"which columns are not stored intraday")
            fails += 1

        if scope == "months":
            months = j.get("months") or []
            if not months:
                print("  months scope returned no months")
                fails += 1
                continue
            # SESSION COUNTS, which is the point of the monthly view existing.
            if not all(m.get("sessions") for m in months):
                print(f"  a month reports no session count: "
                      f"{[(m['scope'], m['sessions']) for m in months]}")
                fails += 1
            for m in months:
                if not m["buckets"]:
                    print(f"  month {m['scope']} has no buckets")
                    fails += 1
                    break
                if all(b.get("ratio") is None for b in m["buckets"]):
                    print(f"  month {m['scope']}: every bucket's ratio is "
                          f"null, so the chart is blank whatever the scale")
                    fails += 1
                    break
                if not all("sessions" in b for b in m["buckets"]):
                    print(f"  month {m['scope']}: buckets carry no per-bucket "
                          f"session count")
                    fails += 1
                    break
            # The monthly query must read ONLY the resolved intraday columns.
            # Reading one that is not stored is how a scope silently returns
            # nulls for the column the chart is drawn from.
            stored = {n for n, _ in scalp_config.INTRADAY_COLUMNS}
            for name in j["columns"].values():
                if name not in stored:
                    print(f"  months scope resolved {name!r}, which is not an "
                          f"intraday column — it would read as null")
                    fails += 1
        else:
            if not j.get("profile"):
                print("  sessions scope returned no profile")
                fails += 1
            if not j.get("repeat"):
                print("  sessions scope returned no repeatability matrix")
                fails += 1
            if not j.get("traded_hours"):
                print("  sessions scope returned no traded hours, so the "
                      "shading the panel exists for cannot be drawn")
                fails += 1

    # A symbol nobody asked about must be refused, not guessed at.
    try:
        await sc.ticker_detail(symbol="  ", date=None, sessions=10,
                               scope="sessions", pool=Pool())
    except Exception as exc:
        if getattr(exc, "status_code", None) != 400:
            print(f"  an empty symbol raised {type(exc).__name__}, not a 400")
            fails += 1
    else:
        print("  an empty symbol was accepted")
        fails += 1

    return fails


async def check_geometry_and_series() -> int:
    """2.1's traded marker and 2.6's history."""
    fails = 0

    c = await _cand(spark_sessions=0)
    by = {r["symbol"]: r for r in c["rows"]}

    # None, not an empty object, for a name never traded. "No result" and "a
    # result of zero" are different readings and the scatter's colour has to
    # tell them apart -- grey versus green is the whole point of the marker.
    if by.get("BBBB", {}).get("traded") is not None:
        print("  an untraded symbol carries a traded marker")
        fails += 1
    t = by.get("AAAA", {}).get("traded")
    if not t:
        print("  a traded symbol carries no marker — 1.3 asked for this and it")
        print("    waited on the upload; it should be there now")
        fails += 1
    elif t.get("pnl_per_min") is None or t["pnl_per_min"] <= 0:
        print(f"  the profitable traded symbol reads {t.get('pnl_per_min')}")
        fails += 1
    loss = by.get("CCCC", {}).get("traded")
    if not loss or loss["pnl_per_min"] >= 0:
        print("  the losing traded symbol does not read negative, so the "
              "scatter would colour it as a winner")
        fails += 1
    if c.get("n_traded") != 2:
        print(f"  n_traded is {c.get('n_traded')}, not 2")
        fails += 1

    # ── /series ──────────────────────────────────────────────────────────
    s = await sc.series(metrics="ratio_tw_mid_5s_rms,spread_bps_tw",
                        symbols="AAAA,CCCC", sessions=30, pool=Pool())
    if not s.get("series"):
        print(f"  /series returned nothing: {s.get('note')}")
        return fails + 1
    if sorted(s["symbols"]) != ["AAAA", "CCCC"]:
        print(f"  /series ignored the requested symbols: {s['symbols']}")
        fails += 1
    for sym, got in s["series"].items():
        if sorted(got) != ["ratio_tw_mid_5s_rms", "spread_bps_tw"]:
            print(f"  /series returned metrics nobody asked for: {sorted(got)}")
            fails += 1
            break
        for m, arr in got.items():
            if len(arr) != len(s["dates"]):
                print(f"  {sym}/{m} has {len(arr)} points for "
                      f"{len(s['dates'])} dates — a series padded to the date "
                      f"axis is what lets six panels share one x")
                fails += 1
                break

    # SHARED BOUNDS. Six panels on their own scales is six shapes and no
    # comparison, which is the one thing the small multiples are for.
    for m in s["metrics"]:
        b = (s.get("bounds") or {}).get(m)
        if not b or b["min"] >= b["max"]:
            print(f"  no usable shared y-bounds for {m}: {b}")
            fails += 1

    # A symbol with no data must be NAMED, not silently absent -- an empty
    # panel reads as still loading.
    s2 = await sc.series(metrics="ratio_tw_mid_5s_rms", symbols="AAAA,ZZZZ",
                         sessions=30, pool=Pool())
    if "missing" not in s2:
        print("  /series does not report symbols it found no data for")
        fails += 1

    try:
        await sc.series(metrics="", symbols=None, sessions=30, pool=Pool())
    except Exception as exc:
        if getattr(exc, "status_code", None) != 400:
            print(f"  /series with no metric raised {type(exc).__name__}")
            fails += 1
    else:
        print("  /series accepted a request naming no metric")
        fails += 1

    return fails


async def check_narrowing() -> int:
    """2.2 and 2.3: the two panels that run without any trade data."""
    fails = 0

    # ── metric correlation ───────────────────────────────────────────────
    c = await sc.metric_correlation(date=None, family=None, limit=90,
                                    cutoff=0.97, pool=Pool())
    if not c.get("matrix"):
        print(f"  /metric-correlation returned no matrix: {c.get('note')}")
        return 1

    n = len(c["metrics"])
    if len(c["matrix"]) != n or any(len(r) != n for r in c["matrix"]):
        print("  the correlation matrix is not square against its labels")
        fails += 1
    diag = [c["matrix"][i][i] for i in range(n)]
    if any(abs(x - 1.0) > 1e-6 for x in diag):
        print(f"  the diagonal is not 1.0: {diag[:4]} — a metric does not "
              f"correlate perfectly with itself, so the matrix is misaligned "
              f"with its own labels")
        fails += 1
    for i in range(min(n, 12)):
        for j in range(min(n, 12)):
            if abs(c["matrix"][i][j] - c["matrix"][j][i]) > 1e-6:
                print("  the matrix is not symmetric")
                fails += 1
                break

    # A constant metric has no ordering. It must be DROPPED AND NAMED, not
    # carried as a row of zeros that reads as 'uncorrelated with everything'.
    if "spread_bps_tw" not in (c.get("constant") or []):
        print("  a constant metric was not reported as constant. Left in, it "
              "becomes a row of zeros that reads as a finding.")
        fails += 1
    if "spread_bps_tw" in c["metrics"]:
        print("  a constant metric is still in the matrix")
        fails += 1

    # Two metrics that are one linear transform apart must be found redundant.
    pair = {"ratio_tw_mid_10s", "ratio_tw_mid_10s_rms"}
    if not any(pair <= set(g) for g in (c.get("redundant") or [])):
        print(f"  two perfectly co-ranked metrics were not grouped as "
              f"redundant: {c.get('redundant')}. That grouping is the whole "
              f"output of this panel.")
        fails += 1

    # ── rank stability ───────────────────────────────────────────────────
    s = await sc.rank_stability(sessions=10, top_n=20, pool=Pool())
    rows = {r["metric"]: r for r in (s.get("rows") or [])}
    if not rows:
        print(f"  /rank-stability returned no rows: {s.get('note')}")
        return fails + 1

    stable = rows.get("ratio_tw_mid_10s")
    flips = rows.get("ratio_last_mid_30s")
    if not stable or not flips:
        print(f"  expected both fixture metrics, got {sorted(rows)}")
        return fails + 1

    if stable["rank_corr"] < 0.99:
        print(f"  a metric that ranks identically every session scored "
              f"{stable['rank_corr']:.3f}, not ~1.0")
        fails += 1
    if flips["rank_corr"] > -0.99:
        print(f"  a metric whose ordering reverses every session scored "
              f"{flips['rank_corr']:.3f}, not ~-1.0. An estimator that cannot "
              f"tell those two apart is measuring nothing.")
        fails += 1
    if s["rows"][0]["metric"] != "ratio_tw_mid_10s":
        print("  rank stability is not sorted with the stable metric first")
        fails += 1

    # The named worst mover, which is what turns a low average into something
    # to look at.
    if not flips.get("worst_jump") or not flips["worst_jump"].get("symbol"):
        print("  the largest single-name rank move is not named")
        fails += 1
    if stable["top_retention"] is None or stable["top_retention"] < 0.99:
        print(f"  a perfectly stable metric retained "
              f"{stable['top_retention']} of its top 20")
        fails += 1

    return fails


async def check_calibration() -> int:
    """Section 2.4: the rank correlation, and the honesty around it."""
    fails = 0
    j = await sc.calibration(target="dollars_per_min", min_pairs=3, pool=Pool())
    rows = j.get("rows") or []
    if not rows:
        print("  /calibration returned no rows against a populated fake")
        return 1

    # Sorted by |rho|, and the metric the fake made monotonic in the target
    # must be at the top. A correlation that ranks a flat metric first is
    # returning noise.
    if abs(rows[0]["rho"]) < abs(rows[-1]["rho"]):
        print("  /calibration is not sorted by |rho|")
        fails += 1
    if rows[0]["metric"] != "ratio_tw_mid_10s_rms":
        print(f"  the perfectly-ranked metric is not first: {rows[0]['metric']}")
        fails += 1
    if abs(rows[0]["rho"] - 1.0) > 1e-9:
        print(f"  a metric that ranks identically with the target scored "
              f"{rows[0]['rho']}, not 1.0")
        fails += 1

    # A metric with one value everywhere has no ordering. None, not zero —
    # zero is a measurement and would sort it among the real answers.
    flat = [r for r in rows if r["metric"] == "quote_bucket_coverage_5s"]
    if flat:
        print("  a constant metric was given a correlation. It has no ordering "
              "to correlate, and a zero here reads as 'no relationship "
              "found' rather than 'not computable'.")
        fails += 1

    # THE HONESTY. Without this the panel is a ranked list that always has
    # something at the top.
    exp = j.get("expected_by_chance")
    if not exp:
        print("  /calibration does not return an expected-by-chance count.")
        print("    With ~232 metrics and a handful of ticker-days, the top of")
        print("    the list is mostly arithmetic and the panel must say so.")
        fails += 1
    else:
        # It has to shrink as the threshold rises, or it is not a tail
        # probability at all.
        vals = [e["expected"] for e in exp]
        if any(b > a + 1e-12 for a, b in zip(vals, vals[1:])):
            print(f"  expected-by-chance does not fall with the threshold: {vals}")
            fails += 1
    if not j.get("sample"):
        print("  the sample size is not returned, so the weakness is implied")
        print("    rather than visible")
        fails += 1

    # A metric whose SIGN opposes the direction its column claims. Derived
    # from scalp_columns' declared `higher_better`, so it is checked wherever
    # an expectation exists rather than for one metric by name.
    if "contradictions" not in j:
        print("  /calibration does not report metrics whose correlation "
              "contradicts the direction their column claims")
        fails += 1
    if len(j.get("contradictions") or []) < 2:
        print(f"  the fixture produced {len(j.get('contradictions') or [])} "
              f"contradictions; two are needed or the grouping path is never "
              f"exercised")
        fails += 1

    # ── the coherence figure must ALWAYS be reachable ────────────────────
    #
    # THE BUG THIS GUARDS. The first version returned None whenever the mean
    # fell below the grouping threshold, and the client rendered the figure
    # only inside the grouped callout -- so a below-threshold result, an empty
    # query and an exception all produced the same thing: the old ungrouped
    # layout with the number nowhere on the page. Reporting gated behind the
    # conclusion it supports.
    coh = j.get("contradiction_coherence")
    if coh is None:
        print("  contradiction_coherence is None. It must always carry a")
        print("    status — a bare null is indistinguishable from the feature")
        print("    never having shipped.")
        fails += 1
    elif "status" not in coh:
        print(f"  contradiction_coherence has no status: {sorted(coh)}")
        fails += 1
    elif coh["status"] in ("coherent", "separate"):
        if coh.get("mean_abs_rho") is None:
            print(f"  status is {coh['status']} with no mean_abs_rho — the "
                  f"number that decides the question is missing")
            fails += 1
        if not coh.get("pairs"):
            print("  no pairwise agreements returned, so the grouping cannot "
                  "be checked by eye")
            fails += 1
    elif not coh.get("reason"):
        print(f"  status {coh['status']} carries no reason, so a failure is "
              f"indistinguishable from a quiet result")
        fails += 1

    # EVERY status the server can emit must have a rendering path. A status
    # the client does not know about falls through to whatever the default
    # branch is, which is how a silent fall-through starts.
    import re as _re
    src = (ROOT / "app" / "routers" / "equities_scalp.py").read_text(encoding="utf-8")
    js = (ROOT / "static" / "js" / "equities_scalp.js").read_text(encoding="utf-8")
    emitted = set(_re.findall(r'"status":\s*"(\w+)"', src))
    emitted |= set(_re.findall(r'"(\w+)" if mean_abs >= [\d.]+ else "(\w+)"', src)[0]
                   if _re.findall(r'"(\w+)" if mean_abs >= [\d.]+ else "(\w+)"', src)
                   else [])
    for st in sorted(emitted):
        if f"'{st}'" not in js and f'"{st}"' not in js:
            print(f"  the server can emit contradiction status {st!r} and the")
            print("    page has no branch for it — it would fall through to")
            print("    the default wording and say the wrong thing")
            fails += 1

    # The target is interpolated into SQL, so the whitelist is load-bearing.
    try:
        await sc.calibration(target="dollars_per_min; DROP TABLE fills--",
                             min_pairs=3, pool=Pool())
    except Exception as exc:
        if getattr(exc, "status_code", None) != 400:
            print(f"  an unknown target raised {type(exc).__name__}, not a 400")
            fails += 1
    else:
        print("  an arbitrary `target` was accepted and interpolated into SQL")
        fails += 1

    return fails


async def check_health() -> int:
    """The data-health panel: does it actually flag the failure it exists for."""
    fails = 0
    j = await sc.health(sessions=10, trailing=10, pool=Pool())
    rows = j.get("rows") or []
    if not rows:
        print("  /health returned no rows against a populated fake")
        return 1

    # Newest first: at 9am the row that matters is last night's, and putting it
    # last means scrolling to find out whether the data is usable.
    if rows[0]["date"] < rows[-1]["date"]:
        print("  /health rows are oldest-first — last night's session should")
        print("    be the first thing on screen")
        fails += 1

    broken = next((r for r in rows if r["date"] == str(BROKEN_DATE)), None)
    if broken is None:
        print(f"  the broken session {BROKEN_DATE} is not in the response")
        return fails + 1

    # THE WHOLE POINT. A 37% drop in arrivals is the signal that took an hour
    # to find by hand.
    if "arrivals" not in broken["flags"]:
        ch = broken["metrics"].get("arrivals", {}).get("change")
        print(f"  the 37%-low session was NOT flagged (change={ch}). This is")
        print("    the exact failure the panel exists for.")
        fails += 1
    if "coverage" not in broken["flags"]:
        print("  eleven missing symbols did not trip the coverage flag")
        fails += 1
    if broken["missing_n"] != MISSING[BROKEN_DATE] or not broken["missing_sample"]:
        print("  missing symbols are counted but not named — 'eight missing'")
        print("    is a number; naming them is what says whether the refusal")
        print("    was correct")
        fails += 1

    # Eight correctly-refused ETFs must be VISIBLE and NOT flagged. Both
    # halves matter: silent is the thing being fixed, and red is the thing
    # that makes a panel ignorable.
    refused = next((r for r in rows if r["date"] == str(REFUSED_DATE)), None)
    if refused is None:
        print(f"  {REFUSED_DATE} is not in the response")
        fails += 1
    else:
        if refused["missing_n"] != MISSING[REFUSED_DATE]:
            print(f"  {REFUSED_DATE}: eight correctly-refused symbols are not")
            print("    reported — this is the gap that should stop being")
            print("    something to rediscover")
            fails += 1
        if "coverage" in refused["flags"]:
            print(f"  {REFUSED_DATE}: 579 of 587 was flagged. That is the")
            print("    system working, and reddening on it makes the panel")
            print("    ignorable.")
            fails += 1

    # A healthy session must NOT be flagged, or the panel is noise and gets
    # ignored, which is worse than not having it.
    healthy = [r for r in rows if r["date"] != str(BROKEN_DATE)
               and r["metrics"].get("arrivals", {}).get("n_trailing", 0) >= 3]
    noisy = [r["date"] for r in healthy
             if r["flags"] and r["date"] != str(REFUSED_DATE)]
    if noisy:
        print(f"  healthy sessions were flagged: {noisy}. A panel that cries")
        print("    wolf is one nobody reads.")
        fails += 1

    # Self-inclusion, checked on the WINDOW rather than on the value.
    #
    # Comparing the baseline against the day's own reading does not work: the
    # baseline is a MEDIAN, and one outlier among ten barely moves it, so a
    # date sitting inside its own window produces a number indistinguishable
    # from a correct one. The window SIZE is the thing that cannot lie — a
    # baseline built from `trailing` strictly-earlier sessions has at most
    # `trailing` entries, and including the date itself makes it one more.
    trailing_n = j.get("trailing")
    for r in rows:
        n = (r["metrics"].get("arrivals") or {}).get("n_trailing")
        if n is not None and trailing_n is not None and n > trailing_n:
            print(f"  {r['date']}: baseline has {n} sessions for a window of")
            print(f"    {trailing_n} — the date is inside the history it is")
            print("    being scored against")
            fails += 1
            break

    # Fetch rejections are not persisted, and the response must SAY that
    # rather than return zero, which would read as "there were none".
    if j.get("fetch_rejects") is not None:
        print("  /health reports a fetch-reject count, but nothing in the")
        print("    pipeline persists one")
        fails += 1
    if not j.get("fetch_rejects_note"):
        print("  the absence of fetch-reject data is silent")
        fails += 1

    return fails


async def check_candidates() -> int:
    """The ranked table: resolution, the filter join, and the pass count."""
    fails = 0

    c = await _cand()

    if not c.get("rows"):
        print("  /candidates returned no rows against a populated fake")
        return fails + 1

    keys = [col["key"] for col in c["columns"]]

    # THE VARIANT MOVES FIVE COLUMNS. Noise alone would leave coverage and the
    # decomposition at whatever horizon happened to resolve first, which reads
    # as a comparison and is not one.
    v = c["variant"]
    for k in ("noise", "ratio", "coverage", "move_rate", "move_bps"):
        if k not in keys:
            continue
        got = next(col["metric"] for col in c["columns"] if col["key"] == k)
        if f"_{v['horizon_s']}s" not in got:
            print(f"  {k} resolved to {got}, which is not at the selected "
                  f"{v['horizon_s']}s horizon")
            fails += 1

    # Every filter must be joined to a column, or the pass count is a number
    # about filters that did not all run.
    for fk in scalp_config.DEFAULT_FILTERS:
        if fk not in sc._FILTER_ROLES:
            print(f"  the filter {fk!r} has no column to apply to — it would")
            print("    move a slider and change nothing")
            fails += 1
    if c.get("inert_filters"):
        print(f"  filters that did not run: {c['inert_filters']} — with the "
              f"full fake catalog every one should apply")
        fails += 1

    # Failing rows are RETAINED, below the passing ones. They are the only
    # evidence that can say whether a threshold sits in the right place.
    if c["n_total"] <= c["n_pass"] and c["n_pass"] == len(c["rows"]):
        pass                                 # everything passed; fine
    seen_fail = False
    for r in c["rows"]:
        if not r["passes"]:
            seen_fail = True
        elif seen_fail:
            print("  a passing row sorts below a failing one")
            fails += 1
            break

    # Nulls last in both directions, or a name with no measurement lands where
    # the best one belongs.
    for descending in (True, False):
        cc = await _cand(sort="ratio", desc=descending, spark_sessions=0)
        vals = [r["values"].get("ratio") for r in cc["rows"] if r["passes"]]
        nulls = [i for i, x in enumerate(vals) if x is None]
        reals = [i for i, x in enumerate(vals) if x is not None]
        if nulls and reals and min(nulls) < max(reals):
            print(f"  desc={descending}: a null ratio sorts above a real one")
            fails += 1

    # A role that cannot resolve is REPORTED, not dropped in silence.
    thin = await _cand(columns="ratio,nonsense", limit=10, spark_sessions=0)
    if "missing" not in thin:
        print("  /candidates does not report which roles failed to resolve")
        fails += 1

    # An unknown extra column must be ignored rather than becoming a column of
    # nulls -- the exact failure the no-hardcoding rule exists for.
    ex = await _cand(columns="ratio", extra="not_a_real_metric", limit=10, spark_sessions=0)
    if any(col["key"] == "not_a_real_metric" for col in ex["columns"]):
        print("  an extra column that is not in the database became a column")
        print("    of nulls rather than being refused")
        fails += 1

    # ── the ticker column must sort BY TICKER ────────────────────────────
    #
    # `symbol` is not a metric column, and the fallback treated it as an
    # unknown key and sorted by ratio instead — so clicking the ticker header
    # reordered the rows into an order with no relation to the column clicked.
    for descending in (False, True):
        ss = await _cand(sort="symbol", desc=descending, spark_sessions=0)
        syms = [r["symbol"] for r in ss["rows"] if r["passes"]]
        want = sorted(syms, reverse=descending)
        if syms != want:
            print(f"  sort=symbol desc={descending} gave {syms}, not {want} — "
                  f"the ticker header sorts by something else")
            fails += 1

    # ── the dashboard's own filters ──────────────────────────────────────
    #
    # Spread in BPS is the primary screen: ten cents on a $100 stock and on a
    # $1,000 stock are the same cents and a tenth of the opportunity.
    for fk in ("min_spread_bps", "min_shares_per_min"):
        if fk not in c.get("thresholds", {}):
            print(f"  {fk} is not in the returned thresholds, so the page "
                  f"cannot draw its slider")
            fails += 1
        if fk not in (c.get("rejected") or {}):
            print(f"  {fk} rejected nothing and is not reported — it did not "
                  f"run, and a pass count that hides that is a number about "
                  f"filters that did not all apply")
            fails += 1
        tight = await _cand(spark_sessions=0, **{fk: 1e9})
        if tight["n_pass"] != 0:
            print(f"  an impossible {fk} still passed {tight['n_pass']} rows")
            fails += 1

    # ── every filter explains itself ─────────────────────────────────────
    #
    # The page puts a "?" beside each slider instead of a paragraph above the
    # table, and it can only do that if the server says what each threshold
    # thresholds and what the metric means.
    fm = c.get("filter_meta") or {}
    for fk in sc._FILTER_ROLES:
        m = fm.get(fk)
        if not m:
            print(f"  no filter_meta for {fk!r} — the slider would have no "
                  f"definition behind its '?'")
            fails += 1
            continue
        if not m.get("why"):
            print(f"  filter {fk!r} carries no reason for existing")
            fails += 1
        if m.get("applied") and not m.get("definition"):
            print(f"  filter {fk!r} thresholds {m.get('metric')!r} with no "
                  f"definition from metric_docs")
            fails += 1
        if (m.get("source") == "dashboard") != (fk in sc.DASHBOARD_FILTERS):
            print(f"  filter {fk!r} is mislabelled as {m.get('source')!r} — "
                  f"the page would present the dashboard's opinion as the "
                  f"pipeline's, or the reverse")
            fails += 1

    # ── trade price as a first-class alternative ─────────────────────────
    both = await _cand(columns="ratio,ratio_trade,noise,noise_trade",
                       spark_sessions=0)
    keys = [col["key"] for col in both["columns"]]
    for k in ("ratio", "ratio_trade", "noise", "noise_trade"):
        if k not in keys:
            print(f"  {k} did not resolve, so the midpoint and trade-price "
                  f"readings cannot be shown side by side")
            fails += 1
    tr = next((col for col in both["columns"] if col["key"] == "ratio_trade"),
              None)
    if tr and "trade_price" not in tr["metric"]:
        print(f"  ratio_trade resolved to {tr['metric']!r}, which is not the "
              f"trade-price basis — it would move with the variant selector "
              f"instead of holding still beside it")
        fails += 1

    # ── ANY column is screenable ─────────────────────────────────────────
    #
    # The principle: the pipeline stores every symbol whether it passes or
    # not, so choosing four metrics in advance to be filterable defeats the
    # reason nothing is dropped. Twice a metric was visible and unfilterable.
    lo = (c.get("col_ranges") or {}).get("spread_bps", {}).get("min")
    hi = (c.get("col_ranges") or {}).get("spread_bps", {}).get("max")
    if lo is None or hi is None:
        print("  no observed range for a displayed column, so the filter pane "
              "would have to guess its slider bounds")
        fails += 1
    else:
        mid = (lo + hi) / 2
        cc = await _cand(filters=f"spread_bps:min:{mid}", spark_sessions=0)
        if cc["n_pass"] >= c["n_pass"]:
            print(f"  a custom constraint at the midpoint of the observed "
                  f"range excluded nothing ({cc['n_pass']} vs {c['n_pass']})")
            fails += 1
        if not cc.get("constraints"):
            print("  /candidates does not report which constraints are active")
            fails += 1

    # SCREENING ON A COLUMN PULLS IT IN. "Filterable" and "visible" must be
    # the same set, or the page has two lists that drift.
    hidden = "off_mid_bps"
    pulled = await _cand(columns="ratio", filters=f"{hidden}:min:0",
                         spark_sessions=0)
    if not any(col["key"] == hidden for col in pulled["columns"]):
        print(f"  screening on {hidden!r} did not add it to the table — "
              f"filterable and visible are two different sets")
        fails += 1

    # A derived column must be computed, screenable and labelled as derived.
    dv = next((col for col in c["columns"] if col.get("derived")), None)
    if not dv:
        print("  no derived column in the response — dollar volume per minute "
              "is shares/min times price and neither is a product")
        fails += 1
    else:
        vals = [r["values"].get(dv["key"]) for r in c["rows"]]
        if not any(v is not None for v in vals):
            print(f"  derived column {dv['key']} is null for every row")
            fails += 1
        dd = await _cand(filters=f"{dv['key']}:min:1e12", spark_sessions=0)
        if dd["n_pass"] != 0:
            print(f"  an impossible threshold on the derived column still "
                  f"passed {dd['n_pass']} rows")
            fails += 1

    # A malformed clause must be refused, not skipped — a silently dropped
    # screen makes the pass count a number about filters that did not run.
    for bad_clause in ("spread_bps:between:4", "spread_bps:min:abc", "nonsense"):
        try:
            await _cand(filters=bad_clause, spark_sessions=0)
        except Exception as exc:
            if getattr(exc, "status_code", None) != 400:
                print(f"  {bad_clause!r} raised {type(exc).__name__}, not 400")
                fails += 1
        else:
            print(f"  malformed filter {bad_clause!r} was accepted and "
                  f"silently ignored")
            fails += 1

    # A constraint on a column that does not resolve is INERT and reported.
    inert = await _cand(filters="not_a_metric_at_all:min:1", spark_sessions=0)
    if not inert.get("inert_filters"):
        print("  a constraint naming an unresolvable column was neither "
              "applied nor reported")
        fails += 1

    # A moved slider has to reach the response, or the page is showing the
    # pipeline's threshold while claiming to show the user's.
    tight = await _cand(spark_sessions=0, min_spread_cents=999.0)
    if tight["thresholds"]["min_spread_cents"] != 999.0:
        print("  a supplied threshold did not override the pipeline default")
        fails += 1
    if tight["n_pass"] != 0:
        print("  an impossible threshold still passed rows")
        fails += 1

    return fails


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

        if (sc._parse_variant(j.get("default_noise") or "")
                or {}).get("statistic") == "median":
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
