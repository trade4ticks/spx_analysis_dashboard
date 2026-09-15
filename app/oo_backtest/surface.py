"""Surface metrics for the OO/Mesosim Backtest page: public.surface_metrics_core.

The table (verified on the VPS, 2026-09-15): 110,258 rows, 458 DOUBLE
PRECISION metric columns keyed (trade_date, quote_time), PK on both and an
index on trade_date. It is CLEAN in a way index_ohlc is not -- non-trading
days have no rows at all, every session has exactly 78 bars (09:35..16:00),
no NaN, no zeros -- so none of market.py's validity machinery applies here.
Bars are start-labeled and aligned with index_ohlc (spot at 09:35 equals
index_ohlc.spx_open at 09:35 at every bar).

COVERAGE differs by metric form (levels from 2020-01-02, z-scores from
2021-04-05 at the time of writing) and moves as history is backfilled, so it
is never hardcoded: coverage() reads each metric's first non-null date from
the table and caches it until the table's date range changes. Before its
first date a metric is NULL in rows that exist; after it, it never gaps.

surface_metrics_catalog (PK column_name) describes every column. The ranked
set is the catalog minus EXCLUDED_FAMILIES, intersected with the table's real
DOUBLE PRECISION columns -- a column name reaches SQL only from that set, so
nothing a request sends is ever interpolated.

THE ENTRY BAR -- NOT YET CONFIRMED. Which bar a trade reads depends on
whether a metric stamped at quote_time T is known AT T (a snapshot at the bar
start, like spot = the bar's open) or only at T+5 (computed over the bar).
That is unconfirmed (2026-09-15), so the rule is a named choice, not an
assumption baked into the SQL:

  at_or_before_entry  the last bar with quote_time <= entry time, the entry's
                      own bar -- correct only if metrics are point-in-time
                      at their quote_time
  previous_bar        the last bar with quote_time <= entry time - 5 min --
                      correct if a bar's metrics include data through its end

BAR_RULE holds the one in use and LOOKAHEAD_CONFIRMED stays False until the
data owner confirms; every response carries both so the page can say so.
Either way the bar is on the ENTRY DATE only: a 09:30 entry has no bar at or
before it (the table starts 09:35) and gets null, never the prior session.
"""
from __future__ import annotations

import asyncio
import logging
import time as _time
from datetime import date, datetime, time

log = logging.getLogger(__name__)

CORE = "surface_metrics_core"
CATALOG = "surface_metrics_catalog"

# Not ranked (data owner, 2026-09-15): the 4 meta columns, and price levels --
# spot and the five forward_* -- which over years mostly measure time trend.
# log_ret is kept.
EXCLUDED_FAMILIES = frozenset({"meta", "spot", "forward"})

BAR_RULES = {
    "at_or_before_entry": "quote_time <= t.tm",
    "previous_bar": "quote_time <= t.tm - INTERVAL '5 minutes'",
}
BAR_RULE = "at_or_before_entry"
LOOKAHEAD_CONFIRMED = False

CATALOG_FIELDS = ("column_name", "family", "tenor", "wing", "form", "base_column", "units",
                  "description", "formula")

CATALOG_SQL = f"SELECT {', '.join(CATALOG_FIELDS)} FROM {CATALOG} ORDER BY column_name"

TABLE_COLUMNS_SQL = """
SELECT column_name, data_type FROM information_schema.columns
WHERE table_schema = 'public' AND table_name = $1
"""

# Index lookups only (PK / trade_date index): a backfill moves the first date,
# a new session moves the last. Either rebuilds the coverage cache.
RANGE_KEY_SQL = f"SELECT min(trade_date) AS first_date, max(trade_date) AS last_date FROM {CORE}"


def quote_ident(name: str) -> str:
    """A Postgres identifier. Names come only from the validated metric set;
    this is the belt to that braces."""
    return '"' + name.replace('"', '""') + '"'


def metric_set(catalog: list[dict], table_columns: dict[str, str]) -> tuple[list[dict], dict]:
    """(ranked catalog rows, report). A catalog row is ranked when its family
    is not excluded and the table really has it as double precision."""
    ranked, missing, wrong_type, excluded = [], [], [], []
    for r in catalog:
        c = r["column_name"]
        if r["family"] in EXCLUDED_FAMILIES:
            excluded.append(c)
        elif c not in table_columns:
            missing.append(c)
        elif table_columns[c] != "double precision":
            wrong_type.append(c)
        else:
            ranked.append(r)
    in_catalog = {r["column_name"] for r in catalog}
    uncatalogued = sorted(c for c in table_columns if c not in in_catalog)
    return ranked, {"catalog_rows": len(catalog), "ranked": len(ranked), "excluded": len(excluded),
                    "missing_from_table": missing, "wrong_type": wrong_type, "uncatalogued": uncatalogued}


def coverage_sql(columns: list[str]) -> str:
    """First non-null trade_date per column, one statement. Each scalar
    subquery walks the PK in date order and stops at the first non-null row,
    so a column costs roughly the rows before its coverage starts, not the
    table. (A single min() FILTER pass reads all 545 MB; the timing script
    measures both.)"""
    parts = [f"(SELECT trade_date FROM {CORE} WHERE {quote_ident(c)} IS NOT NULL "
             f"ORDER BY trade_date, quote_time LIMIT 1) AS {quote_ident(c)}" for c in columns]
    return "SELECT " + ",\n       ".join(parts)


def coverage_scan_sql(columns: list[str]) -> str:
    """The same answer in one full scan -- kept for the timing comparison."""
    parts = [f"min(trade_date) FILTER (WHERE {quote_ident(c)} IS NOT NULL) AS {quote_ident(c)}" for c in columns]
    return "SELECT " + ",\n       ".join(parts) + f"\nFROM {CORE}"


def entry_sql(columns: list[str], rule: str = None) -> str:
    """$1 date[], $2 time[] -- one element per DISTINCT (entry date, entry
    time). One bar per entry for every column at once: the table has no holes,
    so unlike index_ohlc no per-series bar search is needed."""
    rule = rule or BAR_RULE
    if rule not in BAR_RULES:
        raise ValueError(f"unknown bar rule {rule!r}")
    cols = ", ".join(f"b.{quote_ident(c)}" for c in columns)
    inner = ", ".join(quote_ident(c) for c in columns)
    return f"""
SELECT t.d AS entry_date, t.tm AS entry_time, b.quote_time AS bar_time{', ' + cols if cols else ''}
FROM unnest($1::date[], $2::time[]) AS t(d, tm)
LEFT JOIN LATERAL (
    SELECT quote_time{', ' + inner if inner else ''}
    FROM {CORE}
    WHERE trade_date = t.d AND {BAR_RULES[rule]}
    ORDER BY quote_time DESC
    LIMIT 1
) b ON true"""


# ── catalog + coverage cache ────────────────────────────────────────────────

_CACHE: dict = {"key": None, "catalog": None, "metrics": None, "report": None, "coverage": None,
                "built_at": None, "build_s": None}
_LOCK = asyncio.Lock()


async def get_catalog(pool) -> dict:
    """{metrics: [catalog row + min_date], report, first_date, last_date, ...}.
    Rebuilt when the table's first or last date moves."""
    async with pool.acquire() as conn:
        k = await conn.fetchrow(RANGE_KEY_SQL)
    key = (k["first_date"], k["last_date"])
    async with _LOCK:
        if _CACHE["key"] != key or _CACHE["metrics"] is None:
            t0 = _time.monotonic()
            async with pool.acquire() as conn:
                catalog = [dict(r) for r in await conn.fetch(CATALOG_SQL)]
                table_cols = {r["column_name"]: r["data_type"] for r in await conn.fetch(TABLE_COLUMNS_SQL, CORE)}
                ranked, report = metric_set(catalog, table_cols)
                cols = [r["column_name"] for r in ranked]
                cov = dict(await conn.fetchrow(coverage_sql(cols))) if cols else {}
            metrics = [{**r, "min_date": cov.get(r["column_name"]).isoformat() if cov.get(r["column_name"]) else None}
                       for r in ranked]
            _CACHE.update(key=key, catalog=catalog, metrics=metrics, report=report,
                          built_at=datetime.now().isoformat(timespec="seconds"),
                          build_s=round(_time.monotonic() - t0, 2))
            log.info("oo-backtest surface catalog: %d ranked of %d catalog rows, table %s..%s, built in %.2fs; "
                     "missing %s, wrong type %s, uncatalogued %s", report["ranked"], report["catalog_rows"],
                     key[0], key[1], _CACHE["build_s"], report["missing_from_table"], report["wrong_type"],
                     report["uncatalogued"])
    return {"metrics": _CACHE["metrics"], "report": _CACHE["report"],
            "first_date": key[0].isoformat() if key[0] else None,
            "last_date": key[1].isoformat() if key[1] else None,
            "built_at": _CACHE["built_at"], "build_s": _CACHE["build_s"],
            "bar_rule": BAR_RULE, "lookahead_confirmed": LOOKAHEAD_CONFIRMED}


# ── requests from the page ──────────────────────────────────────────────────

MAX_TRADES = 20000


def parse_trades(raw, with_pnl: bool) -> tuple[list[tuple], dict]:
    """[[date, time, pnl?], ...] from the page -> [(date|None, time|None, pnl?)].

    Rows are kept in order (the response aligns to them). A row whose date or
    time does not parse is kept with None and counted -- it gets no bar, and the
    page is told how many, rather than the row silently vanishing."""
    from app.oo_backtest.market import normalize_entry_time
    if not isinstance(raw, list):
        raise ValueError("trades must be a list of [date, time" + (", pnl]" if with_pnl else "]"))
    if len(raw) > MAX_TRADES:
        raise ValueError(f"at most {MAX_TRADES} trades per request ({len(raw)} sent)")
    out, bad_date, bad_time, bad_pnl = [], 0, 0, 0
    for row in raw:
        if not isinstance(row, (list, tuple)) or len(row) < (3 if with_pnl else 2):
            raise ValueError(f"each trade must be [date, time{', pnl' if with_pnl else ''}], got {row!r}")
        try:
            d = date.fromisoformat(str(row[0]))
        except (TypeError, ValueError):
            d, bad_date = None, bad_date + 1
        t = normalize_entry_time(row[1])
        if t is None:
            bad_time += 1
        tt = time.fromisoformat(t) if t else None
        if with_pnl:
            try:
                p = float(row[2])
                if p != p or p in (float("inf"), float("-inf")):
                    raise ValueError
            except (TypeError, ValueError):
                p, bad_pnl = None, bad_pnl + 1
            out.append((d, tt, p))
        else:
            out.append((d, tt))
    return out, {"trades": len(out), "bad_date": bad_date, "bad_time": bad_time, "bad_pnl": bad_pnl}


async def entry_values(pool, trades: list[tuple], columns: list[str], rule: str = None) -> tuple[list[dict], dict]:
    """Per trade (in order): {bar_time, <column>: value|None, ...}, plus a report.
    Distinct (date, time) pairs are looked up once."""
    keys = sorted({(t[0], t[1]) for t in trades if t[0] is not None and t[1] is not None})
    t0 = _time.monotonic()
    rows = []
    if keys:
        async with pool.acquire() as conn:
            rows = await conn.fetch(entry_sql(columns, rule), [k[0] for k in keys], [k[1] for k in keys])
    query_s = round(_time.monotonic() - t0, 3)
    by_key = {(r["entry_date"], r["entry_time"]): r for r in rows}
    out, no_bar = [], 0
    for t in trades:
        r = by_key.get((t[0], t[1]))
        if r is None or r["bar_time"] is None:
            no_bar += 1
            out.append({"bar_time": None, **{c: None for c in columns}})
        else:
            out.append({"bar_time": r["bar_time"], **{c: r[c] for c in columns}})
    return out, {"distinct_entries": len(keys), "no_bar": no_bar, "query_s": query_s,
                 "bar_rule": rule or BAR_RULE, "lookahead_confirmed": LOOKAHEAD_CONFIRMED}
