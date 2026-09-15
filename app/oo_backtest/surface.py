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
is never hardcoded: get_catalog() reads each metric's first non-null date
from the table and caches it until the table changes -- its first date, last
date OR ROW COUNT. The first version keyed on the dates alone and went stale
during a live backfill: a session inserted mid-range moves neither date
(caught on the VPS, 2026-09-15, as two timing runs 50 s apart disagreeing by
one trade). The count catches inserts and deletes; it does not see an UPDATE
that fills values into rows already there. Before its first date a metric is
NULL in rows that exist; after it, it never gaps.

surface_metrics_catalog (PK column_name) describes every column. The ranked
set is the catalog minus EXCLUDED_FAMILIES, intersected with the table's real
DOUBLE PRECISION columns -- a column name reaches SQL only from that set, so
nothing a request sends is ever interpolated.

THE ENTRY BAR: the last bar on the entry date with quote_time <= entry
time -- the entry's own bar. No lookahead: the data owner confirmed
(2026-09-15) that every metric is point-in-time at its quote_time and every
window looks backward from the bar's timestamp. (A "previous bar" rule was
carried while that was open and has been removed.) The bar is on the ENTRY
DATE only: a 09:30 entry has no bar at or before it (the table starts 09:35)
and gets null, never the prior session.
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

BAR_RULE = "at_or_before_entry"
LOOKAHEAD_CONFIRMED = True

# How the ranking chart colours families: eight hues from the dataviz reference
# palette (dark steps, validated as a set). There are ~14 families, more than
# eight hues keep apart, so related families share a hue; the legend toggles
# each family on its own and the hover names it. A family not listed here is
# drawn grey under "Other" -- never given an invented hue. Kept here, not in
# the page JS, so the page names no metric family (the same rule as the
# metric registry).
FAMILY_GROUPS = [
    {"label": "IV", "color": "#3987e5", "families": ["iv"]},
    {"label": "Skew", "color": "#d95926", "families": ["skew"]},
    {"label": "Term structure", "color": "#199e70", "families": ["term_slope", "term_ratio"]},
    {"label": "Convexity", "color": "#c98500", "families": ["convex"]},
    {"label": "Risk reversal", "color": "#d55181", "families": ["rr"]},
    {"label": "VIX index", "color": "#008300", "families": ["vix", "vix_basis"]},
    {"label": "Realized & VRP", "color": "#e66767", "families": ["rv", "vrp", "vrp_ratio"]},
    {"label": "Spot dynamics", "color": "#9085e9", "families": ["log_ret", "spot_vol", "vov"]},
]
OTHER_GROUP = {"label": "Other", "color": "#8a8a8a"}
FORM_LABELS = {"level": "Level", "chg_d": "Daily chg", "chg_1w": "Weekly chg", "z": "Z-score"}

CATALOG_FIELDS = ("column_name", "family", "tenor", "wing", "form", "base_column", "units",
                  "description", "formula")

CATALOG_SQL = f"SELECT {', '.join(CATALOG_FIELDS)} FROM {CATALOG} ORDER BY column_name"

TABLE_COLUMNS_SQL = """
SELECT column_name, data_type FROM information_schema.columns
WHERE table_schema = 'public' AND table_name = $1
"""

# The coverage cache key. min/max are index lookups; count(*) is an
# index-only scan of the trade_date index (~110k entries). A backfill moves
# the first date, a new session the last, a mid-range insert only the count.
RANGE_KEY_SQL = f"""SELECT min(trade_date) AS first_date, max(trade_date) AS last_date,
       count(*) AS row_count FROM {CORE}"""


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


def entry_sql(columns: list[str]) -> str:
    """$1 date[], $2 time[] -- one element per DISTINCT (entry date, entry
    time). One bar per entry for every column at once: the table has no holes,
    so unlike index_ohlc no per-series bar search is needed."""
    cols = ", ".join(f"b.{quote_ident(c)}" for c in columns)
    inner = ", ".join(quote_ident(c) for c in columns)
    return f"""
SELECT t.d AS entry_date, t.tm AS entry_time, b.quote_time AS bar_time{', ' + cols if cols else ''}
FROM unnest($1::date[], $2::time[]) AS t(d, tm)
LEFT JOIN LATERAL (
    SELECT quote_time{', ' + inner if inner else ''}
    FROM {CORE}
    WHERE trade_date = t.d AND quote_time <= t.tm
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
    key = (k["first_date"], k["last_date"], k["row_count"])
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
            "last_date": key[1].isoformat() if key[1] else None, "row_count": key[2],
            "built_at": _CACHE["built_at"], "build_s": _CACHE["build_s"],
            "bar_rule": BAR_RULE, "lookahead_confirmed": LOOKAHEAD_CONFIRMED,
            "family_groups": FAMILY_GROUPS, "other_group": OTHER_GROUP, "form_labels": FORM_LABELS}


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


async def entry_matrix(pool, trades: list[tuple], columns: list[str]):
    """(bar_times, X, report): per trade in order, its entry bar's time (None
    if no bar) and a trades x columns float matrix, NaN where missing. Distinct
    (date, time) pairs are looked up once; the matrix is built from the
    distinct rows and gathered, never cell by cell."""
    import numpy as np
    keys = sorted({(t[0], t[1]) for t in trades if t[0] is not None and t[1] is not None})
    t0 = _time.monotonic()
    rows = []
    if keys:
        async with pool.acquire() as conn:
            rows = await conn.fetch(entry_sql(columns), [k[0] for k in keys], [k[1] for k in keys])
    query_s = round(_time.monotonic() - t0, 3)
    k = len(columns)
    # Row 0 of `distinct` is the all-NaN row for trades with no lookup at all.
    distinct = np.full((len(rows) + 1, k), np.nan)
    pos, bar_of = {}, [None]
    for i, r in enumerate(rows, start=1):
        pos[(r["entry_date"], r["entry_time"])] = i
        bar_of.append(r["bar_time"])
        if r["bar_time"] is not None:
            distinct[i] = [np.nan if v is None else v for v in tuple(r)[3:]]
    gather = np.fromiter((pos.get((t[0], t[1]), 0) for t in trades), dtype=np.int64, count=len(trades))
    X = distinct[gather]
    bar_times = [bar_of[g] for g in gather.tolist()]
    no_bar = sum(1 for b in bar_times if b is None)
    return bar_times, X, {"distinct_entries": len(keys), "no_bar": no_bar, "with_bar": len(trades) - no_bar,
                          "query_s": query_s, "bar_rule": BAR_RULE, "lookahead_confirmed": LOOKAHEAD_CONFIRMED}


async def entry_values(pool, trades: list[tuple], columns: list[str]) -> tuple[list[dict], dict]:
    """Per trade (in order): {bar_time, <column>: value|None, ...}, plus a
    report. For a few columns; the ranking uses entry_matrix."""
    bar_times, X, report = await entry_matrix(pool, trades, columns)
    out = []
    for bt, row in zip(bar_times, X.tolist()):
        out.append({"bar_time": bt, **{c: (None if v != v else v) for c, v in zip(columns, row)}})
    return out, report
