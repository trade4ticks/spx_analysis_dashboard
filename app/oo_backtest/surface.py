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

# How an added metric row shows its catalog `units`. `scale` is applied to the
# values when the page fetches them, so bins, axes and (P6d) filters all work
# in display units: vol_decimal 0.1406 -> 14.06 vol points (data owner,
# 2026-09-15), a log return 0.0037 -> 0.37 %. A unit not listed uses DEFAULT.
UNIT_FORMATS = {
    "vol_decimal": {"scale": 100, "decimals": 2, "suffix": "vol pts"},
    "z_score": {"scale": 1, "decimals": 2, "suffix": "z"},
    "slope": {"scale": 1, "decimals": 4, "suffix": ""},
    "ratio": {"scale": 1, "decimals": 3, "suffix": ""},
    "log_return": {"scale": 100, "decimals": 2, "suffix": "%"},
    "corr": {"scale": 1, "decimals": 3, "suffix": ""},
}
DEFAULT_UNIT_FORMAT = {"scale": 1, "decimals": 4, "suffix": ""}

# Bins for an added row: the Premium rule (~24 bins over p1..p99, outliers in
# end buckets) with "nice" steps -- 1, 2, 2.5, 5 x 10^k around the span -- since
# these scales run from 0.0001 to thousands.
ROW_AUTO_BINS = {"steps": "nice", "targetBins": 24, "pLo": 1, "pHi": 99}

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
            "family_groups": FAMILY_GROUPS, "other_group": OTHER_GROUP, "form_labels": FORM_LABELS,
            "unit_formats": UNIT_FORMATS, "default_unit_format": DEFAULT_UNIT_FORMAT, "row_auto_bins": ROW_AUTO_BINS}


# ── requests from the page ──────────────────────────────────────────────────

# WHAT A REQUEST MAY CARRY (2026-10-19). A surface value is a function of
# the entry (date, time) alone, so the page sends DISTINCT ENTRY BARS -- a
# 90-variant sweep's 288,498 trades are ~3,400 bars, and that count does not
# grow with the variants. The old cap, 20,000 TRADES per request, refused
# exactly that sweep. Measured on a synthetic 85,800-bar x 452-metric table
# (dev machine; the VPS ran a sweep load ~1.75x slower):
#   /surface/values  one column: 85,800 bars looked up in 0.61 s. Nothing
#                    here is heavy; the bound is the request itself, ~28
#                    bytes a bar -> 250,000 bars is ~7 MB of JSON. The table
#                    holds ~110,000 five-minute bars in all.
#   /surface/rank    every metric: memory is the bar matrix (452 x 8 bytes a
#                    bar, filled in chunks -- QUERY_CHUNK_BARS) plus the
#                    stats' blocks: 195 MB at 3,400 bars, 537 MB at 83,000;
#                    ~630 MB at 100,000. The query was 7 s at 83,000 bars.
#                    STATS TIME scales with TRADES, not bars: 10.7 s at
#                    300,000, 22.3 s at 600,000 -- so trades get their own
#                    cap, set so the worst case (100,000 bars, 400,000
#                    trades) stays near 30 s here, ~55 s on the VPS, inside
#                    the tunnel's 100 s.
MAX_VALUE_BARS = 250_000
MAX_RANK_BARS = 100_000
MAX_RANK_TRADES = 400_000


def _too_many_bars(n: int, cap: int) -> ValueError:
    return ValueError(f"at most {cap:,} distinct entry bars (date, time) per request ({n:,} sent)")


def parse_bars(raw, cap: int) -> tuple[list[tuple], dict]:
    """[[date, time], ...] -- the page's DISTINCT entry bars -- -> [(date|None,
    time|None)] in order (the response aligns to them), or ValueError over
    `cap` distinct bars. A bar that does not parse is kept with None and
    counted, as a trade was."""
    rows, rep = parse_trades(raw, with_pnl=False, max_rows=None)
    distinct = len(set(rows))
    if distinct > cap:
        raise _too_many_bars(distinct, cap)
    return rows, {"bars": len(rows), "bad_date": rep["bad_date"], "bad_time": rep["bad_time"]}


def parse_trades(raw, with_pnl: bool, max_rows: int | None = MAX_RANK_TRADES,
                 max_bars: int | None = None) -> tuple[list[tuple], dict]:
    """[[date, time, pnl?], ...] from the page -> [(date|None, time|None, pnl?)].

    Rows are kept in order (the response aligns to them). A row whose date or
    time does not parse is kept with None and counted -- it gets no bar, and the
    page is told how many, rather than the row silently vanishing."""
    from app.oo_backtest.market import normalize_entry_time
    if not isinstance(raw, list):
        raise ValueError("trades must be a list of [date, time" + (", pnl]" if with_pnl else "]"))
    if max_rows is not None and len(raw) > max_rows:
        raise ValueError(f"at most {max_rows:,} trades per request ({len(raw):,} sent)")
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
    if max_bars is not None:
        distinct = len({(r[0], r[1]) for r in out})
        if distinct > max_bars:
            raise _too_many_bars(distinct, max_bars)
    return out, {"trades": len(out), "bad_date": bad_date, "bad_time": bad_time, "bad_pnl": bad_pnl}


def parse_rank_request(body: dict) -> tuple[list[tuple], dict]:
    """[(date, time, pnl)] per TRADE, from either request form:
      {"bars": [[date, time], ...], "bar_of": [i, ...], "pnl": [p, ...]}  -- the
          page's: distinct bars once, each trade an index into them
      {"trades": [[date, time, pnl], ...]}                              -- the old one
    Capped on distinct bars and on trades (see MAX_RANK_BARS / _TRADES)."""
    if "bars" not in body:
        return parse_trades(body.get("trades"), with_pnl=True, max_rows=MAX_RANK_TRADES, max_bars=MAX_RANK_BARS)
    bars, brep = parse_bars(body.get("bars"), MAX_RANK_BARS)
    bar_of, pnl = body.get("bar_of"), body.get("pnl")
    if not isinstance(bar_of, list) or not isinstance(pnl, list) or len(bar_of) != len(pnl):
        raise ValueError("bar_of and pnl must be lists of the same length, one entry per trade")
    if len(bar_of) > MAX_RANK_TRADES:
        raise ValueError(f"at most {MAX_RANK_TRADES:,} trades per request ({len(bar_of):,} sent)")
    out, bad_pnl = [], 0
    for b, p in zip(bar_of, pnl):
        if not isinstance(b, int) or isinstance(b, bool) or not 0 <= b < len(bars):
            raise ValueError(f"bar_of holds {b!r}, not an index into the {len(bars):,} bars")
        try:
            p = float(p)
            if p != p or p in (float("inf"), float("-inf")):
                raise ValueError
        except (TypeError, ValueError):
            p, bad_pnl = None, bad_pnl + 1
        out.append((bars[b][0], bars[b][1], p))
    return out, {"trades": len(out), "bars": len(bars), "bad_date": brep["bad_date"], "bad_time": brep["bad_time"],
                 "bad_pnl": bad_pnl}


async def entry_matrix(pool, trades: list[tuple], columns: list[str]):
    """(bar_times, X, report): per trade in order, its entry bar's time (None
    if no bar) and a trades x columns float matrix, NaN where missing. Distinct
    (date, time) pairs are looked up once; the matrix is built from the
    distinct rows and gathered, never cell by cell."""
    bar_times, distinct, gather, report = await entry_matrix_gathered(pool, trades, columns)
    return bar_times, distinct[gather], report


# The entry-bar query runs this many distinct bars at a time, each chunk's
# rows copied into the float matrix and dropped. Fetched whole, the driver
# holds every value as a Python float first: ~19 KB a bar at 452 metrics,
# measured -- 424 MB at 22,000 bars, ~1.6 GB at 86,000. Chunked, the cost is
# the matrix itself (3.6 KB a bar) plus one chunk.
QUERY_CHUNK_BARS = 2000


async def bars_matrix(pool, keys: list[tuple], columns: list[str]):
    """(bar_time per key with a leading None, float matrix (len(keys)+1) x
    len(columns) with row 0 all NaN, query seconds) for DISTINCT (date, time)
    keys, in their order. Row i+1 is keys[i]; a key with no bar stays NaN."""
    import numpy as np
    distinct = np.full((len(keys) + 1, len(columns)), np.nan)
    bar_of = [None] * (len(keys) + 1)
    pos = {key: i for i, key in enumerate(keys, start=1)}
    sql = entry_sql(columns)
    t0 = _time.monotonic()
    if keys:
        async with pool.acquire() as conn:
            for a in range(0, len(keys), QUERY_CHUNK_BARS):
                chunk = keys[a:a + QUERY_CHUNK_BARS]
                rows = await conn.fetch(sql, [k[0] for k in chunk], [k[1] for k in chunk])
                idx, vals = [], []
                for r in rows:
                    i = pos[(r["entry_date"], r["entry_time"])]
                    bar_of[i] = r["bar_time"]
                    if r["bar_time"] is not None:
                        idx.append(i)
                        vals.append(tuple(r)[3:])
                if idx:
                    v = np.array(vals, dtype=object)
                    v[v == None] = np.nan                       # noqa: E711 -- elementwise, on an object array
                    distinct[idx] = v.astype(float)
                del rows, vals
    return bar_of, distinct, round(_time.monotonic() - t0, 3)


async def entry_matrix_gathered(pool, trades: list[tuple], columns: list[str]):
    """(bar_times, distinct, gather, report): entry_matrix without the gather,
    so X = distinct[gather]. The ranking takes this form: a sweep's 240,000
    trades point at a few thousand distinct bars, and the gathered matrix
    would be ~870 MB (surface_stats.correlations_gathered)."""
    import numpy as np
    keys = sorted({(t[0], t[1]) for t in trades if t[0] is not None and t[1] is not None})
    bar_of, distinct, query_s = await bars_matrix(pool, keys, columns)
    pos = {key: i for i, key in enumerate(keys, start=1)}
    gather = np.fromiter((pos.get((t[0], t[1]), 0) for t in trades), dtype=np.int64, count=len(trades))
    bar_times = [bar_of[g] for g in gather.tolist()]
    no_bar = sum(1 for b in bar_times if b is None)
    return bar_times, distinct, gather, {"distinct_entries": len(keys), "no_bar": no_bar, "with_bar": len(trades) - no_bar,
                          "query_s": query_s, "bar_rule": BAR_RULE, "lookahead_confirmed": LOOKAHEAD_CONFIRMED}


async def entry_values(pool, trades: list[tuple], columns: list[str]) -> tuple[list[dict], dict]:
    """Per trade (in order): {bar_time, <column>: value|None, ...}, plus a
    report. For a few columns; the ranking uses entry_matrix."""
    bar_times, X, report = await entry_matrix(pool, trades, columns)
    out = []
    for bt, row in zip(bar_times, X.tolist()):
        out.append({"bar_time": bt, **{c: (None if v != v else v) for c, v in zip(columns, row)}})
    return out, report
