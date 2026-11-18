"""SPX open-interest, volume and price metrics for the OO/Mesosim Backtest
page: four tables in the OPEN_INTEREST database, described by its
spx_metric_catalog (2026-11-17).

  spx_oi_daily      one row per trade_date            join: the entry DATE
  spx_price_daily   one row per trade_date            join: the entry DATE
  spx_oi_5min       one row per (trade_date, quote_time)  join: the entry BAR
  spx_price_5min    one row per (trade_date, quote_time)  join: the entry BAR

They are read through the EXISTING open_interest pool (app.db.get_oi_pool,
OI_DATABASE_URL). The surface metrics (surface.py) stay on the main pool and
are untouched by this module: nothing here is shared with them but the
request parsing (surface.parse_bars) and the unit formats.

WHAT THIS CODE TAKES ON TRUST -- nothing on this side can check it:

  * THE DAILY TABLES ARE ALREADY LOOKAHEAD-FREE BY DATE. The row dated T
    holds open interest as of the previous session's close and volume from
    the previous session (published the morning of T). So a trade entered on
    T at any time reads row T -- reading T-1 would be a session staler, not
    safer. The Open_Interest build asserts this on every run (every volume
    session contributing to row T is the prior NYSE session); THAT ASSERTION
    IS THE ONLY GUARD. If it is ever dropped from the build, nothing here
    will notice.
  * THE 5-MINUTE ROWS ARE POINT-IN-TIME AT quote_time: a 14:45 row is built
    from inputs known at 14:45:00 (that bar's spx_open, yesterday's OI and
    volume). So the entry bar is the last row on the entry date with
    quote_time <= entry time -- the surface join's rule (surface.BAR_RULE),
    never the previous session. The build asserts this too.

NULLS AFTER COVERAGE STARTS -- unlike surface_metrics_core, these tables have
them, by design: a bar whose spx_open is missing/NaN/<=0 has NULL on every
spot-relative level (a missing spot reads as unknown, never as "no open
interest in band"), and an intraday z-score warms up PER BAR OF DAY (60
readings at 14:45 before the 14:45 z exists; early closes delay afternoon
bars). So a trade can have its row/bar and still no value. The values
endpoint reports, per bar, whether a row/bar was FOUND (`present`), and the
page states three reasons apart: before the metric's data starts, no
row/bar, a row/bar but no value.

Grid mismatch, documented not fixed: these 5-minute tables run 09:30-15:55,
surface_metrics_core 09:35-16:00. A 09:30 entry gets these metrics and no
surface metric; a 16:00 entry reads this table's 15:55 row.

THE CATALOG IS READ LIVE (the user's decision, 2026-11-17): spx_metric_catalog,
primary key (table_name, metric). A metric is served when `surfaced` is true,
its table is one of SOURCES, and the table really has the column with a
numeric type. `surfaced = false` rows are neither served nor ranked. The id a
metric travels under is "<source>.<metric>" (e.g. "oi_5min.pct_oi_atm"): the
same metric name can sit in two tables (`spot` does), and the id names the
table and pool to read. Surface (IV) ids stay bare column names, so nothing
saved before this module changes.

The catalog and coverage are cached and rebuilt when ANY of the four tables'
first date, last date or row count moves, or the catalog's own content does
(an md5 over its rows -- the catalog is small, and built_at alone would miss
a hand edit). The 5-minute job writes all 78 of today's rows on its first
cycle (unprinted bars blank), so the count moves once a session, not per bar.
"""
from __future__ import annotations

import asyncio
import logging
import math
import time as _time
from datetime import datetime

from app.oo_backtest.surface import quote_ident

log = logging.getLogger(__name__)

CATALOG = "spx_metric_catalog"

# The four tables, by the source key an id carries. `join`: "date" = the row
# for the entry date; "bar" = the last row at or before the entry time on
# the entry date.
SOURCES = {
    "oi_daily": {"table": "spx_oi_daily", "join": "date", "label": "OI daily"},
    "oi_5min": {"table": "spx_oi_5min", "join": "bar", "label": "OI 5-minute"},
    "price_daily": {"table": "spx_price_daily", "join": "date", "label": "Price daily"},
    "price_5min": {"table": "spx_price_5min", "join": "bar", "label": "Price 5-minute"},
}
SOURCE_OF_TABLE = {s["table"]: k for k, s in SOURCES.items()}

# The page's sections (the user's decision): IV is the surface pane; these
# are the other three. `cross` (volume against open interest) belongs to BOTH
# OI and Volume. A family in no section is still served and reported.
SECTIONS = [
    {"key": "oi", "label": "OI", "families": ["oi", "oi_spot", "cross"]},
    {"key": "volume", "label": "Volume", "families": ["volume", "volume_spot", "cross"]},
    {"key": "price", "label": "Price", "families": ["price_trend", "price_range", "price_momentum", "price_vol"]},
]

# Family colours, the same eight validated hues surface.FAMILY_GROUPS uses.
# Distinct WITHIN each section (the only place they are drawn side by side);
# `cross` keeps one colour in both sections it appears in.
FAMILY_GROUPS = [
    {"label": "OI", "color": "#3987e5", "families": ["oi"]},
    {"label": "OI, spot-relative", "color": "#199e70", "families": ["oi_spot"]},
    {"label": "Volume vs OI", "color": "#c98500", "families": ["cross"]},
    {"label": "Volume", "color": "#d95926", "families": ["volume"]},
    {"label": "Volume, spot-relative", "color": "#d55181", "families": ["volume_spot"]},
    {"label": "Trend", "color": "#3987e5", "families": ["price_trend"]},
    {"label": "Range", "color": "#199e70", "families": ["price_range"]},
    {"label": "Momentum", "color": "#9085e9", "families": ["price_momentum"]},
    {"label": "Volatility", "color": "#e66767", "families": ["price_vol"]},
]

NUMERIC_TYPES = frozenset({"double precision", "real", "numeric", "integer", "bigint", "smallint"})

CATALOG_FIELDS = ("metric", "table_name", "description", "family", "surfaced", "form", "base_column",
                  "units", "formula")
CATALOG_SQL = f"SELECT {', '.join(CATALOG_FIELDS)} FROM {CATALOG} ORDER BY table_name, metric"
# Changes when any catalog row does -- including built_at, which is in the row.
CATALOG_KEY_SQL = f"SELECT count(*) AS n, md5(coalesce(string_agg(c::text, '|' ORDER BY c.table_name, c.metric), '')) AS h FROM {CATALOG} c"

# current_schema(), not 'public': these tables live wherever the pool's
# search_path puts them (public on the real database), and a gate's scratch
# schema is found the same way.
TABLE_COLUMNS_SQL = """
SELECT table_name, column_name, data_type FROM information_schema.columns
WHERE table_schema = current_schema() AND table_name = ANY($1::text[])
"""


def table_key_sql(table: str) -> str:
    return f"SELECT min(trade_date) AS first_date, max(trade_date) AS last_date, count(*) AS row_count FROM {quote_ident(table)}"


def make_id(source: str, metric: str) -> str:
    return f"{source}.{metric}"


def _valid(col: str, alias: str = "") -> str:
    """SQL: the column as float8, NULL when NULL or NaN. Every value this
    module reads goes through it, so coverage and values agree on what
    "no value" is."""
    q = (alias + "." if alias else "") + quote_ident(col)
    return f"NULLIF({q}::float8, 'NaN'::float8)"


def metric_set(catalog: list[dict], table_columns: dict[str, dict[str, str]]) -> tuple[list[dict], dict]:
    """(served metrics, report). A catalog row is served when surfaced is
    true, its table is one of SOURCES and that table has the column with a
    numeric type. Each served row gains id, source, join and sections."""
    fam_sections: dict[str, list[str]] = {}
    for s in SECTIONS:
        for f in s["families"]:
            fam_sections.setdefault(f, []).append(s["key"])
    served, rep = [], {"catalog_rows": len(catalog), "not_surfaced": 0, "unknown_table": [], "missing_from_table": [],
                       "wrong_type": [], "no_section": []}
    for r in catalog:
        if r.get("surfaced") is not True:
            rep["not_surfaced"] += 1
            continue
        src = SOURCE_OF_TABLE.get(r["table_name"])
        ident = f"{r['table_name']}.{r['metric']}"
        if src is None:
            rep["unknown_table"].append(ident)
            continue
        cols = table_columns.get(r["table_name"], {})
        if r["metric"] not in cols:
            rep["missing_from_table"].append(ident)
            continue
        if cols[r["metric"]] not in NUMERIC_TYPES:
            rep["wrong_type"].append(f"{ident} ({cols[r['metric']]})")
            continue
        sections = fam_sections.get(r["family"], [])
        if not sections:
            rep["no_section"].append(ident)
        served.append({**{k: r.get(k) for k in CATALOG_FIELDS if k != "surfaced"},
                       "id": make_id(src, r["metric"]), "source": src, "join": SOURCES[src]["join"],
                       "sections": sections})
    rep["served"] = len(served)
    rep["unique_names"] = len({m["metric"] for m in served})
    rep["displayed"] = sum(max(1, len(m["sections"])) for m in served)
    return served, rep


def coverage_sql(table: str, join: str, columns: list[str]) -> str:
    """First date with a value, per column, one statement. Each scalar
    subquery walks the primary key in order and stops at the first row with
    a value. ONE date per metric: an intraday z-score warms up per bar of
    day, so some of its bars start later -- the page reports trades past
    this date with no value apart ("a bar but no value"), never as missing
    data."""
    order = "trade_date, quote_time" if join == "bar" else "trade_date"
    parts = [f"(SELECT trade_date FROM {quote_ident(table)} WHERE {_valid(c)} IS NOT NULL "
             f"ORDER BY {order} LIMIT 1) AS {quote_ident(c)}" for c in columns]
    return "SELECT " + ",\n       ".join(parts)


def values_sql(table: str, join: str, column: str) -> str:
    """$1 date[], $2 time[] -- one element per DISTINCT entry (date, time).
    Returns per element: present (a row/bar was found), bar_time (bar joins
    only) and v (the value, NULL for NULL or NaN)."""
    t = quote_ident(table)
    if join == "bar":
        return f"""
SELECT t.d AS entry_date, t.tm AS entry_time, b.quote_time AS bar_time,
       (b.quote_time IS NOT NULL) AS present, b.v
FROM unnest($1::date[], $2::time[]) AS t(d, tm)
LEFT JOIN LATERAL (
    SELECT quote_time, {_valid(column)} AS v
    FROM {t}
    WHERE trade_date = t.d AND quote_time <= t.tm
    ORDER BY quote_time DESC
    LIMIT 1
) b ON true"""
    # The row dated the entry date, full stop: it is already shifted to be
    # lookahead-free (see the module docstring -- the build's assertion is
    # the only guard).
    return f"""
SELECT t.d AS entry_date, t.tm AS entry_time, NULL::time AS bar_time,
       (b.trade_date IS NOT NULL) AS present, {_valid(column, 'b')} AS v
FROM unnest($1::date[], $2::time[]) AS t(d, tm)
LEFT JOIN {t} b ON b.trade_date = t.d"""


# ── catalog + coverage cache ────────────────────────────────────────────────

_CACHE: dict = {"key": None, "metrics": None, "report": None, "built_at": None, "build_s": None, "tables": None}
_LOCK = asyncio.Lock()


async def _cache_key(conn) -> tuple:
    parts = []
    for s in SOURCES.values():
        try:
            r = await conn.fetchrow(table_key_sql(s["table"]))
            parts.append((s["table"], r["first_date"], r["last_date"], r["row_count"]))
        except Exception as exc:  # noqa: BLE001 -- a missing table is reported, not fatal to the others
            parts.append((s["table"], "error", type(exc).__name__, str(exc)[:200]))
    c = await conn.fetchrow(CATALOG_KEY_SQL)
    return tuple(parts) + ((c["n"], c["h"]),)


async def get_catalog(pool) -> dict:
    """{metrics: [served row + min_date], sections, family_groups, report,
    tables, ...}. Rebuilt when a table's dates or row count, or the
    catalog's content, change."""
    async with pool.acquire() as conn:
        key = await _cache_key(conn)
    async with _LOCK:
        if _CACHE["key"] != key or _CACHE["metrics"] is None:
            t0 = _time.monotonic()
            async with pool.acquire() as conn:
                catalog = [dict(r) for r in await conn.fetch(CATALOG_SQL)]
                tables = [s["table"] for s in SOURCES.values()]
                tcols: dict[str, dict[str, str]] = {}
                for r in await conn.fetch(TABLE_COLUMNS_SQL, tables):
                    tcols.setdefault(r["table_name"], {})[r["column_name"]] = r["data_type"]
                served, report = metric_set(catalog, tcols)
                cov: dict[str, object] = {}
                for src, s in SOURCES.items():
                    cols = [m["metric"] for m in served if m["source"] == src]
                    if cols:
                        row = dict(await conn.fetchrow(coverage_sql(s["table"], s["join"], cols)))
                        for c in cols:
                            cov[make_id(src, c)] = row.get(c)
            metrics = [{**m, "min_date": cov[m["id"]].isoformat() if cov.get(m["id"]) else None} for m in served]
            table_info = {p[0]: ({"first_date": p[1].isoformat() if p[1] else None,
                                  "last_date": p[2].isoformat() if p[2] else None, "row_count": p[3]}
                                 if p[1] != "error" else {"error": f"{p[2]}: {p[3]}"})
                          for p in key[:-1]}
            _CACHE.update(key=key, metrics=metrics, report=report, tables=table_info,
                          built_at=datetime.now().isoformat(timespec="seconds"),
                          build_s=round(_time.monotonic() - t0, 2))
            log.info("oo-backtest spx metrics catalog: %d served (%d names, %d displayed) of %d rows, %d not "
                     "surfaced, built in %.2fs; unknown table %s, missing %s, wrong type %s, no section %s",
                     report["served"], report["unique_names"], report["displayed"], report["catalog_rows"],
                     report["not_surfaced"], _CACHE["build_s"], report["unknown_table"],
                     report["missing_from_table"], report["wrong_type"], report["no_section"])
    from app.oo_backtest import surface
    return {"metrics": _CACHE["metrics"], "report": _CACHE["report"], "tables": _CACHE["tables"],
            "built_at": _CACHE["built_at"], "build_s": _CACHE["build_s"],
            "sources": SOURCES, "sections": SECTIONS, "family_groups": FAMILY_GROUPS,
            "form_labels": surface.FORM_LABELS, "unit_formats": surface.UNIT_FORMATS,
            "default_unit_format": surface.DEFAULT_UNIT_FORMAT, "row_auto_bins": surface.ROW_AUTO_BINS,
            "bar_rule": surface.BAR_RULE}


# ── values for one metric ───────────────────────────────────────────────────

QUERY_CHUNK_BARS = 5000


def _num(v):
    """A JSON-safe float, or None (NULL, NaN, +-inf)."""
    if v is None:
        return None
    f = float(v)
    return f if math.isfinite(f) else None


async def entry_values(pool, metric: dict, bars: list[tuple]) -> tuple[list, list, list, dict]:
    """(values, present, bar_times, report) aligned to `bars` (the page's
    distinct (date, time) list; a None date or time is never looked up and
    reads as not present). Looked up once per distinct key."""
    s = SOURCES[metric["source"]]
    keys = sorted({b for b in bars if b[0] is not None and b[1] is not None})
    sql = values_sql(s["table"], s["join"], metric["metric"])
    found: dict = {}
    t0 = _time.monotonic()
    if keys:
        async with pool.acquire() as conn:
            for a in range(0, len(keys), QUERY_CHUNK_BARS):
                chunk = keys[a:a + QUERY_CHUNK_BARS]
                for r in await conn.fetch(sql, [k[0] for k in chunk], [k[1] for k in chunk]):
                    found[(r["entry_date"], r["entry_time"])] = (r["present"], r["bar_time"], _num(r["v"]))
    values, present, bar_times = [], [], []
    for b in bars:
        p, bt, v = found.get(b, (False, None, None))
        values.append(v)
        present.append(bool(p))
        bar_times.append(bt.strftime("%H:%M:%S") if bt else None)
    no_row = sum(1 for p in present if not p)
    no_value = sum(1 for p, v in zip(present, values) if p and v is None)
    return values, present, bar_times, {"distinct_entries": len(keys), "not_present": no_row,
                                        "present_no_value": no_value, "with_value": len(bars) - no_row - no_value,
                                        "query_s": round(_time.monotonic() - t0, 3), "join": s["join"]}


# ── ranking (P4): every served metric against the trades' P/L ───────────────
#
# One request ranks the three sections (OI, Volume, Price). Each section is
# its OWN Benjamini-Hochberg family (the user's decision: adding these must
# not make the 452 surface metrics' adjusted p more conservative, and the
# sections are read apart), so a `cross` metric gets one adjusted p in OI and
# another in Volume; its r, p and filter gain are the same in both. Each
# section may carry its own coverage start (`from`, the page's "common
# coverage only"): only trades entered on or after it are in that section's
# sample. Sections sharing a start share one computation.
#
# "bars" (distinct entries behind a correlation) is per source: a daily
# table's metric holds one value per DATE, a 5-minute table's one per bar.

def matrix_sql(table: str, join: str, columns: list[str]) -> str:
    """values_sql for many columns at once: $1 date[], $2 time[] -> per
    element bar_time (bar joins), present, then every column (NULL for NULL
    or NaN), in `columns` order."""
    t = quote_ident(table)
    if join == "bar":
        inner = ", ".join(f"{_valid(c)} AS {quote_ident(c)}" for c in columns)
        outer = ", ".join(f"b.{quote_ident(c)}" for c in columns)
        return f"""
SELECT t.d AS entry_date, t.tm AS entry_time, b.quote_time AS bar_time,
       (b.quote_time IS NOT NULL) AS present, {outer}
FROM unnest($1::date[], $2::time[]) AS t(d, tm)
LEFT JOIN LATERAL (
    SELECT quote_time, {inner}
    FROM {t}
    WHERE trade_date = t.d AND quote_time <= t.tm
    ORDER BY quote_time DESC
    LIMIT 1
) b ON true"""
    outer = ", ".join(f"{_valid(c, 'b')} AS {quote_ident(c)}" for c in columns)
    return f"""
SELECT t.d AS entry_date, t.tm AS entry_time, NULL::time AS bar_time,
       (b.trade_date IS NOT NULL) AS present, {outer}
FROM unnest($1::date[], $2::time[]) AS t(d, tm)
LEFT JOIN {t} b ON b.trade_date = t.d"""


async def bars_matrix(pool, keys: list[tuple], metrics: list[dict]):
    """For DISTINCT (date, time) keys: (matrix, present, bar_time, query_s).
    matrix is (len(keys)+1) x len(metrics), row 0 all NaN (trades with no
    key point there), row i+1 = keys[i]; present[source] and
    bar_time[source] are per row. One query per source, chunked."""
    import numpy as np
    from app.oo_backtest.surface import QUERY_CHUNK_BARS as CHUNK
    X = np.full((len(keys) + 1, len(metrics)), np.nan)
    present = {src: np.zeros(len(keys) + 1, dtype=bool) for src in SOURCES}
    bar_time = {src: [None] * (len(keys) + 1) for src in SOURCES}
    pos = {k: i for i, k in enumerate(keys, start=1)}
    t0 = _time.monotonic()
    if keys:
        async with pool.acquire() as conn:
            for src, s in SOURCES.items():
                cols = [j for j, m in enumerate(metrics) if m["source"] == src]
                if not cols:
                    continue
                sql = matrix_sql(s["table"], s["join"], [metrics[j]["metric"] for j in cols])
                for a in range(0, len(keys), CHUNK):
                    chunk = keys[a:a + CHUNK]
                    rows = await conn.fetch(sql, [k[0] for k in chunk], [k[1] for k in chunk])
                    idx, vals = [], []
                    for r in rows:
                        i = pos[(r["entry_date"], r["entry_time"])]
                        present[src][i] = bool(r["present"])
                        bar_time[src][i] = r["bar_time"]
                        if r["present"]:
                            idx.append(i)
                            vals.append(tuple(r)[4:])
                    if idx:
                        v = np.array(vals, dtype=object)
                        v[v == None] = np.nan                       # noqa: E711 -- elementwise, on an object array
                        X[np.ix_(idx, cols)] = v.astype(float)
                    del rows, vals
    return X, present, bar_time, round(_time.monotonic() - t0, 3)


def rank_sections(metrics: list[dict], X, trades: list[tuple], gather, present, bar_time,
                  starts: dict | None = None) -> dict:
    """{section key: rows} for SECTIONS. trades = [(date, time, pnl)],
    gather[t] -> row of X; starts = {section key: date | None}. Each row:
    id, metric, source, family, form, units + n, bars, r/p, BH p WITHIN
    ITS SECTION, filter gain fields."""
    import numpy as np
    from app.oo_backtest import surface_stats as ss
    starts = starts or {}
    gather = np.asarray(gather)
    pnl = np.array([np.nan if t[2] is None else t[2] for t in trades], dtype=float)
    dates = [t[0] for t in trades]
    # distinct-entry ids per source: a date for a daily table, (date, bar) for a 5-minute one
    bar_ids = {}
    for src, s in SOURCES.items():
        ids: dict = {}
        if s["join"] == "date":
            bar_ids[src] = np.array([ids.setdefault(d, len(ids)) for d in dates])
        else:
            bar_ids[src] = np.array([ids.setdefault((d, bar_time[src][g]), len(ids)) for d, g in zip(dates, gather.tolist())])
    by_start: dict = {}
    for sec in SECTIONS:
        by_start.setdefault(starts.get(sec["key"]), []).append(sec)
    out: dict = {}
    for start, secs in by_start.items():
        fams = {f for sec in secs for f in sec["families"]}
        p = pnl if start is None else np.where([d is not None and d >= start for d in dates], pnl, np.nan)
        stats: dict = {}
        for src in SOURCES:
            cols = [j for j, m in enumerate(metrics) if m["source"] == src and m["family"] in fams]
            if not cols:
                continue
            for j, st in zip(cols, ss.stat_rows(X[:, cols], list(p), bar_ids[src], gather)):
                stats[j] = st
        for sec in secs:
            rows = [{"id": m["id"], "metric": m["metric"], "source": m["source"], "family": m["family"],
                     "form": m["form"], "units": m["units"], **stats[j]}
                    for j, m in enumerate(metrics) if m["family"] in sec["families"]]
            out[sec["key"]] = ss.with_bh(rows)
    return out
