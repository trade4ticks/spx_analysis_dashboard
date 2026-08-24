"""
Equity IV Analysis page — global (cross-sectional) half.

Mounted at /api/equity-iv against the open_interest database, alongside the
existing OI routers. Nothing here touches the SPX surface DB, and no existing
endpoint is modified — this router is purely additive.

The universe is ~121 tickers. Every metric column is described by
equity_metrics_catalog (602 rows), and that catalog is what drives the metric
pickers: a hand-written dropdown over 600 columns is not maintainable, and
would drift the moment a column is added. So every endpoint here takes column
NAMES from the client and validates them against the catalog before they
reach SQL. asyncpg cannot parameterize an identifier, so that whitelist IS
the injection guard — the same pattern as ticker_analysis.py's
_table_columns(), widened to carry the catalog metadata alongside the name.

Endpoints
  GET /catalog         one row per metric column + the distinct
                       family / tenor / wing / form / units vocabularies.
  GET /calendar        trade dates, with the snapshots available on each.
  GET /cross-section   one row per ticker at a (date, snapshot): the two
                       chosen axis metrics plus the context the scatter
                       needs. Also feeds the universe histogram.
  GET /universe-stats  the four numbers beside the histogram, over the
                       page's history window.
  GET /scanner         the sortable / filterable table.

Which table a column lives in
-----------------------------
equity_metrics holds the ~234 base columns; equity_metrics_z holds the ~368
z columns, PK-joined on (ticker, trade_date, snapshot). The catalog's `form`
says which: 'base' -> equity_metrics, 'z_63' / 'z_252' -> equity_metrics_z.
Callers never need to know; _expr() resolves it. The z join is only emitted
when a z column was actually asked for.

Extrapolated nodes
------------------
equity_surface writes a node even when the target delta falls outside the
fitted smile's domain — the spline returns its boundary value and the row is
written anyway, flagged `extrapolated`. equity_metrics carries the same fact
per node as extrap_{wing}_{tenor}d booleans, plus extrap_rate_short as a
per-ticker summary over tenors <= 30.

This is load-bearing, not cosmetic. The rate varies enormously by name (SPY
0.0%, AAPL 2.6%, T 21.6% on one date), so a metric built on a fabricated node
inherits the fabrication silently, and a cross-sectional skew ranking that
ignores it puts thin chains at the top — T's wing slope reads four times
flatter than AAPL's because the spline hit its boundary, not because T has
flat skew.

Every endpoint here resolves which node flags a metric depends on
(_flags_for) and returns them alongside the value, so the caller can mark or
exclude. With exclude_extrapolated on, the offending VALUE is nulled rather
than the row dropped: the scatter needs both axes but the histogram needs
only x, and nulling per-axis lets one payload serve both without either
silently losing tickers the other still wants.

Nulls
-----
A missing metric is legitimately NULL — a thin chain lacks wing nodes, a
short tenor does not always bracket. Nulls are returned as null and must
render as absent, never as zero.
"""
import re
from datetime import date as date_type, datetime

from fastapi import APIRouter, Depends, HTTPException, Query

from app.db import get_oi_pool

router = APIRouter(tags=["equity-iv"])

METRICS_TABLE = "equity_metrics"
Z_TABLE       = "equity_metrics_z"
CATALOG_TABLE = "equity_metrics_catalog"

# Tenors the surface is fitted at. Used to sanity-check a tenor parsed out of
# a column name before it is turned into an extrap flag name.
TENORS = (7, 14, 21, 30, 60, 90)

# Catalog `wing` value -> the surface nodes the metric is actually built on.
#
# Most are literal: a wing of "25p_atm" is the 25p and atm nodes. The last
# four are not, and are the reason this is a table rather than a split("_"):
#
#   10d / 25d   risk reversals are call minus put at that delta, so BOTH
#               wings, not one.
#   10p_5p      the far-wing cost of a broken-wing butterfly. 5p is not a
#               stored node, so 10p is the only leg whose fabrication is
#               visible here.
#   12.5d       the delta-neutral short of a 25-delta-long 1x2. Solved
#               between the 10p and 25p nodes, so it inherits both.
#   short       likewise, the zero-cost short strike.
#
# A wing not listed here contributes no flags — which is correct for the
# families with no surface-node dependency at all (realized_vol, calendar).
WING_NODES = {
    "10p": ("10p",),
    "25p": ("25p",),
    "atm": ("atm",),
    "25c": ("25c",),
    "10c": ("10c",),
    "10p_25p":     ("10p", "25p"),
    "10p_atm":     ("10p", "atm"),
    "25p_atm":     ("25p", "atm"),
    "25p_25c":     ("25p", "25c"),
    "atm_25c":     ("atm", "25c"),
    "atm_10c":     ("atm", "10c"),
    "10p_25p_atm": ("10p", "25p", "atm"),
    "10p_atm_10c": ("10p", "atm", "10c"),
    "25p_atm_25c": ("25p", "atm", "25c"),
    "atm_25c_10c": ("atm", "25c", "10c"),
    "10d":    ("10p", "10c"),
    "25d":    ("25p", "25c"),
    "10p_5p": ("10p",),
    "12.5d":  ("10p", "25p"),
    "short":  ("10p", "25p"),
}

# Tenors embedded in a column name: skew_30d_25p_atm -> [30];
# term_ratio_30d_90d -> [30, 90]; term_slope_14d_30d_25p -> [14, 30].
# The lookahead keeps it from matching inside "_z_252" or a "1m" suffix.
_TENOR_RE = re.compile(r"_(\d+)d(?=_|$)")

# Context columns every cross-section / scanner row carries, regardless of
# which metrics were asked for. All live in equity_metrics (base form).
CONTEXT_COLS = ("spot", "extrap_rate_short", "median_n_strikes_clean",
                "source", "captured_at")

_catalog_cache: dict | None = None


# ── catalog ──────────────────────────────────────────────────────────────────

async def _catalog(pool) -> dict:
    """Load and cache the metric catalog, cross-checked against the real tables.

    A catalog row whose column does not exist in the table its `form` points
    at is DROPPED rather than trusted. The catalog is metadata maintained
    beside the loader, so it can describe a column that was renamed or never
    shipped; letting such a name through would put an unbacked identifier into
    SQL. Cross-checking against information_schema means this whitelist can
    only ever be a subset of what actually exists.

    Cached for the process lifetime. The catalog changes only when the metric
    set does, which is a deploy, not a request.
    """
    global _catalog_cache
    if _catalog_cache is not None:
        return _catalog_cache

    async with pool.acquire() as conn:
        rows = await conn.fetch(
            f"SELECT column_name, family, tenor, wing, form, base_column, "
            f"       units, description, formula "
            f"FROM {CATALOG_TABLE}"
        )
        real = await conn.fetch(
            "SELECT table_name, column_name FROM information_schema.columns "
            "WHERE table_schema = 'public' AND table_name = ANY($1::text[])",
            [METRICS_TABLE, Z_TABLE],
        )

    live = {(r["table_name"], r["column_name"]) for r in real}
    extrap_cols = {c for (t, c) in live
                   if t == METRICS_TABLE and c.startswith("extrap_")}

    by_col: dict[str, dict] = {}
    for r in rows:
        col   = r["column_name"]
        form  = r["form"] or "base"
        table = METRICS_TABLE if form == "base" else Z_TABLE
        if (table, col) not in live:
            continue
        by_col[col] = {
            "column_name": col,
            "family":      r["family"],
            "tenor":       int(r["tenor"]) if r["tenor"] is not None else None,
            "wing":        r["wing"],
            "form":        form,
            "base_column": r["base_column"],
            "units":       r["units"],
            "description": r["description"],
            "formula":     r["formula"],
        }

    for entry in by_col.values():
        entry["extrap_flags"] = _flags_for(entry, extrap_cols)

    _catalog_cache = {"by_col": by_col, "extrap_cols": extrap_cols}
    return _catalog_cache


def _flags_for(entry: dict, extrap_cols: set) -> list:
    """extrap_* flag columns the metric in `entry` depends on.

    Tenors come from the column NAME when it carries them (term_ratio_30d_90d
    spans two), and fall back to the catalog's single `tenor` otherwise —
    vrp_1m is a 30d metric whose name says "1m". Wings come from WING_NODES.
    Returns [] for a metric with no surface-node dependency.
    """
    wings = WING_NODES.get(entry["wing"] or "", ())
    if not wings:
        return []
    tenors = [int(t) for t in _TENOR_RE.findall(entry["column_name"])
              if int(t) in TENORS]
    if not tenors and entry["tenor"] in TENORS:
        tenors = [entry["tenor"]]
    return [f"extrap_{w}_{t}d" for t in tenors for w in wings
            if f"extrap_{w}_{t}d" in extrap_cols]


def _entry(cat: dict, col: str) -> dict:
    """Catalog entry for `col`, or 400. This is the identifier whitelist."""
    hit = cat["by_col"].get(col)
    if hit is None:
        raise HTTPException(400, f"Unknown metric column: {col!r}")
    return hit


def _expr(entry: dict) -> str:
    """Qualified SQL reference. Safe to interpolate — _entry() vetted the name."""
    alias = "m" if entry["form"] == "base" else "z"
    return '{}."{}"'.format(alias, entry["column_name"])


def _extrap_expr(entry: dict) -> str:
    """Boolean SQL: is any node this metric depends on extrapolated?

    COALESCE to false because a NULL flag means the node was never evaluated,
    which is a null metric — the "is this fabricated" question is then moot,
    and answering TRUE would count it as an exclusion on top of being absent.
    """
    flags = entry["extrap_flags"]
    if not flags:
        return "FALSE"
    return "(" + " OR ".join('COALESCE(m."{}", false)'.format(f) for f in flags) + ")"


def _needs_z(entries) -> bool:
    return any(e["form"] != "base" for e in entries)


def _from_clause(needs_z: bool) -> str:
    base = f"FROM {METRICS_TABLE} m"
    if not needs_z:
        return base
    return (f"{base} JOIN {Z_TABLE} z "
            f"ON z.ticker = m.ticker AND z.trade_date = m.trade_date "
            f"AND z.snapshot = m.snapshot")


def _meta(entry: dict) -> dict:
    """The subset of a catalog entry the client needs to label and format."""
    return {k: entry[k] for k in
            ("column_name", "family", "tenor", "wing", "form", "units",
             "description", "extrap_flags")}


def _jsonable(v):
    if isinstance(v, (datetime, date_type)):
        return v.isoformat()
    return v


# ── date / snapshot resolution ───────────────────────────────────────────────

async def _resolve_slice(conn, date, snapshot):
    """(trade_date, snapshot), defaulting to the latest available of each.

    Snapshot buckets are zero-padded HHMM text ('0945', '1545', and the
    '0935'..'1600' intraday grid), so MAX() is the latest one — opening the
    page at 11am should show 10:55, not yesterday's close.
    """
    if date:
        try:
            d = date_type.fromisoformat(date)
        except ValueError:
            raise HTTPException(400, f"Invalid date: {date!r}")
    else:
        d = await conn.fetchval(f"SELECT max(trade_date) FROM {METRICS_TABLE}")
        if d is None:
            raise HTTPException(404, f"{METRICS_TABLE} is empty")

    if snapshot:
        snap = snapshot
    else:
        snap = await conn.fetchval(
            f"SELECT max(snapshot) FROM {METRICS_TABLE} WHERE trade_date = $1", d
        )
        if snap is None:
            raise HTTPException(404, f"No snapshots for {d}")
    return d, snap


def _window_start(d, window: str):
    """History-window control (3M / 1Y / 2Y / All) -> inclusive start date."""
    days = {"3m": 91, "1y": 365, "2y": 730}.get(window.lower())
    if days is None:
        if window.lower() != "all":
            raise HTTPException(400, f"Invalid window: {window!r}")
        return None
    return date_type.fromordinal(d.toordinal() - days)


# ── endpoints ────────────────────────────────────────────────────────────────

@router.get("/catalog")
async def catalog(pool=Depends(get_oi_pool)):
    """Every metric column plus the vocabularies the pickers group and filter by.

    Fetched once per page load and held client-side; this is what makes the
    metric dropdowns catalog-driven rather than hardcoded. `units` is what the
    client formats by — vol_decimal as a percentage, ratio as a plain number,
    z_score on a diverging scale.
    """
    if not pool:
        return {"error": "OI database not configured", "metrics": []}
    cat = await _catalog(pool)
    metrics = sorted(cat["by_col"].values(),
                     key=lambda r: (r["family"], r["column_name"]))
    return {
        "metrics":  [dict(_meta(m), base_column=m["base_column"]) for m in metrics],
        "families": sorted({m["family"] for m in metrics}),
        "tenors":   sorted({m["tenor"] for m in metrics if m["tenor"] is not None}),
        "wings":    sorted({m["wing"] for m in metrics if m["wing"]}),
        "units":    sorted({m["units"] for m in metrics if m["units"]}),
        "forms":    ["base", "z_63", "z_252"],
    }


@router.get("/calendar")
async def calendar(pool=Depends(get_oi_pool)):
    """Trade dates newest-first, each with the snapshots available on it.

    Drives the date picker and the snapshot picker together, so choosing a
    date can never offer a snapshot that date does not have.
    """
    if not pool:
        return {"error": "OI database not configured", "dates": []}
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            f"SELECT trade_date, array_agg(DISTINCT snapshot ORDER BY snapshot) AS snaps "
            f"FROM {METRICS_TABLE} GROUP BY trade_date ORDER BY trade_date DESC"
        )
    return {"dates": [{"date": str(r["trade_date"]), "snapshots": list(r["snaps"])}
                      for r in rows]}


@router.get("/cross-section")
async def cross_section(
    x:                    str  = Query(..., description="X-axis metric column"),
    y:                    str  = Query(..., description="Y-axis metric column"),
    date:                 str  = Query(None),
    snapshot:             str  = Query(None),
    size:                 str  = Query("median_n_strikes_clean"),
    color:                str  = Query(None),
    exclude_extrapolated: bool = Query(True),
    pool=Depends(get_oi_pool),
):
    """One row per ticker for the scatter, and the x column for the histogram.

    `size` defaults to median_n_strikes_clean — surviving strikes per fitted
    expiry. It is a liquidity PROXY, not liquidity: this database has no
    volume, ADV or market-cap column, and no sector table at all, so the
    spec's "color by sector" has no source. `color` is therefore an optional
    METRIC, rendered on a diverging scale.

    Values are returned with nulls intact. With exclude_extrapolated on, a
    value whose metric depends on a fabricated node is nulled — per axis, so
    the histogram keeps every ticker whose x survives even when its y did not.
    """
    if not pool:
        return {"error": "OI database not configured", "points": []}

    cat    = await _catalog(pool)
    ex, ey = _entry(cat, x), _entry(cat, y)
    esize  = _entry(cat, size)  if size  else None
    ecolor = _entry(cat, color) if color else None
    used   = [e for e in (ex, ey, esize, ecolor) if e]

    sel = [
        "m.ticker",
        f"{_expr(ex)} AS x",
        f"{_expr(ey)} AS y",
        f"{_extrap_expr(ex)} AS x_extrap",
        f"{_extrap_expr(ey)} AS y_extrap",
    ]
    sel += ['m."{0}" AS {0}'.format(c) for c in CONTEXT_COLS]
    if esize:
        sel.append(f"{_expr(esize)} AS size_v")
    if ecolor:
        sel += [f"{_expr(ecolor)} AS color_v",
                f"{_extrap_expr(ecolor)} AS color_extrap"]

    async with pool.acquire() as conn:
        d, snap = await _resolve_slice(conn, date, snapshot)
        rows = await conn.fetch(
            f"SELECT {', '.join(sel)} "
            f"{_from_clause(_needs_z(used))} "
            f"WHERE m.trade_date = $1 AND m.snapshot = $2 "
            f"ORDER BY m.ticker",
            d, snap,
        )

    points, n_x_excl, n_y_excl = [], 0, 0
    for r in rows:
        xv, yv = r["x"], r["y"]
        if exclude_extrapolated and r["x_extrap"] and xv is not None:
            xv, n_x_excl = None, n_x_excl + 1
        if exclude_extrapolated and r["y_extrap"] and yv is not None:
            yv, n_y_excl = None, n_y_excl + 1
        points.append({
            "ticker":      r["ticker"],
            "x":           xv,
            "y":           yv,
            "x_extrap":    r["x_extrap"],
            "y_extrap":    r["y_extrap"],
            "spot":        r["spot"],
            "extrap_rate": r["extrap_rate_short"],
            "size":        r["size_v"]  if esize  else None,
            "color":       r["color_v"] if ecolor else None,
            "source":      r["source"],
            "captured_at": _jsonable(r["captured_at"]),
        })

    return {
        "date":     str(d),
        "snapshot": snap,
        "x": _meta(ex), "y": _meta(ey),
        "size":  _meta(esize)  if esize  else None,
        "color": _meta(ecolor) if ecolor else None,
        "points":    points,
        "n_tickers": len(points),
        "excluded":  {"x": n_x_excl, "y": n_y_excl,
                      "active": bool(exclude_extrapolated)},
    }


@router.get("/universe-stats")
async def universe_stats(
    metric:               str   = Query(..., description="The histogram's metric column"),
    date:                 str   = Query(None),
    snapshot:             str   = Query(None),
    window:               str   = Query("1y", description="3m | 1y | 2y | all"),
    hot:                  float = Query(1.5, description="Z threshold for the 'hot' count"),
    exclude_extrapolated: bool  = Query(True),
    pool=Depends(get_oi_pool),
):
    """The numbers beside the universe histogram.

    Returns today's count above +`hot`, that count's own historical average
    over the window, today's universe median, and where today's cross-name
    DISPERSION ranks among the window's dispersions.

    The point of all four: if the median ticker sits at +0.4 sigma, a ticker
    at +2.0 is part of a market-wide move rather than a name-specific
    opportunity. Same number, different trade — the first mean-reverts on its
    own, the second only if the market does.

    Every date in the window contributes to the historical average, today
    included. With ~250 dates one of them moves it negligibly, and excluding
    it would make "average" mean something other than what the chart shows.
    """
    if not pool:
        return {"error": "OI database not configured"}

    cat = await _catalog(pool)
    e   = _entry(cat, metric)
    val = _expr(e)
    if exclude_extrapolated:
        val = f"CASE WHEN {_extrap_expr(e)} THEN NULL ELSE {val} END"

    async with pool.acquire() as conn:
        d, snap = await _resolve_slice(conn, date, snapshot)
        start = _window_start(d, window)

        # $1 hot, $2 snapshot, $3 date, $4 window start (when bounded).
        params = [hot, snap, d]
        date_clause = "m.trade_date <= $3"
        if start is not None:
            params.append(start)
            date_clause += " AND m.trade_date >= $4"

        rows = await conn.fetch(
            f"WITH v AS ("
            f"  SELECT m.trade_date, {val} AS v "
            f"  {_from_clause(e['form'] != 'base')} "
            f"  WHERE m.snapshot = $2 AND {date_clause}"
            f") "
            f"SELECT trade_date, "
            f"       count(*)                                       AS n, "
            f"       count(*) FILTER (WHERE v > $1)                 AS n_hot, "
            f"       percentile_cont(0.5) WITHIN GROUP (ORDER BY v) AS med, "
            f"       stddev_samp(v)                                 AS disp "
            f"FROM v WHERE v IS NOT NULL "
            f"GROUP BY trade_date ORDER BY trade_date",
            *params,
        )

    series = [{"date": str(r["trade_date"]), "n": r["n"], "n_hot": r["n_hot"],
               "median": r["med"], "dispersion": r["disp"]} for r in rows]
    today = next((s for s in series if s["date"] == str(d)), None)

    hot_avg = (sum(s["n_hot"] for s in series) / len(series)) if series else None

    # Percentile of today's dispersion among the window's, excluding itself
    # from the denominator so a single-date window reports "no rank" rather
    # than a meaningless 0.
    disp_pct = None
    if today and today["dispersion"] is not None:
        disps = [s["dispersion"] for s in series if s["dispersion"] is not None]
        if len(disps) > 1:
            below = sum(1 for v in disps if v < today["dispersion"])
            disp_pct = 100.0 * below / (len(disps) - 1)

    return {
        "date": str(d), "snapshot": snap, "window": window,
        "metric": _meta(e),
        "hot_threshold": hot,
        "today": today,
        "hot_count_avg": hot_avg,
        "dispersion_percentile": disp_pct,
        "n_dates": len(series),
        "series": series,
    }


# Scanner filter operators. Kept as an explicit table rather than passed
# through, for the same reason column names are: the op reaches SQL as text.
_OPS        = {"gt": ">", "gte": ">=", "lt": "<", "lte": "<=", "eq": "=", "ne": "<>"}
_ABS_OPS    = {"absgt": ">", "abslt": "<"}
# NULL-permissive variants. "no earnings within 10 days" is a filter on
# days_to_earnings, which is NULL for every row — no earnings source is wired
# up yet — and a plain `> 10` on NULL is false, so that filter would silently
# return nothing at all. These say "unknown counts as passing", which is what
# the question means when the data is absent.
_NULLOK_OPS = {"nullorgt": ">", "nullorlt": "<"}
_NULL_OPS   = {"isnull": "IS NULL", "notnull": "IS NOT NULL"}
_TEXT_UNITS = {"text", "timestamp"}


def _filter_sql(entry: dict, op: str, raw: str, params: list) -> str:
    """One `col:op:value` clause, with the value parameterized."""
    expr = _expr(entry)
    if op in _NULL_OPS:
        return f"{expr} {_NULL_OPS[op]}"

    if entry["units"] in _TEXT_UNITS:
        if op not in ("eq", "ne"):
            raise HTTPException(
                400, f"Operator {op!r} is numeric but {entry['column_name']} is "
                     f"{entry['units']} — use eq / ne / isnull / notnull.")
        params.append(raw)
        return f"{expr}::text {_OPS[op]} ${len(params)}"

    try:
        params.append(float(raw))
    except ValueError:
        raise HTTPException(400, f"Filter value {raw!r} is not a number")
    n = len(params)
    if op in _ABS_OPS:
        return f"abs({expr}) {_ABS_OPS[op]} ${n}"
    if op in _NULLOK_OPS:
        return f"({expr} IS NULL OR {expr} {_NULLOK_OPS[op]} ${n})"
    if op in _OPS:
        return f"{expr} {_OPS[op]} ${n}"
    raise HTTPException(400, f"Unknown filter operator: {op!r}")


@router.get("/scanner")
async def scanner(
    columns:              str  = Query(..., description="Comma-separated metric columns"),
    date:                 str  = Query(None),
    snapshot:             str  = Query(None),
    filter:               list[str] = Query([], description="Repeatable col:op:value, ANDed"),
    sort:                 str  = Query(None),
    dir:                  str  = Query("desc"),
    limit:                int  = Query(300, ge=1, le=1000),
    exclude_extrapolated: bool = Query(True),
    pool=Depends(get_oi_pool),
):
    """The scanner table: chosen columns, composed filters, one row per ticker.

    Filters are ANDed and each is `col:op:value`, so "skew z > 1.5 AND term
    ratio < 1.0 AND no earnings within 10 days" is

        filter=skew_30d_25p_atm_z_63:gt:1.5
        filter=term_ratio_30d_90d:lt:1.0
        filter=days_to_earnings:nullorgt:10

    Sorting puts NULLS LAST in both directions: a null is a missing metric,
    never an extreme one, and letting it sort to the top of a descending scan
    is how an absent value gets read as a signal.
    """
    if not pool:
        return {"error": "OI database not configured", "rows": []}

    cat  = await _catalog(pool)
    cols = [c.strip() for c in columns.split(",") if c.strip()]
    if not cols:
        raise HTTPException(400, "columns is empty")
    entries = [_entry(cat, c) for c in cols]

    sel = ["m.ticker"]
    for i, e in enumerate(entries):
        sel.append(f"{_expr(e)} AS v{i}")
        sel.append(f"{_extrap_expr(e)} AS e{i}")
    sel += ['m."{0}" AS {0}'.format(c) for c in CONTEXT_COLS]

    params: list = []
    where: list  = []
    used = list(entries)

    for f in filter:
        parts = f.split(":", 2)
        if len(parts) < 2:
            raise HTTPException(400, f"Malformed filter: {f!r} (want col:op:value)")
        fe = _entry(cat, parts[0])
        used.append(fe)
        where.append(_filter_sql(fe, parts[1], parts[2] if len(parts) > 2 else "", params))

    order = "m.ticker"
    if sort:
        se = _entry(cat, sort)
        used.append(se)
        order = f"{_expr(se)} {'DESC' if dir.lower() == 'desc' else 'ASC'} NULLS LAST, m.ticker"

    async with pool.acquire() as conn:
        d, snap = await _resolve_slice(conn, date, snapshot)
        params += [d, snap, limit]
        nd, ns, nl = len(params) - 2, len(params) - 1, len(params)
        clause = f"m.trade_date = ${nd} AND m.snapshot = ${ns}"
        if where:
            clause += " AND " + " AND ".join(where)
        rows = await conn.fetch(
            f"SELECT {', '.join(sel)} "
            f"{_from_clause(_needs_z(used))} "
            f"WHERE {clause} "
            f"ORDER BY {order} "
            f"LIMIT ${nl}",
            *params,
        )

    out = []
    for r in rows:
        vals, flags = {}, {}
        for i, e in enumerate(entries):
            col, v, ex = e["column_name"], r[f"v{i}"], r[f"e{i}"]
            flags[col] = ex
            vals[col]  = None if (exclude_extrapolated and ex) else v
        out.append({
            "ticker":      r["ticker"],
            "values":      vals,
            "extrap":      flags,
            "spot":        r["spot"],
            "extrap_rate": r["extrap_rate_short"],
            "liquidity":   r["median_n_strikes_clean"],
            "source":      r["source"],
            "captured_at": _jsonable(r["captured_at"]),
        })

    return {
        "date": str(d), "snapshot": snap,
        "columns": [_meta(e) for e in entries],
        "rows":      out,
        "n_rows":    len(out),
        "truncated": len(out) >= limit,
        "exclude_extrapolated": bool(exclude_extrapolated),
    }
