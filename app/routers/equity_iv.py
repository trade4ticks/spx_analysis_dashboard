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

    _catalog_cache = {
        "by_col": by_col,
        "extrap_cols": extrap_cols,
        # Every column that really exists on equity_metrics, catalogued or
        # not. The header reads context columns (spot, atm iv, 50dma, ...)
        # that are not metrics and so have no catalog row, and this is what
        # lets it ask for them without either hardcoding a name that might
        # not exist or opening a hole in the identifier whitelist.
        "live_metric_cols": {c for (t, c) in live if t == METRICS_TABLE},
    }
    return _catalog_cache


def _first_live(cat: dict, *candidates: str):
    """First candidate column that exists on equity_metrics, else None.

    The header wants quantities whose exact column name this code cannot
    verify from here. Naming one and hoping is how a page 500s on a rename;
    naming several and taking the first real one degrades to "absent"
    instead, which is the same thing NULL already has to render as.
    """
    live = cat["live_metric_cols"]
    for c in candidates:
        if c in live:
            return c
    return None


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


# ── ticker half ──────────────────────────────────────────────────────────────
#
# Everything below is scoped to ONE ticker, so the queries are small and the
# history windows can be generous. Two rules carry through all of them:
#
#   Per-metric extrapolation, never a ticker rate. A name can be 40%
#   extrapolated chain-wide while the two nodes a given metric rests on are
#   both real — AAL is exactly that. Each value resolves its OWN flags via
#   _extrap_expr, and the chain rate travels separately as context.
#
#   Extrapolated observations are excluded from HISTORICAL DISTRIBUTIONS, not
#   just from today. A percentile band computed over fabricated values
#   describes a normal range that never existed.


# Context columns for the header. Several are quantities this code cannot
# confirm the name of, so each is resolved against the live table by
# _first_live and renders as absent when no candidate exists — the same
# treatment a legitimately NULL metric gets.
HEADER_CANDIDATES = {
    "atm_iv":        ("iv_30d_atm", "atm_iv_30d", "atm_iv"),
    "rv":            ("rv_21d", "rv_1m", "realized_vol_21d", "rv_20d"),
    "term_ratio":    ("term_ratio_30d_90d", "term_ratio_30d_60d"),
    "px_vs_50dma":   ("px_vs_50dma", "price_vs_50dma", "spot_vs_50dma"),
    "days_to_earn":  ("days_to_earnings",),
    "spotvol_beta":  ("spotvol_beta_1m", "spotvol_beta_3m"),
    "spotvol_r2":    ("spotvol_r2_1m", "spotvol_r2_3m"),
}


@router.get("/ticker-header")
async def ticker_header(
    ticker:   str = Query(...),
    date:     str = Query(None),
    snapshot: str = Query(None),
    pool=Depends(get_oi_pool),
):
    """The thin sticky header: spot, ATM IV, RV, and the state chips.

    `days_to_earnings` is currently NULL on every row — no calendar source is
    wired up — so it is returned as null and must render as absent. That is
    the same contract every other metric has, and the reason it is not
    special-cased into a zero.

    `source` distinguishes a 'live' row, captured at an arbitrary instant and
    rounded to the grid bucket, from an 'exact' row out of the anchored
    historical record; `captured_at` holds the true instant. Both are
    returned because a header that says 15:45 when the capture happened at
    15:47:31 is quietly wrong about when it is describing.
    """
    if not pool:
        return {"error": "OI database not configured"}

    cat = await _catalog(pool)
    resolved = {k: _first_live(cat, *cands) for k, cands in HEADER_CANDIDATES.items()}

    sel = ["m.ticker", "m.spot", "m.extrap_rate_short", "m.source",
           "m.captured_at", "m.median_n_strikes_clean"]
    for key, col in resolved.items():
        if col:
            sel.append('m."{}" AS {}'.format(col, key))

    async with pool.acquire() as conn:
        d, snap = await _resolve_slice(conn, date, snapshot)
        row = await conn.fetchrow(
            f"SELECT {', '.join(sel)} FROM {METRICS_TABLE} m "
            f"WHERE m.ticker = $1 AND m.trade_date = $2 AND m.snapshot = $3",
            ticker, d, snap,
        )

    if row is None:
        return {"error": f"No row for {ticker} at {d} {snap}",
                "ticker": ticker, "date": str(d), "snapshot": snap}

    def g(key):
        return row[key] if resolved.get(key) else None

    term = g("term_ratio")
    return {
        "ticker": row["ticker"], "date": str(d), "snapshot": snap,
        "spot":   row["spot"],
        "atm_iv": g("atm_iv"),
        "rv":     g("rv"),
        # Contango when the far tenor is richer than the near one. The ratio
        # is near/far, so < 1 is contango. Null in, null out.
        "term_ratio": term,
        "term_state": None if term is None else ("contango" if term < 1 else "backwardation"),
        "px_vs_50dma":   g("px_vs_50dma"),
        "days_to_earnings": g("days_to_earn"),
        "spotvol_beta":  g("spotvol_beta"),
        "spotvol_r2":    g("spotvol_r2"),
        "extrap_rate":   row["extrap_rate_short"],
        "liquidity":     row["median_n_strikes_clean"],
        "source":        row["source"],
        "captured_at":   _jsonable(row["captured_at"]),
        # Which candidate answered each slot, so the client can label the chip
        # with the column it is actually showing rather than a generic word.
        "resolved":      resolved,
    }


@router.get("/unusual")
async def unusual(
    ticker:               str  = Query(...),
    date:                 str  = Query(None),
    snapshot:             str  = Query(None),
    z_window:             int  = Query(63),
    window:               str  = Query("1y"),
    limit:                int  = Query(40, ge=1, le=200),
    families:             str  = Query(None, description="CSV family filter"),
    exclude_extrapolated: bool = Query(True),
    pool=Depends(get_oi_pool),
):
    """Today's metrics for one ticker, ranked by |z|. Row 4's card strip.

    This is the discovery mechanism the page turns on: ~600 metric columns is
    far more than anyone can page through, so the question "which of these is
    extreme today" has to be answered by the server before the user has to
    know which column to ask about.

    Ranked on |z| from the requested z window. Percentile is computed over
    the page's history window from the BASE column, with extrapolated
    observations dropped — a percentile that counts fabricated history is
    describing a normal range that never occurred.

    A metric whose own nodes are extrapolated today is excluded from the
    ranking under the toggle rather than shown with a marker: this list is
    sorted BY extremeness, and a fabricated node is exactly what manufactures
    a spurious extreme. It stays in the payload, flagged, so the client can
    show what was withheld.
    """
    if not pool:
        return {"error": "OI database not configured", "cards": []}
    if z_window not in (63, 252):
        raise HTTPException(400, f"z_window must be 63 or 252, got {z_window}")

    cat  = await _catalog(pool)
    form = f"z_{z_window}"
    fams = {f.strip() for f in families.split(",") if f.strip()} if families else None

    # Every z column at this window whose base column also exists — the card
    # needs the value, the z and the percentile, and a z with no base has no
    # value to show.
    pairs = []
    for e in cat["by_col"].values():
        if e["form"] != form:
            continue
        base = e["base_column"]
        if not base or base not in cat["by_col"]:
            continue
        if fams and e["family"] not in fams:
            continue
        pairs.append((cat["by_col"][base], e))
    if not pairs:
        return {"error": f"No {form} columns in the catalog", "cards": []}

    sel = ["m.extrap_rate_short"]
    for i, (be, ze) in enumerate(pairs):
        sel += [f"{_expr(be)} AS b{i}", f"{_expr(ze)} AS z{i}",
                f"{_extrap_expr(be)} AS e{i}"]

    async with pool.acquire() as conn:
        d, snap = await _resolve_slice(conn, date, snapshot)
        row = await conn.fetchrow(
            f"SELECT {', '.join(sel)} {_from_clause(True)} "
            f"WHERE m.ticker = $1 AND m.trade_date = $2 AND m.snapshot = $3",
            ticker, d, snap,
        )
        if row is None:
            return {"ticker": ticker, "date": str(d), "snapshot": snap,
                    "cards": [], "error": f"No row for {ticker}"}

        # Rank first, then price percentiles only for what will be shown.
        # Percentile needs a window scan per column; doing all ~368 would be
        # ~368 scans to populate a strip that shows a few dozen.
        ranked = []
        for i, (be, ze) in enumerate(pairs):
            zv, bv, ex = row[f"z{i}"], row[f"b{i}"], row[f"e{i}"]
            if zv is None or bv is None:
                continue
            ranked.append({"i": i, "base": be, "z_entry": ze,
                           "value": float(bv), "z": float(zv), "extrap": bool(ex)})
        ranked.sort(key=lambda c: abs(c["z"]), reverse=True)

        shown = [c for c in ranked if not (exclude_extrapolated and c["extrap"])]
        withheld = len(ranked) - len(shown)
        shown = shown[:limit]

        start = _window_start(d, window)
        for c in shown:
            be = c["base"]
            # Percentile over the SAME snapshot, so an 09:45 reading is
            # ranked against other 09:45 readings rather than against a day
            # that also contains the close.
            pct_sql = (
                f"SELECT count(*) FILTER (WHERE {_expr(be)} <= $4)::float8 "
                f"       / NULLIF(count(*) FILTER (WHERE {_expr(be)} IS NOT NULL), 0) "
                f"{_from_clause(be['form'] != 'base')} "
                f"WHERE m.ticker = $1 AND m.snapshot = $2 AND m.trade_date <= $3 "
                + ("AND m.trade_date >= $5 " if start else "")
                + f"AND NOT {_extrap_expr(be)}"
            )
            args = [ticker, snap, d, c["value"]] + ([start] if start else [])
            c["percentile"] = await conn.fetchval(pct_sql, *args)

    cards = [{
        "column":      c["base"]["column_name"],
        "z_column":    c["z_entry"]["column_name"],
        "family":      c["base"]["family"],
        "tenor":       c["base"]["tenor"],
        "wing":        c["base"]["wing"],
        "units":       c["base"]["units"],
        "description": c["base"]["description"],
        "value":       c["value"],
        "z":           c["z"],
        "percentile":  c["percentile"],
        "extrap":      c["extrap"],
        "extrap_flags": c["base"]["extrap_flags"],
    } for c in shown]

    return {
        "ticker": ticker, "date": str(d), "snapshot": snap,
        "z_window": z_window, "window": window,
        "cards": cards,
        "n_ranked": len(ranked),
        "n_withheld_extrapolated": withheld,
        "exclude_extrapolated": bool(exclude_extrapolated),
        "extrap_rate": row["extrap_rate_short"],
    }


RAILS_DEFAULT = (
    "skew_30d_25p_atm", "skew_30d_10p_atm", "iv_30d_atm",
    "term_ratio_30d_90d", "rr_30d_25d", "zc_width_sigma_30d",
    "vrp_1m", "convexity_30d_25p_atm_25c",
)


@router.get("/rails")
async def rails(
    ticker:               str  = Query(...),
    metrics:              str  = Query(None, description="CSV of base columns"),
    date:                 str  = Query(None),
    snapshot:             str  = Query(None),
    window:               str  = Query("1y"),
    z_window:             int  = Query(63),
    exclude_extrapolated: bool = Query(True),
    pool=Depends(get_oi_pool),
):
    """Row 5's rails: one horizontal distribution bar per metric.

    PERCENTILES, NOT STANDARD DEVIATIONS, and that is a deliberate choice
    rather than a stylistic one. Skew distributions are right-skewed and
    fat-tailed, so a symmetric +/-2SD band is wrong asymmetrically -- too
    wide on the left and too narrow on the right, and the right is the tail
    that matters for a wing that has run. P5/P25/P50/P75/P95 makes no
    distributional assumption at all.

    The z is still returned, from the stored z column, because "where in the
    range" and "how many sigma" answer different questions and the header
    chips speak in sigma. It is a label beside the bar, not the bar's
    geometry.

    Extrapolated observations are dropped from the distribution AND from
    today's marker under the toggle. Leaving them in the history would define
    a normal range partly out of values the spline invented.
    """
    if not pool:
        return {"error": "OI database not configured", "rails": []}

    cat  = await _catalog(pool)
    cols = [c.strip() for c in metrics.split(",") if c.strip()] if metrics \
        else [c for c in RAILS_DEFAULT if c in cat["by_col"]]
    if not cols:
        return {"error": "No usable rail metrics", "rails": []}
    entries = [_entry(cat, c) for c in cols]

    out = []
    async with pool.acquire() as conn:
        d, snap = await _resolve_slice(conn, date, snapshot)
        start = _window_start(d, window)
        for e in entries:
            zcol = None
            for cand in cat["by_col"].values():
                if cand["form"] == f"z_{z_window}" and cand["base_column"] == e["column_name"]:
                    zcol = cand
                    break

            sel = [f"{_expr(e)} AS v", f"{_extrap_expr(e)} AS ex"]
            if zcol:
                sel.append(f"{_expr(zcol)} AS z")
            cur = await conn.fetchrow(
                f"SELECT {', '.join(sel)} "
                f"{_from_clause(e['form'] != 'base' or bool(zcol))} "
                f"WHERE m.ticker = $1 AND m.trade_date = $2 AND m.snapshot = $3",
                ticker, d, snap,
            )

            dist_sql = (
                f"SELECT percentile_cont(0.05) WITHIN GROUP (ORDER BY {_expr(e)}) AS p5, "
                f"       percentile_cont(0.25) WITHIN GROUP (ORDER BY {_expr(e)}) AS p25, "
                f"       percentile_cont(0.50) WITHIN GROUP (ORDER BY {_expr(e)}) AS p50, "
                f"       percentile_cont(0.75) WITHIN GROUP (ORDER BY {_expr(e)}) AS p75, "
                f"       percentile_cont(0.95) WITHIN GROUP (ORDER BY {_expr(e)}) AS p95, "
                f"       count({_expr(e)}) AS n "
                f"{_from_clause(e['form'] != 'base')} "
                f"WHERE m.ticker = $1 AND m.snapshot = $2 AND m.trade_date <= $3 "
                + ("AND m.trade_date >= $4 " if start else "")
                + ("AND NOT " + _extrap_expr(e) if exclude_extrapolated else "")
            )
            args = [ticker, snap, d] + ([start] if start else [])
            dist = await conn.fetchrow(dist_sql, *args)

            v  = cur["v"] if cur else None
            ex = bool(cur["ex"]) if cur else False
            shown = None if (v is None or (exclude_extrapolated and ex)) else float(v)

            pct = None
            if shown is not None and dist and dist["n"]:
                pct_sql = (
                    f"SELECT count(*) FILTER (WHERE {_expr(e)} <= $4)::float8 "
                    f"       / NULLIF(count({_expr(e)}), 0) "
                    f"{_from_clause(e['form'] != 'base')} "
                    f"WHERE m.ticker = $1 AND m.snapshot = $2 AND m.trade_date <= $3 "
                    + ("AND m.trade_date >= $5 " if start else "")
                    + ("AND NOT " + _extrap_expr(e) if exclude_extrapolated else "")
                )
                pargs = [ticker, snap, d, shown] + ([start] if start else [])
                pct = await conn.fetchval(pct_sql, *pargs)

            out.append({
                **_meta(e),
                "value":      shown,
                "raw_value":  None if v is None else float(v),
                "extrap":     ex,
                "z":          (float(cur["z"]) if (zcol and cur and cur["z"] is not None) else None),
                "z_column":   zcol["column_name"] if zcol else None,
                "percentile": pct,
                "p5":  _f(dist, "p5"),  "p25": _f(dist, "p25"),
                "p50": _f(dist, "p50"), "p75": _f(dist, "p75"),
                "p95": _f(dist, "p95"),
                "n":   int(dist["n"]) if dist and dist["n"] else 0,
            })

    return {
        "ticker": ticker, "date": str(d), "snapshot": snap,
        "window": window, "z_window": z_window,
        "rails": out,
        "exclude_extrapolated": bool(exclude_extrapolated),
    }


def _f(row, key):
    """float() a percentile_cont result, tolerating a null row or null value."""
    if row is None or row[key] is None:
        return None
    return float(row[key])

DAILY_SNAPSHOT = "1545"
SERIES_MAX     = 4


async def _daily_baseline(conn, entry, ticker, as_of, snap, z_window):
    """(mu, sigma, n) for one metric over the DAILY close series.

    This is the single source of z for /series, in every mode. The reason is
    the intraday toggle: an intraday reading normalised against intraday
    history would be measured against a baseline that starts on 2026-08-24
    and is sparse before 11:25 that day. A z of 2.4 would then mean "extreme
    relative to the last few hours", while the same number on the daily view
    means "extreme relative to the last quarter" — same label, two scales,
    and the user reading across the toggle has no way to see the swap.

    So intraday z is (v - mu) / sigma against the daily close baseline: the
    intraday point moves, the yardstick does not.

    Deriving daily z the same way rather than reading the stored z column is
    deliberate too — mixing estimators across the toggle reintroduces the
    discontinuity by a different route. The stored value is returned beside
    it as `z_stored` so a divergence shows up instead of hiding.
    """
    sql = (
        f"SELECT avg(v) AS mu, stddev_samp(v) AS sd, count(*) AS n FROM ("
        f"  SELECT {_expr(entry)} AS v "
        f"  {_from_clause(entry['form'] != 'base')} "
        f"  WHERE m.ticker = $1 AND m.snapshot = $2 AND m.trade_date <= $3 "
        f"    AND NOT {_extrap_expr(entry)} AND {_expr(entry)} IS NOT NULL "
        f"  ORDER BY m.trade_date DESC LIMIT $4"
        f") s"
    )
    r = await conn.fetchrow(sql, ticker, snap, as_of, z_window)
    mu = None if r["mu"] is None else float(r["mu"])
    sd = None if r["sd"] is None else float(r["sd"])
    return mu, sd, int(r["n"] or 0)


def _quantile(sorted_vals, p):
    """Linear-interpolated quantile of an already-sorted list."""
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    pos  = p * (len(sorted_vals) - 1)
    lo_i = int(pos)
    hi_i = min(lo_i + 1, len(sorted_vals) - 1)
    return sorted_vals[lo_i] + (sorted_vals[hi_i] - sorted_vals[lo_i]) * (pos - lo_i)


def _rolling_pct_envelope(vals, win, lo_q, hi_q):
    """Trailing percentile band, one (lo, hi) per point, aligned to `vals`.

    Trailing rather than centred: a centred window at index i uses values
    from after i, so the band at any historical point would be drawn partly
    out of the future. On a chart whose whole purpose is "was this unusual
    AT THE TIME", that is the wrong band.

    The band is computed from history strictly BEFORE each point, then the
    point is admitted — so a marker is never inside the distribution it is
    being judged against.

    Nulls and (when excluded) extrapolated observations arrive as None and
    never enter `hist`, so the band describes observed history only.
    """
    out, hist = [], []
    floor = max(8, win // 4)
    for v in vals:
        if len(hist) >= floor:
            w = sorted(hist[-win:])
            out.append((_quantile(w, lo_q), _quantile(w, hi_q)))
        else:
            out.append((None, None))
        if v is not None:
            hist.append(v)
    return out


@router.get("/series")
async def series(
    ticker:               str   = Query(...),
    metrics:              str   = Query(..., description="CSV of columns, max 4"),
    mode:                 str   = Query("daily", description="daily|intraday|candle"),
    snapshot:             str   = Query(None, description="daily mode bucket"),
    window:               str   = Query("1y"),
    z_window:             int   = Query(63),
    envelope:             bool  = Query(True),
    env_window:           int   = Query(63, ge=8, le=504),
    env_lo:               float = Query(0.10, ge=0.0, le=0.5),
    env_hi:               float = Query(0.90, ge=0.5, le=1.0),
    exclude_extrapolated: bool  = Query(True),
    pool=Depends(get_oi_pool),
):
    """Row 5's time series. Up to 4 metrics; the client assigns axis and pane.

    mode:
      daily    — one point per trade_date at `snapshot` (default 1545 close)
      intraday — every snapshot bucket, ordered (trade_date, snapshot)
      candle   — daily OHLC OF THE METRIC, built from the intraday buckets
                 inside each day: open = first bucket, close = last, high and
                 low the extremes. This needs intraday coverage to exist,
                 which begins 2026-08-24 and is sparse before 11:25 that day,
                 so candle mode returns few bars by construction. `n_points`
                 and `first_date` say how few, rather than drawing a stub
                 chart that reads as a loading failure.

    z in every mode comes from the daily baseline — see _daily_baseline().
    """
    if not pool:
        return {"error": "OI database not configured", "series": []}
    if mode not in ("daily", "intraday", "candle"):
        raise HTTPException(400, f"mode must be daily|intraday|candle, got {mode!r}")
    if env_lo >= env_hi:
        raise HTTPException(400, f"env_lo {env_lo} must be below env_hi {env_hi}")

    cat  = await _catalog(pool)
    cols = [c.strip() for c in metrics.split(",") if c.strip()]
    if not cols:
        raise HTTPException(400, "No metrics requested")
    if len(cols) > SERIES_MAX:
        raise HTTPException(400, f"At most {SERIES_MAX} series, got {len(cols)}")
    entries = [_entry(cat, c) for c in cols]

    out = []
    async with pool.acquire() as conn:
        as_of, latest_snap = await _resolve_slice(conn, None, None)
        daily_snap = snapshot or DAILY_SNAPSHOT
        start      = _window_start(as_of, window)

        for e in entries:
            needs_z = e["form"] != "base"
            zcol    = None
            for cand in cat["by_col"].values():
                if cand["form"] == f"z_{z_window}" and cand["base_column"] == e["column_name"]:
                    zcol = cand
                    break

            sel = ["m.trade_date", "m.snapshot", f"{_expr(e)} AS v",
                   f"{_extrap_expr(e)} AS ex"]
            want_stored = bool(zcol) and mode == "daily"
            if want_stored:
                sel.append(f"{_expr(zcol)} AS zs")
                needs_z = True

            where = ["m.ticker = $1"]
            args  = [ticker]
            if mode == "daily":
                args.append(daily_snap)
                where.append(f"m.snapshot = ${len(args)}")
            if start:
                args.append(start)
                where.append(f"m.trade_date >= ${len(args)}")

            rows = await conn.fetch(
                f"SELECT {', '.join(sel)} {_from_clause(needs_z)} "
                f"WHERE {' AND '.join(where)} "
                f"ORDER BY m.trade_date, m.snapshot",
                *args,
            )

            mu, sd, n_base = await _daily_baseline(
                conn, e, ticker, as_of, daily_snap, z_window)

            def zof(v, _mu=mu, _sd=sd):
                if v is None or _mu is None or not _sd:
                    return None
                return (v - _mu) / _sd

            if mode == "candle":
                by_day = {}
                for r in rows:
                    if r["v"] is None or (exclude_extrapolated and r["ex"]):
                        continue
                    by_day.setdefault(r["trade_date"], []).append(float(r["v"]))
                pts = []
                for d in sorted(by_day):
                    vs = by_day[d]                    # already snapshot-ordered
                    pts.append({"t": d.isoformat(), "o": vs[0], "h": max(vs),
                                "l": min(vs), "c": vs[-1], "n": len(vs),
                                "z": zof(vs[-1])})
                closes = [p["c"] for p in pts]
            else:
                pts = []
                for r in rows:
                    v     = None if r["v"] is None else float(r["v"])
                    ex    = bool(r["ex"])
                    shown = None if (v is None or (exclude_extrapolated and ex)) else v
                    p = {"t": r["trade_date"].isoformat(), "v": shown, "extrap": ex,
                         "z": zof(shown)}
                    if mode == "intraday":
                        p["snapshot"] = r["snapshot"]
                    if want_stored:
                        p["z_stored"] = None if r["zs"] is None else float(r["zs"])
                    pts.append(p)
                closes = [p["v"] for p in pts]

            if envelope and pts:
                band = _rolling_pct_envelope(closes, env_window, env_lo, env_hi)
                for p, (lo, hi) in zip(pts, band):
                    p["env_lo"], p["env_hi"] = lo, hi

            out.append({
                **_meta(e),
                "points":     pts,
                "n_points":   len(pts),
                "first_date": pts[0]["t"] if pts else None,
                "last_date":  pts[-1]["t"] if pts else None,
                "baseline":   {"mu": mu, "sigma": sd, "n": n_base,
                               "snapshot": daily_snap, "as_of": as_of.isoformat(),
                               "z_window": z_window},
                "z_stored_column": zcol["column_name"] if want_stored else None,
            })

    return {
        "ticker": ticker, "mode": mode, "window": window, "z_window": z_window,
        "snapshot": daily_snap if mode == "daily" else None,
        "latest_snapshot": latest_snap, "as_of": as_of.isoformat(),
        "z_source": "daily_baseline",
        "envelope": ({"on": True, "window": env_window, "lo": env_lo, "hi": env_hi}
                     if envelope else {"on": False}),
        "exclude_extrapolated": bool(exclude_extrapolated),
        "series": out,
    }
