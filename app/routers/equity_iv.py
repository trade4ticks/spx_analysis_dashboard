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

equity_metrics_z is READ for every z on this page. It stores each snapshot's
z against the ticker's 1545 daily series, with the scored date excluded from
its own window — which is the definition this router previously computed for
itself, and the reason that derivation is gone.

Expect two things from the change. 1545 z-scores grow slightly in magnitude,
around 3-4% on SPY, because removing self-inclusion stops each value pulling
its own mean toward itself. And intraday buckets now carry a z immediately
instead of NULL, since they are scored against 1545 history rather than their
own — with the mean bucket-versus-close drift of the session baked in, which
is uniform within a bucket and so shifts the distribution rather than
reordering tickers inside it.

BASELINE_SNAPSHOT and BASELINE_MIN_N come from app/metrics_config.py, vendored
verbatim from the metrics project and diffed by scripts/check_vendored.py. Two
copies of a baseline definition is what produced two divergent estimators the
first time.

What this page still computes, because equity_metrics does not carry it:
percentile ranks and P5-P95 bands, the rolling envelope on /series, and
everything read off equity_surface.

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
from app.equity_presets import COLUMN_ALIASES, PAIR_FAMILIES, PAIR_FOR_TENOR
from app.metrics_config import TENORS as TENORS_GRID

router = APIRouter(tags=["equity-iv"])

METRICS_TABLE = "equity_metrics"
Z_TABLE       = "equity_metrics_z"
CATALOG_TABLE = "equity_metrics_catalog"
ATM_TABLE       = "equity_atm"
EARNINGS_TABLE  = "earnings_calendar"
EARNINGS_COVERAGE_TABLE = "earnings_coverage"

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
    Wings come from WING_NODES; the tenor comes from the catalog rather than
    the name, since a name can carry a window label that is not its tenor.
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


def _extrap_expr(entry: dict, alias: str = "m") -> str:
    """Boolean SQL: is any node this metric depends on extrapolated?

    COALESCE to false because a NULL flag means the node was never evaluated,
    which is a null metric — the "is this fabricated" question is then moot,
    and answering TRUE would count it as an exclusion on top of being absent.

    `alias` exists because the baseline CTE reads the same flag columns under
    its own alias while scoring history, not today's row.
    """
    flags = entry["extrap_flags"]
    if not flags:
        return "FALSE"
    return "(" + " OR ".join(
        'COALESCE({}."{}", false)'.format(alias, f) for f in flags) + ")"


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

# ── the daily baseline ───────────────────────────────────────────────────────
#
# equity_metrics_z now stores exactly this definition: every snapshot's z
# scored against the ticker's 1545 daily series, with the scored date excluded
# from its own window. So the dashboard reads it rather than recomputing it,
# and the derivation that used to live here is gone — _baseline_cte, _z_expr,
# _baseline_for, _z_from, _daily_baseline, _base_entry, _baseline_join,
# _session_span_days and _reject_z_form with it.
#
# The two constants come from the metrics project's own config, vendored
# verbatim as app/metrics_config.py and diffed by scripts/check_vendored.py.
# Re-declaring them here is what produced two divergent estimators the first
# time; a local copy would move that duplication rather than remove it.
#
# What this page still computes for itself, because equity_metrics does not
# carry it: percentile ranks and P5–P95 bands (rails, unusual, tent), the
# rolling envelope on /series, and everything read off equity_surface — the
# per-node grid views, the curve bands and the sticky-strike counterfactual.
# Those are not z-scores and have no stored equivalent.
from app.metrics_config import BASELINE_SNAPSHOT, BASELINE_MIN_N


def _needs_z(entries) -> bool:
    """True when any entry lives in equity_metrics_z, so the join is needed."""
    return any(e["form"] != "base" for e in entries)


def _z_column(cat: dict, base_col: str, z_window: int):
    """The stored z column for a base metric at this window, or None.

    Not every metric is z-scored — metrics_config excludes whole families,
    price levels and fallback rates among them, on the grounds that a rolling
    z of a trending level is not a reading anyone wants. A base column with no
    z variant returns None and the caller renders it as absent.
    """
    want = f"z_{z_window}"
    for e in cat["by_col"].values():
        if e["form"] == want and e["base_column"] == base_col:
            return e
    return None


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
        "tenors_grid": list(TENORS_GRID),
        # The two tables the tenor-retarget rule needs. Defined once in
        # app/equity_presets.py and shipped from here, so the algorithm is
        # written per runtime but its DATA is not duplicated -- which is the
        # half of it that actually drifts.
        "aliases":       COLUMN_ALIASES,
        "pair_families": sorted(PAIR_FAMILIES),
        "pair_for_tenor": {str(k): list(v) for k, v in PAIR_FOR_TENOR.items()},
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

    A z-form axis is READ from equity_metrics_z. That column now stores every
    snapshot's z against the ticker's 1545 daily series with the scored date
    excluded, which is the definition this router used to compute for itself.
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
        "z_source":  "stored" if _needs_z(used) else None,
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

    Returns today's counts BEYOND EACH TAIL (above +`hot` and below -`hot`),
    the upper count's own historical average over the window, today's
    universe median, and where today's cross-name DISPERSION ranks among the
    window's dispersions.

    Both tails because breadth in one direction is not the complement of
    breadth in the other: 8% rich and 6% cheap is a dispersed universe, 8%
    rich and 0% cheap is a directional one, and a single line cannot tell
    those apart. `hot` is symmetric by construction -- a separate `cold`
    threshold would let the two lines answer different questions.

    The point of all four: if the median ticker sits at +0.4 sigma, a ticker
    at +2.0 is part of a market-wide move rather than a name-specific
    opportunity. Same number, different trade — the first mean-reverts on its
    own, the second only if the market does.

    Two populations, deliberately:

      history  prior sessions at the daily close (BASELINE_SNAPSHOT).
      today    the SELECTED snapshot.

    Both read the stored z, which is scored against the ticker's 1545 daily
    series whatever bucket it sits in -- so an 11:25 count is comparable with
    the daily counts beside it without this endpoint rescoring anything.

    The history stops before today because today is the row being placed
    against it, and hot_count_avg therefore excludes today. At ~250 dates
    that moves it negligibly, and today is measured at a different bucket, so
    averaging it in would mix two populations to save a rounding error.
    """
    if not pool:
        return {"error": "OI database not configured"}

    cat = await _catalog(pool)
    e   = _entry(cat, metric)

    # Fabricated observations are nulled rather than their rows dropped, so a
    # ticker thin on this metric leaves the others in the cross-section alone.
    val = _expr(e)
    if exclude_extrapolated:
        val = f'CASE WHEN {_extrap_expr(e)} THEN NULL ELSE {val} END'

    async with pool.acquire() as conn:
        d, snap = await _resolve_slice(conn, date, snapshot)
        start   = _window_start(d, window)
        params = [hot, BASELINE_SNAPSHOT, d]
        where  = "m.snapshot = $2 AND m.trade_date < $3"
        if start is not None:
            params.append(start)
            where += f" AND m.trade_date >= ${len(params)}"

        rows = await conn.fetch(
            f"SELECT m.trade_date,"
            f" count({val})                                       AS n,"
            f" count(*) FILTER (WHERE {val} > $1)                  AS n_hot,"
            f" count(*) FILTER (WHERE {val} < -$1)                 AS n_cold,"
            f" percentile_cont(0.5) WITHIN GROUP (ORDER BY {val})  AS med,"
            f" stddev_samp({val})                                  AS disp "
            f"{_from_clause(e['form'] != 'base')} WHERE {where} "
            f"GROUP BY m.trade_date ORDER BY m.trade_date",
            *params,
        )

        # Today's per-ticker readings, for the histogram's bars.
        #
        # These have to come from HERE rather than from /cross-section. The
        # panel used to bin the scatter's payload, so it silently plotted
        # whatever the scatter's x axis was -- including whatever a structure
        # preset had just set it to. One column at ~500 rows is a smaller
        # response than the two-column cross-section it replaces.
        # No `hot` here, so the placeholders start at $1: an unused $1 is not
        # merely wasteful, Postgres cannot infer its type and the whole
        # statement fails to prepare.
        vrows = await conn.fetch(
            f"SELECT m.ticker, {val} AS v "
            f"{_from_clause(e['form'] != 'base')} "
            f"WHERE m.trade_date = $1 AND m.snapshot = $2 AND {val} IS NOT NULL "
            f"ORDER BY m.ticker",
            d, snap,
        )

        # Today, at the selected snapshot.
        trow = await conn.fetchrow(
            f"SELECT count({val}) AS n,"
            f" count(*) FILTER (WHERE {val} > $1) AS n_hot,"
            f" count(*) FILTER (WHERE {val} < -$1) AS n_cold,"
            f" percentile_cont(0.5) WITHIN GROUP (ORDER BY {val}) AS med,"
            f" stddev_samp({val}) AS disp "
            f"{_from_clause(e['form'] != 'base')} "
            f"WHERE m.trade_date = $2 AND m.snapshot = $3",
            hot, d, snap,
        )

    series = [{"date": str(r["trade_date"]), "n": r["n"], "n_hot": r["n_hot"],
               "n_cold": r["n_cold"],
               "median": r["med"], "dispersion": r["disp"]} for r in rows]

    today = None
    if trow is not None and trow["n"]:
        today = {"date": str(d), "n": trow["n"], "n_hot": trow["n_hot"],
                 "n_cold": trow["n_cold"],
                 "median": trow["med"], "dispersion": trow["disp"]}

    hot_avg = (sum(s["n_hot"] for s in series) / len(series)) if series else None

    # Today is not in `series`, so nothing has to be excluded from the
    # denominator — the old off-by-one guard is gone with the shape.
    disp_pct = None
    if today and today["dispersion"] is not None:
        disps = [s["dispersion"] for s in series if s["dispersion"] is not None]
        if disps:
            below = sum(1 for v in disps if v < today["dispersion"])
            disp_pct = 100.0 * below / len(disps)

    return {
        "date": str(d), "snapshot": snap, "window": window,
        "metric": _meta(e),
        "hot_threshold": hot,
        "today": today,
        "hot_count_avg": hot_avg,
        "dispersion_percentile": disp_pct,
        "n_dates": len(series),
        "series": series,
        "today_rows": [{"ticker": r["ticker"], "v": r["v"]} for r in vrows],
        "z_source": "stored" if e["form"] != "base" else None,
        "history_basis": {"snapshot": BASELINE_SNAPSHOT, "through": "prior session"},
    }


@router.get("/universe-spot-breadth")
async def universe_spot_breadth(
    date:     str = Query(None),
    snapshot: str = Query(None),
    pool=Depends(get_oi_pool),
):
    """How much of the universe is up, bucket by bucket, TODAY.

    Every other panel on this page reads a vol metric. This one reads price,
    and it is the only panel that moves through the session: `log_ret_*` is a
    settled daily-close series ending at the PRIOR session's close, so at
    11:25 it still reports yesterday's move and will keep reporting it until
    tonight's close is written. Nothing else here answers "what is the tape
    doing right now".

    TWO ANCHORS, because they are different questions:

      vs prior close   the overnight gap plus the day. Available from the
                       first bucket, since last night's close is settled.
      vs today's open  the day alone, gap removed. 60% above yesterday's
                       close with 45% above the open is a market that gapped
                       up and has been sold since.

    The open comes from `underlying_ohlc.open`, NOT from the 0935 snapshot.
    The first snapshot bucket is minutes into the session and already carries
    whatever moved in those minutes; the daily bar's open is the auction
    print. They are not the same number and only one of them is the open.

    The cost of that choice: `underlying_ohlc` is a DAILY table, so today's
    row does not exist until the bar is written after the close. The
    open-anchored series is therefore absent for the whole live session and
    `open_pending` says so, rather than the panel silently drawing one line
    and looking complete.

    FUNDS ARE NOT IN THE BREADTH COUNT. `earnings_coverage.has_earnings` is
    false for a fund -- the one discriminator this database has -- and index
    ETFs are weighted baskets OF the universe being counted, so including
    them double-counts their constituents and compresses the very dispersion
    the panel exists to show. They are counted out and reported as
    `n_fund`, not returned: this row is about implied vol across the covered
    names, and an index reference belongs on a market view built for that
    rather than borrowed into this one.

    SPLITS. `underlying_ohlc` is back-adjusted -- every historical price is
    restated onto the CURRENT share scale -- while a snapshot's
    `underlying_price` is the as-traded quote. Those agree except across a
    split, where the adjusted close is on the far side of the ratio and the
    comparison would read as a 50% or 100% move. Any ticker with a split
    printed on either session is dropped and counted in `n_split`, rather
    than de-adjusted: at one or two names on the rare day it costs nothing,
    and a wrong breadth number is worse than a slightly smaller one.
    """
    if not pool:
        return {"error": "OI database not configured", "series": []}

    # `spot` is a fixed column name, not user input, so it needs no whitelist
    # pass -- but it is checked against the live column set anyway, because
    # "the column vanished upstream" should be a sentence rather than a
    # Postgres error surfacing through the panel as a blank canvas.
    cat = await _catalog(pool)
    if "spot" not in cat["live_metric_cols"]:
        return {"error": "equity_metrics carries no `spot` column",
                "series": []}
    spot = 'm."spot"'

    # Both sessions' split flags, and the two reference prices, per ticker.
    # `prev` is the previous session across the whole table rather than per
    # ticker: a ticker that did not print yesterday has no comparison to make
    # and should drop out, not silently reach further back for one.
    ref_cte = (
        "WITH prev AS ("
        "  SELECT max(trade_date) AS d FROM underlying_ohlc WHERE trade_date < $1"
        "), ref AS ("
        "  SELECT o.ticker, o.close AS prev_close, t.open AS open_px,"
        "         (COALESCE(o.splits, 0) <> 0 OR COALESCE(t.splits, 0) <> 0)"
        "           AS split"
        "  FROM underlying_ohlc o"
        "  JOIN prev p ON o.trade_date = p.d"
        "  LEFT JOIN underlying_ohlc t"
        "    ON t.ticker = o.ticker AND t.trade_date = $1"
        ")"
    )

    # A fund is has_earnings = false. NULL means "not in coverage at all",
    # which is a gap in the table rather than a statement that it is a fund,
    # so those stay in the stock count where they would have been anyway.
    is_fund = "cov.has_earnings IS FALSE"

    async with pool.acquire() as conn:
        d_, snap = await _resolve_slice(conn, date, snapshot)

        rows = await conn.fetch(
            f"{ref_cte} "
            f"SELECT m.snapshot,"
            f" count(*) FILTER (WHERE NOT r.split AND NOT {is_fund}"
            f"   AND {spot} IS NOT NULL AND r.prev_close IS NOT NULL)"
            f"     AS n_close,"
            f" count(*) FILTER (WHERE NOT r.split AND NOT {is_fund}"
            f"   AND {spot} > r.prev_close)"
            f"     AS n_up_close,"
            f" count(*) FILTER (WHERE NOT r.split AND NOT {is_fund}"
            f"   AND {spot} IS NOT NULL AND r.open_px IS NOT NULL)"
            f"     AS n_open,"
            f" count(*) FILTER (WHERE NOT r.split AND NOT {is_fund}"
            f"   AND {spot} > r.open_px)"
            f"     AS n_up_open,"
            f" count(*) FILTER (WHERE r.split) AS n_split,"
            f" count(*) FILTER (WHERE {is_fund}) AS n_fund "
            f"FROM {METRICS_TABLE} m "
            f"JOIN ref r ON r.ticker = m.ticker "
            f"LEFT JOIN {EARNINGS_COVERAGE_TABLE} cov ON cov.ticker = m.ticker "
            f"WHERE m.trade_date = $1 AND m.snapshot <= $2 "
            f"GROUP BY m.snapshot ORDER BY m.snapshot",
            d_, snap,
        )

    def _pct(n, d):
        return (100.0 * n / d) if d else None

    series = [{
        "snapshot":   r["snapshot"],
        "n":          int(r["n_close"] or 0),
        "pct_close":  _pct(r["n_up_close"] or 0, r["n_close"] or 0),
        "n_open":     int(r["n_open"] or 0),
        "pct_open":   _pct(r["n_up_open"] or 0, r["n_open"] or 0),
    } for r in rows]

    last = rows[-1] if rows else None
    return {
        "date": str(d_), "snapshot": snap,
        "series": series,
        # True while the daily bar for today has not been written, which is
        # the whole live session. The client draws the close-anchored line
        # regardless and labels the other one pending.
        "open_pending": not any(s["n_open"] for s in series),
        "n_split": int(last["n_split"] or 0) if last is not None else 0,
        "n_fund":  int(last["n_fund"] or 0) if last is not None else 0,
        "basis": {"open": "underlying_ohlc.open", "close": "prior session close",
                  "funds": "excluded from breadth"},
    }


@router.get("/universe-term-state")
async def universe_term_state(
    metric:               str  = Query(..., description="a term_ratio column"),
    date:                 str  = Query(None),
    snapshot:             str  = Query(None),
    sessions:             int  = Query(20, ge=2, le=250),
    exclude_extrapolated: bool = Query(True),
    pool=Depends(get_oi_pool),
):
    """What fraction of the universe is in contango, session by session.

    The regime filter's CHANGE carries more than its level. A single day at
    31% backwardated is a number; 6% to 31% over a week is the market
    repricing the front end, and only the series shows that.

    A term ratio is near/far, so BELOW 1 is contango — the far tenor is
    richer. Ties at exactly 1.0 are counted as backwardation rather than
    dropped: a flat curve is not contango, and silently discarding them would
    make the two shares not sum to the covered universe.

    History runs at the daily close and stops before the anchor date, with
    today taken at the selected snapshot — the same two-population split the
    rest of the page uses, for the same reason: an 11:25 reading and a set of
    closes are different populations and should not be averaged together.
    """
    if not pool:
        return {"error": "OI database not configured", "series": []}

    cat = await _catalog(pool)
    e   = _entry(cat, metric)
    if e["family"] != "term_ratio":
        raise HTTPException(
            400, f"{metric} is {e['family']}, not term_ratio — contango is a "
                 f"property of a tenor PAIR and this endpoint counts it.")

    val = _expr(e)
    if exclude_extrapolated:
        val = f"CASE WHEN {_extrap_expr(e)} THEN NULL ELSE {val} END"

    counts = (f" count({val})                                  AS n,"
              f" count(*) FILTER (WHERE {val} < 1.0)            AS n_contango")

    async with pool.acquire() as conn:
        d_, snap = await _resolve_slice(conn, date, snapshot)
        rows = await conn.fetch(
            f"SELECT m.trade_date, {counts} "
            f"{_from_clause(e['form'] != 'base')} "
            f"WHERE m.snapshot = $1 AND m.trade_date < $2 "
            f"GROUP BY m.trade_date ORDER BY m.trade_date DESC LIMIT $3",
            BASELINE_SNAPSHOT, d_, sessions,
        )
        trow = await conn.fetchrow(
            f"SELECT {counts} {_from_clause(e['form'] != 'base')} "
            f"WHERE m.snapshot = $1 AND m.trade_date = $2",
            snap, d_,
        )

    series = [{"date": str(r["trade_date"]), "n": int(r["n"] or 0),
               "n_contango": int(r["n_contango"] or 0)}
              for r in reversed(rows)]
    today = None
    if trow is not None and trow["n"]:
        today = {"date": str(d_), "n": int(trow["n"]),
                 "n_contango": int(trow["n_contango"] or 0), "today": True}

    return {
        "date": str(d_), "snapshot": snap, "metric": _meta(e),
        "sessions": sessions,
        "series": series,
        "today": today,
        "basis": {"snapshot": BASELINE_SNAPSHOT, "through": "prior session"},
        "exclude_extrapolated": bool(exclude_extrapolated),
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

    A z column is READ from equity_metrics_z, which now stores every
    snapshot's z against the ticker's 1545 daily series. Filters and ORDER BY
    run on the stored column, so `filter=..._z_63:gt:1.5`, the sort and the
    LIMIT all select from the universe rather than from a page of rows.
    """
    if not pool:
        return {"error": "OI database not configured", "rows": []}

    cat  = await _catalog(pool)
    cols = [c.strip() for c in columns.split(",") if c.strip()]
    if not cols:
        raise HTTPException(400, "columns is empty")
    entries = [_entry(cat, c) for c in cols]

    parsed = []
    for f in filter:
        parts = f.split(":", 2)
        if len(parts) < 2:
            raise HTTPException(400, f"Malformed filter: {f!r} (want col:op:value)")
        parsed.append((_entry(cat, parts[0]), parts[1],
                       parts[2] if len(parts) > 2 else ""))

    se   = _entry(cat, sort) if sort else None
    used = entries + [p[0] for p in parsed] + ([se] if se else [])

    sel = ["m.ticker"]
    for i, e in enumerate(entries):
        sel.append(f"{_expr(e)} AS v{i}")
        sel.append(f"{_extrap_expr(e)} AS e{i}")
    sel += ['m."{0}" AS {0}'.format(c) for c in CONTEXT_COLS]

    params: list = []
    where = [_filter_sql(fe, op, raw, params) for fe, op, raw in parsed]

    order = "m.ticker"
    if se:
        order = (f"{_expr(se)} {'DESC' if dir.lower() == 'desc' else 'ASC'} "
                 f"NULLS LAST, m.ticker")

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
            cn, v, ex = e["column_name"], r[f"v{i}"], r[f"e{i}"]
            flags[cn] = ex
            vals[cn]  = None if (exclude_extrapolated and ex) else v
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
        "z_source":  "stored" if _needs_z(used) else None,
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
    # spot_vol gained the tenor grid, so the columns are now
    # spotvol_{stat}_{tenor}d_{window}. The old flat names are kept as
    # fallbacks so the chip survives either side of the migration.
    "spotvol_beta":  ("spotvol_beta_30d_1m", "spotvol_beta_1m",
                      "spotvol_beta_30d_3m", "spotvol_beta_3m"),
    "spotvol_r2":    ("spotvol_r2_30d_1m", "spotvol_r2_1m",
                      "spotvol_r2_30d_3m", "spotvol_r2_3m"),
}


@router.get("/ticker-header")
async def ticker_header(
    ticker:   str = Query(...),
    date:     str = Query(None),
    snapshot: str = Query(None),
    pool=Depends(get_oi_pool),
):
    """The thin sticky header: spot, ATM IV, RV, and the state chips.

    `days_to_earnings` counts to the next earnings_calendar date at or after
    the trade date. The SESSION for that same date comes from the same row via
    the same LATERAL, not from a second lookup: a session attached to a
    different date than the count counts to would be worse than showing none,
    and would look entirely normal.

    bmo / amc matters at short horizons. A report before the open on day D
    moves D's session; after the close on D it moves D+1's. At one or two days
    out that is the difference between the event landing inside a structure's
    life or outside it, which is a different trade rather than a nuance.

    'unknown' is returned as-is and the client renders the day count with no
    session marker. A ticker with no calendar row gets nulls and renders
    absent, the same contract every other metric has here.

    `has_earnings` comes along from earnings_coverage because "no date" has
    two causes that look identical on screen: a fund that never reports, and a
    stock whose last confirmed date has passed — Yahoo publishes only the next
    one, so the calendar runs out rather than continuing. Both render absent,
    which is right, but the tooltip can say which, and only one of them is a
    reason to go and refresh the calendar.

    Note the count is CALENDAR days, not trading days. metrics_config is
    explicit about why: trading days would make every historical value depend
    on the exchange calendar and shift on recompute.

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
    sel += ["nxt.earnings_date", "nxt.earnings_session", "cov.has_earnings"]

    # The same LATERAL backfill_days_to_earnings.py uses, so the date this
    # resolves to is the date the stored count counted to. LATERAL rather than
    # a correlated scalar subquery so it stops at the first row via
    # ix_earnings_calendar_lookup instead of scanning every future date.
    nxt_join = (
        f"LEFT JOIN LATERAL ("
        f"  SELECT ec.earnings_date, ec.earnings_session "
        f"  FROM {EARNINGS_TABLE} ec "
        f"  WHERE ec.ticker = m.ticker AND ec.earnings_date >= m.trade_date "
        f"  ORDER BY ec.earnings_date LIMIT 1"
        f") nxt ON TRUE "
        f"LEFT JOIN {EARNINGS_COVERAGE_TABLE} cov ON cov.ticker = m.ticker"
    )

    async with pool.acquire() as conn:
        d, snap = await _resolve_slice(conn, date, snapshot)
        row = await conn.fetchrow(
            f"SELECT {', '.join(sel)} FROM {METRICS_TABLE} m {nxt_join} "
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
        "earnings_date":    _jsonable(row["earnings_date"]),
        # 'bmo' | 'amc' | 'unknown' | None. Passed through rather than mapped:
        # 'unknown' is a real state the calendar records, and collapsing it to
        # null would make "we do not know when in the day" indistinguishable
        # from "there is no scheduled report".
        "earnings_session": row["earnings_session"],
        # NULL when the ticker is not in earnings_coverage at all.
        "has_earnings":     row["has_earnings"],
        # The stored count and the count implied by the row the session came
        # from. They agree when the backfill has run; when they do not, the
        # pill would be pairing a day count with a different date's session.
        "earnings_days_calc": (
            None if row["earnings_date"] is None
            else (row["earnings_date"] - d).days),
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

    The two numbers on a card come from different places:

      z           READ from equity_metrics_z, which scores every snapshot
                  against the ticker's 1545 daily series over `z_window`
                  sessions, excluding the scored date from its own window.

      percentile  computed here, because equity_metrics carries no percentile
                  rank. Over the page's history window of daily closes,
                  ending at the prior session -- today is the thing being
                  ranked, and a value inside its own distribution is at the
                  100th percentile by construction.

    The two windows differ on purpose: a sigma means `z_window` sessions,
    while the percentile spans what the rails and the charts are showing.

    A metric whose own nodes are extrapolated today is excluded from the
    ranking under the toggle rather than shown with a marker: this list is
    sorted BY extremeness, and a fabricated node is exactly what manufactures
    a spurious extreme. It stays counted, so the client can show what was
    withheld.
    """
    if not pool:
        return {"error": "OI database not configured", "cards": []}
    if z_window not in (63, 252):
        raise HTTPException(400, f"z_window must be 63 or 252, got {z_window}")

    cat  = await _catalog(pool)
    form = f"z_{z_window}"
    fams = {f.strip() for f in families.split(",") if f.strip()} if families else None

    # Base columns the catalog gives a z variant at this window. Metrics
    # without one are excluded from z-scoring upstream on purpose -- a rolling
    # z of a trending price level is not a reading anyone wants -- so their
    # absence here is the same judgment, not a gap.
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

        ranked, n_thin = [], 0
        for i, (be, ze) in enumerate(pairs):
            bv, zv, ex = row[f"b{i}"], row[f"z{i}"], row[f"e{i}"]
            if bv is None:
                continue
            if zv is None:
                # A live value with no stored score: the window held fewer
                # than BASELINE_MIN_N observations. Counted, not ranked --
                # ranking it at z=0 would bury a real reading mid-strip as
                # though it had been measured.
                n_thin += 1
                continue
            ranked.append({"base": be, "z_entry": ze, "value": float(bv),
                           "z": float(zv), "extrap": bool(ex)})
        ranked.sort(key=lambda c: abs(c["z"]), reverse=True)

        shown = [c for c in ranked if not (exclude_extrapolated and c["extrap"])]
        withheld = len(ranked) - len(shown)
        shown = shown[:limit]

        # One scan for every percentile, not one scan each. The old shape
        # cost `limit` round trips over the same rows.
        if shown:
            start  = _window_start(d, window)
            params = [ticker, BASELINE_SNAPSHOT, d]
            pct_where = ("m.ticker = $1 AND m.snapshot = $2 "
                         "AND m.trade_date < $3")
            if start:
                params.append(start)
                pct_where += f" AND m.trade_date >= ${len(params)}"

            aggs = []
            for i, c in enumerate(shown):
                be = c["base"]
                params.append(c["value"])
                p = len(params)
                live = f'({_expr(be)} IS NOT NULL AND NOT {_extrap_expr(be)})'
                aggs += [
                    f"count(*) FILTER (WHERE {live}) AS d{i}",
                    f"count(*) FILTER (WHERE {live} AND {_expr(be)} <= ${p}) AS l{i}",
                ]
            prow = await conn.fetchrow(
                f"SELECT {', '.join(aggs)} {_from_clause(False)} WHERE {pct_where}",
                *params,
            )
            for i, c in enumerate(shown):
                den = int(prow[f"d{i}"] or 0)
                c["pct_n"] = den
                c["percentile"] = (int(prow[f"l{i}"] or 0) / den) if den else None
        else:
            start = _window_start(d, window)

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
        "percentile":  c.get("percentile"),
        "pct_n":       c.get("pct_n"),
        "extrap":      c["extrap"],
        "extrap_flags": c["base"]["extrap_flags"],
    } for c in shown]

    return {
        "ticker": ticker, "date": str(d), "snapshot": snap,
        "z_window": z_window, "window": window,
        "cards": cards,
        "n_ranked": len(ranked),
        "n_withheld_extrapolated": withheld,
        "n_unscored_thin_baseline": n_thin,
        "exclude_extrapolated": bool(exclude_extrapolated),
        "extrap_rate": row["extrap_rate_short"],
        "z_source": "stored",
        "baseline": {
            "snapshot": BASELINE_SNAPSHOT,
            "z_window": z_window,
            "min_n":    BASELINE_MIN_N,
        },
        "percentile_basis": {
            "snapshot": BASELINE_SNAPSHOT,
            "window":   window,
            "from":     _jsonable(start),
            "through":  "prior session",
        },
    }


# The default rail set, as SLOTS rather than a flat list of names.
#
# The flat version silently dropped any name the catalog did not have, which
# is how "rr_30d_25d" and "convexity_30d_25p_atm_25c" vanished from the panel
# without a word — the set just came back two rails shorter than it was
# written to be. Each slot now carries candidates and a label, resolution is
# reported in the payload, and a slot that resolves to nothing is named on
# screen instead of disappearing.
#
# Order is the reading order: both put wings, then the call wing, then level,
# term, curvature, and the three that price the trade rather than describe
# the surface.
RAILS_SLOTS = (
    ("25Δ put skew",     ("skew_30d_25p_atm",)),
    ("10Δ put skew",     ("skew_30d_10p_atm",)),
    ("25Δ call skew",    ("skew_30d_atm_25c", "skew_30d_25c_atm",
                          "skew_30d_atm_25c_", "callskew_30d_atm_25c")),
    ("ATM IV",           ("iv_30d_atm",)),
    ("term 7d/30d",      ("term_ratio_7d_30d", "term_ratio_7d_21d",
                          "term_ratio_14d_30d")),
    ("put convexity",    ("convex_30d_10p_25p_atm", "convexity_30d_10p_25p_atm",
                          "convex_30d_10p_25p", "convexity_30d_10p_25p_atm_")),
    ("zero-cost width",  ("zc_width_sigma_30d",)),
    # vrp_30d is the post-rename name; vrp_1m is the same column before the
    # RV-window migration lands. Both listed so the panel works either side
    # of it -- the candidate list is exactly what that mechanism is for.
    ("VRP",              ("vrp_30d", "vrp_1m", "vrp_21d")),
    ("spot-vol β",       ("spotvol_beta_30d_1m", "spotvol_beta_1m")),
)


def _resolve_rail_slots(cat):
    """[(label, column_or_None)] for the default rail set.

    Resolved against the CATALOG rather than the live column list, because a
    rail needs the catalog's units and description to render, not just a
    backing column.
    """
    out = []
    for label, cands in RAILS_SLOTS:
        hit = next((c for c in cands if c in cat["by_col"]), None)
        out.append((label, hit))
    return out


@router.get("/rails")
async def rails(
    ticker:               str  = Query(...),
    metrics:              str  = Query(None, description="CSV of base columns"),
    z_windows:            str  = Query(None, description="CSV, one per metric"),
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

    The distribution is built from DAILY CLOSES (see BASELINE_SNAPSHOT) over
    the page's history window, ending at the prior session -- not from the
    selected snapshot's own history. Scored the old way, an intraday bucket
    had one prior observation, so today's value was simultaneously the
    minimum and the maximum of its distribution and every rail read 100th
    percentile. Ending before today matters for the same reason on any
    snapshot: today cannot be one of the observations it is ranked against
    without being at the top of its own range by construction.

    Today's MARKER still comes from the selected snapshot. That is the whole
    point -- an 11:25 reading placed against the daily distribution.

    `z_windows` sets the window PER RAIL, parallel to `metrics`, falling back
    to `z_window` for any position it does not cover. A rail's window is a
    property of that rail because the disagreement between 63 and 252 is
    itself the reading — a metric stretched on both is at an extreme, one
    stretched on the short window alone is in a regime shift — and a single
    page-level window can only ever show one of those.

    The z beside the bar is READ from equity_metrics_z at the same
    (date, snapshot) as the marker — the same number /unusual and the scanner
    show. It is a label, not the bar's geometry: "where in the range" and
    "how many sigma" answer different questions.

    The DISTRIBUTION is still computed here, because equity_metrics carries a
    z but no percentile rank and no P5–P95.

    Extrapolated observations are dropped from the distribution AND from
    today's marker under the toggle. Leaving them in the history would
    define a normal range partly out of values the spline invented.
    """
    if not pool:
        return {"error": "OI database not configured", "rails": []}

    cat = await _catalog(pool)
    slots = None
    if metrics:
        cols = [c.strip() for c in metrics.split(",") if c.strip()]
    else:
        slots = _resolve_rail_slots(cat)
        cols = [c for _lbl, c in slots if c]
    if not cols:
        return {"error": "No usable rail metrics", "rails": [],
                "defaults": [{"slot": l, "column": c} for l, c in (slots or [])]}
    entries = [_entry(cat, c) for c in cols]

    # One window per rail, defaulting to the page's for anything unspecified.
    wins = []
    given = [w.strip() for w in z_windows.split(",")] if z_windows else []
    for i in range(len(entries)):
        raw = given[i] if i < len(given) else ""
        try:
            wins.append(int(raw))
        except ValueError:
            wins.append(z_window)
    bad = [w for w in wins if w not in (63, 252)]
    if bad:
        raise HTTPException(400, f"z_windows must be 63 or 252, got {bad}")

    zcols = [_z_column(cat, e["column_name"], w) for e, w in zip(entries, wins)]

    out = []
    async with pool.acquire() as conn:
        d, snap = await _resolve_slice(conn, date, snapshot)
        start   = _window_start(d, window)

        # Today's values and their stored z, at the SELECTED snapshot.
        sel = []
        for i, e in enumerate(entries):
            sel += [f"{_expr(e)} AS v{i}", f"{_extrap_expr(e)} AS x{i}"]
            if zcols[i] is not None:
                sel.append(f"{_expr(zcols[i])} AS z{i}")
        cur = await conn.fetchrow(
            f"SELECT {', '.join(sel)} "
            f"{_from_clause(_needs_z(entries) or any(z is not None for z in zcols))} "
            f"WHERE m.ticker = $1 AND m.trade_date = $2 AND m.snapshot = $3",
            ticker, d, snap,
        )

        # One scan for every distribution. percentile_cont is an ordered-set
        # aggregate, so all five quantiles for all rails come out of a single
        # pass over the window.
        params = [ticker, BASELINE_SNAPSHOT, d]
        where  = "m.ticker = $1 AND m.snapshot = $2 AND m.trade_date < $3"
        if start:
            params.append(start)
            where += f" AND m.trade_date >= ${len(params)}"

        qaggs = []
        for i, e in enumerate(entries):
            v = _expr(e)
            if exclude_extrapolated:
                # NULL the fabricated observations rather than filtering rows:
                # one row carries every rail, and dropping it for one metric's
                # fabricated node would silently shorten every other rail's
                # window too.
                v = f"(CASE WHEN {_extrap_expr(e)} THEN NULL ELSE {v} END)"
            for q, nm in ((0.05, "p5"), (0.25, "p25"), (0.50, "p50"),
                          (0.75, "p75"), (0.95, "p95")):
                qaggs.append(
                    f"percentile_cont({q}) WITHIN GROUP (ORDER BY {v}) AS {nm}_{i}")
            qaggs.append(f"count({v}) AS n{i}")
        dist = await conn.fetchrow(
            f"SELECT {', '.join(qaggs)} "
            f"{_from_clause(False)} WHERE {where}",
            *params,
        )

        # Today's percentile within that same distribution.
        pparams = list(params)
        paggs, want = [], []
        for i, e in enumerate(entries):
            v  = None if cur is None else cur[f"v{i}"]
            ex = False if cur is None else bool(cur[f"x{i}"])
            if v is None or (exclude_extrapolated and ex):
                continue
            pparams.append(float(v))
            p = len(pparams)
            col = _expr(e)
            live = f"({col} IS NOT NULL"
            if exclude_extrapolated:
                live += f" AND NOT {_extrap_expr(e)}"
            live += ")"
            paggs += [f"count(*) FILTER (WHERE {live}) AS d{i}",
                      f"count(*) FILTER (WHERE {live} AND {col} <= ${p}) AS l{i}"]
            want.append(i)
        prow = None
        if paggs:
            prow = await conn.fetchrow(
                f"SELECT {', '.join(paggs)} "
                f"{_from_clause(False)} WHERE {where}",
                *pparams,
            )

    for i, e in enumerate(entries):
        v  = None if cur is None else cur[f"v{i}"]
        ex = False if cur is None else bool(cur[f"x{i}"])
        shown = None if (v is None or (exclude_extrapolated and ex)) else float(v)

        pct = None
        if prow is not None and i in want:
            den = int(prow[f"d{i}"] or 0)
            pct = (int(prow[f"l{i}"] or 0) / den) if den else None

        zv = None
        if zcols[i] is not None and cur is not None and cur[f"z{i}"] is not None:
            zv = float(cur[f"z{i}"])
        out.append({
            **_meta(e),
            "value":      shown,
            "raw_value":  None if v is None else float(v),
            "extrap":     ex,
            "z":          zv,
            "z_window":   wins[i],
            "z_column":   zcols[i]["column_name"] if zcols[i] is not None else None,
            "percentile": pct,
            "p5":  _f(dist, f"p5_{i}"),  "p25": _f(dist, f"p25_{i}"),
            "p50": _f(dist, f"p50_{i}"), "p75": _f(dist, f"p75_{i}"),
            "p95": _f(dist, f"p95_{i}"),
            "n":   int(dist[f"n{i}"]) if dist and dist[f"n{i}"] else 0,
        })

    return {
        "ticker": ticker, "date": str(d), "snapshot": snap,
        "window": window, "z_window": z_window,
        "rails": out,
        "exclude_extrapolated": bool(exclude_extrapolated),
        "z_source": "stored",
        # Which default slot each rail came from, and which found nothing.
        # Returned only when the caller took the defaults; an explicit metric
        # list is its own answer.
        "defaults": ([{"slot": l, "column": c} for l, c in slots]
                     if slots is not None else None),
        "baseline": {
            "snapshot": BASELINE_SNAPSHOT, "z_window": z_window,
            "min_n": BASELINE_MIN_N,
        },
        "distribution_basis": {
            "snapshot": BASELINE_SNAPSHOT, "window": window,
            "from": _jsonable(start), "through": "prior session",
        },
    }


def _f(row, key):
    """float() a percentile_cont result, tolerating a null row or null value."""
    if row is None or row[key] is None:
        return None
    return float(row[key])

# The bucket the daily VIEW plots by default. Deliberately the same value as
# BASELINE_SNAPSHOT and deliberately a separate name: the caller may plot a
# different bucket's closes, and the baseline scoring them stays at 1545
# regardless. Aliased rather than re-typed so the two cannot drift.
# The bucket the daily VIEW plots by default. Deliberately the same value as
# BASELINE_SNAPSHOT and deliberately a separate name: the caller may plot a
# different bucket's closes, and the z beside each point is scored against
# 1545 regardless — that is now a property of the stored column rather than
# something this module arranges.
DAILY_SNAPSHOT = BASELINE_SNAPSHOT
SERIES_MAX     = 4


def _quantile(sorted_vals, p):
    """Linear-interpolated quantile of an already-sorted list."""
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    pos  = p * (len(sorted_vals) - 1)
    lo_i = int(pos)
    hi_i = min(lo_i + 1, len(sorted_vals) - 1)
    return sorted_vals[lo_i] + (sorted_vals[hi_i] - sorted_vals[lo_i]) * (pos - lo_i)


def _rolling_pct_envelope(vals, win, lo_q, hi_q, admit=None):
    """Trailing percentile band, one (lo, hi) per point, aligned to `vals`.

    Trailing rather than centred: a centred window at index i uses values
    from after i, so the band at any historical point would be drawn partly
    out of the future. On a chart whose whole purpose is "was this unusual
    AT THE TIME", that is the wrong band.

    The band is computed from history strictly BEFORE each point, then the
    point is admitted — so a marker is never inside the distribution it is
    being judged against.

    `admit` is a parallel list of booleans for values that may be SCORED but
    must not be COUNTED — the live partial point at the end of the series.
    A 12:15 reading is a sample of an unfinished session, and letting it into
    the band would put a not-yet-real observation into the definition of
    normal. It still gets a band, because the band was already built from
    prior history alone.

    Nulls and (when excluded) extrapolated observations arrive as None and
    never enter `hist`, so the band describes observed history only.
    """
    out, hist = [], []
    floor = max(8, win // 4)
    for i, v in enumerate(vals):
        if len(hist) >= floor:
            w = sorted(hist[-win:])
            out.append((_quantile(w, lo_q), _quantile(w, hi_q)))
        else:
            out.append((None, None))
        if v is not None and (admit is None or admit[i]):
            hist.append(v)
    return out


@router.get("/series")
async def series(
    ticker:               str   = Query(...),
    metrics:              str   = Query(..., description="CSV of columns, max 4"),
    mode:                 str   = Query("daily", description="daily|intraday|candle"),
    spot:                 bool  = Query(True, description="the background spot reference"),
    snapshot:             str   = Query(None, description="daily mode bucket"),
    date:                 str   = Query(None, description="anchor date; default latest"),
    live_snapshot:        str   = Query(None, description="the page's selected bucket"),
    include_today:        bool  = Query(True),
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

    THE LIVE POINT
    --------------
    The chart shows the world as of (`date`, `live_snapshot`) — the same
    anchor the header, the cards and the rails use. That is what puts today
    on the line: in daily mode the 1545 row does not exist until the close,
    so a series filtered to that bucket ended at the prior session while
    every other panel on the page already showed today.

    So when the anchor date has no settled close yet, today's reading at the
    selected bucket is appended as a final point flagged `partial`. It
    advances with the session because it is read at `live_snapshot` rather
    than pinned to a bucket.

    `partial` is not decoration. A 12:15 reading is a sample of an unfinished
    session, not a close, and the two must not be readable as the same kind
    of observation — the client draws it as an open marker on a dashed
    segment. The flag is set the same way in every mode: the point belongs to
    the anchor date and is not that date's settled close.

    A partial point is kept out of the rolling envelope's history, so the band
    it sits against is built from settled sessions only. Its own band still
    renders, because the envelope was already trailing — computed from history
    strictly before each point. Its z is the stored one for that row, which is
    scored against 1545 dailies excluding the scored date, so the live point
    cannot enter its own yardstick either.

    Once the close lands, the settled 1545 row IS the point and nothing is
    appended — the partial reading is replaced by the real one rather than
    left sitting beside it.

    A faint spot reference rides behind the metrics, off
    equity_atm.underlying_price for the plotted (date, snapshot) — the daily
    view takes each session's 1545 value, the intraday view each bucket's. It
    is chrome: its own hidden scale, no axis labels, drawn behind the band and
    the lines, absent from the legend and from the metric picker, and
    switchable off. Its own scale because spot runs in the hundreds while
    these metrics run 0.05 to 1.1, and one shared axis would flatten every
    metric to a line.

    z is READ from equity_metrics_z per row. Every snapshot's stored z is
    scored against the ticker's 1545 daily series, so the number means the
    same thing on the daily and intraday views without this endpoint
    rescoring anything.

    One consequence worth expecting on the intraday view: an intraday reading
    measured against 1545 closes carries the mean bucket-versus-close drift of
    the session. That drift is uniform within a bucket, so it shifts the
    distribution rather than reordering tickers within it.
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
    appended_live = False
    spot_points = []
    async with pool.acquire() as conn:
        as_of, live_snap = await _resolve_slice(conn, date, live_snapshot)
        daily_snap = snapshot or DAILY_SNAPSHOT
        start      = _window_start(as_of, window)

        def slice_predicate(alias="m"):
            """The (date, snapshot) window every series in this call plots.

            Built once and used by both the metric queries and the spot query,
            so the reference line cannot end up covering a different span than
            the thing it sits behind.
            """
            where = [f"{alias}.ticker = $1"]
            args  = [ticker]
            args.append(as_of)
            p_asof = len(args)
            where.append(f"{alias}.trade_date <= ${p_asof}")
            if mode == "daily":
                args.append(daily_snap)
                where.append(f"{alias}.snapshot = ${len(args)}")
            else:
                args.append(live_snap)
                where.append(f"({alias}.trade_date < ${p_asof} "
                             f"OR {alias}.snapshot <= ${len(args)})")
            if start:
                args.append(start)
                where.append(f"{alias}.trade_date >= ${len(args)}")
            return where, args

        for e in entries:
            zcol = _z_column(cat, e["column_name"], z_window)

            sel = ["m.trade_date", "m.snapshot", f"{_expr(e)} AS v",
                   f"{_extrap_expr(e)} AS ex"]
            if zcol is not None:
                sel.append(f"{_expr(zcol)} AS zs")
            needs_z = (e["form"] != "base") or zcol is not None

            # Upper-bounded at the anchor date, and on the intraday views
            # truncated at the selected bucket, so the chart shows the world as
            # of what the page says rather than running to the newest capture.
            where, args = slice_predicate("m")

            rows = await conn.fetch(
                f"SELECT {', '.join(sel)} {_from_clause(needs_z)} "
                f"WHERE {' AND '.join(where)} "
                f"ORDER BY m.trade_date, m.snapshot",
                *args,
            )

            # In daily mode the settled close may simply not exist yet.
            live_row = None
            settled_today = any(r["trade_date"] == as_of for r in rows)
            if (include_today and mode == "daily" and not settled_today
                    and live_snap != daily_snap):
                lsel = [f"{_expr(e)} AS v", f"{_extrap_expr(e)} AS ex"]
                if zcol is not None:
                    lsel.append(f"{_expr(zcol)} AS zs")
                live_row = await conn.fetchrow(
                    f"SELECT m.trade_date, m.snapshot, {', '.join(lsel)} "
                    f"{_from_clause(needs_z)} "
                    f"WHERE m.ticker = $1 AND m.trade_date = $2 AND m.snapshot = $3",
                    ticker, as_of, live_snap,
                )

            def is_partial(td, snap_):
                """A reading of the anchor session that is not its close.

                Never true on the intraday view. The mark exists to separate a
                settled close from a mid-session sample, and that is a
                distinction the daily line makes and the intraday line cannot:
                there every point is a snapshot and none is a close, so
                flagging today's would mark a whole run for a property all of
                them share. That is an artifact rendered as a signal — and
                once intraday has a few weeks of history it would be the
                entire chart, uniformly.

                Candle mode keeps it, because a candle bar is a whole session
                and today's is an unfinished one, which is the same
                distinction the daily line draws.
                """
                if mode == "intraday":
                    return False
                return td == as_of and snap_ != BASELINE_SNAPSHOT

            if mode == "candle":
                by_day, last_snap, last_z = {}, {}, {}
                for r in rows:
                    if r["v"] is None or (exclude_extrapolated and r["ex"]):
                        continue
                    by_day.setdefault(r["trade_date"], []).append(float(r["v"]))
                    last_snap[r["trade_date"]] = r["snapshot"]   # rows are ordered
                    # The bar's z is the stored z of its CLOSING bucket, not a
                    # re-score of the close: the bar is a summary of readings
                    # that were each already scored.
                    last_z[r["trade_date"]] = (
                        None if zcol is None or r["zs"] is None else float(r["zs"]))
                pts = []
                for dd in sorted(by_day):
                    vs = by_day[dd]                   # already snapshot-ordered
                    pts.append({"t": dd.isoformat(), "o": vs[0], "h": max(vs),
                                "l": min(vs), "c": vs[-1], "n": len(vs),
                                "z": last_z.get(dd),
                                # A bar whose last bucket is not the close is a
                                # bar of a session still being written.
                                "partial": is_partial(dd, last_snap.get(dd)),
                                # NOT called `snapshot`: a candle bar is a
                                # whole day, and the client labels the x axis
                                # with the bucket whenever a point carries
                                # one. That would stamp every daily bar with
                                # a time it does not represent.
                                "last_bucket": last_snap.get(dd)})
                closes = [p["c"] for p in pts]
            else:
                pts = []
                for r in rows:
                    v     = None if r["v"] is None else float(r["v"])
                    ex    = bool(r["ex"])
                    shown = None if (v is None or (exclude_extrapolated and ex)) else v
                    zv = (None if zcol is None or r["zs"] is None
                          else float(r["zs"]))
                    p = {"t": r["trade_date"].isoformat(), "v": shown, "extrap": ex,
                         "z": zv,
                         "partial": is_partial(r["trade_date"], r["snapshot"])}
                    if mode == "intraday" or p["partial"]:
                        p["snapshot"] = r["snapshot"]
                    pts.append(p)

                if live_row is not None and live_row["v"] is not None:
                    ex    = bool(live_row["ex"])
                    shown = None if (exclude_extrapolated and ex) else float(live_row["v"])
                    # snapshot comes from live_snap, not from the row: the
                    # query filtered on it, so it is the same value by
                    # construction, and taking it from the parameter means the
                    # label cannot disagree with what was asked for.
                    pts.append({"t": as_of.isoformat(), "v": shown, "extrap": ex,
                                "z": (None if zcol is None or live_row["zs"] is None
                                      else float(live_row["zs"])),
                                "partial": True, "snapshot": live_snap})
                    appended_live = True

                closes = [p["v"] for p in pts]

            if envelope and pts:
                # A partial point is scored BY the band but never joins it.
                admit = [not p.get("partial") for p in pts]
                band  = _rolling_pct_envelope(closes, env_window, env_lo, env_hi,
                                              admit=admit)
                for p, (lo, hi) in zip(pts, band):
                    p["env_lo"], p["env_hi"] = lo, hi

            out.append({
                **_meta(e),
                "points":     pts,
                "n_points":   len(pts),
                "first_date": pts[0]["t"] if pts else None,
                "last_date":  pts[-1]["t"] if pts else None,
                "n_partial":  sum(1 for p in pts if p.get("partial")),
                "baseline":   {"snapshot": BASELINE_SNAPSHOT,
                               "z_window": z_window,
                               "min_n": BASELINE_MIN_N,
                               "as_of": as_of.isoformat()},
                "z_column": zcol["column_name"] if zcol is not None else None,
            })

        if spot:
            # underlying_price is a property of the (ticker, trade_date,
            # snapshot) row and repeats across every dte in equity_atm, so the
            # grouping collapses that repetition rather than choosing among
            # differing values. max() over identical values is just "the one".
            swhere, sargs = slice_predicate("a")
            srows = await conn.fetch(
                f"SELECT a.trade_date, a.snapshot, max(a.underlying_price) AS v "
                f"FROM {ATM_TABLE} a WHERE {' AND '.join(swhere)} "
                f"GROUP BY a.trade_date, a.snapshot "
                f"ORDER BY a.trade_date, a.snapshot",
                *sargs,
            )
            if mode == "candle":
                # One value per session, the closing bucket's — the bars are
                # daily, so the reference behind them has to be too.
                by_day = {}
                for r in srows:
                    if r["v"] is not None:
                        by_day[r["trade_date"]] = float(r["v"])
                spot_points = [{"t": dd.isoformat(), "v": v}
                               for dd, v in sorted(by_day.items())]
            else:
                for r in srows:
                    if r["v"] is None:
                        continue
                    p = {"t": r["trade_date"].isoformat(), "v": float(r["v"])}
                    # Labelled exactly as the metric points are, so the client
                    # places it in the same slot rather than on a parallel axis
                    # of its own.
                    if mode == "intraday" or (
                            r["trade_date"] == as_of
                            and r["snapshot"] != BASELINE_SNAPSHOT):
                        p["snapshot"] = r["snapshot"]
                    spot_points.append(p)

    return {
        "ticker": ticker, "mode": mode, "window": window, "z_window": z_window,
        "snapshot": daily_snap if mode == "daily" else None,
        "live_snapshot": live_snap,
        "latest_snapshot": live_snap,
        "as_of": as_of.isoformat(),
        "live_point": {
            "appended": appended_live,
            "snapshot": live_snap,
            "date":     as_of.isoformat(),
            # True once the close has landed: the settled row is then the
            # point, and nothing is appended.
            "settled":  live_snap == BASELINE_SNAPSHOT,
        },
        "z_source": "stored",
        # Chrome, not a series: no catalog entry, no legend row, its own hidden
        # scale. The client draws it behind everything and reads it out in the
        # tooltip, which is the only place its absolute level is legible.
        "spot": {"on": bool(spot), "points": spot_points,
                 "source": f"{ATM_TABLE}.underlying_price"},
        "envelope": ({"on": True, "window": env_window, "lo": env_lo, "hi": env_hi}
                     if envelope else {"on": False}),
        "exclude_extrapolated": bool(exclude_extrapolated),
        "series": out,
    }
