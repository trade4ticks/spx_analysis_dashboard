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

equity_metrics_z is, with one exception, NO LONGER READ. Its values are
scored per (ticker, trade_date, snapshot) — each snapshot against its own
history — and the intraday grid only begins 2026-08-24. At the 1200 bucket
that made every stored z null or meaningless: the scanner's z column came
back all em-dashes, the card strip reported nothing scoreable, and every
rail read 100th percentile because a value ranked against one same-bucket
observation is its own maximum.

So a z is now DERIVED at query time against the ticker's daily-close
baseline (see BASELINE_SNAPSHOT), which is what makes +1.8σ mean one thing
whichever snapshot is on screen. A z column name still selects the score and
still names the window — the catalog row is the whitelist entry and the
window label — but the stored value behind it is not what comes back.

The exception is /series, which fetches the stored z as `z_stored` ONLY when
the plotted bucket is the baseline bucket, so a divergence between the two
estimators is visible rather than hidden. /rails and /series reject a z
column outright: neither can derive one without a rolling window, and
serving the stored value there would reintroduce the bug quietly.

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
# Every z and every percentile on this page is measured against the DAILY
# CLOSE series, whatever snapshot is on screen. This is the page's single
# scoring rule and it exists for two reasons.
#
# The first is arithmetic. Scoring a snapshot against its own history means an
# 11:25 reading is ranked against prior 11:25 readings, and the intraday grid
# only starts on 2026-08-24. At the 1200 bucket that distribution is one
# observation deep, so every value is simultaneously the minimum and the
# maximum of it: percentile 100, z undefined, and a page that looks broken
# because it IS measuring nothing.
#
# The second survives the intraday history growing, which is why the rule is
# permanent rather than a stopgap. Five-minute observations inside a session
# are heavily autocorrelated, so a hundred bars carry nowhere near a hundred
# observations' worth of information — the effective sample size is closer to
# the number of SESSIONS than the number of bars. A z computed off the bar
# count would keep looking confident while being backed by a fraction of the
# evidence it claims.
#
# Scoring everything against the same daily distribution also keeps +1.8σ
# meaning one thing across the page. A user switching snapshots is asking
# "where is this now", not "re-scale the axis under me".

BASELINE_SNAPSHOT = "1545"

# Strictly fewer than this many observations and a mean and a standard
# deviation are noise wearing a measurement's clothes. Returning null beats
# returning a confident number off n=3 — that is the failure this whole
# module exists to stop, and it would recur silently at small n.
BASELINE_MIN_N = 20


def _session_span_days(z_window: int) -> int:
    """Calendar days that comfortably contain `z_window` trading sessions.

    Sessions run about 252 a year, so 2x + 30 clears the requirement with
    room for holidays and a stretch of missing captures. It bounds the scan;
    row_number() does the actual selecting, so an over-generous bound costs
    a little IO and cannot change the answer.
    """
    return z_window * 2 + 30


def _base_entry(cat: dict, entry: dict) -> dict:
    """The base-form entry a z entry derives from (or the entry itself).

    The baseline is always computed from the BASE column's daily closes.
    equity_metrics_z is never read for it: those values are stored per
    (ticker, trade_date, snapshot) and carry the same-snapshot scoring this
    replaces, so building a daily baseline out of them would re-import the
    bug it is meant to remove.
    """
    if entry["form"] == "base":
        return entry
    base = cat["by_col"].get(entry["base_column"] or "")
    if base is None:
        raise HTTPException(
            400, f"{entry['column_name']} has no base column in the catalog, "
                 f"so no daily baseline can be built for it")
    return base


def _baseline_cte(cat, z_entries, as_of, z_window, params, ticker=None):
    """SQL for a per-ticker daily-close baseline. Returns (cte_sql, zmap).

    `zmap` maps a z column name to its index, so callers build the score with
    _z_expr(). The CTE yields one row per ticker with mu/sd/n per metric.

    STRICTLY BEFORE `as_of`. If today's own close is inside the window it is
    being scored against, the z is contaminated — the value pulls the mean
    toward itself and inflates the sd, so every reading is shaded toward
    ordinary. At n=252 that is a rounding error; at n=20 it is most of a
    sigma. It also has to be all-or-nothing rather than "exclude it when it
    exists", because a window that silently changes definition at 15:45 every
    day is worse than either rule consistently applied.

    The window is a fixed number of SESSIONS, shared by every metric in the
    call, not each metric's last N non-null observations. /unusual ranks
    metrics against each other by |z|, and that comparison is only meaningful
    if the windows cover the same span: a sparse metric taking its last 63
    observations would reach back years and be scored against a different
    market than the metric beside it. A metric thin inside the window gets a
    smaller n, which is reported, rather than a quietly longer window.
    """
    bases, zmap = [], {}
    for e in z_entries:
        if e["column_name"] in zmap:
            continue
        zmap[e["column_name"]] = len(bases)
        bases.append(_base_entry(cat, e))

    params.append(BASELINE_SNAPSHOT);                     p_snap = len(params)
    params.append(as_of);                                 p_asof = len(params)
    params.append(date_type.fromordinal(
        as_of.toordinal() - _session_span_days(z_window))); p_min = len(params)
    params.append(z_window);                              p_n = len(params)

    tk = ""
    if ticker is not None:
        params.append(ticker)
        tk = f"AND ticker = ${len(params)} "

    aggs = []
    for i, be in enumerate(bases):
        v  = 'b."{}"'.format(be["column_name"])
        # Fabricated observations never enter the baseline. A "normal range"
        # partly built out of values the spline invented is not a range the
        # market ever traded.
        ok = "NOT " + _extrap_expr(be, "b")
        aggs += [
            f"avg({v}) FILTER (WHERE {ok}) AS mu{i}",
            f"stddev_samp({v}) FILTER (WHERE {ok}) AS sd{i}",
            f"count({v}) FILTER (WHERE {ok}) AS n{i}",
        ]

    cte = (
        f"bl_days AS ("
        f" SELECT ticker, trade_date,"
        f" row_number() OVER (PARTITION BY ticker ORDER BY trade_date DESC) AS rn"
        f" FROM {METRICS_TABLE}"
        f" WHERE snapshot = ${p_snap} AND trade_date < ${p_asof}"
        f" AND trade_date >= ${p_min} {tk}"
        f"), "
        f"bl AS ("
        f" SELECT b.ticker,"
        # The real span the baseline covers, reported to the client. This is
        # not decoration: bl_last is what proves the window stops before
        # today. If it ever equals the date on screen, the exclusion broke.
        f" min(b.trade_date) AS bl_first, max(b.trade_date) AS bl_last,"
        f" count(*) AS bl_sessions,"
        f" {', '.join(aggs)}"
        f" FROM {METRICS_TABLE} b"
        f" JOIN bl_days d ON d.ticker = b.ticker AND d.trade_date = b.trade_date"
        f" WHERE b.snapshot = ${p_snap} AND d.rn <= ${p_n}"
        f" GROUP BY b.ticker"
        f")"
    )
    return cte, zmap


def _z_expr(cat, entry, zmap, alias="m", bl="bl"):
    """A metric's SQL value, with z forms scored against the daily baseline.

    A base metric is itself. A z metric becomes (today - mu) / sigma, which
    is why the whole page can stop reading equity_metrics_z: the score is
    derived at query time from the base column and the baseline, so it means
    the same thing at 09:45 as at the close.

    NULLIF guards a zero sigma (a metric constant across the window) and the
    n floor guards a window too thin to describe anything. Both yield NULL,
    which the client already renders as absent.
    """
    if entry["form"] == "base":
        return '{}."{}"'.format(alias, entry["column_name"])
    i  = zmap[entry["column_name"]]
    be = _base_entry(cat, entry)
    return (f'(CASE WHEN {bl}.n{i} >= {BASELINE_MIN_N} THEN '
            f'({alias}."{be["column_name"]}" - {bl}.mu{i}) '
            f'/ NULLIF({bl}.sd{i}, 0) END)')


def _z_window_of(z_entries) -> int:
    """The baseline length implied by a set of z columns.

    The universe endpoints take column NAMES, not a z_window parameter — the
    client encodes the window in the name (`..._z_63` / `..._z_252`) because
    that is what the catalog calls the column. So the window is read back off
    the form rather than passed alongside, which also means the two can never
    disagree about which one was asked for.

    Mixed windows in one request would need two baselines, and no caller does
    it — the page has a single Z-window control. Rejected explicitly rather
    than silently scored against whichever one was seen first.
    """
    wins = {e["form"] for e in z_entries if e["form"] != "base"}
    if not wins:
        return 63
    if len(wins) > 1:
        raise HTTPException(
            400, f"Mixed z windows in one request ({', '.join(sorted(wins))}). "
                 f"Every z column in a request must share a window.")
    form = wins.pop()
    try:
        return int(form.split("_")[-1])
    except ValueError:
        raise HTTPException(400, f"Unrecognised z form: {form!r}")


def _reject_z_form(entries, panel: str, instead: str):
    """Refuse a stored z column where only a derived score is meaningful.

    equity_metrics_z is scored per (ticker, trade_date, snapshot), which is
    the same-snapshot scoring this module exists to replace. The panels that
    can derive a z do; the ones that would have to read the stored column
    say so instead of quietly serving a number that means something
    different at 11:25 than at the close.

    This is a closed door rather than a silent fallback because the failure
    it prevents is invisible: a z column plotted from the stored table looks
    exactly like a correct one.
    """
    bad = [e["column_name"] for e in entries if e["form"] != "base"]
    if bad:
        raise HTTPException(
            400,
            f"{panel} takes base metric columns, not stored z columns "
            f"({', '.join(bad)}). Those are scored per (date, snapshot), so on "
            f"an intraday bucket they measure against that bucket's own "
            f"history. {instead}")


def _baseline_join(bl="bl", alias="m"):
    """LEFT JOIN, not JOIN: a ticker with no usable history still has today's
    values, and dropping its row entirely would make a thin name look absent
    from the universe rather than unscored."""
    return f"LEFT JOIN {bl} ON {bl}.ticker = {alias}.ticker"


async def _baseline_for(conn, cat, entries, ticker, as_of, z_window):
    """{column_name: {mu, sigma, n}} for one ticker, one query.

    Used by the panels that need the baseline as DATA — to show it in a
    footer, to derive a z in Python, or to build a percentile — rather than
    as a SQL expression. Keyed by the entry's own column name so a caller
    can look up either a base or a z column and get the same answer, since
    they share a baseline by construction.
    """
    if not entries:
        return {}
    params: list = []
    cte, zmap = _baseline_cte(cat, entries, as_of, z_window, params, ticker=ticker)
    sel = ["bl_first", "bl_last", "bl_sessions"]
    for col, i in zmap.items():
        sel += [f"mu{i}", f"sd{i}", f"n{i}"]
    row = await conn.fetchrow(f"WITH {cte} SELECT {', '.join(sel)} FROM bl", *params)

    out = {}
    for col, i in zmap.items():
        if row is None:
            out[col] = {"mu": None, "sigma": None, "n": 0,
                        "first": None, "last": None, "sessions": 0}
            continue
        n  = int(row[f"n{i}"] or 0)
        mu = row[f"mu{i}"]
        sd = row[f"sd{i}"]
        thin = n < BASELINE_MIN_N
        out[col] = {
            # Below the floor the numbers are withheld, not shown small: a mu
            # off n=3 renders identically to one off n=250 and there is
            # nothing on screen to tell them apart. `n` still comes back so
            # the client can say why the score is missing.
            "mu":       None if mu is None or thin else float(mu),
            "sigma":    None if sd is None or thin else float(sd),
            "n":        n,
            "first":    _jsonable(row["bl_first"]),
            "last":     _jsonable(row["bl_last"]),
            "sessions": int(row["bl_sessions"] or 0),
        }
    return out


def _z_from(baseline: dict, col: str, v):
    """(v - mu) / sigma, or None when the value or the baseline is unusable."""
    b = baseline.get(col)
    if v is None or not b or b["mu"] is None or not b["sigma"]:
        return None
    return (v - b["mu"]) / b["sigma"]


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

    A z-form axis is DERIVED here against each ticker's daily-close baseline
    rather than read from equity_metrics_z — see BASELINE_SNAPSHOT. The
    stored column is scored per (date, snapshot), so on an intraday bucket
    the default preset (skew z on x) plotted a z against that bucket's own
    one-day history. Deriving it keeps the scatter agreeing with the rails
    and the scanner about what +1.8σ means.
    """
    if not pool:
        return {"error": "OI database not configured", "points": []}

    cat    = await _catalog(pool)
    ex, ey = _entry(cat, x), _entry(cat, y)
    esize  = _entry(cat, size)  if size  else None
    ecolor = _entry(cat, color) if color else None
    used   = [e for e in (ex, ey, esize, ecolor) if e]
    zs     = [e for e in used if e["form"] != "base"]

    async with pool.acquire() as conn:
        d, snap = await _resolve_slice(conn, date, snapshot)

        params: list = []
        cte, zmap = ("", {})
        if zs:
            cte, zmap = _baseline_cte(cat, zs, d, _z_window_of(zs), params)

        def col(e):
            return _z_expr(cat, e, zmap)

        sel = [
            "m.ticker",
            f"{col(ex)} AS x",
            f"{col(ey)} AS y",
            f"{_extrap_expr(ex)} AS x_extrap",
            f"{_extrap_expr(ey)} AS y_extrap",
        ]
        sel += ['m."{0}" AS {0}'.format(c) for c in CONTEXT_COLS]
        if esize:
            sel.append(f"{col(esize)} AS size_v")
        if ecolor:
            sel += [f"{col(ecolor)} AS color_v",
                    f"{_extrap_expr(ecolor)} AS color_extrap"]

        params += [d, snap]
        pd_, ps_ = len(params) - 1, len(params)
        rows = await conn.fetch(
            (f"WITH {cte} " if cte else "")
            + f"SELECT {', '.join(sel)} "
            + f"{_from_clause(False)} "
            + (f"{_baseline_join()} " if zs else "")
            + f"WHERE m.trade_date = ${pd_} AND m.snapshot = ${ps_} "
            + f"ORDER BY m.ticker",
            *params,
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
        "z_source":  "daily_baseline" if zs else None,
        "baseline":  ({"snapshot": BASELINE_SNAPSHOT,
                       "z_window": _z_window_of(zs),
                       "min_n": BASELINE_MIN_N,
                       "through": "prior session"} if zs else None),
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

    Two populations, deliberately:

      history  prior sessions at the daily close (BASELINE_SNAPSHOT). When
               the metric is a z column the z is rolled forward per date with
               a window function — each date scored against ITS OWN prior
               `z_window` sessions, never against the whole span. Scoring the
               history with one baseline taken at the end would let the last
               year of prices set the threshold that judges the start of it.
      today    the SELECTED snapshot, scored against the daily baseline
               through the prior session. That is what makes an 11:25 count
               comparable to the daily counts beside it.

    This split is why the history stops before today rather than including
    it. The old shape filtered every date to the selected snapshot, so at an
    intraday bucket the "historical average" was an average over the one day
    intraday capture has existed. It also means hot_count_avg no longer
    counts today — at ~250 dates that moves it negligibly, and today is now
    measured on a different basis, so averaging it in would mix the two.
    """
    if not pool:
        return {"error": "OI database not configured"}

    cat = await _catalog(pool)
    e   = _entry(cat, metric)
    be  = _base_entry(cat, e)
    zw  = _z_window_of([e]) if e["form"] != "base" else None

    # The daily close of the BASE column, with fabricated observations nulled
    # rather than their rows dropped.
    val = f'm."{be["column_name"]}"'
    if exclude_extrapolated:
        val = f'CASE WHEN {_extrap_expr(be)} THEN NULL ELSE {val} END'

    async with pool.acquire() as conn:
        d, snap = await _resolve_slice(conn, date, snapshot)
        start   = _window_start(d, window)
        # The rolling window needs history BEFORE the display window starts,
        # or the first dates come back unscored.
        warm = None
        if start is not None:
            warm = date_type.fromordinal(
                start.toordinal() - _session_span_days(zw or 63))

        params = [hot, BASELINE_SNAPSHOT, d]
        where  = "m.snapshot = $2 AND m.trade_date < $3"
        if warm is not None:
            params.append(warm)
            where += f" AND m.trade_date >= ${len(params)}"

        if zw is None:
            expr = "b.v"
            roll = ""
        else:
            params.append(zw)
            p_zw = len(params)
            roll = (f", avg(v) OVER w AS mu, stddev_samp(v) OVER w AS sd, "
                    f"count(v) OVER w AS n_w")
            expr = (f"(CASE WHEN b.n_w >= {BASELINE_MIN_N} "
                    f"THEN (b.v - b.mu) / NULLIF(b.sd, 0) END)")

        cut = ""
        if start is not None:
            params.append(start)
            cut = f"WHERE b.trade_date >= ${len(params)}"

        # ROWS BETWEEN <zw> PRECEDING AND 1 PRECEDING is the rule stated as
        # SQL: the frame ends one row BEFORE the current session, so a date is
        # never one of the observations scoring it.
        win = ""
        if zw is not None:
            win = (f" WINDOW w AS (PARTITION BY ticker ORDER BY trade_date "
                   f"ROWS BETWEEN ${p_zw} PRECEDING AND 1 PRECEDING)")

        rows = await conn.fetch(
            f"WITH v AS ("
            f" SELECT m.ticker, m.trade_date, {val} AS v"
            f" {_from_clause(False)} WHERE {where}"
            f"), "
            f"b AS ("
            f" SELECT ticker, trade_date, v{roll} FROM v{win}"
            f") "
            f"SELECT b.trade_date,"
            f" count({expr})                                          AS n,"
            f" count(*) FILTER (WHERE {expr} > $1)                     AS n_hot,"
            f" percentile_cont(0.5) WITHIN GROUP (ORDER BY {expr})     AS med,"
            f" stddev_samp({expr})                                     AS disp "
            f"FROM b {cut} "
            f"GROUP BY b.trade_date ORDER BY b.trade_date",
            *params,
        )

        # Today, at the selected snapshot, against the daily baseline.
        tparams: list = [hot]
        tcte, tzmap = ("", {})
        if zw is not None:
            tcte, tzmap = _baseline_cte(cat, [e], d, zw, tparams)
        texpr = _z_expr(cat, e, tzmap)
        if exclude_extrapolated:
            texpr = f"CASE WHEN {_extrap_expr(e)} THEN NULL ELSE {texpr} END"
        tparams += [d, snap]
        td, ts = len(tparams) - 1, len(tparams)
        trow = await conn.fetchrow(
            (f"WITH {tcte} " if tcte else "")
            + f"SELECT count({texpr}) AS n,"
            + f" count(*) FILTER (WHERE {texpr} > $1) AS n_hot,"
            + f" percentile_cont(0.5) WITHIN GROUP (ORDER BY {texpr}) AS med,"
            + f" stddev_samp({texpr}) AS disp "
            + f"{_from_clause(False)} "
            + (f"{_baseline_join()} " if zw is not None else "")
            + f"WHERE m.trade_date = ${td} AND m.snapshot = ${ts}",
            *tparams,
        )

    series = [{"date": str(r["trade_date"]), "n": r["n"], "n_hot": r["n_hot"],
               "median": r["med"], "dispersion": r["disp"]} for r in rows]

    today = None
    if trow is not None and trow["n"]:
        today = {"date": str(d), "n": trow["n"], "n_hot": trow["n_hot"],
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
        "z_source": "daily_baseline" if zw else None,
        "history_basis": {"snapshot": BASELINE_SNAPSHOT, "through": "prior session",
                          "z_window": zw, "rolling": zw is not None},
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


def _filter_sql(entry: dict, op: str, raw: str, params: list, expr: str = None) -> str:
    """One `col:op:value` clause, with the value parameterized.

    `expr` lets the caller substitute a derived expression for the column —
    the scanner passes the daily-baseline z so that filtering and sorting on
    a z column mean the same thing as reading one.
    """
    if expr is None:
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

    A z column is DERIVED against each ticker's daily-close baseline rather
    than read from equity_metrics_z — see BASELINE_SNAPSHOT. The stored
    column is scored per (date, snapshot), so on an intraday bucket every z
    cell came back empty and every z filter matched nothing.

    Deriving it in SQL rather than in Python after the fetch is what keeps
    `filter=..._z_63:gt:1.5`, the ORDER BY and the LIMIT all meaning what
    they say. Scoring after the rows came back would filter a page of results
    instead of the universe, and "top 300 by skew z" would quietly become
    "300 arbitrary tickers, sorted".
    """
    if not pool:
        return {"error": "OI database not configured", "rows": []}

    cat  = await _catalog(pool)
    cols = [c.strip() for c in columns.split(",") if c.strip()]
    if not cols:
        raise HTTPException(400, "columns is empty")
    entries = [_entry(cat, c) for c in cols]

    # Parse filters before touching SQL: the baseline CTE has to cover every
    # z column in the request — selected, filtered or sorted on — and its
    # parameters have to be bound before any filter value's.
    parsed = []
    for f in filter:
        parts = f.split(":", 2)
        if len(parts) < 2:
            raise HTTPException(400, f"Malformed filter: {f!r} (want col:op:value)")
        parsed.append((_entry(cat, parts[0]), parts[1],
                       parts[2] if len(parts) > 2 else ""))

    se   = _entry(cat, sort) if sort else None
    used = entries + [p[0] for p in parsed] + ([se] if se else [])
    zs   = [e for e in used if e["form"] != "base"]

    async with pool.acquire() as conn:
        d, snap = await _resolve_slice(conn, date, snapshot)

        params: list = []
        cte, zmap = ("", {})
        if zs:
            cte, zmap = _baseline_cte(cat, zs, d, _z_window_of(zs), params)

        def col(e):
            return _z_expr(cat, e, zmap)

        sel = ["m.ticker"]
        for i, e in enumerate(entries):
            sel.append(f"{col(e)} AS v{i}")
            sel.append(f"{_extrap_expr(e)} AS e{i}")
        sel += ['m."{0}" AS {0}'.format(c) for c in CONTEXT_COLS]

        where = [_filter_sql(fe, op, raw, params, expr=col(fe))
                 for fe, op, raw in parsed]

        order = "m.ticker"
        if se:
            order = (f"{col(se)} {'DESC' if dir.lower() == 'desc' else 'ASC'} "
                     f"NULLS LAST, m.ticker")

        params += [d, snap, limit]
        nd, ns, nl = len(params) - 2, len(params) - 1, len(params)
        clause = f"m.trade_date = ${nd} AND m.snapshot = ${ns}"
        if where:
            clause += " AND " + " AND ".join(where)

        rows = await conn.fetch(
            (f"WITH {cte} " if cte else "")
            + f"SELECT {', '.join(sel)} "
            + f"{_from_clause(False)} "
            + (f"{_baseline_join()} " if zs else "")
            + f"WHERE {clause} "
            + f"ORDER BY {order} "
            + f"LIMIT ${nl}",
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
        "z_source":  "daily_baseline" if zs else None,
        "baseline":  ({"snapshot": BASELINE_SNAPSHOT,
                       "z_window": _z_window_of(zs),
                       "min_n": BASELINE_MIN_N,
                       "through": "prior session"} if zs else None),
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

    BOTH numbers on a card come from the daily close series (see
    BASELINE_SNAPSHOT), whatever snapshot is on screen:

      z           (today - mu) / sigma against the last `z_window` sessions,
                  derived here rather than read from equity_metrics_z. The
                  stored column is scored per (date, snapshot), so at an
                  intraday bucket it is a z against that bucket's own short
                  history -- which is why this strip used to report "no
                  metric has both a value and a z" at 1200.

      percentile  rank within the daily closes over the page's history
                  window. Also the reason 100th percentile came back for
                  everything: ranked against one same-bucket observation,
                  every value is its own maximum.

    Both windows END AT THE PRIOR SESSION. Today is the thing being scored,
    so letting it into the distribution scoring it drags the answer toward
    ordinary -- and guarantees the top of any ranking is at the 100th
    percentile by construction.

    Note the two windows differ on purpose: z uses `z_window` sessions
    because that is what a sigma means here, while the percentile uses the
    page's history window because that is the span the rails and the charts
    are showing. They answer different questions.

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

    # The candidate set stays "base columns the catalog gives a z variant at
    # this window". The z column is no longer READ -- its existence is the
    # loader's judgment about which metrics are worth scoring, and that
    # curation is worth keeping even now that the score is derived.
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
    for i, (be, _ze) in enumerate(pairs):
        sel += [f"{_expr(be)} AS b{i}", f"{_extrap_expr(be)} AS e{i}"]

    async with pool.acquire() as conn:
        d, snap = await _resolve_slice(conn, date, snapshot)

        # Today's values only -- no join to equity_metrics_z anywhere.
        row = await conn.fetchrow(
            f"SELECT {', '.join(sel)} {_from_clause(False)} "
            f"WHERE m.ticker = $1 AND m.trade_date = $2 AND m.snapshot = $3",
            ticker, d, snap,
        )
        if row is None:
            return {"ticker": ticker, "date": str(d), "snapshot": snap,
                    "cards": [], "error": f"No row for {ticker}"}

        bases = [be for be, _ in pairs]
        base_stats = await _baseline_for(conn, cat, bases, ticker, d, z_window)

        ranked, n_thin = [], 0
        for i, (be, ze) in enumerate(pairs):
            bv, ex = row[f"b{i}"], row[f"e{i}"]
            if bv is None:
                continue
            v  = float(bv)
            zv = _z_from(base_stats, be["column_name"], v)
            if zv is None:
                # A live value with no score: the baseline is thinner than
                # BASELINE_MIN_N, or flat across the window. Counted, not
                # ranked -- ranking it at z=0 would bury a real reading in
                # the middle of the strip as though it had been measured.
                n_thin += 1
                continue
            ranked.append({"base": be, "z_entry": ze, "value": v,
                           "z": zv, "extrap": bool(ex)})
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

    def stat(c):
        return base_stats.get(c["base"]["column_name"], {})

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
        "z_n":         stat(c).get("n"),
        "extrap":      c["extrap"],
        "extrap_flags": c["base"]["extrap_flags"],
    } for c in shown]

    any_stat = next(iter(base_stats.values()), {})
    return {
        "ticker": ticker, "date": str(d), "snapshot": snap,
        "z_window": z_window, "window": window,
        "cards": cards,
        "n_ranked": len(ranked),
        "n_withheld_extrapolated": withheld,
        "n_unscored_thin_baseline": n_thin,
        "exclude_extrapolated": bool(exclude_extrapolated),
        "extrap_rate": row["extrap_rate_short"],
        "z_source": "daily_baseline",
        "baseline": {
            "snapshot": BASELINE_SNAPSHOT,
            "z_window": z_window,
            "first":    any_stat.get("first"),
            "last":     any_stat.get("last"),
            "sessions": any_stat.get("sessions", 0),
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
    ("VRP 1m",           ("vrp_1m", "vrp_1m_", "vrp_21d")),
    ("spot-vol β 1m",    ("spotvol_beta_1m",)),
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

    The z beside the bar is the daily-baseline z over `z_window` sessions,
    the same number /unusual and the scanner show. It is a label, not the
    bar's geometry: "where in the range" and "how many sigma" answer
    different questions, and the header chips speak in sigma.

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
    _reject_z_form(entries, "A rail", "Pass the base column — each rail "
                   "already shows its daily-baseline z beside the bar.")

    out = []
    async with pool.acquire() as conn:
        d, snap = await _resolve_slice(conn, date, snapshot)
        start   = _window_start(d, window)

        # One baseline query covers every rail.
        stats = await _baseline_for(conn, cat, entries, ticker, d, z_window)

        # Today's values at the SELECTED snapshot, one row.
        sel = []
        for i, e in enumerate(entries):
            sel += [f"{_expr(e)} AS v{i}", f"{_extrap_expr(e)} AS x{i}"]
        cur = await conn.fetchrow(
            f"SELECT {', '.join(sel)} {_from_clause(False)} "
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

        st = stats.get(e["column_name"], {})
        out.append({
            **_meta(e),
            "value":      shown,
            "raw_value":  None if v is None else float(v),
            "extrap":     ex,
            "z":          _z_from(stats, e["column_name"], shown),
            "z_n":        st.get("n"),
            "percentile": pct,
            "p5":  _f(dist, f"p5_{i}"),  "p25": _f(dist, f"p25_{i}"),
            "p50": _f(dist, f"p50_{i}"), "p75": _f(dist, f"p75_{i}"),
            "p95": _f(dist, f"p95_{i}"),
            "n":   int(dist[f"n{i}"]) if dist and dist[f"n{i}"] else 0,
        })

    any_stat = next(iter(stats.values()), {})
    return {
        "ticker": ticker, "date": str(d), "snapshot": snap,
        "window": window, "z_window": z_window,
        "rails": out,
        "exclude_extrapolated": bool(exclude_extrapolated),
        "z_source": "daily_baseline",
        # Which default slot each rail came from, and which found nothing.
        # Returned only when the caller took the defaults; an explicit metric
        # list is its own answer.
        "defaults": ([{"slot": l, "column": c} for l, c in slots]
                     if slots is not None else None),
        "baseline": {
            "snapshot": BASELINE_SNAPSHOT, "z_window": z_window,
            "first": any_stat.get("first"), "last": any_stat.get("last"),
            "sessions": any_stat.get("sessions", 0), "min_n": BASELINE_MIN_N,
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
DAILY_SNAPSHOT = BASELINE_SNAPSHOT
SERIES_MAX     = 4


async def _daily_baseline(conn, cat, entry, ticker, as_of, z_window):
    """(mu, sigma, n) for one metric over the DAILY close series.

    Thin wrapper over _baseline_for() so /series shares one implementation
    with every other panel. It used to have its own, and that copy had both
    of the bugs the shared one is written to avoid:

      * it scored against `trade_date <= as_of`, so today's own close sat
        inside the window judging it — and only escaped that mid-session,
        before the 15:45 row was captured. A window that changes definition
        at the close every day is worse than either rule applied
        consistently.
      * it took each metric's last N NON-NULL observations rather than the
        last N sessions, so a sparse metric was quietly scored against a
        longer and older stretch of market than the metric beside it.
    """
    return (await _baseline_for(conn, cat, [entry], ticker, as_of, z_window)) \
        .get(entry["column_name"], {"mu": None, "sigma": None, "n": 0,
                                    "first": None, "last": None, "sessions": 0})


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

    What a partial point must NEVER do is help score itself:

      * it is kept out of the rolling envelope's history, so the band it sits
        against is built from settled sessions only. Its own band still
        renders, because the envelope was already trailing — computed from
        history strictly before each point.
      * the z baseline never sees it. _baseline_for reads BASELINE_SNAPSHOT
        strictly before the anchor date, so the yardstick is prior sessions
        whatever the live point does.

    Once the close lands, the settled 1545 row IS the point and nothing is
    appended — the partial reading is replaced by the real one rather than
    left sitting beside it.

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
    _reject_z_form(entries, "A series", "Pass the base column — every point "
                   "already carries its daily-baseline z.")

    out = []
    appended_live = False
    async with pool.acquire() as conn:
        as_of, live_snap = await _resolve_slice(conn, date, live_snapshot)
        daily_snap = snapshot or DAILY_SNAPSHOT
        start      = _window_start(as_of, window)

        for e in entries:
            zcol = None
            for cand in cat["by_col"].values():
                if cand["form"] == f"z_{z_window}" and cand["base_column"] == e["column_name"]:
                    zcol = cand
                    break

            sel = ["m.trade_date", "m.snapshot", f"{_expr(e)} AS v",
                   f"{_extrap_expr(e)} AS ex"]
            # z_stored is a like-for-like check on the derived score, so it is
            # only fetched when the plotted bucket IS the baseline bucket.
            # Reading it at any other snapshot would compare the derived daily
            # z against that bucket's own same-snapshot z — two different
            # measurements, displayed as though one were a check on the other.
            needs_z = bool(zcol) and mode == "daily" \
                and daily_snap == BASELINE_SNAPSHOT
            want_stored = needs_z
            if want_stored:
                sel.append(f"{_expr(zcol)} AS zs")

            where = ["m.ticker = $1"]
            args  = [ticker]
            # Upper-bounded at the anchor date. Without this the chart ran to
            # whatever the newest capture was, so picking an earlier date moved
            # every other panel on the page and left this one alone.
            args.append(as_of)
            p_asof = len(args)
            where.append(f"m.trade_date <= ${p_asof}")
            if mode == "daily":
                args.append(daily_snap)
                where.append(f"m.snapshot = ${len(args)}")
            else:
                # The anchor date is truncated at the selected bucket, so the
                # intraday line stops where the page says it is rather than
                # running ahead to the newest capture. Snapshots are
                # zero-padded HHMM text, so this ordering is chronological.
                args.append(live_snap)
                where.append(f"(m.trade_date < ${p_asof} "
                             f"OR m.snapshot <= ${len(args)})")
            if start:
                args.append(start)
                where.append(f"m.trade_date >= ${len(args)}")

            rows = await conn.fetch(
                f"SELECT {', '.join(sel)} {_from_clause(needs_z)} "
                f"WHERE {' AND '.join(where)} "
                f"ORDER BY m.trade_date, m.snapshot",
                *args,
            )

            # Note the baseline takes neither daily_snap nor live_snap. The
            # caller may plot any bucket; the yardstick is 1545 closes strictly
            # before the anchor date either way.
            bl = await _daily_baseline(conn, cat, e, ticker, as_of, z_window)
            mu, sd, n_base = bl["mu"], bl["sigma"], bl["n"]

            def zof(v, _mu=mu, _sd=sd):
                if v is None or _mu is None or not _sd:
                    return None
                return (v - _mu) / _sd

            # In daily mode the settled close may simply not exist yet.
            live_row = None
            settled_today = any(r["trade_date"] == as_of for r in rows)
            if (include_today and mode == "daily" and not settled_today
                    and live_snap != daily_snap):
                live_row = await conn.fetchrow(
                    f"SELECT m.trade_date, m.snapshot, {_expr(e)} AS v, "
                    f"{_extrap_expr(e)} AS ex {_from_clause(False)} "
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
                by_day, last_snap = {}, {}
                for r in rows:
                    if r["v"] is None or (exclude_extrapolated and r["ex"]):
                        continue
                    by_day.setdefault(r["trade_date"], []).append(float(r["v"]))
                    last_snap[r["trade_date"]] = r["snapshot"]   # rows are ordered
                pts = []
                for dd in sorted(by_day):
                    vs = by_day[dd]                   # already snapshot-ordered
                    pts.append({"t": dd.isoformat(), "o": vs[0], "h": max(vs),
                                "l": min(vs), "c": vs[-1], "n": len(vs),
                                "z": zof(vs[-1]),
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
                    p = {"t": r["trade_date"].isoformat(), "v": shown, "extrap": ex,
                         "z": zof(shown),
                         "partial": is_partial(r["trade_date"], r["snapshot"])}
                    if mode == "intraday" or p["partial"]:
                        p["snapshot"] = r["snapshot"]
                    if want_stored:
                        p["z_stored"] = None if r["zs"] is None else float(r["zs"])
                    pts.append(p)

                if live_row is not None and live_row["v"] is not None:
                    ex    = bool(live_row["ex"])
                    shown = None if (exclude_extrapolated and ex) else float(live_row["v"])
                    # snapshot comes from live_snap, not from the row: the
                    # query filtered on it, so it is the same value by
                    # construction, and taking it from the parameter means the
                    # label cannot disagree with what was asked for.
                    pts.append({"t": as_of.isoformat(), "v": shown, "extrap": ex,
                                "z": zof(shown), "partial": True,
                                "snapshot": live_snap})
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
                "baseline":   {"mu": mu, "sigma": sd, "n": n_base,
                               "snapshot": BASELINE_SNAPSHOT,
                               "first": bl["first"], "last": bl["last"],
                               "sessions": bl["sessions"],
                               "as_of": as_of.isoformat(),
                               "z_window": z_window},
                "z_stored_column": zcol["column_name"] if want_stored else None,
            })

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
        "z_source": "daily_baseline",
        "envelope": ({"on": True, "window": env_window, "lo": env_lo, "hi": env_hi}
                     if envelope else {"on": False}),
        "exclude_extrapolated": bool(exclude_extrapolated),
        "series": out,
    }
