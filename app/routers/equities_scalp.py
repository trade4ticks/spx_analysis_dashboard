"""Equities Scalp — read-only endpoints over the `equities_scalp` database.

WHAT THIS OWNS AND WHAT IT DOES NOT. The pipeline in the Open_Interest repo's
`scalp/` computes ~75 metrics per symbol-day and per 15-minute bucket and
writes them here. This module READS them. Nothing on this page writes to
universe, daily_metrics, intraday_metrics, provenance or rankings; the only
writes this project will ever make are the fills tables, which the pipeline
does not own.

NO HARDCODED METRIC NAMES. The metric set is explicitly unsettled -- five
noise variants at three horizons at five statistics, plus flicker and flow
metrics, and a calibration process whose purpose is to delete most of them. A
column list in this file would mean editing this project every time the
pipeline changed one, and the failure mode is silent: a renamed metric becomes
a column of nulls, not an error.

So the catalog is READ from the database. `/meta` returns what
`daily_metrics.metric` actually contains on the latest date, decorated with
the definitions from the vendored metric_docs. scripts/check_scalp_metrics.py
fails the build if a metric-name literal appears in this file or the page's
JS.

LONG, NOT WIDE. daily_metrics is (trade_date, symbol, metric, value), so every
read pivots. That is the pipeline's deliberate choice -- see scalp/db.py -- and
the pivot happens here rather than being pushed onto the client, which would
ship five long rows to draw one wide one.
"""
from __future__ import annotations

import datetime as _dt
import re

from fastapi import APIRouter, Depends, HTTPException, Query

from app.db import get_scalp_pool, pool_status
from app import scalp_config, scalp_metric_docs

router = APIRouter()


# The pipeline's own read-time thresholds and their slider bounds. Vendored
# rather than restated: scripts/check_vendored.py diffs the whole file against
# the pipeline's copy, so a threshold moved upstream cannot sit here at its old
# value looking authoritative.
FILTER_KEYS = tuple(scalp_config.DEFAULT_FILTERS)


# ── the shape of a metric name ───────────────────────────────────────────────
#
# The generated families are `<kind>_<variant>_<horizon>s_<statistic>`, e.g.
# noise_bps_tw_mid_10s_rms. This parses that SHAPE without naming any
# particular variant, horizon or statistic -- the point is to discover which
# ones the pipeline currently emits, not to assert which ones exist.
_VARIANT_RE = re.compile(
    r"^(?P<kind>noise_bps|ratio)_(?P<variant>.+?)_(?P<horizon>\d+)s"
    r"(?:_(?P<stat>[a-z0-9]+))?$"
)


def _parse_variant(metric: str) -> dict | None:
    m = _VARIANT_RE.match(metric)
    if not m:
        return None
    return {"kind": m.group("kind"), "variant": m.group("variant"),
            "horizon_s": int(m.group("horizon")), "statistic": m.group("stat")}


def _catalog_entry(metric: str) -> dict:
    """One metric, with its documentation link and its parsed shape."""
    link = scalp_metric_docs.header_link(metric)
    return {
        "metric":  metric,
        "label":   link.get("label", metric),
        "tooltip": link.get("tooltip"),
        "href":    link.get("href"),
        "section": link.get("section"),
        # None for a metric that is not one of the generated families, which
        # is most of the flow and flicker set.
        "variant": _parse_variant(metric),
    }


# ── dates cross the boundary as strings and must not stay that way ──────────
#
# asyncpg binds by TYPE, not by content: a DATE column needs a datetime.date
# and it will not coerce a string, unlike psycopg2. The error it raises names
# neither the column nor the endpoint --
#
#     invalid input for query argument $1: '2026-08-28'
#     ('str' object has no attribute 'toordinal')
#
# -- and it happens at bind time, so it survives every check that stops at
# whether the SQL parses.
#
# The trap here is specific and will recur through P2-P5: `dates` is stringified
# for the JSON response, and the natural next line picks the anchor out of that
# already-stringified list. The value is a string by the time it is bound, and
# nothing in between looks wrong.
#
# So dates are kept as date objects for the whole of their life inside a
# handler, and stringified once, at the response. Anything that arrives from a
# query parameter goes through _as_date first.


def _as_date(value, field: str = "date"):
    """A query parameter to a datetime.date, or a 400 that says which field.

    None passes through: an absent date means "the latest", which the caller
    resolves. A date object passes through unchanged so this is safe to apply
    twice.
    """
    if value is None or isinstance(value, _dt.date) and not isinstance(value, _dt.datetime):
        return value
    if isinstance(value, _dt.datetime):
        return value.date()
    try:
        return _dt.date.fromisoformat(str(value))
    except (TypeError, ValueError):
        raise HTTPException(
            400, f"{field}={value!r} is not an ISO date (YYYY-MM-DD). "
                 f"asyncpg binds a DATE column by type and will not coerce a "
                 f"string, so this has to be resolved before the query.")


def _no_db(extra: dict | None = None) -> dict:
    """The not-connected state, spelled out — with the startup reason.

    Distinguished from "no data for this date" deliberately: one is a missing
    database and the other is a night the pipeline did not run, and they need
    different actions from whoever is reading the page at 9am.

    The REASON is carried through from startup rather than guessed at here. A
    missing database, wrong credentials and a wiring fault produced identical
    output once, and telling them apart cost an hour of elimination. The pool
    layer already knows which one it was; this just stops throwing that away.
    """
    st = pool_status().get("equities_scalp") or {}
    if not st.get("configured"):
        why = ("No DSN is configured. It derives from DATABASE_URL by default; "
               "set SCALP_DATABASE_URL if the database lives elsewhere.")
    elif st.get("error"):
        why = f"Connecting failed at startup — {st['error']}"
    else:
        # verify_pools() refuses to start on this, so reaching it means the
        # guard was bypassed rather than that the state is normal.
        why = ("Configured, no recorded failure, and no pool. That is a wiring "
               "fault rather than an environment one.")
    out = {
        "connected": False,
        "error": f"No connection to the {scalp_config.PG_DB!r} database. {why}",
        "reason": st or None,
        "dates": [], "latest_date": None, "metrics": [],
    }
    out.update(extra or {})
    return out


@router.get("/meta")
async def meta(
    date: str = Query(None, description="ISO date; defaults to the latest"),
    pool=Depends(get_scalp_pool),
):
    """Everything the page needs before it can draw anything.

    The available dates, the metric catalog as the database actually holds it,
    the noise variants discovered from those metric names, and the pipeline's
    filter defaults and slider bounds.

    The variant list is DERIVED, not declared. Every dropdown of noise variants
    on this page is built from what `daily_metrics` contains on the selected
    date, so a variant the calibration deletes disappears from the page without
    anything here being edited -- and one that is added appears the same way.
    """
    if pool is None:
        return _no_db()

    want = _as_date(date)

    async with pool.acquire() as conn:
        # DATE OBJECTS, not strings. The stringified list below is for the
        # response; the anchor is picked out of THIS one, so what gets bound is
        # never the formatted copy.
        available = [r["trade_date"] for r in await conn.fetch(
            "SELECT DISTINCT trade_date FROM daily_metrics "
            "ORDER BY trade_date DESC LIMIT 90")]
        dates = [str(x) for x in available]
        if not available:
            return {
                "connected": True, "dates": [], "latest_date": None,
                "metrics": [], "variants": [], "statistics": [], "horizons": [],
                "filters": _filter_block(),
                "note": "The database is reachable but daily_metrics is empty "
                        "— the pipeline has not written a session yet.",
            }

        d = want if want in available else available[0]

        rows = await conn.fetch(
            "SELECT metric, count(*) AS n, count(value) AS n_value "
            "FROM daily_metrics WHERE trade_date = $1 "
            "GROUP BY metric ORDER BY metric", d)
        symbols = await conn.fetchval(
            "SELECT count(DISTINCT symbol) FROM daily_metrics "
            "WHERE trade_date = $1", d)

    catalog = []
    for r in rows:
        e = _catalog_entry(r["metric"])
        # Carried so a metric that is present but all-null is visible as such.
        # An all-null column and an absent one look identical in a pivot, and
        # only one of them means the pipeline is broken.
        e["n"] = int(r["n"])
        e["n_value"] = int(r["n_value"])
        catalog.append(e)

    parsed = [e for e in catalog if e["variant"]]
    ratios = [e for e in parsed if e["variant"]["kind"] == "ratio"]
    noises = [e for e in parsed if e["variant"]["kind"] == "noise_bps"]

    def _uniq(items, key):
        seen, out = set(), []
        for it in items:
            v = it["variant"][key]
            if v is not None and v not in seen:
                seen.add(v)
                out.append(v)
        return out

    return {
        "connected": True,
        "dates": dates,
        # Stringified HERE, at the response, and nowhere earlier.
        "date": str(d),
        "latest_date": dates[0],
        "symbols": int(symbols or 0),
        "metrics": catalog,
        # The dropdown's options, and the axes it is built from.
        "noise_metrics": [e["metric"] for e in noises],
        "ratio_metrics": [e["metric"] for e in ratios],
        "variants":   _uniq(noises, "variant"),
        "horizons":   sorted(_uniq(noises, "horizon_s")),
        "statistics": _uniq(noises, "statistic"),
        "default_noise": _default_noise([e["metric"] for e in noises]),
        "filters": _filter_block(),
        "undocumented": scalp_metric_docs.undocumented(
            [e["metric"] for e in catalog]),
    }


# The variant the ranking opens on until calibration says otherwise.
#
# NOT the median statistic, and this is the one place a preference is
# expressed. The median collapses to exactly 0.0 on a sparse-quote name -- when
# more than half of consecutive buckets carry an identical midpoint, the median
# change is zero by construction -- which makes the ratio infinite and sorts
# the least tradeable names to the top. rms is the same measurement without
# that failure.
#
# Written as a PREFERENCE ORDER over what the database actually has, not as a
# literal. If the preferred name is gone, the page opens on something real
# rather than on a column of nulls.
_NOISE_PREFERENCE = ("tw_mid", "last_mid", "trade_price")
# median is ranked BELOW an unrecognised statistic, on purpose. Everything else
# here is taste; this one is a defect. Being unfamiliar is a reason to look at
# a number, whereas median is known to read 0.0 on exactly the names the filter
# is supposed to exclude.
_STAT_PREFERENCE = ("rms", "p75", "p90", "mean")
_STAT_LAST = ("median",)


def _default_noise(available: list[str]) -> str | None:
    if not available:
        return None
    def score(metric: str):
        p = _parse_variant(metric) or {}
        v = p.get("variant") or ""
        s = p.get("statistic") or ""
        if s in _STAT_PREFERENCE:
            stat_rank = _STAT_PREFERENCE.index(s)
        elif s in _STAT_LAST:
            stat_rank = 99
        else:
            stat_rank = 50
        return (
            # STATISTIC OUTRANKS VARIANT. Which midpoint definition is used is
            # a preference; which statistic summarises it is the difference
            # between a number and a zero, so it decides first.
            stat_rank,
            _NOISE_PREFERENCE.index(v) if v in _NOISE_PREFERENCE else 99,
            abs((p.get("horizon_s") or 0) - 10),   # 10s, nearest first
            metric,
        )
    return sorted(available, key=score)[0]


def _filter_block() -> dict:
    """Defaults and slider bounds, straight from the vendored pipeline config.

    Both halves matter. The defaults are the pipeline's current opinion; the
    ranges are the span a threshold can be dragged through, so a value can be
    moved against the data rather than typed blind. Every one of these is a
    READ-TIME filter: the pipeline stores every symbol whether it passes or
    not, and the rows that fail are the only evidence that can say whether a
    threshold is set correctly.
    """
    return {
        "defaults": dict(scalp_config.DEFAULT_FILTERS),
        "ranges": {k: {"min": lo, "max": hi, "step": st}
                   for k, (lo, hi, st) in scalp_config.FILTER_RANGES.items()},
        "keys": list(FILTER_KEYS),
        "ratio_guard": scalp_config.MIN_NOISE_BPS_FOR_RATIO,
        "read_time_only": True,
    }
