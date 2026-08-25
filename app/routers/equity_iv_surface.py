"""
Equity IV Analysis — the surface-reading half (spec rows 6 to 8).

Mounted under the same /api/equity-iv prefix as equity_iv.py, and split off it
for the same reason ticker_chain.py is split from ticker_analysis.py: that
module is already ~1900 lines and these endpoints read different tables.

equity_iv.py reads the METRIC layer — equity_metrics, one row per
(ticker, trade_date, snapshot), pre-computed columns described by a catalog.
This module reads the SURFACE underneath it:

  equity_surface  one row per (ticker, trade_date, snapshot, dte, put_delta)
                  17 tenors x 19 deltas. iv, strike, forward, prices, greeks,
                  and the `extrapolated` flag.
  equity_atm      one row per (ticker, trade_date, snapshot, dte). atm_iv,
                  atm_strike, atm_forward, underlying_price.

Two conventions carry over from equity_iv.py and are not restated at each
endpoint:

DELTA. `put_delta` is a positive integer 5-95, every node priced as a put.
55-95 are ITM puts whose IV came from the OTM CALL quote at that strike, via
put-call parity — genuine call data in put convention. So a 25-delta CALL is
put_delta 75. ATM is NOT put_delta 50: it lives in equity_atm, evaluated at
k=0, and that is what every "vs ATM" view here anchors on.

SCORING. History is daily closes at BASELINE_SNAPSHOT ending at the PRIOR
session; today is the selected snapshot. Every band, z and percentile on this
page follows that rule, so a reading at 11:25 is placed against the daily
distribution rather than against the handful of same-bucket observations that
exist. See the long comment on equity_iv.BASELINE_SNAPSHOT for why.

EXTRAPOLATION. equity_surface.extrapolated is a per-node boolean, which makes
this module's handling far simpler than the metric layer's flag resolution:
a node is fabricated or it is not. Fabricated nodes are excluded from every
historical distribution — otherwise "normal range" is partly built from values
the spline invented — and marked, never silently dropped, in today's curve.
"""
import math
from datetime import date as date_type

from fastapi import APIRouter, Depends, HTTPException, Query

from app.db import get_oi_pool
from app.metrics_config import BASELINE_SNAPSHOT, BASELINE_MIN_N
from app.routers.equity_iv import (
    ATM_TABLE, METRICS_TABLE, _catalog, _entry, _expr, _extrap_expr,
    _first_live, _from_clause, _jsonable, _meta, _needs_z, _resolve_slice,
    _window_start,
)

router = APIRouter(tags=["equity-iv"])

SURFACE_TABLE = "equity_surface"

_axes_cache: dict | None = None


async def _axes(pool) -> dict:
    """The tenors and deltas the surface is actually fitted at.

    Read from the data rather than hardcoded. equity_iv.TENORS exists but is a
    different thing — the tenors that appear in METRIC column names, used to
    turn a name into an extrap flag. The surface carries 17 tenors and 19
    deltas, and a grid built from the metric list would quietly omit the rest.

    Cached for the process lifetime: the fitted grid changes when the loader
    changes, which is a deploy.
    """
    global _axes_cache
    if _axes_cache is not None:
        return _axes_cache
    async with pool.acquire() as conn:
        dtes = await conn.fetch(
            f"SELECT DISTINCT dte FROM {SURFACE_TABLE} ORDER BY dte")
        deltas = await conn.fetch(
            f"SELECT DISTINCT put_delta FROM {SURFACE_TABLE} ORDER BY put_delta")
    _axes_cache = {
        "dtes":   [int(r["dte"]) for r in dtes],
        "deltas": [int(r["put_delta"]) for r in deltas],
    }
    return _axes_cache


def _nearest(values, want):
    """The available axis value closest to `want`. Returns None if empty."""
    if not values:
        return None
    return min(values, key=lambda v: abs(v - want))


# P5 / P25 / P50 / P75 / P95 — the SAME five the rails use, so that "outside
# the band" carries one meaning everywhere on the page. The tent shipped on
# P10/P90 and the rails on P5/P95, which made the identical visual position
# mean two different things one panel apart.
BAND_QUANTILES = ((0.05, "p5"), (0.25, "p25"), (0.50, "p50"),
                  (0.75, "p75"), (0.95, "p95"))


def _band_aggs(expr: str, alias: str) -> str:
    """P5 / P25 / P50 / P75 / P95 of one expression.

    Percentiles rather than mean +/- k*sd, for the reason stated throughout
    this page: these distributions are right-skewed and fat-tailed, so a
    symmetric band is wrong asymmetrically — too wide on one side and too
    narrow on the tail that matters.
    """
    return ", ".join(
        f"percentile_cont({q}) WITHIN GROUP (ORDER BY {expr}) AS {alias}_{nm}"
        for q, nm in BAND_QUANTILES
    ) + f", count({expr}) AS {alias}_n"


def _band_of(row, alias):
    if row is None:
        return None
    out = {k: (None if row[f"{alias}_{k}"] is None else float(row[f"{alias}_{k}"]))
           for _q, k in BAND_QUANTILES}
    out["n"] = int(row[f"{alias}_n"] or 0)
    return out


# ── Row 6: three "today vs history" band panels ─────────────────────────────

@router.get("/curve-band")
async def curve_band(
    ticker:               str  = Query(...),
    kind:                 str  = Query("skew", description="skew|term|skew_term"),
    dte:                  int  = Query(30, description="skew: the tenor plotted"),
    wing:                 int  = Query(25, description="skew_term: the put wing"),
    deltas:               str  = Query("25,75", description="term: CSV put_deltas"),
    date:                 str  = Query(None),
    snapshot:             str  = Query(None),
    window:               str  = Query("1y"),
    exclude_extrapolated: bool = Query(True),
    pool=Depends(get_oi_pool),
):
    """Today's curve against its own historical band. Two of row 6's panels.

    kind:
      skew       IV by delta at one tenor. The smile.
      term       IV by tenor at chosen deltas, plus the ATM line from
                 equity_atm. The term structure.
      skew_term  the PUT-WING SLOPE by tenor: iv(wing) - atm_iv, per tenor.

    That last one is a separate toggle rather than a second series on `term`
    because it answers a different question. Whether the 21-day wing is rich
    relative to the 60-day wing is a different setup from both being rich
    together, and IV-by-tenor cannot separate them — a parallel shift in the
    whole surface moves every tenor's IV and leaves every tenor's SLOPE alone.

    The band is P5/P25/P50/P75/P95 across prior sessions at each x, so it is
    a pointwise envelope, NOT a set of historical curves. A curve tracing the
    P95 line at every delta is not a curve that ever traded; the band says
    "each point, against its own history", which is the question being asked.
    """
    if not pool:
        return {"error": "OI database not configured", "points": []}
    if kind not in ("skew", "term", "skew_term"):
        raise HTTPException(400, f"kind must be skew|term|skew_term, got {kind!r}")

    ax = await _axes(pool)
    keep = "" if not exclude_extrapolated else " AND NOT s.extrapolated"

    async with pool.acquire() as conn:
        d, snap = await _resolve_slice(conn, date, snapshot)
        start = _window_start(d, window)

        hist_where = (f"s.ticker = $1 AND s.snapshot = $2 AND s.trade_date < $3"
                      f"{keep}")
        hist_args  = [ticker, BASELINE_SNAPSHOT, d]
        if start:
            hist_args.append(start)
            hist_where += f" AND s.trade_date >= ${len(hist_args)}"

        if kind == "skew":
            use_dte = _nearest(ax["dtes"], dte)
            if use_dte is None:
                return {"error": "equity_surface has no tenors", "points": []}

            today = await conn.fetch(
                f"SELECT s.put_delta AS x, s.iv AS v, s.strike, s.extrapolated "
                f"FROM {SURFACE_TABLE} s "
                f"WHERE s.ticker = $1 AND s.trade_date = $2 AND s.snapshot = $3 "
                f"  AND s.dte = $4 ORDER BY s.put_delta",
                ticker, d, snap, use_dte,
            )
            band = await conn.fetch(
                f"SELECT s.put_delta AS x, {_band_aggs('s.iv', 'b')} "
                f"FROM {SURFACE_TABLE} s "
                f"WHERE {hist_where} AND s.dte = ${len(hist_args) + 1} "
                f"GROUP BY s.put_delta ORDER BY s.put_delta",
                *hist_args, use_dte,
            )
            axis = {"label": "put_delta", "dte": use_dte}

        elif kind == "term":
            want = [int(x) for x in deltas.split(",") if x.strip().lstrip("-").isdigit()]
            want = [dl for dl in want if dl in ax["deltas"]]
            if not want:
                want = [dl for dl in (25, 75) if dl in ax["deltas"]]

            today = await conn.fetch(
                f"SELECT s.dte AS x, s.put_delta AS series, s.iv AS v, "
                f"       s.strike, s.extrapolated "
                f"FROM {SURFACE_TABLE} s "
                f"WHERE s.ticker = $1 AND s.trade_date = $2 AND s.snapshot = $3 "
                f"  AND s.put_delta = ANY($4::int[]) ORDER BY s.put_delta, s.dte",
                ticker, d, snap, want,
            )
            band = await conn.fetch(
                f"SELECT s.dte AS x, s.put_delta AS series, {_band_aggs('s.iv', 'b')} "
                f"FROM {SURFACE_TABLE} s "
                f"WHERE {hist_where} AND s.put_delta = ANY(${len(hist_args) + 1}::int[]) "
                f"GROUP BY s.dte, s.put_delta ORDER BY s.put_delta, s.dte",
                *hist_args, want,
            )
            axis = {"label": "dte", "deltas": want}

        else:  # skew_term
            use_wing = _nearest(ax["deltas"], wing)
            slope = "(s.iv - a.atm_iv)"
            today = await conn.fetch(
                f"SELECT s.dte AS x, {slope} AS v, s.strike, s.extrapolated "
                f"FROM {SURFACE_TABLE} s "
                f"JOIN {ATM_TABLE} a ON a.ticker = s.ticker "
                f" AND a.trade_date = s.trade_date AND a.snapshot = s.snapshot "
                f" AND a.dte = s.dte "
                f"WHERE s.ticker = $1 AND s.trade_date = $2 AND s.snapshot = $3 "
                f"  AND s.put_delta = $4 ORDER BY s.dte",
                ticker, d, snap, use_wing,
            )
            band = await conn.fetch(
                f"SELECT s.dte AS x, {_band_aggs(slope, 'b')} "
                f"FROM {SURFACE_TABLE} s "
                f"JOIN {ATM_TABLE} a ON a.ticker = s.ticker "
                f" AND a.trade_date = s.trade_date AND a.snapshot = s.snapshot "
                f" AND a.dte = s.dte "
                f"WHERE {hist_where} AND s.put_delta = ${len(hist_args) + 1} "
                f"GROUP BY s.dte ORDER BY s.dte",
                *hist_args, use_wing,
            )
            axis = {"label": "dte", "wing": use_wing}

        # The ATM line, which does not come from equity_surface at all.
        atm_today, atm_band = [], []
        if kind == "term":
            atm_today = await conn.fetch(
                f"SELECT a.dte AS x, a.atm_iv AS v FROM {ATM_TABLE} a "
                f"WHERE a.ticker = $1 AND a.trade_date = $2 AND a.snapshot = $3 "
                f"ORDER BY a.dte",
                ticker, d, snap,
            )
            aargs = [ticker, BASELINE_SNAPSHOT, d]
            awhere = "a.ticker = $1 AND a.snapshot = $2 AND a.trade_date < $3"
            if start:
                aargs.append(start)
                awhere += f" AND a.trade_date >= ${len(aargs)}"
            atm_band = await conn.fetch(
                f"SELECT a.dte AS x, {_band_aggs('a.atm_iv', 'b')} "
                f"FROM {ATM_TABLE} a WHERE {awhere} "
                f"GROUP BY a.dte ORDER BY a.dte",
                *aargs,
            )

    def pack(rows, keyed):
        out = []
        for r in rows:
            item = {"x": int(r["x"]),
                    "v": None if r["v"] is None else float(r["v"])}
            if "strike" in r.keys() and r["strike"] is not None:
                item["strike"] = float(r["strike"])
            if "extrapolated" in r.keys():
                item["extrap"] = bool(r["extrapolated"])
            if keyed:
                item["series"] = int(r["series"])
            out.append(item)
        return out

    def packband(rows, keyed):
        out = []
        for r in rows:
            item = {"x": int(r["x"]), **_band_of(r, "b")}
            if keyed:
                item["series"] = int(r["series"])
            out.append(item)
        return out

    keyed = kind == "term"
    return {
        "ticker": ticker, "date": str(d), "snapshot": snap,
        "kind": kind, "window": window, "axis": axis,
        "today": pack(today, keyed),
        "band":  packband(band, keyed),
        "atm_today": [{"x": int(r["x"]),
                       "v": None if r["v"] is None else float(r["v"])}
                      for r in atm_today],
        "atm_band":  [{"x": int(r["x"]), **_band_of(r, "b")} for r in atm_band],
        "exclude_extrapolated": bool(exclude_extrapolated),
        "basis": {"snapshot": BASELINE_SNAPSHOT, "through": "prior session",
                  "from": _jsonable(start)},
        "available": ax,
    }


# ── Row 6: the tent ─────────────────────────────────────────────────────────

TENT_CANDIDATES = {
    "zc_width":  ("zc_width_sigma_{t}d",),
    "zc_delta":  ("zc_short_delta_{t}d",),
    # The 25-delta LONG leg's sigma. Specced and pre-wired, not yet backfilled
    # — _first_live returns None until it lands and the ghost tent stays dark,
    # then lights up on its own with no code change. Several candidate names
    # because the metrics project owns the final one.
    #
    # It has to be STORED rather than derived here, and not only on the
    # general principle. The ghost's long position is a historical MEDIAN, so
    # deriving it would mean re-reading the surface and re-deriving the
    # convention on every prior session in the window — the same second
    # implementation that put the short marker 0.13 sigma off its own band,
    # replicated ~250 times instead of once.
    "long_sigma": ("long_sigma_{t}d", "zc_long_sigma_{t}d",
                   "long_width_sigma_{t}d"),
    "zc_cost":   ("zc_cost_{t}d", "cost_zero_cost_{t}d"),
    "dn_cost":   ("cost_at_delta_neutral_{t}d", "cost_at_delta_neutral"),
    "dn_width":  ("dn_width_sigma_{t}d", "delta_neutral_width_sigma_{t}d"),
    "dn_delta":  ("dn_short_delta_{t}d", "delta_neutral_short_delta_{t}d"),
}


def _tent_cols(cat, dte):
    """Resolve the tent's metric columns for one tenor, or None per slot.

    Names are templated on the tenor and probed against what equity_metrics
    actually has, rather than hardcoded — the same approach the ticker header
    takes, and for the same reason: a wrong guess renders as a confidently
    empty panel, and `resolved` in the payload says which name answered.
    """
    return {key: _first_live(cat, *(p.format(t=dte) for p in pats))
            for key, pats in TENT_CANDIDATES.items()}


@router.get("/tent")
async def tent(
    ticker:               str  = Query(...),
    dte:                  int  = Query(30),
    long_delta:           int  = Query(25, description="the long leg's put_delta"),
    date:                 str  = Query(None),
    snapshot:             str  = Query(None),
    window:               str  = Query("1y"),
    exclude_extrapolated: bool = Query(True),
    pool=Depends(get_oi_pool),
):
    """The 1x2 put ratio, drawn on a sigma axis, against its own history.

    Long one put at `long_delta`, short two at the zero-cost strike. The panel
    answers "how far out is the market paying me to go today, versus how far
    out it usually pays", so the historical distribution is of the WIDTH —
    zc_width_sigma — not of the payoff.

    Sigma rather than strike or delta because it is the only axis on which
    today's structure and a year of prior ones can share a picture: a strike
    axis moves with spot, and a delta axis compresses exactly where the
    interesting part of the trade is. Width in ATM-implied-move units is
    comparable across dates and across names.

    `cost_at_delta_neutral` rides alongside because the GAP between the
    zero-cost point and the delta-neutral point is the skew reading in the
    units the entry decision actually uses. Steep skew pushes zero-cost
    further out than delta-neutral; flat skew pulls it closer in. Either
    number alone does not say which regime you are in.

    There is ONE source for the sigma axis: the stored zc_width_sigma. Both
    the band and the short marker come from it, so they cannot disagree.

    This module used to derive its own copy of the short's sigma as a
    defensive check, on the reasoning that it could not see the loader's
    convention. That was the wrong response to not knowing — the derivation
    was wrong twice over (linear moneyness where the loader is logarithmic,
    and a delta-node snap where the loader solves), so the panel drew a
    correct band with a wrong marker on it. Two implementations of one
    definition drift; the fix is one implementation, not a check between two.
    See sigma_basis in the payload for how the long leg is placed without
    reintroducing a second one.
    """
    if not pool:
        return {"error": "OI database not configured"}

    cat = await _catalog(pool)
    ax  = await _axes(pool)
    use_dte = _nearest(ax["dtes"], dte)
    if use_dte is None:
        return {"error": "equity_surface has no tenors"}
    cols = _tent_cols(cat, use_dte)

    async with pool.acquire() as conn:
        d, snap = await _resolve_slice(conn, date, snapshot)
        start = _window_start(d, window)

        sel = ['m."{}" AS {}'.format(c, k) for k, c in cols.items() if c]
        today_m = None
        if sel:
            today_m = await conn.fetchrow(
                f"SELECT {', '.join(sel)} FROM {METRICS_TABLE} m "
                f"WHERE m.ticker = $1 AND m.trade_date = $2 AND m.snapshot = $3",
                ticker, d, snap,
            )

        # Both legs' own history, on the page's scoring rule: daily closes,
        # ending at the prior session. One pass — the two share a WHERE, and
        # the ghost needs their medians on the same set of sessions or it is
        # drawing a structure that never existed on any single day.
        band_sel = []
        if cols["zc_width"]:
            band_sel.append(_band_aggs('m."{}"'.format(cols["zc_width"]), "b"))
        if cols["long_sigma"]:
            band_sel.append(_band_aggs('m."{}"'.format(cols["long_sigma"]), "l"))
        band_row = None
        if band_sel:
            bargs  = [ticker, BASELINE_SNAPSHOT, d]
            bwhere = "m.ticker = $1 AND m.snapshot = $2 AND m.trade_date < $3"
            if start:
                bargs.append(start)
                bwhere += f" AND m.trade_date >= ${len(bargs)}"
            band_row = await conn.fetchrow(
                f"SELECT {', '.join(band_sel)} "
                f"FROM {METRICS_TABLE} m WHERE {bwhere}",
                *bargs,
            )

        # Spot and the ATM vol that defines the sigma unit.
        anchor = await conn.fetchrow(
            f"SELECT a.underlying_price AS spot, a.atm_iv, a.atm_strike "
            f"FROM {ATM_TABLE} a "
            f"WHERE a.ticker = $1 AND a.trade_date = $2 AND a.snapshot = $3 "
            f"  AND a.dte = $4",
            ticker, d, snap, use_dte,
        )
        # The metric layer's spot as well. The header shows THAT one, this
        # panel prices off equity_atm, and if the two disagree the page shows
        # a spot beside strikes that were computed against a different one —
        # which is exactly the kind of mismatch that looks like nothing.
        metric_spot = await conn.fetchval(
            f"SELECT m.spot FROM {METRICS_TABLE} m "
            f"WHERE m.ticker = $1 AND m.trade_date = $2 AND m.snapshot = $3",
            ticker, d, snap,
        )

        # The whole delta grid for this tenor. Both legs are interpolated off
        # it rather than snapped to a node — see leg_at().
        legs = await conn.fetch(
            f"SELECT s.put_delta, s.strike, s.iv, s.price, s.extrapolated "
            f"FROM {SURFACE_TABLE} s "
            f"WHERE s.ticker = $1 AND s.trade_date = $2 AND s.snapshot = $3 "
            f"  AND s.dte = $4 ORDER BY s.put_delta",
            ticker, d, snap, use_dte,
        )

    by_delta = {int(r["put_delta"]): r for r in legs}
    spot   = None if anchor is None or anchor["spot"] is None else float(anchor["spot"])
    atm_iv = None if anchor is None or anchor["atm_iv"] is None else float(anchor["atm_iv"])

    # ── Legs, by interpolation ──────────────────────────────────────────────
    # The delta grid is 5 apart and the zero-cost short lands wherever the
    # solve put it — MSTR's was 14.22. Snapping that to node 15 moved the
    # strike from 84.84 to 85.61, and the OI overlay ("is my short sitting on
    # a wall") inherits the wrong strike along with it. So the stored
    # fractional delta is interpolated between its bracketing nodes instead.
    nodes   = sorted(by_delta)
    strikes = [None if by_delta[n]["strike"] is None else float(by_delta[n]["strike"]) for n in nodes]
    ivs     = [None if by_delta[n]["iv"]     is None else float(by_delta[n]["iv"])     for n in nodes]
    prices  = [None if by_delta[n]["price"]  is None else float(by_delta[n]["price"])  for n in nodes]

    def leg_at(dl):
        """One leg at a (possibly fractional) put delta, off the surface grid."""
        if dl is None or len(nodes) < 2:
            return None
        k = _interp(nodes, strikes, dl)
        if k is None:
            return None
        lo = max((n for n in nodes if n <= dl), default=None)
        hi = min((n for n in nodes if n >= dl), default=None)
        # Fabricated if EITHER bracketing node was: interpolating between a
        # real node and an invented one gives a partly invented answer, and
        # calling that clean is how the flag stops meaning anything.
        fab = any(bool(by_delta[n]["extrapolated"])
                  for n in (lo, hi) if n is not None)
        return {"put_delta": dl, "strike": k,
                "iv":     _interp(nodes, ivs, dl),
                "price":  _interp(nodes, prices, dl),
                "extrap": fab,
                "between": [lo, hi]}

    def m(key):
        if today_m is None or not cols.get(key) or today_m[key] is None:
            return None
        return float(today_m[key])

    stored_width = m("zc_width")
    stored_delta = m("zc_delta")

    long_leg  = leg_at(float(long_delta))
    short_leg = leg_at(stored_delta)

    # ── The sigma axis: ONE source ──────────────────────────────────────────
    # zc_width_sigma is the loader's own answer for how far out the short
    # sits, so it is used verbatim. An earlier version derived a second copy
    # here and got it wrong twice over — linear moneyness where the loader is
    # logarithmic, and a snapped strike where the loader solves — which drew a
    # correct band with a wrong marker on the same axis. That looks entirely
    # normal on screen and invalidates precisely the inside-or-outside-the-band
    # judgement the panel exists to support.
    #
    # The long leg still needs a position on that axis and has no stored sigma
    # of its own. Rather than reimplement the definition to place it, the
    # scale is CALIBRATED from the stored pair: one anchor at (short strike,
    # -zc_width_sigma), the origin at (spot, 0). Verified against the data —
    # ln(K/spot) / (atm_iv * sqrt(dte/365)) reproduces the stored width to
    # four decimals across the universe — but nothing here depends on that
    # formula. If the loader changes its denominator the scale absorbs it,
    # because it is read off the loader's output instead of reproduced from
    # its arithmetic. The only assumption left is that sigma is proportional
    # to log-moneyness, which every lognormal convention satisfies.
    sigma_scale = None
    if (stored_width is not None and short_leg and short_leg["strike"]
            and spot and short_leg["strike"] > 0 and spot > 0):
        lm = math.log(short_leg["strike"] / spot)
        if lm:
            sigma_scale = -abs(stored_width) / lm

    def sigma_of(strike):
        if sigma_scale is None or not strike or not spot or strike <= 0:
            return None
        return sigma_scale * math.log(strike / spot)

    if short_leg:
        short_leg["sigma"] = sigma_of(short_leg["strike"])

    # The long leg prefers its own stored sigma and falls back to the
    # calibrated scale only while long_sigma_{t}d is unbackfilled. Both are
    # on the same axis and the same sign convention (stored is positive and
    # increasing with distance; the axis is negative below spot).
    stored_long = m("long_sigma")
    long_source = None
    if long_leg:
        if stored_long is not None:
            long_leg["sigma"] = -abs(stored_long)
            long_source = "stored"
        else:
            long_leg["sigma"] = sigma_of(long_leg["strike"])
            long_source = "calibrated" if long_leg["sigma"] is not None else None

    band      = _band_of(band_row, "b") if (cols["zc_width"] and band_row is not None) else None
    long_band = _band_of(band_row, "l") if (cols["long_sigma"] and band_row is not None) else None

    # ── The ghost: the average structure, from stored medians ───────────────
    # Today's tent alone cannot say WHICH leg moved. The long's sigma is a
    # function of the smile's slope between ATM and 25 delta, so it travels
    # with skew exactly as the short does, off a different segment of the same
    # curve — measured across the universe it moves about half as much as the
    # short and is 0.78 correlated with it. So a narrow-looking tent can be the
    # short coming in, the long going out, or the pair sliding toward spot, and
    # the short's band cannot separate those. The median structure drawn behind
    # today's makes them three visibly different pictures.
    ghost = None
    ghost_reason = None
    if band is None or band.get("p50") is None:
        ghost_reason = "no history for the short leg in this window"
    elif long_band is None:
        ghost_reason = ("long_sigma_{t}d is not in equity_metrics yet — the "
                        "ghost needs the long leg's sigma on every prior "
                        "session, which is a stored column, not something this "
                        "panel should derive").format(t=use_dte)
    elif long_band.get("p50") is None:
        ghost_reason = "no history for the long leg in this window"
    else:
        ghost = {
            "long_sigma":  -abs(long_band["p50"]),
            "short_sigma": -abs(band["p50"]),
            "n": min(band.get("n") or 0, long_band.get("n") or 0),
        }

    sigma_basis = {
        "source":     cols["zc_width"],
        "value":      stored_width,
        "long_source": long_source,
        "long_column": cols["long_sigma"],
        "long_value":  stored_long,
        "scale_per_log_moneyness": sigma_scale,
        "note": ("sigma is the stored metric, not a derivation. The short leg "
                 "sits at -zc_width_sigma by construction. The long leg reads "
                 "long_sigma_{t}d when it exists; until that backfill lands it "
                 "is placed on a scale calibrated from the stored short pair, "
                 "which is exact for today but cannot produce a historical "
                 "median — hence no ghost until the column ships."),
    }

    # Two invariants that must hold for the drawn structure to be a put ratio
    # at all. Neither is decoration: if a strike sits above spot the panel is
    # drawing something that is not the trade, and every downstream consumer
    # of these strikes — the OI overlay especially — inherits it.
    mspot = None if metric_spot is None else float(metric_spot)
    problems = []
    for nm, lg in (("long", long_leg), ("short", short_leg)):
        if lg and lg["strike"] is not None and spot and lg["strike"] > spot:
            problems.append(
                f"{nm} strike {lg['strike']:.2f} is ABOVE spot {spot:.2f}; a put "
                f"ratio's strikes must be below it")
    if (long_leg and short_leg and long_leg["strike"] is not None
            and short_leg["strike"] is not None
            and short_leg["strike"] >= long_leg["strike"]):
        problems.append(
            f"short strike {short_leg['strike']:.2f} is not below the long "
            f"{long_leg['strike']:.2f}; the 1x2 is inverted")
    # zc_width_sigma is DEFINED for a 25-delta-long 1x2. Move the long leg and
    # the stored width — and therefore the band, and the short marker — is
    # describing a different structure from the one drawn.
    if int(long_delta) != 25:
        problems.append(
            f"long leg is {int(long_delta)}-delta, but zc_width_sigma is defined "
            f"for a 25-delta long; the band and the short marker describe a "
            f"25-delta structure, not the one drawn")
    if spot and mspot and abs(spot / mspot - 1.0) > 0.005:
        problems.append(
            f"equity_atm.underlying_price is {spot:.2f} but equity_metrics.spot "
            f"is {mspot:.2f} — a {abs(spot / mspot - 1.0) * 100:.1f}% gap. The "
            f"header shows the second; every strike here is priced against the "
            f"first, so they are not on the same basis.")

    return {
        "ticker": ticker, "date": str(d), "snapshot": snap,
        "dte": use_dte, "window": window,
        "spot": spot, "atm_iv": atm_iv,
        "metric_spot": mspot,
        "geometry_problems": problems,
        "long_leg": long_leg, "short_leg": short_leg,
        "ratio": {"long": 1, "short": 2},
        "zc_width_sigma": stored_width,
        "zc_short_delta": stored_delta,
        "zc_cost":        m("zc_cost"),
        "dn_cost":        m("dn_cost"),
        "dn_width_sigma": m("dn_width"),
        "dn_short_delta": m("dn_delta"),
        "band":      band,
        "long_band": long_band,
        "ghost":     ghost,
        "ghost_unavailable_because": ghost_reason,
        "sigma_basis": sigma_basis,
        "resolved": cols,
        "exclude_extrapolated": bool(exclude_extrapolated),
        "basis": {"snapshot": BASELINE_SNAPSHOT, "through": "prior session",
                  "from": _jsonable(start)},
        "available": ax,
    }


# ── Row 7: sticky-strike decomposition ──────────────────────────────────────

def _interp(xs, ys, x):
    """Linear interpolation on a sorted x grid. None outside the range.

    Deliberately does NOT extrapolate. This is used to re-read one session's
    smile at another session's strikes, and beyond the fitted domain the only
    honest answer is "that strike was not quoted" — inventing one here would
    reproduce, in the client, exactly the fabrication the `extrapolated` flag
    exists to expose.
    """
    if x is None or len(xs) < 2 or x < xs[0] or x > xs[-1]:
        return None
    lo, hi = 0, len(xs) - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if xs[mid] <= x:
            lo = mid
        else:
            hi = mid
    span = xs[hi] - xs[lo]
    if span == 0:
        return ys[lo]
    t = (x - xs[lo]) / span
    return ys[lo] + (ys[hi] - ys[lo]) * t


@router.get("/sticky-strike")
async def sticky_strike(
    ticker:               str  = Query(...),
    dte:                  int  = Query(30),
    date:                 str  = Query(None),
    snapshot:             str  = Query(None),
    prev_date:            str  = Query(None, description="default: prior session"),
    exclude_extrapolated: bool = Query(True),
    pool=Depends(get_oi_pool),
):
    """Sticky-strike vs sticky-delta, decomposed. Three lines in strike space.

      1  the prior session's smile, as it was
      2  today's smile
      3  the prior session's smile RE-READ AT TODAY'S SPOT

    plus the residual, line 2 minus line 3.

    Why this panel exists, and why it is the one that matters most for this
    trading style. If spot drops 1.5% and the surface does not reprice at all,
    the 25-delta strike is now a DIFFERENT strike, and 25d-10d skew reads
    differently for purely mechanical reasons. Line 3 isolates that migration;
    the residual is the actual repricing.

    Skew that is rich because flow bid the wings mean-reverts on its own. Skew
    that is rich because spot moved only reverts if spot bounces — which is a
    directional bet wearing a volatility trade's clothes. Nothing else on the
    page separates those two, and they call for opposite decisions.

    Line 3 is built by moneyness: for today's strike K, the sticky-delta
    reading is the prior smile evaluated at K * S_prev / S_today, which is the
    strike that had the same moneyness then. Interpolated linearly across the
    prior session's fitted nodes, and left NULL past their range.
    """
    if not pool:
        return {"error": "OI database not configured"}

    ax = await _axes(pool)
    use_dte = _nearest(ax["dtes"], dte)
    if use_dte is None:
        return {"error": "equity_surface has no tenors"}

    async with pool.acquire() as conn:
        d, snap = await _resolve_slice(conn, date, snapshot)

        if prev_date:
            try:
                prev = date_type.fromisoformat(prev_date)
            except ValueError:
                raise HTTPException(400, f"Invalid prev_date: {prev_date!r}")
        else:
            # The prior session that actually has a daily close, which is not
            # necessarily yesterday — holidays and gaps in capture both exist.
            prev = await conn.fetchval(
                f"SELECT max(trade_date) FROM {ATM_TABLE} "
                f"WHERE ticker = $1 AND snapshot = $2 AND trade_date < $3",
                ticker, BASELINE_SNAPSHOT, d,
            )
        if prev is None:
            return {"ticker": ticker, "date": str(d), "error":
                    f"No session before {d} for {ticker}", "today": [],
                    "prev": [], "shifted": []}

        keep = " AND NOT s.extrapolated" if exclude_extrapolated else ""
        rows_today = await conn.fetch(
            f"SELECT s.put_delta, s.strike, s.iv, s.extrapolated "
            f"FROM {SURFACE_TABLE} s "
            f"WHERE s.ticker = $1 AND s.trade_date = $2 AND s.snapshot = $3 "
            f"  AND s.dte = $4{keep} ORDER BY s.strike",
            ticker, d, snap, use_dte,
        )
        rows_prev = await conn.fetch(
            f"SELECT s.put_delta, s.strike, s.iv, s.extrapolated "
            f"FROM {SURFACE_TABLE} s "
            f"WHERE s.ticker = $1 AND s.trade_date = $2 AND s.snapshot = $3 "
            f"  AND s.dte = $4{keep} ORDER BY s.strike",
            ticker, prev, BASELINE_SNAPSHOT, use_dte,
        )
        spots = await conn.fetch(
            f"SELECT a.trade_date, a.underlying_price AS spot, a.atm_iv "
            f"FROM {ATM_TABLE} a "
            f"WHERE a.ticker = $1 AND a.dte = $2 "
            f"  AND ((a.trade_date = $3 AND a.snapshot = $4) "
            f"    OR (a.trade_date = $5 AND a.snapshot = $6))",
            ticker, use_dte, d, snap, prev, BASELINE_SNAPSHOT,
        )

    by_date = {r["trade_date"]: r for r in spots}
    s_now  = by_date.get(d)
    s_prev = by_date.get(prev)
    spot_now  = None if not s_now  or s_now["spot"]  is None else float(s_now["spot"])
    spot_prev = None if not s_prev or s_prev["spot"] is None else float(s_prev["spot"])

    def pack(rows):
        return [{"put_delta": int(r["put_delta"]),
                 "strike": None if r["strike"] is None else float(r["strike"]),
                 "iv":     None if r["iv"] is None else float(r["iv"]),
                 "extrap": bool(r["extrapolated"])} for r in rows]

    today_pts = pack(rows_today)
    prev_pts  = pack(rows_prev)

    # The prior smile as an interpolatable grid, sorted and de-duplicated.
    grid = sorted({p["strike"]: p["iv"] for p in prev_pts
                   if p["strike"] is not None and p["iv"] is not None}.items())
    xs = [k for k, _ in grid]
    ys = [v for _, v in grid]

    ratio = (spot_prev / spot_now) if (spot_prev and spot_now) else None
    shifted, residual = [], []
    for p in today_pts:
        k = p["strike"]
        iv3 = _interp(xs, ys, k * ratio) if (k is not None and ratio) else None
        shifted.append({"strike": k, "iv": iv3, "put_delta": p["put_delta"]})
        residual.append({
            "strike": k, "put_delta": p["put_delta"],
            "v": (None if (iv3 is None or p["iv"] is None) else p["iv"] - iv3),
        })

    n_res = sum(1 for r in residual if r["v"] is not None)
    return {
        "ticker": ticker, "date": str(d), "snapshot": snap,
        "prev_date": str(prev), "prev_snapshot": BASELINE_SNAPSHOT,
        "dte": use_dte,
        "spot": spot_now, "prev_spot": spot_prev,
        "spot_return": (None if not (spot_now and spot_prev)
                        else (spot_now / spot_prev) - 1.0),
        "today": today_pts,
        "prev": prev_pts,
        "shifted": shifted,
        "residual": residual,
        "n_residual": n_res,
        # Strikes past the prior session's fitted range have no sticky-delta
        # reading and so no residual. Said out loud, because a residual line
        # that stops short otherwise looks like a rendering fault.
        "n_out_of_domain": len(today_pts) - n_res,
        "exclude_extrapolated": bool(exclude_extrapolated),
        "available": ax,
    }


# ── Row 7: the surface grid ─────────────────────────────────────────────────

GRID_VIEWS = ("iv", "z_iv", "iv_minus_atm", "z_iv_minus_atm", "chg_1d", "chg_5d")


@router.get("/surface-grid")
async def surface_grid(
    ticker:               str  = Query(...),
    view:                 str  = Query("iv_minus_atm"),
    date:                 str  = Query(None),
    snapshot:             str  = Query(None),
    window:               str  = Query("1y"),
    exclude_extrapolated: bool = Query(True),
    pool=Depends(get_oi_pool),
):
    """DTE rows x delta columns, read straight off equity_surface.

    views:
      iv              the raw surface
      z_iv            z of the raw surface
      iv_minus_atm    the SHAPE view, and the default (labelled "Skew")
      z_iv_minus_atm  z of that spread
      chg_1d          today minus the prior session, per node
      chg_5d          today minus five sessions back, per node

    iv_minus_atm is the default because z-scoring a cell's own IV makes the
    grid light up together whenever vol is high or low — every cell moves with
    the level, so that view is largely an expensive ATM IV readout. Scoring the
    SPREAD to ATM removes the level and leaves shape.

    z_iv is offered anyway, on request: "is the whole surface rich" is a real
    question, and the answer to it being a uniformly-lit grid is informative
    once you know that is what the view does. The two sit side by side so the
    comparison is one click.

    The ATM anchor is equity_atm at k=0, never put_delta 50. They are not the
    same node and the difference is exactly the quantity being measured.

    The z window follows the page rule: prior sessions at the daily close, and
    a cell whose window is thinner than BASELINE_MIN_N returns null rather
    than a confident number off a handful of observations.
    """
    if not pool:
        return {"error": "OI database not configured", "cells": []}
    if view not in GRID_VIEWS:
        raise HTTPException(400, f"view must be one of {', '.join(GRID_VIEWS)}")

    ax = await _axes(pool)
    keep = " AND NOT s.extrapolated" if exclude_extrapolated else ""

    async with pool.acquire() as conn:
        d, snap = await _resolve_slice(conn, date, snapshot)
        start = _window_start(d, window)

        join_atm = (f"JOIN {ATM_TABLE} a ON a.ticker = s.ticker "
                    f"AND a.trade_date = s.trade_date "
                    f"AND a.snapshot = s.snapshot AND a.dte = s.dte")

        rows = await conn.fetch(
            f"SELECT s.dte, s.put_delta, s.iv, s.strike, s.extrapolated, "
            f"       s.dte_actual, a.atm_iv, a.underlying_price AS spot "
            f"FROM {SURFACE_TABLE} s {join_atm} "
            f"WHERE s.ticker = $1 AND s.trade_date = $2 AND s.snapshot = $3 "
            f"ORDER BY s.dte, s.put_delta",
            ticker, d, snap,
        )

        stats = {}
        if view in ("z_iv", "z_iv_minus_atm"):
            scored = "s.iv" if view == "z_iv" else "(s.iv - a.atm_iv)"
            sargs  = [ticker, BASELINE_SNAPSHOT, d]
            swhere = "s.ticker = $1 AND s.snapshot = $2 AND s.trade_date < $3"
            if start:
                sargs.append(start)
                swhere += f" AND s.trade_date >= ${len(sargs)}"
            srows = await conn.fetch(
                f"SELECT s.dte, s.put_delta, "
                f"       avg({scored}) AS mu, "
                f"       stddev_samp({scored}) AS sd, "
                f"       count({scored}) AS n "
                f"FROM {SURFACE_TABLE} s {join_atm} "
                f"WHERE {swhere}{keep} "
                f"GROUP BY s.dte, s.put_delta",
                *sargs,
            )
            stats = {(int(r["dte"]), int(r["put_delta"])):
                     (r["mu"], r["sd"], int(r["n"] or 0)) for r in srows}

        prior = {}
        if view in ("chg_1d", "chg_5d"):
            back = 1 if view == "chg_1d" else 5
            ref = await conn.fetchval(
                f"SELECT trade_date FROM ("
                f"  SELECT DISTINCT trade_date FROM {ATM_TABLE} "
                f"  WHERE ticker = $1 AND snapshot = $2 AND trade_date < $3 "
                f"  ORDER BY trade_date DESC LIMIT $4"
                f") t ORDER BY trade_date ASC LIMIT 1",
                ticker, BASELINE_SNAPSHOT, d, back,
            )
            if ref is not None:
                prows = await conn.fetch(
                    f"SELECT s.dte, s.put_delta, s.iv FROM {SURFACE_TABLE} s "
                    f"WHERE s.ticker = $1 AND s.trade_date = $2 AND s.snapshot = $3",
                    ticker, ref, BASELINE_SNAPSHOT,
                )
                prior = {(int(r["dte"]), int(r["put_delta"])): r["iv"]
                         for r in prows}
            ref_date = ref
        else:
            ref_date = None

    cells, n_thin, actual = [], 0, {}
    for r in rows:
        key = (int(r["dte"]), int(r["put_delta"]))
        iv  = None if r["iv"] is None else float(r["iv"])
        atm = None if r["atm_iv"] is None else float(r["atm_iv"])
        ex  = bool(r["extrapolated"])

        if view == "iv":
            v = iv
        elif view == "iv_minus_atm":
            v = None if (iv is None or atm is None) else iv - atm
        elif view in ("z_iv", "z_iv_minus_atm"):
            mu, sd, n = stats.get(key, (None, None, 0))
            spread = iv if view == "z_iv" else (
                None if (iv is None or atm is None) else iv - atm)
            if spread is None or mu is None or not sd or n < BASELINE_MIN_N:
                v = None
                if spread is not None and n and n < BASELINE_MIN_N:
                    n_thin += 1
            else:
                v = (spread - float(mu)) / float(sd)
        else:
            p = prior.get(key)
            v = None if (iv is None or p is None) else iv - float(p)

        cells.append({
            "dte": key[0], "put_delta": key[1], "v": v, "iv": iv, "atm_iv": atm,
            "strike": None if r["strike"] is None else float(r["strike"]),
            "extrap": ex,
        })
        if r["dte_actual"] is not None:
            actual.setdefault(key[0], float(r["dte_actual"]))

    # Per-tenor row metadata. `dte_actual` is the node's TRUE tenor, and a
    # row where it differs from the target tenor was read off a listed expiry
    # rather than blended from the two that bracket it. That distinction is
    # invisible in the values and shows up as a step in the column -- AAPL's
    # 7d row sitting below both 5d and 10d, for instance -- which reads as a
    # data fault unless the grid says which rows are which.
    #
    # This reports the SIGNATURE, from stored data. The fit itself is built in
    # the separate Open_Interest project, so this module cannot confirm the
    # rule that produced it, only that the row carries a real expiry's tenor.
    row_meta = []
    for t in ax["dtes"]:
        a = actual.get(t)
        row_meta.append({
            "dte": t, "dte_actual": a,
            "direct": (a is not None and abs(a - t) > 0.01),
        })

    return {
        "ticker": ticker, "date": str(d), "snapshot": snap,
        "view": view, "window": window,
        "dtes": ax["dtes"], "deltas": ax["deltas"],
        "rows": row_meta,
        "cells": cells,
        "reference_date": _jsonable(ref_date),
        "n_thin_baseline": n_thin,
        "exclude_extrapolated": bool(exclude_extrapolated),
        "basis": {"snapshot": BASELINE_SNAPSHOT, "through": "prior session",
                  "from": _jsonable(start), "min_n": BASELINE_MIN_N},
    }


# ── Row 8: time-scatter ─────────────────────────────────────────────────────

@router.get("/time-scatter")
async def time_scatter(
    ticker:               str  = Query(...),
    x:                    str  = Query(...),
    y:                    str  = Query(...),
    date:                 str  = Query(None),
    snapshot:             str  = Query(None),
    window:               str  = Query("1y"),
    exclude_extrapolated: bool = Query(True),
    pool=Depends(get_oi_pool),
):
    """One dot per prior session for THIS ticker, on the global scatter's axes.

    The global scatter answers "which ticker is unusual". This answers "is
    this ticker on its usual trajectory", which two separate line charts
    cannot: a hysteresis loop — skew rising as IV falls, then both retracing
    by a different route — is plainly visible as a loop here and completely
    invisible as two lines against time.

    Dots carry their date so the client can gradient them by age. Today's dot
    comes from the selected snapshot and is flagged, so the live reading is
    placed on the path rather than mixed into it.

    A z axis reads equity_metrics_z, which carries a z per (ticker,
    trade_date, snapshot) scored against that ticker's 1545 daily series. The
    per-date rolling window this endpoint used to build is exactly what that
    column now stores, so the axis is a column read.
    """
    if not pool:
        return {"error": "OI database not configured", "points": []}

    cat = await _catalog(pool)
    ex_, ey = _entry(cat, x), _entry(cat, y)
    needs = _needs_z([ex_, ey])

    def axis(e):
        col = _expr(e)
        if exclude_extrapolated:
            col = f"CASE WHEN {_extrap_expr(e)} THEN NULL ELSE {col} END"
        return col

    async with pool.acquire() as conn:
        d_, snap = await _resolve_slice(conn, date, snapshot)
        start = _window_start(d_, window)

        # History: daily closes only, so the path is one observation per
        # session rather than a cloud thickened by intraday buckets.
        params = [ticker, BASELINE_SNAPSHOT, d_]
        where  = "m.ticker = $1 AND m.snapshot = $2 AND m.trade_date < $3"
        if start is not None:
            params.append(start)
            where += f" AND m.trade_date >= ${len(params)}"

        rows = await conn.fetch(
            f"SELECT m.trade_date, {axis(ex_)} AS xv, {axis(ey)} AS yv "
            f"{_from_clause(needs)} WHERE {where} "
            f"ORDER BY m.trade_date",
            *params,
        )

        # Today, at the selected snapshot, read exactly the same way.
        trow = await conn.fetchrow(
            f"SELECT {axis(ex_)} AS xv, {axis(ey)} AS yv "
            f"{_from_clause(needs)} "
            f"WHERE m.ticker = $1 AND m.trade_date = $2 AND m.snapshot = $3",
            ticker, d_, snap,
        )

    points = [{"date": str(r["trade_date"]),
               "x": None if r["xv"] is None else float(r["xv"]),
               "y": None if r["yv"] is None else float(r["yv"]),
               "today": False}
              for r in rows]
    tx = None if trow is None or trow["xv"] is None else float(trow["xv"])
    ty = None if trow is None or trow["yv"] is None else float(trow["yv"])
    if tx is not None or ty is not None:
        points.append({"date": str(d_), "x": tx, "y": ty, "today": True})

    return {
        "ticker": ticker, "date": str(d_), "snapshot": snap, "window": window,
        "x": _meta(ex_), "y": _meta(ey),
        "points": points,
        "n_points": sum(1 for p in points if p["x"] is not None and p["y"] is not None),
        "z_source": "stored" if needs else None,
        "exclude_extrapolated": bool(exclude_extrapolated),
    }


# ── Row 8: spot-vol scatter ─────────────────────────────────────────────────

@router.get("/spot-vol")
async def spot_vol(
    ticker:   str = Query(...),
    dte:      int = Query(30),
    date:     str = Query(None),
    snapshot: str = Query(None),
    window:   str = Query("1y"),
    pool=Depends(get_oi_pool),
):
    """Change in ATM IV against underlying return. The cloud behind the beta.

    beta and R-squared are already stored (spotvol_beta_1m/3m, spotvol_r2_1m/3m)
    and shown as header chips. This panel exists because the SHAPE is not in
    those two numbers: non-linearity, a regime break part-way through the
    window, or three outliers carrying the whole fit all produce a perfectly
    respectable-looking beta. A summary statistic cannot tell you it is
    describing two different regimes averaged together.

    Both series come from equity_atm at one tenor — atm_iv and
    underlying_price, the only place spot is stored — as consecutive daily
    closes, so a point is one overnight move. Today's point uses the selected
    snapshot against the prior close, which makes it a partial-session move
    and is flagged as such: it is placed on the cloud, not counted in the fit.
    """
    if not pool:
        return {"error": "OI database not configured", "points": []}

    ax = await _axes(pool)
    use_dte = _nearest(ax["dtes"], dte)
    if use_dte is None:
        return {"error": "equity_surface has no tenors", "points": []}

    async with pool.acquire() as conn:
        d, snap = await _resolve_slice(conn, date, snapshot)
        start = _window_start(d, window)

        args  = [ticker, BASELINE_SNAPSHOT, use_dte, d]
        where = ("a.ticker = $1 AND a.snapshot = $2 AND a.dte = $3 "
                 "AND a.trade_date <= $4")
        if start:
            args.append(start)
            where += f" AND a.trade_date >= ${len(args)}"
        hist = await conn.fetch(
            f"SELECT a.trade_date, a.atm_iv, a.underlying_price AS spot "
            f"FROM {ATM_TABLE} a WHERE {where} ORDER BY a.trade_date",
            *args,
        )
        live = await conn.fetchrow(
            f"SELECT a.trade_date, a.atm_iv, a.underlying_price AS spot "
            f"FROM {ATM_TABLE} a "
            f"WHERE a.ticker = $1 AND a.trade_date = $2 AND a.snapshot = $3 "
            f"  AND a.dte = $4",
            ticker, d, snap, use_dte,
        )

    seq = [(r["trade_date"], r["atm_iv"], r["spot"]) for r in hist
           if r["atm_iv"] is not None and r["spot"] is not None]

    import math
    points = []
    for (pd_, piv, ps), (cd, civ, cs) in zip(seq, seq[1:]):
        if not ps or not cs:
            continue
        points.append({"date": str(cd),
                       "ret": math.log(float(cs) / float(ps)),
                       "d_iv": float(civ) - float(piv),
                       "partial": False})

    # Today, against the last settled close. Flagged, and excluded from the
    # fit: a part-session move regressed against overnight moves is a
    # different measurement wearing the same axes.
    if live is not None and live["atm_iv"] is not None and live["spot"] is not None:
        settled = [s for s in seq if s[0] < d]
        if settled:
            _, piv, ps = settled[-1]
            if ps:
                points.append({"date": str(d),
                               "ret": math.log(float(live["spot"]) / float(ps)),
                               "d_iv": float(live["atm_iv"]) - float(piv),
                               "partial": True})

    fit_pts = [p for p in points if not p["partial"]]
    beta = alpha = r2 = None
    n = len(fit_pts)
    if n >= 3:
        mx = sum(p["ret"] for p in fit_pts) / n
        my = sum(p["d_iv"] for p in fit_pts) / n
        sxx = sum((p["ret"] - mx) ** 2 for p in fit_pts)
        sxy = sum((p["ret"] - mx) * (p["d_iv"] - my) for p in fit_pts)
        syy = sum((p["d_iv"] - my) ** 2 for p in fit_pts)
        if sxx > 0:
            beta  = sxy / sxx
            alpha = my - beta * mx
            r2    = (sxy * sxy) / (sxx * syy) if syy > 0 else None

    return {
        "ticker": ticker, "date": str(d), "snapshot": snap,
        "dte": use_dte, "window": window,
        "points": points,
        "n_fit": n,
        "fit": {"beta": beta, "alpha": alpha, "r2": r2,
                "excludes_today": True},
        "basis": {"snapshot": BASELINE_SNAPSHOT,
                  "note": "overnight moves between consecutive daily closes"},
        "available": ax,
    }
