"""Structure presets: which metrics matter for a given trade.

A structure is NOT a panel. It is a preset over controls that already exist —
tenor, rails, scanner columns and filters, scatter axes. A dedicated
per-structure panel would duplicate the tent, sticky-strike and rails into a
second place and need rebuilding for every new structure; a preset needs a
config entry.

This module is deliberately data plus one resolver, and lives outside the
router, because a later step builds a structured brief from a preset's metric
list to hand to an LLM. That brief has to read the SAME list the rails and the
scanner do — a second hardcoded copy is the thing this file exists to prevent.

WHY THE RESOLVER IS HERE AND NOT ONLY IN THE CLIENT
---------------------------------------------------
The page retargets columns as the tenor moves, and does it in the browser
because a round trip per click is not acceptable. So the rule exists in
JavaScript. Putting a second copy here would be exactly the duplication that
produced two divergent z estimators earlier in this project.

The compromise, and its guard: the ALGORITHM is written twice, once per
runtime, but its DATA — the alias map and the tenor-pair map — is defined here
only and shipped to the client through /catalog. And
scripts/check_tenor_retarget.py runs both implementations over the same
catalog and asserts they return identical answers for every case, so a
divergence cannot be committed.
"""
from __future__ import annotations

import re

from app.metrics_config import TENORS

# ── Renames, so a preset written before a migration resolves after it ───────
#
# TRANSITIONAL. RV_WINDOWS, RET_WINDOWS and SPOTVOL_WINDOWS became tenor-derived
# upstream, turning the 1w/1m/3m labels into the {t}d grid. Both directions are
# listed so a page works against a database on either side of the migration.
# The catalog is always the arbiter: an alias is consulted only when the written
# name is ABSENT and the alias is PRESENT, so it can never shadow a live column.
#
# Delete once the rename has run everywhere.
COLUMN_ALIASES = {
    "vrp_1m": "vrp_30d", "vrp_1w": "vrp_7d", "vrp_3m": "vrp_90d",
    "vrp_ratio_1m": "vrp_ratio_30d", "vrp_ratio_1w": "vrp_ratio_7d",
    "vrp_ratio_3m": "vrp_ratio_90d",
    "rv_1m": "rv_30d", "rv_1w": "rv_7d", "rv_3m": "rv_90d",
    "log_ret_1w": "log_ret_7d", "log_ret_1m": "log_ret_30d",
    "spotvol_beta_1m": "spotvol_beta_30d_1m",
    "spotvol_r2_1m": "spotvol_r2_30d_1m",
    "spotvol_beta_3m": "spotvol_beta_30d_3m",
    "spotvol_r2_3m": "spotvol_r2_30d_3m",
    # ...and the reverse.
    "vrp_30d": "vrp_1m", "vrp_7d": "vrp_1w", "vrp_90d": "vrp_3m",
    "rv_30d": "rv_1m", "rv_7d": "rv_1w", "rv_90d": "rv_3m",
    "log_ret_7d": "log_ret_1w", "log_ret_30d": "log_ret_1m",
    "spotvol_beta_30d_1m": "spotvol_beta_1m",
    "spotvol_r2_30d_1m": "spotvol_r2_1m",
    "spotvol_beta_30d_3m": "spotvol_beta_3m",
    "spotvol_r2_30d_3m": "spotvol_r2_3m",
}

# ── Pair families ──────────────────────────────────────────────────────────
#
# term_ratio and term_slope name a PAIR of tenors, so "the term ratio at tenor
# 21" is not defined and the token swap cannot apply. Pinning them is not right
# either: 30/90 read beside a 7-day structure quietly answers a different
# question than the rest of the row. So they map.
#
# 7 -> 7/30 rather than 7/14 because two very short fits sitting close together
# are both noisy and the contango signal is clearer over a wider gap. 21 shares
# 14/30 because the pair set has no 21-day member.
PAIR_FAMILIES = {"term_ratio", "term_slope"}
PAIR_FOR_TENOR = {7: (7, 30), 14: (14, 30), 21: (14, 30),
                  30: (30, 90), 60: (30, 90), 90: (30, 90)}
_PAIR_RE = re.compile(r"_(\d+)d_(\d+)d")


def alias_if_missing(by_col: dict, col: str) -> str:
    """The written name, or its alias when only the alias exists."""
    if col in by_col:
        return col
    alt = COLUMN_ALIASES.get(col)
    return alt if alt and alt in by_col else col


def retarget(by_col: dict, col: str, tenor: int) -> str:
    """Move a metric column to another tenor. Mirrors the client's retarget().

    Pair families map; everything else swaps the tenor token in the NAME and
    verifies the candidate against the catalog. Keying on (family, wing, tenor)
    would be ambiguous — zc_width_sigma_21d and zc_short_delta_21d share it —
    so the stem carries the identity.

    A column that cannot move is returned unchanged rather than blanked.
    """
    m = by_col.get(col)
    if not m:
        return col

    if m.get("family") in PAIR_FAMILIES:
        want = PAIR_FOR_TENOR.get(tenor)
        if not want:
            return col
        cand = _PAIR_RE.sub(f"_{want[0]}d_{want[1]}d", col, count=1)
        hit = by_col.get(cand)
        if not hit or hit.get("family") != m.get("family") or hit.get("form") != m.get("form"):
            return col
        return cand

    t = m.get("tenor")
    if t is None or t == tenor:
        return col
    token = f"{t}d"
    if token not in col:
        return col
    cand = col.replace(token, f"{tenor}d", 1)
    hit = by_col.get(cand)
    if not hit or hit.get("family") != m.get("family") or hit.get("form") != m.get("form"):
        return col
    return cand


def resolve_column(by_col: dict, col: str, tenor: int) -> str:
    """Alias first, THEN retarget.

    Order matters and is not interchangeable: retarget() returns its input
    unchanged for anything it cannot resolve, which is indistinguishable from
    "already correct". A stale name reaching it stays stale — which is exactly
    how two preset buttons ended up setting one axis and silently keeping the
    other.
    """
    return retarget(by_col, alias_if_missing(by_col, col), tenor)


# ── Built-in structures ────────────────────────────────────────────────────
#
# Adding one is a config entry here, not a code change anywhere.
#
# Column names are written at the preset's own tenor and RESOLVED against the
# catalog on the way out, so a rename upstream surfaces as a named unresolved
# column rather than as a control that quietly does nothing.

BUILTIN_STRUCTURES = [
    {
        "key": "put_ratio",
        "name": "Put Ratio",
        "note": (
            "1x2 put ratio — long one 25-delta put, short two further out, "
            "entered at or near zero cost, held 5-20 days, non-directional. "
            "The preset answers one question: is this name a candidate today."
        ),
        "tenor": 21,
        "rails": [
            # The trade-native reading first: how much cushion the zero-cost
            # structure gives today against how much it usually gives.
            {"b": "zc_width_sigma_21d",  "w": 63},
            {"b": "skew_21d_25p_atm",    "w": 63},   # belly-to-25d slope
            {"b": "skew_21d_10p_25p",    "w": 63},   # the segment actually sold
            {"b": "term_ratio_14d_30d",  "w": 63},   # the regime filter
            {"b": "vrp_21d",             "w": 63},   # a premium to collect at all
            {"b": "spotvol_beta_21d_1m", "w": 63},   # down-move hit on short vega
            # Seven rather than six. Short-dated IV carries event and pin noise
            # that spot does not explain, so R^2 falls as the tenor shortens —
            # a large beta with a weak R^2 is a wide estimate rather than a
            # strong relationship, and beta is the number most likely to look
            # dramatic.
            {"b": "spotvol_r2_21d_1m",   "w": 63},
        ],
        "scanner_columns": [
            {"b": "zc_width_sigma_21d",  "w": 63,    "lock": False},
            {"b": "skew_21d_25p_atm",    "w": 63,    "lock": False},
            {"b": "skew_21d_10p_25p",    "w": 63,    "lock": False},
            {"b": "term_ratio_14d_30d",  "w": "val", "lock": False},
            {"b": "vrp_21d",             "w": "val", "lock": False},
            {"b": "spotvol_beta_21d_1m", "w": "val", "lock": False},
            {"b": "days_to_earnings",    "w": "val", "lock": False},
            {"b": "log_ret_21d",         "w": "val", "lock": False},
            {"b": "spot",                "w": "val", "lock": False},
        ],
        "scanner_filters": [
            {"b": "term_ratio_14d_30d", "op": "lt", "v": "1.0"},
            # nullorgt, NOT gt: an ETF has no earnings date, and a plain
            # "> 20" on NULL is false in SQL, so the whole ETF universe would
            # be filtered out by a test that is meant to exclude events.
            {"b": "days_to_earnings",   "op": "nullorgt", "v": "20"},
            {"b": "log_ret_21d",        "op": "gte", "v": "0"},
        ],
        "scanner_sort": {"b": "zc_width_sigma_21d", "w": 63, "dir": "desc"},
        # "Is skew rich because spot fell?" — the adverse-selection check for
        # this structure, and the reason the pair is worth a preset.
        "scatter_x": {"b": "skew_21d_25p_atm", "z": True},
        "scatter_y": {"b": "log_ret_21d",      "z": False},
    },
]


def _spec_cols(preset: dict):
    """Every column name a preset names, with where it sits."""
    for i, r in enumerate(preset.get("rails") or []):
        yield ("rails", i, r)
    for i, c in enumerate(preset.get("scanner_columns") or []):
        yield ("scanner_columns", i, c)
    for i, f in enumerate(preset.get("scanner_filters") or []):
        yield ("scanner_filters", i, f)
    if preset.get("scanner_sort"):
        yield ("scanner_sort", 0, preset["scanner_sort"])
    for k in ("scatter_x", "scatter_y"):
        if preset.get(k):
            yield (k, 0, preset[k])


def resolve_preset(by_col: dict, preset: dict, tenor: int | None = None) -> dict:
    """A preset with every column resolved against the catalog.

    Returns a copy with `b` rewritten and an `unresolved` list naming anything
    the catalog does not have. Unresolved columns are REPORTED, never dropped:
    a preset silently one column short is a preset that looks like it worked.

    `tenor` overrides the preset's own, so a preset is a starting point rather
    than a fixed snapshot — the same rule the page tenor uses moves every {t}d
    name in it.
    """
    t = int(tenor if tenor is not None else preset.get("tenor") or 30)
    if t not in TENORS:
        t = min(TENORS, key=lambda x: abs(x - t))

    out = dict(preset)
    out["tenor"] = t
    unresolved = []

    for where, i, spec in _spec_cols(preset):
        col = resolve_column(by_col, spec["b"], t)
        if col not in by_col:
            unresolved.append({"where": where, "index": i, "column": spec["b"]})
        # Rebuild the containing structure rather than mutating the input:
        # BUILTIN_STRUCTURES is module state and a request must not edit it.
        if where in ("rails", "scanner_columns", "scanner_filters"):
            lst = [dict(x) for x in out[where]]
            lst[i] = dict(spec, b=col)
            out[where] = lst
        else:
            out[where] = dict(spec, b=col)

    out["unresolved"] = unresolved
    return out
