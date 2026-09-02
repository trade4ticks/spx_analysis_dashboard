"""Which of the ~232 metrics the ranked table opens on, and how it finds them.

THE ONE PLACE METRIC NAMES ARE WRITTEN. scripts/check_scalp_metrics.py bans
metric literals from the router and the page JS; this module is the declared
exception, on the same terms app/equity_presets.py holds for the IV page:

  * every name here is a CANDIDATE, resolved against the live catalog at
    request time. A name that no longer exists does not become a column of
    nulls -- the role falls through to its next candidate, and if none
    resolves the role is dropped and SAYS SO.
  * scripts/check_scalp_metrics.py verifies every literal here against the
    vendored metric_docs, so a rename upstream fails the build rather than
    quietly emptying a column.

That distinction is the whole point. A name used as a blind lookup is the
defect; a name used as a preference that is checked before use is not.

WHY ROLES RATHER THAN A COLUMN LIST. 232 metrics is far too many to render,
and most of them are the same measurement at a different variant, horizon or
statistic. A role says what the column is FOR -- "the spread, in bps" -- and
the candidates say which metrics can play it, best first. That also means the
default set survives the calibration exercise that is about to delete most of
the noise family.

WHAT THE VARIANT SELECTOR DRIVES. Five columns, not one. Picking
`noise_bps_tw_mid_10s_rms` sets the noise column, the ratio that divides by
it, the quote coverage AT THAT HORIZON, and both halves of the move-rate
decomposition at that variant and horizon. They have to move together or the
row is incoherent: noise is measured between consecutive OBSERVED buckets, so
a 10s noise reading beside 30s coverage is comparing two different things and
looks like a comparison.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Role:
    """One column of the ranked table.

    `candidates` are metric names in preference order. `templates` are the
    same thing for the generated families, formatted with the selected
    variant and horizon -- those cannot be listed, since the horizons the
    pipeline emits are its business and not this file's.
    """
    key: str
    label: str
    units: str                       # price | cents | bps | ratio | share | rate | count
    candidates: tuple[str, ...] = ()
    templates: tuple[str, ...] = ()
    # Shown before the user has chosen anything. The rest of the catalog is
    # reachable through the column chooser.
    default: bool = True
    # Larger is better, for the colour ramp and the default sort direction.
    # None where neither direction is good in itself.
    higher_better: bool | None = None
    note: str = ""


# ── the default set ─────────────────────────────────────────────────────────
#
# This IS the brief's list, in its order, with the decomposition folded in
# beside the ratio rather than parked in a separate panel -- move_rate is the
# term noise loses to on sparse names, so it belongs where the ratio is read.
ROLES: tuple[Role, ...] = (
    Role("price", "price", "price",
         candidates=("reference_price",),
         higher_better=None,
         note="Median trade price. Also what picks the round-lot tier, which "
              "is why it is not decoration: at $330 the round lot is 40 "
              "shares, at $1,172 it is 10."),

    Role("spread_cents", "spread ¢", "cents",
         candidates=("spread_cents_tw", "spread_cents_median",
                     "spread_cents_mean"),
         higher_better=True,
         note="Duration-weighted, so a spread that stood for an hour counts "
              "more than one that stood for a tick."),

    Role("spread_bps", "spread bps", "bps",
         candidates=("spread_bps_tw", "spread_bps_median", "spread_bps_mean"),
         higher_better=True,
         note="The numerator of every ranking ratio."),

    # Filled from the variant selector.
    Role("noise", "noise bps", "bps",
         templates=("noise_bps_{v}_{h}s_{stat}", "noise_bps_{v}_{h}s"),
         higher_better=False,
         note="How far the midpoint moves between consecutive observed "
              "buckets. The denominator."),

    Role("ratio", "ratio", "ratio",
         templates=("ratio_{v}_{h}s_{stat}", "ratio_{v}_{h}s"),
         higher_better=True,
         note="Spread over noise. THE UNTESTED HYPOTHESIS the whole pipeline "
              "is organised around — that a stock moving faster than its "
              "spread cannot be scalped passively."),

    Role("coverage", "coverage", "share",
         templates=("quote_bucket_coverage_{h}s",),
         higher_better=True,
         note="Buckets holding a quote observation, over buckets in the "
              "window. NOT a diagnostic: noise is measured between "
              "consecutive OBSERVED buckets, so a name at 0.33 has "
              "observations ~30s apart at a nominal 10s horizon and its "
              "'10s noise' is really 30s noise. This is the only column that "
              "makes the one beside it comparable across names."),

    Role("move_rate", "move rate", "share",
         templates=("move_rate_{v}_{h}s",),
         higher_better=True,
         note="How OFTEN the midpoint moves at all. Noise conflates this with "
              "how far, and loses to this term on sparse names — 96 minutes "
              "for 14 round trips is a move-rate problem, not a noise one."),

    Role("move_bps", "move bps", "bps",
         templates=("move_bps_{v}_{h}s",),
         higher_better=False,
         note="How FAR the midpoint moves among the buckets that moved, with "
              "the unchanged ones taken out. The other half of noise."),

    # NOT keyed "trades_per_min": that is also a metric name, and role
    # keys share a namespace with the raw metric names the column chooser
    # adds, so a key that is also a metric makes "is this a role or a
    # column?" genuinely unanswerable at the merge.
    Role("arrivals", "trades/min", "rate",
         candidates=("trades_per_min",),
         higher_better=True,
         note="Arrivals per minute after excluded prints. Fill opportunity."),

    Role("trade_size", "size", "count",
         candidates=("trade_size_median",),
         higher_better=True,
         note="Median trade size. A 5-share print cannot fill a 50-share "
              "order, and the odd-lot share rises through the middle of the "
              "day."),

    Role("balance", "balance", "share",
         candidates=("two_sided_balance",),
         higher_better=True,
         note="min(at_bid, at_ask) / max(...). 1.0 balanced, 0.0 one-way. "
              "One-way flow is where a passive fill leaves you stuck."),

    Role("off_exchange", "off-exch", "share",
         candidates=("off_exchange_share",),
         higher_better=False,
         note="Printed to a TRF or the ADF rather than to a lit venue — "
              "volume that was never available to a resting limit order."),

    Role("shares", "shares/min", "rate",
         candidates=("shares_per_min",),
         higher_better=True,
         note="Shares per minute after excluded prints. Trades per minute "
              "counts ARRIVALS and is blind to size — fifty one-lots a minute "
              "is not fifty 200-share prints, and only the second is a book "
              "a resting order gets filled against."),

    # ── the trade-price basis, as a first-class alternative ─────────────
    #
    # Four of the five noise variants are built on QUOTE MIDPOINTS. On a thin
    # book the midpoint moves when a resting order is pulled and the next
    # level becomes the best -- one observed case moved a midpoint 19 cents in
    # 30 seconds because a 29-share bid vanished, while the stock barely
    # moved. Midpoint noise there is measuring order flicker, not price.
    #
    # `trade_price` is computed from actual fills and is immune to that. It
    # was one row of fifteen in a table; these roles make it selectable as a
    # column beside the midpoint reading, so the two can disagree visibly on a
    # per-name basis. The variant is BAKED INTO THE TEMPLATE rather than taken
    # from the selector, because the whole point is to hold it fixed while the
    # other column moves.
    Role("noise_trade", "noise (trades)", "bps",
         templates=("noise_bps_trade_price_{h}s_{stat}",
                    "noise_bps_trade_price_{h}s"),
         default=False,
         higher_better=False,
         note="Noise measured from the last TRADE PRICE in each bucket rather "
              "than the quote midpoint. Immune to a midpoint that jumps "
              "because a small resting order was pulled — which on a thin "
              "book is most of what midpoint noise measures."),

    Role("ratio_trade", "ratio (trades)", "ratio",
         templates=("ratio_trade_price_{h}s_{stat}", "ratio_trade_price_{h}s"),
         default=False,
         higher_better=True,
         note="Spread over TRADE-PRICE noise. Read beside the midpoint ratio: "
              "a name where the midpoint ratio is much the worse of the two "
              "has a flickering book rather than a moving price, and that is "
              "a different reason to avoid it — or not to."),

    # ── health only, off the ranked table by default ────────────────────
    #
    # These three are what the data-health panel watches. There is no
    # distinct-exchange-count metric and there deliberately will not be:
    # fetch.py already refuses any symbol-day under MIN_EXCHANGE_CODES, so a
    # Nasdaq-only pull never reaches compute, and a count metric would measure
    # a condition the guard prevents. What DID surface that failure was
    # trades/min reading 37% below trailing, so the panel watches the
    # quantities that move when the tape is wrong rather than one that
    # describes the tape directly.
    Role("unidentified", "unidentified venue", "share",
         candidates=("unidentified_exchange_share",),
         default=False,
         higher_better=False,
         note="Share printed on exchange code 78, which is absent from the "
              "vendor's enum. A jump means the venue mix changed under us."),
)

# The health panel's columns. `arrivals` is the one that actually caught a
# broken fetch; the two share metrics are what say whether the VENUE MIX moved,
# which is the cause rather than the symptom.
HEALTH_KEYS = ("arrivals", "off_exchange", "unidentified")

# WHICH OF THOSE ACTUALLY RAISE A FLAG.
#
# `unidentified` sits at ~0.0001 — exchange code 78 is rare by construction —
# and a percentage change on a number that small is arithmetic, not
# information: 0.0001 to 0.0006 is +516% and flagged all ten sessions. Ten of
# ten flagged is a panel saying nothing.
#
# It stays as a DISPLAY column, because its LEVEL is worth seeing: a jump to
# a few percent would mean the venue enum had gone stale. What it cannot
# support is a relative comparison against itself.
#
# arrivals is the one that caught a broken fetch. off_exchange sits around
# 0.35, where a percentage move is meaningful.
HEALTH_FLAGGING = ("arrivals", "off_exchange")

# ── derived columns ─────────────────────────────────────────────────────────
#
# The pipeline stores components, not products. Dollar volume per minute is
# shares_per_min x reference_price and neither the pipeline nor anything
# downstream had it, despite it being one of the two numbers the strategy was
# selected on BY HAND before any of this existed. Computed in the pivot, so
# it costs one multiplication and no recompute.
#
# Kept as a declared table rather than a special case in the router: the next
# derived column should be a two-line entry here, not another branch.
@dataclass(frozen=True)
class Derived:
    key: str
    label: str
    units: str
    parts: tuple[str, ...]      # metric names, in order
    op: str                     # "mul" | "div"
    higher_better: bool | None = None
    note: str = ""


DERIVED: tuple[Derived, ...] = (
    Derived("dollar_vol_per_min", "$ vol/min", "money",
            parts=("shares_per_min", "reference_price"), op="mul",
            higher_better=True,
            note="Shares per minute times the reference price. One of the two "
                 "numbers this strategy was actually selected on by hand — "
                 "spread as a percentage of price being the other — before any "
                 "of the noise work existed. A book that turns over dollars is "
                 "one a resting order gets filled in; share count alone says "
                 "nothing about that at $8 or at $1,100."),
)

BY_DERIVED = {d.key: d for d in DERIVED}
BY_KEY = {r.key: r for r in ROLES}
DEFAULT_KEYS = tuple(r.key for r in ROLES if r.default)


def literals() -> list[str]:
    """Every fixed metric name written in this file, for the build check."""
    out: list[str] = []
    for r in ROLES:
        out.extend(r.candidates)
    for d in DERIVED:
        out.extend(d.parts)
    return out


def derived_available(d: Derived, available: set) -> bool:
    """A derived column needs EVERY part. A product missing a factor is not a
    smaller number, it is not a number, so it is dropped rather than computed
    from what happens to be there."""
    return all(p in available for p in d.parts)


def describe_derived() -> list[dict]:
    return [{"key": d.key, "label": d.label, "units": d.units,
             "parts": list(d.parts), "op": d.op,
             "higher_better": d.higher_better, "note": d.note}
            for d in DERIVED]


def template_examples() -> list[str]:
    """The generated names this file's templates produce, for the same check.

    Formatted with a variant, horizon and statistic the pipeline actually
    emits, so a template whose SHAPE has drifted -- a renamed family, a moved
    suffix -- fails rather than being checked as a string with braces in it.
    """
    out: list[str] = []
    for r in ROLES:
        for t in r.templates:
            out.append(t.format(v="tw_mid", h=10, stat="rms"))
    return out


def resolve(role: Role, available: set[str], variant: str | None,
            horizon: int | None, stat: str | None) -> str | None:
    """The metric this role should read, or None if nothing can play it.

    Templates first, then fixed candidates, each in declared order. Returning
    None rather than a guess is deliberate: the caller drops the column and
    reports it, which is the behaviour a silently-empty column does not have.
    """
    if role.templates and variant and horizon:
        for t in role.templates:
            # A template mentioning {stat} cannot be filled when the selected
            # variant has none -- the unsuffixed noise family is one metric,
            # not a family of four -- so it is skipped rather than formatted
            # with the string "None".
            if "{stat}" in t and not stat:
                continue
            name = t.format(v=variant, h=horizon, stat=stat)
            if name in available:
                return name
    for c in role.candidates:
        if c in available:
            return c
    return None


def resolve_all(available: set[str], variant: str | None, horizon: int | None,
                stat: str | None, keys: list[str] | None = None) -> dict:
    """Every requested role resolved, plus the ones that could not be.

    `missing` is returned rather than logged. A column that is not there for a
    reason is information -- "this name has no 5s coverage" is a fact about
    the data -- and dropping it silently would make an absent measurement look
    like an absent metric.
    """
    want = [BY_KEY[k] for k in (keys or DEFAULT_KEYS) if k in BY_KEY]
    resolved, missing = {}, []
    for r in want:
        got = resolve(r, available, variant, horizon, stat)
        if got:
            resolved[r.key] = got
        else:
            missing.append(r.key)
    return {"columns": resolved, "missing": missing}


def describe_roles(keys: list[str] | None = None) -> list[dict]:
    """The role table as the client needs it: labels, units, direction, note."""
    want = [BY_KEY[k] for k in (keys or DEFAULT_KEYS) if k in BY_KEY]
    return [{"key": r.key, "label": r.label, "units": r.units,
             "higher_better": r.higher_better, "note": r.note,
             "from_variant": bool(r.templates)} for r in want]
