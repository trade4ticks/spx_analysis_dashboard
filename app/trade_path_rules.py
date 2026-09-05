"""Exit-rule combine SQL for the trade_paths table.

VENDORED from Open_Interest/lib/trade_path_rules.py.
Source of truth lives there — keep in sync. Same arrangement as
app/split_factors.py.

  first vendored  2026-08-15
  re-synced       2026-09-05, against source commit c3fbd56
                  (per-horizon resolution filter + the backstop-length guard)

The re-sync exists because the source moved and nothing noticed: the copy sat
on `WHERE path_status = 'ok'` for as long as it took someone to read both
files side by side. scripts/check_vendored.py covers the VERBATIM copies and
cannot cover this one while it is trimmed — see that script's docstring.

What was copied
---------------
`build_combine_sql` (verbatim logic), plus the two constants it depends on:
`SIDE_PRIORITY` and `HORIZON_RULE_KEY`.

What was deliberately NOT copied
--------------------------------
The source's in-process rule REGISTRY (a list of ~59 `Rule` dataclasses built
at import time) and everything that computes exits from price paths. This
project never computes exits — it reads the precomputed columns. It also
already has the registry available as data: the `trade_path_rules` catalog
table, which the source writes precisely so "the dashboard builds column
names from data rather than hardcoding 118 of them".

So the one signature change from the original: the source resolves rule
metadata from its module-global `BY_KEY`, and this takes an explicit
`by_key` mapping supplied by the caller from that table. Everything else —
the horizon backstop, tie-break ordering, the resolution filter, the emitted
SQL shape — is unchanged.

RuleMeta therefore carries `family` and `params` as well as the columns: the
backstop-length guard mirrored from c3fbd56 compares a selected max_days
rule's `n` against the backstop's, and the source reads those off its own
Rule dataclass.

LOCAL ADDITION, not present upstream
------------------------------------
`include_exit_rule=True` emits one extra column, `exit_rule`, naming which
rule won. Default False, so by default the emitted SQL is byte-identical to
the source's.

It exists because the exit-reason breakdown has to tell a user-selected
`max_days` apart from the auto-appended backstop. Those can be the SAME
column with opposite meanings: exits on `max_days__20` are the user's policy
working when they picked it, and their stops and targets failing to fire when
they did not. The distinction is `horizon_auto_added`, which the caller
already gets in `meta` — but only if it also knows which rule fired per row,
which the source's CASE (returning the return, not the rule) cannot say.

Deriving it outside this function would mean re-deriving the winner, i.e.
duplicating the combine — the exact thing vendoring is meant to prevent. So
the attribution CASE is emitted here, from the same `ordered` list, and is
guaranteed to agree with the return CASE on ties by construction.

Worth pushing upstream on the next sync.

Column names come from the table's `exit_bar_col` / `exit_return_col` and are
never constructed from the rule key.
"""
from __future__ import annotations

import json
from typing import Any, Mapping

# Stop before target before trend before time, so a same-bar tie resolves to
# the stop — the documented same-bar convention that a stop is assumed to
# have fired first.
SIDE_PRIORITY: dict[str, int] = {"stop": 0, "target": 1, "trend": 2, "time": 3}

# The structural backstop appended to every combine. See build_combine_sql.
HORIZON_RULE_KEY = "max_days__20"


class CombineError(ValueError):
    pass


class RuleMeta:
    """Minimal stand-in for the source's `Rule` dataclass.

    The fields build_combine_sql reads. Built from a trade_path_rules row;
    `bar_col`/`ret_col` map to the table's exit_bar_col/exit_return_col.

    `family` and `params` were added when the horizon guard was mirrored from
    upstream (see build_combine_sql): the guard compares a selected max_days
    rule's `n` against the backstop's, which the previous four fields could
    not express. They default to None/{} so any caller constructing a RuleMeta
    positionally with four arguments keeps working.
    """

    __slots__ = ("key", "side", "bar_col", "ret_col", "family", "params")

    def __init__(self, key: str, side: str, bar_col: str, ret_col: str,
                 family: str = None, params: Any = None):
        self.key = key
        self.side = side
        self.bar_col = bar_col
        self.ret_col = ret_col
        self.family = family
        self.params = _coerce_params(params)


def _coerce_params(params: Any) -> dict:
    """Catalog `params` as a dict, from either JSONB shape.

    asyncpg hands back a parsed dict or a JSON string depending on the codec
    in play, and the horizon guard reads a number out of it. An unparseable
    value degrades to {} rather than raising: the guard then simply does not
    fire for that rule, which is the same behaviour as before it existed.
    """
    if isinstance(params, str):
        try:
            params = json.loads(params)
        except (TypeError, ValueError):
            return {}
    return params if isinstance(params, dict) else {}


def _max_days_n(meta) -> "int | None":
    """Sessions for a max_days rule, or None if this is not one.

    The horizon guard needs to compare holding lengths, and `family` is what
    says a rule is a time rule at all -- the rule KEY is not consulted, per
    this module's standing rule that the key's encoding is a naming artefact.
    """
    if meta is None or getattr(meta, "family", None) != "max_days":
        return None
    n = (meta.params or {}).get("n")
    try:
        return int(n)
    except (TypeError, ValueError):
        return None


def by_key_from_rows(rows) -> dict[str, RuleMeta]:
    """{rule_key: RuleMeta} from trade_path_rules rows (asyncpg Records or dicts)."""
    out: dict[str, RuleMeta] = {}
    for r in rows:
        key = r["rule_key"]
        out[key] = RuleMeta(key, r["side"], r["exit_bar_col"], r["exit_return_col"],
                            r["family"], r["params"])
    return out


def build_combine_sql(rule_keys, by_key: Mapping[str, RuleMeta],
                      table: str = "trade_paths",
                      include_unresolved: bool = False,
                      include_exit_rule: bool = False) -> tuple[str, dict[str, Any]]:
    """SQL selecting the winning exit across `rule_keys`, plus metadata.

    THE HORIZON BACKSTOP IS STRUCTURAL, NOT A CONVENTION.

    Postgres's LEAST ignores NULLs, and a NULL exit_bar means "this rule never
    fired". So a policy whose stops all miss would yield LEAST(NULL, NULL) =
    NULL — a trade that never exits, which surfaces downstream as a
    plausible-looking return rather than an error. That is the failure mode
    this function exists to make impossible.

    HORIZON_RULE_KEY is therefore appended to every combine unconditionally.
    It is a no-op whenever any selected rule fires earlier (LEAST picks the
    smaller), and it is the guaranteed exit when none does. There is no code
    path through this function that produces an unbounded exit.

    Note that max_days rules other than the backstop are ordinary selectable
    policy — the backstop is a floor underneath whatever the user picks, not
    a substitute for picking.

    Ties are broken by side priority — stop, then target, then trend, then
    time — implementing the documented same-bar convention that a stop is
    assumed to have fired first.

    Unresolved paths are excluded by default. Those are entries whose horizon
    extends past the end of available data; their horizon exit is genuinely
    unknown, and including them would silently mix "not yet resolved" into
    realised statistics.

    The filter is PER-HORIZON, on the backstop rule's own exit_bar column, not
    on the table-wide `path_status` flag -- see the comment on `where` below
    for why that distinction is load-bearing the moment the catalog grows a
    longer horizon.
    """
    if not rule_keys:
        raise CombineError(
            "no rules selected: a combine with no rules has no exit at all. "
            "Select at least one rule; the horizon backstop "
            f"({HORIZON_RULE_KEY}) is added automatically but is not a policy."
        )
    unknown = [k for k in rule_keys if k not in by_key]
    if unknown:
        raise CombineError(
            f"unknown rule key(s): {unknown}. Valid keys come from the "
            f"trade_path_rules table ({len(by_key)} rules)."
        )
    if HORIZON_RULE_KEY not in by_key:
        raise CombineError(
            f"the horizon backstop {HORIZON_RULE_KEY!r} is missing from "
            "trade_path_rules. Without it a combine has no guaranteed exit; "
            "refusing to build unbounded SQL."
        )

    keys = list(dict.fromkeys(rule_keys))
    horizon_added = HORIZON_RULE_KEY not in keys
    if horizon_added:
        keys.append(HORIZON_RULE_KEY)

    unknown_sides = sorted({by_key[k].side for k in keys} - set(SIDE_PRIORITY))
    if unknown_sides:
        raise CombineError(
            f"rule(s) carry unknown side(s) {unknown_sides}; tie-break order "
            f"is undefined. Known sides: {sorted(SIDE_PRIORITY)}."
        )

    # MIRRORED from the source (c3fbd56). The backstop is the rule GUARANTEED
    # to fire, and the resolution filter below is written against its column,
    # so it must never be shorter than a selected time rule. If it were, the
    # damage would not stop at the filter: LEAST would return the backstop's
    # earlier bar and silently truncate the selection to the backstop's
    # horizon, answering a different question than the one asked. Unreachable
    # while max_days__20 is the longest rule in the catalog -- it becomes
    # reachable the moment a longer horizon is added, which is exactly when it
    # must fail loudly instead of quietly.
    #
    # Reads family/params off RuleMeta, which the dashboard fills from the
    # catalog table's own columns. A rule whose params do not carry a numeric
    # `n` is skipped rather than guessed at.
    h_n = _max_days_n(by_key.get(HORIZON_RULE_KEY))
    if h_n is not None:
        longer = [k for k in keys
                  if (_max_days_n(by_key[k]) or 0) > h_n]
        if longer:
            raise CombineError(
                f"selected time rule(s) {longer} run past the horizon backstop "
                f"{HORIZON_RULE_KEY} ({h_n} sessions). LEAST would truncate them "
                f"to the backstop's exit, and the resolution filter would "
                f"understate the sessions they need. Make the backstop a function "
                f"of the selection before adding horizons longer than {h_n}."
            )

    ordered = sorted(keys, key=lambda k: SIDE_PRIORITY[by_key[k].side])

    bar_cols = [by_key[k].bar_col for k in keys]
    least = "LEAST(" + ", ".join(bar_cols) + ")"

    cases = "\n".join(
        f"        WHEN {by_key[k].bar_col} = w.exit_bar THEN {by_key[k].ret_col}"
        for k in ordered
    )
    # MIRRORED from the source (c3fbd56). Resolution is per-rule data, not a
    # table-wide flag.
    #
    # path_status is a GLOBAL boolean, stamped by build_trade_paths against the
    # build's single longest horizon (MAX_HORIZON_SESSIONS). Filtering on it
    # makes EVERY combine inherit that longest horizon's tail truncation: a
    # one-session policy is denied the same trailing entries as a twenty-
    # session one, for a reason that has nothing to do with the one session it
    # actually needs. Extending the catalog's horizon would therefore silently
    # shrink the eligible population of every existing combination -- including
    # the short ones -- which is the failure this filter exists to prevent.
    #
    # The horizon rule's own exit_bar already carries the same fact per
    # horizon: it is NULL exactly when that rule's final session was not
    # reachable in the available data.
    #
    # EQUIVALENT to path_status = 'ok' while max_days__20 is the longest rule
    # in the catalog, and deliberately so -- build_trade_paths stamps `full`
    # from (si + H - 1) <= last_session, the same predicate that sets
    # sess_end_rel[:, H-1] to NEVER for the horizon rule. The source's test 13
    # verifies that equivalence against the build's own arithmetic.
    #
    # The column is DERIVED from the catalog, never spelled out: the source
    # reads BY_KEY[HORIZON_RULE_KEY].bar_col and this reads the same field off
    # the row the catalog table supplied. Writing 'xb_max_days__20' here would
    # be the one thing this module's docstring forbids -- a column name
    # constructed from a rule key rather than read from the table.
    backstop_col = by_key[HORIZON_RULE_KEY].bar_col
    where = ("" if include_unresolved
             else f"\n    WHERE {backstop_col} IS NOT NULL")

    # LOCAL ADDITION (see module docstring). Same `ordered` list as the return
    # CASE above, so the two cannot disagree about which rule won a tie.
    rule_case = ""
    if include_exit_rule:
        rule_whens = "\n".join(
            f"        WHEN {by_key[k].bar_col} = w.exit_bar THEN '{k}'"
            for k in ordered
        )
        rule_case = f",\n    CASE\n{rule_whens}\n    END AS exit_rule"

    sql = (
        f"WITH w AS (\n"
        f"    SELECT ticker, trade_date, entry_anchor,\n"
        f"           {least} AS exit_bar,\n"
        f"           {', '.join(bar_cols + [by_key[k].ret_col for k in keys])}\n"
        f"    FROM {table}{where}\n"
        f")\n"
        f"SELECT ticker, trade_date, entry_anchor, exit_bar,\n"
        f"    CASE\n{cases}\n"
        f"    END AS exit_return{rule_case}\n"
        f"FROM w"
    )

    meta = {
        "rules": keys,
        "horizon_rule": HORIZON_RULE_KEY,
        "horizon_auto_added": horizon_added,
        "tie_break_order": [by_key[k].side for k in ordered],
        "excludes_unresolved": not include_unresolved,
        # MIRRORED from the source (c3fbd56). Names the rule the resolution
        # filter is written against, and the column it resolved to. Both are
        # derived, so a reader can see WHICH horizon defined "resolved" for
        # this combine rather than having to know the backstop by heart.
        "backstop_rule": HORIZON_RULE_KEY,
        "resolution_column": backstop_col,
    }
    return sql, meta
