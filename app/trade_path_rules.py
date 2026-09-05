"""Exit-rule combine SQL for the trade_paths table.

VENDORED from Open_Interest/lib/trade_path_rules.py.
Source of truth lives there — keep in sync. Same arrangement as
app/split_factors.py.

  first vendored  2026-08-15
  re-synced       2026-09-05, against source commit c3fbd56
                  (per-horizon resolution filter + the backstop-length guard)
  re-synced       2026-09-05, against source commit 48aeb5b
                  (143-rule catalog, 40-session horizon; the backstop became
                   a function of the selection and the guard went away)

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
selection-dependent backstop mirrored from 48aeb5b picks the SHORTEST selected
max_days, and the source reads those fields off its own Rule dataclass.

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


def _is_max_days(meta) -> bool:
    """Whether this rule is a holding-period cap.

    `family` is what says so -- the rule KEY is never consulted, per this
    module's standing rule that the key's encoding is a naming artefact.

    A RuleMeta built without family (the four-argument form) answers False for
    everything, which degrades the selection-dependent backstop below to the
    fixed HORIZON_RULE_KEY -- i.e. exactly the behaviour that preceded it.
    """
    return meta is not None and getattr(meta, "family", None) == "max_days"


def _max_days_n(meta) -> "int | None":
    """Sessions for a max_days rule, or None if it does not carry a usable n."""
    if not _is_max_days(meta):
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


def backstop_for(rule_keys, by_key: Mapping[str, RuleMeta]) -> "tuple[str, bool]":
    """(backstop_rule_key, was_appended) for a selection.

    LOCAL ADDITION, not present upstream, and a pure extraction: the source
    inlines this inside build_combine_sql. It is a separate function here
    because this project has a SECOND implementation of the combine -- the
    grid's vectorised path -- and that path needs the same answer. Two copies
    of "which rule is the backstop" is precisely the drift this module exists
    to prevent, and the numpy path already transcribes enough.

    Under LEAST the shortest selected max_days decides every otherwise
    unresolved trade, so it IS the backstop and nothing is appended. With no
    max_days selected at all, HORIZON_RULE_KEY is appended and is the backstop.

    Worth pushing upstream on the next sync, alongside include_exit_rule.
    """
    time_keys = [k for k in rule_keys if _is_max_days(by_key.get(k))]

    # A max_days rule whose params carry no usable `n` cannot be compared, and
    # picking the shortest is the whole mechanism -- guessing here would write
    # the resolution filter against the wrong column and quietly change which
    # rows resolve. The dashboard fills params from the catalog table, so this
    # is a malformed catalog row, and it fails by name rather than by symptom.
    unreadable = [k for k in time_keys if _max_days_n(by_key[k]) is None]
    if unreadable:
        raise CombineError(
            f"max_days rule(s) {unreadable} carry no numeric 'n' in the "
            f"catalog, so the shortest selected horizon cannot be determined. "
            f"The resolution filter is written against that rule's column; "
            f"choosing one arbitrarily would silently change which rows count "
            f"as resolved."
        )
    if not time_keys:
        return HORIZON_RULE_KEY, True
    return min(time_keys, key=lambda k: _max_days_n(by_key[k])), False


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

A backstop is therefore present in every combine. When the selection contains
    no max_days rule at all, HORIZON_RULE_KEY is appended and is that backstop.
    When it does, the SHORTEST selected max_days already plays the role and
    nothing is appended -- under LEAST that rule is what decides every
    otherwise-unresolved trade, so adding a longer default underneath it would
    change nothing, and adding a shorter one would truncate the selection. There
    is no code path through this function that produces an unbounded exit.

    Note that max_days rules are ordinary selectable policy as well as the
    structural floor -- the two roles coincide, which is why `horizon_auto_added`
    exists to say which one applied.

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

    # MIRRORED from the source (48aeb5b). The backstop is the rule guaranteed
    # to fire, and it is a function of the SELECTION, not a constant.
    #
    # Under LEAST, the shortest max_days selected is the one that decides every
    # otherwise-unresolved trade -- a 5-session cap fires at session 5 whether
    # or not a 40 is also selected. So that rule, not the catalog's longest and
    # not a fixed default, is what the trade actually waits on, and it is what
    # the resolution filter must be written against.
    #
    # Appending a fixed max_days__20 here instead would be wrong in both
    # directions once the catalog runs past it: it would truncate a selected
    # max_days__40 to the backstop's earlier exit, and it would demand 20
    # resolvable sessions from a selection that only ever needed 5.
    #
    # This REPLACES the guard mirrored from c3fbd56, which raised when a
    # selected max_days ran past a fixed backstop. That guard's own message
    # said to make the backstop a function of the selection; this is that.
    backstop, horizon_added = backstop_for(keys, by_key)
    if horizon_added:
        keys.append(HORIZON_RULE_KEY)

    unknown_sides = sorted({by_key[k].side for k in keys} - set(SIDE_PRIORITY))
    if unknown_sides:
        raise CombineError(
            f"rule(s) carry unknown side(s) {unknown_sides}; tie-break order "
            f"is undefined. Known sides: {sorted(SIDE_PRIORITY)}."
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
    backstop_col = by_key[backstop].bar_col
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
        "backstop_rule": backstop,
        "backstop_sessions": _max_days_n(by_key[backstop]),
        "resolution_column": backstop_col,
    }
    return sql, meta
