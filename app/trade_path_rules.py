"""Exit-rule combine SQL for the trade_paths table.

VENDORED from Open_Interest/lib/trade_path_rules.py on 2026-08-15.
Source of truth lives there — keep in sync. Same arrangement as
app/split_factors.py.

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
the horizon backstop, tie-break ordering, path_status filtering, the emitted
SQL shape — is unchanged.

Column names come from the table's `exit_bar_col` / `exit_return_col` and are
never constructed from the rule key.
"""
from __future__ import annotations

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

    Only the three fields build_combine_sql reads. Built from a
    trade_path_rules row; `bar_col`/`ret_col` map to the table's
    exit_bar_col/exit_return_col.
    """

    __slots__ = ("key", "side", "bar_col", "ret_col")

    def __init__(self, key: str, side: str, bar_col: str, ret_col: str):
        self.key = key
        self.side = side
        self.bar_col = bar_col
        self.ret_col = ret_col


def by_key_from_rows(rows) -> dict[str, RuleMeta]:
    """{rule_key: RuleMeta} from trade_path_rules rows (asyncpg Records or dicts)."""
    out: dict[str, RuleMeta] = {}
    for r in rows:
        key = r["rule_key"]
        out[key] = RuleMeta(key, r["side"], r["exit_bar_col"], r["exit_return_col"])
    return out


def build_combine_sql(rule_keys, by_key: Mapping[str, RuleMeta],
                      table: str = "trade_paths",
                      include_unresolved: bool = False) -> tuple[str, dict[str, Any]]:
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

    Unresolved paths (path_status <> 'ok') are excluded by default. Those are
    entries whose horizon extends past the end of available data; their
    horizon exit is genuinely unknown, and including them would silently mix
    "not yet resolved" into realised statistics.
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

    ordered = sorted(keys, key=lambda k: SIDE_PRIORITY[by_key[k].side])

    bar_cols = [by_key[k].bar_col for k in keys]
    least = "LEAST(" + ", ".join(bar_cols) + ")"

    cases = "\n".join(
        f"        WHEN {by_key[k].bar_col} = w.exit_bar THEN {by_key[k].ret_col}"
        for k in ordered
    )
    where = "" if include_unresolved else "\n    WHERE path_status = 'ok'"

    sql = (
        f"WITH w AS (\n"
        f"    SELECT ticker, trade_date, entry_anchor,\n"
        f"           {least} AS exit_bar,\n"
        f"           {', '.join(bar_cols + [by_key[k].ret_col for k in keys])}\n"
        f"    FROM {table}{where}\n"
        f")\n"
        f"SELECT ticker, trade_date, entry_anchor, exit_bar,\n"
        f"    CASE\n{cases}\n"
        f"    END AS exit_return\n"
        f"FROM w"
    )

    meta = {
        "rules": keys,
        "horizon_rule": HORIZON_RULE_KEY,
        "horizon_auto_added": horizon_added,
        "tie_break_order": [by_key[k].side for k in ordered],
        "excludes_unresolved": not include_unresolved,
    }
    return sql, meta
