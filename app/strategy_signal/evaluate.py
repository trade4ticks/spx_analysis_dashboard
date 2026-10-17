"""The Strategy Signal decision. Pure: values in, a state out, no database.

THE RULE. A strategy has an ordered list of states, most favourable first
(TRADE, NO TRADE; or FULL, REDUCED, MINIMAL, NONE). Every signal condition
carries ONE THRESHOLD PER STATE BUT THE LAST: "pctile < 30 / 50 / 70" reads
FULL below 30, REDUCED below 50, MINIMAL below 70. The strategy walks the
states in order and takes the first whose conditions hold -- all of them
(AND) or any of them (OR), the strategy's one choice -- and falls back to the
last state when none does. A binary strategy is the same rule with one
threshold. There is no nesting and no per-state logic: one relationship,
one comparison per condition per level.

THE SCHEDULE IS NOT A CONDITION (changed 2026-10-16). A strategy's
scheduled entry days and times say when it is USUALLY considered; they never
make it NO TRADE. A Wednesday strategy whose conditions fail on Wednesday and
pass on Thursday reads TRADE on Thursday, because entering a day late on
better conditions is a decision the trader makes, not the page. The weekday
was once the first test here (`weekday AND conditions`); it is now reported
as `scheduled_today`, information only, live and on a past date alike.

MISSING IS NOT FALSE. A condition whose metric has no current value is
unknown, and unknowns combine the three-valued way (an AND with a false is
false whatever else is unknown; an OR with a true is true). A level whose
answer is unknown stops the walk and the strategy reads NO DATA: falling
through to a lower level would quietly turn "the data is missing" into
"reduce", which is a decision nobody made.

Manual requirements never enter this. They are shown, not evaluated.
"""
from __future__ import annotations

CMPS = ("<", "<=", ">", ">=", "=")
LOGICS = ("and", "or")
WEEKDAYS = (1, 2, 3, 4, 5)          # ISO: Monday = 1
WEEKDAY_NAMES = {1: "Mon", 2: "Tue", 3: "Wed", 4: "Thu", 5: "Fri", 6: "Sat", 7: "Sun"}


def compare(value, cmp: str, threshold: float) -> bool | None:
    if value is None:
        return None
    if cmp == "<":
        return value < threshold
    if cmp == "<=":
        return value <= threshold
    if cmp == ">":
        return value > threshold
    if cmp == ">=":
        return value >= threshold
    if cmp == "=":
        return abs(value - threshold) <= 1e-9 * max(1.0, abs(threshold))
    raise ValueError(f"unknown comparison {cmp!r}")


def combine(results: list, logic: str) -> bool | None:
    if logic == "and":
        if any(r is False for r in results):
            return False
        return None if any(r is None for r in results) else True
    if any(r is True for r in results):
        return True
    return None if any(r is None for r in results) else False


def decide(cfg: dict, values: dict, iso_weekday: int | None = None) -> dict:
    """{state (index | None), reason, levels, conditions, scheduled_today}.

    `values` maps a metric id to its current value (None when missing).
    reason: "conditions"    decided by the conditions
            "no_conditions" nothing to test -> the first state
            "no_data"       a level could not be answered -> state None
    `iso_weekday` only answers `scheduled_today`; it never moves the state.
    Every automatic requirement belongs in `conditions` -- a later kind of
    condition (a priced structure, say) joins that list and the same
    three-valued walk, so missing data stays NO DATA rather than TRADE.
    """
    states = cfg["states"]
    last = len(states) - 1
    signal = [m for m in cfg["metrics"] if m.get("signal")]
    conds = [{"metric": m["id"], "cmp": m["cmp"], "value": values.get(m["id"]),
              "passes": [compare(values.get(m["id"]), m["cmp"], t) for t in m["thresholds"]]}
             for m in signal]
    out = {"conditions": conds, "levels": [], "weekday": iso_weekday,
           "scheduled_today": iso_weekday in (cfg.get("weekdays") or [])}
    state, reason = 0, "no_conditions"
    if conds:
        state, reason = last, "conditions"
        for i in range(last):
            r = combine([c["passes"][i] for c in conds], cfg["logic"])
            out["levels"].append(r)
            if r is None:
                state, reason = None, "no_data"
                break
            if r:
                state = i
                break
    # market_state/market_reason were "what the conditions alone say" while
    # the weekday could override them; they are now the decision itself,
    # kept so an older page reading them still reads the right thing.
    return {**out, "state": state, "reason": reason, "market_state": state, "market_reason": reason}
