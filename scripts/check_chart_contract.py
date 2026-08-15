#!/usr/bin/env python3
"""Gate: does every consumer of window.FactorCharts satisfy its contract?

Why this exists
---------------
Extracting the shared chart renderers was verified by a round-trip check that
proved each moved body was byte-identical to the original. That check is
structurally incapable of catching a MISSING DEPENDENCY -- identical code
that calls something the new receiver does not have is still identical code.
It reported 13/13 while Factor Trades threw
"cmp._equityModeKey is not a function" on first render.

This gate checks the thing that check could not: that the module never
depends on a method the receiver might lack, and that every component using
it supplies the state it reads.

Three assertions:

  1. The module makes ZERO method calls on its receiver. Intra-module calls
     must route through window.FactorCharts, not through `cmp` -- otherwise
     the receiver silently becomes part of the API surface and every new
     consumer has to re-implement helpers.

  2. Every state field the module reads off `cmp` is declared by each
     consumer component.

  3. Every bare function the shared Jinja macros call in an Alpine
     expression (hmCellBg, _hmCellTitle, ...) exists on each component that
     renders that macro -- the same failure class, arriving via the template
     instead of the module.

Usage
-----
    python scripts/check_chart_contract.py

Exit 0 = every consumer satisfies the contract. Exit 1 = something is
missing, named, per consumer.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODULE = ROOT / "static/js/factor_charts.js"
MACROS = ROOT / "templates/_macros.html"

# component file -> template that renders the heatmap macro (None = does not)
CONSUMERS = {
    ROOT / "static/js/oi_analysis.js":  ROOT / "templates/oi_analysis.html",
    ROOT / "static/js/factor_trades.js": ROOT / "templates/factor_trades.html",
}


def _component_members(src: str) -> set[str]:
    """Property and method names declared on the Alpine component.

    Matches `name(` and `name:` at 4-space indent, which is how every
    component in this codebase declares them.
    """
    out = set(re.findall(r"^    (_?[A-Za-z][\w]*)\s*\(", src, re.M))
    # ES getters/setters are declarations too. Without this a component that
    # exposes a derived field as `get name()` reads as not declaring it at
    # all -- a false failure that would push someone to add a redundant
    # plain field alongside the getter.
    out |= set(re.findall(r"^    (?:get|set)\s+(_?[A-Za-z][\w]*)\s*\(", src, re.M))
    # Declarations are not always alone on a line -- `ticker: '', metric: '',`
    # is one line declaring three fields -- so scan every 4-space-indented
    # line for name: pairs rather than anchoring to line start.
    for line in src.splitlines():
        if line.startswith("    ") and not line.startswith("     "):
            out |= set(re.findall(r"(?<![\w.$])(_?[A-Za-z][\w]*)\s*:", line))
    return out


def main() -> int:
    mod = MODULE.read_text(encoding="utf-8")
    bad = 0

    # 1. no receiver method calls
    receiver_calls = sorted(set(re.findall(r"\bcmp\.(_?[A-Za-z][\w]*)\s*\(", mod)))
    if receiver_calls:
        print("FAIL: module calls methods on its receiver — these must route "
              "through window.FactorCharts instead:")
        for c in receiver_calls:
            print(f"    cmp.{c}(...)")
        bad += 1
    else:
        print("module -> receiver method calls: 0 (module is self-contained)")

    # 2. state contract. A field read ONLY behind a typeof guard is optional
    # by construction — the module already handles its absence — so it is not
    # required of consumers. _labDsCallsThisHover is the case in point: a
    # hover-latency diagnostic assigned dynamically and never declared.
    all_state = set(re.findall(r"\bcmp\.(_?[A-Za-z][\w]*)", mod)) - set(receiver_calls)
    optional = {f for f in re.findall(r"typeof\s+cmp\.(_?[A-Za-z][\w]*)", mod)}
    contract = sorted(all_state - optional)
    print(f"state contract: {len(contract)} required fields"
          + (f", {len(optional)} optional (typeof-guarded)" if optional else ""))

    # 3. macro-called functions
    # Only functions HARDCODED in the macro body are required of every
    # consumer. Anything arriving through cls_expr / click_expr is supplied
    # per call site, so requiring it of all consumers would be wrong --
    # Factor Trades passes toggleCell where Factor Analysis passes
    # toggleHmCell, and both are correct.
    macro_src = MACROS.read_text(encoding="utf-8")
    # Jinja comments first: the macros document their own call sites, and a
    # worked example in a comment is not a dependency. Without this, the
    # example `click_expr='toggleHmCell(ix, iy)'` would be required of every
    # consumer.
    macro_src = re.sub(r"\{#.*?#\}", "", macro_src, flags=re.S)
    macro_src = re.sub(r"\{%-?\s*if (cls_expr|click_expr).*?\{%-?\s*endif\s*-?%\}", "",
                       macro_src, flags=re.S)
    macro_calls = sorted(
        set(re.findall(r"(?<![\w.$])([_A-Za-z][\w]*)\s*\(", macro_src))
        - {"if", "for", "heatmap_grid", "stat_box", "indent", "safe", "expression",
           "var", "times", "fmtPct"})

    for comp, tpl in CONSUMERS.items():
        src = comp.read_text(encoding="utf-8")
        members = _component_members(src)
        missing_state = [f for f in contract if f not in members]
        renders_macro = tpl is not None and "heatmap_grid(" in tpl.read_text(encoding="utf-8")
        missing_fns = [f for f in macro_calls if f not in members] if renders_macro else []

        label = comp.name
        if missing_state or missing_fns:
            bad += 1
            print(f"  {label}: MISSING")
            for f in missing_state:
                print(f"      state    cmp.{f}  — module reads it; component does not declare it")
            for f in missing_fns:
                print(f"      macro fn {f}()  — heatmap_grid calls it in Alpine scope")
        else:
            extra = "  (+ heatmap macro fns)" if renders_macro else ""
            print(f"  {label}: OK — all {len(contract)} state fields{extra}")

    print()
    print("contract satisfied" if not bad else "CONTRACT VIOLATED")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
