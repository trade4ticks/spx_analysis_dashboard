"""Component methods nothing calls — the residue a deleted panel leaves.

WHAT THIS EXISTS FOR. The Equities Scalp fills-upload panel was silently
deleted from its template by an editing helper whose replacement did not
preserve a region it spanned. The endpoints kept working, the JS kept its
uploadFills / fillsRows / uploadIssues methods, and the page rendered with a
calibration panel whose empty state read "upload a statement above" — above
nothing. Nobody noticed for four commits.

Every existing gate passed. check_alpine_refs verifies TEMPLATE to COMPONENT:
every call in the markup resolves to a member. Nothing checked the other
direction, and the other direction is exactly what a vanished panel leaves
behind — handlers with no caller.

THE TEST IS NEITHER, NOT JUST THE TEMPLATE. A member called only from other
JS (renderGeometry, loadSeries) is fine and must not be flagged; a member
called from neither the markup nor any other function in its own file is dead.
That distinction is what makes this precise enough to act on.

RATCHETED, like the tag-balance check. Legacy pages carry orphans that predate
this and are not worth a speculative cleanup; the counts are printed on every
run and any increase fails. A ratchet nobody sees is an exemption, so the
numbers are always shown.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEMPLATES = ROOT / "templates"
STATIC = ROOT / "static"

# Orphans that predate this gate. Recorded per page, never per name: the point
# is that the NUMBER cannot grow, and pinning individual names would turn the
# ratchet into a list nobody maintains.
BASELINE = {
    "equity_iv.html":              9,
    "factor_trades.html":          4,
    "oi_analysis.html":            26,
    "oi_signals.html":             2,
    "research.html":               3,
    "research2.html":              3,
    "ticker_analysis.html":        1,
}

# Lifecycle and framework members are called by Alpine itself or by the
# browser, never by name from either side.
EXEMPT = {
    "init", "destroy", "$nextTick", "$watch", "$refs", "$el", "$store",
    "$dispatch", "$data", "$root", "$id",
}

# Control structures at the component's indentation level look exactly like
# method definitions to a regex — `for (`, `if (`, `catch (`. Excluded by name
# rather than by parsing, for the same reason the matcher is a regex at all:
# a JS parser would be a dependency for a check whose value is being cheap
# enough to run on every commit.
KEYWORDS = {
    "for", "if", "else", "while", "do", "switch", "case", "catch", "try",
    "return", "function", "typeof", "await", "new", "delete", "throw",
    "yield", "in", "of", "instanceof",
}

MEMBER = re.compile(r"^\s{4}(?:async\s+)?([a-zA-Z_$][\w$]*)\s*\(", re.M)


def members(js: str) -> list[str]:
    """Top-level methods of the Alpine component, by indentation.

    Four spaces is the component-literal level in every file here. A JS parser
    would be more correct and would also be a dependency for a check whose
    whole value is that it is cheap enough to always run.
    """
    return [m for m in MEMBER.findall(js)
            if m not in EXEMPT and m not in KEYWORDS]


def check(tpl: Path) -> tuple[int, list[str]]:
    text = tpl.read_text(encoding="utf-8")
    js_files = [STATIC / m for m in
                re.findall(r"asset\(['\"](js/[^'\"]+)['\"]\)", text)]
    js_files = [f for f in js_files if f.is_file()]
    if not js_files:
        return 0, []

    js = "\n".join(f.read_text(encoding="utf-8") for f in js_files)
    # Markup only: a name inside an HTML comment is prose about the page, not
    # a call from it, and counting it would let a commented-out panel keep its
    # handlers alive.
    markup = re.sub(r"<!--.*?-->", "", text, flags=re.S)

    orphans = []
    for name in members(js):
        if re.search(rf"\b{re.escape(name)}\s*\(", markup):
            continue
        # Called from elsewhere in the component. `this.x(` covers the normal
        # case; a bare mention covers a method passed as a value.
        body = re.sub(rf"^\s{{4}}(?:async\s+)?{re.escape(name)}\s*\(",
                      "", js, flags=re.M)
        if re.search(rf"this\.{re.escape(name)}\b", body):
            continue
        orphans.append(name)
    return len(orphans), sorted(orphans)


def main() -> int:
    bad = 0
    known: list[str] = []
    for tpl in sorted(TEMPLATES.glob("*.html")):
        if tpl.name.startswith("_"):
            continue
        n, names = check(tpl)
        if not n:
            continue
        allowed = BASELINE.get(tpl.name, 0)
        if n > allowed:
            bad += 1
            print(f"\n  {tpl.name}: {n} component methods nothing calls, "
                  f"against a baseline of {allowed}")
            print(f"      {', '.join(names[:14])}"
                  f"{' …' if len(names) > 14 else ''}")
            print("      Either dead code, or a panel was deleted from the")
            print("      template and left its handlers behind.")
        else:
            known.append(f"      {tpl.name}: {n} (baseline {allowed})")

    if known:
        print("\n  pre-existing and ratcheted — printed every run, any "
              "increase fails:")
        for line in known:
            print(line)

    print(f"\npages checked: {len(list(TEMPLATES.glob('*.html')))}, "
          f"over baseline: {bad}")
    return 1 if bad else 0


sys.exit(main())
