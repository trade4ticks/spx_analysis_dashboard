"""The chart must not move while you are watching it.

WHAT HAPPENED. Placing an order made a "working" row appear above the plot;
cancelling made it vanish. Both resized the canvas — the pane is a flex
column and the plot takes what is left — so the chart jumped under the cursor
at the one moment it was being read, which is while aiming at a row to click.
The rate-limit banner did the same thing to all four panes at once.

THE RULE, stated so it can be tested rather than remembered:

    nothing that appears and disappears on its own may sit in the layout
    flow above the canvas, or at page level above the panes.

Such things go in an overlay inside the pane, or in the fixed corner stack.
The single exception is the trade bar, which appears when YOU open the ladder
or arm the pane — a deliberate act, once, not something that happens while
you watch.

This parses the template rather than the rendered page because that is where
the rule is broken: someone adds a conditional div in the obvious place, and
nothing that exists today would notice.
"""
from __future__ import annotations

import re
import sys
from html.parser import HTMLParser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TPL = ROOT / "templates" / "equities_live.html"

# The one block allowed to move the pane, because a PERSON moved it: the
# trade bar appears when you open the ladder or arm the pane.
#
# Keyed on the CONDITION, not the class. Keyed on the class, any new block
# reusing `lv-bar trade` inherited the exemption - which is what a
# regression looks like, and this check let one through until it was tested.
ALLOWED_IN_PANE_FLOW = {"pane.ladder || pane.armed"}

# Banners decided at load that then hold still all session. They are in the
# flow because a thing that never changes never reflows anything.
ALLOWED_PAGE_FLOW = {
    "isDelayed()",
    "brokerHealth && !brokerHealth.trading_enabled",
}


class Cols(HTMLParser):
    """Direct children of the pane column, and of the page body, in order."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.stack: list[dict] = []
        self.pane_children: list[dict] = []      # direct children of .lv-col
        self.page_children: list[dict] = []      # direct children of .lv-panes' parent
        self._col_depth: int | None = None
        self._page_depth: int | None = None
        self._all: list[dict] = []
        self.saw_pane = False
        self.pane_index: int | None = None
        self.toasts_fixed = False

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        cls = a.get("class", "") or ""
        rec = {"tag": tag, "cls": cls, "depth": len(self.stack),
               "show": a.get("x-show") or a.get("x-if")}
        self._all.append(rec)
        # The container that holds the panes is what page-level notices
        # share a flow with - not <body>, which holds one wrapper div.
        if "lv-panes" in cls.split() and self._page_depth is None:
            self._page_depth = len(self.stack)
            self.page_children = [r for r in self._all
                                  if r["depth"] == self._page_depth]
        # A direct child of the pane column, before the canvas container.
        if self._col_depth is not None and len(self.stack) == self._col_depth + 1:
            self.pane_children.append(rec)
            if "lv-pane" in cls.split():
                self.saw_pane = True
                self.pane_index = len(self.pane_children) - 1
        if "lv-col" in cls.split() and self._col_depth is None:
            self._col_depth = len(self.stack)
        if "lv-toasts" in cls.split():
            self.toasts_fixed = True
        if tag not in ("br", "img", "input", "hr", "meta", "link"):
            self.stack.append(rec)

    def handle_endtag(self, tag):
        while self.stack:
            top = self.stack.pop()
            if top["tag"] == tag:
                if self._col_depth is not None and len(self.stack) == self._col_depth:
                    self._col_depth = None
                break


def main() -> int:
    src = TPL.read_text(encoding="utf-8")
    p = Cols()
    p.feed(src)
    bad = 0

    def fail(msg: str) -> None:
        nonlocal bad
        bad += 1
        print(f"\n  {msg}")

    # ── the pane column ──────────────────────────────────────────────────
    if not p.saw_pane:
        fail("no .lv-pane found in the pane column; this check is looking at "
             "the wrong markup and is not protecting anything.")
        print(f"\nlayout FAILED: {bad}")
        return 1

    above = p.pane_children[:p.pane_index]
    for rec in above:
        if not rec["show"]:
            continue
        if " ".join((rec["show"] or "").split()) in ALLOWED_IN_PANE_FLOW:
            continue
        fail(f"<{rec['tag']} class=\"{rec['cls']}\"> sits ABOVE the canvas in "
             f"the pane's flow and is conditional on `{rec['show']}`. When "
             f"that flips, every pane below it resizes and the chart moves "
             f"under the cursor. Put it in the .lv-over overlay inside "
             f".lv-pane instead.")

    # And the overlay has to actually be there and actually be an overlay.
    if 'class="lv-over' not in src:
        fail("there is no .lv-over overlay, so there is nowhere for a "
             "conditional notice to go except back into the flow.")
    css = src[:src.index("</style>")] if "</style>" in src else src
    over = re.search(r"\.lv-over\s*\{([^}]*)\}", css)
    if not over or "position:absolute" not in over.group(1).replace(" ", ""):
        fail(f"the .lv-over overlay is not absolutely positioned, so it "
             f"still takes part in the layout: {over.group(1) if over else None}")
    if not over or "pointer-events:none" not in over.group(1).replace(" ", ""):
        fail("the overlay does not pass the mouse through, so it covers the "
             "tape it is drawn over and rows under it cannot be clicked.")

    # ── page level ───────────────────────────────────────────────────────
    for rec in p.page_children:
        if not rec["show"]:
            continue
        if (rec["show"] or "").strip() in ALLOWED_PAGE_FLOW:
            continue
        fail(f"<{rec['tag']} class=\"{rec['cls']}\"> is a page-level block "
             f"conditional on `{rec['show']}`. It pushes EVERY pane down as "
             f"it comes and goes. It belongs in .lv-toasts.")

    if not p.toasts_fixed:
        fail("no .lv-toasts container: the page-level notices have nowhere "
             "to float to.")
    toasts = re.search(r"\.lv-toasts\s*\{([^}]*)\}", css)
    if not toasts or "position:fixed" not in toasts.group(1).replace(" ", ""):
        fail("the .lv-toasts stack is not fixed, so it still reflows the "
             "page it is supposed to float over.")

    # ── the native controls ──────────────────────────────────────────────
    root = None
    for rule in re.finditer(r"([^{}]*)\{([^}]*)\}", css):
        # Strip any comment that ends just before the selector.
        sel = rule.group(1).split("*/")[-1].strip().rstrip(",")
        if sel in (
                "html", ":root", "body", "html, body"):
            if "color-scheme" in rule.group(2):
                root = rule
                break
    if root is None:
        fail("the page never declares `color-scheme: dark` at the root. "
             "Styling an input's background does not reach the parts the "
             "browser draws itself — a number field's spinner arrows and a "
             "checkbox come out white on a dark page without it. A "
             "declaration on the inputs alone is not enough: it leaves the "
             "page itself light, and the next control added inherits that.")
    # White is fine on a marker tick or a badge; it is not fine on anything
    # you type into. Scoped to selectors that name a control, so the rule
    # says what it means rather than banning a colour outright.
    control = re.compile(r"input|select|textarea|\.lv-inp|\.lv-sel|\.lv-qty",
                         re.I)
    white = re.compile(r"background(-color)?\s*:\s*(#fff\b|#ffffff\b|white\b)",
                       re.I)
    for rule in re.finditer(r"([^{}]+)\{([^}]*)\}", css):
        sel, body = rule.group(1).strip(), rule.group(2)
        if not control.search(sel):
            continue
        if white.search(body):
            line = css[:rule.start()].count("\n") + 1
            fail(f"a control is given a white background at line {line} "
                 f"({sel!r}). No white input backgrounds on this page.")
    # And the shared control rule must keep taking its colour from the token,
    # or it drifts away from the rest of the site the next time one moves.
    m = re.search(r"\.lv-sel,\s*\.lv-inp\s*\{([^}]*)\}", css)
    if not m or "var(--raised)" not in m.group(1):
        fail("`.lv-sel, .lv-inp` no longer takes its background from the "
             "shared --raised token, so the inputs will drift away from "
             "every other input on the site.")

    # ── the sizes ────────────────────────────────────────────────────────
    js = (ROOT / "static" / "js" / "equities_live.js").read_text(encoding="utf-8")
    m = re.search(r"^\s*qty:\s*(\d+),", js, re.M)
    if not m:
        fail("the default quantity could not be found in the pane state")
    elif int(m.group(1)) != 10:
        fail(f"the default quantity is {m.group(1)}, not 10. The ladder is "
             f"one click from an order, so the default is the size a "
             f"mis-click costs.")
    m = re.search(r"const LV_QTYS = \[([^\]]*)\]", js)
    want = [10, 20, 30, 50, 100, 200]
    got = [int(x) for x in re.findall(r"\d+", m.group(1))] if m else []
    if got != want:
        fail(f"the quantity quick-picks are {got}, not {want}")
    if "pane.qtyPresets" not in src:
        fail("the quantity buttons are not wired to the presets, so the "
             "list in the JS is not the list on screen")

    # -- the name being traded --------------------------------------------
    # It is not enough that the symbol is SOMEWHERE. The input holds what is
    # being TYPED, which is a different thing: after a watch it can sit empty
    # or hold a half-typed next symbol while the pane trades something else.
    # The moment orders can leave is the moment the symbol matters most.
    if 'class="lv-sym"' not in src:
        fail("there is no permanent symbol display in the pane bar. The "
             "ticker input shows what you are TYPING, not what is being "
             "traded, and after a watch those are not the same string.")
    m = re.search(r'<b class="lv-sym"[^>]*>', src, re.S)
    if not m or "pane.symbol" not in m.group(0):
        fail(f"the symbol display does not read pane.symbol: "
             f"{m.group(0) if m else None}")
    elif "x-show" in m.group(0):
        fail("the symbol display is conditional. It is the one thing on the "
             "pane that must never be absent.")
    # The :disabled BINDING specifically. Matching "pane.armed" anywhere in
    # the tag passed on the :title alone, which explains the lock without
    # applying it - and the gate said nothing until that was injected.
    sym_input = re.search(r'<input class="lv-inp sym".*?>', src, re.S)
    if not sym_input or not re.search(
            r':disabled\s*=\s*"[^"]*pane\.armed', sym_input.group(0)):
        fail("the ticker input is not locked while the pane is armed. "
             "Changing the symbol under an armed pane leaves the arm switch, "
             "the ladder and any working order pointing at a name you are no "
             "longer looking at.")

    # -- x-model names belong to the input, and to nothing else ------------
    #
    # THE BUG THIS EXISTS FOR. `pane.pending` is the ticker input's x-model:
    # a string, bound two-way to a text box. The in-flight move state was
    # given the same name, so releasing a drag set that property to an
    # OBJECT — Alpine wrote it into the input and synced a string back, and
    # the move state was gone a frame later. On screen the order marker went
    # target -> origin -> target across about a second.
    #
    # Nothing caught it. Every trace ran in node, which has no Alpine and no
    # DOM, so the state machine was correct in every harness and wrong in the
    # only place that matters. This is the static check that would have.
    models = set(re.findall(r'x-model(?:\.\w+)*\s*=\s*"pane\.(\w+)"', src))
    if not models:
        fail("no x-model bindings found on the pane; this check is looking "
             "at the wrong markup and is protecting nothing.")
    for name in sorted(models):
        # An object or array literal assigned to a name the DOM owns.
        hit = re.search(r"this\." + name + r"\s*=\s*[\[{]", js)
        if hit:
            line = js[:hit.start()].count("\n") + 1
            fail(f"`pane.{name}` is bound with x-model AND assigned a "
                 f"structure at equities_live.js:{line}. Alpine owns that "
                 f"property: it writes it into the input and syncs a string "
                 f"back, so the structure is destroyed a frame later. Give "
                 f"the state its own name.")
        # And declared as a non-string default, which is the same collision
        # arriving from the other direction.
        decl = re.search(r"^\s*" + name + r":\s*([\[{])", js, re.M)
        if decl:
            line = js[:decl.start()].count("\n") + 1
            fail(f"`pane.{name}` is bound with x-model but declared as a "
                 f"structure at equities_live.js:{line}. An x-model property "
                 f"holds what the input holds, which is text.")

    if bad:
        print(f"\nlayout FAILED: {bad}")
        return 1
    print(f"live layout: {len(above)} blocks above the canvas and none of "
          f"them conditional; page notices float; native controls are dark; "
          f"qty defaults to 10 with quick-picks {want}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
