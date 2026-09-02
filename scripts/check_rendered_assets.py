"""Render every page and assert its markup actually references something.

WHAT SHIPPED. A template emitted

    <script src=    </div>

-- an empty attribute that swallowed the following close tag. The browser
requested a URL literally named "</div", the bundle was never fetched, and
every panel rendered empty. The layout was intact, there was no server error,
and the only signal was a 404 for a nonsense path in the log.

WHY THE EXISTING GATES ALL PASSED:

  check_alpine_syntax    parses x-* EXPRESSIONS. It never looks at the tags
                         around them, and the corruption was in a tag.
  check_alpine_refs      resolves Alpine calls to component members. Same.
  check_asset_versions   scans the template SOURCE for asset() calls. The
                         source was correct — `<script src={{ asset(...) }}>`
                         appeared twice, once corrupted. It counted both.
  check_template_render  renders, but with `asset` UNDEFINED, because
                         importing it meant importing app.main and every
                         router. So every script src came out empty in its
                         output too, and a page emitting `<script src=` looked
                         exactly like a page that did not.

Not one of them looked at the RENDERED OUTPUT with the real helpers bound.
That is the gap, and it is the whole class: any template edit that mangles a
tag rather than an expression is invisible to all four.

So this renders each page with the REAL asset() -- imported from app.assets,
which exists so this can be done without a second copy -- and asserts:

  * every src/href is non-empty
  * no attribute value contains '<', which is what a swallowed tag looks like
  * every local /static/ reference resolves to a file on disk
  * <script> and <div> tags balance, ratcheted against two pre-existing
    imbalances so the gate is not red on the day it is written

The static-file check is the one that would also catch a renamed bundle, a
typo in a path, and a stale reference to a deleted file.
"""
from __future__ import annotations

import sys
from html.parser import HTMLParser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import os                                                 # noqa: E402
import jinja2                                             # noqa: E402


class _StubURL:
    hostname = "localhost"
    scheme = "http"
    path = "/"

    def __str__(self): return "http://localhost/"


class _StubRequest:
    """The minimum of starlette's Request that these templates read."""
    url = _StubURL()
    base_url = _StubURL()
    headers: dict = {}
    query_params: dict = {}
from app.assets import asset                              # noqa: E402

TEMPLATES = ROOT / "templates"
STATIC = ROOT / "static"

# Attributes that name something the browser will go and fetch. An empty one
# is never intentional.
URL_ATTRS = {
    "script": ("src",),
    "link":   ("href",),
    "img":    ("src",),
    "iframe": ("src",),
}

# Void elements never close, so they must not be counted when balancing.
VOID = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link",
        "meta", "param", "source", "track", "wbr"}

# Balanced separately because an unbalanced one of these is what the shipped
# corruption produced, and a page can be visually fine while nesting is wrong.
BALANCE = ("div", "script", "table", "template", "tbody", "thead")

# Imbalances that PREDATE this gate, on pages nobody is currently working on.
# Recorded rather than fixed: browsers auto-correct these, both pages render,
# and silently rewriting the nesting of a page I am not otherwise touching is
# a larger risk than the bug.
#
# This is a RATCHET, not an exemption. The numbers are printed on every run,
# any increase fails, and a decrease fails too so the baseline cannot drift
# above the truth.
# Counted by the PARSER, not by a regex over the source. A regex over `<div`
# reports 3 for oi_analysis because it also counts the string literals inside
# its inline <script> blocks; the parser treats script content as raw text and
# reports 1, which is the number that describes the DOM.
BALANCE_BASELINE = {
    "oi_analysis.html": {"div": 1},     # one <div> never closed
    "research.html":    {"div": -1},    # one </div> more than was opened
}


class Scan(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.problems: list[str] = []
        self.refs: list[tuple[str, str, str, int]] = []
        self.depth: dict = {t: 0 for t in BALANCE}
        self.closed_too_many: list[str] = []

    def handle_starttag(self, tag, attrs):
        line = self.getpos()[0]
        for name, value in attrs:
            # A swallowed tag lands INSIDE an attribute value. The test is
            # for '</' specifically, not a bare '<': these templates are full
            # of Alpine expressions like `page < total` and `n <= 20`, and
            # flagging those would bury the real thing in false positives —
            # the first version of this check produced 25 of them across six
            # pages and one real finding.
            if value and "</" in value:
                self.problems.append(
                    f"line {line}: <{tag} {name}=...> contains '<' — a tag was "
                    f"swallowed into an attribute value: {value[:60]!r}")
            if tag in URL_ATTRS and name in URL_ATTRS[tag]:
                if value is None or not value.strip():
                    self.problems.append(
                        f"line {line}: <{tag} {name}> is EMPTY. The browser "
                        f"will request the page itself, or whatever follows "
                        f"the tag, and the resource is never loaded.")
                else:
                    self.refs.append((tag, name, value.strip(), line))
        if tag in self.depth and tag not in VOID:
            self.depth[tag] += 1

    def handle_startendtag(self, tag, attrs):
        # Self-closing: attributes still matter, nesting does not.
        d = dict(self.depth)
        self.handle_starttag(tag, attrs)
        self.depth = d

    def handle_endtag(self, tag):
        if tag in self.depth:
            # NOT clamped at zero. The first version reset a negative depth to
            # 0, which hid every subsequent imbalance behind the first one --
            # oi_analysis reported 1 unclosed div when the raw count is 3.
            self.depth[tag] -= 1
            if self.depth[tag] == -1:
                self.closed_too_many.append(f"{tag} at line {self.getpos()[0]}")


def render(name: str) -> str:
    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(str(TEMPLATES)),
        autoescape=True,
        keep_trailing_newline=True,
    )
    # The real helper, not a stand-in. A gate that renders with its own copy
    # of asset() passes against a definition the app does not use.
    env.globals["asset"] = asset
    # Both apps bind this, and the shared nav partial reads it to build the
    # cross-port link to Equities Live.
    env.globals["live_port"] = int(os.environ.get("LIVE_PORT", "8001"))
    # FastAPI always puts `request` in a template's context, so a gate that
    # renders without one is rendering something the app never serves — this
    # gate reported fourteen "failed to render" the moment the nav began using
    # it. The stub carries only what templates actually read; anything else
    # they reach for should fail loudly here rather than in a browser.
    return env.get_template(name).render(request=_StubRequest())


def check_page(name: str) -> list[str]:
    try:
        html = render(name)
    except Exception as exc:                              # noqa: BLE001
        return [f"failed to render: {type(exc).__name__}: {exc}"]

    s = Scan()
    s.feed(html)
    out = list(s.problems)

    for tag, attr, value, line in s.refs:
        if value.startswith(("http://", "https://", "//", "data:", "#",
                             "mailto:")):
            continue
        if not value.startswith("/static/"):
            out.append(f"line {line}: <{tag} {attr}={value!r}> is a local path "
                       f"outside /static/ — nothing serves it")
            continue
        rel = value.split("?", 1)[0][len("/static/"):]
        if not (STATIC / rel).is_file():
            out.append(f"line {line}: <{tag} {attr}> points at "
                       f"static/{rel}, which does not exist")

    return out


# Pre-existing imbalances met on this run. Collected rather than printed from
# inside the check, so they appear together under their own heading instead of
# interleaved with whichever page happened to have a real problem.
KNOWN: list[tuple] = []


def check_balance(name: str, html: str) -> list[str]:
    """Tag nesting, ratcheted against what was already broken.

    Separated from the asset checks and given a BASELINE, because two pages
    carry pre-existing imbalances that predate this gate and sit on pages
    nobody is currently working on. Failing on those would make the gate red
    from the day it was written, and a permanently-red gate is one that gets
    ignored -- which is how the defect it exists to catch would get through
    anyway.

    So the baseline is a ratchet: the existing damage is RECORDED and printed
    every run, and any increase fails. New imbalances cannot hide behind old
    ones, and the old ones cannot be quietly forgotten either.
    """
    s = Scan()
    s.feed(html)
    out = []
    for tag, n in sorted(s.depth.items()):
        if not n:
            continue
        allowed = BALANCE_BASELINE.get(name, {}).get(tag, 0)
        if n == allowed:
            KNOWN.append((name, tag, n))
            continue
        if abs(n) > abs(allowed):
            out.append(
                f"{abs(n)} {'unclosed' if n > 0 else 'extra closing'} <{tag}>, "
                f"against a baseline of {abs(allowed)} — nesting is wrong, and "
                f"a page can look correct while it is")
        else:
            out.append(
                f"<{tag}> imbalance improved from {abs(allowed)} to {abs(n)} — "
                f"lower BALANCE_BASELINE[{name!r}] so it cannot regress")
    return out


def self_test() -> int:
    """Prove the scanner fires on the exact markup that shipped.

    Two earlier checks in this project passed while a bug was present because
    the fixture could not produce the failure. This one reproduces it.
    """
    bad = 0
    broken = '<html><body>\n<script src=    </div>\n  </div>\n</body></html>'
    s = Scan()
    s.feed(broken)
    if not any("EMPTY" in p or "swallowed" in p for p in s.problems):
        print("  SELF-TEST: the shipped markup was not flagged")
        bad += 1
    good = ('<html><body><div><script src="/static/js/x.js?v=1"></script>'
            '</div></body></html>')
    s2 = Scan()
    s2.feed(good)
    if s2.problems or any(s2.depth.values()):
        print(f"  SELF-TEST: well-formed markup was flagged: {s2.problems}")
        bad += 1
    if not bad:
        print("self-test: the scanner flags an empty src and passes clean "
              "markup")
    return bad


def main() -> int:
    bad = self_test()
    pages = sorted(p.name for p in TEMPLATES.glob("*.html")
                   if not p.name.startswith("_"))
    for name in pages:
        try:
            html = render(name)
        except Exception as exc:                          # noqa: BLE001
            print(f"\n  {name}\n      failed to render: {exc}")
            bad += 1
            continue
        problems = check_page(name) + check_balance(name, html)
        if problems:
            bad += len(problems)
            print(f"\n  {name}")
            for p in problems:
                print(f"      {p}")
    if KNOWN:
        # Printed EVERY run, not only when something else fails. A ratchet
        # nobody sees is an exemption, and an exemption nobody sees is how the
        # damage it records stops being damage and becomes the shape of the
        # code.
        print("\n  pre-existing and ratcheted — these do not fail the build,")
        print("  but any increase does, and so does a decrease not recorded:")
        for name, tag, n in KNOWN:
            kind = "unclosed" if n > 0 else "extra closing"
            print(f"      {name}: {abs(n)} {kind} <{tag}>")

    print(f"\npages rendered: {len(pages)}, problems: {bad}")
    return 1 if bad else 0


sys.exit(main())
