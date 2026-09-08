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
        # Script tags IN DOCUMENT ORDER, with whether each carries `defer`.
        # Order is the whole point -- see check_alpine_order.
        self.scripts: list[tuple[int, str, bool]] = []
        self.x_data: list[tuple[int, str]] = []

    def handle_starttag(self, tag, attrs):
        line = self.getpos()[0]
        if tag == "script":
            d = dict(attrs)
            self.scripts.append(((line), (d.get("src") or "").strip(),
                                 "defer" in d))
        for name, value in attrs:
            if name == "x-data" and value and value.strip():
                self.x_data.append((line, value.strip()))
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



# ── Alpine bootstrap order ──────────────────────────────────────────────────
#
# THE FAULT THIS EXISTS FOR. equities_scan.html shipped with
#
#     <script defer src=/static/js/equities_scan.js?v=...>
#
# one word different from the two pages that work. Alpine is loaded with
# `defer` in the head; deferred scripts execute in document order after
# parsing, so a deferred bundle at the end of the body runs AFTER Alpine has
# started and dispatched `alpine:init`. The bundle's listener is then added to
# an event that already fired, Alpine.data() is never called, and every
# expression on the page throws "<component> is not defined" — about
# twenty-five errors, one per property, none of which names the cause.
#
# WHY NO EXISTING GATE CAUGHT IT, which is worth being precise about rather
# than blaming the nearest one:
#
#   check_rendered_assets  asked whether the src RESOLVES. It did — the file
#                          exists and the hash was right. Passing was correct.
#   check_alpine_refs      asked whether every expression resolves to a member
#                          of the component. They all did. It executes the
#                          bundle directly under a stub Alpine, so it never
#                          observes when the browser would have run it.
#   node --check           the file parses. It does.
#
# Every one of them was answering its own question correctly. Nothing was
# asking whether the component would be REGISTERED IN TIME, which is a
# property of the rendered markup and belongs here, where the rendered markup
# already is.
#
# The rule is narrow on purpose: it only fires on a page that declares x-data,
# and only about local bundles, so a deferred third-party script that has
# nothing to do with Alpine is left alone.

ALPINE_MARK = "alpinejs"
LOCAL_JS = "/static/js/"


def check_alpine_order(name: str, html: str) -> list[str]:
    """A page's component must be registered before Alpine starts."""
    s = Scan()
    s.feed(html)
    if not s.x_data:
        return []

    # Only a bare identifier names a component defined in a bundle. An inline
    # object literal — x-data="{ open: false }" — defines itself and needs no
    # bundle at all, so requiring one would fail every small page.
    named = [(line, v) for line, v in s.x_data
             if v.replace("_", "").isalnum() and not v[0].isdigit()]
    if not named:
        return []

    out = []
    alpine_at = None
    for i, (_line, src, _defer) in enumerate(s.scripts):
        if ALPINE_MARK in src:
            alpine_at = i
            break

    local = [(i, line, src, defer)
             for i, (line, src, defer) in enumerate(s.scripts)
             if src.startswith(LOCAL_JS)]

    if not local:
        line, comp = named[0]
        out.append(
            f"line {line}: x-data={comp!r} names a component, but the page "
            f"loads NO local bundle from {LOCAL_JS} — nothing can define it, "
            f"and every expression on the page will throw "
            f"'{comp} is not defined'")
        return out

    if alpine_at is None:
        return out

    for i, line, src, defer in local:
        if i > alpine_at and defer:
            out.append(
                f"line {line}: <script defer src={src}> is loaded AFTER "
                f"Alpine and carries `defer`, so it runs after Alpine has "
                f"already dispatched alpine:init. The component registers "
                f"into an event that has fired and the page is dead with "
                f"'{named[0][1]} is not defined'. Drop `defer`: a plain "
                f"script at the end of the body runs during parsing, before "
                f"any deferred one.")
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

    # THE ORDER CHECK, both directions. A check that has never been seen to
    # fire is not evidence of anything, and this one was written after a page
    # shipped dead — so it is proved against the exact markup that shipped,
    # and against the exact markup that works.
    cdn = '<script defer src="https://cdn.jsdelivr.net/npm/alpinejs@3/x.js"></script>'
    shipped = ('<html><head>' + cdn + '</head><body x-data="equitiesScan">'
               '<script defer src="/static/js/equities_scan.js?v=1"></script>'
               '</body></html>')
    if not any("defer" in p and "alpine:init" in p
               for p in check_alpine_order("x.html", shipped)):
        print("  SELF-TEST: a deferred bundle after Alpine was NOT flagged — "
              "this is the markup that shipped dead")
        bad += 1

    works = shipped.replace('<script defer src="/static/js/',
                            '<script src="/static/js/')
    if check_alpine_order("x.html", works):
        print("  SELF-TEST: the working arrangement was flagged: "
              f"{check_alpine_order('x.html', works)}")
        bad += 1

    missing = ('<html><head>' + cdn + '</head><body x-data="equitiesScan">'
               '</body></html>')
    if not any("NO local bundle" in p
               for p in check_alpine_order("x.html", missing)):
        print("  SELF-TEST: a page naming a component with no bundle at all "
              "was not flagged")
        bad += 1

    # An inline object literal defines itself and must not be required to
    # have a bundle, or every small page fails.
    inline = ('<html><head>' + cdn + '</head><body x-data="{ open: false }">'
              '</body></html>')
    if check_alpine_order("x.html", inline):
        print("  SELF-TEST: an inline x-data object was told to find a bundle")
        bad += 1

    if not bad:
        print("self-test: the scanner flags an empty src, a bundle deferred "
              "past Alpine, and a component with no bundle; passes clean "
              "markup and inline x-data")
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
        problems = (check_page(name) + check_balance(name, html)
                    + check_alpine_order(name, html))
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
