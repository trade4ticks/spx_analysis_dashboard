"""The topbar nav: six categories, every page in exactly one of them.

WHY THIS EXISTS. The nav is one partial included by seventeen pages, so a
mistake in it is a mistake on every page at once — and the two ways it can be
wrong are both silent. A page whose `nav_active` matches no item still
renders a perfectly good nav, with nothing highlighted, and looks like a page
you reached from nowhere. A page that drops out of the category list
altogether is simply unreachable, and nothing else in the repo would notice:
the route still works, so every smoke check passes.

So this renders EVERY page template and asks the questions a person would:

  * is this page's key in exactly one category, and is that category marked
  * is the page itself marked inside its own menu, exactly once
  * are the category names buttons rather than links (they have no page)
  * do the six categories hold what they are meant to, in order
  * does every item's href actually go somewhere, with the three
    Equities pages pointing at the OTHER service's port

It renders with the REAL asset() and a stub request, the same way
check_rendered_assets does — a gate that renders with its own helpers is
checking a page the app does not serve.
"""
from __future__ import annotations

import re
import sys
from html.parser import HTMLParser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import jinja2                                             # noqa: E402

from app.assets import asset                              # noqa: E402

TEMPLATES = ROOT / "templates"
LIVE_PORT = 8001

# What the nav is FOR, written out here rather than read from the partial:
# a gate that derives its expectation from the file it is checking agrees
# with every edit, including the wrong ones.
EXPECTED = [
    ("SPX", ["Dashboard", "Heatmap", "Today", "Strategy Signal"]),
    ("Factor", ["Factor Analysis", "Factor Signals", "Factor Trades"]),
    ("Equity", ["Equity IV", "Ticker Analysis"]),
    ("Scalp", ["Equities Scalp", "Equities Live", "Equities Wall",
               "Equities Scan"]),
    ("Backtest", ["OO/Mesosim Backtest", "Backtest Portfolio", "Backtest IV"]),
    ("Research", ["AI Explorer", "Research", "Research 2"]),
]
# The pages that live on the OTHER service, and the path each one has THERE.
# Their nav key is deliberately not their path: "/equities-live" is the key
# the page passes, "/" is where it actually is on port 8001.
LIVE_PATHS = {"Equities Live": "/", "Equities Wall": "/wall",
              "Equities Scan": "/scan"}

FAILS: list[str] = []


def check(cond, msg):
    if not cond:
        FAILS.append(msg)


class _StubURL:
    hostname = "localhost"
    scheme = "http"
    path = "/"

    def __str__(self): return "http://localhost/"


class _StubRequest:
    url = _StubURL()
    headers: dict = {}
    query_params: dict = {}
    scope = {"type": "http"}


def env() -> jinja2.Environment:
    e = jinja2.Environment(loader=jinja2.FileSystemLoader(str(TEMPLATES)),
                           autoescape=True, keep_trailing_newline=True)
    e.globals["asset"] = asset
    e.globals["live_port"] = LIVE_PORT
    return e


class Nav(HTMLParser):
    """The rendered nav, read back as structure rather than as a string."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.depth = 0
        self.cats: list[dict] = []
        self.cur: dict | None = None
        self._in_btn = False
        self._in_link = False

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        cls = a.get("class", "")
        if tag == "nav" and "topbar-nav" in cls:
            self.depth = 1
            return
        if not self.depth:
            return
        if "nav-cat" in cls.split() or cls.startswith("nav-cat "):
            self.cur = {"name": "", "on": "on" in cls.split(),
                        "items": [], "current": a.get("aria-current")}
            self.cats.append(self.cur)
        elif "nav-cat-btn" in cls:
            self._in_btn = True
            if self.cur is not None:
                self.cur["btn"] = a
        elif "nav-link" in cls and self.cur is not None:
            self._in_link = True
            self.cur["items"].append(
                {"label": "", "href": a.get("href"),
                 "on": "on" in cls.split(),
                 "current": a.get("aria-current"), "attrs": a})

    def handle_endtag(self, tag):
        if tag == "nav":
            self.depth = 0
        elif tag == "button":
            self._in_btn = False
        elif tag == "a":
            self._in_link = False

    def handle_data(self, data):
        if self._in_btn and self.cur is not None:
            self.cur["name"] += data.strip()
        elif self._in_link and self.cur is not None and self.cur["items"]:
            self.cur["items"][-1]["label"] += data.strip()


def pages() -> list[tuple[str, str]]:
    """(template, its nav_active) for every page that includes the nav."""
    out = []
    for p in sorted(TEMPLATES.glob("*.html")):
        src = p.read_text(encoding="utf-8")
        if "_nav.html" not in src or p.name == "_nav.html":
            continue
        m = re.search(r'set nav_active\s*=\s*"([^"]*)"', src)
        if not m:
            FAILS.append(f"{p.name} includes the nav but sets no nav_active, "
                         f"so nothing in its bar can be highlighted")
            continue
        out.append((p.name, m.group(1)))
    return out


def main() -> int:
    e = env()
    found = pages()
    check(len(found) >= 18,
          f"only {len(found)} page templates include the nav; there were 18 "
          f"— has a page been dropped from it")

    keys_seen: dict[str, str] = {}
    for name, key in found:
        try:
            html = e.get_template(name).render(request=_StubRequest())
        except Exception as exc:                          # noqa: BLE001
            FAILS.append(f"{name} failed to render: {type(exc).__name__}: {exc}")
            continue
        nav = Nav()
        nav.feed(html)

        shape = [(c["name"], [i["label"] for i in c["items"]])
                 for c in nav.cats]
        if shape != EXPECTED:
            FAILS.append(f"{name}: the nav is {shape}, expected {EXPECTED}")
            continue

        # ── the category names are not links ─────────────────────────────
        for c in nav.cats:
            check("href" not in c.get("btn", {}),
                  f"{name}: the category {c['name']!r} is a link; a category "
                  f"has no page of its own and clicking it must open its menu")
            check(c.get("btn", {}).get("aria-expanded") == "false",
                  f"{name}: {c['name']!r} does not say whether it is open; "
                  f"a screen reader would announce a button that does nothing")
            check(c.get("btn", {}).get("aria-controls"),
                  f"{name}: {c['name']!r} names no menu")

        # ── exactly one category marked, and it holds this page ──────────
        marked = [c["name"] for c in nav.cats if c["on"]]
        owner = [c["name"] for c in nav.cats
                 if any(i["on"] for i in c["items"])]
        check(len(owner) == 1,
              f"{name}: its key {key!r} is in {len(owner)} categories "
              f"({owner}); every page belongs to exactly one")
        check(marked == owner,
              f"{name}: the marked category is {marked} but the page is in "
              f"{owner} — the bar would point at the wrong one")

        # ── and the page is marked inside its own menu, once ─────────────
        current = [i["label"] for c in nav.cats for i in c["items"] if i["on"]]
        check(len(current) == 1,
              f"{name}: {len(current)} items marked as the current page "
              f"({current}); its key is {key!r}")
        aria = [i["label"] for c in nav.cats for i in c["items"]
                if i["current"] == "page"]
        check(aria == current,
              f"{name}: aria-current marks {aria} while the styling marks "
              f"{current}; a screen reader and the screen would disagree")

        # ── every link goes somewhere real ──────────────────────────────
        for c in nav.cats:
            for i in c["items"]:
                href = i["href"] or ""
                check(href.strip() != "",
                      f"{name}: {i['label']!r} has an empty href")
                if i["label"] in LIVE_PATHS:
                    want = f"//{_StubURL.hostname}:{LIVE_PORT}{LIVE_PATHS[i['label']]}"
                    check(href == want,
                          f"{name}: {i['label']!r} points at {href!r}, not "
                          f"{want!r} — that page is on the other service and "
                          f"the port has to come from live_port")
                else:
                    check(href.startswith("/") and not href.startswith("//"),
                          f"{name}: {i['label']!r} points at {href!r}, which "
                          f"is not a path on this service")
        keys_seen[key] = name

    # ── every page in the nav is a page that exists ─────────────────────
    #
    # The other direction: an item whose key no page sets is a link into a
    # nav that can never highlight it, which is how a renamed route leaves a
    # dead entry behind.
    if found:
        html = e.get_template(found[0][0]).render(request=_StubRequest())
        nav = Nav()
        nav.feed(html)
        n_items = sum(len(c["items"]) for c in nav.cats)
        check(n_items == len(EXPECTED and [i for _, ls in EXPECTED for i in ls]),
              f"the nav holds {n_items} pages, expected "
              f"{len([i for _, ls in EXPECTED for i in ls])}")
        hrefs = {i["href"] for c in nav.cats for i in c["items"]}
        local = {h for h in hrefs if h.startswith("/") and not h.startswith("//")}
        unset = local - set(keys_seen)
        check(not unset,
              f"the nav links to {sorted(unset)}, which no page template "
              f"claims with nav_active — a renamed route leaves exactly this")

    # ── the script that opens them ships with it ────────────────────────
    partial = (TEMPLATES / "_nav.html").read_text(encoding="utf-8")
    check("js/nav.js" in partial,
          "the nav partial loads no script, so no menu opens")
    js = (ROOT / "static" / "js" / "nav.js").read_text(encoding="utf-8")
    for key in ("ArrowDown", "ArrowUp", "ArrowLeft", "ArrowRight", "Escape",
                "Home", "End"):
        check(f"'{key}'" in js, f"nav.js handles no {key} key")
    check("aria-expanded" in js,
          "nav.js opens menus without saying so on the button")
    # THE BUBBLING GUARD. The category listener sits on an element that
    # CONTAINS the button, so a key the button handled arrives again with
    # focus already moved into the menu: Down opened the menu, landed on the
    # first page and stepped straight to the second, and every arrow moved
    # two. Only a browser can see it happen — this is the reminder that the
    # guard is load-bearing.
    check("e.target === b" in js,
          "nav.js does not stop the button's own keys reaching the category "
          "listener; both fire for one press and every arrow moves twice")

    print(f"nav: {len(found)} pages rendered, {len(EXPECTED)} categories, "
          f"failures: {len(FAILS)}")
    for m in FAILS:
        print(f"  FAIL {m}")
    return 1 if FAILS else 0


sys.exit(main())
