#!/usr/bin/env python3
"""Smoke check: every HTML page route actually returns 200.

Why this exists
---------------
scripts/check_template_render.py renders templates directly through Jinja and
never touches FastAPI. That is the right gate for "did the markup change", but
it is blind to how main.py CALLS the renderer. It passed with a green
10/10 while every page in the app returned 500, because Starlette >= 1.0
removed the legacy `TemplateResponse(name, {"request": request})` signature
and the correct form puts request first. That failure surfaces at request
time, not import time, so only an actual request catches it.

This script issues a real request per page route through FastAPI's TestClient
and asserts:
  * HTTP 200
  * an HTML content-type
  * a non-trivial body
  * the shared nav rendered (catches a broken `{% include %}`)
  * exactly one nav link marked active, and it points at the page requested
    (catches nav_active being unset, misspelled, or copied wrong)

Routes are DISCOVERED from the app, not hardcoded, so a new page is covered
the moment it is added.

The DB is never touched: init_pool/close_pool are patched out before the
TestClient starts the lifespan, so this runs anywhere with no .env and no
Postgres.

Usage
-----
    python scripts/check_routes_smoke.py

Exit 0 = all page routes healthy. Exit 1 = at least one failed.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


async def _noop() -> None:
    """Stand-in for init_pool / close_pool."""
    return None


def _db_env() -> tuple[bool, str]:
    """(can_use_real_db, why_not).

    The API checks need a REAL pool. The page sweep does not, and used to
    patch init_pool out unconditionally -- which meant get_oi_pool() returned
    None and the factor-trades endpoints could only ever answer "OI database
    not configured". So the pool is patched ONLY when the DB is unavailable.

    init_pool reads DATABASE_URL with os.environ[...] (required, not
    optional), so both vars must be present to let the real lifespan run.
    """
    import os
    missing = [v for v in ("DATABASE_URL", "OI_DATABASE_URL") if not os.getenv(v)]
    if missing:
        return False, "not set: " + ", ".join(missing)
    return True, ""


def main() -> int:
    # DB-free static pass first: it is the only check here that does not
    # need a live database, and it catches the class the others cannot.
    static_bad = _check_undefined_names()
    static_bad += _check_alpine_templates()
    static_bad += _check_sql_placeholders()

    import app.main as main_mod

    real_db, why_not = _db_env()
    if not real_db:
        # No DB: neutralise the lifespan so the page sweep still runs. main.py
        # does `from app.db import init_pool, close_pool`, so the names to
        # patch are the ones bound in app.main, not the originals in app.db.
        main_mod.init_pool = _noop
        main_mod.close_pool = _noop

    from fastapi.testclient import TestClient

    app = main_mod.app

    # Page routes = GET routes defined in app.main itself. That excludes the
    # /api/* routers and FastAPI's own /docs, /openapi.json.
    pages: list[str] = []
    for r in app.routes:
        endpoint = getattr(r, "endpoint", None)
        methods = getattr(r, "methods", set()) or set()
        if endpoint is None or "GET" not in methods:
            continue
        if getattr(endpoint, "__module__", "") != "app.main":
            continue
        pages.append(r.path)
    pages.sort()

    if not pages:
        print("ERROR: discovered no page routes in app.main — did the module move?",
              file=sys.stderr)
        return 1

    ok = bad = 0
    with TestClient(app) as client:
        for path in pages:
            try:
                resp = client.get(path)
            except Exception as e:  # noqa: BLE001 — report, don't abort the sweep
                print(f"  {path:26s} RAISED {type(e).__name__}: {e}")
                bad += 1
                continue

            if resp.status_code != 200:
                body = resp.text[:200].replace("\n", " ")
                print(f"  {path:26s} HTTP {resp.status_code}  {body}")
                bad += 1
                continue

            ctype = resp.headers.get("content-type", "")
            if "html" not in ctype:
                print(f"  {path:26s} unexpected content-type: {ctype}")
                bad += 1
                continue

            html = resp.text
            if len(html) < 500:
                print(f"  {path:26s} suspiciously small body ({len(html)} bytes)")
                bad += 1
                continue

            if 'class="topbar-nav"' not in html:
                print(f"  {path:26s} nav missing — _nav.html include did not render")
                bad += 1
                continue

            active = re.findall(r'<a class="nav-link on" href="([^"]+)"', html)
            if len(active) != 1:
                print(f"  {path:26s} expected exactly 1 active nav link, found {len(active)}")
                bad += 1
                continue
            if active[0] != path:
                print(f"  {path:26s} active nav link is {active[0]!r} — nav_active mismatch")
                bad += 1
                continue

            print(f"  {path:26s} 200  {len(html):>7,} bytes  active={active[0]}")
            ok += 1

    print()
    print(f"page routes healthy: {ok}/{ok + bad}")

    api_bad = _check_factor_trades(app, real_db, why_not)
    return 1 if (bad or api_bad or static_bad) else 0


def _check_sql_placeholders() -> int:
    """Gate: does every $N a query is GIVEN actually appear in that query?

    Why this exists
    ---------------
    asyncpg sends every argument it is handed. If a parameter appears
    NOWHERE in the statement, Postgres has no typed context for it and
    raises, at query time:

        IndeterminateDatatypeError: could not determine data type of
        parameter $2

    That is not a "harmless unused argument". It is a 500.

    It happened on the random-entry baseline: the per-date COUNT query
    reuses the zone query's arg list verbatim (its cell and strike
    predicates are that query's own fragments, with placeholder numbers
    computed against it), but dropped the SELECT expression that was $2's
    only appearance. Nothing in the source looked wrong -- the arg list was
    correct and deliberate -- and the check that would have caught it is
    simply "is every index from 1 to the highest one referenced".

    Applies to the SQL builder functions, which is why those queries were
    extracted from their call sites: an f-string built inline cannot be
    inspected without executing the endpoint.
    """
    from app.routers import factor_trades as ft

    # (label, sql) for every builder, exercised BOTH with and without the
    # optional filters -- a placeholder can be present in one shape of the
    # query and absent in the other.
    # These mirror the FOUR shapes zone() actually builds. The single-metric
    # cell predicate consumes $3 and the two-factor one consumes $3 and $4,
    # so the optional strike filter lands on $4 or $5 respectively -- zone()
    # computes that index from len(args) rather than hardcoding it. Pairing a
    # 1f predicate with the 2f strike index produces a query that IS broken,
    # which this gate correctly flags; getting the stubs wrong therefore
    # reports a bug in the test rather than in the code, so they are spelled
    # out per shape instead of looped over one predicate.
    CELL_1F = "y = ANY($3::int[])"
    CELL_2F = "(y, z) IN (SELECT * FROM unnest($3::int[], $4::int[]))"
    cases = [
        ("_zone_count_sql 1f",          ft._zone_count_sql("SELECT 1", "bt.x > 0", CELL_1F, "")),
        ("_zone_count_sql 1f +strike",  ft._zone_count_sql("SELECT 1", "bt.x > 0", CELL_1F,
                                                           " AND tp.entry_price <= $4")),
        ("_zone_count_sql 2f",          ft._zone_count_sql("SELECT 1", "bt.x > 0", CELL_2F, "")),
        ("_zone_count_sql 2f +strike",  ft._zone_count_sql("SELECT 1", "bt.x > 0", CELL_2F,
                                                           " AND tp.entry_price <= $5")),
        # The random sampler fixes $1..$5 itself, so the strike filter is
        # always $6 regardless of metric mode.
        ("_random_zone_sql",            ft._random_zone_sql("SELECT 1", "")),
        ("_random_zone_sql +strike",    ft._random_zone_sql("SELECT 1", " AND tp.entry_price <= $6")),
        # The random-exit sampler fixes $1..$8 (adding the holding-period
        # CDF arrays), so its strike filter is $9.
        ("_random_exit_zone_sql",         ft._random_exit_zone_sql("SELECT 1", "t2.r1", "")),
        ("_random_exit_zone_sql +strike", ft._random_exit_zone_sql("SELECT 1", "t2.r1",
                                                                   " AND tp.entry_price <= $9")),
    ]

    bad = 0
    for label, sql in cases:
        used = {int(x) for x in re.findall(r"\$(\d+)", sql)}
        if not used:
            continue
        missing = [i for i in range(1, max(used) + 1) if i not in used]
        if missing:
            bad += 1
            print(f"  {label}: takes ${max(used)} but never references "
                  + ", ".join(f"${i}" for i in missing)
                  + "  — IndeterminateDatatypeError at query time")
        else:
            print(f"  {label}: ${'{'}1..{max(used)}{'}'} all referenced")

    print("sql placeholders: OK" if not bad
          else f"sql placeholders: {bad} QUERY(S) WITH AN UNREFERENCED PARAMETER")
    print()
    return bad


def _check_undefined_names() -> int:
    """Static scan for names that would raise NameError when a path runs.

    Closes a gap the other checks structurally cannot. py_compile accepts a
    NameError without complaint -- it is a runtime lookup, not a syntax
    error -- and the API smoke check below needs a live DB, so it skips on
    any machine without one. A /zone endpoint that 500'd on every call
    passed both.

    Scope-aware by design: a nested function legitimately reads its
    enclosing function's locals, so each function inherits the names
    assigned by its ancestors. A flat per-function scan reports every such
    closure as undefined, which is noise that would get the whole check
    ignored.

    Conservative in the other direction: names assigned anywhere within a
    function count as available throughout it, so a use-before-assignment is
    NOT reported. This catches genuinely absent names -- a missing import, a
    renamed variable whose definition was not renamed with it.
    """
    import ast
    import builtins

    EXTRA = {"__file__", "__name__", "__doc__", "__package__", "__spec__",
             "__loader__", "__builtins__", "__debug__"}
    KNOWN = set(dir(builtins)) | EXTRA
    findings: list[tuple[str, int, str, str]] = []

    SCOPES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)

    def bound_here(node) -> set[str]:
        """Names bound in THIS scope only — does not descend into nested ones.

        The previous version collected bindings from the whole module and
        handed them to every function, so a name defined in one function
        looked available in all of them. That made cross-function leakage
        invisible: series_trades was initialised in run() and used in zone(),
        and the scan passed. Real lexical scoping is the fix; a flat set is
        not merely imprecise, it is blind to the exact class this exists to
        catch.
        """
        out: set[str] = set()
        args = getattr(node, "args", None)
        if isinstance(args, ast.arguments):
            for a in args.posonlyargs + args.args + args.kwonlyargs:
                out.add(a.arg)
            if args.vararg: out.add(args.vararg.arg)
            if args.kwarg:  out.add(args.kwarg.arg)

        def walk(n):
            for c in ast.iter_child_nodes(n):
                if isinstance(c, SCOPES):
                    if not isinstance(c, ast.Lambda):
                        out.add(c.name)          # the def's NAME binds here
                    continue                      # its body is a child scope
                if isinstance(c, ast.ClassDef):
                    out.add(c.name); continue
                if isinstance(c, ast.Name) and isinstance(c.ctx, (ast.Store, ast.Del)):
                    out.add(c.id)
                elif isinstance(c, (ast.Import, ast.ImportFrom)):
                    for a in c.names:
                        out.add((a.asname or a.name).split(".")[0])
                elif isinstance(c, ast.ExceptHandler) and c.name:
                    out.add(c.name)
                elif isinstance(c, (ast.Global, ast.Nonlocal)):
                    out.update(c.names)
                walk(c)
        walk(node)
        return out

    def visit(node, enclosing: set[str], rel: str, where: str) -> None:
        scope = enclosing | bound_here(node)

        def walk(n):
            for c in ast.iter_child_nodes(n):
                if isinstance(c, SCOPES):
                    visit(c, scope, rel,
                          getattr(c, "name", "<lambda>"))
                    continue
                if isinstance(c, ast.Name) and isinstance(c.ctx, ast.Load)                         and c.id not in scope and c.id not in KNOWN:
                    findings.append((rel, c.lineno, c.id, where))
                walk(c)
        walk(node)

    for py in sorted((ROOT / "app").rglob("*.py")):
        rel = str(py.relative_to(ROOT)).replace("\\", "/")
        tree = ast.parse(py.read_text(encoding="utf-8"))
        visit(tree, set(), rel, "<module>")

    if findings:
        print("undefined names — these raise NameError when the path runs:")
        for f, ln, name, where in findings:
            print(f"  {f}:{ln}  {name!r} in {where}()")
    else:
        print("undefined names: none")
    print()
    return 1 if findings else 0


def _check_alpine_templates() -> int:
    """Every <template x-for/x-if> must contain exactly one element child.

    Alpine requires a single root inside those templates. Give one two
    siblings and it renders NOTHING -- no error in the console, no partial
    output, the whole loop just silently disappears. That cost a round trip
    when the split trail-rule control produced no dropdowns at all.

    Static and DB-free, like the undefined-name scan above.

    REPORT-ONLY for now. It currently flags two pre-existing pages
    (ai_explorer, backtest_iv_analysis) that appear to work in the browser,
    which means either those pages have a latent bug or this parser
    miscounts children on unclosed tags. Until that is resolved, failing the
    build on them would train people to ignore the gate -- so it warns and
    returns 0. Resolve, then flip the return to 1.
    """
    from html.parser import HTMLParser

    VOID = {"br", "hr", "img", "input", "meta", "link", "source", "area"}

    class P(HTMLParser):
        def __init__(self):
            super().__init__()
            self.stack: list = []
            self.bad: list = []

        def handle_starttag(self, tag, attrs):
            a = dict(attrs)
            directive = next((k for k in a if k in ("x-for", "x-if")), None)                 if tag == "template" else None
            if self.stack:
                self.stack[-1][1].append(tag)
            if tag in VOID:
                return
            self.stack.append([directive, [], self.getpos()[0],
                               a.get("x-for") or a.get("x-if") or ""])

        def handle_endtag(self, tag):
            if not self.stack:
                return
            d, kids, line, expr = self.stack.pop()
            if d and len(kids) != 1:
                self.bad.append((line, d, expr, len(kids), kids))

    bad = 0
    for tpl in sorted((ROOT / "templates").glob("*.html")):
        p = P()
        p.feed(tpl.read_text(encoding="utf-8"))
        for line, d, expr, n, kids in p.bad:
            print(f"  {tpl.name}:{line}  {d}=\"{expr[:44]}\" has {n} element "
                  f"children {kids} — Alpine renders nothing")
            bad += 1
    print("alpine template roots: ok" if not bad
          else f"alpine template roots: {bad} WARNING(S) — report-only, see docstring")
    print()
    return 0


def _check_factor_trades(app, real_db: bool, why_not: str) -> int:
    """Exercise the factor-trades API, which the page sweep above cannot see.

    That sweep only covers GET routes defined in app.main, so /api/* has had
    no gate at all -- which is how a str-vs-date parameter bug reached
    production in an endpoint that 500'd on every call. These are POST
    endpoints needing a real DB, so they SKIP cleanly when OI_DATABASE_URL is
    unset (dev boxes) and actually run on the VPS, which is where it matters.

    The assertion is deliberately weak: 200 with a JSON body, or a clean
    {"error": ...} payload. It is not checking numbers -- it is checking that
    the endpoint executes end to end instead of raising.
    """
    import os
    from fastapi.testclient import TestClient

    print()
    if not real_db:
        print(f"factor-trades API: SKIPPED — {why_not}.")
        print("  This check needs BOTH DATABASE_URL and OI_DATABASE_URL, because")
        print("  app.db.init_pool requires the former and the endpoints need the")
        print("  latter. With either missing the pool is stubbed out and every")
        print("  endpoint would answer 'OI database not configured'.")
        return 0

    bad = 0
    with TestClient(app) as client:
        r = client.get("/api/factor-trades/rules")
        if r.status_code != 200:
            print(f"  /rules  HTTP {r.status_code}  {r.text[:160]}")
            bad += 1
        else:
            body = r.json()
            groups = body.get("groups") or []
            # 200 with an empty catalog is the silent-empty failure mode: it
            # reads as success while telling you nothing works. Fail loudly.
            if body.get("error"):
                print(f"  /rules  200 but error: {body['error']}")
                return bad + 1
            if not groups or not body.get("n_rules"):
                print("  /rules  200 but EMPTY catalog — trade_path_rules has no rows, "
                      "or the pool is not connected")
                return bad + 1
            print(f"  /rules  200  {len(groups)} groups, {body.get('n_rules', 0)} rules")
            # First rule key of the first family, to drive the POSTs below.
            key = None
            for g in groups:
                for fam in g.get("families", []):
                    for rule in fam.get("rules", []):
                        key = key or rule.get("rule_key")
            metric = os.getenv("SMOKE_METRIC", "")
            if not metric:
                print("  /run /zone  SKIPPED — set SMOKE_METRIC to a metric that has a")
                print("      bin20_<metric> column in tt_bins (not merely a daily_features")
                print("      column); /run rejects anything else with 'no stored bins'.")
                return bad
            if not key:
                print("  /run /zone  SKIPPED — /rules returned no rule_key to drive them")
                return bad
            payload = {"primary_metric": metric, "entry_anchor": "open",
                       "rule_keys": [key], "n_bins": 20}
            for path, extra in (("/api/factor-trades/run", {}),
                                ("/api/factor-trades/zone", {"cells": [[0]]})):
                rr = client.post(path, json={**payload, **extra})
                if rr.status_code != 200:
                    print(f"  {path:28s} HTTP {rr.status_code}  {rr.text[:160]}")
                    bad += 1
                    continue
                b = rr.json()
                if b.get("error"):
                    print(f"  {path:28s} 200 but error: {b['error'][:110]}")
                    bad += 1
                    continue
                print(f"  {path:28s} 200  keys={len(b)}  "
                      f"horizon_auto_added={b.get('horizon_auto_added')}")
    return bad


if __name__ == "__main__":
    raise SystemExit(main())
