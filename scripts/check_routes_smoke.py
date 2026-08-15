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


def main() -> int:
    import app.main as main_mod

    # Neutralise the DB before the lifespan runs. main.py does
    # `from app.db import init_pool, close_pool`, so the names to patch are
    # the ones bound in app.main, not the originals in app.db.
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
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
