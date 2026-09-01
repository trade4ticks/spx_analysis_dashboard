#!/usr/bin/env python3
"""Gate: is every static asset cache-busted by CONTENT HASH, not by hand?

Why this exists
---------------
Static files are served by StaticFiles, which sends ETag/Last-Modified, so a
browser holding a cached copy will NOT refetch an unchanged URL. Something in
the URL has to change when the file changes.

This used to be a hand-maintained "?v=N" in each template, and this script
used to check that N was bumped whenever the file changed. That design failed
twice in practice, for the same reason both times: the bump is a separate
action from the edit, so it can be forgotten, and when it is, the browser
silently runs the previous bundle. The symptom is indistinguishable from a
logic bug -- a fix that "does not work", stack traces pointing at code that no
longer exists, and no way to tell from the page which version is loaded.

The number is now DERIVED. `asset()` in app/assets.py hashes the file's contents
and appends the digest, so the URL cannot disagree with the file. This gate no
longer checks that anyone remembered anything; it checks that the mechanism is
still in place and that nothing has drifted back to a hard-coded version.

What it checks
--------------
  1. No template contains a hard-coded ?v= on a /static/ URL.
  2. Every /static/ reference in a template goes through {{ asset(...) }}.
  3. Every path handed to asset() actually exists under static/.
  4. app/main.py still registers the asset() global.

Usage
-----
    python scripts/check_asset_versions.py

Exit 0 = the mechanism is intact. Exit 1 = something regressed.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TEMPLATES = ROOT / "templates"
STATIC = ROOT / "static"

# A literal /static/... URL carrying a hand-written version.
HARDCODED = re.compile(r'["\']/static/[^"\']*\?v=[^"\']*["\']')
# A literal /static/... URL in markup (any form).
LITERAL_STATIC = re.compile(r'(?:src|href)\s*=\s*["\'](/static/[^"\']+)["\']')
# asset('...') / asset("...")
ASSET_CALL = re.compile(r'asset\(\s*["\']([^"\']+)["\']\s*\)')


def main() -> int:
    problems: list[str] = []
    checked = 0
    asset_refs = 0

    for tpl in sorted(TEMPLATES.rglob("*.html")):
        rel = tpl.relative_to(ROOT).as_posix()
        text = tpl.read_text(encoding="utf-8")
        checked += 1

        # Strip HTML comments so prose about "?v=" does not trip the gate.
        body = re.sub(r"<!--.*?-->", "", text, flags=re.S)

        for m in HARDCODED.finditer(body):
            problems.append(
                f"{rel}: hard-coded cache-buster {m.group(0)}\n"
                f"    -> use {{{{ asset('js/whatever.js') }}}} instead"
            )

        for m in LITERAL_STATIC.finditer(body):
            problems.append(
                f"{rel}: literal static URL {m.group(1)} (no content hash)\n"
                f"    -> use {{{{ asset('{m.group(1)[len('/static/'):]}') }}}}"
            )

        for m in ASSET_CALL.finditer(body):
            asset_refs += 1
            target = STATIC / m.group(1)
            if not target.is_file():
                problems.append(
                    f"{rel}: asset('{m.group(1)}') does not exist under static/"
                )

    main_py = (ROOT / "app" / "main.py").read_text(encoding="utf-8")
    # The IMPLEMENTATION moved to app/assets.py so a render gate could import
    # it without pulling in every router; main.py still does the registering.
    # Two files now, and this checks the one that owns each half — reading
    # main.py for the hashing was what made this gate fail on a move that
    # changed no behaviour at all.
    assets_py = (ROOT / "app" / "assets.py").read_text(encoding="utf-8")
    if 'templates.env.globals["asset"]' not in main_py:
        problems.append(
            "app/main.py: the asset() Jinja global is no longer registered — "
            "every {{ asset(...) }} call would raise at render time"
        )
    if "hashlib.sha256" not in assets_py:
        problems.append(
            "app/assets.py: asset() no longer hashes file contents — "
            "the cache-buster would stop tracking the file"
        )

    print(f"templates checked : {checked}")
    print(f"asset() refs      : {asset_refs}")
    if problems:
        print(f"\nPROBLEMS ({len(problems)}):\n")
        for p in problems:
            print("  " + p)
        return 1
    print("\nOK — every static asset is cache-busted by content hash.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
