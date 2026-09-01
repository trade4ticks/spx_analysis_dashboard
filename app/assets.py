"""Cache-busting static URLs, off a CONTENT HASH.

LIVED IN app/main.py UNTIL A TEMPLATE SHIPPED WITH AN EMPTY SCRIPT TAG.
Nothing could check the rendered pages, because rendering one means importing
`asset`, and importing it meant importing app.main — which pulls in every
router and their dependencies. So the render gate rendered with `asset`
UNDEFINED, every script src came out empty in its output too, and a page that
emitted `<script src=` looked exactly like a page that did not.

Extracted here so scripts/check_rendered_assets.py can import the REAL
function. A second copy in the gate would have been worse than no gate: it
would pass against a definition the app does not use.

Templates used to hard-code "?v=NN" and a human had to remember to bump it on
every JS edit. Forgetting once costs a debugging round -- the browser keeps
serving the previous bundle, the symptom looks like a code bug, and nothing
about the page says which version is running. That happened, so the number is
derived rather than maintained.

Hash is of file CONTENT, not mtime: a redeploy that rewrites files without
changing them must not invalidate a warm cache, and an edit that happens to
preserve mtime must not be missed.

Cached in-process, keyed on (mtime_ns, size), so the common case is a stat
rather than a read. A dev editing a file gets a new hash on the next request
without a restart.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent          # project root

_ASSET_CACHE: dict = {}


def asset(path: str) -> str:
    """URL for a file under /static, with a content-hash cache-buster.

    Usage in a template:  <script src="{{ asset('js/app.js') }}"></script>
    Unknown files degrade to an unversioned URL rather than raising -- a
    missing asset should surface as a 404 in the network tab, not a 500 on
    the whole page. scripts/check_rendered_assets.py is what turns that 404
    into a build failure instead of something to notice in a log.
    """
    rel = str(path).lstrip("/")
    full = BASE_DIR / "static" / rel
    try:
        st = full.stat()
    except OSError:
        return f"/static/{rel}"
    stamp = (st.st_mtime_ns, st.st_size)
    hit = _ASSET_CACHE.get(rel)
    if hit and hit[0] == stamp:
        return hit[1]
    try:
        digest = hashlib.sha256(full.read_bytes()).hexdigest()[:10]
    except OSError:
        return f"/static/{rel}"
    url = f"/static/{rel}?v={digest}"
    _ASSET_CACHE[rel] = (stamp, url)
    return url
