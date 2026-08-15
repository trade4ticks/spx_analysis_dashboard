#!/usr/bin/env python3
"""Regression gate for the Jinja template conversion.

Renders every page template exactly as app/main.py does and compares the
result against a baseline snapshot of the pre-refactor files.

Why this exists
---------------
The 10 page templates used to be served with FileResponse — raw static HTML
with zero Jinja tags. Moving to Jinja2Templates so `_nav.html` and
`_macros.html` can be shared has one hard requirement: the rendered bytes
must not change. This script is the proof, and it is the gate to re-run after
ANY edit to a template, macro, or include.

Usage
-----
    # one-time, from a clean checkout BEFORE refactoring:
    python scripts/check_template_render.py --snapshot <dir>

    # thereafter, to verify rendering still matches that baseline:
    python scripts/check_template_render.py --baseline <dir>

Exit code 0 = every template renders byte-identically to its baseline.
Exit code 1 = at least one diverged; the first differing offset is printed
with context on both sides.

Note on keep_trailing_newline: Jinja drops the final newline by default,
which makes every page differ from its source by exactly one byte. main.py
sets keep_trailing_newline=True and so does this script — the two MUST stay
in agreement or this gate silently tests the wrong thing.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import jinja2

ROOT = Path(__file__).resolve().parent.parent
TPL_DIR = ROOT / "templates"

# Partials, not pages. They are included BY pages and never rendered alone,
# so they have no standalone baseline to compare against.
PARTIAL_PREFIX = "_"


def _env() -> jinja2.Environment:
    """Must mirror app/main.py's Jinja2Templates configuration exactly."""
    return jinja2.Environment(
        loader=jinja2.FileSystemLoader(str(TPL_DIR)),
        autoescape=True,
        keep_trailing_newline=True,
    )


def _pages() -> list[Path]:
    return sorted(p for p in TPL_DIR.glob("*.html")
                  if not p.name.startswith(PARTIAL_PREFIX))


def snapshot(dest: Path) -> int:
    dest.mkdir(parents=True, exist_ok=True)
    n = 0
    for p in _pages():
        shutil.copy2(p, dest / p.name)
        n += 1
    print(f"snapshot: {n} page templates -> {dest}")
    return 0


def verify(baseline: Path) -> int:
    if not baseline.is_dir():
        print(f"ERROR: baseline dir not found: {baseline}", file=sys.stderr)
        return 1
    env = _env()
    ok = bad = missing = 0
    total_bytes = 0
    for p in _pages():
        base = baseline / p.name
        if not base.is_file():
            print(f"  {p.name:34s} NO BASELINE (new page — add it to the snapshot)")
            missing += 1
            continue
        want = base.read_text(encoding="utf-8")
        try:
            got = env.get_template(p.name).render()
        except Exception as e:  # noqa: BLE001 — surface any template error
            print(f"  {p.name:34s} RENDER ERROR: {type(e).__name__}: {e}")
            bad += 1
            continue
        if got == want:
            ok += 1
            total_bytes += len(want)
            continue
        bad += 1
        i = next((k for k, (a, b) in enumerate(zip(got, want)) if a != b),
                 min(len(got), len(want)))
        print(f"  {p.name:34s} DIFFERS at char {i}")
        print(f"      rendered: {got[max(0, i - 40):i + 60]!r}")
        print(f"      baseline: {want[max(0, i - 40):i + 60]!r}")

    print()
    print(f"byte-identical: {ok}/{ok + bad + missing} pages "
          f"({total_bytes:,} bytes verified)")
    if missing:
        print(f"  {missing} page(s) had no baseline — not a failure, but unverified")
    return 1 if bad else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--snapshot", metavar="DIR",
                   help="copy current page templates to DIR as a new baseline")
    g.add_argument("--baseline", metavar="DIR",
                   help="verify rendering matches the baseline in DIR")
    a = ap.parse_args()
    return snapshot(Path(a.snapshot)) if a.snapshot else verify(Path(a.baseline))


if __name__ == "__main__":
    raise SystemExit(main())
