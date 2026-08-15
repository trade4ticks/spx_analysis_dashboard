#!/usr/bin/env python3
"""Regression gate: do the page templates still render the same bytes?

Renders every page template with the same Jinja configuration app/main.py
uses, at a git ref AND in the working tree, and compares the two rendered
outputs. Any divergence is printed with the offset and surrounding context.

Both sides are rendered. That matters: once templates contain Jinja tags,
the raw file is no longer the expected output, so a render-vs-raw-file
comparison only works for a pre-Jinja baseline. Rendering both sides makes
the gate valid at any ref, and it degrades gracefully — at a ref whose
templates have no Jinja, rendering is the identity function and this reduces
to the original file comparison.

Two uses
--------
Verifying a refactor that must not change output (the Jinja conversion):

    python scripts/check_template_render.py --ref 1199cad

    1199cad is the last commit before templates were rendered at all; there,
    render == raw file, so a pass proves the conversion changed zero bytes.

Ongoing regression while editing templates, macros, or includes:

    python scripts/check_template_render.py --ref HEAD

    Confirms an edit changed exactly the pages intended and nothing else.
    Expected to FAIL on any deliberate markup change — read the diff, confirm
    it is what you meant, and move the ref forward.

What this does NOT cover
------------------------
It never constructs a request or touches FastAPI, so it cannot see how
main.py CALLS the renderer. A wrong TemplateResponse signature returns 500 on
every page while this reports a clean pass — that happened. Run
scripts/check_routes_smoke.py as well; the two gates cover different
failures.

Exit 0 = every page renders identically at both sides. Exit 1 = divergence,
a render error, or a page that exists on only one side.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import jinja2

ROOT = Path(__file__).resolve().parent.parent
TPL_DIR = ROOT / "templates"

# Partials are included BY pages and never rendered standalone.
PARTIAL_PREFIX = "_"


def _env(directory: Path) -> jinja2.Environment:
    """Must mirror app/main.py's Jinja2Templates configuration exactly.

    keep_trailing_newline is load-bearing: Jinja drops the final newline by
    default, which would make every page differ by one byte. If main.py's
    configuration changes, change it here too or this gate silently tests
    something other than what is served.
    """
    return jinja2.Environment(
        loader=jinja2.FileSystemLoader(str(directory)),
        autoescape=True,
        keep_trailing_newline=True,
    )


def _pages(directory: Path) -> list[str]:
    return sorted(p.name for p in directory.glob("*.html")
                  if not p.name.startswith(PARTIAL_PREFIX))


def _git(*args: str) -> str:
    return subprocess.run(("git", *args), cwd=ROOT, check=True,
                          capture_output=True, text=True).stdout


def _export_templates(ref: str, dest: Path) -> None:
    """Materialise templates/ as it existed at `ref` into `dest`.

    Every file is exported, not just pages: partials must be present for the
    includes and imports in the pages to resolve.
    """
    listing = _git("ls-tree", "-r", "--name-only", ref, "templates/")
    names = [ln.strip() for ln in listing.splitlines() if ln.strip().endswith(".html")]
    if not names:
        raise SystemExit(f"ERROR: ref {ref!r} has no templates/*.html")
    for name in names:
        blob = subprocess.run(("git", "show", f"{ref}:{name}"), cwd=ROOT,
                              check=True, capture_output=True)
        out = dest / Path(name).name
        out.write_bytes(blob.stdout)


def _render_all(directory: Path) -> dict[str, str | Exception]:
    env = _env(directory)
    out: dict[str, str | Exception] = {}
    for name in _pages(directory):
        try:
            out[name] = env.get_template(name).render()
        except Exception as e:  # noqa: BLE001 — report per page, keep sweeping
            out[name] = e
    return out


def verify(ref: str) -> int:
    with tempfile.TemporaryDirectory() as td:
        base_dir = Path(td)
        _export_templates(ref, base_dir)
        base = _render_all(base_dir)
    head = _render_all(TPL_DIR)

    names = sorted(set(base) | set(head))
    ok = bad = 0
    total = 0
    for name in names:
        b, h = base.get(name), head.get(name)
        if b is None:
            print(f"  {name:34s} NEW page (absent at {ref}) — not verified")
            continue
        if h is None:
            print(f"  {name:34s} REMOVED from working tree (present at {ref})")
            bad += 1
            continue
        for side, val in ((ref, b), ("working tree", h)):
            if isinstance(val, Exception):
                print(f"  {name:34s} RENDER ERROR at {side}: "
                      f"{type(val).__name__}: {val}")
        if isinstance(b, Exception) or isinstance(h, Exception):
            bad += 1
            continue
        if b == h:
            ok += 1
            total += len(h)
            continue
        bad += 1
        i = next((k for k, (x, y) in enumerate(zip(h, b)) if x != y),
                 min(len(h), len(b)))
        print(f"  {name:34s} DIFFERS at char {i}")
        print(f"      working tree: {h[max(0, i - 40):i + 60]!r}")
        print(f"      {ref:<12s}: {b[max(0, i - 40):i + 60]!r}")

    print()
    print(f"byte-identical vs {ref}: {ok}/{ok + bad} pages "
          f"({total:,} bytes verified)")
    return 1 if bad else 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ref", required=True, metavar="GITREF",
                    help="git ref to compare rendered output against "
                         "(e.g. HEAD, a tag, or 1199cad for the pre-Jinja baseline)")
    return verify(ap.parse_args().ref)


if __name__ == "__main__":
    raise SystemExit(main())
