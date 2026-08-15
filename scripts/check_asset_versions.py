#!/usr/bin/env python3
"""Gate: did every changed static asset get its ?v= cache-buster bumped?

Why this exists
---------------
Static files are served by StaticFiles, which sends ETag/Last-Modified, so a
browser holding a cached copy will NOT refetch an unchanged URL. The ?v=N
query string in the template is the only thing that forces it. Ship a JS
change without bumping N and the browser keeps running the old file --
silently, and indistinguishably from a logic bug.

That is not hypothetical. factor_charts.js was created at v=1 and then
modified without a bump, so a browser kept serving the pre-fix module while
the deployed file on disk was correct. The symptom was a contract gate that
looked wrong when it was in fact right, and stack traces pointing at code
that no longer existed.

Checks every file under static/ that changed, finds every template
referencing it as <name>?v=N, and asserts N strictly increased. A changed
asset that no template references is reported but not failed -- shared
modules loaded by other means are legitimate.

Usage
-----
    python scripts/check_asset_versions.py                 # working tree vs HEAD
    python scripts/check_asset_versions.py --ref HEAD~1    # working tree vs a ref
    python scripts/check_asset_versions.py --commit <sha>  # audit one commit
    python scripts/check_asset_versions.py --range A..B    # audit a range

Exit 0 = every changed asset was bumped. Exit 1 = at least one was not.
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TPL = ROOT / "templates"


def _git(*args: str) -> str:
    return subprocess.run(("git", *args), cwd=ROOT, check=False,
                          capture_output=True, text=True, encoding="utf-8").stdout


def _versions_at(ref: str | None, name: str) -> dict[str, int]:
    """{template: version} for `name` at a git ref, or in the working tree."""
    out: dict[str, int] = {}
    for tpl in sorted(TPL.glob("*.html")):
        src = (tpl.read_text(encoding="utf-8") if ref is None
               else _git("show", f"{ref}:templates/{tpl.name}"))
        m = re.search(rf"{re.escape(name)}\?v=(\d+)", src)
        if m:
            out[tpl.name] = int(m.group(1))
    return out


def check(before: str, after: str | None, changed: list[str]) -> int:
    assets = [f for f in changed if f.startswith("static/")
              and f.rsplit(".", 1)[-1] in ("js", "css")]
    if not assets:
        print("no static .js/.css changes to check")
        return 0

    bad = 0
    for path in sorted(assets):
        name = Path(path).name
        old = _versions_at(before, name)
        new = _versions_at(after, name)

        # Checked unconditionally, not just when NO template versions the file.
        # A file versioned in ten templates and bare in one is the worse case:
        # it looks cache-busted everywhere you happen to look. The first
        # version of this check only ran when `new` was empty and would have
        # missed exactly that.
        unversioned = [t.name for t in sorted(TPL.glob("*.html"))
                       if re.search(rf'(?:href|src)="/static/[^"?]*{re.escape(name)}"',
                                    t.read_text(encoding="utf-8"))]
        if unversioned:
            bad += 1
            print(f"  {name:26s} CHANGED and referenced with NO ?v= in: "
                  + ", ".join(unversioned))

        if not new:
            # Distinguish "no template references it" (fine — shared modules
            # can load by other means) from "referenced with NO ?v= at all",
            # which is strictly worse than a stale buster: there is no URL
            # change possible, so the browser keeps its copy forever, through
            # hard-refresh included. theme.css was in exactly this state and
            # the earlier version of this gate could not see it, because it
            # only compared version numbers that existed.
            if not unversioned:
                print(f"  {name:26s} changed, referenced by no template — not checked")
            continue
        misses = [t for t, v in new.items() if old.get(t, -1) >= v]
        if misses:
            bad += 1
            for t in misses:
                o = old.get(t)
                print(f"  {name:26s} CHANGED but {t} still ?v="
                      f"{new[t]}" + (f" (was {o})" if o is not None else " (new ref)"))
        else:
            detail = ", ".join(f"{t} {old.get(t, '-')}->{v}" for t, v in new.items())
            print(f"  {name:26s} bumped  ({detail})")

    print()
    if bad:
        print(f"{bad} changed asset(s) not cache-busted — browsers will keep the "
              f"cached copy (an absent ?v= is worse than a stale one: no URL "
              f"change is possible at all)")
    else:
        print("every changed asset was bumped")
    return 1 if bad else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--ref", default="HEAD", help="compare the working tree against this ref")
    g.add_argument("--commit", help="audit a single commit")
    g.add_argument("--range", dest="rng", help="audit a commit range A..B")
    a = ap.parse_args()

    if a.commit:
        changed = _git("show", "--name-only", "--format=", "-r", a.commit).split()
        return check(f"{a.commit}^", a.commit, changed)
    if a.rng:
        before, after = a.rng.split("..")
        changed = _git("diff", "--name-only", a.rng).split()
        return check(before, after, changed)
    changed = _git("diff", "--name-only", a.ref).split()
    return check(a.ref, None, changed)


if __name__ == "__main__":
    raise SystemExit(main())
