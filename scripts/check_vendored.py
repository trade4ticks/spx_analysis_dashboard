"""Verify vendored files still match the source they were copied from.

Some definitions are owned by the Open_Interest project and used here. The
older arrangement -- app/split_factors.py, app/trade_path_rules.py -- copies a
TRIMMED subset with a "keep in sync" comment at the top, which is a promise
rather than a check: nothing notices when the source moves.

app/metrics_config.py is copied VERBATIM instead, precisely so this script can
diff it. Byte-identical means no local header, so the provenance lives here
rather than in the file.

Why this matters more than for the other two: metrics_config declares itself
THE SINGLE SOURCE for BASELINE_SNAPSHOT and BASELINE_MIN_N, and the reason it
says so is that two copies of a baseline definition previously produced two
divergent z estimators -- the dashboard deriving one thing while the metrics
table stored another. A silent copy of that file would recreate exactly the
problem it exists to prevent.

The source lives outside this repo, so it is not always reachable -- the VPS
has no Open_Interest checkout. A missing source SKIPS rather than fails: this
gate is for the machine where the two live side by side, and a check that
cannot run is not a check that failed.
"""
import io
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# The sibling checkout, overridable for a host that keeps it elsewhere.
OI_ROOT = Path(os.getenv("OI_PROJECT_ROOT", ROOT.parent / "Open_Interest"))

# (vendored path, source path relative to OI_ROOT)
VERBATIM = [
    ("app/metrics_config.py", "lib/metrics_config.py"),
]


def main() -> int:
    if not OI_ROOT.is_dir():
        print(f"SKIP: no Open_Interest checkout at {OI_ROOT}")
        print("      (set OI_PROJECT_ROOT to point at one)")
        return 0

    bad = 0
    for rel, src_rel in VERBATIM:
        here = ROOT / rel
        there = OI_ROOT / src_rel
        if not here.is_file():
            print(f"  MISSING vendored file: {rel}")
            bad += 1
            continue
        if not there.is_file():
            print(f"  SKIP {rel}: source absent at {there}")
            continue
        a = here.read_bytes()
        b = there.read_bytes()
        if a == b:
            print(f"  OK {rel} == {src_rel} ({len(a)} bytes)")
            continue
        bad += 1
        print(f"\n  DRIFT {rel} != {src_rel}")
        import difflib
        la = io.open(here, encoding="utf-8").read().splitlines()
        lb = io.open(there, encoding="utf-8").read().splitlines()
        diff = list(difflib.unified_diff(la, lb, "vendored", "source", lineterm=""))
        for line in diff[:40]:
            print("    " + line)
        if len(diff) > 40:
            print(f"    ... {len(diff) - 40} more diff lines")
        print(f"\n    Re-vendor with:  copy {there} -> {here}")

    print(f"\nvendored files checked: {len(VERBATIM)}, drifted: {bad}")
    return 1 if bad else 0


sys.exit(main())
