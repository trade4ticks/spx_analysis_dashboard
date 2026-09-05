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

app/trade_path_rules.py cannot be diffed either way: it is a TRIMMED copy that
takes `by_key` as a parameter where the source reads a module-global registry,
and it carries a local addition (include_exit_rule) the source does not have.
Copying the source whole would mean adopting that in-process registry as the
column source -- the exact hardcoding the catalog table exists to avoid -- so
the trim is deliberate and a byte diff will never pass.

That file is covered by a FINGERPRINT instead: the hash of the source function
it was copied from. This does NOT prove the copy is correct. It proves the
source has not moved since someone last read both files, which is the failure
that actually happened -- the source gained a per-horizon resolution filter
(c3fbd56) and the copy sat on `path_status = 'ok'` until a person noticed. A
fingerprint would have said so the same day.

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
    # The equities-scalp pipeline owns both of these. The dashboard must not
    # import from scalp/ -- `rm -rf scalp/` has to leave this app standing --
    # so they are copied, and copied WHOLE so this script can diff them.
    #
    # scalp_config carries DEFAULT_FILTERS and FILTER_RANGES, which are the
    # thresholds the page's sliders open on and the bounds they move through.
    # A trimmed copy of those would be a threshold sitting here at last
    # month's value while the pipeline used another, with nothing to notice.
    ("app/scalp_config.py", "scalp/config.py"),
    # scalp_metric_docs turns a metric name into its definition and its
    # anchor in METRICS.md, which is what makes every column header a link.
    # It also has to stay in step with the metric set for the same reason.
    ("app/scalp_metric_docs.py", "scalp/metric_docs.py"),
]


# ── Fingerprinted copies ────────────────────────────────────────────────────
#
# A trimmed copy cannot be byte-diffed, so what is checked instead is that the
# SOURCE FUNCTION it was derived from has not changed since the last sync.
#
# NEXT STEP (deliberately not done yet): have Open_Interest extract a DB-free
# `lib/trade_path_rules_core.py` -- SIDE_PRIORITY, HORIZON_RULE_KEY,
# CombineError and build_combine_sql taking `by_key` explicitly -- leaving the
# registry and the numpy exit machinery in the current module. This project
# then vendors the core VERBATIM, moves it into the list above, and this
# fingerprint entry goes away.
#
# Blocked until the registry expansion upstream has landed: extracting a module
# out of a file that is being rewritten to 143 rules is a merge conflict for no
# reason.
#
# `symbol` is resolved through the AST, not by line number, so an edit ANYWHERE
# ELSE in the source file does not trip this. The source is read at its
# COMMITTED HEAD, so work in flight upstream does not trip it either.
FINGERPRINT = [
    {
        "vendored": "app/trade_path_rules.py",
        "source":   "lib/trade_path_rules.py",
        "symbol":   "build_combine_sql",
        # The source commit this copy was last read against and mirrored from.
        "synced":   "c3fbd5650f9592316b7dff674ca2a1746545207c",
        "sha256":   "43af392f0310377542707004e19f507a14d7d367eb5795c523e5923db3c11c87",
    },
]


def _symbol_source(path, symbol, rel=None):
    """Source text of one top-level def/class, via the AST.

    Reads the source at its COMMITTED HEAD, not the working tree. Upstream is
    a working repository: fingerprinting whatever is in the buffer right now
    makes this fire on every in-flight edit, and a check that cries wolf while
    someone is mid-refactor is a check that gets bypassed. You re-sync against
    commits, so commits are what is tracked. Falls back to the working tree
    (and says so) when the source is not a git checkout.

    Returns None when the file will not parse or the symbol is gone. Both are
    reported as failures by the caller, because either means the thing being
    fingerprinted no longer exists in the shape that was copied.
    """
    import ast
    text = _git(["show", "HEAD:" + rel]) if rel else None
    if text is None:
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            return None
        print("    (source is not a git checkout - fingerprinting the working "
              "tree, which moves with uncommitted edits)")
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return None
    for node in tree.body:
        if (isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
                and node.name == symbol):
            return ast.get_source_segment(text, node)
    return None


def _git(args):
    """git output from the source checkout, or None if git is unavailable."""
    import subprocess
    try:
        r = subprocess.run(["git", "-C", str(OI_ROOT)] + args,
                           capture_output=True, timeout=15)
        if r.returncode != 0:
            return None
        out = r.stdout.decode("utf-8", "replace")
        return out if args[:1] == ["show"] else out.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def check_fingerprints() -> int:
    """0 if every fingerprint matches, else the number that drifted."""
    import hashlib
    bad = 0
    for fp in FINGERPRINT:
        here = ROOT / fp["vendored"]
        there = OI_ROOT / fp["source"]
        label = fp["vendored"] + " <- " + fp["source"] + "::" + fp["symbol"]
        if not here.is_file():
            print("  MISSING vendored file: " + fp["vendored"])
            bad += 1
            continue
        if not there.is_file():
            print("  SKIP " + label + ": source absent at " + str(there))
            continue

        seg = _symbol_source(there, fp["symbol"], fp["source"])
        if seg is None:
            bad += 1
            print()
            print("  GONE  " + label)
            print("    " + fp["symbol"] + " is no longer a top-level definition in")
            print("    the source, or the source no longer parses. The copy in")
            print("    " + fp["vendored"] + " was derived from it, so it is now")
            print("    derived from something that does not exist.")
            continue

        got = hashlib.sha256(seg.encode("utf-8")).hexdigest()
        if got == fp["sha256"]:
            print("  OK " + label
                  + " (" + str(len(seg.splitlines())) + " lines, " + got[:12] + ")")
            continue

        bad += 1
        # LOUD, and it says what to do. A tripwire whose message does not
        # explain itself gets bypassed, which is worse than not having one.
        head = _git(["rev-parse", "HEAD"]) or "<source HEAD sha>"
        print()
        print("  " + "=" * 70)
        print("  SOURCE MOVED: " + label)
        print("  " + "=" * 70)
        print("    expected sha256 " + fp["sha256"][:16])
        print("    found    sha256 " + got[:16])
        print()
        print("    " + fp["source"] + "::" + fp["symbol"] + " has changed upstream")
        print("    since " + fp["vendored"] + " was last synced against it.")
        print()
        print("    THIS DOES NOT MEAN THE COPY IS WRONG. It means nobody has read")
        print("    the two side by side since the source changed, which is the one")
        print("    thing a trimmed copy cannot tell you by itself.")
        print()
        print("    WHAT TO DO - in order:")
        print()
        print("      1. Read what changed upstream:")
        print("           git -C " + str(OI_ROOT) + " \\")
        print("               log --oneline " + fp["synced"][:12] + "..HEAD -- "
              + fp["source"])
        print("           git -C " + str(OI_ROOT) + " \\")
        print("               diff " + fp["synced"][:12] + "..HEAD -- " + fp["source"])
        print()
        print("      2. Decide whether the change touches the TRIMMED subset this")
        print("         project copied. " + fp["vendored"] + "'s header says what")
        print("         was copied and what was deliberately left out.")
        print()
        print("      3. If it does, mirror it - and verify it the way the last sync")
        print("         was verified: render the generated SQL before and after over")
        print("         representative selections, and confirm only the intended")
        print("         lines moved.")
        print()
        print("      4. Either way, update this entry once you have looked:")
        print("           scripts/check_vendored.py -> FINGERPRINT")
        print('             "synced": "' + head + '",')
        print('             "sha256": "' + got + '",')
        print()
        print("         Updating the hash WITHOUT doing step 2 silences the check,")
        print("         and is the only way to actually break this.")

        log = _git(["log", "--oneline", fp["synced"] + "..HEAD", "--", fp["source"]])
        if log:
            print()
            print("    upstream commits touching that file since the last sync:")
            for line in log.splitlines()[:12]:
                print("      " + line)
        print()
    return bad


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

    print()
    print("fingerprinted copies (trimmed -- source function hash, not a diff):")
    bad += check_fingerprints()

    print(f"\nvendored files checked: {len(VERBATIM)} verbatim + "
          f"{len(FINGERPRINT)} fingerprinted, drifted: {bad}")
    return 1 if bad else 0


sys.exit(main())
