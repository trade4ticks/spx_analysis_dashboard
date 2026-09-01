"""Run every gate and report one table of exit codes.

WHY THIS EXISTS. The gates were being run as ten separate shell commands with
`echo "name=$?"` after each, which had two costs. Every invocation was a
slightly different string, so the permission allow-list accumulated ~180
one-off entries and still prompted. And "all gates pass" was ten results read
one at a time rather than a single verifiable answer, which is exactly the
shape of claim that gets asserted without being checked -- it has been wrong
here before, when `cmd | tail -1 && echo OK` reported success because a
pipeline's exit code is its LAST stage's.

So: one command, one exit code, and a table that names what ran.

SKIPPING IS LOUD. A gate that cannot run on this host is not a gate that
passed, and the whole point of these scripts is that silent success is the
failure mode. Anything skipped is listed with its reason, counted separately,
and the summary says so in words.

    python scripts/gates.py              every gate
    python scripts/gates.py scalp        only gates whose name contains 'scalp'
    python scripts/gates.py --live       pass --live to the harnesses that take it
    python scripts/gates.py --slow       include the slow ones (off by default)
"""
from __future__ import annotations

import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable


class Gate:
    def __init__(self, name, argv, note="", slow=False, live_flag=False):
        self.name = name
        self.argv = argv
        self.note = note
        self.slow = slow
        self.live_flag = live_flag


def _s(script, *extra, **kw):
    return Gate(script.replace(".py", ""),
                [PY, str(ROOT / "scripts" / script), *extra], **kw)


GATES = [
    # ── syntax and references ────────────────────────────────────────────
    _s("check_alpine_syntax.py", note="every x-* expression parses"),
    _s("check_alpine_refs.py",
       note="every Alpine call resolves to a component member"),
    _s("check_asset_versions.py", note="static assets are content-hashed"),
    Gate("node --check", [], note="the page JS parses"),   # filled in below

    # ── the equity IV page ───────────────────────────────────────────────
    _s("equity_iv_dryrun.py", note="85+ endpoint cases, SQL, contamination"),
    _s("check_tenor_retarget.py",
       note="JS/Python retarget parity, presets, universe stems"),

    # ── the scalp page ───────────────────────────────────────────────────
    _s("scalp_dryrun.py", note="meta/health/candidates/calibration",
       live_flag=True),
    _s("check_scalp_fills.py", note="the Schwab statement parser"),
    _s("check_scalp_metrics.py", note="no hardcoded metric names"),

    # ── infrastructure ───────────────────────────────────────────────────
    _s("check_pool_wiring.py", note="pools bind their module-level names"),
    _s("check_vendored.py", note="vendored files match their source"),

    # ── slower, and not usually what changed ─────────────────────────────
    _s("check_chart_contract.py", slow=True,
       note="chart modules route through window.FactorCharts"),
    _s("check_routes_smoke.py", slow=True, note="every route imports"),

    # ── listed so they appear as SKIP rather than not appearing ──────────
    #
    # A gate this runner cannot invoke has to be VISIBLE. Leaving them out of
    # the list entirely would make the table complete-looking and wrong, which
    # is the failure mode every script in it exists to prevent.
    _s("check_template_render.py", slow=True),
    _s("check_grid_equivalence.py", slow=True),
]

JS_FILES = ["static/js/equity_iv.js", "static/js/equities_scalp.js"]

# A gate can fail because THIS HOST lacks something rather than because the
# code is wrong. Those are reported as ENV, never as PASS -- but also not as
# FAIL, because a runner that is permanently red is one nobody reads.
#
# Narrow on purpose: each entry names the exact module, so a genuine
# ImportError in our own code still fails.
ENV_GAPS = {
    "check_routes_smoke": ("No module named 'anthropic'",
                           "anthropic is not installed on this host"),
}

# Gates that need an argument this runner does not supply. Listed rather than
# discovered, so a gate that starts needing one does not quietly become a skip.
NEEDS_ARGS = {
    "check_template_render": "needs --ref GITREF — run it directly when a "
                             "template changed, comparing against a git ref",
    "check_grid_equivalence": "needs --sweep SWEEP — a targeted comparison, "
                              "run by hand against a specific sweep",
}


def run_one(g: Gate, live: bool):
    t0 = time.monotonic()
    if g.name == "node --check":
        outs, codes = [], []
        for f in JS_FILES:
            p = subprocess.run(["node", "--check", str(ROOT / f)],
                               capture_output=True, text=True, cwd=ROOT,
                               encoding="utf-8", errors="replace")
            codes.append(p.returncode)
            if p.returncode:
                outs.append(f"{f}: {(p.stderr or '').strip().splitlines()[:1]}")
        return g, max(codes or [0]), "\n".join(outs), time.monotonic() - t0, ""

    argv = list(g.argv)
    if live and g.live_flag:
        argv.append("--live")
    # UTF-8 explicitly: these scripts print em-dashes and sigma, and the
    # default console codepage on Windows turns them into mojibake in the
    # captured text even though they render correctly when run directly.
    p = subprocess.run(argv, capture_output=True, text=True, cwd=ROOT,
                       encoding="utf-8", errors="replace")
    # stdout and stderr kept APART. A gate's summary is its last stdout line,
    # and several of these configure logging, so a stray WARNING on stderr
    # would otherwise become the line the table reports.
    return g, p.returncode, (p.stdout or ""), time.monotonic() - t0, (p.stderr or "")


def last_line(text: str) -> str:
    lines = [l.rstrip() for l in text.splitlines() if l.strip()]
    return lines[-1] if lines else ""


def main() -> int:
    args = [a for a in sys.argv[1:]]
    live = "--live" in args
    slow = "--slow" in args
    pats = [a for a in args if not a.startswith("--")]

    gates = [g for g in GATES if slow or not g.slow]
    if pats:
        gates = [g for g in gates if any(p in g.name for p in pats)]
    if not gates:
        print(f"no gate matches {pats}")
        return 2

    with ThreadPoolExecutor(max_workers=6) as ex:
        results = list(ex.map(lambda g: run_one(g, live), gates))
    results.sort(key=lambda r: GATES.index(r[0]))

    width = max(len(g.name) for g, *_ in results)
    passed = failed = skipped = 0
    failures = []

    print()
    for g, code, out, secs, err in results:
        status, detail = "PASS", g.note
        if g.name in NEEDS_ARGS:
            status, detail = "SKIP", NEEDS_ARGS[g.name]
        elif code != 0:
            gap = ENV_GAPS.get(g.name)
            if gap and gap[0] in (out + err):
                status, detail = "ENV ", gap[1]
            else:
                status, detail = "FAIL", (last_line(out) or last_line(err)
                                          or f"exit {code}")
                failures.append((g.name, (out + err)))

        if status == "PASS":
            passed += 1
            summary = last_line(out)
            detail = summary if summary else detail
        elif status == "FAIL":
            failed += 1
        else:
            skipped += 1

        print(f"  {status}  {g.name:<{width}}  {secs:5.1f}s  {detail[:96]}")

    # The full output of anything that failed, after the table, so the table
    # stays readable and the detail is still one screen away.
    for name, out in failures:
        print(f"\n{'─' * 70}\n{name}\n{'─' * 70}")
        print(out.rstrip()[:4000])

    print()
    bits = [f"{passed} passed"]
    if failed:
        bits.append(f"{failed} FAILED")
    if skipped:
        bits.append(f"{skipped} could not run")
    print("  " + ", ".join(bits))
    if skipped:
        # Named again at the bottom. A skip mentioned once in a table is a
        # skip that gets read as a pass.
        for g, code, out, _, err in results:
            if g.name in NEEDS_ARGS or (
                    code != 0 and ENV_GAPS.get(g.name)
                    and ENV_GAPS[g.name][0] in (out + err)):
                print(f"    not run: {g.name}")
    if not slow:
        n = sum(1 for g in GATES if g.slow)
        print(f"    {n} slow gates omitted; add --slow to include them")
    print()
    return 1 if failed else 0


sys.exit(main())
