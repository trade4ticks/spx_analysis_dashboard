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
    python scripts/gates.py --deploy     a gate that could not run on this host
                                         FAILS the run instead of being listed
    python scripts/gates.py --vps --deploy [--ref GITREF]
                                         ONLY the gates meant for the VPS (below);
                                         --ref defaults to ORIG_HEAD, the commit
                                         before the last `git pull`

WHERE GATES RUN. The VPS deliberately lacks node, the sibling checkouts, and
Postgres server binaries (initdb/pg_ctl) -- installing a server package there
would start a second cluster beside the live database. So:

  dev machine   everything except the vps-only pair; in particular
                check_oo_backtest (node, source checkout) and
                check_oo_market_sql (initdb). Run before pushing.
  VPS           check_routes_smoke (the real app, real .env, real DB) and
                check_template_render --report-diffs --ref ORIG_HEAD. Run after
                `git pull`, via --vps. Nothing else is selected there, so a
                deploy check cannot fail on tooling that is absent by design.

VPS SETUP, once. The flow is: pull as root, gate as an unprivileged user
(initdb-style tools refuse root, and a gate has no business running as root).

    sudo useradd --system --create-home --shell /bin/bash gates
    # The repo is owned by root, so git refuses to read it as `gates`
    # ("detected dubious ownership") and check_template_render cannot export
    # the ORIG_HEAD templates. Trust this one path for that user only:
    sudo -u gates git config --global --add safe.directory /spx_analysis_dashboard
    # The repo and .venv must be readable by `gates`. So must .env, for
    # check_routes_smoke: app/db.py calls load_dotenv(), and python-dotenv
    # opens a .env that EXISTS without checking it is readable -- a 600 root
    # file raises PermissionError and the app does not import. Grant the
    # group read (this does give `gates` the DB credentials, which the smoke
    # test's real-database checks need anyway):
    sudo chgrp gates /spx_analysis_dashboard/.env && sudo chmod 640 /spx_analysis_dashboard/.env

Then, on every deploy:

    cd /spx_analysis_dashboard && git pull
    sudo systemctl restart spx-dashboard.service       # never spx-live
    sudo -u gates .venv/bin/python scripts/gates.py --vps --deploy

A gate declared can_skip exits EXIT_SKIPPED (3) when this host cannot run it
(no Postgres binaries, running as root). That is reported as SKIP, never PASS.
Under --deploy it is a FAIL: a deploy check that did not exercise the SQL has
not checked the SQL.
"""
from __future__ import annotations

import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable


EXIT_SKIPPED = 3


class Gate:
    def __init__(self, name, argv, note="", slow=False, live_flag=False, can_skip=False, vps=False):
        self.name = name
        self.argv = argv
        self.note = note
        self.slow = slow
        self.live_flag = live_flag
        # Only a gate that declares it may exit EXIT_SKIPPED has 3 read as a
        # skip; for any other gate a 3 is an ordinary failure.
        self.can_skip = can_skip
        # Selected by --vps: meant to run on the deployed host after a pull.
        self.vps = vps


def _s(script, *extra, **kw):
    return Gate(script.replace(".py", ""),
                [PY, str(ROOT / "scripts" / script), *extra], **kw)


GATES = [
    # ── syntax and references ────────────────────────────────────────────
    _s("check_alpine_syntax.py", note="every x-* expression parses"),
    _s("check_alpine_refs.py",
       note="every Alpine call resolves to a component member"),
    _s("check_asset_versions.py", note="static assets are content-hashed"),
    # One partial, seventeen pages. A page that falls out of the nav is
    # unreachable while every route still answers, so nothing else notices.
    _s("check_nav.py",
       note="every page is in exactly one category, and marked in it"),
    # Renders each page with the REAL asset() and looks at the OUTPUT. The
    # four gates above it all passed while a template shipped a script tag
    # with an empty src — none of them looked at rendered markup.
    _s("check_rendered_assets.py",
       note="every rendered src/href resolves; tags balance"),
    # The OTHER direction from check_alpine_refs: a handler with no caller is
    # what a panel deleted from its template leaves behind.
    _s("check_orphaned_handlers.py",
       note="no component method is left with nothing calling it"),
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
    # The pane accepted a constraint, chipped it, and never sent it. Only
    # driving the shipped page JS can see what the request carried.
    _s("check_scalp_filters.py", can_skip=True,
       note="pane constraints reach the request"),
    # Replay reads PARQUET, so its gate fabricates a session instead: which
    # minute a trade lands in, where the print/candle switch falls, and
    # whether the payload is valid JSON are all decidable without the store.
    _s("check_replay.py", note="replay candles, mode switch, readout"),
    # Runs the shipped component in node: a colour threshold must never
    # change which points draw or how far the axes reach.
    _s("check_scatter_invariants.py",
       note="colour thresholds cannot change membership or extent"),

    # ── the OO/Mesosim backtest page ─────────────────────────────────────
    # The exchange calendar is the authority on whether a day was a session,
    # so a wrong answer here is wrong in the rollup, the deployment axis,
    # coverage, the artifact report and the freshness check at once. Pure
    # Python and offline, so it runs on the VPS too.
    _s("check_market_calendar.py", note="NYSE sessions and expected bars",
       vps=True),
    # It bins in the browser against edges defined in Python; a disagreement
    # about which side of an edge a value falls on is invisible on screen.
    _s("check_oo_backtest.py", can_skip=True,
       note="JS binning == pd.cut; parsers; dev machine (node, source checkout)"),
    # Starts a throwaway Postgres and runs the shipped index_ohlc SQL: early
    # closes, 'NaN' bars, the prior-session row, the entry bar's open.
    # The portfolio page loads SAVED strategies through a cached parse. The
    # interesting failure is a cache that serves what the parser would no
    # longer produce, so the gate compares payloads, not frames.
    _s("check_portfolio.py",
       note="a cached parse produces the same payload as a fresh one"),
    # The page DRIVEN BY CLICKING, in a real browser. Every source-level
    # check passed while the Filters button was disabled and did nothing;
    # only clicking it could see that. Skips where there is no browser.
    _s("check_portfolio_ui.py", can_skip=True,
       note="the buttons a person presses actually do their thing"),
    _s("check_oo_market_sql.py", can_skip=True,
       note="market SQL on a temp cluster; dev machine only (needs initdb)"),
    # Strategy Signal: the decision rule, the percentile, the index/surface
    # clock and the config store, offline; then the page clicked in Edge
    # against the real router's output over fabricated bars.
    _s("check_strategy_signal.py",
       note="decision, percentile, one clock, one fetch per source, store"),
    _s("check_strategy_signal_ui.py", can_skip=True,
       note="cards, charts, Add/Save/Edit clicked; console clean"),

    # ── the live tape ────────────────────────────────────────────────────
    _s("check_live_hub.py",
       note="no aggregation, caps hold, resubscribe on reconnect"),
    # The third tier on the same upstream socket, and the page that is only
    # allowed to watch. Its worst failure is silent: an unsubscribe taken out
    # from under another tier looks like a symbol that went quiet.
    _s("check_wall.py",
       note="tiers share one socket; the cursor is a count; no broker"),
    # The band must hold still under a flat tape, and a rate must be a rate.
    _s("check_live_axis.py",
       note="price band steps only at the edge; trades/min is a fixed minute"),
    # The chart must not move under the cursor. Anything that appears on its
    # own belongs in an overlay, not in the flow above the canvas.
    _s("check_live_layout.py",
       note="no conditional block reflows the chart; controls are dark"),
    # A pane must not act on an order another application placed, and must
    # not stay blocked long after the broker has answered.
    _s("check_live_reconcile.py",
       note="primaryOrder refuses foreign orders; the unresolved window is 5s"),
    # Every way the arrival comparison can be wrong is invisible on screen.
    _s("check_arrival_norm.py",
       note="the 15-minute bucket is right at its edges; today is excluded"),
    # The one module where being wrong costs money rather than time.
    _s("check_broker.py",
       note="three switches, flatten cancels first, guards bound the end"),
    # The second broker: a socket instead of an API, so the failures are
    # different ones. Pure Python against a fake socket — no node, no DAS, no
    # network — so it runs anywhere, the VPS included.
    _s("check_das.py", vps=True,
       note="an unknown status draws; the token is the match; no market data"),

    # ── infrastructure ───────────────────────────────────────────────────
    _s("check_pool_wiring.py", note="pools bind their module-level names"),
    _s("check_vendored.py", note="vendored files match their source"),

    # ── slower, and not usually what changed ─────────────────────────────
    _s("check_chart_contract.py", slow=True,
       note="chart modules route through window.FactorCharts"),
    _s("check_routes_smoke.py", slow=True, vps=True, note="every route imports"),

    # ── listed so they appear as SKIP rather than not appearing ──────────
    #
    # A gate this runner cannot invoke has to be VISIBLE. Leaving them out of
    # the list entirely would make the table complete-looking and wrong, which
    # is the failure mode every script in it exists to prevent.
    _s("check_template_render.py", slow=True, vps=True),
    _s("check_grid_equivalence.py", slow=True),
]

JS_FILES = ["static/js/equity_iv.js", "static/js/equities_scalp.js",
            "static/js/equities_live.js", "static/js/oo_backtest.js"]

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


def _with_ref(g: Gate, ref: str, report: bool) -> Gate:
    """check_template_render with its --ref supplied (and, for a deploy,
    --report-diffs: an intended template change is listed, not failed)."""
    if g.name != "check_template_render":
        return g
    argv = [*g.argv, "--ref", ref] + (["--report-diffs"] if report else [])
    out = Gate(g.name, argv, note=f"rendered vs {ref}", slow=g.slow, vps=g.vps)
    out.ref_supplied = True
    return out


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
    deploy = "--deploy" in args
    vps = "--vps" in args
    ref = None
    if "--ref" in args:
        i = args.index("--ref")
        if i + 1 >= len(args):
            print("--ref needs a git ref")
            return 2
        ref = args.pop(i + 1)
        args.pop(i)
    if vps and ref is None:
        ref = "ORIG_HEAD"
    pats = [a for a in args if not a.startswith("--")]

    if vps:
        gates = [g for g in GATES if g.vps]
    else:
        gates = [g for g in GATES if slow or not g.slow]
    if ref:
        gates = [_with_ref(g, ref, report=vps) for g in gates]
    if pats:
        gates = [g for g in gates if any(p in g.name for p in pats)]
    if not gates:
        print(f"no gate matches {pats}")
        return 2

    with ThreadPoolExecutor(max_workers=6) as ex:
        results = list(ex.map(lambda g: run_one(g, live), gates))
    # By NAME: _with_ref() hands back a new Gate for template_render, which is
    # not the object in GATES.
    order = {g.name: i for i, g in enumerate(GATES)}
    results.sort(key=lambda r: order[r[0].name])

    width = max(len(g.name) for g, *_ in results)
    passed = failed = skipped = 0
    failures = []

    print()
    for g, code, out, secs, err in results:
        status, detail = "PASS", g.note
        if g.name in NEEDS_ARGS and not getattr(g, "ref_supplied", False):
            status, detail = "SKIP", NEEDS_ARGS[g.name]
        elif g.can_skip and code == EXIT_SKIPPED:
            if deploy:
                status, detail = "FAIL", "could not run on this host (--deploy): " + (last_line(out) or "skipped")
                failures.append((g.name, (out + err)))
            else:
                status, detail = "SKIP", last_line(out) or "could not run on this host"
        elif code != 0:
            gap = ENV_GAPS.get(g.name)
            if gap and gap[0] in (out + err) and deploy:
                # Same rule as a skip: a deploy check that could not run a
                # gate has not checked what that gate checks.
                status, detail = "FAIL", f"could not run on this host (--deploy): {gap[1]}"
                failures.append((g.name, (out + err)))
            elif gap and gap[0] in (out + err):
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
            if (g.name in NEEDS_ARGS and not getattr(g, "ref_supplied", False)) or (g.can_skip and code == EXIT_SKIPPED) or (
                    code != 0 and ENV_GAPS.get(g.name)
                    and ENV_GAPS[g.name][0] in (out + err)):
                print(f"    not run: {g.name}")
        if not deploy and any(g.can_skip and code == EXIT_SKIPPED for g, code, *_ in results):
            print("    (a deploy check must not skip these: rerun with --deploy as a non-root user)")
    if not slow and not vps:
        n = sum(1 for g in GATES if g.slow)
        print(f"    {n} slow gates omitted; add --slow to include them")
    print()
    return 1 if failed else 0


sys.exit(main())
