"""The Equities Scalp page's filter pane reaches the server.

WHAT SHIPPED (reported 2026-09-24). The pane accepted a constraint, drew its
chip, took a number — and never put it in the request. `off_exchange_share >
0.40` left rows at 0.36 on screen and the pass count unmoved, and `> 40`
behaved identically because neither was ever asked about.

Every other part of that path was already right: the endpoint takes
`filters`, parses it, pulls the named metric into the pivot, evaluates it in
the same loop as the sliders and reports it as inert if it could not run. The
one missing line was in the browser.

It hid behind a second bug. Until the pane's ranges were fixed the day
before, every database-read metric had no range and its ≥/≤ buttons were
disabled, so a pane constraint could not be created at all — and the only
metric that escaped that, the derived `$ vol/min`, has a named slider, and
sliders were always sent. Two defects in a row, each concealing the other.

So this drives the SHIPPED page JS in node, with the network stubbed, and
reads the URL it builds. A source check ("does the file mention filters")
would have passed against the broken version, since the word appears
throughout.

WHERE IT RUNS: the development machine. Needs `node`.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
JS = ROOT / "static" / "js" / "equities_scalp.js"

EXIT_SKIPPED = 3
FAILS: list[str] = []


def check(cond, msg):
    if not cond:
        FAILS.append(msg)


# The component is built, its state set by hand, and loadCandidates() called.
# `scGetJson` is replaced after the file is evaluated, so what is measured is
# the URL the page would actually have requested.
DRIVER = r"""
const fs = require('fs');
let factory = null;
global.window = { addEventListener: () => {}, location: { search: '' },
                  devicePixelRatio: 1 };
global.document = {
  addEventListener: (ev, fn) => { if (ev === 'alpine:init') fn(); },
  getElementById: () => null, querySelector: () => null,
  querySelectorAll: () => [],
};
global.localStorage = { getItem: () => null, setItem: () => {},
                        removeItem: () => {} };
global.Alpine = { data: (_n, f) => { factory = f; } };
const src = fs.readFileSync(process.argv[1], 'utf8');
const tail = `
  const asked = [];
  scGetJson = async (url) => {
    asked.push(url);
    // Shaped like the endpoint's answer, so nothing downstream explodes and
    // the call gets as far as it would in a browser.
    return { rows: [], columns: [], col_ranges: {}, derived: [], rejected: {},
             constraints: [], inert_filters: [], n_pass: 0, n_total: 0,
             thresholds: {}, variant: {}, spark_dates: [] };
  };
  const c = factory();
  // The parts of the component the request builder reads. Set directly:
  // booting the whole page in node would be testing the fakes.
  c.meta = { connected: true, date: '2026-09-22', filters: { defaults: {} },
             metrics: [] };
  c.filterKeys = [];
  c.extraCols = [];
  c.activeRoleKeys = () => ['price'];
  c.geomInvalidate = () => {};
  c.renderGeometry = () => {};
  c.renderScatter = () => {};
  c.drawSpark = () => {};
  c.afterCandidates = c.afterCandidates || (() => {});
  const out = {};
  const run = async () => {
    // 1. Nothing in the pane: no filters parameter at all.
    c.custom = [];
    await c.loadCandidates();
    out.none = asked[asked.length - 1];
    // 2. One constraint, as the pane's buttons create it.
    c.custom = [{ key: 'off_exchange_share', op: 'min', value: 0.4 }];
    await c.loadCandidates();
    out.one = asked[asked.length - 1];
    // 3. Two, including a max, and a derived key.
    c.custom = [{ key: 'off_exchange_share', op: 'min', value: 0.4 },
                { key: 'spread_cents_tw', op: 'max', value: 12 },
                { key: 'dollar_vol_per_min', op: 'min', value: 1e6 }];
    await c.loadCandidates();
    out.many = asked[asked.length - 1];
    // 4. A constraint still being typed carries no number.
    c.custom = [{ key: 'off_exchange_share', op: 'min', value: null },
                { key: 'spread_cents_tw', op: 'max', value: '' }];
    await c.loadCandidates();
    out.empty = asked[asked.length - 1];
    // 5. Removing the last one removes the parameter with it.
    c.custom = [{ key: 'off_exchange_share', op: 'min', value: 0.4 }];
    await c.loadCandidates();
    c.custom = [];
    await c.loadCandidates();
    out.cleared = asked[asked.length - 1];
    globalThis.__out = out;
  };
  run().then(() => process.stdout.write(JSON.stringify(globalThis.__out)));
`;
eval(src + tail);
"""


def param(url: str, name: str) -> str | None:
    from urllib.parse import parse_qs, urlparse
    q = parse_qs(urlparse(url).query)
    v = q.get(name)
    return v[0] if v else None


def main() -> int:
    if shutil.which("node") is None:
        print("  SKIP  node is not installed — the shipped page JS was not run")
        return EXIT_SKIPPED
    p = subprocess.run(["node", "-e", DRIVER, str(JS)], capture_output=True,
                       text=True, encoding="utf-8")
    if p.returncode or not p.stdout.strip():
        print("  the page JS could not be driven in node:")
        print("   ", (p.stderr or "").strip()[-600:])
        return 1
    out = json.loads(p.stdout)

    # THE FAULT ITSELF: a constraint in the pane is a constraint in the
    # request.
    got = param(out["one"], "filters")
    check(got == "off_exchange_share:min:0.4",
          f"a pane constraint reached the server as {got!r} — it must be "
          f"'off_exchange_share:min:0.4', or the filter is accepted, chipped "
          f"and never applied")

    # The wire format the endpoint parses: key:op:value, comma separated.
    many = param(out["many"], "filters") or ""
    check(many.split(",") == ["off_exchange_share:min:0.4",
                              "spread_cents_tw:max:12",
                              "dollar_vol_per_min:min:1000000"],
          f"several constraints went as {many!r}; the endpoint splits on "
          f"commas and refuses a clause that is not key:op:value")
    check("min" in many and "max" in many,
          f"the direction is lost: {many!r} — a floor sent as a ceiling is "
          f"worse than one not sent at all")

    # An empty pane sends nothing, rather than an empty parameter the
    # endpoint would have to treat as a clause.
    check(param(out["none"], "filters") is None,
          f"an empty pane sent filters={param(out['none'], 'filters')!r}")
    check(param(out["cleared"], "filters") is None,
          "removing the last constraint left it in the request, so clearing "
          "a filter would not bring the rows back")

    # A number not yet typed is not a filter. The endpoint answers a
    # non-numeric threshold with a 400, which on this page reads as the
    # table breaking while someone is typing in it.
    check(param(out["empty"], "filters") is None,
          f"a constraint with no value yet was sent as "
          f"{param(out['empty'], 'filters')!r}; the endpoint 400s on a "
          f"non-numeric threshold")

    # And the request still carries what it carried before.
    check(param(out["one"], "date") == "2026-09-22",
          "the date fell out of the request")

    print(f"scalp filters: pane constraints reach the request, "
          f"failures: {len(FAILS)}")
    for m in FAILS:
        print(f"  FAIL {m}")
    return 1 if FAILS else 0


sys.exit(main())
