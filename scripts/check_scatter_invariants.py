"""The scatter's colour thresholds must not change what is drawn.

WHAT WAS REPORTED. With the upper band at 50 trips, 24 points drew and the x
axis ended at 0.070. With it at 28, the header still read "24 points" but four
more names appeared and the axis reached 0.095. Those four had the highest
values on that axis — they were being clipped, not excluded.

Two defects behind it, and neither was visible from the count:

  * the axis domain came from whatever Chart.js could see in the VISIBLE
    datasets, so a boundary that moved a name between bands could move the
    extent with it;
  * geomPoints() was evaluated separately by the header, the x-show, the note
    and the renderer — four passes over the same rows, with no guarantee the
    number shown was counting the array drawn.

THE INVARIANT, stated once so it can be tested rather than intended:

    membership and extent are functions of the DATA and the FILTERS.
    Colour is a function of membership. Never the reverse.

This runs the shipped component's own logic in node against a fabricated
payload, sweeping the thresholds across their whole range, and asserts the
point set and the domain are byte-identical throughout. It also asserts every
traded point lands in exactly one band, since a point matching none used to
vanish without changing any count.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
JS = ROOT / "static" / "js" / "equities_scalp.js"

DRIVER = r"""
const fs = require('fs');
const src = fs.readFileSync(process.argv[2], 'utf8');

// The component is registered on an alpine:init listener. Stub just enough of
// the browser for the module to load and hand back the factory.
let factory = null;
global.window = { addEventListener: () => {} };
global.document = { addEventListener: (ev, fn) => { if (ev === 'alpine:init') fn(); } };
global.Alpine = { data: (_n, f) => { factory = f; } };
global.Chart = undefined;
global.fetch = () => Promise.reject(new Error('no network in this check'));
eval(src);
if (!factory) { console.log(JSON.stringify({error: 'no component'})); process.exit(0); }

const c = factory();
c.$nextTick = () => {};

// A payload shaped like a real one: traded names spread across trip counts,
// including the boundary values, plus untraded rows and a null.
const rows = [];
const traded = [[226, 0.070], [128, 0.062], [61, 0.058], [49, 0.055],
                [30, 0.095], [28, 0.091], [12, 0.088], [10, 0.084],
                [9, 0.081], [6, 0.078], [2, 0.075], [1, 0.072]];
traded.forEach(([trips, ask], i) => {
  rows.push({ symbol: 'T' + i, passes: true,
              values: { spread_bps: 5 + i, at_ask_share: ask },
              traded: { days: 2, trips, net_pnl: 10 - i, shares: 100 * (i + 1),
                        pnl_per_min: 1 - i * 0.1, pnl_per_trip: 0.5,
                        pnl_per_share: 0.01, win_rate: 0.5 } });
});
for (let i = 0; i < 8; i++) {
  rows.push({ symbol: 'U' + i, passes: true,
              values: { spread_bps: 4 + i, at_ask_share: 0.04 + i * 0.001 },
              traded: null });
}
// One traded name missing the y value entirely — it must be excluded, and the
// exclusion must be REPORTED with a reason.
rows.push({ symbol: 'GAP', passes: true,
            values: { spread_bps: 9, at_ask_share: null },
            traded: { days: 1, trips: 40, net_pnl: 1, shares: 10,
                      pnl_per_min: 0.1, pnl_per_trip: 0.1,
                      pnl_per_share: 0.01, win_rate: 0.5 } });

c.cand = {
  date: '2026-09-02', rows,
  columns: [{ key: 'spread_bps', metric: 'spread_bps_tw', role: true },
            { key: 'at_ask_share', metric: 'at_ask_share', role: false }],
  roles: [{ key: 'spread_bps', label: 'spread bps', units: 'bps' }],
  derived: [],
};
c.meta = { metrics: [], filters: {} };
c.geomX = 'at_ask_share';
c.geomY = 'spread_bps';
c.geomXLog = false;
c.geomYLog = false;
c.geomTradedOnly = true;
c.geomOnly = true;
c.geomMinTrips = 1;

const out = { sweeps: [], gap: null, unassignedSeen: 0 };
for (const colorBy of ['trips', 'pnl_per_share', 'win_rate']) {
  c.geomColorBy = colorBy;
  c.geomHi = null; c.geomLo = null;
  const range = c.geomColorRange();
  if (!range) continue;
  const span = range.max - range.min;
  for (let step = 0; step <= 20; step++) {
    const hi = range.min + (span * step) / 20;
    const lo = range.min + (span * Math.max(0, step - 4)) / 20;
    c.geomHi = hi; c.geomLo = lo;
    c.geomInvalidate();
    const g = c.geomData();
    // Every traded point must land in exactly one band.
    const keys = {};
    let unassigned = 0;
    for (const p of g.pts) {
      if (!p.traded) continue;
      const b = c.geomBandOf(c.geomColorValue(p.traded));
      if (!b || !b.key) unassigned += 1;
      keys[p.symbol] = b ? b.key : null;
    }
    out.unassignedSeen += unassigned;
    out.sweeps.push({
      colorBy, hi, lo,
      symbols: g.pts.map(p => p.symbol).sort().join(','),
      domain: g.domain ? [g.domain.x.min, g.domain.x.max,
                          g.domain.y.min, g.domain.y.max] : null,
      bands: Object.keys(keys).length,
    });
  }
}
c.geomHi = null; c.geomLo = null; c.geomColorBy = 'trips';
c.geomInvalidate();
out.gap = { note: c.geomGapNote(), shown: c.geomData().tradedShown,
            total: c.geomData().tradedTotal,
            reasons: c.geomData().reasons };
console.log(JSON.stringify(out));
"""


def main() -> int:
    driver = ROOT / "scripts" / "_scatter_driver.js"
    driver.write_text(DRIVER, encoding="utf-8")
    try:
        p = subprocess.run(["node", str(driver), str(JS)],
                           capture_output=True, text=True, encoding="utf-8",
                           cwd=ROOT)
    finally:
        driver.unlink(missing_ok=True)

    if p.returncode != 0:
        print("  the component could not be loaded in node:")
        print("   ", (p.stderr or "").strip().splitlines()[-1:] or p.stderr)
        return 1
    try:
        out = json.loads(p.stdout.strip().splitlines()[-1])
    except Exception:
        print("  driver produced no JSON:")
        print("   ", (p.stdout or "")[:400])
        return 1
    if out.get("error"):
        print(f"  {out['error']}")
        return 1

    bad = 0
    sweeps = out["sweeps"]
    if len(sweeps) < 30:
        print(f"  only {len(sweeps)} threshold positions swept — the sweep is "
              f"not exercising the range")
        bad += 1

    # THE INVARIANT. Membership and extent must be identical across every
    # threshold position, for every colour metric.
    by_metric = {}
    for sw in sweeps:
        by_metric.setdefault(sw["colorBy"], []).append(sw)
    for metric, group in by_metric.items():
        syms = {g["symbols"] for g in group}
        if len(syms) != 1:
            bad += 1
            print(f"\n  colouring by {metric}: the POINT SET changes with the "
                  f"threshold — {len(syms)} distinct sets across the sweep.")
            for v in sorted(syms)[:3]:
                print(f"      {v[:110]}")
        doms = {json.dumps(g["domain"]) for g in group}
        if len(doms) != 1:
            bad += 1
            print(f"\n  colouring by {metric}: the AXIS DOMAIN changes with "
                  f"the threshold — {len(doms)} distinct domains.")
            for v in sorted(doms)[:3]:
                print(f"      {v}")

    if out["unassignedSeen"]:
        bad += 1
        print(f"\n  {out['unassignedSeen']} traded point(s) matched no band "
              f"across the sweep — they would vanish without changing any "
              f"count")

    # The gap must be reported WITH ITS REASON.
    gap = out["gap"]
    if gap["total"] - gap["shown"] <= 0:
        bad += 1
        print("\n  the fixture's excluded traded name was not excluded, so "
              "the gap reporting is untested")
    elif not gap["note"]:
        bad += 1
        print("\n  traded names are missing and nothing says why. Reporting "
              "the gap without the reason is the defect being fixed.")
    elif not gap["reasons"]:
        bad += 1
        print("\n  the gap is described but carries no per-reason counts")

    print(f"\nthreshold positions swept: {len(sweeps)} across "
          f"{len(by_metric)} colour metrics, problems: {bad}")
    if not bad:
        print("  membership and extent are invariant under the colour "
              "thresholds; the gap names its reasons")
    return 1 if bad else 0


sys.exit(main())
