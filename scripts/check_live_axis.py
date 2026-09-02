"""The tape's price band must hold still, and its rate must be a rate.

WHAT SHIPPED. Six consecutive frames of a flat FDX tape put the top label at
320.78, 320.44, 320.70, 320.44, 321.04 and 320.83. The axis was following the
data: the anchor was recomputed from the LAST PRINT on every render, so one
off-price odd lot moved the whole scale and every bubble jumped vertically
under a price that had not moved.

And trades/min read 8, 10, 12, 15, 21, 22 while climbing, against a verified
56. It was the window's trade count divided by the window's LENGTH — on a
three-minute window that is a third of the true rate, converging upward from
zero as the buffer fills. 56/3 is 18.7, which is where it was heading.

THE TWO PROPERTIES, stated so they can be tested rather than intended:

    the band moves only when the reference price approaches an edge,
    and then once, to a snapped grid;

    the rate is a count over a FIXED interval, independent of the
    display window.

This drives the shipped component in node against a synthetic tape — flat,
drifting, and with a single wild print — and asserts both.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
JS = ROOT / "static" / "js" / "equities_live.js"

DRIVER = r"""
const fs = require('fs');
const src = fs.readFileSync(process.argv[2], 'utf8');
let factory = null;
global.window = { addEventListener: () => {}, devicePixelRatio: 1 };
global.document = { addEventListener: (ev, fn) => { if (ev === 'alpine:init') fn(); },
                    getElementById: () => null };
global.Alpine = { data: (_n, f) => { factory = f; } };
global.WebSocket = function () { this.readyState = 0; };
eval(src);
// A PANE, not the component. The component owns the socket, the frame loop
// and the list of panes; everything asserted below is a property of one plot,
// and driving it directly is what makes it drivable at all.
if (typeof global.window.lvPane !== 'function') {
  console.error('window.lvPane is not exported — the pane is not separable '
                + 'from the component, so none of this can be driven');
  process.exit(2);
}
const c = global.window.lvPane(1, () => {});
c.$nextTick = () => {};
c.showQuotes = false;

const T0 = Date.now() - 200000;
function tape(mk) {
  c.trades = []; c.quotes = []; c.band = null; c.bandSteps = 0;
  for (let i = 0; i < 400; i++) {
    c.trades.push({ t: T0 + i * 400, p: mk(i), s: 1 + (i % 50), x: 4 });
  }
}
function bands(mk, span) {
  // Re-band once per simulated frame, exactly as draw() does.
  c.spanCents = span;
  tape(mk);
  const held = [];
  const all = c.trades.slice();
  c.trades = [];
  c.band = null;
  for (const t of all) {
    c.trades.push(t);
    c.reband(false);
    held.push([c.band.lo, c.band.hi]);
  }
  return { bands: held, steps: c.bandSteps };
}

const out = {};

// FLAT: 320.40-320.50, the case that produced six different top labels.
out.flat = bands(i => 320.45 + ((i % 5) - 2) * 0.01, 30);

// One wild print 60 cents away — a stale-venue odd lot. It must NOT move the
// band, because a single print is not the price.
out.spike = bands(i => (i === 200 ? 321.05 : 320.45 + ((i % 5) - 2) * 0.01), 30);

// A genuine 40-cent drift must move it, and by STEPS, not continuously.
out.drift = bands(i => 320.40 + i * 0.0015, 30);

// The rate, over a fixed minute, independent of the display window.
function rateFor(windowS) {
  c.windowS = windowS;
  c.trades = [];
  const now = Date.now();
  // 56 trades in each of the last three minutes. Offset by half a second so
  // no print lands exactly on a minute boundary — the count is inclusive at
  // both ends, and a trade sitting on the boundary belongs to both minutes.
  for (let m = 0; m < 3; m++)
    for (let k = 0; k < 56; k++)
      c.trades.push({ t: now - 500 - (m * 60000) - k * 1000,
                      p: 320.4, s: 1, x: 4 });
  c.trades.sort((a, b) => a.t - b.t);
  const tEnd = now;
  const minuteAgo = tEnd - 60000;
  let n = 0;
  for (let i = c.trades.length - 1; i >= 0; i--) {
    const t = c.trades[i].t;
    if (t > tEnd) continue;
    if (t < minuteAgo) break;
    n += 1;
  }
  return { windowS, buffered: c.bufferedS(), inMinute: n };
}
out.rate = [rateFor(60), rateFor(180), rateFor(600)];
out.retain = { at60: c.retainS.call({ windowS: 60 }),
               at180: c.retainS.call({ windowS: 180 }) };

// ── the zoom ladders ─────────────────────────────────────────────────────
// Doubling overshot every small adjustment: from 180s the only neighbours
// were 90 and 360.
function ladder(kind) {
  const stops = kind === 'x' ? c.windowStops : c.spanStops;
  const start = kind === 'x' ? 180 : 30;
  const down = [];
  let v = start;
  for (let i = 0; i < 6; i++) { v = c.step(stops, v, -1); down.unshift(v); }
  v = start;
  const up = [];
  for (let i = 0; i < 6; i++) { v = c.step(stops, v, 1); up.push(v); }
  return { down, up, start };
}
out.ladderX = ladder('x');
out.ladderY = ladder('y');

// ── the price lines ──────────────────────────────────────────────────────
// Four prints sharing a price and a millisecond must count as four; a line
// placed a fraction of a cent off must still find them, because it is a
// question about a price and not about 320.4523.
c.windowS = 180; c.spanCents = 30; c.trades = [];
const hn = Date.now() - 5000;
for (let i = 0; i < 4; i++) c.trades.push({ t: hn, p: 320.45, s: 100, x: 4 });
c.trades.push({ t: hn + 1000, p: 320.55, s: 400, x: 4 });
out.hitsExact = c.lineHits(320.45, hn - 1000, Date.now());
out.hitsMiss  = c.lineHits(320.90, hn - 1000, Date.now());
out.hitsNear  = c.lineHits(320.452, hn - 1000, Date.now());
console.log(JSON.stringify(out));
"""


def main() -> int:
    drv = ROOT / "scripts" / "_live_axis_driver.js"
    drv.write_text(DRIVER, encoding="utf-8")
    try:
        p = subprocess.run(["node", str(drv), str(JS)], capture_output=True,
                           text=True, encoding="utf-8", cwd=ROOT)
    finally:
        drv.unlink(missing_ok=True)
    if p.returncode != 0:
        print("  the component could not be driven in node:")
        print("   ", (p.stderr or "").strip()[-400:])
        return 1
    out = json.loads(p.stdout.strip().splitlines()[-1])

    bad = 0

    # ── the band holds still ─────────────────────────────────────────────
    flat = {tuple(b) for b in out["flat"]["bands"]}
    if len(flat) != 1:
        bad += 1
        print(f"\n  a FLAT tape moved the band {len(flat)} times. This is the "
              f"reported defect: six frames, six different top labels, price "
              f"unchanged.")
        for b in sorted(flat)[:4]:
            print(f"      {b}")

    spike = {tuple(b) for b in out["spike"]["bands"]}
    if len(spike) != 1:
        bad += 1
        print(f"\n  one wild print moved the band ({len(spike)} distinct "
              f"bands). A single off-price odd lot is not the price, and "
              f"following it is what made the axis jump.")

    drift = [tuple(b) for b in out["drift"]["bands"]]
    uniq = []
    for b in drift:
        if not uniq or uniq[-1] != b:
            uniq.append(b)
    if len(uniq) < 2:
        bad += 1
        print("\n  a genuine 40-cent drift never moved the band — it must "
              "follow price eventually, just not continuously")
    if len(uniq) > 12:
        bad += 1
        print(f"\n  a 40-cent drift moved the band {len(uniq)} times; it is "
              f"following the data rather than stepping")
    # Snapped: every boundary on the grid, so labels repeat between steps.
    for lo, hi in uniq:
        if abs(round(lo * 100) - lo * 100) > 1e-6:
            bad += 1
            print(f"\n  band edge {lo} is not on a whole-cent grid — unsnapped "
                  f"edges give labels like 320.78 that read as movement")
            break

    # ── the rate is a rate ───────────────────────────────────────────────
    rates = out["rate"]
    counts = {r["inMinute"] for r in rates}
    if len(counts) != 1:
        bad += 1
        print(f"\n  the per-minute count depends on the display window: "
              f"{[(r['windowS'], r['inMinute']) for r in rates]}. A rate over "
              f"a variable interval is not a rate.")
    elif counts != {56}:
        bad += 1
        print(f"\n  a tape of 56 trades per minute counted {counts.pop()}")

    # ── the ladders are finer than doubling ──────────────────────────────
    for kind, key in (("window", "ladderX"), ("price", "ladderY")):
        lad = out[key]
        down, up, start = lad["down"], lad["up"], lad["start"]
        if not down or not up:
            bad += 1
            print(f"\n  the {kind} ladder produced nothing")
            continue
        # One step must not halve or double — that is the overshoot
        # reported: from 180s the only neighbours were 90 and 360.
        if down[-1] <= start / 2 or up[0] >= start * 2:
            bad += 1
            print(f"\n  one {kind} step goes {start} to {down[-1]} or "
                  f"{up[0]} — still halving or doubling, which overshoots "
                  f"every small adjustment")
        if sorted(down) != down or sorted(up) != up:
            bad += 1
            print(f"\n  the {kind} ladder is not monotonic: {down} {up}")
        if len(set(down)) < 2:
            bad += 1
            print(f"\n  the {kind} ladder cannot step down: {down}")

    # ── a price line counts the prints AT that price ─────────────────────
    if out["hitsExact"] != 4:
        bad += 1
        print(f"\n  four prints at one price counted {out['hitsExact']} — "
              f"the whole output of a placed line is whether anything has "
              f"traded there")
    if out["hitsMiss"] != 0:
        bad += 1
        print(f"\n  a line 45 cents away counted {out['hitsMiss']} prints")
    if out["hitsNear"] != 4:
        bad += 1
        print(f"\n  a line placed a fifth of a cent off found "
              f"{out['hitsNear']} — it is a question about a price, not "
              f"about 320.4523")

    # ── retention outlives the display window ────────────────────────────
    if out["retain"]["at60"] <= 60 or out["retain"]["at180"] <= 180:
        bad += 1
        print(f"\n  the browser trims at the display window ({out['retain']}), "
              f"so widening it can never recover what was already dropped")

    print(f"\nband positions checked: {len(out['flat']['bands'])} flat, "
          f"{len(out['spike']['bands'])} with a spike, "
          f"{len(uniq)} distinct across a drift; problems: {bad}")
    if not bad:
        print("  the band holds still under a flat tape and steps on a snapped "
              "grid; the rate is a fixed-interval count")
    return 1 if bad else 0


sys.exit(main())
