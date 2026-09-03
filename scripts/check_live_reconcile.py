"""A pane must not act on another application's order, or block on a settled one.

TWO PROPERTIES, both about orders the pane did not place.

THE SOURCE CHECK. `primaryOrder()` prefers ids this pane sent, but `ownIds`
is lost on a reload — so after a refresh the fallback is what runs, and it
used to return "the only working order" whichever application had placed it.
thinkorswim orders appear in the same working list. A nudge would then
reprice an order entered by hand somewhere else.

Schwab stamps `tag` itself and a client cannot set one (a body carrying a tag
is rejected outright: 400 tagged, 201 with the identical body untagged,
tested 2026-09-03 by scripts/probe_schwab_tag.py). API orders come back
`TA_<account-derived>`, thinkorswim's come back `API_TOS:AT_LADDER_AS`, and
`broker._norm_order` turns that prefix into `from_api`. The stamp is
per-account, so it cannot say WHICH of ours an order is — but the fallback
only runs with a single candidate, and "is this ours at all" is the question
it was missing.

THE WINDOW. An unresolved placement blocks the pane completely: nothing may
be sent while one is outstanding. The probe order was listed by Schwab ONE
SECOND after the 201, so a 15-second window held the pane shut for fourteen
seconds after the answer had arrived. This asserts the window is a small
multiple of the measured latency rather than an order of magnitude above it.

Drives the shipped pane in node, because both properties are decisions made
in the browser and reading the source cannot show what the function returns.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
JS = ROOT / "static" / "js" / "equities_live.js"

# The measured floor: how long Schwab took to list a real order after
# accepting it. The window has to be comfortably above this.
APPEARED_AFTER_S = 1.0
# And below this, or it is not a window, it is a lockout.
WINDOW_CEILING_S = 8.0

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

if (typeof global.window.lvPane !== 'function') {
  console.error('window.lvPane is not exported — the pane cannot be driven');
  process.exit(2);
}

function pane() {
  const c = global.window.lvPane(1, () => {});
  c.$nextTick = () => {};
  c.symbol = 'FDX';
  c.refreshBroker = async () => {};
  return c;
}

function ord(id, from_api, extra) {
  return Object.assign({ order_id: id, symbol: 'FDX', side: 'BUY', qty: 100,
                         price: 318.50, filled: 0, status: 'WORKING',
                         working: true, from_api: from_api }, extra || {});
}

const out = {};

// ── primaryOrder(): what the single-order controls act on ────────────────
// An id THIS PANE SENT wins outright. It is ours by construction — we have
// the id because we got it back from our own placement — so the stamp is
// not consulted and must not be able to veto it.
{
  const c = pane();
  c.working = [ord('1', false)];
  c.ownIds = ['1'];
  const p = c.primaryOrder();
  out.ownIdWins = p ? p.order_id : null;
}
// The fallback, which is what runs after a reload drops ownIds.
{
  const c = pane();
  c.working = [ord('2', true)];
  c.ownIds = [];
  const p = c.primaryOrder();
  out.fallbackOurs = p ? p.order_id : null;
}
{
  const c = pane();
  c.working = [ord('3', false)];      // placed in thinkorswim
  c.ownIds = [];
  out.fallbackForeign = c.primaryOrder();
  out.foreignWhy = c.noPrimaryWhy('move');
}
{
  const c = pane();
  c.working = [];
  c.ownIds = [];
  out.emptyWhy = c.noPrimaryWhy('move');
}
{
  const c = pane();
  c.working = [ord('4', true), ord('5', true)];
  c.ownIds = [];
  out.fallbackTwo = c.primaryOrder();
  out.twoWhy = c.noPrimaryWhy('move');
}

// ── the give-up window ───────────────────────────────────────────────────
// Schwab keeps answering "absent". Count the looks until the pane stops.
async function giveUp() {
  const c = pane();
  c.brokerCall = async () => ({ ok: true, state: 'absent' });
  c.unresolved = { side: 'BUY', qty: 100, price: 318.50,
                   sentAt: Date.now() / 1000, tries: 0, state: 'looking' };
  for (let i = 0; i < 60; i++) {
    await c.tryResolve();
    if (c.unresolved.state === 'gave-up') return c.unresolved.tries;
  }
  return null;
}

// And it must still resolve the instant the order IS there — a tighter
// window must not have turned into "gives up before Schwab answers".
async function found() {
  const c = pane();
  c.brokerCall = async () => ({ ok: true, state: 'found',
                                order: { order_id: '9', status: 'WORKING' } });
  c.unresolved = { side: 'BUY', qty: 100, price: 318.50,
                   sentAt: Date.now() / 1000, tries: 0, state: 'looking' };
  await c.tryResolve();
  return { cleared: c.unresolved === null, owned: c.ownIds.slice() };
}

(async () => {
  out.giveUpTries = await giveUp();
  out.found = await found();
  console.log(JSON.stringify(out));
})();
"""


def main() -> int:
    drv = ROOT / "scripts" / "_live_reconcile_driver.js"
    drv.write_text(DRIVER, encoding="utf-8")
    try:
        p = subprocess.run(["node", str(drv), str(JS)], capture_output=True,
                           text=True, encoding="utf-8", cwd=ROOT)
    finally:
        drv.unlink(missing_ok=True)
    if p.returncode != 0:
        print("  the pane could not be driven in node:")
        print("   ", (p.stderr or "").strip()[-500:])
        return 1
    out = json.loads(p.stdout.strip().splitlines()[-1])

    bad = 0

    def fail(msg: str) -> None:
        nonlocal bad
        bad += 1
        print(f"\n  {msg}")

    # ── the source check ─────────────────────────────────────────────────
    if out["ownIdWins"] != "1":
        fail("an order this pane SENT was not returned as its primary. The "
             "id came back from our own placement, so it is ours whatever "
             "the stamp says — the source test must not veto it.")

    if out["fallbackOurs"] != "2":
        fail("after a reload the pane no longer recognises its own order: a "
             "single working order stamped TA_ was not returned, so the "
             "controls are dead on an order that IS ours.")

    if out["fallbackForeign"] is not None:
        fail("THE ONE THAT MOVES SOMEONE ELSE'S MONEY. A single working "
             "order NOT stamped as this app's was returned as the primary, "
             "so a nudge would reprice an order placed in thinkorswim.")

    why = (out["foreignWhy"] or "").lower()
    if "not placed from here" not in why:
        fail(f"the refusal does not say WHY: {out['foreignWhy']!r}. Telling "
             f"someone there is no working order, while one sits visibly in "
             f"the list, reads as a broken pane rather than a refusal.")
    if "no working order" in why:
        fail(f"a foreign order was described as no order at all: "
             f"{out['foreignWhy']!r}")

    if "no working order" not in (out["emptyWhy"] or "").lower():
        fail(f"an empty list produced the wrong explanation: "
             f"{out['emptyWhy']!r}")

    if out["fallbackTwo"] is not None:
        fail("two working orders produced a primary. With more than one the "
             "controls must name which rather than picking.")
    if "more than one" not in (out["twoWhy"] or "").lower():
        fail(f"two working orders produced the wrong explanation: "
             f"{out['twoWhy']!r}")

    # ── the window ───────────────────────────────────────────────────────
    tries = out["giveUpTries"]
    if tries is None:
        fail("the pane never gave up. An unresolved placement blocks every "
             "send in the pane, so a window that does not end is a pane that "
             "cannot trade until someone clears it by hand.")
    else:
        # The loop runs on the order poll; the pane's own arithmetic turns
        # looks into seconds, and this reproduces it from the outside.
        secs = tries * 2
        if secs <= APPEARED_AFTER_S:
            fail(f"the window is {secs}s, at or under the {APPEARED_AFTER_S}s "
                 f"a real order took to appear. It would give up before "
                 f"Schwab had answered and report a landed order as "
                 f"unconfirmed.")
        if secs > WINDOW_CEILING_S:
            fail(f"the window is {secs}s. A probe order was listed after "
                 f"{APPEARED_AFTER_S}s, and the pane is blocked from trading "
                 f"for every second of it — this is a lockout, not a margin.")

    if not out["found"]["cleared"]:
        fail("an order that WAS found did not clear the unresolved state, so "
             "the pane stays blocked after the question was answered.")
    if "9" not in out["found"]["owned"]:
        fail("a resolved order was not adopted into ownIds, so the controls "
             "would not act on the order the pane just confirmed is its own.")

    if bad:
        print(f"\nreconcile cases FAILED: {bad}")
        return 1
    print(f"live reconcile: primaryOrder refuses foreign orders; the "
          f"unresolved window is {out['giveUpTries'] * 2}s against a "
          f"{APPEARED_AFTER_S:.0f}s measured appearance")
    return 0


if __name__ == "__main__":
    sys.exit(main())
