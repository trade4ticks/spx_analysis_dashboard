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
  // refreshBroker is DELIBERATELY NOT STUBBED. Which endpoint an action
  // reads back from is one of the things under test here, and a stub would
  // have hidden it - it did, until this comment replaced the stub.
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

// ── a marketable click asks first ────────────────────────────────────────
// bid 318.49 / ask 318.51. A BUY at or above 318.51 lifts the offer; a SELL
// at or below 318.49 hits the bid. Both fill instantly.
function quoted(c) {
  c.quotes = [{ t: Date.now(), bp: 318.49, ap: 318.51 }];
  return c;
}
{
  const c = quoted(pane());
  out.mkt = {
    buyThrough:  c.isMarketable('BUY', 318.52),
    buyAtAsk:    c.isMarketable('BUY', 318.51),
    buyPassive:  c.isMarketable('BUY', 318.45),
    sellThrough: c.isMarketable('SELL', 318.45),
    sellAtBid:   c.isMarketable('SELL', 318.49),
    sellPassive: c.isMarketable('SELL', 318.55),
  };
  c.quotes = [];
  out.mktNoQuote = c.isMarketable('BUY', 318.52);
}
// A passive click goes straight through; a marketable one is held.
{
  const c = quoted(pane());
  c.armed = true;
  let sent = null;
  c.place = async (side, price) => { sent = { side, price }; return { ok: true }; };
  c.rowTooFine = () => false;

  c.clickLadder('buy', 318.45);
  out.passiveSent = sent;
  out.passiveHeld = c.confirm;

  sent = null;
  c.clickLadder('sell', 318.45);          // through the bid
  out.mktSent = sent;
  out.mktHeld = c.confirm ? { side: c.confirm.side, price: c.confirm.price,
                              kind: c.confirm.kind } : null;
  out.mktText = c.confirmText();

  // Declining sends nothing.
  c.confirmCancel();
  out.afterCancel = { sent, confirm: c.confirm };

  // Confirming sends exactly what was held.
  c.clickLadder('sell', 318.45);
  c.confirmSend();
  out.afterConfirm = sent;
}
// A NUDGE that crosses the touch asks too.
{
  const c = quoted(pane());
  c.armed = true;
  c.ladderCents = 1;
  let moved = null;
  c.sendMove = async (o, price) => { moved = { id: o.order_id, price }; };
  c.working = [ord('7', true, { side: 'SELL', price: 318.50 })];
  c.ownIds = ['7'];

  c.nudge(+1);                            // to 318.51, still above the bid
  out.nudgePassive = moved;
  moved = null;
  c.working = [ord('7', true, { side: 'SELL', price: 318.50 })];
  c.nudge(-1);                            // to 318.49 — AT the bid
  out.nudgeMktSent = moved;
  out.nudgeMktHeld = c.confirm ? { kind: c.confirm.kind,
                                   price: c.confirm.price } : null;
  c.confirmSend();
  out.nudgeAfterConfirm = moved;
}

// ── the call budget ──────────────────────────────────────────────────────
// Which endpoint each action reads back from. /broker/state costs TWO Schwab
// calls (orders + positions); /broker/orders costs one.
async function callBudget() {
  const c = pane();
  c.armed = true;
  c.symbol = 'FDX';
  const paths = [];
  c.brokerCall = async (path) => {
    paths.push(String(path).split('?')[0]);
    return { ok: true, working: [], recent: [], order_id: '1' };
  };
  const grab = async (fn) => { paths.length = 0; await fn(); return paths.slice(); };

  const calls = {};
  calls.move = await grab(() =>
    c.sendMove({ order_id: '1', side: 'SELL', qty: 100, filled: 0 }, 318.60));
  c.working = [ord('1', true)];
  c.ownIds = ['1'];
  calls.cancel = await grab(() => c.cancelOrder('1'));
  calls.place = await grab(() => c.place('SELL', 318.60));
  return calls;
}

// ── the poll cadence ─────────────────────────────────────────────────────
// Not a rate this reads off a constant: it drives the real tick and counts
// how many reads actually leave.
async function polls(active) {
  const app = factory();
  app.panes = [pane()];
  app.panes[0].symbol = 'FDX';
  app.panes[0].ladder = active;
  app.lastOrderPollAt = 0;
  app.livePanes = () => app.panes;
  let n = 0;
  global.fetch = async () => ({ json: async () => { n += 1;
    return { ok: true, working: [], recent: [] }; } });
  // Sixty seconds of 2s ticks, with time advanced by hand.
  let t = 1000000;
  const realNow = Date.now;
  Date.now = () => t;
  try {
    for (let i = 0; i < 30; i++) { await app.pollOrders(); t += 2000; }
  } finally { Date.now = realNow; }
  return n;
}

// ── the gutter draws one price scale, and marks the hazard ───────────────
// A recording 2d context. drawLadder and drawGrid are the code most changed
// and the least visible from a test: an exception here blanks the pane, and
// the only symptom is a plot that stops updating.
function recorder() {
  const rec = { texts: [], strokes: [], fills: [], clips: 0 };
  // A PLAIN BACKING OBJECT. Writing state back through the proxy re-enters
  // the set trap and recurses until the stack goes.
  const st = {};
  const ctx = new Proxy({}, {
    get(_t, k) {
      if (k in st) return st[k];
      if (k === 'fillText') {
        return (txt, x, y) => rec.texts.push({ txt: String(txt), x, y });
      }
      if (k === 'clip') return () => { rec.clips += 1; };
      if (k === 'stroke') return () => rec.strokes.push(st.strokeStyle);
      if (k === 'fillRect') {
        return (x, y, w, h) => rec.fills.push({ style: st.fillStyle, x, w });
      }
      if (k === 'measureText') return () => ({ width: 20 });
      return () => {};
    },
    set(_t, k, v) { st[k] = v; return true; },
  });
  return { ctx, rec };
}

function drawn(ladderOn) {
  const c = quoted(pane());
  c.ladder = ladderOn;
  c.armed = true;
  c.ladderCents = 1;
  const T0 = Date.now() - 10000;
  for (let i = 0; i < 40; i++) {
    c.trades.push({ t: T0 + i * 200, p: 318.50, s: 10, x: 4 });
  }
  const padL = 8, padT = 8, plotW = 400, plotH = 300;
  const lo = 318.40, hi = 318.60;
  const Y = pp => padT + (1 - (pp - lo) / (hi - lo)) * plotH;
  const { ctx, rec } = recorder();
  // Both halves of the gutter question: the plot's own axis labels, and the
  // ladder's price column.
  c.drawGrid(ctx, padL, padT, plotW, plotH, lo, hi,
             Date.now() - 60000, Date.now(),
             t => padL + plotW * 0.5, Y, 520);
  if (ladderOn) {
    c.drawLadder(ctx, padL, padT, plotW, plotH, lo, hi, Y);
  }
  const gutterX = padL + plotW;
  return {
    // Price-shaped text drawn INSIDE the gutter, x by x.
    gutterPrices: rec.texts
      .filter(t => /^318\.\d\d$/.test(t.txt) && t.x >= gutterX)
      .map(t => Math.round(t.x - gutterX)),
    clips: rec.clips,
    hazardFills: rec.fills.filter(f => String(f.style).includes('255,140,0')).length,
  };
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
  out.drawLadderOn = drawn(true);
  out.drawLadderOff = drawn(false);
  out.calls = await callBudget();
  out.pollsTrading = await polls(true);
  out.pollsWatching = await polls(false);
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

    # -- the marketable guard --------------------------------------------
    m = out["mkt"]
    for label, got, want in (("a buy through the ask", m["buyThrough"], True),
                             ("a buy AT the ask", m["buyAtAsk"], True),
                             ("a passive buy", m["buyPassive"], False),
                             ("a sell through the bid", m["sellThrough"], True),
                             ("a sell AT the bid", m["sellAtBid"], True),
                             ("a passive sell", m["sellPassive"], False)):
        if got is not want:
            fail(f"{label} was classified {got!r}, expected {want!r}. At the "
                 f"touch counts as marketable: an order AT the offer lifts "
                 f"it.")

    # UNKNOWN IS NOT SAFE. With no quote the answer must be null, so the
    # caller asks, rather than false, which would send silently.
    if out["mktNoQuote"] is not None:
        fail(f"with no NBBO the marketable test returned "
             f"{out['mktNoQuote']!r} instead of null. False here means a "
             f"marketable order goes out unquestioned exactly when the quote "
             f"feed is worst.")

    if out["passiveSent"] is None:
        fail("a PASSIVE click did not send. The guard must not stand between "
             "the click and the order the strategy actually wants.")
    if out["passiveHeld"] is not None:
        fail("a passive click raised the confirmation banner. Asking about "
             "orders that are fine is how a confirmation becomes something "
             "to click through without reading.")

    if out["mktSent"] is not None:
        fail(f"THE ONE THAT COST MONEY. A marketable click sent immediately: "
             f"{out['mktSent']}. It must be held until confirmed.")
    held = out["mktHeld"]
    if not held or held["side"] != "SELL" or abs(held["price"] - 318.45) > 1e-9:
        fail(f"the held order is not the one clicked: {held}")
    txt = (out["mktText"] or "")
    if "318.49" not in txt or "FILL IMMEDIATELY" not in txt:
        fail(f"the banner does not name the edge and the consequence: "
             f"{txt!r}. 'Are you sure' is not a warning.")

    if out["afterCancel"]["sent"] is not None:
        fail("declining the confirmation still sent the order")
    if out["afterCancel"]["confirm"] is not None:
        fail("declining left the banner up")
    if out["afterConfirm"] is None:
        fail("confirming did NOT send. The deliberate override has to work, "
             "or the guard becomes a wall and the next fix will be to "
             "remove it.")

    if out["nudgePassive"] is None:
        fail("a nudge that stays passive was blocked")
    if out["nudgeMktSent"] is not None:
        fail(f"a nudge THROUGH the touch went straight out: "
             f"{out['nudgeMktSent']}. Walking an order one row at a time is "
             f"exactly how it crosses.")
    if not out["nudgeMktHeld"] or out["nudgeMktHeld"]["kind"] != "move":
        fail(f"the held nudge is not recorded as a move: "
             f"{out['nudgeMktHeld']}")
    if out["nudgeAfterConfirm"] is None:
        fail("confirming a nudge did not move the order")

    # -- the call budget --------------------------------------------------
    calls = out["calls"]
    if calls["move"] != ["replace", "orders"]:
        fail(f"a row move reads back from {calls['move'][1:]}, not orders "
             f"alone. /broker/state costs TWO Schwab calls and a replace "
             f"cannot move a position - that third call is the one that "
             f"made repricing unusable at 90/120.")
    if calls["cancel"] != ["cancel", "orders"]:
        fail(f"a cancel reads back from {calls['cancel'][1:]}; cancelling "
             f"cannot move a position either.")
    if calls["place"] != ["order", "state"]:
        fail(f"a placement reads back from {calls['place'][1:]}. This one "
             f"SHOULD read both: a limit can fill on arrival, and then the "
             f"position has changed.")

    # -- the poll cadence -------------------------------------------------
    # 30 ticks of 2s = 60 simulated seconds.
    if out["pollsTrading"] != 30:
        fail(f"with a ladder open the order poll ran {out['pollsTrading']} "
             f"times a minute, not 30. Orders in this account live one to "
             f"six seconds; slowing this loses them.")
    if out["pollsWatching"] != 10:
        fail(f"with every ladder shut the order poll ran "
             f"{out['pollsWatching']} times a minute, not 10. A pane "
             f"watching the tape is not watching the order book, and 30 "
             f"calls a minute of a 90 budget is what left no room to "
             f"reprice.")

    # -- one price scale, and a visible hazard ----------------------------
    # The buy column is the first 26px of the gutter. drawGrid used to write
    # its own price labels at +6px, straight over the clickable column: two
    # scales disagreeing, with the one you click underneath.
    on = out["drawLadderOn"]
    off = out["drawLadderOff"]
    BUY_COL_W = 26

    over = [x for x in on["gutterPrices"] if x < BUY_COL_W]
    if over:
        fail(f"with the ladder open, {len(over)} price labels are drawn "
             f"inside the buy column (at +{sorted(set(over))}px of the "
             f"gutter). That is the second scale painted over the thing you "
             f"click.")
    if not on["gutterPrices"]:
        fail("the ladder drew no prices at all in the gutter — suppressing "
             "the axis labels only works because the ladder replaces them.")
    if not any(x >= BUY_COL_W for x in on["gutterPrices"]):
        fail("no price label landed in the ladder's own price column")

    # With the ladder shut the axis labels are the ONLY scale and must stay.
    # MORE THAN ONE: the last-trade marker is also a price label in the
    # gutter, so "at least one" passes even when every grid label is gone.
    # A scale is a series.
    if len(off["gutterPrices"]) < 3:
        fail(f"with the ladder shut the plot drew {len(off['gutterPrices'])} "
             f"price labels. The axis is the only scale then, and a lone "
             f"last-trade marker is not a scale.")

    if on["hazardFills"] < 1:
        fail("no marketable row was filled with the hazard colour, so the "
             "warning the click depends on is not on screen.")
    if on["clips"] < 1:
        fail("the hazard hatching never clipped, so either it did not draw "
             "or it drew outside its row.")
    if off["hazardFills"]:
        fail("hazard shading was drawn with no ladder up")

    if bad:
        print(f"\nreconcile cases FAILED: {bad}")
        return 1
    print(f"live pane: foreign orders refused; unresolved window "
          f"{out['giveUpTries'] * 2}s vs a {APPEARED_AFTER_S:.0f}s "
          f"appearance; marketable clicks and nudges held for confirmation; "
          f"a row move is 2 calls; the order poll is "
          f"{out['pollsTrading']}/min trading and {out['pollsWatching']}/min "
          f"watching")
    return 0


if __name__ == "__main__":
    sys.exit(main())
