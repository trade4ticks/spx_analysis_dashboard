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
/* The optimistic move, watched from the inside.
 *
 * brokerCall is held open on a deferred promise so the in-flight moment can
 * be inspected: that moment is the entire feature, and a test that only
 * looks at the end state cannot tell "moved on the click" from "moved on the
 * reply".
 */
async function optimistic(reply) {
  const c = quoted(pane());
  c.armed = true;
  c.ladderCents = 1;
  c.working = [ord('1', true, { side: 'BUY', price: 318.50 })];
  c.ownIds = ['1'];

  const paths = [];
  let release = null;
  c.brokerCall = async (path) => {
    paths.push(String(path).split('?')[0]);
    await new Promise(r => { release = r; });
    return reply;
  };

  const done = c.sendMove(c.working[0], 318.45);
  // Let sendMove run up to its await, then look before answering it.
  await new Promise(r => setImmediate(r));
  const inFlight = {
    shown: c.shownPrice(c.working[0]),
    recordPrice: c.working[0].price,
    pending: c.pending ? { to: c.pending.to, from: c.pending.from,
                           state: c.pending.state } : null,
  };
  release();
  await done;
  return {
    inFlight,
    paths,
    after: {
      shown: c.working.length ? c.shownPrice(c.working[0]) : null,
      recordPrice: c.working.length ? c.working[0].price : null,
      orderId: c.working.length ? String(c.working[0].order_id) : null,
      pending: c.pending ? { state: c.pending.state } : null,
      revert: c.revert ? { price: c.revert.price } : null,
      ownIds: c.ownIds.slice(),
      unresolved: c.unresolved ? c.unresolved.state : null,
      rt: c.lastRt,
      rtText: c.rtText(),
      action: c.lastAction,
    },
  };
}

// A read-back must not overwrite the readout an action set.
async function rtNotStolen() {
  const c = quoted(pane());
  c.armed = true;
  c.working = [ord('1', true, { side: 'BUY', price: 318.50 })];
  c.ownIds = ['1'];
  let leg = 'replace';
  global.fetch = async () => ({ json: async () => (
    leg === 'replace' ? { ok: true, order_id: '2', rt_ms: 484 }
                      : { ok: true, rt_ms: 855, working: [], recent: [] }) });
  await c.sendMove(c.working[0], 318.45);
  const afterMove = c.rtText();
  leg = 'orders';
  await c.refreshOrders();
  return { afterMove, afterRead: c.rtText() };
}

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
  const rec = { texts: [], strokes: [], fills: [], arcs: [], lines: [],
                clips: 0, _from: null, _to: null };
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
      if (k === 'stroke') {
        return () => {
          rec.strokes.push(st.strokeStyle);
          // A horizontal segment, for "what is drawn at this price".
          if (rec._from && rec._to && Math.abs(rec._from.y - rec._to.y) < 0.5) {
            rec.lines.push({ y: rec._from.y, style: st.strokeStyle });
          }
        };
      }
      if (k === 'moveTo') return (x, y) => { rec._from = { x, y }; };
      if (k === 'lineTo') return (x, y) => { rec._to = { x, y }; };
      // fill() as well as fillRect(): a filled ARC leaves no rect behind,
      // and "is this mark solid" was exactly the question being asked.
      if (k === 'fill') {
        return () => rec.fills.push({ style: st.fillStyle, x: null, w: null });
      }
      if (k === 'fillRect') {
        return (x, y, w, h) => rec.fills.push({ style: st.fillStyle, x, w });
      }
      if (k === 'arc') {
        return (x, y, r) => rec.arcs.push({ x, y, r });
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

// ── the touch after hours: thin, wide, and sometimes not there ───────────
// Regular hours hid this: quotes arrive several times a second, so the
// newest two-sided quote is always current. Outside them they are sparse or
// stop, and a stale NBBO is a DIFFERENT market - narrower than the one being
// traded, so a price that rests against it is marketable against the real
// one.
{
  const NOW = Date.now();
  const q = (agoS, bp, ap) => ({ t: NOW - agoS * 1000, bp, ap });

  const fresh = pane();
  fresh.quotes = [q(1, 318.49, 318.51)];
  const recent = pane();
  recent.quotes = [q(20, 318.49, 318.51)];
  const stale = pane();
  stale.quotes = [q(300, 318.49, 318.51)];        // five minutes old
  const gone = pane();
  gone.quotes = [];
  // THE INVERSION: a fresh one-sided quote sitting on top of an older
  // two-sided one. The offer is gone; nothing can be said about a buy.
  const oneSided = pane();
  oneSided.quotes = [q(200, 318.49, 318.51), q(2, 318.40, null)];

  out.touch = {
    fresh:    !!fresh.touch(),
    recent:   !!recent.touch(),
    stale:    stale.touch(),
    gone:     gone.touch(),
    oneSided: oneSided.touch(),
    freshAge: fresh.touch() ? Math.round(fresh.touch().ageS) : null,
  };
  // A buy at 318.52 is through the 318.51 ask on the stale quote. Fresh says
  // so; stale must say "cannot tell", not "no".
  out.mktByAge = {
    fresh:    fresh.isMarketable('BUY', 318.52),
    stale:    stale.isMarketable('BUY', 318.52),
    oneSided: oneSided.isMarketable('BUY', 318.52),
    // And the passive case must not silently become "safe" off a stale quote.
    stalePassive: stale.isMarketable('BUY', 318.20),
  };

  // The banner has to say WHICH kind of not-knowing this is.
  const c1 = pane();
  c1.quotes = [q(300, 318.49, 318.51)];
  c1.armed = true;
  c1.rowTooFine = () => false;
  c1.place = async () => {};
  c1.clickLadder('buy', 318.20);
  out.staleBannerHeld = !!c1.confirm;
  out.staleBanner = c1.confirmText();

  const c2 = pane();
  c2.quotes = [];
  c2.armed = true;
  c2.rowTooFine = () => false;
  c2.place = async () => {};
  c2.clickLadder('buy', 318.20);
  out.noQuoteBanner = c2.confirmText();
}

// ── dragging a working order to a new price ──────────────────────────────
// A geometry by hand, so none of this needs a canvas. The gutter starts at
// padL + plotW = 408; the buy column is 408..434, price 434..488, sell
// 488..514. 318.50 sits at y=158 and 318.49 at y=173.
const GEOM = (() => {
  const padL = 8, padT = 8, plotW = 400, plotH = 300;
  const lo = 318.40, hi = 318.60;
  return {
    rect: { left: 0, top: 0, width: 514, height: 328 },
    padL, padT, padB: 20, padR: 106, plotW, plotH,
    tStart: 0, tEnd: 1, lo, hi,
    X: () => padL,
    Y: pp => padT + (1 - (pp - lo) / (hi - lo)) * plotH,
    priceAt: y => lo + (1 - (y - padT) / plotH) * (hi - lo),
  };
})();

function draggable(opts) {
  const c = quoted(pane());
  c.armed = (opts && 'armed' in opts) ? opts.armed : true;
  c.ladder = true;
  c.ladderCents = 1;
  c.geom = () => GEOM;
  c.working = [ord('1', (opts && 'mine' in opts) ? opts.mine : true,
                   { side: 'BUY', price: 318.50 })];
  c.ownIds = ['1'];
  return c;
}
const press = (c, x, y) => c.onDown({ clientX: x, clientY: y,
                                      preventDefault: () => {} });

{
  // The hit test, in both places an order is drawn.
  const c = draggable();
  out.grab = {
    onPlotLine:   !!c.orderAt(200, 158, GEOM),
    inBuyColumn:  !!c.orderAt(420, 158, GEOM),
    inSellColumn: !!c.orderAt(500, 158, GEOM),
    farAway:      !!c.orderAt(200, 220, GEOM),
  };
  out.grabUnarmed = !!draggable({ armed: false }).orderAt(200, 158, GEOM);
  out.grabForeign = !!draggable({ mine: false }).orderAt(200, 158, GEOM);
}
{
  // THE HAZARD: pressing on your own order inside the gutter used to place a
  // SECOND order at that row, because the ladder-zone test came first.
  const c = draggable();
  let placed = null;
  c.clickLadder = (zone, price) => { placed = { zone, price }; };
  press(c, 420, 158);
  out.pressOnMarker = { placed, dragging: !!c.dragOrder };

  // An empty row in the same column is still a click.
  const c2 = draggable();
  let placed2 = null;
  c2.clickLadder = (zone, price) => { placed2 = { zone, price }; };
  press(c2, 420, 218);
  out.pressOnEmptyRow = { placed: placed2, dragging: !!c2.dragOrder };
}
{
  // Drag, then drop somewhere passive: exactly one replace.
  const c = draggable();
  const moves = [];
  c.sendMove = async (o, price) => { moves.push({ id: o.order_id, price }); };
  press(c, 200, 158);
  c.onMove({ clientX: 200, clientY: 173 });
  out.dragPrice = c.dragOrder ? c.dragOrder.price : null;
  out.dragText = c.dragText();
  out.dragMarkerUnmoved = c.working[0].price;   // the REAL marker must not move
  c.onUp();
  out.dropPassive = moves;
  out.dropClearedDrag = c.dragOrder;
}
{
  // Put it back where it came from: no order, no call.
  const c = draggable();
  const moves = [];
  c.sendMove = async () => { moves.push(1); };
  press(c, 200, 158);
  c.onMove({ clientX: 200, clientY: 158 });
  c.onUp();
  out.dropUnmoved = moves.length;
}
{
  // Drop on a marketable row: the same banner the click path raises.
  // bid 318.49, so a BUY at 318.51+ is through the ask (318.51).
  const c = draggable();
  const moves = [];
  c.sendMove = async (o, price) => { moves.push({ id: o.order_id, price }); };
  press(c, 200, 158);
  c.onMove({ clientX: 200, clientY: 128 });     // 318.52, through the ask
  out.dragMktText = c.dragText();
  c.onUp();
  out.dropMktSent = moves.slice();
  out.dropMktHeld = c.confirm ? { kind: c.confirm.kind, price: c.confirm.price }
                              : null;
  c.confirmSend();
  out.dropMktAfterConfirm = moves.slice();
}
{
  // The order retired mid-drag. Replacing by a stale id would resurrect it.
  const c = draggable();
  const moves = [];
  c.sendMove = async () => { moves.push(1); };
  press(c, 200, 158);
  c.onMove({ clientX: 200, clientY: 173 });
  c.working = [];                                // a poll landed: it filled
  c.onUp();
  out.dropVanished = { moves: moves.length, err: c.brokerErr };
}
{
  // The ghost draws, and says what it is about to do.
  const c = draggable();
  press(c, 200, 158);
  c.onMove({ clientX: 200, clientY: 128 });      // marketable
  const { ctx, rec } = recorder();
  c.drawDragGhost(ctx, GEOM.padL, GEOM.padT, GEOM.plotW, GEOM.plotH,
                  GEOM.lo, GEOM.hi, GEOM.Y);
  out.ghostMkt = {
    texts: rec.texts.map(t => t.txt),
    hazard: rec.strokes.filter(x => String(x).includes('255,170,60')).length,
  };
  const c2 = draggable();
  press(c2, 200, 158);
  c2.onMove({ clientX: 200, clientY: 173 });     // passive
  const r2 = recorder();
  c2.drawDragGhost(r2.ctx, GEOM.padL, GEOM.padT, GEOM.plotW, GEOM.plotH,
                   GEOM.lo, GEOM.hi, GEOM.Y);
  out.ghostPassive = {
    texts: r2.rec.texts.map(t => t.txt),
    hazard: r2.rec.strokes.filter(x => String(x).includes('255,170,60')).length,
  };
}

// ── the order handle, the chart pan, and my own fills ────────────────────
const NOW = Date.now();
const iso = ms => new Date(ms).toISOString();
{
  // The handle is right-aligned inside the plot: padL+plotW-68-3 = 337..405,
  // 18px tall, centred on 318.50 at y=158 -> 149..167.
  const c = draggable();
  const r = c.handleRect(c.working[0], GEOM);
  out.handle = {
    rect: r && { x: Math.round(r.x), y: Math.round(r.y),
                 w: Math.round(r.w), h: Math.round(r.h) },
    onBody:   !!c.orderAt(r.x + 10, r.y + 9, GEOM),
    onXZone:  c.inHandleX(r.x + r.w - 6, r.y + 9, r),
    bodyIsNotX: c.inHandleX(r.x + 10, r.y + 9, r),
    // Off the handle AND off the order's own line, which stays grabbable
    // across the plot — the handle is an easier target, not a replacement.
    outside:  !!c.orderAt(r.x - 30, r.y + 70, GEOM),
    onOwnLine: !!c.orderAt(r.x - 30, r.y + 9, GEOM),
  };
  // FIXED IN PIXELS. A ladder row follows the zoom; this must not. Same
  // order, a span four times as wide.
  const wide = draggable();
  wide.geom = () => GEOM;
  const wideGeom = Object.assign({}, GEOM, {
    lo: 317.60, hi: 319.40,
    Y: pp => GEOM.padT + (1 - (pp - 317.60) / 1.80) * GEOM.plotH,
  });
  const r2 = wide.handleRect(wide.working[0], wideGeom);
  out.handleFixed = r2 && { w: Math.round(r2.w), h: Math.round(r2.h) };
}
{
  // The × cancels; the body drags. Two different gestures on one target.
  const c = draggable();
  const r = c.handleRect(c.working[0], GEOM);
  let cancelled = null;
  c.cancelOrder = (id) => { cancelled = id; };
  press(c, r.x + r.w - 6, r.y + 9);
  out.pressX = { cancelled, dragging: !!c.dragOrder };

  const c2 = draggable();
  const r2 = c2.handleRect(c2.working[0], GEOM);
  let cancelled2 = null;
  c2.cancelOrder = (id) => { cancelled2 = id; };
  press(c2, r2.x + 10, r2.y + 9);
  out.pressBody = { cancelled: cancelled2, dragging: !!c2.dragOrder };
}
{
  // THE PRECEDENCE. A press that lands on a handle must move the ORDER, not
  // the chart; a press on bare plot must move the chart.
  const c = draggable();
  const r = c.handleRect(c.working[0], GEOM);
  press(c, r.x + 10, r.y + 9);
  out.handleBeatsPan = { dragOrder: !!c.dragOrder, dragPan: !!c.dragPan };

  const c2 = draggable();
  press(c2, 120, 200);
  out.barePlotPans = { dragOrder: !!c2.dragOrder, dragPan: !!c2.dragPan };

  // A ladder row still places an order rather than panning.
  const c3 = draggable();
  let placed = null;
  c3.clickLadder = (zone, price) => { placed = { zone, price }; };
  press(c3, 420, 218);
  out.ladderStillPlaces = { placed, dragPan: !!c3.dragPan };
}
{
  // The pan itself: the price under the cursor stays under the cursor.
  const c = draggable();
  c.band = { lo: GEOM.lo, hi: GEOM.hi };
  const priceBefore = GEOM.priceAt(200);
  press(c, 120, 200);
  c.onMove({ clientX: 120, clientY: 230 });        // dragged DOWN 30px
  const band = c.band;
  const spanAfter = band.hi - band.lo;
  // Recompute what sits under the new cursor position with the new band.
  const priceAfter = band.lo
    + (1 - ((230 - GEOM.padT) / GEOM.plotH)) * spanAfter;
  out.pan = {
    held: Math.abs(priceAfter - priceBefore) < 1e-6,
    spanKept: Math.abs(spanAfter - (GEOM.hi - GEOM.lo)) < 1e-9,
    manual: c.bandManual,
    movedUp: band.lo > GEOM.lo,
  };
  c.onUp();
  out.panCleared = c.dragPan;

  // A manual band is not drifted away from, and recenter is the way back.
  const d = draggable();
  d.bandManual = true;
  d.band = { lo: 300.00, hi: 300.40 };            // far from the trades
  for (let i = 0; i < 40; i++) d.trades.push({ t: Date.now(), p: 318.50, s: 1, x: 4 });
  d.spanCents = 20;
  d.reband(false);
  out.manualHeld = { lo: d.band.lo, hi: d.band.hi };
  d.recenter();
  out.afterRecenter = { manual: d.bandManual, near: Math.abs(
    (d.band.lo + d.band.hi) / 2 - 318.50) < 1.0 };
}
{
  // MY FILLS, off the order records and never off the tape.
  const c = pane();
  c.working = [Object.assign(ord('1', true, { side: 'BUY', price: 318.50 }), {
    fills: [{ t: iso(NOW - 5000), price: 318.49, qty: 100 }],
  })];
  c.recent = [
    Object.assign(ord('2', true, { side: 'SELL', price: 318.60 }), {
      fills: [{ t: iso(NOW - 9000), price: 318.60, qty: 50 },
              // The SAME execution, as a poll can report it twice.
              { t: iso(NOW - 9000), price: 318.60, qty: 50 },
              // Outside the window.
              { t: iso(NOW - 900000), price: 318.10, qty: 25 }],
    }),
    // Not ours: a thinkorswim order carries no TA_ stamp.
    Object.assign(ord('3', false, { side: 'BUY', price: 318.40 }), {
      fills: [{ t: iso(NOW - 3000), price: 318.40, qty: 999 }],
    }),
  ];
  const got = c.myFills(NOW - 60000, NOW);
  out.fills = {
    n: got.length,
    prices: got.map(f => f.p).sort(),
    sides: got.map(f => f.buy),
    anyForeign: got.some(f => f.s === 999),
  };
  // And they draw as rings on the tape.
  const { ctx, rec } = recorder();
  c.drawMyFills(ctx, GEOM.padL, GEOM.padT, GEOM.plotW, GEOM.plotH,
                318.40, 318.70, NOW - 60000, NOW,
                t => GEOM.padL + GEOM.plotW * 0.5,
                pp => GEOM.padT + (1 - (pp - 318.40) / 0.30) * GEOM.plotH);
  out.fillsDrawn = rec.strokes.filter(
    x => String(x).includes('120,255,170')).length;

  // FIXED SIZE, whatever the quantity. My own fills are small - ten shares
  // against prints of one to three - so sizing them like a print made the
  // mark smallest exactly where it had to be found.
  const radii = (qty) => {
    const c2 = pane();
    c2.working = [Object.assign(ord('9', true, { side: 'BUY', price: 318.50 }), {
      fills: [{ t: iso(NOW - 1000), price: 318.55, qty }],
    })];
    c2.recent = [];
    const rr = recorder();
    c2.drawMyFills(rr.ctx, GEOM.padL, GEOM.padT, GEOM.plotW, GEOM.plotH,
                   318.40, 318.70, NOW - 60000, NOW,
                   t => GEOM.padL + GEOM.plotW * 0.5,
                   pp => GEOM.padT + (1 - (pp - 318.40) / 0.30) * GEOM.plotH);
    return rr.rec.arcs.map(a => Math.round(a.r * 10) / 10);
  };
  out.fillRadii = { ten: radii(10), thousand: radii(1000) };
  // And the middle is not filled in, so the print underneath shows through.
  const rr3 = recorder();
  const c3 = pane();
  c3.working = [Object.assign(ord('9', true, { side: 'BUY', price: 318.50 }), {
    fills: [{ t: iso(NOW - 1000), price: 318.55, qty: 10 }],
  })];
  c3.recent = [];
  c3.drawMyFills(rr3.ctx, GEOM.padL, GEOM.padT, GEOM.plotW, GEOM.plotH,
                 318.40, 318.70, NOW - 60000, NOW,
                 t => GEOM.padL + GEOM.plotW * 0.5,
                 pp => GEOM.padT + (1 - (pp - 318.40) / 0.30) * GEOM.plotH);
  out.fillFilled = rr3.rec.fills.filter(
    f => String(f.style).includes('120,255,170')).length;
}

// ── the release: one code path, one frame, nothing at the old price ──────
// THE GAP THIS CLOSES. Every other drop case stubs sendMove, so none of them
// could see what a real release does — which is exactly where the flash
// lived. This one runs the REAL sendMove with the call held open.
async function releaseFrames() {
  const c = draggable();
  c.quotes = [{ t: Date.now(), bp: 318.20, ap: 318.80 }];   // wide: passive
  let release = null;
  const paths = [];
  c.brokerCall = async (path) => {
    paths.push(String(path).split('?')[0]);
    await new Promise(r => { release = r; });
    return { ok: true, order_id: '2', rt_ms: 484 };
  };
  const shot = () => ({
    marker: c.working[0] ? c.shownPrice(c.working[0]) : null,
    ghost: c.dragOrder ? c.dragOrder.price : null,
    pending: c.pending ? c.pending.to : null,
  });

  press(c, 200, 158);
  c.onMove({ clientX: 200, clientY: 173 });          // 318.50 -> 318.49
  const during = shot();
  const done = c.onUp();
  const atRelease = shot();                          // SAME synchronous step
  await new Promise(r => setImmediate(r));
  const nextTick = shot();
  release();
  await done;
  return { during, atRelease, nextTick, after: shot(), paths };
}

// And nothing may be painted at the price it came from while it is pending.
function pendingPaint() {
  const c = draggable();
  // BOTH prices inside the visible range, or the target is skipped for
  // being off-screen and the test proves nothing.
  c.pending = { order_id: '1', from: 318.55, to: 318.45, qty: 100,
                side: 'BUY', state: 'sending' };
  c.working[0].price = 318.55;
  const { ctx, rec } = recorder();
  const yOf = pp => GEOM.padT + (1 - (pp - GEOM.lo) / (GEOM.hi - GEOM.lo)) * GEOM.plotH;
  c.drawPlotOrders(ctx, GEOM.padL, GEOM.padT, GEOM.plotW, GEOM.plotH,
                   GEOM.lo, GEOM.hi, yOf);
  const yFrom = Math.round(yOf(318.55));
  const yTo = Math.round(yOf(318.45));
  return {
    atFrom: rec.lines.filter(l => Math.round(l.y) === yFrom).length,
    atTo: rec.lines.filter(l => Math.round(l.y) === yTo).length,
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
  out.optOk = await optimistic({ ok: true, order_id: '2', rt_ms: 484 });
  out.optRefused = await optimistic({ ok: false, why: 'refused by the guards' });
  out.optUnknown = await optimistic({ ok: false, indeterminate: true,
                                      why: 'timed out' });
  out.rtSteal = await rtNotStolen();
  {
    const c = quoted(pane());
    c.pending = { order_id: '1', from: 318.50, to: 318.45, state: 'unknown' };
    c.unresolved = { state: 'gave-up' };
    c.clearUnresolved();
    out.clearedPending = c.pending;
  }
  out.release = await releaseFrames();
  out.pendingPaint = pendingPaint();
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
    if calls["move"] != ["replace"]:
        fail(f"a row move makes {calls['move']}, not the replace alone. "
             f"Measured 2026-09-03: the PUT is 484ms, the read-back 855ms, "
             f"and Schwab's propagation is NEGATIVE - the new price is in "
             f"the order list before the read-back asks. That call spent "
             f"855ms and a second of the minute's budget being told what "
             f"the PUT reply already said.")
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

    # -- dragging an order --------------------------------------------------
    gr = out["grab"]
    if not gr["onPlotLine"]:
        fail("a working order could not be grabbed on its line across the "
             "plot, which is where it is most visible against the tape.")
    if not gr["inBuyColumn"]:
        fail("a buy order could not be grabbed from the buy column of the "
             "ladder.")
    if gr["inSellColumn"]:
        fail("a BUY order was grabbable from the SELL column, where it is "
             "not drawn. The grab would come out of empty space.")
    if gr["farAway"]:
        fail("an order was grabbed from far away; the grab radius is wrong "
             "and ordinary clicks would start drags.")
    if out["grabUnarmed"]:
        fail("an order was draggable in an UNARMED pane. A drag ends in a "
             "replace, and a pane that cannot send one must not offer it.")
    if out["grabForeign"]:
        fail("an order not stamped as this app's was draggable — the same "
             "order primaryOrder() refuses to reprice.")

    pm = out["pressOnMarker"]
    if pm["placed"] is not None:
        fail(f"THE HAZARD. Pressing on your own order marker in the gutter "
             f"placed a second order at {pm['placed']}. The grab has to be "
             f"tested before the ladder zone, because the marker is drawn "
             f"inside it.")
    if not pm["dragging"]:
        fail("pressing on the order marker did not start a drag")
    pe = out["pressOnEmptyRow"]
    if pe["placed"] is None or pe["dragging"]:
        fail(f"pressing an EMPTY ladder row no longer places an order: "
             f"{pe}. The grab must not swallow the click path.")

    if abs((out["dragPrice"] or 0) - 318.49) > 1e-9:
        fail(f"the drag snapped to {out['dragPrice']}, not the 318.49 row. "
             f"A ghost that reads a price which cannot be sent is a ghost "
             f"that lied on the way down.")
    if abs(out["dragMarkerUnmoved"] - 318.50) > 1e-9:
        fail(f"the REAL marker moved to {out['dragMarkerUnmoved']} during "
             f"the drag. pane.working is the broker's answer; editing it in "
             f"place makes the pane claim a price Schwab never heard.")
    if "318.49" not in (out["dragText"] or ""):
        fail(f"the drag label does not name the target price: "
             f"{out['dragText']!r}")

    dp = out["dropPassive"]
    if len(dp) != 1:
        fail(f"a drop sent {len(dp)} replaces, not exactly one. The whole "
             f"point of dragging is that forty cents costs what one cent "
             f"costs.")
    elif abs(dp[0]["price"] - 318.49) > 1e-9:
        fail(f"the drop sent {dp[0]['price']}, not the row it was dropped "
             f"on")
    if out["dropClearedDrag"] is not None:
        fail("the drag state survived the drop")
    if out["dropUnmoved"]:
        fail("picking an order up and putting it back sent a replace. That "
             "is a call, a new order id and a queue position, for nothing.")

    if out["dropMktSent"]:
        fail(f"a drop on a marketable row sent immediately: "
             f"{out['dropMktSent']}. The drag path must ask the same "
             f"question the click path asks.")
    if not out["dropMktHeld"] or out["dropMktHeld"]["kind"] != "move":
        fail(f"the held drop is not recorded as a move: "
             f"{out['dropMktHeld']}")
    if len(out["dropMktAfterConfirm"]) != 1:
        fail("confirming the drop did not send the move")
    if "FILLS NOW" not in (out["dragMktText"] or ""):
        fail(f"the drag label gave no warning over a marketable row: "
             f"{out['dragMktText']!r}. The warning has to be on screen "
             f"BEFORE the drop, not only in the banner after it.")

    dv = out["dropVanished"]
    if dv["moves"]:
        fail("an order that filled or was cancelled mid-drag was still "
             "replaced by its stale id, which resurrects an order that is "
             "gone.")
    if "no longer working" not in (dv["err"] or ""):
        fail(f"the vanished-order drop said {dv['err']!r} rather than what "
             f"happened")

    if not any("318.52" in t for t in out["ghostMkt"]["texts"]):
        fail(f"the ghost did not label its target price: "
             f"{out['ghostMkt']['texts']}")
    if not out["ghostMkt"]["hazard"]:
        fail("the ghost over a marketable row was not drawn in the hazard "
             "colour")
    if out["ghostPassive"]["hazard"]:
        fail("a passive drag was drawn as a hazard, which is how a warning "
             "stops meaning anything")

    # -- the optimistic move ----------------------------------------------
    ok = out["optOk"]
    fl = ok["inFlight"]
    if abs(fl["shown"] - 318.45) > 1e-9:
        fail(f"IN FLIGHT the marker is drawn at {fl['shown']}, not the "
             f"318.45 that was asked for. Moving it on the reply instead of "
             f"the click is the 484ms this exists to remove.")
    if abs(fl["recordPrice"] - 318.50) > 1e-9:
        fail(f"the RECORD was overwritten in flight ({fl['recordPrice']}). "
             f"working is the broker's answer; the request belongs in "
             f"`pending` so the two can be told apart.")
    if not fl["pending"] or fl["pending"]["state"] != "sending":
        fail(f"nothing marked the in-flight move as unconfirmed: "
             f"{fl['pending']}. Without it the marker claims a price the "
             f"broker has not agreed to.")
    if abs(fl["pending"]["from"] - 318.50) > 1e-9:
        fail("the pending move does not remember where it came from, so a "
             "refusal has nowhere to put the marker back to")

    af = ok["after"]
    if abs(af["recordPrice"] - 318.45) > 1e-9:
        fail(f"after a 2xx the record still says {af['recordPrice']}. A 2xx "
             f"IS the broker's answer, so this is the moment it stops being "
             f"a guess.")
    if af["pending"] is not None:
        fail("the pending marker survived a confirmed move, so it would go "
             "on drawing as unconfirmed forever")
    if af["orderId"] != "2":
        fail(f"the new order id from the replace was not adopted "
             f"({af['orderId']}). A replace makes a NEW order and the reply "
             f"is the only link to it.")
    if "2" not in af["ownIds"] or "1" in af["ownIds"]:
        fail(f"ownIds did not follow the replace: {af['ownIds']}")

    rf = out["optRefused"]["after"]
    if abs(rf["recordPrice"] - 318.50) > 1e-9:
        fail(f"a REFUSED move left the marker at {rf['recordPrice']}. "
             f"Nothing moved at the broker, so nothing may be shown as "
             f"moved.")
    if rf["pending"] is not None:
        fail("a refused move stayed pending")
    if not rf["revert"] or abs(rf["revert"]["price"] - 318.45) > 1e-9:
        fail(f"a refused move snapped back with nothing to show it happened: "
             f"{rf['revert']}. A marker that quietly returns is a move you "
             f"think went through.")

    un = out["optUnknown"]["after"]
    if not un["pending"] or un["pending"]["state"] != "unknown":
        fail(f"a TIMED-OUT move was resolved one way or the other: "
             f"{un['pending']}. Whether it moved is unknown, and both "
             f"reverting and confirming assert something unfounded.")
    if abs(un["shown"] - 318.45) > 1e-9:
        fail(f"the unknown move is not drawn at the price that was asked "
             f"for ({un['shown']})")
    if un["unresolved"] != "looking":
        fail("a timed-out move did not raise the unresolved state")

    if out["clearedPending"] is not None:
        fail("clearing the unresolved state by hand left the pending marker "
             "up, with nothing remaining that could ever settle it")

    # -- the readout ------------------------------------------------------
    rt = ok["after"]["rt"]
    if rt is None or rt.get("visible") is None or rt.get("confirm") is None:
        fail(f"the move did not record both halves of its own timing: {rt}")
    elif rt["visible"] > rt["confirm"]:
        fail(f"the readout claims the marker moved after the broker "
             f"answered: {rt}")
    txt = ok["after"]["rtText"] or ""
    if "seen" not in txt or "confirmed" not in txt:
        fail(f"the readout does not separate what you feel from what was "
             f"confirmed: {txt!r}")

    st = out["rtSteal"]
    if st["afterMove"] != st["afterRead"]:
        fail(f"a read-back overwrote the action's timing: {st['afterMove']!r}"
             f" became {st['afterRead']!r}. That is exactly how the readout "
             f"came to show 68ms for a 1305ms move.")

    # -- the touch after hours --------------------------------------------
    t = out["touch"]
    if not t["fresh"]:
        fail("a one-second-old NBBO was rejected as stale; the bound is far "
             "too tight and every click in normal hours would be asked "
             "about")
    if not t["recent"]:
        fail("a twenty-second-old NBBO was rejected. The confirmation has to "
             "stay rare enough to be read, or it gets clicked through.")
    if t["stale"] is not None:
        fail(f"a FIVE-MINUTE-OLD NBBO was returned as the touch: {t['stale']}. "
             f"After hours that is a different market - narrower than the "
             f"one being traded - and the marketable test built on it "
             f"answers confidently and wrongly.")
    if t["gone"] is not None:
        fail("an empty quote buffer produced a touch out of nothing")
    if t["oneSided"] is not None:
        fail(f"a fresh one-sided quote was skipped in favour of an older "
             f"two-sided one: {t['oneSided']}. A missing ask is not silence, "
             f"it is the offer being gone, and preferring the stale quote is "
             f"backwards.")

    m = out["mktByAge"]
    if m["fresh"] is not True:
        fail("a buy through the ask on a fresh quote was not marketable")
    if m["stale"] is not None:
        fail(f"against a five-minute-old quote the marketable test answered "
             f"{m['stale']!r}. It has to be null - 'cannot say' - so the "
             f"click is asked about. Any definite answer here is a guess "
             f"about a market nobody has seen for five minutes.")
    if m["stalePassive"] is not None:
        fail(f"a PASSIVE-looking price answered {m['stalePassive']!r} off a "
             f"stale quote. This is the dangerous direction: a definite "
             f"'this will rest' about a touch that has moved.")
    if m["oneSided"] is not None:
        fail(f"with the offer gone, a buy answered {m['oneSided']!r} instead "
             f"of 'cannot say'")

    if not out["staleBannerHeld"]:
        fail("a click against a stale NBBO was not held for confirmation")
    sb = out["staleBanner"] or ""
    if "300s old" not in sb.replace("  ", " "):
        fail(f"the banner does not say how old the NBBO is: {sb!r}. 'No "
             f"NBBO' and 'an NBBO from five minutes ago' are different "
             f"things to be told.")
    if "thin session" not in sb:
        fail(f"the banner does not say the real spread is probably wider: "
             f"{sb!r}")
    nb = out["noQuoteBanner"] or ""
    if "no current NBBO" not in nb:
        fail(f"with no quotes at all the banner should say so, not quote an "
             f"age: {nb!r}")

    # -- the order handle --------------------------------------------------
    h = out["handle"]
    if not h["rect"]:
        fail("a working order in range produced no handle rectangle")
    else:
        if h["rect"]["w"] < 55 or h["rect"]["h"] < 14:
            fail(f"the handle is {h['rect']['w']}x{h['rect']['h']}px — too "
                 f"small to be the grab target it exists to be.")
    if not h["onBody"]:
        fail("the handle body does not hit-test as the order, so the target "
             "that was added to be grabbed cannot be grabbed")
    if not h["onXZone"]:
        fail("the × zone at the right end of the handle does not register")
    if h["bodyIsNotX"]:
        fail("the middle of the handle counts as the × — a drag would cancel "
             "the order instead of moving it, which is the worst possible "
             "confusion between these two.")
    if h["outside"]:
        fail("a point off both the handle and the order's line hit-tested "
             "as the order; the grab radius is wrong")
    if not h["onOwnLine"]:
        fail("the order's line across the plot stopped being grabbable. The "
             "handle is an easier target, not a replacement for it.")
    if out["handleFixed"] != {"w": 68, "h": 18}:
        fail(f"the handle changed size with the zoom: {out['handleFixed']}. "
             f"It is fixed in pixels precisely so a wide span does not leave "
             f"a four-pixel target for a drag that reprices real money.")

    if out["pressX"]["cancelled"] != "1" or out["pressX"]["dragging"]:
        fail(f"pressing the × did not cancel: {out['pressX']}")
    if out["pressBody"]["cancelled"] is not None:
        fail(f"pressing the handle BODY cancelled the order: "
             f"{out['pressBody']}")
    if not out["pressBody"]["dragging"]:
        fail("pressing the handle body did not start a drag")

    # -- gesture precedence ------------------------------------------------
    hp = out["handleBeatsPan"]
    if not hp["dragOrder"] or hp["dragPan"]:
        fail(f"a press on the order handle started a chart pan: {hp}. The "
             f"handle has to take the gesture first or dragging an order "
             f"drags the view instead.")
    bp = out["barePlotPans"]
    if bp["dragOrder"] or not bp["dragPan"]:
        fail(f"a press on bare plot did not start a pan: {bp}")
    ls = out["ladderStillPlaces"]
    if ls["placed"] is None or ls["dragPan"]:
        fail(f"a ladder row press panned the chart instead of placing an "
             f"order: {ls}")

    # -- the pan -----------------------------------------------------------
    pan = out["pan"]
    if not pan["held"]:
        fail("the price under the cursor did not stay under the cursor "
             "during the drag; the chart slides away from the grab")
    if not pan["spanKept"]:
        fail("panning changed the span — a drag moves the range, it does "
             "not zoom it")
    if not pan["movedUp"]:
        fail("dragging DOWN did not move the band up; the direction is "
             "inverted and the chart runs away from the hand")
    if not pan["manual"]:
        fail("a hand-placed band was not marked manual, so the very next "
             "frame's edge test snaps it back the moment price nears a "
             "boundary — which is what makes panning useless")
    if out["panCleared"] is not None:
        fail("the pan state survived the mouse-up")
    mh = out["manualHeld"]
    if abs(mh["lo"] - 300.00) > 1e-9:
        fail(f"the automatic recentre overrode a hand-placed band anyway: "
             f"{mh}")
    ar = out["afterRecenter"]
    if ar["manual"] or not ar["near"]:
        fail(f"recenter did not hand the band back to the automatic edge "
             f"test: {ar}")

    # -- my fills ----------------------------------------------------------
    f = out["fills"]
    if f["n"] != 2:
        fail(f"{f['n']} fills came through, expected 2: one working-order "
             f"fill and one recent, with the duplicate collapsed and the "
             f"out-of-window one dropped. Prices: {f['prices']}")
    if f["anyForeign"]:
        fail("a fill from an order this app did not place was drawn as "
             "mine. The tape already shows everyone else's prints; the "
             "point of this mark is that it is ours.")
    if f["sides"] != [True, False] and f["sides"] != [False, True]:
        fail(f"the fills lost their side: {f['sides']}")
    if not out["fillsDrawn"]:
        fail("no fill ring was drawn, so the answer to 'was anything "
             "trading at my price' is still not on screen")

    ten = out["fillRadii"]["ten"]
    thousand = out["fillRadii"]["thousand"]
    if not ten:
        fail("a 10-share fill drew no ring at all")
    elif ten != thousand:
        fail(f"the fill ring changes size with the quantity: 10 shares gives "
             f"{ten}, 1000 gives {thousand}. My own fills are the small ones "
             f"- ten shares against prints of one to three - so sizing them "
             f"like a print makes the mark smallest exactly where it has to "
             f"be found.")
    elif max(ten) < 8:
        fail(f"the fill ring is {max(ten)}px at its widest, which is not a "
             f"mark you can find at a glance on a busy tape.")
    if out["fillFilled"]:
        fail("the fill mark is filled in, not an open ring - it covers the "
             "print underneath, which is the thing it is meant to be "
             "pointing at.")

    # -- the release ------------------------------------------------------
    rel = out["release"]
    if rel["during"]["ghost"] != 318.49:
        fail(f"the ghost did not follow the drag: {rel['during']}")
    if rel["during"]["marker"] != 318.50:
        fail(f"the record moved during the drag: {rel['during']}")

    ar = rel["atRelease"]
    if ar["ghost"] is not None:
        fail(f"the ghost survived the release: {ar}")
    if ar["pending"] != 318.49:
        fail(f"the release did not set pending: {ar}. The drop would then "
             f"fall back to `working`, which still holds the old price, and "
             f"the marker sits where it started until the PUT lands ~484ms "
             f"later.")
    if ar["marker"] != 318.49:
        fail(f"the marker is not at the target in the SAME step the ghost "
             f"went away: {ar}. Ghost off and pending on have to be one "
             f"render or there is a frame showing the old price.")
    if rel["nextTick"]["marker"] != 318.49:
        fail(f"the marker fell back after the release: {rel['nextTick']}")
    if rel["after"]["marker"] != 318.49 or rel["after"]["pending"] is not None:
        fail(f"the confirmed state is wrong: {rel['after']}")
    if rel["paths"] != ["replace"]:
        fail(f"the drop took a different route to the broker than a nudge: "
             f"{rel['paths']}. Both must go through sendMove, or they drift "
             f"apart again exactly as they did here.")

    pp = out["pendingPaint"]
    if pp["atFrom"]:
        fail(f"{pp['atFrom']} line(s) drawn at the price the order came "
             f"from while the move is pending. On a drag that appears at "
             f"the gesture's starting price for the whole flight and then "
             f"vanishes, which is indistinguishable from the order snapping "
             f"back and then arriving. THIS WAS THE FLASH.")
    if not pp["atTo"]:
        fail("nothing is drawn at the pending target price")

    if bad:
        print(f"\nreconcile cases FAILED: {bad}")
        return 1
    print(f"live pane: foreign orders refused; unresolved window "
          f"{out['giveUpTries'] * 2}s vs a {APPEARED_AFTER_S:.0f}s "
          f"appearance; marketable clicks, nudges and drops held for "
          f"confirmation; a move is ONE call and the marker moves on the "
          f"click, marked pending, reverting visibly on a refusal; the "
          f"order poll is {out['pollsTrading']}/min trading and "
          f"{out['pollsWatching']}/min watching")
    return 0


if __name__ == "__main__":
    sys.exit(main())
