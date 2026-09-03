/* Equities Live — the scrolling tape.
 *
 * CANVAS, NOT A CHART LIBRARY. The window scrolls continuously and every
 * print is drawn at its own timestamp; a charting library's update path
 * rebuilds scales and re-lays out on every frame, and the first thing it
 * would do to make that affordable is bucket the x axis. Bucketing is the
 * one thing this plot exists to avoid.
 *
 * NOTHING IS AGGREGATED ANYWHERE IN THIS FILE. Seven prints sharing a
 * millisecond are seven bubbles at the same x. They overlap; that overlap IS
 * the reading — a single marketable order sweeping seven venues.
 *
 * TWO OBJECTS, NOT ONE. `lvPane()` owns everything about one plot: its
 * symbol, its buffers, its window, its band, its lines, its canvas. The
 * Alpine component owns the socket, the frame loop and the list of panes, and
 * nothing else. Panes were previously the component, which is why there could
 * only ever be one.
 */

const LV_BLUE = '#3498db';
const LV_PINK = '#e84393';

/* NEUTRAL BY DEFAULT.
 *
 * The bubbles were the same blue as the bid line, which reads as "buys"
 * before any decision to read it that way. A print is a print; the colour
 * says nothing until the bichrome toggle is deliberately turned on. */
const LV_TRADE_FILL = 'rgba(206,212,220,0.30)';
const LV_TRADE_RIM  = 'rgba(228,233,240,0.80)';
/* BY AGGRESSOR, NOT BY PROXIMITY.
 *
 * A print above the mid is a buyer lifting the offer, so it is BLUE. Below
 * the mid is a seller hitting the bid, so PINK.
 *
 * These were the other way round, matching the line each dot sat nearest —
 * which is a different claim: "which side of the book is this near" rather
 * than "who crossed the spread". The aggressor is the one that carries
 * information, so a blue dot now sits near the pink ask line and that is
 * correct rather than a mistake. */
/* How long an unresolved placement is looked for before the pane says it
 * could not confirm. MEASURED, not guessed: a probe order placed on
 * 2026-09-03 was listed by Schwab ONE SECOND after the 201. This was 15s,
 * which blocked the pane for fourteen seconds after the answer had arrived —
 * and the block is total, so that is fourteen seconds of not being able to
 * trade the name. Five gives the one-second observation five times its
 * margin and still ends while the move that prompted the order is alive. */
const LV_RESOLVE_WINDOW_S = 5;

/* The poll that drives tryResolve, in seconds. The give-up test counts
 * looks, so it needs this to turn them into elapsed time. */
const LV_RESOLVE_POLL_S = 2;

/* MARKETABLE. Not a side colour — a hazard colour, deliberately outside the
 * blue/pink vocabulary so it cannot be read as "the other side". These rows
 * fill the instant they are clicked. */
/* How near the cursor has to be to a working order to grab it, in pixels.
 * Wider than the 6px used for a price line: this one ends in a replace, and
 * missing the grab silently places a NEW order at the row instead. */
const LV_GRAB_PX = 7;

/* The order poll: fast while trading, slow while only watching. 2s is 30
 * calls a minute against the 90 ordinary traffic may spend; 6s is 10. */
const LV_ORDER_POLL_MS      = 2000;
const LV_ORDER_POLL_IDLE_MS = 6000;

const LV_MKT_FILL  = 'rgba(255,140,0,0.16)';
const LV_MKT_HOT   = 'rgba(255,140,0,0.60)';
const LV_MKT_HATCH = 'rgba(255,170,60,0.55)';

const LV_BUY_FILL  = 'rgba(52,152,219,0.30)';    // lifted the offer
const LV_BUY_RIM   = 'rgba(120,200,255,0.85)';
const LV_SELL_FILL = 'rgba(232,67,147,0.30)';    // hit the bid
const LV_SELL_RIM  = 'rgba(240,130,180,0.85)';

// Price lines: BRIGHT GREY, not yellow. Yellow reads as a signal, and these
// carry no signal — they are a place the eye is holding.
const LV_LINE      = 'rgba(214,218,224,0.90)';
const LV_LINE_DRAG = 'rgba(255,255,255,0.98)';
const LV_LINE_TEXT = '#dfe3e8';
const LV_LINE_COLD = '#8a8a8a';

/* Ladder row heights, in pixels.
 *
 * Under MIN a click is refused — an order landing a cent from where it was
 * aimed is not an acceptable outcome of a near-miss. GOOD is what the
 * automatic increment aims for, comfortably above the refusal line so the
 * rows do not flicker between clickable and not as the band steps.
 */
const LV_ROW_MIN_PX = 5;
const LV_ROW_GOOD_PX = 8;

function lvFmtClock(ms) {
  const d = new Date(ms);
  return String(d.getHours()).padStart(2, '0') + ':'
       + String(d.getMinutes()).padStart(2, '0') + ':'
       + String(d.getSeconds()).padStart(2, '0');
}

function lvFmtNum(v) {
  if (v == null || !isFinite(v)) return '—';
  if (v >= 1000000) return (v / 1000000).toFixed(1) + 'M';
  if (v >= 10000) return Math.round(v / 1000) + 'k';
  if (v >= 1000) return (v / 1000).toFixed(1) + 'k';
  if (v >= 100) return String(Math.round(v));
  return v >= 10 ? v.toFixed(0) : v.toFixed(1);
}

/* One plot. Created by the component, driven by the component's frame loop,
 * and otherwise self-contained — including its own canvas, whose id carries
 * the pane's own number.
 *
 * `send` is injected rather than reached for: the socket belongs to the
 * component, and a pane that could open its own would be a second upstream
 * subscription for the same symbol. */
window.lvPane = function (id, send) {
  return {
    id,
    send: send || (() => {}),

    symbol: '',
    pending: '',
    refused: '',

    windowS: 180,
    // Half-height of the price window, in cents.
    spanCents: 30,
    // THE BAND IS EXPLICIT AND STICKY. It was an `anchor` recomputed from the
    // last print on every frame, which meant a single off-price odd lot moved
    // the whole axis and every bubble jumped vertically under a flat tape —
    // six consecutive frames read 320.78, 320.44, 320.70, 320.44, 321.04,
    // 320.83 with price essentially still.
    //
    // Now it holds until the reference price genuinely approaches an edge,
    // then steps ONCE to a snapped grid. Never per frame, never on a timer.
    band: null,          // {lo, hi}
    bandSteps: 0,        // how many times it has moved, so drift is visible
    paused: false,
    frozenEnd: null,
    showQuotes: true,

    /* OFF BY DEFAULT, and that is the claim being made.
     *
     * Colouring a print by where it sat in the spread asserts that placement
     * carries a side, which has not been established here. Grey asserts
     * nothing. The toggle exists to look, not to conclude. */
    bichrome: false,

    trades: [],
    quotes: [],
    // Every print under the cursor, not the nearest one. At these rates
    // several share a millisecond and land within a pixel or two, and a
    // single number there is ambiguous between one 400-share print and four
    // of 100.
    hover: null,          // {x, y, recs: [...]}
    // Price lines the user placed. Anchored to a PRICE, so they hold still
    // while the plot scrolls under them — the question they answer is "has
    // anything traded at 318.52 in the last three minutes", which is about
    // the level and not about when.
    lines: [],
    dragIdx: -1,
    /* A working order being dragged to a new price.
     *
     * {order, price} — `order` is the record as it was when grabbed, `price`
     * the row under the cursor right now. The ORDER IS NOT MUTATED while
     * dragging: pane.working is the broker's answer, and editing it in place
     * would make the pane claim a price Schwab has never heard of. The drag
     * is drawn as a separate ghost and the real marker stays where the
     * broker last said it was. */
    dragOrder: null,

    // Both rates over a FIXED sixty seconds. See the note in rates().
    tpm: null,
    spm: null,
    tpmPartial: false,

    // This name's own normal for the current 15-minute bucket, from the
    // pipeline's stored history. Never computed here.
    norm: null,
    normFetching: false,
    normAt: 0,

    // ── trading ──────────────────────────────────────────────────────────
    /* ARMED IS OFF AND STAYS OFF until switched on per pane, per session.
     * It is never persisted: a pane that came back armed after a reload
     * would be a pane that can trade because a page was left open. */
    armed: false,
    qty: 100,
    // Cents per ladder row. The row height encodes the increment, which is
    // why nudging is "one row" and not "one cent" — the same key does the
    // right thing at any zoom.
    ladderCents: 1,
    ladderStops: [1, 2, 5],
    // What the BROKER says, replaced wholesale on every read. Never merged
    // with a local guess: a confident wrong list of working orders is the
    // failure that costs money.
    position: null,       // {qty, avg, day_pl}
    working: [],          // [{order_id, side, qty, price, status}]
    recent: [],           // terminal orders, so a fill is visible
    limits: null,         // the shared 120/min budget, as the server sees it
    // TWO AGES, because the two reads happen at different rates and one of
    // them matters far more. Orders in this account live 1-6 seconds;
    // positions move only when one fills.
    ordersAt: 0,          // when the working list was last confirmed, ms
    positionsAt: 0,
    brokerErr: '',
    lastRt: null,         // {visible, confirm, broker} for the last action

    /* A MOVE THE BROKER HAS NOT ANSWERED YET.
     *
     * MEASURED, and the measurement is why this exists: a replace is 484ms
     * at the PUT and 855ms more for the read-back that used to follow it,
     * while Schwab's own propagation is NEGATIVE — the new price is already
     * in the order list before the read-back asks for it. So the whole
     * 1305ms wait was our two round trips confirming something Schwab had
     * already accepted, and the marker sat at the old price for all of it.
     *
     * The marker now moves on the CLICK and this records that it is not yet
     * confirmed. It is drawn differently for exactly as long as that is
     * true — the pane never claims a price the broker has not agreed to, it
     * shows a request AS a request. {order_id, from, to, qty, side, sentAt,
     * state: 'sending' | 'unknown'}. */
    pending: null,

    // A move that came back REFUSED, held briefly so the marker snapping
    // back is something you see happen rather than something you missed.
    revert: null,         // {price, until}
    lastAction: '',
    /* A PLACEMENT WHOSE ANSWER NEVER ARRIVED.
     *
     * Not an error — an unknown. The order may be resting at Schwab right
     * now. While this is set the pane will not place or reprice, because a
     * second order sent on top of an unknown first one is how a timeout
     * becomes a double position. Cancel and flatten stay available: they are
     * how you get out of exactly this. */
    unresolved: null,     // {side, qty, price, sentAt, tries, state, why}
    // A marketable click, held until it is confirmed or dropped. Never
    // persisted and never auto-confirmed: see askFirst().
    confirm: null,        // {kind, side, price, qty, unknown, bid, ask, order}
    /* Order ids this pane created, newest last.
     *
     * A replace returns a NEW id and Schwab provides no link from the old
     * order to it — `replacingOrderCollection` is null on every REPLACED
     * order in the account. So the id from the PUT is the only handle, and
     * throwing it away meant relying on there being exactly one working
     * order to guess from. */
    ownIds: [],
    busy: false,
    // Row the cursor is over in the ladder, and which zone.
    ladderHover: null,    // {price, zone: 'buy'|'sell'}
    ladder: false,        // the gutter costs plot width, so it is opt-in
    ladderManual: false,  // a hand-picked increment, until the next zoom

    canvasId() { return 'lv-canvas-' + this.id; },
    cv() { return document.getElementById(this.canvasId()); },

    // ── subscription ─────────────────────────────────────────────────────
    watch(sym, silent) {
      sym = (sym || '').trim().toUpperCase();
      if (!sym) return;
      this.refused = '';
      const prev = this.symbol;
      this.symbol = sym;
      if (!silent) {
        this.trades = []; this.quotes = []; this.band = null;
        this.norm = null; this.normAt = 0;
      }
      // Refcounted BY THE COMPONENT. Two panes on one symbol share a single
      // upstream subscription, and the first pane to close must not
      // unsubscribe the other one's tape.
      this.send({ kind: 'watch', symbol: sym, prev: silent ? null : prev,
                  window_s: this.windowS });
    },

    /* Kept well BEYOND the display window.
     *
     * This trimmed at `windowS`, so widening the window could never recover
     * what had already been discarded — zoom out from one minute to three and
     * the extra two minutes were gone for good, while the axis went on
     * claiming to cover them. The retention horizon is now fixed and generous;
     * the display window is a view onto it, not the thing that bounds it.
     *
     * The server holds the real ceiling (15 minutes, capped in count too);
     * this only keeps the browser's copy from growing without bound. */
    retainS() { return Math.min(900, Math.max(600, this.windowS * 3)); },

    prune() {
      const cutoff = Date.now() - this.retainS() * 1000;
      while (this.trades.length && this.trades[0].t < cutoff) this.trades.shift();
      while (this.quotes.length && this.quotes[0].t < cutoff) this.quotes.shift();
    },

    /* How much of the displayed window actually has data behind it.
     *
     * The hub only buffers a symbol while someone is watching it, so the
     * first subscribe starts from empty and a three-minute axis sits over
     * however many seconds have passed. That is not eviction and not a
     * failure, but an axis that claims three minutes while showing fifty
     * seconds should say so rather than leave it to be discovered. */
    bufferedS() {
      if (!this.trades.length) return 0;
      return Math.min(this.windowS,
                      Math.round((Date.now() - this.trades[0].t) / 1000));
    },

    // ── geometry ─────────────────────────────────────────────────────────
    resize() {
      const cv = this.cv();
      if (!cv) return;
      const r = cv.parentElement.getBoundingClientRect();
      const dpr = window.devicePixelRatio || 1;
      cv.width = Math.max(1, Math.floor(r.width * dpr));
      cv.height = Math.max(1, Math.floor(r.height * dpr));
      cv.style.width = r.width + 'px';
      cv.style.height = r.height + 'px';
    },

    /* The reference price, robust to one odd print.
     *
     * The last trade alone was the trigger, and a single off-price odd lot at
     * a stale venue was enough to move the axis. A median over the recent
     * prints ignores that; the NBBO mid is used in preference where quotes
     * are on, since it is what the band should be centred on anyway. */
    refPrice() {
      if (this.showQuotes && this.quotes.length) {
        const q = this.quotes[this.quotes.length - 1];
        if (q.bp && q.ap) return (q.bp + q.ap) / 2;
      }
      const n = this.trades.length;
      if (!n) return null;
      const tail = this.trades.slice(Math.max(0, n - 21))
        .map(t => t.p).sort((a, b) => a - b);
      return tail[Math.floor(tail.length / 2)];
    },

    lastPrice() {
      return this.trades.length ? this.trades[this.trades.length - 1].p : null;
    },

    /* The touch: the best bid and offer as last seen.
     *
     * Quotes stream whether or not `showQuotes` is on — that flag only
     * decides whether the band is DRAWN — so this is available even when the
     * NBBO is not on screen. Returns null when either side is missing, and
     * every caller treats null as "cannot say", never as "safe".
     */
    touch() {
      for (let i = this.quotes.length - 1; i >= 0; i--) {
        const q = this.quotes[i];
        if (q && q.bp && q.ap) return { bid: q.bp, ask: q.ap };
      }
      return null;
    },

    /* Would a limit at this price trade immediately against the touch?
     *
     * A BUY at or above the ask lifts the offer; a SELL at or below the bid
     * hits the bid. Both are valid limit orders and both fill instantly,
     * which is the whole problem: a passive strategy never wants either, and
     * nothing on the way to sending one said so.
     *
     * UNKNOWN IS NOT SAFE. With no quote this returns null rather than
     * false, and the caller asks anyway — the one thing that must not happen
     * is a marketable order slipping through because the quote was missing.
     */
    isMarketable(side, price) {
      const t = this.touch();
      if (!t) return null;
      const px = Number(price);
      if (!isFinite(px)) return null;
      return String(side).toUpperCase().startsWith('BUY')
        ? px >= t.ask
        : px <= t.bid;
    },

    /* The grid the band snaps to.
     *
     * Snapping matters as much as the edge test: a band recomputed to
     * centre on whatever the price happened to be produces labels like
     * 320.78 and then 321.04, which look like movement. Snapped to a round
     * increment the labels repeat, so a step is visible AS a step and
     * everything between steps is genuinely still. */
    gridStep() {
      const c = this.spanCents;
      return (c <= 10 ? 2 : c <= 30 ? 5 : c <= 60 ? 10 : 25) / 100;
    },

    /* RECENTRE ONLY AT THE EDGE. Never per frame, never on a timer.
     *
     * The band holds until the reference price crosses INTO the outer 15% of
     * it, then steps once to a snapped band centred on that price. Between
     * steps the axis does not move at all, which is the whole point: position
     * is read against a still background. */
    reband(force) {
      const p = this.refPrice();
      if (p == null) return;
      const half = this.spanCents / 100;
      if (force || !this.band) {
        this.band = this.snapBand(p, half);
        return;
      }
      // Re-derive the band when the SPAN changed, without treating that as a
      // recentre — a zoom is a deliberate act, not drift.
      if (Math.abs((this.band.hi - this.band.lo) - 2 * half) > 1e-9) {
        this.band = this.snapBand((this.band.lo + this.band.hi) / 2, half);
        return;
      }
      const inner = 0.85;
      const mid = (this.band.lo + this.band.hi) / 2;
      if (Math.abs(p - mid) > half * inner) {
        this.band = this.snapBand(p, half);
        this.bandSteps += 1;
      }
    },

    snapBand(centre, half) {
      const g = this.gridStep();
      const lo = Math.floor((centre - half) / g) * g;
      return { lo, hi: lo + Math.ceil((2 * half) / g) * g };
    },

    yRange() {
      if (!this.band) {
        const p = this.refPrice() || 0;
        const h = this.spanCents / 100;
        return { lo: p - h, hi: p + h };
      }
      return { lo: this.band.lo, hi: this.band.hi };
    },

    /* Screen geometry, in one place.
     *
     * draw() and the pointer handlers each computed their own padding, window
     * bounds and scales; two copies of a mapping drift, and a hover that
     * disagrees with what is drawn is worse than no hover. */
    /* The right gutter's width, and how it is divided.
     *
     * ONE DEFINITION, used by the drawing and by the hit-testing. The ladder
     * rows have to line up with the plot's y-axis exactly — that is the
     * requirement — so they are drawn on the same canvas with the same Y()
     * rather than as DOM alongside it. A second copy of the mapping is
     * precisely the drift that a click on the wrong row would come from.
     *
     *   | plot | BUY | PRICE | SELL |
     */
    gutter() {
      if (!this.ladder) return { w: 58, buy: 0, price: 6, sell: 0, cols: false };
      return { w: 106, buy: 26, price: 54, sell: 26, cols: true };
    },

    geom() {
      const cv = this.cv();
      if (!cv) return null;
      const r = cv.getBoundingClientRect();
      const padL = 8, padT = 8, padB = 20;
      const padR = this.gutter().w;
      const plotW = Math.max(10, r.width - padL - padR);
      const plotH = Math.max(10, r.height - padT - padB);
      const tEnd = this.paused ? (this.frozenEnd || Date.now()) : Date.now();
      const tStart = tEnd - this.windowS * 1000;
      const { lo, hi } = this.yRange();
      return {
        rect: r, padL, padR, padT, padB, plotW, plotH, tStart, tEnd, lo, hi,
        X: t => padL + ((t - tStart) / (tEnd - tStart)) * plotW,
        Y: p => padT + (1 - (p - lo) / (hi - lo)) * plotH,
        priceAt: y => lo + (1 - (y - padT) / plotH) * (hi - lo),
      };
    },

    // ── the ladder ───────────────────────────────────────────────────────
    /* Row prices, top down, on the selected increment.
     *
     * Snapped to the increment rather than to the band's edge, so a row
     * boundary does not move when the band steps — the row a limit sits in
     * has to be the same row after a recentre. */
    ladderRows(lo, hi) {
      const step = this.ladderCents / 100;
      const first = Math.ceil(lo / step) * step;
      const rows = [];
      // Bounded. At a 500-cent span and 1-cent rows this would be 1,000
      // rows of two pixels, which is not a thing anyone can click.
      for (let p = first; p <= hi + 1e-9 && rows.length < 240; p += step) {
        rows.push(Math.round(p * 100) / 100);
      }
      return rows;
    },

    /* Whether the increment can be clicked at this zoom, and what to say.
     *
     * A row under about five pixels is not a click target, and silently
     * letting it be one is how an order goes in a cent from where it was
     * aimed. The pane says so and refuses rather than guessing. */
    rowPx(cents) {
      const g = this.geom();
      if (!g) return 0;
      const step = (cents || this.ladderCents) / 100;
      const rows = Math.max(1, (g.hi - g.lo) / step);
      return g.plotH / rows;
    },
    rowTooFine() { return this.rowPx() < LV_ROW_MIN_PX; },

    /* THE INCREMENT FOLLOWS THE ZOOM, and a manual choice wins until the
     * next zoom.
     *
     * Selectable alone was not enough: zoomed out to a 60-cent range, 1-cent
     * rows are 121 rows at 4 pixels — inert, unclickable, and drawn as a
     * striped block that looks like a ladder. This picks the finest
     * increment whose rows are still worth clicking, so zooming out steps
     * 1 -> 2 -> 5 on its own.
     *
     * A manual pick is honoured until the next zoom, because overriding it
     * mid-adjustment would fight the person doing the adjusting; a zoom is
     * the natural moment to hand control back. */
    autoLadder() {
      if (this.ladderManual) return;
      for (const c of this.ladderStops) {
        if (this.rowPx(c) >= LV_ROW_GOOD_PX) { this.ladderCents = c; return; }
      }
      this.ladderCents = this.ladderStops[this.ladderStops.length - 1];
    },

    ladderZone(mx, g) {
      if (!this.ladder || !g) return null;
      const x0 = g.padL + g.plotW;
      const gu = this.gutter();
      if (mx >= x0 && mx < x0 + gu.buy) return 'buy';
      if (mx >= x0 + gu.buy + gu.price
          && mx < x0 + gu.buy + gu.price + gu.sell) return 'sell';
      return null;
    },

    /* The row a y lands in, snapped to the increment. */
    rowAt(y, g) {
      const step = this.ladderCents / 100;
      return Math.round(g.priceAt(y) / step) * step;
    },

    /* The working order under the cursor, or null.
     *
     * Grabbable in BOTH places it is drawn: the line across the plot and the
     * marker in the ladder gutter. Only ours, and only while armed — a drag
     * ends in a replace, and a pane that cannot send one must not offer the
     * gesture. `from_api` keeps the grab off an order placed in thinkorswim
     * for the same reason primaryOrder() refuses to reprice one.
     */
    orderAt(mx, my, g) {
      if (!this.armed || !g) return null;
      const inPlot = mx >= g.padL && mx < g.padL + g.plotW;
      const zone = this.ladderZone(mx, g);
      if (!inPlot && !zone) return null;
      let best = null, bestD = LV_GRAB_PX;
      for (const o of this.working) {
        if (o.price == null || !o.from_api) continue;
        // In the gutter, only the column the order actually sits in: a buy
        // marker is in the buy column, and grabbing "through" the sell
        // column would be grabbing something not drawn there.
        if (zone) {
          const buy = String(o.side || '').toUpperCase().startsWith('BUY');
          if (zone !== (buy ? 'buy' : 'sell')) continue;
        }
        const d = Math.abs(g.Y(Number(o.price)) - my);
        if (d < bestD) { bestD = d; best = o; }
      }
      return best;
    },

    // ── the two rates ────────────────────────────────────────────────────
    /* A ROLLING SIXTY SECONDS, not the window's count over the window's
     * length. That read a third of the true rate on a 3-minute window and
     * climbed from zero as the buffer filled — 8, 10, 12, 15, 21 against a
     * verified 56, which is exactly 56/3 converging. A rate has to be over a
     * fixed interval, and one minute is the interval the reference is quoted
     * in — which matters twice over now that the comparison band is quoted
     * per minute too. */
    rates(tEnd) {
      const minuteAgo = tEnd - 60000;
      let n = 0, sh = 0;
      for (let i = this.trades.length - 1; i >= 0; i--) {
        const t = this.trades[i];
        if (t.t > tEnd) continue;
        if (t.t < minuteAgo) break;
        n += 1;
        sh += (t.s || 0);
      }
      // Under a minute of data cannot report a per-minute rate honestly.
      const ready = this.bufferedS() >= 55;
      this.tpm = ready ? n : null;
      this.spm = ready ? sh : null;
      this.tpmPartial = !ready;
    },

    // ── the frame ────────────────────────────────────────────────────────
    draw() {
      const cv = this.cv();
      if (!cv) return;
      const ctx = cv.getContext('2d');
      const dpr = window.devicePixelRatio || 1;
      const W = cv.width, H = cv.height;
      ctx.setTransform(1, 0, 0, 1, 0, 0);
      ctx.clearRect(0, 0, W, H);
      ctx.scale(dpr, dpr);
      const w = W / dpr, h = H / dpr;

      const padL = 8, padT = 8, padB = 20;
      const padR = this.gutter().w;
      const plotW = Math.max(10, w - padL - padR);
      const plotH = Math.max(10, h - padT - padB);

      if (!this.paused) this.reband(false);
      const tEnd = this.paused ? (this.frozenEnd || Date.now()) : Date.now();
      if (!this.paused) this.frozenEnd = null;
      else if (!this.frozenEnd) this.frozenEnd = Date.now();
      const tStart = tEnd - this.windowS * 1000;
      const { lo, hi } = this.yRange();

      const X = t => padL + ((t - tStart) / (tEnd - tStart)) * plotW;
      const Y = p => padT + (1 - (p - lo) / (hi - lo)) * plotH;

      this.drawGrid(ctx, padL, padT, plotW, plotH, lo, hi, tStart, tEnd, X, Y, w);

      // ── NBBO: a REGION, then two lines ────────────────────────────────
      //
      // Where a print sat relative to bid and ask is the entire question, and
      // two thin muted lines made that something to trace rather than see.
      // The spread is filled so it reads as a band the prints sit inside or
      // outside of — FAINTLY, at background weight, with the edges bright
      // enough to locate exactly. It was brighter and competed with the tape.
      if (this.showQuotes && this.quotes.length) {
        const vis = this.nbboSteps(tStart, tEnd);
        if (vis.length > 1) {
          // A STEP FILL, not a ribbon between quote points. Interpolating
          // between quotes draws a diagonal through prices that were never
          // quoted; the NBBO holds flat until it changes.
          ctx.beginPath();
          ctx.moveTo(X(vis[0].t), Y(vis[0].ap));
          for (let i = 1; i < vis.length; i++) {
            ctx.lineTo(X(vis[i].t), Y(vis[i - 1].ap));
            ctx.lineTo(X(vis[i].t), Y(vis[i].ap));
          }
          for (let i = vis.length - 1; i > 0; i--) {
            ctx.lineTo(X(vis[i].t), Y(vis[i].bp));
            ctx.lineTo(X(vis[i].t), Y(vis[i - 1].bp));
          }
          ctx.lineTo(X(vis[0].t), Y(vis[0].bp));
          ctx.closePath();
          ctx.fillStyle = 'rgba(150,170,190,0.045)';
          ctx.fill();
        }
        ctx.lineWidth = 1.6;
        for (const [key, col] of [['bp', 'rgba(130,190,235,0.95)'],
                                  ['ap', 'rgba(235,150,190,0.95)']]) {
          ctx.beginPath();
          vis.forEach((q, i) => {
            const x = X(q.t), y = Y(q[key]);
            if (i === 0) { ctx.moveTo(x, y); return; }
            ctx.lineTo(x, Y(vis[i - 1][key]));   // hold, then step
            ctx.lineTo(x, y);
          });
          ctx.strokeStyle = col;
          ctx.stroke();
        }
      }

      // ── the tape ───────────────────────────────────────────────────────
      //
      // AREA proportional to share count, not radius. At radius ∝ size a
      // 1-share print beside a 200-share one is invisible; at area ∝ size the
      // 200 is 14x the radius of the 1 and both are on the plot, which is the
      // requirement — the odd lots are the part of the tape being traded in.
      // TRANSPARENT WITH AN OUTLINE, so overlap reads as overlap. At a solid
      // fill one 400-share print and four 100-share prints at the same price
      // are the same disc; with alpha the stack darkens and with a rim the
      // individual prints stay countable.
      //
      // NEUTRAL GREY unless bichrome is on. See the note on `bichrome`.
      let qi = 0;                       // a merge walk, not a search per print
      ctx.lineWidth = 0.9;
      for (const t of this.trades) {
        if (t.t < tStart || t.t > tEnd) continue;
        let fill = LV_TRADE_FILL, rim = LV_TRADE_RIM;
        if (this.bichrome) {
          // The mid AS OF THIS PRINT, from the last quote at or before it.
          // Both arrays are in arrival order, so the pointer only moves
          // forward across the whole frame.
          while (qi + 1 < this.quotes.length
                 && this.quotes[qi + 1].t <= t.t) qi += 1;
          const q = this.quotes[qi];
          const mid = (q && q.bp && q.ap && q.t <= t.t)
            ? (q.bp + q.ap) / 2 : null;
          const side = this.aggressor(t.p, mid);
          if (side === 'buy') { fill = LV_BUY_FILL; rim = LV_BUY_RIM; }
          else if (side === 'sell') { fill = LV_SELL_FILL; rim = LV_SELL_RIM; }
        }
        const r = Math.max(1.3, Math.sqrt(Math.max(1, t.s)) * 0.62);
        ctx.fillStyle = fill;
        ctx.strokeStyle = rim;
        ctx.beginPath();
        ctx.arc(X(t.t), Y(t.p), r, 0, Math.PI * 2);
        ctx.fill();
        if (r > 2) ctx.stroke();
      }

      this.rates(tEnd);

      if (this.hover && this.hover.recs.length) {
        ctx.strokeStyle = '#fff';
        ctx.lineWidth = 1.2;
        for (const t of this.hover.recs) {
          ctx.beginPath();
          ctx.arc(X(t.t), Y(t.p),
                  Math.max(4, Math.sqrt(Math.max(1, t.s)) * 0.62 + 3),
                  0, Math.PI * 2);
          ctx.stroke();
        }
      }

      // ── the placed price lines ────────────────────────────────────────
      //
      // Drawn LAST so they sit over the tape: the point is to see whether
      // anything traded at that level, and a line under the prints is a line
      // being read through them.
      //
      // BRIGHT GREY, not yellow. Yellow reads as a signal and these carry
      // none — they mark a level the eye is holding, nothing more.
      for (const ln of this.lines) {
        if (ln.p < lo || ln.p > hi) continue;
        const y = Y(ln.p);
        ctx.setLineDash([5, 4]);
        ctx.strokeStyle = ln === this.lines[this.dragIdx]
          ? LV_LINE_DRAG : LV_LINE;
        ctx.lineWidth = 1.2;
        ctx.beginPath();
        ctx.moveTo(padL, y); ctx.lineTo(padL + plotW, y);
        ctx.stroke();
        ctx.setLineDash([]);
        // The count is the reading: "nothing has traded here in three
        // minutes" is the answer that decides whether to post there.
        const hits = this.lineHits(ln.p, tStart, tEnd);
        ctx.font = '600 10px sans-serif';
        ctx.fillStyle = hits ? LV_LINE_TEXT : LV_LINE_COLD;
        ctx.textAlign = 'left';
        ctx.fillText(`${ln.p.toFixed(2)}  ${hits} print${hits === 1 ? '' : 's'}`,
                     padL + 4, y - 4);
      }

      // ── working orders, ON THE PLOT ────────────────────────────────────
      //
      // Most of the value of trading from here is seeing the order against
      // the tape: whether anything is printing at the level it rests on. So
      // it is a line across the plot, not only a marker in the gutter.
      //
      // Solid, where a placed price line is dashed — one is a level being
      // watched, the other is real and working at the broker.
      for (const o of this.working) {
        if (o.price == null) continue;
        // WHERE IT IS DRAWN is the pending price while a move is in flight.
        const px = this.shownPrice(o);
        if (px < lo || px > hi) continue;
        const pend = this.pendingFor(o);
        const y = Y(px);
        const buy = String(o.side || '').toUpperCase().startsWith('BUY');
        ctx.strokeStyle = buy ? 'rgba(52,152,219,0.90)'
                              : 'rgba(232,67,147,0.90)';
        ctx.lineWidth = 1.4;
        // A REQUEST IS DASHED; the record is solid. The distinction is the
        // whole licence for moving the marker early — it is not claiming
        // the broker agreed, it is showing what was asked for.
        if (pend) ctx.setLineDash([7, 4]);
        ctx.beginPath();
        ctx.moveTo(padL, y); ctx.lineTo(padL + plotW, y);
        ctx.stroke();
        ctx.setLineDash([]);
        // Where it is moving FROM, faintly, so the move is legible while it
        // is happening rather than only in hindsight.
        if (pend && pend.from >= lo && pend.from <= hi) {
          const yf = Y(pend.from);
          ctx.strokeStyle = 'rgba(255,255,255,0.18)';
          ctx.lineWidth = 1;
          ctx.setLineDash([2, 4]);
          ctx.beginPath();
          ctx.moveTo(padL, yf); ctx.lineTo(padL + plotW, yf);
          ctx.stroke();
          ctx.setLineDash([]);
        }
        ctx.font = '700 10px sans-serif';
        ctx.fillStyle = buy ? '#6cb6e6' : '#f07ab4';
        ctx.textAlign = 'left';
        const left = (o.qty || 0) - (o.filled || 0);
        const tag = pend
          ? (pend.state === 'unknown' ? ' · UNKNOWN' : ' · sending…')
          : (o.status && o.status !== 'WORKING' ? ` · ${o.status}` : '');
        ctx.fillText(`${o.side} ${left}${o.filled ? ' of ' + o.qty : ''}`
                     + ` @ ${px.toFixed(2)}` + tag,
                     padL + 4, y + 11);
      }

      // ── the average price of the open position ─────────────────────────
      //
      // The one line that says whether the trade is working. Drawn as a
      // long-dashed rule so it is not confused with an order.
      const pos = this.position;
      if (pos && pos.avg && Math.abs(pos.qty || 0) > 1e-9
          && pos.avg >= lo && pos.avg <= hi) {
        const y = Y(pos.avg);
        ctx.setLineDash([2, 3]);
        ctx.strokeStyle = 'rgba(255,255,255,0.45)';
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(padL, y); ctx.lineTo(padL + plotW, y);
        ctx.stroke();
        ctx.setLineDash([]);
        ctx.font = '600 9px sans-serif';
        ctx.fillStyle = '#b9bfc7';
        ctx.textAlign = 'left';
        ctx.fillText(`avg ${Number(pos.avg).toFixed(2)}`, padL + 4, y - 3);
      }

      // A REFUSED MOVE, briefly. The marker has already snapped back; this
      // is the only thing that says it went somewhere and came back.
      if (this.revert && Date.now() < this.revert.until
          && this.revert.price >= lo && this.revert.price <= hi) {
        const yr = Y(this.revert.price);
        ctx.strokeStyle = 'rgba(232,67,147,0.85)';
        ctx.lineWidth = 1.2;
        ctx.setLineDash([3, 3]);
        ctx.beginPath();
        ctx.moveTo(padL, yr); ctx.lineTo(padL + plotW, yr);
        ctx.stroke();
        ctx.setLineDash([]);
        ctx.font = '700 10px sans-serif';
        ctx.fillStyle = '#f07ab4';
        ctx.textAlign = 'left';
        ctx.fillText('refused — put back', padL + 4, yr - 4);
      }

      this.drawDragGhost(ctx, padL, padT, plotW, plotH, lo, hi, Y);

      if (this.ladder) {
        this.drawLadder(ctx, padL, padT, plotW, plotH, lo, hi, Y);
      }
    },

    /* THE LADDER, on the same canvas and the same Y().
     *
     * Rows aligned to the plot's y-axis is the requirement, and drawing it
     * here is what makes that structural rather than replicated. A DOM
     * ladder beside the canvas would need its own copy of the mapping, and
     * the failure mode of two copies drifting is a click landing on the
     * wrong row — which is an order at the wrong price.
     */
    /* THE DRAG GHOST: where the order would land, and what that means.
     *
     * Drawn as its own line rather than by moving the real marker, because
     * the real marker is the broker's answer and this is a question. Both
     * are on screen at once during a drag — where it IS, and where it would
     * GO — which is also what makes the distance being covered legible.
     *
     * The marketable warning is the same test the ladder hatches with and
     * the same one the drop will ask about, so the answer cannot differ
     * between what the drag showed and what the banner then says.
     */
    drawDragGhost(ctx, padL, padT, plotW, plotH, lo, hi, Y) {
      const d = this.dragOrder;
      if (!d) return;
      const px = Number(d.price);
      if (!(px >= lo && px <= hi)) return;
      const y = Y(px);
      const mkt = this.isMarketable(d.order.side, px);
      const buy = String(d.order.side || '').toUpperCase().startsWith('BUY');
      const warn = mkt !== false;
      const col = warn ? LV_MKT_HATCH
                       : (buy ? 'rgba(120,200,255,0.95)'
                              : 'rgba(240,130,180,0.95)');

      // The distance travelled, as a band between old price and new. The
      // number is the point of the gesture and a line alone does not give it.
      const yFrom = Y(Number(d.order.price));
      if (Math.abs(yFrom - y) > 1) {
        ctx.fillStyle = warn ? 'rgba(255,140,0,0.10)'
                             : 'rgba(255,255,255,0.055)';
        ctx.fillRect(padL, Math.min(yFrom, y), plotW, Math.abs(yFrom - y));
      }

      ctx.setLineDash([6, 4]);
      ctx.strokeStyle = col;
      ctx.lineWidth = 1.6;
      ctx.beginPath();
      ctx.moveTo(padL, y); ctx.lineTo(padL + plotW, y);
      ctx.stroke();
      ctx.setLineDash([]);

      const label = this.dragText();
      ctx.font = '700 11px sans-serif';
      ctx.textAlign = 'left';
      const w = ctx.measureText(label).width;
      const lx = padL + 6;
      const ly = y - 6;
      ctx.fillStyle = 'rgba(12,14,18,0.85)';
      ctx.fillRect(lx - 3, ly - 11, w + 6, 14);
      ctx.fillStyle = col;
      ctx.fillText(label, lx, ly);
    },

    drawLadder(ctx, padL, padT, plotW, plotH, lo, hi, Y) {
      const gu = this.gutter();
      const x0 = padL + plotW;
      const rows = this.ladderRows(lo, hi);
      const rowH = rows.length > 1
        ? Math.abs(Y(rows[1]) - Y(rows[0])) : plotH;
      const step = this.ladderCents / 100;
      const fine = rowH < LV_ROW_MIN_PX;
      const last = this.lastPrice();

      /* NOTHING CLICKABLE IS DRAWN WHEN NOTHING IS CLICKABLE.
       *
       * This used to stripe 240 zones at four pixels and refuse every click
       * on them — a solid block that looked exactly like a working ladder.
       * Rendering an affordance that does not work is worse than rendering
       * nothing, so at this point the gutter carries the price scale and a
       * sentence saying what to do about it.
       *
       * With autoLadder() running, reaching here at all means even the
       * coarsest increment is too fine for the current zoom. */
      if (fine) {
        ctx.textAlign = 'center';
        for (const p of rows) {
          const yc = Y(p);
          if (yc < padT || yc > padT + plotH) continue;
          // Every fifth cent only, so the scale stays a scale.
          if (Math.abs(Math.round(p * 100) % 5) > 1e-9) continue;
          ctx.font = '9px sans-serif';
          ctx.fillStyle = '#8a8a8a';
          ctx.fillText(p.toFixed(2), x0 + gu.buy + gu.price / 2, yc + 3);
        }
        ctx.font = '700 9px sans-serif';
        ctx.fillStyle = '#e84393';
        ctx.textAlign = 'center';
        ctx.fillText('zoom in', x0 + gu.w / 2, padT + 10);
        ctx.fillText('to trade', x0 + gu.w / 2, padT + 20);
        return;
      }

      ctx.textAlign = 'center';
      for (const p of rows) {
        const yc = Y(p);
        const top = yc - rowH / 2;
        if (top + rowH < padT || top > padT + plotH) continue;

        // Zones. Faint by default; the hovered one lights up, and only when
        // the pane is armed — an unarmed pane must not look clickable.
        for (const [zone, zx, zw] of [['buy', x0, gu.buy],
                                      ['sell', x0 + gu.buy + gu.price, gu.sell]]) {
          const hot = this.ladderHover && this.ladderHover.zone === zone
            && Math.abs(this.ladderHover.price - p) < step / 2;
          // MARKETABLE ROWS ARE MARKED BEFORE THEY ARE CLICKED. A buy at or
          // above the ask and a sell at or below the bid fill instantly.
          // That is a correct limit order and never the intended one here,
          // so it is drawn as a warning rather than explained afterwards.
          const mkt = this.isMarketable(zone, p) === true;
          ctx.fillStyle = hot
            ? (this.armed
                ? (mkt ? LV_MKT_HOT
                       : (zone === 'buy' ? 'rgba(52,152,219,0.45)'
                                         : 'rgba(232,67,147,0.45)'))
                : 'rgba(140,140,140,0.30)')
            : (mkt ? LV_MKT_FILL
                   : (zone === 'buy' ? 'rgba(52,152,219,0.055)'
                                     : 'rgba(232,67,147,0.055)'));
          ctx.fillRect(zx, top, zw, Math.max(1, rowH - 0.5));
          // Hatching, so the warning survives a colourblind eye and a dim
          // screen: colour alone is not a signal a fill can depend on.
          if (mkt && rowH >= 4) {
            ctx.save();
            ctx.beginPath();
            ctx.rect(zx, top, zw, Math.max(1, rowH - 0.5));
            ctx.clip();
            ctx.strokeStyle = LV_MKT_HATCH;
            ctx.lineWidth = 1;
            for (let hx = zx - rowH; hx < zx + zw + rowH; hx += 5) {
              ctx.beginPath();
              ctx.moveTo(hx, top + rowH);
              ctx.lineTo(hx + rowH, top);
              ctx.stroke();
            }
            ctx.restore();
          }
          if (hot && this.armed && rowH >= 8) {
            ctx.font = '700 9px sans-serif';
            ctx.fillStyle = '#fff';
            ctx.fillText(String(this.qty), zx + zw / 2, yc + 3);
          }
        }

        // THE ONLY PRICE SCALE ON THE PANE now that drawGrid yields the
        // gutter, so it carries the weight those labels used to: a size up
        // where the row allows, and the last-trade row banded rather than
        // just tinted, because it is the anchor every other row is read
        // against.
        const isLast = last != null && Math.abs(last - p) < step / 2;
        if (isLast) {
          ctx.fillStyle = 'rgba(255,255,255,0.10)';
          ctx.fillRect(x0 + gu.buy, top, gu.price, Math.max(1, rowH - 0.5));
        }
        ctx.font = (rowH >= 12 ? '600 11px' : rowH >= 9 ? '600 10px' : '9px')
                 + ' sans-serif';
        ctx.fillStyle = isLast ? '#ffffff' : '#cfd4da';
        ctx.fillText(p.toFixed(2), x0 + gu.buy + gu.price / 2, yc + 3);
      }

      // Working orders, in their row. Drawn over the zones so an order is
      // never hidden by a hover.
      for (const o of this.working) {
        if (o.price == null) continue;
        const opx = this.shownPrice(o);
        if (opx < lo || opx > hi) continue;
        const pend = this.pendingFor(o);
        const buy = String(o.side || '').toUpperCase().startsWith('BUY');
        const yc = Y(Math.round(opx / step) * step);
        const zx = buy ? x0 : x0 + gu.buy + gu.price;
        const zw = buy ? gu.buy : gu.sell;
        const oy = yc - Math.max(2, rowH / 2);
        const oh = Math.max(4, rowH - 0.5);
        const col = buy ? 'rgba(52,152,219,0.95)' : 'rgba(232,67,147,0.95)';
        if (pend) {
          // HOLLOW while it is a request, filled once it is the record. The
          // same distinction the dashed line makes on the plot.
          ctx.strokeStyle = col;
          ctx.lineWidth = 1.4;
          ctx.strokeRect(zx + 0.7, oy + 0.7, zw - 1.4, oh - 1.4);
        } else {
          ctx.fillStyle = col;
          ctx.fillRect(zx, oy, zw, oh);
        }
        ctx.font = '700 9px sans-serif';
        ctx.fillStyle = pend ? col : '#fff';
        ctx.textAlign = 'center';
        if (rowH >= 8) {
          ctx.fillText(String((o.qty || 0) - (o.filled || 0)),
                       zx + zw / 2, yc + 3);
        }
      }
    },

    drawGrid(ctx, padL, padT, plotW, plotH, lo, hi, tStart, tEnd, X, Y, w) {
      ctx.lineWidth = 1;

      // ── PRICE LABELS, the most important text on the page ─────────────
      //
      // These are the numbers an order gets typed from, and they were the
      // faintest thing on the plot. Full contrast, larger than the time
      // ticks, and no competing cents column beside them.
      //
      // The cents-from-anchor scale is GONE. It was anchored to whatever the
      // axis happened to be centred on, which moved every frame, so 0¢ landed
      // on 319.44 while price was near 319.50 — a distance reading against a
      // moving zero is worse than no distance reading. If it returns it must
      // hang off something stable: the session open, or a pinned price.
      //
      // ONE PRICE SCALE AT A TIME. These labels are drawn at padL + plotW +
      // 6, which is SIX PIXELS INTO THE GUTTER — and when the ladder is open
      // the gutter's first 26px are the buy column. So the buy column was
      // painted over by a second, differently-spaced set of prices: two
      // scales disagreeing about where a price sits, with the clickable one
      // underneath. The ladder's own price column is a better scale anyway
      // (every row, on the increment being traded), so when it is up, it is
      // the only one. The GRIDLINES stay: they are inside the plot, they
      // cost the gutter nothing, and position is read against them.
      const g = this.gridStep();
      const first = Math.ceil(lo / g) * g;
      ctx.textAlign = 'left';
      for (let p = first; p <= hi + 1e-9; p += g) {
        const y = Y(p);
        ctx.beginPath();
        ctx.moveTo(padL, y);
        ctx.lineTo(padL + plotW, y);
        ctx.strokeStyle = 'rgba(255,255,255,0.06)';
        ctx.stroke();
        if (this.ladder) continue;
        ctx.font = '600 12px sans-serif';
        ctx.fillStyle = '#f2f2f2';
        ctx.fillText(p.toFixed(2), padL + plotW + 6, y + 4);
      }

      // The last trade, marked on the axis. NEUTRAL — it was accent blue,
      // the same blue as the bid line, and a blue mark at the last price
      // reads as a side for the same reason blue bubbles did.
      const last = this.lastPrice();
      if (last != null && last >= lo && last <= hi) {
        const y = Y(last);
        ctx.beginPath();
        ctx.moveTo(padL, y); ctx.lineTo(padL + plotW, y);
        ctx.strokeStyle = 'rgba(255,255,255,0.26)';
        ctx.stroke();
        // The LINE always; the numeral only when nothing else is showing
        // prices. drawLadder marks the last-trade row in the price column.
        if (!this.ladder) {
          ctx.font = '700 12px sans-serif';
          ctx.fillStyle = '#ffffff';
          ctx.fillText(last.toFixed(2), padL + plotW + 6, y + 4);
        }
      }

      // ── time ──────────────────────────────────────────────────────────
      ctx.strokeStyle = 'rgba(255,255,255,0.06)';
      ctx.fillStyle = '#7a7a7a';
      ctx.font = '9px sans-serif';
      ctx.textAlign = 'center';
      const secs = (tEnd - tStart) / 1000;
      // FINER. Thirty-second ticks on a three-minute window gave six lines
      // to read position against; ten gives eighteen, which is what the eye
      // is actually measuring against when a burst lasts fifteen seconds.
      const stepS = secs <= 30 ? 5 : secs <= 120 ? 10 : secs <= 300 ? 15
                  : secs <= 600 ? 30 : 60;
      const firstT = Math.ceil(tStart / (stepS * 1000)) * stepS * 1000;
      // Gridlines at every step; LABELS only where they fit. A label per line
      // at ten-second spacing overlaps into unreadability, and the lines are
      // what position is read against anyway.
      const labelEvery = Math.max(1, Math.ceil((60 / stepS) *
                                   (secs > 300 ? 1 : 0.5)));
      let k = 0;
      for (let t = firstT; t <= tEnd; t += stepS * 1000, k++) {
        const x = X(t);
        ctx.beginPath();
        ctx.moveTo(x, padT); ctx.lineTo(x, padT + plotH);
        ctx.strokeStyle = (k % labelEvery === 0)
          ? 'rgba(255,255,255,0.09)' : 'rgba(255,255,255,0.04)';
        ctx.stroke();
        if (k % labelEvery === 0) {
          ctx.fillStyle = '#7a7a7a';
          ctx.fillText(lvFmtClock(t), x, padT + plotH + 13);
        }
      }

      // ── the size legend ───────────────────────────────────────────────
      //
      // Bubble radius is sqrt(shares) and nothing else — no dependence on the
      // price range, the window, or the frame. Drawn so that is checkable by
      // eye rather than asserted: these four never change size.
      const legend = [1, 10, 100, 500];
      let lx = padL + 6;
      ctx.textAlign = 'left';
      ctx.font = '9px sans-serif';
      for (const sz of legend) {
        const r = Math.max(1.1, Math.sqrt(sz) * 0.62);
        ctx.beginPath();
        ctx.arc(lx + r, padT + plotH - 10, r, 0, Math.PI * 2);
        ctx.fillStyle = 'rgba(206,212,220,0.42)';
        ctx.fill();
        ctx.fillStyle = '#6a6a6a';
        ctx.fillText(String(sz), lx + r - 4, padT + plotH + 4);
        lx += r * 2 + 20;
      }
    },

    /* The NBBO across the window as a STEP FUNCTION that reaches both edges.
     *
     * THE REPORTED DEFECT. The line was drawn between quote points, so the
     * last segment could not exist until the next quote arrived — dots landed
     * in a region the lines had not reached yet, seconds of empty space at
     * the right edge, and the whole page read as jumpy. And on a quiet name
     * where the newest quote predated the window, nothing drew at all.
     *
     * A QUOTE PERSISTS UNTIL IT CHANGES. So: carry the last quote from before
     * the window in as the opening level, take every quote inside it, and
     * extend the last one flat to the right edge. The line is then always
     * complete, and the only thing a new quote does is move it.
     *
     * The synthetic end point is not a quote and is not stored — it is the
     * same quote, still standing. */
    /* WHO CROSSED THE SPREAD, as a value rather than a colour.
     *
     * Above the mid, a buyer lifted the offer; below it, a seller hit the
     * bid. That is the claim, and having it as a named function is what makes
     * it checkable — the mapping was inline in the draw loop and inverted,
     * matching the line each dot sat nearest instead of the aggressor.
     *
     * Returns null when there is no mid, or the print is exactly on it. A
     * midpoint print has no aggressor to name and grey says so. */
    aggressor(price, mid) {
      if (mid == null || price == null) return null;
      if (Math.abs(price - mid) < 1e-9) return null;
      return price > mid ? 'buy' : 'sell';
    },

    nbboSteps(tStart, tEnd) {
      const out = [];
      let carry = null;
      for (const q of this.quotes) {
        if (q.bp == null || q.ap == null || !q.bp || !q.ap) continue;
        if (q.t < tStart) { carry = q; continue; }
        if (q.t > tEnd) break;
        if (carry && !out.length) out.push({ t: tStart, bp: carry.bp, ap: carry.ap });
        out.push(q);
      }
      // Nothing inside the window, but a quote standing from before it: the
      // level is known for the whole window and must be drawn across it.
      if (!out.length && carry) {
        out.push({ t: tStart, bp: carry.bp, ap: carry.ap });
      }
      if (out.length) {
        const last = out[out.length - 1];
        if (last.t < tEnd) out.push({ t: tEnd, bp: last.bp, ap: last.ap });
      }
      return out;
    },

    /* How many prints sat at this price inside the window.
     *
     * A half-cent either side, because a line placed at 318.52 is a question
     * about that price and not about 318.5237. This is the number the line
     * exists to produce: "nothing has traded here in three minutes" is what
     * decides whether posting there is worth it. */
    lineHits(price, tStart, tEnd) {
      let n = 0;
      for (const t of this.trades) {
        if (t.t < tStart || t.t > tEnd) continue;
        if (Math.abs(t.p - price) <= 0.005) n += 1;
      }
      return n;
    },

    // ── pointer ──────────────────────────────────────────────────────────
    onMove(e) {
      const g = this.geom();
      if (!g) return;
      const mx = e.clientX - g.rect.left, my = e.clientY - g.rect.top;

      if (this.dragOrder) {
        // Snapped to the ladder increment as it moves, so the ghost sits on
        // a row that can actually be sent. A drag that reads 318.5237 and
        // then lands on 318.52 is a drag that lied on the way down.
        this.dragOrder.price = this.rowAt(my, g);
        this.hover = null;
        this.ladderHover = null;
        return;
      }

      if (this.dragIdx >= 0) {
        this.lines[this.dragIdx].p = g.priceAt(my);
        this.hover = null;
        return;
      }

      // In the ladder gutter: no tape hover there, and the row under the
      // cursor lights up instead.
      const zone = this.ladderZone(mx, g);
      if (zone) {
        this.ladderHover = { zone, price: this.rowAt(my, g) };
        this.hover = null;
        return;
      }
      this.ladderHover = null;

      // EVERY print under the cursor, not the nearest. Several share a
      // millisecond and land within a pixel or two; one number there cannot
      // distinguish a single 400-share print from four of 100.
      const hits = [];
      for (const t of this.trades) {
        if (t.t < g.tStart || t.t > g.tEnd) continue;
        const dx = g.X(t.t) - mx, dy = g.Y(t.p) - my;
        const r = Math.max(1.3, Math.sqrt(Math.max(1, t.s)) * 0.62);
        const reach = Math.max(7, r + 3);
        if (dx * dx + dy * dy <= reach * reach) hits.push(t);
      }
      hits.sort((x, y) => y.s - x.s);
      this.hover = hits.length ? { x: mx, y: my, recs: hits.slice(0, 10),
                                   more: Math.max(0, hits.length - 10) } : null;
    },

    onDown(e) {
      const g = this.geom();
      if (!g) return;
      const mx = e.clientX - g.rect.left;
      const my = e.clientY - g.rect.top;

      // AN ORDER UNDER THE CURSOR IS A GRAB, NOT A CLICK. This has to be
      // tested before the ladder zone below, because a working order is
      // drawn inside that zone — without it, pressing on your own order in
      // the gutter placed a SECOND order at the same price.
      const grab = this.orderAt(mx, my, g);
      if (grab) {
        e.preventDefault();
        this.dragOrder = { order: grab, price: Number(grab.price) };
        this.hover = null;
        return;
      }

      // A click in a ladder zone is an order. Single click, as a ladder
      // works — the safety is the arm toggle and the guards, not a
      // confirmation dialog that would defeat the point of the speed.
      const zone = this.ladderZone(mx, g);
      if (zone) {
        e.preventDefault();
        this.clickLadder(zone, this.rowAt(my, g));
        return;
      }

      // Grab a line if the cursor is on one.
      let best = -1, bestD = 6;
      this.lines.forEach((ln, i) => {
        const d = Math.abs(g.Y(ln.p) - my);
        if (d < bestD) { bestD = d; best = i; }
      });
      if (best >= 0) {
        this.dragIdx = best;
        e.preventDefault();
      }
    },

    onUp() {
      if (this.dragOrder) return this.dropOrder();

      // Snapped to the cent on release. A line at 318.5237 answers a question
      // nobody asked, and the label would read a price that cannot be posted.
      if (this.dragIdx >= 0) {
        const ln = this.lines[this.dragIdx];
        ln.p = Math.round(ln.p * 100) / 100;
        this.dragIdx = -1;
      }
    },

    /* Let go: one replace, at the row the ghost is on.
     *
     * ONE CALL, like a nudge — a drag across forty cents costs exactly what
     * a one-cent nudge costs, which is the whole reason to drag rather than
     * press the row button forty times.
     *
     * The marketable question is the same question the click path asks, so
     * it goes through the same askFirst(): dropping on a row through the
     * touch raises the same banner, worded for a move.
     */
    dropOrder() {
      const d = this.dragOrder;
      this.dragOrder = null;
      if (!d) return;
      const price = Math.round(Number(d.price) * 100) / 100;
      // Picked up and put back. Not an order, not an error, no call.
      if (Math.abs(price - Number(d.order.price)) < 0.005) return;
      const why = this.blocked();
      if (why) { this.brokerErr = why; return; }
      // THE ORDER MAY HAVE MOVED UNDER THE DRAG. A poll lands every two
      // seconds and a fill or a cancel elsewhere can retire it mid-gesture;
      // replacing by a stale id would resurrect an order that is gone.
      const live = this.working.find(
        o => String(o.order_id) === String(d.order.order_id));
      if (!live) {
        this.brokerErr = 'that order is no longer working — it filled or was '
          + 'cancelled while you were dragging it. Nothing was sent.';
        return;
      }
      if (this.askFirst(live.side, price, live)) {
        this.confirm.kind = 'move';
        this.confirm.qty = (live.qty || 0) - (live.filled || 0);
        return;
      }
      return this.sendMove(live, price);
    },

    /* What the drag is about to do, for the label that follows the cursor.
     * Marketable is asked here so the warning is on screen BEFORE the drop,
     * which is the same promise the ladder's hatching makes to a click. */
    dragText() {
      const d = this.dragOrder;
      if (!d) return '';
      const px = Number(d.price);
      const mkt = this.isMarketable(d.order.side, px);
      const tag = mkt === true ? '  ⚠ FILLS NOW'
                : mkt === null ? '  ⚠ no NBBO' : '';
      return `${d.order.side} ${(d.order.qty || 0) - (d.order.filled || 0)}`
           + ` → ${px.toFixed(2)}${tag}`;
    },

    /* Double-click places a line at that price. Placing is deliberate, so it
     * takes a deliberate gesture — a single click would drop lines while
     * reading the plot. */
    onDouble(e) {
      const g = this.geom();
      if (!g) return;
      const my = e.clientY - g.rect.top;
      if (this.lines.length >= 8) return;
      const price = Math.round(g.priceAt(my) * 100) / 100;
      this.lines.push({ p: price });
    },

    removeLine(i) { this.lines.splice(i, 1); },
    clearLines() { this.lines = []; },

    addLineAtPrice() {
      const p = this.refPrice();
      if (p == null || this.lines.length >= 8) return;
      this.lines.push({ p: Math.round(p * 100) / 100 });
    },

    /* SHARE COUNT AND NOTHING ELSE, at the cursor.
     *
     * The time, venue and condition codes were in a corner box that had to be
     * looked away to read. The one number wanted is how big the print was;
     * where it is in time and price is already where the cursor is. */
    hoverSizes() {
      if (!this.hover) return [];
      return this.hover.recs.map(r => r.s);
    },

    // ── controls ─────────────────────────────────────────────────────────
    /* LADDERS, not multipliers.
     *
     * Doubling and halving overshot every small adjustment: from 180s the
     * only neighbours were 90 and 360. These are round stops, finer at the
     * small end where the adjustments actually are, and they land on numbers
     * that read cleanly on an axis rather than on 234s.
     */
    windowStops: [15, 30, 45, 60, 90, 120, 180, 240, 300, 420, 600, 900],
    spanStops: [2, 3, 5, 8, 10, 15, 20, 30, 40, 60, 80, 120, 200, 300, 500],

    step(stops, cur, dir) {
      let i = stops.indexOf(cur);
      if (i < 0) {
        // Not on a stop — land on the nearest, then move.
        i = stops.reduce((best, v, k) =>
          Math.abs(v - cur) < Math.abs(stops[best] - cur) ? k : best, 0);
        if ((dir > 0 && stops[i] > cur) || (dir < 0 && stops[i] < cur)) {
          return stops[i];
        }
      }
      return stops[Math.max(0, Math.min(stops.length - 1, i + dir))];
    },

    zoomX(dir) {
      this.windowS = this.step(this.windowStops, this.windowS, dir);
    },
    zoomY(dir) {
      this.spanCents = this.step(this.spanStops, this.spanCents, dir);
      // A zoom hands the increment back to the automatic choice.
      this.ladderManual = false;
      this.$nextTick ? this.$nextTick(() => this.autoLadder()) : this.autoLadder();
      // A zoom re-derives the band around its own centre rather than around
      // the price, so widening does not double as a recentre.
      this.reband(false);
    },
    recenter() { this.reband(true); this.bandSteps += 1; },
    togglePause() {
      this.paused = !this.paused;
      if (!this.paused) this.frozenEnd = null;
    },

    windowLabel() {
      return this.windowS >= 60
        ? (this.windowS / 60).toFixed(this.windowS % 60 ? 1 : 0) + 'm'
        : this.windowS + 's';
    },

    // ── this name's own normal ───────────────────────────────────────────
    /* Refetched on a slow timer, because the answer cannot change until the
     * 15-minute bucket does. The server caches per symbol per bucket; this
     * only has to not hammer it. */
    async fetchNorm(force) {
      if (!this.symbol || this.normFetching) return;
      if (!force && Date.now() - this.normAt < 60000) return;
      this.normFetching = true;
      try {
        const r = await fetch('arrival-norm?symbol='
                              + encodeURIComponent(this.symbol));
        const j = await r.json();
        // Guard against a reply for a symbol this pane has since left.
        if (!j.symbol || j.symbol === this.symbol) this.norm = j;
      } catch (err) {
        this.norm = { ok: false, why: 'the comparison could not be fetched: '
                                      + err };
      } finally {
        this.normFetching = false;
        this.normAt = Date.now();
      }
    },

    normOk() { return !!(this.norm && this.norm.ok); },

    /* The scale the band and the live marker share.
     *
     * FROM ZERO. A band drawn over its own min..max makes every quiet period
     * look like a collapse and every busy one like a spike, because the
     * baseline moves with the sample. From zero, "half of normal" is half the
     * distance across, which is the reading. */
    normMax(key) {
      const s = this.norm && this.norm[key];
      if (!s) return 1;
      const live = key === 'trades_per_min' ? this.tpm : this.spm;
      return Math.max(s.max, live || 0, 1e-9) * 1.08;
    },

    normPct(key, v) {
      if (v == null) return null;
      return Math.max(0, Math.min(100, (v / this.normMax(key)) * 100));
    },

    liveOf(key) { return key === 'trades_per_min' ? this.tpm : this.spm; },

    /* Live against the median, which is the sentence the pane is for:
     * "running at 0.6x its own normal for this time of day". */
    normRatio(key) {
      const s = this.norm && this.norm[key];
      const live = this.liveOf(key);
      if (!s || live == null || !s.med) return null;
      return live / s.med;
    },

    normRatioText(key) {
      const r = this.normRatio(key);
      return r == null ? '—' : (r >= 10 ? r.toFixed(0) : r.toFixed(2)) + '×';
    },

    fmt(v) { return lvFmtNum(v); },

    // ═══ trading ════════════════════════════════════════════════════════
    //
    // EVERY ACTION IS A NAMED FUNCTION taking no positional UI state, and
    // the buttons only call them. Hotkeys are wanted later but not yet —
    // binding one should be a line that calls placeBuy(), not an untangling
    // of a click handler.

    /* THIS PAGE IS NOT THE SOURCE OF TRUTH, and this is where that is
     * enforced. `brokerAt` is when Schwab last confirmed the state, and
     * everything below is phrased against it: past the threshold the pane
     * says the list may be wrong rather than showing it as current. */
    brokerAgeS() {
      return this.ordersAt ? (Date.now() - this.ordersAt) / 1000 : null;
    },
    positionAgeS() {
      return this.positionsAt ? (Date.now() - this.positionsAt) / 1000 : null;
    },
    /* Keyed on the ORDER age, not the position age.
     *
     * The working list is the thing that goes wrong fast and the thing a
     * decision is made against. A position a few seconds old is still
     * approximately true; a working list a few seconds old may describe an
     * order that has already filled. */
    brokerStale() {
      const age = this.brokerAgeS();
      return age == null || age > (this.staleAfterS || 4);
    },
    staleAfterS: 4,

    /* The line the pane shows when its order state cannot be trusted.
     *
     * Deliberately blunt. The orders are live at Schwab whatever this page
     * believes, and the dangerous version of this pane is the one that shows
     * a confident wrong list. */
    staleWhy() {
      if (this.brokerErr) return this.brokerErr;
      const age = this.brokerAgeS();
      if (age == null) return 'orders never confirmed with Schwab';
      return `orders last confirmed ${age.toFixed(1)}s ago`
           + ` (they live 1-6s here, so this is a real gap)`;
    },

    async brokerCall(path, body) {
      const t0 = performance.now();
      this.busy = true;
      try {
        const r = await fetch('broker/' + path, {
          method: body ? 'POST' : 'GET',
          headers: { 'Content-Type': 'application/json' },
          body: body ? JSON.stringify(body) : undefined,
        });
        const j = await r.json();
        // TOTAL round trip as the browser sees it, alongside the broker's own
        // leg. Part of why this exists is finding out whether this path beats
        // the click it replaces, and only one of those two numbers answers
        // that question.
        // A READ-BACK IS NOT AN ACTION, so it does not get to claim this
        // readout. It used to: a move set rt from the PUT and then the
        // read-back overwrote it, so the number on screen was the leg
        // nobody waits for and it read 68ms while the move took 1305.
        if (!/^(orders|positions|state)/.test(String(path))) {
          this.lastRt = { total: Math.round(performance.now() - t0),
                          broker: j.rt_ms != null ? Math.round(j.rt_ms) : null,
                          visible: null, confirm: null };
        }
        if (!j.ok) {
          this.brokerErr = j.why || 'the broker refused, without saying why';
        } else {
          this.brokerErr = '';
        }
        return j;
      } catch (err) {
        this.brokerErr = 'the request never completed: ' + err
          + ' — any order already sent is still live at Schwab';
        return { ok: false, why: String(err) };
      } finally {
        this.busy = false;
      }
    },

    /* Read the record. Replaces position and working orders WHOLESALE.
     *
     * Never merged with what was held: an order that has filled or been
     * cancelled elsewhere has to disappear, and a merge is how a phantom
     * order stays on screen. */
    /* ORDERS ONLY — one Schwab call, not two.
     *
     * /broker/state gathers read_orders AND read_positions, so every action
     * that called refreshBroker() spent two calls on the read-back. Placing,
     * repricing and cancelling a RESTING limit cannot move a position: a
     * position moves on a FILL, which the 6-second position poll already
     * carries. So the second call bought nothing and it was spent on the
     * hottest path there is — a row move cost three calls, of which one was
     * the actual replace.
     *
     * Used everywhere the position provably cannot have changed. place() and
     * flatten() still read both, because those can fill.
     */
    async refreshOrders() {
      if (!this.symbol) return;
      const j = await this.brokerCall('orders?symbols='
                                      + encodeURIComponent(this.symbol));
      if (!j.ok) return;
      this.working = (j.working || [])
        .filter(o => (o.symbol || '').toUpperCase() === this.symbol);
      this.recent = j.recent || [];
      this.limits = j.limits || null;
      this.ordersAt = Date.now();
    },

    async refreshBroker() {
      if (!this.symbol) return;
      const j = await this.brokerCall('state?symbols='
                                      + encodeURIComponent(this.symbol));
      if (!j.ok) return;                 // brokerAt deliberately not moved
      this.position = (j.positions || [])
        .find(p => (p.symbol || '').toUpperCase() === this.symbol) || null;
      this.working = (j.working || [])
        .filter(o => (o.symbol || '').toUpperCase() === this.symbol);
      this.recent = j.recent || [];
      this.limits = j.limits || null;
      this.staleAfterS = j.stale_after_s || 4;
      this.ordersAt = Date.now();
      this.positionsAt = Date.now();
    },

    positionQty() { return this.position ? Number(this.position.qty) || 0 : 0; },

    /* Unrealised, from the last print. The tape is the price here — it is on
     * screen and it is newer than anything the broker would quote back. */
    openPl() {
      const p = this.position;
      const last = this.lastPrice();
      if (!p || !p.avg || last == null || Math.abs(p.qty) < 1e-9) return null;
      return (last - p.avg) * p.qty;
    },

    // ── the actions ──────────────────────────────────────────────────────
    clickLadder(zone, price) {
      if (this.rowTooFine()) {
        this.brokerErr = 'rows are under five pixels at this zoom — an order '
          + 'would land a cent from where you aimed. Zoom in or use a coarser '
          + 'row.';
        return;
      }
      const side = zone === 'buy' ? 'BUY' : 'SELL';
      if (this.askFirst(side, price)) {
        this.confirm.kind = 'place';
        return;
      }
      return zone === 'buy' ? this.placeBuy(price) : this.placeSell(price);
    },

    /* Does this price need asking about before it is sent?
     *
     * WHY THIS EXISTS. A limit below the bid on a SELL is a marketable
     * order: correct, legal, and filled the instant it lands. For a passive
     * strategy it is never the intended order, and nothing between the click
     * and the fill said so — the first news of it was the fill.
     *
     * Sets `confirm` and returns true when the caller must stop. UNKNOWN
     * ASKS TOO: with no NBBO the answer is "cannot tell", and treating that
     * as "not marketable" would put the silent fill back exactly where the
     * quote feed is worst. The two cases are worded differently so the
     * banner never claims to know something it does not.
     */
    askFirst(side, price, order) {
      const m = this.isMarketable(side, price);
      if (m === false) return false;
      const t = this.touch();
      this.confirm = {
        kind: 'place', side: String(side).toUpperCase(), price: Number(price),
        qty: Number(this.qty), unknown: m === null,
        bid: t ? t.bid : null, ask: t ? t.ask : null, order: order || null,
      };
      return true;
    },

    /* The sentence on the banner. Built here rather than in the template so
     * it can be asserted in a test. */
    confirmText() {
      const c = this.confirm;
      if (!c) return '';
      const move = c.kind === 'move';
      const what = move
        ? `Moving this ${c.side} to ${c.price.toFixed(2)}`
        : `${c.side} ${c.qty} @ ${c.price.toFixed(2)}`;
      const ask = move ? 'Move it anyway?' : 'Send it anyway?';
      if (c.unknown) {
        return `${what} — there is no NBBO for ${this.symbol}, so whether it `
             + `would fill immediately CANNOT be determined. ${ask}`;
      }
      const edge = c.side === 'BUY' ? `the ask (${c.ask.toFixed(2)})`
                                    : `the bid (${c.bid.toFixed(2)})`;
      const verb = move ? 'puts it at or through' : 'is at or through';
      return `${what} ${verb} ${edge} — it will FILL IMMEDIATELY rather than `
           + `rest. ${ask}`;
    },

    confirmSend() {
      const c = this.confirm;
      if (!c) return;
      this.confirm = null;
      if (c.kind === 'move') return this.sendMove(c.order, c.price);
      return c.side === 'BUY' ? this.placeBuy(c.price)
                              : this.placeSell(c.price);
    },

    confirmCancel() {
      this.confirm = null;
      this.lastAction = 'not sent — marketable';
    },

    placeBuy(price) { return this.place('BUY', price); },
    placeSell(price) { return this.place('SELL', price); },

    /* Nothing may be SENT while a previous send is unresolved. */
    blocked() {
      if (this.unresolved) {
        return `${this.symbol} has an unresolved order — settle it first.`;
      }
      if (!this.armed) return 'this pane is not armed. Nothing was sent.';
      return null;
    },

    async place(side, price) {
      const why = this.blocked();
      if (why) { this.brokerErr = why; return; }
      const qty = Number(this.qty);
      const sentAt = Date.now() / 1000;
      const j = await this.brokerCall('order', {
        symbol: this.symbol, side, qty,
        price: Number(price), armed: true,
        reference: this.lastPrice(), position_qty: this.positionQty(),
      });
      if (j.indeterminate) {
        // NOT AN ERROR AND NOT A SUCCESS. Never retried.
        this.unresolved = { side, qty, price: Number(price), sentAt,
                            tries: 0, state: 'looking', why: j.why };
        this.lastAction = 'UNRESOLVED';
        return j;
      }
      if (j.ok && j.order_id) this.ownIds.push(String(j.order_id));
      this.lastAction = j.ok ? `${side} ${qty} @ ${Number(price).toFixed(2)}`
                             : `${side} refused`;
      // Read back immediately: the reply says the order was accepted, and
      // only the record says where it now rests.
      await this.refreshBroker();
      return j;
    },

    /* BY ONE ROW, not by a cent. The row height already encodes the
     * increment, so the same action does the right thing at any zoom — which
     * is the reason to nudge rather than retype. */
    nudgeUp() { return this.nudge(+1); },
    nudgeDown() { return this.nudge(-1); },

    async nudge(dir) {
      const o = this.primaryOrder();
      if (!o) {
        this.brokerErr = this.noPrimaryWhy('move');
        return;
      }
      const why = this.blocked();
      if (why) { this.brokerErr = why; return; }
      const step = this.ladderCents / 100;
      const price = Math.round((Number(o.price) + dir * step) * 100) / 100;
      // A MOVE CAN CROSS THE TOUCH TOO. Walking a resting order one row at a
      // time is exactly how it ends up through the bid, and the row that
      // does it looks like every other row. Same question, same banner.
      if (this.askFirst(o.side, price, o)) {
        this.confirm.kind = 'move';
        this.confirm.qty = (o.qty || 0) - (o.filled || 0);
        return;
      }
      return this.sendMove(o, price);
    },

    /* The replace itself, once the price is settled. Split from nudge() so
     * the confirmation path sends the identical request rather than a second
     * copy of it that could drift. */
    async sendMove(o, price) {
      const qty = (o.qty || 0) - (o.filled || 0);
      const sentAt = Date.now() / 1000;
      const t0 = performance.now();
      // CAPTURED BEFORE ANYTHING IS MUTATED. `o` is the same object as the
      // entry in `working`, so writing the new id onto it first would leave
      // every later lookup — and the ownIds filter below — searching for an
      // id that no longer exists anywhere.
      const oldId = String(o.order_id);

      // OPTIMISTIC, AND SAID SO. The marker moves now; `pending` is what
      // makes it draw as a request rather than as the record.
      this.pending = { order_id: oldId, from: Number(o.price),
                       to: Number(price), qty, side: o.side, sentAt,
                       state: 'sending' };
      this.revert = null;
      const visible = Math.round(performance.now() - t0);

      const j = await this.brokerCall('replace', {
        order_id: o.order_id, symbol: this.symbol, side: o.side,
        qty, price, armed: true,
        reference: this.lastPrice(), position_qty: this.positionQty(),
      });
      const confirm = Math.round(performance.now() - t0);
      this.lastRt = { total: confirm, visible, confirm,
                      broker: j.rt_ms != null ? Math.round(j.rt_ms) : null };

      if (j.indeterminate) {
        // NEITHER CONFIRMED NOR REVERTED. The order may or may not have
        // moved, so the marker must claim neither: `unknown` keeps it at the
        // requested price and draws it as a question until the reconcile
        // settles it. Reverting here would assert the move did not happen,
        // which is exactly as unfounded as asserting it did.
        this.pending.state = 'unknown';
        this.unresolved = { side: o.side, qty, price, sentAt, tries: 0,
                            state: 'looking', why: j.why, wasReplace: true };
        this.lastAction = 'UNRESOLVED';
        return j;
      }

      if (!j.ok) {
        // REFUSED. Put it back, and make the snap back visible — a marker
        // that quietly returns to where it was is a move you think happened.
        this.revert = { price: Number(price), until: Date.now() + 1500 };
        this.pending = null;
        this.lastAction = 'move refused';
        return j;
      }

      // ACCEPTED. A 2xx IS the broker's answer, so this is no longer a
      // guess and `working` may carry it. THE READ-BACK IS GONE: it cost
      // 855ms and a second call of the minute's budget to be told what this
      // reply already said, and the 2s poll corrects anything Schwab
      // adjusted within one tick.
      const live = this.working.find(x => String(x.order_id) === oldId);
      if (live) {
        live.price = Number(price);
        // A replace makes a NEW order and Schwab gives no link from the old
        // one to it, so the id in the reply is the only handle there is.
        if (j.order_id) live.order_id = String(j.order_id);
      }
      if (j.order_id) {
        this.ownIds = this.ownIds.filter(id => id !== oldId);
        this.ownIds.push(String(j.order_id));
      }
      this.pending = null;
      this.lastAction = `moved to ${price.toFixed(2)}`;
      return j;
    },

    /* The price to DRAW an order at: the pending one while a move is in
     * flight, the broker's otherwise. One place, so the plot line and the
     * ladder marker cannot disagree about where an order is. */
    shownPrice(o) {
      const p = this.pending;
      return (p && String(p.order_id) === String(o.order_id))
        ? p.to : Number(o.price);
    },

    pendingFor(o) {
      const p = this.pending;
      return (p && String(p.order_id) === String(o.order_id)) ? p : null;
    },

    /* The order the single-order controls act on.
     *
     * With more than one working, the controls name which rather than
     * silently picking — so this returns the only one, or nothing.
     */
    primaryOrder() {
      // THIS PANE'S OWN most recent order first. Falling straight through to
      // "the only working order" would let the controls act on an order
      // placed in thinkorswim, which also shows up in this list.
      for (let i = this.ownIds.length - 1; i >= 0; i--) {
        const mine = this.working.find(o => String(o.order_id) === this.ownIds[i]);
        if (mine) return mine;
      }
      // THE FALLBACK IS NOW SOURCE-CHECKED. `ownIds` is lost on a reload and
      // never held an order this pane did not itself send, so the fallback
      // is what runs after a refresh — and it used to hand back "the only
      // working order" whichever application had placed it. A nudge would
      // then reprice an order entered in thinkorswim.
      //
      // `from_api` is Schwab's own `TA_` stamp, which a client cannot set
      // (see broker._norm_order). It is per-account, so it cannot say WHICH
      // of ours this is — but the fallback only runs when there is one
      // candidate, and "is this ours at all" is exactly what was missing.
      // Unstamped means DECLINE: the controls then say nothing is theirs to
      // act on, which is the safe direction to be wrong in.
      const ours = this.working.filter(o => o.from_api);
      return ours.length === 1 ? ours[0] : null;
    },

    /* Why primaryOrder() gave nothing back, in words a person can act on.
     *
     * Three different situations used to collapse into "no working order":
     * none at all, several, and — since the fallback started checking the
     * source — one that belongs to thinkorswim. Saying "there is no order"
     * about an order sitting visibly in the list would read as a bug in the
     * pane rather than a refusal to touch someone else's order. */
    noPrimaryWhy(verb) {
      if (this.working.length > 1) {
        return `more than one order is working — ${verb} them from the list.`;
      }
      if (this.working.length === 1 && !this.working[0].from_api) {
        return `the only working order in ${this.symbol} was not placed from `
             + `here — Schwab did not stamp it as this app's. ${verb} it `
             + `where it was placed, or from the list.`;
      }
      return `no working order in this pane to ${verb}.`;
    },

    async cancelOrder(orderId) {
      // NOT gated on arming. Cancelling is how a mistake is undone, and a
      // safety switch that traps an order is not a safety switch.
      const id = orderId
        || (this.primaryOrder() && this.primaryOrder().order_id);
      if (!id) {
        this.brokerErr = this.noPrimaryWhy('cancel');
        return;
      }
      const j = await this.brokerCall('cancel', { order_id: id });
      this.lastAction = j.ok ? 'cancelled' : 'cancel refused';
      await this.refreshOrders();
      return j;
    },

    async cancelAll() {
      for (const o of this.working.slice()) {
        await this.brokerCall('cancel', { order_id: o.order_id });
      }
      this.lastAction = 'cancelled all';
      await this.refreshOrders();
    },

    /* Cancels first, THEN closes at market. A resting order left working can
     * open a fresh position in the opposite direction the moment the flatten
     * fills; the server does it in that order and this only asks. */
    async flatten() {
      if (!this.armed) {
        this.brokerErr = 'this pane is not armed. Nothing was sent.';
        return;
      }
      const j = await this.brokerCall('flatten',
                                      { symbol: this.symbol, armed: true });
      this.lastAction = j.ok
        ? (j.flat ? 'already flat' : 'flattened at market')
        : 'flatten refused';
      await this.refreshBroker();
      return j;
    },

    /* Settle an unresolved placement against the broker's own record.
     *
     * Driven by the order poll, so it costs one extra call every two seconds
     * and only while something is actually unresolved. It never gives up
     * quietly: after the window it stops looking and SAYS it could not
     * confirm, which is a different and more useful statement than silence.
     */
    async tryResolve() {
      const u = this.unresolved;
      if (!u || u.state === 'ambiguous' || u.state === 'gave-up') return;
      u.tries += 1;
      const j = await this.brokerCall('reconcile', {
        symbol: this.symbol, side: u.side, qty: u.qty,
        price: u.price, sent_at: u.sentAt,
      });
      if (!j.ok) return;                       // try again on the next poll
      if (j.state === 'found') {
        u.state = 'found';
        // Settled: the record is about to be re-read, so the request stops
        // standing in for it.
        this.pending = null;
        if (j.order && j.order.order_id) this.ownIds.push(String(j.order.order_id));
        this.lastAction = `resolved: the order did land (${j.order.status})`;
        this.unresolved = null;
        await this.refreshBroker();
        return;
      }
      if (j.state === 'ambiguous') {
        // NOT GUESSED. See broker.match_placement for why.
        u.state = 'ambiguous';
        u.matches = j.orders;
        return;
      }
      // Absent. Keep looking — Schwab does not list an order the instant it
      // is accepted, and one empty look proves nothing.
      if (u.tries * LV_RESOLVE_POLL_S >= LV_RESOLVE_WINDOW_S) {
        u.state = 'gave-up';
      }
    },

    unresolvedText() {
      const u = this.unresolved;
      if (!u) return '';
      const age = ((Date.now() / 1000) - u.sentAt).toFixed(0);
      const what = `${u.side} ${u.qty} @ ${Number(u.price).toFixed(2)}`;
      if (u.state === 'ambiguous') {
        return `${what} sent ${age}s ago matched ${u.matches.length} orders `
             + `at Schwab. It is not being guessed at — check which is yours `
             + `and clear this.`;
      }
      if (u.state === 'gave-up') {
        // The elapsed time, not the constant: what matters to a person
        // reading this is how long the record was actually searched.
        return `${what} could NOT be confirmed either way after ${age}s of `
             + `looking. It may be resting at Schwab. Check there before `
             + `doing anything else in this pane.`;
      }
      return `${what} sent ${age}s ago — no reply arrived, so whether it `
           + `landed is unknown. Reading the record… (${u.tries})`;
    },

    /* Cleared BY HAND, never on a timer.
     *
     * The block exists because the pane does not know its own position. Only
     * a person who has looked can say it is settled, and a timeout that
     * quietly expired the warning would be the warning failing at its job. */
    clearUnresolved() {
      this.unresolved = null;
      // The pending marker was the unresolved move. Clearing one by hand
      // and leaving the other would keep a requested price on screen with
      // nothing left to settle it.
      this.pending = null;
      this.lastAction = 'unresolved state cleared by hand';
    },

    toggleArm() {
      this.armed = !this.armed;
      // Arming shows the ladder: it is what the arming is for, and hunting
      // for a second toggle at the moment of wanting to trade is friction in
      // the wrong place.
      if (this.armed) { this.ladder = true; this.refreshOrders(); }
    },

    toggleLadder() {
      this.ladder = !this.ladder;
      if (this.ladder) {
        this.ladderManual = false;
        this.$nextTick ? this.$nextTick(() => { this.resize(); this.autoLadder(); })
                       : this.autoLadder();
        this.refreshOrders();
      } else if (this.$nextTick) {
        // The gutter's width changes, so the canvas has to be re-measured.
        this.$nextTick(() => this.resize());
      }
    },

    ladderStep(dir) {
      this.ladderCents = this.step(this.ladderStops, this.ladderCents, dir);
      // A DELIBERATE PICK, honoured until the next zoom.
      this.ladderManual = true;
    },

    /* THE NUMBER YOU FEEL, FIRST.
     *
     * This used to report the PUT and then be overwritten by the read-back,
     * so a move that took 1305ms end to end displayed 68ms — the one leg
     * nobody waits on. A move now shows both halves and in the right order:
     * how long until the marker moved, then how long until the broker
     * agreed. They are different numbers and both are worth knowing; only
     * the first is the one being judged when it feels slow.
     */
    rtText() {
      const r = this.lastRt;
      if (!r) return '—';
      const at = r.broker != null ? ` (${r.broker} at Schwab)` : '';
      if (r.visible == null) return `${r.total}ms${at}`;
      return `${r.visible}ms seen · ${r.confirm}ms confirmed${at}`;
    },
  };
};

/* A SECOND ALPINE SCOPE, declared as one.
 *
 * The template binds directly to pane members and pane methods call each
 * other through `this`, so every check that resolves an expression against
 * "the component" has to know this object exists — otherwise the whole file
 * reads as a wall of unresolved references and the checks that would catch a
 * real typo get switched off to quieten them.
 *
 * The tag is opt-in because the checker CALLS what it is pointed at, and
 * calling every function on window to see what comes back would be running
 * arbitrary page code inside a linter. */
window.lvPane.isComponentScope = true;

document.addEventListener('alpine:init', () => {
  Alpine.data('equitiesLive', () => ({

    // ── connection ───────────────────────────────────────────────────────
    sock: null,
    connected: false,
    status: null,
    refused: '',
    retryIn: 0,

    // ── the panes ────────────────────────────────────────────────────────
    panes: [],
    nextId: 1,
    // 'row' | 'col' | 'grid'. Grid is 2x2 and only means anything at three
    // or four panes.
    layout: 'row',
    // Symbols the hub holds whether or not a pane wants them. Authoritative
    // copy lives on the server; this is what it last told us.
    pinned: [],
    pinPending: '',
    // symbol -> how many panes want it. The socket's own subscription set is
    // a SET, so a second pane on the same symbol is a no-op to it and the
    // first close would have unsubscribed the other pane's tape. The count
    // lives here, and watch/unwatch only cross the wire on 0<->1.
    counts: {},

    init() {
      this.addPane('FDX');
      this.connect();
      this.$nextTick(() => {
        this.resizeAll();
        window.addEventListener('resize', () => this.resizeAll());
        // On WINDOW, not the canvas: a drag that ends off the plot would
        // otherwise leave the line stuck to the cursor. Whichever pane is
        // dragging gets it; the rest ignore it.
        window.addEventListener('mouseup', () => {
          for (const p of this.panes) p.onUp();
        });
        requestAnimationFrame(() => this.frame());
        setInterval(() => {
          for (const p of this.panes) p.fetchNorm(false);
        }, 20000);
        // ONE broker read for every pane, not one per pane.
        //
        // Schwab returns the whole account either way, so per-pane polling
        // would multiply the same two calls by the number of panes against a
        // 120-a-minute ceiling shared with the orders themselves. This is
        // one call per tick whether one pane is open or four.
        //
        // The TICK is 2s; how often it actually reads is pollOrders()'s
        // decision — 2s while a ladder is open or a pane is armed, 6s while
        // the tape is only being watched. Ticking faster than the slow
        // cadence is what lets opening a ladder take effect immediately.
        setInterval(() => this.pollOrders(), LV_ORDER_POLL_MS);
        setInterval(() => this.pollPositions(), 6000);
        this.loadBrokerHealth();
        // The runtime trading flag can be flipped by another service while
        // this page is open, so the arm button's availability is re-read
        // rather than fixed at load. Cheap: it touches no Schwab endpoint
        // and spends none of the 120-a-minute budget.
        setInterval(() => this.loadBrokerHealth(), 10000);
      });
    },

    // ── panes ────────────────────────────────────────────────────────────
    maxPanes() {
      const cap = this.status && this.status.caps
        ? this.status.caps.symbols : 8;
      return Math.min(4, cap);
    },

    /* Grid is offered only where it means something. At two panes a 2x2 is
     * two panes and two holes. */
    layouts() {
      return this.panes.length > 2
        ? [['row', 'side by side'], ['col', 'stacked'], ['grid', '2 x 2']]
        : [['row', 'side by side'], ['col', 'stacked']];
    },

    setLayout(l) {
      this.layout = l;
      // The canvases change size, and a canvas keeps its old backing store
      // until told otherwise — it would just scale the previous frame.
      this.$nextTick(() => this.resizeAll());
    },

    // ── pins ─────────────────────────────────────────────────────────────
    isPinned(sym) { return !!sym && this.pinned.includes(sym); },

    pin(sym) {
      sym = (sym || '').trim().toUpperCase();
      if (sym) this.send({ action: 'pin', symbol: sym });
    },
    unpin(sym) { this.send({ action: 'unpin', symbol: sym }); },
    togglePin(sym) { this.isPinned(sym) ? this.unpin(sym) : this.pin(sym); },
    pinTyped() { this.pin(this.pinPending); this.pinPending = ''; },

    addPane(sym) {
      if (this.panes.length >= this.maxPanes()) return;
      const p = window.lvPane(this.nextId++, msg => this.fromPane(msg));
      p.pending = sym || '';
      this.panes.push(p);
      this.$nextTick(() => {
        p.resize();
        if (sym) { p.watch(sym); p.fetchNorm(true); }
      });
    },

    removePane(i) {
      const p = this.panes[i];
      if (!p) return;
      // The LAST pane on a symbol releases it, and only that one.
      this.release(p.symbol);
      this.panes.splice(i, 1);
      this.$nextTick(() => this.resizeAll());
    },

    /* Every pane message goes through here so refcounting is in one place. */
    fromPane(msg) {
      if (msg.kind !== 'watch') return;
      if (msg.prev && msg.prev !== msg.symbol) this.release(msg.prev);
      this.counts[msg.symbol] = (this.counts[msg.symbol] || 0) + 1;
      // ALWAYS `watch`, never a second verb.
      //
      // This used to send `watch` on 0->1 and `snapshot` after, which made
      // the client's count part of the protocol — and the count could be
      // wrong. `send()` drops when the socket is not open, so a first watch
      // could vanish while the count still went to one, and the next pane
      // asked for a snapshot of a symbol the server had never held: "CRS is
      // not watched on this connection", with nothing at any cap.
      //
      // The server now treats a repeat watch as a snapshot request, so this
      // side does not have to know which case it is in. The count survives
      // only to decide when to unwatch, where being too high merely delays an
      // unsubscribe and can never cut another pane's tape.
      this.send({ action: 'watch', symbol: msg.symbol,
                  window_s: msg.window_s });
    },

    release(sym) {
      if (!sym) return;
      const n = (this.counts[sym] || 0) - 1;
      if (n > 0) { this.counts[sym] = n; return; }
      delete this.counts[sym];
      this.send({ action: 'unwatch', symbol: sym });
    },

    // ── socket ───────────────────────────────────────────────────────────
    connect() {
      const proto = location.protocol === 'https:' ? 'wss' : 'ws';
      // RELATIVE, so the page works on the port and behind the tunnel
      // hostname without a second setting to keep in step.
      const base = location.pathname.replace(/\/[^/]*$/, '/');
      const s = new WebSocket(`${proto}://${location.host}${base}ws`);
      this.sock = s;
      s.onopen = () => {
        this.connected = true;
        this.retryIn = 0;
        // The server remembers nothing across a drop. Re-assert every pane's
        // symbol, and rebuild the counts from the panes rather than trusting
        // a tally that spans a disconnect. Anything queued while the socket
        // was down is discarded in favour of that, because the panes are the
        // authoritative statement of what should be watched.
        this.outbox = [];
        this.counts = {};
        for (const p of this.panes) if (p.symbol) p.watch(p.symbol, true);
        this.send({ action: 'pinned' });
      };
      s.onmessage = ev => this.onMessage(JSON.parse(ev.data));
      s.onclose = () => {
        this.connected = false;
        // Backoff, and the page SAYS it is disconnected. A frozen plot that
        // looks live is the failure mode worth the most care here.
        this.retryIn = Math.min((this.retryIn || 1) * 2, 30);
        setTimeout(() => this.connect(), this.retryIn * 1000);
      };
      s.onerror = () => { /* onclose follows and carries the retry */ };
    },

    /* HELD, NOT DROPPED, when the socket is not open yet.
     *
     * A silent drop is what let the page believe it was subscribed to
     * something the server had never heard of. The queue is bounded and
     * cleared on open, because a backlog of stale subscription changes is
     * worse than none — `onopen` re-asserts every pane from its own state
     * anyway, which is the authoritative version. */
    outbox: [],
    send(o) {
      if (this.sock && this.sock.readyState === 1) {
        this.sock.send(JSON.stringify(o));
        return;
      }
      if (this.outbox.length < 32) this.outbox.push(o);
    },

    onMessage(m) {
      if (m.ev === 'status') {
        this.status = m.data;
        if (m.data && m.data.pinned) this.pinned = m.data.pinned;
        return;
      }
      if (m.ev === 'pinned') { this.pinned = m.data || []; return; }
      if (m.ev === 'refused') {
        // Named on the pane that asked for it where possible, so a symbol cap
        // refusal is attached to the pane that hit it.
        const hit = m.symbol
          ? this.panes.filter(p => p.symbol === m.symbol) : [];
        if (hit.length) { for (const p of hit) p.refused = m.why || 'refused'; }
        else this.refused = m.why || 'refused';
        return;
      }
      if (m.ev === 'snapshot') {
        for (const p of this.panes) {
          if (p.symbol !== m.data.symbol) continue;
          p.trades = (m.data.trades || []).slice();
          p.quotes = (m.data.quotes || []).slice();
          p.reband(true);
        }
        return;
      }
      if (m.ev === 'batch') {
        for (const p of this.panes) {
          const d = m.data[p.symbol];
          if (!d) continue;
          // PAUSE KEEPS BUFFERING. The window freezes; the data does not, so
          // resuming catches up rather than skipping the interval.
          for (const t of d.t) p.trades.push(t);
          for (const q of d.q) p.quotes.push(q);
          p.prune();
        }
      }
    },

    // ── the frame loop ───────────────────────────────────────────────────
    //
    // ONE loop for every pane. Three rAF chains would each draw at their own
    // phase and the panes would tear against each other on the same tape.
    lastDraw: 0,
    frame() {
      const now = performance.now();
      // ~30fps is plenty for a tape and halves the work of 60.
      if (now - this.lastDraw > 33) {
        this.lastDraw = now;
        for (const p of this.panes) p.draw();
      }
      requestAnimationFrame(() => this.frame());
    },

    resizeAll() { for (const p of this.panes) p.resize(); },

    // ── the broker ───────────────────────────────────────────────────────
    brokerHealth: null,
    // When the order poll last actually went out, so the tick can skip
    // beats while nothing is being traded. See pollOrders().
    lastOrderPollAt: 0,

    async loadBrokerHealth() {
      try {
        const r = await fetch('broker/health');
        this.brokerHealth = await r.json();
      } catch (err) { this.brokerHealth = { why: String(err) }; }
    },

    /* TWO POLLS AT DIFFERENT RATES, one read each for all panes.
     *
     * Orders in this account live ONE TO SIX SECONDS — entered, repriced and
     * filled inside what used to be a single four-second poll of both halves
     * together. Positions only move when one of those fills. Pairing them
     * made the cheap read wait on the expensive one and the expensive one
     * run no more often than the cheap one needed.
     *
     * THE BUDGET, which is the reason the rates are what they are. Ordinary
     * traffic may spend 90 calls a minute; the other 30 are reserved so a
     * cancel or a flatten is never the call that gets refused.
     *
     *   watching only   orders 6s + positions 6s   =  20/min
     *   trading         orders 2s + positions 6s   =  40/min
     *
     * A fixed 2s cost 40 whether or not anything was being traded, and the
     * actions themselves used to cost three calls each (the action, plus a
     * read-back of orders AND positions). At 40 idle plus 18 for one
     * place-nudge-nudge-nudge-nudge-cancel round trip, a few clicks reached
     * 90 and the pane stopped. Read-backs are orders-only now — see
     * refreshOrders() — so the same round trip is 12.
     *
     * One read serves every pane: Schwab returns the whole account either
     * way, so per-pane polling would multiply the same calls by the number
     * of panes for no extra information.
     */
    livePanes() {
      return this.panes.filter(p => p.ladder && p.symbol);
    },

    /* Is anyone actually trading right now?
     *
     * A pane with the ladder shut is a pane watching the tape, and the
     * order book is not what it is watching. An unresolved placement counts
     * whatever else is true: that is the pane that most needs a fast answer.
     */
    tradingActive() {
      return this.panes.some(p => p.armed || p.ladder || p.unresolved);
    },

    async pollOrders() {
      const live = this.livePanes();
      if (!live.length) return;
      // CADENCE FOLLOWS INTENT. Two seconds is the right rate while orders
      // are in play and pure waste while none are: at 2s this poll alone is
      // 30 of the 90 calls a minute ordinary traffic may spend, and it was
      // spending them whether or not a ladder was open. The tick stays at
      // 2s and the SLOW case skips beats, so opening a ladder speeds the
      // poll up on the next tick rather than at the end of a long interval.
      const cadence = this.tradingActive() ? LV_ORDER_POLL_MS
                                           : LV_ORDER_POLL_IDLE_MS;
      const now0 = Date.now();
      // The 100ms slack absorbs setInterval jitter: a tick that lands at
      // 1996ms must not be dropped and then waited out all over again.
      if (now0 - this.lastOrderPollAt < cadence - 100) return;
      this.lastOrderPollAt = now0;
      const syms = [...new Set(live.map(p => p.symbol))];
      try {
        const r = await fetch('broker/orders?symbols='
                              + encodeURIComponent(syms.join(',')));
        const j = await r.json();
        if (!j.ok) {
          // The AGE IS NOT ADVANCED on a failure. A pane whose last
          // confirmation is old must keep saying so, and a failed poll is
          // exactly when that matters.
          for (const p of live) p.brokerErr = j.why || 'the order read failed';
          return;
        }
        const now = Date.now();
        for (const p of live) {
          p.working = (j.working || [])
            .filter(o => (o.symbol || '').toUpperCase() === p.symbol);
          p.recent = (j.recent || [])
            .filter(o => (o.symbol || '').toUpperCase() === p.symbol);
          p.limits = j.limits || null;
          p.staleAfterS = j.stale_after_s || 4;
          p.brokerErr = '';
          p.ordersAt = now;
          // An UNRESOLVED placement asks the server to match, rather than
          // matching here: the rule and its ambiguity caveat live in
          // broker.match_placement, and a second copy in the browser would
          // be a second place for that caveat to go stale.
          if (p.unresolved) p.tryResolve();
        }
      } catch (err) {
        for (const p of live) {
          p.brokerErr = 'the order read never completed: ' + err;
        }
      }
    },

    async pollPositions() {
      const live = this.livePanes();
      if (!live.length) return;
      const syms = [...new Set(live.map(p => p.symbol))];
      try {
        const r = await fetch('broker/positions?symbols='
                              + encodeURIComponent(syms.join(',')));
        const j = await r.json();
        if (!j.ok) return;
        const now = Date.now();
        for (const p of live) {
          p.position = (j.positions || [])
            .find(x => (x.symbol || '').toUpperCase() === p.symbol) || null;
          p.positionsAt = now;
        }
      } catch (err) { /* the orders poll carries the visible error */ }
    },

    /* Which panes are blocked on an unresolved placement, named.
     *
     * With four panes open, "something is stuck" is not actionable. This is
     * the symbol list the site bar shows, so the pane can be found without
     * reading four of them. */
    unresolvedPanes() {
      return this.panes.filter(p => p.unresolved);
    },

    tradingEnabled() {
      return !!(this.brokerHealth && this.brokerHealth.trading_enabled);
    },

    statusLine() {
      const s = this.status;
      if (!this.connected) {
        return `disconnected — retrying in ${this.retryIn || 1}s`;
      }
      if (!s) return 'connecting…';
      if (s.problems && s.problems.length) return s.problems[0];
      if (!s.authed) return 'socket open, not authenticated';
      return `${s.feed} · ${s.symbols.length}/${s.caps.symbols} symbols`;
    },

    isDelayed() { return !!(this.status && this.status.delayed); },
  }));
});

// The sentinel the page checks — see the template. Last line, so a syntax
// error anywhere above stops execution before it and the banner still fires.
window.__liveLoaded = true;
