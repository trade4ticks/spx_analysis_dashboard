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
const LV_UP_FILL = 'rgba(232,67,147,0.30)';
const LV_UP_RIM  = 'rgba(240,130,180,0.85)';
const LV_DN_FILL = 'rgba(52,152,219,0.30)';
const LV_DN_RIM  = 'rgba(120,200,255,0.85)';

// Price lines: BRIGHT GREY, not yellow. Yellow reads as a signal, and these
// carry no signal — they are a place the eye is holding.
const LV_LINE      = 'rgba(214,218,224,0.90)';
const LV_LINE_DRAG = 'rgba(255,255,255,0.98)';
const LV_LINE_TEXT = '#dfe3e8';
const LV_LINE_COLD = '#8a8a8a';

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

    // Both rates over a FIXED sixty seconds. See the note in rates().
    tpm: null,
    spm: null,
    tpmPartial: false,

    // This name's own normal for the current 15-minute bucket, from the
    // pipeline's stored history. Never computed here.
    norm: null,
    normFetching: false,
    normAt: 0,

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
    geom() {
      const cv = this.cv();
      if (!cv) return null;
      const r = cv.getBoundingClientRect();
      const padL = 8, padR = 58, padT = 8, padB = 20;
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

      const padL = 8, padR = 58, padT = 8, padB = 20;
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
        const vis = this.quotes.filter(q => q.t >= tStart && q.t <= tEnd
                                            && q.bp && q.ap);
        if (vis.length > 1) {
          ctx.beginPath();
          ctx.moveTo(X(vis[0].t), Y(vis[0].ap));
          for (const q of vis) ctx.lineTo(X(q.t), Y(q.ap));
          for (let i = vis.length - 1; i >= 0; i--) {
            ctx.lineTo(X(vis[i].t), Y(vis[i].bp));
          }
          ctx.closePath();
          ctx.fillStyle = 'rgba(150,170,190,0.045)';
          ctx.fill();
        }
        ctx.lineWidth = 1.6;
        for (const [key, col] of [['bp', 'rgba(130,190,235,0.95)'],
                                  ['ap', 'rgba(235,150,190,0.95)']]) {
          ctx.beginPath();
          let started = false;
          for (const q of this.quotes) {
            const v = q[key];
            if (v == null || q.t < tStart || q.t > tEnd) { started = false; continue; }
            const x = X(q.t), y = Y(v);
            if (!started) { ctx.moveTo(x, y); started = true; }
            else ctx.lineTo(x, y);
          }
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
          if (mid != null && Math.abs(t.p - mid) > 1e-9) {
            const up = t.p > mid;
            fill = up ? LV_UP_FILL : LV_DN_FILL;
            rim = up ? LV_UP_RIM : LV_DN_RIM;
          }
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
        ctx.font = '700 12px sans-serif';
        ctx.fillStyle = '#ffffff';
        ctx.fillText(last.toFixed(2), padL + plotW + 6, y + 4);
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

      if (this.dragIdx >= 0) {
        this.lines[this.dragIdx].p = g.priceAt(my);
        this.hover = null;
        return;
      }

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
      const my = e.clientY - g.rect.top;
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
      // Snapped to the cent on release. A line at 318.5237 answers a question
      // nobody asked, and the label would read a price that cannot be posted.
      if (this.dragIdx >= 0) {
        const ln = this.lines[this.dragIdx];
        ln.p = Math.round(ln.p * 100) / 100;
        this.dragIdx = -1;
      }
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
    stacked: false,
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
      });
    },

    // ── panes ────────────────────────────────────────────────────────────
    maxPanes() {
      const cap = this.status && this.status.caps
        ? this.status.caps.symbols : 4;
      return Math.min(3, cap);
    },

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
      const n = (this.counts[msg.symbol] || 0) + 1;
      this.counts[msg.symbol] = n;
      // WATCH ONLY ON 0->1. The hub reference-counts acquisitions, and this
      // socket's subscription set is a set — so a second `watch` for a symbol
      // already held would raise the hub's count to two while the set stayed
      // at one, and the single `unwatch` this client eventually sends would
      // leave the symbol subscribed forever, occupying one of four slots.
      //
      // A later pane on the same symbol asks only for the backlog, so it
      // opens onto a populated plot rather than filling in over three
      // minutes.
      this.send(n === 1
        ? { action: 'watch', symbol: msg.symbol, window_s: msg.window_s }
        : { action: 'snapshot', symbol: msg.symbol, window_s: msg.window_s });
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
        // a tally that spans a disconnect.
        this.counts = {};
        for (const p of this.panes) if (p.symbol) p.watch(p.symbol, true);
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

    send(o) {
      if (this.sock && this.sock.readyState === 1) this.sock.send(JSON.stringify(o));
    },

    onMessage(m) {
      if (m.ev === 'status') { this.status = m.data; return; }
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
