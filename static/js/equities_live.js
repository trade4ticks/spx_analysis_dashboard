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
 */

const LV_BLUE = '#3498db';
const LV_PINK = '#e84393';

function lvFmtClock(ms) {
  const d = new Date(ms);
  return String(d.getHours()).padStart(2, '0') + ':'
       + String(d.getMinutes()).padStart(2, '0') + ':'
       + String(d.getSeconds()).padStart(2, '0');
}

document.addEventListener('alpine:init', () => {
  Alpine.data('equitiesLive', () => ({

    // ── connection ───────────────────────────────────────────────────────
    sock: null,
    connected: false,
    status: null,
    refused: '',
    retryIn: 0,

    // ── the pane ─────────────────────────────────────────────────────────
    symbol: '',
    pending: 'FDX',
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
    showQuotes: true,

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
    // Trades per minute, from the window itself rather than a counter: the
    // arrival rate is a proxy for fillability and a stale one is worse than
    // none.
    tpm: null,
    tpmPartial: false,
    lastDraw: 0,

    init() {
      this.connect();
      this.$nextTick(() => {
        this.resize();
        window.addEventListener('resize', () => this.resize());
        const cv = document.getElementById('lv-canvas');
        if (cv) {
          cv.addEventListener('mousemove', e => this.onMove(e));
          cv.addEventListener('mousedown', e => this.onDown(e));
          cv.addEventListener('dblclick', e => this.onDouble(e));
          cv.addEventListener('mouseleave', () => { this.hover = null; });
          // On WINDOW, not the canvas: a drag that ends off the plot would
          // otherwise leave the line stuck to the cursor.
          window.addEventListener('mouseup', () => this.onUp());
        }
        requestAnimationFrame(() => this.frame());
      });
    },

    // ── socket ───────────────────────────────────────────────────────────
    connect() {
      const proto = location.protocol === 'https:' ? 'wss' : 'ws';
      const s = new WebSocket(`${proto}://${location.host}/ws`);
      this.sock = s;
      s.onopen = () => {
        this.connected = true;
        this.retryIn = 0;
        if (this.symbol) this.watch(this.symbol, true);
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
      if (m.ev === 'refused') { this.refused = m.why || 'refused'; return; }
      if (m.ev === 'snapshot') {
        if (m.data.symbol !== this.symbol) return;
        this.trades = m.data.trades || [];
        this.quotes = m.data.quotes || [];
        this.reband(true);
        return;
      }
      if (m.ev === 'batch') {
        const d = m.data[this.symbol];
        if (!d) return;
        // PAUSE KEEPS BUFFERING. The window freezes; the data does not, so
        // resuming catches up rather than skipping the interval.
        for (const t of d.t) this.trades.push(t);
        for (const q of d.q) this.quotes.push(q);
        this.prune();
      }
    },

    watch(sym, silent) {
      sym = (sym || '').trim().toUpperCase();
      if (!sym) return;
      this.refused = '';
      if (this.symbol && this.symbol !== sym) {
        this.send({ action: 'unwatch', symbol: this.symbol });
      }
      this.symbol = sym;
      if (!silent) { this.trades = []; this.quotes = []; this.band = null; }
      this.send({ action: 'watch', symbol: sym, window_s: this.windowS });
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
      const cv = document.getElementById('lv-canvas');
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

    // ── the frame loop ───────────────────────────────────────────────────
    frame() {
      const now = performance.now();
      // ~30fps is plenty for a tape and halves the work of 60.
      if (now - this.lastDraw > 33) { this.lastDraw = now; this.draw(); }
      requestAnimationFrame(() => this.frame());
    },

    draw() {
      const cv = document.getElementById('lv-canvas');
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
      // The spread is filled very faintly so it reads as a band the prints sit
      // inside or outside of, with the edges bright enough to locate exactly.
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
          ctx.fillStyle = 'rgba(150,170,190,0.10)';
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
      // are the same blue disc; with alpha the stack darkens and with a rim
      // the individual prints stay countable.
      let count = 0;
      ctx.fillStyle = 'rgba(52,152,219,0.34)';
      ctx.strokeStyle = 'rgba(120,200,255,0.85)';
      ctx.lineWidth = 0.9;
      for (const t of this.trades) {
        if (t.t < tStart || t.t > tEnd) continue;
        count += 1;
        const r = Math.max(1.3, Math.sqrt(Math.max(1, t.s)) * 0.62);
        ctx.beginPath();
        ctx.arc(X(t.t), Y(t.p), r, 0, Math.PI * 2);
        ctx.fill();
        if (r > 2) ctx.stroke();
      }
      // A ROLLING 60-SECOND COUNT, not the window's count divided by the
      // window's length. That read a third of the true rate on a 3-minute
      // window and climbed from zero as the buffer filled — 8, 10, 12, 15,
      // 21 against a verified 56, which is exactly 56/3 converging. A rate
      // has to be over a fixed interval, and one minute is the interval the
      // reference is quoted in.
      const minuteAgo = tEnd - 60000;
      let inMinute = 0;
      for (let i = this.trades.length - 1; i >= 0; i--) {
        const t = this.trades[i].t;
        if (t > tEnd) continue;
        if (t < minuteAgo) break;
        inMinute += 1;
      }
      // Under a minute of data cannot report a per-minute rate honestly.
      this.tpm = this.bufferedS() >= 55 ? inMinute : null;
      this.tpmPartial = this.bufferedS() < 55;

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

      // ── 6. the placed price lines ─────────────────────────────────────
      //
      // Drawn LAST so they sit over the tape: the point is to see whether
      // anything traded at that level, and a line under the prints is a line
      // being read through them.
      for (const ln of this.lines) {
        if (ln.p < lo || ln.p > hi) continue;
        const y = Y(ln.p);
        ctx.setLineDash([5, 4]);
        ctx.strokeStyle = ln === this.lines[this.dragIdx]
          ? 'rgba(255,255,255,0.95)' : 'rgba(255,214,102,0.85)';
        ctx.lineWidth = 1.2;
        ctx.beginPath();
        ctx.moveTo(padL, y); ctx.lineTo(padL + plotW, y);
        ctx.stroke();
        ctx.setLineDash([]);
        // The count is the reading: "nothing has traded here in three
        // minutes" is the answer that decides whether to post there.
        const hits = this.lineHits(ln.p, tStart, tEnd);
        ctx.font = '600 10px sans-serif';
        ctx.fillStyle = hits ? '#ffd666' : '#8a8a8a';
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

      // The last trade, marked on the axis in accent blue — the one price
      // that is not a grid line and the one most often being read.
      const last = this.lastPrice();
      if (last != null && last >= lo && last <= hi) {
        const y = Y(last);
        ctx.beginPath();
        ctx.moveTo(padL, y); ctx.lineTo(padL + plotW, y);
        ctx.strokeStyle = 'rgba(52,152,219,0.30)';
        ctx.stroke();
        ctx.font = '700 12px sans-serif';
        ctx.fillStyle = LV_BLUE;
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
        ctx.fillStyle = 'rgba(52,152,219,0.45)';
        ctx.fill();
        ctx.fillStyle = '#6a6a6a';
        ctx.fillText(String(sz), lx + r - 4, padT + plotH + 4);
        lx += r * 2 + 20;
      }
    },

    /* Screen geometry, in one place.
     *
     * draw() and the pointer handlers each computed their own padding, window
     * bounds and scales; two copies of a mapping drift, and a hover that
     * disagrees with what is drawn is worse than no hover. */
    geom() {
      const cv = document.getElementById('lv-canvas');
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
