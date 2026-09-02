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
    // Half-height of the price window, in cents. Y is expressed as distance
    // from the anchor because that is the reading — judging whether a 10-cent
    // capture is available is a question about distance, not about levels.
    spanCents: 30,
    anchor: null,
    paused: false,
    showQuotes: true,

    trades: [],
    quotes: [],
    hover: null,
    // Trades per minute, from the window itself rather than a counter: the
    // arrival rate is a proxy for fillability and a stale one is worse than
    // none.
    tpm: 0,
    lastDraw: 0,

    init() {
      this.connect();
      this.$nextTick(() => {
        this.resize();
        window.addEventListener('resize', () => this.resize());
        const cv = document.getElementById('lv-canvas');
        if (cv) {
          cv.addEventListener('mousemove', e => this.onHover(e));
          cv.addEventListener('mouseleave', () => { this.hover = null; });
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
        this.reanchor(true);
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
      if (!silent) { this.trades = []; this.quotes = []; this.anchor = null; }
      this.send({ action: 'watch', symbol: sym, window_s: this.windowS });
    },

    /* Dropped by TIME only. The server caps by count as well, but the page's
     * job is the window it is drawing, and trimming by count here would drop
     * the oldest prints of a busy minute while the axis still claimed to
     * cover it. */
    prune() {
      const cutoff = Date.now() - this.windowS * 1000 - 2000;
      while (this.trades.length && this.trades[0].t < cutoff) this.trades.shift();
      while (this.quotes.length && this.quotes[0].t < cutoff) this.quotes.shift();
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

    lastPrice() {
      return this.trades.length ? this.trades[this.trades.length - 1].p : null;
    },

    /* RECENTRE ONLY AT THE EDGE, never on a timer. A plot that jumps every
     * thirty seconds destroys the anchor position is being read against —
     * which is the whole point of a y axis in cents from a fixed price. */
    reanchor(force) {
      const p = this.lastPrice();
      if (p == null) return;
      if (force || this.anchor == null) { this.anchor = p; return; }
      const halfDollars = this.spanCents / 100;
      const edge = halfDollars * 0.85;
      if (Math.abs(p - this.anchor) > edge) this.anchor = p;
    },

    yRange() {
      const a = this.anchor != null ? this.anchor : (this.lastPrice() || 0);
      const h = this.spanCents / 100;
      return { lo: a - h, hi: a + h, anchor: a };
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

      const padL = 8, padR = 62, padT = 8, padB = 20;
      const plotW = Math.max(10, w - padL - padR);
      const plotH = Math.max(10, h - padT - padB);

      if (!this.paused) this.reanchor(false);
      const tEnd = this.paused ? (this.frozenEnd || Date.now()) : Date.now();
      if (!this.paused) this.frozenEnd = null;
      else if (!this.frozenEnd) this.frozenEnd = Date.now();
      const tStart = tEnd - this.windowS * 1000;
      const { lo, hi, anchor } = this.yRange();

      const X = t => padL + ((t - tStart) / (tEnd - tStart)) * plotW;
      const Y = p => padT + (1 - (p - lo) / (hi - lo)) * plotH;

      this.drawGrid(ctx, padL, padT, plotW, plotH, lo, hi, anchor, tStart, tEnd, X, Y, w);

      // ── NBBO, behind everything and muted ──────────────────────────────
      if (this.showQuotes && this.quotes.length) {
        ctx.lineWidth = 1;
        for (const [key, col] of [['bp', 'rgba(120,150,175,0.55)'],
                                  ['ap', 'rgba(175,120,150,0.55)']]) {
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
      let count = 0;
      ctx.fillStyle = 'rgba(52,152,219,0.72)';
      for (const t of this.trades) {
        if (t.t < tStart || t.t > tEnd) continue;
        count += 1;
        const r = Math.max(1.1, Math.sqrt(Math.max(1, t.s)) * 0.62);
        ctx.beginPath();
        ctx.arc(X(t.t), Y(t.p), r, 0, Math.PI * 2);
        ctx.fill();
      }
      this.tpm = this.windowS > 0
        ? Math.round(count / (this.windowS / 60)) : 0;

      if (this.hover && this.hover.rec) {
        const t = this.hover.rec;
        ctx.strokeStyle = '#fff';
        ctx.lineWidth = 1.2;
        ctx.beginPath();
        ctx.arc(X(t.t), Y(t.p), Math.max(4, Math.sqrt(Math.max(1, t.s)) * 0.62 + 3),
                0, Math.PI * 2);
        ctx.stroke();
      }
    },

    drawGrid(ctx, padL, padT, plotW, plotH, lo, hi, anchor, tStart, tEnd, X, Y, w) {
      ctx.strokeStyle = 'rgba(255,255,255,0.06)';
      ctx.fillStyle = '#7a7a7a';
      ctx.font = '9px sans-serif';
      ctx.lineWidth = 1;

      // Y IN CENTS FROM THE ANCHOR, with the absolute price alongside. The
      // question being asked of this axis is "is ten cents available", and a
      // ladder of absolute levels answers it only after arithmetic.
      const stepC = this.spanCents <= 10 ? 2 : this.spanCents <= 30 ? 5
                  : this.spanCents <= 60 ? 10 : 25;
      for (let c = -this.spanCents; c <= this.spanCents; c += stepC) {
        const p = anchor + c / 100;
        if (p < lo || p > hi) continue;
        const y = Y(p);
        ctx.globalAlpha = c === 0 ? 0.35 : 1;
        ctx.beginPath();
        ctx.moveTo(padL, y); ctx.lineTo(padL + plotW, y);
        ctx.strokeStyle = c === 0 ? 'rgba(255,255,255,0.22)'
                                  : 'rgba(255,255,255,0.06)';
        ctx.stroke();
        ctx.globalAlpha = 1;
        ctx.textAlign = 'left';
        ctx.fillStyle = c === 0 ? '#bbb' : '#7a7a7a';
        ctx.fillText((c > 0 ? '+' : '') + c + '¢', padL + plotW + 5, y + 3);
        ctx.fillStyle = '#555';
        ctx.fillText(p.toFixed(2), padL + plotW + 32, y + 3);
      }

      ctx.strokeStyle = 'rgba(255,255,255,0.06)';
      ctx.fillStyle = '#7a7a7a';
      ctx.textAlign = 'center';
      const secs = (tEnd - tStart) / 1000;
      const stepS = secs <= 60 ? 10 : secs <= 180 ? 30 : secs <= 600 ? 60 : 300;
      const first = Math.ceil(tStart / (stepS * 1000)) * stepS * 1000;
      for (let t = first; t <= tEnd; t += stepS * 1000) {
        const x = X(t);
        ctx.beginPath();
        ctx.moveTo(x, padT); ctx.lineTo(x, padT + plotH);
        ctx.stroke();
        ctx.fillText(lvFmtClock(t), x, padT + plotH + 13);
      }
    },

    onHover(e) {
      const cv = document.getElementById('lv-canvas');
      if (!cv) return;
      const r = cv.getBoundingClientRect();
      const mx = e.clientX - r.left, my = e.clientY - r.top;
      const padL = 8, padR = 62, padT = 8, padB = 20;
      const plotW = Math.max(10, r.width - padL - padR);
      const plotH = Math.max(10, r.height - padT - padB);
      const tEnd = this.paused ? (this.frozenEnd || Date.now()) : Date.now();
      const tStart = tEnd - this.windowS * 1000;
      const { lo, hi } = this.yRange();
      const X = t => padL + ((t - tStart) / (tEnd - tStart)) * plotW;
      const Y = p => padT + (1 - (p - lo) / (hi - lo)) * plotH;

      let best = null, bestD = 12 * 12;
      for (const t of this.trades) {
        if (t.t < tStart || t.t > tEnd) continue;
        const dx = X(t.t) - mx, dy = Y(t.p) - my;
        const d = dx * dx + dy * dy;
        if (d < bestD) { bestD = d; best = t; }
      }
      this.hover = best ? { rec: best, x: mx, y: my } : null;
    },

    hoverText() {
      const t = this.hover && this.hover.rec;
      if (!t) return '';
      const ms = String(t.t % 1000).padStart(3, '0');
      return `${lvFmtClock(t.t)}.${ms}  ${t.p.toFixed(2)}  ${t.s} sh  `
           + `venue ${t.x}${t.c && t.c.length ? '  cond ' + t.c.join(',') : ''}`;
    },

    // ── controls ─────────────────────────────────────────────────────────
    zoomX(f) {
      this.windowS = Math.max(15, Math.min(900, Math.round(this.windowS * f)));
    },
    zoomY(f) {
      this.spanCents = Math.max(2, Math.min(500, Math.round(this.spanCents * f)));
    },
    recenter() { this.anchor = this.lastPrice(); },
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
