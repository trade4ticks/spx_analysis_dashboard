/* Replay — the tape for a past session, on the Equities Scalp page.
 *
 * WHAT IT IS FOR, because that decides what it must not become. Four metric
 * families have failed to identify a tradeable name. The common thread is a
 * description being formalised into something subtly different, found only by
 * watching the tape and disagreeing. So this is not a metric, not a screen and
 * not a ranking — it is a way to look at one window and say "this, not that",
 * so a later metric can be built against labelled examples.
 *
 * NOTHING IS AGGREGATED IN THE ZOOMED VIEW. Every trade is drawn at its own
 * timestamp. Prints frequently share a millisecond — one marketable order
 * sweeping venues — and that clustering is the information. Pixel-slice
 * aggregation is exactly what made the other tools useless here.
 *
 * AREA, NOT RADIUS, for share count: at r ∝ size a 1-share print vanishes
 * beside a 200.
 *
 * A SEPARATE ALPINE SCOPE, not additions to equitiesScalp. It shares that
 * page and nothing else, and the scalp bundle is 83 KB that has no reason to
 * grow. The factory is tagged `isComponentScope` so check_alpine_refs picks it
 * up as a secondary scope — the mechanism that gate already has for exactly
 * this.
 */
'use strict';

function replayScope() {
  return {
    // ── state ─────────────────────────────────────────────────────────
    sessions: [],
    symbols: [],
    date: '',
    symbol: '',
    candles: null,
    win: null,
    view: null,              // {t0, t1, p0, p1} in session seconds / dollars
    full: null,              // the reset target
    mode: 'candles',
    printLimit: 20000,
    showNbbo: true,
    loading: false,
    error: '',
    hover: null,
    stats: null,
    dragging: false,
    _drag: null,
    _fetchTimer: null,
    _seq: 0,

    SESSION_SECONDS: 390 * 60,

    // ── lifecycle ─────────────────────────────────────────────────────
    async initReplay() {
      await this.loadSessions();
      window.addEventListener('resize', () => this.draw());
    },

    async loadSessions() {
      this.error = '';
      try {
        const r = await fetch('/api/replay/sessions');
        const d = await r.json();
        if (d.error) { this.error = d.error; return; }
        this.sessions = d.sessions || [];
        this.printLimit = d.print_limit || this.printLimit;
        if (this.sessions.length) await this.pickDate(this.sessions[0].date);
      } catch (e) { this.error = 'could not list sessions: ' + e; }
    },

    async pickDate(d) {
      this.date = d;
      this.symbol = '';
      this.candles = null;
      this.win = null;
      this.error = '';
      try {
        const r = await fetch('/api/replay/symbols?date=' + encodeURIComponent(d));
        const j = await r.json();
        if (j.error) { this.error = j.error; return; }
        this.symbols = j.symbols || [];
      } catch (e) { this.error = 'could not list symbols: ' + e; }
    },

    // A date's symbol count is shown in the picker because 2026-08-14 holds
    // 96 symbols where every other session holds 634-739, and a bare date
    // would offer a day that looks identical and is mostly empty.
    dateLabel(s) { return s.date + '  (' + s.symbols + ')'; },

    async pickSymbol(sym) {
      this.symbol = sym;
      this.error = '';
      this.loading = true;
      this.hover = null;
      try {
        const q = '?symbol=' + encodeURIComponent(sym) +
                  '&date=' + encodeURIComponent(this.date);
        const r = await fetch('/api/replay/candles' + q);
        const d = await r.json();
        if (d.error) { this.error = d.error; this.candles = null; return; }
        this.candles = d;
        this.resetView();
      } catch (e) {
        this.error = 'could not load ' + sym + ': ' + e;
      } finally { this.loading = false; }
    },

    // ── the view ──────────────────────────────────────────────────────
    resetView() {
      const c = this.candles;
      if (!c) return;
      let lo = Infinity, hi = -Infinity;
      for (let i = 0; i < c.minutes; i++) {
        if (c.l[i] !== null && c.l[i] < lo) lo = c.l[i];
        if (c.h[i] !== null && c.h[i] > hi) hi = c.h[i];
      }
      if (!isFinite(lo)) { lo = 0; hi = 1; }
      const pad = (hi - lo) * 0.04 || 0.05;
      this.full = { t0: 0, t1: this.SESSION_SECONDS, p0: lo - pad, p1: hi + pad };
      this.view = Object.assign({}, this.full);
      this.requestWindow();
    },

    // Debounced: a wheel gesture is dozens of events and each one would
    // otherwise be a request. The draw happens immediately from what is
    // already in hand, so the chart tracks the gesture and the data catches
    // up.
    requestWindow() {
      this.draw();
      clearTimeout(this._fetchTimer);
      this._fetchTimer = setTimeout(() => this.fetchWindow(), 130);
    },

    async fetchWindow() {
      if (!this.symbol || !this.view) return;
      const seq = ++this._seq;
      const v = this.view;
      const q = '?symbol=' + encodeURIComponent(this.symbol) +
                '&date=' + encodeURIComponent(this.date) +
                '&t0=' + v.t0.toFixed(3) + '&t1=' + v.t1.toFixed(3) +
                '&nbbo=' + (this.showNbbo ? 'true' : 'false');
      try {
        const r = await fetch('/api/replay/window' + q);
        const d = await r.json();
        // A STALE ANSWER MUST NOT LAND. Zooming fast puts several requests in
        // flight and they can return out of order; without this the chart
        // settles on whichever was slowest, not whichever was asked last.
        if (seq !== this._seq) return;
        if (d.error) { this.error = d.error; return; }
        this.error = '';
        this.win = d;
        this.mode = d.mode;
        this.stats = d.stats;
        this.draw();
      } catch (e) { this.error = 'window failed: ' + e; }
    },

    // ── mouse ─────────────────────────────────────────────────────────
    //
    // NATIVE GESTURES. Scroll zooms time, drag pans, double-click resets.
    // Hunting for buttons in a control panel while trying to read a chart is
    // the thing this is avoiding, so there are no zoom buttons at all.
    onWheel(ev) {
      if (!this.view) return;
      ev.preventDefault();
      const cv = this.$refs.cv;
      const rect = cv.getBoundingClientRect();
      const k = Math.exp((ev.deltaY > 0 ? 1 : -1) * -0.18);
      const v = this.view;
      // Shift (or alt) zooms PRICE. The default gesture is time, because
      // that is the axis being read.
      if (ev.shiftKey || ev.altKey) {
        const fy = (ev.clientY - rect.top) / rect.height;
        const at = v.p1 - fy * (v.p1 - v.p0);
        v.p0 = at - (at - v.p0) * k;
        v.p1 = at + (v.p1 - at) * k;
      } else {
        const fx = (ev.clientX - rect.left) / rect.width;
        const at = v.t0 + fx * (v.t1 - v.t0);
        // Anchored on the cursor, so the point under the pointer stays put.
        let t0 = at - (at - v.t0) * k;
        let t1 = at + (v.t1 - at) * k;
        // A floor of two seconds: past that the axis is narrower than the
        // clustering it exists to show.
        if (t1 - t0 < 2) { const m = (t0 + t1) / 2; t0 = m - 1; t1 = m + 1; }
        v.t0 = t0; v.t1 = t1;
      }
      this.requestWindow();
    },

    onDown(ev) {
      if (!this.view) return;
      this.dragging = true;
      this._drag = { x: ev.clientX, y: ev.clientY,
                     v: Object.assign({}, this.view) };
    },
    onMove(ev) {
      const cv = this.$refs.cv;
      if (!cv || !this.view) return;
      if (this.dragging && this._drag) {
        const rect = cv.getBoundingClientRect();
        const dx = (ev.clientX - this._drag.x) / rect.width;
        const dy = (ev.clientY - this._drag.y) / rect.height;
        const s = this._drag.v;
        const tw = s.t1 - s.t0, pw = s.p1 - s.p0;
        this.view = { t0: s.t0 - dx * tw, t1: s.t1 - dx * tw,
                      p0: s.p0 + dy * pw, p1: s.p1 + dy * pw };
        this.requestWindow();
        return;
      }
      this.hitTest(ev);
    },
    onUp() { this.dragging = false; this._drag = null; },
    onLeave() { this.dragging = false; this._drag = null; this.hover = null;
                this.draw(); },
    onDouble() { this.resetView(); },

    // HOVER SHOWS SHARE COUNT AND NOTHING ELSE. Not time, not venue, not
    // conditions — those are what turns a tape reader into a data browser.
    hitTest(ev) {
      const w = this.win;
      if (!w || this.mode !== 'prints' || !w.t) { this.hover = null; return; }
      const cv = this.$refs.cv;
      const rect = cv.getBoundingClientRect();
      const x = ev.clientX - rect.left, y = ev.clientY - rect.top;
      const v = this.view;
      const W = rect.width, H = rect.height;
      let best = null, bestD = 144;      // within 12px
      for (let i = 0; i < w.t.length; i++) {
        const px = ((w.t[i] - v.t0) / (v.t1 - v.t0)) * W;
        if (px < -20 || px > W + 20) continue;
        const py = H - ((w.p[i] - v.p0) / (v.p1 - v.p0)) * H;
        const d = (px - x) * (px - x) + (py - y) * (py - y);
        if (d < bestD) { bestD = d; best = { x: px, y: py, s: w.s[i] }; }
      }
      const changed = (best === null) !== (this.hover === null) ||
                      (best && this.hover && best.s !== this.hover.s);
      this.hover = best;
      if (changed) this.draw();
    },

    // ── drawing ───────────────────────────────────────────────────────
    draw() {
      const cv = this.$refs.cv;
      if (!cv || !this.view) return;
      const dpr = Math.min(2, window.devicePixelRatio || 1);
      const W = cv.clientWidth, H = cv.clientHeight;
      if (cv.width !== W * dpr || cv.height !== H * dpr) {
        cv.width = W * dpr; cv.height = H * dpr;
      }
      const ctx = cv.getContext('2d');
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, W, H);
      ctx.fillStyle = '#12161b';
      ctx.fillRect(0, 0, W, H);

      const v = this.view;
      const X = (t) => ((t - v.t0) / (v.t1 - v.t0)) * W;
      const Y = (p) => H - ((p - v.p0) / (v.p1 - v.p0)) * H;

      this.drawGrid(ctx, W, H, X, Y);
      if (this.mode === 'prints' && this.win && this.win.t) {
        if (this.showNbbo && this.win.nbbo) this.drawNbbo(ctx, X, Y, W);
        this.drawPrints(ctx, X, Y, W, H);
      } else {
        this.drawCandles(ctx, X, Y, W, H);
      }
      if (this.hover) {
        ctx.strokeStyle = '#e84393';
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.arc(this.hover.x, this.hover.y, 9, 0, 6.283185307179586);
        ctx.stroke();
      }
    },

    drawGrid(ctx, W, H, X, Y) {
      ctx.strokeStyle = 'rgba(255,255,255,0.06)';
      ctx.fillStyle = '#7c8894';
      ctx.font = '10px monospace';
      ctx.lineWidth = 1;
      const v = this.view;
      // Time gridlines at a step that keeps roughly six labels on screen
      // whatever the zoom.
      const span = v.t1 - v.t0;
      const steps = [1, 5, 15, 30, 60, 300, 900, 1800, 3600, 7200];
      let step = steps[steps.length - 1];
      for (const s of steps) { if (span / s <= 8) { step = s; break; } }
      for (let t = Math.ceil(v.t0 / step) * step; t < v.t1; t += step) {
        const x = X(t);
        ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, H); ctx.stroke();
        ctx.fillText(this.clock(t), x + 3, H - 4);
      }
      for (let i = 1; i < 5; i++) {
        const y = (H / 5) * i;
        ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(W, y); ctx.stroke();
        const p = v.p1 - (v.p1 - v.p0) * (i / 5);
        ctx.fillText(p.toFixed(2), 3, y - 3);
      }
    },

    drawPrints(ctx, X, Y, W, H) {
      const w = this.win;
      // ONE Path2D, filled once. Measured: arc() per trade is 3.6 s at 742k
      // and 1.2 s a frame; a single path is 355 ms and 133 ms. At the counts
      // this view actually holds — under 20,000 — it is 60fps either way, and
      // the path costs nothing to prefer.
      const path = new Path2D();
      const k = 0.42;
      for (let i = 0; i < w.t.length; i++) {
        const x = X(w.t[i]);
        if (x < -8 || x > W + 8) continue;
        const y = Y(w.p[i]);
        // AREA proportional to share count: r = k*sqrt(size).
        const r = Math.max(0.6, k * Math.sqrt(w.s[i]));
        path.moveTo(x + r, y);
        path.arc(x, y, r, 0, 6.283185307179586);
      }
      ctx.fillStyle = 'rgba(196,204,212,0.62)';
      ctx.fill(path);
    },

    // Steps that hold their level until the quote changes — not a slope
    // between change points, which would draw a book that drifted when it
    // actually jumped.
    drawNbbo(ctx, X, Y, W) {
      const nb = this.win.nbbo;
      if (!nb || !nb.t.length) return;
      for (const [arr, colour] of [[nb.bid, 'rgba(52,152,219,0.55)'],
                                   [nb.ask, 'rgba(232,67,147,0.55)']]) {
        ctx.strokeStyle = colour;
        ctx.lineWidth = 1;
        ctx.beginPath();
        let lastY = null;
        for (let i = 0; i < nb.t.length; i++) {
          if (arr[i] === null) continue;
          const x = X(nb.t[i]), y = Y(arr[i]);
          if (lastY === null) ctx.moveTo(x, y);
          else { ctx.lineTo(x, lastY); ctx.lineTo(x, y); }
          lastY = y;
        }
        if (lastY !== null) ctx.lineTo(W, lastY);
        ctx.stroke();
      }
    },

    drawCandles(ctx, X, Y, W, H) {
      const c = this.candles;
      if (!c) return;
      const v = this.view;
      const wpx = Math.max(1, (X(60) - X(0)) * 0.72);
      for (let i = 0; i < c.minutes; i++) {
        if (c.o[i] === null) continue;
        const t = i * 60;
        if (t + 60 < v.t0 || t > v.t1) continue;
        const x = X(t + 30);
        const up = c.c[i] >= c.o[i];
        ctx.strokeStyle = up ? 'rgba(107,214,160,0.85)' : 'rgba(232,107,107,0.85)';
        ctx.fillStyle = up ? 'rgba(107,214,160,0.35)' : 'rgba(232,107,107,0.35)';
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(x, Y(c.h[i])); ctx.lineTo(x, Y(c.l[i]));
        ctx.stroke();
        const yo = Y(c.o[i]), yc = Y(c.c[i]);
        const top = Math.min(yo, yc);
        const hgt = Math.max(1, Math.abs(yc - yo));
        ctx.fillRect(x - wpx / 2, top, wpx, hgt);
        ctx.strokeRect(x - wpx / 2, top, wpx, hgt);
      }
      // Volume, along the bottom eighth, so a candle can be read against how
      // much traded in it.
      let vmax = 0;
      for (let i = 0; i < c.minutes; i++) if (c.v[i] > vmax) vmax = c.v[i];
      if (vmax > 0) {
        ctx.fillStyle = 'rgba(224,168,0,0.30)';
        for (let i = 0; i < c.minutes; i++) {
          if (!c.v[i]) continue;
          const t = i * 60;
          if (t + 60 < v.t0 || t > v.t1) continue;
          const h = (c.v[i] / vmax) * (H / 8);
          ctx.fillRect(X(t + 30) - wpx / 2, H - h, wpx, h);
        }
      }
    },

    // ── labels ────────────────────────────────────────────────────────
    clock(sec) {
      const s = Math.max(0, Math.round(sec));
      const m = 9 * 60 + 30 + Math.floor(s / 60);
      const hh = Math.floor(m / 60), mm = m % 60;
      return String(hh).padStart(2, '0') + ':' + String(mm).padStart(2, '0') +
             (this.view && this.view.t1 - this.view.t0 < 300
               ? ':' + String(s % 60).padStart(2, '0') : '');
    },

    // AT FULL-SESSION ZOOM THE PICTURE IS MEANINGLESS, so how far in you are
    // has to be on screen rather than inferred from the axis.
    widthLabel() {
      if (!this.view) return '';
      const s = this.view.t1 - this.view.t0;
      if (s < 90) return s.toFixed(0) + 's window';
      if (s < 5400) return (s / 60).toFixed(1) + ' min window';
      return (s / 3600).toFixed(1) + ' h window';
    },

    // A dense field of dots and a dense field of candles look alike and mean
    // different things, so the mode is stated — with the count and the limit,
    // so the switch explains itself rather than just happening.
    modeLabel() {
      if (!this.win) return '';
      if (this.mode === 'prints') {
        return 'individual prints — ' + this.win.count.toLocaleString() +
               ' trades';
      }
      return '1-minute candles — ' + this.win.count.toLocaleString() +
             ' trades in view, over the ' + this.printLimit.toLocaleString() +
             ' limit; zoom in for prints';
    },

    fmt(v, places) {
      return (v === null || v === undefined || !isFinite(v))
        ? '—' : Number(v).toFixed(places === undefined ? 1 : places);
    },
    fmtInt(v) {
      return (v === null || v === undefined || !isFinite(v))
        ? '—' : Math.round(v).toLocaleString();
    },
    hoverLabel() {
      return this.hover ? this.hover.s.toLocaleString() + ' sh' : '';
    },
  };
}

// The gate's secondary-scope hook: only tagged factories are called, so this
// opts in explicitly rather than the checker running arbitrary page code.
replayScope.isComponentScope = true;
if (typeof window !== 'undefined') window.replayScope = replayScope;
