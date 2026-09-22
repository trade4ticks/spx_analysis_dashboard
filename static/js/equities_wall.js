/* Equities Wall — dozens to a hundred small live tapes, for watching.
 *
 * WHAT IT IS FOR. Opening a tape pane on a name, waiting a minute to see
 * whether it moves, deciding no, and moving on is an hour for thirty names.
 * The wall shows the whole watchlist at once and lets the eye do the picking.
 * There is deliberately NO metric, NO score, NO colour-coding and NO ranking
 * on this page: the scan page's number surfaces names that cannot be traded,
 * and the judgement being made here — is this one bouncy, and is its spread
 * worth having — is one a person makes from the picture in a glance.
 *
 * THE VERTICAL SCALE IS THE POINT. Each pane scales so the spread fills a
 * fixed share of its height (default 60%), centred on the mid. "Bouncy" is
 * relative to the spread's own width: a half-spread move on FDX (7c) and on
 * LLY (50c) are the same trade, and on a shared axis the second looks like
 * everything and the first like nothing. Scaling to the spread also stops
 * absolute width dominating the wall, which is why the spread in cents is
 * printed as a NUMBER — the picture says how it moves, the number says what
 * it costs.
 *
 * WHY THE SCALE DOES NOT JITTER. The height comes from the TYPICAL spread —
 * a median of the last minute, computed server-side — not from the current
 * one. A pane rescaling on one wide quote makes every trade in it collapse
 * to the centre line for a second, and on a wall of a hundred that reads as
 * the symbol going still.
 *
 * COST. One browser connection for the whole page (the service caps browsers
 * at 8), one frame a second, and a pane with nothing new is not redrawn —
 * the server does not even send it. Panes scrolled off the screen are not
 * drawn at all.
 */
'use strict';

// ── the numbers, named ──────────────────────────────────────────────────────
const WL_SHARE_MIN = 0.05;
const WL_SHARE_MAX = 0.95;
// A pane with nothing new still refreshes this often. Without it, a quiet
// pane's bubbles sit at the x they were drawn at while the window slides on
// underneath them, so trades that have fallen out of the window stay on
// screen — and a quiet name looks busier than it is, which is the one error
// this page must not make.
const WL_FORCE_REDRAW_MS = 5000;
// How fast the pane re-centres on the mid, per frame. Slow enough that the
// picture does not slide about under a wandering quote, fast enough to catch
// a real move in a couple of seconds.
const WL_RECENTRE = 0.35;
// Bubble radius: sqrt of the share count, so a 1,000-share print is bigger
// than a 100 but not ten times bigger. 91% of this tape is under 40 shares
// and that part is the part being traded in, so the floor is visible rather
// than a dot.
const WL_BUBBLE_MIN = 1.0;
const WL_BUBBLE_MAX = 6.0;
const WL_BUBBLE_REF = 200;

/* COLOURS COME FROM tape_theme.js, WHICH EQUITIES LIVE ALSO READS: blue bid,
 * pink ask, neutral translucent prints. A wall pane is meant to look like a
 * small pane of that page, and a second copy of a colour here is the two
 * drifting apart the next time either is edited.
 *
 * The RADII are this page's own: a wall pane is a fifth of the size, and the
 * tape page's area-proportional rule would put an 8px disc on a 124px pane
 * for a 200-share print. */

// The per-symbol tape, OUTSIDE Alpine's proxy. A hundred symbols of trade and
// quote arrays behind a reactive proxy means every push is wrapped and every
// frame walks them; nothing in here is read by a template, only by the canvas
// code, so it stays plain. (The headline numbers that ARE read by templates
// live on the component, in `head`.)
const WL_DATA = { panes: {} };


function wlClampShare(v) {
  const f = Number(v);
  if (!isFinite(f)) return null;
  return Math.min(WL_SHARE_MAX, Math.max(WL_SHARE_MIN, f));
}


/* The pane's price window: the spread fills `share` of the height.
 *
 * Returns dollars, plus which input it came from, because a pane scaled off
 * its trades rather than its spread is a pane whose spread number is missing
 * and the two must not look alike.
 */
function wlScale(centre, typicalCents, share, tradeRange) {
  if (!(centre > 0)) return null;
  const s = wlClampShare(share) || 0.6;
  let range, from;
  if (typicalCents > 0) {
    range = (typicalCents / 100) / s;
    from = 'spread';
  } else if (tradeRange > 0) {
    // No quote: the tape is all there is, and a pane drawn to its own prints
    // with a little air around them is still worth looking at.
    range = tradeRange * 1.4;
    from = 'trades';
  } else {
    range = Math.max(centre * 0.0005, 0.01);
    from = 'default';
  }
  return { lo: centre - range / 2, hi: centre + range / 2, range: range,
           from: from };
}


/* Where the pane sits. An EMA toward the mid, except across a gap.
 *
 * Crawling toward a mid a full pane-height away would leave every print off
 * the top of the pane for several seconds — the one moment the name is worth
 * watching. Beyond a pane's height it jumps.
 */
function wlCentre(prev, mid, range) {
  if (!(mid > 0)) return prev;
  if (!(prev > 0)) return mid;
  if (Math.abs(mid - prev) > range) return mid;
  return prev + (mid - prev) * WL_RECENTRE;
}


function wlBubbleR(size) {
  const s = Number(size);
  if (!(s > 0)) return WL_BUBBLE_MIN;
  return Math.min(WL_BUBBLE_MAX,
                  WL_BUBBLE_MIN + 1.6 * Math.sqrt(s / WL_BUBBLE_REF));
}


/* Drop what has fallen out of the window. The arrays are time-ordered, so
 * this is a prefix, and it is done on the CLIENT because the server sends a
 * delta and never re-sends what it has already sent. */
function wlTrim(arr, cutoffMs) {
  let i = 0;
  while (i < arr.length && arr[i][0] < cutoffMs) i++;
  return i ? arr.slice(i) : arr;
}


/* Does this pane get redrawn this frame?
 *
 * Off screen: never — a hundred panes where twenty are visible is a fifth of
 * the work. Nothing new: not until the forced refresh, which is what the
 * server's "a symbol with nothing new is not sent" is for. */
function wlDue(pane, nowMs, visible) {
  if (!visible) return false;
  if (pane.dirty) return true;
  return (nowMs - (pane.drawnAt || 0)) >= WL_FORCE_REDRAW_MS;
}


/* The spread filter, which FADES a pane rather than removing it.
 *
 * NOT A SUBSCRIPTION, and not a filter of the list. Every name on the
 * watchlist keeps streaming and keeps accumulating its two minutes whatever
 * this says, so a symbol that dips under the threshold and comes back has
 * its history intact rather than rebuilding from nothing. And every pane
 * keeps its place in the grid: a wall that reflows as names cross the line
 * is harder to read than one where the failing panes simply go quiet-looking
 * — you are recognising positions, not names.
 *
 * TWO THINGS STOP IT FLICKERING.
 *
 *   * It reads the TYPICAL spread (the server's median of the last minute),
 *     never the instantaneous one. A name sitting on the threshold would
 *     otherwise dim and undim every few seconds on single quotes.
 *   * It dims LATE and undims AT ONCE. A name has to be below for
 *     WL_DIM_AFTER_MS before it fades; the moment it qualifies again it is
 *     full strength. The asymmetry is deliberate — the cost of showing a
 *     too-narrow name for ten more seconds is nothing, and the cost of a
 *     pane strobing at the boundary is that the whole wall is unreadable.
 *
 * A pane that has never quoted has no spread to compare, so it cannot be
 * shown to qualify and fades with the rest — its header reads "—", which is
 * what says why.
 */
const WL_DIM_AFTER_MS = 10000;

function wlDimState(pane, minCents, nowMs) {
  const min = Number(minCents);
  if (!(min > 0)) return { dim: false, belowSince: 0 };
  const typical = pane ? pane.tp : null;
  if (typical != null && isFinite(typical) && typical >= min) {
    return { dim: false, belowSince: 0 };
  }
  const since = pane && pane.belowSince ? pane.belowSince : nowMs;
  return { dim: (nowMs - since) >= WL_DIM_AFTER_MS, belowSince: since };
}


/* The share of the pane this symbol's spread fills: its own override, or the
 * page's setting. The override is a fact about the symbol ("LLY needs more
 * room than the default"), which is why it is saved next to the ticker. */
function wlEffShare(entry, pageShare) {
  const own = entry && entry.scale != null ? wlClampShare(entry.scale) : null;
  return own != null ? own : (wlClampShare(pageShare) || 0.6);
}


function wlParseSymbols(text) {
  const out = [];
  const seen = {};
  for (const raw of String(text || '').split(/[\s,;]+/)) {
    const s = raw.trim().toUpperCase();
    if (!s || !/^[A-Z0-9]+$/.test(s) || seen[s]) continue;
    seen[s] = 1;
    out.push(s);
  }
  return out;
}


function wlFmtSpread(cents) {
  if (cents == null || !isFinite(cents)) return '—';
  return (cents < 10 ? cents.toFixed(1) : cents.toFixed(0)) + 'c';
}


function wlFmtPrice(p) {
  if (p == null || !isFinite(p)) return '';
  return p >= 100 ? p.toFixed(2) : p.toFixed(2);
}


document.addEventListener('alpine:init', () => {
  Alpine.data('equitiesWall', () => ({

    // ── state ───────────────────────────────────────────────────────────
    entries: [],              // [{symbol, scale}] — the server's list
    settings: { window_s: 120, spread_share: 0.60, min_spread_cents: 0 },
    caps: { symbols: 120, retain_s: 180, share_min: 0.05, share_max: 0.95 },
    head: {},                 // sym -> {sp, px, from} — the only reactive tape
    selected: '',
    paneWidth: 200,

    connected: false,
    warning: '',
    busy: false,
    lastTick: null,
    skewMs: 0,
    drawn: 0,
    offscreen: 0,
    dimmed: 0,
    sock: null,
    showManual: false,
    manualText: '',
    saveTimer: null,

    // ── lifecycle ───────────────────────────────────────────────────────
    init() {
      this.restore();
      this.observer = new IntersectionObserver((rows) => {
        for (const r of rows) {
          const vis = r.isIntersecting ? '1' : '0';
          r.target.dataset.vis = vis;
          // Coming back on screen redraws from the buffer, whole: the pane
          // has been ignoring frames while it was away.
          const p = WL_DATA.panes[r.target.dataset.sym];
          if (p && vis === '1') p.dirty = true;
        }
      }, { root: null, rootMargin: '120px' });
      this.connect();
      // ONE TIMER for the whole page. Every pane is drawn from it, so a wall
      // of a hundred is one frame's work a second rather than a hundred
      // independent loops.
      setInterval(() => this.drawAll(), 1000);
      window.addEventListener('resize', () => this.dirtyAll());
    },

    restore() {
      // Pane WIDTH is the one setting that is local: it is about this screen,
      // not about the watchlist, and the same list is read on a laptop and on
      // a 32-inch monitor. The window and the spread share live on the
      // server, with the list.
      try {
        const w = localStorage.getItem('equitiesWall.paneWidth');
        if (w) this.paneWidth = Math.min(420, Math.max(120, Number(w) || 200));
      } catch (e) { /* private window: the default is fine */ }
    },

    connect() {
      const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
      // ONE SOCKET FOR THE WHOLE PAGE. The service caps browser connections
      // at 8; a hundred panes with one each is twelve times over, and the
      // ninth pane would simply be refused.
      const sock = new WebSocket(proto + '//' + location.host + '/wall/ws');
      this.sock = sock;
      sock.onopen = () => { this.connected = true; this.warning = ''; };
      sock.onclose = () => {
        this.connected = false;
        // Said out loud. A wall that quietly stops updating looks exactly
        // like a market that went still, which is what the page is for.
        this.warning = 'disconnected — retrying';
        setTimeout(() => this.connect(), 2000);
      };
      sock.onmessage = (e) => this.onMessage(JSON.parse(e.data));
    },

    onMessage(m) {
      if (m.ev === 'hello' || m.ev === 'watchlist') {
        this.takeState(m);
        if (m.ev === 'watchlist') this.dirtyAll();
      } else if (m.ev === 'tick') {
        this.onTick(m);
      }
    },

    takeState(m) {
      if (m.entries) this.entries = m.entries;
      if (m.settings) this.settings = m.settings;
      if (m.caps) this.caps = m.caps;
      if (m.error) this.warning = 'watchlist: ' + m.error;
    },

    // ── the tape ────────────────────────────────────────────────────────
    onTick(m) {
      // The server's clock, not this browser's. Trade timestamps come from
      // the exchange, and drawing them against a laptop clock a few seconds
      // out puts every print off the edge of every pane.
      this.skewMs = (m.at * 1000) - Date.now();
      this.lastTick = Date.now();
      const windowMs = (m.window_s || this.settings.window_s) * 1000;
      const cutoff = (m.at * 1000) - windowMs;
      for (const [sym, cell] of Object.entries(m.syms || {})) {
        let p = WL_DATA.panes[sym];
        if (!p || cell.full) {
          p = WL_DATA.panes[sym] = { sym: sym, trades: [], quotes: [],
                                     centre: 0, dirty: true, drawnAt: 0 };
        }
        if (cell.q0) p.quotes.push(cell.q0);
        if (cell.q && cell.q.length) p.quotes = p.quotes.concat(cell.q);
        if (cell.t && cell.t.length) p.trades = p.trades.concat(cell.t);
        p.trades = wlTrim(p.trades, cutoff);
        // ONE SAMPLE OLDER THAN THE WINDOW IS KEPT: it is where the band
        // enters the pane from the left. Trimming it would leave a symbol
        // that has not requoted for two minutes — the quiet name this page is
        // for — with no band at all.
        p.quotes = this.trimQuotes(p.quotes, cutoff);
        p.sp = cell.sp;
        p.tp = cell.tp;
        p.mid = cell.mid;
        p.dirty = true;
        if (!this.head[sym]) {
          this.head[sym] = { sp: '—', px: '', from: '', dim: false };
        }
        const h = this.head[sym];
        h.sp = wlFmtSpread(cell.sp);
        h.px = wlFmtPrice(cell.t && cell.t.length
                          ? cell.t[cell.t.length - 1][1] : cell.mid);
        h.from = cell.tp > 0 ? 'spread' : 'trades';
      }
    },

    trimQuotes(arr, cutoffMs) {
      let i = 0;
      while (i + 1 < arr.length && arr[i + 1][0] <= cutoffMs) i++;
      return i ? arr.slice(i) : arr;
    },

    dirtyAll() {
      for (const p of Object.values(WL_DATA.panes)) p.dirty = true;
    },

    // ── drawing ─────────────────────────────────────────────────────────
    /* The fade is decided for EVERY held symbol, every second — not only for
     * the panes being drawn. A pane skipped because it is off screen or has
     * nothing new still has a ten-second timer running against it, and a
     * name that goes quiet must still fade while nobody is looking at it,
     * or scrolling back to it shows a stale answer. The class rides on
     * `head`, which is the reactive side of the page: WL_DATA is outside
     * Alpine's proxy on purpose, so writing the flag there would change
     * nothing on screen. */
    refreshDim(nowMs) {
      const min = this.settings.min_spread_cents;
      let dimmed = 0;
      for (const p of Object.values(WL_DATA.panes)) {
        const st = wlDimState(p, min, nowMs);
        p.belowSince = st.belowSince;
        const h = this.head[p.sym];
        if (h && h.dim !== st.dim) h.dim = st.dim;
        if (st.dim) dimmed++;
      }
      this.dimmed = dimmed;
    },

    drawAll() {
      const grid = this.$refs.grid;
      if (!grid) return;
      const nowMs = Date.now() + this.skewMs;
      this.refreshDim(nowMs);
      let drawn = 0;
      let off = 0;
      for (const el of grid.querySelectorAll('.wl-pane')) {
        if (!el.dataset.obs) { el.dataset.obs = '1'; this.observer.observe(el); }
        const p = WL_DATA.panes[el.dataset.sym];
        if (!p) continue;
        const visible = el.dataset.vis !== '0';
        if (!visible) off++;
        if (!wlDue(p, nowMs, visible)) continue;
        this.drawPane(el, p, nowMs);
        p.dirty = false;
        p.drawnAt = nowMs;
        drawn++;
      }
      this.drawn = drawn;
      this.offscreen = off;
    },

    entryFor(sym) {
      return this.entries.find(e => e.symbol === sym) || null;
    },

    drawPane(el, p, nowMs) {
      const cv = el.querySelector('canvas');
      if (!cv) return;
      const dpr = window.devicePixelRatio || 1;
      const w = cv.clientWidth;
      const h = cv.clientHeight;
      if (!w || !h) return;
      if (cv.width !== Math.round(w * dpr) || cv.height !== Math.round(h * dpr)) {
        cv.width = Math.round(w * dpr);
        cv.height = Math.round(h * dpr);
      }
      const ctx = cv.getContext('2d');
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, w, h);

      const windowMs = this.settings.window_s * 1000;
      const t0 = nowMs - windowMs;
      const share = wlEffShare(this.entryFor(p.sym), this.settings.spread_share);

      let lo = Infinity, hi = -Infinity;
      for (const r of p.trades) { if (r[1] < lo) lo = r[1]; if (r[1] > hi) hi = r[1]; }
      const tradeRange = (hi > lo) ? (hi - lo) : 0;
      const mid = p.mid > 0 ? p.mid
                            : (p.trades.length ? p.trades[p.trades.length - 1][1] : 0);
      const sc0 = wlScale(mid, p.tp, share, tradeRange);
      if (!sc0) {
        // Held, subscribed, and nothing has arrived yet. An empty box is
        // indistinguishable from a pane that is broken, so it says so with
        // the same line a quote-less pane uses.
        this.drawWaiting(ctx, w, h);
        return;
      }
      p.centre = wlCentre(p.centre, mid, sc0.range);
      const sc = wlScale(p.centre, p.tp, share, tradeRange);

      const x = (t) => ((t - t0) / windowMs) * w;
      const y = (price) => h - ((price - sc.lo) / sc.range) * h;

      // ── the quote band ──────────────────────────────────────────────
      // A STEP, not a line between samples: a quote holds until it changes,
      // and interpolating draws a market sliding smoothly between two prices
      // it was never at.
      const q = p.quotes;
      const pts = [];
      for (let i = 0; i < q.length; i++) {
        const from = x(q[i][0]);
        const to = (i + 1 < q.length) ? Math.min(x(q[i + 1][0]), w) : w;
        if (to < 0 || from > w) continue;
        pts.push([Math.max(from, 0), Math.min(to, w), q[i][1], q[i][2]]);
      }
      // NOTHING BETWEEN THE TWO LINES. The tape page draws the spread as two
      // lines and no fill, and a shaded band here made the wall read as
      // something else entirely at a glance across a hundred panes.

      // ── the prints ──────────────────────────────────────────────────
      // Neutral and translucent, as on the tape page: density reads as
      // shade, and a colour would be claiming something about who crossed.
      ctx.fillStyle = TAPE_TRADE_FILL;
      ctx.strokeStyle = TAPE_TRADE_RIM;
      ctx.lineWidth = 0.9;
      for (const r of p.trades) {
        const px = x(r[0]);
        if (px < -4 || px > w + 4) continue;
        const py = y(r[1]);
        if (py < -6 || py > h + 6) continue;
        const rad = wlBubbleR(r[2]);
        ctx.beginPath();
        ctx.arc(px, py, rad, 0, 6.2832);
        ctx.fill();
        // Only discs big enough to have one, the same rule the tape uses:
        // a rim on a 1px dot is just a thicker dot.
        if (rad > 2) ctx.stroke();
      }

      // ── bid and ask, on top ─────────────────────────────────────────
      // BLUE BID, PINK ASK, from the shared file.
      if (pts.length) {
        ctx.lineWidth = 1;
        for (const [col, idx] of [[TAPE_BID, 2], [TAPE_ASK, 3]]) {
          ctx.beginPath();
          for (const seg of pts) {
            ctx.moveTo(seg[0], y(seg[idx]));
            ctx.lineTo(seg[1], y(seg[idx]));
          }
          ctx.strokeStyle = col;
          ctx.stroke();
        }
      }

      // A pane scaled off its trades has no spread to scale to, and must not
      // look like one that does.
      if (sc.from !== 'spread') this.drawWaiting(ctx, w, h);
    },

    drawWaiting(ctx, w, h) {
      ctx.strokeStyle = 'rgba(255,255,255,0.10)';
      ctx.setLineDash([3, 3]);
      ctx.beginPath();
      ctx.moveTo(0, h / 2);
      ctx.lineTo(w, h / 2);
      ctx.stroke();
      ctx.setLineDash([]);
    },

    // ── the watchlist ───────────────────────────────────────────────────
    async save(entries, settings) {
      this.busy = true;
      try {
        const r = await fetch('/wall/watchlist', {
          method: 'POST',
          headers: { 'content-type': 'application/json' },
          body: JSON.stringify({ entries: entries || this.entries,
                                 settings: settings || this.settings }),
        });
        const d = await r.json();
        this.takeState(d);
        // REFUSALS ARE SHOWN. A list that comes back quietly shorter than
        // what was typed is the failure mode of every "paste your symbols"
        // box ever written.
        this.warning = (d.refused && d.refused.length)
          ? d.refused.join(' · ') : '';
      } catch (e) {
        this.warning = 'could not save the watchlist: ' + e;
      } finally {
        this.busy = false;
      }
    },

    saveSoon() {
      // A slider fires on every pixel; the watchlist file does not need 40
      // writes to learn the number stopped moving.
      if (this.saveTimer) clearTimeout(this.saveTimer);
      this.saveTimer = setTimeout(() => this.save(), 500);
    },

    applyManual() {
      const syms = wlParseSymbols(this.manualText);
      // OVERRIDES SURVIVE AN EDIT OF THE LIST. Retyping the tickers is not a
      // statement about LLY's scale, and losing it there would mean
      // re-making the judgement every time a name is added.
      const had = {};
      for (const e of this.entries) had[e.symbol] = e.scale;
      this.save(syms.map(s => ({ symbol: s, scale: had[s] != null ? had[s] : null })));
      this.showManual = false;
    },

    fillManualFromHeld() {
      this.manualText = this.entries.map(e => e.symbol).join(' ');
    },

    addSymbol(text) {
      const syms = wlParseSymbols(text);
      if (!syms.length) return;
      const have = {};
      for (const e of this.entries) have[e.symbol] = 1;
      const next = this.entries.concat(
        syms.filter(s => !have[s]).map(s => ({ symbol: s, scale: null })));
      this.save(next);
    },

    remove(sym) {
      if (this.selected === sym) this.selected = '';
      delete WL_DATA.panes[sym];
      delete this.head[sym];
      this.save(this.entries.filter(e => e.symbol !== sym));
    },

    select(sym) {
      this.selected = (this.selected === sym) ? '' : sym;
    },

    // ── the controls ────────────────────────────────────────────────────
    onWindow() {
      // The SERVER is told too: the frame it builds is cut to the window, and
      // widening it re-sends the whole thing so the new axis fills at once
      // instead of growing into itself over two minutes.
      if (this.sock && this.sock.readyState === 1) {
        this.sock.send(JSON.stringify({ action: 'window',
                                        window_s: this.settings.window_s }));
      }
      this.dirtyAll();
      this.saveSoon();
    },

    onShare() {
      this.dirtyAll();
      this.saveSoon();
    },

    /* MOVING THE CONTROL APPLIES AT ONCE. The ten-second hold exists to stop
     * a name on the boundary strobing; it is not there to make the slider
     * feel broken for ten seconds, which is what waiting would look like.
     * So a threshold change back-dates the timer — every pane below is
     * already past its hold — and crossings from then on are held normally. */
    onSpreadFilter() {
      const nowMs = Date.now() + this.skewMs;
      for (const p of Object.values(WL_DATA.panes)) {
        p.belowSince = nowMs - WL_DIM_AFTER_MS;
      }
      this.refreshDim(nowMs);
      this.saveSoon();
    },

    onPaneWidth() {
      try { localStorage.setItem('equitiesWall.paneWidth', this.paneWidth); }
      catch (e) { /* private window */ }
      this.dirtyAll();
    },

    overrideOf(sym) {
      const e = this.entryFor(sym);
      return e && e.scale != null ? e.scale : this.settings.spread_share;
    },

    hasOverride(sym) {
      const e = this.entryFor(sym);
      return !!(e && e.scale != null);
    },

    setOverride(v) {
      const e = this.entryFor(this.selected);
      if (!e) return;
      e.scale = wlClampShare(v);
      this.dirtyAll();
      this.saveSoon();
    },

    clearOverride() {
      const e = this.entryFor(this.selected);
      if (!e) return;
      e.scale = null;
      this.dirtyAll();
      this.save();
    },

    // ── readouts ────────────────────────────────────────────────────────
    headOf(sym) {
      return this.head[sym] || { sp: '—', px: '', from: '', dim: false };
    },

    paneTitle(sym) {
      const h = this.headOf(sym);
      const share = wlEffShare(this.entryFor(sym), this.settings.spread_share);
      const scaled = h.from === 'spread'
        ? `the spread fills ${Math.round(share * 100)}% of the pane`
        : 'no quote yet — scaled to its own prints';
      // A faded pane says WHY, and says that it is still running: the
      // alternative reading of a faded pane is "this one has stopped".
      const faded = h.dim
        ? `; faded — its typical spread is under ${this.spreadFloorLabel()}, `
          + 'still subscribed and still accumulating'
        : '';
      return `${sym}: spread ${h.sp}, ${scaled}${faded}`;
    },

    spreadFloorLabel() {
      const c = this.settings.min_spread_cents;
      return (c > 0) ? (c < 10 ? c.toFixed(1) : c.toFixed(0)) + 'c' : 'off';
    },

    gridStyle() {
      return `grid-template-columns: repeat(auto-fill, minmax(${this.paneWidth}px, 1fr))`;
    },

    sharePct() { return Math.round(this.settings.spread_share * 100) + '%'; },

    connSummary() {
      if (!this.connected) return 'disconnected';
      const age = this.lastTick ? (Date.now() - this.lastTick) / 1000 : null;
      const held = this.entries.length;
      const stale = (age != null && age > 4) ? ` · last frame ${age.toFixed(0)}s ago` : '';
      return `${held} symbol${held === 1 ? '' : 's'} · ${this.drawn} drawn`
             + (this.offscreen ? ` · ${this.offscreen} off screen` : '')
             // Named, because a faded pane is still subscribed and still
             // accumulating — the count says how much of the wall the filter
             // is currently setting aside, not how much has been dropped.
             + (this.dimmed ? ` · ${this.dimmed} under ${this.spreadFloorLabel()}` : '')
             + ' · one connection' + stale;
    },

    emptyMessage() {
      return this.entries.length
        ? 'waiting for the first frame…'
        : 'No symbols yet. Edit list, paste some tickers, Apply.';
    },
  }));
});
