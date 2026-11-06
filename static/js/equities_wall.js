/* Equities Wall — the whole watchlist as a rotation list.
 *
 * WHAT IT IS FOR (2026-11-06). The watchlist is swept in alphabetical order
 * and whatever currently clears a minimum spread gets worked. So each symbol
 * is ONE ROW — ticker | spread bar | spread in cents — and nothing else: no
 * price, no prints, no volume. It used to be a grid of small tapes, a hundred
 * to a page; nobody was reading the prints, so they went, and the server tier
 * now subscribes quotes only.
 *
 * TOP TO BOTTOM, THEN THE NEXT COLUMN. The list is read beside the DAS
 * watchlist, which runs down. CSS multi-column (`column-width`) does that and
 * stays responsive; a CSS grid fills ACROSS rows and would put B next to A.
 *
 * TWO THRESHOLDS, both presentation only, saved with the list:
 *   floor — below it a row is dimmed IN PLACE (it keeps its position and
 *           keeps streaming), unless "hide below floor" is ticked;
 *   wide  — at or above it the bar is pink; between the two it is blue. The
 *           bars are scaled to `wide`, so moving it re-scales every bar.
 *
 * MARKS are a click: green, "this one made money today". In memory only — no
 * table, no endpoint, no local storage — and a marked row is never dimmed or
 * hidden.
 *
 * COST. One browser connection for the page (the service caps browsers at
 * 8). The server sends, four times a second, only the symbols whose spread
 * changed, and only those rows' bindings re-run.
 */
'use strict';

// Widest-first re-sorts this often, not on every frame. Four re-orders a
// second is a list that never holds still long enough to read; two seconds
// is still fresh for a sweep. Alphabetical never moves.
const WL_RESORT_MS = 2000;


/* Which colour a spread gets. Below the floor wins over wide, so a floor set
 * above `wide` still dims what is under it. A symbol that has not quoted has
 * nothing to clear the floor with, so it is 'none' and dims with the rest. */
function wlTier(sp, floor, wide) {
  if (sp == null || !isFinite(sp)) return 'none';
  if (sp < floor) return 'below';
  if (sp >= wide) return 'wide';
  return 'mid';
}


/* Bar length as a percentage of its track, scaled to the WIDE threshold:
 * at `wide` and above the bar is full. */
function wlBarPct(sp, wide) {
  if (sp == null || !isFinite(sp) || !(wide > 0)) return 0;
  return Math.max(0, Math.min(100, (sp / wide) * 100));
}


/* The display order. Alphabetical, or widest spread first with the names
 * that have not quoted last; ties alphabetical, so equal spreads do not
 * trade places from one re-sort to the next. */
function wlOrder(symbols, spreads, widest) {
  const alpha = symbols.slice().sort();
  if (!widest) return alpha;
  const v = (s) => {
    const x = spreads[s];
    return (x == null || !isFinite(x)) ? -Infinity : x;
  };
  return alpha.sort((a, b) => (v(b) - v(a)) || (a < b ? -1 : a > b ? 1 : 0));
}


/* Is the row on screen at all. Only "hide below floor" removes one, and
 * never a marked one. */
function wlShown(tier, hideBelow, marked) {
  if (marked || !hideBelow) return true;
  return tier === 'mid' || tier === 'wide';
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
  return cents < 10 ? cents.toFixed(1) : cents.toFixed(0);
}


document.addEventListener('alpine:init', () => {
  Alpine.data('equitiesWall', () => ({

    // ── state ───────────────────────────────────────────────────────────
    entries: [],              // [{symbol, scale}] — the server's list, as held
    settings: { min_spread_cents: 59, wide_cents: 120 },
    caps: { symbols: 300, tick_s: 0.25, floor_min: 10, floor_max: 120,
            wide_min: 40, wide_max: 300, wide_step: 5 },
    sp: {},                   // sym -> spread in cents, or null
    marks: {},                // sym -> true; this page session only
    order: [],                // symbols in display order
    widest: false,            // the sort toggle; alphabetical on load
    hideBelow: false,         // off on load: rows hold their place
    sortedAt: 0,

    connected: false,
    warning: '',
    busy: false,
    lastTick: null,
    sock: null,
    showManual: false,
    manualText: '',
    saveTimer: null,

    // ── lifecycle ───────────────────────────────────────────────────────
    init() {
      // The bars wear the TAPE's blue and pink (tape_theme.js, which Equities
      // Live reads too) rather than a second copy of either colour.
      const root = document.documentElement.style;
      root.setProperty('--wl-mid', TAPE_BLUE);
      root.setProperty('--wl-wide', TAPE_PINK);
      this.connect();
      // The connection line's "last frame Ns ago" needs a clock that runs
      // when frames do not.
      setInterval(() => { this.clock = Date.now(); }, 1000);
    },
    clock: 0,

    connect() {
      const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
      const sock = new WebSocket(proto + '//' + location.host + '/wall/ws');
      this.sock = sock;
      sock.onopen = () => { this.connected = true; this.warning = ''; };
      sock.onclose = () => {
        this.connected = false;
        // Said out loud. A list that quietly stops updating looks exactly
        // like spreads that stopped moving.
        this.warning = 'disconnected — retrying';
        setTimeout(() => this.connect(), 2000);
      };
      sock.onmessage = (e) => this.onMessage(JSON.parse(e.data));
    },

    onMessage(m) {
      if (m.ev === 'hello' || m.ev === 'watchlist') this.takeState(m);
      else if (m.ev === 'tick') this.onTick(m);
    },

    takeState(m) {
      if (m.entries) this.entries = m.entries;
      if (m.settings) this.settings = Object.assign({}, this.settings, m.settings);
      if (m.caps) this.caps = Object.assign({}, this.caps, m.caps);
      if (m.error) this.warning = 'watchlist: ' + m.error;
      this.resort();
    },

    onTick(m) {
      this.lastTick = Date.now();
      for (const [sym, v] of Object.entries(m.syms || {})) this.sp[sym] = v;
      if (this.widest && Date.now() - this.sortedAt >= WL_RESORT_MS) this.resort();
    },

    resort() {
      const next = wlOrder(this.entries.map(e => e.symbol), this.sp, this.widest);
      // Assigned only when it changed: a new array re-runs the x-for's diff
      // over every row.
      if (next.length !== this.order.length || next.some((s, i) => s !== this.order[i])) {
        this.order = next;
      }
      this.sortedAt = Date.now();
    },

    // ── rows ────────────────────────────────────────────────────────────
    tierOf(sym) {
      return wlTier(this.sp[sym], this.settings.min_spread_cents,
                    this.settings.wide_cents);
    },

    rowClass(sym) {
      const tier = this.tierOf(sym);
      const marked = !!this.marks[sym];
      return tier + (marked ? ' marked' : '')
        + (wlShown(tier, this.hideBelow, marked) ? '' : ' gone');
    },

    barStyle(sym) {
      return 'width:' + wlBarPct(this.sp[sym], this.settings.wide_cents).toFixed(1) + '%';
    },

    spText(sym) { return wlFmtSpread(this.sp[sym]); },

    toggleMark(sym) {
      if (this.marks[sym]) delete this.marks[sym];
      else this.marks[sym] = true;
    },

    toggleSort() {
      this.widest = !this.widest;
      this.resort();
    },

    // ── the watchlist ───────────────────────────────────────────────────
    async save(entries) {
      this.busy = true;
      try {
        const r = await fetch('/wall/watchlist', {
          method: 'POST',
          headers: { 'content-type': 'application/json' },
          body: JSON.stringify({ entries: entries || this.entries,
                                 settings: this.settings }),
        });
        const d = await r.json();
        this.takeState(d);
        // REFUSALS ARE SHOWN. A list that comes back quietly shorter than
        // what was typed is the failure of every "paste your symbols" box.
        this.warning = (d.refused && d.refused.length)
          ? d.refused.join(' · ') : '';
      } catch (e) {
        this.warning = 'could not save the watchlist: ' + e;
      } finally {
        this.busy = false;
      }
    },

    saveSoon() {
      // A slider fires on every pixel; the file does not need 40 writes to
      // learn the number stopped moving.
      if (this.saveTimer) clearTimeout(this.saveTimer);
      this.saveTimer = setTimeout(() => this.save(), 500);
    },

    applyManual() {
      // An entry's stored fields ride along for a symbol that stays.
      const had = {};
      for (const e of this.entries) had[e.symbol] = e;
      this.save(wlParseSymbols(this.manualText)
        .map(s => had[s] || { symbol: s, scale: null }));
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
      this.save(this.entries.concat(
        syms.filter(s => !have[s]).map(s => ({ symbol: s, scale: null }))));
    },

    // ── readouts ────────────────────────────────────────────────────────
    cents(v) { return Math.round(v) + '¢'; },

    counts() {
      const fl = this.settings.min_spread_cents, wd = this.settings.wide_cents;
      let ok = 0, wide = 0;
      for (const e of this.entries) {
        const t = wlTier(this.sp[e.symbol], fl, wd);
        if (t === 'mid' || t === 'wide') ok++;
        if (t === 'wide') wide++;
      }
      return { ok, wide, marked: Object.keys(this.marks).length };
    },

    connSummary() {
      if (!this.connected) return 'disconnected';
      void this.clock;
      const age = this.lastTick ? (Date.now() - this.lastTick) / 1000 : null;
      const c = this.counts();
      const stale = (age != null && age > 4) ? ` · last frame ${age.toFixed(0)}s ago` : '';
      return `${this.entries.length} symbols · ${c.ok} ≥ floor · ${c.wide} wide`
             + (c.marked ? ` · ${c.marked} marked` : '') + ' · one connection' + stale;
    },

    emptyMessage() {
      return 'No symbols yet. Edit list, paste some tickers, Apply.';
    },
  }));
});
