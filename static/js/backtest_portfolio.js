/* Backtest Portfolio — several saved strategies, combined.
 *
 * P1 CARRIES DATA AND COMPUTES NOTHING. Strategies are chosen, loaded, and
 * their trades held in the browser with their colour, quantity and capital.
 * The stats, the curves and the correlation work are P2-P4, and they will be
 * computed HERE rather than on the server, the same way the single-backtest
 * page does it: the trades go over once and every later change is local.
 *
 * WHAT IS ALREADY DECIDED, so P2 does not re-open it:
 *   qty scales P/L linearly, applied in the browser.
 *   capital seeds from the strategy's saved value and can be overridden
 *     here; portfolio capital is the SUM of the per-strategy figures.
 *   the date range defaults to the UNION of the loaded spans.
 *   P/L is dated by CLOSE, exactly as the single-backtest page dates it.
 *
 * THE TRADES ARE HELD OUTSIDE ALPINE. A portfolio of five strategies is
 * ~10,000 trades of ~30 columns; wrapping that in a reactive proxy costs on
 * every read for no benefit, because no template ever reads a trade. The
 * page's reactive state is the strategy LIST and the summary numbers. Same
 * arrangement as OB_DATA on the other page, and the same trap: anything
 * derived from BP_DATA must be recomputed into reactive state, never read
 * through a getter that Alpine cannot see change.
 */
'use strict';

const BP_DATA = { payloads: {} };      // strategy id -> the server's payload

function bpFmtInt(n) {
  return (n == null || !isFinite(n)) ? '—' : Math.round(n).toLocaleString();
}

function bpFmtMoney(v) {
  if (v == null || !isFinite(v)) return '—';
  const a = Math.abs(v);
  if (a >= 1e6) return '$' + (v / 1e6).toFixed(1) + 'M';
  if (a >= 1e3) return '$' + (v / 1e3).toFixed(0) + 'k';
  return '$' + v.toFixed(0);
}

function bpFmtSeconds(s) {
  if (s == null || !isFinite(s)) return '—';
  return s >= 10 ? s.toFixed(0) + 's' : s.toFixed(1) + 's';
}

/* The union or the intersection of the loaded spans.
 *
 * UNION IS THE DEFAULT (the brief). It uses every trade, at the cost of the
 * early stretch having fewer strategies in it — which is why the page states
 * the span and how many strategies cover it rather than leaving the curve to
 * imply that everything ran from the start. Dates are ISO strings and
 * compare as strings; the payload builder makes them that way on purpose. */
function bpSpan(payloads, mode) {
  const lows = [], highs = [];
  for (const p of payloads) {
    if (p && p.date_min) lows.push(p.date_min);
    if (p && p.date_max) highs.push(p.date_max);
  }
  if (!lows.length || !highs.length) return null;
  return (mode === 'intersection')
    ? { start: lows.reduce((a, b) => (a > b ? a : b)),
        end: highs.reduce((a, b) => (a < b ? a : b)) }
    : { start: lows.reduce((a, b) => (a < b ? a : b)),
        end: highs.reduce((a, b) => (a > b ? a : b)) };
}

/* Portfolio capital: the SUM of each strategy's planned capital, and each
 * strategy's is its per-position capital times its quantity — qty is a
 * multiplier on the position, so it multiplies the capital behind it as well
 * as the P/L it produces. */
function bpTotalCapital(rows) {
  let total = 0;
  for (const r of rows) {
    const cap = Number(r.capital), qty = Number(r.qty);
    if (isFinite(cap) && cap > 0 && isFinite(qty) && qty > 0) total += cap * qty;
  }
  return total;
}


document.addEventListener('alpine:init', () => {
  Alpine.data('backtestPortfolio', () => ({

    // ── state ───────────────────────────────────────────────────────────
    saved: [],            // everything saved on the other page
    colors: [],           // the palette, from the server
    maxStrategies: 12,
    chosen: [],           // [{id, name, qty, capital, savedCapital}]
    loaded: [],           // the server's payloads, in load order
    pick: 0,
    busy: false,
    error: '',
    loadNote: '',
    slowLoad: false,
    rangeMode: 'union',

    async init() {
      await this.loadSaved();
    },

    async loadSaved() {
      try {
        const r = await fetch('/api/backtest-portfolio/strategies');
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        const b = await r.json();
        this.saved = b.strategies || [];
        this.colors = b.colors || [];
        this.maxStrategies = b.max || 12;
      } catch (e) {
        this.error = 'Could not read the saved strategies: ' + e;
      }
    },

    // ── choosing ────────────────────────────────────────────────────────
    addable() {
      const have = new Set(this.chosen.map(c => c.id));
      return this.saved.filter(s => !have.has(s.id));
    },

    optionLabel(s) {
      const n = s.trade_count == null ? '' : ` · ${bpFmtInt(s.trade_count)} trades`;
      const d = s.date_min && s.date_max ? ` · ${s.date_min} → ${s.date_max}` : '';
      return `${s.name}${n}${d}`;
    },

    add() {
      const s = this.saved.find(x => x.id === this.pick);
      if (!s) return;
      if (this.chosen.length >= this.maxStrategies) {
        this.error = `${this.maxStrategies} strategies is the most this page `
                   + `loads at once.`;
        return;
      }
      this.chosen.push({
        id: s.id, name: s.name,
        qty: 1,
        // THE SAVED VALUE SEEDS THE ROW. A strategy saved with $25,000 a
        // position should not silently become $10,000 because this page has
        // a different default; where it saved nothing, the page's default
        // stands and the row says so.
        capital: s.capital_per_position != null ? s.capital_per_position : 10000,
        savedCapital: s.capital_per_position,
      });
      this.pick = 0;
      this.error = '';
    },

    remove(i) {
      const c = this.chosen[i];
      this.chosen.splice(i, 1);
      if (c) {
        delete BP_DATA.payloads[c.id];
        this.loaded = this.loaded.filter(p => p.saved.id !== c.id);
      }
    },

    clearAll() {
      this.chosen = [];
      this.loaded = [];
      BP_DATA.payloads = {};
      this.loadNote = '';
      this.error = '';
    },

    normalise(c) {
      const q = Math.round(Number(c.qty));
      c.qty = (isFinite(q) && q > 0) ? q : 1;
      const cap = Number(c.capital);
      c.capital = (isFinite(cap) && cap >= 0) ? cap : 0;
    },

    colorOf(i) {
      return this.colors.length ? this.colors[i % this.colors.length] : '#3498db';
    },

    // ── loading ─────────────────────────────────────────────────────────
    async load() {
      if (!this.chosen.length || this.busy) return;
      this.busy = true;
      this.error = '';
      this.loadNote = '';
      const t0 = performance.now();
      try {
        const r = await fetch('/api/backtest-portfolio/load', {
          method: 'POST',
          headers: { 'content-type': 'application/json' },
          body: JSON.stringify({ ids: this.chosen.map(c => c.id) }),
        });
        const b = await r.json();
        if (!r.ok) throw new Error(b.detail || `HTTP ${r.status}`);
        this.loaded = b.strategies || [];
        BP_DATA.payloads = {};
        for (const p of this.loaded) BP_DATA.payloads[p.saved.id] = p;
        // The capital the row carries wins over the saved one on a reload:
        // it is what the person set on this page a moment ago.
        for (const c of this.chosen) {
          const p = BP_DATA.payloads[c.id];
          if (p && c.capital == null && p.capital_per_position != null) {
            c.capital = p.capital_per_position;
          }
        }
        const wall = (performance.now() - t0) / 1000;
        const srv = b.load || {};
        // BOTH NUMBERS. The server's time is the work; the wall time includes
        // sending several megabytes of trades to this browser, and when they
        // differ a lot it is the transfer that is slow, not the parse.
        this.loadNote = `${srv.n} loaded in ${bpFmtSeconds(srv.seconds)} `
                      + `(${srv.from_cache} cached, ${srv.parsed} parsed) · `
                      + `${bpFmtSeconds(wall)} in the browser`;
        this.slowLoad = (srv.parsed || 0) > 0;
      } catch (e) {
        this.error = 'Load failed: ' + (e.message || e);
        this.loaded = [];
      } finally {
        this.busy = false;
      }
    },

    // ── readouts ────────────────────────────────────────────────────────
    savedSummary() {
      return this.saved.length
        ? `${this.saved.length} saved · ${this.chosen.length} chosen`
        : 'none saved yet';
    },

    allocSummary() {
      return `${bpFmtMoney(bpTotalCapital(this.chosen))} total`;
    },

    rowMeta(c) {
      const s = this.saved.find(x => x.id === c.id);
      const n = s ? bpFmtInt(s.trade_count) : '—';
      const seeded = c.savedCapital == null ? ' · capital not saved' : '';
      return `${n} trades${seeded}`;
    },

    loadedSummary() {
      const trades = this.loaded.reduce((a, p) => a + (p.n || 0), 0);
      const span = bpSpan(this.loaded, this.rangeMode);
      const s = span ? ` · ${span.start} → ${span.end}` : '';
      return `${this.loaded.length} strategies · ${bpFmtInt(trades)} trades${s}`;
    },

    statusLine() {
      if (!this.chosen.length) return 'Add saved strategies to build a portfolio.';
      if (!this.loaded.length) return `${this.chosen.length} chosen — not loaded yet.`;
      const span = bpSpan(this.loaded, this.rangeMode);
      return `${this.loaded.length} loaded · `
           + `${bpFmtMoney(bpTotalCapital(this.chosen))} capital · `
           + (span ? `${span.start} → ${span.end} (${this.rangeMode})` : 'no dates');
    },

    spanOf(p) {
      return (p.date_min && p.date_max) ? `${p.date_min} → ${p.date_max}` : '—';
    },

    parseOf(p) {
      const q = p.parse || {};
      return `${q.source === 'cache' ? 'cached' : 'parsed'} ${bpFmtSeconds(q.seconds)}`;
    },

    /* Anything the load found that the page should not swallow. A market
     * join that failed, or a trade count that has moved since the strategy
     * was saved, changes what every later number means. */
    warnings() {
      const out = [];
      for (const p of this.loaded) {
        const name = p.saved ? p.saved.name : p.filename;
        if (p.saved_count_changed) {
          out.push(`${name}: ${bpFmtInt(p.saved_count_changed.when_saved)} trades `
                 + `when saved, ${bpFmtInt(p.saved_count_changed.now)} now — the `
                 + `parser has changed since.`);
        }
        const mk = p.market || {};
        if (mk.joined === false) {
          out.push(`${name}: market data did not join (${mk.error || 'no reason given'}) `
                 + `— metric filters will have nothing to filter on.`);
        }
      }
      return out;
    },
  }));
});
