/* Backtest Portfolio — several saved strategies, combined.
 *
 * WHAT IT COMPUTES AND WHERE. Every number on this page comes from
 * backtest_core.js, the file /oo-backtest reads too: obApplyFilters,
 * obStats, obExtraStats, obConcurrency, obSharpe. That is the point of the
 * page — a strategy has to read the same here as it does there — and it is
 * why there is no second implementation of a statistic in this file.
 *
 * SETTLED, so nothing below re-opens it:
 *   qty scales P/L linearly, applied HERE, in the browser, and nowhere else.
 *   capital seeds from the strategy's saved value, is overridable, and the
 *     portfolio's capital is the SUM of capital x qty.
 *   the date range defaults to the UNION of the loaded spans.
 *   P/L is dated by CLOSE, exactly as the single-backtest page dates it.
 *   deployment is half-open [open, close): overnight capital only.
 *
 * THE TRADES ARE HELD OUTSIDE ALPINE (BP_DATA). A five-strategy portfolio is
 * ~10,000 trades of ~30 columns; a reactive proxy over that costs on every
 * read and buys nothing, because no template reads a trade. The reactive
 * state is the strategy list, the filters and the computed rows. The trap
 * that comes with it: anything derived from BP_DATA must be recomputed INTO
 * reactive state, never read through a getter Alpine cannot see change.
 */
'use strict';

const BP_DATA = {
  payloads: {},      // strategy id -> the server's payload
  scaled: {},        // strategy id -> { qty, cols } with pnl x qty
  extents: {},       // "id:metric" -> the observed range, for the sliders
};

/* A bound snapped outward to the registry's step, so a slider's ends are
 * round numbers rather than whatever the extreme trade happened to be. */
function obClampStep(v, step, how) {
  const s = Number(step);
  if (!isFinite(s) || s <= 0) return v;
  const k = v / s;
  return Number(((how === 'floor' ? Math.floor(k) : Math.ceil(k)) * s)
                .toPrecision(12));
}

function bpFmtInt(n) {
  return (n == null || !isFinite(n)) ? '—' : Math.round(n).toLocaleString();
}

function bpFmtMoney(v) {
  if (v == null || !isFinite(v)) return '—';
  const a = Math.abs(v);
  if (a >= 1e6) return (v < 0 ? '-$' : '$') + (a / 1e6).toFixed(1) + 'M';
  if (a >= 1e4) return (v < 0 ? '-$' : '$') + (a / 1e3).toFixed(0) + 'k';
  return (v < 0 ? '-$' : '$') + Math.round(a).toLocaleString();
}

function bpFmtNum(v, dp = 2) {
  return (v == null || !isFinite(v)) ? '—' : v.toFixed(dp);
}

function bpFmtPct(v, dp = 1) {
  return (v == null || !isFinite(v)) ? '—' : v.toFixed(dp) + '%';
}

function bpFmtSeconds(s) {
  if (s == null || !isFinite(s)) return '—';
  return s >= 10 ? s.toFixed(0) + 's' : s.toFixed(1) + 's';
}

/* The union or the intersection of the loaded spans. UNION IS THE DEFAULT:
 * it uses every trade, at the cost of the early stretch having fewer
 * strategies in it — which the page states rather than letting the curve
 * imply everything ran from the start. Dates are ISO strings and compare as
 * strings; the payload builder makes them that way on purpose. */
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
 * strategy's is capital-per-position x qty — qty multiplies the position, so
 * it multiplies the capital behind it as well as the P/L it produces. */
function bpTotalCapital(rows) {
  let total = 0;
  for (const r of rows) {
    const cap = Number(r.capital), qty = Number(r.qty);
    if (isFinite(cap) && cap > 0 && isFinite(qty) && qty > 0) total += cap * qty;
  }
  return total;
}

/* The filter specs for one strategy, in the shape obApplyFilters takes.
 * Built from the SHARED REGISTRY — the old Dash app kept five parallel dicts
 * of metrics and this page keeps none. A filter that is off contributes
 * nothing, so an untouched page filters nothing. */
function bpSpecs(filters, registry, dateSpan) {
  const specs = [];
  for (const m of registry) {
    const f = filters[m.key];
    if (!f || !f.on) continue;
    if (m.type === 'range') {
      const lo = Number(f.lo), hi = Number(f.hi);
      if (!isFinite(lo) || !isFinite(hi)) continue;
      specs.push({ kind: 'range', column: m.column, lo, hi });
    } else if (m.type === 'categorical') {
      if (!f.allowed || !f.allowed.length) continue;
      specs.push({ kind: 'set', column: m.column, allowed: new Set(f.allowed) });
    }
  }
  // THE PORTFOLIO'S DATE RANGE, on the ENTRY date, as the old app had it: a
  // trade belongs to the window it was opened in.
  if (dateSpan && (dateSpan.start || dateSpan.end)) {
    specs.push({ kind: 'date', column: 'date_opened',
                 from: dateSpan.start, to: dateSpan.end });
  }
  return specs;
}

/* What an active filter costs in trades that have no value for it.
 * Metrics have staggered coverage — a filter on one that starts late drops
 * every earlier trade silently — so the page states it, as the single-
 * backtest page does. */
function bpCoverageCost(cols, n, filters, registry) {
  let noValue = 0;
  const cols_ = [];
  for (const m of registry) {
    const f = filters[m.key];
    if (f && f.on && cols[m.column]) cols_.push(m.column);
  }
  if (!cols_.length) return { noValue: 0, columns: [] };
  for (let i = 0; i < n; i++) {
    for (const c of cols_) {
      const v = cols[c][i];
      if (v === null || v === undefined || (typeof v === 'number' && Number.isNaN(v))) {
        noValue++;
        break;
      }
    }
  }
  return { noValue, columns: cols_ };
}


document.addEventListener('alpine:init', () => {
  Alpine.data('backtestPortfolio', () => ({

    // ── state ───────────────────────────────────────────────────────────
    saved: [],
    registry: [],
    colors: [],
    maxStrategies: 12,
    chosen: [],           // [{id, name, qty, capital, savedCapital, filters}]
    loaded: [],
    rows: [],             // the summary table: one per strategy, then TOTAL
    pick: 0,
    editing: 0,           // which strategy's filters are open
    busy: false,
    error: '',
    loadNote: '',
    slowLoad: false,
    rangeMode: 'union',
    tick: 0,              // bumped whenever the rows are recomputed

    async init() {
      await Promise.all([this.loadSaved(), this.loadRegistry()]);
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

    async loadRegistry() {
      try {
        const r = await fetch('/api/backtest-portfolio/registry');
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        const b = await r.json();
        // THE SAME REGISTRY THE OTHER PAGE FILTERS ON, filtered to what can
        // be screened. `year` is a section, not a filter.
        this.registry = (b.metrics || []).filter(m => m.filter);
      } catch (e) {
        this.error = 'Could not read the metric registry: ' + e;
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
        this.error = `${this.maxStrategies} strategies is the most this page loads at once.`;
        return;
      }
      this.chosen.push({
        id: s.id, name: s.name, qty: 1,
        // The saved value seeds the row: a strategy saved with $25,000 a
        // position must not silently become this page's default.
        capital: s.capital_per_position != null ? s.capital_per_position : 10000,
        savedCapital: s.capital_per_position,
        filters: this.blankFilters(),
      });
      this.pick = 0;
      this.error = '';
    },

    blankFilters() {
      const out = {};
      for (const m of this.registry) {
        out[m.key] = (m.type === 'range')
          ? { on: false, lo: m.min, hi: m.max }
          : { on: false, allowed: [] };
      }
      return out;
    },

    remove(i) {
      const c = this.chosen[i];
      this.chosen.splice(i, 1);
      if (c) {
        delete BP_DATA.payloads[c.id];
        delete BP_DATA.scaled[c.id];
        this.loaded = this.loaded.filter(p => p.saved.id !== c.id);
        if (this.editing === c.id) this.editing = 0;
      }
      this.recompute();
    },

    clearAll() {
      this.chosen = [];
      this.loaded = [];
      this.rows = [];
      BP_DATA.payloads = {};
      BP_DATA.scaled = {};
      BP_DATA.extents = {};
      this.loadNote = '';
      this.error = '';
      this.editing = 0;
    },

    normalise(c) {
      const q = Math.round(Number(c.qty));
      c.qty = (isFinite(q) && q > 0) ? q : 1;
      const cap = Number(c.capital);
      c.capital = (isFinite(cap) && cap >= 0) ? cap : 0;
      this.recompute();
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
          method: 'POST', headers: { 'content-type': 'application/json' },
          body: JSON.stringify({ ids: this.chosen.map(c => c.id) }),
        });
        const b = await r.json();
        if (!r.ok) throw new Error(b.detail || `HTTP ${r.status}`);
        this.loaded = b.strategies || [];
        BP_DATA.payloads = {};
        BP_DATA.scaled = {};
        BP_DATA.extents = {};
        for (const p of this.loaded) BP_DATA.payloads[p.saved.id] = p;
        const wall = (performance.now() - t0) / 1000;
        const srv = b.load || {};
        // BOTH NUMBERS: the server's time is the work, the wall time includes
        // sending several megabytes of trades here. When they differ a lot it
        // is the transfer that is slow, not the parse.
        this.loadNote = `${srv.n} loaded in ${bpFmtSeconds(srv.seconds)} `
                      + `(${srv.from_cache} cached, ${srv.parsed} parsed) · `
                      + `${bpFmtSeconds(wall)} in the browser`;
        this.slowLoad = (srv.parsed || 0) > 0;
        this.recompute();
      } catch (e) {
        this.error = 'Load failed: ' + (e.message || e);
        this.loaded = [];
        this.rows = [];
      } finally {
        this.busy = false;
      }
    },

    // ── filters, in the main column ─────────────────────────────────────
    //
    // ONE STRATEGY AT A TIME. Every strategy's panel open at once would be
    // nine metrics times however many strategies on a page whose job is the
    // comparison between them; the sidebar row says which is being edited
    // and how many filters it carries.
    toggleEdit(id) { this.editing = (this.editing === id ? 0 : id); },

    editingRow() { return this.chosen.find(c => c.id === this.editing) || null; },

    editingName() {
      const c = this.editingRow();
      return c ? c.name : '';
    },

    editingColor() {
      const i = this.chosen.findIndex(c => c.id === this.editing);
      return i < 0 ? 'transparent' : this.colorOf(i);
    },

    editingSummary() {
      const c = this.editingRow();
      if (!c) return '';
      const r = this.rows.find(x => x.id === c.id);
      return r ? `${bpFmtInt(r.n)} of ${bpFmtInt(r.nAll)} trades` : '';
    },

    editingCost() {
      const c = this.editingRow();
      const r = c && this.rows.find(x => x.id === c.id);
      if (!r || !r.cost.noValue) return '';
      return `${bpFmtInt(r.cost.noValue)} trades have no value for an active `
           + `filter and are dropped by it — metrics start at different dates.`;
    },

    isOn(m) {
      const c = this.editingRow();
      return !!(c && c.filters[m.key] && c.filters[m.key].on);
    },

    setOn(m, on) {
      const c = this.editingRow();
      if (!c) return;
      c.filters[m.key].on = !!on;
      this.recompute();
    },

    stateOf(m) {
      const c = this.editingRow();
      if (!c) return '';
      const f = c.filters[m.key];
      if (!f || !f.on) return 'off';
      if (m.type === 'categorical') {
        const n = (f.allowed || []).length;
        return n ? `${n} kept` : 'none kept';
      }
      const r = this.rangeOf(m);
      return r ? `${this.fmtVal(m, r.lo)} – ${this.fmtVal(m, r.hi)}` : 'no values';
    },

    /* The slider's bounds come from THIS STRATEGY'S OWN VALUES, not from the
     * registry's nominal range: a VIX slider spanning 9–80 when the log only
     * ever saw 12–31 is a control whose useful travel is a third of its
     * length. The registry supplies the step and the formatting. */
    rangeOf(m) {
      const c = this.editingRow();
      if (!c) return null;
      const ext = this.extentOf(c, m);
      if (!ext) return null;
      const f = c.filters[m.key];
      const lo = (f && isFinite(f.lo)) ? f.lo : ext.min;
      const hi = (f && isFinite(f.hi)) ? f.hi : ext.max;
      return { min: ext.min, max: ext.max, step: m.step || 0.01,
               lo: Math.max(ext.min, Math.min(lo, ext.max)),
               hi: Math.max(ext.min, Math.min(hi, ext.max)), n: ext.n };
    },

    extentOf(c, m) {
      const p = BP_DATA.payloads[c.id];
      if (!p) return null;
      const key = c.id + ':' + m.key;
      if (!(key in BP_DATA.extents)) {
        const ext = obExtent(p.columns[m.column] || []);
        BP_DATA.extents[key] = ext && { min: obClampStep(ext.min, m.step, 'floor'),
                                        max: obClampStep(ext.max, m.step, 'ceil'),
                                        n: ext.n };
      }
      return BP_DATA.extents[key];
    },

    setLo(m, v) {
      const c = this.editingRow(), r = this.rangeOf(m);
      if (!c || !r) return;
      const f = c.filters[m.key];
      f.lo = Math.min(Number(v), r.hi);
      f.on = true;
      this.recompute();
    },

    setHi(m, v) {
      const c = this.editingRow(), r = this.rangeOf(m);
      if (!c || !r) return;
      const f = c.filters[m.key];
      f.hi = Math.max(Number(v), r.lo);
      f.on = true;
      this.recompute();
    },

    fmtVal(m, v) {
      void this.tick;
      if (v == null || !isFinite(v)) return '—';
      if (m.format === 'usd') return bpFmtMoney(v);
      if (m.format === 'pct') return v.toFixed(2) + '%';
      if (m.format === 'ratio') return v.toFixed(2);
      if (m.format === 'int') return String(Math.round(v));
      return v.toFixed(2);
    },

    rangeSummary(m) {
      const r = this.rangeOf(m);
      if (!r) return '';
      return `${bpFmtInt(r.n)} trades have a value · `
           + `${this.fmtVal(m, r.min)} to ${this.fmtVal(m, r.max)} in this log`;
    },

    /* The categories a categorical metric actually has in THIS strategy's
     * trades — from the data, not from a list written here. `exit_reason`
     * has no declared categories because they are whatever the file says. */
    categoriesFor(c, m) {
      const p = BP_DATA.payloads[c.id];
      if (m.categories && m.categories.length) return m.categories;
      if (!p) return [];
      return obDistinct(p.columns[m.column] || [])
        .map(v => ({ value: v, label: String(v) }));
    },

    toggleCategory(c, m, value) {
      if (!c) return;
      const f = c.filters[m.key];
      const i = f.allowed.indexOf(value);
      if (i >= 0) f.allowed.splice(i, 1); else f.allowed.push(value);
      // TICKING THE FIRST BOX TURNS THE FILTER ON; clearing the last leaves
      // it on with nothing kept, which filters everything out. That is a
      // real state a person can reach deliberately, and `stateOf` says
      // "none kept" rather than pretending the filter is off.
      if (f.allowed.length) f.on = true;
      this.recompute();
    },

    isChosenCategory(c, m, value) {
      const f = c.filters[m.key];
      return !!(f && f.allowed.includes(value));
    },

    resetFilters(c) {
      c.filters = this.blankFilters();
      this.recompute();
    },

    activeCount(c) {
      let n = 0;
      for (const m of this.registry) {
        const f = c.filters[m.key];
        if (f && f.on) n++;
      }
      return n;
    },

    filterBadge(c) {
      const n = this.activeCount(c);
      return n ? `${n} filter${n === 1 ? '' : 's'}` : 'no filters';
    },

    // ── the numbers ─────────────────────────────────────────────────────
    span() { return bpSpan(this.loaded, this.rangeMode); },

    /* One strategy's columns with P/L scaled by qty. Cached per (id, qty):
     * the array is rebuilt when the quantity changes and not on every
     * recompute, which is every keystroke in a filter box. */
    scaledCols(c) {
      const p = BP_DATA.payloads[c.id];
      if (!p) return null;
      const hit = BP_DATA.scaled[c.id];
      if (hit && hit.qty === c.qty) return hit.cols;
      const qty = Number(c.qty) > 0 ? Number(c.qty) : 1;
      const cols = Object.assign({}, p.columns,
                                 { pnl: p.columns.pnl.map(v => v * qty) });
      BP_DATA.scaled[c.id] = { qty: c.qty, cols };
      return cols;
    },

    /* Everything the table shows. Recomputed in full on any change — the
     * whole portfolio is a few tens of thousands of trades and the work is
     * milliseconds, so there is no partial-update path to get wrong. */
    recompute() {
      const rows = [];
      const span = this.rangeMode === 'intersection' ? this.span() : null;
      // One session list for everyone: they all come from the same rollup,
      // so the union covers every strategy and the per-strategy deployed
      // series can be summed position by position.
      const sessions = [...new Set(this.loaded.flatMap(
        p => (p.market && p.market.spx_sessions) || []))].sort();

      const pooled = { date_opened: [], date_closed: [], pnl: [], days_in_trade: [] };
      const pctParts = [];          // per-trade P/L %, each against its own capital
      let deployed = new Array(sessions.length).fill(0);

      for (const c of this.chosen) {
        const p = BP_DATA.payloads[c.id];
        if (!p) continue;
        const cols = this.scaledCols(c);
        const specs = bpSpecs(c.filters, this.registry, span);
        const idx = obApplyFilters(cols, p.n, specs);
        const stats = obStats(cols, idx);
        const conc = obConcurrency(p.columns, idx, sessions);
        const capital = (Number(c.capital) || 0) * (Number(c.qty) || 1);
        const extra = obExtraStats(cols, idx, stats, capital, conc.peak);
        const series = obDeployedSeries(conc, sessions, capital);
        for (let i = 0; i < deployed.length; i++) deployed[i] += series[i];

        for (const i of idx) {
          pooled.date_opened.push(p.columns.date_opened[i]);
          pooled.date_closed.push(p.columns.date_closed[i]);
          pooled.pnl.push(cols.pnl[i]);
          pooled.days_in_trade.push(p.columns.days_in_trade[i]);
          if (capital > 0) pctParts.push(cols.pnl[i] / capital * 100);
        }

        rows.push({
          key: 'k' + c.id, id: c.id, name: c.name, color: p.color,
          total: false,
          n: idx.length, nAll: p.n,
          dropped: p.n - idx.length,
          cost: bpCoverageCost(p.columns, p.n, c.filters, this.registry),
          stats, extra, conc,
          sharpe: obSharpe(cols, idx),
          capital,
          peakDeployed: conc.peak * capital,
        });
      }

      // ── the TOTAL row ────────────────────────────────────────────────
      if (rows.length) {
        const all = [...pooled.pnl.keys()];
        const tStats = obStats(pooled, all);
        // PEAK DEPLOYED FOR THE PORTFOLIO is the peak of the SUMMED series,
        // which is not the sum of the per-strategy peaks unless they all peak
        // on the same day. Passing it as `capital` with a peak of 1 is how
        // obExtraStats is told "this is already the denominator".
        const peakDeployed = deployed.length ? Math.max(...deployed) : 0;
        const tExtra = obExtraStats(pooled, all, tStats, peakDeployed, 1);
        // Avg P/L % pools PER-TRADE percentages, each against its own
        // strategy's capital: with one strategy that is exactly the single
        // page's avg P/L / capital, and with several it is the only reading
        // that does not need a "portfolio capital per position" that does
        // not exist.
        tExtra.avg_pnl_pct = pctParts.length
          ? pctParts.reduce((a, b) => a + b, 0) / pctParts.length : null;
        rows.push({
          key: 'total', id: 0, name: 'TOTAL', color: '#ffffff', total: true,
          n: all.length,
          nAll: rows.reduce((a, r) => a + r.nAll, 0),
          dropped: rows.reduce((a, r) => a + r.dropped, 0),
          cost: { noValue: rows.reduce((a, r) => a + r.cost.noValue, 0), columns: [] },
          stats: tStats, extra: tExtra, conc: null,
          sharpe: obSharpe(pooled, all),
          capital: bpTotalCapital(this.chosen),
          peakDeployed,
        });
      }
      this.rows = rows;
      this.tick++;
    },

    // ── readouts ────────────────────────────────────────────────────────
    savedSummary() {
      return this.saved.length
        ? `${this.saved.length} saved · ${this.chosen.length} chosen`
        : 'none saved yet';
    },

    allocSummary() { return `${bpFmtMoney(bpTotalCapital(this.chosen))} total`; },

    rowMeta(c) {
      const s = this.saved.find(x => x.id === c.id);
      const n = s ? bpFmtInt(s.trade_count) : '—';
      const seeded = c.savedCapital == null ? ' · capital not saved' : '';
      return `${n} trades${seeded}`;
    },

    loadedSummary() {
      const trades = this.loaded.reduce((a, p) => a + (p.n || 0), 0);
      const sp = this.span();
      return `${this.loaded.length} strategies · ${bpFmtInt(trades)} trades`
           + (sp ? ` · ${sp.start} → ${sp.end}` : '');
    },

    statusLine() {
      if (!this.chosen.length) return 'Add saved strategies to build a portfolio.';
      if (!this.loaded.length) return `${this.chosen.length} chosen — not loaded yet.`;
      const sp = this.span();
      return `${this.loaded.length} loaded · `
           + `${bpFmtMoney(bpTotalCapital(this.chosen))} capital · `
           + (sp ? `${sp.start} → ${sp.end} (${this.rangeMode})` : 'no dates');
    },

    spanOf(p) {
      return (p.date_min && p.date_max) ? `${p.date_min} → ${p.date_max}` : '—';
    },

    parseOf(p) {
      const q = p.parse || {};
      return `${q.source === 'cache' ? 'cached' : 'parsed'} ${bpFmtSeconds(q.seconds)}`;
    },

    onRangeMode() { this.recompute(); },

    // Cell formatters. `void this.tick` is the reactivity tie: the rows are
    // rebuilt into reactive state, but a cell that read only BP_DATA would
    // never re-render. Same trap the OO page hit twice.
    cell(row, what) {
      void this.tick;
      const s = row.stats, e = row.extra;
      switch (what) {
        case 'n':        return bpFmtInt(row.n);
        case 'win':      return bpFmtPct(s.win_pct);
        case 'total':    return bpFmtMoney(s.total_pnl);
        case 'avg':      return bpFmtMoney(s.avg_pnl);
        case 'avgWin':   return bpFmtMoney(s.avg_win_pnl);
        case 'avgLoss':  return bpFmtMoney(s.avg_loss_pnl);
        case 'pf':       return e.profit_factor === null ? '—'
                              : (isFinite(e.profit_factor) ? bpFmtNum(e.profit_factor) : '∞');
        case 'dd':       return bpFmtMoney(s.max_drawdown);
        case 'calmar':   return bpFmtNum(e.calmar);
        case 'sharpe':   return bpFmtNum(row.sharpe);
        case 'annual':   return bpFmtMoney(e.avg_annual_pnl);
        case 'annualPct': return bpFmtPct(e.avg_annual_return_pct);
        case 'avgPct':   return bpFmtPct(e.avg_pnl_pct, 2);
        case 'days':     return bpFmtNum(s.avg_days_in_trade, 1);
        case 'peak':     return bpFmtMoney(row.peakDeployed);
        default:         return '';
      }
    },

    cellSign(row, what) {
      void this.tick;
      const s = row.stats, e = row.extra;
      const v = { total: s.total_pnl, avg: s.avg_pnl, dd: s.max_drawdown,
                  annual: e.avg_annual_pnl, calmar: e.calmar,
                  sharpe: row.sharpe, annualPct: e.avg_annual_return_pct,
                  avgPct: e.avg_pnl_pct }[what];
      if (v == null || !isFinite(v)) return '';
      return v >= 0 ? 'pos' : 'neg';
    },

    /* What the filters cost, per strategy, stated rather than implied. */
    rowNote(row) {
      void this.tick;
      if (row.total) return '';
      const parts = [];
      if (row.dropped) parts.push(`${bpFmtInt(row.dropped)} of ${bpFmtInt(row.nAll)} filtered out`);
      if (row.cost.noValue) {
        parts.push(`${bpFmtInt(row.cost.noValue)} have no value for an active `
                 + `filter and are dropped by it`);
      }
      if (row.conc && row.conc.sameSession === row.conc.counted && row.conc.counted) {
        parts.push('nothing held overnight, so no capital is deployed by this measure');
      }
      return parts.join(' · ');
    },

    warnings() {
      const out = [];
      for (const p of this.loaded) {
        const name = p.saved ? p.saved.name : p.filename;
        if (p.saved_count_changed) {
          out.push(`${name}: ${bpFmtInt(p.saved_count_changed.when_saved)} trades when `
                 + `saved, ${bpFmtInt(p.saved_count_changed.now)} now — the parser has `
                 + `changed since.`);
        }
        const mk = p.market || {};
        if (mk.joined === false) {
          out.push(`${name}: market data did not join (${mk.error || 'no reason given'}) `
                 + `— the metric filters have nothing to filter on.`);
        }
      }
      return out;
    },
  }));
});
