/* ============================================================================
 * OO/Mesosim Backtest page — analysis of an Option Omega CSV or DeltaRay
 * Mesosim JSON trade log. Replaces the Plotly Dash app in
 * Options-Backtest-Dashboard.
 *
 * EVERYTHING RENDERS OFF THE REGISTRY (/api/oo-backtest/registry). The
 * sidebar filters and the metric sections are loops over that list; no metric
 * column is named in this file. Adding a metric is one registry entry.
 *
 * FILTERING HAPPENS HERE, NOT ON THE SERVER. The whole trade log arrives once,
 * columnar, with ISO date strings; a filter change never makes a request.
 *
 * Phase 2: uploads are joined to main.index_ohlc on the server (VIX levels at
 * the entry bar, gaps against the prior session, both ratio bases); the
 * sidebar shows read-only market freshness, the ratio-basis toggle and per-
 * metric coverage.
 * Phase 2b: saved strategies -- save an upload (name + notes), pick one from
 * the dropdown and Load it (the stored file is re-parsed and re-joined), or
 * Delete it after a confirm.
 * Phase 3: filters (one date range, a dual slider per range metric bounded by
 * the loaded data, checkboxes per categorical metric), the ten summary stats
 * and the cumulative P/L + drawdown charts, all recomputed in the browser.
 * The metric sections are still stubs.
 * ==========================================================================*/

const OB_BLUE = '#3498db';   // positive (theme --accent)
const OB_PINK = '#e84393';   // negative

/* Trade columns live OUTSIDE the Alpine proxy. Thousands of values wrapped in
 * reactive getters is slow to build and slower to iterate, and nothing in the
 * template binds to an individual value. */
const OB_DATA = { columns: null, n: 0, file: null, idx: [] };
/* Chart.js instances, also outside the proxy (Alpine would wrap their internals). */
const OB_CHARTS = { cum: null, dd: null };

/* ── pure helpers (exercised in node by scripts/check_oo_backtest.py) ─────── */

/* Which bin a value falls in, matching pd.cut over the registry's spec.
 * `edges` are the INNER edges; the outer two are -inf/+inf.
 *   closed "left"   bins are [e(i-1), e(i))  -> index = #edges <= v
 *   closed "right"  bins are (e(i-1), e(i)]  -> index = #edges <  v
 * The side is an explicit registry field; anything else is a registry bug and
 * throws rather than silently binning one way.
 * Null / NaN -> -1 (no bin), as pd.cut gives NaN. */
function obBinIndex(v, bins) {
  if (v === null || v === undefined || Number.isNaN(v)) return -1;
  if (bins.closed !== 'left' && bins.closed !== 'right') {
    throw new Error(`bin spec has no closed side: ${JSON.stringify(bins.closed)}`);
  }
  const e = bins.edges;
  let lo = 0, hi = e.length;
  if (bins.closed === 'right') {
    while (lo < hi) { const m = (lo + hi) >> 1; if (e[m] < v) lo = m + 1; else hi = m; }
  } else {
    while (lo < hi) { const m = (lo + hi) >> 1; if (e[m] <= v) lo = m + 1; else hi = m; }
  }
  return lo;
}

/* Non-null numeric extent of a column, or null when it has no values. */
function obExtent(values) {
  let lo = Infinity, hi = -Infinity, n = 0;
  for (const v of values || []) {
    if (v === null || v === undefined || Number.isNaN(v)) continue;
    if (v < lo) lo = v;
    if (v > hi) hi = v;
    n++;
  }
  return n ? { min: lo, max: hi, n } : null;
}

/* Distinct non-null values, sorted. */
function obDistinct(values) {
  const s = new Set();
  for (const v of values || []) if (v !== null && v !== undefined) s.add(v);
  return [...s].sort((a, b) => (a < b ? -1 : a > b ? 1 : 0));
}

const obNull = v => v === null || v === undefined || (typeof v === 'number' && Number.isNaN(v));

/* Row indices passing every ACTIVE filter. A filter is only passed in when it
 * narrows something -- an untouched range lets null values through, a
 * narrowed one drops them (a trade with no VIX9D cannot be "VIX9D 12-20").
 *   {kind:'date',  column, from, to}   ISO strings, inclusive
 *   {kind:'range', column, lo, hi}     inclusive, as filter_dataframe had it
 *   {kind:'set',   column, allowed}    a Set of permitted values */
function obApplyFilters(cols, n, specs) {
  const idx = [];
  outer:
  for (let i = 0; i < n; i++) {
    for (const f of specs) {
      const col = cols[f.column];
      const v = col ? col[i] : null;
      if (f.kind === 'range') {
        if (obNull(v) || v < f.lo || v > f.hi) continue outer;
      } else if (f.kind === 'set') {
        if (!f.allowed.has(v)) continue outer;
      } else if (f.kind === 'date') {
        if (obNull(v) || (f.from && v < f.from) || (f.to && v > f.to)) continue outer;
      }
    }
    idx.push(i);
  }
  return idx;
}

/* Order for anything cumulative: by close date, ties kept in payload order
 * (open date, then file order). A STABLE sort -- pandas' default sort in the
 * source app's stats is not, so its max drawdown could differ across runs
 * when several trades close the same day. */
function obByClose(cols, idx) {
  const dc = cols.date_closed;
  return idx.slice().sort((a, b) => (dc[a] < dc[b] ? -1 : dc[a] > dc[b] ? 1 : a - b));
}

/* The summary figures, as utils/stats.py calculate_stats defines them --
 * including its quirk that a zero-P/L trade counts as a LOSS for Avg Loss
 * (losses are pnl <= 0) but not as a win. */
function obStats(cols, idx) {
  const n = idx.length;
  const empty = { num_trades: 0, win_pct: 0, avg_pnl: 0, total_pnl: 0, avg_days_in_trade: 0,
                  max_drawdown: 0, avg_win_pnl: 0, avg_loss_pnl: 0, max_winner: 0, max_loser: 0 };
  if (!n) return empty;
  const pnl = cols.pnl, dit = cols.days_in_trade || [];
  let total = 0, wins = 0, winSum = 0, losses = 0, lossSum = 0, maxW = -Infinity, maxL = Infinity;
  let ditSum = 0, ditN = 0;
  for (const i of idx) {
    const p = pnl[i];
    total += p;
    if (p > 0) { wins++; winSum += p; } else { losses++; lossSum += p; }
    if (p > maxW) maxW = p;
    if (p < maxL) maxL = p;
    if (!obNull(dit[i])) { ditSum += dit[i]; ditN++; }
  }
  const eq = obEquity(cols, idx);
  return {
    num_trades: n,
    win_pct: wins / n * 100,
    avg_pnl: total / n,
    total_pnl: total,
    avg_days_in_trade: ditN ? ditSum / ditN : 0,
    max_drawdown: eq.maxDD ? eq.maxDD.drawdown : 0,
    avg_win_pnl: wins ? winSum / wins : 0,
    avg_loss_pnl: losses ? lossSum / losses : 0,
    max_winner: maxW,
    max_loser: maxL,
  };
}

/* Cumulative P/L, running peak and drawdown per trade in close order, as
 * calculations.py calculate_drawdown. maxDD is the deepest point (first
 * occurrence), or null when the curve never falls below its peak. */
function obEquity(cols, idx) {
  const order = obByClose(cols, idx);
  const pnl = cols.pnl, dc = cols.date_closed;
  const points = [];
  let cum = 0, peak = -Infinity, maxDD = null;
  for (const i of order) {
    cum += pnl[i];
    if (cum > peak) peak = cum;
    const dd = cum - peak;
    const pt = { row: i, date: dc[i], pnl: pnl[i], cumulative: cum, peak, drawdown: dd };
    points.push(pt);
    if (dd < 0 && (maxDD === null || dd < maxDD.drawdown)) maxDD = pt;
  }
  return { points, maxDD };
}

/* Days since the epoch for an ISO date, for a linear x axis (no date adapter). */
function obDay(iso) { return Date.parse(iso + 'T00:00:00Z') / 86400000; }
function obIsoDay(d) { return new Date(Math.round(d) * 86400000).toISOString().slice(0, 10); }

function obMoney(v, digits = 0) {
  if (obNull(v)) return '—';
  const a = Math.abs(v).toLocaleString('en-US', { minimumFractionDigits: digits, maximumFractionDigits: digits });
  return (v < 0 ? '-$' : '$') + a;
}

/* "SPX Iron Condor 45DTE (1,284 trades, 2021-01-04 – 2026-03-13)" */
function obSavedLabel(s) {
  if (!s) return '';
  const n = Number(s.trade_count || 0).toLocaleString('en-US');
  return `${s.name} (${n} trade${s.trade_count === 1 ? '' : 's'}, ${s.date_min || '—'} – ${s.date_max || '—'})`;
}

/* Snap a data extent outward to the slider's step, so the untouched slider
 * spans every value (a floor/ceil at 0.1 on 0.37 must not cut off 0.37). */
function obStepDecimals(step) { return (String(step).split('.')[1] || '').length; }
function obSnap(v, step, how) {
  const dec = obStepDecimals(step);
  return +((Math[how](v / step + (how === 'floor' ? 1e-9 : -1e-9))) * step).toFixed(dec);
}

function obFmt(v, fmt) {
  if (v === null || v === undefined || Number.isNaN(v)) return '—';
  switch (fmt) {
    case 'usd':   return (v < 0 ? '-$' : '$') + Math.abs(v).toLocaleString(undefined, { maximumFractionDigits: 0 });
    case 'pct':   return v.toFixed(2) + '%';
    case 'ratio': return v.toFixed(3);
    case 'int':   return String(Math.round(v));
    case 'num':   return v.toFixed(2);
    default:      return String(v);
  }
}

document.addEventListener('alpine:init', () => {
  Alpine.data('ooBacktest', () => ({
    registry: [],
    registryError: '',

    // loaded log (metadata only — values are in OB_DATA)
    loaded: false,
    meta: null,          // {n, filename, source, suggested_name, date_min, date_max, market_joined}
    presentColumns: [],  // columns in the payload with at least one non-null value

    uploading: false,
    uploadError: '',
    dragOver: false,

    market: null,
    marketLoading: false,
    marketError: '',
    marketDetail: false,
    coverageError: '',

    // Saved strategies. The dropdown lists them; Load re-parses the stored
    // file and re-joins market data (the same path as an upload).
    savedList: [],
    savedError: '',
    selectedSavedId: '',
    savedBusy: false,
    // Set when the loaded log came from an upload whose File the page still
    // holds -- only then can it be saved (the server re-parses the bytes).
    hasUploadFile: false,
    saveName: '',
    saveNotes: '',
    saveBusy: false,
    saveError: '',
    saveMsg: '',

    // Filters. `ranges` and `cats` are keyed by registry key and built from
    // the loaded data by initFilters(); nothing here is a static default.
    filters: { dateFrom: '', dateTo: '', ranges: {}, cats: {} },
    dateBounds: { min: '', max: '' },
    filteredCount: 0,
    stats: null,
    _rafPending: false,

    // Which ratio basis the sections read: entry-time bars (default) or the
    // entry date's daily closes. Temporary -- one basis is deleted, and this
    // toggle with it, once both have been looked at on a real log.
    ratioBasis: 'entry',

    async init() {
      await Promise.all([this.loadRegistry(), this.loadMarketStatus(), this.loadSavedList()]);
    },

    async loadRegistry() {
      try {
        const r = await fetch('/api/oo-backtest/registry');
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        const body = await r.json();
        this.registry = body.metrics || [];
        this.coverageError = body.coverage_error || '';
      } catch (e) {
        console.error('oo-backtest registry', e);
        this.registryError = `Registry failed to load: ${e.message}`;
      }
    },

    async loadMarketStatus() {
      this.marketLoading = true;
      this.marketError = '';
      try {
        const r = await fetch('/api/oo-backtest/market-status');
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        this.market = await r.json();
      } catch (e) {
        console.error('oo-backtest market-status', e);
        this.marketError = e.message;
      } finally {
        this.marketLoading = false;
      }
    },

    /* ── ingestion ─────────────────────────────────────────────────────── */

    onDrop(ev) {
      this.dragOver = false;
      const f = ev.dataTransfer && ev.dataTransfer.files && ev.dataTransfer.files[0];
      if (f) this.upload(f);
    },

    onPick(ev) {
      const f = ev.target.files && ev.target.files[0];
      if (f) this.upload(f);
      ev.target.value = '';   // re-picking the same file must fire change again
    },

    async upload(file) {
      const name = file.name || '';
      if (!/\.(csv|json)$/i.test(name)) {
        this.uploadError = `${name}: upload an Option Omega .csv or a Mesosim .json`;
        return;
      }
      this.uploading = true;
      this.uploadError = '';
      try {
        const fd = new FormData();
        fd.append('file', file);
        const r = await fetch('/api/oo-backtest/parse', { method: 'POST', body: fd });
        const body = await r.json().catch(() => ({}));
        if (!r.ok) throw new Error(body.detail || `HTTP ${r.status}`);
        this.setTrades(body);
        OB_DATA.file = file;
        this.hasUploadFile = true;
        this.selectedSavedId = '';
        this.saveName = body.suggested_name || '';
        this.saveNotes = '';
        this.saveError = '';
        this.saveMsg = '';
      } catch (e) {
        console.error('oo-backtest upload', e);
        this.uploadError = e.message;
      } finally {
        this.uploading = false;
      }
    },

    setTrades(payload) {
      OB_DATA.columns = payload.columns;
      OB_DATA.n = payload.n;
      this.presentColumns = Object.keys(payload.columns)
        .filter(c => payload.columns[c].some(v => v !== null && v !== undefined));
      const { columns, ...meta } = payload;
      this.meta = meta;
      this.loaded = true;
      this.initFilters();
      this.$nextTick(() => this.recompute());
    },

    /* ── saved strategies ──────────────────────────────────────────────── */

    savedLabel(s) { return obSavedLabel(s); },

    async loadSavedList() {
      this.savedError = '';
      try {
        const r = await fetch('/api/oo-backtest/strategies');
        const body = await r.json().catch(() => ({}));
        if (!r.ok) throw new Error(body.detail || `HTTP ${r.status}`);
        this.savedList = body.strategies || [];
        if (this.selectedSavedId && !this.savedList.some(x => String(x.id) === String(this.selectedSavedId))) {
          this.selectedSavedId = '';
        }
      } catch (e) {
        console.error('oo-backtest saved list', e);
        this.savedError = `Saved strategies unavailable: ${e.message}`;
      }
    },

    selectedSaved() { return this.savedList.find(x => String(x.id) === String(this.selectedSavedId)) || null; },

    async loadSelected() {
      const s = this.selectedSaved();
      if (!s) return;
      this.savedBusy = true;
      this.savedError = '';
      this.uploadError = '';
      try {
        const r = await fetch(`/api/oo-backtest/strategies/${s.id}/load`);
        const body = await r.json().catch(() => ({}));
        if (!r.ok) throw new Error(body.detail || `HTTP ${r.status}`);
        this.setTrades(body);
        // A loaded saved strategy is already saved; there is no File to send.
        OB_DATA.file = null;
        this.hasUploadFile = false;
        this.saveMsg = '';
        this.saveError = '';
      } catch (e) {
        console.error('oo-backtest load saved', e);
        this.savedError = `Could not load "${s.name}": ${e.message}`;
      } finally {
        this.savedBusy = false;
      }
    },

    async saveStrategy(replace = false) {
      if (!OB_DATA.file) return;
      this.saveBusy = true;
      this.saveError = '';
      this.saveMsg = '';
      try {
        const fd = new FormData();
        fd.append('file', OB_DATA.file);
        fd.append('name', this.saveName);
        fd.append('notes', this.saveNotes);
        fd.append('replace', replace ? 'true' : 'false');
        const r = await fetch('/api/oo-backtest/strategies', { method: 'POST', body: fd });
        const body = await r.json().catch(() => ({}));
        if (r.status === 409 && !replace) {
          this.saveBusy = false;
          if (confirm(`${body.detail} Replace it with this upload?`)) return this.saveStrategy(true);
          this.saveError = 'Not saved — choose a different name.';
          return;
        }
        if (!r.ok) throw new Error(body.detail || `HTTP ${r.status}`);
        const saved = body.strategy;
        await this.loadSavedList();
        this.selectedSavedId = String(saved.id);
        this.meta = { ...this.meta, saved };
        this.saveMsg = `Saved as ${obSavedLabel(saved)}` +
          (saved.same_file_as && saved.same_file_as.length ? ` — the same file is also saved as: ${saved.same_file_as.join(', ')}` : '');
      } catch (e) {
        console.error('oo-backtest save', e);
        this.saveError = `Save failed: ${e.message}`;
      } finally {
        this.saveBusy = false;
      }
    },

    async deleteSaved() {
      const s = this.selectedSaved();
      if (!s) return;
      if (!confirm(`Delete saved strategy "${obSavedLabel(s)}"? This cannot be undone.`)) return;
      this.savedError = '';
      try {
        const r = await fetch(`/api/oo-backtest/strategies/${s.id}`, { method: 'DELETE' });
        const body = await r.json().catch(() => ({}));
        if (!r.ok) throw new Error(body.detail || `HTTP ${r.status}`);
        // The trades on screen stay; they are just no longer a saved record.
        if (this.meta && this.meta.saved && this.meta.saved.id === s.id) {
          const { saved, ...rest } = this.meta;
          this.meta = rest;
        }
        this.selectedSavedId = '';
        await this.loadSavedList();
      } catch (e) {
        console.error('oo-backtest delete', e);
        this.savedError = `Delete failed: ${e.message}`;
      }
    },

    /* ── filters ───────────────────────────────────────────────────────── */

    /* Bounds and options from THE LOADED LOG, never static defaults: every
     * range slider opens spanning the data's own extent (snapped outward to
     * its step), every checkbox on. */
    initFilters() {
      const cols = OB_DATA.columns || {};
      const opened = (cols.date_opened || []).filter(Boolean);
      const dmin = opened.length ? opened.reduce((a, b) => (a < b ? a : b)) : '';
      const dmax = opened.length ? opened.reduce((a, b) => (a > b ? a : b)) : '';
      this.dateBounds = { min: dmin, max: dmax };
      const ranges = {}, cats = {};
      for (const m of this.registry.filter(x => x.filter)) {
        if (m.type === 'range') {
          const r = this.rangeStateFor(m);
          if (r) ranges[m.key] = r;
        } else {
          const vals = cols[m.column] || [];
          const counts = new Map();
          for (const v of vals) counts.set(v, (counts.get(v) || 0) + 1);
          let options;
          if (m.categories) {
            options = m.categories.map(c => ({ value: c.value, label: c.label, count: counts.get(c.value) || 0 }));
          } else {
            options = obDistinct(vals).map(v => ({ value: v, label: String(v), count: counts.get(v) }));
            if (counts.has(null)) options.push({ value: null, label: '(none)', count: counts.get(null) });
          }
          cats[m.key] = { options, selected: options.map(o => o.value) };
        }
      }
      this.filters = { dateFrom: dmin, dateTo: dmax, ranges, cats };
    },

    rangeStateFor(m) {
      const ext = obExtent((OB_DATA.columns || {})[this.metricColumn(m)]);
      if (!ext) return null;
      const min = obSnap(ext.min, m.step, 'floor'), max = obSnap(ext.max, m.step, 'ceil');
      return { min, max, lo: min, hi: max, step: m.step, n: ext.n };
    },

    rangeOf(m) { return this.filters.ranges[m.key] || null; },
    catOf(m) { return this.filters.cats[m.key] || null; },

    dateActive() {
      return !!(this.dateBounds.min && ((this.filters.dateFrom && this.filters.dateFrom > this.dateBounds.min)
        || (this.filters.dateTo && this.filters.dateTo < this.dateBounds.max)));
    },

    isActive(m) {
      const r = this.rangeOf(m), c = this.catOf(m);
      if (m.type === 'range') return !!r && (r.lo > r.min || r.hi < r.max);
      return !!c && c.selected.length < c.options.length;
    },

    activeCount() {
      return (this.dateActive() ? 1 : 0) + this.registry.filter(m => m.filter && this.isActive(m)).length;
    },

    setLo(m, raw) {
      const r = this.rangeOf(m);
      if (!r) return;
      r.lo = Math.min(+raw, r.hi);
      this.onFilterChange();
    },

    setHi(m, raw) {
      const r = this.rangeOf(m);
      if (!r) return;
      r.hi = Math.max(+raw, r.lo);
      this.onFilterChange();
    },

    toggleCat(m, value) {
      const c = this.catOf(m);
      if (!c) return;
      const i = c.selected.findIndex(v => v === value);
      if (i >= 0) c.selected.splice(i, 1); else c.selected.push(value);
      this.onFilterChange();
    },

    setAllCats(m, on) {
      const c = this.catOf(m);
      if (!c) return;
      c.selected = on ? c.options.map(o => o.value) : [];
      this.onFilterChange();
    },

    resetFilter(m) {
      if (m.type === 'range') {
        const r = this.rangeOf(m);
        if (r) { r.lo = r.min; r.hi = r.max; }
      } else {
        this.setAllCats(m, true);
        return;
      }
      this.onFilterChange();
    },

    resetDate() {
      this.filters.dateFrom = this.dateBounds.min;
      this.filters.dateTo = this.dateBounds.max;
      this.onFilterChange();
    },

    resetAllFilters() {
      this.initFilters();
      this.onFilterChange();
    },

    setRatioBasis(basis) {
      if (this.ratioBasis === basis) return;
      this.ratioBasis = basis;
      // A basis switch changes the COLUMN a ratio filter reads, so its bounds
      // come from the new column and any narrowing on the old one is dropped.
      for (const m of this.registry.filter(x => x.basis && x.filter)) {
        const r = this.rangeStateFor(m);
        if (r) this.filters.ranges[m.key] = r; else delete this.filters.ranges[m.key];
      }
      this.onFilterChange();
    },

    activeSpecs() {
      const specs = [];
      if (this.dateActive()) {
        specs.push({ kind: 'date', column: 'date_opened', from: this.filters.dateFrom, to: this.filters.dateTo });
      }
      for (const m of this.registry.filter(x => x.filter)) {
        if (!this.isActive(m)) continue;
        if (m.type === 'range') {
          const r = this.rangeOf(m);
          specs.push({ kind: 'range', column: this.metricColumn(m), lo: r.lo, hi: r.hi });
        } else {
          specs.push({ kind: 'set', column: m.column, allowed: new Set(this.catOf(m).selected) });
        }
      }
      return specs;
    },

    /* Slider input fires continuously; recompute at most once per frame. */
    onFilterChange() {
      if (this._rafPending) return;
      this._rafPending = true;
      const run = () => { this._rafPending = false; this.recompute(); };
      if (typeof requestAnimationFrame === 'function') requestAnimationFrame(run); else run();
    },

    recompute() {
      if (!OB_DATA.columns) return;
      const idx = obApplyFilters(OB_DATA.columns, OB_DATA.n, this.activeSpecs());
      OB_DATA.idx = idx;
      this.filteredCount = idx.length;
      this.stats = obStats(OB_DATA.columns, idx);
      this.renderPerformance(obEquity(OB_DATA.columns, idx));
    },

    fmtVal(m, v) { return obFmt(v, m.format); },
    moneyText(v) { return obMoney(v); },

    rangeSummary(m) {
      const r = this.rangeOf(m);
      return r ? `${r.n.toLocaleString()} trades with a value · data ${obFmt(r.min, m.format)} … ${obFmt(r.max, m.format)}` : '';
    },

    /* ── summary stats ─────────────────────────────────────────────────── */

    stat(key) {
      const st = this.stats;
      if (!st) return '—';
      switch (key) {
        case 'num_trades': return st.num_trades.toLocaleString();
        case 'win_pct': return st.win_pct.toFixed(1) + '%';
        case 'avg_days_in_trade': return st.avg_days_in_trade.toFixed(1);
        default: return obMoney(st[key], ['avg_pnl', 'avg_win_pnl', 'avg_loss_pnl'].includes(key) ? 2 : 0);
      }
    },

    statNum(key) { return this.stats ? this.stats[key] : 0; },

    /* ── performance charts ────────────────────────────────────────────── */

    maxDDLine() {
      const st = this.stats;
      if (!st || !st.num_trades) return '';
      const eq = obEquity(OB_DATA.columns, OB_DATA.idx);
      return eq.maxDD ? `max ${obMoney(eq.maxDD.drawdown)} on ${eq.maxDD.date}` : 'no drawdown';
    },

    renderPerformance(eq) {
      if (typeof Chart === 'undefined') return;
      const cumEl = document.getElementById('ob-cum-chart');
      const ddEl = document.getElementById('ob-dd-chart');
      if (!cumEl || !ddEl) return;
      const pts = eq.points;
      const cum = pts.map(p => ({ x: obDay(p.date), y: p.cumulative, p }));
      const dd = pts.map(p => ({ x: obDay(p.date), y: p.drawdown, p }));
      const mark = eq.maxDD ? [{ x: obDay(eq.maxDD.date), y: eq.maxDD.drawdown, p: eq.maxDD }] : [];

      const axis = (money) => ({
        type: 'linear',
        grid: { color: 'rgba(255,255,255,0.05)' },
        border: { display: false },
        ticks: { color: '#9a9a9a', font: { size: 10 }, maxTicksLimit: money ? 6 : 7,
                 callback: money ? (v => obMoney(v)) : (v => obIsoDay(v).slice(0, 7)) },
      });
      // The x axis spans exactly the plotted dates; Chart.js would otherwise
      // round a linear scale out to "nice" values months beyond the data.
      const xr = cum.length ? { min: cum[0].x, max: cum[cum.length - 1].x } : {};
      const base = (tooltipLabel) => ({
        responsive: true, maintainAspectRatio: false, animation: false, parsing: false,
        interaction: { mode: 'nearest', axis: 'x', intersect: false },
        scales: { x: { ...axis(false), ...xr }, y: axis(true) },
        plugins: {
          legend: { display: false },
          tooltip: { callbacks: { title: it => it[0].raw.p.date, label: tooltipLabel } },
        },
      });

      const cumData = { datasets: [{
        data: cum, borderColor: OB_BLUE, backgroundColor: 'rgba(52,152,219,0.10)', fill: 'origin',
        borderWidth: 2, pointRadius: 0, pointHitRadius: 6, tension: 0 }] };
      const ddData = { datasets: [
        { data: dd, borderColor: OB_PINK, backgroundColor: 'rgba(232,67,147,0.16)', fill: 'origin',
          borderWidth: 2, pointRadius: 0, pointHitRadius: 6, tension: 0 },
        // The deepest point: a marker with a surface-coloured ring so it reads
        // over the line, and hover-only like the rest.
        { data: mark, type: 'scatter', pointRadius: 5, pointHoverRadius: 6, pointBackgroundColor: OB_PINK,
          pointBorderColor: '#2d2d2d', pointBorderWidth: 2, showLine: false },
      ] };

      if (OB_CHARTS.cum) { OB_CHARTS.cum.data = cumData; Object.assign(OB_CHARTS.cum.options.scales.x, xr); OB_CHARTS.cum.update('none'); }
      else {
        OB_CHARTS.cum = new Chart(cumEl.getContext('2d'), { type: 'line', data: cumData,
          options: base(c => `Cumulative ${obMoney(c.raw.p.cumulative)} · trade ${obMoney(c.raw.p.pnl)}`) });
      }
      if (OB_CHARTS.dd) { OB_CHARTS.dd.data = ddData; Object.assign(OB_CHARTS.dd.options.scales.x, xr); OB_CHARTS.dd.update('none'); }
      else {
        OB_CHARTS.dd = new Chart(ddEl.getContext('2d'), { type: 'line', data: ddData,
          options: base(c => (c.datasetIndex === 1 ? 'Max drawdown ' : 'Drawdown ') +
                             `${obMoney(c.raw.p.drawdown)} · peak ${obMoney(c.raw.p.peak)}`) });
      }
    },

    /* ── registry-driven views ─────────────────────────────────────────── */

    filterMetrics() { return this.registry.filter(m => m.filter); },
    sectionMetrics() { return this.registry.filter(m => m.section); },

    /* The column a metric reads now: its basis column where it has more than
     * one (the ratios), else its only column. */
    metricColumn(m) { return m.basis ? m.basis[this.ratioBasis] : m.column; },

    hasColumn(m) { return this.presentColumns.includes(this.metricColumn(m)); },

    hasBasisMetrics() { return this.registry.some(m => m.basis); },

    /* Trades in the loaded log that open before a metric's coverage. A range
     * filter drops trades with no value, so filtering on this metric would
     * drop these too -- said before it happens, not discovered after. */
    tradesBeforeCoverage(m) {
      if (!this.loaded || !m.minDate || !OB_DATA.columns) return 0;
      let k = 0;
      for (const d of OB_DATA.columns.date_opened) if (d && d < m.minDate) k++;
      return k;
    },

    coverageNote(m) {
      const k = this.tradesBeforeCoverage(m);
      if (!k) return '';
      const s = k === 1 ? '' : 's';
      return this.isActive(m)
        ? `Data starts ${m.minDate} — this filter is dropping ${k.toLocaleString()} earlier trade${s}`
        : `Data starts ${m.minDate} — filtering on this drops ${k.toLocaleString()} earlier trade${s}`;
    },

    binCount(m) { return m.bins ? m.bins.labels.length : null; },

    /* A range section is skipped when its column is absent or all-null. */
    sectionState(m) {
      if (!this.loaded) return 'empty';
      return this.hasColumn(m) ? 'ready' : 'skipped';
    },

    /* What the parser dropped or had to decide, said on load rather than left
     * for the numbers to hide. `warn` lines are about trades the stats exclude
     * or fills that may not be trustworthy; the rest say where a field came
     * from when the export's version lacked the preferred one. */
    parseNotes() {
      const n = (this.meta && this.meta.notes) || {};
      const out = [];
      const plural = (k, one, many) => `${k} ${k === 1 ? one : many}`;
      if (n.open_positions) {
        out.push({ warn: true, text: `${plural(n.open_positions, 'position', 'positions')} still open at backtest end, excluded` });
      }
      if ((n.missing_data_at_fill || []).length) {
        out.push({ warn: true, text: `${plural(n.missing_data_at_fill.length, 'trade has', 'trades have')} MissingData on the entry or exit bar — that fill may be stale` });
      }
      if ((n.pnl_mismatch || []).length) {
        out.push({ warn: true, text: `${plural(n.pnl_mismatch.length, 'trade', 'trades')}: pos_realized_pnl ≠ pos_pnl (realized used)` });
      }
      if (n.multi_signal_positions) {
        out.push({ warn: false, text: `${plural(n.multi_signal_positions, 'position', 'positions')} fired several exit signals on the exit bar; resolved by precedence (price > adjustments > time)` });
      }
      if ((n.precedence_vs_order || []).length) {
        out.push({ warn: true, text: `${plural(n.precedence_vs_order.length, 'position', 'positions')}: precedence chose a different reason than the last signal in the file — MesoSim's ordering may have changed` });
      }
      if ((n.pnl_contradicts_reason || []).length) {
        out.push({ warn: true, text: `${plural(n.pnl_contradicts_reason.length, 'trade', 'trades')}: exit reason contradicts P/L (profit target with a loss, or stop with a gain)` });
      }
      if ((n.premium_leg_mismatch || []).length) {
        out.push({ warn: true, text: `${plural(n.premium_leg_mismatch.length, 'trade', 'trades')}: entry_net_premium ≠ sum of entry leg fills` });
      }
      if (n.pnl_field === 'pos_pnl') {
        out.push({ warn: false, text: 'P/L from pos_pnl — this MesoSim version has no realized P/L field' });
      }
      if ((n.premium_field || '').includes('leg fills')) {
        out.push({ warn: false, text: 'Premium summed from entry leg fills — no entry_net_premium in this export' });
      }
      return out;
    },

    sourceLabel() {
      if (!this.meta) return '';
      return this.meta.source === 'mesosim_json' ? 'Mesosim JSON' : 'Option Omega CSV';
    },

    loadedLabel() {
      if (!this.meta) return '';
      const name = (this.meta.saved && this.meta.saved.name) || this.meta.suggested_name;
      return `${name} (${this.meta.n.toLocaleString()} trades, ` +
             `${this.meta.date_min} – ${this.meta.date_max})`;
    },

    savedNotes() {
      const out = [];
      const sv = this.meta && this.meta.saved;
      if (sv && !this.hasUploadFile) {
        out.push({ warn: false, text: `Saved strategy, saved ${String(sv.updated_at || '').slice(0, 10)}` });
      }
      const ch = this.meta && this.meta.saved_count_changed;
      if (ch) {
        out.push({ warn: true, text: `Trade count was ${ch.when_saved} when saved and is ${ch.now} now — the parser has changed since` });
      }
      return out;
    },

    /* ── market data: what the join did on this log ───────────────────── */

    /* Said on load, because each of these is invisible in the charts: a 0%
     * entry-time coverage makes every VIX metric a daily-open proxy, and an
     * off-by-one prior close shifts every gap while the page looks normal. */
    marketNotes() {
      const mk = this.meta && this.meta.market;
      if (!mk) return [];
      if (!mk.joined) {
        return [{ warn: true, text: `Market data not joined — ${mk.error || 'unknown reason'}. VIX and gap sections will show as skipped.` }];
      }
      const out = [];
      const e = mk.entry_time || {};
      out.push({
        warn: e.entry_time_found < e.trades,
        text: `Entry time found for ${(e.entry_time_found || 0).toLocaleString()} of ${(e.trades || 0).toLocaleString()} trades` +
              (e.entry_time_fallback_0930 ? ` — ${e.entry_time_fallback_0930} use the 09:30 bar` : ''),
      });
      if (e.before_open) out.push({ warn: true, text: `${e.before_open} entries before 09:30 — no bar that day, VIX levels null` });
      if (e.after_close) out.push({ warn: true, text: `${e.after_close} entries after 16:00 — check the log's time zone` });
      if (mk.trades_without_daily_row) {
        out.push({ warn: true, text: `${mk.trades_without_daily_row} trades open on a date index_ohlc has no session for` });
      }
      // Null levels WITH their reasons: a count alone cannot tell coverage
      // starting later from bars missing inside coverage.
      for (const s of ['vix', 'vix3m', 'vix9d']) {
        const r = mk.null_reasons && mk.null_reasons[s];
        if (r && r.null) {
          out.push({ warn: false, text: `${s.toUpperCase()} level null for ${r.null}: ` +
                     Object.entries(r.reasons).map(([why, n]) => `${n} ${why}`).join('; ') });
        }
      }
      const pushed = Object.entries(mk.entry_bars || {}).filter(([, v]) => v.earlier_than_entry_bar).map(([k, v]) => `${k.toUpperCase()} ${v.earlier_than_entry_bar}`);
      if (pushed.length) out.push({ warn: false, text: `Entry bar was NaN, earlier bar used: ${pushed.join(', ')}` });
      // All-null gaps are what the zero-filled weekends produced for a
      // Monday-only log, with nothing on screen saying so.
      for (const [name, v] of Object.entries(mk.gaps || {})) {
        const total = v.computed + v.null;
        if (v.null) {
          const r = mk.null_reasons && mk.null_reasons[`${name}_gap`];
          const why = r ? ' — ' + Object.entries(r.reasons).map(([k, n]) => `${n} ${k}`).join('; ') : '';
          out.push({ warn: v.null > total * 0.05,
                     text: `${name.toUpperCase()} gap computed for ${v.computed.toLocaleString()} of ${total.toLocaleString()} trades${why}` });
        }
      }
      return out;
    },

    /* ── market-data checks card (main column) ────────────────────────── */

    mk() { return (this.meta && this.meta.market) || {}; },
    hasMarketChecks() { return !!this.mk().null_reasons; },
    nullTrades() {
      const out = [];
      for (const [name, r] of Object.entries(this.mk().null_reasons || {})) {
        for (const t of r.trades || []) out.push({ name, ...t });
      }
      return out;
    },
    diagErrors() { return Object.entries(this.mk().diagnostic_errors || {}).map(([k, v]) => `${k}: ${v}`); },

    /* ── market freshness (read-only) ─────────────────────────────────── */

    marketSummary() {
      if (this.marketLoading) return 'Checking…';
      if (this.marketError) return `Status failed: ${this.marketError}`;
      if (!this.market) return '';
      if (!this.market.ok) return `index_ohlc unavailable — ${this.market.error}`;
      return `index_ohlc through ${this.market.latest_date} ${(this.market.latest_time || '').slice(0, 5)}`;
    },

    marketStale() { return !!(this.market && this.market.ok && this.market.stale); },
    staleText() {
      const m = this.market || {};
      return m.age_days === null || m.age_days === undefined
        ? 'Stale — no valid SPX bar in index_ohlc'
        : `Stale — latest valid SPX bar is ${m.age_days} days old (limit ${m.stale_after_days})`;
    },

    fallbackLines() {
      const fb = (this.market && this.market.close_fallback) || {};
      return Object.entries(fb).map(([s, v]) =>
        `${s.toUpperCase()}: ${v.early_close_days.length} early-close, ${v.full_session_count} full-session` +
        (v.full_session_count ? ` (e.g. ${v.full_session_sample.slice(0, 3).join(', ')})` : '') +
        (v.no_close_days ? `, ${v.no_close_days} days with no close` : ''));
    },

    fallbackWarn() {
      const fb = (this.market && this.market.close_fallback) || {};
      return Object.values(fb).some(v => v.full_session_count > 0);
    },

    /* How the per-series session rule classified the table. The shortest
     * session kept and the longest artifact rejected sit either side of the
     * threshold -- if either comes close to it, the threshold needs a look. */
    sessionLines() {
      const z = this.market && this.market.sessions;
      if (!z) return [];
      const out = [
        `a series has a session with ≥ ${z.min_session_bars} valid bars`,
        `${z.zero_filled_days} zero-filled days (${z.zero_filled_weekdays} weekdays)`,
        `${z.artifact_only_days} artifact-only days (bars, but no series reaches a session)` +
          (z.artifact_only.length ? `: ${z.artifact_only.slice(0, 12).map(a => a.date).join(', ')}` +
                                    (z.artifact_only.length > 12 ? ', …' : '') : ''),
      ];
      for (const [s, v] of Object.entries(z.by_series || {})) {
        const kept = v.shortest_kept ? `${v.shortest_kept.bars} (${v.shortest_kept.date})` : '—';
        const rej = v.longest_rejected ? `${v.longest_rejected.bars} (${v.longest_rejected.date})` : '—';
        let line = `${s.toUpperCase()}: ${v.sessions} sessions; shortest kept ${kept}; longest rejected ${rej}`;
        if (v.missing_on_session_days.length) {
          line += `; missing on ${v.missing_on_session_days.length} session days (e.g. ${v.missing_on_session_days.slice(0, 3).join(', ')})`;
        }
        if (v.zero_bars_in_sessions || v.nan_bars_in_sessions) {
          line += `; invalid bars in sessions: ${v.zero_bars_in_sessions} zero, ${v.nan_bars_in_sessions} NaN`;
        }
        out.push(line);
      }
      return out;
    },

    coverageLines() {
      const c = (this.market && this.market.coverage) || {};
      return Object.entries(c).map(([s, d]) => `${s.toUpperCase()} from ${d || '—'}`);
    },
  }));
});
