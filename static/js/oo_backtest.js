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
 * Phase 1: the scaffold. Upload + parse are real, the registry drives the
 * sidebar and section skeletons, and the market-data probe is shown. Filters,
 * stats and charts are labelled stubs.
 * ==========================================================================*/

const OB_BLUE = '#3498db';   // positive (theme --accent)
const OB_PINK = '#e84393';   // negative

/* Trade columns live OUTSIDE the Alpine proxy. Thousands of values wrapped in
 * reactive getters is slow to build and slower to iterate, and nothing in the
 * template binds to an individual value. */
const OB_DATA = { columns: null, n: 0 };

/* ── pure helpers (exercised in node by scripts/check_oo_backtest.py) ─────── */

/* Which bin a value falls in, matching pd.cut over the registry's spec.
 * `edges` are the INNER edges; the outer two are -inf/+inf.
 *   right=false  bins are [e(i-1), e(i))  -> index = #edges <= v
 *   right=true   bins are (e(i-1), e(i)]  -> index = #edges <  v
 * Null / NaN -> -1 (no bin), as pd.cut gives NaN. */
function obBinIndex(v, bins) {
  if (v === null || v === undefined || Number.isNaN(v)) return -1;
  const e = bins.edges;
  let lo = 0, hi = e.length;
  if (bins.right) {
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

    async init() {
      await Promise.all([this.loadRegistry(), this.loadMarketStatus()]);
    },

    async loadRegistry() {
      try {
        const r = await fetch('/api/oo-backtest/registry');
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        this.registry = (await r.json()).metrics || [];
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
    },

    /* ── registry-driven views ─────────────────────────────────────────── */

    filterMetrics() { return this.registry.filter(m => m.filter); },
    sectionMetrics() { return this.registry.filter(m => m.section); },

    hasColumn(m) { return this.presentColumns.includes(m.column); },

    binCount(m) { return m.bins ? m.bins.labels.length : null; },

    /* What a sidebar card shows about its column before the controls exist:
     * the loaded extent (range) or the distinct values (categorical). */
    columnSummary(m) {
      if (!this.loaded) return '';
      if (!this.hasColumn(m)) return 'no values in this log';
      const vals = OB_DATA.columns[m.column];
      if (m.type === 'range') {
        const x = obExtent(vals);
        return `${obFmt(x.min, m.format)} … ${obFmt(x.max, m.format)}  (${x.n} trades)`;
      }
      const d = obDistinct(vals);
      if (m.categories) {
        const byVal = Object.fromEntries(m.categories.map(c => [c.value, c.label]));
        return d.map(v => byVal[v] ?? String(v)).join(', ');
      }
      return d.join(', ');
    },

    /* A range section is skipped when its column is absent or all-null. */
    sectionState(m) {
      if (!this.loaded) return 'empty';
      return this.hasColumn(m) ? 'ready' : 'skipped';
    },

    sourceLabel() {
      if (!this.meta) return '';
      return this.meta.source === 'mesosim_json' ? 'Mesosim JSON' : 'Option Omega CSV';
    },

    loadedLabel() {
      if (!this.meta) return '';
      return `${this.meta.suggested_name} (${this.meta.n.toLocaleString()} trades, ` +
             `${this.meta.date_min} – ${this.meta.date_max})`;
    },

    /* ── market probe ──────────────────────────────────────────────────── */

    marketSummary() {
      if (this.marketLoading) return 'Checking…';
      if (this.marketError) return `Probe failed: ${this.marketError}`;
      if (!this.market) return '';
      const found = (this.market.candidates || []).filter(c => (c.coverage || []).length);
      // A probe that errored has not shown the table is absent — say which.
      const errs = this.market.errors || [];
      if (!found.length) return errs.length ? `Probe incomplete — ${errs[0]}` : 'No SPX/VIX daily table found';
      return found.map(c => `${c.db}.${c.table}`).join(', ');
    },

    coverageLine(row) {
      return (row.ticker ? row.ticker + ': ' : '') +
             `${row.first || '—'} → ${row.last || '—'}` + (row.n != null ? ` (${row.n})` : '');
    },
  }));
});
