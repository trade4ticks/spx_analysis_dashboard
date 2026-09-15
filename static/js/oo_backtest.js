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
 * metric coverage. Filters, stats and charts are still labelled stubs.
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

    // Which ratio basis the sections read: entry-time bars (default) or the
    // entry date's daily closes. Temporary -- one basis is deleted, and this
    // toggle with it, once both have been looked at on a real log.
    ratioBasis: 'entry',

    async init() {
      await Promise.all([this.loadRegistry(), this.loadMarketStatus()]);
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
      return `Data starts ${m.minDate} — filtering on this drops ${k.toLocaleString()} earlier trade${k === 1 ? '' : 's'}`;
    },

    binCount(m) { return m.bins ? m.bins.labels.length : null; },

    /* What a sidebar card shows about its column before the controls exist:
     * the loaded extent (range) or the distinct values (categorical). */
    columnSummary(m) {
      if (!this.loaded) return '';
      if (!this.hasColumn(m)) return 'no values in this log';
      const vals = OB_DATA.columns[this.metricColumn(m)];
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
      return `${this.meta.suggested_name} (${this.meta.n.toLocaleString()} trades, ` +
             `${this.meta.date_min} – ${this.meta.date_max})`;
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

    zeroDayLines() {
      const z = this.market && this.market.zero_days;
      if (!z) return [];
      const out = [`${z.zero_filled_days} zero-filled days excluded (${z.zero_filled_weekend} weekend, ${z.zero_filled_weekdays} weekday)`];
      if (z.zero_filled_weekdays) out.push(`weekday: ${z.zero_filled_weekday_dates.join(', ')}`);
      const bySeries = o => Object.entries(o || {}).map(([k, v]) => `${k.toUpperCase()} ${v}`).join(', ');
      out.push(`${z.partial_days} trading days with some invalid bars` +
               (z.partial_days ? ` — zero: ${bySeries(z.partial_zero_bars_by_series)}; NaN: ${bySeries(z.partial_nan_bars_by_series)}` : ''));
      for (const d of (z.partial_sample || []).slice(0, 10)) {
        out.push(`  ${d.date}: ${d.valid_bars} valid bars; zero/NaN SPX ${d.spx_zero}/${d.spx_nan}, ` +
                 `VIX ${d.vix_zero}/${d.vix_nan}, VIX3M ${d.vix3m_zero}/${d.vix3m_nan}, VIX9D ${d.vix9d_zero}/${d.vix9d_nan}`);
      }
      return out;
    },

    coverageLines() {
      const c = (this.market && this.market.coverage) || {};
      return Object.entries(c).map(([s, d]) => `${s.toUpperCase()} from ${d || '—'}`);
    },
  }));
});
