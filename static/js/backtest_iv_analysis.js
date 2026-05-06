/* Backtest IV Analysis — Alpine.js app */

// ── Color helpers ──────────────────────────────────────────────────────────────

function _lerp(a, b, t) { return a + (b - a) * Math.max(0, Math.min(1, t)); }

const BLUE  = [41, 128, 245];
const PINK  = [220, 60, 155];
const GRAY  = [52, 52, 58];

function pnlColor(val, maxAbs) {
  if (val === null || val === undefined || isNaN(val)) return `rgb(${GRAY.join(',')})`;
  const t = maxAbs > 0 ? Math.max(-1, Math.min(1, val / maxAbs)) : 0;
  const base = t >= 0 ? BLUE : PINK;
  const s = Math.abs(t);
  return `rgb(${base.map((c, i) => Math.round(_lerp(GRAY[i], c, s))).join(',')})`;
}

function winRateColor(rate) {
  if (rate === null || rate === undefined) return `rgb(${GRAY.join(',')})`;
  const t = (rate - 0.5) * 2;   // -1 to +1, neutral at 0.5
  const base = t >= 0 ? BLUE : PINK;
  const s = Math.abs(t);
  return `rgb(${base.map((c, i) => Math.round(_lerp(GRAY[i], c, s))).join(',')})`;
}

function r2Color(val, maxVal) {
  if (val === null || val === undefined) return `rgb(${GRAY.join(',')})`;
  const s = maxVal > 0 ? Math.min(val / maxVal, 1) : 0;
  return `rgb(${BLUE.map((c, i) => Math.round(_lerp(GRAY[i], c, s))).join(',')})`;
}

function corrColor(val) {
  if (val === null || val === undefined) return `rgb(${GRAY.join(',')})`;
  const base = val >= 0 ? BLUE : PINK;
  const s = Math.min(Math.abs(val), 1);
  return `rgb(${base.map((c, i) => Math.round(_lerp(GRAY[i], c, s))).join(',')})`;
}

function textOnColor(rgb) {
  // Parse r,g,b and return black or white for readable contrast
  const m = rgb.match(/\d+/g);
  if (!m) return '#fff';
  const lum = (0.299 * m[0] + 0.587 * m[1] + 0.114 * m[2]) / 255;
  return lum > 0.5 ? '#111' : '#eee';
}

function fmtPnl(v) {
  if (v === null || v === undefined) return '—';
  return (v >= 0 ? '+' : '') + v.toFixed(0);
}

function fmtPct(v) {
  if (v === null || v === undefined) return '—';
  return (v * 100).toFixed(1) + '%';
}

function fmtR2(v) {
  if (v === null || v === undefined) return '—';
  return v.toFixed(3);
}


// ── Metric multi-select helper ─────────────────────────────────────────────────

function makeMultiSelect(maxCount) {
  return {
    selected: [],

    remove(col) {
      this.selected = this.selected.filter(c => c !== col);
    },

    clear() {
      this.selected = [];
    },
  };
}


// ── Main Alpine app ────────────────────────────────────────────────────────────

document.addEventListener('alpine:init', () => {
  Alpine.data('bivApp', () => ({

    // ── Global state ──
    uploads:          [],
    selectedId:       null,
    uploadInfo:       null,
    ivColumns:        [],
    loading:          false,
    globalError:      null,

    // Date range filter (defaults to upload's full range)
    fullRangeFrom:    null,
    fullRangeTo:      null,
    dateFrom:         null,
    dateTo:           null,
    filteredTradeCount: 0,

    // Bin mode: 'recompute' (default) or 'fixed' (use full-upload boundaries)
    binMode:          'recompute',

    // Metric filters. Each entry: {col, op, min?, max?, value?}.
    //   op = 'between'  → uses min and/or max (either may be null for one-sided)
    //   op = 'gte'/'lte'/'gt'/'lt'/'eq' → uses value
    // Applied in addition to the date filter; same invalidation behaviour.
    filters:          [],

    // Editor state for the +Filter popover (single editor instance —
    // open replaces any current draft). editIdx=null for "add", or the
    // index of an existing filter being modified. posTop/posLeft are
    // recomputed from the +Filter button's bounding rect on open so the
    // popover floats above the section cards instead of being clipped
    // by .ctrl-strip's overflow-x.
    filterEditor: {
      open:      false,
      editIdx:   null,
      draft:     { col: '', op: 'between', min: '', max: '', value: '' },
      colSearch: '',
      posTop:    0,
      posLeft:   0,
    },

    // Section open/closed
    open: { s0: true, s1: true, s2: true, s3: true, s4: false, s5: false, s6: false, s7: false, s8: true },

    // ── Summary stats strip (above S0) ──
    summary: {
      data: null, loading: false, error: null,
      _equityChart: null, _ddChart: null,
    },

    // ── Section 0: Correlation Overview ──
    s0: {
      target: 'pnl',
      data: null, loading: false, error: null,
      _chart: null,
    },

    // ── Section 1: 2D Heatmap ──
    s1: {
      metricA: '', metricB: '', nBuckets: 5, valueField: 'mean_pnl',
      minSampleN: 5,   // cells with n < minSampleN render as gray
      data: null, loading: false, error: null,
      _renderId: 0,    // bumped on each successful compute to force grid re-render
    },

    // ── Section 2: Pairwise ΔR² ──
    s2: {
      ms: null,   // makeMultiSelect(20) — init in init()
      target: 'pnl',
      data: null, loading: false, error: null,
    },

    // ── Section 3: Decile ──
    s3: {
      metric: '', nBuckets: 10,
      data: null, loading: false, error: null,
      _chart: null,
    },

    // ── Section 4: Conditional Slice ──
    s4: {
      fixMetric: '', fixBucket: 0, fixNBuckets: 5,
      varyMetric: '', varyNBuckets: 5,
      data: null, loading: false, error: null,
      _chart: null,
    },

    // ── Section 5: Distribution ──
    s5: {
      metric: '', bucketIdx: null, nBuckets: null,
      data: null, loading: false, error: null,
      showAll: true,
    },

    // ── Section 6: Time Stability ──
    s6: {
      metric: '', nWindows: 6,
      data: null, loading: false, error: null,
      _chart: null,
    },

    // ── Section 7: Feature Redundancy ──
    s7: {
      ms: null,   // makeMultiSelect(25)
      data: null, loading: false, error: null,
    },

    // ── Section 8: Top/Bottom ──
    s8: {
      data: null, loading: false, error: null,
    },

    // ── Init ──
    async init() {
      this.s2.ms = makeMultiSelect(20);
      this.s7.ms = makeMultiSelect(25);
      window.addEventListener('resize', () => this._onFilterPosResize());
      await this.loadUploads();
      // URL-supplied filters apply to whichever upload was auto-selected.
      // selectUpload() clears filters as part of the reset, so we restore here.
      const initial = this._readFiltersFromUrl();
      if (initial.length) {
        this.filters = initial;
        this.applyFilters();
      }
    },

    async loadUploads() {
      this.loading = true;
      try {
        const r = await fetch('/api/backtest-iv/uploads');
        this.uploads = await r.json();
        if (this.uploads.length > 0) {
          await this.selectUpload(this.uploads[0].id);
        }
      } catch (e) {
        this.globalError = e.message;
      } finally {
        this.loading = false;
      }
    },

    async selectUpload(id) {
      this.selectedId  = id;
      this.uploadInfo  = this.uploads.find(u => u.id === id) || null;
      this.ivColumns   = [];
      this.globalError = null;
      // Reset date range to the new upload's full range
      const df = this.uploadInfo?.date_from ? String(this.uploadInfo.date_from).slice(0, 10) : null;
      const dt = this.uploadInfo?.date_to   ? String(this.uploadInfo.date_to).slice(0, 10)   : null;
      this.fullRangeFrom = df;
      this.fullRangeTo   = dt;
      this.dateFrom      = df;
      this.dateTo        = dt;
      this.filteredTradeCount = this.uploadInfo?.trade_count || 0;
      // Reset all section data
      this.s0.data = null; this.s1.data = null; this.s2.data = null; this.s3.data = null;
      this.s4.data = null; this.s5.data = null; this.s6.data = null;
      this.s7.data = null; this.s8.data = null;
      // Reset multi-selects
      if (this.s2.ms) this.s2.ms.clear();
      if (this.s7.ms) this.s7.ms.clear();
      // Different upload, different feature columns — drop any active filters
      // and clear the URL's f= params.
      if (this.filters.length) {
        this.filters = [];
        this._writeFiltersToUrl();
      }

      try {
        const r   = await fetch(`/api/backtest-iv/${id}/columns`);
        const res = await r.json();
        this.ivColumns = res.iv_columns || [];
      } catch (e) {
        this.globalError = e.message;
        return;
      }
      // Auto-load summary, S0, S8
      this.loadSummary();
      this.loadCorrelationOverview();
      this.loadTopBottom();
    },

    // ── Date filter ──
    applyDateFilter() {
      // Guard against an inverted range
      if (this.dateFrom && this.dateTo && this.dateFrom > this.dateTo) {
        this.globalError = 'Invalid date range: from > to';
        return;
      }
      this.globalError = null;
      // Invalidate every section's cached result — user must recompute
      this.summary.data = null;
      this.s0.data = null; this.s1.data = null; this.s2.data = null; this.s3.data = null;
      this.s4.data = null; this.s5.data = null; this.s6.data = null;
      this.s7.data = null; this.s8.data = null;
      // Auto-loaded sections refresh; filteredTradeCount comes from S8 response.
      this.loadSummary();
      this.loadCorrelationOverview();
      this.loadTopBottom();
    },

    resetDateRange() {
      this.dateFrom = this.fullRangeFrom;
      this.dateTo   = this.fullRangeTo;
      this.applyDateFilter();
    },

    setBinMode(mode) {
      if (mode !== 'recompute' && mode !== 'fixed') return;
      if (this.binMode === mode) return;
      this.binMode = mode;
      // Same invalidation behavior as a date change
      this.applyDateFilter();
    },

    get _hasDateFilter() {
      return (this.dateFrom && this.dateFrom !== this.fullRangeFrom) ||
             (this.dateTo   && this.dateTo   !== this.fullRangeTo);
    },

    get _hasMetricFilters() {
      return this.filters.length > 0;
    },

    // ── Metric filters ──
    addFilter(clause) {
      const norm = this._normaliseFilter(clause);
      if (!norm) return;
      this.filters = [...this.filters, norm];
      this._writeFiltersToUrl();
      this.applyFilters();
    },

    updateFilter(idx, clause) {
      if (idx < 0 || idx >= this.filters.length) return;
      const norm = this._normaliseFilter(clause);
      if (!norm) return;
      const next = this.filters.slice();
      next[idx] = norm;
      this.filters = next;
      this._writeFiltersToUrl();
      this.applyFilters();
    },

    removeFilter(idx) {
      if (idx < 0 || idx >= this.filters.length) return;
      this.filters = this.filters.filter((_, i) => i !== idx);
      this._writeFiltersToUrl();
      this.applyFilters();
    },

    clearFilters() {
      if (!this.filters.length) return;
      this.filters = [];
      this._writeFiltersToUrl();
      this.applyFilters();
    },

    applyFilters() {
      // Same invalidation pattern as applyDateFilter — keeps the two filter
      // axes (date + metric) behaving identically from the user's POV.
      this.globalError = null;
      this.summary.data = null;
      this.s0.data = null; this.s1.data = null; this.s2.data = null; this.s3.data = null;
      this.s4.data = null; this.s5.data = null; this.s6.data = null;
      this.s7.data = null; this.s8.data = null;
      this.loadSummary();
      this.loadCorrelationOverview();
      this.loadTopBottom();
    },

    // ── Filter editor (popover) ──
    _computeFilterPos() {
      const popWidth = 340;
      const margin   = 8;
      const viewport = window.innerWidth ||
                       document.documentElement.clientWidth || 1024;

      const btn = this.$refs.filterBtn;
      if (!btn) return { posTop: 80, posLeft: margin };

      const r = btn.getBoundingClientRect();

      // Default placement: right-align popover with button (extends LEFT
      // from the button). Best when the button is near the right side of
      // the viewport — the popover stays adjacent without overflowing.
      let left = r.right - popWidth;

      // If right-alignment would overflow the LEFT edge of the viewport,
      // switch to left-alignment under the button (extends right). This
      // handles the case where the button itself is near the left edge.
      if (left < margin) {
        left = r.left;
      }

      // Final viewport clamp — never let the popover extend past either
      // edge of the viewport, regardless of the button's position.
      const maxLeft = viewport - popWidth - margin;
      if (left > maxLeft) left = Math.max(margin, maxLeft);
      if (left < margin) left = margin;

      return { posTop: r.bottom + 4, posLeft: Math.round(left) };
    },

    _onFilterPosResize() {
      // Re-anchor the popover when the viewport changes while it's open.
      // No-op when closed.
      if (!this.filterEditor.open) return;
      Object.assign(this.filterEditor, this._computeFilterPos());
    },

    openFilterEditor(idx = null) {
      const pos = this._computeFilterPos();
      if (typeof idx === 'number' && idx >= 0 && idx < this.filters.length) {
        const f = this.filters[idx];
        this.filterEditor = {
          open:      true,
          editIdx:   idx,
          draft: {
            col:   f.col,
            op:    f.op,
            min:   (f.op === 'between' && f.min !== null && f.min !== undefined) ? f.min : '',
            max:   (f.op === 'between' && f.max !== null && f.max !== undefined) ? f.max : '',
            value: (f.op !== 'between' && f.value !== null && f.value !== undefined) ? f.value : '',
          },
          colSearch: '',
          ...pos,
        };
      } else {
        this.filterEditor = {
          open:      true,
          editIdx:   null,
          draft:     { col: '', op: 'between', min: '', max: '', value: '' },
          colSearch: '',
          ...pos,
        };
      }
    },

    closeFilterEditor() {
      this.filterEditor.open = false;
    },

    submitFilterEditor() {
      if (!this._filterDraftValid()) return;
      const draft = this.filterEditor.draft;
      const idx   = this.filterEditor.editIdx;
      this.closeFilterEditor();
      if (idx !== null) {
        this.updateFilter(idx, draft);
      } else {
        this.addFilter(draft);
      }
    },

    _filterDraftValid() {
      return this._normaliseFilter(this.filterEditor.draft) !== null;
    },

    // ── Column grouping for the editor's dropdown ──
    _colFamily(c) {
      if (!c) return 'other';
      if (c.startsWith('vix_'))         return 'vix';
      if (c.startsWith('skew_'))        return 'skew';
      if (c.startsWith('term_'))        return 'term';
      if (c.startsWith('convex_'))      return 'convex';
      if (c.startsWith('iv_'))          return 'iv';
      if (c.startsWith('forward_'))     return 'forward';
      if (c.startsWith('portfolio_'))   return 'portfolio';
      if (['pnl','pnl_pct','is_win','days_in_trade'].includes(c)) return 'outcome';
      return 'other';
    },

    filteredColumnList() {
      const q = (this.filterEditor.colSearch || '').toLowerCase().trim();
      // Trade-side columns aren't in ivColumns; expose them as filterable too.
      const tradeCols = ['pnl', 'pnl_pct', 'is_win', 'days_in_trade'];
      const seen = new Set();
      const all  = [];
      for (const c of [...this.ivColumns, ...tradeCols]) {
        if (!seen.has(c)) { seen.add(c); all.push(c); }
      }
      const matched = q ? all.filter(c => c.toLowerCase().includes(q)) : all;
      const groups  = {};
      for (const c of matched) {
        const fam = this._colFamily(c);
        (groups[fam] ||= []).push(c);
      }
      const order = ['portfolio','outcome','vix','skew','term','convex','iv','forward','other'];
      return order
        .filter(f => groups[f]?.length)
        .map(f => ({ family: f, columns: groups[f] }));
    },

    // ── Chip rendering ──
    formatFilterChip(f) {
      if (!f || !f.col) return '';
      if (f.op === 'between') {
        const lo = (f.min === null || f.min === undefined) ? null : f.min;
        const hi = (f.max === null || f.max === undefined) ? null : f.max;
        if (lo !== null && hi !== null) return `${f.col} ∈ [${lo}, ${hi}]`;
        if (lo !== null) return `${f.col} ≥ ${lo}`;
        if (hi !== null) return `${f.col} ≤ ${hi}`;
        return f.col;
      }
      const sym = { gte: '≥', lte: '≤', gt: '>', lt: '<', eq: '=' }[f.op] || f.op;
      return `${f.col} ${sym} ${f.value}`;
    },

    _normaliseFilter(c) {
      if (!c || typeof c !== 'object') return null;
      const col = c.col;
      const op  = c.op;
      if (!col || !op) return null;
      if (op === 'between') {
        const lo = (c.min === '' || c.min === undefined) ? null
                 : (c.min === null ? null : Number(c.min));
        const hi = (c.max === '' || c.max === undefined) ? null
                 : (c.max === null ? null : Number(c.max));
        if (lo === null && hi === null) return null;
        if ((lo !== null && Number.isNaN(lo)) || (hi !== null && Number.isNaN(hi))) return null;
        return { col, op: 'between', min: lo, max: hi };
      }
      if (['gte', 'lte', 'gt', 'lt', 'eq'].includes(op)) {
        const v = (c.value === '' || c.value === undefined || c.value === null)
                ? null : Number(c.value);
        if (v === null || Number.isNaN(v)) return null;
        return { col, op, value: v };
      }
      return null;
    },

    // ── Filter URL serialization ──
    // Format: ?f=col:op:arg1:arg2 (repeated). Matches the backend parser.
    _filterToParam(f) {
      if (!f) return null;
      if (f.op === 'between') {
        const lo = (f.min === null || f.min === undefined) ? '' : String(f.min);
        const hi = (f.max === null || f.max === undefined) ? '' : String(f.max);
        return `${f.col}:between:${lo}:${hi}`;
      }
      if (['gte', 'lte', 'gt', 'lt', 'eq'].includes(f.op)) {
        return `${f.col}:${f.op}:${f.value}`;
      }
      return null;
    },

    _filterFromParam(s) {
      if (!s) return null;
      const parts = String(s).split(':');
      if (parts.length < 3) return null;
      const [col, op] = parts;
      if (op === 'between') {
        if (parts.length < 4) return null;
        const lo = parts[2] === '' ? null : Number(parts[2]);
        const hi = parts[3] === '' ? null : Number(parts[3]);
        if (lo === null && hi === null) return null;
        if ((lo !== null && Number.isNaN(lo)) || (hi !== null && Number.isNaN(hi))) return null;
        return { col, op: 'between', min: lo, max: hi };
      }
      if (['gte', 'lte', 'gt', 'lt', 'eq'].includes(op)) {
        const v = parts[2] === '' ? null : Number(parts[2]);
        if (v === null || Number.isNaN(v)) return null;
        return { col, op, value: v };
      }
      return null;
    },

    _readFiltersFromUrl() {
      try {
        const params = new URLSearchParams(window.location.search);
        return params.getAll('f')
          .map(s => this._filterFromParam(s))
          .filter(Boolean);
      } catch (_) {
        return [];
      }
    },

    _writeFiltersToUrl() {
      try {
        const params = new URLSearchParams(window.location.search);
        params.delete('f');
        for (const f of this.filters) {
          const s = this._filterToParam(f);
          if (s) params.append('f', s);
        }
        const qs = params.toString();
        const next = qs ? `${window.location.pathname}?${qs}` : window.location.pathname;
        window.history.replaceState(null, '', next);
      } catch (_) { /* URL APIs unavailable — silently no-op */ }
    },

    // ── Helpers ──
    _api(path, body) {
      const isPost = body !== undefined;
      let url = `/api/backtest-iv/${this.selectedId}/${path}`;
      let payload = body;

      if (isPost) {
        payload = { ...body };
        if (this.dateFrom) payload.date_from = this.dateFrom;
        if (this.dateTo)   payload.date_to   = this.dateTo;
        if (this.binMode === 'fixed') payload.bin_mode = 'fixed';
        if (this.filters.length) payload.filters = this.filters;
      } else {
        const params = new URLSearchParams();
        if (this.dateFrom) params.set('date_from', this.dateFrom);
        if (this.dateTo)   params.set('date_to',   this.dateTo);
        if (this.binMode === 'fixed') params.set('bin_mode', 'fixed');
        for (const f of this.filters) {
          const s = this._filterToParam(f);
          if (s) params.append('f', s);
        }
        const qs = params.toString();
        if (qs) url += '?' + qs;
      }

      return fetch(url, {
        method:  isPost ? 'POST' : 'GET',
        headers: isPost ? { 'Content-Type': 'application/json' } : {},
        body:    isPost ? JSON.stringify(payload) : undefined,
      }).then(async r => {
        if (!r.ok) {
          const e = await r.json().catch(() => ({}));
          throw new Error(e.detail || `HTTP ${r.status}`);
        }
        return r.json();
      });
    },

    // ── Summary strip (stat cards + equity / drawdown sparklines) ──
    async loadSummary() {
      if (!this.selectedId) return;
      this.summary.loading = true; this.summary.error = null;
      try {
        const data = await this._api('summary', {});
        this.summary.data = data;
        if (typeof data?.n === 'number') this.filteredTradeCount = data.n;
        await this.$nextTick();
        this._renderSummaryCharts();
      } catch (e) {
        this.summary.error = e.message;
      } finally {
        this.summary.loading = false;
      }
    },

    _renderSummaryCharts() {
      this.summary._equityChart = this._destroyChart(this.summary._equityChart);
      this.summary._ddChart     = this._destroyChart(this.summary._ddChart);

      const eq = this.summary.data?.equity_curve   || [];
      const dd = this.summary.data?.drawdown_curve || [];
      if (!eq.length) return;

      const sparkOpts = {
        responsive: true, maintainAspectRatio: false, animation: false,
        plugins: {
          legend: { display: false },
          tooltip: {
            backgroundColor: 'rgba(20,20,20,0.95)',
            borderColor: '#444', borderWidth: 1,
            displayColors: false,
            callbacks: {
              title: items => items[0].label,
              label: ctx => (ctx.parsed.y >= 0 ? '+' : '') + ctx.parsed.y.toFixed(0),
            },
          },
        },
        scales: {
          x: { display: false },
          y: { ticks: { color: '#666', font: { size: 9 }, maxTicksLimit: 3 },
                grid: { color: 'rgba(255,255,255,0.04)' }, border: { display: false } },
        },
        elements: { point: { radius: 0, hitRadius: 6, hoverRadius: 3 } },
      };

      const eqEl = document.getElementById('summary-equity-chart');
      if (eqEl) {
        this.summary._equityChart = new Chart(eqEl.getContext('2d'), {
          type: 'line',
          data: {
            labels: eq.map(p => p.date),
            datasets: [{
              data: eq.map(p => p.equity),
              borderColor: 'rgba(52,152,219,0.95)',
              backgroundColor: 'rgba(52,152,219,0.10)',
              borderWidth: 1.6, fill: true, tension: 0.18,
            }],
          },
          options: sparkOpts,
        });
      }

      const ddEl = document.getElementById('summary-drawdown-chart');
      if (ddEl) {
        this.summary._ddChart = new Chart(ddEl.getContext('2d'), {
          type: 'line',
          data: {
            labels: dd.map(p => p.date),
            datasets: [{
              data: dd.map(p => p.drawdown),
              borderColor: 'rgba(220,60,155,0.95)',
              backgroundColor: 'rgba(220,60,155,0.12)',
              borderWidth: 1.6, fill: true, tension: 0.18,
            }],
          },
          options: sparkOpts,
        });
      }
    },

    // ── Section 0: Correlation Overview ──
    async loadCorrelationOverview() {
      if (!this.selectedId) return;
      this.s0.loading = true; this.s0.error = null;
      try {
        // Built inline since the endpoint takes a non-date 'target' query param
        const params = new URLSearchParams();
        params.set('target', this.s0.target);
        if (this.dateFrom) params.set('date_from', this.dateFrom);
        if (this.dateTo)   params.set('date_to',   this.dateTo);
        for (const f of this.filters) {
          const s = this._filterToParam(f);
          if (s) params.append('f', s);
        }
        const r = await fetch(
          `/api/backtest-iv/${this.selectedId}/correlation-overview?${params}`);
        if (!r.ok) {
          const e = await r.json().catch(() => ({}));
          throw new Error(e.detail || `HTTP ${r.status}`);
        }
        this.s0.data = await r.json();
        await this.$nextTick();
        this._renderS0Chart();
      } catch (e) {
        this.s0.error = e.message;
      } finally {
        this.s0.loading = false;
      }
    },

    _renderS0Chart() {
      this.s0._chart = this._destroyChart(this.s0._chart);
      const el = document.getElementById('s0-chart');
      if (!el || !this.s0.data?.metrics?.length) return;
      const metrics = this.s0.data.metrics;
      const target  = this.s0.data.target;
      const labels  = metrics.map(m => m.metric);
      const data    = metrics.map(m => m.r);
      const colors  = data.map(v =>
        v >= 0 ? 'rgba(41,128,245,0.85)' : 'rgba(220,60,155,0.85)');

      this.s0._chart = new Chart(el.getContext('2d'), {
        type: 'bar',
        data: {
          labels,
          datasets: [{
            data,
            backgroundColor: colors,
            borderWidth: 0,
            categoryPercentage: 0.95,
            barPercentage: 0.95,
          }],
        },
        options: {
          responsive:           true,
          maintainAspectRatio:  false,
          animation:            false,
          plugins: {
            legend:  { display: false },
            tooltip: {
              backgroundColor: 'rgba(20,20,20,0.95)',
              borderColor: '#444', borderWidth: 1,
              callbacks: {
                title: items => metrics[items[0].dataIndex].metric,
                label: ctx => {
                  const m   = metrics[ctx.dataIndex];
                  const tgt = target === 'pnl' ? 'P&L' : 'Win';
                  return `r(${tgt}) = ${m.r >= 0 ? '+' : ''}${m.r.toFixed(4)}  (n=${m.n})`;
                },
              },
            },
          },
          scales: {
            x: {
              ticks: {
                color: '#888', font: { size: 9, family: 'monospace' },
                maxRotation: 90, minRotation: 90, autoSkip: false,
              },
              grid:   { display: false },
              border: { color: '#333' },
            },
            y: {
              min: -1, max: 1,
              ticks: { color: '#888', font: { size: 10 }, stepSize: 0.25 },
              grid:  { color: 'rgba(255,255,255,0.06)' },
              border: { color: '#333' },
            },
          },
        },
      });
    },

    _destroyChart(ref) {
      if (ref) { try { ref.destroy(); } catch (_) {} }
      return null;
    },

    // ── Section 1: Heatmap ──
    async loadHeatmap() {
      if (!this.s1.metricA || !this.s1.metricB || !this.selectedId) return;
      this.s1.loading = true; this.s1.error = null;
      try {
        const data = await this._api('heatmap', {
          metric_a: this.s1.metricA, metric_b: this.s1.metricB, n_buckets: this.s1.nBuckets,
        });
        this.s1._renderId = (this.s1._renderId + 1) % 100000;
        this.s1.data = data;
      } catch (e) { this.s1.error = e.message; }
      finally { this.s1.loading = false; }
    },

    s1CellBg(cell) {
      if (!this.s1.data || !cell) return '#141414';
      const n    = cell.n || 0;
      const minN = this.s1.minSampleN || 0;

      // Tier 1: empty cell — nearly the page background, cell almost vanishes
      if (n === 0) return '#141414';

      // Tier 2: low-sample (0 < n < threshold) — crosshatch on dark base
      if (n < minN) {
        return 'repeating-linear-gradient(45deg, #2e2e2e 0 4px, transparent 4px 8px),'
             + 'repeating-linear-gradient(-45deg, #2e2e2e 0 4px, transparent 4px 8px),'
             + '#1c1c1c';
      }

      // Tier 3: meets threshold — full gradient, scaled across visible cells only
      const vf = this.s1.valueField;
      const visible = this.s1.data.cells.flat().filter(c => (c.n || 0) >= minN);
      if (vf === 'mean_pnl') {
        const maxAbs = Math.max(
          ...visible.map(c => Math.abs(c.mean_pnl || 0)).filter(v => v > 0),
          0,
        );
        return pnlColor(cell.mean_pnl, maxAbs);
      }
      if (vf === 'win_rate') return winRateColor(cell.win_rate);
      const maxN = Math.max(...visible.map(c => c.n || 0), 0);
      return r2Color(cell.n, maxN);
    },

    s1CellText(cell) {
      if (!cell) return '';
      const vf = this.s1.valueField;
      if (vf === 'mean_pnl')  return fmtPnl(cell.mean_pnl);
      if (vf === 'win_rate')  return fmtPct(cell.win_rate);
      return cell.n ?? '—';
    },

    s1CellFg(cell) {
      if (!cell) return '#666';
      const n    = cell.n || 0;
      const minN = this.s1.minSampleN || 0;
      if (n === 0)    return '#333';   // very dim text on near-black
      if (n < minN)   return '#999';   // muted on hatch
      // textOnColor parses rgb(...) digits — only valid for the gradient tier
      return textOnColor(this.s1CellBg(cell));
    },

    // ── Section 2: Pairwise ΔR² ──
    async loadDeltaR2() {
      const metrics = this.s2.ms?.selected || [];
      if (metrics.length < 2 || !this.selectedId) return;
      this.s2.loading = true; this.s2.error = null;
      try {
        this.s2.data = await this._api('delta-r2', { metrics, target: this.s2.target });
      } catch (e) { this.s2.error = e.message; }
      finally { this.s2.loading = false; }
    },

    s2CellBg(val, isDialog) {
      if (val === null || val === undefined) return `rgb(${GRAY.join(',')})`;
      if (!this.s2.data) return `rgb(${GRAY.join(',')})`;
      if (isDialog) return corrColor(val);
      const flat = this.s2.data.matrix.flat().filter(v => v !== null);
      const maxV = Math.max(...flat.filter((_, i) => true));
      return r2Color(val, maxV > 0 ? maxV : 0.01);
    },

    // ── Section 3: Decile ──
    async loadDecile() {
      if (!this.s3.metric || !this.selectedId) return;
      this.s3.loading = true; this.s3.error = null;
      try {
        this.s3.data = await this._api('decile', { metric: this.s3.metric, n_buckets: this.s3.nBuckets });
        await this.$nextTick();
        this._renderDecileChart();
      } catch (e) { this.s3.error = e.message; }
      finally { this.s3.loading = false; }
    },

    _renderDecileChart() {
      this.s3._chart = this._destroyChart(this.s3._chart);
      const el = document.getElementById('s3-chart');
      if (!el || !this.s3.data?.buckets) return;
      const buckets  = this.s3.data.buckets;
      const labels   = buckets.map(b => b.label);
      const pnls     = buckets.map(b => b.mean_pnl ?? 0);
      const maxAbs   = Math.max(...pnls.map(Math.abs), 1);
      const colors   = pnls.map(v => pnlColor(v, maxAbs));
      const winRates = buckets.map(b => (b.win_rate ?? 0) * 100);

      this.s3._chart = new Chart(el.getContext('2d'), {
        type: 'bar',
        data: {
          labels,
          datasets: [
            {
              label: 'Mean P&L',
              data: pnls,
              backgroundColor: colors,
              yAxisID: 'y',
            },
            {
              label: 'Win Rate %',
              data: winRates,
              type: 'line',
              borderColor: 'rgba(200,200,200,0.6)',
              borderWidth: 1.5,
              pointRadius: 3,
              pointBackgroundColor: winRates.map(v => winRateColor(v / 100)),
              fill: false,
              yAxisID: 'y2',
            },
          ],
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: { legend: { labels: { color: '#aaa', font: { size: 11 } } } },
          scales: {
            x: { ticks: { color: '#888', font: { size: 10 } }, grid: { color: '#333' } },
            y:  { ticks: { color: '#aaa', font: { size: 10 } }, grid: { color: '#333' },
                  title: { display: true, text: 'Mean P&L', color: '#aaa', font: { size: 10 } } },
            y2: { position: 'right', min: 0, max: 100,
                  ticks: { color: '#aaa', font: { size: 10 } }, grid: { drawOnChartArea: false },
                  title: { display: true, text: 'Win Rate %', color: '#aaa', font: { size: 10 } } },
          },
        },
      });
    },

    // ── Section 4: Conditional Slice ──
    async loadConditionalSlice() {
      if (!this.s4.fixMetric || !this.s4.varyMetric || !this.selectedId) return;
      this.s4.loading = true; this.s4.error = null;
      try {
        this.s4.data = await this._api('conditional-slice', {
          fix_metric:     this.s4.fixMetric,
          fix_bucket:     parseInt(this.s4.fixBucket),
          fix_n_buckets:  parseInt(this.s4.fixNBuckets),
          vary_metric:    this.s4.varyMetric,
          vary_n_buckets: parseInt(this.s4.varyNBuckets),
        });
        await this.$nextTick();
        this._renderSliceChart();
      } catch (e) { this.s4.error = e.message; }
      finally { this.s4.loading = false; }
    },

    _renderSliceChart() {
      this.s4._chart = this._destroyChart(this.s4._chart);
      const el = document.getElementById('s4-chart');
      if (!el || !this.s4.data?.buckets) return;
      const buckets = this.s4.data.buckets;
      const labels  = buckets.map(b => b.label);
      const pnls    = buckets.map(b => b.mean_pnl ?? 0);
      const maxAbs  = Math.max(...pnls.map(Math.abs), 1);
      const colors  = pnls.map(v => pnlColor(v, maxAbs));

      this.s4._chart = new Chart(el.getContext('2d'), {
        type: 'bar',
        data: {
          labels,
          datasets: [{
            label: 'Mean P&L within slice',
            data: pnls,
            backgroundColor: colors,
          }],
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: { legend: { display: false } },
          scales: {
            x: { ticks: { color: '#888', font: { size: 10 } }, grid: { color: '#333' } },
            y: { ticks: { color: '#aaa', font: { size: 10 } }, grid: { color: '#333' } },
          },
        },
      });
    },

    // ── Section 5: Distribution ──
    async loadDistribution() {
      if (!this.selectedId) return;
      this.s5.loading = true; this.s5.error = null;
      try {
        const body = this.s5.showAll
          ? {}
          : { metric: this.s5.metric, bucket_index: (this.s5.bucketIdx || 1) - 1, n_buckets: this.s5.nBuckets };
        this.s5.data = await this._api('distribution', body);
      } catch (e) { this.s5.error = e.message; }
      finally { this.s5.loading = false; }
    },

    s5BoxLeft(p)   { return this._s5Pct(p);  },
    s5BoxWidth(lo, hi) {
      if (!this.s5.data) return '0%';
      const r = this.s5.data.max - this.s5.data.min;
      if (r === 0) return '0%';
      return ((hi - lo) / r * 100).toFixed(2) + '%';
    },
    _s5Pct(v) {
      if (!this.s5.data) return '0%';
      const r = this.s5.data.max - this.s5.data.min;
      if (r === 0) return '0%';
      return ((v - this.s5.data.min) / r * 100).toFixed(2) + '%';
    },
    s5MedianLeft() { return this._s5Pct(this.s5.data?.p50 ?? 0); },
    s5MeanLeft()   { return this._s5Pct(this.s5.data?.mean ?? 0); },

    // ── Section 6: Time Stability ──
    async loadTimeStability() {
      if (!this.s6.metric || !this.selectedId) return;
      this.s6.loading = true; this.s6.error = null;
      try {
        this.s6.data = await this._api('time-stability', {
          metric: this.s6.metric, n_windows: parseInt(this.s6.nWindows),
        });
        await this.$nextTick();
        this._renderStabilityChart();
      } catch (e) { this.s6.error = e.message; }
      finally { this.s6.loading = false; }
    },

    _renderStabilityChart() {
      this.s6._chart = this._destroyChart(this.s6._chart);
      const el = document.getElementById('s6-chart');
      if (!el || !this.s6.data?.periods) return;
      const periods = this.s6.data.periods;
      const labels  = periods.map(p => p.label + '\n' + p.date_from.slice(0, 7));
      const rs      = periods.map(p => p.r);
      const pnls    = periods.map(p => p.mean_pnl);
      const maxAbsPnl = Math.max(...pnls.map(Math.abs), 1);

      this.s6._chart = new Chart(el.getContext('2d'), {
        data: {
          labels,
          datasets: [
            {
              type: 'line',
              label: 'Pearson r',
              data: rs,
              borderColor: rs.map(v => v >= 0 ? 'rgb(41,128,245)' : 'rgb(220,60,155)'),
              borderWidth: 2,
              pointBackgroundColor: rs.map(v => v >= 0 ? 'rgb(41,128,245)' : 'rgb(220,60,155)'),
              pointRadius: 5,
              fill: false,
              yAxisID: 'yr',
              segment: { borderColor: ctx => rs[ctx.p0DataIndex] >= 0 ? 'rgb(41,128,245)' : 'rgb(220,60,155)' },
            },
            {
              type: 'bar',
              label: 'Mean P&L',
              data: pnls,
              backgroundColor: pnls.map(v => pnlColor(v, maxAbsPnl)),
              yAxisID: 'yp',
            },
          ],
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: { legend: { labels: { color: '#aaa', font: { size: 11 } } } },
          scales: {
            x:  { ticks: { color: '#888', font: { size: 10 } }, grid: { color: '#333' } },
            yr: { position: 'left', min: -1, max: 1,
                  ticks: { color: '#aaa', font: { size: 10 } }, grid: { color: '#333' },
                  title: { display: true, text: 'r', color: '#aaa', font: { size: 10 } } },
            yp: { position: 'right',
                  ticks: { color: '#aaa', font: { size: 10 } }, grid: { drawOnChartArea: false },
                  title: { display: true, text: 'Mean P&L', color: '#aaa', font: { size: 10 } } },
          },
        },
      });
    },

    // ── Section 7: Feature Correlation ──
    async loadFeatureCorr() {
      const metrics = this.s7.ms?.selected || [];
      if (metrics.length < 2 || !this.selectedId) return;
      this.s7.loading = true; this.s7.error = null;
      try {
        this.s7.data = await this._api('feature-correlation', { metrics });
      } catch (e) { this.s7.error = e.message; }
      finally { this.s7.loading = false; }
    },

    // ── Section 8: Top/Bottom ──
    async loadTopBottom() {
      if (!this.selectedId) return;
      this.s8.loading = true; this.s8.error = null;
      try {
        const data = await this._api('top-bottom');
        this.s8.data = data;
        if (typeof data?.n_trades === 'number') {
          this.filteredTradeCount = data.n_trades;
        }
      } catch (e) {
        this.s8.error = e.message;
        this.filteredTradeCount = 0;
      }
      finally { this.s8.loading = false; }
    },

    // ── UI helpers ──
    toggleSection(s) { this.open[s] = !this.open[s]; },

    get noUpload() { return !this.selectedId; },

  }));
});
