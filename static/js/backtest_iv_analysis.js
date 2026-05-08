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


// ── Today's-value Chart.js plugin ─────────────────────────────────────────────
// Draws a dashed gold line + × at the precise fractional x-position of today's
// value within the matching decile bucket. Skipped silently when today's value
// is null or outside all bucket ranges.
const _todayMarkerPlugin = {
  id: 'todayMarker',
  afterDatasetsDraw(chart, _args, opts) {
    if (!opts || opts.todayValue == null || !Array.isArray(opts.buckets)) return;
    const v = Number(opts.todayValue);
    if (!Number.isFinite(v)) return;
    const buckets = opts.buckets;

    let bucketIdx = -1;
    for (let i = 0; i < buckets.length; i++) {
      const b = buckets[i];
      if (b.x_min == null || b.x_max == null) continue;
      const lo = (i === 0)                ? -Infinity : b.x_min;
      const hi = (i === buckets.length-1) ?  Infinity : b.x_max;
      if (v >= lo && v <= hi) { bucketIdx = i; break; }
    }
    if (bucketIdx < 0) return;
    const b   = buckets[bucketIdx];
    const den = (b.x_max - b.x_min) || 1e-9;
    const frac = Math.max(0, Math.min(1, (v - b.x_min) / den));

    const xScale = chart.scales.x;
    if (!xScale) return;
    // For category scales getPixelForValue(idx) returns the bucket center.
    const centerPx = xScale.getPixelForValue(bucketIdx);
    let cellWidth;
    if (buckets.length > 1) {
      const next = bucketIdx === 0
        ? xScale.getPixelForValue(1) - xScale.getPixelForValue(0)
        : xScale.getPixelForValue(bucketIdx) - xScale.getPixelForValue(bucketIdx - 1);
      cellWidth = Math.abs(next);
    } else {
      cellWidth = xScale.right - xScale.left;
    }
    const xPx = centerPx + (frac - 0.5) * cellWidth;
    const top = chart.chartArea.top;
    const bot = chart.chartArea.bottom;

    const ctx = chart.ctx;
    ctx.save();
    ctx.lineWidth   = 2;
    ctx.strokeStyle = 'rgba(220,220,220,0.85)';   // light gray
    ctx.setLineDash([5, 4]);
    ctx.beginPath();
    ctx.moveTo(xPx, top);
    ctx.lineTo(xPx, bot);
    ctx.stroke();
    ctx.setLineDash([]);
    // Filled dot at the top of the line — precise point indicator
    ctx.fillStyle = '#e6e6e6';
    ctx.beginPath();
    ctx.arc(xPx, top + 4, 4, 0, 2 * Math.PI);
    ctx.fill();
    ctx.strokeStyle = 'rgba(0,0,0,0.6)';
    ctx.lineWidth   = 1;
    ctx.stroke();
    // Small numeric label above the dot, kept in-bounds horizontally
    ctx.fillStyle    = 'rgba(220,220,220,0.95)';
    ctx.font         = '11px monospace';
    ctx.textAlign    = 'center';
    ctx.textBaseline = 'bottom';
    const label = Math.abs(v) >= 1000 ? Math.round(v).toString()
                  : Math.abs(v) >= 1   ? v.toFixed(2)
                  :                       v.toFixed(4);
    const labelX = Math.max(xScale.left + 24,
                            Math.min(xScale.right - 24, xPx));
    ctx.fillText('today ' + label, labelX, top - 2);
    ctx.restore();
  },
};


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

    // Bin mode: 'fixed' (default — use full-upload boundaries so deciles
    // mean the same thing across filter slices) or 'recompute' (boundaries
    // from the filtered subset).
    binMode:          'fixed',

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

    // Lazy-cached per-column distribution stats from /column-stats. Keyed
    // by column name. Each entry is either:
    //   undefined — never requested
    //   null      — fetch in flight
    //   false     — fetch failed (column has no numeric values)
    //   object    — {min, max, p01, p05, p50, p95, p99, n}
    columnStats: {},

    // Per-column semantic metadata from /api/meta/columns-catalog. Keyed by
    // column_name. Loaded once on init. Used to group columns in dropdowns,
    // show descriptions on hover, and format values by their units.
    catalog: {},

    // Lazy-cached "today's value" lookups from /api/meta/today-value.
    //   undefined  → not requested
    //   null       → fetch in flight
    //   {value:n}  → success
    //   {value:null} → no surface_metrics_core value (or trade-derived col)
    todayValues: {},

    // Shared searchable picker for the section metric dropdowns. Only one
    // instance — opens for whichever section's button was clicked, writes
    // the picked value back through `path` (dot-notation, e.g. 's3.metric'),
    // closes on selection or click-outside.
    // multiPath: when set, picker is in multi-select mode — picks push into
    // the array at multiPath instead of replacing path, and the popover
    // stays open so the user can pick several metrics in a row.
    metricPicker: {
      open:      false,
      path:      null,
      multiPath: null,
      search:    '',
      posTop:    0,
      posLeft:   0,
    },

    // Active drag state for the slider chip. null when not dragging.
    sliderDrag: null,

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
      // Sort/emphasize method: 'pearson' | 'spearman' | 'consensus' | 'mi'.
      // All three signed bars (Pearson/Spearman) and the MI bar always render;
      // this controls which one drives sort order and gets full-opacity.
      method: 'pearson',
      // Toggled by the fullscreen icon. When true the section is fixed to
      // the viewport and the chart wrapper grows to fill remaining height.
      fullscreen: false,
      data: null, loading: false, error: null,
      _chart: null,
    },

    // ── Section 1: 2D Heatmap ──
    s1: {
      metricA: '', metricB: '', nBuckets: 5, valueField: 'mean_pnl',
      minSampleN: 5,   // cells with n < minSampleN render as gray
      data: null, loading: false, error: null,
      _renderId: 0,    // bumped on each successful compute to force grid re-render
      // Click-toggled cell selection that drives S3 multi-metric filter.
      // Each entry: {ia, ib}. Cleared on metric change.
      selectedCells: [],
      // Last 10 trading days (incl. today) of (metricA, metricB) values.
      // Each entry: {date, a, b}. Loaded from /api/meta/value-trail.
      trail: [],
      // Pre-computed pixel coordinates for the polyline overlay. Recomputed
      // after each heatmap+trail load on a delayed nextTick so the table
      // has actually been laid out before we measure cells.
      trailPoints: '',
    },

    // Global toggle for live (today + trail) overlays. When false, hides
    // S0/S3 today markers and the heatmap × + trail.
    liveOverlays: true,

    // ── Section 2: Pairwise ΔR² ──
    s2: {
      ms: null,   // makeMultiSelect(20) — init in init()
      target: 'pnl',
      data: null, loading: false, error: null,
    },

    // ── Section 3: Decile ──
    // metrics is an array — each entry gets its own decile chart stacked
    // vertically. dataByMetric / charts are keyed by metric name.
    // _loadToken: increments on every loadDecile() so a stale fetch
    //   can't overwrite newer state if the user clicks faster than the
    //   server responds.
    // Canvas elements are created imperatively inside stable wrapper divs
    // (#s3-wrap-<safe>) — no canvas reuse across renders, no Alpine ↔
    // Chart.js timing dependency.
    s3: {
      metrics: [''], nBuckets: 10,
      dataByMetric: {},
      charts: {},
      _loadToken: 0,
      loading: false, error: null,
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
      window.addEventListener('keydown', (e) => {
        if (e.key === 'Escape' && this.s0.fullscreen) this.toggleS0Fullscreen();
      });
      this._loadCatalog();   // fire-and-forget; UI gracefully degrades to prefix matching until loaded
      await this.loadUploads();
      // Restore saved defaults on top of the freshly-selected upload.
      // URL-supplied filters take precedence over saved-default filters
      // since they're explicit per-page-load context.
      const restoredDefaults = this.loadDefaults(/*silent*/ true);
      const initial = this._readFiltersFromUrl();
      if (initial.length) {
        this.filters = initial;
        this._ensureFilterStats();
        this.applyFilters();
      } else if (restoredDefaults) {
        this.applyFilters();
      }
    },

    // ── Save / load / clear defaults (localStorage) ──
    // Captures every metric pick, filter, bucket count, target/method
    // toggle, and panel open state. Date range is intentionally excluded
    // since it's tied to the specific upload's available range and we
    // want it auto-filled per upload.
    _DEFAULTS_KEY: 'biv:defaults:v1',

    saveDefaults() {
      const d = {
        savedAt: new Date().toISOString(),
        binMode: this.binMode,
        open:    { ...this.open },
        filters: this.filters.map(f => ({ ...f })),
        s0: { target: this.s0.target, method: this.s0.method },
        s1: {
          metricA:    this.s1.metricA,
          metricB:    this.s1.metricB,
          nBuckets:   this.s1.nBuckets,
          valueField: this.s1.valueField,
          minSampleN: this.s1.minSampleN,
        },
        s2: {
          selected: [...(this.s2.ms?.selected || [])],
          target:   this.s2.target,
        },
        s3: {
          metrics:  [...this.s3.metrics],
          nBuckets: this.s3.nBuckets,
        },
        s4: {
          fixMetric:    this.s4.fixMetric,
          fixBucket:    this.s4.fixBucket,
          fixNBuckets:  this.s4.fixNBuckets,
          varyMetric:   this.s4.varyMetric,
          varyNBuckets: this.s4.varyNBuckets,
        },
        s5: {
          metric:    this.s5.metric,
          bucketIdx: this.s5.bucketIdx,
          nBuckets:  this.s5.nBuckets,
          showAll:   this.s5.showAll,
        },
        s6: { metric: this.s6.metric, nWindows: this.s6.nWindows },
        s7: { selected: [...(this.s7.ms?.selected || [])] },
      };
      try {
        localStorage.setItem(this._DEFAULTS_KEY, JSON.stringify(d));
        this._flashMessage('Defaults saved.');
      } catch (e) {
        this.globalError = 'Save failed: ' + e.message;
      }
    },

    loadDefaults(silent = false) {
      let d;
      try {
        const raw = localStorage.getItem(this._DEFAULTS_KEY);
        if (!raw) {
          if (!silent) this._flashMessage('No defaults saved yet.');
          return false;
        }
        d = JSON.parse(raw);
      } catch (e) {
        if (!silent) this.globalError = 'Load failed: ' + e.message;
        return false;
      }
      if (d.binMode) this.binMode = d.binMode;
      if (d.open) Object.assign(this.open, d.open);
      if (d.filters) {
        this.filters = d.filters;
        this._writeFiltersToUrl();
        this._ensureFilterStats();
      }
      if (d.s0) Object.assign(this.s0, d.s0);
      if (d.s1) Object.assign(this.s1, d.s1);
      if (d.s2 && this.s2.ms) {
        this.s2.ms.selected = d.s2.selected || [];
        if (d.s2.target) this.s2.target = d.s2.target;
      }
      if (d.s3) {
        this.s3.metrics = (d.s3.metrics?.length ? [...d.s3.metrics] : ['']);
        if (d.s3.nBuckets) this.s3.nBuckets = d.s3.nBuckets;
      }
      if (d.s4) Object.assign(this.s4, d.s4);
      if (d.s5) Object.assign(this.s5, d.s5);
      if (d.s6) Object.assign(this.s6, d.s6);
      if (d.s7 && this.s7.ms) this.s7.ms.selected = d.s7.selected || [];

      // Pre-fetch today's value for any newly-loaded surface metric so
      // markers show as soon as the section computes.
      const allCols = [
        this.s1.metricA, this.s1.metricB,
        this.s4.fixMetric, this.s4.varyMetric,
        this.s5.metric, this.s6.metric,
        ...this.s3.metrics,
      ].filter(Boolean);
      for (const c of allCols) this.ensureTodayValue(c);

      if (!silent) {
        const when = d.savedAt ? ' (saved ' + d.savedAt.slice(0, 16).replace('T', ' ') + ')' : '';
        this._flashMessage('Defaults loaded' + when + '.');
      }
      return true;
    },

    clearDefaults() {
      try {
        localStorage.removeItem(this._DEFAULTS_KEY);
        this._flashMessage('Defaults cleared.');
      } catch (e) {
        this.globalError = 'Clear failed: ' + e.message;
      }
    },

    hasSavedDefaults() {
      try {
        return !!localStorage.getItem(this._DEFAULTS_KEY);
      } catch (_) { return false; }
    },

    _flashMessage(msg) {
      this.globalError = msg;
      const original = msg;
      setTimeout(() => {
        if (this.globalError === original) this.globalError = null;
      }, 2200);
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
      this.columnStats = {};

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
      this._ensureFilterStats();
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
      this._ensureFilterStats();
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
      // Re-anchor open popovers when the viewport changes. No-op when closed.
      if (this.filterEditor.open) {
        Object.assign(this.filterEditor, this._computeFilterPos());
      }
      if (this.metricPicker.open) {
        // Picker doesn't have its anchor element cached, so just close on
        // resize — simpler and the user can reopen.
        this.metricPicker.open = false;
      }
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
        this._ensureOneColStat(f.col);
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
    // ── Today's value lookups (live snapshot of a surface metric) ──
    async _fetchTodayValue(col) {
      if (!col) return;
      this.todayValues[col] = null;
      try {
        const r = await fetch('/api/meta/today-value?col=' + encodeURIComponent(col));
        if (r.ok) {
          this.todayValues[col] = await r.json();
        } else {
          this.todayValues[col] = { col, value: null };
        }
      } catch (_) {
        this.todayValues[col] = { col, value: null };
      }
    },

    ensureTodayValue(col) {
      if (col && !(col in this.todayValues)) this._fetchTodayValue(col);
    },

    todayValueFor(col) {
      const v = this.todayValues[col];
      return (v && typeof v === 'object') ? v : null;
    },

    // For a decile profile result, return {idx (0-9), label} of which bucket
    // today's value falls into, or null if no today value available.
    todayBucket(buckets, todayValue) {
      if (!buckets?.length || todayValue == null) return null;
      for (let i = 0; i < buckets.length; i++) {
        const b = buckets[i];
        if (b.x_min == null || b.x_max == null) continue;
        // Use ≥ for the lowest edge so the smallest values still land in D1.
        const lo = (i === 0) ? -Infinity : b.x_min;
        const hi = (i === buckets.length - 1) ? Infinity : b.x_max;
        if (todayValue >= lo && todayValue <= hi) {
          return { idx: i, label: 'D' + (b.bucket_idx + 1) };
        }
      }
      return null;
    },

    isS1TodayCell(ia, ib) {
      if (!this.s1.data || !this.liveOverlays) return false;
      const a = this.todayValueFor(this.s1.metricA);
      const b = this.todayValueFor(this.s1.metricB);
      if (!a || !b || a.value == null || b.value == null) return false;
      const cell = this.todayHeatmapCell(this.s1.data, a.value, b.value);
      return cell && cell.ia === ia && cell.ib === ib;
    },

    // ── S1 heatmap trail (last N days) ──
    async loadHeatmapTrail() {
      if (!this.s1.metricA || !this.s1.metricB) {
        this.s1.trail = [];
        this.s1.trailPoints = '';
        return;
      }
      try {
        const [aRes, bRes] = await Promise.all([
          fetch('/api/meta/value-trail?days=10&col=' + encodeURIComponent(this.s1.metricA)).then(r => r.json()),
          fetch('/api/meta/value-trail?days=10&col=' + encodeURIComponent(this.s1.metricB)).then(r => r.json()),
        ]);
        const aByDate = {};
        const bByDate = {};
        for (const p of (aRes.trail || [])) aByDate[p.date] = p.value;
        for (const p of (bRes.trail || [])) bByDate[p.date] = p.value;
        const dates = Array.from(new Set([
          ...Object.keys(aByDate), ...Object.keys(bByDate),
        ])).sort();
        this.s1.trail = dates
          .map(d => ({ date: d, a: aByDate[d], b: bByDate[d] }))
          .filter(p => p.a != null && p.b != null);
      } catch (e) {
        console.warn('heatmap trail fetch failed', e);
        this.s1.trail = [];
      }
      // Wait two animation frames so the table has been laid out before
      // we measure cells for the polyline. Single nextTick wasn't enough.
      await this.$nextTick();
      requestAnimationFrame(() => {
        this.s1.trailPoints = this.s1ComputeTrailLine();
      });
    },

    // For a value, return (cellIdx, fracInsideCell) so trail dots can be
    // positioned absolutely inside the table relative to the heatmap area.
    _heatmapPos(val, bounds) {
      if (val == null || !bounds || bounds.length < 2) return null;
      // Clamp into the bin range so points outside the heatmap still show
      // at the nearest edge rather than disappearing.
      let idx = -1;
      if (val < bounds[0]) idx = 0;
      else if (val > bounds[bounds.length - 1]) idx = bounds.length - 2;
      else {
        for (let i = 0; i < bounds.length - 1; i++) {
          if (val >= bounds[i] && val <= bounds[i + 1]) { idx = i; break; }
        }
      }
      if (idx < 0) return null;
      const lo = bounds[idx];
      const hi = bounds[idx + 1];
      const frac = hi <= lo ? 0.5 : Math.max(0, Math.min(1, (val - lo) / (hi - lo)));
      return { idx, frac };
    },

    // Returns trail markers as a list of {dayIdx, ia, ib, fracX, fracY,
    // isToday, date, a, b} positioned within their containing cells.
    s1TrailMarkers() {
      if (!this.liveOverlays || !this.s1.data?.bounds_a || !this.s1.trail?.length) {
        return [];
      }
      const data = this.s1.data;
      const out = [];
      const trail = this.s1.trail;
      const lastDate = trail[trail.length - 1]?.date;
      trail.forEach((p, i) => {
        const xp = this._heatmapPos(p.a, data.bounds_a);
        const yp = this._heatmapPos(p.b, data.bounds_b);
        if (!xp || !yp) return;
        out.push({
          dayIdx:  i,
          isToday: p.date === lastDate,
          ia:      xp.idx,
          ib:      yp.idx,
          fracX:   xp.frac * 100,
          fracY:   yp.frac * 100,
          date:    p.date,
          a:       p.a,
          b:       p.b,
        });
      });
      return out;
    },

    s1TrailMarkersInCell(ia, ib) {
      return this.s1TrailMarkers().filter(m => m.ia === ia && m.ib === ib);
    },

    // Pixel coordinates of the trail polyline relative to the heatmap-wrap.
    // Computed by measuring each containing cell's bounding rect — works
    // regardless of cell width / table layout / header offset.
    s1ComputeTrailLine() {
      const markers = this.s1TrailMarkers();
      if (markers.length < 2) return '';
      // Find the heatmap table; bail quietly if not yet rendered.
      const table = document.querySelector('.biv-heatmap');
      if (!table) return '';
      const wrap = table.closest('.biv-heatmap-wrap');
      if (!wrap) return '';
      const wrapRect = wrap.getBoundingClientRect();
      const rows = table.querySelectorAll('tbody tr');
      const points = [];
      for (const m of markers) {
        const row = rows[m.ib];
        if (!row) continue;
        // Skip the header <th> in column 0; data cells start at index 1.
        const cells = row.querySelectorAll('td');
        const cell  = cells[m.ia];
        if (!cell) continue;
        const r = cell.getBoundingClientRect();
        const x = (r.left - wrapRect.left) + (m.fracX / 100) * r.width
                  + wrap.scrollLeft;
        const y = (r.top - wrapRect.top)   + (m.fracY / 100) * r.height
                  + wrap.scrollTop;
        points.push(x.toFixed(1) + ',' + y.toFixed(1));
      }
      return points.join(' ');
    },

    toggleLiveOverlays() {
      this.liveOverlays = !this.liveOverlays;
      // Re-render section charts so the today line gets removed/restored.
      for (const m of (this.s3.metrics || [])) {
        if (m && this.s3.dataByMetric[m]?.buckets) {
          this._renderDecileChartForMetric(m);
        }
      }
    },

    // Fractional position 0..100 of today's value within its containing
    // heatmap cell. Used to position the precise × marker via CSS percent.
    s1TodayPctX(ia) {
      const data = this.s1.data;
      const a = this.todayValueFor(data?.metric_a);
      if (!data?.bounds_a || !a || a.value == null) return 50;
      const lo = data.bounds_a[ia];
      const hi = data.bounds_a[ia + 1];
      if (hi <= lo) return 50;
      return Math.max(2, Math.min(98, ((a.value - lo) / (hi - lo)) * 100));
    },

    s1TodayPctY(ib) {
      const data = this.s1.data;
      const b = this.todayValueFor(data?.metric_b);
      if (!data?.bounds_b || !b || b.value == null) return 50;
      const lo = data.bounds_b[ib];
      const hi = data.bounds_b[ib + 1];
      if (hi <= lo) return 50;
      return Math.max(2, Math.min(98, ((b.value - lo) / (hi - lo)) * 100));
    },

    // For a heatmap result with bounds_a/bounds_b, return {ia, ib} for the
    // cell containing today's (a, b), or null.
    todayHeatmapCell(data, todayA, todayB) {
      if (!data?.bounds_a || !data?.bounds_b) return null;
      if (todayA == null || todayB == null) return null;
      const findIdx = (val, bounds) => {
        // bounds is length n_buckets+1; cell i covers [bounds[i], bounds[i+1]]
        if (val < bounds[0]) return 0;
        if (val > bounds[bounds.length - 1]) return bounds.length - 2;
        for (let i = 0; i < bounds.length - 1; i++) {
          if (val >= bounds[i] && val <= bounds[i + 1]) return i;
        }
        return null;
      };
      const ia = findIdx(todayA, data.bounds_a);
      const ib = findIdx(todayB, data.bounds_b);
      if (ia == null || ib == null) return null;
      return { ia, ib };
    },

    // ── Catalog (semantic metadata) ──
    async _loadCatalog() {
      try {
        const r = await fetch('/api/meta/columns-catalog');
        if (!r.ok) return;
        const rows = await r.json();
        const byName = {};
        for (const row of rows) {
          if (row?.column_name) byName[row.column_name] = row;
        }
        this.catalog = byName;
      } catch (_) { /* catalog is best-effort; UI falls back to prefix matching */ }
    },

    catalogFor(c) {
      return c ? (this.catalog[c] || null) : null;
    },

    columnDescription(c) {
      const e = this.catalogFor(c);
      return e?.description || '';
    },

    _colFamily(c) {
      if (!c) return 'other';
      // Catalog wins when available — exact, authoritative classification.
      const e = this.catalog[c];
      if (e?.family) return e.family;
      // Fallback prefix matching for non-catalog columns (entry greeks,
      // portfolio context, outcomes, trade attrs) and any unmatched extras.
      if (c.startsWith('vix_'))         return 'vix';
      if (c.startsWith('skew_'))        return 'skew';
      if (c.startsWith('term_'))        return 'term';
      if (c.startsWith('convex_'))      return 'convex';
      if (c.startsWith('iv_'))          return 'iv';
      if (c.startsWith('forward_'))     return 'forward';
      if (c.startsWith('portfolio_'))   return 'portfolio';
      if (c.startsWith('entry_'))       return 'entry';
      if (['pnl','pnl_pct','is_win','days_in_trade'].includes(c)) return 'outcome';
      if (['premium','margin_req','max_profit','max_loss',
           'legs','contracts','spx_open_price'].includes(c)) return 'trade';
      return 'other';
    },

    // Catalog-aware sort order so families with many entries (skew, iv,
    // convex) read predictably: short tenors first, puts before calls,
    // levels before z-scores before changes.
    _tenorRank(t) {
      const order = {'1d':1,'7d':2,'30d':3,'90d':4,'180d':5,
                     '1d_7d':6,'7d_30d':7,'30d_90d':8,'1d_30d':9,
                     '1w':10,'1m':11,'3m':12,'d':13};
      if (t == null) return 99;
      return order[t] ?? 50;
    },
    _wingRank(w) {
      const order = {'10p':1,'25p':2,'atm':3,'25c':4,'10c':5};
      if (w == null) return 99;
      return order[w] ?? 50;
    },
    _formRank(f) {
      const order = {'level':1,'z':2,'chg_d':3,'chg_1w':4};
      return order[f] ?? 5;
    },

    // Grouped metric list for the section dropdowns. Same family ordering
    // and intra-family sort as filteredColumnList(), but no outcome
    // columns (sections select features, not Y variables) and supports
    // an exclude set for multi-selects (S2/S7) plus an optional search
    // string (used by the searchable metric picker).
    metricColumnGroups(excludeArr, searchQuery) {
      const exclude = new Set(excludeArr || []);
      const q = (searchQuery || '').toLowerCase().trim();
      const groups = {};
      for (const c of this.ivColumns) {
        if (exclude.has(c)) continue;
        if (q) {
          const nameHit = c.toLowerCase().includes(q);
          const desc    = (this.columnDescription(c) || '').toLowerCase();
          if (!nameHit && !desc.includes(q)) continue;
        }
        const fam = this._colFamily(c);
        (groups[fam] ||= []).push(c);
      }
      const sortKey = (c) => {
        const e = this.catalogFor(c);
        return [
          this._tenorRank(e?.tenor),
          this._wingRank(e?.wing),
          this._formRank(e?.form),
          c,
        ];
      };
      const cmp = (a, b) => {
        const ka = sortKey(a), kb = sortKey(b);
        for (let i = 0; i < ka.length; i++) {
          if (ka[i] < kb[i]) return -1;
          if (ka[i] > kb[i]) return 1;
        }
        return 0;
      };
      for (const fam of Object.keys(groups)) groups[fam].sort(cmp);
      const order = ['portfolio','entry','trade',
                     'vix','skew','term','term_ratio','term_slope',
                     'convex','rr','iv','forward','vix_basis',
                     'log_ret','rv','vrp','vrp_ratio','vov','spot_vol',
                     'spot','meta','other'];
      return order
        .filter(f => groups[f]?.length)
        .map(f => ({ family: f, columns: groups[f] }));
    },

    // ── Shared searchable metric picker ──
    _pathTokens(path) {
      // 's3.metric' → ['s3', 'metric']
      // 's3.metrics[0]' → ['s3', 'metrics', 0]
      if (!path) return [];
      return path.split(/\.|\[|\]/)
        .filter(s => s !== '')
        .map(s => /^\d+$/.test(s) ? Number(s) : s);
    },

    _getNested(path) {
      const tokens = this._pathTokens(path);
      return tokens.reduce((o, k) => (o == null ? null : o[k]), this);
    },

    _setNested(path, value) {
      const tokens = this._pathTokens(path);
      if (!tokens.length) return;
      const last = tokens.pop();
      const parent = tokens.reduce((o, k) => o[k], this);
      if (parent != null) parent[last] = value;
    },

    _computeMetricPickerPos(btn) {
      const popWidth = 320;
      const margin   = 8;
      const viewport = window.innerWidth ||
                       document.documentElement.clientWidth || 1024;
      if (!btn) return { posTop: 80, posLeft: margin };
      const r = btn.getBoundingClientRect();
      // Prefer left-aligned with the button (extends right). Falls back to
      // right-aligned (extends left) if it would overflow the right edge.
      let left = r.left;
      if (left + popWidth + margin > viewport) {
        left = r.right - popWidth;
      }
      const maxLeft = viewport - popWidth - margin;
      if (left > maxLeft) left = Math.max(margin, maxLeft);
      if (left < margin) left = margin;
      return { posTop: r.bottom + 4, posLeft: Math.round(left) };
    },

    openMetricPicker(path, evt) {
      const btn = evt?.currentTarget || evt?.target;
      const pos = this._computeMetricPickerPos(btn);
      this.metricPicker = {
        open:      true,
        path,
        multiPath: null,
        search:    '',
        ...pos,
      };
    },

    openMetricPickerMulti(arrPath, evt) {
      const btn = evt?.currentTarget || evt?.target;
      const pos = this._computeMetricPickerPos(btn);
      this.metricPicker = {
        open:      true,
        path:      null,
        multiPath: arrPath,   // dot-notation pointing at an array
        search:    '',
        ...pos,
      };
    },

    closeMetricPicker() {
      this.metricPicker.open = false;
    },

    pickMetric(value) {
      if (!value) {
        this.closeMetricPicker();
        return;
      }
      if (this.metricPicker.multiPath) {
        // Multi mode: append to the target array if not already present;
        // keep the picker open so user can keep picking.
        const arr = this._getNested(this.metricPicker.multiPath);
        if (Array.isArray(arr) && !arr.includes(value)) {
          // Mutate via assignment so Alpine reactivity sees a new reference.
          this._setNested(this.metricPicker.multiPath, [...arr, value]);
          this.ensureTodayValue(value);
        }
        return;
      }
      if (this.metricPicker.path) {
        this._setNested(this.metricPicker.path, value);
      }
      this.ensureTodayValue(value);
      this.closeMetricPicker();
    },

    metricPickerCurrent() {
      if (this.metricPicker.multiPath) return null;
      return this._getNested(this.metricPicker.path);
    },

    metricPickerGroups() {
      // In multi mode exclude already-picked entries so the popover stays
      // useful as the user accumulates selections.
      let exclude = null;
      if (this.metricPicker.multiPath) {
        const arr = this._getNested(this.metricPicker.multiPath);
        if (Array.isArray(arr)) exclude = arr;
      }
      return this.metricColumnGroups(exclude, this.metricPicker.search);
    },

    // Shorthand for templates: read a metric value via dot path. Used by the
    // picker trigger buttons so each section binds without losing reactivity.
    metricVal(path) {
      const v = this._getNested(path);
      return (v === null || v === undefined || v === '') ? '' : v;
    },

    filteredColumnList() {
      const q = (this.filterEditor.colSearch || '').toLowerCase().trim();
      const outcomeCols = ['pnl', 'pnl_pct', 'is_win', 'days_in_trade'];
      const seen = new Set();
      const all  = [];
      for (const c of [...this.ivColumns, ...outcomeCols]) {
        if (!seen.has(c)) { seen.add(c); all.push(c); }
      }
      // Match name OR description so a search for "put-call skew" or "realized vol"
      // finds the right columns even when their cryptic names don't contain those words.
      const matched = !q ? all : all.filter(c => {
        if (c.toLowerCase().includes(q)) return true;
        const d = (this.columnDescription(c) || '').toLowerCase();
        return d.includes(q);
      });
      const groups = {};
      for (const c of matched) {
        const fam = this._colFamily(c);
        (groups[fam] ||= []).push(c);
      }
      // Sort within family using catalog metadata when present, name otherwise.
      const sortKey = (c) => {
        const e = this.catalogFor(c);
        return [
          this._tenorRank(e?.tenor),
          this._wingRank(e?.wing),
          this._formRank(e?.form),
          c,
        ];
      };
      const cmp = (a, b) => {
        const ka = sortKey(a), kb = sortKey(b);
        for (let i = 0; i < ka.length; i++) {
          if (ka[i] < kb[i]) return -1;
          if (ka[i] > kb[i]) return 1;
        }
        return 0;
      };
      for (const fam of Object.keys(groups)) groups[fam].sort(cmp);

      const order = ['portfolio','entry','trade','outcome',
                     'vix','skew','term','term_ratio','term_slope',
                     'convex','rr','iv','forward','vix_basis',
                     'log_ret','rv','vrp','vrp_ratio','vov','spot_vol',
                     'spot','meta','other'];
      return order
        .filter(f => groups[f]?.length)
        .map(f => ({ family: f, columns: groups[f] }));
    },

    // ── Column stats (slider track range) ──
    async _fetchColumnStats(col) {
      if (!this.selectedId) return;
      this.columnStats[col] = null;  // mark as fetching
      try {
        const r = await fetch(
          `/api/backtest-iv/${this.selectedId}/column-stats?col=${encodeURIComponent(col)}`);
        if (r.ok) {
          this.columnStats[col] = await r.json();
        } else {
          this.columnStats[col] = false;
        }
      } catch (_) {
        this.columnStats[col] = false;
      }
    },

    _ensureFilterStats() {
      const cols = Array.from(new Set(this.filters.map(f => f.col)));
      for (const col of cols) {
        if (!(col in this.columnStats)) this._fetchColumnStats(col);
      }
    },

    _ensureOneColStat(col) {
      if (col && !(col in this.columnStats)) this._fetchColumnStats(col);
    },

    // ── Slider chip helpers ──
    _isSliderEligible(f) {
      if (!f || f.op !== 'between') return false;
      const s = this.columnStats[f.col];
      if (!s || s === true) return false;
      // Need a non-degenerate range; binary or constant columns fall back to text chip.
      return s.p99 !== s.p01;
    },

    _sliderTrackRange(stats) {
      // Use 1st–99th percentile so a few outliers don't crush the useful range.
      // Handles can still be set outside this range via the popover.
      return { lo: stats.p01, hi: stats.p99 };
    },

    _sliderValueAt(filter, side) {
      // What value to show on the slider for a possibly-null bound. The slider
      // visually anchors a null bound to the track edge; on first drag it's
      // committed as a real number via updateFilter.
      const s = this.columnStats[filter.col];
      if (!s || s === true || s === false) return 0;
      const { lo, hi } = this._sliderTrackRange(s);
      if (side === 'min') return (filter.min === null || filter.min === undefined) ? lo : filter.min;
      return (filter.max === null || filter.max === undefined) ? hi : filter.max;
    },

    _sliderPct(filter, side) {
      const s = this.columnStats[filter.col];
      if (!s || s === true || s === false) return side === 'min' ? 0 : 100;
      const { lo, hi } = this._sliderTrackRange(s);
      const v = this._sliderValueAt(filter, side);
      const span = hi - lo;
      if (span <= 0) return side === 'min' ? 0 : 100;
      const pct = (v - lo) / span * 100;
      return Math.max(0, Math.min(100, pct));
    },

    _formatSliderValue(v) {
      if (v === null || v === undefined || Number.isNaN(v)) return '—';
      const abs = Math.abs(v);
      if (abs >= 1000) return Math.round(v).toLocaleString();
      if (abs >= 10)   return v.toFixed(1);
      if (abs >= 1)    return v.toFixed(2);
      return v.toFixed(4);
    },

    startSliderDrag(evt, idx, mode) {
      // mode: 'min' | 'max' | 'pan'
      evt.preventDefault();
      evt.stopPropagation();
      const f = this.filters[idx];
      const stats = this.columnStats[f.col];
      if (!stats || stats === true || stats === false) return;
      const trackEl = evt.currentTarget.closest('.biv-slider-track-wrap');
      if (!trackEl) return;
      const r = trackEl.getBoundingClientRect();
      this.sliderDrag = {
        idx,
        mode,
        startX:    evt.clientX,
        trackLeft: r.left,
        trackWidth:r.width,
        startMin:  this._sliderValueAt(f, 'min'),
        startMax:  this._sliderValueAt(f, 'max'),
        stats,
      };
      try { evt.target.setPointerCapture?.(evt.pointerId); } catch (_) {}
    },

    doSliderDrag(evt) {
      const d = this.sliderDrag;
      if (!d) return;
      const { lo, hi } = this._sliderTrackRange(d.stats);
      const valuePerPx = (hi - lo) / Math.max(1, d.trackWidth);
      const delta = (evt.clientX - d.startX) * valuePerPx;
      const f = this.filters[d.idx];
      if (!f) return;

      if (d.mode === 'min') {
        const next = Math.max(lo, Math.min(d.startMax, d.startMin + delta));
        f.min = this._roundForCol(next, d.stats);
        if (f.max === null || f.max === undefined) f.max = d.startMax;
      } else if (d.mode === 'max') {
        const next = Math.max(d.startMin, Math.min(hi, d.startMax + delta));
        f.max = this._roundForCol(next, d.stats);
        if (f.min === null || f.min === undefined) f.min = d.startMin;
      } else if (d.mode === 'pan') {
        const span = d.startMax - d.startMin;
        let candMin = d.startMin + delta;
        let candMax = d.startMax + delta;
        if (candMin < lo) { candMin = lo; candMax = lo + span; }
        if (candMax > hi) { candMax = hi; candMin = hi - span; }
        f.min = this._roundForCol(candMin, d.stats);
        f.max = this._roundForCol(candMax, d.stats);
      }
    },

    endSliderDrag(_evt) {
      const d = this.sliderDrag;
      this.sliderDrag = null;
      if (!d) return;
      const f = this.filters[d.idx];
      if (!f) return;
      // Route through updateFilter so URL syncs and sections refetch.
      this.updateFilter(d.idx, { col: f.col, op: 'between', min: f.min, max: f.max });
    },

    _roundForCol(v, stats) {
      // Round to a sensible precision based on column scale. Saves URLs
      // from "vix_30d:between:18.4123459237" type noise.
      const span = stats.p99 - stats.p01;
      if (span >= 1000) return Math.round(v);
      if (span >= 100)  return Math.round(v * 10) / 10;
      if (span >= 10)   return Math.round(v * 100) / 100;
      if (span >= 1)    return Math.round(v * 1000) / 1000;
      return Math.round(v * 10000) / 10000;
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
        payload.bin_mode = this.binMode;
        if (this.filters.length) payload.filters = this.filters;
      } else {
        const params = new URLSearchParams();
        if (this.dateFrom) params.set('date_from', this.dateFrom);
        if (this.dateTo)   params.set('date_to',   this.dateTo);
        params.set('bin_mode', this.binMode);
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

    toggleS0Fullscreen() {
      this.s0.fullscreen = !this.s0.fullscreen;
      // Force the section open if entering fullscreen so the body shows.
      if (this.s0.fullscreen) this.open.s0 = true;
      // Chart.js's responsive observer picks up the new container size on
      // its own, but a manual resize on the next tick keeps it snappy.
      this.$nextTick(() => {
        try { this.s0._chart?.resize(); } catch (_) {}
      });
    },

    setS0Method(method) {
      if (!['pearson','spearman','consensus','mi'].includes(method)) return;
      if (this.s0.method === method) return;
      this.s0.method = method;
      // No fetch needed — the backend returns all values.
      this._renderS0Chart();
    },

    _s0Value(m, method) {
      if (method === 'spearman')   return (m.spearman_r  ?? 0);
      if (method === 'consensus')  return (m.consensus_r ?? 0);
      if (method === 'mi')         return (m.mi          ?? 0);
      return (m.pearson_r ?? m.r ?? 0);
    },

    _renderS0Chart() {
      this.s0._chart = this._destroyChart(this.s0._chart);
      const el = document.getElementById('s0-chart');
      if (!el || !this.s0.data?.metrics?.length) return;

      const target  = this.s0.data.target;
      const method  = this.s0.method || 'pearson';
      // Sort by selected method so the most relevant signals lead. Highlight
      // the method's bars at full opacity; the other one is dimmed but always
      // visible so divergence between the two is obvious at a glance.
      const metrics = [...this.s0.data.metrics].sort(
        (a, b) => Math.abs(this._s0Value(b, method)) - Math.abs(this._s0Value(a, method))
      );
      const labels = metrics.map(m => m.metric);

      const pearsonAlpha   = method === 'pearson'   ? 0.92 : 0.30;
      const spearmanAlpha  = method === 'spearman'  ? 0.92 : 0.30;
      const consensusAlpha = method === 'consensus' ? 0.92 : 0.30;
      const miAlpha        = method === 'mi'        ? 0.92 : 0.30;

      const datasets = [];
      if (method === 'consensus') {
        // Consensus mode is a single-value summary; show one bar plus dimmed
        // pearson/spearman as reference so divergence still reads.
        datasets.push({
          label: 'Consensus',
          data: metrics.map(m => m.consensus_r ?? 0),
          backgroundColor: metrics.map(m => {
            const v = m.consensus_r ?? 0;
            return v >= 0 ? `rgba(46,204,113,${consensusAlpha})`
                          : `rgba(220,60,155,${consensusAlpha})`;
          }),
          borderWidth: 0, categoryPercentage: 0.92, barPercentage: 0.92,
        });
      }
      datasets.push({
        label: 'Pearson r',
        data: metrics.map(m => m.pearson_r ?? m.r ?? 0),
        backgroundColor: metrics.map(m => {
          const v = m.pearson_r ?? m.r ?? 0;
          return v >= 0 ? `rgba(41,128,245,${pearsonAlpha})`
                        : `rgba(220,60,155,${pearsonAlpha})`;
        }),
        borderWidth: 0, categoryPercentage: 0.92, barPercentage: 0.92,
      });
      datasets.push({
        label: 'Spearman ρ',
        data: metrics.map(m => m.spearman_r ?? 0),
        backgroundColor: metrics.map(m => {
          const v = m.spearman_r ?? 0;
          return v >= 0 ? `rgba(241,196,15,${spearmanAlpha})`
                        : `rgba(155,89,182,${spearmanAlpha})`;
        }),
        borderWidth: 0, categoryPercentage: 0.92, barPercentage: 0.92,
      });
      // MI is unsigned (always ≥ 0) — single teal color. Catches non-monotonic
      // dependence (U-shapes, threshold effects) that Pearson and Spearman miss.
      datasets.push({
        label: 'MI (any shape)',
        data: metrics.map(m => m.mi ?? 0),
        backgroundColor: `rgba(26,188,156,${miAlpha})`,
        borderWidth: 0, categoryPercentage: 0.92, barPercentage: 0.92,
      });

      this.s0._chart = new Chart(el.getContext('2d'), {
        type: 'bar',
        data: { labels, datasets },
        options: {
          responsive: true, maintainAspectRatio: false, animation: false,
          plugins: {
            legend: {
              labels: { color: '#aaa', font: { size: 12 }, boxWidth: 14 },
            },
            tooltip: {
              backgroundColor: 'rgba(20,20,20,0.95)',
              borderColor: '#444', borderWidth: 1,
              titleFont: { size: 12 },
              bodyFont:  { size: 12, family: 'monospace' },
              callbacks: {
                title: items => metrics[items[0].dataIndex].metric,
                label: ctx => {
                  const m   = metrics[ctx.dataIndex];
                  const tgt = target === 'pnl' ? 'P&L' : 'Win';
                  const p   = m.pearson_r ?? m.r ?? 0;
                  const s   = m.spearman_r ?? 0;
                  const c   = m.consensus_r ?? 0;
                  const mi  = m.mi ?? 0;
                  const div = m.divergence ?? Math.abs(p - s);
                  const sign = v => (v >= 0 ? '+' : '') + v.toFixed(4);
                  return [
                    `target: ${tgt}   n=${m.n}`,
                    `Pearson r  = ${sign(p)}`,
                    `Spearman ρ = ${sign(s)}`,
                    `MI         = ${mi.toFixed(4)}   (unsigned)`,
                    `Consensus  = ${sign(c)}   (Δ=${div.toFixed(3)})`,
                  ];
                },
              },
            },
          },
          scales: {
            x: {
              ticks: {
                color: '#aaa', font: { size: 11, family: 'monospace' },
                maxRotation: 90, minRotation: 90, autoSkip: false,
              },
              grid:   { display: false },
              border: { color: '#333' },
            },
            y: {
              min: -1, max: 1,
              ticks: { color: '#aaa', font: { size: 12 }, stepSize: 0.25 },
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
      // Cell selection is keyed to the previous heatmap's bin layout, so a
      // recompute (different metrics or bin count) always invalidates it.
      const hadSelection = this.s1.selectedCells.length > 0;
      this.s1.selectedCells = [];
      try {
        const data = await this._api('heatmap', {
          metric_a: this.s1.metricA, metric_b: this.s1.metricB, n_buckets: this.s1.nBuckets,
        });
        this.s1._renderId = (this.s1._renderId + 1) % 100000;
        this.s1.data = data;
      } catch (e) { this.s1.error = e.message; }
      finally { this.s1.loading = false; }
      // Fetch the trail (last N days) for the chosen metric pair.
      this.loadHeatmapTrail();
      // If S3 was filtered by old cells, refresh now that the filter is empty.
      if (hadSelection && this.s3HasResults()) this.loadDecile();
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
    // ── S3: multi-metric decile, optional S1-cell filter ──
    s3AddMetric() {
      this.s3.metrics = [...this.s3.metrics, ''];
    },

    s3RemoveMetric(idx) {
      const m = this.s3.metrics[idx];
      if (m && this.s3.charts[m]) {
        this.s3.charts[m] = this._destroyChart(this.s3.charts[m]);
      }
      if (m) delete this.s3.dataByMetric[m];
      const next = this.s3.metrics.filter((_, i) => i !== idx);
      this.s3.metrics = next.length ? next : [''];
    },

    s3HasResults() {
      return Object.keys(this.s3.dataByMetric || {}).length > 0;
    },

    _safeId(s) {
      return String(s || '').replace(/[^a-zA-Z0-9_-]/g, '_');
    },

    // Translate s1.selectedCells into the backend CellClause shape using
    // the heatmap's known per-axis quantile boundaries.
    _buildCellFilter() {
      const sel = this.s1.selectedCells || [];
      if (sel.length === 0) return [];
      const data = this.s1.data;
      if (!data?.bounds_a || !data?.bounds_b) return [];
      return sel.map(({ ia, ib }) => ({
        metric_a: data.metric_a,
        a_min:    data.bounds_a[ia],
        a_max:    data.bounds_a[ia + 1],
        metric_b: data.metric_b,
        b_min:    data.bounds_b[ib],
        b_max:    data.bounds_b[ib + 1],
      }));
    },

    async loadDecile() {
      if (!this.selectedId) {
        console.log('[loadDecile] aborted — no upload selected');
        return;
      }
      const metrics = (this.s3.metrics || []).filter(m => !!m);
      if (metrics.length === 0) {
        console.log('[loadDecile] aborted — no metrics set');
        return;
      }
      this.s3.loading = true; this.s3.error = null;

      // In-flight cancellation: only the latest call applies its results.
      const token = ++this.s3._loadToken;
      const cellFilter = this._buildCellFilter();
      console.log('[loadDecile] token=' + token,
                  'metrics=', metrics,
                  'cells=' + (this.s1.selectedCells?.length || 0),
                  'cellFilter=', cellFilter);

      let results;
      try {
        results = await Promise.all(metrics.map(metric =>
          this._api('decile', {
            metric,
            n_buckets:   this.s3.nBuckets,
            cell_filter: cellFilter,
          }).then(d => [metric, d])
            .catch(e => [metric, { error: e.message || String(e) }])
        ));
      } catch (e) {
        console.error('[loadDecile] Promise.all failed at token=' + token, e);
        if (token === this.s3._loadToken) {
          this.s3.error = e.message;
          this.s3.loading = false;
        }
        return;
      }
      console.log('[loadDecile] results token=' + token,
                  results.map(([m, d]) => [m, d?.error ? 'ERROR: ' + d.error
                                              : d?.buckets?.length + ' buckets']));

      try {
        // A newer load was kicked off while we were waiting — drop these.
        // Importantly we do NOT touch chart state here; the previous render
        // stays visible until a NEWER successful load replaces it.
        if (token !== this.s3._loadToken) {
          console.log('[loadDecile] stale token=' + token + ', current=' + this.s3._loadToken);
          return;
        }

        const next = {};
        for (const [m, d] of results) next[m] = d;
        this.s3.dataByMetric = next;
        await this.$nextTick();
        if (token !== this.s3._loadToken) {
          console.log('[loadDecile] stale token after nextTick at ' + token);
          return;
        }

        for (const m of metrics) {
          const hasBuckets = !!next[m]?.buckets;
          console.log('[loadDecile]   metric=' + m,
                      'hasBuckets=' + hasBuckets,
                      'error=' + (next[m]?.error || ''));
          if (hasBuckets) {
            this._renderDecileChartForMetric(m);
          } else {
            // Error or no buckets — destroy chart, paint error into wrap.
            this.s3.charts[m] = this._destroyChart(this.s3.charts[m]);
            const wrap = document.getElementById('s3-wrap-' + this._safeId(m));
            if (wrap) {
              const msg = (next[m]?.error || 'No data after filters').replace(/[<>&]/g, '');
              wrap.innerHTML = '<div style="color:#e74c3c;padding:12px;font-size:12px;font-family:monospace">' + msg + '</div>';
            } else {
              console.warn('[loadDecile] wrap not found for ' + m + ' at error path');
            }
          }
        }
      } catch (e) {
        console.error('[loadDecile] post-fetch exception at token=' + token, e);
        if (token === this.s3._loadToken) this.s3.error = e.message;
      } finally {
        if (token === this.s3._loadToken) this.s3.loading = false;
      }
    },

    _renderDecileChartForMetric(metric, retries = 8) {
      const data = this.s3.dataByMetric[metric];
      if (!data?.buckets) {
        console.log('[renderDecile] no buckets for ' + metric + ', skipping');
        return;
      }
      const wrap = document.getElementById('s3-wrap-' + this._safeId(metric));
      if (!wrap) {
        // Wrap div should always be in DOM (uses x-show, not x-if). If
        // missing, retry briefly — defensive only.
        console.warn('[renderDecile] wrap missing for ' + metric + ', retries=' + retries);
        if (retries > 0) {
          setTimeout(() => this._renderDecileChartForMetric(metric, retries - 1), 80);
        }
        return;
      }
      console.log('[renderDecile] rendering ' + metric + ' (' + data.buckets.length + ' buckets)');
      // Tear down the prior chart instance and rebuild the canvas from
      // scratch. Eliminates ANY state contamination from previous renders.
      this.s3.charts[metric] = this._destroyChart(this.s3.charts[metric]);
      wrap.innerHTML = '';
      const canvas = document.createElement('canvas');
      wrap.appendChild(canvas);

      const buckets  = data.buckets;
      const labels   = buckets.map(b => b.label);
      const pnls     = buckets.map(b => b.mean_pnl ?? 0);
      const maxAbs   = Math.max(...pnls.map(Math.abs), 1);
      const colors   = pnls.map(v => pnlColor(v, maxAbs));
      const winRates = buckets.map(b => (b.win_rate ?? 0) * 100);

      // Today's value marker — drawn by an inline plugin so the line lands
      // at the precise fractional x position within the matching bucket.
      // Suppressed entirely when the global Live overlay toggle is off.
      const todayInfo = this.todayValueFor(metric);
      const todayValue = this.liveOverlays ? todayInfo?.value : null;

      try {
      this.s3.charts[metric] = new Chart(canvas.getContext('2d'), {
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
          plugins: {
            legend: { labels: { color: '#aaa', font: { size: 11 } } },
            todayMarker: { todayValue, buckets },
          },
          scales: {
            x: { ticks: { color: '#888', font: { size: 10 } }, grid: { color: '#333' } },
            y:  { ticks: { color: '#aaa', font: { size: 10 } }, grid: { color: '#333' },
                  title: { display: true, text: 'Mean P&L', color: '#aaa', font: { size: 10 } } },
            y2: { position: 'right', min: 0, max: 100,
                  ticks: { color: '#aaa', font: { size: 10 } }, grid: { drawOnChartArea: false },
                  title: { display: true, text: 'Win Rate %', color: '#aaa', font: { size: 10 } } },
          },
        },
        plugins: [_todayMarkerPlugin],
      });
      } catch (e) {
        console.error('S3 chart render failed for', metric, e);
        wrap.innerHTML = '<div style="color:#e74c3c;padding:12px;font-size:11px;font-family:monospace">Chart render failed: ' + (e.message || e) + '</div>';
      }
    },

    // ── S1 cell selection (drives S3 cell filter) ──
    toggleS1Cell(ia, ib) {
      const sel = this.s1.selectedCells || [];
      const idx = sel.findIndex(c => c.ia === ia && c.ib === ib);
      this.s1.selectedCells = idx >= 0
        ? sel.filter((_, i) => i !== idx)
        : [...sel, { ia, ib }];
      // If S3 already has results, re-fetch with the new cell filter.
      if (this.s3HasResults()) this.loadDecile();
    },

    clearS1Selection() {
      if (!this.s1.selectedCells.length) return;
      this.s1.selectedCells = [];
      if (this.s3HasResults()) this.loadDecile();
    },

    isS1CellSelected(ia, ib) {
      return (this.s1.selectedCells || []).some(c => c.ia === ia && c.ib === ib);
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

    _renderSliceChart(retries = 8) {
      if (!this.s4.data?.buckets) return;
      const wrap = document.getElementById('s4-wrap');
      if (!wrap) {
        if (retries > 0) {
          setTimeout(() => this._renderSliceChart(retries - 1), 80);
        } else {
          console.warn('S4 wrap not found');
        }
        return;
      }
      this.s4._chart = this._destroyChart(this.s4._chart);
      wrap.innerHTML = '';
      const canvas = document.createElement('canvas');
      wrap.appendChild(canvas);

      const buckets = this.s4.data.buckets;
      const labels  = buckets.map(b => b.label);
      const pnls    = buckets.map(b => b.mean_pnl ?? 0);
      const maxAbs  = Math.max(...pnls.map(Math.abs), 1);
      const colors  = pnls.map(v => pnlColor(v, maxAbs));

      try {
      this.s4._chart = new Chart(canvas.getContext('2d'), {
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
      } catch (e) {
        console.error('S4 chart render failed', e);
        wrap.innerHTML = '<div style="color:#e74c3c;padding:12px;font-size:11px">Chart render failed: ' + (e.message || e) + '</div>';
      }
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

    _renderStabilityChart(retries = 8) {
      if (!this.s6.data?.periods) return;
      const wrap = document.getElementById('s6-wrap');
      if (!wrap) {
        if (retries > 0) {
          setTimeout(() => this._renderStabilityChart(retries - 1), 80);
        } else {
          console.warn('S6 wrap not found');
        }
        return;
      }
      this.s6._chart = this._destroyChart(this.s6._chart);
      wrap.innerHTML = '';
      const canvas = document.createElement('canvas');
      wrap.appendChild(canvas);

      const periods = this.s6.data.periods;
      const labels  = periods.map(p => p.label + '\n' + (p.date_from || '').slice(0, 7));
      const rs      = periods.map(p => p.r);
      const pnls    = periods.map(p => p.mean_pnl);
      const maxAbsPnl = Math.max(...pnls.map(Math.abs), 1);

      try {
      this.s6._chart = new Chart(canvas.getContext('2d'), {
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
      } catch (e) {
        console.error('S6 chart render failed', e);
        wrap.innerHTML = '<div style="color:#e74c3c;padding:12px;font-size:11px">Chart render failed: ' + (e.message || e) + '</div>';
      }
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
