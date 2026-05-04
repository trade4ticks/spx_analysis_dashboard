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
    _max: maxCount,

    addFromEvent(event) {
      const col = event.target.value;
      event.target.value = '';
      if (!col || this.selected.includes(col)) return;
      if (this.selected.length >= this._max) return;
      this.selected.push(col);
    },

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

    // Section open/closed
    open: { s1: true, s2: true, s3: true, s4: false, s5: false, s6: false, s7: false, s8: true },

    // ── Section 1: 2D Heatmap ──
    s1: {
      metricA: '', metricB: '', nBuckets: 5, valueField: 'mean_pnl',
      data: null, loading: false, error: null,
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
      await this.loadUploads();
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
      // Reset all section data
      this.s1.data = null; this.s2.data = null; this.s3.data = null;
      this.s4.data = null; this.s5.data = null; this.s6.data = null;
      this.s7.data = null; this.s8.data = null;
      // Reset multi-selects
      if (this.s2.ms) this.s2.ms.clear();
      if (this.s7.ms) this.s7.ms.clear();

      try {
        const r   = await fetch(`/api/backtest-iv/${id}/columns`);
        const res = await r.json();
        this.ivColumns = res.iv_columns || [];
      } catch (e) {
        this.globalError = e.message;
        return;
      }
      // Auto-load Section 8
      this.loadTopBottom();
    },

    // ── Helpers ──
    _api(path, body) {
      return fetch(`/api/backtest-iv/${this.selectedId}/${path}`, {
        method:  body !== undefined ? 'POST' : 'GET',
        headers: body !== undefined ? { 'Content-Type': 'application/json' } : {},
        body:    body !== undefined ? JSON.stringify(body) : undefined,
      }).then(async r => {
        if (!r.ok) {
          const e = await r.json().catch(() => ({}));
          throw new Error(e.detail || `HTTP ${r.status}`);
        }
        return r.json();
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
        this.s1.data = await this._api('heatmap', {
          metric_a: this.s1.metricA, metric_b: this.s1.metricB, n_buckets: this.s1.nBuckets,
        });
      } catch (e) { this.s1.error = e.message; }
      finally { this.s1.loading = false; }
    },

    s1CellBg(cell) {
      if (!this.s1.data || !cell) return `rgb(${GRAY.join(',')})`;
      const vf = this.s1.valueField;
      if (vf === 'mean_pnl') {
        const maxAbs = Math.max(...this.s1.data.cells.flat()
          .map(c => Math.abs(c.mean_pnl || 0)).filter(v => v > 0));
        return pnlColor(cell.mean_pnl, maxAbs);
      }
      if (vf === 'win_rate') return winRateColor(cell.win_rate);
      // count
      const maxN = Math.max(...this.s1.data.cells.flat().map(c => c.n || 0));
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
      this.s8.loading = true;
      try {
        this.s8.data = await this._api('top-bottom');
      } catch (e) { this.s8.error = e.message; }
      finally { this.s8.loading = false; }
    },

    // ── UI helpers ──
    toggleSection(s) { this.open[s] = !this.open[s]; },

    get noUpload() { return !this.selectedId; },

  }));
});
