/* ============================================================================
 * Ticker Analysis page — single-ticker view.
 *
 * Additive module (see ticker_analysis_build_brief.md). Nothing here touches
 * the Factor Analysis page or its state.
 *
 * Phase 2a: the core interaction loop —
 *   • full-width price chart (full history, close, split markers)
 *   • metric panes: 20-bin IS fwd-P&L bars (click-to-select, today-in-bin
 *     lean marker) + value-over-time sub-pane (selected-bin band)
 *   • bin selection drives the confluence/union price-chart highlight
 *   • dynamic union/dedup stat strip, recomputed client-side
 *   • family-grouped metric dropdowns; "on price" overlays; spike-preserving
 *     (min/max) value-over-time rendering at full span
 *   • "Today — what's unusual" row: per-metric today value / bin / percentile,
 *     extremes first (from /today-scan); click a cell to add it as a pane
 * Deferred to later phases: saved named layouts (P3), option-chain (P4+).
 * ==========================================================================*/

/* Forward-return horizons offered in the control bar (brief §3.1).
 * These are the actual daily_features column names — shown verbatim. Bins
 * are fixed at 20 on this page, so there is no bin-count control. */
const TA_HORIZONS = [
  'ret_1d_fwd_oc', 'ret_3d_fwd_oc', 'ret_5d_fwd_oc',
  'ret_7d_fwd_oc', 'ret_10d_fwd_oc', 'ret_20d_fwd_oc',
  'ret_1d_fwd_cc', 'ret_3d_fwd_cc', 'ret_5d_fwd_cc',
  'ret_7d_fwd_cc', 'ret_10d_fwd_cc', 'ret_20d_fwd_cc',
];

const TA_BLUE = '#3498db';   // positive / calls / long / above (theme --accent)
const TA_PINK = '#e84393';   // negative / puts / short / below

/* Distinct colors for "on price" metric overlays — deliberately NOT blue or
 * pink (those carry sign meaning). Assigned by order among overlaid panes. */
const TA_OVERLAY_COLORS = ['#f39c12', '#1abc9c', '#9b59b6', '#e67e22', '#00d2ff', '#e056fd'];

/* Chart.js instances and heavy per-pane series live OUTSIDE Alpine's
 * reactive proxy — wrapping Chart internals or 1.7k-row arrays in a Proxy
 * breaks rendering and slows selection. Keyed by pane id. */
const TA_CHARTS = { price: null, bars: {}, series: {} };
const TA_DATA = {};   // paneId -> { bins, series, today }

/* Numpy-matching stats, mirroring oi_analysis.js `_rgComputeStats`:
 * median averages the two middle values for even n; P5/P95 use linear
 * interpolation; std is population (ddof=0). Inputs are return fractions. */
function taStats(vals) {
  if (!vals || !vals.length) return null;
  const n = vals.length;
  const sorted = [...vals].sort((a, b) => a - b);
  const mean = vals.reduce((s, v) => s + v, 0) / n;
  const median = n % 2 === 1
    ? sorted[(n - 1) / 2]
    : (sorted[n / 2 - 1] + sorted[n / 2]) / 2;
  const variance = vals.reduce((s, v) => s + (v - mean) ** 2, 0) / n;
  const std = Math.sqrt(variance);
  const q = p => {
    const idx = (n - 1) * p / 100;
    const lo = Math.floor(idx), hi = Math.ceil(idx);
    return sorted[lo] + (idx - lo) * (sorted[hi] - sorted[lo]);
  };
  const winners = vals.filter(v => v > 0);
  const losers = vals.filter(v => v <= 0);
  return {
    n, mean, median, std, p5: q(5), p95: q(95),
    win_rate: winners.length / n,
    n_win: winners.length,
    avg_win: winners.length ? winners.reduce((s, v) => s + v, 0) / winners.length : 0,
    avg_loss: losers.length ? losers.reduce((s, v) => s + v, 0) / losers.length : 0,
  };
}

const taPct = (v, dp = 2) => (v == null ? '—' : (v * 100).toFixed(dp) + '%');
const taSigned = (v, dp = 2) => (v == null ? '—' : (v >= 0 ? '+' : '') + (v * 100).toFixed(dp) + '%');

document.addEventListener('alpine:init', () => {
  Alpine.data('tickerAnalysis', () => ({
    // ── Control-bar state ────────────────────────────────────────────────
    tickers: [],
    ticker: '',
    horizons: TA_HORIZONS,
    horizon: 'ret_5d_fwd_oc',      // default matches Factor Analysis default
    shadeMode: 'confluence',       // 'confluence' (default) | 'union'  (§5.1)

    metricOptions: [],             // flat list of eligible metric columns
    metricFamilyLookup: {},        // metric -> {family_num, family_name} (for <optgroup>)
    panes: [],                     // [{id, metric, loading, error, selectedBins:[], today, onPrice}]
    paneSeq: 0,
    priceLoading: false,
    priceError: '',

    stats: [],                     // [{label, value, cls}] for the stat strip
    statLabel: 'all dates',

    todayScan: [],                 // [{metric, value, bin, percentile}] extremes-first
    todayScanLoading: false,
    todayScanError: '',

    // ── Lifecycle ────────────────────────────────────────────────────────
    async init() {
      await Promise.all([this.loadTickers(), this.loadMetricOptions()]);
      if (this.ticker) await this.loadTicker();
    },

    async loadTickers() {
      try {
        const r = await fetch('/api/ticker-analysis/tickers');
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        this.tickers = await r.json();
        if (this.tickers.length && !this.ticker) this.ticker = this.tickers[0];
      } catch (e) {
        console.error('[ticker-analysis] loadTickers failed:', e);
        this.tickers = [];
      }
    },

    async loadMetricOptions() {
      // Reuse the Factor Analysis columns endpoint — same eligible metric
      // universe, already grouped/filtered. We use the flat feature list.
      try {
        const r = await fetch('/api/factor-analysis/columns');
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        const j = await r.json();
        this.metricOptions = j.features || [];
        // Build metric→family lookup for <optgroup> rendering (same family
        // hierarchy the Factor Analysis dropdowns use, from metric_classification).
        const lut = {};
        for (const grp of (j.feature_families || [])) {
          for (const m of grp.metrics) lut[m] = { family_num: grp.family_num, family_name: grp.family_name };
        }
        this.metricFamilyLookup = lut;
      } catch (e) {
        console.error('[ticker-analysis] loadMetricOptions failed:', e);
        this.metricOptions = [];
      }
      // Seed one pane so the page is useful on first load.
      if (!this.panes.length && this.metricOptions.length) {
        this.addPane(this.metricOptions[0], /*render=*/false);
      }
    },

    // ── Ticker / horizon changes ─────────────────────────────────────────
    async loadTicker() {
      await Promise.all([
        this.loadPrice(),
        this.loadTodayScan(),
        ...this.panes.map(p => this.loadPaneData(p)),
      ]);
      this.$nextTick(() => {
        this.renderPrice();
        for (const p of this.panes) this.renderPane(p);
        this.recompute();
      });
    },

    async loadTodayScan() {
      if (!this.ticker) return;
      this.todayScanLoading = true; this.todayScanError = '';
      try {
        const r = await fetch(`/api/ticker-analysis/today-scan?ticker=${encodeURIComponent(this.ticker)}`);
        const j = await r.json();
        if (j.error) { this.todayScanError = j.error; this.todayScan = []; }
        else this.todayScan = j.rows || [];
      } catch (e) {
        console.error('[ticker-analysis] loadTodayScan failed:', e);
        this.todayScanError = 'load failed'; this.todayScan = [];
      } finally {
        this.todayScanLoading = false;
      }
    },

    // Badge tint for a "what's unusual" cell: blue for high bins, pink for
    // low, intensity scaled by distance from the median bin.
    scanBadgeStyle(bin) {
      const d = Math.abs(bin - 10.5) / 9.5;                 // 0 (median) .. 1 (extreme)
      const base = bin >= 11 ? '52,152,219' : '232,67,147';
      return `background:rgba(${base},${(0.18 + 0.55 * d).toFixed(2)})`;
    },

    async onTickerChange() {
      // Bin selections are index-based and carry across tickers (brief §4);
      // panes just re-query for the new ticker.
      await this.loadTicker();
    },

    async onHorizonChange() {
      // Forward returns change → reload every pane, then recompute strip/bars.
      await Promise.all(this.panes.map(p => this.loadPaneData(p)));
      this.$nextTick(() => {
        for (const p of this.panes) this.renderPane(p);
        this.recompute();
      });
    },

    // ── Panes ────────────────────────────────────────────────────────────
    addPane(metric = null, render = true) {
      const id = ++this.paneSeq;
      const wanted = metric || (this.metricOptions[0] || '');
      const pane = {
        id,
        metric: wanted,
        loading: false,
        error: '',
        selectedBins: [],
        today: null,
        onPrice: false,
      };
      this.panes.push(pane);
      if (render && this.ticker) {
        this.loadPaneData(pane).then(() => {
          // Guard against the <select> reconciling pane.metric back to its
          // first option before its own option renders (see paneMetricGroups).
          if (pane.metric !== wanted) pane.metric = wanted;
          this.$nextTick(() => {
            this.renderPane(pane);
            this.recompute();
          });
        });
      }
    },

    // Option list for a pane's metric dropdown: the eligible feature universe
    // PLUS this pane's own metric when it isn't in that list (e.g. a metric
    // picked from the "what's unusual" scan, which spans all binned metrics).
    // Guarantees the <select> can always represent pane.metric.
    paneMetricGroups(pane) {
      let list = this.metricOptions;
      if (pane.metric && !list.includes(pane.metric)) list = [...list, pane.metric];
      return this.groupMetricsByFamily(list);
    },

    removePane(id) {
      const i = this.panes.findIndex(p => p.id === id);
      if (i === -1) return;
      const wasOnPrice = this.panes[i].onPrice;
      this.destroyPaneCharts(id);
      delete TA_DATA[id];
      this.panes.splice(i, 1);
      this.$nextTick(() => {
        if (wasOnPrice) this.renderPrice();   // remaining overlays re-color
        this.recompute();
      });
    },

    onPaneMetricChange(pane) {
      pane.selectedBins = [];   // bins are metric-specific; reset on metric swap
      this.loadPaneData(pane).then(() => this.$nextTick(() => {
        this.renderPane(pane);
        if (pane.onPrice) this.renderPrice();   // overlay follows the new metric
        this.recompute();
      }));
    },

    // Group a flat metric list by family via metricFamilyLookup — same
    // hierarchy and ordering the Factor Analysis dropdowns use (families
    // sorted by number, metrics alpha within, unknowns → "Other").
    groupMetricsByFamily(list) {
      const groups = new Map();   // family_num -> {family_num, family_name, metrics:[]}
      for (const m of (list || [])) {
        const fam = this.metricFamilyLookup[m];
        const key = fam ? fam.family_num : 999;
        const label = fam ? fam.family_name : 'Other';
        if (!groups.has(key)) groups.set(key, { family_num: key, family_name: label, metrics: [] });
        groups.get(key).metrics.push(m);
      }
      for (const g of groups.values()) g.metrics.sort((a, b) => String(a).localeCompare(String(b)));
      return [...groups.values()].sort((a, b) => a.family_num - b.family_num);
    },

    // ── "On price" overlays (§5.2) ───────────────────────────────────────
    togglePaneOnPrice(pane) {
      this.renderPrice();   // rebuild price chart with the current overlay set
    },

    paneOverlayColor(id) {
      const on = this.panes.filter(p => p.onPrice);
      const i = on.findIndex(p => p.id === id);
      return i === -1 ? TA_BLUE : TA_OVERLAY_COLORS[i % TA_OVERLAY_COLORS.length];
    },

    async loadPaneData(pane) {
      if (!this.ticker || !pane.metric) return;
      pane.loading = true; pane.error = '';
      try {
        const url = `/api/ticker-analysis/metric?ticker=${encodeURIComponent(this.ticker)}`
                  + `&metric=${encodeURIComponent(pane.metric)}`
                  + `&horizon=${encodeURIComponent(this.horizon)}`;
        const r = await fetch(url);
        const j = await r.json();
        if (j.error) { pane.error = j.error; TA_DATA[pane.id] = null; pane.today = null; }
        else {
          TA_DATA[pane.id] = { bins: j.bins, series: j.series, today: j.today };
          pane.today = j.today;
          // Prune any selected bins that no longer have data for this metric.
          const valid = new Set(j.bins.filter(Boolean).map(b => b.bin));
          pane.selectedBins = pane.selectedBins.filter(b => valid.has(b));
        }
      } catch (e) {
        console.error('[ticker-analysis] loadPaneData failed:', e);
        pane.error = 'load failed';
        TA_DATA[pane.id] = null;
      } finally {
        pane.loading = false;
      }
    },

    // ── Selection ────────────────────────────────────────────────────────
    toggleBin(pane, bin) {
      const i = pane.selectedBins.indexOf(bin);
      if (i === -1) pane.selectedBins.push(bin);
      else pane.selectedBins.splice(i, 1);
      this.updatePaneBarsSelection(pane);
      this.renderPaneSeries(pane);   // band depends on selection
      this.recompute();
    },

    clearSelection() {
      for (const p of this.panes) {
        p.selectedBins = [];
        this.updatePaneBarsSelection(p);
        this.renderPaneSeries(p);
      }
      this.recompute();
    },

    get selectedBinCount() {
      return this.panes.reduce((n, p) => n + p.selectedBins.length, 0);
    },

    saveLayout() { alert('Save layout — coming in Phase 3.'); },

    // ── Data loads ───────────────────────────────────────────────────────
    async loadPrice() {
      if (!this.ticker) return;
      this.priceLoading = true; this.priceError = '';
      try {
        const r = await fetch(`/api/ticker-analysis/price?ticker=${encodeURIComponent(this.ticker)}`);
        const j = await r.json();
        if (j.error) { this.priceError = j.error; TA_DATA._price = null; }
        else TA_DATA._price = { series: j.series, splits: j.splits };
      } catch (e) {
        console.error('[ticker-analysis] loadPrice failed:', e);
        this.priceError = 'load failed';
        TA_DATA._price = null;
      } finally {
        this.priceLoading = false;
      }
    },

    // ── Rendering ────────────────────────────────────────────────────────
    renderPane(pane) {
      this.renderPaneBars(pane);
      this.renderPaneSeries(pane);
    },

    destroyPaneCharts(id) {
      for (const store of [TA_CHARTS.bars, TA_CHARTS.series]) {
        if (store[id]) { store[id].destroy(); delete store[id]; }
      }
    },

    _baseScales() {
      return {
        x: { grid: { color: 'rgba(255,255,255,.05)' }, ticks: { color: '#777', font: { size: 9 }, autoSkip: true, maxTicksLimit: 10, maxRotation: 0 } },
        y: { grid: { color: 'rgba(255,255,255,.06)' }, ticks: { color: '#777', font: { size: 9 } } },
      };
    },

    // Full-width price chart (§5.1) ---------------------------------------
    renderPrice() {
      const cvs = document.getElementById('ta-price');
      if (!cvs) return;
      if (TA_CHARTS.price) { TA_CHARTS.price.destroy(); TA_CHARTS.price = null; }
      const pd = TA_DATA._price;
      if (!pd || !pd.series.length) return;

      const labels = pd.series.map(p => p.date);
      const idxByDate = {};
      labels.forEach((d, i) => { idxByDate[d] = i; });
      const splitIdx = pd.splits.map(s => ({ i: idxByDate[s.date], ratio: s.ratio })).filter(s => s.i != null);

      // "On price" overlays (§5.2): each checked pane's metric, normalized to
      // its own [min,max] and drawn on a hidden 0..1 right axis, distinct color.
      const overlays = this.panes.filter(p => p.onPrice && TA_DATA[p.id]);
      const ovlDatasets = overlays.map((p, k) => {
        const d = TA_DATA[p.id];
        let mn = Infinity, mx = -Infinity;
        for (const pt of d.series) { if (pt.val < mn) mn = pt.val; if (pt.val > mx) mx = pt.val; }
        const rng = mx > mn ? mx - mn : 1;
        const arr = new Array(labels.length).fill(null);
        for (const pt of d.series) { const i = idxByDate[pt.date]; if (i != null) arr[i] = (pt.val - mn) / rng; }
        return {
          label: p.metric, data: arr, yAxisID: 'ovl',
          borderColor: TA_OVERLAY_COLORS[k % TA_OVERLAY_COLORS.length],
          borderWidth: 1, pointRadius: 0, tension: 0, spanGaps: true,
        };
      });

      const scales = this._baseScales();
      scales.ovl = { position: 'right', display: false, min: 0, max: 1 };

      TA_CHARTS.price = new Chart(cvs.getContext('2d'), {
        type: 'line',
        data: { labels, datasets: [
          { label: this.ticker, data: pd.series.map(p => p.close),
            borderColor: '#bdc3c7', borderWidth: 1, pointRadius: 0, tension: 0 },
          ...ovlDatasets,
        ] },
        options: {
          responsive: true, maintainAspectRatio: false, animation: false,
          plugins: {
            legend: { display: overlays.length > 0, position: 'top',
              labels: { color: '#c8c8c8', font: { size: 9 }, boxWidth: 10, boxHeight: 2,
                        filter: item => item.datasetIndex !== 0 } },
            tooltip: { mode: 'index', intersect: false,
              callbacks: { title: items => labels[items[0].dataIndex],
                label: it => it.datasetIndex === 0
                  ? 'close ' + it.parsed.y
                  : it.dataset.label + ' (norm) ' + (it.parsed.y == null ? '—' : it.parsed.y.toFixed(2)) } },
          },
          scales,
        },
        plugins: [{
          id: 'taPriceOverlay',
          beforeDatasetsDraw(chart) {
            const { ctx, chartArea, scales } = chart;
            const x = scales.x;
            // Bin-highlight bands (confluence = opacity by overlap; union = flat)
            const hl = chart.$taHL;
            if (hl && hl.entries.length) {
              ctx.save();
              for (const [i, cnt] of hl.entries) {
                const px = x.getPixelForValue(i);
                const alpha = hl.mode === 'union' ? 0.16 : Math.min(0.40, 0.12 + 0.09 * (cnt - 1) + 0.09);
                ctx.fillStyle = `rgba(52,152,219,${alpha})`;
                ctx.fillRect(px - 1, chartArea.top, 2, chartArea.bottom - chartArea.top);
              }
              ctx.restore();
            }
            // Split markers (pink dashed verticals)
            if (splitIdx.length) {
              ctx.save();
              ctx.setLineDash([3, 3]); ctx.strokeStyle = 'rgba(232,67,147,.55)'; ctx.lineWidth = 1;
              for (const s of splitIdx) {
                const px = x.getPixelForValue(s.i);
                ctx.beginPath(); ctx.moveTo(px, chartArea.top); ctx.lineTo(px, chartArea.bottom); ctx.stroke();
              }
              ctx.restore();
            }
          },
        }],
      });
      this.updatePriceHighlight();
    },

    updatePriceHighlight() {
      const chart = TA_CHARTS.price;
      const pd = TA_DATA._price;
      if (!chart || !pd) return;
      const idxByDate = {};
      pd.series.forEach((p, i) => { idxByDate[p.date] = i; });
      // date -> overlap count across panes with a selection
      const counts = new Map();
      for (const pane of this.panes) {
        const d = TA_DATA[pane.id];
        if (!d || !pane.selectedBins.length) continue;
        const sel = new Set(pane.selectedBins);
        for (const pt of d.series) {
          if (sel.has(pt.bin)) counts.set(pt.date, (counts.get(pt.date) || 0) + 1);
        }
      }
      const entries = [];
      for (const [date, cnt] of counts) {
        const i = idxByDate[date];
        if (i != null) entries.push([i, cnt]);
      }
      chart.$taHL = { entries, mode: this.shadeMode };
      chart.update('none');
    },

    // Metric pane — sub-pane A: 20-bin fwd-P&L bars (§5.2) -----------------
    renderPaneBars(pane) {
      const cvs = document.getElementById(`ta-bars-${pane.id}`);
      if (!cvs) return;
      if (TA_CHARTS.bars[pane.id]) { TA_CHARTS.bars[pane.id].destroy(); delete TA_CHARTS.bars[pane.id]; }
      const d = TA_DATA[pane.id];
      if (!d) return;

      const labels = Array.from({ length: 20 }, (_, i) => i + 1);
      const vals = labels.map(b => { const m = d.bins[b - 1]; return m && m.avg_ret != null ? m.avg_ret * 100 : 0; });
      const self = this;

      TA_CHARTS.bars[pane.id] = new Chart(cvs.getContext('2d'), {
        type: 'bar',
        data: { labels, datasets: [{
          data: vals,
          backgroundColor: vals.map(v => v >= 0 ? 'rgba(52,152,219,.75)' : 'rgba(232,67,147,.75)'),
          borderColor: labels.map(b => pane.selectedBins.includes(b) ? '#fff' : 'transparent'),
          borderWidth: labels.map(b => pane.selectedBins.includes(b) ? 2 : 0),
        }] },
        options: {
          responsive: true, maintainAspectRatio: false, animation: false,
          onClick: (evt, els) => { if (els.length) self.toggleBin(pane, els[0].index + 1); },
          plugins: {
            legend: { display: false },
            tooltip: { callbacks: {
              title: items => `Bin ${items[0].label}`,
              label: it => {
                const m = d.bins[it.dataIndex];
                if (!m) return 'no data';
                return [`range [${m.lo}, ${m.hi}]`,
                        `avg ${(m.avg_ret * 100).toFixed(2)}%`,
                        `win ${(m.win_rate * 100).toFixed(0)}%`,
                        `n ${m.n}`];
              },
            } },
          },
          scales: {
            x: { grid: { display: false }, ticks: { color: '#777', font: { size: 8 } } },
            y: { grid: { color: 'rgba(255,255,255,.06)' }, ticks: { color: '#777', font: { size: 8 }, callback: v => v + '%' } },
          },
        },
        plugins: [{
          id: 'taTodayMarker',
          afterDatasetsDraw(chart) {
            if (!d.today) return;
            const { ctx, chartArea, scales } = chart;
            const x = scales.x;
            const bin = d.today.bin, frac = d.today.frac;
            const center = x.getPixelForValue(bin - 1);
            const band = Math.abs(x.getPixelForValue(1) - x.getPixelForValue(0)) || 12;
            const px = center - band / 2 + frac * band;
            const top = chartArea.top + 2;
            ctx.save();
            // downward triangle
            ctx.fillStyle = '#f1c40f';
            ctx.beginPath();
            ctx.moveTo(px, top + 7); ctx.lineTo(px - 4, top); ctx.lineTo(px + 4, top); ctx.closePath(); ctx.fill();
            // thin tick descending into the bar
            ctx.strokeStyle = 'rgba(241,196,15,.85)'; ctx.lineWidth = 1;
            ctx.beginPath(); ctx.moveTo(px, top + 7); ctx.lineTo(px, chartArea.bottom); ctx.stroke();
            ctx.restore();
          },
        }],
      });
    },

    updatePaneBarsSelection(pane) {
      const chart = TA_CHARTS.bars[pane.id];
      if (!chart) return;
      const labels = chart.data.labels;
      const ds = chart.data.datasets[0];
      ds.borderColor = labels.map(b => pane.selectedBins.includes(b) ? '#fff' : 'transparent');
      ds.borderWidth = labels.map(b => pane.selectedBins.includes(b) ? 2 : 0);
      chart.update('none');
    },

    // Metric pane — sub-pane B: value over time (§5.2) --------------------
    renderPaneSeries(pane) {
      const cvs = document.getElementById(`ta-series-${pane.id}`);
      if (!cvs) return;
      if (TA_CHARTS.series[pane.id]) { TA_CHARTS.series[pane.id].destroy(); delete TA_CHARTS.series[pane.id]; }
      const d = TA_DATA[pane.id];
      if (!d) return;

      // Selected level = the ACTUAL bin cutoff edge of the selected span,
      // not its midpoint. The line sits at the edge facing the center of the
      // distribution, so the shade covers exactly the selected extreme:
      //   • high bins (e.g. 18–20) → line at the LOW edge (min lo of the
      //     lowest selected bin); everything ABOVE is the selected zone.
      //   • low bins (e.g. 1–3)    → line at the HIGH edge (max hi of the
      //     highest selected bin); everything BELOW is the selected zone.
      // Above is always shaded blue, below pink (theme convention).
      let level = null;
      if (pane.selectedBins.length) {
        let lo = Infinity, hi = -Infinity, binSum = 0;
        for (const b of pane.selectedBins) {
          const m = d.bins[b - 1];
          if (m) { lo = Math.min(lo, m.lo); hi = Math.max(hi, m.hi); }
          binSum += b;
        }
        if (lo !== Infinity) {
          const avgBin = binSum / pane.selectedBins.length;
          level = avgBin >= 10.5 ? lo : hi;   // 10.5 = median of bins 1–20
        }
      }

      const labels = d.series.map(p => p.date);
      const vals = d.series.map(p => p.val);
      const ptRadius = vals.map((_, i) => i === vals.length - 1 ? 3 : 0);
      const ptColor = vals.map((_, i) => i === vals.length - 1 ? '#f1c40f' : 'transparent');

      TA_CHARTS.series[pane.id] = new Chart(cvs.getContext('2d'), {
        type: 'line',
        data: { labels, datasets: [{
          // Faint connector through every daily point; the solid spike extent
          // is drawn by the taMinMax plugin below.
          data: vals, borderColor: 'rgba(52,152,219,.30)', borderWidth: 1, tension: 0,
          pointRadius: ptRadius, pointBackgroundColor: ptColor,
        }] },
        options: {
          responsive: true, maintainAspectRatio: false, animation: false,
          plugins: { legend: { display: false }, tooltip: { mode: 'index', intersect: false,
            callbacks: { title: items => labels[items[0].dataIndex], label: it => 'val ' + it.parsed.y } } },
          scales: this._baseScales(),
        },
        plugins: [{
          id: 'taSelectedBand',
          beforeDatasetsDraw(chart) {
            if (level == null) return;
            const { ctx, chartArea, scales } = chart;
            const yPix = scales.y.getPixelForValue(level);
            const yc = Math.max(chartArea.top, Math.min(chartArea.bottom, yPix));
            ctx.save();
            ctx.fillStyle = 'rgba(52,152,219,.10)';            // above → blue
            ctx.fillRect(chartArea.left, chartArea.top, chartArea.right - chartArea.left, yc - chartArea.top);
            ctx.fillStyle = 'rgba(232,67,147,.10)';            // below → pink
            ctx.fillRect(chartArea.left, yc, chartArea.right - chartArea.left, chartArea.bottom - yc);
            ctx.strokeStyle = 'rgba(255,255,255,.35)'; ctx.setLineDash([4, 3]); ctx.lineWidth = 1;
            ctx.beginPath(); ctx.moveTo(chartArea.left, yc); ctx.lineTo(chartArea.right, yc); ctx.stroke();
            ctx.restore();
          },
        }, {
          // Min/max decimation: bucket every daily point by horizontal pixel
          // and draw a vertical line from the bucket's min to its max. Any
          // bucket containing a spike shows that spike's full vertical extent
          // at any pane width — the full daily series, no thinning, no slider.
          id: 'taMinMax',
          afterDatasetsDraw(chart) {
            const { ctx, scales } = chart;
            const xs = scales.x, ys = scales.y;
            const pts = d.series;
            if (!pts.length) return;
            const buckets = new Map();   // px -> {min, max}
            for (let i = 0; i < pts.length; i++) {
              const v = pts[i].val;
              const px = Math.round(xs.getPixelForValue(i));
              const b = buckets.get(px);
              if (!b) buckets.set(px, { min: v, max: v });
              else { if (v < b.min) b.min = v; if (v > b.max) b.max = v; }
            }
            ctx.save();
            ctx.strokeStyle = 'rgba(52,152,219,.9)'; ctx.lineWidth = 1;
            ctx.beginPath();
            for (const [px, b] of buckets) {
              ctx.moveTo(px + 0.5, ys.getPixelForValue(b.max));
              ctx.lineTo(px + 0.5, ys.getPixelForValue(b.min));
            }
            ctx.stroke();
            ctx.restore();
          },
        }],
      });
    },

    // ── Stat strip (§4) — client-side union/dedup ────────────────────────
    recompute() {
      this.updatePriceHighlight();
      this.computeStats();
    },

    computeStats() {
      // Shared date→ret map (ret is identical across metrics for a horizon).
      let retByDate = null;
      for (const p of this.panes) {
        const d = TA_DATA[p.id];
        if (d && d.series.length) {
          retByDate = {};
          for (const pt of d.series) if (pt.ret != null) retByDate[pt.date] = pt.ret;
          break;
        }
      }
      if (!retByDate) { this.stats = []; this.statLabel = 'no data'; return; }

      const anySel = this.panes.some(p => p.selectedBins.length);
      let dates;
      if (anySel) {
        const set = new Set();
        for (const p of this.panes) {
          const d = TA_DATA[p.id];
          if (!d || !p.selectedBins.length) continue;
          const sel = new Set(p.selectedBins);
          for (const pt of d.series) if (sel.has(pt.bin)) set.add(pt.date);
        }
        dates = [...set];
      } else {
        dates = Object.keys(retByDate);
      }

      const rets = [];
      let minMs = Infinity, maxMs = -Infinity;
      for (const dt of dates) {
        const r = retByDate[dt];
        if (r != null) rets.push(r);
        const ms = new Date(dt).getTime();
        if (ms < minMs) minMs = ms;
        if (ms > maxMs) maxMs = ms;
      }

      const s = taStats(rets);
      this.statLabel = anySel
        ? `${dates.length} dates · union of selected bins`
        : 'all dates';
      if (!s) { this.stats = []; return; }

      const years = maxMs > minMs ? (maxMs - minMs) / (365.25 * 86400000) : 1;
      const trdYr = years > 0 ? s.n / years : s.n;
      const posCls = v => (v >= 0 ? 'a' : 'pink');

      this.stats = [
        { label: 'N',        value: String(s.n),                 cls: '' },
        { label: 'Avg Ret',  value: taSigned(s.mean),            cls: posCls(s.mean) },
        { label: 'Median',   value: taSigned(s.median),          cls: posCls(s.median) },
        { label: 'Std Dev',  value: taPct(s.std),                cls: '' },
        { label: 'P5',       value: taSigned(s.p5),              cls: 'pink' },
        { label: 'P95',      value: taSigned(s.p95),             cls: 'a' },
        { label: 'Win %',    value: (s.win_rate * 100).toFixed(1) + '%', cls: posCls(s.win_rate - 0.5) },
        { label: '# Win',    value: String(s.n_win),             cls: '' },
        { label: 'Avg Win',  value: taSigned(s.avg_win),         cls: 'a' },
        { label: 'Avg Loss', value: taSigned(s.avg_loss),        cls: 'pink' },
        { label: 'Trd/Yr',   value: trdYr.toFixed(0),            cls: '' },
      ];
    },
  }));
});
