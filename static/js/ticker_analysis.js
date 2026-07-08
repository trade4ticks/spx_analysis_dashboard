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
 *   • saved layouts: name/save/apply/delete the pane set (ticker-agnostic)
 *   • option chain (split-adjusted, cached, /chain/* endpoints): OI profile,
 *     ΔOI-by-strike, vol-vs-OI scatter, strike×DTE heatmap, flow map,
 *     3D surface (Three.js), IV smile, IV term structure
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
const TA_CHARTS = { price: null, bars: {}, series: {}, chain: null, surface: null,
                    doi: null, voloi: null, smile: null, ivterm: null };
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

    priceFullscreen: false,        // price chart expanded to viewport
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

    layouts: [],                   // [{id, name, layout, created_at}] saved layouts
    selectedLayoutId: '',          // control-bar layout selector

    // Option chain (§5.4): OI profile (P4a) + strike×DTE heatmap (P4b) + flow (P4c)
    chainView: 'profile',          // 'profile' | 'strikeDte' | 'flow'
    chainMetric: 'oi',             // heatmap: 'oi' | 'vol'
    chainFlowMode: 'oi',           // flow: 'oi'|'vol'|'voloi'|'doi'|'dvol'
    chainFlowLookback: 126,        // flow window (sessions) — 6m
    chainFlowN: 1,                 // flow Δ window (sessions)
    chainFlow: null,
    chainSurfaceMetric: 'oi',      // surface: 'oi'|'vol'
    chainSurfaceLookback: 126,     // surface window (sessions)
    chainSurface: null,
    chainDoiN: 5,                  // ΔOI window (sessions)
    chainDoi: null,
    chainVolOi: null,
    chainSmile: null,
    chainIvTermLookback: 252,      // IV-term window (sessions)
    chainIvTerm: null,
    chainDates: [],
    chainDateIdx: 0,
    chainDteMin: 0,
    chainDteMax: 3650,
    chainMoneyness: '',            // '' = off; else ± percent
    chainProfile: null,
    chainHeatmap: null,
    chainLoading: false,
    chainError: '',
    _chainDebounce: null,

    // ── Lifecycle ────────────────────────────────────────────────────────
    async init() {
      await Promise.all([this.loadTickers(), this.loadMetricOptions(), this.loadLayouts()]);
      if (this.ticker) await this.loadTicker();
    },

    async loadTickers() {
      try {
        const r = await fetch('/api/ticker-analysis/tickers');
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        this.tickers = await r.json();
        if (this.tickers.length && !this.ticker) {
          this.ticker = this.tickers.includes('AAPL') ? 'AAPL' : this.tickers[0];
        }
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
      // Chain section loads independently (parquet-backed, cached) so it
      // never delays the metric layer.
      this.loadChainDates().then(() => this.chainReload());
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
      this.panes.push({
        id,
        metric: wanted,
        loading: false,
        error: '',
        selectedBins: [],
        today: null,
        onPrice: false,
      });
      // IMPORTANT: mutate the REACTIVE element Alpine created in the array,
      // not the raw literal above. Setting loading/error/today on the raw
      // object would not trigger the template (it stays stuck on "Loading…").
      const pane = this.panes[this.panes.length - 1];
      if (render && this.ticker) {
        this.loadPaneData(pane).then(() => {
          // Guard against the <select> reconciling pane.metric back to its
          // first option before its own option renders (see paneMetricGroups).
          if (pane.metric !== wanted) pane.metric = wanted;
          this._renderPaneWhenReady(pane);
        });
      }
    },

    // Render a pane's charts once its canvas is actually visible in the
    // layout tree. A freshly-added pane is hidden (x-show, loading=true)
    // while its data loads, so drawing immediately would target a zero-size
    // canvas and show nothing — the reason a new pane came up blank until
    // you changed its metric. Poll across animation frames until laid out.
    _renderPaneWhenReady(pane, tries = 0) {
      const cvs = document.getElementById(`ta-bars-${pane.id}`);
      if ((cvs && cvs.offsetParent !== null) || tries >= 30) {
        this.renderPane(pane);
        this.recompute();
        return;
      }
      requestAnimationFrame(() => this._renderPaneWhenReady(pane, tries + 1));
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
      this.loadPaneData(pane).then(() => {
        this._renderPaneWhenReady(pane);
        if (pane.onPrice) this.$nextTick(() => this.renderPrice());   // overlay follows the new metric
      });
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

    // ── Saved layouts (§6) ───────────────────────────────────────────────
    async loadLayouts() {
      try {
        const r = await fetch('/api/ticker-analysis/layouts');
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        this.layouts = await r.json();
      } catch (e) {
        console.error('[ticker-analysis] loadLayouts failed:', e);
        this.layouts = [];
      }
    },

    async saveLayout() {
      const current = this.layouts.find(l => String(l.id) === String(this.selectedLayoutId));
      const name = prompt('Save layout as:', current ? current.name : '');
      if (!name || !name.trim()) return;
      const layout = {
        panes: this.panes.map(p => ({ metric: p.metric, onPrice: !!p.onPrice })),
        shadeMode: this.shadeMode,
        horizon: this.horizon,
      };
      try {
        const r = await fetch('/api/ticker-analysis/layouts', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ name: name.trim(), layout }),
        });
        const j = await r.json();
        if (j.error) { alert(j.error); return; }
        await this.loadLayouts();
        this.selectedLayoutId = String(j.id);
      } catch (e) {
        console.error('[ticker-analysis] saveLayout failed:', e);
        alert('Save failed');
      }
    },

    async deleteLayout() {
      const lay = this.layouts.find(l => String(l.id) === String(this.selectedLayoutId));
      if (!lay) return;
      if (!confirm(`Delete layout "${lay.name}"?`)) return;
      try {
        await fetch(`/api/ticker-analysis/layouts/${lay.id}`, { method: 'DELETE' });
      } catch (e) {
        console.error('[ticker-analysis] deleteLayout failed:', e);
      }
      this.selectedLayoutId = '';
      await this.loadLayouts();
    },

    onLayoutSelect() {
      const lay = this.layouts.find(l => String(l.id) === String(this.selectedLayoutId));
      if (lay) this.applyLayout(lay.layout);
    },

    applyLayout(layout) {
      if (!layout) return;
      // Tear down current panes/charts.
      for (const p of this.panes) this.destroyPaneCharts(p.id);
      for (const p of this.panes) delete TA_DATA[p.id];
      this.panes = [];

      if (layout.shadeMode) this.shadeMode = layout.shadeMode;
      if (layout.horizon && this.horizons.includes(layout.horizon)) this.horizon = layout.horizon;

      for (const pd of (layout.panes || [])) {
        this.addPane(pd.metric, /*render=*/false);
        this.panes[this.panes.length - 1].onPrice = !!pd.onPrice;
      }

      if (!this.ticker) return;
      Promise.all(this.panes.map(p => this.loadPaneData(p))).then(() => {
        this.$nextTick(() => {
          for (const p of this.panes) this.renderPane(p);
          this.renderPrice();     // rebuilds overlays from onPrice flags
          this.recompute();
        });
      });
    },

    // ── Option chain — OI-by-strike profile (§5.4) ───────────────────────
    async loadChainDates() {
      if (!this.ticker) return;
      try {
        const r = await fetch(`/api/ticker-analysis/chain/dates?ticker=${encodeURIComponent(this.ticker)}`);
        const j = await r.json();
        this.chainDates = j.dates || [];
        this.chainDateIdx = Math.max(0, this.chainDates.length - 1);   // latest
      } catch (e) {
        console.error('[ticker-analysis] loadChainDates failed:', e);
        this.chainDates = [];
      }
    },

    get chainDate() { return this.chainDates[this.chainDateIdx] || ''; },

    chainScrub() {
      clearTimeout(this._chainDebounce);
      this._chainDebounce = setTimeout(() => this.chainReload(), 300);
    },

    // Route the shared controls (date/moneyness/Recompute) to the active view.
    chainReload(force = false) {
      if (this.chainView === 'strikeDte') this.loadChainHeatmap(force);
      else if (this.chainView === 'flow') this.loadChainFlow(force);
      else if (this.chainView === 'surface') this.loadChainSurface(force);
      else if (this.chainView === 'doi') this.loadChainDoi(force);
      else if (this.chainView === 'voloi') this.loadChainVolOi(force);
      else if (this.chainView === 'smile') this.loadChainSmile(force);
      else if (this.chainView === 'ivterm') this.loadChainIvTerm(force);
      else this.loadChainProfile(force);
    },

    setChainView(v) {
      if (this.chainView === v) return;
      if (this.chainView === 'surface' && v !== 'surface') this.disposeSurface();  // stop the render loop
      this.chainView = v;
      this.chainReload();
    },

    async loadChainProfile(force = false) {
      const date = this.chainDate;
      if (!this.ticker || !date) return;
      this.chainLoading = true; this.chainError = '';
      let url = `/api/ticker-analysis/chain/oi-profile?ticker=${encodeURIComponent(this.ticker)}`
              + `&date=${date}&dte_min=${this.chainDteMin || 0}&dte_max=${this.chainDteMax || 3650}`;
      const mPct = parseFloat(this.chainMoneyness);
      if (!isNaN(mPct) && mPct > 0) url += `&moneyness=${mPct / 100}`;
      if (force) url += '&force=1';
      try {
        const r = await fetch(url);
        const j = await r.json();
        if (j.error) { this.chainError = j.error; this.chainProfile = null; }
        else this.chainProfile = j;
      } catch (e) {
        console.error('[ticker-analysis] loadChainProfile failed:', e);
        this.chainError = 'load failed'; this.chainProfile = null;
      } finally {
        this.chainLoading = false;
        this.$nextTick(() => this._renderChainWhenReady('ta-chain-profile', () => this.renderChainProfile()));
      }
    },

    renderChainProfile() {
      const cvs = document.getElementById('ta-chain-profile');
      if (!cvs) return;
      if (TA_CHARTS.chain) { TA_CHARTS.chain.destroy(); TA_CHARTS.chain = null; }
      const p = this.chainProfile;
      if (!p || !p.strikes || !p.strikes.length) return;

      const labels = p.strikes.map(s => s.strike);
      const calls = p.strikes.map(s => s.call_oi);       // positive → right (blue)
      const puts = p.strikes.map(s => -s.put_oi);        // negative → left  (pink)
      const spot = p.spot;

      TA_CHARTS.chain = new Chart(cvs.getContext('2d'), {
        type: 'bar',
        data: { labels, datasets: [
          { label: 'Puts', data: puts, backgroundColor: 'rgba(232,67,147,.75)' },
          { label: 'Calls', data: calls, backgroundColor: 'rgba(52,152,219,.75)' },
        ] },
        options: {
          indexAxis: 'y',
          responsive: true, maintainAspectRatio: false, animation: false,
          scales: {
            x: { stacked: true, grid: { color: 'rgba(255,255,255,.06)' },
                 ticks: { color: '#777', font: { size: 9 }, callback: v => Math.abs(v) } },
            y: { stacked: true, reverse: true, grid: { display: false },
                 ticks: { color: '#777', font: { size: 8 }, autoSkip: true, maxTicksLimit: 26,
                          callback: (v, i) => labels[i] } },
          },
          plugins: {
            legend: { display: true, position: 'top',
                      labels: { color: '#c8c8c8', font: { size: 9 }, boxWidth: 10 } },
            tooltip: { callbacks: {
              title: items => 'Strike ' + labels[items[0].dataIndex],
              label: it => `${it.dataset.label}: ${Math.abs(it.parsed.x).toLocaleString()}` } },
          },
        },
        plugins: [{
          id: 'taSpotLine',
          afterDatasetsDraw(chart) {
            if (spot == null) return;
            const { ctx, chartArea, scales } = chart;
            const y = scales.y;
            // Interpolate spot's y-pixel between the two bracketing strike rows.
            let idx = null;
            for (let i = 0; i < labels.length - 1; i++) {
              if (spot >= labels[i] && spot <= labels[i + 1]) {
                const span = labels[i + 1] - labels[i] || 1;
                idx = i + (spot - labels[i]) / span; break;
              }
            }
            if (idx == null) idx = spot < labels[0] ? 0 : labels.length - 1;
            const i0 = Math.floor(idx), i1 = Math.min(labels.length - 1, i0 + 1), f = idx - i0;
            const py = y.getPixelForValue(i0) + f * (y.getPixelForValue(i1) - y.getPixelForValue(i0));
            ctx.save();
            ctx.strokeStyle = 'rgba(255,255,255,.55)'; ctx.setLineDash([4, 3]); ctx.lineWidth = 1;
            ctx.beginPath(); ctx.moveTo(chartArea.left, py); ctx.lineTo(chartArea.right, py); ctx.stroke();
            ctx.setLineDash([]); ctx.fillStyle = '#fff'; ctx.font = '9px sans-serif';
            ctx.fillText('spot ' + spot, chartArea.left + 4, py - 3);
            ctx.restore();
          },
        }],
      });
    },

    // Strike×DTE heatmap (P4b) --------------------------------------------
    async loadChainHeatmap(force = false) {
      const date = this.chainDate;
      if (!this.ticker || !date) return;
      this.chainLoading = true; this.chainError = '';
      let url = `/api/ticker-analysis/chain/strike-dte?ticker=${encodeURIComponent(this.ticker)}`
              + `&date=${date}&metric=${this.chainMetric}`;
      const mPct = parseFloat(this.chainMoneyness);
      if (!isNaN(mPct) && mPct > 0) url += `&moneyness=${mPct / 100}`;
      if (force) url += '&force=1';
      try {
        const r = await fetch(url);
        const j = await r.json();
        if (j.error) { this.chainError = j.error; this.chainHeatmap = null; }
        else this.chainHeatmap = j;
      } catch (e) {
        console.error('[ticker-analysis] loadChainHeatmap failed:', e);
        this.chainError = 'load failed'; this.chainHeatmap = null;
      } finally {
        this.chainLoading = false;
        this.$nextTick(() => this._renderChainWhenReady('ta-chain-heatmap', () => this.renderChainHeatmap()));
      }
    },

    // Strike×DTE heatmap, canvas-rendered so each cell's color is provably a
    // function of THAT cell's own value (the DOM-grid version mis-bound styles
    // across cells via nested x-for). rows = non-empty DTE buckets, cols =
    // strikes, sequential blue scaled to a p95 ceiling of the displayed cells.
    renderChainHeatmap() {
      const cvs = document.getElementById('ta-chain-heatmap');
      if (!cvs || !cvs.parentElement) return;
      const hm = this.chainHeatmap;
      const wrap = cvs.parentElement;
      const W = wrap.clientWidth, H = wrap.clientHeight;
      if (!W || !H) { if (this.chainView === 'strikeDte') requestAnimationFrame(() => this.renderChainHeatmap()); return; }
      const dpr = window.devicePixelRatio || 1;
      cvs.width = W * dpr; cvs.height = H * dpr; cvs.style.width = W + 'px'; cvs.style.height = H + 'px';
      const ctx = cvs.getContext('2d'); ctx.setTransform(dpr, 0, 0, dpr, 0, 0); ctx.clearRect(0, 0, W, H);
      if (!hm || !hm.strikes || !hm.strikes.length) return;

      const rowsIdx = [];
      for (let i = 0; i < hm.rows.length; i++) if (hm.rows[i].some(v => v > 0)) rowsIdx.push(i);
      const nr = rowsIdx.length, nc = hm.strikes.length;
      if (!nr) return;

      const cells = [];
      for (const i of rowsIdx) for (const v of hm.rows[i]) if (v > 0) cells.push(v);
      const scale = this._percentile(cells, 95) || hm.max || 1;

      const mL = 58, mB = 18, mT = 6, mR = 8;
      const plotW = W - mL - mR, plotH = H - mT - mB;
      const cw = plotW / nc, ch = plotH / nr;

      for (let r = 0; r < nr; r++) {
        const rowvals = hm.rows[rowsIdx[r]];
        for (let c = 0; c < nc; c++) {
          const v = rowvals[c];
          if (!v) continue;
          const a = 0.06 + 0.9 * Math.min(1, v / scale);   // monotone in this cell's own value
          ctx.fillStyle = `rgba(52,152,219,${a.toFixed(3)})`;
          ctx.fillRect(mL + c * cw, mT + r * ch, Math.ceil(cw) + 0.5, Math.ceil(ch) + 0.5);
        }
      }

      // Spot vertical line at the nearest strike column.
      if (hm.spot != null) {
        let best = 0, bd = Infinity;
        hm.strikes.forEach((s, i) => { const dd = Math.abs(s - hm.spot); if (dd < bd) { bd = dd; best = i; } });
        const x = mL + (best + 0.5) * cw;
        ctx.save();
        ctx.strokeStyle = 'rgba(255,255,255,.7)'; ctx.setLineDash([4, 3]); ctx.lineWidth = 1;
        ctx.beginPath(); ctx.moveTo(x, mT); ctx.lineTo(x, mT + plotH); ctx.stroke();
        ctx.setLineDash([]); ctx.fillStyle = '#fff'; ctx.font = '9px sans-serif';
        ctx.textAlign = 'center'; ctx.textBaseline = 'bottom'; ctx.fillText('spot ' + hm.spot, x, mT + 9);
        ctx.restore();
      }

      // Axis labels: DTE buckets (left), strikes (bottom).
      ctx.fillStyle = '#777'; ctx.font = '9px sans-serif';
      ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
      for (let r = 0; r < nr; r++) ctx.fillText(hm.dte_buckets[rowsIdx[r]], mL - 4, mT + (r + 0.5) * ch);
      ctx.textAlign = 'center'; ctx.textBaseline = 'top';
      const colStep = Math.max(1, Math.round(nc / 10));
      for (let c = 0; c < nc; c += colStep) ctx.fillText(hm.strikes[c], mL + (c + 0.5) * cw, mT + plotH + 3);

      cvs._geo = { mL, mT, cw, ch, nc, nr, rowsIdx };
      cvs.onmousemove = (ev) => {
        const g = cvs._geo; if (!g) return;
        const rect = cvs.getBoundingClientRect();
        const c = Math.floor((ev.clientX - rect.left - g.mL) / g.cw);
        const r = Math.floor((ev.clientY - rect.top - g.mT) / g.ch);
        const tip = document.getElementById('ta-hm-tip'); if (!tip) return;
        if (c < 0 || c >= g.nc || r < 0 || r >= g.nr) { tip.style.display = 'none'; return; }
        const v = hm.rows[g.rowsIdx[r]][c];
        tip.style.display = 'block';
        tip.style.left = (ev.clientX - rect.left + 12) + 'px';
        tip.style.top = (ev.clientY - rect.top + 12) + 'px';
        tip.innerHTML = `DTE ${hm.dte_buckets[g.rowsIdx[r]]}<br>strike ${hm.strikes[c]}<br>${(v || 0).toLocaleString()}`;
      };
      cvs.onmouseleave = () => { const tip = document.getElementById('ta-hm-tip'); if (tip) tip.style.display = 'none'; };
    },

    // 95th-percentile of an array (color-scale denominator that resists outliers).
    _percentile(arr, p) {
      if (!arr || !arr.length) return 0;
      const s = [...arr].sort((a, b) => a - b);
      const i = Math.min(s.length - 1, Math.floor((p / 100) * s.length));
      return s[i] || 0;
    },

    // Run a canvas render only once its element is laid out (visible in the
    // layout tree), so first-paint / spot-line-on-load is reliable (S2).
    _renderChainWhenReady(id, fn, tries = 0) {
      const el = document.getElementById(id);
      if ((el && el.offsetParent !== null) || tries >= 30) { fn(); return; }
      requestAnimationFrame(() => this._renderChainWhenReady(id, fn, tries + 1));
    },

    // Flow map — strike × time (P4c) --------------------------------------
    async loadChainFlow(force = false) {
      const date = this.chainDate;
      if (!this.ticker || !date) return;
      this.chainLoading = true; this.chainError = '';
      let url = `/api/ticker-analysis/chain/flow?ticker=${encodeURIComponent(this.ticker)}`
              + `&date=${date}&mode=${this.chainFlowMode}&lookback=${this.chainFlowLookback}`
              + `&n=${this.chainFlowN}&dte_min=${this.chainDteMin || 0}&dte_max=${this.chainDteMax || 3650}`;
      const mPct = parseFloat(this.chainMoneyness);
      if (!isNaN(mPct) && mPct > 0) url += `&moneyness=${mPct / 100}`;
      if (force) url += '&force=1';
      try {
        const r = await fetch(url);
        const j = await r.json();
        if (j.error) { this.chainError = j.error; this.chainFlow = null; }
        else this.chainFlow = j;
      } catch (e) {
        console.error('[ticker-analysis] loadChainFlow failed:', e);
        this.chainError = 'load failed'; this.chainFlow = null;
      } finally {
        this.chainLoading = false;
        this.$nextTick(() => this.renderChainFlow());
      }
    },

    renderChainFlow() {
      const cvs = document.getElementById('ta-chain-flow');
      if (!cvs || !cvs.parentElement) return;
      const f = this.chainFlow;
      const wrap = cvs.parentElement;
      const W = wrap.clientWidth, H = wrap.clientHeight;
      if (!W || !H) {
        // Wrap not laid out yet (view just became visible) — retry next frame.
        if (this.chainView === 'flow') requestAnimationFrame(() => this.renderChainFlow());
        return;
      }
      const dpr = window.devicePixelRatio || 1;
      cvs.width = W * dpr; cvs.height = H * dpr;
      cvs.style.width = W + 'px'; cvs.style.height = H + 'px';
      const ctx = cvs.getContext('2d');
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, W, H);
      if (!f || !f.strikes.length || !f.dates.length) return;

      const mL = 54, mB = 20, mT = 6, mR = 8;
      const plotW = W - mL - mR, plotH = H - mT - mB;
      const nc = f.dates.length, nr = f.strikes.length;
      const cw = plotW / nc, ch = plotH / nr;
      // S3: color-scale to a high percentile of the displayed cells (per mode).
      const cells = [];
      for (let r = 0; r < nr; r++) for (let c = 0; c < nc; c++) {
        const v = f.matrix[r][c];
        if (v != null) { const a = f.signed ? Math.abs(v) : v; if (a > 0) cells.push(a); }
      }
      const max = this._percentile(cells, 95) || f.max || 1;

      for (let r = 0; r < nr; r++) {
        for (let c = 0; c < nc; c++) {
          const v = f.matrix[r][c];
          if (v == null) continue;
          let color;
          if (f.signed) {
            const a = Math.min(1, Math.abs(v) / max);
            if (a <= 0) continue;
            color = v >= 0 ? `rgba(52,152,219,${(0.08 + 0.9 * a).toFixed(3)})`
                           : `rgba(232,67,147,${(0.08 + 0.9 * a).toFixed(3)})`;
          } else {
            const a = Math.min(1, v / max);
            if (a <= 0) continue;
            color = `rgba(52,152,219,${(0.06 + 0.9 * a).toFixed(3)})`;
          }
          ctx.fillStyle = color;
          ctx.fillRect(mL + c * cw, mT + r * ch, Math.ceil(cw) + 0.5, Math.ceil(ch) + 0.5);
        }
      }

      // Spot path across time
      if (f.spots) {
        ctx.strokeStyle = 'rgba(255,255,255,.8)'; ctx.lineWidth = 1;
        ctx.beginPath();
        let started = false;
        for (let c = 0; c < nc; c++) {
          const sp = f.spots[c];
          if (sp == null) continue;
          let best = 0, bd = Infinity;
          for (let r = 0; r < nr; r++) { const dd = Math.abs(f.strikes[r] - sp); if (dd < bd) { bd = dd; best = r; } }
          const x = mL + (c + 0.5) * cw, y = mT + (best + 0.5) * ch;
          if (!started) { ctx.moveTo(x, y); started = true; } else ctx.lineTo(x, y);
        }
        ctx.stroke();
      }

      // Axis labels
      ctx.fillStyle = '#777'; ctx.font = '9px sans-serif';
      ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
      const rowStep = Math.max(1, Math.round(nr / 12));
      for (let r = 0; r < nr; r += rowStep) ctx.fillText(f.strikes[r], mL - 4, mT + (r + 0.5) * ch);
      ctx.textAlign = 'center'; ctx.textBaseline = 'top';
      const colStep = Math.max(1, Math.round(nc / 8));
      for (let c = 0; c < nc; c += colStep) ctx.fillText(f.dates[c].slice(2), mL + (c + 0.5) * cw, mT + plotH + 3);

      cvs._geo = { mL, mT, cw, ch, nc, nr };
      cvs.onmousemove = (ev) => {
        const g = cvs._geo; if (!g) return;
        const rect = cvs.getBoundingClientRect();
        const c = Math.floor((ev.clientX - rect.left - g.mL) / g.cw);
        const r = Math.floor((ev.clientY - rect.top - g.mT) / g.ch);
        const tip = document.getElementById('ta-flow-tip');
        if (!tip) return;
        if (c < 0 || c >= g.nc || r < 0 || r >= g.nr) { tip.style.display = 'none'; return; }
        const v = f.matrix[r][c];
        tip.style.display = 'block';
        tip.style.left = (ev.clientX - rect.left + 12) + 'px';
        tip.style.top = (ev.clientY - rect.top + 12) + 'px';
        tip.innerHTML = `${f.dates[c]}<br>strike ${f.strikes[r]}<br>${v == null ? '—' : v.toLocaleString()}`;
      };
      cvs.onmouseleave = () => { const tip = document.getElementById('ta-flow-tip'); if (tip) tip.style.display = 'none'; };
    },

    // 3D surface — strike × time × signed qty (P4d) -----------------------
    async loadChainSurface(force = false) {
      const date = this.chainDate;
      if (!this.ticker || !date) return;
      this.chainLoading = true; this.chainError = '';
      let url = `/api/ticker-analysis/chain/surface?ticker=${encodeURIComponent(this.ticker)}`
              + `&date=${date}&metric=${this.chainSurfaceMetric}&lookback=${this.chainSurfaceLookback}`
              + `&dte_min=${this.chainDteMin || 0}&dte_max=${this.chainDteMax || 3650}`;
      const mPct = parseFloat(this.chainMoneyness);
      if (!isNaN(mPct) && mPct > 0) url += `&moneyness=${mPct / 100}`;
      if (force) url += '&force=1';
      try {
        const r = await fetch(url);
        const j = await r.json();
        if (j.error) { this.chainError = j.error; this.chainSurface = null; }
        else this.chainSurface = j;
      } catch (e) {
        console.error('[ticker-analysis] loadChainSurface failed:', e);
        this.chainError = 'load failed'; this.chainSurface = null;
      } finally {
        this.chainLoading = false;
        this.$nextTick(() => this.renderChainSurface());
      }
    },

    disposeSurface() {
      const s = TA_CHARTS.surface;
      if (!s) return;
      s.alive = false;
      if (s.animId) cancelAnimationFrame(s.animId);
      try { if (s.controls) s.controls.dispose(); } catch (e) {}
      try { if (s.renderer) { s.renderer.dispose(); s.renderer.domElement.remove(); } } catch (e) {}
      TA_CHARTS.surface = null;
    },

    renderChainSurface() {
      const host = document.getElementById('ta-chain-surface');
      if (!host) return;
      this.disposeSurface();
      const f = this.chainSurface;
      const Wd = host.clientWidth, Hd = host.clientHeight;
      if (!Wd || !Hd) {
        if (this.chainView === 'surface') requestAnimationFrame(() => this.renderChainSurface());
        return;
      }
      if (typeof THREE === 'undefined') { this.chainError = 'Three.js not loaded'; return; }
      if (!f || !f.strikes.length || !f.dates.length) return;

      const nS = f.strikes.length, nD = f.dates.length;
      const SX = 100, SZ = 100, SY = 42;          // grid extents / height scale
      const max = f.max || 1;
      const yAt = (r, c) => { const v = f.matrix[r][c]; return v == null ? 0 : (v / max) * SY; };

      const scene = new THREE.Scene();
      scene.background = new THREE.Color(0x1e1e1e);
      const camera = new THREE.PerspectiveCamera(55, Wd / Hd, 0.1, 2000);
      camera.position.set(SX * 1.0, SY * 2.0, SZ * 1.15);
      const renderer = new THREE.WebGLRenderer({ antialias: true });
      renderer.setPixelRatio(window.devicePixelRatio || 1);
      renderer.setSize(Wd, Hd);
      host.appendChild(renderer.domElement);

      scene.add(new THREE.AmbientLight(0xffffff, 0.65));
      const dir = new THREE.DirectionalLight(0xffffff, 0.7);
      dir.position.set(1, 2, 1); scene.add(dir);

      // Heightfield: X = strike (rows), Z = date (cols), Y = signed net.
      const blue = new THREE.Color(0x3498db), pink = new THREE.Color(0xe84393), base = new THREE.Color(0x2d2d2d);
      const positions = [], colors = [];
      const px = r => (nS > 1 ? r / (nS - 1) - 0.5 : 0) * SX;
      const pz = c => (nD > 1 ? c / (nD - 1) - 0.5 : 0) * SZ;
      for (let r = 0; r < nS; r++) {
        for (let c = 0; c < nD; c++) {
          const y = yAt(r, c);
          positions.push(px(r), y, pz(c));
          const t = Math.min(1, Math.abs(y) / SY);
          const col = base.clone().lerp(y >= 0 ? blue : pink, 0.25 + 0.75 * t);
          colors.push(col.r, col.g, col.b);
        }
      }
      const idx = [];
      const vi = (r, c) => r * nD + c;
      for (let r = 0; r < nS - 1; r++) {
        for (let c = 0; c < nD - 1; c++) {
          idx.push(vi(r, c), vi(r + 1, c), vi(r, c + 1));
          idx.push(vi(r + 1, c), vi(r + 1, c + 1), vi(r, c + 1));
        }
      }
      const geo = new THREE.BufferGeometry();
      geo.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
      geo.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
      geo.setIndex(idx);
      geo.computeVertexNormals();
      const mesh = new THREE.Mesh(geo, new THREE.MeshLambertMaterial({
        vertexColors: true, side: THREE.DoubleSide,
      }));
      scene.add(mesh);

      // Zero plane (strike axis) + grid.
      const grid = new THREE.GridHelper(Math.max(SX, SZ), 12, 0x555555, 0x333333);
      scene.add(grid);
      const plane = new THREE.Mesh(
        new THREE.PlaneGeometry(SX, SZ),
        new THREE.MeshBasicMaterial({ color: 0x888888, transparent: true, opacity: 0.06, side: THREE.DoubleSide }));
      plane.rotation.x = -Math.PI / 2; scene.add(plane);

      // Spot path on the zero plane.
      if (f.spots) {
        const pts = [];
        for (let c = 0; c < nD; c++) {
          const sp = f.spots[c]; if (sp == null) continue;
          let best = 0, bd = Infinity;
          for (let r = 0; r < nS; r++) { const dd = Math.abs(f.strikes[r] - sp); if (dd < bd) { bd = dd; best = r; } }
          pts.push(new THREE.Vector3(px(best), 0.6, pz(c)));
        }
        if (pts.length > 1) {
          const line = new THREE.Line(
            new THREE.BufferGeometry().setFromPoints(pts),
            new THREE.LineBasicMaterial({ color: 0xffffff }));
          scene.add(line);
        }
      }

      const controls = new THREE.OrbitControls(camera, renderer.domElement);
      controls.target.set(0, 0, 0);
      controls.enableDamping = true; controls.dampingFactor = 0.08;
      controls.update();

      const s = { renderer, controls, host, animId: 0, alive: true };
      TA_CHARTS.surface = s;
      const animate = () => {
        if (!s.alive) return;
        s.animId = requestAnimationFrame(animate);
        controls.update();
        renderer.render(scene, camera);
      };
      animate();
    },

    // ΔOI-by-strike (P4e) -------------------------------------------------
    async loadChainDoi(force = false) {
      const date = this.chainDate;
      if (!this.ticker || !date) return;
      this.chainLoading = true; this.chainError = '';
      let url = `/api/ticker-analysis/chain/doi-profile?ticker=${encodeURIComponent(this.ticker)}`
              + `&date=${date}&n=${this.chainDoiN}&dte_min=${this.chainDteMin || 0}&dte_max=${this.chainDteMax || 3650}`;
      const mPct = parseFloat(this.chainMoneyness);
      if (!isNaN(mPct) && mPct > 0) url += `&moneyness=${mPct / 100}`;
      if (force) url += '&force=1';
      try {
        const r = await fetch(url); const j = await r.json();
        if (j.error) { this.chainError = j.error; this.chainDoi = null; }
        else this.chainDoi = j;
      } catch (e) { console.error('[ticker-analysis] loadChainDoi failed:', e); this.chainError = 'load failed'; this.chainDoi = null; }
      finally { this.chainLoading = false; this.$nextTick(() => this._renderChainWhenReady('ta-chain-doi', () => this.renderChainDoi())); }
    },

    renderChainDoi() {
      const cvs = document.getElementById('ta-chain-doi');
      if (!cvs) return;
      if (TA_CHARTS.doi) { TA_CHARTS.doi.destroy(); TA_CHARTS.doi = null; }
      const p = this.chainDoi;
      if (!p || !p.strikes || !p.strikes.length) return;
      const labels = p.strikes.map(s => s.strike);
      const vals = p.strikes.map(s => s.doi);   // signed: + build (blue), − unwind (pink)
      TA_CHARTS.doi = new Chart(cvs.getContext('2d'), {
        type: 'bar',
        data: { labels, datasets: [{
          data: vals,
          backgroundColor: vals.map(v => v >= 0 ? 'rgba(52,152,219,.8)' : 'rgba(232,67,147,.8)'),
        }] },
        options: {
          indexAxis: 'y',
          responsive: true, maintainAspectRatio: false, animation: false,
          scales: {
            x: { grid: { color: 'rgba(255,255,255,.06)' }, ticks: { color: '#777', font: { size: 9 } } },
            y: { reverse: true, grid: { display: false }, ticks: { color: '#777', font: { size: 8 }, autoSkip: true, maxTicksLimit: 26, callback: (v, i) => labels[i] } },
          },
          plugins: {
            legend: { display: false },
            tooltip: { callbacks: { title: it => 'Strike ' + labels[it[0].dataIndex],
              label: it => 'ΔOI ' + (it.parsed.x >= 0 ? '+' : '') + it.parsed.x.toLocaleString() } },
          },
        },
        plugins: [this._profileSpotPlugin(labels, p.spot)],
      });
    },

    // Vol-vs-OI scatter (P4e) ---------------------------------------------
    async loadChainVolOi(force = false) {
      const date = this.chainDate;
      if (!this.ticker || !date) return;
      this.chainLoading = true; this.chainError = '';
      let url = `/api/ticker-analysis/chain/vol-oi?ticker=${encodeURIComponent(this.ticker)}`
              + `&date=${date}&dte_min=${this.chainDteMin || 0}&dte_max=${this.chainDteMax || 3650}`;
      const mPct = parseFloat(this.chainMoneyness);
      if (!isNaN(mPct) && mPct > 0) url += `&moneyness=${mPct / 100}`;
      if (force) url += '&force=1';
      try {
        const r = await fetch(url); const j = await r.json();
        if (j.error) { this.chainError = j.error; this.chainVolOi = null; }
        else this.chainVolOi = j;
      } catch (e) { console.error('[ticker-analysis] loadChainVolOi failed:', e); this.chainError = 'load failed'; this.chainVolOi = null; }
      finally { this.chainLoading = false; this.$nextTick(() => this._renderChainWhenReady('ta-chain-voloi', () => this.renderChainVolOi())); }
    },

    renderChainVolOi() {
      const cvs = document.getElementById('ta-chain-voloi');
      if (!cvs) return;
      if (TA_CHARTS.voloi) { TA_CHARTS.voloi.destroy(); TA_CHARTS.voloi = null; }
      const p = this.chainVolOi;
      if (!p || !p.points || !p.points.length) return;
      const pts = p.points.map(d => ({ x: d.oi, y: d.vol, s: d.strike }));
      // Fresh activity: volume >= standing OI (vol/OI ≥ 1) → accent; else dim.
      const colors = pts.map(d => (d.x > 0 && d.y / d.x >= 1) ? '#3498db' : 'rgba(200,200,200,.35)');
      const maxv = Math.max(1, ...pts.map(d => Math.max(d.x, d.y)));
      TA_CHARTS.voloi = new Chart(cvs.getContext('2d'), {
        data: {
          datasets: [
            { type: 'line', data: [{ x: 0, y: 0 }, { x: maxv, y: maxv }], borderColor: 'rgba(255,255,255,.25)',
              borderDash: [4, 4], borderWidth: 1, pointRadius: 0, fill: false },   // vol = OI reference
            { type: 'scatter', data: pts, pointBackgroundColor: colors, pointRadius: 3, borderWidth: 0 },
          ],
        },
        options: {
          responsive: true, maintainAspectRatio: false, animation: false,
          scales: {
            x: { title: { display: true, text: 'Open interest', color: '#888', font: { size: 9 } },
                 grid: { color: 'rgba(255,255,255,.06)' }, ticks: { color: '#777', font: { size: 9 } } },
            y: { title: { display: true, text: 'Volume (today)', color: '#888', font: { size: 9 } },
                 grid: { color: 'rgba(255,255,255,.06)' }, ticks: { color: '#777', font: { size: 9 } } },
          },
          plugins: {
            legend: { display: false },
            tooltip: { callbacks: { label: it => {
              const d = it.raw; if (d.s == null) return '';
              const vo = d.x > 0 ? (d.y / d.x).toFixed(2) : '∞';
              return `K ${d.s} · OI ${d.x.toLocaleString()} · vol ${d.y.toLocaleString()} · v/OI ${vo}`;
            } } },
          },
        },
      });
    },

    // Shared spot-line plugin for horizontal (strike-on-y) profiles.
    _profileSpotPlugin(labels, spot) {
      return {
        id: 'taProfileSpot',
        afterDatasetsDraw(chart) {
          if (spot == null || !labels.length) return;
          const { ctx, chartArea, scales } = chart;
          const y = scales.y;
          let idx = null;
          for (let i = 0; i < labels.length - 1; i++) {
            if ((spot >= labels[i] && spot <= labels[i + 1]) || (spot <= labels[i] && spot >= labels[i + 1])) {
              const span = labels[i + 1] - labels[i] || 1; idx = i + (spot - labels[i]) / span; break;
            }
          }
          if (idx == null) idx = spot < labels[0] ? 0 : labels.length - 1;
          const i0 = Math.floor(idx), i1 = Math.min(labels.length - 1, i0 + 1), f = idx - i0;
          const py = y.getPixelForValue(i0) + f * (y.getPixelForValue(i1) - y.getPixelForValue(i0));
          ctx.save();
          ctx.strokeStyle = 'rgba(255,255,255,.55)'; ctx.setLineDash([4, 3]); ctx.lineWidth = 1;
          ctx.beginPath(); ctx.moveTo(chartArea.left, py); ctx.lineTo(chartArea.right, py); ctx.stroke();
          ctx.setLineDash([]); ctx.fillStyle = '#fff'; ctx.font = '9px sans-serif';
          ctx.fillText('spot ' + spot, chartArea.left + 4, py - 3);
          ctx.restore();
        },
      };
    },

    // IV smile (P4e) ------------------------------------------------------
    async loadChainSmile(force = false) {
      const date = this.chainDate;
      if (!this.ticker || !date) return;
      this.chainLoading = true; this.chainError = '';
      let url = `/api/ticker-analysis/chain/iv-smile?ticker=${encodeURIComponent(this.ticker)}`
              + `&date=${date}&dte_min=${this.chainDteMin || 0}&dte_max=${this.chainDteMax || 3650}`;
      const mPct = parseFloat(this.chainMoneyness);
      if (!isNaN(mPct) && mPct > 0) url += `&moneyness=${mPct / 100}`;
      if (force) url += '&force=1';
      try {
        const r = await fetch(url); const j = await r.json();
        if (j.error) { this.chainError = j.error; this.chainSmile = null; }
        else this.chainSmile = j;
      } catch (e) { console.error('[ticker-analysis] loadChainSmile failed:', e); this.chainError = 'load failed'; this.chainSmile = null; }
      finally { this.chainLoading = false; this.$nextTick(() => this._renderChainWhenReady('ta-chain-smile', () => this.renderChainSmile())); }
    },

    renderChainSmile() {
      const cvs = document.getElementById('ta-chain-smile');
      if (!cvs) return;
      if (TA_CHARTS.smile) { TA_CHARTS.smile.destroy(); TA_CHARTS.smile = null; }
      const p = this.chainSmile;
      if (!p || !p.strikes || !p.strikes.length) return;
      const calls = p.strikes.map(s => ({ x: s.strike, y: s.call_iv })).filter(d => d.y != null);
      const puts = p.strikes.map(s => ({ x: s.strike, y: s.put_iv })).filter(d => d.y != null);
      const spot = p.spot;
      TA_CHARTS.smile = new Chart(cvs.getContext('2d'), {
        type: 'line',
        data: { datasets: [
          { label: 'Calls', data: calls, borderColor: TA_BLUE, borderWidth: 1.5, pointRadius: 0, tension: 0.15 },
          { label: 'Puts', data: puts, borderColor: TA_PINK, borderWidth: 1.5, pointRadius: 0, tension: 0.15 },
        ] },
        options: {
          responsive: true, maintainAspectRatio: false, animation: false,
          scales: {
            x: { type: 'linear', title: { display: true, text: 'Strike', color: '#888', font: { size: 9 } },
                 grid: { color: 'rgba(255,255,255,.06)' }, ticks: { color: '#777', font: { size: 9 } } },
            y: { title: { display: true, text: 'Implied vol', color: '#888', font: { size: 9 } },
                 grid: { color: 'rgba(255,255,255,.06)' }, ticks: { color: '#777', font: { size: 9 }, callback: v => (v * 100).toFixed(0) + '%' } },
          },
          plugins: {
            legend: { display: true, position: 'top', labels: { color: '#c8c8c8', font: { size: 9 }, boxWidth: 10 } },
            tooltip: { callbacks: { label: it => `${it.dataset.label}: ${(it.parsed.y * 100).toFixed(1)}% @ ${it.parsed.x}` } },
          },
        },
        plugins: [{
          id: 'taSmileSpot',
          afterDatasetsDraw(chart) {
            if (spot == null) return;
            const { ctx, chartArea, scales } = chart;
            const x = scales.x.getPixelForValue(spot);
            if (x < chartArea.left || x > chartArea.right) return;
            ctx.save();
            ctx.strokeStyle = 'rgba(255,255,255,.5)'; ctx.setLineDash([4, 3]); ctx.lineWidth = 1;
            ctx.beginPath(); ctx.moveTo(x, chartArea.top); ctx.lineTo(x, chartArea.bottom); ctx.stroke();
            ctx.setLineDash([]); ctx.fillStyle = '#fff'; ctx.font = '9px sans-serif';
            ctx.fillText('spot ' + spot, x + 3, chartArea.top + 9);
            ctx.restore();
          },
        }],
      });
    },

    // IV term structure over time (P4e) -----------------------------------
    async loadChainIvTerm(force = false) {
      const date = this.chainDate;
      if (!this.ticker || !date) return;
      this.chainLoading = true; this.chainError = '';
      let url = `/api/ticker-analysis/chain/iv-term?ticker=${encodeURIComponent(this.ticker)}`
              + `&date=${date}&lookback=${this.chainIvTermLookback}`;
      if (force) url += '&force=1';
      try {
        const r = await fetch(url); const j = await r.json();
        if (j.error) { this.chainError = j.error; this.chainIvTerm = null; }
        else this.chainIvTerm = j;
      } catch (e) { console.error('[ticker-analysis] loadChainIvTerm failed:', e); this.chainError = 'load failed'; this.chainIvTerm = null; }
      finally { this.chainLoading = false; this.$nextTick(() => this._renderChainWhenReady('ta-chain-ivterm', () => this.renderChainIvTerm())); }
    },

    renderChainIvTerm() {
      const cvs = document.getElementById('ta-chain-ivterm');
      if (!cvs) return;
      if (TA_CHARTS.ivterm) { TA_CHARTS.ivterm.destroy(); TA_CHARTS.ivterm = null; }
      const p = this.chainIvTerm;
      if (!p || !p.dates || !p.dates.length) return;
      const tenorColors = { '7': '#1abc9c', '30': '#3498db', '90': '#9b59b6' };
      const datasets = (p.tenors || [7, 30, 90]).map(t => ({
        label: t + 'd', data: (p.series[String(t)] || []),
        borderColor: tenorColors[String(t)] || '#888', borderWidth: 1.2, pointRadius: 0, tension: 0, spanGaps: true,
      }));
      TA_CHARTS.ivterm = new Chart(cvs.getContext('2d'), {
        type: 'line',
        data: { labels: p.dates, datasets },
        options: {
          responsive: true, maintainAspectRatio: false, animation: false,
          scales: {
            x: { grid: { color: 'rgba(255,255,255,.05)' }, ticks: { color: '#777', font: { size: 9 }, autoSkip: true, maxTicksLimit: 10, maxRotation: 0 } },
            y: { title: { display: true, text: 'ATM IV', color: '#888', font: { size: 9 } },
                 grid: { color: 'rgba(255,255,255,.06)' }, ticks: { color: '#777', font: { size: 9 }, callback: v => (v * 100).toFixed(0) + '%' } },
          },
          plugins: {
            legend: { display: true, position: 'top', labels: { color: '#c8c8c8', font: { size: 9 }, boxWidth: 10 } },
            tooltip: { mode: 'index', intersect: false,
              callbacks: { label: it => `${it.dataset.label}: ${it.parsed.y == null ? '—' : (it.parsed.y * 100).toFixed(1) + '%'}` } },
          },
        },
      });
    },

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

    resetPriceZoom() {
      if (TA_CHARTS.price && TA_CHARTS.price.resetZoom) TA_CHARTS.price.resetZoom();
    },

    togglePriceFullscreen() {
      this.priceFullscreen = !this.priceFullscreen;
      // The container size changes; let the layout settle, then resize the
      // chart so it fills (or un-fills) the new box.
      this.$nextTick(() => {
        requestAnimationFrame(() => { if (TA_CHARTS.price) TA_CHARTS.price.resize(); });
      });
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
        const rawArr = new Array(labels.length).fill(null);
        for (const pt of d.series) {
          const i = idxByDate[pt.date];
          if (i != null) { arr[i] = (pt.val - mn) / rng; rawArr[i] = pt.val; }   // normalize for y; keep raw for tooltip
        }
        return {
          label: p.metric, data: arr, yAxisID: 'ovl',
          rawValues: rawArr, rawRange: [mn, mx],
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
                label: it => {
                  if (it.datasetIndex === 0) return 'close ' + it.parsed.y;
                  // Show the metric's TRUE raw value (line y-position is normalized only).
                  const raw = it.dataset.rawValues?.[it.dataIndex];
                  return it.dataset.label + ': ' + (raw == null ? '—' : (+raw).toPrecision(5));
                } } },
            // Click-drag a horizontal box to zoom into that date range.
            zoom: {
              zoom: {
                drag: { enabled: true, backgroundColor: 'rgba(52,152,219,.15)',
                        borderColor: '#3498db', borderWidth: 1 },
                mode: 'x',
              },
              pan: { enabled: false },
            },
          },
          scales,
        },
        plugins: [{
          id: 'taPriceOverlay',
          beforeDatasetsDraw(chart) {
            const { ctx, chartArea, scales } = chart;
            const x = scales.x;
            // Bin-highlight bands (confluence = opacity by overlap; union = flat)
            // Clip to the plot area so markers stay inside the axes when zoomed.
            const inPlot = px => px >= chartArea.left && px <= chartArea.right;
            const hl = chart.$taHL;
            if (hl && hl.entries.length) {
              ctx.save();
              for (const [i, cnt] of hl.entries) {
                const px = x.getPixelForValue(i);
                if (!inPlot(px)) continue;
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
                if (!inPlot(px)) continue;
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
