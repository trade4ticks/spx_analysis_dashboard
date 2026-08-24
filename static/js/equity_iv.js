/* ============================================================================
 * Equity IV Analysis — global (cross-sectional) half.
 *
 * Phase 1, item 1: the cross-sectional scatter, the universe distribution
 * histogram, and the scanner table. The ticker half (rows 3-9) loads from a
 * selection made here — clicking a dot or a scanner row already sets
 * selectedTicker, so wiring the lower half is additive.
 *
 * Everything metric-shaped is CATALOG-DRIVEN. equity_metrics_catalog has one
 * row per column (602 of them, 234 base + 368 z), carrying family / tenor /
 * wing / form / units / description. A hardcoded dropdown over that many
 * columns is not maintainable, so the pickers are built from the catalog at
 * load and the server validates every name it is handed against the same
 * table.
 *
 * The metric dropdowns list only BASE columns. A per-axis val/z toggle
 * resolves to `<base>_z_<window>` using the page's z-window control, so
 * flipping 63 <-> 252 re-resolves every axis, scanner column and filter at
 * once instead of leaving a stale mix of windows on screen.
 * ==========================================================================*/

'use strict';

const EQ_BLUE = '#3498db';   // theme --accent: positive / above / rich
const EQ_PINK = '#e84393';   // canonical theme pink: negative / below / fabricated
const EQ_GREY = '#8a8a8a';   // neutral midpoint of the diverging ramp
const EQ_SURF = '#2d2d2d';   // theme --surface, used as the ring between overlapping dots

/* Chart.js instances live OUTSIDE Alpine's reactive proxy. Wrapping Chart
 * internals in a Proxy breaks rendering — the same lesson the other pages on
 * this site already learned. */
const EQ_CHARTS = { scatter: null, hist: null };
/* Per-render point metadata the tooltip and the label plugin need, kept off
 * the reactive object for the same reason. */
const EQ_PTS = { scatter: [], hist: [] };

/* Preset axis pairs — the questions asked most often. Clicking through two
 * family dropdowns and two metric dropdowns every time is friction, so these
 * sit above the manual pickers. Columns verified against the catalog.
 *
 * The last one is permanently available: it answers, at a glance, whether the
 * tickers at the edge of the scatter are opportunities or artifacts. */
const EQ_PRESETS = [
  { label: 'skew z × 5d return',     why: 'is skew rich because spot fell?',
    x: { b: 'skew_30d_25p_atm', z: true },  y: { b: 'log_ret_1w', z: false } },
  { label: 'skew z × term ratio',    why: 'rich skew in contango vs backwardation',
    x: { b: 'skew_30d_25p_atm', z: true },  y: { b: 'term_ratio_30d_90d', z: false } },
  { label: 'skew z × VRP',           why: 'rich wings with a vol premium, or without',
    x: { b: 'skew_30d_25p_atm', z: true },  y: { b: 'vrp_1m', z: false } },
  { label: 'IV z × RV z',            why: 'implied stretched relative to what is realising',
    x: { b: 'iv_30d_atm', z: true },        y: { b: 'rv_1m', z: true } },
  { label: 'skew z × spot-vol β',    why: 'which rich-skew names double-hit on a down move',
    x: { b: 'skew_30d_25p_atm', z: true },  y: { b: 'spotvol_beta_1m', z: false } },
  { label: 'zc width z × skew z',    why: 'do vol-space and trade-native readings agree',
    x: { b: 'zc_width_sigma_30d', z: true }, y: { b: 'skew_30d_25p_atm', z: true } },
  { label: 'skew z × extrapolation', why: 'are the outliers real, or just thin chains?',
    permanent: true,
    x: { b: 'skew_30d_25p_atm', z: true },  y: { b: 'extrap_rate_short', z: false } },
];

const EQ_FILTER_OPS = [
  { op: 'gt',       label: '>',            hint: 'greater than' },
  { op: 'gte',      label: '≥',            hint: 'greater than or equal' },
  { op: 'lt',       label: '<',            hint: 'less than' },
  { op: 'lte',      label: '≤',            hint: 'less than or equal' },
  { op: 'eq',       label: '=',            hint: 'equals' },
  { op: 'ne',       label: '≠',            hint: 'not equal' },
  { op: 'absgt',    label: '|x| >',        hint: 'magnitude greater than — either tail' },
  { op: 'abslt',    label: '|x| <',        hint: 'magnitude less than — near the middle' },
  { op: 'nullorgt', label: 'null or >',    hint: 'unknown counts as passing' },
  { op: 'nullorlt', label: 'null or <',    hint: 'unknown counts as passing' },
  { op: 'notnull',  label: 'is present',   hint: 'metric is not null' },
  { op: 'isnull',   label: 'is absent',    hint: 'metric is null' },
];

/* Value formatting is driven by the catalog's `units`, not by guessing from
 * the column name. vol_decimal is a decimal (0.25 = 25%); ratio is a plain
 * number; z_score gets two decimals so +1.8 and +1.9 are distinguishable. */
const EQ_FMT = {
  vol_decimal:        v => (v * 100).toFixed(2) + '%',
  fraction:           v => (v * 100).toFixed(1) + '%',
  log_return:         v => (v * 100).toFixed(2) + '%',
  z_score:            v => v.toFixed(2),
  ratio:              v => v.toFixed(3),
  price:              v => v.toFixed(2),
  sigma:              v => v.toFixed(2) + 'σ',
  put_delta:          v => v.toFixed(1) + 'Δ',
  vol_per_log_strike: v => v.toFixed(3),
  vol_per_log_return: v => v.toFixed(2),
  count:              v => v.toFixed(0),
  days:               v => v.toFixed(0),
  dow:                v => v.toFixed(0),
};

/* Fetch that inspects STATUS, then CONTENT-TYPE, then parses — deliberately
 * in that order. Calling r.json() first reports every failure as a parse
 * error, which is how a proxy's HTML error page once spent an investigation
 * being treated as a data problem. See DEPLOYMENT_NOTES.md. */
async function eqGetJson(url) {
  const r = await fetch(url);
  const ct = r.headers.get('content-type') || '';
  if (!r.ok) {
    const body = (await r.text()).slice(0, 200);
    let detail = body;
    try { detail = JSON.parse(body).detail || body; } catch (e) { /* not JSON */ }
    throw new Error(`HTTP ${r.status} — ${detail}`);
  }
  if (!ct.includes('application/json')) {
    const body = (await r.text()).slice(0, 200);
    throw new Error(
      `Expected JSON, got ${ct || 'no content-type'}. An HTML body means ` +
      `something in front of the app produced it. First bytes: ${body}`);
  }
  return r.json();
}

/* Linear interpolation between two hex colours. */
function eqMix(a, b, t) {
  const p = h => [parseInt(h.slice(1, 3), 16), parseInt(h.slice(3, 5), 16), parseInt(h.slice(5, 7), 16)];
  const [r1, g1, b1] = p(a), [r2, g2, b2] = p(b);
  const m = (x, y) => Math.round(x + (y - x) * t);
  return `rgb(${m(r1, r2)},${m(g1, g2)},${m(b1, b2)})`;
}

/* Diverging ramp: pink -> neutral grey -> blue, with a NEUTRAL midpoint.
 * Never a hue at the middle — that reads as a third category. */
function eqDiverge(v, lo, hi, mid) {
  if (v == null || !isFinite(v)) return EQ_GREY;
  const half = Math.max(Math.abs(hi - mid), Math.abs(mid - lo)) || 1;
  const t = Math.max(-1, Math.min(1, (v - mid) / half));
  return t >= 0 ? eqMix(EQ_GREY, EQ_BLUE, t) : eqMix(EQ_GREY, EQ_PINK, -t);
}

/* Draws a reference line at zero on a z-scored axis. A z axis without a
 * visible origin makes "+1.8" a number rather than a position. */
const eqZeroLines = {
  id: 'eqZeroLines',
  beforeDatasetsDraw(chart, args, opts) {
    const { ctx, scales } = chart;
    ctx.save();
    ctx.strokeStyle = 'rgba(255,255,255,0.18)';
    ctx.lineWidth = 1;
    ctx.setLineDash([4, 4]);
    if (opts.x && scales.x.min < 0 && scales.x.max > 0) {
      const px = scales.x.getPixelForValue(0);
      ctx.beginPath(); ctx.moveTo(px, scales.y.top); ctx.lineTo(px, scales.y.bottom); ctx.stroke();
    }
    if (opts.y && scales.y.min < 0 && scales.y.max > 0) {
      const py = scales.y.getPixelForValue(0);
      ctx.beginPath(); ctx.moveTo(scales.x.left, py); ctx.lineTo(scales.x.right, py); ctx.stroke();
    }
    ctx.restore();
  },
};

/* Selective direct labels: the handful of dots furthest from the centre,
 * plus whatever is selected. Never a label on every point — at 121 tickers
 * that is a wall of text, not a chart. */
const eqDotLabels = {
  id: 'eqDotLabels',
  afterDatasetsDraw(chart, args, opts) {
    const meta = chart.getDatasetMeta(0);
    if (!meta || !meta.data || !meta.data.length) return;
    const pts = EQ_PTS.scatter;
    const { ctx } = chart;
    ctx.save();
    ctx.font = "600 10px 'Segoe UI', system-ui, sans-serif";
    ctx.textAlign = 'center';
    for (const i of opts.indices || []) {
      const el = meta.data[i];
      if (!el || !pts[i]) continue;
      const sel = pts[i].ticker === opts.selected;
      ctx.fillStyle = sel ? '#ecf0f1' : '#c8c8c8';
      ctx.fillText(pts[i].ticker, el.x, el.y - (el.options.radius || 5) - 4);
    }
    ctx.restore();
  },
};

/* Median marker for the distribution histogram. The median is the whole
 * point of the panel — if the median ticker sits at +0.4 sigma, a ticker at
 * +2.0 is riding a market-wide move rather than standing out on its own. */
const eqMedianLine = {
  id: 'eqMedianLine',
  afterDatasetsDraw(chart, args, opts) {
    if (opts.index == null) return;
    const { ctx, scales } = chart;
    const px = scales.x.getPixelForValue(opts.index);
    ctx.save();
    ctx.strokeStyle = EQ_PINK;
    ctx.lineWidth = 2;
    ctx.beginPath(); ctx.moveTo(px, scales.y.top); ctx.lineTo(px, scales.y.bottom); ctx.stroke();
    ctx.fillStyle = EQ_PINK;
    ctx.font = "700 10px 'Segoe UI', system-ui, sans-serif";
    ctx.textAlign = 'center';
    ctx.fillText('median ' + (opts.label || ''), px, scales.y.top - 3);
    ctx.restore();
  },
};

/* House Chart.js defaults, matching charts.js so this page reads as the same
 * system as the Dashboard. Guarded because Chart.js comes from a CDN: if it
 * fails to load, the page should degrade to "charts missing" rather than
 * throwing here and taking the whole Alpine component down with it. That
 * guard is also what lets scripts/check_alpine_refs.py construct this
 * component under its stub Chart, instead of skipping the page. */
if (typeof Chart !== 'undefined' && Chart.defaults) {
  Chart.defaults.color       = '#c8c8c8';
  Chart.defaults.borderColor = 'rgba(255,255,255,0.07)';
  Chart.defaults.font.family = "'Segoe UI', system-ui, sans-serif";
  Chart.defaults.font.size   = 11;
}

document.addEventListener('alpine:init', () => {
  Alpine.data('equityIv', () => ({

    // ── catalog ──────────────────────────────────────────────────────────
    metrics: [],          // every catalog row
    byCol: {},            // column_name -> row
    baseByFamily: {},     // family -> base-form rows
    zBases: {},           // base column -> true when a z variant exists
    families: [],
    catError: '',

    // ── page controls ────────────────────────────────────────────────────
    dates: [], snapshots: [],
    date: '', snapshot: '',
    zWindow: 63,
    histWindow: '1y',
    excludeExtrap: true,
    selectedTicker: null,
    slice: { date: '', snapshot: '', source: '' },

    // ── scatter ──────────────────────────────────────────────────────────
    presets: EQ_PRESETS,
    activePreset: EQ_PRESETS[0].label,
    xFam: 'skew', xBase: 'skew_30d_25p_atm', xZ: true,
    yFam: 'realized_vol', yBase: 'log_ret_1w', yZ: false,
    colorBase: '',
    cs: null, us: null,
    csLoading: false, csError: '',

    // ── scanner ──────────────────────────────────────────────────────────
    scanOpen: true,
    scanCols: [
      { b: 'skew_30d_25p_atm', z: true },
      { b: 'zc_width_sigma_30d', z: true },
      { b: 'iv_30d_atm', z: false },
      { b: 'term_ratio_30d_90d', z: false },
      { b: 'vrp_1m', z: false },
    ],
    scanFilters: [],
    scanSort: '', scanDir: 'desc',
    scan: null, scanLoading: false, scanError: '',
    pickFam: 'skew', pickBase: 'skew_30d_25p_atm', pickZ: true,
    filterDraft: false,
    filtFam: 'skew', filtBase: 'skew_30d_25p_atm', filtZ: true,
    filtOp: 'gt', filtVal: '1.5',
    filterOps: EQ_FILTER_OPS,

    // ── lifecycle ────────────────────────────────────────────────────────

    async init() {
      await this.loadCatalog();
      if (this.catError) return;
      await this.loadCalendar();
      await this.reloadAll();
    },

    async loadCatalog() {
      try {
        const j = await eqGetJson('/api/equity-iv/catalog');
        if (j.error) { this.catError = j.error; this.csError = j.error; return; }
        this.metrics = j.metrics;
        this.families = j.families;
        const by = {}, byFam = {}, zb = {};
        for (const m of j.metrics) {
          by[m.column_name] = m;
          if (m.form === 'base') {
            (byFam[m.family] = byFam[m.family] || []).push(m);
          } else if (m.form === 'z_63' && m.base_column) {
            zb[m.base_column] = true;
          }
        }
        this.byCol = by;
        this.baseByFamily = byFam;
        this.zBases = zb;
      } catch (e) {
        this.catError = String(e.message || e);
        this.csError = this.catError;
      }
    },

    async loadCalendar() {
      try {
        const j = await eqGetJson('/api/equity-iv/calendar');
        this.dates = j.dates || [];
        if (this.dates.length) {
          this.date = this.dates[0].date;
          this.snapshots = this.dates[0].snapshots;
          // Latest available snapshot, not the first — opening the page at
          // 11am should show 10:55, not the morning bucket.
          this.snapshot = this.snapshots[this.snapshots.length - 1];
        }
      } catch (e) {
        this.csError = String(e.message || e);
      }
    },

    onDateChange() {
      const hit = this.dates.find(d => d.date === this.date);
      this.snapshots = hit ? hit.snapshots : [];
      if (this.snapshots.length && !this.snapshots.includes(this.snapshot)) {
        this.snapshot = this.snapshots[this.snapshots.length - 1];
      }
      this.reloadAll();
    },

    async reloadAll() {
      await Promise.all([this.loadCrossSection(), this.loadScanner()]);
    },

    // ── catalog helpers ──────────────────────────────────────────────────

    metricsFor(family) { return this.baseByFamily[family] || []; },

    firstMetric(family) {
      const list = this.metricsFor(family);
      return list.length ? list[0].column_name : '';
    },

    hasZ(base) { return !!this.zBases[base]; },

    /** Base column + the val/z toggle -> the column actually queried. */
    resolve(base, useZ) {
      if (!base) return '';
      return (useZ && this.hasZ(base)) ? `${base}_z_${this.zWindow}` : base;
    },

    unitsOf(col) { const m = this.byCol[col]; return m ? m.units : ''; },
    describe(col) { const m = this.byCol[col]; return m ? m.description : col; },

    xCol() { return this.resolve(this.xBase, this.xZ); },
    yCol() { return this.resolve(this.yBase, this.yZ); },
    colorCol() {
      if (!this.colorBase) return '';
      // Colour answers "how unusual", so prefer the z form when one exists —
      // a diverging ramp needs a meaningful zero.
      return this.resolve(this.colorBase, this.hasZ(this.colorBase));
    },
    xUnits() { return this.unitsOf(this.xCol()); },
    colorUnits() { return this.unitsOf(this.colorCol()); },

    colorChoices() {
      const out = [];
      for (const f of this.families) {
        for (const m of this.metricsFor(f)) {
          if (m.units === 'text' || m.units === 'timestamp' || m.units === 'bool') continue;
          out.push(m);
        }
      }
      return out;
    },

    // ── control handlers ─────────────────────────────────────────────────

    setZWindow(w) {
      if (this.zWindow === w) return;
      this.zWindow = w;
      this.reloadAll();
    },

    setHistWindow(w) {
      if (this.histWindow === w) return;
      this.histWindow = w;
      this.loadUniverseStats();
    },

    toggleExtrap() {
      this.excludeExtrap = !this.excludeExtrap;
      this.reloadAll();
    },

    onFamilyChange(axis) {
      if (axis === 'x') {
        this.xBase = this.firstMetric(this.xFam);
        if (!this.hasZ(this.xBase)) this.xZ = false;
      } else {
        this.yBase = this.firstMetric(this.yFam);
        if (!this.hasZ(this.yBase)) this.yZ = false;
      }
      this.onAxisChange();
    },

    setAxisForm(axis, useZ) {
      const base = axis === 'x' ? this.xBase : this.yBase;
      if (useZ && !this.hasZ(base)) return;   // no z variant — the toggle is inert
      if (axis === 'x') this.xZ = useZ; else this.yZ = useZ;
      this.onAxisChange();
    },

    onAxisChange() {
      this.activePreset = '';
      this.loadCrossSection();
    },

    applyPreset(p) {
      const setAxis = (which, spec) => {
        const m = this.byCol[spec.b];
        if (!m) return;
        if (which === 'x') { this.xFam = m.family; this.xBase = spec.b; this.xZ = spec.z && this.hasZ(spec.b); }
        else               { this.yFam = m.family; this.yBase = spec.b; this.yZ = spec.z && this.hasZ(spec.b); }
      };
      setAxis('x', p.x);
      setAxis('y', p.y);
      this.activePreset = p.label;
      this.loadCrossSection();
    },

    /** Selecting re-renders the scatter to move the highlight ring. The
     *  re-render is deferred a tick because this is also called from the
     *  chart's own onClick, and destroying a Chart from inside its event
     *  dispatch is asking for trouble. */
    selectTicker(t) {
      this.selectedTicker = t;
      setTimeout(() => this.renderScatter(), 0);
    },

    // ── cross-section ────────────────────────────────────────────────────

    async loadCrossSection() {
      if (!this.date || !this.snapshot || this.catError) return;
      this.csLoading = true; this.csError = '';
      try {
        const q = new URLSearchParams({
          x: this.xCol(), y: this.yCol(),
          date: this.date, snapshot: this.snapshot,
          exclude_extrapolated: String(this.excludeExtrap),
        });
        const cc = this.colorCol();
        if (cc) q.set('color', cc);
        const j = await eqGetJson('/api/equity-iv/cross-section?' + q.toString());
        if (j.error) { this.csError = j.error; this.cs = null; }
        else {
          this.cs = j;
          this.slice = { date: j.date, snapshot: j.snapshot, source: this.dominantSource(j.points) };
          this.renderScatter();
          this.renderHistogram();
        }
      } catch (e) {
        this.csError = String(e.message || e);
        this.cs = null;
      } finally {
        this.csLoading = false;
      }
      this.loadUniverseStats();
    },

    /** `live` rows were captured at an arbitrary instant and rounded to the
     *  grid bucket; `exact` rows come from the anchored historical record.
     *  The header shows whichever describes the slice. */
    dominantSource(points) {
      if (!points || !points.length) return '';
      const live = points.filter(p => p.source === 'live').length;
      return live > points.length / 2 ? 'live' : 'exact';
    },

    async loadUniverseStats() {
      if (!this.date || !this.snapshot || this.catError) return;
      try {
        const q = new URLSearchParams({
          metric: this.xCol(), date: this.date, snapshot: this.snapshot,
          window: this.histWindow,
          exclude_extrapolated: String(this.excludeExtrap),
        });
        const j = await eqGetJson('/api/equity-iv/universe-stats?' + q.toString());
        this.us = j.error ? null : j;
      } catch (e) {
        this.us = null;
      }
    },

    csSubtitle() {
      const p = this.presets.find(p => p.label === this.activePreset);
      return p ? p.why : `${this.xCol()} × ${this.yCol()}`;
    },

    scatterCounts() {
      if (!this.cs) return { plotted: '—', nulls: '—', fabricated: '—' };
      const pts = this.cs.points;
      const plotted = pts.filter(p => p.x != null && p.y != null).length;
      const fab = pts.filter(p => p.x_extrap || p.y_extrap).length;
      return { plotted, nulls: pts.length - plotted, fabricated: fab };
    },

    colorRange() {
      if (!this.cs || !this.cs.points.length) return [0, 0];
      const vals = this.cs.points.map(p => p.color).filter(v => v != null);
      if (!vals.length) return [0, 0];
      return [Math.min(...vals), Math.max(...vals)];
    },

    // ── rendering ────────────────────────────────────────────────────────

    renderScatter() {
      const el = document.getElementById('eq-scatter');
      if (!el || !this.cs) return;
      if (EQ_CHARTS.scatter) { EQ_CHARTS.scatter.destroy(); EQ_CHARTS.scatter = null; }

      const pts = this.cs.points.filter(p => p.x != null && p.y != null);
      EQ_PTS.scatter = pts;
      if (!pts.length) return;

      const sizes = pts.map(p => p.size).filter(v => v != null);
      const sMin = sizes.length ? Math.min(...sizes) : 0;
      const sMax = sizes.length ? Math.max(...sizes) : 1;
      const radius = p => {
        if (p.size == null || sMax === sMin) return 6;
        return 4 + 10 * Math.sqrt((p.size - sMin) / (sMax - sMin));
      };

      const useColor = !!this.colorCol();
      const cvals = pts.map(p => p.color).filter(v => v != null);
      const cUnits = this.colorUnits();
      const cMid = cUnits === 'z_score' ? 0
        : (cvals.length ? cvals.slice().sort((a, b) => a - b)[Math.floor(cvals.length / 2)] : 0);
      const cLo = cvals.length ? Math.min(...cvals) : 0;
      const cHi = cvals.length ? Math.max(...cvals) : 1;

      // Extremes get a direct label; everything else is identified on hover.
      const xs = pts.map(p => p.x), ys = pts.map(p => p.y);
      const norm = (v, arr) => {
        const lo = Math.min(...arr), hi = Math.max(...arr);
        return hi === lo ? 0 : (v - lo) / (hi - lo) - 0.5;
      };
      const ranked = pts
        .map((p, i) => ({ i, d: Math.hypot(norm(p.x, xs), norm(p.y, ys)) }))
        .sort((a, b) => b.d - a.d)
        .slice(0, 8)
        .map(r => r.i);
      const selIdx = pts.findIndex(p => p.ticker === this.selectedTicker);
      if (selIdx >= 0 && !ranked.includes(selIdx)) ranked.push(selIdx);

      const self = this;
      EQ_CHARTS.scatter = new Chart(el, {
        type: 'scatter',
        data: {
          datasets: [{
            data: pts.map(p => ({ x: p.x, y: p.y })),
            pointRadius: pts.map(radius),
            pointHoverRadius: pts.map(p => radius(p) + 3),
            backgroundColor: pts.map(p =>
              useColor ? eqDiverge(p.color, cLo, cHi, cMid) : EQ_BLUE),
            // A 2px ring in the surface colour separates overlapping dots.
            // Pink instead when the value rests on a fabricated node, and
            // white when it is the selected ticker — never colour alone.
            borderColor: pts.map(p =>
              p.ticker === self.selectedTicker ? '#ecf0f1'
                : (p.x_extrap || p.y_extrap) ? EQ_PINK : EQ_SURF),
            borderWidth: pts.map(p => (p.ticker === self.selectedTicker ? 3 : 2)),
          }],
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          onClick(evt, els) {
            if (!els.length) return;
            const p = EQ_PTS.scatter[els[0].index];
            if (p) self.selectTicker(p.ticker);
          },
          scales: {
            x: {
              grid: { color: 'rgba(255,255,255,0.06)' },
              ticks: { font: { size: 10 }, callback: v => self.fmtShort(v, self.xUnits()) },
              title: { display: true, text: self.xCol(), color: '#777', font: { size: 9 } },
            },
            y: {
              grid: { color: 'rgba(255,255,255,0.06)' },
              ticks: { font: { size: 10 }, callback: v => self.fmtShort(v, self.unitsOf(self.yCol())) },
              title: { display: true, text: self.yCol(), color: '#777', font: { size: 9 } },
            },
          },
          plugins: {
            legend: { display: false },
            tooltip: {
              backgroundColor: '#3a3a3a', borderColor: '#555', borderWidth: 1,
              titleFont: { size: 11 }, bodyFont: { size: 11 },
              callbacks: {
                title: items => {
                  const p = EQ_PTS.scatter[items[0].dataIndex];
                  return p ? p.ticker : '';
                },
                label: item => {
                  const p = EQ_PTS.scatter[item.dataIndex];
                  if (!p) return '';
                  const rows = [
                    `${self.xCol()}: ${self.fmt(p.x, self.xUnits())}`,
                    `${self.yCol()}: ${self.fmt(p.y, self.unitsOf(self.yCol()))}`,
                    `spot: ${p.spot != null ? p.spot.toFixed(2) : '—'}`,
                    // Labelled "chain" so it is not read as a verdict on the
                    // two axis metrics — those get their own line below.
                    `chain extrap ≤30d: ${p.extrap_rate != null ? (p.extrap_rate * 100).toFixed(1) + '%' : '—'}`,
                  ];
                  if (useColor) rows.push(`${self.colorCol()}: ${self.fmt(p.color, cUnits)}`);
                  if (p.x_extrap) rows.push(`⚠ x rests on a fabricated node (${self.depends(self.xCol())})`);
                  if (p.y_extrap) rows.push(`⚠ y rests on a fabricated node (${self.depends(self.yCol())})`);
                  return rows;
                },
              },
            },
            eqZeroLines: {
              x: self.xUnits() === 'z_score',
              y: self.unitsOf(self.yCol()) === 'z_score',
            },
            eqDotLabels: { indices: ranked, selected: this.selectedTicker },
          },
        },
        plugins: [eqZeroLines, eqDotLabels],
      });
    },

    renderHistogram() {
      const el = document.getElementById('eq-hist');
      if (!el || !this.cs) return;
      if (EQ_CHARTS.hist) { EQ_CHARTS.hist.destroy(); EQ_CHARTS.hist = null; }

      const rows = this.cs.points.filter(p => p.x != null);
      if (!rows.length) return;
      const vals = rows.map(p => p.x);
      const lo = Math.min(...vals), hi = Math.max(...vals);
      const nBins = 20;
      const width = (hi - lo) / nBins || 1;

      const bins = Array.from({ length: nBins }, (_, i) => ({
        lo: lo + i * width, hi: lo + (i + 1) * width, n: 0, tickers: [],
      }));
      for (const p of rows) {
        let i = Math.floor((p.x - lo) / width);
        if (i >= nBins) i = nBins - 1;
        if (i < 0) i = 0;
        bins[i].n += 1;
        bins[i].tickers.push(p.ticker);
      }
      EQ_PTS.hist = bins;

      const sorted = vals.slice().sort((a, b) => a - b);
      const median = sorted.length % 2
        ? sorted[(sorted.length - 1) / 2]
        : (sorted[sorted.length / 2 - 1] + sorted[sorted.length / 2]) / 2;
      let medIdx = Math.floor((median - lo) / width);
      if (medIdx >= nBins) medIdx = nBins - 1;

      const units = this.xUnits();
      const self = this;
      EQ_CHARTS.hist = new Chart(el, {
        type: 'bar',
        data: {
          labels: bins.map(b => self.fmtShort((b.lo + b.hi) / 2, units)),
          datasets: [{
            data: bins.map(b => b.n),
            backgroundColor: EQ_BLUE,
            borderRadius: 3,
            // A 2px gap of surface between adjacent bars, so the bars read as
            // separate marks rather than one block.
            categoryPercentage: 0.96,
            barPercentage: 0.9,
          }],
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          scales: {
            x: {
              grid: { display: false },
              ticks: { font: { size: 9 }, maxRotation: 0, autoSkip: true, maxTicksLimit: 8 },
              title: { display: true, text: self.xCol(), color: '#777', font: { size: 9 } },
            },
            y: {
              grid: { color: 'rgba(255,255,255,0.06)' },
              ticks: { font: { size: 10 }, precision: 0 },
              title: { display: true, text: 'tickers', color: '#777', font: { size: 9 } },
            },
          },
          plugins: {
            legend: { display: false },
            tooltip: {
              backgroundColor: '#3a3a3a', borderColor: '#555', borderWidth: 1,
              titleFont: { size: 11 }, bodyFont: { size: 11 },
              callbacks: {
                title: items => {
                  const b = EQ_PTS.hist[items[0].dataIndex];
                  return b ? `${self.fmtShort(b.lo, units)} … ${self.fmtShort(b.hi, units)}` : '';
                },
                label: item => {
                  const b = EQ_PTS.hist[item.dataIndex];
                  if (!b) return '';
                  const shown = b.tickers.slice(0, 8).join(' ');
                  return [`${b.n} ticker${b.n === 1 ? '' : 's'}`,
                          shown + (b.tickers.length > 8 ? ` +${b.tickers.length - 8}` : '')];
                },
              },
            },
            eqMedianLine: { index: medIdx, label: self.fmtShort(median, units) },
          },
        },
        plugins: [eqMedianLine],
      });
    },

    // ── scanner ──────────────────────────────────────────────────────────

    colHeader(c) {
      return this.resolve(c.b, c.z).replace('_z_63', '·z63').replace('_z_252', '·z252');
    },

    addScanCol() {
      const col = { b: this.pickBase, z: this.pickZ && this.hasZ(this.pickBase) };
      if (this.scanCols.some(c => c.b === col.b && c.z === col.z)) return;
      this.scanCols.push(col);
      this.loadScanner();
    },

    removeScanCol(i) {
      const gone = this.resolve(this.scanCols[i].b, this.scanCols[i].z);
      this.scanCols.splice(i, 1);
      if (this.scanSort === gone) this.scanSort = '';
      this.loadScanner();
    },

    addScanFilter() {
      this.filterDraft = true;
      this.filtFam = this.pickFam;
      this.filtBase = this.pickBase;
      this.filtZ = this.pickZ && this.hasZ(this.pickBase);
    },

    commitFilter() {
      const f = {
        b: this.filtBase,
        z: this.filtZ && this.hasZ(this.filtBase),
        op: this.filtOp,
        v: (this.filtOp === 'isnull' || this.filtOp === 'notnull') ? '' : this.filtVal,
      };
      this.scanFilters.push(f);
      this.filterDraft = false;
      this.loadScanner();
    },

    removeScanFilter(i) {
      this.scanFilters.splice(i, 1);
      this.loadScanner();
    },

    sortBy(col) {
      if (col === '__ticker') { this.scanSort = ''; this.scanDir = 'asc'; }
      else if (this.scanSort === col) { this.scanDir = this.scanDir === 'desc' ? 'asc' : 'desc'; }
      else { this.scanSort = col; this.scanDir = 'desc'; }
      this.loadScanner();
    },

    async loadScanner() {
      if (!this.date || !this.snapshot || this.catError) return;
      if (!this.scanCols.length) { this.scan = { rows: [], n_rows: 0, truncated: false }; return; }
      this.scanLoading = true; this.scanError = '';
      try {
        const q = new URLSearchParams({
          columns: this.scanCols.map(c => this.resolve(c.b, c.z)).join(','),
          date: this.date, snapshot: this.snapshot,
          dir: this.scanDir,
          exclude_extrapolated: String(this.excludeExtrap),
        });
        if (this.scanSort) q.set('sort', this.scanSort);
        for (const f of this.scanFilters) {
          q.append('filter', `${this.resolve(f.b, f.z)}:${f.op}:${f.v}`);
        }
        const j = await eqGetJson('/api/equity-iv/scanner?' + q.toString());
        if (j.error) { this.scanError = j.error; this.scan = null; }
        else this.scan = j;
      } catch (e) {
        this.scanError = String(e.message || e);
        this.scan = null;
      } finally {
        this.scanLoading = false;
      }
    },

    /* ── Per-cell extrapolation marking ────────────────────────────────────
     * A cell is marked when the nodes THAT COLUMN depends on are
     * extrapolated — not when the ticker's chain-wide rate is high. The two
     * are routinely different: AAL can sit at 40% chain-wide because its
     * short-tenor 10-delta nodes do not reach, while skew_30d_25p_atm rests
     * only on 25p@30d and atm@30d, both of which are real. Scoring the cell
     * by the ticker rate makes a genuine signal look fabricated.
     *
     * Three distinct absences, deliberately distinguishable:
     *   dim  —    the metric is genuinely null (thin chain, tenor did not
     *             bracket). Absent, not zero.
     *   pink —*   present but excluded by the toggle, because it rests on a
     *             fabricated node.
     *   pink 1.23* shown and suspect: the toggle is off, so the value is
     *             visible but flagged.
     * Before this, the middle case rendered as a dim em dash identical to
     * the first, so "we threw this away" and "there was nothing here" were
     * the same pixel. */

    isFabricated(row, col) { return !!(row.extrap && row.extrap[col]); },

    cellClass(row, col) {
      if (this.isFabricated(row, col)) return 'eq-fab';
      return row.values[col] == null ? 'eq-null' : '';
    },

    cellText(row, col) {
      const v = row.values[col];
      if (v == null) return '—';
      return this.fmt(v, this.unitsOf(col));
    },

    cellTitle(row, col) {
      if (this.isFabricated(row, col)) {
        const dep = this.depends(col);
        return (row.values[col] == null
          ? 'Excluded: rests on a fabricated node'
          : 'Shown but suspect: rests on a fabricated node')
          + (dep ? ` (${dep})` : '')
          + '. The smile fit returned its boundary value there, so this is not an observation.';
      }
      if (row.values[col] == null) {
        return 'No value at this snapshot — the metric is null, not zero.';
      }
      return '';
    },

    /** The surface nodes a metric rests on, as "25p@30d, atm@30d".
     *  Answers "what would have to be fabricated for this to be wrong". */
    depends(col) {
      const m = this.byCol[col];
      if (!m || !m.extrap_flags || !m.extrap_flags.length) return '';
      return m.extrap_flags
        .map(f => { const p = f.replace('extrap_', '').split('_'); return `${p[0]}@${p[1]}`; })
        .join(', ');
    },

    /** Column-header tooltip: what the metric is, then what it rests on. */
    headerTitle(col) {
      const dep = this.depends(col);
      return this.describe(col) + (dep ? `\n\nDepends on: ${dep}` : '');
    },

    /* ── Chain-wide rate, demoted ──────────────────────────────────────────
     * extrap_rate_short is still worth having — a name at 40% is one whose
     * surface is thin in general — but it is context, not a verdict on any
     * row. It rides as a small dot beside the ticker with the number in the
     * tooltip, rather than as a column that competes with the values. */

    chainClass(row) {
      const r = row.extrap_rate;
      if (r == null || r <= 0) return '';
      if (r >= 0.25) return 'hi';
      return r >= 0.10 ? 'mid' : 'lo';
    },

    tickerTitle(row) {
      if (row.extrap_rate == null) return row.ticker;
      return `${row.ticker} — ${(row.extrap_rate * 100).toFixed(1)}% of nodes at `
           + `tenors ≤30d are extrapolated, chain-wide.\n\nThis says the chain is `
           + `thin, NOT that any metric in this row is affected. The per-cell `
           + `marks say that.`;
    },

    // ── formatting ───────────────────────────────────────────────────────

    /** Full-precision value for tooltips and table cells. A null metric is a
     *  legitimate absence — a thin chain lacks wing nodes — so it renders as
     *  an em dash, never as zero. */
    fmt(v, units) {
      if (v === null || v === undefined) return '—';
      if (typeof v === 'boolean') return v ? 'yes' : 'no';
      if (typeof v !== 'number') return String(v);
      const f = EQ_FMT[units];
      return f ? f(v) : (Math.abs(v) >= 1000 ? v.toFixed(0) : v.toFixed(3));
    },

    /** Compact form for axis ticks and ramp ends. */
    fmtShort(v, units) {
      if (v === null || v === undefined) return '—';
      if (typeof v !== 'number') return String(v);
      if (units === 'vol_decimal' || units === 'fraction' || units === 'log_return') {
        return (v * 100).toFixed(1) + '%';
      }
      if (units === 'z_score' || units === 'sigma') return v.toFixed(1);
      if (Math.abs(v) >= 100) return v.toFixed(0);
      return v.toFixed(2);
    },
  }));
});
