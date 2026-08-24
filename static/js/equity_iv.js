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
const EQ_CHARTS = { scatter: null, hist: null, ts0: null, ts1: null, ts2: null };
/* Per-render point metadata the tooltip and the label plugin need, kept off
 * the reactive object for the same reason. */
const EQ_PTS = { scatter: [], hist: [] };

/* Time-series pane count. The spec caps extra panes at 3–4; three is the
 * point past which each pane is too short to read a level off, given the
 * rails sit beside them and the whole row has to fit one screen. */
const EQ_TS_PANES = 3;

/* Series colours. Blue and pink first because they carry the page's meaning;
 * the next two are neutral so they never read as "positive"/"negative" for a
 * metric that has no such direction. */
const EQ_SERIES_COLORS = ['#3498db', '#e84393', '#f0a30a', '#9b8ec4'];

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

/* Candlesticks for a METRIC, not a price.
 *
 * Chart.js has no candle type and the financial plugin is another CDN
 * dependency on a page that already has two, so this draws them: the owning
 * dataset carries invisible points at each bar's high and low, which is what
 * makes the y-scale fit the wicks, and this plugin paints the body and wick
 * over them from chart.$candles.
 *
 * Colour is close-vs-open, in the page's own vocabulary — blue for a metric
 * that rose through the day, pink for one that fell. It is NOT the
 * up/down-green/red of a price chart, because a rising skew is not "good".
 *
 * A bar built from a single intraday bucket has o == h == l == c and would
 * paint as a one-pixel smear that reads as a rendering fault. Those are drawn
 * as a plain dash instead, and `n` in the tooltip says how many buckets the
 * bar was built from. */
const eqCandles = {
  id: 'eqCandles',
  afterDatasetsDraw(chart) {
    const bars = chart.$candles;
    if (!bars || !bars.length) return;
    const { ctx, scales } = chart;
    const xs = scales.x, ys = scales.y;
    if (!xs || !ys) return;
    // Body width from the actual spacing between the first two bars, so it
    // stays right when the window changes the point count.
    let w = 6;
    if (bars.length > 1) {
      w = Math.abs(xs.getPixelForValue(1) - xs.getPixelForValue(0)) * 0.6;
    }
    w = Math.max(1.5, Math.min(14, w));
    ctx.save();
    bars.forEach((b, i) => {
      const px = xs.getPixelForValue(i);
      const up = b.c >= b.o;
      const col = up ? EQ_BLUE : EQ_PINK;
      ctx.strokeStyle = col; ctx.fillStyle = col; ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(px, ys.getPixelForValue(b.h));
      ctx.lineTo(px, ys.getPixelForValue(b.l));
      ctx.stroke();
      const yo = ys.getPixelForValue(b.o), yc = ys.getPixelForValue(b.c);
      const top = Math.min(yo, yc), h = Math.abs(yc - yo);
      if (h < 1) {
        ctx.beginPath();
        ctx.moveTo(px - w / 2, top); ctx.lineTo(px + w / 2, top); ctx.stroke();
      } else {
        ctx.fillRect(px - w / 2, top, w, h);
      }
    });
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

    // ── ticker half (rows 3–5) ───────────────────────────────────────────
    // Everything below is keyed off selectedTicker and stays empty until a
    // ticker is picked, so the universe half of the page works on its own.
    hdr: null, hdrLoading: false, hdrError: '',

    unusual: null, unusualLoading: false, unusualError: '',
    unusualLimit: 40,

    /* The rail set is configurable; this is the default eight. They are the
     * ones that answer "is this name's surface shaped normally" — level,
     * both skew wings, term, the risk reversal, the zero-cost width, the
     * variance premium and the convexity. Any that the catalog does not have
     * are dropped at load rather than erroring. */
    railMetrics: ['skew_30d_25p_atm', 'skew_30d_10p_atm', 'iv_30d_atm',
                  'term_ratio_30d_90d', 'rr_30d_25d', 'zc_width_sigma_30d',
                  'vrp_1m', 'convexity_30d_25p_atm_25c'],
    rails: null, railsLoading: false, railsError: '',
    railFam: 'skew', railBase: 'skew_30d_25p_atm',

    /* Up to four series. `axis` puts one on the right-hand scale so two
     * metrics in different units can share a pane; `pane` splits them out
     * entirely when the units are too far apart for even a twin axis. */
    seriesSpecs: [{ b: 'skew_30d_25p_atm', axis: 'left', pane: 0 }],
    seriesMode: 'daily',       // daily | intraday
    seriesChart: 'line',       // line | candle
    seriesEnvelope: true,
    envLo: 0.10, envHi: 0.90, envWindow: 63,
    ser: null, serLoading: false, serError: '',
    serFam: 'skew', serBase: 'iv_30d_atm',
    // pane index -> series names candle mode could not draw there
    candleDropped: {},

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
      this.pruneTickerDefaults();
      await this.loadCalendar();
      await this.reloadAll();
    },

    /* The rail and series defaults are written as column names, and the
     * catalog is the whitelist those names have to survive: /rails and
     * /series reject an unknown column with a 400 rather than skipping it,
     * so one retired metric in the default list would take the whole panel
     * down instead of costing it one row. Pruned once, here, against what
     * the catalog actually returned. */
    pruneTickerDefaults() {
      this.railMetrics = this.railMetrics.filter(c => this.byCol[c]);
      this.seriesSpecs = this.seriesSpecs.filter(s => this.byCol[s.b]);
      if (!this.seriesSpecs.length) {
        const first = this.firstMetric(this.families[0] || '');
        if (first) this.seriesSpecs.push({ b: first, axis: 'left', pane: 0 });
      }
      if (!this.byCol[this.railBase]) this.railBase = this.firstMetric(this.railFam);
      if (!this.byCol[this.serBase]) this.serBase = this.firstMetric(this.serFam);
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
      await Promise.all([this.loadCrossSection(), this.loadScanner(), this.loadTicker()]);
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

    /* The history window is the distribution every percentile on the page is
     * taken over, so it drives the rails, the cards' percentiles and the
     * series span as well as the universe stats. Leaving the ticker half on
     * the old window would put a 3M percentile beside a 2Y one with nothing
     * on screen saying they disagree. */
    setHistWindow(w) {
      if (this.histWindow === w) return;
      this.histWindow = w;
      this.loadUniverseStats();
      this.loadTicker();
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
      this.loadTicker();
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

    // ══ Ticker half (rows 3–5) ═══════════════════════════════════════════
    //
    // Loaded as one unit whenever the ticker, date, snapshot, z window,
    // history window or extrapolation toggle changes. Fired in parallel:
    // the four requests are independent and serialising them would put the
    // slowest one in front of the header, which is the part that tells you
    // the click registered.

    async loadTicker() {
      if (!this.selectedTicker) {
        this.hdr = null; this.unusual = null; this.rails = null; this.ser = null;
        return;
      }
      await Promise.all([
        this.loadHeader(), this.loadUnusual(), this.loadRails(), this.loadSeries(),
      ]);
    },

    // ── Row 3: ticker header ─────────────────────────────────────────────

    async loadHeader() {
      if (!this.selectedTicker || !this.date || !this.snapshot) return;
      this.hdrLoading = true; this.hdrError = '';
      try {
        const j = await eqGetJson('/api/equity-iv/ticker-header?' + new URLSearchParams({
          ticker: this.selectedTicker, date: this.date, snapshot: this.snapshot,
        }));
        if (j.error) { this.hdrError = j.error; this.hdr = null; }
        else { this.hdr = j; }
      } catch (e) {
        this.hdrError = String(e.message || e); this.hdr = null;
      } finally {
        this.hdrLoading = false;
      }
    },

    /* Whether a "vs" metric is above or below its reference depends on
     * whether the loader stored a difference or a ratio, and the header
     * resolves its columns at runtime rather than hardcoding one. So the
     * pivot comes from the catalog's units, and when the catalog does not
     * describe the column, the chip shows the number with NO tone rather
     * than guessing a direction and colouring it confidently wrong. */
    pivotFor(col) {
      const m = this.byCol[col];
      if (!m || !m.units) return null;
      if (m.units === 'ratio') return 1;
      if (['fraction', 'log_return', 'vol_decimal', 'sigma', 'z_score'].includes(m.units)) return 0;
      return null;
    },

    toneVs(v, pivot) {
      if (v == null || pivot == null) return '';
      return v > pivot ? 'up' : (v < pivot ? 'down' : '');
    },

    /** State chips for row 3. Each is {k, v, tone, title}. A null value is
     *  rendered as an em dash with tone '' — absent, never zero. */
    hdrChips() {
      const h = this.hdr;
      if (!h) return [];
      const R = h.resolved || {};
      const out = [];

      out.push({
        k: 'Term', v: h.term_state || '—',
        tone: h.term_state === 'contango' ? 'up' : (h.term_state === 'backwardation' ? 'down' : ''),
        title: R.term_ratio
          ? `${R.term_ratio} = ${this.fmt(h.term_ratio, this.unitsOf(R.term_ratio))}. `
            + 'The ratio is near/far, so below 1 is contango — the far tenor is richer.'
          : 'No term-structure ratio column exists on equity_metrics.',
      });

      const pv = this.pivotFor(R.px_vs_50dma);
      out.push({
        k: '50dma', v: this.fmt(h.px_vs_50dma, this.unitsOf(R.px_vs_50dma)),
        tone: this.toneVs(h.px_vs_50dma, pv),
        title: R.px_vs_50dma
          ? `${R.px_vs_50dma}${pv == null ? ' — units unknown to the catalog, so no above/below colour is asserted.' : ''}`
          : 'No 50-day-average column exists on equity_metrics.',
      });

      out.push({
        k: 'Earnings',
        v: h.days_to_earnings == null ? '—' : this.fmt(h.days_to_earnings, 'days'),
        tone: '',
        title: h.days_to_earnings == null
          ? 'days_to_earnings is NULL for every row — no earnings calendar is wired up yet. '
            + 'It renders as absent, not as zero.'
          : 'Trading days to the next earnings date.',
      });

      out.push({
        k: 'Spot-vol β', v: this.fmt(h.spotvol_beta, this.unitsOf(R.spotvol_beta)),
        tone: this.toneVs(h.spotvol_beta, 0),
        title: (R.spotvol_beta || 'spot-vol beta')
          + (h.spotvol_r2 == null ? '' : ` — R² ${this.fmt(h.spotvol_r2, 'ratio')}`)
          + '. A low R² means the beta is fitted through noise; read it with that in mind.',
      });

      out.push({
        k: 'Extrapolated',
        v: h.extrap_rate == null ? '—' : (h.extrap_rate * 100).toFixed(0) + '%',
        tone: h.extrap_rate == null ? '' : (h.extrap_rate >= 0.25 ? 'down' : ''),
        title: 'Share of nodes at tenors ≤30d the smile fit fabricated, chain-wide. '
             + 'This describes the surface, NOT any single metric — the per-metric '
             + 'marks in the rails and cards do that.',
      });

      out.push({
        k: 'Strikes', v: this.fmt(h.liquidity, 'count'), tone: '',
        title: 'median_n_strikes_clean — surviving strikes per fitted expiry. '
             + 'A liquidity proxy; this database has no volume or ADV column.',
      });

      out.push({
        k: 'Captured', v: this.captureText(), tone: '',
        title: h.source === 'live'
          ? 'Captured at an arbitrary instant and rounded to the grid bucket. '
            + 'The bucket label is not the capture time.'
          : 'From the anchored historical record, taken at the bucket itself.',
      });

      return out;
    },

    /* The bucket says 15:45; a live capture may have happened at 15:47:31.
     * Showing the bucket alone would be quietly wrong about what instant the
     * header describes, so the real time is what the chip shows. */
    captureText() {
      const h = this.hdr;
      if (!h || !h.captured_at) return h && h.snapshot ? h.snapshot : '—';
      const t = String(h.captured_at).replace('T', ' ');
      return t.slice(11, 19) || t;
    },

    // ── Row 4: today, what's unusual ─────────────────────────────────────

    async loadUnusual() {
      if (!this.selectedTicker || !this.date || !this.snapshot) return;
      this.unusualLoading = true; this.unusualError = '';
      try {
        const j = await eqGetJson('/api/equity-iv/unusual?' + new URLSearchParams({
          ticker: this.selectedTicker, date: this.date, snapshot: this.snapshot,
          z_window: String(this.zWindow), window: this.histWindow,
          limit: String(this.unusualLimit),
          exclude_extrapolated: String(this.excludeExtrap),
        }));
        if (j.error) { this.unusualError = j.error; this.unusual = null; }
        else { this.unusual = j; }
      } catch (e) {
        this.unusualError = String(e.message || e); this.unusual = null;
      } finally {
        this.unusualLoading = false;
      }
    },

    unusualCards() { return this.unusual ? this.unusual.cards : []; },

    /* Cards are coloured by DIRECTION, never by rank — blue for rich/above,
     * pink for cheap/below — with the tint tracking |z| so the strongest
     * reading is the most saturated. Ranking is already expressed by
     * position in the strip; colouring by it too would say the same thing
     * twice and leave nothing to say which way the metric moved. */
    cardStyle(c) {
      const t = Math.min(1, Math.abs(c.z) / 3);
      const col = c.z >= 0 ? '52,152,219' : '232,67,147';
      return `background:rgba(${col},${(0.06 + t * 0.16).toFixed(3)});`
           + `border-color:rgba(${col},${(0.35 + t * 0.5).toFixed(3)});`;
    },

    cardZClass(c) { return c.z >= 0 ? 'up' : 'down'; },

    pctText(p) {
      if (p == null) return '—';
      return (p * 100).toFixed(0) + 'th';
    },

    cardTitle(c) {
      const dep = this.depends(c.column);
      return (c.description || c.column)
        + `\n\nz ${c.z.toFixed(2)} over ${this.zWindow}d`
        + `\npercentile ${this.pctText(c.percentile)} over ${this.histWindow.toUpperCase()}`
        + (dep ? `\ndepends on: ${dep}` : '')
        + '\n\nClick to chart it.';
    },

    // ── Row 5a: rails ────────────────────────────────────────────────────

    async loadRails() {
      if (!this.selectedTicker || !this.date || !this.snapshot) return;
      if (!this.railMetrics.length) { this.rails = null; return; }
      this.railsLoading = true; this.railsError = '';
      try {
        const j = await eqGetJson('/api/equity-iv/rails?' + new URLSearchParams({
          ticker: this.selectedTicker, metrics: this.railMetrics.join(','),
          date: this.date, snapshot: this.snapshot,
          window: this.histWindow, z_window: String(this.zWindow),
          exclude_extrapolated: String(this.excludeExtrap),
        }));
        if (j.error) { this.railsError = j.error; this.rails = null; }
        else { this.rails = j; }
      } catch (e) {
        this.railsError = String(e.message || e); this.rails = null;
      } finally {
        this.railsLoading = false;
      }
    },

    railRows() { return this.rails ? this.rails.rails : []; },

    /* Rail geometry, as percentages of the track.
     *
     * The domain is p5..p95 EXPANDED to contain today's value when today is
     * outside it, rather than clamping the marker to the end of the track.
     * Clamping would draw a value at the 99th percentile and one at the
     * 300th in exactly the same place, which is the one thing this panel
     * exists to distinguish. A little padding keeps the marker off the edge.
     *
     * Returns null when the distribution is degenerate (no spread, or too
     * few observations), so the row can say so instead of dividing by zero.
     */
    railGeom(r) {
      if (r.p5 == null || r.p95 == null || !r.n) return null;
      let lo = r.p5, hi = r.p95;
      if (r.value != null) { lo = Math.min(lo, r.value); hi = Math.max(hi, r.value); }
      const span = hi - lo;
      if (!(span > 0)) return null;
      const pad = span * 0.06;
      lo -= pad; hi += pad;
      const pos = v => v == null ? null : ((v - lo) / (hi - lo)) * 100;
      const g = {
        outerL: pos(r.p5), outerR: pos(r.p95),
        innerL: pos(r.p25), innerR: pos(r.p75),
        median: pos(r.p50), value: pos(r.value),
        beyond: r.value != null && (r.value < r.p5 || r.value > r.p95),
      };
      g.outerW = g.outerR - g.outerL;
      g.innerW = (g.innerL == null || g.innerR == null) ? null : g.innerR - g.innerL;
      return g;
    },

    /* Style helpers so the template binds one expression per element rather
     * than recomputing the geometry inline four times. */
    railOuterStyle(r) {
      const g = this.railGeom(r);
      return g ? `left:${g.outerL}%;width:${g.outerW}%` : 'display:none';
    },
    railInnerStyle(r) {
      const g = this.railGeom(r);
      return (g && g.innerW != null) ? `left:${g.innerL}%;width:${g.innerW}%` : 'display:none';
    },
    railMedStyle(r) {
      const g = this.railGeom(r);
      return (g && g.median != null) ? `left:${g.median}%` : 'display:none';
    },
    railMarkStyle(r) {
      const g = this.railGeom(r);
      if (!g || g.value == null) return 'display:none';
      // Outside P5–P95 the marker keeps the page's directional colours;
      // inside, it is neutral so "normal" does not read as a signal.
      const col = g.beyond ? (r.value > r.p95 ? EQ_BLUE : EQ_PINK) : '#e8e8e8';
      return `left:${g.value}%;background:${col};box-shadow:0 0 0 1px rgba(0,0,0,.55)`;
    },

    railTitle(r) {
      const u = r.units, f = v => this.fmt(v, u);
      const dep = this.depends(r.column_name);
      return `${r.column_name}\n${r.description || ''}`
        + `\n\nP5 ${f(r.p5)}  P25 ${f(r.p25)}  median ${f(r.p50)}  P75 ${f(r.p75)}  P95 ${f(r.p95)}`
        + `\ntoday ${r.value == null ? '—' : f(r.value)}`
        + `   percentile ${this.pctText(r.percentile)}`
        + (r.z == null ? '' : `   z ${r.z.toFixed(2)}`)
        + `\nn=${r.n} over ${this.histWindow.toUpperCase()}`
        + (dep ? `\ndepends on: ${dep}` : '')
        + '\n\nPercentiles, not standard deviations: these distributions are '
        + 'right-skewed and fat-tailed, so a symmetric band would be too wide '
        + 'on one side and too narrow on the other.';
    },

    /** Why a rail has no marker — excluded vs never there. Different facts. */
    railMissing(r) {
      if (r.value != null) return '';
      if (r.raw_value != null) return 'excluded';   // existed, fabricated node
      return 'absent';
    },

    addRail() {
      if (!this.railBase || this.railMetrics.includes(this.railBase)) return;
      this.railMetrics.push(this.railBase);
      this.loadRails();
    },

    removeRail(i) {
      this.railMetrics.splice(i, 1);
      if (this.railMetrics.length) this.loadRails(); else this.rails = null;
    },

    // ── Row 5b: time series ──────────────────────────────────────────────

    async loadSeries() {
      if (!this.selectedTicker || !this.seriesSpecs.length) { this.ser = null; return; }
      this.serLoading = true; this.serError = '';
      try {
        const q = new URLSearchParams({
          ticker: this.selectedTicker,
          metrics: this.seriesSpecs.map(s => s.b).join(','),
          mode: this.effectiveSeriesMode(),
          window: this.histWindow,
          z_window: String(this.zWindow),
          envelope: String(this.seriesEnvelope),
          env_window: String(this.envWindow),
          env_lo: String(this.envLo), env_hi: String(this.envHi),
          exclude_extrapolated: String(this.excludeExtrap),
        });
        // Daily mode pins the close bucket server-side; intraday and candle
        // read every bucket, so sending one would filter them to nothing.
        const j = await eqGetJson('/api/equity-iv/series?' + q);
        if (j.error) { this.serError = j.error; this.ser = null; }
        else { this.ser = j; this.renderSeries(); }
      } catch (e) {
        this.serError = String(e.message || e); this.ser = null;
      } finally {
        this.serLoading = false;
      }
    },

    /* Candles need more than one bucket per day to have an open and a close,
     * so the chart toggle only means anything on the intraday grid. Asking
     * for candles on the daily view resolves to candles over intraday
     * buckets rather than silently drawing a line — the button said candle. */
    effectiveSeriesMode() {
      return this.seriesChart === 'candle' ? 'candle' : this.seriesMode;
    },

    setSeriesMode(m) {
      if (this.seriesMode === m) return;
      this.seriesMode = m;
      this.loadSeries();
    },

    setSeriesChart(c) {
      if (this.seriesChart === c) return;
      this.seriesChart = c;
      this.loadSeries();
    },

    toggleEnvelope() {
      this.seriesEnvelope = !this.seriesEnvelope;
      this.loadSeries();
    },

    /** Add a metric as a series. Called from the picker and from a card. */
    addSeries(base, opts) {
      if (!base || !this.byCol[base]) return;
      if (this.seriesSpecs.some(s => s.b === base)) return;
      if (this.seriesSpecs.length >= 4) return;
      const o = opts || {};
      this.seriesSpecs.push({
        b: base,
        axis: o.axis || (this.seriesSpecs.length ? 'right' : 'left'),
        pane: o.pane == null ? 0 : o.pane,
      });
      this.loadSeries();
    },

    removeSeries(i) {
      this.seriesSpecs.splice(i, 1);
      this.loadSeries();
    },

    cycleAxis(i) {
      const s = this.seriesSpecs[i];
      s.axis = s.axis === 'left' ? 'right' : 'left';
      this.renderSeries();
    },

    cyclePane(i) {
      const s = this.seriesSpecs[i];
      s.pane = (s.pane + 1) % EQ_TS_PANES;
      this.renderSeries();
    },

    seriesColor(i) { return EQ_SERIES_COLORS[i % EQ_SERIES_COLORS.length]; },

    /** Panes that actually hold a series, so empty canvases stay hidden. */
    activePanes() {
      const used = new Set(this.seriesSpecs.map(s => s.pane));
      return [...Array(EQ_TS_PANES).keys()].filter(p => used.has(p));
    },

    seriesFull() { return this.seriesSpecs.length >= 4; },

    /* Intraday coverage begins 2026-08-24 and is sparse before 11:25 that
     * day. A chart that suddenly shows six points is a real state, not a
     * failure, and saying so is cheaper than the user re-checking their
     * filters. */
    seriesCoverageNote() {
      if (!this.ser || this.ser.mode === 'daily') return '';
      const n = Math.min(...this.ser.series.map(s => s.n_points));
      if (!isFinite(n)) return '';
      const what = this.ser.mode === 'candle' ? 'bars' : 'points';
      return `${n} ${what} — intraday capture starts 2026-08-24 and is sparse `
           + `before 11:25 that day, so this view is short by construction.`;
    },

    /** Where the z on screen came from. The whole point is that it does not
     *  change when the intraday toggle does. */
    baselineNote() {
      if (!this.ser || !this.ser.series.length) return '';
      const b = this.ser.series[0].baseline;
      if (!b || b.mu == null) return '';
      return `z from the ${b.snapshot} daily close over ${b.z_window} days `
           + `(n=${b.n}) in every mode — the intraday point moves, the yardstick does not.`;
    },

    renderSeries() {
      if (typeof Chart === 'undefined' || !this.ser) return;
      for (let p = 0; p < EQ_TS_PANES; p++) {
        const key = 'ts' + p;
        if (EQ_CHARTS[key]) { EQ_CHARTS[key].destroy(); EQ_CHARTS[key] = null; }
      }
      this.candleDropped = {};
      // Destroy first, then paint after Alpine has added or removed the
      // canvases that activePanes() changed — painting into an element the
      // x-for is about to replace leaves a chart bound to a detached node.
      this.$nextTick(() => this.paintSeries());
    },

    /** Series candle mode could not draw in a pane, for the note under it. */
    candleDroppedText() {
      const all = Object.values(this.candleDropped || {}).flat();
      if (!all.length) return '';
      return `Not drawn as candles: ${all.join(', ')} — one candle series per `
           + `pane. Move them to another pane, or switch back to line.`;
    },

    paintSeries() {
      if (typeof Chart === 'undefined' || !this.ser) return;
      const candle = this.ser.mode === 'candle';

      for (const pane of this.activePanes()) {
        const el = document.getElementById('eq-ts-' + pane);
        if (!el) continue;

        let mine = this.seriesSpecs
          .map((s, i) => ({ s, i, d: this.ser.series.find(x => x.column_name === s.b) }))
          .filter(o => o.d && o.s.pane === pane);
        if (!mine.length) continue;

        /* One candle series per pane. Two sets of bodies overlaid at the same
         * x positions is unreadable, and carrying the extra series as
         * invisible high/low points would stretch the y scale for something
         * that never gets drawn — a chart silently rescaled by data you
         * cannot see. The dropped ones are named under the chart; move them
         * to another pane to see them. */
        if (candle && mine.length > 1) {
          this.candleDropped[pane] = mine.slice(1).map(o => o.d.column_name);
          mine = mine.slice(0, 1);
        } else {
          this.candleDropped[pane] = [];
        }

        // Every series in a pane shares one x axis, so the labels come from
        // the longest one and shorter series align to its tail.
        const longest = mine.reduce((a, o) => o.d.points.length > a.points.length ? o.d : a, mine[0].d);
        const labels = longest.points.map(p => this.pointLabel(p));

        const datasets = [];
        const meta = [];
        let usesRight = false;

        for (const o of mine) {
          const colr = this.seriesColor(o.i);
          const yid = o.s.axis === 'right' ? 'yR' : 'y';
          if (yid === 'yR') usesRight = true;
          const pts = o.d.points;
          const pad = labels.length - pts.length;      // shorter series, right-aligned
          const align = arr => new Array(Math.max(0, pad)).fill(null).concat(arr);

          if (candle) {
            // Invisible highs and lows so the scale fits the wicks; the
            // plugin paints the bodies.
            datasets.push({
              label: o.d.column_name, yAxisID: yid,
              data: align(pts.map(p => p.h)),
              borderColor: 'transparent', backgroundColor: 'transparent',
              pointRadius: 0, showLine: false,
            });
            datasets.push({
              label: o.d.column_name + ' low', yAxisID: yid,
              data: align(pts.map(p => p.l)),
              borderColor: 'transparent', backgroundColor: 'transparent',
              pointRadius: 0, showLine: false,
            });
          } else {
            if (this.seriesEnvelope && pts.some(p => p.env_lo != null)) {
              // Band drawn first so the line sits on top of it. `fill:'-1'`
              // ties the upper edge down to the lower one.
              datasets.push({
                label: o.d.column_name + ' P' + Math.round(this.envLo * 100),
                yAxisID: yid, data: align(pts.map(p => p.env_lo)),
                borderColor: 'transparent', pointRadius: 0, fill: false, order: 3,
              });
              datasets.push({
                label: o.d.column_name + ' P' + Math.round(this.envHi * 100),
                yAxisID: yid, data: align(pts.map(p => p.env_hi)),
                borderColor: 'transparent', pointRadius: 0,
                backgroundColor: this.rgba(colr, 0.10), fill: '-1', order: 3,
              });
            }
            datasets.push({
              label: o.d.column_name, yAxisID: yid,
              data: align(pts.map(p => p.v)),
              borderColor: colr, backgroundColor: colr,
              borderWidth: 1.5, pointHoverRadius: 3,
              tension: 0, spanGaps: false, order: 1,
              // A fabricated point is drawn, but marked — the line would
              // otherwise present spline output as observation. Every other
              // point has radius 0, so this is the dataset's only pointRadius.
              pointBackgroundColor: align(pts.map(p => p.extrap ? EQ_PINK : colr)),
              pointRadius: align(pts.map(p => p.extrap ? 2.5 : 0)),
            });
          }
          meta.push({ col: o.d.column_name, units: o.d.units, pts, pad,
                      color: colr, axis: yid });
        }

        // Each axis is labelled in the units of the first series on it. A
        // twin axis exists precisely because the two series are in different
        // units, so formatting both scales the same way defeats it.
        const leftUnits  = (meta.find(m => m.axis === 'y')  || meta[0]).units;
        const rightUnits = (meta.find(m => m.axis === 'yR') || {}).units;

        const self = this;
        const cfg = {
          type: 'line',
          data: { labels, datasets },
          options: {
            responsive: true, maintainAspectRatio: false,
            animation: false,
            interaction: { mode: 'index', intersect: false },
            scales: {
              x: {
                ticks: { maxTicksLimit: 8, autoSkip: true, maxRotation: 0 },
                grid: { display: false },
              },
              y: {
                position: 'left',
                ticks: { callback: v => self.fmtShort(v, leftUnits) },
                grid: { color: 'rgba(255,255,255,0.05)' },
              },
              ...(usesRight ? {
                yR: {
                  position: 'right',
                  ticks: { callback: v => self.fmtShort(v, rightUnits) },
                  grid: { drawOnChartArea: false },
                },
              } : {}),
            },
            plugins: {
              legend: { display: false },
              tooltip: {
                callbacks: {
                  title: items => items.length ? String(items[0].label) : '',
                  label: item => self.seriesTooltip(item, meta, candle),
                },
              },
            },
          },
          plugins: candle ? [eqCandles] : [],
        };

        const chart = new Chart(el.getContext('2d'), cfg);
        if (candle) chart.$candles = meta[0] ? meta[0].pts : [];
        EQ_CHARTS['ts' + pane] = chart;
      }
    },

    pointLabel(p) {
      return p.snapshot ? `${p.t} ${p.snapshot}` : p.t;
    },

    seriesTooltip(item, meta, candle) {
      // Two hidden datasets per candle series; one line row is enough.
      const mi = candle ? Math.floor(item.datasetIndex / 2) : null;
      const m = candle ? meta[mi] : meta.find(x => x.col === item.dataset.label);
      if (candle) {
        if (item.datasetIndex % 2) return null;
        const p = m && m.pts[item.dataIndex - m.pad];
        if (!p) return null;
        return `${m.col}  O ${this.fmt(p.o, m.units)}  H ${this.fmt(p.h, m.units)}  `
             + `L ${this.fmt(p.l, m.units)}  C ${this.fmt(p.c, m.units)}`
             + `  (${p.n} buckets)`
             + (p.z == null ? '' : `  z ${p.z.toFixed(2)}`);
      }
      if (!m) return null;                       // an envelope edge — not a row
      const p = m.pts[item.dataIndex - m.pad];
      if (!p || p.v == null) return null;
      return `${m.col}  ${this.fmt(p.v, m.units)}`
           + (p.z == null ? '' : `  z ${p.z.toFixed(2)}`)
           + (p.extrap ? '  · rests on a fabricated node' : '');
    },

    rgba(hex, a) {
      const n = parseInt(hex.slice(1), 16);
      return `rgba(${(n >> 16) & 255},${(n >> 8) & 255},${n & 255},${a})`;
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
