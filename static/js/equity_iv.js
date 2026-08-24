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
const EQ_CHARTS = {
  scatter: null, hist: null, ts0: null, ts1: null, ts2: null,
  // Rows 6-9. The two curve-band kinds share one canvas but are keyed
  // separately so switching the toggle destroys the outgoing chart
  // rather than orphaning it on an element the new one is about to take.
  'cb-skew': null, 'cb-term': null, 'cb-skew_term': null,
  tent: null, sticky: null, tscat: null, svol: null, oi: null,
};
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
      if (!b) return;                 // an empty slot on the shared x axis
      const px = xs.getPixelForValue(i);
      const up = b.c >= b.o;
      const col = up ? EQ_BLUE : EQ_PINK;
      ctx.strokeStyle = col; ctx.fillStyle = col; ctx.lineWidth = 1;
      // A partial bar is a session still being written — its "close" is
      // whatever the latest bucket happens to be. Drawn hollow with a dashed
      // wick so it cannot be read as a finished day.
      ctx.setLineDash(b.partial ? [3, 2] : []);
      ctx.beginPath();
      ctx.moveTo(px, ys.getPixelForValue(b.h));
      ctx.lineTo(px, ys.getPixelForValue(b.l));
      ctx.stroke();
      ctx.setLineDash([]);
      const yo = ys.getPixelForValue(b.o), yc = ys.getPixelForValue(b.c);
      const top = Math.min(yo, yc), h = Math.abs(yc - yo);
      if (h < 1) {
        ctx.beginPath();
        ctx.moveTo(px - w / 2, top); ctx.lineTo(px + w / 2, top); ctx.stroke();
      } else if (b.partial) {
        ctx.strokeRect(px - w / 2 + 0.5, top + 0.5, w - 1, h - 1);
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

/* Recency ramp for the two scatters.
 *
 * A 141-to-159-dot cloud on a dark-to-blue age gradient makes the recent
 * sessions -- the only ones anybody is reading a trajectory from -- the
 * dimmest thing on the panel, because "recent" and "bright" were fighting a
 * background that is already dark. So the ramp is not a gradient over the
 * whole cloud at all:
 *
 *   older than the last N   uniform dim blue. They are context, and context
 *                           does not need to be individually distinguishable.
 *   the last N              faint-to-bright PINK, so the recent path reads as
 *                           a run rather than as more cloud.
 *   today                   drawn separately, brightest, by the caller.
 *
 * Pink for the recent run because blue is already the cloud; a second blue
 * ramp on top of a blue cloud is the problem this replaces.
 */
const EQ_RECENT_N = 10;

/* The older cloud keeps an age gradient — flattening it to one dim blue threw
 * away the ordering that makes a path readable at all. It runs dark-slate to
 * blue, staying well under the pink so the recent run is still where the eye
 * lands.
 *
 * The recent run is PINK and starts at 0.45 alpha, not 0.30: at 0.30 the
 * oldest few of the ten were indistinguishable from the blue behind them, so
 * ten dots read as about five. */
function eqRecentColor(i, total) {
  const back = total - 1 - i;                       // 0 = most recent
  if (back >= EQ_RECENT_N) {
    const older = Math.max(1, total - EQ_RECENT_N - 1);
    const t = Math.min(1, i / older);               // 0 oldest .. 1 newest
    const rgb = eqMix('#24405a', EQ_BLUE, t);       // "rgb(r,g,b)"
    return rgb.replace('rgb(', 'rgba(')
              .replace(')', `,${(0.35 + t * 0.3).toFixed(2)})`);
  }
  const t = 1 - (back / EQ_RECENT_N);               // 0 faintest .. 1 brightest
  return `rgba(232,67,147,${(0.45 + t * 0.5).toFixed(3)})`;
}

function eqRecentRadius(i, total) {
  const back = total - 1 - i;
  if (back >= EQ_RECENT_N) return 2;
  return 2.5 + 1.5 * (1 - back / EQ_RECENT_N);
}

/* The surface grid's view list, shared by the picker and the label lookup so
 * a new view cannot appear in one and not the other. */
const EQ_GRID_VIEWS = [
  // `fmt` is carried per view, not decided by a chain of comparisons with a
  // fallback. z_iv was added to this list without a matching case in the
  // formatter and fell through to the vol-points default, so every z came
  // out multiplied by 100 — a z of -1.24 rendered as -124. A view that
  // cannot be added without also declaring its units cannot repeat that.
  //
  //   'z'    a z-score. Plain number, one decimal.
  //   'pt'   a vol quantity in decimal form, rendered as vol points.
  //   'pt0'  the same, to whole points.
  { k: 'iv_minus_atm',   label: 'Skew', fmt: 'pt',
    hint: 'Wing IV minus ATM IV, per node. The shape view. Anchored on ' +
          'equity_atm at k=0, not put_delta 50.' },
  { k: 'z_iv_minus_atm', label: 'z of Skew', fmt: 'z',
    hint: 'That spread scored against prior sessions at the daily close.' },
  { k: 'iv',             label: 'raw IV', fmt: 'pt0',
    hint: 'The surface as fitted.' },
  { k: 'z_iv',           label: 'z of raw IV', fmt: 'z',
    hint: 'Every cell moves with the vol level, so this view tends to light ' +
          'up or go dark all at once — which answers "is the whole surface ' +
          'rich" rather than "what shape is it". It is NOT a restatement of ' +
          'raw IV: a node can carry high IV and a low z, because it is ' +
          'usually even higher.' },
  { k: 'chg_1d',         label: '1d change', fmt: 'pt',
    hint: 'Against the prior close, per node.' },
  { k: 'chg_5d',         label: '5d change', fmt: 'pt',
    hint: 'Against five closes back, per node.' },
];

/* Markers for the tent: the historical band of the zero-cost width, plus
 * today's short and long strikes and the delta-neutral point.
 *
 * Drawn as a plugin rather than as datasets because they are vertical
 * annotations on a σ axis, not series — a dataset would put them in the
 * legend, the tooltip and the y-scale calculation, none of which is wanted. */
const eqTentMarks = {
  id: 'eqTentMarks',
  beforeDatasetsDraw(chart, args, opts) {
    if (!opts || !opts.xs) return;
    const { ctx, scales } = chart;
    const xs = scales.x, ys = scales.y;
    if (!xs || !ys) return;
    const px = sig => {
      // The x scale is categorical over `opts.xs`, so a σ value has to be
      // located by interpolating its position in that array.
      const arr = opts.xs;
      if (!arr.length) return null;
      if (sig <= arr[0]) return xs.getPixelForValue(0);
      if (sig >= arr[arr.length - 1]) return xs.getPixelForValue(arr.length - 1);
      let i = 0;
      while (i < arr.length - 1 && arr[i + 1] < sig) i++;
      const span = arr[i + 1] - arr[i];
      const t = span === 0 ? 0 : (sig - arr[i]) / span;
      return xs.getPixelForValue(i) + (xs.getPixelForValue(i + 1) - xs.getPixelForValue(i)) * t;
    };
    ctx.save();

    // The band, as a shaded region on the σ axis: this is where the zero-cost
    // short USUALLY sits, so today's marker inside or outside it is the read.
    const b = opts.band;
    if (b && b.p25 != null && b.p75 != null) {
      const a = px(-Math.abs(b.p75)), c = px(-Math.abs(b.p25));
      if (a != null && c != null) {
        ctx.fillStyle = 'rgba(255,255,255,0.07)';
        ctx.fillRect(Math.min(a, c), ys.top, Math.abs(c - a), ys.bottom - ys.top);
      }
    }
    if (b && b.p5 != null && b.p95 != null) {
      const a = px(-Math.abs(b.p95)), c = px(-Math.abs(b.p5));
      if (a != null && c != null) {
        ctx.strokeStyle = 'rgba(255,255,255,0.16)';
        ctx.setLineDash([2, 3]); ctx.lineWidth = 1;
        [a, c].forEach(x => {
          ctx.beginPath(); ctx.moveTo(x, ys.top); ctx.lineTo(x, ys.bottom); ctx.stroke();
        });
        ctx.setLineDash([]);
      }
    }

    const mark = (sig, col, label) => {
      if (sig == null) return;
      const x = px(sig);
      if (x == null) return;
      ctx.strokeStyle = col; ctx.lineWidth = 1.5;
      ctx.beginPath(); ctx.moveTo(x, ys.top); ctx.lineTo(x, ys.bottom); ctx.stroke();
      ctx.fillStyle = col;
      ctx.font = "700 9px 'Segoe UI', system-ui, sans-serif";
      ctx.textAlign = 'center';
      ctx.fillText(label, x, ys.top - 2);
    };
    mark(opts.shortSigma, EQ_PINK, 'short ×2');
    mark(opts.longSigma, EQ_BLUE, 'long');
    mark(opts.dnSigma, '#f0a30a', 'Δ-neutral');
    ctx.restore();
  },
};

/* Spot markers for the sticky-strike panel. Today's spot and the prior
 * session's, on a strike axis — the distance between them IS the migration
 * the panel is decomposing, so it has to be visible. */
const eqSpotMarks = {
  id: 'eqSpotMarks',
  beforeDatasetsDraw(chart, args, opts) {
    if (!opts) return;
    const { ctx, scales } = chart;
    const xs = scales.x, ys = scales.y;
    if (!xs || !ys) return;
    ctx.save();
    const mark = (k, col, dash, label) => {
      if (k == null) return;
      // A linear scale locates a strike directly. The category version this
      // replaced had to interpolate a position out of the label array, which
      // was both fiddly and wrong whenever the two sessions' strikes differed.
      const x = xs.getPixelForValue(k);
      if (!isFinite(x)) return;
      ctx.strokeStyle = col; ctx.lineWidth = 1; ctx.setLineDash(dash);
      ctx.beginPath(); ctx.moveTo(x, ys.top); ctx.lineTo(x, ys.bottom); ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = col;
      ctx.font = "600 9px 'Segoe UI', system-ui, sans-serif";
      ctx.textAlign = 'center';
      ctx.fillText(label, x, ys.top - 2);
    };
    mark(opts.prevSpot, 'rgba(138,138,138,0.9)', [4, 3], 'prev spot');
    mark(opts.spot, '#e8e8e8', [], 'spot');
    ctx.restore();
  },
};

/* Reference levels on the OI panels: spot, and the OI-weighted average call
 * and put strike.
 *
 * This replaced an overlay of the TRADE's strikes, which drew across two
 * datasets that do not agree on a basis — equity_surface's strikes against a
 * split-adjusted chain ladder — and so could put a mark on a real-looking
 * rung that was not the right one. These three come out of the chain payload
 * itself, so there is nothing to reconcile.
 *
 * Thin, and labelled with their value: the number is what is being compared
 * against the ladder, and making the reader find it in a footer defeats the
 * point of drawing it.
 *
 * Two orientations: the profile and ΔOI charts put strike on a CATEGORY x
 * axis, the flow map on a LINEAR y axis. The caller says which. */
const eqRefLines = {
  id: 'eqRefLines',
  afterDatasetsDraw(chart, args, opts) {
    if (!opts || !opts.marks || !opts.marks.length) return;
    const { ctx, scales } = chart;
    const xs = scales.x, ys = scales.y;
    if (!xs || !ys) return;
    ctx.save();
    ctx.font = "600 9px 'Segoe UI', system-ui, sans-serif";
    ctx.lineWidth = 1;
    opts.marks.forEach((m, i) => {
      const text = `${m.label} ${m.v.toFixed(2)}`;
      ctx.strokeStyle = m.color; ctx.fillStyle = m.color;
      ctx.setLineDash([4, 3]);
      if (opts.horizontal) {
        /* A SHORT segment at the right edge, not a full-width rule.
         *
         * On the flow map the x axis is time, and these three are today's
         * values. Drawn across the whole width they assert a level that held
         * all year, which is a claim about history that was never made and
         * is usually false — the weighted-average strike migrates as the book
         * rolls. At the right edge they read as what they are: where things
         * stand now, against a chart of how they got there.
         *
         * The per-session paths drawn as datasets are the historical version
         * of the same quantity, and they can be compared against these
         * end-markers directly. */
        const y = ys.getPixelForValue(m.v);
        if (!isFinite(y) || y < ys.top || y > ys.bottom) return;
        const seg = Math.min(46, (xs.right - xs.left) * 0.12);
        ctx.beginPath();
        ctx.moveTo(xs.right - seg, y); ctx.lineTo(xs.right, y); ctx.stroke();
        ctx.setLineDash([]);
        ctx.textAlign = 'right';
        ctx.fillText(text, xs.right - 2, y - 4);
      } else {
        // A category axis over the listed ladder: a reference level almost
        // never lands on a rung, so its pixel is interpolated between the two
        // it falls between rather than snapped to the nearer one.
        const arr = (opts.strikes || []).map(Number);
        if (arr.length < 2) return;
        let hi = arr.findIndex(k => k >= m.v);
        if (hi < 0) hi = arr.length - 1;
        const lo = Math.max(0, hi - 1);
        const span = arr[hi] - arr[lo];
        const t = span === 0 ? 0 : (m.v - arr[lo]) / span;
        const x = xs.getPixelForValue(lo)
                + (xs.getPixelForValue(hi) - xs.getPixelForValue(lo)) * t;
        if (!isFinite(x)) return;
        ctx.beginPath(); ctx.moveTo(x, ys.top); ctx.lineTo(x, ys.bottom); ctx.stroke();
        ctx.setLineDash([]);
        // Staggered, so three levels close together do not overprint.
        ctx.textAlign = 'center';
        ctx.fillText(text, x, ys.top - 3 - (i % 2) * 10);
      }
    });
    ctx.restore();
  },
};

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

    /* Empty until the server answers, which is deliberate: the default set
     * used to be written out here AND in the router, and the two drifted —
     * names that the catalog did not have were dropped silently on the way
     * through, so the panel quietly rendered a shorter set than either list
     * described. The first /rails call now omits `metrics` entirely and
     * adopts whatever came back, with the slots that resolved to nothing
     * named on screen. */
    railMetrics: [],
    railDefaults: null,
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

    // ── rows 6–9 state ──────────────────────────────────────────────────
    // Keyed off selectedTicker like the rest of the ticker half, so the
    // universe half still works with nothing selected.

    // Row 6. `cb` holds one payload per curve kind; the skew panel and the
    // term panel render into separate canvases, and the term panel's kind is
    // a toggle between two questions rather than two series.
    cb: {}, cbLoading: {}, cbError: {},
    cbDte: 30,                 // the tenor every per-tenor panel in 6–8 uses
    cbDteChosen: false,        // true once resolved against the real tenor list
    /* The tenor list as STATE, not derived from whichever payload happens to
     * have arrived. The select's options are rendered from this by x-for, and
     * a <select> whose value is bound before its options exist falls back to
     * its first option — which is how the control read 0d while every panel
     * drew 30d. Holding the list in state means the options and the value are
     * set from the same place, and the template re-asserts the DOM value once
     * the options have rendered. */
    dteOptions: [],
    cbWing: 25,
    termKind: 'term',          // term | skew_term
    termDeltas: [25, 75],

    tent: null, tentLoading: false, tentError: '',
    tentLongDelta: 25,

    // Row 7
    sticky: null, stickyLoading: false, stickyError: '',
    grid: null, gridLoading: false, gridError: '',
    gridView: 'iv_minus_atm',

    // Row 8. The Path panel has its OWN axis pickers rather than following
    // the Cross-section ones. Sharing them meant scrolling up to change this
    // panel, and left the global scatter showing whatever the ticker panel
    // wanted rather than what was set for cross-sectional work — two panels
    // fighting over one pair of controls.
    tscat: null, tscatLoading: false, tscatError: '',
    tsFamX: 'skew', tsBaseX: 'skew_30d_25p_atm', tsZX: true,
    tsFamY: 'realized_vol', tsBaseY: 'log_ret_1w', tsZY: false,
    svol: null, svolLoading: false, svolError: '',

    // Row 9 — collapsed by default. This is a periodic-curiosity section,
    // not a check-every-time one, and it hits a different (slower, parquet)
    // data path, so it does not load until it is opened.
    oiOpen: false, oi: null, oiLoading: false, oiError: '', oiNote: '',
    oiTab: 'profile',          // profile | doi | flow
    oiDate: '', oiDates: [], oiDoiN: 5, oiLookback: 252,
    oiSide: 'all',             // all | call | put
    oiRef: null,               // {spot, call, put} reference levels
    oiFlowCall: null, oiFlowPut: null,   // per-session weighted strike paths
    oiFlowZoom: 1,             // multiplier on the measured strike half-range
    /* DTE bands in the chain endpoints' "lo-hi" form, MULTI-select: the
     * endpoints take a CSV of bands, so 0-7 + 8-14 + 15-30 composes into
     * 0-30 without needing a band for every span anyone might want.
     * Empty = every expiry.
     *
     * Deliberately NOT the 1m/3m/6m span buttons: those move the lookback,
     * which is a different axis from which expiries are counted. */
    oiDteSel: [],
    // Finer than the first cut. 0–30 lumped the weekly, the monthly and the
    // next-month expiry into one bucket, which is most of what anyone trading
    // this style is trying to separate.
    oiDteBands: [
      { v: '',         label: 'all' },
      { v: '0-7',      label: '0–7' },
      { v: '8-14',     label: '8–14' },
      { v: '15-30',    label: '15–30' },
      { v: '31-60',    label: '31–60' },
      { v: '61-180',   label: '61–180' },
      { v: '181-3650', label: '180+' },
    ],

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

    /* The ticker header sticks beneath the control bar, and the bar's height
     * depends on how it wraps — which depends on the viewport. Measured rather
     * than guessed, so the two never overlap at an awkward width. */
    /* Called after any change that resizes a panel — full-screen in or out,
     * a note popover opening over a chart. Chart.js instances are responsive
     * and listen for window resize, so one synthetic event re-measures every
     * chart on the page without this having to know which ones exist.
     *
     * nextTick lets Alpine apply the class, rAF lets the browser lay out
     * against it; firing before either leaves the chart sized to the old box.
     */
    relayout() {
      this.$nextTick(() => requestAnimationFrame(() => {
        window.dispatchEvent(new Event('resize'));
      }));
    },

    syncCtrlHeight() {
      const el = document.getElementById('eq-ctrl');
      if (!el) return;
      const h = Math.round(el.getBoundingClientRect().height);
      if (h > 0) document.documentElement.style.setProperty('--eq-ctrl-h', h + 'px');
    },

    async init() {
      this.syncCtrlHeight();
      window.addEventListener('resize', () => this.syncCtrlHeight());
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
      // railMetrics starts empty and is filled from the server's resolution,
      // so there is nothing to prune on the first pass.
      this.railMetrics = this.railMetrics.filter(c => this.byCol[c]);
      this.seriesSpecs = this.seriesSpecs.filter(s => this.byCol[s.b]);
      if (!this.seriesSpecs.length) {
        const first = this.firstMetric(this.families[0] || '');
        if (first) this.seriesSpecs.push({ b: first, axis: 'left', pane: 0 });
      }
      if (!this.byCol[this.railBase]) this.railBase = this.firstMetric(this.railFam);
      if (!this.byCol[this.serBase]) this.serBase = this.firstMetric(this.serFam);
      if (!this.byCol[this.tsBaseX]) this.tsBaseX = this.firstMetric(this.tsFamX);
      if (!this.byCol[this.tsBaseY]) this.tsBaseY = this.firstMetric(this.tsFamY);
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
        this.loadSurface(), this.loadOi(),
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
      this.railsLoading = true; this.railsError = '';
      try {
        const q = new URLSearchParams({
          ticker: this.selectedTicker,
          date: this.date, snapshot: this.snapshot,
          window: this.histWindow, z_window: String(this.zWindow),
          exclude_extrapolated: String(this.excludeExtrap),
        });
        // Omitted on the first call so the server's slot resolution decides
        // the set; sent thereafter so an edited set survives a reload.
        if (this.railMetrics.length) q.set('metrics', this.railMetrics.join(','));
        const j = await eqGetJson('/api/equity-iv/rails?' + q);
        if (j.error) { this.railsError = j.error; this.rails = null; }
        else {
          this.rails = j;
          if (j.defaults) {
            this.railDefaults = j.defaults;
            this.railMetrics = j.rails.map(r => r.column_name);
          }
        }
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

    /** Default slots the catalog had no column for. Named, not dropped. */
    railMissingNote() {
      if (!this.railDefaults) return '';
      const miss = this.railDefaults.filter(s => !s.column).map(s => s.slot);
      if (!miss.length) return '';
      return `No catalog column for: ${miss.join(', ')}. Those rails are absent `
           + `rather than substituted — add one from the picker if you know the `
           + `real column name.`;
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
        // The chart is anchored to the same (date, snapshot) as every other
        // panel. `snapshot` is NOT sent — that parameter picks which bucket's
        // closes the daily line plots, and it stays pinned to 1545.
        // `live_snapshot` is the page's selection, which is what puts today
        // on the line and what makes it advance as the session does.
        if (this.date) q.set('date', this.date);
        if (this.snapshot) q.set('live_snapshot', this.snapshot);
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

    /** The live point's provenance, when there is one on the chart. */
    livePointNote() {
      const lp = this.ser && this.ser.live_point;
      if (!lp) return '';
      const nPartial = (this.ser.series || [])
        .reduce((a, s) => Math.max(a, s.n_partial || 0), 0);
      if (!nPartial) return '';
      const what = lp.appended ? 'The final point is' : 'Points from';
      return `${what} today (${lp.date}) at ${lp.snapshot} — an unfinished `
           + `session sampled at the selected bucket, not a close. Drawn hollow `
           + `on a dashed segment, and kept out of both the envelope and the `
           + `baseline that scores it. It advances as you move the snapshot.`;
    },

    /** Where the z on screen came from. The whole point is that it does not
     *  change when the intraday toggle does. */
    baselineNote() {
      if (!this.ser || !this.ser.series.length) return '';
      const b = this.ser.series[0].baseline;
      if (!b || b.mu == null) return '';
      return `z from the ${b.snapshot} daily close over ${b.z_window} sessions `
           + `(n=${b.n}${b.last ? `, through ${b.last}` : ''}) in every mode — `
           + `the point moves, the yardstick does not.`;
    },

    /* ── One scoring rule, said once ────────────────────────────────────────
     * Every z and percentile on this page — scatter, scanner, cards, rails,
     * charts — is measured against the daily close series ending at the
     * PRIOR session. The line below is what makes that checkable rather than
     * a claim in a docstring: if `last` is ever the date on screen, the
     * exclusion broke, and it is right there to see. */

    /** The daily-baseline provenance line, from whichever payload has it. */
    zBasisNote() {
      const b = (this.rails && this.rails.baseline)
             || (this.unusual && this.unusual.baseline);
      if (!b) return '';
      const through = b.last ? `through ${b.last}` : 'through the prior session';
      return `Scored against ${b.snapshot} daily closes, ${b.z_window} sessions, `
           + `${through}${b.sessions ? ` (${b.sessions} sessions)` : ''}. `
           + `Today is never inside the window scoring it.`;
    },

    /** True when the selected snapshot is not the daily close, which is
     *  exactly the case the whole baseline rule exists for. */
    get onIntradaySnapshot() {
      const b = (this.rails && this.rails.baseline)
             || (this.unusual && this.unusual.baseline);
      return !!(b && this.snapshot && this.snapshot !== b.snapshot);
    },

    /** Metrics that had a value today but no score, because the baseline was
     *  thinner than the floor. Reported rather than shown as z = 0. */
    thinBaselineNote() {
      const u = this.unusual;
      if (!u || !u.n_unscored_thin_baseline) return '';
      const min = u.baseline ? u.baseline.min_n : '';
      return `${u.n_unscored_thin_baseline} metric(s) had a value today but `
           + `fewer than ${min} daily observations to score it against, so they `
           + `carry no z and are not ranked.`;
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

        /* Every series in a pane shares one x axis, built as the union of
         * every series' labels rather than taken from the longest one.
         *
         * The old shape right-aligned each series against the longest, which
         * is only correct while they all end on the same observation. The
         * live point broke that: a metric with a value at 12:15 gains a final
         * point, one that is null there does not, and right-alignment would
         * then slide that series' last SETTLED close into the live slot —
         * drawing yesterday's close as though it were today's reading.
         *
         * Labels are "YYYY-MM-DD" or "YYYY-MM-DD HHMM", both of which sort
         * lexicographically into chronological order, so a plain sort is the
         * axis.
         *
         * One caveat, recorded because it is not obvious: a bare date sorts
         * BEFORE that same date's buckets, so "2026-08-24" (a 15:45 close)
         * would land left of "2026-08-24 0945". That never arises, and the
         * reason is worth knowing before anyone changes it — the server
         * appends a live point only when NO row exists for the anchor date at
         * the daily bucket, and row existence is a property of the row, not
         * of any metric's value in it. So every series in one request agrees
         * about whether today is settled, and a bare date and a bucket for
         * the same day cannot both be on the axis. */
        const labelSet = new Set();
        for (const o of mine) {
          for (const p of o.d.points) labelSet.add(this.pointLabel(p));
        }
        const labels = [...labelSet].sort();
        const slot = new Map(labels.map((l, i) => [l, i]));

        const datasets = [];
        const meta = [];
        let usesRight = false;

        for (const o of mine) {
          const colr = this.seriesColor(o.i);
          const yid = o.s.axis === 'right' ? 'yR' : 'y';
          if (yid === 'yR') usesRight = true;
          const pts = o.d.points;

          // Each point placed at its OWN label's slot; gaps stay null.
          const byIndex = new Array(labels.length).fill(null);
          for (const p of pts) {
            const i = slot.get(this.pointLabel(p));
            if (i != null) byIndex[i] = p;
          }
          const align = get => byIndex.map(p => p == null ? null : get(p));
          // Chart.js wants a number for every point-style entry, so absent
          // slots take the inert value rather than null.
          const alignNum = (get, absent) =>
            byIndex.map(p => p == null ? absent : get(p));

          if (candle) {
            // Invisible highs and lows so the scale fits the wicks; the
            // plugin paints the bodies.
            datasets.push({
              label: o.d.column_name, yAxisID: yid,
              data: align(p => p.h),
              borderColor: 'transparent', backgroundColor: 'transparent',
              pointRadius: 0, showLine: false,
            });
            datasets.push({
              label: o.d.column_name + ' low', yAxisID: yid,
              data: align(p => p.l),
              borderColor: 'transparent', backgroundColor: 'transparent',
              pointRadius: 0, showLine: false,
            });
          } else {
            if (this.seriesEnvelope && pts.some(p => p.env_lo != null)) {
              // Band drawn first so the line sits on top of it. `fill:'-1'`
              // ties the upper edge down to the lower one.
              datasets.push({
                label: o.d.column_name + ' P' + Math.round(this.envLo * 100),
                yAxisID: yid, data: align(p => p.env_lo),
                borderColor: 'transparent', pointRadius: 0, fill: false, order: 3,
              });
              datasets.push({
                label: o.d.column_name + ' P' + Math.round(this.envHi * 100),
                yAxisID: yid, data: align(p => p.env_hi),
                borderColor: 'transparent', pointRadius: 0,
                backgroundColor: this.rgba(colr, 0.10), fill: '-1', order: 3,
              });
            }
            datasets.push({
              label: o.d.column_name, yAxisID: yid,
              data: align(p => p.v),
              borderColor: colr, backgroundColor: colr,
              borderWidth: 1.5, pointHoverRadius: 3,
              tension: 0, spanGaps: false, order: 1,
              // Two kinds of point are marked, and they mean different things:
              //   pink filled — rests on a node the spline fabricated
              //   hollow ring — a PARTIAL reading: an unfinished session
              //                 sampled at the selected bucket, not a close
              // Every other point has radius 0, so these are the dataset's
              // only pointRadius / pointBackgroundColor entries.
              pointBackgroundColor: alignNum(
                p => p.partial ? EQ_SURF : (p.extrap ? EQ_PINK : colr), colr),
              pointBorderColor: alignNum(p => p.extrap ? EQ_PINK : colr, colr),
              pointBorderWidth: alignNum(p => p.partial ? 1.5 : 0, 0),
              pointRadius: alignNum(
                p => p.partial ? 3.5 : (p.extrap ? 2.5 : 0), 0),
              // Dash any segment entering a partial point. A solid line from
              // a close to a 12:15 sample would draw the two as the same kind
              // of observation, which is exactly the claim being avoided.
              segment: {
                borderDash: ctx => {
                  const p = byIndex[ctx.p1DataIndex];
                  return (p && p.partial) ? [4, 3] : undefined;
                },
              },
            });
          }
          meta.push({ col: o.d.column_name, units: o.d.units, byIndex,
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
            // Dead space at the right edge so the line — and especially the
            // live point's marker, which is the rightmost thing on the chart
            // — is not clipped against the frame.
            layout: { padding: { right: 10 } },
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
        // Slot-aligned, so the plugin's index matches the x scale's. Empty
        // slots are null and simply not drawn.
        if (candle) chart.$candles = meta[0] ? meta[0].byIndex : [];
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
        const p = m && m.byIndex[item.dataIndex];
        if (!p) return null;
        return `${m.col}  O ${this.fmt(p.o, m.units)}  H ${this.fmt(p.h, m.units)}  `
             + `L ${this.fmt(p.l, m.units)}  C ${this.fmt(p.c, m.units)}`
             + `  (${p.n} buckets)`
             + (p.z == null ? '' : `  z ${p.z.toFixed(2)}`)
             + this.partialSuffix(p);
      }
      if (!m) return null;                       // an envelope edge — not a row
      const p = m.byIndex[item.dataIndex];
      if (!p || p.v == null) return null;
      return `${m.col}  ${this.fmt(p.v, m.units)}`
           + (p.z == null ? '' : `  z ${p.z.toFixed(2)}`)
           + (p.extrap ? '  · rests on a fabricated node' : '')
           + this.partialSuffix(p);
    },

    /** Says so in words, not only in line style — the dash is easy to miss
     *  on a dense chart, and the difference between a close and a mid-session
     *  sample is the kind of thing that gets read wrong once and acted on. */
    partialSuffix(p) {
      if (!p || !p.partial) return '';
      const at = p.snapshot || p.last_bucket || '';
      return `  · LIVE ${at} — session in progress, not a close`;
    },

    rgba(hex, a) {
      const n = parseInt(hex.slice(1), 16);
      return `rgba(${(n >> 16) & 255},${(n >> 8) & 255},${n & 255},${a})`;
    },

    // ══ Rows 6–8: the surface panels ═════════════════════════════════════
    //
    // These read equity_surface / equity_atm rather than the metric layer, so
    // they load from /api/equity-iv/{curve-band,tent,sticky-strike,
    // surface-grid,time-scatter,spot-vol}. All of them follow the page's
    // scoring rule: today at the selected snapshot, history at the daily
    // close ending at the prior session.

    async loadSurface() {
      if (!this.selectedTicker) {
        this.cb = {}; this.tent = null; this.sticky = null;
        this.grid = null; this.tscat = null; this.svol = null;
        return;
      }
      await Promise.all([
        this.loadCurveBand('skew'),
        this.loadCurveBand(this.termKind),
        this.loadTent(),
        this.loadSticky(),
        this.loadGrid(),
        this.loadTimeScatter(),
        this.loadSpotVol(),
      ]);
    },

    /** Shared query string for every surface panel. */
    _sq(extra) {
      const q = new URLSearchParams({
        ticker: this.selectedTicker,
        window: this.histWindow,
        exclude_extrapolated: String(this.excludeExtrap),
      });
      if (this.date) q.set('date', this.date);
      if (this.snapshot) q.set('snapshot', this.snapshot);
      for (const [k, v] of Object.entries(extra || {})) q.set(k, String(v));
      return q;
    },

    // ── Row 6a/6c: the curve band panels ────────────────────────────────

    async loadCurveBand(kind) {
      this.cbLoading[kind] = true; this.cbError[kind] = '';
      try {
        const j = await eqGetJson('/api/equity-iv/curve-band?' + this._sq({
          kind, dte: this.cbDte, wing: this.cbWing,
          deltas: this.termDeltas.join(','),
        }));
        if (j.error) { this.cbError[kind] = j.error; this.cb[kind] = null; }
        else {
          this.cb[kind] = j;
          if (j.available) this.adoptDtes(j.available.dtes);
          this.renderCurveBand(kind);
        }
      } catch (e) {
        this.cbError[kind] = String(e.message || e); this.cb[kind] = null;
      } finally {
        this.cbLoading[kind] = false;
      }
    },

    setTermKind(k) {
      if (this.termKind === k) return;
      // The old kind's chart is destroyed rather than left behind: both
      // render into the same canvas, and Chart.js will happily keep the
      // previous instance alive and attached to it.
      const old = EQ_CHARTS['cb-' + this.termKind];
      if (old) { old.destroy(); EQ_CHARTS['cb-' + this.termKind] = null; }
      this.termKind = k;
      this.loadCurveBand(k);
    },

    setCbDte(v) {
      const n = parseInt(v, 10);
      if (!isFinite(n) || n === this.cbDte) return;
      this.cbDte = n;
      this.cbDteChosen = true;
      this.loadCurveBand('skew');
      this.loadTent();
      this.loadSticky();
      this.loadSpotVol();
    },

    /** Adopt the surface's real tenor list, and settle the default once. */
    adoptDtes(list) {
      if (!list || !list.length) return;
      if (this.dteOptions.length !== list.length
          || this.dteOptions.some((t, i) => t !== list[i])) {
        this.dteOptions = [...list];
      }
      if (this.cbDteChosen) return;
      this.cbDteChosen = true;
      const want = this.pickDefaultDte(list);
      if (want !== this.cbDte) {
        this.cbDte = want;
        this.loadCurveBand('skew');
        this.loadTent();
        this.loadSticky();
        this.loadSpotVol();
      }
    },

    /* 0DTE is a poor default: the fit is least reliable at the shortest
     * tenor and it is the first thing anyone sees. 30 is the working tenor
     * for this style; if the surface does not carry it, the nearest tenor of
     * at least 7 days wins, and only an all-sub-7 surface can land on 0. */
    pickDefaultDte(list) {
      if (!list || !list.length) return this.cbDte;
      const usable = list.filter(t => t >= 7);
      const pool = usable.length ? usable : list;
      return pool.reduce((a, t) =>
        Math.abs(t - 30) < Math.abs(a - 30) ? t : a, pool[0]);
    },

    /* The wings the skew-term view can be drawn at, in put convention.
     * 10/25 are the put wings; 75/90 are the 25- and 10-delta CALLS. Whether
     * the call wing is rich by tenor is a different question from the put
     * wing, and the panel could previously only ask one of them. */
    cbWings: [10, 25, 75, 90],

    setCbWing(v) {
      const n = parseInt(v, 10);
      if (!isFinite(n) || n === this.cbWing) return;
      this.cbWing = n;
      this.loadCurveBand('skew_term');
    },

    /* Band + today's line, on one canvas.
     *
     * Four fill datasets stacked from P10 up, each filling to the one below,
     * so the shading reads as a nested envelope rather than four ribbons.
     * The median is a line because a median is a location, not an edge. */
    renderCurveBand(kind) {
      if (typeof Chart === 'undefined') return;
      const key = 'cb-' + kind;
      if (EQ_CHARTS[key]) { EQ_CHARTS[key].destroy(); EQ_CHARTS[key] = null; }
      const j = this.cb[kind];
      if (!j) return;
      this.$nextTick(() => this.paintCurveBand(kind));
    },

    paintCurveBand(kind) {
      const j = this.cb[kind];
      if (!j || typeof Chart === 'undefined') return;
      const el = document.getElementById('eq-cb-' + (kind === 'skew' ? 'skew' : 'term'));
      if (!el) return;
      const key = 'cb-' + kind;
      if (EQ_CHARTS[key]) { EQ_CHARTS[key].destroy(); EQ_CHARTS[key] = null; }

      // Multi-series panels (term by delta) key both band and line on the
      // series value; the others have a single implicit series.
      const keyed = kind === 'term';
      const xs = [...new Set([...j.band.map(b => b.x), ...j.today.map(t => t.x),
                              ...(j.atm_band || []).map(b => b.x)])].sort((a, b) => a - b);
      const at = (arr, x, s) => arr.find(r => r.x === x && (!keyed || r.series === s));

      const datasets = [];
      const bandFor = (rows, colr, label) => {
        const g = k => xs.map(x => { const r = at(rows, x); return r ? r[k] : null; });
        datasets.push({ label: label + ' P5', data: g('p5'), borderColor: 'transparent',
                        pointRadius: 0, fill: false, order: 5 });
        datasets.push({ label: label + ' P25', data: g('p25'), borderColor: 'transparent',
                        pointRadius: 0, backgroundColor: this.rgba(colr, 0.07),
                        fill: '-1', order: 5 });
        datasets.push({ label: label + ' P75', data: g('p75'), borderColor: 'transparent',
                        pointRadius: 0, backgroundColor: this.rgba(colr, 0.16),
                        fill: '-1', order: 5 });
        datasets.push({ label: label + ' P95', data: g('p95'), borderColor: 'transparent',
                        pointRadius: 0, backgroundColor: this.rgba(colr, 0.07),
                        fill: '-1', order: 5 });
        datasets.push({ label: label + ' median', data: g('p50'),
                        borderColor: this.rgba(colr, 0.5), borderWidth: 0.75,
                        borderDash: [3, 3], pointRadius: 0, fill: false, order: 4 });
      };

      const series = keyed
        ? [...new Set(j.today.map(t => t.series))].sort((a, b) => a - b)
        : [null];

      series.forEach((s, i) => {
        const colr = this.seriesColor(i);
        bandFor(j.band.filter(b => !keyed || b.series === s), colr,
                keyed ? this.deltaLabel(s) : '');
        const pts = xs.map(x => { const t = at(j.today, x, s); return t ? t.v : null; });
        datasets.push({
          label: keyed ? this.deltaLabel(s) : 'today',
          data: pts, borderColor: colr, backgroundColor: colr,
          borderWidth: 1.25, pointHoverRadius: 4, tension: 0, order: 1,
          // A fabricated node is marked on today's line, never dropped: the
          // band already excludes them from history, and a gap here would
          // read as missing data rather than as a value not to trust.
          pointRadius: xs.map(x => {
            const t = at(j.today, x, s); return (t && t.extrap) ? 2 : 0;
          }),
          pointBackgroundColor: xs.map(x => {
            const t = at(j.today, x, s); return (t && t.extrap) ? EQ_PINK : colr;
          }),
        });
      });

      if (j.atm_band && j.atm_band.length) bandFor(j.atm_band, EQ_GREY, 'ATM');
      if (j.atm_today && j.atm_today.length) {
        datasets.push({
          label: 'ATM',
          data: xs.map(x => { const t = j.atm_today.find(r => r.x === x); return t ? t.v : null; }),
          borderColor: '#e8e8e8', borderWidth: 1.25, borderDash: [6, 3],
          pointRadius: 0, tension: 0, order: 1,
        });
      }

      const self = this;
      EQ_CHARTS[key] = new Chart(el.getContext('2d'), {
        type: 'line',
        data: { labels: xs.map(x => String(x)), datasets },
        options: {
          responsive: true, maintainAspectRatio: false, animation: false,
          layout: { padding: { right: 10 } },
          interaction: { mode: 'index', intersect: false },
          scales: {
            x: { title: { display: true, text: self.cbXLabel(kind),
                          color: '#8a8a8a', font: { size: 10 } },
                 ticks: { maxTicksLimit: 10, maxRotation: 0 },
                 grid: { display: false } },
            y: { ticks: { callback: v => self.fmtShort(v, 'vol_decimal') },
                 grid: { color: 'rgba(255,255,255,0.05)' } },
          },
          plugins: {
            // A legend, because the term panel draws a pink line, a blue line
            // and a white dashed one and nothing on screen said which delta
            // each was. Filtered to the real series: the band edges are four
            // shading datasets per series and would swamp it.
            //
            // Only where there is more than one line to name. The skew and
            // skew-term panels draw a single series, and a legend reading
            // "today" under them is a caption for something already obvious.
            legend: {
              display: keyed || !!(j.atm_today && j.atm_today.length),
              position: 'bottom',
              labels: {
                boxWidth: 10, boxHeight: 2, font: { size: 9 }, padding: 8,
                usePointStyle: false,
                filter: it => !/ (P\d+|median)$/.test(it.text),
              },
            },
            tooltip: {
              // Same reason: listing the four shading datasets turns a
              // five-line tooltip into a twenty-line one.
              filter: it => !/ P\d+$/.test(it.dataset.label),
              callbacks: {
                label: it => `${it.dataset.label}  ${self.fmt(it.parsed.y, 'vol_decimal')}`,
              },
            },
          },
        },
      });
    },

    cbXLabel(kind) {
      if (kind === 'skew') return 'put_delta  (25 = 25Δ put, 75 = 25Δ call)';
      if (kind === 'skew_term') return 'tenor (days)  ·  wing IV − ATM IV';
      return 'tenor (days)';
    },

    /** put_delta in the language of the trade. 75 is a 25-delta CALL. */
    deltaLabel(dl) {
      if (dl == null) return '';
      if (dl === 50) return '50Δ put';
      return dl < 50 ? `${dl}Δ put` : `${100 - dl}Δ call`;
    },

    cbSubtitle(kind) {
      const j = this.cb[kind];
      if (!j) return '';
      const n = j.band.length ? Math.max(...j.band.map(b => b.n || 0)) : 0;
      const a = j.axis || {};
      const what = kind === 'skew' ? `${a.dte}d`
                 : kind === 'skew_term' ? `${this.deltaLabel(a.wing)} − ATM`
                 : (a.deltas || []).map(d => this.deltaLabel(d)).join(', ');
      return `${what} · band from ${n} prior sessions`;
    },

    // ── Row 6b: the tent ────────────────────────────────────────────────

    async loadTent() {
      this.tentLoading = true; this.tentError = '';
      try {
        const j = await eqGetJson('/api/equity-iv/tent?' + this._sq({
          dte: this.cbDte, long_delta: this.tentLongDelta,
        }));
        if (j.error) { this.tentError = j.error; this.tent = null; }
        else {
          this.tent = j;
          if (j.available) this.adoptDtes(j.available.dtes);
          // Adopt the tenor the server actually RESOLVED. It snaps the request
          // to the nearest fitted tenor, and if that differs from what was
          // asked for, the control has to move with it or the label and the
          // chart disagree with nothing on screen to say so.
          if (j.dte != null && j.dte !== this.cbDte) {
            this.cbDte = j.dte;
            this.cbDteChosen = true;
          }
          this.renderTent();
        }
      } catch (e) {
        this.tentError = String(e.message || e); this.tent = null;
      } finally {
        this.tentLoading = false;
      }
    },

    renderTent() {
      if (typeof Chart === 'undefined') return;
      if (EQ_CHARTS.tent) { EQ_CHARTS.tent.destroy(); EQ_CHARTS.tent = null; }
      this.$nextTick(() => this.paintTent());
    },

    /* The 1×2 payoff at expiry, in ATM-implied-move units.
     *
     * Long one put at K_long, short two at K_short. Below K_short the position
     * loses one unit of notional per unit of further decline, which is the
     * whole risk of the structure and the reason the panel draws the payoff
     * rather than just the width. */
    paintTent() {
      const j = this.tent;
      if (!j || typeof Chart === 'undefined') return;
      const el = document.getElementById('eq-tent');
      if (!el) return;
      const L = j.long_leg, S = j.short_leg;
      if (!L || !S || L.sigma == null || S.sigma == null) return;

      /* Framed off the DATA: what has to be legible is the legs plus the
       * historical band, and that span gets ~80% of the width.
       *
       * Fixed windows do not work here. -4σ packed the structure into the
       * right quarter; -2.5σ was better and still left it small whenever the
       * short sits close in — and how far out the zero-cost strike lands is
       * exactly the thing that varies by name and by day. So the span of
       * interest is measured, then padded to 1/0.8 of itself. */
      const band = j.band || {};
      const marks = [S.sigma, L.sigma, 0].concat(
        [band.p5, band.p25, band.p50, band.p75, band.p95]
          .filter(v => v != null).map(v => -Math.abs(v)));
      const iLo = Math.min(...marks), iHi = Math.max(...marks);
      const span = Math.max(0.4, iHi - iLo);
      const pad  = span * (1 / 0.8 - 1) / 2;
      const lo = iLo - pad, hi = iHi + pad;
      const step = (hi - lo) / 160;
      const xs = [], ys = [];
      const net = (j.zc_cost != null) ? j.zc_cost : 0;
      // Payoff in the same sigma units as the x axis, so the vertical scale
      // is "implied moves of P&L" rather than dollars — comparable across
      // dates and names, which is the point of the whole panel.
      for (let s = lo; s <= hi + 1e-9; s += step) {
        const longPay  = Math.max(0, L.sigma - s);
        const shortPay = -2 * Math.max(0, S.sigma - s);
        xs.push(s);
        ys.push(longPay + shortPay - net);
      }

      const self = this;
      EQ_CHARTS.tent = new Chart(el.getContext('2d'), {
        type: 'line',
        data: {
          labels: xs.map(x => x.toFixed(2)),
          datasets: [{
            label: 'payoff', data: ys,
            borderColor: EQ_BLUE, borderWidth: 1.25, pointRadius: 0,
            tension: 0, fill: false,
          }],
        },
        options: {
          responsive: true, maintainAspectRatio: false, animation: false,
          layout: { padding: { right: 10 } },
          interaction: { mode: 'index', intersect: false },
          scales: {
            x: { title: { display: true, text: 'σ from spot (ATM implied move)',
                          color: '#8a8a8a', font: { size: 10 } },
                 ticks: { maxTicksLimit: 9, maxRotation: 0 },
                 grid: { display: false } },
            y: { grid: { color: 'rgba(255,255,255,0.05)' },
                 ticks: { callback: v => v.toFixed(2) } },
          },
          plugins: {
            legend: { display: false },
            tooltip: { callbacks: {
              title: it => `${Number(it[0].label).toFixed(2)}σ`,
              label: it => `payoff ${it.parsed.y.toFixed(3)}σ`,
            } },
            eqTentMarks: {
              band: j.band, longSigma: L.sigma, shortSigma: S.sigma,
              dnSigma: j.dn_width_sigma == null ? null : -Math.abs(j.dn_width_sigma),
              xs,
            },
          },
        },
        plugins: [eqTentMarks],
      });
    },

    tentNote() {
      const j = this.tent;
      if (!j) return '';
      const bits = [];
      if (j.zc_width_sigma != null) {
        bits.push(`zero-cost short at ${Math.abs(j.zc_width_sigma).toFixed(2)}σ`);
      }
      if (j.dn_width_sigma != null) {
        bits.push(`delta-neutral at ${Math.abs(j.dn_width_sigma).toFixed(2)}σ`);
      }
      if (j.band && j.band.p50 != null) {
        bits.push(`usual ${Math.abs(j.band.p50).toFixed(2)}σ (P25–P75 ` +
                  `${Math.abs(j.band.p75).toFixed(2)}–${Math.abs(j.band.p25).toFixed(2)}, ` +
                  `P5–P95 ${Math.abs(j.band.p95).toFixed(2)}–${Math.abs(j.band.p5).toFixed(2)})`);
      }
      return bits.join('  ·  ');
    },

    /** The gap that is the actual skew reading, in the entry decision's units. */
    tentGapNote() {
      const j = this.tent;
      if (!j || j.zc_width_sigma == null || j.dn_width_sigma == null) return '';
      const gap = Math.abs(j.zc_width_sigma) - Math.abs(j.dn_width_sigma);
      const dir = gap > 0 ? 'further out than' : 'closer in than';
      return `Zero-cost sits ${Math.abs(gap).toFixed(2)}σ ${dir} delta-neutral — `
           + `${gap > 0 ? 'steep' : 'flat'} skew. That gap, not either number on `
           + `its own, is the skew reading in the units the entry uses.`;
    },

    /** A convention mismatch would be invisible on screen, so it is stated. */
    tentCheckNote() {
      const c = this.tent && this.tent.sigma_check;
      if (!c || c.agrees !== false) return '';
      return `σ convention mismatch: the stored width is `
           + `${c.stored == null ? '—' : c.stored.toFixed(2)}σ but this panel `
           + `derives ${c.derived == null ? '—' : c.derived.toFixed(2)}σ from the `
           + `surface. The band is right; the payoff diagram's x values are this `
           + `page's convention, not the loader's.`;
    },

    // ── Row 7a: sticky-strike decomposition ─────────────────────────────

    async loadSticky() {
      this.stickyLoading = true; this.stickyError = '';
      try {
        const j = await eqGetJson('/api/equity-iv/sticky-strike?'
          + this._sq({ dte: this.cbDte }));
        if (j.error) { this.stickyError = j.error; this.sticky = null; }
        else { this.sticky = j; this.renderSticky(); }
      } catch (e) {
        this.stickyError = String(e.message || e); this.sticky = null;
      } finally {
        this.stickyLoading = false;
      }
    },

    renderSticky() {
      if (typeof Chart === 'undefined') return;
      if (EQ_CHARTS.sticky) { EQ_CHARTS.sticky.destroy(); EQ_CHARTS.sticky = null; }
      this.$nextTick(() => this.paintSticky());
    },

    /* Three smiles in strike space plus the residual on a twin axis.
     *
     * The residual is the panel's answer, so it gets its own scale rather
     * than being squeezed onto the IV axis where a 40bp repricing is a
     * hairline. Pink and filled, because "how much did the surface actually
     * reprice" is what the eye should land on. */
    /* Three smiles in strike space plus the residual on a twin axis.
     *
     * A LINEAR x axis carrying {x, y} points, not a category axis over shared
     * labels. The first version used categories over the union of both
     * sessions' strikes, which is wrong twice over: strikes are not evenly
     * spaced, and — the reason the panel rendered blank — spot moves, so the
     * delta-derived strikes differ between sessions. Every line then had a
     * value at its own ~19 strikes and null at the other session's, and with
     * pointRadius 0 and spanGaps false each segment was broken by a null.
     * Nothing drew, while the y-axis still scaled off the data and looked
     * entirely healthy.
     *
     * The residual gets its own scale rather than being squeezed onto the IV
     * axis, where a 40bp repricing is a hairline. It is the panel's answer,
     * so it is the filled pink one. */
    paintSticky() {
      const j = this.sticky;
      if (!j || typeof Chart === 'undefined') return;
      const el = document.getElementById('eq-sticky');
      if (!el) return;

      const xy = (rows, f) => rows
        .filter(p => p.strike != null && (f ? f(p) : p.iv) != null)
        .map(p => ({ x: p.strike, y: f ? f(p) : p.iv }))
        .sort((a, b) => a.x - b.x);

      const self = this;
      EQ_CHARTS.sticky = new Chart(el.getContext('2d'), {
        type: 'line',
        data: {
          datasets: [
            { label: 'residual (repricing)', yAxisID: 'yR',
              data: xy(j.residual, r => r.v), parsing: false,
              borderColor: EQ_PINK, backgroundColor: this.rgba(EQ_PINK, 0.14),
              borderWidth: 1, pointRadius: 0, fill: 'origin', tension: 0, order: 5 },
            { label: `${j.prev_date} smile`,
              data: xy(j.prev), parsing: false,
              borderColor: this.rgba('#8a8a8a', 0.85), borderWidth: 1,
              borderDash: [5, 3], pointRadius: 0, tension: 0, order: 2 },
            // Named for what it MEANS, not for how it was built. This is the
            // counterfactual and the most important line on the panel: the
            // prior smile with nothing changed except where spot sits.
            { label: `${j.prev_date} smile, if only spot moved`,
              data: xy(j.shifted), parsing: false,
              borderColor: this.rgba(EQ_BLUE, 0.55), borderWidth: 1,
              borderDash: [2, 2], pointRadius: 0, tension: 0, order: 2 },
            { label: `${j.date} smile`,
              data: xy(j.today), parsing: false,
              borderColor: EQ_BLUE, borderWidth: 1.5, pointRadius: 0,
              tension: 0, order: 1 },
          ],
        },
        options: {
          responsive: true, maintainAspectRatio: false, animation: false,
          layout: { padding: { right: 10 } },
          interaction: { mode: 'nearest', axis: 'x', intersect: false },
          scales: {
            x: { type: 'linear',
                 title: { display: true, text: 'strike', color: '#8a8a8a',
                          font: { size: 10 } },
                 ticks: { maxTicksLimit: 9, maxRotation: 0,
                          callback: v => Number(v).toFixed(0) },
                 grid: { display: false } },
            y: { position: 'left',
                 ticks: { callback: v => self.fmtShort(v, 'vol_decimal') },
                 grid: { color: 'rgba(255,255,255,0.05)' } },
            yR: { position: 'right',
                  ticks: { callback: v => (v * 100).toFixed(1) + 'pt' },
                  grid: { drawOnChartArea: false } },
          },
          plugins: {
            legend: { display: true, position: 'bottom',
                      labels: { boxWidth: 10, boxHeight: 2, font: { size: 9 },
                                padding: 8, usePointStyle: false } },
            tooltip: { callbacks: {
              title: it => 'strike ' + Number(it[0].parsed.x).toFixed(2),
              label: it => it.dataset.yAxisID === 'yR'
                ? `residual ${(it.parsed.y * 100).toFixed(2)} vol pts`
                : `${it.dataset.label}  ${self.fmt(it.parsed.y, 'vol_decimal')}`,
            } },
            eqSpotMarks: { spot: j.spot, prevSpot: j.prev_spot },
          },
        },
        plugins: [eqSpotMarks],
      });
    },

    stickyNote() {
      const j = this.sticky;
      if (!j) return '';
      const r = j.spot_return;
      const move = r == null ? '—' : (r * 100).toFixed(2) + '%';
      let s = `Spot ${move} since ${j.prev_date}. The dashed blue line is that `
            + `session's smile re-read at today's spot — the part of any skew `
            + `change that is strike migration, not repricing. Pink is the rest, `
            + `and the rest is the trade.`;
      if (j.n_out_of_domain) {
        s += `  ${j.n_out_of_domain} of today's strikes fall outside the prior `
           + `session's fitted range, so they have no sticky-delta reading and `
           + `no residual — the line stops rather than extrapolating one.`;
      }
      return s;
    },

    // ── Row 7b: the surface grid ────────────────────────────────────────

    async loadGrid() {
      this.gridLoading = true; this.gridError = '';
      try {
        const j = await eqGetJson('/api/equity-iv/surface-grid?'
          + this._sq({ view: this.gridView }));
        if (j.error) { this.gridError = j.error; this.grid = null; }
        else {
          this.grid = j;
          if (j.dtes) this.adoptDtes(j.dtes);
        }
      } catch (e) {
        this.gridError = String(e.message || e); this.grid = null;
      } finally {
        this.gridLoading = false;
      }
    },

    setGridView(v) {
      if (this.gridView === v) return;
      this.gridView = v;
      this.loadGrid();
    },

    gridViews: EQ_GRID_VIEWS,

    gridCell(dte, dl) {
      if (!this.grid) return null;
      return this.grid.cells.find(c => c.dte === dte && c.put_delta === dl) || null;
    },

    /* Diverging blue/pink about zero, scaled to the view's own range.
     *
     * Scaled per view rather than on one fixed scale because the views are in
     * different units — raw IV is a level around 0.3, a 1-day change is a few
     * hundredths — and a shared scale would render four of the five flat. */
    gridRange() {
      if (!this.grid) return [0, 1];
      const vs = this.grid.cells.map(c => c.v).filter(v => v != null);
      if (!vs.length) return [0, 1];
      if (this.gridView === 'iv') return [Math.min(...vs), Math.max(...vs)];
      const m = Math.max(...vs.map(Math.abs));
      return [-m, m];
    },

    gridStyle(c) {
      if (!c || c.v == null) return 'background:transparent';
      const [lo, hi] = this.gridRange();
      if (this.gridView === 'iv') {
        const t = hi === lo ? 0.5 : (c.v - lo) / (hi - lo);
        return `background:${this.rgba(EQ_BLUE, 0.08 + t * 0.62)}`;
      }
      const m = Math.max(Math.abs(lo), Math.abs(hi)) || 1;
      const t = Math.min(1, Math.abs(c.v) / m);
      const col = c.v >= 0 ? EQ_BLUE : EQ_PINK;
      return `background:${this.rgba(col, 0.06 + t * 0.64)}`;
    },

    /* Which grid rows were read straight off a listed expiry.
     *
     * `dte_actual` is the node's TRUE tenor. A row where it differs from the
     * target tenor came from a real expiry; a row where it matches exactly was
     * interpolated between the two that bracket the target. That is invisible
     * in the numbers and shows up as a step in the column — AAPL's 7d row
     * sitting below both 5d and 10d — which reads as a data fault until the
     * grid says which rows are which. It also MOVES as the expiry calendar
     * moves, so it cannot be learned once and remembered.
     *
     * The surface fit itself is built in the separate Open_Interest project,
     * so this is the stored signature of a direct read, not a claim about the
     * rule that produced it. */
    gridRow(dte) {
      if (!this.grid || !this.grid.rows) return null;
      return this.grid.rows.find(r => r.dte === dte) || null;
    },

    gridRowDirect(dte) {
      const r = this.gridRow(dte);
      return !!(r && r.direct);
    },

    gridRowTitle(dte) {
      const r = this.gridRow(dte);
      if (!r || r.dte_actual == null) return `${dte}-day tenor`;
      if (!r.direct) {
        return `${dte}-day target, blended from the expiries either side `
             + `(dte_actual ${r.dte_actual.toFixed(2)}).`;
      }
      return `${dte}-day target read DIRECTLY off a listed expiry at `
           + `${r.dte_actual.toFixed(2)} days — no blending. Rows around it are `
           + `interpolated, so a step here is the fit changing method, not the `
           + `surface changing shape.`;
    },

    gridDirectNote() {
      if (!this.grid || !this.grid.rows) return '';
      const hits = this.grid.rows.filter(r => r.direct).map(r => r.dte);
      if (!hits.length) return '';
      return `A · beside a tenor marks a row read directly off a listed expiry `
           + `rather than blended from the two either side (${hits.join(', ')}d `
           + `today). Those rows can sit a step away from their neighbours — `
           + `that is the fit changing method, not the surface changing shape, `
           + `and it moves as the expiry calendar moves.`;
    },

    gridFmt() {
      const hit = EQ_GRID_VIEWS.find(v => v.k === this.gridView);
      return hit ? hit.fmt : 'pt';
    },

    gridText(c) {
      if (!c || c.v == null) return '';
      const f = this.gridFmt();
      if (f === 'z')   return c.v.toFixed(1);
      if (f === 'pt0') return (c.v * 100).toFixed(0);
      return (c.v * 100).toFixed(1);
    },

    /** Units for the tooltip, so a z is never read as vol points. */
    gridUnits() { return this.gridFmt() === 'z' ? 'σ' : 'vol pts'; },

    gridTitle(c) {
      if (!c) return '';
      const parts = [`${c.dte}d  ${this.deltaLabel(c.put_delta)}`];
      if (c.iv != null) parts.push(`IV ${this.fmt(c.iv, 'vol_decimal')}`);
      if (c.atm_iv != null) parts.push(`ATM ${this.fmt(c.atm_iv, 'vol_decimal')}`);
      if (c.strike != null) parts.push(`strike ${c.strike.toFixed(2)}`);
      if (c.v != null) {
        parts.push(`${this.gridViewLabel()} ${this.gridText(c)} ${this.gridUnits()}`);
      }
      if (c.extrap) parts.push('node fabricated by the smile fit — not an observation');
      return parts.join('\n');
    },

    gridViewLabel() {
      const hit = EQ_GRID_VIEWS.find(v => v.k === this.gridView);
      return hit ? hit.label : this.gridView;
    },

    gridNote() {
      const j = this.grid;
      if (!j) return '';
      let s = '';
      if (this.gridView === 'iv_minus_atm' || this.gridView === 'z_iv_minus_atm') {
        s = 'Anchored on equity_atm at k=0, not put_delta 50 — they are '
          + 'different nodes and the difference is what is being measured. '
          + 'Spread-to-ATM rather than the cell’s own IV, because z-scoring '
          + 'raw IV lights the whole grid up together whenever vol is high or '
          + 'low and turns a heatmap into an expensive ATM readout.';
      } else if (j.reference_date) {
        s = `Change against the ${j.reference_date} close, per node.`;
      }
      if (j.n_thin_baseline) {
        s += `  ${j.n_thin_baseline} cell(s) had too little history to score and `
           + `are blank rather than shown at zero.`;
      }
      return s;
    },

    // ── Row 8a: the time-scatter ────────────────────────────────────────

    tsXCol() { return this.resolve(this.tsBaseX, this.tsZX); },
    tsYCol() { return this.resolve(this.tsBaseY, this.tsZY); },

    onTsFamily(axis) {
      if (axis === 'x') {
        this.tsBaseX = this.firstMetric(this.tsFamX);
        if (!this.hasZ(this.tsBaseX)) this.tsZX = false;
      } else {
        this.tsBaseY = this.firstMetric(this.tsFamY);
        if (!this.hasZ(this.tsBaseY)) this.tsZY = false;
      }
      this.loadTimeScatter();
    },

    setTsForm(axis, useZ) {
      const base = axis === 'x' ? this.tsBaseX : this.tsBaseY;
      if (useZ && !this.hasZ(base)) return;   // no z variant — the toggle is inert
      if (axis === 'x') this.tsZX = useZ; else this.tsZY = useZ;
      this.loadTimeScatter();
    },

    /** Copy the Cross-section's current pair, for when the two should match. */
    tsCopyGlobal() {
      this.tsFamX = this.xFam; this.tsBaseX = this.xBase; this.tsZX = this.xZ;
      this.tsFamY = this.yFam; this.tsBaseY = this.yBase; this.tsZY = this.yZ;
      this.loadTimeScatter();
    },

    async loadTimeScatter() {
      this.tscatLoading = true; this.tscatError = '';
      try {
        const j = await eqGetJson('/api/equity-iv/time-scatter?' + this._sq({
          x: this.tsXCol(), y: this.tsYCol(),
        }));
        if (j.error) { this.tscatError = j.error; this.tscat = null; }
        else { this.tscat = j; this.renderTimeScatter(); }
      } catch (e) {
        this.tscatError = String(e.message || e); this.tscat = null;
      } finally {
        this.tscatLoading = false;
      }
    },

    renderTimeScatter() {
      if (typeof Chart === 'undefined') return;
      if (EQ_CHARTS.tscat) { EQ_CHARTS.tscat.destroy(); EQ_CHARTS.tscat = null; }
      this.$nextTick(() => this.paintTimeScatter());
    },

    /* ONE age-scatter renderer, shared by Path and Spot-vol.
     *
     * These were two implementations of the same rule -- older points on an
     * age gradient, the most recent EQ_RECENT_N in pink, the current reading
     * highlighted -- and they diverged on screen three times running despite
     * calling the same ramp helpers. A rule written twice is a rule that will
     * diverge again, so there is now one path and the callers differ only in
     * what they hand it.
     *
     * cfg: { key, el, hist, highlight, fit, xTitle, yTitle, xTick, yTick,
     *        label }
     *   hist       oldest-first. Order IS the age, so the caller must not
     *              re-sort it.
     *   highlight  the single current point, drawn brightest, or null.
     *   fit        [{x,y},{x,y}] for a regression line, or null.
     */
    paintAgeScatter(cfg) {
      if (typeof Chart === 'undefined') return;
      const el = document.getElementById(cfg.el);
      if (!el || !cfg.hist.length) return;
      if (EQ_CHARTS[cfg.key]) { EQ_CHARTS[cfg.key].destroy(); EQ_CHARTS[cfg.key] = null; }

      const self = this;
      const hist = cfg.hist;
      const hi   = cfg.highlight ? [cfg.highlight] : [];
      const datasets = [];

      if (cfg.fit && cfg.fit.length === 2) {
        datasets.push({
          label: 'fit', data: cfg.fit, parsing: false, showLine: true,
          borderColor: this.rgba(EQ_BLUE, 0.7), borderWidth: 1,
          borderDash: [5, 3], pointRadius: 0, order: 4,
        });
      }
      datasets.push({
        label: cfg.label || 'history',
        data: hist.map(p => ({ x: p.x, y: p.y })), parsing: false, showLine: false,
        pointRadius: hist.map((_, i) => eqRecentRadius(i, hist.length)),
        pointHoverRadius: 6,
        pointBackgroundColor: hist.map((_, i) => eqRecentColor(i, hist.length)),
        // A thin dark ring on the recent run only. Ten pink dots in a quiet
        // stretch land almost on top of each other and read as about five;
        // the ring keeps them individually countable without making the
        // older cloud noisier.
        pointBorderColor: hist.map((_, i) =>
          (hist.length - 1 - i) < EQ_RECENT_N ? 'rgba(20,20,20,0.85)' : 'transparent'),
        pointBorderWidth: hist.map((_, i) =>
          (hist.length - 1 - i) < EQ_RECENT_N ? 0.75 : 0),
        order: 2,
      });
      if (hi.length) {
        datasets.push({
          label: 'current', data: hi.map(p => ({ x: p.x, y: p.y })),
          parsing: false, showLine: false,
          pointRadius: 6, pointHoverRadius: 8,
          pointBackgroundColor: cfg.highlightHollow ? EQ_SURF : EQ_PINK,
          pointBorderColor: cfg.highlightHollow ? EQ_PINK : '#ffffff',
          pointBorderWidth: cfg.highlightHollow ? 2 : 1.5,
          order: 1,
        });
      }

      const fitOffset = (cfg.fit && cfg.fit.length === 2) ? 1 : 0;
      EQ_CHARTS[cfg.key] = new Chart(el.getContext('2d'), {
        type: 'scatter',
        data: { datasets },
        options: {
          responsive: true, maintainAspectRatio: false, animation: false,
          layout: { padding: { right: 10 } },
          scales: {
            x: { title: { display: true, text: cfg.xTitle, color: '#8a8a8a',
                          font: { size: 10 } },
                 ticks: { callback: v => cfg.xTick(v) },
                 grid: { color: 'rgba(255,255,255,0.05)' } },
            y: { title: { display: true, text: cfg.yTitle, color: '#8a8a8a',
                          font: { size: 10 } },
                 ticks: { callback: v => cfg.yTick(v) },
                 grid: { color: 'rgba(255,255,255,0.05)' } },
          },
          plugins: {
            legend: { display: false },
            tooltip: {
              filter: it => it.datasetIndex !== (fitOffset - 1),
              callbacks: {
                label: it => {
                  const src = it.datasetIndex === fitOffset ? hist : hi;
                  const p = src[it.dataIndex];
                  return p ? cfg.tip(p) : '';
                },
              },
            },
            eqZeroLines: { x: true, y: true },
          },
        },
        plugins: [eqZeroLines],
      });
    },

    /* The point of the panel is the PATH, and a path needs a direction. */
    paintTimeScatter() {
      const j = this.tscat;
      if (!j) return;
      const pts = j.points.filter(p => p.x != null && p.y != null);
      const today = pts.find(p => p.today) || null;
      const hist = pts.filter(p => !p.today);
      const self = this;
      this.paintAgeScatter({
        key: 'tscat', el: 'eq-tscat', hist, highlight: today,
        highlightHollow: false, fit: null,
        xTitle: j.x.column_name, yTitle: j.y.column_name,
        xTick: v => self.fmtShort(v, j.x.units),
        yTick: v => self.fmtShort(v, j.y.units),
        tip: p => `${p.date}  ${self.fmtShort(p.x, j.x.units)} / `
                + `${self.fmtShort(p.y, j.y.units)}`,
      });
    },

    // ── Row 8b: spot-vol scatter ────────────────────────────────────────

    async loadSpotVol() {
      this.svolLoading = true; this.svolError = '';
      try {
        const j = await eqGetJson('/api/equity-iv/spot-vol?'
          + this._sq({ dte: this.cbDte }));
        if (j.error) { this.svolError = j.error; this.svol = null; }
        else { this.svol = j; this.renderSpotVol(); }
      } catch (e) {
        this.svolError = String(e.message || e); this.svol = null;
      } finally {
        this.svolLoading = false;
      }
    },

    renderSpotVol() {
      if (typeof Chart === 'undefined') return;
      if (EQ_CHARTS.svol) { EQ_CHARTS.svol.destroy(); EQ_CHARTS.svol = null; }
      this.$nextTick(() => this.paintSpotVol());
    },

    paintSpotVol() {
      const j = this.svol;
      if (!j) return;
      const hist = j.points.filter(p => !p.partial);
      const live = j.points.find(p => p.partial) || null;
      if (!hist.length) return;

      const xs = hist.map(p => p.ret);
      const lo = Math.min(...xs), hi = Math.max(...xs);
      const fit = (j.fit && j.fit.beta != null)
        ? [{ x: lo, y: j.fit.alpha + j.fit.beta * lo },
           { x: hi, y: j.fit.alpha + j.fit.beta * hi }]
        : null;

      this.paintAgeScatter({
        key: 'svol', el: 'eq-svol',
        hist: hist.map(p => ({ x: p.ret, y: p.d_iv, date: p.date, partial: false })),
        // Hollow, like the live point on the history line and for the same
        // reason: a part-session move is not an overnight move.
        highlight: live ? { x: live.ret, y: live.d_iv, date: live.date, partial: true } : null,
        highlightHollow: true,
        fit,
        xTitle: 'underlying log return', yTitle: 'Δ ATM IV',
        xTick: v => (v * 100).toFixed(1) + '%',
        yTick: v => (v * 100).toFixed(1) + 'pt',
        tip: p => `${p.date}  ret ${(p.x * 100).toFixed(2)}%  `
                + `ΔIV ${(p.y * 100).toFixed(2)}pt`
                + (p.partial ? '  · partial session, not in the fit' : ''),
      });
    },

    svolNote() {
      const j = this.svol;
      if (!j || !j.fit) return '';
      const f = j.fit;
      if (f.beta == null) {
        return `Not enough overnight moves in this window to fit a line (${j.n_fit}).`;
      }
      return `β ${f.beta.toFixed(2)}  ·  R² ${f.r2 == null ? '—' : f.r2.toFixed(2)}`
           + `  ·  ${j.n_fit} overnight moves at ${j.dte}d. The header chips carry `
           + `the stored β; this is the cloud behind it. A respectable β can come `
           + `from two regimes averaged together or three outliers carrying the `
           + `fit, and only the shape says which.`;
    },

    // ── Row 9: open interest, from the existing chain endpoints ─────────

    async loadOi() {
      if (!this.oiOpen || !this.selectedTicker) return;
      this.oiLoading = true; this.oiError = ''; this.oiNote = '';
      try {
        // The chain store is a different dataset from equity_surface — parquet
        // per (ticker, year), keyed on daily_features dates. A ticker in the
        // IV universe is not guaranteed to be in it, and the honest answer
        // when it is not is to say so rather than draw an empty panel.
        if (!this.oiDates.length) {
          const dj = await eqGetJson('/api/ticker-analysis/chain/dates?ticker='
            + encodeURIComponent(this.selectedTicker));
          this.oiDates = dj.dates || [];
        }
        if (!this.oiDates.length) {
          this.oiError = `No option-chain history for ${this.selectedTicker}. `
            + `The OI store is a separate dataset from the IV surface and does `
            + `not cover every ticker in this universe.`;
          this.oi = null;
          return;
        }
        if (!this.oiDate || !this.oiDates.includes(this.oiDate)) {
          // Nearest chain date at or before the page's date; the two datasets
          // are captured independently and do not always have the same days.
          const want = this.date || this.oiDates[this.oiDates.length - 1];
          const usable = this.oiDates.filter(d => d <= want);
          this.oiDate = usable.length ? usable[usable.length - 1]
                                      : this.oiDates[0];
          if (this.oiDate !== want) {
            this.oiNote = `Nearest chain date at or before ${want} is ${this.oiDate}.`;
          }
        }

        const base = { ticker: this.selectedTicker, date: this.oiDate,
                       side: this.oiSide };
        const q = new URLSearchParams(base);
        const dteParam = this.oiDteParam();
        if (dteParam) q.set('dte_bands', dteParam);
        if (this.oiTab === 'doi') q.set('n', String(this.oiDoiN));
        if (this.oiTab === 'flow') { q.set('lookback', String(this.oiLookback)); q.set('mode', 'oi'); }
        const path = this.oiTab === 'profile' ? 'oi-profile'
                   : this.oiTab === 'doi' ? 'doi-profile' : 'flow';

        // The reference lines are computed from the OI PROFILE, which the
        // other two tabs do not return. Fetched alongside them, under the
        // same date / side / DTE filters, so the lines mean the same thing on
        // every tab rather than appearing on one.
        const wantRef = this.oiTab !== 'profile';
        const rq = new URLSearchParams(base);
        if (dteParam) rq.set('dte_bands', dteParam);

        /* The flow map's matrix is total OI per (strike, session) — the
         * endpoint sums both option types unless `side` filters — so a
         * per-session CALL and PUT weighted average cannot be derived from
         * it. It can be derived from two more flow fetches at side=call and
         * side=put, which is cheaper than it looks: each is one parquet pass
         * over the same window, cached server-side on the same key, and the
         * column-wise weighting afterwards is arithmetic on data already in
         * hand. So it costs two extra reads on the FIRST view of the flow tab
         * per (ticker, date, lookback, DTE band) and nothing on re-render or
         * zoom. */
        const sideFlow = (sd) => {
          const fq = new URLSearchParams(base);
          fq.set('side', sd);
          fq.set('lookback', String(this.oiLookback));
          fq.set('mode', 'oi');
          if (dteParam) fq.set('dte_bands', dteParam);
          return eqGetJson('/api/ticker-analysis/chain/flow?' + fq);
        };
        const wantFlowSides = this.oiTab === 'flow';

        const [j, rj, cf, pf] = await Promise.all([
          eqGetJson(`/api/ticker-analysis/chain/${path}?` + q),
          wantRef ? eqGetJson('/api/ticker-analysis/chain/oi-profile?' + rq)
                  : Promise.resolve(null),
          wantFlowSides ? sideFlow('call') : Promise.resolve(null),
          wantFlowSides ? sideFlow('put')  : Promise.resolve(null),
        ]);

        if (j.error) { this.oiError = j.error; this.oi = null; return; }
        this.oiRef = this.oiRefLevels(rj || j);
        this.oiFlowCall = this.oiWeightedPath(cf);
        this.oiFlowPut  = this.oiWeightedPath(pf);
        if (j.empty) {
          this.oi = j;
          this.oiError = `No chain rows for ${this.oiDate} at this DTE / side `
            + `filter. Widen the DTE range or pick another date.`;
        } else { this.oi = j; this.renderOi(); }
      } catch (e) {
        this.oiError = String(e.message || e); this.oi = null;
      } finally {
        this.oiLoading = false;
      }
    },

    /* Three reference levels, all computed from the CHAIN payload itself.
     *
     * This replaces the structure-strike overlay, and the change is not only
     * cosmetic: the structure's strikes come from equity_surface and the
     * chain's ladder is split-adjusted off daily_features, so that overlay
     * was drawing across two bases that do not agree. These three come from
     * one payload, so there is nothing to reconcile.
     *
     * OI-weighted average strike rather than max-OI strike: the single
     * largest strike is often an artefact of one old expiry, while the
     * weighted average says where the positioning actually sits. */
    oiRefLevels(j) {
      const out = { spot: (j && j.spot != null) ? Number(j.spot) : null,
                    call: null, put: null };
      const rows = (j && j.strikes) || [];
      let cw = 0, cs = 0, pw = 0, ps = 0;
      for (const r of rows) {
        const k = Number(r.strike);
        if (!isFinite(k)) continue;
        const c = Number(r.call_oi) || 0, p = Number(r.put_oi) || 0;
        cw += c; cs += c * k; pw += p; ps += p * k;
      }
      if (cw > 0) out.call = cs / cw;
      if (pw > 0) out.put = ps / pw;
      return out;
    },

    /* Column-wise OI-weighted average strike: sum(strike * oi) / sum(oi) per
     * session. Sessions with no OI at all yield null rather than 0, so the
     * line breaks instead of diving to the axis. */
    oiWeightedPath(j) {
      if (!j || j.error || j.empty) return null;
      const strikes = (j.strikes || []).map(Number);
      const matrix = j.matrix || [];
      const dates = j.dates || [];
      if (!strikes.length || !matrix.length) return null;
      const out = [];
      for (let di = 0; di < dates.length; di++) {
        let w = 0, ws = 0;
        for (let si = 0; si < matrix.length; si++) {
          const v = Number((matrix[si] || [])[di]);
          if (!isFinite(v) || v <= 0) continue;
          w += v; ws += v * strikes[si];
        }
        out.push(w > 0 ? { x: di, y: ws / w } : null);
      }
      return out.filter(Boolean);
    },

    toggleOi() {
      this.oiOpen = !this.oiOpen;
      if (this.oiOpen) this.loadOi();
    },

    setOiTab(t) { if (this.oiTab !== t) { this.oiTab = t; this.loadOi(); } },
    setOiSide(v) { if (this.oiSide !== v) { this.oiSide = v; this.loadOi(); } },
    /** Toggle one band. "all" is the empty selection, so picking it clears. */
    toggleOiDte(v) {
      if (!v) {
        if (!this.oiDteSel.length) return;
        this.oiDteSel = [];
      } else {
        const i = this.oiDteSel.indexOf(v);
        if (i >= 0) this.oiDteSel.splice(i, 1);
        else this.oiDteSel.push(v);
      }
      this.loadOi();
    },

    oiDteOn(v) {
      return v ? this.oiDteSel.includes(v) : this.oiDteSel.length === 0;
    },

    /** The CSV the chain endpoints want, or '' for every expiry. */
    oiDteParam() {
      // Ordered by their low bound so the value is stable regardless of the
      // order they were clicked -- it is a cache key on the server.
      return [...this.oiDteSel]
        .sort((a, b) => parseInt(a, 10) - parseInt(b, 10))
        .join(',');
    },

    setOiDateIdx(i) {
      const n = parseInt(i, 10);
      if (!isFinite(n) || !this.oiDates.length) return;
      const d = this.oiDates[Math.max(0, Math.min(this.oiDates.length - 1, n))];
      if (d === this.oiDate) return;
      this.oiDate = d;
      this.oiNote = '';
      this.loadOi();
    },

    /* One session per click. The slider spans the whole store — several
     * hundred sessions across a few hundred pixels — so it moves a week or
     * more per pixel and cannot land on a chosen day. */
    stepOiDate(delta) {
      if (!this.oiDates.length) return;
      this.setOiDateIdx(this.oiDateIdx + delta);
    },

    get oiDateIdx() {
      const i = this.oiDates.indexOf(this.oiDate);
      return i < 0 ? Math.max(0, this.oiDates.length - 1) : i;
    },

    get oiAtStart() { return this.oiDateIdx <= 0; },
    get oiAtEnd() { return this.oiDateIdx >= this.oiDates.length - 1; },

    oiStrikeList() {
      const j = this.oi;
      if (!j) return [];
      if (this.oiTab === 'flow') return (j.strikes || []).map(Number);
      return (j.strikes || []).map(r => Number(r.strike));
    },

    // ── flow zoom ───────────────────────────────────────────────────────

    /* The flow map's strike axis spans every strike ever listed, most of
     * which carry nothing. Drawn full-range the interesting band is a few
     * pixels tall, which is why the Ticker Analysis version reads better —
     * it is not bigger, it is zoomed.
     *
     * The default half-range is measured: the narrowest band around spot that
     * holds ~90% of the map's total absolute flow. Zoom multiplies it. */
    oiFlowRange() {
      const j = this.oi;
      if (!j || this.oiTab !== 'flow') return null;
      const strikes = (j.strikes || []).map(Number);
      const matrix = j.matrix || [];
      if (!strikes.length || !matrix.length) return null;
      const centre = (this.oiRef && this.oiRef.spot) || strikes[Math.floor(strikes.length / 2)];

      const weight = strikes.map((k, si) =>
        (matrix[si] || []).reduce((a, v) => a + Math.abs(Number(v) || 0), 0));
      const total = weight.reduce((a, w) => a + w, 0);
      let half;
      if (total > 0) {
        // Grow a window outward from spot until it holds 90% of the flow.
        const order = strikes.map((k, i) => ({ d: Math.abs(k - centre), w: weight[i] }))
                             .sort((a, b) => a.d - b.d);
        let acc = 0;
        half = order[order.length - 1].d;
        for (const o of order) {
          acc += o.w;
          if (acc >= total * 0.90) { half = o.d; break; }
        }
      } else {
        half = (Math.max(...strikes) - Math.min(...strikes)) / 2;
      }
      half = Math.max(half, centre * 0.02) * this.oiFlowZoom;
      return { min: centre - half, max: centre + half, centre };
    },

    setOiFlowZoom(mult) {
      const z = Math.max(0.15, Math.min(8, this.oiFlowZoom * mult));
      if (z === this.oiFlowZoom) return;
      this.oiFlowZoom = z;
      this.renderOi();
    },

    resetOiFlowZoom() {
      if (this.oiFlowZoom === 1) return;
      this.oiFlowZoom = 1;
      this.renderOi();
    },

    renderOi() {
      if (typeof Chart === 'undefined') return;
      if (EQ_CHARTS.oi) { EQ_CHARTS.oi.destroy(); EQ_CHARTS.oi = null; }
      this.$nextTick(() => this.paintOi());
    },

    /* One renderer per payload shape.
     *
     * The three chain endpoints return three DIFFERENT shapes, and the first
     * version of this panel pointed all three tabs at a single renderer that
     * only understood the profile's. ΔOI came back as {strike, doi} and drew
     * nothing; flow came back as a strike x time matrix whose `strikes` are
     * bare numbers, so reading r.strike gave undefined and the axis filled
     * with NaN. Both looked like "no data" rather than like a bug. */
    paintOi() {
      const j = this.oi;
      if (!j || typeof Chart === 'undefined' || j.empty) return;
      const el = document.getElementById('eq-oi');
      if (!el) return;
      if (this.oiTab === 'profile') return this.paintOiProfile(el, j);
      if (this.oiTab === 'doi') return this.paintOiDoi(el, j);
      return this.paintOiFlow(el, j);
    },

    _oiRefPlugin(horizontal) {
      const r = this.oiRef || {};
      const marks = [];
      if (r.spot != null) marks.push({ v: r.spot, color: '#e8e8e8', label: 'spot' });
      if (r.call != null) marks.push({ v: r.call, color: EQ_BLUE, label: 'call OI avg' });
      if (r.put  != null) marks.push({ v: r.put,  color: EQ_PINK, label: 'put OI avg' });
      return { marks, strikes: this.oiStrikeList(), horizontal: !!horizontal };
    },

    paintOiProfile(el, j) {
      const rows = j.strikes || [];
      EQ_CHARTS.oi = new Chart(el.getContext('2d'), {
        type: 'bar',
        data: {
          labels: rows.map(r => Number(r.strike).toFixed(2)),
          datasets: [
            { label: 'put',  data: rows.map(r => r.put_oi),
              backgroundColor: this.rgba(EQ_PINK, 0.6), borderWidth: 0 },
            { label: 'call', data: rows.map(r => r.call_oi),
              backgroundColor: this.rgba(EQ_BLUE, 0.6), borderWidth: 0 },
          ],
        },
        options: this._oiBarOptions('open interest'),
      });
    },

    paintOiDoi(el, j) {
      const rows = j.strikes || [];
      // Signed: a build and an unwind are opposite facts and must not share a
      // colour. Blue for a build, pink for an unwind — the page's directions.
      const vals = rows.map(r => r.doi);
      const self = this;
      EQ_CHARTS.oi = new Chart(el.getContext('2d'), {
        type: 'bar',
        data: {
          labels: rows.map(r => Number(r.strike).toFixed(2)),
          datasets: [{
            label: 'ΔOI', data: vals, borderWidth: 0,
            backgroundColor: vals.map(v =>
              self.rgba(v >= 0 ? EQ_BLUE : EQ_PINK, 0.65)),
          }],
        },
        options: this._oiBarOptions(
          'ΔOI  (' + (j.date_prev || '') + ' → ' + (j.date || '') + ')'),
      });
    },

    paintOiFlow(el, j) {
      const dates = j.dates || [];
      const strikes = (j.strikes || []).map(Number);
      const matrix = j.matrix || [];
      const scale = j.max || 1;
      const range = this.oiFlowRange();
      const pts = [];
      for (let si = 0; si < matrix.length; si++) {
        const k = strikes[si];
        // Clipped to the visible band rather than left to the scale: several
        // thousand off-screen points cost paint time and nothing else.
        if (range && (k < range.min || k > range.max)) continue;
        const row = matrix[si] || [];
        for (let di = 0; di < row.length; di++) {
          const v = row[di];
          if (v == null || v === 0) continue;
          pts.push({ x: di, y: k, v });
        }
      }
      const self = this;
      EQ_CHARTS.oi = new Chart(el.getContext('2d'), {
        type: 'scatter',
        data: {
          datasets: [
            { label: 'flow', data: pts, parsing: false,
              pointRadius: pts.map(p =>
                Math.max(1, Math.min(7, 1 + 6 * Math.sqrt(Math.abs(p.v) / scale)))),
              pointBackgroundColor: pts.map(p =>
                self.rgba(p.v >= 0 ? EQ_BLUE : EQ_PINK, 0.55)),
              pointBorderWidth: 0, order: 2 },
            // The spot path, so "where OI concentrated" can be read against
            // "where price went" — which is the whole point of the panel.
            { label: 'spot', parsing: false, showLine: true, pointRadius: 0,
              data: (j.spots || []).map((s, i) => (s == null ? null : { x: i, y: s }))
                                   .filter(Boolean),
              borderColor: '#e8e8e8', borderWidth: 1, order: 1 },
            // Where the call and put books sat, session by session. Their
            // spread against each other, and against the spot path, is the
            // thing a single end-of-window number cannot show.
            { label: 'call OI avg', parsing: false, showLine: true, pointRadius: 0,
              data: this.oiFlowCall || [],
              borderColor: this.rgba(EQ_BLUE, 0.9), borderWidth: 1,
              borderDash: [3, 2], order: 1 },
            { label: 'put OI avg', parsing: false, showLine: true, pointRadius: 0,
              data: this.oiFlowPut || [],
              borderColor: this.rgba(EQ_PINK, 0.9), borderWidth: 1,
              borderDash: [3, 2], order: 1 },
          ],
        },
        options: {
          responsive: true, maintainAspectRatio: false, animation: false,
          layout: { padding: { right: 10 } },
          scales: {
            x: { type: 'linear', grid: { display: false },
                 ticks: { maxTicksLimit: 8, maxRotation: 0,
                          callback: v => dates[Math.round(v)] || '' } },
            y: { type: 'linear', grid: { color: 'rgba(255,255,255,0.05)' },
                 min: range ? range.min : undefined,
                 max: range ? range.max : undefined,
                 ticks: { callback: v => Number(v).toFixed(0) } },
          },
          plugins: {
            legend: { display: false },
            tooltip: { filter: it => it.datasetIndex === 0, callbacks: {
              label: it => {
                const p = pts[it.dataIndex];
                if (!p) return '';
                return `${dates[p.x] || ''}  strike ${p.y}  `
                     + `${p.v >= 0 ? '+' : ''}${self.fmtShort(p.v, 'count')}`;
              },
            } },
            // Today's levels, drawn only at the right edge — see eqRefLines.
            eqRefLines: this._oiRefPlugin(true),
          },
        },
        plugins: [eqRefLines],
      });
    },

    _oiBarOptions(yLabel) {
      const self = this;
      return {
        responsive: true, maintainAspectRatio: false, animation: false,
        layout: { padding: { right: 10, top: 14 } },
        interaction: { mode: 'index', intersect: false },
        scales: {
          x: { grid: { display: false },
               ticks: { maxTicksLimit: 12, maxRotation: 0 } },
          y: { title: { display: true, text: yLabel, color: '#8a8a8a',
                        font: { size: 10 } },
               grid: { color: 'rgba(255,255,255,0.05)' },
               ticks: { callback: v => self.fmtShort(v, 'count') } },
        },
        plugins: {
          legend: { display: false },
          eqRefLines: this._oiRefPlugin(false),
        },
      };
    },

    oiRefNote() {
      const r = this.oiRef || {};
      const bits = [];
      if (r.spot != null) bits.push(`spot ${Number(r.spot).toFixed(2)}`);
      if (r.call != null) bits.push(`call OI avg ${r.call.toFixed(2)}`);
      if (r.put  != null) bits.push(`put OI avg ${r.put.toFixed(2)}`);
      return bits.join('  ·  ');
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
