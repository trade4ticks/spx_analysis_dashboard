/**
 * charts.js — Chart.js rendering for each panel type.
 *
 * Public API:
 *   renderPanel(panelId, panelConfig, apiData, metric)
 *   destroyPanel(panelId)
 */

'use strict';

// ── Colour palettes ──────────────────────────────────────────────────────────

const DTE_COLORS = {
  0:   '#ff6b6b',
  1:   '#ff8c42',
  2:   '#ffa931',
  3:   '#ffd166',
  4:   '#e6e600',
  5:   '#bdf441',
  6:   '#7ed957',
  7:   '#3498db',
  8:   '#26c6da',
  9:   '#5dade2',
  10:  '#48a3ff',
  14:  '#2ecc71',
  21:  '#1abc9c',
  30:  '#f0b429',
  45:  '#e67e22',
  60:  '#e74c3c',
  90:  '#9b59b6',
  120: '#8e44ad',
  180: '#c0392b',
  270: '#d35400',
  360: '#7f8c8d',
};

const DELTA_COLORS = {
  5:  '#c0392b', 10: '#e74c3c', 15: '#e67e22', 20: '#f39c12',
  25: '#f1c40f', 30: '#d4e157', 35: '#aed581', 40: '#66bb6a',
  45: '#26a69a', 50: '#3498db',
  55: '#26c6da', 60: '#42a5f5', 65: '#5c6bc0', 70: '#7e57c2',
  75: '#ab47bc', 80: '#ec407a', 85: '#ef5350', 90: '#e53935', 95: '#c62828',
};

// Cycle colours for date-based series
const DATE_PALETTE = [
  '#3498db','#2ecc71','#f39c12','#e74c3c','#9b59b6',
  '#1abc9c','#e67e22','#f1c40f','#5dade2','#a9cce3',
];

function dteColor(dte)   { return DTE_COLORS[dte]   || '#aaa'; }
function deltaColor(pd)  { return DELTA_COLORS[pd]  || '#aaa'; }
function dateColor(i)    { return DATE_PALETTE[i % DATE_PALETTE.length]; }

// ── Chart instance registry ───────────────────────────────────────────────────

const _instances = new Map();   // panelId (int) → Chart
let _showTooltips = true;       // toggled by renderPanel caller

function destroyPanel(panelId) {
  if (_instances.has(panelId)) {
    _instances.get(panelId).destroy();
    _instances.delete(panelId);
  }
}

function getChart(panelId) {
  return _instances.get(panelId);
}

// ── Shared Chart.js defaults ──────────────────────────────────────────────────

Chart.defaults.color          = '#c8c8c8';
Chart.defaults.borderColor    = 'rgba(255,255,255,0.07)';
Chart.defaults.font.family    = "'Segoe UI', system-ui, sans-serif";
Chart.defaults.font.size      = 11;

function baseScales(yLabel = '', yMin = undefined, yMax = undefined, pctFmt = true) {
  return {
    x: {
      grid:  { color: 'rgba(255,255,255,0.06)' },
      ticks: { maxRotation: 0, font: { size: 10 } },
    },
    y: {
      grid:  { color: 'rgba(255,255,255,0.06)' },
      ticks: {
        font: { size: 10 },
        callback: v => pctFmt ? (v * 100).toFixed(2) + '%' : v,
      },
      title: yLabel
        ? { display: true, text: yLabel, color: '#777', font: { size: 9 } }
        : { display: false },
      ...(yMin !== undefined ? { min: yMin } : {}),
      ...(yMax !== undefined ? { max: yMax } : {}),
    },
  };
}

function basePlugins() {
  return {
    legend:  { display: false },
    tooltip: {
      enabled:         _showTooltips,
      backgroundColor: '#3a3a3a',
      borderColor:     '#555',
      borderWidth:     1,
      titleFont:       { size: 11 },
      bodyFont:        { size: 11 },
    },
    zoom: {
      pan:  { enabled: true, mode: 'xy', modifierKey: 'shift' },
      zoom: {
        wheel:  { enabled: true },
        pinch:  { enabled: true },
        drag:   { enabled: true, backgroundColor: 'rgba(52,152,219,0.15)', borderColor: '#3498db', borderWidth: 1 },
        mode:   'xy',
      },
    },
  };
}

// ── Main render dispatcher ────────────────────────────────────────────────────

/**
 * @param {number}  panelId   — 0-3
 * @param {string}  type      — 'skew' | 'term' | 'historical' | 'concavity'
 * @param {object}  data      — API response JSON
 * @param {string}  metric    — 'iv' | 'price' | ...
 */
function renderPanel(panelId, type, data, metric = 'iv', showTooltips = true) {
  const canvas = document.getElementById(`canvas-${panelId}`);
  if (!canvas) return;

  const pctFmt = (metric === 'iv');
  const yLabel = metric === 'iv' ? 'Implied Vol (%)' : metric;

  destroyPanel(panelId);
  _showTooltips = showTooltips;

  let config;
  switch (type) {
    case 'skew':       config = data.mode === 'raw_skew' ? buildRawSkew(data, pctFmt, yLabel) : buildSkew(data, pctFmt, yLabel); break;
    case 'term':       config = data.mode === 'raw_term' ? buildRawTerm(data, pctFmt, yLabel) : buildTerm(data, pctFmt, yLabel); break;
    case 'historical': config = buildHistorical(data, pctFmt, yLabel, data.mode === 'raw_historical' ? 'raw_historical' : 'historical'); break;
    case 'convexity':  config = buildHistorical(data, true, 'Convexity', 'convexity');                       break;
    case 'skew_slope': config = buildHistorical(data, true, 'Skew (sqrt(T)·ΔIV/Δlnk)', 'skew_slope_q');     break;
    case 'term_slope': config = buildHistorical(data, true, 'Forward vol (annualized)', 'forward_vol');     break;
    default:           return;
  }

  const chart = new Chart(canvas, config);
  _instances.set(panelId, chart);
}

// ── Skew ──────────────────────────────────────────────────────────────────────

function buildSkew(data, pctFmt, yLabel) {
  const { series = [], band } = data;

  // X labels: put_delta values from the first series (all share the same grid)
  const labels = series[0]?.put_deltas?.map(pd => deltaLabel(pd)) ?? [];

  const datasets = [];

  // History band: outer (min-max) faint, inner (IQR) stronger, median dashed
  if (band) {
    // Outer max → fills down to outer min (next dataset)
    datasets.push({
      label: 'Band Max', data: band.pmax,
      borderColor: 'transparent', backgroundColor: 'rgba(255,255,255,0.04)',
      tension: 0.4, pointRadius: 0, fill: '+1', borderWidth: 0,
    });
    datasets.push({
      label: 'Band Min', data: band.pmin,
      borderColor: 'transparent', backgroundColor: 'transparent',
      tension: 0.4, pointRadius: 0, fill: false, borderWidth: 0,
    });
    // Inner p75 → fills down to p25 (next dataset)
    datasets.push({
      label: 'Band p75', data: band.p75,
      borderColor: 'transparent', backgroundColor: 'rgba(255,255,255,0.10)',
      tension: 0.4, pointRadius: 0, fill: '+1', borderWidth: 0,
    });
    datasets.push({
      label: 'Band p25', data: band.p25,
      borderColor: 'transparent', backgroundColor: 'transparent',
      tension: 0.4, pointRadius: 0, fill: false, borderWidth: 0,
    });
    // Median dashed centerline
    datasets.push({
      label: 'Band Median', data: band.p50,
      borderColor: 'rgba(255,255,255,0.45)', backgroundColor: 'transparent',
      tension: 0.4, pointRadius: 0, fill: false, borderWidth: 1, borderDash: [4, 3],
    });
  }

  // One line per series (DTE or date)
  series.forEach((s, i) => {
    const color = s.dte !== undefined ? dteColor(s.dte) : dateColor(i);
    datasets.push({
      label:       s.label,
      data:        s.values,
      borderColor: color,
      tension:     0.4,
      pointRadius: 3,
      pointBackgroundColor: color,
      fill:        false,
      borderWidth: 2.2,
      _metrics:    s.metrics,    // full metric set for tooltip
      _putDeltas:  s.put_deltas,
    });
  });

  const plugins = basePlugins();
  plugins.tooltip = {
    ...plugins.tooltip,
    mode: 'index',
    intersect: false,
    filter: (item) => !String(item.dataset.label || '').startsWith('Band'),
    callbacks: {
      title: (items) => {
        if (!items.length) return '';
        const ds = items[0].dataset;
        const pd = ds._putDeltas?.[items[0].dataIndex];
        return pd !== undefined ? `${ds.label} · ${deltaLabel(pd)}` : ds.label;
      },
      label: (ctx) => {
        const m = ctx.dataset._metrics?.[ctx.dataIndex];
        if (!m) return `${ctx.dataset.label}: ${ctx.parsed.y}`;
        const fmt = (v, p=2) => v == null ? '—' : v.toFixed(p);
        const ivPct = m.iv == null ? '—' : (m.iv * 100).toFixed(2) + '%';
        return [
          `${ctx.dataset.label}`,
          `  IV:    ${ivPct}`,
          `  Price: ${fmt(m.price)}`,
          `  Theta: ${fmt(m.theta, 3)}`,
          `  Vega:  ${fmt(m.vega, 3)}`,
          `  Gamma: ${fmt(m.gamma, 5)}`,
        ];
      },
    },
  };

  return {
    type: 'line',
    data: { labels, datasets },
    options: {
      responsive:          true,
      maintainAspectRatio: false,
      animation:           false,
      plugins,
      scales:              baseScales(yLabel, undefined, undefined, pctFmt),
    },
  };
}

// ── Term structure ────────────────────────────────────────────────────────────

function buildTerm(data, pctFmt, yLabel) {
  const { series = [], band } = data;

  // Convert a DTE to its sqrt-scale x position (sqrt(0)=0, sqrt(360)≈19)
  const xOf = (dte) => Math.sqrt(dte);
  const datasets = [];

  // History band: outer min/max faint, inner IQR stronger, dashed median
  if (band && band.dtes && band.dtes.length) {
    const bandPts = (arr) => band.dtes.map((d, i) => ({ x: xOf(d), y: arr[i] }));
    datasets.push({
      label: 'Band Max', data: bandPts(band.pmax),
      borderColor: 'transparent', backgroundColor: 'rgba(255,255,255,0.04)',
      tension: 0.4, pointRadius: 0, fill: '+1', borderWidth: 0,
    });
    datasets.push({
      label: 'Band Min', data: bandPts(band.pmin),
      borderColor: 'transparent', backgroundColor: 'transparent',
      tension: 0.4, pointRadius: 0, fill: false, borderWidth: 0,
    });
    datasets.push({
      label: 'Band p75', data: bandPts(band.p75),
      borderColor: 'transparent', backgroundColor: 'rgba(255,255,255,0.10)',
      tension: 0.4, pointRadius: 0, fill: '+1', borderWidth: 0,
    });
    datasets.push({
      label: 'Band p25', data: bandPts(band.p25),
      borderColor: 'transparent', backgroundColor: 'transparent',
      tension: 0.4, pointRadius: 0, fill: false, borderWidth: 0,
    });
    datasets.push({
      label: 'Band Median', data: bandPts(band.p50),
      borderColor: 'rgba(255,255,255,0.45)', backgroundColor: 'transparent',
      tension: 0.4, pointRadius: 0, fill: false, borderWidth: 1, borderDash: [4, 3],
    });
  }

  series.forEach((s, i) => {
    const color = s.delta !== undefined ? deltaColor(s.delta) : dateColor(i);
    const isDashed = (data.mode === 'by_date' && i > 0);
    const pts = (s.dtes || []).map((d, j) => ({ x: xOf(d), y: s.values[j] }));
    datasets.push({
      label:       s.label,
      data:        pts,
      borderColor: color,
      tension:     0.4,
      pointRadius: 3,
      pointBackgroundColor: color,
      fill:        false,
      borderWidth: 2.2,
      borderDash:  isDashed ? [4, 3] : [],
      _dtes:       s.dtes,
      _metrics:    s.metrics,
    });
  });

  // Tick label set: prefer the union of all series DTEs (already filtered client-side)
  const tickDtes = Array.from(new Set(
    series.flatMap(s => s.dtes || [])
  )).sort((a, b) => a - b);
  const tickValues = tickDtes.map(xOf);

  const plugins = basePlugins();
  plugins.tooltip = {
    ...plugins.tooltip,
    mode: 'index',
    intersect: false,
    filter: (item) => !String(item.dataset.label || '').startsWith('Band'),
    callbacks: {
      title: (items) => {
        if (!items.length) return '';
        const x = items[0].parsed.x;
        const dte = Math.round(x * x);
        return `${dte}D`;
      },
      label: (ctx) => {
        const m = ctx.dataset._metrics?.[ctx.dataIndex];
        if (!m) return `${ctx.dataset.label}: ${ctx.parsed.y}`;
        const fmt = (v, p=2) => v == null ? '—' : v.toFixed(p);
        const ivPct = m.iv == null ? '—' : (m.iv * 100).toFixed(2) + '%';
        return [
          `${ctx.dataset.label}`,
          `  IV:    ${ivPct}`,
          `  Price: ${fmt(m.price)}`,
          `  Theta: ${fmt(m.theta, 3)}`,
          `  Vega:  ${fmt(m.vega, 3)}`,
          `  Gamma: ${fmt(m.gamma, 5)}`,
        ];
      },
    },
  };

  return {
    type: 'line',
    data: { datasets },
    options: {
      responsive:          true,
      maintainAspectRatio: false,
      animation:           false,
      plugins,
      scales: {
        x: {
          type: 'linear',
          min: 0,
          max: xOf(360),
          grid:  { color: 'rgba(255,255,255,0.06)' },
          ticks: {
            font: { size: 10 },
            autoSkip: false,
            callback: (val) => {
              // Only label values that correspond to a real DTE in our set
              const dte = Math.round(val * val);
              return tickValues.some(tv => Math.abs(tv - val) < 1e-9)
                ? dte + 'D' : '';
            },
          },
          afterBuildTicks: (axis) => {
            axis.ticks = tickValues.map(v => ({ value: v }));
          },
        },
        y: {
          grid:  { color: 'rgba(255,255,255,0.06)' },
          ticks: { font: { size: 10 }, callback: v => pctFmt ? (v * 100).toFixed(2) + '%' : v },
          title: yLabel ? { display: true, text: yLabel, color: '#777', font: { size: 9 } } : { display: false },
        },
      },
    },
  };
}

// ── Historical time series ────────────────────────────────────────────────────

function buildHistorical(data, pctFmt, yLabel, flavor = 'historical') {
  const { series = [], dimension } = data;
  if (!series.length) return emptyConfig('No data');

  // All series share the same labels array (real labels, full strings)
  const labels = series[0].labels ?? [];
  const isRaw  = data.mode && data.mode.startsWith('raw_');

  const datasets = series.map((s, i) => {
    let color;
    if (isRaw)                  color = dateColor(i);
    else if (dimension === 'dte') color = dteColor(s.dte);
    else                          color = deltaColor(s.delta);
    return {
      label:       s.label,
      data:        s.values,
      borderColor: color,
      tension:     0.3,
      pointRadius: 0,
      fill:        false,
      borderWidth: 1.8,
      _metrics:    s.metrics,
    };
  });

  const plugins = basePlugins();
  plugins.tooltip = {
    ...plugins.tooltip,
    mode: 'index',
    intersect: false,
    callbacks: {
      title: (items) => items.length ? labels[items[0].dataIndex] || '' : '',
      label: (ctx) => {
        const m = ctx.dataset._metrics?.[ctx.dataIndex];
        const y = ctx.parsed.y;
        const fmt = (v, p=2) => v == null ? '—' : v.toFixed(p);
        if (flavor === 'convexity') {
          if (!m) return `${ctx.dataset.label}: ${y == null ? '—' : (y*100).toFixed(2)+'%'}`;
          const pct = (v) => v == null ? '—' : (v * 100).toFixed(2) + '%';
          return [
            `${ctx.dataset.label}`,
            `  Convexity: ${y == null ? '—' : (y*100).toFixed(2)+'%'}`,
            `  IV left:   ${pct(m.iv_left)}`,
            `  IV center: ${pct(m.iv_center)}`,
            `  IV right:  ${pct(m.iv_right)}`,
          ];
        }
        if (flavor === 'skew_slope_q') {
          const pct = (v) => v == null ? '—' : (v * 100).toFixed(2) + '%';
          if (!m) return `${ctx.dataset.label}: ${pct(y)}`;
          return [
            `${ctx.dataset.label}`,
            `  Skew: ${pct(y)}`,
            `  IV a: ${pct(m.iv_a)}    K a: ${m.k_a == null ? '—' : Math.round(m.k_a)}`,
            `  IV b: ${pct(m.iv_b)}    K b: ${m.k_b == null ? '—' : Math.round(m.k_b)}`,
          ];
        }
        if (flavor === 'forward_vol') {
          const pct = (v) => v == null ? '—' : (v * 100).toFixed(2) + '%';
          if (!m) return `${ctx.dataset.label}: ${pct(y)}`;
          return [
            `${ctx.dataset.label}`,
            `  Fwd vol: ${pct(y)}`,
            `  Fwd var: ${m.fwd_var == null ? '—' : m.fwd_var.toFixed(5)}`,
            `  IV a:    ${pct(m.iv_a)}`,
            `  IV b:    ${pct(m.iv_b)}`,
          ];
        }
        if (flavor === 'raw_historical') {
          const pct = (v) => v == null ? '—' : (v * 100).toFixed(2) + '%';
          if (!m) return `${ctx.dataset.label}: ${pct(y)}`;
          return [
            `${ctx.dataset.label}`,
            `  IV: ${pct(m.iv)}`,
            `  Delta: ${fmt(m.delta, 3)}`,
            `  Mid: ${fmt(m.mid_price)}`,
            `  Bid/Ask: ${fmt(m.bid)}/${fmt(m.ask)}`,
            `  Theta: ${fmt(m.theta, 3)}`,
            `  Vega:  ${fmt(m.vega, 3)}`,
          ];
        }
        // historical flavor
        if (!m) return `${ctx.dataset.label}: ${y}`;
        const ivPct = m.iv == null ? '—' : (m.iv * 100).toFixed(2) + '%';
        return [
          `${ctx.dataset.label}`,
          `  IV:    ${ivPct}`,
          `  Price: ${fmt(m.price)}`,
          `  Theta: ${fmt(m.theta, 3)}`,
          `  Vega:  ${fmt(m.vega, 3)}`,
          `  Gamma: ${fmt(m.gamma, 5)}`,
        ];
      },
    },
  };

  // For convexity show y-axis as percent (multiplied) too
  const yTickCb = (flavor === 'convexity')
    ? (v) => (v * 100).toFixed(2) + '%'
    : (v) => pctFmt ? (v * 100).toFixed(2) + '%' : v;

  return {
    type: 'line',
    data: { labels, datasets },
    options: {
      responsive:          true,
      maintainAspectRatio: false,
      animation:           false,
      plugins,
      scales: {
        x: {
          grid:  { color: 'rgba(255,255,255,0.06)' },
          ticks: { maxRotation: 0, font: { size: 10 }, autoSkip: true, maxTicksLimit: 8 },
        },
        y: {
          grid: { color: 'rgba(255,255,255,0.06)' },
          ticks: { font: { size: 10 }, callback: yTickCb },
          title: yLabel ? { display: true, text: yLabel, color: '#777', font: { size: 9 } } : { display: false },
        },
      },
    },
  };
}

// ── Raw skew (IV vs strike) ──────────────────────────────────────────────────

function buildRawSkew(data, pctFmt = true, yLabel = 'Implied Vol (%)') {
  const { series = [], x_axis } = data;
  if (!series.length) return emptyConfig('No data');

  const useMoney = (x_axis === 'moneyness');
  const datasets = [];

  series.forEach((s, i) => {
    const color = dateColor(i);
    const xArr  = useMoney ? s.moneyness : s.strikes;
    const pts   = xArr.map((x, j) => ({ x, y: s.values[j] }));
    datasets.push({
      label:       s.label,
      data:        pts,
      borderColor: color,
      tension:     0.3,
      pointRadius: 1.5,
      pointBackgroundColor: color,
      fill:        false,
      borderWidth: 2,
      _metrics:    s.metrics,
      _strikes:    s.strikes,
      _rights:     s.rights,
    });
  });

  const plugins = basePlugins();
  plugins.tooltip = {
    ...plugins.tooltip,
    mode: 'nearest',
    intersect: true,
    callbacks: {
      title: (items) => {
        if (!items.length) return '';
        const ds  = items[0].dataset;
        const idx = items[0].dataIndex;
        const k   = ds._strikes?.[idx];
        const r   = ds._rights?.[idx];
        return `${ds.label} · ${k} ${r}`;
      },
      label: (ctx) => {
        const m = ctx.dataset._metrics?.[ctx.dataIndex];
        if (!m) return '';
        const pct = v => v == null ? '—' : (v * 100).toFixed(2) + '%';
        const fmt = (v, p=2) => v == null ? '—' : v.toFixed(p);
        return [
          `  IV: ${pct(m.iv)}`,
          `  Delta: ${fmt(m.delta, 3)}`,
          `  Mid: ${fmt(m.mid_price)}`,
          `  Bid/Ask: ${fmt(m.bid)}/${fmt(m.ask)}`,
          `  Theta: ${fmt(m.theta, 3)}`,
          `  Vega: ${fmt(m.vega, 3)}`,
          `  Gamma: ${fmt(m.gamma, 5)}`,
        ];
      },
    },
  };

  // Underlying marker via annotation is optional; skip for v1
  const underlying = series[0]?.underlying;

  return {
    type: 'line',
    data: { datasets },
    options: {
      responsive: true, maintainAspectRatio: false, animation: false,
      plugins,
      scales: {
        x: {
          type:  'linear',
          grid:  { color: 'rgba(255,255,255,0.06)' },
          ticks: { font: { size: 10 }, maxRotation: 0 },
          title: { display: true,
                   text: useMoney ? 'Moneyness (K/S)' : 'Strike',
                   color: '#777', font: { size: 9 } },
        },
        y: {
          grid:  { color: 'rgba(255,255,255,0.06)' },
          ticks: { font: { size: 10 },
                   callback: v => pctFmt ? (v * 100).toFixed(2) + '%' : v },
          title: { display: true, text: yLabel,
                   color: '#777', font: { size: 9 } },
        },
      },
    },
  };
}

// ── Raw term structure (selected metric vs DTE) ──────────────────────────────

function buildRawTerm(data, pctFmt = true, yLabel = 'Implied Vol (%)') {
  const { series = [], expirations = [] } = data;
  if (!series.length) return emptyConfig('No data');

  const xOf = dte => Math.sqrt(dte);
  const datasets = [];

  series.forEach((s, i) => {
    const color = dateColor(i);
    const pts   = [];
    for (let j = 0; j < s.dtes.length; j++) {
      if (s.values[j] != null && s.dtes[j] != null) {
        pts.push({ x: xOf(s.dtes[j]), y: s.values[j] });
      }
    }
    datasets.push({
      label:       s.label,
      data:        pts,
      borderColor: color,
      tension:     0.4,
      pointRadius: 3,
      pointBackgroundColor: color,
      fill:        false,
      borderWidth: 2.2,
      _metrics:    s.metrics,
      _dtes:       s.dtes,
      _exps:       expirations,
    });
  });

  const allDtes = [...new Set(
    series.flatMap(s => s.dtes.filter(d => d != null))
  )].sort((a, b) => a - b);
  const tickValues = allDtes.map(xOf);

  const plugins = basePlugins();
  plugins.tooltip = {
    ...plugins.tooltip,
    mode: 'nearest',
    intersect: true,
    callbacks: {
      title: (items) => {
        if (!items.length) return '';
        const x   = items[0].parsed.x;
        const dte = Math.round(x * x);
        const ds  = items[0].dataset;
        const idx = items[0].dataIndex;
        const exp = ds._exps?.[idx];
        return `${ds.label} · ${dte}D${exp ? ' (' + exp + ')' : ''}`;
      },
      label: (ctx) => {
        const m = ctx.dataset._metrics?.[ctx.dataIndex];
        if (!m) return '';
        const pct = v => v == null ? '—' : (v * 100).toFixed(2) + '%';
        const fmt = (v, p=2) => v == null ? '—' : v.toFixed(p);
        return [
          `  IV: ${pct(m.iv)}`,
          `  Delta: ${fmt(m.delta, 3)}`,
          `  Mid: ${fmt(m.mid_price)}`,
          `  Theta: ${fmt(m.theta, 3)}`,
          `  Vega: ${fmt(m.vega, 3)}`,
        ];
      },
    },
  };

  return {
    type: 'line',
    data: { datasets },
    options: {
      responsive: true, maintainAspectRatio: false, animation: false,
      plugins,
      scales: {
        x: {
          type: 'linear',
          min: 0,
          grid: { color: 'rgba(255,255,255,0.06)' },
          ticks: {
            font: { size: 10 },
            callback: val => Math.round(val * val) + 'D',
          },
          afterBuildTicks: axis => {
            // Show ~8 evenly-spaced DTE labels (subset of all DTEs)
            const target = 8;
            const step   = Math.max(1, Math.ceil(allDtes.length / target));
            const subset = [];
            for (let i = 0; i < allDtes.length; i += step) subset.push(allDtes[i]);
            const last = allDtes[allDtes.length - 1];
            if (subset[subset.length - 1] !== last) subset.push(last);
            axis.ticks = subset.map(d => ({ value: xOf(d) }));
          },
        },
        y: {
          grid: { color: 'rgba(255,255,255,0.06)' },
          ticks: { font: { size: 10 },
                   callback: v => pctFmt ? (v * 100).toFixed(2) + '%' : v },
          title: { display: true, text: yLabel,
                   color: '#777', font: { size: 9 } },
        },
      },
    },
  };
}

// ── Helpers ───────────────────────────────────────────────────────────────────

function emptyConfig(msg = 'No data') {
  return {
    type: 'line',
    data: { labels: [msg], datasets: [] },
    options: { responsive: true, maintainAspectRatio: false, animation: false },
  };
}

function hexToRgba(hex, alpha) {
  const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
  if (!result) return `rgba(128,128,128,${alpha})`;
  return `rgba(${parseInt(result[1],16)},${parseInt(result[2],16)},${parseInt(result[3],16)},${alpha})`;
}

function deltaLabel(pd) {
  if (pd === 50) return 'ATM';
  if (pd < 50)   return `${pd}Δp`;
  return `${100 - pd}Δc`;
}
