/**
 * app.js — Alpine.js dashboard state.
 *
 * Global state manages: mode (daily/intraday), date, time, metric.
 * Each of the 4 panels has its own type + per-panel controls.
 * Panels are reloaded independently when their controls change.
 */

'use strict';

// ── Constants mirroring pipeline config ──────────────────────────────────────

const ALL_DTES    = [0,1,2,3,4,5,6,7,8,9,10,14,21,30,45,60,90,120,180,270,360];
const ALL_DELTAS  = [5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95];
const COMMON_DTES = [7,14,30,45,60,90,120,180];
const COMMON_DELTAS = [10,25,35,50,65,75,90];

const PANEL_TYPES = ['skew', 'term', 'historical', 'concavity'];

function deltaLabel(pd) {
  if (pd === 50) return 'ATM';
  if (pd < 50)   return `${pd}Δp`;
  return `${100 - pd}Δc`;
}

function offsetDate(dateStr, days) {
  const d = new Date(dateStr + 'T00:00:00');
  d.setDate(d.getDate() + days);
  return d.toISOString().split('T')[0];
}

// ── Default panel configs ─────────────────────────────────────────────────────

function defaultPanel(id) {
  const types = ['skew', 'term', 'historical', 'concavity'];
  const type  = types[id] ?? 'skew';

  return {
    id,
    type,
    loading: false,
    error:   null,
    stats:   null,
    legend:  [],

    // Skew-specific
    skewMode:   'by_dte',    // by_dte | by_date | intraday
    skewDtes:   [7, 30, 90],
    skewDates:  [],
    skewTimes:  [],
    skewDte:    30,          // used in by_date and intraday modes
    showBand:   true,

    // Term-specific
    termMode:    'by_delta',
    termDeltas:  [25, 50, 75],
    termDtes:    [],          // populated from API
    termDteMin:  0,
    termDteMax:  360,
    showTermBand: true,

    // Historical-specific
    histDte:       30,
    histDeltas:    [25, 50, 75],
    histLookback:  180,       // days from current date
    histFreq:      'daily',   // daily | intraday
    histStart:     null,      // set on load

    // Concavity-specific
    concavDte:         30,
    concavLeft:        25,
    concavCenter:      50,
    concavRight:       75,
    concavLookback:    180,
    concavFreq:        'daily',
  };
}

// ── Alpine component ──────────────────────────────────────────────────────────

document.addEventListener('alpine:init', () => {

  Alpine.data('dashboard', () => ({

    // ── Global state ────────────────────────────────────────────────────────
    mode:          'daily',      // 'daily' | 'intraday'
    date:          null,
    time:          '15:45',
    metric:        'iv',
    lookbackDays:  90,

    dates:         [],           // all available trade_dates
    times:         [],           // available quote_times for selected date

    // Intraday window (for historical/concavity panels in intraday mode)
    intradayStart: null,
    intradayEnd:   null,
    intradayWindowDays: 90,

    panels: [0,1,2,3].map(defaultPanel),

    COMMON_DTES,
    COMMON_DELTAS,
    ALL_DTES,
    ALL_DELTAS,

    // ── Initialise ──────────────────────────────────────────────────────────
    async init() {
      await this.loadDates();
      // panels load after dates/times are ready
    },

    async loadDates() {
      try {
        const res  = await fetch('/api/meta/dates');
        this.dates = await res.json();
        if (this.dates.length) {
          this.date = this.dates[0];
          await this.loadTimes();
          await this.loadAllPanels();
        }
      } catch(e) {
        console.error('Failed to load dates', e);
      }
    },

    async loadTimes() {
      if (!this.date) return;
      try {
        const res   = await fetch(`/api/meta/quote_times?date=${this.date}`);
        this.times  = await res.json();
        // Snap to closest available time to target
        if (!this.times.includes(this.time)) {
          this.time = this.closestTime(this.times, this.time);
        }
      } catch(e) {
        console.error('Failed to load times', e);
      }
    },

    closestTime(times, target) {
      if (!times.length) return target;
      const toMins = t => {
        const [h, m] = t.split(':').map(Number);
        return h * 60 + m;
      };
      const tm = toMins(target);
      return times.reduce((best, t) =>
        Math.abs(toMins(t) - tm) < Math.abs(toMins(best) - tm) ? t : best
      );
    },

    // ── Panel loading ────────────────────────────────────────────────────────
    async loadAllPanels() {
      await Promise.all(this.panels.map(p => this.loadPanel(p)));
    },

    async loadPanel(panel) {
      if (!this.date) return;
      panel.loading = true;
      panel.error   = null;
      try {
        const data = await this.fetchPanelData(panel);
        panel.stats  = data.stats  ?? null;
        panel.legend = this.buildLegend(panel.type, data);
        // Chart renders after Alpine reacts (canvas must be in DOM)
        this.$nextTick(() => renderPanel(panel.id, panel.type, data, this.metric));
      } catch(e) {
        panel.error = e.message ?? 'Request failed';
        console.error(`Panel ${panel.id} error`, e);
      }
      panel.loading = false;
    },

    async fetchPanelData(panel) {
      const p = new URLSearchParams({ metric: this.metric });

      switch(panel.type) {

        // ── Skew ─────────────────────────────────────────────────────────
        case 'skew': {
          if (panel.skewMode === 'by_dte') {
            p.set('date', this.date);
            p.set('time', this.time);
            p.set('dtes', panel.skewDtes.join(','));
            if (panel.showBand) {
              p.set('band_days', this.lookbackDays);
              p.set('band_time', this.time);
            }
            return this._get(`/api/skew/by_dte?${p}`);
          }
          if (panel.skewMode === 'by_date') {
            if (!panel.skewDates.length) panel.skewDates = this.dates.slice(0, 3);
            p.set('dates', panel.skewDates.join(','));
            p.set('times', this.time);
            p.set('dte',   panel.skewDte);
            return this._get(`/api/skew/by_date?${p}`);
          }
          // intraday
          p.set('date', this.date);
          p.set('dte',  panel.skewDte);
          if (panel.skewTimes.length) p.set('times', panel.skewTimes.join(','));
          return this._get(`/api/skew/intraday?${p}`);
        }

        // ── Term structure ────────────────────────────────────────────────
        case 'term': {
          if (panel.termMode === 'by_delta') {
            p.set('date',    this.date);
            p.set('time',    this.time);
            p.set('deltas',  panel.termDeltas.join(','));
            p.set('dte_min', panel.termDteMin);
            p.set('dte_max', panel.termDteMax);
            if (panel.showTermBand) {
              p.set('band_days', this.lookbackDays);
              p.set('band_time', this.time);
            }
            return this._get(`/api/term/by_delta?${p}`);
          }
          if (panel.termMode === 'by_date') {
            if (!panel.skewDates.length) panel.skewDates = this.dates.slice(0, 3);
            p.set('dates',   panel.skewDates.join(','));
            p.set('times',   this.time);
            p.set('delta',   panel.termDeltas[0] ?? 50);
            p.set('dte_min', panel.termDteMin);
            p.set('dte_max', panel.termDteMax);
            return this._get(`/api/term/by_date?${p}`);
          }
          // intraday
          p.set('date',    this.date);
          if (panel.skewTimes.length) p.set('times', panel.skewTimes.join(','));
          p.set('delta',   panel.termDeltas[0] ?? 50);
          p.set('dte_min', panel.termDteMin);
          p.set('dte_max', panel.termDteMax);
          return this._get(`/api/term/intraday?${p}`);
        }

        // ── Historical ────────────────────────────────────────────────────
        case 'historical': {
          const end   = this.date;
          const start = this.mode === 'intraday'
            ? (this.intradayStart ?? offsetDate(end, -this.intradayWindowDays))
            : offsetDate(end, -(panel.histLookback));
          p.set('dte',         panel.histDte);
          p.set('deltas',      panel.histDeltas.join(','));
          p.set('start',       start);
          p.set('end',         end);
          p.set('target_time', this.time);
          p.set('freq',        this.mode === 'intraday' ? 'intraday' : 'daily');
          return this._get(`/api/historical?${p}`);
        }

        // ── Concavity ─────────────────────────────────────────────────────
        case 'concavity': {
          const end   = this.date;
          const start = offsetDate(end, -(panel.concavLookback));
          p.set('dte',          panel.concavDte);
          p.set('left_delta',   panel.concavLeft);
          p.set('center_delta', panel.concavCenter);
          p.set('right_delta',  panel.concavRight);
          p.set('start',        start);
          p.set('end',          end);
          p.set('target_time',  this.time);
          p.set('freq',         this.mode === 'intraday' ? 'intraday' : 'daily');
          return this._get(`/api/concavity?${p}`);
        }

        default:
          throw new Error(`Unknown panel type: ${panel.type}`);
      }
    },

    async _get(url) {
      const res = await fetch(url);
      if (!res.ok) {
        const body = await res.text();
        throw new Error(`${res.status}: ${body}`);
      }
      return res.json();
    },

    // ── Legend builder ────────────────────────────────────────────────────────
    buildLegend(type, data) {
      const items = [];
      if (!data.series) return items;
      data.series.forEach((s, i) => {
        items.push({ label: s.label, color: this._seriesColor(type, s, i) });
      });
      if (data.band) {
        items.push({ label: `${this.lookbackDays}D Band`, band: true });
      }
      return items;
    },

    _seriesColor(type, s, i) {
      const DTE_C   = { 7:'#3498db',14:'#2ecc71',21:'#1abc9c',30:'#f0b429',45:'#e67e22',60:'#e74c3c',90:'#9b59b6',120:'#8e44ad',180:'#c0392b',270:'#d35400',360:'#7f8c8d' };
      const DELTA_C = { 10:'#e74c3c',25:'#f1c40f',35:'#aed581',50:'#3498db',65:'#5c6bc0',75:'#ab47bc',90:'#e53935' };
      const PAL     = ['#3498db','#2ecc71','#f39c12','#e74c3c','#9b59b6','#1abc9c','#e67e22'];
      if (s.dte   !== undefined) return DTE_C[s.dte]   ?? PAL[i % PAL.length];
      if (s.delta !== undefined) return DELTA_C[s.delta] ?? PAL[i % PAL.length];
      return PAL[i % PAL.length];
    },

    // ── Control helpers ───────────────────────────────────────────────────────
    toggleSkewDte(panel, dte) {
      const idx = panel.skewDtes.indexOf(dte);
      idx >= 0 ? panel.skewDtes.splice(idx, 1) : panel.skewDtes.push(dte);
      panel.skewDtes.sort((a, b) => a - b);
      this.loadPanel(panel);
    },

    toggleTermDelta(panel, delta) {
      const idx = panel.termDeltas.indexOf(delta);
      idx >= 0 ? panel.termDeltas.splice(idx, 1) : panel.termDeltas.push(delta);
      panel.termDeltas.sort((a, b) => a - b);
      this.loadPanel(panel);
    },

    toggleHistDelta(panel, delta) {
      const idx = panel.histDeltas.indexOf(delta);
      idx >= 0 ? panel.histDeltas.splice(idx, 1) : panel.histDeltas.push(delta);
      panel.histDeltas.sort((a, b) => a - b);
      this.loadPanel(panel);
    },

    setPanelType(panel, type) {
      destroyPanel(panel.id);
      Object.assign(panel, defaultPanel(panel.id), { type });
      this.loadPanel(panel);
    },

    // ── Global control handlers ───────────────────────────────────────────────
    async onDateChange() {
      await this.loadTimes();
      await this.loadAllPanels();
    },

    async onTimeChange() {
      await this.loadAllPanels();
    },

    async onModeChange(newMode) {
      this.mode = newMode;
      if (newMode === 'intraday') {
        this.intradayEnd   = this.date;
        this.intradayStart = offsetDate(this.date, -this.intradayWindowDays);
      }
      await this.loadAllPanels();
    },

    async onMetricChange(m) {
      this.metric = m;
      await this.loadAllPanels();
    },

    // ── Computed helpers ──────────────────────────────────────────────────────
    deltaLabel,

    get latestDate() { return this.dates[0] ?? '—'; },
    get dateDisplay() { return this.date ?? '—'; },
    get timeDisplay() { return this.time ?? '—'; },

  })); // end Alpine.data

}); // end alpine:init
