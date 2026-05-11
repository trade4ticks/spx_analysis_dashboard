'use strict';

const _SIG_PALETTE = [
  '#3498db','#e84393','#2ecc71','#f39c12',
  '#9b59b6','#1abc9c','#e74c3c','#f1c40f',
  '#16a085','#d35400','#8e44ad','#27ae60',
];

document.addEventListener('alpine:init', () => {
  Alpine.data('oiSignals', () => ({

    // ── Triggers ─────────────────────────────────────────────────────────
    triggers: [],
    showForm: false,
    formLoading: false,
    editForm: {
      id: null, name: '', ticker: '', metric: '', outcome: '',
      min_val: '', max_val: '', color: '#3498db',
    },

    // Dropdown options
    availTickers: [],
    availMetrics: [],
    availOutcomes: [],

    // ── Firing ────────────────────────────────────────────────────────────
    firingDate: '',
    firingResults: [],
    firingLoading: false,
    _miniCharts: {},

    // ── Calendar ─────────────────────────────────────────────────────────
    calEntries: [],
    calLoading: false,
    _ganttRange: { start: new Date(), end: new Date(), totalDays: 60 },

    // ── Per-trigger sparkline charts (bottom triggers table) ─────────────
    _trigMiniCharts: {},

    // ─────────────────────────────────────────────────────────────────────
    async init() {
      await this._loadMeta();
      await Promise.all([this.loadTriggers(), this.loadCalendar()]);
      await this.loadFiring();
      this._initSparkHover();
    },

    async _loadMeta() {
      try {
        const [tr, cr] = await Promise.all([
          fetch('/api/oi-analysis/tickers'),
          fetch('/api/oi-analysis/columns'),
        ]);
        if (tr.ok) this.availTickers = await tr.json();
        if (cr.ok) {
          const d = await cr.json();
          this.availMetrics  = d.features  || [];
          this.availOutcomes = d.outcomes  || [];
        }
      } catch (_) {}
    },

    // ── Triggers CRUD ────────────────────────────────────────────────────
    async loadTriggers() {
      try {
        const r = await fetch('/api/oi-signals/triggers');
        if (r.ok) this.triggers = await r.json();
      } catch (_) {}
    },

    openNewForm() {
      this.editForm = {
        id: null, name: '',
        ticker:  this.availTickers[0]  || '',
        metric:  this.availMetrics[0]  || '',
        outcome: this.availOutcomes[0] || '',
        min_val: '', max_val: '',
        color: _SIG_PALETTE[this.triggers.length % _SIG_PALETTE.length],
      };
      this.showForm = true;
    },

    editTrigger(t) {
      this.editForm = {
        id: t.id, name: t.name,
        ticker: t.ticker, metric: t.metric, outcome: t.outcome,
        min_val: t.min_val != null ? this._clean(t.min_val) : '',
        max_val: t.max_val != null ? this._clean(t.max_val) : '',
        color: t.color || '#3498db',
      };
      this.showForm = true;
    },

    async saveTrigger() {
      const f = this.editForm;
      if (!f.name || !f.ticker || !f.metric || !f.outcome) return;
      this.formLoading = true;
      try {
        const body = {
          name: f.name, ticker: f.ticker, metric: f.metric, outcome: f.outcome,
          min_val: f.min_val !== '' && f.min_val !== null ? parseFloat(f.min_val) : null,
          max_val: f.max_val !== '' && f.max_val !== null ? parseFloat(f.max_val) : null,
          color: f.color,
        };
        const url    = f.id ? `/api/oi-signals/triggers/${f.id}` : '/api/oi-signals/triggers';
        const method = f.id ? 'PUT' : 'POST';
        const r = await fetch(url, {
          method,
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(body),
        });
        if (r.ok) {
          this.showForm = false;
          await this.loadTriggers();
          await this.loadFiring();
        }
      } catch (_) {}
      this.formLoading = false;
    },

    async deleteTrigger(id) {
      if (!confirm('Delete this trigger? Its calendar entries will also be removed.')) return;
      await fetch(`/api/oi-signals/triggers/${id}`, { method: 'DELETE' });
      await this.loadTriggers();
      await this.loadFiring();
      await this.loadCalendar();
    },

    // Round-trip a stored float to its most likely "human" value. Older
    // rows were stored as REAL (single-precision) which loses precision
    // around 1.08 → 1.0800000429153442. Rounding to 7 significant digits
    // recovers the user's typed value in the common case without
    // mangling legitimately precise inputs.
    _clean(v) {
      if (v == null || v === '') return v;
      const n = Number(v);
      if (!Number.isFinite(n)) return v;
      return Number(n.toPrecision(7));
    },

    rangeText(t) {
      const lo = this._clean(t.min_val);
      const hi = this._clean(t.max_val);
      if (lo != null && hi != null) return `${lo} – ${hi}`;
      if (lo != null) return `≥ ${lo}`;
      if (hi != null) return `≤ ${hi}`;
      return 'any';
    },

    // Quick {trigger_id → firing result} lookup so the triggers-table row
    // can show its firing badge and reuse the same bin data for its
    // inline sparkline.
    get firingByTrigger() {
      const m = {};
      for (const r of this.firingResults) m[r.trigger?.id] = r;
      return m;
    },

    // ── Firing ────────────────────────────────────────────────────────────
    async loadFiring() {
      this.firingLoading = true;
      // Destroy existing mini charts (cards + trigger-row sparklines)
      for (const c of Object.values(this._miniCharts)) c.destroy();
      this._miniCharts = {};
      for (const c of Object.values(this._trigMiniCharts || {})) c.destroy();
      this._trigMiniCharts = {};
      try {
        const qs = this.firingDate ? `?date=${this.firingDate}` : '';
        const r = await fetch('/api/oi-signals/firing' + qs);
        if (r.ok) {
          const d = await r.json();
          this.firingResults = d.results || [];
          // Auto-populate date from first result with a current_date
          if (!this.firingDate) {
            const hit = this.firingResults.find(x => x.current_date);
            if (hit) this.firingDate = hit.current_date;
          }
        }
      } catch (_) {}
      this.firingLoading = false;
      await this._renderMiniCharts();
    },

    get sortedFiring() {
      // Today's Signals section only shows the cards for triggers that are
      // currently firing — idle ones still live in the triggers table at
      // the bottom with their own mini chart.
      return this.firingResults.filter(r => r.firing);
    },

    async _renderMiniCharts() {
      await this.$nextTick();
      setTimeout(() => {
        for (const r of this.firingResults) {
          if (!r.bins || !r.bins.length) continue;
          // Firing card mini chart (top section — only renders when r.firing
          // since the card itself is filtered).
          const card = document.getElementById('mini-' + r.trigger.id);
          if (card) {
            if (this._miniCharts[r.trigger.id]) this._miniCharts[r.trigger.id].destroy();
            this._miniCharts[r.trigger.id] = _makeMiniChart(card, r);
          }
          // Per-trigger row sparkline (bottom table — always renders for every
          // trigger, firing or idle).
          const row = document.getElementById('trig-mini-' + r.trigger.id);
          if (row) {
            if (this._trigMiniCharts[r.trigger.id]) this._trigMiniCharts[r.trigger.id].destroy();
            this._trigMiniCharts[r.trigger.id] = _makeMiniChart(row, r);
          }
        }
      }, 80);
    },

    // ── Calendar ─────────────────────────────────────────────────────────
    async addToCalendar(triggerId, entryDate) {
      if (!entryDate) return;
      try {
        const r = await fetch('/api/oi-signals/calendar', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ trigger_id: triggerId, entry_date: entryDate }),
        });
        if (r.ok) await this.loadCalendar();
      } catch (_) {}
    },

    async loadCalendar() {
      this.calLoading = true;
      try {
        const r = await fetch('/api/oi-signals/calendar');
        if (r.ok) {
          this.calEntries = await r.json();
          this._updateGanttRange();
        }
      } catch (_) {}
      this.calLoading = false;
    },

    async removeFromCalendar(id) {
      await fetch(`/api/oi-signals/calendar/${id}`, { method: 'DELETE' });
      this.calEntries = this.calEntries.filter(e => e.id !== id);
      this._updateGanttRange();
    },

    // ── Gantt helpers ─────────────────────────────────────────────────────
    _updateGanttRange() {
      const today = new Date();
      today.setHours(0, 0, 0, 0);
      let start = new Date(today);
      start.setDate(start.getDate() - 14);
      let end = new Date(today);
      end.setDate(end.getDate() + 45);

      for (const e of this.calEntries) {
        if (e.entry_date) {
          const d = new Date(e.entry_date + 'T00:00:00');
          if (d < start) { start = new Date(d); start.setDate(start.getDate() - 3); }
        }
        if (e.exit_date) {
          const d = new Date(e.exit_date + 'T00:00:00');
          if (d > end) { end = new Date(d); end.setDate(end.getDate() + 3); }
        }
      }

      const totalDays = Math.round((end - start) / 86400000) + 1;
      this._ganttRange = { start, end, totalDays };
    },

    barStyle(entry) {
      const { start, totalDays } = this._ganttRange;
      const s = new Date((entry.entry_date || '') + 'T00:00:00');
      const e = new Date((entry.exit_date  || entry.entry_date || '') + 'T00:00:00');
      const startOff = Math.max(0, (s - start) / 86400000);
      const endOff   = Math.min(totalDays, (e - start) / 86400000 + 1);
      const leftPct  = (startOff / totalDays) * 100;
      const widthPct = Math.max(0.8, (endOff - startOff) / totalDays * 100);
      const col = entry.color || '#3498db';
      return `left:${leftPct.toFixed(2)}%;width:${widthPct.toFixed(2)}%;background:${col}`;
    },

    get ganttTodayStyle() {
      const { start, totalDays } = this._ganttRange;
      const today = new Date();
      today.setHours(0, 0, 0, 0);
      const off = (today - start) / 86400000;
      const pct = (off / totalDays) * 100;
      if (pct < 0 || pct > 100) return 'display:none';
      return `left:${pct.toFixed(2)}%`;
    },

    get ganttHeaderTicks() {
      const { start, totalDays } = this._ganttRange;
      const ticks = [];
      const d = new Date(start);
      // Align to nearest upcoming Sunday
      const dow = d.getDay();
      if (dow !== 0) d.setDate(d.getDate() + (7 - dow));
      const rangeEnd = new Date(start.getTime() + totalDays * 86400000);
      while (d <= rangeEnd) {
        const off = (d - start) / 86400000;
        const pct = (off / totalDays) * 100;
        ticks.push({
          pct: pct.toFixed(2),
          label: d.toLocaleDateString('en-US', { month: 'short', day: 'numeric' }),
        });
        d.setDate(d.getDate() + 7);
      }
      return ticks;
    },

    // ── Sparkline hover popup ─────────────────────────────────────────────
    _initSparkHover() {
      const popup       = document.getElementById('trig-spark-popup');
      const popupCanvas = document.getElementById('trig-spark-popup-canvas');
      const popupTitle  = document.getElementById('trig-spark-popup-title');
      if (!popup || !popupCanvas) return;

      let popupChart    = null;
      let currentTrigId = null;   // track which trigger is showing to avoid thrash
      const PW = 480, PH = 150;

      const show = (wrap) => {
        const id = parseInt(wrap.dataset.trigId);
        if (id === currentTrigId) return;   // already showing this trigger — skip
        currentTrigId = id;

        const result = this.firingResults.find(r => r.trigger.id === id);
        if (!result || !result.bins?.length) {
          // No data for this trigger — hide any stale popup
          popup.style.display = 'none';
          return;
        }

        const rect = wrap.getBoundingClientRect();
        let left = rect.left;
        let top  = rect.bottom + 6;
        if (left + PW > window.innerWidth  - 8) left = window.innerWidth  - PW - 8;
        if (top  + PH + 32 > window.innerHeight) top  = rect.top - PH - 32;

        if (popupTitle) {
          popupTitle.textContent =
            `${result.trigger.name}  ·  ${result.trigger.ticker}  ·  ${result.trigger.metric}  ·  ${result.trigger.outcome}`;
        }

        popup.style.left    = left + 'px';
        popup.style.top     = top  + 'px';
        popup.style.display = 'block';

        if (popupChart) { popupChart.destroy(); popupChart = null; }
        popupChart = _makePopupChart(popupCanvas, result, PW, PH);
      };

      const hide = () => {
        currentTrigId = null;
        popup.style.display = 'none';
        if (popupChart) { popupChart.destroy(); popupChart = null; }
      };

      document.addEventListener('mouseover', (e) => {
        const wrap = e.target.closest('[data-trig-id]');
        if (wrap) show(wrap); else hide();
      }, { passive: true });

      document.addEventListener('mouseleave', hide, { passive: true });
    },
  }));
});

// ── Mini chart factory (module-level, not on Alpine component) ─────────────
function _makeMiniChart(el, result) {
  const bins     = result.bins || [];
  const todayBin = result.today_bin;

  const labels = bins.map((b, i) => b ? `B${b.bin}` : `B${i + 1}`);
  const data   = bins.map(b => b ? +b.avg_ret.toFixed(3) : 0);

  const bgColors = bins.map(b => {
    const isToday = b && b.bin === todayBin;
    const pos     = !b || b.avg_ret >= 0;
    if (isToday) return pos ? '#3498db' : '#e84393';
    return pos ? 'rgba(52,152,219,0.3)' : 'rgba(232,67,147,0.3)';
  });
  const borderColors = bins.map(b => b && b.bin === todayBin ? '#fff' : 'transparent');
  const borderWidths = bins.map(b => b && b.bin === todayBin ? 1.5 : 0);

  const todayPlugin = {
    id: 'todayLine',
    afterDraw(chart) {
      if (todayBin == null) return;
      const { ctx, scales: { x } } = chart;
      const px = x.getPixelForValue(todayBin - 1);
      ctx.save();
      ctx.strokeStyle = 'rgba(255,255,255,0.6)';
      ctx.lineWidth   = 1.5;
      ctx.setLineDash([2, 2]);
      ctx.beginPath();
      ctx.moveTo(px, chart.chartArea.top);
      ctx.lineTo(px, chart.chartArea.bottom);
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.restore();
    },
  };

  return new Chart(el, {
    type: 'bar',
    data: {
      labels,
      datasets: [{
        data,
        backgroundColor: bgColors,
        borderColor:      borderColors,
        borderWidth:      borderWidths,
        borderSkipped:    false,
      }],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            title:  items => `Bin ${bins[items[0].dataIndex]?.bin ?? '?'} of 20`,
            label:  item  => {
              const b = bins[item.dataIndex];
              if (!b) return [];
              return [
                `Avg ret: ${b.avg_ret.toFixed(2)}%`,
                `Win rate: ${b.win_rate.toFixed(0)}%`,
                `Range: ${b.min_val} – ${b.max_val}`,
                `n = ${b.n}`,
              ];
            },
          },
        },
      },
      scales: {
        x: {
          ticks:  { display: false },
          grid:   { display: false },
          border: { display: false },
        },
        y: {
          ticks:  { color: '#555', font: { size: 8 }, maxTicksLimit: 3 },
          grid:   { color: 'rgba(255,255,255,0.05)' },
          border: { display: false },
        },
      },
    },
    plugins: [todayPlugin],
  });
}

// ── Popup (expanded) chart factory ────────────────────────────────────────
function _makePopupChart(el, result, w, h) {
  // Destroy any lingering Chart.js instance on this canvas before resizing.
  const existing = Chart.getChart(el);
  if (existing) existing.destroy();

  el.width        = w;
  el.height       = h;
  el.style.width  = w + 'px';
  el.style.height = h + 'px';

  const bins     = result.bins || [];
  const todayBin = result.today_bin;
  const data     = bins.map(b => b ? +b.avg_ret.toFixed(3) : 0);

  const bgColors = bins.map(b => {
    const isToday = b && b.bin === todayBin;
    const pos     = !b || b.avg_ret >= 0;
    if (isToday) return pos ? '#3498db' : '#e84393';
    return pos ? 'rgba(52,152,219,0.35)' : 'rgba(232,67,147,0.35)';
  });
  const borderColors = bins.map(b => b && b.bin === todayBin ? '#fff' : 'transparent');
  const borderWidths = bins.map(b => b && b.bin === todayBin ? 2 : 0);

  const todayPlugin = {
    id: 'todayLinePopup',
    afterDraw(chart) {
      if (todayBin == null) return;
      const { ctx, scales: { x } } = chart;
      const px = x.getPixelForValue(todayBin - 1);
      ctx.save();
      ctx.strokeStyle = 'rgba(255,255,255,0.65)';
      ctx.lineWidth   = 2;
      ctx.setLineDash([4, 3]);
      ctx.beginPath();
      ctx.moveTo(px, chart.chartArea.top);
      ctx.lineTo(px, chart.chartArea.bottom);
      ctx.stroke();
      ctx.setLineDash([]);
      // "Today" label
      ctx.fillStyle    = 'rgba(255,255,255,0.5)';
      ctx.font         = '9px sans-serif';
      ctx.textAlign    = 'center';
      ctx.fillText('today', px, chart.chartArea.top - 4);
      ctx.restore();
    },
  };

  return new Chart(el, {
    type: 'bar',
    data: {
      labels: bins.map((b, i) => b ? `B${b.bin}` : `B${i + 1}`),
      datasets: [{
        data,
        backgroundColor: bgColors,
        borderColor:      borderColors,
        borderWidth:      borderWidths,
        borderSkipped:    false,
      }],
    },
    options: {
      responsive:          false,
      maintainAspectRatio: false,
      animation:           false,
      plugins: {
        legend:  { display: false },
        tooltip: { enabled: false },
      },
      scales: {
        x: {
          ticks:  { color: '#666', font: { size: 9 } },
          grid:   { display: false },
          border: { display: false },
        },
        y: {
          ticks:  { color: '#666', font: { size: 9 }, maxTicksLimit: 5 },
          grid:   { color: 'rgba(255,255,255,0.07)' },
          border: { display: false },
        },
      },
    },
    plugins: [todayPlugin],
  });
}
