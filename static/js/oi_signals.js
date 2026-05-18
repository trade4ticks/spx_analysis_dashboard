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

    // ── Portfolios (Signals view) ────────────────────────────────────────
    portfolios: [],                  // [{id, name, ticker, outcome, monitored, system_count}, ...]
    portfolioFirings: [],            // [{type:"system", portfolio_id, system_id, ticker, all_stats, ticker_stats, ...}]
    portfolioFiringsLoading: false,
    monitoredExpanded: {},           // {portfolio_id: bool}

    // ─────────────────────────────────────────────────────────────────────
    async init() {
      await this._loadMeta();
      await Promise.all([this.loadTriggers(), this.loadCalendar(), this.loadPortfolios()]);
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
      // Fire portfolio firings in parallel — independent network call.
      await this.loadPortfolioFirings();
      this.firingLoading = false;
      await this._renderMiniCharts();
    },

    get sortedFiring() {
      // Unified firing list: legacy single-metric triggers + portfolio
      // system firings, each tagged with a `type` so the template can
      // dispatch to the right card markup. Sorted: strong systems first,
      // then triggers, then mixed/weak systems.
      const trigs = this.firingResults
        .filter(r => r.firing)
        .map(r => ({ ...r, type: 'trigger' }));
      const sys = (this.portfolioFirings || []).filter(r => r.firing);
      const verdictOrder = { strong: 0, mixed: 2, weak: 3 };
      const all = [...sys, ...trigs];
      all.sort((a, b) => {
        const av = a.type === 'trigger' ? 1 : (verdictOrder[a.verdict] ?? 2);
        const bv = b.type === 'trigger' ? 1 : (verdictOrder[b.verdict] ?? 2);
        return av - bv;
      });
      return all;
    },

    get monitoredCount() {
      return (this.portfolios || []).filter(p => p.monitored).length;
    },

    // ── Portfolio firings ─────────────────────────────────────────────────
    async loadPortfolios() {
      try {
        const r = await fetch('/api/oi-analysis/portfolios');
        if (r.ok) this.portfolios = await r.json();
      } catch (_) {}
    },

    async loadPortfolioFirings() {
      this.portfolioFiringsLoading = true;
      try {
        const qs = this.firingDate ? `?date=${this.firingDate}` : '';
        const r = await fetch('/api/oi-signals/firing-portfolios' + qs);
        if (r.ok) {
          const d = await r.json();
          this.portfolioFirings = d.results || [];
        } else {
          this.portfolioFirings = [];
        }
      } catch (_) { this.portfolioFirings = []; }
      this.portfolioFiringsLoading = false;
    },

    async toggleMonitored(pid, monitored) {
      try {
        await fetch(`/api/oi-analysis/portfolios/${pid}`, {
          method: 'PUT', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ monitored: !!monitored }),
        });
        const p = this.portfolios.find(x => x.id === pid);
        if (p) p.monitored = !!monitored;
        await this.loadPortfolioFirings();
      } catch (_) {}
    },

    // Group portfolio firings for the Monitored Portfolios section:
    // {portfolio_id: {portfolio_name, systems: {system_id: {system_name, tickers: [...]}}}}
    get firingsByPortfolio() {
      const out = {};
      for (const r of (this.portfolioFirings || [])) {
        if (!r.firing) continue;
        const pid = r.portfolio_id;
        if (!out[pid]) out[pid] = { name: r.portfolio_name, systems: {} };
        if (!out[pid].systems[r.system_id]) {
          out[pid].systems[r.system_id] = { name: r.system_name, tickers: [] };
        }
        out[pid].systems[r.system_id].tickers.push({
          ticker:  r.ticker,
          verdict: r.verdict,
        });
      }
      return out;
    },

    portFiringSystemsFor(pid) {
      return Object.values(this.firingsByPortfolio[pid]?.systems || {});
    },

    scrollToFiring(systemId, ticker) {
      const id = `card-system-${systemId}-${ticker}`;
      const el = document.getElementById(id);
      if (el && el.scrollIntoView) el.scrollIntoView({ behavior: 'smooth', block: 'center' });
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
      let currentTrigId = null;
      let popupEl       = null;
      const PW = 480, PH = 150;

      const show = (wrap) => {
        const id = parseInt(wrap.dataset.trigId);
        if (id === currentTrigId) return;
        currentTrigId = id;

        const result = this.firingResults.find(r => r.trigger.id === id);
        if (!result || !result.bins?.length) { hide(); return; }

        // Remove any previous popup
        if (popupEl) { popupEl.remove(); popupEl = null; }

        const rect = wrap.getBoundingClientRect();
        let left = rect.left;
        let top  = rect.bottom + 6;
        if (left + PW > window.innerWidth  - 8) left = window.innerWidth  - PW - 8;
        if (top  + PH + 32 > window.innerHeight) top  = rect.top - PH - 32;

        // Build popup from scratch each time — avoids any stale canvas state
        const div = document.createElement('div');
        div.style.cssText = [
          'position:fixed', `left:${left}px`, `top:${top}px`,
          'z-index:9999', 'pointer-events:none',
          'background:#1e1e1e', 'border:1px solid #444',
          'border-radius:6px', 'padding:8px 10px 10px',
          'box-shadow:0 8px 32px rgba(0,0,0,.75)',
        ].join(';');

        const titleEl = document.createElement('div');
        titleEl.style.cssText = 'font-size:9px;color:#666;margin-bottom:5px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;max-width:480px';
        titleEl.textContent = `${result.trigger.name}  ·  ${result.trigger.ticker}  ·  ${result.trigger.metric}  ·  ${result.trigger.outcome}`;
        div.appendChild(titleEl);

        const canvas = document.createElement('canvas');
        canvas.width        = PW;
        canvas.height       = PH;
        canvas.style.cssText = `display:block;width:${PW}px;height:${PH}px`;
        div.appendChild(canvas);

        document.body.appendChild(div);
        popupEl = div;

        console.log('[sparkHover] show id=', id, 'bins=', result.bins?.length, 'today_bin=', result.today_bin);
        _drawPopupChart(canvas, result, PW, PH);
      };

      const hide = () => {
        currentTrigId = null;
        if (popupEl) { popupEl.remove(); popupEl = null; }
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

// ── Popup chart: raw Canvas 2D (no Chart.js) ─────────────────────────────
function _drawPopupChart(canvas, result, w, h) {
  // Destroy any Chart.js instance that might be on this canvas from earlier
  const existing = Chart.getChart(canvas);
  if (existing) existing.destroy();

  canvas.width        = w;
  canvas.height       = h;
  canvas.style.width  = w + 'px';
  canvas.style.height = h + 'px';

  const ctx = canvas.getContext('2d');
  if (!ctx) { console.warn('[sparkHover] no canvas context'); return; }

  const bins     = result.bins || [];
  const todayBin = result.today_bin;
  console.log('[sparkHover] drawing', bins.length, 'bins, todayBin=', todayBin);

  // Debug: solid background to confirm canvas is reachable
  ctx.fillStyle = '#2a1a1a';
  ctx.fillRect(0, 0, w, h);

  const PL = 36, PR = 6, PT = 18, PB = 18;
  const cw = w - PL - PR;
  const ch = h - PT - PB;
  const bw = cw / bins.length;

  const vals = bins.map(b => b ? b.avg_ret : 0);
  const maxV = Math.max(0, ...vals);
  const minV = Math.min(0, ...vals);
  const range = (maxV - minV) || 1;
  const zeroY = PT + ch * maxV / range;

  ctx.clearRect(0, 0, w, h);

  // Subtle horizontal grid lines
  ctx.strokeStyle = 'rgba(255,255,255,0.06)';
  ctx.lineWidth   = 1;
  const nLines = 4;
  for (let i = 0; i <= nLines; i++) {
    const y = PT + (ch / nLines) * i;
    ctx.beginPath(); ctx.moveTo(PL, y); ctx.lineTo(w - PR, y); ctx.stroke();
  }

  // Zero line
  ctx.strokeStyle = '#555';
  ctx.lineWidth   = 1;
  ctx.beginPath(); ctx.moveTo(PL, zeroY); ctx.lineTo(w - PR, zeroY); ctx.stroke();

  // Y-axis labels
  ctx.fillStyle  = '#555';
  ctx.font       = '8px sans-serif';
  ctx.textAlign  = 'right';
  ctx.textBaseline = 'middle';
  [[maxV, PT], [0, zeroY], [minV, PT + ch]].forEach(([v, y]) => {
    ctx.fillText(v.toFixed(1) + '%', PL - 3, y);
  });

  // Bars
  for (let i = 0; i < bins.length; i++) {
    const b       = bins[i];
    const v       = b ? b.avg_ret : 0;
    const isToday = b && b.bin === todayBin;
    const pos     = v >= 0;
    const barH    = Math.abs(v) / range * ch;
    const x       = PL + i * bw + 1;
    const y       = pos ? zeroY - barH : zeroY;

    ctx.fillStyle = isToday
      ? (pos ? '#3498db' : '#e84393')
      : (pos ? 'rgba(52,152,219,0.38)' : 'rgba(232,67,147,0.38)');
    ctx.fillRect(x, y, bw - 2, Math.max(barH, 1));

    if (isToday) {
      ctx.strokeStyle = 'rgba(255,255,255,0.7)';
      ctx.lineWidth   = 1;
      ctx.strokeRect(x, y, bw - 2, Math.max(barH, 1));
    }
  }

  // Today dashed vertical line + label
  if (todayBin != null) {
    const tx = PL + (todayBin - 0.5) * bw;
    ctx.save();
    ctx.strokeStyle = 'rgba(255,255,255,0.5)';
    ctx.lineWidth   = 1.5;
    ctx.setLineDash([3, 2]);
    ctx.beginPath(); ctx.moveTo(tx, PT); ctx.lineTo(tx, PT + ch); ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle    = 'rgba(255,255,255,0.4)';
    ctx.font         = '8px sans-serif';
    ctx.textAlign    = 'center';
    ctx.textBaseline = 'bottom';
    ctx.fillText('today', tx, PT - 2);
    ctx.restore();
  }

  // X-axis bin labels every 4th bin
  ctx.fillStyle    = '#555';
  ctx.font         = '8px sans-serif';
  ctx.textAlign    = 'center';
  ctx.textBaseline = 'top';
  for (let i = 0; i < bins.length; i += 4) {
    ctx.fillText('B' + (i + 1), PL + i * bw + bw / 2, PT + ch + 3);
  }
}
