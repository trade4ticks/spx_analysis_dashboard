/* Strategy Signal — is each configured strategy eligible to enter right now?
 *
 * The server decides (app/strategy_signal/evaluate.py) and sends each
 * strategy's state and chart series in ONE call, /board, remade every five
 * minutes. This file draws it and edits the configurations; it computes no
 * metric and no decision of its own.
 *
 * Charts live in SS_CHARTS, OUTSIDE the Alpine proxy: a Chart.js instance
 * wrapped in a reactive proxy is slow and breaks on update.
 */

const SS_BLUE = [0x34, 0x98, 0xdb];     // #3498db — favourable
const SS_PINK = [0xe8, 0x43, 0x93];     // #e84393 — unfavourable
const SS_LINE = '#d6d9dc';
const SS_REFRESH_MS = 5 * 60 * 1000;
const SS_WEEKDAYS = [{ n: 1, label: 'Mon' }, { n: 2, label: 'Tue' }, { n: 3, label: 'Wed' },
                     { n: 4, label: 'Thu' }, { n: 5, label: 'Fri' }];
const SS_CHARTS = {};

/* State i of n on the blue→pink scale. Two states are exactly the theme's
 * blue and pink; more are evenly spaced between them (through purple, never
 * a third hue). */
function ssStateColor(n, i) {
  const f = n <= 1 ? 0 : i / (n - 1);
  const c = SS_BLUE.map((b, k) => Math.round(b + (SS_PINK[k] - b) * f));
  return '#' + c.map(v => v.toString(16).padStart(2, '0')).join('');
}

/* A value for an axis tick or a tooltip, at a precision that suits its size:
 * a ratio near 1 wants four decimals, an index level none. */
function ssFmt(v) {
  if (v === null || v === undefined || !Number.isFinite(v)) return '—';
  const a = Math.abs(v);
  const dp = a >= 1000 ? 0 : a >= 100 ? 1 : a >= 10 ? 2 : a >= 1 ? 3 : 4;
  return v.toLocaleString('en-US', { minimumFractionDigits: dp, maximumFractionDigits: dp });
}

function ssUid() { return 'm' + Math.random().toString(36).slice(2, 8); }

document.addEventListener('alpine:init', () => {
  Alpine.data('strategySignal', () => ({
    board: null,
    loading: false,
    error: '',
    sources: null,
    sourcesError: '',
    editing: false,
    draft: null,          // kept after the panel closes; see the template
    saving: false,
    saveError: '',
    weekdays: SS_WEEKDAYS,
    cmps: ['<', '<=', '>', '>=', '='],

    init() {
      this.loadBoard();
      setInterval(() => { if (!this.editing) this.loadBoard(); }, SS_REFRESH_MS);
    },

    // ── the board ────────────────────────────────────────────────────────
    async loadBoard() {
      if (this.loading) return;
      this.loading = true;
      try {
        const r = await fetch('/api/strategy-signal/board');
        if (!r.ok) throw new Error(`board: HTTP ${r.status} ${(await r.text()).slice(0, 300)}`);
        this.board = await r.json();
        this.error = '';
      } catch (e) {
        this.error = String(e.message || e);
      } finally {
        this.loading = false;
      }
      this.$nextTick(() => this.drawAll());
    },

    entries() { return this.board ? this.board.strategies : []; },

    boardSub() {
      if (!this.board) return this.loading ? 'loading…' : '';
      return `${this.board.weekday} ${this.board.now} ET · refreshes every 5 min`;
    },

    stateIndex(e) { return e.decision.state; },
    stateLabel(e) {
      const i = e.decision.state;
      return i === null ? 'NO DATA' : e.strategy.states[i];
    },
    color(e) {
      const i = e.decision.state;
      return i === null ? 'transparent' : ssStateColor(e.strategy.states.length, i);
    },
    stateColor(n, i) { return ssStateColor(n, i); },

    /* Manual requirements are prominent whenever the automatic answer is to
     * enter at all — every state but the last. They never change it. */
    showManual(e) {
      const i = e.decision.state;
      return e.strategy.manual.length > 0 && i !== null && i < e.strategy.states.length - 1;
    },

    dayTag(s) {
      if (s.weekdays.length === 5) return '';
      return s.weekdays.map(d => SS_WEEKDAYS[d - 1].label).join('/');
    },

    /* The card's one line of identification: the entry days if restricted,
     * then the metrics that decide (or, with none, the ones charted). */
    idLine(e) {
      const s = e.strategy;
      const sig = s.metrics.filter(m => m.signal);
      const ms = (sig.length ? sig : s.metrics).map(m => this.metricInfo(e, m.id).label);
      return [this.dayTag(s), ms.join(' · ')].filter(Boolean).join(' · ') || 'entry days only';
    },

    reason(e) {
      const d = e.decision;
      if (d.reason === 'weekday') {
        return `Not an entry day — ${this.dayTag(e.strategy)} only`;
      }
      if (d.reason === 'no_data') {
        const miss = d.conditions.filter(c => c.value === null)
          .map(c => this.metricInfo(e, c.metric).label);
        return `No current value for ${miss.join(', ')}`;
      }
      if (d.reason === 'no_conditions') return 'No signal conditions — entry day only';
      return '';
    },

    asOf(e) {
      const used = new Set(e.strategy.metrics.map(m => m.id));
      const ts = e.metrics.filter(m => used.has(m.id) && m.as_of).map(m => m.as_of).sort();
      if (!ts.length) return '';
      return (e.stale ? '⚠ ' : '') + 'data through ' + ts[ts.length - 1];
    },

    metricInfo(e, id) {
      return e.metrics.find(m => m.id === id) || { label: id, error: null };
    },

    chartMetrics(e) { return e.strategy.metrics.filter(m => m.chart); },

    chartSub(m) {
      const lb = { '5d': '5 days', '10d': '10 days', '1m': '1 month', '3m': '3 months',
                   '6m': '6 months', '1y': '1 year', '2y': '2 years' }[m.lookback] || m.lookback;
      return `${m.resolution === 'intraday' ? '5-min' : 'daily'} · ${lb}`;
    },

    canvasId(e, m) { return `ss-c-${e.strategy.id}-${m.id}`; },

    scrollTo(e) {
      const el = document.getElementById('ss-sec-' + e.strategy.id);
      if (el) el.scrollIntoView({ behavior: 'smooth', block: 'start' });
    },

    async move(e, dir) {
      await fetch(`/api/strategy-signal/strategies/${e.strategy.id}/move`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ direction: dir }) });
      this.loadBoard();
    },

    // ── charts ───────────────────────────────────────────────────────────
    drawAll() {
      const live = new Set();
      for (const e of this.entries()) {
        for (const m of this.chartMetrics(e)) {
          const id = this.canvasId(e, m);
          live.add(id);
          this.draw(id, e, m);
        }
      }
      for (const id of Object.keys(SS_CHARTS)) {
        if (!live.has(id)) { SS_CHARTS[id].destroy(); delete SS_CHARTS[id]; }
      }
    },

    draw(id, e, m) {
      const el = document.getElementById(id);
      if (!el) return;
      if (SS_CHARTS[id] && SS_CHARTS[id].canvas !== el) { SS_CHARTS[id].destroy(); delete SS_CHARTS[id]; }
      const data = e.charts[m.id] || { t: [], v: [] };
      const t = data.t;
      const intraday = m.resolution === 'intraday';
      const pts = data.v.map((v, i) => ({ x: i, y: v }));
      const n = e.strategy.states.length;
      const datasets = [{ label: this.metricInfo(e, m.id).label, data: pts, borderColor: SS_LINE,
                          borderWidth: 1.6, pointRadius: 0, pointHoverRadius: 4, tension: 0,
                          spanGaps: false }];
      // A decision threshold is a flat line in the colour of the state it
      // leads to: "below this, FULL". Visualisation-only metrics draw none.
      if (m.signal) {
        m.thresholds.forEach((th, i) => {
          datasets.push({ label: `${m.cmp} ${ssFmt(th)} → ${e.strategy.states[i]}`,
                          data: [{ x: 0, y: th }, { x: Math.max(t.length - 1, 1), y: th }],
                          borderColor: ssStateColor(n, i), borderWidth: 1.4, borderDash: [5, 4],
                          pointRadius: 0, pointHoverRadius: 0, isThreshold: true });
        });
      }
      // Intraday ticks sit on each session's first bar, so an overnight gap
      // is a tick, not a stretch of empty axis.
      const dayStarts = [];
      if (intraday) {
        t.forEach((s, i) => { if (i === 0 || s.slice(0, 10) !== t[i - 1].slice(0, 10)) dayStarts.push(i); });
      }
      const short = s => (s ? s.slice(5, 10) : '');
      const opts = {
        animation: false, responsive: true, maintainAspectRatio: false,
        interaction: { mode: 'nearest', axis: 'x', intersect: false },
        plugins: {
          legend: { display: !!m.signal, position: 'top', align: 'end',
                    labels: { color: '#c8c8c8', boxWidth: 14, boxHeight: 1, font: { size: 10 },
                              filter: item => item.datasetIndex > 0 } },
          tooltip: {
            filter: item => item.datasetIndex === 0,
            callbacks: {
              title: items => (items.length ? t[items[0].parsed.x] || '' : ''),
              label: item => `${item.dataset.label}: ${ssFmt(item.parsed.y)}`,
            },
          },
        },
        scales: {
          x: {
            type: 'linear', min: 0, max: Math.max(t.length - 1, 1),
            grid: { color: 'rgba(255,255,255,0.05)' },
            ticks: {
              color: '#9a9a9a', font: { size: 10 }, maxRotation: 0, autoSkip: !intraday,
              maxTicksLimit: intraday ? undefined : 8,
              callback: v => {
                const s = t[Math.round(v)];
                if (!s) return '';
                return (!intraday && t.length > 300) ? s.slice(2, 7) : short(s);
              },
            },
            afterBuildTicks: axis => {
              if (!intraday) return;
              const step = Math.max(1, Math.ceil(dayStarts.length / 12));
              axis.ticks = dayStarts.filter((_, k) => k % step === 0).map(v => ({ value: v }));
            },
          },
          y: {
            grid: { color: 'rgba(255,255,255,0.05)' },
            ticks: { color: '#9a9a9a', font: { size: 10 }, maxTicksLimit: 6, callback: v => ssFmt(v) },
          },
        },
      };
      if (SS_CHARTS[id]) {
        SS_CHARTS[id].data.datasets = datasets;
        SS_CHARTS[id].options = opts;
        SS_CHARTS[id].update('none');
      } else {
        SS_CHARTS[id] = new Chart(el.getContext('2d'), { type: 'line', data: { datasets }, options: opts });
      }
    },

    // ── configuration ────────────────────────────────────────────────────
    async ensureSources() {
      if (this.sources) return;
      this.sourcesError = '';
      try {
        const r = await fetch('/api/strategy-signal/sources');
        if (!r.ok) throw new Error(`metric catalog: HTTP ${r.status} ${(await r.text()).slice(0, 300)}`);
        this.sources = await r.json();
        this.cmps = this.sources.cmps;
      } catch (e) {
        this.sourcesError = String(e.message || e);
      }
    },

    /* Every source, for the datalist: the id is what the input holds, the
     * text is what the browser shows beside it while you type. */
    sourceOptions() {
      if (!this.sources) return [];
      const idx = this.sources.index.map(s => ({ id: s.id, text: `${s.label} — ${s.description}` }));
      const surf = this.sources.surface.map(s => ({
        id: s.id, text: `${s.group} · ${s.form_label}${s.description ? ' — ' + s.description : ''}` }));
      return idx.concat(surf);
    },

    sourceById(id) {
      if (!this.sources || !id) return null;
      return this.sources.index.find(s => s.id === id) || this.sources.surface.find(s => s.id === id) || null;
    },

    srcShort(id) {
      const s = this.sourceById(id);
      if (!s) return id || '';
      return s.label || s.column;
    },

    autoLabel(m) {
      if (!m.a) return '';
      let s = this.srcShort(m.a);
      if (m.op) s += ` ${m.op} ${this.srcShort(m.b)}`;
      if (m.transform) s += ' · pctl 252';
      return s;
    },

    metricHint(m) {
      const parts = [];
      for (const id of [m.a, m.op ? m.b : null]) {
        if (!id) continue;
        const s = this.sourceById(id);
        if (!s) { parts.push(this.sources ? `"${id}" is not an available metric` : ''); continue; }
        if (s.column) {
          parts.push(`${s.column}: ${s.description || s.form_label}` +
                     (s.units ? ` [${s.units}]` : '') + (s.min_date ? `, from ${s.min_date}` : ''));
        } else {
          parts.push(`${s.label}: ${s.description} (index_ohlc close)`);
        }
      }
      if (m.transform) parts.push('percentile of the value among the previous 252 session closes, 0–100');
      return parts.filter(Boolean).join(' · ');
    },

    logicHint() {
      if (!this.draft) return '';
      const days = this.draft.weekdays.length === 5 ? 'an entry day' : 'one of the ticked days';
      return this.draft.logic === 'and'
        ? `Trade needs ${days} AND every signal condition passing.`
        : `Trade needs ${days} AND at least one signal condition passing.`;
    },

    lookbacksFor(m) {
      const all = this.sources ? this.sources.lookbacks : [];
      return m.resolution === 'intraday' ? all.filter(l => l.intraday) : all;
    },

    fixLookback(m) {
      const ok = this.lookbacksFor(m).map(l => l.id);
      if (ok.length && !ok.includes(m.lookback)) m.lookback = m.resolution === 'intraday' ? '10d' : '1y';
    },

    blankMetric() {
      return { id: ssUid(), label: '', a: '', op: '', b: '', transform: '', chart: true,
               resolution: 'daily', lookback: '1y', signal: false, cmp: '<',
               thresholds: Array(this.draft.states.length - 1).fill('') };
    },

    newStrategy() {
      this.draft = { id: null, name: '', notes: '', weekdays: [1, 2, 3, 4, 5],
                     states: ['TRADE', 'NO TRADE'], logic: 'and', metrics: [], manual: [] };
      this.draft.metrics.push(this.blankMetric());
      this.openPanel();
    },

    editStrategy(e) {
      const s = JSON.parse(JSON.stringify(e.strategy));
      s.metrics.forEach(m => {
        m.op = m.op || ''; m.b = m.b || ''; m.transform = m.transform || ''; m.cmp = m.cmp || '<';
        if (!m.signal) m.thresholds = Array(s.states.length - 1).fill('');
      });
      this.draft = s;
      this.openPanel();
    },

    openPanel() {
      this.editing = true;
      this.saveError = '';
      this.ensureSources();
      this.$nextTick(() => { if (this.$refs.cfg) this.$refs.cfg.scrollIntoView({ behavior: 'smooth' }); });
    },

    cancelEdit() { this.editing = false; this.saveError = ''; this.$nextTick(() => this.drawAll()); },

    toggleDay(n) {
      const w = this.draft.weekdays;
      this.draft.weekdays = w.includes(n) ? w.filter(d => d !== n) : [...w, n].sort();
    },

    // States and thresholds move together: each signal metric keeps one
    // threshold per state but the last, so resizing one resizes the other.
    syncThresholds() {
      const k = this.draft.states.length - 1;
      for (const m of this.draft.metrics) {
        const t = m.thresholds.slice(0, k);
        while (t.length < k) t.push('');
        m.thresholds = t;
      }
    },
    addState() {
      const s = this.draft.states;
      s.splice(s.length - 1, 0, `LEVEL ${s.length}`);
      this.syncThresholds();
    },
    removeState(i) {
      this.draft.states.splice(i, 1);
      if (i < this.draft.states.length) this.draft.metrics.forEach(m => m.thresholds.splice(i, 1));
      this.syncThresholds();
    },
    moveState(i, dir) {
      const s = this.draft.states, j = i + dir;
      if (j < 0 || j >= s.length) return;
      [s[i], s[j]] = [s[j], s[i]];
    },
    presetStates(list) {
      this.draft.states = [...list];
      this.syncThresholds();
    },

    addMetric() { this.draft.metrics.push(this.blankMetric()); },

    payload() {
      const d = this.draft;
      return {
        name: d.name, notes: d.notes, weekdays: d.weekdays, states: d.states, logic: d.logic,
        manual: d.manual,
        metrics: d.metrics.map(m => ({
          id: m.id, label: m.label, a: (m.a || '').trim(), op: m.op || null,
          b: m.op ? (m.b || '').trim() : null, transform: m.transform || null,
          chart: m.chart, resolution: m.resolution, lookback: m.lookback,
          signal: m.signal, cmp: m.signal ? m.cmp : null,
          thresholds: m.signal ? m.thresholds.map(v => (String(v).trim() === '' ? null : Number(v))) : [],
        })),
      };
    },

    async save() {
      this.saving = true;
      this.saveError = '';
      try {
        const id = this.draft.id;
        const r = await fetch(id ? `/api/strategy-signal/strategies/${id}` : '/api/strategy-signal/strategies', {
          method: id ? 'PUT' : 'POST', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(this.payload()) });
        if (!r.ok) {
          const body = await r.json().catch(() => ({}));
          throw new Error(body.detail || `HTTP ${r.status}`);
        }
        this.editing = false;
        await this.loadBoard();
      } catch (e) {
        this.saveError = String(e.message || e);
      } finally {
        this.saving = false;
      }
    },

    async remove() {
      if (!this.editing || !this.draft.id) return;
      if (!confirm(`Delete "${this.draft.name}"? This cannot be undone.`)) return;
      this.saving = true;
      try {
        const r = await fetch(`/api/strategy-signal/strategies/${this.draft.id}`, { method: 'DELETE' });
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        this.editing = false;
        await this.loadBoard();
      } catch (e) {
        this.saveError = String(e.message || e);
      } finally {
        this.saving = false;
      }
    },
  }));
});
