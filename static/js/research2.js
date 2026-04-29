document.addEventListener('alpine:init', () => {
  Alpine.data('research2', () => ({

    // ── State ──────────────────────────────────────────────────────────────
    runs:        [],
    selectedRun: null,
    charts:      [],
    results:     [],
    view:        'list',   // 'list' | 'new' | 'run'
    loading:     false,
    error:       null,
    pollTimer:   null,
    lightboxUrl: null,
    _chartInstances: {},

    followups:        [],
    followupQuestion: '',
    followupLoading:  false,
    followupError:    null,

    // Knowledge library
    knowledgeRules:        [],
    newKnowledgeCategory:  'policy',
    newKnowledgeText:      '',
    editingKnowledgeId:    null,
    editingKnowledgeText:  '',

    form: {
      name:      '',
      question:  '',
      table:     'daily_features',
      tickers:   '',
      date_from: '',
      date_to:   '',
      model:     'claude-sonnet-4-6',
    },
    submitting:  false,
    submitError: null,
    availableTickers: [],

    // ── Init ───────────────────────────────────────────────────────────────
    async init() {
      await this.loadRuns();
      this._loadTickers();
      this._loadKnowledge();
    },

    async _loadTickers() {
      try {
        const r = await fetch('/api/research2/tickers');
        if (r.ok) this.availableTickers = await r.json();
      } catch (_) {}
    },

    // ── Run list ───────────────────────────────────────────────────────────
    async loadRuns() {
      try {
        const r = await fetch('/api/research2/runs');
        this.runs = r.ok ? await r.json() : [];
      } catch (_) { this.runs = []; }
    },

    async selectRun(run) {
      this.view        = 'run';
      this.selectedRun = run;
      this.charts      = [];
      this.results     = [];
      this.error       = null;
      this._destroyCharts();
      this._stopPoll();
      await this._loadRunDetail(run.id);
      if (run.status === 'running') this._startPoll(run.id);
    },

    _destroyCharts() {
      Object.values(this._chartInstances).forEach(c => c.destroy());
      this._chartInstances = {};
    },

    async _loadRunDetail(runId) {
      this.loading = true;
      this._destroyCharts();
      try {
        const [runRes, chartsRes, resultsRes, fupsRes] = await Promise.all([
          fetch(`/api/research2/run/${runId}`),
          fetch(`/api/research2/run/${runId}/charts`),
          fetch(`/api/research2/run/${runId}/results`),
          fetch(`/api/research2/run/${runId}/followups`),
        ]);
        if (runRes.ok)     this.selectedRun = await runRes.json();
        if (chartsRes.ok)  this.charts      = await chartsRes.json();
        if (resultsRes.ok) this.results     = await resultsRes.json();
        if (fupsRes.ok)    this.followups   = await fupsRes.json();

        // Render interactive charts after DOM update
        await this.$nextTick();
        this._renderInteractiveCharts();
      } catch (e) {
        this.error = e.message;
      } finally {
        this.loading = false;
      }
    },

    // Computed getters for scan results
    get scans() {
      return this.results.filter(r => r.analysis_type === 'scan' && !r.result?.error)
        .sort((a, b) => (b.result?.composite_score || 0) - (a.result?.composite_score || 0));
    },
    get equityCurveResults() {
      return this.results.filter(r => r.analysis_type?.startsWith('equity_curve'));
    },

    _renderInteractiveCharts() {
      const darkScales = {
        x: { ticks: {color:'#888',font:{size:9},maxRotation:45}, grid: {color:'rgba(255,255,255,0.05)'}, border: {color:'transparent'} },
        y: { ticks: {color:'#888',font:{size:9}}, grid: {color:'rgba(255,255,255,0.05)'}, border: {color:'transparent'} },
      };

      // Decile/bucket profile charts for top scans
      for (const r of this.scans.slice(0, 6)) {
        const bs = (r.result?.bucket_stats || []).filter(b => b != null);
        if (bs.length < 3) continue;
        const el = document.getElementById('r2-decile-' + r.id);
        if (!el) continue;

        const avgs = bs.map(b => (b.avg_ret || 0) * 100);
        const winRates = bs.map(b => (b.win_rate || 0.5) * 100);
        this._chartInstances['r2-decile-' + r.id] = new Chart(el, {
          type: 'bar',
          data: {
            labels: bs.map(b => 'D' + b.bucket),
            datasets: [{
              label: 'Avg Return %',
              data: avgs,
              backgroundColor: avgs.map(v => v >= 0 ? '#3498db' : '#e84393'),
              borderWidth: 0,
            }],
          },
          options: {
            responsive: true, maintainAspectRatio: false, animation: false,
            plugins: {
              legend: { display: false },
              tooltip: {
                backgroundColor: 'rgba(20,20,20,0.92)', borderColor: '#444', borderWidth: 1,
                callbacks: {
                  afterLabel: (ctx) => {
                    const b = bs[ctx.dataIndex];
                    return b ? `WR: ${(b.win_rate*100).toFixed(1)}%  Sharpe: ${(b.sharpe||0).toFixed(3)}  n: ${b.n}` : '';
                  },
                },
              },
            },
            scales: darkScales,
          },
        });
      }

      // Equity curves
      const topEqs = this.results.filter(r => r.analysis_type === 'equity_curve_top');
      for (const r of topEqs.slice(0, 4)) {
        const el = document.getElementById('r2-equity-' + r.id);
        if (!el || !r.result?.points) continue;
        const pts = r.result.points;
        const datasets = [{
          label: r.result.which === 'top2' ? 'Top 2 (D9-D10)' : 'Top (D10)',
          data: pts.map(p => p.value),
          borderColor: '#3498db', backgroundColor: 'transparent',
          borderWidth: 2, pointRadius: 0, tension: 0.1,
        }];
        const bot = this.results.find(x =>
          x.analysis_type === 'equity_curve_bottom' &&
          x.ticker === r.ticker && x.x_col === r.x_col && x.y_col === r.y_col);
        if (bot?.result?.points) {
          datasets.push({
            label: bot.result.which === 'bottom2' ? 'Bottom 2 (D1-D2)' : 'Bottom (D1)',
            data: bot.result.points.map(p => p.value),
            borderColor: '#e84393', backgroundColor: 'transparent',
            borderWidth: 2, pointRadius: 0, tension: 0.1,
          });
        }
        this._chartInstances['r2-equity-' + r.id] = new Chart(el, {
          type: 'line',
          data: {
            labels: pts.map(p => p.date?.slice(0, 7) || ''),
            datasets,
          },
          options: {
            responsive: true, maintainAspectRatio: false, animation: false,
            plugins: {
              legend: { labels: { color: '#aaa', font: { size: 10 } } },
              tooltip: { backgroundColor: 'rgba(20,20,20,0.92)', borderColor: '#444', borderWidth: 1 },
            },
            scales: {
              ...darkScales,
              x: { ...darkScales.x, ticks: { ...darkScales.x.ticks, maxTicksLimit: 10 } },
            },
          },
        });
      }
    },

    // ── Polling ────────────────────────────────────────────────────────────
    _startPoll(runId) {
      this.pollTimer = setInterval(async () => {
        try {
          const r = await fetch(`/api/research2/run/${runId}`);
          if (!r.ok) return;
          const data = await r.json();
          this.selectedRun = data;
          const idx = this.runs.findIndex(x => x.id === runId);
          if (idx >= 0) this.runs[idx] = { ...this.runs[idx], status: data.status };
          if (data.status !== 'running') {
            this._stopPoll();
            await this._loadRunDetail(runId);
          }
        } catch (_) {}
      }, 3000);
    },

    _stopPoll() {
      if (this.pollTimer) { clearInterval(this.pollTimer); this.pollTimer = null; }
    },

    // ── New run form ───────────────────────────────────────────────────────
    newRun() {
      this._stopPoll();
      this.view             = 'new';
      this.selectedRun      = null;
      this.submitError      = null;
      this.followups        = [];
      this.followupQuestion = '';
      this.followupError    = null;
      this.form = {
        name: '', question: '', table: 'daily_features',
        tickers: '', date_from: '', date_to: '',
        model: 'claude-sonnet-4-6',
      };
    },

    setTable(t) { this.form.table = t; },

    addTicker(t) {
      const ex = this.form.tickers.split(',').map(s => s.trim()).filter(Boolean);
      if (!ex.includes(t)) this.form.tickers = [...ex, t].join(', ');
    },

    async submitRun() {
      this.submitError = null;
      if (!this.form.name.trim() || !this.form.question.trim()) {
        this.submitError = 'Name and research question are required.';
        return;
      }
      const parse = s => s.split(',').map(x => x.trim()).filter(Boolean);
      const body = {
        name:      this.form.name.trim(),
        question:  this.form.question.trim(),
        table:     this.form.table,
        tickers:   parse(this.form.tickers),
        date_from: this.form.date_from || null,
        date_to:   this.form.date_to   || null,
        model:     this.form.model,
      };
      this.submitting = true;
      try {
        const r = await fetch('/api/research2/run', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(body),
        });
        if (!r.ok) {
          const err = await r.json().catch(() => ({}));
          throw new Error(err.detail || `HTTP ${r.status}`);
        }
        const data = await r.json();
        await this.loadRuns();
        const run = this.runs.find(x => x.id === data.run_id)
                 || { id: data.run_id, name: body.name, status: 'running',
                      config: { workflow_plan: data.plan } };
        await this.selectRun(run);
      } catch (e) {
        this.submitError = e.message;
      } finally {
        this.submitting = false;
      }
    },

    // ── PDF export ────────────────────────────────────────────────────────
    exportPdf(runId) {
      window.location.href = `/api/research/run/${runId}/pdf`;
    },

    // ── Follow-up ──────────────────────────────────────────────────────────
    async askFollowup() {
      const q = this.followupQuestion.trim();
      if (!q || this.followupLoading) return;
      this.followupError   = null;
      this.followupLoading = true;
      try {
        const r = await fetch(`/api/research2/run/${this.selectedRun.id}/followup`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ question: q }),
        });
        if (!r.ok) {
          const err = await r.json().catch(() => ({}));
          throw new Error(err.detail || `HTTP ${r.status}`);
        }
        const saved = await r.json();
        this.followups.push(saved);
        this.followupQuestion = '';
      } catch (e) {
        this.followupError = e.message;
      } finally {
        this.followupLoading = false;
        await this.$nextTick();
        const el = document.getElementById('r2-followup-bottom');
        if (el) el.scrollIntoView({ behavior: 'smooth' });
      }
    },

    // ── Delete ─────────────────────────────────────────────────────────────
    async deleteRun(runId) {
      if (!confirm('Delete this investigation and all its results?')) return;
      await fetch(`/api/research2/run/${runId}`, { method: 'DELETE' });
      if (this.selectedRun?.id === runId) { this.view = 'list'; this.selectedRun = null; }
      await this.loadRuns();
    },

    // ── Lightbox ───────────────────────────────────────────────────────────
    openLightbox(url)  { this.lightboxUrl = url; },
    closeLightbox()    { this.lightboxUrl = null; },
    chartUrl(chartId)  { return `/api/research/chart/${chartId}.png`; },

    // ── Computed ───────────────────────────────────────────────────────────
    get plan() {
      return this.selectedRun?.config?.workflow_plan || null;
    },

    get isRunning() {
      return this.selectedRun?.status === 'running';
    },

    get progressStep() {
      const s = this.selectedRun?.ai_summary || '';
      if (!s.startsWith('[RUNNING]')) return '';
      return s.replace('[RUNNING]', '').trim();
    },

    get reportText() {
      const s = this.selectedRun?.ai_summary || '';
      if (s.startsWith('[RUNNING]') || !s) return '';
      return s;
    },

    get taskTypeBadgeColor() {
      const colors = {
        'single-factor-scan':        '#1a3a5c',
        'multi-factor-interaction':  '#1a3a2a',
        'regime-analysis':           '#3a2a1a',
        'event-study':               '#2a1a3a',
        'backtest-pnl-attribution':  '#1a2a3a',
        'strategy-entry-condition':  '#3a1a2a',
        'microstructure-investigation': '#1a3a3a',
        'anomaly-investigation':     '#2a3a1a',
      };
      return colors[this.plan?.task_type] || '#2a2a2a';
    },

    get chartTypeLabel() {
      return (t) => ({
        bucket_profile:      'Profile',
        equity_curve:        'Equity',
        scatter:             'Scatter',
        yearly_consistency:  'Yearly',
        correlation_heatmap: 'Heatmap',
        combo_quadrant:      'Combo',
      })[t] || t;
    },

    // ── Knowledge library ────────────────────────────────────────────────
    async _loadKnowledge() {
      try {
        const r = await fetch('/api/research2/knowledge');
        if (r.ok) this.knowledgeRules = await r.json();
      } catch (_) {}
    },

    async addKnowledge() {
      const text = this.newKnowledgeText.trim();
      if (!text) return;
      try {
        const r = await fetch('/api/research2/knowledge', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ category: this.newKnowledgeCategory, rule: text }),
        });
        if (r.ok) {
          this.newKnowledgeText = '';
          await this._loadKnowledge();
        }
      } catch (_) {}
    },

    async saveKnowledgeEdit(id) {
      if (this.editingKnowledgeId !== id) return;
      try {
        await fetch(`/api/research2/knowledge/${id}`, {
          method: 'PATCH',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ rule: this.editingKnowledgeText }),
        });
        this.editingKnowledgeId = null;
        await this._loadKnowledge();
      } catch (_) {}
    },

    async toggleKnowledgeActive(kr) {
      try {
        await fetch(`/api/research2/knowledge/${kr.id}`, {
          method: 'PATCH',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ active: !kr.active }),
        });
        await this._loadKnowledge();
      } catch (_) {}
    },

    async deleteKnowledge(id) {
      try {
        await fetch(`/api/research2/knowledge/${id}`, { method: 'DELETE' });
        await this._loadKnowledge();
      } catch (_) {}
    },

    // ── Helpers ────────────────────────────────────────────────────────────
    statusClass(s) {
      return { complete: 'badge-green', running: 'badge-yellow', error: 'badge-red' }[s] || 'badge-dim';
    },
    fmtDate(dt) { return dt ? String(dt).slice(0, 16).replace('T', ' ') : ''; },
  }));
});
