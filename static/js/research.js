document.addEventListener('alpine:init', () => {
  Alpine.data('research', () => ({

    // ── State ──────────────────────────────────────────────────────────────
    runs:        [],
    selectedRun: null,
    results:     [],
    charts:      [],
    view:        'list',   // 'list' | 'new' | 'run'
    loading:     false,
    error:       null,
    pollTimer:   null,

    // New-run form
    form: {
      name:           '',
      question:       '',
      table:          'daily_features',
      tickers:        '',
      x_columns:      '',
      y_columns:      'ret_1d_fwd, ret_5d_fwd, ret_20d_fwd',
      date_from:      '',
      date_to:        '',
      model:          'claude-sonnet-4-6',
      max_tool_calls: 60,
    },
    availableTickers: [],
    submitting: false,
    submitError: null,

    // ── Init ───────────────────────────────────────────────────────────────
    async init() {
      await this.loadRuns();
      this._loadTickers();
    },

    async _loadTickers() {
      try {
        const r = await fetch('/api/research/tickers');
        if (r.ok) this.availableTickers = await r.json();
      } catch (_) {}
    },

    // ── Run list ───────────────────────────────────────────────────────────
    async loadRuns() {
      try {
        const r = await fetch('/api/research/runs');
        this.runs = r.ok ? await r.json() : [];
      } catch (_) { this.runs = []; }
    },

    async selectRun(run) {
      this.view        = 'run';
      this.selectedRun = run;
      this.results     = [];
      this.charts      = [];
      this.error       = null;
      this._stopPoll();
      await this._loadRunDetail(run.id);
      if (run.status === 'running') this._startPoll(run.id);
    },

    async _loadRunDetail(runId) {
      this.loading = true;
      try {
        const [runRes, resultsRes, chartsRes] = await Promise.all([
          fetch(`/api/research/run/${runId}`),
          fetch(`/api/research/run/${runId}/results`),
          fetch(`/api/research/run/${runId}/charts`),
        ]);
        if (runRes.ok)     this.selectedRun = await runRes.json();
        if (resultsRes.ok) this.results     = await resultsRes.json();
        if (chartsRes.ok)  this.charts      = await chartsRes.json();
      } catch (e) {
        this.error = e.message;
      } finally {
        this.loading = false;
      }
    },

    // ── Polling (for running jobs) ─────────────────────────────────────────
    _startPoll(runId) {
      this.pollTimer = setInterval(async () => {
        try {
          const r = await fetch(`/api/research/run/${runId}`);
          if (!r.ok) return;
          const data = await r.json();
          this.selectedRun = data;
          // Update status in sidebar list
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
      this.view        = 'new';
      this.selectedRun = null;
      this.submitError = null;
    },

    addTicker(ticker) {
      const existing = this.form.tickers.split(',').map(s => s.trim()).filter(Boolean);
      if (!existing.includes(ticker)) {
        this.form.tickers = [...existing, ticker].join(', ');
      }
    },

    setTablePreset(table) {
      this.form.table = table;
      if (table === 'daily_features') {
        this.form.x_columns = 'oi_weighted_strike_all_div_spot, put_call_oi_ratio';
        this.form.y_columns = 'ret_1d_fwd, ret_5d_fwd, ret_20d_fwd';
      } else if (table === 'surface_metrics_core') {
        this.form.x_columns = 'skew_30d_25p_atm, skew_30d_10p_atm, term_ratio_7d_30d';
        this.form.y_columns = 'iv_30d_atm, iv_7d_atm';
        this.form.tickers   = '';
      }
    },

    async submitRun() {
      this.submitError = null;
      const name = this.form.name.trim();
      const question = this.form.question.trim();
      if (!name || !question) {
        this.submitError = 'Run name and research question are required.';
        return;
      }

      const parse = s => s.split(',').map(x => x.trim()).filter(Boolean);

      this.submitting = true;
      try {
        const body = {
          name,
          question,
          table:          this.form.table,
          tickers:        parse(this.form.tickers),
          x_columns:      parse(this.form.x_columns),
          y_columns:      parse(this.form.y_columns),
          date_from:      this.form.date_from || null,
          date_to:        this.form.date_to   || null,
          model:          this.form.model,
          max_tool_calls: parseInt(this.form.max_tool_calls) || 60,
        };
        const r = await fetch('/api/research/run', {
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
        const run = this.runs.find(x => x.id === data.run_id) || { id: data.run_id, name: data.name, status: 'running' };
        await this.selectRun(run);
      } catch (e) {
        this.submitError = e.message;
      } finally {
        this.submitting = false;
      }
    },

    // ── Delete ─────────────────────────────────────────────────────────────
    async deleteRun(runId) {
      if (!confirm('Delete this research run and all its results?')) return;
      await fetch(`/api/research/run/${runId}`, { method: 'DELETE' });
      if (this.selectedRun?.id === runId) { this.view = 'list'; this.selectedRun = null; }
      await this.loadRuns();
    },

    // ── PDF export ─────────────────────────────────────────────────────────
    exportPdf(runId) {
      window.location.href = `/api/research/run/${runId}/pdf`;
    },

    // ── Helpers ────────────────────────────────────────────────────────────
    statusClass(status) {
      return { complete: 'badge-green', running: 'badge-yellow', error: 'badge-red' }[status] || 'badge-dim';
    },

    fmtDate(dt) {
      if (!dt) return '';
      return String(dt).slice(0, 16).replace('T', ' ');
    },

    chartUrl(chartId) {
      return `/api/research/chart/${chartId}.png`;
    },

    // Results helpers
    get correlations() {
      return this.results.filter(r => r.analysis_type === 'correlation');
    },
    get deciles() {
      return this.results.filter(r => r.analysis_type === 'decile');
    },
    get equityCurves() {
      return this.results.filter(r => r.analysis_type?.startsWith('equity_curve'));
    },
    get yearlyConsistency() {
      return this.results.filter(r => r.analysis_type === 'yearly_consistency');
    },

    // Summary table: join correlation + decile + equity_curve by (ticker, x_col, y_col)
    get summaryRows() {
      const key = (ticker, x, y) => `${ticker}||${x}||${y}`;
      const map = {};

      for (const r of this.correlations) {
        const k = key(r.ticker, r.x_col, r.y_col);
        map[k] = map[k] || { ticker: r.ticker, x_col: r.x_col, y_col: r.y_col };
        map[k].pearson_r  = r.result?.pearson_r;
        map[k].pearson_p  = r.result?.pearson_p;
        map[k].n          = r.result?.n;
      }
      for (const r of this.deciles) {
        const k = key(r.ticker, r.x_col, r.y_col);
        map[k] = map[k] || { ticker: r.ticker, x_col: r.x_col, y_col: r.y_col };
        map[k].spread = r.result?.top_bottom_spread;
      }
      for (const r of this.yearlyConsistency) {
        const k = key(r.ticker, r.x_col, r.y_col);
        map[k] = map[k] || { ticker: r.ticker, x_col: r.x_col, y_col: r.y_col };
        map[k].consistency_pct = r.result?.consistency_pct;
      }
      for (const r of this.equityCurves) {
        if (r.analysis_type !== 'equity_curve_top') continue;
        const k = key(r.ticker, r.x_col, r.y_col);
        map[k] = map[k] || { ticker: r.ticker, x_col: r.x_col, y_col: r.y_col };
        map[k].final_equity = r.result?.final_equity;
        map[k].max_drawdown = r.result?.max_drawdown;
      }

      return Object.values(map).sort((a, b) => {
        const sa = Math.abs(a.spread || 0);
        const sb = Math.abs(b.spread || 0);
        return sb - sa;
      });
    },

    pct(v) { return v != null ? (v * 100).toFixed(2) + '%' : '—'; },
    r4(v)  { return v != null ? v.toFixed(4) : '—'; },
    r2(v)  { return v != null ? v.toFixed(2) : '—'; },
  }));
});
