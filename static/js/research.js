document.addEventListener('alpine:init', () => {
  Alpine.data('research', () => ({

    // ── State ──────────────────────────────────────────────────────────────
    runs:        [],
    selectedRun: null,
    results:     [],
    charts:      [],
    followups:   [],
    view:        'list',   // 'list' | 'new' | 'run'
    loading:     false,
    error:       null,
    pollTimer:   null,
    lightboxUrl: null,
    _chartInstances: {},

    // New-run / retry form
    form: {
      name:           '',
      question:       '',
      table:          'daily_features',
      tickers:        '',
      bucketList:     [{name: '', cols: ''}],  // [{name, cols}]
      x_columns:      '',
      y_columns:      '',
      date_from:      '',
      date_to:        '',
      model:          'claude-sonnet-4-6',
      max_tool_calls: 60,
    },
    retryRunId:   null,   // set when retrying a failed run
    availableTickers: [],
    submitting: false,
    submitError: null,

    // Follow-up chat
    followupQuestion: '',
    followupLoading:  false,
    followupError:    null,

    // Column picker
    availableColumns: [],

    // ── Init ───────────────────────────────────────────────────────────────
    async init() {
      this._loadColumns(this.form.table);
      if (typeof Chart !== 'undefined') {
        Chart.defaults.color = '#ccc';
        Chart.defaults.borderColor = '#333';
        // Fill canvas background so toDataURL captures dark BG correctly
        Chart.register({
          id: 'darkBackground',
          beforeDraw: (chart) => {
            const ctx = chart.canvas.getContext('2d');
            ctx.save();
            ctx.globalCompositeOperation = 'destination-over';
            ctx.fillStyle = '#1e1e1e';
            ctx.fillRect(0, 0, chart.width, chart.height);
            ctx.restore();
          },
        });
      }
      await this.loadRuns();
      this._loadTickers();
    },

    async _loadTickers() {
      try {
        const r = await fetch('/api/research/tickers');
        if (r.ok) this.availableTickers = await r.json();
      } catch (_) {}
    },

    async _loadColumns(table) {
      try {
        const r = await fetch(`/api/research/columns?table=${encodeURIComponent(table)}`);
        if (r.ok) this.availableColumns = await r.json();
      } catch (_) {}
    },

    toggleXCol(col) {
      const list = this.form.x_columns.split(',').map(s => s.trim()).filter(Boolean);
      const i = list.indexOf(col);
      i >= 0 ? list.splice(i, 1) : list.push(col);
      this.form.x_columns = list.join(', ');
    },
    toggleYCol(col) {
      const list = this.form.y_columns.split(',').map(s => s.trim()).filter(Boolean);
      const i = list.indexOf(col);
      i >= 0 ? list.splice(i, 1) : list.push(col);
      this.form.y_columns = list.join(', ');
    },
    isXSel(col) { return this.form.x_columns.split(',').map(s => s.trim()).includes(col); },
    isYSel(col) { return this.form.y_columns.split(',').map(s => s.trim()).includes(col); },

    // ── Run list ───────────────────────────────────────────────────────────
    async loadRuns() {
      try {
        const r = await fetch('/api/research/runs');
        this.runs = r.ok ? await r.json() : [];
      } catch (_) { this.runs = []; }
    },

    async selectRun(run) {
      this.view             = 'run';
      this.selectedRun      = run;
      this.results          = [];
      this.charts           = [];
      this.followups        = [];
      this.followupQuestion = '';
      this.followupError    = null;
      this.error            = null;
      this.retryRunId       = null;
      this._stopPoll();
      this._destroyCharts();
      await this._loadRunDetail(run.id);
      if (run.status === 'running') this._startPoll(run.id);
    },

    async _loadRunDetail(runId) {
      this.loading = true;
      this._destroyCharts();
      try {
        const [runRes, resultsRes, chartsRes, fupsRes] = await Promise.all([
          fetch(`/api/research/run/${runId}`),
          fetch(`/api/research/run/${runId}/results`),
          fetch(`/api/research/run/${runId}/charts`),
          fetch(`/api/research/run/${runId}/followups`),
        ]);
        if (runRes.ok)     this.selectedRun = await runRes.json();
        if (resultsRes.ok) this.results     = await resultsRes.json();
        if (chartsRes.ok)  this.charts      = await chartsRes.json();
        if (fupsRes.ok)    this.followups   = await fupsRes.json();
        await this._renderInteractiveCharts();
      } catch (e) {
        this.error = e.message;
      } finally {
        this.loading = false;
      }
    },

    // ── Polling ────────────────────────────────────────────────────────────
    _startPoll(runId) {
      this.pollTimer = setInterval(async () => {
        try {
          const r = await fetch(`/api/research/run/${runId}`);
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
      this.view        = 'new';
      this.selectedRun = null;
      this.retryRunId  = null;
      this.submitError = null;
      // Clear form
      this.form = { name:'', question:'', table:'daily_features', tickers:'',
                    bucketList:[{name:'', cols:''}],
                    x_columns:'', y_columns:'',
                    date_from:'', date_to:'', model:'claude-sonnet-4-6', max_tool_calls:60 };
    },

    // Pre-populate form from a failed run for editing + retry
    editRetry() {
      const run = this.selectedRun;
      if (!run) return;
      const cfg = run.config || {};
      this.form.name           = run.name;
      this.form.question       = run.question;
      this.form.table          = cfg.table          || 'daily_features';
      this.form.tickers        = (cfg.tickers        || []).join(', ');
      this.form.x_columns      = (cfg.x_columns      || []).join(', ');
      this.form.y_columns      = (cfg.y_columns      || []).join(', ');
      this.form.date_from      = cfg.date_from       || '';
      this.form.date_to        = cfg.date_to         || '';
      this.form.model          = cfg.model           || 'claude-sonnet-4-6';
      this.form.max_tool_calls = cfg.max_tool_calls  || 60;
      this.retryRunId  = run.id;
      this.submitError = null;
      this.view        = 'new';
    },

    addTicker(ticker) {
      const existing = this.form.tickers.split(',').map(s => s.trim()).filter(Boolean);
      if (!existing.includes(ticker)) {
        this.form.tickers = [...existing, ticker].join(', ');
      }
    },

    setTablePreset(table) {
      this.form.table = table;
      // Clear columns — user selects from the available chips below
      this.form.x_columns = '';
      this.form.y_columns = '';
      if (table === 'surface_metrics_core') {
        this.form.tickers = '';
      }
      this._loadColumns(table);
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

      // Build buckets from the bucket form
      const buckets = {};
      for (const bk of (this.form.bucketList || [])) {
        const bname = bk.name.trim();
        const bcols = parse(bk.cols);
        if (bname && bcols.length) buckets[bname] = bcols;
      }

      const body = {
        name,
        question,
        table:          this.form.table,
        tickers:        parse(this.form.tickers),
        buckets,
        x_columns:      parse(this.form.x_columns),
        y_columns:      parse(this.form.y_columns),
        date_from:      this.form.date_from || null,
        date_to:        this.form.date_to   || null,
        model:          this.form.model,
      };

      this.submitting = true;
      try {
        const url    = this.retryRunId
          ? `/api/research/run/${this.retryRunId}/retry`
          : '/api/research/run';
        const r = await fetch(url, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(body),
        });
        if (!r.ok) {
          const err = await r.json().catch(() => ({}));
          throw new Error(err.detail || `HTTP ${r.status}`);
        }
        const data = await r.json();
        const runId = data.run_id || this.retryRunId;
        this.retryRunId = null;
        await this.loadRuns();
        const run = this.runs.find(x => x.id === runId) || { id: runId, name, status: 'running' };
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

    // ── Lightbox ───────────────────────────────────────────────────────────
    openLightbox(url)        { this.lightboxUrl = url; },
    closeLightbox()          { this.lightboxUrl = null; },
    openChartFullscreen(id)  {
      const el = document.getElementById(id);
      if (el) this.lightboxUrl = el.toDataURL('image/png');
    },

    // ── Chart.js interactive charts ────────────────────────────────────────
    async _renderInteractiveCharts() {
      if (typeof Chart === 'undefined') return;
      await this.$nextTick();
      this._destroyCharts();

      const darkScales = {
        x: { ticks: { color: '#ccc', font: { size: 11 } }, grid: { color: '#333' } },
        y: { ticks: { color: '#ccc', font: { size: 11 } }, grid: { color: '#333' } },
      };

      // ── Decile bar charts ────────────────────────────────────────────────
      for (const r of this.deciles) {
        const el = document.getElementById('c-decile-' + r.id);
        if (!el || !r.result?.deciles) continue;
        const d = r.result.deciles;
        const vals = d.map(x => +((x.avg_ret || 0) * 100).toFixed(4));
        this._chartInstances['c-decile-' + r.id] = new Chart(el, {
          type: 'bar',
          data: {
            labels: d.map(x => 'D' + x.decile),
            datasets: [{
              label: 'Avg Ret %',
              data: vals,
              backgroundColor: vals.map(v => v >= 0 ? 'rgba(46,204,113,0.75)' : 'rgba(231,76,60,0.75)'),
              borderColor:     vals.map(v => v >= 0 ? '#2ecc71' : '#e74c3c'),
              borderWidth: 1,
            }],
          },
          options: {
            responsive: true, maintainAspectRatio: false,
            plugins: {
              legend: { display: false },
              tooltip: {
                callbacks: {
                  label: ctx => {
                    const dec = d[ctx.dataIndex];
                    return [
                      `Avg Ret:  ${(dec.avg_ret*100).toFixed(3)}%`,
                      `Median:   ${(dec.med_ret*100).toFixed(3)}%`,
                      `Win Rate: ${(dec.win_rate*100).toFixed(1)}%`,
                      `N: ${dec.n}`,
                    ];
                  }
                }
              },
            },
            scales: darkScales,
          },
        });
      }

      // ── Equity curves (render top + bottom together) ─────────────────────
      const topResults = this.results.filter(r => r.analysis_type === 'equity_curve_top');
      for (const r of topResults) {
        const el = document.getElementById('c-equity-' + r.id);
        if (!el || !r.result?.points) continue;
        const pts = r.result.points;
        const datasets = [{
          label: 'Top Decile',
          data:  pts.map(p => p.value),
          borderColor: '#2ecc71', backgroundColor: 'transparent',
          borderWidth: 2, pointRadius: 0, tension: 0.1,
        }];
        const botResult = this.results.find(x =>
          x.analysis_type === 'equity_curve_bottom' &&
          x.ticker === r.ticker && x.x_col === r.x_col && x.y_col === r.y_col
        );
        if (botResult?.result?.points) {
          datasets.push({
            label: 'Bottom Decile',
            data:  botResult.result.points.map(p => p.value),
            borderColor: '#e74c3c', backgroundColor: 'transparent',
            borderWidth: 2, pointRadius: 0, tension: 0.1,
          });
        }
        const finalEq = r.result.final_equity;
        const maxDD   = r.result.max_drawdown;
        this._chartInstances['c-equity-' + r.id] = new Chart(el, {
          type: 'line',
          data: { labels: pts.map(p => p.date), datasets },
          options: {
            responsive: true, maintainAspectRatio: false,
            interaction: { mode: 'index', intersect: false },
            plugins: {
              legend: {
                labels: { color: '#ccc', boxWidth: 12, font: { size: 11 } }
              },
              tooltip: {
                callbacks: {
                  label: ctx => `${ctx.dataset.label}: ${ctx.raw.toFixed(3)}x`,
                  footer: () => finalEq != null
                    ? [`Final: ${finalEq.toFixed(3)}x  |  MaxDD: ${(maxDD*100).toFixed(1)}%`]
                    : [],
                }
              },
            },
            scales: {
              x: { ticks: { color: '#ccc', font: { size: 10 }, maxTicksLimit: 8, maxRotation: 30 }, grid: { color: '#333' } },
              y: { ticks: { color: '#ccc', font: { size: 11 } }, grid: { color: '#333' } },
            },
          },
        });
      }

      // ── Yearly consistency bar charts ────────────────────────────────────
      for (const r of this.yearlyConsistency) {
        const el = document.getElementById('c-yearly-' + r.id);
        if (!el || !r.result?.years) continue;
        const yrs = r.result.years;
        this._chartInstances['c-yearly-' + r.id] = new Chart(el, {
          type: 'bar',
          data: {
            labels: yrs.map(y => String(y.year)),
            datasets: [
              {
                label: 'Top Decile',
                data: yrs.map(y => +((y.top_avg || 0)*100).toFixed(4)),
                backgroundColor: 'rgba(46,204,113,0.75)', borderColor: '#2ecc71', borderWidth: 1,
              },
              {
                label: 'Bottom Decile',
                data: yrs.map(y => +((y.bot_avg || 0)*100).toFixed(4)),
                backgroundColor: 'rgba(231,76,60,0.75)', borderColor: '#e74c3c', borderWidth: 1,
              },
            ],
          },
          options: {
            responsive: true, maintainAspectRatio: false,
            interaction: { mode: 'index', intersect: false },
            plugins: {
              legend: { labels: { color: '#ccc', boxWidth: 12, font: { size: 11 } } },
              tooltip: {
                callbacks: {
                  label: ctx => `${ctx.dataset.label}: ${ctx.raw.toFixed(3)}%`,
                  afterBody: (items) => {
                    const yr = yrs[items[0]?.dataIndex];
                    return yr ? [`Top beats bottom: ${yr.top_beats ? 'Yes' : 'No'}  N: ${yr.n}`] : [];
                  },
                }
              },
            },
            scales: darkScales,
          },
        });
      }
    },

    _destroyCharts() {
      for (const c of Object.values(this._chartInstances)) {
        try { c.destroy(); } catch (_) {}
      }
      this._chartInstances = {};
    },

    // ── Progress ───────────────────────────────────────────────────────────
    get runConfig() {
      const cfg = this.selectedRun?.config;
      if (!cfg) return {};
      return typeof cfg === 'string' ? JSON.parse(cfg) : cfg;
    },

    get expectedTotal() {
      const cfg = this.runConfig;
      const tickers  = Math.max(1, (cfg.tickers || []).length);
      const xCols    = (cfg.x_columns || []).length || 1;
      const yCols    = (cfg.y_columns || []).length || 1;
      const maxCalls = cfg.max_tool_calls || 60;
      return tickers * xCols * yCols + maxCalls;
    },

    get progressPct() {
      if (!this.selectedRun || this.selectedRun.status !== 'running') return 100;
      const cnt = this.selectedRun.result_count || 0;
      return Math.min(94, Math.round(cnt / this.expectedTotal * 100));
    },

    get progressLabel() {
      const cnt = this.selectedRun?.result_count || 0;
      const cfg = this.runConfig;
      const t   = Math.max(1, (cfg.tickers  || []).length);
      const x   = (cfg.x_columns || []).length || 1;
      const y   = (cfg.y_columns || []).length || 1;
      const p1  = t * x * y;
      if (cnt <= p1) return `Phase 1: ${cnt} / ${p1} correlations`;
      return `Phase 2: ${cnt - p1} AI analyses complete`;
    },

    // ── Follow-up dialogue ─────────────────────────────────────────────────
    async askFollowup() {
      const q = this.followupQuestion.trim();
      if (!q || this.followupLoading) return;
      this.followupError   = null;
      this.followupLoading = true;
      try {
        const r = await fetch(`/api/research/run/${this.selectedRun.id}/followup`, {
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
        const el = document.getElementById('followup-bottom');
        if (el) el.scrollIntoView({ behavior: 'smooth' });
      }
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

    // ── Results helpers ────────────────────────────────────────────────────
    get scans() {
      return this.results.filter(r => r.analysis_type === 'scan');
    },
    get correlations() {
      // Legacy support + extract from scan results
      const legacy = this.results.filter(r => r.analysis_type === 'correlation');
      if (legacy.length) return legacy;
      // Map scan results to correlation-shaped objects
      return this.scans.filter(r => r.result?.pearson_r != null);
    },
    get deciles() {
      // Legacy support + extract bucket_stats from scan results
      const legacy = this.results.filter(r => r.analysis_type === 'decile');
      if (legacy.length) return legacy;
      return this.scans.filter(r => r.result?.bucket_stats?.length > 0)
        .map(r => ({
          ...r,
          analysis_type: 'decile',
          result: {
            deciles: r.result.bucket_stats.filter(b => b != null).map(b => ({
              decile: b.bucket, n: b.n, avg_ret: b.avg_ret,
              med_ret: b.med_ret, win_rate: b.win_rate, std_dev: b.std_dev,
            })),
            top_bottom_spread: r.result.tail_spread,
            feature_col: r.result.x_col,
            outcome_col: r.result.y_col,
          },
        }));
    },
    get equityCurves() {
      return this.results.filter(r => r.analysis_type?.startsWith('equity_curve'));
    },
    get yearlyConsistency() {
      // Legacy support + extract from scan robustness
      const legacy = this.results.filter(r => r.analysis_type === 'yearly_consistency');
      if (legacy.length) return legacy;
      return this.scans.filter(r => r.result?.robustness?.yearly_consistency_pct != null)
        .map(r => ({
          ...r,
          analysis_type: 'yearly_consistency',
          result: {
            consistency_pct: r.result.robustness.yearly_consistency_pct,
            wins: r.result.robustness.years_consistent,
            total_years: r.result.robustness.years_checked,
          },
        }));
    },
    get interactions() {
      return this.results.filter(r =>
        r.analysis_type === 'interaction' || r.analysis_type === 'interaction_3f');
    },
    get combos() {
      return this.results.filter(r => r.analysis_type === 'combo');
    },

    // Multi-factor scorecard (PRIMARY output when combos exist)
    get comboRows() {
      return this.combos.map(r => {
        const res = r.result || {};
        const bz = res.best_quadrant || res.best_octant || res.best_zone || {};
        const rob = res.robustness || {};
        return {
          ticker:      r.ticker,
          combo:       (res.combo || []).join(' + '),
          buckets:     (res.buckets_used || []).join('+'),
          y_col:       r.y_col,
          n_factors:   (res.combo || []).length,
          best_zone:   bz.label || '?',
          n:           res.n,
          n_zone:      rob.n_zone || bz.n,
          avg_ret:     rob.avg_ret || bz.avg_ret,
          med_ret:     rob.med_ret || bz.med_ret,
          win_rate:    rob.win_rate || bz.win_rate,
          max_drawdown: rob.max_drawdown,
          final_equity: rob.final_equity,
          consistency: rob.yearly_consistency_pct,
          lift:        res.interaction_lift,
          score:       res.composite_interaction_score,
          r2:          res.ols_r2,
          warnings:    (res.overfit_warnings || []).join('; '),
          baseline:    res.baseline_best_single,
        };
      }).sort((a, b) => (b.score || 0) - (a.score || 0));
    },

    // PNG charts — show ALL generated chart images on screen
    get staticCharts() {
      return this.charts;
    },

    exportComboCsv() {
      const rows = this.comboRows;
      if (!rows.length) return;
      const cols = ['ticker','combo','buckets','y_col','n_factors','best_zone',
                    'n','n_zone','avg_ret','med_ret','win_rate','max_drawdown',
                    'final_equity','consistency','lift','score','r2','warnings'];
      const header = cols.join(',');
      const lines = rows.map(r => cols.map(c => {
        const v = r[c]; return v != null ? String(v).replace(/,/g, ';') : '';
      }).join(','));
      const csv = [header, ...lines].join('\n');
      const blob = new Blob([csv], { type: 'text/csv' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url; a.download = 'combo_scorecard.csv'; a.click();
      URL.revokeObjectURL(url);
    },

    exportScorecardCsv() {
      const rows = this.summaryRows;
      if (!rows.length) return;
      const cols = ['ticker','x_col','y_col','composite_score','pattern',
                    'pearson_r','spearman_r','spread','consistency_pct',
                    'final_equity','max_drawdown','n'];
      const header = cols.join(',');
      const lines = rows.map(r => cols.map(c => {
        const v = r[c];
        return v != null ? v : '';
      }).join(','));
      const csv = [header, ...lines].join('\n');
      const blob = new Blob([csv], { type: 'text/csv' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url; a.download = 'signal_scorecard.csv'; a.click();
      URL.revokeObjectURL(url);
    },

    // Interactive chart sections (types that have structured result data)
    get hasInteractiveCharts() {
      return this.deciles.length > 0 ||
             this.results.some(r => r.analysis_type === 'equity_curve_top') ||
             this.yearlyConsistency.length > 0;
    },

    // Summary table — reads from scan results (new engine) or legacy types
    get summaryRows() {
      const key = (ticker, x, y) => `${ticker}||${x}||${y}`;
      const map = {};

      // New engine: scan results contain everything
      for (const r of this.scans) {
        const k = key(r.ticker, r.x_col, r.y_col);
        const res = r.result || {};
        const rob = res.robustness || {};
        map[k] = {
          ticker: r.ticker, x_col: r.x_col, y_col: r.y_col,
          pearson_r:       res.pearson_r,
          pearson_p:       res.pearson_p,
          spearman_r:      res.spearman_r,
          n:               res.n,
          spread:          res.tail_spread,
          consistency_pct: rob.yearly_consistency_pct,
          pattern:         res.pattern,
          composite_score: res.composite_score,
          monotonicity:    res.monotonicity,
        };
      }

      // Legacy: correlation + decile + yearly separate types
      for (const r of this.results.filter(x => x.analysis_type === 'correlation')) {
        const k = key(r.ticker, r.x_col, r.y_col);
        map[k] = map[k] || { ticker: r.ticker, x_col: r.x_col, y_col: r.y_col };
        map[k].pearson_r = r.result?.pearson_r;
        map[k].pearson_p = r.result?.pearson_p;
        map[k].n         = r.result?.n;
      }
      for (const r of this.results.filter(x => x.analysis_type === 'decile')) {
        const k = key(r.ticker, r.x_col, r.y_col);
        map[k] = map[k] || { ticker: r.ticker, x_col: r.x_col, y_col: r.y_col };
        map[k].spread = r.result?.top_bottom_spread;
      }

      // Equity curves (both old and new engine)
      for (const r of this.equityCurves) {
        if (r.analysis_type !== 'equity_curve_top') continue;
        const k = key(r.ticker, r.x_col, r.y_col);
        map[k] = map[k] || { ticker: r.ticker, x_col: r.x_col, y_col: r.y_col };
        map[k].final_equity = r.result?.final_equity;
        map[k].max_drawdown = r.result?.max_drawdown;
      }

      return Object.values(map).sort((a, b) => {
        const sa = a.composite_score || Math.abs(a.spread || 0) * 1000;
        const sb = b.composite_score || Math.abs(b.spread || 0) * 1000;
        return sb - sa;
      });
    },

    pct(v) { return v != null ? (v * 100).toFixed(2) + '%' : '—'; },
    r4(v)  { return v != null ? v.toFixed(4) : '—'; },
    r2(v)  { return v != null ? v.toFixed(2) : '—'; },
  }));
});
