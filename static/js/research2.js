// Simple markdown-to-HTML for research reports (no external lib)
function _mdToHtml(md) {
  if (!md) return '';
  return md
    .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
    // Headers
    .replace(/^### (.+)$/gm, '<h4 style="color:#ddd;margin:12px 0 4px;font-size:13px">$1</h4>')
    .replace(/^## (.+)$/gm, '<h3 style="color:#eee;margin:16px 0 6px;font-size:14px">$1</h3>')
    .replace(/^# (.+)$/gm, '<h2 style="color:#fff;margin:18px 0 8px;font-size:15px">$1</h2>')
    // Horizontal rule
    .replace(/^---$/gm, '<hr style="border:none;border-top:1px solid var(--border);margin:16px 0">')
    // Bold and italic
    .replace(/\*\*(.+?)\*\*/g, '<strong style="color:#ddd">$1</strong>')
    .replace(/\*(.+?)\*/g, '<em>$1</em>')
    // Bullet lists
    .replace(/^- (.+)$/gm, '<li style="margin:2px 0">$1</li>')
    .replace(/(<li[^>]*>.*<\/li>\n?)+/g, '<ul style="margin:4px 0 8px 16px;padding:0;list-style:disc">$&</ul>')
    // Paragraphs (double newline)
    .replace(/\n\n/g, '</p><p style="margin:6px 0">')
    // Wrap in p
    .replace(/^/, '<p style="margin:6px 0">')
    .replace(/$/, '</p>');
}

document.addEventListener('alpine:init', () => {
  // Utility store for template access to results
  Alpine.store('r2utils', {
    _results: [],
    setResults(r) { this._results = r; },
    getBot(topResult) {
      return this._results.find(x =>
        x.analysis_type === 'equity_curve_bottom' &&
        x.ticker === topResult.ticker &&
        x.x_col === topResult.x_col &&
        x.y_col === topResult.y_col) || null;
    },
  });

  Alpine.data('research2', () => ({

    // ── State ──────────────────────────────────────────────────────────────
    runs:        [],
    selectedRun: null,
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

    // P&L upload
    uploadState: {
      file:       null,
      uploading:  false,
      uploadId:   null,
      uploadName: null,
      preview:    null,
      columns:    null,
      rowCount:   null,
      dateFrom:   null,
      dateTo:     null,
      error:      null,
    },

    // Backtest upload (staged: one file at a time, then finalize)
    backtestUploadState: {
      files:           [],
      uploading:       false,
      finalizing:      false,
      uploadId:        null,
      uploadName:      null,
      source:          null,
      tradeCount:      null,
      matchedCount:    null,
      matchRate:       null,
      dateFrom:        null,
      dateTo:          null,
      strategies:      [],
      columns:         [],
      sources:         [],
      stagedFiles:     [],    // filenames of staged uploads
      totalTrades:     0,
      preview:         null,
      warnings:        [],
      hasDailyPaths:   false,
      pathCount:       0,
      error:           null,
    },

    analysisMode: 'entry',  // 'entry' | 'intratrade' — only relevant when hasDailyPaths

    savedBacktestUploads: [],  // previously finalized uploads loaded from server

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
      this._loadSavedBacktestUploads();
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
        const [runRes, resultsRes, fupsRes] = await Promise.all([
          fetch(`/api/research2/run/${runId}`),
          fetch(`/api/research2/run/${runId}/results`),
          fetch(`/api/research2/run/${runId}/followups`),
        ]);
        if (runRes.ok)     this.selectedRun = await runRes.json();
        if (resultsRes.ok) {
          this.results = await resultsRes.json();
          Alpine.store('r2utils').setResults(this.results);
        }
        if (fupsRes.ok)    this.followups   = await fupsRes.json();

        // Render charts after DOM update (double nextTick for x-for templates)
        await this.$nextTick();
        await this.$nextTick();
        setTimeout(() => {
          this._renderInteractiveCharts();
          this._renderReportCharts();
        }, 50);
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

    _renderReportCharts() {
      const darkOpts = {
        responsive: true, maintainAspectRatio: false, animation: false,
        plugins: {
          legend: { labels: { color: '#aaa', font: { size: 10 } } },
          tooltip: {
            backgroundColor: 'rgba(20,20,20,0.95)', borderColor: '#444', borderWidth: 1,
          },
        },
        scales: {
          x: { ticks: { color: '#888', font: { size: 9 }, maxRotation: 45 },
               grid: { color: 'rgba(255,255,255,0.05)' }, border: { color: 'transparent' } },
          y: { ticks: { color: '#888', font: { size: 9 } },
               grid: { color: 'rgba(255,255,255,0.05)' }, border: { color: 'transparent' } },
        },
      };

      for (let i = 0; i < this.reportSections.length; i++) {
        const sec = this.reportSections[i];
        if (sec.type !== 'chart' || !sec.config) continue;
        const el = document.getElementById('r2-report-chart-' + i);
        if (!el) continue;
        if (this._chartInstances['r2-report-chart-' + i]) continue; // already rendered

        try {
          const cfg = JSON.parse(JSON.stringify(sec.config)); // deep clone
          // Apply dark theme defaults if options not fully specified
          if (!cfg.options) cfg.options = {};
          const o = cfg.options;
          if (!o.responsive) o.responsive = true;
          if (o.maintainAspectRatio === undefined) o.maintainAspectRatio = false;
          if (o.animation === undefined) o.animation = false;
          if (!o.plugins) o.plugins = {};
          if (!o.plugins.legend) o.plugins.legend = darkOpts.plugins.legend;
          if (!o.plugins.tooltip) o.plugins.tooltip = darkOpts.plugins.tooltip;
          if (!o.scales) o.scales = {};
          // Apply dark scale defaults unless already set
          for (const axis of ['x', 'y']) {
            if (!o.scales[axis]) o.scales[axis] = {};
            const ax = o.scales[axis];
            if (!ax.ticks) ax.ticks = {};
            if (!ax.ticks.color) ax.ticks.color = '#888';
            if (!ax.ticks.font) ax.ticks.font = { size: 9 };
            if (!ax.grid) ax.grid = { color: 'rgba(255,255,255,0.05)' };
            if (!ax.border) ax.border = { color: 'transparent' };
          }
          this._chartInstances['r2-report-chart-' + i] = new Chart(el, cfg);
        } catch (e) {
          console.error('Report chart render error at section ' + i, e);
        }
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
      this.uploadState = {
        file: null, uploading: false, uploadId: null, uploadName: null,
        preview: null, columns: null, rowCount: null,
        dateFrom: null, dateTo: null, error: null,
      };
      this.analysisMode = 'entry';
      this.backtestUploadState = {
        files: [], uploading: false, finalizing: false,
        uploadId: null, uploadName: null,
        source: null, sources: [], stagedFiles: [], totalTrades: 0,
        tradeCount: null, matchedCount: null, matchRate: null,
        dateFrom: null, dateTo: null, strategies: [], columns: [],
        preview: null, warnings: [], hasDailyPaths: false, pathCount: 0, error: null,
      };
    },

    // ── P&L upload ────────────────────────────────────────────────────────
    onPnlFileChange(event) {
      const f = event.target.files[0];
      if (f) this.uploadState.file = f;
    },

    async uploadPnl() {
      const f = this.uploadState.file;
      if (!f) return;
      this.uploadState.uploading = true;
      this.uploadState.error     = null;
      const fd = new FormData();
      fd.append('file', f);
      fd.append('name', f.name.replace(/\.csv$/i, ''));
      try {
        const r = await fetch('/api/research2/upload-pnl', { method: 'POST', body: fd });
        if (!r.ok) {
          const err = await r.json().catch(() => ({}));
          throw new Error(err.detail || `HTTP ${r.status}`);
        }
        const data = await r.json();
        this.uploadState.uploadId   = data.upload_id;
        this.uploadState.uploadName = data.name;
        this.uploadState.preview    = data.preview;
        this.uploadState.columns    = data.columns;
        this.uploadState.rowCount   = data.row_count;
        this.uploadState.dateFrom   = data.date_from;
        this.uploadState.dateTo     = data.date_to;
        if (!this.form.date_from && data.date_from) this.form.date_from = data.date_from;
        if (!this.form.date_to   && data.date_to)   this.form.date_to   = data.date_to;
      } catch (e) {
        this.uploadState.error = e.message;
      } finally {
        this.uploadState.uploading = false;
      }
    },

    clearUpload() {
      this.uploadState = {
        file: null, uploading: false, uploadId: null, uploadName: null,
        preview: null, columns: null, rowCount: null,
        dateFrom: null, dateTo: null, error: null,
      };
    },

    // ── Backtest upload ───────────────────────────────────────────────────
    onBacktestFileChange(event) {
      const fs = Array.from(event.target.files || []);
      if (fs.length) this.backtestUploadState.files = fs;
    },

    async uploadBacktest() {
      // Upload files one at a time to staging
      const files = this.backtestUploadState.files;
      if (!files.length) return;
      this.backtestUploadState.uploading = true;
      this.backtestUploadState.error     = null;

      // Derive a name from the first file (used to group staged files)
      if (!this.backtestUploadState.uploadName) {
        this.backtestUploadState.uploadName =
          files[0].name.replace(/\.(csv|json)$/i, '').replace(/_[a-z]{3,4}$/i, '');
      }
      const name = this.backtestUploadState.uploadName;

      for (const file of files) {
        const fd = new FormData();
        fd.append('files', file);
        fd.append('name', name);
        try {
          const r = await fetch('/api/research2/upload-backtest', { method: 'POST', body: fd });
          if (!r.ok) {
            const err = await r.json().catch(() => ({}));
            throw new Error(err.detail || `HTTP ${r.status}`);
          }
          const data = await r.json();
          this.backtestUploadState.source      = data.source;
          this.backtestUploadState.stagedFiles  = data.staged_files || [];
          this.backtestUploadState.totalTrades  = data.total_trades || 0;
          this.backtestUploadState.strategies   = [
            ...new Set([...(this.backtestUploadState.strategies || []), ...(data.strategies || [])])
          ];
          this.backtestUploadState.dateFrom     = this.backtestUploadState.dateFrom
            ? (data.date_from && data.date_from < this.backtestUploadState.dateFrom ? data.date_from : this.backtestUploadState.dateFrom)
            : data.date_from;
          this.backtestUploadState.dateTo       = this.backtestUploadState.dateTo
            ? (data.date_to && data.date_to > this.backtestUploadState.dateTo ? data.date_to : this.backtestUploadState.dateTo)
            : data.date_to;
        } catch (e) {
          this.backtestUploadState.error = `${file.name}: ${e.message}`;
          break;
        }
      }
      // Clear file input so user can add more
      this.backtestUploadState.files = [];
      this.backtestUploadState.uploading = false;
    },

    async finalizeBacktest() {
      const name = this.backtestUploadState.uploadName;
      if (!name) return;
      this.backtestUploadState.finalizing = true;
      this.backtestUploadState.error = null;
      const fd = new FormData();
      fd.append('name', name);
      try {
        const r = await fetch('/api/research2/finalize-backtest', { method: 'POST', body: fd });
        if (!r.ok) {
          const err = await r.json().catch(() => ({}));
          throw new Error(err.detail || `HTTP ${r.status}`);
        }
        const data = await r.json();
        this.backtestUploadState.uploadId      = data.upload_id;
        this.backtestUploadState.tradeCount    = data.trade_count;
        this.backtestUploadState.matchedCount  = data.matched_count;
        this.backtestUploadState.matchRate     = data.match_rate;
        this.backtestUploadState.dateFrom      = data.date_from;
        this.backtestUploadState.dateTo        = data.date_to;
        this.backtestUploadState.strategies    = data.strategies || [];
        this.backtestUploadState.columns       = data.columns || [];
        this.backtestUploadState.sources       = data.sources || [];
        this.backtestUploadState.preview       = data.preview || [];
        this.backtestUploadState.warnings      = data.validation?.warnings || [];
        this.backtestUploadState.hasDailyPaths = data.has_daily_paths || false;
        this.backtestUploadState.pathCount     = data.path_count || 0;
        if (!this.backtestUploadState.hasDailyPaths) this.analysisMode = 'entry';
        if (!this.form.date_from && data.date_from) this.form.date_from = data.date_from;
        if (!this.form.date_to   && data.date_to)   this.form.date_to   = data.date_to;
      } catch (e) {
        this.backtestUploadState.error = e.message;
      } finally {
        this.backtestUploadState.finalizing = false;
      }
    },

    async clearBacktest() {
      // Clear staging table on server
      const name = this.backtestUploadState.uploadName;
      if (name) {
        await fetch(`/api/research2/clear-backtest-staging?name=${encodeURIComponent(name)}`,
                     { method: 'DELETE' }).catch(() => {});
      }
      this.analysisMode = 'entry';
      this.backtestUploadState = {
        files: [], uploading: false, finalizing: false,
        uploadId: null, uploadName: null,
        source: null, sources: [], stagedFiles: [], totalTrades: 0,
        tradeCount: null, matchedCount: null, matchRate: null,
        dateFrom: null, dateTo: null, strategies: [], columns: [],
        preview: null, warnings: [], hasDailyPaths: false, pathCount: 0, error: null,
      };
    },

    async _loadSavedBacktestUploads() {
      try {
        const r = await fetch('/api/research2/backtest-uploads');
        if (r.ok) this.savedBacktestUploads = await r.json();
      } catch (_) {}
    },

    selectSavedBacktestUpload(upload) {
      this.analysisMode = 'entry';
      this.backtestUploadState = {
        files: [], uploading: false, finalizing: false,
        uploadId:     upload.id,
        uploadName:   upload.name,
        source:       upload.source,
        tradeCount:   upload.trade_count,
        matchedCount: upload.matched_count,
        matchRate:    upload.match_rate || 0,
        dateFrom:     upload.date_from,
        dateTo:       upload.date_to,
        strategies:   upload.strategies || [],
        columns:      [],
        sources:      upload.sources || [],
        stagedFiles:  [],
        totalTrades:  upload.trade_count || 0,
        preview:      null,
        warnings:     [],
        hasDailyPaths: upload.has_daily_paths || false,
        pathCount:    upload.path_count || 0,
        error:        null,
      };
      if (!this.backtestUploadState.hasDailyPaths) this.analysisMode = 'entry';
      if (upload.date_from) this.form.date_from = upload.date_from;
      if (upload.date_to)   this.form.date_to   = upload.date_to;
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
        name:                this.form.name.trim(),
        question:            this.form.question.trim(),
        table:               this.form.table,
        tickers:             parse(this.form.tickers),
        date_from:           this.form.date_from || null,
        date_to:             this.form.date_to   || null,
        model:               this.form.model,
        pnl_upload_id:       this.uploadState.uploadId || null,
        backtest_upload_id:  this.backtestUploadState.uploadId || null,
        intratrade_mode:     this.backtestUploadState.hasDailyPaths && this.analysisMode === 'intratrade',
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
      window.location.href = `/api/research2/run/${runId}/pdf`;
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

    // ── Markdown rendering ──────────────────────────────────────────────
    _mdToHtml(md) { return _mdToHtml(md); },

    // ── Lightbox ───────────────────────────────────────────────────────────
    openLightbox(url)  { this.lightboxUrl = url; },
    closeLightbox()    { this.lightboxUrl = null; },

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

    get reportSections() {
      const s = this.selectedRun?.ai_summary || '';
      if (!s || s.startsWith('[RUNNING]')) return [];
      try {
        const parsed = JSON.parse(s);
        if (Array.isArray(parsed)) return parsed;
      } catch (_) {}
      // Legacy plain-text report: wrap in a single markdown section
      return [{ type: 'markdown', content: s }];
    },

    get reportText() {
      // Legacy accessor — extracts prose only (for PDF export, etc.)
      return this.reportSections
        .filter(s => s.type === 'markdown')
        .map(s => s.content || '')
        .join('\n\n');
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
        'pnl-iv-correlation':        '#2a1a3a',
        'backtest-regime-analysis':  '#1a3a2a',
      };
      return colors[this.plan?.task_type] || '#2a2a2a';
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
