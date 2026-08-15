'use strict';

// Factor Trades — exit-policy lab.
//
// The four shared panes (equity+DD, annual P&L, activity, ticker breakdown)
// render through window.FactorCharts, the SAME code Recall/Zone/Portfolio
// use — not a copy. Those functions take the component as their first
// argument and read state off it by name, so this component declares the
// same field names they expect (see the "FactorCharts contract" block
// below). Anything cosmetic about these charts changes in one place.

document.addEventListener('alpine:init', () => {
  Alpine.data('factorTrades', () => ({
    metrics: [], ruleGroups: [], cutoffDate: '',
    // metric -> {family_num, family_name}; drives the <optgroup>
    // grouping so this page's dropdowns read like Factor Analysis'.
    metricFamilyLookup: {},
    mode: '2f', primaryMetric: '', secondaryMetric: '', entryAnchor: 'open',
    selected: {},                 // family -> rule_key (absent = family off)
    perTrade: 2000, dailyCap: 10000, maxStrike: 1000,
    loading: false, error: '',
    runs: [], currentIdx: -1, lockedIdx: -1,
    runData: null, lockedRun: null, zoneData: null, lockedZone: null,
    gridView: 'edited', showDD: true, fsId: null, _fsHome: null,
    selectedCells: [],            // [[bp, bs], ...]

    // ── FactorCharts contract ────────────────────────────────────────────
    // window.FactorCharts.* reads these off the component by name. They must
    // keep these exact names; renaming any of them silently breaks a chart
    // rather than raising. Defaults mirror Recall's so the panes look and
    // behave identically out of the box.
    _charts: {},
    secDetail: null, data: null,
    // Heatmap colour scale. hmCellBg/_hmCellTitle live in FactorCharts and
    // read these; _hmRange is recomputed from the grid whenever it changes.
    heatmapData: null, _hmRange: null, hmMinSampleN: 0,
    metric: '', secSelectedMetric: '',
    // Keyed by section, because FactorCharts reads
    // cmp.activityMode?.[sectionKey] — a bare string silently
    // reads undefined and pins the chart to Count.
    activityMode: { ft: 'trades', sec: 'trades', port: 'trades' },
    dedupeConc: { primary: false, sec: false, corr: false, port: false },
    secBubbleMinN: 0,
    // 'ft' is this page's key (FactorCharts._equityModeKey maps ft-*
    // canvases to it). Dollar mode is the point of this page: every
    // axis is dollars derived from the rail's sizing controls.
    equityAggMode:      { ft: 'dollar_capped', zone: 'dollar_capped', sec: 'daily',
                          recall: 'dollar_capped', port: 'dollar_capped' },
    equityDollarParams: { ft:   { perTrade: 2000, dailyCap: 10000, maxStrike: 1000 },
                          zone: { perTrade: 2000, dailyCap: 10000 },
                          sec:  { perTrade: 2000, dailyCap: 10000 },
                          recall: { perTrade: 2000, dailyCap: 10000 },
                          port: { perTrade: 2000, dailyCap: 10000 } },

    async init() {
      try {
        const [cols, rules, tt] = await Promise.all([
          fetch('/api/factor-analysis/columns').then(r => r.ok ? r.json() : null),
          fetch('/api/factor-trades/rules').then(r => r.ok ? r.json() : null),
          fetch('/api/factor-analysis/tt-cutoff').then(r => r.ok ? r.json() : null),
        ]);
        this.metrics = cols?.features || [];
        for (const g of (cols?.feature_families || [])) {
          for (const mm of g.metrics) {
            this.metricFamilyLookup[mm] = { family_num: g.family_num,
                                            family_name: g.family_name };
          }
        }
        this.primaryMetric   = this.metrics[0] || '';
        this.secondaryMetric = this.metrics[1] || '';
        this.ruleGroups = rules?.groups || [];
        this.cutoffDate = tt?.cutoff_date || '';
        if (rules?.error) this.error = rules.error;
      } catch (e) { this.error = String(e); }
    },

    // A family is on when it has a chosen rule_key. Toggling on defaults to
    // that family's first precomputed value.
    toggleFamily(f) {
      if (this.selected[f.family]) delete this.selected[f.family];
      else this.selected[f.family] = f.rules[0]?.rule_key;
      this.selected = { ...this.selected };
    },
    // Full screen: MOVE the canvas into the overlay and back again, so the
    // Chart.js instance survives and only needs a resize. Re-creating it
    // would mean re-deriving the dollar series for a resize.
    openFs(id) {
      const c = document.getElementById(id);
      if (!c) return;
      this._fsHome = c.parentElement;
      this.fsId = id;
      this.$nextTick(() => {
        const body = document.getElementById('ft-ov-body');
        if (body) { body.appendChild(c); this._charts[id]?.resize(); }
      });
    },
    closeFs() {
      const id = this.fsId, home = this._fsHome;
      const c = id && document.getElementById(id);
      if (c && home) home.appendChild(c);   // move back BEFORE clearing state
      this.fsId = null; this._fsHome = null;
      this.$nextTick(() => this._charts[id]?.resize());
    },

    setActivityMode(m) {
      this.activityMode = { ft: m, sec: m, port: m };
      this.renderCharts();
    },

    clearAll() {
      this.selected = {};
      this.selectedCells = [];
      this.zoneData = null; this.lockedZone = null;
      this.error = '';
    },
    ruleKeys() { return Object.values(this.selected).filter(Boolean); },

    policyLabel() {
      const ks = this.ruleKeys();
      if (!ks.length) return 'no rules — backstop only';
      return ks.join(' + ');
    },

    async run(random = false) {
      this.loading = true; this.error = '';
      try {
        const body = {
          primary_metric: this.primaryMetric,
          secondary_metric: this.mode === '2f' ? this.secondaryMetric : null,
          entry_anchor: this.entryAnchor,
          rule_keys: this.ruleKeys(),
          n_bins: 20,
          label: random ? 'random entries' : null,
        };
        const r = await fetch('/api/factor-trades/run', {
          method: 'POST', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(body),
        });
        const d = await r.json();
        if (!r.ok || d.error) { this.error = d.error || ('HTTP ' + r.status); return; }
        this.runData = d;
        this.metric = d.primary_metric;
        this.secSelectedMetric = d.secondary_metric || '';
        // FactorCharts.hmCellBg reads heatmapData + _hmRange for the gradient.
        this.heatmapData = { grid: this.gridRows };
        window.FactorCharts._hmRecomputeRange(this);
        this.runs.push(d);
        this.currentIdx = this.runs.length - 1;
        this.selectedCells = [];
        this.zoneData = null;
      } catch (e) { this.error = String(e); }
      finally { this.loading = false; }
    },

    lockRun(i) {
      this.lockedIdx = (this.lockedIdx === i) ? -1 : i;
      this.lockedRun = this.lockedIdx >= 0 ? this.runs[this.lockedIdx] : null;
      this.lockedZone = null;
    },

    // ── Heatmap ──────────────────────────────────────────────────────────
    // The macro reads gridMeta.x_labels / .y_labels and iterates gridRows.
    get gridMeta() {
      const n = this.runData?.n_bins || 20;
      const lab = Array.from({ length: n }, (_, i) => 'B' + (i + 1));
      return { x_labels: lab, y_labels: this.runData?.mode === '1f' ? [''] : lab };
    },
    get gridRows() {
      const cur = this.runData?.grid || [];
      const lok = this.lockedRun?.grid || [];
      if (this.gridView === 'locked') return lok;
      if (this.gridView !== 'change') return cur;
      // Change = edited minus locked, cell-wise. Cells absent on either side
      // are null rather than treated as zero, so "no data" never reads as
      // "no change".
      return cur.map((row, iy) => row.map((c, ix) => {
        const l = lok?.[iy]?.[ix];
        if (!c || !l) return null;
        return { ...c, avg_ret: (c.avg_ret || 0) - (l.avg_ret || 0),
                 n: Math.min(c.n || 0, l.n || 0) };
      }));
    },
    // The heatmap macro calls these in Alpine scope. Delegating keeps the
    // gradient and the tooltip identical to every other heatmap on the site.
    groupMetricsByFamily(...a) { return window.FactorCharts.groupMetricsByFamily(this, ...a); },
    hmCellBg(...a)     { return window.FactorCharts.hmCellBg(this, ...a); },
    _hmCellTitle(...a) { return window.FactorCharts._hmCellTitle(this, ...a); },

    isCellSelected(ix, iy) {
      return this.selectedCells.some(c => c[0] === ix && c[1] === iy);
    },
    toggleCell(ix, iy) {
      const i = this.selectedCells.findIndex(c => c[0] === ix && c[1] === iy);
      if (i >= 0) this.selectedCells.splice(i, 1);
      else this.selectedCells.push([ix, iy]);
      this.loadZone();
    },

    async loadZone() {
      if (!this.runData || !this.selectedCells.length) { this.zoneData = null; return; }
      try {
        const body = {
          primary_metric: this.runData.primary_metric,
          secondary_metric: this.runData.secondary_metric,
          entry_anchor: this.runData.entry_anchor,
          rule_keys: this.runData.rules,
          n_bins: this.runData.n_bins,
          cells: this.selectedCells,
        };
        const r = await fetch('/api/factor-trades/zone', {
          method: 'POST', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(body),
        });
        const d = await r.json();
        if (!r.ok || d.error) { this.error = d.error || ('HTTP ' + r.status); return; }
        this.zoneData = d;
        this.secDetail = d;   // FactorCharts reads secDetail for defaults
        this.$nextTick(() => this.renderCharts());
      } catch (e) { this.error = String(e); }
    },

    // Sizing from the rail feeds the shared dollar-capped equity path, so the
    // $/trade and daily-cap inputs drive the same maths Recall uses.
    // Total / annualised / max-DD in dollars, off the same capped series the
    // equity pane draws — so the stat bar and the chart cannot disagree.
    _dollarStats(ds) {
      const eq = ds.equity || [];
      if (!eq.length) return null;
      const total = eq[eq.length - 1].value;
      let peak = 0, maxDD = 0;
      for (const p of eq) { if (p.value > peak) peak = p.value;
                            maxDD = Math.min(maxDD, p.value - peak); }
      const d0 = new Date(eq[0].date), d1 = new Date(eq[eq.length - 1].date);
      const years = Math.max((d1 - d0) / 31557600000, 1e-9);
      const annual = total / years;
      // Calmar is ANNUALISED return over max drawdown. Using total return
      // overstates it by the length of the window -- roughly 7.5x over a
      // 7.5-year sample -- and makes it incomparable with any published
      // Calmar figure.
      return { total_ret_usd: total, avg_annual_usd: annual,
               max_dd_usd: maxDD,
               calmar: maxDD < 0 ? (annual / Math.abs(maxDD)) : null };
    },

    renderCharts() {
      const FC = window.FactorCharts;
      if (!FC || !this.zoneData) return;
      this.equityDollarParams.ft = {
        perTrade: this.perTrade, dailyCap: this.dailyCap,
        maxStrike: this.maxStrike,
      };
      // Dollar stats for the three boxes that cannot come from the backend:
      // they depend on the rail's sizing, which is a client-side control.
      const ds = window.FactorCharts._computeDollarSeries(
        this, this.zoneData.combined_trades || [],
        this.perTrade, this.dailyCap, this.maxStrike);
      this.dollarStats = this._dollarStats(ds);
      try {
        FC._renderSecEquity(this, 'ft-equity', this.zoneData, true);
        FC._renderZoneYearly(this, 'ft-yearly', this.zoneData);
        FC._renderSecActivity(this, 'ft-activity-edited', this.zoneData);
        FC._renderSecBubble(this, 'ft-bubble-edited', this.zoneData);
        if (this.lockedZone) {
          FC._renderSecActivity(this, 'ft-activity-locked', this.lockedZone);
          FC._renderSecBubble(this, 'ft-bubble-locked', this.lockedZone);
        }
      } catch (e) { console.error('FactorCharts render failed', e); }
    },

    // ── Stat rows ────────────────────────────────────────────────────────
    // One row until a run is locked, then locked / edited / change. Never
    // two values in one box.
    statRows() {
      const src = this.zoneData || this.runData;
      if (!src) return [];
      const pct = v => v == null ? '—' : (v * 100).toFixed(3) + '%';
      const fmt$ = v => {
        if (v == null) return '—';
        const a = Math.abs(v), sg = v < 0 ? '-' : '';
        if (a >= 1e6) return sg + '$' + (a / 1e6).toFixed(2) + 'M';
        if (a >= 1e3) return sg + '$' + (a / 1e3).toFixed(1) + 'k';
        return sg + '$' + a.toFixed(0);
      };
      const mk = (key, label, s, win) => {
        if (!s) return null;
        return {
          key, label, window: win,
          nTickers: (s.n_tickers ?? 0).toLocaleString(),
          n: (s.n ?? 0).toLocaleString(),
          avgRet: pct(s.avg_ret), avgRetRaw: s.avg_ret ?? 0,
          median: pct(s.median),  medianRaw: s.median ?? 0,
          stdDev: pct(s.std_dev),
          p5: pct(s.p5),   p5Raw: s.p5 ?? 0,
          p95: pct(s.p95), p95Raw: s.p95 ?? 0,
          winRate: s.win_rate != null ? (s.win_rate * 100).toFixed(1) + '%' : '—',
          nWin: (s.n_win ?? 0).toLocaleString(),
          avgWin: pct(s.avg_win),   avgWinRaw: s.avg_win ?? 0,
          avgLoss: pct(s.avg_loss), avgLossRaw: s.avg_loss ?? 0,
          trdYr: (s.trades_per_year ?? 0).toFixed(1),
          // Calmar is null when there is no drawdown — render "—" rather
          // than an infinity dressed up as a great number.
          calmar: s.calmar != null ? s.calmar.toFixed(2) : '—', calmarRaw: s.calmar ?? 0,
          // Dollar figures come from the sizing controls via
          // _computeDollarSeries. Until that path is wired these read "—"
          // rather than a percent mislabelled as dollars, which is what made
          // Max DD render as -1826.885%.
          totalRet: s.total_ret_usd != null ? fmt$(s.total_ret_usd) : '—',
          totalRetRaw: s.total_ret_usd ?? 0,
          avgAnnRet: s.avg_annual_usd != null ? fmt$(s.avg_annual_usd) : '—',
          avgAnnRetRaw: s.avg_annual_usd ?? 0,
          maxDD: s.max_dd_usd != null ? fmt$(s.max_dd_usd) : '—',
          maxDDRaw: s.max_dd_usd ?? 0,
          avgHold: (s.avg_hold ?? 0).toFixed(2) + ' sess',
        };
      };
      const edited = mk('edited', this.lockedRun ? 'Edited' : 'Current',
                        { ...src.train, ...(this.dollarStats || {}) }, 'train');
      if (!this.lockedRun) return [edited].filter(Boolean);
      const lockedSrc = this.lockedZone || this.lockedRun;
      const locked = mk('locked', 'Locked', lockedSrc.train, 'train');
      const d = (a, b) => (a ?? 0) - (b ?? 0);
      const st = src.train, lt = lockedSrc.train;
      const diff = (locked && edited) ? {
        key: 'change', label: 'Change', window: 'train',
        nTickers: d(st?.n_tickers, lt?.n_tickers).toLocaleString(),
        n: d(st?.n, lt?.n).toLocaleString(),
        avgRet: pct(d(st?.avg_ret, lt?.avg_ret)), avgRetRaw: d(st?.avg_ret, lt?.avg_ret),
        median: pct(d(st?.median, lt?.median)), medianRaw: d(st?.median, lt?.median),
        stdDev: pct(d(st?.std_dev, lt?.std_dev)),
        p5: pct(d(st?.p5, lt?.p5)), p5Raw: d(st?.p5, lt?.p5),
        p95: pct(d(st?.p95, lt?.p95)), p95Raw: d(st?.p95, lt?.p95),
        winRate: (d(st?.win_rate, lt?.win_rate) * 100).toFixed(1) + '%',
        nWin: d(st?.n_win, lt?.n_win).toLocaleString(),
        avgWin: pct(d(st?.avg_win, lt?.avg_win)), avgWinRaw: d(st?.avg_win, lt?.avg_win),
        avgLoss: pct(d(st?.avg_loss, lt?.avg_loss)), avgLossRaw: d(st?.avg_loss, lt?.avg_loss),
        trdYr: d(st?.trades_per_year, lt?.trades_per_year).toFixed(1),
        calmar: (st?.calmar != null && lt?.calmar != null)
                ? d(st.calmar, lt.calmar).toFixed(2) : '—',
        calmarRaw: d(st?.calmar, lt?.calmar),
        totalRet: '—', totalRetRaw: 0, avgAnnRet: '—', avgAnnRetRaw: 0,
        maxDD: '—', maxDDRaw: 0,
      } : null;
      return [locked, edited, diff].filter(Boolean);
    },

    exportCsv() {
      const t = this.zoneData?.combined_trades || [];
      if (!t.length) return;
      const head = ['ticker', 'trade_date', 'entry_price', 'ret_pct', 'exit_bar', 'exit_rule', 'window'];
      const rows = t.map(x => [x.ticker, x.trade_date,
        x.entry_price ?? '', (x.ret * 100).toFixed(6), x.exit_bar,
        x.exit_rule, x.window].join(','));
      const blob = new Blob([[head.join(','), ...rows].join(String.fromCharCode(10))],
                            { type: 'text/csv' });
      const a = document.createElement('a');
      a.href = URL.createObjectURL(blob);
      a.download = `factor_trades_${this.runData?.primary_metric || 'run'}.csv`;
      a.click(); URL.revokeObjectURL(a.href);
    },
  }));
});
