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
    // Every pane here is train/test, so the cutoff marker is always wanted.
    get cutoffLineDate() { return this.cutoffDate; },
    // Fixed x-domain for the three time-series panes, both windows. Read by
    // FactorCharts._seriesAxis; pages that omit it keep data-driven bounds.
    seriesAxis: { from: '2019-01-01', to: null },
    mode: '2f', primaryMetric: '', secondaryMetric: '', entryAnchor: 'open',
    selected: {},                 // family -> rule_key (absent = family off)
    perTrade: 2000, dailyCap: 10000, maxStrike: 1000,
    loading: false, error: '',
    runs: [], currentIdx: -1, lockedIdx: -1,
    runData: null, lockedRun: null, zoneData: null, lockedZone: null,
    // Page-level window. TRAIN by default: the workflow is to iterate on
    // train and treat switching to test as a decision, not a default view.
    window: 'train',
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
    // Capital by default: the question this page asks is what a policy ties
    // up, not how many tickets it writes.
    activityMode: { ft: 'capital', sec: 'trades', port: 'trades' },
    dedupeConc: { primary: false, sec: false, corr: false, port: false },
    secBubbleMinN: 0,
    // 'ft' is this page's key (FactorCharts._equityModeKey maps ft-*
    // canvases to it). Dollar mode is the point of this page: every
    // axis is dollars derived from the rail's sizing controls.
    equityAggMode:      { ft: 'dollar_capped', zone: 'dollar_capped', sec: 'daily',
                          recall: 'dollar_capped', port: 'dollar_capped' },
    equityDollarParams: { ft:   { perTrade: 2000, dailyCap: 10000 },
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

    // A family whose rules carry TWO parameters (trail: activation x trail
    // distance) renders as two stacked dropdowns instead of one list of every
    // pair. Derived from the params, not hardcoded to `trail`, so any family
    // the registry grows with two dimensions splits automatically.
    famDims(f) {
      const keys = new Set();
      for (const r of f.rules || []) for (const k of Object.keys(r.params || {})) keys.add(k);
      return [...keys].sort();
    },
    isSplitFamily(f) { return this.famDims(f).length >= 2; },
    // Distinct values for one dimension, in numeric order.
    famVals(f, dim) {
      const vs = new Set();
      for (const r of f.rules || []) {
        const v = (r.params || {})[dim];
        if (v != null) vs.add(v);
      }
      return [...vs].sort((a, b) => (+a) - (+b));
    },
    // Which value of `dim` the currently-selected rule uses.
    famSel(f, dim) {
      const key = this.selected[f.family];
      const r = (f.rules || []).find(x => x.rule_key === key);
      return r ? (r.params || {})[dim] : null;
    },
    // Pick the rule matching the requested value on `dim`, holding every
    // other dimension at its current value where possible — that is the
    // whole point of splitting the control.
    setFamDim(f, dim, val) {
      const dims = this.famDims(f);
      const cur = {};
      for (const d of dims) cur[d] = this.famSel(f, d);
      cur[dim] = val;
      let hit = (f.rules || []).find(r =>
        dims.every(d => String((r.params || {})[d]) === String(cur[d])));
      // No exact pair exists (the grid is not always complete) — fall back to
      // matching the dimension the user just changed.
      if (!hit) hit = (f.rules || []).find(r => String((r.params || {})[dim]) === String(val));
      if (hit) { this.selected[f.family] = hit.rule_key; this.selected = { ...this.selected }; }
    },

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

    // _hmRange drives hmCellBg's gradient. It was only recomputed at run
    // time, so switching window or grid view left the previous window's
    // range in place -- which is why cells lost their colours.
    _refreshGrid() {
      this.heatmapData = { grid: this.gridRows };
      window.FactorCharts._hmRecomputeRange(this);
    },
    setGridView(v) { this.gridView = v; this._refreshGrid(); },

    setWindow(w) {
      if (w === this.window) return;
      this.window = w;
      this._refreshGrid();
      // Both the run and the zone are window-scoped server-side, so the
      // whole page has to be refetched rather than re-filtered client-side.
      if (this.runData) this.run();
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
      const prev = this.runData;
      try {
        const body = {
          primary_metric: this.primaryMetric,
          secondary_metric: this.mode === '2f' ? this.secondaryMetric : null,
          entry_anchor: this.entryAnchor,
          rule_keys: this.ruleKeys(),
          n_bins: 20,
          max_strike: this.maxStrike,
          window: this.window,
          label: random ? 'random entries' : null,
        };
        const r = await fetch('/api/factor-trades/run', {
          method: 'POST', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(body),
        });
        const d = await r.json();
        if (!r.ok || d.error) { this.error = d.error || ('HTTP ' + r.status); return; }
        // Selection survives a re-run when the METRIC PAIR is unchanged: the
        // bins are the same bins, so the cells still mean the same thing.
        // Change a metric and they do not, so it is cleared. Re-selecting on
        // every exit-parameter tweak was the main friction in the loop.
        const samePair = prev
          && prev.primary_metric === d.primary_metric
          && (prev.secondary_metric || null) === (d.secondary_metric || null)
          && prev.n_bins === d.n_bins;
        this.runData = d;
        this.metric = d.primary_metric;
        this.secSelectedMetric = d.secondary_metric || '';
        // FactorCharts.hmCellBg reads heatmapData + _hmRange for the gradient.
        this._refreshGrid();
        this.runs.push(d);
        this.currentIdx = this.runs.length - 1;
        if (!samePair) { this.selectedCells = []; this.zoneData = null; }
        this._refreshGrid();
        this.$nextTick(() => this._scrollRunsRight());
        if (samePair && this.selectedCells.length) {
          // Same cells, new policy — refetch the zone rather than leaving the
          // panes showing the previous run's trades.
          this.loadZone();
        }
      } catch (e) { this.error = String(e); }
      finally { this.loading = false; }
    },

    _scrollRunsRight() {
      // New runs append, so the strip must anchor RIGHT or Current scrolls
      // out of view exactly when it becomes the thing you want to see.
      const el = document.getElementById('ft-runs');
      if (el) el.scrollLeft = el.scrollWidth;
    },

    deleteRun(i) {
      this.runs.splice(i, 1);
      // Indices shift; a stale lockedIdx would silently point at a different
      // run than the one that was locked.
      if (this.lockedIdx === i) { this.lockedIdx = -1; this.lockedRun = null; this.lockedZone = null; }
      else if (this.lockedIdx > i) this.lockedIdx -= 1;
      if (this.currentIdx === i) this.currentIdx = Math.min(i, this.runs.length - 1);
      else if (this.currentIdx > i) this.currentIdx -= 1;
      this.$nextTick(() => this._scrollRunsRight());
    },

    async lockRun(i) {
      this.lockedIdx = (this.lockedIdx === i) ? -1 : i;
      this.lockedRun = this.lockedIdx >= 0 ? this.runs[this.lockedIdx] : null;
      this.lockedZone = null;
      this._refreshGrid();
      if (!this.lockedRun || !this.selectedCells.length) return;
      // Fetch the locked run's OWN zone series, using ITS parameters and ITS
      // window -- not the rail's current state. A locked TRAIN run compared
      // against a TEST run would otherwise cross populations with nothing on
      // screen saying so.
      try {
        const r = this.lockedRun;
        const resp = await fetch('/api/factor-trades/zone', {
          method: 'POST', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            primary_metric: r.primary_metric, secondary_metric: r.secondary_metric,
            entry_anchor: r.entry_anchor, rule_keys: r.rules,
            n_bins: r.n_bins, max_strike: r.max_strike,
            window: r.window || 'train', cells: this.selectedCells,
          }),
        });
        const d = await resp.json();
        if (resp.ok && !d.error) { this.lockedZone = d; this.renderCharts(); }
      } catch (e) { console.error('[factor-trades] locked zone fetch failed', e); }
    },

    // ── Heatmap ──────────────────────────────────────────────────────────
    // The macro reads gridMeta.x_labels / .y_labels and iterates gridRows.
    get gridMeta() {
      const n = this.runData?.n_bins || 20;
      const lab = Array.from({ length: n }, (_, i) => 'B' + (i + 1));
      return { x_labels: lab, y_labels: this.runData?.mode === '1f' ? [''] : lab };
    },
    // The grid carries BOTH windows per cell: avg_ret/n are train, and
    // test_avg/test_n are test. Selection happens on train, so that is the
    // default, but the toggle switches which face is shown rather than
    // pinning it. Mapping here (not server-side) keeps one grid payload.
    _faceOf(grid) {
      if (this.window !== 'test') return grid;
      return (grid || []).map(row => (row || []).map(c => c && ({
        ...c, avg_ret: c.test_avg ?? 0, n: c.test_n ?? 0,
      })));
    },
    get gridRows() {
      const cur = this._faceOf(this.runData?.grid || []);
      const lok = this._faceOf(this.lockedRun?.grid || []);
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
          // Must match the run's population or the zone is a different trade set.
          max_strike: this.runData.max_strike ?? this.maxStrike,
          window: this.window,
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


    // Exit reasons. Page-local rather than in FactorCharts: no other page has
    // an exit policy, so there is no second consumer to keep in step, and a
    // shared module should not carry a chart only one page can use.
    //
    // Horizontal bars, one per rule that actually closed a trade, ordered by
    // share. The backstop is drawn in a distinct colour because it means the
    // opposite of the others: a bar for fixed_stop__2 is that stop working,
    // while a bar for the auto-appended max_days__20 is every selected rule
    // FAILING to fire and the horizon catching the trade by default. Same
    // chart, opposite readings — see horizon_auto_added on the run card.
    // Two sub-charts, not one dual-axis pane. The shared-axis version put
    // grey avg-hold bars on the same rows as the coloured share bars with
    // scales at opposite edges, and there was no way to tell by looking
    // which scale a given bar was drawn against.
    _renderExitReasons() {
      const el   = document.getElementById('ft-reasons');
      const elH  = document.getElementById('ft-hold');
      const rows = this.zoneData?.exit_reasons || [];
      for (const k of ['ft-reasons', 'ft-hold']) {
        if (this._charts[k]) { this._charts[k].destroy(); delete this._charts[k]; }
      }
      if (!el || !rows.length) return;
      const labels = rows.map(r => r.label);
      const data   = rows.map(r => +(r.frac * 100).toFixed(2));
      // Colour by side: targets blue, stops pink, time grey, trend teal.
      // The backstop keeps amber and overrides its side, because it means
      // the opposite of a rule working -- nothing fired and the horizon
      // caught the trade.
      const SIDE_COLOR = {
        target: 'rgba(52,152,219,0.78)',
        stop:   'rgba(232,67,147,0.72)',
        time:   'rgba(150,150,150,0.60)',
        trend:  'rgba(26,188,156,0.70)',
      };
      const colors = rows.map(r => r.is_backstop
        ? 'rgba(224,176,102,0.80)'
        : (SIDE_COLOR[r.side] || 'rgba(150,150,150,0.60)'));
      this._charts['ft-reasons'] = new Chart(el.getContext('2d'), {
        type: 'bar',
        data: { labels, datasets: [
          { data, backgroundColor: colors, borderColor: colors, borderWidth: 1 },
        ] },
        options: {
          indexAxis: 'y',
          responsive: true, maintainAspectRatio: false, animation: false,
          plugins: {
            legend: { display: false },
            tooltip: {
              backgroundColor: 'rgba(20,20,20,.95)', borderColor: '#444', borderWidth: 1,
              callbacks: {
                label: (c) => {
                  const r = rows[c.dataIndex];
                  return [`${r.frac != null ? (r.frac * 100).toFixed(1) : '—'}% of trades`,
                          `avg hold ${(r.avg_hold ?? 0).toFixed(2)} sess`,
                          `n = ${(r.n ?? 0).toLocaleString()}`,
                          r.is_backstop
                            ? 'backstop — no selected rule fired'
                            : `${r.side} · ${r.rule_key}`];
                },
              },
            },
          },
          scales: {
            x: { ticks: { color: '#888', font: { size: 9 }, callback: v => v + '%' },
                 grid: { color: '#222' }, beginAtZero: true },
            y: { ticks: { color: '#aaa', font: { size: 10 } },
                 grid: { display: false } },
          },
        },
      });

      // Avg hold, same rows in the same order. Labels are repeated rather
      // than shared down a gutter so this chart is readable on its own --
      // a stop firing at 1.2 sessions against a backstop at 20 is the
      // shape worth seeing, and it should not need a row-count across the
      // pane to attribute it.
      if (!elH) return;
      this._charts['ft-hold'] = new Chart(elH.getContext('2d'), {
        type: 'bar',
        data: { labels, datasets: [{
          data: rows.map(r => +(r.avg_hold ?? 0).toFixed(2)),
          // Same side colours, held back to a wash so the eye reads this
          // pane as the quieter companion to the share pane.
          backgroundColor: colors.map(c => c.replace(/[\d.]+\)$/, '0.34)')),
          borderColor: colors, borderWidth: 1,
        }] },
        options: {
          indexAxis: 'y',
          responsive: true, maintainAspectRatio: false, animation: false,
          plugins: {
            legend: { display: false },
            tooltip: {
              backgroundColor: 'rgba(20,20,20,.95)', borderColor: '#444', borderWidth: 1,
              callbacks: {
                label: (c) => {
                  const r = rows[c.dataIndex];
                  return [`avg hold ${(r.avg_hold ?? 0).toFixed(2)} sessions`,
                          `n = ${(r.n ?? 0).toLocaleString()}`];
                },
              },
            },
          },
          scales: {
            x: { ticks: { color: '#888', font: { size: 9 }, callback: v => v + 'd' },
                 grid: { color: '#222' }, beginAtZero: true },
            y: { ticks: { color: '#aaa', font: { size: 10 } },
                 grid: { display: false } },
          },
        },
      });
    },


    // Avg return bucketed by ENTRY PRICE. Test window only, matching the
    // pane title. Buckets are fixed rather than quantile-derived so the
    // x-axis means the same thing across runs — the question is "does this
    // edge live in cheap or expensive names", and a bucket that moves with
    // the data cannot answer it.
    _priceBuckets: [[0,25],[25,50],[50,100],[100,150],[150,200],[200,300],
                    [300,400],[400,500],[500,750],[750,1000],[1000,Infinity]],
    // Distribution of trades by P&L bucket. Server-computed from
    // window_trades for the same reason the price bins are.
    _renderPnlDist() {
      const el = document.getElementById('ft-pnldist');
      if (this._charts['ft-pnldist']) { this._charts['ft-pnldist'].destroy(); delete this._charts['ft-pnldist']; }
      const b = this.zoneData?.pnl_dist || [];
      if (!el || !b.length) return;
      // Same bar language as Annual P&L and the price bins: alpha = n.
      // This pane is single-window by construction (server-computed from
      // window_trades), so nothing here is ever hatched.
      const FC = window.FactorCharts;
      const mx = Math.max(...b.map(x => x.n), 1);
      const paints = b.map(x => FC._barPaint(
        FC._barRgb(x.n ? (x.lo < 0 ? 'neg' : 'pos') : 'none'),
        (x.n || 0) / mx, false));
      this._charts['ft-pnldist'] = new Chart(el.getContext('2d'), {
        type: 'bar',
        data: { labels: b.map(x => x.label), datasets: [{
          data: b.map(x => x.n),
          backgroundColor: paints.map(p => p.background),
          borderColor:     paints.map(p => p.border),
          borderWidth:     paints.map(p => p.borderWidth),
        }] },
        options: {
          responsive: true, maintainAspectRatio: false, animation: false,
          plugins: { legend: { display: false },
            tooltip: { backgroundColor: 'rgba(20,20,20,.95)', borderColor: '#444', borderWidth: 1,
              callbacks: { label: (c) => `${c.raw.toLocaleString()} trades` } } },
          scales: {
            x: { ticks: { color: '#888', font: { size: 9 }, maxRotation: 60 }, grid: { display: false } },
            y: { ticks: { color: '#888', font: { size: 9 } }, grid: { color: '#222' }, beginAtZero: true },
          },
        },
      });
    },

    _renderPriceBins() {
      const el = document.getElementById('ft-pricebins');
      if (this._charts['ft-pricebins']) { this._charts['ft-pricebins'].destroy(); delete this._charts['ft-pricebins']; }
      // Server-computed from window_trades, so this pane cannot pick up the
      // wider series population by accident.
      const bins = this.zoneData?.price_bins || [];
      if (!el || !bins.length) return;
      const lbl = bins.map(b => b.label);
      const avg = bins.map(b => b.avg_ret == null ? null : +(b.avg_ret * 100).toFixed(3));
      // Same bar language as Annual P&L: alpha = n, borders on every bar.
      // Single-window pane, so nothing is hatched.
      const FC = window.FactorCharts;
      const mx = Math.max(...bins.map(b => b.n), 1);
      const paints = avg.map((v, i) => FC._barPaint(
        FC._barRgb(v == null ? 'none' : (v >= 0 ? 'pos' : 'neg')),
        (bins[i].n || 0) / mx, false));
      this._charts['ft-pricebins'] = new Chart(el.getContext('2d'), {
        type: 'bar',
        data: { labels: lbl, datasets: [{
          data: avg,
          backgroundColor: paints.map(p => p.background),
          borderColor:     paints.map(p => p.border),
          borderWidth:     paints.map(p => p.borderWidth),
        }] },
        options: {
          responsive: true, maintainAspectRatio: false, animation: false,
          plugins: {
            legend: { display: false },
            tooltip: { backgroundColor: 'rgba(20,20,20,.95)', borderColor: '#444', borderWidth: 1,
              callbacks: { label: (c) => {
                const b = bins[c.dataIndex];
                return b.n ? [`avg ${(b.avg_ret * 100).toFixed(3)}%`, `n = ${b.n.toLocaleString()}`]
                           : ['no trades in this bucket'];
              } } },
          },
          scales: {
            x: { ticks: { color: '#888', font: { size: 9 }, maxRotation: 60,
                          callback: (v, i) => lbl[i] + (bins[i].n ? `  n=${bins[i].n}` : '') },
                 grid: { display: false } },
            y: { ticks: { color: '#888', font: { size: 9 }, callback: v => v + '%' },
                 grid: { color: '#222' } },
          },
        },
      });
    },

    renderCharts() {
      const FC = window.FactorCharts;
      if (!FC || !this.zoneData) return;
      this.equityDollarParams.ft = {
        perTrade: this.perTrade, dailyCap: this.dailyCap,
      };
      // Dollar stats for the three boxes that cannot come from the backend:
      // they depend on the rail's sizing, which is a client-side control.
      const ds = window.FactorCharts._computeDollarSeries(
        this, this.zoneData.window_trades || [],
        this.perTrade, this.dailyCap);
      this.dollarStats = this._dollarStats(ds);
      // The locked row's dollar boxes were blank because these were only ever
      // computed for the edited zone. Same function, same sizing inputs, the
      // locked population.
      this.lockedDollarStats = this.lockedZone
        ? this._dollarStats(window.FactorCharts._computeDollarSeries(
            this, this.lockedZone.window_trades || [], this.perTrade, this.dailyCap))
        : null;
      // In TRAIN the two populations are identical by construction, so the
      // equity curve's endpoint must equal Total Ret. If it does not, the
      // arrays have been crossed somewhere. Console-loud rather than silent:
      // a mismatch here is a data-integrity bug, not a rounding artefact.
      if (this.window === 'train' && this.dollarStats) {
        const seriesDs = window.FactorCharts._computeDollarSeries(
          this, this.zoneData.series_trades || [], this.perTrade, this.dailyCap);
        const endPt = seriesDs.equity.length
          ? seriesDs.equity[seriesDs.equity.length - 1].value : 0;
        const diff = Math.abs(endPt - this.dollarStats.total_ret_usd);
        if (diff > 0.01) {
          console.error('[factor-trades] TRAIN invariant violated: equity endpoint',
            endPt, '!= Total Ret', this.dollarStats.total_ret_usd,
            '- series_trades and window_trades have been crossed');
        } else {
          // Logged on success too. A silent assertion is indistinguishable
          // from one that never ran, and "these two numbers look close
          // enough" is exactly the judgement it exists to replace.
          console.info('[factor-trades] TRAIN invariant ok: equity endpoint',
            endPt.toFixed(2), '== Total Ret', this.dollarStats.total_ret_usd.toFixed(2));
        }
      }
      // THE one place the two populations meet. The three time-series panes
      // get series_trades under the name FactorCharts expects; every other
      // consumer reads a server-computed single-window aggregate and never
      // sees a trade list at all. One line to audit instead of a convention
      // to remember.
      const seriesView = { ...this.zoneData,
                           combined_trades: this.zoneData.series_trades
                                            || this.zoneData.window_trades || [] };
      // Each pane in its own try. One shared try/catch meant a throw in the
      // second renderer silently blanked the four after it -- one bug
      // presenting as five broken panes, with nothing to say which was the
      // cause. Now a failure costs exactly its own pane and names itself.
      const panes = [
        ['equity',          () => FC._renderSecEquity(this, 'ft-equity', seriesView, true)],
        ['annual P&L',      () => FC._renderZoneYearly(this, 'ft-yearly', seriesView)],
        ['activity',        () => FC._renderSecActivity(this, 'ft-activity-edited', seriesView)],
        ['ticker breakdown',() => FC._renderSecBubble(this, 'ft-bubble-edited', this.zoneData)],
        ['exit reasons',    () => this._renderExitReasons()],
        ['price bins',      () => this._renderPriceBins()],
        ['P&L distribution',() => this._renderPnlDist()],
      ];
      if (this.lockedZone) {
        panes.push(['activity (locked)', () => FC._renderSecActivity(this, 'ft-activity-locked', this.lockedZone)]);
        panes.push(['ticker (locked)',   () => FC._renderSecBubble(this, 'ft-bubble-locked', this.lockedZone)]);
      }
      for (const [name, fn] of panes) {
        try { fn(); }
        catch (e) { console.error(`[factor-trades] ${name} pane failed to render`, e); }
      }
    },

    // ── Stat rows ────────────────────────────────────────────────────────
    // One row until a run is locked, then locked / edited / change. Never
    // two values in one box.
    statRows() {
      const src = this.zoneData || this.runData;
      if (!src) return [];
      const pct = v => v == null ? '—' : (v * 100).toFixed(3) + '%';
      // Annualised dollar return over dollar max drawdown. Blank unless BOTH
      // inputs are present, so Calmar can never appear beside empty boxes.
      const calmarRawOf = (s) => {
        const a = s?.avg_annual_usd, d = s?.max_dd_usd;
        return (a != null && d != null && d < 0) ? (a / Math.abs(d)) : null;
      };
      const calmarOf = (s) => {
        const v = calmarRawOf(s);
        return v == null ? '—' : v.toFixed(2);
      };
      const fmt$ = v => {
        if (v == null) return '—';
        const a = Math.abs(v), sg = v < 0 ? '-' : '';
        if (a >= 1e6) return sg + '$' + (a / 1e6).toFixed(2) + 'M';
        if (a >= 1e3) return sg + '$' + (a / 1e3).toFixed(1) + 'k';
        return sg + '$' + a.toFixed(0);
      };
      // Two explicit sources rather than a spread. A merge silently supplies
      // whatever the right-hand side happens not to define -- which is how a
      // server-side percent Calmar ended up displayed beside blank dollar
      // boxes. Server stats and dollar stats are now read from their own
      // argument, so a missing dollar figure reads blank instead of
      // inheriting a different definition of the same name.
      const mk = (key, label, s, $d, win) => {
        if (!s) return null;
        return {
          key, label, window: win,
          nTickers: (s.n_tickers ?? 0).toLocaleString(),
          effTickers: (s.eff_tickers ?? 0).toFixed(1),
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
          // Calmar is derived HERE from the dollar figures shown beside it,
          // never from the server's percent-based calmar. Those are two
          // different definitions, and falling back to the other one produced
          // a populated Calmar sitting next to a blank Max DD -- a number
          // whose stated inputs were empty, which is worse than no number.
          calmar: calmarOf($d), calmarRaw: calmarRawOf($d),
          // Dollar figures come from the sizing controls via
          // _computeDollarSeries. Until that path is wired these read "—"
          // rather than a percent mislabelled as dollars, which is what made
          // Max DD render as -1826.885%.
          totalRet: $d?.total_ret_usd != null ? fmt$($d.total_ret_usd) : '—',
          totalRetRaw: $d?.total_ret_usd ?? 0,
          avgAnnRet: $d?.avg_annual_usd != null ? fmt$($d.avg_annual_usd) : '—',
          avgAnnRetRaw: $d?.avg_annual_usd ?? 0,
          maxDD: $d?.max_dd_usd != null ? fmt$($d.max_dd_usd) : '—',
          maxDDRaw: $d?.max_dd_usd ?? 0,
          avgHold: (s.avg_hold ?? 0).toFixed(2) + ' sess',
          // Same figure as Avg Hold, kept on the bar as well as the run
          // card: it is the one stat you check against every policy tweak.
          avgDit: (s.avg_hold ?? 0).toFixed(2) + ' sess',
        };
      };
      // The stat bar is always the TEST window: it is the verdict, and a
      // number that silently switched windows would be the worst kind of
      // wrong. Selection happens on the heatmap, which is train.
      const edited = mk('edited', this.lockedRun ? 'Edited' : 'Current',
                        src[this.window], this.dollarStats, this.window);
      if (!this.lockedRun) return [edited].filter(Boolean);
      const lockedSrc = this.lockedZone || this.lockedRun;
      const locked = mk('locked', 'Locked',
                        lockedSrc[this.window], this.lockedDollarStats, this.window);
      const d = (a, b) => (a ?? 0) - (b ?? 0);
      const E = this.dollarStats, L = this.lockedDollarStats;
      const dd = (k) => (E?.[k] != null && L?.[k] != null) ? (E[k] - L[k]) : null;
      const cE = calmarRawOf(E), cL = calmarRawOf(L);
      const st = src[this.window], lt = lockedSrc[this.window];
      const diff = (locked && edited) ? {
        key: 'change', label: 'Change', window: this.window,
        nTickers: d(st?.n_tickers, lt?.n_tickers).toLocaleString(),
        effTickers: d(st?.eff_tickers, lt?.eff_tickers).toFixed(1),
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
        // Dollar deltas, and a Calmar delta only when both sides have one.
        totalRet: fmt$(dd('total_ret_usd')), totalRetRaw: dd('total_ret_usd') ?? 0,
        avgAnnRet: fmt$(dd('avg_annual_usd')), avgAnnRetRaw: dd('avg_annual_usd') ?? 0,
        maxDD: fmt$(dd('max_dd_usd')), maxDDRaw: dd('max_dd_usd') ?? 0,
        calmar: (cE != null && cL != null) ? (cE - cL).toFixed(2) : '—',
        calmarRaw: (cE != null && cL != null) ? (cE - cL) : 0,
        avgDit: d(st?.avg_hold, lt?.avg_hold).toFixed(2) + ' sess',
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
