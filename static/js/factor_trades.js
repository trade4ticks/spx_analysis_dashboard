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
    mode: '2f', primaryMetric: '', secondaryMetric: '', entryAnchor: 'open',
    selected: {},                 // family -> rule_key (absent = family off)
    perTrade: 2000, dailyCap: 10000, maxConcurrent: 5,
    loading: false, error: '',
    runs: [], currentIdx: -1, lockedIdx: -1,
    runData: null, lockedRun: null, zoneData: null, lockedZone: null,
    gridView: 'edited', showDD: true,
    selectedCells: [],            // [[bp, bs], ...]

    // ── FactorCharts contract ────────────────────────────────────────────
    // window.FactorCharts.* reads these off the component by name. They must
    // keep these exact names; renaming any of them silently breaks a chart
    // rather than raising. Defaults mirror Recall's so the panes look and
    // behave identically out of the box.
    _charts: {},
    secDetail: null, zoneData_: null, data: null,
    activityMode: 'trades',
    dedupeConc: { primary: false, sec: false, corr: false, port: false },
    secBubbleMinN: 0,
    equityAggMode:      { zone: 'dollar_capped', sec: 'daily', recall: 'dollar_capped', port: 'dollar_capped' },
    equityDollarParams: { zone: { perTrade: 2000, dailyCap: 10000 },
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
    renderCharts() {
      const FC = window.FactorCharts;
      if (!FC || !this.zoneData) return;
      for (const k of Object.keys(this.equityDollarParams)) {
        this.equityDollarParams[k] = { perTrade: this.perTrade, dailyCap: this.dailyCap };
      }
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
      const mk = (key, label, s) => {
        if (!s) return null;
        const dd = s.max_dd ?? null, cal = s.calmar ?? null;
        return {
          key, label,
          n: (s.n ?? 0).toLocaleString(),
          avgRet: ((s.avg_ret ?? 0) * 100).toFixed(3) + '%', avgRetRaw: s.avg_ret ?? 0,
          winRate: s.win_rate != null ? (s.win_rate * 100).toFixed(1) + '%' : '—',
          calmar: cal != null ? cal.toFixed(2) : '—', calmarRaw: cal ?? 0,
          maxDD: dd != null ? (dd * 100).toFixed(2) + '%' : '—', maxDDRaw: dd ?? 0,
          avgHold: (s.avg_hold ?? 0).toFixed(1) + 'd',
        };
      };
      const edited = mk('edited', this.lockedRun ? 'Edited' : 'Current', src.train);
      if (!this.lockedRun) return [edited].filter(Boolean);
      const lockedSrc = this.lockedZone || this.lockedRun;
      const locked = mk('locked', 'Locked', lockedSrc.train);
      const diff = (locked && edited) ? {
        key: 'change', label: 'Change',
        n: ((src.train?.n ?? 0) - (lockedSrc.train?.n ?? 0)).toLocaleString(),
        avgRet: (((src.train?.avg_ret ?? 0) - (lockedSrc.train?.avg_ret ?? 0)) * 100).toFixed(3) + '%',
        avgRetRaw: (src.train?.avg_ret ?? 0) - (lockedSrc.train?.avg_ret ?? 0),
        winRate: '—', calmar: '—', calmarRaw: 0, maxDD: '—', maxDDRaw: 0,
        avgHold: (((src.train?.avg_hold ?? 0) - (lockedSrc.train?.avg_hold ?? 0))).toFixed(1) + 'd',
      } : null;
      return [locked, edited, diff].filter(Boolean);
    },

    exportCsv() {
      const t = this.zoneData?.combined_trades || [];
      if (!t.length) return;
      const head = ['ticker', 'trade_date', 'ret_pct', 'exit_bar', 'exit_rule', 'window'];
      const rows = t.map(x => [x.ticker, x.trade_date,
        (x.ret * 100).toFixed(6), x.exit_bar, x.exit_rule, x.window].join(','));
      const blob = new Blob([[head.join(','), ...rows].join(String.fromCharCode(10))],
                            { type: 'text/csv' });
      const a = document.createElement('a');
      a.href = URL.createObjectURL(blob);
      a.download = `factor_trades_${this.runData?.primary_metric || 'run'}.csv`;
      a.click(); URL.revokeObjectURL(a.href);
    },
  }));
});
