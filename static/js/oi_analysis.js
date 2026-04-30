'use strict';

document.addEventListener('alpine:init', () => {
  Alpine.data('oiAnalysis', () => ({
    // Selectors
    tickers: [], features: [], outcomes: [],
    ticker: '', metric: '', outcome: '',
    selectedDeciles: new Set([1, 10]),
    equityMode: 'concurrent',  // 'concurrent' | 'non_overlapping'

    // Data
    data: null,
    loading: false,
    error: null,
    _charts: {},

    async init() {
      const [tkRes, colRes] = await Promise.all([
        fetch('/api/oi-analysis/tickers'),
        fetch('/api/oi-analysis/columns'),
      ]);
      if (tkRes.ok) {
        this.tickers = ['ALL', ...(await tkRes.json())];
        this.ticker = this.tickers[1] || 'ALL';
      }
      if (colRes.ok) {
        const cols = await colRes.json();
        this.features = cols.features || [];
        this.outcomes = cols.outcomes || [];
        if (this.features.length) this.metric = this.features[0];
        if (this.outcomes.length) this.outcome = this.outcomes[0];
      }
    },

    async loadAnalysis() {
      if (!this.ticker || !this.metric || !this.outcome) return;
      this.loading = true;
      this.error = null;
      this._destroyCharts();
      try {
        const r = await fetch(
          `/api/oi-analysis/analyze?ticker=${encodeURIComponent(this.ticker)}`
          + `&metric=${encodeURIComponent(this.metric)}`
          + `&outcome=${encodeURIComponent(this.outcome)}`);
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        this.data = await r.json();
        if (this.data.error) { this.error = this.data.error; return; }
        await this.$nextTick();
        await this.$nextTick();
        setTimeout(() => this._renderCharts(), 80);
      } catch (e) {
        this.error = e.message;
      } finally {
        this.loading = false;
      }
    },

    toggleDecile(d) {
      if (this.selectedDeciles.has(d)) {
        if (this.selectedDeciles.size > 1) this.selectedDeciles.delete(d);
      } else {
        this.selectedDeciles.add(d);
      }
      this.selectedDeciles = new Set(this.selectedDeciles); // trigger reactivity
      this._renderEquity();
    },

    selectAllDeciles() { this.selectedDeciles = new Set([1,2,3,4,5,6,7,8,9,10]); this._renderEquity(); },
    selectExtremes()   { this.selectedDeciles = new Set([1,10]); this._renderEquity(); },

    isDecileSelected(d) { return this.selectedDeciles.has(d); },

    setEquityMode(m) { this.equityMode = m; this._renderEquity(); },

    _destroyCharts() {
      Object.values(this._charts).forEach(c => c.destroy());
      this._charts = {};
    },

    _darkScales() {
      return {
        x: { ticks:{color:'#888',font:{size:9},maxRotation:45}, grid:{color:'rgba(255,255,255,0.05)'}, border:{color:'transparent'} },
        y: { ticks:{color:'#888',font:{size:9}}, grid:{color:'rgba(255,255,255,0.05)'}, border:{color:'transparent'} },
      };
    },

    _renderCharts() {
      if (!this.data) return;
      this._renderDecileBar();
      this._renderEquity();
      this._renderYearly();
      this._renderYearlyConsistency();
    },

    _renderDecileBar() {
      const el = document.getElementById('chart-decile');
      if (!el) return;
      if (this._charts['decile']) this._charts['decile'].destroy();

      const stats = (this.data.decile_stats || []).filter(d => d);
      const avgs = stats.map(d => d.avg_ret * 100);
      const self = this;

      this._charts['decile'] = new Chart(el, {
        type: 'bar',
        data: {
          labels: stats.map(d => 'D' + d.bucket),
          datasets: [{
            data: avgs,
            backgroundColor: stats.map(d =>
              self.selectedDeciles.has(d.bucket)
                ? (d.avg_ret >= 0 ? '#3498db' : '#e84393')
                : 'rgba(100,100,100,0.3)'),
            borderColor: stats.map(d =>
              self.selectedDeciles.has(d.bucket) ? '#fff' : 'transparent'),
            borderWidth: stats.map(d => self.selectedDeciles.has(d.bucket) ? 1 : 0),
          }],
        },
        options: {
          responsive: true, maintainAspectRatio: false, animation: false,
          onClick: (e, elements) => {
            if (elements.length) {
              const idx = elements[0].index;
              self.toggleDecile(stats[idx].bucket);
              self._renderDecileBar(); // re-render colors
            }
          },
          plugins: {
            legend: { display: false },
            tooltip: {
              backgroundColor: 'rgba(20,20,20,0.95)',
              borderColor: '#444', borderWidth: 1,
              callbacks: {
                label: ctx => {
                  const d = stats[ctx.dataIndex];
                  return [
                    `Avg: ${(d.avg_ret*100).toFixed(3)}%`,
                    `WR: ${(d.win_rate*100).toFixed(1)}%`,
                    `Sharpe: ${d.sharpe.toFixed(3)}`,
                    `n: ${d.n}`,
                    `Range: ${d.min_val.toFixed(4)} – ${d.max_val.toFixed(4)}`,
                  ];
                },
              },
            },
          },
          scales: this._darkScales(),
        },
      });
    },

    _renderEquity() {
      const el = document.getElementById('chart-equity');
      if (!el || !this.data) return;
      if (this._charts['equity']) this._charts['equity'].destroy();

      const eqData = this.data.equity_by_decile || {};
      const colors = ['#e74c3c','#e67e22','#f1c40f','#2ecc71','#1abc9c',
                       '#3498db','#9b59b6','#e84393','#fd79a8','#636e72'];
      const datasets = [];
      for (const d of this.selectedDeciles) {
        const eq = eqData[d]?.[this.equityMode];
        if (!eq?.points?.length) continue;
        datasets.push({
          label: `D${d}`,
          data: eq.points.map(p => p.value * 100),
          borderColor: colors[(d-1) % colors.length],
          backgroundColor: 'transparent',
          borderWidth: 1.5, pointRadius: 0, tension: 0.1,
        });
      }

      const allLabels = datasets.length
        ? eqData[Array.from(this.selectedDeciles)[0]]?.[this.equityMode]?.points?.map(p => p.date?.slice(0,7)) || []
        : [];

      this._charts['equity'] = new Chart(el, {
        type: 'line',
        data: { labels: allLabels, datasets },
        options: {
          responsive: true, maintainAspectRatio: false, animation: false,
          plugins: {
            legend: { labels: { color:'#aaa', font:{size:10} } },
            tooltip: {
              backgroundColor: 'rgba(20,20,20,0.95)',
              borderColor: '#444', borderWidth: 1,
              mode: 'index', intersect: false,
            },
          },
          scales: {
            ...this._darkScales(),
            x: { ...this._darkScales().x, ticks: { ...this._darkScales().x.ticks, maxTicksLimit: 12 } },
            y: { ...this._darkScales().y, ticks: { ...this._darkScales().y.ticks,
                  callback: v => v.toFixed(0) + '%' } },
          },
        },
      });
    },

    _renderYearly() {
      const el = document.getElementById('chart-yearly');
      if (!el || !this.data?.yearly) return;
      if (this._charts['yearly']) this._charts['yearly'].destroy();

      const yearly = this.data.yearly;
      this._charts['yearly'] = new Chart(el, {
        type: 'bar',
        data: {
          labels: yearly.map(y => y.year),
          datasets: [{
            label: 'Avg Return',
            data: yearly.map(y => y.avg_ret * 100),
            backgroundColor: yearly.map(y => y.avg_ret >= 0 ? '#3498db' : '#e84393'),
            borderWidth: 0,
          }],
        },
        options: {
          responsive: true, maintainAspectRatio: false, animation: false,
          plugins: {
            legend: { display: false },
            tooltip: {
              callbacks: {
                label: ctx => {
                  const y = yearly[ctx.dataIndex];
                  return [`Avg: ${(y.avg_ret*100).toFixed(3)}%`, `WR: ${(y.win_rate*100).toFixed(0)}%`, `n: ${y.n}`];
                },
              },
            },
          },
          scales: {
            ...this._darkScales(),
            y: { ...this._darkScales().y, ticks: { ...this._darkScales().y.ticks,
                  callback: v => v.toFixed(2) + '%' } },
          },
        },
      });
    },

    _renderYearlyConsistency() {
      const el = document.getElementById('chart-consistency');
      if (!el || !this.data?.yearly_consistency?.length) return;
      if (this._charts['consistency']) this._charts['consistency'].destroy();

      const yc = this.data.yearly_consistency;
      this._charts['consistency'] = new Chart(el, {
        type: 'bar',
        data: {
          labels: yc.map(y => y.year),
          datasets: [
            { label: 'D10 (top)', data: yc.map(y => y.top_avg*100), backgroundColor: '#3498db' },
            { label: 'D1 (bottom)', data: yc.map(y => y.bot_avg*100), backgroundColor: '#e84393' },
          ],
        },
        options: {
          responsive: true, maintainAspectRatio: false, animation: false,
          plugins: {
            legend: { labels:{color:'#aaa',font:{size:10}} },
            tooltip: { backgroundColor:'rgba(20,20,20,0.95)', borderColor:'#444', borderWidth:1 },
          },
          scales: {
            ...this._darkScales(),
            y: { ...this._darkScales().y, ticks: { ...this._darkScales().y.ticks,
                  callback: v => v.toFixed(2) + '%' } },
          },
        },
      });
    },

    // Helpers
    pct(v) { return v != null ? (v*100).toFixed(2) + '%' : '—'; },
    r4(v)  { return v != null ? v.toFixed(4) : '—'; },
  }));
});
