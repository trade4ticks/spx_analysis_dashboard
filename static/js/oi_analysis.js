'use strict';

document.addEventListener('alpine:init', () => {
  Alpine.data('oiAnalysis', () => ({
    // Selectors
    tickers: [], features: [], outcomes: [],
    ticker: '', metric: '', outcome: '',
    dateFrom: '2020-01-01', dateTo: new Date().toISOString().slice(0, 10),
    selectedDeciles: new Set([1, 10]),
    equityMode: 'concurrent',  // 'concurrent' | 'non_overlapping'

    // Data
    data: null,
    loading: false,
    error: null,
    _charts: {},
    fsChartId: null,

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
        let url = `/api/oi-analysis/analyze?ticker=${encodeURIComponent(this.ticker)}`
          + `&metric=${encodeURIComponent(this.metric)}`
          + `&outcome=${encodeURIComponent(this.outcome)}`;
        if (this.dateFrom) url += `&date_from=${this.dateFrom}`;
        if (this.dateTo) url += `&date_to=${this.dateTo}`;
        const r = await fetch(url);
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
      this._onDecileChange();
    },

    selectAllDeciles() { this.selectedDeciles = new Set([1,2,3,4,5,6,7,8,9,10]); this._onDecileChange(); },
    selectExtremes()   { this.selectedDeciles = new Set([1,10]); this._onDecileChange(); },
    selectNone()       { this.selectedDeciles = new Set(); this._onDecileChange(); },

    _onDecileChange() {
      this._renderDecileBar();
      this._renderEquity();
      this._renderDrawdown();
      this._renderYearly();
      this._renderRollingCorr();
      this._renderReturnDist();
      this._renderTradeCalendar();
      this._renderDOW();
    },

    isDecileSelected(d) { return this.selectedDeciles.has(d); },

    setEquityMode(m) { this.equityMode = m; this._renderEquity(); this._renderDrawdown(); this._renderRollingCorr(); },

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
      this._renderAllCharts();
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
      let longestLabels = [];

      for (const d of this.selectedDeciles) {
        const eq = eqData[d]?.[this.equityMode];
        if (!eq?.points?.length) continue;
        const labels = eq.points.map(p => p.date?.slice(0,7));
        if (labels.length > longestLabels.length) longestLabels = labels;
        datasets.push({
          label: `D${d}`,
          data: eq.points.map(p => p.value * 100),
          borderColor: colors[(d-1) % colors.length],
          backgroundColor: 'transparent',
          borderWidth: 1.5, pointRadius: 0, tension: 0.1,
        });
      }

      // Aggregate line (average of all selected deciles)
      if (this.selectedDeciles.size >= 2) {
        const selArr = Array.from(this.selectedDeciles);
        const eqs = selArr.map(d => eqData[d]?.[this.equityMode]?.points || []);
        const maxLen = Math.max(...eqs.map(e => e.length));
        if (maxLen > 0) {
          const aggData = [];
          for (let i = 0; i < maxLen; i++) {
            let sum = 0, cnt = 0;
            for (const eq of eqs) {
              if (i < eq.length) { sum += eq[i].value; cnt++; }
            }
            aggData.push(cnt ? (sum / cnt) * 100 : null);
          }
          datasets.push({
            label: 'Aggregate',
            data: aggData,
            borderColor: '#fff', backgroundColor: 'transparent',
            borderWidth: 2.5, pointRadius: 0, tension: 0.1,
            borderDash: [6, 3],
          });
        }
      }

      // Spot price overlay on secondary axis
      const spotSeries = this.data.spot_series || [];
      if (spotSeries.length > 0) {
        datasets.push({
          label: 'Spot Price',
          data: spotSeries.map(s => s.value),
          borderColor: 'rgba(255,255,255,0.15)', backgroundColor: 'transparent',
          borderWidth: 1, pointRadius: 0, tension: 0.1,
          yAxisID: 'y1',
        });
        if (spotSeries.length > longestLabels.length) {
          longestLabels = spotSeries.map(s => s.date?.slice(0,7));
        }
      }

      this._charts['equity'] = new Chart(el, {
        type: 'line',
        data: { labels: longestLabels, datasets },
        options: {
          responsive: true, maintainAspectRatio: false, animation: false,
          plugins: {
            legend: { labels: { color:'#aaa', font:{size:10} } },
            tooltip: {
              backgroundColor: 'rgba(20,20,20,0.95)',
              borderColor: '#444', borderWidth: 1,
              mode: 'index', intersect: false,
              filter: item => item.dataset.label !== 'Spot Price',
            },
          },
          scales: {
            ...this._darkScales(),
            x: { ...this._darkScales().x, ticks: { ...this._darkScales().x.ticks, maxTicksLimit: 12 } },
            y: { ...this._darkScales().y, ticks: { ...this._darkScales().y.ticks,
                  callback: v => v.toFixed(0) + '%' },
                 title: { display:true, text:'Cum Return %', color:'#888', font:{size:9} } },
            y1: { display: spotSeries.length > 0, position: 'right',
                  grid: { drawOnChartArea: false },
                  ticks: { color:'rgba(255,255,255,0.2)', font:{size:8} },
                  title: { display:true, text:'Spot', color:'rgba(255,255,255,0.2)', font:{size:8} } },
          },
        },
      });
    },

    _renderYearly() {
      const el = document.getElementById('chart-yearly');
      if (!el || !this.data?.trade_calendar) return;
      if (this._charts['yearly']) this._charts['yearly'].destroy();

      const cal = this.data.trade_calendar || [];
      const selDec = this.selectedDeciles;
      // Filter to selected deciles, group by year
      const filtered = selDec.size > 0
        ? cal.filter(c => selDec.has(c.decile))
        : cal;
      const byYear = {};
      for (const c of filtered) {
        if (!byYear[c.year]) byYear[c.year] = { rets: [], wins: 0 };
        byYear[c.year].rets.push(c.ret);
        if (c.ret > 0) byYear[c.year].wins++;
      }
      const years = Object.keys(byYear).sort();
      const avgs = years.map(yr => {
        const r = byYear[yr].rets;
        return r.length ? r.reduce((a,b) => a+b, 0) / r.length * 100 : 0;
      });
      const decLabel = selDec.size > 0 ? Array.from(selDec).sort((a,b)=>a-b).map(d=>'D'+d).join('+') : 'All';

      this._charts['yearly'] = new Chart(el, {
        type: 'bar',
        data: {
          labels: years,
          datasets: [{
            label: `Avg Return (${decLabel})`,
            data: avgs,
            backgroundColor: avgs.map(v => v >= 0 ? '#3498db' : '#e84393'),
            borderWidth: 0,
          }],
        },
        options: {
          responsive: true, maintainAspectRatio: false, animation: false,
          plugins: {
            legend: { labels:{color:'#aaa',font:{size:10}} },
            tooltip: {
              callbacks: {
                label: ctx => {
                  const yr = years[ctx.dataIndex];
                  const info = byYear[yr];
                  const avg = info.rets.reduce((a,b)=>a+b,0) / info.rets.length;
                  const wr = info.wins / info.rets.length;
                  return [`Avg: ${(avg*100).toFixed(3)}%`, `WR: ${(wr*100).toFixed(0)}%`, `n: ${info.rets.length}`];
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

    // ── Boxplot (5th-95th percentile + IQR + median) ───────────────────
    _renderBoxplot() {
      const el = document.getElementById('chart-boxplot');
      if (!el || !this.data?.decile_stats) return;
      if (this._charts['boxplot']) this._charts['boxplot'].destroy();

      const stats = (this.data.decile_stats || []).filter(d => d && d.returns?.length >= 5);
      const labels = stats.map(d => 'D' + d.bucket);
      const boxData = stats.map(d => {
        const s = [...d.returns].sort((a,b) => a-b);
        const pct = p => s[Math.floor(s.length * p)] * 100;
        return { p5: pct(0.05), q1: pct(0.25), med: pct(0.5), q3: pct(0.75), p95: pct(0.95) };
      });

      // IQR as floating bars, whiskers as error-bar-like lines, median as points
      this._charts['boxplot'] = new Chart(el, {
        type: 'bar',
        data: {
          labels,
          datasets: [
            // Whisker range (5th-95th) as thin faint bars behind
            { label: 'P5-P95', data: boxData.map(b => [b.p5, b.p95]),
              backgroundColor: 'rgba(255,255,255,0.05)', borderColor: 'rgba(255,255,255,0.2)',
              borderWidth: 1, barPercentage: 0.15 },
            // IQR as thicker bars
            { label: 'IQR', data: boxData.map(b => [b.q1, b.q3]),
              backgroundColor: 'rgba(52,152,219,0.3)', borderColor: '#3498db',
              borderWidth: 1, barPercentage: 0.5 },
            // Median as point line
            { label: 'Median', data: boxData.map(b => b.med), type: 'line',
              borderColor: '#fff', backgroundColor: 'transparent',
              borderWidth: 2, pointRadius: 4, pointBackgroundColor: '#fff', pointBorderWidth: 0 },
          ],
        },
        options: {
          responsive: true, maintainAspectRatio: false, animation: false,
          plugins: {
            legend: { display: false },
            tooltip: {
              callbacks: {
                label: ctx => {
                  const b = boxData[ctx.dataIndex];
                  return [`P5: ${b.p5.toFixed(2)}%`, `Q1: ${b.q1.toFixed(2)}%`,
                          `Med: ${b.med.toFixed(2)}%`, `Q3: ${b.q3.toFixed(2)}%`,
                          `P95: ${b.p95.toFixed(2)}%`];
                },
              },
            },
          },
          scales: {
            ...this._darkScales(),
            y: { ...this._darkScales().y, ticks: { ...this._darkScales().y.ticks,
                  callback: v => v.toFixed(1) + '%' } },
          },
        },
      });
    },

    // ── Drawdown chart ──────────────────────────────────────────────────
    _renderDrawdown() {
      const el = document.getElementById('chart-drawdown');
      if (!el || !this.data?.equity_by_decile) return;
      if (this._charts['drawdown']) this._charts['drawdown'].destroy();

      const eqData = this.data.equity_by_decile;
      const colors = ['#e74c3c','#e67e22','#f1c40f','#2ecc71','#1abc9c',
                       '#3498db','#9b59b6','#e84393','#fd79a8','#636e72'];
      const datasets = [];
      for (const d of this.selectedDeciles) {
        const eq = eqData[d]?.[this.equityMode];
        if (!eq?.points?.length) continue;
        // Compute drawdown from equity points
        let peak = 0;
        const dd = eq.points.map(p => {
          peak = Math.max(peak, p.value);
          return (p.value - peak) * 100;
        });
        datasets.push({
          label: `D${d}`,
          data: dd,
          borderColor: colors[(d-1) % colors.length],
          backgroundColor: 'transparent',
          borderWidth: 1.5, pointRadius: 0, tension: 0.1, fill: true,
          backgroundColor: colors[(d-1) % colors.length] + '15',
        });
      }

      // Aggregate drawdown
      if (this.selectedDeciles.size >= 2 && datasets.length >= 2) {
        const selArr = Array.from(this.selectedDeciles);
        const eqs = selArr.map(d => eqData[d]?.[this.equityMode]?.points || []);
        const maxLen = Math.max(...eqs.map(e => e.length));
        if (maxLen > 0) {
          let peak = 0;
          const aggDD = [];
          for (let i = 0; i < maxLen; i++) {
            let sum = 0, cnt = 0;
            for (const eq of eqs) { if (i < eq.length) { sum += eq[i].value; cnt++; } }
            const val = cnt ? sum / cnt : 0;
            peak = Math.max(peak, val);
            aggDD.push((val - peak) * 100);
          }
          datasets.push({
            label: 'Aggregate', data: aggDD,
            borderColor: '#fff', backgroundColor: 'transparent',
            borderWidth: 2.5, pointRadius: 0, tension: 0.1, borderDash: [6,3],
          });
        }
      }

      const allLabels = datasets.length
        ? eqData[Array.from(this.selectedDeciles)[0]]?.[this.equityMode]?.points?.map(p => p.date?.slice(0,7)) || []
        : [];

      this._charts['drawdown'] = new Chart(el, {
        type: 'line',
        data: { labels: allLabels, datasets },
        options: {
          responsive: true, maintainAspectRatio: false, animation: false,
          plugins: {
            legend: { labels:{color:'#aaa',font:{size:10}} },
            tooltip: { backgroundColor:'rgba(20,20,20,0.95)', borderColor:'#444', borderWidth:1,
                       mode:'index', intersect:false },
          },
          scales: {
            ...this._darkScales(),
            x: { ...this._darkScales().x, ticks:{...this._darkScales().x.ticks, maxTicksLimit:12} },
            y: { ...this._darkScales().y, ticks:{...this._darkScales().y.ticks,
                  callback: v => v.toFixed(1)+'%' },
                 max: 0 },
          },
        },
      });
    },

    // ── 2D Heatmap ──────────────────────────────────────────────────────
    heatmapMetric: '',
    heatmapData: null,
    heatmapLoading: false,

    async loadHeatmap() {
      if (!this.heatmapMetric || !this.data) return;
      this.heatmapLoading = true;
      try {
        const r = await fetch(
          `/api/oi-analysis/heatmap?ticker=${encodeURIComponent(this.ticker)}`
          + `&metric_x=${encodeURIComponent(this.metric)}`
          + `&metric_y=${encodeURIComponent(this.heatmapMetric)}`
          + `&outcome=${encodeURIComponent(this.outcome)}&bins=5`);
        if (r.ok) {
          this.heatmapData = await r.json();
          await this.$nextTick();
          this._renderHeatmap();
        }
      } catch (_) {}
      this.heatmapLoading = false;
    },

    _renderHeatmap() {
      const el = document.getElementById('chart-heatmap');
      if (!el || !this.heatmapData?.grid) return;
      if (this._charts['heatmap']) this._charts['heatmap'].destroy();

      const grid = this.heatmapData.grid;
      const xLabels = this.heatmapData.x_labels;
      const yLabels = this.heatmapData.y_labels;
      const bins = grid.length;

      // Build scatter dataset with colored points
      const points = [];
      let minRet = Infinity, maxRet = -Infinity;
      for (let i = 0; i < bins; i++) {
        for (let j = 0; j < bins; j++) {
          const cell = grid[i][j];
          if (!cell) continue;
          points.push({ x: i, y: j, r: Math.min(Math.max(cell.n / 5, 8), 25),
                        avg: cell.avg_ret, wr: cell.win_rate, n: cell.n });
          minRet = Math.min(minRet, cell.avg_ret);
          maxRet = Math.max(maxRet, cell.avg_ret);
        }
      }

      const range = Math.max(Math.abs(minRet), Math.abs(maxRet)) || 0.01;
      const colors = points.map(p => {
        const t = p.avg / range;
        return t >= 0
          ? `rgba(52,152,219,${Math.min(Math.abs(t)*0.8+0.2, 1)})`
          : `rgba(232,67,147,${Math.min(Math.abs(t)*0.8+0.2, 1)})`;
      });

      this._charts['heatmap'] = new Chart(el, {
        type: 'bubble',
        data: {
          datasets: [{
            data: points,
            backgroundColor: colors,
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
                  const p = points[ctx.dataIndex];
                  return [
                    `X: ${xLabels[p.x]}`, `Y: ${yLabels[p.y]}`,
                    `Avg: ${(p.avg*100).toFixed(3)}%`,
                    `WR: ${(p.wr*100).toFixed(1)}%`, `n: ${p.n}`,
                  ];
                },
              },
            },
          },
          scales: {
            x: { ...this._darkScales().x, type:'linear', min:-0.5, max:bins-0.5,
                 ticks: { ...this._darkScales().x.ticks, autoSkip: false, stepSize: 1,
                          callback: v => Number.isInteger(v) && v >= 0 && v < bins ? xLabels[v] : '' },
                 title: { display:true, text:this.heatmapData.metric_x, color:'#888', font:{size:10} } },
            y: { ...this._darkScales().y, type:'linear', min:-0.5, max:bins-0.5,
                 ticks: { ...this._darkScales().y.ticks, autoSkip: false, stepSize: 1,
                          callback: v => Number.isInteger(v) && v >= 0 && v < bins ? yLabels[v] : '' },
                 title: { display:true, text:this.heatmapData.metric_y, color:'#888', font:{size:10} } },
          },
        },
      });
    },

    // ── AI Summary ──────────────────────────────────────────────────────
    aiSummary: '',
    aiLoading: false,

    async generateAISummary() {
      if (!this.data) return;
      this.aiLoading = true;
      this.aiSummary = '';
      try {
        const r = await fetch(
          `/api/oi-analysis/ai-summary?ticker=${encodeURIComponent(this.ticker)}`
          + `&metric=${encodeURIComponent(this.metric)}`
          + `&outcome=${encodeURIComponent(this.outcome)}`);
        if (r.ok) {
          const d = await r.json();
          this.aiSummary = d.summary || '(no summary)';
        }
      } catch (e) {
        this.aiSummary = 'Error: ' + e.message;
      }
      this.aiLoading = false;
    },

    // ── Rolling correlation ────────────────────────────────────────────
    _renderRollingCorr() {
      const el = document.getElementById('chart-rolling');
      if (!el || !this.data?.rolling_corr?.length) return;
      if (this._charts['rolling']) this._charts['rolling'].destroy();

      const rc = this.data.rolling_corr;
      this._charts['rolling'] = new Chart(el, {
        type: 'line',
        data: {
          labels: rc.map(r => r.date?.slice(0, 7)),
          datasets: [{
            label: 'Spearman ρ (252d)',
            data: rc.map(r => r.spearman),
            borderColor: '#3498db', backgroundColor: 'transparent',
            borderWidth: 1.5, pointRadius: 0, tension: 0.2,
          }],
        },
        options: {
          responsive: true, maintainAspectRatio: false, animation: false,
          plugins: {
            legend: { labels:{color:'#aaa',font:{size:10}} },
            tooltip: { backgroundColor:'rgba(20,20,20,0.95)', borderColor:'#444', borderWidth:1 },
          },
          scales: {
            ...this._darkScales(),
            x: { ...this._darkScales().x, ticks:{...this._darkScales().x.ticks, maxTicksLimit:10} },
            y: { ...this._darkScales().y, ticks:{...this._darkScales().y.ticks,
                  callback: v => v.toFixed(3) } },
          },
        },
      });
    },

    // ── Return distribution (histogram with background) ─────────────────
    _renderReturnDist() {
      const el = document.getElementById('chart-dist');
      if (!el || !this.data?.decile_stats) return;
      if (this._charts['dist']) this._charts['dist'].destroy();

      const allRets = [];
      const selRets = [];
      for (const d of (this.data.decile_stats || [])) {
        if (!d?.returns) continue;
        allRets.push(...d.returns.map(r => r * 100));
        if (this.selectedDeciles.has(d.bucket)) {
          selRets.push(...d.returns.map(r => r * 100));
        }
      }
      if (!allRets.length) return;

      // Build histogram bins
      const nBins = 40;
      const mn = Math.max(Math.min(...allRets), -15);
      const mx = Math.min(Math.max(...allRets), 15);
      const step = (mx - mn) / nBins;
      const labels = [];
      const allCounts = new Array(nBins).fill(0);
      const selCounts = new Array(nBins).fill(0);
      for (let i = 0; i < nBins; i++) labels.push((mn + step * (i + 0.5)).toFixed(1));
      for (const v of allRets) {
        const b = Math.min(Math.floor((v - mn) / step), nBins - 1);
        if (b >= 0) allCounts[b]++;
      }
      for (const v of selRets) {
        const b = Math.min(Math.floor((v - mn) / step), nBins - 1);
        if (b >= 0) selCounts[b]++;
      }

      const decLabel = this.selectedDeciles.size > 0
        ? Array.from(this.selectedDeciles).sort((a,b)=>a-b).map(d=>'D'+d).join('+') : 'None';

      this._charts['dist'] = new Chart(el, {
        type: 'bar',
        data: {
          labels,
          datasets: [
            { label: 'All Deciles', data: allCounts,
              backgroundColor: 'rgba(255,255,255,0.08)', borderWidth: 0, barPercentage: 1, categoryPercentage: 1 },
            { label: decLabel, data: selCounts,
              backgroundColor: 'rgba(52,152,219,0.5)', borderWidth: 0, barPercentage: 1, categoryPercentage: 1 },
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
            x: { ...this._darkScales().x, title:{display:true,text:'Return %',color:'#888',font:{size:10}} },
            y: { ...this._darkScales().y, title:{display:true,text:'Count',color:'#888',font:{size:10}} },
          },
        },
      });
    },

    // ── Trade calendar (month × year heatmap) ───────────────────────────
    _renderTradeCalendar() {
      const el = document.getElementById('chart-calendar');
      if (!el || !this.data?.trade_calendar) return;
      if (this._charts['calendar']) this._charts['calendar'].destroy();

      const cal = this.data.trade_calendar || [];
      const selDec = this.selectedDeciles;
      const filtered = selDec.size > 0 ? cal.filter(c => selDec.has(c.decile)) : cal;

      // Group by year × month
      const byYM = {};
      for (const c of filtered) {
        const k = `${c.year}-${c.month}`;
        if (!byYM[k]) byYM[k] = [];
        byYM[k].push(c.ret);
      }

      const years = [...new Set(filtered.map(c => c.year))].sort();
      const months = [1,2,3,4,5,6,7,8,9,10,11,12];
      const monthNames = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'];

      // Build bubble data
      const points = [];
      let minAvg = Infinity, maxAvg = -Infinity;
      for (let yi = 0; yi < years.length; yi++) {
        for (let mi = 0; mi < 12; mi++) {
          const rets = byYM[`${years[yi]}-${mi+1}`];
          if (!rets?.length) continue;
          const avg = rets.reduce((a,b)=>a+b,0) / rets.length;
          minAvg = Math.min(minAvg, avg);
          maxAvg = Math.max(maxAvg, avg);
          points.push({ x: mi, y: yi, r: Math.min(Math.max(rets.length, 4), 15),
                        avg, n: rets.length, year: years[yi], month: mi+1 });
        }
      }

      const range = Math.max(Math.abs(minAvg), Math.abs(maxAvg)) || 0.01;
      const colors = points.map(p =>
        p.avg >= 0
          ? `rgba(52,152,219,${Math.min(Math.abs(p.avg/range)*0.8+0.2,1)})`
          : `rgba(232,67,147,${Math.min(Math.abs(p.avg/range)*0.8+0.2,1)})`);

      this._charts['calendar'] = new Chart(el, {
        type: 'bubble',
        data: { datasets: [{ data: points, backgroundColor: colors, borderWidth: 0 }] },
        options: {
          responsive: true, maintainAspectRatio: false, animation: false,
          plugins: {
            legend: { display: false },
            tooltip: {
              callbacks: {
                label: ctx => {
                  const p = points[ctx.dataIndex];
                  return [`${monthNames[p.month-1]} ${p.year}`,
                          `Avg: ${(p.avg*100).toFixed(3)}%`, `n: ${p.n}`];
                },
              },
            },
          },
          scales: {
            x: { ...this._darkScales().x, type:'linear', min:-0.5, max:11.5,
                 ticks:{...this._darkScales().x.ticks, callback: v => monthNames[Math.round(v)] || ''} },
            y: { ...this._darkScales().y, type:'linear', min:-0.5, max: years.length - 0.5,
                 ticks:{...this._darkScales().y.ticks, callback: v => years[Math.round(v)] || ''} },
          },
        },
      });
    },

    // ── Day of week P&L ────────────────────────────────────────────────
    _renderDOW() {
      const el = document.getElementById('chart-dow');
      if (!el || !this.data?.dow_data) return;
      if (this._charts['dow']) this._charts['dow'].destroy();

      const dowNames = ['Mon','Tue','Wed','Thu','Fri'];
      const selDec = this.selectedDeciles;
      const filtered = selDec.size > 0
        ? this.data.dow_data.filter(d => selDec.has(d.decile))
        : this.data.dow_data;

      const byDow = {};
      for (const d of filtered) {
        if (!byDow[d.dow]) byDow[d.dow] = [];
        byDow[d.dow].push(d.ret);
      }
      const avgs = dowNames.map((_, i) => {
        const rets = byDow[i] || [];
        return rets.length ? rets.reduce((a,b)=>a+b,0) / rets.length * 100 : 0;
      });
      const counts = dowNames.map((_, i) => (byDow[i] || []).length);

      this._charts['dow'] = new Chart(el, {
        type: 'bar',
        data: {
          labels: dowNames,
          datasets: [{ data: avgs,
            backgroundColor: avgs.map(v => v >= 0 ? '#3498db' : '#e84393'),
            borderWidth: 0 }],
        },
        options: {
          responsive: true, maintainAspectRatio: false, animation: false,
          plugins: {
            legend: { display: false },
            tooltip: { callbacks: { label: ctx => [
              `Avg: ${avgs[ctx.dataIndex].toFixed(3)}%`,
              `n: ${counts[ctx.dataIndex]}`,
            ] } },
          },
          scales: {
            ...this._darkScales(),
            y: { ...this._darkScales().y, ticks: { ...this._darkScales().y.ticks,
                  callback: v => v.toFixed(2) + '%' } },
          },
        },
      });
    },

    // ── Win rate by decile ───────────────────────────────────────────────
    _renderWinRate() {
      const el = document.getElementById('chart-winrate');
      if (!el || !this.data?.decile_stats) return;
      if (this._charts['winrate']) this._charts['winrate'].destroy();

      const stats = (this.data.decile_stats || []).filter(d => d);
      this._charts['winrate'] = new Chart(el, {
        type: 'bar',
        data: {
          labels: stats.map(d => 'D' + d.bucket),
          datasets: [{
            data: stats.map(d => d.win_rate * 100),
            backgroundColor: stats.map(d => d.win_rate >= 0.5 ? '#3498db' : '#e84393'),
            borderWidth: 0,
          }],
        },
        options: {
          responsive: true, maintainAspectRatio: false, animation: false,
          plugins: {
            legend: { display: false },
            tooltip: { callbacks: { label: ctx => {
              const d = stats[ctx.dataIndex];
              return [`WR: ${(d.win_rate*100).toFixed(1)}%`, `n: ${d.n}`];
            } } },
          },
          scales: {
            ...this._darkScales(),
            y: { ...this._darkScales().y, min: 30, max: 70,
                 ticks: { ...this._darkScales().y.ticks, callback: v => v + '%' } },
          },
        },
      });
    },

    // ── Update _renderCharts to include new charts ──────────────────────
    _renderAllCharts() {
      this._renderDecileBar();
      this._renderEquity();
      this._renderYearly();
      this._renderBoxplot();
      this._renderDrawdown();
      this._renderRollingCorr();
      this._renderReturnDist();
      this._renderTradeCalendar();
      this._renderDOW();
      this._renderWinRate();
    },

    // Fullscreen — re-render the chart into the fullscreen canvas
    openFullscreen(chartId) {
      const key = chartId.replace('chart-', '');
      this.fsChartId = chartId;
      this.$nextTick(() => {
        if (this._charts['_fs']) { this._charts['_fs'].destroy(); delete this._charts['_fs']; }
        // Re-render the specific chart into the fs canvas by calling its render method
        const fsEl = document.getElementById('fs-canvas');
        if (!fsEl) return;
        // Swap canvas ID temporarily so render methods target the fullscreen canvas
        const origEl = document.getElementById(chartId);
        if (origEl) origEl.id = chartId + '-orig';
        fsEl.id = chartId;
        // Call the appropriate render method
        const renderMap = {
          'chart-decile': () => this._renderDecileBar(),
          'chart-equity': () => this._renderEquity(),
          'chart-yearly': () => this._renderYearly(),
          'chart-rolling': () => this._renderRollingCorr(),
          'chart-boxplot': () => this._renderBoxplot(),
          'chart-drawdown': () => this._renderDrawdown(),
          'chart-dist': () => this._renderReturnDist(),
          'chart-calendar': () => this._renderTradeCalendar(),
          'chart-dow': () => this._renderDOW(),
          'chart-winrate': () => this._renderWinRate(),
        };
        const fn = renderMap[chartId];
        if (fn) {
          fn();
          // Move the chart instance to _fs key
          this._charts['_fs'] = this._charts[key];
          delete this._charts[key];
        }
        // Restore IDs
        fsEl.id = 'fs-canvas';
        if (origEl) origEl.id = chartId;
      });
    },

    closeFullscreen() {
      if (this._charts['_fs']) { this._charts['_fs'].destroy(); delete this._charts['_fs']; }
      this.fsChartId = null;
      // Re-render the chart back into its original canvas
      this.$nextTick(() => this._renderAllCharts());
    },

    // Helpers
    pct(v) { return v != null ? (v*100).toFixed(2) + '%' : '—'; },
    r4(v)  { return v != null ? v.toFixed(4) : '—'; },
  }));
});
