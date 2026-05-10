'use strict';

document.addEventListener('alpine:init', () => {
  Alpine.store('metricPicker', {
    selected: [],
    toggle(m) {
      if (!m) return;
      if (this.selected.includes(m)) {
        this.selected = this.selected.filter(x => x !== m);
      } else {
        this.selected = [...this.selected, m];
      }
    },
    clear() { this.selected = []; },
    get pairCount() {
      const n = this.selected.length;
      return n >= 2 ? n * (n - 1) / 2 : 0;
    },
  });

  Alpine.data('oiAnalysis', () => ({
    // Selectors
    tickers: [], features: [], outcomes: [],
    ticker: '', metric: '', outcome: '',
    dateFrom: '2020-01-01', dateTo: new Date().toISOString().slice(0, 10),
    selectedDeciles: new Set([1, 10]),
    selectedBins20: new Set([1, 20]),
    equityMode: 'concurrent',  // 'concurrent' | 'non_overlapping'
    decileBins: 10,
    decileBinsData: null,

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
      // Load score matrix (independent of analysis)
      this.smInit();
    },

    async loadAnalysis() {
      if (!this.ticker || !this.metric || !this.outcome) return;
      this.loading = true;
      this.error = null;
      this.decileBins = 10;
      this.decileBinsData = null;
      this.heatmapData = null;
      this.hmXData = null;
      this.hmYData = null;
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

    selectAllDeciles() {
      if (this.decileBins === 20) this.selectedBins20 = new Set([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]);
      else this.selectedDeciles = new Set([1,2,3,4,5,6,7,8,9,10]);
      this._onDecileChange();
    },
    selectExtremes() {
      if (this.decileBins === 20) this.selectedBins20 = new Set([1, 20]);
      else this.selectedDeciles = new Set([1, 10]);
      this._onDecileChange();
    },
    selectNone() {
      if (this.decileBins === 20) this.selectedBins20 = new Set();
      else this.selectedDeciles = new Set();
      this._onDecileChange();
    },

    _onDecileChange() {
      this._renderDecileBar();
      this._renderEquity();
      this._renderDrawdown();
      this._renderYearly();
      this._renderRollingCorr();
      this._renderReturnDist();
      this._renderTradeCalendar();
      this._renderDOW();
      this._renderActivity();
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

      const bins = this.decileBins;
      const stats = (this.decileBinsData || this.data.decile_stats || []).filter(d => d);
      const avgs = stats.map(d => d.avg_ret * 100);
      const self = this;

      const _has20 = !!(self.data?.trade_calendar?.[0]?.decile20);
      const _isSelected = (d) => {
        if (bins === 10) return self.selectedDeciles.has(d.bucket);
        if (bins === 5)  return self.selectedDeciles.has(d.bucket*2-1) || self.selectedDeciles.has(d.bucket*2);
        if (bins === 20) return _has20 ? self.selectedBins20.has(d.bucket)
                                       : self.selectedDeciles.has(Math.ceil(d.bucket/2));
        return false;
      };

      this._charts['decile'] = new Chart(el, {
        type: 'bar',
        data: {
          labels: stats.map(d => 'B' + d.bucket),
          datasets: [{
            data: avgs,
            backgroundColor: stats.map(d => _isSelected(d)
              ? (d.avg_ret >= 0 ? '#3498db' : '#e84393') : 'rgba(100,100,100,0.3)'),
            borderColor:     stats.map(d => _isSelected(d) ? '#fff' : 'transparent'),
            borderWidth:     stats.map(d => _isSelected(d) ? 1 : 0),
          }],
        },
        options: {
          responsive: true, maintainAspectRatio: false, animation: false,
          onClick: (e, elements) => {
            if (!elements.length) return;
            const d = stats[elements[0].index];
            if (bins === 10) {
              self.toggleDecile(d.bucket);
            } else if (bins === 5) {
              const lo = d.bucket * 2 - 1, hi = d.bucket * 2;
              if (self.selectedDeciles.has(lo) && self.selectedDeciles.has(hi)) {
                self.selectedDeciles.delete(lo); self.selectedDeciles.delete(hi);
              } else {
                self.selectedDeciles.add(lo); self.selectedDeciles.add(hi);
              }
              self.selectedDeciles = new Set(self.selectedDeciles);
              self._onDecileChange();
            } else if (bins === 20) {
              if (_has20) {
                // Single-ticker: true independent 20-bin selection
                if (self.selectedBins20.has(d.bucket)) {
                  if (self.selectedBins20.size > 1) self.selectedBins20.delete(d.bucket);
                } else {
                  self.selectedBins20.add(d.bucket);
                }
                self.selectedBins20 = new Set(self.selectedBins20);
              } else {
                // ALL mode: map to parent decile
                self.toggleDecile(Math.ceil(d.bucket / 2));
                return; // toggleDecile calls _onDecileChange + _renderDecileBar
              }
              self._onDecileChange();
            }
            self._renderDecileBar();
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
                    `Sharpe: ${d.sharpe?.toFixed(3) ?? '—'}`,
                    `n: ${d.n}`,
                    d.min_val != null ? `Range: ${d.min_val.toFixed(4)} – ${d.max_val.toFixed(4)}` : '',
                  ].filter(Boolean);
                },
              },
            },
          },
          scales: this._darkScales(),
        },
      });
    },

    _computeDecile5Bins() {
      const ds = (this.data?.decile_stats || []).filter(Boolean);
      const bins = [];
      for (let i = 0; i < 5; i++) {
        const d1 = ds.find(d => d.bucket === i*2+1) || {};
        const d2 = ds.find(d => d.bucket === i*2+2) || {};
        const n1 = d1.n || 0, n2 = d2.n || 0, n = n1 + n2;
        if (!n) continue;
        bins.push({
          bucket:   i + 1,
          n,
          avg_ret:  ((d1.avg_ret||0)*n1 + (d2.avg_ret||0)*n2) / n,
          win_rate: ((d1.win_rate||0)*n1 + (d2.win_rate||0)*n2) / n,
          sharpe:   ((d1.sharpe||0) + (d2.sharpe||0)) / 2,
          std_dev:  ((d1.std_dev||0) + (d2.std_dev||0)) / 2,
          min_val:  d1.min_val, max_val: d2.max_val ?? d1.max_val,
        });
      }
      return bins;
    },

    async setDecileBins(n) {
      if (!this.data) return;
      this.decileBins = n;
      if (n === 5) {
        this.decileBinsData = this._computeDecile5Bins();
      } else if (n === 10) {
        this.decileBinsData = null;
        this.selectedDeciles = new Set([1, 10]);
      } else {
        this.selectedBins20 = new Set([1, 20]);
        try {
          const r = await fetch(
            `/api/oi-analysis/metric-bins?ticker=${encodeURIComponent(this.ticker)}`
            + `&metric=${encodeURIComponent(this.metric)}`
            + `&outcome=${encodeURIComponent(this.outcome)}&bins=20`
            + (this.dateFrom ? `&date_from=${this.dateFrom}` : '')
            + (this.dateTo ? `&date_to=${this.dateTo}` : ''));
          if (r.ok) { const d = await r.json(); this.decileBinsData = d.buckets || null; }
        } catch (_) {}
      }
      this._renderDecileBar();
      this._renderEquity();
      this._renderYearly();
      this._renderDOW();
    },

    _getEquity20Curve(bin, mode) {
      const horizon = this.data.horizon || 1;
      const cal = (this.data.trade_calendar || []).filter(c => c.decile20 === bin);
      if (!cal.length) return { points: [], n_trades: 0, cum_return: 0, win_rate: 0 };
      const sorted = cal.slice().sort((a, b) => a.date.localeCompare(b.date));
      let cum = 0, peak = 0, wins = 0, lastDate = null;
      const points = [];
      for (const c of sorted) {
        if (mode === 'non_overlapping' && lastDate !== null) {
          const diffDays = (new Date(c.date) - new Date(lastDate)) / 86400000;
          if (diffDays < horizon) continue;
        }
        lastDate = c.date;
        cum += c.ret;
        peak = Math.max(peak, cum);
        if (c.ret > 0) wins++;
        points.push({ date: c.date, value: cum });
      }
      const n = points.length;
      return { points, n_trades: n, cum_return: cum, win_rate: n ? wins / n : 0 };
    },

    _renderEquity() {
      const el = document.getElementById('chart-equity');
      if (!el || !this.data) return;
      if (this._charts['equity']) this._charts['equity'].destroy();

      const use20 = this.decileBins === 20 && !!(this.data.trade_calendar?.[0]?.decile20);
      const selectedKeys = use20 ? this.selectedBins20 : this.selectedDeciles;
      const binLabel = use20 ? 'B' : 'D';

      let eqData;
      if (use20) {
        eqData = {};
        for (const bin of selectedKeys) {
          eqData[bin] = {
            concurrent:      this._getEquity20Curve(bin, 'concurrent'),
            non_overlapping: this._getEquity20Curve(bin, 'non_overlapping'),
          };
        }
      } else {
        eqData = this.data.equity_by_decile || {};
      }

      const spotSeries = this.data.spot_series || [];
      const colors = ['#e74c3c','#e67e22','#f1c40f','#2ecc71','#1abc9c',
                       '#3498db','#9b59b6','#e84393','#fd79a8','#636e72'];

      // Build a master timeline from spot (dense) — all other datasets map onto this
      const timeline = spotSeries.length > 0
        ? spotSeries.map(s => s.date)
        : [];  // fallback: build from first equity dataset

      // If no spot, use the longest equity curve dates as timeline
      if (!timeline.length) {
        let longest = [];
        for (const d of selectedKeys) {
          const pts = eqData[d]?.[this.equityMode]?.points || [];
          if (pts.length > longest.length) longest = pts;
        }
        timeline.push(...longest.map(p => p.date));
      }

      const dateIndex = {};
      timeline.forEach((d, i) => dateIndex[d] = i);

      // Map equity curve onto the timeline (null for dates without trades, spanGaps)
      const datasets = [];
      for (const d of selectedKeys) {
        const eq = eqData[d]?.[this.equityMode];
        if (!eq?.points?.length) continue;
        const mapped = new Array(timeline.length).fill(null);
        for (const p of eq.points) {
          const idx = dateIndex[p.date];
          if (idx !== undefined) mapped[idx] = p.value * 100;
        }
        datasets.push({
          label: `${binLabel}${d}`,
          data: mapped,
          borderColor: colors[(d-1) % colors.length],
          backgroundColor: 'transparent',
          borderWidth: 1.5, pointRadius: 0, tension: 0.1,
          spanGaps: true,
        });
      }

      // Aggregate line — sum all trades from all selected bins (additive portfolio)
      if (selectedKeys.size >= 2) {
        const selArr = Array.from(selectedKeys);
        // Build a carried-forward cumulative for each decile across the full timeline
        const carried = selArr.map(d => {
          const pts = eqData[d]?.[this.equityMode]?.points || [];
          const valByDate = {};
          for (const p of pts) valByDate[p.date] = p.value;
          const arr = new Array(timeline.length).fill(null);
          let last = 0;
          for (let i = 0; i < timeline.length; i++) {
            if (valByDate[timeline[i]] !== undefined) last = valByDate[timeline[i]];
            arr[i] = last;
          }
          return arr;
        });

        // Sum all deciles at each point — combined portfolio of all selected decile trades
        const mapped = timeline.map((_, i) => {
          const sum = carried.reduce((a, c) => a + c[i], 0);
          return sum * 100;
        });

        datasets.push({
          label: 'Aggregate', data: mapped,
          borderColor: '#fff', backgroundColor: 'transparent',
          borderWidth: 2.5, pointRadius: 0, tension: 0.1,
          borderDash: [6, 3],
        });
      }

      // Spot overlay
      if (spotSeries.length > 0) {
        datasets.push({
          label: 'Spot Price',
          data: spotSeries.map(s => s.value),
          borderColor: 'rgba(255,255,255,0.15)', backgroundColor: 'transparent',
          borderWidth: 1, pointRadius: 0, tension: 0.1,
          yAxisID: 'y1',
        });
      }

      this._charts['equity'] = new Chart(el, {
        type: 'line',
        data: { labels: timeline.map(d => d?.slice(0,7)), datasets },
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
      const use20y = this.decileBins === 20 && !!(cal[0]?.decile20);
      const selDecY = use20y ? this.selectedBins20 : this.selectedDeciles;
      const decFieldY = use20y ? 'decile20' : 'decile';
      // Filter to selected deciles/bins, group by year
      const filtered = selDecY.size > 0
        ? cal.filter(c => selDecY.has(c[decFieldY]))
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
      const _lbl = use20y ? 'B' : 'D';
      const decLabel = selDecY.size > 0 ? Array.from(selDecY).sort((a,b)=>a-b).map(d=>_lbl+d).join('+') : 'All';

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
      const spotSeries = this.data.spot_series || [];
      const colors = ['#e74c3c','#e67e22','#f1c40f','#2ecc71','#1abc9c',
                       '#3498db','#9b59b6','#e84393','#fd79a8','#636e72'];

      // Reuse the spot timeline for consistent x-axis
      const timeline = spotSeries.length > 0 ? spotSeries.map(s => s.date) : [];
      if (!timeline.length) {
        for (const d of this.selectedDeciles) {
          const pts = eqData[d]?.[this.equityMode]?.points || [];
          if (pts.length > timeline.length) { timeline.length = 0; timeline.push(...pts.map(p => p.date)); }
        }
      }
      const dateIndex = {};
      timeline.forEach((d, i) => dateIndex[d] = i);

      const datasets = [];
      for (const d of this.selectedDeciles) {
        const eq = eqData[d]?.[this.equityMode];
        if (!eq?.points?.length) continue;
        // Compute drawdown then map onto timeline
        let peak = 0;
        const ddByDate = {};
        for (const p of eq.points) {
          peak = Math.max(peak, p.value);
          ddByDate[p.date] = (p.value - peak) * 100;
        }
        const mapped = new Array(timeline.length).fill(null);
        for (const [dt, v] of Object.entries(ddByDate)) {
          const idx = dateIndex[dt];
          if (idx !== undefined) mapped[idx] = v;
        }
        datasets.push({
          label: `D${d}`, data: mapped,
          borderColor: colors[(d-1) % colors.length],
          backgroundColor: colors[(d-1) % colors.length] + '15',
          borderWidth: 1.5, pointRadius: 0, tension: 0.1, fill: true,
          spanGaps: true,
        });
      }

      // Aggregate drawdown — carry forward last known value per decile, average all
      if (this.selectedDeciles.size >= 2) {
        const selArr = Array.from(this.selectedDeciles);
        // Build a carried-forward cumulative for each decile across the full timeline
        const carried = selArr.map(d => {
          const pts = eqData[d]?.[this.equityMode]?.points || [];
          const valByDate = {};
          for (const p of pts) valByDate[p.date] = p.value;
          const arr = new Array(timeline.length).fill(null);
          let last = 0;
          for (let i = 0; i < timeline.length; i++) {
            if (valByDate[timeline[i]] !== undefined) last = valByDate[timeline[i]];
            arr[i] = last;
          }
          return arr;
        });
        // Average the carried-forward values, then compute drawdown from the average
        let peak = 0;
        const mapped = timeline.map((_, i) => {
          const avg = carried.reduce((a, c) => a + c[i], 0) / carried.length;
          peak = Math.max(peak, avg);
          return (avg - peak) * 100;
        });
        datasets.push({
          label: 'Aggregate', data: mapped,
          borderColor: '#fff', backgroundColor: 'transparent',
          borderWidth: 2.5, pointRadius: 0, tension: 0.1, borderDash: [6,3],
        });
      }

      this._charts['drawdown'] = new Chart(el, {
        type: 'line',
        data: { labels: timeline.map(d => d?.slice(0,7)), datasets },
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
    heatmapBins: 5,
    hmBins1d: 10,
    hmXData: null,
    hmYData: null,
    _hmRange: null,

    async loadHeatmap() {
      if (!this.heatmapMetric || !this.data) return;
      this.heatmapLoading = true;
      this.heatmapData = null;
      this.hmXData = null;
      this.hmYData = null;
      this._hmRange = null;
      try {
        const r = await fetch(
          `/api/oi-analysis/heatmap?ticker=${encodeURIComponent(this.ticker)}`
          + `&metric_x=${encodeURIComponent(this.metric)}`
          + `&metric_y=${encodeURIComponent(this.heatmapMetric)}`
          + `&outcome=${encodeURIComponent(this.outcome)}&bins=${this.heatmapBins}`);
        if (r.ok) {
          const d = await r.json();
          let max = 0;
          for (const row of (d.grid || [])) {
            for (const c of row) {
              if (c && c.n) max = Math.max(max, Math.abs(c.avg_ret || 0));
            }
          }
          this._hmRange = max || 0.01;
          this.heatmapData = d;
        }
        await this.loadHmBins1d();
      } catch (_) {}
      this.heatmapLoading = false;
    },

    setHeatmapBins(n) {
      this.heatmapBins = n;
      this.loadHeatmap();
    },

    async setHmBins1d(n) {
      this.hmBins1d = n;
      await this.loadHmBins1d();
    },

    async loadHmBins1d() {
      if (!this.data || !this.heatmapMetric) return;
      const base = `/api/oi-analysis/metric-bins?ticker=${encodeURIComponent(this.ticker)}`
        + `&outcome=${encodeURIComponent(this.outcome)}&bins=${this.hmBins1d}`
        + (this.dateFrom ? `&date_from=${this.dateFrom}` : '')
        + (this.dateTo ? `&date_to=${this.dateTo}` : '');
      try {
        const [rx, ry] = await Promise.all([
          fetch(base + `&metric=${encodeURIComponent(this.metric)}`),
          fetch(base + `&metric=${encodeURIComponent(this.heatmapMetric)}`),
        ]);
        if (rx.ok) { const d = await rx.json(); this.hmXData = d.buckets || null; }
        if (ry.ok) { const d = await ry.json(); this.hmYData = d.buckets || null; }
      } catch (_) {}
      await this.$nextTick();
      this._renderHmBar1d('chart-hm-x', this.hmXData, this.metric);
      this._renderHmBar1d('chart-hm-y', this.hmYData, this.heatmapMetric);
    },

    _renderHmBar1d(canvasId, buckets, title) {
      const el = document.getElementById(canvasId);
      if (!el || !buckets?.length) return;
      if (this._charts[canvasId]) this._charts[canvasId].destroy();
      const stats = buckets.filter(Boolean);
      const avgs = stats.map(d => d.avg_ret * 100);
      const maxAbs = Math.max(...avgs.map(Math.abs), 0.001);
      this._charts[canvasId] = new Chart(el, {
        type: 'bar',
        data: {
          labels: stats.map(d => 'B' + d.bucket),
          datasets: [{
            data: avgs,
            backgroundColor: avgs.map(v => {
              const t = Math.min(Math.abs(v) / maxAbs, 1);
              return v >= 0
                ? `rgba(52,152,219,${(0.2 + t * 0.7).toFixed(2)})`
                : `rgba(232,67,147,${(0.2 + t * 0.7).toFixed(2)})`;
            }),
            borderWidth: 0,
          }],
        },
        options: {
          responsive: true, maintainAspectRatio: false, animation: false,
          plugins: {
            legend: { display: false },
            title: { display: true, text: title, color: '#666', font: { size: 9 }, padding: { top: 0, bottom: 3 } },
            tooltip: {
              backgroundColor: 'rgba(20,20,20,0.95)', borderColor: '#444', borderWidth: 1,
              callbacks: {
                label: ctx => {
                  const d = stats[ctx.dataIndex];
                  return [
                    `Avg: ${(d.avg_ret*100).toFixed(3)}%`,
                    `WR: ${(d.win_rate*100).toFixed(1)}%`,
                    `Sharpe: ${d.sharpe?.toFixed(3) ?? '—'}`,
                    `n: ${d.n}`,
                  ];
                },
              },
            },
          },
          scales: this._darkScales(),
        },
      });
    },

    hmCellBg(cell) {
      if (!cell || !cell.n) return 'rgba(40,40,40,0.5)';
      const t = Math.max(-1, Math.min(1, (cell.avg_ret || 0) / (this._hmRange || 0.01)));
      if (t >= 0) return `rgba(52,152,219,${(0.15 + t * 0.7).toFixed(2)})`;
      return `rgba(232,67,147,${(0.15 + (-t) * 0.7).toFixed(2)})`;
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
      const use20d = this.decileBins === 20 && !!(this.data.dow_data?.[0]?.decile20);
      const selDecD = use20d ? this.selectedBins20 : this.selectedDeciles;
      const decFieldD = use20d ? 'decile20' : 'decile';
      const filtered = selDecD.size > 0
        ? this.data.dow_data.filter(d => selDecD.has(d[decFieldD]))
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
      const wrs = dowNames.map((_, i) => {
        const rets = byDow[i] || [];
        return rets.length ? rets.filter(r => r > 0).length / rets.length * 100 : 0;
      });

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
              `WR: ${wrs[ctx.dataIndex].toFixed(1)}%`,
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

    // ── Trade activity: entries/open per day (computed client-side from trade_calendar) ──
    _renderActivity() {
      const el = document.getElementById('chart-activity');
      if (!el || !this.data?.trade_calendar) return;
      if (this._charts['activity']) this._charts['activity'].destroy();

      const cal = this.data.trade_calendar || [];
      if (!cal.length) return;

      const use20a = this.decileBins === 20 && !!(cal[0]?.decile20);
      const selDec = use20a ? this.selectedBins20 : this.selectedDeciles;
      const decField = use20a ? 'decile20' : 'decile';
      const filtered = selDec.size > 0 ? cal.filter(c => selDec.has(c[decField])) : cal;
      if (!filtered.length) return;

      const horizon = this.data.horizon || 1;
      const horizonCalDays = Math.round(horizon * 1.4);

      // Count entries per date
      const entriesByDate = {};
      for (const c of filtered) {
        entriesByDate[c.date] = (entriesByDate[c.date] || 0) + 1;
      }
      const allDates = Object.keys(entriesByDate).sort();
      const dateMs = allDates.map(d => new Date(d).getTime());
      const entered = allDates.map(d => entriesByDate[d]);

      // Count open positions: entries within the past horizonCalDays calendar days of each date
      const open = allDates.map((_, i) => {
        const cutoffMs = dateMs[i] - horizonCalDays * 86400000;
        let count = 0;
        for (let j = i; j >= 0 && dateMs[j] >= cutoffMs; j--) {
          count += entriesByDate[allDates[j]];
        }
        return count;
      });

      this._charts['activity'] = new Chart(el, {
        type: 'bar',
        data: {
          labels: allDates.map(d => d.slice(0, 7)),
          datasets: [
            {
              type: 'line',
              label: 'Open Trades',
              data: open,
              borderColor: 'rgba(46,204,113,0.6)',
              backgroundColor: 'rgba(46,204,113,0.08)',
              fill: true, tension: 0.3, pointRadius: 0, borderWidth: 1.5,
              order: 1,
            },
            {
              type: 'bar',
              label: 'Entered',
              data: entered,
              backgroundColor: 'rgba(52,152,219,0.7)',
              barThickness: 2,
              order: 2,
            },
          ],
        },
        options: {
          responsive: true, maintainAspectRatio: false, animation: false,
          plugins: {
            legend: { labels: { color: '#aaa', font: { size: 10 } } },
            tooltip: {
              backgroundColor: 'rgba(20,20,20,0.95)', borderColor: '#444', borderWidth: 1,
              mode: 'index', intersect: false,
              callbacks: {
                title: ctx => allDates[ctx[0]?.dataIndex] || '',
                label: ctx => `${ctx.dataset.label}: ${ctx.raw}`,
              },
            },
          },
          scales: {
            ...this._darkScales(),
            x: { ...this._darkScales().x, ticks: { ...this._darkScales().x.ticks, maxTicksLimit: 12 } },
            y: {
              ...this._darkScales().y,
              title: { display: true, text: 'Count', color: '#888', font: { size: 9 } },
              ticks: { ...this._darkScales().y.ticks, stepSize: 1 },
            },
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
      this._renderActivity();
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
          'chart-activity': () => this._renderActivity(),
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

    // ── Score Matrix ──────────────────────────────────────────────────
    smRows: [],
    smMeta: { count: 0, tickers: [], metrics: [], fwd_rets: [], avg_score: 0, gte50: 0, gte70: 0, last_run: null },
    smStatus: { running: false, message: '', last_run: null },
    smFilterTicker: '',
    smFilterMetric: '',
    smFilterFwd: '',
    smMinScore: 0,
    smSortKey: 'composite_score',
    smSortDir: 'desc',
    smPollTimer: null,
    smSelectedMetric: '',
    smSelectedFwd: '',
    smSelectedTicker: '',
    smSummary: { by_metric: [], by_fwd: [], by_ticker: [], by_fwd_ticker: [] },

    // ── Interaction Scan ──
    ifClusters: [],
    ifLastScannedMetrics: [],
    ifStatus: { running: false, message: '', last_run: null },
    ifPollTimer: null,
    ifRows: [],          // ranked interaction-matrix rows
    ifFwdFilter: '',
    ifSelected: null,    // currently-drilled combo {feat_a, feat_b}
    ifDetail: [],        // quadrant rows for selected combo
    ifDetailTicker: '',
    ifDetailFwd: '',
    ifDetailRow: null,   // single row shown in heatmap

    smColumns: [
      { key: 'composite_score', label: 'Score' },
      { key: 'ticker', label: 'Ticker' },
      { key: 'metric', label: 'Metric' },
      { key: 'fwd_ret', label: 'Fwd Ret' },
      { key: 'pattern', label: 'Pattern' },
      { key: 'spearman_r', label: 'Spearman' },
      { key: 'monotonicity', label: 'Mono' },
      { key: 'yearly_pct', label: 'Consist' },
      { key: 'd10_d1_spread', label: 'D10-D1' },
      { key: 'd10_wr', label: 'D10 WR' },
      { key: 'best_sharpe', label: 'Sharpe' },
      { key: 'n_obs', label: 'N' },
      { key: 'mi', label: 'MI' },
      { key: 'pearson_r', label: 'Pearson' },
      { key: 'loyo_fragile', label: 'LOYO' },
    ],

    async smInit() {
      try {
        const [metaRes, statusRes] = await Promise.all([
          fetch('/api/oi-analysis/score-matrix/meta'),
          fetch('/api/oi-analysis/batch-score-status'),
        ]);
        if (metaRes.ok) this.smMeta = await metaRes.json();
        if (statusRes.ok) this.smStatus = await statusRes.json();
        if (this.smMeta.count > 0) {
          await this.loadScoreMatrix();
          await this.loadSmSummary();
          // Re-render ticker charts after layout settles (x-show may delay dimensions)
          setTimeout(() => {
            this._renderSmTickerChart();
            this._renderSmTickerFwdChart();
          }, 150);
        }
        if (this.smStatus.running) this._smStartPoll();
        if (this.ifStatus.running) this._ifStartPoll();
      } catch (_) {}
    },

    async loadScoreMatrix() {
      const params = new URLSearchParams({
        sort_by: this.smSortKey === 'd10_d1_spread' ? 'composite_score' : this.smSortKey,
        order: this.smSortDir,
        min_score: this.smMinScore,
      });
      if (this.smFilterTicker) params.set('ticker', this.smFilterTicker);
      if (this.smFilterMetric) params.set('metric', this.smFilterMetric);
      if (this.smFilterFwd) params.set('fwd_ret', this.smFilterFwd);

      try {
        const r = await fetch('/api/oi-analysis/score-matrix?' + params);
        if (r.ok) this.smRows = await r.json();
        // Refresh meta too
        const m = await fetch('/api/oi-analysis/score-matrix/meta');
        if (m.ok) this.smMeta = await m.json();
      } catch (_) {}
    },

    smSort(key) {
      if (this.smSortKey === key) {
        this.smSortDir = this.smSortDir === 'desc' ? 'asc' : 'desc';
      } else {
        this.smSortKey = key;
        this.smSortDir = 'desc';
      }
      // Sort client-side for d10_d1_spread (computed), server-side for others
      if (key === 'd10_d1_spread') {
        const dir = this.smSortDir === 'desc' ? -1 : 1;
        this.smRows.sort((a, b) => {
          const sa = (a.d10_avg || 0) - (a.d1_avg || 0);
          const sb = (b.d10_avg || 0) - (b.d1_avg || 0);
          return (sb - sa) * dir;
        });
      } else {
        this.loadScoreMatrix();
      }
    },

    async runBatchScore() {
      try {
        const r = await fetch('/api/oi-analysis/run-batch-score', { method: 'POST' });
        if (r.ok) {
          const data = await r.json();
          this.smStatus = { running: true, message: data.message, last_run: this.smStatus.last_run };
          this._smStartPoll();
        }
      } catch (_) {}
    },

    _smStartPoll() {
      if (this.smPollTimer) return;
      this.smPollTimer = setInterval(async () => {
        try {
          const r = await fetch('/api/oi-analysis/batch-score-status');
          if (r.ok) {
            this.smStatus = await r.json();
            if (!this.smStatus.running) {
              clearInterval(this.smPollTimer);
              this.smPollTimer = null;
              await this.loadScoreMatrix();
              await this.loadSmSummary();
            }
          }
        } catch (_) {}
      }, 3000);
    },

    async loadSmSummary(metric, fwdRet, ticker) {
      // Use current selections if args not provided
      if (metric === undefined) metric = this.smSelectedMetric;
      if (fwdRet === undefined) fwdRet = this.smSelectedFwd;
      if (ticker === undefined) ticker = this.smSelectedTicker;
      const params = new URLSearchParams();
      if (metric) params.set('metric', metric);
      if (fwdRet) params.set('fwd_ret', fwdRet);
      if (ticker) params.set('ticker', ticker);
      try {
        const r = await fetch('/api/oi-analysis/score-matrix/summary?' + params);
        if (r.ok) {
          this.smSummary = await r.json();
          this.smSelectedMetric = metric || '';
          this.smSelectedFwd = fwdRet || '';
          this.smSelectedTicker = ticker || '';
          // Sync table filters to match chart selections
          this.smFilterMetric = this.smSelectedMetric;
          this.smFilterFwd = this.smSelectedFwd;
          this.smFilterTicker = this.smSelectedTicker;
          this.loadScoreMatrix();
          this.$nextTick(() => {
            this._renderSmMetricChart();
            this._renderSmFwdChart();
            this._renderSmTickerChart();
            this._renderSmTickerFwdChart();
          });
        }
      } catch (_) {}
    },

    smClearFilters() {
      this.smSelectedMetric = '';
      this.smSelectedFwd = '';
      this.smSelectedTicker = '';
      this.smFilterMetric = '';
      this.smFilterFwd = '';
      this.smFilterTicker = '';
      this.smMinScore = 0;
      this.loadSmSummary('', '', '');
    },

    _smTooltipCallback(data) {
      return (ctx) => {
        if (ctx.datasetIndex !== 0) return '';
        const d = data[ctx.dataIndex];
        return `Std: ${d.std_score}  |  Max: ${d.max_score}  |  ≥50: ${d.gte50}/${d.n}`;
      };
    },

    _renderSmMetricChart() {
      const el = document.getElementById('chart-sm-metric');
      if (!el) return;
      if (this._charts['sm-metric']) this._charts['sm-metric'].destroy();
      const data = this.smSummary.by_metric || [];
      if (!data.length) return;

      const self = this;
      this._charts['sm-metric'] = new Chart(el, {
        type: 'bar',
        data: {
          labels: data.map(d => d.metric),
          datasets: [
            {
              label: 'Avg Score',
              data: data.map(d => d.avg_score),
              backgroundColor: data.map(d =>
                d.metric === self.smSelectedMetric ? '#3498db' :
                d.avg_score >= 40 ? 'rgba(52,152,219,0.5)' :
                d.avg_score >= 30 ? 'rgba(149,165,166,0.5)' :
                'rgba(232,67,147,0.3)'),
              borderWidth: data.map(d => d.metric === self.smSelectedMetric ? 2 : 0),
              borderColor: '#3498db',
            },
            {
              label: 'Std Dev',
              data: data.map(d => d.std_score),
              backgroundColor: 'rgba(255,255,255,0.08)',
              borderWidth: 0,
            },
          ],
        },
        options: {
          responsive: true, maintainAspectRatio: false, animation: false,
          onClick: (evt, elements) => {
            if (elements.length > 0 && elements[0].datasetIndex === 0) {
              const clicked = data[elements[0].index].metric;
              // Toggle: click same metric to deselect
              const newMetric = clicked === self.smSelectedMetric ? '' : clicked;
              self.loadSmSummary(newMetric, self.smSelectedFwd);
            }
          },
          plugins: {
            legend: { labels: { color: '#aaa', font: { size: 9 } } },
            tooltip: {
              backgroundColor: 'rgba(20,20,20,0.95)', borderColor: '#444', borderWidth: 1,
              callbacks: { afterLabel: self._smTooltipCallback(data) },
            },
          },
          scales: {
            x: { ticks: { color: '#888', font: { size: 7 }, maxRotation: 90, minRotation: 45 },
                 grid: { color: 'rgba(255,255,255,0.03)' }, border: { color: 'transparent' } },
            y: { ticks: { color: '#888', font: { size: 9 } },
                 grid: { color: 'rgba(255,255,255,0.05)' }, border: { color: 'transparent' } },
          },
        },
      });
    },

    _renderSmFwdChart() {
      const el = document.getElementById('chart-sm-fwd');
      if (!el) return;
      if (this._charts['sm-fwd']) this._charts['sm-fwd'].destroy();
      const data = this.smSummary.by_fwd || [];
      if (!data.length) return;

      const self = this;
      this._charts['sm-fwd'] = new Chart(el, {
        type: 'bar',
        data: {
          labels: data.map(d => d.fwd_ret),
          datasets: [{
            label: 'Avg Score',
            data: data.map(d => d.avg_score),
            backgroundColor: data.map(d =>
              d.fwd_ret === self.smSelectedFwd ? '#3498db' :
              d.avg_score >= 40 ? 'rgba(52,152,219,0.5)' :
              d.avg_score >= 30 ? 'rgba(149,165,166,0.5)' : 'rgba(232,67,147,0.3)'),
            borderWidth: data.map(d => d.fwd_ret === self.smSelectedFwd ? 2 : 0),
            borderColor: '#3498db',
          }],
        },
        options: {
          responsive: true, maintainAspectRatio: false, animation: false,
          onClick: (evt, elements) => {
            if (elements.length > 0) {
              const clicked = data[elements[0].index].fwd_ret;
              const newFwd = clicked === self.smSelectedFwd ? '' : clicked;
              self.loadSmSummary(self.smSelectedMetric, newFwd);
            }
          },
          plugins: {
            legend: { display: false },
            tooltip: {
              backgroundColor: 'rgba(20,20,20,0.95)', borderColor: '#444', borderWidth: 1,
              callbacks: { afterLabel: self._smTooltipCallback(data) },
            },
          },
          scales: {
            x: { ticks: { color: '#888', font: { size: 8 } },
                 grid: { color: 'rgba(255,255,255,0.03)' }, border: { color: 'transparent' } },
            y: { ticks: { color: '#888', font: { size: 9 } },
                 grid: { color: 'rgba(255,255,255,0.05)' }, border: { color: 'transparent' } },
          },
        },
      });
    },

    _renderSmTickerChart() {
      const el = document.getElementById('chart-sm-ticker');
      if (!el) return;
      if (this._charts['sm-ticker']) this._charts['sm-ticker'].destroy();
      const data = this.smSummary.by_ticker || [];
      if (!data.length) return;

      const self = this;
      this._charts['sm-ticker'] = new Chart(el, {
        type: 'bar',
        data: {
          labels: data.map(d => d.ticker),
          datasets: [{
            label: 'Avg Score',
            data: data.map(d => d.avg_score),
            backgroundColor: data.map(d =>
              d.ticker === self.smSelectedTicker ? '#3498db' :
              d.avg_score >= 40 ? 'rgba(52,152,219,0.5)' :
              d.avg_score >= 30 ? 'rgba(149,165,166,0.5)' :
              'rgba(232,67,147,0.3)'),
            borderWidth: data.map(d => d.ticker === self.smSelectedTicker ? 2 : 0),
            borderColor: '#3498db',
          }],
        },
        options: {
          responsive: true, maintainAspectRatio: false, animation: false,
          onClick: (evt, elements) => {
            if (elements.length > 0) {
              const clicked = data[elements[0].index].ticker;
              const newTicker = clicked === self.smSelectedTicker ? '' : clicked;
              self.loadSmSummary(self.smSelectedMetric, self.smSelectedFwd, newTicker);
            }
          },
          plugins: {
            legend: { display: false },
            tooltip: {
              backgroundColor: 'rgba(20,20,20,0.95)', borderColor: '#444', borderWidth: 1,
              callbacks: { afterLabel: self._smTooltipCallback(data) },
            },
          },
          scales: {
            x: { ticks: { color: '#888', font: { size: 8 }, maxRotation: 45, minRotation: 45 },
                 grid: { color: 'rgba(255,255,255,0.03)' }, border: { color: 'transparent' } },
            y: { ticks: { color: '#888', font: { size: 9 } },
                 grid: { color: 'rgba(255,255,255,0.05)' }, border: { color: 'transparent' } },
          },
        },
      });
    },

    _renderSmTickerFwdChart() {
      const el = document.getElementById('chart-sm-ticker-fwd');
      if (!el) return;
      if (this._charts['sm-ticker-fwd']) this._charts['sm-ticker-fwd'].destroy();
      const data = this.smSummary.by_fwd_ticker || [];
      if (!data.length) return;

      const self = this;
      this._charts['sm-ticker-fwd'] = new Chart(el, {
        type: 'bar',
        data: {
          labels: data.map(d => d.fwd_ret),
          datasets: [{
            label: 'Avg Score',
            data: data.map(d => d.avg_score),
            backgroundColor: data.map(d =>
              d.fwd_ret === self.smSelectedFwd ? '#3498db' :
              d.avg_score >= 40 ? 'rgba(52,152,219,0.5)' :
              d.avg_score >= 30 ? 'rgba(149,165,166,0.5)' : 'rgba(232,67,147,0.3)'),
            borderWidth: data.map(d => d.fwd_ret === self.smSelectedFwd ? 2 : 0),
            borderColor: '#3498db',
          }],
        },
        options: {
          responsive: true, maintainAspectRatio: false, animation: false,
          onClick: (evt, elements) => {
            if (elements.length > 0) {
              const clicked = data[elements[0].index].fwd_ret;
              const newFwd = clicked === self.smSelectedFwd ? '' : clicked;
              self.loadSmSummary(self.smSelectedMetric, newFwd, self.smSelectedTicker);
            }
          },
          plugins: {
            legend: { display: false },
            tooltip: {
              backgroundColor: 'rgba(20,20,20,0.95)', borderColor: '#444', borderWidth: 1,
              callbacks: { afterLabel: self._smTooltipCallback(data) },
            },
          },
          scales: {
            x: { ticks: { color: '#888', font: { size: 8 } },
                 grid: { color: 'rgba(255,255,255,0.03)' }, border: { color: 'transparent' } },
            y: { ticks: { color: '#888', font: { size: 9 } },
                 grid: { color: 'rgba(255,255,255,0.05)' }, border: { color: 'transparent' } },
          },
        },
      });
    },

    // ── Feature Clusters ──────────────────────────────────────────────
    async loadClusters() {
      try {
        const r = await fetch('/api/oi-analysis/feature-clusters');
        if (r.ok) this.ifClusters = await r.json();
      } catch (_) {}
    },

    // ── 2F Interaction Scanner ────────────────────────────────────────
    async run2fScan() {
      const metrics = [...this.$store.metricPicker.selected];
      if (metrics.length < 2) return;
      this.ifLastScannedMetrics = metrics;
      try {
        const r = await fetch('/api/oi-analysis/run-2f-scan', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ metrics }),
        });
        if (r.ok) {
          const d = await r.json();
          this.ifStatus = { running: true,
                            message: d.message, last_run: this.ifStatus.last_run };
          this._ifStartPoll();
        }
      } catch (_) {}
    },

    _ifStartPoll() {
      if (this.ifPollTimer) return;
      this.ifPollTimer = setInterval(async () => {
        try {
          const r = await fetch('/api/oi-analysis/2f-scan-status');
          if (r.ok) {
            this.ifStatus = await r.json();
            if (!this.ifStatus.running) {
              clearInterval(this.ifPollTimer);
              this.ifPollTimer = null;
              await this.loadInteractionMatrix();
            }
          }
        } catch (_) {}
      }, 3000);
    },

    async loadInteractionMatrix() {
      const params = new URLSearchParams();
      if (this.ifFwdFilter) params.set('fwd_ret', this.ifFwdFilter);
      (this.ifLastScannedMetrics || []).forEach(m => params.append('metrics', m));
      try {
        const r = await fetch('/api/oi-analysis/interaction-matrix?' + params);
        if (r.ok) this.ifRows = await r.json();
      } catch (_) {}
    },

    async drillInteraction(row) {
      this.ifSelected = row;
      this.ifDetailTicker = '';
      this.ifDetailFwd = row.fwd_ret || '';
      await this.loadInteractionDetail();
    },

    async loadInteractionDetail() {
      if (!this.ifSelected) return;
      const params = new URLSearchParams({
        feat_a: this.ifSelected.feat_a,
        feat_b: this.ifSelected.feat_b,
      });
      if (this.ifDetailTicker) params.set('ticker', this.ifDetailTicker);
      if (this.ifDetailFwd) params.set('fwd_ret', this.ifDetailFwd);
      try {
        const r = await fetch('/api/oi-analysis/interaction-detail?' + params);
        if (r.ok) {
          this.ifDetail = (await r.json()).sort((a, b) => (b.interaction_lift || 0) - (a.interaction_lift || 0));
          this.ifDetailRow = this.ifDetail[0] || null;
          this._pickDetailRow();
        }
      } catch (_) {}
    },

    _pickDetailRow() {
      if (!this.ifDetail.length) { this.ifDetailRow = null; return; }
      const match = this.ifDetail.find(d =>
        (!this.ifDetailTicker || d.ticker === this.ifDetailTicker) &&
        (!this.ifDetailFwd    || d.fwd_ret === this.ifDetailFwd));
      this.ifDetailRow = match || this.ifDetail[0] || null;
    },

    ifQuadrantColor(q) {
      if (!q || q.avg_ret == null) return 'rgba(80,80,80,0.3)';
      const v = q.avg_ret;
      const abs = Math.max(...(this.ifDetailRow?.quadrants || []).map(x => Math.abs(x.avg_ret || 0)), 0.001);
      const t = Math.max(-1, Math.min(1, v / abs));
      if (t >= 0) return `rgba(52,152,219,${0.15 + t * 0.65})`;
      return `rgba(232,67,147,${0.15 + (-t) * 0.65})`;
    },

    ifQuadCell(feat_a_high, feat_b_high) {
      const label = (feat_a_high ? 'H' : 'L') + (feat_b_high ? 'H' : 'L');
      return (this.ifDetailRow?.quadrants || []).find(q => {
        // Label stored as "feat_a_H+feat_b_H" or shorthand "HH"
        if (q.label && q.label.length <= 4) return q.label === label;
        const parts = (q.label || '').split('+');
        const aH = parts[0]?.endsWith('_H');
        const bH = parts[1]?.endsWith('_H');
        return aH === feat_a_high && bH === feat_b_high;
      }) || null;
    },

    drillIntoScore(row) {
      // Set the analysis selectors to this row's values and trigger analysis
      this.ticker = row.ticker;
      this.metric = row.metric;
      this.outcome = row.fwd_ret;
      this.$nextTick(() => this.loadAnalysis());
      // Scroll to top
      document.querySelector('.oi-body')?.scrollTo({ top: 0, behavior: 'smooth' });
    },
  }));
});
