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
    dateFrom: '', dateTo: new Date().toISOString().slice(0, 10),
    // selectedBins20 is the sole selection state (1-20). D1+D10 in 10-bin = bins {1,2,19,20}.
    selectedBins20: new Set([1, 2, 19, 20]),
    equityMode: 'concurrent',   // 'concurrent' | 'non_overlapping'
    equityXMode: 'calendar',   // 'calendar' | 'sequential'
    decileBins: 10,
    decileBinsData: null,

    // Trade Data table view mode + sort. The bin filter is shared with
    // the flat trade view via selectedBins20 already.
    //   'trades'    — flat per-trade list (default; existing behaviour)
    //   'by_ticker' — pivoted: one row per ticker with n / avg_ret /
    //                 win_rate / min / max, sortable on any column.
    tradeView:    'trades',
    tradeSortKey: 'date',
    tradeSortDir: 'desc',

    // Data
    data: null,
    loading: false,
    error: null,
    _charts: {},
    fsChartId: null,

    // Secondary Signal Scanner
    secStatus: { loaded: false, loading: false, error: null },
    secCacheKey: null,
    secBaseline: null,
    secMetrics: [],
    secMaxAbsLift: 0,
    secSelectedMetric: null,
    secDetail: null,
    secDetailLoading: false,
    secBinCount: 10,
    secSelectedSecBins: [10],
    secBubbleMinN: 1,

    async init() {
      // Trade-table column sort: header onclick calls a window function
      // directly since the headers are built via innerHTML (no Alpine bindings).
      // Using window avoids duplicate document listeners if init fires again.
      window._oiTradeSort = (key) => this._tradeSort(key);

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
      // Preserve decileBins and selectedBins20 across re-analyze.
      // decileBinsData is recomputed below once new data arrives.
      this.heatmapData = null;
      this.hmXData = null;
      this.hmYData = null;
      this._destroyCharts();
      // Clear secondary scanner cache when primary analysis changes
      this.secStatus = { loaded: false, loading: false, error: null };
      this.secCacheKey = null;
      this.secBaseline = null;
      this.secMetrics = [];
      this.secSelectedMetric = null;
      this.secDetail = null;
      this.secBinCount = 10;
      this.secSelectedSecBins = [10];
      this.secBubbleMinN = 1;
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
        // Recompute bar chart data for current mode with fresh decile_stats_20.
        this.decileBinsData = this.decileBins !== 10 ? this._computeDecileNBins(this.decileBins) : null;
        await this.$nextTick();
        await this.$nextTick();
        setTimeout(() => this._renderCharts(), 80);
        if (this.heatmapMetric) await this.loadHeatmap();
      } catch (e) {
        this.error = e.message;
      } finally {
        this.loading = false;
      }
    },

    // Map a display-bucket number (1..decileBins) to the set of 20-bin indices it covers.
    _bins20For(displayBin) {
      const g = 20 / this.decileBins;
      const lo = (displayBin - 1) * g + 1;
      return Array.from({length: g}, (_, i) => lo + i);
    },

    // Derive the effective 10-bin decile set from selectedBins20 (for charts using equity_by_decile).
    _effectiveDeciles() {
      const s = new Set();
      for (const b of this.selectedBins20) s.add(Math.ceil(b / 2));
      return s;
    },

    // D1–D10 quick-buttons (hidden in 20-bin mode): toggle both 20-bin members of that decile.
    toggleDecile(d) {
      const lo = d * 2 - 1, hi = d * 2;
      const allOn = this.selectedBins20.has(lo) && this.selectedBins20.has(hi);
      if (allOn) {
        if (this.selectedBins20.size > 2) { this.selectedBins20.delete(lo); this.selectedBins20.delete(hi); }
      } else {
        this.selectedBins20.add(lo); this.selectedBins20.add(hi);
      }
      this.selectedBins20 = new Set(this.selectedBins20);
      this._onDecileChange();
    },

    // isDecileSelected: used by D1–D10 buttons and the decile stats table row highlight.
    isDecileSelected(d) {
      return this.selectedBins20.has(d * 2 - 1) || this.selectedBins20.has(d * 2);
    },

    selectAllDeciles() {
      this.selectedBins20 = new Set(Array.from({length: 20}, (_, i) => i + 1));
      this._onDecileChange();
    },
    selectExtremes() {
      const g = 20 / this.decileBins;
      const lo = Array.from({length: g}, (_, i) => i + 1);
      const hi = Array.from({length: g}, (_, i) => 21 - g + i);
      this.selectedBins20 = new Set([...lo, ...hi]);
      this._onDecileChange();
    },
    selectNone() {
      this.selectedBins20 = new Set();
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
      this._renderTradeTable();
      if (this.secStatus.loaded && !this.secStatus.loading) this.secScan();
    },


    setEquityMode(m)  { this.equityMode = m;  this._renderEquity(); this._renderDrawdown(); this._renderRollingCorr(); },
    setEquityXMode(m) { this.equityXMode = m; this._renderEquity(); },

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
      const g = 20 / bins;  // group size: bins 20→1, 10→2, 5→4
      const stats = (this.decileBinsData || this.data.decile_stats || []).filter(d => d);
      const avgs = stats.map(d => d.avg_ret * 100);
      const self = this;

      // A display bucket is selected if ANY of its 20-bin members are in selectedBins20.
      const _isSelected = (d) => {
        const lo = (d.bucket - 1) * g + 1;
        for (let b = lo; b < lo + g; b++) { if (self.selectedBins20.has(b)) return true; }
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
            const lo = (d.bucket - 1) * g + 1;
            const binSet = Array.from({length: g}, (_, i) => lo + i);
            const allSelected = binSet.every(b => self.selectedBins20.has(b));
            if (allSelected) {
              if (self.selectedBins20.size > binSet.length) binSet.forEach(b => self.selectedBins20.delete(b));
            } else {
              binSet.forEach(b => self.selectedBins20.add(b));
            }
            self.selectedBins20 = new Set(self.selectedBins20);
            self._onDecileChange();
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

    // Aggregate decile_stats_20 (always 20 bins) into n display groups.
    _computeDecileNBins(n) {
      const ds20 = (this.data?.decile_stats_20 || []).filter(Boolean);
      if (!ds20.length) return null;
      const g = 20 / n;
      const bins = [];
      for (let i = 0; i < n; i++) {
        const group = ds20.slice(i * g, (i + 1) * g).filter(Boolean);
        if (!group.length) { bins.push(null); continue; }
        const totalN = group.reduce((a, d) => a + d.n, 0);
        if (!totalN) { bins.push(null); continue; }
        bins.push({
          bucket:   i + 1,
          n:        totalN,
          avg_ret:  group.reduce((a, d) => a + d.avg_ret * d.n, 0) / totalN,
          win_rate: group.reduce((a, d) => a + d.win_rate * d.n, 0) / totalN,
          sharpe:   group.reduce((a, d) => a + (d.sharpe || 0), 0) / group.length,
          std_dev:  group.reduce((a, d) => a + (d.std_dev || 0), 0) / group.length,
          min_val:  group[0].min_val,
          max_val:  group[group.length - 1].max_val,
        });
      }
      return bins;
    },

    setDecileBins(n) {
      if (!this.data) return;
      // Translate selectedBins20 to new granularity before changing decileBins.
      // Map each currently-selected 20-bin index to its display bucket in the new mode,
      // then expand back to full 20-bin groups — so "top bucket" stays "top bucket".
      const newG = 20 / n;
      const mappedDisplayBuckets = new Set();
      for (const b of this.selectedBins20) mappedDisplayBuckets.add(Math.ceil(b / newG));
      const newBins20 = new Set();
      for (const db of mappedDisplayBuckets) {
        const lo = (db - 1) * newG + 1;
        for (let b = lo; b < lo + newG; b++) newBins20.add(b);
      }
      this.selectedBins20 = newBins20;
      this.decileBins = n;
      this.decileBinsData = n === 10 ? null : this._computeDecileNBins(n);
      this._renderDecileBar();
      this._renderEquity();
      this._renderYearly();
      this._renderDOW();
      this._renderActivity();
    },

    // Core equity curve builder from any trade_calendar subset.
    _getEquityCurveFromCal(cal, mode) {
      const horizon = this.data.horizon || 1;
      if (!cal.length) return { points: [], n_trades: 0, cum_return: 0, win_rate: 0 };
      const sorted = cal.slice().sort((a, b) => a.date.localeCompare(b.date));
      let cum = 0, wins = 0, lastDate = null;
      const points = [];
      for (const c of sorted) {
        if (mode === 'non_overlapping' && lastDate !== null) {
          const diffDays = (new Date(c.date) - new Date(lastDate)) / 86400000;
          if (diffDays < horizon) continue;
        }
        lastDate = c.date;
        cum += c.ret;
        if (c.ret > 0) wins++;
        points.push({ date: c.date, value: cum });
      }
      const n = points.length;
      return { points, n_trades: n, cum_return: cum, win_rate: n ? wins / n : 0 };
    },

    _getEquity20Curve(bin, mode) {
      const cal = (this.data.trade_calendar || []).filter(c => c.decile20 === bin);
      return this._getEquityCurveFromCal(cal, mode);
    },

    _renderEquity() {
      const el = document.getElementById('chart-equity');
      if (!el || !this.data) return;
      if (this._charts['equity']) this._charts['equity'].destroy();

      const cal = this.data.trade_calendar || [];
      const has20 = !!(cal[0]?.decile20);
      const g = 20 / this.decileBins;
      const binLabel = this.decileBins === 20 ? 'B' : 'D';
      const spotSeries = this.data.spot_series || [];
      const colors = ['#e74c3c','#e67e22','#f1c40f','#2ecc71','#1abc9c',
                       '#3498db','#9b59b6','#e84393','#fd79a8','#636e72'];

      // Which display buckets are selected?
      const selectedDisplayBuckets = new Set();
      for (const b of this.selectedBins20) selectedDisplayBuckets.add(Math.ceil(b / g));

      // Build equity curve per display bucket
      const eqData = {};
      for (const db of selectedDisplayBuckets) {
        if (has20) {
          const lo = (db - 1) * g + 1, hi = db * g;
          const binCal = cal.filter(c => c.decile20 >= lo && c.decile20 <= hi);
          const concCurve = this._getEquityCurveFromCal(binCal, 'concurrent');
          eqData[db] = {
            concurrent:      concCurve,
            non_overlapping: this._getEquityCurveFromCal(binCal, 'non_overlapping'),
          };
          // Sanity check: cumulative return must equal sum of individual returns.
          // If this fails it means the data source for the bar chart and equity diverged.
          const ds20Group = (this.data.decile_stats_20 || []).slice(lo - 1, hi).filter(Boolean);
          if (ds20Group.length) {
            const expectedCum = ds20Group.reduce((a, d) => a + d.avg_ret * d.n, 0);
            const actualCum = concCurve.cum_return;
            if (Math.abs(expectedCum - actualCum) > 0.01) {
              console.warn(`[equity sanity] B${lo}-B${hi}: expected Σ=${expectedCum.toFixed(4)}, got ${actualCum.toFixed(4)} — data source mismatch!`);
            }
          }
        } else {
          // Fallback: server-side equity_by_decile (10-bin only)
          const ebd = this.data.equity_by_decile || {};
          eqData[db] = ebd[db] || { concurrent: { points: [] }, non_overlapping: { points: [] } };
        }
      }

      // Build timeline
      let timeline;
      if (this.equityXMode === 'calendar') {
        // Union of all trade dates across selected curves — one x position per unique date
        const allDates = new Set();
        for (const db of selectedDisplayBuckets) {
          const pts = eqData[db]?.[this.equityMode]?.points || [];
          for (const p of pts) allDates.add(p.date);
        }
        if (spotSeries.length > 0) spotSeries.forEach(s => allDates.add(s.date));
        timeline = Array.from(allDates).sort();
      } else {
        // Sequential: spot_series (all trading days) or longest equity curve
        timeline = spotSeries.length > 0 ? spotSeries.map(s => s.date) : [];
        if (!timeline.length) {
          let longest = [];
          for (const db of selectedDisplayBuckets) {
            const pts = eqData[db]?.[this.equityMode]?.points || [];
            if (pts.length > longest.length) longest = pts;
          }
          timeline.push(...longest.map(p => p.date));
        }
      }
      const dateIndex = {};
      timeline.forEach((d, i) => dateIndex[d] = i);

      const datasets = [];
      for (const db of selectedDisplayBuckets) {
        const eq = eqData[db]?.[this.equityMode];
        if (!eq?.points?.length) continue;
        const mapped = new Array(timeline.length).fill(null);
        for (const p of eq.points) {
          const idx = dateIndex[p.date];
          if (idx !== undefined) mapped[idx] = p.value * 100;
        }
        datasets.push({
          label: `${binLabel}${db}`,
          data: mapped,
          borderColor: colors[(db - 1) % colors.length],
          backgroundColor: 'transparent',
          borderWidth: 1.5, pointRadius: 0, tension: 0.1, spanGaps: true,
        });
      }

      // Aggregate line
      if (selectedDisplayBuckets.size >= 2) {
        const selArr = Array.from(selectedDisplayBuckets);
        const carried = selArr.map(db => {
          const pts = eqData[db]?.[this.equityMode]?.points || [];
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
        const mapped = timeline.map((_, i) => carried.reduce((a, c) => a + c[i], 0) * 100);
        datasets.push({
          label: 'Aggregate', data: mapped,
          borderColor: '#fff', backgroundColor: 'transparent',
          borderWidth: 2.5, pointRadius: 0, tension: 0.1, borderDash: [6, 3],
        });
      }

      if (spotSeries.length > 0) {
        datasets.push({
          label: 'Spot Price', data: spotSeries.map(s => s.value),
          borderColor: 'rgba(255,255,255,0.15)', backgroundColor: 'transparent',
          borderWidth: 1, pointRadius: 0, tension: 0.1, yAxisID: 'y1',
        });
      }

      this._charts['equity'] = new Chart(el, {
        type: 'line',
        data: { labels: timeline.map(d => d?.slice(0, 7)), datasets },
        options: {
          responsive: true, maintainAspectRatio: false, animation: false,
          plugins: {
            legend: { labels: { color: '#aaa', font: { size: 10 } } },
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
                 title: { display: true, text: 'Cum Return %', color: '#888', font: { size: 9 } } },
            y1: { display: spotSeries.length > 0, position: 'right',
                  grid: { drawOnChartArea: false },
                  ticks: { color: 'rgba(255,255,255,0.2)', font: { size: 8 } },
                  title: { display: true, text: 'Spot', color: 'rgba(255,255,255,0.2)', font: { size: 8 } } },
          },
        },
      });
    },

    _renderYearly() {
      const el = document.getElementById('chart-yearly');
      if (!el || !this.data?.trade_calendar) return;
      if (this._charts['yearly']) this._charts['yearly'].destroy();

      const cal = this.data.trade_calendar || [];
      const has20y = !!(cal[0]?.decile20);
      const filtered = has20y && this.selectedBins20.size > 0
        ? cal.filter(c => this.selectedBins20.has(c.decile20))
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
      const g_y = 20 / this.decileBins;
      const _lbl = this.decileBins === 20 ? 'B' : 'D';
      const selDispY = new Set([...this.selectedBins20].map(b => Math.ceil(b / g_y)));
      const decLabel = selDispY.size > 0 ? Array.from(selDispY).sort((a,b)=>a-b).map(d=>_lbl+d).join('+') : 'All';

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
      const effDec = this._effectiveDeciles();
      const spotSeries = this.data.spot_series || [];
      const colors = ['#e74c3c','#e67e22','#f1c40f','#2ecc71','#1abc9c',
                       '#3498db','#9b59b6','#e84393','#fd79a8','#636e72'];

      const timeline = spotSeries.length > 0 ? spotSeries.map(s => s.date) : [];
      if (!timeline.length) {
        for (const d of effDec) {
          const pts = eqData[d]?.[this.equityMode]?.points || [];
          if (pts.length > timeline.length) { timeline.length = 0; timeline.push(...pts.map(p => p.date)); }
        }
      }
      const dateIndex = {};
      timeline.forEach((d, i) => dateIndex[d] = i);

      const datasets = [];
      for (const d of effDec) {
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

      if (effDec.size >= 2) {
        const selArr = Array.from(effDec);
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
          + `&outcome=${encodeURIComponent(this.outcome)}&bins=${this.heatmapBins}`
          + (this.dateFrom ? `&date_from=${this.dateFrom}` : '')
          + (this.dateTo   ? `&date_to=${this.dateTo}`     : ''));
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
        if (rx.ok) {
          const d = await rx.json();
          console.log('[hmBins1d X]', this.metric, d.error || `n=${d.n} buckets=${(d.buckets||[]).length}`, d);
          this.hmXData = d.buckets || null;
        } else {
          console.warn('[hmBins1d X] HTTP', rx.status, await rx.text());
        }
        if (ry.ok) {
          const d = await ry.json();
          console.log('[hmBins1d Y]', this.heatmapMetric, d.error || `n=${d.n} buckets=${(d.buckets||[]).length}`, d);
          this.hmYData = d.buckets || null;
        } else {
          console.warn('[hmBins1d Y] HTTP', ry.status, await ry.text());
        }
      } catch (e) { console.error('[hmBins1d] fetch failed', e); }
      await this.$nextTick();
      this._renderHmBar1d('chart-hm-x', this.hmXData, this.metric);
      this._renderHmBar1d('chart-hm-y', this.hmYData, this.heatmapMetric);
    },

    _renderHmBar1d(canvasId, buckets, title, retries = 6) {
      const el = document.getElementById(canvasId);
      if (!el) {
        // Canvas lives inside an x-if that's flipped by heatmapData changes
        // — Alpine sometimes hasn't (re)created it yet by the time we get
        // here. Retry briefly so the chart actually paints.
        if (retries > 0) {
          setTimeout(() => this._renderHmBar1d(canvasId, buckets, title, retries - 1), 80);
        } else {
          console.warn('[hmBar1d]', canvasId, 'canvas never appeared');
        }
        return;
      }
      if (!buckets?.length) {
        console.warn('[hmBar1d]', canvasId, 'no buckets — buckets =', buckets);
        return;
      }
      console.log('[hmBar1d]', canvasId, 'rendering with', buckets.filter(Boolean).length, 'non-null buckets');
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

      const effDec2 = this._effectiveDeciles();
      const allRets = [];
      const selRets = [];
      for (const d of (this.data.decile_stats || [])) {
        if (!d?.returns) continue;
        allRets.push(...d.returns.map(r => r * 100));
        if (effDec2.has(d.bucket)) {
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

      const decLabel = effDec2.size > 0
        ? Array.from(effDec2).sort((a,b)=>a-b).map(d=>'D'+d).join('+') : 'None';

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
      const has20c = !!(cal[0]?.decile20);
      const filtered = has20c && this.selectedBins20.size > 0
        ? cal.filter(c => this.selectedBins20.has(c.decile20))
        : cal;

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
      const has20d = !!(this.data.dow_data?.[0]?.decile20);
      const filtered = has20d && this.selectedBins20.size > 0
        ? this.data.dow_data.filter(d => this.selectedBins20.has(d.decile20))
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

      const has20a = !!(cal[0]?.decile20);
      const filtered = has20a && this.selectedBins20.size > 0
        ? cal.filter(c => this.selectedBins20.has(c.decile20))
        : cal;
      if (!filtered.length) return;

      const horizon = this.data.horizon || 1;

      // Count entries per trade-entry date
      const entriesByDate = {};
      for (const c of filtered) entriesByDate[c.date] = (entriesByDate[c.date] || 0) + 1;

      // True calendar x-axis: spot_series = all trading days (single-ticker);
      // ALL mode: union of entry dates (covers most trading days across tickers)
      const spotSeries = this.data.spot_series || [];
      const tradingDays = spotSeries.length > 0
        ? spotSeries.map(s => s.date)
        : [...new Set(cal.map(c => c.date))].sort();

      const entered = tradingDays.map(d => entriesByDate[d] || 0);

      // Open positions on day i = entries in the N-trading-day window ending at i.
      // A trade entered on day T is open for exactly N trading days (T..T+N-1),
      // so on day D (index i) count entries from index max(0, i-N+1) to i.
      // Maximum open count can never exceed N.
      const open = tradingDays.map((_, i) => {
        const startIdx = Math.max(0, i - horizon + 1);
        let count = 0;
        for (let j = startIdx; j <= i; j++) count += entriesByDate[tradingDays[j]] || 0;
        return count;
      });

      this._charts['activity'] = new Chart(el, {
        type: 'bar',
        data: {
          labels: tradingDays.map(d => d.slice(0, 7)),
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
                title: ctx => tradingDays[ctx[0]?.dataIndex] || '',
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

    // ── Trade data table ──────────────────────────────────────────────────
    setTradeView(mode) {
      if (mode !== 'trades' && mode !== 'by_ticker') return;
      if (this.tradeView === mode) return;
      this.tradeView = mode;
      // Reset sort to a sensible default for each view.
      if (mode === 'by_ticker') {
        this.tradeSortKey = 'avg_ret';
        this.tradeSortDir = 'desc';
      } else {
        this.tradeSortKey = 'date';
        this.tradeSortDir = 'desc';
      }
      this._renderTradeTable();
    },

    _tradeSort(key) {
      // Click a column header to sort. Same key flips direction.
      if (this.tradeSortKey === key) {
        this.tradeSortDir = this.tradeSortDir === 'desc' ? 'asc' : 'desc';
      } else {
        this.tradeSortKey = key;
        this.tradeSortDir = 'desc';
      }
      this._renderTradeTable();
    },

    _renderTradeTable() {
      const headEl = document.getElementById('trade-table-head');
      const bodyEl = document.getElementById('trade-table-body');
      const cntEl  = document.getElementById('trade-table-count');
      if (!bodyEl || !this.data?.trade_calendar) return;

      const cal = this.data.trade_calendar || [];
      const has20 = !!(cal[0]?.decile20);
      // Bin filter from the decile pane above — same filter both views see.
      const filtered = has20 && this.selectedBins20.size > 0
        ? cal.filter(c => this.selectedBins20.has(c.decile20))
        : cal;

      if (this.tradeView === 'by_ticker') {
        this._renderTradeTableByTicker(headEl, bodyEl, cntEl, filtered);
      } else {
        this._renderTradeTableFlat(headEl, bodyEl, cntEl, filtered);
      }
    },

    _renderTradeTableFlat(headEl, bodyEl, cntEl, filtered) {
      const key = this.tradeSortKey;
      const dir = this.tradeSortDir === 'asc' ? 1 : -1;
      const strKeys = new Set(['date', 'ticker', 'exit_date']);
      const sortVal = (c) => {
        switch (key) {
          case 'date':       return c.date       || '';
          case 'ticker':     return c.ticker     || '';
          case 'metric_val': return c.metric_val ?? -Infinity;
          case 'spot_entry': return c.spot_entry ?? -Infinity;
          case 'spot_exit':  return c.spot_exit  ?? -Infinity;
          case 'ret':        return c.ret        ?? -Infinity;
          case 'exit_date':  return c.exit_date  || '';
          case 'bin':        return c.decile20   || c.decile || 0;
          default:           return '';
        }
      };
      const sorted = filtered.slice().sort((a, b) => {
        const va = sortVal(a), vb = sortVal(b);
        if (strKeys.has(key)) return dir * String(va).localeCompare(String(vb));
        return dir * (va - vb);
      });
      const LIMIT = 250;
      const rows = sorted.slice(0, LIMIT);

      if (cntEl) {
        cntEl.textContent = filtered.length > LIMIT
          ? `Showing ${LIMIT} of ${filtered.length.toLocaleString()} trades — export CSV for all`
          : `${filtered.length.toLocaleString()} trades`;
      }
      // Column widths
      const colsEl = document.getElementById('trade-table-cols');
      if (colsEl) {
        colsEl.innerHTML = `
          <col style="width:11%">
          <col style="width:9%">
          <col>
          <col style="width:11%">
          <col style="width:11%">
          <col style="width:11%">
          <col style="width:11%">
          <col style="width:7%">
        `;
      }
      const arrow = (k) => this.tradeSortKey === k ? (this.tradeSortDir === 'desc' ? ' ▼' : ' ▲') : '';
      const hdr = (k, label, isNum) => {
        const color = this.tradeSortKey === k ? '#3498db' : 'var(--dim)';
        const cls   = isNum ? 'class="num"' : '';
        const align = isNum ? '' : 'text-align:left;';
        return `<th ${cls}
                    style="${align}color:${color};font-weight:600;cursor:pointer;user-select:none"
                    onclick="window._oiTradeSort('${k}')">
                  ${label}${arrow(k)}
                </th>`;
      };
      if (headEl) {
        headEl.innerHTML = `<tr style="border-bottom:1px solid var(--border)">
          ${hdr('date',       'Date',                  false)}
          ${hdr('ticker',     'Ticker',                false)}
          ${hdr('metric_val', this.metric || 'Metric', true)}
          ${hdr('spot_entry', 'Entry Spot',            true)}
          ${hdr('spot_exit',  'Exit Spot',             true)}
          ${hdr('ret',        'Ret %',                 true)}
          ${hdr('exit_date',  'Exit Date',             false)}
          ${hdr('bin',        'Bin',                   true)}
        </tr>`;
      }
      bodyEl.innerHTML = rows.map(c => {
        const entrySpot = c.spot_entry ?? null;
        const exitSpot  = c.spot_exit  ?? null;
        const exitDate  = c.exit_date  || '';
        const retPct    = (c.ret * 100).toFixed(3);
        const sign      = c.ret >= 0 ? '+' : '';
        const color     = c.ret >= 0 ? '#3498db' : '#e84393';
        return `<tr>
          <td>${c.date}</td>
          <td>${c.ticker || ''}</td>
          <td class="num">${c.metric_val != null ? c.metric_val.toFixed(4) : ''}</td>
          <td class="num">${entrySpot != null ? entrySpot.toLocaleString('en-US', {minimumFractionDigits:2, maximumFractionDigits:2}) : ''}</td>
          <td class="num">${exitSpot  != null ? exitSpot .toLocaleString('en-US', {minimumFractionDigits:2, maximumFractionDigits:2}) : ''}</td>
          <td class="num" style="color:${color}">${sign}${retPct}%</td>
          <td>${exitDate}</td>
          <td class="num" style="color:#888">${c.decile20 || c.decile || ''}</td>
        </tr>`;
      }).join('');
    },

    _renderTradeTableByTicker(headEl, bodyEl, cntEl, filtered) {
      // Group filtered rows by ticker → aggregate. Sort by selected column.
      const groups = new Map();
      for (const c of filtered) {
        const k = c.ticker || '(unknown)';
        let g = groups.get(k);
        if (!g) { g = []; groups.set(k, g); }
        g.push(c.ret);
      }
      const stats = [];
      for (const [ticker, rets] of groups.entries()) {
        const n = rets.length;
        if (!n) continue;
        let sum = 0, wins = 0, minV = rets[0], maxV = rets[0];
        for (const r of rets) {
          sum += r;
          if (r > 0) wins += 1;
          if (r < minV) minV = r;
          if (r > maxV) maxV = r;
        }
        stats.push({
          ticker,
          n,
          avg_ret:  sum / n,
          win_rate: wins / n,
          min_ret:  minV,
          max_ret:  maxV,
        });
      }
      const dir = this.tradeSortDir === 'asc' ? 1 : -1;
      const key = this.tradeSortKey;
      stats.sort((a, b) => {
        const va = a[key], vb = b[key];
        if (key === 'ticker') return dir * String(va).localeCompare(String(vb));
        return dir * ((va ?? 0) - (vb ?? 0));
      });

      if (cntEl) {
        const binsTxt = (this.selectedBins20.size === 0)
          ? 'all bins'
          : `${this.selectedBins20.size} bin${this.selectedBins20.size > 1 ? 's' : ''} selected`;
        cntEl.textContent = `${stats.length} tickers · ${filtered.length.toLocaleString()} trades · ${binsTxt}`;
      }

      // Column widths — every column gets a deterministic share so the
      // headers and cells stay aligned regardless of content length.
      const colsEl = document.getElementById('trade-table-cols');
      if (colsEl) {
        colsEl.innerHTML = `
          <col>                    <!-- Ticker (flex) -->
          <col style="width:11%">  <!-- N -->
          <col style="width:18%">  <!-- Avg Ret % -->
          <col style="width:15%">  <!-- Win Rate -->
          <col style="width:15%">  <!-- Min Ret % -->
          <col style="width:15%">  <!-- Max Ret % -->
        `;
      }
      // Sortable headers — arrow on the active column, blue highlight.
      const arrow = (k) => this.tradeSortKey === k
        ? (this.tradeSortDir === 'desc' ? ' ▼' : ' ▲') : '';
      const hdr = (k, label, isNum) => {
        const color = this.tradeSortKey === k ? '#3498db' : 'var(--dim)';
        const cls   = isNum ? 'class="num"' : '';
        const align = isNum ? '' : 'text-align:left;';
        return `<th ${cls}
                    style="${align}color:${color};font-weight:600;cursor:pointer;user-select:none"
                    onclick="window._oiTradeSort('${k}')">
                  ${label}${arrow(k)}
                </th>`;
      };
      if (headEl) {
        headEl.innerHTML = `<tr style="border-bottom:1px solid var(--border)">
          ${hdr('ticker',  'Ticker',    false)}
          ${hdr('n',       'N',         true)}
          ${hdr('avg_ret', 'Avg Ret %', true)}
          ${hdr('win_rate','Win Rate',  true)}
          ${hdr('min_ret', 'Min Ret %', true)}
          ${hdr('max_ret', 'Max Ret %', true)}
        </tr>`;
      }
      bodyEl.innerHTML = stats.map(s => {
        const avgPct  = (s.avg_ret * 100).toFixed(3);
        const minPct  = (s.min_ret * 100).toFixed(2);
        const maxPct  = (s.max_ret * 100).toFixed(2);
        const avgSign = s.avg_ret >= 0 ? '+' : '';
        const avgClr  = s.avg_ret >= 0 ? '#3498db' : '#e84393';
        const wrPct   = (s.win_rate * 100).toFixed(1);
        const wrClr   = s.win_rate >= 0.5 ? '#3498db' : '#e84393';
        return `<tr>
          <td style="font-weight:600">${s.ticker}</td>
          <td class="num">${s.n}</td>
          <td class="num" style="color:${avgClr};font-weight:600">${avgSign}${avgPct}%</td>
          <td class="num" style="color:${wrClr}">${wrPct}%</td>
          <td class="num" style="color:#888">${minPct}%</td>
          <td class="num" style="color:#888">${maxPct}%</td>
        </tr>`;
      }).join('');
    },

    exportTradeCSV() {
      if (!this.data?.trade_calendar) return;
      const cal = this.data.trade_calendar || [];
      const has20 = !!(cal[0]?.decile20);
      const filtered = has20 && this.selectedBins20.size > 0
        ? cal.filter(c => this.selectedBins20.has(c.decile20))
        : cal;

      const metric = this.metric || 'metric';

      const header = `trade_date,ticker,${metric},entry_spot,exit_spot,ret_pct,exit_date,bin20`;
      const rows = filtered.slice().sort((a, b) => a.date.localeCompare(b.date)).map(c => {
        const entrySpot = c.spot_entry ?? '';
        const exitSpot = c.spot_exit != null ? c.spot_exit.toFixed(2) : '';
        return [
          c.date, c.ticker || '', c.metric_val ?? '', entrySpot,
          exitSpot, (c.ret * 100).toFixed(6), c.exit_date || '', c.decile20 || c.decile || '',
        ].join(',');
      });

      const csv = [header, ...rows].join('\n');
      const blob = new Blob([csv], { type: 'text/csv' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `trades_${this.ticker}_${this.metric}_${new Date().toISOString().slice(0,10)}.csv`;
      a.click();
      URL.revokeObjectURL(url);
    },

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
      this._renderTradeTable();
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
      { key: 'composite_score', label: 'Score',    align: 'right'  },
      { key: 'ticker',          label: 'Ticker',   align: 'left'   },
      { key: 'metric',          label: 'Metric',   align: 'left'   },
      { key: 'fwd_ret',         label: 'Fwd Ret',  align: 'left'   },
      { key: 'pattern',         label: 'Pattern',  align: 'left'   },
      { key: 'spearman_r',      label: 'Spearman', align: 'right'  },
      { key: 'monotonicity',    label: 'Mono',     align: 'right'  },
      { key: 'yearly_pct',      label: 'Consist',  align: 'right'  },
      { key: 'd10_d1_spread',   label: 'D10-D1',   align: 'right'  },
      { key: 'd10_wr',          label: 'D10 WR',   align: 'right'  },
      { key: 'best_sharpe',     label: 'Sharpe',   align: 'right'  },
      { key: 'n_obs',           label: 'N',        align: 'right'  },
      { key: 'mi',              label: 'MI',       align: 'right'  },
      { key: 'pearson_r',       label: 'Pearson',  align: 'right'  },
      { key: 'loyo_fragile',    label: 'LOYO',     align: 'center' },
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
          // loadSmSummary internally calls loadScoreMatrix, so the table
          // populates as part of this single await. The setTimeout below
          // re-renders the two ticker charts after layout settles.
          await this.loadSmSummary();
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

    // ── Secondary Signal Scanner ──────────────────────────────────────────────
    _secFilteredDates() {
      const cal = this.data?.trade_calendar || [];
      const has20 = !!(cal[0]?.decile20);
      const entries = (!has20 || this.selectedBins20.size === 0)
        ? cal
        : cal.filter(c => this.selectedBins20.has(c.decile20));
      // Always encode as "ticker|date" so the backend can filter per-(ticker,date) in ALL mode.
      return entries.map(c => `${c.ticker}|${c.date}`);
    },

    async secLoad() {
      if (!this.data || this.secStatus.loading) return;
      this.secStatus = { loaded: false, loading: true, error: null };
      this.secSelectedMetric = null;
      this.secDetail = null;
      this.secSelectedSecBins = [this.secBinCount];
      try {
        const r = await fetch('/api/oi-analysis/secondary-load', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            ticker:         this.ticker,
            metric:         this.metric,
            outcome:        this.outcome,
            date_from:      this.dateFrom || '',
            date_to:        this.dateTo || '',
            filtered_dates: this._secFilteredDates(),
          }),
        });
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        const d = await r.json();
        if (d.error) throw new Error(d.error);
        this.secCacheKey = d.cache_key;
        this.secBaseline = d.baseline;
        this.secMetrics  = d.metrics || [];
        this.secMaxAbsLift = Math.max(0.0001, ...this.secMetrics.map(m => Math.abs(m.lift)));
        this.secStatus = { loaded: true, loading: false, error: null };
        await this.$nextTick();
        await this.$nextTick();
        setTimeout(() => this._renderSecBar(), 60);
      } catch (e) {
        this.secStatus = { loaded: false, loading: false, error: e.message };
      }
    },

    async secScan() {
      if (!this.secCacheKey || this.secStatus.loading) return;
      this.secStatus.loading = true;
      try {
        const r = await fetch('/api/oi-analysis/secondary-scan', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            cache_key:      this.secCacheKey,
            filtered_dates: this._secFilteredDates(),
            ticker:         this.ticker,
          }),
        });
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        const d = await r.json();
        if (d.error) {
          if (d.error === 'cache_miss') {
            this.secStatus = { loaded: false, loading: false, error: null };
            return;
          }
          throw new Error(d.error);
        }
        this.secBaseline = d.baseline;
        this.secMetrics  = d.metrics || [];
        this.secMaxAbsLift = Math.max(0.0001, ...this.secMetrics.map(m => Math.abs(m.lift)));
        // Reset detail if the selected metric's position changed significantly
        if (this.secSelectedMetric) await this.secDrillMetric(this.secSelectedMetric, false);
        this._renderSecBar();
      } catch (_) {}
      finally { this.secStatus.loading = false; }
    },

    async secDrillMetric(metricName, resetBins = true) {
      if (!this.secCacheKey) return;
      this.secSelectedMetric = metricName;
      if (resetBins) this.secSelectedSecBins = [5];
      this.secDetailLoading = true;
      this.secDetail = null;
      try {
        const r = await fetch('/api/oi-analysis/secondary-detail', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            cache_key:      this.secCacheKey,
            metric_b:       metricName,
            filtered_dates: this._secFilteredDates(),
            sec_bins:       this.secSelectedSecBins,
            sec_bin_count:  this.secBinCount,
            ticker:         this.ticker,
          }),
        });
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        const d = await r.json();
        if (d.error) throw new Error(d.error);
        this.secDetail = d;
        await this.$nextTick();
        await this.$nextTick();
        setTimeout(() => this._renderSecDetail(), 60);
      } catch (_) {}
      finally { this.secDetailLoading = false; }
    },

    async secToggleSecBin(bin) {
      const idx = this.secSelectedSecBins.indexOf(bin);
      if (idx >= 0) {
        if (this.secSelectedSecBins.length > 1) {
          this.secSelectedSecBins = this.secSelectedSecBins.filter(b => b !== bin);
        }
      } else {
        this.secSelectedSecBins = [...this.secSelectedSecBins, bin];
      }
      if (this.secSelectedMetric) await this.secDrillMetric(this.secSelectedMetric, false);
    },

    async secSetBinCount(n) {
      this.secBinCount = n;
      this.secSelectedSecBins = [n];  // reset to top bin
      if (this.secSelectedMetric) await this.secDrillMetric(this.secSelectedMetric, false);
    },

    _renderSecBar() {
      const inner = document.getElementById('sec-bar-inner');
      const canvas = document.getElementById('sec-bar-canvas');
      if (!canvas || !inner || !this.secMetrics.length) return;
      if (this._charts['sec-bar']) { this._charts['sec-bar'].destroy(); delete this._charts['sec-bar']; }

      const metrics = this.secMetrics;
      const lifts   = metrics.map(m => +(m.lift * 100).toFixed(4));
      const colors   = lifts.map(l => l >= 0 ? 'rgba(52,152,219,0.75)' : 'rgba(232,67,147,0.75)');
      const borders  = lifts.map(l => l >= 0 ? '#3498db' : '#e84393');

      // Dynamic width: at least container width, at most barW*n
      const barW = Math.max(10, Math.min(22, 1400 / metrics.length));
      const chartW = Math.max(inner.parentElement.clientWidth || 600, metrics.length * (barW + 3) + 60);
      inner.style.width = chartW + 'px';

      const ctx = canvas.getContext('2d');
      this._charts['sec-bar'] = new Chart(ctx, {
        type: 'bar',
        data: {
          labels: metrics.map(m => m.name),
          datasets: [{
            data: lifts,
            backgroundColor: colors,
            borderColor: borders,
            borderWidth: 1,
            barThickness: barW,
          }],
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          animation: false,
          onClick: (e, elements) => {
            if (!elements.length) return;
            const name = metrics[elements[0].index]?.name;
            if (name) this.secDrillMetric(name);
          },
          plugins: {
            legend: { display: false },
            tooltip: {
              callbacks: {
                label: ctx => {
                  const m = metrics[ctx.dataIndex];
                  return [
                    `Lift: ${(m.lift * 100).toFixed(3)}%`,
                    `WR lift: ${(m.win_lift * 100).toFixed(1)}%`,
                    `Best bin: B${m.top_bin} of ${this.secBinCount}`,
                    `n top bin: ${m.n_top}  total: ${m.n}`,
                  ];
                },
              },
            },
          },
          scales: {
            x: {
              ticks: {
                color: ctx => metrics[ctx.index]?.name === this.secSelectedMetric ? '#3498db' : '#666',
                font: { size: 8, family: 'monospace' },
                maxRotation: 90,
                minRotation: 45,
              },
              grid: { color: '#1e1e1e' },
            },
            y: {
              ticks: { color: '#888', font: { size: 9 }, callback: v => v.toFixed(2) + '%' },
              grid: { color: '#2a2a2a' },
              title: { display: true, text: 'Lift (%)', color: '#555', font: { size: 9 } },
            },
          },
        },
      });
    },

    _renderSecDetail() {
      if (!this.secDetail) return;
      this._renderSecBinsChart();
      this._renderSecEquity();
      this._renderSecYearly();
      this._renderSecActivity();
      this._renderSecBubble();
    },

    _renderSecBubble() {
      const canvas = document.getElementById('sec-bubble-canvas');
      if (!canvas || !this.secDetail?.tickers?.length) return;
      if (this._charts['sec-bubble']) { this._charts['sec-bubble'].destroy(); delete this._charts['sec-bubble']; }

      const minN = this.secBubbleMinN || 1;
      const tickers = this.secDetail.tickers.filter(t => t.n >= minN);
      if (!tickers.length) return;

      // Radius: positive contrib scaled 3–20; negative → 2
      const maxContrib = Math.max(1, ...tickers.filter(t => t.contrib_pct > 0).map(t => t.contrib_pct));
      // Color: pink (#e84393) at wr=0, blue (#3498db) at wr=1
      const mkColor = (wr, a) => {
        const r = Math.round(232 + (52  - 232) * wr);
        const g = Math.round(67  + (152 - 67)  * wr);
        const b = Math.round(147 + (219 - 147) * wr);
        return `rgba(${r},${g},${b},${a})`;
      };

      const datasets = tickers.map(t => ({
        label: t.ticker,
        data: [{ x: t.n, y: +(t.avg_ret * 100).toFixed(4), r: t.contrib_pct > 0 ? Math.max(3, (t.contrib_pct / maxContrib) * 20) : 2 }],
        backgroundColor: mkColor(t.win_rate, 0.65),
        borderColor:     mkColor(t.win_rate, 1),
        borderWidth: 1,
      }));

      this._charts['sec-bubble'] = new Chart(canvas.getContext('2d'), {
        type: 'bubble',
        data: { datasets },
        options: {
          responsive: true, maintainAspectRatio: false, animation: false,
          plugins: {
            legend: { display: false },
            tooltip: {
              backgroundColor: 'rgba(20,20,20,0.95)', borderColor: '#444', borderWidth: 1,
              callbacks: {
                label: ctx => {
                  const t = tickers[ctx.datasetIndex];
                  return [`${t.ticker}  n:${t.n}  avg:${(t.avg_ret*100).toFixed(3)}%  WR:${(t.win_rate*100).toFixed(1)}%  contrib:${t.contrib_pct.toFixed(1)}%`];
                },
              },
            },
          },
          scales: {
            ...this._darkScales(),
            x: { ...this._darkScales().x,
                 title: { display: true, text: 'Trade Count', color: '#888', font: { size: 9 } } },
            y: { ...this._darkScales().y,
                 title: { display: true, text: 'Avg Return %', color: '#888', font: { size: 9 } } },
          },
        },
      });
    },

    secSetBubbleMinN(n) {
      this.secBubbleMinN = +n;
      this._renderSecBubble();
    },

    _renderSecBinsChart() {
      const canvas = document.getElementById('sec-bins-canvas');
      if (!canvas) return;
      if (this._charts['sec-bins']) { this._charts['sec-bins'].destroy(); delete this._charts['sec-bins']; }
      const bins = (this.secDetail.bins || []).filter(b => b);
      if (!bins.length) return;
      const ctx = canvas.getContext('2d');
      const avgRets = bins.map(b => (b.avg_ret || 0) * 100);
      const selected = this.secSelectedSecBins;
      this._charts['sec-bins'] = new Chart(ctx, {
        type: 'bar',
        data: {
          labels: bins.map(b => `B${b.bin}`),
          datasets: [{
            data: avgRets,
            backgroundColor: bins.map(b =>
              selected.includes(b.bin)
                ? (b.avg_ret >= 0 ? 'rgba(52,152,219,0.85)' : 'rgba(232,67,147,0.85)')
                : (b.avg_ret >= 0 ? 'rgba(52,152,219,0.25)' : 'rgba(232,67,147,0.25)')
            ),
            borderColor: bins.map(b => b.avg_ret >= 0 ? '#3498db' : '#e84393'),
            borderWidth: 1,
          }],
        },
        options: {
          responsive: true, maintainAspectRatio: false, animation: false,
          onClick: (e, elements) => {
            if (!elements.length) return;
            this.secToggleSecBin(bins[elements[0].index].bin);
          },
          plugins: {
            legend: { display: false },
            tooltip: {
              callbacks: {
                label: ctx => {
                  const b = bins[ctx.dataIndex];
                  return [`Avg: ${(b.avg_ret*100).toFixed(3)}%`, `WR: ${(b.win_rate*100).toFixed(1)}%`, `n: ${b.n}`];
                },
              },
            },
          },
          scales: {
            x: { ticks: { color: '#888', font: { size: 10 } }, grid: { color: '#222' } },
            y: {
              ticks: { color: '#888', font: { size: 9 }, callback: v => v.toFixed(2) + '%' },
              grid: { color: '#222' },
            },
          },
        },
      });
    },

    _renderSecEquity() {
      const canvas = document.getElementById('sec-equity-canvas');
      if (!canvas || !this.secDetail) return;
      if (this._charts['sec-equity']) { this._charts['sec-equity'].destroy(); delete this._charts['sec-equity']; }
      const eqP = this.secDetail.equity_primary || [];
      const eqC = this.secDetail.equity_combined || [];
      if (!eqP.length) return;
      const ctx = canvas.getContext('2d');

      // Align combined curve to primary date axis: hold last value on non-combined dates
      const cMap = Object.fromEntries(eqC.map(p => [p.date, +(p.value * 100).toFixed(4)]));
      let lastCombined = 0;
      const combinedAligned = eqP.map(p => {
        if (cMap[p.date] !== undefined) lastCombined = cMap[p.date];
        return lastCombined;
      });

      this._charts['sec-equity'] = new Chart(ctx, {
        type: 'line',
        data: {
          labels: eqP.map(p => p.date),
          datasets: [
            {
              label: 'Primary filter',
              data: eqP.map(p => +(p.value * 100).toFixed(4)),
              borderColor: '#3498db',
              backgroundColor: 'rgba(52,152,219,0.08)',
              borderWidth: 1.5,
              pointRadius: 0,
              fill: false,
              tension: 0,
            },
            {
              label: '+ Secondary filter',
              data: combinedAligned,
              borderColor: '#e84393',
              backgroundColor: 'transparent',
              borderWidth: 1.5,
              pointRadius: 0,
              fill: false,
              tension: 0,
              spanGaps: true,
            },
          ],
        },
        options: {
          responsive: true, maintainAspectRatio: false, animation: false,
          plugins: {
            legend: {
              display: true,
              labels: { color: '#888', font: { size: 9 }, boxWidth: 12 },
            },
            tooltip: { mode: 'index', intersect: false },
          },
          scales: {
            x: { display: false },
            y: {
              ticks: { color: '#888', font: { size: 9 }, callback: v => v.toFixed(1) + '%' },
              grid: { color: '#222' },
            },
          },
        },
      });
    },

    _renderSecYearly() {
      const canvas = document.getElementById('sec-yearly-canvas');
      if (!canvas || !this.secDetail?.yearly?.length) return;
      if (this._charts['sec-yearly']) { this._charts['sec-yearly'].destroy(); delete this._charts['sec-yearly']; }
      const yearly = this.secDetail.yearly;
      const ctx = canvas.getContext('2d');
      this._charts['sec-yearly'] = new Chart(ctx, {
        type: 'bar',
        data: {
          labels: yearly.map(y => y.year),
          datasets: [
            {
              label: 'Primary',
              data: yearly.map(y => +(y.primary_avg * 100).toFixed(3)),
              backgroundColor: 'rgba(52,152,219,0.45)',
              borderColor: '#3498db',
              borderWidth: 1,
            },
            {
              label: '+ Secondary',
              data: yearly.map(y => +(y.combined_avg * 100).toFixed(3)),
              backgroundColor: 'rgba(232,67,147,0.45)',
              borderColor: '#e84393',
              borderWidth: 1,
            },
          ],
        },
        options: {
          responsive: true, maintainAspectRatio: false, animation: false,
          plugins: {
            legend: {
              display: true,
              labels: { color: '#888', font: { size: 9 }, boxWidth: 12 },
            },
            tooltip: {
              callbacks: {
                label: ctx => {
                  const y = yearly[ctx.dataIndex];
                  if (ctx.datasetIndex === 0) return [`Avg: ${(y.primary_avg*100).toFixed(3)}%`, `WR: ${(y.primary_wr*100).toFixed(1)}%`, `n: ${y.primary_n}`];
                  return [`Avg: ${(y.combined_avg*100).toFixed(3)}%`, `WR: ${(y.combined_wr*100).toFixed(1)}%`, `n: ${y.combined_n}`];
                },
              },
            },
          },
          scales: {
            x: { ticks: { color: '#888', font: { size: 9 } }, grid: { color: '#222' } },
            y: {
              ticks: { color: '#888', font: { size: 9 }, callback: v => v.toFixed(2) + '%' },
              grid: { color: '#222' },
            },
          },
        },
      });
    },

    _renderSecActivity() {
      const canvas = document.getElementById('sec-activity-canvas');
      if (!canvas || !this.secDetail?.combined_trade_dates?.length) return;
      if (this._charts['sec-activity']) { this._charts['sec-activity'].destroy(); delete this._charts['sec-activity']; }

      const dates = this.secDetail.combined_trade_dates;
      const horizon = this.secDetail.horizon || 1;

      // Entry count per trading day
      const entriesByDate = {};
      for (const d of dates) entriesByDate[d] = (entriesByDate[d] || 0) + 1;

      // True calendar x-axis: spot_series covers all trading days in single-ticker mode.
      // In ALL mode spot_series is empty — fall back to trade_calendar (all tickers, all bins),
      // which spans the full date range. Mirrors primary _renderActivity() exactly.
      const spotSeries = this.data?.spot_series || [];
      const cal = this.data?.trade_calendar || [];
      const tradingDays = spotSeries.length > 0
        ? spotSeries.map(s => s.date)
        : [...new Set(cal.length > 0 ? cal.map(c => c.date) : dates)].sort();

      const entered = tradingDays.map(d => entriesByDate[d] || 0);

      // Open positions on day i = entries in the N-trading-day window [i-N+1 .. i]
      // (exact same logic as the primary _renderActivity)
      const open = tradingDays.map((_, i) => {
        const start = Math.max(0, i - horizon + 1);
        let count = 0;
        for (let j = start; j <= i; j++) count += entriesByDate[tradingDays[j]] || 0;
        return count;
      });

      const ctx = canvas.getContext('2d');
      this._charts['sec-activity'] = new Chart(ctx, {
        type: 'bar',
        data: {
          labels: tradingDays.map(d => d.slice(0, 7)),
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
                title: ctx => tradingDays[ctx[0]?.dataIndex] || '',
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
  }));
});
