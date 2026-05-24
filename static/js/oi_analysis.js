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
    // Page-wide bin mode. Drives every binning analysis on the page so
    // primary / corr explorer / portfolio aggregate all use the same flavor.
    //   'in_sample'    — full-history bin thresholds
    //   'walk_forward' — per-ticker bisect_left against running history, 252d warmup
    pageMode: 'walk_forward',
    // Only used when pageMode === 'train_test'. Defaults to 2024-01-01 so
    // users get a meaningful default test window without typing a date.
    cutoffDate: '2024-01-01',
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
    secMaxAbsScore: 0,
    secSelectedMetric: null,
    secDetail: null,
    secDetailLoading: false,
    secBinCount: 10,
    secSelectedSecBins: [10],
    secBubbleMinN: 1,
    secScanMeta: null,

    // Multi-Metric Correlation Explorer
    corrPanelOpen: false,
    corrBinCount: 20,
    corrMiniData: null,
    corrMiniLoading: false,
    corrSelections: {},
    corrResult: null,
    corrLoading: false,
    corrBubbleMinN: 1,

    // System Portfolio (third tier — persisted research portfolios)
    portfolios: [],          // [{id, name, ticker, outcome, date_from, date_to, system_count}, ...]
    portfolioId: null,       // currently-selected portfolio id (or null)
    portfolio: null,         // {portfolio: {...}, systems: [...]}
    portAggregate: null,     // last /aggregate response
    portLoading: false,
    editingSystemId: null,   // when set, the "Add" button becomes "Save Changes"
    portIsShort: false,      // direction toggle for the system being added/edited
    portBubbleMinN: 1,       // bubble chart min-n filter
    // System Library — anchor-agnostic system templates reusable across portfolios
    librarySystems: [],      // [{id, name, primary_metric, primary_bins, secondaries, ...}]
    libraryExpanded: false,

    // Trade Activity dedupe — when on, a new entry for a ticker is skipped
    // while a previous trade of the same ticker is still inside its horizon.
    // Independent per chart so the user can A/B them visually.
    dedupeConc: { primary: false, sec: false, corr: false, port: false },

    // All-Ticker Metric Bins (top-of-page browser, independent of analysis)
    topBinsExpanded: false,
    topBinsLoading:  false,
    topBinsData:     null,
    topBinsOutcome:  'ret_5d_fwd_oc',

    // Threshold Drift (walk-forward bin boundaries over time)
    tdExpanded:     false,
    tdLoading:      false,
    tdData:         null,
    tdMetric:       '',
    tdOutcome:      'ret_5d_fwd_oc',
    tdBinsToShow:   [1, 20],
    tdMode:         'ratio',   // 'ratio' (all tickers, dimensionless) | 'native_single'
    tdSingleTicker: '',

    // IC.5 — Signal Stability (universe-wide IC leaderboard + scatter)
    // Lazy-loaded on first expand (user-initiated, same as All-Ticker Metric
    // Bins). Also reloads when the section is open and mode changes. Fresh
    // compute takes ~2-3 min on the VPS; cached reads are sub-second.
    icBatchData:      null,
    icBatchLoading:   false,
    icBatchError:     null,
    icBatchKey:       null,           // last-loaded "ticker:outcome:mode:cutoff" key
    icBatchStatus:    null,           // 'not_ready' | 'computing' | 'queued' | 'failed' | 'timeout' | null
    icBatchPollTimer: null,           // setInterval handle for polling
    icBatchPollStart: null,           // Date.now() when polling started
    icBatchRefreshAt: null,           // Date.now() when ⟳ Refresh was last triggered

    // IC.7 — Signal Decomposition (per-ticker breakdown, ALL mode only)
    icDecompData:     null,
    icDecompLoading:  false,
    icDecompError:    null,
    icDecompKey:      null,   // last-loaded "metric:outcome:mode:cutoff"
    icDecompYMode:    'raw',  // 'raw' | 'basket' — Y-axis mode for bubble scatter

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
        this.ticker = 'ALL';
      }
      if (colRes.ok) {
        const cols = await colRes.json();
        this.features = cols.features || [];
        this.outcomes = cols.outcomes || [];
        if (this.features.length) this.metric = this.features[0];
        if (this.outcomes.length) this.outcome = this.outcomes[0];
        // Pre-fill Threshold Drift's metric picker with the first feature.
        if (this.features.length && !this.tdMetric) this.tdMetric = this.features[0];

        // ALWAYS prefer ret_5d_fwd_oc for all outcome pickers.
        const _preferred = 'ret_5d_fwd_oc';
        const _pick = this.outcomes.includes(_preferred) ? _preferred : (this.outcomes[0] || '');
        // Set state AND force the DOM SELECT directly. Belt-and-suspenders
        // because some browsers restore the previously-chosen option from
        // form-state cache even with autocomplete="off", and Alpine's
        // reactivity won't re-fire if state hasn't changed.
        this.outcome        = _pick;  // main analysis outcome
        await this.$nextTick();
        this.topBinsOutcome = _pick;
        this.tdOutcome      = _pick;
        await this.$nextTick();
        // Belt-and-suspenders: walk every SELECT with our marker IDs and
        // force its DOM value directly. Necessary when the browser has
        // form-state cache that overrides JS-assigned defaults.
        const _forceSelect = (id, val) => {
          const el = document.getElementById(id);
          if (el && val) el.value = val;
        };
        _forceSelect('select-topbins-outcome', _pick);
        _forceSelect('select-td-outcome',      _pick);
      }
      // Load score matrix (independent of analysis)
      this.smInit();
      // Portfolios list (third-tier — research portfolios persisted server-side)
      this.loadPortfolios();
      // System library (anchor-agnostic reusable system templates)
      this.loadLibrarySystems();
      // Signal Survey — always-visible; load on init so charts appear without
      // requiring an Analyze click. Single-ticker auto-triggers; ALL stays idle
      // until explicit ⟳ Refresh (OOM guard).
      this.loadIcBatch();
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
      this.corrPanelOpen = false;
      this.corrMiniData = null;
      this.corrSelections = {};
      this.corrResult = null;
      try {
        let url = `/api/oi-analysis/analyze?ticker=${encodeURIComponent(this.ticker)}`
          + `&metric=${encodeURIComponent(this.metric)}`
          + `&outcome=${encodeURIComponent(this.outcome)}`;
        if (this.dateFrom) url += `&date_from=${this.dateFrom}`;
        if (this.dateTo) url += `&date_to=${this.dateTo}`;
        if (this.pageMode === 'walk_forward') url += '&walk_forward=true';
        if (this.pageMode === 'train_test')   url += `&cutoff_date=${encodeURIComponent(this.cutoffDate)}`;
        const r = await fetch(url);
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        this.data = await r.json();
        if (this.data.error) { this.error = this.data.error; return; }
        // Recompute bar chart data for current mode with fresh decile_stats_20.
        this.decileBinsData = this.decileBins !== 10 ? this._computeDecileNBins(this.decileBins) : null;
        await this.$nextTick();
        await this.$nextTick();
        setTimeout(() => this._renderCharts(), 80);
        // IC.5: always reload when key changes or data is absent (section always visible).
        // Clear stale data first so panes show status rather than wrong-mode charts.
        const _icKey = this._icBatchKey();
        if (_icKey !== this.icBatchKey || !this.icBatchData) {
          if (_icKey !== this.icBatchKey) this.icBatchData = null;
          this.loadIcBatch();
        }
        // IC.7: reload decomp when metric/mode changes (ALL mode only; single-ticker has no decomp).
        if (this.ticker === 'ALL') {
          const _dcKey = this._icDecompKey();
          if (_dcKey !== this.icDecompKey || !this.icDecompData) {
            if (_dcKey !== this.icDecompKey) this.icDecompData = null;
            this.loadIcDecomp();
          }
        }
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
      // Preserve charts that aren't tied to the primary analysis — Score
      // Matrix bars (sm-*), System Portfolio visuals (port-*), and the
      // top-of-page Threshold Drift line (td) live independently and
      // shouldn't blank out every time Analyze runs.
      for (const k of Object.keys(this._charts)) {
        // Preserve charts not tied to the primary analysis result.
        if (k.startsWith('sm-') || k.startsWith('port-') || k === 'td' || k.startsWith('ic-')) continue;
        this._charts[k].destroy();
        delete this._charts[k];
      }
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
      // walk_forward / train_test heatmap are only supported in ALL mode
      // (single-ticker /heatmap uses absolute np.percentile edges by design).
      // For single-ticker + non-in-sample pageMode, leave the mode flags off
      // so the heatmap silently runs in-sample rather than 422'ing.
      let wf = '';
      if (this.ticker === 'ALL') {
        if (this.pageMode === 'walk_forward') wf = '&walk_forward=true';
        else if (this.pageMode === 'train_test') wf = `&cutoff_date=${encodeURIComponent(this.cutoffDate)}`;
      }
      try {
        const r = await fetch(
          `/api/oi-analysis/heatmap?ticker=${encodeURIComponent(this.ticker)}`
          + `&metric_x=${encodeURIComponent(this.metric)}`
          + `&metric_y=${encodeURIComponent(this.heatmapMetric)}`
          + `&outcome=${encodeURIComponent(this.outcome)}&bins=${this.heatmapBins}`
          + (this.dateFrom ? `&date_from=${this.dateFrom}` : '')
          + (this.dateTo   ? `&date_to=${this.dateTo}`     : '')
          + wf);
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
      // /metric-bins (post-Step-5.5-continuation) supports walk_forward
      // and (Step 6) train_test for both ALL and single-ticker modes via
      // the Assigner. Send the page-wide mode so the side bin charts
      // agree with the rest of the page (top All-Ticker Metric Bins,
      // standalone primary chart).
      let wf = '';
      if (this.pageMode === 'walk_forward') wf = '&walk_forward=true';
      else if (this.pageMode === 'train_test') wf = `&cutoff_date=${encodeURIComponent(this.cutoffDate)}`;
      const base = `/api/oi-analysis/metric-bins?ticker=${encodeURIComponent(this.ticker)}`
        + `&outcome=${encodeURIComponent(this.outcome)}&bins=${this.hmBins1d}`
        + (this.dateFrom ? `&date_from=${this.dateFrom}` : '')
        + (this.dateTo ? `&date_to=${this.dateTo}` : '')
        + wf;
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

    // ── Rolling IC + sign-stability (IC.2) ─────────────────────────────
    // Reads `data.rolling_ic.series`, colors each segment by sign_class
    // (same=green, opposite=red, neutral=grey), draws the reference IC as
    // a dashed horizontal line, and in train_test mode draws a vertical
    // dashed marker at the cutoff date.
    _renderRollingCorr() {
      const el = document.getElementById('chart-rolling');
      const payload = this.data?.rolling_ic;
      if (!el || !payload?.series?.length) {
        if (this._charts['rolling']) { this._charts['rolling'].destroy(); this._charts['rolling'] = null; }
        return;
      }
      if (this._charts['rolling']) this._charts['rolling'].destroy();

      const series  = payload.series;
      const refIc   = Number(payload.reference_ic ?? 0);
      const epsilon = Number(payload.epsilon ?? 0);  // noise-floor half-width
      const cutoff  = payload.cutoff_date;  // ISO date string or null

      // 5-day context series: align to main series x-axis by date so both
      // share the same label array regardless of warmup-length differences.
      // (5d warmup << 252d warmup, so every main-series date is covered.)
      const shortByDate = new Map((payload.short_series || []).map(p => [p.date, p.ic]));
      const shortData   = series.map(p => shortByDate.get(p.date) ?? null);

      // ── Y-axis range: always include 0 and the full ±ε band ─────────────
      const icValues = series.map(p => p.ic).filter(v => v != null);
      const dataMin  = icValues.length ? Math.min(...icValues) : 0;
      const dataMax  = icValues.length ? Math.max(...icValues) : 0;
      const absEps   = Math.abs(epsilon);
      const rawMin   = Math.min(dataMin, -absEps, 0);
      const rawMax   = Math.max(dataMax,  absEps, 0);
      const pad      = Math.max((rawMax - rawMin) * 0.12, 0.005);
      const yMin     = rawMin - pad;
      const yMax     = rawMax + pad;

      const SIGN_COLORS = {
        same:     'rgba(76, 175, 80, 0.95)',   // green
        opposite: 'rgba(229, 57, 53, 0.95)',   // red
        neutral:  'rgba(140, 140, 140, 0.55)', // dim grey
      };

      // Find cutoff index for vertical-line drawing in train_test mode.
      let cutoffIdx = -1;
      if (cutoff) {
        for (let i = 0; i < series.length; i++) {
          if (series[i].date >= cutoff) { cutoffIdx = i; break; }
        }
      }

      // Custom plugin for the cutoff vertical line. Drawn on top of the
      // datasets so it sits above the IC line.
      const cutoffPlugin = {
        id: 'icCutoffLine',
        afterDatasetsDraw(chart) {
          if (cutoffIdx < 0) return;
          const xScale = chart.scales.x, yScale = chart.scales.y;
          const xPx = xScale.getPixelForValue(cutoffIdx);
          const ctx = chart.ctx;
          ctx.save();
          ctx.strokeStyle = 'rgba(255, 193, 7, 0.75)';
          ctx.setLineDash([5, 4]);
          ctx.lineWidth = 1.25;
          ctx.beginPath();
          ctx.moveTo(xPx, yScale.top);
          ctx.lineTo(xPx, yScale.bottom);
          ctx.stroke();
          // Label
          ctx.fillStyle = 'rgba(255, 193, 7, 0.95)';
          ctx.font = '10px sans-serif';
          ctx.textAlign = 'left';
          ctx.fillText(`cutoff ${cutoff}`, xPx + 4, yScale.top + 10);
          ctx.restore();
        },
      };

      // ── Noise-floor band: faint grey rect spanning −ε to +ε ─────────────
      // Drawn before datasets so the IC line stays on top. If epsilon is 0
      // or null (e.g. suppressed metrics) the plugin exits immediately.
      const noiseFloorPlugin = {
        id: 'icNoiseFloor',
        beforeDatasetsDraw(chart) {
          if (absEps <= 0) return;
          const { scales: { x: xsc, y: ysc }, ctx: c } = chart;
          const yTop    = ysc.getPixelForValue(absEps);
          const yBottom = ysc.getPixelForValue(-absEps);
          c.save();
          c.fillStyle = 'rgba(160, 160, 160, 0.13)';
          c.fillRect(xsc.left, yTop, xsc.right - xsc.left, yBottom - yTop);
          c.restore();
        },
      };

      this._charts['rolling'] = new Chart(el, {
        type: 'line',
        data: {
          labels: series.map(p => p.date?.slice(0, 7)),
          datasets: [
            // 5-day IC context overlay — regime texture, NOT a signal line.
            // Drawn first so it sits behind the main 252d line and reference
            // line. Flat neutral color, no segment hook, thinner stroke.
            // Y-axis range is not expanded for 5d extremes — spikes clip.
            {
              label: 'IC (21d context)',
              data:  shortData,
              borderColor:     'rgba(180, 180, 180, 0.28)',
              backgroundColor: 'transparent',
              borderWidth: 0.75, pointRadius: 0, tension: 0.15,
              // No segment: coloring must stay flat to distinguish it from
              // the sign-classified 252d line — color encodes nothing here.
            },
            // IC line with per-segment coloring.
            {
              label: `IC (252d)`,
              data: series.map(p => p.ic),
              borderColor: SIGN_COLORS.neutral, // default for any segment we can't classify
              backgroundColor: 'transparent',
              borderWidth: 1.5, pointRadius: 0, tension: 0.2,
              segment: {
                // Chart.js segment hook: `ctx.p1DataIndex` is the right
                // endpoint of the segment. Color by the destination
                // point's sign_class.
                borderColor: (ctx) => {
                  const cls = series[ctx.p1DataIndex]?.sign_class;
                  return SIGN_COLORS[cls] || SIGN_COLORS.neutral;
                },
              },
            },
            // Reference IC dashed horizontal line.
            {
              label: `Reference IC (${refIc.toFixed(3)})`,
              data: series.map(() => refIc),
              borderColor: 'rgba(255, 193, 7, 0.55)',
              backgroundColor: 'transparent',
              borderWidth: 1, borderDash: [6, 4],
              pointRadius: 0, tension: 0,
            },
          ],
        },
        options: {
          responsive: true, maintainAspectRatio: false, animation: false,
          plugins: {
            legend: { labels: { color:'#aaa', font:{ size:10 } } },
            tooltip: {
              backgroundColor:'rgba(20,20,20,0.95)', borderColor:'#444', borderWidth:1,
              callbacks: {
                afterLabel: (item) => {
                  const p = series[item.dataIndex];
                  if (!p) return '';
                  return `class: ${p.sign_class}`;
                },
              },
            },
          },
          scales: {
            ...this._darkScales(),
            x: { ...this._darkScales().x, ticks:{...this._darkScales().x.ticks, maxTicksLimit:10} },
            y: { ...this._darkScales().y,
                 min: yMin, max: yMax,  // always include 0 and ±ε band
                 ticks:{...this._darkScales().y.ticks, callback: v => v.toFixed(3) } },
          },
        },
        plugins: [noiseFloorPlugin, cutoffPlugin],
      });
    },

    // Human-readable summary string for the rolling-IC pane subtitle.
    // Returns empty string when no data (subtitle hidden by the template).
    rollingIcSubtitle() {
      const p = this.data?.rolling_ic;
      if (!p?.series?.length) return '';
      const ss = p.sign_stability || {};
      // ic_mode: "single_ticker" or "cross_sectional" (IC.3). Disambiguates
      // which computation produced this chart — the ticker selector usually
      // suffices, but having it in the subtitle makes the data lineage explicit.
      const modeTxt = p.ic_mode === 'cross_sectional' ? 'cross-sectional' : 'single-ticker';
      const refTxt = `ref ${(p.reference_ic ?? 0).toFixed(3)}`;
      const epsTxt = `ε ${(p.epsilon ?? 0).toFixed(3)}`;
      if (ss.suppressed) {
        const reason = ss.suppression_reason === 'reference_below_noise'
          ? 'reference below noise floor'
          : (ss.suppression_reason || 'no decisive windows');
        return `[${modeTxt}] Stability: — (${reason}) · ${refTxt} · ${epsTxt}`;
      }
      const stab    = (ss.stability == null) ? '—' : `${(ss.stability * 100).toFixed(1)}%`;
      const neutPct = ss.n_total ? (100 * ss.n_neutral / ss.n_total).toFixed(1) : '0.0';
      return `[${modeTxt}] Stability: ${stab} · ${neutPct}% neutral · ${refTxt} · ${epsTxt}`;
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

      // Build histogram bins — avoid spread on large arrays (V8 call-stack limit)
      const nBins = 40;
      let minRet = Infinity, maxRet = -Infinity;
      for (const v of allRets) { if (v < minRet) minRet = v; if (v > maxRet) maxRet = v; }
      const mn = Math.max(minRet, -15);
      const mx = Math.min(maxRet,  15);
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

      // True calendar x-axis: spot_series = all trading days (single-ticker);
      // ALL mode: union of entry dates (covers most trading days across tickers)
      const spotSeries = this.data.spot_series || [];
      const tradingDays = spotSeries.length > 0
        ? spotSeries.map(s => s.date)
        : [...new Set(cal.map(c => c.date))].sort();

      // Optionally drop entries that overlap a still-open trade of the same ticker.
      const entries = filtered.map(c => ({ ticker: c.ticker, date: c.date }));
      const kept = this.dedupeConc.primary
        ? this._dedupeConcurrent(entries, tradingDays, horizon)
        : entries;
      const entriesByDate = {};
      for (const t of kept) entriesByDate[t.date] = (entriesByDate[t.date] || 0) + 1;

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
      // Map canvas IDs to their _charts key (for non-standard naming)
      const keyOverride = {
        'sec-bar-canvas':       'sec-bar',
        'sec-equity-canvas':    'sec-equity',
        'sec-bins-canvas':      'sec-bins',
        'sec-activity-canvas':  'sec-activity',
        'sec-bubble-canvas':    'sec-bubble',
        'sec-yearly-canvas':    'sec-yearly',
        'corr-equity-canvas':   'corr-equity',
        'corr-yearly-canvas':   'corr-yearly',
        'corr-activity-canvas': 'corr-activity',
        'corr-bubble-canvas':   'corr-bubble',
        'chart-port-equity':    'port-equity',
        'chart-port-yearly':    'port-yearly',
        'chart-port-activity':  'port-activity',
        'chart-port-bubble':    'port-bubble',
        // Signal Survey (IC.5 + IC.7)
        'chart-ic-leaderboard': 'ic-leader',
        'chart-ic-scatter':     'ic-scatter',
        'chart-ic-beeswarm':    'ic-beeswarm',
        'chart-ic-decomp':      'ic-decomp',
        'chart-ic-lorenz':      'ic-lorenz',
      };
      const key = keyOverride[chartId] || chartId.replace('chart-', '');
      this.fsChartId = chartId;
      this.$nextTick(() => {
        if (this._charts['_fs']) { this._charts['_fs'].destroy(); delete this._charts['_fs']; }
        // Destroy the original chart before the ID swap so its canvas is free
        // when closeFullscreen re-renders into it. Without this the orphaned
        // Chart.js instance on the original canvas blocks re-render silently.
        if (this._charts[key]) { this._charts[key].destroy(); delete this._charts[key]; }
        // Re-render the specific chart into the fs canvas by calling its render method
        const fsEl = document.getElementById('fs-canvas');
        if (!fsEl) return;
        // Swap canvas ID temporarily so render methods target the fullscreen canvas
        const origEl = document.getElementById(chartId);
        if (origEl) origEl.id = chartId + '-orig';
        fsEl.id = chartId;
        // Call the appropriate render method
        const renderMap = {
          'chart-decile':        () => this._renderDecileBar(),
          'chart-equity':        () => this._renderEquity(),
          'chart-yearly':        () => this._renderYearly(),
          'chart-rolling':       () => this._renderRollingCorr(),
          'chart-boxplot':       () => this._renderBoxplot(),
          'chart-drawdown':      () => this._renderDrawdown(),
          'chart-dist':          () => this._renderReturnDist(),
          'chart-calendar':      () => this._renderTradeCalendar(),
          'chart-dow':           () => this._renderDOW(),
          'chart-winrate':       () => this._renderWinRate(),
          'chart-activity':      () => this._renderActivity(),
          // Secondary scanner
          'sec-bar-canvas':       () => this._renderSecBar(),
          'sec-equity-canvas':    () => this._renderSecEquity(),
          'sec-bins-canvas':      () => this._renderSecBinsChart(),
          'sec-activity-canvas':  () => this._renderSecActivity(),
          'sec-bubble-canvas':    () => this._renderSecBubble(),
          'sec-yearly-canvas':    () => this._renderSecYearly(),
          // Correlation intersection detail
          'corr-equity-canvas':   () => this._renderCorrEquity(),
          'corr-yearly-canvas':   () => this._renderCorrYearly(),
          'corr-activity-canvas': () => this._renderCorrActivity(),
          'corr-bubble-canvas':   () => this._renderCorrBubble(),
          // System Portfolio
          'chart-port-equity':    () => this._renderPortEquity(),
          'chart-port-yearly':    () => this._renderPortYearly(),
          'chart-port-activity':  () => this._renderPortActivity(),
          'chart-port-bubble':    () => this._renderPortBubble(),
          // Signal Survey
          'chart-ic-leaderboard': () => this._renderIcLeaderboard(),
          'chart-ic-scatter':     () => this._renderIcScatter(),
          'chart-ic-beeswarm':    () => this._renderIcBeeswarm(),
          'chart-ic-decomp':      () => this._renderIcDecomp(),
          'chart-ic-lorenz':      () => this._renderIcLorenz(),
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

    openTradeTableFullscreen() {
      this.fsChartId = 'trade-table';
      this.$nextTick(() => {
        // Clone the rendered table DOM into the overlay mount point
        const src  = document.getElementById('trade-table-body')?.closest('table');
        const dest = document.getElementById('fs-trade-table-mount');
        if (src && dest) {
          dest.innerHTML = '';
          dest.appendChild(src.cloneNode(true));
          // Copy the colgroup + thead from the original table wrapper
          const colgroup = document.getElementById('trade-table-cols');
          const thead    = document.getElementById('trade-table-head');
          const tbl      = dest.querySelector('table');
          if (tbl && colgroup) tbl.prepend(colgroup.cloneNode(true));
          if (tbl && thead)    tbl.querySelector('thead')?.replaceWith(thead.cloneNode(true));
          tbl?.setAttribute('style', 'border-collapse:collapse;font-family:monospace;width:100%;table-layout:fixed');
        }
      });
    },

    closeFullscreen() {
      const wasChartId = this.fsChartId;
      if (this._charts['_fs']) { this._charts['_fs'].destroy(); delete this._charts['_fs']; }
      this.fsChartId = null;
      this.$nextTick(() => {
        // Re-render only the chart that was in fullscreen — re-rendering all
        // charts takes 3-5 s and is unnecessary.
        const renderMap = {
          'chart-decile':        () => this._renderDecileBar(),
          'chart-equity':        () => this._renderEquity(),
          'chart-yearly':        () => this._renderYearly(),
          'chart-rolling':       () => this._renderRollingCorr(),
          'chart-boxplot':       () => this._renderBoxplot(),
          'chart-drawdown':      () => this._renderDrawdown(),
          'chart-dist':          () => this._renderReturnDist(),
          'chart-calendar':      () => this._renderTradeCalendar(),
          'chart-dow':           () => this._renderDOW(),
          'chart-winrate':       () => this._renderWinRate(),
          'chart-activity':      () => this._renderActivity(),
          'sec-bar-canvas':      () => this._renderSecBar(),
          'sec-equity-canvas':   () => this._renderSecEquity(),
          'sec-bins-canvas':     () => this._renderSecBinsChart(),
          'sec-activity-canvas': () => this._renderSecActivity(),
          'sec-bubble-canvas':   () => this._renderSecBubble(),
          'sec-yearly-canvas':   () => this._renderSecYearly(),
          'corr-equity-canvas':  () => this._renderCorrEquity(),
          'corr-yearly-canvas':  () => this._renderCorrYearly(),
          'corr-activity-canvas':() => this._renderCorrActivity(),
          'corr-bubble-canvas':  () => this._renderCorrBubble(),
          'chart-port-equity':   () => this._renderPortEquity(),
          'chart-port-yearly':   () => this._renderPortYearly(),
          'chart-port-activity': () => this._renderPortActivity(),
          'chart-port-bubble':   () => this._renderPortBubble(),
          // Signal Survey
          'chart-ic-leaderboard': () => this._renderIcLeaderboard(),
          'chart-ic-scatter':     () => this._renderIcScatter(),
          'chart-ic-beeswarm':    () => this._renderIcBeeswarm(),
          'chart-ic-decomp':      () => this._renderIcDecomp(),
          'chart-ic-lorenz':      () => this._renderIcLorenz(),
        };
        const fn = renderMap[wasChartId];
        if (fn) fn();
      });
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
        const smMode = this.pageMode === 'walk_forward' ? 'walk_forward' : this.pageMode === 'train_test' ? 'train_test' : 'in_sample';
        // For train_test, the score matrix is partitioned by cutoff_date.
        // Send the active cutoff so the right slice loads.
        const cutoffQ = smMode === 'train_test'
          ? `&cutoff_date=${encodeURIComponent(this.cutoffDate)}` : '';
        const [metaRes, statusRes] = await Promise.all([
          fetch('/api/oi-analysis/score-matrix/meta?mode=' + smMode + cutoffQ),
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
      const smMode = this.pageMode === 'walk_forward' ? 'walk_forward' : this.pageMode === 'train_test' ? 'train_test' : 'in_sample';
      const params = new URLSearchParams({
        sort_by: this.smSortKey === 'd10_d1_spread' ? 'composite_score' : this.smSortKey,
        order: this.smSortDir,
        min_score: this.smMinScore,
        mode: smMode,
      });
      // For train_test, scope to the active cutoff_date.
      if (smMode === 'train_test') params.set('cutoff_date', this.cutoffDate);
      if (this.smFilterTicker) params.set('ticker', this.smFilterTicker);
      if (this.smFilterMetric) params.set('metric', this.smFilterMetric);
      if (this.smFilterFwd) params.set('fwd_ret', this.smFilterFwd);

      try {
        const r = await fetch('/api/oi-analysis/score-matrix?' + params);
        if (r.ok) this.smRows = await r.json();
        // Refresh meta too (mirror the cutoff_date filter).
        const metaParams = new URLSearchParams({ mode: smMode });
        if (smMode === 'train_test') metaParams.set('cutoff_date', this.cutoffDate);
        const m = await fetch('/api/oi-analysis/score-matrix/meta?' + metaParams);
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
        const r = await fetch('/api/oi-analysis/run-batch-score', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            walk_forward: this.pageMode === 'walk_forward',
            cutoff_date:  this.pageMode === 'train_test' ? this.cutoffDate : null,
          }),
        });
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
      const smMode = this.pageMode === 'walk_forward' ? 'walk_forward'
                   : this.pageMode === 'train_test'   ? 'train_test'
                   : 'in_sample';
      const params = new URLSearchParams({ mode: smMode });
      // For train_test, scope to the active cutoff_date.
      if (smMode === 'train_test') params.set('cutoff_date', this.cutoffDate);
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
      this.corrMiniData = null;
      this.corrSelections = {};
      this.corrResult = null;
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
            sec_bin_count:  this.secBinCount,
          }),
        });
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        const d = await r.json();
        if (d.error) throw new Error(d.error);
        this.secCacheKey = d.cache_key;
        this.secBaseline = d.baseline;
        this.secMetrics  = d.metrics || [];
        this.secMaxAbsScore = Math.max(0.0001, ...this.secMetrics.map(m => Math.abs(m.score)));
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
            cache_key:             this.secCacheKey,
            filtered_dates:        this._secFilteredDates(),
            ticker:                this.ticker,
            sec_bin_count:         this.secBinCount,
            selected_primary_bins: [...this.selectedBins20].sort((a, b) => a - b),
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
        this.secMaxAbsScore = Math.max(0.0001, ...this.secMetrics.map(m => Math.abs(m.score)));
        this.secScanMeta = { mode: d.mode, dropped_warmup_n: d.dropped_warmup_n, universe_n: d.universe_n, start_date: d.start_date };
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
            cache_key:             this.secCacheKey,
            metric_b:              metricName,
            filtered_dates:        this._secFilteredDates(),
            sec_bins:              this.secSelectedSecBins,
            sec_bin_count:         this.secBinCount,
            ticker:                this.ticker,
            selected_primary_bins: [...this.selectedBins20].sort((a, b) => a - b),
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
      await this.secScan();           // re-rank leaderboard with new bin count
      if (this.secSelectedMetric) await this.secDrillMetric(this.secSelectedMetric, false);
    },

    _renderSecBar() {
      const inner = document.getElementById('sec-bar-inner');
      const canvas = document.getElementById('sec-bar-canvas');
      if (!canvas || !inner || !this.secMetrics.length) return;
      if (this._charts['sec-bar']) { this._charts['sec-bar'].destroy(); delete this._charts['sec-bar']; }

      const metrics = this.secMetrics;
      // score is in raw return units (same as spread); * 100 converts to %.
      const scores  = metrics.map(m => +(m.score * 100).toFixed(5));

      // Color encodes spread direction (same IC leaderboard convention).
      // Opacity encodes sample sufficiency via w: 0.25 (thin) → 0.90 (full).
      // w is absolute-reference-normalised so a pale bar means thin bins
      // regardless of how many primary bins were selected.
      const bgColors  = metrics.map(m => {
        const op = (0.25 + (m.w ?? 1) * 0.65).toFixed(2);
        return (m.spread ?? 0) >= 0 ? `rgba(52,152,219,${op})` : `rgba(232,67,147,${op})`;
      });
      const borders   = metrics.map(m =>
        (m.spread ?? 0) >= 0 ? 'rgba(52,152,219,0.6)' : 'rgba(232,67,147,0.6)');

      // Dynamic width: at least container width, at most barW*n
      const barW   = Math.max(10, Math.min(22, 1400 / metrics.length));
      const chartW = Math.max(inner.parentElement.clientWidth || 600, metrics.length * (barW + 3) + 60);
      inner.style.width = chartW + 'px';

      const secBinCount = this.secBinCount;
      const ctx = canvas.getContext('2d');
      this._charts['sec-bar'] = new Chart(ctx, {
        type: 'bar',
        data: {
          labels: metrics.map(m => m.name),
          datasets: [{
            data:            scores,
            backgroundColor: bgColors,
            borderColor:     borders,
            borderWidth:     1,
            barThickness:    barW,
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
                  const wPct = m.w != null ? (m.w * 100).toFixed(0) + '%' : '—';
                  return [
                    `Score: ${(m.score * 100).toFixed(4)}%`,
                    `Spread: ${(m.spread * 100).toFixed(3)}%  (n_top=${m.n_top}, n_bot=${m.n_bottom})`,
                    `Breadth: ${(m.breadth * 100).toFixed(1)}%  (${m.n_qualifying_tickers} tickers)`,
                    `Sample weight: ${wPct}  (qualifying bins: ${m.n_qualifying_bins} of ${secBinCount})`,
                    `WR spread: ${(m.win_lift * 100).toFixed(1)}%`,
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
              ticks: { color: '#888', font: { size: 9 }, callback: v => v.toFixed(3) + '%' },
              grid: { color: '#2a2a2a' },
              title: { display: true, text: 'Score = weighted spread × breadth (%)', color: '#555', font: { size: 9 } },
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

      // Trade-weighted avg ret across the visible tickers (n-weighted) → %
      const totalN = tickers.reduce((s, t) => s + (t.n || 0), 0);
      const avgPct = totalN > 0
        ? tickers.reduce((s, t) => s + (t.avg_ret || 0) * (t.n || 0), 0) / totalN * 100
        : 0;

      this._charts['sec-bubble'] = new Chart(canvas.getContext('2d'), {
        type: 'bubble',
        data: { datasets },
        plugins: [this._avgRetLinePlugin(avgPct, 'avg')],
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

    secDownloadCSV() {
      const trades = this.secDetail?.combined_trades;
      if (!trades?.length) return;
      const sec_metric  = this.secSelectedMetric || 'secondary';
      const prim_metric = this.metric             || 'primary';
      const header = [
        'ticker', 'trade_date', `${prim_metric}_val`, `${sec_metric}_val`,
        'spot_entry', 'exit_date', 'spot_exit', 'ret_pct',
      ].join(',');
      const fmt = (v, d = 6) => v == null ? '' : Number(v).toFixed(d);
      const rows = trades.map(t => [
        t.ticker || '',
        t.trade_date || '',
        fmt(t.primary_val),
        fmt(t.secondary_val),
        fmt(t.spot_entry, 2),
        t.exit_date || '',
        fmt(t.spot_exit, 2),
        t.ret != null ? (t.ret * 100).toFixed(6) : '',
      ].join(','));
      this._downloadCsv([header, ...rows].join('\n'),
        `sec_${this.metric}_${sec_metric}_${new Date().toISOString().slice(0,10)}.csv`);
    },

    _downloadCsv(csvText, filename) {
      const blob = new Blob([csvText], { type: 'text/csv' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url; a.download = filename;
      document.body.appendChild(a); a.click(); document.body.removeChild(a);
      URL.revokeObjectURL(url);
    },

    // Filter trade entries so a new entry for ticker T is skipped while a
    // prior entry for T is still inside its horizon (i.e. an "open trade"
    // of that ticker). Entries can be `{ticker, date}` or `{ticker, trade_date}`.
    _dedupeConcurrent(entries, tradingDays, horizon) {
      if (!entries?.length || !tradingDays?.length || !(horizon > 0)) return entries || [];
      const idxBy = new Map();
      tradingDays.forEach((d, i) => idxBy.set(d, i));
      const sorted = [...entries].sort((a, b) => {
        const ad = a.date || a.trade_date || '';
        const bd = b.date || b.trade_date || '';
        return ad < bd ? -1 : (ad > bd ? 1 : 0);
      });
      const lastByTkr = new Map();
      const keep = [];
      for (const e of sorted) {
        const t  = e.ticker || '?';
        const dk = e.date || e.trade_date || '';
        const i  = idxBy.get(dk);
        if (i == null) { keep.push(e); continue; }
        const last = lastByTkr.get(t);
        if (last == null || (i - last) >= horizon) {
          keep.push(e);
          lastByTkr.set(t, i);
        }
      }
      return keep;
    },

    toggleDedupeConc(key) {
      this.dedupeConc[key] = !this.dedupeConc[key];
      this.dedupeConc = { ...this.dedupeConc };  // nudge Alpine reactivity
      if (key === 'primary') this._renderActivity();
      else if (key === 'sec')  this._renderSecActivity();
      else if (key === 'corr') this._renderCorrActivity();
      else if (key === 'port') this._renderPortActivity();
    },

    // Threshold Drift (walk-forward bin boundaries over time)
    get tdHasSeries() {
      const s = this.tdData?.series_ratio || this.tdData?.series_native || {};
      return Object.values(s).some(arr => Array.isArray(arr) && arr.length > 0);
    },

    tdToggleBin(b) {
      const i = this.tdBinsToShow.indexOf(b);
      if (i >= 0) this.tdBinsToShow.splice(i, 1);
      else        this.tdBinsToShow.push(b);
      this.tdBinsToShow = [...this.tdBinsToShow].sort((a, b) => a - b);
      if (this.tdData) this.loadTd(true);
    },

    async toggleTd() {
      this.tdExpanded = !this.tdExpanded;
      if (this.tdExpanded && this.tdMetric && !this.tdData) {
        await this.loadTd();
      }
      // Re-render even if data was already loaded — the canvas may have
      // been hidden by the x-if.
      this.$nextTick(() => this._renderTdChart());
    },

    async loadTd(forceRefresh = false) {
      if (!this.tdMetric || !this.tdBinsToShow.length) return;
      this.tdLoading = true;
      try {
        if (forceRefresh) {
          try {
            await fetch('/api/oi-analysis/threshold-drift/invalidate', { method: 'POST' });
          } catch (_) {}
        }
        const params = new URLSearchParams({
          metric:  this.tdMetric,
          outcome: this.tdOutcome || 'ret_5d_fwd_oc',
          ticker:  'ALL',
          n_bins:  '20',
          bins:    this.tdBinsToShow.join(','),
        });
        const r = await fetch('/api/oi-analysis/threshold-drift?' + params);
        if (!r.ok) {
          const txt = await r.text().catch(() => '');
          this.tdData = { error: `HTTP ${r.status}${txt ? ': ' + txt.slice(0, 200) : ''}`,
                          series: {}, in_sample_ref: {} };
          return;
        }
        this.tdData = await r.json();
        await this.$nextTick();
        this._renderTdChart();
      } catch (e) {
        console.error('loadTd', e);
        this.tdData = { error: e.message, series: {}, in_sample_ref: {} };
      } finally {
        this.tdLoading = false;
      }
    },

    _renderTdChart() {
      const canvas = document.getElementById('chart-threshold-drift');
      if (!canvas || !this.tdData) return;
      if (this._charts['td']) { this._charts['td'].destroy(); delete this._charts['td']; }

      // ── Select series + reference per the active mode ────────────────
      // Mode 'ratio' (default): aggregate per-ticker drift ratios (walk-forward
      // threshold / that ticker's full-history threshold). Dimensionless; the
      // reference line is at 1.0 ("matches today's threshold").
      // Mode 'native_single': raw threshold values for ONE picked ticker.
      // Y-axis label and reference values differ per mode.
      const mode = this.tdMode || 'ratio';
      let series, refValues, yLabel;
      if (mode === 'native_single') {
        // Materialise this single ticker's per-bin time series in the same
        // shape as the aggregated series so the chart code below is shared.
        const t = this.tdSingleTicker || (this.tdData.tickers_eligible || [])[0];
        if (t && !this.tdSingleTicker) this.tdSingleTicker = t;
        const tkrData = this.tdData.per_ticker?.[t] || {};
        const tkrFull = this.tdData.per_ticker_full?.[t] || {};
        series = {};
        refValues = {};
        for (const b of (this.tdData.bins || [])) {
          const arr = tkrData[String(b)] || [];
          series[String(b)] = arr.map(p => ({ date: p.date, median: p.threshold }));
          refValues[String(b)] = tkrFull[String(b)] ?? null;
        }
        yLabel = `Threshold value (${t || '?'})`;
      } else {
        series    = this.tdData.series_ratio || {};
        refValues = {};
        for (const b of (this.tdData.bins || [])) refValues[String(b)] = 1.0;
        yLabel = "× today's full-history threshold (1.0 = stable)";
      }

      const dateSet = new Set();
      for (const arr of Object.values(series)) for (const p of arr) dateSet.add(p.date);
      const dates = [...dateSet].sort();
      if (!dates.length) return;

      // Color palette per bin (consistent regardless of selection order).
      const binColor = (b) => {
        const map = {
          1:  ['#e84393', 'rgba(232,67,147,0.18)'],
          5:  ['#f39c12', 'rgba(243,156,18,0.18)'],
          10: ['#95a5a6', 'rgba(149,165,166,0.18)'],
          15: ['#1abc9c', 'rgba(26,188,156,0.18)'],
          20: ['#3498db', 'rgba(52,152,219,0.18)'],
        };
        return map[b] || ['#aaa', 'rgba(170,170,170,0.18)'];
      };

      // Build datasets: per bin → median line + IQR band (filled between q25/q75).
      // In native_single mode there are no q25/q75 — just the single value.
      const datasets = [];
      const bins = (this.tdData.bins || []);
      const showBand = mode === 'ratio';
      for (const b of bins) {
        const arr = series[String(b)] || [];
        if (!arr.length) continue;
        const byDate = Object.fromEntries(arr.map(p => [p.date, p]));
        const [stroke, fill] = binColor(b);
        const mid = dates.map(d => (d in byDate) ? +(byDate[d].median).toFixed(6) : null);
        if (showBand) {
          const lo = dates.map(d => (d in byDate && byDate[d].q25 != null) ? +(byDate[d].q25).toFixed(6) : null);
          const hi = dates.map(d => (d in byDate && byDate[d].q75 != null) ? +(byDate[d].q75).toFixed(6) : null);
          datasets.push({
            label: `B${b} q75`,
            data:  hi,
            borderColor: 'transparent', backgroundColor: 'transparent',
            pointRadius: 0, fill: false, tension: 0, spanGaps: true,
          });
          datasets.push({
            label: `B${b} band`,
            data:  lo,
            borderColor: 'transparent', backgroundColor: fill,
            pointRadius: 0, fill: '-1', tension: 0, spanGaps: true,
          });
        }
        datasets.push({
          label: showBand ? `B${b} (median)` : `B${b}`,
          data:  mid,
          borderColor: stroke,
          backgroundColor: stroke,
          pointRadius: 0,
          borderWidth: 1.5,
          tension: 0,
          spanGaps: true,
          fill: false,
        });
      }

      this._charts['td'] = new Chart(canvas.getContext('2d'), {
        type: 'line',
        data: { labels: dates, datasets },
        plugins: [{
          id: 'inSampleRefs',
          afterDraw: (chart) => {
            const yScale = chart.scales.y;
            const xScale = chart.scales.x;
            if (!yScale || !xScale) return;
            const ctx = chart.ctx;
            for (const b of bins) {
              const ref = refValues[String(b)];
              if (ref == null) continue;
              const [stroke] = binColor(b);
              const y = yScale.getPixelForValue(ref);
              ctx.save();
              ctx.strokeStyle = stroke;
              ctx.globalAlpha = 0.6;
              ctx.lineWidth = 1;
              ctx.setLineDash([4, 4]);
              ctx.beginPath();
              ctx.moveTo(xScale.left,  y);
              ctx.lineTo(xScale.right, y);
              ctx.stroke();
              ctx.setLineDash([]);
              ctx.fillStyle = stroke;
              ctx.font = '9px sans-serif';
              ctx.textAlign = 'right';
              ctx.textBaseline = 'bottom';
              const fmt = (mode === 'ratio') ? ref.toFixed(2) + 'x' : ref.toFixed(4);
              ctx.fillText(`B${b} now=${fmt}`, xScale.right - 4, y - 1);
              ctx.restore();
            }
          },
        }],
        options: {
          responsive: true, maintainAspectRatio: false, animation: false,
          plugins: {
            legend: {
              labels: {
                color: '#aaa', font: { size: 10 },
                filter: (item) => !item.text.includes('q75') && !item.text.includes('band'),
              },
            },
            tooltip: {
              mode: 'index', intersect: false,
              filter: (item) => !item.dataset.label.includes('q75') && !item.dataset.label.includes('band'),
              callbacks: {
                title: ctx => dates[ctx[0]?.dataIndex] || '',
                label: ctx => {
                  if (ctx.raw == null) return '';
                  return `${ctx.dataset.label}: ${ctx.raw.toFixed(4)}`;
                },
              },
            },
          },
          scales: {
            ...this._darkScales(),
            x: { ...this._darkScales().x,
                 ticks: { ...this._darkScales().x.ticks, maxTicksLimit: 12 } },
            y: { ...this._darkScales().y,
                 title: { display: true, text: yLabel,
                          color: '#888', font: { size: 9 } } },
          },
        },
      });
    },

    // All-Ticker Metric Bins (top-of-page collapsable browser)
    async toggleTopBins() {
      this.topBinsExpanded = !this.topBinsExpanded;
      if (this.topBinsExpanded && !this.topBinsData) {
        await this.loadTopBins();
      }
    },

    async loadTopBins(forceRefresh = false) {
      this.topBinsLoading = true;
      // Pin the dropdown to ret_5d_fwd_oc on first load — Alpine's x-model
      // can fall out of sync when outcomes arrive after init.
      if (this.outcomes?.length && !this.outcomes.includes(this.topBinsOutcome)) {
        this.topBinsOutcome = this.outcomes.includes('ret_5d_fwd_oc')
          ? 'ret_5d_fwd_oc' : this.outcomes[0];
      }
      try {
        if (forceRefresh) {
          try {
            await fetch('/api/oi-analysis/global-metric-bins/invalidate', { method: 'POST' });
          } catch (_) {}
        }
        const params = new URLSearchParams({
          outcome:      this.topBinsOutcome || 'ret_5d_fwd_oc',
          ticker:       'ALL',
          n_bins:       '20',
          walk_forward: this.pageMode === 'walk_forward' ? '1' : '0',
        });
        if (this.pageMode === 'train_test') params.set('cutoff_date', this.cutoffDate);
        const r = await fetch('/api/oi-analysis/global-metric-bins?' + params);
        if (!r.ok) {
          const txt = await r.text().catch(() => '');
          this.topBinsData = { metrics: [], total_rows: 0,
                               error: `HTTP ${r.status}${txt ? ': ' + txt.slice(0, 200) : ''}` };
          return;
        }
        const d = await r.json();
        // Compute _zeroTopPct + _total per metric for the diverging-bar layout
        // (same pattern the corr explorer uses for its mini charts).
        for (const m of (d.metrics || [])) {
          const maxPos = Math.max(0, ...m.bins);
          const maxNeg = Math.abs(Math.min(0, ...m.bins));
          m._total      = Math.max(0.0001, (maxPos + maxNeg) * 1.06);
          m._zeroTopPct = (maxPos / m._total * 100).toFixed(2);
        }
        this.topBinsData = d;
      } catch (e) {
        console.error('loadTopBins', e);
        this.topBinsData = { metrics: [], total_rows: 0, error: e.message };
      } finally {
        this.topBinsLoading = false;
      }
    },

    // Faint horizontal dotted gray line on bubble charts marking the
    // section's weighted-average return across visible tickers.
    _avgRetLinePlugin(avgPct, label) {
      return {
        id: 'avgRetLine',
        afterDraw(chart) {
          if (!Number.isFinite(avgPct)) return;
          const yScale = chart.scales.y;
          const xScale = chart.scales.x;
          if (!yScale || !xScale) return;
          const y = yScale.getPixelForValue(avgPct);
          const ctx = chart.ctx;
          ctx.save();
          ctx.strokeStyle = 'rgba(170,170,170,0.55)';
          ctx.lineWidth = 1;
          ctx.setLineDash([3, 3]);
          ctx.beginPath();
          ctx.moveTo(xScale.left,  y);
          ctx.lineTo(xScale.right, y);
          ctx.stroke();
          ctx.setLineDash([]);
          ctx.fillStyle = 'rgba(170,170,170,0.75)';
          ctx.font = '10px sans-serif';
          ctx.textAlign = 'right';
          ctx.textBaseline = 'bottom';
          ctx.fillText(
            (label || 'avg') + ' ' + avgPct.toFixed(3) + '%',
            xScale.right - 4, y - 2);
          ctx.restore();
        },
      };
    },

    // ── Multi-Metric Correlation Explorer ────────────────────────────────────

    async corrTogglePanel() {
      this.corrPanelOpen = !this.corrPanelOpen;
      if (this.corrPanelOpen && !this.corrMiniData && this.secCacheKey) {
        await this.corrLoadMiniData();
      }
    },

    async corrSetBinCount(n) {
      this.corrBinCount = n;
      this.corrMiniData = null;
      this.corrSelections = {};
      this.corrResult = null;
      if (this.corrPanelOpen && this.secCacheKey) await this.corrLoadMiniData();
    },

    // Page-wide mode toggle. Cascades through every binning surface on
    // the page: All-Ticker Metric Bins, /analyze, corr explorer (mini +
    // result), System Portfolio aggregate, Score Matrix, and the heatmap
    // (grid + side bars). Called by the segmented mode toggle AND by the
    // cutoff-date input's @change — in train_test mode the cutoffDate may
    // have shifted even though the mode label didn't, so the early
    // return only applies when the new mode is in_sample/walk_forward.
    async setPageMode(m) {
      if (m === this.pageMode && m !== 'train_test') return;
      this.pageMode = m;
      // Top bins: always clear so the next open (or current expanded view)
      // re-fetches in the new mode.
      this.topBinsData = null;
      if (this.topBinsExpanded) {
        this.loadTopBins();  // fire-and-forget; don't block primary reloads
      }
      this.corrMiniData = null;
      const hadCorrResult = !!(this.corrResult && !this.corrResult.error);
      this.corrResult = null;
      if (this.data && !this.error) {
        await this.loadAnalysis();
      }
      if (this.corrPanelOpen && this.secCacheKey) {
        await this.corrLoadMiniData();
        if (hadCorrResult && this.corrSelectedCount() >= 2) {
          await this.runCorrelation();
        }
      }
      if (this.portfolioId && this.portAggregate) {
        this.loadPortfolioAggregate();  // fire-and-forget
      }
      if (this.smMeta.count > 0) {
        this.smInit();  // reload score matrix in new mode
      }
      // Heatmap (Step 5.5 continuation): re-fetch when mode flips. If a
      // heatmap is currently rendered (heatmapData != null) it needs to
      // re-bin under the new spec — both the 2D grid (via /heatmap) and
      // the side bin charts (via /metric-bins).
      if (this.heatmapData && this.heatmapMetric) {
        this.loadHeatmap();  // fire-and-forget; chains into loadHmBins1d()
      }
      // IC.5: mode change shifts reference IC (train_test cutoff vs full
      // history). Reload if the section is open. Clear stale data so the
      // user sees status panels rather than wrong-mode bars.
      this.icBatchData = null; this.loadIcBatch();
      // IC.7: mode change shifts reference IC — always reload in ALL mode.
      if (this.ticker === 'ALL') { this.icDecompData = null; this.loadIcDecomp(); }
    },

    async corrLoadMiniData() {
      this.corrMiniLoading = true;
      try {
        const r = await fetch('/api/oi-analysis/secondary-corr-bins', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({
            cache_key:      this.secCacheKey,
            filtered_dates: this._secFilteredDates(),
            ticker:         this.ticker,
            n_bins:         this.corrBinCount,
            walk_forward:   this.pageMode === 'walk_forward',
            cutoff_date:    this.pageMode === 'train_test' ? this.cutoffDate : null,
            selected_primary_bins: [...this.selectedBins20].sort((a, b) => a - b),
          }),
        });
        const d = await r.json();
        if (!d.error) {
          d.metrics.forEach(m => {
            const maxPos = Math.max(0, ...m.bins);
            const maxNeg = Math.abs(Math.min(0, ...m.bins));
            // 6% headroom so tallest bar doesn't touch edge
            m._total    = Math.max(0.0001, (maxPos + maxNeg) * 1.06);
            m._maxPos   = maxPos;
            m._zeroTopPct = (maxPos / m._total * 100).toFixed(2);  // zero-line % from top
          });
        }
        this.corrMiniData = d;
      } catch (e) {
        this.corrMiniData = { error: e.message };
      } finally {
        this.corrMiniLoading = false;
      }
    },

    corrToggleBin(metric, bin) {
      const sel = { ...this.corrSelections };
      const cur = sel[metric] ? [...sel[metric]] : [];
      const idx = cur.indexOf(bin);
      if (idx >= 0) {
        cur.splice(idx, 1);
        if (cur.length === 0) delete sel[metric]; else sel[metric] = cur;
      } else {
        sel[metric] = [...cur, bin];
      }
      this.corrSelections = sel;
    },

    corrIsBinSelected(metric, bin) {
      return (this.corrSelections[metric] || []).includes(bin);
    },

    corrSelectedCount() {
      return Object.keys(this.corrSelections).length;
    },

    corrClearAll() {
      this.corrSelections = {};
      this.corrResult = null;
    },

    async runCorrelation() {
      if (this.corrSelectedCount() < 2 || !this.secCacheKey) return;
      this.corrLoading = true;
      this.corrResult = null;
      try {
        const selections = Object.entries(this.corrSelections)
          .map(([metric, bins]) => ({ metric, bins }));
        const r = await fetch('/api/oi-analysis/secondary-correlation', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({
            cache_key:      this.secCacheKey,
            filtered_dates: this._secFilteredDates(),
            ticker:         this.ticker,
            n_bins:         this.corrBinCount,
            selections,
            walk_forward:   this.pageMode === 'walk_forward',
            cutoff_date:    this.pageMode === 'train_test' ? this.cutoffDate : null,
            selected_primary_bins: [...this.selectedBins20].sort((a, b) => a - b),
          }),
        });
        const d = await r.json();
        this.corrResult = d;
        if (!d.error) {
          await this.$nextTick();
          await this.$nextTick();
          setTimeout(() => this._renderCorrDetail(), 60);
        }
      } catch (_) {}
      finally { this.corrLoading = false; }
    },

    corrCellStyle(phi, isDiag) {
      const base = 'text-align:center;padding:7px 14px;border:1px solid rgba(255,255,255,0.05);min-width:58px;border-radius:2px;font-size:11px';
      if (isDiag) return `background:#1c1c1c;color:#555;${base}`;
      const a = (0.12 + Math.min(1, Math.abs(phi)) * 0.72).toFixed(2);
      const fg = Math.abs(phi) > 0.4 ? '#fff' : '#999';
      const bg = phi >= 0 ? `rgba(52,152,219,${a})` : `rgba(232,67,147,${a})`;
      return `background:${bg};color:${fg};${base}`;
    },

    corrCellTitle(i, j) {
      if (!this.corrResult) return '';
      if (i === j) return `${this.corrResult.metrics[i]}  n=${this.corrResult.n_each[i]}`;
      const m1 = this.corrResult.metrics[i], m2 = this.corrResult.metrics[j];
      const phi = this.corrResult.phi[i][j];
      const ov  = this.corrResult.overlap[i][j];
      return `${m1} × ${m2}\nφ = ${phi.toFixed(3)}\nOverlap: ${ov} trades`;
    },

    corrSetBubbleMinN(n) {
      this.corrBubbleMinN = +n;
      this._renderCorrBubble();
    },

    _tickerCoverage(tickers) {
      const total = new Set((this.data?.trade_calendar || []).map(c => c.ticker)).size;
      const n = (tickers || []).length;
      return `${n} / ${total} tkrs (${total > 0 ? Math.round(n / total * 100) : 0}%)`;
    },

    corrDownloadCSV() {
      const trades = this.corrResult?.combined_trades;
      if (!trades?.length) return;
      const prim = this.metric || 'primary';
      // Selected secondary metrics, in the same order the response uses
      const secMetrics = this.corrResult?.metrics || [];
      const header = [
        'ticker', 'trade_date', `${prim}_val`,
        ...secMetrics.map(m => `${m}_val`),
        'spot_entry', 'exit_date', 'spot_exit', 'ret_pct',
      ].join(',');
      const fmt = (v, d = 6) => v == null ? '' : Number(v).toFixed(d);
      const rows = trades.map(t => {
        const extras = (t.extra || {});
        return [
          t.ticker || '',
          t.trade_date || '',
          fmt(t.primary_val),
          ...secMetrics.map(m => fmt(extras[m])),
          fmt(t.spot_entry, 2),
          t.exit_date || '',
          fmt(t.spot_exit, 2),
          t.ret != null ? (t.ret * 100).toFixed(6) : '',
        ].join(',');
      });
      this._downloadCsv([header, ...rows].join('\n'),
        `corr_${prim}_${new Date().toISOString().slice(0,10)}.csv`);
    },

    corrStats() {
      const yearly = this.corrResult?.yearly;
      if (!yearly?.length) return null;
      const totalN  = yearly.reduce((s, y) => s + (y.combined_n  || 0), 0);
      if (!totalN) return null;
      const avgRet  = yearly.reduce((s, y) => s + (y.combined_avg || 0) * (y.combined_n || 0), 0) / totalN;
      const winRate = yearly.reduce((s, y) => s + (y.combined_wr  || 0) * (y.combined_n || 0), 0) / totalN;
      const winners = Math.round(winRate * totalN);
      const best  = yearly.reduce((b, y) => y.combined_avg > b.avg ? { yr: y.year, avg: y.combined_avg } : b, { yr: null, avg: -Infinity });
      const worst = yearly.reduce((b, y) => y.combined_avg < b.avg ? { yr: y.year, avg: y.combined_avg } : b, { yr: null, avg:  Infinity });
      const eq  = this.corrResult.equity_combined || [];
      const cum = eq.length ? +(eq[eq.length - 1].value * 100).toFixed(2) : null;
      // Trade utilization: normalized overlap metric
      // 0% = perfect correlation (all bins fire same trades), 100% = fully exclusive (zero overlap)
      // Formula: (union - min_n) / (sum_n - min_n)
      const nEach = this.corrResult.n_each || [];
      const sumN  = nEach.reduce((s, n) => s + n, 0);
      const minN  = nEach.length ? Math.min(...nEach) : 0;
      const union = this.corrResult.combined_n || 0;
      const util  = sumN > minN ? +((union - minN) / (sumN - minN) * 100).toFixed(1) : 100;
      return {
        totalN, avgRet, winRate, winners, losers: totalN - winners, best, worst, cum, util,
        winnerAvg: this.corrResult.winner_avg_ret ?? null,
        loserAvg:  this.corrResult.loser_avg_ret  ?? null,
      };
    },

    _renderCorrDetail() {
      if (!this.corrResult || this.corrResult.error) return;
      this._renderCorrEquity();
      this._renderCorrYearly();
      this._renderCorrActivity();
      this._renderCorrBubble();
    },

    _renderCorrEquity() {
      const canvas = document.getElementById('corr-equity-canvas');
      if (!canvas || !this.corrResult) return;
      if (this._charts['corr-equity']) { this._charts['corr-equity'].destroy(); delete this._charts['corr-equity']; }
      const eqP = this.corrResult.equity_primary  || [];
      const eqC = this.corrResult.equity_combined || [];
      if (!eqP.length) return;
      const cMap = Object.fromEntries(eqC.map(p => [p.date, +(p.value * 100).toFixed(4)]));
      let lastC = 0;
      const combinedAligned = eqP.map(p => {
        if (cMap[p.date] !== undefined) lastC = cMap[p.date];
        return lastC;
      });
      this._charts['corr-equity'] = new Chart(canvas.getContext('2d'), {
        type: 'line',
        data: {
          labels: eqP.map(p => p.date.slice(0, 7)),
          datasets: [
            { label: 'Primary',  data: eqP.map(p => +(p.value * 100).toFixed(4)),
              borderColor: '#3498db', backgroundColor: 'rgba(52,152,219,0.06)',
              borderWidth: 1.5, pointRadius: 0, tension: 0.2, fill: true },
            { label: 'Union', data: combinedAligned,
              borderColor: '#e84393', backgroundColor: 'rgba(232,67,147,0.06)',
              borderWidth: 1.5, pointRadius: 0, tension: 0.2, fill: true, spanGaps: true },
          ],
        },
        options: {
          responsive: true, maintainAspectRatio: false, animation: false,
          plugins: {
            legend: { labels: { color: '#aaa', font: { size: 10 } } },
            tooltip: { mode: 'index', intersect: false,
              callbacks: { title: ctx => eqP[ctx[0]?.dataIndex]?.date || '' } },
          },
          scales: {
            ...this._darkScales(),
            x: { ...this._darkScales().x, ticks: { ...this._darkScales().x.ticks, maxTicksLimit: 10 } },
            y: { ...this._darkScales().y, title: { display: true, text: 'Cum Return %', color: '#888', font: { size: 9 } } },
          },
        },
      });
    },

    _renderCorrYearly() {
      const canvas = document.getElementById('corr-yearly-canvas');
      if (!canvas || !this.corrResult?.yearly?.length) return;
      if (this._charts['corr-yearly']) { this._charts['corr-yearly'].destroy(); delete this._charts['corr-yearly']; }
      const yearly = this.corrResult.yearly;
      this._charts['corr-yearly'] = new Chart(canvas.getContext('2d'), {
        type: 'bar',
        data: {
          labels: yearly.map(y => y.year),
          datasets: [
            { label: 'Primary',  data: yearly.map(y => +(y.primary_avg  * 100).toFixed(3)),
              backgroundColor: 'rgba(52,152,219,0.65)', borderColor: '#3498db', borderWidth: 1 },
            { label: 'Union', data: yearly.map(y => +(y.combined_avg * 100).toFixed(3)),
              backgroundColor: 'rgba(232,67,147,0.65)', borderColor: '#e84393', borderWidth: 1 },
          ],
        },
        options: {
          responsive: true, maintainAspectRatio: false, animation: false,
          plugins: { legend: { labels: { color: '#aaa', font: { size: 10 } } },
            tooltip: { callbacks: { label: ctx => {
              const y = yearly[ctx.dataIndex];
              if (ctx.datasetIndex === 0) return [`Avg: ${(y.primary_avg*100).toFixed(3)}%`, `WR: ${(y.primary_wr*100).toFixed(1)}%`, `n: ${y.primary_n}`];
              return [`Avg: ${(y.combined_avg*100).toFixed(3)}%`, `WR: ${(y.combined_wr*100).toFixed(1)}%`, `n: ${y.combined_n}`];
            } } } },
          scales: {
            ...this._darkScales(),
            y: { ...this._darkScales().y, title: { display: true, text: 'Avg Return %', color: '#888', font: { size: 9 } } },
          },
        },
      });
    },

    _renderCorrActivity() {
      const canvas = document.getElementById('corr-activity-canvas');
      if (!canvas || !this.corrResult) return;
      if (this._charts['corr-activity']) { this._charts['corr-activity'].destroy(); delete this._charts['corr-activity']; }
      const trades = this.corrResult.combined_trades
        || (this.corrResult.combined_trade_dates || []).map(d => ({ ticker: '?', trade_date: d }));
      if (!trades.length) return;
      const horizon = this.corrResult.horizon || 1;
      const spotSeries  = this.data?.spot_series || [];
      const cal         = this.data?.trade_calendar || [];
      const dates = trades.map(t => t.trade_date || t.date);
      const tradingDays = spotSeries.length > 0
        ? spotSeries.map(s => s.date)
        : [...new Set(cal.length > 0 ? cal.map(c => c.date) : dates)].sort();
      const kept = this.dedupeConc.corr
        ? this._dedupeConcurrent(trades, tradingDays, horizon)
        : trades;
      const entriesByDate = {};
      for (const t of kept) {
        const d = t.trade_date || t.date;
        entriesByDate[d] = (entriesByDate[d] || 0) + 1;
      }
      const entered = tradingDays.map(d => entriesByDate[d] || 0);
      const open    = tradingDays.map((_, i) => {
        let count = 0;
        for (let j = Math.max(0, i - horizon + 1); j <= i; j++) count += entriesByDate[tradingDays[j]] || 0;
        return count;
      });
      this._charts['corr-activity'] = new Chart(canvas.getContext('2d'), {
        type: 'bar',
        data: {
          labels: tradingDays.map(d => d.slice(0, 7)),
          datasets: [
            { type: 'line', label: 'Open Trades', data: open,
              borderColor: 'rgba(46,204,113,0.6)', backgroundColor: 'rgba(46,204,113,0.08)',
              fill: true, tension: 0.3, pointRadius: 0, borderWidth: 1.5, order: 1 },
            { type: 'bar',  label: 'Entered', data: entered,
              backgroundColor: 'rgba(52,152,219,0.7)', barThickness: 2, order: 2 },
          ],
        },
        options: {
          responsive: true, maintainAspectRatio: false, animation: false,
          plugins: { legend: { labels: { color: '#aaa', font: { size: 10 } } },
            tooltip: { mode: 'index', intersect: false,
              callbacks: { title: ctx => tradingDays[ctx[0]?.dataIndex] || '',
                           label: ctx => `${ctx.dataset.label}: ${ctx.raw}` } } },
          scales: {
            ...this._darkScales(),
            x: { ...this._darkScales().x, ticks: { ...this._darkScales().x.ticks, maxTicksLimit: 12 } },
            y: { ...this._darkScales().y, title: { display: true, text: 'Count', color: '#888', font: { size: 9 } } },
          },
        },
      });
    },

    _renderCorrBubble() {
      const canvas = document.getElementById('corr-bubble-canvas');
      if (!canvas || !this.corrResult?.tickers?.length) return;
      if (this._charts['corr-bubble']) { this._charts['corr-bubble'].destroy(); delete this._charts['corr-bubble']; }
      const minN = this.corrBubbleMinN || 1;
      const tickers = this.corrResult.tickers.filter(t => t.n >= minN);
      if (!tickers.length) return;
      const maxContrib = Math.max(1, ...tickers.filter(t => t.contrib_pct > 0).map(t => t.contrib_pct));
      const mkColor = (wr, a) => {
        const r = Math.round(232 + (52  - 232) * wr);
        const g = Math.round(67  + (152 - 67)  * wr);
        const b = Math.round(147 + (219 - 147) * wr);
        return `rgba(${r},${g},${b},${a})`;
      };
      const datasets = tickers.map(t => ({
        label: t.ticker,
        data: [{ x: t.n, y: +(t.avg_ret * 100).toFixed(4),
                 r: t.contrib_pct > 0 ? Math.max(3, (t.contrib_pct / maxContrib) * 20) : 2 }],
        backgroundColor: mkColor(t.win_rate, 0.65),
        borderColor:     mkColor(t.win_rate, 1),
        borderWidth: 1,
      }));
      const totalN = tickers.reduce((s, t) => s + (t.n || 0), 0);
      const avgPct = totalN > 0
        ? tickers.reduce((s, t) => s + (t.avg_ret || 0) * (t.n || 0), 0) / totalN * 100
        : 0;
      this._charts['corr-bubble'] = new Chart(canvas.getContext('2d'), {
        type: 'bubble',
        data: { datasets },
        plugins: [this._avgRetLinePlugin(avgPct, 'avg')],
        options: {
          responsive: true, maintainAspectRatio: false, animation: false,
          plugins: { legend: { display: false },
            tooltip: { backgroundColor: 'rgba(20,20,20,0.95)', borderColor: '#444', borderWidth: 1,
              callbacks: { label: ctx => {
                const t = tickers[ctx.datasetIndex];
                return [`${t.ticker}  n:${t.n}  avg:${(t.avg_ret*100).toFixed(3)}%  WR:${(t.win_rate*100).toFixed(1)}%  contrib:${t.contrib_pct.toFixed(1)}%`];
              } } } },
          scales: {
            ...this._darkScales(),
            x: { ...this._darkScales().x, title: { display: true, text: 'Trade Count', color: '#888', font: { size: 9 } } },
            y: { ...this._darkScales().y, title: { display: true, text: 'Avg Return %', color: '#888', font: { size: 9 } } },
          },
        },
      });
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
      if (!canvas || !this.secDetail) return;
      if (this._charts['sec-activity']) { this._charts['sec-activity'].destroy(); delete this._charts['sec-activity']; }

      // Prefer the enriched combined_trades (has ticker per entry) so the
      // dedupe-concurrent toggle can work per ticker. Fall back to plain
      // combined_trade_dates for older payloads.
      const trades = this.secDetail.combined_trades
        || (this.secDetail.combined_trade_dates || []).map(d => ({ ticker: '?', trade_date: d }));
      if (!trades.length) return;
      const horizon = this.secDetail.horizon || 1;

      const spotSeries = this.data?.spot_series || [];
      const cal = this.data?.trade_calendar || [];
      const dates = trades.map(t => t.trade_date || t.date);
      const tradingDays = spotSeries.length > 0
        ? spotSeries.map(s => s.date)
        : [...new Set(cal.length > 0 ? cal.map(c => c.date) : dates)].sort();

      const kept = this.dedupeConc.sec
        ? this._dedupeConcurrent(trades, tradingDays, horizon)
        : trades;
      const entriesByDate = {};
      for (const t of kept) {
        const d = t.trade_date || t.date;
        entriesByDate[d] = (entriesByDate[d] || 0) + 1;
      }

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

    // ── System Portfolio (third tier) ──────────────────────────────────────

    get canAddSystem() {
      // Need: a portfolio loaded, a primary analysis loaded with a metric,
      // at least one primary bin selected, and at least one secondary in the
      // corr explorer. The Portfolio section can be browsed without an
      // Analyze, but adding/editing requires one.
      if (!this.portfolioId || !this.portfolio) return false;
      if (!this.data || !this.metric) return false;
      if (!this.selectedBins20 || this.selectedBins20.size === 0) return false;
      const sels = this.corrSelections || {};
      return Object.values(sels).some(b => Array.isArray(b) && b.length > 0);
    },

    get portAggSysMap() {
      const m = {};
      for (const s of (this.portAggregate?.systems || [])) m[s.id] = s;
      return m;
    },

    get editingSystemName() {
      if (!this.editingSystemId || !this.portfolio?.systems) return '';
      const s = this.portfolio.systems.find(x => x.id === this.editingSystemId);
      return s ? s.name : '';
    },

    // Mirrors corrStats() — same shape so the stats card template can render.
    portStats() {
      const r = this.portAggregate;
      if (!r) return null;
      const yearly = r.yearly || [];
      if (!yearly.length) return null;
      const totalN  = yearly.reduce((s, y) => s + (y.combined_n  || 0), 0);
      if (!totalN) return null;
      const avgRet  = yearly.reduce((s, y) => s + (y.combined_avg || 0) * (y.combined_n || 0), 0) / totalN;
      const winRate = yearly.reduce((s, y) => s + (y.combined_wr  || 0) * (y.combined_n || 0), 0) / totalN;
      const winners = Math.round(winRate * totalN);
      const best  = yearly.reduce((b, y) => y.combined_avg > b.avg ? { yr: y.year, avg: y.combined_avg } : b, { yr: null, avg: -Infinity });
      const worst = yearly.reduce((b, y) => y.combined_avg < b.avg ? { yr: y.year, avg: y.combined_avg } : b, { yr: null, avg:  Infinity });
      const eq  = r.equity_combined || [];
      const cum = eq.length ? +(eq[eq.length - 1].value * 100).toFixed(2) : null;
      // Trade utilization across pair firings (matches corr explorer).
      const nEach = r.n_each || [];
      const sumN  = nEach.reduce((s, n) => s + n, 0);
      const minN  = nEach.length ? Math.min(...nEach) : 0;
      const union = r.combined_n || 0;
      const util  = sumN > minN ? +((union - minN) / (sumN - minN) * 100).toFixed(1) : 100;
      return {
        totalN, avgRet, winRate, winners, losers: totalN - winners, best, worst, cum, util,
        winnerAvg: r.winner_avg_ret ?? null,
        loserAvg:  r.loser_avg_ret  ?? null,
      };
    },

    portSetBubbleMinN(n) {
      this.portBubbleMinN = +n;
      this._renderPortBubble();
    },

    async loadPortfolios() {
      try {
        const r = await fetch('/api/oi-analysis/portfolios');
        if (r.ok) this.portfolios = await r.json();
      } catch (_) {}
    },

    async selectPortfolio(id) {
      this._destroyPortCharts();
      this.portAggregate = null;
      if (!id) { this.portfolio = null; return; }
      this.portLoading = true;
      try {
        const r = await fetch(`/api/oi-analysis/portfolios/${id}`);
        if (r.ok) this.portfolio = await r.json();
        else { this.portfolio = null; return; }
        await this.loadPortfolioAggregate();
      } finally { this.portLoading = false; }
    },

    async createPortfolio() {
      const name = prompt('Portfolio name:', `Research ${new Date().toISOString().slice(0, 10)}`);
      if (!name) return;
      const body = {
        name: name.trim(),
        ticker:    this.ticker,
        outcome:   this.outcome,
        date_from: this.dateFrom || null,
        date_to:   this.dateTo   || null,
      };
      try {
        const r = await fetch('/api/oi-analysis/portfolios', {
          method: 'POST', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(body),
        });
        if (!r.ok) { alert('Create failed: ' + await r.text()); return; }
        const p = await r.json();
        await this.loadPortfolios();
        this.portfolioId = p.id;
        await this.selectPortfolio(p.id);
      } catch (e) { alert('Create error: ' + e.message); }
    },

    async renamePortfolio() {
      if (!this.portfolioId || !this.portfolio) return;
      const current = this.portfolio.portfolio?.name || '';
      const name = prompt('New name:', current);
      if (!name || name.trim() === current) return;
      try {
        const r = await fetch(`/api/oi-analysis/portfolios/${this.portfolioId}`, {
          method: 'PUT', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ name: name.trim() }),
        });
        if (!r.ok) { alert('Rename failed: ' + await r.text()); return; }
        await this.loadPortfolios();
        if (this.portfolio?.portfolio) this.portfolio.portfolio.name = name.trim();
      } catch (e) { alert('Rename error: ' + e.message); }
    },

    async deletePortfolio() {
      if (!this.portfolioId) return;
      const name = this.portfolio?.portfolio?.name || '?';
      if (!confirm(`Delete portfolio "${name}" and all its systems?`)) return;
      try {
        const r = await fetch(`/api/oi-analysis/portfolios/${this.portfolioId}`, { method: 'DELETE' });
        if (!r.ok) { alert('Delete failed: ' + await r.text()); return; }
        this.portfolioId = null;
        this.portfolio = null;
        this.portAggregate = null;
        this._destroyPortCharts();
        await this.loadPortfolios();
      } catch (e) { alert('Delete error: ' + e.message); }
    },

    async addCurrentSystem() {
      if (!this.canAddSystem) return;
      const primary_bins = [...this.selectedBins20].sort((a, b) => a - b);
      const secondaries = [];
      for (const [metric, bins] of Object.entries(this.corrSelections || {})) {
        if (Array.isArray(bins) && bins.length) {
          secondaries.push({
            metric,
            bins: [...bins].sort((a, b) => a - b),
            bin_count: this.corrBinCount || 10,
          });
        }
      }
      const body = {
        primary_metric:    this.metric,
        primary_bins,
        primary_bin_count: 20,
        secondaries,
        is_short: this.portIsShort,
      };
      const editingId = this.editingSystemId;
      const url = editingId
        ? `/api/oi-analysis/portfolios/${this.portfolioId}/systems/${editingId}`
        : `/api/oi-analysis/portfolios/${this.portfolioId}/systems`;
      const method = editingId ? 'PUT' : 'POST';
      try {
        const r = await fetch(url, {
          method, headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(body),
        });
        if (!r.ok) {
          alert((editingId ? 'Save changes' : 'Add system') + ' failed: ' + await r.text());
          return;
        }
        // Clear edit mode on success
        this.editingSystemId = null;
        this.portIsShort = false;
        await this.selectPortfolio(this.portfolioId);
        await this.loadPortfolios();
      } catch (e) { alert((editingId ? 'Save changes' : 'Add system') + ' error: ' + e.message); }
    },

    cancelEditSystem() {
      this.editingSystemId = null;
      this.portIsShort = false;
    },

    async toggleSystem(sid, enabled) {
      try {
        await fetch(`/api/oi-analysis/portfolios/${this.portfolioId}/systems/${sid}`, {
          method: 'PUT', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ enabled }),
        });
        // Optimistic update
        const sys = this.portfolio?.systems?.find(s => s.id === sid);
        if (sys) sys.enabled = enabled;
        await this.loadPortfolioAggregate();
      } catch (_) {}
    },

    async renameSystem(sid, currentName) {
      const name = prompt('Rename system:', currentName);
      if (!name || name.trim() === currentName) return;
      try {
        await fetch(`/api/oi-analysis/portfolios/${this.portfolioId}/systems/${sid}`, {
          method: 'PUT', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ name: name.trim() }),
        });
        const sys = this.portfolio?.systems?.find(s => s.id === sid);
        if (sys) sys.name = name.trim();
        // Re-render aggregate labels
        await this.loadPortfolioAggregate();
      } catch (_) {}
    },

    async deleteSystem(sid) {
      const sys = this.portfolio?.systems?.find(s => s.id === sid);
      if (!confirm(`Remove "${sys?.name || 'system'}" from this portfolio?\n\n` +
                   `This does not delete it from the Library — saved copies stay available for other portfolios.`)) return;
      try {
        await fetch(`/api/oi-analysis/portfolios/${this.portfolioId}/systems/${sid}`, { method: 'DELETE' });
        await this.selectPortfolio(this.portfolioId);
        await this.loadPortfolios();
      } catch (_) {}
    },

    async editSystem(sys) {
      this.portIsShort = sys.is_short || false;
      // Mark the system as the one being edited. The "+ Add Current Setup
      // as System" button will become "Save Changes to <name>" and will
      // PUT instead of POST.
      this.editingSystemId = sys.id;

      const anchor = this.portfolio?.portfolio;
      if (!anchor) return;

      // Decide whether we need to re-run /analyze. Skip it when the page
      // is already on the same anchor + primary metric — only the primary
      // bin selection and corr selections need adjusting in that case.
      const anchorMatches =
        this.ticker   === anchor.ticker &&
        this.outcome  === anchor.outcome &&
        (this.dateFrom || '') === (anchor.date_from || '') &&
        (this.dateTo   || '') === (anchor.date_to   || '');
      const metricMatches = this.metric === sys.primary_metric;

      if (!anchorMatches || !metricMatches) {
        // Full reload path — only when something material changed.
        this.ticker   = anchor.ticker;
        this.outcome  = anchor.outcome;
        this.dateFrom = anchor.date_from || '';
        this.dateTo   = anchor.date_to   || '';
        this.metric   = sys.primary_metric;
        this.selectedBins20 = new Set(sys.primary_bins || []);
        await this.loadAnalysis();
        if (this.error) return;
        await this.secLoad();
        if (!this.secStatus.loaded) return;
      } else {
        // Fast path — page already on the right anchor + metric. Just
        // swap the primary bin selection and re-prime the corr cache.
        this.selectedBins20 = new Set(sys.primary_bins || []);
        // Re-running secLoad refreshes the cache against the new
        // primary bin filter (its `_secFilteredDates()` reads
        // selectedBins20).
        await this.secLoad();
        if (!this.secStatus.loaded) return;
      }

      // Restore the corr explorer state.
      this.corrPanelOpen = true;
      this.corrBinCount  = (sys.secondaries?.[0]?.bin_count) || 10;
      this.corrMiniData  = null;
      await this.corrLoadMiniData();
      const sels = {};
      for (const sec of (sys.secondaries || [])) {
        sels[sec.metric] = [...(sec.bins || [])];
      }
      this.corrSelections = sels;

      // Scroll the corr explorer into view so the user sees the restored state.
      this.$nextTick(() => {
        const el = document.getElementById('corr-equity-canvas')
                || document.querySelector('.if-pill');
        if (el && el.scrollIntoView) el.scrollIntoView({ behavior: 'smooth', block: 'start' });
      });
    },

    async loadPortfolioAggregate() {
      if (!this.portfolioId) return;
      this.portLoading = true;
      try {
        const r = await fetch(`/api/oi-analysis/portfolios/${this.portfolioId}/aggregate`, {
          method: 'POST', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            walk_forward: this.pageMode === 'walk_forward',
            cutoff_date:  this.pageMode === 'train_test' ? this.cutoffDate : null,
          }),
        });
        if (!r.ok) { this.portAggregate = null; return; }
        this.portAggregate = await r.json();
        await this.$nextTick();
        this._renderPortCharts();
      } catch (e) {
        console.error('portfolio aggregate', e);
      } finally { this.portLoading = false; }
    },

    _destroyPortCharts() {
      for (const k of ['port-equity', 'port-yearly', 'port-activity', 'port-bubble']) {
        if (this._charts[k]) { this._charts[k].destroy(); delete this._charts[k]; }
      }
    },

    _renderPortCharts() {
      if (!this.portAggregate || !(this.portAggregate.combined_n > 0)) {
        this._destroyPortCharts();
        return;
      }
      this._destroyPortCharts();
      this._renderPortEquity();
      this._renderPortYearly();
      this._renderPortActivity();
      this._renderPortBubble();
    },

    _renderPortEquity() {
      // Mirrors _renderCorrEquity: dual line (primary universe vs portfolio
      // union), tooltip title is the full date (daily), labels are YYYY-MM
      // for legibility on the axis.
      const canvas = document.getElementById('chart-port-equity');
      if (!canvas || !this.portAggregate) return;
      if (this._charts['port-equity']) { this._charts['port-equity'].destroy(); delete this._charts['port-equity']; }
      const eqP = this.portAggregate.equity_primary  || [];
      const eqC = this.portAggregate.equity_combined || [];
      if (!eqP.length) return;
      const cMap = Object.fromEntries(eqC.map(p => [p.date, +(p.value * 100).toFixed(4)]));
      let lastC = 0;
      const combinedAligned = eqP.map(p => {
        if (cMap[p.date] !== undefined) lastC = cMap[p.date];
        return lastC;
      });
      this._charts['port-equity'] = new Chart(canvas.getContext('2d'), {
        type: 'line',
        data: {
          labels: eqP.map(p => p.date.slice(0, 7)),
          datasets: [
            { label: 'Primary', data: eqP.map(p => +(p.value * 100).toFixed(4)),
              borderColor: '#3498db', backgroundColor: 'rgba(52,152,219,0.06)',
              borderWidth: 1.5, pointRadius: 0, tension: 0.2, fill: true },
            { label: 'Union', data: combinedAligned,
              borderColor: '#e84393', backgroundColor: 'rgba(232,67,147,0.06)',
              borderWidth: 1.5, pointRadius: 0, tension: 0.2, fill: true, spanGaps: true },
          ],
        },
        options: {
          responsive: true, maintainAspectRatio: false, animation: false,
          plugins: {
            legend: { labels: { color: '#aaa', font: { size: 10 } } },
            tooltip: { mode: 'index', intersect: false,
              callbacks: { title: ctx => eqP[ctx[0]?.dataIndex]?.date || '' } },
          },
          scales: {
            ...this._darkScales(),
            x: { ...this._darkScales().x, ticks: { ...this._darkScales().x.ticks, maxTicksLimit: 10 } },
            y: { ...this._darkScales().y, title: { display: true, text: 'Cum Return %', color: '#888', font: { size: 9 } } },
          },
        },
      });
    },

    _renderPortYearly() {
      // Mirrors _renderCorrYearly: two bar series (primary vs union) per year
      // with n + WR in the tooltip.
      const canvas = document.getElementById('chart-port-yearly');
      if (!canvas || !this.portAggregate?.yearly?.length) return;
      if (this._charts['port-yearly']) { this._charts['port-yearly'].destroy(); delete this._charts['port-yearly']; }
      const yearly = this.portAggregate.yearly;
      this._charts['port-yearly'] = new Chart(canvas.getContext('2d'), {
        type: 'bar',
        data: {
          labels: yearly.map(y => y.year),
          datasets: [
            { label: 'Primary',  data: yearly.map(y => +(y.primary_avg  * 100).toFixed(3)),
              backgroundColor: 'rgba(52,152,219,0.65)', borderColor: '#3498db', borderWidth: 1 },
            { label: 'Union', data: yearly.map(y => +(y.combined_avg * 100).toFixed(3)),
              backgroundColor: 'rgba(232,67,147,0.65)', borderColor: '#e84393', borderWidth: 1 },
          ],
        },
        options: {
          responsive: true, maintainAspectRatio: false, animation: false,
          plugins: { legend: { labels: { color: '#aaa', font: { size: 10 } } },
            tooltip: { callbacks: { label: ctx => {
              const y = yearly[ctx.dataIndex];
              if (ctx.datasetIndex === 0) return [`Avg: ${(y.primary_avg*100).toFixed(3)}%`, `WR: ${(y.primary_wr*100).toFixed(1)}%`, `n: ${y.primary_n}`];
              return [`Avg: ${(y.combined_avg*100).toFixed(3)}%`, `WR: ${(y.combined_wr*100).toFixed(1)}%`, `n: ${y.combined_n}`];
            } } } },
          scales: {
            ...this._darkScales(),
            y: { ...this._darkScales().y, title: { display: true, text: 'Avg Return %', color: '#888', font: { size: 9 } } },
          },
        },
      });
    },

    _renderPortActivity() {
      // Mirrors _renderCorrActivity: daily granularity with entered bars +
      // open trades line. Trading days come from the backend's primary
      // universe (every trade-eligible date in the anchor date range).
      const canvas = document.getElementById('chart-port-activity');
      if (!canvas || !this.portAggregate) return;
      if (this._charts['port-activity']) { this._charts['port-activity'].destroy(); delete this._charts['port-activity']; }
      const trades = this.portAggregate.combined_trades
        || (this.portAggregate.combined_trade_dates || []).map(d => ({ ticker: '?', trade_date: d }));
      if (!trades.length) return;
      const horizon = this.portAggregate.horizon || 1;
      // Trading days: prefer equity_primary's date spine, fall back to union dates.
      const spine = this.portAggregate.equity_primary || [];
      const dates = trades.map(t => t.trade_date || t.date);
      const tradingDays = spine.length
        ? [...new Set(spine.map(p => p.date))].sort()
        : [...new Set(dates)].sort();
      const kept = this.dedupeConc.port
        ? this._dedupeConcurrent(trades, tradingDays, horizon)
        : trades;
      const entriesByDate = {};
      for (const t of kept) {
        const d = t.trade_date || t.date;
        entriesByDate[d] = (entriesByDate[d] || 0) + 1;
      }
      const entered = tradingDays.map(d => entriesByDate[d] || 0);
      const open    = tradingDays.map((_, i) => {
        let count = 0;
        for (let j = Math.max(0, i - horizon + 1); j <= i; j++) count += entriesByDate[tradingDays[j]] || 0;
        return count;
      });
      this._charts['port-activity'] = new Chart(canvas.getContext('2d'), {
        type: 'bar',
        data: {
          labels: tradingDays.map(d => d.slice(0, 7)),
          datasets: [
            { type: 'line', label: 'Open Trades', data: open,
              borderColor: 'rgba(46,204,113,0.6)', backgroundColor: 'rgba(46,204,113,0.08)',
              fill: true, tension: 0.3, pointRadius: 0, borderWidth: 1.5, order: 1 },
            { type: 'bar',  label: 'Entered', data: entered,
              backgroundColor: 'rgba(52,152,219,0.7)', barThickness: 2, order: 2 },
          ],
        },
        options: {
          responsive: true, maintainAspectRatio: false, animation: false,
          plugins: { legend: { labels: { color: '#aaa', font: { size: 10 } } },
            tooltip: { mode: 'index', intersect: false,
              callbacks: { title: ctx => tradingDays[ctx[0]?.dataIndex] || '',
                           label: ctx => `${ctx.dataset.label}: ${ctx.raw}` } } },
          scales: {
            ...this._darkScales(),
            x: { ...this._darkScales().x, ticks: { ...this._darkScales().x.ticks, maxTicksLimit: 12 } },
            y: { ...this._darkScales().y, title: { display: true, text: 'Count', color: '#888', font: { size: 9 } } },
          },
        },
      });
    },

    _renderPortBubble() {
      // Mirrors _renderCorrBubble: x = trade count, y = avg ret %, bubble
      // size = contribution to total P&L, color = win rate (pink→blue).
      const canvas = document.getElementById('chart-port-bubble');
      if (!canvas || !this.portAggregate?.tickers?.length) return;
      if (this._charts['port-bubble']) { this._charts['port-bubble'].destroy(); delete this._charts['port-bubble']; }
      const minN = this.portBubbleMinN || 1;
      const tickers = this.portAggregate.tickers.filter(t => t.n >= minN);
      if (!tickers.length) return;
      const maxContrib = Math.max(1, ...tickers.filter(t => t.contrib_pct > 0).map(t => t.contrib_pct));
      const mkColor = (wr, a) => {
        const r = Math.round(232 + (52  - 232) * wr);
        const g = Math.round(67  + (152 - 67)  * wr);
        const b = Math.round(147 + (219 - 147) * wr);
        return `rgba(${r},${g},${b},${a})`;
      };
      const datasets = tickers.map(t => ({
        label: t.ticker,
        data: [{ x: t.n, y: +(t.avg_ret * 100).toFixed(4),
                 r: t.contrib_pct > 0 ? Math.max(3, (t.contrib_pct / maxContrib) * 20) : 2 }],
        backgroundColor: mkColor(t.win_rate, 0.65),
        borderColor:     mkColor(t.win_rate, 1),
        borderWidth: 1,
      }));
      const totalN = tickers.reduce((s, t) => s + (t.n || 0), 0);
      const avgPct = totalN > 0
        ? tickers.reduce((s, t) => s + (t.avg_ret || 0) * (t.n || 0), 0) / totalN * 100
        : 0;
      this._charts['port-bubble'] = new Chart(canvas.getContext('2d'), {
        type: 'bubble',
        data: { datasets },
        plugins: [this._avgRetLinePlugin(avgPct, 'avg')],
        options: {
          responsive: true, maintainAspectRatio: false, animation: false,
          plugins: { legend: { display: false },
            tooltip: { backgroundColor: 'rgba(20,20,20,0.95)', borderColor: '#444', borderWidth: 1,
              callbacks: { label: ctx => {
                const t = tickers[ctx.datasetIndex];
                return [`${t.ticker}  n:${t.n}  avg:${(t.avg_ret*100).toFixed(3)}%  WR:${(t.win_rate*100).toFixed(1)}%  contrib:${t.contrib_pct.toFixed(1)}%`];
              } } } },
          scales: {
            ...this._darkScales(),
            x: { ...this._darkScales().x, title: { display: true, text: 'Trade Count', color: '#888', font: { size: 9 } } },
            y: { ...this._darkScales().y, title: { display: true, text: 'Avg Return %', color: '#888', font: { size: 9 } } },
          },
        },
      });
    },

    portSysCellTitle(i, j) {
      if (!this.portAggregate) return '';
      const labs = this.portAggregate.system_labels || [];
      const ov = this.portAggregate.overlap_systems || [];
      const phi = this.portAggregate.phi_systems || [];
      if (i === j) return `${labs[i]}  n=${ov[i]?.[i] ?? 0}`;
      return `${labs[i]} × ${labs[j]}\nφ = ${(phi[i]?.[j] ?? 0).toFixed(3)}\nOverlap: ${ov[i]?.[j] ?? 0} trades`;
    },

    portPairCellTitle(i, j) {
      if (!this.portAggregate) return '';
      const labs = this.portAggregate.pair_labels || [];
      const ov = this.portAggregate.overlap_pairs || [];
      const phi = this.portAggregate.phi_pairs || [];
      if (i === j) return `${labs[i]}  n=${ov[i]?.[i] ?? 0}`;
      return `${labs[i]} × ${labs[j]}\nφ = ${(phi[i]?.[j] ?? 0).toFixed(3)}\nOverlap: ${ov[i]?.[j] ?? 0} trades`;
    },

    async portCsvDownload() {
      if (!this.portAggregate) return;
      const trades = this.portAggregate.combined_trades || [];
      if (!trades.length) return;
      const header = [
        'ticker', 'trade_date', 'spot_entry', 'exit_date', 'spot_exit',
        'ret_pct', 'fired_systems',
      ].join(',');
      const fmt = (v, d = 6) => v == null ? '' : Number(v).toFixed(d);
      const rows = trades.map(t => [
        t.ticker || '',
        t.trade_date || '',
        fmt(t.spot_entry, 2),
        t.exit_date || '',
        fmt(t.spot_exit, 2),
        t.ret != null ? (t.ret * 100).toFixed(6) : '',
        // Quote the fired_systems list since it can contain commas
        '"' + (t.fired_systems || []).join(' | ') + '"',
      ].join(','));
      this._downloadCsv([header, ...rows].join('\n'),
        `portfolio_${this.portfolioId}_union_${new Date().toISOString().slice(0,10)}.csv`);
    },

    // ── System Library ────────────────────────────────────────────────────

    async loadLibrarySystems() {
      try {
        const r = await fetch('/api/oi-analysis/library/systems');
        if (r.ok) this.librarySystems = await r.json();
      } catch (_) {}
    },

    async saveSystemToLibrary(sys) {
      // POST a copy of the portfolio system into the library. Default name
      // is the system's current name; user can rename in the library
      // section after.
      const proposed = prompt('Save to Library as:', sys.name || 'System');
      if (!proposed) return;
      const body = {
        name:              proposed.trim(),
        primary_metric:    sys.primary_metric,
        primary_bins:      sys.primary_bins || [],
        primary_bin_count: sys.primary_bin_count || 20,
        secondaries:       (sys.secondaries || []).map(s => ({
          metric: s.metric, bins: s.bins || [], bin_count: s.bin_count || 10,
        })),
        is_short: sys.is_short || false,
      };
      try {
        const r = await fetch('/api/oi-analysis/library/systems', {
          method: 'POST', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(body),
        });
        if (!r.ok) { alert('Save to Library failed: ' + await r.text()); return; }
        await this.loadLibrarySystems();
      } catch (e) { alert('Save to Library error: ' + e.message); }
    },

    async addLibrarySystemToPortfolio(lid) {
      if (!this.portfolioId) {
        alert('Select or create a portfolio first.');
        return;
      }
      try {
        const r = await fetch(
          `/api/oi-analysis/portfolios/${this.portfolioId}/systems/from-library/${lid}`,
          { method: 'POST' });
        if (!r.ok) { alert('Add from library failed: ' + await r.text()); return; }
        await this.selectPortfolio(this.portfolioId);
        await this.loadPortfolios();
      } catch (e) { alert('Add from library error: ' + e.message); }
    },

    async renameLibrarySystem(lid, currentName) {
      const name = prompt('Rename library system:', currentName);
      if (!name || name.trim() === currentName) return;
      try {
        await fetch(`/api/oi-analysis/library/systems/${lid}`, {
          method: 'PUT', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ name: name.trim() }),
        });
        await this.loadLibrarySystems();
      } catch (_) {}
    },

    async deleteLibrarySystem(lid) {
      const item = this.librarySystems.find(x => x.id === lid);
      if (!confirm(`Delete library system "${item?.name || '?'}"? (Existing portfolio copies are untouched.)`)) return;
      try {
        await fetch(`/api/oi-analysis/library/systems/${lid}`, { method: 'DELETE' });
        await this.loadLibrarySystems();
      } catch (_) {}
    },

    // ── IC.5 — Signal Stability (universe-wide leaderboard + scatter) ────
    // Fetches /ic-batch for the current ticker + outcome + mode, then
    // renders either the leaderboard (horizontal bar, sign_stability ↓) or
    // the scatter (IC strength × stability). Click on any bar/dot sets the
    // Metric selector and fires /analyze below.

    // Canonical cache-key for the current fetch context. Stored after a
    // successful load so expand/mode-change can detect stale data.
    _icBatchKey() {
      const cut = this.pageMode === 'train_test' ? this.cutoffDate : '';
      return `${this.ticker}:${this.outcome}:${this.pageMode}:${cut}`;
    },

    async loadIcBatch() {
      if (!this.ticker || !this.outcome) return;
      this.icBatchLoading = true;
      this.icBatchError = null;
      try {
        let url = `/api/oi-analysis/ic-batch?ticker=${encodeURIComponent(this.ticker)}`
          + `&outcome=${encodeURIComponent(this.outcome)}`;
        if (this.pageMode === 'train_test') url += `&cutoff_date=${encodeURIComponent(this.cutoffDate)}`;
        const r = await fetch(url);
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        const d = await r.json();

        if (d.status === 'not_ready') {
          // No cache entry. Single-ticker auto-triggers a background job so
          // the UX matches the old "expand and wait" behavior without the 524.
          // ALL-mode requires an explicit ⟳ Refresh click (2-3 min job).
          this.icBatchStatus = 'not_ready';
          this.icBatchData   = null;
          if (this.ticker !== 'ALL') {
            // refreshIcBatch() owns the polling-state transition from here —
            // don't call _stopIcBatchPolling() first so a queued cycle keeps
            // its timer running across successive not_ready → busy → not_ready
            // → computing ticks.
            await this.refreshIcBatch();
          } else {
            this._stopIcBatchPolling();
          }
        } else if (d.status === 'computing') {
          // Background job running — begin / continue polling.
          this.icBatchStatus = 'computing';
          this.icBatchData   = null;
          this._startIcBatchPolling();
        } else if (d.status === 'failed') {
          // Background job crashed — surface the error, stop polling.
          this.icBatchStatus = 'failed';
          this.icBatchError  = d.error || 'Background computation failed';
          this.icBatchData   = null;
          this._stopIcBatchPolling();
        } else {
          // Normal response: cached data.
          if (d.error) throw new Error(d.error);
          // Stale-cache guard: if a ⟳ Refresh was triggered, ignore any
          // cache entry older than the refresh click — keep polling until
          // the background job writes a fresh result.
          // Stale-cache guard: if ⟳ Refresh was clicked, reject any cache entry
          // older than the click. Uses epoch-ms from server (cached_at_ms) so
          // there is no timezone string parsing — NaN is impossible.
          if (this.icBatchRefreshAt && d.cached_at_ms) {
            if (d.cached_at_ms < this.icBatchRefreshAt) {
              this.icBatchStatus = 'computing';
              this._startIcBatchPolling();
              return; // don't render stale data; next poll will re-check
            }
          }
          this.icBatchRefreshAt = null; // fresh data confirmed, clear flag
          this.icBatchStatus = null;
          this.icBatchData   = d;
          this.icBatchKey    = this._icBatchKey();
          this._stopIcBatchPolling();
          await this.$nextTick();
          this._renderIcBatch();
        }
      } catch (e) {
        this.icBatchStatus = 'failed';
        this.icBatchError  = e.message;
        this._stopIcBatchPolling();
      } finally {
        this.icBatchLoading = false;
      }
    },

    async refreshIcBatch() {
      // POST /ic-batch/refresh for any ticker (single or ALL).
      // Returns immediately; background job writes cache; poll picks it up.
      if (!this.ticker || !this.outcome) return;
      this.icBatchLoading  = true;
      this.icBatchError    = null;
      this.icBatchData     = null;
      this.icBatchRefreshAt = Date.now(); // used by loadIcBatch to reject stale cache hits
      try {
        let url = `/api/oi-analysis/ic-batch/refresh?ticker=${encodeURIComponent(this.ticker)}`
          + `&outcome=${encodeURIComponent(this.outcome)}`;
        if (this.pageMode === 'train_test') url += `&cutoff_date=${encodeURIComponent(this.cutoffDate)}`;
        const r = await fetch(url, { method: 'POST' });
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        const d = await r.json();
        if (d.error) throw new Error(d.error);

        if (d.status === 'busy') {
          // Another ticker's job is occupying the slot. Treat as a queue:
          // keep (or start) polling so the next not_ready → auto-trigger
          // cycle fires automatically once the slot frees.
          this.icBatchStatus = 'queued';
          this._startIcBatchPolling();
        } else {
          // 'computing' or 'already_computing' — job is running for this key.
          this.icBatchStatus = 'computing';
          this._startIcBatchPolling();
        }
      } catch (e) {
        this.icBatchStatus = 'failed';
        this.icBatchError  = e.message;
        this._stopIcBatchPolling();
      } finally {
        this.icBatchLoading = false;
      }
    },

    _startIcBatchPolling() {
      if (this.icBatchPollTimer) return; // already polling
      const POLL_MS    = 30_000;          // 30 s between checks
      const TIMEOUT_MS = 15 * 60 * 1000; // 15-min hard stop
      this.icBatchPollStart = Date.now();
      this.icBatchPollTimer = setInterval(async () => {
        if (Date.now() - this.icBatchPollStart > TIMEOUT_MS) {
          this._stopIcBatchPolling();
          this.icBatchStatus = 'timeout';
          this.icBatchError  = 'Computation is taking longer than 15 min. '
            + 'Try expanding the section again in a few minutes, or check the server log.';
          return;
        }
        try { await this.loadIcBatch(); } catch (_) { /* loadIcBatch handles its own errors */ }
      }, POLL_MS);
    },

    _stopIcBatchPolling() {
      if (this.icBatchPollTimer) {
        clearInterval(this.icBatchPollTimer);
        this.icBatchPollTimer = null;
      }
      this.icBatchPollStart = null;
    },


    icBatchSubtitle() {
      if (this.icBatchLoading) return '';
      if (this.icBatchStatus === 'not_ready') return '— no cached data · click ⟳ Refresh to compute';
      if (this.icBatchStatus === 'computing')  return '— computing in background · polling every 30 s…';
      if (this.icBatchStatus === 'queued')     return '— queued · waiting for current IC job to finish…';
      if (this.icBatchStatus === 'failed' || this.icBatchStatus === 'timeout') return '';
      if (!this.icBatchData?.metrics?.length) return '';
      const n    = this.icBatchData.metrics.length;
      const nSup = this.icBatchData.metrics.filter(m => m.suppressed).length;
      const mode = this.ticker === 'ALL' ? 'cross-sectional' : 'time-series';
      let s = `· ${n} metrics · ${nSup} suppressed · ${mode}`;
      if (this.icBatchData.cutoff_date) s += ` · cutoff ${this.icBatchData.cutoff_date}`;
      return s;
    },

    _renderIcBatch() {
      this._renderIcLeaderboard();
      this._renderIcScatter();
      this._renderIcBeeswarm();
    },

    _renderIcLeaderboard() {
      const el = document.getElementById('chart-ic-leaderboard');
      const innerEl = document.getElementById('ic-leaderboard-inner');
      if (!el || !this.icBatchData?.metrics?.length) return;
      if (this._charts['ic-leader']) { this._charts['ic-leader'].destroy(); this._charts['ic-leader'] = null; }

      const metrics = this.icBatchData.metrics;
      // Sort: non-suppressed by IC strength desc, suppressed alphabetically at bottom.
      const nonSup = metrics.filter(m => !m.suppressed)
                            .sort((a, b) => (b.long_run_ic_abs || 0) - (a.long_run_ic_abs || 0));
      const sup    = metrics.filter(m => m.suppressed)
                            .sort((a, b) => a.name.localeCompare(b.name));
      const sorted = [...nonSup, ...sup];

      const maxAbsIc = Math.max(...nonSup.map(m => m.long_run_ic_abs || 0), 0.001);
      // Vertical column chart: fixed height, width grows with metric count so
      // x-axis labels have room (container is overflow-x:auto).
      const chartH  = 380;
      const chartW  = Math.max(sorted.length * 11, 900);

      if (innerEl) {
        innerEl.style.height = chartH + 'px';
        innerEl.style.minWidth = chartW + 'px';
      }

      const labels = sorted.map(m => m.name);
      const values = sorted.map(m => m.suppressed ? 0 : (m.long_run_ic_abs || 0));
      const bgColors = sorted.map(m => {
        if (m.suppressed) return 'rgba(100,100,100,0.25)';
        const t = Math.min((m.long_run_ic_abs || 0) / maxAbsIc, 1);
        const op = (0.25 + t * 0.65).toFixed(2);
        return (m.long_run_ic || 0) >= 0
          ? `rgba(52,152,219,${op})` : `rgba(232,67,147,${op})`;
      });

      const self = this;
      this._charts['ic-leader'] = new Chart(el, {
        type: 'bar',
        data: {
          labels,
          datasets: [{ data: values, backgroundColor: bgColors, borderWidth: 0, maxBarThickness: 12 }],
        },
        options: {
          responsive: true, maintainAspectRatio: false, animation: false,
          onClick: (e, elements) => {
            if (!elements.length) return;
            const m = sorted[elements[0].index];
            if (m && !m.suppressed) self._icBatchClickMetric(m.name);
          },
          plugins: {
            legend: { display: false },
            tooltip: {
              backgroundColor: 'rgba(20,20,20,0.95)', borderColor: '#444', borderWidth: 1,
              callbacks: {
                title: ctx => sorted[ctx[0]?.dataIndex]?.name || '',
                label: ctx => {
                  const m = sorted[ctx.dataIndex];
                  if (!m) return '';
                  if (m.suppressed) return [`Suppressed: ${m.suppression_reason || 'no decisive windows'}`];
                  return [
                    `Stability: ${m.sign_stability != null ? (m.sign_stability * 100).toFixed(1) + '%' : '—'}`,
                    `IC: ${(m.long_run_ic || 0).toFixed(4)}  abs: ${(m.long_run_ic_abs || 0).toFixed(4)}`,
                    `ε: ${(m.epsilon || 0).toFixed(4)}  windows: ${m.n_windows}`,
                    `same:${m.n_same}  opp:${m.n_opposite}  neut:${m.n_neutral}`,
                  ];
                },
              },
            },
          },
          scales: {
            x: {
              ...this._darkScales().x,
              ticks: {
                ...this._darkScales().x.ticks,
                maxRotation: 90, minRotation: 45,
                font: { size: 7 },
              },
            },
            y: {
              ...this._darkScales().y,
              min: 0, max: Math.max(maxAbsIc * 1.15, 0.01),
              title: { display: true, text: 'IC Strength (|IC|)', color: '#888', font: { size: 9 } },
              ticks: { ...this._darkScales().y.ticks, callback: v => v.toFixed(4) },
            },
          },
        },
      });
    },

    _renderIcScatter() {
      const el = document.getElementById('chart-ic-scatter');
      if (!el || !this.icBatchData?.metrics?.length) return;
      if (this._charts['ic-scatter']) { this._charts['ic-scatter'].destroy(); this._charts['ic-scatter'] = null; }

      const metrics  = this.icBatchData.metrics;
      const nonSup   = metrics.filter(m => !m.suppressed);
      const sup      = metrics.filter(m =>  m.suppressed);
      // Y-axis is Gini: exclude metrics with null gini; they'll appear automatically
      // once cross-sectional data exists (no name-based exclusion).
      const hasGini  = nonSup.filter(m => m.concentration_gini != null);
      const noGini   = nonSup.filter(m => m.concentration_gini == null);
      const supGini  = sup.filter(m =>    m.concentration_gini != null);
      const maxAbsIc = Math.max(...hasGini.map(m => m.long_run_ic_abs || 0), 0.001);
      const xMax     = Math.max(maxAbsIc * 1.1, 0.05);

      const _mkPt = m => ({
        x:           m.long_run_ic_abs || 0,
        y:           m.concentration_gini,
        name:        m.name,
        ic:          m.long_run_ic || 0,
        gini:        m.concentration_gini,
        effective_n: m.effective_n,
        stability:   m.sign_stability,
        n_same:      m.n_same,
        n_opposite:  m.n_opposite,
        suppressed:  m.suppressed,
      });
      const _color = m => {
        if (m.suppressed) return 'rgba(100,100,100,0.28)';
        const t = Math.min((m.long_run_ic_abs || 0) / maxAbsIc, 1);
        const op = (0.4 + t * 0.5).toFixed(2);
        return (m.long_run_ic || 0) >= 0
          ? `rgba(52,152,219,${op})` : `rgba(232,67,147,${op})`;
      };

      // Quadrant guide-lines + labels: strength (x) × breadth (y, inverted: low Gini = broad).
      const quadrantPlugin = {
        id: 'icScatterQuadrant',
        afterDatasetsDraw(chart) {
          const { ctx, chartArea: { left, right, top, bottom }, scales: { x: sx, y: sy } } = chart;
          const xMid = sx.getPixelForValue(xMax * 0.5);
          const yMid = sy.getPixelForValue(0.5);
          ctx.save();
          ctx.strokeStyle = 'rgba(255,255,255,0.06)';
          ctx.setLineDash([4, 4]); ctx.lineWidth = 1;
          ctx.beginPath(); ctx.moveTo(xMid, top);  ctx.lineTo(xMid, bottom); ctx.stroke();
          ctx.beginPath(); ctx.moveTo(left, yMid);  ctx.lineTo(right, yMid);  ctx.stroke();
          ctx.restore();
          ctx.save();
          ctx.font = '9px sans-serif'; ctx.fillStyle = 'rgba(255,255,255,0.15)';
          ctx.fillText('weak / concentrated',   left + 4,  top    + 14);
          ctx.fillText('strong / concentrated', xMid + 6,  top    + 14);
          ctx.fillText('weak / broad',          left + 4,  bottom -  6);
          ctx.fillText('strong / broad ★',      xMid + 6,  bottom -  6);
          ctx.restore();
        },
      };

      const self = this;
      this._charts['ic-scatter'] = new Chart(el, {
        type: 'scatter',
        data: {
          datasets: [
            {
              label: 'Metrics',
              data: hasGini.map(_mkPt),
              backgroundColor: hasGini.map(_color),
              pointRadius: 5, pointHoverRadius: 7,
            },
            {
              label: 'Suppressed',
              data: supGini.map(_mkPt),
              backgroundColor: 'rgba(100,100,100,0.25)',
              pointRadius: 3, pointHoverRadius: 5,
            },
          ],
        },
        options: {
          responsive: true, maintainAspectRatio: false, animation: false,
          onClick: (e, elements) => {
            if (!elements.length) return;
            const ds = elements[0].datasetIndex === 0 ? hasGini : supGini;
            const m  = ds[elements[0].index];
            if (m && !m.suppressed) self._icBatchClickMetric(m.name);
          },
          plugins: {
            legend: { display: false },
            subtitle: noGini.length > 0 ? {
              display: true,
              text: `${noGini.length} metrics excluded — no cross-sectional coverage`,
              color: '#666', font: { size: 9 }, padding: { top: 0, bottom: 3 }, align: 'start',
            } : { display: false },
            tooltip: {
              backgroundColor: 'rgba(20,20,20,0.95)', borderColor: '#444', borderWidth: 1,
              callbacks: {
                title: ctx => {
                  const ds = ctx[0]?.datasetIndex === 0 ? hasGini : supGini;
                  return ds[ctx[0]?.dataIndex]?.name || '';
                },
                label: ctx => {
                  const pt = ctx.raw;
                  if (pt.suppressed) return 'Suppressed (no decisive windows)';
                  const nStr = pt.effective_n != null ? pt.effective_n.toFixed(1) : '—';
                  return [
                    `IC abs: ${pt.x.toFixed(4)}  (${pt.ic >= 0 ? '+' : ''}${pt.ic.toFixed(4)})`,
                    `Gini: ${pt.gini != null ? pt.gini.toFixed(3) : '—'}  eff N: ${nStr}`,
                    `Stability: ${pt.stability != null ? (pt.stability * 100).toFixed(1) + '%' : '—'}`,
                    `same: ${pt.n_same}  opp: ${pt.n_opposite}`,
                  ];
                },
              },
            },
          },
          scales: {
            x: {
              ...this._darkScales().x,
              min: 0, max: xMax,
              title: { display: true, text: 'IC Strength (|IC|)', color: '#888', font: { size: 9 } },
              ticks: { ...this._darkScales().x.ticks, callback: v => v.toFixed(3) },
            },
            y: {
              ...this._darkScales().y,
              min: 0, max: 1.02,
              title: { display: true, text: 'Concentration (Gini)   0 = broad   1 = concentrated', color: '#888', font: { size: 9 } },
              ticks: { ...this._darkScales().y.ticks, callback: v => v.toFixed(1) },
            },
          },
        },
        plugins: [quadrantPlugin],
      });
    },

    // ── IC.7 Breadth Beeswarm ─────────────────────────────────────────

    _renderIcBeeswarm() {
      const el = document.getElementById('chart-ic-beeswarm');
      if (!el || !this.icBatchData?.metrics?.length) return;
      if (this._charts['ic-beeswarm']) {
        this._charts['ic-beeswarm'].destroy();
        this._charts['ic-beeswarm'] = null;
      }

      const metrics = this.icBatchData.metrics;

      // Separate plottable metrics (have gini) from excluded ones (null = no
      // cross-sectional data for that metric — iv_25d_call_30d etc).
      // undefined gini = stale cache, not expected to reach here.
      const hasGini = metrics.filter(m => m.concentration_gini != null);
      const noGini  = metrics.filter(m => m.concentration_gini == null);
      const nonSup  = hasGini.filter(m => !m.suppressed);
      const sup     = hasGini.filter(m =>  m.suppressed);

      if (hasGini.length === 0) return; // nothing to plot

      const maxAbsIc = Math.max(...nonSup.map(m => m.long_run_ic_abs || 0), 0.001);

      // Dot radius in pixels — uniform across all dots (it's a beeswarm, not a bubble).
      const DOT_R = 6;
      // Approximate canvas dimensions for beeswarm collision geometry.
      // The pane is 1/3 of the survey grid; chart height is the 380px pane
      // body minus the subtitle padding.
      const CW = Math.max(el.parentElement?.clientWidth || 400, 180);
      const CH = 330;      // usable canvas height (px)
      const X_SPAN = 1.04; // gini axis spans -0.02 .. 1.02
      const Y_SPAN = 1.30; // Y axis spans -0.65 .. 0.65
      // Dot radius in data-unit coordinates
      const xr = DOT_R / CW * X_SPAN;
      const yr = DOT_R / CH * Y_SPAN;

      // Simple 1-D beeswarm: process metrics sorted by gini (left → right),
      // find the nearest Y level (±k full-diameters from centre) where the
      // incoming dot does not overlap any already-placed dot.
      const _beeswarm = (arr) => {
        const sorted = [...arr].sort((a, b) => a.concentration_gini - b.concentration_gini);
        const placed = [];
        for (const m of sorted) {
          const gx = m.concentration_gini;
          let gy = 0;
          outer:
          for (let k = 0; k <= 40; k++) {
            const ys = k === 0 ? [0] : [k * yr * 2.1, -k * yr * 2.1];
            for (const cy of ys) {
              if (Math.abs(cy) > 0.60) continue;   // clip at axis edge
              let ok = true;
              for (const p of placed) {
                const dxn = (gx - p.gx) / xr;
                const dyn = (cy - p.gy) / yr;
                if (dxn * dxn + dyn * dyn < 4.2) { ok = false; break; }
              }
              if (ok) { gy = cy; break outer; }
            }
          }
          placed.push({ gx, gy, _m: m });
        }
        return placed;
      };

      const pNonSup = _beeswarm(nonSup);
      const pSup    = _beeswarm(sup);

      // Color: same palette and opacity curve as Signal Scatter for consistency.
      // Blue = positive IC, pink = negative IC; opacity encodes IC magnitude.
      const _color = (m) => {
        const t  = Math.min((m.long_run_ic_abs || 0) / maxAbsIc, 1);
        const op = (0.28 + t * 0.62).toFixed(2);
        return (m.long_run_ic || 0) >= 0
          ? `rgba(52,152,219,${op})` : `rgba(232,67,147,${op})`;
      };

      const nonSupData = pNonSup.map(p => ({ x: p.gx, y: p.gy, r: DOT_R, _m: p._m }));
      const supData    = pSup.map(p =>    ({ x: p.gx, y: p.gy, r: DOT_R, _m: p._m }));

      const self = this;

      this._charts['ic-beeswarm'] = new Chart(el, {
        type: 'bubble',
        data: {
          datasets: [
            {
              label:           'Active',
              data:            nonSupData,
              backgroundColor: pNonSup.map(p => _color(p._m)),
              borderWidth:     0,
            },
            {
              label:           'Suppressed',
              data:            supData,
              backgroundColor: 'rgba(100,100,100,0.10)',
              borderWidth:     0,
            },
          ],
        },
        options: {
          responsive: true, maintainAspectRatio: false, animation: false,
          onClick(e, elements) {
            if (!elements.length) return;
            if (elements[0].datasetIndex !== 0) return; // suppressed → inert
            const m = nonSupData[elements[0].index]?._m;
            if (m) self._icBatchClickMetric(m.name);
          },
          plugins: {
            legend: { display: false },
            // Show excluded-metric count as subtitle when any are absent.
            subtitle: noGini.length > 0 ? {
              display: true,
              text:    `${noGini.length} metrics excluded — no cross-sectional coverage`,
              color:   '#666',
              font:    { size: 9 },
              padding: { top: 0, bottom: 3 },
              align:   'start',
            } : { display: false },
            tooltip: {
              backgroundColor: 'rgba(20,20,20,0.95)', borderColor: '#444', borderWidth: 1,
              callbacks: {
                label: ctx => {
                  const m    = ctx.raw._m;
                  const gini = m.concentration_gini.toFixed(3);
                  const effn = m.effective_n != null ? m.effective_n.toFixed(1) : '—';
                  if (m.suppressed) {
                    return [
                      `${m.name}  [suppressed]`,
                      `Gini ${gini}  ·  eff N ${effn}`,
                      `Reason: ${m.suppression_reason || 'no decisive windows'}`,
                    ];
                  }
                  const ic   = (m.long_run_ic   || 0).toFixed(4);
                  const stab = m.sign_stability != null
                    ? (m.sign_stability * 100).toFixed(1) + '%' : '—';
                  return [
                    m.name,
                    `IC ${ic}  ·  stability ${stab}`,
                    `Gini ${gini}  ·  eff N ${effn}`,
                  ];
                },
              },
            },
          },
          scales: {
            x: {
              ...this._darkScales().x,
              min: -0.02, max: 1.02,
              title: {
                display: true,
                text:    'Concentration (Gini)   0 = equal   1 = one ticker dominates',
                color:   '#888',
                font:    { size: 9 },
              },
              ticks: {
                ...this._darkScales().x.ticks,
                callback: v => v.toFixed(1),
                maxTicksLimit: 7,
              },
            },
            y: {
              display: false,    // Y axis hidden — only used for spread
              min: -0.65,
              max:  0.65,
            },
          },
        },
      });
    },

    // ── IC.7 Signal Decomposition ──────────────────────────────────────

    _icDecompKey() {
      const cut = this.pageMode === 'train_test' ? this.cutoffDate : '';
      return `${this.metric}:${this.outcome}:${this.pageMode}:${cut}`;
    },


    async loadIcDecomp() {
      if (this.ticker !== 'ALL' || !this.metric || !this.outcome) return;
      this.icDecompLoading = true;
      this.icDecompError   = null;
      try {
        let url = `/api/oi-analysis/ic-decomp?metric=${encodeURIComponent(this.metric)}`
          + `&outcome=${encodeURIComponent(this.outcome)}`;
        if (this.pageMode === 'train_test') url += `&cutoff_date=${encodeURIComponent(this.cutoffDate)}`;
        const r = await fetch(url);
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        const d = await r.json();
        if (d.error) throw new Error(d.error);
        this.icDecompData = d;
        this.icDecompKey  = this._icDecompKey();
        this.$nextTick(() => { this._renderIcDecomp(); this._renderIcLorenz(); });
      } catch (e) {
        this.icDecompError = e.message;
      } finally {
        this.icDecompLoading = false;
      }
    },

    _renderIcDecomp() {
      const el = document.getElementById('chart-ic-decomp');
      if (!el || !this.icDecompData?.tickers?.length) return;
      if (this._charts['ic-decomp']) {
        this._charts['ic-decomp'].destroy();
        this._charts['ic-decomp'] = null;
      }

      const d       = this.icDecompData;
      // Only plot tickers that fired on the flagged side at least once
      const tickers = d.tickers.filter(t => t.avg_ret_flagged != null && t.n_flagged > 0);
      const refIc   = d.reference_ic || 0;

      // Bubble size: radius scaled by sqrt(n_flagged) so area ∝ n_flagged.
      // Clamp 3–18px so small-sample dots are still clickable.
      const maxNF = Math.max(...tickers.map(t => t.n_flagged), 1);
      const bubbleR = t => Math.max(3, Math.min(18, Math.sqrt(t.n_flagged / maxNF) * 14));

      // Color: same direction as reference_ic → blue, opposite → pink.
      // Opacity encodes magnitude so noise tickers fade to background.
      const maxAbsScore = Math.max(...tickers.map(t => Math.abs(t.score)), 1e-9);
      const bgColors = tickers.map(t => {
        const mag = Math.min(Math.abs(t.score) / maxAbsScore, 1);
        const op  = (0.35 + mag * 0.50).toFixed(2);
        const sameSign = refIc >= 0 ? t.score >= 0 : t.score < 0;
        return sameSign ? `rgba(52,152,219,${op})` : `rgba(232,67,147,${op})`;
      });

      // Y-axis mode: 'raw' = avg_ret_flagged; 'basket' = avg_ret_flagged_vs_basket.
      // Both are direction-normalised (predicted-winner side). 'basket' subtracts the
      // cross-sectional mean return over each ticker's own flagged days — per-ticker
      // correction, NOT a uniform axis shift. Clearly labelled as vs 128-ticker basket.
      const useBasket = this.icDecompYMode === 'basket';
      const yVal  = t => (useBasket
        ? (t.avg_ret_flagged_vs_basket ?? t.avg_ret_flagged)
        : t.avg_ret_flagged) * 100;
      const yTitle = useBasket
        ? 'Avg return when flagged vs 128-ticker basket  (%)'
        : 'Avg return when flagged  (%)';

      const bubbleData = tickers.map(t => ({
        x:  t.sign_agreement_rate,
        y:  yVal(t),
        r:  bubbleR(t),
        _t: t,
      }));

      // Reference-line plugin: vertical at x=0.5 (random baseline),
      // horizontal at y=0 (break-even). Drawn before datasets.
      const refLinePlugin = {
        id: 'icDecompRefLines',
        beforeDatasetsDraw(chart) {
          const { ctx, scales: { x: xs, y: ys } } = chart;
          ctx.save();
          ctx.strokeStyle = 'rgba(180,180,180,0.35)';
          ctx.lineWidth   = 1;
          ctx.setLineDash([4, 3]);
          const x05 = xs.getPixelForValue(0.5);
          ctx.beginPath(); ctx.moveTo(x05, ys.top);  ctx.lineTo(x05, ys.bottom); ctx.stroke();
          const y0  = ys.getPixelForValue(0);
          ctx.beginPath(); ctx.moveTo(xs.left, y0);  ctx.lineTo(xs.right, y0);   ctx.stroke();
          ctx.restore();
        },
      };

      const self = this;
      this._charts['ic-decomp'] = new Chart(el, {
        type: 'bubble',
        data: { datasets: [{ data: bubbleData, backgroundColor: bgColors, borderWidth: 0 }] },
        options: {
          responsive: true, maintainAspectRatio: false, animation: false,
          onClick(e, elements) {
            if (!elements.length) return;
            const tkr = bubbleData[elements[0].index]._t.ticker;
            self.ticker = tkr;
            const sel = document.querySelector('select[x-model="ticker"]');
            if (sel) sel.value = tkr;
            self.loadAnalysis();
          },
          plugins: {
            legend: { display: false },
            tooltip: {
              callbacks: {
                label: ctx => {
                  const t    = ctx.raw._t;
                  const agr  = (t.sign_agreement_rate * 100).toFixed(1);
                  const ret  = yVal(t).toFixed(3);
                  const hr   = t.hit_rate_flagged != null
                    ? ` · hit ${(t.hit_rate_flagged * 100).toFixed(1)}%` : '';
                  const yLbl = useBasket ? 'vs basket' : 'avg ret';
                  return [`${t.ticker}`,
                          `sign agr: ${agr}%  ·  ${yLbl}: ${ret}%${hr}`,
                          `n flagged: ${t.n_flagged}  ·  score: ${t.score.toFixed(6)}`];
                },
              },
            },
          },
          scales: {
            x: {
              ...this._darkScales().x,
              min: 0.35, max: 0.65,
              title: { display: true, text: 'Sign agreement rate  (0.5 = random)',
                       color: '#888', font: { size: 9 } },
              ticks: { ...this._darkScales().x.ticks,
                       callback: v => (v * 100).toFixed(0) + '%', maxTicksLimit: 7 },
            },
            y: {
              ...this._darkScales().y,
              title: { display: true, text: yTitle, color: '#888', font: { size: 9 } },
              ticks: { ...this._darkScales().y.ticks,
                       callback: v => v.toFixed(2) + '%', maxTicksLimit: 7 },
            },
          },
        },
        plugins: [refLinePlugin],
      });
    },

    // ── IC.7 Lorenz Curve ─────────────────────────────────────────────

    _renderIcLorenz() {
      const el = document.getElementById('chart-ic-lorenz');
      if (!el || !this.icDecompData?.tickers?.length) return;
      if (this._charts['ic-lorenz']) {
        this._charts['ic-lorenz'].destroy();
        this._charts['ic-lorenz'] = null;
      }

      const tickers = this.icDecompData.tickers;
      const gini    = this.icDecompData.concentration_gini;
      const effN    = this.icDecompData.effective_n;
      const n       = tickers.length;

      // Sort absolute scores ascending — smallest contributors first.
      const absSorted = tickers
        .map(t => Math.abs(t.score))
        .sort((a, b) => a - b);

      const total = absSorted.reduce((s, v) => s + v, 0);
      if (total === 0) return; // degenerate: all scores zero

      // Build Lorenz points: (0,0) → cumulative (fraction of tickers, fraction of |score|).
      const lorenzPts = [{ x: 0, y: 0 }];
      let cumSum = 0;
      absSorted.forEach((v, i) => {
        cumSum += v;
        lorenzPts.push({ x: (i + 1) / n, y: cumSum / total });
      });

      // Diagonal = perfect equality line.
      const diagPts = [{ x: 0, y: 0 }, { x: 1, y: 1 }];

      // Custom plugin: shade the area between the diagonal and the Lorenz
      // curve (below the diagonal = concentration). Runs before datasets so
      // both lines render on top of the fill.
      const shadePlugin = {
        id: 'lorenzShade',
        beforeDatasetsDraw(chart) {
          const { ctx, scales: { x: xs, y: ys } } = chart;
          ctx.save();
          ctx.beginPath();
          // Trace diagonal forward (0,0) → (1,1)
          ctx.moveTo(xs.getPixelForValue(0), ys.getPixelForValue(0));
          ctx.lineTo(xs.getPixelForValue(1), ys.getPixelForValue(1));
          // Trace Lorenz curve backward from (1,1) to (0,0)
          for (let i = lorenzPts.length - 1; i >= 0; i--) {
            ctx.lineTo(
              xs.getPixelForValue(lorenzPts[i].x),
              ys.getPixelForValue(lorenzPts[i].y),
            );
          }
          ctx.closePath();
          ctx.fillStyle = 'rgba(52,152,219,0.10)';
          ctx.fill();
          ctx.restore();
        },
      };

      // Title: Gini + eff N aligned to the right so it floats over the chart
      // without blocking the curve (concentrated metrics bow far below the
      // diagonal, so the top-right corner is always empty).
      const giniStr = gini  != null ? `Gini = ${gini.toFixed(3)}` : '';
      const effStr  = effN  != null ? `  ·  eff N = ${effN.toFixed(1)} / ${n}` : '';

      this._charts['ic-lorenz'] = new Chart(el, {
        type: 'line',
        data: {
          datasets: [
            {
              // Diagonal reference line (equal distribution)
              data:        diagPts,
              borderColor: 'rgba(180,180,180,0.50)',
              borderWidth: 1,
              borderDash:  [4, 3],
              pointRadius: 0,
              tension:     0,
              order:       2,
            },
            {
              // Lorenz curve
              data:        lorenzPts,
              borderColor: 'rgba(52,152,219,0.85)',
              borderWidth: 1.5,
              pointRadius: 0,
              tension:     0,
              fill:        false,
              order:       1,
            },
          ],
        },
        options: {
          responsive: true, maintainAspectRatio: false, animation: false,
          plugins: {
            legend: { display: false },
            title: (giniStr.length > 0) ? {
              display:  true,
              text:     giniStr + effStr,
              color:    '#888',
              font:     { size: 10 },
              align:    'end',
              padding:  { top: 6, bottom: 0 },
            } : { display: false },
            tooltip: {
              callbacks: {
                title: ctx => {
                  const pct = (ctx[0].raw.x * 100).toFixed(0);
                  return `Bottom ${pct}% of tickers by |IC contribution|`;
                },
                label: ctx => {
                  if (ctx.datasetIndex === 1) {
                    return `account for ${(ctx.raw.y * 100).toFixed(1)}% of total |IC contribution|`;
                  }
                  return 'equal distribution';
                },
              },
            },
          },
          scales: {
            x: {
              ...this._darkScales().x,
              type: 'linear',
              min: 0, max: 1,
              title: {
                display: true,
                text:    'Cumulative share of tickers (sorted by |contribution| ↑)',
                color:   '#888',
                font:    { size: 9 },
              },
              ticks: {
                ...this._darkScales().x.ticks,
                callback:     v => (v * 100).toFixed(0) + '%',
                maxTicksLimit: 6,
              },
            },
            y: {
              ...this._darkScales().y,
              min: 0, max: 1,
              title: {
                display: true,
                text:    'Cumulative share of |IC contribution|',
                color:   '#888',
                font:    { size: 9 },
              },
              ticks: {
                ...this._darkScales().y.ticks,
                callback:     v => (v * 100).toFixed(0) + '%',
                maxTicksLimit: 6,
              },
            },
          },
        },
        plugins: [shadePlugin],
      });
    },

    // Set the Metric selector to `name` and trigger /analyze.
    // Belt-and-suspenders: update both Alpine state and the DOM select value
    // (same pattern as init()'s _forceSelect) so browser form-cache can't
    // resist the programmatic change.
    _icBatchClickMetric(name) {
      this.metric = name;
      const el = document.querySelector('select[x-model="metric"]');
      if (el) el.value = name;
      this.loadAnalysis();
    },
  }));
});
