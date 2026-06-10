'use strict';

// ── Quantile compare-mode constants ───────────────────────────────────────────
// P3: renamed from LADDER_* — used by both Horizon mode (6-bar groups)
// and Entry mode's horizon sub-control. The palette runs short→long.
const HORIZON_LIST    = [1, 3, 5, 7, 10, 20];
const HORIZON_PALETTE = ['#74b9ff', '#0984e3', '#48dbfb', '#00b894', '#fdcb6e', '#ff7675'];

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
    // Metric family groupings — built from /columns feature_families.
    // Drives <optgroup> rendering in all metric dropdowns via groupMetricsByFamily().
    metricFamilyLookup: {},  // metric_name -> {family_num, family_name}
    ticker: '', metric: '', outcome: '',
    // P2: Signal Survey has its own outcome (separate from the main chart's
    // `outcome`). Persisted across page reloads via localStorage.
    surveyOutcome: '',
    // Decomp pane's drilled-into metric. Decoupled from Analyze's
    // this.metric so a leaderboard click does NOT mutate the Analyze
    // section's metric selector. Set by _icBatchClickMetric or
    // loadIcDecomp(name) — the only entry points that surface a metric
    // into the Survey's decomp visuals.
    surveyDecompMetric: '',
    dateFrom: '', dateTo: new Date().toISOString().slice(0, 10),
    // Page-wide bin mode. Drives every binning analysis on the page so
    // primary / corr explorer / portfolio aggregate all use the same flavor.
    //   'in_sample'    — full-history bin thresholds
    //   'walk_forward' — per-ticker bisect_left against running history, 252d warmup
    pageMode: 'in_sample',
    // Group 8: cutoffDate is no longer user-selectable. Auto-discovered
    // from tt_bins via GET /tt-cutoff on page load. Defaulting to a
    // sane string so the initial render before the fetch resolves doesn't
    // emit `undefined` in URLs; gets overwritten by the actual value.
    cutoffDate: '2024-01-01',
    // selectedBins20 is the sole selection state (1-20). D1+D10 in 10-bin = bins {1,2,19,20}.
    selectedBins20: new Set([1, 2, 19, 20]),
    equityMode: 'concurrent',   // 'concurrent' | 'non_overlapping'
    equityXMode: 'calendar',   // 'calendar' | 'sequential'
    decileBins: 10,                 // P3: always 10 (5/10/20 toggle removed)
    // Load-button pending flags. Bin-click / granularity changes mark pending;
    // Load button (or mode/outcome switch for primary) clears and re-renders.
    primaryPending: false,
    secPending: false,
    _secBinCountChanged: false,  // secondary Load needs secScan when bin count changed
    decileBinsData: null,
    // P3: lower section mode. Renamed from decileCompareMode for clarity —
    // the new names describe the *dimension* being analyzed, not chart shape:
    //   single        — one outcome, 10 bars
    //   entry         — OC vs CC at one horizon, 20 bars
    //   horizon       — 6 horizons at one anchor, 60 bars
    //   overnight_gap — mean(ret_1d_fwd_cc − ret_1d_fwd_oc) per bin, 10 bars
    decileMode: 'single',
    decileHorizonAnchor: 'oc',      // 'oc' | 'cc' — Horizon mode sub-toggle (was decileLadderAnchor)
    decileEntryHorizon: 5,          // 1|3|5|7|10|20 — Entry mode sub-toggle
    decileActiveOutcome: 'ret_5d_fwd_oc',  // The outcome currently driving Single mode + downstream visuals.
                                           // Default ret_5d_fwd_oc; user changes via Single mode inline
                                           // dropdown OR (P4) bar-click promotion in Entry/Horizon modes.

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
    secScanKey: null,
    secPolling: false,
    _secDrillVersion: 0,    // P1: version counter — discards stale secDrillMetric responses
    secScannerStale: false,  // M1: scanner results stale due to primary-context change
    _secCacheParams: null,   // M2: {ticker,metric,outcome,dateFrom,dateTo} at cache build time
    _lastHeatmapKey: null,   // loadHeatmapIfChanged guard — tracks last rendered heatmap inputs
    secAdvisoryOpen: true,   // M1: advisory block (scanner+minis) expand/collapse — visual only

    // Multi-Metric Correlation Explorer
    corrPanelOpen: false,
    // corrBinCount removed — unified into secBinCount so detail and minis always match
    corrMiniData: null,
    corrMiniLoading: false,
    corrMiniComputedBins: null,  // P4: primary-bin selection at last mini compute
    corrSelections: {},
    corrResult: null,
    corrLoading: false,
    corrBubbleMinN: 1,

    // Signal Portfolio (third tier — persisted research portfolios)
    portfolios: [],          // [{id, name, ticker, outcome, date_from, date_to, signal_count}, ...]
    portfolioId: null,       // currently-selected portfolio id (or null)
    portfolio: null,         // {portfolio: {...}, signals: [...]}
    portAggregate: null,     // last /aggregate response
    portLoading: false,
    portSignalPick: '',      // signal_id selected in the "add signal" dropdown
    portBubbleMinN: 1,       // bubble chart min-n filter

    // Trade Activity dedupe — when on, a new entry for a ticker is skipped
    // while a previous trade of the same ticker is still inside its horizon.
    // Independent per chart so the user can A/B them visually.
    dedupeConc: { primary: false, sec: false, corr: false, port: false },

    // All-Ticker Metric Bins (top-of-page browser, independent of analysis)
    // Bucket A step 4: local mode state per the per-pane independence rule.
    // Pane reads topBinsMode / topBinsCutoffDate, NOT this.pageMode.
    // topBinsDataByMode holds per-mode slots; switching topBinsMode swaps
    // which slot is displayed (in-memory only). Outcome dropdown and
    // cutoff input changes also do NOT auto-fetch — only ⟳ Refresh
    // triggers a network round trip.
    topBinsExpanded: false,
    topBinsLoading:  false,
    topBinsData:     null,
    topBinsOutcome:  'ret_5d_fwd_oc',
    topBinsMode:        'in_sample',
    topBinsCutoffDate:  '2024-01-01',
    topBinsDataByMode: {
      in_sample:    null,  // most-recent topBinsData dict for this mode; null = never loaded
      walk_forward: null,
      train_test:   null,
    },
    topBinsStatusByMode: {
      in_sample:    'empty',  // 'empty' | 'loading' | 'ready' | 'no_data' | 'error'
      walk_forward: 'empty',
      train_test:   'empty',
    },
    // Init-time metadata: {exists, cached_at} from /global-metric-bins/meta
    // for the default selectors. Lets the collapsed-pane breadcrumb show
    // "last: …" before the pane is ever expanded. Cleared and replaced
    // by a per-mode slot's own cached_at once the user expands + loads.
    topBinsMeta: null,

    // P1: 12-outcome analyze bundle (drives P3+ mode views).
    // Two-stage render: /analyze fires first for ret_5d_fwd_oc immediately;
    // then loadAnalyzeBundle() pulls the full 12-outcome bundle in the
    // background. ALL-mode bundles are persisted to analyze_cache (cache
    // key omits FROM/TO); single-ticker bundles compute inline (~2-5s) and
    // live only in this Alpine state.
    analyzeBundle:       null,    // {trade_meta?, per_outcome_returns?, per_bin, rolling_ic, ...}
    analyzeBundleStatus: 'idle',  // 'idle' | 'computing' | 'ready' | 'failed' | 'not_computed'
    analyzeBundleKey:    null,    // tracks current (ticker, metric, mode, cutoff) so we don't redundantly refetch
    analyzeBundleError:  null,    // surfaces server-side errors for inspection
    _analyzeBundlePollTimer: null,
    // v6 lazy-load: ALL-mode bundle arrives slim (per_bin + rolling_ic, ~1.3MB).
    // trade_meta and per_outcome_returns[outcome] fetch on demand the first
    // time a user action needs them (Gap mode, non-default promotion, Flat
    // trade table for non-default outcome). Results cached on analyzeBundle
    // for the session. _deferredLoading drives a spinner overlay; the two
    // in-flight maps de-dupe concurrent triggers so a rapid click sequence
    // doesn't fire duplicate fetches. Single-ticker bundles return inline
    // with all 4 parts present — the ensure helpers short-circuit on that path.
    _deferredLoading:    false,
    _tradeMetaInFlight:  null,    // Promise | null
    _outcomesInFlight:   {},      // { outcome: Promise } per-outcome dedupe

    // Threshold Drift (walk-forward bin boundaries over time)
    // Bucket A step 7: Threshold Drift owns its own binning mode +
    // cutoff. Pane reads tdBinMode (NOT this.pageMode), and tdScope is
    // the visualization toggle (renamed from tdMode to free the word
    // "mode" for the actual binning mode). IS / TT modes show a clean
    // placeholder — the in-sample threshold-drift computation is a
    // FUTURE task (it's a settled-threshold time series, NOT the WF
    // logic with a flag flipped), so no endpoint exists yet for those
    // selector positions. Refresh is gated to walk_forward.
    tdExpanded:     false,
    tdLoading:      false,
    tdData:         null,
    tdMetric:       '',
    tdOutcome:      'ret_5d_fwd_oc',
    tdBinsToShow:   [1, 20],
    tdScope:        'ratio',   // 'ratio' (all tickers, dimensionless) | 'native_single' — RENAMED from tdMode
    tdSingleTicker: '',
    tdBinMode:       'in_sample',         // local binning mode for this pane
    tdCutoffDate:    '2024-01-01',
    tdDataByMode: {
      walk_forward: null,                // { tdData, loaded_at }
      in_sample:    null,
      train_test:   null,
    },
    // Init-time metadata: {exists, cached_at} from /threshold-drift/meta
    // for the default selectors (set after the columns fetch). Lets the
    // collapsed-pane breadcrumb show "last: …" before the pane is
    // expanded. Cached_at is the in-process load timestamp — clears on
    // server restart since the threshold-drift cache is in-memory only.
    tdMeta: null,

    // IC.5 — Signal Stability (universe-wide IC leaderboard + scatter)
    // Lazy-loaded on first expand (user-initiated, same as All-Ticker Metric
    // Bins). Also reloads when the section is open and mode changes. Fresh
    // compute takes ~2-3 min on the VPS; cached reads are sub-second.
    // Bucket A step 5: Signal Survey owns its own mode + cutoff state.
    // Pane reads surveyMode / surveyCutoffDate, NOT this.pageMode /
    // this.cutoffDate. Per-mode in-memory cache (surveyDataByMode); the
    // mode-pill swap is in-memory only. Outcome dropdown and cutoff
    // input changes also do NOT auto-fetch — only the ⟳ Refresh button
    // dispatches a network round trip.
    surveyMode:          'in_sample',
    surveyCutoffDate:    '2024-01-01',
    // Per-(slot bucket, slot key) cache. Two buckets, not three:
    //   "default"     ← shared by IS and WF (rolling IC is rank-correlation
    //                    based and mode-independent; the backend cache also
    //                    keys IS and WF to the same row at mode_tag="default").
    //   "train_test"  ← TT only; gets its own bucket because reference_ic
    //                    is the pre-cutoff mean instead of the full mean
    //                    (genuinely different leaderboard numbers).
    // Switching IS↔WF is a no-op slot-wise — both pills point at the same
    // bucket — so the swap is truly instant after either has been loaded.
    // Slot key inside a bucket:
    //   default:     surveyOutcome                            (e.g., "ret_5d_fwd_oc")
    //   train_test:  `${surveyOutcome}|${surveyCutoffDate}`   (cutoff partitions data)
    // Each leaf holds { icBatchData, icBatchKey, icBatchStatus,
    //                   icBatchError, icDecompData, icDecompKey,
    //                   icDecompError }.
    surveyDataByMode: {
      default:    {},
      train_test: {},
    },
    // Per-mode coarse status — written for telemetry; currently no readers.
    surveyStatusByMode: {
      in_sample:    'empty',  // 'empty' | 'loading' | 'ready' | 'no_data' | 'error'
      walk_forward: 'empty',
      train_test:   'empty',
    },
    icBatchData:      null,
    icBatchLoading:   false,
    icBatchError:     null,
    icBatchKey:       null,           // last-loaded "ticker:outcome:mode:cutoff" key
    icBatchSeq:       0,              // incremented on every loadIcBatch() call; stale responses check seq before writing state
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

      const [tkRes, colRes, ttRes] = await Promise.all([
        fetch('/api/factor-analysis/tickers'),
        fetch('/api/factor-analysis/columns'),
        // Group 8: discover the TT cutoff frozen in tt_bins. User no
        // longer picks it; backend reads it once and the frontend uses
        // whatever the table holds. If the fetch fails, the existing
        // 2024-01-01 default stays — TT mode still works against
        // whatever the build froze.
        fetch('/api/factor-analysis/tt-cutoff'),
      ]);
      if (ttRes && ttRes.ok) {
        try {
          const tt = await ttRes.json();
          if (tt && tt.cutoff_date) this.cutoffDate = tt.cutoff_date;
        } catch (_) {}
      }
      if (tkRes.ok) {
        this.tickers = ['ALL', ...(await tkRes.json())];
        this.ticker = 'ALL';
      }
      if (colRes.ok) {
        const cols = await colRes.json();
        this.features = cols.features || [];
        this.outcomes = cols.outcomes || [];
        // Build metric→family lookup for <optgroup> rendering across all dropdowns.
        if (cols.feature_families?.length) {
          const lut = {};
          for (const grp of cols.feature_families) {
            for (const m of grp.metrics) {
              lut[m] = { family_num: grp.family_num, family_name: grp.family_name };
            }
          }
          this.metricFamilyLookup = lut;
        }
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
        this.outcome        = _pick;  // main analysis outcome (chart + downstream)
        // P2: Signal Survey outcome is now persisted via localStorage.
        // Fall back to _pick (ret_5d_fwd_oc) if no saved value or if the
        // saved one isn't in the discovered outcomes list (e.g., schema
        // change since last visit).
        try {
          const _saved = localStorage.getItem('factor-analysis.signalSurvey.outcome');
          this.surveyOutcome = (_saved && this.outcomes.includes(_saved)) ? _saved : _pick;
        } catch (_) {
          this.surveyOutcome = _pick;
        }
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
      // Signals list — needed by the portfolio signal-picker dropdown
      this.loadSignals();
      // Signal Survey — always-visible; load on init so charts appear without
      // requiring a click. Single-ticker auto-triggers a compute on cache miss;
      // ALL stays idle until explicit ⟳ Refresh (OOM guard).
      //
      // Bucket A step 5: the watchers that USED to live here ($watch on
      // this.outcome and this.pageMode, both clearing IC-batch state and
      // re-firing loadIcBatch) are removed. Signal Survey now owns its
      // own surveyMode + surveyCutoffDate; outcome lives in surveyOutcome
      // (which is independent of this.outcome). Mode-pill / outcome-
      // dropdown / cutoff-input changes are in-memory only — the user
      // clicks ⟳ Refresh to dispatch any network work. This means
      // changing the Analyze section's mode or promoting a lower-section
      // outcome no longer affects the Signal Survey pane.
      // v150: Re-fire trade table when bundle becomes ready — initial _renderTradeTable()
      // fires before the bundle is populated (bundle is async, ~50ms single ticker).
      // This $watch fires exactly once per analyze cycle (bundle goes null→object),
      // catching that "bundle now ready" event and running through the synchronous
      // bundle path — the same path manual bin/outcome changes use reliably.
      this.$watch('analyzeBundle', (val) => {
        if (val) this._renderTradeTable();
      });
      this.loadIcBatch();
      // Init-time breadcrumb metadata for the 4 panes that don't
      // auto-load on init (Corner Scan 2F/1F, All-Ticker Metric Bins,
      // Threshold Drift). Cheap metadata-only endpoints — no payload
      // transferred. Lets the collapsed-pane headers show "last: …"
      // immediately on page load instead of "no data yet".
      // Score Matrix + Signal Survey already do this (smInit /
      // loadIcBatch); this brings the other 4 into parity.
      this._loadInitMetadata();
    },

    async _loadInitMetadata() {
      // Defaults map to the same query string each pane uses on its
      // first ⟳ Refresh — see loadCs2f / loadTopBins / loadTd. If the
      // user reloads with non-default selections persisted in
      // localStorage (Signal Survey only today), those panes already
      // own their own metadata path.
      try {
        // Corner Scan 2F + 1F. The /corner-scan/meta response carries
        // BOTH 2F and 1F stats, so one fetch populates both breadcrumbs
        // (cs2fBreadcrumb + cs1fBreadcrumb both read this.cs2fMeta /
        // this.cs1fMeta respectively, and both default to walk_forward
        // mode at init).
        fetch('/api/factor-analysis/corner-scan/meta?mode=walk_forward')
          .then(r => r.ok ? r.json() : null)
          .then(d => { if (d) { this.cs2fMeta = d; this.cs1fMeta = d; if (d.metrics?.length) this.csMetrics = d.metrics; } })
          .catch(() => {});
        // All-Ticker Metric Bins. Default selectors mirror the first
        // ⟳ Refresh: ALL ticker, ret_5d_fwd_oc, 20 bins, walk_forward.
        fetch('/api/factor-analysis/global-metric-bins/meta'
              + '?ticker=ALL&outcome=ret_5d_fwd_oc&n_bins=20&walk_forward=1')
          .then(r => r.ok ? r.json() : null)
          .then(d => { if (d) this.topBinsMeta = d; })
          .catch(() => {});
        // Threshold Drift. Needs the default tdMetric (set after the
        // columns fetch above). Skip if columns aren't loaded yet or
        // tdMetric is empty — the breadcrumb will fall through to "no
        // data yet" in that case, same as before this commit.
        if (this.tdMetric) {
          const url = '/api/factor-analysis/threshold-drift/meta'
            + `?metric=${encodeURIComponent(this.tdMetric)}`
            + `&outcome=${encodeURIComponent(this.tdOutcome || 'ret_5d_fwd_oc')}`
            + `&ticker=ALL&n_bins=20&bins=1%2C20`;
          fetch(url)
            .then(r => r.ok ? r.json() : null)
            .then(d => { if (d) this.tdMeta = d; })
            .catch(() => {});
        }
      } catch (_) { /* non-fatal — breadcrumbs just stay "no data yet" */ }
    },

    async loadAnalysis() {
      if (!this.ticker || !this.metric || !this.outcome) return;
      this.loading = true;
      this.error = null;
      // v8 defensive reset: any in-flight deferred fetch / chart spinner
      // / in-flight promise tracker from a prior render must not bleed
      // across an Analyze cycle. Without this, the chart overlay could
      // stay "Loading data for the selected outcome…" until the stale
      // fetches resolve against the OLD cache_key (or never resolve, if
      // the server hangs them). Stale _outcomesInFlight entries could
      // also dedupe a freshly-needed fetch back to a promise that's
      // about to write the old metric's data into the new bundle. By
      // clearing all of this at the top of every Analyze, we ensure the
      // page can never wedge on the cross-Analyze handoff.
      this._deferredLoading   = false;
      this._tradeMetaInFlight = null;
      this._outcomesInFlight  = {};
      // User directive: every Analyze click resets the lower-section
      // view to Single mode at the default outcome (ret_5d_fwd_oc),
      // regardless of what mode/outcome was active when clicked.
      // Analyze is a "start fresh from the top" action and Single +
      // default is the natural landing spot. This also eliminates the
      // fragile cross-Analyze mode-restoration logic — the path that
      // caused the Gap-bars-don't-render bug and the null-deref wedge.
      // If the user wants Gap mode (or a promoted outcome, or Entry /
      // Horizon mode) for the newly-analyzed metric, they click into
      // it AFTER Analyze. decileEntryHorizon / decileHorizonAnchor /
      // decileBins / selectedBins20 stay preserved — those are sub-
      // control state, not the active view itself.
      this.decileMode          = 'single';
      this.outcome             = 'ret_5d_fwd_oc';
      this.decileActiveOutcome = 'ret_5d_fwd_oc';
      this.heatmapData = null;
      this.hmXData = null;
      this.hmYData = null;
      // Defensive wrap: a failure inside chart teardown must not crash
      // loadAnalysis partway through. Pre-v8 _destroyCharts had no
      // null-guard and threw TypeError on a null chart slot (Gap mode's
      // rolling-IC slot is the canonical case), wedging the entire
      // Analyze cycle before any network request fired. _destroyCharts
      // is now null-guarded internally, but the outer try here is the
      // last-resort safety net so any future teardown edge case can
      // only log and continue, never wedge the page.
      try { this._destroyCharts(); }
      catch (e) { console.warn('[loadAnalysis] _destroyCharts crashed, continuing:', e); }
      // M2: scanner persists across primary changes — stale-but-visible, not wiped.
      // Context key = ticker|metric|outcome|dates (NO mode — scanner is WF-locked).
      // Mode-only toggle (WF↔TT same context): leave scanner completely untouched.
      // Context change (new metric/ticker/outcome/dates): keep results, mark stale.
      {
        const _curCtx  = `${this.ticker}|${this.metric}|${this.outcome}|${this.dateFrom||''}|${this.dateTo||''}`;
        const _scanCtx = this._secCacheParams
          ? `${this._secCacheParams.ticker}|${this._secCacheParams.metric}|${this._secCacheParams.outcome}|${this._secCacheParams.dateFrom}|${this._secCacheParams.dateTo}`
          : null;
        const _ctxChanged = (_scanCtx !== null) && (_curCtx !== _scanCtx);
        if (_ctxChanged) {
          // Row cache + scan job are for the old context — invalidate them.
          // Scanner results (secStatus/secMetrics/secBaseline) stay — just marked stale.
          this.secCacheKey = null;
          this.secScanKey  = null;
          this.secPolling  = false;
          if (this.secStatus.loaded) this.secScannerStale = true;
        }
        // Mode-only (_ctxChanged === false): touch nothing — scanner and row cache both valid.
      }
      // secSelectedMetric and secDetail intentionally NOT cleared — persists across
      // primary-side changes. Auto-drill logic (M2) recomputes confirmation layer.
      this.secBinCount = 10;
      this.secSelectedSecBins = [10];
      this.secBubbleMinN = 1;
      this.corrPanelOpen = false;
      this.corrMiniData = null;
      this.corrMiniComputedBins = null;
      this.corrSelections = {};
      this.corrResult = null;
      try {
        let url = `/api/factor-analysis/analyze?ticker=${encodeURIComponent(this.ticker)}`
          + `&metric=${encodeURIComponent(this.metric)}`
          + `&outcome=${encodeURIComponent(this.outcome)}`;
        if (this.dateFrom) url += `&date_from=${this.dateFrom}`;
        if (this.dateTo) url += `&date_to=${this.dateTo}`;
        if (this.pageMode === 'walk_forward') url += '&walk_forward=true';
        if (this.pageMode === 'train_test')   url += `&cutoff_date=${encodeURIComponent(this.cutoffDate)}`;
        // Client-side timing: measures transmission + JSON-parse cost (diag, mirrors server _tlog).
        const _ct0 = performance.now();
        const r = await fetch(url);
        const _ct1 = performance.now();
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        this.data = await r.json();
        const _ct2 = performance.now();
        if (this.data.error) {
          this.error = this.data.error;
          // Clear stale bundle from the previous metric so it can't leak
          // into the next Analyze call. Without this, the prior good
          // metric's bundle lingers in memory when /analyze returns an
          // error (e.g. null-by-design metric → "Insufficient data"), and
          // a subsequent switch back to the good metric briefly shows stale
          // data before the fresh bundle arrives.
          this.analyzeBundle    = null;
          this.analyzeBundleKey = null;
          return;
        }
        // P4: snapshot the outcome-tied fields of the /analyze response (always
        // ret_5d_fwd_oc) so we can restore them when the user picks
        // ret_5d_fwd_oc as active again after a bundle-driven swap. We keep
        // boxplot returns + yearly_consistency + min_val/max_val that the
        // bundle slice doesn't carry. Snap each fresh fetch — replaces any
        // prior snapshot from a prior (ticker, metric, mode) selection.
        this._originalAnalyzeData = {
          trade_calendar:    this.data.trade_calendar,
          decile_stats:      this.data.decile_stats,
          decile_stats_20:   this.data.decile_stats_20,
          equity_by_decile:  this.data.equity_by_decile,
          rolling_ic:        this.data.rolling_ic,
          yearly_stats:      this.data.yearly_stats,
          dow_stats:         this.data.dow_stats,
          activity_by_date:  this.data.activity_by_date,
          horizon:           this.data.horizon,
        };
        // Default-active outcome resets to the /analyze outcome on every fresh fetch.
        this.decileActiveOutcome = this.outcome;
        // Recompute bar chart data for current mode with fresh decile_stats_20.
        this.decileBinsData = this.decileBins !== 10 ? this._computeDecileNBins(this.decileBins) : null;
        await this.$nextTick();
        await this.$nextTick();
        setTimeout(() => {
          this._renderCharts();
          const _ct3 = performance.now();
          // _handler_ms: time inside the Python handler (measured at return stmt).
          // serialize_est: (server+1stbyte) - handler — the FastAPI JSON-serialize gap.
          const _handlerS  = (this.data._handler_ms || 0) / 1000;
          const _serEst    = Math.max(0, (_ct1 - _ct0) / 1000 - _handlerS).toFixed(2);
          console.log(
            `[loadAnalysis] handler: ${_handlerS.toFixed(2)}s` +
            `  serialize_est: ${_serEst}s` +
            `  body+parse: ${((_ct2-_ct1)/1000).toFixed(2)}s` +
            `  render: ${((_ct3-_ct2)/1000).toFixed(2)}s` +
            `  total: ${((_ct3-_ct0)/1000).toFixed(2)}s  [${this.ticker}|${this.pageMode}]`
          );
        }, 80);
        // Bucket A step 5: Signal Survey cascade removed. Survey is a
        // fully independent pane — it reads surveyMode / surveyOutcome
        // (not pageMode / outcome) and only its own ⟳ Refresh button
        // fetches new data. Clicking Analyze used to also re-load IC
        // Batch + IC Decomp when the cache key changed (e.g., ticker
        // changed). It no longer does. If the user changes the Analyze
        // ticker, the Survey continues to show the previously-loaded
        // ticker's data until the user clicks Refresh in the Survey pane.
        // M3: heatmap is now driven by secDrillMetric — no longer fires on analyze.
        // P1: fire-and-forget kick of the 12-outcome bundle so P3+ mode views
        // have it ready. Single-ticker computes inline (~2-5s); ALL goes
        // through the background-job + poll path (~5 min on cache miss).
        // The bundle key is reset here too so a stale bundle from the prior
        // (ticker, metric, mode, cutoff) doesn't bleed into the new view.
        this.analyzeBundle    = null;
        this.analyzeBundleKey = null;
        this.loadAnalyzeBundle();

        // M2: secondary confirmation layer auto-recompute after primary changes.
        // entry type + horizon are subsumed by outcome — no separate check needed.
        if (this.secSelectedMetric) {
          const newParams = {
            ticker:   this.ticker,
            metric:   this.metric,
            outcome:  this.outcome,
            dateFrom: this.dateFrom || '',
            dateTo:   this.dateTo   || '',
          };
          const cacheStillValid = this.secCacheKey && this._secCacheParams &&
            JSON.stringify(newParams) === JSON.stringify(this._secCacheParams);

          // Mark scanner stale when primary context changed (not on mode-only toggle —
          // scanner is WF-locked, so toggling WF↔TT doesn't invalidate scanner rankings).
          if (this.secStatus.loaded && !cacheStillValid) this.secScannerStale = true;

          if (cacheStillValid) {
            // Mode/bin change only — row cache still valid; recompute with live mode params.
            this.secDrillMetric(this.secSelectedMetric, false);
          } else {
            // Metric/ticker/outcome/dates changed — need row refetch for new primary context.
            this.secCacheKey = null;
            console.log(`[loadAnalysis] firing _prepareSecRowsThenDrill at ${performance.now().toFixed(0)}ms`);
            this._prepareSecRowsThenDrill();
          }
          this._secCacheParams = newParams;
        }
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
      this._onDecileChangeLight();
    },

    // isDecileSelected: used by D1–D10 buttons and the decile stats table row highlight.
    isDecileSelected(d) {
      return this.selectedBins20.has(d * 2 - 1) || this.selectedBins20.has(d * 2);
    },

    selectAllDeciles() {
      this.selectedBins20 = new Set(Array.from({length: 20}, (_, i) => i + 1));
      this._onDecileChangeLight();
    },
    selectExtremes() {
      const g = 20 / this.decileBins;
      const lo = Array.from({length: g}, (_, i) => i + 1);
      const hi = Array.from({length: g}, (_, i) => 21 - g + i);
      this.selectedBins20 = new Set([...lo, ...hi]);
      this._onDecileChangeLight();
    },
    selectNone() {
      this.selectedBins20 = new Set();
      this._onDecileChangeLight();
    },

    _onDecileChange() {
      this._renderDecileBar();
      this._renderEquity();
      this._renderDrawdown();
      this._renderYearly();
      this._renderRollingCorr();
      this._renderReturnDist();
      this._renderDOW();
      this._renderActivity();
      this._renderTradeTable();
      // M2: advisory layer goes stale — user must explicitly reload scanner
      if (this.secStatus.loaded) this.secScannerStale = true;
      // M2: confirmation layer is live — recompute immediately if a secondary is selected
      if (this.secSelectedMetric && this.secCacheKey) {
        this.secDrillMetric(this.secSelectedMetric, false);
      }
      this._computeSelectedStats();
    },

    // ── Load-button helpers ───────────────────────────────────────────────
    // loadPrimary(): called by the primary Load button (and by mode/outcome
    // switches that should re-render immediately and clear any pending state).
    loadPrimary() {
      this.primaryPending = false;
      this._onDecileChange();
    },

    // _onDecileChangeLight(): called by bin-click / granularity-change paths
    // that are now gated behind the Load button. Only updates the bar
    // highlight and stats line; marks primaryPending so the Load button
    // becomes enabled.
    _onDecileChangeLight() {
      this._renderDecileBar();
      this._computeSelectedStats();
      if (this.secStatus.loaded) this.secScannerStale = true;
      this.primaryPending = true;
    },

    // loadSecondary(): called by the secondary Load button. Runs secScan
    // (only when bin count changed) and re-drills the selected metric.
    async loadSecondary() {
      this.secPending = false;
      if (this._secBinCountChanged) {
        this._secBinCountChanged = false;
        await this.secScan();
        if (this.corrPanelOpen && this.secCacheKey) await this.corrLoadMiniData();
      }
      if (this.secSelectedMetric) await this.secDrillMetric(this.secSelectedMetric, false);
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
        // sec-bar is WF-locked and mode-toggle-invariant — preserving it avoids
        // blanking the scanner lift bar on WF↔TT toggles. _applySecResults()
        // recreates it when scanner results actually change.
        // recall-* preserved alongside the other independent-section
        // charts (sm-*, port-*, ic-*, td, sec-bar). Without this guard
        // every primary /analyze sweep destroys the Recall view's
        // four Chart.js instances even though their canvases remain
        // in the DOM, leaving the recall section "blank" until you
        // re-click a signal. The recall canvases are managed entirely
        // by recallSignal / recallReanalyze / recallCancel — they have
        // no business being torn down by main-flow Analyze.
        if (k.startsWith('sm-') || k.startsWith('port-') || k === 'td' || k.startsWith('ic-') || k === 'sec-bar' || k.startsWith('recall-')) continue;
        // Null-guard each slot before calling .destroy(). Some render
        // functions explicitly null their slot when the chart has no
        // data — _renderRollingCorr does this at 'rolling' when
        // rolling_ic.series is empty, which is Gap mode's permanent
        // state (the gap synth never emits a rolling-IC series). Before
        // this guard, clicking Analyze while in Gap mode threw
        // TypeError: Cannot read properties of null (reading 'destroy')
        // and wedged the whole loadAnalysis call before any fetch fired.
        // try/catch on the .destroy() call itself is belt-and-suspenders
        // for any future Chart.js teardown edge case so one bad chart
        // can't stop the loop and leave the rest of the page half-destroyed.
        const ch = this._charts[k];
        if (ch) {
          try { ch.destroy(); }
          catch (e) { console.warn(`[_destroyCharts] destroy() failed for '${k}':`, e); }
        }
        delete this._charts[k];
      }
    },

    _darkScales() {
      return {
        x: { ticks:{color:'#888',font:{size:9},maxRotation:45}, grid:{color:'rgba(255,255,255,0.05)'}, border:{color:'transparent'} },
        y: { ticks:{color:'#888',font:{size:9}}, grid:{color:'rgba(255,255,255,0.05)'}, border:{color:'transparent'} },
      };
    },

    // Like _darkScales() but forces maxRotation:0 so Chart.js allocates only the
    // actual label height for the x-axis — prevents the blank gap that appears when
    // labels stay horizontal but Chart.js pre-reserved height for 45° rotation.
    _darkScalesNR() {
      const s = this._darkScales();
      return { ...s, x: { ...s.x, ticks: { ...s.x.ticks, maxRotation: 0 } } };
    },

    _renderCharts() {
      if (!this.data) return;
      this._renderAllCharts();
      this._computeSelectedStats();
    },

    // ── P3: lower-section chart rendering ─────────────────────────────────
    // Dispatch on decileMode. Single / Entry / Horizon read from the
    // analyze_cache bundle (analyzeBundle.per_bin[outcome]); Overnight Gap
    // computes from analyzeBundle.per_outcome_returns at render time.
    // Single mode falls back to this.data.decile_stats (the /analyze single-
    // outcome payload) when the bundle hasn't arrived yet — so the initial
    // ~25s /analyze view is always usable. Entry, Horizon, and Overnight Gap
    // require the bundle (controls are disabled in template until ready).

    _renderDecileBar() {
      const el = document.getElementById('chart-decile');
      if (!el) return;
      if (this._charts['decile']) this._charts['decile'].destroy();
      if (this.decileMode === 'overnight_gap') { this._renderDecileBarOvernightGap(el); return; }
      if (this.decileMode === 'entry')         { this._renderDecileBarEntry(el);        return; }
      if (this.decileMode === 'horizon')       { this._renderDecileBarHorizon(el);      return; }
      this._renderDecileBarSingle(el);
    },

    // Bundle's per_bin is 20-bin granularity only. This returns the
    // normalized 20-bin array (or null if not in bundle). Field names are
    // unified with the existing decile_stats shape downstream (std_dev,
    // sharpe-derived). The chart's 5/10-bin views are produced by
    // _aggregateBin20ToN below, run client-side.
    _bundlePerBin20(outcome) {
      const pb = this.analyzeBundle?.per_bin?.[outcome];
      if (!Array.isArray(pb)) return null;
      return pb.map(r => r ? {
        bucket:   r.bin,
        n:        r.n,
        avg_ret:  r.avg_ret,
        median:   r.median,
        win_rate: r.win_rate,
        std_dev:  r.std,
        sharpe:   (r.std && r.std > 0) ? (r.avg_ret / r.std) : null,
      } : null);
    },

    // Aggregate 20-bin stats array to a coarser n-bin view (n ∈ {5, 10, 20}).
    // avg_ret and win_rate are trade-count-weighted; n is summed; median
    // and std_dev are *not* derivable from per-bin stats so they're null
    // at coarser granularities (the bundle keeps them at 20-bin only, per
    // the user directive).
    _aggregateBin20ToN(perBin20, n) {
      if (!perBin20 || !perBin20.length) return [];
      if (n === 20) return perBin20;
      const g = 20 / n;
      const out = [];
      for (let i = 0; i < n; i++) {
        const group = perBin20.slice(i * g, (i + 1) * g).filter(Boolean);
        if (!group.length) { out.push(null); continue; }
        const totalN = group.reduce((a, d) => a + d.n, 0);
        if (!totalN) { out.push(null); continue; }
        out.push({
          bucket:   i + 1,
          n:        totalN,
          avg_ret:  group.reduce((a, d) => a + d.avg_ret * d.n, 0) / totalN,
          win_rate: group.reduce((a, d) => a + d.win_rate * d.n, 0) / totalN,
          median:   null,    // not derivable from per-bin stats
          std_dev:  null,
          sharpe:   null,
        });
      }
      return out;
    },

    // Bundle-backed stats for an outcome at the current display granularity.
    _bundlePerBin(outcome) {
      const pb20 = this._bundlePerBin20(outcome);
      if (!pb20) return null;
      return this._aggregateBin20ToN(pb20, this.decileBins);
    },

    // Stats for the *active* outcome at the current display granularity.
    // Prefers the bundle; falls back to this.data.decile_stats* (which
    // /analyze populates at 10-bin AND 20-bin) when the active outcome is
    // the /analyze-driven default (ret_5d_fwd_oc) and the bundle hasn't
    // arrived yet.
    _activeOutcomeStats() {
      const pb20 = this._bundlePerBin20(this.decileActiveOutcome);
      if (pb20) return this._aggregateBin20ToN(pb20, this.decileBins).filter(Boolean);
      if (this.decileActiveOutcome !== 'ret_5d_fwd_oc') return [];
      // Fall back to /analyze's decile_stats. /analyze always emits
      // decile_stats (10-bin) and decile_stats_20 (20-bin), so 10 and 20
      // views are direct lookups; 5 aggregates from the 20-bin.
      if (this.decileBins === 10) return (this.data?.decile_stats || []).filter(Boolean);
      if (this.decileBins === 20) return (this.data?.decile_stats_20 || []).filter(Boolean);
      // 5-bin: aggregate the legacy 20-bin payload through the same path
      const ds20 = (this.data?.decile_stats_20 || []).map(d => d ? {
        bucket: d.bucket, n: d.n,
        avg_ret: d.avg_ret, win_rate: d.win_rate,
        median: d.med_ret, std_dev: d.std_dev,
        sharpe: d.sharpe,
      } : null);
      return this._aggregateBin20ToN(ds20, 5).filter(Boolean);
    },

    _renderDecileBarSingle(el) {
      const stats = this._activeOutcomeStats();
      if (!stats.length) return;
      const avgs = stats.map(d => d.avg_ret * 100);
      const self = this;
      // selectedBins20 is the canonical 20-bin set. Display bucket k at
      // current granularity decileBins maps to bins {(k-1)g+1..kg} where
      // g = 20/decileBins. Selection appears "on" if ANY 20-bin member of
      // the display bucket is in selectedBins20.
      const _isSel = (d) => self._displayBucketIsSelected(d.bucket);

      this._charts['decile'] = new Chart(el, {
        type: 'bar',
        data: {
          labels: stats.map(d => 'B' + d.bucket),
          datasets: [{
            data: avgs,
            backgroundColor: stats.map(d => _isSel(d)
              ? (d.avg_ret >= 0 ? '#3498db' : '#e84393') : 'rgba(100,100,100,0.3)'),
            borderColor:     stats.map(d => _isSel(d) ? '#fff' : 'transparent'),
            borderWidth:     stats.map(d => _isSel(d) ? 1 : 0),
          }],
        },
        options: {
          responsive: true, maintainAspectRatio: false, animation: false,
          layout: { padding: { bottom: 0 } },
          onClick: (e, elements) => {
            if (!elements.length) return;
            const d = stats[elements[0].index];
            self._clickBarSingle(d.bucket);
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
                    d.sharpe != null ? `Sharpe: ${d.sharpe.toFixed(3)}` : '',
                    `n: ${d.n}`,
                    d.min_val != null ? `Range: ${d.min_val.toFixed(4)} – ${d.max_val.toFixed(4)}` : '',
                  ].filter(Boolean);
                },
              },
            },
          },
          scales: this._darkScalesNR(),
        },
      });
    },

    // Display bucket k (1..decileBins) is "selected" if ANY 20-bin member
    // of the bucket is in selectedBins20. Used by every chart renderer.
    _displayBucketIsSelected(bucket) {
      const g = 20 / this.decileBins;
      const lo = (bucket - 1) * g + 1;
      for (let b = lo; b < lo + g; b++) if (this.selectedBins20.has(b)) return true;
      return false;
    },

    _hexToRgba(hex, alpha) {
      const r = parseInt(hex.slice(1, 3), 16);
      const g = parseInt(hex.slice(3, 5), 16);
      const b = parseInt(hex.slice(5, 7), 16);
      return `rgba(${r},${g},${b},${alpha.toFixed(2)})`;
    },

    _outcomeAnchor(outcome) {
      const m = (outcome || this.decileActiveOutcome || '').match(/_(oc|cc)$/i);
      return m ? m[1].toLowerCase() : 'oc';
    },

    _outcomeHorizon(outcome) {
      const m = (outcome || this.decileActiveOutcome || '').match(/ret_(\d+)d_fwd/i);
      return m ? +m[1] : null;
    },

    // Entry mode: at a chosen horizon, compare OC vs CC. 20 bars (10 bins × 2 anchors).
    _renderDecileBarEntry(el) {
      const h = this.decileEntryHorizon;
      const ocStats = this._bundlePerBin(`ret_${h}d_fwd_oc`) || [];
      const ccStats = this._bundlePerBin(`ret_${h}d_fwd_cc`) || [];
      if (!ocStats.length && !ccStats.length) return;
      const primaryStats = ocStats.length ? ocStats : ccStats;
      const self  = this;
      const labels = primaryStats.map(d => d ? 'B' + d.bucket : '');

      // Active outcome gets the brighter fill + star marker; the other is dim.
      const activeAnchor = this._outcomeAnchor(this.decileActiveOutcome);
      const ocPrim = activeAnchor === 'oc';
      // Selection highlight is anchored to the active (primary) dataset only.
      // Bars in the non-primary dataset never reflect selection — clicking a
      // primary bar must not also light up the sibling bar in the cluster.
      const isSelOC = (d) => ocPrim && d && self._displayBucketIsSelected(d.bucket);
      const isSelCC = (d) => !ocPrim && d && self._displayBucketIsSelected(d.bucket);

      const datasets = [
        {
          label: `OC (${h}d)${ocPrim ? ' ★' : ''}`,
          data:  ocStats.map(d => d ? d.avg_ret * 100 : null),
          backgroundColor: ocStats.map(d => {
            if (!d) return 'transparent';
            const a = ocPrim ? (isSelOC(d) ? 0.92 : 0.50) : 0.26;
            return d.avg_ret >= 0 ? `rgba(52,152,219,${a})` : `rgba(232,67,147,${a})`;
          }),
          borderWidth: ocPrim ? ocStats.map(d => isSelOC(d) ? 1 : 0) : 0,
          borderColor: '#fff',
        },
        {
          label: `CC (${h}d)${!ocPrim ? ' ★' : ''}`,
          data:  ccStats.map(d => d ? d.avg_ret * 100 : null),
          backgroundColor: ccStats.map(d => {
            if (!d) return 'transparent';
            const a = !ocPrim ? (isSelCC(d) ? 0.92 : 0.50) : 0.26;
            return d.avg_ret >= 0 ? `rgba(46,204,113,${a})` : `rgba(230,126,34,${a})`;
          }),
          borderWidth: !ocPrim ? ccStats.map(d => isSelCC(d) ? 1 : 0) : 0,
          borderColor: '#fff',
        },
      ];

      this._charts['decile'] = new Chart(el, {
        type: 'bar',
        data: { labels, datasets },
        options: {
          responsive: true, maintainAspectRatio: false, animation: false,
          layout: { padding: { bottom: 0 } },
          onClick: (e, elements) => {
            if (!elements.length) return;
            const el0 = elements[0];
            const d = primaryStats[el0.index];
            if (d) self._clickBarEntry(d.bucket, el0.datasetIndex);
          },
          plugins: {
            legend: { display: true, labels: { color: '#888', font: { size: 10 }, boxWidth: 10, padding: 8 } },
            tooltip: {
              backgroundColor: 'rgba(20,20,20,0.95)', borderColor: '#444', borderWidth: 1,
              callbacks: {
                label: ctx => {
                  const d = ctx.datasetIndex === 0 ? ocStats[ctx.dataIndex]
                                                   : ccStats[ctx.dataIndex];
                  if (!d) return `${ctx.dataset.label.replace(' ★','')}: —`;
                  return [
                    `${ctx.dataset.label.replace(' ★','')}`,
                    `Avg: ${(d.avg_ret*100).toFixed(3)}%`,
                    `WR: ${(d.win_rate*100).toFixed(1)}%`,
                    d.sharpe != null ? `Sharpe: ${d.sharpe.toFixed(3)}` : '',
                    `n: ${d.n}`,
                  ].filter(Boolean);
                },
              },
            },
          },
          scales: {
            ...this._darkScalesNR(),
            y: { ...this._darkScales().y, ticks: { ...this._darkScales().y.ticks, callback: v => v.toFixed(2)+'%' } },
          },
        },
      });
    },

    // Horizon mode: at a chosen anchor, show all 6 horizons. 60 bars (10 × 6).
    _renderDecileBarHorizon(el) {
      const a = this.decileHorizonAnchor;
      const series = HORIZON_LIST.map(h => this._bundlePerBin(`ret_${h}d_fwd_${a}`) || []);
      const primaryStats = series.find(s => s.length) || [];
      if (!primaryStats.length) return;
      const self  = this;
      const labels = primaryStats.map(d => d ? 'B' + d.bucket : '');

      // Active outcome gets the star marker (when the active outcome matches
      // anchor + one of the 6 horizons in this group).
      const activeH = this._outcomeHorizon(this.decileActiveOutcome);
      const activeAnchor = this._outcomeAnchor(this.decileActiveOutcome);

      const datasets = HORIZON_LIST.map((h, i) => {
        const stats = series[i];
        const isPrimary = (h === activeH) && (a === activeAnchor);
        const col = HORIZON_PALETTE[i];
        // Selection highlight is anchored to the primary (active) dataset.
        // Non-primary horizons stay at their dim baseline so a click on the
        // primary's bucket B3 doesn't also brighten the other 5 horizons' B3.
        const isSel = (d) => isPrimary && d && self._displayBucketIsSelected(d.bucket);
        return {
          label: `${h}d${isPrimary ? ' ★' : ''}`,
          data:  stats.map(d => d ? d.avg_ret * 100 : null),
          backgroundColor: stats.map(d => {
            if (!d) return 'transparent';
            return self._hexToRgba(col,
              isPrimary ? (isSel(d) ? 0.95 : 0.50) : 0.28);
          }),
          borderWidth: isPrimary ? stats.map(d => isSel(d) ? 1 : 0) : 0,
          borderColor: col,
        };
      });

      this._charts['decile'] = new Chart(el, {
        type: 'bar',
        data: { labels, datasets },
        options: {
          responsive: true, maintainAspectRatio: false, animation: false,
          layout: { padding: { bottom: 0 } },
          onClick: (e, elements) => {
            if (!elements.length) return;
            const el0 = elements[0];
            const d = primaryStats[el0.index];
            if (d) self._clickBarHorizon(d.bucket, el0.datasetIndex);
          },
          plugins: {
            legend: { display: true, labels: { color: '#888', font: { size: 10 }, boxWidth: 10, padding: 8 } },
            tooltip: {
              backgroundColor: 'rgba(20,20,20,0.95)', borderColor: '#444', borderWidth: 1,
              callbacks: {
                label: ctx => {
                  const d = series[ctx.datasetIndex]?.[ctx.dataIndex];
                  if (!d) return `${ctx.dataset.label.replace(' ★','')}: —`;
                  return [
                    `${ctx.dataset.label.replace(' ★','')}`,
                    `Avg: ${(d.avg_ret*100).toFixed(3)}%`,
                    `WR: ${(d.win_rate*100).toFixed(1)}%`,
                    d.sharpe != null ? `Sharpe: ${d.sharpe.toFixed(3)}` : '',
                    `n: ${d.n}`,
                  ].filter(Boolean);
                },
              },
            },
          },
          scales: {
            ...this._darkScalesNR(),
            y: { ...this._darkScales().y, ticks: { ...this._darkScales().y.ticks, callback: v => v.toFixed(2)+'%' } },
          },
        },
      });
    },

    // Overnight Gap per-bin stats (20-bin) for the Quantile bar chart.
    // v7: read straight from the precomputed per_bin["overnight_gap"]
    // already in the slim payload — no fetch, no synthesis, no
    // trade_meta needed. The renderer aggregates to current display
    // granularity via _aggregateBin20ToN. v6 computed gap at render
    // time by zipping ret_1d_fwd_cc + ret_1d_fwd_oc + trade_meta (~56
    // MB transfer + ~1s client synthesis); v7 serves the same 20-row
    // result from the slim payload in microseconds.
    _overnightGapPerBin20() {
      const pb = this.analyzeBundle?.per_bin?.overnight_gap;
      if (!Array.isArray(pb)) return null;
      return pb.map(r => r ? {
        bucket:   r.bin,
        n:        r.n,
        avg_ret:  r.avg_ret,
        win_rate: r.win_rate,
        std_dev:  r.std,
        sharpe:   (r.std && r.std > 0) ? (r.avg_ret / r.std) : null,
      } : null);
    },

    _renderDecileBarOvernightGap(el) {
      const stats20 = this._overnightGapPerBin20();
      if (!stats20) return;
      const stats = this._aggregateBin20ToN(stats20, this.decileBins).filter(Boolean);
      if (!stats.length) return;
      const avgs = stats.map(d => d.avg_ret * 100);
      const self = this;
      const _isSel = (d) => self._displayBucketIsSelected(d.bucket);

      this._charts['decile'] = new Chart(el, {
        type: 'bar',
        data: {
          labels: stats.map(d => 'B' + d.bucket),
          datasets: [{
            data: avgs,
            backgroundColor: stats.map(d => _isSel(d)
              ? (d.avg_ret >= 0 ? '#9b59b6' : '#e74c3c')   // gap palette differs from Single
              : 'rgba(155,89,182,0.30)'),
            borderColor:     stats.map(d => _isSel(d) ? '#fff' : 'transparent'),
            borderWidth:     stats.map(d => _isSel(d) ? 1 : 0),
          }],
        },
        options: {
          responsive: true, maintainAspectRatio: false, animation: false,
          layout: { padding: { bottom: 0 } },
          onClick: (e, elements) => {
            if (!elements.length) return;
            const d = stats[elements[0].index];
            self._clickBarOvernightGap(d.bucket);
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
                    `Gap: ${(d.avg_ret*100).toFixed(3)}%`,
                    `WR: ${(d.win_rate*100).toFixed(1)}%`,
                    d.sharpe != null ? `Sharpe: ${d.sharpe.toFixed(3)}` : '',
                    `n: ${d.n}`,
                  ].filter(Boolean);
                },
              },
            },
          },
          scales: this._darkScalesNR(),
        },
      });
    },

    // ── P4: click handlers + data swap ────────────────────────────────────
    // The decile bar is the canonical selector for the lower section. Each
    // mode has its own subset semantics:
    //   Single        → 1 subset (the current granularity bins). Click toggles.
    //   Entry         → 2 subsets at the active horizon: OC bins and CC bins.
    //                   Click within active subset = toggle. Click on the OTHER
    //                   subset = promote that outcome, single-select that bin.
    //   Horizon       → 6 subsets at the active anchor (1d/3d/5d/7d/10d/20d).
    //                   Same rule: same-horizon click = toggle; cross-horizon
    //                   click = promote + single-select.
    //   Overnight Gap → 1 subset. Click toggles.

    // Multi-select toggle for a display bucket (1..decileBins). The bucket
    // maps to g = 20/decileBins contiguous 20-bin indices; the toggle applies
    // to ALL members atomically (so the on/off state is consistent across
    // granularity boundaries).
    _toggleBucketInSelection(bucket) {
      const g = 20 / this.decileBins;
      const lo = (bucket - 1) * g + 1;
      const members = [];
      for (let b = lo; b < lo + g; b++) members.push(b);
      const allOn = members.every(b => this.selectedBins20.has(b));
      const next = new Set(this.selectedBins20);
      if (allOn) {
        for (const b of members) next.delete(b);
      } else {
        for (const b of members) next.add(b);
      }
      this.selectedBins20 = next;
      this._onDecileChangeLight();   // visual highlight only; Load button triggers full render
    },

    // Replace the selection with a single display bucket (used after a
    // cross-subset click triggers outcome promotion: clear prior selection,
    // single-select the just-clicked bucket).
    _setSingleBucket(bucket) {
      const g = 20 / this.decileBins;
      const lo = (bucket - 1) * g + 1;
      const next = new Set();
      for (let b = lo; b < lo + g; b++) next.add(b);
      this.selectedBins20 = next;
      this._onDecileChangeLight();   // visual highlight only; Load button triggers full render
    },

    // Promote the active outcome (Entry/Horizon cross-subset click, or the
    // Single-mode dropdown). Re-points downstream visuals via _swapData.
    // The caller is responsible for triggering re-render (typically
    // _setSingleBucket or _onDecileChange).
    async _promoteOutcome(newOutcome) {
      if (this.decileActiveOutcome === newOutcome) return;
      this.decileActiveOutcome = newOutcome;
      this.outcome = newOutcome;
      // v6: _swapDataForActiveOutcome lazy-loads trade_meta + outcome
      // data for non-default outcomes. Await so callers can rely on
      // this.data being fresh by the time we return — _clickBar* below
      // calls _setSingleBucket after this, which triggers the chart
      // re-renders that need to read the new this.data.
      await this._swapDataForActiveOutcome();
    },

    _clickBarSingle(bucket)                  { this._toggleBucketInSelection(bucket); },
    _clickBarOvernightGap(bucket)            { this._toggleBucketInSelection(bucket); },

    async _clickBarEntry(bucket, datasetIdx) {
      // Dataset 0 = OC, dataset 1 = CC.
      const clickedAnchor = (datasetIdx === 0) ? 'oc' : 'cc';
      const activeAnchor  = this._outcomeAnchor(this.decileActiveOutcome);
      if (clickedAnchor === activeAnchor) {
        this._toggleBucketInSelection(bucket);
      } else {
        await this._promoteOutcome(`ret_${this.decileEntryHorizon}d_fwd_${clickedAnchor}`);
        this._setSingleBucket(bucket);
      }
    },

    async _clickBarHorizon(bucket, datasetIdx) {
      // datasetIdx i ∈ [0, HORIZON_LIST.length-1] → HORIZON_LIST[i] horizon.
      const clickedH = HORIZON_LIST[datasetIdx];
      const activeH  = this._outcomeHorizon(this.decileActiveOutcome);
      if (clickedH === activeH) {
        this._toggleBucketInSelection(bucket);
      } else {
        await this._promoteOutcome(`ret_${clickedH}d_fwd_${this.decileHorizonAnchor}`);
        this._setSingleBucket(bucket);
      }
    },

    // Build the outcome-specific data slice from the bundle, matching the
    // shape that /analyze would emit if that outcome had been the primary.
    // Used to mutate this.data when active outcome changes. Returns null if
    // the bundle isn't loaded yet (caller leaves this.data untouched).
    _buildOutcomeDataSlice(outcome) {
      const b = this.analyzeBundle;
      if (!b) return null;
      const tm    = b.trade_meta;
      const ret   = b.per_outcome_returns?.[outcome];
      const perBin = b.per_bin?.[outcome];
      if (!Array.isArray(tm) || !ret || !Array.isArray(perBin)) return null;

      // trade_calendar: one row per trade where the outcome value is non-null.
      // Shape matches /analyze's slim trade_calendar (date, ret, ticker, decile20).
      // We also accumulate per-bin (10-bin) raw return arrays here so the
      // Return Distribution histogram can read decile_stats[i].returns
      // for swapped outcomes — without this it filters every bar out via
      // `d.returns?.length >= 5` and renders blank.
      const tc = [];
      const returnsByBin10 = Array.from({length: 10}, () => []);
      for (let i = 0; i < ret.trade_ids.length; i++) {
        const m = tm[ret.trade_ids[i]];
        if (!m) continue;
        const r = ret.ret_pcts[i];
        tc.push({
          date:     m.trade_date,
          ret:      r,
          ticker:   m.ticker,
          decile20: m.bin_20,
        });
        const bin10 = Math.ceil(m.bin_20 / 2);
        if (bin10 >= 1 && bin10 <= 10) returnsByBin10[bin10 - 1].push(r);
      }

      // decile_stats_20 = per_bin (already 20-bin); decile_stats (10-bin)
      // aggregates pairs (trade-count-weighted). median/std/sharpe at the
      // 10-bin view aren't recoverable from per-bin summary so they're
      // null — same constraint as the bar-chart's _aggregateBin20ToN.
      const ds20 = perBin.map(r => (r && r.n) ? {
        bucket:   r.bin,
        n:        r.n,
        avg_ret:  r.avg_ret,
        med_ret:  r.median,
        win_rate: r.win_rate,
        std_dev:  r.std,
        sharpe:   (r.std && r.std > 0) ? (r.avg_ret / r.std) : 0,
        min_val:  null,
        max_val:  null,
        returns:  [],     // bundle doesn't carry per-trade returns; boxplot will skip
      } : null);
      const ds10 = [];
      for (let i = 0; i < 10; i++) {
        const pair = [ds20[i*2], ds20[i*2 + 1]].filter(Boolean);
        if (!pair.length) { ds10.push(null); continue; }
        const totalN = pair.reduce((a, d) => a + d.n, 0);
        if (!totalN) { ds10.push(null); continue; }
        ds10.push({
          bucket:   i + 1,
          n:        totalN,
          avg_ret:  pair.reduce((a, d) => a + d.avg_ret  * d.n, 0) / totalN,
          med_ret:  null,
          win_rate: pair.reduce((a, d) => a + d.win_rate * d.n, 0) / totalN,
          std_dev:  null,
          sharpe:   null,
          min_val:  null,
          max_val:  null,
          returns:  returnsByBin10[i],
        });
      }

      // Yearly / DoW / Activity aggregates derived from trade_calendar. Same
      // (year, decile20) / (dow, decile20) / (date, decile20) keys as the
      // server-side stats so downstream renderers don't change.
      const yrAcc  = new Map();
      const dowAcc = new Map();
      const actAcc = new Map();
      for (const t of tc) {
        const yr = +t.date.slice(0, 4);
        // ISO date → Mon=1..Sun=7 via JS Date getDay (Sun=0..Sat=6). The
        // server uses Python weekday() (Mon=0..Sun=6) — match that.
        const jd = new Date(t.date + 'T12:00:00Z').getUTCDay();
        const dow = (jd + 6) % 7;      // shift Sun=0..Sat=6 → Mon=0..Sun=6
        const yk = `${yr}|${t.decile20}`;
        if (!yrAcc.has(yk)) yrAcc.set(yk, {year: yr, decile20: t.decile20, sum: 0, wins: 0, n: 0});
        const ya = yrAcc.get(yk);
        ya.sum += t.ret; ya.wins += (t.ret > 0 ? 1 : 0); ya.n++;
        if (dow <= 4) {
          const dk = `${dow}|${t.decile20}`;
          if (!dowAcc.has(dk)) dowAcc.set(dk, {dow, decile20: t.decile20, sum: 0, wins: 0, n: 0});
          const da = dowAcc.get(dk);
          da.sum += t.ret; da.wins += (t.ret > 0 ? 1 : 0); da.n++;
        }
        const ak = `${t.date}|${t.decile20}`;
        if (!actAcc.has(ak)) actAcc.set(ak, {date: t.date, decile20: t.decile20, n: 0});
        actAcc.get(ak).n++;
      }
      const yearly_stats = Array.from(yrAcc.values()).map(a => ({
        year: a.year, decile20: a.decile20,
        n: a.n, avg_ret: a.sum / a.n, win_rate: a.wins / a.n,
      })).sort((a, b) => a.year - b.year || a.decile20 - b.decile20);
      const dow_stats = Array.from(dowAcc.values()).map(a => ({
        dow: a.dow, decile20: a.decile20,
        n: a.n, avg_ret: a.sum / a.n, win_rate: a.wins / a.n,
      })).sort((a, b) => a.dow - b.dow || a.decile20 - b.decile20);
      const activity_by_date = Array.from(actAcc.values())
        .sort((a, b) => a.date.localeCompare(b.date) || a.decile20 - b.decile20);

      // rolling_ic: /analyze emits {series, reference_ic, epsilon, cutoff_date,
      // short_series}. Bundle stores only the classified series. We wrap it
      // and compute reference_ic = mean(ic) as the backend does in non-
      // train_test mode. epsilon (noise-floor band) and short_series (5d
      // context line) aren't in the bundle — they're omitted, so the band
      // collapses to zero for swapped outcomes. Bundle schema would need
      // a bump to carry them; deferred.
      const icSeries = b.rolling_ic?.[outcome] || [];
      const icVals = icSeries.map(p => p.ic).filter(v => v != null);
      const refIc = icVals.length ? icVals.reduce((a, v) => a + v, 0) / icVals.length : 0;

      return {
        trade_calendar:    tc,
        decile_stats:      ds10,
        decile_stats_20:   ds20,
        rolling_ic: {
          series:        icSeries,
          reference_ic:  refIc,
          epsilon:       0,
          cutoff_date:   null,
          short_series:  [],
        },
        yearly_stats,
        dow_stats,
        activity_by_date,
        horizon:           this._outcomeHorizon(outcome) || 1,
      };
    },

    // P6: synthesized data slice for Overnight Gap mode. Each trade's
    // "return" is the per-trade gap = ret_1d_fwd_cc − ret_1d_fwd_oc.
    // Trades are intersected on trade_id between the two columns so
    // every row has both legs. Bin assignment comes from trade_meta
    // (anchor-outcome filtered, n_bins=20) — same as the main Quantile
    // pane in Gap mode and the X-sidebar after the P5 fixup, so all
    // three agree exactly.
    //
    // Rolling IC is intentionally empty: the bundle stores IC per real
    // outcome, not for the synthetic gap. Computing it client-side
    // means a 252-day rolling Spearman over ~150K trades (~minutes in
    // JS without a worker). The Rolling Signal Strength chart will
    // render blank in Gap mode until a bundle schema bump adds it.
    _buildGapDataSlice() {
      // v7: read straight from the precomputed overnight_gap outcome.
      // Server computed gap = ret_1d_fwd_cc − ret_1d_fwd_oc per trade
      // and shipped it as 4 inline parallel arrays (ret_pcts,
      // trade_dates, tickers, bin_20s) plus per_bin["overnight_gap"]
      // (already in slim). No client-side synthesis, no trade_meta
      // lookup. Drops Gap-mode entry from ~17s to ~6-11s.
      const b = this.analyzeBundle;
      if (!b) return null;
      const gap = b.per_outcome_returns?.overnight_gap;
      const gapPerBin = b.per_bin?.overnight_gap;
      if (!gap || !Array.isArray(gap.trade_dates) || !Array.isArray(gapPerBin)) return null;

      const tc = [];
      const returnsByBin10 = Array.from({length: 10}, () => []);
      for (let i = 0; i < gap.trade_dates.length; i++) {
        const ret   = gap.ret_pcts[i];
        const bin20 = gap.bin_20s[i];
        tc.push({
          date:     gap.trade_dates[i],
          ret:      ret,
          ticker:   gap.tickers[i],
          decile20: bin20,
        });
        const bin10 = Math.ceil(bin20 / 2);
        if (bin10 >= 1 && bin10 <= 10) returnsByBin10[bin10 - 1].push(ret);
      }

      // Per-bin stats (20-bin) from the precomputed server aggregate.
      // Shape mirrors what _buildOutcomeDataSlice produces for real
      // outcomes so downstream renderers (decile bar, return dist,
      // boxplot) consume it uniformly.
      const ds20 = gapPerBin.map(r => (r && r.n) ? {
        bucket:   r.bin,
        n:        r.n,
        avg_ret:  r.avg_ret,
        med_ret:  r.median,
        win_rate: r.win_rate,
        std_dev:  r.std,
        sharpe:   (r.std && r.std > 0) ? (r.avg_ret / r.std) : 0,
        min_val:  null,
        max_val:  null,
        returns:  [],
      } : null);

      // 10-bin aggregation (trade-count-weighted), with per-bin returns
      // arrays for the Return Distribution histogram.
      const ds10 = [];
      for (let i = 0; i < 10; i++) {
        const pair = [ds20[i*2], ds20[i*2 + 1]].filter(Boolean);
        if (!pair.length) { ds10.push(null); continue; }
        const totalN = pair.reduce((a, d) => a + d.n, 0);
        if (!totalN) { ds10.push(null); continue; }
        ds10.push({
          bucket:   i + 1,
          n:        totalN,
          avg_ret:  pair.reduce((a, d) => a + d.avg_ret  * d.n, 0) / totalN,
          med_ret:  null,
          win_rate: pair.reduce((a, d) => a + d.win_rate * d.n, 0) / totalN,
          std_dev:  null,
          sharpe:   null,
          min_val:  null,
          max_val:  null,
          returns:  returnsByBin10[i],
        });
      }

      // Yearly / DoW / Activity — same aggregation shape as the real-
      // outcome branch in _buildOutcomeDataSlice.
      const yrAcc  = new Map();
      const dowAcc = new Map();
      const actAcc = new Map();
      for (const t of tc) {
        const yr = +t.date.slice(0, 4);
        const jd = new Date(t.date + 'T12:00:00Z').getUTCDay();
        const dow = (jd + 6) % 7;
        const yk = `${yr}|${t.decile20}`;
        if (!yrAcc.has(yk)) yrAcc.set(yk, {year: yr, decile20: t.decile20, sum: 0, wins: 0, n: 0});
        const ya = yrAcc.get(yk);
        ya.sum += t.ret; ya.wins += (t.ret > 0 ? 1 : 0); ya.n++;
        if (dow <= 4) {
          const dk = `${dow}|${t.decile20}`;
          if (!dowAcc.has(dk)) dowAcc.set(dk, {dow, decile20: t.decile20, sum: 0, wins: 0, n: 0});
          const da = dowAcc.get(dk);
          da.sum += t.ret; da.wins += (t.ret > 0 ? 1 : 0); da.n++;
        }
        const ak = `${t.date}|${t.decile20}`;
        if (!actAcc.has(ak)) actAcc.set(ak, {date: t.date, decile20: t.decile20, n: 0});
        actAcc.get(ak).n++;
      }
      const yearly_stats = Array.from(yrAcc.values()).map(a => ({
        year: a.year, decile20: a.decile20,
        n: a.n, avg_ret: a.sum / a.n, win_rate: a.wins / a.n,
      })).sort((a, b) => a.year - b.year || a.decile20 - b.decile20);
      const dow_stats = Array.from(dowAcc.values()).map(a => ({
        dow: a.dow, decile20: a.decile20,
        n: a.n, avg_ret: a.sum / a.n, win_rate: a.wins / a.n,
      })).sort((a, b) => a.dow - b.dow || a.decile20 - b.decile20);
      const activity_by_date = Array.from(actAcc.values())
        .sort((a, b) => a.date.localeCompare(b.date) || a.decile20 - b.decile20);

      return {
        trade_calendar:    tc,
        decile_stats:      ds10,
        decile_stats_20:   ds20,
        rolling_ic: {
          series: [], reference_ic: 0, epsilon: 0, cutoff_date: null, short_series: [],
        },
        yearly_stats,
        dow_stats,
        activity_by_date,
        horizon: 1,
      };
    },

    // Rebuild equity_by_decile (10-bin) from a trade_calendar. Reads
    // this.data.horizon for non_overlapping spacing — caller must set
    // this.data.horizon to the new outcome's horizon FIRST.
    _buildEquityByDecileFromCal(cal) {
      const out = {};
      for (let d = 1; d <= 10; d++) {
        const binCal = cal.filter(c => Math.ceil(c.decile20 / 2) === d);
        out[d] = {
          concurrent:      this._getEquityCurveFromCal(binCal, 'concurrent'),
          non_overlapping: this._getEquityCurveFromCal(binCal, 'non_overlapping'),
        };
      }
      return out;
    },

    // Mutate this.data so downstream renderers (equity, drawdown, rolling
    // IC, yearly, dow, activity, decile stats, trade table) reflect the
    // current active outcome. Three branches:
    //   1. ret_5d_fwd_oc + _originalAnalyzeData snapshot → restore from
    //      snapshot (keeps boxplot returns, yearly_consistency etc.).
    //   2. Bundle is loaded → build slice from bundle.
    //   3. No bundle yet → leave this.data unchanged (soft-fail; charts
    //      will show stale ret_5d_fwd_oc data with the new outcome label).
    //
    // v6 lazy-load: Gap mode and non-default outcomes need trade_meta +
    // per_outcome_returns[outcome] which arrive on demand. This method
    // awaits those fetches with _deferredLoading=true so callers (and
    // their downstream render calls) see fresh this.data when they
    // resume after the await.
    async _swapDataForActiveOutcome() {
      if (!this.data) return;
      // P6: Overnight Gap mode takes precedence — the lower section's
      // "outcome" is the synthetic gap (cc - oc), not whatever
      // decileActiveOutcome happens to be. decileActiveOutcome stays
      // pointing at the prior real outcome so leaving Gap mode restores
      // cleanly via the normal outcome branch below.
      if (this.decileMode === 'overnight_gap') {
        // v7: gap is precomputed server-side as a synthetic 13th outcome.
        // per_bin["overnight_gap"] is already in the slim payload (bar
        // chart renders instantly); per_outcome_returns["overnight_gap"]
        // is the lazy slice (~8 MB) carrying inline trade_dates / tickers
        // / bin_20s so we don't need trade_meta or the 1d_cc / 1d_oc
        // slices. Drops Gap-mode entry from ~17s (v6: 56 MB fetch + parse
        // + client synthesis) to ~6-11s (mostly equity build + chart
        // render floor, shared with other modes).
        await this._runDeferred([
          this._ensureOutcome('overnight_gap'),
        ]);
        const slice = this._buildGapDataSlice();
        if (slice) {
          Object.assign(this.data, slice);
          this.data.equity_by_decile = this._buildEquityByDecileFromCal(slice.trade_calendar);
        }
        // M3: heatmap no longer auto-fires on outcome change — it's driven by secDrillMetric.
        this._computeSelectedStats();
        return;
      }
      const outcome = this.decileActiveOutcome;
      if (outcome === 'ret_5d_fwd_oc' && this._originalAnalyzeData) {
        Object.assign(this.data, this._originalAnalyzeData);
      } else {
        // Non-default outcome — need trade_meta + per_outcome_returns[outcome].
        await this._runDeferred([
          this._ensureTradeMeta(),
          this._ensureOutcome(outcome),
        ]);
        const slice = this._buildOutcomeDataSlice(outcome);
        if (slice) {
          Object.assign(this.data, slice);
          // equity_by_decile depends on horizon (now updated) + new trade_calendar.
          this.data.equity_by_decile = this._buildEquityByDecileFromCal(slice.trade_calendar);
        }
      }
      // M3: heatmap no longer auto-fires on outcome change — it's driven by secDrillMetric.
      this._computeSelectedStats();
    },

    // ── P3 setters ────────────────────────────────────────────────────────

    async setDecileMode(mode) {
      // P3: bin selection clears on any subset-defining change (per the
      // spec; P4 refines multi-select rules within each subset).
      const prevMode = this.decileMode;
      this.decileMode = mode;
      this.selectedBins20 = new Set();
      // P6: when the mode toggles into or out of Overnight Gap, swap
      // this.data so the downstream visuals (equity, drawdown, yearly,
      // dow, activity, decile stats, trade table) reflect either the
      // synthetic gap or the previously-active real outcome. The swap
      // also reloads the heatmap (via _swapDataForActiveOutcome) so we
      // don't need a separate loadHeatmap call here.
      //
      // v6: Gap-mode entry triggers lazy-load of trade_meta + 1d_cc + 1d_oc.
      // _swapDataForActiveOutcome handles the await; we then continue to
      // _renderDecileBar with fresh this.data.
      const gapBefore = prevMode === 'overnight_gap';
      const gapAfter  = mode === 'overnight_gap';
      if (gapBefore !== gapAfter) {
        await this._swapDataForActiveOutcome();
      }
      // Mode switch re-renders immediately and clears any pending bin selection.
      this.loadPrimary();
      this._renderDecileBar();
    },

    async setDecileHorizonAnchor(anchor) {
      this.decileHorizonAnchor = anchor;
      // Clear bin selection — different anchor = different subset
      this.selectedBins20 = new Set();
      // In Horizon mode the anchor pill defines the subset, so promote the
      // active outcome to {currentHorizon}d_{newAnchor} so the star follows
      // the user's anchor choice.
      if (this.decileMode === 'horizon') {
        const h = this._outcomeHorizon(this.decileActiveOutcome) || 5;
        const newOutcome = `ret_${h}d_fwd_${anchor}`;
        if (this.decileActiveOutcome !== newOutcome) {
          this.decileActiveOutcome = newOutcome;
          this.outcome = newOutcome;
          await this._swapDataForActiveOutcome();
        }
      }
      this.loadPrimary();
      if (this.decileMode === 'horizon') this._renderDecileBar();
    },

    async setDecileEntryHorizon(h) {
      this.decileEntryHorizon = h;
      this.selectedBins20 = new Set();
      // In Entry mode the horizon pill defines the subset, so promote the
      // active outcome to {newH}d_{currentAnchor}.
      if (this.decileMode === 'entry') {
        const a = this._outcomeAnchor(this.decileActiveOutcome);
        const newOutcome = `ret_${h}d_fwd_${a}`;
        if (this.decileActiveOutcome !== newOutcome) {
          this.decileActiveOutcome = newOutcome;
          this.outcome = newOutcome;
          await this._swapDataForActiveOutcome();
        }
      }
      this.loadPrimary();
      if (this.decileMode === 'entry') this._renderDecileBar();
    },

    async setDecileActiveOutcome(outcome) {
      if (this.decileActiveOutcome === outcome) return;
      this.decileActiveOutcome = outcome;
      // Same outcome on this.outcome too so downstream visuals (Equity, IC,
      // etc.) reflect the new choice. /analyze isn't re-fired; this.data is
      // swapped from the bundle slice (or restored from _originalAnalyzeData
      // when outcome = the /analyze default).
      this.outcome = outcome;
      await this._swapDataForActiveOutcome();
      this.loadPrimary();
    },

    decileChartTitle() {
      if (this.decileMode === 'overnight_gap') return 'Quantile Avg Return — Entry-day overnight gap (CC − OC)';
      if (this.decileMode === 'entry')         return `Quantile Avg Return — OC vs CC · ${this.decileEntryHorizon}d`;
      if (this.decileMode === 'horizon')       return `Quantile Avg Return — Horizon Ladder (${this.decileHorizonAnchor.toUpperCase()})`;
      return `Quantile Avg Return — ${this.decileActiveOutcome}`;
    },

    // ── P3 breadcrumb + cache timestamp helpers ───────────────────────────
    // Breadcrumb communicates "what filter is the lower section applying".
    // The spec called for "drop the redundant 'active' wording" — this
    // version reads as "<Mode> mode · <sub-control value>".
    decileBreadcrumb() {
      if (this.decileMode === 'overnight_gap') return 'Overnight Gap strategy';
      if (this.decileMode === 'entry')         return `Entry mode · ${this.decileEntryHorizon}d horizon`;
      if (this.decileMode === 'horizon')       return `Horizon mode · ${this.decileHorizonAnchor.toUpperCase()} anchor`;
      return `Single mode · ${this.decileActiveOutcome}`;
    },

    // Analyze-section breadcrumb. Same "last: YYYY-MM-DD HH:MM:SS · <Mode>"
    // shape used by the 6 lower-section panes (smBreadcrumb / surveyBreadcrumb
    // / cs2fBreadcrumb / topBinsBreadcrumb / tdBreadcrumb). Reads
    // analyzeBundle.computed_at (set inside _compute_analyze_bundle_sync,
    // always present on a loaded bundle) and analyzeBundle.mode (the mode
    // the bundle was COMPUTED under — may differ from this.pageMode if the
    // user changed the mode pill since the last Analyze click).
    analyzeBreadcrumb() {
      const b = this.analyzeBundle;
      if (!b || !b.computed_at) return 'no data yet';
      const ts = String(b.computed_at).slice(0, 19).replace('T', ' ');
      const mode = b.mode || 'unknown';
      const label = mode === 'walk_forward' ? 'Walk-fwd'
                  : mode === 'in_sample'    ? 'In-sample'
                  : mode === 'train_test'   ? 'Train-test'
                  : mode;
      return `last: ${ts} · ${label}`;
    },

    // Confirm-modal-gated refresh — recomputes the bundle for the current
    // (ticker, metric, mode, cutoff). ALL mode → ~5 min background;
    // single-ticker → ~2-5s inline. Doesn't touch upper-section views.
    decileRefresh() {
      if (!confirm('Recompute analysis for this metric? ALL mode takes ~5 min in the background; single-ticker takes ~2-5 seconds.')) return;
      this.analyzeBundle = null;
      this.analyzeBundleKey = null;
      this.refreshAnalyzeBundle();
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
      // P3: bundle is 20-bin; all chart paths aggregate via
      // _aggregateBin20ToN(_, n). decileBinsData (the legacy
      // _computeDecileNBins fallback for the chart-only 5/20 view from
      // this.data) is still used by the *initial* Single-mode render
      // when the bundle hasn't loaded yet AND the active outcome is
      // ret_5d_fwd_oc. Other modes/outcomes wait for the bundle.
      this.decileBinsData = null;
      // Granularity switch is gated behind Load — update bar highlight +
      // stats only; full re-render fires when user clicks Load.
      this._onDecileChangeLight();
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
      if (!el || !this.data) return;
      if (this._charts['yearly']) this._charts['yearly'].destroy();

      // W1: use pre-aggregated yearly_stats (server-side) instead of trade_calendar.
      const ystats = this.data.yearly_stats;
      if (!ystats?.length) return;

      const hasBins = ystats.some(y => y.decile20 != null);
      const filtered = hasBins && this.selectedBins20.size > 0
        ? ystats.filter(y => this.selectedBins20.has(y.decile20))
        : ystats;

      // N-weighted avg and win-rate, grouped by year.
      const byYear = {};
      for (const y of filtered) {
        if (!byYear[y.year]) byYear[y.year] = { n: 0, sumRet: 0, wins: 0 };
        byYear[y.year].n      += y.n;
        byYear[y.year].sumRet += y.avg_ret * y.n;
        byYear[y.year].wins   += y.win_rate * y.n;
      }
      const years = Object.keys(byYear).sort();
      const avgs = years.map(yr => {
        const b = byYear[yr];
        return b.n > 0 ? b.sumRet / b.n * 100 : 0;
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
          layout: { padding: { bottom: 0 } },
          plugins: {
            legend: { labels:{color:'#aaa',font:{size:10}} },
            tooltip: {
              callbacks: {
                label: ctx => {
                  const yr = years[ctx.dataIndex];
                  const b  = byYear[yr];
                  const avg = b.n > 0 ? b.sumRet / b.n : 0;
                  const wr  = b.n > 0 ? b.wins   / b.n : 0;
                  return [`Avg: ${(avg*100).toFixed(3)}%`, `WR: ${(wr*100).toFixed(0)}%`, `n: ${b.n}`];
                },
              },
            },
          },
          scales: {
            ...this._darkScalesNR(),
            y: { ...this._darkScales().y, ticks: { ...this._darkScales().y.ticks,
                  callback: v => v.toFixed(2) + '%' } },
          },
        },
      });
    },

    // ── Return Distribution (histogram: selected bins vs all) ─────────────
    // Uses /analyze's per-trade decile_stats[].returns array (only present
    // for the /analyze default outcome; after a P4 bundle-driven swap the
    // returns array is empty, so this chart silently renders nothing for
    // non-ret_5d_fwd_oc outcomes — that's intentional given the bundle's
    // memory budget).
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
          layout: { padding: { bottom: 0 } },
          plugins: {
            legend: { labels:{color:'#aaa',font:{size:10}} },
            tooltip: { backgroundColor:'rgba(20,20,20,0.95)', borderColor:'#444', borderWidth:1 },
          },
          scales: {
            ...this._darkScalesNR(),
            x: { ...this._darkScales().x, ticks: { ...this._darkScales().x.ticks, maxRotation: 0, autoSkip: true, maxTicksLimit: 10 },
                 title:{display:true,text:'Return %',color:'#888',font:{size:10}} },
            y: { ...this._darkScales().y, title:{display:true,text:'Count',color:'#888',font:{size:10}} },
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

    // ── 2D Heatmap ── M3: Y-axis is now secSelectedMetric; no standalone dropdown ──
    heatmapData: null,
    heatmapLoading: false,
    heatmapBins: 20,
    hmBins1d: 10,
    hmXData: null,
    hmYData: null,
    // Gate F replacement: n-slider for the heatmap. Cells with n below
    // `hmMinSampleN` render as hatched gray (no gradient color), AND are
    // excluded from the gradient scale's min/max computation. Same
    // threshold drives both — see `_hmRange` getter below.
    hmMinSampleN: 50,
    _hmRange: null,

    // ── Zone Analyze (Stage 1) ────────────────────────────────────────────────
    // Cell selection state — Set of "ix-iy" strings. Alpine's reactivity
    // requires re-assigning the Set (not mutating in place) to trigger re-renders.
    // Wrapped in a proxy-friendly plain object. Only active in IS mode.
    hmSelectedCells: new Set(),   // Set<"ix-iy">
    zoneOpen:         false,      // zone section expanded
    zoneLoading:      false,
    zoneData:         null,       // last /secondary-zone-analyze response
    // Empty by default — the Analyze Zone dropdown shows the primary
    // section's active outcome (`this.outcome`) until the user
    // explicitly picks a different one. analyzeZone / saveSignal use
    // `this.zoneOutcome || this.outcome` so the analysis follows the
    // primary too. Setting any value here means the zone analysis has
    // diverged from the primary.
    zoneOutcome:      '',

    // Signals — saved named cell-sets
    signals:          [],
    signalsExpanded:  false,
    signalName:       '',
    signalSaving:     false,
    signalSaveMsg:    '',
    // Saved Signals table — sort, checkbox, expand, batch action state.
    // sigCheckedIds is keyed by signal_id so sorting doesn't drop checks
    // and reload re-applies them to whichever rows still exist.
    sigSortKey:       'agg_avg_ret',
    sigSortDir:       'desc',
    sigCheckedIds:    {},
    sigExpanded:      {},
    sigBatchBusy:     false,
    sigBatchMsg:      '',

    // Recall view — independent state from the main analysis flow so
    // opening a saved signal can't clobber a primary/secondary pair
    // already loaded in the main heatmap. recallSelectedCells matches
    // hmSelectedCells's "ix-iy" key format so the set-equality dirty
    // check compares apples to apples.
    recallSig:           null,
    recallExpanded:      false,
    recallLoading:       false,
    recallHeatmapData:   null,
    recallZoneData:      null,
    recallSelectedCells: new Set(),
    recallSaving:        false,
    recallSaveMsg:       '',
    // Editable name lives in its own scratch field — we don't mutate
    // recallSig.name directly because recallSig points at the same
    // object instance as the row in `signals[]`, and any direct edit
    // would flash the new name in the list before the save completes.
    recallEditedName:    '',

    // P5: in Overnight Gap mode, the heatmap and sidebar bin charts use
    // the synthetic outcome `overnight_gap`, which the backend resolves
    // to per-trade (ret_1d_fwd_cc − ret_1d_fwd_oc). Outside Gap mode,
    // they follow whatever the active outcome is.
    _heatmapOutcome() {
      return this.decileMode === 'overnight_gap' ? 'overnight_gap' : this.outcome;
    },

    // Returns a string key of every parameter that drives the heatmap grid.
    // loadHeatmapIfChanged() skips the fetch when none of these change.
    _heatmapKey() {
      return JSON.stringify([
        this.ticker, this.metric, this.secSelectedMetric,
        this.pageMode, this.cutoffDate, this.heatmapBins,
        this._heatmapOutcome(), this.dateFrom, this.dateTo,
      ]);
    },

    // Called from secDrillMetric() instead of loadHeatmap() directly.
    // Skips the fetch when only bin selection changed (grid is invariant to
    // which bins are selected; it always shows all bins).
    async loadHeatmapIfChanged() {
      if (!this.secSelectedMetric || !this.data) return;
      if (this._heatmapKey() === this._lastHeatmapKey) return;
      await this.loadHeatmap();
    },

    async loadHeatmap() {
      // M3: Y-axis is now the selected secondary metric; no standalone dropdown.
      if (!this.secSelectedMetric || !this.data) return;
      // Stamp the key so loadHeatmapIfChanged() won't re-fire for the same inputs.
      this._lastHeatmapKey = this._heatmapKey();
      this.heatmapLoading = true;
      this.heatmapData = null;
      this.hmXData = null;
      this.hmYData = null;
      this._hmRange = null;
      // walk_forward / train_test heatmap: all tickers now use stored-bin
      // rank bins (B1..BN) for all three modes — pass mode flags unconditionally.
      let wf = '';
      if (this.pageMode === 'walk_forward') wf = '&walk_forward=true';
      else if (this.pageMode === 'train_test') wf = `&cutoff_date=${encodeURIComponent(this.cutoffDate)}`;
      try {
        const r = await fetch(
          `/api/factor-analysis/heatmap?ticker=${encodeURIComponent(this.ticker)}`
          + `&metric_x=${encodeURIComponent(this.metric)}`
          + `&metric_y=${encodeURIComponent(this.secSelectedMetric)}`
          + `&outcome=${encodeURIComponent(this._heatmapOutcome())}&bins=${this.heatmapBins}`
          + (this.dateFrom ? `&date_from=${this.dateFrom}` : '')
          + (this.dateTo   ? `&date_to=${this.dateTo}`     : '')
          + wf);
        if (r.ok) {
          const d = await r.json();
          // Gate F replacement: `_hmRange` is no longer set here. It's
          // computed by `_hmComputeRange()` from cells whose n meets
          // `hmMinSampleN`, so dragging the slider re-scopes the scale
          // without a refetch. Just store the data.
          this.heatmapData = d;
          this._hmRecomputeRange();
        }
        await this.loadHmBins1d();
      } catch (_) {}
      this.heatmapLoading = false;
    },

    setHeatmapBins(n) {
      if (n !== this.heatmapBins) {
        // Grid resolution changed — cell indices are no longer valid; clear selection.
        this.hmSelectedCells = new Set();
        this.zoneData = null;
        this._destroyZoneCharts();
      }
      this.heatmapBins = n;
      this.loadHeatmap();
    },

    async setHmBins1d(n) {
      this.hmBins1d = n;
      await this.loadHmBins1d();
    },

    async loadHmBins1d() {
      if (!this.data || !this.secSelectedMetric) return;
      // /metric-bins (post-Step-5.5-continuation) supports walk_forward
      // and (Step 6) train_test for both ALL and single-ticker modes via
      // the Assigner. Send the page-wide mode so the side bin charts
      // agree with the rest of the page (top All-Ticker Metric Bins,
      // standalone primary chart).
      let wf = '';
      if (this.pageMode === 'walk_forward') wf = '&walk_forward=true';
      else if (this.pageMode === 'train_test') wf = `&cutoff_date=${encodeURIComponent(this.cutoffDate)}`;
      const base = `/api/factor-analysis/metric-bins?ticker=${encodeURIComponent(this.ticker)}`
        + `&outcome=${encodeURIComponent(this._heatmapOutcome())}&bins=${this.hmBins1d}`
        + (this.dateFrom ? `&date_from=${this.dateFrom}` : '')
        + (this.dateTo ? `&date_to=${this.dateTo}` : '')
        + wf;

      // Overnight Gap mode: serve the X-axis sidebar directly from the
      // bundle so its values match the main Quantile pane exactly. The
      // bundle's bin_20 assignments come from the anchor-outcome
      // (ret_5d_fwd_oc) row-validity set; /metric-bins for the synthetic
      // gap outcome filters on (cc AND oc) non-null, which is a different
      // row set and produces a different ticker exclusion — symptom is
      // the residual ~few-bp gap between sidebar and main pane after
      // canonical-20 binning. Bundle-derived path bypasses /metric-bins
      // for X and is faster (no round-trip).
      // Y still fetches because its metric differs from the main pane
      // and the bundle doesn't carry per-bin stats for arbitrary metrics.
      if (this.decileMode === 'overnight_gap' && this.analyzeBundle) {
        const x20 = this._overnightGapPerBin20();
        const xAgg = x20 ? this._aggregateBin20ToN(x20, this.hmBins1d) : null;
        this.hmXData = xAgg ? xAgg.filter(Boolean) : null;
        try {
          const ry = await fetch(base + `&metric=${encodeURIComponent(this.secSelectedMetric)}`);
          if (ry.ok) {
            const d = await ry.json();
            this.hmYData = d.buckets || null;
          }
        } catch (e) { console.error('[hmBins1d gap Y] fetch failed', e); }
        await this.$nextTick();
        this._renderHmBar1d('chart-hm-x', this.hmXData, this.metric);
        this._renderHmBar1d('chart-hm-y', this.hmYData, this.secSelectedMetric);
        return;
      }

      try {
        const [rx, ry] = await Promise.all([
          fetch(base + `&metric=${encodeURIComponent(this.metric)}`),
          fetch(base + `&metric=${encodeURIComponent(this.secSelectedMetric)}`),
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
          console.log('[hmBins1d Y]', this.secSelectedMetric, d.error || `n=${d.n} buckets=${(d.buckets||[]).length}`, d);
          this.hmYData = d.buckets || null;
        } else {
          console.warn('[hmBins1d Y] HTTP', ry.status, await ry.text());
        }
      } catch (e) { console.error('[hmBins1d] fetch failed', e); }
      await this.$nextTick();
      this._renderHmBar1d('chart-hm-x', this.hmXData, this.metric);
      this._renderHmBar1d('chart-hm-y', this.hmYData, this.secSelectedMetric);
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
      // Three tiers driven by the SINGLE `hmMinSampleN` threshold —
      // same number determines hatching AND gradient inclusion. Critical
      // invariant: a cell rendered with a gradient color is also in the
      // population whose min/max defined that gradient. If a low-n cell
      // with a wild return stayed in the scale, every real cell would
      // collapse to near-uniform shade by relativity — the user-stated
      // reason for keeping hatch and scale on the same threshold.
      if (!cell || !cell.n) return 'rgba(40,40,40,0.5)';   // n=0: empty
      const n = cell.n;
      const minN = this.hmMinSampleN || 0;
      // Tier 2: 0 < n < threshold — hatched gray, no gradient color.
      // Pattern matches the Regime Heatmap style for visual consistency.
      if (n < minN) {
        return 'repeating-linear-gradient(45deg, #2e2e2e 0 4px, transparent 4px 8px),'
             + 'repeating-linear-gradient(-45deg, #2e2e2e 0 4px, transparent 4px 8px),'
             + '#1c1c1c';
      }
      // Tier 3: n >= threshold — gradient, scaled across visible cells only.
      const t = Math.max(-1, Math.min(1, (cell.avg_ret || 0) / (this._hmRange || 0.01)));
      if (t >= 0) return `rgba(52,152,219,${(0.15 + t * 0.7).toFixed(2)})`;
      return `rgba(232,67,147,${(0.15 + (-t) * 0.7).toFixed(2)})`;
    },

    // Recompute `_hmRange` from cells whose n meets `hmMinSampleN`. Called
    // when heatmap data arrives AND from the `hmMinSampleN` watcher so
    // dragging the slider rescopes the gradient. Cells below threshold
    // drop out of the max-abs computation; the gradient stretches across
    // the survivors. This is the difference from a hatch-only n-slider.
    _hmRecomputeRange() {
      if (!this.heatmapData) { this._hmRange = null; return; }
      const minN = this.hmMinSampleN || 0;
      const grids = [
        ...(this.heatmapData.grid || []),
        ...(this.heatmapData.train_grid || []),
        ...(this.heatmapData.test_grid || []),
      ];
      let max = 0;
      for (const row of grids) {
        for (const c of row) {
          if (c && (c.n || 0) >= minN) {
            max = Math.max(max, Math.abs(c.avg_ret || 0));
          }
        }
      }
      this._hmRange = max || 0.01;
    },

    // Wire the slider's reactivity. Called from x-init in the heatmap
    // template (see template). Cheap idempotent setup — re-running is
    // harmless because Alpine deduplicates identical watchers.
    _hmInitWatchers() {
      if (this._hmWatchersInit) return;
      this._hmWatchersInit = true;
      this.$watch('hmMinSampleN', () => this._hmRecomputeRange());
    },

    // Tooltip for a heatmap cell. In train-test mode, appends the frozen
    // training-set thresholds for both axes so the user can see what metric
    // value ranges each bin corresponds to.
    _hmCellTitle(cell, ix, iy) {
      if (!cell || !cell.n) return 'n=0';
      let s = `n=${cell.n}  avg=${((cell.avg_ret||0)*100).toFixed(3)}%  wr=${((cell.win_rate||0)*100).toFixed(1)}%`;
      const xt = this.heatmapData?.x_thresholds;
      const yt = this.heatmapData?.y_thresholds;
      if (xt && yt) {
        const fmt = v => v !== undefined ? v.toFixed(4) : '?';
        s += `\nX (${this.metric}): ${fmt(xt[ix])} – ${fmt(xt[ix+1])}`;
        s += `\nY (${this.secSelectedMetric}): ${fmt(yt[iy])} – ${fmt(yt[iy+1])}`;
      }
      return s;
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
          `/api/factor-analysis/ai-summary?ticker=${encodeURIComponent(this.ticker)}`
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
    // ── Day of week P&L ────────────────────────────────────────────────
    _renderDOW() {
      const el = document.getElementById('chart-dow');
      if (!el || !this.data) return;
      if (this._charts['dow']) this._charts['dow'].destroy();

      // W1: use pre-aggregated dow_stats (server-side) instead of dow_data.
      const dstats = this.data.dow_stats;
      if (!dstats?.length) return;

      const dowNames = ['Mon','Tue','Wed','Thu','Fri'];
      const hasBins = dstats.some(d => d.decile20 != null);
      const filtered = hasBins && this.selectedBins20.size > 0
        ? dstats.filter(d => this.selectedBins20.has(d.decile20))
        : dstats;

      // N-weighted avg_ret and win_rate by day-of-week.
      const byDow = {};
      for (const d of filtered) {
        if (!byDow[d.dow]) byDow[d.dow] = { sumN: 0, sumRet: 0, sumWr: 0 };
        byDow[d.dow].sumN   += d.n;
        byDow[d.dow].sumRet += d.avg_ret * d.n;
        byDow[d.dow].sumWr  += d.win_rate * d.n;
      }
      const avgs   = dowNames.map((_, i) =>
        byDow[i] && byDow[i].sumN ? byDow[i].sumRet / byDow[i].sumN * 100 : 0);
      const counts = dowNames.map((_, i) => byDow[i]?.sumN || 0);
      const wrs    = dowNames.map((_, i) =>
        byDow[i] && byDow[i].sumN ? byDow[i].sumWr / byDow[i].sumN * 100 : 0);

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
          layout: { padding: { bottom: 0 } },
          plugins: {
            legend: { display: false },
            tooltip: { callbacks: { label: ctx => [
              `Avg: ${avgs[ctx.dataIndex].toFixed(3)}%`,
              `WR: ${wrs[ctx.dataIndex].toFixed(1)}%`,
              `n: ${counts[ctx.dataIndex]}`,
            ] } },
          },
          scales: {
            ...this._darkScalesNR(),
            y: { ...this._darkScales().y, ticks: { ...this._darkScales().y.ticks,
                  callback: v => v.toFixed(2) + '%' } },
          },
        },
      });
    },

    // ── Trade activity: entries/open per day ──────────────────────────────────
    _renderActivity() {
      const el = document.getElementById('chart-activity');
      if (!el || !this.data) return;
      if (this._charts['activity']) this._charts['activity'].destroy();

      // W1: use pre-aggregated activity_by_date (server-side).
      // dedupeConc.primary is dropped — slim trade_calendar no longer carries ticker.
      const actData = this.data.activity_by_date;
      if (!actData?.length) return;

      const horizon = this.data.horizon || 1;
      const hasBins = actData.some(a => a.decile20 != null);
      const filtered = hasBins && this.selectedBins20.size > 0
        ? actData.filter(a => this.selectedBins20.has(a.decile20))
        : actData;

      // Sum entry counts by date across selected bins.
      const entriesByDate = {};
      for (const a of filtered) {
        entriesByDate[a.date] = (entriesByDate[a.date] || 0) + a.n;
      }

      // Trading-day axis: spot_series for single-ticker; activity dates for ALL.
      const spotSeries  = this.data.spot_series || [];
      const tradingDays = spotSeries.length > 0
        ? spotSeries.map(s => s.date)
        : [...new Set(actData.map(a => a.date))].sort();

      const entered = tradingDays.map(d => entriesByDate[d] || 0);
      // Open = entries in the N-trading-day window [i-N+1 .. i]
      const open = tradingDays.map((_, i) => {
        let count = 0;
        for (let j = Math.max(0, i - horizon + 1); j <= i; j++) {
          count += entriesByDate[tradingDays[j]] || 0;
        }
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
      if (!bodyEl || !this.data) return;

      if (this.tradeView === 'by_ticker') {
        // By-ticker view aggregates from slim trade_calendar (has ticker+ret+decile20).
        const cal = this.data.trade_calendar || [];
        const has20 = !!(cal[0]?.decile20);
        const filtered = has20 && this.selectedBins20.size > 0
          ? cal.filter(c => this.selectedBins20.has(c.decile20))
          : cal;
        this._renderTradeTableByTicker(headEl, bodyEl, cntEl, filtered);
      } else {
        // Flat view fetches full trade details from /trades endpoint (W1).
        this._renderTradeTableFlat(headEl, bodyEl, cntEl).catch(e => {
          if (bodyEl) bodyEl.innerHTML =
            `<tr><td colspan="8" style="color:#e84393;padding:8px">Error: ${e.message}</td></tr>`;
        });
      }
    },

    // Build the Flat trade list from the analyze_cache bundle. Used for
    // non-default outcomes (the /trades server cache only carries the
    // /analyze default = ret_5d_fwd_oc; other outcomes returned
    // not_cached → "Run Analyze first" placeholder). Bundle has every
    // field the table renders: trade_meta gives ticker/date/metric_val/
    // entry_spot/bin_20; per_outcome_returns gives ret/exit_date/exit_spot.
    //
    // P6: outcome="overnight_gap" intersects ret_1d_fwd_cc and
    // ret_1d_fwd_oc by trade_id and stores the per-row gap (cc - oc)
    // in `ret`. exit_date/spot come from the cc leg (both legs land at
    // close of trade_date for 1-day horizon).
    _buildFlatTradesFromBundle(outcome) {
      const b = this.analyzeBundle;
      if (!b) return null;

      // v8: Gap uses the precomputed overnight_gap slice — every flat-
      // table field is carried inline (entry_dates_cc, entry_spots_cc,
      // entry_spots_oc, metric_vals + the chart fields from v7). No
      // trade_meta dependency, no 1d_cc / 1d_oc dependency. This is what
      // eliminates the 56 MB fetch trigger that caused the 24s Gap-mode
      // entry and the Analyze-in-Gap-mode wedge symptom.
      if (outcome === 'overnight_gap') {
        const gap = b.per_outcome_returns?.overnight_gap;
        if (!gap || !Array.isArray(gap.trade_dates)) return null;
        const out = [];
        for (let i = 0; i < gap.trade_dates.length; i++) {
          out.push({
            date:       gap.entry_dates_cc[i],   // entry T−1
            ticker:     gap.tickers[i],
            metric_val: gap.metric_vals[i],
            spot_entry: gap.entry_spots_cc[i],   // close T−1
            spot_exit:  gap.entry_spots_oc[i],   // open T
            ret:        gap.ret_pcts[i],         // gap = cc − oc
            exit_date:  gap.trade_dates[i],      // T = OC-anchor exit
            decile20:   gap.bin_20s[i],
          });
        }
        return out;
      }

      // Real outcomes still use trade_meta + per_outcome_returns[outcome].
      // The trade-meta dependency here is unchanged from v6/v7 because
      // non-gap outcomes carry only ret/exit_date/exit_spot in their
      // slice; entry fields and metric_val come from trade_meta.
      const tm = b.trade_meta;
      if (!Array.isArray(tm)) return null;
      const ret = b.per_outcome_returns?.[outcome];
      if (!ret) return null;
      const anchor = this._outcomeAnchor(outcome);
      const dateKey = `entry_date_${anchor}`;
      const spotKey = `entry_spot_${anchor}`;
      const out = [];
      for (let i = 0; i < ret.trade_ids.length; i++) {
        const m = tm[ret.trade_ids[i]];
        if (!m) continue;
        out.push({
          date:       m[dateKey],
          ticker:     m.ticker,
          metric_val: m.metric_val,
          spot_entry: m[spotKey],
          spot_exit:  ret.exit_spots[i],
          ret:        ret.ret_pcts[i],
          exit_date:  ret.exit_dates[i],
          decile20:   m.bin_20,
        });
      }
      return out;
    },

    _sortFlatTrades(rows, key, dir) {
      // Map "bin" → "decile20" the same way /trades does server-side; the
      // table header sends sort_key="bin" but bundle rows carry decile20.
      const resolved = key === 'bin' ? 'decile20' : key;
      const strKeys = new Set(['date', 'ticker', 'exit_date']);
      const sgn = dir === 'asc' ? 1 : -1;
      const cmp = (a, b) => {
        const va = a[resolved], vb = b[resolved];
        if (strKeys.has(resolved)) {
          return sgn * String(va ?? '').localeCompare(String(vb ?? ''));
        }
        const na = (va == null) ? -Infinity : va;
        const nb = (vb == null) ? -Infinity : vb;
        return sgn * (na - nb);
      };
      return rows.slice().sort(cmp);
    },

    async _renderTradeTableFlat(headEl, bodyEl, cntEl) {
      // Bundle-backed for non-default outcomes; /trades fetch otherwise.
      // See _buildFlatTradesFromBundle for context.
      if (!this.data) return;
      if (bodyEl) bodyEl.innerHTML =
        '<tr><td colspan="8" style="text-align:center;color:#888;padding:12px">Loading…</td></tr>';

      let rows, total;
      // In Gap mode the effective outcome is the synthetic overnight_gap,
      // even though this.outcome still points at the prior real outcome
      // (so we can restore on Gap-mode exit).
      const effectiveOutcome = this.decileMode === 'overnight_gap'
        ? 'overnight_gap' : this.outcome;

      // v6 lazy-load behaviour by outcome class:
      //   - Gap mode / non-default outcome: explicit user action — chart
      //     overlay spinner shows during the await (existing).
      //   - Default outcome (ret_5d_fwd_oc): the most-used view. Bundle
      //     prefetch was kicked off in the background when slim arrived
      //     (_prefetchDefaultOutcomeAfterBundle). If the prefetch has
      //     already resolved, render from bundle immediately. If it's
      //     still in flight, show a passive "Loading trade data…"
      //     placeholder — NO chart spinner, NO await. When the prefetch
      //     resolves it triggers a re-render and the table populates.
      //     This keeps the main visuals fully decoupled from the trade
      //     fetch.
      //   - Single-ticker: bundle is inline complete; lazy-fetch helpers
      //     short-circuit; falls through to bundle rendering below.
      if (this.ticker === 'ALL' && this.analyzeBundle) {
        if (effectiveOutcome === 'overnight_gap') {
          // v8: precomputed gap slice carries every flat-table field
          // inline — no trade_meta, no 1d_cc / 1d_oc fetches. The same
          // _ensureOutcome('overnight_gap') call that _swapData fires
          // covers this render too (dedup'd via in-flight map). On a
          // warm v8 cache the data is usually already present from the
          // Gap-mode-entry swap that just happened.
          await this._runDeferred([
            this._ensureOutcome('overnight_gap'),
          ]);
        } else if (effectiveOutcome !== 'ret_5d_fwd_oc') {
          await this._runDeferred([
            this._ensureTradeMeta(),
            this._ensureOutcome(effectiveOutcome),
          ]);
        } else {
          // Default outcome — passive placeholder if data not yet loaded.
          // The background prefetch will trigger a re-render when ready.
          const haveTradeMeta = !!this.analyzeBundle.trade_meta;
          const haveOutcome   = !!this.analyzeBundle.per_outcome_returns?.[effectiveOutcome];
          if (!haveTradeMeta || !haveOutcome) {
            if (bodyEl) bodyEl.innerHTML =
              '<tr><td colspan="8" style="text-align:center;color:#888;padding:12px;font-style:italic">'
              + 'Loading trade data…'
              + '</td></tr>';
            return;
          }
        }
      }

      const bundleRows = this._buildFlatTradesFromBundle(effectiveOutcome);

      if (bundleRows) {
        const filtered = this.selectedBins20.size > 0
          ? bundleRows.filter(t => this.selectedBins20.has(t.decile20))
          : bundleRows;
        const sorted = this._sortFlatTrades(filtered, this.tradeSortKey, this.tradeSortDir);
        total = sorted.length;
        rows  = sorted.slice(0, 250);
      } else {
        // Defensive fallback for edge cases (single-ticker bundle still
        // loading, or ALL-mode with no analyzeBundle yet). The v6
        // ALL-mode default-outcome path is handled above and never
        // reaches this branch — see the lazy-load gate at the top.
        const mode = this.pageMode === 'train_test' ? 'train_test'
                   : this.pageMode === 'in_sample'  ? 'in_sample'
                   : 'walk_forward';
        const params = new URLSearchParams({
          ticker:   this.ticker,
          metric:   this.metric,
          outcome:  this.outcome,
          mode,
          sort_key: this.tradeSortKey,
          sort_dir: this.tradeSortDir,
          page:     0,
          page_size: 250,
        });
        if (mode === 'train_test' && this.cutoffDate) params.set('cutoff_date', this.cutoffDate);
        for (const b of this.selectedBins20) params.append('decile20', b);

        const r = await fetch(`/api/factor-analysis/trades?${params}`);
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        const d = await r.json();

        if (d.error === 'not_cached') {
          if (bodyEl) bodyEl.innerHTML =
            '<tr><td colspan="8" style="text-align:center;color:#888;padding:12px">Run Analyze first</td></tr>';
          return;
        }
        rows  = d.trades || [];
        total = d.total  || 0;
      }

      if (cntEl) {
        cntEl.textContent = total > 250
          ? `Showing 250 of ${total.toLocaleString()} trades — export CSV for all`
          : `${total.toLocaleString()} trades`;
      }
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
      if (bodyEl) {
        bodyEl.innerHTML = rows.map(t => {
          const entrySpot = t.spot_entry ?? null;
          const exitSpot  = t.spot_exit  ?? null;
          const exitDate  = t.exit_date  || '';
          const retPct    = (t.ret * 100).toFixed(3);
          const sign      = t.ret >= 0 ? '+' : '';
          const color     = t.ret >= 0 ? '#3498db' : '#e84393';
          return `<tr>
            <td>${t.date}</td>
            <td>${t.ticker || ''}</td>
            <td class="num">${t.metric_val != null ? t.metric_val.toFixed(4) : ''}</td>
            <td class="num">${entrySpot != null ? entrySpot.toLocaleString('en-US', {minimumFractionDigits:2, maximumFractionDigits:2}) : ''}</td>
            <td class="num">${exitSpot  != null ? exitSpot .toLocaleString('en-US', {minimumFractionDigits:2, maximumFractionDigits:2}) : ''}</td>
            <td class="num" style="color:${color}">${sign}${retPct}%</td>
            <td>${exitDate}</td>
            <td class="num" style="color:#888">${t.decile20 || t.decile || ''}</td>
          </tr>`;
        }).join('');
      }
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
      // Default outcome → server /trades/csv (cache is warm, streams the
      // full filtered set). Non-default outcomes → client-side build
      // from the bundle, same row shape and column order as the server
      // (oi_analysis.py:1278). Bin filter (selectedBins20) honored on
      // both paths.
      if (!this.data) return;
      const mode = this.pageMode === 'train_test' ? 'train_test'
                 : this.pageMode === 'in_sample'  ? 'in_sample'
                 : 'walk_forward';
      // Gap mode → effective outcome is the synthetic overnight_gap; the
      // CSV builder takes the same path as the Flat view.
      const effectiveOutcome = this.decileMode === 'overnight_gap'
        ? 'overnight_gap' : this.outcome;
      const isDefault = effectiveOutcome === 'ret_5d_fwd_oc';
      const bundleRows = !isDefault ? this._buildFlatTradesFromBundle(effectiveOutcome) : null;
      if (bundleRows) {
        const filtered = this.selectedBins20.size > 0
          ? bundleRows.filter(t => this.selectedBins20.has(t.decile20))
          : bundleRows;
        filtered.sort((a, b) => String(a.date).localeCompare(String(b.date)));
        // CSV escape for any field that contains a comma, quote, or newline.
        const esc = (v) => {
          if (v == null) return '';
          const s = String(v);
          return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
        };
        const header = ['date', 'ticker', this.metric || 'metric',
                        'spot_entry', 'spot_exit', 'ret_pct', 'exit_date', 'bin20'];
        const lines = [header.map(esc).join(',')];
        for (const t of filtered) {
          lines.push([
            t.date, t.ticker,
            t.metric_val ?? '',
            t.spot_entry ?? '',
            t.spot_exit  ?? '',
            (t.ret != null ? (t.ret * 100).toFixed(6) : ''),
            t.exit_date ?? '',
            t.decile20  ?? '',
          ].map(esc).join(','));
        }
        const blob = new Blob([lines.join('\n')], { type: 'text/csv;charset=utf-8' });
        const today = new Date().toISOString().slice(0, 10);
        const a = document.createElement('a');
        a.href = URL.createObjectURL(blob);
        a.download = `trades_${this.ticker}_${this.metric}_${today}.csv`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        // Free the object URL on the next tick; the click handler has
        // already kicked off the download by then.
        setTimeout(() => URL.revokeObjectURL(a.href), 0);
        return;
      }
      const params = new URLSearchParams({
        ticker:  this.ticker,
        metric:  this.metric,
        outcome: this.outcome,
        mode,
      });
      if (mode === 'train_test' && this.cutoffDate) params.set('cutoff_date', this.cutoffDate);
      for (const b of this.selectedBins20) params.append('decile20', b);
      const a = document.createElement('a');
      a.href = `/api/factor-analysis/trades/csv?${params}`;
      a.click();
    },

    _renderAllCharts() {
      this._renderDecileBar();
      this._renderEquity();
      this._renderYearly();
      this._renderDrawdown();
      this._renderRollingCorr();
      this._renderReturnDist();
      this._renderDOW();
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
          'chart-dist':          () => this._renderReturnDist(),
          'chart-drawdown':      () => this._renderDrawdown(),
          'chart-dow':           () => this._renderDOW(),
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
          'chart-dist':          () => this._renderReturnDist(),
          'chart-drawdown':      () => this._renderDrawdown(),
          'chart-dow':           () => this._renderDOW(),
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
    // Bucket A step 3: local mode state per the per-pane independence
    // rule. The Score Matrix pane reads smMode / smCutoffDate, NOT the
    // Analyze section's pageMode / cutoffDate. The pane's per-mode
    // results live in smDataByMode; switching smMode swaps which slot
    // is displayed (in-memory only — never auto-fetches). The Refresh
    // button (and Run Scan, and Apply) fetches for the active smMode
    // and writes into the matching slot.
    smMode: 'in_sample',
    smCutoffDate: '2024-01-01',
    smDataByMode: {
      in_sample:    null,  // { meta, rows, summary } once loaded; null = never loaded
      walk_forward: null,
      train_test:   null,
    },
    smStatusByMode: {
      in_sample:    'empty',  // 'empty' | 'loading' | 'ready' | 'no_data' | 'error'
      walk_forward: 'empty',
      train_test:   'empty',
    },
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
    smExpanded: false,

    // ── Corner Scan ─────────────────────────────────────────────────────────
    // Bucket A step 6: Corner Scan 2F local mode. Pane reads cs2fMode,
    // NOT this.pageMode. Per-mode slot cache holds meta + rows + total
    // independently for each binning mode. Mode-pill swap is in-memory
    // only. Refresh fetches for the active local mode and writes the
    // slot. Refresh is DISABLED for IS/TT until the in-sample / train-
    // test corner-scan batch job exists (a future task).
    csMetrics: [],   // sorted distinct eligible metrics — populated from /corner-scan/meta; shared by cs2f + cs1f dropdowns
    cs2fExpanded: false, cs2fLoading: false, cs2fMeta: null, cs2fRows: [], cs2fTotal: 0,
    cs2fPage: 1,  // current page (50 rows/page, server-side)
    cs2fSortKey: 'd_ret_per_day', cs2fSortDir: 'desc',
    cs2fFilterP: '', cs2fFilterS: '', cs2fFilterDir: '', cs2fFilterOutcome: '', cs2fMinN: 300,
    cs2fMode:       'in_sample',
    cs2fCutoffDate: '2024-01-01',
    cs2fDataByMode: {
      walk_forward: null,  // { meta, rows, total, page }
      in_sample:    null,
      train_test:   null,
    },

    cs1fExpanded: false, cs1fLoading: false, cs1fRows: [], cs1fTotal: 0,
    cs1fPage: 1,  // current page (50 rows/page, server-side)
    cs1fSortKey: 'd_ret_per_day', cs1fSortDir: 'desc',
    cs1fFilterMetric: '', cs1fFilterExtreme: '', cs1fFilterOutcome: '', cs1fMinN: 300,
    cs1fMeta:       null,  // pane-local meta (previously read from cs2fMeta — decoupled in step 6)
    cs1fMode:       'in_sample',
    cs1fCutoffDate: '2024-01-01',
    cs1fDataByMode: {
      walk_forward: null,
      in_sample:    null,
      train_test:   null,
    },

    surveyExpanded: false,
    selectedStats: null,

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

    toggleSm() {
      this.smExpanded = !this.smExpanded;
      if (this.smExpanded && this.smMeta.count > 0) {
        setTimeout(() => {
          this._renderSmMetricChart();
          this._renderSmFwdChart();
          this._renderSmTickerChart();
          this._renderSmTickerFwdChart();
        }, 50);
      }
    },

    toggleSurvey() {
      this.surveyExpanded = !this.surveyExpanded;
      if (this.surveyExpanded) {
        // Re-render all IC charts after the pane becomes visible in the DOM.
        setTimeout(() => {
          if (this.icBatchData) this._renderIcBatch();
          if (this.icDecompData) { this._renderIcDecomp(); this._renderIcLorenz(); }
        }, 50);
      }
    },

    _computeSelectedStats() {
      if (!this.data?.trade_calendar?.length) { this.selectedStats = null; return; }
      const tc = this.data.trade_calendar;
      const rets = [];
      const hasSel = this.selectedBins20.size > 0;
      for (const t of tc) {
        if (!hasSel || this.selectedBins20.has(t.decile20)) rets.push(t.ret);
      }
      if (!rets.length) { this.selectedStats = null; return; }
      const n = rets.length;
      const sorted = [...rets].sort((a, b) => a - b);
      const sum = rets.reduce((a, b) => a + b, 0);
      const mean = sum / n;
      const wins   = rets.filter(r => r > 0);
      const losses = rets.filter(r => r <= 0);
      let ssq = 0;
      for (const r of rets) { const d = r - mean; ssq += d * d; }
      const std  = Math.sqrt(ssq / n);
      const p5   = sorted[Math.max(0, Math.floor(n * 0.05))];
      const p95  = sorted[Math.min(n - 1, Math.floor(n * 0.95))];
      let years = 1;
      const filtDates = tc
        .filter(t => !hasSel || this.selectedBins20.has(t.decile20))
        .map(t => t.date);
      if (filtDates.length > 1) {
        const sd  = [...filtDates].sort();
        const ms  = new Date(sd[sd.length - 1]) - new Date(sd[0]);
        const yrs = ms / (365.25 * 24 * 3600 * 1000);
        if (yrs > 0.1) years = yrs;
      }
      this.selectedStats = {
        n,
        avg_ret:         mean,
        median:          sorted[Math.floor((n - 1) / 2)],
        std,
        p5,
        p95,
        win_rate:        wins.length / n,
        n_winners:       wins.length,
        avg_winners:     wins.length  ? wins.reduce((a, b)   => a + b, 0) / wins.length   : 0,
        avg_losers:      losses.length ? losses.reduce((a, b) => a + b, 0) / losses.length : 0,
        trades_per_year: n / years,
      };
    },

    // Mode label for display (also reused by the consistent ".bucketA-
    // mode-tag" span pattern across all 6 panes).
    _smModeLabel(m) {
      if (m === 'in_sample')    return 'In-sample';
      if (m === 'train_test')   return 'Train-test';
      return 'Walk-forward';
    },

    // Pane-header breadcrumb. Format is identical across all 6 panes:
    //   "last: YYYY-MM-DD HH:MM:SS · <Mode label>"
    // Falls back to "no data yet" prefix when the slot has never been
    // loaded. last_run is already formatted server-side as the first 19
    // chars of the timestamp (see score-matrix/meta:2158) so we use it
    // verbatim.
    smBreadcrumb() {
      const slot = this.smDataByMode[this.smMode];
      const label = this._smModeLabel(this.smMode);
      if (!slot || !slot.meta || slot.meta.count === 0) {
        return `no data yet · ${label}`;
      }
      const ts = slot.meta.last_run || 'unknown';
      return `last: ${ts} · ${label}`;
    },

    // Mode-pill click handler. In-memory only — swaps display from the
    // slot for the chosen mode; NEVER auto-fetches. User clicks Refresh
    // to load.
    setSmMode(m) {
      if (m === this.smMode && m !== 'train_test') return;
      this.smMode = m;
      this._smSwapDisplayFromSlot();
    },

    // Cutoff-date input change in train_test mode is also in-memory only.
    setSmCutoffDate(d) {
      this.smCutoffDate = d;
      // No fetch — user clicks Refresh to load for the new cutoff.
    },

    // Re-point the visible state (smMeta / smRows / smSummary) at the
    // slot for the active smMode. If the slot is empty, reset display to
    // an empty meta so the placeholder shows.
    _smSwapDisplayFromSlot() {
      const slot = this.smDataByMode[this.smMode];
      if (slot && slot.meta) {
        this.smMeta = slot.meta;
        this.smRows = slot.rows || [];
        this.smSummary = slot.summary || { by_metric: [], by_fwd: [], by_ticker: [], by_fwd_ticker: [] };
      } else {
        this.smMeta = { count: 0, tickers: [], metrics: [], fwd_rets: [], avg_score: 0, gte50: 0, gte70: 0, last_run: null };
        this.smRows = [];
        this.smSummary = { by_metric: [], by_fwd: [], by_ticker: [], by_fwd_ticker: [] };
      }
      if (this.smExpanded && this.smMeta.count > 0) {
        this.$nextTick(() => {
          this._renderSmMetricChart();
          this._renderSmFwdChart();
          this._renderSmTickerChart();
          this._renderSmTickerFwdChart();
        });
      }
    },

    // Write a freshly-loaded {meta, rows, summary} into the slot for the
    // current smMode AND point the display at it. status reflects the
    // fetched count.
    _smStoreSlot(meta, rows, summary) {
      this.smDataByMode[this.smMode] = { meta, rows: rows || [], summary: summary || null };
      const cnt = (meta && meta.count) ? meta.count : 0;
      this.smStatusByMode[this.smMode] = cnt > 0 ? 'ready' : 'no_data';
      this._smSwapDisplayFromSlot();
    },

    async smInit() {
      // Initial load fires once on page boot for the default smMode
      // (walk_forward). Subsequent mode switches do NOT auto-fetch —
      // user clicks Refresh. Background-job polling continues as before.
      try {
        const m = this.smMode;
        const cutoffQ = m === 'train_test'
          ? `&cutoff_date=${encodeURIComponent(this.smCutoffDate)}` : '';
        this.smStatusByMode[m] = 'loading';
        const [metaRes, statusRes] = await Promise.all([
          fetch('/api/factor-analysis/score-matrix/meta?mode=' + m + cutoffQ),
          fetch('/api/factor-analysis/batch-score-status'),
        ]);
        let meta = null;
        if (metaRes.ok) meta = await metaRes.json();
        if (statusRes.ok) this.smStatus = await statusRes.json();
        if (meta && meta.count > 0) {
          // loadSmSummary uses smMode + smCutoffDate and writes the slot
          // via _smStoreSlot. It calls loadScoreMatrix internally.
          await this.loadSmSummary();
          setTimeout(() => {
            this._renderSmTickerChart();
            this._renderSmTickerFwdChart();
          }, 150);
        } else if (meta) {
          // No rows for this mode yet — display the placeholder.
          this._smStoreSlot(meta, [], null);
        }
        if (this.smStatus.running) this._smStartPoll();
        if (this.ifStatus.running) this._ifStartPoll();
      } catch (_) {}
    },

    async loadScoreMatrix() {
      const m = this.smMode;
      const params = new URLSearchParams({
        sort_by: this.smSortKey === 'd10_d1_spread' ? 'composite_score' : this.smSortKey,
        order: this.smSortDir,
        min_score: this.smMinScore,
        mode: m,
      });
      if (m === 'train_test') params.set('cutoff_date', this.smCutoffDate);
      if (this.smFilterTicker) params.set('ticker', this.smFilterTicker);
      if (this.smFilterMetric) params.set('metric', this.smFilterMetric);
      if (this.smFilterFwd) params.set('fwd_ret', this.smFilterFwd);

      this.smStatusByMode[m] = 'loading';
      try {
        const r = await fetch('/api/factor-analysis/score-matrix?' + params);
        const rows = r.ok ? await r.json() : [];
        const metaParams = new URLSearchParams({ mode: m });
        if (m === 'train_test') metaParams.set('cutoff_date', this.smCutoffDate);
        const mr = await fetch('/api/factor-analysis/score-matrix/meta?' + metaParams);
        const meta = mr.ok ? await mr.json() : null;
        // Preserve any prior summary in the slot (loadSmSummary writes it).
        const priorSummary = this.smDataByMode[m]?.summary || null;
        this._smStoreSlot(meta, rows, priorSummary);
      } catch (_) {
        this.smStatusByMode[m] = 'error';
      }
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

    // ── Corner Scan ─────────────────────────────────────────────────────────
    // Shared mode-label helper. The 2F and 1F panes own independent
    // mode state but render the same label vocabulary.
    _cornerScanModeLabel(m) {
      if (m === 'in_sample')  return 'In-sample';
      if (m === 'train_test') return 'Train-test';
      return 'Walk-forward';
    },

    // Breadcrumb format identical to the other 5 Bucket A panes:
    //   "last: YYYY-MM-DD HH:MM:SS · <Mode label>"
    // The 2F-side timestamp comes from MAX(scanned_at) over rows with
    // the active mode (returned by /corner-scan/meta?mode=<m>).
    cs2fBreadcrumb() {
      const label = this._cornerScanModeLabel(this.cs2fMode);
      const meta = this.cs2fMeta;
      if (!meta || !meta.scanned_at_2f) {
        return `no data yet · ${label}`;
      }
      const ts = String(meta.scanned_at_2f).slice(0, 19).replace('T', ' ');
      return `last: ${ts} · ${label}`;
    },
    cs1fBreadcrumb() {
      const label = this._cornerScanModeLabel(this.cs1fMode);
      const meta = this.cs1fMeta;
      if (!meta || !meta.scanned_at_1f) {
        return `no data yet · ${label}`;
      }
      const ts = String(meta.scanned_at_1f).slice(0, 19).replace('T', ' ');
      return `last: ${ts} · ${label}`;
    },

    // Mode-pill / cutoff setters — in-memory swap only. Never auto-fetch.
    setCs2fMode(m) {
      if (m === this.cs2fMode && m !== 'train_test') return;
      this.cs2fMode = m;
      this._cs2fSwapDisplayFromSlot();
    },
    setCs2fCutoffDate(d) {
      this.cs2fCutoffDate = d;
    },
    setCs1fMode(m) {
      if (m === this.cs1fMode && m !== 'train_test') return;
      this.cs1fMode = m;
      this._cs1fSwapDisplayFromSlot();
    },
    setCs1fCutoffDate(d) {
      this.cs1fCutoffDate = d;
    },

    // Refresh is supported for walk_forward and in_sample (both built).
    // train_test corner-scan batch does not exist yet — kept disabled.
    cs2fCanRefresh() { return this.cs2fMode !== 'train_test'; },
    cs1fCanRefresh() { return this.cs1fMode !== 'train_test'; },

    // Pull the active mode's slot into the top-level rows/meta/total
    // fields the template binds to. Empty slot → empty state +
    // placeholder.
    _cs2fSwapDisplayFromSlot() {
      const slot = this.cs2fDataByMode[this.cs2fMode];
      if (slot) {
        this.cs2fMeta  = slot.meta;
        this.cs2fRows  = slot.rows  || [];
        this.cs2fTotal = slot.total || 0;
        this.cs2fPage  = slot.page  || 1;  // restore the page that was last fetched for this mode
      } else {
        this.cs2fMeta  = null;
        this.cs2fRows  = [];
        this.cs2fTotal = 0;
        this.cs2fPage  = 1;
      }
    },
    _cs1fSwapDisplayFromSlot() {
      const slot = this.cs1fDataByMode[this.cs1fMode];
      if (slot) {
        this.cs1fMeta  = slot.meta;
        this.cs1fRows  = slot.rows  || [];
        this.cs1fTotal = slot.total || 0;
        this.cs1fPage  = slot.page  || 1;
      } else {
        this.cs1fMeta  = null;
        this.cs1fRows  = [];
        this.cs1fTotal = 0;
        this.cs1fPage  = 1;
      }
    },

    _cs2fStoreSlot(meta, rows, total) {
      this.cs2fDataByMode[this.cs2fMode] = { meta, rows: rows || [], total: total || 0, page: this.cs2fPage };
      this._cs2fSwapDisplayFromSlot();
    },
    _cs1fStoreSlot(meta, rows, total) {
      this.cs1fDataByMode[this.cs1fMode] = { meta, rows: rows || [], total: total || 0, page: this.cs1fPage };
      this._cs1fSwapDisplayFromSlot();
    },

    async toggleCs2f() {
      this.cs2fExpanded = !this.cs2fExpanded;
      // First-expand auto-load for any mode with data (WF + IS).
      // TT skips because cs2fCanRefresh() returns false for it.
      if (this.cs2fExpanded
          && !this.cs2fDataByMode[this.cs2fMode]
          && this.cs2fCanRefresh()) {
        await this.loadCs2f();
      }
    },
    // resetPage=true (default): resets to page 1 — use for Refresh, sort, filter changes.
    // resetPage=false: keeps current page — use for prev/next/goto navigation.
    async loadCs2f(resetPage = true) {
      if (!this.cs2fCanRefresh()) return;
      if (resetPage) this.cs2fPage = 1;
      this.cs2fLoading = true;
      const cutoffQ = this.cs2fMode === 'train_test'
        ? `&cutoff_date=${encodeURIComponent(this.cs2fCutoffDate)}` : '';
      // Meta: counts + scanned_at for the breadcrumb. Always re-fetched
      // on Refresh so the timestamp reflects the latest scan.
      let meta = null;
      try {
        const r = await fetch(`/api/factor-analysis/corner-scan/meta?mode=${this.cs2fMode}${cutoffQ}`);
        if (r.ok) { meta = await r.json(); if (meta?.metrics?.length) this.csMetrics = meta.metrics; }
      } catch (_) {}
      const p = new URLSearchParams({
        sort_key: this.cs2fSortKey, sort_dir: this.cs2fSortDir,
        min_d_n:  this.cs2fMinN,   limit:    50,
        offset:   (this.cs2fPage - 1) * 50,
        mode:     this.cs2fMode,
      });
      if (this.cs2fMode === 'train_test') p.set('cutoff_date', this.cs2fCutoffDate);
      if (this.cs2fFilterP)       p.set('primary_metric',   this.cs2fFilterP);
      if (this.cs2fFilterS)       p.set('secondary_metric', this.cs2fFilterS);
      if (this.cs2fFilterDir)     p.set('corner_direction', this.cs2fFilterDir);
      if (this.cs2fFilterOutcome) p.set('outcome',          this.cs2fFilterOutcome);
      let rows = [], total = 0;
      try {
        const r = await fetch('/api/factor-analysis/corner-scan/2f?' + p);
        if (r.ok) { const d = await r.json(); rows = d.rows || []; total = d.total || 0; }
      } catch (_) {}
      this._cs2fStoreSlot(meta, rows, total);
      this.cs2fLoading = false;
    },
    cs2fSort(key) {
      if (this.cs2fSortKey === key) {
        this.cs2fSortDir = this.cs2fSortDir === 'desc' ? 'asc' : 'desc';
      } else {
        this.cs2fSortKey = key;
        this.cs2fSortDir = 'desc';
      }
      // Sort change resets to page 1 and re-fetches.
      if (this.cs2fCanRefresh()) this.loadCs2f();
    },
    cs2fTotalPages() { return Math.max(1, Math.ceil(this.cs2fTotal / 50)); },
    cs2fPrevPage()   { if (this.cs2fPage > 1) { this.cs2fPage--; this.loadCs2f(false); } },
    cs2fNextPage()   { if (this.cs2fPage < this.cs2fTotalPages()) { this.cs2fPage++; this.loadCs2f(false); } },
    cs2fGoToPage(n)  {
      const pg = Math.max(1, Math.min(this.cs2fTotalPages(), parseInt(n) || 1));
      if (pg !== this.cs2fPage) { this.cs2fPage = pg; this.loadCs2f(false); }
    },

    async toggleCs1f() {
      this.cs1fExpanded = !this.cs1fExpanded;
      if (this.cs1fExpanded
          && !this.cs1fDataByMode[this.cs1fMode]
          && this.cs1fCanRefresh()) {
        await this.loadCs1f();
      }
    },
    async loadCs1f(resetPage = true) {
      if (!this.cs1fCanRefresh()) return;
      if (resetPage) this.cs1fPage = 1;
      this.cs1fLoading = true;
      const cutoffQ = this.cs1fMode === 'train_test'
        ? `&cutoff_date=${encodeURIComponent(this.cs1fCutoffDate)}` : '';
      let meta = null;
      try {
        const r = await fetch(`/api/factor-analysis/corner-scan/meta?mode=${this.cs1fMode}${cutoffQ}`);
        if (r.ok) { meta = await r.json(); if (meta?.metrics?.length) this.csMetrics = meta.metrics; }
      } catch (_) {}
      const p = new URLSearchParams({
        sort_key: this.cs1fSortKey, sort_dir: this.cs1fSortDir,
        min_d_n:  this.cs1fMinN,   limit:    50,
        offset:   (this.cs1fPage - 1) * 50,
        mode:     this.cs1fMode,
      });
      if (this.cs1fMode === 'train_test') p.set('cutoff_date', this.cs1fCutoffDate);
      if (this.cs1fFilterMetric)  p.set('metric',   this.cs1fFilterMetric);
      if (this.cs1fFilterExtreme) p.set('extreme',  this.cs1fFilterExtreme);
      if (this.cs1fFilterOutcome) p.set('outcome',  this.cs1fFilterOutcome);
      let rows = [], total = 0;
      try {
        const r = await fetch('/api/factor-analysis/corner-scan/1f?' + p);
        if (r.ok) { const d = await r.json(); rows = d.rows || []; total = d.total || 0; }
      } catch (_) {}
      this._cs1fStoreSlot(meta, rows, total);
      this.cs1fLoading = false;
    },
    cs1fSort(key) {
      if (this.cs1fSortKey === key) {
        this.cs1fSortDir = this.cs1fSortDir === 'desc' ? 'asc' : 'desc';
      } else {
        this.cs1fSortKey = key;
        this.cs1fSortDir = 'desc';
      }
      if (this.cs1fCanRefresh()) this.loadCs1f();
    },
    cs1fTotalPages() { return Math.max(1, Math.ceil(this.cs1fTotal / 50)); },
    cs1fPrevPage()   { if (this.cs1fPage > 1) { this.cs1fPage--; this.loadCs1f(false); } },
    cs1fNextPage()   { if (this.cs1fPage < this.cs1fTotalPages()) { this.cs1fPage++; this.loadCs1f(false); } },
    cs1fGoToPage(n)  {
      const pg = Math.max(1, Math.min(this.cs1fTotalPages(), parseInt(n) || 1));
      if (pg !== this.cs1fPage) { this.cs1fPage = pg; this.loadCs1f(false); }
    },

    async runBatchScore() {
      try {
        const m = this.smMode;
        const r = await fetch('/api/factor-analysis/run-batch-score', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            walk_forward: m === 'walk_forward',
            cutoff_date:  m === 'train_test' ? this.smCutoffDate : null,
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
          const r = await fetch('/api/factor-analysis/batch-score-status');
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
      const m = this.smMode;
      const params = new URLSearchParams({ mode: m });
      if (m === 'train_test') params.set('cutoff_date', this.smCutoffDate);
      if (metric) params.set('metric', metric);
      if (fwdRet) params.set('fwd_ret', fwdRet);
      if (ticker) params.set('ticker', ticker);
      try {
        const r = await fetch('/api/factor-analysis/score-matrix/summary?' + params);
        if (r.ok) {
          this.smSummary = await r.json();
          // Persist into the active mode's slot so a mode-switch swap
          // restores the same summary without re-fetching.
          if (this.smDataByMode[m]) {
            this.smDataByMode[m].summary = this.smSummary;
          }
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
        const r = await fetch('/api/factor-analysis/feature-clusters');
        if (r.ok) this.ifClusters = await r.json();
      } catch (_) {}
    },

    // ── 2F Interaction Scanner ────────────────────────────────────────
    async run2fScan() {
      const metrics = [...this.$store.metricPicker.selected];
      if (metrics.length < 2) return;
      this.ifLastScannedMetrics = metrics;
      try {
        const r = await fetch('/api/factor-analysis/run-2f-scan', {
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
          const r = await fetch('/api/factor-analysis/2f-scan-status');
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
        const r = await fetch('/api/factor-analysis/interaction-matrix?' + params);
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
        const r = await fetch('/api/factor-analysis/interaction-detail?' + params);
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

    _applySecResults(d) {
      this.secBaseline = d.baseline;
      this.secMetrics  = d.metrics || [];
      this.secMaxAbsScore = Math.max(0.0001, ...this.secMetrics.map(m => Math.abs(m.score)));
      this.secScanMeta = {
        mode:             d.mode || 'walk_forward',
        universe_n:       d.universe_n,
        start_date:       d.start_date,
        data_as_of:       d.data_as_of || null,
      };
      this.secStatus = { loaded: true, loading: false, error: null };
      this.secScannerStale = false;  // M1: fresh scan results clear stale flag
      this.$nextTick(() => this.$nextTick(() => setTimeout(() => this._renderSecBar(), 60)));
    },

    // ── Zone Analyze methods ──────────────────────────────────────────────────

    toggleHmCell(ix, iy) {
      // Only active in IS mode; WF/TT cells are view-only.
      if (this.pageMode !== 'in_sample') return;
      const key = `${ix}-${iy}`;
      const next = new Set(this.hmSelectedCells);
      if (next.has(key)) next.delete(key);
      else               next.add(key);
      this.hmSelectedCells = next;
      // Clear stale zone result when selection changes
      this.zoneData = null;
      this._destroyZoneCharts();
    },

    isHmCellSelected(ix, iy) {
      return this.hmSelectedCells.has(`${ix}-${iy}`);
    },

    clearHmSelection() {
      this.hmSelectedCells = new Set();
      this.zoneData = null;
      this.zoneOpen = false;
      this._destroyZoneCharts();
    },

    // Returns [[ix, iy], ...] for API calls
    hmCellSet() {
      return [...this.hmSelectedCells].map(k => k.split('-').map(Number));
    },

    async analyzeZone() {
      if (!this.hmSelectedCells.size) return;
      if (!this.metric || !this.secSelectedMetric) return;
      this.zoneLoading = true;
      this.zoneData = null;
      this._destroyZoneCharts();
      this.zoneOpen = true;
      try {
        const r = await fetch('/api/factor-analysis/secondary-zone-analyze', {
          method:  'POST',
          headers: { 'Content-Type': 'application/json' },
          body:    JSON.stringify({
            primary_metric:   this.metric,
            secondary_metric: this.secSelectedMetric,
            outcome:          this.zoneOutcome || this.outcome || 'ret_5d_fwd_oc',
            n_bins:           this.heatmapBins,
            cell_set:         this.hmCellSet(),
            ticker:           this.ticker,
            date_from:        this.dateFrom  || null,
            date_to:          this.dateTo    || null,
          }),
        });
        if (r.ok) {
          this.zoneData = await r.json();
          await this.$nextTick();
          await this.$nextTick();
          setTimeout(() => this._renderZoneCharts(), 60);
        } else {
          this.zoneData = { error: `HTTP ${r.status}` };
        }
      } catch (e) {
        this.zoneData = { error: e.message };
      }
      this.zoneLoading = false;
    },

    _destroyZoneCharts() {
      for (const k of ['zone-equity', 'zone-yearly', 'zone-activity', 'zone-bubble']) {
        if (this._charts[k]) { this._charts[k].destroy(); delete this._charts[k]; }
      }
    },

    _renderZoneCharts() {
      if (!this.zoneData || this.zoneData.error) return;
      this._renderSecEquity('chart-zone-equity', this.zoneData, true);
      this._renderZoneYearly('chart-zone-yearly', this.zoneData);
      this._renderSecActivity('chart-zone-activity', this.zoneData);
      this._renderSecBubble('chart-zone-bubble', this.zoneData);
    },

    // Parameterized so the Recall view can call with its own canvas ID
    // and data source without reimplementing the chart. Main flow keeps
    // the same defaults; nothing else changes.
    _renderZoneYearly(canvasId, data) {
      canvasId = canvasId || 'chart-zone-yearly';
      const src = data || this.zoneData;
      const canvas = document.getElementById(canvasId);
      if (!canvas || !src || !src.yearly || !src.yearly.length) return;
      // Chart-key derived from canvasId so main ('zone-yearly') and
      // recall ('recall-yearly') don't collide in this._charts.
      const chartKey = canvasId.replace(/^chart-/, '');
      if (this._charts[chartKey]) { this._charts[chartKey].destroy(); delete this._charts[chartKey]; }
      const yearly = src.yearly;
      // n-count gradient: dim bars for thin years, vivid for well-populated ones
      const ns = yearly.map(y => y.n);
      const minN = Math.min(...ns), maxN = Math.max(...ns);
      const nPct = y => maxN > minN ? (y.n - minN) / (maxN - minN) : 1;
      const alpha = y => (0.2 + nPct(y) * 0.6).toFixed(2);
      const bgColor = y => y.avg_ret >= 0
        ? `rgba(52,152,219,${alpha(y)})` : `rgba(232,67,147,${alpha(y)})`;
      const borderColor = y => y.avg_ret >= 0 ? '#3498db' : '#e84393';
      const ctx = canvas.getContext('2d');
      this._charts[chartKey] = new Chart(ctx, {
        type: 'bar',
        data: {
          labels:   yearly.map(y => y.year),
          datasets: [{
            label:           'Avg Ret',
            data:            yearly.map(y => +(y.avg_ret * 100).toFixed(3)),
            backgroundColor: yearly.map(bgColor),
            borderColor:     yearly.map(borderColor),
            borderWidth:     1,
          }],
        },
        options: {
          responsive: true, maintainAspectRatio: false, animation: false,
          plugins: {
            legend: { display: false },
            tooltip: {
              backgroundColor: 'rgba(20,20,20,0.95)', borderColor: '#444', borderWidth: 1,
              callbacks: {
                label: ctx => {
                  const y = yearly[ctx.dataIndex];
                  return [`Avg: ${(y.avg_ret*100).toFixed(3)}%`, `WR: ${(y.win_rate*100).toFixed(1)}%`, `n: ${y.n}`];
                },
              },
            },
          },
          scales: {
            x: { ticks: { color: '#888', font: { size: 9 } }, grid: { color: '#222' } },
            y: { ticks: { color: '#888', font: { size: 9 }, callback: v => v.toFixed(2) + '%' },
                 grid:  { color: '#222' } },
          },
        },
      });
    },

    // _renderZoneEquity, _renderZoneActivity, _renderZoneBubble removed —
    // zone charts now delegate to the parameterized _renderSecEquity /
    // _renderSecActivity / _renderSecBubble (identical visuals, zone data source).

    zoneDownloadCSV() {
      if (!this.zoneData?.tickers?.length) return;
      const rows = [['ticker', 'n', 'avg_ret', 'win_rate', 'contrib_pct']];
      for (const t of this.zoneData.tickers) {
        rows.push([t.ticker, t.n, (t.avg_ret*100).toFixed(4)+'%',
                   (t.win_rate*100).toFixed(2)+'%', t.contrib_pct.toFixed(2)+'%']);
      }
      // Also include equity curve
      const eq = this.zoneData.equity_primary || [];
      if (eq.length) {
        rows.push([]);
        rows.push(['date', 'cum_ret_pct']);
        for (const p of eq) rows.push([p.date, (p.value*100).toFixed(4)+'%']);
      }
      const csv = rows.map(r => r.join(',')).join('\n');
      const blob = new Blob([csv], { type: 'text/csv' });
      const a = document.createElement('a');
      a.href = URL.createObjectURL(blob);
      a.download = `zone_${this.metric}_${this.secSelectedMetric}_${this.zoneOutcome || this.outcome}.csv`;
      a.click();
      URL.revokeObjectURL(a.href);
    },

    // ── Signals CRUD + table helpers ─────────────────────────────────────────

    async loadSignals() {
      const r = await fetch('/api/factor-analysis/signals');
      if (r.ok) this.signals = (await r.json()).signals || [];
    },

    async saveSignal() {
      if (!this.signalName.trim() || !this.hmSelectedCells.size) return;
      if (!this.metric || !this.secSelectedMetric) return;
      this.signalSaving = true;
      this.signalSaveMsg = '';
      try {
        const r = await fetch('/api/factor-analysis/signals', {
          method:  'POST',
          headers: { 'Content-Type': 'application/json' },
          body:    JSON.stringify({
            name:             this.signalName.trim(),
            primary_metric:   this.metric,
            secondary_metric: this.secSelectedMetric,
            outcome:          this.zoneOutcome || this.outcome || 'ret_5d_fwd_oc',
            n_bins:           this.heatmapBins,
            cell_set:         this.hmCellSet(),
          }),
        });
        if (r.ok) {
          this.signalSaveMsg = '✓ Saved';
          this.signalName    = '';
          // POST now computes stats inline, so reloading shows them right
          // away without an extra Refresh.
          await this.loadSignals();
        } else {
          this.signalSaveMsg = '✗ Error';
        }
      } catch (_) {
        this.signalSaveMsg = '✗ Error';
      }
      this.signalSaving = false;
      setTimeout(() => { this.signalSaveMsg = ''; }, 3000);
    },

    // ── Sort dispatcher (Saved Signals table) ──────────────────────────

    sigSortBy(key) {
      if (this.sigSortKey === key) {
        this.sigSortDir = this.sigSortDir === 'asc' ? 'desc' : 'asc';
      } else {
        this.sigSortKey = key;
        // String / date columns default asc; numeric defaults desc
        // (largest first — best avg_ret, biggest n, freshest stats date).
        const ASC_DEFAULT = new Set([
          'name', 'primary', 'primary_family',
          'secondary', 'secondary_family', 'outcome',
        ]);
        this.sigSortDir = ASC_DEFAULT.has(key) ? 'asc' : 'desc';
      }
    },
    sigSortClass(key) {
      return this.sigSortKey === key ? 'sort-active' : '';
    },
    sigSortArrow(key) {
      if (this.sigSortKey !== key) return '';
      return this.sigSortDir === 'asc' ? '▲' : '▼';
    },
    _sigSortValueFor(s, key) {
      switch (key) {
        case 'name':             return (s.name             || '').toLowerCase();
        case 'primary':          return (s.primary_metric   || '').toLowerCase();
        case 'primary_family':   return (this.signalFamilyName(s.primary_metric)   || '').toLowerCase();
        case 'secondary':        return (s.secondary_metric || '').toLowerCase();
        case 'secondary_family': return (this.signalFamilyName(s.secondary_metric) || '').toLowerCase();
        case 'outcome':          return (s.outcome          || '').toLowerCase();
        case 'agg_avg_ret':      return s.agg_avg_ret;
        case 'agg_n':            return s.agg_n;
        case 'stats_updated_at': return s.stats_updated_at;
        default:                 return 0;
      }
    },
    get sortedSignals() {
      const arr = (this.signals || []).slice();
      const key = this.sigSortKey;
      const dir = this.sigSortDir === 'asc' ? 1 : -1;
      // Nulls sink to the bottom regardless of direction — a missing
      // stat is "no data," not a tiny number.
      const isMissing = (v) =>
        v === null || v === undefined ||
        (typeof v === 'number' && !isFinite(v));
      arr.sort((a, b) => {
        const va = this._sigSortValueFor(a, key);
        const vb = this._sigSortValueFor(b, key);
        const ma = isMissing(va), mb = isMissing(vb);
        if (ma && mb) return 0;
        if (ma) return 1;
        if (mb) return -1;
        if (typeof va === 'string' || typeof vb === 'string') {
          return String(va).localeCompare(String(vb)) * dir;
        }
        return (va - vb) * dir;
      });
      return arr;
    },

    // ── Checkbox + batch actions ───────────────────────────────────────

    sigCheckedCount() {
      let n = 0;
      for (const k in this.sigCheckedIds) if (this.sigCheckedIds[k]) n++;
      return n;
    },
    sigAllChecked() {
      return this.signals.length > 0 &&
             this.signals.every(s => !!this.sigCheckedIds[s.id]);
    },
    sigToggle(id) {
      this.sigCheckedIds = { ...this.sigCheckedIds, [id]: !this.sigCheckedIds[id] };
    },
    sigToggleAll(checked) {
      const next = {};
      if (checked) for (const s of this.signals) next[s.id] = true;
      this.sigCheckedIds = next;
    },
    sigToggleExpand(id) {
      this.sigExpanded = { ...this.sigExpanded, [id]: !this.sigExpanded[id] };
    },

    async refreshSelectedSignals() {
      const ids = Object.keys(this.sigCheckedIds)
        .filter(k => this.sigCheckedIds[k]).map(Number);
      if (!ids.length) return;
      this.sigBatchBusy = true;
      this.sigBatchMsg  = 'Refreshing…';
      try {
        const r = await fetch('/api/factor-analysis/signals/refresh', {
          method:  'POST',
          headers: { 'Content-Type': 'application/json' },
          body:    JSON.stringify({ signal_ids: ids }),
        });
        if (r.ok) {
          const { refreshed, failed } = await r.json();
          const failN = (failed && failed.length) || 0;
          this.sigBatchMsg = `Refreshed ${refreshed} signal${refreshed === 1 ? '' : 's'}` +
                             (failN ? ` · ${failN} failed` : '');
          await this.loadSignals();
        } else {
          this.sigBatchMsg = '✗ Refresh failed';
        }
      } catch (_) {
        this.sigBatchMsg = '✗ Network error';
      }
      this.sigBatchBusy = false;
      setTimeout(() => { this.sigBatchMsg = ''; }, 4000);
    },

    async deleteSelectedSignals() {
      const ids = Object.keys(this.sigCheckedIds)
        .filter(k => this.sigCheckedIds[k]).map(Number);
      if (!ids.length) return;
      if (!confirm(`Delete ${ids.length} signal${ids.length === 1 ? '' : 's'}?`)) return;
      this.sigBatchBusy = true;
      this.sigBatchMsg  = 'Deleting…';
      try {
        const r = await fetch('/api/factor-analysis/signals/delete-batch', {
          method:  'POST',
          headers: { 'Content-Type': 'application/json' },
          body:    JSON.stringify({ signal_ids: ids }),
        });
        if (r.ok) {
          const { deleted } = await r.json();
          this.sigBatchMsg = `Deleted ${deleted} signal${deleted === 1 ? '' : 's'}`;
          this.sigCheckedIds = {};
          await this.loadSignals();
        } else {
          this.sigBatchMsg = '✗ Delete failed';
        }
      } catch (_) {
        this.sigBatchMsg = '✗ Network error';
      }
      this.sigBatchBusy = false;
      setTimeout(() => { this.sigBatchMsg = ''; }, 4000);
    },

    // ── Display helpers ─────────────────────────────────────────────────

    signalFamilyName(metric) {
      const fam = this.metricFamilyLookup[metric];
      return fam ? fam.family_name : '—';
    },
    sigTruncateMetric(name) {
      // Long metric names like oi_weighted_all_div_spot_co get clipped
      // for column width; full name shows in the cell's title attr.
      if (!name) return '';
      return name.length > 24 ? name.slice(0, 22) + '…' : name;
    },
    sigFormatPct(v) {
      if (v === null || v === undefined) return '—';
      const x = +v;
      if (!isFinite(x)) return '—';
      return (x * 100).toFixed(2) + '%';
    },
    sigPosNeg(v) {
      if (v === null || v === undefined) return '';
      const x = +v;
      if (!isFinite(x) || x === 0) return '';
      return x > 0 ? 'pos' : 'neg';
    },
    sigSmallN(n) {
      return (typeof n === 'number' && n < 30) ? 'small-n' : '';
    },

    // Per-cell color on a fixed ±3% divergent scale — same canonical
    // hex anchors as .pos / .neg everywhere else on the page (no
    // per-signal or per-global max_abs calibration). 4-5% cells
    // saturate to max intensity; a cell's color means the same return
    // in every thumbnail.
    cellColor(avgRet) {
      if (avgRet === null || avgRet === undefined) return 'rgb(60,60,60)';
      const r = Math.max(-0.03, Math.min(0.03, +avgRet));
      const t = r / 0.03;   // -1 → +1
      // Neutral rgb(60,60,60). Blue end #3498db (52,152,219). Pink end #e84393 (232,67,147).
      let R, G, B;
      if (t > 0) {
        R = Math.round(60 + (52  - 60) * t);
        G = Math.round(60 + (152 - 60) * t);
        B = Math.round(60 + (219 - 60) * t);
      } else if (t < 0) {
        const u = -t;
        R = Math.round(60 + (232 - 60) * u);
        G = Math.round(60 + (67  - 60) * u);
        B = Math.round(60 + (147 - 60) * u);
      } else {
        return 'rgb(60,60,60)';
      }
      return `rgb(${R},${G},${B})`;
    },
    cellOpacity(n) {
      // Linear small-n dimming: 0.35 below 10, 1.0 at 100+, smooth
      // between. Same honesty cue as the firing page's small-n dim.
      if (n === null || n === undefined || n < 10) return 0.35;
      if (n >= 100) return 1.0;
      return 0.35 + 0.65 * (n - 10) / 90;
    },

    sortedPerCellStats(sig) {
      return ((sig && sig.per_cell_stats) || [])
        .slice()
        .sort((a, b) => (b.avg_ret || 0) - (a.avg_ret || 0));
    },

    // ── Recall / Edit flow ──────────────────────────────────────────────
    //
    // Clicking a saved signal opens it in the dedicated Recall view
    // (own heatmap + four zone visuals, separate from the main flow).
    // The recall path skips the slow primary-analyze + secondary-scan
    // chains; both data calls (/heatmap and /secondary-zone-analyze)
    // read directly from stored bins and run in parallel — ~10s total.
    // Editing happens on the recall heatmap; Save updates the signal
    // in place via PUT /signals/{id}, which recomputes stats inline.

    async recallSignal(sig) {
      if (!sig) return;
      // Dirty check uses set-equality (order-independent) so that
      // toggling a cell off then back on, or the same cells in a
      // different array order, does NOT register as a dirty edit.
      if (this.recallSig && this.recallSig.id !== sig.id && this._recallIsDirty()) {
        const msg = `Discard unsaved edits on "${this.recallSig.name}"?`;
        if (!confirm(msg)) return;
      }
      this.recallSig = sig;
      this.recallEditedName = sig.name || '';
      this.recallSelectedCells = new Set(
        (sig.cell_set || []).map(c => c[0] + '-' + c[1]));
      this.recallExpanded = true;
      await this._recallFireRequests();
    },

    async _recallFireRequests() {
      if (!this.recallSig) return;
      const sig = this.recallSig;
      this.recallLoading = true;
      this.recallHeatmapData = null;
      this.recallZoneData = null;
      this._destroyRecallCharts();
      try {
        // Two parallel fetches — neither depends on the other or on
        // the slow main-flow preamble. Heatmap reads is_bins directly,
        // zone-analyze reads is_bins + daily_features for the
        // cell-set. Both ~5-10s; parallel = the slower of the two.
        const heatmapUrl =
          `/api/factor-analysis/heatmap?ticker=ALL`
          + `&metric_x=${encodeURIComponent(sig.primary_metric)}`
          + `&metric_y=${encodeURIComponent(sig.secondary_metric)}`
          + `&outcome=${encodeURIComponent(sig.outcome)}`
          + `&bins=${sig.n_bins}`;
        const [hmResp, zoneResp] = await Promise.all([
          fetch(heatmapUrl),
          fetch('/api/factor-analysis/secondary-zone-analyze', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              primary_metric:   sig.primary_metric,
              secondary_metric: sig.secondary_metric,
              outcome:          sig.outcome,
              n_bins:           sig.n_bins,
              cell_set:         sig.cell_set || [],
              ticker:           'ALL',
            }),
          }),
        ]);
        if (hmResp.ok)   this.recallHeatmapData = await hmResp.json();
        if (zoneResp.ok) this.recallZoneData    = await zoneResp.json();
      } catch (e) {
        console.error('Recall fetch failed', e);
      }
      this.recallLoading = false;
      await this.$nextTick();
      setTimeout(() => this._renderRecallCharts(), 60);
    },

    async recallReanalyze() {
      // Re-run zone-analyze on the CURRENT recall cells (after the
      // user has edited the selection). Doesn't touch the heatmap
      // grid — the grid is fixed by metric pair.
      if (!this.recallSig || !this.recallSelectedCells.size) return;
      const sig = this.recallSig;
      this.recallLoading = true;
      this.recallZoneData = null;
      this._destroyRecallCharts();
      try {
        const r = await fetch('/api/factor-analysis/secondary-zone-analyze', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            primary_metric:   sig.primary_metric,
            secondary_metric: sig.secondary_metric,
            outcome:          sig.outcome,
            n_bins:           sig.n_bins,
            cell_set:         this.recallCellSet(),
            ticker:           'ALL',
          }),
        });
        if (r.ok) this.recallZoneData = await r.json();
      } catch (_) {}
      this.recallLoading = false;
      await this.$nextTick();
      setTimeout(() => this._renderRecallCharts(), 60);
    },

    async recallSave() {
      if (!this.recallSig || !this.recallSelectedCells.size) return;
      const trimmedName = (this.recallEditedName || '').trim();
      if (!trimmedName) {
        this.recallSaveMsg = '✗ Name required';
        setTimeout(() => { this.recallSaveMsg = ''; }, 3000);
        return;
      }
      this.recallSaving = true;
      this.recallSaveMsg = '';
      try {
        // Include name in the body whenever it differs from the saved
        // signal's name (server only writes it when present). Sending
        // the unchanged name is also fine — backend uses COALESCE.
        const body = { cell_set: this.recallCellSet() };
        if (trimmedName !== this.recallSig.name) body.name = trimmedName;
        const r = await fetch(
          `/api/factor-analysis/signals/${this.recallSig.id}`,
          {
            method:  'PUT',
            headers: { 'Content-Type': 'application/json' },
            body:    JSON.stringify(body),
          });
        if (r.ok) {
          const updated = await r.json();
          // Update the in-memory recallSig so subsequent dirty checks
          // compare against the just-saved cell_set + name (no
          // false-positive 'unsaved edits' confirmation right after).
          this.recallSig = {
            ...this.recallSig,
            name:             updated.name,
            cell_set:         updated.cell_set,
            agg_avg_ret:      updated.agg_avg_ret,
            agg_n:            updated.agg_n,
            per_cell_stats:   updated.per_cell_stats,
            stats_updated_at: updated.stats_updated_at,
          };
          this.recallEditedName = updated.name;
          this.recallSaveMsg = '✓ Saved';
          // Refresh the Saved Signals list so the row's stats,
          // thumbnail, and name reflect the edit immediately.
          await this.loadSignals();
        } else {
          this.recallSaveMsg = '✗ Save failed';
        }
      } catch (_) {
        this.recallSaveMsg = '✗ Network error';
      }
      this.recallSaving = false;
      setTimeout(() => { this.recallSaveMsg = ''; }, 4000);
    },

    recallCancel() {
      this.recallSig = null;
      this.recallEditedName = '';
      this.recallExpanded = false;
      this.recallHeatmapData = null;
      this.recallZoneData = null;
      this.recallSelectedCells = new Set();
      this._destroyRecallCharts();
    },

    // Heatmap cell handlers — recall-namespaced. Same "ix-iy" key
    // format as hmSelectedCells so the set-equality dirty check
    // compares apples to apples.
    toggleRecallCell(ix, iy) {
      const key = `${ix}-${iy}`;
      const next = new Set(this.recallSelectedCells);
      if (next.has(key)) next.delete(key);
      else               next.add(key);
      this.recallSelectedCells = next;
    },
    isRecallCellSelected(ix, iy) {
      return this.recallSelectedCells.has(`${ix}-${iy}`);
    },
    clearRecallSelection() {
      this.recallSelectedCells = new Set();
    },
    recallCellSet() {
      return [...this.recallSelectedCells].map(k => k.split('-').map(Number));
    },

    // Dirty if either the cell set or the name has been edited since
    // the last save. Cell-set check is order-independent set equality;
    // name check is whitespace-trimmed string compare. Toggling a cell
    // off then back on, or the same cells in a different array order,
    // or an unchanged name with surrounding whitespace — none of these
    // register as dirty.
    _recallIsDirty() {
      if (!this.recallSig) return false;
      const editedName = (this.recallEditedName || '').trim();
      const savedName  = (this.recallSig.name    || '').trim();
      if (editedName !== savedName) return true;
      const saved = new Set(
        (this.recallSig.cell_set || []).map(c => c[0] + '-' + c[1]));
      const current = this.recallSelectedCells;
      if (saved.size !== current.size) return true;
      for (const k of saved) if (!current.has(k)) return true;
      return false;
    },

    _destroyRecallCharts() {
      for (const k of ['recall-equity', 'recall-yearly',
                       'recall-activity', 'recall-bubble']) {
        if (this._charts[k]) {
          this._charts[k].destroy();
          delete this._charts[k];
        }
      }
    },

    _renderRecallCharts() {
      // Delegates to the SAME chart renderers the main zone view uses
      // — different canvas ids and data source, identical visuals.
      if (!this.recallZoneData || this.recallZoneData.error) return;
      this._renderSecEquity('chart-recall-equity',   this.recallZoneData, true);
      this._renderZoneYearly('chart-recall-yearly',  this.recallZoneData);
      this._renderSecActivity('chart-recall-activity', this.recallZoneData);
      this._renderSecBubble('chart-recall-bubble',   this.recallZoneData);
    },

    // Build the thumbnail SVG as a markup string. We can't use Alpine's
    // <template x-for> inside an <svg> element because the browser parses
    // <template> content in HTML mode, and the cloned <rect> / <line>
    // children end up in the HTML namespace, not SVG — they look correct
    // in the DOM tree but never paint. Building the SVG as a string and
    // injecting via x-html lets innerHTML's inline-SVG parser place
    // every node in the SVG namespace correctly.
    signalThumbnailSVG(sig) {
      const n     = (sig && sig.n_bins) || 0;
      const step  = n > 0 ? 20 / n : 20;
      const cells = (sig && sig.per_cell_stats) || [];

      const parts = [
        '<svg width="110" height="110" viewBox="0 0 20 20"'
          + ' style="display:block;background:#1a1a1a;cursor:pointer">',
        // Frame
        '<rect x="0" y="0" width="20" height="20" fill="none"'
          + ' stroke="rgba(255,255,255,.18)" stroke-width="0.08"/>',
      ];
      // Gridlines at the signal's own n_bins resolution — viewer can
      // see whether they're looking at a 3-bin, 5-bin, 10-bin or 20-bin
      // signal just from the gridline density.
      for (let i = 1; i < n; i++) {
        const pos = (i * step).toFixed(4);
        parts.push(
          `<line x1="${pos}" y1="0" x2="${pos}" y2="20"`
            + ' stroke="rgba(255,255,255,.08)" stroke-width="0.05"/>',
          `<line x1="0" y1="${pos}" x2="20" y2="${pos}"`
            + ' stroke="rgba(255,255,255,.08)" stroke-width="0.05"/>'
        );
      }
      // Selected cells. No Y flip — the main 2D heatmap renders in
      // SVG-native orientation (y=0 at top, high secondary-bin rows
      // at the bottom). High iy goes to high SVG y so a zone in the
      // thumbnail visually matches the same zone on the heatmap.
      for (const cell of cells) {
        const x       = (cell.ix * step).toFixed(4);
        const y       = (cell.iy * step).toFixed(4);
        const w       = step.toFixed(4);
        const fill    = this.cellColor(cell.avg_ret);
        const opacity = this.cellOpacity(cell.n).toFixed(3);
        parts.push(
          `<rect x="${x}" y="${y}" width="${w}" height="${w}"`
            + ` fill="${fill}" fill-opacity="${opacity}"`
            + ' stroke="rgba(255,255,255,.30)" stroke-width="0.06"/>'
        );
      }
      parts.push('</svg>');
      return parts.join('');
    },

    // Group a flat metric list by family using metricFamilyLookup.
    // Works for any array of metric-name strings (features, csMetrics, smMeta.metrics, …).
    // keepOrder=false  → groups sorted by family_num (default; use for static lists).
    // keepOrder=true   → groups appear in first-occurrence order from `list`
    //                    (use for score-sorted lists so hottest family appears first).
    groupMetricsByFamily(list, keepOrder = false) {
      const groups = new Map();  // family_num → {family_num, family_name, metrics:[]}
      for (const m of (list || [])) {
        const fam = this.metricFamilyLookup[m];
        const key = fam ? fam.family_num : 999;
        const label = fam ? fam.family_name : 'Other';
        if (!groups.has(key)) groups.set(key, { family_num: key, family_name: label, metrics: [] });
        groups.get(key).metrics.push(m);
      }
      // Sort metrics alphabetically within each family group.
      // Use explicit localeCompare so the sort is unambiguous regardless of
      // whether Alpine's reactivity layer has wrapped the string elements.
      if (!keepOrder) for (const g of groups.values())
        g.metrics.sort((a, b) => String(a).localeCompare(String(b)));
      // DIAG: unconditional — fires on every call to confirm function is reached
      // and to expose what keys the Map holds (number vs string) and whether F3
      // metrics land in key 3 or somewhere else.
      console.log('[DIAG groupMetricsByFamily] called — list.length:', (list||[]).length,
        '| keepOrder:', keepOrder,
        '| groupKeys:', [...groups.keys()],
        '| F3 metrics:', groups.get(3)?.metrics?.slice(0,5));
      const result = [...groups.values()];
      if (!keepOrder) result.sort((a, b) => a.family_num - b.family_num);
      return result;
    },

    // Returns secMetricsForBar() regrouped by family for <optgroup> rendering.
    // Preserves score-sort order: families appear in order of their highest-ranked
    // metric; within each family metrics keep their score rank.
    // Falls back to family_num sort when scanner is not loaded.
    secMetricsGrouped() {
      const flat = this.secMetricsForBar();  // [{name, score}, …] already sorted
      const groups = new Map();
      for (const m of flat) {
        const fam = this.metricFamilyLookup[m.name];
        const key = fam ? fam.family_num : 999;
        const label = fam ? fam.family_name : 'Other';
        if (!groups.has(key)) groups.set(key, { family_num: key, family_name: label, metrics: [] });
        groups.get(key).metrics.push(m);
      }
      const result = [...groups.values()];
      // When scanner is not loaded, sort groups by family_num for consistency.
      if (!this.secMetrics.length) result.sort((a, b) => a.family_num - b.family_num);
      return result;
    },

    // M1/M3: returns the full static feature list for the control-bar <select>.
    // Source: this.features (loaded at page-start from /columns — same list that
    // feeds the primary metric dropdown). Scanner is NOT required; it only provides
    // optional ranking (lift scores) that sorts metrics when available.
    // Excludes the currently-selected primary metric to avoid self-reference.
    // When scanner is loaded, sorts by |score| desc (top-ranked first).
    // When scanner is not loaded, sorts alphabetically.
    secMetricsForBar() {
      const scanMap = {};
      for (const m of this.secMetrics) scanMap[m.name] = m.score;

      const list = (this.features || [])
        .filter(f => f !== this.metric)   // exclude active primary metric
        .map(f => ({ name: f, score: scanMap[f] ?? null }));

      if (this.secMetrics.length > 0) {
        // Scanner loaded: rank by |score| descending, then alphabetical as tiebreak
        list.sort((a, b) => {
          const sa = a.score !== null ? Math.abs(a.score) : -1;
          const sb = b.score !== null ? Math.abs(b.score) : -1;
          return sb - sa || a.name.localeCompare(b.name);
        });
      } else {
        list.sort((a, b) => a.name.localeCompare(b.name));
      }

      // Safety: if secSelectedMetric is set but not in features (shouldn't happen),
      // prepend a stub entry so the select always shows the current value.
      if (this.secSelectedMetric && !list.some(m => m.name === this.secSelectedMetric)) {
        list.unshift({ name: this.secSelectedMetric, score: null });
      }
      return list;
    },

    // R2: single entry point for secondary metric selection (dropdown @change + lift-bar click).
    // Routes to secDrillMetric() when rows are already cached, else fetches rows first.
    secSelectMetric(name) {
      if (!name) { this.secDetail = null; return; }
      if (this.secCacheKey) {
        this.secDrillMetric(name, false);
      } else {
        this._prepareSecRowsThenDrill();
      }
    },

    // M2: primary metric/ticker/outcome/dates changed → fetch rows for new primary context
    // (no scanner dispatch), then redrill the current secondary.
    async _prepareSecRowsThenDrill() {
      if (!this.secSelectedMetric) return;
      this.secDetailLoading = true;
      // Issue 4 diagnostics: measure server round-trip time separately from drill time.
      const _t0 = performance.now();
      console.log(`[secPrepare] START — ${this.ticker} / ${this.metric} / ${this.outcome}`);
      try {
        const r = await fetch('/api/factor-analysis/secondary-prepare-rows', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            ticker:    this.ticker,
            metric:    this.metric,
            outcome:   this.outcome,
            date_from: this.dateFrom || '',
            date_to:   this.dateTo   || '',
          }),
        });
        const _t1 = performance.now();
        console.log(`[secPrepare] /secondary-prepare-rows responded in ${((_t1-_t0)/1000).toFixed(2)}s`);
        const d = await r.json();
        if (d.error || !d.cache_key) {
          this._clearSecDetailCharts();
          this.secDetail = { error: d.error || 'Could not prepare secondary data for new primary.' };
          return;
        }
        this.secCacheKey = d.cache_key;
        this._secCacheParams = {
          ticker:   this.ticker,
          metric:   this.metric,
          outcome:  this.outcome,
          dateFrom: this.dateFrom || '',
          dateTo:   this.dateTo   || '',
        };
        // secDrillMetric carries live walk_forward + cutoff_date from toggle —
        // no second call to loadHeatmap needed here (it fires inside secDrillMetric).
        await this.secDrillMetric(this.secSelectedMetric, false);
        const _t2 = performance.now();
        console.log(`[secPrepare] full prepare+drill done in ${((_t2-_t0)/1000).toFixed(2)}s total`);
      } catch (e) {
        this._clearSecDetailCharts();
        this.secDetail = { error: e.message };
      } finally {
        this.secDetailLoading = false;
      }
    },

    async _pollSecScore() {
      if (this.secPolling) return;
      this.secPolling = true;
      const MAX_POLLS = 300;  // 15 min at 3s — matches IC batch timeout
      let polls = 0;
      try {
        while (this.secPolling && polls < MAX_POLLS) {
          await new Promise(res => setTimeout(res, 3000));
          polls++;
          if (!this.secScanKey) break;
          const r = await fetch('/api/factor-analysis/secondary-score-status', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ scan_key: this.secScanKey }),
          });
          if (!r.ok) break;
          const d = await r.json();
          if (d.status === 'done') {
            this.secPolling = false;
            this._applySecResults(d);
            if (this.secSelectedMetric) await this.secDrillMetric(this.secSelectedMetric, false);
            this._renderSecBar();
            // Minis stay out-of-sync after scan completes — user must click
            // Compute Correlation explicitly. Out-of-sync banner covers this.
            break;
          } else if (d.status === 'error' || d.status === 'not_found') {
            this.secPolling = false;
            this.secStatus = { loaded: false, loading: false, error: d.error || d.status };
            break;
          }
          // status === 'computing' — keep polling
        }
        if (polls >= MAX_POLLS) {
          this.secStatus = { loaded: false, loading: false,
                             error: 'Score computation timed out — try Reload.' };
        }
      } finally {
        this.secPolling = false;
        if (!this.secStatus.loaded) {
          this.secStatus = { ...this.secStatus, loading: false };
        }
      }
    },

    async secLoad() {
      if (!this.data || this.secStatus.loading || this.secPolling) return;
      this.secStatus = { loaded: false, loading: true, error: null };
      // secSelectedMetric and secDetail intentionally NOT cleared here — scanner
      // reload should not destroy the current confirmation selection.
      this.secSelectedSecBins = [this.secBinCount];
      this.corrMiniData = null;
      this.corrSelections = {};
      this.corrResult = null;
      try {
        const r = await fetch('/api/factor-analysis/secondary-load', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            ticker:                this.ticker,
            metric:                this.metric,
            outcome:               this.outcome,
            date_from:             this.dateFrom || '',
            date_to:               this.dateTo || '',
            filtered_dates:        this._secFilteredDates(),
            sec_bin_count:         this.secBinCount,
            selected_primary_bins: [...this.selectedBins20].sort((a, b) => a - b),
            mode:                  this.pageMode || 'walk_forward',
            cutoff_date:           this.cutoffDate || '',
          }),
        });
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        const d = await r.json();
        if (d.error && !d.scan_key) throw new Error(d.error);
        this.secCacheKey = d.cache_key;
        this.secScanKey  = d.scan_key;
        // M2: snapshot params so loadAnalysis() can detect row-cache validity later
        this._secCacheParams = {
          ticker:   this.ticker,
          metric:   this.metric,
          outcome:  this.outcome,
          dateFrom: this.dateFrom || '',
          dateTo:   this.dateTo   || '',
        };
        if (d.status === 'done') {
          this._applySecResults(d);
        } else if (d.status === 'computing') {
          this._pollSecScore();  // fire-and-forget; loading stays true
        } else if (d.status === 'busy') {
          throw new Error(d.error || 'Another heavy job is running — try again shortly.');
        } else {
          throw new Error(`Unexpected status: ${d.status}`);
        }
      } catch (e) {
        this.secStatus = { loaded: false, loading: false, error: e.message };
      }
    },

    async secScan() {
      if (!this.secCacheKey || this.secStatus.loading || this.secPolling) return;
      this.secStatus.loading = true;
      let keepLoading = false;
      try {
        const r = await fetch('/api/factor-analysis/secondary-scan', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            cache_key:             this.secCacheKey,
            filtered_dates:        this._secFilteredDates(),
            ticker:                this.ticker,
            sec_bin_count:         this.secBinCount,
            selected_primary_bins: [...this.selectedBins20].sort((a, b) => a - b),
            mode:                  this.pageMode || 'walk_forward',
            cutoff_date:           this.cutoffDate || '',
          }),
        });
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        const d = await r.json();
        if (d.error === 'cache_miss') {
          this.secStatus = { loaded: false, loading: false, error: null };
          return;
        }
        this.secScanKey = d.scan_key;
        if (d.status === 'done') {
          this._applySecResults(d);
          if (this.secSelectedMetric) await this.secDrillMetric(this.secSelectedMetric, false);
          this._renderSecBar();
          // Minis stay out-of-sync after scan completes — user must click
          // Compute Correlation explicitly. Out-of-sync banner covers this.
        } else if (d.status === 'computing') {
          keepLoading = true;
          this._pollSecScore();
        } else if (d.status === 'busy') {
          this.secStatus = { loaded: this.secStatus.loaded, loading: false,
                             error: d.error || 'Another heavy job is running — try again shortly.' };
        }
      } catch (_) {}
      finally { if (!keepLoading) this.secStatus.loading = false; }
    },

    async secDrillMetric(metricName, resetBins = true) {
      if (!this.secCacheKey) return;
      // P1: version counter cancels stale responses when a newer call supersedes this one.
      const version = ++this._secDrillVersion;
      this.secSelectedMetric = metricName;
      if (resetBins) this.secSelectedSecBins = [5];
      this.secDetailLoading = true;
      this.secDetail = null;
      try {
        const r = await fetch('/api/factor-analysis/secondary-detail', {
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
            walk_forward:          this.pageMode === 'walk_forward',
            cutoff_date:           this.pageMode === 'train_test' ? this.cutoffDate : null,
          }),
        });
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        const d = await r.json();
        if (version !== this._secDrillVersion) return;  // stale — newer call in flight
        if (d.error) {
          const msg = d.error === 'insufficient_data'
            ? 'Insufficient data for this metric in the selected primary bins.'
            : d.error;
          this._clearSecDetailCharts();
          this.secDetail = { error: msg };
          return;
        }
        this.secDetail = d;
        await this.$nextTick();
        await this.$nextTick();
        setTimeout(() => this._renderSecDetail(), 60);
        // M3: heatmap X-axis = primary metric, Y-axis = secSelectedMetric.
        // Uses loadHeatmapIfChanged() — skips the fetch when only bin selection
        // changed (bin selection never affects the heatmap grid; all cells are
        // always shown). Rebuilds when metric, mode, granularity, or outcome change.
        this.loadHeatmapIfChanged();
      } catch (e) {
        if (version !== this._secDrillVersion) return;
        const msg = e.message === 'insufficient_data'
          ? 'Insufficient data for this metric in the selected primary bins.'
          : e.message;
        this._clearSecDetailCharts();
        this.secDetail = { error: msg };
      } finally { this.secDetailLoading = false; }
    },

    secToggleSecBin(bin) {
      const idx = this.secSelectedSecBins.indexOf(bin);
      if (idx >= 0) {
        if (this.secSelectedSecBins.length > 1) {
          this.secSelectedSecBins = this.secSelectedSecBins.filter(b => b !== bin);
        }
      } else {
        this.secSelectedSecBins = [...this.secSelectedSecBins, bin];
      }
      // Update bar highlight immediately (same pattern as primary _onDecileChangeLight);
      // full re-drill (equity/n-counts) is gated behind secondary Load.
      if (this.secDetail) this._renderSecBinsChart();
      this.secPending = true;
    },

    secSetBinCount(n) {
      this.secBinCount = n;
      this.secSelectedSecBins = [n];  // reset to top bin
      this.corrMiniData = null;       // n_bins changed — invalidate minis
      this.corrMiniComputedBins = null;
      this.corrSelections = {};
      this.corrResult = null;
      // Gated behind Load — secScan + drill fire when user clicks Load.
      this._secBinCountChanged = true;
      this.secPending = true;
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

    // Destroy all secondary canvas charts without re-drawing them.
    // Called on the error / insufficient-data path so stale charts from
    // the previously-loaded secondary don't bleed through.
    // sec-equity-canvas is auto-torn-down by Alpine's x-if guard
    // (secDetail && !secDetail.error), but the other four are always
    // in the DOM and must be explicitly cleared here.
    // Chart.js destroy() removes the instance but NOT the canvas pixels —
    // ctx.clearRect() is required to blank the visuals.
    _clearSecDetailCharts() {
      for (const key of ['sec-bins', 'sec-equity', 'sec-yearly', 'sec-activity', 'sec-bubble']) {
        if (this._charts[key]) { this._charts[key].destroy(); delete this._charts[key]; }
      }
      // Explicitly clear pixel data on the always-in-DOM canvases.
      for (const id of ['sec-bins-canvas', 'sec-yearly-canvas', 'sec-activity-canvas', 'sec-bubble-canvas']) {
        const canvas = document.getElementById(id);
        if (canvas) { const ctx = canvas.getContext('2d'); if (ctx) ctx.clearRect(0, 0, canvas.width, canvas.height); }
      }
      // Clear the 2-D heatmap (Alpine reactive — nulling the data collapses x-if blocks).
      // Also reset the guard key so loadHeatmapIfChanged() re-arms for the next valid metric.
      this.heatmapData = null;
      this._lastHeatmapKey = null;
    },

    _renderSecDetail() {
      if (!this.secDetail) return;
      this._renderSecBinsChart();
      this._renderSecEquity();
      this._renderSecYearly();
      this._renderSecActivity();
      this._renderSecBubble();
    },

    _renderSecBubble(canvasId = 'sec-bubble-canvas', detail = null) {
      detail = detail || this.secDetail;
      const _key = canvasId.replace(/-canvas$/, '').replace(/^chart-/, '');
      const canvas = document.getElementById(canvasId);
      if (!canvas || !detail?.tickers?.length) return;
      if (this._charts[_key]) { this._charts[_key].destroy(); delete this._charts[_key]; }

      const minN = this.secBubbleMinN || 1;
      const tickers = detail.tickers.filter(t => t.n >= minN);
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

      this._charts[_key] = new Chart(canvas.getContext('2d'), {
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
      else if (key === 'port') this._renderSecActivity('chart-port-activity', this.portAggregate);
    },

    // Threshold Drift (walk-forward bin boundaries over time)
    get tdHasSeries() {
      const s = this.tdData?.series_ratio || this.tdData?.series_native || {};
      return Object.values(s).some(arr => Array.isArray(arr) && arr.length > 0);
    },

    // tdBinsToShow is now a fixed [1, 20] — the chart always renders both
    // and the user toggles bin visibility via the chart legend (which
    // toggles the median line AND its IQR band together; see the custom
    // legend.onClick in _renderTdChart). The per-bin pill toggle UI was
    // removed because in practice only the extreme bins are useful.

    // Bucket A step 7: Threshold Drift helpers / setters / breadcrumb.

    _tdModeLabel(m) {
      if (m === 'in_sample')  return 'In-sample';
      if (m === 'train_test') return 'Train-test';
      return 'Walk-forward';
    },

    // Breadcrumb: "last: YYYY-MM-DD HH:MM:SS · <Mode label>".
    // Threshold Drift's cache is client-side in-memory (no DB table), so
    // the timestamp is when the data was last loaded into this session's
    // slot — analogous to scanned_at on the corner-scan tables.
    tdBreadcrumb() {
      const label = this._tdModeLabel(this.tdBinMode);
      const slot = this.tdDataByMode[this.tdBinMode];
      // Authoritative: this-session slot loaded by the user expanding +
      // refreshing the pane.
      if (slot && slot.tdData && !slot.tdData.error) {
        const ts = slot.loaded_at
          ? new Date(slot.loaded_at).toISOString().slice(0, 19).replace('T', ' ')
          : 'unknown';
        return `last: ${ts} · ${label}`;
      }
      // Fallback: init-time meta for default selectors. Threshold-drift
      // cache is in-memory only; cached_at clears on server restart,
      // and the meta call only fired for walk_forward defaults.
      if (this.tdBinMode === 'walk_forward'
          && this.tdMeta?.exists
          && this.tdMeta?.cached_at) {
        const ts = String(this.tdMeta.cached_at).slice(0, 19).replace('T', ' ');
        return `last: ${ts} · ${label}`;
      }
      return `no data yet · ${label}`;
    },

    // Refresh is only meaningful for walk_forward today. IS / TT are
    // future work — the IS computation is materially different from WF
    // (settled-threshold position over time, not rolling threshold), so
    // it's NOT a flag flip on the existing endpoint. Until that lands,
    // the selector positions are placeholder-only.
    tdCanRefresh() { return this.tdBinMode === 'walk_forward'; },

    setTdBinMode(m) {
      if (m === this.tdBinMode && m !== 'train_test') return;
      this.tdBinMode = m;
      this._tdSwapDisplayFromSlot();
    },
    setTdCutoffDate(d) { this.tdCutoffDate = d; },

    // Metric / outcome setters — no auto-fetch. User clicks Refresh.
    // The scope toggle (tdScope) re-renders without refetch, which is
    // already correct.
    setTdMetric(m)  { this.tdMetric = m; },
    setTdOutcome(o) { this.tdOutcome = o; },

    _tdSwapDisplayFromSlot() {
      const slot = this.tdDataByMode[this.tdBinMode];
      this.tdData = slot ? slot.tdData : null;
      if (this.tdExpanded && this.tdData && !this.tdData.error) {
        this.$nextTick(() => this._renderTdChart());
      }
    },

    _tdStoreSlot(data) {
      this.tdDataByMode[this.tdBinMode] = {
        tdData:    data,
        loaded_at: Date.now(),
      };
      this._tdSwapDisplayFromSlot();
    },

    async toggleTd() {
      this.tdExpanded = !this.tdExpanded;
      // First-expand auto-load only for walk_forward (the only mode with
      // a working endpoint today). Loads if there's no slot data yet AND
      // a metric is picked. IS / TT just show the placeholder.
      if (this.tdExpanded
          && this.tdMetric
          && !this.tdDataByMode[this.tdBinMode]
          && this.tdBinMode === 'walk_forward') {
        await this.loadTd();
      }
      // Re-render even if data was already loaded — the canvas may have
      // been hidden by the x-if.
      this.$nextTick(() => this._renderTdChart());
    },

    async loadTd(forceRefresh = false) {
      // Gated to walk_forward — IS / TT have no endpoint yet, so we
      // don't fire a request to a non-existent path.
      if (!this.tdCanRefresh()) return;
      if (!this.tdMetric || !this.tdBinsToShow.length) return;
      this.tdLoading = true;
      try {
        const params = new URLSearchParams({
          metric:  this.tdMetric,
          outcome: this.tdOutcome || 'ret_5d_fwd_oc',
          ticker:  'ALL',
          n_bins:  '20',
          bins:    this.tdBinsToShow.join(','),
          force:   forceRefresh ? '1' : '0',
        });
        const r = await fetch('/api/factor-analysis/threshold-drift?' + params);
        if (!r.ok) {
          const txt = await r.text().catch(() => '');
          this._tdStoreSlot({
            error: `HTTP ${r.status}${txt ? ': ' + txt.slice(0, 200) : ''}`,
            series: {}, in_sample_ref: {},
          });
          return;
        }
        const d = await r.json();
        this._tdStoreSlot(d);
        await this.$nextTick();
        this._renderTdChart();
      } catch (e) {
        console.error('loadTd', e);
        this._tdStoreSlot({ error: e.message, series: {}, in_sample_ref: {} });
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
      // Bucket A step 7: tdMode renamed to tdScope (the visualization
      // toggle: 'ratio' / 'native_single'). Local variable name `mode`
      // kept to minimize chart-render churn.
      const mode = this.tdScope || 'ratio';
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
                // Hide the invisible q75/band datasets from the legend —
                // only the median lines are user-visible legend entries.
                filter: (item) => !item.text.includes('q75') && !item.text.includes('band'),
              },
              // Toggle the median line AND its companion IQR band datasets
              // together — clicking "B1 (median)" hides the B1 q75 fence,
              // the B1 band fill, and the B1 line as one unit. Without this,
              // toggling the median leaves a stray IQR band on the chart.
              onClick: (e, legendItem, legend) => {
                const chart = legend.chart;
                const m = (legendItem.text || '').match(/^B(\d+)/);
                if (!m) return;
                const binPrefix = `B${m[1]}`;
                // Determine the new hidden state from the clicked dataset.
                const clickedMeta = chart.getDatasetMeta(legendItem.datasetIndex);
                const willHide = !clickedMeta.hidden;
                // Apply that state to every dataset belonging to this bin
                // (median + q75 fence + band fill). In native_single mode
                // the q75/band datasets don't exist; the loop is a no-op
                // for those and just toggles the single median dataset.
                chart.data.datasets.forEach((ds, idx) => {
                  const label = ds.label || '';
                  if (label === binPrefix
                      || label.startsWith(`${binPrefix} `)) {
                    chart.getDatasetMeta(idx).hidden = willHide;
                  }
                });
                chart.update();
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

    // Bucket A — shared label / breadcrumb helpers (mirror Score Matrix's).
    _topBinsModeLabel(m) {
      if (m === 'in_sample')  return 'In-sample';
      if (m === 'train_test') return 'Train-test';
      return 'Walk-forward';
    },

    // Breadcrumb: "last: YYYY-MM-DD HH:MM:SS · <Mode label>" — identical
    // format across all 6 panes. cached_at is an ISO-ish timestamp from
    // the backend; we slice it to the first 19 chars to match Score
    // Matrix's `str(scanned_at)[:19]` server-side format.
    topBinsBreadcrumb() {
      const slot = this.topBinsDataByMode[this.topBinsMode];
      const label = this._topBinsModeLabel(this.topBinsMode);
      // Authoritative: per-mode slot once the user has expanded + loaded.
      if (slot && !slot.error && slot.metrics?.length) {
        const ts = slot.cached_at ? String(slot.cached_at).slice(0, 19) : 'unknown';
        return `last: ${ts} · ${label}`;
      }
      // Fallback: init-time meta for the default selectors. The meta
      // call only fired for walk_forward defaults, so it's only
      // meaningful when the current mode is walk_forward.
      if (this.topBinsMode === 'walk_forward'
          && this.topBinsMeta?.exists
          && this.topBinsMeta?.cached_at) {
        const ts = String(this.topBinsMeta.cached_at).slice(0, 19);
        return `last: ${ts} · ${label}`;
      }
      return `no data yet · ${label}`;
    },

    // Mode-pill / cutoff / outcome setters — all in-memory only.
    setTopBinsMode(m) {
      if (m === this.topBinsMode && m !== 'train_test') return;
      this.topBinsMode = m;
      this._topBinsSwapDisplayFromSlot();
    },
    setTopBinsCutoffDate(d) {
      this.topBinsCutoffDate = d;
    },
    setTopBinsOutcome(o) {
      // No auto-fetch. The displayed slot may still carry the prior
      // outcome's bins until the user clicks ⟳ Refresh; the dropdown
      // shows the chosen outcome and topBinsData.outcome shows what's
      // actually displayed, so any mismatch is visible in the UI.
      this.topBinsOutcome = o;
    },

    _topBinsSwapDisplayFromSlot() {
      const slot = this.topBinsDataByMode[this.topBinsMode];
      this.topBinsData = slot || null;
    },

    _topBinsStoreSlot(d) {
      const m = this.topBinsMode;
      this.topBinsDataByMode[m] = d;
      if (d && d.error) {
        this.topBinsStatusByMode[m] = 'error';
      } else if (d && d.metrics?.length) {
        this.topBinsStatusByMode[m] = 'ready';
      } else {
        this.topBinsStatusByMode[m] = 'no_data';
      }
      this._topBinsSwapDisplayFromSlot();
    },

    async toggleTopBins() {
      this.topBinsExpanded = !this.topBinsExpanded;
      // First-expand auto-load: if nothing has ever been loaded for the
      // active local mode, fetch once. Subsequent mode / outcome / cutoff
      // changes do NOT auto-fetch — user clicks ⟳ Refresh.
      if (this.topBinsExpanded && !this.topBinsDataByMode[this.topBinsMode]) {
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
      const m = this.topBinsMode;
      this.topBinsStatusByMode[m] = 'loading';
      try {
        const params = new URLSearchParams({
          outcome:      this.topBinsOutcome || 'ret_5d_fwd_oc',
          ticker:       'ALL',
          n_bins:       '20',
          walk_forward: m === 'walk_forward' ? '1' : '0',
          force:        forceRefresh ? '1' : '0',
        });
        if (m === 'train_test') params.set('cutoff_date', this.topBinsCutoffDate);
        const r = await fetch('/api/factor-analysis/global-metric-bins?' + params);
        if (!r.ok) {
          const txt = await r.text().catch(() => '');
          this._topBinsStoreSlot({
            metrics: [], total_rows: 0,
            error: `HTTP ${r.status}${txt ? ': ' + txt.slice(0, 200) : ''}`,
          });
          return;
        }
        const d = await r.json();
        // Compute _zeroTopPct + _total per metric for the diverging-bar layout
        // (same pattern the corr explorer uses for its mini charts).
        for (const m_ of (d.metrics || [])) {
          const maxPos = Math.max(0, ...m_.bins);
          const maxNeg = Math.abs(Math.min(0, ...m_.bins));
          m_._total      = Math.max(0.0001, (maxPos + maxNeg) * 1.06);
          m_._zeroTopPct = (maxPos / m_._total * 100).toFixed(2);
        }
        this._topBinsStoreSlot(d);
      } catch (e) {
        console.error('loadTopBins', e);
        this._topBinsStoreSlot({
          metrics: [], total_rows: 0, error: e.message,
        });
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
      if (this.corrPanelOpen && !this.corrMiniData) {
        if (this.secCacheKey) {
          await this.corrLoadMiniData();
        } else {
          // Rows not cached yet (scanner never ran) — prepare rows fast then compute minis.
          // Minis depend on the conditioned-population rows, not the lift ranking.
          await this._prepareSecRowsForMinis();
        }
      }
    },

    // Fetch /secondary-prepare-rows (~0.08s) to populate secCacheKey, then load minis.
    // Lets the minis open without requiring the scanner lift scan to have run first.
    async _prepareSecRowsForMinis() {
      this.corrMiniLoading = true;   // show spinner during row-prep phase
      try {
        const r = await fetch('/api/factor-analysis/secondary-prepare-rows', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            ticker:    this.ticker,
            metric:    this.metric,
            outcome:   this.outcome,
            date_from: this.dateFrom || '',
            date_to:   this.dateTo   || '',
          }),
        });
        const d = await r.json();
        if (d.error || !d.cache_key) {
          this.corrMiniData = { error: d.error || 'Could not prepare secondary rows.' };
          this.corrMiniLoading = false;
          return;
        }
        this.secCacheKey = d.cache_key;
        this._secCacheParams = {
          ticker:    this.ticker,   metric:   this.metric,
          outcome:   this.outcome,  dateFrom: this.dateFrom || '',
          dateTo:    this.dateTo || '',
        };
        await this.corrLoadMiniData();  // corrLoadMiniData() manages corrMiniLoading to completion
      } catch (e) {
        this.corrMiniData = { error: e.message };
        this.corrMiniLoading = false;
      }
    },

    // corrSetBinCount removed — use secSetBinCount() which now drives both detail and minis

    // Page-wide mode toggle. Cascades through every binning surface on
    // the page: All-Ticker Metric Bins, /analyze, corr explorer (mini +
    // result), System Portfolio aggregate, Score Matrix, and the heatmap
    // (grid + side bars). Called by the segmented mode toggle AND by the
    // Bucket A step 2: the Analyze-section mode pill (and the cutoff-date
    // input that calls setPageMode('train_test') on @change) are now
    // INERT — they only update this.pageMode (and the bound this.cutoffDate
    // via x-model) and do nothing else.
    //
    // The Analyze button is the only thing that fires loadAnalysis() now,
    // so users can change mode + cutoff + ticker + metric + outcome and
    // then dispatch one fetch. This matches the design rule: the Analyze
    // section's mode has NO connection to any of the 6 top-level panes.
    //
    // What used to live here that is now gone:
    //   - this.topBinsData / loadTopBins cascade  → All-Ticker Metric Bins
    //                                                gets local mode in step 4
    //   - corrMiniData / corrResult / corrLoadMiniData / runCorrelation
    //                                              → loadAnalysis() now
    //                                                handles these via its
    //                                                own cascade, fired by
    //                                                the Analyze button
    //   - loadAnalysis() itself                    → moved to the Analyze
    //                                                button click
    //   - loadPortfolioAggregate()                 → fires inside
    //                                                loadAnalysis()'s
    //                                                cascade when needed
    //   - smInit() (Score Matrix re-init)          → Score Matrix gets
    //                                                local mode in step 3
    //
    // Signal Survey still listens to pageMode via the $watch at the top
    // of init() until step 5 decouples it. The watcher keeps firing
    // loadIcBatch on mode change in this intermediate state.
    //
    // The early-return guard stays: in train_test mode the cutoffDate
    // may have shifted even though the mode label didn't (the
    // cutoff-date input @change fires setPageMode('train_test')), so
    // we allow re-entry when mode is train_test even if the label is
    // unchanged — currently a no-op, but kept so a future caller can
    // hook a reaction without re-introducing the gate.
    setPageMode(m) {
      if (m === this.pageMode && m !== 'train_test') return;
      this.pageMode = m;
    },

    async corrLoadMiniData() {
      this.corrMiniLoading = true;
      try {
        const r = await fetch('/api/factor-analysis/secondary-corr-bins', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({
            cache_key:      this.secCacheKey,
            filtered_dates: this._secFilteredDates(),
            ticker:         this.ticker,
            n_bins:         this.secBinCount,
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
        if (!d.error) {
          // P4: snapshot the primary-bin selection the minis were computed against
          this.corrMiniComputedBins = [...this.selectedBins20].sort((a, b) => a - b);
        }
      } catch (e) {
        this.corrMiniData = { error: e.message };
      } finally {
        this.corrMiniLoading = false;
      }
    },

    // P4: true when current primary-bin selection differs from last mini compute
    corrMiniOutOfSync() {
      if (!this.corrMiniComputedBins || !this.corrMiniData || this.corrMiniData.error) return false;
      const cur = [...this.selectedBins20].sort((a, b) => a - b);
      return JSON.stringify(cur) !== JSON.stringify(this.corrMiniComputedBins);
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
      // P2: always refresh the 100 mini charts first (the primary purpose of this button).
      // The phi-correlation matrix (requires 2+ selected metrics) runs after.
      if (!this.secCacheKey) return;
      await this.corrLoadMiniData();
      if (this.corrSelectedCount() < 2) return;
      // phi correlation matrix:
      this.corrLoading = true;
      this.corrResult = null;
      try {
        const selections = Object.entries(this.corrSelections)
          .map(([metric, bins]) => ({ metric, bins }));
        const r = await fetch('/api/factor-analysis/secondary-correlation', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({
            cache_key:      this.secCacheKey,
            filtered_dates: this._secFilteredDates(),
            ticker:         this.ticker,
            n_bins:         this.secBinCount,
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

    _renderSecEquity(canvasId = 'sec-equity-canvas', detail = null, singleSeries = false) {
      detail = detail || this.secDetail;
      const _key = canvasId.replace(/-canvas$/, '').replace(/^chart-/, '');
      const canvas = document.getElementById(canvasId);
      if (!canvas || !detail) return;
      if (this._charts[_key]) { this._charts[_key].destroy(); delete this._charts[_key]; }
      const eqP = detail.equity_primary || [];
      const eqC = detail.equity_combined || [];
      if (!eqP.length) return;
      const ctx = canvas.getContext('2d');

      let datasets;
      if (singleSeries) {
        // Zone mode: single curve — equity_primary is the zone curve
        datasets = [{
          label: 'Zone equity',
          data: eqP.map(p => +(p.value * 100).toFixed(4)),
          borderColor: '#e84393',
          backgroundColor: 'rgba(232,67,147,0.08)',
          borderWidth: 1.5,
          pointRadius: 0,
          fill: false,
          tension: 0,
        }];
      } else {
        // Sec-detail mode: primary + combined curves
        const cMap = Object.fromEntries(eqC.map(p => [p.date, +(p.value * 100).toFixed(4)]));
        let lastCombined = 0;
        const combinedAligned = eqP.map(p => {
          if (cMap[p.date] !== undefined) lastCombined = cMap[p.date];
          return lastCombined;
        });
        datasets = [
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
        ];
      }

      this._charts[_key] = new Chart(ctx, {
        type: 'line',
        data: { labels: eqP.map(p => p.date), datasets },
        options: {
          responsive: true, maintainAspectRatio: false, animation: false,
          plugins: {
            legend: {
              display: !singleSeries,
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

    _renderSecActivity(canvasId = 'sec-activity-canvas', detail = null) {
      detail = detail || this.secDetail;
      const _key = canvasId.replace(/-canvas$/, '').replace(/^chart-/, '');
      const canvas = document.getElementById(canvasId);
      if (!canvas || !detail) return;
      if (this._charts[_key]) { this._charts[_key].destroy(); delete this._charts[_key]; }

      // Prefer the enriched combined_trades (has ticker per entry) so the
      // dedupe-concurrent toggle can work per ticker. Fall back to plain
      // combined_trade_dates for older payloads.
      const trades = detail.combined_trades
        || (detail.combined_trade_dates || []).map(d => ({ ticker: '?', trade_date: d }));
      if (!trades.length) return;
      const horizon = detail.horizon || 1;

      const spotSeries = this.data?.spot_series || [];
      const cal = this.data?.trade_calendar || [];
      const dates = trades.map(t => t.trade_date || t.date);
      const tradingDays = spotSeries.length > 0
        ? spotSeries.map(s => s.date)
        : [...new Set(cal.length > 0 ? cal.map(c => c.date) : dates)].sort();

      // Derive dedupeConc key from canvasId: chart-port-activity → 'port', else 'sec'
      const _dedupeKey = canvasId.includes('port') ? 'port' : 'sec';
      const kept = this.dedupeConc[_dedupeKey]
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
      this._charts[_key] = new Chart(ctx, {
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

    // ── Signal Portfolio (third tier) ──────────────────────────────────────

    portSetBubbleMinN(n) {
      this.portBubbleMinN = +n;
      this._renderSecBubble('chart-port-bubble', this.portAggregate);
    },

    async loadPortfolios() {
      try {
        const r = await fetch('/api/factor-analysis/portfolios');
        if (r.ok) this.portfolios = await r.json();
      } catch (_) {}
    },

    async selectPortfolio(id) {
      this._destroyPortCharts();
      this.portAggregate = null;
      if (!id) { this.portfolio = null; return; }
      this.portLoading = true;
      try {
        const r = await fetch(`/api/factor-analysis/portfolios/${id}`);
        if (r.ok) this.portfolio = await r.json();
        else { this.portfolio = null; return; }
        await this.loadPortfolioAggregate();
      } finally { this.portLoading = false; }
    },

    async createPortfolio() {
      const name = prompt('Portfolio name:', `Research ${new Date().toISOString().slice(0, 10)}`);
      if (!name) return;
      // Outcome is NOT set at creation — it is derived from the first signal added.
      try {
        const r = await fetch('/api/factor-analysis/portfolios', {
          method: 'POST', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ name: name.trim() }),
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
        const r = await fetch(`/api/factor-analysis/portfolios/${this.portfolioId}`, {
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
      if (!confirm(`Delete portfolio "${name}" and all its signals?`)) return;
      try {
        const r = await fetch(`/api/factor-analysis/portfolios/${this.portfolioId}`, { method: 'DELETE' });
        if (!r.ok) { alert('Delete failed: ' + await r.text()); return; }
        this.portfolioId = null;
        this.portfolio = null;
        this.portAggregate = null;
        this._destroyPortCharts();
        await this.loadPortfolios();
      } catch (e) { alert('Delete error: ' + e.message); }
    },

    async addSignalToPortfolio() {
      const signalId = parseInt(this.portSignalPick);
      if (!signalId || !this.portfolioId) return;
      try {
        const r = await fetch(`/api/factor-analysis/portfolios/${this.portfolioId}/signals`, {
          method: 'POST', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ signal_id: signalId }),
        });
        if (!r.ok) { alert('Add signal failed: ' + await r.text()); return; }
        this.portSignalPick = '';
        await this.selectPortfolio(this.portfolioId);
        await this.loadPortfolios();
      } catch (e) { alert('Add signal error: ' + e.message); }
    },

    async removeSignalFromPortfolio(psId) {
      const sig = this.portfolio?.signals?.find(s => s.id === psId);
      if (!confirm(`Remove "${sig?.name || 'signal'}" from this portfolio?`)) return;
      try {
        const r = await fetch(
          `/api/factor-analysis/portfolios/${this.portfolioId}/signals/${psId}`,
          { method: 'DELETE' });
        if (!r.ok) { alert('Remove failed: ' + await r.text()); return; }
        await this.selectPortfolio(this.portfolioId);
        await this.loadPortfolios();
      } catch (_) {}
    },

    async togglePortfolioSignal(psId, enabled) {
      try {
        await fetch(`/api/factor-analysis/portfolios/${this.portfolioId}/signals/${psId}`, {
          method: 'PUT', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ enabled }),
        });
        // Optimistic update
        const sig = this.portfolio?.signals?.find(s => s.id === psId);
        if (sig) sig.enabled = enabled;
        await this.loadPortfolioAggregate();
      } catch (_) {}
    },

    async loadPortfolioAggregate() {
      if (!this.portfolioId) return;
      this.portLoading = true;
      try {
        // Portfolio aggregate is always IS — no mode params sent.
        const r = await fetch(`/api/factor-analysis/portfolios/${this.portfolioId}/aggregate`, {
          method: 'POST', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({}),
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
      if (!this.portAggregate || !(this.portAggregate.n > 0)) {
        this._destroyPortCharts();
        return;
      }
      this._destroyPortCharts();
      // Equity + activity + bubble delegate to the parameterized _renderSec* methods
      // (singleSeries=true → one pink curve, no primary-universe blue line).
      this._renderSecEquity('chart-port-equity', this.portAggregate, true);
      this._renderPortYearly();
      this._renderSecActivity('chart-port-activity', this.portAggregate);
      this._renderSecBubble('chart-port-bubble', this.portAggregate);
    },

    _renderPortYearly() {
      // Single-series yearly bars with n-count gradient (same pattern as _renderZoneYearly).
      const canvas = document.getElementById('chart-port-yearly');
      if (!canvas || !this.portAggregate?.yearly?.length) return;
      if (this._charts['port-yearly']) { this._charts['port-yearly'].destroy(); delete this._charts['port-yearly']; }
      const yearly = this.portAggregate.yearly;
      const ns = yearly.map(y => y.n);
      const minN = Math.min(...ns), maxN = Math.max(...ns);
      const nPct  = y => maxN > minN ? (y.n - minN) / (maxN - minN) : 1;
      const alpha = y => (0.2 + nPct(y) * 0.6).toFixed(2);
      const bgColor = y => y.avg_ret >= 0
        ? `rgba(52,152,219,${alpha(y)})` : `rgba(232,67,147,${alpha(y)})`;
      this._charts['port-yearly'] = new Chart(canvas.getContext('2d'), {
        type: 'bar',
        data: {
          labels: yearly.map(y => y.year),
          datasets: [{
            label: 'Union Avg Ret',
            data: yearly.map(y => +(y.avg_ret * 100).toFixed(3)),
            backgroundColor: yearly.map(bgColor),
            borderColor: yearly.map(y => y.avg_ret >= 0 ? '#3498db' : '#e84393'),
            borderWidth: 1,
          }],
        },
        options: {
          responsive: true, maintainAspectRatio: false, animation: false,
          plugins: { legend: { display: false },
            tooltip: { callbacks: { label: ctx => {
              const y = yearly[ctx.dataIndex];
              return [`Avg: ${(y.avg_ret*100).toFixed(3)}%`, `WR: ${(y.win_rate*100).toFixed(1)}%`, `n: ${y.n}`];
            } } } },
          scales: {
            ...this._darkScales(),
            y: { ...this._darkScales().y, title: { display: true, text: 'Avg Return %', color: '#888', font: { size: 9 } } },
          },
        },
      });
    },

    portSigCellTitle(i, j) {
      if (!this.portAggregate) return '';
      const labs = this.portAggregate.system_labels || [];
      const ov = this.portAggregate.overlap_systems || [];
      const phi = this.portAggregate.phi_systems || [];
      if (i === j) return `${labs[i]}  n=${ov[i]?.[i] ?? 0}`;
      return `${labs[i]} × ${labs[j]}\nφ = ${(phi[i]?.[j] ?? 0).toFixed(3)}\nOverlap: ${ov[i]?.[j] ?? 0} trades`;
    },

    async portCsvDownload() {
      if (!this.portAggregate) return;
      const trades = this.portAggregate.combined_trades || [];
      if (!trades.length) return;
      const header = ['ticker', 'trade_date', 'spot_entry', 'exit_date', 'spot_exit', 'ret_pct'].join(',');
      const fmt = (v, d = 6) => v == null ? '' : Number(v).toFixed(d);
      const rows = trades.map(t => [
        t.ticker || '',
        t.trade_date || '',
        fmt(t.spot_entry, 2),
        t.exit_date || '',
        fmt(t.spot_exit, 2),
        t.ret != null ? (t.ret * 100).toFixed(6) : '',
      ].join(','));
      this._downloadCsv([header, ...rows].join('\n'),
        `portfolio_${this.portfolioId}_union_${new Date().toISOString().slice(0,10)}.csv`);
    },

    // ── P1: analyze_cache 12-outcome bundle ───────────────────────────────
    // Two-stage render flow:
    //   1. loadAnalysis() fires /analyze → single-outcome view (ret_5d_fwd_oc default)
    //   2. loadAnalyzeBundle() fires /analyze-bundle → all 12 outcomes
    // The bundle is what P3+ mode buttons consume. P1 just plumbs the data
    // path and surfaces a status indicator; no UI rendering off the bundle yet.

    _analyzeBundleMode() {
      return this.pageMode === 'walk_forward' ? 'walk_forward'
           : this.pageMode === 'train_test'   ? 'train_test'
           : 'in_sample';
    },
    _analyzeBundleKey() {
      // Cache key for analyze-bundle, excluding outcome and FROM/TO.
      // FROM/TO is display windowing only (applied client-side). Outcome
      // is irrelevant because the bundle contains all 12.
      const cd = this.pageMode === 'train_test' ? this.cutoffDate : '';
      return `${this.ticker}:${this.metric}:${this._analyzeBundleMode()}:${cd}`;
    },
    _analyzeBundleBaseUrl() {
      let url = '/api/factor-analysis/analyze-bundle'
        + `?ticker=${encodeURIComponent(this.ticker)}`
        + `&metric=${encodeURIComponent(this.metric)}`
        + `&mode=${this._analyzeBundleMode()}`;
      if (this.pageMode === 'train_test')
        url += `&cutoff_date=${encodeURIComponent(this.cutoffDate)}`;
      return url;
    },

    async loadAnalyzeBundle() {
      if (!this.ticker || !this.metric) return;
      this.analyzeBundleStatus = 'computing';
      this.analyzeBundleError  = null;
      this._stopAnalyzeBundlePolling();
      try {
        const r = await fetch(this._analyzeBundleBaseUrl());
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        const d = await r.json();
        if (d.error && !d.status) throw new Error(d.error);

        if (d.status === 'ready') {
          // Single-ticker GET embeds the bundle inline (small payload,
          // fast). ALL-mode GET returns a lean status only — payload
          // fetched separately via /analyze-bundle/payload so polling
          // never carries the ~130MB body. The shape disambiguator is
          // `bundle` being present in the response.
          if (d.bundle !== undefined) {
            this.analyzeBundle       = d.bundle;
            this.analyzeBundleStatus = 'ready';
            this.analyzeBundleKey    = this._analyzeBundleKey();
          } else {
            await this._fetchAnalyzeBundlePayload();
          }
        } else if (d.status === 'computing') {
          // ALL-mode background job already in flight — start polling.
          this.analyzeBundleStatus = 'computing';
          this.analyzeBundle       = null;
          this._startAnalyzeBundlePolling();
        } else if (d.status === 'not_computed') {
          // ALL-mode cache miss with no running job. Auto-trigger refresh
          // so the user gets a bundle without an extra click.
          this.analyzeBundleStatus = 'computing';
          this.analyzeBundle       = null;
          await this.refreshAnalyzeBundle();
        } else {
          this.analyzeBundleStatus = 'failed';
          this.analyzeBundleError  = `unexpected response: ${JSON.stringify(d).slice(0, 200)}`;
        }
      } catch (e) {
        this.analyzeBundleStatus = 'failed';
        this.analyzeBundleError  = e.message;
      }
    },

    async _fetchAnalyzeBundlePayload() {
      // One-shot ALL-mode SLIM payload fetch (~1.3 MB, v6) — called after
      // polling sees status=ready, or after the initial GET on an already-
      // cached ALL-mode result. Slim payload carries per_bin + rolling_ic
      // + scalar metadata; trade_meta (~41 MB) and per_outcome_returns
      // (~7.4 MB per outcome) load lazily via _ensureTradeMeta /
      // _ensureOutcome the first time a user action needs them. On
      // failure, surface the error through the same status fields the
      // rest of the bundle flow uses.
      const baseUrl = this._analyzeBundleBaseUrl();
      const url     = baseUrl.replace('/analyze-bundle?', '/analyze-bundle/payload?');
      const r = await fetch(url);
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      const d = await r.json();
      if (d.error) throw new Error(d.error);
      this.analyzeBundle       = d.bundle;
      this.analyzeBundleStatus = 'ready';
      this.analyzeBundleKey    = this._analyzeBundleKey();
      // Fresh slim bundle — discard any v6 lazy-load promises queued
      // against the prior cache_key. trade_meta / per_outcome_returns on
      // the new bundle are absent by definition (just-arrived slim).
      this._tradeMetaInFlight = null;
      this._outcomesInFlight  = {};
      // Hybrid prefetch: kick off trade_meta + the default outcome's
      // per_outcome_returns silently in the background so the default-
      // outcome flat trade table populates without the user waiting on
      // a click. Fire-and-forget — main visuals already rendered from
      // /analyze before slim arrival; this never blocks any chart. By
      // the time the user scrolls/clicks to the trade table, the data is
      // usually already on the bundle and the table renders instantly.
      // If the user clicks first, the flat-view render shows a passive
      // "Loading trade data…" placeholder until the prefetch resolves.
      this._prefetchDefaultOutcomeAfterBundle();
    },

    async _prefetchDefaultOutcomeAfterBundle() {
      // Background prefetch of the current outcome's trade data + shared
      // trade_meta. Used by the slim-bundle path so the default-outcome
      // flat trade table populates silently after the main visuals are
      // on screen. ALL-mode only — single-ticker bundles already carry
      // everything inline. Gap mode skipped — entering Gap mode triggers
      // its own (different) lazy-fetch with a chart spinner because the
      // user explicitly switched into it.
      if (this.ticker !== 'ALL') return;
      if (!this.analyzeBundle) return;
      const outcome = this.outcome;
      if (!outcome || outcome === 'overnight_gap') return;
      try {
        await Promise.all([
          this._ensureTradeMeta(),
          this._ensureOutcome(outcome),
        ]);
        // Bundle now has trade_meta + per_outcome_returns[outcome].
        // Re-render the trade table — the prior render likely showed
        // the "Loading trade data…" placeholder; this swap populates
        // the rows without any spinner overlay or user action.
        this._renderTradeTable();
      } catch (e) {
        // Silent failure — the placeholder stays. User can click
        // Recompute to re-fire the bundle flow including this prefetch.
        console.warn('[prefetch] default-outcome trade-data fetch failed:', e);
      }
    },

    async _ensureTradeMeta() {
      // Lazy-load trade_meta (~41 MB) the first time a user action needs
      // it. Cached on this.analyzeBundle.trade_meta for the session.
      // Single-ticker bundles always have trade_meta inline → no fetch.
      // Concurrent triggers (rapid clicks) share the same in-flight promise.
      if (this.analyzeBundle?.trade_meta) return;
      if (this.ticker !== 'ALL') return;
      if (this._tradeMetaInFlight) return this._tradeMetaInFlight;
      // v8: snapshot the bundle key the fetch was started for so a
      // stale promise resolving after the user clicked Analyze (and
      // moved on to a different cache_key) can't write old-metric data
      // into the new bundle. Silent no-op on key mismatch.
      const startedForKey = this._analyzeBundleKey();
      this._tradeMetaInFlight = (async () => {
        try {
          const url = this._analyzeBundleBaseUrl()
            .replace('/analyze-bundle?', '/analyze-bundle/trade-meta?');
          const r = await fetch(url);
          if (!r.ok) throw new Error(`HTTP ${r.status}`);
          const d = await r.json();
          if (d.error) throw new Error(d.error);
          if (this.analyzeBundle && this._analyzeBundleKey() === startedForKey) {
            this.analyzeBundle.trade_meta = d.trade_meta;
          }
        } finally {
          this._tradeMetaInFlight = null;
        }
      })();
      return this._tradeMetaInFlight;
    },

    async _ensureOutcome(outcome) {
      // Lazy-load one outcome's per_outcome_returns (~7.4 MB) the first
      // time a user action needs it. Cached on
      // this.analyzeBundle.per_outcome_returns[outcome] for the session.
      // Single-ticker bundles always have all outcomes inline → no fetch.
      // Per-outcome dedupe so a Gap-mode toggle that calls _ensureOutcome
      // for 1d_cc + 1d_oc in parallel doesn't double-fetch each.
      if (this.analyzeBundle?.per_outcome_returns?.[outcome]) return;
      if (this.ticker !== 'ALL') return;
      if (this._outcomesInFlight[outcome]) return this._outcomesInFlight[outcome];
      // v8: same cache-key snapshot guard as _ensureTradeMeta.
      const startedForKey = this._analyzeBundleKey();
      this._outcomesInFlight[outcome] = (async () => {
        try {
          const url = this._analyzeBundleBaseUrl()
            .replace('/analyze-bundle?', '/analyze-bundle/outcome?')
            + `&outcome=${encodeURIComponent(outcome)}`;
          const r = await fetch(url);
          if (!r.ok) throw new Error(`HTTP ${r.status}`);
          const d = await r.json();
          if (d.error) throw new Error(d.error);
          if (this.analyzeBundle && this._analyzeBundleKey() === startedForKey) {
            this.analyzeBundle.per_outcome_returns ??= {};
            this.analyzeBundle.per_outcome_returns[outcome] = d.data;
          }
        } finally {
          delete this._outcomesInFlight[outcome];
        }
      })();
      return this._outcomesInFlight[outcome];
    },

    // Wrap a set of deferred-fetch promises with the loading-overlay flag.
    // Idempotent — safe to call with zero needs (immediate return).
    async _runDeferred(promises) {
      const needs = promises.filter(Boolean);
      if (!needs.length) return;
      this._deferredLoading = true;
      try { await Promise.all(needs); }
      finally { this._deferredLoading = false; }
    },

    async refreshAnalyzeBundle() {
      // ALL mode: POST kicks off the background compute, then poll.
      // Single-ticker: GET computes inline; no separate refresh path.
      if (!this.ticker || !this.metric) return;
      if (this.ticker !== 'ALL') {
        return this.loadAnalyzeBundle();
      }
      this.analyzeBundleStatus = 'computing';
      this.analyzeBundleError  = null;
      try {
        const url = '/api/factor-analysis/analyze-bundle/refresh'
          + `?ticker=ALL`
          + `&metric=${encodeURIComponent(this.metric)}`
          + `&mode=${this._analyzeBundleMode()}`
          + (this.pageMode === 'train_test'
              ? `&cutoff_date=${encodeURIComponent(this.cutoffDate)}` : '');
        const r = await fetch(url, { method: 'POST' });
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        const d = await r.json();
        if (d.error) throw new Error(d.error);
        // 'computing' or 'busy' (another job already running for this key) —
        // both states converge to "wait and poll".
        this._startAnalyzeBundlePolling();
      } catch (e) {
        this.analyzeBundleStatus = 'failed';
        this.analyzeBundleError  = e.message;
      }
    },

    _startAnalyzeBundlePolling() {
      this._stopAnalyzeBundlePolling();
      const startTime = Date.now();
      const POLL_INTERVAL_MS = 10000;       // 10s between polls
      const POLL_TIMEOUT_MS  = 10 * 60 * 1000;   // 10 min ceiling
      this._analyzeBundlePollTimer = setInterval(async () => {
        if (Date.now() - startTime > POLL_TIMEOUT_MS) {
          this._stopAnalyzeBundlePolling();
          this.analyzeBundleStatus = 'failed';
          this.analyzeBundleError  = 'poll_timeout_10min';
          return;
        }
        try {
          const r = await fetch(this._analyzeBundleBaseUrl());
          if (!r.ok) return;   // transient
          const d = await r.json();
          if (d.status === 'ready') {
            // Stop polling FIRST so a slow payload fetch doesn't get
            // overlapped by another setInterval tick (the prior pile-up
            // pattern is what caused the event-loop starvation we just
            // fixed). Then download the 130MB body one time.
            this._stopAnalyzeBundlePolling();
            try {
              await this._fetchAnalyzeBundlePayload();
            } catch (e) {
              this.analyzeBundleStatus = 'failed';
              this.analyzeBundleError  = e.message;
            }
          } else if (d.status === 'not_computed' && d.previous_error) {
            this.analyzeBundleStatus = 'failed';
            this.analyzeBundleError  = d.previous_error;
            this._stopAnalyzeBundlePolling();
          }
          // 'computing' → keep polling
        } catch (_) {
          // transient; keep polling
        }
      }, POLL_INTERVAL_MS);
    },

    _stopAnalyzeBundlePolling() {
      if (this._analyzeBundlePollTimer) {
        clearInterval(this._analyzeBundlePollTimer);
        this._analyzeBundlePollTimer = null;
      }
    },

    // ── IC.5 — Signal Stability (universe-wide leaderboard + scatter) ────
    // Fetches /ic-batch for the current ticker + outcome + mode, then
    // renders either the leaderboard (horizontal bar, sign_stability ↓) or
    // the scatter (IC strength × stability). Click on any bar/dot sets the
    // Metric selector and fires /analyze below.

    // P2: Signal Survey's local outcome control. The on-change handler in
    // the template calls this with the new value. Persists to localStorage
    // and triggers a Signal Survey reload (the main chart is unaffected).
    setSurveyOutcome(newOutcome) {
      // Bucket A: dropdown change is an INSTANT in-memory swap to that
      // outcome's previously-cached slot — no network. If the (mode,
      // outcome) combo has never been loaded, the swap yields a null
      // display state and the placeholder shows ("not yet computed for
      // this outcome — click ⟳ Refresh"). User clicks Refresh to
      // compute + cache, and from then on subsequent visits to that
      // outcome are instant.
      if (newOutcome === this.surveyOutcome) return;
      try {
        localStorage.setItem('factor-analysis.signalSurvey.outcome', newOutcome);
      } catch (_) { /* private mode etc. — non-fatal */ }
      // Same persist-then-swap pattern as the mode pill, applied to the
      // outcome dimension.
      this._stopIcBatchPolling();
      this.icBatchSeq++;
      this._surveyStoreSlot();
      this.surveyOutcome = newOutcome;
      this._surveySwapDisplayFromSlot();
      // If the swap landed on an empty client slot, try a server-cache
      // lookup. Pure read — no compute kickoff — so a server-cache miss
      // yields the "not_ready · click Refresh" placeholder rather than
      // an implicit ~1–2 min single-ticker auto-compute.
      if (!this.icBatchData) this.loadIcBatch({ allowAutoCompute: false });
    },

    // ── Bucket A local-mode helpers (Signal Survey) ─────────────────────
    _surveyModeLabel(m) {
      if (m === 'in_sample')  return 'In-sample';
      if (m === 'train_test') return 'Train-test';
      return 'Walk-forward';
    },

    // Breadcrumb: "last: YYYY-MM-DD HH:MM:SS · <Mode label>" — identical
    // shape across all 6 panes. The IC-batch response carries cached_at
    // (ISO-ish string) which we slice to the first 19 chars to match
    // the YYYY-MM-DD HH:MM:SS format used by Score Matrix etc.
    surveyBreadcrumb() {
      const label = this._surveyModeLabel(this.surveyMode);
      if (!this.icBatchData?.metrics?.length) {
        return `no data yet · ${label}`;
      }
      const ts = this.icBatchData.cached_at
        ? String(this.icBatchData.cached_at).slice(0, 19)
        : 'unknown';
      return `last: ${ts} · ${label}`;
    },

    setSurveyMode(m) {
      if (m === this.surveyMode && m !== 'train_test') return;
      // IS↔WF are aliased to the same bucket — same data slot, no swap
      // dance needed. Just flip the pill highlight (and the breadcrumb's
      // mode label) so the UI reflects the selected pill, but DO NOT
      // bump icBatchSeq or stop polling — an in-flight load for the
      // shared bucket should continue uninterrupted.
      const oldBucket = this._surveyBucketFor(this.surveyMode);
      const newBucket = this._surveyBucketFor(m);
      if (oldBucket === newBucket) {
        this.surveyMode = m;
        return;
      }
      // Cross-bucket switch (anything ↔ TT): persist current and swap.
      this._stopIcBatchPolling();
      this.icBatchSeq++;
      this._surveyStoreSlot();
      this.surveyMode = m;
      this._surveySwapDisplayFromSlot();
      // Empty new slot → server cache lookup (no compute kickoff).
      if (!this.icBatchData) this.loadIcBatch({ allowAutoCompute: false });
    },

    setSurveyCutoffDate(d) {
      // Cutoff is only a slot dimension in train_test (only the TT bucket
      // partitions by cutoff). Outside TT, just remember the value for
      // next time the user enters TT.
      if (d === this.surveyCutoffDate) return;
      if (this.surveyMode !== 'train_test') {
        this.surveyCutoffDate = d;
        return;
      }
      this._stopIcBatchPolling();
      this.icBatchSeq++;
      this._surveyStoreSlot();
      this.surveyCutoffDate = d;
      this._surveySwapDisplayFromSlot();
      // Empty new slot → server cache lookup (no compute kickoff).
      if (!this.icBatchData) this.loadIcBatch({ allowAutoCompute: false });
    },

    // Map a mode to its slot bucket. IS and WF both bucket to "default"
    // (mathematically identical for rolling IC + backend keys both to
    // mode_tag="default"); TT gets its own bucket.
    _surveyBucketFor(mode) {
      return mode === 'train_test' ? 'train_test' : 'default';
    },

    // Slot key inside a bucket — outcome only for default; outcome+cutoff
    // for train_test.
    _surveySlotKey() {
      return this.surveyMode === 'train_test'
        ? `${this.surveyOutcome}|${this.surveyCutoffDate}`
        : this.surveyOutcome;
    },

    // Pull surveyDataByMode[bucket][slotKey] into the top-level icBatch*
    // / icDecomp* fields the template binds to. If the slot is empty,
    // reset top-level to nulls so the placeholder renders.
    _surveySwapDisplayFromSlot() {
      const slot = this.surveyDataByMode[this._surveyBucketFor(this.surveyMode)]?.[this._surveySlotKey()];
      if (slot) {
        this.icBatchData   = slot.icBatchData;
        this.icBatchKey    = slot.icBatchKey;
        this.icBatchStatus = slot.icBatchStatus;
        this.icBatchError  = slot.icBatchError;
        this.icDecompData  = slot.icDecompData;
        this.icDecompKey   = slot.icDecompKey;
        this.icDecompError = slot.icDecompError;
      } else {
        this.icBatchData   = null;
        this.icBatchKey    = null;
        this.icBatchStatus = null;
        this.icBatchError  = null;
        this.icDecompData  = null;
        this.icDecompKey   = null;
        this.icDecompError = null;
      }
      if (this.surveyExpanded && this.icBatchData?.metrics?.length) {
        this.$nextTick(() => {
          this._renderIcBatch();
          if (this.icDecompData) { this._renderIcDecomp(); this._renderIcLorenz(); }
        });
      }
    },

    // Snapshot the currently-displayed IC state into the slot for the
    // active (bucket, slotKey). Called after every successful load /
    // refresh / decomp response, and right before a setter swap.
    _surveyStoreSlot() {
      const b = this._surveyBucketFor(this.surveyMode);
      const k = this._surveySlotKey();
      if (!this.surveyDataByMode[b]) this.surveyDataByMode[b] = {};
      this.surveyDataByMode[b][k] = {
        icBatchData:   this.icBatchData,
        icBatchKey:    this.icBatchKey,
        icBatchStatus: this.icBatchStatus,
        icBatchError:  this.icBatchError,
        icDecompData:  this.icDecompData,
        icDecompKey:   this.icDecompKey,
        icDecompError: this.icDecompError,
      };
      // Status telemetry, keyed by surveyMode for visual consistency
      // (IS and WF report the same status because they share the bucket;
      // not currently read by any UI but kept for future debugging).
      const m = this.surveyMode;
      const cnt = this.icBatchData?.metrics?.length || 0;
      if (this.icBatchStatus === 'failed' || this.icBatchStatus === 'timeout') {
        this.surveyStatusByMode[m] = 'error';
      } else if (this.icBatchStatus === 'computing' || this.icBatchStatus === 'queued') {
        this.surveyStatusByMode[m] = 'loading';
      } else if (cnt > 0) {
        this.surveyStatusByMode[m] = 'ready';
      } else {
        this.surveyStatusByMode[m] = 'no_data';
      }
    },

    // Canonical cache-key for the current fetch context. Stored after a
    // successful load so expand/mode-change can detect stale data.
    _icBatchKey() {
      // W6: in_sample and walk_forward produce identical backend IC results
      // (rolling IC is mode-independent; only reference_ic post-processing differs,
      // and that's recomputed each time from the cached series).  Mapping both to
      // the same JS key prevents unnecessary cache-busting on mode toggle.
      // train_test gets a distinct key because the backend uses a different cache
      // entry (different reference IC based on pre-cutoff windows only).
      if (this.surveyMode === 'train_test') {
        return `${this.ticker}:${this.surveyOutcome}:train_test:${this.surveyCutoffDate}`;
      }
      return `${this.ticker}:${this.surveyOutcome}:default:`;
    },

    // Pure cache read against ic_batch_cache via GET /ic-batch.
    // `allowAutoCompute` (default true) controls whether a single-ticker
    // cache miss auto-fires refreshIcBatch (the ~1–2 min compute). When
    // called from a setter on an empty client slot, we pass
    // allowAutoCompute=false so the server is just queried for an
    // already-cached row; if there isn't one, the user sees the
    // "not_ready · click Refresh" placeholder rather than triggering an
    // implicit compute.
    async loadIcBatch(opts = {}) {
      const { allowAutoCompute = true } = opts;
      if (!this.ticker || !this.surveyOutcome) return;
      // Sequence guard: stamp this call. Any response that arrives after the
      // outcome/mode has changed will see a mismatched seq and bail without
      // touching state — prevents a stale "computing" response for 7d from
      // overwriting the just-rendered 5d data when the user switches mid-flight.
      const _seq = ++this.icBatchSeq;
      this.icBatchLoading = true;
      this.icBatchError = null;
      try {
        let url = `/api/factor-analysis/ic-batch?ticker=${encodeURIComponent(this.ticker)}`
          + `&outcome=${encodeURIComponent(this.surveyOutcome)}`;
        if (this.surveyMode === 'train_test') url += `&cutoff_date=${encodeURIComponent(this.surveyCutoffDate)}`;
        const r = await fetch(url);
        if (_seq !== this.icBatchSeq) return; // outcome/mode changed while awaiting — discard
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        const d = await r.json();
        if (_seq !== this.icBatchSeq) return; // changed during json parse — discard

        if (d.status === 'not_ready') {
          // No cache entry. Single-ticker auto-triggers a background job so
          // the UX matches the old "expand and wait" behavior without the 524.
          // ALL-mode requires an explicit ⟳ Refresh click (2-3 min job).
          this.icBatchStatus = 'not_ready';
          this.icBatchData   = null;
          if (this.ticker !== 'ALL' && allowAutoCompute) {
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
        // Persist whatever we just landed into the active mode's slot
        // so a mode-pill swap can restore it later. Runs for every
        // status branch (not_ready / computing / failed / data).
        this._surveyStoreSlot();
      } catch (e) {
        if (_seq !== this.icBatchSeq) return; // stale error — discard
        this.icBatchStatus = 'failed';
        this.icBatchError  = e.message;
        this._stopIcBatchPolling();
        this._surveyStoreSlot();
      } finally {
        if (_seq === this.icBatchSeq) this.icBatchLoading = false;
      }
    },

    async refreshIcBatch() {
      // POST /ic-batch/refresh for any ticker (single or ALL).
      // Returns immediately; background job writes cache; poll picks it up.
      if (!this.ticker || !this.surveyOutcome) return;
      // Seq guard: read current seq without bumping (loadIcBatch owns bumping).
      // If outcome/mode changes while the POST is in-flight, icBatchSeq will
      // have been bumped by the new loadIcBatch() call, and the stale POST
      // response will bail before writing any state.
      const _seq = this.icBatchSeq;
      this.icBatchLoading  = true;
      this.icBatchError    = null;
      this.icBatchData     = null;
      this.icBatchRefreshAt = Date.now(); // used by loadIcBatch to reject stale cache hits
      try {
        let url = `/api/factor-analysis/ic-batch/refresh?ticker=${encodeURIComponent(this.ticker)}`
          + `&outcome=${encodeURIComponent(this.surveyOutcome)}`;
        if (this.surveyMode === 'train_test') url += `&cutoff_date=${encodeURIComponent(this.surveyCutoffDate)}`;
        const r = await fetch(url, { method: 'POST' });
        if (_seq !== this.icBatchSeq) return; // outcome/mode changed mid-flight — discard
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        const d = await r.json();
        if (_seq !== this.icBatchSeq) return; // changed during json parse — discard
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
        this._surveyStoreSlot();
      } catch (e) {
        if (_seq !== this.icBatchSeq) return; // stale error — discard
        this.icBatchStatus = 'failed';
        this.icBatchError  = e.message;
        this._stopIcBatchPolling();
        this._surveyStoreSlot();
      } finally {
        if (_seq === this.icBatchSeq) this.icBatchLoading = false;
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
      // Supplementary subtitle alongside the standardized breadcrumb +
      // mode tag in the header. Shows horizon + status / metric counts.
      //
      // Bucket A: horizon comes from icBatchData.outcome — the actually-
      // displayed data's outcome — NOT this.surveyOutcome (the pending
      // dropdown value). Otherwise changing the outcome dropdown shifts
      // the breadcrumb's horizon before the user clicks Refresh,
      // misrepresenting what's on screen. Falls back to surveyOutcome
      // only when no data is displayed (so the pre-load placeholder
      // shows the pending selection).
      const displayedOutcome = this.icBatchData?.outcome || this.surveyOutcome;
      const _hz  = (displayedOutcome || '').match(/(\d+d)/)?.[0] || '';
      const _mod = this.surveyMode === 'train_test' ? 'train/test'
                 : this.surveyMode === 'walk_forward' ? 'walk-forward' : 'in-sample';
      const _ctx = _hz ? `${_hz} · ${_mod}` : _mod;
      if (this.icBatchLoading) return `${_ctx}`;
      if (this.icBatchStatus === 'not_ready') return `${_ctx} — not computed · click ⟳ Refresh to run`;
      if (this.icBatchStatus === 'computing')  return `${_ctx} — computing · polling every 30 s…`;
      if (this.icBatchStatus === 'queued')     return `${_ctx} — queued · waiting for IC job to finish…`;
      if (this.icBatchStatus === 'failed' || this.icBatchStatus === 'timeout') return _ctx;
      if (!this.icBatchData?.metrics?.length) return _ctx;
      const n    = this.icBatchData.metrics.length;
      const nSup = this.icBatchData.metrics.filter(m => m.suppressed).length;
      const xsec = this.ticker === 'ALL' ? 'cross-sectional' : 'time-series';
      let s = `${_ctx} · ${n} metrics · ${nSup} suppressed · ${xsec}`;
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
      // Bucket A: keyed on surveyDecompMetric (Survey-local), NOT
      // this.metric (Analyze section's metric). The two are intentionally
      // independent — drilling into a metric in the Survey's decomp pane
      // does not touch the Analyze section's selectors.
      const cut = this.surveyMode === 'train_test' ? this.surveyCutoffDate : '';
      return `${this.surveyDecompMetric}:${this.surveyOutcome}:${this.surveyMode}:${cut}`;
    },


    async loadIcDecomp(metric) {
      // `metric` arg is the Survey-local metric to drill into. Pass it
      // explicitly so callers don't have to mutate this.metric (which
      // would side-effect the Analyze section's selector). Falls back to
      // surveyDecompMetric for a "re-fetch current decomp" semantic.
      if (metric) this.surveyDecompMetric = metric;
      const m = this.surveyDecompMetric;
      if (this.ticker !== 'ALL' || !m || !this.surveyOutcome) return;
      this.icDecompLoading = true;
      this.icDecompError   = null;
      try {
        let url = `/api/factor-analysis/ic-decomp?metric=${encodeURIComponent(m)}`
          + `&outcome=${encodeURIComponent(this.surveyOutcome)}`;
        if (this.surveyMode === 'train_test') url += `&cutoff_date=${encodeURIComponent(this.surveyCutoffDate)}`;
        const r = await fetch(url);
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        const d = await r.json();
        if (d.error) throw new Error(d.error);
        this.icDecompData = d;
        this.icDecompKey  = this._icDecompKey();
        this.$nextTick(() => { this._renderIcDecomp(); this._renderIcLorenz(); });
        this._surveyStoreSlot();
      } catch (e) {
        this.icDecompError = e.message;
        this._surveyStoreSlot();
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

      // Bucket A: ticker-click in the decomp bubble chart used to set
      // this.ticker + force the ticker SELECT + fire loadAnalysis(), same
      // anti-pattern as the leaderboard metric-click. Removed. Hovering
      // a bubble still shows the tooltip; clicking is a no-op. The
      // Signal Survey is for fast exploration; deep-diving a ticker is
      // a deliberate separate action via the Analyze ticker selector.
      this._charts['ic-decomp'] = new Chart(el, {
        type: 'bubble',
        data: { datasets: [{ data: bubbleData, backgroundColor: bgColors, borderWidth: 0 }] },
        options: {
          responsive: true, maintainAspectRatio: false, animation: false,
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
    //
    // P2 (drill-in): propagate the Signal Survey outcome to the main chart's
    // `outcome` before firing /analyze. The user picked a metric in the
    // context of `surveyOutcome` (e.g., they were exploring 10d signals);
    // dropping them into the main chart at a different outcome (the
    // default ret_5d_fwd_oc) would be confusing. This matches what was
    // implicit pre-P2 when the upper bar dropdown drove both surfaces.
    _icBatchClickMetric(name) {
      // Bucket A: leaderboard click is a Survey-INTERNAL action that
      // populates the per-ticker decomposition panes for the chosen
      // metric and does NOTHING ELSE. The Analyze section is not
      // touched — no this.metric mutation, no this.outcome mutation,
      // no DOM force-set, no loadAnalysis() call. The Survey is for
      // fast exploration; deep-diving a metric in the primary visuals
      // is a deliberate separate action (user selects the metric in
      // the Analyze controls and clicks Analyze).
      //
      // Single-ticker (this.ticker !== 'ALL') has no decomp; the call
      // is gated on ALL mode. loadIcDecomp returns immediately for
      // non-ALL tickers regardless.
      if (this.ticker === 'ALL') this.loadIcDecomp(name);
    },
  }));
});
