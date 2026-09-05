'use strict';

// Factor Trades — exit-policy lab.
//
// The four shared panes (equity+DD, annual P&L, activity, ticker breakdown)
// render through window.FactorCharts, the SAME code Recall/Zone/Portfolio
// use — not a copy. Those functions take the component as their first
// argument and read state off it by name, so this component declares the
// same field names they expect (see the "FactorCharts contract" block
// below). Anything cosmetic about these charts changes in one place.

// Memo for the two-parameter marginalisation.
//
// Deliberately module-scope, NOT a field on the component. The scale it
// produces has to be readable from the span-slider bounds as well as from
// the grid, and the previous shape -- the gridHeat getter writing
// this._gridScale as a side effect -- meant the slider's min/max/step were
// correct only if the grid happened to be evaluated first. When it was not,
// maxAbs was still 0, gridSpanMax fell back to 1, and a Calmar matrix
// running 1.72-4.38 got a slider capped at 1 with every cell saturated.
//
// Keeping the cache out of the Alpine object also keeps it non-reactive:
// writing a reactive field from inside a getter is how render loops start.
// One factorTrades component exists per page, so a module-level cache is safe.
const _GRID_SCALE_0 = { autoSpan: 0.01, maxAbs: 0, lo: 0, hi: 0, mid: 0,
                        midIsNull: false, n: 0 };
let _gridHeatKey = null, _gridHeatVal = null, _gridScaleVal = _GRID_SCALE_0;
// Identity of the gridData the cache was built from. Comparing the object
// itself is what makes a fresh run invalidate reliably; a combo COUNT would
// collide whenever a re-run swept the same shape.
let _gridHeatData = null;

// Second memo, for gridStats. That getter runs _computeDollarSeries for every
// combination x BOTH windows -- the expensive work on this pane -- and it was
// uncached, so every read redid all of it. Both the 1D panels (gridMarginals)
// and the 2D matrix read it, so a single metric switch paid for it more than
// once. That is the multi-second pause on switching.
//
// It depends on the DATA and the DOLLAR SIZING only. Not on metric or window:
// _gridStats returns every metric for both windows in one object, so switching
// either is a re-read of a different field, not a recompute.
let _gridStatsKey = null, _gridStatsVal = null, _gridStatsData = null;

// ── TEMPORARY DIAGNOSTIC (remove once the matrix stale-value bug is closed) ──
// Off unless you set  window.FT_DEBUG = true  in the console.
//
// What it separates: whether gridHeat produces the NEW metric's numbers, and
// whether the DOM cell bindings are then painted with those numbers or with
// the previous ones. If COMPUTE reports the new metric while cellBg/fmt report
// the old values, the getter is fine and the x-for is not re-binding `cell`.
let _dbgWin = 0, _dbgCount = 0;
function _ftDbg(tag, obj) {
  if (typeof window === 'undefined' || !window.FT_DEBUG) return;
  console.log('[FT ' + tag + ']', obj);
}
// Throttled: the cell bindings fire once per cell per paint, so cap the noise
// at 3 lines per 250ms burst -- enough to see what a repaint carried.
function _ftDbgCell(tag, obj) {
  if (typeof window === 'undefined' || !window.FT_DEBUG) return;
  const now = Date.now();
  if (now - _dbgWin > 250) { _dbgWin = now; _dbgCount = 0; }
  if (_dbgCount++ >= 3) return;
  console.log('[FT ' + tag + ']', obj);
}

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

    // ── Signal / Portfolio ───────────────────────────────────────────────
    // ONE control panel, one toggle. In portfolio mode the metric pickers are
    // replaced by the saved-signal list and NOTHING ELSE CHANGES: exits,
    // sizing, window, baselines, grid and suite all apply to the combined
    // deduped set exactly as they apply to a zone. A portfolio is just a
    // trade set, so every pane below reads it without knowing the difference.
    //
    // The signals are the ones saved on Factor Analysis. There is no second
    // save mechanism here -- one store, nothing to drift.
    selMode: 'signal',            // 'signal' | 'portfolio'
    signals: [], signalsError: '', maxSelectable: 31,
    selectedSignalIds: [],
    get isPortfolio() { return this.selMode === 'portfolio'; },
    get eligibleSignals() { return this.signals.filter(s => s.eligible); },
    // The gate every action shares: a zone in signal mode, at least one
    // signal in portfolio mode. Stated once so a button cannot enable itself
    // under a selection the server would refuse.
    get hasSelection() {
      return this.isPortfolio ? this.selectedSignalIds.length > 0
                              : this.selectedCells.length > 0;
    },
    selected: {},                 // family -> rule_key (absent = family off)
    perTrade: 2000, dailyCap: 10000, maxStrike: 1000,
    loading: false, error: '',
    runs: [], currentIdx: -1, lockedIdx: -1,
    runData: null, lockedRun: null, zoneData: null, lockedZone: null,
    // Page-level window. TRAIN by default: the workflow is to iterate on
    // train and treat switching to test as a decision, not a default view.
    window: 'train',
    gridView: 'edited', showDD: true, fsId: null, _fsHome: null,
    // Synced crosshair (FactorCharts._crosshairPlugin). ON here: the whole
    // page is one policy read across panes, so lining up a date across
    // equity and activity is the normal question.
    crosshairSync: true, crosshairDate: null,
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
        const [cols, rules, tt, sigs] = await Promise.all([
          fetch('/api/factor-analysis/columns').then(r => r.ok ? r.json() : null),
          fetch('/api/factor-trades/rules').then(r => r.ok ? r.json() : null),
          fetch('/api/factor-analysis/tt-cutoff').then(r => r.ok ? r.json() : null),
          // Eligibility is decided server-side and arrives per signal, so the
          // disabled checkbox and the request the server would refuse cannot
          // disagree. Ineligible signals are LISTED, not hidden: a signal that
          // silently vanished is indistinguishable from one that was deleted.
          fetch('/api/factor-trades/signals').then(r => r.ok ? r.json() : null),
        ]);
        this.signals = sigs?.signals || [];
        this.signalsError = sigs?.error || '';
        if (sigs?.max_selectable) this.maxSelectable = sigs.max_selectable;
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
    // A bare "2" in a trail dropdown does not say 2 percent, 2 ATRs or 2
    // days. The unit is inferred from the PARAMETER NAME using the same
    // conventions _rule_label uses server-side, so the two cannot drift to
    // different answers for the same key. An unrecognised name renders the
    // raw value rather than guessing a unit onto it -- the dim name beside
    // the control is then the only claim being made.
    _dimUnit(dim) {
      const d = String(dim || '').toLowerCase();
      if (d.includes('pct') || d.includes('percent')) return 'pct';
      if (d === 'k' || d.includes('atr')) return 'atr';
      if (d === 'n' || d === 'days' || d === 'bars' || d.includes('day')) return 'day';
      return null;
    },
    famValLabel(dim, v) {
      const u = this._dimUnit(dim);
      if (u === 'pct') return v + '%';
      if (u === 'atr') return v + 'x ATR';
      if (u === 'day') return v + 'd';
      return String(v);
    },
    famDimHint(dim) {
      const u = this._dimUnit(dim);
      return u === 'pct' ? `${dim} — percent`
           : u === 'atr' ? `${dim} — multiples of ATR`
           : u === 'day' ? `${dim} — trading days`
           : `${dim} — unit not declared by the rule catalog; values shown raw`;
    },
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
    // Headers for canvases that full-screen as part of a split pane. Must
    // stay in step with the .ft-subhead text in the template.
    _FS_LABEL: { 'ft-reasons': 'Share of exits', 'ft-hold': 'Avg hold' },
    // Takes one id or a list. A pane split into sub-charts has to open as a
    // unit -- expanding only the left half of exit reasons drops the very
    // comparison the split was made for.
    openFs(id) {
      const ids = (Array.isArray(id) ? id : [id]).filter(Boolean);
      const els = ids.map(i => document.getElementById(i));
      if (els.some(e => !e)) return;
      this._fsHome = els.map(e => e.parentElement);
      this.fsId = ids;
      this.$nextTick(() => {
        const body = document.getElementById('ft-ov-body');
        if (!body) return;
        body.style.display = ids.length > 1 ? 'flex' : '';
        // BUILD THE WHOLE LAYOUT FIRST, RESIZE AFTERWARDS. Resizing inside
        // the append loop measured chart 0 while its cell was still the
        // only child of the overlay, so Chart.js sized that canvas to the
        // FULL overlay width and wrote it to the canvas' inline style.
        // Appending cell 1 then halved cell 0's box, but nothing resized
        // chart 0 again -- so the left canvas kept its full-width style and
        // drew straight across the right-hand chart. That is what produced
        // the share axis' "50%" ticks colliding with the hold axis' "40d"
        // and share gridlines running through the max_days bar. Any future
        // DOM mutation here must stay ahead of the resize pass.
        els.forEach((c, k) => {
          if (ids.length === 1) { body.appendChild(c); return; }
          // Each canvas needs its OWN positioned, sized box; Chart.js
          // measures the offset parent, and two canvases sharing one
          // would both size to the full width and overlap.
          const cell = document.createElement('div');
          cell.className = 'ft-ov-cell';
          // A rule between the panels, with real gutter either side. With
          // only a small gap the two charts read as one wide chart with a
          // seam in it, which is the confusion the split was made to fix.
          const last = k === els.length - 1;
          cell.style.cssText = 'flex:1 1 0;min-width:0;display:flex;flex-direction:column'
            + (k ? ';border-left:1px solid var(--border);padding-left:22px' : '')
            + (last ? '' : ';padding-right:22px');
          // The header travels with the chart. Full screen is where the
          // two halves sit furthest apart, so it is where an unlabelled
          // sub-chart is hardest to attribute.
          const h = document.createElement('div');
          // Styled inline, not via .ft-subhead: that rule styles its
          // CHILDREN, and this header is a leaf.
          h.style.cssText = 'flex:0 0 auto;text-align:center;font-size:11px;'
                          + 'font-weight:700;letter-spacing:.4px;padding-bottom:6px;'
                          + 'text-transform:uppercase;color:var(--muted)';
          h.textContent = this._FS_LABEL[ids[k]] || '';
          const host = document.createElement('div');
          host.style.cssText = 'flex:1;min-height:0;position:relative';
          host.appendChild(c);
          cell.appendChild(h);
          cell.appendChild(host);
          body.appendChild(cell);
        });
        // Every cell is in place and the flex boxes have settled; only now
        // does each chart measure a box that will not change under it.
        ids.forEach(i => this._charts[i]?.resize());
      });
    },
    closeFs() {
      const ids = Array.isArray(this.fsId) ? this.fsId : (this.fsId ? [this.fsId] : []);
      const homes = Array.isArray(this._fsHome) ? this._fsHome : [this._fsHome];
      // Move every canvas back BEFORE clearing state or tearing down the
      // wrapper cells, or they are destroyed along with the overlay.
      ids.forEach((i, k) => {
        const c = document.getElementById(i);
        if (c && homes[k]) homes[k].appendChild(c);
      });
      const body = document.getElementById('ft-ov-body');
      if (body) {
        body.querySelectorAll('.ft-ov-cell').forEach(n => n.remove());
        body.style.display = '';
      }
      this.fsId = null; this._fsHome = null;
      this.$nextTick(() => ids.forEach(i => this._charts[i]?.resize()));
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

    // Switching Signal <-> Portfolio CLEARS the current run rather than
    // leaving it on screen. A zone's charts and stat bar sitting under a
    // panel that says "Portfolio" is exactly the population confusion the
    // rest of this page is built to prevent: every number would be the old
    // trade set, and nothing on screen would say so. The signal selection and
    // the cell selection are both kept, so toggling back is free.
    setSelMode(m) {
      if (m === this.selMode) return;
      this.selMode = m;
      this.runData = null; this.zoneData = null; this.secDetail = null;
      this.lockedRun = null; this.lockedZone = null; this.lockedIdx = -1;
      this.suiteData = null; this.gridData = null; this.gridError = '';
      this.heatmapData = null; this._hmRange = null;
      this.error = '';
      if (m === 'portfolio') this.entryAnchor = 'open';
    },

    setWindow(w) {
      if (w === this.window) return;
      this.window = w;
      this._refreshGrid();
      // Both the run and the zone are window-scoped server-side, so the
      // whole page has to be refetched rather than re-filtered client-side.
      if (!this.runData) return;
      // A BASELINE MUST STAY A BASELINE across a window switch. run() is the
      // real-entry path; calling it here replaced the baseline with an
      // ordinary run and nothing on screen said the baseline had been
      // dropped -- you would be reading real-entry numbers having asked for
      // random ones.
      //
      // Re-running is the right answer rather than clearing, and it is
      // cheap: the sample is drawn per DATE across the zone's whole history
      // with no window filter, so train and test are two reports of the SAME
      // draw. Pinning the seed to the card being re-scoped guarantees that
      // -- switching window must re-scope the baseline, never re-roll it.
      if (this.runData.randomize) {
        if (this.runData.seed != null) this.baselineSeed = this.runData.seed;
        this.runBaseline(false, this.runData.baseline_kind || 'entry');
      } else {
        this.run();
      }
    },

    clearAll() {
      this.selected = {};
      this.selectedCells = [];
      this.selectedSignalIds = [];
      this.zoneData = null; this.lockedZone = null;
      this.error = '';
    },
    ruleKeys() { return Object.values(this.selected).filter(Boolean); },

    // ── The selection half of every request body ─────────────────────────
    // ONE builder, used by /zone, /grid, /suite and the baselines. The four
    // call sites previously each spelled out the metric pair and the cells,
    // which is four places for a portfolio to be forgotten in. Now there is
    // one, and it mirrors the server's single _resolve_selection.
    //
    // `src` is the run the request descends from (runData or a locked run) so
    // a locked card refetches under ITS OWN selection, not the rail's current
    // one.
    selectionBody(src) {
      const s = src || {};
      if ((s.mode || (this.isPortfolio ? 'portfolio' : '2f')) === 'portfolio') {
        return { signal_ids: (s.signals || []).length
                   ? s.signals.map(x => x.id) : this.selectedSignalIds.slice() };
      }
      return {
        primary_metric:   s.primary_metric   ?? this.primaryMetric,
        secondary_metric: s.secondary_metric ?? (this.mode === '2f' ? this.secondaryMetric : null),
        n_bins:           s.n_bins ?? 20,
        cells:            this.selectedCells,
      };
    },

    toggleSignal(sig) {
      if (!sig.eligible) return;
      // Every selectable signal enters at the open by construction (the
      // server only marks an _oc outcome eligible), so the anchor is not
      // a free choice here. Pinned on selection rather than at request
      // time, so the rail shows what will actually be traded.
      this.entryAnchor = 'open';
      const i = this.selectedSignalIds.indexOf(sig.id);
      if (i >= 0) { this.selectedSignalIds.splice(i, 1); return; }
      if (this.selectedSignalIds.length >= this.maxSelectable) {
        this.error = `at most ${this.maxSelectable} signals can be combined`;
        return;
      }
      this.selectedSignalIds.push(sig.id);
    },
    signalSelected(id) { return this.selectedSignalIds.includes(id); },

    // Bit i of sig_mask is signal i of the run's OWN ordered signal list, not
    // of the rail's current selection — those diverge as soon as a run is
    // locked and the selection is edited underneath it.
    signalLegend(src) { return (src || this.runData || {}).signals || []; },

    policyLabel() {
      const ks = this.ruleKeys();
      if (!ks.length) return 'no rules — backstop only';
      return ks.join(' + ');
    },

    // ── Random-entry baseline ────────────────────────────────────────────
    // Not a /run: the heatmap is a binning of the REAL population and has no
    // meaning for randomly-chosen entries. A baseline is a /zone with the
    // same policy, same anchor, same max strike, same window and the same
    // per-date trade counts, drawn from random tickers -- so it answers "is
    // this policy signal-specific, or would any long position under these
    // exits look like this".
    //
    // It needs a zone to exist, because the zone IS the distribution being
    // matched. Without one there is no shape to sample against.
    baselineSeed: null,       // null => the server's fixed default
    // Two baseline types, deliberately sharing the SAME entry draw for a
    // given seed. Their difference is then purely the exit rule, which is
    // what separates "does the metric pick better names than chance" from
    // "do the exit rules beat simply being in the market that long".
    // Confounded until you can run both.
    //
    // Three baselines complete the 2x2 against the policy run:
    //   policy      signal entries + rule   exits
    //   entry       random entries + rule   exits   -> selection vs chance
    //   exit        signal entries + random exits   -> exit rules ON MY TRADES
    //   entry_exit  random entries + random exits   -> the floor
    // The two random-EXIT kinds share an exit salt, and the two random-ENTRY
    // kinds share an entry ordering hash, so any pair differs by exactly one
    // factor. RE-SAMPLING RE-ROLLS BOTH DRAWS: they derive from one seed.
    BASELINE_KINDS: {
      entry:      { label: 'random entries', card: 'RANDOM ENTRIES',
                    row: 'rand entry',      colour: '#3498db' },
      exit:       { label: 'random exits', card: 'RANDOM EXITS',
                    row: 'rand exit',       colour: '#e84393' },
      entry_exit: { label: 'random entries + exits', card: 'RANDOM ENTRIES + EXITS',
                    row: 'rand entry+exit', colour: '#1abc9c' },
    },
    // ── One place that turns a fetch into {data} or {error} ──────────────
    //
    // The order here is the whole point, and getting it wrong sent a real
    // investigation to the wrong place: the previous code called r.json()
    // FIRST and only inspected r.ok afterwards, so a proxy's HTML error page
    // threw in the parser and was reported as "the response could not be
    // parsed — it is probably too large. Sweep fewer families." The response
    // began with <!DOCTYPE. It was not too large; it was not JSON, and the
    // status and content-type both said so before a single byte was parsed.
    //
    // So: status, then content-type, then parse — and when it is not JSON,
    // say what actually arrived rather than inferring a cause.
    async _postJson(url, body) {
      let r;
      try {
        r = await fetch(url, { method: 'POST',
                               headers: { 'Content-Type': 'application/json' },
                               body: JSON.stringify(body) });
      } catch (e) {
        // The connection itself failed — no status to report.
        return { error: `network error contacting ${url}: ${e}` };
      }
      const ct = (r.headers.get('content-type') || '').toLowerCase();
      if (!ct.includes('json')) {
        let head = '';
        try { head = (await r.text()).slice(0, 200).replace(/\s+/g, ' ').trim(); }
        catch (e) { head = '(body unreadable: ' + e + ')'; }
        const looksHtml = /^<!doctype|^<html/i.test(head);
        return { error:
          `${url} returned HTTP ${r.status} ${r.statusText || ''} as `
          + `${ct || 'an unspecified content type'}, not JSON.\n`
          + (looksHtml
              ? 'The body is an HTML page. This application only returns '
              + 'JSON, so it came from something in front of it — a reverse '
              + 'proxy timing out or refusing the response. Check the read '
              + 'timeout and body-size limits there; the app itself may well '
              + 'have logged 200.\n'
              : '')
          + `First bytes received: ${head}` };
      }
      if (!r.ok) {
        let d = null;
        try { d = await r.json(); } catch (e) { /* fall through to the status */ }
        return { error: d?.error || `HTTP ${r.status} ${r.statusText || ''}` };
      }
      try {
        const d = await r.json();
        if (d && d.error) return { error: d.error, data: d };
        return { data: d };
      } catch (e) {
        // JSON content-type, 200, and still unparseable — a genuinely
        // truncated or oversized body, which is the ONLY case where blaming
        // size is warranted.
        return { error: `${url} returned JSON that could not be parsed — the `
                      + `body was probably truncated in transit. ${e}` };
      }
    },

    // ── Parameter response grid ──────────────────────────────────────────
    // Second thin layer on the batch runner: the suite varies the seed, this
    // varies the parameters. The question is whether each parameter's effect
    // is SMOOTH AND CONSISTENT, not which combination scores best — so the
    // primary view is marginal, with a spread band, and there is deliberately
    // no ranking of combinations anywhere except the train/test scatter,
    // which exists to test whether the surface means anything at all.
    gridOpen: false,
    gridSweep: [],            // family names
    gridData: null,
    gridRunning: false,
    gridElapsed: 0,
    _gridTimer: null,
    gridMetric: 'calmar',
    gridWindow: 'train',      // which window the panels read
    gridPairX: '', gridPairY: '',
    // Its own error slot. Routing grid failures into the shared `error`
    // meant the pane's x-show went false and the whole thing evaporated --
    // progress text included -- with the reason parked in the rail where it
    // read as unrelated. A failed grid must say so where the grid was.
    gridError: '',
    gridShowScatter: false,
    _gridRange: 0.01,

    get gridFamilies() {
      const out = [];
      for (const g of (this.ruleGroups || [])) {
        for (const f of (g.families || [])) {
          if ((f.rules || []).length > 1) out.push({ family: f.family, n: f.rules.length });
        }
      }
      return out;
    },
    // {family: Set-like array of rule_keys}. Absent family = all values.
    gridValues: {},
    gridFamilyRules(fam) {
      for (const g of (this.ruleGroups || []))
        for (const f of (g.families || []))
          if (f.family === fam) return f.rules || [];
      return [];
    },
    gridValueOn(fam, rk) {
      const sel = this.gridValues[fam];
      return !sel || sel.includes(rk);
    },
    gridToggleValue(fam, rk) {
      const all = this.gridFamilyRules(fam).map(r => r.rule_key);
      let sel = this.gridValues[fam] ? [...this.gridValues[fam]] : [...all];
      const i = sel.indexOf(rk);
      if (i >= 0) sel.splice(i, 1); else sel.push(rk);
      // A family with nothing selected is not a sweep. Refuse the last
      // uncheck rather than sending a request the server will reject.
      if (!sel.length) return;
      this.gridValues = { ...this.gridValues, [fam]: all.filter(k => sel.includes(k)) };
    },
    gridFamilyValueCount(fam) {
      const sel = this.gridValues[fam];
      return sel ? sel.length : this.gridFamilyRules(fam).length;
    },
    // A 2D selection -- up to 9 families x 16 values -- does not fit a 290px
    // rail, and no amount of padding makes it fit. It opens in a modal
    // instead, reusing the full-screen overlay pattern already on this page.
    // The rail keeps a one-line summary per family and stays calm.
    gridCfgOpen: false,
    _gridShiftAnchor: {},          // family -> last clicked rule_key

    // Ordered families are the common case, so a range selection saves the
    // seven clicks it takes to drop three values.
    gridChipClick(fam, rk, ev) {
      const all = this.gridFamilyRules(fam).map(r => r.rule_key);
      if (ev && ev.shiftKey && this._gridShiftAnchor[fam] != null) {
        const a0 = all.indexOf(this._gridShiftAnchor[fam]);
        const a1 = all.indexOf(rk);
        if (a0 >= 0 && a1 >= 0) {
          const [lo, hi] = a0 <= a1 ? [a0, a1] : [a1, a0];
          const range = all.slice(lo, hi + 1);
          // Shift-click SETS the range rather than toggling each member, so
          // the result does not depend on what those members happened to be.
          this.gridValues = { ...this.gridValues, [fam]: range };
          return;
        }
      }
      this._gridShiftAnchor[fam] = rk;
      this.gridToggleValue(fam, rk);
    },
    gridSetAll(fam, on) {
      const all = this.gridFamilyRules(fam).map(r => r.rule_key);
      // "None" would make the family unsweepable, so it keeps the first
      // value rather than producing a request the server must reject.
      this.gridValues = { ...this.gridValues, [fam]: on ? all : all.slice(0, 1) };
    },
    gridDropped(fam) {
      const sel = this.gridValues[fam];
      if (!sel) return [];
      return this.gridFamilyRules(fam).filter(r => !sel.includes(r.rule_key));
    },
    // What this family's deselection is buying, in the unit that matters.
    // Shown before committing rather than discovered after a run.
    gridFamilyCost(fam) {
      if (!this.gridSweep.includes(fam)) return null;
      const dropped = this.gridDropped(fam);
      if (!dropped.length) return null;
      const full = this.gridFamilyRules(fam).length;
      const now = this.gridFamilyValueCount(fam);
      const other = this.gridComboCount / Math.max(1, now);
      return { dropped: dropped.map(r => r.label).join(', '),
               was: Math.round(other * full), is: this.gridComboCount };
    },
    // trail carries two sub-fields; a flat 16-chip run hides that structure.
    // Grouped by the first dimension so the grid reads as activation x
    // distance. Families with one dimension return a single unnamed group.
    gridChipGroups(fam) {
      const f = { family: fam, rules: this.gridFamilyRules(fam) };
      const dims = this.famDims(f);
      if (dims.length < 2) return [{ label: '', rules: f.rules }];
      const d0 = dims[0];
      const out = new Map();
      for (const r of f.rules) {
        const k = String((r.params || {})[d0] ?? '');
        if (!out.has(k)) out.set(k, []);
        out.get(k).push(r);
      }
      return [...out.entries()]
        .sort((a, b) => (+a[0]) - (+b[0]))
        .map(([k, rules]) => ({ label: `${d0} ${this.famValLabel(d0, k)}`, rules }));
    },
    gridSummary(fam) {
      const sel = this.gridValues[fam];
      const rules = this.gridFamilyRules(fam);
      const on = sel ? rules.filter(r => sel.includes(r.rule_key)) : rules;
      return on.map(r => r.label).join(', ');
    },
    gridToggleFamily(fam) {
      const i = this.gridSweep.indexOf(fam);
      if (i >= 0) this.gridSweep.splice(i, 1);
      else this.gridSweep.push(fam);
      this.gridSweep = [...this.gridSweep];
    },
    get gridComboCount() {
      if (!this.gridSweep.length) return 0;
      let n = 1;
      for (const fam of this.gridSweep) n *= this.gridFamilyValueCount(fam);
      // + the null, + one horizon-only reference per swept max_days value.
      const nref = this.gridSweep.includes('max_days')
        ? this.gridFamilyValueCount('max_days') : 0;
      return n + 1 + nref;
    },
    // Payload, not query time, is the binding cap now. Trades per combination
    // is only known after a run, so before one this uses the last grid's
    // count and says so by falling back to a round number.
    get gridTradesEach() { return this.gridData?.n_trades || 2500; },
    get gridPayloadMB() {
      return (this.gridComboCount * this.gridTradesEach * 9) / 1e6;
    },
    // One query for the whole sweep now, so the estimate no longer scales
    // with combinations -- it is a single scan plus vectorised arithmetic,
    // and the transfer of the returns arrays. Verification is the exception:
    // it deliberately reinstates per-combination queries.
    gridEstimate() {
      const q = 2;                                  // the single column fetch
      const xfer = this.gridComboCount * 0.004;     // ~4ms per combo of JSON
      return Math.max(2, Math.round(q + xfer + (this.gridVerify ? this.gridVerify * 1.2 : 0)));
    },
    // Diff N combinations against build_combine_sql before the grid renders.
    // Off by default: it costs exactly the per-combination queries the
    // restructuring removed. On when you want the oracle consulted.
    gridVerify: 0,

    async runGrid() {
      if (!this.runData) { this.error = 'run a policy first'; return; }
      if (!this.hasSelection) {
        this.error = this.isPortfolio ? 'select at least one signal first'
                                      : 'select a zone first';
        return;
      }
      if (!this.gridSweep.length) { this.error = 'pick at least one family to sweep'; return; }
      const src = this.runData;
      this.gridRunning = true; this.error = ''; this.gridError = '';
      this.gridElapsed = 0;
      this._gridTimer = setInterval(() => { this.gridElapsed += 1; }, 1000);
      try {
        const { data: d, error: err } = await this._postJson(
          '/api/factor-trades/grid', {
            ...this.selectionBody(src),
            entry_anchor: src.entry_anchor, rule_keys: src.rules,
            max_strike: src.max_strike ?? this.maxStrike,
            window: this.window,
            sweep_families: this.gridSweep,
            sweep_values: this.gridSweep.reduce((o, f) => {
              if (this.gridValues[f]) o[f] = this.gridValues[f];
              return o;
            }, {}),
            verify: +this.gridVerify || 0,
          });
        if (err) { this.gridError = err; return; }
        this.gridData = d;
        this.gridOpen = true;
        // First two swept families, and NEVER the same family on both axes —
        // a parameter plotted against itself gives one combination per cell,
        // which is what n=1 everywhere was saying.
        this.gridPairX = d.sweep?.[0]?.family || '';
        this.gridPairY = d.sweep?.[1]?.family || '';
        this.$nextTick(() => this.renderGrid());
      } catch (e) { this.gridError = String(e); }
      finally {
        this.gridRunning = false;
        if (this._gridTimer) { clearInterval(this._gridTimer); this._gridTimer = null; }
      }
    },

    // Axes must name two DIFFERENT families. Setting one to the other's
    // family swaps them rather than silently plotting a parameter against
    // itself.
    setGridPair(axis, fam) {
      if (axis === 'x') {
        if (fam === this.gridPairY) this.gridPairY = this.gridPairX;
        this.gridPairX = fam;
      } else {
        if (fam === this.gridPairX) this.gridPairX = this.gridPairY;
        this.gridPairY = fam;
      }
    },

    // Per-combination metrics. Dollars come from the SAME _computeDollarSeries
    // the stat bar uses, applied to the shared skeleton with this
    // combination's returns spliced in — one implementation, and the grid
    // cannot describe a differently-sized system than the rest of the page.
    _gridStats(c, win) {
      const d = this.gridData, sk = d.skeleton;
      const want = win === 'train' ? 1 : 0;
      const trades = [];
      for (let i = 0; i < sk.w.length; i++) {
        if (sk.w[i] !== want) continue;
        const ret = c.r[i];
        if (ret == null) continue;
        trades.push({ ticker: d.tickers[sk.t[i]], trade_date: d.dates[sk.d[i]],
                      ret, spot_entry_raw: sk.p[i] });
      }
      const base = { avg_ret: c[win]?.avg_ret ?? null, n: c[win]?.n ?? 0,
                     avg_hold: c[win]?.avg_hold ?? null };
      if (!trades.length) {
        return { ...base, total_ret: null, max_dd: null, calmar: null, exit_share: null };
      }
      const ds = window.FactorCharts._computeDollarSeries(
        this, trades, this.perTrade, this.dailyCap);
      const $d = this._dollarStats(ds);
      // Exit share of the rules this combination actually selected — the
      // question "how often did this stop fire" only means anything about
      // the rules in the combination.
      let share = null;
      const rk = Object.values(c.params || {});
      if (rk.length) {
        const rs = c.reasons?.[win] || {};
        share = rk.reduce((a, k) => a + (rs[k] || 0), 0);
      }
      return { ...base, total_ret: $d?.total_ret_usd ?? null,
               max_dd: $d?.max_dd_usd ?? null, calmar: $d?.calmar ?? null,
               exit_share: share };
    },

    // Every combination x both windows, computed once per data/sizing change.
    get gridStats() {
      const d = this.gridData;
      if (!d) return null;
      // perTrade / dailyCap are live inputs and they feed
      // _computeDollarSeries, so calmar, max_dd and total_ret all move when
      // position sizing changes. They belong in the key; metric and window
      // do not.
      const key = `${this.perTrade}|${this.dailyCap}`;
      if (key === _gridStatsKey && _gridStatsData === d && _gridStatsVal) {
        return _gridStatsVal;
      }
      _gridStatsVal = (d.combos || []).map(c => c.error ? null : ({
        combo: c,
        train: this._gridStats(c, 'train'),
        test:  this._gridStats(c, 'test'),
      }));
      _gridStatsKey = key;
      _gridStatsData = d;
      return _gridStatsVal;
    },

    // THE PRIMARY VIEW. For each swept family, the metric averaged across
    // every setting of every OTHER family, with the spread across those
    // settings. A narrow band means the parameter's effect does not depend
    // on context; a wide band means it interacts with something, which is
    // the cue to open the 2D view and find out what.
    get gridMarginals() {
      const d = this.gridData, st = this.gridStats;
      if (!d || !st) return [];
      const win = this.gridWindow, mk = this.gridMetric;
      return (d.sweep || []).map(f => {
        const points = f.values.map(v => {
          const vals = [];
          for (const s of st) {
            if (!s || s.combo.is_null) continue;      // the null combo is a
            if (s.combo.params[f.family] !== v.rule_key) continue;  // reference,
            const x = s[win]?.[mk];                   // not a grid point
            if (x != null && isFinite(x)) vals.push(+x);
          }
          if (!vals.length) return { label: v.label, mean: null, lo: null, hi: null,
                                     sdLo: null, sdHi: null, n: 0 };
          const mean = vals.reduce((a, b) => a + b, 0) / vals.length;
          const sd = vals.length > 1
            ? Math.sqrt(vals.reduce((a, b) => a + (b - mean) ** 2, 0) / vals.length) : 0;
          // p5-p95 rather than true min-max: one broken combination should
          // not set the outer band, and the band should not widen simply
          // because the sweep got bigger. With few combinations per point it
          // degenerates toward min-max, which is the honest result.
          const p = this._pct(vals);
          return { label: v.label, mean, n: vals.length,
                   lo: p(5), hi: p(95),
                   sdLo: mean - sd, sdHi: mean + sd,
                   rawLo: Math.min(...vals), rawHi: Math.max(...vals) };
        });
        // values ride along so the max_days panel can look its reference up
        // by rule_key rather than by position, which would break the moment
        // a value filter reorders or drops one.
        return { family: f.family, values: f.values, points };
      });
    },

    // Linear-interpolated percentile, the same convention numpy uses, so a
    // band drawn here and a percentile quoted server-side agree.
    _pct(vals) {
      const a = [...vals].sort((x, y) => x - y);
      return (q) => {
        if (!a.length) return null;
        if (a.length === 1) return a[0];
        const i = (q / 100) * (a.length - 1);
        const lo = Math.floor(i), hi = Math.ceil(i);
        return lo === hi ? a[lo] : a[lo] + (a[hi] - a[lo]) * (i - lo);
      };
    },

    // Null combination as a reference line on every panel — it is the thing
    // every parameter setting has to beat, so it belongs ON the chart rather
    // than in a number to be remembered.
    // "Do nothing" is not one number when max_days is swept: horizon-only
    // at 5 days and at 20 days are different baselines. The server emits one
    // reference combination per swept horizon; this reads them.
    //
    //   max_days panel  -> the CURVE, one reference per x value
    //   other panels    -> the mean over the SELECTED horizons, matching how
    //                      those panels marginalise
    //   max_days unswept-> the single null, a flat line as before
    get gridRefs() {
      const d = this.gridData, st = this.gridStats;
      if (!d || !st) return { byValue: {}, mean: null, kind: 'none' };
      const win = this.gridWindow, mk = this.gridMetric;
      const byValue = {};
      const vals = [];
      for (const [rk, ix] of Object.entries(d.ref_index || {})) {
        const v = st[ix]?.[win]?.[mk];
        if (v != null && isFinite(v)) { byValue[rk] = v; vals.push(v); }
      }
      if (vals.length) {
        return { byValue, kind: 'curve',
                 mean: vals.reduce((a, b) => a + b, 0) / vals.length };
      }
      const nul = st.find(x => x && x.combo.is_null);
      const v = nul ? nul[win]?.[mk] : null;
      return { byValue: {}, kind: (v != null && isFinite(v)) ? 'flat' : 'none',
               mean: (v != null && isFinite(v)) ? v : null };
    },
    // Kept for the flat case; panels use gridRefs directly.
    get gridNullValue() { return this.gridRefs.mean; },

    gridFmt(v) {
      _ftDbgCell('fmt', { metric: this.gridMetric, raw: v });
      if (v == null || !isFinite(v)) return '—';
      const u = (this.gridData?.metrics || []).find(m => m.key === this.gridMetric)?.unit;
      if (u === 'ratio') return (+v).toFixed(2);
      if (u === 'sess') return (+v).toFixed(1);
      if (u === 'usd') {
        const a = Math.abs(v), sg = v < 0 ? '-' : '';
        if (a >= 1e6) return sg + '$' + (a / 1e6).toFixed(2) + 'M';
        if (a >= 1e3) return sg + '$' + (a / 1e3).toFixed(1) + 'k';
        return sg + '$' + a.toFixed(0);
      }
      return ((+v) * 100).toFixed(2) + '%';
    },

    // ── 2D view ──────────────────────────────────────────────────────────
    // Marginalised over every family that is not one of the two axes, so a
    // cell is "this pair at these settings, averaged over the rest" rather
    // than one arbitrary slice.
    get gridHeat() {
      const d = this.gridData;
      if (!d || !this.gridPairX || !this.gridPairY) return null;
      const fx = (d.sweep || []).find(f => f.family === this.gridPairX);
      const fy = (d.sweep || []).find(f => f.family === this.gridPairY);
      if (!fx || !fy || fx.family === fy.family) return null;
      // Memo, checked BEFORE touching gridStats -- that getter maps every
      // combination, so reading it above the cache check would make a "hit"
      // as expensive as a miss. The span slider is deliberately NOT in the
      // key: changing the span must recolour, never re-marginalise.
      // Every input that changes what this matrix DISPLAYS:
      //   pairX/pairY  which families are on the axes
      //   window       train vs test
      //   metric       which column of the stats object is read
      //   colourBy     moves the anchor, which moves autoSpan
      //   perTrade/dailyCap  dollar sizing -> calmar, max_dd, total_ret
      // Deliberately absent: the span slider (recolour only, never
      // re-marginalise) and gridValues (a pre-RUN filter -- the matrix is
      // built from d.sweep, i.e. what the completed run actually swept).
      const memoKey = [this.gridPairX, this.gridPairY, this.gridWindow,
                       this.gridMetric, this.gridColourBy,
                       this.perTrade, this.dailyCap].join('|');
      if (memoKey === _gridHeatKey && _gridHeatData === d && _gridHeatVal) {
        _ftDbg('gridHeat MEMO-HIT', {
          metric: this.gridMetric, window: this.gridWindow,
          firstCell: _gridHeatVal.grid?.[0]?.[0]?.avg_ret });
        return _gridHeatVal;
      }
      const st = this.gridStats;
      if (!st) return null;
      const win = this.gridWindow, mk = this.gridMetric;
      const grid = fy.values.map(vy => fx.values.map(vx => {
        const vals = [];
        for (const s of st) {
          if (!s || s.combo.is_null) continue;
          if (s.combo.params[fx.family] !== vx.rule_key) continue;
          if (s.combo.params[fy.family] !== vy.rule_key) continue;
          const x = s[win]?.[mk];
          if (x != null && isFinite(x)) vals.push(+x);
        }
        if (!vals.length) return { n: 0, avg_ret: 0 };
        return { n: vals.length,
                 avg_ret: vals.reduce((a, b) => a + b, 0) / vals.length };
      }));
      // SCALE MAX, ANCHORED AT ZERO.
      //
      // The ramp is _hmPaint's: zero is dark, brightness grows with distance
      // from zero, terminating at #3498db / #e84393. So the only thing this
      // has to produce is the MAGNITUDE at which the ramp saturates.
      //
      // A previous version stretched the ramp so the matrix MINIMUM went
      // dark. That is wrong: with Calmars running 1.72-3.82 it painted 1.72
      // near-black, which reads as zero when it is not zero. Zero is dark;
      // nothing else is.
      //
      // autoSpan is p95 of the distance from the anchor -- a default, not a
      // rule. The span slider overrides it, because the value ranges differ
      // enormously between metrics (tight for Calmar, wide for dollars) and
      // no auto-scaling heuristic beats seeing the matrix and setting it.
      const anchor = this._gridAnchor;
      const dists = [], vals = [];
      for (const row of grid) for (const c of row) if (c.n) {
        vals.push(c.avg_ret);
        dists.push(Math.abs(c.avg_ret - anchor));
      }
      const pd = this._pct(dists);
      const mid = this.gridRefs.mean;
      const maxAbs = dists.length ? Math.max(...dists) : 0;
      _gridScaleVal = {
        autoSpan: dists.length ? (pd(95) || maxAbs || 0.01) : 0.01,
        maxAbs,
        lo: vals.length ? Math.min(...vals) : 0,
        hi: vals.length ? Math.max(...vals) : 0,
        mid: (mid != null && isFinite(mid)) ? mid : (vals.length ? this._pct(vals)(50) : 0),
        midIsNull: mid != null && isFinite(mid),
        n: vals.length,
      };
      _gridHeatKey = memoKey;
      _gridHeatData = d;
      _ftDbg('gridHeat COMPUTE', {
        metric: this.gridMetric, window: this.gridWindow,
        firstCell: grid?.[0]?.[0]?.avg_ret,
        cells: grid.flat().filter(c => c.n).length });
      _gridHeatVal = { grid, x_labels: fx.values.map(v => v.label),
                       y_labels: fy.values.map(v => v.label) };
      return _gridHeatVal;
    },
    // Reading the scale ENSURES it is computed, rather than hoping the grid
    // was rendered first. Memoised, so this is a key comparison after the
    // first call -- cheap enough for the slider bounds and the colourbar.
    get _gridScale() {
      this.gridHeat;
      return _gridScaleVal || _GRID_SCALE_0;
    },
    // Anchor for the ramp. The ramp itself always treats ZERO as dark, so
    // "vs baseline" is expressed by subtracting the null from each cell
    // before painting -- the caller shifts the value, the ramp is untouched.
    //   'value'    anchored at ZERO, same as the node heatmap.
    //   'baseline' anchored at the null: what beats doing nothing.
    gridColourBy: 'value',
    setGridColourBy(m) { this.gridColourBy = m; },
    get _gridAnchor() {
      if (this.gridColourBy !== 'baseline') return 0;
      const m = this.gridRefs.mean;
      return (m != null && isFinite(m)) ? m : 0;
    },

    // ── Span control ─────────────────────────────────────────────────────
    // ONE control: the magnitude at which the ramp reaches full #3498db /
    // #e84393. Zero stays pinned to the dark end; this moves the top only,
    // and the same magnitude applies to both signs so +2.0 and -2.0 are
    // equally bright. Values past it clamp at the terminal colour.
    //
    // null means "auto" -- p95 of the displayed cells' distance from the
    // anchor, recomputed per matrix, since Calmar and dollar metrics have
    // nothing in common range-wise.
    gridSpanManual: null,
    get gridSpan() {
      const m = this.gridSpanManual;
      if (m != null && isFinite(m) && m > 0) return m;
      return this._gridScale.autoSpan || 0.01;
    },
    get gridSpanIsAuto() { return this.gridSpanManual == null; },
    // Slider bounds track the matrix so one slider serves every metric.
    // Top is a little past the largest cell, so the ramp can be relaxed
    // until nothing clamps as well as tightened until most things do.
    get gridSpanMax() {
      const m = this._gridScale.maxAbs || 0;
      return m > 0 ? m * 1.25 : 1;
    },
    get gridSpanMin()  { return this.gridSpanMax / 200; },
    get gridSpanStep() { return this.gridSpanMax / 500; },
    setGridSpan(v) {
      const f = parseFloat(v);
      if (!isFinite(f) || f <= 0) return;
      this.gridSpanManual = Math.max(this.gridSpanMin,
                                     Math.min(this.gridSpanMax, f));
    },
    gridSpanAuto() { this.gridSpanManual = null; },

    // PERFORMANCE: this is called once per cell on every render, and the
    // span slider re-runs exactly these bindings while it is dragged. It
    // must stay O(1) and must NOT touch `gridHeat` -- that getter
    // re-marginalises over every swept combination, and reading it here is
    // what made the old rank mode crawl (it rebuilt and re-sorted the whole
    // matrix once per cell, so O(cells^2 x combos) per repaint).
    //
    // Nothing here reads gridHeat, so changing the span invalidates only the
    // colour bindings: cells recolour, the grid is not rebuilt or re-laid
    // out, and no request is made.
    // Span read used ONLY by the per-cell paint. Unlike `gridSpan` it does
    // not go through _gridScale, so it never re-enters the gridHeat getter:
    // 25 cells x several bindings was ~170 memo lookups per render, all of
    // them asking a question the grid had already answered. Safe because a
    // cell can only exist if gridHeat just produced it, which is the same
    // pass that writes _gridScaleVal -- so the value is current by
    // construction rather than by ordering luck.
    get _gridSpanForPaint() {
      const m = this.gridSpanManual;
      if (m != null && isFinite(m) && m > 0) return m;
      return (_gridScaleVal && _gridScaleVal.autoSpan) || 0.01;
    },
    gridCellBg(cell) {
      _ftDbgCell('cellBg', { metric: this.gridMetric, cellValue: cell && cell.avg_ret,
                             span: this._gridSpanForPaint });
      if (!cell || !cell.n) return window.FactorCharts._hmPaint(1, 0, cell);
      // Same function the node heatmap and trade-activity grid call, same
      // default path, no opts. minSampleN 0: `n` here is a count of
      // combinations averaged, not trades, so hatching does not apply.
      return window.FactorCharts._hmPaint(
        this._gridSpanForPaint, 0,
        { n: cell.n, avg_ret: (cell.avg_ret || 0) - this._gridAnchor });
    },
    gridCellTitle(cell) {
      if (!cell || !cell.n) return 'no combinations';
      const S = this._gridScale;
      const d = (cell.avg_ret ?? 0) - S.mid;
      const shifted = Math.abs((cell.avg_ret ?? 0) - this._gridAnchor);
      return `${this.gridFmt(cell.avg_ret)} — mean over ${cell.n} combination(s)`
           + `
${d >= 0 ? '+' : ''}${this.gridFmt(d)} vs `
           + (S.midIsNull ? 'do nothing' : 'the matrix median')
           + (shifted > this.gridSpan ? `
beyond the scale max (${this.gridFmt(this.gridSpan)}) — clamped` : '');
    },
    // Colourbar stops, so the ramp can be read rather than guessed at.
    // Symmetric -span..+span around the anchor, which is what the ramp
    // actually does — showing only the positive half would hide that zero
    // is the dark end.
    get gridBarStops() {
      const out = [];
      const span = this.gridSpan;
      for (let i = 0; i <= 20; i++) {
        const t = (i / 20) * 2 - 1;             // -1 .. +1
        out.push(window.FactorCharts._hmPaint(span, 0, { n: 1, avg_ret: t * span }));
      }
      return out;
    },
    get gridBarLo() { return this._gridAnchor - this.gridSpan; },
    get gridBarHi() { return this._gridAnchor + this.gridSpan; },
    // Where the null sits on the bar. In vs-baseline it IS the anchor, so
    // dead centre; in value mode it is wherever "do nothing" falls.
    get gridBarMidPct() {
      const S = this._gridScale;
      const lo = this.gridBarLo, hi = this.gridBarHi;
      if (!S.n || hi === lo) return 50;
      return Math.max(0, Math.min(100, ((S.mid - lo) / (hi - lo)) * 100));
    },
    // True when the null falls outside the painted range, in which case the
    // tick is pinned to an edge and says so rather than implying the
    // reference sits inside the data.
    get gridBarNullOff() {
      const S = this._gridScale;
      if (!S.n || !S.midIsNull) return '';
      return S.mid > this.gridBarHi ? 'above'
           : (S.mid < this.gridBarLo ? 'below' : '');
    },
    // How many cells are clamped at a terminal colour — worth stating, since
    // clamping is expected but silently clamping everything is not useful.
    get gridClampedN() {
      const g = this.gridHeat;
      if (!g) return 0;
      const span = this.gridSpan, a = this._gridAnchor;
      let k = 0;
      for (const row of g.grid) for (const c of row) {
        if (c.n && Math.abs((c.avg_ret || 0) - a) > span) k++;
      }
      return k;
    },

    // ── Rank scatter ─────────────────────────────────────────────────────
    // Free: is_train is a column, so every combination already carries both
    // windows. Does a good train combination stay good out of sample? If the
    // cloud is shapeless the surface is noise and the marginal panels above
    // are describing nothing.
    get gridScatter() {
      const st = this.gridStats;
      if (!st) return [];
      const rows = st.filter(Boolean)
        .map(s => ({ train: s.train?.calmar, test: s.test?.calmar,
                     isNull: s.combo.is_null, labels: s.combo.labels }))
        .filter(r => r.train != null && r.test != null && isFinite(r.train) && isFinite(r.test));
      const ranked = rows.filter(r => !r.isNull).sort((a, b) => b.train - a.train).slice(0, 20);
      const nul = rows.find(r => r.isNull);
      return nul ? [...ranked, nul] : ranked;
    },

    renderGrid() {
      if (!this.gridData) return;
      const panes = [['marginals', () => this._renderGridMarginals()],
                     ['scatter',   () => this._renderGridScatter()]];
      for (const [name, fn] of panes) {
        try { fn(); } catch (e) { console.error(`[factor-trades] grid ${name} failed`, e); }
      }
    },

    // ONE PANEL. The null is a reference, not a benchmark: it is a 20-day
    // hold, and a policy returning 80% of it in 25% of the days is BETTER,
    // because it frees capital. A second plot devoted to value/null encoded
    // the opposite -- that everything is judged against it -- and cost a
    // whole axis to carry one ratio.
    //
    // The ratio is now a muted row of text under the x labels. Same
    // information, nothing to mistake for a data series, no axis to misread.
    _renderGridMarginals() {
      const refs = this.gridRefs;
      const fmt = (v) => this.gridFmt(v);
      for (const m of this.gridMarginals) {
        const id = 'grid-mg-' + m.family;
        const el = document.getElementById(id);
        if (this._charts[id]) { this._charts[id].destroy(); delete this._charts[id]; }
        if (!el) continue;
        const labels = m.points.map(p => p.label);
        const refSeries = (m.family === 'max_days' && refs.kind === 'curve')
          ? m.values.map(v => refs.byValue[v.rule_key] ?? null)
          : (refs.mean != null ? labels.map(() => refs.mean) : null);

        const ds = [
          { data: m.points.map(p => p.hi), borderWidth: 0, pointRadius: 0,
            fill: '+1', backgroundColor: 'rgba(52,152,219,.08)' },
          { data: m.points.map(p => p.lo), borderWidth: 0, pointRadius: 0, fill: false },
          { data: m.points.map(p => p.sdHi), borderWidth: 0, pointRadius: 0,
            fill: '+1', backgroundColor: 'rgba(52,152,219,.20)' },
          { data: m.points.map(p => p.sdLo), borderWidth: 0, pointRadius: 0, fill: false },
          { data: m.points.map(p => p.mean), borderColor: '#3498db', borderWidth: 2,
            pointRadius: 3, pointBackgroundColor: '#3498db', fill: false, tension: 0 },
        ];
        if (!this.gridShowBands) ds.splice(0, 4);
        const meanIx = this.gridShowBands ? 4 : 0;
        if (refSeries) {
          ds.push({ data: refSeries, borderColor: 'rgba(200,200,200,.8)',
                    borderWidth: 1.5, borderDash: [5, 4], pointRadius: 0,
                    fill: false, tension: 0, label: 'do nothing' });
        }

        // The axis is computed from the DATA series only. The reference is
        // still drawn; it just cannot vote on the range, so a baseline
        // several times the policies cannot squeeze them into a sliver.
        let lo = Infinity, hi = -Infinity;
        for (const d of (this.gridShowBands ? ds.slice(0, 5) : ds.slice(0, 1))) {
          for (const v of d.data) if (v != null && isFinite(v)) {
            lo = Math.min(lo, v); hi = Math.max(hi, v);
          }
        }
        if (!isFinite(lo)) { lo = 0; hi = 1; }
        const pad = (hi - lo) * 0.08 || Math.abs(hi || 1) * 0.08;
        const yMin = lo - pad, yMax = hi + pad;
        const nullV = refs.mean;
        // Per-point ratio, drawn under the tick labels.
        const ratios = m.points.map((p, i) => {
          const r = refSeries ? refSeries[i] : null;
          if (!r || p.mean == null || !isFinite(p.mean)) return null;
          return p.mean / r;
        });

        this._charts[id] = new Chart(el.getContext('2d'), {
          type: 'line',
          data: { labels, datasets: ds },
          options: {
            responsive: true, maintainAspectRatio: false, animation: false,
            // Room for the ratio row without shrinking the plot into it.
            layout: { padding: { bottom: 14 } },
            plugins: {
              legend: { display: false },
              tooltip: {
                backgroundColor: 'rgba(20,20,20,.95)', borderColor: '#444', borderWidth: 1,
                filter: (i) => i.datasetIndex === meanIx,
                callbacks: { label: (c) => {
                  const p = m.points[c.dataIndex];
                  const out = [`mean ${fmt(p.mean)}`,
                               `+/-1sd ${fmt(p.sdLo)} … ${fmt(p.sdHi)}`,
                               `p5-p95 ${fmt(p.lo)} … ${fmt(p.hi)}`,
                               `over ${p.n} combination(s)`];
                  const r = refSeries ? refSeries[c.dataIndex] : null;
                  if (r != null) {
                    out.push(`do nothing: ${fmt(r)}`);
                    const q = ratios[c.dataIndex];
                    if (q != null) out.push(`${q.toFixed(2)}x it`);
                  }
                  return out;
                } },
              },
            },
            scales: {
              x: { ticks: { color: '#888', font: { size: 10 } }, grid: { display: false } },
              y: { min: yMin, max: yMax,
                   ticks: { color: '#888', font: { size: 10 }, callback: fmt },
                   grid: { color: '#222' } },
            },
          },
          plugins: [{
            id: 'gridNullText',
            afterDatasetsDraw(chart) {
              const xs = chart.scales.x, ys = chart.scales.y;
              if (!xs || !ys) return;
              const c = chart.ctx;
              c.save();
              if (nullV != null) {
                // Named, in the corner, in real units. No arrow: the null is
                // a reference point, and an arrow pointing at it implies a
                // ceiling to be measured against, which it is not.
                c.font = '10px monospace';
                c.fillStyle = 'rgba(210,210,210,.9)';
                c.textAlign = 'right';
                c.fillText(`do nothing: ${fmt(nullV)}`, xs.right - 2, ys.top + 11);
              }
              // The ratio row: muted, smaller, under the tick labels.
              c.font = '9px monospace';
              c.fillStyle = 'rgba(150,150,150,.85)';
              c.textAlign = 'center';
              ratios.forEach((q, i) => {
                if (q == null) return;
                c.fillText(q.toFixed(2) + 'x', xs.getPixelForValue(i), xs.bottom + 11);
              });
              c.restore();
            },
          }],
        });
      }
    },
    gridShowBands: true,
    toggleGridBands() { this.gridShowBands = !this.gridShowBands; this.$nextTick(() => this.renderGrid()); },
    setGridMetric(m) {
      _ftDbg('setGridMetric', { from: this.gridMetric, to: m });
      this.gridMetric = m; this.$nextTick(() => this.renderGrid());
    },
    setGridWindow(w) {
      _ftDbg('setGridWindow', { from: this.gridWindow, to: w });
      this.gridWindow = w; this.$nextTick(() => this.renderGrid());
    },

    _renderGridScatter() {
      const id = 'grid-scatter';
      const el = document.getElementById(id);
      if (this._charts[id]) { this._charts[id].destroy(); delete this._charts[id]; }
      if (!el || !this.gridShowScatter) return;
      const pts = this.gridScatter;
      if (!pts.length) return;
      this._charts[id] = new Chart(el.getContext('2d'), {
        type: 'scatter',
        data: { datasets: [
          { data: pts.filter(p => !p.isNull).map(p => ({ x: p.train, y: p.test })),
            backgroundColor: 'rgba(52,152,219,.75)', pointRadius: 4 },
          { data: pts.filter(p => p.isNull).map(p => ({ x: p.train, y: p.test })),
            backgroundColor: '#e84393', pointRadius: 7, pointStyle: 'rectRot' },
        ] },
        options: {
          responsive: true, maintainAspectRatio: false, animation: false,
          plugins: {
            legend: { display: false },
            tooltip: { backgroundColor: 'rgba(20,20,20,.95)', borderColor: '#444',
              borderWidth: 1, callbacks: { label: (c) => {
                const p = pts[c.datasetIndex === 1 ? pts.length - 1 : c.dataIndex];
                return [`train ${c.parsed.x.toFixed(2)} → test ${c.parsed.y.toFixed(2)}`,
                        p?.isNull ? 'do nothing'
                                  : Object.values(p?.labels || {}).join(' / ')];
              } } },
          },
          scales: {
            x: { title: { display: true, text: 'train Calmar', color: '#888' },
                 ticks: { color: '#888', font: { size: 9 } }, grid: { color: '#222' } },
            y: { title: { display: true, text: 'test Calmar', color: '#888' },
                 ticks: { color: '#888', font: { size: 9 } }, grid: { color: '#222' } },
          },
        },
      });
    },
    toggleGridScatter() {
      this.gridShowScatter = !this.gridShowScatter;
      this.$nextTick(() => this.renderGrid());
    },

    gridCsv() {
      const d = this.gridData, st = this.gridStats;
      if (!d || !st) return;
      const esc = (x) => {
        const s = String(x ?? '');
        return /[",\n]/.test(s) ? '"' + s.replace(/"/g, '""') + '"' : s;
      };
      const fams = (d.sweep || []).map(f => f.family);
      const out = [];
      out.push(['# Factor Trades — parameter response grid']);
      out.push(['# units', d.units_note]);
      out.push(['# sizing', '$' + this.perTrade + '/trade', '$' + this.dailyCap + '/day cap']);
      out.push(['# cutoff', d.cutoff_date]);
      out.push(['# entry_anchor', d.entry_anchor, 'max_strike', d.max_strike ?? '']);
      for (const r of this._csvSelectionRows(this.runData)) out.push(r);
      out.push(['# cells', JSON.stringify(d.cells)]);
      out.push(['# held rules', (d.held || []).join(' | ')]);
      out.push(['# combinations', d.n_combos, 'trades each', d.n_trades]);
      for (const e of (d.errors || [])) out.push(['# FAILED', e.key, e.error]);
      out.push([]);
      const mcols = (d.metrics || []).map(m => m.key);
      out.push([...fams, 'is_null',
                ...mcols.map(k => 'train_' + k), ...mcols.map(k => 'test_' + k)]);
      for (const s of st) {
        if (!s) continue;
        out.push([...fams.map(f => s.combo.labels?.[f] ?? ''),
                  s.combo.is_null ? 'yes' : '',
                  ...mcols.map(k => s.train?.[k] ?? ''),
                  ...mcols.map(k => s.test?.[k] ?? '')]);
      }
      const blob = new Blob([out.map(r => r.map(esc).join(',')).join('\n')],
                            { type: 'text/csv' });
      const a = document.createElement('a');
      a.href = URL.createObjectURL(blob);
      a.download = `factor_trades_grid_${fams.join('_')}.csv`;
      a.click();
      URL.revokeObjectURL(a.href);
    },

    // ── Baseline suite ───────────────────────────────────────────────────
    // One batch: the policy plus three baseline types x N seeds, train and
    // test. Replaces re-sampling by hand and transcribing into a spreadsheet.
    //
    // Deliberately NOT interactive while it runs -- 6N+2 queries is a batch,
    // and pretending otherwise would mean partial tables that look finished.
    suiteN: 10,
    // Collapsed by default and re-opened by a run: the pane is consulted
    // occasionally, but an expanded table pushes the charts -- the things
    // watched while iterating -- down the page on every node change.
    suiteOpen: false,
    suiteData: null,
    suiteRunning: false,
    suiteElapsed: 0,
    _suiteTimer: null,

    async runSuite() {
      if (!this.runData) { this.error = 'run a policy first'; return; }
      if (!this.hasSelection) {
        this.error = (this.isPortfolio ? 'select at least one signal first'
                                       : 'select a zone first')
                   + ' — the suite matches every baseline '
                   + 'to its trade count and date distribution';
        return;
      }
      const src = this.runData;
      this.suiteRunning = true; this.error = ''; this.suiteElapsed = 0;
      // A wall clock rather than a percentage: the server returns one
      // response at the end, so any progress bar would be invented. Elapsed
      // seconds against a stated estimate is the honest version.
      this._suiteTimer = setInterval(() => { this.suiteElapsed += 1; }, 1000);
      try {
        const { data: d, error: err } = await this._postJson(
          '/api/factor-trades/suite', {
            ...this.selectionBody(src),
            entry_anchor: src.entry_anchor, rule_keys: src.rules,
            max_strike: src.max_strike ?? this.maxStrike,
            window: this.window,
            seed: this.baselineSeed, n_draws: +this.suiteN || 10,
          });
        if (err) { this.error = err; return; }
        this.suiteData = d;
        this.suiteOpen = true;
        this.$nextTick(() => this._suiteVerifyAgainstStatBar());
      } catch (e) { this.error = String(e); }
      finally {
        this.suiteRunning = false;
        if (this._suiteTimer) { clearInterval(this._suiteTimer); this._suiteTimer = null; }
      }
    },

    // Higher is better for every metric, INCLUDING max_dd — it is negative,
    // so closer to zero wins. Backwards would invert one column and read as
    // a finding rather than a bug.
    SUITE_HIGHER_IS_BETTER: { avg_ret: true, total_ret: true,
                              avg_annual: true, max_dd: true, calmar: true },

    // Unpack one packed window into the shape _computeDollarSeries wants.
    // Field names are the ones that function reads; nothing is renamed on
    // the way through, because a rename is where the two could diverge.
    _suiteTrades(d, packed) {
      const out = [];
      const T = d.tickers, D = d.dates;
      for (let i = 0; i < packed.r.length; i++) {
        out.push({
          ticker: T[packed.t[i]],
          trade_date: D[packed.d[i]],
          ret: packed.r[i],
          spot_entry_raw: packed.p[i],
        });
      }
      return out;
    },

    // THE POINT OF ALL THIS: dollars come from the SAME function that fills
    // the stat bar, with the SAME rail sizing. Not a port, not a re-derivation
    // — the identical call. A return-unit Calmar would describe a system with
    // unlimited capital; with a $10k daily cap a day firing 40 names cannot
    // take them all, and drawdown accrues against deployed capital.
    _suiteRunStats(d, run, win) {
      const packed = run.trades?.[win];
      const base = { avg_ret: run.avg_ret?.[win] ?? null, n: run.n?.[win] ?? 0 };
      if (!packed || !packed.r.length) {
        return { ...base, total_ret: null, avg_annual: null,
                 max_dd: null, calmar: null };
      }
      const ds = window.FactorCharts._computeDollarSeries(
        this, this._suiteTrades(d, packed), this.perTrade, this.dailyCap);
      const $d = this._dollarStats(ds);
      return { ...base,
               total_ret:  $d?.total_ret_usd ?? null,
               avg_annual: $d?.avg_annual_usd ?? null,
               max_dd:     $d?.max_dd_usd ?? null,
               calmar:     $d?.calmar ?? null };
    },

    _suiteSummary(values) {
      const v = values.filter(x => x != null).map(Number).filter(Number.isFinite);
      if (!v.length) return { n: 0, mean: null, sd: null, min: null, max: null };
      const mean = v.reduce((a, b) => a + b, 0) / v.length;
      // Population sd: these ARE all the draws taken, not a sample of a
      // larger set of draws.
      const sd = v.length > 1
        ? Math.sqrt(v.reduce((a, b) => a + (b - mean) ** 2, 0) / v.length) : 0;
      return { n: v.length, mean, sd, min: Math.min(...v), max: Math.max(...v) };
    },

    // Derived, not stored: sizing lives in the rail, so changing $/trade or
    // the daily cap must re-price a suite already on screen rather than
    // leave it showing figures from the previous sizing.
    get suiteTable() {
      const d = this.suiteData;
      if (!d) return null;
      const metrics = d.metrics || [];
      const policyRow = (d.rows || []).find(r => r.kind === null);
      const polStats = {};
      for (const win of ['train', 'test']) {
        const run = policyRow?.runs?.[0];
        polStats[win] = run ? this._suiteRunStats(d, run, win) : null;
      }
      const rows = (d.rows || []).map(row => {
        const cells = {};
        for (const win of ['train', 'test']) {
          const per = (row.runs || []).map(r => this._suiteRunStats(d, r, win));
          const c = {};
          for (const m of metrics) {
            const vals = per.map(x => x[m.key]);
            if (row.kind === null) {
              c[m.key] = { value: vals[0] ?? null };
              continue;
            }
            const s = this._suiteSummary(vals);
            const pv = polStats[win]?.[m.key];
            const cmp = vals.filter(x => x != null).map(Number);
            if (pv == null || !cmp.length) {
              s.beats = null; s.of = cmp.length;
            } else {
              const hib = this.SUITE_HIGHER_IS_BETTER[m.key] !== false;
              // Counted only over draws that produced a value, and reported
              // with THAT denominator, so a failed draw cannot inflate it.
              s.beats = cmp.filter(x => hib ? pv > x : pv < x).length;
              s.of = cmp.length;
            }
            c[m.key] = s;
          }
          cells[win] = c;
        }
        return { ...row, train: cells.train, test: cells.test };
      });
      return { rows, policy: polStats };
    },

    // The check the port-to-server option could only ever approximate: the
    // suite's policy row and the stat bar are the same function over the
    // same trades, so they must agree to the cent. Logged on both outcomes
    // — a silent pass proves nothing ran.
    _suiteVerifyAgainstStatBar() {
      const t = this.suiteTable, sb = this.dollarStats;
      if (!t || !sb) return;
      const p = t.policy?.[this.window];
      if (!p) return;
      const pairs = [['total_ret', 'total_ret_usd'], ['avg_annual', 'avg_annual_usd'],
                     ['max_dd', 'max_dd_usd']];
      const bad = pairs.filter(([a, b]) =>
        p[a] != null && sb[b] != null && Math.abs(p[a] - sb[b]) > 0.01);
      if (bad.length) {
        console.error('[factor-trades] SUITE POLICY ROW != STAT BAR — same '
          + 'function, same sizing, so this means the two are seeing different '
          + 'trade sets:', bad.map(([a, b]) => `${a} ${p[a]} vs ${sb[b]}`).join(' | '));
      } else {
        console.log('[factor-trades] suite policy row matches the stat bar '
          + `(${this.window}): total ${(p.total_ret ?? 0).toFixed(2)}, `
          + `maxDD ${(p.max_dd ?? 0).toFixed(2)}`);
      }
    },

    suiteEstimate() {
      // 6N+2 queries at ~1.4s, divided by the server's concurrency cap.
      const n = +this.suiteN || 10;
      const q = 3 * n + 1;
      const c = this.suiteData?.concurrency || 5;
      return Math.round(q * 1.4 / c);
    },

    // One formatter, driven by the unit the server declares per metric.
    // Avg Ret is a per-trade percentage; the three portfolio figures are
    // dollars; Calmar is a ratio OF dollars.
    suiteFmt(metric, v) {
      if (v == null) return '—';
      const unit = (this.suiteData?.metrics || []).find(m => m.key === metric)?.unit;
      if (unit === 'ratio') return (+v).toFixed(2);
      if (unit === 'usd') {
        const a = Math.abs(+v), sg = (+v) < 0 ? '-' : '';
        if (a >= 1e6) return sg + '$' + (a / 1e6).toFixed(2) + 'M';
        if (a >= 1e3) return sg + '$' + (a / 1e3).toFixed(1) + 'k';
        return sg + '$' + a.toFixed(0);
      }
      return ((+v) * 100).toFixed(2) + '%';
    },
    // The headline. "beats 0 of 10" answers the question directly where two
    // means ask the reader to hold a distribution in their head.
    suiteBeatsClass(cell) {
      if (!cell || cell.beats == null || !cell.of) return '';
      const f = cell.beats / cell.of;
      // Green only at a clean sweep: "beat 9 of 10" is one draw away from
      // ambiguous and should not read as a result.
      return f >= 1 ? 'sb-win' : (f === 0 ? 'sb-loss' : 'sb-mid');
    },

    // Selection provenance for a CSV. A table of numbers with no record of
    // WHICH trade set produced it is not reusable a week later, and in
    // portfolio mode the metric pair alone no longer identifies that set.
    _csvSelectionRows(src) {
      const out = [];
      const s = src || {};
      if ((s.mode || '') === 'portfolio') {
        out.push(['# selection', 'portfolio', (s.signals || []).length + ' signals']);
        for (const sig of (s.signals || [])) {
          out.push(['# signal', 'bit ' + sig.bit, sig.id, sig.name,
                    sig.primary_metric, sig.secondary_metric,
                    'n_bins ' + sig.n_bins, sig.n_cells + ' cells']);
        }
        out.push(['# dedup', 'one trade per ticker+date across all signals']);
      } else {
        out.push(['# selection', 'zone', s.primary_metric ?? '',
                  s.secondary_metric ?? '(single)']);
        out.push(['# cells', (this.selectedCells || []).length]);
      }
      return out;
    },

    suiteCsv() {
      const d = this.suiteData;
      if (!d) return;
      const esc = (x) => {
        const s = String(x ?? '');
        return /[",\n]/.test(s) ? '"' + s.replace(/"/g, '""') + '"' : s;
      };
      const out = [];
      // Provenance first: a table of sixty numbers with no record of which
      // zone, policy and seeds produced it is not reusable a week later.
      out.push(['# Factor Trades — baseline suite']);
      out.push(['# units', d.units_note]);
      out.push(['# cutoff', d.cutoff_date]);
      out.push(['# entry_anchor', d.entry_anchor]);
      out.push(['# max_strike', d.max_strike ?? '']);
      for (const r of this._csvSelectionRows(this.runData)) out.push(r);
      out.push(['# cells', JSON.stringify(d.cells)]);
      out.push(['# rules', (d.rule_keys || []).join(' | ')]);
      out.push(['# draws', d.n_draws, 'base_seed', d.base_seed]);
      // Dollar figures are meaningless without the sizing that produced
      // them — a suite exported at $2k/$10k is not comparable to one at
      // $5k/$25k, and nothing else in the file would say which it was.
      out.push(['# sizing', '$' + this.perTrade + '/trade',
                '$' + this.dailyCap + '/day cap']);
      out.push(['# seeds', (d.seeds || []).join(' ')]);
      if (d.hold) out.push(['# drawn hold (sessions)', d.hold.avg_sessions]);
      for (const e of (d.errors || [])) out.push(['# FAILED', e.key, e.error]);
      out.push([]);
      out.push(['window', 'run type', 'metric', 'unit', 'draws',
                'policy', 'mean', 'sd', 'min', 'max', 'policy beats', 'of']);
      const t = this.suiteTable;
      for (const win of ['train', 'test']) {
        for (const row of (t.rows || [])) {
          for (const m of (d.metrics || [])) {
            const c = row[win]?.[m.key] || {};
            const pv = t.policy?.[win]?.[m.key];
            const isPol = row.kind === null;
            out.push([win, row.label, m.label, m.unit, row.draws,
                      isPol ? (c.value ?? '') : (pv ?? ''),
                      isPol ? '' : (c.mean ?? ''), isPol ? '' : (c.sd ?? ''),
                      isPol ? '' : (c.min ?? ''), isPol ? '' : (c.max ?? ''),
                      isPol ? '' : (c.beats ?? ''), isPol ? '' : (c.of ?? '')]);
          }
        }
      }
      const blob = new Blob([out.map(r => r.map(esc).join(',')).join('\n')],
                            { type: 'text/csv' });
      const a = document.createElement('a');
      a.href = URL.createObjectURL(blob);
      a.download = `factor_trades_suite_${d.n_draws}x_${d.base_seed}.csv`;
      a.click();
      URL.revokeObjectURL(a.href);
    },

    async runBaseline(resample = false, kind = 'entry') {
      if (!this.runData) { this.error = 'run a policy first'; return; }
      if (!this.hasSelection) {
        this.error = this.isPortfolio
          ? 'select at least one signal first — the baseline matches the '
            + 'portfolio\'s trade count and date distribution, so it needs '
            + 'one to match'
          : 'select a zone first — the baseline matches its trade '
            + 'count and date distribution, so it needs one to match';
        return;
      }
      // Fixed by default so the baseline does NOT move under you while you
      // iterate on exit rules -- otherwise a change in the numbers could be
      // the policy or could be a different draw, and there is no way to tell
      // which. Re-sample is the deliberate opposite: same zone, new draw,
      // which is how you see the baseline's own sampling variance.
      if (resample) this.baselineSeed = Math.floor(Math.random() * 2147483647);
      this.loading = true; this.error = '';
      try {
        const src = this.runData;
        const body = {
          ...this.selectionBody(src),
          entry_anchor: src.entry_anchor, rule_keys: src.rules,
          max_strike: src.max_strike ?? this.maxStrike,
          window: this.window,
          randomize: true, seed: this.baselineSeed, baseline_kind: kind,
        };
        const r = await fetch('/api/factor-trades/zone', {
          method: 'POST', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(body),
        });
        const d = await r.json();
        if (!r.ok || d.error) { this.error = d.error || ('HTTP ' + r.status); return; }
        // A run card built from the policy that produced it, so the strip
        // shows what was actually run. grid is null on purpose: showing the
        // real run's grid beside a random-entry card would claim a zone
        // selection that these trades were not drawn from.
        const card = {
          ...src, grid: null,
          randomize: true, seed: d.baseline?.seed ?? this.baselineSeed,
          baseline_kind: d.baseline?.kind || kind,
          baseline: d.baseline || null,
          window: d.window,
          train: d.train, test: d.test,
          exit_reasons: d.exit_reasons,
          label: 'baseline — ' + (this.BASELINE_KINDS[kind]?.label || kind),
        };
        this.runs.push(card);
        this.currentIdx = this.runs.length - 1;
        this.runData = card;
        this.zoneData = d;
        this.secDetail = d;
        this._refreshGrid();
        this.$nextTick(() => { this._scrollRunsRight(); this.renderCharts(); });
      } catch (e) { this.error = String(e); }
      finally { this.loading = false; }
    },

    async run(random = false) {
      // Kept for the old call signature; the baseline is its own path now.
      if (random) return this.runBaseline(false);
      // PORTFOLIO MODE HAS NO /run. /run exists to build the 20x20 heatmap,
      // and a heatmap is a property of one metric pair -- a portfolio spans
      // several pairs at several resolutions, so there is no single grid to
      // draw. The signal checkboxes ARE the selection, so there is also
      // nothing for a second step to select. /zone alone, and runData is the
      // zone payload: it already carries every field the run card reads.
      if (this.isPortfolio) return this.runPortfolio();
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

    // One request, not two: the zone payload IS the run payload here, minus
    // the grid a portfolio does not have. Pushed onto `runs` by the same code
    // that pushes a signal run, so lock / delete / compare are unchanged.
    async runPortfolio() {
      if (!this.selectedSignalIds.length) {
        this.error = 'select at least one saved signal';
        return;
      }
      this.loading = true; this.error = '';
      try {
        const body = {
          signal_ids: this.selectedSignalIds.slice(),
          entry_anchor: this.entryAnchor,
          rule_keys: this.ruleKeys(),
          max_strike: this.maxStrike,
          window: this.window,
        };
        const r = await fetch('/api/factor-trades/zone', {
          method: 'POST', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(body),
        });
        const d = await r.json();
        if (!r.ok || d.error) { this.error = d.error || ('HTTP ' + r.status); return; }
        // Both, deliberately. runData drives the card and the panels gated on
        // a run existing; zoneData drives the charts. They are the same object
        // here, which is exactly what "a portfolio is just a trade set" means.
        this.runData = d;
        this.zoneData = d;
        this.secDetail = d;
        this.heatmapData = null; this._hmRange = null;
        this.runs.push(d);
        this.currentIdx = this.runs.length - 1;
        this.$nextTick(() => { this._scrollRunsRight(); this.renderCharts(); });
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
      if (!this.lockedRun || !this.hasSelection) return;
      // Fetch the locked run's OWN zone series, using ITS parameters and ITS
      // window -- not the rail's current state. A locked TRAIN run compared
      // against a TEST run would otherwise cross populations with nothing on
      // screen saying so.
      try {
        const r = this.lockedRun;
        const resp = await fetch('/api/factor-trades/zone', {
          method: 'POST', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            ...this.selectionBody(r),
            entry_anchor: r.entry_anchor, rule_keys: r.rules,
            max_strike: r.max_strike,
            window: r.window || 'train',
            // Deterministic seed: locking a baseline refetches the identical
            // sample, so the LOCKED column is the draw that was on screen
            // when it was locked, not a fresh one.
            randomize: !!r.randomize, seed: r.seed ?? null,
            baseline_kind: r.baseline_kind || 'entry',
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
      if (!this.runData || !this.hasSelection) { this.zoneData = null; return; }
      try {
        const body = {
          ...this.selectionBody(this.runData),
          entry_anchor: this.runData.entry_anchor,
          rule_keys: this.runData.rules,
          // Must match the run's population or the zone is a different trade set.
          max_strike: this.runData.max_strike ?? this.maxStrike,
          window: this.window,
          // A baseline run stays a baseline when the zone is refetched --
          // clicking a cell or switching window must not silently turn it
          // back into a real-entry run under the same card.
          randomize: !!this.runData.randomize,
          seed: this.runData.seed ?? null,
          baseline_kind: this.runData.baseline_kind || 'entry',
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
      const SIDE_COLOR = {
        target: 'rgba(52,152,219,0.78)',
        stop:   'rgba(232,67,147,0.72)',
        time:   'rgba(150,150,150,0.60)',
        trend:  'rgba(26,188,156,0.70)',
      };
      // The backstop still overrides its side -- it means the opposite of a
      // rule working, since nothing fired and the horizon caught the trade
      // -- but it does that OUTLINED rather than in a colour of its own.
      //
      // Both values are the time-exit grey, because the backstop IS a time
      // exit: same hue, faint fill, and the theme's --muted for the border,
      // read from the stylesheet rather than hand-picked so a theme change
      // carries here. A light neutral next to saturated blue reads warm by
      // simultaneous contrast, so "looks grey in isolation" is not the test
      // -- being literally the same channel values as the bar beside it is.
      const muted = (getComputedStyle(document.documentElement)
                       .getPropertyValue('--muted') || '').trim() || '#c8c8c8';
      const BACKSTOP = { fill: 'rgba(150,150,150,0.16)', line: muted, w: 2 };
      const fills = rows.map(r => r.is_backstop ? BACKSTOP.fill
        : (SIDE_COLOR[r.side] || 'rgba(150,150,150,0.60)'));
      const lines = rows.map((r, i) => r.is_backstop ? BACKSTOP.line : fills[i]);
      const widths = rows.map(r => r.is_backstop ? BACKSTOP.w : 1);
      this._charts['ft-reasons'] = new Chart(el.getContext('2d'), {
        type: 'bar',
        data: { labels, datasets: [
          { data, backgroundColor: fills, borderColor: lines, borderWidth: widths },
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
          // IDENTICAL paint to the share half. The wash that used to be
          // applied here made the same rule two different greys across the
          // two sub-charts -- max_days 20d read lighter on the left than on
          // the right -- and a colour that means "this rule" cannot also
          // mean "this is the quieter pane". The headers already say which
          // half is which; that was never colour's job.
          backgroundColor: fills, borderColor: lines, borderWidth: widths,
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
      const FC = window.FactorCharts;
      const mx = Math.max(...b.map(x => x.n), 1);
      const paints = b.map(x => FC._barPaint(
        FC._barRgb(x.n ? (x.lo < 0 ? 'neg' : 'pos') : 'none'), (x.n || 0) / mx));
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
      const FC = window.FactorCharts;
      const mx = Math.max(...bins.map(b => b.n), 1);
      const paints = avg.map((v, i) => FC._barPaint(
        FC._barRgb(v == null ? 'none' : (v >= 0 ? 'pos' : 'neg')),
        (bins[i].n || 0) / mx));
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
      // Formatters and the Calmar guard now live in FactorCharts, shared
      // with the portfolio bar. These are thin aliases so the Change row
      // below reads the same as it did -- and, more importantly, so there is
      // exactly ONE definition of Calmar on the project. The guard blanks
      // unless both dollar inputs are present AND the drawdown is negative;
      // a rewrite is how the percent-based fallback came back last time.
      const pct  = v => window.FactorCharts._statPct(v);
      const fmt$ = v => window.FactorCharts._statFmt$(v);
      const calmarRawOf = (s) => window.FactorCharts._statCalmarRaw(s);
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
          // Shared mapper — the same one the portfolio bar uses. Everything
          // page-specific stays below it.
          ...window.FactorCharts.statRowValues(s, $d),
          // p5 / p95 are not on the shared bar but the Change row below
          // still diffs them.
          p5: pct(s.p5),   p5Raw: s.p5 ?? 0,
          p95: pct(s.p95), p95Raw: s.p95 ?? 0,
          avgHold: (s.avg_hold ?? 0).toFixed(2) + ' sess',
          // This page's 16th box. Mean EXIT BAR, which varies here because
          // exits are rule-driven -- the one stat checked against every
          // policy tweak. The portfolio page has no exit rules, so its 16th
          // box is Time in Market instead.
          avgDit: (s.avg_hold ?? 0).toFixed(2) + ' sess',
        };
      };
      // The stat bar is always the TEST window: it is the verdict, and a
      // number that silently switched windows would be the worst kind of
      // wrong. Selection happens on the heatmap, which is train.
      // A baseline against a real run is the comparison this page exists
      // for, so it must NOT warn -- but 'Locked' with no qualifier leaves the
      // most important row on the page unlabelled, and random-entry numbers
      // read exactly like real ones. The window banner covers crossing
      // train/test; nothing covered crossing real/random until here.
      const bMark = (run, name) => {
        if (!run?.randomize) return name;
        const k = this.BASELINE_KINDS[run.baseline_kind || 'entry'];
        return name + ' (' + (k?.row || 'baseline') + ')';
      };
      const editedIsBase = !!this.runData?.randomize;
      const lockedIsBase = !!this.lockedRun?.randomize;
      const edited = mk('edited',
                        bMark(this.runData, this.lockedRun ? 'Edited' : 'Current'),
                        src[this.window], this.dollarStats, this.window);
      if (!this.lockedRun) return [edited].filter(Boolean);
      const lockedSrc = this.lockedZone || this.lockedRun;
      const locked = mk('locked', bMark(this.lockedRun, 'Locked'),
                        lockedSrc[this.window], this.lockedDollarStats, this.window);
      const d = (a, b) => (a ?? 0) - (b ?? 0);
      const E = this.dollarStats, L = this.lockedDollarStats;
      const dd = (k) => (E?.[k] != null && L?.[k] != null) ? (E[k] - L[k]) : null;
      const cE = calmarRawOf(E), cL = calmarRawOf(L);
      const st = src[this.window], lt = lockedSrc[this.window];
      const diff = (locked && edited) ? {
        key: 'change',
        // Exactly one side random makes this row the signal-vs-random delta
        // -- the actual output of the baseline -- rather than a policy delta.
        label: (editedIsBase !== lockedIsBase) ? 'vs baseline' : 'Change',
        window: this.window,
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
      // Attribution columns only when the rows actually carry a mask. A
      // portfolio BASELINE is mode=portfolio with randomly drawn tickers and
      // no mask, so `mode` alone is the wrong thing to branch on.
      const tagged = !!this.zoneData?.has_sig_mask;
      const legend = this.signalLegend(this.zoneData);
      const head = ['ticker', 'trade_date', 'entry_price', 'ret_pct',
                    'exit_bar', 'exit_rule', 'window'];
      if (tagged) {
        // Both the raw mask and one column per signal: the mask is the exact
        // combination's identity (group by it), the named columns are what a
        // spreadsheet pivot actually filters on.
        head.push('sig_mask', 'n_signals', ...legend.map(s => 'sig_' + s.id));
      }
      const rows = t.map(x => {
        const base = [x.ticker, x.trade_date, x.entry_price ?? '',
                      (x.ret * 100).toFixed(6), x.exit_bar, x.exit_rule, x.window];
        if (tagged) {
          const m = x.sig_mask | 0;
          base.push(m, this._popcount(m), ...legend.map(s => (m >> s.bit) & 1));
        }
        return base.join(',');
      });
      const blob = new Blob([[head.join(','), ...rows].join(String.fromCharCode(10))],
                            { type: 'text/csv' });
      const a = document.createElement('a');
      a.href = URL.createObjectURL(blob);
      const stem = this.isPortfolio ? 'portfolio'
                                    : (this.runData?.primary_metric || 'run');
      a.download = `factor_trades_${stem}.csv`;
      a.click(); URL.revokeObjectURL(a.href);
    },

    // How many signals claimed one trade. Bit-twiddling popcount rather than
    // a string count, because it runs once per row on sets of ~25k.
    _popcount(v) {
      v = v - ((v >> 1) & 0x55555555);
      v = (v & 0x33333333) + ((v >> 2) & 0x33333333);
      return (((v + (v >> 4)) & 0x0F0F0F0F) * 0x01010101) >> 24;
    },

  }));
});
