/* Equities Scalp — page component.
 *
 * PHASE 1: the scaffold. What is real here is the metric catalog and the
 * filter bar, both driven entirely by /api/equities-scalp/meta. Everything
 * else on the page is a labelled stub.
 *
 * NO METRIC NAMES IN THIS FILE. The metric set is unsettled by design and a
 * calibration exercise exists to delete most of it; a name written here would
 * become a column of nulls rather than an error the day the pipeline renamed
 * it. Every metric this page knows about arrives from the server, which read
 * it from the database. scripts/check_scalp_metrics.py fails the build on a
 * metric-shaped literal in this file.
 */

const SC_BLUE = '#3498db';
const SC_PINK = '#e84393';
const SC_CHARTS = { geom: null, overlay: null, mult: {}, profile: null };

/* Constant-ratio lines for the spread-against-noise scatter.
 *
 * On LOG-LOG axes a constant spread/noise ratio is a straight line of slope
 * 1, which is the whole reason the panel uses log scales: it turns "good" from
 * a number you look up per point into a DIRECTION you read off the picture.
 * On linear axes those lines are rays through the origin that crowd together
 * near it, and the geometry the panel exists to show is unreadable. */
const scRatioLines = {
  id: 'scRatioLines',
  beforeDatasetsDraw(chart, args, opts) {
    const {ctx, scales: {x, y}} = chart;
    const lines = (opts && opts.ratios) || [];
    ctx.save();
    for (const r of lines) {
      ctx.beginPath();
      ctx.strokeStyle = 'rgba(255,255,255,0.10)';
      ctx.setLineDash([3, 3]);
      ctx.lineWidth = 1;
      // y = r * x, sampled at the axis ends.
      const x0 = x.min, x1 = x.max;
      ctx.moveTo(x.getPixelForValue(x0), y.getPixelForValue(x0 * r));
      ctx.lineTo(x.getPixelForValue(x1), y.getPixelForValue(x1 * r));
      ctx.stroke();
      const py = y.getPixelForValue(x1 * r);
      if (py > y.top + 6 && py < y.bottom - 2) {
        ctx.setLineDash([]);
        ctx.fillStyle = 'rgba(255,255,255,0.28)';
        ctx.font = '9px sans-serif';
        ctx.textAlign = 'right';
        ctx.fillText('ratio ' + r, x.getPixelForValue(x1) - 3, py - 3);
      }
    }
    ctx.restore();
  },
};

/* The hours actually traded, shaded behind the session profile.
 *
 * The panel's whole point is that the good window may not be the window being
 * used — two sessions ended at 13:30 and 14:34 while both spread and noise
 * recover into the close. Drawn BEHIND the lines so it reads as ground rather
 * than as another series, and as one band per session so a day that stopped
 * early is visible as itself rather than averaged away. */
const scTradedHours = {
  id: 'scTradedHours',
  beforeDatasetsDraw(chart, args, opts) {
    const bands = (opts && opts.bands) || [];
    if (!bands.length) return;
    const {ctx, scales: {x, y}} = chart;
    ctx.save();
    for (const b of bands) {
      if (b.from == null || b.to == null) continue;
      const x0 = x.getPixelForValue(b.from);
      const x1 = x.getPixelForValue(b.to);
      ctx.fillStyle = 'rgba(78,201,160,0.10)';
      ctx.fillRect(x0, y.top, Math.max(1, x1 - x0), y.bottom - y.top);
    }
    ctx.restore();
  },
};

async function scGetJson(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${r.status} ${r.statusText}`);
  return r.json();
}

document.addEventListener('alpine:init', () => {
  Alpine.data('equitiesScalp', () => ({

    loading: true,
    // Shaped so every x-show on the page can be evaluated before the first
    // fetch returns. An undefined .metrics would throw inside a template loop
    // and take the whole component down before it could report why.
    meta: { connected: false, error: '', dates: [], metrics: [], date: null,
            symbols: 0, undocumented: [] },
    filters: {},
    filterKeys: [],
    ranges: {},
    // null, not 0: "not counted yet" and "nothing passes" are different
    // answers and the second one is a finding.
    passCount: null,

    // ── fills and calibration ───────────────────────────────────────────
    fills: null,
    uploading: false,
    uploadResult: null,
    uploadError: '',
    calib: null,
    calibLoading: false,
    calibTarget: 'dollars_per_min',
    calibFilter: '',

    /* ── sorting, shared by every research table ─────────────────────────
     *
     * Only the ranked candidates table sorted, and the tables that most need
     * it are the ones that do not: rank stability's own banner points at
     * top-20 retention as the column to read, and there was no way to order
     * by it.
     *
     * One mechanism rather than three, because three would diverge — one
     * table would keep nulls-last and the others would lose it, and a metric
     * with no measurement would sort into the position the best one belongs
     * in.
     *
     * DEFAULTS ARE THE PANEL'S OWN CONCLUSION. Rank stability opens on
     * top-20 held, because its banner says rank_corr is flat and the range
     * lives in the other column — opening on rank_corr would contradict the
     * sentence directly above the table. */
    sorts: {
      stab:  { key: 'top_retention', desc: true },
      calib: { key: 'abs_rho',       desc: true },
      fills: { key: 'trade_date',    desc: true },
    },
    corrOrder: 'cluster',

    setSortOn(table, key) {
      const s = this.sorts[table];
      if (!s) return;
      if (s.key === key) s.desc = !s.desc;
      else { s.key = key; s.desc = true; }
    },

    sortMark(table, key) {
      const s = this.sorts[table];
      if (!s || s.key !== key) return '';
      return s.desc ? '▾' : '▴';
    },

    /* `abs_x` sorts by magnitude, and a dotted key reaches into a nested
     * object — worst_jump.places is the only thing on these tables that is
     * not a top-level field, and special-casing it would be the start of
     * three different accessors. */
    sortVal(row, key) {
      if (key.startsWith('abs_')) {
        const v = this.sortVal(row, key.slice(4));
        return v == null ? null : Math.abs(v);
      }
      return key.split('.').reduce((o, k) => (o == null ? null : o[k]), row);
    },

    sortRows(rows, table) {
      const s = this.sorts[table];
      if (!s) return rows;
      const sign = s.desc ? -1 : 1;
      return rows.slice().sort((a, b) => {
        const x = this.sortVal(a, s.key), y = this.sortVal(b, s.key);
        // NULLS LAST IN BOTH DIRECTIONS. A missing measurement is not a small
        // value, and letting it float to the top of an ascending sort puts
        // the names with no data where the best ones belong.
        if (x == null || y == null) {
          return (x == null && y == null) ? 0 : (x == null ? 1 : -1);
        }
        if (typeof x === 'string' || typeof y === 'string') {
          return sign * String(x).localeCompare(String(y));
        }
        return sign * (x < y ? -1 : x > y ? 1 : 0);
      });
    },

    // ── 2.1 geometry / 2.6 over time ────────────────────────────────────
    /* Passing names only, BY DEFAULT.
     *
     * Unfiltered, the sparse-quote names crush everything into one corner:
     * their noise collapses toward zero, the log x-axis stretches down to
     * 1e-10 to hold them, and the structure the panel exists to show is a
     * dot. Those names are exactly what the filters exclude, so drawing them
     * by default spends the whole plot on the rows already ruled out.
     *
     * The unfiltered view answers "what did the filters exclude", which is a
     * deliberate question and not the opening one. */
    geomOnly: true,
    ser: null,
    serLoading: false,
    serSymbols: '',

    // ── 2.2 / 2.3, neither of which needs a single fill ─────────────────
    corr: null,
    corrLoading: false,
    corrFamily: '',
    stab: null,
    stabLoading: false,
    stabFilter: '',

    // ── P3 ticker detail ────────────────────────────────────────────────
    tdSymbol: '',
    td: null,
    tdLoading: false,
    tdError: '',
    tdScope: 'sessions',
    cmp: null,
    cmpSymbols: '',
    cmpFilter: '',

    // ── data health ─────────────────────────────────────────────────────
    health: null,
    healthLoading: false,

    // ── the ranked table ────────────────────────────────────────────────
    cand: null,
    candLoading: false,
    candError: '',
    // The selected noise metric. Everything about the variant -- which
    // midpoint, which horizon, which statistic -- is carried IN THE NAME,
    // so one string is the whole selection and there is no way for the
    // three parts to disagree with each other.
    noise: '',
    sortKey: '',
    sortDesc: true,
    // Role keys the user has turned off, and raw metric names they have
    // added. Kept as the DIFFERENCE from the default set rather than as an
    // absolute list, so a role added to scalp_columns.py later appears
    // without the stored preference having to know about it.
    hiddenRoles: [],
    extraCols: [],
    chooserOpen: false,
    chooserFilter: '',
    // Failing rows come back so a threshold can be judged against what it
    // excludes, but they are off by default -- at 9am the question is what
    // to trade, not whether the filter is right.
    showFails: false,

    async init() {
      try {
        const j = await scGetJson('/api/equities-scalp/meta');
        this.meta = Object.assign(this.meta, j);
        const f = j.filters || {};
        this.filterKeys = f.keys || [];
        this.ranges = f.ranges || {};
        // A COPY of the defaults. Binding the sliders straight to the served
        // object would let a drag mutate what the page believes the pipeline's
        // defaults are, and there would then be nothing to reset to.
        this.filters = Object.assign({}, f.defaults || {});
        this.noise = j.default_noise || '';
        if (this.meta.connected) {
          await Promise.all([
            this.meta.date ? this.loadCandidates() : Promise.resolve(),
            this.meta.date ? this.loadHealth() : Promise.resolve(),
            this.loadFills(), this.loadCalibration(),
            this.loadCorrelation(), this.loadStability(),
          ]);
        }
      } catch (e) {
        this.meta.connected = false;
        this.meta.error = String(e.message || e);
      } finally {
        this.loading = false;
      }
    },

    range(k) {
      return this.ranges[k] || { min: 0, max: 1, step: 0.01 };
    },

    /* Slider labels, derived from the config key rather than listed.
     * The filter set comes from the pipeline's DEFAULT_FILTERS, so a
     * threshold added upstream appears here with a readable name instead of
     * needing a matching edit in this file. */
    filterLabel(k) {
      return k.replace(/^min_/, 'min ')
              .replace(/^max_/, 'max ')
              .replace(/_cents$/, ' ¢')
              .replace(/_bps$/, ' bps')
              .replace(/_per_min$/, '/min')
              .replace(/_/g, ' ');
    },

    fmtFilter(k, v) {
      if (v == null) return '—';
      // Coverage is a share, not a rate; everything else reads better at one
      // decimal than at the step's own precision.
      if (k.indexOf('coverage') >= 0) return (v * 100).toFixed(0) + '%';
      const step = this.range(k).step || 0.1;
      return step >= 1 ? String(Math.round(v)) : Number(v).toFixed(1);
    },

    onFilterChange() { this.loadCandidates(); },

    // ── the ranked table ────────────────────────────────────────────────

    async loadHealth() {
      if (!this.meta.connected) return;
      this.healthLoading = true;
      try {
        const j = await scGetJson('/api/equities-scalp/health?sessions=10');
        this.health = j.error ? null : j;
      } catch (e) {
        this.health = null;
      } finally {
        this.healthLoading = false;
      }
    },

    healthCols() { return (this.health && this.health.watched) || []; },

    /* A change against trailing, as a signed percentage. The SIGN is kept —
     * arrivals collapsing and arrivals doubling are different problems, and
     * an absolute value would render them identically. */
    fmtChange(ch) {
      if (ch == null) return '—';
      const p = ch * 100;
      return (p >= 0 ? '+' : '') + p.toFixed(0) + '%';
    },

    /* Amber below the flag threshold, red at or above it. Two levels rather
     * than one because a 20% move is worth a look and a 37% move is the
     * incident this panel was built for. */
    changeClass(ch) {
      if (ch == null) return '';
      const thr = (this.health && this.health.thresholds.move) || 0.25;
      const a = Math.abs(ch);
      if (a >= thr) return 'bad';
      if (a >= thr * 0.6) return 'warn';
      return '';
    },

    /* The row's own summary. A red cell says a number moved; this says what
     * that means, which is what stops the panel needing to be interpreted
     * from scratch every morning. */
    healthNote(r) {
      const parts = [];
      if (r.flags.indexOf('coverage') >= 0) {
        parts.push(`${r.n_symbols} of ${r.universe_n} symbols — a compute run `
                 + `that did not finish looks exactly like this`);
      } else if (r.missing_n) {
        parts.push(`${r.missing_n} symbol${r.missing_n === 1 ? '' : 's'} absent `
                 + `(${(r.missing_sample || []).slice(0, 6).join(' ')}${r.missing_n > 6 ? ' …' : ''}) `
                 + `— refused at fetch, no data, or not computed; the pipeline `
                 + `does not record which`);
      }
      if (r.flags.indexOf('arrivals') >= 0) {
        parts.push('arrivals moved against their own history — this is the '
                 + 'shape a venue-incomplete fetch took last time');
      }
      if (r.flags.indexOf('n_metrics') >= 0) {
        parts.push(`${r.n_metrics} distinct metrics, short of the trailing `
                 + `count — a metric family stopped being written`);
      }
      return parts.join('. ');
    },

    // ── P3 ticker detail ────────────────────────────────────────────────

    async loadTickerDetail() {
      const sym = (this.tdSymbol || '').trim().toUpperCase();
      if (!sym) { this.td = null; return; }
      this.tdLoading = true; this.tdError = '';
      try {
        const q = new URLSearchParams({ symbol: sym, scope: this.tdScope });
        if (this.meta.date) q.set('date', this.meta.date);
        const j = await scGetJson('/api/equities-scalp/ticker-detail?' + q);
        if (j.error) { this.tdError = j.error; this.td = null; }
        else { this.td = j; this.renderProfile(); }
      } catch (e) {
        this.tdError = String(e.message || e); this.td = null;
      } finally { this.tdLoading = false; }
    },

    pickTicker(sym) {
      this.tdSymbol = sym;
      this.loadTickerDetail();
      const el = document.getElementById('detail');
      if (el) el.scrollIntoView({ behavior: 'smooth', block: 'start' });
    },

    /* Minutes since midnight, so a clock time can be a linear x. A category
     * axis would space 09:30 and 15:45 evenly with everything between and
     * make a gap in the data look like continuous trading. */
    tdMins(t) {
      const p = String(t || '').split(':');
      return p.length < 2 ? null : (+p[0]) * 60 + (+p[1]);
    },

    tdBands() {
      const h = (this.td && this.td.traded_hours) || [];
      return h.map(x => ({ from: this.tdMins(x.first), to: this.tdMins(x.last) }));
    },

    renderProfile() {
      if (typeof Chart === 'undefined' || !this.td) return;
      if (SC_CHARTS.profile) { SC_CHARTS.profile.destroy(); SC_CHARTS.profile = null; }
      this.$nextTick(() => {
        const el = document.getElementById('sc-profile');
        const rows = (this.td.profile || []);
        if (!el || !rows.length) return;
        const self = this;
        const pt = k => rows.map(r => ({ x: self.tdMins(r.t), y: r[k] }))
                            .filter(p => p.x != null && p.y != null);
        const sets = [];
        if (this.td.columns.spread_bps)
          sets.push({ label: 'spread bps', data: pt('spread_bps'),
                      borderColor: SC_BLUE, yAxisID: 'y' });
        if (this.td.columns.noise)
          sets.push({ label: 'noise bps', data: pt('noise'),
                      borderColor: SC_PINK, yAxisID: 'y' });
        if (this.td.columns.ratio)
          sets.push({ label: 'ratio', data: pt('ratio'),
                      borderColor: '#4ec9a0', yAxisID: 'y2', borderDash: [4, 3] });
        SC_CHARTS.profile = new Chart(el.getContext('2d'), {
          type: 'line',
          data: { datasets: sets.map(s => Object.assign({
            borderWidth: 1.4, tension: 0, pointRadius: 0, parsing: false,
            spanGaps: true, fill: false }, s)) },
          options: {
            responsive: true, maintainAspectRatio: false, animation: false,
            interaction: { mode: 'nearest', axis: 'x', intersect: false },
            scales: {
              x: { type: 'linear', min: 9 * 60 + 30, max: 16 * 60,
                   ticks: { stepSize: 60, font: { size: 9 },
                            callback: v => String(Math.floor(v / 60)).padStart(2, '0')
                                           + ':' + String(v % 60).padStart(2, '0') },
                   grid: { color: 'rgba(255,255,255,0.05)' } },
              // SPREAD AND NOISE SHARE AN AXIS. Where they converge there is
              // nothing to capture, and separate axes would rescale them into
              // looking parallel. The ratio gets its own because it is a
              // different quantity, not a third bps series.
              y: { position: 'left', grid: { color: 'rgba(255,255,255,0.05)' },
                   ticks: { font: { size: 9 } },
                   title: { display: true, text: 'bps', color: '#777',
                            font: { size: 9 } } },
              y2: { position: 'right', grid: { display: false },
                    ticks: { font: { size: 9 }, color: '#4ec9a0' },
                    title: { display: true, text: 'ratio', color: '#4ec9a0',
                             font: { size: 9 } } },
            },
            plugins: {
              legend: { position: 'bottom',
                        labels: { boxWidth: 9, font: { size: 9 },
                                  usePointStyle: true, pointStyle: 'line' } },
              scTradedHours: { bands: this.tdBands() },
              tooltip: { callbacks: { title: it => {
                const v = it[0].parsed.x;
                return String(Math.floor(v / 60)).padStart(2, '0') + ':'
                     + String(v % 60).padStart(2, '0');
              } } },
            },
          },
          plugins: [scTradedHours],
        });
      });
    },

    /* The repeatability heatmap's colour. Scaled to the matrix's OWN range
     * rather than an absolute one: the question is whether the good window
     * lands in the same place, and a fixed scale would render a name whose
     * ratios are all small as a uniform blank. */
    tdRange() {
      const v = [];
      for (const row of ((this.td && this.td.repeat) || [])) {
        for (const c of row.cells) if (c != null) v.push(c);
      }
      if (!v.length) return null;
      v.sort((a, b) => a - b);
      // 5th to 95th, so one extreme session does not flatten every other cell.
      return { lo: v[Math.floor(v.length * 0.05)],
               hi: v[Math.floor(v.length * 0.95)] };
    },

    tdCell(v) {
      if (v == null) return 'background:repeating-linear-gradient(45deg,'
                          + 'transparent,transparent 3px,rgba(255,255,255,.05) 3px,'
                          + 'rgba(255,255,255,.05) 6px)';
      const r = this.tdRange();
      if (!r || r.hi <= r.lo) return '';
      const t = Math.max(0, Math.min(1, (v - r.lo) / (r.hi - r.lo)));
      return `background:rgba(78,201,160,${(0.08 + t * 0.82).toFixed(3)})`;
    },

    /* Whether the good window actually repeats, as a number.
     *
     * Per session, which bucket held the highest ratio; then how tightly those
     * cluster. A consistent shape is a name to plan a morning around; a best
     * hour that wanders is not, however good the average looks. */
    tdRepeatNote() {
      const rows = (this.td && this.td.repeat) || [];
      const buckets = (this.td && this.td.buckets) || [];
      const best = [];
      for (const row of rows) {
        let bi = null, bv = null;
        row.cells.forEach((c, i) => { if (c != null && (bv == null || c > bv)) { bv = c; bi = i; } });
        if (bi != null) best.push(bi);
      }
      if (best.length < 3) return '';
      const mean = best.reduce((a, b) => a + b, 0) / best.length;
      const sd = Math.sqrt(best.reduce((a, b) => a + (b - mean) ** 2, 0) / best.length);
      const at = buckets[Math.round(mean)] || '';
      const spread = sd * 15;   // buckets are 15 minutes
      return `Across ${best.length} sessions the best bucket averages `
           + `${at.slice(0, 5)}, with a spread of ±${spread.toFixed(0)} minutes. `
           + (spread <= 45
              ? 'That is a window to plan a morning around.'
              : 'That wanders too much to plan around — the average profile '
                + 'above is hiding a window that is not in the same place '
                + 'twice.');
    },

    async loadCompare() {
      const syms = (this.cmpSymbols || '').trim();
      if (!syms) { this.cmp = null; return; }
      try {
        const q = new URLSearchParams({ symbols: syms });
        if (this.meta.date) q.set('date', this.meta.date);
        const j = await scGetJson('/api/equities-scalp/compare?' + q);
        this.cmp = j.error ? null : j;
      } catch (e) { this.cmp = null; }
    },

    cmpRows() {
      const q = (this.cmpFilter || '').toLowerCase();
      const rows = (this.cmp && this.cmp.metrics) || [];
      return q ? rows.filter(r => r.metric.toLowerCase().indexOf(q) >= 0) : rows;
    },

    /* Provenance as a share of the tape, which is the reading. "0.94" beside
     * an item name is a number; "94% of the tape retained" is a fact about
     * whether the row above it can be trusted. */
    provRows() {
      const p = (this.td && this.td.provenance) || [];
      return p.map(r => ({
        item: r.item, value: r.value,
        pct: /share|rate/.test(r.item) && r.value != null && r.value <= 1
             ? (r.value * 100).toFixed(1) + '%' : null,
      }));
    },

    // ── fills ───────────────────────────────────────────────────────────

    async loadFills() {
      try {
        const j = await scGetJson('/api/equities-scalp/fills');
        this.fills = j.error ? null : j;
      } catch (e) { this.fills = null; }
    },

    async uploadFills(ev) {
      const f = ev.target.files && ev.target.files[0];
      if (!f) return;
      this.uploading = true;
      this.uploadError = '';
      this.uploadResult = null;
      try {
        const body = new FormData();
        body.append('file', f);
        const r = await fetch('/api/equities-scalp/upload-fills',
                              { method: 'POST', body });
        const j = await r.json();
        if (!r.ok) {
          // FastAPI puts a 400's message in `detail`. Surfacing the raw
          // status instead would hide the one sentence that says which
          // column the parser could not find.
          this.uploadError = j.detail || `${r.status} ${r.statusText}`;
        } else {
          this.uploadResult = j;
          await Promise.all([this.loadFills(), this.loadCalibration()]);
        }
      } catch (e) {
        this.uploadError = String(e.message || e);
      } finally {
        this.uploading = false;
        // Cleared so re-selecting the SAME file fires change again — a
        // re-upload after fixing a statement is the common case, and an
        // input that silently ignores it looks like a failed upload.
        ev.target.value = '';
      }
    },

    /* The parser's report, as things that need doing rather than counts.
     * An unclosed position is first because it is the one that has silently
     * corrupted a day's statistics before. */
    uploadIssues() {
      const j = this.uploadResult;
      if (!j || !j.report) return [];
      const r = j.report, out = [];
      if (r.n_unclosed) {
        out.push({ level: 'bad', text:
          `${r.n_unclosed} position${r.n_unclosed === 1 ? '' : 's'} still open `
          + `at the end of a session (`
          + r.unclosed.map(u => `${u.symbol} ${u.shares} from ${u.since.slice(11)}`).join(', ')
          + `). Excluded from the trips above rather than completed at the `
          + `last price — but every statistic for that ticker-day is now `
          + `computed from an incomplete picture.` });
      }
      if (r.n_unparsed) {
        out.push({ level: 'bad', text:
          `${r.n_unparsed} trade row${r.n_unparsed === 1 ? '' : 's'} could not `
          + `be read: ` + r.rows_unparsed.slice(0, 4)
              .map(u => `line ${u.line} "${u.text}"`).join('; ') });
      }
      if (r.reversals && r.reversals.length) {
        out.push({ level: 'warn', text:
          `${r.reversals.length} execution(s) carried a position through zero `
          + `and were split. That should not happen in a one-position-at-a-time `
          + `strategy.` });
      }
      if (j.archive_error) {
        out.push({ level: 'warn', text:
          `The statement was parsed and stored, but the raw file could not be `
          + `archived (${j.archive_error}). A parser fix would need the `
          + `statement exported again.` });
      }
      return out;
    },

    // ── calibration ─────────────────────────────────────────────────────

    async loadCalibration() {
      this.calibLoading = true;
      try {
        const j = await scGetJson('/api/equities-scalp/calibration?target='
                                  + encodeURIComponent(this.calibTarget));
        this.calib = j.error ? null : j;
      } catch (e) {
        this.calib = null;
      } finally {
        this.calibLoading = false;
      }
    },

    calibRows() {
      const q = (this.calibFilter || '').toLowerCase();
      let rows = (this.calib && this.calib.rows) || [];
      if (q) rows = rows.filter(r => r.metric.toLowerCase().indexOf(q) >= 0);
      return this.sortRows(rows, 'calib');
    },

    fillsRows() {
      return this.sortRows((this.fills && this.fills.rows) || [], 'fills');
    },

    /* How many sessions before this table means anything.
     *
     * Solved rather than asserted: the expected-by-chance count at |rho| >= 0.5
     * falls below one when the sample reaches about 25 ticker-days. Saying
     * "watch it over weeks" is advice; saying how many more is a number. */
    calibNeeded() {
      const c = this.calib;
      if (!c || !c.n_pairs) return null;
      const have = c.n_pairs;
      // erfc(0.5 * sqrt(n-1) / sqrt(2)) * n_metrics < 1
      for (let n = have; n <= 400; n++) {
        const z = 0.5 * Math.sqrt(n - 1);
        // Abramowitz-Stegun 7.1.26 is plenty for a guidance number.
        const t = 1 / (1 + 0.3275911 * Math.abs(z) / Math.SQRT2);
        const erf = 1 - (((((1.061405429 * t - 1.453152027) * t)
                    + 1.421413741) * t - 0.284496736) * t + 0.254829592)
                    * t * Math.exp(-(z * z) / 2);
        const p = 1 - erf;
        if (p * c.n_metrics < 1) return { need: n, have };
      }
      return null;
    },

    // ── 2.2 metric correlation ──────────────────────────────────────────

    async loadCorrelation() {
      this.corrLoading = true;
      try {
        const q = new URLSearchParams();
        if (this.meta.date) q.set('date', this.meta.date);
        if (this.corrFamily) q.set('family', this.corrFamily);
        const j = await scGetJson('/api/equities-scalp/metric-correlation?' + q);
        this.corr = j.error ? null : j;
      } catch (e) { this.corr = null; } finally { this.corrLoading = false; }
    },

    /* One cell of the matrix. |rho| drives opacity and the sign drives the
     * hue, so a block of near-duplicates reads as a solid square rather than
     * as a number to be compared. */
    /* The matrix is ordered, not sorted.
     *
     * Its default leaf ordering comes from hierarchical clustering and is the
     * reason the blocks are adjacent — sorting the rows alphabetically would
     * destroy the one thing the picture shows. So this offers an ordering
     * CHOICE instead, and says what each one costs:
     *
     *   cluster     families adjacent; the blocks are readable
     *   redundancy  most-duplicated first; answers "what can I delete"
     *   name        alphabetical; find a specific metric, blocks scattered
     */
    corrIndices() {
      const c = this.corr;
      if (!c || !c.matrix.length) return [];
      const n = c.metrics.length;
      const idx = Array.from({ length: n }, (_, i) => i);
      if (this.corrOrder === 'name') {
        return idx.sort((a, b) => c.metrics[a].localeCompare(c.metrics[b]));
      }
      if (this.corrOrder === 'redundancy') {
        const mean = idx.map(i => {
          let s = 0;
          for (let j = 0; j < n; j++) if (j !== i) s += Math.abs(c.matrix[i][j]);
          return n > 1 ? s / (n - 1) : 0;
        });
        return idx.sort((a, b) => mean[b] - mean[a]);
      }
      return idx;
    },

    /* How duplicated one metric is, for the ordering above and the tooltip. */
    corrMean(i) {
      const c = this.corr;
      if (!c) return 0;
      const n = c.metrics.length;
      let s = 0;
      for (let j = 0; j < n; j++) if (j !== i) s += Math.abs(c.matrix[i][j]);
      return n > 1 ? s / (n - 1) : 0;
    },

    corrCell(v) {
      const a = Math.min(1, Math.abs(v));
      const c = v >= 0 ? '52,152,219' : '232,67,147';
      return `background:rgba(${c},${(a * a).toFixed(3)})`;
    },

    /* How much of the metric set is one metric wearing several names. The
     * headline of this panel: 75 noise columns collapsing to a handful is
     * the finding, not the matrix. */
    corrSaving() {
      const g = (this.corr && this.corr.redundant) || [];
      if (!g.length) return null;
      const inGroups = g.reduce((a, x) => a + x.length, 0);
      return { groups: g.length, members: inGroups,
               removable: inGroups - g.length,
               of: (this.corr.metrics || []).length };
    },

    // ── 2.3 rank stability ──────────────────────────────────────────────

    async loadStability() {
      this.stabLoading = true;
      try {
        const j = await scGetJson('/api/equities-scalp/rank-stability?sessions=10');
        this.stab = j.error ? null : j;
      } catch (e) { this.stab = null; } finally { this.stabLoading = false; }
    },

    /* WHAT A HIGH SCORE HERE DOES NOT MEAN.
     *
     * Measured on the real universe, day-to-day rank correlation came back at
     * 0.95-0.99 for nearly every metric — reference_price 0.999, spreads
     * 0.97-0.99, every noise family at 0.957, with no separation between
     * median and rms. That is the NULL for a universe this size and this
     * stable, not a finding, and reading it as one would turn a panel that
     * separates nothing into a second line of evidence it cannot provide.
     *
     * Computed from the data rather than asserted: if the whole column sits
     * in a narrow band, the panel says so itself. */
    stabSpread() {
      const rows = (this.stab && this.stab.rows) || [];
      const v = rows.map(r => r.rank_corr).filter(x => x != null)
                    .sort((a, b) => a - b);
      if (v.length < 5) return null;
      const q = p => v[Math.min(v.length - 1, Math.floor(p * v.length))];
      const t = rows.map(r => r.top_retention).filter(x => x != null)
                    .sort((a, b) => a - b);
      const tq = p => t.length ? t[Math.min(t.length - 1, Math.floor(p * t.length))] : null;
      return {
        n: v.length,
        lo: v[0], hi: v[v.length - 1],
        // MEDIAN AND IQR, not min and max. The first version tested
        // (max - min) < 0.1 and the banner never appeared: across 232
        // metrics one outlier is enough to widen the range past any
        // threshold, so a min/max test asks "is every metric alike"
        // when the question is "are nearly all of them alike".
        p25: q(0.25), median: q(0.5), p75: q(0.75),
        iqr: q(0.75) - q(0.25),
        // The plainest statement of the same thing, and the one the banner
        // leads with.
        shareHigh: v.filter(x => x >= 0.9).length / v.length,
        topLo: t.length ? t[0] : null,
        topHi: t.length ? t[t.length - 1] : null,
        topP25: tq(0.25), topP75: tq(0.75),
        topIqr: t.length ? tq(0.75) - tq(0.25) : null,
      };
    },

    /* Whether this panel is separating anything today.
     *
     * Nearly all metrics above 0.9 with a narrow interquartile range means
     * the column is the NULL for a universe this size, not a finding — and
     * that has to be said whether it is true or not, because a column of
     * 0.97s left to be inferred from reads as evidence it is not. */
    stabIsNull() {
      const s = this.stabSpread();
      return !!(s && s.shareHigh >= 0.8 && s.iqr < 0.1);
    },

    stabRows() {
      const q = (this.stabFilter || '').toLowerCase();
      let rows = (this.stab && this.stab.rows) || [];
      if (q) rows = rows.filter(r => r.metric.toLowerCase().indexOf(q) >= 0);
      return this.sortRows(rows, 'stab');
    },

    /* Does one statistic hold its head better than another?
     *
     * The comparison the banner makes possible once the table can be ordered
     * by top-20 retention. Grouped by the STATISTIC suffix, because that is
     * what calibration said separates and rank correlation was too flat to
     * confirm or deny — median against rms is the specific question, and a
     * median of medians is the honest summary at this spread. */
    stabByStat() {
      const rows = (this.stab && this.stab.rows) || [];
      const by = {};
      for (const r of rows) {
        const v = r.variant;
        const stat = (v && v.statistic) || null;
        if (!stat || r.top_retention == null) continue;
        (by[stat] = by[stat] || []).push(r.top_retention);
      }
      const out = Object.keys(by).map(stat => {
        const v = by[stat].slice().sort((a, b) => a - b);
        return { stat, n: v.length,
                 median: v[Math.floor(v.length / 2)],
                 lo: v[0], hi: v[v.length - 1] };
      });
      return out.sort((a, b) => b.median - a.median);
    },

    /* Below 0.5 a metric is re-drawing its ranking every morning. That is
     * disqualifying whatever it calibrates to — a signal that cannot be acted
     * on the next day is not one. */
    stabClass(v) {
      if (v == null) return '';
      if (v >= 0.8) return 'strong';
      if (v >= 0.5) return 'mid';
      return 'bad';
    },

    /* The named worst mover, in words. A low average is a number; "AGX moved
     * 431 places of 587" is the thing that gets looked into. */
    jumpNote(r) {
      const j = r.worst_jump;
      if (!j) return '';
      return `${j.symbol} moved ${j.places} places of ${j.of} between two `
           + `sessions (${j.from.toFixed(3)} → ${j.to.toFixed(3)})`;
    },

    // ── 2.1 spread against noise ────────────────────────────────────────

    /* Built from the ranked table's own payload rather than a second fetch.
     * The scatter and the table are the same rows seen two ways, and a
     * separate query could disagree with the table sitting above it. */
    geomPoints() {
      const c = this.cand;
      if (!c) return [];
      const nx = 'noise', ny = 'spread_bps';
      if (!c.columns.some(x => x.key === nx) ||
          !c.columns.some(x => x.key === ny)) return [];
      return c.rows
        .filter(r => r.values[nx] > 0 && r.values[ny] > 0)
        .filter(r => !this.geomOnly || r.passes)
        .map(r => ({ x: r.values[nx], y: r.values[ny], symbol: r.symbol,
                     passes: r.passes, traded: r.traded }));
    },

    renderGeometry() {
      if (typeof Chart === 'undefined') return;
      if (SC_CHARTS.geom) { SC_CHARTS.geom.destroy(); SC_CHARTS.geom = null; }
      this.$nextTick(() => {
        const el = document.getElementById('sc-geom');
        const pts = this.geomPoints();
        if (!el || !pts.length) return;
        const self = this;

        // Traded names are drawn LAST and larger, so they sit on top of the
        // cloud. The point of the panel is reading an unknown candidate
        // against a known outcome, which needs the known ones findable.
        const untraded = pts.filter(p => !p.traded);
        const traded = pts.filter(p => p.traded);

        SC_CHARTS.geom = new Chart(el.getContext('2d'), {
          type: 'scatter',
          data: { datasets: [
            { label: 'not traded', data: untraded, parsing: false,
              pointRadius: 2.5, pointBorderWidth: 0,
              pointBackgroundColor: untraded.map(p =>
                p.passes ? 'rgba(52,152,219,0.45)' : 'rgba(138,138,138,0.22)') },
            { label: 'traded', data: traded, parsing: false,
              pointRadius: 6, pointBorderWidth: 1.5,
              pointBorderColor: '#1b1b1b',
              // Green made money, pink lost it. Not a ramp: at this sample
              // size the SIGN is the reading and a gradient would imply a
              // precision the data does not have.
              pointBackgroundColor: traded.map(p =>
                (p.traded.pnl_per_min == null) ? '#8a8a8a'
                  : (p.traded.pnl_per_min >= 0 ? '#4ec9a0' : SC_PINK)) },
          ] },
          options: {
            responsive: true, maintainAspectRatio: false, animation: false,
            scales: {
              x: { type: 'logarithmic', title: { display: true, text: 'noise bps',
                   color: '#777', font: { size: 10 } },
                   grid: { color: 'rgba(255,255,255,0.05)' },
                   ticks: { font: { size: 9 } } },
              y: { type: 'logarithmic', title: { display: true, text: 'spread bps',
                   color: '#777', font: { size: 10 } },
                   grid: { color: 'rgba(255,255,255,0.05)' },
                   ticks: { font: { size: 9 } } },
            },
            plugins: {
              legend: { display: false },
              scRatioLines: { ratios: [1, 2, 4, 8] },
              tooltip: { callbacks: { label: it => {
                const p = it.raw;
                const base = `${p.symbol}  spread ${p.y.toFixed(1)} / noise `
                           + `${p.x.toFixed(2)} = ratio ${(p.y / p.x).toFixed(2)}`;
                if (!p.traded) return base;
                const t = p.traded;
                return [base, `traded ${t.days} session${t.days === 1 ? '' : 's'}, `
                            + `${t.trips} trips, `
                            + `${t.pnl_per_min == null ? '—'
                                : (t.pnl_per_min >= 0 ? '+' : '')
                                  + t.pnl_per_min.toFixed(2)} $/min`];
              } } },
            },
          },
          plugins: [scRatioLines],
        });
      });
    },

    geomNote() {
      const c = this.cand;
      if (!c) return '';
      const n = c.n_traded || 0;
      if (!n) return 'No traded names yet — upload a statement and the known '
                   + 'outcomes appear as larger points.';
      return `${n} name${n === 1 ? '' : 's'} with realised results, drawn `
           + `larger: green made money, pink lost it. An unknown candidate `
           + `sitting in a green neighbourhood is the reading this panel `
           + `exists for.`;
    },

    // ── 2.6 over time ───────────────────────────────────────────────────

    async loadSeries() {
      if (!this.cand) return;
      const ratio = (this.cand.columns.find(c => c.key === 'ratio') || {}).metric;
      const spread = (this.cand.columns.find(c => c.key === 'spread_bps') || {}).metric;
      const noise = (this.cand.columns.find(c => c.key === 'noise') || {}).metric;
      if (!ratio) return;
      this.serLoading = true;
      try {
        const q = new URLSearchParams({
          metrics: [ratio, spread, noise].filter(Boolean).join(','),
          sessions: '30',
        });
        if (this.serSymbols.trim()) q.set('symbols', this.serSymbols.trim());
        const j = await scGetJson('/api/equities-scalp/series?' + q);
        this.ser = j.error ? null : j;
        if (this.ser) { this.renderMultiples(); this.renderOverlay(); }
      } catch (e) { this.ser = null; } finally { this.serLoading = false; }
    },

    serRatio()  { return (this.ser && this.ser.metrics[0]) || ''; },
    serSpread() { return (this.ser && this.ser.metrics[1]) || ''; },
    serNoise()  { return (this.ser && this.ser.metrics[2]) || ''; },

    /* Six panels on IDENTICAL axes. Six panels on their own scales is six
     * shapes and no comparison, which is the one thing this is for. */
    renderMultiples() {
      if (typeof Chart === 'undefined' || !this.ser) return;
      Object.values(SC_CHARTS.mult).forEach(c => c && c.destroy());
      SC_CHARTS.mult = {};
      this.$nextTick(() => {
        const j = this.ser, m = this.serRatio();
        const b = (j.bounds || {})[m] || {};
        for (const sym of j.symbols) {
          const el = document.getElementById('sc-mult-' + sym);
          const data = ((j.series[sym] || {})[m]) || [];
          if (!el || !data.length) continue;
          SC_CHARTS.mult[sym] = new Chart(el.getContext('2d'), {
            type: 'line',
            data: { labels: j.dates, datasets: [{
              data, borderColor: SC_BLUE, borderWidth: 1.3, tension: 0,
              pointRadius: 0, spanGaps: true,
              fill: { target: 'origin', above: 'rgba(52,152,219,0.10)' },
            }] },
            options: {
              responsive: true, maintainAspectRatio: false, animation: false,
              scales: {
                x: { display: false },
                y: { min: b.min, max: b.max,
                     grid: { color: 'rgba(255,255,255,0.05)' },
                     ticks: { font: { size: 8 }, maxTicksLimit: 3 } },
              },
              plugins: {
                legend: { display: false },
                tooltip: { callbacks: {
                  title: it => j.dates[it[0].dataIndex],
                  label: it => it.parsed.y == null ? '' : it.parsed.y.toFixed(2),
                } },
              },
            },
          });
        }
      });
    },

    /* Spread and noise on ONE axis, for one name. Where the lines converge
     * there is nothing to capture however wide the spread looks in cents —
     * which is invisible when the two are plotted apart. */
    renderOverlay() {
      if (typeof Chart === 'undefined' || !this.ser) return;
      if (SC_CHARTS.overlay) { SC_CHARTS.overlay.destroy(); SC_CHARTS.overlay = null; }
      this.$nextTick(() => {
        const el = document.getElementById('sc-overlay');
        const j = this.ser, sym = j.symbols[0];
        if (!el || !sym || !j.series[sym]) return;
        const sp = j.series[sym][this.serSpread()] || [];
        const no = j.series[sym][this.serNoise()] || [];
        if (!sp.length && !no.length) return;
        SC_CHARTS.overlay = new Chart(el.getContext('2d'), {
          type: 'line',
          data: { labels: j.dates, datasets: [
            { label: 'spread bps', data: sp, borderColor: SC_BLUE,
              borderWidth: 1.4, tension: 0, pointRadius: 0, spanGaps: true },
            { label: 'noise bps', data: no, borderColor: SC_PINK,
              borderWidth: 1.4, tension: 0, pointRadius: 0, spanGaps: true },
          ] },
          options: {
            responsive: true, maintainAspectRatio: false, animation: false,
            interaction: { mode: 'index', intersect: false },
            scales: {
              x: { grid: { display: false },
                   ticks: { maxTicksLimit: 6, maxRotation: 0, font: { size: 8 } } },
              // ONE axis, deliberately. Two would rescale them into looking
              // parallel and destroy the only thing the panel shows.
              y: { grid: { color: 'rgba(255,255,255,0.05)' },
                   ticks: { font: { size: 9 } } },
            },
            plugins: {
              legend: { position: 'bottom',
                        labels: { boxWidth: 9, font: { size: 9 },
                                  usePointStyle: true, pointStyle: 'line' } },
            },
          },
        });
      });
    },

    /* One callout or several.
     *
     * Four contradictions is a different object from one. Four independently
     * broken premises would be remarkable; four metrics that co-vary across
     * the universe and point the same way is ONE uncontrolled variable, and
     * listing them separately reads as the first when the evidence says the
     * second. The server measures which it is — mean |rho| among the
     * contradicting metrics across the whole universe, which needs no fills
     * and so can be said at a sample size where nothing else can. */
    contraCoh() { return (this.calib && this.calib.contradiction_coherence) || null; },

    /* Grouped ONLY when the evidence says so — but the evidence is reported
     * either way. The first version rendered the coherence figure inside the
     * grouped callout, so a mean below the threshold hid the very number that
     * decides the question. Reporting gated behind the conclusion is the same
     * defect as a guard that reports healthy about the wrong thing. */
    contraGrouped() {
      const c = this.contraCoh();
      return !!(c && c.status === 'coherent');
    },

    contraShow() {
      return !!(this.calib && (this.calib.contradictions || []).length >= 2);
    },

    contraHeading() {
      const c = this.contraCoh();
      if (!c) return 'Metrics correlating against their column\'s direction';
      if (c.status === 'coherent')
        return 'These point the same way because they are the same thing';
      if (c.status === 'separate')
        return 'These point the same way and are NOT the same thing';
      if (c.status === 'error')
        return 'Could not decide whether these are one thing or several';
      return 'Metrics correlating against their column\'s direction';
    },

    contraNote() {
      const cal = this.calib;
      if (!cal) return '';
      const c = this.contraCoh();
      const n = (cal.contradictions || []).length;
      if (!c || c.status === 'not_applicable') {
        return `${n} metric${n === 1 ? '' : 's'} correlate against the `
             + `direction the column claims.`;
      }
      if (c.status === 'error') {
        return `The coherence calculation failed — ${c.reason}. Without it `
             + `there is no way to tell one uncontrolled variable from `
             + `${n} broken premises, so they are listed separately below. `
             + `This is a bug, not a finding.`;
      }
      if (c.status === 'unavailable') {
        return c.reason;
      }
      const p = c.mean_abs_rho.toFixed(2);
      if (c.status === 'coherent') {
        return `${n} metrics correlate against the direction their columns `
             + `claim — and they correlate with EACH OTHER at a mean |ρ| of `
             + `${p} across ${c.universe} symbols, above the ${c.threshold} `
             + `line. That is the signature of one uncontrolled variable `
             + `rather than ${n} broken premises: they are largely the same `
             + `measurement, so they were always going to agree. Which `
             + `variable is not answerable from this data.`;
      }
      return `${n} metrics correlate against the direction their columns claim, `
           + `and their mean |ρ| with each other is ${p} across `
           + `${c.universe} symbols — below the ${c.threshold} line. So on `
           + `this evidence they are ${n} separate things to explain, not one `
           + `confound. Watch the number as sessions accumulate: it is a `
           + `property of the metrics, so it moves only if the universe does.`;
    },

    rhoClass(rho) {
      const a = Math.abs(rho);
      if (a >= 0.7) return 'strong';
      if (a >= 0.5) return 'mid';
      return '';
    },

    async loadCandidates() {
      if (!this.meta.connected || !this.meta.date) return;
      this.candLoading = true; this.candError = '';
      try {
        const q = new URLSearchParams({ date: this.meta.date });
        if (this.noise) q.set('noise', this.noise);
        const cols = this.activeRoleKeys();
        if (cols.length) q.set('columns', cols.join(','));
        if (this.extraCols.length) q.set('extra', this.extraCols.join(','));
        if (this.sortKey) q.set('sort', this.sortKey);
        q.set('desc', String(this.sortDesc));
        // Only thresholds the user has actually moved. Sending the whole set
        // every time would freeze the pipeline's defaults into the URL, so a
        // default changed upstream would stop reaching the page.
        for (const k of this.filterKeys) {
          const v = this.filters[k];
          const dflt = (this.meta.filters && this.meta.filters.defaults || {})[k];
          if (v != null && v !== dflt) q.set(k, String(v));
        }
        const j = await scGetJson('/api/equities-scalp/candidates?' + q);
        if (j.error) { this.candError = j.error; this.cand = null; }
        else {
          this.cand = j;
          this.noise = j.noise || this.noise;
          this.passCount = j.n_pass;
          // Same rows, two more views. Drawn from this payload rather than
          // re-fetched, so the scatter cannot disagree with the table.
          this.renderGeometry();
          this.loadSeries();
        }
      } catch (e) {
        this.candError = String(e.message || e); this.cand = null;
      } finally {
        this.candLoading = false;
      }
    },

    /* The role keys currently shown. Derived from the server's own role list
     * minus what the user hid, so this never has to name one. */
    activeRoleKeys() {
      const all = (this.cand && this.cand.roles || []).map(r => r.key);
      if (!all.length) return [];
      return all.filter(k => this.hiddenRoles.indexOf(k) < 0);
    },

    toggleRole(k) {
      const i = this.hiddenRoles.indexOf(k);
      if (i < 0) this.hiddenRoles.push(k); else this.hiddenRoles.splice(i, 1);
      this.loadCandidates();
    },

    toggleExtra(metric) {
      const i = this.extraCols.indexOf(metric);
      if (i < 0) this.extraCols.push(metric); else this.extraCols.splice(i, 1);
      this.loadCandidates();
    },

    /* The column chooser lists the WHOLE catalog, not the leftovers. A metric
     * already on screen shows as checked rather than being filtered out of
     * the list -- otherwise turning one off means hunting for where it went. */
    chooserRows() {
      const q = (this.chooserFilter || '').toLowerCase();
      const roleMetric = {};
      for (const c of (this.cand && this.cand.columns || [])) {
        if (c.role) roleMetric[c.metric] = c.key;
      }
      return (this.meta.metrics || [])
        .filter(m => !q || m.metric.toLowerCase().indexOf(q) >= 0)
        .map(m => ({
          metric: m.metric,
          section: m.section,
          tooltip: m.tooltip,
          href: m.href,
          // A metric can be on screen for either reason, and they are undone
          // differently -- one is a role, the other an extra column.
          asRole: roleMetric[m.metric] || null,
          asExtra: this.extraCols.indexOf(m.metric) >= 0,
        }));
    },

    setSort(key) {
      if (this.sortKey === key) this.sortDesc = !this.sortDesc;
      else { this.sortKey = key; this.sortDesc = true; }
      this.loadCandidates();
    },

    /* Column headers. The role's label where there is one, the metric's own
     * name where the column came from the chooser -- a raw metric has no
     * friendlier name and inventing one would hide which metric it is. */
    colLabel(c) {
      const r = (this.cand && this.cand.roles || []).find(x => x.key === c.key);
      return r ? r.label : c.key;
    },

    colMeta(c) {
      const r = (this.cand && this.cand.roles || []).find(x => x.key === c.key);
      const m = (this.meta.metrics || []).find(x => x.metric === c.metric);
      return {
        note: r ? r.note : (m && m.tooltip) || '',
        href: m && m.href,
        metric: c.metric,
        units: r ? r.units : null,
      };
    },

    /* Units decide the format, not the magnitude. A share printed as "0.99"
     * and a ratio printed as "0.99" are different readings, and guessing from
     * the number would render both the same way. */
    fmtVal(c, v) {
      if (v == null) return '—';
      const u = this.colMeta(c).units;
      if (u === 'share') return (v * 100).toFixed(0) + '%';
      if (u === 'price') return v.toFixed(2);
      if (u === 'cents') return v.toFixed(1);
      if (u === 'ratio') return v.toFixed(2);
      if (u === 'count') return Math.round(v).toLocaleString();
      if (Math.abs(v) >= 1000) return Math.round(v).toLocaleString();
      return v.toFixed(2);
    },

    /* Which threshold a row failed, in words, for the row's title. A struck
     * row that does not say why is just a struck row. */
    failNote(row) {
      if (!row.fails || !row.fails.length) return '';
      return 'Excluded by ' + row.fails.map(f => {
        const t = (this.cand && this.cand.thresholds || {})[f];
        return `${this.filterLabel(f)} (${this.fmtFilter(f, t)})`;
      }).join(', ');
    },

    visibleRows() {
      const rows = (this.cand && this.cand.rows) || [];
      return this.showFails ? rows : rows.filter(r => r.passes);
    },

    /* The 10-day ratio sparkline, as an SVG path. STABILITY, not level: the
     * line is scaled to its own range, so what reads is whether the name
     * holds still, which the ratio column beside it cannot say. */
    sparkPath(row, w = 54, h = 14) {
      const v = row.spark;
      if (!v) return '';
      const pts = v.map((y, i) => [i, y]).filter(p => p[1] != null);
      if (pts.length < 2) return '';
      const ys = pts.map(p => p[1]);
      const lo = Math.min(...ys), hi = Math.max(...ys);
      const span = (hi - lo) || 1;
      const n = v.length - 1 || 1;
      return pts.map((p, k) => {
        const x = (p[0] / n) * w;
        const y = h - ((p[1] - lo) / span) * h;
        return `${k ? 'L' : 'M'}${x.toFixed(1)},${y.toFixed(1)}`;
      }).join(' ');
    },

    /* How far the ratio swings across the window, relative to its own level.
     * A name reading 0.069 one day and 1.727 the next is measuring the
     * measurement, and that is invisible in a single day's column. */
    sparkSwing(row) {
      const v = (row.spark || []).filter(x => x != null);
      if (v.length < 2) return null;
      const lo = Math.min(...v), hi = Math.max(...v);
      if (!lo) return null;
      return hi / lo;
    },
  }));
});

// The sentinel the page checks. LAST LINE, deliberately: a syntax error
// anywhere above stops execution before this runs, so the banner fires for a
// broken bundle as well as an absent one.
window.__scalpLoaded = true;
