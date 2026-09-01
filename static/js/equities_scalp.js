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
const SC_CHARTS = { geom: null, overlay: null, mult: {} };

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
      const rows = (this.calib && this.calib.rows) || [];
      return q ? rows.filter(r => r.metric.toLowerCase().indexOf(q) >= 0) : rows;
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
      const v = rows.map(r => r.rank_corr).filter(x => x != null);
      if (v.length < 5) return null;
      const lo = Math.min(...v), hi = Math.max(...v);
      const t = rows.map(r => r.top_retention).filter(x => x != null);
      return {
        lo, hi, band: hi - lo,
        // Under 0.1 of spread across every metric there is nothing to pick
        // between them on this axis.
        flat: (hi - lo) < 0.1,
        topLo: t.length ? Math.min(...t) : null,
        topHi: t.length ? Math.max(...t) : null,
        topBand: t.length ? Math.max(...t) - Math.min(...t) : null,
      };
    },

    stabRows() {
      const q = (this.stabFilter || '').toLowerCase();
      const rows = (this.stab && this.stab.rows) || [];
      return q ? rows.filter(r => r.metric.toLowerCase().indexOf(q) >= 0) : rows;
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
    contraGrouped() {
      const c = this.calib;
      return !!(c && (c.contradictions || []).length >= 2
                && c.contradiction_coherence
                && c.contradiction_coherence.coherent);
    },

    contraNote() {
      const c = this.calib;
      if (!c) return '';
      const co = c.contradiction_coherence;
      const n = (c.contradictions || []).length;
      if (!co) {
        return `${n} metric${n === 1 ? '' : 's'} correlate against the `
             + `direction the column claims.`;
      }
      const pct = (co.mean_abs_rho).toFixed(2);
      if (co.coherent) {
        return `${n} metrics correlate against the direction their columns `
             + `claim — and they correlate with EACH OTHER at a mean |ρ| of `
             + `${pct} across ${co.universe} symbols. That is the signature of `
             + `one uncontrolled variable rather than ${n} broken premises: `
             + `they are largely the same measurement, so they were always `
             + `going to agree. Which variable is not answerable from this `
             + `data.`;
      }
      return `${n} metrics correlate against the direction their columns claim, `
           + `and they are only weakly related to each other (mean |ρ| ${pct} `
           + `across ${co.universe} symbols). So these are ${n} separate `
           + `things to explain rather than one.`;
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
