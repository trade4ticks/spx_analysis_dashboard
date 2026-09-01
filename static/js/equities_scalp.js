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
        if (this.meta.connected && this.meta.date) {
          await Promise.all([this.loadCandidates(), this.loadHealth()]);
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
