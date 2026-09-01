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

    onFilterChange() {
      // Phase 2 re-queries the candidate list here. Deliberately left
      // unimplemented rather than stubbed with a fake number: a count that
      // moves when a slider moves would read as a measurement.
      this.passCount = null;
    },
  }));
});
