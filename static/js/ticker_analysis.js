/* ============================================================================
 * Ticker Analysis page — single-ticker view.
 *
 * Additive module (see ticker_analysis_build_brief.md). Nothing here touches
 * the Factor Analysis page or its state.
 *
 * Phase 1 (this file): control-bar scaffold. Populates the ticker universe,
 * wires the horizon selector (fixed list per §3.1), the Confluence/Union
 * shade toggle, and Clear/Save stubs. The metric panes, price chart, dynamic
 * stat strip, saved layouts, and chain views arrive in later phases.
 * ==========================================================================*/

/* Forward-return horizons offered in the control bar (brief §3.1).
 * `_oc` = open→close of the forward window; `_cc` = close→close. Bins are
 * fixed at 20 on this page, so there is no bin-count control. */
const TA_HORIZONS = (() => {
  const days = [1, 3, 5, 7, 10, 20];
  const out = [];
  for (const d of days) out.push({ value: `ret_${d}d_fwd_oc`, label: `${d}D fwd (o→c)` });
  for (const d of days) out.push({ value: `ret_${d}d_fwd_cc`, label: `${d}D fwd (c→c)` });
  return out;
})();

document.addEventListener('alpine:init', () => {
  Alpine.data('tickerAnalysis', () => ({
    // ── Control-bar state ────────────────────────────────────────────────
    tickers: [],
    ticker: '',
    horizons: TA_HORIZONS,
    horizon: 'ret_5d_fwd_oc',      // default matches Factor Analysis default
    shadeMode: 'confluence',       // 'confluence' (default) | 'union'  (§5.1)

    // Selected bins across all metric panes. Keyed later by paneId → Set of
    // bin indices; Phase 1 keeps it empty so the count pill reads 0.
    selectedBins: {},

    // ── Lifecycle ────────────────────────────────────────────────────────
    async init() {
      await this.loadTickers();
    },

    async loadTickers() {
      try {
        const r = await fetch('/api/ticker-analysis/tickers');
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        this.tickers = await r.json();
        if (this.tickers.length && !this.ticker) {
          this.ticker = this.tickers[0];   // default to first (§1)
        }
      } catch (e) {
        console.error('[ticker-analysis] loadTickers failed:', e);
        this.tickers = [];
      }
    },

    // ── Derived ──────────────────────────────────────────────────────────
    get selectedBinCount() {
      let n = 0;
      for (const k in this.selectedBins) {
        const s = this.selectedBins[k];
        n += s ? (s.size ?? s.length ?? 0) : 0;
      }
      return n;
    },

    // ── Control-bar handlers (Phase 2 wires these to data loads) ─────────
    onTickerChange() {
      // Selections are bin-index based, so they carry across tickers; the
      // panes just re-query for the new ticker in Phase 2.
    },

    onHorizonChange() {
      // Recompute trigger for the stat strip / bars lands in Phase 2.
    },

    clearSelection() {
      this.selectedBins = {};
    },

    saveLayout() {
      // Saved layouts (Postgres-backed) arrive in Phase 3.
      alert('Save layout — coming in Phase 3.');
    },
  }));
});
