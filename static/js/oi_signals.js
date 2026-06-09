'use strict';

// Stage 2: ticker-centric firing view backed by GET /api/factor-signals/firing.
// Renders verified-correct payload numbers exactly as-is — Stage 1 tied stats
// to ground truth in pgAdmin, so the bug surface here is rendering fidelity,
// not stat math. Tracked-signals add lives here; untrack moves to the roster
// in Stage 3. Gantt helpers below are preserved verbatim from the pre-rebuild
// JS so Stage 3 can rewire the calendar without re-deriving the Gantt math.

document.addEventListener('alpine:init', () => {
  Alpine.data('oiSignals', () => ({

    // ── Firing engine state ─────────────────────────────────────────────
    firingData: { as_of: null, n_tracked: 0, n_firing_tickers: 0, tickers: [] },
    firingDate: '',           // optional ISO date; '' means "use server MAX(trade_date)"
    firingLoading: false,
    expandedTickers: {},       // ticker -> bool

    // ── Tracked-signals add controls ────────────────────────────────────
    availSignals:    [],       // GET /api/factor-analysis/signals
    portfolios:      [],       // GET /api/factor-analysis/portfolios
    addSignalId:     '',
    addPortfolioId:  '',
    toast:           '',
    _toastTimer:     null,

    // ── Calendar (kept inert for Stage 3 rewire) ────────────────────────
    calEntries:  [],
    _ganttRange: { start: new Date(), end: new Date(), totalDays: 60 },

    // ─────────────────────────────────────────────────────────────────────

    async init() {
      // Three independent fetches in parallel: signals list + portfolios list
      // + firing payload. Tracked signals are reflected in the firing payload
      // (n_tracked + tickers), so no separate /tracked call here.
      this._updateGanttRange();
      await Promise.all([
        this._loadSignals(),
        this._loadPortfolios(),
        this.loadFiring(),
      ]);
    },

    async _loadSignals() {
      try {
        const r = await fetch('/api/factor-analysis/signals');
        if (r.ok) {
          const j = await r.json();
          this.availSignals = (j.signals || []).slice()
            .sort((a, b) => (a.name || '').localeCompare(b.name || ''));
        }
      } catch (_) {}
    },

    async _loadPortfolios() {
      try {
        const r = await fetch('/api/factor-analysis/portfolios');
        if (r.ok) {
          this.portfolios = (await r.json()).slice()
            .sort((a, b) => (a.name || '').localeCompare(b.name || ''));
        }
      } catch (_) {}
    },

    // ── Firing payload ──────────────────────────────────────────────────

    async loadFiring() {
      this.firingLoading = true;
      try {
        const qs = this.firingDate ? `?date=${encodeURIComponent(this.firingDate)}` : '';
        const r = await fetch(`/api/factor-signals/firing${qs}`);
        if (r.ok) {
          const data = await r.json();
          // Defensive defaults so the template can render before the first
          // real payload arrives without throwing on undefined access.
          this.firingData = {
            as_of:            data.as_of            ?? null,
            n_tracked:        data.n_tracked        ?? 0,
            n_firing_tickers: data.n_firing_tickers ?? 0,
            tickers:          data.tickers          ?? [],
          };
          // After the first load, anchor the date picker visibly to the as_of
          // the server resolved. Only sync if the picker is still empty — the
          // user may have an explicit selection we shouldn't clobber.
          if (!this.firingDate && this.firingData.as_of) {
            this.firingDate = this.firingData.as_of;
          }
        }
      } catch (_) {}
      this.firingLoading = false;
    },

    hasFirings() {
      return (this.firingData.tickers || []).length > 0;
    },

    // ── Ticker row expand/collapse ──────────────────────────────────────

    toggleTicker(t) {
      this.expandedTickers = {
        ...this.expandedTickers,
        [t]: !this.expandedTickers[t],
      };
    },
    expandAll() {
      const next = {};
      for (const r of (this.firingData.tickers || [])) next[r.ticker] = true;
      this.expandedTickers = next;
    },
    collapseAll() {
      this.expandedTickers = {};
    },

    // ── Tracked-signals add paths ───────────────────────────────────────

    async addTrackedSignal() {
      const sid = parseInt(this.addSignalId);
      if (!sid) return;
      try {
        const r = await fetch('/api/factor-signals/tracked', {
          method:  'POST',
          headers: { 'Content-Type': 'application/json' },
          body:    JSON.stringify({ signal_id: sid }),
        });
        const sig = this.availSignals.find(s => s.id === sid);
        const label = sig ? sig.name : `signal ${sid}`;
        if (r.ok) {
          this._showToast(`Added '${label}'`);
          await this.loadFiring();
        } else {
          const detail = (await r.json().catch(() => ({}))).detail || 'add failed';
          this._showToast(`Add failed: ${detail}`, true);
        }
      } catch (_) {
        this._showToast('Add failed — network error', true);
      }
      this.addSignalId = '';
    },

    async addTrackedFromPortfolio() {
      const pid = parseInt(this.addPortfolioId);
      if (!pid) return;
      try {
        const r = await fetch('/api/factor-signals/tracked/from-portfolio', {
          method:  'POST',
          headers: { 'Content-Type': 'application/json' },
          body:    JSON.stringify({ portfolio_id: pid }),
        });
        if (r.ok) {
          const { added, already_present, total_in_portfolio } = await r.json();
          this._showToast(
            `Added ${added} of ${total_in_portfolio} signals` +
            (already_present ? ` (${already_present} already tracked)` : '')
          );
          await this.loadFiring();
        } else {
          const detail = (await r.json().catch(() => ({}))).detail || 'add failed';
          this._showToast(`Add failed: ${detail}`, true);
        }
      } catch (_) {
        this._showToast('Add failed — network error', true);
      }
      this.addPortfolioId = '';
    },

    _showToast(msg, isError) {
      this.toast = msg;
      if (this._toastTimer) clearTimeout(this._toastTimer);
      this._toastTimer = setTimeout(() => { this.toast = ''; }, isError ? 5500 : 4000);
    },

    // ── Dropdown labels ─────────────────────────────────────────────────

    signalLabel(s) {
      const cells = (s.cell_set || []).length;
      return `${s.name} · ${s.primary_metric} × ${s.secondary_metric} · ${s.outcome} · ${cells} cells @ ${s.n_bins}`;
    },

    // ── Formatters / class helpers (render layer only) ───────────────────

    pct(v) {
      // Stat blocks in /firing already carry return-space floats (e.g. 0.045
      // = 4.5%). Multiply by 100 for percent display. Show '—' for null.
      if (v === null || v === undefined) return '—';
      const x = +v;
      if (!isFinite(x)) return '—';
      return (x * 100).toFixed(2) + '%';
    },
    pctOf1(v) {
      // win_rate already on the 0..1 scale; same formula but force the same
      // 1-decimal style as the rest of the page.
      if (v === null || v === undefined) return '—';
      const x = +v;
      if (!isFinite(x)) return '—';
      return (x * 100).toFixed(1) + '%';
    },
    cvOrDash(v) {
      // null is the explicit 'near-zero mean — undefined' sentinel.
      if (v === null || v === undefined) return '—';
      const x = +v;
      if (!isFinite(x)) return '—';
      return x.toFixed(2);
    },
    posneg(v) {
      if (v === null || v === undefined) return '';
      const x = +v;
      if (!isFinite(x) || x === 0) return '';
      return x > 0 ? 'pos' : 'neg';
    },
    smallN(n) {
      // Visual dim cue for samples under 30. NOT a gate — the value still
      // renders, just in the muted color, so the user notices the small n
      // without the page hiding numbers.
      return (typeof n === 'number' && n < 30) ? 'small-n' : '';
    },

    // ── Gantt helpers (kept from pre-rebuild for Stage 3 rewire) ────────

    async removeFromCalendar(id) {
      // Stage 3 will POST a real signal-keyed delete; for now this is a
      // local-only remove so the inert UI doesn't error if a stub row
      // is injected during testing.
      this.calEntries = this.calEntries.filter(e => e.id !== id);
      this._updateGanttRange();
    },

    _updateGanttRange() {
      const today = new Date();
      today.setHours(0, 0, 0, 0);
      let start = new Date(today);
      start.setDate(start.getDate() - 14);
      let end = new Date(today);
      end.setDate(end.getDate() + 45);

      for (const e of this.calEntries) {
        if (e.entry_date) {
          const d = new Date(e.entry_date + 'T00:00:00');
          if (d < start) { start = new Date(d); start.setDate(start.getDate() - 3); }
        }
        if (e.exit_date) {
          const d = new Date(e.exit_date + 'T00:00:00');
          if (d > end) { end = new Date(d); end.setDate(end.getDate() + 3); }
        }
      }

      const totalDays = Math.round((end - start) / 86400000) + 1;
      this._ganttRange = { start, end, totalDays };
    },

    barStyle(entry) {
      const { start, totalDays } = this._ganttRange;
      const s = new Date((entry.entry_date || '') + 'T00:00:00');
      const e = new Date((entry.exit_date  || entry.entry_date || '') + 'T00:00:00');
      const startOff = Math.max(0, (s - start) / 86400000);
      const endOff   = Math.min(totalDays, (e - start) / 86400000 + 1);
      const leftPct  = (startOff / totalDays) * 100;
      const widthPct = Math.max(0.8, (endOff - startOff) / totalDays * 100);
      const col = entry.color || '#3498db';
      return `left:${leftPct.toFixed(2)}%;width:${widthPct.toFixed(2)}%;background:${col}`;
    },

    get ganttTodayStyle() {
      const { start, totalDays } = this._ganttRange;
      const today = new Date();
      today.setHours(0, 0, 0, 0);
      const off = (today - start) / 86400000;
      const pct = (off / totalDays) * 100;
      if (pct < 0 || pct > 100) return 'display:none';
      return `left:${pct.toFixed(2)}%`;
    },

    get ganttHeaderTicks() {
      const { start, totalDays } = this._ganttRange;
      const ticks = [];
      const d = new Date(start);
      const dow = d.getDay();
      if (dow !== 0) d.setDate(d.getDate() + (7 - dow));
      const rangeEnd = new Date(start.getTime() + totalDays * 86400000);
      while (d <= rangeEnd) {
        const off = (d - start) / 86400000;
        const pct = (off / totalDays) * 100;
        ticks.push({
          pct:   pct.toFixed(2),
          label: d.toLocaleDateString('en-US', { month: 'short', day: 'numeric' }),
        });
        d.setDate(d.getDate() + 7);
      }
      return ticks;
    },

  }));
});
