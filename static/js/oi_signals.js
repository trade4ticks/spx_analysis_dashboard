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
    // Firing payload is now keyed by (ticker, outcome) row — a ticker
    // firing on multiple outcome horizons (5d AND 20d) produces multiple
    // rows. avg_ret is only comparable WITHIN one horizon, so pooling
    // across horizons produces nonsense. Each row is single-outcome by
    // construction; every downstream stat is comparable.
    //
    // all_signals = every tracked signal, ordered by signal_id ASC.
    // Drives the fixed-position columns of the firing-grid pane below
    // the table. Columns do NOT shift with the firing set day-to-day —
    // muscle memory: "column 5 is signal #29".
    firingData: { as_of: null, n_tracked: 0, n_firing_rows: 0,
                  n_firing_tickers: 0, rows: [], all_signals: [] },

    // Firing-grid pane state. gridStatMode flips every cell, every
    // row-end, every column-foot together — never mixed on screen.
    // gridMinN fades cells whose n (current mode) is below the slider
    // so dark shades on thin samples don't read as edge.
    gridStatMode: 'ticker',   // 'ticker' | 'all'
    gridMinN:     30,
    firingDate: '',           // optional ISO date; '' means "use server MAX(trade_date)"
    firingLoading: false,
    // Expansion is keyed by `${ticker}|${outcome}` since a ticker can
    // appear on more than one row now.
    expandedRows: {},

    // ── Sort state ──────────────────────────────────────────────────────
    // Default: most-confirmed tickers at the top (#sigs desc). Click a
    // header to switch column; click the same header again to flip the
    // direction. Sort only reorders the head rows — each tbody owns its
    // own detail row, so expansions ride along through every reorder.
    sortKey: 'n_signals_firing',
    sortDir: 'desc',           // 'asc' | 'desc'

    // ── Tracked-signals add controls ────────────────────────────────────
    availSignals:    [],       // GET /api/factor-analysis/signals
    portfolios:      [],       // GET /api/factor-analysis/portfolios
    addSignalId:     '',
    addPortfolioId:  '',
    toast:           '',
    _toastTimer:     null,

    // ── Calendar (wired in Stage 3) ─────────────────────────────────────
    calEntries:  [],
    calLoading:  false,
    _ganttRange: { start: new Date(), end: new Date(), totalDays: 60 },

    // ── Roster (watchlist-with-performance, every tracked signal) ───────
    roster:           [],
    rosterLoading:    false,
    rosterCollapsed:  true,         // collapsed by default — firing view is primary
    rosterSortKey:    'tracked_at',
    rosterSortDir:    'desc',       // newest first

    // ─────────────────────────────────────────────────────────────────────

    async init() {
      // Five independent fetches in parallel. Tracked-signals state is
      // reflected in the firing + roster payloads (n_tracked, tickers,
      // roster items), so no separate /tracked call here.
      this._updateGanttRange();
      await Promise.all([
        this._loadSignals(),
        this._loadPortfolios(),
        this.loadFiring(),
        this.loadCalendar(),
        this.loadRoster(),
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
            n_firing_rows:    data.n_firing_rows    ?? 0,
            n_firing_tickers: data.n_firing_tickers ?? 0,
            rows:             data.rows             ?? [],
            all_signals:      data.all_signals      ?? [],
          };
          if (!this.firingDate && this.firingData.as_of) {
            this.firingDate = this.firingData.as_of;
          }
        }
      } catch (_) {}
      this.firingLoading = false;
    },

    hasFirings() {
      return (this.firingData.rows || []).length > 0;
    },

    // ── Firing-grid helpers ─────────────────────────────────────────────
    //
    // The grid sits above the existing ticker table — same firing data,
    // wide read instead of deep read. Rows are the same (ticker, outcome)
    // entries the table uses; columns are every tracked signal in
    // signal_id order. Cell stats reuse what's already in the /firing
    // payload — ticker_slice per firing entry for TICKER mode, overall
    // for ALL mode, scope_b/scope_a for row-end union dedup. No parallel
    // computation; the table and grid read the same source values so
    // they can't disagree.

    // Shade + thumbnail wrappers delegate to the shared SignalThumb
    // globals — guarantees no scale drift between a grid cell shaded by
    // cellColor and the column-header thumbnail above it (literally the
    // same function).
    signalThumbnailSVG(sig) {
      return window.SignalThumb.thumbnailSVG(sig, { width: 70, clickable: false });
    },
    cellColor(avgRet) {
      return window.SignalThumb.cellColor(avgRet);
    },
    cellOpacity(n) {
      return window.SignalThumb.cellOpacity(n);
    },

    // Per-row lookup: Map of signal_id → its signals_firing entry. Built
    // once per row in template via x-init / x-effect to keep per-cell
    // access O(1).
    gridBuildFiringMap(row) {
      const m = new Map();
      for (const sf of (row.signals_firing || [])) m.set(sf.signal_id, sf);
      return m;
    },

    // Per-cell value in the current stat mode. Returns null for empty
    // cells (ticker didn't fire this signal). For filled cells, returns
    // { avg_ret, n } drawn from ticker_slice (TICKER) or overall (ALL).
    gridCellValue(firingMap, sid) {
      const sf = firingMap.get(sid);
      if (!sf) return null;
      if (this.gridStatMode === 'all') {
        return { avg_ret: sf.overall && sf.overall.avg_ret, n: sf.overall && sf.overall.n };
      }
      // Default TICKER mode.
      return { avg_ret: sf.ticker_slice && sf.ticker_slice.avg_ret,
               n:       sf.ticker_slice && sf.ticker_slice.n };
    },

    // Row-end union-deduped aggregate for the current stat mode. Reads
    // the exact field the ticker table reads above (scope_b for ticker
    // mode, scope_a for ALL mode), so the two views literally share the
    // source number and can't drift.
    gridRowAgg(row) {
      return this.gridStatMode === 'all' ? (row.scope_a || null)
                                          : (row.scope_b || null);
    },

    // Per-column ticker count — how many rows fired this signal today.
    // The frequency read at the column foot. Same in both modes (firing
    // count is mode-independent).
    gridColTickerCount(sid) {
      let n = 0;
      for (const row of (this.firingData.rows || [])) {
        for (const sf of (row.signals_firing || [])) {
          if (sf.signal_id === sid) { n++; break; }
        }
      }
      return n;
    },

    // Combined cell opacity: cellOpacity(n) for the small-n shade
    // honesty cue, multiplied by the min-n slider fade (0.32 when below
    // the slider, 1.0 otherwise). A cell with n=8 hits BOTH dimmings
    // and goes very faint — exactly the "ignore me" reading the slider
    // is for.
    gridCellOpacity(cell) {
      if (!cell || cell.n == null) return 0.35;
      const base = this.cellOpacity(cell.n);
      const fade = (cell.n < this.gridMinN) ? 0.32 : 1.0;
      return base * fade;
    },

    // Stat-mode + slider setters. Pure draw — no fetch.
    setGridStatMode(mode) {
      this.gridStatMode = (mode === 'all') ? 'all' : 'ticker';
    },
    setGridMinN(v) {
      const n = parseInt(v, 10);
      if (!isNaN(n) && n >= 0) this.gridMinN = n;
    },

    // Number formatting reused by the cell render.
    gridFmtPct(x) {
      if (x == null || !isFinite(x)) return '—';
      return (x * 100).toFixed(2) + '%';
    },
    gridFmtN(x) {
      if (x == null) return '—';
      return Number(x).toLocaleString();
    },

    // Composite (ticker, outcome) row key. A ticker firing on two
    // outcome horizons is two rows, and they expand/collapse independently.
    rowKey(row) {
      return (row.ticker || '') + '|' + (row.outcome || '');
    },

    // ── Sorting ─────────────────────────────────────────────────────────
    // sortBy: first click on a new column uses a column-appropriate
    // default direction (alpha for ticker, descending for everything
    // numeric — biggest at the top is what you usually want); repeat
    // clicks flip. Active-column arrow + highlight live in sortArrow /
    // sortClass below, read by the thead template.
    sortBy(key) {
      if (this.sortKey === key) {
        this.sortDir = this.sortDir === 'asc' ? 'desc' : 'asc';
      } else {
        this.sortKey = key;
        // String columns default ASC (alphabetical); numeric DESC (largest first).
        this.sortDir = (key === 'ticker' || key === 'outcome') ? 'asc' : 'desc';
      }
    },
    sortClass(key) {
      return this.sortKey === key ? 'sort-active' : '';
    },
    sortArrow(key) {
      if (this.sortKey !== key) return '';
      return this.sortDir === 'asc' ? '▲' : '▼';
    },

    // Sort-key dispatcher — maps the column code from the template's
    // @click to the actual field on a ticker row. Kept in one place so
    // adding a sortable column = one line here + one <th> in the template.
    _sortValueFor(row, key) {
      const a = row.scope_a || {};
      const s = row.scope_a_stability || {};
      const b = row.scope_b || {};
      switch (key) {
        case 'ticker':           return row.ticker;
        case 'outcome':          return row.outcome;
        case 'n_signals_firing': return row.n_signals_firing;
        case 'a_n':    return a.n;
        case 'a_avg':  return a.avg_ret;
        case 'a_med':  return a.median;
        case 'a_std':  return a.std_dev;
        case 'a_p5':   return a.p5;
        case 'a_p95':  return a.p95;
        case 'a_win':  return a.win_rate;
        case 'a_avgw': return a.avg_win;
        case 'a_avgl': return a.avg_loss;
        // Stability — positive-years column sorts by RATIO (9/9 ranks
        // above 8/10). Returns null on no-data so it sinks under the
        // null-handler rule below.
        case 's_pyr':
          return s.total_years > 0
            ? s.positive_years / s.total_years
            : null;
        case 's_cva':  return s.cv_yearly_avg_ret;       // may be null near zero
        case 's_dn':   return s.dispersion_yearly_n;
        case 'b_n':    return b.n;
        case 'b_avg':  return b.avg_ret;
        case 'b_win':  return b.win_rate;
        default:       return 0;
      }
    },

    get sortedTickers() {
      const arr = (this.firingData.rows || []).slice();
      const key = this.sortKey;
      const dir = this.sortDir === 'asc' ? 1 : -1;
      // Nulls / NaN / undefined always sink to the bottom regardless of
      // direction — a missing CV is "no data," not a tiny number. Same
      // convention as a standard spreadsheet sort.
      const isMissing = (v) =>
        v === null || v === undefined ||
        (typeof v === 'number' && !isFinite(v));
      arr.sort((rowA, rowB) => {
        const va = this._sortValueFor(rowA, key);
        const vb = this._sortValueFor(rowB, key);
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

    // ── Row expand/collapse — keyed by (ticker, outcome) ────────────────

    toggleRow(key) {
      this.expandedRows = {
        ...this.expandedRows,
        [key]: !this.expandedRows[key],
      };
    },
    expandAll() {
      const next = {};
      for (const r of (this.firingData.rows || [])) next[this.rowKey(r)] = true;
      this.expandedRows = next;
    },
    collapseAll() {
      this.expandedRows = {};
    },

    // ── Tracked-signals add paths ───────────────────────────────────────

    async addTrackedSignal() {
      // Called by the "Add" button next to the signal dropdown — NOT on
      // bare selection. The user picks first, then commits.
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
          // Refresh BOTH firing and roster so the new signal shows up in
          // the bottom list right away — no manual reload.
          await Promise.all([this.loadFiring(), this.loadRoster()]);
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
      // Called by the "Add" button next to the portfolio dropdown.
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
          await Promise.all([this.loadFiring(), this.loadRoster()]);
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

    // ── Calendar (Open Positions Gantt) ──────────────────────────────────

    async loadCalendar() {
      this.calLoading = true;
      try {
        const r = await fetch('/api/factor-signals/calendar');
        if (r.ok) {
          this.calEntries = await r.json();
          this._updateGanttRange();
        }
      } catch (_) {}
      this.calLoading = false;
    },

    async addToCalendar(ticker, outcome, entryDate) {
      // Calendar identity is (ticker, outcome, entry_date). No signal_id
      // — the calendar doesn't track which signals fired, just that the
      // ticker is on for that horizon on that day. Same ticker+outcome
      // re-added on the same day is a no-op (ON CONFLICT DO NOTHING).
      try {
        const r = await fetch('/api/factor-signals/calendar', {
          method:  'POST',
          headers: { 'Content-Type': 'application/json' },
          body:    JSON.stringify({
            ticker:     ticker,
            outcome:    outcome,
            entry_date: entryDate,
          }),
        });
        if (r.ok) {
          this._showToast(`Added to calendar: ${ticker} · ${outcome}`);
          await this.loadCalendar();
        } else {
          const detail = (await r.json().catch(() => ({}))).detail || 'add failed';
          this._showToast(`Calendar add failed: ${detail}`, true);
        }
      } catch (_) {
        this._showToast('Calendar add failed — network error', true);
      }
    },

    async removeFromCalendar(id) {
      try {
        await fetch(`/api/factor-signals/calendar/${id}`, { method: 'DELETE' });
      } catch (_) {}
      this.calEntries = this.calEntries.filter(e => e.id !== id);
      this._updateGanttRange();
    },

    // Deterministic per-ticker bar color. The HUE comes from a hash of
    // the ticker symbol (two large coprime multipliers spread visually
    // close hashes apart, so JPM/JNJ/JCI don't all land within 10° of
    // each other) — so ADBE is always the same color FAMILY across days
    // and across horizons. The LIGHTNESS varies by outcome horizon so a
    // ticker on simultaneous positions at different horizons renders as
    // distinguishable shades of the same hue family — ADBE-1d lighter,
    // ADBE-5d darker, both still recognizably "ADBE".
    tickerColor(ticker, outcome) {
      const s = String(ticker || '').toUpperCase();
      let h = 0;
      for (let i = 0; i < s.length; i++) {
        h = (h * 131 + s.charCodeAt(i) * 977) >>> 0;
      }
      const hue = h % 360;

      // Parse horizon from outcome (e.g. ret_5d_fwd_oc → 5). Map to a
      // lightness ladder that puts visible distance between the common
      // horizons (1d / 3d / 5d / 10d) without losing the ticker hue.
      let lightness = 55;
      if (outcome) {
        const m = String(outcome).match(/(\d+)d/);
        if (m) {
          const h_days = parseInt(m[1], 10);
          if      (h_days <= 1)  lightness = 68;
          else if (h_days <= 2)  lightness = 62;
          else if (h_days <= 3)  lightness = 56;
          else if (h_days <= 5)  lightness = 46;
          else if (h_days <= 10) lightness = 36;
          else                   lightness = 28;
        }
      }
      return `hsl(${hue}, 65%, ${lightness}%)`;
    },

    // ── Roster (every tracked signal, firing or not) ────────────────────

    async loadRoster() {
      this.rosterLoading = true;
      try {
        const r = await fetch('/api/factor-signals/roster');
        if (r.ok) {
          const j = await r.json();
          this.roster = j.tracked || [];
        }
      } catch (_) {}
      this.rosterLoading = false;
    },

    toggleRoster() {
      this.rosterCollapsed = !this.rosterCollapsed;
    },

    async untrackSignal(signalId, signalName) {
      // Removes from tracked_signals → drops from BOTH the firing view
      // (top) and the roster (bottom). Both reloaded after delete so the
      // page reflects the change immediately.
      if (!confirm(`Untrack "${signalName}"?`)) return;
      try {
        await fetch(`/api/factor-signals/tracked/${signalId}`, { method: 'DELETE' });
        this._showToast(`Untracked '${signalName}'`);
        await Promise.all([this.loadFiring(), this.loadRoster()]);
      } catch (_) {
        this._showToast('Untrack failed — network error', true);
      }
    },

    // Roster sort dispatcher — separate state from firing sort so each
    // table remembers its own column independently.
    sortRosterBy(key) {
      if (this.rosterSortKey === key) {
        this.rosterSortDir = this.rosterSortDir === 'asc' ? 'desc' : 'asc';
      } else {
        this.rosterSortKey = key;
        // tracked_at default DESC (newest first); name default ASC; other
        // numeric columns default DESC (largest first).
        this.rosterSortDir = key === 'name' ? 'asc' : 'desc';
      }
    },
    rosterSortClass(key) {
      return this.rosterSortKey === key ? 'sort-active' : '';
    },
    rosterSortArrow(key) {
      if (this.rosterSortKey !== key) return '';
      return this.rosterSortDir === 'asc' ? '▲' : '▼';
    },
    _rosterValueFor(row, key) {
      const o = row.overall   || {};
      const s = row.stability || {};
      switch (key) {
        case 'tracked_at': return row.tracked_at;
        case 'name':       return (row.name || '').toLowerCase();
        case 'outcome':    return row.outcome;
        case 'o_n':    return o.n;
        case 'o_avg':  return o.avg_ret;
        case 'o_med':  return o.median;
        case 'o_std':  return o.std_dev;
        case 'o_p5':   return o.p5;
        case 'o_p95':  return o.p95;
        case 'o_win':  return o.win_rate;
        case 'o_avgw': return o.avg_win;
        case 'o_avgl': return o.avg_loss;
        case 's_pyr':  return s.total_years > 0 ? s.positive_years / s.total_years : null;
        case 's_cva':  return s.cv_yearly_avg_ret;
        case 's_dn':   return s.dispersion_yearly_n;
        default:       return 0;
      }
    },
    get sortedRoster() {
      const arr = (this.roster || []).slice();
      const key = this.rosterSortKey;
      const dir = this.rosterSortDir === 'asc' ? 1 : -1;
      const isMissing = (v) =>
        v === null || v === undefined ||
        (typeof v === 'number' && !isFinite(v));
      arr.sort((rowA, rowB) => {
        const va = this._rosterValueFor(rowA, key);
        const vb = this._rosterValueFor(rowB, key);
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

    // ── Gantt helpers (preserved verbatim — math unchanged from Stage 2) ─

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
      // Color is derived from the (ticker, outcome) — the hue is the
      // ticker's family, lightness varies by outcome horizon so
      // simultaneous positions on the same ticker at different
      // horizons are visibly distinguishable.
      const col = this.tickerColor(entry.ticker, entry.outcome);
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
