/* Equities Scan — a live quietness grid.
 *
 * WHAT THIS IS FOR, so the design choices below have something to answer to.
 * Finding a name to trade means opening a tape pane, waiting sixty seconds for
 * enough prints to judge it, deciding no, and moving on. Thirty names is an
 * hour. The grid does not need to predict anything — an episode-level analysis
 * over 170 episodes found every lead-in metric at p=1.000, so it cannot tell
 * you what will be good. It only has to be better than random at ordering
 * which name you open a pane on, which is a much lower bar and worth real
 * money.
 *
 * WINDOWED DOM, NOT CANVAS. Measured at 120 cells a row: a full repaint with
 * every row in the document is 390 ms at 600 rows and 29 ms with only the
 * viewport's rows present. Windowing is the variable that matters — canvas
 * buys another 10x on top and costs the hover, the pin and the tooltip
 * becoming hit-testing written by hand.
 *
 * THE COLOUR ALONE DOES NOT DISCRIMINATE. Measured live, the quiet ratio ran
 * p10 0.07, p50 0.42, p90 1.22 — most names are under 1.0 most of the time, so
 * a ramp spread over the whole range puts nearly everything in two shades.
 * Hence two ANCHORS the user places, the live percentiles printed beside them
 * so the placing is informed, and a reset that snaps them to p10/p90. Not a
 * ramp fitted to one morning's data, and not a relative score — the anchors
 * stay where they are put until they are moved.
 */
'use strict';

document.addEventListener('alpine:init', () => {
  Alpine.data('equitiesScan', () => ({

    // ── state ───────────────────────────────────────────────────────────
    rows: [],                 // sorted, all of them
    visible: [],              // only what the viewport shows
    bySym: {},                // sym -> row, so a tick updates in place
    cells: {},                // sym -> [[ratio, range, dollars, trades], ...]
    held: [],
    gridMinutes: 120,
    firstMinute: 0,
    currentMinute: null,

    volumeFloor: 750000,
    floorLog: 5.875,          // the slider is logarithmic; see onFloor
    ratioLow: 0.10,
    ratioHigh: 1.20,
    pct: { p10: null, p50: null, p90: null },

    // Both MINIMA: below either, a name has nothing to capture however it
    // behaves. Defaults come from /scan/defaults so the cents figure is the
    // pipeline's own min_spread_cents rather than a number invented here.
    minSpreadCents: 5,
    minSpreadBps: 0,
    hiddenBySpread: 0,
    unknownSpread: 0,

    pinned: {},
    hovered: null,
    showManual: false,
    manualText: '',
    seedMinRange: 5,
    seedMinDollar: 100000,

    busy: false,
    warning: '',
    connected: false,
    lastTick: null,
    seedCounts: null,
    scrollTop: 0,
    rowHeight: 18,
    expandedHeight: 40,
    sock: null,

    // ── lifecycle ───────────────────────────────────────────────────────
    async init() {
      this.restore();
      await this.loadDefaults();
      this.connect();
      // The grid re-windows on resize; the row set does not change, only how
      // many of them are in the document.
      window.addEventListener('resize', () => this.window_());
      setInterval(() => this.tickWatchdog(), 4000);
    },

    async loadDefaults() {
      // SERVED, NOT BAKED IN. The volume floor is arithmetic the user did —
      // a few percent of a minute's flow at their round-trip size — and it
      // belongs somewhere it can be changed without a rebuild. Local storage
      // wins over it, because a value the user set outranks a default.
      try {
        const r = await fetch('/scan/defaults');
        const d = await r.json();
        this.gridMinutes = d.grid_minutes || this.gridMinutes;
        if (!this.hasStored('floor')) {
          this.volumeFloor = d.volume_floor;
          this.floorLog = this.dollarsToLog(d.volume_floor);
        }
        if (!this.hasStored('anchors')) {
          this.ratioLow = d.ratio_low;
          this.ratioHigh = d.ratio_high;
        }
        if (!this.hasStored('spread')) {
          this.minSpreadCents = d.min_spread_cents;
          this.minSpreadBps = d.min_spread_bps;
        }
      } catch (e) {
        this.warning = 'could not read /scan/defaults: ' + e;
      }
    },

    // ── persistence ─────────────────────────────────────────────────────
    //
    // Local storage, per the brief: the anchors and the floor survive a
    // refresh AND a service restart, because they are the user's judgement
    // about where the resolution should sit and re-making it every deploy is
    // the thing that makes a control not worth setting.
    storeKey(k) { return 'equitiesScan.' + k; },
    hasStored(k) {
      try { return localStorage.getItem(this.storeKey(k)) !== null; }
      catch (e) { return false; }
    },
    save() {
      try {
        localStorage.setItem(this.storeKey('floor'), String(this.volumeFloor));
        localStorage.setItem(this.storeKey('anchors'),
          JSON.stringify([this.ratioLow, this.ratioHigh]));
        localStorage.setItem(this.storeKey('symbols'), JSON.stringify(this.held));
        localStorage.setItem(this.storeKey('seed'),
          JSON.stringify([this.seedMinRange, this.seedMinDollar]));
        localStorage.setItem(this.storeKey('spread'),
          JSON.stringify([this.minSpreadCents, this.minSpreadBps]));
      } catch (e) { /* private mode, quota — the page still works */ }
    },
    restore() {
      try {
        const f = localStorage.getItem(this.storeKey('floor'));
        if (f !== null) {
          this.volumeFloor = Number(f);
          this.floorLog = this.dollarsToLog(this.volumeFloor);
        }
        const a = localStorage.getItem(this.storeKey('anchors'));
        if (a) {
          const v = JSON.parse(a);
          this.ratioLow = v[0];
          this.ratioHigh = v[1];
        }
        const s = localStorage.getItem(this.storeKey('seed'));
        if (s) {
          const v = JSON.parse(s);
          this.seedMinRange = v[0];
          this.seedMinDollar = v[1];
        }
        const sp = localStorage.getItem(this.storeKey('spread'));
        if (sp) {
          const v = JSON.parse(sp);
          this.minSpreadCents = v[0];
          this.minSpreadBps = v[1];
        }
      } catch (e) { /* ignore — defaults are fine */ }
    },

    // ── the socket ──────────────────────────────────────────────────────
    connect() {
      const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
      const sock = new WebSocket(proto + '//' + location.host + '/scan/ws');
      this.sock = sock;
      sock.onopen = () => {
        this.connected = true;
        this.warning = '';
        this.requestGrid();
      };
      sock.onclose = () => {
        this.connected = false;
        // Reconnect, but say so. A grid that quietly stops updating looks
        // exactly like a market that went still, which is the one thing this
        // page exists to tell apart.
        this.warning = 'disconnected — retrying';
        setTimeout(() => this.connect(), 2000);
      };
      sock.onmessage = (e) => this.onMessage(JSON.parse(e.data));
    },

    requestGrid() {
      if (this.sock && this.sock.readyState === 1) {
        this.sock.send(JSON.stringify({ action: 'grid',
                                        minutes: this.gridMinutes }));
      }
    },

    onMessage(m) {
      if (m.ev === 'grid') {
        this.cells = m.cells || {};
        this.firstMinute = m.first_minute;
        this.currentMinute = m.current_minute;
        this.held = m.symbols || [];
        this.rebuild();
      } else if (m.ev === 'tick') {
        this.applyTick(m);
      } else if (m.ev === 'hello' || m.ev === 'status') {
        this.onStatus(m.data);
      }
    },

    onStatus(st) {
      if (!st) return;
      const scan = st.scan || {};
      if (scan.truncated_count) {
        // A truncated ring draws a SHORT range bar, which is
        // indistinguishable from a narrow one. Named rather than absorbed.
        this.warning = scan.truncated_count + ' symbol(s) truncated: ' +
          (scan.truncated || []).slice(0, 6).join(' ') +
          ' — raise LIVE_SCAN_RING_MAX; their range bars read short';
      }
      if (st.delayed) this.warning = 'DELAYED FEED — this is not live';
    },

    applyTick(m) {
      this.lastTick = Date.now();
      this.currentMinute = m.minute;
      const live = m.live || {};
      // A NEW MINUTE MEANS THE GRID SCROLLED. Ask for it again rather than
      // shifting cells here: the server owns which minute is which, and two
      // implementations of "everything moves left by one" is how an axis ends
      // up off by one for an hour.
      if (this.lastMinuteSeen !== undefined &&
          this.lastMinuteSeen !== m.minute) {
        this.requestGrid();
      }
      this.lastMinuteSeen = m.minute;

      for (const sym in live) {
        const v = live[sym];
        const row = this.bySym[sym];
        if (!row) continue;
        row.ratio = v[0];
        row.range = v[1];
        row.dollars = v[2];
        row.trades = v[3];
        row.price = v[4];
        row.spreadC = v[5];
        row.spreadB = v[6];
      }
      this.recomputePercentiles();
      this.resort();
    },

    tickWatchdog() {
      // The tape's own staleness rule: say it rather than let a frozen grid
      // read as a quiet market.
      if (this.connected && this.lastTick &&
          Date.now() - this.lastTick > 20000) {
        this.warning = 'no tick for ' +
          Math.round((Date.now() - this.lastTick) / 1000) + 's';
      }
    },

    // ── the ramp ────────────────────────────────────────────────────────
    //
    // Brightest at the low anchor, dark at the high, gradient between,
    // UNIFORMLY DARK ABOVE. Not normalised to the current distribution: the
    // anchors stay where they are put until they are moved, so a cell's
    // colour means the same thing today as it did an hour ago.
    colourFor(ratio) {
      if (ratio === null || ratio === undefined || !isFinite(ratio)) {
        return 'transparent';
      }
      const lo = this.ratioLow, hi = this.ratioHigh;
      if (ratio >= hi) return '#1b2027';
      const span = Math.max(1e-6, hi - lo);
      const k = 1 - Math.max(0, Math.min(1, (ratio - lo) / span));
      // One hue, moving in lightness and saturation. Two hues would read as
      // two categories, and quietness is a quantity.
      const l = 14 + k * 44;
      const s = 32 + k * 58;
      return 'hsl(204 ' + s.toFixed(0) + '% ' + l.toFixed(0) + '%)';
    },

    // The other two bands get their own ramps, so an expanded row shows three
    // different quantities rather than three shades of the same one.
    volColour(d) {
      if (d === null || !isFinite(d)) return 'transparent';
      const k = Math.max(0, Math.min(1, Math.log10(Math.max(1, d)) / 7));
      return 'hsl(45 ' + (30 + k * 60).toFixed(0) + '% ' +
             (10 + k * 45).toFixed(0) + '%)';
    },
    rangeColour(c) {
      if (c === null || !isFinite(c)) return 'transparent';
      const k = Math.max(0, Math.min(1, c / 40));
      return 'hsl(282 ' + (30 + k * 55).toFixed(0) + '% ' +
             (12 + k * 42).toFixed(0) + '%)';
    },

    // ── rows ────────────────────────────────────────────────────────────
    rebuild() {
      const rows = [];
      this.bySym = {};
      for (const sym of this.held) {
        const row = {
          sym: sym, ratio: null, range: null, dollars: null, trades: null,
          price: null, spreadC: null, spreadB: null, qual: 0, html: '',
        };
        rows.push(row);
        this.bySym[sym] = row;
      }
      this.rows = rows;
      this.renderAll();
      this.recomputePercentiles();
      this.resort();
    },

    // A minute counts if it was QUIET AND HAD VOLUME. Both, because a dead
    // name is quiet by default — that is the whole reason the gate exists.
    qualifying(series) {
      if (!series) return 0;
      let n = 0;
      for (const c of series) {
        if (!c) continue;
        if (c[0] !== null && c[0] < this.ratioHigh && c[2] !== null &&
            c[2] >= this.volumeFloor) n++;
      }
      return n;
    },

    renderAll() {
      for (const row of this.rows) this.renderRow(row);
    },

    // Built as an HTML string rather than as elements. One assignment per row
    // replaces 120 nodes, where creating them individually is 120 layout
    // invalidations for a row that is about to be positioned anyway.
    renderRow(row) {
      const series = this.cells[row.sym];
      row.qual = this.qualifying(series);
      row.html = this.bandHtml(series, 'quiet');
      if (this.isOpen(row.sym)) {
        row.html =
          '<div class="sc-bands">' +
          '<div class="sc-band">' + this.bandHtml(series, 'quiet') + '</div>' +
          '<div class="sc-band">' + this.bandHtml(series, 'vol') + '</div>' +
          '<div class="sc-band">' + this.bandHtml(series, 'range') + '</div>' +
          '</div>';
      }
    },

    bandHtml(series, kind) {
      const n = this.gridMinutes;
      const out = new Array(n);
      for (let i = 0; i < n; i++) {
        const c = series ? series[i] : null;
        let bg = 'transparent', cls = 'sc-cell';
        if (c) {
          if (kind === 'quiet') {
            // THE GATE OVERRIDES THE RATIO. Struck out below the floor
            // whatever the quietness says.
            const gated = c[2] === null || c[2] < this.volumeFloor;
            bg = this.colourFor(c[0]);
            if (gated) { cls += ' gated'; bg = '#191d23'; }
          } else if (kind === 'vol') {
            bg = this.volColour(c[2]);
          } else {
            bg = this.rangeColour(c[1]);
          }
        }
        out[i] = '<div class="' + cls + '" style="background:' + bg + '"></div>';
      }
      return out.join('');
    },

    // UNKNOWN IS NOT TIGHT. A symbol whose book has not quoted yet — the
    // first seconds after a seed, or a name nobody is making a market in —
    // has a NaN spread, and hiding it would empty the grid at startup and
    // silently drop names for a reason that is not "the spread is too tight".
    // It passes, renders dim and italic, and is counted so the state is
    // visible rather than inferred.
    passesSpread(r) {
      if (r.spreadC === null || r.spreadC === undefined ||
          !isFinite(r.spreadC)) return true;
      if (r.spreadC < this.minSpreadCents) return false;
      if (r.spreadB !== null && isFinite(r.spreadB) &&
          r.spreadB < this.minSpreadBps) return false;
      return true;
    },

    resort() {
      // ORDER BY QUALIFYING MINUTES. Not a score, and not a ranking that
      // asserts one name is better than another — the evidence does not
      // support that. It is a count of minutes that were quiet and had
      // volume, and the decision stays with the person reading it.
      this.rows.sort((a, b) => (b.qual - a.qual) ||
                               (b.dollars || 0) - (a.dollars || 0) ||
                               a.sym.localeCompare(b.sym));
      this.window_();
    },

    recomputePercentiles() {
      const v = [];
      for (const r of this.rows) {
        if (r.ratio !== null && isFinite(r.ratio)) v.push(r.ratio);
      }
      if (!v.length) { this.pct = { p10: null, p50: null, p90: null }; return; }
      v.sort((a, b) => a - b);
      const at = (q) => v[Math.min(v.length - 1, Math.floor(v.length * q))];
      this.pct = { p10: at(0.10), p50: at(0.50), p90: at(0.90) };
    },

    // ── windowing ───────────────────────────────────────────────────────
    onScroll() {
      this.scrollTop = this.$refs.scroll ? this.$refs.scroll.scrollTop : 0;
      this.window_();
    },

    window_() {
      const el = this.$refs.scroll;
      const h = el ? el.clientHeight : 700;
      const top = el ? el.scrollTop : 0;
      // Positions are cumulative because an expanded row is taller than a
      // collapsed one, so a row's offset is not its index times a constant.
      let y = 0;
      const vis = [];
      let hidden = 0, unknown = 0;
      for (const r of this.rows) {
        // The filter is applied HERE rather than by rebuilding a second
        // array, so a row that starts or stops passing does not disturb the
        // sort order or the scroll position — the list simply gets shorter.
        if (r.spreadC !== null && isFinite(r.spreadC)) {
          if (!this.passesSpread(r)) { hidden++; continue; }
        } else {
          unknown++;
        }
        const rh = this.isOpen(r.sym) ? this.expandedHeight : this.rowHeight;
        r._top = y;
        r._h = rh;
        if (y + rh >= top - 40 && y <= top + h + 40) vis.push(r);
        y += rh;
      }
      this._total = y;
      this.visible = vis;
      this.hiddenBySpread = hidden;
      this.unknownSpread = unknown;
    },

    totalHeight() { return this._total || 0; },
    gridWidth() { return this.gridMinutes * 7; },
    rowStyle(r) { return 'top:' + r._top + 'px;height:' + r._h + 'px'; },
    rowClass(r) {
      return (this.pinned[r.sym] ? 'pinned ' : '') +
             (this.isOpen(r.sym) ? 'open' : '');
    },

    // ── hover and pin ───────────────────────────────────────────────────
    //
    // NOT A MODE. A mode means remembering which one you are in and getting
    // it wrong at the moment it matters. Hovering expands in place, leaving
    // collapses, clicking pins it open, and several pins coexist for when the
    // choice is between two names.
    isOpen(sym) { return this.hovered === sym || !!this.pinned[sym]; },

    hover(sym) {
      if (this.hovered === sym) return;
      const prev = this.hovered;
      this.hovered = sym;
      if (prev && this.bySym[prev]) this.renderRow(this.bySym[prev]);
      if (this.bySym[sym]) this.renderRow(this.bySym[sym]);
      this.window_();
    },

    unhover(sym) {
      if (this.hovered !== sym) return;
      this.hovered = null;
      if (this.bySym[sym]) this.renderRow(this.bySym[sym]);
      this.window_();
    },

    togglePin(sym) {
      if (this.pinned[sym]) delete this.pinned[sym];
      else this.pinned[sym] = true;
      if (this.bySym[sym]) this.renderRow(this.bySym[sym]);
      this.window_();
    },

    // ── controls ────────────────────────────────────────────────────────
    //
    // The floor slider is LOGARITHMIC. Dollar flow spans four orders of
    // magnitude across the universe, and a linear slider would put every
    // usable value in the first two millimetres of travel.
    dollarsToLog(d) { return Math.log10(Math.max(1, d)); },
    onFloor() {
      this.volumeFloor = Math.round(Math.pow(10, this.floorLog));
      this.renderAll();
      this.resort();
      this.save();
    },

    onAnchors() {
      // The high anchor must stay above the low one or the ramp inverts and
      // quiet reads as loud, which is worse than no colour at all.
      if (this.ratioHigh <= this.ratioLow) {
        this.ratioHigh = this.ratioLow + 0.01;
      }
      this.renderAll();
      this.resort();
      this.save();
    },

    onSpread() {
      // No re-render: the cells are coloured by quietness and the filter only
      // decides which ROWS exist, so re-windowing is the whole of the work.
      this.window_();
      this.save();
    },

    snapAnchors() {
      // One click when the market's character changes, rather than dragging.
      if (this.pct.p10 === null) return;
      this.ratioLow = Math.round(this.pct.p10 * 100) / 100;
      this.ratioHigh = Math.max(this.ratioLow + 0.01,
                                Math.round(this.pct.p90 * 100) / 100);
      this.onAnchors();
    },

    // ── the symbol set ──────────────────────────────────────────────────
    async seedList() {
      this.busy = true;
      this.warning = '';
      try {
        const q = new URLSearchParams({
          min_range_cents: this.seedMinRange,
          min_dollar_per_min: this.seedMinDollar,
        });
        const r = await fetch('/scan/seed?' + q);
        const d = await r.json();
        if (d.error) { this.warning = 'seed failed: ' + d.error; return; }
        this.seedCounts = d.counts;
        await this.setSymbols((d.symbols || []).map((x) => x.symbol));
      } catch (e) {
        this.warning = 'seed failed: ' + e;
      } finally {
        this.busy = false;
        this.save();
      }
    },

    async applyManual() {
      const syms = this.manualText.toUpperCase().split(/[^A-Z0-9.]+/)
        .filter((s) => s.length);
      this.busy = true;
      try { await this.setSymbols(syms); } finally { this.busy = false; }
      this.save();
    },

    fillManualFromHeld() { this.manualText = this.held.join(' '); },

    async setSymbols(syms) {
      const r = await fetch('/scan/symbols', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ symbols: syms }),
      });
      const d = await r.json();
      if (d.error) { this.warning = d.error; return; }
      this.held = d.held || [];
      if (d.refused && d.refused.length) {
        this.warning = d.refused.length + ' refused: ' +
                       d.refused.slice(0, 3).join('; ');
      }
      this.requestGrid();
    },

    // ── formatting ──────────────────────────────────────────────────────
    fmtDollars(d) {
      if (d === null || d === undefined || !isFinite(d)) return '—';
      if (d >= 1e9) return (d / 1e9).toFixed(1) + 'B';
      if (d >= 1e6) return (d / 1e6).toFixed(1) + 'M';
      if (d >= 1e3) return Math.round(d / 1e3) + 'k';
      return String(Math.round(d));
    },
    fmtRange(c) {
      return (c === null || !isFinite(c)) ? '—' : c.toFixed(1) + 'c';
    },
    fmtPrice(p) {
      return (p === null || !isFinite(p)) ? '—' : p.toFixed(2);
    },
    fmtPct(v) { return v === null ? '—' : v.toFixed(2); },
    fmtSpreadC(c) {
      return (c === null || c === undefined || !isFinite(c))
        ? '—' : c.toFixed(1) + 'c';
    },
    fmtSpreadB(b) {
      return (b === null || b === undefined || !isFinite(b))
        ? '' : b.toFixed(1);
    },
    spreadClass(r) {
      return (r.spreadC === null || r.spreadC === undefined ||
              !isFinite(r.spreadC)) ? 'unknown' : '';
    },
    spreadTitle(r) {
      if (r.spreadC === null || !isFinite(r.spreadC)) {
        return r.sym + ' — no usable quotes yet. Unknown, not tight: it is ' +
               'not being filtered out.';
      }
      return r.sym + '  time-weighted quoted spread over the last ' +
             '5 minutes: ' + r.spreadC.toFixed(2) + ' cents, ' +
             r.spreadB.toFixed(2) + ' bps';
    },
    spreadSummary() {
      const parts = [];
      if (this.hiddenBySpread) parts.push(this.hiddenBySpread + ' hidden');
      if (this.unknownSpread) parts.push(this.unknownSpread + ' unquoted');
      return parts.join(', ');
    },

    // The bars are scaled against fixed ceilings, not against the current
    // maximum: a bar that rescales when one name spikes makes every other bar
    // move for a reason that has nothing to do with them.
    rangePct(r) {
      if (r.range === null || !isFinite(r.range)) return 0;
      return Math.max(2, Math.min(100, (r.range / 40) * 100));
    },
    volPct(r) {
      if (r.dollars === null || !isFinite(r.dollars)) return 0;
      const k = Math.log10(Math.max(1, r.dollars)) / 7;
      return Math.max(2, Math.min(100, k * 100));
    },

    nowColour(r) {
      if (r.dollars !== null && r.dollars < this.volumeFloor) return '#191d23';
      return this.colourFor(r.ratio);
    },
    nowTitle(r) {
      return r.sym + '  ratio ' + this.fmtPct(r.ratio) +
             '  range ' + this.fmtRange(r.range) +
             '  ' + this.fmtDollars(r.dollars) + '/min' +
             '  spread ' + this.fmtSpreadC(r.spreadC) +
             '  ' + (r.trades === null ? '—' : r.trades) + ' trades' +
             (r.dollars !== null && r.dollars < this.volumeFloor
               ? '  — under the floor' : '');
    },

    setSummary() {
      const c = this.seedCounts;
      return this.held.length + ' held' +
        (c ? '  (' + c.both + ' of ' + c.total + ' qualified and not ' +
             'spread-excluded)' : '');
    },
    connSummary() {
      if (!this.connected) return 'disconnected';
      const age = this.lastTick
        ? Math.round((Date.now() - this.lastTick) / 1000) + 's ago' : 'waiting';
      return 'live · last tick ' + age;
    },
    sortSummary() {
      return 'sorted by qualifying minutes — quiet AND above the floor';
    },
    emptyMessage() {
      return this.busy ? 'loading…'
        : 'no symbols held — press Seed, or Edit list to paste your own';
    },
  }));
});
