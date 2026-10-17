/* Strategy Signal — is each configured strategy eligible to enter right now?
 *
 * The server decides (app/strategy_signal/evaluate.py) and sends each
 * strategy's state and chart series in ONE call, /board, remade every five
 * minutes. This file draws it and edits the configurations; it computes no
 * metric and no decision of its own.
 *
 * Charts live in SS_CHARTS, OUTSIDE the Alpine proxy: a Chart.js instance
 * wrapped in a reactive proxy is slow and breaks on update.
 */

const SS_BLUE = [0x34, 0x98, 0xdb];     // #3498db — favourable
const SS_PINK = [0xe8, 0x43, 0x93];     // #e84393 — unfavourable
const SS_LINE = '#d6d9dc';
const SS_REFRESH_MS = 5 * 60 * 1000;
const SS_WEEKDAYS = [{ n: 1, label: 'Mon', full: 'Monday' }, { n: 2, label: 'Tue', full: 'Tuesday' },
                     { n: 3, label: 'Wed', full: 'Wednesday' }, { n: 4, label: 'Thu', full: 'Thursday' },
                     { n: 5, label: 'Fri', full: 'Friday' }];
const SS_CHARTS = {};

/* The weekly calendar: 30-minute rows over the regular session, in Eastern
 * time. An entry time is placed in the row it falls in (10:15 -> the 10:00
 * row); 16:00 belongs to the last row. The calendar is a SCHEDULE: where a
 * card sits says when the strategy is usually considered, never whether it
 * trades -- the state on the card is the same decision as its section's. */
const SS_CAL_OPEN = 9 * 60 + 30;
const SS_CAL_CLOSE = 16 * 60;
const SS_CAL_STEP = 30;
const SS_CAL_SLOTS = (SS_CAL_CLOSE - SS_CAL_OPEN) / SS_CAL_STEP;   // 13
const SS_CLOCK_MS = 30 * 1000;
const SS_TZ = 'America/New_York';
const SS_PREF_LAYOUT = 'ss.layout';
const SS_PREF_COLLAPSED = 'ss.collapsed';
const SS_WD = { Mon: 1, Tue: 2, Wed: 3, Thu: 4, Fri: 5, Sat: 6, Sun: 7 };

function ssMinutes(hhmm) {
  const [h, m] = String(hhmm || '').split(':').map(Number);
  return Number.isFinite(h) && Number.isFinite(m) ? h * 60 + m : null;
}

function ssHhmm(min) {
  return `${String(Math.floor(min / 60)).padStart(2, '0')}:${String(min % 60).padStart(2, '0')}`;
}

/* "10:00" -> 1 (the 10:00 row); 16:00 is the last row's end, so row 12. */
function ssSlotOf(hhmm) {
  const m = ssMinutes(hhmm);
  if (m === null) return null;
  return Math.max(0, Math.min(SS_CAL_SLOTS - 1, Math.floor((m - SS_CAL_OPEN) / SS_CAL_STEP)));
}

/* A moment as Eastern wall time -- by the zone's own rules, so the
 * daylight-saving change is the browser's tz database, not arithmetic here. */
const SS_ET_FMT = new Intl.DateTimeFormat('en-US', {
  timeZone: SS_TZ, year: 'numeric', month: '2-digit', day: '2-digit', weekday: 'short',
  hour: '2-digit', minute: '2-digit', second: '2-digit', hourCycle: 'h23' });
function ssEt(d) {
  const p = {};
  for (const x of SS_ET_FMT.formatToParts(d)) p[x.type] = x.value;
  const h = Number(p.hour) % 24;
  return { iso: `${p.year}-${p.month}-${p.day}`, wd: SS_WD[p.weekday],
           min: h * 60 + Number(p.minute) + Number(p.second) / 60 };
}

/* Strategy order, then time: the one order every list on the page uses. */
function ssByPosition(a, b) {
  return (a.strategy.position - b.strategy.position) || (a.strategy.id - b.strategy.id);
}

/* Where each strategy goes at the top of the page. A calendar strategy has
 * one card per scheduled (day, time); a scheduled day with no time yet sits
 * in the "no time" row for that day. A calendar strategy with no scheduled
 * day at all would be on no row, so it is shown beside the calendar instead
 * -- a strategy is never on neither. Card strategies are never on the
 * calendar. */
function ssPlace(entries) {
  const slots = {}, untimed = {}, side = [];
  const put = (map, key, item) => { (map[key] = map[key] || []).push(item); };
  for (const e of [...entries].sort(ssByPosition)) {
    const s = e.strategy;
    if (s.display === 'card') { side.push({ e, unscheduled: false }); continue; }
    const days = s.weekdays || [];
    if (!days.length) { side.push({ e, unscheduled: true }); continue; }
    const times = s.entry_times || [];
    for (const d of days) {
      const mine = times.filter(t => t.day === d);
      if (!mine.length) put(untimed, d, { e, day: d, time: null });
      for (const t of mine) put(slots, `${d}|${ssSlotOf(t.time)}`, { e, day: d, time: t.time });
    }
  }
  for (const k of Object.keys(slots)) {
    slots[k].sort((a, b) => (ssMinutes(a.time) - ssMinutes(b.time)) || ssByPosition(a.e, b.e));
  }
  return { slots, untimed, side };
}

function ssLoad(key, fallback) {
  try { const v = localStorage.getItem(key); return v === null ? fallback : JSON.parse(v); }
  catch (e) { return fallback; }
}
function ssStore(key, v) {
  try { localStorage.setItem(key, JSON.stringify(v)); } catch (e) { /* private window: per-visit only */ }
}

/* State i of n on the blue→pink scale. Two states are exactly the theme's
 * blue and pink; more are evenly spaced between them (through purple, never
 * a third hue). */
function ssStateColor(n, i) {
  const f = n <= 1 ? 0 : i / (n - 1);
  const c = SS_BLUE.map((b, k) => Math.round(b + (SS_PINK[k] - b) * f));
  return '#' + c.map(v => v.toString(16).padStart(2, '0')).join('');
}

/* A value for an axis tick or a tooltip, at a precision that suits its size:
 * a ratio near 1 wants four decimals, an index level none. */
function ssFmt(v) {
  if (v === null || v === undefined || !Number.isFinite(v)) return '—';
  const a = Math.abs(v);
  const dp = a >= 1000 ? 0 : a >= 100 ? 1 : a >= 10 ? 2 : a >= 1 ? 3 : 4;
  return v.toLocaleString('en-US', { minimumFractionDigits: dp, maximumFractionDigits: dp });
}

/* One comparison, as the server makes it (evaluate.compare). Used only to
 * decide which side of a threshold to shade; the decision is the server's. */
function ssCompare(v, cmp, t) {
  if (cmp === '<') return v < t;
  if (cmp === '<=') return v <= t;
  if (cmp === '>') return v > t;
  if (cmp === '>=') return v >= t;
  return Math.abs(v - t) <= 1e-9 * Math.max(1, Math.abs(t));
}

const SS_ZONE_ALPHA = 0.13;

function ssRgba(hex, a) {
  const n = parseInt(hex.slice(1), 16);
  return `rgba(${n >> 16}, ${(n >> 8) & 255}, ${n & 255}, ${a})`;
}

/* The decision zones of ONE condition on its own axis: the value ranges
 * between its thresholds, each labelled with the state that range leads to
 * (the first level whose threshold it passes, else the last state), in that
 * state's colour. Adjacent ranges with the same state merge. `=` has no
 * range to shade -- only the line itself passes. */
function ssZones(cmp, thresholds, n) {
  if (cmp === '=' || !thresholds.length) return [];
  const pts = [...new Set(thresholds)].sort((a, b) => a - b);
  const edges = [null, ...pts, null];
  const zones = [];
  for (let k = 0; k < edges.length - 1; k++) {
    const lo = edges[k], hi = edges[k + 1];
    const probe = lo === null ? hi - 1 : hi === null ? lo + 1 : (lo + hi) / 2;
    let st = thresholds.findIndex(t => ssCompare(probe, cmp, t));
    if (st < 0) st = n - 1;
    const last = zones[zones.length - 1];
    if (last && last.state === st) last.hi = hi;
    else zones.push({ lo, hi, state: st, color: ssRgba(ssStateColor(n, st), SS_ZONE_ALPHA) });
  }
  return zones;
}

/* Draws the zones behind the data, clipped to the plot area. */
const ssZonePlugin = {
  id: 'ssZones',
  beforeDatasetsDraw(chart, args, opts) {
    const zones = opts && opts.zones;
    if (!zones || !zones.length) return;
    const { ctx, chartArea: a, scales: { y } } = chart;
    ctx.save();
    for (const z of zones) {
      const top = z.hi === null ? a.top : Math.max(a.top, Math.min(a.bottom, y.getPixelForValue(z.hi)));
      const bot = z.lo === null ? a.bottom : Math.max(a.top, Math.min(a.bottom, y.getPixelForValue(z.lo)));
      if (bot > top) { ctx.fillStyle = z.color; ctx.fillRect(a.left, top, a.right - a.left, bot - top); }
    }
    ctx.restore();
  },
};

/* [1,2,3,4] -> "Mon–Thu"; [1,3,5] -> "Mon, Wed, Fri". */
function ssDaysText(days) {
  if (!days.length) return 'none';
  const d = [...days].sort((a, b) => a - b);
  const runs = [];
  for (const x of d) {
    const r = runs[runs.length - 1];
    if (r && x === r[1] + 1) r[1] = x; else runs.push([x, x]);
  }
  const nm = i => SS_WEEKDAYS[i - 1].label;
  return runs.map(([a, b]) => (a === b ? nm(a) : b === a + 1 ? `${nm(a)}, ${nm(b)}` : `${nm(a)}–${nm(b)}`)).join(', ');
}

/* ── the trade definition, in words ─────────────────────────────────────────
 * ONE formatter for the hover popover and the section's panel, so the two
 * cannot describe a trade differently. Information only: nothing here is
 * read by the decision. Empty fields produce no row. */
const SS_TIP_OPEN_MS = 220;      // hover this long before it opens (no flicker on a pass-over)
const SS_TIP_CLOSE_MS = 300;     // and this long to reach it before it closes

function ssNum(v) { return Number.isInteger(v) ? String(v) : String(+Number(v).toFixed(4)); }

function ssLevelText(l) {
  if (!l) return '';
  const v = l.value === null || l.value === undefined ? ''
    : l.unit === 'usd' ? `$${ssNum(l.value)}` : l.unit === 'pts' ? `${ssNum(l.value)} pts` : `${ssNum(l.value)}%`;
  return [v, l.basis].filter(Boolean).join(' — ');
}

/* "SELL 1 × 30Δ Call, 1 DTE"; "BUY 1 × Call, same strike as leg 1 (short call), 6 DTE". */
function ssLegText(leg, legs) {
  const right = leg.right === 'call' ? 'Call' : 'Put', st = leg.strike || {};
  let k = right, extra = '';
  if (st.method === 'delta') k = `${ssNum(st.value)}Δ ${right}`;
  else if (st.method === 'atm') k = `ATM ${right}`;
  else if (st.method === 'offset') extra = `, SPX ${st.value < 0 ? '−' : '+'}${ssNum(Math.abs(st.value))}`;
  else if (st.method === 'same') {
    const i = legs.findIndex(l => l.id === st.ref), r = legs[i];
    extra = r ? `, same strike as leg ${i + 1} (${r.side === 'sell' ? 'short' : 'long'} ${r.right})`
              : ', same strike as a leg that no longer exists';
  }
  return `${leg.side.toUpperCase()} ${leg.qty} × ${k}${extra}, ${leg.dte} DTE${leg.note ? ' — ' + leg.note : ''}`;
}

/* The schedule grouped by time: "Tue / Wed / Thu 15:00 ET · Mon (no time)". */
function ssEntryText(s) {
  const days = s.weekdays || [], times = s.entry_times || [];
  const groups = new Map();
  for (const d of days) {
    const key = times.filter(t => t.day === d).map(t => t.time).join(', ');
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key).push(SS_WEEKDAYS[d - 1].label);
  }
  return [...groups].map(([t, ds]) => `${ds.join(' / ')} ${t ? t + ' ET' : '(no time)'}`).join(' · ');
}

function ssExitText(ex) {
  if (!ex) return '';
  return [ex.exit_time ? `${ex.exit_time} ET` : '', ex.exit_dte !== null && ex.exit_dte !== undefined ? `${ex.exit_dte} DTE` : '',
          ex.max_dit !== null && ex.max_dit !== undefined ? `max ${ex.max_dit} DIT` : ''].filter(Boolean).join(' · ');
}

/* [{k, v}] or {k, legs: [...]} rows, in reading order, for one strategy. */
function ssTradeRows(s) {
  const t = s.trade;
  if (!t) return [];
  const rows = [];
  const add = (k, v) => { if (v) rows.push({ k, v }); };
  add('Entry', ssEntryText(s));
  add('Entry exceptions', t.entry_notes);
  if ((t.legs || []).length) rows.push({ k: 'Legs', legs: t.legs.map(l => ssLegText(l, t.legs)) });
  add('Profit target', ssLevelText(t.target));
  add('Stop loss', ssLevelText(t.stop));
  add('Exit', ssExitText(t.exit));
  add('Exit rules', t.exit && t.exit.rules);
  add('Management', t.management);
  return rows;
}

/* The card the popover belongs to, outside the proxy like SS_CHARTS (a DOM
 * element in a reactive proxy is wrapped for nothing). */
const SS_TIP_ANCHOR = { el: null };

/* The staged image File, outside the Alpine proxy like SS_CHARTS: it is
 * only ever handed to FormData, never rendered. */
const SS_STAGED = { file: null };

function ssUid() { return 'm' + Math.random().toString(36).slice(2, 8); }

document.addEventListener('alpine:init', () => {
  Alpine.data('strategySignal', () => ({
    board: null,
    evalDate: '',         // '' = live; else a past date evaluated at its close
    loading: false,
    error: '',
    sources: null,
    sourcesError: '',
    editing: false,
    draft: null,          // kept after the panel closes; see the template
    saving: false,
    saveError: '',
    weekdays: SS_WEEKDAYS,
    cmps: ['<', '<=', '>', '>=', '='],
    clock: new Date(),    // ticks every SS_CLOCK_MS; the calendar's time line reads it
    // Per-screen preferences, local storage like the Wall's pane width: how
    // this monitor lays the page out is not a fact about the strategies.
    layout: ssLoad(SS_PREF_LAYOUT, 'full') === 'two' ? 'two' : 'full',
    collapsed: ssLoad(SS_PREF_COLLAPSED, {}) || {},
    tip: null,            // {id, top, left}: the trade-definition popover, one for the page
    tipTimer: null,
    tdOpen: {},           // section Trade Definition panels opened this visit (closed by default)

    init() {
      this.loadBoard();
      setInterval(() => { if (!this.editing && !this.evalDate) this.loadBoard(); }, SS_REFRESH_MS);
      setInterval(() => { this.clock = new Date(); this.placeNowLine(); }, SS_CLOCK_MS);
      // Rows grow with their cards, so the line's height is measured, not
      // computed -- again whenever the calendar changes size.
      this.$nextTick(() => {
        if (window.ResizeObserver && this.$refs.cal) {
          new ResizeObserver(() => this.placeNowLine()).observe(this.$refs.cal);
        }
      });
    },

    // ── the board ────────────────────────────────────────────────────────
    async loadBoard() {
      if (this.loading) return;
      this.loading = true;
      try {
        const q = this.evalDate ? `?as_of=${encodeURIComponent(this.evalDate)}` : '';
        const r = await fetch('/api/strategy-signal/board' + q);
        if (!r.ok) throw new Error(`board: HTTP ${r.status} ${(await r.text()).slice(0, 300)}`);
        this.board = await r.json();
        this.error = '';
      } catch (e) {
        this.error = String(e.message || e);
      } finally {
        this.loading = false;
      }
      this.$nextTick(() => { this.drawAll(); this.placeNowLine(); });
    },

    entries() { return this.board ? this.board.strategies : []; },

    // ── the weekly calendar and the standalone cards ─────────────────────
    placement() { return ssPlace(this.entries()); },
    sideItems() { return this.placement().side; },
    sideEntries() { return this.sideItems().map(x => x.e); },
    unscheduled(e) { return e.strategy.display !== 'card' && !(e.strategy.weekdays || []).length; },
    calSlots() {
      return Array.from({ length: SS_CAL_SLOTS }, (_, i) => ({ i, label: ssHhmm(SS_CAL_OPEN + i * SS_CAL_STEP) }));
    },
    calCell(day, slot) { return this.placement().slots[`${day}|${slot}`] || []; },
    calUntimed(day) { return this.placement().untimed[day] || []; },
    hasUntimed() { return Object.keys(this.placement().untimed).length > 0; },

    /* The live moment in Eastern time, or null on a past date: a past
     * evaluation is at that day's close, and a "now" line over it would
     * describe a different day than everything else on the page. */
    liveEt() { return this.board && this.board.live ? ssEt(this.clock) : null; },
    isTodayCol(n) { const et = this.liveEt(); return !!et && et.wd === n; },
    isEvalCol(n) { return !!this.board && !this.board.live && this.board.iso_weekday === n; },

    /* Minutes past midnight ET for the time line, or null when it would
     * mislead: not live, not a session (weekend, holiday), the board is for
     * another day, or outside the session's own hours (an early close ends
     * at 13:00). */
    nowMinutes() {
      const et = this.liveEt(), b = this.board;
      if (!et || !b.is_session || !b.session || et.iso !== b.date) return null;
      const open = ssMinutes(b.session.open), close = ssMinutes(b.session.close);
      return et.min >= open && et.min <= close ? et.min : null;
    },

    calCorner() {
      const b = this.board;
      if (!b) return '';
      if (!b.live) return 'past';
      const et = this.liveEt();
      return et ? ssHhmm(Math.floor(et.min)) : '';
    },

    calNote() {
      const b = this.board;
      if (!b) return '';
      if (!b.live) return `Schedule only — the states are at the close of ${b.weekday} ${b.date}, not live`;
      if (this.nowMinutes() !== null) return '';
      const et = this.liveEt();
      if (!b.is_session || (et && et.iso !== b.date)) return 'Market closed today';
      const close = b.session ? b.session.close : '16:00';
      return et && et.min < ssMinutes(b.session ? b.session.open : '09:30') ? 'Before the open' : `After the close (${close} ET)`;
    },

    placeNowLine() {
      const line = this.$refs.nowLine, grid = this.$refs.calGrid;
      if (!line || !grid) return;
      const m = this.nowMinutes();
      const slot = m === null ? null : Math.min(SS_CAL_SLOTS - 1, Math.floor((m - SS_CAL_OPEN) / SS_CAL_STEP));
      const cell = slot === null ? null : grid.querySelector(`.ss-cal-time[data-slot="${slot}"]`);
      const first = grid.querySelector('.ss-cal-day');
      if (!cell || !first) { line.style.display = 'none'; return; }
      const frac = (m - SS_CAL_OPEN - slot * SS_CAL_STEP) / SS_CAL_STEP;
      line.style.display = 'block';
      line.style.top = `${grid.offsetTop + cell.offsetTop + frac * cell.offsetHeight}px`;
      line.style.left = `${grid.offsetLeft + first.offsetLeft}px`;
      line.style.width = `${grid.offsetWidth - first.offsetLeft}px`;
      line.title = `${ssHhmm(Math.floor(m))} ET`;
    },

    miniTitle(c) {
      const s = c.e.strategy, d = SS_WEEKDAYS[c.day - 1].label;
      const when = c.time ? `${d} ${c.time} ET` : `${d}, no time set`;
      return `${s.name} — scheduled ${when} — ${this.stateLabel(c.e)}${c.e.stale ? ' (data not current)' : ''}`;
    },

    /* "Tue 10:00 · Fri 11:30", with untimed days by name: "Mon (no time)". */
    scheduleText(s) {
      const days = s.weekdays || [], times = s.entry_times || [];
      if (!days.length) return 'no scheduled days';
      const parts = [];
      const untimed = days.filter(d => !times.some(t => t.day === d));
      for (const d of days) {
        for (const t of times.filter(x => x.day === d)) parts.push(`${SS_WEEKDAYS[d - 1].label} ${t.time}`);
      }
      if (untimed.length) parts.push(`${ssDaysText(untimed)} (no time)`);
      return parts.join(' · ');
    },

    imageUrl(s, size) {
      return s.image ? `/api/strategy-signal/strategies/${s.id}/image?size=${size}&v=${s.image.v}` : '';
    },

    // ── the trade definition: popover and panel ─────────────────────────
    hasTrade(e) { return !!(e && e.strategy.trade); },
    tradeRows(e) { return e ? ssTradeRows(e.strategy) : []; },
    tradeSummary(s) {
      const t = s.trade;
      if (!t) return '';
      const n = (t.legs || []).length;
      return [n ? `${n} leg${n > 1 ? 's' : ''}` : '', t.target ? 'target ' + ssLevelText(t.target).split(' — ')[0] : '',
              t.stop ? 'stop ' + ssLevelText(t.stop).split(' — ')[0] : ''].filter(Boolean).join(' · ');
    },

    /* Hover opens after a short pause and closes after a short grace, which
     * the popover itself cancels -- so the pointer can travel into it to
     * read. A card without a definition opens nothing. Clicking the card
     * still goes to its section; the popover never takes the click. */
    tipEnter(e, ev) {
      if (!this.hasTrade(e)) return;
      const el = ev.currentTarget;
      clearTimeout(this.tipTimer);
      this.tipTimer = setTimeout(() => this.openTip(e, el), this.tip ? 60 : SS_TIP_OPEN_MS);
    },
    tipLeave() {
      clearTimeout(this.tipTimer);
      this.tipTimer = setTimeout(() => { this.tip = null; }, SS_TIP_CLOSE_MS);
    },
    tipHold() { clearTimeout(this.tipTimer); },
    closeTip() { clearTimeout(this.tipTimer); this.tip = null; },

    /* The ⓘ on a card, for screens without hover: toggles the popover and
     * stops there, so the card's own click (go to the section) does not fire. */
    tipToggle(e, ev) {
      ev.stopPropagation();
      if (this.tip && this.tip.id === e.strategy.id) { this.closeTip(); return; }
      this.openTip(e, ev.currentTarget.closest('.ss-mini, .ss-card'));
    },

    openTip(e, el) {
      if (!el || !el.isConnected) return;
      SS_TIP_ANCHOR.el = el;
      this.tip = { id: e.strategy.id, top: -9999, left: -9999 };
      this.$nextTick(() => this.placeTip());
    },
    /* Beside the card (right if it fits, else left), else below it, kept
     * inside the window. */
    placeTip() {
      const pop = this.$refs.tip, el = SS_TIP_ANCHOR.el;
      if (!pop || !this.tip || !el || !el.isConnected) return;
      const r = el.getBoundingClientRect(), w = pop.offsetWidth, h = pop.offsetHeight;
      const vw = document.documentElement.clientWidth, vh = document.documentElement.clientHeight, gap = 8;
      let left = r.right + gap + w <= vw ? r.right + gap : r.left - gap - w;
      let top = r.top;
      if (left < gap) { left = Math.min(Math.max(gap, r.left), vw - w - gap); top = r.bottom + gap; }
      top = Math.max(gap, Math.min(top, vh - h - gap));
      this.tip = { id: this.tip.id, top, left };
    },
    /* The page scrolled: the popover follows its card while the card is on
     * screen, and closes once it is not -- a definition beside nothing. */
    tipScrolled() {
      if (!this.tip) return;
      const el = SS_TIP_ANCHOR.el;
      const r = el && el.isConnected ? el.getBoundingClientRect() : null;
      // Measured against the scrolling area, not the window: a card scrolled
      // up under the top bar is off screen though still inside the window.
      const box = el && el.closest('.ss-main') ? el.closest('.ss-main').getBoundingClientRect()
        : { top: 0, bottom: document.documentElement.clientHeight };
      if (!r || r.bottom <= box.top || r.top >= box.bottom) { this.closeTip(); return; }
      this.placeTip();
    },
    tipEntry() { return this.tip ? this.entries().find(x => x.strategy.id === this.tip.id) || null : null; },
    // Visibility is HERE, not x-show: x-show hides at once but shows a frame
    // later, so a close inside that frame (a scroll, a click away) was
    // followed by a stale show -- an empty popover left on screen. An object,
    // too: a string :style would replace the whole attribute.
    tipStyle() {
      return this.tip ? { display: 'block', top: `${this.tip.top}px`, left: `${this.tip.left}px` } : { display: 'none' };
    },

    isTdOpen(e) { return !!this.tdOpen[e.strategy.id]; },
    toggleTd(e) { this.tdOpen = { ...this.tdOpen, [e.strategy.id]: !this.tdOpen[e.strategy.id] }; },

    // ── layout and collapsing ───────────────────────────────────────────
    setLayout(v) {
      this.layout = v;
      ssStore(SS_PREF_LAYOUT, v);
    },
    isCollapsed(e) { return !!this.collapsed[e.strategy.id]; },
    toggleCollapse(e) {
      const c = { ...this.collapsed };
      const opening = !!c[e.strategy.id];
      if (opening) delete c[e.strategy.id]; else c[e.strategy.id] = true;
      this.collapsed = c;
      ssStore(SS_PREF_COLLAPSED, c);
      // A chart drawn while its section was hidden measured nothing; it is
      // given its size once the section is shown again, not left at zero.
      if (opening) {
        this.$nextTick(() => {
          const pre = `ss-c-${e.strategy.id}-`;
          for (const id of Object.keys(SS_CHARTS)) if (id.startsWith(pre)) SS_CHARTS[id].resize();
        });
      }
    },

    boardSub() {
      if (!this.board) return this.loading ? 'loading…' : '';
      if (!this.board.live) return `As of the close of ${this.board.weekday} ${this.board.date}`;
      return `${this.board.weekday} ${this.board.now} ET · refreshes every 5 min`;
    },

    todayIso() { return ssEt(new Date()).iso; },     // the exchange's today, not the laptop's

    pickDate(v) {
      this.evalDate = v && v !== this.todayIso() ? v : '';
      this.loadBoard();
    },

    pastBanner() {
      const b = this.board;
      const shut = b.is_session ? '' : " The exchange was closed that day, so the data is the previous session's.";
      return `Evaluated at the close of ${b.weekday_full} ${b.date}, not live — the same rules run on ` +
             `that day's last observations, with charts ending there.${shut}`;
    },

    markText(m) { return m === 'pass' ? 'pass' : m === 'fail' ? 'fail' : 'no value'; },

    /* Every input to the decision as a row: each signal condition with its
     * current value and what it needs. A multi-level condition lists each
     * level's threshold with its own pass/fail, because "which level did
     * this reach" is the question. The schedule is not an input and has no
     * row; it is stated under the checklist. */
    whyRows(e) {
      const s = e.strategy, d = e.decision, n = s.states.length;
      const rows = [];
      for (const c of d.conditions) {
        const m = s.metrics.find(x => x.id === c.metric) || { thresholds: [] };
        const needs = c.passes.map((p, i) => ({
          text: `${n > 2 ? s.states[i] + ' ' : 'needs '}${c.cmp} ${ssFmt(m.thresholds[i])}`,
          mark: p === null ? 'none' : p ? 'pass' : 'fail' }));
        const best = c.passes.findIndex(p => p === true);
        const mark = c.value === null ? 'none' : (n > 2 ? (best >= 0 ? 'pass' : 'fail') : needs[0].mark);
        rows.push({ name: this.metricInfo(e, c.metric).label, value: ssFmt(c.value), mark, needs });
      }
      return rows;
    },

    /* The verdict in words, naming what decided it, then the schedule --
     * stated, never applied. */
    summaryLine(e) {
      const s = e.strategy, d = e.decision, logic = s.logic.toUpperCase();
      const verdict = d.reason === 'no_conditions'
        ? `No signal conditions → ${s.states[d.state]}`
        : d.state === null
          ? `Conditions (${logic}) cannot be decided — a value is missing → NO DATA`
          : `Conditions (${logic}) → ${s.states[d.state]}`;
      if (!(s.weekdays || []).length) return verdict;
      const day = this.board.weekday;
      return `${verdict} · ${day} is ${d.scheduled_today ? '' : 'not '}a scheduled day (the schedule does not change the signal)`;
    },

    /* The one phrase a card carries, and only when the state did not come
     * from the market: missing data. No values. */
    cardNote(e) {
      if (e.decision.reason === 'no_data') return 'a signal value is missing';
      return '';
    },

    stateIndex(e) { return e.decision.state; },
    stateLabel(e) {
      const i = e.decision.state;
      return i === null ? 'NO DATA' : e.strategy.states[i];
    },
    color(e) {
      const i = e.decision.state;
      return i === null ? 'transparent' : ssStateColor(e.strategy.states.length, i);
    },
    stateColor(n, i) { return ssStateColor(n, i); },

    /* Manual requirements are prominent whenever the automatic answer is to
     * enter at all — every state but the last. They never change it. */
    showManual(e) {
      const i = e.decision.state;
      return e.strategy.manual.length > 0 && i !== null && i < e.strategy.states.length - 1;
    },

    /* The card's one line of identification: the metrics that decide (or,
     * with none, the ones charted). */
    idLine(e) {
      const s = e.strategy;
      const sig = s.metrics.filter(m => m.signal);
      const ms = (sig.length ? sig : s.metrics).map(m => this.metricInfo(e, m.id).label);
      return ms.join(' · ') || 'no signal conditions';
    },

    reason(e) {
      const d = e.decision;
      if (d.reason === 'no_data') {
        const miss = d.conditions.filter(c => c.value === null)
          .map(c => this.metricInfo(e, c.metric).label);
        return `No current value for ${miss.join(', ')}`;
      }
      if (d.reason === 'no_conditions') return 'No signal conditions';
      return '';
    },

    asOf(e) {
      const used = new Set(e.strategy.metrics.map(m => m.id));
      const ts = e.metrics.filter(m => used.has(m.id) && m.as_of).map(m => m.as_of).sort();
      if (!ts.length) return '';
      return (e.stale ? '⚠ ' : '') + 'data through ' + ts[ts.length - 1];
    },

    metricInfo(e, id) {
      return e.metrics.find(m => m.id === id) || { label: id, error: null };
    },

    chartMetrics(e) { return e.strategy.metrics.filter(m => m.chart); },

    chartSub(e, m) {
      const several = e.strategy.metrics.filter(x => x.signal).length > 1;
      const lb = { '5d': '5 days', '10d': '10 days', '1m': '1 month', '3m': '3 months',
                   '6m': '6 months', '1y': '1 year', '2y': '2 years' }[m.lookback] || m.lookback;
      const shade = m.signal && m.cmp !== '=' && several ? ' · shading: this condition alone' : '';
      return `${m.resolution === 'intraday' ? '5-min' : 'daily'} · ${lb}${shade}`;
    },

    canvasId(e, m) { return `ss-c-${e.strategy.id}-${m.id}`; },

    /* From a calendar or side card to the strategy's section, opening it
     * first if it was collapsed. */
    scrollTo(e) {
      if (this.isCollapsed(e)) this.toggleCollapse(e);
      this.$nextTick(() => {
        const el = document.getElementById('ss-sec-' + e.strategy.id);
        if (el) el.scrollIntoView({ behavior: 'smooth', block: 'start' });
      });
    },

    async move(e, dir) {
      await fetch(`/api/strategy-signal/strategies/${e.strategy.id}/move`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ direction: dir }) });
      this.loadBoard();
    },

    // ── charts ───────────────────────────────────────────────────────────
    drawAll() {
      const live = new Set();
      for (const e of this.entries()) {
        for (const m of this.chartMetrics(e)) {
          const id = this.canvasId(e, m);
          live.add(id);
          this.draw(id, e, m);
        }
      }
      for (const id of Object.keys(SS_CHARTS)) {
        if (!live.has(id)) { SS_CHARTS[id].destroy(); delete SS_CHARTS[id]; }
      }
    },

    draw(id, e, m) {
      const el = document.getElementById(id);
      if (!el) return;
      if (SS_CHARTS[id] && SS_CHARTS[id].canvas !== el) { SS_CHARTS[id].destroy(); delete SS_CHARTS[id]; }
      const data = e.charts[m.id] || { t: [], v: [] };
      const t = data.t;
      const intraday = m.resolution === 'intraday';
      const pts = data.v.map((v, i) => ({ x: i, y: v }));
      const n = e.strategy.states.length;
      const datasets = [{ label: this.metricInfo(e, m.id).label, data: pts, borderColor: SS_LINE,
                          borderWidth: 1.6, pointRadius: 0, pointHoverRadius: 4, tension: 0,
                          spanGaps: false }];
      // A decision threshold is a flat line in the colour of the state it
      // leads to: "below this, FULL". Visualisation-only metrics draw none.
      if (m.signal) {
        m.thresholds.forEach((th, i) => {
          datasets.push({ label: `${m.cmp} ${ssFmt(th)} → ${e.strategy.states[i]}`,
                          data: [{ x: 0, y: th }, { x: Math.max(t.length - 1, 1), y: th }],
                          borderColor: ssStateColor(n, i), borderWidth: 1, borderDash: [5, 4],
                          pointRadius: 0, pointHoverRadius: 0, isThreshold: true });
        });
      }
      // Intraday ticks sit on each session's first bar, so an overnight gap
      // is a tick, not a stretch of empty axis.
      const dayStarts = [];
      if (intraday) {
        t.forEach((s, i) => { if (i === 0 || s.slice(0, 10) !== t[i - 1].slice(0, 10)) dayStarts.push(i); });
      }
      const short = s => (s ? s.slice(5, 10) : '');
      const opts = {
        animation: false, responsive: true, maintainAspectRatio: false,
        interaction: { mode: 'nearest', axis: 'x', intersect: false },
        plugins: {
          ssZones: { zones: m.signal ? ssZones(m.cmp, m.thresholds, n) : [] },
          legend: { display: !!m.signal, position: 'top', align: 'end',
                    labels: { color: '#c8c8c8', boxWidth: 14, boxHeight: 1, font: { size: 10 },
                              filter: item => item.datasetIndex > 0 } },
          tooltip: {
            filter: item => item.datasetIndex === 0,
            callbacks: {
              title: items => (items.length ? t[items[0].parsed.x] || '' : ''),
              label: item => `${item.dataset.label}: ${ssFmt(item.parsed.y)}`,
            },
          },
        },
        scales: {
          x: {
            type: 'linear', min: 0, max: Math.max(t.length - 1, 1),
            grid: { color: 'rgba(255,255,255,0.05)' },
            ticks: {
              color: '#9a9a9a', font: { size: 10 }, maxRotation: 0, autoSkip: !intraday,
              maxTicksLimit: intraday ? undefined : 8,
              callback: v => {
                const s = t[Math.round(v)];
                if (!s) return '';
                return (!intraday && t.length > 300) ? s.slice(2, 7) : short(s);
              },
            },
            afterBuildTicks: axis => {
              if (!intraday) return;
              const step = Math.max(1, Math.ceil(dayStarts.length / 12));
              axis.ticks = dayStarts.filter((_, k) => k % step === 0).map(v => ({ value: v }));
            },
          },
          y: {
            grid: { color: 'rgba(255,255,255,0.05)' },
            ticks: { color: '#9a9a9a', font: { size: 10 }, maxTicksLimit: 6, callback: v => ssFmt(v) },
          },
        },
      };
      if (SS_CHARTS[id]) {
        SS_CHARTS[id].data.datasets = datasets;
        SS_CHARTS[id].options = opts;
        SS_CHARTS[id].update('none');
      } else {
        SS_CHARTS[id] = new Chart(el.getContext('2d'), { type: 'line', data: { datasets }, options: opts,
                                                          plugins: [ssZonePlugin] });
      }
    },

    // ── configuration ────────────────────────────────────────────────────
    async ensureSources() {
      if (this.sources) return;
      this.sourcesError = '';
      try {
        const r = await fetch('/api/strategy-signal/sources');
        if (!r.ok) throw new Error(`metric catalog: HTTP ${r.status} ${(await r.text()).slice(0, 300)}`);
        this.sources = await r.json();
        this.cmps = this.sources.cmps;
      } catch (e) {
        this.sourcesError = String(e.message || e);
      }
    },

    /* Every source, for the datalist: the id is what the input holds, the
     * text is what the browser shows beside it while you type. */
    sourceOptions() {
      if (!this.sources) return [];
      const idx = this.sources.index.map(s => ({ id: s.id, text: `${s.label} — ${s.description}` }));
      const surf = this.sources.surface.map(s => ({
        id: s.id, text: `${s.group} · ${s.form_label}${s.description ? ' — ' + s.description : ''}` }));
      return idx.concat(surf);
    },

    sourceById(id) {
      if (!this.sources || !id) return null;
      return this.sources.index.find(s => s.id === id) || this.sources.surface.find(s => s.id === id) || null;
    },

    srcShort(id) {
      const s = this.sourceById(id);
      if (!s) return id || '';
      return s.label || s.column;
    },

    autoLabel(m) {
      if (!m.a) return '';
      let s = this.srcShort(m.a);
      if (m.op) s += ` ${m.op} ${this.srcShort(m.b)}`;
      if (m.transform) s += ' · pctl 252';
      return s;
    },

    metricHint(m) {
      const parts = [];
      for (const id of [m.a, m.op ? m.b : null]) {
        if (!id) continue;
        const s = this.sourceById(id);
        if (!s) { parts.push(this.sources ? `"${id}" is not an available metric` : ''); continue; }
        if (s.column) {
          parts.push(`${s.column}: ${s.description || s.form_label}` +
                     (s.units ? ` [${s.units}]` : '') + (s.min_date ? `, from ${s.min_date}` : ''));
        } else {
          parts.push(`${s.label}: ${s.description} (index_ohlc close)`);
        }
      }
      if (m.transform) parts.push('percentile of the value among the previous 252 session closes, 0–100');
      return parts.filter(Boolean).join(' · ');
    },

    logicHint() {
      if (!this.draft) return '';
      return this.draft.logic === 'and'
        ? 'A state needs every signal condition passing. The schedule never enters it.'
        : 'A state needs at least one signal condition passing. The schedule never enters it.';
    },

    lookbacksFor(m) {
      const all = this.sources ? this.sources.lookbacks : [];
      return m.resolution === 'intraday' ? all.filter(l => l.intraday) : all;
    },

    fixLookback(m) {
      const ok = this.lookbacksFor(m).map(l => l.id);
      if (ok.length && !ok.includes(m.lookback)) m.lookback = m.resolution === 'intraday' ? '10d' : '1y';
    },

    blankMetric() {
      return { id: ssUid(), label: '', a: '', op: '', b: '', transform: '', chart: true,
               resolution: 'daily', lookback: '1y', signal: false, cmp: '<',
               thresholds: Array(this.draft.states.length - 1).fill('') };
    },

    /* The editor holds times per day ({1: ['10:00'], ...}); the store holds
     * [{day, time}]. A day's times are kept only while the day is ticked. */
    timesByDay(list) {
      const t = { 1: [], 2: [], 3: [], 4: [], 5: [] };
      for (const x of list || []) if (t[x.day]) t[x.day].push(x.time);
      return t;
    },

    newStrategy() {
      // No days and no times: a schedule is something the user states, so
      // nothing is pre-ticked.
      this.draft = { id: null, name: '', notes: '', weekdays: [], times: this.timesByDay([]),
                     display: 'calendar', image: null, imageAction: null, imagePreview: '',
                     trade: this.draftTrade(null), tradeOpen: false,
                     states: ['TRADE', 'NO TRADE'], logic: 'and', metrics: [], manual: [] };
      this.draft.metrics.push(this.blankMetric());
      this.openPanel();
    },

    editStrategy(e) {
      const s = JSON.parse(JSON.stringify(e.strategy));
      s.metrics.forEach(m => {
        m.op = m.op || ''; m.b = m.b || ''; m.transform = m.transform || ''; m.cmp = m.cmp || '<';
        if (!m.signal) m.thresholds = Array(s.states.length - 1).fill('');
      });
      s.weekdays = s.weekdays || [];
      s.times = this.timesByDay(s.entry_times);
      s.display = s.display || 'calendar';
      s.trade = this.draftTrade(s.trade);
      s.tradeOpen = false;      // collapsed; its summary line says what is in it
      s.imageAction = null;     // null | 'set' | 'remove' -- applied by Save, after the config
      s.imagePreview = '';
      this.draft = s;
      this.openPanel();
    },

    /* The editor's trade: every field present, numbers as input strings. */
    draftTrade(t) {
      const str = v => (v === null || v === undefined ? '' : String(v));
      const lvl = l => ({ value: str(l && l.value), unit: (l && l.unit) || 'pct', basis: (l && l.basis) || '' });
      t = t || {};
      const ex = t.exit || {};
      return {
        entry_notes: t.entry_notes || '',
        legs: (t.legs || []).map(l => ({ id: l.id, side: l.side, right: l.right, qty: str(l.qty), dte: str(l.dte),
                                          method: l.strike.method, value: str(l.strike.value), ref: l.strike.ref || '',
                                          note: l.note || '' })),
        target: lvl(t.target), stop: lvl(t.stop),
        exit: { max_dit: str(ex.max_dit), exit_dte: str(ex.exit_dte), exit_time: ex.exit_time || '', rules: ex.rules || '' },
        management: t.management || '',
      };
    },
    addLeg() {
      this.draft.trade.legs.push({ id: 'L' + Math.random().toString(36).slice(2, 7), side: 'sell', right: 'put',
                                   qty: '1', dte: '', method: 'delta', value: '', ref: '', note: '' });
    },
    removeLeg(i) { this.draft.trade.legs.splice(i, 1); },
    moveLeg(i, dir) {
      const L = this.draft.trade.legs, j = i + dir;
      if (j < 0 || j >= L.length) return;
      [L[i], L[j]] = [L[j], L[i]];       // ids move with their legs; references follow them
    },
    /* The legs a "same strike as" may point at: every other leg, by its
     * current number. */
    refOptions(leg) {
      return this.draft.trade.legs.map((l, i) => ({ id: l.id, text: `Leg ${i + 1} · ${l.side.toUpperCase()} ${l.right} ${l.dte || '?'} DTE` }))
        .filter(o => o.id !== leg.id);
    },
    /* A same-strike reference whose leg was removed (or never chosen). Shown
     * on the leg and refused by the server -- never repointed silently. */
    brokenRef(leg) {
      return leg.method === 'same' && !this.draft.trade.legs.some(l => l.id === leg.ref && l.id !== leg.id);
    },
    /* The schedule as it stands in the editor (ticked days and their times),
     * in the popover's words -- the trade definition reuses it, never a copy. */
    draftEntryText() {
      const d = this.draft, entry_times = [];
      for (const n of d.weekdays) for (const t of d.times[n] || []) if (String(t).trim()) entry_times.push({ day: n, time: t });
      return ssEntryText({ weekdays: d.weekdays, entry_times });
    },
    draftTradeSummary() {
      const t = this.draft.trade, n = t.legs.length, parts = [];
      if (n) parts.push(`${n} leg${n > 1 ? 's' : ''}`);
      if (String(t.target.value).trim() || t.target.basis) parts.push('profit target');
      if (String(t.stop.value).trim() || t.stop.basis) parts.push('stop');
      if (t.entry_notes || t.management || Object.values(t.exit).some(v => String(v).trim())) parts.push('notes');
      return parts.length ? parts.join(' · ') : 'none — optional, information only';
    },
    tradePayload() {
      const t = this.draft.trade;
      const num = v => (String(v).trim() === '' ? null : Number(v));
      const lvl = l => ({ value: num(l.value), unit: l.unit, basis: l.basis });
      return {
        v: 1, entry_notes: t.entry_notes,
        legs: t.legs.map(l => ({ id: l.id, side: l.side, right: l.right, qty: num(l.qty), dte: num(l.dte),
                                 strike: { method: l.method,
                                           value: l.method === 'delta' || l.method === 'offset' ? num(l.value) : null,
                                           ref: l.method === 'same' ? (l.ref || null) : null },
                                 note: l.note })),
        target: lvl(t.target), stop: lvl(t.stop),
        exit: { max_dit: num(t.exit.max_dit), exit_dte: num(t.exit.exit_dte), exit_time: t.exit.exit_time || null,
                rules: t.exit.rules },
        management: t.management,
      };
    },

    addTime(n) {
      this.draft.times[n].push('10:00');
      if (!this.draft.weekdays.includes(n)) this.draft.weekdays = [...this.draft.weekdays, n].sort();
    },
    removeTime(n, i) { this.draft.times[n].splice(i, 1); },

    // ── the risk graph, staged until Save ───────────────────────────────
    pickImage(ev) {
      const f = ev.target.files && ev.target.files[0];
      ev.target.value = '';
      if (!f) return;
      if (this.draft.imagePreview) URL.revokeObjectURL(this.draft.imagePreview);
      SS_STAGED.file = f;
      this.draft.imageAction = 'set';
      this.draft.imagePreview = URL.createObjectURL(f);
      this.draft.imageName = f.name;
    },
    dropImage() {
      if (this.draft.imagePreview) URL.revokeObjectURL(this.draft.imagePreview);
      SS_STAGED.file = null;
      this.draft.imagePreview = '';
      this.draft.imageAction = this.draft.image ? 'remove' : null;
    },
    draftImageSrc() {
      const d = this.draft;
      if (d.imageAction === 'set') return d.imagePreview;
      if (d.imageAction === 'remove' || !d.image) return '';
      return this.imageUrl(d, 'thumb');
    },
    imageHint() {
      const d = this.draft;
      if (d.imageAction === 'set') return `${d.imageName || 'new image'} — uploaded when you Save`;
      if (d.imageAction === 'remove') return 'removed when you Save';
      if (d.image) return `${d.image.w} × ${d.image.h} px`;
      return 'PNG, JPEG, WebP or GIF, up to 8 MB — scaled to fit, never stretched';
    },

    openPanel() {
      this.editing = true;
      this.saveError = '';
      this.ensureSources();
      this.$nextTick(() => { if (this.$refs.cfg) this.$refs.cfg.scrollIntoView({ behavior: 'smooth' }); });
    },

    cancelEdit() {
      if (this.draft && this.draft.imagePreview) URL.revokeObjectURL(this.draft.imagePreview);
      SS_STAGED.file = null;
      this.editing = false; this.saveError = ''; this.$nextTick(() => this.drawAll());
    },

    toggleDay(n) {
      const w = this.draft.weekdays;
      if (w.includes(n)) this.draft.times[n] = [];
      this.draft.weekdays = w.includes(n) ? w.filter(d => d !== n) : [...w, n].sort();
    },

    // States and thresholds move together: each signal metric keeps one
    // threshold per state but the last, so resizing one resizes the other.
    syncThresholds() {
      const k = this.draft.states.length - 1;
      for (const m of this.draft.metrics) {
        const t = m.thresholds.slice(0, k);
        while (t.length < k) t.push('');
        m.thresholds = t;
      }
    },
    addState() {
      const s = this.draft.states;
      s.splice(s.length - 1, 0, `LEVEL ${s.length}`);
      this.syncThresholds();
    },
    removeState(i) {
      this.draft.states.splice(i, 1);
      if (i < this.draft.states.length) this.draft.metrics.forEach(m => m.thresholds.splice(i, 1));
      this.syncThresholds();
    },
    moveState(i, dir) {
      const s = this.draft.states, j = i + dir;
      if (j < 0 || j >= s.length) return;
      [s[i], s[j]] = [s[j], s[i]];
    },
    presetStates(list) {
      this.draft.states = [...list];
      this.syncThresholds();
    },

    addMetric() { this.draft.metrics.push(this.blankMetric()); },

    payload() {
      const d = this.draft;
      const entry_times = [];
      for (const n of d.weekdays) {
        for (const t of d.times[n] || []) if (String(t).trim()) entry_times.push({ day: n, time: String(t).trim() });
      }
      return {
        name: d.name, notes: d.notes, weekdays: d.weekdays, entry_times, display: d.display,
        trade: this.tradePayload(),
        states: d.states, logic: d.logic,
        manual: d.manual,
        metrics: d.metrics.map(m => ({
          id: m.id, label: m.label, a: (m.a || '').trim(), op: m.op || null,
          b: m.op ? (m.b || '').trim() : null, transform: m.transform || null,
          chart: m.chart, resolution: m.resolution, lookback: m.lookback,
          signal: m.signal, cmp: m.signal ? m.cmp : null,
          thresholds: m.signal ? m.thresholds.map(v => (String(v).trim() === '' ? null : Number(v))) : [],
        })),
      };
    },

    async save() {
      const broken = this.draft.trade.legs.findIndex(l => this.brokenRef(l));
      if (broken >= 0) {
        this.draft.tradeOpen = true;
        this.saveError = `Leg ${broken + 1} is set to the same strike as a leg that no longer exists — choose another leg or another method.`;
        return;
      }
      this.saving = true;
      this.saveError = '';
      try {
        const id = this.draft.id;
        const r = await fetch(id ? `/api/strategy-signal/strategies/${id}` : '/api/strategy-signal/strategies', {
          method: id ? 'PUT' : 'POST', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(this.payload()) });
        if (!r.ok) {
          const body = await r.json().catch(() => ({}));
          throw new Error(body.detail || `HTTP ${r.status}`);
        }
        const saved = ((await r.json().catch(() => ({}))).strategy) || {};
        // From here the strategy exists: a failed image step leaves the
        // panel open on it (an edit, not a second create) with the image
        // still staged, so Save again retries only what did not land.
        if (saved.id) this.draft.id = saved.id;
        await this.saveImage(this.draft.id);
        this.editing = false;
        SS_STAGED.file = null;
        await this.loadBoard();
      } catch (e) {
        this.saveError = String(e.message || e);
      } finally {
        this.saving = false;
      }
    },

    async saveImage(id) {
      const act = this.draft.imageAction;
      if (!act || !id) return;
      let r;
      if (act === 'set') {
        const fd = new FormData();
        fd.append('file', SS_STAGED.file);
        r = await fetch(`/api/strategy-signal/strategies/${id}/image`, { method: 'PUT', body: fd });
      } else {
        r = await fetch(`/api/strategy-signal/strategies/${id}/image`, { method: 'DELETE' });
      }
      if (!r.ok) {
        const body = await r.json().catch(() => ({}));
        throw new Error(`The strategy was saved, but the image was not: ${body.detail || 'HTTP ' + r.status}`);
      }
      this.draft.imageAction = null;
    },

    async remove() {
      if (!this.editing || !this.draft.id) return;
      if (!confirm(`Delete "${this.draft.name}"? This cannot be undone.`)) return;
      this.saving = true;
      try {
        const r = await fetch(`/api/strategy-signal/strategies/${this.draft.id}`, { method: 'DELETE' });
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        this.editing = false;
        await this.loadBoard();
      } catch (e) {
        this.saveError = String(e.message || e);
      } finally {
        this.saving = false;
      }
    },
  }));
});
