/* Backtest Portfolio — several saved strategies, combined.
 *
 * WHAT IT COMPUTES AND WHERE. Every number on this page comes from
 * backtest_core.js, the file /oo-backtest reads too: obApplyFilters,
 * obStats, obExtraStats, obConcurrency, obSharpe. That is the point of the
 * page — a strategy has to read the same here as it does there — and it is
 * why there is no second implementation of a statistic in this file.
 *
 * SETTLED, so nothing below re-opens it:
 *   qty scales P/L linearly, applied HERE, in the browser, and nowhere else.
 *   capital seeds from the strategy's saved value, is overridable, and the
 *     portfolio's capital is the SUM of capital x qty.
 *   the date range defaults to the UNION of the loaded spans.
 *   P/L is dated by CLOSE, exactly as the single-backtest page dates it.
 *   deployment is half-open [open, close): overnight capital only.
 *
 * THE TRADES ARE HELD OUTSIDE ALPINE (BP_DATA). A five-strategy portfolio is
 * ~10,000 trades of ~30 columns; a reactive proxy over that costs on every
 * read and buys nothing, because no template reads a trade. The reactive
 * state is the strategy list, the filters and the computed rows. The trap
 * that comes with it: anything derived from BP_DATA must be recomputed INTO
 * reactive state, never read through a getter Alpine cannot see change.
 */
'use strict';

const BP_DATA = {
  // Per (strategy, surface metric) values, so removing and re-adding a
  // metric costs nothing. Cleared by a load: the file behind a saved
  // strategy can have been re-saved since.
  surface: {},
  loadToken: 0,
  payloads: {},      // strategy id -> the server's payload
  scaled: {},        // strategy id -> { qty, cols } with pnl x qty
  extents: {},       // "id:metric" -> the observed range, for the sliders
  curves: {},        // the drawn series, rebuilt on every recompute
};

/* Chart.js instances. Outside Alpine for the same reason the trades are: a
 * chart is not state a template reads, and a reactive proxy around one is a
 * proxy around every point in it. */
const BP_CHARTS = { eq: null, dd: null, cap: null, sc: null, roll: null,
                    dist: null, overlap: null, risk: null };

/* The rolling-risk palette, the old app's: blue Sharpe, purple Sortino,
 * amber win rate on its own axis. */
const BP_SHARPE = '#3498db';
const BP_SORTINO = '#9b59b6';
const BP_WINRATE = '#f39c12';
/* The old app's histogram bin, in dollars of P/L. */
const BP_DIST_BIN = 100;

/* A hex colour at an opacity, for the rolling lines. */
function obRgbaFrom(hex, a) {
  const h = (hex || '#3498db').replace('#', '');
  const n = parseInt(h.length === 3 ? h.split('').map(c => c + c).join('') : h, 16);
  return `rgba(${(n >> 16) & 255},${(n >> 8) & 255},${n & 255},${a})`;
}

/* Badge colours per metric, ported from the old app's METRIC_BADGE_COLORS:
 * a pale ground with darker text of the same family, which is what makes a
 * row of them readable at a glance on a dark card. Keyed by the REGISTRY's
 * key (the old app's "dow" is our "day_of_week"); anything the registry
 * grows later falls back to the old default rather than being invented. */
const BP_BADGE = {
  vix:          ['#cce5ff', '#0d4a8a'],
  vix3m:        ['#d4d8ff', '#2d3494'],
  vix9d:        ['#e8d5ff', '#5b21b6'],
  gap:          ['#fff3c4', '#855a00'],
  vix_gap:      ['#ffe8cc', '#8a3a00'],
  premium:      ['#c6f6d5', '#166534'],
  vix3m_vix:    ['#b2f0f7', '#065a6b'],
  vix_vix9d:    ['#b2f0eb', '#075c56'],
  day_of_week:  ['#e2e8f0', '#374151'],
  exit_reason:  ['#d1d5db', '#1f2937'],
};
const BP_BADGE_FALLBACK = ['#1e2030', '#888888'];

const BP_BLUE = '#3498db';
const BP_PINK = '#e84393';
/* The portfolio's own line: white, over the strategies' colours. It is the
 * answer the page exists for, so it is not one of eight hues competing. */
const BP_TOTAL = '#e8ecf1';

/* A bound snapped outward to the registry's step, so a slider's ends are
 * round numbers rather than whatever the extreme trade happened to be. */
function obClampStep(v, step, how) {
  const s = Number(step);
  if (!isFinite(s) || s <= 0) return v;
  const k = v / s;
  return Number(((how === 'floor' ? Math.floor(k) : Math.ceil(k)) * s)
                .toPrecision(12));
}

function bpFmtInt(n) {
  return (n == null || !isFinite(n)) ? '—' : Math.round(n).toLocaleString();
}

/* FULL VALUES, ALWAYS. No $31k beside $7,853: two rows of a table that
 * abbreviate at different thresholds cannot be compared at a glance, which
 * is the entire job of a summary table. The same goes for the monthly grid,
 * the year totals, the axis ticks and the tooltips -- if something stops
 * fitting, widen it. */
function bpFmtMoney(v) {
  if (v == null || !isFinite(v)) return '—';
  const a = Math.round(Math.abs(v)).toLocaleString('en-US');
  return (v < 0 ? '-$' : '$') + a;
}

function bpFmtNum(v, dp = 2) {
  return (v == null || !isFinite(v)) ? '—' : v.toFixed(dp);
}

function bpFmtPct(v, dp = 1) {
  return (v == null || !isFinite(v)) ? '—' : v.toFixed(dp) + '%';
}

function bpFmtSeconds(s) {
  if (s == null || !isFinite(s)) return '—';
  return s >= 10 ? s.toFixed(0) + 's' : s.toFixed(1) + 's';
}

/* The union or the intersection of the loaded spans. UNION IS THE DEFAULT:
 * it uses every trade, at the cost of the early stretch having fewer
 * strategies in it — which the page states rather than letting the curve
 * imply everything ran from the start. Dates are ISO strings and compare as
 * strings; the payload builder makes them that way on purpose. */
function bpSpan(payloads, mode) {
  const lows = [], highs = [];
  for (const p of payloads) {
    if (p && p.date_min) lows.push(p.date_min);
    if (p && p.date_max) highs.push(p.date_max);
  }
  if (!lows.length || !highs.length) return null;
  return (mode === 'intersection')
    ? { start: lows.reduce((a, b) => (a > b ? a : b)),
        end: highs.reduce((a, b) => (a < b ? a : b)) }
    : { start: lows.reduce((a, b) => (a < b ? a : b)),
        end: highs.reduce((a, b) => (a > b ? a : b)) };
}

/* THE TWO SHADINGS. Both are the SAME GREY at two densities, and each is a
 * BINARY state rather than a scale:
 *
 *   strategy coverage — either every loaded strategy is live, or not
 *   metric coverage   — either every active filter could be evaluated, or not
 *
 * Grading them ("1 of 3" darker than "2 of 3") invited the reading that the
 * shade measured something, and a second hue or a texture would have read as
 * a different KIND of information. Where the two overlap they simply
 * compound and the stretch is darker still; that is not special-cased,
 * because "neither condition holds here" is exactly what darker should mean.
 */
const BP_SHADE = '154,154,154';
const BP_SHADE_STRATEGY = 0.09;
const BP_SHADE_METRIC = 0.20;

/* WHEN FEWER THAN ALL STRATEGIES WERE LIVE.
 *
 * On union dates every strategy's trades count, so the equity curve steepens
 * each time one starts and flattens as one finishes — which reads as the
 * portfolio getting better and then worse when it is only the membership
 * changing.
 *
 * TAKEN FROM THE DRAWN CURVES, not from the payload's date_min/date_max.
 * Those two mean different things — date_min is the earliest OPEN and
 * date_max the latest CLOSE — while these charts plot by CLOSE, so a
 * strategy's span started at its first trade's ENTRY while its line did not
 * begin until that trade EXITED. On positions held days to months that is a
 * gap of months, and the band claimed a strategy was live over a stretch
 * where it had drawn nothing. Reading the spans off the curves makes the two
 * agree by construction rather than by keeping two date bases in step.
 *
 * It also means the bands follow what is actually plotted: a filter that
 * removes a strategy's early trades moves its line, and the band moves with
 * it.
 *
 * Bands are merged by WHETHER ALL ARE LIVE, not by how many — a stretch that
 * goes from one strategy to two is one band, because both are "not all".
 * Fewer than two strategies with a line gets no bands: there is nothing to
 * be fewer than, and shading the whole chart would say something false. */
function bpLiveBands(series) {
  const spans = [];
  for (const sv of series || []) {
    if (!sv || sv.total || !sv.points || !sv.points.length) continue;
    const pts = sv.points;
    spans.push([obDay(pts[0].date), obDay(pts[pts.length - 1].date) + 1]);
  }
  const total = spans.length;
  if (total < 2) return { total, bands: [] };
  const cuts = [...new Set(spans.flat())].sort((a, b) => a - b);
  const bands = [];
  for (let i = 0; i < cuts.length - 1; i++) {
    const from = cuts[i], to = cuts[i + 1];
    let count = 0;
    for (const s of spans) if (s[0] <= from && s[1] >= to) count++;
    const partial = count < total;
    const last = bands[bands.length - 1];
    if (last && last.partial === partial && last.to === from) { last.to = to; continue; }
    bands.push({ from, to, partial });
  }
  return { total, bands };
}

/* Drawn UNDER the datasets, so the curves stay legible on top. The bands
 * ride in options.plugins.bpLiveShade rather than in the plugin array,
 * because `draw` reuses a live chart and only reassigns data and options — a
 * constructor array is read once and never again. */
function bpFillBand(chart, from, to, alpha) {
  const area = chart.chartArea, x = chart.scales && chart.scales.x;
  if (!area || !x) return false;
  const lo = Math.max(area.left, Math.min(area.right, x.getPixelForValue(from)));
  const hi = Math.max(area.left, Math.min(area.right, x.getPixelForValue(to)));
  if (hi - lo < 0.5) return false;
  const ctx = chart.ctx;
  ctx.save();
  ctx.fillStyle = `rgba(${BP_SHADE},${alpha})`;
  ctx.fillRect(lo, area.top, hi - lo, area.bottom - area.top);
  ctx.restore();
  return true;
}

const bpLiveShade = {
  id: 'bpLiveShade',
  beforeDatasetsDraw(chart, args, opts) {
    const bands = (opts && opts.bands) || [];
    if (!bands.length) return;
    for (const b of bands) {
      if (!b.partial) continue;
      bpFillBand(chart, b.from, b.to, BP_SHADE_STRATEGY);
    }
  },
};

/* Portfolio capital: the SUM of each strategy's planned capital, and each
 * strategy's is capital-per-position x qty — qty multiplies the position, so
 * it multiplies the capital behind it as well as the P/L it produces. */
function bpTotalCapital(rows) {
  let total = 0;
  for (const r of rows) {
    const cap = Number(r.capital), qty = Number(r.qty);
    if (isFinite(cap) && cap > 0 && isFinite(qty) && qty > 0) total += cap * qty;
  }
  return total;
}

/* The filter specs for one strategy, in the shape obApplyFilters takes.
 * Built from the SHARED REGISTRY — the old Dash app kept five parallel dicts
 * of metrics and this page keeps none. A filter that is off contributes
 * nothing, so an untouched page filters nothing. */
function bpSpecs(filters, registry, dateSpan) {
  const specs = [];
  for (const m of registry) {
    const f = filters[m.key];
    if (!f || !f.on) continue;
    if (m.type === 'range') {
      const lo = Number(f.lo), hi = Number(f.hi);
      if (!isFinite(lo) || !isFinite(hi)) continue;
      specs.push({ kind: 'range', column: m.column, lo, hi });
    } else if (m.type === 'categorical') {
      if (!f.allowed || !f.allowed.length) continue;
      specs.push({ kind: 'set', column: m.column, allowed: new Set(f.allowed) });
    }
  }
  // THE PORTFOLIO'S DATE RANGE, on the ENTRY date, as the old app had it: a
  // trade belongs to the window it was opened in.
  if (dateSpan && (dateSpan.start || dateSpan.end)) {
    specs.push({ kind: 'date', column: 'date_opened',
                 from: dateSpan.start, to: dateSpan.end });
  }
  return specs;
}

/* A slider step for a metric whose scale we do not know in advance.
 * Surface metrics range from 0.0001 to hundreds depending on the family, so
 * the step comes from the DATA's own span rather than a registry field that
 * does not exist for them. ~100 notches, snapped to a 1/2/2.5/5 decade so
 * the readout has a sane number of decimals. */
function bpNiceStep(span) {
  if (!(span > 0) || !isFinite(span)) return 0.01;
  const raw = span / 100;
  const mag = Math.pow(10, Math.floor(Math.log10(raw)));
  const mult = [1, 2, 2.5, 5, 10].find(m => raw <= m * mag) || 10;
  return Number((mult * mag).toPrecision(12));
}

/* FILTERS THAT CANNOT SEE THE WHOLE HISTORY.
 *
 * A filter either judges a trade or is blind to it, and the two are not the
 * same thing. Day of Week judges every trade ever: a Tuesday excluded by a
 * Mon/Wed/Fri filter is genuinely gone. VIX can only judge from the day
 * index_ohlc starts, and a surface z-score from the day that metric starts;
 * before then there is nothing to judge by, and dropping those trades
 * silently shortened the equity curve to wherever the metric began while the
 * date-range card went on advertising the full span.
 *
 * `minDate` is the registry's own coverage date, filled at request time from
 * the data — never a constant here. A metric without one is always
 * evaluable.
 */
function bpLenientColumns(filters, registry) {
  const out = new Map();
  for (const m of registry) {
    const f = filters && filters[m.key];
    if (f && f.on && m.minDate) out.set(m.column, m.minDate);
  }
  return out;
}

/* Trades an active blind-able filter could not judge even though its metric
 * HAD data by then -- no bar at the entry time. They are scattered through
 * the series rather than forming a stretch, so they are dropped exactly as
 * the table drops them; this counts them so the page can say where they
 * went instead of leaving a gap between the table and the shaded stretch. */
function bpNoBarCount(cols, n, lenient) {
  if (!lenient.size) return 0;
  const entry = cols.date_opened || [];
  let out = 0;
  for (let i = 0; i < n; i++) {
    for (const [column, from] of lenient) {
      const col = cols[column];
      if (!obNull(col ? col[i] : null)) continue;
      if (entry[i] && entry[i] >= from) { out++; break; }
    }
  }
  return out;
}

/* The date from which every active blind-able filter can finally be
 * evaluated: the LATEST coverage among them, not the earliest. With VIX from
 * 2017 and a z-score from 2021 both on, nothing before 2021 has passed both,
 * so nothing before 2021 is fully filtered. */
function bpUnfilteredTo(chosen, registry) {
  let to = null;
  for (const c of chosen || []) {
    // A surface metric has a coverage date like any other, and it is the
    // one most likely to set the boundary -- z-scores start years late.
    for (const m of [...registry, ...(c.surface || [])]) {
      const f = c.filters && c.filters[m.key];
      if (!f || !f.on || !m.minDate) continue;
      if (!to || m.minDate > to) to = m.minDate;
    }
  }
  return to;
}

/* The same grey as the strategy shade, denser. Not a texture and not a
 * second hue: on these pages blue and pink carry profit and loss everywhere,
 * and anything else reads as a new kind of information rather than as
 * "nothing is being asserted about this stretch". */
const bpUnfilteredShade = {
  id: 'bpUnfilteredShade',
  beforeDatasetsDraw(chart, args, opts) {
    const to = opts && opts.to;
    const x = chart.scales && chart.scales.x;
    if (to != null && x) {
      bpFillBand(chart, x.min, to, BP_SHADE_METRIC);
    }
  },
};

/* What an active filter costs in trades that have no value for it.
 * Metrics have staggered coverage — a filter on one that starts late drops
 * every earlier trade silently — so the page states it, as the single-
 * backtest page does. */
function bpCoverageCost(cols, n, filters, registry) {
  let noValue = 0;
  const cols_ = [];
  for (const m of registry) {
    const f = filters[m.key];
    if (f && f.on && cols[m.column]) cols_.push(m.column);
  }
  if (!cols_.length) return { noValue: 0, columns: [] };
  for (let i = 0; i < n; i++) {
    for (const c of cols_) {
      const v = cols[c][i];
      if (v === null || v === undefined || (typeof v === 'number' && Number.isNaN(v))) {
        noValue++;
        break;
      }
    }
  }
  return { noValue, columns: cols_ };
}


document.addEventListener('alpine:init', () => {
  Alpine.data('backtestPortfolio', () => ({

    // ── state ───────────────────────────────────────────────────────────
    saved: [],
    registry: [],
    surfPick: '',
    /* The surface catalog, fetched once and lazily: 452 metrics and a ~4s
     * index walk on the VPS, so a portfolio nobody filters never pays it. */
    surf: { catalog: null, groups: [], other: null, unitFormats: {},
            defaultUnit: null, loading: false, error: '' },
    colors: [],
    maxStrategies: 12,
    chosen: [],           // [{id, name, qty, capital, savedCapital, filters}]
    loaded: [],
    rows: [],             // the summary table: one per strategy, then TOTAL
    pick: 0,
    editing: 0,           // which strategy's filters are open
    busy: false,
    error: '',
    loadNote: '',
    slowLoad: false,
    rangeMode: 'union',
    tick: 0,              // bumped whenever the rows are recomputed
    // The monthly grid is small and IS read by the template, so unlike the
    // curves it lives in reactive state.
    months: { years: [], labels: [], cells: {}, totals: {}, max: 0 },
    deploy: { peak: 0, peakDay: null, sessions: 0 },
    // P4. Small enough to be reactive; the weekly series they are built from
    // is not, and lives in BP_DATA.
    corr: { names: [], matrix: [], pairs: [], weeks: 0, metrics: [] },
    pairA: 0, pairB: 0,
    rollWeeks: 26,
    riskWindow: 90,
    risk: { days: 0, window: 90, fits: true },
    profiles: [],
    profilePick: 0,
    profileName: '',
    profileMsg: '',
    profileClash: null,

    async init() {
      await Promise.all([this.loadSaved(), this.loadRegistry(),
                         this.loadProfiles()]);
    },

    // ── saved profiles ──────────────────────────────────────────────────
    //
    // A profile is a POINTER at saved strategies plus the numbers applied to
    // them. It holds no trades, so the strategy on the other page stays the
    // one source of its file — and a strategy deleted since is reported by
    // name on load rather than quietly leaving a smaller portfolio.
    async loadProfiles() {
      try {
        const r = await fetch('/api/backtest-portfolio/profiles');
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        this.profiles = (await r.json()).profiles || [];
      } catch (e) {
        this.profileMsg = 'Could not read the profiles: ' + e;
      }
    },

    profileLabel(pf) {
      const n = pf.n_strategies;
      return `${pf.name} · ${n} strateg${n === 1 ? 'y' : 'ies'}`;
    },

    profileSub() {
      return this.profiles.length ? `${this.profiles.length} saved` : 'none saved';
    },

    /* What the page IS, in the shape the store takes. Built from the live
     * rows rather than from anything cached, so what is saved is what is on
     * screen. */
    profilePayload() {
      return {
        strategies: this.chosen.map(c => ({
          id: c.id, qty: c.qty, capital: c.capital, filters: c.filters,
        })),
        range_mode: this.rangeMode,
        roll_weeks: this.rollWeeks,
      };
    },

    async saveProfile(replace) {
      const name = (this.profileName || '').trim();
      if (!name) { this.profileMsg = 'Give the combination a name first.'; return; }
      this.busy = true;
      this.profileMsg = '';
      this.profileClash = null;
      try {
        const r = await fetch('/api/backtest-portfolio/profiles', {
          method: 'POST', headers: { 'content-type': 'application/json' },
          body: JSON.stringify({ name, replace: !!replace,
                                 payload: this.profilePayload() }),
        });
        const b = await r.json();
        if (r.status === 409) {
          // The name exists. Offer to replace THAT one rather than asking
          // for a new name the page already knows is taken.
          this.profileClash = b.detail || { detail: 'That name is taken.' };
          return;
        }
        if (!r.ok) throw new Error(b.detail || `HTTP ${r.status}`);
        await this.loadProfiles();
        this.profilePick = b.profile.id;
        this.profileMsg = `Saved "${b.profile.name}".`;
      } catch (e) {
        this.profileMsg = 'Not saved: ' + (e.message || e);
      } finally {
        this.busy = false;
      }
    },

    async loadProfile() {
      if (!this.profilePick) return;
      this.busy = true;
      this.profileMsg = '';
      let ok = false;
      try {
        const r = await fetch(`/api/backtest-portfolio/profiles/${this.profilePick}`);
        const b = await r.json();
        if (!r.ok) throw new Error(b.detail || `HTTP ${r.status}`);
        const pf = b.profile;
        const missing = new Set(pf.missing || []);
        this.chosen = pf.payload.strategies
          .filter(s => !missing.has(s.id))
          .map(s => ({
            id: s.id, name: (pf.names || {})[String(s.id)] || `#${s.id}`,
            qty: s.qty, capital: s.capital,
            savedCapital: null, filters: s.filters || {},
          }));
        for (const c of this.chosen) this.ensureFilters(c.id);
        this.rangeMode = pf.payload.range_mode || 'union';
        this.rollWeeks = pf.payload.roll_weeks || 26;
        this.profileName = pf.name;
        this.editing = 0;
        this.profileMsg = missing.size
          ? `Loaded "${pf.name}" — ${missing.size} strateg`
            + `${missing.size === 1 ? 'y has' : 'ies have'} been deleted since `
            + `and could not be loaded.`
          : `Loaded "${pf.name}".`;
        ok = true;
      } catch (e) {
        this.profileMsg = 'Not loaded: ' + (e.message || e);
      } finally {
        // RELEASED BEFORE THE LOAD. `load()` refuses to run while `busy` is
        // set -- it is the guard against a second click -- so fetching the
        // trades from inside this busy window restored every setting and
        // then quietly fetched nothing. The profile read is finished here;
        // the trade load takes the flag again on its own.
        this.busy = false;
      }
      if (ok) await this.load();
    },

    async deleteProfile() {
      const pf = this.profiles.find(x => x.id === this.profilePick);
      if (!pf) return;
      this.busy = true;
      try {
        const r = await fetch(`/api/backtest-portfolio/profiles/${pf.id}`,
                              { method: 'DELETE' });
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        this.profilePick = 0;
        await this.loadProfiles();
        this.profileMsg = `Deleted "${pf.name}".`;
      } catch (e) {
        this.profileMsg = 'Not deleted: ' + (e.message || e);
      } finally {
        this.busy = false;
      }
    },

    async loadSaved() {
      try {
        const r = await fetch('/api/backtest-portfolio/strategies');
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        const b = await r.json();
        this.saved = b.strategies || [];
        this.colors = b.colors || [];
        this.maxStrategies = b.max || 12;
      } catch (e) {
        this.error = 'Could not read the saved strategies: ' + e;
      }
    },

    async loadRegistry() {
      try {
        const r = await fetch('/api/backtest-portfolio/registry');
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        const b = await r.json();
        // THE SAME REGISTRY THE OTHER PAGE FILTERS ON, filtered to what can
        // be screened. `year` is a section, not a filter.
        this.registry = (b.metrics || []).filter(m => m.filter);
      } catch (e) {
        this.error = 'Could not read the metric registry: ' + e;
      }
    },

    // ── choosing ────────────────────────────────────────────────────────
    addable() {
      const have = new Set(this.chosen.map(c => c.id));
      return this.saved.filter(s => !have.has(s.id));
    },

    optionLabel(s) {
      const n = s.trade_count == null ? '' : ` · ${bpFmtInt(s.trade_count)} trades`;
      const d = s.date_min && s.date_max ? ` · ${s.date_min} → ${s.date_max}` : '';
      return `${s.name}${n}${d}`;
    },

    add() {
      const s = this.saved.find(x => x.id === this.pick);
      if (!s) return;
      if (this.chosen.length >= this.maxStrategies) {
        this.error = `${this.maxStrategies} strategies is the most this page loads at once.`;
        return;
      }
      this.chosen.push({
        id: s.id, name: s.name, qty: 1,
        // The saved value seeds the row: a strategy saved with $25,000 a
        // position must not silently become this page's default.
        capital: s.capital_per_position != null ? s.capital_per_position : 10000,
        savedCapital: s.capital_per_position,
        filters: this.blankFilters(),
      });
      this.pick = 0;
      this.error = '';
    },

    blankFilters() {
      const out = {};
      for (const m of this.registry) {
        out[m.key] = (m.type === 'range')
          ? { on: false, lo: m.min, hi: m.max }
          : { on: false, allowed: [] };
      }
      return out;
    },

    remove(i) {
      const c = this.chosen[i];
      this.chosen.splice(i, 1);
      if (c) {
        delete BP_DATA.payloads[c.id];
        delete BP_DATA.scaled[c.id];
        this.loaded = this.loaded.filter(p => p.saved.id !== c.id);
        if (this.editing === c.id) this.editing = 0;
      }
      this.recompute();
    },

    clearAll() {
      this.chosen = [];
      this.loaded = [];
      this.rows = [];
      BP_DATA.payloads = {};
      BP_DATA.scaled = {};
      BP_DATA.extents = {};
      BP_DATA.curves = {};
      BP_DATA.weekly = null;
      this.corr = { names: [], matrix: [], pairs: [], weeks: 0, metrics: [] };
      this.months = { years: [], labels: [], cells: {}, totals: {}, max: 0 };
      this.loadNote = '';
      this.error = '';
      this.editing = 0;
    },

    normalise(c) {
      const q = Math.round(Number(c.qty));
      c.qty = (isFinite(q) && q > 0) ? q : 1;
      const cap = Number(c.capital);
      c.capital = (isFinite(cap) && cap >= 0) ? cap : 0;
      this.recompute();
    },

    /* THE REGISTRY THIS STRATEGY FILTERS ON: the shared one plus whatever
     * surface metrics were added to IT. Per strategy on purpose -- adding a
     * metric to one row must not put a slider on every other. */
    registryFor(c) {
      return (c && c.surface && c.surface.length)
        ? [...this.registry, ...c.surface] : this.registry;
    },

    async loadSurfaceCatalog() {
      if (this.surf.catalog || this.surf.loading) return;
      this.surf.loading = true;
      this.surf.error = '';
      try {
        // THE OO PAGE'S ENDPOINT, not a second one. The catalog is cached in
        // that process on the table's own key, so this is a cheap call.
        const r = await fetch('/api/oo-backtest/surface/catalog');
        const b = await r.json();
        if (!r.ok) throw new Error(b.detail || `HTTP ${r.status}`);
        this.surf.catalog = b.metrics || [];
        this.surf.groups = b.family_groups || [];
        this.surf.other = b.other_group || { label: 'Other' };
        this.surf.unitFormats = b.unit_formats || {};
        this.surf.defaultUnit = b.default_unit_format
          || { scale: 1, decimals: 4, suffix: '' };
      } catch (e) {
        this.surf.error = 'Surface metrics unavailable: ' + (e.message || e);
      } finally {
        this.surf.loading = false;
        this.tick++;
      }
    },

    /* One optgroup per family, in the legend's order, exactly as the OO page
     * groups them -- the display names come from the server so neither page
     * names a family itself. */
    surfaceOptions() {
      void this.tick;
      const cat = this.surf.catalog || [];
      if (!cat.length) return [];
      const fams = new Set(cat.map(m => m.family));
      const groups = (this.surf.groups || [])
        .map(g => ({ ...g, families: (g.families || []).filter(f => fams.has(f)) }))
        .filter(g => g.families.length);
      const seen = new Set(groups.flatMap(g => g.families));
      const rest = [...fams].filter(f => !seen.has(f)).sort();
      if (rest.length) groups.push({ ...(this.surf.other || { label: 'Other' }), families: rest });
      const out = [];
      for (const g of groups) {
        for (const f of g.families) {
          const ms = cat.filter(m => m.family === f)
            .sort((a, b) => (a.column_name < b.column_name ? -1 : 1));
          out.push({ label: `${g.label} · ${f}`, options: ms.map(m => ({
            value: m.column_name,
            label: m.column_name + (m.description ? ' — ' + String(m.description).slice(0, 60) : ''),
          })) });
        }
      }
      return out;
    },

    async addSurfaceMetric(id, column) {
      if (!column) return;
      const c = this.chosen.find(x => x.id === id);
      const meta = (this.surf.catalog || []).find(m => m.column_name === column);
      if (!c || !meta) return;
      if (!c.surface) c.surface = [];
      if (c.surface.some(m => m.surfColumn === column)) return;
      const u = this.surf.unitFormats[meta.units] || this.surf.defaultUnit
             || { scale: 1, decimals: 4, suffix: '' };
      const entry = {
        key: `surface__${column}`, column: `surface__${column}`,
        surfColumn: column, label: column, type: 'range', filter: true,
        categories: null, min: null, max: null, step: null,
        // The object form of `format`; fmtVal branches on it. Surface values
        // arrive in the metric's own units and are scaled to display units
        // once, here, so everything downstream reads display units.
        format: { decimals: u.decimals, suffix: u.suffix }, scale: u.scale,
        minDate: meta.min_date || null, description: meta.description || '',
        family: meta.family, removable: true, loading: true, error: '',
      };
      c.surface = [...c.surface, entry];
      this.tick++;
      await this.fetchSurface(c, entry);
    },

    removeSurfaceMetric(id, key) {
      const c = this.chosen.find(x => x.id === id);
      if (!c || !c.surface) return;
      c.surface = c.surface.filter(m => m.key !== key);
      if (c.filters) delete c.filters[key];
      const p = BP_DATA.payloads[c.id];
      if (p && p.columns) delete p.columns[key];
      delete BP_DATA.extents[c.id + ':' + key];
      delete BP_DATA.scaled[c.id];
      this.recompute();
    },

    /* Values for EVERY trade of this strategy, cached per (strategy, metric)
     * so removing and re-adding costs nothing. The server is asked for the
     * whole log, not the filtered subset, so a later filter change is free. */
    async fetchSurface(c, entry) {
      const p = BP_DATA.payloads[c.id];
      if (!p || !p.columns) {
        entry.loading = false;
        entry.error = 'load the portfolio first';
        this.tick++;
        return;
      }
      const cacheKey = c.id + '|' + entry.surfColumn;
      const token = BP_DATA.loadToken;
      const hit = BP_DATA.surface[cacheKey];
      if (hit && hit.n === p.n) {
        p.columns[entry.column] = hit.values;
        this.finishSurface(c, entry, hit.report);
        return;
      }
      const cols = p.columns;
      const trades = cols.date_opened.map((d, i) => [d, (cols.time_opened || [])[i] ?? null]);
      try {
        const r = await fetch('/api/oo-backtest/surface/values', {
          method: 'POST', headers: { 'content-type': 'application/json' },
          body: JSON.stringify({ column: entry.surfColumn, trades }),
        });
        const b = await r.json().catch(() => ({}));
        if (!r.ok) throw new Error(b.detail || `HTTP ${r.status}`);
        // A LOAD SINCE THIS WAS ASKED FOR, or the row removed meanwhile:
        // writing now would put a column on the wrong trades.
        if (token !== BP_DATA.loadToken || !(c.surface || []).includes(entry)) return;
        if (!Array.isArray(b.values) || b.values.length !== p.n) {
          throw new Error(`server returned ${(b.values || []).length} values `
                        + `for ${p.n} trades`);
        }
        const vals = b.values.map(v => (v === null || v === undefined ? null : v * entry.scale));
        p.columns[entry.column] = vals;
        BP_DATA.surface[cacheKey] = { values: vals, n: p.n, report: b.report || {} };
        this.finishSurface(c, entry, b.report || {});
      } catch (e) {
        if (token !== BP_DATA.loadToken) return;
        entry.loading = false;
        entry.error = String(e.message || e);
        this.tick++;
      }
    },

    finishSurface(c, entry, report) {
      const p = BP_DATA.payloads[c.id];
      const ext = obExtent((p.columns || {})[entry.column] || []);
      entry.loading = false;
      entry.error = '';
      entry.report = report || {};
      if (ext) {
        entry.step = bpNiceStep(ext.max - ext.min);
        entry.min = Math.min(obClampStep(ext.min, entry.step, 'floor'), ext.min);
        entry.max = Math.max(obClampStep(ext.max, entry.step, 'ceil'), ext.max);
      }
      delete BP_DATA.extents[c.id + ':' + entry.key];
      // scaledCols CACHES A COPY of the columns, so a column added after it
      // was built is invisible to every filter -- and a range spec drops
      // nulls, so the filter did not narrow, it emptied the strategy.
      delete BP_DATA.scaled[c.id];
      if (!c.filters) c.filters = {};
      if (!c.filters[entry.key]) {
        c.filters[entry.key] = { on: false, lo: entry.min, hi: entry.max };
      }
      this.recompute();
    },

    /* WHAT FILTERING ON THIS WOULD COST, for THIS strategy, split the way
     * the single-backtest page splits it: trades that entered before the
     * metric's data starts, and trades whose entry had no bar. The first is
     * a stretch of history, the second is scattered -- and a filter drops
     * both, which is why it is said before the slider moves and not after. */
    surfaceCost(c, m) {
      void this.tick;
      const p = c && BP_DATA.payloads[c.id];
      const vals = p && p.columns && p.columns[m.column];
      if (!vals) return null;
      const dates = p.columns.date_opened;
      let before = 0, noBar = 0;
      for (let i = 0; i < vals.length; i++) {
        if (vals[i] !== null && vals[i] !== undefined) continue;
        if (m.minDate && dates[i] && dates[i] < m.minDate) before++; else noBar++;
      }
      return { n: vals.length, dropped: before + noBar, before, noBar };
    },

    surfaceCostText(c, m) {
      const k = this.surfaceCost(c, m);
      if (!k) return '';
      if (!k.dropped) return 'every trade in this strategy has a value';
      const why = [
        k.before ? `${bpFmtInt(k.before)} entered before its data starts (${m.minDate})` : '',
        k.noBar ? `${bpFmtInt(k.noBar)} with no bar at the entry time` : '',
      ].filter(Boolean).join(', ');
      const verb = (c.filters[m.key] || {}).on ? 'is dropping' : 'would drop';
      return `Filtering on this ${verb} ${bpFmtInt(k.dropped)} of `
           + `${bpFmtInt(k.n)} trades — ${why}`;
    },

    colorOf(i) {
      return this.colors.length ? this.colors[i % this.colors.length] : '#3498db';
    },

    // ── loading ─────────────────────────────────────────────────────────
    async load() {
      if (!this.chosen.length || this.busy) return;
      this.busy = true;
      this.error = '';
      this.loadNote = '';
      const t0 = performance.now();
      try {
        const r = await fetch('/api/backtest-portfolio/load', {
          method: 'POST', headers: { 'content-type': 'application/json' },
          body: JSON.stringify({ ids: this.chosen.map(c => c.id) }),
        });
        const b = await r.json();
        if (!r.ok) throw new Error(b.detail || `HTTP ${r.status}`);
        this.loaded = b.strategies || [];
        BP_DATA.payloads = {};
        BP_DATA.scaled = {};
        BP_DATA.extents = {};
        // A LOAD REPLACES THE COLUMNS, so any surface metric added before it
        // is now pointing at nothing, and any request still in flight is for
        // the previous trades. Bump the token, drop the cache (the file
        // behind a saved strategy may have been re-saved), and refetch.
        BP_DATA.loadToken++;
        BP_DATA.surface = {};
        for (const p of this.loaded) BP_DATA.payloads[p.saved.id] = p;
        const wall = (performance.now() - t0) / 1000;
        const srv = b.load || {};
        // BOTH NUMBERS: the server's time is the work, the wall time includes
        // sending several megabytes of trades here. When they differ a lot it
        // is the transfer that is slow, not the parse.
        this.loadNote = `${srv.n} loaded in ${bpFmtSeconds(srv.seconds)} `
                      + `(${srv.from_cache} cached, ${srv.parsed} parsed) · `
                      + `${bpFmtSeconds(wall)} in the browser`;
        this.slowLoad = (srv.parsed || 0) > 0;
        this.recompute();
        for (const c of this.chosen) {
          for (const m of (c.surface || [])) {
            m.loading = true;
            m.error = '';
            this.fetchSurface(c, m);
          }
        }
      } catch (e) {
        this.error = 'Load failed: ' + (e.message || e);
        this.loaded = [];
        this.rows = [];
      } finally {
        this.busy = false;
      }
    },

    // ── filters, in the main column ─────────────────────────────────────
    //
    // ONE STRATEGY AT A TIME. Every strategy's panel open at once would be
    // nine metrics times however many strategies on a page whose job is the
    // comparison between them; the sidebar row says which is being edited
    // and how many filters it carries.
    /* NOT GATED ON THE PORTFOLIO BEING LOADED. It was, with
     * `:disabled="!loaded.length"` on the button, and that is a regression
     * the user hit: before the panel moved out of the sidebar it opened
     * whenever a strategy was in the list, and a disabled button that keeps
     * its text colour looks exactly like a working one that does nothing.
     * Filters set before a load are applied by the load, so there is
     * nothing to protect; what the panel cannot do yet is show RANGES, and
     * it says which of the two it is. */
    toggleEdit(id) {
      this.ensureFilters(id);
      this.loadSurfaceCatalog();
      this.editing = (this.editing === id ? 0 : id);
    },

    /* Fill in any metric the registry has that this strategy's filter map
     * does not. `add()` builds the map from the registry as it stands, so a
     * strategy added before /registry answered -- or after a metric is added
     * upstream -- would otherwise have a hole, and the panel reads
     * `c.filters[m.key].on` straight into a TypeError. */
    ensureFilters(id) {
      const c = this.chosen.find(x => x.id === id);
      if (!c) return;
      if (!c.filters) c.filters = {};
      for (const m of this.registryFor(c)) {
        if (c.filters[m.key]) continue;
        c.filters[m.key] = (m.type === 'range')
          ? { on: false, lo: m.min, hi: m.max }
          : { on: false, allowed: [] };
      }
    },

    editingRow() { return this.chosen.find(c => c.id === this.editing) || null; },

    editingName() {
      const c = this.editingRow();
      return c ? c.name : '';
    },

    editingColor() {
      const i = this.chosen.findIndex(c => c.id === this.editing);
      return i < 0 ? 'transparent' : this.colorOf(i);
    },

    editingSummary() {
      const c = this.editingRow();
      if (!c) return '';
      const r = this.rows.find(x => x.id === c.id);
      return r ? `${bpFmtInt(r.n)} of ${bpFmtInt(r.nAll)} trades` : '';
    },

    editingCost() {
      const c = this.editingRow();
      const r = c && this.rows.find(x => x.id === c.id);
      if (!r || !r.cost.noValue) return '';
      return `${bpFmtInt(r.cost.noValue)} trades have no value for an active `
           + `filter and are dropped by it — metrics start at different dates.`;
    },

    isOn(m) {
      const c = this.editingRow();
      return !!(c && c.filters[m.key] && c.filters[m.key].on);
    },

    setOn(m, on) {
      const c = this.editingRow();
      if (!c) return;
      const f = c.filters[m.key];
      f.on = !!on;
      // SWITCHING A CATEGORICAL ON KEEPS EVERYTHING, then you untick what
      // you do not want. On with nothing chosen is INERT -- bpSpecs skips a
      // set filter with no members, matching the old app -- so the cell
      // would read as an active filter that changes nothing, which is the
      // failure the scalp page's inert-filter line exists to prevent.
      if (f.on && m.type === 'categorical' && !f.allowed.length) {
        f.allowed = this.categoriesFor(c, m).map(x => x.value);
      }
      this.recompute();
    },

    stateOf(m) {
      const c = this.editingRow();
      if (!c) return '';
      const f = c.filters[m.key];
      if (!f || !f.on) return 'off';
      if (m.type === 'categorical') {
        const n = (f.allowed || []).length;
        const all = this.categoriesFor(c, m).length;
        // Nothing chosen does not filter (see setOn), and says so rather
        // than implying it keeps nothing.
        if (!n) return 'nothing chosen — not filtering';
        return n === all ? `all ${n} kept` : `${n} of ${all} kept`;
      }
      const r = this.rangeOf(m);
      return r.missing ? 'no values'
        : `${this.fmtVal(m, r.lo)} – ${this.fmtVal(m, r.hi)}`;
    },

    /* The slider's bounds come from THIS STRATEGY'S OWN VALUES, not from the
     * registry's nominal range: a VIX slider spanning 9–80 when the log only
     * ever saw 12–31 is a control whose useful travel is a third of its
     * length. The registry supplies the step and the formatting. */
    rangeOf(m) {
      // NEVER NULL. The panel binds :min, :max and :step from this, and
      // Alpine evaluates those bindings once more as the x-if around them is
      // torn down -- with the row already gone. Returning a shape with
      // `missing` set means no binding can dereference nothing; an
      // expression error there does not stop the page, it just leaves the
      // rest of that render pass stale, which is how a filter panel took the
      // summary table down with it.
      const blank = { min: 0, max: 1, step: 1, lo: 0, hi: 1, n: 0, missing: true };
      const c = this.editingRow();
      if (!c) return blank;
      const ext = this.extentOf(c, m);
      if (!ext) return blank;
      const f = c.filters[m.key];
      const lo = (f && isFinite(f.lo)) ? f.lo : ext.min;
      const hi = (f && isFinite(f.hi)) ? f.hi : ext.max;
      return { missing: false, min: ext.min, max: ext.max, step: m.step || 0.01,
               lo: Math.max(ext.min, Math.min(lo, ext.max)),
               hi: Math.max(ext.min, Math.min(hi, ext.max)), n: ext.n };
    },

    extentOf(c, m) {
      const p = BP_DATA.payloads[c.id];
      if (!p) return null;
      const key = c.id + ':' + m.key;
      if (!(key in BP_DATA.extents)) {
        const ext = obExtent(p.columns[m.column] || []);
        // SNAPPED OUTWARD, AND NEVER INSIDE THE REAL EXTENT. obClampStep
        // rounds through toPrecision(12), so a ceil can land a float's
        // breadth BELOW the true maximum -- and a slider parked at its own
        // maximum then drops the very trades that set it. Seen as two
        // trades vanishing from a full-range filter.
        BP_DATA.extents[key] = ext && {
          min: Math.min(obClampStep(ext.min, m.step, 'floor'), ext.min),
          max: Math.max(obClampStep(ext.max, m.step, 'ceil'), ext.max),
          n: ext.n };
      }
      return BP_DATA.extents[key];
    },

    setLo(m, v) {
      const c = this.editingRow(), r = this.rangeOf(m);
      if (!c || r.missing) return;
      const f = c.filters[m.key];
      f.lo = Math.min(Number(v), r.hi);
      f.on = true;
      this.recompute();
    },

    setHi(m, v) {
      const c = this.editingRow(), r = this.rangeOf(m);
      if (!c || r.missing) return;
      const f = c.filters[m.key];
      f.hi = Math.max(Number(v), r.lo);
      f.on = true;
      this.recompute();
    },

    fmtVal(m, v) {
      void this.tick;
      if (v == null || !isFinite(v)) return '—';
      if (m.format === 'usd') return bpFmtMoney(v);
      if (m.format === 'pct') return v.toFixed(2) + '%';
      if (m.format === 'ratio') return v.toFixed(2);
      if (m.format === 'int') return String(Math.round(v));
      // Surface metrics carry {decimals, suffix} instead of a name: their
      // scales run from 0.0001 to hundreds, so two decimals is not a
      // readout, it is a row of zeros.
      if (m.format && typeof m.format === 'object') {
        return v.toFixed(m.format.decimals ?? 4)
             + (m.format.suffix ? ' ' + m.format.suffix : '');
      }
      return v.toFixed(2);
    },

    /* Why a range metric has no slider: not loaded yet, or loaded and the
     * column is empty. Different problems, different fixes, and "no values
     * in this log" would be a lie before the load. */
    noValueNote(m) {
      const c = this.editingRow();
      if (c && !BP_DATA.payloads[c.id]) {
        return 'load the portfolio to see this strategy\'s values';
      }
      return `no values in this log (${m.column})`;
    },

    rangeSummary(m) {
      const r = this.rangeOf(m);
      if (r.missing) return '';
      return `${bpFmtInt(r.n)} trades have a value · `
           + `${this.fmtVal(m, r.min)} to ${this.fmtVal(m, r.max)} in this log`;
    },

    /* The categories a categorical metric actually has in THIS strategy's
     * trades — from the data, not from a list written here. `exit_reason`
     * has no declared categories because they are whatever the file says. */
    categoriesFor(c, m) {
      // NULL-SAFE because Alpine evaluates an x-for's expression once more
      // as the x-if around it is torn down -- with `editing` already 0, so
      // `editingRow()` is null. Closing the panel logged an expression
      // error every time; nothing broke, which is how it went unnoticed.
      if (m.categories && m.categories.length) return m.categories;
      const p = c && BP_DATA.payloads[c.id];
      if (!p) return [];
      return obDistinct(p.columns[m.column] || [])
        .map(v => ({ value: v, label: String(v) }));
    },

    toggleCategory(c, m, value) {
      if (!c) return;
      const f = c.filters[m.key];
      const i = f.allowed.indexOf(value);
      if (i >= 0) f.allowed.splice(i, 1); else f.allowed.push(value);
      // TICKING THE FIRST BOX TURNS THE FILTER ON; clearing the last leaves
      // it on with nothing kept, which filters everything out. That is a
      // real state a person can reach deliberately, and `stateOf` says
      // "none kept" rather than pretending the filter is off.
      if (f.allowed.length) f.on = true;
      this.recompute();
    },

    isChosenCategory(c, m, value) {
      const f = c && c.filters && c.filters[m.key];
      return !!(f && f.allowed.includes(value));
    },

    resetFilters(c) {
      c.filters = this.blankFilters();
      this.recompute();
    },

    activeCount(c) {
      let n = 0;
      for (const m of this.registryFor(c)) {
        const f = c.filters[m.key];
        if (f && f.on) n++;
      }
      return n;
    },

    /* What a card shows about its filters: one badge per ACTIVE filter,
     * worded as the old app worded them — "VIX3M/VIX Ratio: 0.7–1.1",
     * "Premium: 55.0–3480.0", "Day of Week: Fri". Without these you cannot
     * tell which of several strategies is filtered without opening each
     * panel in turn, which is the whole point of having them side by side.
     */
    badges(c) {
      void this.tick;
      const out = [];
      for (const m of this.registryFor(c)) {
        const f = c.filters && c.filters[m.key];
        if (!f || !f.on) continue;
        const [bg, fg] = BP_BADGE[m.key] || BP_BADGE_FALLBACK;
        out.push({ key: m.key, bg, fg, text: this.badgeText(c, m, f) });
      }
      return out;
    },

    badgeText(c, m, f) {
      if (m.type === 'categorical') {
        const all = this.categoriesFor(c, m);
        const chosen = f.allowed || [];
        // ALL OF THEM IS NOT A NARROWING, so the badge is just the label --
        // the old app's rule, and it keeps a row of badges meaningful.
        if (!chosen.length || chosen.length >= all.length) return m.label;
        const labels = chosen.map(v => {
          const hit = all.find(x => String(x.value) === String(v));
          return hit ? hit.label : String(v);
        });
        const shown = labels.slice(0, 2).join(',');
        return `${m.label}: ${shown}${labels.length > 2 ? '…' : ''}`;
      }
      const lo = Number(f.lo), hi = Number(f.hi);
      if (!isFinite(lo) || !isFinite(hi)) return m.label;
      return `${m.label}: ${lo.toFixed(1)}–${hi.toFixed(1)}`;
    },

    filterBadge(c) {
      const n = this.activeCount(c);
      return n ? `${n} filter${n === 1 ? '' : 's'}` : 'no filters';
    },

    // ── the numbers ─────────────────────────────────────────────────────
    span() { return bpSpan(this.loaded, this.rangeMode); },

    /* One strategy's columns with P/L scaled by qty. Cached per (id, qty):
     * the array is rebuilt when the quantity changes and not on every
     * recompute, which is every keystroke in a filter box. */
    scaledCols(c) {
      const p = BP_DATA.payloads[c.id];
      if (!p) return null;
      const hit = BP_DATA.scaled[c.id];
      if (hit && hit.qty === c.qty) return hit.cols;
      const qty = Number(c.qty) > 0 ? Number(c.qty) : 1;
      const cols = Object.assign({}, p.columns,
                                 { pnl: p.columns.pnl.map(v => v * qty) });
      BP_DATA.scaled[c.id] = { qty: c.qty, cols };
      return cols;
    },

    /* Everything the table shows. Recomputed in full on any change — the
     * whole portfolio is a few tens of thousands of trades and the work is
     * milliseconds, so there is no partial-update path to get wrong. */
    recompute() {
      const rows = [];
      const span = this.rangeMode === 'intersection' ? this.span() : null;
      // One session list for everyone: they all come from the same rollup,
      // so the union covers every strategy and the per-strategy deployed
      // series can be summed position by position.
      const sessions = [...new Set(this.loaded.flatMap(
        p => (p.market && p.market.spx_sessions) || []))].sort();

      const pooled = { date_opened: [], date_closed: [], pnl: [], days_in_trade: [],
                       // The metric columns ride along so the metric/P-L
                       // correlations below read the same filtered trades the
                       // table counted, rather than re-deriving them.
                       cols: {}, idx: [] };
      for (const m of this.registry) pooled.cols[m.column] = [];
      const pctParts = [];          // per-trade P/L %, each against its own capital
      let deployed = new Array(sessions.length).fill(0);
      // The CHART's series. Identical to `deployed` unless a filter is
      // blind to part of the history, in which case Capital deployed draws
      // the whole span like the two charts beside it. The TABLE's peak
      // stays on the strict set, so the two can differ and the card says so.
      let deployedDraw = new Array(sessions.length).fill(0);
      // THE CHARTS' OWN POOL. Same trades as `pooled` unless a filter is
      // blind to part of the history, in which case the equity and drawdown
      // charts keep drawing it and shade it instead of stopping short.
      const pooledDraw = { pnl: [], date_closed: [] };
      let unfilteredTo = null;      // the metric shade's right edge, by close
      let noBar = 0;                // dropped for no bar at entry, not shaded

      for (const c of this.chosen) {
        const p = BP_DATA.payloads[c.id];
        if (!p) continue;
        const cols = this.scaledCols(c);
        const reg = this.registryFor(c);
        const specs = bpSpecs(c.filters, reg, span);
        const idx = obApplyFilters(cols, p.n, specs);
        // The charts' set: identical to the table's unless some active
        // filter is blind before its coverage, in which case those trades
        // are kept and the stretch is shaded rather than cut off.
        const lenient = bpLenientColumns(c.filters, reg);
        const idxDraw = lenient.size
          ? obApplyFilters(cols, p.n, specs, lenient) : idx;
        noBar += bpNoBarCount(cols, p.n, lenient);
        const kept = lenient.size ? new Set(idx) : null;
        for (const i of idxDraw) {
          pooledDraw.pnl.push(cols.pnl[i]);
          pooledDraw.date_closed.push(p.columns.date_closed[i]);
          // THE HATCH ENDS AT THE LAST TRADE IT IS EXPLAINING, by CLOSE
          // date, because that is the axis these curves are drawn on.
          // Ending it at the metric's coverage date instead would leave a
          // trade entered before coverage but closed after it drawn
          // unfiltered OUTSIDE the shade -- mixing with nothing marking it,
          // which is the one thing this is here to prevent.
          if (kept && !kept.has(i)) {
            const d = p.columns.date_closed[i];
            if (d && (!unfilteredTo || d > unfilteredTo)) unfilteredTo = d;
          }
        }
        const stats = obStats(cols, idx);
        const conc = obConcurrency(p.columns, idx, sessions);
        const capital = (Number(c.capital) || 0) * (Number(c.qty) || 1);
        const extra = obExtraStats(cols, idx, stats, capital, conc.peak);
        const series = obDeployedSeries(conc, sessions, capital);
        for (let i = 0; i < deployed.length; i++) deployed[i] += series[i];
        const concDraw = lenient.size
          ? obConcurrency(p.columns, idxDraw, sessions) : conc;
        const seriesDraw = lenient.size
          ? obDeployedSeries(concDraw, sessions, capital) : series;
        for (let i = 0; i < deployedDraw.length; i++) deployedDraw[i] += seriesDraw[i];

        for (const i of idx) {
          pooled.idx.push(pooled.pnl.length);
          pooled.date_opened.push(p.columns.date_opened[i]);
          pooled.date_closed.push(p.columns.date_closed[i]);
          pooled.pnl.push(cols.pnl[i]);
          pooled.days_in_trade.push(p.columns.days_in_trade[i]);
          for (const m of this.registry) {
            const src = p.columns[m.column];
            pooled.cols[m.column].push(src ? src[i] : null);
          }
          if (capital > 0) pctParts.push(cols.pnl[i] / capital * 100);
        }

        rows.push({
          key: 'k' + c.id, id: c.id, name: c.name, color: p.color,
          total: false, idx, idxDraw,
          n: idx.length, nAll: p.n,
          dropped: p.n - idx.length,
          cost: bpCoverageCost(p.columns, p.n, c.filters, reg),
          stats, extra, conc,
          sharpe: obSharpe(cols, idx),
          capital,
          peakDeployed: conc.peak * capital,
        });
      }

      // ── the TOTAL row ────────────────────────────────────────────────
      if (rows.length) {
        const all = [...pooled.pnl.keys()];
        const tStats = obStats(pooled, all);
        // PEAK DEPLOYED FOR THE PORTFOLIO is the peak of the SUMMED series,
        // which is not the sum of the per-strategy peaks unless they all peak
        // on the same day. Passing it as `capital` with a peak of 1 is how
        // obExtraStats is told "this is already the denominator".
        const peakDeployed = deployed.length ? Math.max(...deployed) : 0;
        const tExtra = obExtraStats(pooled, all, tStats, peakDeployed, 1);
        // Avg P/L % pools PER-TRADE percentages, each against its own
        // strategy's capital: with one strategy that is exactly the single
        // page's avg P/L / capital, and with several it is the only reading
        // that does not need a "portfolio capital per position" that does
        // not exist.
        tExtra.avg_pnl_pct = pctParts.length
          ? pctParts.reduce((a, b) => a + b, 0) / pctParts.length : null;
        rows.push({
          key: 'total', id: 0, name: 'TOTAL', color: '#ffffff', total: true,
          n: all.length,
          nAll: rows.reduce((a, r) => a + r.nAll, 0),
          dropped: rows.reduce((a, r) => a + r.dropped, 0),
          cost: { noValue: rows.reduce((a, r) => a + r.cost.noValue, 0), columns: [] },
          stats: tStats, extra: tExtra, conc: null,
          sharpe: obSharpe(pooled, all),
          capital: bpTotalCapital(this.chosen),
          peakDeployed,
        });
      }
      this.rows = rows;

      // ── what the charts draw ─────────────────────────────────────────
      //
      // Built HERE, in the same pass that builds the table, from the same
      // filtered indices. A second pass that re-derived them could disagree
      // with the numbers above it, which is the one thing a chart beside a
      // table must not do.
      const curves = { eq: [], dd: [], cap: [], sessions };
      for (const r of rows) {
        if (r.total) continue;
        const c = this.chosen.find(x => x.id === r.id);
        const cols = c && this.scaledCols(c);
        if (!cols) continue;
        curves.eq.push({ name: r.name, color: r.color,
                         points: obEquity(cols, r.idxDraw).points });
      }
      if (pooledDraw.pnl.length) {
        const peq = obEquity(pooledDraw, [...pooledDraw.pnl.keys()]);
        const daily = peq.points;
        curves.eq.push({ name: 'TOTAL', color: BP_TOTAL, points: daily, total: true });
        curves.dd = daily;
        curves.maxDD = peq.maxDD;
      }
      // Where the metric shade stops: the latest coverage among every active
      // blind-able filter on any strategy. Null when nothing active is
      // blind, and then these charts are exactly what they always were.
      curves.unfilteredTo = unfilteredTo;
      curves.unfilteredN = pooledDraw.pnl.length - pooled.pnl.length;
      curves.noBarN = noBar;
      // Why, as opposed to how far: the latest coverage among the active
      // blind filters. Stated in the key; the edge itself is the data.
      curves.coverageFrom = bpUnfilteredTo(this.chosen, this.registry);
      curves.cap = deployedDraw;
      // The table's figure, kept so the card can say when the drawn peak
      // runs above it rather than leaving two numbers to be compared.
      curves.capPeakStrict = deployed.length ? Math.max(...deployed) : 0;
      curves.capPeakDrawn = deployedDraw.length ? Math.max(...deployedDraw) : 0;
      // P6 reads these: the portfolio's daily P/L (by close date, the old
      // app's series) and each strategy's own concurrency, which the overlap
      // chart draws beside the portfolio total.
      const dailyMap = obDailyPnl(pooled, pooled.idx);
      const dailyDates = [...dailyMap.keys()].sort();
      curves.daily = { dates: dailyDates,
                       values: dailyDates.map(d => dailyMap.get(d)) };
      curves.overlap = [];
      for (const r of rows) {
        if (r.total || !r.conc || !r.conc.days.length) continue;
        curves.overlap.push({ name: r.name, color: r.color, conc: r.conc });
      }
      curves.dist = [];
      for (const r of rows) {
        if (r.total) continue;
        const c = this.chosen.find(x => x.id === r.id);
        const cols = c && this.scaledCols(c);
        if (!cols) continue;
        curves.dist.push({ name: r.name, color: r.color,
                           pnl: r.idx.map(i => cols.pnl[i]) });
      }
      BP_DATA.curves = curves;

      const peakAt = deployed.indexOf(Math.max(...(deployed.length ? deployed : [0])));
      this.deploy = {
        peak: deployed.length ? Math.max(...deployed) : 0,
        peakDay: peakAt >= 0 ? sessions[peakAt] : null,
        sessions: sessions.length,
      };
      this.buildMonths(pooled);
      this.buildCorrelations(rows, pooled);
      this.tick++;
      // After the reactive state, so the template's cards exist to draw into
      // on the first pass.
      this.$nextTick(() => this.renderCharts());
    },

    /* P4: how the strategies move together, and what moves P/L.
     *
     * WEEKLY throughout -- see obAlignWeekly. Built from the same filtered
     * indices as the table, so a filter narrows the correlations too.
     */
    buildCorrelations(rows, pooled) {
      const live = rows.filter(r => !r.total && r.n > 0);
      const series = [];
      for (const r of live) {
        const c = this.chosen.find(x => x.id === r.id);
        const cols = c && this.scaledCols(c);
        series.push(cols ? obWeeklyPnl(cols, r.idx) : new Map());
      }
      const aligned = obAlignWeekly(series);
      BP_DATA.weekly = { weeks: aligned.weeks, cols: aligned.cols,
                         names: live.map(r => r.name),
                         colors: live.map(r => r.color) };

      // The matrix, and the pair list the scatter and the rolling chart read.
      const n = live.length;
      const matrix = [];
      const pairs = [];
      for (let i = 0; i < n; i++) {
        const row = [];
        for (let j = 0; j < n; j++) {
          const r = (i === j) ? 1 : obPearson(aligned.cols[i], aligned.cols[j]);
          row.push(r);
          if (j > i) pairs.push({ i, j, r, label: `${live[i].name} / ${live[j].name}` });
        }
        matrix.push(row);
      }
      if (this.pairA >= n || this.pairB >= n || this.pairA === this.pairB) {
        this.pairA = 0;
        this.pairB = n > 1 ? 1 : 0;
      }

      // METRIC vs P/L, Spearman, over the POOLED filtered trades. Pooling
      // mixes strategies that traded at different times, so this says what
      // the portfolio's P/L moved with -- not what any one strategy's did.
      // Stated under the table rather than left to be assumed.
      const metrics = [];
      for (const m of this.registry) {
        if (m.type !== 'range') continue;
        const xs = [], ys = [];
        for (const i of pooled.idx) {
          const v = pooled.cols[m.column] ? pooled.cols[m.column][i] : null;
          if (obNull(v)) continue;
          xs.push(v);
          ys.push(pooled.pnl[i]);
        }
        metrics.push({ key: m.key, label: m.label, n: xs.length,
                       rho: xs.length >= 10 ? obSpearman(xs, ys) : null });
      }
      metrics.sort((a, b) => Math.abs(b.rho || 0) - Math.abs(a.rho || 0));

      this.corr = { names: live.map(r => r.name), colors: live.map(r => r.color),
                    matrix, pairs, weeks: aligned.weeks.length, metrics };
    },

    /* The monthly grid: P/L by close month, years down, months across. */
    buildMonths(pooled) {
      const by = obMonthlyPnl(pooled, [...pooled.pnl.keys()]);
      const years = [...new Set([...by.keys()].map(k => k.slice(0, 4)))].sort();
      const cells = {}, totals = {};
      let max = 0;
      for (const [k, v] of by) {
        cells[k] = v;
        const y = k.slice(0, 4);
        totals[y] = (totals[y] || 0) + v;
        if (Math.abs(v) > max) max = Math.abs(v);
      }
      this.months = {
        years, cells, totals, max,
        labels: ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
      };
    },

    // ── the charts ──────────────────────────────────────────────────────
    //
    // Chart.js, the same build and the same conventions as the other page:
    // linear x in epoch days (no date adapter), thin lines, no point
    // markers, hover only. An existing chart is UPDATED rather than
    // rebuilt -- a new Chart on every filter change leaks canvases and
    // throws away the zoom.
    renderCharts() {
      if (typeof Chart === 'undefined') return;
      const c = BP_DATA.curves;
      if (!c || !c.eq) return;
      const axis = (money) => ({
        type: 'linear',
        grid: { color: 'rgba(255,255,255,0.05)' },
        border: { display: false },
        ticks: { color: '#9a9a9a', font: { size: 10 }, maxTicksLimit: money ? 6 : 7,
                 callback: money ? (v => bpFmtMoney(v)) : (v => obIsoDay(v).slice(0, 7)) },
      });
      const span = (pts) => (pts && pts.length
        ? { min: obDay(pts[0].date), max: obDay(pts[pts.length - 1].date) } : {});
      const base = (label, xr) => ({
        responsive: true, maintainAspectRatio: false, animation: false, parsing: false,
        interaction: { mode: 'nearest', axis: 'x', intersect: false },
        scales: { x: { ...axis(false), ...xr }, y: axis(true) },
        plugins: {
          legend: { display: false },
          tooltip: { callbacks: { title: it => obIsoDay(it[0].parsed.x), label } },
        },
      });

      // HOW MANY STRATEGIES ARE LIVE, shaded under the three DATE charts --
      // and only those three. The pairwise scatter's x is dollars and the
      // distribution's is a P/L bin, so a band drawn there would be nonsense
      // dressed as information.
      const shade = this.liveShade();
      const shaded = (o) => {
        o.plugins.bpLiveShade = { bands: shade.bands, total: shade.total };
        return o;
      };
      // The metric shade goes on the two charts that now draw the whole
      // span -- equity and drawdown. Capital deployed keeps the table's
      // trades, because it answers what would have been at risk UNDER the
      // filter, which is a different question.
      const unf = (o) => {
        o.plugins.bpUnfilteredShade = { to: c.unfilteredTo ? obDay(c.unfilteredTo) : null };
        return o;
      };

      // EQUITY: a line per strategy plus the portfolio, sharing an axis.
      const total = c.eq.find(s => s.total);
      const xr = span(total ? total.points : (c.eq[0] && c.eq[0].points));
      const eqData = { datasets: c.eq.map(sv => ({
        label: sv.name,
        data: sv.points.map(p => ({ x: obDay(p.date), y: p.cumulative })),
        borderColor: sv.color,
        borderWidth: sv.total ? 1.8 : 1,
        pointRadius: 0, pointHitRadius: 5, tension: 0,
        fill: sv.total ? 'origin' : false,
        backgroundColor: sv.total ? 'rgba(232,236,241,0.06)' : undefined,
        order: sv.total ? 0 : 1,
      })) };
      this.draw('eq', 'bp-eq-chart', eqData,
                unf(shaded(base(it => `${it.dataset.label}: ${bpFmtMoney(it.parsed.y)}`, xr))),
                [bpLiveShade, bpUnfilteredShade]);

      // DRAWDOWN: the portfolio's only, with its deepest point marked.
      const dd = c.dd || [];
      const mark = c.maxDD
        ? [{ x: obDay(c.maxDD.date), y: c.maxDD.drawdown }] : [];
      const ddData = { datasets: [
        { data: dd.map(p => ({ x: obDay(p.date), y: p.drawdown })),
          borderColor: BP_PINK, backgroundColor: 'rgba(232,67,147,0.16)',
          fill: 'origin', borderWidth: 1, pointRadius: 0, pointHitRadius: 5, tension: 0 },
        { data: mark, type: 'scatter', pointRadius: 5, pointHoverRadius: 6,
          pointBackgroundColor: BP_PINK, pointBorderColor: '#2d2d2d',
          pointBorderWidth: 2, showLine: false },
      ] };
      this.draw('dd', 'bp-dd-chart', ddData,
                unf(shaded(base(it => (it.datasetIndex === 1 ? 'Deepest: ' : 'Drawdown ')
                            + bpFmtMoney(it.parsed.y), xr))),
                [bpLiveShade, bpUnfilteredShade]);

      // CAPITAL DEPLOYED: a step per session, since it changes at a close.
      const cap = [];
      for (let i = 0; i < c.sessions.length; i++) {
        if (c.cap[i] || (i && c.cap[i - 1])) {
          cap.push({ x: obDay(c.sessions[i]), y: c.cap[i] });
        }
      }
      const capData = { datasets: [{
        data: cap, borderColor: BP_BLUE, backgroundColor: 'rgba(52,152,219,0.12)',
        fill: 'origin', borderWidth: 1, pointRadius: 0, pointHitRadius: 5,
        stepped: 'before' }] };
      // THE PAIRWISE SCATTER: one point a week, the two strategies' P/L.
      const w = BP_DATA.weekly;
      if (w && w.cols.length > 1) {
        const a = w.cols[this.pairA] || [], b = w.cols[this.pairB] || [];
        const pts = a.map((v, i) => ({ x: v, y: b[i] }));
        const money = {
          type: 'linear', grid: { color: 'rgba(255,255,255,0.05)' },
          border: { display: false },
          ticks: { color: '#9a9a9a', font: { size: 10 }, maxTicksLimit: 6,
                   callback: v => bpFmtMoney(v) },
        };
        this.draw('sc', 'bp-sc-chart', { datasets: [{
            type: 'scatter', data: pts, pointRadius: 3, pointHoverRadius: 5,
            backgroundColor: 'rgba(52,152,219,0.55)',
            borderColor: 'rgba(52,152,219,0.9)', borderWidth: 0.5 }] },
          { responsive: true, maintainAspectRatio: false, animation: false,
            parsing: false,
            scales: { x: money, y: money },
            plugins: { legend: { display: false }, tooltip: { callbacks: {
              title: () => '', label: it => `${w.names[this.pairA]} `
                + `${bpFmtMoney(it.parsed.x)} · ${w.names[this.pairB]} `
                + `${bpFmtMoney(it.parsed.y)}` } } } });

        // ROLLING PAIRWISE CORRELATION, every pair, over a window of WEEKS.
        const xs = w.weeks.map(d => obDay(d));
        const sets = [];
        for (const pr of this.corr.pairs) {
          const r = obRollingCorr(w.cols[pr.i], w.cols[pr.j], this.rollWeeks);
          sets.push({
            label: pr.label,
            data: r.map((v, i) => (v === null ? null : { x: xs[i], y: v })).filter(Boolean),
            borderColor: pr.i === this.pairA && pr.j === this.pairB
              ? BP_TOTAL : obRgbaFrom(w.colors[pr.i], 0.75),
            borderWidth: pr.i === this.pairA && pr.j === this.pairB ? 1.8 : 1,
            pointRadius: 0, pointHitRadius: 5, tension: 0, fill: false,
          });
        }
        this.draw('roll', 'bp-roll-chart', { datasets: sets },
          { responsive: true, maintainAspectRatio: false, animation: false,
            parsing: false,
            scales: {
              x: { type: 'linear', grid: { color: 'rgba(255,255,255,0.05)' },
                   border: { display: false },
                   ticks: { color: '#9a9a9a', font: { size: 10 },
                            maxTicksLimit: 7,
                            callback: v => obIsoDay(v).slice(0, 7) } },
              y: { type: 'linear', min: -1, max: 1,
                   grid: { color: 'rgba(255,255,255,0.05)' },
                   border: { display: false },
                   ticks: { color: '#9a9a9a', font: { size: 10 },
                            callback: v => v.toFixed(1) } },
            },
            plugins: { legend: { display: false }, tooltip: { callbacks: {
              title: it => obIsoDay(it[0].parsed.x),
              label: it => `${it.dataset.label}: ${it.parsed.y.toFixed(2)}` } } } });
      }

      // The annual bars are NOT drawn here. They are DOM cells in the
      // monthly grid's own rows (`yearBar`), because a canvas beside the
      // table kept its own vertical rhythm and a year's bar drifted off
      // that year's row of months.

      // ── P/L DISTRIBUTION: overlaid histograms, one per strategy ──────
      if (c.dist && c.dist.length) {
        // One bin set for everyone, or the bars do not line up and
        // "overlaid" becomes "interleaved".
        const all = c.dist.flatMap(d => d.pnl);
        const base = obHistogram(all, BP_DIST_BIN);
        const at = new Map(base.edges.map((e, i) => [e, i]));
        const sets = c.dist.map(d => {
          const h = obHistogram(d.pnl, BP_DIST_BIN);
          const counts = new Array(base.edges.length).fill(0);
          h.edges.forEach((e, i) => {
            const k = at.get(e);
            if (k !== undefined) counts[k] = h.counts[i];
          });
          return { label: d.name, data: counts,
                   backgroundColor: obRgbaFrom(d.color, 0.6),
                   borderWidth: 0, grouped: false };
        });
        this.drawBar('dist', 'bp-dist-chart',
          { labels: base.edges.map(e => bpFmtMoney(e)), datasets: sets },
          { responsive: true, maintainAspectRatio: false, animation: false,
            scales: {
              x: { grid: { display: false }, border: { display: false },
                   ticks: { color: '#9a9a9a', font: { size: 10 },
                            maxTicksLimit: 9, autoSkip: true } },
              y: { grid: { color: 'rgba(255,255,255,0.05)' },
                   border: { display: false },
                   ticks: { color: '#9a9a9a', font: { size: 10 },
                            maxTicksLimit: 5, precision: 0 } },
            },
            plugins: { legend: { display: false }, tooltip: { callbacks: {
              title: it => `${it[0].label} to `
                + bpFmtMoney(base.edges[it[0].dataIndex] + BP_DIST_BIN),
              label: it => `${it.dataset.label}: ${it.parsed.y} trades` } } } });
      }

      // ── STRATEGIES ACTIVE PER DAY ────────────────────────────────────
      if (c.overlap && c.overlap.length) {
        const sets = c.overlap.map(o => ({
          label: o.name,
          data: o.conc.days.map((d, i) => ({ x: obDay(d), y: o.conc.counts[i] })),
          borderColor: o.color, borderWidth: 1, pointRadius: 0,
          pointHitRadius: 5, tension: 0, stepped: 'before', fill: false,
        }));
        // The portfolio total, dotted, as the old app drew it.
        const totals = new Map();
        for (const o of c.overlap) {
          o.conc.days.forEach((d, i) => {
            totals.set(d, (totals.get(d) || 0) + o.conc.counts[i]);
          });
        }
        const days = [...totals.keys()].sort();
        sets.push({
          label: 'Portfolio total',
          data: days.map(d => ({ x: obDay(d), y: totals.get(d) })),
          borderColor: BP_TOTAL, borderWidth: 1.8, borderDash: [4, 3],
          pointRadius: 0, pointHitRadius: 5, tension: 0, stepped: 'before',
          fill: false,
        });
        this.draw('overlap', 'bp-overlap-chart', { datasets: sets },
          { responsive: true, maintainAspectRatio: false, animation: false,
            parsing: false,
            interaction: { mode: 'nearest', axis: 'x', intersect: false },
            scales: {
              x: { type: 'linear', grid: { color: 'rgba(255,255,255,0.05)' },
                   border: { display: false },
                   ticks: { color: '#9a9a9a', font: { size: 10 },
                            maxTicksLimit: 7,
                            callback: v => obIsoDay(v).slice(0, 7) } },
              y: { beginAtZero: true,
                   grid: { color: 'rgba(255,255,255,0.05)' },
                   border: { display: false },
                   ticks: { color: '#9a9a9a', font: { size: 10 },
                            precision: 0 } },
            },
            plugins: { legend: { display: false }, tooltip: { callbacks: {
              title: it => obIsoDay(it[0].parsed.x),
              label: it => `${it.dataset.label}: ${it.parsed.y} open` } } } });
      }

      // ── ROLLING RISK: Sharpe and Sortino left, win rate right ────────
      const dy = c.daily;
      if (dy && dy.values.length) {
        const w = this.riskWindow;
        this.risk = { days: dy.values.length, window: w,
                      fits: dy.values.length >= w };
        const xs = dy.dates.map(d => obDay(d));
        const pair = (arr) => arr.map((v, i) => (v === null ? null
          : { x: xs[i], y: v })).filter(Boolean);
        // THE AXIS SPANS THE SERIES, not the points. A window longer than
        // the portfolio has close-days produces no points at all, and
        // Chart.js then scales a linear x axis from zero -- which reads as
        // "1970" and looks like a broken chart rather than a window that
        // does not fit. The card says which it is.
        const xr = xs.length ? { min: xs[0], max: xs[xs.length - 1] } : {};
        const sharpe = pair(obRollingSharpe(dy.values, w));
        const sortino = pair(obRollingSortino(dy.values, w));
        const wins = pair(obRollingWinRate(dy.values, w));
        this.draw('risk', 'bp-risk-chart', { datasets: [
          { label: `Sharpe (${w})`, data: sharpe, borderColor: BP_SHARPE,
            borderWidth: 1.6, pointRadius: 0, pointHitRadius: 5, tension: 0,
            yAxisID: 'y' },
          { label: `Sortino (${w})`, data: sortino, borderColor: BP_SORTINO,
            borderWidth: 1.6, pointRadius: 0, pointHitRadius: 5, tension: 0,
            yAxisID: 'y' },
          { label: `Win rate % (${w})`, data: wins, borderColor: BP_WINRATE,
            borderWidth: 1.2, borderDash: [3, 3], pointRadius: 0,
            pointHitRadius: 5, tension: 0, yAxisID: 'y1' },
        ] }, {
          responsive: true, maintainAspectRatio: false, animation: false,
          parsing: false,
          interaction: { mode: 'nearest', axis: 'x', intersect: false },
          scales: {
            x: { type: 'linear', ...xr,
                 grid: { color: 'rgba(255,255,255,0.05)' },
                 border: { display: false },
                 ticks: { color: '#9a9a9a', font: { size: 10 },
                          maxTicksLimit: 7,
                          callback: v => obIsoDay(v).slice(0, 7) } },
            // TWO AXES because the old app had two: a ratio and a
            // percentage do not share a scale, and the card names which
            // line belongs to which side.
            y: { type: 'linear', position: 'left',
                 grid: { color: 'rgba(255,255,255,0.05)' },
                 border: { display: false },
                 ticks: { color: '#9a9a9a', font: { size: 10 },
                          callback: v => v.toFixed(1) } },
            y1: { type: 'linear', position: 'right', min: 0, max: 100,
                  grid: { display: false }, border: { display: false },
                  ticks: { color: BP_WINRATE, font: { size: 10 },
                           callback: v => v + '%' } },
          },
          plugins: { legend: { display: false }, tooltip: { callbacks: {
            title: it => obIsoDay(it[0].parsed.x),
            label: it => `${it.dataset.label}: ${it.parsed.y.toFixed(2)}` } } },
        });
      }

      this.draw('cap', 'bp-cap-chart', capData,
                unf(shaded(base(it => `${bpFmtMoney(it.parsed.y)} deployed`,
                            cap.length ? { min: cap[0].x, max: cap[cap.length - 1].x } : {}))),
                [bpLiveShade, bpUnfilteredShade]);
    },

    drawBar(key, id, data, options) {
      const el = document.getElementById(id);
      if (!el) return;
      const live = BP_CHARTS[key];
      if (live && live.canvas === el) {
        live.data = data;
        live.options = options;
        live.update('none');
        return;
      }
      if (live) live.destroy();
      BP_CHARTS[key] = new Chart(el.getContext('2d'), { type: 'bar', data, options });
    },

    /* The live-strategy bands, and the key that says what they mean.
     * Unexplained shading is worse than none: a grey stretch nobody can
     * name reads as a rendering fault. */
    liveShade() {
      void this.tick;
      // The DRAWN curves, so the bands and the lines cannot disagree.
      return bpLiveBands((BP_DATA.curves || {}).eq || []);
    },

    /* The metric shade's key. It names the filter that SET THE BOUNDARY --
     * the latest coverage among the active ones -- rather than listing
     * every blind filter: the boundary is one date and one metric put it
     * there. Null when no active filter is blind. */
    unfilteredKey() {
      void this.tick;
      const c = BP_DATA.curves || {};
      if (!c.unfilteredTo) return null;
      let label = '';
      for (const ch of this.chosen) {
        for (const m of this.registryFor(ch)) {
          const f = ch.filters && ch.filters[m.key];
          if (f && f.on && m.minDate === c.coverageFrom) label = m.label;
        }
      }
      return { to: c.unfilteredTo, from: c.coverageFrom, label,
               n: c.unfilteredN || 0, noBar: c.noBarN || 0,
               bg: `rgba(${BP_SHADE},${BP_SHADE_METRIC})` };
    },

    /* THE OVERLAP. Two shades are drawn but THREE tones appear, because
     * where both conditions hold the fills compound. The key named two of
     * them, so the darkest region on screen matched nothing in it. The
     * combined alpha is what compositing actually produces --
     * 1 - (1-a)(1-b) -- not a third constant to keep in step. */
    bothKey() {
      void this.tick;
      const c = BP_DATA.curves || {};
      const lk = this.liveKey();
      if (!lk || !c.unfilteredTo) return null;
      // Only when they really overlap: the metric shade runs from the left
      // edge, so it overlaps if any partial band starts before it ends.
      const to = obDay(c.unfilteredTo);
      const hit = this.liveShade().bands.some(b => b.partial && b.from < to);
      if (!hit) return null;
      const a = 1 - (1 - BP_SHADE_STRATEGY) * (1 - BP_SHADE_METRIC);
      return { bg: `rgba(${BP_SHADE},${a.toFixed(3)})` };
    },

    /* ONE ENTRY, not one per count: the shade is a state, not a scale. */
    liveKey() {
      const s = this.liveShade();
      if (!s.bands.some(b => b.partial)) return null;
      return { total: s.total, bg: `rgba(${BP_SHADE},${BP_SHADE_STRATEGY})` };
    },

    draw(key, id, data, options, plugins) {
      const el = document.getElementById(id);
      if (!el) return;
      const live = BP_CHARTS[key];
      if (live && live.canvas === el) {
        live.data = data;
        live.options = options;
        live.update('none');
        return;
      }
      if (live) live.destroy();
      BP_CHARTS[key] = new Chart(el.getContext('2d'),
                                 { type: 'line', data, options, plugins: plugins || [] });
    },

    // ── readouts ────────────────────────────────────────────────────────
    perfSub() {
      void this.tick;
      const t = this.rows.find(r => r.total);
      if (!t) return '';
      const dd = t.stats.max_drawdown;
      return `${bpFmtMoney(t.stats.total_pnl)} total · deepest drawdown `
           + `${bpFmtMoney(dd)}`;
    },

    deploySub() {
      void this.tick;
      if (!this.deploy.peak) return 'nothing held overnight';
      return `peak ${bpFmtMoney(this.deploy.peak)}`
           + (this.deploy.peakDay ? ` on ${this.deploy.peakDay}` : '');
    },

    monthsSub() {
      void this.tick;
      const n = Object.keys(this.months.cells).length;
      return n ? `${n} months · biggest ${bpFmtMoney(this.months.max)}` : '';
    },

    /* One month's cell: text, and a blue/pink wash whose opacity is the
     * month's size against the biggest month in the grid — the same rule the
     * bar charts on the other page use, so the two read alike. */
    monthCell(year, i) {
      void this.tick;
      const key = `${year}-${String(i + 1).padStart(2, '0')}`;
      const v = this.months.cells[key];
      if (v === undefined) return { text: '·', bg: 'transparent', empty: true, title: `${key}: no closes` };
      const a = this.months.max ? 0.12 + 0.68 * (Math.abs(v) / this.months.max) : 0.12;
      const rgb = v >= 0 ? '52,152,219' : '232,67,147';
      return { text: bpFmtMoney(v), bg: `rgba(${rgb},${a.toFixed(3)})`,
               empty: false, title: `${key}: ${bpFmtMoney(v)}` };
    },

    /* The annual bar for one year, as percentages of its own grid cell --
     * so it is laid out by the row it belongs to and cannot drift off it.
     * The per-year TOTAL is not printed beside December any more: the bar
     * carries it (and its tooltip states it), and the number was the same
     * figure twice. */
    yearBar(year) {
      void this.tick;
      const totals = this.months.totals;
      const v = totals[year] || 0;
      const vals = Object.values(totals);
      const biggest = Math.max(1, ...vals.map(Math.abs));
      // ZERO SITS IN THE MIDDLE ONLY WHEN SOME YEAR LOST MONEY. Reserving
      // half the column for a direction nothing uses would halve every
      // bar's resolution to draw white space.
      const anyNeg = vals.some(x => x < 0);
      const zero = anyNeg ? 50 : 0;
      const width = (anyNeg ? 50 : 100) * Math.abs(v) / biggest;
      return { zero, width, left: v >= 0 ? zero : zero - width,
               bg: v >= 0 ? BP_BLUE : BP_PINK,
               title: `${year}: ${bpFmtMoney(v)}` };
    },

    /* What the bar column's width means, stated once under it rather than
     * as a number on every row. */
    yearScale() {
      void this.tick;
      const vals = Object.values(this.months.totals);
      if (!vals.length) return '';
      const biggest = Math.max(1, ...vals.map(Math.abs));
      return vals.some(x => x < 0)
        ? `±${bpFmtMoney(biggest)}`
        : `0 to ${bpFmtMoney(biggest)}`;
    },

    // ── P4 readouts ─────────────────────────────────────────────────────
    corrCell(i, j) {
      void this.tick;
      const r = this.corr.matrix[i] && this.corr.matrix[i][j];
      if (r === null || r === undefined) return { text: '—', bg: 'transparent' };
      // Blue for together, pink for apart, opacity by strength: the page's
      // two colours, used the way every other chart here uses them.
      const a = 0.10 + 0.60 * Math.abs(r);
      const rgb = r >= 0 ? '52,152,219' : '232,67,147';
      return { text: r.toFixed(2), bg: `rgba(${rgb},${a.toFixed(3)})` };
    },

    corrSub() {
      void this.tick;
      if (!this.corr.weeks) return '';
      return `${this.corr.weeks} weeks · Pearson on weekly P/L`;
    },

    pairSub() {
      void this.tick;
      const p = this.corr.pairs.find(x => x.i === this.pairA && x.j === this.pairB);
      return p && p.r !== null ? `r = ${p.r.toFixed(2)}` : '';
    },

    onPair() { this.renderCharts(); },

    onRoll() { this.renderCharts(); },

    onRisk() { this.renderCharts(); },

    riskSub() {
      void this.tick;
      if (!this.risk.days) return '';
      const d = `${bpFmtInt(this.risk.days)} days with a close`;
      return this.risk.fits ? d
        : `${d} — fewer than the ${this.risk.window} this window needs, so `
          + `nothing is drawn`;
    },

    metricRow(m) {
      void this.tick;
      if (m.rho === null) return { text: '—', bg: 'transparent', n: m.n };
      const a = 0.10 + 0.60 * Math.abs(m.rho);
      const rgb = m.rho >= 0 ? '52,152,219' : '232,67,147';
      return { text: m.rho.toFixed(3), bg: `rgba(${rgb},${a.toFixed(3)})`, n: m.n };
    },

    savedSummary() {
      return this.saved.length
        ? `${this.saved.length} saved · ${this.chosen.length} chosen`
        : 'none saved yet';
    },

    allocSummary() { return `${bpFmtMoney(bpTotalCapital(this.chosen))} total`; },

    rowMeta(c) {
      const s = this.saved.find(x => x.id === c.id);
      const n = s ? bpFmtInt(s.trade_count) : '—';
      const seeded = c.savedCapital == null ? ' · capital not saved' : '';
      return `${n} trades${seeded}`;
    },

    loadedSummary() {
      const trades = this.loaded.reduce((a, p) => a + (p.n || 0), 0);
      const sp = this.span();
      return `${this.loaded.length} strategies · ${bpFmtInt(trades)} trades`
           + (sp ? ` · ${sp.start} → ${sp.end}` : '');
    },

    statusLine() {
      if (!this.chosen.length) return 'Add saved strategies to build a portfolio.';
      if (!this.loaded.length) return `${this.chosen.length} chosen — not loaded yet.`;
      const sp = this.span();
      return `${this.loaded.length} loaded · `
           + `${bpFmtMoney(bpTotalCapital(this.chosen))} capital · `
           + (sp ? `${sp.start} → ${sp.end} (${this.rangeMode})` : 'no dates');
    },

    spanOf(p) {
      return (p.date_min && p.date_max) ? `${p.date_min} → ${p.date_max}` : '—';
    },

    parseOf(p) {
      const q = p.parse || {};
      return `${q.source === 'cache' ? 'cached' : 'parsed'} ${bpFmtSeconds(q.seconds)}`;
    },

    onRangeMode() { this.recompute(); },

    // Cell formatters. `void this.tick` is the reactivity tie: the rows are
    // rebuilt into reactive state, but a cell that read only BP_DATA would
    // never re-render. Same trap the OO page hit twice.
    cell(row, what) {
      void this.tick;
      const s = row.stats, e = row.extra;
      switch (what) {
        case 'n':        return bpFmtInt(row.n);
        case 'win':      return bpFmtPct(s.win_pct);
        case 'total':    return bpFmtMoney(s.total_pnl);
        case 'avg':      return bpFmtMoney(s.avg_pnl);
        case 'avgWin':   return bpFmtMoney(s.avg_win_pnl);
        case 'avgLoss':  return bpFmtMoney(s.avg_loss_pnl);
        case 'pf':       return e.profit_factor === null ? '—'
                              : (isFinite(e.profit_factor) ? bpFmtNum(e.profit_factor) : '∞');
        case 'dd':       return bpFmtMoney(s.max_drawdown);
        case 'calmar':   return bpFmtNum(e.calmar);
        case 'sharpe':   return bpFmtNum(row.sharpe);
        case 'annual':   return bpFmtMoney(e.avg_annual_pnl);
        case 'annualPct': return bpFmtPct(e.avg_annual_return_pct);
        case 'avgPct':   return bpFmtPct(e.avg_pnl_pct, 2);
        case 'days':     return bpFmtNum(s.avg_days_in_trade, 1);
        case 'peak':     return bpFmtMoney(row.peakDeployed);
        default:         return '';
      }
    },

    cellSign(row, what) {
      void this.tick;
      const s = row.stats, e = row.extra;
      const v = { total: s.total_pnl, avg: s.avg_pnl, dd: s.max_drawdown,
                  annual: e.avg_annual_pnl, calmar: e.calmar,
                  sharpe: row.sharpe, annualPct: e.avg_annual_return_pct,
                  avgPct: e.avg_pnl_pct }[what];
      if (v == null || !isFinite(v)) return '';
      return v >= 0 ? 'pos' : 'neg';
    },

    /* What the filters cost, per strategy, stated rather than implied. */
    rowNote(row) {
      void this.tick;
      if (row.total) return '';
      const parts = [];
      if (row.dropped) parts.push(`${bpFmtInt(row.dropped)} of ${bpFmtInt(row.nAll)} filtered out`);
      if (row.cost.noValue) {
        parts.push(`${bpFmtInt(row.cost.noValue)} have no value for an active `
                 + `filter and are dropped by it`);
      }
      if (row.conc && row.conc.sameSession === row.conc.counted && row.conc.counted) {
        parts.push('nothing held overnight, so no capital is deployed by this measure');
      }
      return parts.join(' · ');
    },

    warnings() {
      const out = [];
      for (const p of this.loaded) {
        const name = p.saved ? p.saved.name : p.filename;
        if (p.saved_count_changed) {
          out.push(`${name}: ${bpFmtInt(p.saved_count_changed.when_saved)} trades when `
                 + `saved, ${bpFmtInt(p.saved_count_changed.now)} now — the parser has `
                 + `changed since.`);
        }
        const mk = p.market || {};
        if (mk.joined === false) {
          out.push(`${name}: market data did not join (${mk.error || 'no reason given'}) `
                 + `— the metric filters have nothing to filter on.`);
        }
      }
      return out;
    },
  }));
});
