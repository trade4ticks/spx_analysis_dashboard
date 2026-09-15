/* ============================================================================
 * OO/Mesosim Backtest page — analysis of an Option Omega CSV or DeltaRay
 * Mesosim JSON trade log. Replaces the Plotly Dash app in
 * Options-Backtest-Dashboard.
 *
 * EVERYTHING RENDERS OFF THE REGISTRY (/api/oo-backtest/registry). The
 * sidebar filters and the metric sections are loops over that list; no metric
 * column is named in this file. Adding a metric is one registry entry.
 *
 * FILTERING HAPPENS HERE, NOT ON THE SERVER. The whole trade log arrives once,
 * columnar, with ISO date strings; a filter change never makes a request.
 *
 * Phase 2: uploads are joined to main.index_ohlc on the server (VIX levels at
 * the entry bar, gaps against the prior session, both ratio bases); the
 * sidebar shows read-only market freshness, the ratio-basis toggle and per-
 * metric coverage.
 * Phase 2b: saved strategies -- save an upload (name + notes), pick one from
 * the dropdown and Load it (the stored file is re-parsed and re-joined), or
 * Delete it after a confirm.
 * Phase 3: filters (one date range, a dual slider per range metric bounded by
 * the loaded data, checkboxes per categorical metric), the ten summary stats
 * and the cumulative P/L + drawdown charts, all recomputed in the browser.
 * Phase 4: the ten metric sections -- avg and total P/L by bin, and P/L vs the
 * metric with an OLS line (categorical metrics: no scatter; year adds win
 * rate) -- recomputed on every filter change from the filtered rows.
 * Later: five more summary figures (Calmar, avg annual P/L, profit factor,
 * avg annual return %, avg P/L %), a capital-per-position input, and the
 * Deployment chart of concurrent positions per SPX session.
 * ==========================================================================*/

const OB_BLUE = '#3498db';   // positive (theme --accent)
const OB_PINK = '#e84393';   // negative

/* Trade columns live OUTSIDE the Alpine proxy. Thousands of values wrapped in
 * reactive getters is slow to build and slower to iterate, and nothing in the
 * template binds to an individual value. */
const OB_DATA = { columns: null, n: 0, file: null, idx: [], sessions: [], conc: null, autoBins: {}, rank: null,
                  logToken: 0 };
/* Chart.js instances, also outside the proxy (Alpine would wrap their internals). */
const OB_CHARTS = { cum: null, dd: null, deploy: null, rank: null, rankAxis: null, sec: {} };
const OB_DEFAULT_CAPITAL = 10000;

/* ── pure helpers (exercised in node by scripts/check_oo_backtest.py) ─────── */

/* Which bin a value falls in, matching pd.cut over the registry's spec.
 * `edges` are the INNER edges; the outer two are -inf/+inf.
 *   closed "left"   bins are [e(i-1), e(i))  -> index = #edges <= v
 *   closed "right"  bins are (e(i-1), e(i)]  -> index = #edges <  v
 * The side is an explicit registry field; anything else is a registry bug and
 * throws rather than silently binning one way.
 * Null / NaN -> -1 (no bin), as pd.cut gives NaN. */
function obBinIndex(v, bins) {
  if (v === null || v === undefined || Number.isNaN(v)) return -1;
  if (bins.closed !== 'left' && bins.closed !== 'right') {
    throw new Error(`bin spec has no closed side: ${JSON.stringify(bins.closed)}`);
  }
  const e = bins.edges;
  let lo = 0, hi = e.length;
  if (bins.closed === 'right') {
    while (lo < hi) { const m = (lo + hi) >> 1; if (e[m] < v) lo = m + 1; else hi = m; }
  } else {
    while (lo < hi) { const m = (lo + hi) >> 1; if (e[m] <= v) lo = m + 1; else hi = m; }
  }
  return lo;
}

/* Non-null numeric extent of a column, or null when it has no values. */
function obExtent(values) {
  let lo = Infinity, hi = -Infinity, n = 0;
  for (const v of values || []) {
    if (v === null || v === undefined || Number.isNaN(v)) continue;
    if (v < lo) lo = v;
    if (v > hi) hi = v;
    n++;
  }
  return n ? { min: lo, max: hi, n } : null;
}

/* Distinct non-null values, sorted. */
function obDistinct(values) {
  const s = new Set();
  for (const v of values || []) if (v !== null && v !== undefined) s.add(v);
  return [...s].sort((a, b) => (a < b ? -1 : a > b ? 1 : 0));
}

const obNull = v => v === null || v === undefined || (typeof v === 'number' && Number.isNaN(v));

/* Row indices passing every ACTIVE filter. A filter is only passed in when it
 * narrows something -- an untouched range lets null values through, a
 * narrowed one drops them (a trade with no VIX9D cannot be "VIX9D 12-20").
 *   {kind:'date',  column, from, to}   ISO strings, inclusive
 *   {kind:'range', column, lo, hi}     inclusive, as filter_dataframe had it
 *   {kind:'set',   column, allowed}    a Set of permitted values */
function obApplyFilters(cols, n, specs) {
  const idx = [];
  outer:
  for (let i = 0; i < n; i++) {
    for (const f of specs) {
      const col = cols[f.column];
      const v = col ? col[i] : null;
      if (f.kind === 'range') {
        if (obNull(v) || v < f.lo || v > f.hi) continue outer;
      } else if (f.kind === 'set') {
        if (!f.allowed.has(v)) continue outer;
      } else if (f.kind === 'date') {
        if (obNull(v) || (f.from && v < f.from) || (f.to && v > f.to)) continue outer;
      }
    }
    idx.push(i);
  }
  return idx;
}

/* Order for anything cumulative: by close date, ties kept in payload order
 * (open date, then file order). A STABLE sort -- pandas' default sort in the
 * source app's stats is not, so its max drawdown could differ across runs
 * when several trades close the same day. */
function obByClose(cols, idx) {
  const dc = cols.date_closed;
  return idx.slice().sort((a, b) => (dc[a] < dc[b] ? -1 : dc[a] > dc[b] ? 1 : a - b));
}

/* The summary figures, as utils/stats.py calculate_stats defines them --
 * including its quirk that a zero-P/L trade counts as a LOSS for Avg Loss
 * (losses are pnl <= 0) but not as a win. */
function obStats(cols, idx) {
  const n = idx.length;
  const empty = { num_trades: 0, win_pct: 0, avg_pnl: 0, total_pnl: 0, avg_days_in_trade: 0,
                  max_drawdown: 0, avg_win_pnl: 0, avg_loss_pnl: 0, max_winner: 0, max_loser: 0 };
  if (!n) return empty;
  const pnl = cols.pnl, dit = cols.days_in_trade || [];
  let total = 0, wins = 0, winSum = 0, losses = 0, lossSum = 0, maxW = -Infinity, maxL = Infinity;
  let ditSum = 0, ditN = 0;
  for (const i of idx) {
    const p = pnl[i];
    total += p;
    if (p > 0) { wins++; winSum += p; } else { losses++; lossSum += p; }
    if (p > maxW) maxW = p;
    if (p < maxL) maxL = p;
    if (!obNull(dit[i])) { ditSum += dit[i]; ditN++; }
  }
  const eq = obEquity(cols, idx);
  return {
    num_trades: n,
    win_pct: wins / n * 100,
    avg_pnl: total / n,
    total_pnl: total,
    avg_days_in_trade: ditN ? ditSum / ditN : 0,
    max_drawdown: eq.maxDD ? eq.maxDD.drawdown : 0,
    avg_win_pnl: wins ? winSum / wins : 0,
    avg_loss_pnl: losses ? lossSum / losses : 0,
    max_winner: maxW,
    max_loser: maxL,
  };
}

/* Cumulative P/L, running peak and drawdown per trade in close order, as
 * calculations.py calculate_drawdown. maxDD is the deepest point (first
 * occurrence), or null when the curve never falls below its peak. */
function obEquity(cols, idx) {
  const order = obByClose(cols, idx);
  const pnl = cols.pnl, dc = cols.date_closed;
  const points = [];
  let cum = 0, peak = -Infinity, maxDD = null;
  for (const i of order) {
    cum += pnl[i];
    if (cum > peak) peak = cum;
    const dd = cum - peak;
    const pt = { row: i, date: dc[i], pnl: pnl[i], cumulative: cum, peak, drawdown: dd };
    points.push(pt);
    if (dd < 0 && (maxDD === null || dd < maxDD.drawdown)) maxDD = pt;
  }
  return { points, maxDD };
}

/* Concurrent open positions per SPX session, over the span of the given
 * trades (first entry to last exit). A trade is open on every session from
 * its entry date to its exit date, both inclusive -- day granularity, so a
 * trade closed at 10:00 and one opened at 15:30 the same day count as two.
 * `sessions` is the sorted ISO list from the server; days outside the span are
 * dropped. A trade with no exit date is in no count (the parser excludes
 * still-open positions; this is the backstop, and it is counted, not hidden).
 * offSession counts trades whose entry or exit date is not in the list.
 * Returns { days, counts, peak, peakDay, offSession, unclosed }. */
function obConcurrency(cols, idx, sessions) {
  const out = { days: [], counts: [], peak: 0, peakDay: null, offSession: 0, unclosed: 0 };
  if (!sessions || !sessions.length || !idx.length) return out;
  const dOpen = cols.date_opened, dClose = cols.date_closed;
  let lo = null, hi = null;
  for (const i of idx) {
    if (!dClose[i]) continue;
    if (lo === null || dOpen[i] < lo) lo = dOpen[i];
    if (hi === null || dClose[i] > hi) hi = dClose[i];
  }
  if (lo === null) { out.unclosed = idx.length; return out; }
  // first index with sessions[k] >= d  /  > d
  const lower = d => { let a = 0, b = sessions.length; while (a < b) { const m = (a + b) >> 1; if (sessions[m] < d) a = m + 1; else b = m; } return a; };
  const upper = d => { let a = 0, b = sessions.length; while (a < b) { const m = (a + b) >> 1; if (sessions[m] <= d) a = m + 1; else b = m; } return a; };
  const k0 = lower(lo), k1 = upper(hi);          // span is sessions[k0 .. k1-1]
  const diff = new Array(Math.max(0, k1 - k0) + 1).fill(0);
  const known = new Set(sessions);
  for (const i of idx) {
    if (!dClose[i]) { out.unclosed++; continue; }
    if (!known.has(dOpen[i]) || !known.has(dClose[i])) out.offSession++;
    const a = lower(dOpen[i]) - k0, b = upper(dClose[i]) - k0;   // sessions [a, b)
    if (b > a) { diff[a]++; diff[b]--; }
  }
  let run = 0;
  for (let k = 0; k < k1 - k0; k++) {
    run += diff[k];
    out.days.push(sessions[k0 + k]);
    out.counts.push(run);
    if (run > out.peak) { out.peak = run; out.peakDay = sessions[k0 + k]; }
  }
  return out;
}

/* The five figures beyond stats.py's ten.
 *   years            (last exit - first entry) / 365.25 over these trades
 *   avg_annual_pnl   total P/L / years
 *   calmar           avg annual P/L / |max drawdown $|   (null with no drawdown)
 *   profit_factor    gross wins / |gross losses|         (Infinity with wins and
 *                                                         no losses, null with neither)
 *   avg_annual_return_pct  avg annual P/L / (peak concurrency x capital) x 100
 *   avg_pnl_pct            avg P/L / capital x 100
 * The two capital figures are null when capital is not a positive number.
 * `stats` is obStats' result; `peak` is obConcurrency's. */
function obExtraStats(cols, idx, stats, capital, peak) {
  const out = { years: null, avg_annual_pnl: null, calmar: null, profit_factor: null,
                avg_annual_return_pct: null, avg_pnl_pct: null };
  if (!idx.length) return out;
  let lo = null, hi = null, gw = 0, gl = 0;
  for (const i of idx) {
    const o = cols.date_opened[i], c = cols.date_closed[i], p = cols.pnl[i];
    if (o && (lo === null || o < lo)) lo = o;
    if (c && (hi === null || c > hi)) hi = c;
    if (p > 0) gw += p; else if (p < 0) gl += p;
  }
  const years = lo && hi ? (obDay(hi) - obDay(lo)) / 365.25 : 0;
  if (years > 0) {
    out.years = years;
    out.avg_annual_pnl = stats.total_pnl / years;
    if (stats.max_drawdown < 0) out.calmar = out.avg_annual_pnl / Math.abs(stats.max_drawdown);
  }
  out.profit_factor = gl < 0 ? gw / Math.abs(gl) : (gw > 0 ? Infinity : null);
  const cap = typeof capital === 'number' && capital > 0 ? capital : null;
  if (cap) {
    out.avg_pnl_pct = stats.avg_pnl / cap * 100;
    if (out.avg_annual_pnl !== null && peak > 0) out.avg_annual_return_pct = out.avg_annual_pnl / (peak * cap) * 100;
  }
  return out;
}



/* Where the BH boundary line goes in a sorted bar list: after the LAST bar
 * that survives. Benjamini-Hochberg is not a pure |r| threshold -- the
 * adjusted p also depends on each metric's n -- so survivors need not be an
 * unbroken run from the left. The line goes after the last one, and the bars
 * that sit on the wrong side of it are counted, so the line never quietly
 * misstates which bars survive.
 *   index         bars[0 .. index-1] are left of the line; null when none survive
 *   failLeft      non-survivors left of the line
 *   survivors     survivors in the list */
function obBhBoundary(bars) {
  let last = -1, survivors = 0;
  bars.forEach((b, i) => { if (b.survives) { last = i; survivors++; } });
  if (last < 0) return { index: null, failLeft: 0, survivors: 0 };
  return { index: last + 1, failLeft: last + 1 - survivors, survivors };
}

/* A row filter's range state from the row's own values (display units):
 * bounds snapped outward to a slider step one tenth of the bin step. */
function obRowFilterState(values, binStep, prevScope) {
  const ext = obExtent(values);
  if (!ext) return null;
  const step = obClean(binStep ? binStep / 10 : Math.max((ext.max - ext.min) / 100, 1e-6));
  const min = obSnap(ext.min, step, 'floor'), max = obSnap(ext.max, step, 'ceil');
  return { min, max, lo: min, hi: max, step, n: ext.n, scope: prevScope || 'row' };
}

/* ── surface metrics ranking (P6b) ─────────────────────────────────────── */

/* Family colours and form labels come from the server with the catalog
 * (surface.FAMILY_GROUPS / FORM_LABELS): the page names no metric family. */
const OB_BH_Q = 0.05;       // BH false-discovery level; the dashed line follows the last bar that clears it

function obSurfGroupOf(family, groups, other) {
  return (groups || []).find(g => g.families.includes(family)) || other || { label: 'Other', color: '#8a8a8a' };
}

/* The bars to draw, in order. rows: /surface/rank rows. opts:
 *   method       'spearman' | 'pearson' -- the value drawn, sorted on and
 *                whose BH p decides the outline
 *   form         'all' or one catalog form
 *   hidden       Set of hidden families
 *   groups/other the catalog's family_groups / other_group, for bar colour
 * Sorted by |value| descending, ties by column name. A metric with no value
 * for the method (too few trades, or constant) is not a bar; it is counted.
 * alpha is obBarAlpha against the largest n IN VIEW. */
function obRankView(rows, opts) {
  const m = opts.method;
  const inView = rows.filter(r => (opts.form === 'all' || r.form === opts.form) && !opts.hidden.has(r.family));
  const bars = inView.filter(r => r[m] !== null && r[m] !== undefined);
  bars.sort((a, b) => Math.abs(b[m]) - Math.abs(a[m]) || (a.column < b.column ? -1 : a.column > b.column ? 1 : 0));
  const maxN = bars.reduce((x, r) => Math.max(x, r.n), 0);
  return {
    bars: bars.map(r => ({ ...r, value: r[m], pBh: r[`${m}_p_bh`],
                           survives: r[`${m}_p_bh`] !== null && r[`${m}_p_bh`] < OB_BH_Q,
                           alpha: obBarAlpha(r.n, maxN), group: obSurfGroupOf(r.family, opts.groups, opts.other) })),
    inView: inView.length,
    undefinedInView: inView.length - bars.length,
    survivorsInView: bars.filter(r => r[`${m}_p_bh`] !== null && r[`${m}_p_bh`] < OB_BH_Q).length,
    survivorsAll: rows.filter(r => r[`${m}_p_bh`] !== null && r[`${m}_p_bh`] < OB_BH_Q).length,
    computedAll: rows.filter(r => r[m] !== null && r[m] !== undefined).length,
  };
}

/* What "common coverage only" would cost, over the given trades. minDates:
 * the coverage start of every metric in view. commonStart is the LATEST of
 * them (every metric in view has data from there), earliestStart the earliest.
 * dropped counts trades entered in [earliestStart, commonStart): trades some
 * metric in view currently uses and that common coverage would remove. Trades
 * before earliestStart have no metric value either way and are not a cost. */
function obCommonCoverage(dates, idx, minDates) {
  const starts = minDates.filter(Boolean).sort();
  const out = { commonStart: null, earliestStart: null, dropped: 0, beforeEarliest: 0, kept: 0 };
  if (!starts.length) return out;
  out.earliestStart = starts[0];
  out.commonStart = starts[starts.length - 1];
  for (const i of idx) {
    const d = dates[i];
    if (!d) continue;
    if (d < out.earliestStart) out.beforeEarliest++;
    else if (d < out.commonStart) out.dropped++;
    else out.kept++;
  }
  return out;
}

/* The request body's trades: [entry date, entry time, P/L] for each row in
 * idx, optionally only those entered on/after `from`. */
function obRankTrades(cols, idx, from) {
  const out = [];
  const t = cols.time_opened || [];
  for (const i of idx) {
    const d = cols.date_opened[i];
    if (from && (!d || d < from)) continue;
    out.push([d, t[i] ?? null, cols.pnl[i]]);
  }
  return out;
}

/* Wrap a long catalog description for a canvas tooltip. */
function obWrap(text, width) {
  const words = String(text || '').split(/\s+/).filter(Boolean);
  const lines = [];
  let line = '';
  for (const w of words) {
    if (line && (line + ' ' + w).length > width) { lines.push(line); line = w; } else line = line ? line + ' ' + w : w;
  }
  if (line) lines.push(line);
  return lines;
}

function obP(p) {
  if (p === null || p === undefined) return '—';
  return p < 0.001 ? p.toExponential(1) : p.toFixed(3);
}

/* P/L by bin for one metric section, as calculations.py calculate_bin_stats
 * over pd.cut: per bin count, total, mean and win rate (pnl > 0), bins in
 * label order. Empty bins: a range metric keeps them (see below), a
 * categorical one has none to keep. Trades with no value in `column` are in
 * no bin.
 *   range metric       bins from m.bins via obBinIndex
 *   categorical metric one bin per value: m.categories order first, then any
 *                      value the log has that the list lacks, labelled by the
 *                      value itself -- a Saturday trade is shown, not dropped.
 * Returns { valued, rows:[{label, count, total, avg, win}], xs, ys, rowsIdx }
 * where xs/ys/rowsIdx are the (value, pnl, row) triples the scatter plots. */
function obSectionData(cols, idx, m, column) {
  const vals = cols[column] || [], pnl = cols.pnl;
  const acc = new Map();   // label -> {extra, key, label, count, total, wins}
  const xs = [], ys = [], rowsIdx = [];
  const cats = m.categories ? new Map(m.categories.map((c, i) => [c.value, { i, label: c.label }])) : null;
  for (const i of idx) {
    const v = vals[i];
    if (obNull(v)) continue;
    // `extra` 0 sorts by bin/category index; 1 is a value outside a fixed
    // category list (or any value of a data-derived one), sorted by value.
    let extra = 0, key, label;
    if (m.type === 'range') {
      key = obBinIndex(v, m.bins); label = m.bins.labels[key];
    } else if (cats && cats.has(v)) {
      key = cats.get(v).i; label = cats.get(v).label;
    } else {
      extra = 1; key = v; label = String(v);
    }
    let b = acc.get(extra + ':' + key);
    if (!b) acc.set(extra + ':' + key, b = { extra, key, label, count: 0, total: 0, wins: 0 });
    b.count++; b.total += pnl[i]; if (pnl[i] > 0) b.wins++;
    xs.push(v); ys.push(pnl[i]); rowsIdx.push(i);
  }
  // A range metric keeps EVERY bin, empty ones included, so the x axis stays
  // to scale: dropping them put ">40" beside "18-20". Deliberately unlike
  // calculate_bin_stats (observed=True). An empty bin is count 0 with null
  // avg/total/win, which Chart.js draws as nothing.
  if (m.type === 'range') {
    m.bins.labels.forEach((label, key) => {
      if (!acc.has('0:' + key)) acc.set('0:' + key, { extra: 0, key, label, count: 0, total: 0, wins: 0 });
    });
  }
  const rows = [...acc.values()].sort((a, b) =>
    a.extra - b.extra || (a.key < b.key ? -1 : a.key > b.key ? 1 : 0)
  ).map(b => (b.count
    ? { label: b.label, count: b.count, total: b.total, avg: b.total / b.count, win: b.wins / b.count * 100 }
    : { label: b.label, count: 0, total: null, avg: null, win: null }));
  return { valued: xs.length, rows, xs, ys, rowsIdx };
}

/* Bar opacity from a bin's trade count relative to the largest bin in that
 * chart: floor + (1 - floor) * (count / max) ^ gamma. Relative only -- a chart
 * whose bins are all small still spreads across the full range. Tuning knobs:
 * gamma 1 is linear; above 1 pushes thin bins dimmer, below 1 lifts them (0.5
 * was the square root, which left n=1 of 531 clearly visible). */
const OB_ALPHA_FLOOR = 0.12;
const OB_ALPHA_GAMMA = 1.0;
function obBarAlpha(count, maxCount) {
  if (!count || !maxCount) return 0;
  return OB_ALPHA_FLOOR + (1 - OB_ALPHA_FLOOR) * Math.pow(count / maxCount, OB_ALPHA_GAMMA);
}

/* Percentile by linear interpolation between order statistics -- numpy's
 * default, so the gate can compare against np.percentile. `sorted` ascending. */
function obPercentile(sorted, p) {
  if (!sorted.length) return null;
  const h = (sorted.length - 1) * p / 100, lo = Math.floor(h), hi = Math.min(lo + 1, sorted.length - 1);
  return sorted[lo] + (h - lo) * (sorted[hi] - sorted[lo]);
}

/* Candidate "nice" steps for a span: 1, 2, 2.5, 5 x 10^k for the decade of
 * span/target and the one either side, ascending. A zero span (every value in
 * p1..p99 equal) takes its decade from the value itself. */
function obNiceSteps(pLo, pHi, target) {
  const span = pHi - pLo > 0 ? pHi - pLo : (Math.abs(pHi) || 1);
  const e = Math.floor(Math.log10(span / target));
  const out = [];
  for (let k = e - 1; k <= e + 1; k++) for (const m of [1, 2, 2.5, 5]) out.push(obClean(m * Math.pow(10, k)));
  return out;
}

/* Round to 12 significant digits: 24 * 0.005 is 0.12000000000000001. */
function obClean(x) { return Number(x.toPrecision(12)); }

function obDecimals(x) {
  const s = String(obClean(x));
  if (s.includes('e-')) return Number(s.split('e-')[1]) + ((s.split('e-')[0].split('.')[1] || '').length);
  return (s.split('.')[1] || '').length;
}

/* Data-driven bins for a registry metric with binning 'auto'. The inner edges
 * run lo, lo+step, ..., hi where lo/hi are the pLo/pHi percentiles snapped
 * OUT to the step; each candidate step gives (hi - lo) / step bins and the one
 * nearest targetBins wins (a tie goes to the smaller step). Values below lo
 * and at/above hi land in the two end buckets, "<lo" and "≥hi" (left-closed,
 * so a value exactly at hi is in the top bucket -- hence ≥, not >).
 * auto.steps is a list, or "nice" (obNiceSteps). Edges are cleaned to 12
 * significant digits; labels use at least as many decimals as the step.
 * Returns a bins object obBinIndex/obSectionData take, plus `step`, or null
 * when the column has no values. */
function obAutoBins(values, auto, fmt) {
  const v = (values || []).filter(x => !obNull(x)).sort((a, b) => a - b);
  if (!v.length) return null;
  const pLo = obPercentile(v, auto.pLo), pHi = obPercentile(v, auto.pHi);
  let best = null;
  const steps = auto.steps === 'nice' ? obNiceSteps(pLo, pHi, auto.targetBins) : auto.steps;
  for (const step of steps) {
    const lo = obClean(Math.floor(pLo / step) * step);
    let hi = obClean(Math.ceil(pHi / step) * step);
    if (hi <= lo) hi = obClean(lo + step);
    const n = Math.round((hi - lo) / step);
    if (!best || Math.abs(n - auto.targetBins) < Math.abs(best.n - auto.targetBins)) best = { step, lo, hi, n };
  }
  const labelFmt = fmt && typeof fmt === 'object'
    ? { ...fmt, decimals: Math.max(fmt.decimals ?? 0, obDecimals(best.step)) } : fmt;
  const f = x => (fmt === 'usd' ? obMoney(x) : obFmt(x, labelFmt));
  const edges = [];
  for (let k = 0; k <= best.n; k++) edges.push(obClean(best.lo + k * best.step));
  const labels = [`<${f(best.lo)}`];
  for (let k = 0; k < best.n; k++) labels.push(`${f(edges[k])} to ${f(edges[k + 1])}`);
  labels.push(`≥${f(best.hi)}`);
  return { edges, labels, closed: 'left', labelEdge: 'both', step: best.step, auto: true };
}

/* '#3498db', 0.5 -> 'rgba(52,152,219,0.5)' */
function obRgba(hex, a) {
  const n = parseInt(hex.slice(1), 16);
  return `rgba(${n >> 16},${(n >> 8) & 255},${n & 255},${+a.toFixed(3)})`;
}

/* Ordinary least squares of y on x, as scipy.stats.linregress (the fit
 * calculate_correlation reports). Null below 3 points, as that function
 * returns, or when every x is identical (no line exists). r is null when every
 * y is identical (Pearson r is undefined there). */
function obOLS(xs, ys) {
  const n = xs.length;
  if (n < 3) return null;
  let mx = 0, my = 0;
  for (let i = 0; i < n; i++) { mx += xs[i]; my += ys[i]; }
  mx /= n; my /= n;
  let sxx = 0, syy = 0, sxy = 0;
  for (let i = 0; i < n; i++) {
    const dx = xs[i] - mx, dy = ys[i] - my;
    sxx += dx * dx; syy += dy * dy; sxy += dx * dy;
  }
  if (sxx === 0) return null;
  const slope = sxy / sxx;
  const r = syy === 0 ? null : Math.max(-1, Math.min(1, sxy / Math.sqrt(sxx * syy)));
  return { n, slope, intercept: my - slope * mx, r, r2: r === null ? null : r * r };
}

/* Days since the epoch for an ISO date, for a linear x axis (no date adapter). */
function obDay(iso) { return Date.parse(iso + 'T00:00:00Z') / 86400000; }
function obIsoDay(d) { return new Date(Math.round(d) * 86400000).toISOString().slice(0, 10); }

function obMoney(v, digits = 0) {
  if (obNull(v)) return '—';
  const a = Math.abs(v).toLocaleString('en-US', { minimumFractionDigits: digits, maximumFractionDigits: digits });
  return (v < 0 ? '-$' : '$') + a;
}

/* "SPX Iron Condor 45DTE (1,284 trades, 2021-01-04 – 2026-03-13)" */
function obSavedLabel(s) {
  if (!s) return '';
  const n = Number(s.trade_count || 0).toLocaleString('en-US');
  return `${s.name} (${n} trade${s.trade_count === 1 ? '' : 's'}, ${s.date_min || '—'} – ${s.date_max || '—'})`;
}

/* Snap a data extent outward to the slider's step, so the untouched slider
 * spans every value (a floor/ceil at 0.1 on 0.37 must not cut off 0.37). */
function obStepDecimals(step) { return (String(step).split('.')[1] || '').length; }
function obSnap(v, step, how) {
  const dec = obStepDecimals(step);
  return +((Math[how](v / step + (how === 'floor' ? 1e-9 : -1e-9))) * step).toFixed(dec);
}

function obFmt(v, fmt) {
  if (v === null || v === undefined || Number.isNaN(v)) return '—';
  // An added surface row carries {decimals, suffix} from the catalog's units;
  // the number alone is formatted here (axes), the suffix is used in text.
  if (fmt && typeof fmt === 'object') return v.toFixed(fmt.decimals ?? 4);
  switch (fmt) {
    case 'usd':   return (v < 0 ? '-$' : '$') + Math.abs(v).toLocaleString(undefined, { maximumFractionDigits: 0 });
    case 'pct':   return v.toFixed(2) + '%';
    case 'ratio': return v.toFixed(3);
    case 'int':   return String(Math.round(v));
    case 'num':   return v.toFixed(2);
    default:      return String(v);
  }
}

document.addEventListener('alpine:init', () => {
  Alpine.data('ooBacktest', () => ({
    registry: [],
    registryError: '',

    // loaded log (metadata only — values are in OB_DATA)
    loaded: false,
    meta: null,          // {n, filename, source, suggested_name, date_min, date_max, market_joined}
    presentColumns: [],  // columns in the payload with at least one non-null value

    uploading: false,
    uploadError: '',
    dragOver: false,

    market: null,
    marketLoading: false,
    marketError: '',
    marketDetail: false,
    coverageError: '',

    // Saved strategies. The dropdown lists them; Load re-parses the stored
    // file and re-joins market data (the same path as an upload).
    savedList: [],
    savedError: '',
    selectedSavedId: '',
    savedBusy: false,
    // Set when the loaded log came from an upload whose File the page still
    // holds -- only then can it be saved (the server re-parses the bytes).
    hasUploadFile: false,
    saveName: '',
    saveNotes: '',
    saveBusy: false,
    saveError: '',
    saveMsg: '',

    // Filters. `ranges` and `cats` are keyed by registry key and built from
    // the loaded data by initFilters(); nothing here is a static default.
    filters: { dateFrom: '', dateTo: '', ranges: {}, cats: {} },
    dateBounds: { min: '', max: '' },
    filteredCount: 0,
    stats: null,
    // Per metric-section summary {valued, bins, fit}, keyed by registry key;
    // the chart data itself stays outside the proxy.
    sections: {},
    _rafPending: false,

    // Which ratio basis the sections read: entry-time bars (default) or the
    // entry date's daily closes. Temporary -- one basis is deleted, and this
    // toggle with it, once both have been looked at on a real log.
    ratioBasis: 'entry',

    // Capital per position: a display input. It feeds Avg Annual Return %,
    // Avg P/L % and the Deployment chart's dollar axis, recomputed here with no
    // request -- except that on a SAVED strategy the committed value is
    // written back (PUT .../capital) so a reload keeps it.
    capitalInput: String(OB_DEFAULT_CAPITAL),
    capitalMsg: '',
    autoSteps: {},     // registry key -> step of that metric's auto bins for this log

    // Surface metrics ranking. Rank rows live in OB_DATA.rank; `result` is the
    // summary of the last computation, `sig` what it was computed on.
    surf: { catalog: null, catalogBusy: false, catalogError: '', catalogInfo: null,
            groups: [], other: null, formLabels: {}, unitFormats: {}, defaultUnitFormat: null, rowAutoBins: null,
            method: 'spearman', form: 'all', hidden: {}, common: false,
            busy: false, error: '', result: null },
    // Bumped on every recompute. The filtered rows live outside the proxy
    // (OB_DATA.idx), so anything derived from them reads this to re-render
    // when they change -- without it the coverage headline stayed blank.
    idxTick: 0,
    surfRows: [],      // added surface metric sections, registry-shaped (P6c)
    marketChecksOpen: false,   // the diagnostics pane starts collapsed
    extra: null,       // obExtraStats
    deploy: null,      // {peak, peakDay, offSession, unclosed, days, hasSessions}

    async init() {
      await Promise.all([this.loadRegistry(), this.loadMarketStatus(), this.loadSavedList()]);
    },

    async loadRegistry() {
      try {
        const r = await fetch('/api/oo-backtest/registry');
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        const body = await r.json();
        this.registry = body.metrics || [];
        this.coverageError = body.coverage_error || '';
      } catch (e) {
        console.error('oo-backtest registry', e);
        this.registryError = `Registry failed to load: ${e.message}`;
      }
    },

    async loadMarketStatus() {
      this.marketLoading = true;
      this.marketError = '';
      try {
        const r = await fetch('/api/oo-backtest/market-status');
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        this.market = await r.json();
      } catch (e) {
        console.error('oo-backtest market-status', e);
        this.marketError = e.message;
      } finally {
        this.marketLoading = false;
      }
    },

    /* ── ingestion ─────────────────────────────────────────────────────── */

    onDrop(ev) {
      this.dragOver = false;
      const f = ev.dataTransfer && ev.dataTransfer.files && ev.dataTransfer.files[0];
      if (f) this.upload(f);
    },

    onPick(ev) {
      const f = ev.target.files && ev.target.files[0];
      if (f) this.upload(f);
      ev.target.value = '';   // re-picking the same file must fire change again
    },

    async upload(file) {
      const name = file.name || '';
      if (!/\.(csv|json)$/i.test(name)) {
        this.uploadError = `${name}: upload an Option Omega .csv or a Mesosim .json`;
        return;
      }
      this.uploading = true;
      this.uploadError = '';
      try {
        const fd = new FormData();
        fd.append('file', file);
        const r = await fetch('/api/oo-backtest/parse', { method: 'POST', body: fd });
        const body = await r.json().catch(() => ({}));
        if (!r.ok) throw new Error(body.detail || `HTTP ${r.status}`);
        this.setTrades(body);
        OB_DATA.file = file;
        this.hasUploadFile = true;
        this.selectedSavedId = '';
        this.saveName = body.suggested_name || '';
        this.saveNotes = '';
        this.saveError = '';
        this.saveMsg = '';
      } catch (e) {
        console.error('oo-backtest upload', e);
        this.uploadError = e.message;
      } finally {
        this.uploading = false;
      }
    },

    setTrades(payload) {
      OB_DATA.columns = payload.columns;
      OB_DATA.n = payload.n;
      OB_DATA.logToken++;
      this.presentColumns = Object.keys(payload.columns)
        .filter(c => payload.columns[c].some(v => v !== null && v !== undefined));
      const { columns, ...meta } = payload;
      OB_DATA.sessions = (payload.market && payload.market.spx_sessions) || [];
      if (payload.saved) {
        const c = payload.saved.capital_per_position;
        this.capitalInput = String(c === null || c === undefined ? OB_DEFAULT_CAPITAL : c);
      }
      this.capitalMsg = '';
      this.meta = meta;
      this.computeAutoBins();
      // A new log invalidates any ranking of the previous one.
      OB_DATA.rank = null;
      this.surf.result = null;
      this.loaded = true;
      this.loadSurfaceCatalog();
      for (const row of this.surfRows) this.fetchSurfRow(row.key);
      this.initFilters();
      this.$nextTick(() => this.recompute());
    },

    /* ── saved strategies ──────────────────────────────────────────────── */

    savedLabel(s) { return obSavedLabel(s); },

    async loadSavedList() {
      this.savedError = '';
      try {
        const r = await fetch('/api/oo-backtest/strategies');
        const body = await r.json().catch(() => ({}));
        if (!r.ok) throw new Error(body.detail || `HTTP ${r.status}`);
        this.savedList = body.strategies || [];
        if (this.selectedSavedId && !this.savedList.some(x => String(x.id) === String(this.selectedSavedId))) {
          this.selectedSavedId = '';
        }
      } catch (e) {
        console.error('oo-backtest saved list', e);
        this.savedError = `Saved strategies unavailable: ${e.message}`;
      }
    },

    selectedSaved() { return this.savedList.find(x => String(x.id) === String(this.selectedSavedId)) || null; },

    async loadSelected() {
      const s = this.selectedSaved();
      if (!s) return;
      this.savedBusy = true;
      this.savedError = '';
      this.uploadError = '';
      try {
        const r = await fetch(`/api/oo-backtest/strategies/${s.id}/load`);
        const body = await r.json().catch(() => ({}));
        if (!r.ok) throw new Error(body.detail || `HTTP ${r.status}`);
        this.setTrades(body);
        // A loaded saved strategy is already saved; there is no File to send.
        OB_DATA.file = null;
        this.hasUploadFile = false;
        this.saveMsg = '';
        this.saveError = '';
      } catch (e) {
        console.error('oo-backtest load saved', e);
        this.savedError = `Could not load "${s.name}": ${e.message}`;
      } finally {
        this.savedBusy = false;
      }
    },

    async saveStrategy(replace = false) {
      if (!OB_DATA.file) return;
      this.saveBusy = true;
      this.saveError = '';
      this.saveMsg = '';
      try {
        const fd = new FormData();
        fd.append('file', OB_DATA.file);
        fd.append('name', this.saveName);
        fd.append('notes', this.saveNotes);
        fd.append('capital_per_position', this.capital() === null ? '' : String(this.capital()));
        fd.append('replace', replace ? 'true' : 'false');
        const r = await fetch('/api/oo-backtest/strategies', { method: 'POST', body: fd });
        const body = await r.json().catch(() => ({}));
        if (r.status === 409 && !replace) {
          this.saveBusy = false;
          if (confirm(`${body.detail} Replace it with this upload?`)) return this.saveStrategy(true);
          this.saveError = 'Not saved — choose a different name.';
          return;
        }
        if (!r.ok) throw new Error(body.detail || `HTTP ${r.status}`);
        const saved = body.strategy;
        await this.loadSavedList();
        this.selectedSavedId = String(saved.id);
        this.meta = { ...this.meta, saved };
        this.saveMsg = `Saved as ${obSavedLabel(saved)}` +
          (saved.same_file_as && saved.same_file_as.length ? ` — the same file is also saved as: ${saved.same_file_as.join(', ')}` : '');
      } catch (e) {
        console.error('oo-backtest save', e);
        this.saveError = `Save failed: ${e.message}`;
      } finally {
        this.saveBusy = false;
      }
    },

    async deleteSaved() {
      const s = this.selectedSaved();
      if (!s) return;
      if (!confirm(`Delete saved strategy "${obSavedLabel(s)}"? This cannot be undone.`)) return;
      this.savedError = '';
      try {
        const r = await fetch(`/api/oo-backtest/strategies/${s.id}`, { method: 'DELETE' });
        const body = await r.json().catch(() => ({}));
        if (!r.ok) throw new Error(body.detail || `HTTP ${r.status}`);
        // The trades on screen stay; they are just no longer a saved record.
        if (this.meta && this.meta.saved && this.meta.saved.id === s.id) {
          const { saved, ...rest } = this.meta;
          this.meta = rest;
        }
        this.selectedSavedId = '';
        await this.loadSavedList();
      } catch (e) {
        console.error('oo-backtest delete', e);
        this.savedError = `Delete failed: ${e.message}`;
      }
    },

    /* ── filters ───────────────────────────────────────────────────────── */

    /* Bounds and options from THE LOADED LOG, never static defaults: every
     * range slider opens spanning the data's own extent (snapped outward to
     * its step), every checkbox on. */
    initFilters() {
      const cols = OB_DATA.columns || {};
      const opened = (cols.date_opened || []).filter(Boolean);
      const dmin = opened.length ? opened.reduce((a, b) => (a < b ? a : b)) : '';
      const dmax = opened.length ? opened.reduce((a, b) => (a > b ? a : b)) : '';
      this.dateBounds = { min: dmin, max: dmax };
      const ranges = {}, cats = {};
      for (const m of this.registry.filter(x => x.filter)) {
        if (m.type === 'range') {
          const r = this.rangeStateFor(m);
          if (r) ranges[m.key] = r;
        } else {
          const vals = cols[m.column] || [];
          const counts = new Map();
          for (const v of vals) counts.set(v, (counts.get(v) || 0) + 1);
          let options;
          if (m.categories) {
            options = m.categories.map(c => ({ value: c.value, label: c.label, count: counts.get(c.value) || 0 }));
          } else {
            options = obDistinct(vals).map(v => ({ value: v, label: String(v), count: counts.get(v) }));
            if (counts.has(null)) options.push({ value: null, label: '(none)', count: counts.get(null) });
          }
          cats[m.key] = { options, selected: options.map(o => o.value) };
        }
      }
      this.filters = { dateFrom: dmin, dateTo: dmax, ranges, cats };
    },

    rangeStateFor(m) {
      const ext = obExtent((OB_DATA.columns || {})[this.metricColumn(m)]);
      if (!ext) return null;
      const min = obSnap(ext.min, m.step, 'floor'), max = obSnap(ext.max, m.step, 'ceil');
      return { min, max, lo: min, hi: max, step: m.step, n: ext.n };
    },

    rangeOf(m) { return this.filters.ranges[m.key] || null; },
    catOf(m) { return this.filters.cats[m.key] || null; },

    dateActive() {
      return !!(this.dateBounds.min && ((this.filters.dateFrom && this.filters.dateFrom > this.dateBounds.min)
        || (this.filters.dateTo && this.filters.dateTo < this.dateBounds.max)));
    },

    isActive(m) {
      const r = this.rangeOf(m), c = this.catOf(m);
      if (m.type === 'range') return !!r && (r.lo > r.min || r.hi < r.max);
      return !!c && c.selected.length < c.options.length;
    },

    activeCount() {
      return (this.dateActive() ? 1 : 0) + this.registry.filter(m => m.filter && this.isActive(m)).length
        + this.surfRows.filter(m => this.rowFilterOnPage(m)).length;
    },

    setLo(m, raw) {
      const r = this.rangeOf(m);
      if (!r) return;
      r.lo = Math.min(+raw, r.hi);
      this.onFilterChange();
    },

    setHi(m, raw) {
      const r = this.rangeOf(m);
      if (!r) return;
      r.hi = Math.max(+raw, r.lo);
      this.onFilterChange();
    },

    toggleCat(m, value) {
      const c = this.catOf(m);
      if (!c) return;
      const i = c.selected.findIndex(v => v === value);
      if (i >= 0) c.selected.splice(i, 1); else c.selected.push(value);
      this.onFilterChange();
    },

    setAllCats(m, on) {
      const c = this.catOf(m);
      if (!c) return;
      c.selected = on ? c.options.map(o => o.value) : [];
      this.onFilterChange();
    },

    resetFilter(m) {
      if (m.type === 'range') {
        const r = this.rangeOf(m);
        if (r) { r.lo = r.min; r.hi = r.max; }
      } else {
        this.setAllCats(m, true);
        return;
      }
      this.onFilterChange();
    },

    resetDate() {
      this.filters.dateFrom = this.dateBounds.min;
      this.filters.dateTo = this.dateBounds.max;
      this.onFilterChange();
    },

    resetAllFilters() {
      this.initFilters();
      for (const m of this.surfRows) if (m.rowFilter) { m.rowFilter.lo = m.rowFilter.min; m.rowFilter.hi = m.rowFilter.max; }
      this.onFilterChange();
    },

    setRatioBasis(basis) {
      if (this.ratioBasis === basis) return;
      this.ratioBasis = basis;
      this.computeAutoBins();   // an auto-binned metric with a basis bins its new column
      // A basis switch changes the COLUMN a ratio filter reads, so its bounds
      // come from the new column and any narrowing on the old one is dropped.
      for (const m of this.registry.filter(x => x.basis && x.filter)) {
        const r = this.rangeStateFor(m);
        if (r) this.filters.ranges[m.key] = r; else delete this.filters.ranges[m.key];
      }
      this.onFilterChange();
    },

    activeSpecs() {
      const specs = [];
      if (this.dateActive()) {
        specs.push({ kind: 'date', column: 'date_opened', from: this.filters.dateFrom, to: this.filters.dateTo });
      }
      for (const m of this.registry.filter(x => x.filter)) {
        if (!this.isActive(m)) continue;
        if (m.type === 'range') {
          const r = this.rangeOf(m);
          specs.push({ kind: 'range', column: this.metricColumn(m), lo: r.lo, hi: r.hi });
        } else {
          specs.push({ kind: 'set', column: m.column, allowed: new Set(this.catOf(m).selected) });
        }
      }
      // An added row's filter joins the page only in page scope.
      for (const m of this.surfRows) {
        if (this.rowFilterOnPage(m)) specs.push({ kind: 'range', column: m.column, lo: m.rowFilter.lo, hi: m.rowFilter.hi });
      }
      return specs;
    },

    /* Slider input fires continuously; recompute at most once per frame. */
    onFilterChange() {
      if (this._rafPending) return;
      this._rafPending = true;
      const run = () => { this._rafPending = false; this.recompute(); };
      if (typeof requestAnimationFrame === 'function') requestAnimationFrame(run); else run();
    },

    recompute() {
      if (!OB_DATA.columns) return;
      const idx = obApplyFilters(OB_DATA.columns, OB_DATA.n, this.activeSpecs());
      OB_DATA.idx = idx;
      this.idxTick++;
      this.filteredCount = idx.length;
      this.stats = obStats(OB_DATA.columns, idx);
      OB_DATA.conc = obConcurrency(OB_DATA.columns, idx, OB_DATA.sessions);
      this.renderPerformance(obEquity(OB_DATA.columns, idx));
      this.renderSections(idx);
      this.recomputeCapital();
    },


    /* ── surface metrics ranking (P6b) ─────────────────────────────────── */

    surfaceForms() {
      const labels = this.surf.formLabels || {};
      const present = new Set((this.surf.catalog || []).map(m => m.form));
      return [{ value: 'all', label: 'All forms' },
              ...Object.keys(labels).filter(f => present.has(f)).map(f => ({ value: f, label: labels[f] })),
              ...[...present].filter(f => !(f in labels)).sort().map(f => ({ value: f, label: f }))];
    },
    surfaceGroups() {
      const fams = new Set((this.surf.catalog || []).map(m => m.family));
      const defs = this.surf.groups || [];
      const groups = defs.map(g => ({ ...g, families: g.families.filter(f => fams.has(f)) })).filter(g => g.families.length);
      const other = [...fams].filter(f => !defs.some(g => g.families.includes(f))).sort();
      if (other.length) groups.push({ ...(this.surf.other || { label: 'Other', color: '#8a8a8a' }), families: other });
      return groups;
    },
    familyCount(f) { return (this.surf.catalog || []).filter(m => m.family === f && (this.surf.form === 'all' || m.form === this.surf.form)).length; },
    familyHidden(f) { return !!this.surf.hidden[f]; },
    toggleFamily(f) {
      this.surf.hidden = { ...this.surf.hidden, [f]: !this.surf.hidden[f] };
      this.renderRanking();
    },
    setSurfMethod(m) { this.surf.method = m; this.renderRanking(); },
    setSurfForm(f) { this.surf.form = f; this.renderRanking(); },

    async loadSurfaceCatalog() {
      if (this.surf.catalog || this.surf.catalogBusy) return;
      this.surf.catalogBusy = true;
      this.surf.catalogError = '';
      try {
        const r = await fetch('/api/oo-backtest/surface/catalog');
        const body = await r.json().catch(() => ({}));
        if (!r.ok) throw new Error(body.detail || `HTTP ${r.status}`);
        this.surf.catalog = body.metrics;
        this.surf.groups = body.family_groups || [];
        this.surf.other = body.other_group || null;
        this.surf.formLabels = body.form_labels || {};
        this.surf.unitFormats = body.unit_formats || {};
        this.surf.defaultUnitFormat = body.default_unit_format || null;
        this.surf.rowAutoBins = body.row_auto_bins || null;
        this.surf.catalogInfo = { first: body.first_date, last: body.last_date, lookahead: body.lookahead_confirmed };
      } catch (e) {
        console.error('oo-backtest surface catalog', e);
        this.surf.catalogError = `Surface metrics unavailable: ${e.message}`;
      } finally {
        this.surf.catalogBusy = false;
      }
    },

    /* Metrics the view shows (form + families), from the catalog -- the set
     * common coverage is computed over, available before any ranking. */
    surfViewMetrics() {
      return (this.surf.catalog || []).filter(m => (this.surf.form === 'all' || m.form === this.surf.form)
                                                  && !this.surf.hidden[m.family]);
    },

    surfCoverage() {
      void this.idxTick;
      if (!OB_DATA.columns) return null;
      return obCommonCoverage(OB_DATA.columns.date_opened, OB_DATA.idx, this.surfViewMetrics().map(m => m.min_date));
    },

    /* Identifies what a ranking was computed on: the filtered rows and the
     * common-coverage start. A different one now means the chart is stale. */
    surfSignature() {
      void this.idxTick;
      const idx = OB_DATA.idx || [];
      let h = 0;
      for (const i of idx) h = (h * 31 + i + 1) % 1000000007;
      const cov = this.surf.common ? (this.surfCoverage() || {}).commonStart : '';
      return `${idx.length}:${h}:${cov || ''}`;
    },
    surfStale() { return !!this.surf.result && this.surf.result.sig !== this.surfSignature(); },

    async rankSurface() {
      if (!OB_DATA.columns || this.surf.busy) return;
      await this.loadSurfaceCatalog();
      const cov = this.surfCoverage();
      const from = this.surf.common && cov ? cov.commonStart : null;
      const trades = obRankTrades(OB_DATA.columns, OB_DATA.idx, from);
      const sig = this.surfSignature();
      const beforeEarliest = cov ? obCommonCoverage(OB_DATA.columns.date_opened,
        OB_DATA.idx.filter(i => !from || (OB_DATA.columns.date_opened[i] || '') >= from),
        (this.surf.catalog || []).map(m => m.min_date)).beforeEarliest : 0;
      this.surf.busy = true;
      this.surf.error = '';
      try {
        const t0 = performance.now();
        const r = await fetch('/api/oo-backtest/surface/rank', {
          method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ trades }) });
        const body = await r.json().catch(() => ({}));
        if (!r.ok) throw new Error(body.detail || `HTTP ${r.status}`);
        OB_DATA.rank = body.rows;
        const ns = body.rows.map(x => x.n).filter(n => n > 0);
        this.surf.result = {
          sig, from, sent: trades.length, filtered: OB_DATA.idx.length, report: body.report,
          beforeCoverage: beforeEarliest, nMin: ns.length ? Math.min(...ns) : 0, nMax: ns.length ? Math.max(...ns) : 0,
          at: new Date().toTimeString().slice(0, 5), ms: Math.round(performance.now() - t0),
          lookahead: body.lookahead_confirmed,
        };
        this.$nextTick(() => this.renderRanking());
      } catch (e) {
        console.error('oo-backtest surface rank', e);
        this.surf.error = `Ranking failed: ${e.message}`;
      } finally {
        this.surf.busy = false;
      }
    },

    /* The prominent line: how many of the trades actually count. */
    surfHeadline() {
      const res = this.surf.result;
      if (!res) {
        const cov = this.surfCoverage();
        if (!cov || !cov.earliestStart) return '';
        const n = OB_DATA.idx.length;
        return `${cov.beforeEarliest.toLocaleString()} of ${n.toLocaleString()} filtered trades entered before the metrics in view start (${cov.earliestStart}) and will have no value`;
      }
      const rep = res.report;
      const noBar = rep.no_bar;
      const other = Math.max(0, noBar - res.beforeCoverage);
      return `${rep.with_bar.toLocaleString()} of ${res.sent.toLocaleString()} trades have a metric bar` +
        (noBar ? ` — ${noBar.toLocaleString()} don't: ${res.beforeCoverage.toLocaleString()} entered before coverage` +
                 (other ? `, ${other.toLocaleString()} with no bar at the entry time (e.g. 09:30)` : '') : '');
    },

    surfSubline() {
      const res = this.surf.result;
      if (!res) return '';
      const parts = [];
      if (res.from) parts.push(`common coverage from ${res.from}: ${res.sent.toLocaleString()} of ${res.filtered.toLocaleString()} filtered trades sent`);
      parts.push(`n per metric ${res.nMin.toLocaleString()}–${res.nMax.toLocaleString()}`);
      parts.push(`${res.report.distinct_entries.toLocaleString()} distinct entries`);
      parts.push(`computed ${res.at} in ${(res.ms / 1000).toFixed(1)}s`);
      return parts.join(' · ');
    },

    surfCommonText() {
      const cov = this.surfCoverage();
      if (!cov || !cov.commonStart) return '';
      const shown = this.surfViewMetrics().length;
      if (cov.commonStart === cov.earliestStart) return `every metric in view starts ${cov.commonStart}; no cost`;
      return `from ${cov.commonStart} (latest start of the ${shown} metrics in view): drops ${cov.dropped.toLocaleString()} ` +
             `of ${(cov.dropped + cov.kept).toLocaleString()} covered trades`;
    },

    toggleCommon() {
      this.surf.common = !this.surf.common;
    },

    surfView() {
      // The rank rows live outside the proxy (OB_DATA.rank); reading
      // surf.result -- replaced on every ranking -- is what makes the footer
      // and the BH warning re-render. Without it both stayed blank on screen
      // while every direct call returned the right text.
      void this.surf.result;
      if (!OB_DATA.rank) return null;
      return obRankView(OB_DATA.rank, { method: this.surf.method, form: this.surf.form,
                                        hidden: new Set(Object.keys(this.surf.hidden).filter(k => this.surf.hidden[k])),
                                        groups: this.surf.groups, other: this.surf.other });
    },

    surfFooter() {
      const v = this.surfView();
      if (!v) return '';
      const name = this.surf.method === 'spearman' ? 'Spearman' : 'Pearson';
      const bh = obBhBoundary(v.bars);
      const line = bh.index === null ? 'none in view survive, so no line' : 'left of the dashed line';
      return `${v.bars.length} bars · ${v.survivorsInView} in view survive Benjamini-Hochberg at q ${OB_BH_Q} (${name}; ` +
             `${v.survivorsAll} of ${v.computedAll} computed), ${line}` +
             (v.undefinedInView ? ` · ${v.undefinedInView} in view with no correlation (too few values or constant)` : '') +
             ' · opacity by n · click a bar to add its section below';
    },

    /* The prominent note for bars left of the BH line that do NOT survive.
     * Not noise: metrics sharing a coverage start share an n, so a later-
     * starting form (z-scores) sits systematically at the low end of n and
     * can fail BH at an |r| that passes for a level metric. Said with the
     * forms and the n ranges, so the cause is visible. */
    surfBhWarning() {
      const v = this.surfView();
      if (!v) return '';
      const bh = obBhBoundary(v.bars);
      if (!bh.failLeft) return '';
      const left = v.bars.slice(0, bh.index);
      const fail = left.filter(b => !b.survives), pass = left.filter(b => b.survives);
      const byForm = {};
      for (const b of fail) byForm[b.form] = (byForm[b.form] || 0) + 1;
      const labels = this.surf.formLabels || {};
      const forms = Object.entries(byForm).sort((a, b) => b[1] - a[1])
        .map(([f, k]) => `${k} ${labels[f] || f}`).join(', ');
      const range = bs => {
        const ns = bs.map(b => b.n);
        const lo = Math.min(...ns), hi = Math.max(...ns);
        return lo === hi ? lo.toLocaleString() : `${lo.toLocaleString()}–${hi.toLocaleString()}`;
      };
      const s = fail.length === 1 ? '' : 's';
      return `${fail.length} bar${s} left of the BH line do${fail.length === 1 ? 'es' : ''} NOT survive (${forms}): ` +
             `n ${range(fail)} against ${range(pass)} for the bars that do. A later coverage start means fewer trades, ` +
             `so the same |r| is weaker evidence. To compare like with like, view one form or use Common coverage only.`;
    },

    /* Tooltip text for the ranking's three statistical choices. */
    surfTip(which) {
      const n = this.surfView() ? this.surfView().computedAll : (this.surf.catalog || []).length || null;
      // Coverage starts per form, read from the catalog -- never written in.
      const starts = {};
      for (const m of this.surf.catalog || []) {
        if (m.min_date && (!starts[m.form] || m.min_date < starts[m.form])) starts[m.form] = m.min_date;
      }
      const labels = this.surf.formLabels || {};
      const startText = Object.entries(starts).sort((a, b) => (a[1] < b[1] ? -1 : 1))
        .map(([f, d]) => `${labels[f] || f} from ${d}`).join(', ');
      switch (which) {
        case 'spearman':
          return 'Spearman ρ: correlation of RANKS. Asks whether P/L tends to rise or fall as the metric rises, ' +
                 'whatever the shape of the relationship. One huge winner or an extreme metric value cannot ' +
                 'dominate it. Prefer it when a relationship may be monotone but not a straight line.';
        case 'pearson':
          return 'Pearson r: LINEAR correlation of the raw values — how well a straight line fits P/L against the ' +
                 'metric. A few large-P/L trades or extreme metric values can create or hide it. Where it disagrees ' +
                 'with Spearman, outliers or a curved relationship are usually the reason.';
        case 'bh':
          return `Benjamini-Hochberg false-discovery control at q ${OB_BH_Q}, over every metric computed` +
                 (n ? ` (${n})` : '') + ' — hidden families and other forms still count, because they were tested. ' +
                 'Bars left of the dashed line have an adjusted p below q: of those, about 5% are expected to be ' +
                 'false discoveries' + (n ? `, against ~${Math.round(n * 0.05)} of ${n} looking "significant" by chance at raw p < 0.05` : '') + '. ' +
                 'It is not a cutoff on |r|: the adjusted p also depends on each metric\'s n. The p-values assume ' +
                 'independent trades; trades sharing an entry bar share a value (see distinct bars in the hover).';
        case 'common':
          return 'Each metric is correlated over the trades that have a value for it, and metrics start on different ' +
                 'dates' + (startText ? ` (${startText})` : '') + ', so without this a z-score bar and a level ' +
                 'bar describe different periods. Common coverage only sends trades from the latest start among the ' +
                 'metrics in view, so every bar is measured on the same trades — at the cost stated beside it.';
        default:
          return '';
      }
    },

    renderRanking() {
      if (typeof Chart === 'undefined') return;
      const v = this.surfView();
      const el = document.getElementById('ob-rank-chart');
      const axisEl = document.getElementById('ob-rank-axis');
      const inner = document.getElementById('ob-rank-inner');
      if (!v || !el || !axisEl || !inner) return;
      const cat = new Map((this.surf.catalog || []).map(m => [m.column_name, m]));
      const BAR_PX = 12, AXIS_PX = 52;
      inner.style.width = Math.max(inner.parentElement.clientWidth, v.bars.length * BAR_PX + 16) + 'px';
      const maxAbs = Math.max(0.05, ...v.bars.map(b => Math.abs(b.value))) * 1.1;
      const lim = Math.ceil(maxAbs * 20) / 20;
      const yScale = { min: -lim, max: lim, border: { display: false } };
      const method = this.surf.method;
      const tipLines = b => {
        const m = cat.get(b.column) || {};
        const pr = `Pearson r ${b.pearson === null ? '—' : b.pearson.toFixed(3)} (p ${obP(b.pearson_p)}, BH ${obP(b.pearson_p_bh)})`;
        const sp = `Spearman ρ ${b.spearman === null ? '—' : b.spearman.toFixed(3)} (p ${obP(b.spearman_p)}, BH ${obP(b.spearman_p_bh)})`;
        return [...obWrap(m.description, 60), ...(m.formula ? obWrap('= ' + m.formula, 60) : []),
                `n ${b.n.toLocaleString()} · ${b.bars.toLocaleString()} distinct bars · from ${m.min_date || '—'}`,
                method === 'spearman' ? sp + (b.survives ? '  ✓ BH' : '') : sp,
                method === 'pearson' ? pr + (b.survives ? '  ✓ BH' : '') : pr,
                `${b.family} · ${b.form}${b.tenor ? ' · ' + b.tenor : ''}${b.wing ? ' · ' + b.wing : ''}`];
      };
      // The BH boundary: a faint vertical line after the last surviving bar,
      // drawn by a per-chart plugin (no outline on the bars themselves).
      const bh = obBhBoundary(v.bars);
      const bhLine = {
        id: 'obBhLine',
        afterDatasetsDraw(chart) {
          if (bh.index === null) return;
          const meta = chart.getDatasetMeta(0);
          const a = meta.data[bh.index - 1], b = meta.data[bh.index];
          if (!a) return;
          const x = b ? (a.x + b.x) / 2 : a.x + a.width;
          const { top, bottom } = chart.chartArea;
          const ctx = chart.ctx;
          ctx.save();
          ctx.strokeStyle = 'rgba(255,255,255,0.28)';
          ctx.lineWidth = 1;
          ctx.setLineDash([3, 3]);
          ctx.beginPath(); ctx.moveTo(x + 0.5, top); ctx.lineTo(x + 0.5, bottom); ctx.stroke();
          ctx.setLineDash([]);
          ctx.fillStyle = 'rgba(255,255,255,0.45)';
          ctx.font = '10px sans-serif';
          ctx.fillText(`BH q ${OB_BH_Q}`, x + 4, top + 10);
          ctx.restore();
        },
      };
      const cfg = {
        type: 'bar',
        plugins: [bhLine],
        data: { labels: v.bars.map(b => b.column), datasets: [{
          data: v.bars.map(b => b.value),
          backgroundColor: v.bars.map(b => obRgba(b.group.color, b.alpha)),
          borderSkipped: false, barPercentage: 0.8, categoryPercentage: 1.0 }] },
        options: {
          responsive: true, maintainAspectRatio: false, animation: false,
          layout: { padding: { top: 6, bottom: 6 } },
          // Click a bar: add that metric's section row below the ranking.
          onClick: (evt, _els, chart) => {
            const hit = (chart || OB_CHARTS.rank).getElementsAtEventForMode(evt, 'nearest', { intersect: true }, false);
            if (hit.length) this.surfClickBar(hit[0].index);
          },
          onHover: (evt, els, chart) => { (chart || OB_CHARTS.rank).canvas.style.cursor = els.length ? 'pointer' : 'default'; },
          scales: {
            x: { display: false },
            y: { ...yScale, ticks: { display: false }, afterFit: sc => { sc.width = 0; },
                 grid: { color: ctx => (ctx.tick && ctx.tick.value === 0 ? 'rgba(255,255,255,0.35)' : 'rgba(255,255,255,0.05)') } },
          },
          plugins: { legend: { display: false },
                     tooltip: { callbacks: { title: it => v.bars[it[0].dataIndex].column,
                                             label: it => tipLines(v.bars[it.dataIndex]) } } },
        },
      };
      // The y axis lives in its own narrow chart OUTSIDE the scroller, so it
      // stays put while the bars scroll. Same scale, padding and no x axis,
      // so the two plot areas share their top and bottom exactly.
      const axisCfg = {
        type: 'bar', data: { labels: [''], datasets: [{ data: [null] }] },
        options: { responsive: true, maintainAspectRatio: false, animation: false,
                   layout: { padding: { top: 6, bottom: 6 } },
                   scales: { x: { display: false },
                             y: { ...yScale, grid: { display: false },
                                  afterFit: sc => { sc.width = AXIS_PX; },
                                  ticks: { color: '#9a9a9a', font: { size: 10 }, callback: x => x.toFixed(2) } } },
                   plugins: { legend: { display: false }, tooltip: { enabled: false } } },
      };
      for (const [key, node, c] of [['rank', el, cfg], ['rankAxis', axisEl, axisCfg]]) {
        const ch = OB_CHARTS[key];
        // Plugins are fixed at construction, and the BH line closes over this
        // render's bars: rebuild the bar chart rather than update it.
        if (ch && (ch.canvas !== node || key === 'rank')) { ch.destroy(); OB_CHARTS[key] = null; }
        if (OB_CHARTS[key]) { OB_CHARTS[key].data = c.data; OB_CHARTS[key].options = c.options; OB_CHARTS[key].resize(); OB_CHARTS[key].update('none'); }
        else OB_CHARTS[key] = new Chart(node.getContext('2d'), c);
      }
    },


    /* ── added surface metric rows (P6c) ───────────────────────────────── */

    /* A metric picked from the ranking (bar click or dropdown) becomes a
     * section row shaped like a registry entry. Its values are fetched ONCE for
     * every trade in the log, stored as a client column (surface__<name>) and
     * scaled to display units; from then on the row is binned, filtered and
     * drawn by the same code as the built-in sections, with no request. A new
     * log refetches each row's values (the old ones belong to other trades). */
    builtinSections() { return this.registry.filter(m => m.section); },
    addedSections() { return this.surfRows; },

    surfUnitFormat(units) {
      return (this.surf.unitFormats || {})[units] || this.surf.defaultUnitFormat || { scale: 1, decimals: 4, suffix: '' };
    },

    surfRowFor(column) { return this.surfRows.find(r => r.surfColumn === column) || null; },

    /* The dropdown: families in legend order, each an optgroup of its metrics. */
    surfOptionGroups() {
      const cat = this.surf.catalog || [];
      const out = [];
      for (const g of this.surfaceGroups()) {
        for (const f of g.families) {
          const ms = cat.filter(m => m.family === f).sort((a, b) => (a.column_name < b.column_name ? -1 : 1));
          out.push({ label: `${g.label} · ${f}`, options: ms.map(m => ({
            value: m.column_name, added: !!this.surfRowFor(m.column_name),
            label: `${m.column_name}${m.description ? ' — ' + String(m.description).slice(0, 70) : ''}` })) });
        }
      }
      return out;
    },

    surfClickBar(index) {
      const v = this.surfView();
      if (v && v.bars[index]) this.addSurfRow(v.bars[index].column);
    },

    async addSurfRow(column) {
      if (!column || !OB_DATA.columns) return;
      await this.loadSurfaceCatalog();
      const existing = this.surfRowFor(column);
      if (existing) { this.scrollToRow(existing); return; }
      const meta = (this.surf.catalog || []).find(m => m.column_name === column);
      if (!meta) { this.surf.error = `Unknown surface metric ${column}`; return; }
      const u = this.surfUnitFormat(meta.units);
      const row = {
        key: `surf_${column}`, label: column, description: meta.description || '', surfColumn: column,
        column: `surface__${column}`, units: meta.units, type: 'range', binning: 'auto',
        auto: this.surf.rowAutoBins || { steps: 'nice', targetBins: 24, pLo: 1, pHi: 99 },
        bins: { edges: null, labels: null, closed: 'left', labelEdge: 'both' }, categories: null,
        hasScatter: true, section: true, filter: false, winRate: false, pane: null, basis: null, series: [],
        format: { decimals: u.decimals, suffix: u.suffix }, scale: u.scale, minDate: meta.min_date,
        family: meta.family, form: meta.form, removable: true, loading: true, error: '', report: null, rowFilter: null,
      };
      this.surfRows = [...this.surfRows, row];
      await this.fetchSurfRow(row.key);
      const added = this.surfRows.find(r => r.key === row.key);
      if (added) this.$nextTick(() => this.scrollToRow(added));
    },

    async fetchSurfRow(key) {
      const row = this.surfRows.find(r => r.key === key);
      if (!row || !OB_DATA.columns) return;
      const token = OB_DATA.logToken;
      row.loading = true;
      row.error = '';
      const cols = OB_DATA.columns;
      const trades = cols.date_opened.map((d, i) => [d, (cols.time_opened || [])[i] ?? null]);
      try {
        const r = await fetch('/api/oo-backtest/surface/values', {
          method: 'POST', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ column: row.surfColumn, trades }) });
        const body = await r.json().catch(() => ({}));
        if (!r.ok) throw new Error(body.detail || `HTTP ${r.status}`);
        if (token !== OB_DATA.logToken || !this.surfRows.includes(row)) return;   // another log, or removed meanwhile
        if (body.values.length !== cols.date_opened.length) {
          throw new Error(`server returned ${body.values.length} values for ${cols.date_opened.length} trades`);
        }
        const vals = body.values.map(v => (v === null || v === undefined ? null : v * row.scale));
        cols[row.column] = vals;
        this.presentColumns = this.presentColumns.filter(c => c !== row.column)
          .concat(vals.some(v => v !== null) ? [row.column] : []);
        row.report = { withValue: vals.filter(v => v !== null).length, trades: vals.length, noBar: body.report.no_bar };
        this.computeAutoBins();
        this.initRowFilter(row);
      } catch (e) {
        console.error('oo-backtest surface values', e);
        if (token === OB_DATA.logToken) row.error = `Could not fetch ${row.surfColumn}: ${e.message}`;
      } finally {
        if (token === OB_DATA.logToken) {
          row.loading = false;
          this.$nextTick(() => this.recompute());
        }
      }
    },

    removeSurfRow(m) {
      const wasOnPage = this.rowFilterOnPage(m);
      for (const which of ['avg', 'total', 'win', 'scatter']) {
        const id = this.canvasId(m, which);
        if (OB_CHARTS.sec[id]) { OB_CHARTS.sec[id].destroy(); delete OB_CHARTS.sec[id]; }
      }
      if (OB_DATA.columns) delete OB_DATA.columns[m.column];
      this.presentColumns = this.presentColumns.filter(c => c !== m.column);
      this.surfRows = this.surfRows.filter(r => r.key !== m.key);
      delete OB_DATA.autoBins[m.key];
      const { [m.key]: _gone, ...rest } = this.sections;
      this.sections = rest;
      if (wasOnPage) this.onFilterChange();
    },

    scrollToRow(m) {
      const el = typeof document !== 'undefined' && document.getElementById && document.getElementById('ob-sec-card-' + m.key);
      if (el && el.scrollIntoView) el.scrollIntoView({ behavior: 'smooth', block: 'start' });
    },

    /* What the skipped placeholder says: an added row has no column in the
     * log, only a coverage start the log's trades may all predate. */
    skippedText(m) {
      if (m.removable) {
        return `No trade in this log has a value for ${m.label}` + (m.minDate ? ` — its data starts ${m.minDate}` : '');
      }
      return 'Skipped — this log has no values in ' + this.metricColumn(m) +
             (this.meta && this.meta.market && !this.meta.market.joined ? ' (market data not joined)' : '');
    },

    columnLabel(m) { return m.removable ? m.label : this.metricColumn(m); },


    /* ── per-row filters (P6d) ─────────────────────────────────────────── */

    /* Each added row has its own range filter with a scope:
     *   row   (default) narrows only that row's charts; the page is untouched
     *   page  a filter like the sidebar's: every stat, chart and section, and
     *         any trade WITHOUT a value for the metric is dropped once narrowed.
     * Page scope is the old app's "Filter Scope" trap -- a z-score filter
     * silently discarding every pre-2021 trade -- so its cost is stated while
     * the page scope is selected, before and while it narrows. */
    initRowFilter(row) {
      const prev = row.rowFilter ? row.rowFilter.scope : null;
      row.rowFilter = obRowFilterState((OB_DATA.columns || {})[row.column], this.autoSteps[row.key], prev);
    },

    rowFilterActive(m) { const f = m.rowFilter; return !!f && (f.lo > f.min || f.hi < f.max); },
    rowFilterOnPage(m) { return m.rowFilter && m.rowFilter.scope === 'page' && this.rowFilterActive(m); },

    setRowLo(m, raw) { const f = m.rowFilter; if (!f) return; f.lo = Math.min(+raw, f.hi); this.onFilterChange(); },
    setRowHi(m, raw) { const f = m.rowFilter; if (!f) return; f.hi = Math.max(+raw, f.lo); this.onFilterChange(); },
    resetRowFilter(m) { const f = m.rowFilter; if (!f) return; f.lo = f.min; f.hi = f.max; this.onFilterChange(); },
    setRowScope(m, scope) {
      if (!m.rowFilter || m.rowFilter.scope === scope) return;
      m.rowFilter.scope = scope;
      this.onFilterChange();
    },

    /* The page's filtered rows WITHOUT this row's own filter -- what the
     * row's cost and its row-scope narrowing are measured against. */
    idxWithout(m) {
      const own = m.column;
      const specs = this.activeSpecs().filter(sp => sp.column !== own);
      return obApplyFilters(OB_DATA.columns, OB_DATA.n, specs);
    },

    /* What whole-page scope costs through the coverage gap: trades the page
     * would otherwise show that have no value for this metric, split into
     * "entered before its data starts" and "no bar at the entry time". */
    rowScopeCost(m) {
      void this.idxTick;
      if (!m.rowFilter || !OB_DATA.columns || !OB_DATA.columns[m.column]) return null;
      const vals = OB_DATA.columns[m.column], dates = OB_DATA.columns.date_opened;
      const base = this.idxWithout(m);
      let before = 0, noBar = 0;
      for (const i of base) {
        if (vals[i] !== null && vals[i] !== undefined) continue;
        if (m.minDate && dates[i] && dates[i] < m.minDate) before++; else noBar++;
      }
      return { base: base.length, dropped: before + noBar, before, noBar };
    },

    rowScopeText(m) {
      const c = this.rowScopeCost(m);
      if (!c || m.rowFilter.scope !== 'page') return '';
      const why = [c.before ? `${c.before.toLocaleString()} entered before its data starts (${m.minDate})` : '',
                   c.noBar ? `${c.noBar.toLocaleString()} with no bar at the entry time` : ''].filter(Boolean).join(', ');
      if (!c.dropped) return 'Whole page: every filtered trade has a value — no coverage cost';
      const verb = this.rowFilterActive(m) ? 'dropping' : 'moving the slider will drop';
      return `Whole page: ${verb} ${c.dropped.toLocaleString()} of ${c.base.toLocaleString()} filtered trades with no value — ${why}`;
    },

    rowFilterText(m) {
      const f = m.rowFilter;
      if (!f) return '';
      const u = m.format && m.format.suffix ? ' ' + m.format.suffix : '';
      const dec = Math.max(m.format.decimals ?? 2, obDecimals(f.step));
      return `${f.lo.toFixed(dec)} – ${f.hi.toFixed(dec)}${u}`;
    },

    /* ── capital per position ──────────────────────────────────────────── */

    /* The entered amount, or null when blank / not a positive number. */
    capital() {
      const raw = String(this.capitalInput ?? '');
      const v = Number(raw.replace(/[$,\s]/g, ''));
      return raw.trim() !== '' && Number.isFinite(v) && v > 0 ? v : null;
    },

    /* Everything capital feeds, and nothing else: no re-filter, no re-bin. */
    recomputeCapital() {
      if (!OB_DATA.columns || !this.stats) return;
      const conc = OB_DATA.conc;
      this.extra = obExtraStats(OB_DATA.columns, OB_DATA.idx, this.stats, this.capital(), conc ? conc.peak : 0);
      this.deploy = conc && { peak: conc.peak, peakDay: conc.peakDay, offSession: conc.offSession,
                              unclosed: conc.unclosed, days: conc.days.length, hasSessions: OB_DATA.sessions.length > 0 };
      this.renderDeployment();
    },

    /* On commit (change, not every keystroke): a saved strategy keeps it. */
    async commitCapital() {
      const sv = this.meta && this.meta.saved;
      this.capitalMsg = '';
      if (!sv) return;
      const cap = this.capital();
      if (cap === null && String(this.capitalInput ?? '').trim() !== '') {
        this.capitalMsg = 'Not saved — enter a positive dollar amount';
        return;
      }
      try {
        const r = await fetch(`/api/oo-backtest/strategies/${sv.id}/capital`, {
          method: 'PUT', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ capital_per_position: cap }) });
        const body = await r.json().catch(() => ({}));
        if (!r.ok) throw new Error(body.detail || `HTTP ${r.status}`);
        this.meta = { ...this.meta, saved: { ...sv, capital_per_position: body.strategy.capital_per_position } };
        this.capitalMsg = `Saved with "${sv.name}"`;
      } catch (e) {
        console.error('oo-backtest capital', e);
        this.capitalMsg = `Not saved: ${e.message}`;
      }
    },

    deployTitle() {
      const d = this.deploy;
      if (!d || !d.peak) return '';
      const cap = this.capital();
      return `peak ${d.peak} on ${d.peakDay}` + (cap ? ` · ${obMoney(d.peak * cap)}` : '');
    },

    deployNote() {
      const d = this.deploy;
      if (!d) return '';
      if (!d.hasSessions) return 'No session list — market data not joined';
      const parts = [];
      if (d.offSession) parts.push(`${d.offSession} trade${d.offSession === 1 ? '' : 's'} open or close on a day with no SPX session`);
      if (d.unclosed) parts.push(`${d.unclosed} trade${d.unclosed === 1 ? '' : 's'} with no exit date, not counted`);
      return parts.join(' · ');
    },

    renderDeployment() {
      if (typeof Chart === 'undefined') return;
      const el = document.getElementById('ob-deploy-chart');
      const conc = OB_DATA.conc;
      if (OB_CHARTS.deploy && OB_CHARTS.deploy.canvas !== el) { OB_CHARTS.deploy.destroy(); OB_CHARTS.deploy = null; }
      if (!el || !conc) return;
      const cap = this.capital();
      const pts = conc.days.map((d, k) => ({ x: obDay(d), y: conc.counts[k], d }));
      // ONE line. The dollar axis is the position axis times capital: both
      // scales are pinned to the same range, so the right one only relabels.
      // A whole-number step from 1-2-5 giving about five ticks, the top rounded
      // up to it; the dollar axis ticks at the same step times capital, so the
      // two sets of labels sit level.
      const step = [1, 2, 5, 10, 20, 50, 100, 200, 500].find(s => conc.peak * 1.05 / s <= 5) || 1000;
      const yMax = Math.max(step, Math.ceil(conc.peak * 1.05 / step) * step);
      const tick = { color: '#9a9a9a', font: { size: 10 } };
      const xr = pts.length ? { min: pts[0].x, max: pts[pts.length - 1].x } : {};
      const cfg = {
        type: 'line',
        data: { datasets: [{ data: pts, stepped: true, borderColor: OB_BLUE, backgroundColor: 'rgba(52,152,219,0.10)',
                             fill: 'origin', borderWidth: 1, pointRadius: 0, pointHitRadius: 6 }] },
        options: {
          responsive: true, maintainAspectRatio: false, animation: false, parsing: false,
          interaction: { mode: 'nearest', axis: 'x', intersect: false },
          scales: {
            x: { type: 'linear', ...xr, grid: { color: 'rgba(255,255,255,0.05)' }, border: { display: false },
                 ticks: { ...tick, maxTicksLimit: 6, callback: v => obIsoDay(v).slice(0, 7) } },
            y: { min: 0, max: yMax, grid: { color: 'rgba(255,255,255,0.05)' }, border: { display: false },
                 ticks: { ...tick, precision: 0, stepSize: step },
                 title: { display: true, text: 'positions', color: '#9a9a9a', font: { size: 10 } } },
            y2: { display: !!cap, position: 'right', min: 0, max: yMax * (cap || 1), grid: { display: false },
                  border: { display: false }, ticks: { ...tick, stepSize: step * (cap || 1), callback: v => obMoney(v) } },
          },
          plugins: { legend: { display: false },
                     tooltip: { callbacks: { title: it => it[0].raw.d,
                                             label: it => `${it.raw.y} open` + (cap ? ` · ${obMoney(it.raw.y * cap)} deployed` : '') } } },
        },
      };
      if (OB_CHARTS.deploy) { OB_CHARTS.deploy.data = cfg.data; OB_CHARTS.deploy.options = cfg.options; OB_CHARTS.deploy.update('none'); }
      else OB_CHARTS.deploy = new Chart(el.getContext('2d'), cfg);
    },

    fmtVal(m, v) { return obFmt(v, m.format); },
    moneyText(v) { return obMoney(v); },

    rangeSummary(m) {
      const r = this.rangeOf(m);
      return r ? `${r.n.toLocaleString()} trades with a value · data ${obFmt(r.min, m.format)} … ${obFmt(r.max, m.format)}` : '';
    },

    /* ── summary stats ─────────────────────────────────────────────────── */

    stat(key) {
      const st = this.stats;
      if (!st) return '—';
      switch (key) {
        case 'num_trades': return st.num_trades.toLocaleString();
        case 'win_pct': return st.win_pct.toFixed(1) + '%';
        case 'avg_days_in_trade': return st.avg_days_in_trade.toFixed(1);
        case 'avg_annual_pnl': return this.extraVal(key) === null ? '—' : obMoney(this.extraVal(key));
        case 'calmar': return this.extraVal(key) === null ? '—' : this.extraVal(key).toFixed(2);
        case 'profit_factor': {
          const v = this.extraVal(key);
          return v === null ? '—' : v === Infinity ? '∞' : v.toFixed(2);
        }
        // Blank, not "—", until a capital amount is entered.
        case 'avg_annual_return_pct':
        case 'avg_pnl_pct': {
          const v = this.extraVal(key);
          return v === null ? (this.capital() === null ? '' : '—') : v.toFixed(2) + '%';
        }
        default: return obMoney(st[key], ['avg_pnl', 'avg_win_pnl', 'avg_loss_pnl'].includes(key) ? 2 : 0);
      }
    },

    statNum(key) {
      if (this.stats && key in this.stats) return this.stats[key];
      const v = this.extraVal(key);
      // Profit factor reads blue at or above 1, pink below.
      if (key === 'profit_factor') return v === null ? 0 : v - 1;
      return v === null ? 0 : v;
    },

    extraVal(key) { return this.extra && this.extra[key] !== undefined ? this.extra[key] : null; },

    /* ── performance charts ────────────────────────────────────────────── */

    maxDDLine() {
      const st = this.stats;
      if (!st || !st.num_trades) return '';
      const eq = obEquity(OB_DATA.columns, OB_DATA.idx);
      return eq.maxDD ? `max ${obMoney(eq.maxDD.drawdown)} on ${eq.maxDD.date}` : 'no drawdown';
    },

    renderPerformance(eq) {
      if (typeof Chart === 'undefined') return;
      const cumEl = document.getElementById('ob-cum-chart');
      const ddEl = document.getElementById('ob-dd-chart');
      if (!cumEl || !ddEl) return;
      const pts = eq.points;
      const cum = pts.map(p => ({ x: obDay(p.date), y: p.cumulative, p }));
      const dd = pts.map(p => ({ x: obDay(p.date), y: p.drawdown, p }));
      const mark = eq.maxDD ? [{ x: obDay(eq.maxDD.date), y: eq.maxDD.drawdown, p: eq.maxDD }] : [];

      const axis = (money) => ({
        type: 'linear',
        grid: { color: 'rgba(255,255,255,0.05)' },
        border: { display: false },
        ticks: { color: '#9a9a9a', font: { size: 10 }, maxTicksLimit: money ? 6 : 7,
                 callback: money ? (v => obMoney(v)) : (v => obIsoDay(v).slice(0, 7)) },
      });
      // The x axis spans exactly the plotted dates; Chart.js would otherwise
      // round a linear scale out to "nice" values months beyond the data.
      const xr = cum.length ? { min: cum[0].x, max: cum[cum.length - 1].x } : {};
      const base = (tooltipLabel) => ({
        responsive: true, maintainAspectRatio: false, animation: false, parsing: false,
        interaction: { mode: 'nearest', axis: 'x', intersect: false },
        scales: { x: { ...axis(false), ...xr }, y: axis(true) },
        plugins: {
          legend: { display: false },
          tooltip: { callbacks: { title: it => it[0].raw.p.date, label: tooltipLabel } },
        },
      });

      const cumData = { datasets: [{
        data: cum, borderColor: OB_BLUE, backgroundColor: 'rgba(52,152,219,0.10)', fill: 'origin',
        borderWidth: 1, pointRadius: 0, pointHitRadius: 6, tension: 0 }] };
      const ddData = { datasets: [
        { data: dd, borderColor: OB_PINK, backgroundColor: 'rgba(232,67,147,0.16)', fill: 'origin',
          borderWidth: 1, pointRadius: 0, pointHitRadius: 6, tension: 0 },
        // The deepest point: a marker with a surface-coloured ring so it reads
        // over the line, and hover-only like the rest.
        { data: mark, type: 'scatter', pointRadius: 5, pointHoverRadius: 6, pointBackgroundColor: OB_PINK,
          pointBorderColor: '#2d2d2d', pointBorderWidth: 2, showLine: false },
      ] };

      if (OB_CHARTS.cum) { OB_CHARTS.cum.data = cumData; Object.assign(OB_CHARTS.cum.options.scales.x, xr); OB_CHARTS.cum.update('none'); }
      else {
        OB_CHARTS.cum = new Chart(cumEl.getContext('2d'), { type: 'line', data: cumData,
          options: base(c => `Cumulative ${obMoney(c.raw.p.cumulative)} · trade ${obMoney(c.raw.p.pnl)}`) });
      }
      if (OB_CHARTS.dd) { OB_CHARTS.dd.data = ddData; Object.assign(OB_CHARTS.dd.options.scales.x, xr); OB_CHARTS.dd.update('none'); }
      else {
        OB_CHARTS.dd = new Chart(ddEl.getContext('2d'), { type: 'line', data: ddData,
          options: base(c => (c.datasetIndex === 1 ? 'Max drawdown ' : 'Drawdown ') +
                             `${obMoney(c.raw.p.drawdown)} · peak ${obMoney(c.raw.p.peak)}`) });
      }
    },

    /* ── registry-driven views ─────────────────────────────────────────── */

    filterMetrics() { return this.registry.filter(m => m.filter); },
    sectionMetrics() { return [...this.registry.filter(m => m.section), ...this.surfRows]; },

    /* The column a metric reads now: its basis column where it has more than
     * one (the ratios), else its only column. */
    metricColumn(m) { return m.basis ? m.basis[this.ratioBasis] : m.column; },

    hasColumn(m) { return this.presentColumns.includes(this.metricColumn(m)); },

    hasBasisMetrics() { return this.registry.some(m => m.basis); },

    /* Trades in the loaded log that open before a metric's coverage. A range
     * filter drops trades with no value, so filtering on this metric would
     * drop these too -- said before it happens, not discovered after. */
    tradesBeforeCoverage(m) {
      if (!this.loaded || !m.minDate || !OB_DATA.columns) return 0;
      let k = 0;
      for (const d of OB_DATA.columns.date_opened) if (d && d < m.minDate) k++;
      return k;
    },

    coverageNote(m) {
      const k = this.tradesBeforeCoverage(m);
      if (!k) return '';
      const s = k === 1 ? '' : 's';
      return this.isActive(m)
        ? `Data starts ${m.minDate} — this filter is dropping ${k.toLocaleString()} earlier trade${s}`
        : `Data starts ${m.minDate} — filtering on this drops ${k.toLocaleString()} earlier trade${s}`;
    },

    /* The bins a range metric uses now: the registry's fixed spec, or for
     * binning 'auto' the edges built from THIS log (all of it, not the
     * filtered rows -- a filter must not move the edges under the bars). */
    binsFor(m) { return m.binning === 'auto' ? (OB_DATA.autoBins[m.key] || null) : m.bins; },

    computeAutoBins() {
      OB_DATA.autoBins = {};
      const steps = {};
      for (const m of [...this.registry, ...this.surfRows].filter(x => x.binning === 'auto')) {
        const b = obAutoBins((OB_DATA.columns || {})[this.metricColumn(m)], m.auto, m.format);
        if (b) { OB_DATA.autoBins[m.key] = b; steps[m.key] = b.step; }
      }
      this.autoSteps = steps;
    },

    binCount(m) { const b = this.binsFor(m); return b ? b.labels.length : null; },

    binsTitle(m) {
      const b = this.binsFor(m);
      if (!b) return m.type === 'range' ? '' : 'categorical';
      return `${b.labels.length} bins, ${b.closed}-closed` + (b.labelEdge === 'left' ? ', labelled by left edge' : '')
        + (m.binning === 'auto' ? `, auto from this log's p${m.auto.pLo}–p${m.auto.pHi}` : '');
    },

    /* ── metric sections ───────────────────────────────────────────────── */

    /* Three states, each worded differently on screen so none reads as a
     * broken chart:
     *   skipped  the LOG has no values in the column (absent or all null)
     *   nodata   the log has values, but no trade in the current filter does
     *   ready    charts drawn from `sections[key]`
     * Filled by renderSections(); a section recompute never makes a request. */
    sectionState(m) {
      if (!this.loaded) return 'empty';
      if (m.loading) return 'loading';
      if (m.error) return 'error';
      if (!this.hasColumn(m)) return 'skipped';
      const s = this.sections[m.key];
      return s && s.valued ? 'ready' : 'nodata';
    },

    sectionSub(m) {
      const s = this.sections[m.key];
      const col = this.columnLabel(m);
      if (m.removable && m.report && (!s || !s.valued)) {
        return `${m.report.withValue.toLocaleString()} of ${m.report.trades.toLocaleString()} trades in the log have a value`;
      }
      if (!s || !s.valued) return col;
      const of = s.valued === this.filteredCount ? '' : ` of ${this.filteredCount.toLocaleString()}`;
      // An auto-binned metric names its step: two logs with different steps
      // are not bar-for-bar comparable, and this is where that shows.
      const auto = m.binning === 'auto' && this.autoSteps[m.key] !== undefined
        ? ` · ${this.stepText(m)} bins (auto)` : '';
      const cover = (m.removable && m.minDate ? ` · data from ${m.minDate}` : '') +
        (m.rowFilter && m.rowFilter.scope === 'row' && this.rowFilterActive(m) ? ' · row filter applied' : '');
      return `${col} · ${s.valued.toLocaleString()}${of} trades with a value · ${m.type === 'range' ? `${s.bins} of ${this.binCount(m)} bins filled` : `${s.bins} values`}${auto}${cover}`;
    },

    stepText(m) {
      const step = this.autoSteps[m.key];
      if (step === undefined) return '';
      if (m.format && typeof m.format === 'object') {
        const t = step.toFixed(Math.max(m.format.decimals ?? 0, obDecimals(step)));
        return m.format.suffix ? `${t} ${m.format.suffix}` : t;
      }
      return obFmt(step, m.format);
    },

    fitText(m) {
      const f = this.sections[m.key] && this.sections[m.key].fit;
      if (!f) return 'no fit (fewer than 3 points, or one x value)';
      // Cents where the slope is small: premium's is well under $1 per $1.
      // An added surface row states the slope per ONE BIN STEP: "per 1 unit"
      // of a metric whose whole range is 0.008 reads as -$22,589.
      const step = this.autoSteps[m.key];
      if (m.format && typeof m.format === 'object' && step) {
        const d = f.slope * step;
        const slope = obMoney(d, Math.abs(d) < 10 ? 2 : 0) + ' per ' + this.stepText(m);
        return f.r === null ? `slope ${slope}` : `r ${f.r.toFixed(3)} · R² ${f.r2.toFixed(3)} · slope ${slope}`;
      }
      const per = m.format === 'pct' ? '1%' : m.format === 'ratio' ? '1.0' : m.format === 'usd' ? '$1' : '1 pt';
      const slope = obMoney(f.slope, Math.abs(f.slope) < 10 ? 2 : 0) + ' per ' + per;
      return f.r === null ? `slope ${slope}` : `r ${f.r.toFixed(3)} · R² ${f.r2.toFixed(3)} · slope ${slope}`;
    },

    canvasId(m, which) { return `ob-sec-${m.key}-${which}`; },

    /* "bin" for a range metric; the category's own name otherwise
     * ("Day of Week" -> "day of week", "P&L by Year" -> "year"). */
    binNoun(m) { return m.type === 'range' ? 'bin' : m.label.replace(/^P&L by /, '').toLowerCase(); },

    renderSections(idx) {
      const sections = {};
      for (const m of this.sectionMetrics()) {
        if (!this.hasColumn(m)) continue;
        const bins = m.type === 'range' ? this.binsFor(m) : null;
        if (m.type === 'range' && !bins) continue;
        const mm = bins === m.bins ? m : { ...m, bins };
        // A row-scoped filter narrows this section only.
        let rowIdx = idx;
        if (m.rowFilter && m.rowFilter.scope === 'row' && this.rowFilterActive(m)) {
          const vals = OB_DATA.columns[m.column], f = m.rowFilter;
          rowIdx = idx.filter(i => vals[i] !== null && vals[i] >= f.lo && vals[i] <= f.hi);
        }
        const d = obSectionData(OB_DATA.columns, rowIdx, mm, this.metricColumn(m));
        const fit = m.hasScatter ? obOLS(d.xs, d.ys) : null;
        sections[m.key] = { valued: d.valued, bins: d.rows.filter(r => r.count).length, fit };
        if (d.valued) this.drawSection(m, d, fit);
      }
      this.sections = sections;
    },

    drawSection(m, d, fit) {
      if (typeof Chart === 'undefined') return;
      // Sign picks the hue; the bin's trade count picks the opacity, so a
      // thin bin recedes without losing its profit/loss colour.
      const maxCount = Math.max(0, ...d.rows.map(r => r.count));
      const signColor = (v, k) => obRgba(v >= 0 ? OB_BLUE : OB_PINK, obBarAlpha(d.rows[k].count, maxCount));
      const grid = { color: 'rgba(255,255,255,0.05)' };
      const tick = { color: '#9a9a9a', font: { size: 10 } };
      const labels = d.rows.map(r => r.label);
      const rowTip = r => [`${r.count.toLocaleString()} trade${r.count === 1 ? '' : 's'}`,
                           `Avg ${obMoney(r.avg, 2)} · Total ${obMoney(r.total)}`, `Win ${r.win.toFixed(1)}%`];
      const bar = (values, yTick, color) => ({
        type: 'bar',
        data: { labels, datasets: [{ data: values, backgroundColor: values.map((v, k) => (v === null ? 'transparent' : color(v, k))),
                                     borderRadius: 4, borderSkipped: 'start', maxBarThickness: 36 }] },
        options: {
          responsive: true, maintainAspectRatio: false, animation: false,
          scales: {
            x: { grid: { display: false }, border: { display: false },
                 ticks: { ...tick, autoSkip: true, maxRotation: 60, minRotation: 0 } },
            y: { grid, border: { display: false }, ticks: { ...tick, maxTicksLimit: 6, callback: yTick } },
          },
          plugins: { legend: { display: false },
                     tooltip: { callbacks: { title: it => it[0].label, label: it => rowTip(d.rows[it.dataIndex]) } } },
        },
      });
      this.upsertChart(this.canvasId(m, 'avg'), bar(d.rows.map(r => r.avg), v => obMoney(v), signColor));
      this.upsertChart(this.canvasId(m, 'total'), bar(d.rows.map(r => r.total), v => obMoney(v), signColor));
      if (m.winRate) {
        const cfg = bar(d.rows.map(r => r.win), v => v + '%',
                        (v, k) => obRgba(OB_BLUE, obBarAlpha(d.rows[k].count, maxCount)));
        Object.assign(cfg.options.scales.y, { min: 0, max: 100 });
        this.upsertChart(this.canvasId(m, 'win'), cfg);
      }
      if (!m.hasScatter) return;

      const cols = OB_DATA.columns;
      const pts = d.xs.map((x, k) => ({ x, y: d.ys[k], row: d.rowsIdx[k] }));
      let lo = Infinity, hi = -Infinity;
      for (const x of d.xs) { if (x < lo) lo = x; if (x > hi) hi = x; }
      const line = fit ? [{ x: lo, y: fit.intercept + fit.slope * lo }, { x: hi, y: fit.intercept + fit.slope * hi }] : [];
      this.upsertChart(this.canvasId(m, 'scatter'), {
        type: 'scatter',
        data: { datasets: [
          { data: pts, pointRadius: 2.5, pointHoverRadius: 5, pointHitRadius: 4, borderWidth: 0,
            pointBackgroundColor: pts.map(p => (p.y >= 0 ? 'rgba(52,152,219,0.55)' : 'rgba(232,67,147,0.55)')) },
          { data: line, type: 'line', borderColor: '#e0e0e0', borderWidth: 2, borderDash: [5, 4],
            pointRadius: 0, pointHitRadius: 0, fill: false },
        ] },
        options: {
          responsive: true, maintainAspectRatio: false, animation: false, parsing: false,
          scales: {
            x: { type: 'linear', grid, border: { display: false }, ticks: { ...tick, maxTicksLimit: 7, callback: v => obFmt(v, m.format) } },
            y: { grid, border: { display: false }, ticks: { ...tick, maxTicksLimit: 6, callback: v => obMoney(v) } },
          },
          plugins: { legend: { display: false },
                     tooltip: { filter: it => it.datasetIndex === 0,
                                callbacks: { title: it => cols.date_opened[it[0].raw.row],
                                             label: it => `${m.label} ${obFmt(it.raw.x, m.format)} · P/L ${obMoney(it.raw.y)}` } } },
        },
      });
    },

    /* Create a section chart, or update it in place. A canvas Alpine has
     * replaced is a different element: that chart is destroyed, not updated
     * into a detached node. */
    upsertChart(id, cfg) {
      const el = document.getElementById(id);
      const old = OB_CHARTS.sec[id];
      if (old && old.canvas !== el) { old.destroy(); delete OB_CHARTS.sec[id]; }
      if (!el) return;
      const ch = OB_CHARTS.sec[id];
      if (ch) { ch.data = cfg.data; ch.options = cfg.options; ch.update('none'); }
      else OB_CHARTS.sec[id] = new Chart(el.getContext('2d'), cfg);
    },

    /* What the parser dropped or had to decide, said on load rather than left
     * for the numbers to hide. `warn` lines are about trades the stats exclude
     * or fills that may not be trustworthy; the rest say where a field came
     * from when the export's version lacked the preferred one. */
    parseNotes() {
      const n = (this.meta && this.meta.notes) || {};
      const out = [];
      const plural = (k, one, many) => `${k} ${k === 1 ? one : many}`;
      if (n.open_positions) {
        out.push({ warn: true, text: `${plural(n.open_positions, 'position', 'positions')} still open at backtest end, excluded` });
      }
      if ((n.missing_data_at_fill || []).length) {
        out.push({ warn: true, text: `${plural(n.missing_data_at_fill.length, 'trade has', 'trades have')} MissingData on the entry or exit bar — that fill may be stale` });
      }
      if ((n.pnl_mismatch || []).length) {
        out.push({ warn: true, text: `${plural(n.pnl_mismatch.length, 'trade', 'trades')}: pos_realized_pnl ≠ pos_pnl (realized used)` });
      }
      if (n.multi_signal_positions) {
        out.push({ warn: false, text: `${plural(n.multi_signal_positions, 'position', 'positions')} fired several exit signals on the exit bar; resolved by precedence (price > adjustments > time)` });
      }
      if ((n.precedence_vs_order || []).length) {
        out.push({ warn: true, text: `${plural(n.precedence_vs_order.length, 'position', 'positions')}: precedence chose a different reason than the last signal in the file — MesoSim's ordering may have changed` });
      }
      if ((n.pnl_contradicts_reason || []).length) {
        out.push({ warn: true, text: `${plural(n.pnl_contradicts_reason.length, 'trade', 'trades')}: exit reason contradicts P/L (profit target with a loss, or stop with a gain)` });
      }
      if ((n.premium_leg_mismatch || []).length) {
        out.push({ warn: true, text: `${plural(n.premium_leg_mismatch.length, 'trade', 'trades')}: entry_net_premium ≠ sum of entry leg fills` });
      }
      if (n.pnl_field === 'pos_pnl') {
        out.push({ warn: false, text: 'P/L from pos_pnl — this MesoSim version has no realized P/L field' });
      }
      if ((n.premium_field || '').includes('leg fills')) {
        out.push({ warn: false, text: 'Premium summed from entry leg fills — no entry_net_premium in this export' });
      }
      return out;
    },

    sourceLabel() {
      if (!this.meta) return '';
      return this.meta.source === 'mesosim_json' ? 'Mesosim JSON' : 'Option Omega CSV';
    },

    loadedLabel() {
      if (!this.meta) return '';
      const name = (this.meta.saved && this.meta.saved.name) || this.meta.suggested_name;
      return `${name} (${this.meta.n.toLocaleString()} trades, ` +
             `${this.meta.date_min} – ${this.meta.date_max})`;
    },

    savedNotes() {
      const out = [];
      const sv = this.meta && this.meta.saved;
      if (sv && !this.hasUploadFile) {
        out.push({ warn: false, text: `Saved strategy, saved ${String(sv.updated_at || '').slice(0, 10)}` });
      }
      const ch = this.meta && this.meta.saved_count_changed;
      if (ch) {
        out.push({ warn: true, text: `Trade count was ${ch.when_saved} when saved and is ${ch.now} now — the parser has changed since` });
      }
      return out;
    },

    /* ── market data: what the join did on this log ───────────────────── */

    /* Said on load, because each of these is invisible in the charts: a 0%
     * entry-time coverage makes every VIX metric a daily-open proxy, and an
     * off-by-one prior close shifts every gap while the page looks normal. */
    marketNotes() {
      const mk = this.meta && this.meta.market;
      if (!mk) return [];
      if (!mk.joined) {
        return [{ warn: true, text: `Market data not joined — ${mk.error || 'unknown reason'}. VIX and gap sections will show as skipped.` }];
      }
      const out = [];
      const e = mk.entry_time || {};
      out.push({
        warn: e.entry_time_found < e.trades,
        text: `Entry time found for ${(e.entry_time_found || 0).toLocaleString()} of ${(e.trades || 0).toLocaleString()} trades` +
              (e.entry_time_fallback_0930 ? ` — ${e.entry_time_fallback_0930} use the 09:30 bar` : ''),
      });
      if (e.before_open) out.push({ warn: true, text: `${e.before_open} entries before 09:30 — no bar that day, VIX levels null` });
      if (e.after_close) out.push({ warn: true, text: `${e.after_close} entries after 16:00 — check the log's time zone` });
      if (mk.trades_without_daily_row) {
        out.push({ warn: true, text: `${mk.trades_without_daily_row} trades open on a date index_ohlc has no session for` });
      }
      // Null levels WITH their reasons: a count alone cannot tell coverage
      // starting later from bars missing inside coverage.
      for (const s of ['vix', 'vix3m', 'vix9d']) {
        const r = mk.null_reasons && mk.null_reasons[s];
        if (r && r.null) {
          out.push({ warn: false, text: `${s.toUpperCase()} level null for ${r.null}: ` +
                     Object.entries(r.reasons).map(([why, n]) => `${n} ${why}`).join('; ') });
        }
      }
      const pushed = Object.entries(mk.entry_bars || {}).filter(([, v]) => v.earlier_than_entry_bar).map(([k, v]) => `${k.toUpperCase()} ${v.earlier_than_entry_bar}`);
      if (pushed.length) out.push({ warn: false, text: `Entry bar was NaN, earlier bar used: ${pushed.join(', ')}` });
      // All-null gaps are what the zero-filled weekends produced for a
      // Monday-only log, with nothing on screen saying so.
      for (const [name, v] of Object.entries(mk.gaps || {})) {
        const total = v.computed + v.null;
        if (v.null) {
          const r = mk.null_reasons && mk.null_reasons[`${name}_gap`];
          const why = r ? ' — ' + Object.entries(r.reasons).map(([k, n]) => `${n} ${k}`).join('; ') : '';
          out.push({ warn: v.null > total * 0.05,
                     text: `${name.toUpperCase()} gap computed for ${v.computed.toLocaleString()} of ${total.toLocaleString()} trades${why}` });
        }
      }
      return out;
    },

    /* ── market-data checks card (main column) ────────────────────────── */

    mk() { return (this.meta && this.meta.market) || {}; },
    hasMarketChecks() { return !!this.mk().null_reasons; },
    nullTrades() {
      const out = [];
      for (const [name, r] of Object.entries(this.mk().null_reasons || {})) {
        for (const t of r.trades || []) out.push({ name, ...t });
      }
      return out;
    },
    diagErrors() { return Object.entries(this.mk().diagnostic_errors || {}).map(([k, v]) => `${k}: ${v}`); },

    /* ── market freshness (read-only) ─────────────────────────────────── */

    marketSummary() {
      if (this.marketLoading) return 'Checking…';
      if (this.marketError) return `Status failed: ${this.marketError}`;
      if (!this.market) return '';
      if (!this.market.ok) return `index_ohlc unavailable — ${this.market.error}`;
      return `index_ohlc through ${this.market.latest_date} ${(this.market.latest_time || '').slice(0, 5)}`;
    },

    marketStale() { return !!(this.market && this.market.ok && this.market.stale); },
    staleText() {
      const m = this.market || {};
      return m.age_days === null || m.age_days === undefined
        ? 'Stale — no valid SPX bar in index_ohlc'
        : `Stale — latest valid SPX bar is ${m.age_days} days old (limit ${m.stale_after_days})`;
    },

    fallbackLines() {
      const fb = (this.market && this.market.close_fallback) || {};
      return Object.entries(fb).map(([s, v]) =>
        `${s.toUpperCase()}: ${v.early_close_days.length} early-close, ${v.full_session_count} full-session` +
        (v.full_session_count ? ` (e.g. ${v.full_session_sample.slice(0, 3).join(', ')})` : '') +
        (v.no_close_days ? `, ${v.no_close_days} days with no close` : ''));
    },

    fallbackWarn() {
      const fb = (this.market && this.market.close_fallback) || {};
      return Object.values(fb).some(v => v.full_session_count > 0);
    },

    /* How the per-series session rule classified the table. The shortest
     * session kept and the longest artifact rejected sit either side of the
     * threshold -- if either comes close to it, the threshold needs a look. */
    sessionLines() {
      const z = this.market && this.market.sessions;
      if (!z) return [];
      const out = [
        `a series has a session with ≥ ${z.min_session_bars} valid bars`,
        `${z.zero_filled_days} zero-filled days (${z.zero_filled_weekdays} weekdays)`,
        `${z.artifact_only_days} artifact-only days (bars, but no series reaches a session)` +
          (z.artifact_only.length ? `: ${z.artifact_only.slice(0, 12).map(a => a.date).join(', ')}` +
                                    (z.artifact_only.length > 12 ? ', …' : '') : ''),
      ];
      for (const [s, v] of Object.entries(z.by_series || {})) {
        const kept = v.shortest_kept ? `${v.shortest_kept.bars} (${v.shortest_kept.date})` : '—';
        const rej = v.longest_rejected ? `${v.longest_rejected.bars} (${v.longest_rejected.date})` : '—';
        let line = `${s.toUpperCase()}: ${v.sessions} sessions; shortest kept ${kept}; longest rejected ${rej}`;
        if (v.missing_on_session_days.length) {
          line += `; missing on ${v.missing_on_session_days.length} session days (e.g. ${v.missing_on_session_days.slice(0, 3).join(', ')})`;
        }
        if (v.zero_bars_in_sessions || v.nan_bars_in_sessions) {
          line += `; invalid bars in sessions: ${v.zero_bars_in_sessions} zero, ${v.nan_bars_in_sessions} NaN`;
        }
        out.push(line);
      }
      return out;
    },

    coverageLines() {
      const c = (this.market && this.market.coverage) || {};
      return Object.entries(c).map(([s, d]) => `${s.toUpperCase()} from ${d || '—'}`);
    },
  }));
});
