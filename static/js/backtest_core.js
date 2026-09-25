/* The backtest pages' shared calculations.
 *
 * ONE DEFINITION, TWO PAGES. /oo-backtest reads one saved strategy and
 * /backtest-portfolio combines several; both filter trades, both compute the
 * summary figures, and both count concurrent positions. A second copy of any
 * of that would be two answers to "what did this strategy do", and the whole
 * point of the portfolio page is that a strategy reads the same on both.
 *
 * WHAT BELONGS HERE: pure functions over the columnar trade payload --
 * filtering, ordering, statistics, the equity curve, concurrency, and the
 * formatters those need. Nothing that touches the DOM, Alpine, Chart.js or a
 * URL; anything that needs one of those belongs to a page.
 *
 * The `ob` prefix is kept rather than renamed: it is the name in the first
 * page's gates and expressions, and a rename would be a large diff for no
 * behaviour. It means "backtest", not "the OO page".
 *
 * Loaded BEFORE each page's bundle and not deferred: these are top-level
 * declarations the bundles read at load.
 *
 * NO 'use strict' DIRECTIVE, deliberately, and it matches oo_backtest.js
 * which has none either. The gates run the SHIPPED functions by eval()ing
 * this file together with a page's bundle in node: a strict eval keeps its
 * declarations to itself, so every driver would see `obStats is not defined`
 * and the only way to test the shipped code would be to stop testing the
 * shipped code. The browser loads the two as separate script tags and is
 * unaffected either way.
 */

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
function obApplyFilters(cols, n, specs, lenient) {
  const idx = [];
  outer:
  for (let i = 0; i < n; i++) {
    for (const f of specs) {
      const col = cols[f.column];
      const v = col ? col[i] : null;
      const isNull = obNull(v);
      // A FILTER THAT CANNOT BE EVALUATED MAY BE TOLD TO PASS. `lenient` is a
      // set of columns whose nulls mean "no data to judge this trade by",
      // not "judged and rejected" -- a metric before its coverage starts.
      // Dropping is still the default and the summary table always uses it;
      // this exists so a chart can draw the whole history and SHADE the
      // stretch a filter was blind to, rather than silently shortening
      // itself to wherever the metric happens to begin.
      if (isNull && lenient && lenient.has(f.column)) continue;
      if (f.kind === 'range') {
        if (isNull || v < f.lo || v > f.hi) continue outer;
      } else if (f.kind === 'set') {
        if (!f.allowed.has(v)) continue outer;
      } else if (f.kind === 'date') {
        if (isNull || (f.from && v < f.from) || (f.to && v > f.to)) continue outer;
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
 * trades (first entry to last exit). HALF-OPEN [entry, exit): a position is
 * counted on every session it is held THROUGH and not on the one it closes
 * on, because the measure is capital still at risk at the close. A trade
 * that opens and closes in one session is therefore in no count, and is
 * returned as `sameSession` so the page can say so.
 * `sessions` is the sorted ISO list from the server; days outside the span are
 * dropped. A trade with no exit date is in no count (the parser excludes
 * still-open positions; this is the backstop, and it is counted, not hidden).
 * offSession counts trades whose entry or exit date is not in the list.
 * Returns { days, counts, peak, peakDay, offSession, unclosed, sameSession,
 * counted }. */
function obConcurrency(cols, idx, sessions) {
  const out = { days: [], counts: [], peak: 0, peakDay: null, offSession: 0,
                unclosed: 0, sameSession: 0, counted: 0 };
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
    out.counted++;
    // HALF-OPEN [open, close): a position is deployed on the sessions it is
    // held THROUGH, and not on the one it closes on.
    //
    // The measure is OVERNIGHT CAPITAL — what is still at risk when the
    // market shuts. A strategy entering every Friday and closing the next
    // Friday holds ONE position at all times; counting the close day as well
    // drew two every Friday, which is the same position counted twice on the
    // day it changes hands. Changed 2026-09-25 on both pages; the old Dash
    // app's capital_deployed had this right and this one did not.
    //
    // The consequence is deliberate: a trade opened and closed inside one
    // session is never held overnight and contributes nothing here. That is
    // correct for this measure and wrong-LOOKING for a 0DTE log, so those
    // trades are counted and the pane says so rather than drawing a flat
    // zero line and leaving it to be discovered.
    const a = lower(dOpen[i]) - k0, b = lower(dClose[i]) - k0;   // sessions [a, b)
    if (b > a) { diff[a]++; diff[b]--; }
    else out.sameSession++;
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


/* Days since the epoch for an ISO date, for a linear x axis (no date adapter). */
function obDay(iso) { return Date.parse(iso + 'T00:00:00Z') / 86400000; }


function obIsoDay(d) { return new Date(Math.round(d) * 86400000).toISOString().slice(0, 10); }


function obMoney(v, digits = 0) {
  if (obNull(v)) return '—';
  const a = Math.abs(v).toLocaleString('en-US', { minimumFractionDigits: digits, maximumFractionDigits: digits });
  return (v < 0 ? '-$' : '$') + a;
}


/* Sharpe, AS THE OLD DASH APP COMPUTED IT, kept on the user's instruction.
 *
 *     mean(daily) / stdev(daily) * sqrt(252)
 *
 * where `daily` is P/L SUMMED BY CLOSE DATE over the days that HAD a close.
 * Two things about that are worth saying out loud wherever the number is
 * shown, because neither is what a reader assumes:
 *
 *   * The days between closes are not zero-filled. A strategy closing twice
 *     a week has ~100 observations a year, not 252, so multiplying by
 *     sqrt(252) annualises a sample that is not daily. The figure is
 *     therefore higher than a daily-return Sharpe of the same equity curve,
 *     and the two are not comparable.
 *   * It is a Sharpe of DOLLARS, not returns: no capital, no risk-free
 *     rate. It ranks strategies against each other on this page and means
 *     nothing next to a published Sharpe.
 *
 * Kept rather than corrected so the numbers match the app this page
 * replaces; the label carries the caveat. Sample stdev (n-1), as pandas'
 * .std() has it -- the source used pandas.
 */
function obSharpe(cols, idx) {
  if (!idx || idx.length < 2) return null;
  const dc = cols.date_closed, pnl = cols.pnl;
  const byDay = new Map();
  for (const i of idx) {
    const d = dc[i];
    if (!d) continue;
    byDay.set(d, (byDay.get(d) || 0) + pnl[i]);
  }
  const v = [...byDay.values()];
  if (v.length < 2) return null;
  const mean = v.reduce((a, b) => a + b, 0) / v.length;
  let ss = 0;
  for (const x of v) ss += (x - mean) * (x - mean);
  const sd = Math.sqrt(ss / (v.length - 1));
  return sd > 0 ? mean / sd * Math.sqrt(252) : null;
}


/* One strategy's capital deployed per session: concurrency x its capital.
 * Returned aligned to `sessions` so several strategies can be summed into a
 * portfolio series -- the PEAK of that sum is the portfolio's peak deployed
 * capital, which is NOT the sum of the per-strategy peaks unless they all
 * peak on the same day. */
function obDeployedSeries(conc, sessions, capital) {
  const out = new Array(sessions.length).fill(0);
  if (!conc || !conc.days.length || !(capital > 0)) return out;
  const at = new Map();
  for (let i = 0; i < sessions.length; i++) at.set(sessions[i], i);
  for (let k = 0; k < conc.days.length; k++) {
    const j = at.get(conc.days[k]);
    if (j !== undefined) out[j] = conc.counts[k] * capital;
  }
  return out;
}


/* One point per CLOSE DATE from a trade-level equity curve.
 *
 * The curve is computed per TRADE (obEquity) because that is where max
 * drawdown is defined and what the summary table reports; drawing every
 * trade would be ten thousand points a strategy. So each date keeps its
 * day-END cumulative and its WORST drawdown of the day -- the worst
 * trade-level point always falls on some day, so the chart's minimum equals
 * the table's Max DD exactly rather than being a slightly shallower
 * day-boundary reading of it. */
function obDailyCurve(eq) {
  const out = [];
  for (const p of eq.points) {
    const last = out.length ? out[out.length - 1] : null;
    if (last && last.date === p.date) {
      last.cumulative = p.cumulative;
      last.peak = p.peak;
      if (p.drawdown < last.drawdown) last.drawdown = p.drawdown;
    } else {
      out.push({ date: p.date, cumulative: p.cumulative, peak: p.peak,
                 drawdown: p.drawdown });
    }
  }
  return out;
}


/* P/L by calendar month, keyed "YYYY-MM", dated by CLOSE like everything
 * else on these pages. */
function obMonthlyPnl(cols, idx) {
  const out = new Map();
  for (const i of idx) {
    const d = cols.date_closed[i];
    if (!d) continue;
    const k = d.slice(0, 7);
    out.set(k, (out.get(k) || 0) + cols.pnl[i]);
  }
  return out;
}


/* ── correlation, weekly ─────────────────────────────────────────────────
 *
 * WEEKLY, NOT DAILY, and deliberately: strategies close on their own
 * schedules, so a daily series is mostly zeros and a correlation of mostly
 * zeros measures how often two strategies happened to close on the same day
 * rather than whether they move together. The old Dash app resampled its
 * matrix to weeks for exactly that reason -- and then computed its ROLLING
 * pairwise correlation daily, so the two disagreed about what a correlation
 * was. Both are weekly here (agreed 2026-09-25).
 *
 * The week is the one pandas' 'W' gives: the SUNDAY ending it, so the
 * buckets match the old app's.
 */
function obWeekEnding(iso) {
  const d = new Date(iso + 'T00:00:00Z');
  d.setUTCDate(d.getUTCDate() + (7 - d.getUTCDay()) % 7);   // forward to Sunday
  return d.toISOString().slice(0, 10);
}


/* Weekly P/L for one set of trades: Map(week-ending -> summed P/L). */
function obWeeklyPnl(cols, idx) {
  const out = new Map();
  for (const i of idx) {
    const d = cols.date_closed[i];
    if (!d) continue;
    const k = obWeekEnding(d);
    out.set(k, (out.get(k) || 0) + cols.pnl[i]);
  }
  return out;
}


/* Several strategies' weekly series aligned on one week axis.
 *
 * ZERO-FILLED, then weeks where EVERY strategy is zero are dropped -- the
 * old app's rule. A week in which nothing closed anywhere is not evidence
 * that two strategies agree; keeping those weeks pulls every correlation
 * toward +1 by padding both series with matching zeros. */
function obAlignWeekly(series) {
  const weeks = [...new Set(series.flatMap(s => [...s.keys()]))].sort();
  const cols = series.map(s => weeks.map(w => s.get(w) || 0));
  const keep = weeks.map((_, i) => cols.some(c => c[i] !== 0));
  return {
    weeks: weeks.filter((_, i) => keep[i]),
    cols: cols.map(c => c.filter((_, i) => keep[i])),
  };
}


function obPearson(xs, ys) {
  const n = Math.min(xs.length, ys.length);
  if (n < 3) return null;
  let sx = 0, sy = 0;
  for (let i = 0; i < n; i++) { sx += xs[i]; sy += ys[i]; }
  const mx = sx / n, my = sy / n;
  let num = 0, dx = 0, dy = 0;
  for (let i = 0; i < n; i++) {
    const a = xs[i] - mx, b = ys[i] - my;
    num += a * b; dx += a * a; dy += b * b;
  }
  return (dx > 0 && dy > 0) ? num / Math.sqrt(dx * dy) : null;
}


/* Ranks with TIES AVERAGED, as scipy and pandas do it. Ties are not rare
 * here -- a metric like Day of Week has five distinct values across
 * thousands of trades -- and ranking them 1..n in arbitrary order would
 * manufacture an ordering the data does not have. */
function obRank(values) {
  const idx = values.map((v, i) => i).sort((a, b) => values[a] - values[b]);
  const out = new Array(values.length);
  let i = 0;
  while (i < idx.length) {
    let j = i;
    while (j + 1 < idx.length && values[idx[j + 1]] === values[idx[i]]) j++;
    const avg = (i + j) / 2 + 1;
    for (let k = i; k <= j; k++) out[idx[k]] = avg;
    i = j + 1;
  }
  return out;
}


function obSpearman(xs, ys) {
  const n = Math.min(xs.length, ys.length);
  if (n < 3) return null;
  return obPearson(obRank(xs.slice(0, n)), obRank(ys.slice(0, n)));
}


/* Rolling Pearson over a window of OBSERVATIONS (weeks here). Returns one
 * value per position from `window - 1` on; earlier positions are null so the
 * series stays aligned with its x axis rather than being silently shifted. */
function obRollingCorr(xs, ys, window) {
  const out = new Array(xs.length).fill(null);
  if (window < 3) return out;
  for (let end = window - 1; end < xs.length; end++) {
    const a = xs.slice(end - window + 1, end + 1);
    const b = ys.slice(end - window + 1, end + 1);
    out[end] = obPearson(a, b);
  }
  return out;
}


/* ── rolling risk ────────────────────────────────────────────────────────
 *
 * The old app's definitions, kept (the user asked for them): the series is
 * P/L SUMMED BY CLOSE DATE over the days that had a close, and the window
 * counts OBSERVATIONS of that series rather than calendar days. At a couple
 * of closes a week, a "90" window is closer to nine months than to three --
 * which is worth knowing and is why the card says so.
 */
function obDailyPnl(cols, idx) {
  const out = new Map();
  for (const i of idx) {
    const d = cols.date_closed[i];
    if (!d) continue;
    out.set(d, (out.get(d) || 0) + cols.pnl[i]);
  }
  return out;
}


function obMean(v) {
  let s = 0;
  for (const x of v) s += x;
  return v.length ? s / v.length : null;
}


/* Sample standard deviation (n-1), as pandas' .std() has it. */
function obStdev(v) {
  if (v.length < 2) return null;
  const m = obMean(v);
  let ss = 0;
  for (const x of v) ss += (x - m) * (x - m);
  return Math.sqrt(ss / (v.length - 1));
}


const OB_ANNUALISE = Math.sqrt(252);


/* mean / stdev * sqrt(252) over a rolling window. Nulls until the window
 * fills, so the series stays aligned with its dates. */
function obRollingSharpe(values, window) {
  const out = new Array(values.length).fill(null);
  for (let e = window - 1; e < values.length; e++) {
    const w = values.slice(e - window + 1, e + 1);
    const sd = obStdev(w);
    out[e] = (sd && sd > 0) ? obMean(w) / sd * OB_ANNUALISE : null;
  }
  return out;
}


/* Sortino: the same, against DOWNSIDE deviation only -- the stdev of the
 * losing days in the window. Fewer than two of them is not a deviation, so
 * the point is null rather than a large number from one loss. */
function obRollingSortino(values, window) {
  const out = new Array(values.length).fill(null);
  for (let e = window - 1; e < values.length; e++) {
    const w = values.slice(e - window + 1, e + 1);
    const neg = w.filter(v => v < 0);
    const sd = neg.length > 1 ? obStdev(neg) : null;
    out[e] = (sd && sd > 0) ? obMean(w) / sd * OB_ANNUALISE : null;
  }
  return out;
}


/* The share of days in the window that made money, as a percentage. Days,
 * not trades: this series is one value per close date. */
function obRollingWinRate(values, window) {
  const out = new Array(values.length).fill(null);
  for (let e = window - 1; e < values.length; e++) {
    const w = values.slice(e - window + 1, e + 1);
    out[e] = w.filter(v => v > 0).length / w.length * 100;
  }
  return out;
}


/* Counts per fixed-width bin, for a histogram. `size` is the bin width in
 * the values' own units ($100 in the old app). Returns the left edge of
 * each bin and its count, over the whole range so gaps read as gaps. */
function obHistogram(values, size) {
  const v = values.filter(x => !obNull(x));
  if (!v.length || !(size > 0)) return { edges: [], counts: [] };
  let lo = Infinity, hi = -Infinity;
  for (const x of v) { if (x < lo) lo = x; if (x > hi) hi = x; }
  const first = Math.floor(lo / size) * size;
  const last = Math.floor(hi / size) * size;
  const n = Math.round((last - first) / size) + 1;
  const counts = new Array(n).fill(0);
  for (const x of v) counts[Math.round((Math.floor(x / size) * size - first) / size)]++;
  const edges = [];
  for (let i = 0; i < n; i++) edges.push(first + i * size);
  return { edges, counts };
}
