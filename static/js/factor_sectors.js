// Factor Trades — the sector filter on the result panes.
//
// A DISPLAY filter over trades already in the browser. It changes no request,
// no trade and no size. Every function here is pure (no Alpine, no DOM), so
// check_ft_sectors runs them in node against the server's own Python.
//
// TWO RULES this file exists to keep:
//
//  1. SIZE ONCE, THEN FILTER. A trade's dollar size is min($/trade,
//     daily cap / N) with N the tickers that fired that DAY across every
//     sector. Re-sizing a filtered subset divides the cap among fewer names
//     and inflates every position. So ftStampSizes runs the page's own
//     _computeDollarSeries over the WHOLE list and stamps each trade's
//     usd_size; _computeDollarSeries then keeps a stamped size. Filtered
//     curves therefore add up to the unfiltered one.
//
//  2. A FILTERED STAT IS THE SUBSET'S OWN, not its contribution. Equity
//     decomposes by sector; drawdown, Calmar, win rate and the rest do not.
//     The page says so wherever a filtered number is shown.
//
// Loaded after factor_charts.js and before factor_trades.js.

const FT_BARS_PER_SESSION = 390;
// Up to this many selected sectors are drawn as their own lines; more than
// that is one line for the selection. Four is what the existing palette
// (SignalThumb.SLOT_PALETTE's first four) distinguishes cleanly.
const FT_SECTOR_LINES = 4;


// ── The map ──────────────────────────────────────────────────────────────
function ftSectorIndex(payload) {
  const of = new Map();
  const order = [];
  for (const b of (payload && payload.buckets) || []) {
    order.push(b.name);
    for (const t of b.tickers) of.set(t, b.name);
  }
  return { order, of, unassigned: (payload && payload.unassigned) || 'Unassigned' };
}

function ftSectorOf(idx, ticker) {
  return (idx && idx.of.get(ticker)) || (idx ? idx.unassigned : 'Unassigned');
}

// What the map and daily_features disagree on, both ways.
function ftMapCheck(idx, liveTickers) {
  if (!idx || !Array.isArray(liveTickers) || !liveTickers.length) return null;
  const live = new Set(liveTickers);
  const notInData = [...idx.of.keys()].filter(t => !live.has(t)).sort();
  const unmapped = liveTickers.filter(t => !idx.of.has(t)).sort();
  return { notInData, unmapped, nLive: live.size, nMapped: idx.of.size };
}


// ── Sizing, once ─────────────────────────────────────────────────────────
// Copies of `trades` carrying usd_size from ONE _computeDollarSeries pass
// over the whole list. A trade that pass skipped (no usable price) gets no
// stamp, and is skipped again wherever it goes.
function ftStampSizes(FC, cmp, trades, perTrade, dailyCap, cap) {
  const ds = FC._computeDollarSeries(cmp, trades || [], perTrade, dailyCap, cap);
  const size = new Map();
  // (ticker, trade_date) is unique within a zone's list (one position per
  // ticker per day -- see _computeDollarSeries). Were it ever not, both rows
  // share a price and a day's N, so they share a size.
  for (const x of ds.tradeDollarSizes) size.set(x.ticker + '|' + x.trade_date, x.dollar_size);
  return (trades || []).map(t => {
    const s = size.get(t.ticker + '|' + t.trade_date);
    return s == null ? { ...t } : { ...t, usd_size: s };
  });
}


// ── The per-sector capital cap: its inputs ───────────────────────────────
// The cap itself is in _computeDollarSeries (factor_charts.js), so the stat
// bar, the grid and the suite cannot size differently. These build what it
// reads.

// Sessions held AFTER the entry session: ceil(exit_bar / 390) - 1, floored
// at 0. The Peak sector exposure definition, and the server's _nights
// (factor_trades.py), which encodes the grid's and suite's holds.
function ftNights(exitBar) {
  const eb = +exitBar;
  if (exitBar == null || !isFinite(eb)) return null;
  return Math.max(0, Math.ceil(eb / FT_BARS_PER_SESSION) - 1);
}

// The server's hold code (_hold_pack): `hw` characters per trade, "." = none.
const FT_HOLD_ALPHABET = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz-_';
function ftHoldDecode(h, hw) {
  const w = hw || 1, n = (h || '').length / w, out = new Array(n);
  for (let i = 0; i < n; i++) {
    const c0 = h[i * w];
    if (c0 === '.') { out[i] = null; continue; }
    let v = 0;
    for (let j = 0; j < w; j++) v = v * 64 + FT_HOLD_ALPHABET.indexOf(h[i * w + j]);
    out[i] = v;
  }
  return out;
}

// The cap's options for _computeDollarSeries, or null when it is off.
// `calendar`: every session (a zone's trading_days, the grid's and suite's
// calendar). A trade's hold is its `nights` when the payload carries one
// (grid, suite), else from its exit_bar (zone).
function ftCapOpts(usd, idx, calendar) {
  if (!(usd > 0) || !idx || !Array.isArray(calendar) || !calendar.length) return null;
  const at = new Map(calendar.map((d, i) => [d, i]));
  return {
    usd: +usd,
    D: calendar.length,
    sectorOf: tk => ftSectorOf(idx, tk),
    sessionOf: d => at.get(d),
    nightsOf: t => (t.nights !== undefined ? t.nights : ftNights(t.exit_bar)),
  };
}


// ── The server's single-window aggregates, over a subset ─────────────────
// Each is a port of the Python that builds the same field in /zone, held to
// it by check_ft_sectors. With every sector on the page shows the SERVER's
// values; these run only on a subset.

// numpy.percentile's default (linear) interpolation.
function _ftPct(sorted, q) {
  const n = sorted.length;
  if (!n) return 0;
  const h = (n - 1) * q / 100, lo = Math.floor(h), hi = Math.ceil(h);
  return sorted[lo] + (sorted[hi] - sorted[lo]) * (h - lo);
}

function _ftDays(a, b) {
  return Math.round((Date.parse(b) - Date.parse(a)) / 86400000);
}

// _window_stats(rets, holds, tickers, span) in factor_trades.py. Order
// matters for max_dd: the cumulative curve runs in the list's order, which
// is the server's row order.
function ftWindowStats(trades) {
  const n = trades.length;
  const counts = new Map();
  for (const t of trades) counts.set(t.ticker, (counts.get(t.ticker) || 0) + 1);
  if (!n) {
    return { n: 0, n_tickers: counts.size, eff_tickers: 0, avg_ret: 0, median: 0,
             std_dev: 0, p5: 0, p95: 0, win_rate: 0, n_win: 0, avg_win: 0, avg_loss: 0,
             trades_per_year: 0, calmar: null, max_dd: 0, avg_hold: 0 };
  }
  const a = trades.map(t => +t.ret || 0);
  let sum = 0, cum = 0, peak = -Infinity, maxDD = 0, nWin = 0, sWin = 0, sLoss = 0;
  for (const x of a) {
    sum += x; cum += x;
    if (cum > peak) peak = cum;
    if (cum - peak < maxDD) maxDD = cum - peak;
    if (x > 0) { nWin++; sWin += x; } else { sLoss += x; }
  }
  const mean = sum / n;
  let ss = 0;
  for (const x of a) ss += (x - mean) * (x - mean);
  const sorted = a.slice().sort((x, y) => x - y);
  let d0 = trades[0].trade_date, d1 = d0;
  for (const t of trades) { if (t.trade_date < d0) d0 = t.trade_date; if (t.trade_date > d1) d1 = t.trade_date; }
  const span = n > 1 ? _ftDays(d0, d1) : 1;
  const years = Math.max(span / 365.25, 1e-9);
  let c2 = 0, c1 = 0;
  for (const c of counts.values()) { c1 += c; c2 += c * c; }
  let hold = 0;
  for (const t of trades) hold += +t.exit_bar || 0;
  return {
    n,
    n_tickers: counts.size,
    eff_tickers: c2 ? (c1 * c1) / c2 : 0,
    avg_ret: mean,
    median: _ftPct(sorted, 50),
    std_dev: Math.sqrt(ss / n),
    p5: _ftPct(sorted, 5),
    p95: _ftPct(sorted, 95),
    win_rate: nWin / n,
    n_win: nWin,
    avg_win: nWin ? sWin / nWin : 0,
    avg_loss: (n - nWin) ? sLoss / (n - nWin) : 0,
    trades_per_year: n / years,
    total_ret: cum,
    avg_annual: cum / years,
    calmar: maxDD < 0 ? ((cum / years) / Math.abs(maxDD)) : null,
    max_dd: maxDD,
    avg_hold: hold / n / FT_BARS_PER_SESSION,
  };
}

// _price_bins: the server's rows give the labels; `edges` the buckets.
function ftPriceBins(trades, serverRows, edges) {
  const acc = edges.map(() => [0, 0]);
  for (const t of trades) {
    const px = t.entry_price;
    if (px == null) continue;
    for (let i = 0; i < edges.length; i++) {
      if (edges[i][0] <= px && px < edges[i][1]) { acc[i][0]++; acc[i][1] += +t.ret || 0; break; }
    }
  }
  return serverRows.map((r, i) => ({ ...r, n: acc[i][0],
                                     avg_ret: acc[i][0] ? acc[i][1] / acc[i][0] : null }));
}

// _pnl_dist: the buckets are contiguous, so each row's `lo` and the next
// row's give its edges -- no second copy of PNL_BUCKETS.
function ftPnlDist(trades, serverRows) {
  const counts = serverRows.map(() => 0);
  for (const t of trades) {
    const r = t.ret;
    if (r == null) continue;
    for (let i = 0; i < serverRows.length; i++) {
      const hi = i + 1 < serverRows.length ? serverRows[i + 1].lo : Infinity;
      if (serverRows[i].lo <= r && r < hi) { counts[i]++; break; }
    }
  }
  return serverRows.map((r, i) => ({ ...r, n: counts[i] }));
}

// _ticker_breakdown, unrounded (the server rounds for the wire).
function ftTickerBreakdown(trades) {
  const by = new Map();
  for (const t of trades) {
    if (!by.has(t.ticker)) by.set(t.ticker, []);
    by.get(t.ticker).push(+t.ret || 0);
  }
  let total = 0;
  for (const rs of by.values()) for (const x of rs) total += x;
  const out = [];
  for (const [ticker, rs] of by) {
    const n = rs.length, s = rs.reduce((p, x) => p + x, 0);
    out.push({ ticker, n, avg_ret: n ? s / n : 0,
               win_rate: n ? rs.filter(x => x > 0).length / n : 0,
               contrib_pct: total ? s / total * 100 : 0 });
  }
  return out.sort((x, y) => y.n - x.n);     // stable: ties keep first-seen order
}

// _exit_breakdown: counts, shares and holds from the subset; each rule's
// label, side and backstop flag from the server's row for it.
function ftExitBreakdown(trades, serverRows) {
  const meta = new Map((serverRows || []).map(r => [r.rule_key, r]));
  const n = new Map(), hold = new Map();
  for (const t of trades) {
    n.set(t.exit_rule, (n.get(t.exit_rule) || 0) + 1);
    hold.set(t.exit_rule, (hold.get(t.exit_rule) || 0) + (+t.exit_bar || 0));
  }
  const tot = trades.length || 1;
  return [...n.entries()].sort((a, b) => b[1] - a[1]).map(([rk, k]) => {
    const m = meta.get(rk) || { rule_key: rk, family: rk, side: '', label: rk, is_backstop: false };
    return { ...m, n: k, frac: k / tot, avg_hold: hold.get(rk) / k / FT_BARS_PER_SESSION };
  });
}


// ── The view ─────────────────────────────────────────────────────────────
// A zone payload as the panes read it, restricted to `keep` (a Set of
// sector names). Sizes come from the WHOLE zone. `win` is the page window,
// whose stats the stat bar reads as view[win].
function ftSectorView(zone, keep, idx, sized, priceEdges, win) {
  const inKeep = t => keep.has(ftSectorOf(idx, t.ticker));
  const w = sized.window.filter(inKeep);
  const s = sized.series.filter(inKeep);
  return {
    ...zone,
    window_trades: w, combined_trades: w, series_trades: s,
    combined_trade_dates: w.map(t => t.trade_date),
    price_bins: ftPriceBins(w, zone.price_bins || [], priceEdges),
    pnl_dist: ftPnlDist(w, zone.pnl_dist || []),
    tickers: ftTickerBreakdown(w),
    exit_reasons: ftExitBreakdown(w, zone.exit_reasons || []),
    [win]: ftWindowStats(w),
    sector_filtered: true,
  };
}


// ── Peak concurrent sector exposure ──────────────────────────────────────
// The largest dollar amount one sector had open at once, over `sized`
// trades, and that amount as a share of EVERYTHING open on the same
// session (every sector -- the book a cap is a fraction of).
//
// `only`: a Set of sector names to search, or null for every sector. The
// page passes its chip selection, so one chip gives that sector's own peak;
// several give the largest single-sector peak among them. The share's
// denominator is always the whole book, whatever `only` says.
//
// "Open on session k" = entered on or before k and not exited before k:
// k in [entry, entry + nights], nights = ceil(exit_bar / 390) - 1. That is
// the exposure a cap checked at entry would see -- it counts the day's own
// entries and positions closing later that day. It is NOT the Trade
// Activity pane's figure, which counts only what is held over a close, so
// this reads higher than that pane's peak on the same trades.
//
// Exposure only rises on entry days, so the peak is always on one; trades
// running past the calendar's end cannot move it.
function ftPeakSectorExposure(sized, tradingDays, idx, only) {
  const D = tradingDays.length;
  const at = new Map(tradingDays.map((d, i) => [d, i]));
  const diff = new Map();
  const totDiff = new Float64Array(D + 1);
  let unplaced = 0;
  for (const t of sized) {
    if (t.usd_size == null) continue;
    const i = at.get(t.trade_date);
    if (i === undefined) { unplaced++; continue; }
    // The cap's own hold (ftNights), so the peak and the cap cannot disagree
    // about how long a position is open.
    const nights = t.nights !== undefined ? (t.nights ?? 0) : (ftNights(t.exit_bar) ?? 0);
    const sec = ftSectorOf(idx, t.ticker);
    if (!diff.has(sec)) diff.set(sec, new Float64Array(D + 1));
    const a = diff.get(sec), end = Math.min(i + nights + 1, D);
    a[i] += t.usd_size; a[end] -= t.usd_size;
    totDiff[i] += t.usd_size; totDiff[end] -= t.usd_size;
  }
  const total = new Float64Array(D);
  for (let k = 0, run = 0; k < D; k++) { run += totDiff[k]; total[k] = run; }
  let best = null;
  for (const [sec, a] of diff) {
    if (only && !only.has(sec)) continue;
    let run = 0;
    for (let k = 0; k < D; k++) {
      run += a[k];
      if (!best || run > best.usd + 1e-9) best = { usd: run, sector: sec, date: tradingDays[k], k };
    }
  }
  if (!best || !(best.usd > 0)) {
    return { usd: 0, sector: null, date: null, total: null, share: null, unplaced };
  }
  const tot = total[best.k];
  return { usd: best.usd, sector: best.sector, date: best.date, total: tot,
           share: tot > 0 ? best.usd / tot : null, unplaced };
}
