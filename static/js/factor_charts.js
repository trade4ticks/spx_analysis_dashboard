'use strict';

// Shared chart renderers for the Factor family of pages.
//
// These four panes -- equity+drawdown, annual P&L, trade activity, ticker
// breakdown -- plus the nine helpers they depend on used to live as METHODS
// on the oiAnalysis Alpine component. They were already parameterized on
// (canvasId, data), which is what let Zone, Recall and Portfolio share them,
// but being methods made them unreachable from any other page. Factor Trades
// needed the same four, and copying them would have created exactly the
// parallel implementation this project keeps working to avoid.
//
// So they moved here verbatim, with one mechanical change: `this.` became
// `cmp.`, and every function takes the component as its first argument. The
// bodies are otherwise character-for-character what they were -- verified by
// substituting back and diffing against the pre-extraction source.
//
// cmp must supply, by these exact names:
//   state   _charts, activityMode, data, dedupeConc, equityAggMode,
//           equityDollarParams, secBubbleMinN, secDetail, zoneData
// Every helper the renderers call is on this object, so cmp does not need to
// provide them. oi_analysis.js delegates with `this`, which already has all
// of the above; Factor Trades supplies an object using the same names.
//
// Loaded BEFORE the page JS in every template that renders these charts.
// Same pattern as signal_thumbnail.js.

window.FactorCharts = {

  _renderSecEquity(cmp, canvasId = 'sec-equity-canvas', detail = null, singleSeries = false) {
    detail = detail || cmp.secDetail;
    const _key = canvasId.replace(/-canvas$/, '').replace(/^chart-/, '');
    const canvas = document.getElementById(canvasId);
    if (!canvas || !detail) return;
    if (cmp._charts[_key]) { cmp._charts[_key].destroy(); delete cmp._charts[_key]; }
    // Per-pane mode: each of the four single-line equity views
    // (zone / sec / recall / port) carries its own toggle state.
    const modeKey = window.FactorCharts._equityModeKey(cmp, canvasId);
    const mode    = (cmp.equityAggMode && cmp.equityAggMode[modeKey]) || 'daily';

    // Dollar-capped branch — zone/recall/port only. Secondary is
    // never in dollar mode (its toggle stays binary, untouched).
    // Computes equity, drawdown, and y-axis tick formatters from
    // combined_trades rather than the backend's pre-built %
    // equity_primary series, so $/trade and daily-cap changes
    // re-render without a server call.
    const isDollar = (mode === 'dollar_capped') && (modeKey !== 'sec');
    if (isDollar) {
      const params  = cmp.equityDollarParams[modeKey] || { perTrade: 2000, dailyCap: 10000 };
      const trades  = detail.combined_trades || [];
      const dollarS = window.FactorCharts._computeDollarSeries(cmp, trades, params.perTrade, params.dailyCap);
      if (!dollarS.equity.length) return;
      const toMs = d => new Date(d).getTime();
      const eqPxy = dollarS.equity.map(p => ({ x: toMs(p.date), y: +p.value.toFixed(2) }));
      const dd    = window.FactorCharts._drawdownFromDollarEquity(cmp, dollarS.equity);
      const ddxy  = dd.map(p => ({ x: toMs(p.date), y: +p.value.toFixed(2) }));
      const ctx   = canvas.getContext('2d');
      const fmt$  = v => {
        const abs = Math.abs(v);
        const sign = v < 0 ? '-' : '';
        if (abs >= 1e6) return sign + '$' + (abs / 1e6).toFixed(2) + 'M';
        if (abs >= 1e3) return sign + '$' + (abs / 1e3).toFixed(1) + 'k';
        return sign + '$' + abs.toFixed(0);
      };
      cmp._charts[_key] = new Chart(ctx, {
        plugins: [...window.FactorCharts._cutoffPlugin(cmp, 'time'),
                  ...window.FactorCharts._crosshairPlugin(cmp, 'time')],
        type: 'line',
        data: { datasets: [
          { label: 'Equity ($)', data: eqPxy, borderColor: '#3498db',
            backgroundColor: 'rgba(52,152,219,0.08)', borderWidth: 1.5,
            pointRadius: 0, fill: false, tension: 0, stepped: 'after', yAxisID: 'y' },
          { label: 'Drawdown ($)', data: ddxy, borderColor: 'rgba(232,67,147,.32)',
            backgroundColor: 'transparent', borderWidth: 1,
            pointRadius: 0, fill: false, tension: 0, stepped: 'after', yAxisID: 'y1' },
        ] },
        options: {
          responsive: true, maintainAspectRatio: false, animation: false,
          parsing: false,
          plugins: {
            legend: { display: false },
            tooltip: {
              mode: 'index', intersect: false,
              callbacks: {
                title: items => items.length
                  ? new Date(items[0].parsed.x).toISOString().slice(0, 10) : '',
                label: item => `${item.dataset.label}: ${fmt$(item.parsed.y)}`,
              },
            },
          },
          scales: {
            x: {
              type: 'linear',
              min: (window.FactorCharts._seriesAxis(cmp) || {}).fromMs ?? eqPxy[0].x,
              max: (window.FactorCharts._seriesAxis(cmp) || {}).toMs ?? eqPxy[eqPxy.length - 1].x,
              ticks: { color: '#888', font: { size: 9 }, maxTicksLimit: 10,
                autoSkip: true, autoSkipPadding: 16,
                callback: val => new Date(val).toISOString().slice(0, 7) },
              grid: { color: '#222' },
            },
            y:  { position: 'left',  ticks: { color: '#888', font: { size: 9 }, callback: fmt$ },
                  grid: { color: '#222' } },
            y1: { position: 'right', ticks: { color: 'rgba(232,67,147,.6)', font: { size: 9 }, callback: fmt$ },
                  grid: { drawOnChartArea: false }, max: 0 },
          },
        },
      });
      return;
    }

    const eqP = window.FactorCharts._equityForMode(cmp, detail.equity_primary  || [], mode);
    const eqC = window.FactorCharts._equityForMode(cmp, detail.equity_combined || [], mode);
    if (!eqP.length) return;

    // Real time x-axis: each point carries its ACTUAL DATE as an
    // epoch-ms x-value (Chart.js linear scale positions by the
    // numeric x, so a 16-month gap occupies 16 months of pixels —
    // not one index slot). stepped:'after' on the line datasets
    // holds the value flat from each point to the next, then steps
    // to the new value, so no-trade gaps render as horizontal
    // segments without a diagonal implying gains accrued. Chart.js
    // category axis (or auto-skipped categorical) can't do this —
    // it spaces by index, not by date. (No date adapter is loaded
    // for type:'time'; epoch-ms on linear gets the same result with
    // no new dependency.)
    const toMs    = d => new Date(d).getTime();
    const eqPxy   = eqP.map(p => ({ x: toMs(p.date), y: +(p.value * 100).toFixed(4) }));
    const eqCxy   = eqC.map(p => ({ x: toMs(p.date), y: +(p.value * 100).toFixed(4) }));
    const drawdown = window.FactorCharts._drawdownFromEquity(cmp, eqP);
    const ddxy    = drawdown.map(p => ({ x: toMs(p.date), y: +p.value.toFixed(3) }));

    const ctx = canvas.getContext('2d');

    let datasets;
    if (singleSeries) {
      // Zone / recall / portfolio mode: single curve. Equity in
      // canonical project blue (#3498db); drawdown overlay below
      // uses canonical pink on the secondary y-axis.
      datasets = [{
        label: 'Equity',
        data: eqPxy,
        borderColor: '#3498db',
        backgroundColor: 'rgba(52,152,219,0.08)',
        borderWidth: 1.5,
        pointRadius: 0,
        fill: false,
        tension: 0,
        stepped: 'after',
        yAxisID: 'y',
      }];
    } else {
      // Sec-detail mode: primary + combined curves. Each carries
      // its own {x: date, y: pct} so the two lines independently
      // sit at their real dates on the shared time axis.
      datasets = [
        {
          label: 'Primary filter',
          data: eqPxy,
          borderColor: '#3498db',
          backgroundColor: 'rgba(52,152,219,0.08)',
          borderWidth: 1.5,
          pointRadius: 0,
          fill: false,
          tension: 0,
          stepped: 'after',
          yAxisID: 'y',
        },
        {
          label: '+ Secondary filter',
          data: eqCxy,
          borderColor: '#e84393',
          backgroundColor: 'transparent',
          borderWidth: 1.5,
          pointRadius: 0,
          fill: false,
          tension: 0,
          stepped: 'after',
          yAxisID: 'y',
        },
      ];
    }

    // Drawdown overlay — faint pink line on a secondary y-axis.
    // Recomputed peak-to-trough from whichever equity series the
    // mode toggle just produced, plotted at real dates with the
    // same step semantics so it stays flat across no-trade gaps.
    datasets.push({
      label: 'Drawdown',
      data: ddxy,
      borderColor: 'rgba(232,67,147,.32)',
      backgroundColor: 'transparent',
      borderWidth: 1,
      pointRadius: 0,
      fill: false,
      tension: 0,
      stepped: 'after',
      yAxisID: 'y1',
    });

    cmp._charts[_key] = new Chart(ctx, {
        plugins: [...window.FactorCharts._cutoffPlugin(cmp, 'time'),
                  ...window.FactorCharts._crosshairPlugin(cmp, 'time')],
      type: 'line',
      data: { datasets },   // no labels — points carry their own x
      options: {
        responsive: true, maintainAspectRatio: false, animation: false,
        parsing: false,     // data is already {x, y} — skip parsing
        plugins: {
          legend: { display: false },
          tooltip: {
            mode: 'index', intersect: false,
            callbacks: {
              title: (items) => {
                if (!items.length) return '';
                const ms = items[0].parsed.x;
                return new Date(ms).toISOString().slice(0, 10);   // YYYY-MM-DD
              },
              label: (item) => {
                const lab = item.dataset.label || '';
                const v   = item.parsed.y;
                return `${lab}: ${v.toFixed(2)}%`;
              },
            },
          },
        },
        scales: {
          // LINEAR x-axis on epoch milliseconds — Chart.js positions
          // each point at its actual numeric x, so a 16-month gap
          // takes 16 months of horizontal pixels (not 1 slot). No
          // date adapter needed (which would be required for
          // type:'time' on Chart.js 4). YYYY-MM tick labels via
          // formatter.
          x: {
            type: 'linear',
            // Hug the data range — no padding on either end. With
            // bounds:'data' Chart.js still pads to "nice" tick
            // boundaries, so set min/max explicitly to the first
            // and last data points.
            min: (window.FactorCharts._seriesAxis(cmp) || {}).fromMs ?? eqPxy[0].x,
            max: (window.FactorCharts._seriesAxis(cmp) || {}).toMs ?? eqPxy[eqPxy.length - 1].x,
            ticks: {
              color: '#888', font: { size: 9 },
              maxTicksLimit: 10,
              autoSkip: true,
              autoSkipPadding: 16,
              callback: (val) => {
                const d = new Date(val);
                return d.toISOString().slice(0, 7);   // YYYY-MM
              },
            },
            grid: { color: '#222' },
          },
          y: {
            position: 'left',
            ticks: { color: '#888', font: { size: 9 }, callback: v => v.toFixed(1) + '%' },
            grid: { color: '#222' },
          },
          y1: {
            position: 'right',
            ticks: {
              color: 'rgba(232,67,147,.6)', font: { size: 9 },
              callback: v => v.toFixed(0) + '%',
            },
            grid: { drawOnChartArea: false },
            max: 0,
          },
        },
      },
    });
  },

  _renderZoneYearly(cmp, canvasId, data) {
    canvasId = canvasId || 'chart-zone-yearly';
    const src = data || cmp.zoneData;
    const canvas = document.getElementById(canvasId);
    if (!canvas || !src) return;
    // Chart-key derived from canvasId so main ('zone-yearly') and
    // recall ('recall-yearly') don't collide in cmp._charts.
    const chartKey = canvasId.replace(/^chart-/, '');
    if (cmp._charts[chartKey]) { cmp._charts[chartKey].destroy(); delete cmp._charts[chartKey]; }

    // Mode-aware: which section is this canvas in, and what mode is
    // that section's equity toggle in? Read the trade-level data
    // (combined_trades) and recompute — the backend's `yearly` field
    // is per-trade-mean only, so we can't trust it once the toggle
    // is on a different mode.
    const modeKey = window.FactorCharts._equityModeKey(cmp, canvasId);
    const mode    = (cmp.equityAggMode && cmp.equityAggMode[modeKey]) || 'daily';
    const dollarParams = cmp.equityDollarParams[modeKey];
    const yearly = window.FactorCharts._yearlyForMode(cmp, src.combined_trades || [], mode, dollarParams);
    if (!yearly.length) return;

    // Pad to the fixed domain so TRAIN shows empty 2024-2026 slots rather
    // than a shorter axis. Padding here, not in _yearlyForMode, keeps the
    // aggregation honest -- these are placeholders with n=0, not years with
    // zero return.
    const _ax = window.FactorCharts._seriesAxis(cmp);
    if (_ax) {
      const have = new Set(yearly.map(y => +y.year));
      for (let y = _ax.fromYear; y <= _ax.toYear; y++) {
        // Must match _yearlyForMode's shape exactly. An earlier version
        // omitted win_rate, which threw inside the colour callback -- and
        // because every renderer shared one try/catch, that single throw
        // blanked five panes.
        if (!have.has(y)) yearly.push({ year: y, value: 0, n: 0, win_rate: 0 });
      }
      yearly.sort((a, b) => (+a.year) - (+b.year));
    }

    const isDollar = (mode === 'dollar_capped');
    // Alpha carries n and only n; see _barPaint. min/max come from the
    // years actually on screen, so TRAIN grades 2019-2023 against each
    // other and TEST grades its full span against each other -- one
    // population per mode, one scale.
    const ns = yearly.map(y => y.n);
    const minN = Math.min(...ns), maxN = Math.max(...ns);
    const nPct = y => maxN > minN ? (y.n - minN) / (maxN - minN) : 1;
    const paints = yearly.map(y => window.FactorCharts._barPaint(
      window.FactorCharts._barRgb(y.value >= 0 ? 'pos' : 'neg'), nPct(y)));
    const fmt$ = v => {
      const abs = Math.abs(v);
      const sign = v < 0 ? '-' : '';
      if (abs >= 1e6) return sign + '$' + (abs / 1e6).toFixed(2) + 'M';
      if (abs >= 1e3) return sign + '$' + (abs / 1e3).toFixed(1) + 'k';
      return sign + '$' + abs.toFixed(0);
    };
    const ctx = canvas.getContext('2d');
    cmp._charts[chartKey] = new Chart(ctx, {
      plugins: window.FactorCharts._cutoffPlugin(cmp, 'cat', yearly.map(y => y.year)),
      type: 'bar',
      data: {
        labels:   yearly.map(y => y.year),
        datasets: [{
          label:           isDollar ? 'Annual P&L' : 'Avg Ret',
          data:            yearly.map(y => isDollar ? +y.value.toFixed(2) : +(y.value * 100).toFixed(3)),
          backgroundColor: paints.map(p => p.background),
          borderColor:     paints.map(p => p.border),
          borderWidth:     paints.map(p => p.borderWidth),
        }],
      },
      options: {
        responsive: true, maintainAspectRatio: false, animation: false,
        plugins: {
          legend: { display: false },
          tooltip: {
            backgroundColor: 'rgba(20,20,20,0.95)', borderColor: '#444', borderWidth: 1,
            callbacks: {
              label: ctx => {
                const y = yearly[ctx.dataIndex];
                const primary = isDollar
                  ? `P&L: ${fmt$(y.value)}`
                  : `Avg: ${(y.value*100).toFixed(3)}%`;
                return [primary, `WR: ${(y.win_rate*100).toFixed(1)}%`, `n: ${y.n}`];
              },
            },
          },
        },
        scales: {
          x: { ticks: { color: '#888', font: { size: 9 } }, grid: { color: '#222' } },
          y: { ticks: { color: '#888', font: { size: 9 },
                        callback: v => isDollar ? fmt$(v) : v.toFixed(2) + '%' },
               grid:  { color: '#222' } },
        },
      },
    });
  },

  _renderSecActivity(cmp, canvasId = 'sec-activity-canvas', detail = null) {
    detail = detail || cmp.secDetail;
    const _key = canvasId.replace(/-canvas$/, '').replace(/^chart-/, '');
    const canvas = document.getElementById(canvasId);
    if (!canvas || !detail) return;
    if (cmp._charts[_key]) { cmp._charts[_key].destroy(); delete cmp._charts[_key]; }

    // Prefer the enriched combined_trades (has ticker per entry) so the
    // dedupe-concurrent toggle can work per ticker. Fall back to plain
    // combined_trade_dates for older payloads.
    const trades = detail.combined_trades
      || (detail.combined_trade_dates || []).map(d => ({ ticker: '?', trade_date: d }));
    if (!trades.length) return;
    const horizon = detail.horizon || 1;

    // Trading-day calendar resolution, preferred → last-resort:
    //   1. detail.trading_days — dense ride-along from the section's
    //      own endpoint (secondary-zone-analyze, portfolios/aggregate).
    //      Available on initial page load without requiring a prior
    //      primary Analyze; spans the trade set's date range exactly,
    //      so the activity-pane x-axis lines up with the equity curve.
    //   2. cmp.data.spot_series — populated only after primary /analyze.
    //   3. cmp.data.trade_calendar — same gate as (2).
    //   4. Sparse fired-trade dates (last-resort safety net; only fires
    //      if all three above are missing — used to be the broken
    //      initial-load path and is now effectively dead code).
    const detailDays = detail.trading_days || [];
    const spotSeries = cmp.data?.spot_series || [];
    const cal = cmp.data?.trade_calendar || [];
    const dates = trades.map(t => t.trade_date || t.date);
    const tradingDays = detailDays.length > 0
      ? detailDays
      : spotSeries.length > 0
        ? spotSeries.map(s => s.date)
        : [...new Set(cal.length > 0 ? cal.map(c => c.date) : dates)].sort();

    // Derive dedupeConc key from canvasId: chart-port-activity → 'port', else 'sec'
    const _dedupeKey = canvasId.includes('port') ? 'port' : 'sec';
    const kept = cmp.dedupeConc[_dedupeKey]
      ? window.FactorCharts._dedupeConcurrent(cmp, trades, tradingDays, horizon)
      : trades;

    // Section key for activity-mode lookup. Secondary stays on
    // count-only (no dollar mode available); all others can switch
    // to capital when their equity toggle is on dollar-capped.
    const sectionKey = window.FactorCharts._equityModeKey(cmp, canvasId);
    const isDollarEquity = (cmp.equityAggMode?.[sectionKey] === 'dollar_capped');
    const wantsCapital   = (cmp.activityMode?.[sectionKey] === 'capital');
    const isCapital      = isDollarEquity && wantsCapital && sectionKey !== 'sec';

    // Per-trade weight: 1 for count view, dollar_size for capital
    // view. Single windowing function runs over a weights-by-date
    // map — same H, same boundaries in both views.
    let weightByDate;
    // null in count mode -> every trade weighs 1; a Map in capital mode.
    let tradeWeight = null;
    if (isCapital) {
      const params = cmp.equityDollarParams[sectionKey] || { perTrade: 2000, dailyCap: 10000 };
      // Run the dollar series over the dedupe-filtered trade list
      // so the dedupeConcurrent toggle propagates into the Capital
      // view too: skipped trades contribute neither to bars nor to
      // the rolling line. dayDeployedByDate already aggregates per
      // day, so the same windowing loop below sums it over H days
      // to get rolling deployed capital — identical math to the
      // count path, dollar weights instead of unit weights.
      const { dayDeployedByDate, tradeDollarSizes } =
        window.FactorCharts._computeDollarSeries(cmp,
          kept, params.perTrade, params.dailyCap,
        );
      weightByDate = dayDeployedByDate;
      // Per-TRADE dollar size, needed because open positions are now walked
      // per trade rather than summed from a per-day aggregate. Keyed on
      // (ticker, trade_date), which is unique within a trade list.
      tradeWeight = new Map(
        tradeDollarSizes.map(x => [x.ticker + '|' + x.trade_date, x.dollar_size]));
    } else {
      weightByDate = new Map();
      for (const t of kept) {
        const d = t.trade_date || t.date;
        weightByDate.set(d, (weightByDate.get(d) || 0) + 1);
      }
    }

    const entered = tradingDays.map(d => weightByDate.get(d) || 0);

    // Open positions = what is carried OVERNIGHT, counted once per session at
    // the close. Intraday churn is noise for a multi-day system; the question
    // is what is being held and what capital it ties up.
    //
    // Per-trade hold when the trade carries one, fixed horizon when it does
    // not. exit_bar counts 1-minute bars over the regular session, so
    // sessions = exit_bar / 390 and the number of closes the position is
    // carried through is ceil(sessions) - 1:
    //
    //   exit_bar 390  -> 1.00 sessions -> 0 nights  (opened and closed same
    //                    session: counts in Entered, contributes nothing to
    //                    Open, and occupies no later day)
    //   exit_bar 558  -> 1.43 sessions -> 1 night
    //
    // A naive ceil() would give that same-session trade a night it never had,
    // overstating concurrency and — once the sizing cap is live — wrongly
    // blocking other entries.
    //
    // Trades WITHOUT exit_bar fall back to `horizon` nights, which reproduces
    // the previous fixed-window behaviour exactly. That is what keeps Recall,
    // Zone and Portfolio byte-identical: they pass a real fixed horizon and
    // their trades carry no exit_bar, so every trade takes the fallback and
    // the resulting window is the old [i-H+1 .. i] sum.
    const BARS_PER_SESSION = 390;
    const dayIndex = new Map(tradingDays.map((d, i) => [d, i]));
    const open = new Array(tradingDays.length).fill(0);
    for (const t of kept) {
      const d = t.trade_date || t.date;
      const i = dayIndex.get(d);
      if (i === undefined) continue;
      const nights = (t.exit_bar != null && isFinite(+t.exit_bar))
        ? Math.max(0, Math.ceil((+t.exit_bar) / BARS_PER_SESSION) - 1)
        : horizon;
      const w = tradeWeight
        ? (tradeWeight.get((t.ticker || '') + '|' + d) || 0)
        : 1;
      for (let j = i; j < Math.min(i + nights, tradingDays.length); j++) open[j] += w;
    }

    const fmt$ = v => {
      const abs = Math.abs(v);
      const sign = v < 0 ? '-' : '';
      if (abs >= 1e6) return sign + '$' + (abs / 1e6).toFixed(2) + 'M';
      if (abs >= 1e3) return sign + '$' + (abs / 1e3).toFixed(1) + 'k';
      return sign + '$' + abs.toFixed(0);
    };

    const ctx = canvas.getContext('2d');
    cmp._charts[_key] = new Chart(ctx, {
      plugins: [...window.FactorCharts._cutoffPlugin(cmp, 'cat', tradingDays),
                ...window.FactorCharts._crosshairPlugin(cmp, 'cat', tradingDays)],
      type: 'bar',
      data: {
        labels: tradingDays.map(d => d.slice(0, 7)),
        datasets: [
          {
            type: 'line',
            label: isCapital ? 'Deployed (rolling)' : 'Open Trades',
            data: open,
            borderColor: 'rgba(46,204,113,0.6)',
            backgroundColor: window.FactorCharts._preCutoffFade(
              cmp, tradingDays, 'rgba(46,204,113,0.08)', 'rgba(46,204,113,0.03)'),
            fill: true, tension: 0.3, pointRadius: 0, borderWidth: 1.5,
            order: 1,
          },
          {
            type: 'bar',
            label: isCapital ? 'Deployed' : 'Entered',
            data: entered,
            backgroundColor: window.FactorCharts._preCutoffFade(
              cmp, tradingDays, 'rgba(52,152,219,0.7)', 'rgba(52,152,219,0.22)'),
            barThickness: 2,
            order: 2,
          },
        ],
      },
      options: {
        responsive: true, maintainAspectRatio: false, animation: false,
        plugins: {
          legend: { labels: { color: '#aaa', font: { size: 10 } } },
          tooltip: {
            backgroundColor: 'rgba(20,20,20,0.95)', borderColor: '#444', borderWidth: 1,
            mode: 'index', intersect: false,
            callbacks: {
              title: ctx => tradingDays[ctx[0]?.dataIndex] || '',
              label: ctx => isCapital
                ? `${ctx.dataset.label}: ${fmt$(ctx.raw)}`
                : `${ctx.dataset.label}: ${ctx.raw}`,
            },
          },
        },
        scales: {
          ...window.FactorCharts._darkScales(cmp),
          x: { ...window.FactorCharts._darkScales(cmp).x, ticks: { ...window.FactorCharts._darkScales(cmp).x.ticks, maxTicksLimit: 12 } },
          y: isCapital ? {
            ...window.FactorCharts._darkScales(cmp).y,
            title: { display: true, text: 'Capital ($)', color: '#888', font: { size: 9 } },
            ticks: { ...window.FactorCharts._darkScales(cmp).y.ticks, callback: fmt$ },
          } : {
            ...window.FactorCharts._darkScales(cmp).y,
            title: { display: true, text: 'Count', color: '#888', font: { size: 9 } },
            ticks: { ...window.FactorCharts._darkScales(cmp).y.ticks, stepSize: 1 },
          },
        },
      },
    });
  },

  _renderSecBubble(cmp, canvasId = 'sec-bubble-canvas', detail = null) {
    detail = detail || cmp.secDetail;
    const _key = canvasId.replace(/-canvas$/, '').replace(/^chart-/, '');
    const canvas = document.getElementById(canvasId);
    if (!canvas || !detail?.tickers?.length) return;
    if (cmp._charts[_key]) { cmp._charts[_key].destroy(); delete cmp._charts[_key]; }

    const minN = cmp.secBubbleMinN || 1;
    const tickers = detail.tickers.filter(t => t.n >= minN);
    if (!tickers.length) return;

    // Radius: positive contrib scaled 3–20; negative → 2
    const maxContrib = Math.max(1, ...tickers.filter(t => t.contrib_pct > 0).map(t => t.contrib_pct));
    // Color: pink (#e84393) at wr=0, blue (#3498db) at wr=1
    const mkColor = (wr, a) => {
      const r = Math.round(232 + (52  - 232) * wr);
      const g = Math.round(67  + (152 - 67)  * wr);
      const b = Math.round(147 + (219 - 147) * wr);
      return `rgba(${r},${g},${b},${a})`;
    };

    const datasets = tickers.map(t => ({
      label: t.ticker,
      data: [{ x: t.n, y: +(t.avg_ret * 100).toFixed(4), r: t.contrib_pct > 0 ? Math.max(3, (t.contrib_pct / maxContrib) * 20) : 2 }],
      backgroundColor: mkColor(t.win_rate, 0.65),
      borderColor:     mkColor(t.win_rate, 1),
      borderWidth: 1,
    }));

    // Trade-weighted avg ret across the visible tickers (n-weighted) → %
    const totalN = tickers.reduce((s, t) => s + (t.n || 0), 0);
    const avgPct = totalN > 0
      ? tickers.reduce((s, t) => s + (t.avg_ret || 0) * (t.n || 0), 0) / totalN * 100
      : 0;

    cmp._charts[_key] = new Chart(canvas.getContext('2d'), {
      type: 'bubble',
      data: { datasets },
      plugins: [window.FactorCharts._avgRetLinePlugin(cmp, avgPct, 'avg')],
      options: {
        responsive: true, maintainAspectRatio: false, animation: false,
        plugins: {
          legend: { display: false },
          tooltip: {
            backgroundColor: 'rgba(20,20,20,0.95)', borderColor: '#444', borderWidth: 1,
            callbacks: {
              label: ctx => {
                const t = tickers[ctx.datasetIndex];
                return [`${t.ticker}  n:${t.n}  avg:${(t.avg_ret*100).toFixed(3)}%  WR:${(t.win_rate*100).toFixed(1)}%  contrib:${t.contrib_pct.toFixed(1)}%`];
              },
            },
          },
        },
        scales: {
          ...window.FactorCharts._darkScales(cmp),
          x: { ...window.FactorCharts._darkScales(cmp).x,
               title: { display: true, text: 'Trade Count', color: '#888', font: { size: 9 } } },
          y: { ...window.FactorCharts._darkScales(cmp).y,
               title: { display: true, text: 'Avg Return %', color: '#888', font: { size: 9 } } },
        },
      },
    });
  },

  // Faint vertical marker at the train/test cutoff, for panes that plot the
  // FULL series. Lives here because the three panes that need it sit on two
  // different scale types: equity is a linear axis of epoch-ms (deliberately,
  // so a gap occupies its real width), while annual P&L and activity are
  // CATEGORY axes indexed by label. One plugin, two conversions.
  //
  // Returns [] when cmp.cutoffDate is unset, so pages without a train/test
  // split are unaffected and this can be spread into any plugins array.
  // Pre-cutoff points render muted, post-cutoff full strength, on all three
  // time-series panes. In TEST mode those panes span the full history while
  // every single-window number beside them covers test only -- so the equity
  // endpoint deliberately will NOT equal Total Ret. Making the pre-cutoff
  // stretch visibly different is what stops that being rediscovered as a bug
  // months later. Returns the full colour when there is no cutoff.
  // Fixed x-domain for the time-series panes, when the page asks for one.
  // Without it the axis rescales on every toggle and you have to read the
  // tick labels to know which window you are in. Pinned, the shape alone
  // says it -- half-filled is train, full is test -- and the cutoff line
  // stays in the same place on screen. Returns null when unset, so pages
  // that do not opt in keep data-driven bounds.
  _seriesAxis(cmp) {
    const a = cmp.seriesAxis;
    if (!a || !a.from) return null;
    const to = a.to || new Date().toISOString().slice(0, 10);
    return { fromMs: new Date(a.from).getTime(), toMs: new Date(to).getTime(),
             fromYear: +a.from.slice(0, 4), toYear: +to.slice(0, 4) };
  },

  // ── Bar visual language ──────────────────────────────────────────────
  // ALPHA MEANS SAMPLE SIZE, and nothing else. Faint = thin n, vivid =
  // well-populated. No second signal may be written to this channel.
  //
  // A pre/post-cutoff distinction WAS layered on top of the gradient here,
  // also as alpha, and the two cancelled: in TRAIN every bar is pre-cutoff
  // so the gradient washed out to one flat tone, and in TEST a post-cutoff
  // year on 756 trades read BRIGHTER than a pre-cutoff year on 1439 --
  // backwards -- because the muting overwrote the alpha the gradient had
  // just set. The vertical cutoff line already marks the boundary, so the
  // bars do not need to carry it and the channel stays single-purpose.
  //
  // The gradient scales against whatever set is on screen: one population
  // per mode, so TRAIN's bars grade against each other and TEST's against
  // each other.
  //
  // Used by Annual P&L, price bins and the P&L distribution, so one look
  // means one thing everywhere on the page.
  //
  // rgb: "r,g,b" string. nRatio: 0..1 (pass 1 when n is unknown/uniform).
  _barPaint(rgb, nRatio) {
    const r = Number.isFinite(nRatio) ? Math.max(0, Math.min(1, nRatio)) : 1;
    return {
      background:  `rgba(${rgb},${(0.20 + 0.60 * r).toFixed(3)})`,
      border:      `rgba(${rgb},0.9)`,
      borderWidth: 1,
    };
  },

  // The page's two bar hues, so a positive bar is the same blue in every
  // pane. 'none' is the empty-bucket grey.
  _barRgb(kind) {
    return kind === 'none' ? '120,120,120'
         : kind === 'neg'  ? '232,67,147'
         :                   '52,152,219';
  },

  _preCutoffFade(cmp, dates, full, muted) {
    const iso = cmp.cutoffLineDate;
    if (!iso) return full;
    return (dates || []).map(d => (String(d) < iso ? muted : full));
  },

  _cutoffPlugin(cmp, kind, labels) {
    // Deliberately NOT cmp.cutoffDate. On Factor Analysis that field is
    // populated from /tt-cutoff on every page load regardless of mode, so
    // keying off it would draw a train/test marker on in-sample Recall
    // charts. cutoffLineDate is the explicit opt-in: a page returns a date
    // only when it actually wants the line.
    const iso = cmp.cutoffLineDate;
    if (!iso) return [];
    return [{
      id: 'ftCutoff',
      afterDatasetsDraw(chart) {
        const xs = chart.scales.x, ys = chart.scales.y;
        if (!xs || !ys) return;
        let px = null;
        if (kind === 'time') {
          px = xs.getPixelForValue(new Date(iso).getTime());
        } else {
          // Category axis: first label at or after the cutoff. Labels may be
          // truncated for display (YYYY-MM on activity, YYYY on annual), so
          // compare on the same prefix length rather than assuming full ISO.
          const L = (labels || []).map(String);
          const n = L.length ? L[0].length : 10;
          const key = iso.slice(0, n);
          const i = L.findIndex(d => d >= key);
          if (i >= 0) {
            // getPixelForValue on a category scale returns the bar's CENTRE,
            // which draws the line through the middle of the first
            // post-cutoff bar and reads as though that bar is split. The
            // boundary is the midpoint between it and the bar before it.
            px = (i > 0)
              ? (xs.getPixelForValue(i - 1) + xs.getPixelForValue(i)) / 2
              : xs.getPixelForValue(i) - (xs.getPixelForValue(1) - xs.getPixelForValue(0)) / 2;
          }
        }
        if (px == null || !isFinite(px)) return;
        const c = chart.ctx;
        c.save();
        c.strokeStyle = 'rgba(200,200,200,0.35)';
        c.setLineDash([4, 4]);
        c.lineWidth = 1;
        c.beginPath(); c.moveTo(px, ys.top); c.lineTo(px, ys.bottom); c.stroke();
        c.restore();
      },
    }];
  },

  // ── Synced crosshair ─────────────────────────────────────────────────
  // Hovering one participating pane draws a vertical line at the SAME DATE
  // on every other participating pane. Line only, no readout.
  //
  // OPT-IN: nothing draws unless cmp.crosshairSync is true, so Recall, Zone
  // and Portfolio are untouched until a page turns it on.
  //
  // SYNC IS ON THE DATE VALUE, NEVER THE INDEX. Equity is a linear epoch-ms
  // axis carrying one point per trade date; activity is a category axis
  // carrying every session. Index n is a different date in each, so an
  // index-synced line would drift further apart the more sessions passed
  // without a trade. Each chart converts pixel -> its own nearest date on
  // the way out and date -> its own nearest position on the way in, which
  // is what makes the match work in both directions between two axes that
  // do not share a domain.
  //
  // Visually it must not read as the cutoff line: the cutoff is part of the
  // chart (dashed, dim, fixed), the crosshair is a pointer (solid, bright,
  // capped, follows the mouse).
  _crosshairPlugin(cmp, kind, dates) {
    if (!cmp.crosshairSync) return [];

    // ms values of a time chart's points, read off the chart rather than
    // passed in, so the two equity call sites do not have to hand over a
    // parallel array that could fall out of step with what they plot.
    const timePos = (chart) => {
      const ds = chart.data?.datasets?.[0]?.data || [];
      return ds.map(p => (p && typeof p === 'object') ? p.x : null)
               .filter(v => Number.isFinite(v));
    };
    const nearestNum = (arr, v) => {
      let best = null, bd = Infinity;
      for (const a of arr) { const d = Math.abs(a - v); if (d < bd) { bd = d; best = a; } }
      return best;
    };
    const isoOfMs = (ms) => new Date(ms).toISOString().slice(0, 10);
    const nearestIdx = (isoArr, iso) => {
      const t = new Date(iso).getTime();
      let best = -1, bd = Infinity;
      for (let i = 0; i < isoArr.length; i++) {
        const d = Math.abs(new Date(isoArr[i]).getTime() - t);
        if (d < bd) { bd = d; best = i; }
      }
      return best;
    };

    return [{
      id: 'ftCrosshair',
      // Marks the chart as a participant so a hover on one can redraw the
      // others without a registry the caller has to maintain.
      install(chart) { chart.$ftCrosshair = true; },
      afterEvent(chart, a) {
        const t = a.event?.type;
        if (t !== 'mousemove' && t !== 'mouseout') return;
        const xs = chart.scales.x;
        if (!xs) return;
        let next = null;
        if (t === 'mousemove') {
          const px = a.event.x;
          if (px >= xs.left && px <= xs.right) {
            const v = xs.getValueForPixel(px);
            if (kind === 'cat') {
              const i = Math.max(0, Math.min((dates || []).length - 1, Math.round(v)));
              next = (dates || [])[i] || null;
            } else {
              const ms = nearestNum(timePos(chart), v);
              next = ms == null ? null : isoOfMs(ms);
            }
          }
        }
        if (next === cmp.crosshairDate) return;
        cmp.crosshairDate = next;
        // Redraw the OTHER participants. draw() clears and repaints from
        // cached data -- no re-derivation of the dollar series, which is
        // what makes this cheap enough to run on every mousemove.
        for (const c of Object.values(cmp._charts || {})) {
          if (c && c !== chart && c.$ftCrosshair) { try { c.draw(); } catch (e) { /* chart torn down mid-hover */ } }
        }
        a.changed = true;               // repaint the chart under the cursor
      },
      afterDatasetsDraw(chart) {
        const iso = cmp.crosshairDate;
        if (!iso) return;
        const xs = chart.scales.x, ys = chart.scales.y;
        if (!xs || !ys) return;
        let px = null;
        if (kind === 'cat') {
          const i = nearestIdx(dates || [], iso);
          if (i < 0) return;
          px = xs.getPixelForValue(i);
        } else {
          const ms = nearestNum(timePos(chart), new Date(iso).getTime());
          if (ms == null) return;
          px = xs.getPixelForValue(ms);
        }
        if (px == null || !isFinite(px)) return;
        const c = chart.ctx;
        c.save();
        // Solid and bright against the cutoff's dashed rgba(200,200,200,.35):
        // different line style AND different weight, so the two never have to
        // be told apart by colour alone.
        c.strokeStyle = 'rgba(255,255,255,0.55)';
        c.lineWidth = 1;
        c.setLineDash([]);
        c.beginPath(); c.moveTo(px, ys.top); c.lineTo(px, ys.bottom); c.stroke();
        // A cap at the top reads as a pointer rather than a chart feature.
        c.fillStyle = 'rgba(255,255,255,0.75)';
        c.beginPath();
        c.moveTo(px - 4, ys.top); c.lineTo(px + 4, ys.top); c.lineTo(px, ys.top + 5);
        c.closePath(); c.fill();
        c.restore();
      },
    }];
  },

  _avgRetLinePlugin(cmp, avgPct, label) {
    return {
      id: 'avgRetLine',
      afterDraw(chart) {
        if (!Number.isFinite(avgPct)) return;
        const yScale = chart.scales.y;
        const xScale = chart.scales.x;
        if (!yScale || !xScale) return;
        const y = yScale.getPixelForValue(avgPct);
        const ctx = chart.ctx;
        ctx.save();
        ctx.strokeStyle = 'rgba(170,170,170,0.55)';
        ctx.lineWidth = 1;
        ctx.setLineDash([3, 3]);
        ctx.beginPath();
        ctx.moveTo(xScale.left,  y);
        ctx.lineTo(xScale.right, y);
        ctx.stroke();
        ctx.setLineDash([]);
        ctx.fillStyle = 'rgba(170,170,170,0.75)';
        ctx.font = '10px sans-serif';
        ctx.textAlign = 'right';
        ctx.textBaseline = 'bottom';
        ctx.fillText(
          (label || 'avg') + ' ' + avgPct.toFixed(3) + '%',
          xScale.right - 4, y - 2);
        ctx.restore();
      },
    };
  },

  _computeDollarSeries(cmp, trades, perTrade, dailyCap) {
    // ── HOVER-LAG INSTRUMENTATION (Lab.diag) ─────────────────────────────
    // _labDsCallsThisHover is set to 0 by the hover/leave handlers before
    // they call anything; we count every _computeDollarSeries invocation
    // that occurs while that flag is defined.
    if (typeof cmp._labDsCallsThisHover === 'number') {
      cmp._labDsCallsThisHover++;
    }
    // ─────────────────────────────────────────────────────────────────────
    const empty = {
      equity: [], dayPnlByDate: new Map(),
      dayDeployedByDate: new Map(), tradeDollarSizes: [],
    };
    if (!trades || !trades.length) return empty;
    // Group by date and count distinct tickers per day so dilution
    // matches reality (a ticker firing under multiple cells in the
    // same signal is still one position; combined_trades carries
    // deduped rows for portfolio and single-signal-deduped rows
    // for zone/recall, so distinct-counting is correct in both).
    const tradesByDate = new Map();
    const tickersByDate = new Map();
    for (const t of trades) {
      const d = t.trade_date;
      if (!d) continue;
      if (!tradesByDate.has(d)) {
        tradesByDate.set(d, []);
        tickersByDate.set(d, new Set());
      }
      tradesByDate.get(d).push(t);
      if (t.ticker) tickersByDate.get(d).add(t.ticker);
    }
    const dates = Array.from(tradesByDate.keys()).sort();
    const dayPnlByDate      = new Map();
    const dayDeployedByDate = new Map();
    const tradeDollarSizes  = [];
    const equity = [];
    let cum = 0;
    for (const d of dates) {
      const dayTrades = tradesByDate.get(d);
      const N = Math.max(1, tickersByDate.get(d).size);
      const perTickerAlloc = Math.min(perTrade, dailyCap / N);
      let dayPnl      = 0;
      let dayDeployed = 0;
      for (const t of dayTrades) {
        // Size off the AS-TRADED price, not the back-adjusted one.
        // spot_entry is restated onto today's share scale: forward splits
        // shrink old prices, reverse splits inflate them. The scale error
        // normally cancels (floor(alloc/px) * px ≈ alloc), but it stops
        // cancelling the moment the floor binds — and on a reverse-split
        // ticker the adjusted price can exceed perTickerAlloc outright, so
        // floor() yields 0, the 1-share minimum below kicks in, and the
        // trade deploys the full inflated price instead of the allocation.
        // spot_entry_raw = spot_entry / adj_factor, supplied by the server.
        // Fall back to the adjusted price when it's absent (split-free
        // ticker, where factor is 1.0 and the two are identical, or a
        // pre-v14 cached payload) so behaviour degrades to the old path
        // rather than dropping the trade.
        const pxAdj = +t.spot_entry;
        const pxRawCand = +t.spot_entry_raw;
        const px  = (isFinite(pxRawCand) && pxRawCand > 0) ? pxRawCand : pxAdj;
        const ret = +t.ret;
        if (!isFinite(px) || px <= 0 || !isFinite(ret)) continue;
        let shares = Math.floor(perTickerAlloc / px);
        if (shares < 1) shares = 1;   // 1-share min, no redistribution
        const dollarSize = shares * px;
        const tradePnl   = dollarSize * ret;
        dayPnl      += tradePnl;
        dayDeployed += dollarSize;
        tradeDollarSizes.push({
          ticker:      t.ticker,
          trade_date:  d,
          dollar_size: dollarSize,
          dollar_pnl:  tradePnl,
        });
      }
      cum += dayPnl;
      dayPnlByDate.set(d, dayPnl);
      dayDeployedByDate.set(d, dayDeployed);
      equity.push({ date: d, value: cum });
    }
    return { equity, dayPnlByDate, dayDeployedByDate, tradeDollarSizes };
  },

  _darkScales(cmp) {
    return {
      x: { ticks:{color:'#888',font:{size:9},maxRotation:45}, grid:{color:'rgba(255,255,255,0.05)'}, border:{color:'transparent'} },
      y: { ticks:{color:'#888',font:{size:9}}, grid:{color:'rgba(255,255,255,0.05)'}, border:{color:'transparent'} },
    };
  },

  _dedupeConcurrent(cmp, entries, tradingDays, horizon) {
    if (!entries?.length || !tradingDays?.length || !(horizon > 0)) return entries || [];
    const idxBy = new Map();
    tradingDays.forEach((d, i) => idxBy.set(d, i));
    const sorted = [...entries].sort((a, b) => {
      const ad = a.date || a.trade_date || '';
      const bd = b.date || b.trade_date || '';
      return ad < bd ? -1 : (ad > bd ? 1 : 0);
    });
    const lastByTkr = new Map();
    const keep = [];
    for (const e of sorted) {
      const t  = e.ticker || '?';
      const dk = e.date || e.trade_date || '';
      const i  = idxBy.get(dk);
      if (i == null) { keep.push(e); continue; }
      const last = lastByTkr.get(t);
      if (last == null || (i - last) >= horizon) {
        keep.push(e);
        lastByTkr.set(t, i);
      }
    }
    return keep;
  },

  _drawdownFromDollarEquity(cmp, eqPoints) {
    let peak = 0;
    const out = [];
    for (const p of eqPoints) {
      if (p.value > peak) peak = p.value;
      out.push({ date: p.date, value: p.value - peak });
    }
    return out;
  },

  _drawdownFromEquity(cmp, eqPoints) {
    let peak = 1;
    const out = [];
    for (const p of eqPoints) {
      const equity = 1 + p.value;
      if (equity > peak) peak = equity;
      out.push({
        date:  p.date,
        value: ((equity - peak) / peak) * 100,
      });
    }
    return out;
  },

  _equityForMode(cmp, rawPoints, mode) {
    if (!rawPoints || !rawPoints.length) return [];
    if (mode === 'pertrade') return rawPoints;
    const byDate = {};
    const dates  = [];
    let prev = 0;
    for (const p of rawPoints) {
      const ret = p.value - prev;
      prev = p.value;
      if (byDate[p.date] === undefined) {
        byDate[p.date] = { sum: 0, n: 0 };
        dates.push(p.date);
      }
      byDate[p.date].sum += ret;
      byDate[p.date].n   += 1;
    }
    // Backend already sorts by date, so dates[] is in order; no resort needed.
    let cum = 0;
    const out = [];
    for (const d of dates) {
      const dailyAvg = byDate[d].sum / byDate[d].n;
      cum += dailyAvg;
      out.push({ date: d, value: cum });
    }
    return out;
  },

  _equityModeKey(cmp, canvasId) {
    const c = canvasId || '';
    // Factor Trades needs its own key: every ft-* canvas would otherwise fall
    // through to 'sec', and 'sec' is explicitly barred from dollar mode by
    // the isDollar checks, so the page could never show dollar axes.
    if (c.includes('ft-'))    return 'ft';
    if (c.includes('zone'))   return 'zone';
    if (c.includes('recall')) return 'recall';
    if (c.includes('port'))   return 'port';
    return 'sec';
  },

  _yearlyForMode(cmp, trades, mode, dollarParams) {
    if (!trades || !trades.length) return [];
    // Bucket trades by year and by (year, date)
    const byYear = new Map();          // year → trade returns[]
    const byYearDate = new Map();      // year → Map(date → return[])
    for (const t of trades) {
      const d = t.trade_date;
      const r = +t.ret;
      if (!d || !isFinite(r)) continue;
      const y = +d.slice(0, 4);
      if (!byYear.has(y)) {
        byYear.set(y, []);
        byYearDate.set(y, new Map());
      }
      byYear.get(y).push(r);
      const dmap = byYearDate.get(y);
      if (!dmap.has(d)) dmap.set(d, []);
      dmap.get(d).push(r);
    }
    const years = Array.from(byYear.keys()).sort();

    if (mode === 'dollar_capped') {
      // Reuse _computeDollarSeries so the year bars come from the
      // SAME dayPnlByDate the equity curve does — guarantees
      // reconciliation (sum of bars = endpoint of curve).
      const params = dollarParams || { perTrade: 2000, dailyCap: 10000 };
      const { dayPnlByDate } = window.FactorCharts._computeDollarSeries(cmp, trades, params.perTrade, params.dailyCap);
      const yearPnl = new Map();
      const yearN   = new Map();
      for (const [d, pnl] of dayPnlByDate.entries()) {
        const y = +d.slice(0, 4);
        yearPnl.set(y, (yearPnl.get(y) || 0) + pnl);
        yearN.set(y, (yearN.get(y) || 0) + (byYearDate.get(y)?.get(d)?.length || 0));
      }
      return years.map(y => ({
        year:    y,
        value:   yearPnl.get(y) || 0,
        n:       yearN.get(y)   || 0,
        win_rate: (() => {
          const rs = byYear.get(y) || [];
          return rs.length ? rs.filter(r => r > 0).length / rs.length : 0;
        })(),
      }));
    }

    if (mode === 'daily') {
      // Per-day mean → mean of those daily means for the year.
      return years.map(y => {
        const dmap = byYearDate.get(y);
        const dailyAvgs = [];
        for (const arr of dmap.values()) {
          const s = arr.reduce((a, b) => a + b, 0);
          dailyAvgs.push(s / arr.length);
        }
        const ys = byYear.get(y);
        return {
          year:    y,
          value:   dailyAvgs.reduce((a, b) => a + b, 0) / dailyAvgs.length,
          n:       ys.length,
          win_rate: ys.filter(r => r > 0).length / ys.length,
        };
      });
    }

    // pertrade — mean of trade returns per year
    return years.map(y => {
      const ys = byYear.get(y);
      return {
        year:    y,
        value:   ys.reduce((a, b) => a + b, 0) / ys.length,
        n:       ys.length,
        win_rate: ys.filter(r => r > 0).length / ys.length,
      };
    });
  },

  // Range-parameterised so a SECOND heatmap on the same page cannot borrow
  // -- or clobber -- the first one's colour scale. The body is unchanged
  // from hmCellBg; only the two reads off the component became arguments.
  // The parameter is minSampleN, not minN: the body already declares a
  // local minN and a same-named parameter shadows it.
  // opts (all optional): {mid, lo, hi}. Default -- no opts -- is the
  // original behaviour exactly: diverge around zero, symmetric +/-range.
  // A second heatmap can instead anchor the midpoint somewhere meaningful
  // (the null combination) with independent endpoints, without a second
  // colour formula that could drift from this one.
  _hmPaint(range, minSampleN, cell, opts) {
    // Three tiers driven by the SINGLE `hmMinSampleN` threshold —
    // same number determines hatching AND gradient inclusion. Critical
    // invariant: a cell rendered with a gradient color is also in the
    // population whose min/max defined that gradient. If a low-n cell
    // with a wild return stayed in the scale, every real cell would
    // collapse to near-uniform shade by relativity — the user-stated
    // reason for keeping hatch and scale on the same threshold.
    if (!cell || !cell.n) return 'rgba(40,40,40,0.5)';   // n=0: empty
    const n = cell.n;
    const minN = minSampleN || 0;
    // Tier 2: 0 < n < threshold — hatched gray, no gradient color.
    // Pattern matches the Regime Heatmap style for visual consistency.
    if (n < minN) {
      return 'repeating-linear-gradient(45deg, #2e2e2e 0 4px, transparent 4px 8px),'
           + 'repeating-linear-gradient(-45deg, #2e2e2e 0 4px, transparent 4px 8px),'
           + '#1c1c1c';
    }
    // Tier 3: n >= threshold — gradient, scaled across visible cells only.
    const v = cell.avg_ret || 0;
    let t;
    if (opts && Number.isFinite(opts.mid)) {
      // Distance from the midpoint, normalised by the reach on THAT side.
      // Asymmetric on purpose: a matrix clustered just above its null has a
      // short blue reach and a long red one, and forcing them equal is what
      // collapses every cell into one slice of the ramp.
      const d = v - opts.mid;
      const up = (Number.isFinite(opts.hi) ? opts.hi : opts.mid) - opts.mid;
      const dn = opts.mid - (Number.isFinite(opts.lo) ? opts.lo : opts.mid);
      t = d >= 0 ? (up > 0 ? d / up : 0) : (dn > 0 ? d / dn : 0);
      t = Math.max(-1, Math.min(1, t));
    } else {
      t = Math.max(-1, Math.min(1, v / (range || 0.01)));
    }
    if (t >= 0) return `rgba(52,152,219,${(0.15 + t * 0.7).toFixed(2)})`;
    return `rgba(232,67,147,${(0.15 + (-t) * 0.7).toFixed(2)})`;
  },

  hmCellBg(cmp, cell) {
    return window.FactorCharts._hmPaint(cmp._hmRange, cmp.hmMinSampleN, cell);
  },

  _hmCellTitle(cmp, cell, ix, iy) {
    if (!cell || !cell.n) return 'n=0';
    let s = `n=${cell.n}  avg=${((cell.avg_ret||0)*100).toFixed(3)}%  wr=${((cell.win_rate||0)*100).toFixed(1)}%`;
    const xt = cmp.heatmapData?.x_thresholds;
    const yt = cmp.heatmapData?.y_thresholds;
    if (xt && yt) {
      const fmt = v => v !== undefined ? v.toFixed(4) : '?';
      s += `\nX (${cmp.metric}): ${fmt(xt[ix])} – ${fmt(xt[ix+1])}`;
      s += `\nY (${cmp.secSelectedMetric}): ${fmt(yt[iy])} – ${fmt(yt[iy+1])}`;
    }
    return s;
  },

  _hmRecomputeRange(cmp) {
    if (!cmp.heatmapData) { cmp._hmRange = null; return; }
    const minN = cmp.hmMinSampleN || 0;
    const grids = [
      ...(cmp.heatmapData.grid || []),
      ...(cmp.heatmapData.train_grid || []),
      ...(cmp.heatmapData.test_grid || []),
    ];
    let max = 0;
    for (const row of grids) {
      for (const c of row) {
        if (c && (c.n || 0) >= minN) {
          max = Math.max(max, Math.abs(c.avg_ret || 0));
        }
      }
    }
    cmp._hmRange = max || 0.01;
  },

  groupMetricsByFamily(cmp, list, keepOrder = false) {
    const groups = new Map();  // family_num → {family_num, family_name, metrics:[]}
    for (const m of (list || [])) {
      const fam = cmp.metricFamilyLookup[m];
      const key = fam ? fam.family_num : 999;
      const label = fam ? fam.family_name : 'Other';
      if (!groups.has(key)) groups.set(key, { family_num: key, family_name: label, metrics: [] });
      groups.get(key).metrics.push(m);
    }
    // Sort metrics alphabetically within each family group.
    // Use explicit localeCompare so the sort is unambiguous regardless of
    // whether Alpine's reactivity layer has wrapped the string elements.
    if (!keepOrder) for (const g of groups.values())
      g.metrics.sort((a, b) => String(a).localeCompare(String(b)));
    const result = [...groups.values()];
    if (!keepOrder) result.sort((a, b) => a.family_num - b.family_num);
    return result;
  },

};
