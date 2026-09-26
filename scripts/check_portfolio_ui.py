"""The Backtest Portfolio page, DRIVEN BY CLICKING, in a real browser.

WHY THIS EXISTS, in the words of the fault that produced it. The Filters
button stopped opening the panel. Every check passed: the panel was `x-if`,
it was in the main column, its arithmetic was right, and a browser check even
read its geometry on screen. All of them opened the panel by CALLING
`toggleEdit()`. Nothing clicked the button, so `:disabled="!loaded.length"`
-- added with the panel and looking identical to an enabled button, because
the disabled style kept the text colour -- went unnoticed until it was used.

So: this clicks what a person clicks, on the page as rendered, and reads what
the page does. It also captures `window.onerror` and `console.error`, because
an Alpine expression error does not stop the page -- it logs and leaves the
control inert, which is exactly the shape of the bug above. Closing the panel
was logging one on every click and nothing said so.

WHERE IT RUNS: a machine with Edge. The VPS has none, so it SKIPS there
rather than passing -- a check that cannot run has not run. Registered in
gates.py with can_skip.
"""
from __future__ import annotations

import datetime as dt
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

EXIT_SKIPPED = 3
EDGE_CANDIDATES = [
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    "/usr/bin/microsoft-edge",
    "/usr/bin/chromium",
    "/usr/bin/google-chrome",
]


def find_browser() -> str | None:
    for p in EDGE_CANDIDATES:
        if Path(p).exists():
            return p
    return None


def build_page() -> str:
    """The real template, with its own CSS and JS inlined so file:// works."""
    import jinja2
    import pandas as pd

    from app.assets import asset
    from app.oo_backtest.registry import registry_with_coverage

    class _URL:
        hostname = "localhost"; scheme = "http"; path = "/"
        def __str__(self): return "http://localhost/"

    class _Req:
        url = _URL(); headers = {}; query_params = {}; scope = {"type": "http"}

    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(str(ROOT / "templates")),
        autoescape=True, keep_trailing_newline=True)
    env.globals["asset"] = asset
    env.globals["live_port"] = 8001
    html = env.get_template("backtest_portfolio.html").render(request=_Req())
    html = re.sub(
        r'<link rel="stylesheet" href=[^>]*css/([a-z_]+\.css)[^>]*>',
        lambda m: "<style>\n" + (ROOT / "static/css" / m.group(1)).read_text(encoding="utf-8") + "\n</style>",
        html)
    html = re.sub(
        r"<script src=[^>]*?/static/js/([a-z_]+\.js)[^>]*></script>",
        lambda m: "<script>\n" + (ROOT / "static/js" / m.group(1)).read_text(encoding="utf-8") + "\n</script>",
        html)

    sessions = [d.date().isoformat() for d in pd.bdate_range("2017-01-03", "2026-12-31")]

    def payload(sid, name, color, cap, n, start, step, hold, pnls, vix_from,
                first_hold=None):
        # first_hold: the FIRST trade held far longer than the rest, which is
        # ordinary for options held days to months and is what pulls a
        # strategy's earliest OPEN away from its earliest CLOSE. The charts
        # plot by close, so the two are not interchangeable.
        d0 = dt.date.fromisoformat(start)
        op = [(d0 + dt.timedelta(days=i * step)).isoformat() for i in range(n)]
        holds = [hold] * n
        if first_hold:
            holds[0] = first_hold
        cl = [(dt.date.fromisoformat(o) + dt.timedelta(days=holds[i])).isoformat()
              for i, o in enumerate(op)]
        return {
            "saved": {"id": sid, "name": name, "capital_per_position": cap,
                      "trade_count": n},
            "filename": f"{name}.json", "color": color, "n": n,
            "date_min": op[0], "date_max": cl[-1],
            "capital_per_position": cap, "qty": 1,
            "parse": {"source": "cache", "seconds": 0.04, "rows": n},
            "market": {"joined": True, "spx_sessions": sessions},
            "columns": {
                "date_opened": op, "date_closed": cl,
                "pnl": [pnls[i % len(pnls)] for i in range(n)],
                "days_in_trade": holds,
                "day_of_week": [dt.date.fromisoformat(d).weekday() for d in op],
                "exit_reason": ["profit target"] * n,
                # Gaps, but no coverage START: a missing premium is missing
                # data, not a metric that did not exist yet.
                "premium": [None if i % 7 == 0 else 100.0 + i for i in range(n)],
                # A metric whose coverage starts late, so a filter on it has
                # a cost the panel has to state.
                # Nulls of TWO KINDS, as the real column has: the first
                # vix_from trades predate the series (before coverage), and
                # a scattered few later ones have no bar at the entry time
                # (a 09:30 entry). Only the first kind is a STRETCH.
                "vix_level": [None if (i < vix_from or i % 17 == 5)
                               else 14 + (i % 9) for i in range(n)],
            },
        }

    a = payload(1, "monthly", "#3498db", 25000, 40, "2023-01-03", 30, 30, [100.0], 0)
    # Its first trade is held 400 days, so its earliest OPEN (2013-01-02) and
    # the start of its LINE are well over a year apart.
    old = payload(3, "since2013", "#4ec9a0", 20000, 84, "2013-01-02", 60, 20,
                  [250.0, -90.0], 30, first_hold=400)
    b = payload(2, "weekly", "#e84393", 10000, 40, "2023-01-06", 7, 7, [300.0, -100.0], 10)
    # index_ohlc's real per-series starts, so the registry carries the
    # minDates the hatch is derived from. Fixture values: the app reads
    # these from the data at request time and hardcodes none of them.
    coverage = {"spx": "2017-01-03", "vix": "2017-01-03",
                "vix3m": "2017-10-24", "vix9d": "2018-06-08"}
    reg = [m for m in registry_with_coverage(coverage) if m.get("filter")]
    return html.replace("</head>", (STUB % (json.dumps([a, b, old]), json.dumps(reg))) + "</head>", 1)


# NOT a raw string, and it cannot casually become one: several sequences in
# here are written for Python to process. The cost is that a JS apostrophe
# escaped as a backslash-quote is EATEN at parse time, leaving an unterminated
# JS string and a page that reports nothing at all. Use double quotes for any
# JS string containing an apostrophe.
STUB = """
<script>
window.__errs = [];
window.onerror = (m) => window.__errs.push('error: ' + m);
window.addEventListener('unhandledrejection', e => window.__errs.push('promise: ' + e.reason));
(function () {
  const ce = console.error, cw = console.warn;
  console.error = (...a) => { window.__errs.push('console.error: ' + a.map(String).join(' ')); ce(...a); };
  console.warn = (...a) => { window.__errs.push('console.warn: ' + a.map(String).join(' ')); cw(...a); };
})();

const LOADED = %s;
window.fetch = async (u) => ({ok: true, json: async () => {
  if (String(u).includes('/strategies')) return {strategies: [
    {id:1,name:'monthly',trade_count:40,capital_per_position:25000},
    {id:2,name:'weekly',trade_count:40,capital_per_position:10000},
    {id:3,name:'since2013',trade_count:84,capital_per_position:20000}],
    colors:['#3498db','#e84393'], max:12};
  if (String(u).includes('/registry')) return {metrics: %s};
  // THE OO PAGE'S SURFACE ENDPOINTS, reused rather than duplicated.
  if (String(u).includes('/surface/catalog')) return {
    metrics: [
      {column_name: 'iv_30d_atm', family: 'iv', form: 'level', units: 'vol_decimal',
       description: 'ATM implied vol, 30d', min_date: '2020-01-02'},
      {column_name: 'z_iv_30d_atm', family: 'iv', form: 'z', units: 'zscore',
       description: 'ATM implied vol, 30d, z-scored', min_date: '2021-04-05'},
      {column_name: 'skew_25d', family: 'skew', form: 'level', units: 'vol_decimal',
       description: '25-delta skew', min_date: '2020-01-02'}],
    family_groups: [{label: 'IV', color: '#3987e5', families: ['iv']},
                    {label: 'Skew', color: '#d95926', families: ['skew']}],
    other_group: {label: 'Other', color: '#8a8a8a'},
    unit_formats: {vol_decimal: {scale: 100, decimals: 2, suffix: 'vol pts'},
                   zscore: {scale: 1, decimals: 2, suffix: ''}},
    default_unit_format: {scale: 1, decimals: 4, suffix: ''},
  };
  if (String(u).includes('/load')) return {strategies: LOADED,
    load: {seconds: 0.2, n: 2, from_cache: 2, parsed: 0}};
  return {};
}});

// A profile store, in memory, behaving like the endpoints: 409 on a name
// that exists, and a load that reports which strategies are gone.
window.__surfCalls = [];
window.__profiles = [];
const realFetch = window.fetch;
window.fetch = async (u, init) => {
  const url = String(u);
  if (url.includes('/surface/values')) {
    const body = JSON.parse(init.body);
    window.__surfCalls.push({column: body.column, n: body.trades.length});
    const from = body.column === 'z_iv_30d_atm' ? '2021-04-05' : '2020-01-02';
    const values = body.trades.map((t, i) =>
      (t[0] < from ? null : (i %% 13 === 4 ? null : 0.10 + (i %% 17) * 0.002)));
    return {ok: true, status: 200, json: async () => ({
      column: body.column, values,
      bar_times: values.map(v => (v === null ? null : '09:35:00')),
      report: {trades: values.length, no_bar: values.filter(v => v === null).length}})};
  }
  if (url.includes('/profiles')) {
    const m = url.match(/profiles\/(\d+)/);
    if (init && init.method === 'POST') {
      const body = JSON.parse(init.body);
      const clash = window.__profiles.find(p => p.name === body.name);
      if (clash && !body.replace) {
        return { ok: false, status: 409, json: async () => (
          { detail: { detail: 'A profile named "' + body.name + '" already exists.',
                      existing_id: clash.id } }) };
      }
      const rec = clash || { id: window.__profiles.length + 1, name: body.name };
      rec.payload = body.payload;
      rec.n_strategies = body.payload.strategies.length;
      if (!clash) window.__profiles.push(rec);
      return { ok: true, status: 200, json: async () => ({ profile: rec }) };
    }
    if (init && init.method === 'DELETE') {
      window.__profiles = window.__profiles.filter(p => String(p.id) !== m[1]);
      return { ok: true, status: 200, json: async () => ({ deleted: +m[1] }) };
    }
    if (m) {
      const rec = window.__profiles.find(p => String(p.id) === m[1]);
      // Strategy 2 is "deleted": the page must say so, not quietly load one.
      const missing = rec.payload.strategies.map(s => s.id).filter(i => i === 99);
      return { ok: true, status: 200, json: async () => ({ profile: {
        ...rec, missing, names: { '1': 'monthly', '2': 'weekly', '3': 'since2013' } } }) };
    }
    return { ok: true, status: 200, json: async () => (
      { profiles: JSON.parse(JSON.stringify(window.__profiles)) }) };
  }
  return realFetch(u, init);
};

window.addEventListener('load', () => setTimeout(async () => {
  const out = [];
  // A BACKSTOP REPORTER. If the driver hangs on an await that never settles,
  // the page produces nothing and the gate can only say "it did not get that
  // far" -- which is what it said, twice, while the real answer was one
  // unresolved promise away.
  setTimeout(() => {
    if (document.getElementById('report')) return;
    const pre = document.createElement('pre');
    pre.id = 'report';
    pre.textContent = out.concat(['FAIL|the driver did not finish|hung|done',
      'errs|' + (window.__errs.slice(0, 3).join(' ~ ') || 'clean')]).join(String.fromCharCode(10));
    document.body.appendChild(pre);
  }, 40000);
  try {
  const ok = (k, got, want) => out.push(
    (String(got) === String(want) ? 'ok|' : 'FAIL|') + k + '|' + got + '|' + want);
  const wait = ms => new Promise(r => setTimeout(r, ms));
  const c = Alpine.$data(document.querySelector('[x-data]'));
  const side = () => [...document.querySelectorAll('.ob-side button')];
  const filterBtns = () => side().filter(b => /filter|editing/i.test(b.textContent));
  const panels = () => document.querySelectorAll('.bp-fgrid').length;

  // ADD TWO STRATEGIES THE WAY A PERSON DOES: choose in the dropdown, then
  // press Add. The SELECT is driven with a real input event rather than by
  // assigning to the component -- writing `pick` directly raced x-model,
  // which re-synced the select and wrote its old value back, so the second
  // Add did nothing. Driving the control is the point of this file.
  const selectBy = (text) => [...document.querySelectorAll('.ob-side select')]
    .find(el => el.options[0] && el.options[0].textContent.includes(text));
  const choose = async (id) => {
    const sel = selectBy('Add a saved strategy');
    sel.value = String(id);
    // BOTH EVENTS: Alpine's x-model listens for `change` on a <select>, not
    // `input`, so dispatching only the latter left `pick` at 0 and the Add
    // button disabled -- and a click on a disabled button does nothing,
    // silently, which is the same shape as the bug being tested for.
    sel.dispatchEvent(new Event('input', { bubbles: true }));
    sel.dispatchEvent(new Event('change', { bubbles: true }));
    await wait(40);
    side().find(b => b.textContent.trim() === 'Add').click();
    await wait(60);
  };
  await choose(1);
  await choose(2);
  await choose(3);
  ok('three strategies added', c.chosen.length, 3);

  // THE BUTTON WORKS BEFORE A LOAD. It was disabled here, which is the
  // regression this file exists for.
  const early = filterBtns();
  ok('a filter button per strategy', early.length, 3);
  ok('not disabled before loading', early[0].disabled, false);
  early[0].click();
  await wait(120);
  ok('clicking opens the panel', panels(), 1);
  // Across the PANEL, not the first cell: that one is categorical (Day of
  // Week) and has no range to be missing.
  ok('the panel says values need a load',
     /load the portfolio/i.test(document.querySelector('.bp-fgrid').textContent), true);
  early[0].click();
  await wait(100);
  ok('clicking again closes it', panels(), 0);

  // LOAD, then drive the panel by clicking.
  side().find(b => /load portfolio/i.test(b.textContent)).click();
  await wait(250);
  ok('the summary table drew', document.querySelectorAll('.bp-table tr.total').length, 1);

  const btn = filterBtns()[1];
  btn.click();
  await wait(150);
  ok('the second strategy opens', c.editing, 2);
  ok('its panel is on screen',
     document.querySelector('.bp-fgrid').offsetHeight > 0, true);
  ok('the panel is in the main column',
     !!document.querySelector('.ob-main .bp-fgrid'), true);
  // ABOVE THE SUMMARY, where the old app had it: configure, then read the
  // results below. The table is sticky, so its numbers stay on screen while
  // a control up here moves.
  const panelCard = document.querySelector('.bp-fgrid').closest('.ob-card');
  const sumCard = document.querySelector('.bp-summary');
  ok('the panel sits ABOVE the summary',
     !!(panelCard.compareDocumentPosition(sumCard)
        & Node.DOCUMENT_POSITION_FOLLOWING), true);
  ok('no sliders in the sidebar',
     document.querySelectorAll('.ob-side .ob-dual').length, 0);

  // A CHECKBOX, CLICKED, changes the table.
  const before = document.querySelector('.bp-table tr.total td:nth-child(2)').textContent;
  const boxes = [...document.querySelectorAll('.bp-fcell input[type=checkbox]')];
  const vixBox = boxes.find(b => /vix level/i.test(b.closest('.bp-fcell').textContent));
  vixBox.click();
  await wait(150);
  ok('ticking a filter changes the table',
     document.querySelector('.bp-table tr.total td:nth-child(2)').textContent !== before, true);
  ok('and the row states its cost',
     /no value for an active filter/.test(document.body.textContent), true);

  // ── P3: THE CURVES ────────────────────────────────────────────────
  // Charts are canvases; what matters is that they were built, carry the
  // data the table reports, and agree with it at the one number both show.
  const eq = Chart.getChart('bp-eq-chart');
  const dd = Chart.getChart('bp-dd-chart');
  const cap = Chart.getChart('bp-cap-chart');
  ok('the equity chart exists', !!eq, true);
  ok('a line per strategy plus the portfolio', eq && eq.data.datasets.length,
     c.chosen.length + 1);
  ok('the last dataset is the portfolio',
     eq && eq.data.datasets[eq.data.datasets.length - 1].label, 'TOTAL');
  // The curve ENDS at the portfolio's total P/L -- the table's own figure.
  const tot = eq.data.datasets.find(d => d.label === 'TOTAL').data;
  const tableTotal = document.querySelector('.bp-table tr.total td:nth-child(4)').textContent;
  ok('the curve ends at the table total, nothing being hatched',
     BP_DATA.curves.unfilteredTo ? tableTotal : bpFmtMoney(tot[tot.length - 1].y),
     tableTotal);

  ok('the drawdown chart exists', !!dd, true);
  const ddPts = dd.data.datasets[0].data;
  const trough = Math.min(...ddPts.map(p => p.y));
  const tableDD = document.querySelector('.bp-table tr.total td:nth-child(9)').textContent;
  // THE TROUGH IS THE TABLE'S MAX DD, not a shallower day-end reading.
  // ...when nothing is shaded. Once a filter is blind to part of the
  // history these charts deliberately draw trades the table does not count,
  // so the trough can be deeper than the table's Max DD. Same exception as
  // the curve's end total above.
  ok('the trough equals the table Max DD, nothing being shaded',
     BP_DATA.curves.unfilteredTo ? tableDD : bpFmtMoney(trough), tableDD);
  ok('the deepest point is marked', dd.data.datasets[1].data.length, 1);

  // ── TRADES THAT PREDATE THE MARKET DATA ───────────────────────────
  // index_ohlc starts 2017-01-03, so this strategy's first four years have
  // no VIX, no gap and no SPX session. They are still trades: they must
  // survive the load, be counted in the summary, and be drawn. Every other
  // fixture in this file spans a narrow recent window, which is why nothing
  // here could have caught a truncation at the market data's start.
  const cOld = Alpine.$data(document.querySelector('[x-data]'));
  const pOld = cOld.loaded.find(x => x.saved.id === 3);
  ok('the pre-2017 strategy loaded', !!pOld, true);
  ok('it really does predate the market data', pOld.date_min < '2017-01-03', true);
  ok('its trades survived the load', pOld.n, 84);
  ok('none of its trades were dropped server-side',
     pOld.columns.date_opened.filter(d => d < '2017-01-03').length > 0, true);
  const rowOld = cOld.rows.find(r => r.id === 3);
  ok('it has a summary row', !!rowOld, true);
  ok('the summary counts every one of its trades', rowOld.n, 84);
  ok('the summary drops none of them', rowOld.dropped, 0);
  // The portfolio total counts them too -- a strategy can be in the table
  // and still be missing from the pooled figures underneath it.
  const totOld = cOld.rows.find(r => r.total);
  ok('the TOTAL row includes them',
     totOld.nAll, cOld.rows.filter(r => !r.total).reduce((a, r) => a + r.nAll, 0));
  // AND ON THE CHARTS. Its own curve, the portfolio curve, and the axis.
  const itsCurve = BP_DATA.curves.eq.find(sv => sv.name === 'since2013');
  ok('its equity curve reaches back past the market data',
     itsCurve && itsCurve.points[0].date < '2017-01-03', true);
  const totCurve = BP_DATA.curves.eq.find(sv => sv.total);
  ok('so does the portfolio curve',
     totCurve && totCurve.points[0].date < '2017-01-03', true);
  const eqOld = Chart.getChart('bp-eq-chart');
  ok('and the axis is not truncated to the market data',
     obIsoDay(eqOld.scales.x.min) < '2017-01-03', true);
  // The drawdown chart shares that axis, so it must reach back as well.
  const ddOld = Chart.getChart('bp-dd-chart');
  ok('the drawdown axis reaches back too',
     obIsoDay(ddOld.scales.x.min) < '2017-01-03', true);

  // ── THE LIVE-STRATEGY SHADING ─────────────────────────────────────
  // On union dates the curve steepens as each strategy starts and flattens
  // as each finishes. The bands say so. The fixture is two strategies with
  // DIFFERENT spans, so there is a real edge to find.
  const cS = Alpine.$data(document.querySelector('[x-data]'));
  const shade = cS.liveShade();
  ok('there are three strategies to be fewer than', shade.total, 3);
  ok('the bands cover a rise and a fall', shade.bands.length >= 3, true);
  // A BINARY STATE, NOT A SCALE: either every strategy is live or some is
  // not. A stretch going from one strategy to two is ONE band, because both
  // are "not all" -- there is no separate treatment for 1 of 3 vs 2 of 3.
  let merged = true;
  for (let i = 1; i < shade.bands.length; i++) {
    if (shade.bands[i].partial === shade.bands[i - 1].partial) merged = false;
    if (shade.bands[i].from !== shade.bands[i - 1].to) merged = false;
  }
  ok('bands are contiguous and alternate state', merged, true);
  ok('some stretch has them all', shade.bands.some(b => !b.partial), true);
  ok('and the ends do not', shade.bands[0].partial
     && shade.bands[shade.bands.length - 1].partial, true);
  // THE BANDS ARE THE DRAWN LINES' OWN EDGES, in the charts' own units.
  //
  // These charts plot by CLOSE date. The payload's date_min is the earliest
  // OPEN and date_max the latest CLOSE -- a mixed basis -- so a strategy
  // whose first position was held for months had a span that began long
  // before its line did, and the band called it live over a stretch where it
  // had drawn nothing. The fixture's third strategy holds its first trade
  // 400 days precisely so the two cannot be confused.
  const lines = BP_DATA.curves.eq.filter(sv => !sv.total && sv.points.length);
  const starts = lines.map(sv => obDay(sv.points[0].date)).sort((a, b) => a - b);
  const ends = lines.map(sv => obDay(sv.points[sv.points.length - 1].date) + 1);
  ok('the first band starts where the first line starts',
     shade.bands[0].from, starts[0]);
  ok('the last band ends where the last line ends',
     shade.bands[shade.bands.length - 1].to, Math.max(...ends));
  // ALL LIVE ONLY ONCE EVERY LINE HAS STARTED.
  const firstFull = shade.bands.find(b => !b.partial);
  ok('all-live begins when the last line begins',
     firstFull.from, starts[starts.length - 1]);
  // AND IT IS NOT THE EARLIEST OPEN. A line begins at the EARLIEST CLOSE,
  // which is a whole holding period after the earliest entry -- and not
  // even the first trade's, since obByClose orders by exit and this
  // strategy holds its first position 400 days while the next closes in 20.
  // A band built on date_min would start at the entry and be wrong by that
  // gap, which on positions held days to months is months.
  const pLong = cS.loaded.find(x => x.saved.id === 3);
  const lineLong = BP_DATA.curves.eq.find(
    sv => sv.name === cS.rows.find(r => r.id === 3).name);
  const gap = obDay(lineLong.points[0].date) - obDay(pLong.date_min);
  ok('its earliest open is well before its line starts', gap > 60, true);
  ok('and the band starts with the line, not the open',
     starts[0], obDay(lineLong.points[0].date));
  ok('which is not where date_min is',
     shade.bands[0].from === obDay(pLong.date_min), false);
  // ONE STRATEGY HAS NOTHING TO BE FEWER THAN.
  ok('a single strategy gets no bands',
     bpLiveBands([cS.loaded[0]]).bands.length, 0);
  // BOTH SHADES ARE THE SAME GREY at two densities -- no second hue, no
  // texture. The only difference between them is how dense they are.
  ok('the two shades are one colour', BP_SHADE, '154,154,154');
  ok('and differ only in density',
     BP_SHADE_METRIC > BP_SHADE_STRATEGY, true);
  // THE PLUGIN IS ON THE THREE DATE CHARTS AND NOWHERE ELSE: the pairwise
  // scatter's x is dollars, so a band there would be nonsense.
  const hasShade = (ch) => !!(ch && (ch.config.plugins || [])
    .some(pl => pl.id === 'bpLiveShade'));
  ok('equity is shaded', hasShade(eq), true);
  ok('drawdown is shaded', hasShade(dd), true);
  ok('capital deployed is shaded', hasShade(cap), true);
  // AND THE BANDS SURVIVE AN UPDATE. draw() reuses the instance and only
  // reassigns options, so a constructor-array plugin is read once -- the
  // band data has to live in options.plugins to keep working.
  ok('the bands ride in options, not the constructor',
     eq.options.plugins.bpLiveShade.bands.length, shade.bands.length);
  // THE KEY IS ONE ENTRY, not one per count.
  const lk = cS.liveKey();
  ok('the key is a single entry', !!lk && !Array.isArray(lk), true);
  ok('it names the total', lk.total, 3);
  const legendEl = document.querySelector('.bp-liveleg');
  ok('the key is on the page', !!legendEl, true);
  ok('it says what is being counted',
     /strategies live/i.test(legendEl.textContent), true);

  ok('the deployed chart exists', !!cap, true);
  ok('deployment is a step', cap && cap.data.datasets[0].stepped, 'before');
  const capPeak = Math.max(...cap.data.datasets[0].data.map(p => p.y));
  const tablePeak = document.querySelector('.bp-table tr.total td:nth-child(16)').textContent;
  // ...when nothing is shaded. Once the chart draws trades the table does
  // not count, its peak can run above the table's figure -- the same
  // exception as the curve's end total and the drawdown trough.
  ok('its peak is the table peak, nothing being shaded',
     BP_DATA.curves.unfilteredTo ? tablePeak : bpFmtMoney(capPeak), tablePeak);

  // THE MONTHLY GRID sums to the same total.
  const cells = [...document.querySelectorAll('.bp-mcell')];
  ok('the monthly grid drew', cells.length > 0, true);
  const c2 = Alpine.$data(document.querySelector('[x-data]'));
  const monthSum = Object.values(c2.months.cells).reduce((a, b) => a + b, 0);
  ok('the months sum to the total', bpFmtMoney(monthSum), tableTotal);
  const yearSum = Object.values(c2.months.totals).reduce((a, b) => a + b, 0);
  ok('the years sum to the total', bpFmtMoney(yearSum), tableTotal);

  // A FILTER MOVES THE CURVES, not just the table.
  const endBefore = tot[tot.length - 1].y;
  // A DAY-OF-WEEK CATEGORY, unticked: the fixture's trades fall on several
  // weekdays, so dropping one genuinely narrows. (Exit Reason is one value
  // in this fixture, so filtering on it would change nothing and the
  // assertion would be testing the fixture.)
  const dowCell = [...document.querySelectorAll('.bp-fcell')]
    .find(el => /day of week/i.test(el.textContent));
  dowCell.querySelector('.bp-fhead input').click();      // switch it on
  await wait(120);
  ok('switching a categorical on keeps everything',
     /^all [0-9]+ kept$/.test(dowCell.querySelector('.bp-fstate').textContent.trim()), true);
  // FRIDAY, because this fixture's weekly strategy opens every Friday: any
  // other day would change nothing and the assertion would be testing the
  // fixture rather than the page.
  const friday = [...dowCell.querySelectorAll('.ob-check')]
    .find(l => /fri/i.test(l.textContent));
  friday.querySelector('input').click();
  await wait(200);
  const eq2 = Chart.getChart('bp-eq-chart');
  const tot2 = eq2.data.datasets.find(d => d.label === 'TOTAL').data;
  ok('the curve moved with the filter',
     tot2.length !== 0 && tot2[tot2.length - 1].y !== endBefore, true);

  // THE CARD'S FIELDS DO NOT OVERLAP. Two lines: name above, numbers below.
  const card = document.querySelector('.bp-card');
  const name = card.querySelector('.bp-name').getBoundingClientRect();
  const qty = card.querySelector('input[type=number]').getBoundingClientRect();
  ok('qty sits BELOW the name', qty.top >= name.bottom - 1, true);
  ok('qty is labelled', /qty/i.test(card.querySelector('.bp-flabel').textContent), true);
  ok('the card fits the sidebar',
     card.scrollWidth <= card.clientWidth + 1, true);

  // ── P4: CORRELATION ───────────────────────────────────────────────
  // Clear the day filter so both strategies are back in the portfolio.
  dowCell.querySelector('.bp-fhead input').click();
  await wait(200);
  const cc = Alpine.$data(document.querySelector('[x-data]'));
  ok('the matrix is square', cc.corr.matrix.length, cc.corr.names.length);
  ok('its diagonal is 1', cc.corr.matrix[0][0], 1);
  ok('it is symmetric',
     Math.abs(cc.corr.matrix[0][1] - cc.corr.matrix[1][0]) < 1e-12, true);
  ok('one pair per combination', cc.corr.pairs.length, 3);
  ok('the correlation is a real number',
     cc.corr.pairs[0].r !== null && Math.abs(cc.corr.pairs[0].r) <= 1, true);
  ok('it is weekly, not daily', cc.corr.weeks > 0 && cc.corr.weeks < 250, true);

  // The matrix agrees with the primitive computed straight from the series.
  const wk = BP_DATA.weekly;
  ok('the matrix equals obPearson on the same series',
     Math.abs(cc.corr.matrix[0][1] - obPearson(wk.cols[0], wk.cols[1])) < 1e-12, true);
  // Weeks where nothing closed anywhere are dropped.
  // ACROSS EVERY STRATEGY, not just the first two: a week is dropped only
  // when nothing closed ANYWHERE, so testing a pair would pass on a week
  // that a third strategy kept alive.
  ok('no all-zero weeks survive',
     wk.cols[0].some((v, i) => wk.cols.every(col => col[i] === 0)), false);

  ok('the scatter drew', !!Chart.getChart('bp-sc-chart'), true);
  ok('a point per week',
     Chart.getChart('bp-sc-chart').data.datasets[0].data.length, wk.weeks.length);
  ok('the rolling chart drew', !!Chart.getChart('bp-roll-chart'), true);
  ok('a rolling line per pair',
     Chart.getChart('bp-roll-chart').data.datasets.length, cc.corr.pairs.length);
  ok('rolling is bounded to -1..1',
     Chart.getChart('bp-roll-chart').options.scales.y.min, -1);

  // The metric table ranks by |rho| and never invents one below ten values.
  ok('metrics are listed', cc.corr.metrics.length > 0, true);
  const thin = cc.corr.metrics.filter(m => m.n < 10 && m.rho !== null);
  ok('no correlation from fewer than ten values', thin.length, 0);
  const rhos = cc.corr.metrics.filter(m => m.rho !== null).map(m => Math.abs(m.rho));
  ok('sorted by strength',
     rhos.every((v, i) => i === 0 || rhos[i - 1] >= v - 1e-12), true);

  // ── P6: DISTRIBUTION, OVERLAP, ROLLING RISK ───────────────────────
  const dist = Chart.getChart('bp-dist-chart');
  ok('the distribution drew', !!dist, true);
  ok('a series per strategy', dist && dist.data.datasets.length, cc.chosen.length);
  ok('the bars overlay rather than interleave',
     dist && dist.data.datasets[0].grouped, false);
  // Every trade lands in exactly one bin, so the counts sum to the trades.
  const binned = dist.data.datasets.reduce(
    (a, d) => a + d.data.reduce((x, y) => x + y, 0), 0);
  ok('every trade is in a bin', binned, cc.rows.find(r => r.total).n);

  const ov = Chart.getChart('bp-overlap-chart');
  ok('the overlap chart drew', !!ov, true);
  ok('a line per strategy plus the total', ov && ov.data.datasets.length,
     cc.chosen.length + 1);
  // FOUND BY ITS DASH, not by its index: the total is last however many
  // strategies there are, and an index here silently tested a strategy.
  ok('the total is dotted',
     !!(ov && ov.data.datasets[ov.data.datasets.length - 1].borderDash), true);
  // It counts the same way Capital deployed does: the portfolio total's
  // peak times capital is that chart's peak, so the two cannot disagree
  // about what a position is.
  const ovPeak = Math.max(...ov.data.datasets[2].data.map(p => p.y));
  ok('the overlap peak is a real count', ovPeak > 0, true);

  const risk = Chart.getChart('bp-risk-chart');
  ok('the risk chart drew', !!risk, true);
  ok('three series', risk && risk.data.datasets.length, 3);
  ok('win rate is on its own axis',
     risk && risk.data.datasets[2].yAxisID, 'y1');
  ok('that axis is a percentage', risk && risk.options.scales.y1.max, 100);
  ok('the window is in the labels',
     /\(90\)/.test(risk.data.datasets[0].label), true);

  // CHANGING THE WINDOW redraws with the new one.
  const riskSel = [...document.querySelectorAll('.ob-main select')]
    .find(el => [...el.options].some(o => o.value === '180'));
  riskSel.selectedIndex = 2;
  riskSel.dispatchEvent(new Event('input', { bubbles: true }));
  riskSel.dispatchEvent(new Event('change', { bubbles: true }));
  await wait(200);
  ok('the window control works', cc.riskWindow, 180);
  ok('and the chart says so',
     /\(180\)/.test(Chart.getChart('bp-risk-chart').data.datasets[0].label), true);
  // A WINDOW LONGER THAN THE DATA draws nothing -- and says so, instead of
  // leaving an empty chart whose axis falls back to 1970 and reads as broken.
  const fits = cc.risk.days >= 180;
  ok('the card states the window against the data',
     /days with a close/.test(document.body.textContent), true);
  if (!fits) {
    ok('it says the window does not fit',
       /fewer than the 180/.test(document.body.textContent), true);
    const rc = Chart.getChart('bp-risk-chart');
    ok('and the axis still spans the series, not 1970',
       rc.options.scales.x.min > obDay('2000-01-01'), true);
  }

  // ── THE CARDS SHOW THEIR FILTERS ──────────────────────────────────
  // Without these you cannot tell which of several strategies is filtered
  // without opening each panel in turn, which is the point of having them
  // side by side. The old app put them here and this was missed.
  const cards = [...document.querySelectorAll('.bp-card')];
  const weeklyCard = cards[1];
  const badges = [...weeklyCard.querySelectorAll('.bp-badge')]
    .map(b => b.textContent.trim());
  ok('the filtered strategy shows badges', badges.length > 0, true);
  ok('a range badge reads label and bounds',
     badges.some(t => t.includes('VIX Level:') && t.includes('–')), true);
  ok('the unfiltered strategy shows none',
     cards[0].querySelectorAll('.bp-badge').length, 0);
  // A range badge carries one decimal, as the old app wrote them.
  const vixBadge = badges.find(t => t.startsWith('VIX Level:'));
  ok('one decimal on the bounds', /[0-9]\.[0-9][^0-9]/.test(vixBadge + ' '), true);
  // The card carries the strategy's colour on its edge.
  ok('the card wears its colour',
     getComputedStyle(cards[0]).borderLeftWidth, '3px');

  // ── THE ANNUAL BARS SHARE THE MONTHS' ROWS ────────────────────────
  // They used to be a canvas beside the table, keeping its own vertical
  // rhythm, so a year's bar sat at a different height than that year's row
  // of months. The fix is structural -- one grid, one row per year -- and
  // this is the assertion that holds it there: measure both and compare.
  const bars = [...document.querySelectorAll('.bp-bcell')];
  ok('a bar per year', bars.length, cc.months.years.length);
  const rows = [...document.querySelectorAll('.bp-mrow')];
  let worst = 0;
  rows.forEach(r => {
    const cell = r.querySelector('.bp-mcell').getBoundingClientRect();
    const bar = r.querySelector('.bp-bcell').getBoundingClientRect();
    worst = Math.max(worst, Math.abs(cell.top - bar.top),
                     Math.abs(cell.height - bar.height));
  });
  ok('every bar is level with its own months', worst <= 1, true);
  // Direction and length come from that year's total.
  const firstYear = cc.months.years[0];
  const bar0 = cc.yearBar(firstYear);
  ok('the bar is coloured by sign',
     bar0.bg, cc.months.totals[firstYear] >= 0 ? '#3498db' : '#e84393');
  ok('the widest year fills the column',
     Math.round(Math.max(...cc.months.years.map(y => cc.yearBar(y).width))), 100);
  // THE REDUNDANT PER-YEAR NUMBER IS GONE. The bar carries that figure.
  ok('no year-total column beside December',
     document.querySelectorAll('.bp-ycell').length, 0);

  // NO HORIZONTAL SCROLL on the months. Twelve full figures, no scrollbar:
  // the reason the cells are flexible rather than min-width'd.
  const mgrid = document.querySelector('.bp-mgrid');
  const mwrap = document.querySelector('.bp-mwrap');
  ok('twelve month columns drew',
     getComputedStyle(mgrid).gridTemplateColumns.split(' ').length, 14);
  // MEASURED, NOT ASKED FOR A SCROLLBAR. .bp-mwrap overflows visibly at
  // this width, and a visible overflow reports scrollWidth == clientWidth --
  // so 'does it scroll' passed with the grid hanging 400px out of the card.
  // What fits is a question about edges, so compare edges.
  const gridBox = mgrid.getBoundingClientRect();
  const wrapBox = mwrap.getBoundingClientRect();
  const lastMonth = [...rows[0].querySelectorAll('.bp-mcell')].pop();
  const barCell = rows[0].querySelector('.bp-bcell').getBoundingClientRect();
  ok('the grid fits its column',
     gridBox.right <= wrapBox.right + 1, true);
  ok('December fits inside the grid',
     lastMonth.getBoundingClientRect().right <= gridBox.right + 1, true);
  ok('the bar column fits inside the grid',
     barCell.right <= gridBox.right + 1, true);
  // AND IT STILL FITS WHEN THE COLUMN IS NARROWER. At this window the old
  // fixed-width cells happened to fit too, so measuring only here proves
  // nothing: the scrollbar the user saw appeared because the table was given
  // a third of the row. Squeeze it and measure again -- month tracks that
  // flex survive, a fixed min-width does not.
  // Overlap is what a fixed cell width actually causes: the grid keeps its
  // width and the CELLS spill over each other and into the bars. So the
  // question is whether any cell crosses the next one.
  const overlap = (w) => {
    if (w) mwrap.style.width = w + 'px';
    void mwrap.offsetWidth;
    let worstGap = 0;
    rows.forEach(r => {
      const cs = [...r.querySelectorAll('.bp-mcell')].map(e => e.getBoundingClientRect());
      const bc = r.querySelector('.bp-bcell').getBoundingClientRect();
      for (let i = 0; i < cs.length - 1; i++)
        worstGap = Math.min(worstGap, cs[i + 1].left - cs[i].right);
      worstGap = Math.min(worstGap, bc.left - cs[cs.length - 1].right);
    });
    mwrap.style.width = '';
    void mwrap.offsetWidth;
    return worstGap;
  };
  ok('no cell overlaps its neighbour', overlap(0) >= -1, true);
  ok('nor at 900px', overlap(900) >= -1, true);
  ok('nor at 760px', overlap(760) >= -1, true);
  // A FULL FIGURE IS NOT CLIPPED. The cells shrank to fit rather than
  // abbreviating, so the thing to prove is that the text still fits in one.
  const widest = [...document.querySelectorAll('.bp-mcell')]
    .filter(el => el.textContent.trim().startsWith('$'))
    .sort((a, b) => b.textContent.length - a.textContent.length)[0];
  ok('the widest money cell is not clipped',
     !widest || widest.scrollWidth <= widest.clientWidth + 1, true);
  const monthCard = mgrid.closest('.ob-card');
  ok('nor does the card holding them',
     monthCard.scrollWidth <= monthCard.clientWidth + 1, true);

  // ── P5: PROFILES ──────────────────────────────────────────────────
  // Saved and reloaded by CLICKING, with the filters and the allocation it
  // was saved with.
  cc.chosen[0].qty = 4;
  cc.normalise(cc.chosen[0]);
  await wait(80);
  const nameBox = [...document.querySelectorAll('.ob-side input')]
    .find(el => el.placeholder && el.placeholder.startsWith('Name'));
  nameBox.value = 'live book';
  nameBox.dispatchEvent(new Event('input', { bubbles: true }));
  await wait(40);
  const saveBtn = side().find(b => b.textContent.trim() === 'Save');
  saveBtn.click();
  await wait(150);
  ok('the profile saved', window.__profiles.length, 1);
  ok('it carried the quantity', window.__profiles[0].payload.strategies[0].qty, 4);
  ok('it carried the filters',
     !!window.__profiles[0].payload.strategies[1].filters.day_of_week, true);
  ok('it carried the range mode',
     window.__profiles[0].payload.range_mode, cc.rangeMode);

  // A SECOND SAVE UNDER THE SAME NAME ASKS, it does not overwrite.
  saveBtn.click();
  await wait(150);
  ok('a duplicate name asks first', !!cc.profileClash, true);
  ok('and names the profile it would replace',
     cc.profileClash.existing_id, 1);
  side().find(b => /replace it/i.test(b.textContent)).click();
  await wait(150);
  ok('replacing clears the prompt', cc.profileClash, null);
  ok('and does not make a second profile', window.__profiles.length, 1);

  // RELOADING RESTORES THE COMBINATION.
  cc.chosen = [];
  cc.rows = [];
  await wait(80);
  const pfSel = selectBy('Saved combinations');
  // selectedIndex, not .value: picking the option a person would pick is
  // faithful whatever the ids are, and assigning .value silently does
  // nothing when the option list has not rendered yet -- which left the
  // model null and the Load button disabled.
  pfSel.selectedIndex = 1;
  pfSel.dispatchEvent(new Event('input', { bubbles: true }));
  pfSel.dispatchEvent(new Event('change', { bubbles: true }));
  await wait(60);
  ok('choosing a profile sets the model', cc.profilePick, window.__profiles[0].id);
  side().find(b => b.textContent.trim() === 'Load').click();
  await wait(300);
  ok('the profile loaded its strategies', cc.chosen.length, 3);
  ok('with the quantity it was saved with', cc.chosen[0].qty, 4);
  ok('with its names, not ids', cc.chosen[0].name, 'monthly');
  ok('and the table came back', document.querySelectorAll('.bp-table tr.total').length, 1);
  ok('the profile load fetched the trades too', cc.rows.length > 0, true);

  // FULL NUMBERS. $31k beside $7,853 cannot be compared at a glance, which
  // is the whole job of a summary table. Checked without a regex: every
  // backslash in this driver has to survive a Python string on the way in,
  // and two attempts at an escaped one broke the page instead.
  const money = [...document.querySelectorAll('.bp-table td, .bp-mcell, .bp-bfoot')]
    .map(el => el.textContent.trim())
    .filter(t => t.startsWith('$') || t.startsWith('-$'));
  const abbreviated = money.filter(t => t.endsWith('k') || t.endsWith('M'));
  ok('no abbreviated money on the page', abbreviated.length, 0);
  const totalCell = document.querySelector('.bp-table tr.total td:nth-child(4)')
    .textContent.trim();
  ok('big totals carry separators',
     totalCell.length < 6 || totalCell.includes(','), true);
  // ── A FILTER THAT CANNOT SEE THE WHOLE HISTORY ────────────────────
  // The distinction the whole feature rests on: Day of Week judges every
  // trade ever, so one it rejects is GENUINELY GONE. VIX cannot judge
  // anything before index_ohlc starts, so those trades are drawn and
  // hatched instead of silently shortening the curve.
  const cU = Alpine.$data(document.querySelector('[x-data]'));
  const sU = cU.chosen.find(x => x.id === 3);
  // FROM A CLEAN SLATE. Earlier checks leave filters on, and a blind filter
  // on ANY strategy hatches these charts -- right, but it makes "nothing
  // active" impossible to assert unless it is arranged first.
  for (const ch of cU.chosen) cU.resetFilters(ch);
  cU.recompute();
  await wait(150);
  ok('the slate is clean', BP_DATA.curves.unfilteredTo, null);

  // 1. A FILTER WITH NO COVERAGE LIMIT HATCHES NOTHING.
  // MONDAYS ONLY, so it genuinely REJECTS trades rather than keeping
  // everything -- the point is that what it rejects is gone from the charts
  // too, because it could judge those trades and did.
  sU.filters.day_of_week = { on: true, allowed: [0] };
  cU.recompute();
  await wait(150);
  ok('a filter that can judge everything shades nothing',
     BP_DATA.curves.unfilteredTo, null);
  ok('and draws no key for it', !!cU.unfilteredKey(), false);
  const rowD = cU.rows.find(r => r.id === 3);
  ok('it really did reject trades', rowD.dropped > 0, true);
  // A GAP IS NOT A COVERAGE START. Premium is missing on some trades but
  // has no minDate, so filtering on it drops them from the charts as well
  // -- there is no stretch of history it was blind to.
  sU.filters.day_of_week = { on: false, allowed: [] };
  sU.filters.premium = { on: true, lo: 0, hi: 1e9 };
  cU.recompute();
  await wait(150);
  const rowP = cU.rows.find(r => r.id === 3);
  ok('the premium gaps really are dropped', rowP.dropped > 0, true);
  ok('a gap in a metric is not a coverage start',
     BP_DATA.curves.unfilteredTo, null);
  ok('and those trades are dropped, not drawn',
     rowP.idxDraw.length, rowP.n);
  sU.filters.premium = { on: false, lo: 0, hi: 1e9 };
  cU.recompute();
  await wait(150);
  // A TUESDAY EXCLUDED BY A MONDAY FILTER IS GENUINELY GONE, not shaded:
  // the charts drop it exactly as the table does.
  ok('and the charts drop them too, not shade them',
     rowD.idxDraw.length, rowD.n);

  // 2. VIX IS BLIND BEFORE index_ohlc.
  const vixM = cU.registry.find(m => m.key === 'vix');
  ok('the fixture registry carries a coverage date', !!vixM.minDate, true);
  // Day of Week off again: leaving it on would narrow what VIX is blind
  // to, and this step is about VIX alone.
  sU.filters.day_of_week = { on: false, allowed: [] };
  sU.filters.vix = { on: true, lo: 9, hi: 80 };
  cU.recompute();
  await wait(150);
  // The hatch ends at the last UNFILTERED trade to CLOSE. That is not the
  // coverage date and must not be asserted to be: a trade entered before
  // coverage can close after it, and if none closes late the hatch stops
  // earlier. What must hold is that the key names the coverage date as the
  // reason, and that every added-back trade is inside the hatch (below).
  ok('the key names the coverage date as the reason',
     cU.unfilteredKey().from, vixM.minDate);
  ok('and names the metric that set it', cU.unfilteredKey().label, vixM.label);
  // THE SUMMARY IS UNCHANGED -- it still drops what the filter could not
  // judge, which is what was asked for.
  const rowU = cU.rows.find(r => r.id === 3);
  ok('the summary still drops them', rowU.dropped > 0, true);
  // THE CHARTS DO NOT.
  ok('the charts keep them', rowU.idxDraw.length > rowU.n, true);
  const nameU = cU.rows.find(r => r.id === 3).name;
  const itsU = BP_DATA.curves.eq.find(sv => sv.name === nameU);
  ok('the curve reaches back past the coverage date',
     itsU.points[0].date < vixM.minDate, true);
  const eqU = Chart.getChart('bp-eq-chart');
  const ddU = Chart.getChart('bp-dd-chart');
  ok('the equity axis is no longer cut to it',
     obIsoDay(eqU.scales.x.min) < vixM.minDate, true);
  ok('nor is the drawdown axis',
     obIsoDay(ddU.scales.x.min) < vixM.minDate, true);
  ok('equity carries the metric shade',
     eqU.options.plugins.bpUnfilteredShade.to, obDay(BP_DATA.curves.unfilteredTo));
  ok('drawdown carries the metric shade',
     ddU.options.plugins.bpUnfilteredShade.to, obDay(BP_DATA.curves.unfilteredTo));
  // THE INVARIANT THAT MAKES THE HATCH HONEST: every trade the chart adds
  // back sits inside it. One drawn unfiltered outside the hatch is mixing
  // with nothing marking it, which is the whole thing this prevents.
  const pU = BP_DATA.payloads[3];
  const keptU = new Set(rowU.idx);
  const addedU = rowU.idxDraw.filter(i => !keptU.has(i));
  ok('the chart added trades back', addedU.length > 0, true);
  ok('and every one of them closes inside the shade',
     addedU.every(i => pU.columns.date_closed[i] <= BP_DATA.curves.unfilteredTo), true);
  ok('the key counts exactly those trades', cU.unfilteredKey().n, addedU.length);
  // CAPITAL DEPLOYED DRAWS THE WHOLE SPAN TOO. Three charts in one pane
  // behaving differently is worse than the inconsistency the clipping
  // avoided, so it is shaded like the other two -- while the TABLE's peak
  // stays strict, which is the divergence the card has to state.
  const capU = Chart.getChart('bp-cap-chart');
  ok('capital deployed carries the metric shade',
     capU.options.plugins.bpUnfilteredShade.to, obDay(BP_DATA.curves.unfilteredTo));
  ok('its drawn peak is at or above the table figure',
     BP_DATA.curves.capPeakDrawn >= BP_DATA.curves.capPeakStrict, true);
  ok('and the card says the two can differ',
     /peak deployed/i.test(capU.canvas.closest('.ob-card').textContent), true);

  // THE THIRD TONE. Two shades are drawn but three appear, because they
  // compound where both hold. The key has to name what is on screen.
  const both = cU.bothKey();
  ok('the overlap has its own key entry', !!both, true);
  const composited = 1 - (1 - BP_SHADE_STRATEGY) * (1 - BP_SHADE_METRIC);
  ok('its tone is what compositing produces',
     both.bg, 'rgba(' + BP_SHADE + ',' + composited.toFixed(3) + ')');
  ok('which is darker than either alone',
     composited > BP_SHADE_METRIC && composited > BP_SHADE_STRATEGY, true);
  const keyEls = [...document.querySelectorAll('.bp-liveleg')];
  ok('three key rows render', keyEls.length >= 3, true);
  ok('one of them names both conditions',
     keyEls.some(el => /both at once/i.test(el.textContent)), true);

  // 3. TWO BLIND FILTERS HATCH TO THE LATER COVERAGE, NOT THE EARLIER:
  // nothing before the later one has passed both.
  const v9 = cU.registry.find(m => m.key === 'vix9d');
  sU.filters.vix9d = { on: true, lo: 5, hi: 90 };
  cU.recompute();
  await wait(150);
  ok('the later coverage wins', cU.unfilteredKey().from, v9.minDate);
  ok('and it really is the later of the two', v9.minDate > vixM.minDate, true);
  ok('the shade grew with it',
     BP_DATA.curves.unfilteredTo >= v9.minDate, true);
  // AND IT STOPS SOON AFTER THE COVERAGE DATE, not at the end of the data.
  // A scattered no-bar null late in the series used to drag it years past
  // the boundary; the shade is only ever as long as the trades it explains.
  const held = Math.max(...BP_DATA.payloads[3].columns.days_in_trade) + 1;
  const edge = obDay(BP_DATA.curves.unfilteredTo) - obDay(v9.minDate);
  ok('the shade stops within one holding period of coverage',
     edge >= 0 && edge <= held, true);
  ok('and nowhere near the end of the data',
     BP_DATA.curves.unfilteredTo < BP_DATA.payloads[3].date_max, true);

  // 4. BOTH KEYS ON THE PAGE, AND THEY ARE NOT THE SAME THING.
  const keyU = cU.unfilteredKey();
  ok('the key names the boundary', keyU.from, v9.minDate);
  ok('the key names the metric that set the boundary', keyU.label, v9.label);
  ok('the key counts the unfiltered trades', keyU.n > 0, true);
  // BOTH KEYS RENDER, and they are two rows rather than one combined one.
  const keyRows = [...document.querySelectorAll('.bp-liveleg')];
  ok('both keys render', keyRows.length >= 2, true);
  ok('one is the strategy key',
     keyRows.some(el => /strategies live/i.test(el.textContent)), true);
  ok('the other is the metric key',
     keyRows.some(el => /not filtered/i.test(el.textContent)), true);
  ok('the metric key names its coverage date',
     keyRows.some(el => el.textContent.includes(v9.minDate)), true);

  // 5. AND IT ALL COMES BACK OFF.
  sU.filters.vix = { on: false, lo: 9, hi: 80 };
  sU.filters.vix9d = { on: false, lo: 5, hi: 90 };
  sU.filters.day_of_week = { on: false, allowed: [] };
  cU.recompute();
  await wait(150);
  ok('clearing the filters clears the shade',
     BP_DATA.curves.unfilteredTo, null);
  // A HATCH WITHOUT A NAMED REASON IS A BUG: anything drawn unfiltered has
  // to be explained by a coverage-limited filter, or the page is shading a
  // stretch it cannot account for.
  ok('a shade never appears without a reason',
     !BP_DATA.curves.unfilteredTo || !!BP_DATA.curves.coverageFrom, true);
  ok('and the charts are what they were',
     cU.rows.find(r => r.id === 3).dropped, 0);

  // ── SURFACE METRICS AS PER-STRATEGY FILTERS ───────────────────────
  const cS2 = Alpine.$data(document.querySelector('[x-data]'));
  for (const ch of cS2.chosen) cS2.resetFilters(ch);
  cS2.recompute();
  await wait(120);

  // The panel is where the picker lives, and opening it loads the catalog.
  const sfSid = 3, sfOther = 1;
  cS2.toggleEdit(sfSid);
  await wait(200);
  ok('the surface catalog loaded', (cS2.surf.catalog || []).length, 3);
  // ── THE PICKER IS SEARCHABLE ──────────────────────────────────────
  // 452 metrics with unguessable names, so the query has to reach the
  // DESCRIPTION too -- matching column names only would be a list you can
  // already scroll.
  ok('unfiltered, every metric is offered',
     cS2.surfaceOptions().reduce((n, g) => n + g.options.length, 0), 3);
  cS2.surfQuery = 'skew';
  ok('a name query narrows it',
     cS2.surfaceOptions().reduce((n, g) => n + g.options.length, 0), 1);
  cS2.surfQuery = 'z-scored';
  ok('a DESCRIPTION query finds what the name does not say',
     cS2.surfaceOptions().flatMap(g => g.options.map(o => o.value)), ['z_iv_30d_atm']);
  ok('and empty groups disappear rather than sitting there empty',
     cS2.surfaceOptions().length, 1);
  cS2.surfQuery = 'zzzz';
  ok('no matches says so rather than showing an empty list',
     /nothing matches/.test(cS2.surfaceCount()), true);
  cS2.surfQuery = '';
  ok('clearing the query restores the list', cS2.surfaceCount(), '3 metrics');

  ok('grouped by family, in the legend order',
     cS2.surfaceOptions().map(g => g.label).join(' | '), 'IV · iv | Skew · skew');

  const sfBefore = cS2.rows.find(r => r.id === sfSid).n;
  const sfOtherBefore = cS2.rows.find(r => r.id === sfOther).n;
  await cS2.addSurfaceMetric(sfSid, 'iv_30d_atm');
  await wait(200);
  const sc = cS2.chosen.find(c => c.id === sfSid);
  const sfM = (sc.surface || [])[0];
  ok('the metric was added to that strategy', !!sfM, true);
  ok('and to NO sfOther strategy',
     (cS2.chosen.find(c => c.id === sfOther).surface || []).length, 0);
  ok('it was fetched once, for every trade of that strategy',
     window.__surfCalls.length && window.__surfCalls[0].n,
     BP_DATA.payloads[sfSid].n);

  // VALUES ALIGN BY INDEX. A shifted column is the failure that looks like
  // data, so the length is checked against the trade count, not trusted.
  const sfVals = BP_DATA.payloads[sfSid].columns[sfM.column];
  ok('the column is as long as the trade list', sfVals.length, BP_DATA.payloads[sfSid].n);
  ok('and is scaled to display units',
     Math.max(...sfVals.filter(v => v !== null)) > 1, true);
  ok("it enters that strategy's registry",
     cS2.registryFor(sc).some(x => x.key === sfM.key), true);
  ok('and not the shared one', cS2.registry.some(x => x.key === sfM.key), false);
  ok('the filter cell exists for it',
     [...document.querySelectorAll('.bp-fcell')]
       .some(el => el.textContent.includes('iv_30d_atm')), true);

  // THE COVERAGE COST, STATED BEFORE THE SLIDER MOVES. Split the way the
  // single-backtest page splits it: a stretch of history the metric does not
  // reach, and scattered entries with no bar.
  const sfCost = cS2.surfaceCost(sc, sfM);
  const sfDates = BP_DATA.payloads[sfSid].columns.date_opened;
  const sfHandBefore = sfVals.filter((v, i) => v === null && sfDates[i] < '2020-01-02').length;
  const sfHandNoBar = sfVals.filter((v, i) => v === null && sfDates[i] >= '2020-01-02').length;
  ok('the sfCost counts every trade with no value',
     sfCost.dropped, sfVals.filter(v => v === null).length);
  ok('sfBefore-coverage matches a hand count', sfCost.before, sfHandBefore);
  ok('no-bar matches a hand count', sfCost.noBar, sfHandNoBar);
  ok('both kinds really occur in the fixture',
     sfCost.before > 0 && sfCost.noBar > 0, true);
  const sfTxt = cS2.surfaceCostText(sc, sfM);
  ok('the text names the coverage date', sfTxt.includes('2020-01-02'), true);
  ok('and says it WOULD drop, sfBefore the filter is on',
     sfTxt.includes('would drop'), true);

  // TURNING IT ON NARROWS THIS STRATEGY AND LEAVES THE OTHERS ALONE.
  sc.filters[sfM.key] = { on: true, lo: sfM.min, hi: sfM.max };
  cS2.recompute();
  await wait(150);
  const sfAfter = cS2.rows.find(r => r.id === sfSid).n;
  ok('the filtered strategy lost exactly the trades with no value',
     sfBefore - sfAfter, sfCost.dropped);
  ok('the sfOther strategy is untouched',
     cS2.rows.find(r => r.id === sfOther).n, sfOtherBefore);
  const sfBadge = cS2.badges(sc).find(b => b.key === sfM.key);
  ok('the badge names it', !!sfBadge, true);
  // AT THE METRIC'S OWN PRECISION. The built-ins print one decimal; this one
  // is in vol points to two, and its unit is on the badge -- one decimal
  // would have read as a range of zeros for a small-scale metric.
  ok("the badge carries the metric's own decimals and unit",
     /10\.00–13\.2[0-9] vol pts/.test(sfBadge.text), true);
  ok('and the text now says it IS dropping',
     cS2.surfaceCostText(sc, sfM).includes('is dropping'), true);

  // IT IS A COVERAGE-LIMITED FILTER, so it drives the charts' metric shade.
  ok('it sets the unfiltered boundary',
     BP_DATA.curves.coverageFrom, '2020-01-02');

  // A SECOND METRIC WITH A LATER START WINS THE BOUNDARY.
  await cS2.addSurfaceMetric(sfSid, 'z_iv_30d_atm');
  await wait(200);
  const sfMz = sc.surface.find(x => x.surfColumn === 'z_iv_30d_atm');
  sc.filters[sfMz.key] = { on: true, lo: sfMz.min, hi: sfMz.max };
  cS2.recompute();
  await wait(150);
  ok('the later coverage sets the boundary',
     BP_DATA.curves.coverageFrom, '2021-04-05');

  // REMOVING PUTS IT BACK, and re-adding does NOT hit the server again.
  const sfCalls = window.__surfCalls.length;
  cS2.removeSurfaceMetric(sfSid, sfMz.key);
  await wait(150);
  ok('removing drops the cell', (sc.surface || []).length, 1);
  ok('and its column', BP_DATA.payloads[sfSid].columns[sfMz.column], undefined);
  await cS2.addSurfaceMetric(sfSid, 'z_iv_30d_atm');
  await wait(200);
  ok('re-adding is served from the cache', window.__surfCalls.length, sfCalls);

  // ── A PROFILE CARRIES THE ADDED METRICS ───────────────────────────
  // The filter VALUE always survived. Without the list beside it the value
  // came back describing nothing: bpSpecs walks the registry, not the
  // filter map, so the slider was gone and the trades were not filtered --
  // a filter you could see in the payload and not on the page.
  const sfNarrow = { on: true, lo: sfM.min, hi: sfM.min + (sfM.max - sfM.min) / 2 };
  sc.filters[sfM.key] = { ...sfNarrow };
  cS2.recompute();
  await wait(150);
  const sfNarrowed = cS2.rows.find(r => r.id === sfSid).n;
  ok('a narrowed surface filter drops more than the nulls',
     sfNarrowed < sfBefore - sfCost.dropped, true);

  cS2.profileName = 'surface book';
  await cS2.saveProfile(false);
  await wait(200);
  const saved = window.__profiles.find(x => x.name === 'surface book');
  ok('the profile saved', !!saved, true);
  const savedStrat = saved.payload.strategies.find(x => x.id === sfSid);
  ok('it carries the metric list',
     savedStrat.surface.includes('iv_30d_atm')
     && savedStrat.surface.includes('z_iv_30d_atm'), true);
  ok('and the strategy with none carries an empty list',
     (saved.payload.strategies.find(x => x.id === sfOther).surface || []).length, 0);

  // Wipe the page, then bring it back from the profile alone.
  cS2.clearAll();
  await wait(120);
  ok('cleared', cS2.chosen.length, 0);
  cS2.profilePick = saved.id;
  await cS2.loadProfile();
  await wait(400);
  const sc2 = cS2.chosen.find(c => c.id === sfSid);
  ok('the strategies came back', cS2.chosen.length, 3);
  ok('with their added metrics', (sc2.surface || []).length, 2);
  ok('as real registry entries, not orphan filter keys',
     cS2.registryFor(sc2).some(x => x.key === sfM.key), true);
  ok('their values were refetched for this strategy',
     (BP_DATA.payloads[sfSid].columns[sfM.column] || []).length,
     BP_DATA.payloads[sfSid].n);
  ok('the filter is still ON', (sc2.filters[sfM.key] || {}).on, true);
  ok('with the bounds it was saved with, not the full range',
     [sc2.filters[sfM.key].lo, sc2.filters[sfM.key].hi],
     [sfNarrow.lo, sfNarrow.hi]);
  ok('so the strategy is filtered exactly as it was',
     cS2.rows.find(r => r.id === sfSid).n, sfNarrowed);
  ok('and the badge is back on its card',
     cS2.badges(sc2).some(b => b.key === sfM.key), true);

  } catch (e) {
    // A THROW IS A FINDING, not a lost run: without this the page simply
    // never reports and the gate can only say "it did not get that far".
    out.push('FAIL|the driver threw|' + (e && e.message ? e.message : e) + '|no throw');
  }
  out.push('errs|' + (window.__errs.length ? window.__errs.slice(0, 3).join(' ~ ') : 'clean'));
  const pre = document.createElement('pre');
  pre.id = 'report';
  pre.textContent = out.join('\\n');
  document.body.appendChild(pre);
}, 400));
</script>
"""


def main() -> int:
    browser = find_browser()
    if browser is None:
        print("  SKIP  no Edge/Chromium on this host — the page was NOT driven")
        return EXIT_SKIPPED
    try:
        page = build_page()
    except Exception as exc:                              # noqa: BLE001
        print(f"  the page could not be rendered: {type(exc).__name__}: {exc}")
        return 1
    tmp = ROOT / "scripts" / "_portfolio_ui.html"
    tmp.write_text(page, encoding="utf-8")
    try:
        # `--shot PATH` also writes the picture. The gate reads numbers; a
        # person reviewing a layout change wants to look at it, and building
        # the page twice from two scripts is how the two drift.
        if "--shot" in sys.argv:
            shot = sys.argv[sys.argv.index("--shot") + 1]
            subprocess.run(
                [browser, "--headless=new", "--disable-gpu",
                 "--window-size=1600,3800", f"--screenshot={shot}",
                 "--virtual-time-budget=60000", tmp.as_uri()],
                capture_output=True, timeout=180)
            print(f"  wrote {shot}")
        p = subprocess.run(
            [browser, "--headless=new", "--disable-gpu", "--window-size=1600,1000",
             "--dump-dom", "--virtual-time-budget=60000", tmp.as_uri()],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=180)
    finally:
        tmp.unlink(missing_ok=True)

    m = re.search(r'<pre id="report">(.*?)</pre>', p.stdout, re.S)
    if not m:
        print("  the page never reported — it did not get that far")
        print("   ", (p.stdout or "")[-400:].replace("\n", " ")[:400])
        return 1

    import html as _html
    fails = 0
    for line in _html.unescape(m.group(1)).strip().splitlines():
        parts = line.split("|")
        if parts[0] == "errs":
            if parts[1] != "clean":
                print(f"  FAIL  the console is not clean: {parts[1]}")
                fails += 1
            continue
        if len(parts) < 4:
            # A NOTE rather than an assertion: the driver can report context
            # ("what the page said when it failed") and a strict parser here
            # turned that into an IndexError instead of printing it.
            print(f"  {line.strip()}")
            continue
        status, name, got, want = parts[0], parts[1], parts[2], parts[3]
        if status != "ok":
            print(f"  FAIL  {name}: {got} (want {want})")
            fails += 1
    n = sum(1 for ln in _html.unescape(m.group(1)).strip().splitlines()
            if ln.startswith("ok|") or ln.startswith("FAIL|"))
    print(f"portfolio UI: {n} clicked assertions, failures: {fails}")
    return 1 if fails else 0


sys.exit(main())
