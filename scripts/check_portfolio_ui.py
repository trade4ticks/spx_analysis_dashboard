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

    sessions = [d.date().isoformat() for d in pd.bdate_range("2023-01-02", "2026-12-31")]

    def payload(sid, name, color, cap, n, start, step, hold, pnls, vix_from):
        d0 = dt.date.fromisoformat(start)
        op = [(d0 + dt.timedelta(days=i * step)).isoformat() for i in range(n)]
        cl = [(dt.date.fromisoformat(o) + dt.timedelta(days=hold)).isoformat() for o in op]
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
                "days_in_trade": [hold] * n,
                "day_of_week": [dt.date.fromisoformat(d).weekday() for d in op],
                "exit_reason": ["profit target"] * n,
                # A metric whose coverage starts late, so a filter on it has
                # a cost the panel has to state.
                "vix_level": [None if i < vix_from else 14 + (i % 9) for i in range(n)],
            },
        }

    a = payload(1, "monthly", "#3498db", 25000, 40, "2023-01-03", 30, 30, [100.0], 0)
    b = payload(2, "weekly", "#e84393", 10000, 40, "2023-01-06", 7, 7, [300.0, -100.0], 10)
    reg = [m for m in registry_with_coverage(None) if m.get("filter")]
    return html.replace("</head>", (STUB % (json.dumps([a, b]), json.dumps(reg))) + "</head>", 1)


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
    {id:2,name:'weekly',trade_count:40,capital_per_position:10000}],
    colors:['#3498db','#e84393'], max:12};
  if (String(u).includes('/registry')) return {metrics: %s};
  if (String(u).includes('/load')) return {strategies: LOADED,
    load: {seconds: 0.2, n: 2, from_cache: 2, parsed: 0}};
  return {};
}});

// A profile store, in memory, behaving like the endpoints: 409 on a name
// that exists, and a load that reports which strategies are gone.
window.__profiles = [];
const realFetch = window.fetch;
window.fetch = async (u, init) => {
  const url = String(u);
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
        ...rec, missing, names: { '1': 'monthly', '2': 'weekly' } } }) };
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
  }, 3000);
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
  ok('two strategies added', c.chosen.length, 2);

  // THE BUTTON WORKS BEFORE A LOAD. It was disabled here, which is the
  // regression this file exists for.
  const early = filterBtns();
  ok('a filter button per strategy', early.length, 2);
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
  ok('a line per strategy plus the portfolio', eq && eq.data.datasets.length, 3);
  ok('the last dataset is the portfolio',
     eq && eq.data.datasets[eq.data.datasets.length - 1].label, 'TOTAL');
  // The curve ENDS at the portfolio's total P/L -- the table's own figure.
  const tot = eq.data.datasets.find(d => d.label === 'TOTAL').data;
  const tableTotal = document.querySelector('.bp-table tr.total td:nth-child(4)').textContent;
  ok('the curve ends at the table total',
     bpFmtMoney(tot[tot.length - 1].y), tableTotal);

  ok('the drawdown chart exists', !!dd, true);
  const ddPts = dd.data.datasets[0].data;
  const trough = Math.min(...ddPts.map(p => p.y));
  const tableDD = document.querySelector('.bp-table tr.total td:nth-child(9)').textContent;
  // THE TROUGH IS THE TABLE'S MAX DD, not a shallower day-end reading.
  ok('the trough equals the table Max DD', bpFmtMoney(trough), tableDD);
  ok('the deepest point is marked', dd.data.datasets[1].data.length, 1);

  ok('the deployed chart exists', !!cap, true);
  ok('deployment is a step', cap && cap.data.datasets[0].stepped, 'before');
  const capPeak = Math.max(...cap.data.datasets[0].data.map(p => p.y));
  const tablePeak = document.querySelector('.bp-table tr.total td:nth-child(16)').textContent;
  ok('its peak is the table peak', bpFmtMoney(capPeak), tablePeak);

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
  ok('one pair for two strategies', cc.corr.pairs.length, 1);
  ok('the correlation is a real number',
     cc.corr.pairs[0].r !== null && Math.abs(cc.corr.pairs[0].r) <= 1, true);
  ok('it is weekly, not daily', cc.corr.weeks > 0 && cc.corr.weeks < 250, true);

  // The matrix agrees with the primitive computed straight from the series.
  const wk = BP_DATA.weekly;
  ok('the matrix equals obPearson on the same series',
     Math.abs(cc.corr.matrix[0][1] - obPearson(wk.cols[0], wk.cols[1])) < 1e-12, true);
  // Weeks where nothing closed anywhere are dropped.
  ok('no all-zero weeks survive',
     wk.cols[0].some((v, i) => v === 0 && wk.cols[1][i] === 0), false);

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
  ok('a series per strategy', dist && dist.data.datasets.length, 2);
  ok('the bars overlay rather than interleave',
     dist && dist.data.datasets[0].grouped, false);
  // Every trade lands in exactly one bin, so the counts sum to the trades.
  const binned = dist.data.datasets.reduce(
    (a, d) => a + d.data.reduce((x, y) => x + y, 0), 0);
  ok('every trade is in a bin', binned, cc.rows.find(r => r.total).n);

  const ov = Chart.getChart('bp-overlap-chart');
  ok('the overlap chart drew', !!ov, true);
  ok('a line per strategy plus the total', ov && ov.data.datasets.length, 3);
  ok('the total is dotted',
     !!(ov && ov.data.datasets[2].borderDash), true);
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

  // ── THE ANNUAL BARS, beside the monthly grid ──────────────────────
  const yr = Chart.getChart('bp-year-chart');
  ok('the annual bar chart drew', !!yr, true);
  ok('a bar per year', yr && yr.data.labels.length, cc.months.years.length);
  ok('the bars are the year totals',
     yr && yr.data.datasets[0].data[0],
     cc.months.totals[cc.months.years[0]]);

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
  ok('the profile loaded its strategies', cc.chosen.length, 2);
  ok('with the quantity it was saved with', cc.chosen[0].qty, 4);
  ok('with its names, not ids', cc.chosen[0].name, 'monthly');
  ok('and the table came back', document.querySelectorAll('.bp-table tr.total').length, 1);
  ok('the profile load fetched the trades too', cc.rows.length > 0, true);

  // FULL NUMBERS. $31k beside $7,853 cannot be compared at a glance, which
  // is the whole job of a summary table. Checked without a regex: every
  // backslash in this driver has to survive a Python string on the way in,
  // and two attempts at an escaped one broke the page instead.
  const money = [...document.querySelectorAll('.bp-table td, .bp-mcell, .bp-ycell')]
    .map(el => el.textContent.trim())
    .filter(t => t.startsWith('$') || t.startsWith('-$'));
  const abbreviated = money.filter(t => t.endsWith('k') || t.endsWith('M'));
  ok('no abbreviated money on the page', abbreviated.length, 0);
  const totalCell = document.querySelector('.bp-table tr.total td:nth-child(4)')
    .textContent.trim();
  ok('big totals carry separators',
     totalCell.length < 6 || totalCell.includes(','), true);
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
                 "--virtual-time-budget=6000", tmp.as_uri()],
                capture_output=True, timeout=180)
            print(f"  wrote {shot}")
        p = subprocess.run(
            [browser, "--headless=new", "--disable-gpu", "--window-size=1600,1000",
             "--dump-dom", "--virtual-time-budget=6000", tmp.as_uri()],
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
