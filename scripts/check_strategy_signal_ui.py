"""The Strategy Signal page, DRIVEN BY CLICKING, in headless Edge.

The board it draws is produced by THE REAL ROUTER over fabricated bars --
store and fetch are replaced, the decision, series and payload code are not
-- so the page is read against what the server really sends. The page's
fetch is stubbed in the browser; its own JS and template are the shipped
files, inlined so file:// works (the check_portfolio_ui pattern).

What it asks, as a person would:
  the schedule   a strategy off its scheduled day still reads TRADE when its
                 conditions pass; the checklist has no day row.
  the calendar   cards on the right day and 30-minute row; several times per
                 strategy; two strategies in one row side by side without
                 overlapping; a scheduled day with no time in the "no time"
                 row; standalone cards beside it and never on it; a calendar
                 strategy with no day beside it too -- every strategy
                 somewhere; one state per strategy wherever it is drawn
                 (mini card, side card, section) because all read one
                 decision; today's column and the time line in EASTERN time,
                 interpolated within its row; no line after the close or on
                 a past date; clicking a card opens a collapsed section.
  layout         two columns side by side without horizontal overflow, back
                 to full width; both and the collapsed set read from and
                 written to local storage.
  images         thumb URLs carry the version; Upload / Replace / Remove are
                 staged and sent by Save, after the config; a failed upload
                 keeps the panel open and says the strategy was saved.
  the editor     times per day, display, a payload the store accepts.
And the console stays clean -- an Alpine expression error leaves a control
inert without stopping the page.

`--shot PATH` also writes a screenshot. No Edge: SKIP, not PASS.
"""
from __future__ import annotations

import asyncio
import json
import math
import re
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

EXIT_SKIPPED = 3
EDGE_CANDIDATES = [
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    "/usr/bin/microsoft-edge", "/usr/bin/chromium", "/usr/bin/google-chrome",
]
NOW = pd.Timestamp("2026-09-22 14:30", tz="America/New_York")   # a Tuesday
PAST = "2026-09-18"                                              # the Friday before

VIX_SIG = lambda cmp, th: [{"id": "v", "label": "", "a": "index:vix", "op": None, "b": None, "transform": None,  # noqa: E731
                            "chart": False, "resolution": "daily", "lookback": "1y", "signal": True, "cmp": cmp,
                            "thresholds": [th]}]

STRATEGIES = [
    # Allantis' shape: watched for weeks, a standalone card, with a picture.
    {"id": 1, "position": 1, "name": "Skew allocation", "weekdays": [], "entry_times": [], "display": "card",
     "image": {"v": 1790812800000, "w": 1200, "h": 600, "tw": 480, "th": 240},
     "states": ["FULL", "REDUCED", "MINIMAL", "NONE"], "logic": "and", "manual": [],
     "notes": "Allocation scales down as 90d skew gets rich against its own year.",
     "metrics": [
         {"id": "raw", "label": "", "a": "surface:skew_90d_25p_25c", "op": None, "b": None, "transform": None,
          "chart": True, "resolution": "daily", "lookback": "1y", "signal": False, "cmp": None, "thresholds": []},
         {"id": "pct", "label": "90d skew pctl", "a": "surface:skew_90d_25p_25c", "op": None, "b": None,
          "transform": "pctile_252", "chart": True, "resolution": "daily", "lookback": "6m", "signal": True,
          "cmp": "<", "thresholds": [30, 50, 101]}]},
    # Two scheduled times, two days.
    {"id": 2, "position": 2, "name": "Short-term put spread", "weekdays": [2, 5], "display": "calendar",
     "entry_times": [{"day": 2, "time": "10:00"}, {"day": 5, "time": "11:30"}], "image": None,
     "states": ["TRADE", "NO TRADE"], "logic": "and", "manual": ["Premium must be >= $2.50"],
     "notes": "VIX/VIX9D below 1.14 with a flat-to-upward front term structure.",
     "metrics": [
         {"id": "r", "label": "", "a": "index:vix", "op": "/", "b": "index:vix9d", "transform": None,
          "chart": True, "resolution": "intraday", "lookback": "10d", "signal": True, "cmp": "<",
          "thresholds": [1.14]},
         {"id": "t", "label": "", "a": "surface:term_ratio_7d_30d", "op": None, "b": None, "transform": None,
          "chart": True, "resolution": "intraday", "lookback": "10d", "signal": True, "cmp": ">",
          "thresholds": [0.8375]}]},
    # A row saved BEFORE the schedule fields: Friday only, no times, no
    # display. It used to read NO TRADE on a Tuesday; it reads its market.
    {"id": 3, "position": 3, "name": "Friday iron fly", "weekdays": [5],
     "states": ["TRADE", "NO TRADE"], "logic": "and", "manual": ["Check entry delta"], "notes": "",
     "metrics": [
         {"id": "r", "label": "", "a": "index:vix", "op": None, "b": None, "transform": None,
          "chart": True, "resolution": "daily", "lookback": "3m", "signal": True, "cmp": "<",
          "thresholds": [40]}]},
    # On the calendar by setting, but on no day: shown beside it, not lost.
    {"id": 4, "position": 4, "name": "Needs missing data", "weekdays": [], "entry_times": [], "display": "calendar",
     "image": None, "states": ["TRADE", "NO TRADE"], "logic": "and", "manual": [], "notes": "",
     "metrics": [
         {"id": "g", "label": "", "a": "surface:iv_30d_atm", "op": None, "b": None, "transform": None,
          "chart": False, "resolution": "daily", "lookback": "1y", "signal": True, "cmp": ">",
          "thresholds": [0.1]}]},
    # Two strategies in ONE row (Wed 10:00-10:30): 10:15 and 10:00.
    {"id": 5, "position": 5, "name": "Dbl-Cal 7/9 DTE", "weekdays": [3], "display": "calendar",
     "entry_times": [{"day": 3, "time": "10:15"}],
     "image": {"v": 1790812900000, "w": 800, "h": 400, "tw": 480, "th": 240},
     "states": ["TRADE", "NO TRADE"], "logic": "and", "manual": [], "notes": "", "metrics": VIX_SIG(">", 40)},
    {"id": 6, "position": 6, "name": "Day Trade 1:7 DTE Diagonal", "weekdays": [3], "display": "calendar",
     "entry_times": [{"day": 3, "time": "10:00"}], "image": None,
     "states": ["TRADE", "NO TRADE"], "logic": "and", "manual": [], "notes": "", "metrics": VIX_SIG("<", 40)},
]

SOURCES = {
    "index": [{"id": "index:vix", "label": "VIX", "description": "CBOE 30-day volatility index"},
              {"id": "index:vix9d", "label": "VIX9D", "description": "CBOE 9-day volatility index"}],
    "surface": [{"id": "surface:skew_90d_25p_25c", "column": "skew_90d_25p_25c", "family": "skew",
                 "group": "Skew", "form": "level", "form_label": "Level", "units": "vol_decimal",
                 "description": "90d 25-delta put minus call IV", "min_date": "2020-01-02"},
                {"id": "surface:term_ratio_7d_30d", "column": "term_ratio_7d_30d", "family": "term_ratio",
                 "group": "Term structure", "form": "level", "form_label": "Level", "units": "ratio",
                 "description": "7d ATM IV / 30d ATM IV", "min_date": "2020-01-02"}],
    "surface_last_date": "2026-09-22",
    "operators": ["+", "-", "*", "/"],
    "transforms": [{"id": "pctile_252", "label": "Percentile, trailing 252 sessions", "description": ""}],
    "lookbacks": [{"id": k, "label": k, "sessions": 0, "intraday": k in ("5d", "10d", "1m", "3m")}
                  for k in ("5d", "10d", "1m", "3m", "6m", "1y", "2y")],
    "cmps": ["<", "<=", ">", ">=", "="],
}


def board_json() -> str:
    """The board, from the real router, over fabricated bars."""
    from app.routers import strategy_signal as router
    from app.strategy_signal import library as lib, store

    def wave(src, i, k):
        x = i + k / 78
        if src == "index:vix":
            return 17 + 2 * math.sin(x / 7)
        if src == "index:vix9d":
            return 16.5 + 2 * math.sin(x / 7) + 0.3 * math.sin(x * 2)
        if src == "surface:term_ratio_7d_30d":
            return 0.9 + 0.05 * math.sin(x / 3)
        if src == "surface:skew_90d_25p_25c":
            return 0.06 + 0.01 * math.sin(x / 40) + 0.002 * math.sin(x)
        raise KeyError(src)

    async def fetch(pool, src, sessions):
        if src == "surface:iv_30d_atm":
            return pd.Series(dtype=float)
        idx, vals = [], []
        for i, d in enumerate(sessions):
            base = datetime.fromisoformat(d) + timedelta(hours=9, minutes=35)
            last = 78 if d < NOW.date().isoformat() else 60      # today is in progress
            for k in range(last):
                idx.append(base + timedelta(minutes=5 * k))
                vals.append(wave(src, i, k))
        return pd.Series(vals, index=pd.DatetimeIndex(idx))

    async def listing(pool):
        # What store._row hands back: the read-time defaults applied, so the
        # legacy row (id 3) arrives exactly as a pre-schedule row would.
        return [{**store.with_defaults(s), "image": s.get("image")} for s in json.loads(json.dumps(STRATEGIES))]

    lib.fetch_bars, store.list_strategies, lib.now_et = fetch, listing, (lambda: NOW)
    # Live (a Tuesday) and a past Friday, both from the real router.
    return json.dumps({"live": asyncio.run(router.board(as_of=None, pool=None)),
                       "past": asyncio.run(router.board(as_of=PAST, pool=None))})


DRIVER = r"""
<script>
// Headless virtual time runs no animation frames, and Alpine SHOWS an x-show
// element from one (as Chart.js resizes from one): without this every x-show
// stays hidden and a chart drawn while collapsed never takes its size
// (the check_oo_notes_ui shim).
window.requestAnimationFrame = cb => setTimeout(() => cb(performance.now()), 0);
// Stored preferences as a previous visit left them: two columns, section 3
// collapsed. The page must START that way -- that is the persistence.
try { localStorage.setItem('ss.layout', JSON.stringify('two'));
      localStorage.setItem('ss.collapsed', JSON.stringify({ 3: true })); } catch (e) {}
(function () {
  const errs = [];
  window.addEventListener('error', e => errs.push('onerror: ' + e.message + ' @ ' + String((e.error && e.error.stack) || '').split('\n').slice(1, 4).join(' ')));
  const ce = console.error.bind(console);
  console.error = (...a) => { errs.push('console.error: ' + a.map(String).join(' ')); ce(...a); };
  const sent = [];
  let failImage = false;
  window.fetch = async (url, opts) => {
    const u = String(url), m = (opts && opts.method) || 'GET';
    const ok = body => ({ ok: true, status: 200, json: async () => body, text: async () => JSON.stringify(body) });
    if (u.endsWith('/board')) return ok(BOARDS.live);
    if (u.endsWith('/board?as_of=' + BOARDS.past.date)) return ok(BOARDS.past);
    if (u.endsWith('/sources')) return ok(SOURCES);
    if (u.includes('/strategies')) {
      sent.push({ u, m, body: opts && opts.body });
      if (u.endsWith('/image') && failImage) {
        failImage = false;
        return { ok: false, status: 400, json: async () => ({ detail: 'not an image' }), text: async () => '' };
      }
      return ok({ strategy: {} });
    }
    return { ok: false, status: 404, json: async () => ({}), text: async () => 'no stub for ' + u };
  };
  const out = [];
  const eq = (name, got, want) => out.push(`${JSON.stringify(got) === JSON.stringify(want) ? 'ok' : 'FAIL'}|${name}|${JSON.stringify(got)}|${JSON.stringify(want)}`);
  const wait = ms => new Promise(r => setTimeout(r, ms));
  const $ = s => document.querySelector(s), $$ = s => [...document.querySelectorAll(s)];
  const txt = el => (el ? el.textContent.trim().replace(/\s+/g, ' ') : null);
  const rgb = hex => `rgb(${parseInt(hex.slice(1, 3), 16)}, ${parseInt(hex.slice(3, 5), 16)}, ${parseInt(hex.slice(5, 7), 16)})`;
  const setVal = (el, v) => { el.value = v; el.dispatchEvent(new Event('input', { bubbles: true })); el.dispatchEvent(new Event('change', { bubbles: true })); };
  const byName = n => BOARD.strategies.find(e => e.strategy.name === n);
  const stOf = e => (e.decision.state === null ? 'NO DATA' : e.strategy.states[e.decision.state]);
  const cell = (d, s) => $(`.ss-cal-cell[data-day="${d}"][data-slot="${s}"]`);
  const minis = el => (el ? [...el.querySelectorAll('.ss-mini')].map(m => txt(m.querySelector('.ss-mini-name'))) : null);
  const secOf = id => document.getElementById('ss-sec-' + id);
  const visible = el => !!el && el.offsetParent !== null && getComputedStyle(el).display !== 'none';

  async function run() {
    for (let i = 0; i < 100 && !$$('.ss-sec').length; i++) await wait(50);
    await wait(300);
    const app = Alpine.$data(document.body);

    // ── the clock: Eastern by the zone's rules, across daylight saving ───
    eq('ET in summer (EDT): 18:30Z is 14:30 Tue', [ssEt(new Date('2026-09-22T18:30:00Z')).wd, ssEt(new Date('2026-09-22T18:30:00Z')).min], [2, 870]);
    eq('ET in winter (EST): 15:00Z is 10:00', ssEt(new Date('2026-01-15T15:00:00Z')).min, 600);
    eq('ET the Monday after spring-forward: 14:00Z is 10:00', ssEt(new Date('2026-03-09T14:00:00Z')).min, 600);
    eq('ET the Monday after fall-back: 15:00Z is 10:00', ssEt(new Date('2026-11-02T15:00:00Z')).min, 600);
    eq('ET date rolls at Eastern midnight, not UTC', ssEt(new Date('2026-09-23T02:00:00Z')).iso, '2026-09-22');
    eq('row of an entry time', ['09:30', '10:00', '10:15', '10:29', '15:59', '16:00'].map(ssSlotOf), [0, 1, 1, 1, 12, 12]);

    // ── the schedule no longer decides ───────────────────────────────────
    const fly = byName('Friday iron fly');
    eq('a Friday strategy on a Tuesday reads its market (TRADE)', stOf(fly), 'TRADE');
    eq('legacy row arrives with its day, no times, the calendar', [fly.strategy.weekdays, fly.strategy.entry_times, fly.strategy.display], [[5], [], 'calendar']);
    eq('checklist has no day row', [...secOf(3).querySelectorAll('.ss-why-row .ss-why-name')].map(txt), ['VIX']);
    eq('summary states the schedule without applying it', txt(secOf(3).querySelector('.ss-why-sum')),
       'Conditions (AND) → TRADE · Tue is not a scheduled day (the schedule does not change the signal)');
    eq('reason line is empty for a market verdict', txt(secOf(3).querySelector('.ss-reason')), '');

    // ── the calendar ─────────────────────────────────────────────────────
    eq('calendar: five weekdays', $$('.ss-cal-day').map(txt), ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']);
    eq('calendar: 30-minute rows 09:30..15:30', $$('.ss-cal-time[data-slot]').map(txt),
       ['09:30', '10:00', '10:30', '11:00', '11:30', '12:00', '12:30', '13:00', '13:30', '14:00', '14:30', '15:00', '15:30']);
    eq('Tue 10:00 → the Tuesday 10:00 row', minis(cell(2, 1)), ['Short-term put spread']);
    eq('Fri 11:30 → the Friday 11:30 row (second scheduled time)', minis(cell(5, 4)), ['Short-term put spread']);
    eq('one strategy, two scheduled times → two cards', $$('.ss-mini').filter(m => txt(m.querySelector('.ss-mini-name')) === 'Short-term put spread').length, 2);
    eq('two strategies in one row, earlier time first', minis(cell(3, 1)), ['Day Trade 1:7 DTE Diagonal', 'Dbl-Cal 7/9 DTE']);
    { const [p, q] = [...cell(3, 1).querySelectorAll('.ss-mini')].map(m => m.getBoundingClientRect());
      eq('... on one line when the cell has room for both', Math.abs(p.top - q.top) < 1, true); }
    const [ma, mb] = [...cell(3, 1).querySelectorAll('.ss-mini')].map(m => m.getBoundingClientRect());
    eq('... side by side, not overlapping', !(ma.right > mb.left && mb.right > ma.left && ma.bottom > mb.top && mb.bottom > ma.top), true);
    eq('... and inside their cell', ma.width > 40 && mb.right <= cell(3, 1).getBoundingClientRect().right + 0.5, true);
    eq('original time in the tooltip', cell(3, 1).querySelectorAll('.ss-mini')[1].title,
       'Dbl-Cal 7/9 DTE — scheduled Wed 10:15 ET — NO TRADE');
    eq('scheduled day with no time → the "no time" row', $$('.ss-cal-untimed .ss-cal-cell').map(c => minis(c)), [[], [], [], [], ['Friday iron fly']]);
    eq('standalone card is not on the calendar', $$('.ss-cal .ss-mini-name').map(txt).includes('Skew allocation'), false);
    eq('side: the standalone card and the dayless calendar strategy', $$('.ss-side .ss-card .ss-cname').map(txt), ['Skew allocation', 'Needs missing data']);
    eq('dayless calendar strategy says why it is beside', !!$$('.ss-side .ss-card')[1].querySelector('.ss-unsched'), true);
    const topNames = new Set([...$$('.ss-cal .ss-mini-name'), ...$$('.ss-side .ss-cname')].map(n => txt(n).replace(/^⚠ /, '')));
    eq('every strategy is somewhere at the top', BOARD.strategies.every(e => topNames.has(e.strategy.name)), true);
    const sideW = $('.ss-side').getBoundingClientRect().width, topW = $('.ss-top').getBoundingClientRect().width;
    eq('side column is a narrow share beside the calendar', sideW / topW > 0.12 && sideW / topW < 0.22, true);
    eq('calendar and side are side by side', Math.abs($('.ss-side').getBoundingClientRect().top - $('.ss-cal').getBoundingClientRect().top) < 2, true);

    // ── one decision, three places ───────────────────────────────────────
    const shown = {};
    for (const m of $$('.ss-mini')) (shown[txt(m.querySelector('.ss-mini-name'))] = shown[txt(m.querySelector('.ss-mini-name'))] || new Set()).add(txt(m.querySelector('.ss-mini-state')));
    for (const c of $$('.ss-side .ss-card')) (shown[txt(c.querySelector('.ss-cname'))] = shown[txt(c.querySelector('.ss-cname'))] || new Set()).add(txt(c.querySelector('.ss-pill')));
    const agree = BOARD.strategies.map(e => {
      const sec = txt(secOf(e.strategy.id).querySelector('.ss-sec-hdr .ss-pill'));
      return [e.strategy.name, [...shown[e.strategy.name]].concat([sec]).every(v => v === stOf(e))];
    });
    eq('calendar, side card and section report the same state', agree.filter(a => !a[1]).map(a => a[0]), []);
    eq('mini card state colour is the state colour', cell(2, 1).querySelector('.ss-mini-state').style.background, rgb('#3498db'));
    eq('NO TRADE mini is pink', cell(3, 1).querySelectorAll('.ss-mini-state')[1].style.background, rgb('#e84393'));
    eq('NO DATA side card', txt($$('.ss-side .ss-card')[1].querySelector('.ss-pill')), 'NO DATA');
    eq('mini cards carry no numbers', $$('.ss-mini').map(m => /\d\.\d|\d{2}:\d{2}/.test(txt(m))).some(Boolean), false);

    // ── today and the time line (Eastern) ────────────────────────────────
    app.clock = new Date('2026-09-22T18:30:00Z'); await wait(50);   // the board's Tuesday, 14:30 ET
    eq('today column (Tue) is highlighted', $$('.ss-cal-day').map(d => d.classList.contains('today')), [false, true, false, false, false]);
    eq('every Tuesday cell carries it', $$('.ss-cal-cell.today').length, 14);
    app.clock = new Date('2026-09-22T18:45:00Z');       // 14:45 EDT: half-way down the 14:30 row
    await wait(50); app.placeNowLine();
    const line = $('.ss-now'), row = $('.ss-cal-time[data-slot="10"]');
    const lr = line.getBoundingClientRect(), rr = row.getBoundingClientRect();
    eq('time line shown in session', visible(line), true);
    eq('time line sits at the actual time, not the row edge', Math.abs(lr.top - (rr.top + rr.height / 2)) <= 1.5, true);
    const d1 = $$('.ss-cal-day')[0].getBoundingClientRect(), d5 = $$('.ss-cal-day')[4].getBoundingClientRect();
    eq('time line spans all five weekdays', Math.abs(lr.left - d1.left) <= 1 && Math.abs(lr.right - d5.right) <= 1, true);
    eq('corner shows the ET time', txt($('.ss-cal-corner')), '14:45');
    app.clock = new Date('2026-09-22T21:00:00Z'); await wait(50); app.placeNowLine();
    eq('after the close: no line, says so', [visible(line), txt($('.ss-cal-note'))], [false, 'After the close (16:00 ET)']);
    app.clock = new Date('2026-09-22T13:00:00Z'); await wait(50); app.placeNowLine();
    eq('before the open: no line', [visible(line), txt($('.ss-cal-note'))], [false, 'Before the open']);
    app.clock = new Date('2026-09-22T18:45:00Z'); await wait(50); app.placeNowLine();

    // ── layout: started in two columns, section 3 collapsed (stored) ────
    const secs = $$('.ss-sec');
    eq('stored layout applied on load (two columns)', $('.ss-secs').classList.contains('two'), true);
    const r0 = secs[0].getBoundingClientRect(), r1 = secs[1].getBoundingClientRect();
    eq('two sections share a row', Math.abs(r0.top - r1.top) < 2 && r1.left > r0.right, true);
    eq('no horizontal overflow', document.documentElement.scrollWidth <= document.documentElement.clientWidth, true);
    const ch = id => Chart.getChart(document.getElementById(id));
    const vr = ch('ss-c-2-r');
    eq('charts fit their half-width section', !!vr && vr.width > 150 && vr.width <= secs[1].getBoundingClientRect().width, true);
    eq('stored collapse applied on load (section 3 closed)', [visible(secOf(3).querySelector('.ss-sec-body')), txt(secOf(3).querySelector('.ss-fold'))], [false, '▸']);
    $$('.ss-cal-untimed .ss-mini')[0].click();
    await wait(300);
    eq('clicking its card opens the collapsed section', visible(secOf(3).querySelector('.ss-sec-body')), true);
    eq('... and remembers it open', JSON.parse(localStorage.getItem('ss.collapsed')), {});
    const fly3 = ch('ss-c-3-r');
    eq('a chart drawn while collapsed has a size once opened', !!fly3 && fly3.width > 100, true);
    secOf(2).querySelector('.ss-fold').click(); await wait(100);
    eq('collapse a section: body hidden, remembered', [visible(secOf(2).querySelector('.ss-sec-body')), JSON.parse(localStorage.getItem('ss.collapsed'))], [false, { 2: true }]);
    secOf(2).querySelector('.ss-fold').click(); await wait(100);
    $$('.ss-bar .ss-seg button').find(b => txt(b) === 'Full width').click(); await wait(200);
    eq('Full width: one per row, remembered', [$('.ss-secs').classList.contains('two'), secs[1].getBoundingClientRect().top > secs[0].getBoundingClientRect().bottom - 1, JSON.parse(localStorage.getItem('ss.layout'))], [false, true, 'full']);

    // ── images ───────────────────────────────────────────────────────────
    const img1 = '/api/strategy-signal/strategies/1/image?size=thumb&v=1790812800000';
    eq('side card shows the thumbnail', $$('.ss-side .ss-card')[0].querySelector('img').getAttribute('src'), img1);
    eq('mini card shows its thumbnail', cell(3, 1).querySelectorAll('.ss-mini')[1].querySelector('img').getAttribute('src'),
       '/api/strategy-signal/strategies/5/image?size=thumb&v=1790812900000');
    eq('no image → no img', cell(3, 1).querySelectorAll('.ss-mini')[0].querySelector('img'), null);
    eq('section heading links the full image', secOf(1).querySelector('.ss-sec-hdr a').getAttribute('href'),
       '/api/strategy-signal/strategies/1/image?size=full&v=1790812800000');
    const sImg = $$('.ss-side .ss-card')[0].querySelector('img');
    eq('thumbnail keeps its aspect ratio', getComputedStyle(sImg).objectFit, 'contain');

    // ── order ────────────────────────────────────────────────────────────
    secOf(2).querySelector('button[title="Move down"]').click(); await wait(100);
    const mv = sent.find(s => s.u.endsWith('/strategies/2/move'));
    eq('▼ sends a move the server stores', mv && JSON.parse(mv.body), { direction: 1 });

    // ── a past date: no live line, the evaluated day marked ──────────────
    const dateIn = $('.ss-bar input[type=date]');
    setVal(dateIn, BOARDS.past.date);
    for (let i = 0; i < 40 && !$('.ss-past'); i++) await wait(50);
    await wait(200);
    eq('past date shows the banner', txt($('.ss-past')).startsWith('Evaluated at the close of Friday 2026-09-18'), true);
    eq('past date: no time line, no today column', [visible(line), $$('.ss-cal .today').length], [false, 0]);
    eq('past date: the evaluated weekday is marked', $$('.ss-cal-day').map(d => d.classList.contains('evald')), [false, false, false, false, true]);
    eq('past date: the calendar says it is not live', txt($('.ss-cal-note')), 'Schedule only — the states are at the close of Fri 2026-09-18, not live');
    const vrPast = Chart.getChart(document.getElementById('ss-c-2-r'));
    eq('charts end on the chosen day', vrPast && BOARDS.past.strategies[1].charts.r.t.slice(-1)[0], '2026-09-18 16:00');
    eq('the past board decides without the weekday too', BOARDS.past.strategies.map(e => e.decision.reason).includes('weekday'), false);
    $$('.ss-bar .ss-btn').find(b => txt(b) === 'Live').click();
    for (let i = 0; i < 40 && $('.ss-past'); i++) await wait(50);
    await wait(100); app.placeNowLine();
    eq('Live returns: line and today back', [!!$('.ss-past'), visible(line), $$('.ss-cal-day.today').length], [false, true, 1]);

    // ── Add → schedule, display → Save ───────────────────────────────────
    $$('.ss-bar .ss-btn').find(b => txt(b) === '+ Add Strategy').click();
    await wait(300);
    eq('Add opens the panel', !!$('.ss-cfg'), true);
    const dayRow = n => $$('.ss-cfg .ss-dayrow').find(r => txt(r.querySelector('.ss-chk')) === n);
    eq('a new strategy has no scheduled day', $$('.ss-cfg .ss-dayrow input[type=checkbox]').map(i => i.checked), [false, false, false, false, false]);
    setVal($('.ss-cfg input[placeholder^="e.g. Short"]'), 'Test strat');
    dayRow('Tue').querySelector('input[type=checkbox]').click(); await wait(50);
    dayRow('Fri').querySelector('.ss-btn').click(); await wait(50);
    eq('+ time ticks its day', dayRow('Fri').querySelector('input[type=checkbox]').checked, true);
    setVal(dayRow('Fri').querySelector('input[type=time]'), '11:30');
    dayRow('Fri').querySelector('.ss-btn').click(); await wait(50);
    setVal(dayRow('Fri').querySelectorAll('input[type=time]')[1], '14:05');
    eq('Tue shows "no time set"', visible(dayRow('Tue').querySelector('.ss-hint')), true);
    $$('.ss-cfg .ss-seg button').find(b => txt(b) === 'Standalone card').click(); await wait(50);
    setVal($('.ss-metric input.src'), 'index:vix');
    await wait(50);
    const sigBox = $$('.ss-metric .ss-chk').find(l => txt(l) === 'Use for signal').querySelector('input');
    sigBox.click(); await wait(100);
    setVal($('.ss-metric input.num'), '22.5');
    $$('.ss-cfg button').find(b => txt(b) === 'Full … None').click(); await wait(100);
    eq('preset states resize thresholds', $$('.ss-metric input.num').length, 3);
    $$('.ss-cfg button').find(b => txt(b) === 'Trade / No trade').click(); await wait(100);
    eq('and back', $$('.ss-metric input.num').length, 1);
    $$('.ss-cfg button').find(b => txt(b) === '+ Requirement').click(); await wait(50);
    setVal($('.ss-cfg input[placeholder^="e.g. Premium"]'), 'Check theta');
    sent.length = 0;
    $$('.ss-cfg button').find(b => txt(b) === 'Save').click();
    await wait(400);
    const post = sent.find(s => s.m === 'POST');
    const body = post ? JSON.parse(post.body) : {};
    eq('Save POSTs', !!post, true);
    eq('payload name / days / times / display', [body.name, body.weekdays, body.entry_times, body.display],
       ['Test strat', [2, 5], [{ day: 5, time: '11:30' }, { day: 5, time: '14:05' }], 'card']);
    eq('payload metric', body.metrics && [body.metrics[0].a, body.metrics[0].signal, body.metrics[0].cmp, body.metrics[0].thresholds],
       ['index:vix', true, '<', [22.5]]);
    eq('payload manual', body.manual, ['Check theta']);
    eq('no image chosen → no image request', sent.filter(s => s.u.endsWith('/image')).length, 0);
    window.__payload = body;
    eq('panel collapses after save', !!$('.ss-cfg'), false);

    // ── Edit: filled, then an image staged and sent by Save ─────────────
    secOf(2).querySelector('.ss-sec-hdr .ss-btn').click();
    await wait(300);
    eq('Edit opens the panel filled', [!!$('.ss-cfg'), $('.ss-cfg input[placeholder^="e.g. Short"]').value,
       $$('.ss-metric').length, $$('.ss-metric input.num').map(i => i.value)],
       [true, 'Short-term put spread', 2, ['1.14', '0.8375']]);
    eq('Edit shows the scheduled times', $$('.ss-cfg input[type=time]').map(i => i.value), ['10:00', '11:30']);
    eq('Edit shows the display', txt($$('.ss-cfg .ss-seg button.on').find(x => /calendar|card/i.test(txt(x)))), 'Weekly calendar');
    eq('no image yet', [!!$('.ss-cfg .ss-imgnone'), txt($$('.ss-cfg button').find(b => /Upload|Replace/.test(txt(b))))], [true, 'Upload…']);
    const fileIn = $('.ss-cfg input[type=file]');
    const dt = new DataTransfer();
    dt.items.add(new File([new Uint8Array([137, 80, 78, 71])], 'risk.png', { type: 'image/png' }));
    fileIn.files = dt.files;
    fileIn.dispatchEvent(new Event('change', { bubbles: true }));
    await wait(100);
    eq('picked image previews before Save', [($('.ss-cfg .ss-imgprev') || {}).src && $('.ss-cfg .ss-imgprev').src.startsWith('blob:'), txt($('.ss-cfg .ss-f .ss-hint:not([x-show])') && [...$$('.ss-cfg .ss-hint')].find(h => /when you Save/.test(txt(h))))],
       [true, 'risk.png — uploaded when you Save']);
    eq('nothing is sent before Save', sent.filter(s => s.u.endsWith('/image')).length, 0);
    sent.length = 0;
    failImage = true;
    $$('.ss-cfg button').find(b => txt(b) === 'Save').click();
    await wait(400);
    eq('config first, then the image', sent.map(s => s.m + ' ' + s.u.replace('/api/strategy-signal', '')), ['PUT /strategies/2', 'PUT /strategies/2/image']);
    eq('failed upload keeps the panel and says the strategy was saved', [!!$('.ss-cfg'), txt($('.ss-cfg .ss-err'))],
       [true, 'The strategy was saved, but the image was not: not an image']);
    sent.length = 0;
    $$('.ss-cfg button').find(b => txt(b) === 'Save').click();
    await wait(400);
    const up = sent.find(s => s.u.endsWith('/strategies/2/image'));
    eq('Save again uploads the staged file as multipart', !!up && up.m === 'PUT' && up.body instanceof FormData && up.body.get('file').name, 'risk.png');
    eq('panel closes once both landed', !!$('.ss-cfg'), false);

    // Replace / remove on a strategy that has one.
    secOf(1).querySelector('.ss-sec-hdr .ss-btn').click();
    await wait(300);
    eq('existing image shown, offered for replacement', [$('.ss-cfg .ss-imgprev').getAttribute('src'), !!$$('.ss-cfg button').find(b => txt(b) === 'Replace…')], [img1, true]);
    eq('edit shows the saved transform and lookback', $$('.ss-metric').map(el => [...el.querySelectorAll('select')].map(x => x.value)),
       [['', '', 'daily', '1y'], ['', 'pctile_252', 'daily', '6m', '<']]);
    $$('.ss-cfg button').find(b => txt(b) === 'Remove').click(); await wait(50);
    eq('Remove is staged', [!!$('.ss-cfg .ss-imgnone'), !![...$$('.ss-cfg .ss-hint')].find(h => txt(h) === 'removed when you Save')], [true, true]);
    sent.length = 0;
    $$('.ss-cfg button').find(b => txt(b) === 'Save').click();
    await wait(400);
    eq('Save sends the config, then DELETE image', sent.map(s => s.m + ' ' + s.u.replace('/api/strategy-signal', '')), ['PUT /strategies/1', 'DELETE /strategies/1/image']);
    secOf(1).querySelector('.ss-sec-hdr .ss-btn').click(); await wait(300);
    $$('.ss-cfg button').find(b => txt(b) === 'Cancel').click(); await wait(200);
    eq('Cancel collapses, nothing sent', [!!$('.ss-cfg'), sent.length], [false, 2]);

    const pre = document.createElement('pre');
    pre.id = 'report';
    pre.textContent = out.join('\n') + '\nerrs|' + (errs.length ? errs.join(' ;; ') : 'clean') +
                      '\npayload|' + JSON.stringify(window.__payload || {});
    document.body.appendChild(pre);
  }
  // A throw inside the driver is reported, not swallowed into "never reported".
  const guarded = () => run().catch(err => {
    out.push(`FAIL|the driver threw|${String(err && err.stack || err).replace(/\n/g, ' ')}|nothing`);
    const pre = document.createElement('pre');
    pre.id = 'report';
    pre.textContent = out.join('\n') + '\nerrs|' + (errs.length ? errs.join(' ;; ') : 'clean');
    document.body.appendChild(pre);
  });
  document.addEventListener('alpine:initialized', () => setTimeout(guarded, 50));
})();
</script>
"""


def build_page(board: str) -> str:
    import jinja2
    from app.assets import asset

    class _URL:
        hostname = "localhost"; scheme = "http"; path = "/"
        def __str__(self): return "http://localhost/"

    class _Req:
        url = _URL(); headers = {}; query_params = {}; scope = {"type": "http"}

    env = jinja2.Environment(loader=jinja2.FileSystemLoader(str(ROOT / "templates")),
                             autoescape=True, keep_trailing_newline=True)
    env.globals["asset"] = asset
    env.globals["live_port"] = 8001
    html = env.get_template("strategy_signal.html").render(request=_Req())
    # crossorigin, so an error thrown inside a CDN script reports its message
    # instead of the opaque "Script error." a file:// page otherwise gets.
    html = html.replace('<script defer src="https://', '<script defer crossorigin="anonymous" src="https://')
    html = html.replace('<script src="https://', '<script crossorigin="anonymous" src="https://')
    html = re.sub(r'<link rel="stylesheet" href=[^>]*css/([a-z_]+\.css)[^>]*>',
                  lambda m: "<style>\n" + (ROOT / "static/css" / m.group(1)).read_text(encoding="utf-8") + "\n</style>",
                  html)
    stub = f"<script>const BOARDS = {board}; const BOARD = BOARDS.live; const SOURCES = {json.dumps(SOURCES)};</script>" + DRIVER
    html = re.sub(r"<script src=[^>]*?/static/js/([a-z_]+\.js)[^>]*></script>",
                  lambda m: (stub if m.group(1) == "strategy_signal.js" else "") + "<script>\n" +
                  (ROOT / "static/js" / m.group(1)).read_text(encoding="utf-8") + "\n</script>", html)
    return html


def main() -> int:
    browser = next((p for p in EDGE_CANDIDATES if Path(p).exists()), None)
    if browser is None:
        print("  SKIP  no Edge/Chromium on this host — the page was NOT driven")
        return EXIT_SKIPPED
    board = board_json()
    tmp = ROOT / "scripts" / "_strategy_signal_ui.html"
    tmp.write_text(build_page(board), encoding="utf-8")
    try:
        if "--shot" in sys.argv:
            shot = sys.argv[sys.argv.index("--shot") + 1]
            html = tmp.read_text(encoding="utf-8").replace("setTimeout(guarded, 50)", "0")
            shot_page = ROOT / "scripts" / "_strategy_signal_shot.html"
            shot_page.write_text(html, encoding="utf-8")
            subprocess.run([browser, "--headless=new", "--disable-gpu", "--window-size=1500,2600",
                            f"--screenshot={shot}", "--virtual-time-budget=20000", shot_page.as_uri()],
                           capture_output=True, timeout=180)
            shot_page.unlink(missing_ok=True)
            print(f"  wrote {shot}")
        p = subprocess.run([browser, "--headless=new", "--disable-gpu", "--window-size=1500,1000",
                            "--dump-dom", "--virtual-time-budget=30000", tmp.as_uri()],
                           capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=180)
    finally:
        tmp.unlink(missing_ok=True)

    m = re.search(r'<pre id="report">(.*?)</pre>', p.stdout, re.S)
    if not m:
        print("  the page never reported —", (p.stdout or "")[-400:].replace("\n", " "))
        return 1
    import html as _html
    from app.strategy_signal import store
    fails = n = 0
    for line in _html.unescape(m.group(1)).strip().splitlines():
        parts = line.split("|", 3)
        if parts[0] == "errs":
            if parts[1] != "clean":
                print(f"  FAIL  the console is not clean: {parts[1]}")
                fails += 1
            continue
        if parts[0] == "payload":
            # What the page sends must be what the store accepts.
            try:
                store.clean_config(json.loads(parts[1]), {"index:vix", "index:vix9d"})
                print("  ok    the saved payload passes store.clean_config")
            except Exception as exc:                          # noqa: BLE001
                print(f"  FAIL  the saved payload is refused by the store: {exc}")
                fails += 1
            continue
        n += 1
        ok = parts[0] == "ok"
        print(("  ok    " if ok else "  FAIL  ") + parts[1] + ("" if ok else f": {parts[2]} (want {parts[3]})"))
        fails += not ok
    print(f"strategy signal UI: {n} assertions, failures: {fails}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
