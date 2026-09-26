"""The Strategy Signal page, DRIVEN BY CLICKING, in headless Edge.

The board it draws is produced by THE REAL ROUTER over fabricated bars --
store and fetch are replaced, the decision, series and payload code are not
-- so the page is read against what the server really sends. The page's
fetch is stubbed in the browser; its own JS and template are the shipped
files, inlined so file:// works (the check_portfolio_ui pattern).

What it asks, as a person would: one card per strategy, each showing its
state in the right colour; the manual requirement visible on a TRADE and
absent on a NO TRADE; a Friday-only strategy reading NO TRADE on a Tuesday;
a strategy whose metric has no data reading NO DATA; a chart per charted
metric, with threshold lines only where the metric decides; notes as a
footnote; Add opens the panel, Save sends a payload the store accepts and
collapses it; Edit opens it filled. And the console stays clean -- an Alpine
expression error leaves a control inert without stopping the page.

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

STRATEGIES = [
    {"id": 1, "position": 1, "name": "Skew allocation", "weekdays": [1, 2, 3, 4, 5],
     "states": ["FULL", "REDUCED", "MINIMAL", "NONE"], "logic": "and", "manual": [],
     "notes": "Allocation scales down as 90d skew gets rich against its own year.",
     "metrics": [
         {"id": "raw", "label": "", "a": "surface:skew_90d_25p_25c", "op": None, "b": None, "transform": None,
          "chart": True, "resolution": "daily", "lookback": "1y", "signal": False, "cmp": None, "thresholds": []},
         {"id": "pct", "label": "90d skew pctl", "a": "surface:skew_90d_25p_25c", "op": None, "b": None,
          "transform": "pctile_252", "chart": True, "resolution": "daily", "lookback": "6m", "signal": True,
          "cmp": "<", "thresholds": [30, 50, 101]}]},
    {"id": 2, "position": 2, "name": "Short-term put spread", "weekdays": [1, 2, 3, 4, 5],
     "states": ["TRADE", "NO TRADE"], "logic": "and", "manual": ["Premium must be >= $2.50"],
     "notes": "VIX/VIX9D below 1.14 with a flat-to-upward front term structure.",
     "metrics": [
         {"id": "r", "label": "", "a": "index:vix", "op": "/", "b": "index:vix9d", "transform": None,
          "chart": True, "resolution": "intraday", "lookback": "10d", "signal": True, "cmp": "<",
          "thresholds": [1.14]},
         {"id": "t", "label": "", "a": "surface:term_ratio_7d_30d", "op": None, "b": None, "transform": None,
          "chart": True, "resolution": "intraday", "lookback": "10d", "signal": True, "cmp": ">",
          "thresholds": [0.8375]}]},
    {"id": 3, "position": 3, "name": "Friday iron fly", "weekdays": [5],
     "states": ["TRADE", "NO TRADE"], "logic": "and", "manual": ["Check entry delta"], "notes": "",
     "metrics": [
         {"id": "r", "label": "", "a": "index:vix", "op": None, "b": None, "transform": None,
          "chart": True, "resolution": "daily", "lookback": "3m", "signal": True, "cmp": "<",
          "thresholds": [40]}]},
    {"id": 4, "position": 4, "name": "Needs missing data", "weekdays": [1, 2, 3, 4, 5],
     "states": ["TRADE", "NO TRADE"], "logic": "and", "manual": [], "notes": "",
     "metrics": [
         {"id": "g", "label": "", "a": "surface:iv_30d_atm", "op": None, "b": None, "transform": None,
          "chart": False, "resolution": "daily", "lookback": "1y", "signal": True, "cmp": ">",
          "thresholds": [0.1]}]},
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
        return json.loads(json.dumps(STRATEGIES))

    lib.fetch_bars, store.list_strategies, lib.now_et = fetch, listing, (lambda: NOW)
    return json.dumps(asyncio.run(router.board(pool=None)))


DRIVER = r"""
<script>
(function () {
  const errs = [];
  window.addEventListener('error', e => errs.push('onerror: ' + e.message));
  const ce = console.error.bind(console);
  console.error = (...a) => { errs.push('console.error: ' + a.map(String).join(' ')); ce(...a); };
  const sent = [];
  window.fetch = async (url, opts) => {
    const u = String(url), m = (opts && opts.method) || 'GET';
    const ok = body => ({ ok: true, status: 200, json: async () => body, text: async () => JSON.stringify(body) });
    if (u.endsWith('/board')) return ok(BOARD);
    if (u.endsWith('/sources')) return ok(SOURCES);
    if (u.includes('/strategies')) { sent.push({ u, m, body: opts && opts.body }); return ok({ strategy: {} }); }
    return { ok: false, status: 404, json: async () => ({}), text: async () => 'no stub for ' + u };
  };
  const out = [];
  const eq = (name, got, want) => out.push(`${JSON.stringify(got) === JSON.stringify(want) ? 'ok' : 'FAIL'}|${name}|${JSON.stringify(got)}|${JSON.stringify(want)}`);
  const wait = ms => new Promise(r => setTimeout(r, ms));
  const $ = s => document.querySelector(s), $$ = s => [...document.querySelectorAll(s)];
  const txt = el => (el ? el.textContent.trim().replace(/\s+/g, ' ') : null);
  const rgb = hex => `rgb(${parseInt(hex.slice(1, 3), 16)}, ${parseInt(hex.slice(3, 5), 16)}, ${parseInt(hex.slice(5, 7), 16)})`;
  const setVal = (el, v) => { el.value = v; el.dispatchEvent(new Event('input', { bubbles: true })); el.dispatchEvent(new Event('change', { bubbles: true })); };

  async function run() {
    for (let i = 0; i < 100 && !$$('.ss-card').length; i++) await wait(50);
    await wait(300);
    const cards = $$('.ss-card');
    eq('one card per strategy', cards.length, 4);
    const pill = i => txt(cards[i].querySelector('.ss-pill'));
    eq('skew allocation state', pill(0), BOARD.strategies[0].strategy.states[BOARD.strategies[0].decision.state]);
    eq('binary strategy TRADE', pill(1), 'TRADE');
    eq('TRADE is theme blue', cards[1].querySelector('.ss-pill').style.background, rgb('#3498db'));
    eq('manual requirement shown on TRADE', txt(cards[1].querySelector('.ss-check')), 'CHECK Premium must be >= $2.50');
    eq('Friday-only on a Tuesday', pill(2), 'NO TRADE');
    eq('NO TRADE is theme pink', cards[2].querySelector('.ss-pill').style.background, rgb('#e84393'));
    eq('no CHECK on NO TRADE', cards[2].querySelector('.ss-check'), null);
    eq('missing data reads NO DATA', pill(3), 'NO DATA');
    eq('card id line names the decision metrics', txt(cards[1].querySelector('.ss-cid')), 'VIX / VIX9D · term_ratio_7d_30d');
    eq('card id line names the entry day', txt(cards[2].querySelector('.ss-cid')), 'Fri · VIX');
    eq('cards carry no numbers', cards.map(c => /\d\.\d/.test(txt(c).replace('$2.50', ''))), [false, false, false, false]);

    const secs = $$('.ss-sec');
    eq('one section per strategy', secs.length, 4);
    eq('charts per section', secs.map(s => s.querySelectorAll('canvas').length), [2, 2, 1, 0]);
    const ch = id => Chart.getChart(document.getElementById(id));
    const raw = ch('ss-c-1-raw'), pct = ch('ss-c-1-pct'), vr = ch('ss-c-2-r');
    eq('charts exist', [!!raw, !!pct, !!vr], [true, true, true]);
    if (raw && pct && vr) {
      eq('visualisation-only metric draws no threshold', raw.data.datasets.length, 1);
      eq('allocation metric draws one threshold per level', pct.data.datasets.length, 4);
      eq('threshold coloured by the state it leads to', pct.data.datasets[1].borderColor, '#3498db');
      eq('intraday 10 sessions of points', vr.data.datasets[0].data.length, BOARD.strategies[1].charts.r.t.length);
      eq('intraday ticks are session starts', vr.scales.x.ticks.length <= 12 && vr.scales.x.ticks.length >= 9, true);
    }
    eq('Friday reason', txt(secs[2].querySelector('.ss-reason')), 'Not an entry day — Fri only');
    eq('notes are a footnote, not a box', [txt(secs[0].querySelector('.ss-notes')), secs[0].querySelectorAll('textarea').length],
       ['Notes. Allocation scales down as 90d skew gets rich against its own year.', 0]);

    // Add → fill → Save
    $$('.ss-bar .ss-btn').find(b => txt(b) === '+ Add Strategy').click();
    await wait(300);
    eq('Add opens the panel', !!$('.ss-cfg'), true);
    setVal($('.ss-cfg input[placeholder^="e.g. Short"]'), 'Test strat');
    const fri = $$('.ss-cfg .ss-chk').filter(l => txt(l) === 'Fri')[0].querySelector('input');
    const mon = $$('.ss-cfg .ss-chk').filter(l => txt(l) === 'Mon')[0].querySelector('input');
    mon.click(); await wait(50);
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
    $$('.ss-cfg button').find(b => txt(b) === 'Save').click();
    await wait(400);
    const post = sent.find(s => s.m === 'POST');
    const body = post ? JSON.parse(post.body) : {};
    eq('Save POSTs', !!post, true);
    eq('payload name / days / states', [body.name, body.weekdays, body.states], ['Test strat', [2, 3, 4, 5], ['TRADE', 'NO TRADE']]);
    eq('payload metric', body.metrics && [body.metrics[0].a, body.metrics[0].signal, body.metrics[0].cmp, body.metrics[0].thresholds],
       ['index:vix', true, '<', [22.5]]);
    eq('payload manual', body.manual, ['Check theta']);
    window.__payload = body;
    eq('panel collapses after save', !!$('.ss-cfg'), false);

    // Edit
    secs[1].querySelector('.ss-sec-hdr .ss-btn').click();
    await wait(300);
    eq('Edit opens the panel filled', [!!$('.ss-cfg'), $('.ss-cfg input[placeholder^="e.g. Short"]').value,
       $$('.ss-metric').length, $$('.ss-metric input.num').map(i => i.value)],
       [true, 'Short-term put spread', 2, ['1.14', '0.8375']]);
    $$('.ss-cfg button').find(b => txt(b) === 'Cancel').click();
    await wait(200);
    eq('Cancel collapses', !!$('.ss-cfg'), false);

    // Selects whose options arrive after the model must still show it.
    secs[0].querySelector('.ss-sec-hdr .ss-btn').click();
    await wait(300);
    eq('edit shows the saved transform and lookback', $$('.ss-metric').map(el => [...el.querySelectorAll('select')].map(x => x.value)),
       [['', '', 'daily', '1y'], ['', 'pctile_252', 'daily', '6m', '<']]);
    $$('.ss-cfg button').find(b => txt(b) === 'Cancel').click();
    await wait(200);

    const pre = document.createElement('pre');
    pre.id = 'report';
    pre.textContent = out.join('\n') + '\nerrs|' + (errs.length ? errs.join(' ;; ') : 'clean') +
                      '\npayload|' + JSON.stringify(window.__payload || {});
    document.body.appendChild(pre);
  }
  document.addEventListener('alpine:initialized', () => setTimeout(run, 50));
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
    stub = f"<script>const BOARD = {board}; const SOURCES = {json.dumps(SOURCES)};</script>" + DRIVER
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
            html = tmp.read_text(encoding="utf-8").replace("setTimeout(run, 50)", "0")
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
