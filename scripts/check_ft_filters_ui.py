"""Factor Trades' metric filters, DRIVEN BY CLICKING, in headless Edge.

The payloads the page is served are produced by THE REAL HANDLERS (/run and
/zone) over check_ft_population's recording fake pool, so the page reads the
server's shapes rather than hand-written ones. The page JS and template are
the shipped files; fetch is stubbed in the browser and every request body is
kept, because what the page SENDS is half of what this checks.

What it asks, as a person would:
  + Add filter makes a row that says it is not applied until complete;
  choosing a metric shows its train-window percentiles; Run sends the filter
  and max strike together; the run card names the filter and what it removed
  ("failed" and "no value" apart); a saved, locked card and a run with a
  different filter raise the different-population banner; the min-n slider
  defaults to 50 and the share of cells below it follows the slider; the
  trade CSV carries the filter in its header; portfolio mode keeps the
  filter rows and sends them, and labels the signal list's n as unfiltered.
And the console stays clean.

No Edge: SKIP, not PASS.
"""
from __future__ import annotations

import asyncio
import copy
import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

EXIT_SKIPPED = 3
EDGE = [r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
        "/usr/bin/microsoft-edge", "/usr/bin/chromium", "/usr/bin/google-chrome"]


def _harness():
    spec = importlib.util.spec_from_file_location("ftpop", ROOT / "scripts" / "check_ft_population.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["ftpop"] = mod
    spec.loader.exec_module(mod)
    return mod


def payloads() -> dict:
    """Real /run and /zone responses for the bodies the driver will send."""
    h = _harness()
    import app.routers.factor_trades as ft
    import app.routers.oi_analysis as oia

    def call(fn, req):
        oia._TT_CUTOFF_CACHED = None
        return json.loads(json.dumps(asyncio.run(fn(req=req, pool=h.FakePool())), default=str))

    # An exit rule is required: the server refuses a run with none.
    rk = ["fixed_stop__5"]
    out = {}
    for v in (0.05, 0.1):
        f = [{"metric": "ret_5d", "op": ">", "value": v}]
        r = call(ft.run, ft.RunReq(primary_metric="a", secondary_metric="b", n_bins=20,
                                   rule_keys=rk, max_strike=1000, filters=f))
        # A grid with cells of different sizes, so min-n has something to split.
        g = [[None] * 20 for _ in range(20)]
        for (iy, ix, n) in ((2, 1, 12), (4, 3, 60), (6, 5, 140), (8, 7, 30)):
            g[iy][ix] = {"n": n, "avg_ret": 0.004 * (ix - 4), "test_n": n // 3, "test_avg": 0.001}
        r["grid"] = g
        out[f"run {v}"] = r
        out[f"zone {v}"] = call(ft.zone, ft.ZoneReq(primary_metric="a", secondary_metric="b",
                                                    n_bins=20, rule_keys=rk, max_strike=1000,
                                                    filters=f, cells=[[1, 2]]))
    out["portfolio"] = call(ft.zone, ft.ZoneReq(signal_ids=[1], n_bins=20, rule_keys=rk, max_strike=1000,
                                                filters=[{"metric": "ret_5d", "op": ">", "value": 0.1}]))
    return out


STUBS = {
    "columns": {"features": ["a", "b", "ret_5d"], "outcomes": ["ret_5d_fwd_oc"],
                "feature_families": [{"family_num": 1, "family_name": "Fam", "metrics": ["a", "b", "ret_5d"]}]},
    "cutoff": {"cutoff_date": "2023-01-02"},
    "signals": {"signals": [{"id": 1, "name": "sig1", "primary_metric": "a", "secondary_metric": "b",
                             "outcome": "ret_5d_fwd_oc", "n_bins": 20, "n_cells": 1, "agg_n": 500,
                             "agg_avg_ret": 0.002, "per_cell_stats": [], "status": "Test",
                             "color_slot": 0, "corner": None, "selection_mode": "train_test",
                             "selection_cutoff": "2023-01-02", "eligible": True, "reason": None}],
                "cutoff_date": "2023-01-02", "max_selectable": 31},
    "stats": {"metric": "ret_5d", "window": "train", "n": 1000, "n_value": 980, "no_value_share": 0.02,
              "pcts": [{"p": 5, "v": -0.081}, {"p": 25, "v": -0.02}, {"p": 50, "v": 0.004},
                       {"p": 75, "v": 0.027}, {"p": 95, "v": 0.09}]},
}

DRIVER = r"""
<script>
(function () {
  const errs = [];
  window.addEventListener('error', e => errs.push('onerror: ' + e.message + ' @ ' + String((e.error && e.error.stack) || '').split('\n').slice(1, 3).join(' ')));
  const ce = console.error.bind(console);
  console.error = (...a) => { errs.push('console.error: ' + a.map(String).join(' ')); ce(...a); };
  const sent = [];
  let csv = null;
  const realBlob = window.Blob;
  window.Blob = function (parts, opts) { csv = parts.join(''); return new realBlob(parts, opts); };
  URL.createObjectURL = () => 'blob:x';
  HTMLAnchorElement.prototype.click = function () {};
  window.fetch = async (url, opts) => {
    const u = String(url);
    const body = opts && opts.body ? JSON.parse(opts.body) : null;
    if (body) sent.push({ u, body });
    const ok = b => ({ ok: true, status: 200, json: async () => b, text: async () => JSON.stringify(b) });
    const fv = body && body.filters && body.filters[0] ? body.filters[0].value : null;
    if (u.includes('/factor-analysis/columns')) return ok(STUBS.columns);
    if (u.includes('/tt-cutoff')) return ok(STUBS.cutoff);
    if (u.includes('/factor-trades/rules')) return ok({ groups: [] });
    if (u.includes('/factor-trades/signals')) return ok(STUBS.signals);
    if (u.includes('/filter-stats')) return ok(STUBS.stats);
    if (u.endsWith('/factor-trades/run')) return ok(PAY['run ' + fv]);
    if (u.endsWith('/factor-trades/zone')) return ok(body.signal_ids ? PAY.portfolio : PAY['zone ' + fv]);
    return { ok: false, status: 404, json: async () => ({}), text: async () => 'no stub ' + u };
  };
  const out = [];
  const eq = (name, got, want) => out.push(`${JSON.stringify(got) === JSON.stringify(want) ? 'ok' : 'FAIL'}|${name}|${JSON.stringify(got)}|${JSON.stringify(want)}`);
  const wait = ms => new Promise(r => setTimeout(r, ms));
  const $ = s => document.querySelector(s), $$ = s => [...document.querySelectorAll(s)];
  const txt = el => (el ? el.textContent.trim().replace(/\s+/g, ' ') : null);
  const vis = el => !!(el && el.offsetParent !== null);
  const setVal = (el, v, ev) => { el.value = v; for (const e of (ev || ['input', 'change'])) el.dispatchEvent(new Event(e, { bubbles: true })); };
  const btn = t => $$('button').find(b => txt(b) === t);
  const lastBody = path => { const r = sent.filter(x => x.u.endsWith(path)); return r.length ? r[r.length - 1].body : null; };

  async function run() {
    for (let i = 0; i < 60 && !btn('+ Add filter'); i++) await wait(50);
    await wait(200);
    // ── add, pick, type ────────────────────────────────────────────────
    btn('+ Add filter').click(); await wait(100);
    eq('a new row says it is not applied', txt($('.ft-filter-off')), 'not applied — pick a metric');
    const sel = $('.ft-filter select');
    setVal(sel, 'ret_5d', ['change']); await wait(200);
    eq('choosing a metric shows its train percentiles', txt($('.ft-filter-st')),
       'train: p5 -0.081 p25 -0.02 med 0.004 p75 0.027 p95 0.09 · 2.0% no value');
    eq('the value box suggests the median', $('.ft-filter-line input').placeholder, 'median 0.004');
    eq('still not applied without a value', txt($('.ft-filter-off')), 'not applied — enter a value');
    setVal($('.ft-filter-line input'), '0.05'); await wait(100);
    // Content, not visibility: x-show hides on the same expression, but a
    // hide after a show waits on Alpine's transition frame, which headless
    // virtual time does not run.
    eq('a complete row is applied (no warning left)', txt($('.ft-filter-off')), '');

    // ── run ────────────────────────────────────────────────────────────
    btn('Run').click(); await wait(400);
    eq('the run did not error', Alpine.$data(document.body).error, '');
    const rb = lastBody('/run');
    eq('Run sends the filter and max strike together', rb && [rb.filters, rb.max_strike],
       [[{ metric: 'ret_5d', op: '>', value: 0.05 }], 1000]);
    const card = () => txt($$('.ft-run').slice(-1)[0]);
    eq('the card names the filter', card().includes('filter ret_5d > 0.05'), true);
    eq('the card reports kept, failed and no value apart',
       /kept 6 of 10/.test(card()) && /ret_5d: 2 failed · 1 no value/.test(card()), true);

    // ── min n ──────────────────────────────────────────────────────────
    const slider = $('input[type=range][max="200"]');
    eq('min n defaults to 50', slider && slider.value, '50');
    const hmNote = () => txt(slider.parentElement);
    eq('share of cells below the threshold', hmNote().includes('2 of 4 cells (50%) below n 50'), true);
    setVal(slider, '20'); await wait(100);
    eq('the share follows the slider', hmNote().includes('1 of 4 cells (25%) below n 20'), true);

    // ── save, lock, change the filter, run again ───────────────────────
    btn('+ Save to strip').click(); await wait(200);
    // The saved card's lock, found through Alpine's own scope for it.
    const lock = $$('.ft-lock').find(l => { try { return !!Alpine.$data(l).c.saved; } catch (e) { return false; } });
    eq('the saved card has a lock', !!lock, true);
    if (lock) lock.click();
    await wait(200);
    setVal($('.ft-filter-line input'), '0.1'); await wait(100);
    btn('Run').click(); await wait(500);
    const banner = $$('div').find(d => d.getAttribute('x-show') === 'lockedPopMismatch');
    eq('a locked card with another filter raises the population banner, naming both',
       banner ? txt(banner) : null,
       'Locked and current runs cover different populations — locked: max strike $1000 · ret_5d > 0.05 — current: max strike $1000 · ret_5d > 0.1. The Change row compares two different trade sets.');

    // ── select a cell, export ──────────────────────────────────────────
    const cells = $$('.hm-cell-selectable');
    if (cells.length) cells.find(c => (c.getAttribute('title') || '').length) ?.click();
    await wait(600);
    const zb = lastBody('/zone');
    eq('the zone request carries the population', zb && [zb.filters, zb.max_strike],
       [[{ metric: 'ret_5d', op: '>', value: 0.1 }], 1000]);
    const exp = btn('Export CSV');
    if (exp) exp.click();
    await wait(100);
    const head = (csv || '').split('\n').filter(l => l.startsWith('#'));
    eq('the trade CSV header carries the filter', head.includes('# filter,ret_5d,>,0.1'), true);
    eq('and max strike', head.includes('# max_strike,1000'), true);

    // ── portfolio ──────────────────────────────────────────────────────
    $$('span').find(s => txt(s) === 'Portfolio').click(); await wait(300);
    eq('portfolio keeps the filter rows', [vis(btn('+ Add filter')), $$('.ft-filter').filter(vis).length], [true, 1]);
    eq('the signal list labels its n unfiltered', document.body.innerHTML.includes('n(FA, unfiltered)'), true);
    const sigBox = $$('input[type=checkbox]').find(c => c.closest('tr') && /sig1/.test(txt(c.closest('tr'))));
    if (sigBox) { sigBox.click(); await wait(100); }
    const comp = Alpine.$data(document.body);
    if (!comp.selectedSignalIds.length) comp.selectedSignalIds = [1];
    btn('Run').click(); await wait(500);
    const pb = lastBody('/zone');
    eq('a portfolio run sends the filter', pb && [pb.signal_ids, pb.filters],
       [[1], [{ metric: 'ret_5d', op: '>', value: 0.1 }]]);
    eq('the portfolio card names the filter', card().includes('filter ret_5d > 0.1'), true);

    const pre = document.createElement('pre');
    pre.id = 'report';
    pre.textContent = out.join('\n') + '\nerrs|' + (errs.length ? errs.join(' ;; ') : 'clean');
    document.body.appendChild(pre);
  }
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


def build_page(pay: dict) -> str:
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
    html = env.get_template("factor_trades.html").render(request=_Req())
    html = html.replace('<script defer src="https://', '<script defer crossorigin="anonymous" src="https://')
    html = html.replace('<script src="https://', '<script crossorigin="anonymous" src="https://')
    html = re.sub(r'<link rel="stylesheet" href=[^>]*css/([a-z_]+\.css)[^>]*>',
                  lambda m: "<style>\n" + (ROOT / "static/css" / m.group(1)).read_text(encoding="utf-8") + "\n</style>",
                  html)
    stub = (f"<script>const PAY = {json.dumps(pay)}; const STUBS = {json.dumps(STUBS)};</script>" + DRIVER)
    first = [True]

    def inline(m):
        pre = stub if first[0] else ""
        first[0] = False
        return pre + "<script>\n" + (ROOT / "static/js" / m.group(1)).read_text(encoding="utf-8") + "\n</script>"
    return re.sub(r"<script src=[^>]*?/static/js/([a-z_]+\.js)[^>]*></script>", inline, html)


def main() -> int:
    browser = next((p for p in EDGE if Path(p).exists()), None)
    if browser is None:
        print("  SKIP  no Edge/Chromium on this host — the page was NOT driven")
        return EXIT_SKIPPED
    tmp = ROOT / "scripts" / "_ft_filters_ui.html"
    tmp.write_text(build_page(payloads()), encoding="utf-8")
    try:
        p = subprocess.run([browser, "--headless=new", "--disable-gpu", "--window-size=1600,1200",
                            "--dump-dom", "--virtual-time-budget=40000", tmp.as_uri()],
                           capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=240)
    finally:
        tmp.unlink(missing_ok=True)
    m = re.search(r'<pre id="report">(.*?)</pre>', p.stdout, re.S)
    if not m:
        print("  the page never reported —", (p.stdout or "")[-400:].replace("\n", " "))
        return 1
    import html as _html
    fails = n = 0
    for line in _html.unescape(m.group(1)).strip().splitlines():
        parts = line.split("|", 3)
        if parts[0] == "errs":
            if parts[1] != "clean":
                print(f"  FAIL  the console is not clean: {parts[1][:1500]}")
                fails += 1
            continue
        n += 1
        ok = parts[0] == "ok"
        print(("  ok    " if ok else "  FAIL  ") + parts[1] + ("" if ok else f": {parts[2][:300]} (want {parts[3][:200]})"))
        fails += not ok
    print(f"factor-trades filters UI: {n} assertions, failures: {fails}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
