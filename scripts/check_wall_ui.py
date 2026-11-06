"""The Equities Wall page at 300 symbols, 100 a page -- clicked, in headless Edge.

The real template and bundle, with the one WebSocket replaced by a fake that
speaks the server's frame format (hello, then a tick a second carrying every
symbol). Nothing reaches the live service or the market-data account.

What it holds, and what it MEASURES (printed, so the numbers in the commit are
the page's own):

  * 100 panes in the document, list order; the pager CLICKED to page two puts
    entries 101-200 there and draws every one of them AT ONCE, from tape that
    was already in the browser -- no message to the server on a flip.
  * every one of the 300 symbols keeps arriving while it is off-page.
  * one draw pass over the visible hundred, and one tick's processing of all
    300, each well inside the one-second budget.
  * pane geometry at the window size(s) given: columns, rows, canvas size,
    and whether a hundred fit without scrolling, per pane-width setting.
  * no console error and no exception (an Alpine expression error leaves a
    control inert without stopping the page).

No browser (the VPS) -> SKIP.
"""
from __future__ import annotations

import json
import re
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import edge_cdp                                           # noqa: E402
import check_oo_notes_ui as N                             # noqa: E402

EXIT_SKIPPED = 3
FAILS: list[str] = []
N_ASSERT = [0]


def check(cond, msg):
    N_ASSERT[0] += 1
    if not cond:
        FAILS.append(msg)


DRIVER = r"""
<script>
(() => {
  const H = document.documentElement;
  const mark = (s) => { H.setAttribute('data-progress', s);
    H.setAttribute('data-trail', (H.getAttribute('data-trail') || '') + ' > ' + s); };
  const errors = [];
  window.addEventListener('error', (e) => errors.push('onerror: ' + e.message));
  const ce = console.error.bind(console);
  console.error = (...a) => { errors.push('console.error: ' + a.join(' ')); ce(...a); };
  try { localStorage.clear(); } catch (e) {}

  const N = 300, PER = 100;
  const syms = Array.from({length: N}, (_, i) => 'S' + String(i).padStart(3, '0'));
  const px = {}; syms.forEach((s, i) => { px[s] = 20 + i; });
  const sends = [];
  const received = {};           // symbol -> ticks that carried it
  let sock = null;

  // Roughly the wall's real shape: most names a few prints a second, a few
  // busy ones, and the 200 ms sampled band (up to five quotes a second).
  function cell(s, i, nowMs, full) {
    const busy = i < 10 ? 40 : (i < 60 ? 6 : 2);
    const t = [], q = [];
    const n = full ? busy * 120 : busy;
    const span = full ? 120000 : 1000;
    for (let k = 0; k < n; k++) {
      px[s] += (Math.random() - 0.5) * 0.02;
      t.push([nowMs - span + (k + 0.5) * span / n, +px[s].toFixed(2), [1, 10, 100][k % 3]]);
    }
    const nq = full ? 600 : 5, half = 0.03 + (i % 7) * 0.02;
    for (let k = 0; k < nq; k++) {
      q.push([nowMs - span + (k + 0.5) * span / nq, +(px[s] - half).toFixed(2), +(px[s] + half).toFixed(2)]);
    }
    const c = {t, q, sp: half * 200, tp: half * 200, mid: px[s]};
    if (full) { c.full = true; c.q0 = [nowMs - span - 50, q[0][1], q[0][2]]; }
    return c;
  }
  function frame(full) {
    const nowMs = Date.now(), out = {};
    syms.forEach((s, i) => { out[s] = cell(s, i, nowMs, full); received[s] = (received[s] || 0) + 1; });
    return {ev: 'tick', at: nowMs / 1000, window_s: 120, syms: out};
  }
  class FakeWS {
    constructor(url) { this.url = url; this.readyState = 0; sock = this;
      setTimeout(() => { this.readyState = 1; this.onopen && this.onopen();
        this.emit({ev: 'hello', data: {}, entries: syms.map(s => ({symbol: s, scale: null})),
                   settings: {window_s: 120, spread_share: 0.6, min_spread_cents: 0},
                   caps: {symbols: N, per_page: PER, retain_s: 180, share_min: 0.05, share_max: 0.95, tick_s: 1}});
        this.emit(frame(true));
        this.timer = setInterval(() => this.emit(frame(false)), 1000);
      }, 30); }
    emit(m) { const t0 = performance.now(); this.onmessage && this.onmessage({data: JSON.stringify(m)});
              if (m.ev === 'tick') window.__tickMs = (window.__tickMs || []).concat(performance.now() - t0); }
    send(x) { sends.push(x); }
    close() {}
  }
  window.WebSocket = FakeWS;

  const sleep = (ms) => new Promise(r => setTimeout(r, ms));
  const frameDone = () => new Promise(r => requestAnimationFrame(() => setTimeout(r, 0)));
  async function waitFor(fn, what, ms = 20000) {
    const t0 = Date.now();
    while (Date.now() - t0 < ms) { try { if (fn()) return; } catch (e) {} await sleep(50); }
    throw new Error('timed out waiting for ' + what);
  }
  function canvasInk(cv) {
    if (!cv.width || !cv.height) return 0;
    const d = cv.getContext('2d').getImageData(0, 0, cv.width, cv.height).data;
    let n = 0; for (let i = 3; i < d.length; i += 16) if (d[i]) n++;
    return n;
  }
  function paneSyms() { return [...document.querySelectorAll('.wl-pane')].map(e => e.dataset.sym); }

  async function run() {
    const out = {errors, layout: []};
    mark('alpine');
    await waitFor(() => window.Alpine && document.querySelector('[x-data]')._x_dataStack, 'Alpine');
    const C = () => Alpine.$data(document.querySelector('[x-data]'));
    mark('first frame');
    await waitFor(() => document.querySelectorAll('.wl-pane').length > 0 && C().lastTick, 'panes');
    await sleep(1500);
    out.panesPage1 = paneSyms();
    out.pagerButtons = [...document.querySelectorAll('.wl-pager .wl-btn')].map(b => b.textContent.trim());

    // ── layout per pane width ──────────────────────────────────────────
    mark('layout');
    const scroll = document.querySelector('.wl-scroll');
    for (const w of [120, 140, 160, 180, 200, 220, 240]) {
      C().paneWidth = w; await Alpine.nextTick(); await frameDone();
      const panes = [...document.querySelectorAll('.wl-pane')];
      const topOf = (p) => Math.round(p.getBoundingClientRect().top + scroll.scrollTop);
      const tops = new Set(panes.map(topOf));
      const first = Math.min(...tops);
      const cv = panes[0].querySelector('canvas').getBoundingClientRect();
      out.layout.push({paneWidth: w, cols: panes.filter(p => topOf(p) === first).length,
                       rows: tops.size, canvasW: Math.round(cv.width), canvasH: Math.round(cv.height),
                       gridViewH: scroll.clientHeight, gridH: scroll.scrollHeight,
                       fits: scroll.scrollHeight <= scroll.clientHeight + 1,
                       viewport: [innerWidth, innerHeight]});
    }
    // Every pane on screen (160 fits a hundred at this size), so the pass
    // below draws all of them; off-screen panes are skipped by design.
    C().paneWidth = 160; await Alpine.nextTick(); await frameDone(); await sleep(300);
    // ── the draw pass over the visible hundred, all dirty ──────────────
    mark('draw cost');
    const draws = [];
    for (let k = 0; k < 6; k++) {
      C().dirtyAll();
      const t0 = performance.now(); C().drawAll(); draws.push(performance.now() - t0);
      out.drawnAll = C().drawn;
      await sleep(200);
    }
    out.drawMs = draws.sort((a, b) => a - b);


    // ── flip to page two BY CLICKING, and time it ──────────────────────
    mark('flip');
    const sendsBefore = sends.length;
    const btn = [...document.querySelectorAll('.wl-pager .wl-btn')].find(b => b.textContent.trim() === '101–200');
    const comp = C();
    const t0 = performance.now();
    btn.click();
    await waitFor(() => paneSyms()[0] === 'S100' && comp.drawn >= 90, 'page two drawn', 3000)
      .catch(e => { out.flipTimeout = String(e); });
    out.flipMs = performance.now() - t0;
    await frameDone();
    out.panesPage2 = paneSyms();
    out.drawnAfterFlip = comp.drawn;
    const cvs = [...document.querySelectorAll('.wl-pane canvas')];
    out.inked = cvs.filter(c => canvasInk(c) > 20).length;
    out.sendsOnFlip = sends.slice(sendsBefore);
    out.page = comp.page;
    out.summary = comp.connSummary();

    // every symbol kept arriving, on- and off-page
    const before = Object.assign({}, received);
    await sleep(2200);
    out.stillArriving = syms.filter(s => (received[s] || 0) > (before[s] || 0)).length;
    out.tickMs = (window.__tickMs || []).slice(2).sort((a, b) => a - b);

    // last page and back; a shortened list clamps the page
    mark('page three');
    [...document.querySelectorAll('.wl-pager .wl-btn')].find(b => b.textContent.trim() === '201–300').click();
    await waitFor(() => paneSyms()[0] === 'S200', 'page three', 3000).catch(e => { out.p3Timeout = String(e); });
    out.panesPage3 = paneSyms();
    out.nextDisabled = [...document.querySelectorAll('.wl-pager .wl-btn')].pop().disabled;
    sock.emit({ev: 'watchlist', entries: syms.slice(0, 150).map(s => ({symbol: s, scale: null}))});
    await Alpine.nextTick(); await frameDone();
    out.pageAfterShrink = comp.page;
    out.panesAfterShrink = paneSyms();
    out.storedPage = localStorage.getItem('equitiesWall.page');
    mark('done');
    return out;
  }
  window.addEventListener('DOMContentLoaded', () => {
    run().then(o => { const r = document.createElement('pre'); r.id = 'report';
                      r.textContent = JSON.stringify(o); document.body.appendChild(r); })
         .catch(e => { const r = document.createElement('pre'); r.id = 'report';
                       r.textContent = JSON.stringify({fatal: String(e), errors}); document.body.appendChild(r); });
  });
})();
</script>
"""


def build_page() -> str:
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
    html = env.get_template("equities_wall.html").render(request=_Req())
    html = re.sub(
        r'<link rel="stylesheet" href=[^>]*css/([a-z_]+\.css)[^>]*>',
        lambda m: "<style>\n" + (ROOT / "static/css" / m.group(1)).read_text(encoding="utf-8") + "\n</style>",
        html)
    html = re.sub(
        r"<script src=[^>]*?/static/js/([a-z_]+\.js)[^>]*></script>",
        lambda m: "<script>\n" + (ROOT / "static/js" / m.group(1)).read_text(encoding="utf-8") + "\n</script>",
        html)
    return html.replace("</head>", DRIVER + "</head>", 1)


def drive(browser: str, window: str) -> dict:
    tmp = Path(tempfile.mkdtemp(prefix="wallui-")) / "wall.html"
    tmp.write_text(build_page(), encoding="utf-8")
    return json.loads(edge_cdp.run_page(browser, tmp, timeout_s=120, window=window))


def main() -> int:
    browser = N.find_browser()
    if browser is None:
        print("  SKIP  no Edge/Chromium on this host -- the page was NOT driven")
        return EXIT_SKIPPED
    # 2560x1440 with the window full screen, and maximised under a tab strip,
    # an address bar and the taskbar (~140 px less).
    for window in ("2560,1440", "2560,1300"):
        o = drive(browser, window)
        if o.get("fatal"):
            FAILS.append(f"[{window}] the page did not run: {o['fatal']}; errors {o.get('errors')}")
            continue
        tag = f"[{window}]"
        check(not o["errors"], f"{tag} console errors: {o['errors'][:3]}")
        check(o["panesPage1"] == [f"S{i:03d}" for i in range(100)],
              f"{tag} page one is not the first hundred in list order "
              f"({len(o['panesPage1'])} panes, starting {o['panesPage1'][:2]})")
        check(o["pagerButtons"] == ["‹", "1–100", "101–200", "201–300", "›"],
              f"{tag} pager reads {o['pagerButtons']}")
        check(o["drawnAll"] == 100, f"{tag} one draw pass drew {o['drawnAll']} of 100 dirty panes")
        check(not o.get("flipTimeout"), f"{tag} {o.get('flipTimeout')}")
        check(o["panesPage2"] == [f"S{i:03d}" for i in range(100, 200)],
              f"{tag} after clicking 101-200 the panes are {o['panesPage2'][:2]}...")
        check(o["flipMs"] < 400,
              f"{tag} a flip took {o['flipMs']:.0f} ms to draw page two; it must not "
              f"wait for the next one-second tick")
        check(o["inked"] >= 95,
              f"{tag} only {o['inked']} of page two's canvases have anything drawn "
              f"right after the flip -- their tape was not warm")
        check(not o["sendsOnFlip"],
              f"{tag} the flip sent {o['sendsOnFlip']} to the server; pages must be a "
              f"view, with every symbol still streaming")
        check(o["stillArriving"] == 300,
              f"{tag} only {o['stillArriving']} of 300 symbols kept arriving while two "
              f"pages were off screen")
        check(o["panesPage3"][:1] == ["S200"] and o["nextDisabled"] is True,
              f"{tag} page three / its next button: {o['panesPage3'][:1]}, {o['nextDisabled']}")
        check(o["pageAfterShrink"] == 1 and o["panesAfterShrink"][:1] == ["S100"]
              and len(o["panesAfterShrink"]) == 50,
              f"{tag} a list shortened to 150 under page three left the page at "
              f"{o['pageAfterShrink']} showing {len(o['panesAfterShrink'])} panes")
        check(o["storedPage"] is not None, f"{tag} the page is not remembered on this screen")
        draw_med = o["drawMs"][len(o["drawMs"]) // 2]
        tick = o["tickMs"]
        tick_med = tick[len(tick) // 2] if tick else float("nan")
        check(draw_med < 250, f"{tag} drawing 100 panes takes {draw_med:.0f} ms of every second")
        check(tick and tick_med < 250, f"{tag} one tick of 300 symbols takes {tick_med:.0f} ms")

        print(f"  {tag} viewport {o['layout'][0]['viewport']}: draw 100 panes median "
              f"{draw_med:.1f} ms (worst {o['drawMs'][-1]:.1f}); tick of 300 symbols median "
              f"{tick_med:.1f} ms; flip to page two drawn in {o['flipMs']:.0f} ms, "
              f"{o['inked']}/100 canvases inked")
        for L in o["layout"]:
            print(f"      pane width {L['paneWidth']:>3}: {L['cols']:>2} x {L['rows']:>2}, canvas "
                  f"{L['canvasW']} x {L['canvasH']} px, "
                  + ("all 100 on screen" if L["fits"] else
                     f"scrolls ({L['gridH']} px of panes in {L['gridViewH']} px)"))

    print(f"\nwall ui: {N_ASSERT[0]} assertions, failures: {len(FAILS)}")
    for f in FAILS:
        print(f"  FAIL {f}")
    return 1 if FAILS else 0


if __name__ == "__main__":
    raise SystemExit(main())
