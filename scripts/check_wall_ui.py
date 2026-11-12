"""The Equities Wall rotation list at 300 symbols -- clicked, in headless Edge.

The real template and bundle, with the one WebSocket replaced by a fake that
speaks the server's frame format (hello, then ticks carrying the windows --
[p10, median, p90] -- that changed), and `fetch` replaced by one that echoes
the save. The fake ticks every 250 ms, faster than the server's 2 s, so the
run is short; the page takes its cadence from the frames.
Nothing reaches the live service or the market-data account.

What it holds, and what it MEASURES (printed):

  * 300 rows, alphabetical (the hello's list is shuffled), flowing DOWN each
    column and then on to the next -- read from the rows' geometry -- ~250 px
    wide, 16 px tall, all on one screen without scrolling.
  * colour from p90 against the two thresholds, read from computed styles;
    the bar is ANCHORED AT ZERO and ends at p90, dim (34%, 22% below the
    floor) up to p10 and full colour from p10, with the median tick over
    it -- measured from the rendered geometry and the computed gradient; a
    name gapping 20c -> 65c (median under the floor, p90 over it) is lit,
    not dimmed; a name steady at $1 draws LONGER than it, and mostly dim; a
    flat 55c name keeps a bright sliver; both sliders moved by input events
    re-colour and re-scale.
  * a click marks a row green and a second click clears it; a marked row is
    never dimmed or hidden.
  * "hide below floor" removes the dim rows and nothing else; unticked, every
    row is back in place.
  * the sort button: widest first, then alphabetical again.
  * the time one tick takes to apply, at 300 symbols.
  * no console error and no exception.

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

  const N = 300;
  const syms = Array.from({length: N}, (_, i) => 'S' + String(i).padStart(3, '0'));
  // Spreads 5..200c, spread over the list so every column has every tier.
  // sp is each symbol's p90; its window is [p90/3, p90/2, p90], except the
  // two named cases below.
  const sp = {}; syms.forEach((s, i) => { sp[s] = 5 + ((i * 37) % 196); });
  sp.S000 = null;                                   // never quoted
  sp.S001 = 65; sp.S002 = 55; sp.S003 = 100;
  const special = { S001: [20, 22, 65],             // tight, gapping wide
                    S002: [55, 55, 55],             // parked flat
                    S003: [96, 100, 100] };         // steady at a dollar
  const win = (s, v) => (v == null ? null : special[s] && v === sp[s] ? special[s].slice()
                          : [Math.round(v / 3), Math.round(v / 2), v]);
  const all = () => { const o = {}; syms.forEach(s => { o[s] = win(s, sp[s]); }); return o; };
  const shuffled = syms.slice().sort((a, b) => ((a.charCodeAt(3) * 7 + a.charCodeAt(2)) % 11) - ((b.charCodeAt(3) * 7 + b.charCodeAt(2)) % 11));
  const sends = [], posts = [];
  let sock = null, paused = false;

  window.fetch = async (url, opt) => {
    const body = JSON.parse(opt.body);
    posts.push(body);
    return { json: async () => ({ entries: body.entries, settings: body.settings, refused: [] }) };
  };

  class FakeWS {
    constructor(url) { this.url = url; this.readyState = 0; sock = this;
      setTimeout(() => { this.readyState = 1; this.onopen && this.onopen();
        this.emit({ev: 'hello', data: {}, entries: shuffled.map(s => ({symbol: s, scale: null})),
                   settings: {min_spread_cents: 59, wide_cents: 120},
                   caps: {symbols: N, tick_s: 2, window_s: 30, pct_lo: 10, pct_hi: 90}});
        this.emit({ev: 'tick', at: Date.now() / 1000, syms: all()});
        // Deltas: forty names a tick change by a cent and change back, so
        // the picture the assertions read is the same between ticks.
        let flip = 0;
        this.timer = setInterval(() => {
          if (paused) return;
          flip ^= 1; const d = {};
          for (let k = 0; k < 40; k++) { const s = syms[4 + ((k * 7 + flip) % 296)]; d[s] = win(s, sp[s] == null ? null : sp[s] + flip); }
          if (!flip) Object.keys(d).forEach(s => { d[s] = win(s, sp[s]); });
          this.emit({ev: 'tick', at: Date.now() / 1000, syms: d});
        }, 250);
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
  const rows = () => [...document.querySelectorAll('.wl-row')];
  const row = (s) => document.querySelector('.wl-row[data-sym="' + s + '"]');
  const visibleSyms = () => rows().filter(r => getComputedStyle(r).display !== 'none').map(r => r.dataset.sym);
  function look(s) {
    const r = row(s), f = r.querySelector('.wl-fill'), t = r.querySelector('.wl-track');
    const m = r.querySelector('.wl-med');
    const cs = getComputedStyle(r);
    const tb = t.getBoundingClientRect(), fb = f.getBoundingClientRect(), mb = m.getBoundingClientRect();
    const at = (x) => Math.round((x - tb.left) / tb.width * 1000) / 10;
    // The two segments are one element's gradient: the dim stop, its
    // alpha, the boundary (a share of the FILL) and the full colour, as the
    // browser computed them.
    const bgi = getComputedStyle(f).backgroundImage;
    const g = bgi.match(/color\(srgb ([\d.]+) ([\d.]+) ([\d.]+) \/ ([\d.]+)\) ([\d.]+)%, (rgb\([^)]*\)) ([\d.]+)%/);
    const to255 = (x) => Math.round(+x * 255);
    const pShare = g ? +g[5] / 100 : NaN;
    return { cls: r.className, opacity: +cs.opacity, bg: cs.backgroundColor,
             bgi,
             fill: g ? g[6] : null,
             dim: g ? 'rgb(' + to255(g[1]) + ', ' + to255(g[2]) + ', ' + to255(g[3]) + ')' : null,
             dimA: g ? +g[4] : null,
             shown: getComputedStyle(f).display !== 'none' && fb.width > 0,
             left: at(fb.left), hi: at(fb.right), mid: at(mb.left),
             split: at(fb.left + fb.width * pShare),
             brightPct: Math.round(fb.width * (1 - pShare) / tb.width * 1000) / 10,
             fillPx: Math.round(fb.width * 10) / 10,
             inTrack: fb.left >= tb.left - 0.5 && fb.right <= tb.right + 0.5,
             trackPx: Math.round(tb.width),
             medPx: Math.round(mb.width * 10) / 10,
             medColor: getComputedStyle(m).backgroundColor,
             title: (r.dispatchEvent(new Event('mouseenter')), r.getAttribute('title')),
             text: r.querySelector('.wl-cents').textContent.trim(),
             tk: getComputedStyle(r.querySelector('.wl-tk')).color };
  }
  function flow() {
    // Down a column, then on to the next: consecutive visible rows either
    // share a left edge and step down, or move right and start higher.
    const rs = rows().filter(r => getComputedStyle(r).display !== 'none').map(r => r.getBoundingClientRect());
    let bad = 0, cols = 1;
    for (let i = 1; i < rs.length; i++) {
      const a = rs[i - 1], b = rs[i];
      if (Math.abs(a.left - b.left) < 1) { if (!(b.top > a.top)) bad++; }
      else if (b.left > a.left && b.top < a.top) cols++;
      else bad++;
    }
    return { bad, cols, w: Math.round(rs[0].width), h: Math.round(rs[0].height), n: rs.length };
  }
  function slide(model, v) {
    const el = [...document.querySelectorAll('input[type=range]')].find(e => e.getAttribute('x-model.number') === 'settings.' + model);
    el.value = String(v); el.dispatchEvent(new Event('input', {bubbles: true}));
  }

  async function run() {
    const out = {errors};
    mark('alpine');
    await waitFor(() => window.Alpine && document.querySelector('[x-data]')._x_dataStack, 'Alpine');
    const C = () => Alpine.$data(document.querySelector('[x-data]'));
    mark('first frame');
    await waitFor(() => rows().length === N && C().lastTick, 'rows');
    await sleep(800);
    paused = true; await sleep(300);
    // The ticks change and change back; settle on the base picture.
    sock.emit({ev: 'tick', at: Date.now() / 1000, syms: all()});
    await Alpine.nextTick(); await frameDone();

    out.order = rows().map(r => r.dataset.sym);
    out.flow = flow();
    const sc = document.querySelector('.wl-scroll');
    out.fits = sc.scrollHeight <= sc.clientHeight + 1;
    out.listH = Math.round(document.querySelector('.wl-list').getBoundingClientRect().height);
    out.viewH = sc.clientHeight;
    out.sortLabel = document.querySelector('.wl-sort').textContent.trim();
    out.hideChecked = document.querySelector('input[type=checkbox]').checked;

    // ── tiers at the default 59 / 120 ──────────────────────────────────
    mark('tiers');
    const pick = (lo, hi) => syms.find(s => sp[s] != null && sp[s] >= lo && sp[s] < hi);
    out.symBelow = pick(10, 40); out.symMid = pick(70, 100); out.symWide = pick(130, 200);
    out.symSixty = syms.find(s => sp[s] === 60) || pick(60, 61);
    out.below = look(out.symBelow); out.mid = look(out.symMid); out.wide = look(out.symWide);
    out.none = look('S000');
    out.gapper = look('S001'); out.flat = look('S002'); out.steady = look('S003');
    out.legend = [...document.querySelectorAll('.wl-note')].map(e => e.textContent.trim());
    out.sixty = look(out.symSixty);
    out.spOf = {below: sp[out.symBelow], mid: sp[out.symMid], wide: sp[out.symWide], sixty: sp[out.symSixty]};

    // ── the sliders ────────────────────────────────────────────────────
    mark('sliders');
    slide('wide_cents', 240); await Alpine.nextTick(); await frameDone();
    out.sixtyAt240 = look(out.symSixty);
    out.wideAt240 = look(out.symWide);
    slide('floor' === 'x' ? '' : 'min_spread_cents', 20); await Alpine.nextTick(); await frameDone();
    out.belowAt20 = look(out.symBelow);
    out.flatAt240 = look('S002');
    await sleep(700);                               // the debounced save
    out.posts = posts.map(p => p.settings);
    slide('min_spread_cents', 59); slide('wide_cents', 120);
    await Alpine.nextTick(); await frameDone();

    // ── marks ──────────────────────────────────────────────────────────
    mark('marks');
    row(out.symBelow).click(); row(out.symMid).click();
    await Alpine.nextTick(); await frameDone();
    out.markedBelow = look(out.symBelow); out.markedMid = look(out.symMid);
    row(out.symMid).click(); await Alpine.nextTick(); await frameDone();
    out.unmarkedMid = look(out.symMid);

    // ── hide below floor (CLICKED) ─────────────────────────────────────
    mark('hide');
    document.querySelector('input[type=checkbox]').click();
    await Alpine.nextTick(); await frameDone();
    const vis = visibleSyms();
    out.hideVisible = vis.length;
    out.hideExpected = syms.filter(s => (sp[s] != null && sp[s] >= 59) || s === out.symBelow).length;
    out.hideKeepsGapper = vis.includes('S001');
    out.hideKeepsMarked = vis.includes(out.symBelow);
    out.hideDropsBelow = !vis.includes(pick(20, 50) === out.symBelow ? syms.find(s => sp[s] != null && sp[s] < 59 && s !== out.symBelow) : pick(20, 50));
    out.hideFlow = flow();
    out.hideAlpha = vis.join() === vis.slice().sort().join();
    document.querySelector('input[type=checkbox]').click();
    await Alpine.nextTick(); await frameDone();
    out.unhidden = visibleSyms().length;

    // ── sort (CLICKED) ─────────────────────────────────────────────────
    mark('sort');
    document.querySelector('.wl-sort').click();
    await Alpine.nextTick(); await frameDone();
    out.wideOrder = rows().map(r => r.dataset.sym);
    out.sortLabel2 = document.querySelector('.wl-sort').textContent.trim();
    out.wideFlow = flow();
    document.querySelector('.wl-sort').click();
    await Alpine.nextTick(); await frameDone();
    out.backToAlpha = rows().map(r => r.dataset.sym).join() === syms.join();

    // ── export (CLICKED) ───────────────────────────────────────────────
    // The download is caught at the blob and the anchor's click. Taken A-Z
    // with hide off, then widest-first with hide ON and an above-floor row
    // marked: symBelow stays marked throughout and must never appear.
    mark('export');
    const got = [];
    let lastBlob = null;
    const realURL = URL.createObjectURL, realClick = HTMLAnchorElement.prototype.click;
    URL.createObjectURL = (b) => { lastBlob = b; return 'blob:gate'; };
    HTMLAnchorElement.prototype.click = function () {
      if (this.download) got.push({name: this.download, blob: lastBlob}); else realClick.call(this);
    };
    const exportBtn = () => document.querySelector('.wl-export');
    out.exportDisabled = exportBtn().disabled;
    exportBtn().click();
    document.querySelector('input[type=checkbox]').click();
    document.querySelector('.wl-sort').click();
    row(out.symMid).click();
    await Alpine.nextTick(); await frameDone();
    out.exportShownWide = rows().map(r => r.dataset.sym);
    exportBtn().click();
    URL.createObjectURL = realURL; HTMLAnchorElement.prototype.click = realClick;
    out.exports = [];
    for (const g of got) {
      const buf = new Uint8Array(await g.blob.arrayBuffer());
      out.exports.push({name: g.name, type: g.blob.type, text: await g.blob.text(),
                        head: [...buf.slice(0, 3)]});
    }
    out.anchorsLeft = document.querySelectorAll('a[download]').length;
    row(out.symMid).click();
    document.querySelector('.wl-sort').click();
    document.querySelector('input[type=checkbox]').click();
    await Alpine.nextTick(); await frameDone();

    // ── live updates: one row's number changes in place ────────────────
    mark('update');
    paused = false;
    sock.emit({ev: 'tick', at: Date.now() / 1000, syms: {S299: [10, 30, 99]}});  // S299: no background tick touches it
    await Alpine.nextTick(); await frameDone();
    out.updated = look('S299').text;
    // WORST CASE: all 300 spreads change in one tick, timed through Alpine's
    // flush of the row bindings (nextTick) -- the work the frame costs.
    paused = true;
    const full = [];
    for (let k = 0; k < 8; k++) {
      const d = {}; syms.forEach((s, i) => { const v = 5 + ((i * 37 + k + 1) % 196); d[s] = [Math.round(v / 3), Math.round(v / 2), v]; });
      const t0 = performance.now();
      sock.emit({ev: 'tick', at: Date.now() / 1000, syms: d});
      await Alpine.nextTick();
      full.push(performance.now() - t0);
      await sleep(250);
    }
    out.fullTickMs = full.sort((a, b) => a - b);
    paused = false;
    await sleep(1500);
    out.tickMs = (window.__tickMs || []).slice(2).sort((a, b) => a - b);
    out.sends = sends;
    out.summary = C().connSummary();
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


BLUE, PINK, GREY = "rgb(52, 152, 219)", "rgb(232, 67, 147)", "rgb(119, 119, 119)"
GREEN_BG, GREEN = "rgba(46, 204, 113, 0.22)", "rgb(46, 204, 113)"


def p90_of(sym: str):
    """The fixture's p90 per symbol (the driver's `sp`)."""
    i = int(sym[1:])
    return {0: None, 1: 65, 2: 55, 3: 100}.get(i, 5 + ((i * 37) % 196))


def main() -> int:
    browser = N.find_browser()
    if browser is None:
        print("  SKIP  no Edge/Chromium on this host -- the page was NOT driven")
        return EXIT_SKIPPED
    syms = [f"S{i:03d}" for i in range(300)]
    # 2560x1440 full screen, maximised under browser chrome and the taskbar,
    # and a laptop.
    for window in ("2560,1440", "2560,1300", "1440,860"):
        o = drive(browser, window)
        tag = f"[{window}]"
        if o.get("fatal"):
            FAILS.append(f"{tag} the page did not run: {o['fatal']}; errors {o.get('errors')}")
            continue
        check(not o["errors"], f"{tag} console errors: {o['errors'][:3]}")
        check(o["order"] == syms, f"{tag} rows are not alphabetical: {o['order'][:4]}")
        f = o["flow"]
        check(f["bad"] == 0 and f["cols"] >= 2,
              f"{tag} rows do not flow down then across: {f}")
        check(f["h"] == 16 and 250 <= f["w"] <= 320,
              f"{tag} a row is {f['w']} x {f['h']} px, want ~250 x 16")
        # 2560 wide, the screen the list is read on, holds all 300; a laptop
        # cannot (5 columns of 60 rows at ~250 x 16), and is only reported.
        if window.startswith("2560"):
            check(o["fits"], f"{tag} 300 rows scroll: {o['listH']} px in {o['viewH']} px")
        check(o["sortLabel"] == "A → Z" and o["hideChecked"] is False,
              f"{tag} opens as {o['sortLabel']!r}, hide {o['hideChecked']}")

        def seg(x, colour, alpha):
            """Both segments in the row's colour, the dim one at `alpha`."""
            return (x["fill"] == colour and x["dim"] == colour
                    and x["dimA"] is not None and abs(x["dimA"] - alpha) < 0.005)

        b, m, w, n = o["below"], o["mid"], o["wide"], o["none"]
        check("below" in b["cls"] and b["opacity"] == 0.3 and seg(b, GREY, 0.22),
              f"{tag} a {o['spOf']['below']}c row under a 59c floor: {b}")
        check("mid" in m["cls"] and m["opacity"] == 1 and seg(m, BLUE, 0.34),
              f"{tag} a {o['spOf']['mid']}c row between floor and wide: {m}")
        check("wide" in w["cls"] and w["opacity"] == 1 and seg(w, PINK, 0.34)
              and abs(w["left"]) <= 0.5 and abs(w["hi"] - 100) <= 1 and w["inTrack"],
              f"{tag} a {o['spOf']['wide']}c row over 120c wide: {w}")
        check("none" in n["cls"] and n["opacity"] == 0.3 and n["text"] == "—"
              and not n["shown"],
              f"{tag} a symbol with no quote: {n}")
        # THE ANCHORED BAR: 0 -> p90 on the wide scale, the dim/bright
        # boundary at p10, the tick at the median, the number p90. 1% of a
        # ~130 px track is ~1.3 px.
        six, v = o["sixty"], o["spOf"]["sixty"]
        lo, md = round(v / 3), round(v / 2)
        check(abs(six["left"]) <= 0.5 and abs(six["hi"] - v / 1.2) <= 1.5
              and abs(six["split"] - lo / 1.2) <= 1.5
              and abs(six["mid"] - md / 1.2) <= 1.5 and six["text"] == str(v),
              f"{tag} a [{lo}, {md}, {v}] window drew from {six['left']}% to "
              f"{six['hi']}%, bright from {six['split']}%, tick {six['mid']}%, "
              f"number {six['text']!r}; want 0 / {v / 1.2:.1f} / {lo / 1.2:.1f} "
              f"/ {md / 1.2:.1f}% and {v}")
        check(six["medPx"] == 2 and six["medColor"] not in (BLUE, PINK, GREY),
              f"{tag} the median tick is {six['medPx']} px in {six['medColor']}")
        g, fl, st = o["gapper"], o["flat"], o["steady"]
        check("mid" in g["cls"] and g["opacity"] == 1 and seg(g, BLUE, 0.34)
              and g["text"] == "65" and abs(g["left"]) <= 0.5
              and abs(g["split"] - 20 / 1.2) <= 1.5
              and abs(g["mid"] - 22 / 1.2) <= 1.5 and abs(g["hi"] - 65 / 1.2) <= 1.5,
              f"{tag} 20c gapping to 65c (median 22, p90 65) under a 59c "
              f"floor must be lit by p90, bright from 20c, the tick low: {g}")
        check("mid" in st["cls"] and st["text"] == "100"
              and st["fillPx"] > 1.4 * g["fillPx"]
              and st["brightPct"] < 0.25 * g["brightPct"],
              f"{tag} a name steady at $1 must draw LONGER than the 20-65c "
              f"jumper and mostly dim: steady {st['fillPx']} px, bright "
              f"{st['brightPct']}%; jumper {g['fillPx']} px, bright "
              f"{g['brightPct']}%")
        check("below" in fl["cls"] and fl["opacity"] == 0.3 and fl["text"] == "55"
              and abs(fl["hi"] - 55 / 1.2) <= 1.5 and seg(fl, GREY, 0.22)
              and 0.5 <= fl["brightPct"] <= 0.7,
              f"{tag} a flat 55c name must run to 55 with a ~0.6% bright "
              f"sliver at its end: {fl}")
        check(g["title"] and "p90 65" in g["title"] and "median 22" in g["title"],
              f"{tag} the row's hover says {g['title']!r}")
        check(any("bar to p90 of the last 30 s, bright from p10" in t
                  and "number p90" in t for t in o["legend"]),
              f"{tag} the legend reads {o['legend']}")
        s240 = o["sixtyAt240"]
        check(abs(s240["hi"] - v / 2.4) <= 1.5 and abs(s240["split"] - lo / 2.4) <= 1.5
              and abs(s240["left"]) <= 0.5 and o["wideAt240"]["fill"] == BLUE
              and abs(o["flatAt240"]["hi"] - 55 / 2.4) <= 1.5,
              f"{tag} moving wide to 240 did not rescale/recolour: "
              f"{s240}, {o['wideAt240']}")
        check(o["belowAt20"]["opacity"] == 1 and "mid" in o["belowAt20"]["cls"],
              f"{tag} a floor of 20 left a {o['spOf']['below']}c row dim: {o['belowAt20']}")
        check(o["posts"] and o["posts"][-1] == {"min_spread_cents": 20, "wide_cents": 240},
              f"{tag} the thresholds were saved as {o['posts'][-1:]}")

        mb, mm, um = o["markedBelow"], o["markedMid"], o["unmarkedMid"]
        check("marked" in mb["cls"] and mb["opacity"] == 1 and mb["bg"] == GREEN_BG
              and mb["tk"] == GREEN and abs(mb["dimA"] - 0.34) < 0.005,
              f"{tag} a marked row under the floor is not green and full strength: {mb}")
        check("marked" in mm["cls"] and mm["bg"] == GREEN_BG,
              f"{tag} a click did not mark the row: {mm}")
        check("marked" not in um["cls"] and um["bg"] != GREEN_BG,
              f"{tag} a second click did not clear the mark: {um}")

        check(o["hideVisible"] == o["hideExpected"] and o["hideKeepsMarked"]
              and o["hideDropsBelow"] and o["hideKeepsGapper"],
              f"{tag} hide below floor shows {o['hideVisible']} (want "
              f"{o['hideExpected']}), marked kept {o['hideKeepsMarked']}, "
              f"below dropped {o['hideDropsBelow']}")
        check(o["hideFlow"]["bad"] == 0 and o["hideAlpha"],
              f"{tag} the hidden list does not flow down in order: {o['hideFlow']}")
        check(o["unhidden"] == 300, f"{tag} unticking left {o['unhidden']} rows")

        wide_vals = [p90_of(s) for s in o["wideOrder"]]
        nums = [v for v in wide_vals if v is not None]
        check(nums == sorted(nums, reverse=True) and wide_vals[-1] is None
              and o["sortLabel2"] == "widest first" and o["wideFlow"]["bad"] == 0,
              f"{tag} widest first is not descending down the columns: "
              f"{o['wideOrder'][:5]} / {o['sortLabel2']}")
        check(o["backToAlpha"], f"{tag} the second click did not restore A-Z")
        # EXPORT: the rows at or above the 59c floor, in the order shown,
        # marks and hide-below-floor irrelevant; one ticker a line, no
        # header, no BOM; named on the Eastern clock.
        ex = o.get("exports") or []
        above = [s for s in syms if p90_of(s) is not None and p90_of(s) >= 59]
        widest = sorted(above, key=lambda s: (-p90_of(s), s))
        shown_wide = [s for s in o.get("exportShownWide", []) if s in set(above)]
        check(len(ex) == 2 and not o["exportDisabled"],
              f"{tag} two clicks gave {len(ex)} downloads (disabled "
              f"{o.get('exportDisabled')})")
        if len(ex) == 2:
            a_rows, w_rows = (e["text"].split("\r\n") for e in ex)
            check(a_rows[-1] == "" and a_rows[:-1] == above,
                  f"{tag} the A-Z export is {a_rows[:4]}... ({len(a_rows) - 1}); "
                  f"want the {len(above)} at or above 59c, alphabetical")
            check(w_rows[-1] == "" and w_rows[:-1] == widest == shown_wide,
                  f"{tag} the widest-first export is {w_rows[:4]}...; want "
                  f"{widest[:4]}... as shown on screen")
            check(o["symBelow"] not in a_rows + w_rows and o["symMid"] in w_rows,
                  f"{tag} a mark changed what was exported")
            for e in ex:
                check(re.fullmatch(r"wall_\d{4}-\d\d-\d\d_\d{4}\.csv", e["name"])
                      and e["head"] != [0xEF, 0xBB, 0xBF] and e["type"] == "text/csv"
                      and "," not in e["text"] and '"' not in e["text"],
                      f"{tag} export file {e['name']!r} {e['type']} head {e['head']}")
        check(o.get("anchorsLeft") == 0, f"{tag} the download link was left in the page")
        check(o["updated"] == "99", f"{tag} a tick did not update the row: {o['updated']}")
        check(not o["sends"], f"{tag} the page sent {o['sends']} on its socket")
        # RENDER BUDGETS, set from what the page must do, not from a
        # quiet-machine measurement. A frame arrives every WALL_TICK_S (2 s);
        # a full repaint of all 300 rows must be a small share of that, so
        # the page keeps up: under 20% of a frame (400 ms) at the MEDIAN.
        # Measured here, alone: ~45 ms; with the whole suite competing for
        # the CPU: 90-110 ms median, single runs past 300 ms. The old limits
        # (100 ms median, 50 ms for the WORST 40-row tick) were the alone
        # figures plus a little, and failed at random under the suite. The
        # worst case is reported, never asserted: one slow sample is the
        # machine, not the page. A real regression -- a per-row cost that
        # grows, a layout thrash -- moves the median by multiples.
        frame_ms = 2000
        tick = o["tickMs"]
        tick_med = tick[len(tick) // 2] if tick else float("nan")
        check(tick and tick_med < 50,
              f"{tag} a 40-row tick takes {tick_med} ms at the median")
        full = o["fullTickMs"]
        check(full and full[len(full) // 2] < 0.2 * frame_ms,
              f"{tag} a tick changing all 300 rows takes {full} ms to render "
              f"(median over 20% of the {frame_ms} ms frame)")
        print(f"  {tag} {f['cols']} columns of ~{(300 + f['cols'] - 1) // f['cols']} rows, "
              f"row {f['w']} x {f['h']} px, list {o['listH']} px in {o['viewH']} px "
              f"({'fits' if o['fits'] else 'scrolls'}); a 40-row tick median "
              f"{tick_med:.2f} ms; all 300 changed, through the render flush: "
              f"median {full[len(full) // 2]:.1f} ms, worst {full[-1]:.1f} ms")

    print(f"\nwall ui: {N_ASSERT[0]} assertions, failures: {len(FAILS)}")
    for f in FAILS:
        print(f"  FAIL {f}")
    return 1 if FAILS else 0


if __name__ == "__main__":
    raise SystemExit(main())
