"""Editing a SAVED strategy's notes on the OO/Mesosim page, BY CLICKING.

The store half (`store.set_notes`) is held by check_oo_market_sql against a
real Postgres. This holds the page half, which no source-reading check can:
that the button shows on a loaded saved strategy and not on an upload, that
Edit opens a box holding the current notes, that Save sends exactly the box
to `PUT /strategies/{id}/notes` and shows what the SERVER returned in both
places the page shows notes (the loaded strategy and the saved-list line),
that Cancel leaves the saved notes as they were, and that a refused save
says so and keeps the draft.

Headless Edge against the real template and bundle, fetch stubbed. Captures
window.onerror / console.error, because an Alpine expression error leaves a
control inert without stopping the page. No browser (the VPS) -> SKIP.
"""
from __future__ import annotations

import html as _html
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

SAVED = [{"id": 7, "name": "alpha", "notes": "first thoughts", "trade_count": 3,
          "capital_per_position": None, "source": "mesosim", "filename": "alpha.json",
          "date_min": "2024-01-02", "date_max": "2024-03-01",
          "updated_at": "2026-10-01T00:00:00+00:00"}]

# A raw string: the JS below is passed through untouched, and the one
# substitution is a plain replace of __SAVED__.
DRIVER = r"""
<script>
// HEADLESS VIRTUAL TIME DOES NOT RUN ANIMATION FRAMES, and Alpine SHOWS an
// x-show element from requestAnimationFrame (it hides synchronously). So
// x-text updated here while every x-show stayed hidden, and the gate failed
// or passed by luck. Frames become timers, which virtual time does advance.
// The page's own code is untouched; this is the harness, not a fix.
window.requestAnimationFrame = cb => setTimeout(() => cb(performance.now()), 0);
window.cancelAnimationFrame = id => clearTimeout(id);
window.__errs = [];
window.onerror = (m) => window.__errs.push('error: ' + m);
window.addEventListener('unhandledrejection', e => window.__errs.push('promise: ' + e.reason));
(function () {
  const ce = console.error;
  console.error = (...a) => { window.__errs.push('console.error: ' + a.map(String).join(' ')); ce(...a); };
  const cw = console.warn;
  console.warn = (...a) => { window.__errs.push('console.warn: ' + a.map(String).join(' ')); cw(...a); };
})();
const SAVED = __SAVED__;
window.__puts = [];
window.__refuse = false;
window.fetch = async (u, init) => {
  const url = String(u);
  const reply = (status, body) => ({ ok: status < 400, status, json: async () => body });
  const m = url.match(/\/strategies\/(\d+)\/notes$/);
  if (m && init && init.method === 'PUT') {
    const body = JSON.parse(init.body);
    window.__puts.push({ id: +m[1], notes: body.notes });
    if (window.__refuse) return reply(400, { detail: 'Notes are longer than 4000 characters.' });
    // The server trims; the page must show what it RETURNED, not the box.
    const sv = { ...SAVED[0], notes: String(body.notes).trim() };
    return reply(200, { strategy: sv });
  }
  if (/\/strategies$/.test(url)) return reply(200, { strategies: SAVED });
  return reply(200, {});
};

window.addEventListener('load', () => setTimeout(async () => {
  const out = [];
  const ok = (k, got, want) => out.push(
    (String(got) === String(want) ? 'ok|' : 'FAIL|') + k + '|' + got + '|' + want);
  const wait = ms => new Promise(r => setTimeout(r, ms));
  // UNTIL, not a fixed wait: under headless virtual time a render can land
  // a frame after a timer, so a fixed wait passes or fails by luck.
  const until = async (cond, ms = 3000) => {
    for (let t = 0; t < ms; t += 25) { if (cond()) return true; await wait(25); }
    return cond();
  };
  const report = () => {
    const pre = document.createElement('pre');
    pre.id = 'report';
    pre.textContent = out.concat(['errs|' + (window.__errs.slice(0, 6).join(' ~ ') || 'clean')])
      .join(String.fromCharCode(10));
    document.body.appendChild(pre);
  };
  try {
    // WAIT FOR THE REAL COMPONENT. Before Alpine has initialised <body>,
    // $data returns a detached object: writes to it render nothing, and a
    // gate driving it fails only on the runs where the page was slow.
    let c = null;
    for (let i = 0; i < 100 && !(c && typeof c.commitNotes === 'function'); i++) {
      await wait(50);
      c = document.body._x_dataStack ? Alpine.$data(document.body) : null;
    }
    ok('the page component is up', !!c && typeof c.commitNotes, 'function');
    await wait(300);
    const shown = el => !!el && el.offsetParent !== null;
    const btnText = t => [...document.querySelectorAll('button')]
      .find(b => b.textContent.trim() === t && shown(b));
    const box = () => document.querySelector('.ob-notes-edit textarea');

    // AN UPLOAD has nothing saved to edit.
    c.meta = { filename: 'upload.json', source: 'mesosim_json', notes: {}, n: 3,
               date_min: '2024-01-02', date_max: '2024-03-01', suggested_name: 'upload' };
    c.loaded = true;
    await until(() => /upload/.test(document.querySelector('.topbar-right').textContent));
    ok('no notes button on an upload', !!(btnText('Edit notes') || btnText('Add notes')), false);

    // A LOADED SAVED strategy: what the load path leaves in meta.saved.
    c.selectedSavedId = '7';
    c.meta = { filename: 'alpha.json', source: 'mesosim_json', notes: {}, n: 3,
               date_min: '2024-01-02', date_max: '2024-03-01', saved: { ...SAVED[0] } };
    await wait(150);
    const loadedNotes = () => [...document.querySelectorAll('.ob-loaded .ob-dim')]
      .filter(shown).map(e => e.textContent.trim());
    ok('the saved notes are shown',
       await until(() => loadedNotes().includes('first thoughts')), true);
    await until(() => btnText('Edit notes'));
    const edit = btnText('Edit notes');
    ok('Edit notes is on screen', !!edit, true);

    // CANCEL changes nothing.
    edit.click();
    ok('the box opens', await until(() => shown(box())), true);
    await until(() => document.activeElement === box());
    ok('holding the current notes', box().value, 'first thoughts');
    ok('it has focus', document.activeElement === box(), true);
    box().value = 'thrown away';
    box().dispatchEvent(new Event('input', { bubbles: true }));
    btnText('Cancel').click();
    ok('Cancel closes it', await until(() => !shown(box())), true);
    ok('and sends nothing', window.__puts.length, 0);
    ok('the saved notes are still shown', loadedNotes().includes('first thoughts'), true);

    // SAVE sends the box and shows what the server returned.
    await until(() => btnText('Edit notes'));
    btnText('Edit notes').click();
    await until(() => shown(box()));
    ok('re-opening starts from the saved notes, not the cancelled draft',
       box().value, 'first thoughts');
    box().value = '  second thoughts\nline two  ';
    box().dispatchEvent(new Event('input', { bubbles: true }));
    btnText('Save notes').click();
    await until(() => !shown(box()));
    ok('one PUT', window.__puts.length, 1);
    ok('to this strategy', window.__puts[0] && window.__puts[0].id, 7);
    ok('carrying the box', JSON.stringify(window.__puts[0] && window.__puts[0].notes),
       JSON.stringify('  second thoughts\nline two  '));
    ok('the box closes', !shown(box()), true);
    ok('the loaded strategy shows the server\'s notes',
       loadedNotes().includes('second thoughts\nline two'), true);
    ok('so does the saved list', c.selectedSaved().notes, 'second thoughts\nline two');
    ok('it says so', /Notes saved/.test(document.body.textContent), true);

    // A REFUSED save keeps the box open with the draft, and says why.
    window.__refuse = true;
    await until(() => btnText('Edit notes'));
    btnText('Edit notes').click();
    await until(() => shown(box()));
    box().value = 'too long, pretend';
    box().dispatchEvent(new Event('input', { bubbles: true }));
    btnText('Save notes').click();
    await until(() => /Not saved:/.test(document.body.textContent));
    ok('a refusal keeps the box open', shown(box()), true);
    ok('with the draft', box().value, 'too long, pretend');
    ok('and says why', /Not saved: Notes are longer/.test(document.body.textContent), true);
    ok('the saved notes are unchanged', c.meta.saved.notes, 'second thoughts\nline two');
    // The refusal is expected; only errors other than it count.
    window.__errs = window.__errs.filter(e => !/oo-backtest notes/.test(e));
    window.__refuse = false;
    btnText('Cancel').click();
    await until(() => !shown(box()));

    // NO NOTES: the button says Add.
    c.meta = { ...c.meta, saved: { ...c.meta.saved, notes: '' } };
    ok('with no notes the button reads Add notes', await until(() => !!btnText('Add notes')), true);
  } catch (e) {
    out.push('FAIL|the driver threw|' + e + '|nothing');
  }
  report();
}, 400));
</script>
"""


def find_browser() -> str | None:
    for p in EDGE_CANDIDATES:
        if Path(p).exists():
            return p
    return None


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
    html = env.get_template("oo_backtest.html").render(request=_Req())
    html = re.sub(
        r'<link rel="stylesheet" href=[^>]*css/([a-z_]+\.css)[^>]*>',
        lambda m: "<style>\n" + (ROOT / "static/css" / m.group(1)).read_text(encoding="utf-8") + "\n</style>",
        html)
    html = re.sub(
        r"<script src=[^>]*?/static/js/([a-z_]+\.js)[^>]*></script>",
        lambda m: "<script>\n" + (ROOT / "static/js" / m.group(1)).read_text(encoding="utf-8") + "\n</script>",
        html)
    return html.replace("</head>", DRIVER.replace("__SAVED__", json.dumps(SAVED)) + "</head>", 1)


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
    tmp = ROOT / "scripts" / "_oo_notes_ui.html"
    tmp.write_text(page, encoding="utf-8")
    try:
        p = subprocess.run(
            [browser, "--headless=new", "--disable-gpu", "--window-size=1600,1000",
             "--dump-dom", "--virtual-time-budget=30000", tmp.as_uri()],
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=180)
    finally:
        tmp.unlink(missing_ok=True)

    m = re.search(r'<pre id="report">(.*?)</pre>', p.stdout, re.S)
    if not m:
        print("  the page never reported — it did not get that far")
        return 1
    fails = n = 0
    for line in _html.unescape(m.group(1)).strip().splitlines():
        parts = line.split("|")
        if parts[0] == "errs":
            if parts[1] != "clean":
                print(f"  FAIL  the console is not clean: {parts[1]}")
                fails += 1
            continue
        if len(parts) < 4:
            print(f"  {line.strip()}")
            continue
        n += 1
        if parts[0] != "ok":
            print(f"  FAIL  {parts[1]}: {parts[2]} (want {parts[3]})")
            fails += 1
        else:
            print(f"  ok    {parts[1]}")
    print(f"OO notes UI: {n} clicked assertions, failures: {fails}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
