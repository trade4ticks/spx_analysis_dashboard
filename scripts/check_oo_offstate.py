"""The OO/Mesosim page with NO NEW FEATURE SWITCHED ON is the page as recorded.

A baseline gate for the train/test split (2026-10-25), and for anything else
that promises to be absent until turned on. With Split OFF the page must be
indistinguishable from what shipped before the split existed: same DOM, same
computations, same numbers, same requests. "Same" is not judged by reading
the new code -- it is held against a RECORDING made at 8f996e8, the commit
before any split code, so the comparison is never the new code with itself.

The recording (scripts/fixtures/oo_offstate_baseline.json.gz) holds the
INPUTS too -- the registry, the catalog and every payload the page is fed --
so a later server change cannot move this gate; only the page can.

Driven by clicking, in headless Edge, against the real template and bundle,
fetch stubbed:
  1. a single log dropped on the page
  2. Day of Week: one weekday unticked BY CLICK
  3. Export CSV clicked (the file text captured from the Blob)
  4. Rank metrics clicked (the /surface/rank request body captured)
  5. reset all filters; a 4-variant sweep dropped -- ALL
  6. one variant picked in the selector
  7. a 3-parameter sweep dropped -- ALL, the grid pooling the hidden one
At each step: the rendered DOM (every element: tag, class, shown or not,
its own text, input state, inline style), every Chart.js chart's data and
titles, and the component's computed figures (stats, extra, the filtered
indices, the equity and deployment series, the grid).

THE EXCLUSION: any element marked data-gate-exclude="train-test-card" is
left out of the DOM comparison, with its subtree -- the sidebar card the
split adds is the one thing allowed to exist with the split off. What that
card may hold while Off is asserted separately, by the split's own gate.

  python scripts/check_oo_offstate.py            compare with the recording
  python scripts/check_oo_offstate.py --record   (re)record -- only on a
      commit whose page is the one to keep; say why in the commit message
  python scripts/check_oo_offstate.py --rerecord-from REF
      re-record with THIS driver but the page (templates, static) exactly
      as it was at REF, and the recording's own inputs -- for a change in
      how the page is DRIVEN, never in what it is. Used once (2026-10-25):
      the move from --virtual-time-budget to real time, where virtual time
      had recorded an artifact (Alpine never shows an x-show element on
      virtual time once hidden: the drop zone's label after a second upload).

No browser (the VPS) -> SKIP.
"""
from __future__ import annotations

import gzip
import html as _html
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import check_oo_notes_ui as N  # noqa: E402  (page rendering + browser lookup)

EXIT_SKIPPED = 3
BASELINE = ROOT / "scripts" / "fixtures" / "oo_offstate_baseline.json.gz"
EXCLUDE = "train-test-card"
# What the page is built from, for the "recorded from uncommitted code" warning.
PAGE_FILES = ["templates/oo_backtest.html", "templates/_nav.html", "templates/_macros.html", "static/js/oo_backtest.js",
              "static/js/backtest_core.js", "static/js/nav.js", "static/css"]

SNAPSHOT_JS = r"""
  const shown = el => !!el && el.getClientRects().length > 0 && getComputedStyle(el).visibility !== 'hidden';
  const txt = el => (el ? el.textContent.replace(/\s+/g, ' ').trim() : '');

  // The rendered DOM, one line per element: depth, tag, class, H when not
  // shown, input state, inline style, then its OWN text (child elements are
  // their own lines). Alpine's x-* attributes are template source, not
  // render, and are not read; an x-if that renders nothing is simply absent.
  const dom = () => {
    const lines = [];
    const walk = (el, d) => {
      if (el.id === 'report' || el.tagName === 'SCRIPT' || el.tagName === 'STYLE' || el.tagName === 'TEMPLATE') return;
      // The split's card: absent from the recording, so absent here too.
      if (el.getAttribute && el.getAttribute('data-gate-exclude') === EXCLUDE) { window.__excluded++; return; }
      const own = [...el.childNodes].filter(n => n.nodeType === 3).map(n => n.textContent).join(' ').replace(/\s+/g, ' ').trim();
      const bits = [d, el.tagName.toLowerCase(), el.getAttribute('class') || '', shown(el) ? '' : 'H'];
      if (el.tagName === 'INPUT') bits.push(`${el.type}=${el.type === 'file' ? '' : el.value}${el.checked ? ' checked' : ''}`);
      if (el.tagName === 'SELECT') bits.push('=' + el.value + ' [' + [...el.options].map(o => o.value + ':' + o.textContent.trim()).join(';') + ']');
      if (el.disabled) bits.push('disabled');
      // A canvas's inline size is Chart.js's resize, which lands on its own
      // frame; what it draws is compared under "charts".
      if (el.getAttribute('style') && el.tagName !== 'CANVAS') bits.push('style=' + el.getAttribute('style'));
      if (el.getAttribute('title')) bits.push('title=' + el.getAttribute('title'));
      bits.push(own);
      lines.push(bits.join('|'));
      for (const ch of el.children) walk(ch, d + 1);
    };
    walk(document.body, 0);
    return lines;
  };
  // Every chart: the data drawn and the words around it. Functions (scriptable
  // options) are not data and drop out of JSON.
  const charts = () => [...document.querySelectorAll('canvas')].map(cv => {
    const ch = Chart.getChart(cv);
    if (!ch) return { id: cv.id || '', chart: null };
    const o = ch.options || {}, sc = o.scales || {};
    return {
      id: cv.id || '', type: ch.config.type, labels: ch.data.labels,
      datasets: ch.data.datasets.map(ds => ({ label: ds.label, type: ds.type, data: ds.data, hidden: !!ds.hidden,
                                             backgroundColor: ds.backgroundColor, borderColor: ds.borderColor,
                                             borderDash: ds.borderDash, yAxisID: ds.yAxisID })),
      title: o.plugins && o.plugins.title && o.plugins.title.text,
      scales: Object.fromEntries(Object.entries(sc).map(([k, s]) => [k, { min: s.min, max: s.max, type: s.type,
                                                                          title: s.title && s.title.text }])),
    };
  });
  const state = (c) => JSON.parse(JSON.stringify({
    stats: c.stats, extra: c.extra, filteredCount: c.filteredCount, activeCount: c.activeCount(),
    idx: OB_DATA.idx, eq: OB_DATA.eq, conc: OB_DATA.conc, vk: c.vk, variantSel: c.variantSel, grids: c.grids,
    dateBounds: c.dateBounds, autoSteps: c.autoSteps,
    vm: OB_DATA.vm && OB_DATA.vm.map(f => ({ id: f.id, stats: f.stats, calmar: f.calmar })),
  }));
  // The ranking stamps its own clock time and duration; nothing else does.
  const scrub = lines => lines.map(l => l.replace(/\b\d\d:\d\d\b/g, 'HH:MM').replace(/\b\d+ ?ms\b/g, 'N ms'));
"""

DRIVER = r"""
<script>
// NO requestAnimationFrame shim. It mapped frames to setTimeout(0) because
// virtual time runs no frames; this page now runs in REAL time (edge_cdp),
// where real frames run -- and the shim was itself a race: Alpine shows an
// x-show element on the next frame and hides one at once, so with frames as
// zero-delay timers and a stubbed upload that finishes instantly, a late
// SHOW landed after the hide and "Parsing 4 files..." stuck on screen after
// every sweep upload that followed a single file (12 of 12 with the shim,
// 0 of 12 without, 2026-10-25). The drop zone read either way run to run.
window.__errs = [];
window.onerror = (m) => window.__errs.push('error: ' + m);
window.addEventListener('unhandledrejection', e => window.__errs.push('promise: ' + e.reason));
(function () {
  const ce = console.error;
  console.error = (...a) => { window.__errs.push('console.error: ' + a.map(String).join(' ')); ce(...a); };
})();
const FIX = __FIX__;
const EXCLUDE = '__EXCLUDE__';
window.__next = 'alone';
window.__excluded = 0;
window.__requests = [];
window.__blobs = [];
const realURL = URL.createObjectURL.bind(URL);
URL.createObjectURL = (b) => { window.__blobs.push(b); return realURL(b); };
HTMLAnchorElement.prototype.click = function () {};          // no download navigation
window.fetch = async (u, init) => {
  const url = String(u);
  const reply = (status, body) => ({ ok: status < 400, status, json: async () => JSON.parse(JSON.stringify(body)) });
  if (/\/registry$/.test(url)) return reply(200, { metrics: FIX.registry, coverage_error: null });
  if (/\/strategies$/.test(url) && !(init && init.method === 'POST')) return reply(200, { strategies: [] });
  if (/\/market-status$/.test(url)) return reply(200, { ok: false, error: 'stub' });
  if (/\/parse-batch$/.test(url)) {
    window.__requests.push({ url: 'parse-batch', files: init.body.getAll('files').map(f => f.name) });
    return reply(200, FIX[window.__next]);
  }
  if (/\/surface\/catalog$/.test(url)) return reply(200, FIX.catalog);
  if (/\/surface\/rank$/.test(url)) {
    const body = JSON.parse(init.body);
    window.__requests.push({ url: 'surface/rank', body });
    return reply(200, { rows: FIX.rankRows, report: { no_bar: 0, with_bar: body.pnl.length, distinct_entries: body.bars.length },
                        lookahead_confirmed: true });
  }
  window.__requests.push({ url, stub: 404 });
  return reply(404, { detail: 'stub' });
};

window.addEventListener('load', () => setTimeout(async () => {
  const wait = ms => new Promise(r => setTimeout(r, ms));
  const until = async (cond, ms = 6000) => { for (let t = 0; t < ms; t += 25) { if (cond()) return true; await wait(25); } return cond(); };
  const steps = [], notes = [];
__SNAPSHOT__  const snap = async (name, c) => {
    document.documentElement.setAttribute('data-progress', name);
    await wait(300);
    window.__excluded = 0;
    const d = scrub(dom());
    steps.push({ name, dom: d, charts: charts(), state: state(c), excluded: window.__excluded });
  };
  const drop = (names) => {
    const dt = new DataTransfer();
    for (const n of names) dt.items.add(new File(['x'], n, { type: 'text/csv' }));
    const input = document.querySelector('.ob-drop input[type=file]');
    input.files = dt.files;
    input.dispatchEvent(new Event('change', { bubbles: true }));
  };
  const card = t => [...document.querySelectorAll('.ob-side .ob-card')].find(e => txt(e.querySelector('.ob-card-title')) === t);
  try {
    let c = null;
    for (let i = 0; i < 100 && !(c && typeof c.recompute === 'function'); i++) {
      await wait(50); c = document.body._x_dataStack ? Alpine.$data(document.body) : null;
    }
    await until(() => c.registry.length > 0);
    await snap('0 empty page', c);

    window.__next = 'alone'; drop([FIX.aloneName]);
    await until(() => c.loaded && c.filteredCount > 0);
    await snap('1 single log', c);

    const box = [...card('Day of Week').querySelectorAll('.ob-checks input')].find(x => x.checked);
    const before = c.filteredCount;
    box.click();
    await until(() => c.filteredCount !== before);
    await snap('2 Day of Week: first weekday unticked by click', c);

    const exp = [...document.querySelectorAll('button')].find(b => txt(b) === 'Export CSV');
    exp.click();
    await until(() => window.__blobs.length > 0);
    const csv = window.__blobs.length ? await window.__blobs[window.__blobs.length - 1].text() : null;

    const rank = [...document.querySelectorAll('button')].find(b => txt(b) === 'Rank metrics');
    rank.click();
    await until(() => c.surf.result && !c.surf.busy);
    await snap('3 Rank metrics clicked', c);

    [...document.querySelectorAll('.ob-status a')].find(a => /reset all filters/.test(txt(a))).click();
    await until(() => c.activeCount() === 0);
    window.__next = 'sweep'; drop(FIX.sweepNames);
    await until(() => c.sweep && c.filteredCount === FIX.sweep.n);
    await snap('4 sweep at ALL', c);

    const sel = document.getElementById('ob-variant');
    sel.value = sel.options[3].value;
    sel.dispatchEvent(new Event('change', { bubbles: true }));
    await until(() => c.variantSel !== 'all');
    await snap('5 one variant picked', c);

    window.__next = 'sweep3'; drop(FIX.sweep3Names);
    await until(() => c.sweep && c.filteredCount === FIX.sweep3.n && c.variantSel === 'all');
    await snap('6 three-parameter sweep at ALL', c);

    window.__result = { steps, csv, requests: window.__requests, errs: window.__errs };
  } catch (e) {
    window.__result = { error: String(e) + (e && e.stack ? ' ' + e.stack : ''), steps, errs: window.__errs };
  }
  const pre = document.createElement('pre');
  pre.id = 'report';
  pre.textContent = JSON.stringify(window.__result);
  document.body.appendChild(pre);
}, 400));
</script>
"""


def inputs() -> dict:
    """What the page is fed, built by the real server code (record time only)."""
    import check_oo_sweep_ui as S
    from app.oo_backtest.registry import registry_with_coverage
    sweep, alone, _expect, _files, sweep3 = S.payloads()
    cat = [{"column_name": f"m{j}", "family": "iv", "form": "level", "tenor": None, "wing": None,
            "min_date": "2024-03-01", "units": None, "description": f"metric {j}", "formula": None} for j in range(6)]
    return {
        "registry": registry_with_coverage(None), "alone": alone, "sweep": sweep, "sweep3": sweep3,
        "aloneName": S.NAMES[2], "sweepNames": S.NAMES, "sweep3Names": S.NAMES3,
        "catalog": {"metrics": cat, "family_groups": [{"label": "IV", "color": "#3498db", "families": ["iv"]}],
                    "other_group": None, "form_labels": {"level": "level"}, "unit_formats": {},
                    "default_unit_format": {"scale": 1, "decimals": 2, "suffix": ""}, "row_auto_bins": None,
                    "built_at": "x"},
        "rankRows": [{"column": f"m{j}", "family": "iv", "tenor": None, "wing": None, "form": "level", "n": 40,
                      "bars": 40, "pearson": 0.3 - j * 0.05, "pearson_p": 1e-3 * (j + 1), "pearson_p_bh": 1e-2 * (j + 1),
                      "spearman": 0.3 - j * 0.05, "spearman_p": 1e-3 * (j + 1), "spearman_p_bh": 1e-2 * (j + 1)}
                     for j in range(6)],
    }


def drive(browser: str, fix: dict) -> dict:
    html = N.build_page()
    html = html.replace(N.DRIVER.replace("__SAVED__", json.dumps(N.SAVED)), "")
    page = html.replace("</head>", DRIVER.replace("__SNAPSHOT__", SNAPSHOT_JS).replace("__FIX__", json.dumps(fix)).replace("__EXCLUDE__", EXCLUDE)
                        + "</head>", 1)
    import os

    import edge_cdp
    tmp = ROOT / "scripts" / f"_oo_offstate_{os.getpid()}.html"   # unique: a gate may run beside itself
    tmp.write_text(page, encoding="utf-8")
    # REAL TIME over the DevTools protocol (edge_cdp), not --virtual-time-budget:
    # see edge_cdp's docstring for the flake that retired the budget.
    try:
        return json.loads(edge_cdp.run_page(browser, tmp, timeout_s=240))
    finally:
        tmp.unlink(missing_ok=True)


def first_diffs(a, b, path="", out=None, limit=12):
    """Where two JSON values differ, as readable paths (the first `limit`)."""
    out = [] if out is None else out
    if len(out) >= limit:
        return out
    if type(a) is not type(b):
        out.append(f"{path}: {json.dumps(a)[:160]}  ->  {json.dumps(b)[:160]}")
    elif isinstance(a, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a or k not in b:
                out.append(f"{path}.{k}: {'absent' if k not in a else json.dumps(a[k])[:120]}  ->  "
                           f"{'absent' if k not in b else json.dumps(b[k])[:120]}")
            else:
                first_diffs(a[k], b[k], f"{path}.{k}", out, limit)
    elif isinstance(a, list):
        if len(a) != len(b):
            out.append(f"{path}: {len(a)} items  ->  {len(b)} items")
        for i, (x, y) in enumerate(zip(a, b)):
            first_diffs(x, y, f"{path}[{i}]", out, limit)
    elif a != b:
        out.append(f"{path}: {json.dumps(a)[:160]}  ->  {json.dumps(b)[:160]}")
    return out


def dom_diff(a: list[str], b: list[str], limit=12) -> list[str]:
    import difflib
    d = [l for l in difflib.unified_diff(a, b, "recorded", "now", n=1, lineterm="") if l[:1] in "+-@"][:limit * 3]
    return [l[:200] for l in d]


def main() -> int:
    record = "--record" in sys.argv
    rerecord = sys.argv[sys.argv.index("--rerecord-from") + 1] if "--rerecord-from" in sys.argv else None
    wt = None
    browser = N.find_browser()
    if browser is None:
        print("  SKIP  no Edge/Chromium on this host -- the page was NOT driven")
        return EXIT_SKIPPED
    if record:
        fix = inputs()
    elif rerecord:
        # The page from REF, in a worktree beside this one; the inputs from
        # the recording; this driver.
        old = json.loads(gzip.decompress(BASELINE.read_bytes()))
        fix = old["inputs"]
        wt = ROOT.parent / f"_offstate_rerecord_{rerecord}"
        subprocess.run(["git", "worktree", "add", "-q", "--detach", str(wt), rerecord], cwd=ROOT, check=True)
        N.ROOT = wt
    else:
        if not BASELINE.exists():
            print(f"  FAIL  no recording at {BASELINE.relative_to(ROOT)} -- nothing to compare against")
            return 1
        base = json.loads(gzip.decompress(BASELINE.read_bytes()))
        fix = base["inputs"]
    try:
        got = drive(browser, fix)
    finally:
        if wt is not None:
            N.ROOT = ROOT
            subprocess.run(["git", "worktree", "remove", "--force", str(wt)], cwd=ROOT)
    fails = 0
    if got.get("error"):
        print(f"  FAIL  the driver threw: {got['error'][:600]}")
        fails += 1
    if got.get("errs"):
        print(f"  FAIL  console: {' ~ '.join(got['errs'][:6])[:600]}")
        fails += 1
    if rerecord:
        if fails:
            print("\nnot re-recorded: the run itself failed")
            return 1
        rec = {"recorded_at": rerecord, "page_from": rerecord, "driver": "real time (edge_cdp)",
               "dirty": False, "inputs": fix, "result": got}
        BASELINE.write_bytes(gzip.compress(json.dumps(rec, sort_keys=True).encode(), mtime=0))
        n = sum(len(st["dom"]) for st in got["steps"])
        print(f"re-recorded {len(got['steps'])} steps ({n:,} DOM lines) from the page at {rerecord}, driven in real time")
        return 0
    if record:
        if fails:
            print("\nnot recorded: the run itself failed")
            return 1
        rec = {"recorded_at": subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                                             cwd=ROOT).stdout.strip(),
               "dirty": bool(subprocess.run(["git", "status", "--porcelain", "--", *PAGE_FILES],
                                            capture_output=True, text=True, cwd=ROOT).stdout.strip()),
               "inputs": fix, "result": got}
        BASELINE.parent.mkdir(exist_ok=True)
        BASELINE.write_bytes(gzip.compress(json.dumps(rec, sort_keys=True).encode(), mtime=0))
        n = sum(len(s["dom"]) for s in got["steps"])
        print(f"recorded {len(got['steps'])} steps ({n:,} DOM lines, "
              f"{sum(len(s['charts']) for s in got['steps'])} charts) at {rec['recorded_at']}"
              + (" -- WARNING: the page's files have uncommitted changes" if rec["dirty"] else ""))
        return 0

    want = base["result"]
    print(f"recorded at {base['recorded_at']}; {len(want['steps'])} steps")
    names = [s["name"] for s in want["steps"]]
    if [s["name"] for s in got["steps"]] != names:
        print(f"  FAIL  steps: {[s['name'] for s in got['steps']]} != {names}")
        fails += 1
    for w, g in zip(want["steps"], got["steps"]):
        # The exclusion covers ONE element (the split's card) or none -- it
        # must not be able to hide anything else.
        ex = g.get("excluded", 0)
        print(("  ok    " if ex <= 1 else "  FAIL  ") + f"{w['name']}: {ex} element(s) excluded (at most 1)")
        fails += ex > 1
        for part in ("dom", "charts", "state"):
            if part == "dom":
                same = w["dom"] == g["dom"]
                why = [] if same else dom_diff(w["dom"], g["dom"])
            else:
                same = w[part] == g[part]
                why = [] if same else first_diffs(w[part], g[part], part)
            print(("  ok    " if same else "  FAIL  ") + f"{w['name']}: {part}"
                  + (f" ({len(w['dom']):,} elements)" if part == "dom" else ""))
            for line in why:
                print("          " + line)
            fails += not same
    for part, label in (("csv", "Export CSV text"), ("requests", "requests the page sent (incl. the /surface/rank body)")):
        same = want.get(part) == got.get(part)
        print(("  ok    " if same else "  FAIL  ") + label)
        if not same:
            for line in first_diffs(want.get(part), got.get(part), part):
                print("          " + line)
        fails += not same
    print()
    print(f"OO off-state: {len(names) * 4 + 2} comparisons with the recording, failures: {fails}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
