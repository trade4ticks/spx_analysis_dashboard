"""Parameter sweeps on the OO/Mesosim page, BY CLICKING (P2).

check_oo_sweep holds the server half. This holds what only the page can get
wrong, in headless Edge against the real template and bundle, fetch stubbed
with payloads the REAL router built (analyze_batch, no database):

  * dropping several files sends ONE /parse-batch request with every file,
    each gzipped; a refused batch says why and leaves the loaded log alone
  * the load report says what every file was read as
  * the variant selector lists ALL (the default) and every variant
  * at ALL the scale-dependent figures are the MEAN PER VARIANT -- checked
    against stats.py's calculate_stats run on each variant's own file, not
    against the page's own arithmetic -- while # trades, Win % and the
    per-trade figures pool; those five boxes are tagged, the rest are not;
    the cumulative curve ends at the mean total; the note says which is which
  * one variant selected IS that variant's file loaded alone: every summary
    figure, the status line, every section's subtitle and the date bounds
    read identically in the two
  * a filter set at ALL survives switching variant
  * Export CSV carries param_* columns
  * the surface ranking marks no significance over pooled variants, and does
    for one variant
  * a multi-file sweep offers no Save (P4); a single file still does
  * no console errors (an Alpine expression error leaves a control inert)

No browser (the VPS) -> SKIP.
"""
from __future__ import annotations

import asyncio
import html as _html
import io
import json
import random
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import pandas as pd  # noqa: E402

import check_oo_notes_ui as N  # noqa: E402  (page rendering + browser lookup)

EXIT_SKIPPED = 3

HEADER = ["Date Opened", "Time Opened", "Opening Price", "Legs", "Premium", "Closing Price",
          "Date Closed", "Time Closed", "Avg. Closing Cost", "Reason For Close", "P/L", "P/L %",
          "No. of Contracts", "Funds at Close", "Margin Req.", "Strategy", "Gap", "Movement"]
VARIANTS = [(30, 25), (30, 50), (45, 25), (45, 50)]
NAMES = [f"sweep_dte{d}_pt{p}.csv" for d, p in VARIANTS]


def make_csv(seed: int) -> str:
    """~60 trades through 2024. Each variant its own P/L path, so the mean of
    the variants' drawdowns is NOT the pooled drawdown (the gate needs that)."""
    rng = random.Random(seed)
    days = pd.bdate_range("2024-01-02", "2024-12-20")
    rows = []
    for _ in range(60):
        i = rng.randrange(len(days) - 6)
        d0, d1 = days[i], days[i + rng.randrange(1, 6)]
        pnl = round(rng.gauss(40 if seed % 2 else -10, 380), 2)
        rows.append([d0.date().isoformat(), "10:00:00", "4800", "1 P STO", str(round(-300 - rng.random() * 700, 2)),
                     "4810", d1.date().isoformat(), "15:59:00", "0", rng.choice(["Profit Target", "Stop Loss"]),
                     str(pnl), "10", "1", "100000", "5000", "IC", "0.3", "20"])
    out = io.StringIO()
    pd.DataFrame(rows, columns=HEADER).to_csv(out, index=False)
    return out.getvalue()


def payloads() -> tuple[dict, dict, dict, dict]:
    """(sweep payload, the third variant's file alone, expected figures, files)."""
    import logging

    from app.oo_backtest import data_loader, market_calendar
    from app.oo_backtest.stats import calculate_stats
    from app.routers.oo_backtest import analyze_batch
    logging.getLogger("app.routers.oo_backtest").setLevel(logging.CRITICAL)   # no pool: "not joined" is expected
    files = {n: make_csv(i) for i, n in enumerate(NAMES)}

    def with_sessions(p):
        p["market"]["spx_sessions"] = market_calendar.sessions(p["date_min"], p["date_max"])
        return p
    sweep = with_sessions(asyncio.run(analyze_batch([(n, c.encode()) for n, c in files.items()], None)))
    alone = with_sessions(asyncio.run(analyze_batch([(NAMES[2], files[NAMES[2]].encode())], None)))

    per = []
    for n in NAMES:
        df = data_loader.parse_upload(files[n].encode(), n)
        st = calculate_stats(df)
        closes = pd.to_datetime(df["date_closed"])
        years = (closes.max() - closes.min()).days / 365.25
        ann = st["total_pnl"] / years
        per.append({"total": st["total_pnl"], "dd": st["max_drawdown"], "ann": ann,
                    "calmar": ann / abs(st["max_drawdown"]) if st["max_drawdown"] < 0 else None})
    allp = pd.concat([data_loader.parse_upload(files[n].encode(), n) for n in NAMES])
    mean = lambda xs: sum(xs) / len(xs)  # noqa: E731
    expect = {"total": mean([p["total"] for p in per]), "dd": mean([p["dd"] for p in per]),
              "ann": mean([p["ann"] for p in per]),
              "calmar": mean([p["calmar"] for p in per if p["calmar"] is not None]),
              "pooled_total": float(allp["pnl"].sum()), "pooled_dd": calculate_stats(allp)["max_drawdown"],
              "trades": len(allp), "win": float((allp["pnl"] > 0).mean() * 100)}
    return sweep, alone, expect, files


DRIVER = r"""
<script>
window.requestAnimationFrame = cb => setTimeout(() => cb(performance.now()), 0);
window.cancelAnimationFrame = id => clearTimeout(id);
window.__errs = [];
window.onerror = (m) => window.__errs.push('error: ' + m);
window.addEventListener('unhandledrejection', e => window.__errs.push('promise: ' + e.reason));
(function () {
  const ce = console.error;
  console.error = (...a) => { window.__errs.push('console.error: ' + a.map(String).join(' ')); ce(...a); };
})();
const FIX = __FIX__;
window.__sent = [];
window.__next = 'sweep';
window.fetch = async (u, init) => {
  const url = String(u);
  const reply = (status, body) => ({ ok: status < 400, status, json: async () => JSON.parse(JSON.stringify(body)) });
  if (/\/registry$/.test(url)) return reply(200, { metrics: FIX.registry, coverage_error: null });
  if (/\/strategies$/.test(url)) return reply(200, { strategies: [] });
  if (/\/market-status$/.test(url)) return reply(200, { ok: false, error: 'stub' });
  if (/\/parse-batch$/.test(url)) {
    const parts = init.body.getAll('files');
    const heads = [];
    for (const f of parts) { const b = new Uint8Array(await f.slice(0, 2).arrayBuffer()); heads.push(b[0] === 0x1f && b[1] === 0x8b); }
    window.__sent.push({ names: parts.map(f => f.name), gz: heads });
    if (window.__next === 'refuse') return reply(400, { detail: 'sweep_x.csv has dte; the other files have dte, pt. Nothing was loaded.' });
    return reply(200, window.__next === 'alone' ? FIX.alone : FIX.sweep);
  }
  if (/\/surface\/catalog$/.test(url)) return reply(200, FIX.catalog);
  if (/\/surface\/rank$/.test(url)) {
    const body = JSON.parse(init.body);
    return reply(200, { rows: FIX.rankRows, report: { no_bar: 0, with_bar: body.trades.length, distinct_entries: 50 },
                        lookahead_confirmed: true });
  }
  return reply(404, { detail: 'stub' });
};

window.addEventListener('load', () => setTimeout(async () => {
  const out = [];
  const ok = (k, got, want) => out.push((JSON.stringify(got) === JSON.stringify(want) ? 'ok|' : 'FAIL|') + k + '|' +
                                         JSON.stringify(got) + '|' + JSON.stringify(want));
  const near = (k, got, want, tol) => out.push((Math.abs(got - want) <= tol ? 'ok|' : 'FAIL|') + k + '|' + got + '|' + want);
  const wait = ms => new Promise(r => setTimeout(r, ms));
  const until = async (cond, ms = 4000) => { for (let t = 0; t < ms; t += 25) { if (cond()) return true; await wait(25); } return cond(); };
  const report = () => {
    const pre = document.createElement('pre');
    pre.id = 'report';
    pre.textContent = out.concat(['errs|' + (window.__errs.slice(0, 6).join(' ~ ') || 'clean')]).join(String.fromCharCode(10));
    document.body.appendChild(pre);
  };
  const shown = el => !!el && el.getClientRects().length > 0 && getComputedStyle(el).visibility !== 'hidden';
  const txt = el => (el ? el.textContent.replace(/\s+/g, ' ').trim() : '');
  const money = t => Number(String(t).replace(/[$,\s]/g, ''));
  const box = label => [...document.querySelectorAll('.ob-summary .stat-item')]
    .find(b => b.querySelector('.stat-label').childNodes[0].textContent.trim() === label);
  const boxVal = label => txt(box(label) && box(label).querySelector('.stat-value'));
  const boxTag = label => { const t = box(label) && box(label).querySelector('.stat-tag'); return t && shown(t) ? txt(t) : ''; };
  const saveNote = () => [...document.querySelectorAll('.ob-parse-note')].find(e => /cannot be saved yet/.test(txt(e)));
  const LABELS = ['# Trades', 'Win %', 'Total P/L', 'Avg P/L', 'Avg P/L %', 'Avg Win', 'Avg Loss', 'Max Win', 'Max Loss',
                  'Profit Factor', 'Avg Annual P/L', 'Avg Annual Return %', 'Max DD', 'Calmar', 'Avg Days'];
  const snapshot = (c) => ({
    stats: Object.fromEntries(LABELS.map(l => [l, boxVal(l)])),
    status: txt(document.querySelector('.ob-status span')),
    subs: [...document.querySelectorAll('.ob-card-sub')].filter(shown).map(txt).filter(t => /trades with a value/.test(t)),
    bounds: c.dateBounds, deploy: c.deployTitle(), dd: c.maxDDLine(), steps: c.autoSteps,
  });
  const drop = (names) => {
    const dt = new DataTransfer();
    for (const n of names) dt.items.add(new File([FIX.files[n]], n, { type: 'text/csv' }));
    const input = document.querySelector('.ob-drop input[type=file]');
    input.files = dt.files;
    input.dispatchEvent(new Event('change', { bubbles: true }));
  };
  try {
    let c = null;
    for (let i = 0; i < 100 && !(c && typeof c.setVariant === 'function'); i++) {
      await wait(50); c = document.body._x_dataStack ? Alpine.$data(document.body) : null;
    }
    ok('the page component is up', !!c && typeof c.setVariant, 'function');
    await until(() => c.registry.length > 0);
    ok('the file input takes several files', document.querySelector('.ob-drop input[type=file]').multiple, true);

    // ── one request, every file, gzipped ────────────────────────────────
    drop(FIX.names);
    ok('the sweep loads', await until(() => c.loaded && !!c.sweep && c.filteredCount > 0), true);
    ok('one request with all four files', window.__sent.length === 1 && window.__sent[0].names, FIX.names);
    ok('each part gzipped before upload', window.__sent[0].gz, [true, true, true, true]);

    // ── the load report ─────────────────────────────────────────────────
    ok('the headline says what was read', txt(document.querySelector('.ob-sweep .ob-parse-note')),
       '4 variants · dte, pt (from the filenames) · 4 files');
    const toggle = [...document.querySelectorAll('.ob-sweep a')].find(a => /each file/.test(txt(a)));
    toggle.click();
    await until(() => shown(document.querySelector('.ob-sweep-files')));
    const lines = [...document.querySelectorAll('.ob-sweep-files div')].map(txt);
    ok('every file is listed with what its name was read as', lines.map(l => l.replace(/ · \d+ trades.*$/, '')),
       FIX.names.map((n, i) => `${n}: read as dte=${FIX.variants[i][0]} · pt=${FIX.variants[i][1]} (not read: sweep)`));

    // ── the selector ────────────────────────────────────────────────────
    const sel = document.querySelector('#ob-variant');
    ok('the variant selector is on screen', shown(sel), true);
    ok('it lists ALL and every variant', [...sel.options].map(o => txt(o).replace(/ \(\d+ trades\)$/, '')),
       ['ALL — 4 variants', 'dte=30 · pt=25', 'dte=30 · pt=50', 'dte=45 · pt=25', 'dte=45 · pt=50']);
    ok('ALL is selected on load', sel.value, 'all');

    // ── ALL: the mean per variant, against stats.py per file ────────────
    const E = FIX.expect;
    ok('# Trades pools every trade', boxVal('# Trades'), E.trades.toLocaleString());
    near('Win % pools every trade', parseFloat(boxVal('Win %')), E.win, 0.051);
    near('Total P/L is the mean of the variants\' totals (stats.py per file)', money(boxVal('Total P/L')), E.total, 0.51);
    near('Max DD is the mean of the variants\' own Max DD (stats.py per file)', money(boxVal('Max DD')), E.dd, 0.51);
    near('Avg Annual P/L is the mean of the variants\'', money(boxVal('Avg Annual P/L')), E.ann, 0.51);
    near('Calmar is the mean of the variants\'', parseFloat(boxVal('Calmar')), E.calmar, 0.0051);
    ok('and none of them is the pooled basket',
       [Math.abs(money(boxVal('Total P/L')) - E.pooled_total) > 1, Math.abs(money(boxVal('Max DD')) - E.pooled_dd) > 1], [true, true]);
    ok('the five per-variant boxes are tagged, the others are not',
       LABELS.filter(l => boxTag(l) === 'mean / variant'),
       ['Total P/L', 'Avg Annual P/L', 'Avg Annual Return %', 'Max DD', 'Calmar']);
    ok('the note beside the selector says which is which', /MEAN PER VARIANT/.test(txt(document.querySelector('.ob-variant-note'))) &&
       /Avg P\/L, Win %/.test(txt(document.querySelector('.ob-variant-note'))), true);
    const cum = OB_CHARTS.cum.data.datasets[0].data;
    near('the cumulative curve ends at the mean total', cum[cum.length - 1].y, E.total, 0.01);
    ok('the drawdown title says it is the mean of the variants\' drawdowns', /mean of the variants' drawdowns/.test(c.maxDDLine()), true);
    ok('the section totals say they are per variant',
       [...document.querySelectorAll('.chart-title')].filter(shown).map(txt).filter(t => /^Total P\/L by/.test(t))
         .every(t => /mean per variant/.test(t)), true);
    ok('the deployment title is the mean of the variants\' peaks', /mean peak [\d.]+ per variant/.test(c.deployTitle()), true);

    // ── export carries the parameters ───────────────────────────────────
    const csv = c.exportText().split(/\r\n/);
    ok('Export CSV leads with param_* columns', csv[0].split(',').slice(0, 2), ['param_dte', 'param_pt']);
    ok('every row has its variant\'s values', csv.slice(1, -1).every(l => /^(30|45),(25|50),/.test(l)), true);

    // ── a filter set at ALL survives switching variant ──────────────────
    const dow = c.registry.find(m => m.key === 'day_of_week');
    c.toggleCat(dow, 0);           // untick Monday, as its checkbox does
    await until(() => c.activeCount() === 1);

    // ── the ranking: no significance over pooled variants ───────────────
    await c.rankSurface();
    await until(() => !!c.surf.result);
    ok('pooled: the note says significance is not marked', await until(() => shown(document.querySelector('.ob-rank-pooled'))) &&
       /same market day appears once per variant/.test(txt(document.querySelector('.ob-rank-pooled'))), true);
    ok('pooled: no Benjamini-Hochberg in the footer', /Benjamini/.test(c.surfFooter()), false);

    // ── one variant IS its own file ─────────────────────────────────────
    sel.value = '2';
    sel.dispatchEvent(new Event('change', { bubbles: true }));
    await until(() => c.variantSel === '2' && c.viewN === FIX.sweep.sweep.variants[2].n);
    await wait(300);
    ok('the Monday filter survived the switch', [c.activeCount(), c.catOf(dow).selected.includes(0)], [1, false]);
    ok('one variant: no box is tagged', LABELS.filter(l => boxTag(l)), []);
    ok('one variant: the ranking is stale (another sample)', c.surfStale(), true);
    await c.rankSurface();
    await until(() => !c.surf.busy);
    ok('one variant: no pooled note, BH in the footer', [await until(() => !shown(document.querySelector('.ob-rank-pooled'))),
       /Benjamini/.test(c.surfFooter())], [true, true]);
    // Clear the filter so both sides start from the same untouched page.
    c.setAllCats(dow, true);
    await wait(300);
    const asVariant = snapshot(c);

    // The same file loaded ALONE, by dropping it.
    window.__next = 'alone';
    drop([FIX.names[2]]);
    await until(() => c.loaded && !c.sweep && window.__sent.length === 2);
    await wait(400);
    ok('a single file: one part, no selector', [window.__sent[1].names, shown(sel)], [[FIX.names[2]], false]);
    const alone = snapshot(c);
    ok('the comparison has something to compare: sections with subtitles, a deployment title',
       [alone.subs.length >= 3, !!alone.deploy, Object.values(alone.stats).every(v => v !== '')], [true, true, true]);
    ok('and the single file shows no saved-sweep note', shown(saveNote()), false);
    for (const k of Object.keys(alone.stats)) ok(`variant view == its file alone: ${k}`, asVariant.stats[k], alone.stats[k]);
    ok('variant view == its file alone: status line', asVariant.status, alone.status);
    ok('variant view == its file alone: every section subtitle', asVariant.subs, alone.subs);
    ok('variant view == its file alone: date bounds, auto-bin steps, drawdown and deployment titles',
       [asVariant.bounds, asVariant.steps, asVariant.dd, asVariant.deploy], [alone.bounds, alone.steps, alone.dd, alone.deploy]);
    ok('a single uploaded file can be saved', await until(() => shown(document.querySelector('.ob-save'))), true);

    // ── a refused batch leaves the page alone ───────────────────────────
    window.__next = 'sweep';
    drop(FIX.names);
    await until(() => !!c.sweep);
    ok('a multi-file sweep offers no Save, and says why',
       [await until(() => !shown(document.querySelector('.ob-save'))), await until(() => shown(saveNote()))], [true, true]);
    window.__next = 'refuse';
    drop(['sweep_x.csv'].concat(FIX.names.slice(1)).map(n => FIX.files[n] ? n : FIX.names[0]));
    await until(() => !!c.uploadError);
    const upErr = document.querySelector('.ob-drop').parentElement.querySelector(':scope > .ob-err');
    ok('a refused batch says why, under the drop zone', /Nothing was loaded/.test(c.uploadError) &&
       await until(() => shown(upErr) && /Nothing was loaded/.test(txt(upErr))), true);
    ok('and the loaded sweep is untouched', [c.sweep && c.sweep.variants.length, c.viewN], [4, FIX.sweep.n]);
    // The refusal is logged on purpose; only other errors count.
    window.__errs = window.__errs.filter(e => !/oo-backtest upload/.test(e));
  } catch (e) {
    out.push('FAIL|the driver threw|' + e + (e && e.stack ? ' ' + e.stack.split(String.fromCharCode(10))[1] : '') + '|nothing');
  }
  report();
}, 400));
</script>
"""


def build_page(fix: dict) -> str:
    html = N.build_page()
    html = html.replace(N.DRIVER.replace("__SAVED__", json.dumps(N.SAVED)), "")
    return html.replace("</head>", DRIVER.replace("__FIX__", json.dumps(fix)) + "</head>", 1)


def main() -> int:
    browser = N.find_browser()
    if browser is None:
        print("  SKIP  no Edge/Chromium on this host — the page was NOT driven")
        return EXIT_SKIPPED
    from app.oo_backtest.registry import registry_with_coverage
    sweep, alone, expect, files = payloads()
    cat = [{"column_name": f"m{j}", "family": "iv", "form": "level", "tenor": None, "wing": None,
            "min_date": "2020-01-02", "units": None, "description": f"metric {j}", "formula": None} for j in range(6)]
    fix = {
        "registry": registry_with_coverage(None), "sweep": sweep, "alone": alone, "expect": expect,
        "files": files, "names": NAMES, "variants": VARIANTS,
        "catalog": {"metrics": cat, "family_groups": [{"label": "IV", "color": "#3498db", "families": ["iv"]}],
                    "other_group": None, "form_labels": {"level": "level"}, "unit_formats": {},
                    "default_unit_format": {"scale": 1, "decimals": 2, "suffix": ""}, "row_auto_bins": None,
                    "built_at": "x"},
        "rankRows": [{"column": f"m{j}", "family": "iv", "tenor": None, "wing": None, "form": "level", "n": 240,
                      "bars": 50, "pearson": 0.3 - j * 0.05, "pearson_p": 1e-6 * (j + 1), "pearson_p_bh": 1e-5 * (j + 1),
                      "spearman": 0.3 - j * 0.05, "spearman_p": 1e-6 * (j + 1), "spearman_p_bh": 1e-5 * (j + 1)}
                     for j in range(6)],
    }
    try:
        page = build_page(fix)
    except Exception as exc:                              # noqa: BLE001
        print(f"  the page could not be rendered: {type(exc).__name__}: {exc}")
        return 1
    tmp = ROOT / "scripts" / "_oo_sweep_ui.html"
    tmp.write_text(page, encoding="utf-8")
    try:
        p = subprocess.run(
            [browser, "--headless=new", "--disable-gpu", "--window-size=1600,1200",
             "--dump-dom", "--virtual-time-budget=60000", tmp.as_uri()],
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=240)
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
            clean = parts[1] == "clean"
            print(("  ok    " if clean else "  FAIL  ") + "console: " + parts[1][:400])
            fails += not clean
            n += 1
            continue
        n += 1
        if parts[0] == "ok":
            print(f"  ok    {parts[1]}")
        else:
            fails += 1
            print(f"  FAIL  {parts[1]}\n          got  {parts[2][:300]}\n          want {parts[3][:300] if len(parts) > 3 else ''}")
    print()
    print(f"OO sweep UI: {n} clicked assertions, failures: {fails}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
