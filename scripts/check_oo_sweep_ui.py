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
  * parameter FILTERS (2026-10-19): a card per parameter, every value
    ticked; a slice (pt=25) narrows everything, the mean-per-variant figures
    average the surviving variants (stats.py), the grid collapses to the
    ticked values, the selector lists only survivors; one survivor drops the
    "mean / variant" tags and brings the BH line back; picking a variant
    greys the cards and keeps the slice for ALL; unticking every value is the
    no-trades state; "reset all filters" clears them
  * THE ORDINARY CATEGORICAL FILTERS STAY LIVE: Day of Week and Exit Reason
    enabled and filtering BY CLICK at ALL, with a variant picked, and on a
    single log -- only sweep-parameter cards ever grey out. (43312b4 disabled
    them for good and 120 assertions passed: every one drove them by calling
    toggleCat, never by clicking.)
  * saving (P4): a multi-file sweep's Save sends EVERY file, gzipped, to
    /strategies/sweep as one strategy; a single file saves as before; a
    saved sweep loads back with ALL selected, and a changed variant list is
    said
  * HEADER BANDS (2026-10-27): on every pivot a column band naming the
    COLUMN parameter over exactly the data columns, and per box ONE row band
    naming the ROW parameter beside exactly the data rows -- names, spans
    and measured positions, in Train and Full view and on a 1-value axis
  * the parameter grid, EVERY PAIR AT ONCE (2026-10-27; it had Across /
    Down selectors): a 2-parameter sweep renders one pair, a 3-parameter
    sweep three -- dte×pt, dte×sl, pt×sl, in the sweep's parameter order --
    a 1-parameter sweep, a single log and one variant none; every cell of
    every grid against stats.py for all six metrics, with and without a
    filter; ONE colour scale across every grid (each cell's shade is its
    |value| against the largest anywhere; no grid normalised to its own
    range); the thin outline per grid (a fixture where per-grid and global
    medians disagree); the order stable across a filter, a variant switch
    and Train / Full; Full view one box per pair holding two pivots
  * grid shading is magnitude, and a cell under half its grid's median cell
    trades is outlined
  * the parameter sections: avg and win rate pool, a value's TOTAL is the
    mean over the variants holding that value
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


def make_csv(seed: int, n: int = 60) -> str:
    """~60 trades through 2024. Each variant its own P/L path, so the mean of
    the variants' drawdowns is NOT the pooled drawdown (the gate needs that)."""
    rng = random.Random(seed)
    days = pd.bdate_range("2024-01-02", "2024-12-20")
    rows = []
    for _ in range(n):
        i = rng.randrange(len(days) - 6)
        d0, d1 = days[i], days[i + rng.randrange(1, 6)]
        pnl = round(rng.gauss(40 if seed % 2 else -10, 380), 2)
        rows.append([d0.date().isoformat(), "10:00:00", "4800", "1 P STO", str(round(-300 - rng.random() * 700, 2)),
                     "4810", d1.date().isoformat(), "15:59:00", "0", rng.choice(["Profit Target", "Stop Loss"]),
                     str(pnl), "10", "1", "100000", "5000", "IC", "0.3", "20"])
    out = io.StringIO()
    pd.DataFrame(rows, columns=HEADER).to_csv(out, index=False)
    return out.getvalue()


VARIANTS3 = [(d, p, sl) for d in (30, 45) for p in (25, 50) for sl in (100, 200)]
NAMES3 = [f"sweep_dte{d}_pt{p}_sl{sl}.csv" for d, p, sl in VARIANTS3]


def figures(df: pd.DataFrame) -> dict:
    """One variant's own figures, by stats.py -- what the page's cells are held to."""
    from app.oo_backtest.stats import calculate_stats
    st = calculate_stats(df)
    closes = pd.to_datetime(df["date_closed"])
    years = (closes.max() - closes.min()).days / 365.25
    dd = st["max_drawdown"]
    return {"n": len(df), "total": float(df["pnl"].sum()), "wins": int((df["pnl"] > 0).sum()), "dd": dd,
            "calmar": (st["total_pnl"] / years) / abs(dd) if dd < 0 and years > 0 else None}


def cell(figs: list[dict]) -> dict:
    """A grid cell from its variants' figures, by the page's rule."""
    n = sum(f["n"] for f in figs)
    cal = [f["calmar"] for f in figs if f["calmar"] is not None]
    return {"n": n, "avg": sum(f["total"] for f in figs) / n, "win": sum(f["wins"] for f in figs) / n * 100,
            "total": sum(f["total"] for f in figs) / len(figs), "maxdd": sum(f["dd"] for f in figs) / len(figs),
            "calmar": sum(cal) / len(cal) if cal else None}


def pair_cells(frames: dict, variants: list, params: list, frame_filter=None) -> dict:
    """{ "x|y": { "xv,yv": cell } } for every pair, by the page's rule, from
    each variant's own frame (optionally filtered)."""
    out = {}
    for i in range(len(params)):
        for j in range(i + 1, len(params)):
            x, y = params[i], params[j]
            grid = {}
            for xv in sorted({v[x] for v, _ in variants}):
                for yv in sorted({v[y] for v, _ in variants}):
                    ns = [n for v, n in variants if v[x] == xv and v[y] == yv]
                    if ns:
                        fr = [frames[n] if frame_filter is None else frame_filter(frames[n]) for n in ns]
                        grid[f"{xv},{yv}"] = cell([figures(f) for f in fr])
            out[f"{x}|{y}"] = grid
    return out


def thin_cells(grids: dict) -> dict:
    """Per grid, the cells under half that grid's OWN median trade count."""
    import statistics
    out = {}
    for k, g in grids.items():
        med = statistics.median([c["n"] for c in g.values() if c["n"]])
        out[k] = sorted(key for key, c in g.items() if c["n"] and c["n"] < 0.5 * med)
    return out


VARIANTSP = [(d, p, sl) for d in (30, 45) for p in (25, 50) for sl in (50, 100, 150, 200)]
NAMESP = [f"pairs_dte{d}_pt{p}_sl{sl}.csv" for d, p, sl in VARIANTSP]
NAMES1 = ["one_dte30.csv", "one_dte45.csv"]


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
    # The 2-parameter grid: each cell is one variant.
    frames = {n: data_loader.parse_upload(files[n].encode(), n) for n in NAMES}
    expect["grid2"] = {f"{d},{p}": cell([figures(frames[n])]) for (d, p), n in zip(VARIANTS, NAMES)}

    # A 3-parameter sweep: cells pool across sl.
    # dte=45 x pt=50 is THIN: 20 trades per variant against 60 -- the grid must outline it.
    files3 = {n: make_csv(20 + i, 20 if VARIANTS3[i][:2] == (45, 50) else 60) for i, n in enumerate(NAMES3)}
    sweep3 = with_sessions(asyncio.run(analyze_batch([(n, c.encode()) for n, c in files3.items()], None)))
    fr3 = {n: data_loader.parse_upload(files3[n].encode(), n) for n in NAMES3}
    no_mon = {n: f[f["date_opened"].dt.dayofweek != 0] for n, f in fr3.items()}
    grid3, grid3_nomon = {}, {}
    for d in (30, 45):
        for p_ in (25, 50):
            ns = [n for (dd, pp, _), n in zip(VARIANTS3, NAMES3) if dd == d and pp == p_]
            grid3[f"{d},{p_}"] = cell([figures(fr3[n]) for n in ns])
            grid3_nomon[f"{d},{p_}"] = cell([figures(no_mon[n]) for n in ns])
    # The dte section: avg and win pool, total is the mean over that value's variants.
    sec = {}
    for d in (30, 45):
        ns = [n for (dd, _, _), n in zip(VARIANTS3, NAMES3) if dd == d]
        c = cell([figures(fr3[n]) for n in ns])
        sec[str(d)] = {"avg": c["avg"], "total": c["total"], "win": c["win"]}
    # Parameter-filter slices: pt=25 (4 variants), and one variant alone.
    pt25 = [n for (_, pp, _), n in zip(VARIANTS3, NAMES3) if pp == 25]
    one = NAMES3[VARIANTS3.index((30, 25, 100))]
    expect.update({"grid3": grid3, "grid3_nomon": grid3_nomon, "sec_dte": sec,
                   "slice_pt25": cell([figures(fr3[n]) for n in pt25]), "slice_one": cell([figures(fr3[one])]),
                   "n3": sum(len(f) for f in fr3.values())})
    # EVERY PAIR: the 3-parameter sweep's three grids, with and without the
    # Monday filter; the parameter-filter slice (pt=25 only).
    v3 = [({"dte": d, "pt": p_, "sl": sl}, n) for (d, p_, sl), n in zip(VARIANTS3, NAMES3)]
    nomon = lambda f: f[f["date_opened"].dt.dayofweek != 0]  # noqa: E731
    expect["pairs3"] = pair_cells(fr3, v3, ["dte", "pt", "sl"])
    expect["pairs3_nomon"] = pair_cells(fr3, v3, ["dte", "pt", "sl"], nomon)
    expect["pairs3_pt25"] = pair_cells(fr3, [x for x in v3 if x[0]["pt"] == 25], ["dte", "pt", "sl"])
    # A sweep where the thin outline per GRID differs from a GLOBAL one: sl
    # has four values, so dte×pt cells pool four variants and the others two;
    # the dte=45 pt=50 variants are thin (25 trades against 60). dte×pt's
    # (45,50) cell, 100 trades, is under half ITS median (240) but not under
    # half the median of every cell on the card (120).
    filesP = {n: make_csv(40 + i, 25 if VARIANTSP[i][:2] == (45, 50) else 60) for i, n in enumerate(NAMESP)}
    sweepP = with_sessions(asyncio.run(analyze_batch([(n, c.encode()) for n, c in filesP.items()], None)))
    frP = {n: data_loader.parse_upload(filesP[n].encode(), n) for n in NAMESP}
    vP = [({"dte": d, "pt": p_, "sl": sl}, n) for (d, p_, sl), n in zip(VARIANTSP, NAMESP)]
    expect["pairsP"] = pair_cells(frP, vP, ["dte", "pt", "sl"])
    expect["thinP"] = thin_cells(expect["pairsP"])
    all_n = sorted(c["n"] for g in expect["pairsP"].values() for c in g.values())
    import statistics
    gmed = statistics.median(all_n)
    expect["thinP_global"] = {k: sorted(key for key, c in g.items() if c["n"] < 0.5 * gmed)
                              for k, g in expect["pairsP"].items()}
    # A ONE-parameter sweep: no pair, no grid.
    files1 = {n: make_csv(70 + i) for i, n in enumerate(NAMES1)}
    sweep1 = with_sessions(asyncio.run(analyze_batch([(n, c.encode()) for n, c in files1.items()], None)))
    files.update(files3)
    files.update(filesP)
    files.update(files1)
    expect["sweepP"], expect["sweep1"] = sweepP, sweep1
    return sweep, alone, expect, files, sweep3


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
  if (/\/strategies\/sweep$/.test(url)) {
    const parts = init.body.getAll('files');
    const heads = [];
    for (const f of parts) { const b = new Uint8Array(await f.slice(0, 2).arrayBuffer()); heads.push(b[0] === 0x1f && b[1] === 0x8b); }
    window.__saved = { names: parts.map(f => f.name), gz: heads, name: init.body.get('name'), single: init.body.get('file') };
    window.__savedList = [{ id: 9, name: init.body.get('name'), notes: '', trade_count: FIX.sweep.n, variants: 4,
                            date_min: FIX.sweep.date_min, date_max: FIX.sweep.date_max, capital_per_position: null,
                            updated_at: '2026-10-18T00:00:00+00:00' }];
    return reply(200, { strategy: window.__savedList[0] });
  }
  if (/\/strategies\/9\/load$/.test(url)) {
    return reply(200, { ...FIX.sweep, saved: window.__savedList[0], parse: { source: 'parse', files: 4 },
                        saved_variants_changed: { when_saved: ['dte=30 · pt=25 (61)'], now: ['dte=30 · pt=25 (60)'] } });
  }
  if (/\/strategies$/.test(url) && init && init.method === 'POST') {
    window.__saved = { single: init.body.get('file') && init.body.get('file').name, name: init.body.get('name') };
    return reply(200, { strategy: { id: 10, name: init.body.get('name'), trade_count: 60 } });
  }
  if (/\/strategies$/.test(url)) return reply(200, { strategies: window.__savedList || [] });
  if (/\/market-status$/.test(url)) return reply(200, { ok: false, error: 'stub' });
  if (/\/parse-batch$/.test(url)) {
    const parts = init.body.getAll('files');
    const heads = [];
    for (const f of parts) { const b = new Uint8Array(await f.slice(0, 2).arrayBuffer()); heads.push(b[0] === 0x1f && b[1] === 0x8b); }
    window.__sent.push({ names: parts.map(f => f.name), gz: heads });
    if (window.__next === 'refuse') return reply(400, { detail: 'sweep_x.csv has dte; the other files have dte, pt. Nothing was loaded.' });
    return reply(200, window.__next === 'alone' ? FIX.alone : window.__next === 'sweep3' ? FIX.sweep3
                     : window.__next === 'sweepP' ? FIX.expect.sweepP : window.__next === 'sweep1' ? FIX.expect.sweep1 : FIX.sweep);
  }
  if (/\/surface\/catalog$/.test(url)) return reply(200, FIX.catalog);
  if (/\/surface\/rank$/.test(url)) {
    const body = JSON.parse(init.body);
    return reply(200, { rows: FIX.rankRows, report: { no_bar: 0, with_bar: body.pnl.length, distinct_entries: body.bars.length },
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
  // An ORDINARY categorical filter card (not a sweep parameter), driven by
  // clicking: enabled, and the click filters. Returns [enabled, narrowed,
  // counted, restored]; the page is left as it was found.
  const regCard = t => [...document.querySelectorAll('.ob-side .ob-card')].find(e =>
    txt(e.querySelector('.ob-card-title')) === t && ![...e.querySelectorAll('.ob-card-sub')].some(x => shown(x) && txt(x) === 'sweep parameter'));
  const clickWorks = async (title) => {
    const c = Alpine.$data(document.body);
    const card = regCard(title);
    const boxes = card ? [...card.querySelectorAll('.ob-checks input')] : [];
    const enabled = boxes.length > 0 && boxes.every(b => !b.disabled) && !card.classList.contains('ob-param-off');
    const before = c.filteredCount, active = c.activeCount();
    // A TICKED box: unticking it must narrow (one the test already unticked would widen).
    const box = boxes.find(x => x.checked);
    if (!box) return [false, false, false, false];
    const wasActive = c.isActive(c.filterDefs().find(m => m.label === title));
    box.click();
    await until(() => c.filteredCount !== before);
    const narrowed = c.filteredCount < before, counted = c.activeCount() === active + (wasActive ? 0 : 1);
    box.click();
    await until(() => c.filteredCount === before);
    return [enabled, narrowed, counted, c.filteredCount === before];
  };
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

    // ── the scatter sample keeps the extremes ───────────────────────────
    {
      let seed = 11; const rnd = () => (seed = (seed * 16807) % 2147483647) / 2147483647;
      const gauss = () => Math.sqrt(-2 * Math.log(rnd() + 1e-12)) * Math.cos(2 * Math.PI * rnd());
      ok('under the cap every point is drawn', obScatterPick(Array.from({ length: 4000 }, gauss)).pick, null);
      const ys = Array.from({ length: 240000 }, () => Math.round(gauss() * 40000) / 100);
      const r = obScatterPick(ys);
      const byY = ys.map((y, i) => [y, i]).sort((a, b) => a[0] - b[0] || a[1] - b[1]);
      const want = new Set([...byY.slice(0, 200), ...byY.slice(-200)].map(p => p[1]));
      const got = new Set(r.pick);
      ok('240,000 points: at most 5,000 drawn, every one of the 200 best and 200 worst among them',
         [r.pick.length <= 5000, r.stride > 1, [...want].every(i => got.has(i))], [true, true, true]);
      // Heavy ties at both edges: each tail is still exactly 200, ties in trade order.
      const tied = Array.from({ length: 240000 }, (_, i) => (i % 1000 === 0 ? 9 : i % 997 === 0 ? -9 : 0));
      const t = obScatterPick(tied);
      // (Ties past the first 200 are ordinary points and may be drawn by the stride too.)
      const tset = new Set(t.pick);
      const firstOf = v => tied.map((y, i) => [y, i]).filter(p => p[0] === v).slice(0, 200).map(p => p[1]);
      ok('ties at the tail edges: the first 200 of the highest and of the lowest, in trade order, all drawn',
         [firstOf(9).every(i => tset.has(i)), firstOf(-9).every(i => tset.has(i)), t.pick.length <= 5000], [true, true, true]);
    }

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
    ok('ALL: Day of Week enabled, and a click filters (narrows, counts, restores)', await clickWorks('Day of Week'), [true, true, true, true]);
    ok('ALL: Exit Reason enabled, and a click filters', await clickWorks('Exit Reason'), [true, true, true, true]);

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

    // ── the parameter grid: EVERY pair (2 parameters -> one) ────────────
    const gridCard = () => [...document.querySelectorAll('.ob-card')].find(e => /Parameter grid/.test(txt(e.querySelector('.ob-card-title'))));
    const setSel = async (id, v) => { const e = document.getElementById(id); e.value = v; e.dispatchEvent(new Event('change', { bubbles: true })); await wait(150); };
    // Every pair box as data: its key, axes, and cells (train pivot; the
    // test pivot too in Full view).
    const readPairs = () => [...gridCard().querySelectorAll('.ob-pgrid-pair')].map(box => {
      const tbl = t => !t ? null : {
        xs: [...t.querySelectorAll('thead tr:not(.ob-pgrid-bandrow) th')].map(txt).filter(Boolean),
        ys: [...t.querySelectorAll('tbody th:not(.ob-pgrid-band)')].map(txt),
        cells: [...t.querySelectorAll('tbody tr')].map(tr => [...tr.querySelectorAll('td')].map(td => ({
          text: txt(td.querySelector('.ob-pgrid-v')), thin: td.classList.contains('thin'),
          bg: getComputedStyle(td).backgroundColor, inl: td.style.background }))) };
      return { key: box.dataset.pair.replace('|', '×'), title: txt(box.querySelector('.ob-pgrid-pair-title')),
               train: tbl(box.querySelector('.ob-pgrid-wrap > table.ob-pgrid')), test: tbl(box.querySelector('.ob-pgrid-test table.ob-pgrid')),
               note: txt(box.querySelector('.ob-pgrid-pair-note')) };
    });
    const FMT = { avg: v => money(v), total: v => money(v), maxdd: v => money(v), win: v => parseFloat(v), calmar: v => parseFloat(v),
                  n: v => Number(String(v).replace(/,/g, '')) };
    const TOL = { avg: 0.006, total: 0.51, maxdd: 0.51, win: 0.051, calmar: 0.0051, n: 0 };
    // The header bands of every box, against its axes: the column band names
    // the COLUMN parameter and spans exactly the data columns (colspan AND
    // measured edges), the row band names the ROW parameter and spans exactly
    // the data rows; ONE row band per box. Returns the problems found.
    const bandProblems = () => {
      const bad = [];
      const near = (a, b) => Math.abs(a - b) <= 1.5;
      for (const box of gridCard().querySelectorAll('.ob-pgrid-pair')) {
        const [x, y] = box.dataset.pair.split('|');
        const rowBands = box.querySelectorAll('th.ob-pgrid-band-row');
        if (rowBands.length !== 1) bad.push(`${x}×${y}: ${rowBands.length} row bands`);
        const tables = [box.querySelector('.ob-pgrid-wrap > table.ob-pgrid'), box.querySelector('.ob-pgrid-test table.ob-pgrid')].filter(Boolean);
        tables.forEach((t, ti) => {
          const which = ti ? 'test' : 'train';
          const cb = t.querySelector('th.ob-pgrid-band-col');
          const heads = [...t.querySelectorAll('thead tr:not(.ob-pgrid-bandrow) th')].filter(th => txt(th));
          if (!cb) { bad.push(`${x}×${y} ${which}: no column band`); return; }
          if (txt(cb) !== x) bad.push(`${x}×${y} ${which}: column band says "${txt(cb)}", the columns are ${x}`);
          if (cb.colSpan !== heads.length) bad.push(`${x}×${y} ${which}: column band spans ${cb.colSpan} of ${heads.length} columns`);
          const r = cb.getBoundingClientRect(), a = heads[0].getBoundingClientRect(), z = heads[heads.length - 1].getBoundingClientRect();
          if (!near(r.left, a.left) || !near(r.right, z.right)) bad.push(`${x}×${y} ${which}: column band ${r.left}-${r.right} vs columns ${a.left}-${z.right}`);
        });
        const rb = rowBands[0], t0 = tables[0];
        if (rb) {
          const rows = [...t0.querySelectorAll('tbody tr')];
          if (txt(rb) !== y) bad.push(`${x}×${y}: row band says "${txt(rb)}", the rows are ${y}`);
          if (rb.rowSpan !== rows.length) bad.push(`${x}×${y}: row band spans ${rb.rowSpan} of ${rows.length} rows`);
          const r = rb.getBoundingClientRect(), a = rows[0].getBoundingClientRect(), z = rows[rows.length - 1].getBoundingClientRect();
          if (!near(r.top, a.top) || !near(r.bottom, z.bottom)) bad.push(`${x}×${y}: row band ${r.top}-${r.bottom} vs rows ${a.top}-${z.bottom}`);
          // In Full view the test pivot's rows sit level with the band's.
          if (tables[1]) {
            const tr = [...tables[1].querySelectorAll('tbody tr')];
            if (tr.length !== rows.length || !near(tr[0].getBoundingClientRect().top, a.top) || !near(tr[tr.length - 1].getBoundingClientRect().bottom, z.bottom))
              bad.push(`${x}×${y}: the test pivot's rows are not level with the row band`);
          }
        }
      }
      return bad;
    };
    const alphaOf = bg => { const m = bg.match(/rgba?\(([^)]+)\)/); const p = m[1].split(','); return p.length === 4 ? +p[3] : 1; };
    // Every cell of every pair against the reference, for all six metrics.
    const checkPairs = async (E, label) => {
      for (const mt of ['avg', 'total', 'win', 'maxdd', 'calmar', 'n']) {
        await setSel('ob-grid-m', mt);
        let worst = 0, cells = 0, missing = [];
        for (const pr of readPairs()) {
          const G = E[pr.key.replace('×', '|')] || E[pr.key];
          if (!G) { missing.push(pr.key); continue; }
          pr.train.ys.forEach((yv, r) => pr.train.xs.forEach((xv, k) => {
            const want = G[xv + ',' + yv];
            if (!want) return;
            cells++;
            worst = Math.max(worst, want[mt] === null ? (pr.train.cells[r][k].text === '—' ? 0 : 1e9)
                                                      : Math.abs(FMT[mt](pr.train.cells[r][k].text) - want[mt]));
          }));
        }
        out.push((worst <= TOL[mt] && !missing.length ? 'ok|' : 'FAIL|') +
                 `${label} ${mt}: every cell of every grid = stats.py on its variants (${cells} cells)|worst diff ${worst} ${missing}|<= ${TOL[mt]}`);
      }
      await setSel('ob-grid-m', 'avg');
    };
    ok('the grid is on screen at ALL', await until(() => shown(gridCard()) && c.grids.length > 0), true);
    ok('2 parameters: ONE pair -- dte × pt, dte across, pt down -- and no Across / Down selectors',
       [readPairs().map(p => [p.key, p.title, p.train.xs, p.train.ys]), !!document.getElementById('ob-grid-x'),
        !!document.getElementById('ob-grid-y'), document.getElementById('ob-grid-m').value],
       [[['dte×pt', 'dte × pt', ['30', '45'], ['25', '50']]], false, false, 'avg']);
    await checkPairs({ 'dte×pt': FIX.expect.grid2 }, '2-param');
    ok('the box says each cell is one variant', /each cell is one variant/.test(readPairs()[0].note), true);
    const bgs = [...gridCard().querySelectorAll('.ob-pgrid td')].map(td => getComputedStyle(td).backgroundColor);
    ok('cells shade blue (profit) or pink (loss): ' + bgs.join(' '),
       bgs.every(b => /^rgba?\((52, 152, 219|232, 67, 147)(, [\d.]+)?\)$/.test(b)), true);

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
    ok('one variant: Day of Week still enabled, and a click still filters', await clickWorks('Day of Week'), [true, true, true, true]);
    ok('one variant: Exit Reason still enabled, and a click still filters', await clickWorks('Exit Reason'), [true, true, true, true]);
    ok('one variant: no parameter grid, no parameter sections',
       [await until(() => !shown(gridCard())), c.paramSections().length, !!document.getElementById('ob-sec-card-param__dte')], [true, 0, false]);
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
    ok('a single log: no parameter grid, no pair box', [shown(gridCard()), c.grids.length,
       document.querySelectorAll('.ob-pgrid-pair').length], [false, 0, 0]);
    ok('a single log: Day of Week enabled, and a click filters', await clickWorks('Day of Week'), [true, true, true, true]);
    const alone = snapshot(c);
    ok('the comparison has something to compare: sections with subtitles, a deployment title',
       [alone.subs.length >= 3, !!alone.deploy, Object.values(alone.stats).every(v => v !== '')], [true, true, true]);
    for (const k of Object.keys(alone.stats)) ok(`variant view == its file alone: ${k}`, asVariant.stats[k], alone.stats[k]);
    ok('variant view == its file alone: status line', asVariant.status, alone.status);
    ok('variant view == its file alone: every section subtitle', asVariant.subs, alone.subs);
    ok('variant view == its file alone: date bounds, auto-bin steps, drawdown and deployment titles',
       [asVariant.bounds, asVariant.steps, asVariant.dd, asVariant.deploy], [alone.bounds, alone.steps, alone.dd, alone.deploy]);
    ok('a single uploaded file can be saved', await until(() => shown(document.querySelector('.ob-save'))), true);
    ok('labelled as a log, not a sweep', txt(document.querySelector('.ob-save .ob-lbl')), 'Save this log');

    // ── a refused batch leaves the page alone ───────────────────────────
    window.__next = 'sweep';
    drop(FIX.names);
    await until(() => !!c.sweep);
    // ── saving a sweep: every file, gzipped, one strategy ───────────────
    ok('a multi-file sweep offers Save, labelled for the sweep',
       [await until(() => shown(document.querySelector('.ob-save'))),
        txt(document.querySelector('.ob-save .ob-lbl'))], [true, 'Save this sweep (every file, one strategy)']);
    const nameBox = document.querySelector('.ob-save input[type=text]');
    nameBox.value = 'dte sweep'; nameBox.dispatchEvent(new Event('input', { bubbles: true }));
    await wait(50);
    [...document.querySelectorAll('.ob-save button')].find(b => txt(b) === 'Save').click();
    await until(() => !!window.__saved);
    ok('Save sends every file, gzipped, with the name, to /strategies/sweep',
       [window.__saved.names, window.__saved.gz, window.__saved.name, window.__saved.single], [FIX.names, [true, true, true, true], 'dte sweep', null]);
    ok('and says it was saved, as a sweep', await until(() => /Saved as dte sweep \(240 trades, sweep of 4 variants/.test(c.saveMsg)), true);
    // ── loading a saved sweep: ALL, the record, a changed list said ─────
    c.setVariant('2'); await wait(150);
    c.selectedSavedId = '9';
    await c.loadSelected();
    await until(() => c.meta && c.meta.saved && c.meta.saved.id === 9);
    await wait(200);
    ok('a saved sweep loads with ALL selected and the selector back', [c.variantSel, c.sweep && c.sweep.variants.length, shown(sel)], ['all', 4, true]);
    const notes = [...document.querySelectorAll('.ob-loaded .ob-parse-note')].map(txt);
    ok('the sidebar says it is a saved sweep', notes.some(t => /^Saved parameter sweep, 4 variants/.test(t)), true);
    ok('and that its files now resolve to other variants than when saved',
       notes.some(t => /now resolve to other variants than when this sweep was saved/.test(t)), true);
    ok('a loaded saved sweep offers no Save (it is saved)', shown(document.querySelector('.ob-save')), false);
    window.__next = 'refuse';
    drop(['sweep_x.csv'].concat(FIX.names.slice(1)).map(n => FIX.files[n] ? n : FIX.names[0]));
    await until(() => !!c.uploadError);
    const upErr = document.querySelector('.ob-drop').parentElement.querySelector(':scope > .ob-err');
    ok('a refused batch says why, under the drop zone', /Nothing was loaded/.test(c.uploadError) &&
       await until(() => shown(upErr) && /Nothing was loaded/.test(txt(upErr))), true);
    ok('and the loaded sweep is untouched', [c.sweep && c.sweep.variants.length, c.viewN], [4, FIX.sweep.n]);
    // The refusal is logged on purpose; only other errors count.
    window.__errs = window.__errs.filter(e => !/oo-backtest upload/.test(e));

    // ── three parameters: cells pool across the hidden one ──────────────
    window.__next = 'sweep3';
    drop(FIX.names3);
    await until(() => c.sweep && c.sweep.params.length === 3 && c.grids.length === 3);
    await wait(300);
    const PAIRS3 = ['dte×pt', 'dte×sl', 'pt×sl'];
    ok('3 parameters: THREE pairs, in the sweep\'s parameter order', [c.sweep.variants.length, c.sweep.params,
       readPairs().map(p => [p.key, p.title, p.train.xs, p.train.ys])],
       [8, ['dte', 'pt', 'sl'], [['dte×pt', 'dte × pt', ['30', '45'], ['25', '50']], ['dte×sl', 'dte × sl', ['30', '45'], ['100', '200']],
                                ['pt×sl', 'pt × sl', ['25', '50'], ['100', '200']]]]);
    ok('header bands, every box: the column band names the column parameter over exactly its columns, ONE row band ' +
       'names the row parameter beside exactly its rows', bandProblems(), []);
    ok('each box says what its cells pool across', readPairs().map(p => p.note.split(' · ')[0]),
       ['each cell pools across sl (2 values)', 'each cell pools across pt (2 values)', 'each cell pools across dte (2 values)']);
    await checkPairs(FIX.expect.pairs3, '3-param');
    // ONE colour scale across every grid: each cell's alpha is ITS |value|
    // against the largest |value| on the whole card.
    {
      const E = FIX.expect.pairs3;
      const maxAbs = Math.max(...Object.values(E).flatMap(g => Object.values(g).map(x => Math.abs(x.avg))));
      let worst = 0, perGridMax = [], worstAt = '', domWorst = 0;
      for (const pr of readPairs()) {
        let gmax = 0;
        pr.train.ys.forEach((yv, r) => pr.train.xs.forEach((xv, k) => {
          const want = OB_ALPHA_FLOOR + (1 - OB_ALPHA_FLOOR) * Math.abs(E[pr.key.replace('×', '|')][xv + ',' + yv].avg) / maxAbs;
          // The alpha the page COMPUTED (its cell's bg string, 3 decimals):
          // the browser quantises any read-back to 8 bits (0.587 -> "0.59").
          const gi = c.grids.findIndex(x => x.key === pr.key.replace('×', '|'));
          const a = alphaOf(c.grids[gi].train.rows[r][k].bg);
          // ...and what is ON SCREEN is that colour, to the browser's precision.
          domWorst = Math.max(domWorst, Math.abs(alphaOf(pr.train.cells[r][k].bg) - a));
          if (Math.abs(a - want) > worst) worstAt = `${pr.key} ${xv},${yv}: set ${a}, want ${want.toFixed(4)}`;
          worst = Math.max(worst, Math.abs(a - want)); gmax = Math.max(gmax, a);
        }));
        perGridMax.push(gmax);
      }
      out.push((worst < 0.0006 ? 'ok|' : 'FAIL|') + 'ONE scale: every cell\'s shade is its abs(value) against the largest abs(value) on the card|worst alpha diff ' + worst + ' at ' + worstAt + '|< 0.0006 (3-decimal rounding)');
      out.push((domWorst < 0.006 ? 'ok|' : 'FAIL|') + 'and each cell on screen wears that shade|worst diff ' + domWorst + '|< 0.006 (8-bit alpha)');
      ok('no grid is normalised to its own range: only the grid holding the card\'s largest value reaches full strength',
         perGridMax.filter(a => a >= 0.999).length, 1);
    }
    // The dte×pt (45, 50) cell is thin in its own grid.
    ok('the thin outline: dte×pt (45, 50), 40 trades against its grid\'s median 120', readPairs().map(p =>
       p.train.cells.flatMap((row, r) => row.map((cl, k) => cl.thin ? p.train.xs[k] + ',' + p.train.ys[r] : null)).filter(Boolean)),
       [['45,50'], [], []]);
    // The dte section: pooled avg and win, total = mean over that value's 4 variants.
    const chartData = which => OB_CHARTS.sec[`ob-sec-param__dte-${which}`].data.datasets[0].data;
    ok('a section per parameter, in order, before the metric sections', [...document.querySelectorAll('[id^="ob-sec-card-param__"]')]
       .filter(shown).map(e => e.id), ['ob-sec-card-param__dte', 'ob-sec-card-param__pt', 'ob-sec-card-param__sl']);
    const SD = FIX.expect.sec_dte;
    near('dte section, Avg P/L at 45 pools its trades', chartData('avg')[1], SD['45'].avg, 1e-6);
    near('dte section, Total P/L at 45 is the mean of its 4 variants', chartData('total')[1], SD['45'].total, 1e-6);
    near('dte section, win rate at 30 pools', chartData('win')[0], SD['30'].win, 1e-6);
    ok('no scatter in a parameter section', shown(document.getElementById('ob-sec-param__dte-scatter')), false);
    // A filter: every cell follows it.
    const dow3 = c.registry.find(m => m.key === 'day_of_week');
    const order0 = readPairs().map(p => p.key);       // BEFORE the filter
    c.toggleCat(dow3, 0);
    await until(() => c.activeCount() === 1);
    await wait(300);
    await checkPairs(FIX.expect.pairs3_nomon, 'filtered (no Mondays)');
    ok('the pair order does not move with a filter', readPairs().map(p => p.key), order0);

    // ── parameter filters ───────────────────────────────────────────────
    c.setAllCats(dow3, true);
    await until(() => c.activeCount() === 0);
    await wait(200);
    const pcard = p => [...document.querySelectorAll('.ob-side .ob-card')].find(e =>
      txt(e.querySelector('.ob-card-title')) === p && /sweep parameter/.test(txt(e.querySelector('.ob-card-hdr'))));
    const pbox = (p, v) => [...pcard(p).querySelectorAll('.ob-check')].find(l => txt(l.querySelector('span')) === String(v))
      .querySelector('input');
    const isParamCard = e => [...e.querySelectorAll('.ob-card-sub')].some(x => shown(x) && txt(x) === 'sweep parameter');
    const filterCards = [...document.querySelectorAll('.ob-side .ob-card')].filter(e => e.querySelector('.ob-checks, .ob-dual'));
    ok('a filter card per parameter, first in the panel',
       [filterCards.filter(isParamCard).map(e => txt(e.querySelector('.ob-card-title'))),
        filterCards.slice(0, 3).every(isParamCard)], [['dte', 'pt', 'sl'], true]);
    ok('values with trade counts, every one ticked', [...pcard('pt').querySelectorAll('.ob-check')].map(l =>
       [txt(l.querySelectorAll('span')[0]), l.querySelector('input').checked]), [['25', true], ['50', true]]);
    pbox('pt', 50).click();
    await until(() => c.activeCount() === 1 && c.filteredCount === FIX.expect.slice_pt25.n);
    await wait(300);
    const SP = FIX.expect.slice_pt25, EN = FIX.expect.n3;
    ok('unticking pt=50 narrows the page to the pt=25 trades', [boxVal('# Trades'), txt(document.querySelector('.ob-status span'))],
       [SP.n.toLocaleString(), `Showing ${SP.n.toLocaleString()} of ${EN.toLocaleString()} trades`]);
    near('Total P/L averages the 4 SURVIVING variants (stats.py)', money(boxVal('Total P/L')), SP.total, 0.51);
    near('Max DD averages the 4 surviving variants (stats.py)', money(boxVal('Max DD')), SP.maxdd, 0.51);
    ok('it counts as an active filter', /1 filter active/.test(txt(document.querySelector('.ob-status'))), true);
    ok('the selector lists only the surviving variants', [...sel.options].map(o => txt(o).replace(/ \(\d+ trades\)$/, '')),
       ['ALL — 4 of 8 variants (parameter filters)', 'dte=30 · pt=25 · sl=100', 'dte=30 · pt=25 · sl=200',
        'dte=45 · pt=25 · sl=100', 'dte=45 · pt=25 · sl=200']);
    ok('header bands on a ONE-value axis: the bands span the single row / column', bandProblems(), []);
    ok('THE SINGLE-VALUE CASE (one pt ticked, dte and sl still varying): every grid keeps its place -- dte×pt one row, pt×sl ' +
       'one column -- never a missing one',
       readPairs().map(p => [p.key, p.train.xs, p.train.ys]),
       [['dte×pt', ['30', '45'], ['25']], ['dte×sl', ['30', '45'], ['100', '200']], ['pt×sl', ['25'], ['100', '200']]]);
    ok('and dte×sl, which pools across pt, counts its 1 ticked value', /pools across pt \(1 value\)/.test(readPairs()[1].note), true);
    await checkPairs(FIX.expect.pairs3_pt25, 'pt=25 slice');
    // Down to ONE surviving variant: the page is single-variant in effect.
    pbox('dte', 45).click(); await wait(100);
    pbox('sl', 200).click();
    await until(() => c.filteredCount === FIX.expect.slice_one.n);
    await wait(300);
    const SO = FIX.expect.slice_one;
    ok('one survivor: its own trades', boxVal('# Trades'), SO.n.toLocaleString());
    near('one survivor: Total P/L is its own', money(boxVal('Total P/L')), SO.total, 0.51);
    ok('one survivor: no "mean / variant" tags', LABELS.filter(l => boxTag(l)), []);
    await c.rankSurface();
    await until(() => !c.surf.busy && !!c.surf.result);
    ok('one survivor: the ranking marks significance again (BH in the footer, no pooled note)',
       [/Benjamini/.test(c.surfFooter()), await until(() => !shown(document.querySelector('.ob-rank-pooled')))], [true, true]);
    // Picking a variant greys the cards; ALL brings the slice back.
    sel.value = String(c.sliceVariants()[0].id);
    sel.dispatchEvent(new Event('change', { bubbles: true }));
    await until(() => c.variantSel !== 'all');
    await wait(300);
    ok('a picked variant greys the parameter cards and says why',
       [pbox('pt', 25).disabled, shown(pcard('pt').querySelector('.ob-param-off-note')), c.activeCount()], [true, true, 0]);
    const greyed = [...document.querySelectorAll('.ob-side .ob-card')].filter(e =>
      e.classList.contains('ob-param-off') || [...e.querySelectorAll('.ob-checks input')].some(b => b.disabled));
    ok('ONLY the sweep-parameter cards grey out', greyed.map(e => txt(e.querySelector('.ob-card-title'))), ['dte', 'pt', 'sl']);
    ok('and with them greyed, Day of Week still filters by click', await clickWorks('Day of Week'), [true, true, true, true]);
    sel.value = 'all';
    sel.dispatchEvent(new Event('change', { bubbles: true }));
    await until(() => c.variantSel === 'all');
    await wait(300);
    ok('back at ALL the slice is as it was', [c.catOf(c.paramRows[1]).selected, boxVal('# Trades'), pbox('pt', 25).disabled],
       [[25], SO.n.toLocaleString(), false]);
    ok('ALL with a parameter slice: Exit Reason enabled and filtering by click', await clickWorks('Exit Reason'), [true, true, true, true]);
    // Every value of one parameter unticked: the no-trades state.
    pbox('pt', 25).click();
    await until(() => c.filteredCount === 0);
    const noTrades = [...document.querySelectorAll('.ob-main .ob-empty')].find(e => /No trades match/.test(txt(e)));
    ok('every pt value unticked: the ordinary no-trades state', [await until(() => shown(noTrades)), sel.options[0].textContent.trim()],
       [true, 'ALL — 0 of 8 variants (parameter filters)']);
    // "reset all filters" clears them.
    [...document.querySelectorAll('.ob-status a')].find(a => /reset all filters/.test(txt(a))).click();
    await until(() => c.filteredCount === EN);
    await wait(200);
    ok('reset all filters ticks every parameter value again', [['dte', 30], ['dte', 45], ['pt', 25], ['pt', 50], ['sl', 100], ['sl', 200]]
       .map(([p, v]) => pbox(p, v).checked).concat([sel.options[0].textContent.trim()]),
       [true, true, true, true, true, true, 'ALL — 8 variants']);

    // ── the order is stable: a variant and back, Train / Full ───────────
    const keys = () => readPairs().map(p => p.key);
    sel.value = '3'; sel.dispatchEvent(new Event('change', { bubbles: true }));
    await until(() => c.variantSel === '3'); await wait(300);
    ok('one variant picked: no pair boxes', [c.grids.length, document.querySelectorAll('.ob-pgrid-pair').length], [0, 0]);
    sel.value = 'all'; sel.dispatchEvent(new Event('change', { bubbles: true }));
    await until(() => c.variantSel === 'all' && c.grids.length === 3); await wait(300);
    ok('back to ALL: the same three pairs in the same order', keys(), ['dte×pt', 'dte×sl', 'pt×sl']);
    const tt = document.querySelector('[data-gate-exclude="train-test-card"]');
    const segB = l => [...tt.querySelectorAll('.seg-btn')].find(b => txt(b) === l);
    segB('On').click(); await wait(400);
    ok('Train view: the same three pairs, one pivot each', [keys(), readPairs().map(p => !!p.test)],
       [['dte×pt', 'dte×sl', 'pt×sl'], [false, false, false]]);
    segB('Full').click(); await wait(400);
    ok('Full view: ONE box per pair (3, not 6), each holding the train and test pivots, in the same order',
       [document.querySelectorAll('.ob-pgrid-pair').length, keys(), readPairs().map(p => !!p.train && !!p.test),
        readPairs().map(p => p.test ? p.test.cells.length === p.train.cells.length && p.test.xs.join() === p.train.xs.join() : false)],
       [3, ['dte×pt', 'dte×sl', 'pt×sl'], [true, true, true], [true, true, true]]);
    ok('Full view header bands: a column band on EACH pivot, ONE row band per box (not per pivot), the test rows level with it',
       bandProblems(), []);
    segB('Off').click(); await wait(400);
    ok('Split Off: the same three pairs again', keys(), ['dte×pt', 'dte×sl', 'pt×sl']);

    // ── thin outlines are PER GRID ───────────────────────────────────────
    // sl has four values here, so dte×pt cells pool four variants and the
    // other grids two: the grids' medians differ, and per-grid and global
    // thresholds outline different cells.
    window.__next = 'sweepP';
    drop(FIX.namesP);
    await until(() => c.sweep && c.sweep.variants.length === 16 && readPairs().length === 3
                      && readPairs()[1].train.ys.length === 4, 8000); await wait(400);
    const EP = FIX.expect.pairsP;
    ok('16 variants: three pairs, sl with four values', readPairs().map(p => [p.key, p.train.xs, p.train.ys]),
       [['dte×pt', ['30', '45'], ['25', '50']], ['dte×sl', ['30', '45'], ['50', '100', '150', '200']],
        ['pt×sl', ['25', '50'], ['50', '100', '150', '200']]]);
    await checkPairs(EP, '16-variant');
    ok('header bands on a four-value axis', bandProblems(), []);
    const thinNow = Object.fromEntries(readPairs().map(p => [p.key.replace('×', '|'),
      p.train.cells.flatMap((row, r) => row.map((cl, k) => cl.thin ? p.train.xs[k] + ',' + p.train.ys[r] : null)).filter(Boolean).sort()]));
    ok('the fixture separates the rules: per-grid and global thresholds outline different cells',
       JSON.stringify(FIX.expect.thinP) !== JSON.stringify(FIX.expect.thinP_global), true);
    ok('thin outlines are per grid, each against its OWN median: ' + JSON.stringify(thinNow), thinNow, FIX.expect.thinP);

    // ── one parameter: no pair, no grid ──────────────────────────────────
    window.__next = 'sweep1';
    drop(FIX.names1);
    await until(() => c.sweep && c.sweep.params.length === 1); await wait(400);
    ok('a ONE-parameter sweep: no grid card, no pair box', [shown(gridCard()), c.grids.length,
       document.querySelectorAll('.ob-pgrid-pair').length], [false, 0, 0]);
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
    sweep, alone, expect, files, sweep3 = payloads()
    cat = [{"column_name": f"m{j}", "family": "iv", "form": "level", "tenor": None, "wing": None,
            "min_date": "2020-01-02", "units": None, "description": f"metric {j}", "formula": None} for j in range(6)]
    fix = {
        "registry": registry_with_coverage(None), "sweep": sweep, "alone": alone, "expect": expect, "sweep3": sweep3,
        "names3": NAMES3, "namesP": NAMESP, "names1": NAMES1,
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
