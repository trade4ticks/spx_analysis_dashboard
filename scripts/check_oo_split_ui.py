"""The train/test split on the OO/Mesosim page, BY CLICKING (P1).

Headless Edge, the real template and bundle, fetch stubbed with payloads the
REAL router built (analyze_batch, no database). Every control is driven the
way a person drives it: the Off/On and Train/Full buttons clicked, the
cutoff typed into its date field, the Day of Week box clicked, the date
range typed, Export and Rank metrics clicked.

THE CENTRAL CHECK IS A DIFFERENTIAL, not a search for numbers. The same log
is loaded twice: as it is (A), and with every TEST-period trade's P/L, P/L %
and exit reason changed (B). With Split On and view Train, A and B must
render IDENTICALLY -- every element, its text, its title, every chart's data,
every computed figure, the Export CSV text and the /surface/rank request --
because nothing about the test period may reach the page. A search for
"the test total" finds only the figures someone thought of; a differential
finds any of them. The same change must SHOW in Full view, or the
differential proved nothing. A third load (C) appends copies of test trades,
so the test COUNT changes: Train view must not move except the Data source
card's description of the file (the log's own size), which is excluded from
that comparison and said so.

The Train vs Test panel (P2): absent from the DOM in Train view and with
Split Off; in Full view every figure of its four rows (train / test x
unfiltered / filtered) equals pandas + stats.py on those trades -- at ALL
over a sweep by the Summary's rule, the scale-dependent figures as the mean
of each variant's own, with each row's variant count shown -- and the lift
row follows the brief's formulas; the drawdown note always, the per-variant
note only when several variants are pooled; a period with no trades says so.
THE CACHE of the two unfiltered rows: a clicked metric filter and a view
switch do NOT rebuild them; a typed cutoff, a picked variant and a typed
date range DO, and each time the rows equal pandas on the new universe.

P3: the dashed cutoff line is drawn on BOTH performance charts in Full view
-- at the cutoff's pixel, labelled "entry cutoff <date>", #8a8a8a, 1px,
4 4 dash -- and on neither in Train view or with Split Off. A sweep's grid in
Full view is two pivots: train on the left, test on the right, each cell
against pandas on that variant's filtered trades of that period; the test
pivot repeats no row labels; one shading scale across both. Train view: one
pivot, the train trades, no test pivot in the DOM.

Also, by click and against stats.py / an independent cutoff:
  * Split Off: the card holds only the Off/On control. On -> Train -> Full ->
    Off returns the page to exactly its pre-split state
  * the default cutoff is the entry date nearest 70% of the log's entry span
  * Train view's figures are stats.py on the train trades (entry <= cutoff),
    with and without a clicked filter; Full view's figures are the Off page's
  * train + test = the filtered total, at several typed cutoffs, each count
    against pandas; the sidebar shows no test count in Train view
  * moving the date range or the variant does not move the cutoff
  * the ranking always sends the filtered TRAIN trades, in both views, and
    says "ranked on train only"; switching view does not stale it
  * Export CSV writes what the view shows
  * edge cases: cutoff before the first trade, after the last, outside the
    date range, a six-trade log -- each a legible state; no NaN, undefined
    or Infinity anywhere on any screen
  * a sweep: the same differential at ALL and with one variant picked
  * no console errors (an Alpine expression error leaves a control inert)

SAVING (P4), clicked, with the save / load endpoints stubbed and recording
what the page sends: a saved strategy loads with its split On and its
cutoff, in TRAIN; switching view is not an unsaved change; typing a cutoff
shows "unsaved" and sends NOTHING until the card's Save is clicked, which
sends exactly the split; a strategy saved before the split existed loads
Off, and turning it on gives the 70% default; a strategy loaded while the
page is in Full view opens in Train; an upload's Save carries split_on and
the cutoff.

SPEED (2026-10-25). Each log's browser loads (A, B, C) run in parallel,
three at a time -- the flake that kept this serial was traced to the
requestAnimationFrame shim, not to concurrency. A PLANTED-FAULT run can be
scoped to the browser work and the assertions that should catch it:

  --log SUBSTR    only the logs whose label contains it
  --runs A,B,C    only these loads (B: test trades changed; C: doubled)
  --grep REGEX    only the assertions whose message matches

The driver and console-error checks always count. A scope that matches NO
assertion is a failure, never a quiet "not caught". A scoped run is for
proving one assertion catches one fault; the clean full run is what says the
code is right.

No browser (the VPS) -> SKIP.
"""
from __future__ import annotations

import asyncio
import csv
import html as _html
import io
import json
import re
import subprocess
import sys
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import pandas as pd  # noqa: E402

import check_oo_notes_ui as N  # noqa: E402  (page rendering + browser lookup)
import check_oo_offstate as O  # noqa: E402  (the shared DOM / chart / state snapshot)
import check_oo_sweep_ui as S  # noqa: E402  (the CSV fixture maker)

EXIT_SKIPPED = 3
FAILS: list[str] = []
N_OK = [0]
SCOPE = {"log": None, "runs": {"A", "B", "C"}, "grep": None}
MATCHED = [0]
ALWAYS = ("the driver ran to the end", "no console errors")


def check(ok: bool, msg: str, detail: str = "") -> None:
    if SCOPE["grep"] and not msg.startswith(ALWAYS) and not re.search(SCOPE["grep"], msg):
        return
    if not msg.startswith(ALWAYS):
        MATCHED[0] += 1
    print(("  ok    " if ok else "  FAIL  ") + msg + ("" if ok or not detail else f"\n          {detail[:600]}"))
    if ok:
        N_OK[0] += 1
    else:
        FAILS.append(msg)


# ── fixtures ────────────────────────────────────────────────────────────────

def ref_cutoff(dates: list[str]) -> str:
    """The default cutoff, written independently of the page: the entry date
    nearest 70% of the calendar span, ties to the earlier date."""
    ds = sorted(set(d for d in dates if d))
    lo, hi = date.fromisoformat(ds[0]), date.fromisoformat(ds[-1])
    target = lo.toordinal() + 0.7 * (hi.toordinal() - lo.toordinal())
    return min(ds, key=lambda d: (abs(date.fromisoformat(d).toordinal() - target), d))


def rows_of(text: str) -> tuple[list[str], list[list[str]]]:
    r = list(csv.reader(io.StringIO(text)))
    return r[0], r[1:]


def write(header, rows) -> str:
    out = io.StringIO()
    csv.writer(out, lineterminator="\n").writerows([header, *rows])
    return out.getvalue()


def alter_test(text: str, cutoff: str) -> str:
    """B: every trade entered after the cutoff gets a different P/L, P/L % and
    exit reason. Dates, premium and every market input stay, so nothing the
    page takes from the whole log (bin edges, slider extents, the option
    LISTS) can change -- only test outcomes."""
    h, rows = rows_of(text)
    i_d, i_p, i_pp, i_r = h.index("Date Opened"), h.index("P/L"), h.index("P/L %"), h.index("Reason For Close")
    for r in rows:
        if r[i_d] > cutoff:
            r[i_p] = str(round(-1.7 * float(r[i_p]) + 13.0, 2))
            r[i_pp] = "-37"
            r[i_r] = "Stop Loss" if r[i_r] == "Profit Target" else "Profit Target"
    return write(h, rows)


def duplicate_test(text: str, cutoff: str) -> str:
    """C: every test trade twice -- the test COUNT changes, nothing else."""
    h, rows = rows_of(text)
    i_d = h.index("Date Opened")
    return write(h, rows + [list(r) for r in rows if r[i_d] > cutoff])


def payload(files: dict[str, str]) -> dict:
    import logging

    from app.oo_backtest import market_calendar
    from app.routers.oo_backtest import analyze_batch
    logging.getLogger("app.routers.oo_backtest").setLevel(logging.CRITICAL)   # no pool: "not joined" is expected
    p = asyncio.run(analyze_batch([(n, c.encode()) for n, c in files.items()], None))
    p["market"]["spx_sessions"] = market_calendar.sessions(p["date_min"], p["date_max"])
    return p


def frame(files: dict[str, str]) -> pd.DataFrame:
    from app.oo_backtest import data_loader
    parts = []
    for n, c in files.items():
        d = data_loader.parse_upload(c.encode(), n)
        d["vfile"] = n                     # one file, one variant
        parts.append(d)
    return pd.concat(parts, ignore_index=True)


def one_variant(df: pd.DataFrame) -> dict:
    """A variant's own figures, by stats.py; Calmar as the page defines it
    (avg annual P/L over |Max DD|, years close-to-close)."""
    from app.oo_backtest.stats import calculate_stats
    st = calculate_stats(df)
    closes = pd.to_datetime(df["date_closed"])
    years = (closes.max() - closes.min()).days / 365.25
    dd = float(st["max_drawdown"])
    total = float(df["pnl"].sum())
    return {"total": total, "dd": dd, "calmar": (total / years) / abs(dd) if dd < 0 and years > 0 else None}


def panel_ref(df: pd.DataFrame, per_variant: bool) -> dict | None:
    """One panel row, written independently: pooled n / win / avg / PF; at ALL
    over several variants the mean of each variant's total, Max DD, Calmar."""
    if not len(df):
        return None
    groups = [g for _, g in df.groupby("vfile")] if per_variant else [df]
    per = [one_variant(g) for g in groups]
    mean = lambda xs: (sum(xs) / len(xs)) if xs else None  # noqa: E731
    pooled = one_variant(df)
    several = per_variant and len(groups) > 1
    gw, gl = float(df.loc[df["pnl"] > 0, "pnl"].sum()), float(df.loc[df["pnl"] < 0, "pnl"].sum())
    return {"n": len(df), "win": float((df["pnl"] > 0).mean() * 100), "avg": float(df["pnl"].mean()),
            "total": mean([x["total"] for x in per]) if several else pooled["total"],
            "dd": mean([x["dd"] for x in per]) if several else pooled["dd"],
            "calmar": mean([x["calmar"] for x in per if x["calmar"] is not None]) if several else pooled["calmar"],
            "pf": gw / abs(gl) if gl < 0 else (float("inf") if gw > 0 else None),
            "k": len(groups) if per_variant else 1}


def close(a, b) -> bool:
    a = float("inf") if a == "Infinity" else a
    if a is None or b is None:
        return a is None and b is None
    if a == float("inf") or b == float("inf"):
        return a == b
    return abs(a - b) <= 1e-6 * max(1.0, abs(a), abs(b))


def stats_of(df: pd.DataFrame) -> dict:
    from app.oo_backtest.stats import calculate_stats
    st = calculate_stats(df)
    return {"n": len(df), "total": round(float(df["pnl"].sum()), 6), "dd": round(float(st["max_drawdown"]), 6)}


CATALOG = {"metrics": [{"column_name": f"m{j}", "family": "iv", "form": "level", "tenor": None, "wing": None,
                        "min_date": "2024-03-01", "units": None, "description": f"metric {j}", "formula": None}
                       for j in range(6)],
           "family_groups": [{"label": "IV", "color": "#3498db", "families": ["iv"]}], "other_group": None,
           "form_labels": {"level": "level"}, "unit_formats": {},
           "default_unit_format": {"scale": 1, "decimals": 2, "suffix": ""}, "row_auto_bins": None, "built_at": "x"}
RANK_ROWS = [{"column": f"m{j}", "family": "iv", "tenor": None, "wing": None, "form": "level", "n": 40, "bars": 40,
              "pearson": 0.3 - j * 0.05, "pearson_p": 1e-3 * (j + 1), "pearson_p_bh": 1e-2 * (j + 1),
              "spearman": 0.3 - j * 0.05, "spearman_p": 1e-3 * (j + 1), "spearman_p_bh": 1e-2 * (j + 1)}
             for j in range(6)]

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
window.__excluded = 0;
window.__ranks = [];
window.__blobs = [];
const realURL = URL.createObjectURL.bind(URL);
URL.createObjectURL = (b) => { window.__blobs.push(b); return realURL(b); };
HTMLAnchorElement.prototype.click = function () {};
window.fetch = async (u, init) => {
  const url = String(u);
  const reply = (status, body) => ({ ok: status < 400, status, json: async () => JSON.parse(JSON.stringify(body)) });
  if (/\/registry$/.test(url)) return reply(200, { metrics: FIX.registry, coverage_error: null });
  if (/\/strategies$/.test(url)) return reply(200, { strategies: [] });
  if (/\/market-status$/.test(url)) return reply(200, { ok: false, error: 'stub' });
  if (/\/parse-batch$/.test(url)) return reply(200, FIX.payload);
  if (/\/surface\/catalog$/.test(url)) return reply(200, FIX.catalog);
  if (/\/surface\/rank$/.test(url)) {
    const body = JSON.parse(init.body);
    window.__ranks.push(body);
    return reply(200, { rows: FIX.rankRows, report: { no_bar: 0, with_bar: body.pnl.length, distinct_entries: body.bars.length },
                        lookahead_confirmed: true });
  }
  return reply(404, { detail: 'stub' });
};

window.addEventListener('load', () => setTimeout(async () => {
  const wait = ms => new Promise(r => setTimeout(r, ms));
  const until = async (cond, ms = 6000) => { for (let t = 0; t < ms; t += 25) { if (cond()) return true; await wait(25); } return cond(); };
  const steps = [], facts = {};
__SNAPSHOT__
  // A TRAIL of (step, virtual ms, wall ms): if the page is dumped before it
  // reports, the dump says where the time went.
  const WALL0 = Date.now();
  const mark = (name) => {
    const t = document.documentElement;
    t.setAttribute('data-progress', name);
    t.setAttribute('data-trail', (t.getAttribute('data-trail') || '') + `${name}@${Math.round(performance.now())}v/${Date.now() - WALL0}w;`);
  };
  window.__mark = mark;
  const snap = async (name, c) => {
    mark(name);
    await wait(350);
    window.__excluded = 0;
    steps.push({ name, dom: scrub(dom()), charts: charts(), state: state(c) });
  };
  const settle = () => wait(350);
  const splitCard = () => document.querySelector('[data-gate-exclude="train-test-card"]');
  const seg = label => [...splitCard().querySelectorAll('.seg-btn')].find(b => txt(b) === label);
  const counts = () => [...splitCard().querySelectorAll('.ob-split-counts div')].map(txt);
  const status = () => txt(document.querySelector('.ob-status'));
  const showing = () => txt(document.querySelector('.ob-status span'));
  const card = t => [...document.querySelectorAll('.ob-side .ob-card')].find(e => txt(e.querySelector('.ob-card-title')) === t);
  const button = t => [...document.querySelectorAll('button')].find(b => txt(b) === t);
  const cutoffEl = () => document.getElementById('ob-cutoff');
  const typeCutoff = async (v) => { window.__mark('cutoff ' + v); const el = cutoffEl(); el.value = v; el.dispatchEvent(new Event('change', { bubbles: true })); await settle(); };
  const typeDate = async (which, v) => {
    const el = document.querySelector(`input[aria-label="Entry date ${which}"]`);
    el.value = v; el.dispatchEvent(new Event('input', { bubbles: true })); el.dispatchEvent(new Event('change', { bubbles: true }));
    await settle();
  };
  const exportText = async () => { window.__mark('export'); const n = window.__blobs.length; button('Export CSV').click(); await until(() => window.__blobs.length > n);
                                   return window.__blobs.length > n ? await window.__blobs[window.__blobs.length - 1].text() : null; };
  const rank = async (c) => { window.__mark('rank'); const n = window.__ranks.length; (button('Rank metrics') || button('Recompute')).click();
                              await until(() => window.__ranks.length > n && !c.surf.busy); await settle();
                              return window.__ranks.length > n ? window.__ranks[window.__ranks.length - 1] : null; };
  const panel = () => {
    const card = document.querySelector('.ob-tt');
    if (!card) return null;
    return { note: txt(card.querySelector('.ob-card-sub')),
             // Cells only: an x-if leaves a <template> inside the row.
             rows: [...card.querySelectorAll('.ob-tt-table tbody tr')].map(tr => [...tr.children].filter(td => td.tagName === 'TD').map(td => txt(td))),
             // Infinity (a period with no losing trade: PF) would become null in JSON.
             data: JSON.parse(JSON.stringify(Alpine.$data(document.body).panel, (k, v) => (v === Infinity ? 'Infinity' : v))),
             builds: OB_DATA.panelUBuilds || 0 };
  };
  const cutLine = () => ['ob-cum-chart', 'ob-dd-chart'].map(id => {
    const ch = Chart.getChart(document.getElementById(id));
    if (!ch) return 'no chart';
    const d = ch.$cutoffDrawn;
    return d ? { ...d, want: d.outside ? null : ch.scales.x.getPixelForValue(obDay(Alpine.$data(document.body).split.cutoff)) } : null;
  });
  const grids = () => {
    const card = [...document.querySelectorAll('.ob-main .ob-card')].find(e => txt(e.querySelector('.ob-card-title')) === 'Parameter grid');
    if (!card || !shown(card)) return null;
    const read = t => t ? [...t.querySelectorAll('tbody tr')].map(tr => [...tr.querySelectorAll('td')].map(td => ({
      text: txt(td.querySelector('.ob-pgrid-v')), sub: txt(td.querySelector('.ob-pgrid-n')), bg: getComputedStyle(td).backgroundColor }))) : null;
    const left = card.querySelector('.ob-pgrid-wrap > table.ob-pgrid'), test = card.querySelector('.ob-pgrid-test table.ob-pgrid');
    return { train: read(left), test: read(test), testRowLabels: test ? test.querySelectorAll('tbody th').length : null,
             heads: [...card.querySelectorAll('.ob-pgrid-head')].map(txt), sub: txt(card.querySelector('.ob-card-sub')) };
  };
  const emptyShown = () => { const e = [...document.querySelectorAll('.ob-main .ob-empty')].find(shown); return e ? txt(e) : ''; };
  const sumSub = () => { const e = document.querySelector('.ob-summary .ob-card-sub'); return e ? txt(e) : ''; };
  const rankSub = () => [...document.querySelectorAll('.ob-main .ob-card')].filter(e => /Surface metrics vs P\/L/.test(txt(e.querySelector('.ob-card-title'))))
                          .map(e => [...e.querySelectorAll('.ob-card-hdr .ob-card-sub')].map(txt)).flat();
  try {
    let c = null;
    for (let i = 0; i < 100 && !(c && typeof c.recompute === 'function'); i++) {
      await wait(50); c = document.body._x_dataStack ? Alpine.$data(document.body) : null;
    }
    await until(() => c.registry.length > 0);
    const dt = new DataTransfer();
    for (const n of FIX.names) dt.items.add(new File(['x'], n, { type: 'text/csv' }));
    const input = document.querySelector('.ob-drop input[type=file]');
    input.files = dt.files;
    input.dispatchEvent(new Event('change', { bubbles: true }));
    await until(() => c.loaded && c.filteredCount > 0);
    await snap('off', c);
    const sc = splitCard();
    facts.cardOff = { text: txt(sc), controls: sc.querySelectorAll('input, select, textarea, button').length,
                      segs: [...sc.querySelectorAll('.seg-btn')].filter(shown).map(txt), offOn: seg('Off').classList.contains('on') };
    facts.cutoffDefault = c.split.cutoff;

    seg('On').click(); await settle();
    facts.afterOn = { trainOn: seg('Train').classList.contains('on'), fullOn: seg('Full').classList.contains('on'),
                      cutoffShown: cutoffEl() ? cutoffEl().value : null, counts: counts(), status: status(), sumSub: sumSub(),
                      panel: panel(), panelState: c.panel, line: cutLine() };
    await snap('train', c);
    seg('Full').click(); await settle();
    facts.afterFull = { counts: counts(), status: status(), sumSub: sumSub(), panel: panel(), line: cutLine() };
    await snap('full', c);
    seg('Off').click(); await settle();
    facts.afterOff = { panel: panel(), panelState: c.panel, line: cutLine() };
    await snap('off again', c);

    // ── Train view, a clicked filter ─────────────────────────────────────
    seg('On').click(); await settle();
    const mon = [...card('Day of Week').querySelectorAll('.ob-check')].find(l => txt(l).startsWith('Mon'));
    facts.dowEnabled = !mon.querySelector('input').disabled;
    const before = c.filteredCount;
    mon.querySelector('input').click(); await until(() => c.filteredCount !== before); await settle();
    facts.trainDow = { counts: counts(), status: status(), panel: panel(), panelState: c.panel, grids: grids(), line: cutLine() };
    await snap('train dow', c);
    facts.csvTrain = await exportText();
    facts.rankTrain = await rank(c);
    facts.rankSubTrain = rankSub();
    await snap('train dow ranked', c);

    // ── Full view ────────────────────────────────────────────────────────
    seg('Full').click(); await settle();
    facts.fullDow = { counts: counts(), status: status(), stale: c.surfStale(), panel: panel(), grids: grids(), line: cutLine() };
    await snap('full dow', c);
    facts.rankFull = await rank(c);
    facts.rankSubFull = rankSub();
    facts.csvFull = await exportText();

    // ── a variant (sweeps) ───────────────────────────────────────────────
    if (FIX.variant) {
      const sel = document.getElementById('ob-variant');
      sel.value = FIX.variant; sel.dispatchEvent(new Event('change', { bubbles: true }));
      await until(() => c.variantSel === FIX.variant); await settle();
      facts.cutoffAfterVariant = cutoffEl().value;
      seg('Train').click(); await settle();
      await snap('train dow variant', c);
      seg('Full').click(); await settle();
      facts.fullVariant = { counts: counts(), status: status(), panel: panel() };
      sel.value = 'all'; sel.dispatchEvent(new Event('change', { bubbles: true }));
      await until(() => c.variantSel === 'all'); await settle();
      // A PARAMETER FILTER is variant selection too: untick one value of the
      // first parameter, BY CLICK, in Full view; read the panel; tick it back.
      const pcard = card(FIX.param.name);
      const pbox = [...pcard.querySelectorAll('.ob-check')].find(l => txt(l).startsWith(String(FIX.param.value)));
      const before2 = c.filteredCount;
      pbox.querySelector('input').click(); await until(() => c.filteredCount !== before2); await settle();
      facts.fullParam = { panel: panel() };
      // Between two PARTIAL selections the spec is present on both sides, so
      // only the ticked values tell them apart: {other} -> {} -> {this}.
      const other = [...pcard.querySelectorAll('.ob-check')].find(l => !txt(l).startsWith(String(FIX.param.value)));
      other.querySelector('input').click(); await settle();
      facts.fullParamNone = { panel: panel() };
      pbox.querySelector('input').click(); await settle();
      facts.fullParamSwap = { panel: panel() };
      other.querySelector('input').click(); await until(() => c.filteredCount === before2); await settle();
    }

    // ── the date range moves the universe, never the cutoff ─────────────
    await typeDate('from', FIX.dateFrom);
    facts.afterDate = { cutoff: cutoffEl().value, counts: counts(), status: status(), sumSub: sumSub(), panel: panel() };
    seg('Train').click(); await settle();
    facts.afterDateTrain = { cutoff: cutoffEl().value, sumSub: sumSub(), status: status() };
    seg('Full').click(); await settle();
    await typeDate('from', c.dateBounds.min);

    // ── typed cutoffs ─────────────────────────────────────────────────────
    facts.cutoffs = [];
    for (const co of FIX.cutoffs) {
      await typeCutoff(co);
      const full = { cutoff: cutoffEl().value, counts: counts(), status: status(), panel: panel(), line: cutLine() };
      seg('Train').click(); await settle();
      const train = { counts: counts(), status: status(), empty: emptyShown(), sumSub: sumSub() };
      await snap('train at ' + co, c);
      seg('Full').click(); await settle();
      facts.cutoffs.push({ co, full, train });
    }
    // A malformed date is refused; the field shows the cutoff in force.
    const inForce = c.split.cutoff;
    await typeCutoff('2024-02-30x');
    facts.badCutoff = { value: cutoffEl().value, inForce, now: c.split.cutoff };
    await typeCutoff(FIX.cutoffs[1]);
    const reset = [...splitCard().querySelectorAll('a.ob-reset')].find(shown);
    // A real click event: the gate stubs <a>.click() to stop downloads.
    if (reset) reset.dispatchEvent(new MouseEvent('click', { bubbles: true, cancelable: true }));
    await settle();
    facts.afterReset = cutoffEl().value;

    // ── the cutoff outside the date range ───────────────────────────────
    await typeDate('from', FIX.outside.from);
    await typeCutoff(FIX.outside.cutoff);
    seg('Train').click(); await settle();
    facts.outside = { status: status(), empty: emptyShown(), sumSub: sumSub(), counts: counts() };
    await snap('train cutoff outside range', c);
    seg('Full').click(); await settle();
    facts.outsideFull = { counts: counts(), status: status() };
    await typeDate('from', c.dateBounds.min);

    window.__result = { steps, facts, errs: window.__errs };
  } catch (e) {
    window.__result = { error: String(e) + (e && e.stack ? ' ' + e.stack : ''), steps, facts, errs: window.__errs };
  }
  const pre = document.createElement('pre');
  pre.id = 'report';
  pre.textContent = JSON.stringify(window.__result);
  document.body.appendChild(pre);
}, 400));
</script>
"""


SAVED_DRIVER = r"""
<script>
window.__errs = [];
window.onerror = (m) => window.__errs.push('error: ' + m);
window.addEventListener('unhandledrejection', e => window.__errs.push('promise: ' + e.reason));
(function () { const ce = console.error; console.error = (...a) => { window.__errs.push('console.error: ' + a.map(String).join(' ')); ce(...a); }; })();
const FIX = __FIX__;
window.__puts = []; window.__posts = [];
const saved = Object.fromEntries(FIX.saved.map(sv => [sv.id, sv]));
window.fetch = async (u, init) => {
  const url = String(u);
  const reply = (status, body) => ({ ok: status < 400, status, json: async () => JSON.parse(JSON.stringify(body)) });
  if (/\/registry$/.test(url)) return reply(200, { metrics: FIX.registry, coverage_error: null });
  if (/\/market-status$/.test(url)) return reply(200, { ok: false, error: 'stub' });
  if (/\/surface\/catalog$/.test(url)) return reply(200, FIX.catalog);
  if (/\/parse-batch$/.test(url)) return reply(200, FIX.payload);
  let m = url.match(/\/strategies\/(\d+)\/split$/);
  if (m) {
    const body = JSON.parse(init.body); window.__puts.push({ id: +m[1], body });
    return reply(200, { strategy: { ...saved[m[1]], split_on: body.split_on, split_cutoff: body.split_cutoff } });
  }
  m = url.match(/\/strategies\/(\d+)\/load$/);
  if (m) return reply(200, { ...FIX.payload, saved: saved[m[1]] });
  if (/\/strategies$/.test(url) && init && init.method === 'POST') {
    const f = init.body; const rec = { name: f.get('name'), split_on: f.get('split_on'), split_cutoff: f.get('split_cutoff') };
    window.__posts.push(rec);
    return reply(200, { strategy: { ...FIX.saved[0], id: 3, name: rec.name, split_on: rec.split_on === 'true',
                                    split_cutoff: rec.split_cutoff || null } });
  }
  if (/\/strategies$/.test(url)) return reply(200, { strategies: FIX.saved });
  return reply(404, { detail: 'stub' });
};
window.addEventListener('load', () => setTimeout(async () => {
  const wait = ms => new Promise(r => setTimeout(r, ms));
  const until = async (cond, ms = 6000) => { for (let t = 0; t < ms; t += 25) { if (cond()) return true; await wait(25); } return cond(); };
  const txt = el => (el ? el.textContent.replace(/\s+/g, ' ').trim() : '');
  const facts = {};
  const card = () => document.querySelector('[data-gate-exclude="train-test-card"]');
  const seg = l => [...card().querySelectorAll('.seg-btn')].find(b => txt(b) === l);
  const on = l => !!seg(l) && seg(l).classList.contains('on');
  const unsaved = () => !!card().querySelector('.ob-split-save');
  const cutoff = () => { const e = document.getElementById('ob-cutoff'); return e ? e.value : null; };
  const state = () => ({ on: on('On'), off: on('Off'), train: on('Train'), full: on('Full'), cutoff: cutoff(),
                         unsaved: unsaved(), puts: window.__puts.length, card: txt(card()) });
  const settle = () => wait(400);
  try {
    let c = null;
    for (let i = 0; i < 200 && !(c && c.recompute && c.registry.length); i++) { await wait(50); c = document.body._x_dataStack ? Alpine.$data(document.body) : null; }
    await until(() => c.savedList.length === FIX.saved.length);
    const pick = async id => {
      const sel = [...document.querySelectorAll('.ob-side select')].find(x => /saved strategy/.test(txt(x.options[0])));
      sel.value = String(id); sel.dispatchEvent(new Event('change', { bubbles: true }));
      await settle();
      [...document.querySelectorAll('.ob-side button')].find(b => txt(b) === 'Load').click();
      await until(() => c.loaded && c.meta && c.meta.saved && c.meta.saved.id === id); await settle();
    };
    await pick(1);
    facts.s1 = state();
    seg('Full').click(); await settle();
    facts.s1full = state();
    const el = document.getElementById('ob-cutoff');
    el.value = FIX.typed; el.dispatchEvent(new Event('change', { bubbles: true })); await settle();
    facts.typed = state();
    await wait(1200);
    facts.typedLater = state();
    card().querySelector('.ob-split-save button').click();
    await until(() => window.__puts.length > 0); await settle();
    facts.afterSave = { ...state(), put: window.__puts[0] };
    await pick(2);                                   // loaded while the page is in Full view
    facts.s2 = state();
    seg('On').click(); await settle();
    facts.s2on = state();
    seg('Off').click(); await settle();
    facts.s2offAgain = state();
    seg('On').click(); await settle();
    card().querySelector('.ob-split-save button').click();
    await until(() => window.__puts.length > 1); await settle();
    facts.s2saved = { ...state(), put: window.__puts[1] };
    seg('Full').click(); await settle();
    await pick(1);
    facts.s1again = state();
    // An upload, then its Save box.
    const dt = new DataTransfer(); dt.items.add(new File(['x'], 'upload.csv', { type: 'text/csv' }));
    const inp = document.querySelector('.ob-drop input[type=file]'); inp.files = dt.files; inp.dispatchEvent(new Event('change', { bubbles: true }));
    await until(() => c.hasUploadFile && !c.uploading); await settle();
    facts.upload = state();
    // Re-found: loading the strategy saved Off removed the field (x-if).
    const el2 = document.getElementById('ob-cutoff');
    el2.value = FIX.typedUpload; el2.dispatchEvent(new Event('change', { bubbles: true })); await settle();
    const nm = document.querySelector('.ob-save input[type=text]'); nm.value = 'uploaded with split'; nm.dispatchEvent(new Event('input', { bubbles: true }));
    await settle();
    [...document.querySelectorAll('.ob-save button')].find(b => txt(b) === 'Save').click();
    await until(() => window.__posts.length > 0); await settle();
    facts.uploadSaved = { ...state(), post: window.__posts[0] };
    window.__result = { facts, errs: window.__errs };
  } catch (e) {
    window.__result = { error: String(e) + (e && e.stack ? ' ' + e.stack : ''), facts, errs: window.__errs };
  }
  const pre = document.createElement('pre'); pre.id = 'report'; pre.textContent = JSON.stringify(window.__result);
  document.body.appendChild(pre);
}, 300));
</script>
"""


def check_saved(browser: str) -> None:
    """P4: the split saved with a strategy, clicked."""
    import uuid

    import edge_cdp
    from app.oo_backtest.registry import registry_with_coverage
    print("saving the split (P4)")
    files = {"split_log.csv": S.make_csv(2)}
    pa = payload(files)
    auto = ref_cutoff(pa["columns"]["date_opened"])
    row = {"notes": "", "source": "oo_csv", "filename": "split_log.csv", "file_sha256": "x", "file_bytes": 1,
           "trade_count": pa["n"], "date_min": pa["date_min"], "date_max": pa["date_max"], "capital_per_position": None,
           "created_at": "2026-10-25T00:00:00+00:00", "updated_at": "2026-10-25T00:00:00+00:00", "variants": None}
    fix = {"registry": registry_with_coverage(None), "catalog": CATALOG, "payload": pa,
           "saved": [{**row, "id": 1, "name": "with split", "split_on": True, "split_cutoff": "2024-05-01"},
                     {**row, "id": 2, "name": "saved before the split", "split_on": False, "split_cutoff": None}],
           "typed": "2024-06-03", "typedUpload": "2024-07-01"}
    html = N.build_page().replace(N.DRIVER.replace("__SAVED__", json.dumps(N.SAVED)), "")
    tmp = ROOT / "scripts" / f"_oo_split_saved_{uuid.uuid4().hex[:12]}.html"
    tmp.write_text(html.replace("</head>", SAVED_DRIVER.replace("__FIX__", json.dumps(fix)) + "</head>", 1), encoding="utf-8")
    try:
        out = judge_run(json.loads(edge_cdp.run_page(browser, tmp, timeout_s=300)))
    finally:
        tmp.unlink(missing_ok=True)
    f = out.get("facts", {})
    s1, s1f, ty, tyl, sv = (f.get(k, {}) for k in ("s1", "s1full", "typed", "typedLater", "afterSave"))
    check(s1.get("on") is True and s1.get("train") is True and s1.get("cutoff") == "2024-05-01" and s1.get("unsaved") is False,
          f"a strategy saved with the split On loads On, in TRAIN, with its cutoff, nothing unsaved ({s1})")
    check(s1f.get("full") is True and s1f.get("unsaved") is False, "switching to Full is not an unsaved change (the view is not saved)")
    check(ty.get("unsaved") is True and ty.get("cutoff") == "2024-06-03" and ty.get("puts") == 0
          and tyl.get("puts") == 0 and tyl.get("unsaved") is True,
          "typing a cutoff shows 'unsaved' and sends NOTHING, a second later too")
    check(sv.get("put") == {"id": 1, "body": {"split_on": True, "split_cutoff": "2024-06-03"}} and sv.get("unsaved") is False,
          f"clicking the card's Save sends exactly the split, and 'unsaved' clears ({sv.get('put')})")
    s2, s2on, s2off, s2s, s1a = (f.get(k, {}) for k in ("s2", "s2on", "s2offAgain", "s2saved", "s1again"))
    check(s2.get("off") is True and s2.get("unsaved") is False and s2.get("card") == "Train / Test Split Off On",
          f"a strategy saved before the split existed loads Off -- the card holds only Off/On ({s2.get('card')!r})")
    check(s2on.get("cutoff") == auto and s2on.get("train") is True and s2on.get("unsaved") is True,
          f"turning it On gives the 70% default ({s2on.get('cutoff')} = {auto}), in Train, unsaved")
    check(s2off.get("unsaved") is False, "turning it Off again matches what is saved: nothing unsaved")
    check(s2s.get("put") == {"id": 2, "body": {"split_on": True, "split_cutoff": auto}} and s2s.get("unsaved") is False,
          "its Save sends On and the default cutoff")
    check(s1a.get("train") is True and s1a.get("full") is False and s1a.get("cutoff") == "2024-05-01",
          "a strategy loaded while the page is in Full view opens in TRAIN")
    up, us = f.get("upload", {}), f.get("uploadSaved", {})
    check(up.get("unsaved") is False, "an upload has no 'unsaved' marker (its Save box saves the split)")
    check(us.get("post") == {"name": "uploaded with split", "split_on": "true", "split_cutoff": "2024-07-01"}
          and us.get("unsaved") is False,
          f"an upload's Save carries split_on and the cutoff ({us.get('post')})")
    print()


def drive(browser: str, fix: dict, exclude: str = "__nothing__") -> dict:
    html = N.build_page()
    html = html.replace(N.DRIVER.replace("__SAVED__", json.dumps(N.SAVED)), "")
    page = html.replace("</head>", DRIVER.replace("__SNAPSHOT__", O.SNAPSHOT_JS).replace("__FIX__", json.dumps(fix))
                        .replace("__EXCLUDE__", exclude) + "</head>", 1)
    import uuid

    import edge_cdp
    # Unique per load: loads run in parallel threads of one process.
    tmp = ROOT / "scripts" / f"_oo_split_ui_{uuid.uuid4().hex[:12]}.html"
    tmp.write_text(page, encoding="utf-8")
    # REAL TIME, over the DevTools protocol (edge_cdp): the page's own report
    # is waited for, and a timeout names the last step it marked. It ran on
    # --virtual-time-budget until 2026-10-25, and twice the page was printed
    # partway through under the parallel suite, cause not found.
    try:
        text = edge_cdp.run_page(browser, tmp, timeout_s=300)
    finally:
        tmp.unlink(missing_ok=True)
    return json.loads(text)


def judge_run(out: dict) -> dict:
    """The checks every load gets, on the main thread (loads run in parallel)."""
    if out.get("error"):
        check(False, "the driver ran to the end", out["error"])
    check(not out.get("errs"), "no console errors", " ~ ".join(out.get("errs") or [])[:600])
    return out


FILE_SIZE = re.compile(r"\([\d,]+ trades, \d{4}-\d{2}-\d{2} – \d{4}-\d{2}-\d{2}\)")


def without_file_summary(s: dict) -> dict:
    """The step with the Data source card's description of the loaded FILE
    (.ob-loaded: its trade count, the sweep's per-file lines) cut out of the
    DOM -- the walker's lines are depth-first with the depth first."""
    out, cut_at = [], None
    for line in s["dom"]:
        depth = int(line.split("|", 1)[0])
        if FILE_SIZE.search(line):           # the topbar's copy of the same label
            continue
        if cut_at is not None and depth > cut_at:
            continue
        cut_at = None
        if "ob-loaded" in line.split("|")[2].split():
            cut_at = depth
            continue
        out.append(line)
    return {**s, "dom": out}


def step(run: dict, name: str) -> dict:
    return next((s for s in run["steps"] if s["name"] == name), {"dom": [], "charts": [], "state": {}})


def same(a: dict, b: dict, parts=("dom", "charts", "state")) -> tuple[bool, str]:
    for part in parts:
        if a[part] != b[part]:
            if part == "dom":
                return False, "; ".join(O.dom_diff(a["dom"], b["dom"], 4))
            return False, "; ".join(O.first_diffs(a[part], b[part], part, limit=4))
    return True, ""


def count_line(lines: list[str], which: str) -> dict:
    """'Filtered · Train 742 · Test 318' -> {'train': 742, 'test': 318}."""
    line = next((l for l in lines if l.startswith(which)), "")
    got = {k.lower(): int(v.replace(",", "")) for k, v in re.findall(r"(Train|Test) ([\d,]+)", line)}
    return got


def showing(status: str) -> tuple[int, int]:
    m = re.search(r"Showing ([\d,]+) of ([\d,]+) trades", status)
    return (int(m.group(1).replace(",", "")), int(m.group(2).replace(",", ""))) if m else (-1, -1)


BAD_TEXT = re.compile(r"\bNaN\b|\bundefined\b|\bInfinity\b")


def no_bad_text(run: dict, label: str) -> None:
    # On screen only (not 'H'), and not the market pane's own label "zero/NaN
    # bars" -- the stubbed market status leaves that hidden pane half-filled.
    hits = [l for s in run["steps"] for l in s["dom"]
            if l.split("|")[3] != "H" and BAD_TEXT.search(l.replace("zero/NaN bars", ""))]
    def shown_only(v):
        # What reached the screen: not the panel's raw numbers ("data"), not
        # the component's state, not request bodies.
        if isinstance(v, dict):
            return {k: shown_only(x) for k, x in v.items() if k not in ("data", "panelState")}
        return [shown_only(x) for x in v] if isinstance(v, list) else v
    facts_txt = json.dumps(shown_only({k: v for k, v in run["facts"].items()
                                       if k not in ("csvTrain", "csvFull", "rankTrain", "rankFull")}))
    hits += [m.group(0) for m in BAD_TEXT.finditer(facts_txt)]
    check(not hits, f"{label}: no NaN, undefined or Infinity on any screen", "; ".join(map(str, hits[:4])))


# ── one log, end to end ──────────────────────────────────────────────────────

def check_log(browser: str, label: str, files: dict[str, str], variant: str | None = None) -> None:
    from app.oo_backtest.registry import registry_with_coverage
    print(label)
    pa = payload(files)
    cut = ref_cutoff(pa["columns"]["date_opened"])
    df = frame(files)
    df["d"] = df["date_opened"].dt.strftime("%Y-%m-%d")
    d0, d1 = df["d"].min(), df["d"].max()
    # Typed cutoffs: before the first trade, three inside, after the last.
    inside = sorted(df["d"].unique())
    cutoffs = [(date.fromisoformat(d0) - timedelta(days=40)).isoformat(),
               inside[len(inside) // 4], inside[len(inside) // 2], inside[(3 * len(inside)) // 4],
               (date.fromisoformat(d1) + timedelta(days=40)).isoformat()]
    date_from = inside[len(inside) // 3]
    outside = {"from": inside[(2 * len(inside)) // 3], "cutoff": inside[len(inside) // 5]}

    b_files = {n: alter_test(c, cut) for n, c in files.items()}
    c_files = {n: duplicate_test(c, cut) for n, c in files.items()}
    vfile, param = None, None
    if variant:
        p0 = pa["sweep"]["params"][0]
        param = {"name": p0, "value": pa["sweep"]["values"][p0][0]}
        vfile = pa["sweep"]["variants"][2]["file"]
        variant = str(pa["sweep"]["variants"][2]["id"])
    base = {"registry": registry_with_coverage(None), "catalog": CATALOG, "rankRows": RANK_ROWS, "names": list(files),
            "cutoffs": cutoffs, "dateFrom": date_from, "outside": outside, "variant": variant,
            "param": param}
    from concurrent.futures import ThreadPoolExecutor
    fixes = {"A": {**base, "payload": pa}}
    if "B" in SCOPE["runs"]:
        fixes["B"] = {**base, "payload": payload(b_files)}
    if "C" in SCOPE["runs"]:
        fixes["C"] = {**base, "payload": payload(c_files)}
    # A always loads (every check reads it); B and C as scoped. Three at a time.
    with ThreadPoolExecutor(max_workers=3) as ex:
        futures = {k: ex.submit(drive, browser, f) for k, f in fixes.items()}
        got = {k: f.result() for k, f in futures.items()}
    A = judge_run(got["A"])
    B = judge_run(got["B"]) if "B" in got else None
    C = judge_run(got["C"]) if "C" in got else None
    fa = A["facts"]

    # ── the default cutoff, and the card while off ──────────────────────
    check(fa.get("cutoffDefault") == cut, f"default cutoff {fa.get('cutoffDefault')} = the entry date nearest 70% of "
                                          f"{d0} … {d1} ({cut})")
    co = fa.get("cardOff", {})
    check(co.get("controls") == 0 and co.get("segs") == ["Off", "On"] and co.get("offOn") is True
          and co.get("text") == "Train / Test Split Off On",
          "Split Off: the card holds only the Off/On control (Off on)", json.dumps(co))
    ao = fa.get("afterOn", {})
    check(ao.get("trainOn") is True and ao.get("fullOn") is False and ao.get("cutoffShown") == cut,
          "clicking On opens in Train, the cutoff field showing the default", json.dumps(ao))

    # ── Off returns the page exactly ─────────────────────────────────────
    ok, why = same(step(A, "off"), step(A, "off again"))
    check(ok, "On → Train → Full → Off: the DOM, every chart and every figure are the pre-split page's again", why)
    # Full computes what Off computes.
    # Except the parameter grid: in Full view its left pivot is the TRAIN
    # trades by design (P3) -- checked against pandas below.
    # (and the calmar the Off grid caches onto each variant's entry as it
    # draws -- a side effect of the grid, not a figure anything shows).
    def no_grid(st):
        state = {k: v for k, v in st["state"].items() if k != "grid"}
        if state.get("vm"):
            state["vm"] = [{k: v for k, v in f.items() if k != "calmar"} for f in state["vm"]]
        return {**st, "state": state}
    ok, why = same(no_grid(step(A, "off")), no_grid(step(A, "full")), parts=("charts", "state"))
    check(ok, "Full view's charts and figures are exactly the Off page's (only headers differ; the grid is P3's train/test pair)", why)

    # ── THE DIFFERENTIAL ─────────────────────────────────────────────────
    # B and C change only trades entered after the DEFAULT cutoff, so the
    # differential holds wherever the train period ends on or before it; at
    # a later typed cutoff those trades are train, and should show.
    def at_or_before_default(name):
        m = re.search(r"train at (\d{4}-\d{2}-\d{2})", name)
        return not m or m.group(1) <= cut
    train_steps = [s["name"] for s in A["steps"] if s["name"].startswith("train") and at_or_before_default(s["name"])]
    later = [s["name"] for s in A["steps"] if s["name"].startswith("train at") and not at_or_before_default(s["name"])]
    for name in (later if B else []):
        ok, _ = same(step(A, name), step(B, name), parts=("state",))
        check(not ok, f"'{name}' (after the default cutoff): the changed trades ARE train there, and show")
    if B:
        for name in train_steps:
            ok, why = same(step(A, name), step(B, name))
            check(ok, f"Train view '{name}': changing every TEST trade's P/L and exit reason changes nothing on the "
                      f"page (DOM, titles, charts, figures)", why)
        check(fa.get("csvTrain") == B["facts"].get("csvTrain"), "Train view: the Export CSV is identical with the test trades changed")
        check(fa.get("rankTrain") == B["facts"].get("rankTrain"), "Train view: the /surface/rank request is identical with the test trades changed")
        ok, _ = same(step(A, "full"), step(B, "full"), parts=("state",))
        check(not ok, "...and the same change DOES show in Full view (the differential can see test trades)")
    if C:
        for name in train_steps:
            ok, why = same(without_file_summary(step(A, name)), without_file_summary(step(C, name)))
            check(ok, f"Train view '{name}': doubling the TEST trades changes nothing but the Data source card's file size", why)
        ok, _ = same(step(A, "full"), step(C, "full"), parts=("state",))
        check(not ok, "...and doubling them DOES show in Full view")

    # ── Train figures are stats.py on the train trades ──────────────────
    def st(run, name):
        s = step(run, name)["state"]
        return {"n": s.get("filteredCount"), "total": round(float((s.get("stats") or {}).get("total_pnl") or 0), 6),
                "dd": round(float((s.get("stats") or {}).get("max_drawdown") or 0), 6)}
    tr = df[df["d"] <= cut]
    want = stats_of(tr) if variant is None else None
    if want:
        check(st(A, "train") == want, f"Train view = stats.py on the {want['n']} train trades: {st(A, 'train')}", json.dumps(want))
        nomon = tr[tr["date_opened"].dt.dayofweek != 0]
        check(st(A, "train dow") == stats_of(nomon),
              "Train view with Mon unticked BY CLICK = stats.py on the train trades not entered on a Monday",
              json.dumps([st(A, 'train dow'), stats_of(nomon)]))
    check(fa.get("dowEnabled") is True, "Day of Week stays enabled with the split on (the 43312b4 shape)")

    # ── counts: train + test = the filtered total ───────────────────────
    nomon_all = df[df["date_opened"].dt.dayofweek != 0]
    fd = fa.get("fullDow", {})
    cl = count_line(fd.get("counts", []), "Filtered")
    shown_n, of_n = showing(fd.get("status", ""))
    check(cl.get("train", -1) + cl.get("test", -1) == shown_n == len(nomon_all)
          and cl.get("train") == int((nomon_all["d"] <= cut).sum()),
          f"Full view, Mon unticked: Train {cl.get('train')} + Test {cl.get('test')} = Showing {shown_n} = pandas",
          json.dumps(fd))
    td = fa.get("trainDow", {})
    check(count_line(td.get("counts", []), "Filtered").get("test") is None
          and count_line(td.get("counts", []), "Unfiltered").get("test") is None,
          "Train view: the card shows no test count", json.dumps(td.get("counts")))
    t_shown, t_of = showing(td.get("status", ""))
    check((t_shown, t_of) == (int((nomon_all["d"] <= cut).sum()), int((df["d"] <= cut).sum())) and "train period" in td.get("status", ""),
          f"Train view status: 'Showing {t_shown} of {t_of} trades · train period' -- the train trades, filtered and all",
          td.get("status", ""))
    for e in fa.get("cutoffs", []):
        c2 = e["co"]
        cl = count_line(e["full"]["counts"], "Filtered")
        sh, _ = showing(e["full"]["status"])
        want_tr = int((nomon_all["d"] <= c2).sum())
        check(cl.get("train", -1) + cl.get("test", -1) == sh and cl.get("train") == want_tr and e["full"]["cutoff"] == c2,
              f"typed cutoff {c2}: Train {cl.get('train')} + Test {cl.get('test')} = {sh} (pandas train {want_tr})",
              json.dumps(e["full"]))
        tsh, _ = showing(e["train"]["status"])
        if want_tr == 0:
            check(tsh == 0 and "No trades in the train period" in e["train"]["empty"],
                  f"cutoff {c2} before the first trade: Train view says so ('{e['train']['empty'][:70]}…')", json.dumps(e["train"]))
        else:
            check(tsh == want_tr and not e["train"]["empty"], f"cutoff {c2}: Train view shows its {want_tr} train trades",
                  json.dumps(e["train"]))
        if c2 > d1:
            check(cl.get("test") == 0, f"cutoff {c2} after the last trade: Test 0, shown as a count", json.dumps(e["full"]))
    bc = fa.get("badCutoff", {})
    check(bc.get("value") == bc.get("inForce") == bc.get("now"), "a malformed cutoff is refused; the field shows the one in force",
          json.dumps(bc))
    check(fa.get("afterReset") == cut, f"reset returns the cutoff to the default ({fa.get('afterReset')})")

    # ── the cutoff stays put ─────────────────────────────────────────────
    ad = fa.get("afterDate", {})
    check(ad.get("cutoff") == cut, f"typing a date-range start ({date_from}) leaves the cutoff at {ad.get('cutoff')}")
    uni = df[(df["d"] >= date_from)]
    ucl = count_line(ad.get("counts", []), "Unfiltered")
    check(ucl == {"train": int((uni["d"] <= cut).sum()), "test": int((uni["d"] > cut).sum())},
          f"the date range narrows the universe first, the cutoff splits what is left: {ucl}", json.dumps(ad))
    check(ad.get("sumSub", "").startswith(f"full period · {date_from} – ")
          and fa.get("afterDateTrain", {}).get("sumSub") == f"train period · {date_from} – {cut}",
          f"Summary header: '{fa.get('afterDateTrain', {}).get('sumSub')}'", json.dumps([ad.get("sumSub"), fa.get("afterDateTrain")]))
    if variant is not None:
        check(fa.get("cutoffAfterVariant") == cut, "picking a variant leaves the cutoff where it was")

    # ── outside the date range ───────────────────────────────────────────
    ou = fa.get("outside", {})
    check("No trades in the train period" in ou.get("empty", "") and showing(ou.get("status", ""))[0] == 0
          and "none" in ou.get("sumSub", "") + ou.get("empty", ""),
          f"cutoff {outside['cutoff']} before a date range starting {outside['from']}: Train view says so", json.dumps(ou))

    # ── ranking: train only, both views ──────────────────────────────────
    rt, rf = fa.get("rankTrain") or {}, fa.get("rankFull") or {}
    bars_ok = all(b[0] <= cut for b in rt.get("bars", []))
    check(len(rt.get("pnl", [])) == int((nomon_all["d"] <= cut).sum()) and bars_ok,
          f"Rank metrics in Train view sends the {len(rt.get('pnl', []))} filtered train trades, every entry on or before the cutoff")
    check(rf == rt, "Rank metrics in FULL view sends exactly the same train trades")
    check(fd.get("stale") is False, "switching Train → Full does not mark the ranking stale (it is the same ranking)")
    check("ranked on train only" in fa.get("rankSubTrain", []) and "ranked on train only" in fa.get("rankSubFull", []),
          "the ranking card says 'ranked on train only' in both views")

    # ── export writes what the view shows ────────────────────────────────
    def csv_dates(text):
        if not text:
            return []
        rows = list(csv.reader(io.StringIO(text.lstrip("﻿"))))
        i = rows[0].index("date_opened")
        return [r[i] for r in rows[1:] if r]
    dtr, dfu = csv_dates(fa.get("csvTrain")), csv_dates(fa.get("csvFull"))
    check(len(dtr) == int((nomon_all["d"] <= cut).sum()) and all(x <= cut for x in dtr),
          f"Export CSV in Train view: the {len(dtr)} filtered train trades, none after the cutoff")
    check(len(dfu) == len(nomon_all), f"Export CSV in Full view: all {len(dfu)} filtered trades")

    # ── the Train vs Test panel ──────────────────────────────────────────
    per_v = variant is not None
    check(fa.get("afterOn", {}).get("panel") is None and fa.get("afterOn", {}).get("panelState") is None
          and fa.get("trainDow", {}).get("panel") is None and fa.get("trainDow", {}).get("panelState") is None,
          "Train view: the panel is not in the DOM and was not even computed")
    check(fa.get("afterOff", {}).get("panel") is None and fa.get("afterOff", {}).get("panelState") is None,
          "Split Off: no panel")

    def check_panel(pn, d_all, d_filt, what, cutoff, pv=None):
        pv = per_v if pv is None else pv
        if not pn:
            check(False, f"{what}: the panel is on the page in Full view")
            return
        ok_all, why = True, []
        for key, sel in (("train", lambda d: d[d["d"] <= cutoff]), ("test", lambda d: d[d["d"] > cutoff])):
            for side, src in (("u", d_all), ("f", d_filt)):
                want = panel_ref(sel(src), pv)
                got = (pn["data"] or {}).get(key, {}).get(side)
                if want is None or got is None:
                    if (want is None) != (got is None):
                        ok_all = False
                        why.append(f"{key}.{side}: {got} vs {want}")
                    continue
                for f in ("n", "win", "avg", "total", "dd", "calmar", "pf", "k"):
                    if not close(got.get(f), want[f]):
                        ok_all = False
                        why.append(f"{key}.{side}.{f}: page {got.get(f)} vs reference {want[f]}")
        check(ok_all, f"{what}: all four rows equal pandas + stats.py on their trades"
                      + (" (total / Max DD / Calmar as the mean of each variant's own)" if pv else ""),
              "; ".join(why[:6]))
        return ok_all

    fdp = fa.get("fullDow", {}).get("panel")
    check_panel(fa.get("afterFull", {}).get("panel"), df, df, "Full view, no filter", cut)
    check_panel(fdp, df, nomon_all, "Full view, Mon unticked", cut)
    if fdp:
        # The lift row, against the brief's formulas, as printed.
        tr_u, tr_f = panel_ref(df[df["d"] <= cut], per_v), panel_ref(nomon_all[nomon_all["d"] <= cut], per_v)
        lift = [""] + fdp["rows"][2]          # the lift row has no group cell; pad so indices match
        want_kept = f"{tr_f['n'] / tr_u['n'] * 100:.0f}% kept"
        dpp = tr_f["win"] - tr_u["win"]
        want_pp = ("+" if dpp > 0 else "") + f"{dpp:.1f}pp"
        ddl = (abs(tr_u["dd"]) - abs(tr_f["dd"])) / abs(tr_u["dd"]) * 100 if tr_u["dd"] else None
        want_dd = "—" if ddl is None else ("+" if ddl > 0 else "") + f"{ddl:.0f}%"
        apct = (tr_f["avg"] - tr_u["avg"]) / abs(tr_u["avg"]) * 100
        check(lift[1] == "lift" and lift[2] == want_kept and lift[3] == want_pp and lift[5] == "n/a" and lift[6] == want_dd
              and lift[4].endswith(("+" if apct > 0 else "") + f"{apct:.1f}%"),
              f"train lift row: {lift[2:]} (kept, pp, avg with its %, n/a, Max DD %)",
              json.dumps([want_kept, want_pp, apct, want_dd]))
        rows = [r for r in fdp["rows"]]
        check(len(rows) == 6 and rows[0][0].startswith("TRAIN") and rows[3][0].startswith("TEST"),
              "two blocks of three rows, periods as rows: TRAIN then TEST", json.dumps([r[:2] for r in rows]))
        note = fdp["note"]
        check("drawdown measured within each period from zero, not read off the full curve" in note,
              "the header says drawdown is per period, from zero", note)
        pooled_note = "MEANS PER VARIANT" in note
        check(pooled_note == per_v, ("the header says Total P/L, Max DD and Calmar are per-variant means"
                                     if per_v else "a single log: no per-variant note"), note)
        if per_v:
            ks = [r[2] for r in rows if r[1] in ("unfiltered", "filtered")]
            check(all(re.search(r"\d+ variants?$", k) for k in ks), f"each row shows its variant count: {ks}")
    # ── the cache: what rebuilds it and what does not ────────────────────
    b_full, b_dow = (fa.get("afterFull", {}).get("panel") or {}).get("builds"), (fdp or {}).get("builds")
    check(b_full is not None and b_full == b_dow,
          f"the unfiltered rows are NOT rebuilt by a clicked metric filter or by Off / Train / Full (builds {b_full} → {b_dow})")
    ad_p = fa.get("afterDate", {}).get("panel")
    if ad_p:
        check(ad_p["builds"] > (b_dow or 0), f"a typed date range rebuilds them (builds {b_dow} → {ad_p['builds']})")
        check_panel(ad_p, df[df["d"] >= date_from], nomon_all[nomon_all["d"] >= date_from],
                    f"after typing the date range {date_from}: the unfiltered rows are the narrowed universe's", cut)
    cut_builds = [(e["co"], (e["full"].get("panel") or {}).get("builds")) for e in fa.get("cutoffs", [])]
    check(all(b2 > b1 for (_, b1), (_, b2) in zip(cut_builds, cut_builds[1:])),
          f"every typed cutoff rebuilds them: {cut_builds}")
    fv = fa.get("fullVariant", {}).get("panel")
    if variant is not None and fv:
        vf = vfile
        check(fv["builds"] > (b_dow or 0), f"picking a variant rebuilds them (builds {b_dow} → {fv['builds']})")
        check_panel(fv, df[df["vfile"] == vf], nomon_all[nomon_all["vfile"] == vf],
                    f"one variant picked ({vf}): every row is that variant's alone", cut, pv=False)
        fp = fa.get("fullParam", {}).get("panel")
        keep = [v["file"] for v in pa["sweep"]["variants"] if v["params"][param["name"]] != param["value"]]
        check(fp is not None and fp["builds"] > fv["builds"],
              f"unticking {param['name']}={param['value']} (a parameter filter) rebuilds them")
        if fp:
            check_panel(fp, df[df["vfile"].isin(keep)], nomon_all[nomon_all["vfile"].isin(keep)],
                        f"{param['name']}={param['value']} unticked: the universe is the {len(keep)} remaining variants", cut)
        fn = fa.get("fullParamNone", {}).get("panel")
        check(fn is not None and all(fn["data"][k]["u"] is None for k in ("train", "test")),
              f"every {param['name']} value unticked: no unfiltered rows at all (not the previous selection's)",
              json.dumps(fn and fn["rows"]))
        fs = fa.get("fullParamSwap", {}).get("panel")
        only = [v["file"] for v in pa["sweep"]["variants"] if v["params"][param["name"]] == param["value"]]
        if fs:
            check_panel(fs, df[df["vfile"].isin(only)], nomon_all[nomon_all["vfile"].isin(only)],
                        f"only {param['name']}={param['value']} ticked (from the other value alone, via none): "
                        f"the {len(only)} variants holding it", cut)
    if variant is not None and fv:
        ks = [fv["data"][k][side]["k"] for k in ("train", "test") for side in ("u", "f") if fv["data"][k][side]]
        check("MEANS PER VARIANT" not in fv["note"] and ks and all(x == 1 for x in ks),
              f"one variant picked: no per-variant note, every row one variant ({ks})", fv["note"])
    for e in fa.get("cutoffs", []):
        pn = e["full"].get("panel")
        if e["co"] < d0:
            check(pn and any("no trades in the train period" in " ".join(r) for r in pn["rows"]),
                  f"cutoff {e['co']} before the first trade: the panel's TRAIN block says 'no trades in the train period'",
                  json.dumps(pn and pn["rows"]))
        elif e["co"] > d1:
            check(pn and any("no trades in the test period" in " ".join(r) for r in pn["rows"]),
                  f"cutoff {e['co']} after the last trade: the TEST block says 'no trades in the test period'",
                  json.dumps(pn and pn["rows"]))
        else:
            check_panel(pn, df, nomon_all, f"panel at typed cutoff {e['co']}", e["co"])

    # ── P3: the cutoff line ──────────────────────────────────────────────
    def line_ok(ln):
        return (isinstance(ln, list) and len(ln) == 2 and all(isinstance(x, dict) and not x.get("outside")
                and x.get("label") == f"entry cutoff {cut}" and x.get("color") == "#8a8a8a" and x.get("width") == 1
                and x.get("dash") == [4, 4] and abs(x["px"] - x["want"]) < 0.5 for x in ln))
    check(line_ok(fa.get("afterFull", {}).get("line")) and line_ok(fa.get("fullDow", {}).get("line")),
          f"Full view: the dashed cutoff line on BOTH charts at the cutoff's pixel, 'entry cutoff {cut}', #8a8a8a 1px 4 4",
          json.dumps(fa.get("afterFull", {}).get("line")))
    check(fa.get("afterOn", {}).get("line") == [None, None] and fa.get("trainDow", {}).get("line") == [None, None]
          and fa.get("afterOff", {}).get("line") == [None, None],
          "Train view and Split Off: no cutoff line on either chart",
          json.dumps([fa.get("afterOn", {}).get("line"), fa.get("afterOff", {}).get("line")]))
    late = next((e for e in fa.get("cutoffs", []) if e["co"] > d1), None)
    if late:
        check(all(isinstance(x, dict) and x.get("outside") for x in late["full"].get("line") or []),
              f"a cutoff after the last close ({late['co']}): no line drawn off the chart",
              json.dumps(late["full"].get("line")))

    # ── P3: the two pivots ───────────────────────────────────────────────
    if variant is not None:
        sw = pa["sweep"]
        gx, gy = sw["params"][0], sw["params"][1]

        def money2(v):
            return ("-" if v < 0 else "") + f"${abs(v):,.2f}"

        def want_grid(frame_):
            rows = []
            for yv in sw["values"][gy]:
                row = []
                for xv in sw["values"][gx]:
                    f = next(v["file"] for v in sw["variants"] if v["params"][gx] == xv and v["params"][gy] == yv)
                    d = frame_[frame_["vfile"] == f]
                    row.append({"text": money2(float(d["pnl"].mean())) if len(d) else "—",
                                "sub": f"{len(d):,} trades" if len(d) else "no trades"})
                rows.append(row)
            return rows

        def strip(g):
            return [[{"text": c["text"], "sub": c["sub"]} for c in row] for row in g] if g else g
        gd = fa.get("fullDow", {}).get("grids") or {}
        w_tr, w_te = want_grid(nomon_all[nomon_all["d"] <= cut]), want_grid(nomon_all[nomon_all["d"] > cut])
        check(strip(gd.get("train")) == w_tr, "Full view: the LEFT pivot is the train trades, every cell (avg P/L, trades) = pandas",
              json.dumps([strip(gd.get("train")), w_tr]))
        check(strip(gd.get("test")) == w_te, "Full view: the RIGHT pivot is the test trades, every cell = pandas",
              json.dumps([strip(gd.get("test")), w_te]))
        check(gd.get("testRowLabels") == 0 and len(gd.get("test") or []) == len(gd.get("train") or [None]),
              "the test pivot repeats no row labels and has the train pivot's rows", json.dumps(gd.get("testRowLabels")))
        check(len(gd.get("heads", [])) == 2 and gd["heads"][0].startswith("Train ·") and gd["heads"][1].startswith("Test ·"),
              f"the two heads: {gd.get('heads')}")
        # One scale: only the single strongest |value| across BOTH grids is at full strength.
        alph = lambda bg: float(bg.rsplit(",", 1)[1].rstrip(")")) if bg.startswith("rgba") else 1.0  # noqa: E731
        cells = [c for g in (gd.get("train") or [], gd.get("test") or []) for row in g for c in row if c["text"] not in ("", "—")]
        vals = [abs(float(c["text"].replace("$", "").replace(",", ""))) for c in cells]
        full_strength = [c for c in cells if alph(c["bg"]) >= 0.999]
        top = max(vals) if vals else None
        check(len(full_strength) == vals.count(top) and all(abs(float(c["text"].replace("$", "").replace(",", ""))) == top
                                                             for c in full_strength),
              "one shading scale across both pivots: only the largest |value| of the two is at full strength",
              json.dumps([(c["text"], c["bg"]) for c in cells]))
        tg = fa.get("trainDow", {}).get("grids") or {}
        check(tg.get("test") is None and not tg.get("heads") and strip(tg.get("train")) == w_tr,
              "Train view: one pivot -- the train trades -- and no test pivot in the DOM", json.dumps(tg)[:300])

    no_bad_text(A, label)
    print()


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--log"), ap.add_argument("--runs"), ap.add_argument("--grep")
    a = ap.parse_args()
    SCOPE["log"], SCOPE["grep"] = a.log, a.grep
    if a.runs:
        SCOPE["runs"] = {r.strip().upper() for r in a.runs.split(",")}
    scoped = bool(a.log or a.runs or a.grep)
    browser = N.find_browser()
    if browser is None:
        print("  SKIP  no Edge/Chromium on this host -- the page was NOT driven")
        return EXIT_SKIPPED
    logs = [("one log (60 trades through 2024)", lambda: {"split_log.csv": S.make_csv(2)}, False),
            ("a six-trade log", lambda: {"tiny_log.csv": S.make_csv(7, n=6)}, False),
            ("a 4-variant sweep (ALL, then one variant)", lambda: {n: S.make_csv(i) for i, n in enumerate(S.NAMES)}, True)]
    try:
        for label, files, variant in logs:
            if SCOPE["log"] and SCOPE["log"] not in label:
                continue
            check_log(browser, label, files(), variant=variant or None)
        if not SCOPE["log"] or SCOPE["log"] in "saving the split":
            check_saved(browser)
    except Exception as exc:                              # noqa: BLE001
        import traceback
        traceback.print_exc()
        check(False, f"the gate raised {type(exc).__name__}: {exc}")
    if scoped:
        if not MATCHED[0]:
            check(False, f"the scope (log={a.log}, runs={a.runs}, grep={a.grep}) matched NO assertion -- it proves nothing")
        print(f"OO split UI, SCOPED (log={a.log}, runs={a.runs}, grep={a.grep}): "
              f"{MATCHED[0]} assertions in scope, failures: {len(FAILS)}")
        return 1 if FAILS else 0
    print(f"OO split UI: {N_OK[0] + len(FAILS)} assertions, failures: {len(FAILS)}")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
