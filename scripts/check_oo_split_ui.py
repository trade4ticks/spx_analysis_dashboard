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


def check(ok: bool, msg: str, detail: str = "") -> None:
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
    return pd.concat([data_loader.parse_upload(c.encode(), n) for n, c in files.items()], ignore_index=True)


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
  const snap = async (name, c) => {
    document.documentElement.setAttribute('data-progress', name);
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
  const typeCutoff = async (v) => { const el = cutoffEl(); el.value = v; el.dispatchEvent(new Event('change', { bubbles: true })); await settle(); };
  const typeDate = async (which, v) => {
    const el = document.querySelector(`input[aria-label="Entry date ${which}"]`);
    el.value = v; el.dispatchEvent(new Event('input', { bubbles: true })); el.dispatchEvent(new Event('change', { bubbles: true }));
    await settle();
  };
  const exportText = async () => { const n = window.__blobs.length; button('Export CSV').click(); await until(() => window.__blobs.length > n);
                                   return window.__blobs.length > n ? await window.__blobs[window.__blobs.length - 1].text() : null; };
  const rank = async (c) => { const n = window.__ranks.length; (button('Rank metrics') || button('Recompute')).click();
                              await until(() => window.__ranks.length > n && !c.surf.busy); await settle();
                              return window.__ranks.length > n ? window.__ranks[window.__ranks.length - 1] : null; };
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
                      cutoffShown: cutoffEl() ? cutoffEl().value : null, counts: counts(), status: status(), sumSub: sumSub() };
    await snap('train', c);
    seg('Full').click(); await settle();
    facts.afterFull = { counts: counts(), status: status(), sumSub: sumSub() };
    await snap('full', c);
    seg('Off').click(); await settle();
    await snap('off again', c);

    // ── Train view, a clicked filter ─────────────────────────────────────
    seg('On').click(); await settle();
    const mon = [...card('Day of Week').querySelectorAll('.ob-check')].find(l => txt(l).startsWith('Mon'));
    facts.dowEnabled = !mon.querySelector('input').disabled;
    const before = c.filteredCount;
    mon.querySelector('input').click(); await until(() => c.filteredCount !== before); await settle();
    facts.trainDow = { counts: counts(), status: status() };
    await snap('train dow', c);
    facts.csvTrain = await exportText();
    facts.rankTrain = await rank(c);
    facts.rankSubTrain = rankSub();
    await snap('train dow ranked', c);

    // ── Full view ────────────────────────────────────────────────────────
    seg('Full').click(); await settle();
    facts.fullDow = { counts: counts(), status: status(), stale: c.surfStale() };
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
      facts.fullVariant = { counts: counts(), status: status() };
      sel.value = 'all'; sel.dispatchEvent(new Event('change', { bubbles: true }));
      await until(() => c.variantSel === 'all'); await settle();
    }

    // ── the date range moves the universe, never the cutoff ─────────────
    await typeDate('from', FIX.dateFrom);
    facts.afterDate = { cutoff: cutoffEl().value, counts: counts(), status: status(), sumSub: sumSub() };
    seg('Train').click(); await settle();
    facts.afterDateTrain = { cutoff: cutoffEl().value, sumSub: sumSub(), status: status() };
    seg('Full').click(); await settle();
    await typeDate('from', c.dateBounds.min);

    // ── typed cutoffs ─────────────────────────────────────────────────────
    facts.cutoffs = [];
    for (const co of FIX.cutoffs) {
      await typeCutoff(co);
      const full = { cutoff: cutoffEl().value, counts: counts(), status: status() };
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


def drive(browser: str, fix: dict, exclude: str = "__nothing__") -> dict:
    html = N.build_page()
    html = html.replace(N.DRIVER.replace("__SAVED__", json.dumps(N.SAVED)), "")
    page = html.replace("</head>", DRIVER.replace("__SNAPSHOT__", O.SNAPSHOT_JS).replace("__FIX__", json.dumps(fix))
                        .replace("__EXCLUDE__", exclude) + "</head>", 1)
    import os
    tmp = ROOT / "scripts" / f"_oo_split_ui_{os.getpid()}.html"   # unique: a gate may run beside itself
    tmp.write_text(page, encoding="utf-8")
    import tempfile
    prof = tempfile.mkdtemp(prefix="edge-gate-")
    try:
        p = subprocess.run(
            # A PRIVATE profile per run: gates run six at a time, and a second
            # Edge on the default profile hands its page to the first and exits
            # with nothing -- "the page never reported".
            [browser, "--headless=new", "--disable-gpu", "--window-size=1600,1200", f"--user-data-dir={prof}",
             "--dump-dom", "--virtual-time-budget=90000", tmp.as_uri()],
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=300)
    finally:
        tmp.unlink(missing_ok=True)
        import shutil
        shutil.rmtree(prof, ignore_errors=True)
    m = re.search(r'<pre id="report">(.*?)</pre>', p.stdout, re.S)
    if not m:
        prog = re.search(r'data-progress="([^"]*)"', p.stdout)
        alpine = "Alpine loaded" if "_x_dataStack" in p.stdout or 'x-data="ooBacktest"' in p.stdout else ""
        raise RuntimeError("the page never reported -- it did not get that far: "
                           f"last step {prog.group(1) if prog else '(none: before the first snapshot)'}, "
                           f"{len(p.stdout):,} bytes of DOM, exit {p.returncode}, stderr: {(p.stderr or '').strip()[-300:]}")
    out = json.loads(_html.unescape(m.group(1)))
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
    facts_txt = json.dumps({k: v for k, v in run["facts"].items() if k not in ("csvTrain", "csvFull", "rankTrain", "rankFull")})
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
    if variant:
        variant = str(pa["sweep"]["variants"][2]["id"])
    base = {"registry": registry_with_coverage(None), "catalog": CATALOG, "rankRows": RANK_ROWS, "names": list(files),
            "cutoffs": cutoffs, "dateFrom": date_from, "outside": outside, "variant": variant}
    A = drive(browser, {**base, "payload": pa})
    B = drive(browser, {**base, "payload": payload(b_files)})
    C = drive(browser, {**base, "payload": payload(c_files)})
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
    ok, why = same(step(A, "off"), step(A, "full"), parts=("charts", "state"))
    check(ok, "Full view's charts and figures are exactly the Off page's (only headers differ)", why)

    # ── THE DIFFERENTIAL ─────────────────────────────────────────────────
    # B and C change only trades entered after the DEFAULT cutoff, so the
    # differential holds wherever the train period ends on or before it; at
    # a later typed cutoff those trades are train, and should show.
    def at_or_before_default(name):
        m = re.search(r"train at (\d{4}-\d{2}-\d{2})", name)
        return not m or m.group(1) <= cut
    train_steps = [s["name"] for s in A["steps"] if s["name"].startswith("train") and at_or_before_default(s["name"])]
    later = [s["name"] for s in A["steps"] if s["name"].startswith("train at") and not at_or_before_default(s["name"])]
    for name in later:
        ok, _ = same(step(A, name), step(B, name), parts=("state",))
        check(not ok, f"'{name}' (after the default cutoff): the changed trades ARE train there, and show")
    for name in train_steps:
        ok, why = same(step(A, name), step(B, name))
        check(ok, f"Train view '{name}': changing every TEST trade's P/L and exit reason changes nothing on the "
                  f"page (DOM, titles, charts, figures)", why)
    check(fa.get("csvTrain") == B["facts"].get("csvTrain"), "Train view: the Export CSV is identical with the test trades changed")
    check(fa.get("rankTrain") == B["facts"].get("rankTrain"), "Train view: the /surface/rank request is identical with the test trades changed")
    ok, _ = same(step(A, "full"), step(B, "full"), parts=("state",))
    check(not ok, "...and the same change DOES show in Full view (the differential can see test trades)")
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

    no_bad_text(A, label)
    print()


def main() -> int:
    browser = N.find_browser()
    if browser is None:
        print("  SKIP  no Edge/Chromium on this host -- the page was NOT driven")
        return EXIT_SKIPPED
    try:
        check_log(browser, "one log (60 trades through 2024)", {"split_log.csv": S.make_csv(2)})
        check_log(browser, "a six-trade log", {"tiny_log.csv": S.make_csv(7, n=6)})
        sweep = {n: S.make_csv(i) for i, n in enumerate(S.NAMES)}
        check_log(browser, "a 4-variant sweep (ALL, then one variant)", sweep, variant=True)
    except Exception as exc:                              # noqa: BLE001
        import traceback
        traceback.print_exc()
        check(False, f"the gate raised {type(exc).__name__}: {exc}")
    print(f"OO split UI: {N_OK[0] + len(FAILS)} assertions, failures: {len(FAILS)}")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
