"""Backtest Portfolio P1: the load path, and the cache under it.

WHAT IS AT STAKE. This page exists to combine SAVED strategies, and a saved
strategy is a FILE that gets re-parsed on every load — ~3.0 s for a
2,100-trade MesoSim log on the development machine, and the VPS has measured
about half this speed on comparable work. Five of those is a page nobody
waits for, so the parse is cached. A cache on numbers people trade from has
exactly one interesting failure: serving something that is no longer what the
parser would produce.

So the property under test is not "the cache is fast". It is:

    the payload built from a CACHED parse is identical, byte for byte, to
    the payload built by parsing the file again

Identical at the PAYLOAD, because that is what reaches the browser and what
every later number on both pages is computed from. A frame that differs in
some dtype nothing reads would be a curiosity; a payload that differs is two
different answers for one file.

And the key: the cache is keyed on the file's sha AND a fingerprint of the
parser's own source, so an edit to data_loader.py invalidates every entry.
That is deliberately not a constant someone bumps by hand — the failure it
prevents is a stale parse that looks plausible.

NO DATABASE NEEDED. The store is faked; what is exercised is the cache
policy, the encoding, and the shape the page is handed.
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.oo_backtest import data_loader, parsed_cache           # noqa: E402
from app.oo_backtest.payload import trades_to_payload           # noqa: E402

FIXTURES = ROOT / "scripts" / "fixtures" / "mesosim"
FAILS: list[str] = []


def check(cond, msg):
    if not cond:
        FAILS.append(msg)


class FakeStore:
    """The two columns the cache touches, per strategy id."""

    def __init__(self):
        self.rows: dict[int, dict] = {}
        self.reads = 0
        self.writes = 0

    async def read_parsed(self, pool, sid, *, sha, fingerprint):
        self.reads += 1
        r = self.rows.get(sid)
        if not r or r["sha"] != sha or r["fingerprint"] != fingerprint:
            return None
        return r["blob"]

    async def write_parsed(self, pool, sid, *, sha, fingerprint, blob, rows):
        self.writes += 1
        self.rows[sid] = {"sha": sha, "fingerprint": fingerprint,
                          "blob": blob, "rows": rows}


def fixture() -> tuple[bytes, str]:
    f = sorted(FIXTURES.glob("*.json"))[0]
    return f.read_bytes(), f.name


def parse(content, filename):
    return data_loader.parse_upload(content, filename)


# ── the round trip ──────────────────────────────────────────────────────────
def case_payload_survives_the_cache():
    """A cached parse produces the SAME payload as parsing again.

    The whole point. If this ever differs, the portfolio page and the
    single-backtest page are showing two different readings of one file, and
    which one you got depends on whether the cache happened to be warm.
    """
    content, name = fixture()
    fresh = parse(content, name)
    revived = parsed_cache.decode(parsed_cache.encode(fresh))

    check(list(revived.columns) == list(fresh.columns),
          f"columns changed through the cache: "
          f"{set(fresh.columns) ^ set(revived.columns)}")
    check(len(revived) == len(fresh),
          f"{len(revived)} rows out, {len(fresh)} in")

    a = json.dumps(trades_to_payload(fresh), sort_keys=True, default=str)
    b = json.dumps(trades_to_payload(revived), sort_keys=True, default=str)
    if a != b:
        # Name the first column that differs, or the message is useless.
        pa, pb = trades_to_payload(fresh), trades_to_payload(revived)
        bad = [c for c in pa["columns"]
               if pa["columns"].get(c) != pb["columns"].get(c)]
        FAILS.append(f"the payload differs after a cache round trip; first "
                     f"columns affected: {bad[:4]}")


def case_dates_and_nulls_survive():
    """The two things a JSON round trip breaks if nobody looks.

    Dates must come back as datetimes (the market join and every date filter
    depend on it) and missing values must come back as NaN/NaT rather than
    the string "nan", which compares as a value and would silently pass
    filters.
    """
    import pandas as pd
    content, name = fixture()
    fresh = parse(content, name)
    revived = parsed_cache.decode(parsed_cache.encode(fresh))
    for col in ("date_opened", "date_closed"):
        check(pd.api.types.is_datetime64_any_dtype(revived[col]),
              f"{col} came back as {revived[col].dtype}, not a datetime — the "
              f"market join would not match a single trade")
        check(list(revived[col]) == list(fresh[col]),
              f"{col} values changed through the cache")
    # A null in a float column must be null, not the STRING 'nan'.
    nulls = [c for c in fresh.columns if fresh[c].isna().any()]
    for c in nulls[:6]:
        vals = list(revived[c])
        check(not any(isinstance(v, str) and v.lower() in ("nan", "nat")
                      for v in vals),
              f"{c} has a missing value that came back as the string 'nan'; "
              f"it would pass a numeric filter as a value")


# ── the key ─────────────────────────────────────────────────────────────────
def case_fingerprint_follows_the_parser():
    """The cache key changes when the parser does, without anyone bumping it."""
    import hashlib
    src = (ROOT / "app" / "oo_backtest" / "data_loader.py").read_bytes()
    want = hashlib.sha256(src).hexdigest()[:16]
    check(parsed_cache.FINGERPRINT == want,
          f"the fingerprint {parsed_cache.FINGERPRINT!r} is not a hash of "
          f"data_loader.py ({want!r}) — a parser change would go on serving "
          f"the old parse, and the stale numbers would look plausible")
    check(len(parsed_cache.FINGERPRINT) >= 12,
          "the fingerprint is too short to be distinct")


def case_cache_is_used_and_invalidated():
    """Second load reads; a changed file or parser does not."""
    content, name = fixture()
    store = FakeStore()
    parsed_cache.store = store                      # the fake, for this case
    calls = {"n": 0}

    def counting_parse(c, n):
        calls["n"] += 1
        return parse(c, n)

    async def run():
        sha = "sha-of-the-file"
        first = await parsed_cache.load(object(), strategy_id=7, sha=sha,
                                        content=content, filename=name,
                                        parse=counting_parse)
        second = await parsed_cache.load(object(), strategy_id=7, sha=sha,
                                         content=content, filename=name,
                                         parse=counting_parse)
        # A DIFFERENT FILE under the same id: a save replaced it.
        third = await parsed_cache.load(object(), strategy_id=7, sha="other-sha",
                                        content=content, filename=name,
                                        parse=counting_parse)
        return first, second, third

    first, second, third = asyncio.run(run())
    check(first[1]["source"] == "parsed",
          f"the first load did not parse: {first[1]}")
    check(store.writes == 2,
          f"{store.writes} cache writes across three loads; expected two — "
          f"the first parse and the one whose file had changed")
    check(second[1]["source"] == "cache",
          f"the second load parsed again ({second[1]}); the cache is not "
          f"being used and every portfolio load pays full price")
    check(calls["n"] == 2,
          f"the parser ran {calls['n']} times for three loads; expected two "
          f"(the first, and the one whose file changed)")
    check(third[1]["source"] == "parsed",
          f"a changed file was served from the cache ({third[1]}) — the page "
          f"would show the OLD file's trades under the new file's name")
    check(len(second[0]) == len(first[0]),
          "the cached frame has a different number of rows")

    # THE REAL QUERY, because everything above ran against a fake store and
    # a fake cannot be wrong in the way the SQL can. Planting the removal of
    # `parsed_for_sha` from the WHERE clause passed every behavioural case
    # here: the fake still checked it. Both keys must be IN THE STATEMENT.
    store_src = (ROOT / "app" / "oo_backtest" / "store.py").read_text(encoding="utf-8")
    start = store_src.index("async def read_parsed")
    sql = store_src[start:store_src.index("async def write_parsed")]
    for key in ("parsed_for_sha = $2", "parsed_fingerprint = $3"):
        check(key in sql,
              f"read_parsed's query does not test {key.split(' =')[0]} — a "
              f"cache entry made from a different file, or by a different "
              f"parser, would be served as this one's")
    wsql = store_src[store_src.index("async def write_parsed"):]
    check("file_sha256 = $3" in wsql,
          "write_parsed does not check the file is still the one it parsed; "
          "a save during a parse would file the OLD file's trades against "
          "the new one")

    # A parser change invalidates every entry, without touching the rows.
    old = parsed_cache.FINGERPRINT
    try:
        parsed_cache.FINGERPRINT = "deadbeefdeadbeef"

        async def again():
            return await parsed_cache.load(object(), strategy_id=7,
                                           sha="sha-of-the-file",
                                           content=content, filename=name,
                                           parse=counting_parse)
        out = asyncio.run(again())
        check(out[1]["source"] == "parsed",
              "a new parser fingerprint still read the old cache entry")
    finally:
        parsed_cache.FINGERPRINT = old


def case_a_broken_entry_is_survived():
    """Corrupt bytes cost a parse, never a failed load.

    The cache is derived data. There is nothing to recover and nothing to
    report to the person looking at the page — but it must not be possible
    for a bad row to take the page down.
    """
    content, name = fixture()
    store = FakeStore()
    parsed_cache.store = store
    store.rows[3] = {"sha": "s", "fingerprint": parsed_cache.FINGERPRINT,
                     "blob": b"not gzip at all", "rows": 0}

    async def run():
        return await parsed_cache.load(object(), strategy_id=3, sha="s",
                                       content=content, filename=name,
                                       parse=parse)
    df, note = asyncio.run(run())
    check(note["source"] == "parsed" and len(df) > 0,
          f"a corrupt cache entry did not fall back to parsing: {note}")
    check(store.writes == 1, "the corrupt entry was not replaced")


# ── what the page is handed ─────────────────────────────────────────────────
def case_the_router_reuses_one_path():
    """The portfolio load goes through the OO page's parse/join/payload.

    Source-level, because the alternative is two code paths producing two
    sets of trade numbers for one file — and the cheapest way for that to
    happen is someone writing a second, simpler loader here.
    """
    src = (ROOT / "app" / "routers" / "backtest_portfolio.py").read_text(encoding="utf-8")
    # IMPORTED **AND CALLED**. A check for the import line alone passed a
    # planted `import _analyze as _a`, because the old name is a substring of
    # the new line — the same trap that got the wall's faded-class check.
    check("from app.routers.oo_backtest import _analyze, load_parsed" in src,
          "the portfolio router does not import the OO page's _analyze and "
          "load_parsed; a second parse/join/payload path is two answers for "
          "one file")
    check("await _analyze(content" in src and "await load_parsed(pool" in src,
          "the portfolio router imports the shared load path and does not "
          "call it")
    check("parse_upload" not in src and "trades_to_payload" not in src,
          "the portfolio router parses or builds a payload of its own")
    for token in ("qty", "COLORS", "MAX_STRATEGIES"):
        check(token in src, f"the router no longer defines {token}")
    # qty scales P/L in the BROWSER. Nothing server-side may multiply a P/L:
    # two places that scale is a portfolio silently squared.
    check("pnl" not in src.replace("pnl_per", ""),
          "the portfolio router touches pnl; qty scaling belongs in the page")

    js = (ROOT / "static" / "js" / "backtest_portfolio.js").read_text(encoding="utf-8")
    check("bpTotalCapital" in js and "qty" in js,
          "the page does not carry qty and capital")
    check("date_min" in js and "union" in js,
          "the page does not derive a span, or does not default to union")


def case_the_page_is_wired_up():
    html = (ROOT / "templates" / "backtest_portfolio.html").read_text(encoding="utf-8")
    check('x-data="backtestPortfolio"' in html, "the page declares no component")
    check("backtest_portfolio.js" in html, "the page loads no bundle")
    check("css/backtest.css" in html,
          "the page does not read the shared backtest stylesheet, so it will "
          "drift from its sibling")
    oo = (ROOT / "templates" / "oo_backtest.html").read_text(encoding="utf-8")
    check("css/backtest.css" in oo,
          "the OO page no longer reads the shared stylesheet it now depends on")
    shared = (ROOT / "static" / "css" / "backtest.css").read_text(encoding="utf-8")
    for rule in (".ob-card", ".ob-side", ".ob-stats", ".ob-sel"):
        check(rule in shared, f"{rule} is missing from the shared stylesheet")
        # And is NOT still defined inline on the OO page, or the two copies
        # are exactly the drift this file exists to prevent.
        check(f"\n    {rule} " not in oo and f"\n    {rule}{{" not in oo,
              f"{rule} is defined both inline on the OO page and in the "
              f"shared stylesheet")
    main = (ROOT / "app" / "main.py").read_text(encoding="utf-8")
    check('"/backtest-portfolio"' in main and "backtest_portfolio.router" in main,
          "the page or its API is not registered")



# ── P2: what the page computes ──────────────────────────────────────────────
#
# The SHIPPED functions, executed. These are the definitions the user settled
# and they are the ones a portfolio is read from, so they are checked as
# arithmetic rather than by looking at a table and finding it plausible.

JS_DRIVER = r"""
const fs = require('fs');
global.document = { addEventListener: () => {} };
global.Alpine = { data: () => {} };
const core = fs.readFileSync(process.argv[1], 'utf8');
const page = fs.readFileSync(process.argv[2], 'utf8');
const tail = `
  const out = {};
  const sessions = [];
  for (let d = new Date('2023-01-02'); d < new Date('2023-04-01');
       d.setUTCDate(d.getUTCDate() + 1)) {
    const wd = d.getUTCDay();
    if (wd !== 0 && wd !== 6) sessions.push(d.toISOString().slice(0, 10));
  }
  // A WEEKLY ROLL: open Monday, close the next Monday, over and over.
  const op = [], cl = [], pnl = [];
  for (let k = 0; k < 8; k++) {
    const a = new Date('2023-01-02'); a.setUTCDate(a.getUTCDate() + k * 7);
    const b = new Date('2023-01-02'); b.setUTCDate(b.getUTCDate() + (k + 1) * 7);
    op.push(a.toISOString().slice(0, 10));
    cl.push(b.toISOString().slice(0, 10));
    pnl.push(k % 2 ? -100 : 300);
  }
  const cols = { date_opened: op, date_closed: cl, pnl: pnl,
                 days_in_trade: op.map(() => 7),
                 vix_level: op.map((_, i) => (i < 3 ? null : 15 + i)) };
  const all = [...op.keys()];

  // HALF-OPEN DEPLOYMENT: one position at a time, never two on a roll day.
  const conc = obConcurrency(cols, all, sessions);
  out.peak = conc.peak;
  out.counts = [...new Set(conc.counts)].sort();
  out.sameSession = conc.sameSession;

  // qty scales P/L and the capital behind it.
  const scaled = Object.assign({}, cols, { pnl: cols.pnl.map(v => v * 3) });
  out.total1 = obStats(cols, all).total_pnl;
  out.total3 = obStats(scaled, all).total_pnl;
  out.capital = bpTotalCapital([{ qty: 3, capital: 10000 },
                                { qty: 1, capital: 25000 }]);

  // The deployed series sums across strategies; the peak of the sum is not
  // the sum of the peaks unless they peak together.
  const s1 = obDeployedSeries(conc, sessions, 10000);
  out.seriesPeak = Math.max(...s1);
  const doubled = s1.map(v => v * 2);
  out.sumPeak = Math.max(...doubled.map((v, i) => v + s1[i] * 0));

  // Sharpe: the old app's definition, over days that HAD a close.
  out.sharpe = obSharpe(cols, all);
  const byDay = new Map();
  for (const i of all) byDay.set(cl[i], (byDay.get(cl[i]) || 0) + pnl[i]);
  const v = [...byDay.values()];
  const mean = v.reduce((a, b) => a + b, 0) / v.length;
  let ss = 0; for (const x of v) ss += (x - mean) * (x - mean);
  out.sharpeWant = mean / Math.sqrt(ss / (v.length - 1)) * Math.sqrt(252);
  out.sharpeDays = v.length;

  // The registry drives the filters: OFF contributes no spec, ON does.
  const reg = [{ key: 'vix', column: 'vix_level', type: 'range', min: 9, max: 80 },
               { key: 'dow', column: 'day_of_week', type: 'categorical' }];
  out.specsOff = bpSpecs({ vix: { on: false, lo: 9, hi: 80 } }, reg, null).length;
  const specsOn = bpSpecs({ vix: { on: true, lo: 9, hi: 80 } }, reg, null);
  out.specsOn = specsOn.length;
  out.specColumn = specsOn[0].column;
  // A filter on a late-starting metric drops the trades with no value.
  out.filtered = obApplyFilters(cols, op.length, specsOn).length;
  // The date range narrows on the ENTRY date.
  const dated = bpSpecs({}, reg, { start: op[2], end: op[5] });
  out.dateKind = dated[0].kind + ':' + dated[0].column;
  out.dateFiltered = obApplyFilters(cols, op.length, dated).length;

  // Union and intersection of two spans.
  const ps = [{ date_min: '2023-01-01', date_max: '2023-06-30' },
              { date_min: '2023-03-01', date_max: '2023-12-31' }];
  out.union = bpSpan(ps, 'union').start + '..' + bpSpan(ps, 'union').end;
  out.inter = bpSpan(ps, 'intersection').start + '..' + bpSpan(ps, 'intersection').end;
  globalThis.__out = out;
`;
eval(core + String.fromCharCode(10) + page + String.fromCharCode(10) + tail);
process.stdout.write(JSON.stringify(globalThis.__out));
"""


def case_p2_arithmetic():
    """qty, capital, half-open deployment, Sharpe, and the registry filters."""
    import shutil
    import subprocess
    if shutil.which("node") is None:
        FAILS.append("node is not installed — the shipped page JS was NOT run")
        return
    core = ROOT / "static" / "js" / "backtest_core.js"
    page = ROOT / "static" / "js" / "backtest_portfolio.js"
    p = subprocess.run(["node", "-e", JS_DRIVER, str(core), str(page)],
                       capture_output=True, text=True, encoding="utf-8")
    if p.returncode:
        FAILS.append(f"the page JS did not run: {p.stderr.strip()[:300]}")
        return
    out = json.loads(p.stdout)

    # HALF-OPEN: a weekly roll is ONE position, including on the day it rolls.
    check(out["peak"] == 1,
          f"a weekly roll peaks at {out['peak']} concurrent positions; "
          f"half-open [open, close) means the position closing and the one "
          f"opening on the same day are never both counted")
    check(out["counts"] == [0, 1] or out["counts"] == [1],
          f"the roll's daily counts are {out['counts']}, expected 1 "
          f"throughout (0 only after the final close)")
    check(out["sameSession"] == 0,
          f"{out['sameSession']} weekly trades were counted as intraday")

    # qty scales P/L linearly and multiplies the capital behind it.
    check(out["total3"] == out["total1"] * 3,
          f"qty 3 gave {out['total3']} against {out['total1']} at qty 1")
    check(out["capital"] == 55000,
          f"portfolio capital {out['capital']}, expected 3x10,000 + 1x25,000 "
          f"= 55,000 — qty multiplies the capital behind the position")

    # Sharpe is the old app's, and it is over days that had a close.
    check(out["sharpe"] is not None
          and abs(out["sharpe"] - out["sharpeWant"]) < 1e-9,
          f"Sharpe {out['sharpe']} does not match mean/stdev x sqrt(252) over "
          f"P/L summed by close date ({out['sharpeWant']})")
    check(out["sharpeDays"] == 8,
          f"Sharpe used {out['sharpeDays']} observations for 8 closes; the "
          f"days between closes must NOT be zero-filled (that is the old "
          f"app's definition, kept deliberately)")

    # The registry drives the filters, and a late metric costs trades.
    check(out["specsOff"] == 0 and out["specsOn"] == 1
          and out["specColumn"] == "vix_level",
          f"an off filter produced {out['specsOff']} specs and an on one "
          f"{out['specsOn']} on {out['specColumn']!r}")
    check(out["filtered"] == 5,
          f"filtering on a metric with three null values kept "
          f"{out['filtered']} of 8; a trade with no value cannot satisfy it")
    check(out["dateKind"] == "date:date_opened",
          f"the portfolio date range applies to {out['dateKind']}, not the "
          f"entry date — a trade belongs to the window it was opened in")
    check(out["dateFiltered"] == 4,
          f"the date range kept {out['dateFiltered']} of 8")

    # Union and intersection.
    check(out["union"] == "2023-01-01..2023-12-31",
          f"union is {out['union']}")
    check(out["inter"] == "2023-03-01..2023-06-30",
          f"intersection is {out['inter']}")


def case_the_page_computes_nothing_of_its_own():
    """Every statistic comes from the shared core, not from this page.

    The portfolio's whole claim is that a strategy reads the same on both
    pages. A statistic reimplemented here would break that quietly, and the
    cheapest way for it to happen is someone adding "just one more column".
    """
    js = (ROOT / "static" / "js" / "backtest_portfolio.js").read_text(encoding="utf-8")
    for fn in ("obStats", "obExtraStats", "obConcurrency", "obApplyFilters",
               "obSharpe", "obDeployedSeries"):
        check(f"{fn}(" in js, f"the page does not call {fn}")
        check(f"function {fn}" not in js,
              f"the page defines its own {fn}; it must come from "
              f"backtest_core.js, which /oo-backtest reads too")
    core = (ROOT / "static" / "js" / "backtest_core.js").read_text(encoding="utf-8")
    oo = (ROOT / "static" / "js" / "oo_backtest.js").read_text(encoding="utf-8")
    for fn in ("obStats", "obExtraStats", "obConcurrency", "obApplyFilters"):
        check(f"function {fn}" in core, f"{fn} is not in the shared core")
        check(f"function {fn}" not in oo,
              f"{fn} is still defined in oo_backtest.js as well as the core")
    html = (ROOT / "templates" / "backtest_portfolio.html").read_text(encoding="utf-8")
    check("js/backtest_core.js" in html,
          "the portfolio page does not load the shared calculations")
    oo_html = (ROOT / "templates" / "oo_backtest.html").read_text(encoding="utf-8")
    check("js/backtest_core.js" in oo_html,
          "the OO page does not load the shared calculations it now depends on")



def case_filters_live_in_the_main_column():
    """Filter editing is in the main column; the sidebar keeps qty and capital.

    THE REASON, so it is not undone by tidying: the sidebar is ~400px and
    permanent, filter editing is occasional and wants width. Stacked one per
    row in the sidebar the nine metrics were already tight and the list
    grows; in the main column they run two or more abreast. qty and capital
    stay inline in the sidebar rows because they are adjusted repeatedly
    while watching the table, which is the opposite case.
    """
    html = (ROOT / "templates" / "backtest_portfolio.html").read_text(encoding="utf-8")
    side = html.split('class="ob-side"')[1].split('class="ob-main"')[0]
    main = html.split('class="ob-main"')[1]

    check("bp-fgrid" in main and "bp-fgrid" not in side,
          "the filter panel is not in the main column")
    check("ob-dual" in main and "ob-dual" not in side,
          "there are range sliders in the sidebar; that is where they used "
          "to be and the point of the move was to get them out")
    # qty and capital stay where they are.
    check('x-model.number="c.qty"' in side and 'x-model.number="c.capital"' in side,
          "qty and capital left the sidebar rows; they are adjusted "
          "repeatedly while watching the table and belong beside it")
    # One panel at a time, and only when open.
    check('<template x-if="editing">' in main,
          "the panel is not x-if'd on `editing` — with x-show it stays in the "
          "document for a strategy that is not being edited, and every "
          "expression inside it evaluates against null")
    # The table stays visible while the panel is worked.
    check("bp-summary" in main and "position:sticky" in html,
          "the summary table does not pin, so working a filter scrolls the "
          "numbers it is supposed to move off the screen")
    # The controls are the shared ones, not a second set.
    css = (ROOT / "static" / "css" / "backtest.css").read_text(encoding="utf-8")
    for rule in (".ob-dual", ".ob-checks", ".ob-range-vals"):
        check(rule in css, f"{rule} is not in the shared stylesheet")
    oo = (ROOT / "templates" / "oo_backtest.html").read_text(encoding="utf-8")
    check(".ob-dual {" not in oo,
          "the OO page still defines the dual slider inline as well as "
          "reading it from the shared sheet")


CASES = [
    ("payload survives the cache", case_payload_survives_the_cache),
    ("dates and nulls survive",    case_dates_and_nulls_survive),
    ("fingerprint follows parser", case_fingerprint_follows_the_parser),
    ("cache used and invalidated", case_cache_is_used_and_invalidated),
    ("a broken entry is survived", case_a_broken_entry_is_survived),
    ("one load path",              case_the_router_reuses_one_path),
    ("the page is wired up",       case_the_page_is_wired_up),
    ("P2 arithmetic",              case_p2_arithmetic),
    ("filters in the main column", case_filters_live_in_the_main_column),
    ("no second implementation",   case_the_page_computes_nothing_of_its_own),
]


def main() -> int:
    real_store = parsed_cache.store
    for name, fn in CASES:
        before = len(FAILS)
        try:
            fn()
        except Exception as exc:                          # noqa: BLE001
            FAILS.append(f"raised {type(exc).__name__}: {exc}")
        finally:
            parsed_cache.store = real_store
        for m in FAILS[before:]:
            if m:
                print(f"  FAIL {name}: {m}")
    real = [m for m in FAILS if m]
    print(f"\nportfolio cases: {len(CASES)}, failures: {len(real)}")
    return 1 if real else 0


sys.exit(main())
