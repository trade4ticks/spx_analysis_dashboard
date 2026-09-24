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


CASES = [
    ("payload survives the cache", case_payload_survives_the_cache),
    ("dates and nulls survive",    case_dates_and_nulls_survive),
    ("fingerprint follows parser", case_fingerprint_follows_the_parser),
    ("cache used and invalidated", case_cache_is_used_and_invalidated),
    ("a broken entry is survived", case_a_broken_entry_is_survived),
    ("one load path",              case_the_router_reuses_one_path),
    ("the page is wired up",       case_the_page_is_wired_up),
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
