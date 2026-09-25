"""OO/Mesosim Backtest: the browser must bin exactly as pandas does.

WHY. The page filters and re-bins in JavaScript, but the bin boundaries are
tuned and are defined once, in Python (app/oo_backtest/calculations.py). If
the JS disagreed with pd.cut about which side of an edge a value sits on,
every section would still render, the totals would still add up, and a trade
at VIX 20.00 would sit in the 19 bar. Nothing on screen could reveal that.

So this runs the SHIPPED obBinIndex() in node against every range metric in
the registry, at every edge, next to every edge and at random, and compares
bin-for-bin with the create_*_bins helper the pandas path uses. The ratio
bins are right-closed where the others are left-closed, which is exactly the
kind of difference a port gets wrong; the check plants that fault and requires
it to be caught before it trusts a pass.

Also checked, against fabricated logs rather than files on disk:
  * both vendor formats parse, in each spelling the loader handles
    (tab/comma CSV; StrategyName/BacktestName; pos_pnl/pos_realized_pnl)
  * the payload carries ISO date strings, open-date order, and only
    whitelisted columns -- a planted vrp_calc column must not reach it
  * the registry is JSON without NaN/inf, has every field, and its sections
    come in the brief's order
  * no dropped-scope name (SharpTwo, skew, regime) appears in the page JS or
    the registry
  * REAL MesoSim exports (3.1 and 2.13, trimmed into scripts/fixtures/mesosim)
    reduce to one trade per closed position, report open positions, resolve
    two exit signals on one bar to the later, keep the 12:30 early-close
    entry, and flag MissingData / P&L disagreement when planted

    python scripts/check_oo_backtest.py

WHERE IT RUNS: the development machine. It needs `node` (to execute the shipped
JS) and the Options-Backtest-Dashboard checkout beside this repo. The VPS has
neither, deliberately; there this exits 3 (NOT FULLY RUN), never 0.
"""
from __future__ import annotations

import json
import math
import random
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from app.oo_backtest import calculations as calc  # noqa: E402
from app.oo_backtest import data_loader  # noqa: E402
from app.oo_backtest.payload import TRADE_COLUMNS, allowed_column, trades_to_payload  # noqa: E402
from app.oo_backtest.registry import REGISTRY, registry_with_coverage  # noqa: E402
from app.routers.oo_backtest import _parse  # noqa: E402

JS = ROOT / "static" / "js" / "oo_backtest.js"
# The shared half, loaded first by the page and by every driver here.
CORE = ROOT / "static" / "js" / "backtest_core.js"

FAILS: list[str] = []
# Parts this host cannot run (no node, no sibling checkout of the source app).
# Any entry makes the whole check exit EXIT_SKIPPED rather than PASS: a run
# that skipped the JS parity check has not checked JS parity.
NOT_RUN: list[str] = []
EXIT_SKIPPED = 3


def check(ok: bool, msg: str) -> None:
    print(("  ok    " if ok else "  FAIL  ") + msg)
    if not ok:
        FAILS.append(msg)


# Which pandas helper each range metric's column is binned by in the source.
HELPERS = {
    "gap": calc.create_gap_bins,
    "vix_gap": calc.create_vix_gap_bins,
    "vix": calc.create_vix_bins,
    "vix3m": calc.create_vix_bins,
    "vix9d": calc.create_vix_bins,
    "vix3m_vix": calc.create_ratio_bins,
    "vix_vix9d": calc.create_ratio_bins,
}

DRIVER = r"""
const fs = require('fs');
global.document = { addEventListener: () => {} };
// The shared calculations, then the page: the bundle's functions are
// top-level declarations that read them, exactly as the browser loads
// the two script tags.
eval(fs.readFileSync(process.argv[2], 'utf8') + String.fromCharCode(10)
     + fs.readFileSync(process.argv[1], 'utf8'));
const job = JSON.parse(fs.readFileSync(0, 'utf8'));
const out = {};
for (const [key, t] of Object.entries(job)) {
  out[key] = t.values.map(v => obBinIndex(v, t.bins));
}
process.stdout.write(JSON.stringify(out));
"""


def probe_values(bins: dict) -> list:
    rng = random.Random(7)
    e = bins["edges"]
    vals = []
    for x in e:
        vals += [x, math.nextafter(x, -math.inf), math.nextafter(x, math.inf), x - 1e-6, x + 1e-6]
    span = (e[-1] - e[0]) or 1.0
    vals += [rng.uniform(e[0] - span, e[-1] + span) for _ in range(400)]
    # Two-decimal values, as VIX closes and their ratios actually arrive.
    vals += [round(rng.uniform(e[0] - 0.1 * span, e[-1] + 0.1 * span), 2) for _ in range(400)]
    vals += [e[0] - 1e6, e[-1] + 1e6, None]
    return vals


def pandas_codes(metric_key: str, column: str, values: list) -> list[int]:
    df = pd.DataFrame({column: pd.Series([math.nan if v is None else v for v in values], dtype=float)})
    binned = HELPERS[metric_key](df, column)
    labels = {m["key"]: m for m in REGISTRY}[metric_key]["bins"]["labels"]
    idx = {lab: i for i, lab in enumerate(labels)}
    return [-1 if pd.isna(b) else idx[b] for b in binned[f"{column}_bin"]]


def run_js(job: dict) -> dict:
    p = subprocess.run(["node", "-e", DRIVER, str(JS), str(CORE)], input=json.dumps(job),
                       capture_output=True, text=True, encoding="utf-8")
    if p.returncode:
        raise RuntimeError(p.stderr.strip())
    return json.loads(p.stdout)


def check_binning() -> None:
    print("binning: shipped JS vs pd.cut")
    import shutil
    if shutil.which("node") is None:
        print("  SKIP  node is not installed — the shipped JS cannot be executed on this host")
        NOT_RUN.append("JS binning parity (node not installed)")
        return
    # The registry goes through JSON exactly as the endpoint sends it.
    reg = json.loads(json.dumps(REGISTRY, allow_nan=False))
    # Fixed bins only: an auto-binned metric's edges depend on the log and have
    # no pandas helper; check_auto_bins covers those.
    ranges = [m for m in reg if m["type"] == "range" and m["binning"] == "fixed"]
    check(set(HELPERS) == {m["key"] for m in ranges},
          f"every fixed-bin range metric has a pandas helper mapped ({len(ranges)})")

    job, expect = {}, {}
    for m in ranges:
        vals = probe_values(m["bins"])
        job[m["key"]] = {"values": vals, "bins": m["bins"]}
        expect[m["key"]] = pandas_codes(m["key"], m["column"], vals)
    got = run_js(job)
    for m in ranges:
        bad = [i for i, (a, b) in enumerate(zip(got[m["key"]], expect[m["key"]])) if a != b]
        detail = "" if not bad else f" — first at value {job[m['key']]['values'][bad[0]]!r}: js {got[m['key']][bad[0]]} vs pandas {expect[m['key']][bad[0]]}"
        check(not bad, f"{m['key']}: {len(expect[m['key']])} values agree{detail}")

    # Planted fault: flip the ratio bins to left-closed. If the comparison
    # cannot tell, every pass above is vacuous.
    m = next(x for x in ranges if x["key"] == "vix3m_vix")
    flipped = dict(m["bins"], closed="left" if m["bins"]["closed"] == "right" else "right")
    fault = run_js({"f": {"values": job["vix3m_vix"]["values"], "bins": flipped}})["f"]
    check(fault != expect["vix3m_vix"], "a left/right-closed swap on the ratio bins IS detected")

    # A spec without an explicit side must not bin at all.
    unsided = {k: v for k, v in m["bins"].items() if k != "closed"}
    try:
        run_js({"f": {"values": [1.0], "bins": unsided}})
        check(False, "a bin spec with no `closed` side is refused by obBinIndex")
    except RuntimeError as exc:
        check("no closed side" in str(exc), "a bin spec with no `closed` side is refused by obBinIndex")
    check({x["key"]: x["bins"]["closed"] for x in ranges if x["key"] in ("vix3m_vix", "vix_vix9d", "vix")}
          == {"vix3m_vix": "right", "vix_vix9d": "right", "vix": "left"},
          "ratio bins are explicitly right-closed, the others left-closed")


SOURCE_ROOT = ROOT.parent / "Options-Backtest-Dashboard"


def check_against_source() -> None:
    """The spec refactor must bin exactly as the source app's own helpers did.

    Loads the ORIGINAL utils/calculations.py from the sibling checkout and
    compares labels value-for-value. The ratio bins had no helper in the
    source -- they were inline in app.py's render callback -- so that
    expression is reproduced here verbatim. Skips, loudly, where the source
    checkout is absent (the VPS has none).
    """
    print("spec refactor vs the source app's helpers")
    src = SOURCE_ROOT / "utils" / "calculations.py"
    if not src.exists():
        print(f"  SKIP  {src} not present — nothing to compare against on this host")
        NOT_RUN.append("source-app bin comparison (no Options-Backtest-Dashboard checkout)")
        return
    import importlib.util
    sys.path.insert(0, str(SOURCE_ROOT))          # the source does `from config import …`
    try:
        spec = importlib.util.spec_from_file_location("_source_calculations", src)
        orig = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(orig)
    finally:
        sys.path.remove(str(SOURCE_ROOT))

    rng = random.Random(11)
    vals = ([rng.uniform(-25, 90) for _ in range(3000)] + [round(rng.uniform(0.5, 2.0), 2) for _ in range(2000)]
            + [float(x) for x in range(-20, 81)] + [x / 10 for x in range(-30, 31)]
            + [x / 100 for x in range(50, 201)] + [math.nan])
    df = pd.DataFrame({"v": vals})
    pairs = [("gap", orig.create_gap_bins, calc.create_gap_bins),
             ("vix_gap", orig.create_vix_gap_bins, calc.create_vix_gap_bins),
             ("premium", orig.create_premium_bins, calc.create_premium_bins),
             ("vix", orig.create_vix_bins, calc.create_vix_bins)]
    for name, old, new in pairs:
        a = old(df * (100 if name == "premium" else 1), "v")["v_bin"].astype(object)
        b = new(df * (100 if name == "premium" else 1), "v")["v_bin"].astype(object)
        same = all((pd.isna(x) and pd.isna(y)) or x == y for x, y in zip(a, b))
        check(same, f"{name}: identical labels to the source helper over {len(vals)} values")

    # app.py:1112-1118, verbatim.
    ratio_bins = [0] + [0.7 + i * 0.04 for i in range(21)] + [10]
    ratio_labels = ["<0.70"] + [f"{0.7 + i * 0.04:.2f}" for i in range(20)] + [">1.50"]
    a = pd.cut(df["v"], bins=ratio_bins, labels=ratio_labels).astype(object)
    b = calc.create_ratio_bins(df, "v")["v_bin"].astype(object)
    # The only permitted difference: values outside the source's (0, 10],
    # which it dropped and this bins at the ends.
    inside = (df["v"] > 0) & (df["v"] <= 10)
    same = all((pd.isna(x) and pd.isna(y)) or x == y for x, y, k in zip(a, b, inside) if k)
    check(same, "ratio: identical labels to the source's inline bins inside (0, 10]")


def check_registry() -> None:
    print("registry")
    try:
        json.dumps(REGISTRY, allow_nan=False)
        check(True, "serialises without NaN/inf")
    except ValueError as exc:
        check(False, f"serialises without NaN/inf: {exc}")
    need = {"key", "label", "column", "type", "min", "max", "step", "bins", "hasScatter", "minDate",
            "series", "basis", "binning", "auto", "pane"}
    for m in REGISTRY:
        miss = need - set(m)
        check(not miss, f"{m['key']}: has {sorted(need)}" + (f" — missing {sorted(miss)}" if miss else ""))
    order = [m["key"] for m in REGISTRY if m["section"]]
    brief = ["day_of_week", "year", "gap", "vix_gap", "premium", "vix",
             "vix3m", "vix9d", "vix3m_vix", "vix_vix9d"]
    check(order == brief, f"section order matches the brief: {order}")
    check(len({m["key"] for m in REGISTRY}) == len(REGISTRY), "keys are unique")
    for m in REGISTRY:
        if m["type"] == "categorical":
            check(not m["hasScatter"], f"{m['key']}: categorical has no scatter")
        else:
            cols = list(m["basis"].values()) if m["basis"] else [m["column"]]
            check(all(c in TRADE_COLUMNS for c in cols) and m["column"] in cols,
                  f"{m['key']}: columns {cols} are in the payload whitelist")
    cov = registry_with_coverage({"spx": "2017-01-03", "vix": "2017-01-03", "vix3m": "2017-01-03", "vix9d": "2019-05-01"})
    by = {m["key"]: m for m in cov}
    check(by["vix9d"]["minDate"] == "2019-05-01" and by["vix_vix9d"]["minDate"] == "2019-05-01"
          and by["vix3m_vix"]["minDate"] == "2017-01-03" and by["premium"]["minDate"] is None,
          "minDate from coverage: a two-series metric takes the later first date; non-market metrics stay None")
    check(all(m["minDate"] is None for m in REGISTRY), "the static REGISTRY is not mutated by coverage")
    r = calc.ratio_bin_spec()
    check(len(r["labels"]) == 22 and r["labels"][0] == "<0.70" and r["labels"][-1] == ">1.50"
          and abs(r["bins"][1] - 0.70) < 1e-12 and abs(r["bins"][-2] - 1.50) < 1e-9,
          f"ratio bins: 22, <0.70 … >1.50 ({len(r['labels'])})")


DROPPED = re.compile(r"vrp|iv_rv|\bvov\b|realized_vol|spot_vol|sharpe|sharptwo|skew|regime", re.I)


def check_dropped_scope() -> None:
    print("dropped scope")
    js = JS.read_text(encoding="utf-8")
    hits = sorted(set(m.group(0) for m in DROPPED.finditer(js)))
    check(not hits, f"page JS names no SharpTwo/skew/regime metric {hits or ''}")
    reg = json.dumps(REGISTRY)
    hits = sorted(set(m.group(0) for m in DROPPED.finditer(reg)))
    check(not hits, f"registry names no SharpTwo/skew/regime metric {hits or ''}")
    hits = [c for c in TRADE_COLUMNS if DROPPED.search(c)]
    check(not hits, f"payload whitelist has no dropped column {hits or ''}")

    df = pd.DataFrame({"date_opened": pd.to_datetime(["2024-01-02"]),
                       "date_closed": pd.to_datetime(["2024-01-03"]),
                       "pnl": [1.0], "vrp_calc": [math.nan], "regime": ["x"]})
    cols = set(trades_to_payload(df)["columns"])
    check(not ({"vrp_calc", "regime"} & cols), "a planted vrp_calc/regime column does not reach the payload")


OO_HEADER = ["Date Opened", "Time Opened", "Opening Price", "Legs", "Premium", "Closing Price",
             "Date Closed", "Time Closed", "Avg. Closing Cost", "Reason For Close", "P/L", "P/L %",
             "No. of Contracts", "Funds at Close", "Margin Req.", "Strategy", "Gap", "Movement"]
OO_ROWS = [
    ["2024-03-05", "10:00:00", "5100", "1 Mar 8 P 5000 STO", "1250", "5120", "2024-03-08", "15:59:00",
     "0", "Expired", "1200", "96", "1", "101200", "5000", "IC 45", "0.35", "20"],
    ["2024-03-04", "10:00:00", "5080", "1 Mar 8 P 4980 STO", "900", "5010", "2024-03-06", "11:00:00",
     "2100", "Stop Loss", "-1200", "-133", "1", "100000", "5000", "IC 45", "-0.62", "-70"],
    ["2023-12-29", "10:00:00", "4780", "1 Jan 5 P 4700 STO", "700", "4790", "2024-01-02", "15:59:00",
     "0", "Expired", "650", "93", "1", "101200", "5000", "IC 45", "", "10"],
]


def mesosim(key: str, pnl_key: str) -> str:
    ev = [{"EventType": "Start", "Message": f"Backtest {key} mytest MesoSimVersion: 3"}]
    def pos(pid, t0, t1, pnl, reason):
        return [
            {"EventType": "EnterPosition", "PositionId": pid, "SimTime": t0, "Vars": {"pos_margin": 4000}},
            {"EventType": "EntryTrade", "PositionId": pid, "SimTime": t0,
             "TradeEvent": {"Price": -2.5, "Qty": 1, "Contract": {"Multiplier": 100}}},
            {"EventType": "EntryTrade", "PositionId": pid, "SimTime": t0,
             "TradeEvent": {"Price": 1.0, "Qty": 1, "Contract": {"Multiplier": 100}}},
            {"EventType": "ExitSignal", "PositionId": pid, "SimTime": t1, "Message": reason},
            {"EventType": "ExitPosition", "PositionId": pid, "SimTime": t1, "Vars": {pnl_key: pnl}},
        ]
    ev += pos(2, "2024-02-06T10:00:00", "2024-02-09T15:00:00", -300.0, "Stop Loss hit")
    ev += pos(1, "2024-02-05T10:00:00", "2024-02-07T15:00:00", 150.0, "Profit Target reached")
    # An open position (no exit) must be skipped, not half-parsed.
    ev += [{"EventType": "EnterPosition", "PositionId": 3, "SimTime": "2024-02-08T10:00:00", "Vars": {}}]
    return json.dumps(ev)


ISO = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def check_payload(p: dict, label: str, n: int) -> None:
    cols = p["columns"]
    check(p["n"] == n and all(len(v) == n for v in cols.values()), f"{label}: {n} trades, columns aligned")
    check(all(allowed_column(k) for k in cols), f"{label}: only whitelisted columns")
    dates = cols["date_opened"] + cols["date_closed"]
    check(all(isinstance(d, str) and ISO.match(d) for d in dates), f"{label}: dates are ISO strings")
    check(cols["date_opened"] == sorted(cols["date_opened"]), f"{label}: ordered by open date")
    json.dumps(p, allow_nan=False)
    check(True, f"{label}: JSON without NaN")


def check_parsers() -> None:
    print("parsers")
    for sep, name in (("\t", "tab"), (",", "comma")):
        text = "\n".join(sep.join(r) for r in [OO_HEADER] + OO_ROWS)
        p = _parse(text.encode(), f"log_{name}.csv")
        check_payload(p, f"OO CSV ({name})", 3)
        c = p["columns"]
        check(c["date_opened"][0] == "2023-12-29" and c["year"][0] == 2023 and c["day_of_week"][0] == 4,
              f"OO CSV ({name}): derived year/day_of_week from the open date")
        # Option Omega's Gap is a different definition from open-minus-prior-
        # close; without market data the page must show no SPX gap at all,
        # not the vendor's number under that heading.
        check("gap" not in c and "csv_gap" not in c,
              f"OO CSV ({name}): the vendor Gap column reaches the payload under no name")
        check(p["source"] == "oo_csv" and p["suggested_name"] == "IC 45", f"OO CSV ({name}): source + name")

    for key, pnl_key in (("StrategyName:", "pos_realized_pnl"), ("BacktestName:", "pos_pnl")):
        p = _parse(mesosim(key, pnl_key).encode(), "run.json")
        label = f"Mesosim ({key} {pnl_key})"
        check_payload(p, label, 2)
        c = p["columns"]
        check(c["pnl"] == [150.0, -300.0], f"{label}: P/L read, open position skipped")
        check(c["exit_reason"] == ["Profit Target", "Stop Loss"], f"{label}: exit reasons simplified")
        check(c["premium"] == [-150.0, -150.0], f"{label}: premium from the entry trades")
        check(p["suggested_name"] == "mytest", f"{label}: strategy name from the Start event")

    try:
        _parse(b"foo,bar\n1,2\n", "wrong.csv")
        check(False, "a CSV without Date Opened is rejected")
    except KeyError:
        check(True, "a CSV without Date Opened is rejected (KeyError -> 400 in the route)")


FIXTURES = ROOT / "scripts" / "fixtures" / "mesosim"


def _by_pid(p: dict) -> dict:
    c = p["columns"]
    return {pid: {k: v[i] for k, v in c.items()} for i, pid in enumerate(c["position_id"])}



def check_surface_stats() -> None:
    """Surface ranking maths (app/oo_backtest/surface_stats.py) against scipy
    directly, BH against scipy.stats.false_discovery_control, and the ranked
    metric set against the real catalog (surface_metrics_catalog.csv, diffed
    identical to the live table on 2026-09-15)."""
    print("surface metrics: ranking stats vs scipy, BH, metric set")
    import numpy as np
    from scipy import stats as sps
    from app.oo_backtest import surface, surface_stats as ss

    rng = random.Random(31)
    n = 700
    pnl = [round(rng.gauss(20, 400), 2) for _ in range(n)]
    x = [None if i < 150 else pnl[i] * 0.0004 + rng.gauss(0.15, 0.03) for i in range(n)]   # late coverage, weak signal
    bars = [(i // 3,) for i in range(n)]                                                      # three trades per bar
    got = ss.correlate(x, pnl, bars)
    xs = np.array([v for v in x if v is not None]); ys = np.array([pnl[i] for i in range(n) if x[i] is not None])
    pr, sr = sps.pearsonr(xs, ys), sps.spearmanr(xs, ys)
    close = lambda a, b: a is not None and math.isclose(a, float(b), rel_tol=1e-12, abs_tol=1e-15)   # noqa: E731
    check(got["n"] == 550 and got["bars"] == len({bars[i] for i in range(n) if x[i] is not None})
          and close(got["pearson"], pr.statistic) and close(got["pearson_p"], pr.pvalue)
          and close(got["spearman"], sr.statistic) and close(got["spearman_p"], sr.pvalue),
          f"correlate: n={got['n']} (pairwise, nulls dropped), bars={got['bars']}, r/rho and p equal scipy")
    check(got["bars"] < got["n"], "distinct entry bars are counted separately from n (shared bars)")
    for label, xv, yv in (("two points", [1.0, 2.0], [1.0, 3.0]), ("constant metric", [0.2] * 10, list(range(10))),
                          ("constant P/L", list(range(10)), [5.0] * 10)):
        g = ss.correlate(xv, yv, list(range(len(xv))))
        check(g["pearson"] is None and g["spearman_p"] is None, f"{label}: no correlation, not a NaN")

    ps = [rng.random() ** 3 for _ in range(452)] + [None, None]
    rng.shuffle(ps)
    bh = ss.benjamini_hochberg(ps)
    ref = sps.false_discovery_control(np.array([p for p in ps if p is not None]), method="bh")
    check(all(math.isclose(a, b, rel_tol=1e-12) for a, b in zip([b for b in bh if b is not None], ref)),
          f"BH equals scipy.stats.false_discovery_control over {len(ref)} p-values; None stays None")
    check([i for i, b in enumerate(bh) if b is None] == [i for i, p in enumerate(ps) if p is None],
          "a metric with no p is excluded from m and stays None")
    # Raw p*m/k is 0.03, 0.03, 0.021: the running minimum from the top pulls
    # the first two down to 0.021. Without it they would stay 0.03.
    toy = ss.benjamini_hochberg([0.01, 0.02, 0.021])
    check(all(math.isclose(a, b) for a, b in zip(toy, [0.021, 0.021, 0.021])),
          f"BH by hand: [0.01, 0.02, 0.021] -> all 0.021 (the step-down minimum applies) ({toy})")

    cat = pd.read_csv(ROOT / "surface_metrics_catalog.csv").astype(object).where(lambda d: d.notna(), None).to_dict("records")
    table = {r["column_name"]: "double precision" for r in cat if r["family"] != "meta"}
    table.update({"trade_date": "date", "quote_time": "time without time zone", "day_of_week": "integer",
                  "days_to_monthly_opex": "integer"})
    ranked, rep = surface.metric_set(cat, table)
    fams = {r["family"] for r in ranked}
    check(rep["catalog_rows"] == 462 and rep["ranked"] == 452 and rep["excluded"] == 10
          and not ({"meta", "spot", "forward"} & fams) and "log_ret" in fams,
          f"ranked set: 462 catalog rows - 4 meta - spot - 5 forward_* = {rep['ranked']}; log_ret kept")
    check({r["form"] for r in ranked} == {"level", "chg_d", "chg_1w", "z"},
          f"forms present: {sorted({r['form'] for r in ranked})}")
    planted = dict(table, **{"iv_30d_atm": "text"})
    planted.pop("vix_30d")
    planted["not_in_catalog"] = "double precision"
    _, prep = surface.metric_set(cat, planted)
    check(prep["wrong_type"] == ["iv_30d_atm"] and prep["missing_from_table"] == ["vix_30d"]
          and "not_in_catalog" in prep["uncatalogued"],
          "planted: a non-double column, a catalogued column missing from the table and an uncatalogued one are reported")
    check(surface.quote_ident('a"b') == '"a""b"', "identifiers are quoted with embedded quotes doubled")
    check(surface.BAR_RULE == "at_or_before_entry" and surface.LOOKAHEAD_CONFIRMED is True
          and not hasattr(surface, "BAR_RULES") and "INTERVAL" not in surface.entry_sql(["iv_30d_atm"]),
          "entry bar fixed at at_or_before_entry, lookahead confirmed; the previous_bar alternative is gone")
    check_surface_matrix()

    tr, trep = surface.parse_trades([["2024-01-02", "15:30:00", 12.5], ["bad", "10:00", 1], ["2024-01-03", "", 2],
                                     ["2024-01-04", "3:30 PM", "x"]], with_pnl=True)
    check(len(tr) == 4 and trep == {"trades": 4, "bad_date": 1, "bad_time": 1, "bad_pnl": 1}
          and tr[3][1] == __import__("datetime").time(15, 30),
          f"parse_trades keeps every row in order and counts what did not parse ({trep})")


def check_surface_matrix() -> None:
    """The vectorised ranking against per-metric scipy calls at full size:
    2,097 trades x 452 metrics, shaped like the VPS run -- metrics starting on
    different dates (so n differs per metric), ~36% of trades with no bar,
    ties, a constant metric, a metric with too few values, P/L the page could
    not parse, and trades sharing a bar."""
    import time as _t
    import numpy as np
    from scipy import stats as sps
    from app.oo_backtest import surface_stats as ss

    rng = np.random.default_rng(37)
    T, C = 2097, 452
    pnl = rng.normal(20, 400, T).round(2)
    pnl[rng.choice(T, 6, replace=False)] = np.nan                   # unparseable P/L
    bar_ids = np.arange(T) // 2                                       # two trades per bar
    no_bar = np.zeros(T, bool); no_bar[:754] = True                   # the pre-coverage 36%
    starts = rng.choice([754, 800, 910, 1150, 1203], C)               # coverage start per metric
    X = rng.normal(0, 1, (T, C)) + np.outer(pnl, rng.normal(0, 0.002, C))
    X[:, ::7] = X[:, ::7].round(1)                                    # heavy ties for Spearman
    for j in range(C):
        X[: starts[j], j] = np.nan
    X[no_bar] = np.nan
    X[np.isfinite(X[:, 3]), 3] = 0.15                                 # constant wherever covered
    X[:, 5] = np.nan; X[2000:2002, 5] = [1.0, 2.0]                     # only two values
    X[:, 9] = np.nan                                                  # no coverage at all
    X[1500, 11] = np.nan                                              # a metric with its own null pattern

    t0 = _t.monotonic()
    got = ss.correlations(X, pnl, bar_ids)
    took = _t.monotonic() - t0

    bad, checked = [], 0
    for j in range(C):
        m = np.isfinite(X[:, j]) & np.isfinite(pnl)
        n = int(m.sum())
        want_bars = len(np.unique(bar_ids[m]))
        if got["n"][j] != n or got["bars"][j] != want_bars:
            bad.append((j, "n/bars", got["n"][j], n)); continue
        x, y = X[m, j], pnl[m]
        if n < 3 or np.all(x == x[0]) or np.all(y == y[0]):
            if not all(np.isnan(got[f][j]) for f in ("pearson", "pearson_p", "spearman", "spearman_p")):
                bad.append((j, "should be undefined")); continue
            continue
        pr, sr = sps.pearsonr(x, y), sps.spearmanr(x, y)
        for f, want in (("pearson", pr.statistic), ("pearson_p", pr.pvalue),
                        ("spearman", sr.statistic), ("spearman_p", sr.pvalue)):
            # Not bit-identical (summation order differs): r within 1e-12
            # relative, p within 1e-9 -- p magnifies r's last-digit noise.
            # Measured: r 2e-14, p 1.6e-12.
            if not math.isclose(got[f][j], want, rel_tol=1e-9 if f.endswith("_p") else 1e-12, abs_tol=1e-300):
                bad.append((j, f, got[f][j], want))
        checked += 1
    ns = np.sort(got["n"][got["n"] > 0])
    check(not bad, f"all {C} metrics: n, bars, r, rho and both p equal per-metric scipy "
                   f"({checked} with a correlation){' — ' + str(bad[:3]) if bad else ''}")
    check(len(set(ns.tolist())) >= 5 and ns[0] < ns[len(ns) // 2] < ns[-1],
          f"n really differs per metric, so pairwise dropping is exercised (min {ns[0]}, median {ns[len(ns) // 2]}, max {ns[-1]})")
    check(np.isnan(got["pearson"][[3, 5, 9]]).all() and got["n"][5] == 2 and got["n"][9] == 0,
          "constant, two-value and uncovered metrics have no correlation; their n is still reported")
    check(took < 0.3, f"2,097 x 452 in {took * 1000:.0f} ms here (target < 300 ms on the VPS)")

    rows = ss.rank([{"column_name": f"m{j}", "family": "f", "tenor": None, "wing": None, "form": "level"} for j in range(C)],
                   X, [None if np.isnan(p) else float(p) for p in pnl], bar_ids)
    ps = [r["spearman_p"] for r in rows]
    check(all(r["spearman_p_bh"] is None for r, p in zip(rows, ps) if p is None)
          and sum(p is not None for p in ps) == checked,
          f"rank(): BH over the {checked} metrics with a p; the rest stay None")


def check_real_mesosim() -> None:
    """Real MesoSim events, trimmed to the positions that exercise each rule.

    v3_1: allantis v2 (MesoSim 3.1.11) — pos_realized_pnl, entry_net_premium,
          a position with two exit signals on its exit bar, the 2023-07-03
          12:30 early-close entry, a max-adjustments exit, two open positions.
    v2_13: allantis weekly Fri (MesoSim 2.13.4) — no realized P/L field, no
          entry_net_premium, TemplateName, a double signal ending in a stop.
    """
    print("real MesoSim exports (trimmed)")
    v3_raw = (FIXTURES / "v3_1_allantis_v2_mon.json").read_bytes()
    p = _parse(v3_raw, "v3.events.json")
    t, n = _by_pid(p), p["notes"]
    events = json.loads(v3_raw)
    fills = sum(e["EventType"] == "EntryTrade" for e in events)
    check(p["n"] == 4 and fills > 4 * p["n"],
          f"v3.1: one trade per closed PositionId ({p['n']}), not per leg fill ({fills})")
    check(n["open_positions"] == 2 and n["open_position_ids"] == [222, 223] and not ({222, 223} & set(t)),
          "v3.1: open positions 222, 223 excluded and reported")
    check(t[0]["pnl"] == 1461.56 and n["pnl_field"] == "pos_realized_pnl", "v3.1: P/L from pos_realized_pnl")
    check(t[0]["premium"] == 11215 and n["premium_field"] == "entry_net_premium",
          "v3.1: premium from entry_net_premium")
    check(t[0]["margin_req"] == 16640.6, "v3.1: margin from pos_margin")
    check(t[26]["exit_reason"] == "Profit Target" and n["multi_signal_positions"] == 1,
          "v3.1: two signals on the exit bar -> the later one (Profit Target), counted")
    check(t[122]["exit_reason"] == "Max Adjustments", "v3.1: 'Maximum adjustment count reached (25 >= 25)' -> Max Adjustments")
    reasons = set(p["columns"]["exit_reason"])
    check(not any(re.search(r"[\d:()]", r) for r in reasons), f"v3.1: no threshold survives in a label {sorted(reasons)}")
    check((t[106]["date_opened"], t[106]["time_opened"]) == ("2023-07-03", "12:30:00"),
          "v3.1: early-close entry kept as 2023-07-03 12:30:00 (P2 entry-time join fixture)")
    check(all(x == "15:30:00" for pid, x in ((k, r["time_opened"]) for k, r in t.items()) if pid != 106),
          "v3.1: every other entry at 15:30:00")
    check(p["suggested_name"] == "*allantis - v2: 5+4+1, PT SPX*.35, 60DIT, mon, 2021-2026"
          and t[0]["strategy"] == "allantis", "v3.1: save name from BacktestName, strategy from StrategyName")
    df_v3 = data_loader.parse_mesosim_json(v3_raw, "v3.events.json")
    check("entry_var_profit_target" in df_v3.columns and "entry_var_pos_vega" in df_v3.columns,
          "v3.1: EnterPosition Vars kept on the parsed DataFrame (entry_var_*)")
    check(not any(k.startswith("entry_var_") for k in p["columns"]),
          "v3.1: ...and none of them reach the payload")
    check(not n["precedence_vs_order"] and not n["pnl_contradicts_reason"] and not n["premium_leg_mismatch"],
          "v3.1: precedence agrees with file order, P/L agrees with reasons, legs sum to entry_net_premium")
    # P&L agreement as an assertion, on every trade the fixture has.
    pt = df_v3[df_v3["exit_reason"] == "Profit Target"]
    check(len(pt) and bool((pt["pnl"] >= pt["entry_var_profit_target"]).all()),
          f"v3.1: every Profit Target exit has P/L >= its target ({len(pt)})")
    check(not n["missing_data_at_fill"] and not n["pnl_mismatch"], "v3.1: no MissingData at fills, no P/L disagreement")

    # Planted faults. Each must change the output the way the rule says.
    ev = json.loads(v3_raw)
    enter0 = next(e for e in ev if e["EventType"] == "EnterPosition" and e["PositionId"] == 0)
    ev.append({"EventType": "MissingData", "PositionId": 0, "SimTime": enter0["SimTime"],
               "Message": "No data for given contract 'X' at current time"})
    exit0 = next(e for e in ev if e["EventType"] == "ExitPosition" and e["PositionId"] == 0)
    exit0["Vars"]["pos_pnl"] = exit0["Vars"]["pos_realized_pnl"] + 50
    ev.append({"EventType": "ExitSignal", "PositionId": 26, "SimTime": "2099-01-01T15:30:00",
               "Message": "Reached stop loss: 1"})
    # A time signal written AFTER position 26's profit-target signal on the
    # same bar, ahead of the exit. File order now says "time"; precedence must
    # still say "Profit Target", and must report the disagreement.
    exit26 = next(i for i, e in enumerate(ev) if e["EventType"] == "ExitPosition" and e["PositionId"] == 26)
    ev.insert(exit26, {"EventType": "ExitSignal", "PositionId": 26, "SimTime": ev[exit26]["SimTime"],
                       "Message": "Max time in trade reached: 60 days"})
    # A leg fill on position 0's entry bar that entry_net_premium does not include.
    ev.append({"EventType": "EntryTrade", "PositionId": 0, "SimTime": enter0["SimTime"],
               "TradeEvent": {"Price": 1.0, "Qty": 1, "Contract": {"Multiplier": 100}}})
    f = _parse(json.dumps(ev).encode(), "planted.json")
    ft, fn = _by_pid(f), f["notes"]
    check(fn["missing_data_at_fill"] == [{"position_id": 0, "at": ["entry"]}] and ft[0]["missing_data_at_fill"] == "entry",
          "planted: MissingData on the entry bar is flagged on that trade")
    check(len(fn["pnl_mismatch"]) == 1 and ft[0]["pnl"] == 1461.56,
          "planted: pos_pnl disagreeing with realized is reported, realized still used")
    check(ft[26]["exit_reason"] == "Profit Target", "planted: a signal AFTER the exit does not change the reason")
    check([x["position_id"] for x in fn["precedence_vs_order"]] == [26],
          "planted: time signal written last on the exit bar -> precedence keeps Profit Target, disagreement reported")
    check([x["position_id"] for x in fn["premium_leg_mismatch"]] == [0] and ft[0]["premium"] == 11215,
          "planted: legs not summing to entry_net_premium are reported; entry_net_premium still used")

    for msg, want in [("Delta limit breached (12 >= 10): 0.35", "Delta limit breached"),
                      ("Reached stop loss: 3690.06", "Stop Loss"),
                      ("Max time in trade reached: 60 days", "Max Time in Trade"),
                      ("", "Unknown")]:
        got = data_loader._simplify_exit_reason(msg)
        check(got == want, f"exit reason {msg!r} -> {want!r} (got {got!r})")

    p = _parse((FIXTURES / "v2_13_allantis_weekly.json").read_bytes(), "v213.events.json")
    t, n = _by_pid(p), p["notes"]
    check(n["pnl_field"] == "pos_pnl" and n["premium_field"] == "leg fills",
          "v2.13: P/L falls back to pos_pnl, premium to leg fills, and the page is told")
    check(all(v is None for v in p["columns"]["margin_req"]), "v2.13: no pos_margin -> margin null, not stop_loss")
    check(all(v == round(v, 2) for v in p["columns"]["premium"]), "v2.13: leg-sum premium rounded to cents")
    check(t[54]["exit_reason"] == "Stop Loss" and t[54]["pnl"] < 0, "v2.13: time + stop on the exit bar -> Stop Loss")
    check(not n["precedence_vs_order"] and not n["pnl_contradicts_reason"], "v2.13: precedence agrees with order and P/L")
    check((t[42]["date_opened"], t[42]["time_opened"]) == ("2024-11-29", "12:30:00"),
          "v2.13: early-close entry 2024-11-29 12:30:00 kept")
    check(n["open_positions"] == 1 and 72 not in t, "v2.13: open position 72 excluded and reported")
    check(p["suggested_name"] == "allantis - weekly entry - Fri" and t[0]["strategy"] == "allantis",
          "v2.13: full BacktestName kept (the source cut it to 'allantis'); TemplateName as strategy")


STATS_DRIVER = r"""
const fs = require('fs');
global.document = { addEventListener: () => {} };
// The shared calculations, then the page: the bundle's functions are
// top-level declarations that read them, exactly as the browser loads
// the two script tags.
eval(fs.readFileSync(process.argv[2], 'utf8') + String.fromCharCode(10)
     + fs.readFileSync(process.argv[1], 'utf8'));
const job = JSON.parse(fs.readFileSync(0, 'utf8'));
const out = {};
for (const [key, t] of Object.entries(job)) {
  const specs = (t.specs || []).map(s => s.kind === 'set' ? { ...s, allowed: new Set(s.allowed) } : s);
  const idx = t.idx || obApplyFilters(t.cols, t.n, specs);
  out[key] = { idx, stats: obStats(t.cols, idx), equity: obEquity(t.cols, idx) };
}
process.stdout.write(JSON.stringify(out));
"""


def run_stats_js(job: dict) -> dict:
    p = subprocess.run(["node", "-e", STATS_DRIVER, str(JS), str(CORE)], input=json.dumps(job),
                       capture_output=True, text=True, encoding="utf-8")
    if p.returncode:
        raise RuntimeError(p.stderr.strip())
    return json.loads(p.stdout)


def check_stats_parity() -> None:
    """The page's summary stats, equity curve and filters vs the Python they
    replace (utils/stats.py calculate_stats, calculations.py
    calculate_drawdown and filter_dataframe's inclusive bounds)."""
    print("summary stats, equity and filters: shipped JS vs stats.py / calculations.py")
    import shutil
    if shutil.which("node") is None:
        print("  SKIP  node is not installed")
        NOT_RUN.append("JS stats/equity/filter parity (node not installed)")
        return
    from app.oo_backtest import stats as pystats

    rng = random.Random(3)
    n = 400
    base = pd.Timestamp("2021-01-04")
    opened = sorted(base + pd.Timedelta(days=rng.randint(0, 900)) for _ in range(n))
    df = pd.DataFrame({
        "date_opened": opened,
        # Unique close dates here: stats.py sorts with pandas' default (not
        # stable) sort, so ties are compared separately below.
        "date_closed": [base + pd.Timedelta(days=1000 + i * 2 + rng.randint(0, 1)) for i in rng.sample(range(n), n)],
        "pnl": [0.0 if i % 37 == 0 else round(rng.gauss(40, 600), 2) for i in range(n)],
        "days_in_trade": [None if i == 5 else rng.randint(0, 60) for i in range(n)],
        "vix_level": [None if i % 11 == 0 else round(rng.uniform(10, 40), 2) for i in range(n)],
        "exit_reason": [rng.choice(["Profit Target", "Stop Loss", "Expired"]) for _ in range(n)],
    })
    df["day_of_week"] = df["date_opened"].dt.dayofweek
    cols = {c: [None if pd.isna(v) else (v.strftime("%Y-%m-%d") if isinstance(v, pd.Timestamp) else v)
                for v in df[c]] for c in df.columns}

    # Filters: VIX 12-35 inclusive, Mon-Thu, dates within 2021-03..2023-03, not Expired.
    specs = [{"kind": "range", "column": "vix_level", "lo": 12.0, "hi": 35.0},
             {"kind": "set", "column": "day_of_week", "allowed": [0, 1, 2, 3]},
             {"kind": "date", "column": "date_opened", "from": "2021-03-01", "to": "2023-03-31"},
             {"kind": "set", "column": "exit_reason", "allowed": ["Profit Target", "Stop Loss"]}]
    want_mask = ((df["vix_level"] >= 12) & (df["vix_level"] <= 35) & df["day_of_week"].isin([0, 1, 2, 3])
                 & (df["date_opened"] >= "2021-03-01") & (df["date_opened"] <= "2023-03-31")
                 & df["exit_reason"].isin(["Profit Target", "Stop Loss"]))
    ties = df.copy()
    ties["date_closed"] = [base + pd.Timedelta(days=1000 + (i // 3)) for i in range(n)]   # three per day
    ties_cols = dict(cols, date_closed=[d.strftime("%Y-%m-%d") for d in ties["date_closed"]])

    got = run_stats_js({"all": {"cols": cols, "n": n, "idx": list(range(n))},
                        "filtered": {"cols": cols, "n": n, "specs": specs},
                        "none": {"cols": cols, "n": n, "specs": [{"kind": "range", "column": "vix_level", "lo": 99, "hi": 100}]},
                        "ties": {"cols": ties_cols, "n": n, "idx": list(range(n))}})

    def close(a, b):
        return math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-6)

    for key, sub in (("all", df), ("filtered", df[want_mask])):
        want = pystats.calculate_stats(sub.copy())
        js = got[key]["stats"]
        bad = [k for k in js if not close(float(js[k]), float(want[k]))]
        check(not bad, f"{key}: all ten stats equal stats.py over {len(sub)} trades" +
              (f" — differ: {[(k, js[k], want[k]) for k in bad]}" if bad else ""))
    check(got["filtered"]["idx"] == [i for i in range(n) if want_mask.iloc[i]],
          f"filters select exactly the pandas rows (null VIX excluded, bounds inclusive) ({len(got['filtered']['idx'])})")
    check(got["none"]["idx"] == [] and got["none"]["stats"]["num_trades"] == 0 and got["none"]["equity"]["maxDD"] is None,
          "a filter matching nothing gives zero trades, zero stats, no drawdown point")

    dd = calc.calculate_drawdown(df.copy())
    js_pts = got["all"]["equity"]["points"]
    check(len(js_pts) == len(dd) and all(close(p["cumulative"], c) and close(p["drawdown"], d)
                                         for p, c, d in zip(js_pts, dd["cumulative_pnl"], dd["drawdown"])),
          "cumulative P/L and drawdown equal calculate_drawdown point for point")
    mdd = got["all"]["equity"]["maxDD"]
    check(mdd is not None and close(mdd["drawdown"], dd["drawdown"].min())
          and mdd["date"] == dd.loc[dd["drawdown"].idxmin(), "date_closed"].strftime("%Y-%m-%d"),
          f"max-DD point is the deepest drawdown, and its date ({mdd and mdd['date']})")

    # Ties: three trades per close date. Against a STABLE pandas sort the JS
    # must agree exactly; that it keeps payload order is the point.
    t = ties.sort_values("date_closed", kind="mergesort")
    cum = t["pnl"].cumsum()
    stable_dd = float((cum - cum.cummax()).min())
    check(close(got["ties"]["stats"]["max_drawdown"], stable_dd),
          f"same-day closes: max drawdown equals a stable-order pandas sort ({got['ties']['stats']['max_drawdown']:.2f})")


SECTION_DRIVER = r"""
const fs = require('fs');
global.document = { addEventListener: () => {} };
// The shared calculations, then the page: the bundle's functions are
// top-level declarations that read them, exactly as the browser loads
// the two script tags.
eval(fs.readFileSync(process.argv[2], 'utf8') + String.fromCharCode(10)
     + fs.readFileSync(process.argv[1], 'utf8'));
const job = JSON.parse(fs.readFileSync(0, 'utf8'));
const out = {};
for (const [key, t] of Object.entries(job)) {
  const d = obSectionData(t.cols, t.idx, t.metric, t.column);
  out[key] = { rows: d.rows, valued: d.valued, fit: obOLS(d.xs, d.ys),
               swapped: obOLS(d.ys, d.xs), raw: t.raw ? obOLS(t.raw.xs, t.raw.ys) : null,
               alpha: { max: obBarAlpha(400, 400), one_of_400: obBarAlpha(1, 400),
                        quarter: obBarAlpha(100, 400), zero: obBarAlpha(0, 400),
                        p531: obBarAlpha(531, 531), p249: obBarAlpha(249, 531), p6: obBarAlpha(6, 531), p1: obBarAlpha(1, 531) } };
}
process.stdout.write(JSON.stringify(out));
"""


def check_section_parity() -> None:
    """Every metric section's numbers vs the Python they replace: per-bin
    count / total / mean / win rate against calculations.py calculate_bin_stats
    over the same pd.cut, and the scatter's OLS line against
    calculate_correlation (scipy linregress / pearsonr). Run over the whole
    log AND a filtered subset, since the page recomputes from filtered rows."""
    print("metric sections: shipped JS vs calculate_bin_stats / calculate_correlation")
    import shutil
    if shutil.which("node") is None:
        print("  SKIP  node is not installed")
        NOT_RUN.append("JS metric-section parity (node not installed)")
        return
    reg = json.loads(json.dumps(REGISTRY, allow_nan=False))
    sections = [m for m in reg if m["section"] and m.get("binning") != "auto"]
    rng = random.Random(19)
    n = 900

    def values_for(m):
        if m["key"] == "day_of_week":
            # A 5 (Saturday) is outside the fixed list: it must get its own bar.
            return [None if i % 41 == 0 else (5 if i % 97 == 0 else rng.randint(0, 4)) for i in range(n)]
        if m["key"] == "year":
            return [rng.randint(2017, 2026) for _ in range(n)]
        e = m["bins"]["edges"]
        span = (e[-1] - e[0]) or 1.0
        pool = list(e) + [round(rng.uniform(e[0] - 0.2 * span, e[-1] + 0.2 * span), 2) for _ in range(n)]
        # A hole of several bins in the middle, so the kept-empty-bin path runs.
        k = len(e) // 2
        pool = [v for v in pool if not (e[k - 2] <= v <= e[k + 2])]
        return [None if i % 23 == 0 else rng.choice(pool) for i in range(n)]

    pnl = [0.0 if i % 31 == 0 else round(rng.gauss(30, 500), 2) for i in range(n)]
    subset = [i for i in range(n) if rng.random() < 0.4]
    job, want = {}, {}
    for m in sections:
        col = m["basis"]["entry"] if m["basis"] else m["column"]
        vals = values_for(m)
        cols = {col: vals, "pnl": pnl}
        for scope, idx in (("all", list(range(n))), ("subset", subset)):
            key = f"{m['key']}:{scope}"
            job[key] = {"cols": cols, "idx": idx, "metric": m, "column": col}
            df = pd.DataFrame({col: pd.Series([math.nan if vals[i] is None else vals[i] for i in idx], dtype=float),
                               "pnl": [pnl[i] for i in idx]})
            if m["type"] == "range":
                df = calc.apply_bin_spec(df, col, {"bins": [-math.inf] + m["bins"]["edges"] + [math.inf],
                                                   "labels": m["bins"]["labels"],
                                                   "right": m["bins"]["closed"] == "right"})
                bs = calc.calculate_bin_stats(df, f"{col}_bin")
                labels = list(bs["bin"].astype(str))
                corr = calc.calculate_correlation(df, col)
            else:
                names = {c["value"]: c["label"] for c in (m["categories"] or [])}
                bs = calc.calculate_bin_stats(df.dropna(subset=[col]).astype({col: int}), col)
                labels = [names.get(int(v), str(int(v))) for v in bs["bin"]]
                corr = None
            want[key] = {"rows": [{"label": lab, "count": int(r["count"]), "total": float(r["total_pnl"]),
                                   "avg": float(r["avg_pnl"]), "win": float(r["win_rate"])}
                                  for lab, (_, r) in zip(labels, bs.iterrows())],
                         "valued": int(df[col].notna().sum()), "corr": corr}

    # Degenerate fits: two points; every x the same.
    job["deg"] = {"cols": {"v": [1.0], "pnl": [1.0]}, "idx": [0], "metric": sections[2], "column": "v",
                  "raw": {"xs": [3.0, 3.0, 3.0, 3.0], "ys": [1.0, -2.0, 5.0, 0.0]}}
    job["two"] = {"cols": {"v": [1.0, 2.0], "pnl": [10.0, -4.0]}, "idx": [0, 1], "metric": sections[2], "column": "v"}
    # Sparse VIX: two far-apart readings must keep every bin between them.
    vix_m = next(m for m in sections if m["key"] == "vix")
    job["sparse"] = {"cols": {"v": [18.5, 45.0, 18.2], "pnl": [100.0, -50.0, 20.0]}, "idx": [0, 1, 2],
                     "metric": vix_m, "column": "v"}

    p = subprocess.run(["node", "-e", SECTION_DRIVER, str(JS), str(CORE)], input=json.dumps(job),
                       capture_output=True, text=True, encoding="utf-8")
    if p.returncode:
        check(False, f"section driver ran ({p.stderr.strip()[:300]})")
        return
    got = json.loads(p.stdout)

    def close(a, b):
        return a is not None and b is not None and math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-6)

    for key, w in want.items():
        g = got[key]
        m = job[key]["metric"]
        # Compared BY LABEL on the bins pandas reports. The page deliberately
        # keeps empty range bins (calculate_bin_stats drops them), so the lists
        # differ by design; what must agree is every filled bin's values, and
        # every extra JS bin must be one pandas left out for having no trades.
        filled = [r for r in g["rows"] if r["count"]]
        same_rows = (len(filled) == len(w["rows"]) and all(
            a["label"] == b["label"] and a["count"] == b["count"]
            and all(close(a[k], b[k]) for k in ("total", "avg", "win"))
            for a, b in zip(filled, w["rows"])))
        first = next((i for i, (a, b) in enumerate(zip(filled, w["rows"])) if a != b), None)
        check(same_rows and g["valued"] == w["valued"],
              f"{key}: {len(w['rows'])} filled bins equal calculate_bin_stats ({w['valued']} valued)"
              + ("" if same_rows else f" — js {filled[first] if first is not None else len(filled)} vs pandas "
                 f"{w['rows'][first] if first is not None else len(w['rows'])}"))
        empty = [r for r in g["rows"] if not r["count"]]
        if m["type"] == "range":
            pandas_labels = {r["label"] for r in w["rows"]}
            check([r["label"] for r in g["rows"]] == m["bins"]["labels"]
                  and all(r["avg"] is None and r["total"] is None and r["win"] is None
                          and r["label"] not in pandas_labels for r in empty),
                  f"{key}: every bin kept in label order; the {len(empty)} empty ones are null and are the bins pandas omits")
        else:
            check(not empty, f"{key}: categorical — no empty bins")
        c = w["corr"]
        if c is not None:
            f = g["fit"]
            ok = f is not None and all(close(f[a], c[b]) for a, b in
                                       (("slope", "slope"), ("intercept", "intercept"), ("r", "correlation"), ("r2", "r_squared")))
            check(ok, f"{key}: OLS slope/intercept/r/R² equal calculate_correlation"
                  + ("" if ok else f" — js {f} vs {c}"))
            check(f is not None and not close(g["swapped"]["slope"], c["slope"]),
                  f"{key}: a fit of x on y (planted swap) IS detected")

    dow = got["day_of_week:all"]["rows"]
    check([r["label"] for r in dow] == ["Mon", "Tue", "Wed", "Thu", "Fri", "5"],
          f"day of week: bars in Mon-Fri order, an off-list value kept as its own bar ({[r['label'] for r in dow]})")
    sp = got["sparse"]["rows"]
    lab = vix_m["bins"]["labels"]
    i18, i45 = lab.index("18"), lab.index("45")
    check(len(sp) == len(lab) and sp[i18]["count"] == 2 and sp[i45]["count"] == 1
          and sum(r["count"] for r in sp) == 3 and i45 - i18 > 1,
          f"sparse VIX: 18.x and 45 stay {i45 - i18} bins apart, not adjacent ({len(sp)} bins)")
    alpha = got["sparse"]["alpha"]
    check(abs(alpha["max"] - 1) < 1e-12 and abs(alpha["quarter"] - (0.12 + 0.88 * 0.25)) < 1e-12
          and abs(alpha["one_of_400"] - (0.12 + 0.88 / 400)) < 1e-12 and alpha["zero"] == 0
          and abs(alpha["p531"] - 1) < 1e-12 and abs(alpha["p249"] - 0.5327) < 1e-3
          and abs(alpha["p6"] - 0.1299) < 1e-3 and abs(alpha["p1"] - 0.1217) < 1e-3,
          f"bar opacity: 0.12 + 0.88 x (count/max)^1 — Premium's 531/249/6/1 of 531 -> "
          f"{alpha['p531']:.2f}/{alpha['p249']:.2f}/{alpha['p6']:.2f}/{alpha['p1']:.2f}; empty 0")
    check(got["two"]["fit"] is None and got["deg"]["raw"] is None,
          "no OLS line from fewer than 3 points, or when every x is identical")



AUTO_DRIVER = r"""
const fs = require('fs');
global.document = { addEventListener: () => {} };
// The shared calculations, then the page: the bundle's functions are
// top-level declarations that read them, exactly as the browser loads
// the two script tags.
eval(fs.readFileSync(process.argv[2], 'utf8') + String.fromCharCode(10)
     + fs.readFileSync(process.argv[1], 'utf8'));
const job = JSON.parse(fs.readFileSync(0, 'utf8'));
const out = {};
for (const [key, t] of Object.entries(job)) {
  const bins = obAutoBins(t.values, t.auto, t.auto.steps === 'nice' ? { decimals: 2, suffix: '' } : 'usd');
  if (!bins) { out[key] = null; continue; }
  const idx = t.values.map((_, i) => i);
  const d = obSectionData({ v: t.values, pnl: t.pnl }, idx, { type: 'range', bins, categories: null }, 'v');
  out[key] = { bins, codes: t.values.map(v => obBinIndex(v, bins)), rows: d.rows };
}
process.stdout.write(JSON.stringify(out));
"""


def auto_bins_reference(values: list, auto: dict) -> dict | None:
    """The auto-binning rule, written independently in numpy: percentiles by
    np.percentile's default (linear), each step's span snapped outward, the
    step whose bin count is nearest targetBins (ties to the smaller step)."""
    import numpy as np
    clean = lambda x: float(f"{x:.12g}")   # noqa: E731 -- Number(x.toPrecision(12))
    v = np.array([x for x in values if x is not None], dtype=float)
    if not len(v):
        return None
    p_lo, p_hi = np.percentile(v, auto["pLo"]), np.percentile(v, auto["pHi"])
    steps = auto["steps"]
    if steps == "nice":
        span = p_hi - p_lo if p_hi - p_lo > 0 else (abs(p_hi) or 1.0)
        e = math.floor(math.log10(span / auto["targetBins"]))
        steps = [clean(m * 10.0 ** k) for k in range(e - 1, e + 2) for m in (1, 2, 2.5, 5)]
    best = None
    for step in steps:
        lo = clean(math.floor(p_lo / step) * step)
        hi = clean(math.ceil(p_hi / step) * step)
        if hi <= lo:
            hi = clean(lo + step)
        n = round((hi - lo) / step)
        if best is None or abs(n - auto["targetBins"]) < abs(best[3] - auto["targetBins"]):
            best = (step, lo, hi, n)
    step, lo, hi, n = best
    return {"step": step, "edges": [clean(lo + k * step) for k in range(n + 1)], "n": n}


def check_auto_bins() -> None:
    """Premium (binning 'auto'): edges built from the log. No fixed pandas
    helper exists to compare against, so the rule is checked against an
    independent numpy implementation, the bin assignment against pd.cut over
    those edges, and the per-bin figures against calculate_bin_stats."""
    print("auto bins (Premium): shipped JS vs a numpy reference")
    import shutil
    if shutil.which("node") is None:
        print("  SKIP  node is not installed")
        NOT_RUN.append("auto-bin parity (node not installed)")
        return
    reg = json.loads(json.dumps(REGISTRY, allow_nan=False))
    autos = [m for m in reg if m.get("binning") == "auto"]
    check([m["key"] for m in autos] == ["premium"] and all(m["binning"] == "fixed" for m in reg
                                                          if m["type"] == "range" and m["key"] != "premium"),
          f"Premium is the only auto-binned metric; every other range metric is fixed ({[m['key'] for m in autos]})")
    auto = autos[0]["auto"]
    check(auto == {"steps": [10, 25, 50, 100, 250], "targetBins": 24, "pLo": 1, "pHi": 99},
          f"premium auto spec: ~24 bins over p1..p99, steps $10/$25/$50/$100/$250 ({auto})")

    rng = random.Random(29)
    cases = {
        # a ~$5.00 premium strategy (dollars per contract), with a few outliers either side
        "five_dollar": [round(rng.gauss(510, 55), 0) for _ in range(780)] + [40, 60, 1500, 2100, 2600, 90, 3000],
        "wide_credit_debit": [round(rng.uniform(-3000, 6000), 0) for _ in range(500)],
        "narrow": [round(rng.uniform(480, 500), 0) for _ in range(300)],
        "constant": [500.0] * 50,
        "single": [737.0],
        "negative_only": [round(rng.gauss(-640, 120), 0) for _ in range(400)],
        "with_nulls": [None if i % 9 == 0 else round(rng.gauss(300, 80), 0) for i in range(400)],
    }
    job = {k: {"values": v, "auto": auto, "pnl": [round(rng.gauss(20, 300), 2) for _ in v]} for k, v in cases.items()}
    # Added surface rows: "nice" steps over arbitrary scales, in display units.
    from app.oo_backtest import surface as _surface
    nice = _surface.ROW_AUTO_BINS
    nice_cases = {
        "nice_vol_pts": [round(rng.gauss(18, 5), 4) for _ in range(1300)],          # vol_decimal x 100
        "nice_z": [round(rng.gauss(0, 1.1), 6) for _ in range(1200)],
        "nice_tiny_slope": [rng.gauss(0.004, 0.0015) for _ in range(900)],
        "nice_negative_skew": [rng.gauss(-0.21, 0.04) for _ in range(900)],
        "nice_constant": [0.25] * 40,
        "nice_zero": [0.0] * 40,
    }
    for k, v in nice_cases.items():
        cases[k] = v
        job[k] = {"values": v, "auto": nice, "pnl": [round(rng.gauss(20, 300), 2) for _ in v]}
    p = subprocess.run(["node", "-e", AUTO_DRIVER, str(JS), str(CORE)], input=json.dumps(job),
                       capture_output=True, text=True, encoding="utf-8")
    if p.returncode:
        check(False, f"auto-bin driver ran ({p.stderr.strip()[:300]})")
        return
    got = json.loads(p.stdout)

    def close(a, b):
        return math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-6)

    for key, vals in cases.items():
        g, ref = got[key], auto_bins_reference(vals, job[key]["auto"])
        b = g["bins"]
        same_edges = b["step"] == ref["step"] and len(b["edges"]) == len(ref["edges"]) and all(
            close(x, y) for x, y in zip(b["edges"], ref["edges"]))
        check(same_edges and len(b["labels"]) == ref["n"] + 2 and b["labels"][0].startswith("<")
              and b["labels"][-1].startswith("≥") and b["closed"] == "left"
              and len(set(b["labels"])) == len(b["labels"]),
              f"{key}: step {b['step']}, {ref['n']} bins + 2 end buckets, edges equal the reference, labels distinct "
              f"({b['labels'][0]} … {b['labels'][-1]})")
        spec = {"bins": [-math.inf] + ref["edges"] + [math.inf], "labels": b["labels"], "right": False}
        df = pd.DataFrame({"v": pd.Series([math.nan if x is None else x for x in vals], dtype=float), "pnl": job[key]["pnl"]})
        df = calc.apply_bin_spec(df, "v", spec)
        want_codes = [-1 if pd.isna(x) else b["labels"].index(x) for x in df["v_bin"]]
        check(g["codes"] == want_codes, f"{key}: every value lands in the bin pd.cut puts it in")
        bs = calc.calculate_bin_stats(df, "v_bin")
        filled = [r for r in g["rows"] if r["count"]]
        ok = len(filled) == len(bs) and all(
            a["label"] == str(r["bin"]) and a["count"] == int(r["count"]) and close(a["total"], float(r["total_pnl"]))
            and close(a["avg"], float(r["avg_pnl"])) and close(a["win"], float(r["win_rate"]))
            for a, (_, r) in zip(filled, bs.iterrows()))
        check(ok and len(g["rows"]) == len(b["labels"]),
              f"{key}: filled bins equal calculate_bin_stats; empty bins kept ({len(filled)} of {len(g['rows'])} filled)")

    nv = got["nice_vol_pts"]["bins"]
    check(18 <= len(nv["labels"]) - 2 <= 32 and nv["step"] in (0.5, 1, 2, 2.5)
          and all(float(f"{e:.12g}") == e for e in nv["edges"]),
          f"vol points: a nice step ({nv['step']}), about 24 bins, edges with no float dust ({nv['labels'][1]})")
    ts = got["nice_tiny_slope"]["bins"]
    check(ts["step"] < 0.001 and len(set(ts["labels"])) == len(ts["labels"]),
          f"a 0.004-scale slope gets a sub-0.001 step and labels with enough decimals to stay distinct ({ts['labels'][1]})")
    f5 = got["five_dollar"]
    check(20 <= len(f5["bins"]["labels"]) - 2 <= 28 and f5["rows"][0]["count"] > 0 and f5["rows"][-1]["count"] > 0,
          f"~$5 premium: about 24 bins, and the outliers past p1/p99 sit in the end buckets "
          f"(${f5['bins']['step']} step, {f5['rows'][0]['count']} low, {f5['rows'][-1]['count']} high)")
    old = next(m for m in REGISTRY if m["key"] == "premium")
    check(old["bins"]["edges"] is None and old["bins"]["labels"] is None,
          "the registry carries no fixed Premium edges for the page to fall back on")


RANK_DRIVER = r"""
const fs = require('fs');
let factory;
global.document = { addEventListener: (e, fn) => fn(), getElementById: () => null };
global.Alpine = { data: (_n, f) => { factory = f; } };
// The shared calculations, then the page: the bundle's functions are
// top-level declarations that read them, exactly as the browser loads
// the two script tags.
eval(fs.readFileSync(process.argv[2], 'utf8') + String.fromCharCode(10)
     + fs.readFileSync(process.argv[1], 'utf8'));
const job = JSON.parse(fs.readFileSync(0, 'utf8'));
const out = {};

// Pure helpers.
const rows = job.rows;
const view = o => obRankView(rows, { method: o.method, form: o.form, hidden: new Set(o.hidden || []),
                                     groups: job.groups, other: job.other });
out.spear = view({ method: 'spearman', form: 'all' });
out.pear = view({ method: 'pearson', form: 'all' });
out.z = view({ method: 'spearman', form: 'z' });
out.noIv = view({ method: 'spearman', form: 'all', hidden: ['iv'] });
const dates = ['2019-06-03', '2020-01-02', '2020-06-01', '2021-04-05', '2021-04-06', '2022-01-03', null];
out.cov = obCommonCoverage(dates, [0, 1, 2, 3, 4, 5, 6], ['2020-01-02', '2021-04-06', null, '2021-01-05']);
out.covOne = obCommonCoverage(dates, [0, 1, 2], ['2020-01-02', '2020-01-02']);
const cols = { date_opened: dates, time_opened: ['10:00:00', '15:30:00', null, '09:30:00', '12:00:00', '15:55:00', '10:00:00'],
               pnl: [1, 2, 3, 4, 5, 6, 7] };
out.tradesAll = obRankTrades(cols, [0, 1, 2, 3, 4, 5], null);
const sv = a => a.map(x => ({ survives: x }));
out.bh = { prefix: obBhBoundary(sv([true, true, false, false])), gap: obBhBoundary(sv([true, false, true, false, false])),
           none: obBhBoundary(sv([false, false])), all: obBhBoundary(sv([true, true])) };
out.tradesFrom = obRankTrades(cols, [0, 1, 2, 3, 4, 5, 6], '2021-04-06');

// Component, with the server stubbed.
const sent = [];
global.fetch = async (url, init) => {
  const j = b => ({ ok: true, status: 200, json: async () => b });
  if (url.endsWith('/surface/catalog')) return j({ metrics: job.catalog, first_date: '2020-01-02', last_date: '2026-09-14',
                                                    lookahead_confirmed: true, family_groups: job.groups, other_group: job.other,
                                                    form_labels: job.formLabels });
  if (url.endsWith('/surface/rank')) {
    const body = JSON.parse(init.body);
    sent.push(body.trades);
    return j({ rows: sent.length > 3 ? job.rows2 : rows, report: { with_bar: 3, no_bar: body.trades.length - 3, distinct_entries: 3 },
               lookahead_confirmed: true });
  }
  throw new Error('unexpected ' + url);
};
(async () => {
  const c = factory();
  c.$nextTick = f => f && f();
  c.registry = job.registry;
  c.setTrades(job.payload);
  await new Promise(r => setTimeout(r, 0));
  out.catalogLoaded = !!c.surf.catalog;
  out.headBefore = c.surfHeadline();
  await c.rankSurface();
  out.sent1 = sent[0];
  out.head1 = c.surfHeadline();
  out.stale1 = c.surfStale();
  c.filters.dateFrom = '2021-03-01'; c.recompute();
  out.staleAfterFilter = c.surfStale();
  await c.rankSurface();
  out.sent2 = sent[1];
  out.stale2 = c.surfStale();
  out.commonText = c.surfCommonText();
  c.resetAllFilters(); c.recompute();
  c.toggleCommon();
  out.staleAfterCommon = c.surfStale();
  await c.rankSurface();
  out.sent3 = sent[2];
  out.sub3 = c.surfSubline();
  out.noWarn = c.surfBhWarning();
  await c.rankSurface();
  out.bhWarn = c.surfBhWarning();
  out.tips = { bh: c.surfTip('bh'), common: c.surfTip('common'), spearman: c.surfTip('spearman'), pearson: c.surfTip('pearson') };
  c.setTrades(job.payload);
  out.clearedOnNewLog = c.surf.result === null;
  out.forms = c.surfaceForms().map(f => f.label);
  out.groups = c.surfaceGroups().map(g => g.label + ':' + g.families.join('+'));
  process.stdout.write(JSON.stringify(out));
})().catch(e => { console.error(e.stack); process.exit(1); });
"""


def check_surface_ranking_ui() -> None:
    """P6b in node with the shipped JS: the bar view (sort, method, form,
    hidden families, BH outline, opacity against the largest n in view), what
    common coverage costs and sends, and the component's request, headline and
    staleness against a stubbed server."""
    print("surface ranking chart (shipped JS)")
    import shutil
    if shutil.which("node") is None:
        print("  SKIP  node is not installed")
        NOT_RUN.append("surface ranking UI (node not installed)")
        return

    def row(col, fam, form, n, sp, pe, sp_bh, pe_bh):
        return {"column": col, "family": fam, "form": form, "tenor": "30d", "wing": "atm", "n": n, "bars": n,
                "spearman": sp, "spearman_p": sp_bh, "spearman_p_bh": sp_bh,
                "pearson": pe, "pearson_p": pe_bh, "pearson_p_bh": pe_bh}
    rows = [
        row("iv_a", "iv", "level", 1348, 0.10, -0.30, 0.20, 0.001),
        row("z_iv_a", "iv", "z", 1203, -0.25, 0.05, 0.01, 0.60),
        row("skew_a", "skew", "level", 400, 0.20, 0.20, 0.04, 0.04),
        row("rv_a", "rv", "chg_d", 1300, -0.02, -0.02, 0.90, 0.90),
        row("dead", "vov", "z", 2, None, None, None, None),
        row("new_fam", "brand_new", "level", 1000, 0.15, 0.15, 0.30, 0.30),
    ]
    # Ranked by |rho|: a survivor, two z-score non-survivors at lower n, then
    # a level survivor -- non-survivors LEFT of the line.
    rows2 = [row("lv_1", "iv", "level", 1351, 0.30, 0.30, 0.001, 0.001),
             row("z_1", "iv", "z", 1203, 0.28, 0.28, 0.07, 0.07),
             row("z_2", "skew", "z", 1210, -0.27, -0.27, 0.08, 0.08),
             row("lv_2", "skew", "level", 1340, 0.26, 0.26, 0.02, 0.02),
             row("lv_3", "rv", "level", 1348, 0.05, 0.05, 0.60, 0.60)]
    catalog = [{"column_name": r["column"], "family": r["family"], "form": r["form"], "min_date": d,
                "description": "desc " + r["column"], "formula": "f", "units": "vol_decimal"}
               for r, d in zip(rows, ["2020-01-02", "2021-04-05", "2020-01-02", "2021-01-05", "2021-04-05", "2020-01-02"])]
    cols = {"date_opened": ["2018-01-02", "2019-05-06", "2020-03-02", "2021-02-01", "2021-06-01", "2022-06-01"],
            "date_closed": ["2018-01-05", "2019-05-09", "2020-03-05", "2021-02-04", "2021-06-04", "2022-06-04"],
            "time_opened": ["15:30:00"] * 6, "pnl": [10.0, -5.0, 20.0, 0.0, 7.0, -3.0], "days_in_trade": [3] * 6,
            "day_of_week": [1, 0, 0, 0, 1, 2]}
    payload = {"n": 6, "columns": cols, "date_min": "2018-01-02", "date_max": "2022-06-04", "notes": {},
               "suggested_name": "t", "market": {"joined": True, "spx_sessions": []}}
    reg = json.loads(json.dumps(REGISTRY))
    from app.oo_backtest import surface
    p = subprocess.run(["node", "-e", RANK_DRIVER, str(JS), str(CORE)],
                       input=json.dumps({"rows": rows, "rows2": rows2, "catalog": catalog, "payload": payload, "registry": reg,
                                         "groups": surface.FAMILY_GROUPS, "other": surface.OTHER_GROUP,
                                         "formLabels": surface.FORM_LABELS}),
                       capture_output=True, text=True, encoding="utf-8")
    if p.returncode:
        check(False, f"rank driver ran ({p.stderr.strip()[:400]})")
        return
    o = json.loads(p.stdout)
    names = lambda v: [b["column"] for b in v["bars"]]   # noqa: E731
    check(names(o["spear"]) == ["z_iv_a", "skew_a", "new_fam", "iv_a", "rv_a"],
          f"Spearman: sorted by |rho| descending, sign kept ({names(o['spear'])})")
    check(names(o["pear"]) == ["iv_a", "skew_a", "new_fam", "z_iv_a", "rv_a"] and o["pear"]["bars"][0]["value"] == -0.30,
          f"Pearson re-sorts on |r| and draws r ({names(o['pear'])})")
    check(names(o["z"]) == ["z_iv_a"] and o["z"]["undefinedInView"] == 1,
          "form filter z: only z metrics; the z metric with no correlation is counted, not drawn")
    sv = {b["column"]: b["survives"] for b in o["spear"]["bars"]}
    pv = {b["column"]: b["survives"] for b in o["pear"]["bars"]}
    check(sv == {"z_iv_a": True, "skew_a": True, "new_fam": False, "iv_a": False, "rv_a": False}
          and pv["iv_a"] is True and pv["z_iv_a"] is False and o["spear"]["survivorsAll"] == 2,
          "BH outline follows the sorting method's adjusted p at q 0.05")
    a = {b["column"]: b["alpha"] for b in o["spear"]["bars"]}
    a2 = {b["column"]: b["alpha"] for b in o["noIv"]["bars"]}
    check(abs(a["skew_a"] - (0.12 + 0.88 * 400 / 1348)) < 1e-9 and a2["skew_a"] > a["skew_a"]
          and "iv_a" not in a2 and o["noIv"]["survivorsAll"] == 2,
          "opacity is n against the largest n IN VIEW; hiding iv re-packs and lifts the rest; BH count over all computed is unchanged")
    bh = o["bh"]
    check(bh["prefix"] == {"index": 2, "failLeft": 0, "survivors": 2} and bh["gap"] == {"index": 3, "failLeft": 1, "survivors": 2}
          and bh["none"]["index"] is None and bh["all"]["index"] == 2,
          "BH line goes after the last survivor; a non-survivor left of it (smaller n) is counted, not hidden")
    other = next(b for b in o["spear"]["bars"] if b["column"] == "new_fam")
    check(other["group"]["label"] == "Other", "a family with no assigned hue is drawn grey under Other")
    cv = o["cov"]
    check((cv["earliestStart"], cv["commonStart"], cv["beforeEarliest"], cv["dropped"], cv["kept"])
          == ("2020-01-02", "2021-04-06", 1, 3, 2),
          f"common coverage: from the latest start; drops trades in [earliest, common) — the earliest start day "
          f"itself counts as dropped, the common start day is kept ({cv})")
    check(o["covOne"]["dropped"] == 0 and o["covOne"]["commonStart"] == o["covOne"]["earliestStart"],
          "one shared start: common coverage costs nothing")
    check(o["tradesAll"][0] == ["2019-06-03", "10:00:00", 1] and o["tradesAll"][2] == ["2020-06-01", None, 3]
          and [t[0] for t in o["tradesFrom"]] == ["2021-04-06", "2022-01-03"],
          "request trades: [date, time, pnl], null time kept; common coverage sends only trades on/after its start")

    check(o["catalogLoaded"] and "entered before the metrics in view start (2020-01-02)" in o["headBefore"]
          and o["headBefore"].startswith("2 of 6"),
          f"before ranking, the headline says how many filtered trades predate coverage ({o['headBefore']})")
    check(len(o["sent1"]) == 6 and o["head1"].startswith("3 of 6 trades have a metric bar")
          and "3 don't: 2 entered before coverage, 1 with no bar at the entry time" in o["head1"] and o["stale1"] is False,
          f"after ranking: prominent 'with a bar' count and the no-bar split ({o['head1']})")
    check(o["staleAfterFilter"] is True and len(o["sent2"]) == 2 and o["stale2"] is False,
          "a filter change marks the ranking stale (no request); Recompute sends only the filtered trades")
    check(o["staleAfterCommon"] is True and [t[0] for t in o["sent3"]] == ["2021-06-01", "2022-06-01"]
          and "common coverage from 2021-04-05" in o["sub3"],
          f"enabling common coverage marks it stale; the next request sends only trades from the common start ({o['sub3']})")
    check(o["clearedOnNewLog"], "loading another log clears the previous ranking")
    check(o["noWarn"] == "" and o["bhWarn"].startswith("2 bars left of the BH line do NOT survive (2 Z-score)")
          and "n 1,203–1,210 against 1,340–1,351" in o["bhWarn"] and "Common coverage only" in o["bhWarn"],
          f"prominent BH warning names the forms and n ranges of non-survivors left of the line ({o['bhWarn'][:120]})")
    t = o["tips"]
    check("Level from 2020-01-02" in t["common"] and "Z-score from 2021-04-05" in t["common"]
          and "2021 at the time" not in t["common"] and "of 5 looking" in t["bh"] and "not a cutoff on |r|" in t["bh"]
          and "RANKS" in t["spearman"] and "LINEAR" in t["pearson"],
          "tooltips: BH, Spearman, Pearson, common coverage -- coverage starts and the chance count come from the data, not text")
    check(o["forms"] == ["All forms", "Level", "Daily chg", "Z-score"]
          and o["groups"] == ["IV:iv", "Skew:skew", "Realized & VRP:rv", "Spot dynamics:vov", "Other:brand_new"],
          f"form buttons and legend groups come from the catalog response; an unassigned family lands in Other "
          f"({o['forms']}; {o['groups']})")
    cat_rows = pd.read_csv(ROOT / "surface_metrics_catalog.csv")
    ranked_fams = set(cat_rows.loc[~cat_rows["family"].isin(surface.EXCLUDED_FAMILIES), "family"])
    assigned = {f for g in surface.FAMILY_GROUPS for f in g["families"]}
    check(ranked_fams == assigned and len(surface.FAMILY_GROUPS) == 8,
          f"every ranked family in the real catalog has one of the 8 hues, and no hue names a family that is not there "
          f"(unassigned {sorted(ranked_fams - assigned)}, stale {sorted(assigned - ranked_fams)})")


ROWS_DRIVER = r"""
const fs = require('fs');
let factory;
global.document = { addEventListener: (e, fn) => fn(), getElementById: () => null };
global.Alpine = { data: (_n, f) => { factory = f; } };
// The shared calculations first, in the SAME eval: `obNull` is a const,
// which does not leak from one eval to the next.
eval(fs.readFileSync(process.argv[2], 'utf8') + String.fromCharCode(10)
     + fs.readFileSync(process.argv[1], 'utf8') + ';globalThis.OBD = OB_DATA;');
const job = JSON.parse(fs.readFileSync(0, 'utf8'));
const out = { valuesCalls: [] };
global.fetch = async (url, init) => {
  const j = (b, ok = true) => ({ ok, status: ok ? 200 : 400, json: async () => b });
  if (url.endsWith('/surface/catalog')) return j(job.catalogBody);
  if (url.endsWith('/surface/values')) {
    const body = JSON.parse(init.body);
    out.valuesCalls.push({ column: body.column, n: body.trades.length, first: body.trades[0] });
    if (body.column === 'broken') return j({ detail: 'Unknown or unranked surface metric' }, false);
    const vals = job.values[body.column].slice(0, body.trades.length);
    return j({ values: vals, report: { no_bar: vals.filter(v => v === null).length } });
  }
  if (url.endsWith('/surface/rank')) return j({ rows: job.rankRows, report: { with_bar: 1, no_bar: 0, distinct_entries: 1 } });
  throw new Error('unexpected ' + url);
};
(async () => {
  const c = factory();
  c.$nextTick = f => f && f();
  c.registry = job.registry;
  c.setTrades(job.payload);
  await new Promise(r => setTimeout(r, 0));
  const R = k => c.surfRows.find(r => r.surfColumn === k);

  await c.addSurfRow('iv_30d_atm');
  const iv = R('iv_30d_atm');
  out.iv = { key: iv.key, format: iv.format, scale: iv.scale, state: c.sectionState(iv), sub: c.sectionSub(iv),
             col: OBD.columns['surface__iv_30d_atm'], bins: c.binsFor(iv), step: c.stepText(iv),
             fit: c.fitText(iv), inSections: c.sectionMetrics().map(m => m.key).includes(iv.key),
             valued: c.sections[iv.key] && c.sections[iv.key].valued };
  await c.addSurfRow('iv_30d_atm');
  out.dupCalls = out.valuesCalls.length;

  c.filters.dateFrom = '2021-06-01'; c.recompute();
  out.filtered = { valued: c.sections[iv.key].valued, count: c.filteredCount, calls: out.valuesCalls.length,
                   binsSame: JSON.stringify(c.binsFor(iv)) === JSON.stringify(out.iv.bins) };
  c.resetAllFilters(); c.recompute();

  // ── per-row filter and scope (P6d)
  const dow = () => c.sections.day_of_week.valued;
  const rf = iv.rowFilter;
  out.rf0 = { min: rf.min, max: rf.max, step: rf.step, scope: rf.scope, n: rf.n, text: c.rowScopeText(iv) };
  c.setRowLo(iv, 20); c.setRowHi(iv, 26);
  out.rowScope = { count: c.filteredCount, rowValued: c.sections[iv.key].valued, dow: dow(), active: c.activeCount(),
                   specs: c.activeSpecs().length, sub: c.sectionSub(iv), text: c.rowScopeText(iv) };
  const inRange = OBD.columns['surface__iv_30d_atm'].filter(v => v !== null && v >= 20 && v <= 26).length;
  out.inRange = inRange;
  c.setRowScope(iv, 'page');
  out.pageScope = { count: c.filteredCount, dow: dow(), active: c.activeCount(), text: c.rowScopeText(iv),
                    stale: null };
  c.resetRowFilter(iv);
  out.pageUntouched = { count: c.filteredCount, active: c.activeCount(), text: c.rowScopeText(iv) };
  c.setRowHi(iv, 26);
  c.resetAllFilters();
  out.afterResetAll = { count: c.filteredCount, lo: iv.rowFilter.lo === iv.rowFilter.min, hi: iv.rowFilter.hi === iv.rowFilter.max,
                        scope: iv.rowFilter.scope };
  c.setRowHi(iv, 26);
  out.beforeRemovePage = c.filteredCount;
  c.setRowScope(iv, 'row');
  c.resetRowFilter(iv);

  await c.addSurfRow('z_iv_30d_atm');
  out.z = { state: c.sectionState(R('z_iv_30d_atm')), skipped: c.skippedText(R('z_iv_30d_atm')),
            rowFilter: R('z_iv_30d_atm').rowFilter };
  // A page-scoped filter on a row that is then removed: the page recovers.
  const ivRow = R('iv_30d_atm');
  c.setRowScope(ivRow, 'page'); c.setRowHi(ivRow, 22);
  out.pageBeforeRemoval = c.filteredCount;
  await c.addSurfRow('broken');
  out.broken = { state: c.sectionState(R('broken')), error: R('broken').error };

  out.groups = c.surfOptionGroups().map(g => g.label + ':' + g.options.map(o => (o.added ? '+' : '') + o.value).join(','));

  await c.rankSurface();
  c.surfRows.filter(r => r.surfColumn !== 'iv_30d_atm').forEach(r => c.removeSurfRow(r));
  c.removeSurfRow(R('iv_30d_atm'));
  out.afterPageRowRemoved = c.filteredCount;
  out.removed = { rows: c.surfRows.length, col: 'surface__iv_30d_atm' in OBD.columns,
                  present: c.presentColumns.includes('surface__iv_30d_atm'), sections: Object.keys(c.sections).filter(k => k.startsWith('surf_')) };
  c.surfClickBar(0);
  await new Promise(r => setTimeout(r, 0));
  out.clicked = c.surfRows.map(r => r.surfColumn);

  const before = out.valuesCalls.length;
  c.setTrades(job.payload2);
  await new Promise(r => setTimeout(r, 0));
  out.newLog = { calls: out.valuesCalls.length - before, n: out.valuesCalls[out.valuesCalls.length - 1].n,
                 col: OBD.columns['surface__' + out.clicked[0]] };
  process.stdout.write(JSON.stringify(out));
})().catch(e => { console.error(e.stack); process.exit(1); });
"""


def check_surface_rows() -> None:
    """P6c in node with the shipped JS against a stubbed server: adding a row
    fetches ONE metric for EVERY trade in the log, scales it to display units,
    bins it with nice steps, and from then on follows page filters without a
    request; duplicates, errors, coverage-only rows, removal, a bar click and
    a new log are each exercised."""
    print("added surface metric rows (shipped JS)")
    import shutil
    if shutil.which("node") is None:
        print("  SKIP  node is not installed")
        NOT_RUN.append("surface metric rows (node not installed)")
        return
    from app.oo_backtest import surface
    rng = random.Random(41)
    n = 60
    dates = [f"2021-{1 + i // 6:02d}-{1 + i % 6 * 4:02d}" for i in range(n)]
    cols = {"date_opened": dates, "date_closed": dates, "time_opened": ["15:30:00"] * n,
            "pnl": [round(rng.gauss(10, 200), 2) for _ in range(n)], "days_in_trade": [1] * n,
            "day_of_week": [i % 5 for i in range(n)]}
    payload = {"n": n, "columns": cols, "date_min": dates[0], "date_max": dates[-1], "notes": {},
               "suggested_name": "t", "market": {"joined": True, "spx_sessions": []}}
    payload2 = {**payload, "n": 40, "columns": {k: v[:40] for k, v in cols.items()}}
    values = {"iv_30d_atm": [None if i < 5 else round(rng.uniform(0.11, 0.32), 4) for i in range(n)],
              "z_iv_30d_atm": [None] * n}
    catalog = [{"column_name": "iv_30d_atm", "family": "iv", "form": "level", "units": "vol_decimal",
                "description": "30d ATM implied vol", "formula": "sigma", "min_date": "2021-01-10"},
               {"column_name": "z_iv_30d_atm", "family": "iv", "form": "z", "units": "z_score",
                "description": "z of iv", "formula": "z", "min_date": "2023-04-05"},
               {"column_name": "broken", "family": "skew", "form": "level", "units": "mystery",
                "description": "", "formula": "", "min_date": "2020-01-02"}]
    body = {"metrics": catalog, "family_groups": surface.FAMILY_GROUPS, "other_group": surface.OTHER_GROUP,
            "form_labels": surface.FORM_LABELS, "unit_formats": surface.UNIT_FORMATS,
            "default_unit_format": surface.DEFAULT_UNIT_FORMAT, "row_auto_bins": surface.ROW_AUTO_BINS}
    rank_rows = [{"column": "z_iv_30d_atm", "family": "iv", "form": "z", "tenor": None, "wing": None, "n": 50, "bars": 50,
                  "spearman": 0.3, "spearman_p": 0.01, "spearman_p_bh": 0.02, "pearson": 0.2, "pearson_p": 0.1, "pearson_p_bh": 0.2}]
    reg = json.loads(json.dumps(REGISTRY))
    p = subprocess.run(["node", "-e", ROWS_DRIVER, str(JS), str(CORE)],
                       input=json.dumps({"registry": reg, "payload": payload, "payload2": payload2, "values": values,
                                         "catalogBody": body, "rankRows": rank_rows}),
                       capture_output=True, text=True, encoding="utf-8")
    if p.returncode:
        check(False, f"rows driver ran ({p.stderr.strip()[:400]})")
        return
    o = json.loads(p.stdout)
    iv = o["iv"]
    first = o["valuesCalls"][0]
    check(first["column"] == "iv_30d_atm" and first["n"] == n and first["first"] == [dates[0], "15:30:00"],
          f"adding a row fetches that one metric for every trade in the log ({first['n']} of {n}), as [date, time]")
    want = [None if v is None else v * 100 for v in values["iv_30d_atm"]]
    check(iv["col"] == want and iv["scale"] == 100 and iv["format"] == {"decimals": 2, "suffix": "vol pts"},
          "vol_decimal values are scaled to vol points on arrival (0.1406 -> 14.06), formatted to 2 decimals")
    check(iv["state"] == "ready" and iv["inSections"] and iv["valued"] == 55
          and "55 of 60 trades with a value" in iv["sub"] and "vol pts bins (auto)" in iv["sub"] and "data from 2021-01-10" in iv["sub"],
          f"the row is a ready section of the page, rendered by the shared section code ({iv['sub']})")
    check(iv["bins"]["labels"][0].startswith("<") and iv["bins"]["labels"][-1].startswith("≥")
          and iv["step"].endswith("vol pts") and f"per {iv['step']}" in iv["fit"],
          f"auto bins with a nice step and end buckets; step and OLS slope in the metric's units ({iv['step']}; {iv['fit']})")
    check(o["dupCalls"] == 1, "adding the same metric again makes no request and no second row")
    f = o["filtered"]
    check(f["calls"] == 1 and f["valued"] < 55 and f["count"] < n and f["binsSame"],
          f"a page filter re-bins the row from the filtered trades with no request, and does not move its edges ({f})")
    r0 = o["rf0"]
    ivvals = [v * 100 for v in values["iv_30d_atm"] if v is not None]
    check(r0["min"] <= min(ivvals) and r0["max"] >= max(ivvals) and r0["scope"] == "row" and r0["n"] == 55
          and abs(r0["step"] - 0.1) < 1e-12 and r0["text"] == "",
          f"row filter: bounds from the row's own values in vol points, step a tenth of the bin step, row scope by default ({r0})")
    rs = o["rowScope"]
    check(rs["count"] == n and rs["dow"] == n and rs["rowValued"] == o["inRange"] and rs["active"] == 0 and rs["specs"] == 0
          and "row filter applied" in rs["sub"] and rs["text"] == "",
          f"row scope narrows only that row ({rs['rowValued']} in 20-26 vol pts); page count, other sections and "
          f"the active-filter count are untouched")
    ps = o["pageScope"]
    check(ps["count"] == o["inRange"] and ps["dow"] == o["inRange"] and ps["active"] == 1
          and ps["text"] == "Whole page: dropping 5 of 60 filtered trades with no value — 3 entered before its data starts (2021-01-10), 2 with no bar at the entry time",
          f"whole-page scope filters every section and states the coverage cost, split before-coverage / no bar ({ps['text']})")
    pu = o["pageUntouched"]
    check(pu["count"] == n and pu["active"] == 0 and pu["text"].startswith("Whole page: moving the slider will drop 5 of 60"),
          f"page scope with the slider untouched drops nothing, but says what narrowing will cost ({pu['text'][:60]})")
    ar = o["afterResetAll"]
    check(ar["count"] == n and ar["lo"] and ar["hi"] and ar["scope"] == "page",
          "reset all filters also resets row filters (keeping each row's scope)")
    check(o["pageBeforeRemoval"] < n and o["afterPageRowRemoved"] == n,
          f"removing a row whose filter was on the page re-filters the page ({o['pageBeforeRemoval']} -> {o['afterPageRowRemoved']})")
    check(o["z"]["rowFilter"] is None, "a row with no values gets no filter control")
    check(o["z"]["state"] == "skipped" and "its data starts 2023-04-05" in o["z"]["skipped"],
          f"a metric whose coverage starts after every trade says so instead of drawing nothing ({o['z']['skipped']})")
    check(o["broken"]["state"] == "error" and "Unknown or unranked" in o["broken"]["error"],
          "a failed fetch leaves an error row (removable), not a silent empty section")
    check(o["groups"] == ["IV · iv:+iv_30d_atm,+z_iv_30d_atm", "Skew · skew:+broken"],
          f"dropdown: optgroups per family in legend order, added metrics marked ({o['groups']})")
    r = o["removed"]
    check(r == {"rows": 0, "col": False, "present": False, "sections": []},
          "remove drops the row, its client column and its section state")
    check(o["clicked"] == ["z_iv_30d_atm"], "clicking a ranking bar adds that bar's metric (sorted view index)")
    nl = o["newLog"]
    check(nl["calls"] == 1 and nl["n"] == 40,
          f"a new log refetches each added row for its own trades ({nl['calls']} call, {nl['n']} trades)")
    cat = pd.read_csv(ROOT / "surface_metrics_catalog.csv")
    ranked_units = set(cat.loc[~cat["family"].isin(surface.EXCLUDED_FAMILIES), "units"])
    check(ranked_units <= set(surface.UNIT_FORMATS),
          f"every unit in the real ranked catalog has a display format ({sorted(ranked_units)}; "
          f"missing {sorted(ranked_units - set(surface.UNIT_FORMATS))})")


DEPLOY_DRIVER = r"""
const fs = require('fs');
global.document = { addEventListener: () => {} };
// The shared calculations, then the page: the bundle's functions are
// top-level declarations that read them, exactly as the browser loads
// the two script tags.
eval(fs.readFileSync(process.argv[2], 'utf8') + String.fromCharCode(10)
     + fs.readFileSync(process.argv[1], 'utf8'));
const job = JSON.parse(fs.readFileSync(0, 'utf8'));
const out = {};
for (const [key, t] of Object.entries(job)) {
  const idx = t.idx || [...Array(t.cols.pnl.length).keys()];
  const conc = obConcurrency(t.cols, idx, t.sessions);
  const stats = obStats(t.cols, idx);
  out[key] = { conc, stats, extra: obExtraStats(t.cols, idx, stats, t.capital, conc.peak) };
}
process.stdout.write(JSON.stringify(out, (k, v) => (v === Infinity ? 'Infinity' : v)));
"""


def _weekdays(lo: str, hi: str, drop=()) -> list[str]:
    days = pd.bdate_range(lo, hi)
    return [d.strftime("%Y-%m-%d") for d in days if d.strftime("%Y-%m-%d") not in set(drop)]


def _weekly_handover() -> dict:
    """Enter every Friday, close the next Friday. Always exactly one open."""
    fridays = [d.strftime("%Y-%m-%d")
               for d in pd.date_range("2021-01-08", "2021-12-31", freq="W-FRI")]
    return {"date_opened": fridays[:-1], "date_closed": fridays[1:],
            "pnl": [10.0] * (len(fridays) - 1),
            "days_in_trade": [7] * (len(fridays) - 1)}


def check_deployment_and_extra_stats() -> None:
    """Concurrent positions and the five added figures: shipped JS against a
    brute-force count and a direct pandas computation written here, not ported
    from the JS. Includes a six-month stretch with nothing open (must be a run
    of zeros, not a gap), a filtered subset, and the real MesoSim fixture's
    still-open positions (must contribute nothing)."""
    print("deployment + added stats: shipped JS vs brute force / pandas")
    import shutil
    if shutil.which("node") is None:
        print("  SKIP  node is not installed")
        NOT_RUN.append("deployment/extra-stats parity (node not installed)")
        return
    from app.oo_backtest import stats as pystats
    from app.oo_backtest.market import session_days

    rng = random.Random(23)
    holidays = ["2021-07-05", "2021-11-25", "2022-01-17"]
    sessions = _weekdays("2021-01-04", "2022-12-30", holidays)
    # Trades avoid 2021-09-01 .. 2022-02-28 entirely: six months flat at zero.
    usable = [d for d in sessions if not ("2021-08-20" <= d <= "2022-02-28")]
    trades = []
    for _ in range(160):
        a = rng.randrange(len(usable) - 30)
        b = min(a + rng.randint(0, 25), len(usable) - 1)
        if usable[a] <= "2021-08-20" < usable[b]:
            b = a
        trades.append((usable[a], usable[b], 0.0 if rng.random() < 0.05 else round(rng.gauss(50, 400), 2)))
    trades.append(("2021-07-05", "2021-07-07", 10.0))   # opens on a holiday: counted from the next session
    trades.sort()
    cols = {"date_opened": [t[0] for t in trades], "date_closed": [t[1] for t in trades],
            "pnl": [t[2] for t in trades], "days_in_trade": [1] * len(trades)}
    subset = [i for i in range(len(trades)) if rng.random() < 0.5]

    def brute(idx):
        # HALF-OPEN [open, close). The measure is OVERNIGHT capital: a
        # position is deployed on the sessions it is held through and not on
        # the one it closes on. Written here as the spec, not ported from the
        # JS -- a strategy entering every Friday and closing the next Friday
        # must read a constant 1, not 2 on Fridays.
        lo = min(cols["date_opened"][i] for i in idx)
        hi = max(cols["date_closed"][i] for i in idx)
        days = [d for d in sessions if lo <= d <= hi]
        return days, [sum(1 for i in idx
                          if cols["date_opened"][i] <= d < cols["date_closed"][i])
                      for d in days]

    def extra(idx, capital, peak):
        df = pd.DataFrame({k: [cols[k][i] for i in idx] for k in cols})
        df["date_opened"] = pd.to_datetime(df["date_opened"])
        df["date_closed"] = pd.to_datetime(df["date_closed"])
        st = pystats.calculate_stats(df.copy())
        # CLOSE-TO-CLOSE, as obExtraStats measures it and as the old
        # Render app did. Measuring from the first ENTRY stretched the
        # window by the first position's holding period and understated
        # every figure derived from it.
        years = (df["date_closed"].max() - df["date_closed"].min()).days / 365.25
        ann = st["total_pnl"] / years
        gw, gl = df.loc[df["pnl"] > 0, "pnl"].sum(), df.loc[df["pnl"] < 0, "pnl"].sum()
        return {"avg_annual_pnl": ann, "calmar": ann / abs(st["max_drawdown"]),
                "profit_factor": gw / abs(gl), "avg_pnl_pct": st["avg_pnl"] / capital * 100,
                "avg_annual_return_pct": ann / (peak * capital) * 100}

    # Real MesoSim 3.1 fixture: 4 closed trades through 2023-12-12, and two
    # positions (222, 223) still open from 2026-03. Sessions run to 2026-12-31.
    mp = _parse((FIXTURES / "v3_1_allantis_v2_mon.json").read_bytes(), "v3.events.json")
    mc = {k: mp["columns"][k] for k in ("date_opened", "date_closed", "pnl", "days_in_trade")}
    m_sessions = _weekdays("2021-01-04", "2026-12-31")
    planted = {k: list(v) for k, v in mc.items()}
    for d in ("2026-03-16", "2026-03-23"):   # what "running to the end" would look like
        planted["date_opened"].append(d); planted["date_closed"].append("2026-12-31")
        planted["pnl"].append(0.0); planted["days_in_trade"].append(None)

    job = {"all": {"cols": cols, "sessions": sessions, "capital": 10000},
           "subset": {"cols": cols, "sessions": sessions, "capital": 2500, "idx": subset},
           "nocap": {"cols": cols, "sessions": sessions, "capital": None},
           "nolosses": {"cols": {"date_opened": ["2021-01-04", "2021-02-01"], "date_closed": ["2021-01-05", "2021-02-02"],
                                 "pnl": [5.0, 7.0], "days_in_trade": [1, 1]}, "sessions": sessions, "capital": 1000},
           "nosessions": {"cols": cols, "sessions": [], "capital": 10000},
           # THE HAND-OVER. Enter every Friday, close the next Friday: one
           # position is held at all times, and the Friday a trade closes is
           # the Friday the next one opens. Counting the close day as well
           # drew 2 every Friday -- the same position counted twice on the
           # day it changes hands -- which is what half-open fixes.
           "handover": {"cols": _weekly_handover(), "sessions": sessions,
                        "capital": 10000},
           # NOTHING HELD OVERNIGHT: a 0DTE log. Zero is the right answer for
           # capital held through a close, and the pane has to say so rather
           # than draw a flat line and let it be discovered.
           "intraday": {"cols": {"date_opened": ["2021-03-01", "2021-03-02", "2021-03-03"],
                                 "date_closed": ["2021-03-01", "2021-03-02", "2021-03-03"],
                                 "pnl": [12.0, -4.0, 8.0], "days_in_trade": [0, 0, 0]},
                        "sessions": sessions, "capital": 10000},
           "meso": {"cols": mc, "sessions": m_sessions, "capital": 10000},
           "meso_planted": {"cols": planted, "sessions": m_sessions, "capital": 10000}}
    p = subprocess.run(["node", "-e", DEPLOY_DRIVER, str(JS), str(CORE)], input=json.dumps(job),
                       capture_output=True, text=True, encoding="utf-8")
    if p.returncode:
        check(False, f"deployment driver ran ({p.stderr.strip()[:300]})")
        return
    got = json.loads(p.stdout)

    def close(a, b):
        return a is not None and b is not None and math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-9)

    for key, idx in (("all", list(range(len(trades)))), ("subset", subset)):
        days, counts = brute(idx)
        c = got[key]["conc"]
        check(c["days"] == days and c["counts"] == counts,
              f"{key}: concurrency equals a brute-force count on every one of {len(days)} sessions")
        check(c["peak"] == max(counts) and c["peakDay"] == days[counts.index(max(counts))],
              f"{key}: peak {c['peak']} on {c['peakDay']} (the first day it is reached)")
        want = extra(idx, job[key]["capital"], max(counts))
        bad = {k: (got[key]["extra"][k], v) for k, v in want.items() if not close(got[key]["extra"][k], v)}
        check(not bad, f"{key}: Avg Annual P/L, Calmar, Profit Factor, Avg Annual Return %, Avg P/L % equal pandas"
              + (f" — differ {bad}" if bad else ""))

    c = got["all"]["conc"]
    gap = [n for d, n in zip(c["days"], c["counts"]) if "2021-09-01" <= d <= "2022-02-28"]
    check(len(gap) > 120 and not any(gap),
          f"six months with nothing open: {len(gap)} consecutive sessions at zero, not a collapsed gap")
    check("2021-07-05" not in c["days"] and c["offSession"] == 1,
          f"a trade opening on a non-session day is counted from the next session and reported ({c['offSession']})")
    check(got["nocap"]["extra"]["avg_pnl_pct"] is None and got["nocap"]["extra"]["avg_annual_return_pct"] is None
          and got["nocap"]["extra"]["calmar"] is not None,
          "no capital: the two % figures are null; capital-free figures still computed")
    check(got["nolosses"]["extra"]["profit_factor"] == "Infinity", "profit factor with wins and no losses is Infinity")
    check(got["nosessions"]["conc"]["days"] == [] and got["nosessions"]["conc"]["peak"] == 0,
          "no session list (market not joined): an empty series, no invented calendar")

    # HALF-OPEN, the two cases that name what it means.
    hv = got["handover"]["conc"]
    # Every session carries exactly one, EXCEPT the last: the series runs to
    # the final close, and on that day the last trade is gone with nothing
    # opened behind it. Zero there is the same rule, not an exception to it.
    check(hv["peak"] == 1 and set(hv["counts"][:-1]) == {1} and hv["counts"][-1] == 0,
          f"weekly hand-over: peak {hv['peak']}, counts {sorted(set(hv['counts']))} "
          f"(last {hv['counts'][-1]}) — one position is held at all times, and "
          f"the Friday one closes is the Friday the next opens; counting the "
          f"close day draws 2 there")
    check(hv["sameSession"] == 0,
          f"{hv['sameSession']} hand-over trades were counted as intraday")

    it = got["intraday"]["conc"]
    check(it["peak"] == 0 and not any(it["counts"]),
          f"a log that never holds overnight reports peak {it['peak']}; by this "
          f"measure -- capital still at risk at the close -- it deploys none")
    check(it["sameSession"] == 3 and it["counted"] == 3,
          f"the intraday trades are not counted as such ({it['sameSession']} of "
          f"{it['counted']}), so the pane could not explain its flat zero line")

    m, pl = got["meso"]["conc"], got["meso_planted"]["conc"]
    check(mp["notes"]["open_positions"] == 2 and mp["n"] == 4 and m["days"][-1] == "2023-12-12"
          and all(d <= "2023-12-12" for d in m["days"]) and m["unclosed"] == 0,
          f"MesoSim: the 2 still-open positions are not trades, so the series ends at the last exit ({m['days'][-1]})")
    check(pl["days"][-1] == "2026-12-31" and pl["counts"][pl["days"].index("2026-03-23")] == 2,
          "planted: had they been kept with an open-ended exit they WOULD run to the end — the check can tell")

    # session_days: SPX sessions only, clipped to first entry .. last exit.
    daily = pd.DataFrame({"trade_date": pd.to_datetime(["2026-04-06", "2026-04-07", "2026-04-08", "2026-04-09", "2026-04-10"]).date,
                          "spx_session": [True, True, False, True, True]})
    tdf = pd.DataFrame({"date_opened": pd.to_datetime(["2026-04-07"]), "date_closed": pd.to_datetime(["2026-04-09"])})
    sd = session_days(daily, tdf)
    check(sd == ["2026-04-07", "2026-04-09"],
          f"session_days: SPX sessions within the log's span; 2026-04-08 (no SPX bars) is not one ({sd})")


COMPONENT_DRIVER = r"""
const fs = require('fs');
let factory;
global.document = { addEventListener: (e, fn) => fn(), getElementById: () => null };
global.Alpine = { data: (_n, f) => { factory = f; } };
// The shared calculations, then the page: the bundle's functions are
// top-level declarations that read them, exactly as the browser loads
// the two script tags.
eval(fs.readFileSync(process.argv[2], 'utf8') + String.fromCharCode(10)
     + fs.readFileSync(process.argv[1], 'utf8'));
const job = JSON.parse(fs.readFileSync(0, 'utf8'));
const c = factory();
c.$nextTick = f => f && f();
c.registry = job.registry;
c.setTrades(job.payload);
const R = k => c.registry.find(m => m.key === k);
const out = {};
const copy = v => JSON.parse(JSON.stringify(v));   // snapshots, not live references
const snap = () => ({ count: c.filteredCount, specs: c.activeSpecs().map(s => s.kind + ':' + s.column), active: c.activeCount() });
out.initial = copy({ ...snap(), vix: c.rangeOf(R('vix')), premium: c.rangeOf(R('premium')), dow: c.catOf(R('day_of_week')),
                exit: c.catOf(R('exit_reason')), dates: c.dateBounds, total: c.stats.total_pnl });
c.setHi(R('vix'), c.rangeOf(R('vix')).max); out.vixFull = snap();
c.setHi(R('vix'), 25); out.vixNarrow = copy({ ...snap(), vix: c.rangeOf(R('vix')) });
c.setLo(R('vix'), 99); out.loClamped = copy(c.rangeOf(R('vix')));
c.resetFilter(R('vix'));
c.toggleCat(R('day_of_week'), 0); out.noMonday = snap();
c.setAllCats(R('exit_reason'), false); out.noExit = snap();
c.resetAllFilters(); out.reset = snap();
c.filters.dateFrom = '2021-03-01'; c.onFilterChange(); out.date = snap();
c.resetAllFilters();
c.setHi(R('vix3m_vix'), c.rangeOf(R('vix3m_vix')).min); const before = copy(c.rangeOf(R('vix3m_vix')));
c.setRatioBasis('close'); out.basis = { before, after: c.rangeOf(R('vix3m_vix')), specs: c.activeSpecs().map(s => s.column) };
c.setRatioBasis('entry'); c.resetAllFilters();
const states = () => Object.fromEntries(c.sectionMetrics().map(m => [m.key, c.sectionState(m)]));
out.sectionsAll = { states: states(), vix: c.sections.vix, sub: c.sectionSub(R('vix')) };
// Only the 2021-02-01 trade, whose VIX is null: the log HAS VIX, this filter does not.
c.filters.dateFrom = '2021-02-01'; c.filters.dateTo = '2021-02-01'; c.onFilterChange();
out.sectionsOne = { count: c.filteredCount, states: states() };
c.resetAllFilters();
const premBins = () => JSON.stringify(c.binsFor(R('premium')));
out.autoBefore = premBins(); out.autoSub = c.sectionSub(R('premium'));
c.setHi(R('premium'), 1000); out.autoAfter = premBins(); out.autoCount = c.filteredCount;
c.resetAllFilters();
const cap = () => ({ count: c.filteredCount, pnlPct: c.stat('avg_pnl_pct'), retPct: c.stat('avg_annual_return_pct'),
                     pf: c.stat('profit_factor'), calmar: c.stat('calmar'), extra: c.extra, deploy: c.deploy });
out.cap10k = cap();
c.capitalInput = ''; c.recomputeCapital(); out.capBlank = cap();
c.capitalInput = '100'; c.recomputeCapital(); out.cap100 = cap();
c.capitalInput = '50'; c.recomputeCapital(); out.cap5k = cap();
c.capitalInput = '-3'; c.recomputeCapital(); out.capBad = cap();
c.setTrades({ ...job.payload, saved: { id: 7, name: 's', capital_per_position: 2500 } }); out.capSaved = c.capitalInput;
c.setTrades({ ...job.payload, saved: { id: 8, name: 't', capital_per_position: null } }); out.capSavedNull = c.capitalInput;
process.stdout.write(JSON.stringify(out));
"""


def check_component_filters() -> None:
    """The page component's filter state, driven in node with the shipped JS:
    bounds from the data, what counts as an ACTIVE filter, clamping, reset, the
    ratio-basis switch, and that the shown count follows."""
    print("filter component (shipped JS, stubbed Alpine)")
    import shutil
    if shutil.which("node") is None:
        print("  SKIP  node is not installed")
        NOT_RUN.append("filter component behaviour (node not installed)")
        return
    cols = {
        "date_opened": ["2021-01-04", "2021-02-01", "2021-03-01", "2021-04-05", "2021-05-03", "2021-06-07"],
        "date_closed": ["2021-01-20", "2021-02-15", "2021-03-19", "2021-04-20", "2021-05-20", "2021-06-21"],
        "pnl": [100.0, -50.0, 200.0, 0.0, -300.0, 75.0],
        "days_in_trade": [16, 14, 18, 15, 17, 14],
        "day_of_week": [0, 0, 0, 0, 0, 1],
        "exit_reason": ["Profit Target", "Stop Loss", "Profit Target", "Expired", "Stop Loss", None],
        "premium": [1250.0, 900.0, 1600.0, -500.0, 1250.0, 900.0],
        "vix_level": [17.2, None, 22.9, 31.5, 19.0, 25.0],
        "vix3m_vix_ratio_entry": [1.10, None, 1.02, 0.95, 1.08, 1.11],
        "vix3m_vix_ratio_close": [1.20, 1.05, 0.99, 0.90, 1.15, 1.30],
    }
    payload = {"n": 6, "columns": cols, "date_min": "2021-01-04", "date_max": "2021-06-21", "notes": {},
               "suggested_name": "t", "market": {"joined": True}}
    reg = json.loads(json.dumps(REGISTRY))
    p = subprocess.run(["node", "-e", COMPONENT_DRIVER, str(JS), str(CORE)], input=json.dumps({"registry": reg, "payload": payload}),
                       capture_output=True, text=True, encoding="utf-8")
    if p.returncode:
        check(False, f"component driver ran ({p.stderr.strip()[:300]})")
        return
    o = json.loads(p.stdout)
    i = o["initial"]
    check(i["count"] == 6 and i["specs"] == [] and i["active"] == 0 and i["total"] == 25.0,
          f"on load: every trade shown, no filter active ({i['count']}, {i['specs']})")
    check((i["vix"]["min"], i["vix"]["max"], i["vix"]["n"]) == (17, 32, 5) and (i["premium"]["min"], i["premium"]["max"]) == (-500, 1600),
          f"range bounds come from the data, snapped outward to the step (vix {i['vix']['min']}..{i['vix']['max']}, premium {i['premium']['min']}..{i['premium']['max']})")
    check([x["label"] for x in i["dow"]["options"]] == ["Mon", "Tue", "Wed", "Thu", "Fri"] and len(i["dow"]["selected"]) == 5
          and [x["label"] for x in i["exit"]["options"]] == ["Expired", "Profit Target", "Stop Loss", "(none)"],
          f"day-of-week offers Mon-Fri; exit reasons come from the log, a null reason included; all on "
          f"({[x['label'] for x in i['dow']['options']]}, {[x['label'] for x in i['exit']['options']]})")
    check(i["dates"] == {"min": "2021-01-04", "max": "2021-06-07"}, f"date bounds from the entry dates ({i['dates']})")
    check(o["vixFull"]["specs"] == [] and o["vixFull"]["count"] == 6,
          "a slider moved back to its extent is not a filter (null VIX still shown)")
    check(o["vixNarrow"]["specs"] == ["range:vix_level"] and o["vixNarrow"]["count"] == 4,
          f"narrowing VIX to 17-25 keeps 17.2, 22.9, 19.0 and 25.0 (inclusive) and drops the null and 31.5 ({o['vixNarrow']['count']})")
    check(o["loClamped"]["lo"] == o["loClamped"]["hi"] == 25,
          f"a lower handle dragged past the upper one stops at it (lo {o['loClamped']['lo']}, hi {o['loClamped']['hi']})")
    check(o["noMonday"]["count"] == 1 and o["noMonday"]["specs"] == ["set:day_of_week"], "unticking Monday leaves the Tuesday trade")
    check(o["noExit"]["count"] == 0, "no exit reasons ticked -> zero trades (the empty state)")
    check(o["reset"]["count"] == 6 and o["reset"]["active"] == 0, "reset all filters restores every trade")
    check(o["date"]["count"] == 4 and o["date"]["specs"] == ["date:date_opened"], "a from-date of 2021-03-01 keeps the last four")
    b = o["basis"]
    check(b["before"]["max"] < 1.2 and b["after"]["max"] == 1.3 and b["specs"] == [],
          f"switching ratio basis rebuilds that filter from the new column and drops the old narrowing ({b['after']})")
    ab = json.loads(o["autoBefore"])
    check(ab["step"] == 100 and ab["edges"][0] == -500 and ab["edges"][-1] == 1600 and o["autoBefore"] == o["autoAfter"]
          and o["autoCount"] < 6 and "$100 bins (auto)" in o["autoSub"],
          f"Premium auto bins come from the whole log, do not move when a filter narrows it, and the header names the step "
          f"({ab['step']}, {ab['edges'][0]}..{ab['edges'][-1]}; {o['autoSub']})")
    k10, kb, k5, kbad = o["cap10k"], o["capBlank"], o["cap5k"], o["capBad"]
    k100 = o["cap100"]
    check(k10["pnlPct"] == "0.04%" and k100["pnlPct"] == "4.17%" and k5["pnlPct"] == "8.33%" and k5["count"] == k10["count"] == 6,
          f"capital changes Avg P/L % (avg $4.17: of $10k, $100, $50) without touching the trades ({k10['pnlPct']}, {k100['pnlPct']}, {k5['pnlPct']})")
    check(kb["pnlPct"] == "" and kb["retPct"] == "" and kbad["pnlPct"] == "" and kb["pf"] == k10["pf"] and kb["calmar"] == k10["calmar"],
          f"blank or non-positive capital: both % figures blank, the others unchanged ({kb['pnlPct']!r}, {kbad['pnlPct']!r})")
    check(o["capSaved"] == "2500" and o["capSavedNull"] == "10000",
          f"a loaded saved strategy brings its capital; none stored means the $10,000 default ({o['capSaved']}, {o['capSavedNull']})")
    check(k10["deploy"]["hasSessions"] is False and k10["deploy"]["peak"] == 0,
          "no spx_sessions in the payload: deployment has no series and says so (hasSessions false)")
    sa, so = o["sectionsAll"], o["sectionsOne"]
    want_all = {"day_of_week": "ready", "year": "skipped", "gap": "skipped", "vix_gap": "skipped", "premium": "ready",
                "vix": "ready", "vix3m": "skipped", "vix9d": "skipped", "vix3m_vix": "ready", "vix_vix9d": "skipped"}
    check(sa["states"] == want_all,
          f"on load: a section is ready where the log has values, skipped where the column is absent ({sa['states']})")
    check(sa["vix"]["valued"] == 5 and sa["vix"]["fit"] is not None and "5 of 6 trades" in sa["sub"],
          f"VIX section: 5 of 6 trades have a value, and the header says so ({sa['sub']})")
    check(so["count"] == 1 and so["states"]["vix"] == "nodata" and so["states"]["premium"] == "ready"
          and so["states"]["gap"] == "skipped",
          f"a filter leaving no VIX value makes that section 'nodata', distinct from 'skipped' ({so['states']})")


def main() -> int:
    check_registry()
    check_against_source()
    check_binning()
    check_stats_parity()
    check_section_parity()
    check_auto_bins()
    check_component_filters()
    check_deployment_and_extra_stats()
    check_surface_ranking_ui()
    check_surface_rows()
    check_dropped_scope()
    check_parsers()
    check_real_mesosim()
    check_surface_stats()
    print()
    if FAILS:
        print(f"FAIL: {len(FAILS)} check(s) failed" + (f"; also not run: {'; '.join(NOT_RUN)}" if NOT_RUN else ""))
        return 1
    if NOT_RUN:
        print(f"NOT FULLY RUN: everything that ran passed, but {len(NOT_RUN)} part(s) could not run here: "
              + "; ".join(NOT_RUN))
        return EXIT_SKIPPED
    print("PASS: OO/Mesosim Backtest registry, binning parity, payload and parsers")
    return 0


if __name__ == "__main__":
    sys.exit(main())
