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

    python scripts/check_oo_backtest.py
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
from app.oo_backtest.payload import TRADE_COLUMNS, trades_to_payload  # noqa: E402
from app.oo_backtest.registry import REGISTRY  # noqa: E402
from app.routers.oo_backtest import _parse  # noqa: E402

JS = ROOT / "static" / "js" / "oo_backtest.js"

FAILS: list[str] = []


def check(ok: bool, msg: str) -> None:
    print(("  ok    " if ok else "  FAIL  ") + msg)
    if not ok:
        FAILS.append(msg)


# Which pandas helper each range metric's column is binned by in the source.
HELPERS = {
    "gap": calc.create_gap_bins,
    "vix_gap": calc.create_vix_gap_bins,
    "premium": calc.create_premium_bins,
    "vix": calc.create_vix_bins,
    "vix3m": calc.create_vix_bins,
    "vix9d": calc.create_vix_bins,
    "vix3m_vix": calc.create_ratio_bins,
    "vix_vix9d": calc.create_ratio_bins,
}

DRIVER = r"""
const fs = require('fs');
global.document = { addEventListener: () => {} };
eval(fs.readFileSync(process.argv[1], 'utf8'));
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
    p = subprocess.run(["node", "-e", DRIVER, str(JS)], input=json.dumps(job),
                       capture_output=True, text=True, encoding="utf-8")
    if p.returncode:
        raise RuntimeError(p.stderr.strip())
    return json.loads(p.stdout)


def check_binning() -> None:
    print("binning: shipped JS vs pd.cut")
    # The registry goes through JSON exactly as the endpoint sends it.
    reg = json.loads(json.dumps(REGISTRY, allow_nan=False))
    ranges = [m for m in reg if m["type"] == "range"]
    check(set(HELPERS) == {m["key"] for m in ranges},
          f"every range metric has a pandas helper mapped ({len(ranges)})")

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
    flipped = dict(m["bins"], right=not m["bins"]["right"])
    fault = run_js({"f": {"values": job["vix3m_vix"]["values"], "bins": flipped}})["f"]
    check(fault != expect["vix3m_vix"], "a left/right-closed swap on the ratio bins IS detected")


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
    need = {"key", "label", "column", "type", "min", "max", "step", "bins", "hasScatter", "minDate"}
    for m in REGISTRY:
        miss = need - set(m)
        check(not miss, f"{m['key']}: has {sorted(need)}" + (f" — missing {sorted(miss)}" if miss else ""))
    order = [m["column"] for m in REGISTRY if m["section"]]
    brief = ["day_of_week", "year", "gap", "vix_overnight_gap", "premium", "vix_level",
             "vix3m_level", "vix9d_level", "vix3m_vix_ratio", "vix_vix9d_ratio"]
    check(order == brief, f"section order matches the brief: {order}")
    check(len({m["key"] for m in REGISTRY}) == len(REGISTRY), "keys are unique")
    for m in REGISTRY:
        if m["type"] == "categorical":
            check(not m["hasScatter"], f"{m['key']}: categorical has no scatter")
        else:
            check(m["column"] in TRADE_COLUMNS, f"{m['key']}: column {m['column']} is in the payload whitelist")
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
    check(set(cols) <= set(TRADE_COLUMNS), f"{label}: only whitelisted columns")
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
        check(c["gap"][0] is None and c["gap"][1] == -0.62, f"OO CSV ({name}): blank Gap is null, not 0")
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


def main() -> int:
    check_registry()
    check_against_source()
    check_binning()
    check_dropped_scope()
    check_parsers()
    print()
    if FAILS:
        print(f"FAIL: {len(FAILS)} check(s) failed")
        return 1
    print("PASS: OO/Mesosim Backtest registry, binning parity, payload and parsers")
    return 0


if __name__ == "__main__":
    sys.exit(main())
