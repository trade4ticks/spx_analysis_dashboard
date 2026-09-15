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
    import shutil
    if shutil.which("node") is None:
        print("  SKIP  node is not installed — the shipped JS cannot be executed on this host")
        NOT_RUN.append("JS binning parity (node not installed)")
        return
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
            "series", "basis"}
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
eval(fs.readFileSync(process.argv[1], 'utf8'));
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
    p = subprocess.run(["node", "-e", STATS_DRIVER, str(JS)], input=json.dumps(job),
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
eval(fs.readFileSync(process.argv[1], 'utf8'));
const job = JSON.parse(fs.readFileSync(0, 'utf8'));
const out = {};
for (const [key, t] of Object.entries(job)) {
  const d = obSectionData(t.cols, t.idx, t.metric, t.column);
  out[key] = { rows: d.rows, valued: d.valued, fit: obOLS(d.xs, d.ys),
               swapped: obOLS(d.ys, d.xs), raw: t.raw ? obOLS(t.raw.xs, t.raw.ys) : null,
               alpha: { max: obBarAlpha(400, 400), one_of_400: obBarAlpha(1, 400),
                        quarter: obBarAlpha(100, 400), zero: obBarAlpha(0, 400) } };
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
    sections = [m for m in reg if m["section"]]
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

    p = subprocess.run(["node", "-e", SECTION_DRIVER, str(JS)], input=json.dumps(job),
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
    check(alpha["max"] == 1 and alpha["one_of_400"] > 0.25 and abs(alpha["quarter"] - 0.625) < 1e-9
          and alpha["zero"] == 0,
          f"bar opacity: sqrt of count/max, floored at 0.25, empty bin 0 ({alpha})")
    check(got["two"]["fit"] is None and got["deg"]["raw"] is None,
          "no OLS line from fewer than 3 points, or when every x is identical")



DEPLOY_DRIVER = r"""
const fs = require('fs');
global.document = { addEventListener: () => {} };
eval(fs.readFileSync(process.argv[1], 'utf8'));
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
        lo = min(cols["date_opened"][i] for i in idx)
        hi = max(cols["date_closed"][i] for i in idx)
        days = [d for d in sessions if lo <= d <= hi]
        return days, [sum(1 for i in idx if cols["date_opened"][i] <= d <= cols["date_closed"][i]) for d in days]

    def extra(idx, capital, peak):
        df = pd.DataFrame({k: [cols[k][i] for i in idx] for k in cols})
        df["date_opened"] = pd.to_datetime(df["date_opened"])
        df["date_closed"] = pd.to_datetime(df["date_closed"])
        st = pystats.calculate_stats(df.copy())
        years = (df["date_closed"].max() - df["date_opened"].min()).days / 365.25
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
           "meso": {"cols": mc, "sessions": m_sessions, "capital": 10000},
           "meso_planted": {"cols": planted, "sessions": m_sessions, "capital": 10000}}
    p = subprocess.run(["node", "-e", DEPLOY_DRIVER, str(JS)], input=json.dumps(job),
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
eval(fs.readFileSync(process.argv[1], 'utf8'));
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
    p = subprocess.run(["node", "-e", COMPONENT_DRIVER, str(JS)], input=json.dumps({"registry": reg, "payload": payload}),
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
    check_component_filters()
    check_deployment_and_extra_stats()
    check_dropped_scope()
    check_parsers()
    check_real_mesosim()
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
