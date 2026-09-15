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


def main() -> int:
    check_registry()
    check_against_source()
    check_binning()
    check_stats_parity()
    check_component_filters()
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
