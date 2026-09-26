"""Gate: Factor Trades' population reaches EVERY query, and nothing else moved.

Offline. Drives the real endpoint handlers -- /run, /zone (policy and all
three random baselines), /suite, /grid -- against a RECORDING fake pool that
answers each statement with plausible rows, and asserts two things:

  1. UNCHANGED WHEN UNUSED. With no metric filters, every statement the new
     code issues is byte-identical, args included, to what the previous
     commit's code issued -- with and without max_strike. The Population
     builder replaced max_strike's eleven hand-numbered copies, so this is
     the check that the refactor itself changed nothing.

  2. EVERYWHERE WHEN USED. With filters, every statement that selects trades
     (any that joins trade_paths) carries the daily_features join and every
     filter's predicate, and each predicate's placeholder points at the
     argument holding its own threshold. A filter that reached the heatmap
     but not a baseline sampler or the grid is the failure this exists for.

Plus the guard: forward returns, unknown metrics and bad operators are
refused, and the percentile endpoint sees train rows only.

The previous code is read from git (HEAD, or --ref), so this needs git.
"""
from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import re
import subprocess
import sys
import tempfile
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

FAILS: list[str] = []
CUTOFF = date(2023, 1, 2)


def check(ok: bool, msg: str) -> None:
    print(("  ok    " if ok else "  FAIL  ") + msg)
    if not ok:
        FAILS.append(msg)


RULES = [
    {"rule_key": "fixed_stop__5", "family": "fixed_stop", "side": "stop", "fill_mode": "close",
     "params": {"pct": 5}, "exit_bar_col": "fs5_bar", "exit_return_col": "fs5_ret", "is_horizon": False},
    {"rule_key": "max_days__5", "family": "max_days", "side": "time", "fill_mode": "close",
     "params": {"n": 5}, "exit_bar_col": "md5_bar", "exit_return_col": "md5_ret", "is_horizon": False},
    {"rule_key": "max_days__20", "family": "max_days", "side": "time", "fill_mode": "close",
     "params": {"n": 20}, "exit_bar_col": "md20_bar", "exit_return_col": "md20_ret", "is_horizon": True},
]
DAYS = [date(2022, 12, 28) + timedelta(days=i) for i in range(8)]


class Row(dict):
    """A Record stand-in. Any *_bar / *_ret column the grid asks for exists."""
    def __missing__(self, k):
        if k.endswith("_bar"):
            return 30.0
        if k.endswith("_ret"):
            return 0.01
        raise KeyError(k)


def trade(d, i=0):
    return Row(ticker=f"T{i}", trade_date=d, exit_bar=30.0, exit_return=0.01,
               exit_rule="max_days__20", entry_price=50.0 + i, is_train=d < CUTOFF, sig_mask=1)


def respond(sql: str, args: tuple) -> list:
    s = " ".join(sql.split())
    if "FROM signals WHERE id = ANY" in s:
        return [Row(id=i, name=f"sig{i}", primary_metric="a", secondary_metric="b",
                    outcome="ret_5d_fwd_oc", n_bins=20, cell_set=json.dumps([[i, i + 1]]),
                    agg_n=None, agg_avg_ret=None, per_cell_stats=None, stats_updated_at=None,
                    status="Test", color_slot=None, corner=None, selection_mode="train_test",
                    selection_cutoff=CUTOFF) for i in args[0]]
    if "FROM trade_path_rules" in s:
        return [Row(r) for r in RULES]
    if "MAX(cutoff_date)" in s:
        return [Row(cutoff=CUTOFF)]
    if "information_schema.columns" in s and "tt_bins" in s:
        return [Row(column_name=c) for c in ("bin20_a", "bin20_b")]
    if "information_schema.columns" in s and "daily_features" in s:
        return [Row(column_name=c) for c in ("a", "b", "ret_5d", "rv_20d", "ret_5d_fwd_oc")]
    if "FROM metric_classification" in s:
        return [Row(metric=m) for m in ("a", "b", "ret_5d", "rv_20d")]
    if "AS n_before" in s:
        k = len(re.findall(r"AS nv\d+", s))
        r = Row(is_train=True, n_before=10, n_after=6)
        for i in range(k):
            r[f"nv{i}"], r[f"fail{i}"] = 1, 2
        return [r]
    if "percentile_cont" in s:
        return [Row(n=100, n_value=90, q=[0.1, 0.2, 0.3, 0.4, 0.5])]
    if "GROUP BY bp" in s:
        return [Row(bp=1, bs=2, exit_rule="max_days__20", is_train=True, n=3, avg_ret=0.01, avg_hold=30.0)]
    if "AS sess" in s:
        return [Row(sess=5, n=3)]
    if "COUNT(*) AS k" in s and "GROUP BY c.trade_date" in s:
        return [Row(trade_date=d, k=2) for d in DAYS]
    if "COUNT(*) AS k" in s:
        return [Row(k=len(DAYS) * 2)]
    if "unnest($3::date[], $4::int[])" in s:
        return [trade(d, i) for d, k in zip(args[2], args[3]) for i in range(k)]
    if "SELECT DISTINCT trade_date FROM tt_bins" in s:
        return [Row(trade_date=d) for d in DAYS]
    if "FROM c" in s or "FROM tt_bins bt JOIN trade_paths" in s:
        return [trade(d, i) for d in DAYS for i in range(2)]
    return []


class FakeConn:
    def __init__(self, log):
        self.log = log

    async def fetch(self, sql, *args):
        self.log.append((sql, args))
        return respond(sql, args)

    async def fetchrow(self, sql, *args):
        rows = await self.fetch(sql, *args)
        return rows[0] if rows else None

    async def fetchval(self, sql, *args):
        r = await self.fetchrow(sql, *args)
        return next(iter(r.values())) if r else None


class _Acq:
    def __init__(self, conn):
        self.conn = conn

    async def __aenter__(self):
        return self.conn

    async def __aexit__(self, *exc):
        return False


class FakePool:
    def __init__(self):
        self.log: list = []

    def acquire(self):
        return _Acq(FakeConn(self.log))


def load_old(ref: str):
    src = subprocess.run(["git", "show", f"{ref}:app/routers/factor_trades.py"], cwd=ROOT,
                         capture_output=True, text=True, encoding="utf-8", check=True).stdout
    path = Path(tempfile.mkdtemp()) / "factor_trades_old.py"
    path.write_text(src, encoding="utf-8")
    spec = importlib.util.spec_from_file_location("factor_trades_old", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["factor_trades_old"] = mod      # pydantic resolves annotations through it
    spec.loader.exec_module(mod)
    return mod


BASE = dict(primary_metric="a", secondary_metric="b", rule_keys=["fixed_stop__5"],
            n_bins=20, cells=[[1, 2], [3, 4]])


def calls(mod, extra: dict):
    """(label, coroutine factory) for every population-bearing endpoint path."""
    b = {**BASE, **extra}
    run_b = {k: v for k, v in b.items() if k != "cells"}
    # PORTFOLIO MODE: saved signals instead of a metric pair and cells. /run
    # refuses it (no single heatmap), so /zone carries the provenance there.
    pf = {**extra, "signal_ids": [1, 2], "rule_keys": ["fixed_stop__5"], "n_bins": 20}
    return [
        ("portfolio zone", lambda p: mod.zone(req=mod.ZoneReq(**pf), pool=p)),
        ("portfolio zone entry", lambda p: mod.zone(req=mod.ZoneReq(**pf, randomize=True, baseline_kind="entry"), pool=p)),
        ("portfolio suite", lambda p: mod.suite(req=mod.SuiteReq(**pf, n_draws=2, concurrency=1), pool=p)),
        ("portfolio grid", lambda p: mod.grid(req=mod.GridReq(**pf, sweep_families=["max_days"]), pool=p)),
        ("run", lambda p: mod.run(req=mod.RunReq(**run_b), pool=p)),
        ("zone", lambda p: mod.zone(req=mod.ZoneReq(**b), pool=p)),
        ("zone entry", lambda p: mod.zone(req=mod.ZoneReq(**b, randomize=True, baseline_kind="entry"), pool=p)),
        ("zone exit", lambda p: mod.zone(req=mod.ZoneReq(**b, randomize=True, baseline_kind="exit"), pool=p)),
        ("zone entry_exit", lambda p: mod.zone(req=mod.ZoneReq(**b, randomize=True, baseline_kind="entry_exit"), pool=p)),
        ("suite", lambda p: mod.suite(req=mod.SuiteReq(**b, n_draws=2, concurrency=1), pool=p)),
        ("grid", lambda p: mod.grid(req=mod.GridReq(**b, sweep_families=["max_days"]), pool=p)),
        # verify= re-runs combinations through build_combine_sql and diffs them
        # against numpy. The fake's rows are not a consistent database, so that
        # diff fails by design here; this run exists for its STATEMENTS.
        ("grid verify", lambda p: mod.grid(req=mod.GridReq(**b, sweep_families=["max_days"], verify=2), pool=p)),
    ]


def drive(mod, extra):
    out = {}
    import app.routers.oi_analysis as oia
    for label, make in calls(mod, extra):
        oia._TT_CUTOFF_CACHED = None        # cached across calls; each run starts cold
        pool = FakePool()
        resp = asyncio.run(make(pool))
        out[label] = (resp, pool.log)
    return out


def ran(label, resp) -> tuple[bool, str]:
    err = resp.get("error") if isinstance(resp, dict) else None
    if label == "grid verify":
        return bool(err and err.startswith("VERIFY FAILED")), "reached its verify step"
    return not err, err or ""


def population_statements(log):
    return [(sql, args) for sql, args in log if "trade_paths" in sql and "information_schema" not in sql]


def check_unchanged(old, new):
    print("unchanged when unused")
    for strike in (None, 250.0):
        o = drive(old, {"max_strike": strike})
        n = drive(new, {"max_strike": strike})
        for label in o:
            ro, lo = o[label]
            rn, ln = n[label]
            ok, why = ran(label, rn)
            check(ok, f"{label} (max_strike={strike}) ran {why}")
            same = [(" ".join(a.split()), list(map(repr, x))) for a, x in lo] == \
                   [(" ".join(a.split()), list(map(repr, x))) for a, x in ln]
            if not same:
                for i, ((a, x), (b, y)) in enumerate(zip(lo, ln)):
                    if " ".join(a.split()) != " ".join(b.split()) or list(map(repr, x)) != list(map(repr, y)):
                        print("      first difference at statement", i)
                        print("      old:", " ".join(a.split())[:300], x)
                        print("      new:", " ".join(b.split())[:300], y)
                        break
            check(same, f"{label} (max_strike={strike}): {len(ln)} statements identical to the previous code")
            check(len(population_statements(ln)) > 0, f"{label}: issues population statements at all")


def check_everywhere(new):
    print("everywhere when used")
    filters = [{"metric": "ret_5d", "op": ">", "value": 0.0512},
               {"metric": "rv_20d", "op": "<", "value": 0.4321}]
    res = drive(new, {"max_strike": 250.0, "filters": filters})
    for label, (resp, log) in res.items():
        ok, why = ran(label, resp)
        check(ok, f"{label} with filters ran {why}")
        stmts = population_statements(log)
        bad = []
        for sql, args in stmts:
            flat = " ".join(sql.split())
            if "LEFT JOIN LATERAL" not in flat or "FROM daily_features d" not in flat:
                bad.append(("no daily_features join", flat[:160]))
                continue
            for f in filters:
                m = re.search(rf'df\."{f["metric"]}"::float8 {re.escape(f["op"])} \$(\d+)', flat)
                if not m:
                    bad.append((f"no {f['metric']} predicate", flat[:160]))
                elif not (int(m.group(1)) <= len(args) and args[int(m.group(1)) - 1] == f["value"]):
                    bad.append((f"{f['metric']} placeholder ${m.group(1)} is not its threshold", flat[:160]))
            m = re.search(r"tp\.entry_price <= \$(\d+)", flat)
            if not m or args[int(m.group(1)) - 1] != 250.0:
                bad.append(("max_strike placeholder wrong", flat[:160]))
            top = max([int(x) for x in re.findall(r"\$(\d+)", flat)] or [0])
            if top > len(args):
                bad.append((f"${top} with {len(args)} args", flat[:160]))
        for why, what in bad[:3]:
            print(f"      {why}: {what}")
        check(stmts and not bad, f"{label}: all {len(stmts)} trade-selecting statements carry both filters and max_strike")
        if isinstance(resp, dict) and "filters" in resp:
            check([f["metric"] for f in resp["filters"]] == ["ret_5d", "rv_20d"], f"{label}: echoes its filters")
    for label in ("run", "zone", "portfolio zone"):
        rep = res[label][0].get("filter_report")
        check(bool(rep) and rep["train"]["filters"][0]["no_value"] == 1 and rep["train"]["filters"][0]["failed"] == 2,
              f"{label}: reports no-value and failed separately")
    for label in ("zone entry", "zone exit", "zone entry_exit"):
        check(res[label][0].get("filter_report") is None, f"{label}: a baseline carries no filter report of its own")


def check_guard(new):
    print("guard")
    for f, why in (({"metric": "ret_5d_fwd_oc", "op": ">", "value": 0}, "a forward return"),
                   ({"metric": "not_a_column", "op": ">", "value": 0}, "an unknown metric"),
                   ({"metric": "ret_5d", "op": ">=", "value": 0}, "an operator other than < >"),
                   ({"metric": 'ret_5d"; DROP TABLE x; --', "op": ">", "value": 0}, "an injected name")):
        pool = FakePool()
        resp = asyncio.run(new.run(req=new.RunReq(**{k: v for k, v in BASE.items() if k != "cells"},
                                                  filters=[f]), pool=pool))
        refused = isinstance(resp, dict) and resp.get("error")
        touched = any("trade_paths" in s for s, _ in pool.log)
        check(bool(refused) and not touched, f"refuses {why} before any trade query")
    pool = FakePool()
    r = asyncio.run(new.filter_stats(metric="ret_5d", entry_anchor="open", pool=pool))
    sql = " ".join(next(s for s, _ in pool.log if "percentile_cont" in s).split())
    check("tp.trade_date < $2::date" in sql and r["pcts"][2]["v"] == 0.3 and abs(r["no_value_share"] - 0.1) < 1e-9,
          "filter-stats: train window only, median and no-value share returned")
    r = asyncio.run(new.filter_stats(metric="ret_5d_fwd_oc", entry_anchor="open", pool=FakePool()))
    check("error" in r, "filter-stats refuses a forward return too")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default="HEAD")
    a = ap.parse_args()
    import app.routers.factor_trades as new
    try:
        old = load_old(a.ref)
    except Exception as exc:                                  # noqa: BLE001
        print(f"FAIL: could not load the previous code from git {a.ref}: {exc}")
        return 1
    if "Population" in vars(old):
        print(f"  note  {a.ref} already has Population; the unchanged-when-unused "
              f"check compares the builder with itself")
    check_unchanged(old, new)
    check_everywhere(new)
    check_guard(new)
    print()
    if FAILS:
        print(f"FAIL: {len(FAILS)} factor-trades population check(s) failed")
        return 1
    print("PASS: factor-trades population — unchanged when unused, in every query when used, guarded")
    return 0


if __name__ == "__main__":
    sys.exit(main())
