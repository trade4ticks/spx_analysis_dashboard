"""Factor Trades' sector filter: the map, the sizing trap, and parity.

Offline: no database, no browser. The zone payloads come from THE REAL /zone
handler over check_ft_population's fake pool, fed designed trades (several
sectors, days where the daily cap binds, an unmapped ticker, both windows).
The page's functions (static/js/factor_sectors.js, and _computeDollarSeries
from factor_charts.js) run in node on those payloads.

What it holds:
  map     17 buckets, every ticker in exactly one, the agreed placements
  zone    the /zone refactor (_ticker_breakdown, _exit_breakdown pulled out)
          returns byte-identical payloads to HEAD's /zone, both windows
  parity  every sector on: the browser's copies of price bins, P&L
          distribution, ticker breakdown, exit reasons and the window stats
          equal what the server sent. A sector subset: they equal the
          server's own Python run on that subset
  sizing  sizes stamped once over the whole zone; per-sector dollar P&L adds
          up to the total; re-sizing the subset instead gives a DIFFERENT
          answer on this fixture (so the check can see the trap); and
          _computeDollarSeries on unstamped trades is HEAD's, exactly
  peak    peak concurrent sector exposure equals a brute-force day-by-day
          count
  check   the map-vs-daily_features diff names both directions

Runs on the dev machine (node). Exit 1 on any failure.
"""
from __future__ import annotations

import asyncio
import importlib.util
import json
import math
import random
import subprocess
import sys
import tempfile
from collections import Counter
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

FAILS: list[str] = []
N_OK = [0]


def check(ok: bool, msg: str, detail: str = "") -> None:
    print(("  ok    " if ok else "  FAIL  ") + msg + ("" if ok or not detail else f" -- {detail}"))
    if ok:
        N_OK[0] += 1
    else:
        FAILS.append(msg)


def _harness():
    spec = importlib.util.spec_from_file_location("ftpop", ROOT / "scripts" / "check_ft_population.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["ftpop"] = mod
    spec.loader.exec_module(mod)
    return mod


# ── The fixture ───────────────────────────────────────────────────────────
SESSIONS = [d for d in (date(2022, 10, 3) + timedelta(days=i) for i in range(182)) if d.weekday() < 5]
TICKERS = ["NVDA", "AMD", "JPM", "BAC", "MARA", "COIN", "UNH", "CVS", "XLE", "SPY", "ZZZZ"]
PRICES = {"NVDA": 180.0, "AMD": 70.0, "JPM": 140.0, "BAC": 33.0, "MARA": 9.0, "COIN": 60.0,
          "UNH": 520.0, "CVS": 95.0, "XLE": 88.0, "SPY": 390.0, "ZZZZ": 2500.0}


def fixture_rows(h) -> list:
    rnd = random.Random(11)
    rows = []
    for d in SESSIONS:
        # 2-10 names a day: above five the $10,000 daily cap binds, so a
        # day's N decides every size that day.
        for tk in sorted(rnd.sample(TICKERS, rnd.randint(2, 10))):
            eb = rnd.choice([rnd.uniform(60, 390), rnd.uniform(391, 390 * 9)])
            rows.append(h.Row(ticker=tk, trade_date=d, exit_bar=round(eb, 1),
                              exit_return=round(rnd.gauss(0.002, 0.04), 6),
                              exit_rule=rnd.choice(["fixed_stop__5", "max_days__20"]),
                              entry_price=PRICES[tk] * rnd.uniform(0.9, 1.1),
                              is_train=d < h.CUTOFF, sig_mask=1))
    return rows


def zones(h, ft_mod, rows) -> dict:
    """The real /zone, both windows, over the fixture rows."""
    import app.routers.oi_analysis as oia
    orig = h.respond

    def respond(sql, args):
        s = " ".join(sql.split())
        r = orig(sql, args)
        # Every trade-row query, except the random baseline's own draw.
        if r and "unnest($3::date[], $4::int[])" not in s and "exit_return" in r[0] and "ticker" in r[0]:
            return rows
        return r
    h.respond = respond
    h.DAYS = SESSIONS
    out = {}
    try:
        for win in ("train", "test"):
            oia._TT_CUTOFF_CACHED = None
            req = ft_mod.ZoneReq(primary_metric="a", secondary_metric="b", n_bins=20,
                                 rule_keys=["fixed_stop__5"], max_strike=1000,
                                 cells=[[1, 2]], window=win)
            out[win] = json.loads(json.dumps(asyncio.run(ft_mod.zone(req=req, pool=h.FakePool())),
                                             default=str))
    finally:
        h.respond = orig
    return out


# ── node ──────────────────────────────────────────────────────────────────
NODE = r"""
const fs = require('fs'), vm = require('vm');
const [inPath, chartsPath, headChartsPath, sectorsPath] = process.argv.slice(2);
const IN = JSON.parse(fs.readFileSync(inPath, 'utf8'));
global.window = global;
vm.runInThisContext(fs.readFileSync(headChartsPath, 'utf8'));
const HEAD = window.FactorCharts;
vm.runInThisContext(fs.readFileSync(chartsPath, 'utf8'));
const FC = window.FactorCharts;
vm.runInThisContext(fs.readFileSync(sectorsPath, 'utf8'));
const cmp = {};
const idx = ftSectorIndex(IN.sectors);
const out = {};
for (const win of ['train', 'test']) {
  const z = IN.zones[win];
  const all = z.window_trades;
  const o = out[win] = {};
  o.all = { stats: ftWindowStats(all), price_bins: ftPriceBins(all, z.price_bins, IN.edges),
            pnl_dist: ftPnlDist(all, z.pnl_dist), tickers: ftTickerBreakdown(all),
            exit_reasons: ftExitBreakdown(all, z.exit_reasons) };
  const sized = { window: ftStampSizes(FC, cmp, all, 2000, 10000),
                  series: ftStampSizes(FC, cmp, z.series_trades, 2000, 10000) };
  const keep = new Set(IN.keep);
  const v = ftSectorView(z, keep, idx, sized, IN.edges, win);
  o.sub = { n: v.window_trades.length, tickers_in: [...new Set(v.window_trades.map(t => t.ticker))].sort(),
            stats: v[win], price_bins: v.price_bins, pnl_dist: v.pnl_dist, tickers: v.tickers,
            exit_reasons: v.exit_reasons, series_n: v.series_trades.length };
  // Sizing: per-sector P&L sums to the total.
  const tot = FC._computeDollarSeries(cmp, sized.series, 2000, 10000).equity;
  let sumEnds = 0;
  for (const name of [...idx.order, idx.unassigned]) {
    const eq = FC._computeDollarSeries(cmp, sized.series.filter(t => ftSectorOf(idx, t.ticker) === name), 2000, 10000).equity;
    if (eq.length) sumEnds += eq[eq.length - 1].value;
  }
  o.total_end = tot.length ? tot[tot.length - 1].value : 0;
  o.sector_sum = sumEnds;
  // The trap, planted: size the subset on its own.
  const subRaw = z.series_trades.filter(t => keep.has(ftSectorOf(idx, t.ticker)));
  const resized = FC._computeDollarSeries(cmp, subRaw, 2000, 10000).equity;
  const kept = FC._computeDollarSeries(cmp, v.series_trades, 2000, 10000).equity;
  o.sub_end_kept = kept.length ? kept[kept.length - 1].value : 0;
  o.sub_end_resized = resized.length ? resized[resized.length - 1].value : 0;
  // Stamped sizes are the ones the whole-list pass produced.
  const whole = FC._computeDollarSeries(cmp, all, 2000, 10000).tradeDollarSizes;
  o.stamped = sized.window.filter(t => t.usd_size != null).map(t => t.usd_size);
  o.whole = whole.map(x => x.dollar_size);
  // Unstamped trades: the new function is HEAD's, field for field.
  const a = FC._computeDollarSeries(cmp, z.series_trades, 2000, 10000);
  const b = HEAD._computeDollarSeries(cmp, z.series_trades, 2000, 10000);
  o.head_same = JSON.stringify([a.equity, [...a.dayPnlByDate], [...a.dayDeployedByDate], a.tradeDollarSizes])
             === JSON.stringify([b.equity, [...b.dayPnlByDate], [...b.dayDeployedByDate], b.tradeDollarSizes]);
  o.peak = ftPeakSectorExposure(sized.window, z.trading_days, idx);
  o.sized_window = sized.window.map(t => [t.ticker, t.trade_date, t.usd_size ?? null, t.exit_bar]);
}
out.map_check = ftMapCheck(idx, IN.live);
process.stdout.write(JSON.stringify(out));
"""


def run_node(payload: dict) -> dict:
    head = subprocess.run(["git", "show", "HEAD:static/js/factor_charts.js"], cwd=ROOT,
                          capture_output=True, text=True, encoding="utf-8", check=True).stdout
    tmp = Path(tempfile.mkdtemp())
    (tmp / "in.json").write_text(json.dumps(payload), encoding="utf-8")
    (tmp / "head_charts.js").write_text(head, encoding="utf-8")
    (tmp / "run.js").write_text(NODE, encoding="utf-8")
    p = subprocess.run(["node", str(tmp / "run.js"), str(tmp / "in.json"),
                        str(ROOT / "static/js/factor_charts.js"), str(tmp / "head_charts.js"),
                        str(ROOT / "static/js/factor_sectors.js")],
                       capture_output=True, text=True, encoding="utf-8", timeout=120)
    if p.returncode:
        raise RuntimeError("node failed: " + p.stderr[-1500:])
    return json.loads(p.stdout)


def close(a, b, tol=1e-9) -> bool:
    if a is None or b is None:
        return a is None and b is None
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return math.isclose(a, b, rel_tol=tol, abs_tol=tol)
    return a == b


def rows_match(js: list, py: list, keys: dict) -> tuple[bool, str]:
    """keys: field -> tolerance (None = exact)."""
    if len(js) != len(py):
        return False, f"{len(js)} rows vs {len(py)}"
    for i, (a, b) in enumerate(zip(js, py)):
        for k, tol in keys.items():
            ok = (a.get(k) == b.get(k)) if tol is None else close(a.get(k), b.get(k), tol)
            if not ok:
                return False, f"row {i} {k}: {a.get(k)!r} vs {b.get(k)!r}"
    return True, ""


def main() -> int:
    from app import factor_sectors as fs
    import app.routers.factor_trades as ft

    # ── map ───────────────────────────────────────────────────────────────
    names = [n for n, _ in fs.BUCKETS]
    allt = [t for _, ts in fs.BUCKETS for t in ts]
    dup = [t for t, c in Counter(allt).items() if c > 1]
    check(len(fs.BUCKETS) == 14, f"14 buckets ({len(fs.BUCKETS)})")
    gics = ["Communication Services", "Consumer Discretionary", "Consumer Staples", "Energy",
            "Financials", "Health Care", "Industrials", "Information Technology", "Materials",
            "Real Estate", "Utilities"]
    check(names == gics + ["Crypto", "Broad index", "Bonds"],
          "the 11 GICS sectors in GICS order, then Crypto, Broad index, Bonds", " | ".join(names))
    check(not dup, "every ticker is in exactly one bucket", ", ".join(dup))
    check(len(allt) == 121, f"121 tickers ({len(allt)})")
    dead = [t for t in ("ALL", "AMC", "APLD", "CORZ", "ETHA", "GME", "IBIT", "QBTS") if t in allt]
    check(not dead, "the eight tickers not in daily_features are gone", " ".join(dead))
    check(fs.UNASSIGNED not in names, "no bucket is called Unassigned")
    m = fs.sector_map()
    agreed = {**{t: "Crypto" for t in ("MSTR", "MARA", "RIOT", "CIFR", "COIN", "HOOD")},
              **{t: "Broad index" for t in ("SPY", "QQQ", "IWM", "DIA", "ARKK")},
              "TLT": "Bonds", "IEF": "Bonds",
              "TSLA": "Consumer Discretionary", "NVDA": "Information Technology",
              "XLK": "Information Technology", "XLF": "Financials", "KRE": "Financials",
              "XLE": "Energy", "XOP": "Energy", "XLV": "Health Care", "XBI": "Health Care",
              "XLP": "Consumer Staples", "XLI": "Industrials", "XLB": "Materials",
              "XLU": "Utilities", "V": "Financials", "MA": "Financials", "PYPL": "Financials",
              "WMT": "Consumer Staples", "UBER": "Industrials", "GOOGL": "Communication Services"}
    bad = [f"{t}->{m.get(t)}" for t, b in agreed.items() if m.get(t) != b]
    check(not bad, "the user's buckets, sector ETFs in their own sector, GICS placements", " ".join(bad))
    check([b["name"] for b in fs.payload()["buckets"]] == names, "/sectors returns the buckets in chip order")

    # ── zone, before and after the refactor ──────────────────────────────
    h = _harness()
    rows = fixture_rows(h)
    new = zones(h, ft, rows)
    old_mod = h.load_old("HEAD")
    old = zones(h, old_mod, rows)
    for win in ("train", "test"):
        check(json.dumps(new[win], sort_keys=True) == json.dumps(old[win], sort_keys=True),
              f"/zone ({win}) is byte-identical to HEAD's after pulling out the two breakdowns")
    for win in ("train", "test"):
        z = new[win]
        check(len(z["window_trades"]) > 100 and len({t["ticker"] for t in z["window_trades"]}) == len(TICKERS),
              f"the {win} fixture has every ticker and >100 trades ({len(z['window_trades'])})")

    keep = ["Information Technology", "Crypto", "Unassigned"]
    edges = [[lo, hi if hi != float("inf") else 1e308] for lo, hi in ft.PRICE_BUCKETS]
    live = ["NVDA", "AMD", "JPM", "ZZZZ", "SPY"]
    out = run_node({"zones": new, "sectors": fs.payload(), "edges": edges, "keep": keep, "live": live})

    smap = fs.sector_map()
    sec = lambda t: smap.get(t, fs.UNASSIGNED)
    stat_keys = {k: 1e-9 for k in ("n", "n_tickers", "eff_tickers", "avg_ret", "median", "std_dev",
                                   "p5", "p95", "win_rate", "n_win", "avg_win", "avg_loss",
                                   "trades_per_year", "total_ret", "avg_annual", "calmar",
                                   "max_dd", "avg_hold")}
    for win in ("train", "test"):
        z, o = new[win], out[win]
        # ── parity, every sector on ────────────────────────────────────────
        js, py = o["all"]["stats"], z[win]
        bad = [k for k in stat_keys if not close(js.get(k), py.get(k), 1e-9)]
        check(not bad, f"{win}: window stats equal the server's, every sector on", ", ".join(f"{k} {js.get(k)} vs {py.get(k)}" for k in bad))
        ok, why = rows_match(o["all"]["price_bins"], z["price_bins"], {"label": None, "n": None, "avg_ret": 1e-9})
        check(ok, f"{win}: price bins equal the server's", why)
        ok, why = rows_match(o["all"]["pnl_dist"], z["pnl_dist"], {"label": None, "n": None, "lo": None})
        check(ok, f"{win}: P&L distribution equals the server's", why)
        # The server rounds for the wire (avg 6 dp, win 4, contrib 2).
        ok, why = rows_match(o["all"]["tickers"], z["tickers"],
                             {"ticker": None, "n": None, "avg_ret": 5e-7, "win_rate": 5e-5, "contrib_pct": 5e-3})
        check(ok, f"{win}: ticker breakdown equals the server's (to its rounding)", why)
        ok, why = rows_match(o["all"]["exit_reasons"], z["exit_reasons"],
                             {"rule_key": None, "label": None, "side": None, "is_backstop": None,
                              "n": None, "frac": 1e-12, "avg_hold": 1e-9})
        check(ok, f"{win}: exit reasons equal the server's", why)

        # ── parity, a subset, against the server's own Python ─────────────
        sub = [t for t in z["window_trades"] if sec(t["ticker"]) in keep]
        check(o["sub"]["n"] == len(sub) and o["sub"]["tickers_in"] == sorted({t["ticker"] for t in sub}),
              f"{win}: the subset is exactly the kept sectors' trades ({len(sub)})")
        span = ((date.fromisoformat(max(t["trade_date"] for t in sub))
                 - date.fromisoformat(min(t["trade_date"] for t in sub))).days if len(sub) > 1 else 1)
        py = ft._window_stats([t["ret"] for t in sub], [t["exit_bar"] for t in sub],
                              Counter(t["ticker"] for t in sub), span)
        js = o["sub"]["stats"]
        bad = [k for k in stat_keys if not close(js.get(k), py.get(k), 1e-9)]
        check(not bad, f"{win}: subset stats equal _window_stats on the subset", ", ".join(f"{k} {js.get(k)} vs {py.get(k)}" for k in bad))
        ok, why = rows_match(o["sub"]["price_bins"], ft._price_bins(sub), {"label": None, "n": None, "avg_ret": 1e-9})
        check(ok, f"{win}: subset price bins equal _price_bins on the subset", why)
        ok, why = rows_match(o["sub"]["pnl_dist"], ft._pnl_dist(sub), {"label": None, "n": None})
        check(ok, f"{win}: subset P&L distribution equals _pnl_dist on the subset", why)
        by = {}
        for t in sub:
            by.setdefault(t["ticker"], []).append(t["ret"])
        ok, why = rows_match(o["sub"]["tickers"], ft._ticker_breakdown(by),
                             {"ticker": None, "n": None, "avg_ret": 5e-7, "win_rate": 5e-5, "contrib_pct": 5e-3})
        check(ok, f"{win}: subset ticker breakdown equals _ticker_breakdown on the subset", why)
        reasons, holds = {}, {}
        for t in sub:
            reasons[t["exit_rule"]] = reasons.get(t["exit_rule"], 0) + 1
            holds.setdefault(t["exit_rule"], []).append(t["exit_bar"])
        meta = {r["rule_key"]: r for r in h.RULES}
        ok, why = rows_match(o["sub"]["exit_reasons"],
                             ft._exit_breakdown(reasons, holds, meta, z["horizon_auto_added"], len(sub) or 1),
                             {"rule_key": None, "label": None, "n": None, "frac": 1e-12, "avg_hold": 1e-9})
        check(ok, f"{win}: subset exit reasons equal _exit_breakdown on the subset", why)

        # ── sizing ────────────────────────────────────────────────────────
        check(o["stamped"] == o["whole"], f"{win}: stamped sizes are the whole-list sizing pass's")
        check(abs(o["sector_sum"] - o["total_end"]) < 0.01,
              f"{win}: per-sector equity ends add up to the total ({o['sector_sum']:.2f} vs {o['total_end']:.2f})")
        check(abs(o["sub_end_kept"] - o["sub_end_resized"]) > 1.0,
              f"{win}: re-sizing the subset would change its P&L on this fixture "
              f"({o['sub_end_kept']:.2f} kept vs {o['sub_end_resized']:.2f} re-sized), so the sum check can see the trap")
        check(o["head_same"], f"{win}: _computeDollarSeries on unstamped trades is HEAD's, exactly")

        # ── peak exposure, brute force ────────────────────────────────────
        days = z["trading_days"]
        at = {d: i for i, d in enumerate(days)}
        best = (0.0, None, None)
        for k, day in enumerate(days):
            per = {}
            for tk, d, size, eb in o["sized_window"]:
                if size is None or d not in at:
                    continue
                i = at[d]
                nights = max(0, math.ceil(eb / 390) - 1)
                if i <= k <= i + nights:
                    per[sec(tk)] = per.get(sec(tk), 0.0) + size
            for s_, v in per.items():
                if v > best[0] + 1e-9:
                    best = (v, s_, day)
        p = o["peak"]
        check(abs(p["usd"] - best[0]) < 1e-6 and p["sector"] == best[1] and p["date"] == best[2],
              f"{win}: peak sector exposure {p['usd']:.2f} {p['sector']} {p['date']} equals the day-by-day count "
              f"({best[0]:.2f} {best[1]} {best[2]})")

    mc = out["map_check"]
    check("ZZZZ" in mc["unmapped"] and "SPY" not in mc["unmapped"] and "BAC" in mc["notInData"]
          and "NVDA" not in mc["notInData"], "the map check names both directions (unmapped ZZZZ, BAC not in data)")

    print(f"check_ft_sectors: {N_OK[0]} ok, {len(FAILS)} failed")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
