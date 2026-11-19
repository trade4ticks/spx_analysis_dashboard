"""Gate: the Heatmap page's IV z-score view (3mZ / 1yZ), offline.

`GET /api/heatmap/iv_zscore` and the page code that draws it arrived in an
uncommitted change dated 2026-10-25 that nobody remembers writing; the user
kept it on condition that it is gated like the rest (2026-11-19). Two halves:

  handler  the router's own function, driven against a RECORDING fake pool:
           an unknown lookback, a malformed date or time is a 400; the
           window is exactly 90 / 365 calendar days ENDING ON (and
           including) the viewed date; the viewed time and both thresholds
           (ZSCORE_MIN_DAYS 20, ZSCORE_MIN_STDEV 1e-6) reach the SQL as
           parameters; the response shape (v = z, n of a node with no
           history = 0, window.days = the largest n).
  page     static/js/heatmap.js in node: the colour is a FIXED +/-3 scale
           (clamped, 0 at the dark midpoint, and a function of z alone --
           the same z is the same colour whatever the other cells hold);
           a blank z names its reason (no IV / n below 20 / deviation ~0);
           the z views exist for IV only; responses are cached per
           (date, time, lookback) and land under the snapshot they were
           asked for.

The SQL itself -- does it compute those numbers -- is check_heatmap_sql
(a temporary Postgres; --live, read-only, on the VPS).
WHERE IT RUNS: the development machine (needs node for the page half).
"""
from __future__ import annotations

import asyncio
import json
import shutil
import subprocess
import sys
from datetime import date, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
JS = ROOT / "static" / "js" / "heatmap.js"

EXIT_SKIPPED = 3
FAILS: list[str] = []
OKS = 0


def check(ok: bool, msg: str) -> None:
    global OKS
    print(("  ok    " if ok else "  FAIL  ") + msg)
    if ok:
        OKS += 1
    else:
        FAILS.append(msg)


# ── the handler, against a recording fake pool ──────────────────────────────

class FakeConn:
    def __init__(self, rows):
        self.rows, self.calls = rows, []

    async def fetch(self, sql, *args):
        self.calls.append((sql, args))
        return self.rows


class FakePool:
    def __init__(self, rows):
        self.conn = FakeConn(rows)

    def acquire(self):
        pool = self

        class _Ctx:
            async def __aenter__(self):
                return pool.conn

            async def __aexit__(self, *a):
                return False
        return _Ctx()


def check_handler() -> None:
    print("handler: /api/heatmap/iv_zscore")
    from fastapi import HTTPException

    from app.routers import heatmap as hm

    def call(rows=(), **kw):
        args = {"date": "2026-03-31", "time": "15:45", "lookback": "90d"}
        args.update(kw)
        pool = FakePool(list(rows))
        try:
            return asyncio.run(hm.heatmap_iv_zscore(pool=pool, **args)), pool, None
        except HTTPException as e:
            return None, pool, e

    for kw, what in [({"lookback": "30d"}, "an unknown lookback"),
                     ({"lookback": ""}, "an empty lookback"),
                     ({"date": "2026-02-30"}, "an impossible date"),
                     ({"date": "yesterday"}, "a word for a date"),
                     ({"time": "25:00"}, "an impossible time")]:
        _, pool, err = call(**kw)
        check(err is not None and err.status_code == 400 and not pool.conn.calls,
              f"{what} is a 400 and never reaches the database ({err.status_code if err else 'no error'})")

    check(hm.ZSCORE_LOOKBACK_DAYS == {"90d": 90, "1y": 365} and hm.ZSCORE_MIN_DAYS == 20
          and hm.ZSCORE_MIN_STDEV == 1e-6,
          f"the windows are 90 / 365 calendar days, blank below 20 days or a deviation <= 1e-6 "
          f"({hm.ZSCORE_LOOKBACK_DAYS}, {hm.ZSCORE_MIN_DAYS}, {hm.ZSCORE_MIN_STDEV})")

    for lb, first in [("90d", date(2026, 1, 1)), ("1y", date(2025, 4, 1))]:
        _, pool, _ = call(lookback=lb)
        sql, args = pool.conn.calls[0]
        check(args[:3] == (first, date(2026, 3, 31), time(15, 45)),
              f"{lb}: the window is {first}..2026-03-31 inclusive -- ending ON the viewed date -- and the "
              f"viewed time is a parameter ({args[:3]})")
        check(args[3:] == (20, 1e-6), f"{lb}: both thresholds reach the SQL as parameters ({args[3:]})")
    check("quote_time = '15:45'" in sql and "stddev_samp" in sql and "s.quote_time = $3::time" in sql,
          "history is the 15:45 snapshot only; the viewed snapshot is the requested time")
    _, pool, _ = call(time="10:05")
    check(pool.conn.calls[0][1][2] == time(10, 5), "a mid-session view asks for that snapshot")

    rows = [
        {"dte": 7, "put_delta": 25, "iv": 0.21, "n": 61, "mean": 0.18, "sd": 0.02, "z": 1.5},
        {"dte": 7, "put_delta": 50, "iv": 0.20, "n": 19, "mean": 0.19, "sd": 0.01, "z": None},
        {"dte": 30, "put_delta": 25, "iv": 0.22, "n": None, "mean": None, "sd": None, "z": None},
    ]
    body, _, _ = call(rows)
    cur = {(c["dte"], c["put_delta"]): c for c in body["current"]}
    check(cur[(7, 25)]["v"] == 1.5 and cur[(7, 25)]["iv"] == 0.21 and cur[(7, 25)]["n"] == 61,
          "each node carries v = z, its IV, mean, sd and n")
    check(cur[(7, 50)]["v"] is None and cur[(7, 50)]["n"] == 19, "a blank z keeps its n (19)")
    check(cur[(30, 25)]["n"] == 0 and cur[(30, 25)]["v"] is None,
          "a node with no 15:45 history at all reads n = 0, z blank")
    check(body["mode"] == "iv_zscore" and body["lookback"] == "90d"
          and body["window"] == {"start": "2026-01-01", "end": "2026-03-31", "days": 61, "min_days": 20},
          f"the window block: start, end, days = the largest n, min_days ({body['window']})")
    body, _, _ = call([])
    check(body["current"] == [] and body["window"]["days"] == 0, "no snapshot at that time: no nodes, 0 days")


# ── the page, in node ────────────────────────────────────────────────────────

DRIVER = r"""
const fs = require('fs');
let factory = null;
global.window = { addEventListener: () => {} };
global.document = { addEventListener: (ev, fn) => { if (ev === 'alpine:init') fn(); },
                    getElementById: () => null, querySelector: () => null };
global.Alpine = { data: (_n, f) => { factory = f; } };
eval(fs.readFileSync(process.argv[1], 'utf8') + `
;(async () => {
  const out = {};
  // THE SCALE: fixed, clamped, symmetric -- read off the colour the CELL
  // gets, not a call that names the bounds itself.
  const win = { start: '2026-01-01', end: '2026-03-31', days: 61, min_days: 20 };
  const row = (v, n, iv) => ({ dte: 7, put_delta: 25, v, n, iv, mean: 0.18, sd: 0.02 });
  out.colours = [-9, -3, -1.5, 0, 1.5, 3, 9].map(z => zCell(row(z, 61, 0.2), win, '3mZ').bg);
  out.range = Z_RANGE;
  out.cells = {
    z:      zCell(row(1.234, 61, 0.21), win, '3mZ'),
    neg:    zCell(row(-0.5, 61, 0.21), win, '3mZ'),
    noRow:  zCell(undefined, win, '3mZ'),
    noIv:   zCell(row(null, 61, null), win, '3mZ'),
    thin:   zCell(row(null, 12, 0.21), win, '3mZ'),
    flat:   zCell(row(null, 61, 0.21), win, '1yZ'),
  };
  // The colour is a function of z alone: the same z under another window,
  // another node, another n, is the same colour.
  out.sameZ = [zCell(row(1.234, 61, 0.21), win, '3mZ').bg,
               zCell({ ...row(1.234, 250, 0.55), dte: 365 }, { ...win, days: 250 }, '1yZ').bg];

  // THE COMPONENT: modes, the cache, the request.
  const c = factory();
  const asked = [];
  let answer = (url) => ({ mode: 'iv_zscore', window: win, current: [], url });
  global.fetch = async (url) => { asked.push(url); const body = answer(url);
                                  return { ok: true, json: async () => body }; };
  c.renderGrid = () => {};
  c.loadGrid = async () => {};
  c.date = '2026-03-31'; c.time = '15:45'; c.mode = 'iv';
  await c.onSubmodeChange('z3m');
  out.firstUrl = asked[0];
  out.lookbacks = [c.zLookback];
  c.submode = 'z1y'; out.lookbacks.push(c.zLookback);
  c.submode = 'raw'; out.lookbacks.push(c.zLookback);
  c.submode = 'z3m';
  await c.loadZscore();
  out.cached = asked.length;              // still 1
  out.zInfo = c.zInfo;
  // A thin window says so.
  c.zCache['2026-03-31|15:45|90d'] = { window: { ...win, days: 12 }, current: [] };
  out.thinInfo = c.zInfo;
  // A response lands under the snapshot it was ASKED for, even when the
  // selection moves on before it arrives.
  let release;
  global.fetch = async (url) => { asked.push(url);
    await new Promise(r => { release = r; });
    return { ok: true, json: async () => ({ window: win, current: [], url }) }; };
  c.date = '2026-03-30';
  const pending = c.loadZscore();
  c.date = '2026-03-27';
  release();
  await pending;
  out.landed = Object.keys(c.zCache).sort();
  out.lateUrl = c.zCache['2026-03-30|15:45|90d'] && c.zCache['2026-03-30|15:45|90d'].url;
  out.shownForNew = c.zData;             // 03-27 was never fetched: nothing shown
  // Z IS IV ONLY: switching to Skew drops a z submode; asking for one there is refused.
  c.date = '2026-03-31';
  await c.onModeChange('skew');
  out.afterSkew = [c.mode, c.submode, c.zLookback];
  await c.onSubmodeChange('z1y');
  out.zInSkew = c.submode;
  console.log(JSON.stringify(out));
})().catch(e => { console.log(JSON.stringify({ threw: String(e && e.stack || e) })); });
`);
"""


def check_page() -> bool:
    print("page: static/js/heatmap.js in node")
    if shutil.which("node") is None:
        print("  SKIPPED: the page half -- node is not installed")
        return False
    p = subprocess.run(["node", "-e", DRIVER, str(JS)], capture_output=True, text=True,
                       encoding="utf-8", timeout=60)
    try:
        out = json.loads(p.stdout.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError):
        check(False, f"the node driver produced no report: {p.stdout[-400:]} {p.stderr[-800:]}")
        return True
    if "threw" in out:
        check(False, f"the node driver threw: {out['threw'][:600]}")
        return True
    pink, dark, blue = "#ff1a8c", "#2a2a2a", "#1a8cff"
    col = out["colours"]
    check(out["range"] == 3, f"the scale is +/-3 ({out['range']})")
    check(col[0] == pink and col[1] == pink, f"z <= -3 is full pink, clamped ({col[:2]})")
    check(col[3] == dark, f"z = 0 is the dark midpoint ({col[3]})")
    check(col[5] == blue and col[6] == blue, f"z >= +3 is full blue, clamped ({col[5:]})")
    check(col[2] not in (pink, dark) and col[4] not in (blue, dark) and col[2] != col[4],
          f"in between it interpolates, low pink and high blue ({col[2]}, {col[4]})")
    check(out["sameZ"][0] == out["sameZ"][1],
          f"the colour depends on z alone -- the same z under another window, node and n is the same "
          f"colour ({out['sameZ']})")

    cells = out["cells"]
    check(cells["z"]["text"] == "+1.23" and cells["neg"]["text"] == "-0.50",
          f"a z prints signed to two places ({cells['z']['text']}, {cells['neg']['text']})")
    check(cells["z"]["lines"][0] == "3mZ: +1.23" and cells["z"]["lines"][2]
          == "n=61 days at 15:45  (2026-01-01 → 2026-03-31)",
          f"the hover names the view, n and the window ({cells['z']['lines']})")
    for k, want in [("noRow", "(no IV at this node)"), ("noIv", "(no IV at this node)"),
                    ("thin", "(n below 20, not reported)"), ("flat", "(stdev ≈ 0, not reported)")]:
        line = cells[k]["lines"][0]
        check(cells[k]["text"] == "—" and line.endswith(want) and cells[k]["bg"] == "#222",
              f"a blank z ({k}) shows '—' on the blank colour and says why: {line!r}")

    check(out["firstUrl"] == "/api/heatmap/iv_zscore?date=2026-03-31&time=15:45&lookback=90d",
          f"3mZ asks for the 90d window at the viewed snapshot ({out['firstUrl']})")
    check(out["lookbacks"] == ["90d", "1y", None], f"3mZ -> 90d, 1yZ -> 1y, Raw -> none ({out['lookbacks']})")
    check(out["cached"] == 1, f"the same snapshot and window is fetched once ({out['cached']} requests)")
    check(out["zInfo"] == "2026-01-01 → 2026-03-31 · 61 days at 15:45",
          f"the controls line states the window and its days ({out['zInfo']!r})")
    check(out["thinInfo"].endswith("— thin: under 20, z suppressed"),
          f"a window under 20 days says so ({out['thinInfo']!r})")
    check(out["lateUrl"] == "/api/heatmap/iv_zscore?date=2026-03-30&time=15:45&lookback=90d"
          and "2026-03-27|15:45|90d" not in out["landed"],
          f"a late response lands under the snapshot it was asked for, not the one now shown ({out['landed']})")
    check(out["shownForNew"] is None, "and the newly selected snapshot shows nothing until its own arrives")
    check(out["afterSkew"] == ["skew", "raw", None], f"switching to Skew drops a z view ({out['afterSkew']})")
    check(out["zInSkew"] == "raw", f"and a z view cannot be chosen there ({out['zInSkew']})")
    return True


def main() -> int:
    check_handler()
    ran_page = check_page()
    print()
    print(f"heatmap: {OKS} ok")
    if FAILS:
        print(f"FAIL: {len(FAILS)} heatmap z-score check(s) failed")
        return 1
    if not ran_page:
        print("SKIPPED: the page half -- node is not installed")
        return EXIT_SKIPPED
    print("PASS: heatmap IV z-score -- handler and page")
    return 0


if __name__ == "__main__":
    sys.exit(main())
