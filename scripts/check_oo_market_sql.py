"""OO/Mesosim Backtest: run the market-data SQL against a REAL Postgres.

WHY THIS EXISTS. app/oo_backtest/market.py is the page's only reader of
index_ohlc, and every rule in it is the kind that fails quietly: a close
pinned to 15:55 drops early-close days, a prior close taken from
trade_date - 1 is wrong after every weekend, an entry join that reads a bar's
CLOSE is five minutes of lookahead, a NaN stored as 'NaN' rather than NULL
passes an IS NOT NULL test. None of that can be seen by reading the SQL, and
the development box has no copy of the table -- so this starts a throwaway
Postgres cluster, loads bars shaped exactly like the writer's
(Thetadata_Raw_SPX/push_index_ohlc.py: trade_date DATE, quote_time TIME,
DOUBLE PRECISION OHLC, NaN as 'NaN'), and runs the shipped statements
through the shipped join_market().

Fabricated days, all ET, start-labeled. ZERO-FILLED means 78 bars of 0.0 --
exactly how the real table stores every weekend and NYSE holiday. The first
version of this check left non-trading days ABSENT, which is why it passed
while every Monday gap on the VPS came out null.

  2023-06-24/25   Sat/Sun  zero-filled, before any session (coverage must not start here)
  2023-06-29 Thu  full; junk 16:00 row and a pre-market row that must be ignored;
                  VIX3M absent (coverage starts the next day); first day -> no gap
  2023-06-30 Fri  full; SPX 15:55 close is 'NaN' -> close from 15:50 (a
                  full-session fallback, reported); VIX 15:55 open NaN
  2023-07-01/02   Sat/Sun  zero-filled weekend
  2023-07-03 Mon  EARLY CLOSE, last bar 12:55; gap must use FRIDAY 06-30
  2023-07-04 Tue  zero-filled holiday
  2023-07-05 Wed  full; gap must use 07-03's 12:55 close; one bar's SPX high is
                  'NaN' and another's SPX low is 0.0 (neither may reach high/low)
  2023-07-06 Thu  full; the 10:00 VIX open is 0.0 (an entry there takes 09:55)
  2023-07-07 Fri  HOLIDAY WITH VIX ARTIFACTS: 25 valid VIX bars (09:30-11:30),
                  everything else zero -- the shape VIX ingestion writes on
                  real holidays. Not a session for any series. Then 07-08/09
                  zero-filled weekend.
  2023-07-10 Mon  full; SPX and VIX gaps must use THURSDAY 07-06, not the
                  artifact Friday (the 2026-07-06 bug)
  2023-07-11 Tue  FULL VIX SESSION, NO SPX (78 VIX bars; SPX/VIX3M/VIX9D zero)
                  -- the 2026-04-08 shape: a VIX session, not an SPX one
  2023-07-12 Wed  full; SPX gap must use 07-10 (skipping 07-11), VIX gap 07-11
  2023-07-15/16   Sat/Sun  zero-filled, after the last session (freshness must not report them)

Planted faults, each of which must change the answer: a close pinned to
15:55, a prior close from trade_date - 1, an entry join reading the bar
close, a raw max() over a NaN high, a "previous row" that includes the
zero-filled days, and a "previous row" over days with ANY valid bar (which
takes the VIX-artifact holiday).

EXITS 3 (skipped), never 0, where it cannot run: no Postgres server binaries,
no asyncpg, or running as root (initdb refuses root). scripts/gates.py shows
that as SKIP, and as FAIL under --deploy. Set PG_BIN to the directory holding
initdb/pg_ctl if they are somewhere unusual.

    python scripts/check_oo_market_sql.py
"""
from __future__ import annotations

import asyncio
import glob
import math
import os
import shutil
import socket
import subprocess
import sys
import tempfile
from datetime import date, datetime, time, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from app.oo_backtest import market  # noqa: E402

FAILS: list[str] = []

# Not 0. A check that did not run must not be counted as one that passed.
EXIT_SKIPPED = 3


def check(ok: bool, msg: str) -> None:
    print(("  ok    " if ok else "  FAIL  ") + msg)
    if not ok:
        FAILS.append(msg)


def find_bin() -> Path | None:
    exe = ".exe" if os.name == "nt" else ""
    cands = [os.environ.get("PG_BIN", "")]
    cands += sorted(glob.glob(r"C:\Program Files\PostgreSQL\*\bin"), reverse=True)
    cands += sorted(glob.glob("/usr/lib/postgresql/*/bin"), reverse=True)
    which = shutil.which("initdb")
    if which:
        cands.append(str(Path(which).parent))
    for c in cands:
        if c and (Path(c) / f"initdb{exe}").exists() and (Path(c) / f"pg_ctl{exe}").exists():
            return Path(c)
    return None


# ── fabricated bars ─────────────────────────────────────────────────────────

DAYS = [date(2023, 6, 29), date(2023, 6, 30), date(2023, 7, 3), date(2023, 7, 5),
        date(2023, 7, 6), date(2023, 7, 10), date(2023, 7, 11), date(2023, 7, 12)]
EARLY = date(2023, 7, 3)
NAN = "NaN"
ZERO_WEEKEND = [date(2023, 6, 24), date(2023, 6, 25), date(2023, 7, 1), date(2023, 7, 2),
                date(2023, 7, 8), date(2023, 7, 9), date(2023, 7, 15), date(2023, 7, 16)]
# 07-04 is the only real holiday in this range, so it is where artifact bars
# belong: valid VIX values written on a day the exchange was SHUT.
ZERO_HOLIDAY: list = []
ARTIFACT_HOLIDAY = date(2023, 7, 4)          # 25 valid VIX bars on a CLOSED day
# 07-07 is a Friday and an ordinary session. The writer left it with those
# same 25 VIX bars and nothing else, which under the calendar is what it
# really is: a TRADING DAY with SPX missing and VIX partial.
SPARSE_SESSION = date(2023, 7, 7)
ARTIFACT_BARS = 25
VIX_ONLY = date(2023, 7, 11)                 # full VIX session, SPX/VIX3M/VIX9D zero
WED2 = date(2023, 7, 12)
SPX_DAYS = [d for d in [date(2023, 6, 29), date(2023, 6, 30), date(2023, 7, 3), date(2023, 7, 5),
                        date(2023, 7, 6), date(2023, 7, 10), date(2023, 7, 12)]]
WED, THU, MON2 = date(2023, 7, 5), date(2023, 7, 6), date(2023, 7, 10)
NAN_HIGH_BAR, ZERO_LOW_BAR = time(10, 20), time(11, 10)   # on WED
ZERO_VIX_OPEN_BAR = time(10, 0)                            # on THU


def session_times(d: date) -> list[time]:
    last = time(12, 55) if d == EARLY else time(15, 55)
    out, t = [], time(9, 30)
    while t <= last:
        out.append(t)
        m = t.hour * 60 + t.minute + 5
        t = time(m // 60, m % 60)
    return out


def px(series: str, di: int, k: int) -> dict:
    base = {"spx": 4000.0, "vix": 15.0, "vix3m": 17.0, "vix9d": 14.0}[series]
    step = {"spx": 0.5, "vix": 0.01, "vix3m": 0.01, "vix9d": 0.02}[series]
    o = base + di * {"spx": 25.0, "vix": 1.0, "vix3m": 1.0, "vix9d": 1.0}[series] + k * step
    return {"open": round(o, 4), "high": round(o + 0.3, 4), "low": round(o - 0.3, 4), "close": round(o + 0.1, 4)}


def build_rows() -> list[tuple]:
    rows = []
    for di, d in enumerate(DAYS):
        for k, t in enumerate(session_times(d)):
            vals = {}
            for s in market.SERIES:
                p = px(s, di, k)
                for f in ("open", "high", "low", "close"):
                    vals[f"{s}_{f}"] = p[f]
            if d == DAYS[0]:
                for f in ("open", "high", "low", "close"):
                    vals[f"vix3m_{f}"] = None           # VIX3M coverage starts 06-30
            if d == DAYS[1] and t == time(15, 55):
                vals["spx_close"] = NAN                 # 'NaN', not NULL
                vals["vix_open"] = NAN
            if d == WED and t == NAN_HIGH_BAR:
                vals["spx_high"] = NAN                  # max() over raw highs would be NaN
            if d == WED and t == ZERO_LOW_BAR:
                vals["spx_low"] = 0.0                   # min() over raw lows would be 0
            if d == THU and t == ZERO_VIX_OPEN_BAR:
                vals["vix_open"] = 0.0                  # a zero bar inside a real session
            if d == VIX_ONLY:
                for sr in ("spx", "vix3m", "vix9d"):    # the 2026-04-08 shape
                    for f in ("open", "high", "low", "close"):
                        vals[f"{sr}_{f}"] = 0.0
            rows.append((d, t, vals))
        if d not in (EARLY, VIX_ONLY):
            # 16:00 partial row with absurd values: must never be read.
            rows.append((d, time(16, 0), {f"{s}_{f}": 99999.0 for s in market.SERIES
                                         for f in ("open", "high", "low", "close")}))
    # Zero-filled non-trading days, as the real table has them: every bar of
    # the session window, every column 0.0.
    for d in ZERO_WEEKEND + ZERO_HOLIDAY:
        for t in session_times(date(2023, 6, 29)):
            rows.append((d, t, {c: 0.0 for c in OHLC_COLS_ALL}))
    # The holiday VIX artifacts: the first 25 bars carry plausible VIX values,
    # every other value on the day is zero.
    for k, t in enumerate(session_times(date(2023, 6, 29))):
        vals = {c: 0.0 for c in OHLC_COLS_ALL}
        if k < ARTIFACT_BARS:
            for f, v in px("vix", 4, k).items():
                vals[f"vix_{f}"] = v
        rows.append((ARTIFACT_HOLIDAY, t, vals))
    # THE SAME 25 BARS ON A TRADING DAY. Identical data, opposite meaning:
    # on 07-04 it is a value written to a closed market, on 07-07 it is a
    # session the pipeline under-filled. The old rule could not tell them
    # apart -- both were "no series reaches 34 bars" -- which is exactly the
    # hiding the calendar removes.
    for k, t in enumerate(session_times(date(2023, 6, 29))):
        vals = {c: 0.0 for c in OHLC_COLS_ALL}
        if k < ARTIFACT_BARS:
            for f, v in px("vix", 4, k).items():
                vals[f"vix_{f}"] = v
        rows.append((SPARSE_SESSION, t, vals))
    # A pre-market row on day one, also never read.
    rows.append((DAYS[0], time(9, 25), {f"{s}_{f}": -1.0 for s in market.SERIES
                                        for f in ("open", "high", "low", "close")}))
    return rows


OHLC_COLS_ALL = [f"{s}_{f}" for s in market.SERIES for f in ("open", "high", "low", "close")]


def expected_hi_lo(d: date) -> tuple[float, float]:
    di, ts = DAYS.index(d), session_times(d)
    hi = max(px("spx", di, k)["high"] for k, t in enumerate(ts) if not (d == WED and t == NAN_HIGH_BAR))
    lo = min(px("spx", di, k)["low"] for k, t in enumerate(ts) if not (d == WED and t == ZERO_LOW_BAR))
    return hi, lo


def expected_close(s: str, d: date) -> tuple[float, time]:
    di = DAYS.index(d)
    ts = session_times(d)
    k = len(ts) - 1
    if s == "spx" and d == DAYS[1]:
        k -= 1
    return px(s, di, k)["close"], ts[k]


def bar_open(s: str, d: date, t: time) -> float:
    return px(s, DAYS.index(d), session_times(d).index(t))["open"]


# ── cluster ─────────────────────────────────────────────────────────────────

def free_port() -> int:
    with socket.socket() as so:
        so.bind(("127.0.0.1", 0))
        return so.getsockname()[1]


class Cluster:
    def __init__(self, bindir: Path):
        self.bin = bindir
        self.dir = Path(tempfile.mkdtemp(prefix="oo_market_pg_"))
        self.port = free_port()

    def run(self, *argv, timeout=120):
        # Output goes to a FILE, never a pipe: `pg_ctl start` hands its stdout
        # to the server it launches, so a captured pipe never reaches EOF and
        # subprocess.run waits for a process that is meant to keep running.
        # (This hung the first version of this script on Windows.)
        out = self.dir / f"{argv[0]}.out"
        with open(out, "w") as fh:
            p = subprocess.run([str(self.bin / argv[0]), *argv[1:]], stdout=fh, stderr=subprocess.STDOUT,
                               stdin=subprocess.DEVNULL, timeout=timeout)
        if p.returncode:
            raise RuntimeError(f"{argv[0]} failed: {out.read_text(errors='replace').strip()}")

    def __enter__(self):
        data = self.dir / "data"
        self.run("initdb", "-D", str(data), "-U", "postgres", "-A", "trust", "-E", "UTF8", "--no-sync")
        try:
            self.run("pg_ctl", "-D", str(data), "-l", str(self.dir / "log.txt"), "-w", "-t", "60",
                     "-o", f"-p {self.port} -h 127.0.0.1 -c fsync=off", "start")
        except Exception:
            self.__exit__(None, None, None)
            raise
        return self

    def __exit__(self, *exc):
        try:
            if (self.dir / "data" / "postmaster.pid").exists():
                self.run("pg_ctl", "-D", str(self.dir / "data"), "-m", "fast", "-w", "stop")
        except Exception as e:  # noqa: BLE001 — a stuck stop is reported, the rmtree still runs
            print(f"  WARN  could not stop the temporary cluster cleanly: {e}")
        finally:
            shutil.rmtree(self.dir, ignore_errors=True)

    @property
    def dsn(self):
        return f"postgresql://postgres@127.0.0.1:{self.port}/postgres"


OHLC_COLS = [f"{s}_{f}" for s in market.SERIES for f in ("open", "high", "low", "close")]


async def load(pool) -> None:
    async with pool.acquire() as conn:
        cols = ",\n".join(f"{c} DOUBLE PRECISION" for c in OHLC_COLS)
        await conn.execute(f"""CREATE TABLE index_ohlc (
            trade_date DATE NOT NULL, quote_time TIME NOT NULL, {cols},
            PRIMARY KEY (trade_date, quote_time))""")
        recs = []
        for d, t, vals in build_rows():
            recs.append((d, t, *[float("nan") if vals.get(c) == NAN else vals.get(c) for c in OHLC_COLS]))
        await conn.copy_records_to_table("index_ohlc", records=recs, columns=["trade_date", "quote_time", *OHLC_COLS])
        n_nan = await conn.fetchval("SELECT count(*) FROM index_ohlc WHERE spx_close = 'NaN'::float8")
        assert n_nan == 1, n_nan   # the fixture really stores NaN, not NULL
        n_zero_days = await conn.fetchval("""SELECT count(*) FROM (SELECT trade_date FROM index_ohlc
                                             GROUP BY trade_date HAVING max(spx_close) = 0 AND max(vix_close) = 0) z""")
        assert n_zero_days == len(ZERO_WEEKEND + ZERO_HOLIDAY), n_zero_days   # really zero-filled, not absent
        n_art = await conn.fetchval("SELECT count(*) FROM index_ohlc WHERE trade_date = $1 AND vix_close > 0",
                                    ARTIFACT_HOLIDAY)
        assert n_art == ARTIFACT_BARS, n_art


async def run(dsn: str) -> None:
    import asyncpg
    pool = await asyncpg.create_pool(dsn, min_size=1, max_size=3)
    try:
        await load(pool)
        await check_daily(pool)
        await check_entry_join(pool)
        await check_diagnostics(pool)
        await check_planted(pool)
        await check_end_to_end(pool)
        await check_saved_strategies(pool)
        await check_saved_sweeps(pool)
        await check_portfolio_profiles(pool)
        await check_surface(pool)
        await check_surface_sweep_scale(pool)
    finally:
        await pool.close()


async def check_daily(pool) -> None:
    print("daily rollup")
    daily, fresh = await market.get_daily(pool)
    by = {r["trade_date"]: r for r in daily.to_dict("records")}
    # ONE ROW PER EXCHANGE SESSION, whatever the table holds for it. 07-07 is
    # a Friday with 25 VIX bars and nothing else: under the old rule it was
    # not a row at all, which is how a half-empty trading day stayed
    # invisible. Now it is a row with everything null and session flags off.
    from app.oo_backtest import market_calendar as mc
    cal_days = [date.fromisoformat(d) for d in mc.sessions(date(2023, 6, 24), date(2023, 7, 16))]
    check(sorted(by) == cal_days,
          f"one row per NYSE session in the table's range -- {len(cal_days)} of "
          f"them, including sessions the fixture has no bars for at all "
          f"({len(by)} rows)")
    empty = [d for d in cal_days if d not in DAYS and d != SPARSE_SESSION]
    check(len(empty) == 5 and all(pd.isna(by[d]["spx_close"]) for d in empty),
          f"a session the table never wrote is still a row, with nothing in "
          f"it ({[d.isoformat() for d in empty]})")
    check(ARTIFACT_HOLIDAY not in by,
          "07-04 is a holiday, so its 25 VIX bars are not a rollup row at all")
    for d in ZERO_WEEKEND:
        check(d not in by, f"{d} is a weekend and not a row")
    sp = by[SPARSE_SESSION]
    check(not sp["spx_session"] and not sp["vix_session"],
          "07-07: no series is complete, so no session flag is set")
    check(pd.isna(sp["spx_close"]) and pd.isna(sp["vix_close"]),
          "07-07: 25 of 78 bars is short by more than the tolerance, so not "
          "even VIX keeps a price")
    check(sp["expected_bars"] == 78, "07-07 expects 78 bars, from its own close")
    check(by[EARLY]["expected_bars"] == 42,
          "the early close expects 42 -- which a single floor could not say")
    check(fresh["latest_date"] == "2023-07-12" and fresh["latest_time"] == "15:55:00"
          and fresh["latest_raw_date"] == "2023-07-16",
          f"freshness is the last VALID session bar, not the trailing zero-filled weekend "
          f"({fresh['latest_date']} {fresh['latest_time']}; raw {fresh['latest_raw_date']})")
    for d in SPX_DAYS:
        r = by[d]
        want, wt = expected_close("spx", d)
        check(r["spx_close"] == want and r["spx_close_time"] == wt,
              f"{d}: SPX close {r['spx_close']} from {r['spx_close_time']} (want {want} @ {wt})")
        check(r["spx_open"] == bar_open("spx", d, time(9, 30)), f"{d}: SPX open from the 09:30 bar, not 09:25")
        hi, lo = expected_hi_lo(d)
        check(r["spx_high"] == hi and r["spx_low"] == lo,
              f"{d}: high/low ignore the 16:00 row, a 'NaN' high and a zero low ({r['spx_high']}/{r['spx_low']} vs {hi}/{lo})")
    check(by[DAYS[3]]["spx_prev_date"] == EARLY, "07-05's previous SPX session is 07-03 (not the 07-04 holiday)")
    check(by[DAYS[3]]["spx_prev_close"] == expected_close("spx", EARLY)[0],
          "07-05's prior close is 07-03's 12:55 close")
    check(by[DAYS[2]]["spx_prev_date"] == DAYS[1] and by[DAYS[2]]["spx_prev_close"] == expected_close("spx", DAYS[1])[0],
          "Monday 07-03's previous SPX session is FRIDAY 06-30, not the zero-filled Sunday")
    check(by[MON2]["spx_prev_date"] == THU and by[MON2]["spx_prev_close"] == expected_close("spx", THU)[0]
          and by[MON2]["vix_prev_date"] == THU and by[MON2]["vix_prev_close"] == expected_close("vix", THU)[0],
          "Monday 07-10: SPX AND VIX previous session is THURSDAY 07-06, not the VIX-artifact Friday (2026-07-06 bug)")
    vo = by[VIX_ONLY]
    check(vo["vix_session"] and not vo["spx_session"] and vo["spx_bars"] == 0 and pd.isna(vo["spx_close"])
          and pd.isna(vo["spx_high"]) and vo["vix_close"] == expected_close("vix", VIX_ONLY)[0],
          "07-11 (full VIX, no SPX) is a VIX session and not an SPX one; SPX fields null")
    check(by[WED2]["spx_prev_date"] == MON2 and by[WED2]["vix_prev_date"] == VIX_ONLY,
          "07-12: SPX previous session is 07-10 (skips the VIX-only day); VIX previous session is 07-11")
    check(pd.isna(vo["spx_prev_date"]) and vo["vix_prev_date"] == MON2,
          "07-11 has no SPX previous session of its own (not an SPX session); its VIX previous session is 07-10")
    check(by[DAYS[3]]["vix_prev_close"] == expected_close("vix", EARLY)[0],
          "VIX prior close follows the same trading-day rows")
    check(pd.isna(by[DAYS[0]]["spx_prev_close"]), "first date: no prior close -> null")
    check(pd.isna(by[DAYS[0]]["vix3m_close"]) and by[DAYS[1]]["vix3m_close"] is not None,
          "VIX3M null before its coverage")

    rep = market.fallback_report(daily)
    check(rep["spx"]["early_close_days"] == ["2023-07-03"], f"SPX early-close fallback: {rep['spx']['early_close_days']}")
    check(rep["spx"]["full_session_count"] == 1 and rep["spx"]["full_session_sample"] == ["2023-06-30"],
          "SPX full-session fallback counted once (the NaN 15:55 close)")
    check(rep["vix"]["full_session_count"] == 0, "VIX: no full-session fallback (its 15:55 close is fine)")
    cov = market.coverage(daily)
    check(cov == {"spx": "2023-06-29", "vix": "2023-06-29", "vix3m": "2023-06-30", "vix9d": "2023-06-29"},
          f"coverage from real closes, not from the zero-filled 06-24 {cov}")
    labels = market.bar_labels()
    y = labels[0]
    check(y["days"] == 7 and y["days_with_0930"] == 7 and y["days_with_1555"] == 5   # 06-30 NaN close, 07-03 early
          and y["days_with_valid_1600"] == 6 and y["days_with_premarket"] == 1
          and y["spx_incomplete_sessions"] == len(cal_days) - 7,
          f"bar-label diagnostics count SPX-COMPLETE sessions; the rest are "
          f"sessions SPX did not fill {y}")
    z = market.sessions()
    check(z["market"] == "NYSE" and z["sessions"] == len(cal_days),
          f"the report counts exchange sessions, not days with enough bars ({z['sessions']})")
    # AN ARTIFACT IS NOW A DAY THE MARKET WAS SHUT, full stop -- no threshold.
    check(z["artifact_days"] == 1 and z["artifacts"][0]["date"] == "2023-07-04"
          and z["artifacts"][0]["vix"] == 25,
          f"07-04 carries 25 VIX bars on a CLOSED day: an artifact ({z['artifacts']})")
    sx, vx = z["by_series"]["spx"], z["by_series"]["vix"]
    # THE CATEGORY THAT DID NOT EXIST. A trading day with no SPX data is a
    # hole in the pipeline; it used to read as "not an SPX session", which is
    # what a holiday reads as.
    missing = [m["date"] for m in sx["missing"]]
    check("2023-07-11" in missing,
          f"SPX is MISSING on 07-11 -- a trading day with no SPX bars, no "
          f"longer indistinguishable from a holiday ({missing})")
    check("2023-07-07" in missing, "and on 07-07, the under-filled Friday")
    check(sx["complete"] == 7, f"SPX is complete on 7 sessions ({sx['complete']})")
    check(sx["missing_days"] == len(cal_days) - 7,
          f"and missing on the other {len(cal_days) - 7}")
    # PARTIAL IS ITS OWN THING: some data, short by more than the tolerance.
    check(vx["partial"] and vx["partial"][0]["date"] == "2023-07-07"
          and vx["partial"][0]["bars"] == 25 and vx["partial"][0]["expected"] == 78
          and vx["partial"][0]["short"] == 53,
          f"VIX on 07-07 is PARTIAL: 25 of 78, short 53 ({vx['partial']})")
    check(vx["by_shortfall"].get(53) == 1,
          f"shortfalls are grouped by size, so \"short by 53 on 1 session\" "
          f"is readable ({vx['by_shortfall']})")
    check("short by 53" in vx["summary"] and "no data at all" in vx["summary"],
          f"and the summary says both in words ({vx['summary']})")
    check(z["tolerance"] == 2, "the tolerance is reported with the numbers it shaped")
    # 06-30: 'NaN' SPX close + 'NaN' VIX open; 07-05: 'NaN' SPX high + zero SPX
    # low; 07-06: zero VIX open -- counted inside sessions, zero and NaN apart.
    check((sx["zero_bars_in_sessions"], sx["nan_bars_in_sessions"], vx["zero_bars_in_sessions"], vx["nan_bars_in_sessions"])
          == (1, 2, 1, 1), f"invalid bars inside sessions counted per series ({sx['zero_bars_in_sessions']}/"
                           f"{sx['nan_bars_in_sessions']}, {vx['zero_bars_in_sessions']}/{vx['nan_bars_in_sessions']})")


def trades() -> pd.DataFrame:
    rows = [
        # (date, time, why)
        (EARLY, "12:30:00"),        # 0 early-close entry -> 12:30 bar
        (DAYS[0], "09:30:00"),      # 1 09:30 entry -> 09:30 bar (not 04:30 / 14:30)
        (DAYS[3], "09:29:00"),      # 2 before the open -> null, NOT the prior session
        (DAYS[3], None),            # 3 no entry time -> 09:30 bar, counted
        (DAYS[0], "16:00:00"),      # 4 at the close -> 15:55 bar, never the 16:00 row
        (DAYS[1], "3:57 PM"),       # 5 VIX 15:55 open is NaN -> 15:50 bar; VIX3M/VIX9D keep 15:55
        (DAYS[3], "10:02:30"),      # 6 mid-bar -> 10:00 bar's OPEN
        (MON2, "09:30:00"),         # 7 Monday after a zero-filled Friday + weekend
        (THU, "10:00:00"),          # 8 VIX 10:00 open is 0.0 -> 09:55 for VIX only
        (VIX_ONLY, "10:00:00"),     # 9 full VIX session, no SPX: VIX computes, SPX null
        (WED2, "09:30:00"),         # 10 SPX gap vs 07-10, VIX gap vs 07-11
        (ARTIFACT_HOLIDAY, "10:00:00"),  # 11 a VIX artifact bar exists at 10:00 -> level null
    ]
    return pd.DataFrame({"date_opened": pd.to_datetime([d for d, _ in rows]),
                         "time_opened": [t for _, t in rows], "pnl": [1.0] * len(rows)})


async def check_entry_join(pool) -> None:
    print("entry-time join")
    df, rep = await market.join_market(pool, trades())
    L = df.to_dict("records")
    check(L[0]["vix_level"] == bar_open("vix", EARLY, time(12, 30)) and L[0]["vix_bar_time"] == "12:30:00",
          "early-close 2023-07-03 12:30 entry lands on the 12:30 bar")
    check(L[1]["vix_bar_time"] == "09:30:00" and L[1]["vix_level"] == bar_open("vix", DAYS[0], time(9, 30)),
          "09:30 entry lands on the 09:30:00 bar")
    check(all(pd.isna(L[2][f"{s}_level"]) for s in ("vix", "vix3m", "vix9d")),
          "09:29 entry -> null levels, never the previous session")
    check(L[3]["vix_bar_time"] == "09:30:00", "missing entry time -> 09:30 bar")
    check(L[4]["vix_bar_time"] == "15:55:00" and L[4]["vix_level"] != 99999.0, "16:00 entry -> 15:55 bar, 16:00 row unread")
    check(L[5]["vix_bar_time"] == "15:50:00" and L[5]["vix3m_bar_time"] == "15:55:00",
          "NaN VIX open on the entry bar -> 15:50 for VIX only; VIX3M keeps 15:55 ('3:57 PM' parsed)")
    check(L[6]["vix_level"] == bar_open("vix", DAYS[3], time(10, 0)), "mid-bar entry takes the bar OPEN")
    check(pd.isna(L[1]["vix3m_level"]), "VIX3M before its coverage -> null")
    check(L[8]["vix_bar_time"] == "09:55:00" and L[8]["vix_level"] == bar_open("vix", THU, time(9, 55))
          and L[8]["vix3m_bar_time"] == "10:00:00",
          "a ZERO VIX open on the entry bar -> 09:55 for VIX only, never a level of 0")
    check(L[9]["vix_level"] == bar_open("vix", VIX_ONLY, time(10, 0)) and pd.isna(L[9]["gap"])
          and math.isclose(L[9]["vix_overnight_gap"],
                           (bar_open("vix", VIX_ONLY, time(9, 30)) - expected_close("vix", MON2)[0])
                           / expected_close("vix", MON2)[0] * 100),
          "VIX-only day: VIX level and VIX gap (vs 07-10) compute; SPX gap null")
    check(pd.isna(L[11]["vix_level"]) and pd.isna(L[11]["vix_bar_time"]),
          "an entry on the VIX-artifact holiday gets no VIX level, though a VIX bar exists at that time")
    e = rep["entry_time"]
    check(e["entry_time_found"] == 11 and e["entry_time_fallback_0930"] == 1 and e["before_open"] == 1,
          f"entry-time coverage reported {e}")
    check(rep["entry_bars"]["vix"]["earlier_than_entry_bar"] == 2, "bars pushed back by NaN and by zero are counted")

    print("gaps and ratios")
    d3c = expected_close("spx", EARLY)[0]
    want = (bar_open("spx", DAYS[3], time(9, 30)) - d3c) / d3c * 100
    check(math.isclose(L[6]["gap"], want, rel_tol=1e-12), f"07-05 SPX gap vs 07-03 12:55 close ({L[6]['gap']:.5f}%)")
    check(pd.isna(L[1]["gap"]), "first date gap is null, not zero (the zero-filled 06-25 is not its prior day)")
    fc = expected_close("spx", DAYS[1])[0]
    check(math.isclose(L[0]["gap"], (bar_open("spx", EARLY, time(9, 30)) - fc) / fc * 100),
          f"Monday 07-03 SPX gap is against FRIDAY's close ({L[0]['gap']:.5f}%), not null")
    tc = expected_close("spx", THU)[0]
    check(math.isclose(L[7]["gap"], (bar_open("spx", MON2, time(9, 30)) - tc) / tc * 100),
          f"Monday 07-10 SPX gap is against THURSDAY's close ({L[7]['gap']:.5f}%)")
    c10 = expected_close("spx", MON2)[0]
    check(math.isclose(L[10]["gap"], (bar_open("spx", WED2, time(9, 30)) - c10) / c10 * 100),
          f"07-12 SPX gap is against 07-10, skipping the VIX-only 07-11 ({L[10]['gap']:.5f}%)")
    v11 = expected_close("vix", VIX_ONLY)[0]
    check(math.isclose(L[10]["vix_overnight_gap"], (bar_open("vix", WED2, time(9, 30)) - v11) / v11 * 100),
          "07-12 VIX gap is against 07-11's VIX close")
    vt = expected_close("vix", THU)[0]
    check(math.isclose(L[7]["vix_overnight_gap"], (bar_open("vix", MON2, time(9, 30)) - vt) / vt * 100),
          "Monday 07-10 VIX gap is against THURSDAY's VIX close, not the artifact Friday's")
    g = rep["gaps"]
    check(g["spx"] == {"computed": 8, "null": 4},
          f"gap computed/null counts (06-29 x2, VIX-only day, artifact holiday) {g['spx']}")
    vprev = expected_close("vix", DAYS[1])[0]
    check(math.isclose(L[0]["vix_overnight_gap"], (bar_open("vix", EARLY, time(9, 30)) - vprev) / vprev * 100),
          "VIX gap: 09:30 open vs prior row's close")
    check(math.isclose(L[6]["vix3m_vix_ratio_entry"], L[6]["vix3m_level"] / L[6]["vix_level"]),
          "entry-basis ratio from the entry bars")
    vc, v3c = expected_close("vix", DAYS[3])[0], expected_close("vix3m", DAYS[3])[0]
    check(math.isclose(L[6]["vix3m_vix_ratio_close"], v3c / vc), "close-basis ratio from the daily closes")

async def check_diagnostics(pool) -> None:
    """Why a value is null -- each against the fixture trade that causes it."""
    print("null reasons")
    _, rep_ = await market.join_market(pool, trades())
    nr = rep_["null_reasons"]
    not_session = "entry date is not an exchange session (the market was shut)"
    check(nr["spx_gap"]["reasons"] == {"first SPX session in the table (no earlier SPX session)": 2,
                                        "SPX has no bars at all on this exchange session (short by more than 2)": 1,
                                        not_session: 1},
          f"null SPX gaps: first session x2, the VIX-only day names SPX, the artifact holiday is no session "
          f"({nr['spx_gap']['reasons']})")
    check(nr["vix_gap"]["reasons"] == {"first VIX session in the table (no earlier VIX session)": 2, not_session: 1},
          f"null VIX gaps: the VIX-only day is NOT among them ({nr['vix_gap']['reasons']})")
    check(nr["vix3m"]["reasons"] == {"entry date before VIX3M coverage (2023-06-30)": 2, "entry before 09:30": 1,
                                      "VIX3M has no bars at all on this exchange session (short by more than 2)": 1,
                                      not_session: 1},
          f"null VIX3M levels explained ({nr['vix3m']['reasons']})")
    check(nr["vix"]["reasons"] == {"entry before 09:30": 1, not_session: 1},
          f"null VIX levels: pre-open and the artifact holiday ({nr['vix']['reasons']})")
    check(rep_["diagnostic_errors"] == {}, f"no diagnostic raised ({rep_['diagnostic_errors']})")


async def check_planted(pool) -> None:
    print("planted faults")
    async with pool.acquire() as conn:
        pinned = await conn.fetchval("""SELECT NULLIF(spx_close, 'NaN'::float8) FROM index_ohlc
                                        WHERE trade_date = $1 AND quote_time = TIME '15:55'""", EARLY)
        cal_prev = await conn.fetchval("""SELECT NULLIF(spx_close, 'NaN'::float8) FROM index_ohlc
                                          WHERE trade_date = $1::date - 1 AND quote_time = TIME '15:55'""", DAYS[3])
        raw_high = await conn.fetchval("SELECT max(spx_high) FROM index_ohlc WHERE trade_date = $1 AND "
                                       "quote_time BETWEEN TIME '09:30' AND TIME '15:55'", WED)
        raw_prev_row_close = await conn.fetchval("""SELECT spx_close FROM index_ohlc
            WHERE trade_date = (SELECT max(trade_date) FROM index_ohlc WHERE trade_date < $1)
              AND quote_time = TIME '15:55'""", EARLY)
        bar_close = await conn.fetchval("""SELECT vix_close FROM index_ohlc
                                           WHERE trade_date = $1 AND quote_time = TIME '10:00'""", DAYS[3])
        # "Any valid bar" as the session rule: the day before Monday 07-10
        # that has one is the VIX-artifact Friday.
        any_valid_prev = await conn.fetchval("""SELECT max(trade_date) FROM index_ohlc
            WHERE trade_date < $1 AND quote_time BETWEEN TIME '09:30' AND TIME '15:55'
              AND ((spx_close > 0 AND spx_close <> 'NaN') OR (vix_close > 0 AND vix_close <> 'NaN'))""", MON2)
    daily, _ = await market.get_daily(pool)
    by = {r["trade_date"]: r for r in daily.to_dict("records")}
    check(pinned is None and by[EARLY]["spx_close"] is not None,
          "a close pinned to 15:55 loses the early-close day; the rollup does not")
    check(cal_prev in (None, 0.0) and by[DAYS[3]]["spx_prev_close"] == expected_close("spx", EARLY)[0],
          f"a prior close from trade_date - 1 lands on the zero-filled holiday ({cal_prev}); the previous row does not")
    check(isinstance(raw_high, float) and math.isnan(raw_high) and not math.isnan(by[WED]["spx_high"]),
          f"a raw max() over a 'NaN' high IS NaN in Postgres ({raw_high}); the rollup's high is {by[WED]['spx_high']}")
    check(any_valid_prev == SPARSE_SESSION and by[MON2]["spx_prev_date"] == THU and by[MON2]["vix_prev_date"] == THU,
          f"an 'any valid bar' rule takes the under-filled Friday ({any_valid_prev}) as Monday's "
          f"previous day; completeness takes Thursday -- and the day IS a session, so the "
          f"distinction is now about the DATA, not about whether the market was open")
    check(raw_prev_row_close == 0.0 and by[EARLY]["spx_prev_date"] == DAYS[1],
          "a previous row over ALL dates gives Monday a close of 0 (the zero-filled Sunday); the rollup gives Friday")
    df, _ = await market.join_market(pool, trades())
    check(df.loc[6, "vix_level"] != bar_close, "the entry join is not reading the bar's close (5 min lookahead)")


async def check_saved_sweeps(pool) -> None:
    """A parameter sweep saved as ONE strategy (P4), through the real
    endpoints on this database: saved with every file, listed with its
    variant count, loaded back as the same payload an upload of the same
    files gives, a changed variant list said, a corrupt or missing file
    refused, a single file saved over it taking its files away, deleted with
    its files, and refused by the portfolio with the way round it."""
    print("saved sweeps (endpoints, this database)")
    import gzip as _gz
    import io as _io
    import json as _json

    import httpx
    from fastapi import FastAPI

    from app.db import get_pool
    from app.oo_backtest import store
    from app.routers import backtest_portfolio as BP
    from app.routers import oo_backtest as R

    app = FastAPI()
    app.include_router(R.router, prefix="/api/oo-backtest")
    app.include_router(BP.router)               # it carries its own /api/backtest-portfolio prefix
    app.dependency_overrides[get_pool] = lambda: pool
    hdr = ["Date Opened", "Time Opened", "Date Closed", "Time Closed", "P/L", "Legs", "Premium"]

    def oo(rows):
        out = _io.StringIO()
        pd.DataFrame(rows, columns=hdr).to_csv(out, index=False)
        return out.getvalue().encode()
    files = {
        "sw_dte7_pt25.csv": oo([["2023-07-03", "12:30:00", "2023-07-05", "15:59:00", 100, "x", 1],
                               ["2023-07-05", "10:00:00", "2023-07-06", "15:59:00", -40, "x", 1]]),
        "sw_dte7_pt50.csv": oo([["2023-07-06", "10:00:00", "2023-07-10", "15:59:00", 60, "x", 1]]),
        "sw_dte14_pt25.csv": oo([["2023-07-05", "10:00:00", "2023-07-12", "15:59:00", -10, "x", 1],
                                ["2023-07-10", "10:00:00", "2023-07-12", "15:59:00", 30, "x", 1]]),
    }
    parts = [("files", (n, _gz.compress(c))) for n, c in files.items()]
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t", timeout=60) as c:
        r = await c.post("/api/oo-backtest/strategies/sweep", files=parts,
                         data={"name": "dte sweep", "notes": "three files", "capital_per_position": "25000"})
        sv = r.json().get("strategy", {}) if r.status_code == 200 else {}
        check(r.status_code == 200 and sv.get("variants") == 3 and sv.get("trade_count") == 5
              and sv.get("filename") == "3 files" and sv.get("capital_per_position") == 25000,
              f"a 3-file sweep saves as ONE strategy: 3 variants, 5 trades ({r.status_code} {sv})")
        sid = sv.get("id")
        async with pool.acquire() as conn:
            stored = await conn.fetch("SELECT ord, filename, file_bytes FROM oo_backtest_strategy_files "
                                      "WHERE strategy_id = $1 ORDER BY ord", sid)
        check([(x["filename"], x["file_bytes"]) for x in stored] == [(n, len(b)) for n, b in files.items()],
              "every file stored, in upload order, at its original size")
        r = await c.post("/api/oo-backtest/strategies/sweep", files=parts, data={"name": "dte sweep"})
        check(r.status_code == 409 and r.json().get("existing_id") == sid, "the same name again: 409 with its id")
        r = await c.post("/api/oo-backtest/strategies/sweep", files=parts[:1], data={"name": "one"})
        check(r.status_code == 400 and "two or more files" in r.json().get("detail", ""),
              "one file is not a sweep: refused, pointing at a normal save")
        bad = parts + [("files", ("sw_dte7_sl100.csv", _gz.compress(files["sw_dte7_pt50.csv"])))]
        r = await c.post("/api/oo-backtest/strategies/sweep", files=bad, data={"name": "bad"})
        check(r.status_code == 400 and "sw_dte7_sl100.csv" in r.json().get("detail", ""),
              "a batch that is not one sweep is refused as the upload refuses it, naming the file")
        lst = (await c.get("/api/oo-backtest/strategies")).json()["strategies"]
        check(any(s["id"] == sid and s["variants"] == 3 for s in lst), "the list carries the variant count")

        # Load: the same payload as uploading the same files, ALL, saved meta.
        r = await c.get(f"/api/oo-backtest/strategies/{sid}/load")
        got = r.json()
        up = (await c.post("/api/oo-backtest/parse-batch", files=parts)).json()
        strip = lambda p: {k: v for k, v in p.items() if k not in ("saved", "parse", "suggested_name", "sweep", "market")}  # noqa: E731
        check(r.status_code == 200 and r.headers.get("content-encoding") == "gzip" and strip(got) == strip(up)
              and got["sweep"]["variants"] == up["sweep"]["variants"],
              "loading it back gives exactly the upload's trades and variants")
        check(got["saved"]["id"] == sid and got["saved"]["variants"] == 3 and "saved_variants_changed" not in got
              and got["suggested_name"] == "dte sweep", "with its saved record, and nothing said to have changed")

        # The recorded list no longer matches what the files resolve to.
        async with pool.acquire() as conn:
            rec = _json.loads(await conn.fetchval("SELECT sweep FROM oo_backtest_strategies WHERE id = $1", sid))
            rec["variants"][0]["label"] = "dte=7 · pt=99"
            await conn.execute("UPDATE oo_backtest_strategies SET sweep = $2::jsonb WHERE id = $1", sid, _json.dumps(rec))
        got = (await c.get(f"/api/oo-backtest/strategies/{sid}/load")).json()
        ch = got.get("saved_variants_changed") or {}
        check("dte=7 · pt=99 (2)" in ch.get("when_saved", []) and "dte=7 · pt=25 (2)" in ch.get("now", []),
              "a variant list that no longer matches the files is said, then and now")

        # Corruption: a changed file, a missing file.
        async with pool.acquire() as conn:
            await conn.execute("UPDATE oo_backtest_strategy_files SET file_gz = $3 WHERE strategy_id = $1 AND ord = $2",
                               sid, 1, _gz.compress(b"changed"))
        r = await c.get(f"/api/oo-backtest/strategies/{sid}/load")
        check(r.status_code == 500 and "sw_dte7_pt50.csv no longer matches its hash" in r.json().get("detail", ""),
              "a stored file that changed is refused, naming it")
        async with pool.acquire() as conn:
            await conn.execute("UPDATE oo_backtest_strategy_files SET file_gz = $3 WHERE strategy_id = $1 AND ord = $2",
                               sid, 1, _gz.compress(files["sw_dte7_pt50.csv"]))
            await conn.execute("DELETE FROM oo_backtest_strategy_files WHERE strategy_id = $1 AND ord = 2", sid)
        r = await c.get(f"/api/oo-backtest/strategies/{sid}/load")
        check(r.status_code == 500 and "not the ones it was saved with" in r.json().get("detail", ""),
              "a sweep missing a file is refused, not analysed smaller")

        # Replace with a full sweep again, then the portfolio's refusal.
        r = await c.post("/api/oo-backtest/strategies/sweep", files=parts, data={"name": "dte sweep", "replace": "true"})
        check(r.status_code == 200 and r.json()["strategy"]["id"] == sid, "replace=true re-saves the sweep under its id")
        r = await c.post("/api/backtest-portfolio/load", json={"ids": [sid]})
        det = r.json().get("detail", "")
        check(r.status_code == 400 and '"dte sweep" is a parameter sweep (3 variants)' in det
              and "re-export that variant from Option Omega (or re-run it through the MCP API) and save it as a "
                  "normal strategy" in det,
              "the portfolio refuses a sweep and says how to use one variant")

        # A single file saved over the name takes the sweep's files with it.
        r = await c.post("/api/oo-backtest/strategies", files={"file": ("one.csv", files["sw_dte7_pt25.csv"])},
                         data={"name": "dte sweep", "replace": "true"})
        async with pool.acquire() as conn:
            left = await conn.fetchval("SELECT count(*) FROM oo_backtest_strategy_files WHERE strategy_id = $1", sid)
        lst = {s["id"]: s for s in (await c.get("/api/oo-backtest/strategies")).json()["strategies"]}
        check(r.status_code == 200 and left == 0 and lst[sid]["variants"] is None and lst[sid]["trade_count"] == 2,
              "a single file saved over the sweep's name: a normal strategy, its files gone")
        r = await c.get(f"/api/oo-backtest/strategies/{sid}/load")
        check(r.status_code == 200 and r.json().get("sweep") is None and r.json()["n"] == 2, "and it loads as one log")

        # Delete takes the files with it.
        r = await c.post("/api/oo-backtest/strategies/sweep", files=parts, data={"name": "to delete"})
        did = r.json()["strategy"]["id"]
        await c.delete(f"/api/oo-backtest/strategies/{did}")
        async with pool.acquire() as conn:
            left = await conn.fetchval("SELECT count(*) FROM oo_backtest_strategy_files WHERE strategy_id = $1", did)
        check(left == 0, "deleting a sweep deletes its files")
        await c.delete(f"/api/oo-backtest/strategies/{sid}")


async def check_saved_strategies(pool) -> None:
    """Saved strategies: the stored record is the original file, byte for byte,
    and a load goes through the same parse + join as an upload."""
    print("saved strategies")
    import gzip
    import hashlib
    from app.oo_backtest import store
    from app.routers.oo_backtest import _parse_df, _payload

    raw = (ROOT / "scripts" / "fixtures" / "mesosim" / "v3_1_allantis_v2_mon.json").read_bytes()
    raw2 = (ROOT / "scripts" / "fixtures" / "mesosim" / "v2_13_allantis_weekly.json").read_bytes()
    common = dict(source="mesosim_json", filename="v3.events.json", date_min=date(2021, 1, 4), date_max=date(2023, 12, 12))

    a = await store.save_strategy(pool, name="  allantis   v2 ", notes="first", content=raw, trade_count=4,
                                  replace=False, **common)
    check(a["name"] == "allantis v2" and a["trade_count"] == 4 and a["same_file_as"] == [],
          f"saved, name whitespace-normalised ({a['name']!r})")
    try:
        await store.save_strategy(pool, name="allantis v2", notes="dup", content=raw2, trade_count=5,
                                  replace=False, **common)
        check(False, "saving over an existing name without replace is refused")
    except store.NameTaken as exc:
        check(exc.existing_id == a["id"], "saving over an existing name without replace is refused (NameTaken, id given)")

    b = await store.save_strategy(pool, name="allantis v2", notes="replaced", content=raw2, trade_count=5,
                                  replace=True, **common)
    check(b["id"] == a["id"] and b["trade_count"] == 5 and b["notes"] == "replaced" and b["updated_at"] >= a["updated_at"],
          "replace overwrites in place: same id, new file facts and notes")

    c = await store.save_strategy(pool, name="copy of weekly", notes="", content=raw2, trade_count=5,
                                  replace=False, **common)
    check(c["same_file_as"] == ["allantis v2"], f"the same file under another name is flagged ({c['same_file_as']})")

    lst = await store.list_strategies(pool)
    check([s["name"] for s in lst] == ["copy of weekly", "allantis v2"] and all("file_gz" not in s for s in lst),
          "list is newest-saved first and carries no file content")

    meta, content = await store.load_strategy_file(pool, b["id"])
    check(content == raw2 and hashlib.sha256(content).hexdigest() == meta["file_sha256"],
          "load returns the original bytes exactly")

    # Round trip: a stored file through the load path equals a direct parse.
    await store.save_strategy(pool, name="v3 fixture", notes="", content=raw, trade_count=4, replace=False, **common)
    v3 = next(s for s in await store.list_strategies(pool) if s["name"] == "v3 fixture")
    _, stored = await store.load_strategy_file(pool, v3["id"])
    j1, r1 = await market.join_market(pool, _parse_df(stored, "v3.events.json"))
    j2, r2 = await market.join_market(pool, _parse_df(raw, "v3.events.json"))
    p1, p2 = _payload(j1, "v3.events.json", r1), _payload(j2, "v3.events.json", r2)
    check(p1["columns"] == p2["columns"] and p1["notes"] == p2["notes"],
          "a saved strategy loads to the same trades and notes as uploading the file")

    # Planted corruption: the stored bytes no longer match the stored hash.
    async with pool.acquire() as conn:
        await conn.execute(f"UPDATE {store.TABLE} SET file_gz = $1 WHERE id = $2", gzip.compress(b"[]"), c["id"])
    try:
        await store.load_strategy_file(pool, c["id"])
        check(False, "a stored file that no longer matches its hash is refused")
    except ValueError as exc:
        check("corrupt" in str(exc), "a stored file that no longer matches its hash is refused")

    for bad, why in (("   ", "blank"), ("x" * (store.NAME_MAX + 1), "too long")):
        try:
            await store.save_strategy(pool, name=bad, notes="", content=raw, trade_count=1, replace=False, **common)
            check(False, f"a {why} name is refused")
        except ValueError:
            check(True, f"a {why} name is refused")

    # Capital per position: saved with the file, read back, changed alone.
    cap = await store.save_strategy(pool, name="with capital", notes="", content=raw, trade_count=4, replace=False,
                                    capital_per_position="12500", **common)
    check(cap["capital_per_position"] == 12500.0, f"capital per position is stored on save ({cap['capital_per_position']})")
    no_cap = await store.save_strategy(pool, name="no capital", notes="", content=raw, trade_count=4, replace=False,
                                       capital_per_position="", **common)
    check(no_cap["capital_per_position"] is None, "a blank capital is stored as NULL (the page default)")
    moved = await store.set_capital(pool, cap["id"], 7500)
    meta_c, _ = await store.load_strategy_file(pool, cap["id"])
    check(moved["capital_per_position"] == 7500.0 and meta_c["capital_per_position"] == 7500.0
          and moved["updated_at"] == cap["updated_at"],
          "set_capital changes only the capital; a load returns it; updated_at (the list order) is untouched")
    for bad in (0, -5, "abc"):
        try:
            await store.set_capital(pool, cap["id"], bad)
            check(False, f"capital {bad!r} is refused")
        except ValueError:
            check(True, f"capital {bad!r} is refused")
    check(await store.set_capital(pool, 999999, 100) is None, "set_capital on a missing id returns None (404)")

    # Notes: edited after the save, alone.
    renoted = await store.set_notes(pool, cap["id"], "  re-worded\nsecond line  ")
    meta_n, raw_n = await store.load_strategy_file(pool, cap["id"])
    check(renoted["notes"] == "re-worded\nsecond line" and meta_n["notes"] == renoted["notes"]
          and meta_n["capital_per_position"] == 7500.0 and raw_n == raw
          and renoted["updated_at"] == cap["updated_at"],
          "set_notes changes only the notes (trimmed); file, capital and updated_at untouched")
    check((await store.set_notes(pool, cap["id"], ""))["notes"] == "", "notes can be cleared")
    for bad, why in (("x" * (store.NOTES_MAX + 1), "too long"), (123, "non-text")):
        try:
            await store.set_notes(pool, cap["id"], bad)
            check(False, f"{why} notes are refused")
        except ValueError:
            check(True, f"{why} notes are refused")
    check(await store.set_notes(pool, 999999, "x") is None, "set_notes on a missing id returns None (404)")

    # Migration: a table created before the column existed gains it on first use.
    # (A table from before sweeps has no files table either: drop that first.)
    async with pool.acquire() as conn:
        await conn.execute(f"DROP TABLE IF EXISTS {store.FILES_TABLE}")
        await conn.execute(f"DROP TABLE {store.TABLE}")
        await conn.execute(store.CREATE_SQL.replace("    capital_per_position DOUBLE PRECISION,\n", ""))
        before = await conn.fetchval("SELECT count(*) FROM information_schema.columns WHERE table_name = $1 "
                                     "AND column_name = 'capital_per_position'", store.TABLE)
    await store.list_strategies(pool)
    async with pool.acquire() as conn:
        after = await conn.fetchval("SELECT count(*) FROM information_schema.columns WHERE table_name = $1 "
                                    "AND column_name = 'capital_per_position'", store.TABLE)
        sweep_col = await conn.fetchval("SELECT count(*) FROM information_schema.columns WHERE table_name = $1 "
                                        "AND column_name = 'sweep'", store.TABLE)
        files_tbl = await conn.fetchval("SELECT to_regclass($1) IS NOT NULL", store.FILES_TABLE)
    check(sweep_col == 1 and files_tbl, "a pre-sweep table gains the sweep column and the files table on first use")
    check(before == 0 and after == 1, f"an existing table without the column gains it (before {before}, after {after})")
    a = await store.save_strategy(pool, name="allantis v2", notes="", content=raw, trade_count=4, replace=False, **common)

    check(await store.delete_strategy(pool, a["id"]) is True, "delete removes the row")
    check(await store.delete_strategy(pool, a["id"]) is False and await store.load_strategy_file(pool, a["id"]) is None,
          "a deleted id deletes nothing twice and loads as missing")


async def check_portfolio_profiles(pool) -> None:
    """Saved portfolio profiles: a combination, validated on the way in.

    A profile POINTS AT saved strategies and carries the numbers applied to
    them. What matters here is that what comes out is what a page can use --
    a store whose rows have to be re-validated by every reader is not a
    store -- and that a name collision is an answer rather than an
    overwrite.
    """
    print("portfolio profiles")
    from app.oo_backtest import portfolio_store as ps

    body = {
        "strategies": [
            {"id": 1, "qty": 2, "capital": 25000,
             "filters": {"vix": {"on": True, "lo": 12, "hi": 30},
                         "day_of_week": {"on": True, "allowed": [0, 4]}}},
            {"id": 2, "qty": 1, "capital": 10000, "filters": {}},
        ],
        "range_mode": "intersection",
        "roll_weeks": 13,
    }
    a = await ps.save_profile(pool, name="  live   book ", notes="two",
                              payload=body)
    check(a["name"] == "live book" and a["n_strategies"] == 2,
          f"the name was not collapsed, or the count is wrong: {a}")

    # A NAME COLLISION IS AN ANSWER. Two portfolios called "live book" where
    # one silently replaced the other is a combination nobody can reproduce.
    try:
        await ps.save_profile(pool, name="live book", notes="dup", payload=body)
        check(False, "a duplicate profile name was accepted silently")
    except ps.NameTaken as exc:
        check(exc.existing_id == a["id"],
              f"the collision named id {exc.existing_id}, not {a['id']} -- "
              f"the page offers to replace THAT one")

    b = await ps.save_profile(pool, name="live book", notes="replaced",
                              payload=body, replace=True)
    check(b["id"] == a["id"] and b["notes"] == "replaced",
          f"replace made a second row: {b['id']} vs {a['id']}")

    got = await ps.load_profile(pool, a["id"])
    check(got["payload"] == ps.clean_payload(body),
          f"the payload changed through the store: {got['payload']}")
    check(got["payload"]["strategies"][0]["filters"]["day_of_week"]["allowed"]
          == [0, 4],
          "a categorical filter's chosen values did not survive")
    check(got["payload"]["range_mode"] == "intersection"
          and got["payload"]["roll_weeks"] == 13,
          f"the page settings did not survive: {got['payload']}")

    # VALIDATED ON THE WAY IN, with a reason.
    bad = [
        ({"strategies": []}, "empty"),
        ({"strategies": [{"id": 0}]}, "a zero id"),
        ({"strategies": [{"id": 1}, {"id": 1}]}, "the same strategy twice"),
        ({"strategies": [{"id": 1, "qty": 0}]}, "a zero quantity"),
        ({"strategies": [{"id": 1, "capital": -5}]}, "negative capital"),
        ({"strategies": [{"id": 1}], "range_mode": "sideways"}, "a bad range mode"),
        ({"strategies": [{"id": 1, "filters": {"vix": {"lo": "wide"}}}]},
         "a non-numeric bound"),
    ]
    for payload, what in bad:
        try:
            await ps.save_profile(pool, name=f"bad {what}", notes="",
                                  payload=payload)
            check(False, f"{what} was accepted into a profile")
        except ValueError:
            pass

    # The filter map holds NO METRIC NAMES of its own: a key the registry no
    # longer has is stored as given, because the registry owns that list and
    # a second copy here would drift from it.
    odd = await ps.save_profile(
        pool, name="future metric", notes="",
        payload={"strategies": [{"id": 3, "qty": 1, "capital": 1,
                                 "filters": {"not_a_metric_yet":
                                             {"on": False, "lo": 0, "hi": 1}}}]})
    back = await ps.load_profile(pool, odd["id"])
    check("not_a_metric_yet" in back["payload"]["strategies"][0]["filters"],
          "an unknown metric key was dropped; the registry decides what "
          "exists, not this table")

    # ── THE ADDED SURFACE METRICS TRAVEL WITH THE PROFILE ────────────
    # The filter VALUE already survived; without the list beside it the
    # value comes back describing nothing, because bpSpecs walks the
    # registry and not the filter map.
    surf = await ps.save_profile(
        pool, name="with surface", notes="",
        payload={"strategies": [
            {"id": 1, "qty": 1, "capital": 1000,
             "surface": ["iv_30d_atm", "z_iv_30d_atm", "iv_30d_atm"],
             "filters": {"surface__iv_30d_atm": {"on": True, "lo": 10.5, "hi": 13.0}}},
            {"id": 2, "qty": 1, "capital": 1000}]})
    back = await ps.load_profile(pool, surf["id"])
    a, b = back["payload"]["strategies"]
    check(a["surface"] == ["iv_30d_atm", "z_iv_30d_atm"],
          f"the metric list survives, duplicates collapsed ({a['surface']})")
    check(b["surface"] == [],
          "a strategy with none gets an empty list, not a missing key")
    check(a["filters"]["surface__iv_30d_atm"] == {"on": True, "lo": 10.5, "hi": 13.0},
          "and its filter keeps the bounds it was saved with")
    # SHAPE ONLY -- the catalog owns which columns exist, and a second list
    # here would drift from it. What is rejected is what cannot be a column.
    for bad, what in [(["../etc/passwd"], "a path"),
                      (["a" * 201], "an absurd name"),
                      ([f"m{i}" for i in range(25)], "more than the cap"),
                      ("iv_30d_atm", "a bare string instead of a list"),
                      ([123], "a number")]:
        try:
            await ps.save_profile(pool, name=f"bad surf {what}", notes="",
                                  payload={"strategies": [{"id": 1, "surface": bad}]})
            check(False, f"{what} was accepted as a surface metric")
        except ValueError:
            pass
    # A column the catalog does not have is NOT rejected here: this table has
    # no business holding a second copy of the catalog, and /surface/values
    # refuses it at load time where the catalog actually is.
    okd = await ps.save_profile(pool, name="future metric col", notes="",
                                payload={"strategies": [{"id": 1, "surface": ["not_a_column_yet"]}]})
    check((await ps.load_profile(pool, okd["id"]))["payload"]["strategies"][0]["surface"]
          == ["not_a_column_yet"],
          "an unknown column is stored as given, for the catalog to refuse later")
    await ps.delete_profile(pool, surf["id"])
    await ps.delete_profile(pool, okd["id"])

    lst = await ps.list_profiles(pool)
    check([x["name"] for x in lst][:1] == ["future metric"],
          f"profiles are not newest-updated first: {[x['name'] for x in lst]}")
    check(await ps.delete_profile(pool, odd["id"]) is True
          and await ps.delete_profile(pool, odd["id"]) is False,
          "delete did not report whether it removed anything")


async def check_end_to_end(pool) -> None:
    """The route's own sequence: parse a REAL MesoSim fixture, join, build the
    payload. Three of its four trades fall outside the fabricated sessions and
    must get null levels, counted. The fourth is the spec's early-close fixture
    -- position 106, entered 2023-07-03 12:30 -- which is deliberately the
    fabricated early-close date, so it must land on that session's 12:30 bar
    through the real parser and the real SQL together."""
    print("end to end: real MesoSim fixture -> parse -> join -> payload")
    from app.oo_backtest.payload import allowed_column
    from app.routers.oo_backtest import _parse_df, _payload
    raw = (ROOT / "scripts" / "fixtures" / "mesosim" / "v3_1_allantis_v2_mon.json").read_bytes()
    df = _parse_df(raw, "v3.events.json")
    joined, rep = await market.join_market(pool, df)
    p = _payload(joined, "v3.events.json", rep)
    check(p["notes"].get("open_positions") == 2, "parse notes survive the market join")
    check(all(allowed_column(c) for c in p["columns"]), "payload after the join carries only whitelisted columns")
    e = rep["entry_time"]
    check(e["entry_time_found"] == e["trades"] == 4 and e["entry_time_fallback_0930"] == 0,
          f"MesoSim entry time found for every trade ({e['entry_time_found']} of {e['trades']})")
    c = p["columns"]
    i106 = c["position_id"].index(106)
    others = [v for i, v in enumerate(c["vix_level"]) if i != i106]
    check(rep["trades_without_daily_row"] == 3 and all(v is None for v in others),
          "trades outside the table's sessions are counted (3) and get null levels, not a neighbour's")
    check(c["vix_level"][i106] == bar_open("vix", EARLY, time(12, 30)) and c["vix_bar_time"][i106] == "12:30:00",
          "position 106 (2023-07-03 12:30, parsed from real MesoSim events) lands on the early-close 12:30 bar")
    # The Deployment axis: the rollup's own SPX sessions inside the log's span,
    # read through the real SQL -- the VIX-only 07-11 and artifact holiday 07-07
    # must not be days.
    # THE DEPLOYMENT AXIS IS A MARKET FACT, not a fact about our table. It
    # comes from the calendar, so it covers the log's span whatever
    # index_ohlc holds -- including 07-07 and 07-11, the two sessions this
    # fixture deliberately under-fills, because a position was held over
    # them either way.
    from app.oo_backtest import market_calendar as mc
    want_days = mc.sessions(min(c["date_opened"]), max(c["date_closed"]))
    check(rep["spx_sessions"] == want_days and not rep["diagnostic_errors"],
          f"the payload's spx_sessions are the EXCHANGE sessions in the log's span "
          f"({len(rep['spx_sessions'])} days vs {len(want_days)})")
    check("2023-07-11" in rep["spx_sessions"] and "2023-07-07" in rep["spx_sessions"],
          "the two sessions with no SPX data are still days on the axis")
    check("2023-07-04" not in rep["spx_sessions"],
          "but the holiday is not, however many artifact bars it carries")

    # A SWEEP goes through the same join once, over every file's trades: each
    # trade must get exactly the levels its own file gets loaded alone.
    print("end to end: a two-file sweep -> one join -> payload")
    from app.routers.oo_backtest import analyze_batch, _analyze
    import io as _io
    hdr = ["Date Opened", "Time Opened", "Date Closed", "Time Closed", "P/L", "Legs", "Premium"]
    def oo(rows):
        out = _io.StringIO()
        pd.DataFrame(rows, columns=hdr).to_csv(out, index=False)
        return out.getvalue().encode()
    a = oo([["2023-07-03", "12:30:00", "2023-07-05", "15:59:00", 100, "x", 1],
            ["2023-07-05", "10:00:00", "2023-07-06", "15:59:00", -40, "x", 1]])
    b = oo([["2023-07-06", "10:00:00", "2023-07-10", "15:59:00", 60, "x", 1],
            ["2023-07-03", "12:30:00", "2023-07-06", "15:59:00", 20, "x", 1]])
    sw = await analyze_batch([("sw_pt50.csv", a), ("sw_pt25.csv", b)], pool)
    alone = {n: await _analyze(c, n, pool) for n, c in (("sw_pt50.csv", a), ("sw_pt25.csv", b))}
    sc = sw["columns"]
    check(sw["market"]["joined"] and sw["n"] == 4 and [v["label"] for v in sw["sweep"]["variants"]] == ["pt=25", "pt=50"],
          "the sweep joined, 4 trades, variants in numeric order")
    lv = ("vix_level", "vix3m_level", "vix9d_level", "gap", "vix_bar_time", "vix3m_vix_ratio_entry")
    ok = True
    for vid, name in ((0, "sw_pt25.csv"), (1, "sw_pt50.csv")):
        ac = alone[name]["columns"]
        mine = [i for i, v in enumerate(sc["variant"]) if v == vid]
        got = sorted(tuple(sc[k][i] for k in ("date_opened", "pnl") + lv) for i in mine)
        want = sorted(tuple(ac[k][i] for k in ("date_opened", "pnl") + lv) for i in range(alone[name]["n"]))
        ok = ok and got == want
    check(ok, "every sweep trade carries exactly the market values its own file gets alone")
    check(sc["vix_level"][sc["date_opened"].index("2023-07-03")] == bar_open("vix", EARLY, time(12, 30)),
          "the early-close 12:30 entry still lands on its own bar inside a sweep")


# ── surface metrics ─────────────────────────────────────────────────────────
#
# Shaped like public.surface_metrics_core as verified on the VPS: rows only on
# sessions (a missing day has NO rows, not zero rows), 78 bars 09:35..16:00,
# DOUBLE PRECISION metrics, NULL before a metric's coverage and never after.
# Each iv value encodes its own bar -- day index * 10000 + minutes since
# midnight -- so a join that picks the wrong bar picks the wrong number.
#
#   S1 2024-03-04  iv, spot, forward present; chg_d, z, vrp NULL
#   S2 2024-03-05  + chg_d
#   (2024-03-06 absent: no rows at all)
#   S3 2024-03-07  + z
#   S4 2024-03-08  + vrp_3m
# Catalog quirks: meta and spot/forward rows (excluded), ghost_col (catalogued,
# not in the table), label_col (in the table as TEXT), uncat_col (in the
# table, not catalogued).

S_DAYS = [date(2024, 3, 4), date(2024, 3, 5), date(2024, 3, 7), date(2024, 3, 8)]
S_ABSENT = date(2024, 3, 6)
S_BACKFILL = date(2024, 3, 1)
S_COLS = ["spot", "forward_30d", "iv_30d_atm", "chg_d_iv_30d_atm", "z_iv_30d_atm", "vrp_3m"]
S_START = {"spot": 0, "forward_30d": 0, "iv_30d_atm": 0, "chg_d_iv_30d_atm": 1, "z_iv_30d_atm": 2, "vrp_3m": 3}


def s_bars():
    t = datetime.combine(date(2000, 1, 1), time(9, 35))
    while t.time() <= time(16, 0):
        yield t.time()
        t += timedelta(minutes=5)


def s_code(day_idx: int, t: time) -> float:
    return float(day_idx * 10000 + t.hour * 60 + t.minute)


async def load_surface(pool) -> None:
    async with pool.acquire() as conn:
        await conn.execute("""CREATE TABLE surface_metrics_catalog (
            column_name TEXT PRIMARY KEY, family TEXT, tenor TEXT, wing TEXT, form TEXT,
            base_column TEXT, units TEXT, description TEXT, formula TEXT)""")
        cat = [("trade_date", "meta", None, None, "level", None, "date", "Trading date", None),
               ("quote_time", "meta", None, None, "level", None, "time", "Quote time", None),
               ("spot", "spot", None, None, "level", None, "price", "Spot", None),
               ("forward_30d", "forward", "30d", None, "level", None, "price", "Forward", None),
               ("iv_30d_atm", "iv", "30d", "atm", "level", None, "vol_decimal", "IV 30d ATM", "sigma"),
               ("chg_d_iv_30d_atm", "iv", "30d", "atm", "chg_d", "iv_30d_atm", "vol_decimal", "1d change", "d"),
               ("z_iv_30d_atm", "iv", "30d", "atm", "z", "iv_30d_atm", "z_score", "z", "z"),
               ("vrp_3m", "vrp", "3m", None, "level", None, "vol_decimal", "VRP 3m", "v"),
               ("ghost_col", "iv", "7d", "atm", "level", None, "vol_decimal", "not in the table", None),
               ("label_col", "iv", "7d", "atm", "level", None, "vol_decimal", "text in the table", None)]
        await conn.executemany("INSERT INTO surface_metrics_catalog VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9)", cat)
        cols = ", ".join(f"{c} DOUBLE PRECISION" for c in S_COLS)
        await conn.execute(f"""CREATE TABLE surface_metrics_core (
            trade_date DATE NOT NULL, quote_time TIME NOT NULL, {cols}, label_col TEXT, uncat_col DOUBLE PRECISION,
            PRIMARY KEY (trade_date, quote_time))""")
        await conn.execute("CREATE INDEX ON surface_metrics_core (trade_date)")
        await conn.copy_records_to_table("surface_metrics_core", records=list(s_records(S_DAYS, 1)),
                                         columns=["trade_date", "quote_time", *S_COLS, "label_col", "uncat_col"])
        n = await conn.fetchval("SELECT count(*) FROM surface_metrics_core")
        assert n == 78 * len(S_DAYS), n
        assert await conn.fetchval("SELECT count(*) FROM surface_metrics_core WHERE trade_date = $1", S_ABSENT) == 0


def s_records(days, first_idx):
    for k, d in enumerate(days):
        idx = first_idx + k
        for t in s_bars():
            code = s_code(idx, t)
            vals = [code if k >= S_START[c] else None for c in S_COLS]
            yield (d, t, *vals, "x", 1.0)


async def check_surface(pool) -> None:
    """The surface SQL through a real Postgres: catalog + coverage (and its
    rebuild on a backfill), the entry bar under both rules, no reach-back to
    the prior session, and the rank/values routes."""
    print("surface metrics: catalog, coverage, entry bar, routes")
    from fastapi import HTTPException
    from app.oo_backtest import surface
    from app.routers import oo_backtest as routes

    await load_surface(pool)
    cat = await surface.get_catalog(pool)
    by = {m["column_name"]: m for m in cat["metrics"]}
    rep = cat["report"]
    check(sorted(by) == ["chg_d_iv_30d_atm", "iv_30d_atm", "vrp_3m", "z_iv_30d_atm"],
          f"ranked: the four metrics; meta, spot and forward excluded ({sorted(by)})")
    check(rep["missing_from_table"] == ["ghost_col"] and rep["wrong_type"] == ["label_col"]
          and rep["uncatalogued"] == ["uncat_col"] and rep["excluded"] == 4,
          f"catalog/table disagreements are reported, not guessed around ({rep})")
    want = {"iv_30d_atm": "2024-03-04", "chg_d_iv_30d_atm": "2024-03-05", "z_iv_30d_atm": "2024-03-07", "vrp_3m": "2024-03-08"}
    check({c: by[c]["min_date"] for c in want} == want,
          f"min_date per metric is its first non-null date in the table ({ {c: by[c]['min_date'] for c in want} })")
    built = cat["built_at"]
    again = await surface.get_catalog(pool)
    check(again["built_at"] == built, "an unchanged table is served from the cache")

    async with pool.acquire() as conn:     # backfill: an earlier session with levels only
        await conn.executemany("INSERT INTO surface_metrics_core VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10)",
                               [(S_BACKFILL, t, s_code(0, t), s_code(0, t), s_code(0, t), None, None, None, "x", 1.0)
                                for t in s_bars()])
    bf = {m["column_name"]: m["min_date"] for m in (await surface.get_catalog(pool))["metrics"]}
    check(bf["iv_30d_atm"] == "2024-03-01" and bf["z_iv_30d_atm"] == "2024-03-07",
          f"a backfill moves the first date and rebuilds coverage: iv now {bf['iv_30d_atm']}, z unchanged {bf['z_iv_30d_atm']}")

    S2, S3 = S_DAYS[1], S_DAYS[2]
    trades = [(S2, time(10, 2)), (S2, time(9, 30)), (S2, time(9, 35)), (S2, time(16, 20)),
              (S_ABSENT, time(12, 0)), (S_DAYS[0], time(12, 0)), (S2, time(10, 2)), (None, None), (S3, time(15, 30))]
    cols = ["iv_30d_atm", "z_iv_30d_atm"]
    at, at_rep = await surface.entry_values(pool, trades, cols)
    iv = lambda rows, i: rows[i]["iv_30d_atm"]   # noqa: E731
    check(iv(at, 0) == s_code(2, time(10, 0)), "10:02 entry reads its own 10:00 bar")
    check(iv(at, 1) is None and at[1]["bar_time"] is None,
          "09:30 entry: no bar on the entry date at or before it -> null; the prior session's 16:00 is NOT used")
    check(iv(at, 2) == s_code(2, time(9, 35)), "09:35 entry reads the 09:35 bar")
    check(iv(at, 3) == s_code(2, time(16, 0)), "16:20 entry reads the 16:00 bar")
    check(iv(at, 4) is None and iv(at, 7) is None, "a date with no rows, and a row with no date/time, get null")
    check(iv(at, 5) == s_code(1, time(12, 0)) and at[5]["z_iv_30d_atm"] is None,
          "before a metric's coverage its value is null while an earlier-starting metric on the same bar has one")
    check(iv(at, 6) == iv(at, 0) and at_rep["distinct_entries"] == 7 and at_rep["no_bar"] == 3
          and at_rep["with_bar"] == 6,
          f"a repeated entry is looked up once and filled for both ({at_rep['distinct_entries']} distinct, {at_rep['no_bar']} with no bar)")
    check(at[8]["z_iv_30d_atm"] == s_code(3, time(15, 30)), "15:30 entry on a z-covered day reads z from the 15:30 bar")

    # Routes, called with the pool directly.
    body = {"trades": [[d.isoformat(), t.strftime("%H:%M:%S"), float(k)] for d in S_DAYS
                       for k, t in enumerate([time(10, 0), time(11, 0), time(12, 0), time(13, 0)])]}
    r = await routes.surface_rank(body, pool=pool)
    rows = {x["column"]: x for x in r["rows"]}
    check(sorted(rows) == sorted(want) and rows["iv_30d_atm"]["n"] == 16 and rows["z_iv_30d_atm"]["n"] == 8
          and rows["vrp_3m"]["n"] == 4 and rows["chg_d_iv_30d_atm"]["n"] == 12,
          f"rank: n per metric follows coverage (iv 16, chg_d 12, z 8, vrp 4)")
    check(all(x["pearson_p_bh"] is not None and x["spearman_p_bh"] is not None for x in r["rows"])
          and rows["iv_30d_atm"]["bars"] == 16 and r["lookahead_confirmed"] is True
          and r["bar_rule"] == "at_or_before_entry" and r["report"]["with_bar"] == 16,
          "rank: both BH p-values on every metric; distinct bars; trades with a bar reported; bar rule stated")
    for bad_body, why in (({"trades": body["trades"], "bar_rule": "previous_bar"}, "the removed previous_bar rule"),
                          ({"trades": "nope"}, "trades that are not a list")):
        try:
            await routes.surface_rank(bad_body, pool=pool)
            check(False, f"rank refuses {why}")
        except HTTPException as exc:
            check(exc.status_code == 400, f"rank refuses {why} (400)")
    v = await routes.surface_values({"column": "z_iv_30d_atm", "trades": [[S3.isoformat(), "15:30"], [S2.isoformat(), "15:30"]]},
                                    pool=pool)
    check(v["values"] == [s_code(3, time(15, 30)), None] and v["bar_times"] == ["15:30:00", "15:30:00"],
          "values: aligned to the trades sent; null before coverage even where the bar exists")
    for col in ("spot", "ghost_col", 'iv_30d_atm"; DROP TABLE surface_metrics_core; --'):
        try:
            await routes.surface_values({"column": col, "trades": [[S3.isoformat(), "15:30"]]}, pool=pool)
            check(False, f"values refuses column {col!r}")
        except HTTPException as exc:
            check(exc.status_code == 400, f"values refuses column {col[:30]!r} (400): only ranked metrics reach SQL")
    async with pool.acquire() as conn:
        still = await conn.fetchval("SELECT count(*) FROM surface_metrics_core")
    check(still == 78 * (len(S_DAYS) + 1), "the table is intact after the injection attempt")

    # A MID-RANGE insert -- the live-backfill shape that moves neither the
    # first nor the last date. The 2024-03-06 gap gets a session with vrp_3m,
    # which otherwise starts 2024-03-08: coverage must see it.
    before = {m["column_name"]: m["min_date"] for m in (await surface.get_catalog(pool))["metrics"]}
    async with pool.acquire() as conn:
        await conn.executemany("INSERT INTO surface_metrics_core VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10)",
                               [(S_ABSENT, t, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, "x", 1.0) for t in s_bars()])
    after_cat = await surface.get_catalog(pool)
    after = {m["column_name"]: m["min_date"] for m in after_cat["metrics"]}
    check(before["vrp_3m"] == "2024-03-08" and after["vrp_3m"] == "2024-03-06"
          and after_cat["first_date"] == "2024-03-01" and after_cat["last_date"] == "2024-03-08",
          f"a mid-range insert (dates unchanged) still rebuilds coverage: vrp_3m {before['vrp_3m']} -> {after['vrp_3m']}")


async def check_surface_sweep_scale(pool) -> None:
    """Both surface routes at SWEEP SCALE -- 90 variants, 288,540 trades -- with
    the request built by the page's own obEntryBars (the shipped JS, in node).
    The old cap, 20,000 TRADES, refused exactly this; every earlier check here
    ran below it, which is how that went unseen."""
    print("surface metrics at sweep scale: distinct bars, not trades")
    import json as _json
    import random as _random
    import subprocess as _sp

    from fastapi import HTTPException
    from app.oo_backtest import surface
    from app.routers import oo_backtest as routes

    # Every minute 09:35..16:00 on the fixture's sessions: the entry times a
    # sweep's variants share. Each variant: 3,206 trades over those times.
    minutes = [time(m // 60, m % 60) for m in range(9 * 60 + 35, 16 * 60 + 1)]
    keys = [(d.isoformat(), t.strftime("%H:%M:%S")) for d in S_DAYS for t in minutes]
    rng = _random.Random(7)

    def sweep(variants):
        rows = [(*keys[rng.randrange(len(keys))], round(rng.gauss(20, 300), 2)) for _ in range(variants * 3206)]
        return {"date_opened": [r[0] for r in rows], "time_opened": [r[1] for r in rows], "pnl": [r[2] for r in rows]}
    cols90, cols30 = sweep(90), sweep(30)

    core = (ROOT / "static" / "js" / "backtest_core.js").read_text(encoding="utf-8")
    driver = core + """
const fs = require('fs'); const job = JSON.parse(fs.readFileSync(0, 'utf8'));
const out = {};
for (const [k, cols] of Object.entries(job)) {
  const idx = cols.pnl.map((_, i) => i);
  const eb = obEntryBars(cols, idx);
  out[k] = { bars: eb.bars, bar_of: eb.barOf };
}
process.stdout.write(JSON.stringify(out));"""
    p = _sp.run(["node", "-e", driver], input=_json.dumps({"v90": cols90, "v30": cols30}), capture_output=True,
                text=True, encoding="utf-8", timeout=180)
    if p.returncode:
        check(False, f"the page's obEntryBars ran in node ({p.stderr.strip()[:300]})")
        return
    eb = _json.loads(p.stdout)
    n90, n30 = len(cols90["pnl"]), len(cols30["pnl"])
    b90, b30 = len(eb["v90"]["bars"]), len(eb["v30"]["bars"])
    check(n90 == 288_540 and b90 == len({(d, t) for d, t in zip(cols90["date_opened"], cols90["time_opened"])}) and b90 <= len(keys),
          f"90 variants: {n90:,} trades -> {b90:,} distinct entry bars sent (the page's obEntryBars)")
    check(b30 == b90, f"the bar count does not grow with the variants: 30 variants -> {b30:,}, 90 -> {b90:,}")
    body_bytes = len(_json.dumps({"column": "iv_30d_atm", "bars": eb["v90"]["bars"]}))
    old_bytes = len(_json.dumps({"column": "iv_30d_atm", "trades": [[d, t] for d, t in zip(cols90["date_opened"], cols90["time_opened"])]}))
    check(n90 > 20_000 and body_bytes < old_bytes / 50,
          f"/surface/values request: {body_bytes / 1e3:,.0f} KB as bars, against {old_bytes / 1e6:.1f} MB as trades "
          f"(which the old 20,000-trade cap refused)")

    # /surface/values: the page's bars, answered per bar, mapped back per trade.
    v = await routes.surface_values({"column": "iv_30d_atm", "bars": eb["v90"]["bars"]}, pool=pool)
    per_trade = [v["values"][k] for k in eb["v90"]["bar_of"]]
    sub = list(range(0, n90, 61))[:5000]
    legacy = await routes.surface_values({"column": "iv_30d_atm",
                                          "trades": [[cols90["date_opened"][i], cols90["time_opened"][i]] for i in sub]},
                                         pool=pool)
    check(len(v["values"]) == b90 and [per_trade[i] for i in sub] == legacy["values"],
          f"values: one per bar sent, and per trade identical to the per-trade request ({len(sub):,} compared)")

    # /surface/rank: the page's form at 288,540 trades == the old form on the same trades.
    rb = {"bars": eb["v90"]["bars"], "bar_of": eb["v90"]["bar_of"], "pnl": cols90["pnl"]}
    r_new = await routes.surface_rank(rb, pool=pool)
    r_old = await routes.surface_rank({"trades": [[d, t, x] for d, t, x in
                                                  zip(cols90["date_opened"], cols90["time_opened"], cols90["pnl"])]}, pool=pool)
    rep = r_new["report"]
    check(rep["trades"] == n90 and rep["bars"] == b90 and rep["with_bar"] + rep["no_bar"] == n90,
          f"rank: {n90:,} trades over {b90:,} bars accepted; with/without a bar counted per trade ({rep['with_bar']:,} / {rep['no_bar']:,})")
    check(r_new["rows"] == r_old["rows"], "rank: the bar form gives exactly the per-trade form's result, every metric")

    # The caps, worded for what they count.
    many = lambda n: [[(date(2000, 1, 1) + timedelta(days=i)).isoformat(), "10:00:00"] for i in range(n)]  # noqa: E731
    for call, body, want in (
        (routes.surface_values, {"column": "iv_30d_atm", "bars": many(surface.MAX_VALUE_BARS + 1)},
         f"at most {surface.MAX_VALUE_BARS:,} distinct entry bars (date, time) per request ({surface.MAX_VALUE_BARS + 1:,} sent)"),
        (routes.surface_rank, {"bars": many(surface.MAX_RANK_BARS + 1), "bar_of": [0], "pnl": [1.0]},
         f"at most {surface.MAX_RANK_BARS:,} distinct entry bars (date, time) per request ({surface.MAX_RANK_BARS + 1:,} sent)"),
        (routes.surface_rank, {"bars": [["2024-03-04", "10:00:00"]], "bar_of": [0] * (surface.MAX_RANK_TRADES + 1),
                               "pnl": [1.0] * (surface.MAX_RANK_TRADES + 1)},
         f"at most {surface.MAX_RANK_TRADES:,} trades per request ({surface.MAX_RANK_TRADES + 1:,} sent)"),
        (routes.surface_rank, {"bars": [["2024-03-04", "10:00:00"]], "bar_of": [3], "pnl": [1.0]},
         "bar_of holds 3, not an index into the 1 bars"),
    ):
        try:
            await call(body, pool=pool)
            check(False, f"refused: {want}")
        except HTTPException as exc:
            check(exc.status_code == 400 and exc.detail == want, f"refused (400): {exc.detail}")


def main() -> int:
    if os.name == "posix" and os.geteuid() == 0:
        print("SKIP: running as root — initdb refuses to create a cluster as root; run as an ordinary user")
        return EXIT_SKIPPED
    bindir = find_bin()
    if bindir is None:
        print("SKIP: no Postgres server binaries (initdb, pg_ctl) found — set PG_BIN to run this check")
        return EXIT_SKIPPED
    print(f"postgres binaries: {bindir}")
    try:
        import asyncpg  # noqa: F401
    except ImportError:
        print("SKIP: asyncpg not installed")
        return EXIT_SKIPPED
    with Cluster(bindir) as c:
        asyncio.run(run(c.dsn))
    print()
    if FAILS:
        print(f"FAIL: {len(FAILS)} market SQL check(s) failed")
        return 1
    print("PASS: market SQL — rollup, early close, prior row, entry bar, gaps, null reasons, saved strategies, surface metrics, planted faults")
    return 0


if __name__ == "__main__":
    sys.exit(main())
