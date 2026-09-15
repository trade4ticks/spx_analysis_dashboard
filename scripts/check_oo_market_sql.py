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
from datetime import date, time, timedelta
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
ZERO_HOLIDAY = [date(2023, 7, 4)]
ARTIFACT_HOLIDAY = date(2023, 7, 7)          # 25 valid VIX bars, nothing else
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
    finally:
        await pool.close()


async def check_daily(pool) -> None:
    print("daily rollup")
    daily, fresh = await market.get_daily(pool)
    by = {r["trade_date"]: r for r in daily.to_dict("records")}
    check(sorted(by) == DAYS,
          f"one row per day with a session in some series; zero-filled days and the VIX-artifact holiday are not rows ({len(by)})")
    check(ARTIFACT_HOLIDAY not in by, "the holiday with 25 VIX artifact bars is not a rollup row")
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
          and y["non_spx_session_days"] == len(ZERO_WEEKEND) + 3,   # + 07-04, 07-07, 07-11
          f"bar-label diagnostics count SPX sessions only {y}")
    z = market.sessions()
    check(z["zero_filled_days"] == len(ZERO_WEEKEND + ZERO_HOLIDAY) and z["zero_filled_weekday_dates"] == ["2023-07-04"],
          f"zero-filled days reported, weekdays named ({z['zero_filled_days']}, {z['zero_filled_weekday_dates']})")
    check(z["artifact_only_days"] == 1 and z["artifact_only"][0]["date"] == "2023-07-07" and z["artifact_only"][0]["vix"] == 25,
          f"the VIX-artifact holiday is reported as artifact-only ({z['artifact_only']})")
    sx, vx = z["by_series"]["spx"], z["by_series"]["vix"]
    check(sx["sessions"] == 7 and sx["shortest_kept"] == {"date": "2023-07-03", "bars": 42}
          and sx["missing_on_session_days"] == ["2023-07-11"],
          f"SPX: 7 sessions, shortest kept the 42-bar early close, missing on the VIX-only day ({sx})")
    check(vx["sessions"] == 8 and vx["longest_rejected"] == {"date": "2023-07-07", "bars": 25}
          and vx["shortest_kept"]["bars"] == 42,
          f"VIX: longest artifact rejected is 25 bars on 07-07; shortest session kept 42 ({vx['longest_rejected']}, {vx['shortest_kept']})")
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
    not_session = "entry date is not a session in index_ohlc (no series has a session that day)"
    check(nr["spx_gap"]["reasons"] == {"first SPX session in the table (no earlier SPX session)": 2,
                                        "no SPX session that day (0 valid SPX bars; 34 required)": 1,
                                        not_session: 1},
          f"null SPX gaps: first session x2, the VIX-only day names SPX, the artifact holiday is no session "
          f"({nr['spx_gap']['reasons']})")
    check(nr["vix_gap"]["reasons"] == {"first VIX session in the table (no earlier VIX session)": 2, not_session: 1},
          f"null VIX gaps: the VIX-only day is NOT among them ({nr['vix_gap']['reasons']})")
    check(nr["vix3m"]["reasons"] == {"entry date before VIX3M coverage (2023-06-30)": 2, "entry before 09:30": 1,
                                      "no VIX3M session that day (0 valid VIX3M bars; 34 required)": 1, not_session: 1},
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
    check(any_valid_prev == ARTIFACT_HOLIDAY and by[MON2]["spx_prev_date"] == THU and by[MON2]["vix_prev_date"] == THU,
          f"an 'any valid bar' rule takes the artifact holiday ({any_valid_prev}) as Monday's previous day; the session rule takes Thursday")
    check(raw_prev_row_close == 0.0 and by[EARLY]["spx_prev_date"] == DAYS[1],
          "a previous row over ALL dates gives Monday a close of 0 (the zero-filled Sunday); the rollup gives Friday")
    df, _ = await market.join_market(pool, trades())
    check(df.loc[6, "vix_level"] != bar_close, "the entry join is not reading the bar's close (5 min lookahead)")


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

    # Migration: a table created before the column existed gains it on first use.
    async with pool.acquire() as conn:
        await conn.execute(f"DROP TABLE {store.TABLE}")
        await conn.execute(store.CREATE_SQL.replace("    capital_per_position DOUBLE PRECISION,\n", ""))
        before = await conn.fetchval("SELECT count(*) FROM information_schema.columns WHERE table_name = $1 "
                                     "AND column_name = 'capital_per_position'", store.TABLE)
    await store.list_strategies(pool)
    async with pool.acquire() as conn:
        after = await conn.fetchval("SELECT count(*) FROM information_schema.columns WHERE table_name = $1 "
                                    "AND column_name = 'capital_per_position'", store.TABLE)
    check(before == 0 and after == 1, f"an existing table without the column gains it (before {before}, after {after})")
    a = await store.save_strategy(pool, name="allantis v2", notes="", content=raw, trade_count=4, replace=False, **common)

    check(await store.delete_strategy(pool, a["id"]) is True, "delete removes the row")
    check(await store.delete_strategy(pool, a["id"]) is False and await store.load_strategy_file(pool, a["id"]) is None,
          "a deleted id deletes nothing twice and loads as missing")


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
    daily, _ = await market.get_daily(pool)
    spx_days = sorted(d.isoformat() for d in daily.loc[daily["spx_session"].astype(bool), "trade_date"])
    span = [d for d in spx_days if min(c["date_opened"]) <= d <= max(c["date_closed"])]
    check(rep["spx_sessions"] == span and "2023-07-11" not in rep["spx_sessions"]
          and "2023-07-07" not in rep["spx_sessions"] and not rep["diagnostic_errors"],
          f"the payload's spx_sessions are the rollup's SPX sessions in the log's span ({rep['spx_sessions']})")


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
    print("PASS: market SQL — rollup, early close, prior row, entry bar, gaps, null reasons, saved strategies, planted faults")
    return 0


if __name__ == "__main__":
    sys.exit(main())
