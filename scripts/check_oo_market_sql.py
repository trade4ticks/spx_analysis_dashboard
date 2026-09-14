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

Fabricated sessions, all ET, start-labeled:
  2023-06-29 Thu  full; junk 16:00 row and a pre-market row that must be ignored;
                  VIX3M absent (coverage starts the next day); first day -> no gap
  2023-06-30 Fri  full; SPX 15:55 close is 'NaN' -> close from 15:50 (a
                  full-session fallback, reported); VIX 15:55 open NaN
  (weekend)
  2023-07-03 Mon  EARLY CLOSE, last bar 12:55
  (07-04 holiday)
  2023-07-05 Wed  full; its gap must use 07-03's 12:55 close

Planted faults, each of which must change the answer: a close pinned to
15:55, a prior close from trade_date - 1, and an entry join reading the bar
close.

SKIPS -- loudly -- where no Postgres server binaries are found. Set PG_BIN to
the directory holding initdb/pg_ctl if they are somewhere unusual.

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

DAYS = [date(2023, 6, 29), date(2023, 6, 30), date(2023, 7, 3), date(2023, 7, 5)]
EARLY = date(2023, 7, 3)
NAN = "NaN"


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
            rows.append((d, t, vals))
        if d != EARLY:
            # 16:00 partial row with absurd values: must never be read.
            rows.append((d, time(16, 0), {f"{s}_{f}": 99999.0 for s in market.SERIES
                                         for f in ("open", "high", "low", "close")}))
    # A pre-market row on day one, also never read.
    rows.append((DAYS[0], time(9, 25), {f"{s}_{f}": -1.0 for s in market.SERIES
                                        for f in ("open", "high", "low", "close")}))
    return rows


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


async def run(dsn: str) -> None:
    import asyncpg
    pool = await asyncpg.create_pool(dsn, min_size=1, max_size=3)
    try:
        await load(pool)
        await check_daily(pool)
        await check_entry_join(pool)
        await check_planted(pool)
        await check_end_to_end(pool)
    finally:
        await pool.close()


async def check_daily(pool) -> None:
    print("daily rollup")
    daily, fresh = await market.get_daily(pool)
    by = {r["trade_date"]: r for r in daily.to_dict("records")}
    check(sorted(by) == DAYS, f"one row per trade_date, weekend/holiday absent ({len(by)})")
    check(fresh["latest_date"] == "2023-07-05" and fresh["latest_time"] == "16:00:00",
          f"freshness reads the table's latest bar ({fresh['latest_date']} {fresh['latest_time']})")
    for d in DAYS:
        r = by[d]
        want, wt = expected_close("spx", d)
        check(r["spx_close"] == want and r["spx_close_time"] == wt,
              f"{d}: SPX close {r['spx_close']} from {r['spx_close_time']} (want {want} @ {wt})")
        check(r["spx_open"] == bar_open("spx", d, time(9, 30)), f"{d}: SPX open from the 09:30 bar, not 09:25")
        hi = max(px("spx", DAYS.index(d), k)["high"] for k in range(len(session_times(d))))
        check(r["spx_high"] == hi, f"{d}: high ignores the 16:00 row ({r['spx_high']} vs {hi})")
    check(by[DAYS[3]]["prev_trade_date"] == EARLY, "07-05's previous row is 07-03 (not the 07-04 holiday)")
    check(by[DAYS[3]]["spx_prev_close"] == expected_close("spx", EARLY)[0],
          "07-05's prior close is 07-03's 12:55 close")
    check(by[DAYS[2]]["spx_prev_close"] == expected_close("spx", DAYS[1])[0],
          "07-03 (Monday) prior close is Friday 06-30's, across the weekend")
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
          f"coverage per series from real non-null closes {cov}")
    labels = market.bar_labels()
    y = labels[0]
    check(y["days"] == 4 and y["days_with_0930"] == 4 and y["days_with_valid_1600"] == 3 and y["days_with_premarket"] == 1,
          f"bar-label diagnostics count what is there {y}")


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
    e = rep["entry_time"]
    check(e["entry_time_found"] == 6 and e["entry_time_fallback_0930"] == 1 and e["before_open"] == 1,
          f"entry-time coverage reported {e}")
    check(rep["entry_bars"]["vix"]["earlier_than_entry_bar"] == 1, "a bar pushed back by NaN is counted")

    print("gaps and ratios")
    d3c = expected_close("spx", EARLY)[0]
    want = (bar_open("spx", DAYS[3], time(9, 30)) - d3c) / d3c * 100
    check(math.isclose(L[6]["gap"], want, rel_tol=1e-12), f"07-05 SPX gap vs 07-03 12:55 close ({L[6]['gap']:.5f}%)")
    check(pd.isna(L[1]["gap"]), "first date gap is null, not zero")
    vprev = expected_close("vix", DAYS[1])[0]
    check(math.isclose(L[0]["vix_overnight_gap"], (bar_open("vix", EARLY, time(9, 30)) - vprev) / vprev * 100),
          "VIX gap: 09:30 open vs prior row's close")
    check(math.isclose(L[6]["vix3m_vix_ratio_entry"], L[6]["vix3m_level"] / L[6]["vix_level"]),
          "entry-basis ratio from the entry bars")
    vc, v3c = expected_close("vix", DAYS[3])[0], expected_close("vix3m", DAYS[3])[0]
    check(math.isclose(L[6]["vix3m_vix_ratio_close"], v3c / vc), "close-basis ratio from the daily closes")

    print("vendor Gap cross-check")
    daily, _ = await market.get_daily(pool)
    good = df.copy()
    good["csv_gap"] = good["gap"]
    x = market.gap_crosscheck(good, daily)
    check(x["best_alignment"] == "previous_row" and x["best_unit"] == "pct" and x["disagree"] == 0,
          f"a vendor column matching the page -> previous_row/pct, no disagreement ({x['compared']} compared)")
    d = daily.sort_values("trade_date").reset_index(drop=True)
    two_back = dict(zip(d["trade_date"], d["spx_close"].shift(2)))
    opens = dict(zip(d["trade_date"], d["spx_open"]))
    bad = df.copy()
    bad["csv_gap"] = [market._pct(opens.get(t), two_back.get(t)) for t in pd.to_datetime(bad["date_opened"]).dt.date]
    y = market.gap_crosscheck(bad, daily)
    check(y["best_alignment"] == "two_rows_back",
          f"an off-by-one vendor column is IDENTIFIED as off by one ({y['best_alignment']})")


async def check_planted(pool) -> None:
    print("planted faults")
    async with pool.acquire() as conn:
        pinned = await conn.fetchval("""SELECT NULLIF(spx_close, 'NaN'::float8) FROM index_ohlc
                                        WHERE trade_date = $1 AND quote_time = TIME '15:55'""", EARLY)
        cal_prev = await conn.fetchval("""SELECT NULLIF(spx_close, 'NaN'::float8) FROM index_ohlc
                                          WHERE trade_date = $1::date - 1 AND quote_time = TIME '15:55'""", DAYS[3])
        bar_close = await conn.fetchval("""SELECT vix_close FROM index_ohlc
                                           WHERE trade_date = $1 AND quote_time = TIME '10:00'""", DAYS[3])
    daily, _ = await market.get_daily(pool)
    by = {r["trade_date"]: r for r in daily.to_dict("records")}
    check(pinned is None and by[EARLY]["spx_close"] is not None,
          "a close pinned to 15:55 loses the early-close day; the rollup does not")
    check(cal_prev is None and by[DAYS[3]]["spx_prev_close"] is not None,
          "a prior close from trade_date - 1 finds nothing after a holiday; the previous row does")
    df, _ = await market.join_market(pool, trades())
    check(df.loc[6, "vix_level"] != bar_close, "the entry join is not reading the bar's close (5 min lookahead)")


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
    check(rep["gap_crosscheck"] is None, "no vendor Gap column -> no cross-check claimed")


def main() -> int:
    if os.name == "posix" and os.geteuid() == 0:
        print("SKIP: running as root — initdb refuses to create a cluster as root; run as an ordinary user")
        return 0
    bindir = find_bin()
    if bindir is None:
        print("SKIP: no Postgres server binaries (initdb, pg_ctl) found — set PG_BIN to run this check")
        return 0
    print(f"postgres binaries: {bindir}")
    try:
        import asyncpg  # noqa: F401
    except ImportError:
        print("SKIP: asyncpg not installed")
        return 0
    with Cluster(bindir) as c:
        asyncio.run(run(c.dsn))
    print()
    if FAILS:
        print(f"FAIL: {len(FAILS)} market SQL check(s) failed")
        return 1
    print("PASS: market SQL — rollup, early close, prior row, entry bar, gaps, cross-check, planted faults")
    return 0


if __name__ == "__main__":
    sys.exit(main())
