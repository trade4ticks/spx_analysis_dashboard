"""How many symbols can one socket and one process carry?

THIS IS A MEASUREMENT, NOT A FEATURE. The Equities Scan page is a grid of
N symbols x 120 minutes, and whether that page is a curated watchlist or a
scanner over the whole universe is decided by a number nobody has measured
yet. Guessing it wrong is expensive in the direction that looks fine in
development: 50 symbols on a quiet afternoon proves nothing about 600 at the
open.

Run it against the real feed, during real hours, on the box that will host it.

    python scripts/measure_scan_capacity.py --steps 50,200,600 --seconds 120

WHAT IS MEASURED, and why each one is here rather than just "messages/sec":

  frames/s, records/s   the raw arrival rate. The headline number, and the
                        only one that scales with the symbol count in a way
                        you can extrapolate.
  bytes/s               what the socket actually costs. Frames/s alone hides
                        that a batched frame carries dozens of records.
  busy fraction         THE SATURATION SIGNAL. The share of wall-clock the
                        ingest loop spends inside ingest rather than awaiting
                        the next frame. At 1.0 the process is the bottleneck
                        and every other number below it is already a lie --
                        the feed is not slower, we are just behind it.
  cpu %                 of one core, from /proc. Related to busy fraction but
                        not the same: busy fraction can sit at 0.4 while CPU
                        reads 40% of a core, and it is the FRACTION that says
                        how much headroom is left before frames queue.
  rss                   the memory question, and the one with an OOM behind
                        it. The box has been OOM-killed. Reported per step
                        AND as growth across steps, because a leak and a
                        legitimately larger buffer look identical at one
                        reading.
  sip lag               now - the exchange timestamp on the record, p50/p95.
                        Includes real feed latency, so the ABSOLUTE value is
                        not ours to judge -- what matters is the TREND within
                        a step. Rising lag with a busy fraction below 1 means
                        the upstream is behind; rising lag with it at 1 means
                        we are.
  rollup ms             what the page's own arithmetic costs, separately from
                        ingest. See below.

WHY THE ROLLUP IS MEASURED SEPARATELY. Ingest is JSON parsing and appending;
the grid also has to COMPUTE something per symbol every few seconds -- the
quiet ratio over the trailing 60s window, the p10-p90 range and the dollar
volume over the trailing five minutes. That is a numpy pass per symbol, and at
600 symbols it is not obviously free. Measuring only ingest would report a
comfortable number for half the work. The two are reported separately so a
capacity finding is attributable: "the socket is fine, the arithmetic is not"
and "the arithmetic is fine, the socket is not" are different pages.

THE ROLLUP USES THE VENDORED quiet.py, not a reimplementation. If the number
here came from a second copy of the arithmetic it would be measuring something
the page will not run.

--- The buffer design is part of the measurement -----------------------------

The grid shows 120 minutes, and the obvious implementation -- keep 120 minutes
of trades per symbol -- does not survive arithmetic. At ~56 trades/min a busy
name is ~6,700 records over two hours; at 600 symbols that is four million,
and four million dicts of six fields is gigabytes on a box with twelve free.

So raw trades are retained only as long as the LONGEST LIVE WINDOW needs them
(the 5-minute range, plus a margin), and the two hours of history live as 120
finished per-minute CELLS -- four floats each. That is the design the page
will have to use, so it is the design measured here; a memory number for a
buffer nobody can ship is not a capacity finding.

Records are held in per-symbol NUMPY RING BUFFERS rather than a deque of
dicts, for the same reason: a 3-tuple is ~120 bytes against a dict's ~400, and
a preallocated float ring is 24, with the additional property that the rollup
does not have to rebuild an array from a deque on every pass.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import websockets

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from live import config                                   # noqa: E402
from app import scalp_quiet as quiet                      # noqa: E402


# -- process accounting -----------------------------------------------------
#
# /proc rather than psutil: the VPS venv does not have psutil, and a capacity
# harness that cannot run until someone installs a package is a harness that
# does not get run. Both files are stable kernel interfaces.

_CLK = os.sysconf("SC_CLK_TCK") if hasattr(os, "sysconf") else 100.0


def cpu_seconds() -> float:
    """User+system CPU consumed by this process, or nan off Linux."""
    try:
        with open("/proc/self/stat", "rb") as fh:
            raw = fh.read()
        # utime and stime are fields 14 and 15, 1-indexed -- but the process
        # NAME (field 2) may contain spaces and parentheses, so counting from
        # the left is wrong. Everything after the LAST close-paren is
        # fixed-width, which is the documented way to parse this file.
        tail = raw[raw.rindex(b")") + 2:].split()
        return (int(tail[11]) + int(tail[12])) / _CLK
    except (OSError, ValueError, IndexError):
        return float("nan")


def rss_mb() -> float:
    try:
        with open("/proc/self/status", "r") as fh:
            for line in fh:
                if line.startswith("VmRSS:"):
                    return float(line.split()[1]) / 1024.0
    except OSError:
        pass
    return float("nan")


# -- per-symbol state -------------------------------------------------------

class SymbolBuf:
    """A fixed-capacity ring of trades, plus the finished per-minute cells.

    The ring is sized from a RATE, not from a trade count pulled out of the
    air: retain_s seconds at max_rate trades/sec. Overflow overwrites the
    oldest, which is the correct behaviour -- a name printing faster than the
    ring holds has more recent trades than any window needs, and dropping the
    oldest is what the window would have done anyway. It is COUNTED, because a
    ring that is quietly too small understates memory and truncates the range.
    """

    __slots__ = ("t", "p", "s", "n", "head", "cap", "cells", "overflows")

    def __init__(self, cap: int, cells: int):
        self.cap = cap
        self.t = np.zeros(cap, dtype="float64")
        self.p = np.zeros(cap, dtype="float64")
        self.s = np.zeros(cap, dtype="float64")
        self.head = 0        # next write position
        self.n = 0           # records held, <= cap
        self.overflows = 0
        # The 120-minute history: one cell per minute, four floats --
        # ratio, range_c, dollar_per_min, trades. Exactly what a cell needs to
        # be coloured, gated, and expanded into the three bands.
        self.cells = np.full((cells, 4), np.nan, dtype="float64")

    def push(self, t_ms: float, price: float, size: float) -> None:
        i = self.head
        self.t[i] = t_ms
        self.p[i] = price
        self.s[i] = size
        self.head = (i + 1) % self.cap
        if self.n < self.cap:
            self.n += 1
        else:
            self.overflows += 1

    def ordered(self):
        """(t, p, s) oldest-first. One roll when the ring has wrapped.

        np.roll copies, which is the cost of using a ring at all; it is paid
        once per rollup per symbol rather than once per trade, which is the
        trade that makes the ring worth having.
        """
        if self.n < self.cap:
            return self.t[:self.n], self.p[:self.n], self.s[:self.n]
        k = -self.head
        return np.roll(self.t, k), np.roll(self.p, k), np.roll(self.s, k)


# -- the rollup: what the page actually computes -----------------------------

def rollup_one(buf: SymbolBuf, now_s: float, *, quiet_window_s: float,
               slow_window_s: float, min_trades: int) -> tuple:
    """One symbol's current cell: (ratio, range_cents, dollar_per_min, n).

    Quiet uses a 60-SECOND window and the slow quantities use FIVE MINUTES,
    deliberately and not as an oversight. Quiet is the thing that changes and
    the reason the page is open, so it has to be responsive; range and dollar
    volume answer "is this name worth anything at all", which is not a
    per-minute question, and a bar that jumps while you glance at it for a
    tenth of a second is worse than no bar.

    The quiet ratio comes from the VENDORED window_series, at the window and
    step the pipeline uses, so the grid and the stored metric are the same
    arithmetic rather than two things that ought to agree.
    """
    t, p, s = buf.ordered()
    if t.size == 0:
        return (np.nan, np.nan, np.nan, 0)
    t_sec = t / 1000.0

    # The quiet ratio needs the CURRENT window and its PREDECESSOR -- the
    # shift is a difference between the two -- so the span asked for is a
    # window plus a step, not a window.
    step_s = quiet.step_for(quiet_window_s)
    span = quiet_window_s + step_s
    ser = quiet.window_series(t_sec, p, s, window_s=quiet_window_s,
                              start_s=now_s - span, end_s=now_s,
                              step_s=step_s, min_trades=min_trades)
    ratio = float("nan")
    if ser["ratio"].size:
        # The LAST window is the current one. Earlier entries are history the
        # per-minute cells already hold.
        r = ser["ratio"][-1]
        ratio = float(r) if np.isfinite(r) else float("nan")

    slow = t_sec >= (now_s - slow_window_s)
    n_slow = int(slow.sum())
    if n_slow >= min_trades:
        ps = p[slow]
        q10, q90 = np.percentile(ps, (10.0, 90.0))
        range_c = float((q90 - q10) * 100.0)
        dollars = float(np.dot(ps, s[slow])) / (slow_window_s / 60.0)
    else:
        range_c = float("nan")
        dollars = float("nan")
    return (ratio, range_c, dollars, n_slow)


# -- the measurement --------------------------------------------------------

class Step:
    def __init__(self, symbols, channels, args):
        self.symbols = symbols
        self.channels = channels
        self.args = args
        cap = max(64, int(args.retain_s * args.max_rate))
        self.bufs = {s: SymbolBuf(cap, args.cells) for s in symbols}
        self.frames = 0
        self.records = 0
        self.trades = 0
        self.quotes = 0
        self.bytes = 0
        self.busy_s = 0.0
        self.lag_early: list[float] = []
        self.lag_late: list[float] = []
        self.rollup_ms: list[float] = []
        self.status_msgs: list[str] = []
        self.unknown_syms = 0
        self.half_at = float("inf")
        # Set rather than raised from inside ingest: ingest is called from the
        # measurement loop and is where the frame accounting happens, so it
        # finishes the frame it is holding and lets the loop decide.
        self.taken: str | None = None

    def ingest(self, raw) -> None:
        """One upstream frame. Times itself: this is the busy fraction."""
        t0 = time.perf_counter()
        self.frames += 1
        self.bytes += (len(raw) if isinstance(raw, (bytes, bytearray))
                       else len(raw.encode("utf-8", "replace")))
        try:
            msgs = json.loads(raw)
        except ValueError:
            self.busy_s += time.perf_counter() - t0
            return
        if isinstance(msgs, dict):
            msgs = [msgs]
        now = time.time()
        now_ms = now * 1000.0
        # Lag is split into the step's two halves rather than kept as one
        # distribution: the absolute value carries real feed latency and is
        # not ours to judge, but a p50 higher in the second half than the
        # first says we are falling behind, which is entirely ours.
        lags = self.lag_late if now >= self.half_at else self.lag_early
        for m in msgs:
            self.records += 1
            ev = m.get("ev")
            if ev == "T":
                self.trades += 1
                buf = self.bufs.get(m.get("sym"))
                if buf is None:
                    self.unknown_syms += 1
                    continue
                ts, price, size = m.get("t"), m.get("p"), m.get("s")
                if ts is None or price is None or size is None:
                    continue
                buf.push(float(ts), float(price), float(size))
                lags.append(now_ms - float(ts))
            elif ev == "Q":
                self.quotes += 1
            elif ev == "status":
                self.status_msgs.append(f"{m.get('status')}: {m.get('message')}")
                if m.get("status") == "max_connections":
                    self.taken = str(m.get("message") or "max_connections")
        self.busy_s += time.perf_counter() - t0

    def do_rollup(self) -> None:
        """Every symbol's current cell, timed as one pass."""
        now_s = time.time()
        t0 = time.perf_counter()
        for buf in self.bufs.values():
            rollup_one(buf, now_s, quiet_window_s=self.args.quiet_window_s,
                       slow_window_s=self.args.slow_window_s,
                       min_trades=self.args.min_trades)
        self.rollup_ms.append((time.perf_counter() - t0) * 1000.0)


def pct(values, q):
    v = [x for x in values if x == x]
    return float(np.percentile(v, q)) if v else float("nan")


class ConnectionTaken(Exception):
    """The account's single upstream connection is already in use.

    MEASURED, not assumed: the feed permits ONE concurrent websocket per
    account. A second one authenticates, accepts every subscription, and is
    then closed with 1008 and `max_connections` -- and the service that was
    already connected reconnects, which closes the new one, which reconnects.
    Two processes on one key do not share the feed; they evict each other in a
    loop, and both show a plausible-looking partial tape while doing it.

    This is the reason the scan page cannot open a socket of its own. It is
    raised as a distinct exception rather than surfacing as
    ConnectionClosedError, because "you are at the connection limit" and "the
    socket dropped" call for completely different responses and the traceback
    for the second is what the first used to look like.
    """


async def run_step(symbols, args) -> dict:
    """Subscribe, measure for `seconds`, report. One socket, opened fresh.

    A fresh socket per step rather than one socket resubscribed: a step's
    numbers must not carry the previous step's buffers, and a resubscribe
    leaves the server's idea of the subscription set to be trusted rather than
    established.
    """
    channels = [c.strip().upper() for c in args.channels.split(",") if c.strip()]
    st = Step(symbols, channels, args)
    url = config.feed_url()
    print("")
    print("=" * 76)
    print(f"  {len(symbols)} symbols x {','.join(channels)}   "
          f"{args.seconds:.0f}s measured   {url}")
    print("=" * 76)

    async with websockets.connect(url, ping_interval=20, ping_timeout=20,
                                  max_queue=args.max_queue) as ws:
        await ws.send(json.dumps({"action": "auth", "params": config.API_KEY}))

        # Subscriptions go out in CHUNKS. A single 600-symbol params string is
        # ~5KB, and whether that is accepted is a server-side question nobody
        # here can answer -- chunking removes it as a variable, so a refusal
        # is about the symbol COUNT rather than about one large message.
        n = args.chunk
        for i in range(0, len(symbols), n):
            part = symbols[i:i + n]
            params = ",".join(f"{c}.{s}" for s in part for c in channels)
            await ws.send(json.dumps({"action": "subscribe", "params": params}))
            await asyncio.sleep(0.05)

        # Warm-up. Subscriptions land over some hundreds of milliseconds and
        # the first frames arrive against a partial set, so counting from the
        # first frame would report a rate for a subscription that did not
        # exist yet. It is also when a refusal arrives, which is why status
        # messages are printed here rather than only summarised at the end.
        warm_until = time.time() + args.warmup
        while time.time() < warm_until:
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=1.0)
            except asyncio.TimeoutError:
                continue
            try:
                msgs = json.loads(raw)
            except ValueError:
                continue
            for m in (msgs if isinstance(msgs, list) else [msgs]):
                if m.get("ev") == "status":
                    line = f"{m.get('status')}: {m.get('message')}"
                    st.status_msgs.append(line)
                    print(f"  upstream>  {line}")
                    if m.get("status") == "max_connections":
                        raise ConnectionTaken(m.get("message") or line)

        cpu0, rss0 = cpu_seconds(), rss_mb()
        t_start = time.time()
        stop_at = t_start + args.seconds
        st.half_at = t_start + args.seconds / 2.0
        next_rollup = t_start + args.rollup_s

        while True:
            now = time.time()
            if now >= stop_at:
                break
            try:
                raw = await asyncio.wait_for(ws.recv(),
                                             timeout=max(0.05, stop_at - now))
            except asyncio.TimeoutError:
                pass
            else:
                st.ingest(raw)
                if st.taken:
                    raise ConnectionTaken(st.taken)
            if time.time() >= next_rollup:
                st.do_rollup()
                next_rollup = time.time() + args.rollup_s

        cpu1, rss1 = cpu_seconds(), rss_mb()
        elapsed = max(1e-9, time.time() - t_start)

    active = sum(1 for b in st.bufs.values() if b.n > 0)
    over = sum(b.overflows for b in st.bufs.values())
    out = {
        "symbols": len(symbols),
        "channels": ",".join(channels),
        "seconds": round(elapsed, 1),
        "frames_per_s": st.frames / elapsed,
        "records_per_s": st.records / elapsed,
        "trades_per_s": st.trades / elapsed,
        "quotes_per_s": st.quotes / elapsed,
        "kb_per_s": st.bytes / elapsed / 1024.0,
        "busy_fraction": st.busy_s / elapsed,
        "cpu_pct_of_core": (cpu1 - cpu0) / elapsed * 100.0,
        "rss_mb": rss1,
        "rss_growth_mb": rss1 - rss0,
        "lag_p50_early_ms": pct(st.lag_early, 50),
        "lag_p50_late_ms": pct(st.lag_late, 50),
        "lag_p95_late_ms": pct(st.lag_late, 95),
        "rollup_ms_p50": pct(st.rollup_ms, 50),
        "rollup_ms_p95": pct(st.rollup_ms, 95),
        "rollups": len(st.rollup_ms),
        "symbols_with_trades": active,
        "ring_overflows": over,
        "unknown_symbol_records": st.unknown_syms,
        "status": st.status_msgs[:10],
    }
    report(out, args)
    return out


def report(r: dict, args) -> None:
    print(f"  arrival    {r['frames_per_s']:8.1f} frames/s   "
          f"{r['records_per_s']:9.1f} records/s   "
          f"({r['trades_per_s']:.1f} T, {r['quotes_per_s']:.1f} Q)   "
          f"{r['kb_per_s']:.0f} KB/s")
    print(f"  process    busy {r['busy_fraction'] * 100:5.1f}% of wall   "
          f"cpu {r['cpu_pct_of_core']:5.1f}% of one core   "
          f"rss {r['rss_mb']:.0f} MB (+{r['rss_growth_mb']:.0f} in step)")
    print(f"  lag        p50 {r['lag_p50_early_ms']:8.0f} ms first half -> "
          f"{r['lag_p50_late_ms']:8.0f} ms second half "
          f"(p95 late {r['lag_p95_late_ms']:.0f} ms)")
    if r["rollups"]:
        print(f"  rollup     {r['rollup_ms_p50']:8.1f} ms p50   "
              f"{r['rollup_ms_p95']:8.1f} ms p95   for {r['symbols']} symbols "
              f"every {args.rollup_s:.0f}s ({r['rollups']} passes)")
    else:
        print("  rollup     not run (--no-rollup)")
    print(f"  coverage   {r['symbols_with_trades']}/{r['symbols']} symbols "
          f"printed at least once")
    if r["ring_overflows"]:
        print(f"  WARNING    {r['ring_overflows']} ring overflows -- "
              f"--max-rate is too low for these names; memory is understated "
              f"and the range bar is truncated")
    if r["unknown_symbol_records"]:
        print(f"  WARNING    {r['unknown_symbol_records']} records arrived for "
              f"symbols never subscribed")
    for s in r["status"]:
        print(f"  upstream>  {s}")

    # The verdict is STATED, not left to be inferred from four numbers. A
    # capacity harness whose output needs interpreting is one whose answer
    # gets remembered as "it seemed fine".
    verdict = []
    if r["busy_fraction"] > 0.8:
        verdict.append("SATURATED -- the ingest loop is the bottleneck")
    elif r["busy_fraction"] > 0.5:
        verdict.append("busy -- under half the loop is left for headroom")
    drift = r["lag_p50_late_ms"] - r["lag_p50_early_ms"]
    if drift == drift and drift > 500:
        verdict.append(f"lag grew {drift:.0f} ms within the step")
    if r["rollups"] and r["rollup_ms_p50"] > args.rollup_s * 1000 * 0.5:
        verdict.append("the rollup alone eats half its own interval")
    if r["symbols_with_trades"] < 0.5 * r["symbols"]:
        verdict.append("under half the symbols printed -- the market may be "
                       "closed, and this is not a capacity number")
    print(f"  VERDICT    {'; '.join(verdict) if verdict else 'comfortable'}")


# -- symbols ----------------------------------------------------------------

def holder_of_the_connection() -> str | None:
    """Whether the live service is already holding the one socket.

    Asked BEFORE connecting rather than discovered afterwards. Taking the
    connection does not fail politely: this process is evicted, spx-live
    reconnects, this process reconnects, and the two trade the socket back and
    forth for as long as the run lasts -- during which the tape page shows a
    live-looking plot fed by a connection that keeps dropping, and every
    number this harness produces is measured against a fraction of the feed.

    Best-effort. A service that cannot be reached is not evidence that nothing
    holds the connection, so the answer is advisory and the run continues.
    """
    try:
        import urllib.request
        url = f"http://127.0.0.1:{config.PORT}/status"
        with urllib.request.urlopen(url, timeout=2) as r:
            st = json.load(r)
    except Exception:                                     # noqa: BLE001
        return None
    if st.get("connected"):
        syms = st.get("symbols") or []
        return (f"spx-live on port {config.PORT} is connected "
                f"(up {st.get('uptime_s', 0):.0f}s, {len(syms)} symbols, "
                f"{st.get('reconnects', 0)} reconnects)")
    return None


async def symbols_from_db(limit: int) -> list[str]:
    """The busiest names in the most recent universe, dollar volume first.

    The scan is a tool for finding something to trade, so the capacity test
    has to be run against names that PRINT. Measuring 600 randomly chosen
    tickers would report the message rate of a sleeping market and call it
    capacity.
    """
    import asyncpg
    from urllib.parse import urlsplit, urlunsplit
    dsn = os.getenv("SCALP_DATABASE_URL")
    if not dsn:
        parts = urlsplit(os.environ["DATABASE_URL"])
        dsn = urlunsplit(parts._replace(path="/equities_scalp"))
    con = await asyncpg.connect(dsn)
    try:
        rows = await con.fetch(
            """select symbol from universe
               where trade_date = (select max(trade_date) from universe)
               order by dollar_volume desc nulls last
               limit $1""", limit)
    finally:
        await con.close()
    return [r[0] for r in rows]


# -- the self-test ----------------------------------------------------------

def self_test() -> int:
    """Prove the harness measures WORK, not an early return.

    The failure this exists to catch has happened in this project before, in
    exactly this shape: a timed loop over a computation that quietly produces
    nothing runs FASTER than one that works, so a broken rollup reports as
    excellent capacity. Every number in the report would then be a measurement
    of `return nan`.

    So the rollup is run against a synthetic tape whose answers are known, and
    a NaN ratio is a FAILURE rather than a shrug. Also checks the ring after
    it has wrapped, because ordering is the one part of a ring buffer that is
    wrong silently and only under load.
    """
    fails = []

    def check(cond, msg):
        if not cond:
            fails.append(msg)

    # /proc parsing. A nan here means every cpu and rss figure in the report
    # is nan, which reads as "not measured" only if you notice it.
    if sys.platform.startswith("linux"):
        c0 = cpu_seconds()
        check(c0 == c0 and c0 >= 0,
              f"cpu_seconds() returned {c0} on Linux -- /proc/self/stat "
              f"parsing is wrong and every CPU figure would be nan")
        junk = [x * x for x in range(400000)]
        c1 = cpu_seconds()
        check(c1 > c0, f"cpu_seconds() did not advance across real work "
                       f"({c0} -> {c1}); it is not reading the counter")
        del junk
        r = rss_mb()
        check(r == r and r > 1, f"rss_mb() returned {r}; VmRSS was not found")
    else:
        print("  (not Linux: /proc accounting is untested here and will "
              "report nan -- run the real measurement on the VPS)")

    # The ring, wrapped. Written 1.5x its capacity, so half of it has been
    # overwritten and `ordered` has to un-rotate.
    cap = 100
    b = SymbolBuf(cap, 120)
    for i in range(150):
        b.push(float(i), 10.0 + i, 1.0)
    t, p, s = b.ordered()
    check(t.size == cap, f"ring holds {t.size} after 150 pushes into {cap}")
    check(b.overflows == 50, f"ring counted {b.overflows} overflows, want 50")
    check(list(t[:3]) == [50.0, 51.0, 52.0],
          f"wrapped ring came back out of order: starts {list(t[:3])}, "
          f"want [50.0, 51.0, 52.0]")
    check(t[-1] == 149.0, f"wrapped ring's newest is {t[-1]}, want 149.0")
    check(bool(np.all(np.diff(t) == 1.0)),
          "wrapped ring is not monotonic in time")

    # The rollup, against a tape with a KNOWN answer. Two 60s windows: the
    # first jitters around 100.00, the second around 100.20 -- so the level
    # shifts about 20 cents against an IQR of a few cents, and the ratio must
    # come out well above 1. A quiet tape (no shift) must come out near 0.
    now = time.time()
    rng = np.random.default_rng(7)

    def tape(shift_dollars):
        buf = SymbolBuf(4000, 120)
        # 40 trades/min over the 80 seconds the rollup asks for, plus the
        # five-minute slow window behind it.
        for k in range(400):
            age = 300.0 * (1.0 - k / 400.0)          # 300s ago -> now
            base = 100.0 + (shift_dollars if age < 20.0 else 0.0)
            buf.push((now - age) * 1000.0,
                     base + float(rng.normal(0, 0.01)), 100.0)
        return buf

    loud = rollup_one(tape(0.20), now, quiet_window_s=60.0,
                      slow_window_s=300.0, min_trades=10)
    still = rollup_one(tape(0.0), now, quiet_window_s=60.0,
                       slow_window_s=300.0, min_trades=10)
    for name, got in (("moved", loud), ("still", still)):
        check(got[0] == got[0],
              f"{name} tape produced a NaN quiet ratio -- the rollup is "
              f"computing nothing and its timing is meaningless")
        check(got[1] == got[1] and got[1] > 0,
              f"{name} tape produced range {got[1]}, want a positive span")
        check(got[2] == got[2] and got[2] > 0,
              f"{name} tape produced {got[2]} dollars/min, want positive")
        check(got[3] >= 10, f"{name} tape saw {got[3]} slow-window trades")
    check(loud[0] > still[0],
          f"a tape that moved 20 cents scored {loud[0]:.3f}, no louder than "
          f"one that did not ({still[0]:.3f}) -- the ratio is not responding "
          f"to the shift it exists to measure")
    check(still[0] < 1.0,
          f"an unmoved tape scored {still[0]:.3f}; a still name must read "
          f"quiet or the grid lights up on nothing")

    # Dollar volume is checkable exactly, and it is worth checking rather
    # than eyeballing: 400 trades of 100 shares near $100 is $4.0M over the
    # 300-second slow window, so $800k a minute. The bounds are loose because
    # this is a wiring check on the /60 normalisation -- getting that wrong by
    # a factor of five is what the amber bar would silently render.
    check(7e5 < still[2] < 9e5,
          f"dollars/min came out {still[2]:.0f}; 400 x 100sh x ~$100 over "
          f"300s is ~8.0e5, so the per-minute normalisation is wrong")

    for f in fails:
        print(f"  FAIL  {f}")
    print(f"\n  self-test: {'PASS' if not fails else str(len(fails)) + ' FAILED'}")
    return 1 if fails else 0


async def main() -> int:
    ap = argparse.ArgumentParser(
        description="Measure the Equities Scan symbol ceiling.")
    ap.add_argument("--self-test", action="store_true",
                    help="prove the rollup computes something, without a "
                         "market or a key")
    ap.add_argument("--steps", default="50,200,600",
                    help="symbol counts to measure, in order")
    ap.add_argument("--seconds", type=float, default=120.0,
                    help="measured seconds per step, after warm-up")
    ap.add_argument("--warmup", type=float, default=10.0)
    ap.add_argument("--channels", default="T",
                    help="T for trades only (all the scan needs); T,Q to "
                         "price in the tape page's quote load as well")
    ap.add_argument("--chunk", type=int, default=200,
                    help="symbols per subscribe message")
    ap.add_argument("--symbols-file", default=None)
    ap.add_argument("--rollup-s", type=float, default=5.0)
    ap.add_argument("--quiet-window-s", type=float, default=60.0)
    ap.add_argument("--slow-window-s", type=float, default=300.0)
    ap.add_argument("--min-trades", type=int, default=quiet.MIN_TRADES)
    ap.add_argument("--retain-s", type=float, default=360.0,
                    help="seconds of raw trades kept per symbol; must exceed "
                         "the slow window")
    ap.add_argument("--max-rate", type=float, default=20.0,
                    help="trades/second the ring is sized for, per symbol")
    ap.add_argument("--cells", type=int, default=120,
                    help="per-minute history cells held per symbol")
    ap.add_argument("--max-queue", type=int, default=4096)
    ap.add_argument("--no-rollup", action="store_true",
                    help="ingest only, to separate the two costs")
    ap.add_argument("--force", action="store_true",
                    help="measure even though something else holds the "
                         "account's one connection")
    ap.add_argument("--out", default=None, help="write results as JSON")
    args = ap.parse_args()

    if args.self_test:
        return self_test()

    problems = config.problems()
    if problems:
        for p in problems:
            print(f"configuration: {p}")
        return 2
    if args.retain_s <= args.slow_window_s:
        print(f"--retain-s {args.retain_s} does not cover --slow-window-s "
              f"{args.slow_window_s}; the range bar would read short.")
        return 2
    if args.no_rollup:
        args.rollup_s = 1e9

    steps = [int(x) for x in args.steps.split(",") if x.strip()]
    want = max(steps)
    if args.symbols_file:
        with open(args.symbols_file) as fh:
            pool = [ln.strip().upper() for ln in fh
                    if ln.strip() and not ln.startswith("#")]
    else:
        pool = await symbols_from_db(want)
    if len(pool) < want:
        print(f"only {len(pool)} symbols available; the {want} step will "
              f"measure {len(pool)}")
    holder = holder_of_the_connection()
    if holder:
        print("")
        print("  REFUSING TO START: " + holder)
        print("")
        print("  The account permits ONE concurrent websocket. Starting a")
        print("  second one does not queue or share -- the two evict each")
        print("  other for the length of the run, so this harness would")
        print("  measure a feed it keeps being thrown off, and the tape page")
        print("  would drop repeatedly while looking live.")
        print("")
        print("    systemctl stop spx-live")
        print("    " + " ".join(sys.argv))
        print("    systemctl start spx-live")
        print("")
        print("  --force to measure anyway (the numbers will be wrong).")
        if not args.force:
            return 3
        print("  --force given; continuing against a contended connection.")

    print(f"feed:   {config.FEED} -> {config.feed_url()}")
    print(f"pool:   {len(pool)} symbols, busiest first: "
          f"{', '.join(pool[:8])} ...")
    print(f"rollup: quiet {args.quiet_window_s:.0f}s "
          f"(step {quiet.step_for(args.quiet_window_s):.0f}s), "
          f"range/volume {args.slow_window_s:.0f}s, "
          f"every {args.rollup_s:.0f}s")

    results = []
    for n in steps:
        syms = pool[:n]
        if not syms:
            continue
        try:
            results.append(await run_step(syms, args))
        except ConnectionTaken as exc:
            print("")
            print("  " + "=" * 72)
            print("  ABORTED: the account's ONE upstream connection is in use")
            print("  " + "=" * 72)
            print(f"    upstream said: {exc}")
            print("")
            print("    The feed permits a single concurrent websocket per")
            print("    account. This process authenticated, was accepted for")
            print("    every subscription, and was then evicted -- and the")
            print("    service that already held the connection reconnected,")
            print("    which evicts this one again. They do not share it.")
            print("")
            print("    Stop the holder for the length of the measurement:")
            print("      systemctl stop spx-live && <this command>"
                  " ; systemctl start spx-live")
            return 3
        except websockets.exceptions.ConnectionClosed as exc:
            # Distinguished from the case above ON PURPOSE. A 1008 after a
            # max_connections status is the limit; a close without one is an
            # ordinary drop, and reporting both the same way is how the limit
            # went unnoticed as "the socket is flaky".
            print(f"\n  step of {n} symbols ended early: {exc}")
            print("  (no max_connections status was seen, so this is an "
                  "ordinary drop rather than the connection limit)")
            continue
        # A pause between steps so the previous socket is closed server-side
        # before the next opens. Two overlapping connections on one key is a
        # different test from the one being run.
        await asyncio.sleep(3.0)

    print("")
    print("=" * 76)
    print("  summary")
    print("=" * 76)
    print(f"  {'syms':>6} {'rec/s':>9} {'KB/s':>8} {'busy':>7} {'cpu':>7} "
          f"{'rss MB':>8} {'rollup ms':>10} {'lag drift':>10}")
    for r in results:
        drift = r["lag_p50_late_ms"] - r["lag_p50_early_ms"]
        print(f"  {r['symbols']:>6} {r['records_per_s']:>9.1f} "
              f"{r['kb_per_s']:>8.0f} {r['busy_fraction'] * 100:>6.1f}% "
              f"{r['cpu_pct_of_core']:>6.1f}% {r['rss_mb']:>8.0f} "
              f"{r['rollup_ms_p50']:>10.1f} {drift:>10.0f}")

    if args.out:
        Path(args.out).write_text(json.dumps(
            {"when": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
             "feed": config.FEED, "args": vars(args), "steps": results},
            indent=2))
        print(f"\n  wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
