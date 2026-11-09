"""The wall tier, its watchlist, its frames and its page, on a fabricated feed.

NO MARKET NEEDED. The wall is a rotation list (2026-11-06): one row per
symbol, ticker | range bar | p90 in cents, on the QUOTE channel only. Since
2026-11-09 each symbol is a 30 s window of 100 ms buckets, and the page is
driven by its p90.

WHAT IS BEING PROTECTED, in order of how bad it would be:

  * THE OTHER TIERS' CHANNELS. Three tiers share one upstream socket, and
    they no longer want the same channels: a pane and the scan take trades
    and quotes, the wall quotes only. A wall-only symbol must be `Q.` alone;
    a pane opening on it must add `T.` (or the pane's tape is empty), and
    closing must drop `T.` and NOT `Q.` (or the wall row freezes) -- and the
    same for every pair of tiers. Every one of these failures looks like a
    market that went quiet.
  * THE WINDOW IS THE STEP FUNCTION, SAMPLED: every bucket holds the spread
    standing at its end, carried forward through silence (a quiet name must
    not vanish), time-weighted (a furious tight requoter must not bury a
    still wide stretch) -- held to a brute-force reference on random feeds.
  * THE FRAME IS A DELTA OF WINDOWS, and the frame still goes out when empty.
  * THE WATCHLIST SURVIVES A RESTART, with both thresholds.
  * THE THRESHOLDS ARE PRESENTATION: they never reach the hub.
  * THE PAGE: columns flow top to bottom, colours and bar length follow the
    two thresholds, marks are memory only, alphabetical by default.
  * THE PAGE NEVER TALKS TO A BROKER. This page is for watching.
"""
from __future__ import annotations

import asyncio
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from live import config                                   # noqa: E402
from live.hub import Hub                                  # noqa: E402
from live.wall import WallSym                             # noqa: E402
from live.wall_runner import WallRunner                   # noqa: E402
from live.wall_store import WallStore                     # noqa: E402

FAILS: list[str] = []

JS = ROOT / "static" / "js" / "equities_wall.js"
TPL = ROOT / "templates" / "equities_wall.html"


def check(cond, msg):
    if not cond:
        FAILS.append(msg)


def trade(sym, t, p, s):
    return {"ev": "T", "sym": sym, "t": t, "p": p, "s": s, "x": 4, "z": 1}


def quote(sym, t, bp, ap):
    return {"ev": "Q", "sym": sym, "t": t, "bp": bp, "ap": ap,
            "bs": 1, "as": 1}


def recording_hub() -> Hub:
    """A hub whose upstream sends are captured rather than sent."""
    h = Hub()
    h.sent = []

    async def _send(payload):
        h.sent.append(payload)

    h._send = _send
    return h


def params(hub) -> str:
    return " | ".join(f"{p['action']}:{p['params']}" for p in hub.sent)


def sent(hub, action) -> list[str]:
    return [p["params"] for p in hub.sent if p["action"] == action]


class FakeSock:
    """A browser socket that keeps what it was sent."""

    def __init__(self):
        self.frames = []

    async def send_json(self, payload):
        self.frames.append(payload)


def runner(hub, tmp: Path) -> WallRunner:
    return WallRunner(hub, WallStore(tmp / "wall_watchlist.json"))


# ── the shared socket, per channel ─────────────────────────────────────────
async def case_wall_subscribes_quotes_only():
    """A wall-only symbol is `Q.` and nothing else, on and off."""
    h = recording_hub()
    await h.wall_set(["FDX", "LLY"])
    check(sent(h, "subscribe") == ["Q.FDX,Q.LLY"],
          f"the wall subscribed {sent(h, 'subscribe')}; it shows a spread "
          f"and nothing else, so trades are paid for and thrown away")
    h.sent.clear()
    await h.wall_set([])
    check(sent(h, "unsubscribe") == ["Q.FDX,Q.LLY"] and not sent(h, "subscribe"),
          f"dropping the wall sent {params(h)!r}; the last holder must take "
          f"the quote channel with it, and only that")


async def case_tiers_keep_each_others_channels():
    """Every pair of tiers, both directions, one channel at a time.

    The failures: the pane's tape empty because the wall already held the
    symbol and nothing sent `T.`; or the wall's row frozen because a pane
    closing unsubscribed both channels.
    """
    # a pane opens on a wall symbol: T is added, alone
    h = recording_hub()
    await h.wall_set(["FDX"])
    h.sent.clear()
    await h.acquire("FDX")
    check(h.sent == [{"action": "subscribe", "params": "T.FDX"}],
          f"a pane opening on a wall-only symbol sent {params(h)!r}; it "
          f"needs the trade channel, and only that is missing")
    # and closes: T goes, Q stays
    h.sent.clear()
    await h.release("FDX")
    check(h.sent == [{"action": "unsubscribe", "params": "T.FDX"}],
          f"a pane closing on a wall symbol sent {params(h)!r}; dropping Q "
          f"would freeze the wall's row")

    # the scan, the same both ways
    h = recording_hub()
    await h.wall_set(["FDX"])
    h.sent.clear()
    await h.scan_set(["FDX"])
    check(h.sent == [{"action": "subscribe", "params": "T.FDX"}],
          f"the scan taking a wall symbol sent {params(h)!r}")
    h.sent.clear()
    await h.scan_set([])
    check(h.sent == [{"action": "unsubscribe", "params": "T.FDX"}],
          f"the scan dropping a wall symbol sent {params(h)!r}")

    # the wall joins and leaves a symbol a pane or the scan holds: nothing
    for holder in ("pane", "scan"):
        h = recording_hub()
        if holder == "pane":
            await h.acquire("FDX")
        else:
            await h.scan_set(["FDX"])
        await h.wall_set(["FDX"])
        check(sent(h, "subscribe") == ["T.FDX,Q.FDX"],
              f"with the {holder} holding FDX the wall re-subscribed: "
              f"{params(h)!r}")
        h.sent.clear()
        await h.wall_set([])
        check(h.sent == [],
              f"the wall dropping FDX sent {params(h)!r} while the {holder} "
              f"still reads both channels")

    # a pane and the scan between themselves are unchanged: both channels
    h = recording_hub()
    await h.scan_set(["FDX"])
    await h.acquire("FDX")
    h.sent.clear()
    await h.release("FDX")
    check(h.sent == [], f"a pane closing on a scan symbol sent {params(h)!r}")
    await h.scan_set([])
    check(sent(h, "unsubscribe") == ["T.FDX,Q.FDX"],
          f"the last holder left {params(h)!r}; both channels must go")


async def case_reconnect_restores_each_channel():
    """A reconnect re-subscribes every tier, each on its own channels."""
    h = recording_hub()
    await h.scan_set(["AAPL"])
    await h.wall_set(["FDX", "LLY", "AAPL"])
    await h.acquire("NVDA")
    syms = sorted(set(h.refs) | set(h.scan) | set(h.wall))
    from live.hub import _params
    got = ",".join(_params(s, h._wants(s)) for s in syms)
    check(got == "T.AAPL,Q.AAPL,Q.FDX,Q.LLY,T.NVDA,Q.NVDA",
          f"a reconnect would subscribe {got}")
    src = (ROOT / "live" / "hub.py").read_text(encoding="utf-8")
    check("set(self.refs) | set(self.scan) | set(self.wall)" in src
          and "_params(s, self._wants(s)) for s in syms" in src,
          "_session() does not rebuild its subscription from every tier's "
          "own channels; a reconnect would restore some pages and not others")


# ── ingest ──────────────────────────────────────────────────────────────────
async def case_wall_takes_quotes_and_ignores_trades():
    """A trade for a wall symbol (live because a pane holds it) is ignored."""
    h = recording_hub()
    await h.scan_set(["FDX"])
    await h.wall_set(["FDX"])
    await h.acquire("FDX")
    now = time.time() * 1000
    h._ingest(json.dumps([trade("FDX", now, 334.4, 100),
                          quote("FDX", now, 334.37, 334.44)]))
    w = h.wall["FDX"]
    check(w.spread_cents() == 7.0, f"the wall's spread is {w.spread_cents()}")
    check(not hasattr(w, "trades"), "the wall still keeps a trade store")
    check(h.scan["FDX"].n == 1 and len(h.trades["FDX"]) == 1,
          "the scan or the pane missed the trade")
    check(h.tier_in["wall"] == [0, 1],
          f"wall arrivals counted as {h.tier_in['wall']}; trades must stay 0")


async def case_every_quote_is_taken():
    """Nothing is sampled: the current spread is the newest message."""
    h = recording_hub()
    await h.wall_set(["FDX"])
    now = time.time() * 1000
    h._ingest(json.dumps([quote("FDX", now + i, 334.40, 334.41 + i * 0.01)
                          for i in range(50)]))
    w = h.wall["FDX"]
    check(w.quotes_seen == 50 and w.spread_cents() == 50.0,
          f"{w.quotes_seen} quotes taken, spread {w.spread_cents()}; every "
          f"message is the current spread, 50c after the last")


async def case_a_crossed_quote_is_dropped_and_counted():
    """A zero bid would read as a whole-ask spread, in the wide colour."""
    h = recording_hub()
    await h.wall_set(["FDX"])
    w = h.wall["FDX"]
    now = time.time() * 1000
    h._ingest(json.dumps([quote("FDX", now, 334.37, 334.44)]))
    h._ingest(json.dumps([quote("FDX", now + 300, 0, 334.44),
                          quote("FDX", now + 600, 334.50, 334.44),
                          quote("FDX", now + 900, 334.40, 0)]))
    check(w.spread_cents() == 7.0,
          f"a crossed or one-sided quote became the spread: {w.spread_cents()}")
    check(w.crossed == 3, f"{w.crossed} of 3 unusable quotes were counted")
    check(h.wall_status()["crossed"] == 3, "status hides the crossed count")


# ── frames ──────────────────────────────────────────────────────────────────
async def case_first_frame_is_every_symbol(tmp: Path):
    """A new page gets every row at once, null for one that has not quoted."""
    h = recording_hub()
    r = runner(h, tmp)
    await r.apply([{"symbol": "FDX"}, {"symbol": "KO"}])
    h._ingest(json.dumps([quote("FDX", time.time() * 1000, 334.37, 334.44)]))
    st = r.subscribe(FakeSock())
    f = r.frame(st)
    check(f["ev"] == "tick" and f["syms"] == {"FDX": [7, 7, 7], "KO": None},
          f"the first frame was {f['syms']}")


async def case_frame_is_a_delta_of_windows(tmp: Path):
    """Only a changed window is sent; a quote that moves both sides is not."""
    h = recording_hub()
    r = runner(h, tmp)
    await r.apply([{"symbol": "FDX"}, {"symbol": "KO"}])
    t0 = bucket_aligned(-60_000)
    h._ingest(json.dumps([quote("FDX", t0, 334.37, 334.44),
                          quote("KO", t0, 60.00, 60.01)]))
    st = r.subscribe(FakeSock())
    r.frame(st, (t0 + 1_000) / 1000)
    # Both sides up a cent: spread unchanged, window unchanged, nothing sent.
    h._ingest(json.dumps([quote("FDX", t0 + 5_000, 334.38, 334.45)]))
    f = r.frame(st, (t0 + 6_000) / 1000)
    check(f["syms"] == {},
          f"a quote that did not change the window was sent: {f['syms']}")
    check("at" in f, "an empty frame must still go out, stamped")
    # KO widens to 3c for a second: p90 moves, the rest does not yet.
    h._ingest(json.dumps([quote("KO", t0 + 9_000, 60.00, 60.03)]))
    f = r.frame(st, (t0 + 10_000) / 1000)
    check(f["syms"] == {"KO": [1, 1, 3]},
          f"KO's changed window came as {f['syms']}")
    # Two sockets keep their own record of what each was sent.
    st2 = r.subscribe(FakeSock())
    check(r.frame(st2, (t0 + 10_000) / 1000)["syms"]
          == {"FDX": [7, 7, 7], "KO": [1, 1, 3]},
          "a second page did not get every row on its first frame")
    # The windows are computed ONCE per tick, whatever the page count.
    calls = [0]
    real = WallSym.window

    def counted(self, now_ms):
        calls[0] += 1
        return real(self, now_ms)
    WallSym.window = counted
    try:
        r.subscribe(FakeSock())
        await r._tick()
    finally:
        WallSym.window = real
    check(calls[0] == 2,
          f"one tick over 2 symbols and 3 pages computed {calls[0]} windows")


async def case_one_socket_carries_300_rows(tmp: Path):
    """300 symbols, one socket, one frame each tick, built quickly."""
    h = recording_hub()
    r = runner(h, tmp)
    syms = [f"S{i:03d}" for i in range(config.WALL_MAX_SYMBOLS)]
    await r.apply([{"symbol": s} for s in syms])
    # Thirty seconds of a busy feed: every symbol requotes every 50 ms, so
    # every bucket of every window holds a value, and they differ.
    t_feed = time.time() * 1000 - 31_000
    for k in range(0, 31_000, 50):
        h._ingest(json.dumps([quote(s, t_feed + k, 10.0,
                                    10.0 + ((i + k // 50) % 90) / 100)
                              for i, s in enumerate(syms)]))
    check(all(w.filled == config.WALL_BUCKETS for w in h.wall.values()),
          "thirty seconds of quotes did not fill every window")
    sock = FakeSock()
    st = r.subscribe(sock)
    t0 = time.perf_counter()
    for _ in range(20):
        wins = r.windows()
    per_win = (time.perf_counter() - t0) / 20 * 1000
    t0 = time.perf_counter()
    for _ in range(40):
        r.frame(st, None, wins)
    per = (time.perf_counter() - t0) / 40 * 1000
    await r._tick()
    check(len(sock.frames) == 1, f"one tick sent {len(sock.frames)} frames")
    check(per_win < 100 and per < 5,
          f"300 windows took {per_win:.2f} ms, a frame {per:.2f} ms")
    check(config.WALL_TICK_S <= 2.0,
          f"the tick is {config.WALL_TICK_S}s; slower than 2 s and a name "
          f"that has just opened up is late")
    one = json.dumps({"S000": wins["S000"]}, separators=(",", ":"))[1:-1]
    print(f"  at 300 full windows: windows {per_win:.2f} ms a tick, a frame "
          f"{per:.3f} ms a page; one symbol on the wire {len(one)} bytes "
          f"({one}); tick {config.WALL_TICK_S}s")


# ── the window ─────────────────────────────────────────────────────────────
def bucket_aligned(offset_ms: float) -> float:
    """A time `offset_ms` from now, on a bucket boundary (in the past for a
    negative offset -- push_quote clamps a quote to its arrival time)."""
    b = config.WALL_BUCKET_MS
    return ((time.time() * 1000 + offset_ms) // b) * b


def reference_window(quotes, now_ms):
    """Brute force: the step function sampled at the END of every bucket of
    the window, nearest-rank percentiles. `quotes` is [(t_ms, cents)] in
    arrival order, each arriving at its own timestamp. A quote never lands in
    a bucket before one already written."""
    import math
    B, N = config.WALL_BUCKET_MS, config.WALL_BUCKETS
    if not quotes:
        return None
    placed, last_b = [], None
    for t, sp in quotes:
        qb = int(t // B)
        qb = qb if last_b is None else max(qb, last_b)
        last_b = qb
        placed.append((qb, sp))
    now_b = max(int(now_ms // B), last_b)
    vals = []
    for k in range(max(placed[0][0], now_b - N + 1), now_b + 1):
        standing = None
        for qb, sp in placed:
            if qb <= k:
                standing = sp
        vals.append(standing)
    vals.sort()

    def rank(p):
        return vals[max(1, math.ceil(p / 100 * len(vals) - 1e-9)) - 1]
    return [rank(config.WALL_PCT_LO), rank(50.0), rank(config.WALL_PCT_HI)]


def fed(quotes, now_ms) -> list | None:
    w = WallSym()
    for t, sp in quotes:
        w.push_quote(t, 100.0, 100.0 + sp / 100.0, now_ms=t)
    out = w.window(now_ms)
    return list(out) if out is not None else None


async def case_window_constants():
    """30 s of 100 ms buckets, p10/p90, a 2 s tick -- single-sourced, sent."""
    check((config.WALL_WINDOW_S, config.WALL_BUCKET_MS, config.WALL_BUCKETS,
           config.WALL_PCT_LO, config.WALL_PCT_HI, config.WALL_TICK_S)
          == (30, 100, 300, 10, 90, 2.0),
          "the window is not 30 s of 100 ms buckets, p10/p90, a 2 s tick")
    caps = WallRunner(recording_hub(), WallStore(Path("_unused.json"))).state()["caps"]
    check(caps.get("window_s") == 30 and caps.get("pct_lo") == 10
          and caps.get("pct_hi") == 90 and caps.get("tick_s") == 2.0,
          f"the page is not told the window: {caps}")
    js = JS.read_text(encoding="utf-8")
    check("WL_RESORT_MS" not in js and "sortedAt" not in js,
          "widest first still keeps a timer of its own")


async def case_window_carries_forward():
    """One quote, then twenty seconds of silence: every bucket holds it."""
    t0 = bucket_aligned(-600_000)
    w = WallSym()
    w.push_quote(t0, 10.00, 10.55, now_ms=t0)
    win = w.window(t0 + 20_000)
    check(list(win) == [55, 55, 55] and w.filled == 201,
          f"a name quoting once and sitting still read {win} over "
          f"{w.filled} buckets; it is at 55c for every one of them")
    win = w.window(t0 + 120_000)
    check(list(win) == [55, 55, 55] and w.filled == config.WALL_BUCKETS,
          f"two minutes later it read {win}: a standing quote never ages out")
    check(len(w.ring) == config.WALL_BUCKETS, f"the ring grew to {len(w.ring)}")
    check(WallSym().window(t0) is None, "a symbol with no quote has a window")


async def case_window_is_time_weighted():
    """The user's case: 20c most of the time, gapping to 65c a few seconds in
    every thirty. Tight requotes furiously; wide stands still."""
    t0 = bucket_aligned(-600_000)
    quotes = [(t0 + k, 20) for k in range(0, 26_000, 20)]   # 1,300 at 20c
    quotes.append((t0 + 26_000, 65))                         # ONE, 4 s at 65c
    now = t0 + 29_999
    got = fed(quotes, now)
    check(got == [20, 20, 65],
          f"20c requoting while tight, 65c standing for 4 s of 30, read "
          f"{got}; p90 must see the 65c stretch at its full length (counted "
          f"per quote it would read 20)")
    # A flat 55 and an oscillating 20/57 are told apart by the bar.
    flat = fed([(t0, 55)], now)
    osc = fed([(t0 + k, 57 if (k // 1000) % 3 == 0 else 20)
               for k in range(0, 30_000, 1000)], now)
    check(flat == [55, 55, 55] and osc[0] == 20 and osc[1] == 20
          and osc[2] == 57,
          f"flat 55 read {flat}, 20c gapping to 57c read {osc}")
    # The wide stretch leaves the window thirty seconds later.
    later = fed(quotes + [(t0 + 30_000, 20)], t0 + 61_000)
    check(later == [20, 20, 20],
          f"31 s after the gap closed the window still read {later}")


async def case_window_clock_is_clamped():
    """A quote stamped ahead of this box goes in now's bucket; an older one
    updates the newest bucket -- buckets never go backwards."""
    B = config.WALL_BUCKET_MS
    t0 = bucket_aligned(-600_000)
    w = WallSym()
    w.push_quote(t0, 10.00, 10.20, now_ms=t0)
    w.push_quote(t0 + 5_000, 10.00, 10.80, now_ms=t0 + 1_000)   # skewed ahead
    check(w.b_last == int((t0 + 1_000) // B),
          "a quote stamped in the future was placed in the future")
    w.push_quote(t0 + 200, 10.00, 10.40, now_ms=t0 + 1_050)     # arrives late
    check(w.b_last == int((t0 + 1_000) // B) and w.cur == 40
          and w.ring[w.b_last % config.WALL_BUCKETS] == 40,
          "a late quote did not become the newest bucket's standing spread")
    win = w.window(t0 + 1_999)
    check(list(win) == [20, 20, 40] and w.filled == 20,
          f"the window read {win} over {w.filled} buckets")


async def case_window_matches_reference():
    """Random feeds -- bursts inside one bucket, silences, gaps longer than
    the window -- held to the brute-force sampler at random read times."""
    import random
    rng = random.Random(11)
    bad, reads = [], 0
    for trial in range(150):
        t = bucket_aligned(-3_600_000) + rng.randrange(100)
        quotes, w = [], WallSym()
        for _ in range(rng.randrange(1, 120)):
            t += rng.choice((1, 7, 40, 99, 100, 101, 350, 2_000, 9_000, 45_000))
            sp = rng.choice((1, 2, 3, 5, 8, 13, 20, 35, 57, 80, 120, 0.5))
            w.push_quote(t, 50.0, 50.0 + sp / 100.0, now_ms=t)
            quotes.append((t, sp))
            if rng.random() < 0.2:
                now = t + rng.randrange(0, 40_000)
                got, want = list(w.window(now)), reference_window(quotes, now)
                reads += 1
                if got != want:
                    bad.append((trial, got, want))
                t = max(t, now)
        now = t + rng.randrange(0, 5_000)
        got, want = list(w.window(now)), reference_window(quotes, now)
        reads += 1
        if got != want:
            bad.append((trial, got, want))
    check(reads > 300, f"only {reads} windows were compared")
    check(not bad, f"{len(bad)} of {reads} windows differ from the reference, "
                   f"e.g. {bad[:3]}")


# ── the watchlist ───────────────────────────────────────────────────────────
async def case_watchlist_round_trips(tmp: Path):
    """The list and both thresholds survive a restart; old fields ride along."""
    path = tmp / "wl.json"
    s = WallStore(path)
    s.set([{"symbol": "fdx"}, {"symbol": "LLY", "scale": 0.35},
           {"symbol": "LLY"}, "NVDA"],
          {"min_spread_cents": 45, "wide_cents": 150})
    s.save()
    back = WallStore(path)
    n = back.load()
    check(n == 3 and back.symbols() == ["FDX", "LLY", "NVDA"],
          f"{back.symbols()} reloaded")
    check(back.settings == {"min_spread_cents": 45, "wide_cents": 150},
          f"the thresholds came back {back.settings}")
    check({e["symbol"]: e["scale"] for e in back.entries}["LLY"] == 0.35,
          "an old per-pane scale was dropped on save instead of carried")


async def case_thresholds_clamp_and_default(tmp: Path):
    """Floor 10-120 (default 59, a saved 0 reads as 59); wide 40-300 on 5s."""
    s = WallStore(tmp / "c.json")
    check(s.settings == {"min_spread_cents": 59.0, "wide_cents": 120.0},
          f"the defaults are {s.settings}")
    cases = [({"min_spread_cents": 0}, "min_spread_cents", 59.0),
             ({"min_spread_cents": 4}, "min_spread_cents", 10.0),
             ({"min_spread_cents": 999}, "min_spread_cents", 120.0),
             ({"min_spread_cents": 72}, "min_spread_cents", 72.0),
             ({"wide_cents": 12}, "wide_cents", 40.0),
             ({"wide_cents": 9999}, "wide_cents", 300.0),
             ({"wide_cents": 137}, "wide_cents", 135.0),
             ({"wide_cents": 250}, "wide_cents", 250.0)]
    for settings, key, want in cases:
        s.set([{"symbol": "FDX"}], settings)
        check(s.settings[key] == want,
              f"{settings} was stored as {s.settings[key]}, want {want}")
    s.set([{"symbol": "FDX"}], {"wide_cents": "wide"})
    check(s.settings["wide_cents"] == 250.0,
          "an unparseable value moved the setting instead of leaving it")
    # A file from the tape wall: floor 0 ("off"), window and share settings.
    old = tmp / "old.json"
    old.write_text(json.dumps({"version": 1, "settings": {
        "window_s": 120, "spread_share": 0.6, "min_spread_cents": 0},
        "entries": [{"symbol": "FDX", "scale": 0.4}]}), encoding="utf-8")
    o = WallStore(old)
    check(o.load() == 1 and o.settings == {"min_spread_cents": 59.0,
                                           "wide_cents": 120.0},
          f"the old file loaded as {o.settings}")

    # The cap holds, and says so.
    s2 = WallStore(tmp / "cap.json")
    ref2 = s2.set([{"symbol": f"S{i:04d}"}
                   for i in range(config.WALL_MAX_SYMBOLS + 5)])
    check(len(s2.entries) == config.WALL_MAX_SYMBOLS and len(ref2) == 5,
          f"the cap let {len(s2.entries)} through with {len(ref2)} refusals")
    refused = WallStore(tmp / "r.json").set([{"symbol": "BRK B"}, {"symbol": ""}])
    check(len(refused) == 2 and any("BRK B" in r for r in refused),
          f"refusals {refused}; a refused ticker must be named")


async def case_sliders_match_the_clamps():
    """Each slider's range and step is the server's, or a value snaps back."""
    tpl = TPL.read_text(encoding="utf-8")

    def slider(model):
        for tag in re.findall(r"<input[^>]*>", tpl, re.S):
            if f'x-model.number="settings.{model}"' in tag:
                return {k: float(re.search(rf'{k}="([\d.]+)"', tag).group(1))
                        for k in ("min", "max", "step")}
        return None

    fl, wd = slider("min_spread_cents"), slider("wide_cents")
    check(fl == {"min": config.WALL_FLOOR_MIN, "max": config.WALL_FLOOR_MAX,
                 "step": 1.0},
          f"the floor slider is {fl}")
    check(wd == {"min": config.WALL_WIDE_MIN, "max": config.WALL_WIDE_MAX,
                 "step": config.WALL_WIDE_STEP},
          f"the wide slider is {wd}")
    check((config.WALL_FLOOR_CENTS, config.WALL_FLOOR_MIN,
           config.WALL_FLOOR_MAX) == (59, 10, 120)
          and (config.WALL_WIDE_CENTS, config.WALL_WIDE_MIN,
               config.WALL_WIDE_MAX, config.WALL_WIDE_STEP) == (120, 40, 300, 5),
          "the thresholds' defaults and ranges are not the specified ones")


async def case_thresholds_do_not_touch_subscriptions(tmp: Path):
    """Moving either threshold changes nothing the hub holds or sends."""
    h = recording_hub()
    r = runner(h, tmp)
    await r.apply([{"symbol": "FDX"}, {"symbol": "KO"}])
    h.sent.clear()
    out = await r.apply(r.store.entries,
                        {"min_spread_cents": 80, "wide_cents": 200})
    check(sorted(h.wall) == ["FDX", "KO"] and not out["added"]
          and not out["dropped"] and h.sent == [],
          f"a threshold change reached the tier: {params(h)!r}")
    back = WallStore(tmp / "wall_watchlist.json")
    back.load()
    check(back.settings == {"min_spread_cents": 80, "wide_cents": 200},
          f"the thresholds did not survive a reload: {back.settings}")


async def case_a_corrupt_watchlist_is_named(tmp: Path):
    """Half a file, or a hand-edited one, must not read as an empty list."""
    path = tmp / "bad.json"
    path.write_text('{"entries": [{"symbol": "FDX"}', encoding="utf-8")
    s = WallStore(path)
    check(s.load() == 0 and s.entries == [] and s.last_error,
          "a corrupt watchlist was swallowed or loaded entries")
    s2 = WallStore(tmp / "absent.json")
    check(s2.load() == 0 and s2.last_error is None,
          f"a missing file was reported as an error: {s2.last_error}")


async def case_apply_keeps_hub_file_and_pages_in_step(tmp: Path):
    """One call sets the tier, the file and every open page."""
    h = recording_hub()
    r = runner(h, tmp)
    a, b = FakeSock(), FakeSock()
    r.subscribe(a)
    r.subscribe(b)
    out = await r.apply([{"symbol": "FDX"}, {"symbol": "LLY"}],
                        {"wide_cents": 150})
    check(sorted(h.wall) == ["FDX", "LLY"], f"the hub holds {sorted(h.wall)}")
    saved = json.loads((tmp / "wall_watchlist.json").read_text(encoding="utf-8"))
    check([e["symbol"] for e in saved["entries"]] == ["FDX", "LLY"]
          and saved["settings"]["wide_cents"] == 150,
          f"the file holds {saved}")
    for name, sock in (("the editing page", a), ("the OTHER page", b)):
        check("watchlist" in [f["ev"] for f in sock.frames],
              f"{name} was not told the list changed")
    check(out["added"] == ["FDX", "LLY"] and not out["refused"],
          f"apply reported {out['added']} / {out['refused']}")
    st = r._subs[a]
    r.frame(st)
    await r.apply([{"symbol": "FDX"}])
    check("LLY" not in st["sent"],
          f"a dropped symbol left its last-sent spread: {list(st['sent'])}")


async def case_restore_holds_the_saved_list(tmp: Path):
    """Startup: the file is read and held, quotes only, in one message."""
    path = tmp / "restore.json"
    s = WallStore(path)
    s.set([{"symbol": "FDX"}, {"symbol": "LLY"}])
    s.save()
    h = recording_hub()
    r = WallRunner(h, WallStore(path))
    n = await r.restore()
    check(n == 2 and sorted(h.wall) == ["FDX", "LLY"],
          f"restore loaded {n} and held {sorted(h.wall)}")
    check(sent(h, "subscribe") == ["Q.FDX,Q.LLY"],
          f"restore subscribed {sent(h, 'subscribe')}")


async def case_no_paging_left():
    """WALL_PER_PAGE and the pager are gone, server and page."""
    check(not hasattr(config, "WALL_PER_PAGE"), "WALL_PER_PAGE is still set")
    for rel in ("live/wall_runner.py", "static/js/equities_wall.js",
                "templates/equities_wall.html"):
        text = (ROOT / rel).read_text(encoding="utf-8")
        for tok in ("per_page", "setPage", "pageEntries", "pageCount"):
            check(tok not in text, f"{rel} still has {tok}")


# ── what this page is not ───────────────────────────────────────────────────
async def case_the_wall_never_talks_to_a_broker():
    """It is for watching. Parsed, not grepped: an import, a name or an
    attribute mentioning the broker, in the wall's modules and routes."""
    import ast

    def broker_refs(tree) -> list[str]:
        out = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                out += [a.name for a in node.names if "broker" in a.name]
            elif isinstance(node, ast.ImportFrom):
                mod = node.module or ""
                if "broker" in mod:
                    out.append(mod)
                out += [f"{mod}.{a.name}" for a in node.names
                        if "broker" in a.name.lower()]
            elif isinstance(node, ast.Name) and "broker" in node.id.lower():
                out.append(node.id)
            elif isinstance(node, ast.Attribute) and "broker" in node.attr.lower():
                out.append(node.attr)
        return out

    for name in ("wall.py", "wall_runner.py", "wall_store.py"):
        tree = ast.parse((ROOT / "live" / name).read_text(encoding="utf-8"))
        refs = broker_refs(tree)
        check(not refs, f"live/{name} references {sorted(set(refs))}")
    main = ast.parse((ROOT / "live" / "main.py").read_text(encoding="utf-8"))
    seen = []
    for node in ast.walk(main):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) \
                and node.name.startswith("wall_"):
            seen.append(node.name)
            refs = broker_refs(node)
            check(not refs, f"live/main.py:{node.name}() references "
                            f"{sorted(set(refs))}")
    check(sorted(seen) == ["wall_page", "wall_watchlist", "wall_watchlist_set",
                           "wall_ws"],
          f"the wall's endpoints are {sorted(seen)}; this check scans routes "
          f"by name and a renamed one would be scanned by nothing")


async def case_the_page_cannot_trade():
    """Nothing on the page reaches a broker -- the markup included."""
    for rel in ("static/js/equities_wall.js", "templates/equities_wall.html"):
        text = (ROOT / rel).read_text(encoding="utf-8").lower()
        for tok in ("/broker", "brokercall", "armed", "flatten",
                    "sendorder", "placeorder", "sendmove", "ladder"):
            check(tok not in text, f"{rel} mentions {tok!r}")


# ── the page, executed ──────────────────────────────────────────────────────
JS_DRIVER = r"""
const fs = require('fs');
global.document = { addEventListener: () => {} };
const src = fs.readFileSync(process.argv[1], 'utf8');
const tail = `
  const out = {};
  out.tiers = [wlTier(null, 59, 120), wlTier(58, 59, 120), wlTier(59, 59, 120),
               wlTier(119, 59, 120), wlTier(120, 59, 120), wlTier(250, 59, 120),
               wlTier(100, 150, 120)];
  out.bars = [wlBarPct(60, 120), wlBarPct(120, 120), wlBarPct(300, 120),
              wlBarPct(null, 120), wlBarPct(60, 200)];
  out.hi = [wlHi([20, 35, 57]), wlHi(null), wlHi([1, 2])];
  out.range = [wlRange([30, 45, 60], 120), wlRange([55, 55, 55], 110),
               wlRange([100, 200, 300], 120), wlRange(null, 120)];
  const sp = {MMM: [10, 20, 70], AAPL: [12, 12, 12], ZTS: [5, 6, 130], KO: null,
              BA: [60, 65, 70]};
  out.alpha = wlOrder(['ZTS', 'MMM', 'AAPL', 'KO', 'BA'], sp, false);
  out.widest = wlOrder(['ZTS', 'MMM', 'AAPL', 'KO', 'BA'], sp, true);
  out.shown = [wlShown('below', false, false), wlShown('below', true, false),
               wlShown('below', true, true), wlShown('none', true, false),
               wlShown('mid', true, false), wlShown('wide', true, false)];
  out.parse = wlParseSymbols('fdx, lly\\nNVDA fdx  BRK.B');
  out.fmt = [wlFmtSpread(7), wlFmtSpread(59.6), wlFmtSpread(null)];
  globalThis.__out = out;
`;
eval(src + tail);
process.stdout.write(JSON.stringify(globalThis.__out));
"""


async def case_js_rows():
    """The shipped functions: colour, bar length, order, hiding."""
    if shutil.which("node") is None:
        FAILS.append("node is not installed -- the shipped JS was NOT executed")
        return
    p = subprocess.run(["node", "-e", JS_DRIVER, str(JS)],
                       capture_output=True, text=True, encoding="utf-8")
    if p.returncode:
        FAILS.append(f"the page JS did not run: {p.stderr.strip()[:300]}")
        return
    o = json.loads(p.stdout)
    check(o["tiers"] == ["none", "below", "mid", "mid", "wide", "wide", "below"],
          f"tiers {o['tiers']}: at/above wide pink, floor..wide blue, below "
          f"floor dim (and below the floor wins when it is set above wide)")
    check(o["bars"] == [50, 100, 100, 0, 30],
          f"bars {o['bars']}: length is spread / wide, full at wide")
    check(o["alpha"] == ["AAPL", "BA", "KO", "MMM", "ZTS"],
          f"alphabetical is {o['alpha']}")
    check(o["widest"] == ["ZTS", "BA", "MMM", "AAPL", "KO"],
          f"widest first is {o['widest']} (ties alphabetical, unquoted last)")
    check(o["shown"] == [True, False, True, False, True, True],
          f"hide-below-floor gives {o['shown']}; a marked row always shows")
    check(o["parse"] == ["FDX", "LLY", "NVDA"], f"parsed {o['parse']}")
    check(o["fmt"] == ["7.0", "60", "—"], f"formatted {o['fmt']}")
    check(o["hi"] == [57, None, None], f"p90 read as {o['hi']}")
    r0, r1, r2, r3 = o["range"]
    check(r0 == {"lo": 25, "width": 25, "mid": 37.5}
          and r1 == {"lo": 50, "width": 0, "mid": 50}
          and abs(r2["lo"] - 100 / 1.2) < 1e-9 and abs(r2["width"] - (100 - 100 / 1.2)) < 1e-9
          and r2["mid"] == 100 and r3 is None,
          f"range bars {o['range']}: p10..p90 on the wide scale, median between")


async def case_the_page_markup():
    """Columns flow down, rows are compact, nothing else is on a row."""
    html = TPL.read_text(encoding="utf-8")
    js = JS.read_text(encoding="utf-8")
    css = html.split("</style>")[0]
    lst = re.search(r"\.wl-list\s*\{([^}]*)\}", css)
    check(lst and "column-width:250px" in lst.group(1).replace(" ", ""),
          "the list is not a multi-column flow with ~250px columns")
    check(lst and "grid" not in lst.group(1),
          "the list container is a grid, which fills across, not down")
    row = re.search(r"\.wl-row\s*\{([^}]*)\}", css)
    check(row and "break-inside:avoid" in row.group(1).replace(" ", "")
          and "height:16px" in row.group(1).replace(" ", "")
          and "tabular-nums" in row.group(1) and "monospace" in row.group(1),
          "a row is not 16px, monospace, tabular and unbroken across columns")
    tmpl = re.search(r'<template x-for="sym in order".*?</template>', html, re.S)
    check(tmpl is not None, "the list does not iterate `order`")
    if tmpl:
        spans = re.findall(r'class="(wl-[a-z]+)"', tmpl.group(0))
        check(spans == ["wl-row", "wl-tk", "wl-track", "wl-fill", "wl-med",
                        "wl-cents"],
              f"a row holds {spans}; ticker, range bar with its median "
              f"tick, and cents, nothing else")
        check('@click="toggleMark(sym)"' in tmpl.group(0),
              "a row click does not toggle the mark")
    check("<canvas" not in html and "getContext" not in js,
          "the canvases are still on the page")
    check("widest: false" in js and "hideBelow: false" in js,
          "the page does not open alphabetical with hide-below off")
    # MARKS ARE MEMORY ONLY.
    code = "\n".join(ln for ln in js.splitlines()
                     if not ln.lstrip().startswith(("*", "/*", "//")))
    check("localStorage" not in code and "sessionStorage" not in code
          and "indexedDB" not in code,
          "the page writes to browser storage; marks are this session only")
    check(code.count("new WebSocket") == 1,
          f"the page constructs {code.count('new WebSocket')} WebSockets")
    check("TAPE_BLUE" in js and "TAPE_PINK" in js
          and html.find("tape_theme.js") < html.find("equities_wall.js"),
          "the bars do not take the tape's blue and pink from tape_theme.js "
          "(loaded first)")
    main = (ROOT / "live" / "main.py").read_text(encoding="utf-8")
    check('"equities_wall.html"' in main and '@app.get("/wall"' in main,
          "no route serves the page")
    check('x-data="equitiesWall"' in html and "defer src={{ asset" not in html,
          "the component is missing, or its bundle is deferred")
    # Initialised once (Alpine calls init() itself).
    body = "\n".join(ln for ln in html.splitlines()
                     if "NO `x-init" not in ln and not ln.lstrip().startswith("#"))
    check('x-init="init()"' not in body,
          "init() is named in x-init as well: it runs twice, two sockets")
    nav = (ROOT / "templates" / "_nav.html").read_text(encoding="utf-8")
    check("/wall" in nav, "the wall is not on the nav")


CASES = [
    ("wall is quotes only",       case_wall_subscribes_quotes_only),
    ("tiers keep channels",       case_tiers_keep_each_others_channels),
    ("reconnect per channel",     case_reconnect_restores_each_channel),
    ("trades ignored",            case_wall_takes_quotes_and_ignores_trades),
    ("every quote taken",         case_every_quote_is_taken),
    ("crossed quotes dropped",    case_a_crossed_quote_is_dropped_and_counted),
    ("first frame is all",        case_first_frame_is_every_symbol),
    ("window constants",          case_window_constants),
    ("window carries forward",    case_window_carries_forward),
    ("window time-weighted",      case_window_is_time_weighted),
    ("window clock clamped",      case_window_clock_is_clamped),
    ("window = reference",        case_window_matches_reference),
    ("frame is a delta",          case_frame_is_a_delta_of_windows),
    ("one socket, 300 rows",      case_one_socket_carries_300_rows),
    ("watchlist round-trips",     case_watchlist_round_trips),
    ("thresholds clamp",          case_thresholds_clamp_and_default),
    ("sliders match clamps",      case_sliders_match_the_clamps),
    ("thresholds spare the tier", case_thresholds_do_not_touch_subscriptions),
    ("corrupt list is named",     case_a_corrupt_watchlist_is_named),
    ("apply keeps all in step",   case_apply_keeps_hub_file_and_pages_in_step),
    ("restore holds the list",    case_restore_holds_the_saved_list),
    ("no paging left",            case_no_paging_left),
    ("never talks to a broker",   case_the_wall_never_talks_to_a_broker),
    ("the page cannot trade",     case_the_page_cannot_trade),
    ("rows, executed",            case_js_rows),
    ("the page markup",           case_the_page_markup),
]


async def main() -> int:
    import inspect
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        for name, fn in CASES:
            before = len(FAILS)
            try:
                if "tmp" in inspect.signature(fn).parameters:
                    await fn(tmp)
                else:
                    await fn()
            except Exception as exc:                      # noqa: BLE001
                FAILS.append(f"raised {type(exc).__name__}: {exc}")
            for m in FAILS[before:]:
                print(f"  FAIL {name}: {m}")
    print(f"\nwall cases: {len(CASES)}, failures: {len(FAILS)}")
    return 1 if FAILS else 0


sys.exit(asyncio.run(main()))
