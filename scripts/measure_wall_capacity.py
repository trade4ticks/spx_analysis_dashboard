"""What the wall tier costs the live service at N symbols, measured offline.

READ-ONLY AND OFFLINE: no market-data socket is opened (the account permits
one, and a second evicts spx-live), nothing is written. It drives the SHIPPED
Hub._ingest and WallRunner.frame with a synthetic feed, so it can be run here
or on the VPS during the session:

    .venv/bin/python scripts/measure_wall_capacity.py              # 300 symbols
    .venv/bin/python scripts/measure_wall_capacity.py --symbols 220 --scale 2

QUOTES ONLY (2026-11-06): the wall subscribes the Q channel and nothing
else, so the feed simulated here is quotes. A trade for a wall symbol arrives
only when a pane or the scan holds it too, and is that tier's cost.

THE RATE IS AN UPPER BOUND FROM A MEASUREMENT, not a guess at a watchlist.
At the open on 2026-09-08 the scan's 600 symbols ran 3,850 trades/s, and with
quotes added the records were ~3.1x that. The busiest ~50 names carry most of
it, so a 300-name list is assumed to carry ALL of it -- quotes at 2.1x that
trade rate, ~8,100 a second -- as if the watchlist were the 300 busiest names
in the universe. A list of quiet names costs much less. Per-symbol rates are
Zipf-shaped so a few names are very busy, as measured.

THE SPREAD WINDOW (2026-11-09): every quote now also writes into the
symbol's 30 s ring of 100 ms buckets, and each tick sorts every window once.
The synthetic spreads JUMP (one to four times a base width, per quote) so the
windows actually move and the frames carry what a live list would.

What it reports, per simulated second at steady state (the window full):
  ingest   parsing and storing every upstream message (the hub's reader)
  frame    computing the windows and building one page's delta, every
           WALL_TICK_S, and their size (also per symbol sent)
and the share of one core both take at the stated rate. The VPS measured
about half this machine's speed (CLAUDE.md), so the VPS line doubles it.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from live import config                                   # noqa: E402
from live.hub import Hub                                  # noqa: E402
from live.wall_runner import WallRunner                   # noqa: E402
from live.wall_store import WallStore                     # noqa: E402

# Measured 2026-09-08 (see the module docstring), as an upper bound.
TRADES_PER_S = 3850.0
QUOTES_PER_TRADE = 2.1
# Upstream frames carry many messages each; the hub parses one frame at a time.
MSGS_PER_FRAME = 40
VPS_SLOWDOWN = 2.0


def build_second(syms, weights, n_t, n_q, t0_ms, px, rng):
    """One simulated second of upstream text frames, time-ordered."""
    msgs = []
    for _ in range(n_t):
        s = rng.choices(syms, weights)[0]
        px[s] *= 1 + rng.gauss(0, 0.0002)
        msgs.append((t0_ms + rng.random() * 1000.0,
                     {"ev": "T", "sym": s, "p": round(px[s], 2),
                      "s": rng.choice((1, 5, 10, 25, 100, 200, 500)),
                      "x": 4, "z": 1, "c": [12]}))
    for _ in range(n_q):
        s = rng.choices(syms, weights)[0]
        half = max(0.005, px[s] * 0.0004) * rng.choice((1, 1, 1, 2, 4))
        msgs.append((t0_ms + rng.random() * 1000.0,
                     {"ev": "Q", "sym": s, "bp": round(px[s] - half, 2),
                      "ap": round(px[s] + half, 2), "bs": 1, "as": 1,
                      "bx": 4, "ax": 4, "z": 1}))
    msgs.sort(key=lambda m: m[0])
    out = []
    for i in range(0, len(msgs), MSGS_PER_FRAME):
        chunk = []
        for t, m in msgs[i:i + MSGS_PER_FRAME]:
            m["t"] = int(t)
            chunk.append(m)
        out.append(json.dumps(chunk))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", type=int, default=300)
    ap.add_argument("--scale", type=float, default=1.0,
                    help="multiply the measured upstream rate")
    ap.add_argument("--seconds", type=int, default=30,
                    help="simulated seconds")
    ap.add_argument("--budget", type=float, default=0.50,
                    help="share of one VPS core the wall may use (PASS/OVER)")
    a = ap.parse_args()

    rng = random.Random(7)
    syms = [f"S{i:03d}" for i in range(a.symbols)]
    weights = [1.0 / (i + 1) ** 1.1 for i in range(a.symbols)]
    px = {s: rng.uniform(15, 400) for s in syms}
    n_t = 0                                 # the wall takes no trades
    n_q = int(TRADES_PER_S * QUOTES_PER_TRADE * a.scale)
    tick_ms = config.WALL_TICK_S * 1000.0

    config.WALL_MAX_SYMBOLS = max(config.WALL_MAX_SYMBOLS, a.symbols)
    hub = Hub()
    import asyncio
    asyncio.run(hub.wall_set(syms))
    runner = WallRunner(hub, WallStore(Path(".") / "_unused_wall.json"))
    st = runner.subscribe(object())

    t0 = time.time() * 1000.0 - a.seconds * 1000.0
    ingest_s, frame_s, frame_kb, n_msgs = [], [], [], 0
    per_sym, n_sent, next_tick = [], [], t0 + tick_ms
    for sec in range(a.seconds):
        frames = build_second(syms, weights, n_t, n_q, t0 + sec * 1000.0, px, rng)
        c0 = time.perf_counter()
        for f in frames:
            hub._ingest(f)
        ingest_s.append(time.perf_counter() - c0)
        # The second's feed is ingested whole, then every tick that falls in
        # it computes the windows and builds the page's delta.
        c0 = time.perf_counter()
        kb = 0.0
        end = t0 + (sec + 1) * 1000.0
        while next_tick <= end:
            f = runner.frame(st, next_tick / 1000.0)
            body = json.dumps(f, separators=(",", ":"))   # as Starlette sends it
            kb += len(body) / 1024.0
            if f["syms"]:
                per_sym.append((len(json.dumps(f["syms"], separators=(",", ":"))) - 2)
                               / len(f["syms"]))
            n_sent.append(len(f["syms"]))
            next_tick += tick_ms
        frame_s.append(time.perf_counter() - c0)
        frame_kb.append(kb)
        n_msgs = n_t + n_q

    steady = slice(-10, None)
    ing = sorted(ingest_s[steady])
    frm = sorted(frame_s[steady])
    kb = sorted(frame_kb[steady])
    med = lambda v: v[len(v) // 2]
    worst = lambda v: v[-1]
    core_here = med(ing) + med(frm)
    core_vps = core_here * VPS_SLOWDOWN
    ws = hub.wall_status()

    print(f"wall: {a.symbols} symbols, {n_q:,} quotes/s, no trades "
          f"({a.scale:g}x the measured upper bound), {a.seconds} s simulated")
    print(f"  ingest  median {med(ing) * 1000:7.1f} ms per second of feed  "
          f"(worst {worst(ing) * 1000:.1f}; {med(ing) / n_msgs * 1e6:.2f} us/msg)")
    sent_steady = sorted(n_sent[len(n_sent) // 2:])
    print(f"  frames  median {med(frm) * 1000:7.1f} ms per page per second    "
          f"(worst {worst(frm) * 1000:.1f}; one every {config.WALL_TICK_S:g} s), "
          f"{med(kb):.1f} KB a second (worst {worst(kb):.1f} KB)")
    print(f"  wire    {sum(per_sym) / max(1, len(per_sym)):.1f} bytes a symbol "
          f"sent; {med(sent_steady)} of {a.symbols} symbols changed a tick "
          f"(median, steady state)")
    print(f"  held    a {config.WALL_WINDOW_S} s window a symbol "
          f"({config.WALL_BUCKETS} buckets); {ws['quotes_seen']:,} taken, "
          f"{ws['crossed']} crossed")
    print(f"  share of one core: {core_here * 100:.0f}% here, "
          f"~{core_vps * 100:.0f}% on the VPS (x{VPS_SLOWDOWN:g}), one page open")
    verdict = "PASS" if core_vps <= a.budget else "OVER"
    print(f"{verdict}: budget {a.budget * 100:.0f}% of one VPS core")
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
