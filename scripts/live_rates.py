"""Message rate per tier on the running spx-live, from two /status readings.

READ-ONLY. Two GETs of /status, N seconds apart, and arithmetic on the
counters. It opens no market-data socket, so it cannot evict the service, and
it changes nothing -- safe to run on the VPS during the session:

    .venv/bin/python scripts/live_rates.py            # 30 s sample
    .venv/bin/python scripts/live_rates.py --secs 60 --url http://127.0.0.1:8001

The per-tier figures are ARRIVALS for symbols that tier holds, counted before
any sampling, so a symbol two tiers hold counts under both and the tiers do not
sum to the total. Each answers "what would this tier cost on its own".
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.request


def status(url: str) -> dict:
    with urllib.request.urlopen(url.rstrip("/") + "/status", timeout=10) as r:
        return json.load(r)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8001")
    ap.add_argument("--secs", type=float, default=30.0)
    a = ap.parse_args()

    s0, t0 = status(a.url), time.monotonic()
    if "tier_in" not in s0:
        print("this spx-live has no per-tier counters yet -- it is still on "
              "the code from before they were added; restart it (after the "
              "close) and run this again.")
        return 1
    time.sleep(a.secs)
    s1, t1 = status(a.url), time.monotonic()
    dt = t1 - t0

    held = {"pane": len(s1.get("symbols") or []),
            "scan": (s1.get("scan") or {}).get("held", 0),
            "wall": (s1.get("wall") or {}).get("held", 0)}
    print(f"over {dt:.1f} s   connected={s1.get('connected')} "
          f"authed={s1.get('authed')} feed={s1.get('feed')}")
    print(f"{'tier':6} {'held':>5} {'trades/s':>10} {'quotes/s':>10} {'total/s':>10}")
    for tier in ("pane", "scan", "wall"):
        a0, a1 = s0["tier_in"][tier], s1["tier_in"][tier]
        tr = (a1["trades"] - a0["trades"]) / dt
        qt = (a1["quotes"] - a0["quotes"]) / dt
        print(f"{tier:6} {held[tier]:>5} {tr:>10.1f} {qt:>10.1f} {tr + qt:>10.1f}")
    total = (s1["msgs_in"] - s0["msgs_in"]) / dt
    print(f"{'socket':6} {'':>5} {'':>10} {'':>10} {total:>10.1f}   "
          f"(every message on the upstream socket, status lines included)")

    pins = s1.get("pinned") or []
    print(f"\npane symbols: {', '.join(s1.get('symbols') or []) or 'none'}"
          f"   pinned: {', '.join(pins) or 'none'}")
    sr = s1.get("scan_runner") or {}
    print(f"scan: {sr.get('subscribers', 0)} page(s) open, {sr.get('held', 0)} held, "
          f"{sr.get('wanted', 0)} remembered, release pending={sr.get('release_pending')}, "
          f"releases={sr.get('releases')}")
    w = s1.get("wall") or {}
    print(f"wall: {w.get('held')} of {w.get('cap')} held, "
          f"{w.get('buffer_mb', 0):.1f} MB buffered, "
          f"{w.get('truncated_count', 0)} truncated, "
          f"{len(w.get('quoteless') or [])} never quoted"
          + (f" ({', '.join(w['truncated'])})" if w.get("truncated") else ""))
    wr = s1.get("wall_runner") or {}
    print(f"wall pages open: {wr.get('subscribers')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
