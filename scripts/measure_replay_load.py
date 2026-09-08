"""What does one session of trade_quote parquet cost to read?

STRICTLY READ-ONLY. Opens parquet files, times reads, computes an envelope in
memory. It touches no service, no database, and writes nothing anywhere.

The Replay chart's design turns on one number nobody has: how long a full
session takes to come off disk and be reduced to something a browser can
draw. The render half is already measured (scripts/bench_replay_render.html):
the zoomed view is free below ~20,000 trades, and at full session the cost is
the per-point loop rather than the rasterisation -- which is why the overview
has to be reduced SERVER-SIDE and the browser never sees 742k rows.

That leaves the read. This measures it against the heaviest files actually on
disk rather than an average one, because the worst case is what decides
whether the overview can be computed per request or has to be cached.

    python scripts/measure_replay_load.py
    python scripts/measure_replay_load.py --symbol AAPL --top 5
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np

# pyarrow is imported INSIDE main(), not here. The envelope below is the part
# with arithmetic worth checking, and a module-level parquet import means it
# cannot be exercised on a machine that only has numpy -- which is where the
# first version's bug would have been caught before it ran on the box.

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# The five columns a chart needs, out of the thirteen the vendor stores.
# Parquet is columnar, so this is the difference between reading a session and
# reading a third of one. Resolved against scalp/schema.py's candidate lists
# rather than hardcoded, because the vendor documents two naming schemes and
# the pipeline resolves rather than assumes.
WANT = {
    "time":  ["trade_timestamp", "timestamp", "ms_of_day", "time", "datetime"],
    "price": ["trade_price", "price", "last"],
    "size":  ["trade_size", "size", "quantity", "shares"],
    "bid":   ["bid", "bid_price", "nbbo_bid"],
    "ask":   ["ask", "ask_price", "nbbo_ask"],
}


def raw_dir() -> Path:
    d = os.environ.get("SCALP_DATA_DIR")
    if not d:
        print("SCALP_DATA_DIR is not set. It has no default on purpose -- see "
              "app/scalp_config.py.")
        raise SystemExit(2)
    return Path(d) / "raw"


def resolve(names) -> dict:
    lower = {c.lower(): c for c in names}
    out = {}
    for purpose, cands in WANT.items():
        for c in cands:
            if c.lower() in lower:
                out[purpose] = lower[c.lower()]
                break
    return out


def envelope(t, p, s, width: int = 1400) -> tuple:
    """The overview: per pixel column, the price span and the largest print.

    THE STRATEGY FOR THE FULL-SESSION VIEW ONLY. It is not a metric and never
    reaches the zoomed view, where every print is drawn at its own timestamp.
    Measured here because the render bench showed the cost at full session is
    this scan, not the drawing -- 1,400 columns rasterise in nothing while
    scanning 742k trades to build them costs 94 ms a frame in JS. Done once,
    server-side, in numpy, it should be a different number entirely.
    """
    lo, hi = t[0], t[-1]
    span = max(1e-9, hi - lo)
    col = np.minimum(((t - lo) / span * width).astype("int64"), width - 1)

    px_lo = np.full(width, np.nan)
    px_hi = np.full(width, np.nan)
    px_big = np.zeros(width)
    if t.size == 0:
        return px_lo, px_hi, px_big

    # REDUCEAT, NOT `np.minimum.at`. Two reasons, and the first is that the
    # obvious version was WRONG: `np.minimum.at` against an array seeded with
    # NaN stays NaN forever, because min(NaN, x) is NaN. The whole envelope
    # came back empty and the payload said "(0 non-empty)", which is the only
    # reason it was noticed -- it still produced a plausible 25 KB of nulls.
    #
    # The second is speed. `.at` is the unbuffered scatter-reduce and is slow
    # by design. Trades arrive in time order, so `col` is NON-DECREASING, and
    # that turns the whole thing into segment reductions: searchsorted for the
    # bin edges, then one reduceat per quantity. Empty bins between two
    # non-empty ones contribute no elements, so a segment running from one
    # non-empty start to the next holds exactly that bin's trades.
    if t.size > 1 and not np.all(np.diff(t) >= 0):
        order = np.argsort(t, kind="stable")
        t, p, s, col = t[order], p[order], s[order], col[order]

    edges = np.searchsorted(col, np.arange(width + 1), side="left")
    starts, ends = edges[:-1], edges[1:]
    live = np.flatnonzero(starts < ends)
    if live.size:
        idx = starts[live]
        px_lo[live] = np.minimum.reduceat(p, idx)
        px_hi[live] = np.maximum.reduceat(p, idx)
        px_big[live] = np.maximum.reduceat(s, idx)
    return px_lo, px_hi, px_big


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--symbol", default=None,
                    help="measure this symbol's heaviest day, not the store's")
    ap.add_argument("--top", type=int, default=3,
                    help="how many of the heaviest files to time")
    ap.add_argument("--repeat", type=int, default=3)
    args = ap.parse_args()

    import pyarrow.parquet as pq

    root = raw_dir()
    if not root.is_dir():
        print(f"no raw directory at {root}")
        return 2

    syms = sorted(p.name for p in root.iterdir() if p.is_dir())
    print(f"store: {root}")
    print(f"symbols: {len(syms)}")

    files = []
    for sym in ([args.symbol.upper()] if args.symbol else syms):
        d = root / sym
        if not d.is_dir():
            continue
        for f in d.glob("*.parquet"):
            try:
                date.fromisoformat(f.stem)
            except ValueError:
                continue
            files.append((f.stat().st_size, sym, f.stem, f))
    if not files:
        print("no parquet files found")
        return 1

    days = sorted({d for _, _, d, _ in files})
    total = sum(sz for sz, _, _, _ in files)
    print(f"sessions on disk: {len(days)}  ({days[0]} .. {days[-1]})")
    print(f"files: {len(files)}, {total / 1024**3:.2f} GB total, "
          f"{total / len(files) / 1024**2:.1f} MB mean")

    # PER-DAY COVERAGE, because the date picker lists what is actually there
    # and a day with three symbols is not a session anyone can replay.
    by_day = {}
    for _, sym, d, _ in files:
        by_day[d] = by_day.get(d, 0) + 1
    print("\nsymbols per session:")
    for d in days:
        print(f"  {d}  {by_day[d]:>4}")

    files.sort(reverse=True)
    print(f"\nheaviest {args.top} files:")
    for sz, sym, d, f in files[:args.top]:
        print(f"  {sym:6} {d}  {sz / 1024**2:7.1f} MB")

    print("\n" + "=" * 74)
    for sz, sym, d, f in files[:args.top]:
        pf = pq.ParquetFile(f)
        n_rows = pf.metadata.num_rows
        cols = [pf.schema_arrow.names[i]
                for i in range(len(pf.schema_arrow.names))]
        res = resolve(cols)
        missing = set(WANT) - set(res)
        print(f"\n{sym} {d} -- {n_rows:,} rows, {sz / 1024**2:.1f} MB on disk")
        if missing:
            print(f"  COULD NOT RESOLVE: {sorted(missing)}")
            print(f"  columns present: {cols}")
            continue
        print(f"  columns: {len(cols)} stored, using "
              f"{[res[k] for k in ('time', 'price', 'size', 'bid', 'ask')]}")

        def timed(fn, label):
            best = float("inf")
            for _ in range(args.repeat):
                t0 = time.perf_counter()
                out = fn()
                best = min(best, time.perf_counter() - t0)
            print(f"  {label:<34} {best * 1000:8.1f} ms")
            return out, best

        _, t_all = timed(lambda: pq.read_table(f), "read ALL columns")
        tbl, t_five = timed(
            lambda: pq.read_table(f, columns=[res[k] for k in WANT]),
            "read the 5 the chart needs")
        print(f"  {'saving from column selection':<34} "
              f"{(1 - t_five / t_all) * 100:7.0f}%")

        # To numpy, which is what the envelope and the wire format need.
        def to_arrays():
            t = tbl.column(res["time"]).to_numpy(zero_copy_only=False)
            if not np.issubdtype(np.asarray(t[:1]).dtype, np.number):
                t = tbl.column(res["time"]).cast("int64").to_numpy()
            t = np.asarray(t, dtype="float64")
            p = np.asarray(tbl.column(res["price"]).to_numpy(
                zero_copy_only=False), dtype="float64")
            s = np.asarray(tbl.column(res["size"]).to_numpy(
                zero_copy_only=False), dtype="float64")
            return t, p, s

        (t, p, s), t_np = timed(to_arrays, "to numpy arrays")
        _, t_env = timed(lambda: envelope(t, p, s), "envelope over the session")

        print(f"  {'READ + ENVELOPE, cold path':<34} "
              f"{(t_five + t_np + t_env) * 1000:8.1f} ms")

        # WHAT ACTUALLY GOES ON THE WIRE. The overview is ~1,400 columns; a
        # zoomed window is every trade in it. Both are what the browser gets,
        # and neither is the session.
        px_lo, px_hi, px_big = envelope(t, p, s)
        finite = int(np.isfinite(px_lo).sum())
        env_json = len(json.dumps({
            "lo": [None if not np.isfinite(v) else round(float(v), 4)
                   for v in px_lo],
            "hi": [None if not np.isfinite(v) else round(float(v), 4)
                   for v in px_hi],
            "big": [int(v) for v in px_big]}))
        print(f"  {'overview payload (1400 cols)':<34} "
              f"{env_json / 1024:8.1f} KB  ({finite} non-empty)")

        # A THREE-MINUTE WINDOW, which is the resolution the tool is for.
        span = t[-1] - t[0]
        mid = t[0] + span * 0.5
        three_min = span * (3.0 / 390.0)
        sel = (t >= mid) & (t < mid + three_min)
        k = int(sel.sum())
        rows = [[round(float(a), 6), round(float(b), 4), int(c)]
                for a, b, c in zip(t[sel], p[sel], s[sel])]
        print(f"  {'3-minute window':<34} {k:8,} trades, "
              f"{len(json.dumps(rows)) / 1024:.0f} KB JSON")
        print(f"  {'trades per minute (session mean)':<34} "
              f"{n_rows / 390.0:8,.0f}")

    print("\n" + "=" * 74)
    print("  Nothing was written. No service or database was contacted.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
