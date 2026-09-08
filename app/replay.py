"""Replay: the tape for a past session, read from parquet.

WHY THIS EXISTS, because it decides what it must not do. Four metric families
have failed to identify a tradeable name -- noise, spread over noise, quiet
windows, and the live scan grid. The common thread is a description being
formalised into something subtly different, discovered only by watching the
tape and disagreeing. So this is not a metric, not a screen and not a ranking.
It is a way to look at a specific window and say "this one, not that one", so
a future metric can be built against labelled examples instead of a
description of them.

Everything here is READ-ONLY against parquet. Nothing is written anywhere and
no database is touched.

--- What the browser gets, and what it does not ----------------------------

Measured on the box: NVDA 2026-08-27 is 4,325,698 rows at 11,092 trades a
minute, and reading the five columns a chart needs takes 47 ms. Shipping that
session to a browser is ~30 MB of JSON for a picture in which no individual
print is visible, so it is never sent.

  the overview   390 one-minute candles, open/high/low/close/volume/trades.
                 ~25 KB whatever the symbol. Candles rather than a min/max
                 envelope because the envelope says what the range was and
                 the candle says whether price was going anywhere within it,
                 which is what picks a window worth zooming into.
  a window       every trade in it, columnar, plus the NBBO CHANGE POINTS.

--- The switch is on trade COUNT, not on time ------------------------------

NVDA at 11,092 trades a minute reaches twenty thousand in under two minutes.
FDX at 118 a minute would not reach it in two hours. A time-width threshold
would be wrong for both, in opposite directions, so the server counts what is
actually in the window and says which mode it is answering in.

The count is a searchsorted against a cached array, so deciding costs nothing.

--- Sessions are cached, not files -----------------------------------------

Zooming is the whole point, and re-reading a 37 MB parquet on every zoom would
put 443 ms between a scroll and a redraw. Row-group skipping was measured and
does not help: NVDA has five groups and AAPL has one, so the file cannot be
sliced usefully by time. So a whole symbol-day is decoded once and kept.

TWO of them, LRU. NVDA is ~104 MB as arrays and the box has been OOM-killed
twice; two is enough to flip between a name and the one before it without
being enough to hurt.
"""
from __future__ import annotations

import os
import threading
from collections import OrderedDict
from datetime import date as _date
from pathlib import Path

import numpy as np

from app.scalp_spread import spread_tw_window

# Regular trading hours, which is what the fetch stores. 09:30 to 16:00 ET is
# 390 minutes; the axis is seconds since the open, so the session is 0..23400.
SESSION_MINUTES = 390
SESSION_SECONDS = SESSION_MINUTES * 60

# The five of nineteen stored columns a chart needs. Parquet is columnar, so
# naming them saves 64-66% of the read -- measured.
WANT = {
    "time":  ["trade_timestamp", "timestamp", "ms_of_day", "time", "datetime"],
    "price": ["trade_price", "price", "last"],
    "size":  ["trade_size", "size", "quantity", "shares"],
    "bid":   ["bid", "bid_price", "nbbo_bid"],
    "ask":   ["ask", "ask_price", "nbbo_ask"],
}

# Above this many trades in view, individual prints stop being drawn. 20,000
# is where the render bench stops holding 60fps; past it the page shows the
# candles for the same range and SAYS SO, because a dense field of dots and a
# dense field of candles look alike at a glance and mean different things.
PRINT_LIMIT = int(os.environ.get("REPLAY_PRINT_LIMIT", "20000"))

_CACHE: OrderedDict = OrderedDict()
_CACHE_MAX = int(os.environ.get("REPLAY_CACHE_SESSIONS", "2"))
_LOCK = threading.Lock()


class ReplayError(RuntimeError):
    """Something the caller should be told in words, not a 500."""


def raw_dir() -> Path:
    d = os.environ.get("SCALP_DATA_DIR")
    if not d:
        raise ReplayError(
            "SCALP_DATA_DIR is not set. It has no default on purpose -- see "
            "app/scalp_config.py -- so replay cannot find the parquet store.")
    return Path(d) / "raw"


def _resolve(names) -> dict:
    """Column names as the vendor actually returned them.

    Candidate-based rather than hardcoded, matching scalp/schema.py: the
    vendor documents two naming schemes and the pipeline resolves rather than
    assumes. Exact, case-insensitive, never substring -- `size` must not pick
    up `bid_size`, which is in the same frame.
    """
    lower = {c.lower(): c for c in names}
    out = {}
    for purpose, cands in WANT.items():
        for c in cands:
            if c.lower() in lower:
                out[purpose] = lower[c.lower()]
                break
    missing = set(WANT) - set(out)
    if missing:
        raise ReplayError(
            f"could not resolve {sorted(missing)} in the parquet. "
            f"Columns present: {list(names)}")
    return out


# ── what is on disk ─────────────────────────────────────────────────────────

def sessions() -> list[dict]:
    """Session dates with the number of symbols each actually has.

    THE COUNT IS NOT DECORATION. 2026-08-14 holds 96 symbols where every other
    session holds 634-739, so a bare list of dates would offer a day that
    looks identical and is mostly empty. A picker that shows the count lets a
    thin day be recognised rather than discovered.
    """
    root = raw_dir()
    if not root.is_dir():
        raise ReplayError(f"no parquet store at {root}")
    by_day: dict = {}
    for sym_dir in root.iterdir():
        if not sym_dir.is_dir():
            continue
        for f in sym_dir.glob("*.parquet"):
            try:
                _date.fromisoformat(f.stem)
            except ValueError:
                continue
            by_day[f.stem] = by_day.get(f.stem, 0) + 1
    return [{"date": d, "symbols": by_day[d]} for d in sorted(by_day, reverse=True)]


def symbols_for(day: str) -> list[str]:
    root = raw_dir()
    if not root.is_dir():
        return []
    return sorted(p.name for p in root.iterdir()
                  if p.is_dir() and (p / f"{day}.parquet").is_file())


def path_for(symbol: str, day: str) -> Path:
    return raw_dir() / symbol.upper() / f"{day}.parquet"


# ── loading ─────────────────────────────────────────────────────────────────

def load(symbol: str, day: str) -> dict:
    """One symbol-day as arrays, cached. Raises ReplayError if absent.

    A MISSING FILE IS NOT AN EMPTY CHART. Drawing nothing for a date with no
    parquet is indistinguishable from drawing a session where nothing traded,
    and the whole point of this tool is telling those apart.
    """
    key = (symbol.upper(), day)
    with _LOCK:
        hit = _CACHE.get(key)
        if hit is not None:
            _CACHE.move_to_end(key)
            return hit

    p = path_for(symbol, day)
    if not p.is_file():
        raise ReplayError(
            f"no parquet for {symbol.upper()} on {day}. Raw retention is 45 "
            f"days and coverage varies by session -- check the date list.")

    import pyarrow.parquet as pq
    pf = pq.ParquetFile(p)
    res = _resolve(pf.schema_arrow.names)
    tbl = pq.read_table(p, columns=[res[k] for k in WANT])

    t = tbl.column(res["time"]).to_numpy(zero_copy_only=False)
    # SECONDS SINCE 09:30 OF THIS SESSION, not epoch. The axis is a session,
    # the numbers stay small enough to keep millisecond resolution in a float,
    # and the client never has to know what timezone the box is in.
    #
    # THE ORIGIN COMES FROM THE DATE, NOT FROM THE FIRST TRADE. Flooring the
    # first print to its minute is right for AAPL, whose first trade is at
    # 09:30:00.004, and wrong for any name that does not print until 09:31 --
    # the whole session would shift a minute left and every candle would be
    # attributed to the wrong minute, consistently enough to look fine.
    #
    # The stored timestamps are naive ET wall-clock, so the reference is built
    # the same way and the subtraction is exact whatever the box's zone is.
    t = np.asarray(t, dtype="datetime64[ns]").astype("int64") / 1e9
    open_s = (np.datetime64(f"{day}T09:30:00", "ns").astype("int64") / 1e9)
    t = t - open_s

    out = {
        "symbol": symbol.upper(), "date": day,
        "t": np.asarray(t, dtype="float64"),
        "p": np.asarray(tbl.column(res["price"]).to_numpy(zero_copy_only=False),
                        dtype="float32"),
        "s": np.asarray(tbl.column(res["size"]).to_numpy(zero_copy_only=False),
                        dtype="float32"),
        "bid": np.asarray(tbl.column(res["bid"]).to_numpy(zero_copy_only=False),
                          dtype="float32"),
        "ask": np.asarray(tbl.column(res["ask"]).to_numpy(zero_copy_only=False),
                          dtype="float32"),
    }
    # Trades arrive in time order and every window query is a searchsorted, so
    # a file that is not sorted would silently return the wrong window rather
    # than fail. Cheap to check once per load, impossible to notice later.
    if out["t"].size > 1 and not np.all(np.diff(out["t"]) >= 0):
        order = np.argsort(out["t"], kind="stable")
        for k in ("t", "p", "s", "bid", "ask"):
            out[k] = out[k][order]

    with _LOCK:
        _CACHE[key] = out
        _CACHE.move_to_end(key)
        while len(_CACHE) > _CACHE_MAX:
            _CACHE.popitem(last=False)
    return out


def cache_status() -> dict:
    with _LOCK:
        held = [{"symbol": k[0], "date": k[1],
                 "rows": int(v["t"].size),
                 "mb": sum(v[c].nbytes for c in ("t", "p", "s", "bid", "ask"))
                       / 1048576.0}
                for k, v in _CACHE.items()]
    return {"sessions": held, "max": _CACHE_MAX}


# ── the overview ────────────────────────────────────────────────────────────

def candles(symbol: str, day: str) -> dict:
    """390 one-minute bars: open, high, low, close, volume, trades.

    CANDLES RATHER THAN A MIN/MAX ENVELOPE, and the difference is the point.
    An envelope says what the price range was in that minute; a candle says
    that AND whether price was going somewhere within it. Picking a window
    worth zooming into is a question about direction as much as spread, and
    the envelope cannot answer it.

    Bars come from segment reductions, not a Python loop: trades are time
    ordered so the minute index is non-decreasing, searchsorted gives the bar
    edges, and one reduceat per quantity does 4.3 million rows in tens of
    milliseconds.
    """
    d = load(symbol, day)
    t, p, s = d["t"], d["p"], d["s"]
    n = SESSION_MINUTES
    o = np.full(n, np.nan); h = np.full(n, np.nan)
    lo = np.full(n, np.nan); c = np.full(n, np.nan)
    vol = np.zeros(n); cnt = np.zeros(n, dtype="int64")

    if t.size:
        minute = np.clip((t // 60).astype("int64"), 0, n - 1)
        edges = np.searchsorted(minute, np.arange(n + 1), side="left")
        starts, ends = edges[:-1], edges[1:]
        live = np.flatnonzero(starts < ends)
        if live.size:
            idx = starts[live]
            o[live] = p[idx]
            c[live] = p[ends[live] - 1]
            h[live] = np.maximum.reduceat(p, idx)
            lo[live] = np.minimum.reduceat(p, idx)
            vol[live] = np.add.reduceat(s.astype("float64"), idx)
            cnt[live] = (ends - starts)[live]

    return {
        "symbol": d["symbol"], "date": d["date"],
        "minutes": n,
        "o": _nums(o, 4), "h": _nums(h, 4), "l": _nums(lo, 4), "c": _nums(c, 4),
        "v": [int(x) for x in vol], "n": [int(x) for x in cnt],
        "rows": int(t.size),
        "t_first": float(t[0]) if t.size else None,
        "t_last": float(t[-1]) if t.size else None,
    }


# ── a window ────────────────────────────────────────────────────────────────

def window(symbol: str, day: str, t0: float, t1: float,
           limit: int | None = None, nbbo: bool = True) -> dict:
    """Everything in [t0, t1), or a refusal to draw it print by print.

    THE MODE IS DECIDED HERE AND NAMED IN THE RESPONSE. The client must not
    infer it from how much came back, because "few trades" and "too many to
    send" would look the same at the edge.
    """
    limit = PRINT_LIMIT if limit is None else int(limit)
    d = load(symbol, day)
    t = d["t"]
    i0 = int(np.searchsorted(t, t0, side="left"))
    i1 = int(np.searchsorted(t, t1, side="left"))
    count = max(0, i1 - i0)

    out = {
        "symbol": d["symbol"], "date": d["date"],
        "t0": float(t0), "t1": float(t1),
        "count": count, "limit": limit,
        "mode": "prints" if count < limit else "candles",
        "stats": _stats(d, i0, i1, t0, t1),
    }
    if out["mode"] != "prints":
        # Nothing is sent. The client already holds the 390 candles and draws
        # the ones inside the window; shipping a million trades it cannot
        # render would be slow AND useless, in that order.
        return out

    sl = slice(i0, i1)
    # COLUMNAR, not a list of triples. Same numbers, roughly a third fewer
    # bytes, because every row otherwise carries its own brackets and commas.
    out["t"] = _nums(t[sl], 3)
    out["p"] = _nums(d["p"][sl], 4)
    out["s"] = [int(x) for x in d["s"][sl]]
    if nbbo:
        out["nbbo"] = _nbbo_steps(d, i0, i1)
    return out


def _nbbo_steps(d: dict, i0: int, i1: int) -> dict:
    """The quote as STEP CHANGES, not one point per trade.

    Every trade row carries the prevailing bid and ask, so the NBBO costs
    nothing to read -- but sending one point per trade would double the
    payload to draw a line that is flat between changes. Only the rows where
    bid or ask actually moved are sent, and the client holds each level until
    the next one.
    """
    b, a, t = d["bid"][i0:i1], d["ask"][i0:i1], d["t"][i0:i1]
    if b.size == 0:
        return {"t": [], "bid": [], "ask": []}
    changed = np.empty(b.size, dtype=bool)
    changed[0] = True
    changed[1:] = (b[1:] != b[:-1]) | (a[1:] != a[:-1])
    idx = np.flatnonzero(changed)
    return {"t": _nums(t[idx], 3), "bid": _nums(b[idx], 4),
            "ask": _nums(a[idx], 4)}


def _stats(d: dict, i0: int, i1: int, t0: float, t1: float) -> dict:
    """What is in the visible window. A READOUT, not a screen.

    No thresholds, no colouring, no judgement -- these describe the picture
    rather than grading it. That restraint is the point: every metric family
    so far has failed by grading before anyone agreed what good looked like.
    """
    n = max(0, i1 - i0)
    minutes = max(1e-9, (t1 - t0) / 60.0)
    if n == 0:
        return {"trades": 0, "trades_per_min": 0.0, "shares_per_min": 0.0,
                "p10_p90_cents": None, "spread_cents_tw": None,
                "minutes": minutes}
    p = d["p"][i0:i1].astype("float64")
    s = d["s"][i0:i1].astype("float64")
    q10, q90 = np.percentile(p, (10.0, 90.0))
    sp_c, _sp_b, _obs, _crossed = spread_tw_window(
        d["t"][i0:i1] * 1000.0, d["bid"][i0:i1], d["ask"][i0:i1],
        end_ms=t1 * 1000.0)
    return {
        "trades": n,
        "trades_per_min": n / minutes,
        "shares_per_min": float(s.sum()) / minutes,
        "p10_p90_cents": float((q90 - q10) * 100.0),
        "spread_cents_tw": None if sp_c != sp_c else float(sp_c),
        "minutes": minutes,
    }


def _nums(arr, places: int) -> list:
    """Finite floats rounded for the wire; NaN becomes null.

    NaN IS NOT JSON. Python's encoder emits a bare `NaN`, which is not in the
    grammar -- a browser rejects the whole frame rather than one value, so a
    minute with no trades would take the entire chart with it.
    """
    a = np.asarray(arr, dtype="float64")
    ok = np.isfinite(a)
    r = np.round(a, places)
    return [None if not k else float(v) for k, v in zip(ok, r)]
