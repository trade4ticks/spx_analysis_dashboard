"""This name's own normal arrival rate, for this time of day.

WHY THIS EXISTS. "Forty trades a minute" is not a reading. Forty is busy for
one name and dead for another, and both are busy at 09:35 and dead at 12:15.
The number that means something is forty against what THIS symbol normally
does in THIS fifteen-minute bucket, so that is what the pane shows.

AGAINST THE SAME CLOCK TIME, NEVER A DAILY AVERAGE. Every name looks like it
is deteriorating at noon, because every name is. A daily average would report
the lunch lull as a per-symbol anomaly on every symbol simultaneously, which
is the one thing this comparison must not do.

THE ONE READ OF equities_scalp. The tape brief says this service connects to
no database, and that stands everywhere else: no writes, no shared pool, no
router, and a failure here degrades one small pane rather than the plot. This
reads `intraday_metrics` — eleven sessions at fifteen-minute grain — once per
symbol per bucket, from its own tiny read-only pool.

FIFTEEN MINUTES IS THE GRAIN AND THERE IS NO FINER. The pipeline stores these
buckets; anything finer needs a recompute over raw prints, which is not a
thing a live page gets to trigger.

WHAT IS DELIBERATELY NOT HERE. No noise band and no "too noisy to trade"
indicator. Noise has not survived calibration, it is contaminated by small
orders flickering the midpoint, and putting it on screen during live trading
would lend it an authority it has not earned. The plot already shows an
unstable quote directly.
"""
from __future__ import annotations

import datetime as dt
import logging
import os
import time

log = logging.getLogger("live.norms")

# The pipeline's buckets are stamped on the market clock: the first is 09:30
# and the last 15:45. Reading them against any other clock silently compares
# a symbol to a different part of its own day.
MARKET_TZ = "America/New_York"
BUCKET_MINUTES = 15
SESSION_OPEN = dt.time(9, 30)
SESSION_CLOSE = dt.time(16, 0)

# The two columns this pane reads, named because the user named them. Both are
# stored per bucket by the pipeline; neither is derived here.
METRICS = ("trades_per_min", "shares_per_min")

# A bucket's answer cannot change until the bucket does — the history behind
# it only moves when the pipeline runs overnight. One query per symbol per
# bucket, which is what was asked for.
_CACHE: dict[tuple[str, str], tuple[float, dict]] = {}
_CACHE_TTL_S = BUCKET_MINUTES * 60

_pool = None
_pool_error: str | None = None


# ── the market clock ────────────────────────────────────────────────────────
def tz_problem() -> str | None:
    """Why the market clock cannot be read, or None.

    Windows ships no zone database, so `ZoneInfo("America/New_York")` raises
    there unless `tzdata` is installed. The tempting fallback — use local time
    — would compare a symbol against the wrong quarter of its own day and look
    entirely plausible doing it, so this reports instead of guessing.
    """
    try:
        from zoneinfo import ZoneInfo
        ZoneInfo(MARKET_TZ)
        return None
    except Exception as exc:                                # noqa: BLE001
        return (f"no timezone database for {MARKET_TZ} ({type(exc).__name__}) "
                f"— install `tzdata`. Without it the 15-minute bucket cannot "
                f"be identified, and guessing with local time would compare "
                f"this symbol against the wrong part of its own day.")


def market_now() -> dt.datetime | None:
    if tz_problem():
        return None
    from zoneinfo import ZoneInfo
    return dt.datetime.now(ZoneInfo(MARKET_TZ))


def bucket_for(when: dt.datetime) -> dt.time | None:
    """The 15-minute bucket `when` falls in, or None outside the session.

    Pure and separately testable: the boundaries are where this can be wrong,
    and being wrong by one bucket at 09:44:59 is invisible on screen.
    """
    t = when.time()
    if t < SESSION_OPEN or t >= SESSION_CLOSE:
        return None
    minute = (t.minute // BUCKET_MINUTES) * BUCKET_MINUTES
    return dt.time(t.hour, minute)


# ── percentiles ─────────────────────────────────────────────────────────────
def summarise(values: list[float]) -> dict | None:
    """Min, quartiles, median and max over a SMALL sample, stated as such.

    Eleven sessions is eleven. Deciles over eleven points are two order
    statistics wearing a percentile's name, so the outer band here is the
    actual observed range and the inner one the quartiles — both of which mean
    what they say at this n.
    """
    xs = sorted(v for v in values if v is not None)
    if not xs:
        return None
    return {"n": len(xs), "min": xs[0], "max": xs[-1],
            "p25": _quantile(xs, 0.25), "med": _quantile(xs, 0.50),
            "p75": _quantile(xs, 0.75)}


def _quantile(xs: list[float], q: float) -> float:
    """Linear interpolation between order statistics (numpy's default)."""
    if len(xs) == 1:
        return xs[0]
    pos = (len(xs) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(xs) - 1)
    frac = pos - lo
    return xs[lo] * (1 - frac) + xs[hi] * frac


# ── the database ────────────────────────────────────────────────────────────
def _dsn() -> str | None:
    explicit = os.environ.get("SCALP_DATABASE_URL")
    if explicit:
        return explicit
    base = os.environ.get("DATABASE_URL")
    if not base:
        return None
    from urllib.parse import urlsplit, urlunsplit
    parts = urlsplit(base)
    name = os.environ.get("SCALP_PG_DB", "equities_scalp")
    return urlunsplit(parts._replace(path="/" + name))


async def pool():
    """A tiny read-only pool, created on first use and never by startup.

    Deliberately not opened in the lifespan: the tape must start and run with
    the database down, missing or not yet built. A pane that cannot show a
    comparison says so; it does not stop prints from arriving.
    """
    global _pool, _pool_error
    if _pool is not None:
        return _pool
    dsn = _dsn()
    if not dsn:
        _pool_error = ("neither SCALP_DATABASE_URL nor DATABASE_URL is set, so "
                       "there is nothing to compare against")
        return None
    try:
        import asyncpg
        _pool = await asyncpg.create_pool(dsn, min_size=1, max_size=2,
                                          timeout=8, command_timeout=8)
        _pool_error = None
    except Exception as exc:                                # noqa: BLE001
        # Recorded and reported, never swallowed. A comparison pane that is
        # silently blank is indistinguishable from a symbol with no history.
        _pool_error = f"{type(exc).__name__}: {exc}"
        log.warning("scalp pool unavailable: %s", _pool_error)
        _pool = None
    return _pool


async def close() -> None:
    global _pool
    if _pool is not None:
        await _pool.close()
        _pool = None


# ── the reading ─────────────────────────────────────────────────────────────
async def arrival_norm(symbol: str, now: dt.datetime | None = None) -> dict:
    sym = (symbol or "").strip().upper()
    if not sym.isalnum():
        return {"ok": False, "why": f"{symbol!r} is not a symbol"}

    # The zone database is needed to ask what time it is in New York, and for
    # nothing else. A caller that supplies the clock — the harness, or a
    # backfill — has already answered that question, so the check belongs
    # HERE rather than at the top where it gated the whole function.
    when = now
    if when is None:
        tz = tz_problem()
        if tz:
            return {"ok": False, "symbol": sym, "why": tz}
        when = market_now()

    bucket = bucket_for(when) if when else None
    if bucket is None:
        return {"ok": False, "symbol": sym, "tz": MARKET_TZ,
                "clock": when.strftime("%H:%M") if when else None,
                "why": (f"outside the session — the buckets run "
                        f"{SESSION_OPEN:%H:%M} to {SESSION_CLOSE:%H:%M} "
                        f"market time")}

    label = bucket.strftime("%H:%M")
    key = (sym, label)
    hit = _CACHE.get(key)
    if hit and time.time() - hit[0] < _CACHE_TTL_S:
        return hit[1]

    p = await pool()
    if p is None:
        return {"ok": False, "symbol": sym, "bucket": label,
                "why": f"no read of equities_scalp: {_pool_error}"}

    # trade_date < today, ALWAYS. Today's own partial session must never
    # become part of the normal it is being compared against — that is a
    # comparison of a thing with itself, and it would tighten the band exactly
    # when the live value is extreme.
    today = when.date()
    try:
        rows = await p.fetch(
            f"""select trade_date, {", ".join(METRICS)}
                  from intraday_metrics
                 where symbol = $1 and bucket_time = $2 and trade_date < $3
                 order by trade_date""",
            sym, bucket, today)
    except Exception as exc:                                # noqa: BLE001
        log.warning("arrival_norm(%s, %s) failed: %s", sym, label, exc)
        return {"ok": False, "symbol": sym, "bucket": label,
                "why": f"the history query failed: {type(exc).__name__}: {exc}"}

    if not rows:
        return {"ok": False, "symbol": sym, "bucket": label,
                "why": (f"{sym} has no stored history at {label} — the "
                        f"pipeline's universe is the 587 names it screens, "
                        f"and this is not one of them")}

    out = {
        "ok": True, "symbol": sym, "bucket": label, "tz": MARKET_TZ,
        "grain_min": BUCKET_MINUTES,
        "sessions": len(rows),
        "first": str(rows[0]["trade_date"]), "last": str(rows[-1]["trade_date"]),
    }
    for m in METRICS:
        out[m] = summarise([r[m] for r in rows])
    # Rows can exist with the values null — a bucket the pipeline reached but
    # could not measure. That is not a comparison, and reporting ok on it puts
    # a band on screen with nothing behind it.
    missing = [m for m in METRICS if not out[m]]
    if missing:
        return {"ok": False, "symbol": sym, "bucket": label,
                "why": (f"{len(rows)} sessions stored at {label} but "
                        f"{', '.join(missing)} is empty in all of them")}
    _CACHE[key] = (time.time(), out)
    return out
