"""Configuration for Equities Live.

SEPARATE SERVICE, SEPARATE PORT, deliberately. If this crashes it must not
take the dashboards with it, so it shares no process, no database and no
router with them — only the stylesheet, read from disk.

THE HOST IS NOT INLINE. Real-time and delayed are two different sockets and a
15-minute-old tape renders identically to a live one, which is the dangerous
part. The choice is a setting, the page states which it got, and the delayed
case is announced rather than inferred.
"""
from __future__ import annotations

import os
from pathlib import Path

try:
    from dotenv import load_dotenv
    # The key lives with the options project. Loaded from there rather than
    # copied, so there is one place it can be rotated.
    for candidate in (Path(__file__).resolve().parents[2] / "Open_Interest" / ".env",
                      Path(__file__).resolve().parents[1] / ".env"):
        if candidate.is_file():
            load_dotenv(candidate)
    load_dotenv()
except ImportError:                                    # pragma: no cover
    pass

ROOT = Path(__file__).resolve().parents[1]

# ── the upstream socket ─────────────────────────────────────────────────────
REALTIME_URL = os.environ.get("LIVE_WS_REALTIME",
                              "wss://socket.massive.com/stocks")
DELAYED_URL  = os.environ.get("LIVE_WS_DELAYED",
                              "wss://delayed.massive.com/stocks")

# "realtime" | "delayed". Anything else is refused at startup rather than
# quietly defaulting, because defaulting to the wrong one is the failure that
# looks like success.
FEED = os.environ.get("LIVE_FEED", "realtime").strip().lower()

API_KEY = (os.environ.get("POLYGON_API_KEY")
           or os.environ.get("MASSIVE_API_KEY") or "")

PORT = int(os.environ.get("LIVE_PORT", "8001"))
HOST = os.environ.get("LIVE_HOST", "0.0.0.0")


def feed_url() -> str:
    return REALTIME_URL if FEED == "realtime" else DELAYED_URL


def feed_is_delayed() -> bool:
    return FEED != "realtime"


# ── hard caps ───────────────────────────────────────────────────────────────
#
# The box has been OOM-killed twice this week and already runs three
# dashboards, Postgres, the ThetaData terminal and batch jobs. Memory here
# should be trivial, but "should be" is what an unbounded buffer is before it
# is not — so every growth axis has a ceiling and the page is told when one
# binds.
#
# The arithmetic, so the numbers are not arbitrary: FDX runs ~56 trades and
# ~283 quotes a minute. At the 15-minute ceiling that is ~5,100 records for one
# symbol; at four symbols, ~20,000. A record is a small dict of six numbers.
MAX_SYMBOLS = int(os.environ.get("LIVE_MAX_SYMBOLS", "4"))
MAX_WINDOW_S = int(os.environ.get("LIVE_MAX_WINDOW_S", str(15 * 60)))
DEFAULT_WINDOW_S = int(os.environ.get("LIVE_DEFAULT_WINDOW_S", "180"))

# A second ceiling in COUNT, because a halt-and-reopen or a news print can put
# a minute's worth of tape into a second and the time bound alone would not
# hold. Whichever binds first wins.
MAX_TRADES_PER_SYMBOL = int(os.environ.get("LIVE_MAX_TRADES", "40000"))
MAX_QUOTES_PER_SYMBOL = int(os.environ.get("LIVE_MAX_QUOTES", "60000"))

# Browser sockets. Each is a fan-out target, not a subscription of its own.
MAX_CLIENTS = int(os.environ.get("LIVE_MAX_CLIENTS", "8"))

# ── transport batching ──────────────────────────────────────────────────────
#
# NOT aggregation. Every trade keeps its own timestamp, its own price and its
# own size, and nothing is merged, bucketed or deduplicated — seven prints at
# .401 stay seven prints at .401, because that clustering is the information.
# This only decides how often the accumulated records are put on the wire, so
# 300 quotes a minute do not become 300 WebSocket frames.
FLUSH_MS = int(os.environ.get("LIVE_FLUSH_MS", "100"))

# ── reconnect ───────────────────────────────────────────────────────────────
RECONNECT_BASE_S = 1.0
RECONNECT_MAX_S = 30.0


def problems() -> list[str]:
    """Configuration faults worth refusing to start over.

    Returned rather than raised so the caller can decide; the page reports
    them, because a service that will not connect is more useful saying why
    than not running at all.
    """
    out = []
    if FEED not in ("realtime", "delayed"):
        out.append(f"LIVE_FEED is {FEED!r}; expected 'realtime' or 'delayed'.")
    if not API_KEY:
        out.append("POLYGON_API_KEY is not set — the upstream socket cannot "
                   "authenticate. It lives in the Open_Interest project's "
                   ".env.")
    if MAX_SYMBOLS < 1:
        out.append("LIVE_MAX_SYMBOLS must be at least 1.")
    return out
