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
    # The Schwab credentials live with the portfolio dashboard, which owns the
    # token file too — loaded from there rather than copied, so there is one
    # place they can be rotated. Same reasoning as the Polygon key above.
    _schwab_env = os.environ.get("SCHWAB_ENV_FILE",
                                 "/root/Portfolio_Dashboard/.env")
    for candidate in (Path(__file__).resolve().parents[2] / "Open_Interest" / ".env",
                      Path(_schwab_env),
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
# Raised from four. A 2x2 grid is four panes on its own, and pinned symbols
# hold a reference each — four of both would sit exactly on the old ceiling
# and refuse the fifth. Eight busy names at the 15-minute ceiling is ~41,000
# records of six numbers, which is still nothing next to Postgres on this box.
MAX_SYMBOLS = int(os.environ.get("LIVE_MAX_SYMBOLS", "8"))

# ── the persistent watchlist ────────────────────────────────────────────────
#
# Symbols held whether or not a pane is watching them, so closing the browser
# does not throw away the buffer and leave the next pane reading "buffering
# 55s of 180s". Pins can also be set and cleared from the page; this is only
# the set restored on restart.
PINNED = [s.strip().upper() for s in
          os.environ.get("LIVE_PINNED", "").split(",") if s.strip()]
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


# ── trading ─────────────────────────────────────────────────────────────────
#
# OFF BY DEFAULT, and that is not timidity. This is order-placing code, and a
# service that can trade the moment it starts can trade because of a stray
# request, a replayed fetch, or a page left open. Turning it on is one line in
# .env and a restart, which is the right amount of deliberateness.
#
# It is the outermost of three switches. The pane's arm toggle is the second
# and the guards below are the third; all three are checked server-side.
TRADING_ENABLED = os.environ.get("LIVE_TRADING_ENABLED", "").strip().lower() \
    in ("1", "true", "yes", "on")

# The shared secret that lets a caller ENABLE trading over HTTP. Disabling
# never needs it — a control that fails closed at the worst moment is worse
# than one that anybody can use to stop trading.
#
# This service is reachable from the internet through the cloudflared tunnel,
# and behind it every request appears to come from localhost, so filtering by
# address would prove nothing. Without a token set, enabling over HTTP is
# refused outright rather than left open.
CONTROL_TOKEN = os.environ.get("LIVE_CONTROL_TOKEN", "").strip()

SCHWAB_API_KEY = os.environ.get("SCHWAB_API_KEY", "")
SCHWAB_API_SECRET = os.environ.get("SCHWAB_API_SECRET", "")
# Skips an account lookup per restart; optional.
SCHWAB_ACCOUNT_HASH = os.environ.get("SCHWAB_ACCOUNT_HASH", "")

# THE PORTFOLIO DASHBOARD OWNS THIS FILE. Shared rather than duplicated,
# because Schwab rotates the refresh token on every refresh and two files
# would mean two rotations racing each other with no way to reconcile. See
# the long note in broker._token() for what is and is not protected.
SCHWAB_TOKEN_FILE = os.environ.get(
    "SCHWAB_TOKEN_FILE", "/root/Portfolio_Dashboard/schwab_tokens.json")

# Refresh only inside the last minute of the access token's 30, so this
# service and the portfolio dashboard are rarely both due at once.
REFRESH_MARGIN_S = int(os.environ.get("SCHWAB_REFRESH_MARGIN_S", "60"))

# 120 A MINUTE, TOTAL, across every pane. A round trip with four repricings on
# entry and three on exit is eight calls — nothing on average, and able to
# burst when three panes reprice together. The reserve is quota that only
# cancel, flatten and the state read may spend, so getting flat is never the
# call that gets refused.
SCHWAB_CALLS_PER_MIN = int(os.environ.get("SCHWAB_CALLS_PER_MIN", "120"))
SCHWAB_CALL_RESERVE = int(os.environ.get("SCHWAB_CALL_RESERVE", "30"))

# ── the fat-finger guards ───────────────────────────────────────────────────
#
# One mistyped quantity at eight-second holds is expensive, and a mistyped
# PRICE is worse than a mistyped size — 31.85 for 318.50 is a marketable order
# at a tenth of the price, which no share limit catches.
MAX_ORDER_SHARES = int(os.environ.get("LIVE_MAX_ORDER_SHARES", "500"))
MAX_POSITION_SHARES = int(os.environ.get("LIVE_MAX_POSITION_SHARES", "1000"))
MAX_NOTIONAL = float(os.environ.get("LIVE_MAX_NOTIONAL", "50000"))
MAX_LIMIT_DISTANCE_PCT = float(
    os.environ.get("LIVE_MAX_LIMIT_DISTANCE_PCT", "5"))

# How stale the order state may be before the page says so rather than
# showing it as current. Two poll intervals plus a round trip.
STALE_AFTER_S = float(os.environ.get("LIVE_STALE_AFTER_S", "12"))


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
