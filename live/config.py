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

# ── the scan tier ───────────────────────────────────────────────────────────
#
# A SECOND CLASS OF HOLDER ON THE SAME SOCKET, and not a preference.
#
# Measured: the account permits ONE concurrent websocket. A second one
# authenticates, is accepted for every subscription, and is then closed with
# 1008 and `max_connections` -- while the service that already held the
# connection reconnects and evicts the newcomer in turn. Two processes on one
# key do not share the feed, they trade it back and forth, and both show a
# plausible partial tape while doing it. So the scan cannot have a socket of
# its own; it holds symbols on this one.
#
# The two tiers want different things from the same connection, which is why
# they are counted separately rather than sharing MAX_SYMBOLS:
#
#   a pane   wants trades AND quotes, and fifteen minutes of both, for one of
#            eight symbols it is drawing in full detail.
#   the scan wants trades ONLY, for six minutes, across hundreds of symbols it
#            is reducing to one number each.
#
# Measured at the open on the VPS, trades only: 600 symbols is 3,850 records
# and 579 KB a second, 30% of one core, with the ingest loop busy 6% of
# wall-clock. Nothing there is close to binding. The ceiling is set at 600
# because the universe holds 740 names and the box carries it comfortably, not
# because anything broke -- and going wide is cheap, since 50 symbols already
# carry most of the volume and the other 550 add under half as much again.
SCAN_MAX_SYMBOLS = int(os.environ.get("LIVE_SCAN_MAX_SYMBOLS", "600"))

# SIX MINUTES, not fifteen. The longest thing the scan computes is a
# five-minute range, and the retention only has to cover it with a margin.
# Fifteen would be two and a half times the memory for data nothing reads.
SCAN_RETAIN_S = float(os.environ.get("LIVE_SCAN_RETAIN_S", "360"))

# Ring sizing. Symbols START at the small figure and grow toward the large one
# as their own rate demands -- see the note in scan.py. The measured mistake
# was a single assumed rate for every symbol, which truncated the busy names
# and over-allocated the quiet ones by 3x at the same time.
#
# 512 records is a minute at 8 trades/sec, which covers most of the universe
# outright. 72,000 is six minutes at 200/sec, which is more than the busiest
# name printed at the open; at 1.7 MB it is affordable for the handful that
# ever reach it.
SCAN_RING_START = int(os.environ.get("LIVE_SCAN_RING_START", "512"))
SCAN_RING_MAX = int(os.environ.get("LIVE_SCAN_RING_MAX", "72000"))

# The scan's two lookbacks, which are deliberately different from each other.
# See rollup_one: quiet has to be responsive because it is what changes;
# range and volume must not be, because a bar that jumps while you glance at
# it is worse than no bar.
SCAN_QUIET_WINDOW_S = float(os.environ.get("LIVE_SCAN_QUIET_WINDOW_S", "60"))
SCAN_SLOW_WINDOW_S = float(os.environ.get("LIVE_SCAN_SLOW_WINDOW_S", "300"))

# How often the live "now" column is recomputed. Quiet is the thing that
# changes and the reason the page is open, so it has to be responsive; five
# seconds is twelve passes a minute, which measured at ~8% of one core for 430
# symbols.
SCAN_TICK_S = float(os.environ.get("LIVE_SCAN_TICK_S", "5"))

# Where the grid's minutes are kept so a deploy does not blank them. Files are
# per session date and are NOT pruned -- see scan_history for why yesterday's
# grid being loadable is a feature rather than an oversight.
SCAN_HISTORY_DIR = os.environ.get(
    "LIVE_SCAN_HISTORY_DIR", str(ROOT / "data" / "scan_history"))

# How many minutes the grid shows. The store keeps the whole session; this is
# only how much of it goes over the wire on connect.
SCAN_GRID_MINUTES = int(os.environ.get("LIVE_SCAN_GRID_MINUTES", "120"))

# ── the page's own defaults ─────────────────────────────────────────────────
#
# SERVED, NOT BAKED INTO THE JAVASCRIPT, so the number a person reasons about
# lives in one place and can be moved without a rebuild. The page persists
# whatever the user sets in local storage; these are only what it opens on the
# first time.
#
# THE VOLUME FLOOR IS ARITHMETIC, NOT A PERCENTILE. At $17k round trips,
# staying under a few percent of a minute's flow puts the floor near $750k/min.
# A moving percentile would slide underneath the user as the symbol set
# changes, which is the same objection as a colour ramp fitted to whatever this
# morning happened to look like.
SCAN_VOLUME_FLOOR = float(os.environ.get("LIVE_SCAN_VOLUME_FLOOR", "750000"))

# The colour anchors: brightest at the low, dark at the high, uniformly dark
# above. Measured 2026-09-08 the ratio ran p10 0.07 / p50 0.42 / p90 1.22, so
# a ramp spread over the full 0-2.4 puts nearly every name in two shades. These
# open near that spread and the page shows the LIVE percentiles beside them, so
# the anchors can be placed against what the market is doing rather than
# against a remembered morning.
SCAN_RATIO_LOW = float(os.environ.get("LIVE_SCAN_RATIO_LOW", "0.10"))
SCAN_RATIO_HIGH = float(os.environ.get("LIVE_SCAN_RATIO_HIGH", "1.20"))

# THE SPREAD FLOOR, in cents and in bps, and both are MINIMA.
#
# A name whose quoted spread is too tight has nothing to capture however much
# it trades -- INTC sits near the top of the grid on 1-2 cents, and no amount
# of tuning the quiet ratio removes it, because quietness is not what is wrong
# with it.
#
# 5 cents is the pipeline's own figure: scalp/config.py's
# DEFAULT_FILTERS['min_spread_cents']. Matching it means the page opens on the
# same screen the universe was built with rather than on a number invented
# here. The bps minimum defaults OFF, because the cents figure is the one with
# a decision behind it and a second active threshold nobody chose would hide
# names for a reason that is not written down anywhere.
SCAN_MIN_SPREAD_CENTS = float(os.environ.get("LIVE_SCAN_MIN_SPREAD_CENTS", "5"))
SCAN_MIN_SPREAD_BPS = float(os.environ.get("LIVE_SCAN_MIN_SPREAD_BPS", "0"))

# How long a quote may be credited with standing, in seconds.
#
# A quote that stands across a halt, a feed gap, or a subscription that has
# just been restored would otherwise dominate a five-minute time-weighted
# average with a price nobody could have traded. 30 seconds is half the quiet
# window: long enough that an ordinarily still book is weighted honestly, short
# enough that an outage cannot own the window.
SCAN_QUOTE_DWELL_CAP_S = float(
    os.environ.get("LIVE_SCAN_QUOTE_DWELL_CAP_S", "30"))

# THE LONGEST THE ROLLUP MAY HOLD THE EVENT LOOP, in milliseconds.
#
# Measured: the rollup costs ~0.85 ms per symbol, so 430 symbols is a 367 ms
# pass. Run as one uninterrupted loop that is 367 ms during which NOTHING else
# in this process runs -- not the upstream reader, and not Hub.pump(), which
# flushes to every browser every 100 ms. The tape page would stall for three
# and a half flush intervals every five seconds, on the same connection the
# scan shares, and the scan page would make it obvious.
#
# 8 ms is half a 60fps frame, so a pass can begin and finish inside one frame's
# budget without the browser noticing. The slice is a TIME, not a symbol count,
# so it holds as the per-symbol cost changes and on a slower box -- a count
# tuned to today's 0.85 ms is a block that silently grows.
SCAN_ROLLUP_SLICE_MS = float(os.environ.get("LIVE_SCAN_ROLLUP_SLICE_MS", "8"))


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
# showing it as current.
#
# FOUR SECONDS, from the account's own record: orders here live ONE TO SIX
# seconds — entered, repriced and filled inside what used to be a single poll
# interval. Twelve seconds would have called a list "current" that had
# already missed an order's entire life. This is two order-poll intervals
# plus the ~850ms the read itself takes.
STALE_AFTER_S = float(os.environ.get("LIVE_STALE_AFTER_S", "4"))

# The two poll cadences. Orders change constantly and cost ~850ms; positions
# change only when something fills and cost ~370ms. Together: 30 + 10 = 40
# calls a minute against the 90 available to ordinary traffic, leaving 50 for
# placement and repricing and the 30-call reserve untouched.
ORDER_POLL_S = float(os.environ.get("LIVE_ORDER_POLL_S", "2"))
POSITION_POLL_S = float(os.environ.get("LIVE_POSITION_POLL_S", "6"))


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
