"""The wall tier, its watchlist and its frames, against a fabricated feed.

NO MARKET NEEDED. What is most likely to be wrong here is decidable from a
synthetic message stream, and none of it can be checked by looking at a wall
of a hundred panes and deciding it looks about right -- a wrong pane on a wall
looks like a quiet symbol, which is the one reading the page exists to give.

WHAT IS BEING PROTECTED, in order of how bad it would be:

  * THE OTHER TIERS' SUBSCRIPTIONS. Three tiers now share one upstream socket
    because the account permits exactly one. A drop that forgets to ask
    whether another tier still holds the symbol unsubscribes a row the scan
    is drawing or a pane someone has an order resting on, and BOTH look like
    a market that went quiet.
  * NOTHING IS AGGREGATED, and no record is delivered twice or skipped. The
    cursor is a count for a reason: seven prints share a millisecond.
  * THE WATCHLIST SURVIVES A RESTART, with its per-symbol overrides. It is a
    hundred names chosen by hand and spx-live restarts on every deploy.
  * THE SCALE'S INPUT IS A MEDIAN. One wide quote must not rescale a pane.
  * THE PAGE NEVER TALKS TO A BROKER. This page is for watching.
"""
from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from live import config                                   # noqa: E402
from live.hub import Hub                                  # noqa: E402
from live.wall_runner import WallRunner                   # noqa: E402
from live.wall_store import WallStore                     # noqa: E402

FAILS: list[str] = []


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


class FakeSock:
    """A browser socket that keeps what it was sent."""

    def __init__(self):
        self.frames = []

    async def send_json(self, payload):
        self.frames.append(payload)


def runner(hub, tmp: Path) -> WallRunner:
    return WallRunner(hub, WallStore(tmp / "wall_watchlist.json"))


# ── the shared socket ───────────────────────────────────────────────────────
async def case_tiers_do_not_unsubscribe_each_other():
    """Every pair of tiers, both directions. One socket, three holders.

    THE FAILURE THIS CATCHES: the wall was added and the scan's drop path
    still asked only "does a pane hold it". Dropping a symbol from the scan
    would then unsubscribe it underneath the wall, and the wall's pane would
    go flat -- which reads as a symbol that stopped trading.
    """
    # scan drops, wall holds
    h = recording_hub()
    await h.wall_set(["FDX"])
    await h.scan_set(["FDX"])
    h.sent.clear()
    await h.scan_set([])
    check(not any(p["action"] == "unsubscribe" for p in h.sent),
          f"the scan dropping FDX unsubscribed it while the WALL held it: "
          f"{params(h)}")

    # wall drops, scan holds
    h = recording_hub()
    await h.scan_set(["FDX"])
    await h.wall_set(["FDX"])
    h.sent.clear()
    await h.wall_set([])
    check(not any(p["action"] == "unsubscribe" for p in h.sent),
          f"the wall dropping FDX unsubscribed it while the SCAN held it: "
          f"{params(h)}")

    # wall drops, a pane holds
    h = recording_hub()
    await h.acquire("FDX")
    await h.wall_set(["FDX"])
    h.sent.clear()
    await h.wall_set([])
    check(not any(p["action"] == "unsubscribe" for p in h.sent),
          f"the wall dropping FDX unsubscribed it while a PANE held it: "
          f"{params(h)}")

    # a pane closes, the wall holds
    h = recording_hub()
    await h.wall_set(["FDX"])
    await h.acquire("FDX")
    h.sent.clear()
    await h.release("FDX")
    check(not any(p["action"] == "unsubscribe" for p in h.sent),
          f"a pane closing unsubscribed FDX while the WALL held it: "
          f"{params(h)}")

    # and the last holder DOES unsubscribe, or the socket accumulates symbols
    h = recording_hub()
    await h.wall_set(["FDX"])
    h.sent.clear()
    await h.wall_set([])
    check([p["action"] for p in h.sent] == ["unsubscribe"],
          f"the LAST holder dropping FDX sent {params(h)!r} — nothing else "
          f"held it, so the subscription must go")


async def case_wall_does_not_resubscribe_a_held_symbol():
    """A symbol another tier already holds is not subscribed twice."""
    h = recording_hub()
    await h.scan_set(["FDX"])
    h.sent.clear()
    await h.wall_set(["FDX", "LLY"])
    subs = [p["params"] for p in h.sent if p["action"] == "subscribe"]
    check(subs == ["T.LLY,Q.LLY"],
          f"the wall subscribed {subs} — FDX was already live on the scan's "
          f"claim, and a duplicate makes the release path's mirror wrong")


async def case_reconnect_restores_the_wall():
    """A reconnect that comes back without the wall is a page of flat panes."""
    h = recording_hub()
    await h.scan_set(["AAPL"])
    await h.wall_set(["FDX", "LLY"])
    await h.acquire("NVDA")
    syms = sorted(set(h.refs) | set(h.scan) | set(h.wall))
    check(syms == ["AAPL", "FDX", "LLY", "NVDA"],
          f"the resubscribe set is {syms}; every tier's symbols must be in it")
    src = (ROOT / "live" / "hub.py").read_text(encoding="utf-8")
    check("set(self.refs) | set(self.scan) | set(self.wall)" in src,
          "_session() does not rebuild its subscription list from all three "
          "tiers; a reconnect would restore some pages and not others")


# ── ingest ──────────────────────────────────────────────────────────────────
async def case_trades_reach_the_wall_unaggregated():
    """Seven prints in one millisecond stay seven bubbles."""
    h = recording_hub()
    await h.wall_set(["FDX"])
    now = time.time() * 1000
    h._ingest(json.dumps([trade("FDX", now, 334.40 + i * 0.01, 1 + i)
                          for i in range(7)]))
    w = h.wall["FDX"]
    check(w.trades.n == 7 and w.trades_seen == 7,
          f"seven prints at one timestamp became {w.trades.n} records — the "
          f"clustering IS the information the wall is drawing")
    t, p, s = w.trades.ordered()
    check(sorted(s.tolist()) == [1, 2, 3, 4, 5, 6, 7],
          f"sizes {s.tolist()} — a bubble is sized by its share count and "
          f"every print keeps its own")


async def case_ingest_feeds_every_tier_that_holds_it():
    """One symbol, three tiers, one message: all three stores see it."""
    h = recording_hub()
    await h.scan_set(["FDX"])
    await h.wall_set(["FDX"])
    await h.acquire("FDX")
    now = time.time() * 1000
    h._ingest(json.dumps([trade("FDX", now, 334.4, 100),
                          quote("FDX", now, 334.37, 334.44)]))
    check(h.wall["FDX"].trades.n == 1, "the wall missed the trade")
    check(h.scan["FDX"].n == 1, "the scan missed the trade")
    check(len(h.trades["FDX"]) == 1, "the pane missed the trade")
    check(h.wall["FDX"].last_quote is not None, "the wall missed the quote")
    check(h.spread["FDX"].acc.sum() > 0, "the scan missed the quote")
    check(len(h.quotes["FDX"]) == 1, "the pane missed the quote")

    # And a symbol only the wall holds does not leak into the others.
    h2 = recording_hub()
    await h2.wall_set(["LLY"])
    h2._ingest(json.dumps([trade("LLY", now, 1158.6, 5)]))
    check("LLY" not in h2.scan and "LLY" not in h2.trades,
          "a wall-only symbol allocated a scan ring or a pane deque")


async def case_quotes_are_sampled_but_now_is_not():
    """The ring is thinned; the current spread is not."""
    h = recording_hub()
    await h.wall_set(["FDX"])
    w = h.wall["FDX"]
    now = time.time() * 1000
    # Two seconds of quotes, 10 ms apart, each a cent wider than the last.
    msgs = [quote("FDX", now + i * 10, 334.40, 334.41 + i * 0.0001)
            for i in range(200)]
    h._ingest(json.dumps(msgs))
    expect = int(2000 / config.WALL_QUOTE_SAMPLE_MS) + 1
    check(abs(w.quotes.n - expect) <= 1,
          f"200 quotes over two seconds stored {w.quotes.n} samples; at "
          f"{config.WALL_QUOTE_SAMPLE_MS:.0f} ms that should be ~{expect}")
    check(w.last_quote[0] == now + 1990,
          f"the CURRENT quote is {w.last_quote} — it must be the newest "
          f"message, not the newest sample, or the spread number on the pane "
          f"is up to a sample interval stale")


async def case_a_crossed_quote_is_dropped_and_counted():
    """A zero bid would put the band across the whole pane."""
    h = recording_hub()
    await h.wall_set(["FDX"])
    w = h.wall["FDX"]
    now = time.time() * 1000
    h._ingest(json.dumps([quote("FDX", now, 334.37, 334.44)]))
    good = w.last_quote
    h._ingest(json.dumps([quote("FDX", now + 300, 0, 334.44),
                          quote("FDX", now + 600, 334.50, 334.44),
                          quote("FDX", now + 900, 334.40, 0)]))
    check(w.last_quote == good,
          f"a crossed or one-sided quote became the current quote: "
          f"{w.last_quote} — the pane's whole vertical scale is built from it")
    check(w.crossed == 3,
          f"{w.crossed} of 3 unusable quotes were counted; discarding has to "
          f"be visible or a symbol quoting nothing but crossed markets looks "
          f"like one that simply is not quoting")
    check(w.quotes.n == 1, "an unusable quote reached the band's ring")


# ── the scale's input ───────────────────────────────────────────────────────
async def case_typical_spread_is_a_median():
    """One wide quote must not rescale the pane.

    The pane's height is a multiple of the typical spread. If that were a
    mean -- or worse, the last value -- a single print-through quote would
    make every trade in the pane collapse to the centre line for a second,
    on a wall where that reads as the symbol going still.
    """
    h = recording_hub()
    await h.wall_set(["FDX"])
    w = h.wall["FDX"]
    now = time.time() * 1000
    # Thirty seconds at a 7 cent spread, then one quote at 70.
    msgs = [quote("FDX", now - 30000 + i * 250, 334.37, 334.44)
            for i in range(120)]
    msgs.append(quote("FDX", now, 334.00, 334.70))
    h._ingest(json.dumps(msgs))
    sp, tp, mid = w.spreads(now, config.WALL_SPREAD_WINDOW_S * 1000)
    check(abs(sp - 70.0) < 0.01,
          f"the CURRENT spread is {sp:.2f}c, not 70c — it is a fact about now")
    check(abs(tp - 7.0) < 0.01,
          f"the TYPICAL spread moved to {tp:.2f}c on one wide quote; it is a "
          f"median of the last minute so the scale does not jump")
    check(abs(mid - 334.35) < 0.001,
          f"the mid is {mid}, not the current quote's midpoint")


async def case_spread_falls_back_to_the_last_quote():
    """A symbol that quoted once and went quiet still has a spread.

    This is the quiet name the wall is FOR. With no samples inside the
    window, a median of nothing is NaN, and a NaN scale draws an empty pane
    that is indistinguishable from a symbol with no data at all.
    """
    h = recording_hub()
    await h.wall_set(["QUIET"])
    w = h.wall["QUIET"]
    now = time.time() * 1000
    h._ingest(json.dumps([quote("QUIET", now - 300000, 10.00, 10.06)]))
    sp, tp, mid = w.spreads(now, config.WALL_SPREAD_WINDOW_S * 1000)
    check(abs(sp - 6.0) < 0.01 and abs(tp - 6.0) < 0.01,
          f"a five-minute-old quote gave ({sp}, {tp}); the spread is old, "
          f"not absent, and the pane must still be able to scale itself")
    check(abs(mid - 10.03) < 0.001, f"the mid is {mid}")

    # No quote at all is a different thing, and says so rather than guessing.
    h2 = recording_hub()
    await h2.wall_set(["NEW"])
    sp2, tp2, mid2 = h2.wall["NEW"].spreads(now, 60000)
    check(sp2 != sp2 and mid2 != mid2,
          f"a symbol that has never quoted reported ({sp2}, {mid2}) instead "
          f"of nothing; a made-up scale is worse than a blank pane")


# ── frames ──────────────────────────────────────────────────────────────────
async def case_cursor_is_a_count_not_a_timestamp(tmp: Path):
    """Seven prints in one millisecond are delivered once each.

    A timestamp cursor sends the first and drops the other six -- silently,
    and only on the symbols that print fastest, which are the ones being
    watched.
    """
    h = recording_hub()
    await h.wall_set(["FDX"])
    r = runner(h, tmp)
    sock = FakeSock()
    st = r.subscribe(sock)
    r.frame(st)                                # opening frame: nothing yet
    now = time.time() * 1000
    h._ingest(json.dumps([trade("FDX", now, 334.40 + i * 0.01, 1)
                          for i in range(7)]))
    f1 = r.frame(st)
    got = f1["syms"]["FDX"]["t"]
    check(len(got) == 7,
          f"a frame delivered {len(got)} of seven prints sharing a "
          f"millisecond — the cursor must be a COUNT")
    f2 = r.frame(st)
    check("FDX" not in f2["syms"],
          "the same seven prints were delivered twice; a redraw would stack "
          "duplicate bubbles on top of each other")


async def case_nothing_new_is_not_sent_but_the_frame_is(tmp: Path):
    """A silent symbol is absent; the frame still arrives.

    Both halves matter. Skipping the symbol is what lets the page leave that
    pane alone; skipping the FRAME would leave the page unable to tell a
    quiet market from a dead feed.
    """
    h = recording_hub()
    await h.wall_set(["FDX", "LLY"])
    r = runner(h, tmp)
    sock = FakeSock()
    st = r.subscribe(sock)
    r.frame(st)
    now = time.time() * 1000
    h._ingest(json.dumps([trade("FDX", now, 334.4, 100)]))
    f = r.frame(st)
    check(list(f["syms"]) == ["FDX"],
          f"the frame carried {list(f['syms'])}; LLY had nothing new and "
          f"must not be sent")
    f2 = r.frame(st)
    check(f2["ev"] == "tick" and f2["syms"] == {},
          f"a tick with nothing new was suppressed ({f2}); the page would "
          f"have no way to tell a quiet market from a stopped feed")


async def case_first_frame_carries_the_window_and_the_band(tmp: Path):
    """A pane opens drawn, not empty — including a band it can draw from.

    `q0` is the band's state at the LEFT EDGE: without it, a symbol that has
    not requoted inside the window draws no bid-ask at all, and a symbol
    whose first sample is thirty seconds in draws a band that begins in the
    middle of its own pane.
    """
    h = recording_hub()
    await h.wall_set(["FDX"])
    r = runner(h, tmp)
    now = time.time() * 1000
    h._ingest(json.dumps([
        quote("FDX", now - 300000, 334.37, 334.44),       # before the window
        trade("FDX", now - 400000, 334.40, 100),          # before the window
        trade("FDX", now - 30000, 334.41, 200),
    ]))
    st = r.subscribe(FakeSock())
    cell = r.frame(st)["syms"]["FDX"]
    check(cell.get("full") is True, "the first frame is not marked full")
    check(len(cell["t"]) == 1,
          f"the first frame carried {len(cell['t'])} trades; one of the two "
          f"is older than the window and must not be drawn")
    check(cell.get("q0") and abs(cell["q0"][1] - 334.37) < 1e-9,
          f"no band at the left edge ({cell.get('q0')}); this symbol has not "
          f"requoted inside the window, which is the quiet name the wall is "
          f"for")


async def case_changing_the_window_refills_it(tmp: Path):
    """A wider window re-sends, or the extra minutes stay blank until lived."""
    h = recording_hub()
    await h.wall_set(["FDX"])
    r = runner(h, tmp)
    now = time.time() * 1000
    h._ingest(json.dumps([trade("FDX", now - 100000, 334.4, 100),
                          trade("FDX", now - 10000, 334.5, 100)]))
    sock = FakeSock()
    st = r.subscribe(sock, window_s=60)
    first = r.frame(st)["syms"]["FDX"]["t"]
    check(len(first) == 1, f"a 60s window drew {len(first)} of two trades")
    r.set_window(sock, 180)
    again = r.frame(st)["syms"]["FDX"]["t"]
    check(len(again) == 2,
          f"widening the window delivered {len(again)} trades; the older one "
          f"is inside the new axis and the pane has never been sent it")


async def case_one_socket_carries_every_pane(tmp: Path):
    """A hundred panes, one connection. The service caps browsers at eight."""
    h = recording_hub()
    syms = [f"SYM{i:03d}" for i in range(100)]
    added, _, refused = await h.wall_set(syms)
    check(len(added) == 100 and not refused,
          f"the wall held {len(added)} of 100 symbols: {refused[:3]}")
    now = time.time() * 1000
    h._ingest(json.dumps([trade(s, now, 10.0, 100) for s in syms]
                         + [quote(s, now, 9.99, 10.01) for s in syms]))
    r = runner(h, tmp)
    sock = FakeSock()
    st = r.subscribe(sock)
    t0 = time.perf_counter()
    f = r.frame(st)
    ms = (time.perf_counter() - t0) * 1000
    check(len(f["syms"]) == 100,
          f"one frame carried {len(f['syms'])} of 100 symbols; the page opens "
          f"ONE socket for the whole wall")
    check(ms < 100,
          f"building a 100-symbol frame took {ms:.0f} ms — this runs once a "
          f"second in the same process as the tape and the live ladder")
    mb = h.wall_status()["buffer_mb"]
    check(mb < 64,
          f"100 symbols hold {mb:.1f} MB; the box has been OOM-killed twice")


async def case_status_names_truncation():
    """A ring at its ceiling draws a SHORT tape, which reads as a quiet one."""
    h = recording_hub()
    await h.wall_set(["FDX"])
    w = h.wall["FDX"]
    w.trades.cap_max = w.trades.cap          # forbid growing, force eviction
    now = time.time() * 1000
    h._ingest(json.dumps([trade("FDX", now + i, 334.4, 1)
                          for i in range(w.trades.cap + 50)]))
    st = h.wall_status()
    check(st["truncated"] == ["FDX"] and st["truncated_count"] == 1,
          f"a ring evicting inside its retention window was not reported: "
          f"{st['truncated']}")
    check(st["held"] == 1 and st["trades_held"] == w.trades.cap,
          f"status disagrees with the store: {st['held']}, "
          f"{st['trades_held']}")


# ── the watchlist ───────────────────────────────────────────────────────────
async def case_watchlist_round_trips(tmp: Path):
    """The list and its per-symbol overrides survive a restart.

    spx-live restarts on every deploy. A watchlist of a hundred hand-picked
    names that came back empty -- or came back without the overrides, which
    are the numbers that make FDX and LLY comparable -- would be rebuilt by
    hand several times a day.
    """
    path = tmp / "wl.json"
    s = WallStore(path)
    s.set([{"symbol": "fdx"}, {"symbol": "LLY", "scale": 0.35},
           {"symbol": "LLY"}, "NVDA"],
          {"window_s": 90, "spread_share": 0.45})
    s.save()

    back = WallStore(path)
    n = back.load()
    check(n == 3, f"{n} entries reloaded, 3 saved (LLY was listed twice)")
    check(back.symbols() == ["FDX", "LLY", "NVDA"],
          f"{back.symbols()} — a lowercase ticker must normalise")
    got = {e["symbol"]: e["scale"] for e in back.entries}
    check(got["LLY"] == 0.35,
          f"LLY's override came back {got['LLY']}; it is the number that "
          f"makes a 50 cent spread comparable with a 7 cent one")
    check(got["FDX"] is None,
          f"FDX invented an override of {got['FDX']} instead of following "
          f"the page's share")
    check(back.settings["window_s"] == 90
          and back.settings["spread_share"] == 0.45,
          f"the page's settings came back {back.settings}")


async def case_watchlist_clamps_and_refuses(tmp: Path):
    """Out-of-range numbers are clamped; nonsense is refused, by name."""
    s = WallStore(tmp / "wl2.json")
    refused = s.set([{"symbol": "FDX", "scale": 9.0},
                     {"symbol": "LLY", "scale": -1},
                     {"symbol": "NVDA", "scale": "x"},
                     {"symbol": "BRK B"}, {"symbol": ""}],
                    {"window_s": 99999})
    got = {e["symbol"]: e["scale"] for e in s.entries}
    check(got.get("FDX") == config.WALL_SHARE_MAX
          and got.get("LLY") == config.WALL_SHARE_MIN,
          f"shares were not clamped: {got}")
    check(got.get("NVDA") is None,
          f"an unparseable share became {got.get('NVDA')}")
    check(len(refused) == 2 and any("BRK B" in r for r in refused),
          f"refusals {refused} — a refused ticker must be named, or the page "
          f"shows a list that is quietly shorter than what was typed")
    check(s.settings["window_s"] == config.WALL_RETAIN_S,
          f"a window of 99999s was stored as {s.settings['window_s']}; the "
          f"store holds {config.WALL_RETAIN_S}s of tape and an axis longer "
          f"than the buffer draws emptiness as quiet")

    # The cap holds, and says so.
    s2 = WallStore(tmp / "wl3.json")
    ref2 = s2.set([{"symbol": f"S{i:04d}"}
                   for i in range(config.WALL_MAX_SYMBOLS + 5)])
    check(len(s2.entries) == config.WALL_MAX_SYMBOLS and len(ref2) == 5,
          f"the {config.WALL_MAX_SYMBOLS}-symbol cap let "
          f"{len(s2.entries)} through with {len(ref2)} refusals")


async def case_a_corrupt_watchlist_is_named(tmp: Path):
    """Half a file, or a hand-edited one, must not read as an empty list."""
    path = tmp / "bad.json"
    path.write_text('{"entries": [{"symbol": "FDX"}', encoding="utf-8")
    s = WallStore(path)
    n = s.load()
    check(n == 0 and s.entries == [], "a corrupt file loaded entries")
    check(s.last_error,
          "a corrupt watchlist was swallowed; 'my list is gone' and 'my list "
          "did not load' want different answers from the journal")
    check(s.status()["error"] and s.status()["entries"] == 0,
          f"status hides the failure: {s.status()}")

    # A missing file is the first run, not a fault.
    s2 = WallStore(tmp / "absent.json")
    check(s2.load() == 0 and s2.last_error is None,
          f"a missing file was reported as an error: {s2.last_error}")


async def case_apply_keeps_hub_file_and_pages_in_step(tmp: Path):
    """One call sets the tier, the file and every open page.

    Two of the three would be enough to be wrong in a way nobody sees until
    the next deploy.
    """
    h = recording_hub()
    r = runner(h, tmp)
    a = FakeSock()
    b = FakeSock()
    r.subscribe(a)
    r.subscribe(b)
    out = await r.apply([{"symbol": "FDX"}, {"symbol": "LLY", "scale": 0.3}],
                        {"spread_share": 0.5})
    check(sorted(h.wall) == ["FDX", "LLY"],
          f"the hub holds {sorted(h.wall)} after apply")
    saved = json.loads((tmp / "wall_watchlist.json").read_text(encoding="utf-8"))
    check([e["symbol"] for e in saved["entries"]] == ["FDX", "LLY"],
          f"the file holds {saved['entries']}")
    check(saved["settings"]["spread_share"] == 0.5,
          f"the settings were not saved: {saved['settings']}")
    for name, sock in (("the editing page", a), ("the OTHER page", b)):
        evs = [f["ev"] for f in sock.frames]
        check("watchlist" in evs,
              f"{name} was not told the list changed ({evs}); two tabs would "
              f"disagree about which symbols exist")
    check(out["added"] == ["FDX", "LLY"] and not out["refused"],
          f"apply reported {out['added']} / {out['refused']}")

    # A removal clears the socket's cursor, or the symbol's last records
    # would be re-sent if it were added back.
    await r.apply([{"symbol": "FDX"}])
    st = r._subs[a]
    check("LLY" not in st["cursors"],
          f"a dropped symbol left a cursor behind: {list(st['cursors'])}")
    check(sorted(h.wall) == ["FDX"], f"the hub holds {sorted(h.wall)}")


async def case_restore_holds_the_saved_list(tmp: Path):
    """Startup: the file is read and the symbols are held before any page."""
    path = tmp / "restore.json"
    s = WallStore(path)
    s.set([{"symbol": "FDX"}, {"symbol": "LLY", "scale": 0.3}])
    s.save()
    h = recording_hub()
    r = WallRunner(h, WallStore(path))
    n = await r.restore()
    check(n == 2 and sorted(h.wall) == ["FDX", "LLY"],
          f"restore loaded {n} entries and held {sorted(h.wall)}")
    subs = [p["params"] for p in h.sent if p["action"] == "subscribe"]
    check(subs == ["T.FDX,Q.FDX,T.LLY,Q.LLY"],
          f"restore subscribed {subs} — one message, both channels")
    check(r.state()["entries"][1]["scale"] == 0.3,
          "the override did not survive into the state the page reads")


# ── what this page is not ───────────────────────────────────────────────────
async def case_the_wall_never_talks_to_a_broker():
    """It is for watching. Nothing here places, moves or cancels an order.

    A SOURCE SCAN, not a behavioural check, because the property is "this
    code cannot do that at all" -- and the cheapest way for it to stop being
    true is an import added while borrowing something from the tape page,
    which no behavioural check would ever see.

    Parsed rather than grepped: `os.replace` writes the watchlist and
    `CancelledError` is how the loop stops, so a substring scan for the verbs
    finds nothing but false alarms. What is banned is a reference to the
    broker -- an import of it, or any name or attribute that mentions it.
    """
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
        check(not refs,
              f"live/{name} references {sorted(set(refs))} — the wall "
              f"watches; nothing on it may reach a broker")

    # And the page's own endpoints. `_status()` carries the trading state for
    # the tape page, so the wall's routes are checked directly rather than
    # the whole module.
    main = ast.parse((ROOT / "live" / "main.py").read_text(encoding="utf-8"))
    seen = []
    for node in ast.walk(main):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) \
                and node.name.startswith("wall_"):
            seen.append(node.name)
            refs = broker_refs(node)
            check(not refs,
                  f"live/main.py:{node.name}() references {sorted(set(refs))}"
                  f" — a wall endpoint must not reach a broker")
    check(sorted(seen) == ["wall_page", "wall_watchlist", "wall_watchlist_set",
                           "wall_ws"],
          f"the wall's endpoints are {sorted(seen)}; this check scans routes "
          f"by name and a renamed one would be scanned by nothing")


# ── the page ────────────────────────────────────────────────────────────────
#
# The shipped JS is EXECUTED, not read. The scale is the page's whole claim --
# that a half-spread move looks the same on a 7 cent name and a 50 cent one --
# and it is arithmetic, so it can be checked exactly rather than by looking at
# a wall and forming an impression.

JS = ROOT / "static" / "js" / "equities_wall.js"

JS_DRIVER = r"""
const fs = require('fs');
global.document = { addEventListener: () => {} };
const src = fs.readFileSync(process.argv[1], 'utf8');
const tail = `
  const out = {};
  // FDX at 7c and LLY at 50c, both at 60% of the pane.
  const fdx = wlScale(334.40, 7.0, 0.60, 0);
  const lly = wlScale(1158.60, 50.0, 0.60, 0);
  out.fdxShare = (0.07 / fdx.range);
  out.llyShare = (0.50 / lly.range);
  out.fdxCentred = Math.abs((fdx.lo + fdx.hi) / 2 - 334.40);
  out.llyCentred = Math.abs((lly.lo + lly.hi) / 2 - 1158.60);
  // A half-spread move is the same FRACTION of the pane in both.
  out.fdxHalf = (0.035 / fdx.range);
  out.llyHalf = (0.25 / lly.range);
  // An override changes only its own pane.
  out.override = (0.50 / wlScale(1158.60, 50.0, 0.30, 0).range);
  out.effOwn = wlEffShare({symbol:'LLY', scale:0.3}, 0.6);
  out.effPage = wlEffShare({symbol:'FDX', scale:null}, 0.6);
  out.effClamped = wlEffShare({symbol:'X', scale:9}, 0.6);
  // No quote: scaled to its own prints, and it says so.
  out.noQuoteFrom = wlScale(10.0, null, 0.6, 0.20).from;
  out.quoteFrom = fdx.from;
  // Re-centring: an EMA, except across a gap.
  out.ema = wlCentre(100.0, 100.10, 0.20);
  out.jump = wlCentre(100.0, 101.00, 0.20);
  out.first = wlCentre(0, 100.0, 0.20);
  // Bubbles: monotone, bounded, and a one-lot is still visible.
  out.r1 = wlBubbleR(1);
  out.r100 = wlBubbleR(100);
  out.r1000 = wlBubbleR(1000);
  out.r1e6 = wlBubbleR(1e6);
  // Redraw rules.
  out.dueOffscreen = wlDue({dirty:true, drawnAt:0}, 10000, false);
  out.dueDirty = wlDue({dirty:true, drawnAt:9999}, 10000, true);
  out.dueQuiet = wlDue({dirty:false, drawnAt:9000}, 10000, true);
  out.dueStale = wlDue({dirty:false, drawnAt:0}, 10000, true);
  out.forceMs = WL_FORCE_REDRAW_MS;
  // Trimming, and the symbol box.
  out.trim = wlTrim([[1,1,1],[5,1,1],[9,1,1]], 5).length;
  out.parse = wlParseSymbols('fdx, lly\\nNVDA fdx  BRK.B');
  out.fmt = [wlFmtSpread(7), wlFmtSpread(50), wlFmtSpread(null)];
  // ── the spread filter ────────────────────────────────────────────────
  // THE MEDIAN, NOT THE TICK: sp is wide, tp is narrow, and the pane fades.
  const wide_now = {sym:'X', sp: 20, tp: 2, belowSince: 0, trades:[1,2,3]};
  const t0 = 1000000;
  out.offNever = wlDimState({sym:'X', sp:1, tp:1}, 0, t0).dim;
  const a = wlDimState(wide_now, 5, t0);          // first tick below
  out.notYet = a.dim;
  wide_now.belowSince = a.belowSince;
  out.holdMid = wlDimState(wide_now, 5, t0 + 9000).dim;
  out.dimsAfter = wlDimState(wide_now, 5, t0 + 10000).dim;
  out.holdMs = WL_DIM_AFTER_MS;
  // QUALIFYING UNDIMS AT ONCE, and clears the timer with it.
  const back = wlDimState({sym:'X', sp:1, tp:9, belowSince: t0}, 5, t0 + 30000);
  out.undim = back.dim;
  out.undimResets = back.belowSince;
  // Never quoted: cannot be shown to qualify.
  out.unknown = wlDimState({sym:'Q', sp:null, tp:null, belowSince: t0},
                           5, t0 + 20000).dim;
  // AND THE TAPE IS UNTOUCHED. The filter draws; it does not drop data.
  out.keptTrades = wide_now.trades.length;
  globalThis.__out = out;
`;
eval(src + tail);
process.stdout.write(JSON.stringify(globalThis.__out));
"""


async def case_js_scale_makes_names_comparable():
    """The shipped JS, executed: the spread fills the share it is given.

    THE PAGE'S WHOLE CLAIM. "Bouncy" is relative to the spread's own width,
    so a half-spread move must occupy the same fraction of the pane on FDX
    (7c) and LLY (50c). On a shared axis it does not, and the wide name looks
    like everything while the narrow one looks like nothing.
    """
    import shutil
    import subprocess
    if shutil.which("node") is None:
        FAILS.append("node is not installed — the shipped JS was NOT executed")
        return
    p = subprocess.run(["node", "-e", JS_DRIVER, str(JS)],
                       capture_output=True, text=True, encoding="utf-8")
    if p.returncode:
        FAILS.append(f"the page JS did not run: {p.stderr.strip()[:300]}")
        return
    out = json.loads(p.stdout)

    check(abs(out["fdxShare"] - 0.60) < 1e-9 and abs(out["llyShare"] - 0.60) < 1e-9,
          f"the spread fills {out['fdxShare']:.3f} of FDX's pane and "
          f"{out['llyShare']:.3f} of LLY's, asked for 0.60")
    check(abs(out["fdxHalf"] - out["llyHalf"]) < 1e-12,
          f"a half-spread move is {out['fdxHalf']:.4f} of FDX's pane and "
          f"{out['llyHalf']:.4f} of LLY's — the two are supposed to look the "
          f"same, and the spread NUMBER is what tells them apart")
    check(out["fdxCentred"] < 1e-9 and out["llyCentred"] < 1e-9,
          f"the window is not centred on the mid: {out['fdxCentred']}, "
          f"{out['llyCentred']}")
    check(abs(out["override"] - 0.30) < 1e-9,
          f"a per-pane override of 0.30 gave {out['override']:.3f}")
    check(out["effOwn"] == 0.3 and out["effPage"] == 0.6,
          f"the override is not preferred over the page setting: "
          f"{out['effOwn']}, {out['effPage']}")
    check(out["effClamped"] <= 0.95,
          f"a share of 9 was not clamped ({out['effClamped']}); a spread "
          f"filling nine panes leaves nowhere for the trades")
    check(out["quoteFrom"] == "spread" and out["noQuoteFrom"] == "trades",
          f"a pane with no quote does not say so ({out['noQuoteFrom']}); it "
          f"is not scaled to a spread and must not look like one that is")


async def case_js_redraw_and_recentre():
    """Off screen is never drawn; nothing new waits; a gap does not crawl."""
    import shutil
    import subprocess
    if shutil.which("node") is None:
        FAILS.append("node is not installed — the shipped JS was NOT executed")
        return
    p = subprocess.run(["node", "-e", JS_DRIVER, str(JS)],
                       capture_output=True, text=True, encoding="utf-8")
    if p.returncode:
        FAILS.append(f"the page JS did not run: {p.stderr.strip()[:300]}")
        return
    out = json.loads(p.stdout)

    check(out["dueOffscreen"] is False,
          "a pane scrolled off the screen was redrawn; at a hundred panes "
          "that is most of the page's work done for nobody")
    check(out["dueDirty"] is True,
          "a pane with new data was not redrawn")
    check(out["dueQuiet"] is False,
          "a pane with nothing new was redrawn a second later; the server "
          "does not even send it")
    check(out["dueStale"] is True and out["forceMs"] <= 10000,
          f"a quiet pane is never refreshed (force={out['forceMs']} ms); its "
          f"bubbles would sit still while the window slid out from under "
          f"them, and old prints would stay on screen — a quiet name looking "
          f"busier than it is")
    check(abs(out["ema"] - 100.035) < 1e-9,
          f"re-centring is not an EMA: {out['ema']}")
    check(out["jump"] == 101.0,
          f"a gap bigger than the pane crawled toward the new mid "
          f"({out['jump']}); every print would be off the top of the pane "
          f"for several seconds, which is the moment worth watching")
    check(out["first"] == 100.0, f"the first centre was {out['first']}")
    check(out["r1"] >= 1.0 and out["r100"] > out["r1"]
          and out["r1000"] > out["r100"] and out["r1e6"] <= 9.0,
          f"bubble radii {out['r1']}, {out['r100']}, {out['r1000']}, "
          f"{out['r1e6']} — 91% of this tape is under 40 shares and a one-lot "
          f"must still be visible, while a block must not swallow the pane")
    check(out["trim"] == 2, f"the window trim kept {out['trim']} of 3")
    check(out["parse"] == ["FDX", "LLY", "NVDA"],
          f"the symbol box parsed {out['parse']} — duplicates and a dotted "
          f"class ticker the feed does not take must not reach the server")
    check(out["fmt"] == ["7.0c", "50c", "—"],
          f"the spread reads {out['fmt']}")


async def case_the_spread_filter_fades_and_does_not_flicker():
    """Below the floor fades the pane; it never removes or unsubscribes it.

    THE FLICKER IS THE WHOLE DIFFICULTY. A name sitting on the threshold
    would dim and undim every few seconds on single quotes, and on a wall of
    a hundred that is the page becoming unreadable — so the comparison is
    against the server's MEDIAN of the last minute (`tp`), never the
    instantaneous spread (`sp`), and a pane has to be below for ten seconds
    before it fades while qualifying undims it at once.
    """
    import shutil
    import subprocess
    if shutil.which("node") is None:
        FAILS.append("node is not installed — the shipped JS was NOT executed")
        return
    p = subprocess.run(["node", "-e", JS_DRIVER, str(JS)],
                       capture_output=True, text=True, encoding="utf-8")
    if p.returncode:
        FAILS.append(f"the page JS did not run: {p.stderr.strip()[:300]}")
        return
    out = json.loads(p.stdout)

    check(out["offNever"] is False,
          "a floor of 0 faded a pane; the filter is off at zero")
    check(out["notYet"] is False and out["holdMid"] is False,
          f"a pane faded before its hold elapsed ({out['notYet']}, "
          f"{out['holdMid']}) — a name on the boundary would strobe")
    check(out["dimsAfter"] is True and out["holdMs"] == 10000,
          f"a pane below the floor for {out['holdMs']} ms did not fade "
          f"({out['dimsAfter']})")
    check(out["undim"] is False and out["undimResets"] == 0,
          f"a name that qualifies again was not restored at once "
          f"({out['undim']}) or kept its timer ({out['undimResets']}), which "
          f"would fade it again the moment it dipped")
    # THE MEDIAN IS THE INPUT. This case's pane has a 20c last quote and a 2c
    # median: reading `sp` would leave it bright.
    check(out["dimsAfter"] is True,
          "the filter read the instantaneous spread rather than the median")
    check(out["unknown"] is True,
          "a symbol that has never quoted stayed at full strength under a "
          "spread floor; it cannot be shown to qualify")
    check(out["keptTrades"] == 3,
          f"the filter touched the pane's tape ({out['keptTrades']} of 3 "
          f"records) — a name that dips under the floor must keep its two "
          f"minutes and not rebuild when it comes back")


async def case_the_filter_does_not_touch_subscriptions(tmp: Path):
    """Setting the floor changes nothing the hub holds.

    The whole point of fading rather than filtering: a symbol under the
    threshold goes on streaming, so when it widens again its window is
    already full. A filter that reached the tier would give back an empty
    pane and two minutes of waiting.
    """
    h = recording_hub()
    r = runner(h, tmp)
    await r.apply([{"symbol": "FDX"}, {"symbol": "KO"}])
    h.sent.clear()
    before = sorted(h.wall)
    out = await r.apply(r.store.entries, {"min_spread_cents": 8})
    check(sorted(h.wall) == before == ["FDX", "KO"],
          f"the tier changed when the floor moved: {sorted(h.wall)}")
    check(not out["added"] and not out["dropped"],
          f"the floor added {out['added']} and dropped {out['dropped']}")
    check(h.sent == [],
          f"the floor sent {params(h)!r} upstream; a presentation setting "
          f"must not reach the socket")
    check(r.store.settings["min_spread_cents"] == 8,
          f"the floor was not saved: {r.store.settings}")

    # AND IT SURVIVES A RESTART, like the window and the share.
    back = WallStore(tmp / "wall_watchlist.json")
    back.load()
    check(back.settings["min_spread_cents"] == 8,
          f"the floor did not survive a reload: {back.settings}")
    # Nonsense and out-of-range values clamp rather than refusing the list.
    s = WallStore(tmp / "floor.json")
    s.set([{"symbol": "FDX"}], {"min_spread_cents": -4})
    lo = s.settings["min_spread_cents"]
    s.set([{"symbol": "FDX"}], {"min_spread_cents": 9e9})
    hi = s.settings["min_spread_cents"]
    s.set([{"symbol": "FDX"}], {"min_spread_cents": "wide"})
    bad = s.settings["min_spread_cents"]
    check(lo == 0 and hi == config.WALL_MAX_SPREAD_FILTER and bad == hi,
          f"the floor did not clamp: {lo}, {hi}, and an unparseable value "
          f"became {bad} instead of leaving the setting alone")

    # THE PANES STAY WHERE THEY ARE. The page draws every entry and marks the
    # failing ones; it does not draw a filtered list.
    html = (ROOT / "templates" / "equities_wall.html").read_text(encoding="utf-8")
    check('x-for="e in entries"' in html,
          "the grid iterates something other than the whole list, so panes "
          "would reflow as names cross the threshold")
    # THE CLASS THE TEMPLATE ADDS IS THE CLASS THAT FADES. Tested as one
    # fact rather than two: a substring check for the rule passed happily
    # when the CSS was renamed to `.wl-pane.dimmed` and the binding still
    # added `dim`, which is a pane that never fades at all.
    import re
    bound = re.search(r"dim \? '([a-z-]+)'", html)
    styled = re.search(r"\.wl-pane\.([a-z-]+)\s*\{[^}]*opacity", html)
    check(bound is not None and styled is not None
          and bound.group(1) == styled.group(1),
          f"the pane's faded class does not match its rule: the template "
          f"adds {bound.group(1) if bound else None!r} and the stylesheet "
          f"fades {styled.group(1) if styled else None!r}")


async def case_the_page_holds_one_connection():
    """One socket for the whole wall, and one timer that draws it.

    The service caps browser connections at MAX_CLIENTS (8). A socket per
    pane is twelve times over at a hundred panes, and the ninth pane would
    simply be refused — with the page looking like eight live names and
    ninety-two dead ones.
    """
    src = JS.read_text(encoding="utf-8")
    code = "\n".join(ln for ln in src.splitlines()
                     if not ln.lstrip().startswith(("*", "/*", "//")))
    check(code.count("new WebSocket") == 1,
          f"the page JS constructs {code.count('new WebSocket')} WebSockets; "
          f"one page is one connection, whatever the pane count")
    check(code.count("setInterval") == 1,
          f"the page JS starts {code.count('setInterval')} timers; every pane "
          f"is drawn from the one frame loop")
    # And the draw loop is the thing that runs once a second, not a
    # requestAnimationFrame chain pretending to be throttled.
    check("requestAnimationFrame" not in code,
          "the wall redraws on animation frames; a hundred canvases at 60fps "
          "is a page that melts a laptop to show tape read in glances")
    check("drawAll(), 1000" in code.replace(" ", "").replace("=>this.", "")
          or "drawAll(), 1000" in code,
          "the draw loop does not run at one frame a second")


async def case_the_page_cannot_trade():
    """Nothing on the page reaches a broker — the markup included."""
    for rel in ("static/js/equities_wall.js", "templates/equities_wall.html"):
        text = (ROOT / rel).read_text(encoding="utf-8").lower()
        # Named precisely, because the page legitimately says "placeholder"
        # and a scan for "place" would fail on the symbol box's own hint.
        for tok in ("/broker", "brokercall", "armed", "flatten",
                    "sendorder", "placeorder", "sendmove", "ladder"):
            check(tok not in text,
                  f"{rel} mentions {tok!r}; the wall is for watching, and "
                  f"the ladder, the arming and the order entry live on the "
                  f"tape page")


async def case_the_colours_come_from_the_tape_page():
    """A wall pane is a small Equities Live pane, so the colours are shared.

    ONE DEFINITION, in static/js/tape_theme.js, read by both bundles. Two
    copies agree until one of them is edited, and the drift is invisible
    until the two pages are open side by side — which is exactly how this
    page is used.
    """
    theme = (ROOT / "static" / "js" / "tape_theme.js").read_text(encoding="utf-8")
    for name in ("TAPE_BID", "TAPE_ASK", "TAPE_TRADE_FILL", "TAPE_TRADE_RIM"):
        check(f"const {name}" in theme,
              f"{name} is not defined in tape_theme.js")

    wall = (ROOT / "static" / "js" / "equities_wall.js").read_text(encoding="utf-8")
    live = (ROOT / "static" / "js" / "equities_live.js").read_text(encoding="utf-8")
    check("TAPE_BID" in wall and "TAPE_ASK" in wall,
          "the wall does not draw its bid and ask from the shared colours")
    check("TAPE_TRADE_FILL" in wall and "TAPE_TRADE_RIM" in wall,
          "the wall's prints are not the tape page's neutral grey")
    check("TAPE_BID" in live and "TAPE_TRADE_FILL" in live,
          "Equities Live no longer reads the shared colours, so the file "
          "that exists to keep the two together is keeping only one of them")

    # THE LITERALS LIVE IN ONE FILE. A colour written out again anywhere else
    # is the drift this is meant to prevent, whichever page writes it.
    for lit in ("rgba(130,190,235", "rgba(235,150,190",
                "rgba(206,212,220", "rgba(228,233,240"):
        where = [f.name for f in (ROOT / "static" / "js").glob("*.js")
                 if lit in f.read_text(encoding="utf-8")]
        check(where == ["tape_theme.js"],
              f"{lit}…) is written out in {where}; the tape colours are "
              f"defined once, in tape_theme.js")

    # BOTH BUNDLES LOAD against the shared file, and neither loads without
    # it. The tape page's constants are TOP LEVEL, so getting the order wrong
    # is not a wrong colour — it is a ReferenceError before Alpine starts and
    # a blank trading page.
    import shutil
    import subprocess
    if shutil.which("node") is None:
        FAILS.append("node is not installed — the bundles were NOT loaded")
    else:
        for name in ("equities_wall.js", "equities_live.js"):
            src = f'global.document={{addEventListener:()=>{{}}}};' \
                  f'global.window={{}};' \
                  f'eval(require("fs").readFileSync({str(ROOT / "static" / "js" / "tape_theme.js")!r},"utf8")' \
                  f'+require("fs").readFileSync({str(ROOT / "static" / "js" / name)!r},"utf8"))'
            p = subprocess.run(["node", "-e", src], capture_output=True,
                               text=True, encoding="utf-8")
            check(p.returncode == 0,
                  f"{name} does not load beside tape_theme.js: "
                  f"{p.stderr.strip()[:200]}")
        alone = f'global.document={{addEventListener:()=>{{}}}};' \
                f'global.window={{}};' \
                f'eval(require("fs").readFileSync({str(ROOT / "static" / "js" / "equities_live.js")!r},"utf8"))'
        p = subprocess.run(["node", "-e", alone], capture_output=True,
                           text=True, encoding="utf-8")
        check(p.returncode != 0 and "TAPE_" in p.stderr,
              "the tape bundle loads without tape_theme.js, so it is not "
              "really reading the shared colours and the two pages can still "
              "drift")

    # NO FILL BETWEEN THE LINES. Two lines and nothing between, as on the
    # tape page — a shaded band made the wall read as something else at a
    # glance across a hundred panes.
    draw = wall.split("── the prints")[0]
    check("fillRect" not in draw,
          "the wall shades the spread between the bid and the ask; the tape "
          "page draws two lines and nothing between them")
    # And the page's own blue/pink are gone with it.
    check("WL_BLUE" not in wall and "WL_PINK" not in wall,
          "the wall still carries its own copy of the tape's blue and pink")


async def case_the_page_is_wired_up():
    """The template names the component, the route serves it, nav links it."""
    html = (ROOT / "templates" / "equities_wall.html").read_text(encoding="utf-8")
    check('x-data="equitiesWall"' in html, "the page declares no component")
    check("equities_wall.js" in html and "defer src={{ asset" not in html,
          "the page bundle is missing, or deferred (which would register the "
          "component after alpine:init has already fired)")
    # The shared colours have to be READ before the bundle that reads them.
    for page, bundle in (("equities_wall.html", "equities_wall.js"),
                         ("equities_live.html", "equities_live.js")):
        text = (ROOT / "templates" / page).read_text(encoding="utf-8")
        i, j = text.find("tape_theme.js"), text.find(bundle + "') }}")
        check(i != -1 and j != -1 and i < j,
              f"{page} does not load tape_theme.js before {bundle}; the "
              f"bundle's colour constants would be a ReferenceError and the "
              f"page would not start at all")
    check('x-ref="grid"' in html and "wl-pane" in html,
          "the grid the draw loop walks is not in the markup")
    main = (ROOT / "live" / "main.py").read_text(encoding="utf-8")
    check('"equities_wall.html"' in main and '@app.get("/wall"' in main,
          "no route serves the page")
    # THE COMPONENT IS INITIALISED ONCE.
    #
    # Alpine 3 calls a data object's own init() automatically. The other
    # pages in this app ALSO name it in x-init, which runs it twice — on this
    # page that meant two WebSockets and two draw loops per tab, measured by
    # rendering the page in a browser and counting the sockets (2, then 1).
    # No gate that reads source could see it, so this one holds the rule for
    # the page where a second connection is the thing the design forbids.
    js = (ROOT / "static" / "js" / "equities_wall.js").read_text(encoding="utf-8")
    declares = "init() {" in js
    body = "\n".join(ln for ln in html.splitlines()
                     if "NO `x-init" not in ln and not ln.lstrip().startswith("#"))
    check(declares and 'x-init="init()"' not in body,
          "the component's init() is named in x-init as well as being "
          "Alpine's own hook, so it runs twice — two sockets and two draw "
          "loops on a page whose whole design is one of each")

    nav = (ROOT / "templates" / "_nav.html").read_text(encoding="utf-8")
    check("/wall" in nav,
          "the wall is not on the nav; a page nothing links to is one nobody "
          "opens")


CASES = [
    ("tiers share one socket",   case_tiers_do_not_unsubscribe_each_other),
    ("no duplicate subscribe",   case_wall_does_not_resubscribe_a_held_symbol),
    ("reconnect restores it",    case_reconnect_restores_the_wall),
    ("no aggregation",           case_trades_reach_the_wall_unaggregated),
    ("ingest feeds every tier",  case_ingest_feeds_every_tier_that_holds_it),
    ("quotes sampled, now is not", case_quotes_are_sampled_but_now_is_not),
    ("crossed quotes dropped",   case_a_crossed_quote_is_dropped_and_counted),
    ("typical spread is median", case_typical_spread_is_a_median),
    ("spread falls back",        case_spread_falls_back_to_the_last_quote),
    ("cursor is a count",        case_cursor_is_a_count_not_a_timestamp),
    ("silence is not sent",      case_nothing_new_is_not_sent_but_the_frame_is),
    ("first frame draws",        case_first_frame_carries_the_window_and_the_band),
    ("window change refills",    case_changing_the_window_refills_it),
    ("one socket, 100 panes",    case_one_socket_carries_every_pane),
    ("truncation is named",      case_status_names_truncation),
    ("watchlist round-trips",    case_watchlist_round_trips),
    ("watchlist clamps",         case_watchlist_clamps_and_refuses),
    ("corrupt list is named",    case_a_corrupt_watchlist_is_named),
    ("apply keeps all in step",  case_apply_keeps_hub_file_and_pages_in_step),
    ("restore holds the list",   case_restore_holds_the_saved_list),
    ("never talks to a broker",  case_the_wall_never_talks_to_a_broker),
    ("the scale is comparable",  case_js_scale_makes_names_comparable),
    ("redraw and re-centre",     case_js_redraw_and_recentre),
    ("spread floor fades",       case_the_spread_filter_fades_and_does_not_flicker),
    ("floor spares the tier",    case_the_filter_does_not_touch_subscriptions),
    ("one connection, one loop", case_the_page_holds_one_connection),
    ("the page cannot trade",    case_the_page_cannot_trade),
    ("colours are shared",       case_the_colours_come_from_the_tape_page),
    ("the page is wired up",     case_the_page_is_wired_up),
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
