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
    check(sorted(seen) == ["wall_watchlist", "wall_watchlist_set", "wall_ws"],
          f"the wall's endpoints are {sorted(seen)}; this check scans routes "
          f"by name and a renamed one would be scanned by nothing")


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
