"""The wall's watchlist, kept across a restart.

WHY IT IS SERVER-SIDE. The list is the page -- a hundred names chosen over
weeks -- and local storage loses it to a cleared browser, a second machine, or
the other monitor. spx-live also restarts on every deploy, several times a
day; a watchlist that came back empty would have to be rebuilt by hand each
time, which is the same argument the scan's history file already won.

WHY A FILE AND NOT POSTGRES. It is one small object, written when a person
edits a list and read once at startup, by one process, with no query beyond
"give me the list". A table, a pool and a migration for that is machinery
bought for nothing. The scan's history made the same call for 6.6 MB a day.

WHAT AN ENTRY CARRIES. The ticker and its SCALE OVERRIDE, together, because
the override is a fact about that symbol on this wall -- "LLY needs more room
than the default" -- and storing it anywhere else is a second list to keep in
step with the first. An entry with no override uses the page's share.

WRITTEN WHOLE, VIA A TEMPORARY FILE AND A RENAME. A partial write is a
watchlist that loads as a syntax error, and the replacement for that is not a
backup, it is never having half a file on disk.
"""
from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path

from live import config

log = logging.getLogger("live.wall_store")

VERSION = 1


def clean_symbol(s) -> str:
    return (s or "").strip().upper()


def clean_share(v, default=None):
    """A share of pane height, or None to follow the page's setting.

    Clamped rather than refused: the bound is a drawing limit (a spread that
    fills 99% of the pane leaves nowhere for the trades to be), and a stored
    list that refuses to load because one number is out of range would cost
    the whole watchlist.
    """
    if v is None or v == "":
        return default
    try:
        f = float(v)
    except (TypeError, ValueError):
        return default
    if f != f:                                            # NaN
        return default
    return min(config.WALL_SHARE_MAX, max(config.WALL_SHARE_MIN, f))


def clean_spread_floor(v, default=None):
    """A minimum spread in cents, or 0 for no filter.

    Clamped rather than refused, like the share: it is a DRAWING threshold,
    and a watchlist that refuses to load because one number is out of range
    costs the whole list.
    """
    default = config.WALL_MIN_SPREAD_CENTS if default is None else default
    try:
        f = float(v)
    except (TypeError, ValueError):
        return default
    if f != f:
        return default
    return min(config.WALL_MAX_SPREAD_FILTER, max(0.0, f))


def clean_window(v, default=None):
    default = config.WALL_WINDOW_S if default is None else default
    try:
        f = float(v)
    except (TypeError, ValueError):
        return default
    if f != f or f <= 0:
        return default
    # The store keeps WALL_RETAIN_S of tape; a window longer than that would
    # draw an axis the buffer cannot fill, which is the "buffering 55s of
    # 180s" complaint the pin set was invented to answer.
    return min(config.WALL_RETAIN_S, max(10.0, f))


class WallStore:
    """The watchlist and the page's settings. One instance, loaded at start."""

    def __init__(self, path: str | Path | None = None):
        self.path = Path(path or config.WALL_STORE_PATH)
        self.entries: list[dict] = []
        self.settings: dict = {
            "window_s": config.WALL_WINDOW_S,
            "spread_share": config.WALL_SPREAD_SHARE,
            # PRESENTATION ONLY. It is kept beside the other two because it is
            # the same kind of thing -- how the wall is drawn, saved with the
            # list so it survives a refresh and a restart -- but unlike them
            # nothing downstream of it reaches the hub: see the note on
            # WALL_MIN_SPREAD_CENTS.
            "min_spread_cents": config.WALL_MIN_SPREAD_CENTS,
        }
        self.updated: float | None = None
        self.loaded_from: str | None = None
        self.last_error: str | None = None
        self.writes = 0

    # ── reading ─────────────────────────────────────────────────────────
    def symbols(self) -> list[str]:
        return [e["symbol"] for e in self.entries]

    def state(self) -> dict:
        return {
            "entries": [dict(e) for e in self.entries],
            "settings": dict(self.settings),
            "updated": self.updated,
            "path": str(self.path),
            "loaded_from": self.loaded_from,
            "error": self.last_error,
        }

    def load(self) -> int:
        """Read the file. Returns the number of entries restored.

        A missing file is the first run and is not an error. A CORRUPT file
        is: it is named and kept, rather than quietly replaced with an empty
        list, because "my watchlist is gone" and "my watchlist did not load"
        want different answers from whoever reads the journal.
        """
        self.last_error = None
        if not self.path.is_file():
            return 0
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            entries = raw.get("entries") or []
            if not isinstance(entries, list):
                raise ValueError("entries is not a list")
            self.entries = self._clean_entries(entries)[0]
            st = raw.get("settings") or {}
            self.settings = {
                "window_s": clean_window(st.get("window_s")),
                "spread_share": clean_share(st.get("spread_share"),
                                            config.WALL_SPREAD_SHARE),
                "min_spread_cents": clean_spread_floor(
                    st.get("min_spread_cents")),
            }
            self.updated = raw.get("updated")
            self.loaded_from = str(self.path)
        except Exception as exc:                          # noqa: BLE001
            self.last_error = f"{type(exc).__name__}: {exc}"
            log.warning("wall watchlist did not load from %s: %s",
                        self.path, self.last_error)
            return 0
        return len(self.entries)

    # ── writing ─────────────────────────────────────────────────────────
    def _clean_entries(self, entries) -> tuple[list[dict], list[str]]:
        out, seen, refused = [], set(), []
        for e in entries:
            if isinstance(e, str):
                e = {"symbol": e}
            if not isinstance(e, dict):
                refused.append(f"{e!r} is not an entry.")
                continue
            sym = clean_symbol(e.get("symbol"))
            if not sym or not sym.isalnum():
                refused.append(f"{e.get('symbol')!r} is not a symbol.")
                continue
            if sym in seen:
                continue
            if len(seen) >= config.WALL_MAX_SYMBOLS:
                refused.append(f"{sym}: at the "
                               f"{config.WALL_MAX_SYMBOLS}-symbol wall cap.")
                continue
            seen.add(sym)
            out.append({"symbol": sym, "scale": clean_share(e.get("scale"))})
        return out, refused

    def set(self, entries, settings=None) -> list[str]:
        """Replace the list wholesale. Returns the refusals.

        Wholesale, matching Hub.wall_set: the page edits a list and posts the
        list, so the store and the tier cannot drift into disagreeing about
        which symbols exist.
        """
        self.entries, refused = self._clean_entries(entries)
        if settings:
            if "window_s" in settings:
                self.settings["window_s"] = clean_window(
                    settings.get("window_s"), self.settings["window_s"])
            if "spread_share" in settings:
                self.settings["spread_share"] = clean_share(
                    settings.get("spread_share"), self.settings["spread_share"])
            if "min_spread_cents" in settings:
                self.settings["min_spread_cents"] = clean_spread_floor(
                    settings.get("min_spread_cents"),
                    self.settings["min_spread_cents"])
        self.updated = time.time()
        return refused

    def save(self) -> None:
        """Write the whole object through a temporary file and a rename."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        blob = {"version": VERSION, "updated": self.updated or time.time(),
                "settings": self.settings, "entries": self.entries}
        tmp = self.path.with_suffix(".json.tmp")
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(blob, fh, indent=1)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, self.path)
        self.writes += 1

    def status(self) -> dict:
        return {"entries": len(self.entries), "writes": self.writes,
                "path": str(self.path), "loaded_from": self.loaded_from,
                "error": self.last_error, "updated": self.updated}
