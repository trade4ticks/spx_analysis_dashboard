"""The wall's watchlist, kept across a restart.

WHY IT IS SERVER-SIDE. The list is the page -- a few hundred names chosen over
weeks -- and local storage loses it to a cleared browser, a second machine, or
the other monitor. spx-live also restarts on every deploy, several times a
day; a watchlist that came back empty would have to be rebuilt by hand each
time, which is the same argument the scan's history file already won.

WHY A FILE AND NOT POSTGRES. It is one small object, written when a person
edits a list and read once at startup, by one process, with no query beyond
"give me the list". A table, a pool and a migration for that is machinery
bought for nothing. The scan's history made the same call for 6.6 MB a day.

WHAT AN ENTRY CARRIES. The ticker, plus the per-pane scale override from
when the wall drew tapes -- carried, unused, so it is not lost (see
clean_scale). The settings are the two spread thresholds, floor and wide.

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


def clean_scale(v):
    """An entry's old per-pane scale override, carried and never used.

    The panes are gone (2026-11-06) and nothing reads it; it is kept on the
    entry rather than dropped so the first save after the change does not
    throw away settings someone chose by hand. Not range-checked: nothing
    draws with it.
    """
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if f != f else f


def _cents(v):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if f != f else f


def clean_floor(v, default=None):
    """The floor in cents, clamped to WALL_FLOOR_MIN..MAX.

    0 OR LESS READS AS THE DEFAULT, not as the minimum. Before the list, 0
    meant "no floor" and was the saved default; the floor can no longer be
    off, and the value that answers that file is the new default (59c), not
    the slider's bottom (10c). Clamped rather than refused: a stored list
    that will not load because of one number costs the whole watchlist.
    """
    default = config.WALL_FLOOR_CENTS if default is None else default
    f = _cents(v)
    if f is None or f <= 0:
        return default
    return min(config.WALL_FLOOR_MAX, max(config.WALL_FLOOR_MIN, f))


def clean_wide(v, default=None):
    """The wide threshold in cents: WALL_WIDE_MIN..MAX on WALL_WIDE_STEP."""
    default = config.WALL_WIDE_CENTS if default is None else default
    f = _cents(v)
    if f is None or f <= 0:
        return default
    f = round(f / config.WALL_WIDE_STEP) * config.WALL_WIDE_STEP
    return min(config.WALL_WIDE_MAX, max(config.WALL_WIDE_MIN, f))


class WallStore:
    """The watchlist and the page's settings. One instance, loaded at start."""

    def __init__(self, path: str | Path | None = None):
        self.path = Path(path or config.WALL_STORE_PATH)
        self.entries: list[dict] = []
        # PRESENTATION ONLY, both: nothing downstream of either reaches the
        # hub. Saved with the list so they survive a refresh and a restart.
        self.settings: dict = {
            "min_spread_cents": config.WALL_FLOOR_CENTS,
            "wide_cents": config.WALL_WIDE_CENTS,
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
                "min_spread_cents": clean_floor(st.get("min_spread_cents")),
                "wide_cents": clean_wide(st.get("wide_cents")),
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
            out.append({"symbol": sym, "scale": clean_scale(e.get("scale"))})
        return out, refused

    def set(self, entries, settings=None) -> list[str]:
        """Replace the list wholesale. Returns the refusals.

        Wholesale, matching Hub.wall_set: the page edits a list and posts the
        list, so the store and the tier cannot drift into disagreeing about
        which symbols exist.
        """
        self.entries, refused = self._clean_entries(entries)
        if settings:
            if "min_spread_cents" in settings:
                self.settings["min_spread_cents"] = clean_floor(
                    settings.get("min_spread_cents"),
                    self.settings["min_spread_cents"])
            if "wide_cents" in settings:
                self.settings["wide_cents"] = clean_wide(
                    settings.get("wide_cents"), self.settings["wide_cents"])
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
