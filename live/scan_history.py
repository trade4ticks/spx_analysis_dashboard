"""The grid's two hours of minutes, kept across a restart.

WHY A FILE. spx-live restarts on every deploy, several times a day, and each
restart used to blank the whole grid. Two hours of history is the page's entire
context -- a row you are about to trade is worth looking at BECAUSE of what the
last ninety minutes did -- so losing it to a push costs more than the machinery
to keep it.

WHY NOT POSTGRES. A full session is 430 symbols x 960 minutes x 4 float32 =
6.6 MB, written once a minute and read once at startup, by one process, with no
query beyond "give me today". A table, a pool, a migration and a schema for
that is machinery bought for no benefit. A file is the right size of answer.

A SIDE EFFECT WORTH HAVING: the files are per session date and are not deleted,
so yesterday's grid is still on disk and can be loaded by date.

--- What a cell is ---------------------------------------------------------

Four numbers per symbol per minute: the quiet ratio, the p10-p90 range in
cents, dollars per minute, and the trade count. The first colours the cell, the
second and third are what the page's two bars show, and the fourth is what
decides whether the minute is gated -- a dead name is quiet by default and must
not light up the grid.

float32, not float64. These are drawn, not computed with; the arithmetic
happens upstream in float64 and lands here to be looked at. It halves both the
file and the memory for no visible difference.

--- Minutes are absolute, not relative -------------------------------------

A cell is indexed by its MINUTE OF THE SESSION DAY, counted from
SESSION_START_ET, rather than by its position in a ring. A ring index means the
reloaded file has to be rotated back into place against the current time, which
is the kind of arithmetic that is wrong by one for an hour and then right
again. An absolute index reloads by being read.

It also makes the gap after a restart explicit: minutes the service was down
are NaN, and NaN renders as an empty cell rather than as a quiet one. A restart
must not look like a calm market.
"""
from __future__ import annotations

import asyncio
import logging
import os
import time
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

log = logging.getLogger("live.scan_history")

ET = ZoneInfo("America/New_York")

# 04:00 to 20:00 ET. Wider than the regular session on purpose: the tape runs
# pre- and post-market, and a grid that starts at 09:30 would drop the hour
# before the open, which is when the day's character is decided.
SESSION_START_ET = 4 * 60
SESSION_MINUTES = 16 * 60

# ratio, range_cents, dollars_per_min, trades
CELL_WIDTH = 4


def session_date(now: float | None = None) -> str:
    """The ET calendar date whose session `now` belongs to.

    Taken in ET rather than UTC, so a file boundary lands at ET midnight and
    not in the middle of a session for anyone running the box on UTC.
    """
    dt = datetime.fromtimestamp(time.time() if now is None else now, ET)
    return dt.strftime("%Y-%m-%d")


def minute_index(now: float | None = None) -> int | None:
    """Minutes since SESSION_START_ET, or None outside the window."""
    dt = datetime.fromtimestamp(time.time() if now is None else now, ET)
    idx = dt.hour * 60 + dt.minute - SESSION_START_ET
    return idx if 0 <= idx < SESSION_MINUTES else None


def minute_epoch(date: str, idx: int) -> float:
    """The unix time at the START of a session minute. For the page's axis."""
    y, m, d = (int(x) for x in date.split("-"))
    base = datetime(y, m, d, tzinfo=ET) + timedelta(
        minutes=SESSION_START_ET + idx)
    return base.timestamp()


class ScanHistory:
    """Per-symbol minute cells for one session date, flushed to a file.

    Symbols come and go as the scan set changes, so storage is per symbol
    rather than one rectangular array: a name added at noon should not cost a
    morning of NaN, and a name dropped should keep what it had in case it comes
    back.
    """

    def __init__(self, directory: str | Path, date: str | None = None):
        self.dir = Path(directory)
        self.date = date or session_date()
        self.cells: dict[str, np.ndarray] = {}
        self.loaded_from: str | None = None
        self.flushes = 0
        self.last_flush: float | None = None
        self.last_error: str | None = None
        self._writing = False

    # ── storage ─────────────────────────────────────────────────────────
    def path_for(self, date: str | None = None) -> Path:
        return self.dir / f"scan_cells_{date or self.date}.npz"

    def _row(self, symbol: str) -> np.ndarray:
        row = self.cells.get(symbol)
        if row is None:
            row = np.full((SESSION_MINUTES, CELL_WIDTH), np.nan,
                          dtype="float32")
            self.cells[symbol] = row
        return row

    def write(self, symbol: str, idx: int, cell) -> None:
        """One finished minute: exactly CELL_WIDTH numbers, no more.

        SPREAD IS NOT STORED HERE. It is a live per-symbol screen and a
        column, not something the grid draws per minute -- the cells are
        coloured by quietness and the expanded row has three bands, none of
        them spread. Widening every stored minute by a third to carry a value
        only ever read at `now` would cost a third of the session file and
        invalidate every file already on disk, for nothing anyone looks at.

        The caller therefore passes the trade rollup's leading fields and
        leaves the quote fields behind. If that ever needs to change, the
        shape check in load() is what will catch the old files.

        Out-of-range indices are ignored rather than raised: a rollup landing
        a second either side of the session window is ordinary at 04:00 and
        20:00, and raising there would take the whole minute task down for a
        cell nobody will look at.
        """
        if idx is None or not (0 <= idx < SESSION_MINUTES):
            return
        self._row(symbol)[idx] = cell

    def slice(self, symbols, first: int, count: int) -> dict:
        """`count` minutes ending at `first + count`, per symbol.

        Returned as lists of nullable numbers rather than NaN: NaN is not
        valid JSON, and every encoder that accepts it emits something no
        parser on the other end agrees about.
        """
        first = max(0, first)
        last = min(SESSION_MINUTES, first + count)
        out = {}
        for sym in symbols:
            row = self.cells.get(sym)
            if row is None:
                out[sym] = None
                continue
            seg = row[first:last]
            out[sym] = [
                None if not np.isfinite(v[0]) and not np.isfinite(v[3])
                else [_num(v[0], 3), _num(v[1], 1), _num(v[2], 0),
                      _num(v[3], 0)]
                for v in seg]
        return out

    # ── the file ────────────────────────────────────────────────────────
    def load(self, date: str | None = None) -> int:
        """Read a session's cells back. Returns the symbol count.

        A missing file is NOT an error: the first run of a session date has
        nothing to load, and that is the common case rather than a fault.
        """
        p = self.path_for(date)
        if not p.is_file():
            return 0
        try:
            with np.load(p, allow_pickle=False) as z:
                syms = [str(s) for s in z["symbols"]]
                arr = z["cells"]
        except (OSError, ValueError, KeyError) as exc:
            # A CORRUPT FILE MUST NOT STOP THE SERVICE. It is two hours of
            # drawing, not state anything depends on; losing it is a bad
            # morning, refusing to start is a dead tape.
            self.last_error = f"{type(exc).__name__}: {exc}"
            log.warning("scan history at %s is unreadable (%s); starting empty",
                        p, self.last_error)
            return 0
        if arr.ndim != 3 or arr.shape[1] != SESSION_MINUTES \
                or arr.shape[2] != CELL_WIDTH or len(syms) != arr.shape[0]:
            self.last_error = (f"shape {arr.shape} for {len(syms)} symbols does "
                               f"not match {SESSION_MINUTES}x{CELL_WIDTH}")
            log.warning("scan history at %s has the wrong shape (%s); "
                        "starting empty", p, self.last_error)
            return 0
        self.cells = {s: arr[i].astype("float32")
                      for i, s in enumerate(syms)}
        self.loaded_from = str(p)
        log.info("scan history: %d symbols restored from %s", len(syms), p)
        return len(syms)

    def _blob(self) -> tuple:
        """A COPY of the current cells, for a writer running off-thread.

        Copied here, on the loop, rather than in the writer: the arrays keep
        being written while the file is being compressed, and handing the live
        ones to a thread is a torn file and a race in one move.
        """
        syms = sorted(self.cells)
        if not syms:
            return (), np.zeros((0, SESSION_MINUTES, CELL_WIDTH), "float32")
        return syms, np.stack([self.cells[s] for s in syms])

    async def flush(self) -> bool:
        """Write the session file. Returns whether anything was written.

        OFF THE EVENT LOOP. Compressing ~6.6 MB is tens of milliseconds, and
        this project just spent a change removing a 367 ms stall from this same
        loop -- reintroducing one on a timer would undo it for a file nobody is
        waiting on.

        ATOMIC. Written to a temporary name and renamed, so a crash mid-write
        leaves the previous session file intact rather than a truncated one
        that loads as garbage.

        Re-entrancy is refused rather than queued: if a flush is still running
        when the next minute arrives, the next one has nothing new to say that
        the one after will not say better.
        """
        if self._writing:
            return False
        syms, arr = self._blob()
        if not syms:
            return False
        self._writing = True
        try:
            await asyncio.to_thread(self._write_blob, syms, arr)
            self.flushes += 1
            self.last_flush = time.time()
            self.last_error = None
            return True
        except OSError as exc:
            # Reported, not raised. The grid keeps working from memory; what
            # is lost is only the ability to survive a restart, and the page
            # is told so rather than discovering it at the next deploy.
            self.last_error = f"{type(exc).__name__}: {exc}"
            log.warning("scan history flush failed: %s", self.last_error)
            return False
        finally:
            self._writing = False

    def _write_blob(self, syms, arr) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        final = self.path_for()
        tmp = final.with_suffix(".npz.tmp")
        # WRITTEN THROUGH AN OPEN HANDLE, not by passing the path.
        #
        # np.savez_compressed APPENDS ".npz" to a filename that does not
        # already end in it. Given "scan_cells_2026-09-08.npz.tmp" it writes
        # "scan_cells_2026-09-08.npz.tmp.npz", and the os.replace below then
        # looks for a file that was never created -- so every flush raised
        # FileNotFoundError, was caught and logged as a failed flush, and the
        # history silently never persisted at all. A file object is left
        # alone, which is the only way to name the temporary file after what
        # it is rather than after what numpy will accept.
        with open(tmp, "wb") as fh:
            np.savez_compressed(fh, symbols=np.array(syms), cells=arr)
        os.replace(tmp, final)

    def status(self) -> dict:
        arrays = len(self.cells)
        return {
            "date": self.date,
            "symbols": arrays,
            "minutes": SESSION_MINUTES,
            "start_et_minute": SESSION_START_ET,
            "memory_mb": arrays * SESSION_MINUTES * CELL_WIDTH * 4 / 1048576.0,
            "path": str(self.path_for()),
            "loaded_from": self.loaded_from,
            "flushes": self.flushes,
            "last_flush_age_s": (time.time() - self.last_flush)
                                if self.last_flush else None,
            "error": self.last_error,
        }

    def sessions(self, limit: int = 30) -> list[str]:
        """Session dates with a file on disk, newest first.

        The files are not deleted, so yesterday's grid is loadable. This is
        what lets the page offer it.
        """
        try:
            names = sorted((p.stem[len("scan_cells_"):]
                            for p in self.dir.glob("scan_cells_*.npz")),
                           reverse=True)
        except OSError:
            return []
        return names[:limit]


def _num(v, places):
    """A finite float rounded for the wire, or None.

    NaN IS NOT JSON. Python's encoder emits bare `NaN`, which is not in the
    grammar; browsers reject it and the whole frame is lost rather than one
    cell. None is `null`, which every parser agrees means "no value" -- and
    "no value" is exactly what a minute with no trades is.
    """
    if v is None or not np.isfinite(v):
        return None
    r = round(float(v), places)
    return int(r) if places == 0 else r
