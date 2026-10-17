"""Saved Strategy Signal configurations (main database).

ONE ROW PER STRATEGY, THE CONFIGURATION AS JSONB -- the pattern of
backtest_portfolio_profiles. A strategy is a handful of fields and a short
list of metrics; columns per field would be a migration every time the
page grows a setting, and nothing queries inside the config.

VALIDATED ON THE WAY IN, into exactly the shape the page and the evaluator
read back. Sources are checked against the metric library's live catalog,
so a strategy can only name a metric that exists; one that later disappears
from the table is reported by the board as an error on that metric, not
dropped from the saved config.

THE SCHEDULE (2026-10-16). `weekdays` -- once "eligible entry days", a
condition -- is now the SCHEDULED entry days, information only; the stored
key is unchanged so every saved strategy keeps its days with no migration.
`entry_times` adds optional times, [{day, time}], several per day allowed;
a scheduled day with no time is still scheduled ("time not set"). `display`
puts the strategy on the weekly calendar or on a standalone card. Rows saved
before these existed read back with the defaults in `_row` (the calendar, no
times) -- nothing is rewritten, nothing is invented.

THE RISK GRAPH is an uploaded image in its own table, keyed by strategy and
dropped with it, as BYTEA -- the convention of oo_backtest_strategies'
uploaded file. Never in the JSONB: the config is read on every board call,
the image only when a page shows it. A thumbnail is made once at upload
(images.prepare) so the calendar never pulls the full file.

The tables are created lazily and idempotently, like the other stores.
"""
from __future__ import annotations

import json
import math

from app.strategy_signal import evaluate as ev, library as lib

TABLE = "strategy_signal_strategies"
NAME_MAX = 80
NOTES_MAX = 8000
STATE_MAX = 24
MANUAL_MAX = 160
LABEL_MAX = 60
MAX_STATES = 6
MAX_METRICS = 8
MAX_MANUAL = 8
MAX_STRATEGIES = 30
MAX_ENTRY_TIMES = 20
DISPLAYS = ("calendar", "card")
DEFAULT_DISPLAY = "calendar"
# Entry times the calendar can place: the regular session, 09:30-16:00 ET.
FIRST_TIME, LAST_TIME = "09:30", "16:00"
IMAGES = "strategy_signal_images"

CREATE_SQL = f"""
CREATE TABLE IF NOT EXISTS {TABLE} (
    id          SERIAL PRIMARY KEY,
    name        TEXT NOT NULL UNIQUE,
    position    INTEGER NOT NULL DEFAULT 0,
    config      JSONB NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now()
)
"""

CREATE_IMAGES_SQL = f"""
CREATE TABLE IF NOT EXISTS {IMAGES} (
    strategy_id   INTEGER PRIMARY KEY REFERENCES {TABLE}(id) ON DELETE CASCADE,
    content_type  TEXT NOT NULL,
    image         BYTEA NOT NULL,
    thumb         BYTEA NOT NULL,
    thumb_type    TEXT NOT NULL,
    width         INTEGER NOT NULL,
    height        INTEGER NOT NULL,
    thumb_width   INTEGER NOT NULL,
    thumb_height  INTEGER NOT NULL,
    updated_at    TIMESTAMPTZ NOT NULL DEFAULT now()
)
"""


def _text(v, *, name, cap, required=False) -> str:
    s = " ".join(str(v or "").split())
    if required and not s:
        raise ValueError(f"{name} is required.")
    if len(s) > cap:
        raise ValueError(f"{name} is longer than {cap} characters.")
    return s


def _number(v, *, name) -> float:
    try:
        f = float(v)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a number, not {v!r}.")
    if not math.isfinite(f):
        raise ValueError(f"{name} must be a finite number.")
    return f


def clean_time(v, *, where: str) -> str:
    """'9:5' / '09:05' -> '09:05'; inside the session or refused."""
    parts = str(v or "").strip().split(":")
    if len(parts) != 2 or not all(p.strip().isdigit() for p in parts):
        raise ValueError(f"{where}: {v!r} is not a time (HH:MM).")
    h, m = int(parts[0]), int(parts[1])
    if not (0 <= h <= 23 and 0 <= m <= 59):
        raise ValueError(f"{where}: {v!r} is not a time (HH:MM).")
    t = f"{h:02d}:{m:02d}"
    if not FIRST_TIME <= t <= LAST_TIME:
        raise ValueError(f"{where}: {t} is outside the session ({FIRST_TIME}-{LAST_TIME} ET).")
    return t


def clean_entry_times(raw) -> list[dict]:
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise ValueError("Entry times must be a list.")
    out = set()
    for i, e in enumerate(raw):
        if not isinstance(e, dict) or e.get("day") not in ev.WEEKDAYS:
            raise ValueError(f"Entry time {i + 1}: needs a weekday 1 (Mon) .. 5 (Fri).")
        out.add((e["day"], clean_time(e.get("time"), where=f"Entry time {i + 1}")))
    if len(out) > MAX_ENTRY_TIMES:
        raise ValueError(f"At most {MAX_ENTRY_TIMES} scheduled entry times.")
    return [{"day": d, "time": t} for d, t in sorted(out)]


def clean_metric(raw, *, n_states: int, sources: set[str], seen: set[str], idx: int) -> dict:
    if not isinstance(raw, dict):
        raise ValueError("Each metric must be an object.")
    where = f"Metric {idx + 1}"
    mid = _text(raw.get("id") or f"m{idx + 1}", name=f"{where} id", cap=20, required=True)
    if mid in seen:
        mid = f"m{idx + 1}"
        while mid in seen:
            mid += "x"
    seen.add(mid)
    a = raw.get("a")
    if a not in sources:
        raise ValueError(f"{where}: {a or '(nothing)'} is not an available metric.")
    op = raw.get("op") or None
    b = None
    if op is not None:
        if op not in lib.OPERATORS:
            raise ValueError(f"{where}: {op!r} is not one of + - * /.")
        b = raw.get("b")
        if b not in sources:
            raise ValueError(f"{where}: the second metric, {b or '(nothing)'}, is not an available metric.")
    transform = raw.get("transform") or None
    if transform is not None and transform not in lib.TRANSFORMS:
        raise ValueError(f"{where}: unknown transform {transform!r}.")
    chart, signal = bool(raw.get("chart")), bool(raw.get("signal"))
    if not (chart or signal):
        raise ValueError(f"{where} shows no chart and is not used for the signal -- tick one or remove it.")
    resolution = raw.get("resolution") or "daily"
    if resolution not in lib.RESOLUTIONS:
        raise ValueError(f"{where}: resolution must be intraday or daily.")
    lookback = raw.get("lookback") or ("10d" if resolution == "intraday" else "1y")
    if lookback not in lib.LOOKBACKS:
        raise ValueError(f"{where}: unknown lookback {lookback!r}.")
    if resolution == "intraday" and lookback not in lib.INTRADAY_LOOKBACKS:
        raise ValueError(f"{where}: an intraday chart goes back at most 3 months.")
    m = {"id": mid, "label": _text(raw.get("label"), name=f"{where} label", cap=LABEL_MAX),
         "a": a, "op": op, "b": b, "transform": transform,
         "chart": chart, "resolution": resolution, "lookback": lookback,
         "signal": signal, "cmp": None, "thresholds": []}
    if signal:
        cmp = raw.get("cmp")
        if cmp not in ev.CMPS:
            raise ValueError(f"{where}: comparison must be one of {' '.join(ev.CMPS)}.")
        th = raw.get("thresholds")
        if not isinstance(th, list) or len(th) != n_states - 1:
            raise ValueError(f"{where}: needs {n_states - 1} threshold(s), one per state but the last.")
        m["cmp"] = cmp
        m["thresholds"] = [_number(t, name=f"{where} threshold") for t in th]
    return m


def clean_config(raw, sources: set[str]) -> dict:
    if not isinstance(raw, dict):
        raise ValueError("A strategy must be an object.")
    name = _text(raw.get("name"), name="Name", cap=NAME_MAX, required=True)
    notes = (raw.get("notes") or "").strip()
    if len(notes) > NOTES_MAX:
        raise ValueError(f"Notes are longer than {NOTES_MAX} characters.")

    states = raw.get("states")
    if not isinstance(states, list):
        raise ValueError("Output states must be a list.")
    states = [_text(s, name="A state", cap=STATE_MAX) for s in states]
    states = [s for s in states if s]
    if not 2 <= len(states) <= MAX_STATES:
        raise ValueError(f"Between 2 and {MAX_STATES} output states, please.")
    if len({s.lower() for s in states}) != len(states):
        raise ValueError("Output states must be different from each other.")

    # Scheduled days are information, so none at all is allowed (a strategy
    # watched for weeks at a time has no day). A day with an entry time is
    # scheduled whether or not it was ticked.
    days = raw.get("weekdays") or []
    if not isinstance(days, list) or any(d not in ev.WEEKDAYS for d in days):
        raise ValueError("Scheduled entry days must be a list of weekdays 1 (Mon) .. 5 (Fri).")
    entry_times = clean_entry_times(raw.get("entry_times"))
    days = sorted(set(days) | {e["day"] for e in entry_times})
    display = raw.get("display") or DEFAULT_DISPLAY
    if display not in DISPLAYS:
        raise ValueError("Summary display is the weekly calendar or a standalone card.")

    logic = raw.get("logic", "and")
    if logic not in ev.LOGICS:
        raise ValueError("Conditions combine with AND or OR.")

    metrics = raw.get("metrics") or []
    if not isinstance(metrics, list) or len(metrics) > MAX_METRICS:
        raise ValueError(f"At most {MAX_METRICS} metrics per strategy.")
    seen: set[str] = set()
    metrics = [clean_metric(m, n_states=len(states), sources=sources, seen=seen, idx=i)
               for i, m in enumerate(metrics)]

    manual = raw.get("manual") or []
    if not isinstance(manual, list):
        raise ValueError("Manual requirements must be a list.")
    manual = [_text(s, name="A manual requirement", cap=MANUAL_MAX) for s in manual]
    manual = [s for s in manual if s]
    if len(manual) > MAX_MANUAL:
        raise ValueError(f"At most {MAX_MANUAL} manual requirements.")

    return {"name": name, "notes": notes, "weekdays": days, "entry_times": entry_times,
            "display": display, "states": states,
            "logic": logic, "metrics": metrics, "manual": manual}


async def ensure_table(conn) -> None:
    await conn.execute(CREATE_SQL)
    await conn.execute(CREATE_IMAGES_SQL)


def _image_meta(r) -> dict | None:
    if r is None or "image_at" not in r.keys() or r["image_at"] is None:
        return None
    return {"v": int(r["image_at"].timestamp() * 1000), "w": r["image_w"], "h": r["image_h"],
            "tw": r["image_tw"], "th": r["image_th"]}


def with_defaults(cfg: dict) -> dict:
    """A saved config with every field the page reads. Defaults for rows
    saved before a field existed -- read time only; the stored config is
    never rewritten to add them."""
    return {"weekdays": [], "entry_times": [], "display": DEFAULT_DISPLAY, **cfg}


def _row(r) -> dict:
    cfg = r["config"]
    cfg = json.loads(cfg) if isinstance(cfg, str) else cfg
    return {"id": r["id"], "position": r["position"], **with_defaults(cfg), "name": r["name"],
            "image": _image_meta(r),
            "updated_at": r["updated_at"].isoformat() if r["updated_at"] else None}


LIST_SQL = f"""
SELECT s.*, i.updated_at AS image_at, i.width AS image_w, i.height AS image_h,
       i.thumb_width AS image_tw, i.thumb_height AS image_th
FROM {TABLE} s LEFT JOIN {IMAGES} i ON i.strategy_id = s.id
ORDER BY s.position, s.id"""


async def list_strategies(pool) -> list[dict]:
    async with pool.acquire() as conn:
        await ensure_table(conn)
        rows = await conn.fetch(LIST_SQL)
    return [_row(r) for r in rows]


async def create(pool, cfg: dict) -> dict:
    async with pool.acquire() as conn:
        await ensure_table(conn)
        if await conn.fetchval(f"SELECT count(*) FROM {TABLE}") >= MAX_STRATEGIES:
            raise ValueError(f"{MAX_STRATEGIES} strategies is the most this page holds.")
        if await conn.fetchval(f"SELECT 1 FROM {TABLE} WHERE name = $1", cfg["name"]):
            raise ValueError(f'A strategy named "{cfg["name"]}" already exists.')
        pos = await conn.fetchval(f"SELECT coalesce(max(position), 0) + 1 FROM {TABLE}")
        r = await conn.fetchrow(
            f"INSERT INTO {TABLE} (name, position, config) VALUES ($1, $2, $3) RETURNING *",
            cfg["name"], pos, json.dumps(cfg))
    return _row(r)


async def update(pool, sid: int, cfg: dict) -> dict | None:
    async with pool.acquire() as conn:
        await ensure_table(conn)
        clash = await conn.fetchval(f"SELECT id FROM {TABLE} WHERE name = $1 AND id <> $2", cfg["name"], sid)
        if clash:
            raise ValueError(f'A strategy named "{cfg["name"]}" already exists.')
        r = await conn.fetchrow(
            f"UPDATE {TABLE} SET name = $2, config = $3, updated_at = now() WHERE id = $1 RETURNING *",
            sid, cfg["name"], json.dumps(cfg))
    return _row(r) if r else None


async def move(pool, sid: int, direction: int) -> bool:
    """Swap a strategy with its neighbour in the page order."""
    async with pool.acquire() as conn:
        await ensure_table(conn)
        async with conn.transaction():
            rows = await conn.fetch(f"SELECT id FROM {TABLE} ORDER BY position, id")
            ids = [r["id"] for r in rows]
            if sid not in ids:
                return False
            i = ids.index(sid)
            j = i + (1 if direction > 0 else -1)
            if 0 <= j < len(ids):
                ids[i], ids[j] = ids[j], ids[i]
            for pos, x in enumerate(ids, start=1):
                await conn.execute(f"UPDATE {TABLE} SET position = $2 WHERE id = $1", x, pos)
    return True


async def delete(pool, sid: int) -> bool:
    async with pool.acquire() as conn:
        await ensure_table(conn)
        out = await conn.execute(f"DELETE FROM {TABLE} WHERE id = $1", sid)
    return out.endswith("1")


# ── the risk graph ──────────────────────────────────────────────────────────

async def set_image(pool, sid: int, img: dict) -> dict | None:
    """Store or replace a strategy's image (images.prepare's output). None
    when there is no such strategy."""
    async with pool.acquire() as conn:
        await ensure_table(conn)
        if not await conn.fetchval(f"SELECT 1 FROM {TABLE} WHERE id = $1", sid):
            return None
        r = await conn.fetchrow(
            f"""INSERT INTO {IMAGES} (strategy_id, content_type, image, thumb, thumb_type,
                                      width, height, thumb_width, thumb_height, updated_at)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, clock_timestamp())
                ON CONFLICT (strategy_id) DO UPDATE SET
                  content_type = EXCLUDED.content_type, image = EXCLUDED.image,
                  thumb = EXCLUDED.thumb, thumb_type = EXCLUDED.thumb_type,
                  width = EXCLUDED.width, height = EXCLUDED.height,
                  thumb_width = EXCLUDED.thumb_width, thumb_height = EXCLUDED.thumb_height,
                  updated_at = clock_timestamp()
                RETURNING updated_at AS image_at, width AS image_w, height AS image_h,
                          thumb_width AS image_tw, thumb_height AS image_th""",
            sid, img["content_type"], img["data"], img["thumb"], img["thumb_type"],
            img["width"], img["height"], img["thumb_width"], img["thumb_height"])
    return _image_meta(r)


async def get_image(pool, sid: int, size: str) -> tuple[str, bytes] | None:
    """(content type, bytes) of the thumbnail or the uploaded file."""
    col, typ = ("thumb", "thumb_type") if size == "thumb" else ("image", "content_type")
    async with pool.acquire() as conn:
        await ensure_table(conn)
        r = await conn.fetchrow(f"SELECT {typ} AS t, {col} AS b FROM {IMAGES} WHERE strategy_id = $1", sid)
    return (r["t"], bytes(r["b"])) if r else None


async def delete_image(pool, sid: int) -> bool:
    async with pool.acquire() as conn:
        await ensure_table(conn)
        out = await conn.execute(f"DELETE FROM {IMAGES} WHERE strategy_id = $1", sid)
    return out.endswith("1")
