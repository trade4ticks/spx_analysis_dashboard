"""Saved portfolio profiles: a combination, not a copy of the trades.

WHAT A PROFILE IS. The strategies chosen, each with its quantity, its
capital and its filters, plus the page's date-range mode. It is a POINTER at
saved strategies and a set of numbers to apply to them -- it holds no trades
and no parse, so the strategy it names goes on being the single source of
that file. Reloading a profile after re-saving the underlying strategy picks
up the new file, which is the behaviour anyone would expect and the reason
this is not a snapshot.

WHAT THAT COSTS, said out loud rather than discovered: a profile can name a
strategy that has since been DELETED. Loading one reports which ids are gone
instead of quietly loading a smaller portfolio -- a four-strategy portfolio
that silently becomes three is a set of numbers nobody can reproduce.

VALIDATED ON THE WAY IN. The payload is a page's state, so it is structural
JSON rather than free text: ids are integers, quantities are positive
integers, capital is a non-negative number, filters are the registry's shape.
Anything else is refused with a reason. The whole point of a store is that
what comes out is what a page can use.
"""
from __future__ import annotations

import json

TABLE = "backtest_portfolio_profiles"
NAME_MAX = 200
NOTES_MAX = 4000
# A profile is a few dozen numbers and a filter map. The cap is here so a
# malformed client cannot write megabytes into a row nothing will read.
PAYLOAD_MAX = 256 * 1024
MAX_STRATEGIES = 12
CAPITAL_MAX = 1e9
QTY_MAX = 10_000

CREATE_SQL = f"""
CREATE TABLE IF NOT EXISTS {TABLE} (
    id          SERIAL PRIMARY KEY,
    name        TEXT NOT NULL UNIQUE,
    notes       TEXT NOT NULL DEFAULT '',
    payload     JSONB NOT NULL,
    n_strategies INTEGER NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now()
)
"""

_LIST_COLS = "id, name, notes, n_strategies, created_at, updated_at"


class NameTaken(Exception):
    """A profile with this name exists and replace was not requested."""

    def __init__(self, name: str, existing_id: int):
        super().__init__(f'A profile named "{name}" already exists.')
        self.name = name
        self.existing_id = existing_id


def clean_name(name: str | None) -> str:
    n = " ".join((name or "").split())
    if not n:
        raise ValueError("A name is required.")
    if len(n) > NAME_MAX:
        raise ValueError(f"Name is longer than {NAME_MAX} characters.")
    return n


def clean_notes(notes: str | None) -> str:
    n = (notes or "").strip()
    if len(n) > NOTES_MAX:
        raise ValueError(f"Notes are longer than {NOTES_MAX} characters.")
    return n


def _num(v, *, name, lo, hi, integer=False):
    try:
        f = float(v)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a number, not {v!r}.")
    if f != f or not (lo <= f <= hi):
        raise ValueError(f"{name} must be between {lo:g} and {hi:g}.")
    return int(f) if integer else f


def clean_filters(raw) -> dict:
    """The registry-shaped filter map, checked without naming a metric.

    NO METRIC NAMES HERE. The registry owns which metrics exist and this
    store has no business holding a second list -- so keys are accepted as
    given and only their SHAPE is enforced. A metric dropped upstream leaves
    a filter nothing reads, which the page back-fills on load.
    """
    if raw in (None, ""):
        return {}
    if not isinstance(raw, dict):
        raise ValueError("filters must be an object.")
    out: dict = {}
    for key, f in raw.items():
        if not isinstance(key, str) or not key or len(key) > 100:
            raise ValueError(f"{key!r} is not a metric key.")
        if not isinstance(f, dict):
            raise ValueError(f"filter {key!r} is not an object.")
        on = bool(f.get("on"))
        if "allowed" in f and f.get("allowed") is not None:
            allowed = f.get("allowed")
            if not isinstance(allowed, list) or len(allowed) > 200:
                raise ValueError(f"filter {key!r} has a bad category list.")
            for v in allowed:
                if not isinstance(v, (int, float, str, bool)):
                    raise ValueError(f"filter {key!r} has a bad category.")
            out[key] = {"on": on, "allowed": allowed}
        else:
            out[key] = {"on": on,
                        "lo": _num(f.get("lo", 0), name=f"{key} lo",
                                   lo=-1e12, hi=1e12),
                        "hi": _num(f.get("hi", 0), name=f"{key} hi",
                                   lo=-1e12, hi=1e12)}
    return out


MAX_SURFACE_PER_STRATEGY = 20


def clean_surface(raw) -> list:
    """The surface metrics ADDED to one strategy, as column names.

    NO METRIC NAMES HERE EITHER, for the same reason clean_filters holds
    none: the catalog owns which columns exist, and a second list in this
    table would drift from it. Only the SHAPE is enforced -- a string of a
    sane length that looks like a column -- and an unknown one is rejected
    by /surface/values at load time, where the catalog actually is.

    A profile stores the LIST, not the values: values are fetched per
    strategy on load, so a saved profile cannot pin a metric to whatever the
    surface table held on the day it was saved.
    """
    if raw in (None, ""):
        return []
    if not isinstance(raw, list):
        raise ValueError("surface must be a list of column names.")
    if len(raw) > MAX_SURFACE_PER_STRATEGY:
        raise ValueError(f"{len(raw)} surface metrics on one strategy; "
                         f"{MAX_SURFACE_PER_STRATEGY} is the most a profile holds.")
    out = []
    for col in raw:
        if not isinstance(col, str) or not col or len(col) > 200:
            raise ValueError(f"{col!r} is not a surface column name.")
        if not col.replace("_", "").isalnum():
            raise ValueError(f"{col!r} is not a surface column name.")
        if col not in out:
            out.append(col)
    return out


def clean_payload(raw) -> dict:
    """A page's state, checked into the shape the page reads back."""
    if not isinstance(raw, dict):
        raise ValueError("A profile needs an object to save.")
    strategies = raw.get("strategies")
    if not isinstance(strategies, list) or not strategies:
        raise ValueError("A profile needs at least one strategy.")
    if len(strategies) > MAX_STRATEGIES:
        raise ValueError(f"{len(strategies)} strategies; {MAX_STRATEGIES} is "
                         f"the most a portfolio holds.")
    seen, out = set(), []
    for s in strategies:
        if not isinstance(s, dict):
            raise ValueError("Each strategy must be an object.")
        sid = _num(s.get("id"), name="strategy id", lo=1, hi=2**31 - 1,
                   integer=True)
        if sid in seen:
            raise ValueError(f"Strategy {sid} is listed twice.")
        seen.add(sid)
        out.append({
            "id": sid,
            "qty": _num(s.get("qty", 1), name="qty", lo=1, hi=QTY_MAX,
                        integer=True),
            "capital": _num(s.get("capital", 0), name="capital", lo=0,
                            hi=CAPITAL_MAX),
            "filters": clean_filters(s.get("filters")),
            "surface": clean_surface(s.get("surface")),
        })
    mode = raw.get("range_mode", "union")
    if mode not in ("union", "intersection"):
        raise ValueError(f"range_mode {mode!r} is not union or intersection.")
    payload = {
        "strategies": out,
        "range_mode": mode,
        "roll_weeks": _num(raw.get("roll_weeks", 26), name="roll_weeks",
                           lo=2, hi=520, integer=True),
    }
    if len(json.dumps(payload)) > PAYLOAD_MAX:
        raise ValueError("The profile is too large to store.")
    return payload


async def ensure_table(conn) -> None:
    await conn.execute(CREATE_SQL)


def _row(r) -> dict:
    d = dict(r)
    for k in ("created_at", "updated_at"):
        if d.get(k) is not None:
            d[k] = d[k].isoformat()
    return d


async def list_profiles(pool) -> list[dict]:
    async with pool.acquire() as conn:
        await ensure_table(conn)
        rows = await conn.fetch(
            f"SELECT {_LIST_COLS} FROM {TABLE} ORDER BY updated_at DESC")
    return [_row(r) for r in rows]


async def save_profile(pool, *, name: str, notes: str, payload: dict,
                       replace: bool = False) -> dict:
    """Insert, or replace an existing name when asked.

    A NAME COLLISION IS AN ANSWER, not an overwrite: two portfolios called
    "live" where one silently replaced the other is a combination someone
    can no longer reproduce.
    """
    name = clean_name(name)
    notes = clean_notes(notes)
    body = clean_payload(payload)
    async with pool.acquire() as conn:
        await ensure_table(conn)
        existing = await conn.fetchrow(f"SELECT id FROM {TABLE} WHERE name = $1",
                                       name)
        if existing and not replace:
            raise NameTaken(name, existing["id"])
        if existing:
            r = await conn.fetchrow(
                f"UPDATE {TABLE} SET notes = $2, payload = $3, "
                f"n_strategies = $4, updated_at = now() WHERE id = $1 "
                f"RETURNING {_LIST_COLS}",
                existing["id"], notes, json.dumps(body), len(body["strategies"]))
        else:
            r = await conn.fetchrow(
                f"INSERT INTO {TABLE} (name, notes, payload, n_strategies) "
                f"VALUES ($1, $2, $3, $4) RETURNING {_LIST_COLS}",
                name, notes, json.dumps(body), len(body["strategies"]))
    return _row(r)


async def load_profile(pool, profile_id: int) -> dict | None:
    async with pool.acquire() as conn:
        await ensure_table(conn)
        r = await conn.fetchrow(
            f"SELECT {_LIST_COLS}, payload FROM {TABLE} WHERE id = $1",
            profile_id)
    if r is None:
        return None
    d = _row(r)
    raw = d.pop("payload")
    # asyncpg hands JSONB back as text unless a codec is registered; the
    # store parses it here so every caller sees the same thing.
    d["payload"] = json.loads(raw) if isinstance(raw, str) else raw
    return d


async def delete_profile(pool, profile_id: int) -> bool:
    async with pool.acquire() as conn:
        await ensure_table(conn)
        out = await conn.execute(f"DELETE FROM {TABLE} WHERE id = $1", profile_id)
    return out.endswith("1")
