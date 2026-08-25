"""Structure presets for the Equity IV page.

Mounted under /api/equity-iv alongside equity_iv.py and equity_iv_surface.py.

WHERE USER PRESETS LIVE, and why it is not localStorage
--------------------------------------------------------
This app already keeps two kinds of user state, and they are not
interchangeable:

  localStorage   transient per-browser UI defaults — backtest_iv_analysis's
                 saveDefaults, the Signal Survey outcome. Per machine, lost on
                 a new browser, invisible to the server.

  Postgres       named user-created artefacts other parts of the app read —
                 oi_research_portfolios, signals, corner_scan_notes,
                 tracked_signals.

A structure preset is the second kind, and one requirement settles it: a later
step builds a structured brief from a preset's metric list, server-side. A
preset living in a browser cannot be read by that. So this follows the
oi_research_portfolios idiom — a table, a _DDL string, _ensure_tables on
demand — rather than inventing anything.

Built-ins come from app/equity_presets.py and are NOT stored in the table.
Keeping them in code means adding a BWB preset is a config entry, and means a
built-in cannot be edited into something that no longer matches what the code
around it assumes. They can be duplicated; the copy is an ordinary user row.
"""
from __future__ import annotations

import json

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from app.db import get_oi_pool
from app.equity_presets import (
    BUILTIN_STRUCTURES, COLUMN_ALIASES, PAIR_FAMILIES, PAIR_FOR_TENOR,
    resolve_preset,
)
from app.routers.equity_iv import _catalog

router = APIRouter(tags=["equity-iv"])

_DDL = """
CREATE TABLE IF NOT EXISTS equity_structure_presets (
    id         SERIAL PRIMARY KEY,
    name       TEXT NOT NULL UNIQUE,
    tenor      INTEGER NOT NULL,
    -- The whole preset body: rails, scanner columns, filters, sort, axes.
    -- JSONB rather than a column per field because the shape is the preset's
    -- own business and will grow -- a BWB preset may want legs a put ratio
    -- has no use for -- and because the brief reads it whole.
    payload    JSONB NOT NULL,
    note       TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);
"""


async def _ensure_tables(pool):
    async with pool.acquire() as conn:
        await conn.execute(_DDL)


class StructureIn(BaseModel):
    name: str
    tenor: int
    rails: list = []
    scanner_columns: list = []
    scanner_filters: list = []
    scanner_sort: dict | None = None
    scatter_x: dict | None = None
    scatter_y: dict | None = None
    note: str | None = None


def _row_to_preset(r) -> dict:
    body = r["payload"]
    if isinstance(body, str):
        body = json.loads(body)
    return {
        "key": f"user:{r['id']}", "id": r["id"], "name": r["name"],
        "note": r["note"], "tenor": r["tenor"], "builtin": False, **body,
    }


@router.get("/structures")
async def list_structures(
    tenor: int = Query(None, description="OVERRIDE: resolve every preset at "
                                        "this tenor instead of its own"),
    pool=Depends(get_oi_pool),
):
    """Built-ins and saved presets, every column resolved against the catalog.

    Resolution happens HERE rather than being left to the client because the
    column names in a preset are literals, and literals are the one thing on
    this page with no lookup behind them — which is precisely how two
    cross-section preset buttons ended up setting one axis and silently
    keeping the other after an upstream rename.

    Anything that does not resolve comes back named in `unresolved` rather
    than dropped. A preset quietly one column short is a preset that looks
    like it worked.

    `tenor` OVERRIDES every preset's own, and is for a caller that wants one
    preset read at a horizon it was not written for -- the brief-builder
    asking "what would Put Ratio look like at 30d". The page must NOT pass it
    when listing: doing so stamps the page's current tenor onto every preset
    and erases the one each was saved with, which is exactly what made a copy
    saved at 7d come back as whatever the page happened to be showing. A
    preset's tenor is part of the preset.
    """
    if not pool:
        return {"error": "OI database not configured", "structures": []}

    cat = await _catalog(pool)
    by_col = cat["by_col"]

    await _ensure_tables(pool)
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT id, name, tenor, payload, note FROM equity_structure_presets "
            "ORDER BY lower(name)"
        )

    out = []
    for p in BUILTIN_STRUCTURES:
        out.append(resolve_preset(by_col, dict(p, builtin=True), tenor))
    for r in rows:
        out.append(resolve_preset(by_col, _row_to_preset(r), tenor))

    return {
        "structures": out,
        # The client's retarget() needs the same two tables this module used.
        # Shipped from here so there is ONE definition of each -- the algorithm
        # is written per runtime, the data is not.
        "aliases": COLUMN_ALIASES,
        "pair_families": sorted(PAIR_FAMILIES),
        "pair_for_tenor": {str(k): list(v) for k, v in PAIR_FOR_TENOR.items()},
    }


@router.post("/structures")
async def save_structure(body: StructureIn, pool=Depends(get_oi_pool)):
    """Save the current configuration under a name, or update a saved one.

    A built-in's name is refused rather than shadowed: two entries with one
    name in the picker is a worse outcome than being told to choose another,
    and silently overriding a built-in would make the code and the screen
    disagree about what "Put Ratio" means.
    """
    if not pool:
        raise HTTPException(503, "OI database not configured")

    name = (body.name or "").strip()
    if not name:
        raise HTTPException(400, "A preset needs a name")
    if any(name.lower() == p["name"].lower() for p in BUILTIN_STRUCTURES):
        raise HTTPException(
            400, f"{name!r} is a built-in preset. Built-ins cannot be replaced — "
                 f"duplicate it under another name instead.")

    payload = {
        "rails": body.rails, "scanner_columns": body.scanner_columns,
        "scanner_filters": body.scanner_filters,
        "scanner_sort": body.scanner_sort,
        "scatter_x": body.scatter_x, "scatter_y": body.scatter_y,
    }

    await _ensure_tables(pool)
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "INSERT INTO equity_structure_presets (name, tenor, payload, note) "
            "VALUES ($1, $2, $3::jsonb, $4) "
            "ON CONFLICT (name) DO UPDATE SET "
            "  tenor = EXCLUDED.tenor, payload = EXCLUDED.payload, "
            "  note = EXCLUDED.note, updated_at = NOW() "
            "RETURNING id, name, tenor, payload, note",
            name, int(body.tenor), json.dumps(payload), body.note,
        )
    return {"saved": _row_to_preset(row)}


@router.delete("/structures/{preset_id}")
async def delete_structure(preset_id: int, pool=Depends(get_oi_pool)):
    """Delete a saved preset. Built-ins are not in the table, so not reachable."""
    if not pool:
        raise HTTPException(503, "OI database not configured")
    await _ensure_tables(pool)
    async with pool.acquire() as conn:
        gone = await conn.fetchval(
            "DELETE FROM equity_structure_presets WHERE id = $1 RETURNING id",
            preset_id,
        )
    if gone is None:
        raise HTTPException(404, f"No saved preset with id {preset_id}")
    return {"deleted": gone}
