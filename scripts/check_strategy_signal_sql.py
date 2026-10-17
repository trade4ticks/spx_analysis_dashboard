"""Gate: Strategy Signal's store SQL against a REAL Postgres. Needs initdb.

A temporary cluster (check_oo_market_sql's Cluster) -- nothing touches the
real database. What it pins:

  migration   a row written BEFORE entry_times / display existed (inserted
              as raw JSONB, the shape already on the VPS) lists with its
              weekdays intact, no times, the calendar display -- and the
              stored JSONB is not rewritten by reading it.
  order       move() persists: a fresh list comes back in the new order.
  image       upload, replace (a new version), thumb vs full bytes and
              types, remove, a strategy that does not exist, and deleting
              the strategy drops its image (ON DELETE CASCADE). Through the
              router's own handlers, multipart UploadFile included.

PG_BIN points at initdb/pg_ctl if they are somewhere unusual. No binaries:
SKIP, not PASS.
"""
from __future__ import annotations

import asyncio
import io
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from check_oo_market_sql import Cluster, EXIT_SKIPPED, find_bin  # noqa: E402

FAILS: list[str] = []


def check(ok: bool, msg: str) -> None:
    print(("  ok    " if ok else "  FAIL  ") + msg)
    if not ok:
        FAILS.append(msg)


def png(w, h):
    from PIL import Image
    b = io.BytesIO()
    Image.new("RGB", (w, h), (232, 67, 147)).save(b, "PNG")
    return b.getvalue()


def cfg(name, **kw):
    return {"name": name, "notes": "", "weekdays": [3], "entry_times": [], "display": "calendar",
            "states": ["TRADE", "NO TRADE"], "logic": "and", "metrics": [], "manual": [], **kw}


async def run(dsn: str) -> None:
    import asyncpg
    from fastapi import HTTPException, UploadFile
    from app.routers import strategy_signal as router
    from app.strategy_signal import images, store

    pool = await asyncpg.create_pool(dsn, min_size=1, max_size=3)
    try:
        print("migration")
        async with pool.acquire() as conn:
            # The table as the first release created it, and a row in its shape.
            await conn.execute(store.CREATE_SQL)
            legacy = {"name": "Allantis", "notes": "skew", "weekdays": [1, 2, 3, 4, 5],
                      "states": ["FULL", "MEH", "NONE"], "logic": "and", "metrics": [], "manual": ["check"]}
            await conn.execute(f"INSERT INTO {store.TABLE} (name, position, config) VALUES ($1, 1, $2)",
                               "Allantis", json.dumps(legacy))
        rows = await store.list_strategies(pool)
        a = rows[0]
        check(len(rows) == 1 and a["weekdays"] == [1, 2, 3, 4, 5] and a["entry_times"] == []
              and a["display"] == "calendar" and a["image"] is None and a["manual"] == ["check"],
              "a pre-existing row lists with its days, no times, the calendar, no image")
        async with pool.acquire() as conn:
            raw = await conn.fetchval(f"SELECT config FROM {store.TABLE} WHERE id = $1", a["id"])
            has_img_table = await conn.fetchval("SELECT to_regclass($1) IS NOT NULL", store.IMAGES)
        check(json.loads(raw) == legacy, "reading it does not rewrite the stored config")
        check(has_img_table, "the image table is created alongside, lazily")

        print("create, update, order")
        b = await store.create(pool, cfg("CRD", entry_times=[{"day": 3, "time": "10:00"}]))
        c = await store.create(pool, cfg("T-1", display="card", weekdays=[]))
        names = [r["name"] for r in await store.list_strategies(pool)]
        check(names == ["Allantis", "CRD", "T-1"], f"created in order (got {names})")
        await store.move(pool, c["id"], -1)
        await store.move(pool, c["id"], -1)
        names = [r["name"] for r in await store.list_strategies(pool)]
        check(names == ["T-1", "Allantis", "CRD"], f"move persists, a fresh list reads the new order (got {names})")
        await store.update(pool, b["id"], cfg("CRD", entry_times=[{"day": 2, "time": "10:00"},
                                                                  {"day": 5, "time": "11:30"}], weekdays=[2, 5]))
        got = next(r for r in await store.list_strategies(pool) if r["name"] == "CRD")
        check(got["entry_times"] == [{"day": 2, "time": "10:00"}, {"day": 5, "time": "11:30"}] and got["weekdays"] == [2, 5],
              "entry times round-trip")

        print("image")
        sid = a["id"]
        up = await router.put_image(sid, UploadFile(file=io.BytesIO(png(1200, 600)), filename="risk.png"), pool)
        m1 = up["image"]
        check((m1["w"], m1["h"], m1["tw"], m1["th"]) == (1200, 600, 480, 240), "upload: sizes reported")
        listed = next(r for r in await store.list_strategies(pool) if r["id"] == sid)
        check(listed["image"] == m1, "the list carries the same image meta")
        th = await router.get_image(sid, "thumb", pool)
        full = await router.get_image(sid, "full", pool)
        check(th.media_type == "image/webp" and full.media_type == "image/png", "thumb is webp, full is the upload's type")
        check(full.body == png(1200, 600) and len(th.body) < len(full.body) * 4, "full is the uploaded bytes")
        check("immutable" in th.headers.get("cache-control", ""), "cacheable, since the URL carries the version")
        await asyncio.sleep(0.01)
        up = await router.put_image(sid, UploadFile(file=io.BytesIO(png(300, 300)), filename="new.png"), pool)
        m2 = up["image"]
        check(m2["v"] > m1["v"] and (m2["w"], m2["tw"]) == (300, 300), "replace: a new version, new sizes")
        try:
            await router.put_image(sid, UploadFile(file=io.BytesIO(b"GIF89a not really"), filename="x.gif"), pool)
            check(False, "a broken file is refused with 400")
        except HTTPException as exc:
            check(exc.status_code == 400, "a broken file is refused with 400")
        check(next(r for r in await store.list_strategies(pool) if r["id"] == sid)["image"] == m2,
              "a refused upload leaves the previous image")
        try:
            await router.put_image(99999, UploadFile(file=io.BytesIO(png(10, 10)), filename="x.png"), pool)
            check(False, "no such strategy -> 404")
        except HTTPException as exc:
            check(exc.status_code == 404, "no such strategy -> 404")
        await router.delete_image(sid, pool)
        check(next(r for r in await store.list_strategies(pool) if r["id"] == sid)["image"] is None, "remove clears it")
        try:
            await router.get_image(sid, "thumb", pool)
            check(False, "a removed image is a 404")
        except HTTPException as exc:
            check(exc.status_code == 404, "a removed image is a 404")
        await store.set_image(pool, b["id"], images.prepare(png(50, 50)))
        await store.delete(pool, b["id"])
        async with pool.acquire() as conn:
            left = await conn.fetchval(f"SELECT count(*) FROM {store.IMAGES} WHERE strategy_id = $1", b["id"])
        check(left == 0, "deleting the strategy drops its image")
    finally:
        await pool.close()


def main() -> int:
    if os.name == "posix" and os.geteuid() == 0:
        print("SKIP: running as root — initdb refuses to create a cluster as root")
        return EXIT_SKIPPED
    bindir = find_bin()
    if bindir is None:
        print("SKIP: no Postgres server binaries (initdb, pg_ctl) found — set PG_BIN to run this check")
        return EXIT_SKIPPED
    try:
        import asyncpg  # noqa: F401
        import PIL  # noqa: F401
    except ImportError as exc:
        print(f"SKIP: {exc.name} not installed")
        return EXIT_SKIPPED
    with Cluster(bindir) as c:
        asyncio.run(run(c.dsn))
    print()
    if FAILS:
        print(f"FAIL: {len(FAILS)} strategy-signal SQL check(s) failed")
        return 1
    print("PASS: strategy signal SQL — legacy rows, schedule fields, order, images")
    return 0


if __name__ == "__main__":
    sys.exit(main())
