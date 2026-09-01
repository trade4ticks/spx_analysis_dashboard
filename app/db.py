import os
import asyncpg
from dotenv import load_dotenv
from urllib.parse import urlsplit, urlunsplit

load_dotenv()

_pool: asyncpg.Pool | None = None
_oi_pool: asyncpg.Pool | None = None
_scalp_pool: asyncpg.Pool | None = None

# The equities-scalp pipeline keeps its own DATABASE on the same server as the
# dashboards, deliberately: its table names (universe, daily_metrics, rankings)
# are generic enough to collide, and a separate database means the whole
# project can be dropped without touching the IV or factor work.
#
# "Same server, different database" is exactly what a derived DSN expresses, so
# the default is DATABASE_URL with the database name swapped. SCALP_DATABASE_URL
# overrides it for a host that keeps the two apart.
SCALP_DB_NAME = os.getenv("SCALP_PG_DB", "equities_scalp")


def _scalp_dsn() -> str | None:
    explicit = os.getenv("SCALP_DATABASE_URL")
    if explicit:
        return explicit
    base = os.getenv("DATABASE_URL")
    if not base:
        return None
    parts = urlsplit(base)
    return urlunsplit(parts._replace(path="/" + SCALP_DB_NAME))


async def init_pool() -> None:
    global _pool, _oi_pool
    _pool = await asyncpg.create_pool(
        dsn=os.environ["DATABASE_URL"],
        min_size=4,
        max_size=20,
        command_timeout=30,
    )
    oi_dsn = os.getenv("OI_DATABASE_URL")
    if oi_dsn:
        _oi_pool = await asyncpg.create_pool(
            dsn=oi_dsn,
            min_size=2,
            max_size=10,
            command_timeout=30,
        )
    # The scalp database may not exist yet -- the pipeline creates it on first
    # run. A failure here must not take the whole app down with it, since every
    # other page is unaffected by its absence; the page reports "not
    # connected" instead, which is a state it has to handle anyway.
    scalp_dsn = _scalp_dsn()
    if scalp_dsn:
        try:
            _scalp_pool = await asyncpg.create_pool(
                dsn=scalp_dsn,
                min_size=1,
                max_size=5,
                command_timeout=30,
            )
        except Exception:
            _scalp_pool = None


async def close_pool() -> None:
    global _pool, _oi_pool, _scalp_pool
    if _pool:
        await _pool.close()
        _pool = None
    if _oi_pool:
        await _oi_pool.close()
        _oi_pool = None
    if _scalp_pool:
        await _scalp_pool.close()
        _scalp_pool = None


async def get_pool() -> asyncpg.Pool:
    """FastAPI dependency — returns the shared connection pool."""
    if _pool is None:
        raise RuntimeError("Database pool not initialised")
    return _pool


async def get_oi_pool() -> asyncpg.Pool | None:
    """Returns the open_interest pool, or None if not configured."""
    return _oi_pool


async def get_scalp_pool() -> asyncpg.Pool | None:
    """Returns the equities_scalp pool, or None if absent or unreachable."""
    return _scalp_pool
