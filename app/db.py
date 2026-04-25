import os
import asyncpg
from dotenv import load_dotenv

load_dotenv()

_pool: asyncpg.Pool | None = None
_oi_pool: asyncpg.Pool | None = None


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


async def close_pool() -> None:
    global _pool, _oi_pool
    if _pool:
        await _pool.close()
        _pool = None
    if _oi_pool:
        await _oi_pool.close()
        _oi_pool = None


async def get_pool() -> asyncpg.Pool:
    """FastAPI dependency — returns the shared connection pool."""
    if _pool is None:
        raise RuntimeError("Database pool not initialised")
    return _pool


async def get_oi_pool() -> asyncpg.Pool | None:
    """Returns the open_interest pool, or None if not configured."""
    return _oi_pool
