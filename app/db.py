import logging
import os
import asyncpg
from dotenv import load_dotenv
from urllib.parse import urlsplit, urlunsplit

load_dotenv()

log = logging.getLogger(__name__)

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


def _safe_dsn(dsn: str) -> str:
    """A DSN with the password removed, for a log line.

    Logging the failure is the whole point of the handler below, and a raw DSN
    would put the database password in the journal — so the redaction happens
    here rather than being remembered at each call site.
    """
    try:
        parts = urlsplit(dsn)
        if parts.password:
            host = parts.hostname or ""
            if parts.port:
                host = f"{host}:{parts.port}"
            netloc = f"{parts.username or ''}:***@{host}"
            return urlunsplit(parts._replace(netloc=netloc))
        return dsn
    except Exception:
        return "<unparseable DSN>"


# ── what init_pool() actually did ────────────────────────────────────────────
#
# Recorded per pool so verify_pools() can tell the three cases apart, because
# they are indistinguishable from the outside and have produced an hour of
# elimination between them:
#
#   not configured   no DSN in the environment. Legitimate; the pages that
#                    read it report themselves unavailable.
#   configured, failed   a DSN was present and connecting raised. Also
#                    legitimate for the optional pools — the scalp database
#                    does not exist until the pipeline's first run.
#   configured, neither  a DSN was present, nothing raised, and there is still
#                    no pool. That is not a state the environment can produce.
#                    It means the code failed to assign, which is exactly the
#                    bug this file shipped: init_pool() declared
#                    `global _pool, _oi_pool` and assigned _scalp_pool, so the
#                    assignment created a local and the module-level name
#                    stayed None. Every symptom was identical to a missing
#                    database.
_STATUS: dict[str, dict] = {}


def _record(name: str, dsn: str | None, attr: str, error: str | None,
            required: bool) -> None:
    """Record what happened, reading the pool from the MODULE namespace.

    `attr` is the module-level variable's NAME, not its value, and that is the
    whole mechanism. Passing the value would defeat the check completely: with
    a missing `global`, the assignment binds a local, and a caller writing
    `_record(..., _scalp_pool, ...)` hands over that same local — non-None,
    freshly created, connected. The status would read healthy while the
    module-level name the dependency actually returns stayed None, and
    verify_pools() would confirm a working pool that no request can reach.

    Looking the name up in globals() is invisible to a local binding, so the
    contradiction the guard is built to detect — configured, no exception, no
    pool — is exactly what a missing `global` produces.
    """
    pool = globals().get(attr)
    _STATUS[name] = {"configured": bool(dsn), "dsn": _safe_dsn(dsn) if dsn else None,
                     "connected": pool is not None, "error": error,
                     "required": required, "attr": attr}


async def init_pool() -> None:
    global _pool, _oi_pool, _scalp_pool

    dsn = os.environ["DATABASE_URL"]
    _pool = await asyncpg.create_pool(
        dsn=dsn,
        min_size=4,
        max_size=20,
        command_timeout=30,
    )
    _record("main", dsn, "_pool", None, required=True)

    oi_dsn = os.getenv("OI_DATABASE_URL")
    if oi_dsn:
        _oi_pool = await asyncpg.create_pool(
            dsn=oi_dsn,
            min_size=2,
            max_size=10,
            command_timeout=30,
        )
    _record("open_interest", oi_dsn, "_oi_pool", None, required=False)

    # The scalp database may not exist yet -- the pipeline creates it on first
    # run. A failure here must not take the whole app down with it, since every
    # other page is unaffected by its absence; the page reports "not
    # connected" instead, which is a state it has to handle anyway.
    #
    # But it must SAY SO. A swallowed exception here made a missing database,
    # wrong credentials and a plain bug produce byte-identical output, and the
    # only way to tell them apart was to eliminate them one at a time.
    scalp_dsn = _scalp_dsn()
    scalp_err = None
    if scalp_dsn:
        try:
            _scalp_pool = await asyncpg.create_pool(
                dsn=scalp_dsn,
                min_size=1,
                max_size=5,
                command_timeout=30,
            )
        except Exception as exc:
            _scalp_pool = None
            scalp_err = f"{type(exc).__name__}: {exc}"
            log.warning(
                "scalp pool not created for %s — %s. The Equities Scalp page "
                "will report itself not connected; every other page is "
                "unaffected.", _safe_dsn(scalp_dsn), scalp_err,
            )
    _record("equities_scalp", scalp_dsn, "_scalp_pool", scalp_err, required=False)


def verify_pools() -> None:
    """Fail startup on a pool that was configured, did not fail, and is absent.

    THE CLASS OF BUG THIS CATCHES is a path that never runs during development
    and produces a plausible-looking result in production. A missing `global`
    is the purest example: the code reads correctly, the pool really is
    created, the connection really does succeed, and the module-level name is
    still None because the assignment bound a local. Every downstream symptom
    is identical to the database being absent.

    A reference checker cannot see this — the name exists, it is spelled right,
    and it is assigned. What distinguishes the bug is a CONTRADICTION between
    two observations that are individually unremarkable: a DSN was configured,
    nothing raised, and there is no pool. The environment cannot produce that
    combination; only the code can.

    So the assertion is not "every pool must exist". That would break the case
    the optional pools are designed for — the scalp database legitimately does
    not exist before the pipeline's first run. It is "every outcome must be
    explicable", which is a weaker claim that this bug still cannot satisfy.
    """
    if not _STATUS:
        raise RuntimeError(
            "verify_pools() ran before init_pool(). Nothing has been "
            "attempted, so there is nothing to verify and a clean result "
            "would be meaningless.")

    problems = []
    for name, st in _STATUS.items():
        if st["required"] and not st["connected"]:
            problems.append(f"  {name}: required, and not connected.")
            continue
        if st["configured"] and not st["connected"] and not st["error"]:
            problems.append(
                f"  {name}: a DSN is configured ({st['dsn']}), connecting did "
                f"not raise, and there is no pool.\n"
                f"    The environment cannot produce this. Check that "
                f"init_pool() declares this pool in its `global` statement — "
                f"without it the assignment binds a local and the module-level "
                f"name stays None.")

    if problems:
        raise RuntimeError(
            "Database pools did not initialise as configured:\n"
            + "\n".join(problems))

    for name, st in _STATUS.items():
        if not st["configured"]:
            log.info("pool %s: not configured", name)
        elif st["connected"]:
            log.info("pool %s: connected to %s", name, st["dsn"])
        else:
            log.warning("pool %s: configured but unavailable — %s",
                        name, st["error"])


def pool_status() -> dict:
    """What each pool did at startup. Read by the health endpoint."""
    return {k: dict(v) for k, v in _STATUS.items()}


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
    _STATUS.clear()


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
