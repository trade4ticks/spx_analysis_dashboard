"""Prove verify_pools() catches a missing `global`, by reproducing one.

WHY THIS SCRIPT EXISTS. The bug it guards against is the fourth of its kind on
this dashboard: a code path that never runs during development and produces a
plausible result in production. `init_pool()` declared

    global _pool, _oi_pool

and then assigned `_scalp_pool`. The assignment created a local, the
module-level name stayed None, and the page reported itself not connected --
identical to a missing database and to bad credentials, which is what made it
cost an hour of elimination rather than a glance.

The reference checker cannot see this. The name exists, it is spelled
correctly, and it is assigned. Nothing about the source is wrong in isolation.

So the check is behavioural, and it has to be run against a DELIBERATELY
BROKEN init_pool -- otherwise this script is itself an unexecuted path,
asserting that a guard fires without ever making it fire. Both directions are
exercised: the correct wiring must pass, and the broken wiring must fail.

There is a second, subtler assertion here. verify_pools() must NOT fail when a
configured database is genuinely unreachable, because that is the case the
optional pools are designed for -- the scalp database does not exist until the
pipeline's first run. A guard that fired on that would be turned off within a
week, and then would not be there for the real thing.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app import db                                       # noqa: E402


def reset():
    db._STATUS.clear()
    db._pool = db._oi_pool = db._scalp_pool = None


class FakePool:
    pass


def case_all_connected() -> str | None:
    """Everything configured and up. Must pass."""
    reset()
    db._pool = FakePool()
    db._record("main", "postgresql://u:p@h/db", "_pool", None, required=True)
    db._oi_pool = FakePool()
    db._record("open_interest", "postgresql://u:p@h/oi", "_oi_pool", None, False)
    db._scalp_pool = FakePool()
    db._record("equities_scalp", "postgresql://u:p@h/s", "_scalp_pool", None, False)
    try:
        db.verify_pools()
    except RuntimeError as exc:
        return f"a fully-connected app was rejected: {exc}"
    return None


def case_not_configured() -> str | None:
    """An optional pool with no DSN at all. Must pass — that is normal."""
    reset()
    db._pool = FakePool()
    db._record("main", "postgresql://u:p@h/db", "_pool", None, required=True)
    db._record("equities_scalp", None, "_scalp_pool", None, required=False)
    try:
        db.verify_pools()
    except RuntimeError as exc:
        return f"an unconfigured optional pool was rejected: {exc}"
    return None


def case_configured_but_unreachable() -> str | None:
    """Configured, connect raised, no pool. Must PASS.

    This is the case the scalp pool exists to tolerate: the database is not
    created until the pipeline's first run. A guard that failed here would be
    disabled inside a week and would not be present for the real bug.
    """
    reset()
    db._pool = FakePool()
    db._record("main", "postgresql://u:p@h/db", "_pool", None, required=True)
    db._record("equities_scalp", "postgresql://u:p@h/s", "_scalp_pool",
               "InvalidCatalogNameError: database does not exist", False)
    try:
        db.verify_pools()
    except RuntimeError as exc:
        return (f"a configured-but-absent optional database was treated as a "
                f"code fault: {exc}")
    return None


def case_missing_global() -> str | None:
    """THE BUG. Configured, nothing raised, no pool. Must FAIL."""
    reset()
    db._pool = FakePool()
    db._record("main", "postgresql://u:p@h/db", "_pool", None, required=True)
    # Exactly what init_pool() records when the assignment bound a local:
    # a DSN was present, no exception was caught, and the module-level name is
    # still None.
    db._record("equities_scalp", "postgresql://u:p@h/s", "_scalp_pool", None, False)
    try:
        db.verify_pools()
    except RuntimeError as exc:
        msg = str(exc)
        if "global" not in msg:
            return ("the missing-global case failed, but the message does not "
                    "name the cause — the point of the guard is that it says "
                    "what to look at")
        return None
    return ("a configured pool that neither connected nor failed was accepted. "
            "That is the exact signature of the shipped bug.")


def case_required_down() -> str | None:
    """The main pool absent. Must FAIL — nothing works without it."""
    reset()
    db._record("main", "postgresql://u:p@h/db", "_pool", None, required=True)
    try:
        db.verify_pools()
    except RuntimeError:
        return None
    return "the app started with no main pool"


def case_before_init() -> str | None:
    """Verifying before init_pool() must fail rather than report success.

    A clean result from an empty status table is the most misleading answer
    this function could give: it would confirm the wiring of pools that had
    never been attempted.
    """
    reset()
    try:
        db.verify_pools()
    except RuntimeError:
        return None
    return "verify_pools() reported success before init_pool() ran"


def case_password_redacted() -> str | None:
    """The failure message must not carry the database password.

    verify_pools() prints the DSN so the reader knows which database it means,
    and this runs at startup where the output lands in a log.
    """
    reset()
    db._pool = FakePool()
    db._record("main", "postgresql://u:p@h/db", "_pool", None, required=True)
    db._record("equities_scalp",
               "postgresql://portfolio:hunter2@localhost/equities_scalp",
               "_scalp_pool", None, False)
    try:
        db.verify_pools()
    except RuntimeError as exc:
        if "hunter2" in str(exc):
            return "the failure message leaks the database password"
        if "***" not in str(exc):
            return "the DSN was not redacted in the recognisable way"
        return None
    return "the missing-global case did not fail, so redaction went unchecked"


# ── against the REAL init_pool() ─────────────────────────────────────────
#
# Everything above tests verify_pools() against a hand-built status table,
# which proves the guard reasons correctly and proves nothing about whether
# init_pool() feeds it honestly. The shipped bug lived in init_pool.
#
# So these run the real function with asyncpg stubbed out, and then check the
# thing that actually broke: that the MODULE-LEVEL names are bound, which is
# what the FastAPI dependencies return. A status table saying "connected"
# while get_scalp_pool() returns None is the precise failure, and only reading
# both can see it.


class _FakeAsyncpg:
    """create_pool that succeeds without a database."""
    def __init__(self, fail_for=()):
        self.fail_for = fail_for
        self.Pool = FakePool

    async def create_pool(self, dsn=None, **kw):
        for frag in self.fail_for:
            if frag in (dsn or ""):
                raise RuntimeError("database \"x\" does not exist")
        return FakePool()


def _run_init(env, fail_for=()):
    """init_pool() under a stubbed asyncpg and a controlled environment."""
    import asyncio
    reset()
    real_asyncpg, real_env = db.asyncpg, dict(os.environ)
    db.asyncpg = _FakeAsyncpg(fail_for)
    try:
        for k in ("DATABASE_URL", "OI_DATABASE_URL", "SCALP_DATABASE_URL"):
            os.environ.pop(k, None)
        os.environ.update(env)
        asyncio.run(db.init_pool())
        db.verify_pools()
        return None
    except RuntimeError as exc:
        return str(exc)
    finally:
        db.asyncpg = real_asyncpg
        os.environ.clear()
        os.environ.update(real_env)


def case_init_binds_module_names() -> str | None:
    """The real init_pool must bind every module-level name it creates."""
    err = _run_init({
        "DATABASE_URL":       "postgresql://u:p@h/main",
        "OI_DATABASE_URL":    "postgresql://u:p@h/oi",
        "SCALP_DATABASE_URL": "postgresql://u:p@h/scalp",
    })
    if err:
        return f"a fully-configured init_pool() was rejected: {err}"
    # The dependencies read the module namespace, so that is what is checked
    # -- not the status table, which a local binding could have populated.
    for attr in ("_pool", "_oi_pool", "_scalp_pool"):
        if getattr(db, attr) is None:
            return (f"init_pool() left db.{attr} as None after creating a pool "
                    f"for it. The assignment bound a local — add {attr} to the "
                    f"`global` statement.")
    return None


def case_init_missing_global_is_caught() -> str | None:
    """THE REGRESSION. Patch out the global and confirm startup refuses.

    Without this, the guard is itself an unexecuted path: it would assert that
    a fault is detected without ever producing one. The source is rewritten to
    the exact broken form that shipped — `global _pool, _oi_pool` — compiled,
    and run.
    """
    import types, asyncio
    src = (ROOT / "app" / "db.py").read_text(encoding="utf-8")
    broken = src.replace("global _pool, _oi_pool, _scalp_pool",
                         "global _pool, _oi_pool", 1)
    if broken == src:
        return ("could not find the `global` statement to break — this check "
                "has stopped testing anything")

    mod = types.ModuleType("db_broken")
    mod.__file__ = "db_broken.py"
    exec(compile(broken, "db_broken.py", "exec"), mod.__dict__)
    mod.asyncpg = _FakeAsyncpg()

    real_env = dict(os.environ)
    try:
        for k in ("OI_DATABASE_URL", "SCALP_DATABASE_URL"):
            os.environ.pop(k, None)
        os.environ["DATABASE_URL"] = "postgresql://u:p@h/main"
        asyncio.run(mod.init_pool())
    finally:
        os.environ.clear()
        os.environ.update(real_env)

    # First: the bug must actually be present, or the rest proves nothing.
    if mod._scalp_pool is not None:
        return ("the deliberately-broken module still bound _scalp_pool, so "
                "the reproduction failed and the guard went untested")
    try:
        mod.verify_pools()
    except RuntimeError as exc:
        if "global" not in str(exc):
            return "the guard fired but did not name the cause"
        return None
    return ("init_pool() with `global _pool, _oi_pool` — the exact form that "
            "shipped — passed verify_pools(). The guard does not catch the "
            "bug it was written for.")


CASES = [
    ("all connected",                 case_all_connected),
    ("optional pool not configured",  case_not_configured),
    ("configured but unreachable",    case_configured_but_unreachable),
    ("MISSING GLOBAL",                case_missing_global),
    ("required pool down",            case_required_down),
    ("verify before init",            case_before_init),
    ("password redacted",             case_password_redacted),
    ("real init binds module names",  case_init_binds_module_names),
    ("MISSING GLOBAL, end to end",    case_init_missing_global_is_caught),
]


def main() -> int:
    bad = 0
    for name, fn in CASES:
        problem = fn()
        if problem:
            print(f"  FAIL {name}: {problem}")
            bad += 1
    reset()
    print(f"\npool wiring cases: {len(CASES)}, failures: {bad}")
    return 1 if bad else 0


sys.exit(main())
