"""app/metric_filter.py
Single source of truth for the eligible-metric filter applied to
daily_features columns across all dashboard surfaces and offline scripts.

Exclusion rule
--------------
  Family 2  (spot snapshots: spot_pc / spot_co)     — always excluded;
    these are identity/key fields, not analysis metrics.
  Family 4 + 5  _pc suffix only                      — excluded; the OI-by-strike
    and OI-change _pc columns capture prior-close data unavailable at morning
    analysis time (the _co variants are available and are kept).
  Family 12 _pc kept  (vol_weighted, opt_vol columns) — EVENING-tier, valid metrics.
    The old blanket endswith("_pc") filter wrongly blocked these after the
    _co → _pc rename.
  Everything else                                    — included.

Graceful fallback
-----------------
  If metric_classification is absent (e.g. fresh environment),
  get_excluded_metrics() returns None.  build_feature_cols() treats None as
  "family filter unavailable" and falls back to the old blanket
  not c.endswith("_pc") ban so callers degrade safely without extra logic.

Usage
-----
  from app.metric_filter import get_excluded_metrics, build_feature_cols

  async with pool.acquire() as conn:
      col_rows  = await conn.fetch(<schema_query>)
      excl_set  = await get_excluded_metrics(conn)

  all_cols      = [r["column_name"] for r in col_rows]
  outcomes      = {c for c in all_cols if "ret_" in c and "fwd" in c}
  feature_cols  = build_feature_cols(all_cols, outcomes, excl_set,
                                     also_exclude={primary_metric})
"""
from __future__ import annotations

__all__ = ["get_excluded_metrics", "build_feature_cols"]

_EXCLUSION_SQL = """
    SELECT metric FROM metric_classification
    WHERE  family_num = 2
        OR (family_num IN (4, 5) AND RIGHT(metric, 3) = '_pc')
"""


async def get_excluded_metrics(conn) -> set | None:
    """Return the set of metric names excluded from eligible-feature lists.

    Parameters
    ----------
    conn : asyncpg connection (already acquired — bare connection or pool conn)

    Returns
    -------
    set[str]  when metric_classification is present and the query succeeded.
              The set may be empty only if the table has no rows matching the
              rule — highly unlikely in production.
    None      when metric_classification does not exist or any DB error occurs;
              callers should treat this as "family filter unavailable" and pass
              None to build_feature_cols() which will apply a safe fallback.
    """
    try:
        rows = await conn.fetch(_EXCLUSION_SQL)
        return {r["metric"] for r in rows}
    except Exception:
        return None


def build_feature_cols(
    all_num_cols: list[str],
    outcomes,
    excl_set: "set | None",
    *,
    also_exclude=None,
) -> list[str]:
    """Return eligible feature columns in their original ordinal order.

    Parameters
    ----------
    all_num_cols  : ordered list of all numeric column names from daily_features
    outcomes      : set or list of forward-return outcome column names to skip
    excl_set      : exclusion set from get_excluded_metrics(); pass None to
                    trigger the old blanket not-endswith("_pc") fallback
    also_exclude  : optional extra names to skip (e.g. the current primary
                    metric when building the secondary candidate list)
    """
    out_set = set(outcomes)
    extra   = set(also_exclude) if also_exclude else set()
    if excl_set is None:
        # metric_classification absent — degrade to old blanket ban so the
        # endpoint stays functional even in a fresh environment.
        return [c for c in all_num_cols
                if c not in out_set and c not in extra
                and not c.endswith("_pc")]
    skip = excl_set | out_set | extra
    return [c for c in all_num_cols if c not in skip]
