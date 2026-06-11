"""app/metric_filter.py
Single source of truth for the eligible-metric filter applied to
daily_features columns across all dashboard surfaces and offline scripts.

Eligibility model — inclusion / allowlist
-----------------------------------------
  A metric appears in feature lists ONLY when it has an explicit
  eligible_as_metric = true entry in metric_classification.

  Fail-safe default: anything that is NOT explicitly true is excluded:
    • eligible_as_metric = false  → excluded  (explicitly ineligible)
    • eligible_as_metric IS NULL  → excluded  (present but unclassified)
    • absent from metric_classification entirely → excluded
                                                  (not in allowlist ⇒ hidden)

  This means a raw volume column (total_vol, total_call_vol, …) that lives
  in daily_features as a formula input but is not in metric_classification
  is automatically hidden — no explicit exclusion entry needed.

  Two exclusion reasons are encoded in metric_classification by the loader
  (scripts/load_metric_classification.py):

  (a) Deliberate _pc suppression — Family 4 + 5 _pc variants use yesterday's
      close (C_{T-1}) as the spot reference; only the _co variants are used.
      Hidden intentionally; data exists and computes correctly.

  (b) Raw drift metrics — dollar-denominated price levels and raw OI counts
      (Family 2 spot snapshots, Family 3 OI levels / strike prices, certain
      Family 4/5 _co counts and minus_spot values, Family 5 raw OI count
      differences).  Per-ticker walk-forward bins are confounded with price
      regime and secular OI growth.  Normalised siblings (_div_spot, pct_,
      zscore_) remain eligible.

Graceful fallback
-----------------
  If metric_classification is absent (e.g. fresh environment),
  get_excluded_metrics() returns None.  build_feature_cols() treats None as
  "classification unavailable" and falls back to the old blanket
  not c.endswith("_pc") ban so callers degrade safely without extra logic.

Usage
-----
  from app.metric_filter import get_excluded_metrics, build_feature_cols

  async with pool.acquire() as conn:
      col_rows  = await conn.fetch(<schema_query>)
      excl_set  = await get_excluded_metrics(conn)

  all_cols      = [r["column_name"] for r in col_rows]
  outcomes      = [c for c in all_cols if "ret_" in c and "fwd" in c]
  feature_cols  = build_feature_cols(all_cols, outcomes, excl_set,
                                     also_exclude={primary_metric})
"""
from __future__ import annotations

__all__ = ["get_excluded_metrics", "build_feature_cols"]


async def get_excluded_metrics(conn) -> "set | None":
    """Return the allowlist of metric names eligible as signal inputs.

    Despite the historic name "get_excluded_metrics", this function now
    implements an **inclusion** model: it returns the SET OF ELIGIBLE
    metrics (those with eligible_as_metric = true in metric_classification).
    Pass the result directly to build_feature_cols(), which handles the
    inclusion semantics correctly.

    Fail-safe: only metrics with an explicit eligible_as_metric = true entry
    are included.  Anything else — false, NULL, or absent from the table —
    is excluded by omission.  No additional exclusion pass is needed.

    Returns
    -------
    set[str]  allowlist of eligible metric names (may be empty if the
              table exists but has no eligible rows — unlikely in production).
    None      when metric_classification does not exist or any DB error occurs;
              callers should treat this as "classification unavailable" and
              pass None to build_feature_cols() which will apply a safe fallback.
    """
    try:
        rows = await conn.fetch(
            "SELECT metric FROM metric_classification WHERE eligible_as_metric = true"
        )
        return {r["metric"] for r in rows}
    except Exception:
        return None


def build_feature_cols(
    all_num_cols: "list[str]",
    outcomes,
    elig_set: "set | None",
    *,
    also_exclude=None,
) -> "list[str]":
    """Return eligible feature columns in their original ordinal order.

    Parameters
    ----------
    all_num_cols  : ordered list of all numeric column names from daily_features
    outcomes      : set or list of forward-return outcome column names.
                    When elig_set is provided, outcomes are excluded because
                    Family-7 metrics have eligible_as_metric = false and are
                    therefore absent from elig_set.  When elig_set is None,
                    outcomes are excluded explicitly via out_set for the
                    fallback path.
    elig_set      : allowlist from get_excluded_metrics(); pass None to trigger
                    the old blanket not-endswith("_pc") fallback.
                    When provided, ONLY metrics explicitly in this set are
                    returned.  Metrics absent from metric_classification are
                    excluded automatically — no explicit exclusion entry needed.
    also_exclude  : optional extra names to skip (e.g. the current primary
                    metric when building the secondary candidate list)
    """
    extra = set(also_exclude) if also_exclude else set()

    if elig_set is None:
        # metric_classification absent — degrade to old blanket ban so the
        # endpoint stays functional even in a fresh environment.
        out_set = set(outcomes)
        return [c for c in all_num_cols
                if c not in out_set and c not in extra
                and not c.endswith("_pc")]

    # Inclusion model: keep only metrics explicitly marked eligible=true.
    # Outcomes (Family 7, eligible=false), excluded families, drift metrics,
    # _pc variants, and any column absent from metric_classification are all
    # not in elig_set and are automatically excluded.
    return [c for c in all_num_cols if c in elig_set and c not in extra]
