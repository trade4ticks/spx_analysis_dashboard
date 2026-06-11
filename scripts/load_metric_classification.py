#!/usr/bin/env python3
"""Load metric classification from the data dictionary into the DB.

Reads `daily_features_data_dictionary.md` (or a path you specify) and writes
every metric's family, tier, and eligibility into the `metric_classification`
table in the OI database.  Re-running refreshes the table; the dictionary file
is the source of truth.

Usage:
    python scripts/load_metric_classification.py \\
        --dict-path daily_features_data_dictionary.md \\
        [--dry-run]   # print rows, do not touch DB
        [--verify]    # after write, print spot-check results
"""
from __future__ import annotations

import argparse
import asyncio
import os
import re
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(_ROOT / ".env")

import asyncpg  # noqa: E402

# ── DDL ───────────────────────────────────────────────────────────────────────

_DDL = """
CREATE TABLE IF NOT EXISTS metric_classification (
    metric             TEXT PRIMARY KEY,
    family_num         INTEGER NOT NULL,
    family_name        TEXT NOT NULL,
    tier               TEXT NOT NULL,
    eligible_as_metric BOOLEAN NOT NULL,
    updated_at         TIMESTAMPTZ DEFAULT NOW()
);
"""

# ── Exclusions ────────────────────────────────────────────────────────────────

# Family numbers whose metrics are NEVER eligible as primary/secondary.
_INELIGIBLE_FAMILIES: set[int] = {
    1,  # Identity (ticker, trade_date) — key fields, not metrics
    2,  # Spot snapshots (spot_pc, spot_co) — raw dollar price levels; confounded
        # with price regime and secular drift over the 7-year history.
    7,  # Forward returns — prediction TARGETS, not features.
        # Critical: they are EVENING-tier so the tier check alone would not
        # exclude them; the family check is the authoritative guard.
}

# Individual metrics excluded from eligibility regardless of family.
# Two exclusion reasons are combined here:
#
#  (a) Deliberate _pc suppression (Family 4 + 5 _pc variants):
#      These divide OI-weighted strikes by YESTERDAY'S close (C_{T-1}).  The
#      _co variants use today's open and are the correct reference for morning
#      analysis.  The _pc variants are intentionally hidden; they are NOT a
#      data-quality issue — the data exists and the columns compute correctly.
#
#  (b) Raw drift metrics (Family 3 levels, Family 4 _co raw counts,
#      Family 5 raw count differences):
#      Dollar-denominated price levels and raw contract counts that scale with
#      the underlying's price and with secular OI growth.  Their per-ticker
#      walk-forward bins are confounded with time and price regime.  The
#      normalised siblings (_div_spot, pct_, zscore_) are kept eligible.
#
# Columns in this set remain in daily_features — ineligible = hidden from
# signal selection, NOT deleted.  "Storage is cheap, dropping is irreversible."
_INELIGIBLE_METRICS: frozenset[str] = frozenset({

    # ── (b) Family 3 — raw OI levels & dollar-denominated strike prices ────
    "total_oi", "call_oi", "put_oi",
    "max_oi_strike_call", "max_oi_strike_put",
    "oi_weighted_call",     "oi_weighted_put",     "oi_weighted_all",
    "oi_weighted_all_0_30d",  "oi_weighted_call_0_30d",  "oi_weighted_put_0_30d",
    "oi_weighted_all_31_90d", "oi_weighted_call_31_90d", "oi_weighted_put_31_90d",

    # ── (a) Family 4 — ALL _pc variants (prior-close spot reference) ───────
    "oi_weighted_call_minus_spot_pc",
    "oi_weighted_put_minus_spot_pc",
    "oi_weighted_all_minus_spot_pc",
    "oi_weighted_call_div_spot_pc",
    "oi_weighted_put_div_spot_pc",
    "oi_weighted_all_div_spot_pc",
    "oi_weighted_all_0_30d_div_spot_pc",
    "oi_weighted_call_0_30d_div_spot_pc",
    "oi_weighted_put_0_30d_div_spot_pc",
    "oi_weighted_all_31_90d_div_spot_pc",
    "oi_weighted_call_31_90d_div_spot_pc",
    "oi_weighted_put_31_90d_div_spot_pc",
    "oi_weighted_next_monthly_div_spot_pc",
    "oi_within_5pct_pc",   "oi_within_10pct_pc",
    "oi_above_spot_pc",    "oi_below_spot_pc",
    "oi_above_below_ratio_pc",
    "pct_oi_within_5pct_pc",  "pct_oi_within_10pct_pc",
    "pct_oi_above_spot_pc",   "pct_oi_below_spot_pc",

    # ── (b) Family 4 — _co raw counts & dollar-level minus_spot ───────────
    # (_co div_spot, ratio, pct_ siblings remain eligible)
    "oi_weighted_call_minus_spot_co",
    "oi_weighted_put_minus_spot_co",
    "oi_weighted_all_minus_spot_co",
    "oi_within_5pct_co",  "oi_within_10pct_co",
    "oi_above_spot_co",   "oi_below_spot_co",

    # ── (a) Family 5 — ALL _pc variants (prior-close spot reference) ───────
    "d1_oi_weighted_all_div_spot_change_pc",
    "d5_oi_weighted_all_div_spot_change_pc",
    "zscore_oi_weighted_all_div_spot_3m_pc",
    "zscore_oi_above_below_ratio_3m_pc",

    # ── (b) Family 5 — raw OI count differences (not pct_change) ──────────
    # (d_total_oi_pct_change, zscore_, ratio_ siblings remain eligible)
    "d1_total_oi_change", "d5_total_oi_change", "d20_total_oi_change",
})

# ── Parser ────────────────────────────────────────────────────────────────────

_TIER_PAT = r"(MORNING|EVENING|both \(key\))"

# Standard single-column row:  | `col_name` | ... | TIER |
_RE_SINGLE = re.compile(
    r"^\|\s*`([a-zA-Z_]\w*)`\s*\|.+\|\s*" + _TIER_PAT + r"\s*\|\s*$"
)

# Paired _pc/_co columns (Families 4 and 5):
#   | `col_name_pc` / `_co` | ... | TIER |
# Expands to TWO entries: col_name_pc and col_name_co
_RE_PAIRED = re.compile(
    r"^\|\s*`([a-zA-Z_]\w*_pc)`\s*/\s*`_co`\s*\|.+\|\s*" + _TIER_PAT + r"\s*\|\s*$"
)

# Family section header:  ## Family N — Name
_RE_FAMILY = re.compile(r"^##\s+Family\s+(\d+)\s+[—–\-]\s+(.+)", re.UNICODE)


def parse_dictionary(path: Path) -> list[dict]:
    """Parse the data dictionary and return a list of metric rows.

    Each row: {metric, family_num, family_name, tier, eligible_as_metric}
    """
    text = path.read_text(encoding="utf-8")
    metrics: list[dict] = []
    current_family_num: int | None = None
    current_family_name: str = ""

    for line in text.splitlines():
        # Detect family section header
        m = _RE_FAMILY.match(line)
        if m:
            current_family_num = int(m.group(1))
            current_family_name = m.group(2).strip()
            continue

        if current_family_num is None:
            continue  # haven't seen a family header yet

        # Detect paired _pc / _co columns
        m = _RE_PAIRED.match(line)
        if m:
            base_pc = m.group(1)      # e.g. col_name_pc
            tier = m.group(2)
            base_co = base_pc[:-2] + "co"  # replace trailing _pc with _co
            for col_name in (base_pc, base_co):
                metrics.append(_make_row(col_name, current_family_num,
                                         current_family_name, tier))
            continue

        # Detect single column
        m = _RE_SINGLE.match(line)
        if m:
            col_name = m.group(1)
            tier = m.group(2)
            metrics.append(_make_row(col_name, current_family_num,
                                      current_family_name, tier))

    return metrics


def _make_row(metric: str, family_num: int, family_name: str, tier: str) -> dict:
    eligible = True
    if family_num in _INELIGIBLE_FAMILIES:
        eligible = False
    elif metric in _INELIGIBLE_METRICS:
        eligible = False
    return {
        "metric": metric,
        "family_num": family_num,
        "family_name": family_name,
        "tier": tier,
        "eligible_as_metric": eligible,
    }

# ── DB helpers ────────────────────────────────────────────────────────────────

_UPSERT = """
INSERT INTO metric_classification
    (metric, family_num, family_name, tier, eligible_as_metric, updated_at)
VALUES ($1, $2, $3, $4, $5, NOW())
ON CONFLICT (metric) DO UPDATE
    SET family_num         = EXCLUDED.family_num,
        family_name        = EXCLUDED.family_name,
        tier               = EXCLUDED.tier,
        eligible_as_metric = EXCLUDED.eligible_as_metric,
        updated_at         = NOW();
"""


async def write_to_db(conn: asyncpg.Connection, rows: list[dict]) -> None:
    await conn.execute(_DDL)
    data = [
        (r["metric"], r["family_num"], r["family_name"],
         r["tier"], r["eligible_as_metric"])
        for r in rows
    ]
    await conn.executemany(_UPSERT, data)


async def verify(conn: asyncpg.Connection) -> None:
    """Print spot-checks against the freshly-written table."""
    spot_checks = [
        # ── Eligible metrics that must remain eligible ─────────────────────
        ("put_call_oi_ratio",             "MORNING", True,  "OI ratio — keep eligible"),
        ("rv_20d",                        "EVENING", True,  "vol metric — keep eligible"),
        ("iv_25d_call_30d",               "EVENING", True,  "30d skew metric (now active)"),
        ("atm_iv_30d",                    "EVENING", True,  "IV metric — keep eligible"),
        ("oi_weighted_all_div_spot_co",   "MORNING", True,  "_co div_spot — keep eligible"),
        ("pct_oi_within_5pct_co",         "MORNING", True,  "_co pct_ — keep eligible"),
        ("oi_above_below_ratio_co",       "MORNING", True,  "_co ratio — keep eligible"),
        ("d1_total_oi_pct_change",        "MORNING", True,  "pct_change — keep eligible"),
        # ── Family 7 forward returns must stay ineligible ──────────────────
        ("ret_5d_fwd_oc",                 "EVENING", False, "Family 7 forward return"),
        # ── Family 2 spot (now via _INELIGIBLE_FAMILIES) ──────────────────
        ("spot_pc",                       "MORNING", False, "Family 2 spot — dollar level"),
        # ── (b) New drift exclusions: raw OI levels ────────────────────────
        ("total_oi",                      "MORNING", False, "F3 raw OI level — drift"),
        ("oi_weighted_all",               "MORNING", False, "F3 OI-weighted strike — drift"),
        ("oi_within_5pct_co",             "MORNING", False, "F4 _co raw count — drift"),
        ("d1_total_oi_change",            "MORNING", False, "F5 raw OI count diff — drift"),
        # ── (a) _pc suppression: folded into eligibility, must stay hidden ──
        ("oi_weighted_all_div_spot_pc",   "MORNING", False, "F4 _pc — prior-close suppressed"),
        ("oi_above_below_ratio_pc",       "MORNING", False, "F4 _pc ratio — prior-close suppressed"),
        ("zscore_oi_above_below_ratio_3m_pc", "MORNING", False, "F5 _pc zscore — prior-close suppressed"),
    ]

    print("\n── Spot checks ────────────────────────────────────────────────")
    all_pass = True
    for metric, exp_tier, exp_elig, label in spot_checks:
        row = await conn.fetchrow(
            "SELECT tier, eligible_as_metric FROM metric_classification WHERE metric=$1",
            metric,
        )
        if row is None:
            print(f"  MISSING  {metric:<40} ({label})")
            all_pass = False
            continue
        ok = row["tier"] == exp_tier and row["eligible_as_metric"] == exp_elig
        mark = "✓" if ok else "✗ FAIL"
        print(
            f"  {mark}  {metric:<40} tier={row['tier']:<12} "
            f"eligible={str(row['eligible_as_metric']):<5}  ({label})"
        )
        if not ok:
            all_pass = False

    # Family 7 check: ALL 12 forward returns must be eligible=false
    fam7_wrong = await conn.fetchval(
        """SELECT COUNT(*) FROM metric_classification
           WHERE family_num=7 AND eligible_as_metric=true"""
    )
    fam7_total = await conn.fetchval(
        "SELECT COUNT(*) FROM metric_classification WHERE family_num=7"
    )
    fam7_ok = fam7_wrong == 0
    print(
        f"\n  {'✓' if fam7_ok else '✗ FAIL'}  Family 7: {fam7_total} forward returns "
        f"present, {fam7_wrong} erroneously eligible (must be 0)"
    )
    if not fam7_ok:
        all_pass = False

    # Family 2 check: spot_pc and spot_co must be ineligible
    fam2_wrong = await conn.fetchval(
        """SELECT COUNT(*) FROM metric_classification
           WHERE family_num=2 AND eligible_as_metric=true"""
    )
    fam2_ok = fam2_wrong == 0
    print(
        f"  {'✓' if fam2_ok else '✗ FAIL'}  Family 2: spot snapshots "
        f"(spot_pc/co) — {fam2_wrong} erroneously eligible (must be 0)"
    )
    if not fam2_ok:
        all_pass = False

    # _pc suppression check: no Family 4/5 _pc metric may be eligible
    pc_wrong = await conn.fetchval(
        """SELECT COUNT(*) FROM metric_classification
           WHERE family_num IN (4,5)
             AND RIGHT(metric,3) = '_pc'
             AND eligible_as_metric=true"""
    )
    pc_ok = pc_wrong == 0
    print(
        f"  {'✓' if pc_ok else '✗ FAIL'}  Family 4/5 _pc suppression: "
        f"{pc_wrong} _pc metrics erroneously eligible (must be 0)"
    )
    if not pc_ok:
        all_pass = False

    # Summary counts
    totals = await conn.fetchrow(
        """SELECT COUNT(*) AS total,
                  COUNT(*) FILTER (WHERE eligible_as_metric) AS n_elig,
                  COUNT(*) FILTER (WHERE NOT eligible_as_metric) AS n_inelig,
                  COUNT(*) FILTER (WHERE tier='MORNING') AS n_morning,
                  COUNT(*) FILTER (WHERE tier='EVENING') AS n_evening
           FROM metric_classification"""
    )
    print(
        f"\n  Total rows:      {totals['total']}\n"
        f"  Eligible:        {totals['n_elig']}\n"
        f"  Ineligible:      {totals['n_inelig']}\n"
        f"  MORNING tier:    {totals['n_morning']}\n"
        f"  EVENING tier:    {totals['n_evening']}\n"
    )
    print("── " + ("All spot checks PASSED ✓" if all_pass else "FAILURES detected ✗ — review above") + " ──")


# ── Main ──────────────────────────────────────────────────────────────────────

async def run(dict_path: Path, dry_run: bool, do_verify: bool) -> None:
    oi_dsn = os.getenv("OI_DATABASE_URL")
    if not oi_dsn:
        print("ERROR: OI_DATABASE_URL not set (check .env or environment).")
        sys.exit(1)

    rows = parse_dictionary(dict_path)

    if not rows:
        print("ERROR: No metrics parsed from dictionary. Check --dict-path.")
        sys.exit(1)

    eligible_count = sum(1 for r in rows if r["eligible_as_metric"])
    print(f"Parsed {len(rows)} metrics from {dict_path.name}:")
    print(f"  eligible_as_metric=true:  {eligible_count}")
    print(f"  eligible_as_metric=false: {len(rows) - eligible_count}")

    if dry_run:
        print("\n── DRY RUN — rows that would be written ────────────────────")
        for r in rows:
            elig_str = "eligible" if r["eligible_as_metric"] else "EXCLUDED"
            print(
                f"  Family {r['family_num']:>2}  {r['tier']:<12}  "
                f"{elig_str:<8}  {r['metric']}"
            )
        print(f"\nDry-run complete ({len(rows)} rows). No DB changes made.")
        return

    print(f"\nConnecting to OI database…")
    conn = await asyncpg.connect(dsn=oi_dsn)
    try:
        await write_to_db(conn, rows)
        print(f"Written {len(rows)} rows to metric_classification.")

        if do_verify:
            await verify(conn)
    finally:
        await conn.close()


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--dict-path",
        default=str(_ROOT / "daily_features_data_dictionary.md"),
        help="Path to data dictionary markdown file (default: project root)",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print parsed rows without writing to DB",
    )
    p.add_argument(
        "--verify",
        action="store_true",
        help="After writing, print spot-check results",
    )
    args = p.parse_args()

    asyncio.run(run(
        dict_path=Path(args.dict_path),
        dry_run=args.dry_run,
        do_verify=args.verify,
    ))


if __name__ == "__main__":
    main()
