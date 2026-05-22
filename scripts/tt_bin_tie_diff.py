"""Step 7k tied-value bin diff (targeted comparison on real data).

For several (ticker, feature) tuples and a train-test cutoff, compute
bin assignments under BOTH:

  - side='right' (Score Matrix's prior _tt_bin_matrix behavior)
  - side='left'  (the shared train_test_bin_matrix_per_ticker helper)

and report every test row whose bin assignment shifts between the two
methods. Tied test values (test value matches one or more training values)
are the only ones that can possibly shift, and even then only when the
tie crosses a bin boundary.

Run on the VPS where OI_DATABASE_URL points at the local Postgres instance:

    cd /spx_analysis_dashboard
    python scripts/tt_bin_tie_diff.py

Output is printed to stdout — paste back to the dev session.
"""
import asyncio
import math
import os

import asyncpg
import numpy as np
from dotenv import load_dotenv


load_dotenv()

CUTOFF = "2024-01-01"
N_BINS = 10

TICKERS = ["SPY", "QQQ", "AAL", "NVDA", "TSLA", "COIN", "AAPL", "AMD"]

# Mix of discrete-leaning and continuous features. Strikes are integers
# and frequently repeat (same strike holds max OI for many days), so
# max_oi_strike_* should produce the densest ties. The pct_* and ratio
# columns are continuous and should produce sparse-to-no ties.
FEATURES = [
    "max_oi_strike_call",
    "max_oi_strike_put",
    "total_oi",
    "call_oi",
    "put_call_oi_ratio",
    "pct_oi_in_front_expiry",
    "top5_strikes_pct_total_oi",
    "weighted_avg_dte",
]


def _safe(v):
    if v is None:
        return float("nan")
    try:
        x = float(v)
        return x if not math.isnan(x) else float("nan")
    except (TypeError, ValueError):
        return float("nan")


def _bin_old(train_sorted, n_train, v, n_bins):
    """Score Matrix's PRIOR side='right' behavior (one bin too high on ties)."""
    if math.isnan(v):
        return 0
    rank = int(np.searchsorted(train_sorted, v, side="right"))
    return min(int(rank / n_train * n_bins), n_bins - 1) + 1


def _bin_new(train_sorted, n_train, v, n_bins):
    """Shared helper's side='left' behavior — correct per the invariance argument."""
    if math.isnan(v):
        return 0
    rank = int(np.searchsorted(train_sorted, v, side="left"))
    return min((rank * n_bins) // n_train, n_bins - 1) + 1


async def main():
    dsn = os.environ["OI_DATABASE_URL"]
    conn = await asyncpg.connect(dsn)

    print(f"Cutoff: {CUTOFF}   n_bins: {N_BINS}")
    print(f"Tickers: {TICKERS}")
    print()

    grand_shifts = 0
    grand_tested = 0
    grand_ties = 0

    for ticker in TICKERS:
        print(f"--- {ticker} ---")
        for feat in FEATURES:
            try:
                rows = await conn.fetch(
                    f'SELECT trade_date, "{feat}" AS v FROM daily_features '
                    f'WHERE ticker = $1 AND "{feat}" IS NOT NULL '
                    f'ORDER BY trade_date',
                    ticker,
                )
            except Exception as e:
                print(f"  {feat:30s} : query failed ({e})")
                continue

            train_vals = []
            test_pairs = []
            for r in rows:
                date_s = str(r["trade_date"])
                v = _safe(r["v"])
                if math.isnan(v):
                    continue
                if date_s < CUTOFF:
                    train_vals.append(v)
                else:
                    test_pairs.append((date_s, v))

            if len(train_vals) < N_BINS or not test_pairs:
                print(f"  {feat:30s} : skipped (train={len(train_vals)}, test={len(test_pairs)})")
                continue

            train_sorted = np.sort(np.array(train_vals, dtype=np.float64))
            train_set = set(train_vals)
            n_train = len(train_sorted)

            shifts = []
            shifts_by_delta: dict = {}
            ties = 0
            for date, v in test_pairs:
                if v in train_set:
                    ties += 1
                b_old = _bin_old(train_sorted, n_train, v, N_BINS)
                b_new = _bin_new(train_sorted, n_train, v, N_BINS)
                if b_old != b_new:
                    shifts.append((date, v, b_old, b_new))
                    delta = b_new - b_old
                    shifts_by_delta[delta] = shifts_by_delta.get(delta, 0) + 1

            n_test = len(test_pairs)
            pct = 100 * len(shifts) / n_test
            tie_pct = 100 * ties / n_test
            shift_summary = (
                "  ".join(f"Δ={d:+d}: {n}" for d, n in sorted(shifts_by_delta.items()))
                if shifts_by_delta
                else "—"
            )
            print(
                f"  {feat:30s} : shifts {len(shifts):5d}/{n_test:5d} "
                f"({pct:5.2f}%)  ties {ties:5d} ({tie_pct:5.2f}%)  "
                f"shift dist: {shift_summary}"
            )

            for date, v, b_old, b_new in shifts[:3]:
                print(f"      e.g. {date}  v={v}  old bin {b_old} → new bin {b_new}")

            grand_shifts += len(shifts)
            grand_tested += n_test
            grand_ties += ties
        print()

    print(f"GRAND TOTAL: {grand_shifts} shifted / {grand_tested} test rows "
          f"({100*grand_shifts/max(grand_tested,1):.3f}%)")
    print(f"             {grand_ties} rows tied with training "
          f"({100*grand_ties/max(grand_tested,1):.3f}%)")
    print()
    print("Note: every shift is a tied value (only ties can possibly shift), but not")
    print("every tie causes a shift — only those whose rank diff crosses a bin boundary.")
    print("So #shifts ≤ #ties is expected.")

    await conn.close()


if __name__ == "__main__":
    asyncio.run(main())
