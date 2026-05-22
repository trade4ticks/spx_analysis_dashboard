"""Step 7k tied-value bin diff (targeted comparison on real data).

For every numeric feature column in `daily_features` and a fixed train-test
cutoff, compute bin assignments under BOTH:

  - side='right' (Score Matrix's prior _tt_bin_matrix behavior)
  - side='left'  (the shared train_test_bin_matrix_per_ticker helper)

across several tickers, then report which features have any test rows that
shift bins between the two methods. Tied test values (test value equals
one or more training values) are the only ones that can shift, and even
then only when the rank delta crosses a bin boundary.

The output is the complete list of features whose pre-Step-7k train-test
Score Matrix values could have been affected by the side='right' bug.

Run on the VPS where OI_DATABASE_URL points at the local Postgres instance:

    cd /spx_analysis_dashboard
    python scripts/tt_bin_tie_diff.py
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

# Mix of index ETFs, large caps, and a low-priced ticker (AAL) — picked to
# expose discreteness on both ends of the price scale. Discrete features
# manifest across all tickers, so this sample is sufficient for feature
# identification (which is structural). Adding more tickers would refine
# per-ticker shift counts but not change the affected-feature list.
TICKERS = ["SPY", "QQQ", "AAL", "NVDA", "TSLA", "COIN", "AAPL", "AMD"]


def _safe(v):
    if v is None:
        return float("nan")
    try:
        x = float(v)
        return x if not math.isnan(x) else float("nan")
    except (TypeError, ValueError):
        return float("nan")


def _bin_old(train_sorted, n_train, v, n_bins):
    """Score Matrix's PRIOR side='right' behavior."""
    if math.isnan(v):
        return 0
    rank = int(np.searchsorted(train_sorted, v, side="right"))
    return min(int(rank / n_train * n_bins), n_bins - 1) + 1


def _bin_new(train_sorted, n_train, v, n_bins):
    """Shared helper's side='left' behavior."""
    if math.isnan(v):
        return 0
    rank = int(np.searchsorted(train_sorted, v, side="left"))
    return min((rank * n_bins) // n_train, n_bins - 1) + 1


async def main():
    dsn = os.environ["OI_DATABASE_URL"]
    conn = await asyncpg.connect(dsn)

    # Discover all numeric feature columns via the same query the
    # dashboard's /columns endpoint uses.
    col_rows = await conn.fetch("""
        SELECT column_name FROM information_schema.columns
        WHERE table_name = 'daily_features' AND table_schema = 'public'
        AND data_type IN ('double precision','numeric','real',
                          'integer','bigint','smallint')
        AND column_name NOT IN ('id','ticker','trade_date','created_at','updated_at')
        ORDER BY ordinal_position
    """)
    all_cols = [r["column_name"] for r in col_rows]
    outcomes = {c for c in all_cols if "ret_" in c and "fwd" in c}
    features = [c for c in all_cols if c not in outcomes and not c.endswith("_pc")]

    print(f"Cutoff: {CUTOFF}   n_bins: {N_BINS}")
    print(f"Tickers: {TICKERS}")
    print(f"Features discovered: {len(features)}")
    print()

    per_feature: dict = {f: {
        "shifts":           0,
        "tested":           0,
        "ties":             0,
        "min_delta":        0,   # most negative delta seen (worst downward)
        "any_pos_delta":    False,
        "tickers_affected": set(),
    } for f in features}

    for ticker in TICKERS:
        print(f"  scanning {ticker} ...", flush=True)
        for feat in features:
            try:
                rows = await conn.fetch(
                    f'SELECT trade_date, "{feat}" AS v FROM daily_features '
                    f'WHERE ticker = $1 AND "{feat}" IS NOT NULL '
                    f'ORDER BY trade_date',
                    ticker,
                )
            except Exception:
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
                continue

            train_sorted = np.sort(np.array(train_vals, dtype=np.float64))
            train_set = set(train_vals)
            n_train = len(train_sorted)

            n_shifts = 0
            n_ties = 0
            for date, v in test_pairs:
                if v in train_set:
                    n_ties += 1
                b_old = _bin_old(train_sorted, n_train, v, N_BINS)
                b_new = _bin_new(train_sorted, n_train, v, N_BINS)
                if b_old != b_new:
                    n_shifts += 1
                    delta = b_new - b_old
                    if delta < per_feature[feat]["min_delta"]:
                        per_feature[feat]["min_delta"] = delta
                    if delta > 0:
                        per_feature[feat]["any_pos_delta"] = True

            per_feature[feat]["shifts"] += n_shifts
            per_feature[feat]["tested"] += len(test_pairs)
            per_feature[feat]["ties"] += n_ties
            if n_shifts > 0:
                per_feature[feat]["tickers_affected"].add(ticker)

    print()
    print("=" * 100)
    print("AFFECTED FEATURES (≥1 shifted test row across any of the sampled tickers)")
    print("=" * 100)
    affected = sorted(
        [(f, d) for f, d in per_feature.items() if d["shifts"] > 0],
        key=lambda x: -x[1]["shifts"],
    )
    print(f"{len(affected)} of {len(features)} features had ≥1 shift")
    print()
    if affected:
        print(f"{'feature':45s}  {'shifts':>8s} / {'tested':>8s}  {'pct':>7s}  "
              f"{'tickers':>9s}  {'worst Δ':>8s}")
        print("-" * 100)
        for feat, d in affected:
            pct = 100 * d["shifts"] / max(d["tested"], 1)
            n_tkrs = len(d["tickers_affected"])
            worst = f"{d['min_delta']:+d}" + ("!" if d["any_pos_delta"] else "")
            print(f"{feat:45s}  {d['shifts']:8d} / {d['tested']:8d}  "
                  f"{pct:6.2f}%  {n_tkrs:4d}/{len(TICKERS):<2d}  {worst:>8s}")

    print()
    print("=" * 100)
    print("UNAFFECTED FEATURES (0 shifted rows across all sampled tickers)")
    print("=" * 100)
    unaffected = sorted([f for f, d in per_feature.items() if d["shifts"] == 0])
    print(f"{len(unaffected)} features. List:")
    # Print in 3 columns
    rows_unaffected = [unaffected[i:i + 3] for i in range(0, len(unaffected), 3)]
    for row in rows_unaffected:
        print("  " + "  ".join(f"{f:32s}" for f in row))

    print()
    grand_shifts  = sum(d["shifts"] for d in per_feature.values())
    grand_tested  = sum(d["tested"] for d in per_feature.values())
    any_pos       = any(d["any_pos_delta"] for d in per_feature.values())
    print(f"Grand total: {grand_shifts} shifts across {grand_tested} test rows "
          f"({100*grand_shifts/max(grand_tested,1):.4f}%)")
    if any_pos:
        print("WARNING: at least one feature had a POSITIVE-delta shift — unexpected!")
    else:
        print("All shifts had negative delta (side='right' was always too high). "
              "Consistent with the invariance argument: side='left' is correct.")

    await conn.close()


if __name__ == "__main__":
    asyncio.run(main())
