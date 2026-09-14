"""Calculations for binning and derived metrics.

COPIED from Options-Backtest-Dashboard/utils/calculations.py (211198c). The
bin boundaries are tuned and are unchanged. Two structural edits:

  * Each create_*_bins helper is split into a *_bin_spec() that returns the
    edges and labels, and a pd.cut over that spec. The page bins in the
    browser, so the edges have to reach JavaScript; deriving them from the
    same function the pandas path uses means there is one definition, not a
    Python copy and a JS copy. scripts/check_oo_backtest.py runs the shipped
    JS binning against pd.cut over every spec and fails on any disagreement.
  * The ratio bins, which the source hardcoded inline in its render callback
    (app.py, twice), are a spec here like the others.

filter_dataframe() loses its SharpTwo/regime parameters (dropped scope).
"""

import numpy as np
import pandas as pd
from scipy import stats

from app.oo_backtest.config import (
    VIX_BIN_RANGE, GAP_BIN_COUNT, GAP_BIN_RANGE, VIX_GAP_BIN_RANGE, PREMIUM_BINS, DAY_OF_WEEK,
    RATIO_BIN_RANGE, RATIO_BIN_STEP,
)

INF = float("inf")


def _spec(bins: list, labels: list, right: bool) -> dict:
    """A bin definition: full edge list (outer edges +/-inf) and one label per bin."""
    assert len(labels) == len(bins) - 1, (len(bins), len(labels))
    return {"bins": bins, "labels": labels, "right": right}


def apply_bin_spec(df: pd.DataFrame, column: str, spec: dict) -> pd.DataFrame:
    df = df.copy()
    df[f"{column}_bin"] = pd.cut(df[column], bins=spec["bins"], labels=spec["labels"],
                                 right=spec["right"])
    return df


def vix_bin_spec() -> dict:
    """Integer bins from VIX_BIN_RANGE[0] to VIX_BIN_RANGE[1], with edge bins for < and >."""
    min_val, max_val = VIX_BIN_RANGE

    # Create bins: <min, min, min+1, ..., max-1, >=max
    bins = [-INF] + list(range(min_val, max_val + 1)) + [INF]
    labels = [f"<{min_val}"] + [str(i) for i in range(min_val, max_val)] + [f"≥{max_val}"]
    return _spec(bins, labels, right=False)


def gap_bin_spec() -> dict:
    """GAP_BIN_COUNT percentage bins from GAP_BIN_RANGE[0] to GAP_BIN_RANGE[1]."""
    min_val, max_val = GAP_BIN_RANGE
    step = (max_val - min_val) / GAP_BIN_COUNT

    bins = [-INF] + [min_val + i * step for i in range(GAP_BIN_COUNT + 1)] + [INF]
    labels = [f"<{min_val:.1f}%"]
    for i in range(GAP_BIN_COUNT):
        start = min_val + i * step
        end = min_val + (i + 1) * step
        labels.append(f"{start:.1f}% to {end:.1f}%")
    labels.append(f"≥{max_val:.1f}%")
    return _spec(bins, labels, right=False)


def premium_bin_spec() -> dict:
    bins = [-INF] + PREMIUM_BINS + [INF]
    labels = [f"<${PREMIUM_BINS[0]}"]
    for i in range(len(PREMIUM_BINS) - 1):
        labels.append(f"${PREMIUM_BINS[i]} to ${PREMIUM_BINS[i+1]}")
    labels.append(f"≥${PREMIUM_BINS[-1]}")
    return _spec(bins, labels, right=False)


def vix_gap_bin_spec() -> dict:
    """VIX gaps are typically larger than SPX gaps, so we use a wider range."""
    min_val, max_val = VIX_GAP_BIN_RANGE
    num_bins = 20  # Same number of bins as SPX gap
    step = (max_val - min_val) / num_bins

    bins = [-INF] + [min_val + i * step for i in range(num_bins + 1)] + [INF]
    labels = [f"<{min_val:.0f}%"]
    for i in range(num_bins):
        start = min_val + i * step
        end = min_val + (i + 1) * step
        labels.append(f"{start:.0f}% to {end:.0f}%")
    labels.append(f"≥{max_val:.0f}%")
    return _spec(bins, labels, right=False)


def ratio_bin_spec() -> dict:
    """<0.70, then 0.70 -> 1.50 in steps of 0.04, then >1.50 (22 bins).

    As the source had it: RIGHT-closed (pandas' default, unlike every helper
    above), labelled by each bin's left edge. The source's outer edges were 0
    and 10, which dropped a ratio outside them; -inf/inf here. VIX term
    ratios cannot leave (0, 10], so no trade changes bin.
    """
    lo, hi = RATIO_BIN_RANGE
    n = round((hi - lo) / RATIO_BIN_STEP)
    bins = [-INF] + [lo + i * RATIO_BIN_STEP for i in range(n + 1)] + [INF]
    labels = [f"<{lo:.2f}"] + [f"{lo + i * RATIO_BIN_STEP:.2f}" for i in range(n)] + [f">{hi:.2f}"]
    return _spec(bins, labels, right=True)


def create_vix_bins(df: pd.DataFrame, column: str) -> pd.DataFrame:
    return apply_bin_spec(df, column, vix_bin_spec())


def create_gap_bins(df: pd.DataFrame, column: str = "gap") -> pd.DataFrame:
    return apply_bin_spec(df, column, gap_bin_spec())


def create_premium_bins(df: pd.DataFrame, column: str = "premium") -> pd.DataFrame:
    return apply_bin_spec(df, column, premium_bin_spec())


def create_vix_gap_bins(df: pd.DataFrame, column: str = "vix_overnight_gap") -> pd.DataFrame:
    return apply_bin_spec(df, column, vix_gap_bin_spec())


def create_ratio_bins(df: pd.DataFrame, column: str) -> pd.DataFrame:
    return apply_bin_spec(df, column, ratio_bin_spec())


def create_day_of_week_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Add day of week labels."""
    df = df.copy()
    df["day_of_week_label"] = df["day_of_week"].map(DAY_OF_WEEK)
    return df


def calculate_bin_stats(df: pd.DataFrame, bin_column: str, value_column: str = "pnl") -> pd.DataFrame:
    """
    Calculate statistics for each bin.

    Returns DataFrame with:
        - bin: bin label
        - count: number of trades
        - total_pnl: sum of P/L
        - avg_pnl: mean P/L
        - median_pnl: median P/L
        - win_rate: percentage of winning trades
        - std_pnl: standard deviation of P/L
    """
    grouped = df.groupby(bin_column, observed=True).agg(
        count=(value_column, "count"),
        total_pnl=(value_column, "sum"),
        avg_pnl=(value_column, "mean"),
        median_pnl=(value_column, "median"),
        std_pnl=(value_column, "std"),
    ).reset_index()

    # Calculate win rate
    win_rates = df.groupby(bin_column, observed=True).apply(
        lambda x: (x[value_column] > 0).mean() * 100, include_groups=False
    ).reset_index(name="win_rate")

    grouped = grouped.merge(win_rates, on=bin_column)
    grouped = grouped.rename(columns={bin_column: "bin"})

    return grouped


def calculate_correlation(df: pd.DataFrame, x_column: str, y_column: str = "pnl") -> dict:
    """
    Calculate correlation and regression statistics between two columns.

    Returns dict with:
        - correlation: Pearson correlation coefficient
        - p_value: p-value for correlation
        - slope: regression slope
        - intercept: regression intercept
        - r_squared: R² value
    """
    # Drop NaN values
    valid = df[[x_column, y_column]].dropna()

    if len(valid) < 3:
        return {
            "correlation": None,
            "p_value": None,
            "slope": None,
            "intercept": None,
            "r_squared": None,
        }

    x = valid[x_column].values
    y = valid[y_column].values

    # Correlation
    corr, p_value = stats.pearsonr(x, y)

    # Linear regression
    slope, intercept, r_value, _, _ = stats.linregress(x, y)

    return {
        "correlation": corr,
        "p_value": p_value,
        "slope": slope,
        "intercept": intercept,
        "r_squared": r_value ** 2,
    }


def calculate_cumulative_pnl(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate cumulative P/L over time.

    Returns DataFrame with date and cumulative_pnl columns.
    """
    df = df.copy()
    df = df.sort_values("date_closed")
    df["cumulative_pnl"] = df["pnl"].cumsum()
    return df[["date_closed", "pnl", "cumulative_pnl"]]


def calculate_drawdown(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate drawdown from cumulative P/L.

    Returns DataFrame with date, cumulative_pnl, peak, drawdown, drawdown_pct columns.
    """
    df = df.copy()
    df = df.sort_values("date_closed")
    df["cumulative_pnl"] = df["pnl"].cumsum()

    # Calculate running peak
    df["peak"] = df["cumulative_pnl"].cummax()

    # Drawdown is current value minus peak (negative when in drawdown)
    df["drawdown"] = df["cumulative_pnl"] - df["peak"]

    # Drawdown percentage (relative to peak, avoiding division by zero)
    df["drawdown_pct"] = np.where(
        df["peak"] != 0,
        (df["drawdown"] / df["peak"]) * 100,
        0
    )

    return df[["date_closed", "cumulative_pnl", "peak", "drawdown", "drawdown_pct"]]


def filter_dataframe(
    df: pd.DataFrame,
    date_range: tuple | None = None,
    exit_reasons: list | None = None,
    days_of_week: list | None = None,
    vix_range: tuple | None = None,
    vix3m_range: tuple | None = None,
    vix9d_range: tuple | None = None,
    gap_range: tuple | None = None,
    premium_range: tuple | None = None,
    vix3m_vix_ratio_range: tuple | None = None,
    vix_vix9d_ratio_range: tuple | None = None,
    vix_gap_range: tuple | None = None,
    wins_only: bool = False,
    losses_only: bool = False,
) -> pd.DataFrame:
    """
    Apply multiple filters to the dataframe.

    All filters are optional; None means no filter applied.
    """
    df = df.copy()

    # Ensure date columns are datetime (they may be strings after JSON round-trip)
    if "date_opened" in df.columns:
        df["date_opened"] = pd.to_datetime(df["date_opened"])
    if "date_closed" in df.columns:
        df["date_closed"] = pd.to_datetime(df["date_closed"])

    if date_range:
        start, end = date_range
        if start:
            df = df[df["date_opened"] >= pd.to_datetime(start)]
        if end:
            df = df[df["date_opened"] <= pd.to_datetime(end)]

    if exit_reasons:
        df = df[df["exit_reason"].isin(exit_reasons)]

    if days_of_week:
        df = df[df["day_of_week"].isin(days_of_week)]

    if vix_range and "vix_level" in df.columns:
        min_v, max_v = vix_range
        if min_v is not None:
            df = df[df["vix_level"] >= min_v]
        if max_v is not None:
            df = df[df["vix_level"] <= max_v]

    if vix3m_range and "vix3m_level" in df.columns:
        min_v, max_v = vix3m_range
        if min_v is not None:
            df = df[df["vix3m_level"] >= min_v]
        if max_v is not None:
            df = df[df["vix3m_level"] <= max_v]

    if vix9d_range and "vix9d_level" in df.columns:
        min_v, max_v = vix9d_range
        if min_v is not None:
            df = df[df["vix9d_level"] >= min_v]
        if max_v is not None:
            df = df[df["vix9d_level"] <= max_v]

    if gap_range and "gap" in df.columns:
        min_v, max_v = gap_range
        if min_v is not None:
            df = df[df["gap"] >= min_v]
        if max_v is not None:
            df = df[df["gap"] <= max_v]

    if premium_range:
        min_v, max_v = premium_range
        if min_v is not None:
            df = df[df["premium"] >= min_v]
        if max_v is not None:
            df = df[df["premium"] <= max_v]

    # VIX3M/VIX ratio filter (inverted ratio)
    if vix3m_vix_ratio_range and "vix3m_vix_ratio" in df.columns:
        min_v, max_v = vix3m_vix_ratio_range
        if min_v is not None:
            df = df[df["vix3m_vix_ratio"] >= min_v]
        if max_v is not None:
            df = df[df["vix3m_vix_ratio"] <= max_v]

    # VIX/VIX9D ratio filter
    if vix_vix9d_ratio_range and "vix_vix9d_ratio" in df.columns:
        min_v, max_v = vix_vix9d_ratio_range
        if min_v is not None:
            df = df[df["vix_vix9d_ratio"] >= min_v]
        if max_v is not None:
            df = df[df["vix_vix9d_ratio"] <= max_v]

    # VIX overnight gap filter
    if vix_gap_range and "vix_overnight_gap" in df.columns:
        min_v, max_v = vix_gap_range
        if min_v is not None:
            df = df[df["vix_overnight_gap"] >= min_v]
        if max_v is not None:
            df = df[df["vix_overnight_gap"] <= max_v]

    if wins_only:
        df = df[df["pnl"] > 0]
    elif losses_only:
        df = df[df["pnl"] <= 0]

    return df
