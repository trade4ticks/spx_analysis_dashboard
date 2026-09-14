"""Bin and market-data constants for the OO/Mesosim Backtest page.

Trimmed from Options-Backtest-Dashboard/config.py (211198c). The bin ranges
are tuned and are kept at their source values; what is gone is the Dash
styling (CHART_HEIGHT, STATS_CARD_STYLE), the SQLite path, and
SKEW_BIN_RANGE, which belonged to the dropped skew section.
"""

# Market data settings
MARKET_DATA_START = "2010-01-01"  # Backfill to 2010

# yfinance tickers
TICKERS = {
    "SPX": "^GSPC",
    "VIX": "^VIX",
    "VIX3M": "^VIX3M",
    "VIX9D": "^VIX9D",
}

# Bin configurations
VIX_BIN_RANGE = (9, 50)  # Integer bins from 9 to 50+
GAP_BIN_COUNT = 20  # Number of bins for overnight gap percentages
GAP_BIN_RANGE = (-2.0, 2.0)  # SPX Gap percentage range
VIX_GAP_BIN_RANGE = (-10.0, 10.0)  # VIX gaps are larger than SPX
VIX9D_RATIO_RANGE = (0.7, 1.5)  # VIX/VIX9D ratio range

# Ratio bins. The source app hardcoded these inline in its render callback;
# they live here now so both ratio sections share one definition.
RATIO_BIN_RANGE = (0.70, 1.50)
RATIO_BIN_STEP = 0.04

# Premium bins (dollars)
PREMIUM_BINS = [-2000, -1500, -1000, -750, -500, -250, 0, 250, 500, 750, 1000, 1500, 2000]

# Day of week mapping
DAY_OF_WEEK = {
    0: "Monday",
    1: "Tuesday",
    2: "Wednesday",
    3: "Thursday",
    4: "Friday",
    5: "Saturday",
    6: "Sunday",
}
