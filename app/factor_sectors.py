"""Ticker -> sector bucket, for the Factor Trades result panes.

THE 11 GICS SECTORS, plus three buckets only where GICS fails (the user's
call, 2026-11-07, replacing a first "what moves together" taxonomy):
  Crypto       the names that move with bitcoin, which GICS would scatter
               across Financials and Technology
  Broad index  index products, which are every sector at once (ARKK too)
  Bonds        Treasury ETFs, which have no sector
Everything else is in its GICS sector, with the 2023 GICS changes applied
(payments -- V, MA, PYPL -- in Financials; WMT, COST, TGT in Consumer
Staples; UBER in Industrials). A sector ETF counts against its own sector at
full weight (XLF in Financials).

ASSIGNED BY JUDGEMENT, not by GICS -- these products have no GICS sector and
sit in the nearest of the fourteen (the page's map is the place to change
them): country ETFs EEM EWY EWZ FXI KWEB -> Broad index; GLD SLV ->
Materials; USO UNG -> Energy; VXX -> Broad index.

WHY A FILE, NOT A TABLE. No ticker-attribute table exists in either database
(the scalp DB's `universe` is a different list), and a ticker's bucket
should be reviewed as a diff. The page diffs this map against
`SELECT DISTINCT ticker FROM daily_features` (/api/factor-analysis/tickers)
and says what is missing on either side; a traded ticker missing here is
shown as UNASSIGNED, never dropped. Every ticker is in exactly one bucket
(check_ft_sectors holds it).

The order below is the chip order on the page: GICS order, then the three.
"""
from __future__ import annotations

UNASSIGNED = "Unassigned"

BUCKETS: list[tuple[str, tuple[str, ...]]] = [
    ("Communication Services", ("GOOGL", "META", "NFLX", "DIS", "SNAP", "SPOT", "T", "VZ")),
    ("Consumer Discretionary", ("AMZN", "TSLA", "HD", "NKE", "SBUX", "LULU", "CMG", "F", "RIVN",
                                "LCID", "NIO", "BABA", "JD", "CVNA", "CCL", "PTON")),
    ("Consumer Staples",       ("WMT", "COST", "TGT", "KO", "XLP")),
    ("Energy",                 ("XOM", "CVX", "OXY", "XLE", "XOP", "USO", "UNG")),
    ("Financials",             ("JPM", "BAC", "C", "WFC", "GS", "V", "MA", "PYPL", "SOFI",
                                "XLF", "KRE")),
    ("Health Care",            ("UNH", "CVS", "HUM", "JNJ", "PFE", "LLY", "MRNA", "HIMS",
                                "XLV", "XBI")),
    ("Industrials",            ("CAT", "GE", "BA", "FDX", "UPS", "UBER", "AAL", "DAL", "UAL",
                                "HTZ", "PLUG", "RKLB", "XLI")),
    ("Information Technology", ("AAPL", "MSFT", "NVDA", "AMD", "AVGO", "MU", "INTC", "AMAT",
                                "TSM", "MRVL", "SMCI", "WDC", "ADBE", "CRM", "NOW", "INTU",
                                "ORCL", "PANW", "CRWD", "SNOW", "SHOP", "PLTR", "IBM", "CSCO",
                                "IONQ", "XLK")),
    ("Materials",              ("VALE", "XLB", "GLD", "SLV")),
    ("Real Estate",            ("OPEN",)),
    ("Utilities",              ("XLU",)),
    ("Crypto",                 ("MSTR", "MARA", "RIOT", "CIFR", "COIN", "HOOD")),
    ("Broad index",            ("SPY", "QQQ", "IWM", "DIA", "ARKK",
                                "EEM", "EWY", "EWZ", "FXI", "KWEB", "VXX")),
    ("Bonds",                  ("TLT", "IEF")),
]


def sector_map() -> dict[str, str]:
    return {t: name for name, tickers in BUCKETS for t in tickers}


def payload() -> dict:
    """What the page fetches once: the buckets in chip order."""
    return {"buckets": [{"name": n, "tickers": list(t)} for n, t in BUCKETS],
            "unassigned": UNASSIGNED}
