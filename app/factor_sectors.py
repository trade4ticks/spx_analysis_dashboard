"""Ticker -> sector bucket, for the Factor Trades result panes.

WHY NOT GICS. A quarter of the universe is not single-name equity -- index,
sector, country, commodity, bond and crypto-linked products -- and GICS would
scatter the crypto names across Financials and Tech when they move with
bitcoin. These buckets group what MOVES TOGETHER, which is the question a
concentration cap asks. Agreed with the user, 2026-11-07.

WHY A FILE, NOT A TABLE. No ticker-attribute table exists in either database
(the scalp DB's `universe` is a different list), and a ticker's bucket is a
judgement that should be reviewed as a diff. The page diffs this map against
`SELECT DISTINCT ticker FROM daily_features` (/api/factor-analysis/tickers)
and says what is missing on either side, so a new or dead ticker shows up on
screen rather than silently landing in "Unassigned".

RULES.
  * A sector ETF counts against its sector at full weight (XLF in Financials).
  * Broad index products are their own bucket: they are everything at once.
  * Every ticker is in exactly one bucket (check_ft_sectors holds it).
  * A traded ticker missing from here is shown as UNASSIGNED, never dropped.

The order below is the chip order on the page.
"""
from __future__ import annotations

UNASSIGNED = "Unassigned"

BUCKETS: list[tuple[str, tuple[str, ...]]] = [
    ("Broad index",             ("SPY", "QQQ", "IWM", "DIA")),
    ("Mega-cap & platforms",    ("AAPL", "MSFT", "GOOGL", "AMZN", "META", "NFLX", "SPOT", "UBER",
                                 "XLK")),
    ("Semis & AI hardware",     ("NVDA", "AMD", "AVGO", "MU", "INTC", "AMAT", "TSM", "MRVL", "SMCI",
                                 "WDC")),
    ("Software",                ("ADBE", "CRM", "NOW", "INTU", "ORCL", "PANW", "CRWD", "SNOW", "SHOP",
                                 "PLTR", "IBM", "CSCO")),
    ("Crypto",                  ("MSTR", "MARA", "RIOT", "CIFR", "COIN", "HOOD", "IBIT", "ETHA",
                                 "CORZ")),
    # ARKK is here, not under Broad index: it holds TSLA / COIN-type names and
    # moves with them. TSLA and the EV names likewise.
    ("Speculative growth",      ("ARKK", "TSLA", "RIVN", "LCID", "PLUG", "IONQ", "QBTS", "RKLB",
                                 "APLD", "OPEN", "CVNA", "SOFI", "HIMS", "GME", "AMC", "PTON",
                                 "SNAP")),
    ("China & EM",              ("FXI", "KWEB", "BABA", "JD", "NIO", "EEM", "EWY", "EWZ", "VALE")),
    ("Financials",              ("JPM", "BAC", "C", "WFC", "GS", "ALL", "V", "MA", "PYPL",
                                 "XLF", "KRE")),
    ("Energy",                  ("XOM", "CVX", "OXY", "XLE", "XOP", "USO", "UNG")),
    ("Precious metals",         ("GLD", "SLV")),
    # Split from pharma at the user's request: managed care fires as a
    # cluster, which is exactly the correlated exposure a cap is for.
    ("Managed care",            ("UNH", "CVS", "HUM")),
    ("Pharma & biotech",        ("JNJ", "PFE", "LLY", "MRNA", "XLV", "XBI")),
    ("Consumer",                ("WMT", "COST", "TGT", "KO", "HD", "SBUX", "NKE", "LULU", "CMG",
                                 "DIS", "XLP")),
    ("Travel",                  ("AAL", "DAL", "UAL", "CCL", "HTZ")),
    ("Industrials & materials", ("CAT", "GE", "BA", "FDX", "UPS", "F", "XLI", "XLB")),
    # Utilities and telecom trade on rates, so they sit with the bonds.
    ("Rates & yield",           ("TLT", "IEF", "XLU", "T", "VZ")),
    # Long volatility: moves against everything else, so it stays alone.
    ("Volatility",              ("VXX",)),
]


def sector_map() -> dict[str, str]:
    return {t: name for name, tickers in BUCKETS for t in tickers}


def payload() -> dict:
    """What the page fetches once: the buckets in chip order."""
    return {"buckets": [{"name": n, "tickers": list(t)} for n, t in BUCKETS],
            "unassigned": UNASSIGNED}
