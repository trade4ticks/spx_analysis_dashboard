"""OO/Mesosim Backtest: parameter sweeps, offline (P1 -- the server half).

A sweep is several trade logs of one strategy loaded as ONE trade set, each
trade tagged with its variant. Everything here fails quietly if it is wrong:
a filename read as the wrong parameters, two variants merged into one, a
trade the vendor excluded kept in every figure. So:

  * filenames: what is read from each, and what is ignored -- dte45, dte_45,
    dte=45, decimals, negatives, a word value, a key given twice
  * precedence: param_* columns beat the filename; a single file's NAME is
    never read; a file with neither, alone, is the old single-log payload
  * refusals that load nothing and name the file: a different parameter set,
    the same variant twice ("45" from a column equals 45 from a name), a
    blank param cell, the same file twice, a batch with no parameters at all
  * variant ids follow numeric order (5 < 10 < 45, never "10" < "5")
  * isIgnored: dropped at parse, counted, absent column untouched, a value
    that is neither true nor false refused
  * the payload: one int `variant` column, whitelisted, every trade tagged,
    counts per variant adding up to the trade count
  * the endpoints, through FastAPI: gzipped parts are decompressed, the
    response is compressed here (the middleware leaves it alone), a refusal is
    a 400 naming the file, the size cap refuses, /parse still answers
  * the hand-rolled encoder sends what jsonable_encoder + JSONResponse would

No database: the market join fails into its "not joined" report, as it does
when index_ohlc is unreachable. The SQL gate runs a batch through a real join.

    python scripts/check_oo_sweep.py
"""
from __future__ import annotations

import gzip
import io
import json
import logging
import sys
from datetime import date, datetime, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from app.oo_backtest import data_loader, sweep  # noqa: E402
from app.oo_backtest.payload import allowed_column  # noqa: E402

FAILS: list[str] = []


def check(ok: bool, msg: str) -> None:
    print(("  ok    " if ok else "  FAIL  ") + msg)
    if not ok:
        FAILS.append(msg)


def raises(fn, needle: str, msg: str) -> None:
    try:
        fn()
    except (sweep.SweepError, ValueError) as exc:
        check(needle in str(exc), f"{msg}  [{exc}]")
        return
    check(False, f"{msg}  [did not refuse]")


HEADER = ["Date Opened", "Time Opened", "Opening Price", "Legs", "Premium", "Closing Price",
          "Date Closed", "Time Closed", "Avg. Closing Cost", "Reason For Close", "P/L", "P/L %",
          "No. of Contracts", "Funds at Close", "Margin Req.", "Strategy", "Gap", "Movement"]


def row(d0: str, d1: str, pnl: float, t0: str = "10:00:00") -> list[str]:
    return [d0, t0, "5100", "1 Mar 8 P 5000 STO", "1250", "5120", d1, "15:59:00", "0", "Expired",
            str(pnl), "10", "1", "100000", "5000", "IC", "0.3", "20"]


ROWS = [row("2024-03-04", "2024-03-06", 100), row("2024-03-05", "2024-03-08", -50),
        row("2024-03-07", "2024-03-08", 25)]


def csv(rows=ROWS, extra: dict | None = None) -> bytes:
    """An Option Omega CSV; `extra` = {column: [value per row]}."""
    extra = extra or {}
    head = HEADER + list(extra)
    out = io.StringIO()
    pd.DataFrame([r + [extra[c][i] for c in extra] for i, r in enumerate(rows)], columns=head).to_csv(out, index=False)
    return out.getvalue().encode()


def parse(content: bytes, name: str) -> pd.DataFrame:
    df = data_loader.parse_upload(content, name)
    ok, msg = data_loader.validate_data(df)
    if not ok:
        raise ValueError(msg)
    return df


def combine(*files):
    return sweep.combine([(n, parse(c, n)) for n, c in files])


# ── filenames ───────────────────────────────────────────────────────────────

def check_filenames() -> None:
    print("filenames")
    cases = [
        ("sweep_dte45_pt50_sl200.csv", {"dte": 45, "pt": 50, "sl": 200}, ["sweep"]),
        ("dte=45 pt=0.5.csv", {"dte": 45, "pt": 0.5}, []),
        ("sweep_dte_45_pt_50.csv", {"dte": 45, "pt": 50}, ["sweep"]),
        ("ic-delta0.16-dte7.csv", {"delta": 0.16, "dte": 7}, ["ic"]),
        ("side=put_dte45.csv", {"side": "put", "dte": 45}, []),
        ("off-10_dte7.csv", {"off": -10, "dte": 7}, []),
        ("DTE45_PT50.csv", {"dte": 45, "pt": 50}, []),
        ("pt.5_dte45.csv", {"pt": 0.5, "dte": 45}, []),
        ("trade-log 2024-01-05.csv", {}, ["trade", "log", "2024-01-05"]),
        # A MesoSim export's name. "v2" and "SPX_.35" read as parameters, the
        # weekday that actually differs ("mon") does not -- which is why each
        # file's reading is REPORTED: five such files are refused as one variant.
        ("mesosim-allantis._allantis - v2_ 5+4+1, PT SPX_.35, 60DIT, mon, 2021-2026-260503_012305.events.json",
         {"v": 2, "spx": 0.35}, None),
    ]
    for name, want, ignored in cases:
        got = sweep.params_from_filename(name)
        ok = got["params"] == want and all(type(got["params"][k]) is type(want[k]) for k in want)
        if ignored is not None:
            ok = ok and got["ignored"] == ignored
        check(ok, f"{name!r} -> {got['params']} (ignored {got['ignored']})")
    check(sweep.params_from_filename("sweep_dte45_pt50_sl200.csv")["order"] == ["dte", "pt", "sl"],
          "keys keep the filename's order")
    raises(lambda: sweep.params_from_filename("dte45_DTE50.csv"), 'dte45_DTE50.csv: "dte" appears twice',
           "a key given twice in one name is refused, naming the file")


# ── combining ───────────────────────────────────────────────────────────────

def check_combine() -> None:
    print("combine")
    one = parse(csv(), "anything_dte45.csv")
    df, sw = sweep.combine([("anything_dte45.csv", one)])
    check(sw is None and df is one and sweep.VARIANT_COLUMN not in df.columns,
          "one file, no param_* columns: the file's own frame, no sweep -- its NAME is not read")

    df, sw = combine(("mcp.csv", csv(extra={"param_delta": [16, 16, 30], "param_pt": ["50", "50", "50"]})))
    check(sw is not None and sw["params"] == ["delta", "pt"] and [v["label"] for v in sw["variants"]]
          == ["delta=16 · pt=50", "delta=30 · pt=50"], "one MCP file with param_* columns: one variant per combination")
    check(df[sweep.VARIANT_COLUMN].tolist() == [0, 0, 1] and not any(c.startswith("param_") for c in df.columns),
          "rows tagged with their variant; the param_* columns themselves are dropped")
    check(sw["files"][0]["source"] == "columns" and sw["files"][0]["variants"] == 2, "the file reports its source")

    files = [(f"sweep_dte{d}_pt50.csv", csv()) for d in (45, 5, 10)]
    df, sw = combine(*files)
    check([v["params"]["dte"] for v in sw["variants"]] == [5, 10, 45], "variant ids in NUMERIC order: 5, 10, 45")
    check([v["file"] for v in sw["variants"]] == ["sweep_dte5_pt50.csv", "sweep_dte10_pt50.csv", "sweep_dte45_pt50.csv"],
          "each variant names its file")
    check(df[sweep.VARIANT_COLUMN].tolist() == [2] * 3 + [0] * 3 + [1] * 3, "each file's trades carry its variant id")
    check(sum(v["n"] for v in sw["variants"]) == len(df) == 9, "per-variant counts add up to the trade count")
    check(sw["values"] == {"dte": [5, 10, 45], "pt": [50]}, "distinct values per parameter, sorted")
    check(sw["suggested_name"] == "sweep", f"suggested name from the shared stem ({sw['suggested_name']!r})")
    check(all(f["source"] == "filename" and f["ignored_tokens"] == ["sweep"] for f in sw["files"]),
          "each file reports what its name was read as, and what was ignored")

    # Columns beat the filename.
    df, sw = combine(("sweep_dte45.csv", csv(extra={"param_dte": [7, 7, 7]})),
                     ("sweep_dte60.csv", csv()))
    check([v["params"]["dte"] for v in sw["variants"]] == [7, 60] and sw["files"][0]["source"] == "columns",
          "a file with param_* columns uses them, not its name")

    # Key order may differ between files; the set may not.
    _, sw = combine(("a_dte45_pt50.csv", csv()), ("b_pt75_dte45.csv", csv()))
    check(sw["params"] == ["dte", "pt"] and len(sw["variants"]) == 2, "dte/pt and pt/dte are the same parameter set")

    raises(lambda: combine(("s_dte45_pt50.csv", csv()), ("s_dte50_pt50.csv", csv()), ("s_dte55_pt50.csv", csv()),
                           ("s_dte45_sl200.csv", csv())),
           "s_dte45_sl200.csv has dte, sl; the other files have dte, pt", "a different parameter set names THAT file")
    raises(lambda: combine(("s_dte45.csv", csv()), ("s_dte50.csv", csv()), ("export.csv", csv())),
           "export.csv has no parameters", "a file with no parameters among a sweep is named")
    raises(lambda: combine(("a.csv", csv()), ("b.csv", csv())), "None of these 2 files carries parameters",
           "a batch with nothing to tell the files apart is refused")
    raises(lambda: combine(("x_dte45.csv", csv()), ("y_dte45.csv", csv())),
           "x_dte45.csv and y_dte45.csv are the same variant (dte=45)", "the same variant twice names both files")
    raises(lambda: combine(("x.csv", csv(extra={"param_dte": ["45", "45", "45"]})), ("y_dte45.csv", csv())),
           "same variant", '"45" from a column and 45 from a filename are the same variant')
    raises(lambda: combine(("x.csv", csv(extra={"param_dte": [45, "", 45]}))),
           'x.csv: 1 of 3 rows have no value in "param_dte"', "a blank param cell is refused")
    raises(lambda: combine(("x.csv", csv(extra={"param_dte": [1, 1, 1], "PARAM_DTE": [2, 2, 2]}))),
           "are the same parameter", "two param_* columns differing only in case are refused")
    raises(lambda: sweep.combine([("a_dte1.csv", parse(csv(), "a.csv")), ("a_dte1.csv", parse(csv(), "a.csv"))]),
           "a_dte1.csv is in the batch twice", "the same file twice is refused")
    # Strings sort after numbers, alphabetically.
    _, sw = combine(("s_side=put.csv", csv()), ("s_side=call.csv", csv()))
    check([v["params"]["side"] for v in sw["variants"]] == ["call", "put"], "word values sort alphabetically")


# ── isIgnored ───────────────────────────────────────────────────────────────

def check_ignored() -> None:
    print("isIgnored")
    df = parse(csv(extra={"isIgnored": ["false", "true", ""]}), "mcp.csv")
    check(len(df) == 2 and df.attrs["parse_notes"]["ignored_dropped"] == 1 and "isIgnored" not in df.columns,
          "a true row is dropped and counted; a blank reads as false")
    df = parse(csv(extra={"isIgnored": [True, True, False]}), "mcp.csv")
    check(len(df) == 1 and df.attrs["parse_notes"]["ignored_dropped"] == 2, "a boolean column is read as booleans")
    df = parse(csv(extra={"isIgnored": [False, False, False]}), "mcp.csv")
    check(len(df) == 3 and df.attrs["parse_notes"] == {"ignored_dropped": 0},
          "the column present with nothing ignored reports 0 dropped")
    df = parse(csv(), "site.csv")
    check(len(df) == 3 and df.attrs.get("parse_notes") == {}, "no column: nothing dropped, nothing reported")
    raises(lambda: parse(csv(extra={"isIgnored": ["true", "maybe", "false"]}), "mcp.csv"),
           "mcp.csv: column \"isIgnored\" holds 'maybe'", "a value that is neither true nor false is refused")
    raises(lambda: parse(csv(extra={"isIgnored": ["true"] * 3}), "mcp.csv"), "no trade data",
           "a file with every trade ignored is an empty log")
    df, sw = combine(("a_dte1.csv", csv(extra={"isIgnored": ["true", "false", "false"]})),
                     ("a_dte2.csv", csv(extra={"isIgnored": ["true", "true", "false"]})))
    check(df.attrs["parse_notes"]["ignored_dropped"] == 3 and [f["ignored_dropped"] for f in sw["files"]] == [1, 2]
          and len(df) == 3, "a sweep sums the dropped count, and each file reports its own")


# ── the MCP export: a third format ──────────────────────────────────────────

# Exactly as exported (2026-10-18), and the website export's header beside it.
MCP_HEADER = ("param_delta,param_pt,dateOpened,timeOpened,dateClosed,timeClosed,daysInTrade,underlying,legs,"
              "numberOfContracts,premiumPerContract,averageClosingCostPerContract,openingUnderlyingPrice,"
              "closingUnderlyingPrice,reasonClosed,profit,profitPercentage,fundsAtClose,buyingPowerRequired,"
              "openingFees,closingFees,shortLongRatio,closingShortLongRatio,gap,intradayMovement,"
              "highestProfitPercentage,highestLossPercentage,isIgnored").split(",")
SITE_HEADER = ["Date Opened", "Time Opened", "Opening Price", "Legs", "Premium", "Closing Price", "Date Closed",
               "Time Closed", "Days in Trade", "Avg. Closing Cost", "Reason For Close", "P/L", "P/L %",
               "No. of Contracts", "Funds at Close", "Margin Req.", "Underlying", "Strategy",
               "Opening Commissions + Fees", "Closing Commissions + Fees", "Opening Short/Long Ratio",
               "Closing Short/Long Ratio", "Gap", "Movement", "Max Profit", "Max Loss"]
# The same three trades, as each export would carry them: open date, time,
# close date, time, legs, contracts, premium, closing cost, open px, close px,
# reason, P/L, P/L %, funds, margin, open fee, close fee, s/l, closing s/l,
# gap, movement, highest profit %, highest loss %.
TRADES = [
    ("2024-03-04", "10:00:00", "2024-03-06", "15:59:00", "1 Mar 8 P 5000 STO", 2, 125.0, 20.0, 5100.5, 5120.25,
     "Profit Target", 210.0, 84.0, 100210, 5000, 2.6, 2.6, 0.5, 0.4, 0.31, 20.1, 90.0, -15.0),
    ("2024-03-05", "10:00:00", "2024-03-08", "11:00:00", "1 Mar 8 P 4980 STO", 1, 90.0, 210.0, 5080, 5010,
     "Stop Loss", -120.0, -133.3, 100090, 4000, 1.3, 1.3, 0.5, 0.6, -0.62, -70, 10.0, -140.0),
    ("2024-03-07", "10:30:00", "2024-03-08", "15:59:00", "1 Mar 8 P 4990 STO", 1, 70.0, 0.0, 5110, 5130,
     "Expired", 70.0, 100.0, 100160, 3500, 1.3, 0, 0.5, 0.5, 0.1, 10, 100.0, -5.0),
]


def mcp_csv(params=((16, 50),) * 3, ignored=("false",) * 3, dates=None, drop: str | None = None) -> bytes:
    rows = []
    for i, t in enumerate(TRADES):
        (d0, t0, d1, t1, legs, n, prem, cc, opx, cpx, why, pl, plp, funds, bp, of, cf, sl, csl, gap, mv, hi, lo) = t
        if dates:
            d0, d1 = dates[i]
        rows.append([params[i][0], params[i][1], d0, t0, d1, t1, 2, "SPX", legs, n, prem, cc, opx, cpx, why, pl,
                     plp, funds, bp, of, cf, sl, csl, gap, mv, hi, lo, ignored[i]])
    df = pd.DataFrame(rows, columns=MCP_HEADER)
    if drop:
        df = df.drop(columns=[drop])
    out = io.StringIO()
    df.to_csv(out, index=False)
    return out.getvalue().encode()


def site_csv() -> bytes:
    rows = []
    for (d0, t0, d1, t1, legs, n, prem, cc, opx, cpx, why, pl, plp, funds, bp, of, cf, sl, csl, gap, mv, hi, lo) in TRADES:
        rows.append([d0, t0, opx, legs, prem, cpx, d1, t1, 2, cc, why, pl, plp, n, funds, bp, "SPX", "IC", of, cf,
                     sl, csl, gap, mv, "", ""])
    out = io.StringIO()
    pd.DataFrame(rows, columns=SITE_HEADER).to_csv(out, index=False)
    return out.getvalue().encode()


def check_mcp() -> None:
    print("MCP export format")
    from app.routers.oo_backtest import _parse
    mapped = set(data_loader.MCP_COLUMN_MAPPING)
    unread = [c for c in MCP_HEADER if c not in mapped and not c.startswith("param_") and c != "isIgnored"]
    check(unread == ["daysInTrade", "underlying"],
          f"every MCP column is mapped, a sweep parameter, isIgnored, or deliberately unread ({unread})")
    check(data_loader.csv_format(MCP_HEADER) == "oo_mcp_csv" and data_loader.csv_format(SITE_HEADER) == "oo_csv",
          "the header decides the format: MCP vs the website's")

    df = parse(mcp_csv(), "mcp.csv")
    check(df.attrs["source_format"] == "oo_mcp_csv", "an MCP file parses as oo_mcp_csv")
    p = _parse(mcp_csv(), "mcp.csv")
    c = p["columns"]
    check(p["source"] == "oo_mcp_csv" and p["n"] == 3, "payload source oo_mcp_csv, 3 trades")
    check(c["highest_profit_pct"] == [90.0, 10.0, 100.0] and c["highest_loss_pct"] == [-15.0, -140.0, -5.0],
          "highestProfitPercentage / highestLossPercentage reach the payload")
    check(c["margin_req"] == [5000.0, 4000.0, 3500.0], "buyingPowerRequired is the trade's margin_req")
    check("gap" not in c and "csv_gap" not in c and "csv_gap" in df.columns,
          "the vendor's gap is csv_gap and reaches the payload under no name -- never as our SPX gap")

    # The same trades through both exports agree on every shared field.
    w = _parse(site_csv(), "site.csv")["columns"]
    shared = ["date_opened", "time_opened", "date_closed", "time_closed", "days_in_trade", "legs", "contracts",
              "premium", "pnl", "pnl_pct", "margin_req", "exit_reason", "spx_open_price", "spx_close_price",
              "is_win", "day_of_week", "year"]
    diff = [k for k in shared if c.get(k) != w.get(k)]
    check(not diff, f"the same trades from the website CSV and the MCP export: identical shared columns {diff or ''}")
    check(_parse(site_csv(), "site.csv")["source"] == "oo_csv" and "highest_profit_pct" not in w,
          "the website export (its full current header) still parses as oo_csv, with no excursion columns")

    df = parse(mcp_csv(dates=[("2024-03-04T00:00:00Z", "2024-03-06T00:00:00Z")] * 3), "mcp.csv")
    check([str(d.date()) for d in df["date_opened"]] == ["2024-03-04"] * 3,
          "a zoned midnight-UTC date keeps its written date (not shifted to the day before)")
    df = parse(mcp_csv(ignored=("false", "true", "false")), "mcp.csv")
    check(len(df) == 2 and df.attrs["parse_notes"]["ignored_dropped"] == 1, "isIgnored applies to the MCP export")
    raises(lambda: parse(mcp_csv(drop="profit"), "mcp.csv"), 'mcp.csv: missing column "profit"',
           "an MCP header missing a required column names it in the MCP's own spelling")

    # One MCP file, two variants, an ignored row: a sweep.
    df, sw = combine(("mcp.csv", mcp_csv(params=((16, 50), (30, 50), (16, 50)), ignored=("false", "false", "true"))))
    check(bool(sw) and [v["label"] for v in sw["variants"]] == ["delta=16 · pt=50", "delta=30 · pt=50"]
          and sw["files"][0]["ignored_dropped"] == 1 and len(df) == 2,
          "an MCP file carrying two variants loads as a sweep, its ignored row dropped")


# ── payload and endpoints ───────────────────────────────────────────────────

def check_endpoints() -> None:
    print("endpoints")
    from fastapi import FastAPI
    from fastapi.encoders import jsonable_encoder
    from fastapi.testclient import TestClient

    from app.db import get_pool
    from app.routers import oo_backtest as R
    from starlette.middleware.gzip import GZipMiddleware

    # Without a pool the join fails and logs its traceback, as it should on a
    # real failure; here it is the expected path, so it is not printed.
    logging.getLogger("app.routers.oo_backtest").setLevel(logging.CRITICAL)
    app = FastAPI()
    app.add_middleware(GZipMiddleware, minimum_size=1024)    # as app/main.py has it
    app.include_router(R.router, prefix="/api/oo-backtest")
    app.dependency_overrides[get_pool] = lambda: None        # the join reports "not joined"
    c = TestClient(app)

    def post(files, path="/api/oo-backtest/parse-batch"):
        return c.post(path, files=[("files" if "batch" in path else "file", f) for f in files])

    files = [(f"sweep_dte{d}_pt{p}.csv", gzip.compress(csv())) for d in (30, 45) for p in (25, 50)]
    r = post(files)
    body = r.json() if r.status_code == 200 else {}
    check(r.status_code == 200 and r.headers.get("content-encoding") == "gzip",
          f"a gzipped four-file batch loads; the response is gzip ({r.status_code} {r.headers.get('content-encoding')})")
    sw = body.get("sweep") or {}
    cols = body.get("columns", {})
    check(body.get("n") == 12 and len(sw.get("variants", [])) == 4 and len(cols.get("variant", [])) == 12
          and set(cols["variant"]) == {0, 1, 2, 3}, "12 trades, 4 variants, every trade tagged")
    check(all(allowed_column(k) for k in cols) and all(isinstance(v, int) for v in cols.get("variant", [])),
          "the variant column is a whitelisted int column")
    check(body.get("filename") == "4 files" and body.get("source") == "oo_csv" and body.get("suggested_name") == "sweep",
          "batch filename / source / suggested name")
    check(set(sw.get("timing", {})) == {"parse_s", "join_payload_s"}, "the sweep reports its own server timing")
    r = post([("m1.csv", mcp_csv(params=((16, 50),) * 3)), ("m2.csv", gzip.compress(mcp_csv(params=((30, 50),) * 3))),
              ("site_delta45_pt50.csv", site_csv())])
    b = r.json() if r.status_code == 200 else {}
    check(r.status_code == 200 and b.get("source") == "mixed" and len((b.get("sweep") or {}).get("variants", [])) == 3,
          f"two MCP files and a website CSV in one batch: source 'mixed', 3 variants ({r.status_code})")
    check(body.get("market", {}).get("joined") is False, "no pool here: the join reports itself not joined")

    # /parse is the batch path with one file: today's payload plus `sweep: null`.
    r1 = c.post("/api/oo-backtest/parse", files={"file": ("site.csv", csv())})
    b1 = r1.json()
    old = R._parse(csv(), "site.csv")
    same = {k: v for k, v in b1.items() if k not in ("market", "sweep")} == \
        {k: v for k, v in json.loads(json.dumps(old)).items() if k != "market"}
    check(r1.status_code == 200 and b1.get("sweep") is None and same,
          "/parse with one plain file: the same payload as before, plus sweep: null")
    r1 = c.post("/api/oo-backtest/parse", files={"file": ("mcp.csv", csv(extra={"param_pt": [25, 50, 50]}))})
    check(r1.status_code == 200 and len((r1.json().get("sweep") or {}).get("variants", [])) == 2,
          "/parse with one MCP file holding two variants: a sweep")

    r = post([("s_dte45_pt50.csv", csv()), ("s_dte50_pt50.csv", csv()), ("s_dte45_sl1.csv", csv())])
    check(r.status_code == 400 and "s_dte45_sl1.csv" in r.json().get("detail", "") and "Nothing was loaded" in r.json()["detail"],
          f"a mismatched batch is a 400 naming the file ({r.json().get('detail')!r})")
    r = post([("s_dte45.csv", csv()), ("s_dte50.csv", b"foo,bar\n1,2\n")])
    check(r.status_code == 400 and r.json()["detail"].startswith("s_dte50.csv:") and "Nothing was loaded" in r.json()["detail"],
          f"a file that does not parse is named ({r.json().get('detail')!r})")
    r = post([("s_dte45.csv", gzip.compress(csv())[:-8] + b"garbage!")])
    check(r.status_code == 400, f"a corrupt gzip part is refused ({r.status_code})")
    keep = sweep.MAX_BATCH_BYTES
    try:
        sweep.MAX_BATCH_BYTES = len(csv()) * 2 + 10
        r = post([(f"s_dte{d}.csv", gzip.compress(csv())) for d in (1, 2, 3)])
        check(r.status_code == 400 and "s_dte3.csv" in r.json()["detail"] and "MB uncompressed" in r.json()["detail"],
              f"over the size cap: refused, naming where it was reached ({r.json().get('detail')!r})")
    finally:
        sweep.MAX_BATCH_BYTES = keep

    # The encoder: byte for byte what JSONResponse(jsonable_encoder(x)) sends.
    sample = {"d": date(2026, 4, 8), "dt": datetime(2026, 4, 8, 9, 30), "t": time(15, 55), "s": "é ü", "n": None,
              "f": 1.5, "l": [1, 2.25, "x"], "nested": {"missed": [date(2026, 1, 2)]}}
    from fastapi.responses import JSONResponse
    sample.update({"tiny": 1e-05, "huge": 1.5e17, "neg0": -0.0, "third": 1 / 3, "ints": {1: "a", 2: [3, 4.0]}})
    want_s, want_b = JSONResponse(jsonable_encoder(sample)).body, JSONResponse(jsonable_encoder(body)).body
    # Both encoders: orjson when installed, and the stdlib fallback it
    # replaces when it is not. The same VALUES always; bytes noted.
    keep = R._orjson
    try:
        for name, mod in (("orjson", keep), ("stdlib fallback", None)):
            if name == "orjson" and mod is None:
                print("  note  orjson is not installed here: only the stdlib fallback is checked")
                continue
            R._orjson = mod
            got_s, got_b = R.encode_json(sample), R.encode_json(body)
            check(json.loads(got_s) == json.loads(want_s) and json.loads(got_b) == json.loads(want_b),
                  f"{name}: the same JSON values as JSONResponse(jsonable_encoder(...)) -- dates, times, unicode, "
                  f"int keys, tiny/huge floats, -0.0, a whole sweep payload")
            check(got_b == want_b, f"{name}: byte-identical for a whole sweep payload")
    finally:
        R._orjson = keep


def check_rank_memory() -> None:
    """The surface ranking at sweep scale: the same numbers as the direct
    matrix, in a fraction of its memory."""
    print("surface ranking at sweep scale")
    import time as _t
    import tracemalloc

    import numpy as np

    from app.oo_backtest import surface_stats as ss
    rng = np.random.default_rng(3)
    D, C = 300, 40
    distinct = rng.normal(size=(D, C))
    distinct[:120, ::3] = np.nan                       # late coverage: a second null pattern
    distinct[0] = np.nan                               # the "no bar" row
    distinct[:, 7] = 0.5                               # constant
    gather = rng.integers(0, D, size=6000)
    pnl = rng.normal(30, 400, size=6000)
    pnl[::97] = np.nan
    bars = gather.copy()
    want = ss.correlations(distinct[gather], pnl, bars)
    keep = ss.BLOCK_CELLS
    try:
        ss.BLOCK_CELLS = 6000 * 3                     # three columns a block: chunking exercised
        got = ss.correlations_gathered(distinct, gather, pnl, bars)
    finally:
        ss.BLOCK_CELLS = keep
    same = all(np.array_equal(got[f], want[f]) for f in ("n", "bars"))
    close = all(np.allclose(got[f], want[f], rtol=1e-12, atol=1e-15, equal_nan=True)
                for f in ("pearson", "pearson_p", "spearman", "spearman_p"))
    check(same and close, "gathered + chunked == the full matrix (n, bars exact; r, p within 1e-12)")
    one = ss.correlations_gathered(distinct, gather, pnl, bars)
    check(all(np.array_equal(one[f], want[f], equal_nan=True) for f in want),
          "below the block size: bit-identical to the full-matrix path")

    # 240,000 trades over 2,500 distinct bars x 452 metrics.
    distinct = rng.normal(size=(2500, 452))
    distinct[:900, 200:] = np.nan
    distinct[0] = np.nan
    gather = rng.integers(0, 2500, size=240_000)
    pnl = rng.normal(30, 400, size=240_000)
    tracemalloc.start()
    t0 = _t.monotonic()
    ss.correlations_gathered(distinct, gather, pnl, gather)
    took = _t.monotonic() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    full = 240_000 * 452 * 8
    check(peak < 300e6, f"240,000 trades x 452 metrics: peak {peak / 1e6:.0f} MB, against {full / 1e6:.0f} MB for the "
                        f"gathered matrix alone ({took:.1f}s)")


def main() -> int:
    check_filenames()
    check_combine()
    check_ignored()
    check_mcp()
    check_endpoints()
    check_rank_memory()
    print()
    if FAILS:
        print(f"FAIL: {len(FAILS)} check(s) failed")
        return 1
    print("PASS: OO sweeps — filenames, precedence, refusals, isIgnored, MCP format, variant payload, batch endpoints")
    return 0


if __name__ == "__main__":
    sys.exit(main())
