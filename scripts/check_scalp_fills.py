"""The Schwab statement parser, against statements built to break it.

WHY THIS IS ITS OWN SCRIPT. app/scalp_fills.py touches no database and no
filesystem, so it can be tested exhaustively without either — and it is the
one piece of this project where a quiet mistake becomes a wrong number rather
than an empty panel. Every downstream statistic, including the calibration
that decides which metric drives the ranking, is computed from what this
returns.

THE ASSERTIONS ARE WRITTEN TO FAIL IF THE CODE WERE WRONG, not to confirm the
output looks right. Two earlier checks in this project passed while a bug was
present because the fixture could not produce the failure: an odd-lot
assertion whose data could not violate it, and a self-inclusion check
comparing a median against one value it could not move. So each case here
states the WRONG answer it is guarding against, and several assert on
arithmetic computed by hand rather than on whatever the parser happens to
return.
"""
from __future__ import annotations

import sys
from datetime import date, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app import scalp_fills as F                          # noqa: E402

HEAD = ("Transactions  for account XXXX-1234\n"
        "\n"
        "Date,Time,Type,REF #,Description,Misc Fees,Amount\n")


def stmt(*lines: str) -> bytes:
    return (HEAD + "\n".join(lines) + "\n").encode()


def parse(*lines: str):
    return F.parse_statement(stmt(*lines), "statement.csv")


FAILS: list[str] = []


def check(cond, msg):
    if not cond:
        FAILS.append(msg)


# ── one long round trip, arithmetic done by hand ────────────────────────────
def case_simple_long():
    trips, rep = parse(
        "08/26/2026,09:41:02,TRD,101,BOT +50 FDX @334.47,,-16723.50",
        "08/26/2026,09:41:11,TRD,102,SOLD -50 FDX @334.55,0.02,16727.48",
    )
    check(len(trips) == 1, f"one buy and one sell made {len(trips)} trips")
    if not trips:
        return
    t = trips[0]
    # 50 x (334.55 - 334.47) = 4.00, less 0.02 in fees.
    check(abs(t.net_pnl - 3.98) < 1e-9,
          f"P&L is {t.net_pnl}, not 3.98 — 50 x 0.08 less 0.02 of fees")
    check(t.is_long, "a BOT-then-SOLD trip was not recorded as long")
    check(t.peak_shares == 50, f"peak shares is {t.peak_shares}, not 50")
    check(abs(t.entry_price - 334.47) < 1e-9, "entry price is not the buy price")
    check(abs(t.capital - 50 * 334.47) < 1e-6, "capital is not shares x entry")
    check(t.duration_s == 9, f"duration is {t.duration_s}s, not 9")
    check(t.legs == 2, "leg count is wrong")


# ── a short trip: sold first, bought back ───────────────────────────────────
def case_short():
    trips, _ = parse(
        "08/26/2026,11:15:00,TRD,106,SOLD -30 LII @512.00,0.01,15359.99",
        "08/26/2026,11:16:30,TRD,107,BOT +30 LII @511.80,,-15354.00",
    )
    check(len(trips) == 1, "a short round trip did not close")
    if not trips:
        return
    t = trips[0]
    check(not t.is_long, "a SOLD-then-BOT trip was recorded as LONG. The sign "
                         "convention is inverted and every short's P&L will "
                         "carry the wrong sign.")
    # 30 x (512.00 - 511.80) = 6.00, less 0.01.
    check(abs(t.net_pnl - 5.99) < 1e-9,
          f"short P&L is {t.net_pnl}, not 5.99 — a short that falls 20c makes "
          f"money, and a sign error here reads as a loss")
    check(abs(t.entry_price - 512.00) < 1e-9,
          "a short's entry price should be where it was SOLD")


# ── partial fills: several legs, one excursion ──────────────────────────────
def case_multi_leg():
    trips, _ = parse(
        "08/26/2026,10:02:00,TRD,103,BOT +40 FDX @333.10,,-13324.00",
        "08/26/2026,10:02:07,TRD,104,BOT +40 FDX @333.05,,-13322.00",
        "08/26/2026,10:02:19,TRD,105,SOLD -80 FDX @333.20,0.03,26655.97",
    )
    check(len(trips) == 1,
          f"three legs of one excursion became {len(trips)} trips — a round "
          f"trip is a run between flat points, not a pairing of fills")
    if not trips:
        return
    t = trips[0]
    check(abs(t.net_pnl - 9.97) < 1e-9, f"P&L is {t.net_pnl}, not 9.97")
    check(t.peak_shares == 80,
          f"peak shares is {t.peak_shares}, not 80 — capital at risk is the "
          f"largest position reached, not the size of any one leg")
    check(abs(t.entry_price - 333.075) < 1e-9,
          "entry price is not the share-weighted average of the opening legs")
    check(t.legs == 3, "leg count is wrong")


# ── THE ONE THAT MATTERS: a position still open at the end ──────────────────
def case_unclosed():
    trips, rep = parse(
        "08/26/2026,09:41:02,TRD,101,BOT +50 FDX @334.47,,-16723.50",
        "08/26/2026,09:41:11,TRD,102,SOLD -50 FDX @334.55,0.02,16727.48",
        "08/26/2026,15:59:00,TRD,108,BOT +25 DLTR @71.10,,-1777.50",
    )
    check(len(trips) == 1,
          f"{len(trips)} trips from one closed excursion and one open "
          f"position. An open position must NOT be completed at the last "
          f"price — that is what corrupted a session's statistics before.")
    check(len(rep.unclosed) == 1,
          "the open DLTR position was not reported. Silence here is the exact "
          "failure this parser exists to prevent.")
    if rep.unclosed:
        u = rep.unclosed[0]
        check(u["symbol"] == "DLTR" and u["shares"] == 25,
              f"the unclosed report is wrong: {u}")
    check(all(t.symbol != "DLTR" for t in trips),
          "an unclosed position was written as a completed trip")


# ── rows that are not executions ────────────────────────────────────────────
def case_non_trades():
    trips, rep = parse(
        "08/26/2026,09:41:02,TRD,101,BOT +50 FDX @334.47,,-16723.50",
        "08/26/2026,09:41:11,TRD,102,SOLD -50 FDX @334.55,0.02,16727.48",
        "08/26/2026,,DIV,,ORDINARY DIVIDEND FDX,,12.00",
        "08/26/2026,,JRN,,JOURNAL TO BROKERAGE,,0.00",
        ",,,,,,",
    )
    check(len(trips) == 1,
          "a dividend or journal row became a trade. Filtering on the TRD "
          "type is what keeps a description containing a ticker from parsing "
          "as an execution.")
    check(not rep.rows_unparsed,
          f"non-trade rows were reported as unparsed: {rep.rows_unparsed}")


# ── a trade row the parser cannot read ──────────────────────────────────────
def case_unparsed_is_reported():
    trips, rep = parse(
        "08/26/2026,09:41:02,TRD,101,BOT +50 FDX @334.47,,-16723.50",
        "08/26/2026,09:41:11,TRD,102,SOLD -50 FDX @334.55,0.02,16727.48",
        "08/26/2026,10:00:00,TRD,109,EXCHANGE OR EXERCISE SOMETHING ODD,,0.00",
    )
    check(len(rep.rows_unparsed) == 1,
          "an unreadable TRD row was skipped silently. 'Twelve rows did not "
          "parse' is unactionable, and nothing at all is worse.")
    if rep.rows_unparsed:
        u = rep.rows_unparsed[0]
        check("line" in u and u["line"] > 0,
              "the unparsed row has no line number, so it cannot be found")
        check(u["text"], "the unparsed row's text was not kept")


# ── a single execution carrying the position through zero ───────────────────
def case_reversal():
    trips, rep = parse(
        "08/26/2026,09:00:00,TRD,201,BOT +50 XYZ @10.00,,-500.00",
        "08/26/2026,09:00:30,TRD,202,SOLD -80 XYZ @10.10,0.01,808.00",
        "08/26/2026,09:01:00,TRD,203,BOT +30 XYZ @10.05,,-301.50",
    )
    check(len(trips) == 2,
          f"a reversal made {len(trips)} trips, not 2 — the leg through zero "
          f"has to be split so the closing part belongs to the trip it closes")
    check(rep.reversals,
          "an execution carried the position through zero and nothing said "
          "so. In a one-position-at-a-time strategy that is worth seeing.")
    if len(trips) == 2:
        # Long 50 closed at 10.10: 50 x 0.10 = 5.00, less the 0.01 fee.
        check(abs(trips[0].net_pnl - 4.99) < 1e-9,
              f"the closing half is {trips[0].net_pnl}, not 4.99")
        check(trips[0].is_long and not trips[1].is_long,
              "the two halves of a reversal are not long-then-short")
        # Short 30 at 10.10 bought back at 10.05: 30 x 0.05 = 1.50.
        check(abs(trips[1].net_pnl - 1.50) < 1e-9,
              f"the opened half is {trips[1].net_pnl}, not 1.50")


# ── attention minutes is WALL CLOCK, not summed holds ───────────────────────
def case_attention_is_wall_clock():
    trips, _ = parse(
        "08/26/2026,09:30:00,TRD,301,BOT +10 LII @500.00,,-5000.00",
        "08/26/2026,09:30:08,TRD,302,SOLD -10 LII @500.10,0.01,5000.99",
        "08/26/2026,11:06:00,TRD,303,BOT +10 LII @501.00,,-5010.00",
        "08/26/2026,11:06:08,TRD,304,SOLD -10 LII @501.10,0.01,5010.99",
    )
    rows = F.daily_rows(trips)
    check(len(rows) == 1, "two trips in one name became more than one row")
    if not rows:
        return
    r = rows[0]
    # 09:30:00 -> 11:06:08 is 96.13 minutes. Summed holds would be 16 seconds,
    # which would rate this name ~360x better.
    check(abs(r["attention_minutes"] - 96.1333) < 0.01,
          f"attention is {r['attention_minutes']:.2f} minutes. It must be wall "
          f"clock from first entry to last exit — summing hold durations "
          f"gives 0.27 minutes here and rates a name that consumed an hour "
          f"and a half as though the waiting were free.")
    check(r["trips"] == 2 and abs(r["net_pnl"] - 1.98) < 1e-9,
          f"daily aggregate is wrong: {r['trips']} trips, {r['net_pnl']}")
    check(abs(r["dollars_per_min"] - 1.98 / 96.1333) < 1e-6,
          "$/min is not net P&L over attention minutes")


# ── a header that is not the first row ──────────────────────────────────────
def case_header_search():
    data = ("My Account Transactions\n\n\n"
            "Date,Time,Type,REF #,Description,Misc Fees,Amount\n"
            "08/26/2026,09:41:02,TRD,101,BOT +50 FDX @334.47,,-16723.50\n"
            "08/26/2026,09:41:11,TRD,102,SOLD -50 FDX @334.55,0.02,16727.48\n"
            ).encode()
    trips, _ = F.parse_statement(data, "s.csv")
    check(len(trips) == 1,
          "the header was not found past the title rows. Assuming row 0 fails "
          "silently — the title becomes the column names and no trades are "
          "found at all.")


# ── a file that is not a statement ──────────────────────────────────────────
def case_refuses_garbage():
    try:
        F.parse_statement(b"hello,world\n1,2\n", "s.csv")
    except F.ParseError as exc:
        check("date" in str(exc).lower() or "header" in str(exc).lower(),
              "the refusal does not say what it was looking for")
        return
    FAILS.append("a file with no statement columns parsed without complaint")


def case_empty():
    for data, name in ((b"", "s.csv"), (b"\n\n", "s.csv")):
        try:
            F.parse_statement(data, name)
        except F.ParseError:
            continue
        FAILS.append(f"an empty file ({name}) parsed without complaint")


# ── xlsx, if openpyxl is installed ──────────────────────────────────────────
def case_xlsx():
    try:
        from openpyxl import Workbook
    except ImportError:
        print("  SKIP xlsx: openpyxl not installed")
        return
    import io
    wb = Workbook()
    ws = wb.active
    ws.append(["Transactions for account XXXX"])
    ws.append(["Date", "Time", "Type", "REF #", "Description", "Misc Fees"])
    ws.append([date(2026, 8, 26), datetime(2026, 8, 26, 9, 41, 2), "TRD", 101,
               "BOT +50 FDX @334.47", None])
    ws.append([date(2026, 8, 26), datetime(2026, 8, 26, 9, 41, 11), "TRD", 102,
               "SOLD -50 FDX @334.55", 0.02])
    buf = io.BytesIO()
    wb.save(buf)
    trips, _ = F.parse_statement(buf.getvalue(), "s.xlsx")
    check(len(trips) == 1, "the xlsx path did not produce a trip")
    if trips:
        check(abs(trips[0].net_pnl - 3.98) < 1e-9,
              "xlsx and csv disagree on P&L — the cells arrive as real dates "
              "and floats rather than strings, and one of the two paths is "
              "coercing them differently")


CASES = [
    ("simple long",            case_simple_long),
    ("short trip",             case_short),
    ("multi-leg excursion",    case_multi_leg),
    ("UNCLOSED position",      case_unclosed),
    ("non-trade rows",         case_non_trades),
    ("unparsed row reported",  case_unparsed_is_reported),
    ("reversal through zero",  case_reversal),
    ("attention = wall clock", case_attention_is_wall_clock),
    ("header past the title",  case_header_search),
    ("refuses a non-statement", case_refuses_garbage),
    ("refuses an empty file",  case_empty),
    ("xlsx matches csv",       case_xlsx),
]


def main() -> int:
    for name, fn in CASES:
        before = len(FAILS)
        try:
            fn()
        except Exception as exc:
            FAILS.append(f"{name}: raised {type(exc).__name__}: {exc}")
        if len(FAILS) > before:
            for m in FAILS[before:]:
                print(f"  FAIL {name}: {m}")
    print(f"\nparser cases: {len(CASES)}, failures: {len(FAILS)}")
    return 1 if FAILS else 0


sys.exit(main())
