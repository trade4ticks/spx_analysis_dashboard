#!/usr/bin/env python3
"""Read-only audit: which saved signals were selected on which bin table?

READ THIS BEFORE READING THE OUTPUT
-----------------------------------
The obvious test -- recompute each signal's stats on is_bins and see whether
they match what is stored -- CANNOT answer the provenance question. Both
paths that ever write signal stats (POST /signals and the recall update at
oi_analysis.py:9708) call the same _compute_signal_stats(), and that function
hardcodes `JOIN is_bins` (oi_analysis.py:8984). Stored stats are is_bins-
derived for EVERY signal, whatever view it was saved from. A TT-selected
signal has IS-computed stats sitting on it and reconciles perfectly.

So this script runs two independent checks, and only the second speaks to
provenance.

CHECK A -- RECONCILIATION (does NOT prove provenance)
    Stored stats vs an is_bins recompute today. A mismatch means the stored
    numbers are stale: is_bins was rebuilt after the signal was saved, or the
    signal predates a change to the stats path. Worth knowing on its own --
    stale stats are wrong numbers on the signals list -- but a PASS here says
    nothing about which grid the zone was drawn on.

CHECK B -- CELL OCCUPANCY (this is the provenance evidence)
    A person selects cells they can see have data in them. So for every
    selected cell, count the rows it would contain under each bin table:

        is_bins            full-history in-sample edges
        tt_bins (train)    frozen edges, trade_date <  cutoff
        tt_bins (test)     frozen edges, trade_date >= cutoff
        wf_bins            walk-forward edges

    A cell that is EMPTY under is_bins but POPULATED under tt_bins-train is
    a cell nobody would have picked off an IS heatmap, because on that grid
    it was blank. Signals containing such cells were drawn on a non-IS grid.

    The reverse is weaker evidence and is reported but not treated as proof:
    a zone can legitimately contain a cell that is empty in one window.

WHAT A CLEAN RESULT MEANS
    No signal has an is_bins-empty / tt_bins-populated cell => nothing
    contradicts the assumption that every signal is IS-selected. That is
    absence of evidence, not proof: a TT-drawn zone whose cells happen to be
    populated under both grids is indistinguishable after the fact, because
    the save path recorded no mode. The honest ceiling on this audit is
    "no signal is provably non-IS", and the fix for the ambiguity is the
    selection_mode column going forward, not more forensics.

Usage (VPS, project root, venv active):
    python scripts/audit_signal_provenance.py
    python scripts/audit_signal_provenance.py --verbose   # per-cell detail

Writes nothing. Exit 1 if any signal is provably non-IS-selected.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(_ROOT / ".env")

import asyncpg  # noqa: E402

_SAFE = set("abcdefghijklmnopqrstuvwxyz_0123456789")
_GAP = "overnight_gap"
_TOL = 2e-6          # stored stats are rounded to 6 dp


def _safe(*names: str) -> bool:
    return all(n and all(c in _SAFE for c in n) for n in names)


def _outcome_sql(outcome: str) -> tuple[str, str]:
    if outcome == _GAP:
        return ("AVG(df.ret_1d_fwd_cc - df.ret_1d_fwd_oc)",
                "df.ret_1d_fwd_cc IS NOT NULL AND df.ret_1d_fwd_oc IS NOT NULL")
    return f"AVG(df.{outcome})", f"df.{outcome} IS NOT NULL"


async def _cols_of(conn, table: str) -> set:
    rows = await conn.fetch(
        """SELECT column_name FROM information_schema.columns
           WHERE table_name = $1 AND table_schema = 'public'
             AND column_name LIKE 'bin20_%'""", table)
    return {r["column_name"] for r in rows}


async def _cell_stats(conn, table, prim, sec, outcome, n_bins, xs, ys,
                      window, cutoff):
    """Per-cell (n, avg_ret) for one signal against one bin table + window.

    Mirrors _compute_signal_stats' formula exactly:
        cell_idx = ((bin20 - 1) * n_bins) / 20      (integer division)
    """
    expr, filt = _outcome_sql(outcome)
    params: list = [n_bins, xs, ys]
    date_sql = ""
    if window == "train":
        params.append(cutoff); date_sql = f" AND df.trade_date <  ${len(params)}"
    elif window == "test":
        params.append(cutoff); date_sql = f" AND df.trade_date >= ${len(params)}"
    sql = f"""
        SELECT ((b.bin20_{prim} - 1) * $1::int) / 20 AS ix,
               ((b.bin20_{sec}  - 1) * $1::int) / 20 AS iy,
               {expr}::float8 AS avg_ret,
               COUNT(*) AS n
        FROM daily_features df
        JOIN {table} b USING (ticker, trade_date)
        WHERE b.bin20_{prim} > 0 AND b.bin20_{sec} > 0
          AND {filt}
          AND (((b.bin20_{prim} - 1) * $1::int) / 20,
               ((b.bin20_{sec}  - 1) * $1::int) / 20)
              IN (SELECT * FROM unnest($2::int[], $3::int[]))
          {date_sql}
        GROUP BY ix, iy
    """
    out = {}
    for r in await conn.fetch(sql, *params):
        out[(int(r["ix"]), int(r["iy"]))] = (
            int(r["n"]),
            float(r["avg_ret"]) if r["avg_ret"] is not None else None)
    return out


def _agg(per_cell: dict) -> tuple[int, float | None]:
    tot = sum(n for n, _ in per_cell.values())
    if not tot:
        return 0, None
    wsum = sum((a or 0.0) * n for n, a in per_cell.values())
    return tot, round(wsum / tot, 6)


async def run(args) -> None:
    dsn = os.getenv("OI_DATABASE_URL")
    if not dsn:
        print("ERROR: OI_DATABASE_URL not set.")
        sys.exit(1)
    conn = await asyncpg.connect(dsn=dsn)
    try:
        cutoff = await conn.fetchval("SELECT MAX(cutoff_date) FROM tt_bins")
        have = {t: await _cols_of(conn, t)
                for t in ("is_bins", "tt_bins", "wf_bins")}
        print(f"tt_bins frozen cutoff : {cutoff}")
        print(f"bin20 columns         : "
              + "  ".join(f"{t}={len(c)}" for t, c in have.items()))

        sigs = await conn.fetch(
            """SELECT id, name, primary_metric, secondary_metric, outcome,
                      n_bins, cell_set, agg_avg_ret, agg_n, per_cell_stats,
                      created_at, stats_updated_at
               FROM signals ORDER BY id""")
        print(f"signals               : {len(sigs)}\n")
        if not sigs:
            print("Nothing to audit.")
            return

        recon_ok = recon_bad = recon_skip = 0
        provably_non_is: list = []
        ambiguous: list = []
        rows_out: list = []

        for s in sigs:
            prim, sec, outc = (s["primary_metric"], s["secondary_metric"],
                               s["outcome"])
            n_bins = int(s["n_bins"])
            raw = s["cell_set"] or "[]"
            cells = json.loads(raw) if isinstance(raw, str) else raw
            label = f"#{s['id']} {s['name'][:26]}"

            if not cells or not _safe(prim, sec) or (
                    outc != _GAP and not _safe(outc)):
                rows_out.append((label, "SKIP", "unsafe identifier or no cells",
                                 "", ""))
                recon_skip += 1
                continue
            xs = [int(c[0]) for c in cells]
            ys = [int(c[1]) for c in cells]
            want = {(int(c[0]), int(c[1])) for c in cells}

            missing = [t for t in ("is_bins", "tt_bins")
                       if f"bin20_{prim}" not in have[t]
                       or f"bin20_{sec}" not in have[t]]
            if missing:
                rows_out.append((label, "SKIP",
                                 f"metric absent from {','.join(missing)}", "", ""))
                recon_skip += 1
                continue

            # ── CHECK A: reconciliation against is_bins ──────────────────
            is_pc = await _cell_stats(conn, "is_bins", prim, sec, outc,
                                      n_bins, xs, ys, "all", cutoff)
            is_n, is_avg = _agg(is_pc)
            st_n = int(s["agg_n"] or 0)
            st_avg = float(s["agg_avg_ret"]) if s["agg_avg_ret"] is not None else None
            if s["agg_n"] is None and s["agg_avg_ret"] is None:
                recon = "NO-STATS"
                recon_skip += 1
            elif st_n == is_n and (
                    (st_avg is None and is_avg is None)
                    or (st_avg is not None and is_avg is not None
                        and abs(st_avg - is_avg) < _TOL)):
                recon = "match"
                recon_ok += 1
            else:
                recon = f"DRIFT stored n={st_n} vs is_bins n={is_n}"
                recon_bad += 1

            # ── CHECK B: cell occupancy across bin tables ────────────────
            tt_tr = await _cell_stats(conn, "tt_bins", prim, sec, outc,
                                      n_bins, xs, ys, "train", cutoff)
            tt_te = await _cell_stats(conn, "tt_bins", prim, sec, outc,
                                      n_bins, xs, ys, "test", cutoff)
            wf_pc = {}
            if (f"bin20_{prim}" in have["wf_bins"]
                    and f"bin20_{sec}" in have["wf_bins"]):
                wf_pc = await _cell_stats(conn, "wf_bins", prim, sec, outc,
                                          n_bins, xs, ys, "all", cutoff)

            # A cell nobody could have picked off an IS grid: blank there,
            # populated on the frozen train grid.
            smoking = sorted(c for c in want
                             if is_pc.get(c, (0,))[0] == 0
                             and tt_tr.get(c, (0,))[0] > 0)
            # Weaker, reported only: populated in IS, blank on the TT train
            # grid. Consistent with an IS selection; not evidence either way.
            reverse = sorted(c for c in want
                             if is_pc.get(c, (0,))[0] > 0
                             and tt_tr.get(c, (0,))[0] == 0)

            if smoking:
                verdict = f"NON-IS ({len(smoking)} cell(s) blank in is_bins)"
                provably_non_is.append((s["id"], s["name"], smoking))
            else:
                verdict = "consistent with IS"
                ambiguous.append(s["id"])

            if args.explain_drift and recon.startswith("DRIFT"):
                # WHY a signal drifts has two very different answers.
                #
                # is_bins is IN-SAMPLE: its edges are full-history quantiles,
                # so a rebuild does not just append new dates -- it RE-LABELS
                # every historical row against moved boundaries. A zone can
                # therefore lose rows it used to contain.
                #
                # Split the fresh count at the horizon that existed when the
                # stats were last written:
                #   history part != stored  -> edges moved, rows re-labelled
                #   history part == stored  -> pure append, only new dates
                horizon = s["stats_updated_at"] or s["created_at"]
                hz = horizon.date() if hasattr(horizon, "date") else horizon
                hist = await _cell_stats(conn, "is_bins", prim, sec, outc,
                                         n_bins, xs, ys, "train", hz)
                hist_n, _ = _agg(hist)
                new_n = is_n - hist_n
                relabel = hist_n - st_n
                print(f"  -- {label} drift breakdown (horizon {hz})")
                print(f"     stored at save time            : {st_n:>8,}")
                print(f"     is_bins today, dates <= horizon: {hist_n:>8,}"
                      f"   -> re-labelled {relabel:+,}")
                print(f"     is_bins today, dates >  horizon: {new_n:>8,}"
                      f"   -> genuinely new")
                print(f"     is_bins today, total           : {is_n:>8,}"
                      f"   -> net {is_n - st_n:+,}")
                if relabel:
                    print("     => edges MOVED: a rebuild re-binned historical")
                    print("        rows, so this zone does not mean today what")
                    print("        it meant when it was saved.")
                else:
                    print("     => pure append: the zone is unchanged, it just")
                    print("        has more dates in it.")

            occ = (f"is={sum(n for n,_ in is_pc.values())} "
                   f"ttTr={sum(n for n,_ in tt_tr.values())} "
                   f"ttTe={sum(n for n,_ in tt_te.values())} "
                   f"wf={sum(n for n,_ in wf_pc.values())}")
            rows_out.append((label, recon, verdict, occ,
                             f"{len(want)} cells, rev={len(reverse)}"))

            if args.verbose and (smoking or recon.startswith("DRIFT")):
                print(f"  -- {label} detail")
                for c in sorted(want):
                    print(f"     cell {c}: is={is_pc.get(c,(0,None))[0]:>6}  "
                          f"ttTrain={tt_tr.get(c,(0,None))[0]:>6}  "
                          f"ttTest={tt_te.get(c,(0,None))[0]:>6}  "
                          f"wf={wf_pc.get(c,(0,None))[0]:>6}")

        print("=" * 108)
        print(f"{'signal':<32} {'CHECK A recon':<34} {'CHECK B provenance':<34} rows")
        print("-" * 108)
        for label, recon, verdict, occ, extra in rows_out:
            print(f"{label:<32} {recon:<34} {verdict:<34} {occ}")

        print("\n" + "=" * 108)
        print("SUMMARY")
        print("=" * 108)
        print(f"  CHECK A  stored stats reconcile with is_bins : {recon_ok}")
        print(f"           stale / drifted                     : {recon_bad}")
        print(f"           skipped (no stats, unsafe, missing) : {recon_skip}")
        print(f"  CHECK B  provably NON-IS selected            : {len(provably_non_is)}")
        print(f"           consistent with IS                  : {len(ambiguous)}")

        if provably_non_is:
            print("\n  Provably non-IS signals — these contain cells that are")
            print("  blank under is_bins, so the zone cannot have been drawn")
            print("  on an IS heatmap:")
            for sid, name, cells in provably_non_is:
                print(f"    #{sid} {name}: cells {cells[:6]}"
                      + (" ..." if len(cells) > 6 else ""))
            print("\n  A blanket backfill to selection_mode='in_sample' would")
            print("  mislabel these.")
        else:
            print("\n  No signal is provably non-IS.")
            print("  This is absence of evidence, not proof: the save path")
            print("  recorded no mode, so a TT-drawn zone whose cells are")
            print("  populated under both grids leaves no trace. Treat it as")
            print("  'nothing contradicts the IS assumption'.")

        if recon_bad:
            print(f"\n  {recon_bad} signal(s) have stale stored stats. That is")
            print("  independent of provenance -- is_bins was likely rebuilt")
            print("  after they were saved. Their displayed numbers are wrong")
            print("  today and a stats refresh would fix them.")

        sys.exit(1 if provably_non_is else 0)
    finally:
        await conn.close()


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--verbose", action="store_true",
                    help="Per-cell occupancy for flagged signals")
    ap.add_argument("--explain-drift", action="store_true",
                    help="Split CHECK A's delta into new dates vs re-labelled "
                         "history, which separates the two causes of drift")
    asyncio.run(run(ap.parse_args()))


if __name__ == "__main__":
    main()
