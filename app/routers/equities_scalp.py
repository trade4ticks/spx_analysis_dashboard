"""Equities Scalp — read-only endpoints over the `equities_scalp` database.

WHAT THIS OWNS AND WHAT IT DOES NOT. The pipeline in the Open_Interest repo's
`scalp/` computes ~75 metrics per symbol-day and per 15-minute bucket and
writes them here. This module READS them. Nothing on this page writes to
universe, daily_metrics, intraday_metrics, provenance or rankings; the only
writes this project will ever make are the fills tables, which the pipeline
does not own.

NO HARDCODED METRIC NAMES. The metric set is explicitly unsettled -- five
noise variants at three horizons at five statistics, plus flicker and flow
metrics, and a calibration process whose purpose is to delete most of them. A
column list in this file would mean editing this project every time the
pipeline changed one, and the failure mode is silent: a renamed metric becomes
a column of nulls, not an error.

So the catalog is READ from the database. `/meta` returns what
`daily_metrics.metric` actually contains on the latest date, decorated with
the definitions from the vendored metric_docs. scripts/check_scalp_metrics.py
fails the build if a metric-name literal appears in this file or the page's
JS.

LONG, NOT WIDE. daily_metrics is (trade_date, symbol, metric, value), so every
read pivots. That is the pipeline's deliberate choice -- see scalp/db.py -- and
the pivot happens here rather than being pushed onto the client, which would
ship five long rows to draw one wide one.
"""
from __future__ import annotations

import datetime as _dt
import os
import re

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile

from app.db import get_scalp_pool, pool_status
from app import scalp_config, scalp_columns, scalp_fills, scalp_metric_docs

router = APIRouter()


# The pipeline's own read-time thresholds and their slider bounds. Vendored
# rather than restated: scripts/check_vendored.py diffs the whole file against
# the pipeline's copy, so a threshold moved upstream cannot sit here at its old
# value looking authoritative.
FILTER_KEYS = tuple(scalp_config.DEFAULT_FILTERS)


# ── the shape of a metric name ───────────────────────────────────────────────
#
# The generated families are `<kind>_<variant>_<horizon>s_<statistic>`, e.g.
# noise_bps_tw_mid_10s_rms. This parses that SHAPE without naming any
# particular variant, horizon or statistic -- the point is to discover which
# ones the pipeline currently emits, not to assert which ones exist.
_VARIANT_RE = re.compile(
    r"^(?P<kind>noise_bps|ratio)_(?P<variant>.+?)_(?P<horizon>\d+)s"
    r"(?:_(?P<stat>[a-z0-9]+))?$"
)


def _parse_variant(metric: str) -> dict | None:
    m = _VARIANT_RE.match(metric)
    if not m:
        return None
    return {"kind": m.group("kind"), "variant": m.group("variant"),
            "horizon_s": int(m.group("horizon")), "statistic": m.group("stat")}


def _catalog_entry(metric: str) -> dict:
    """One metric, with its documentation link and its parsed shape."""
    link = scalp_metric_docs.header_link(metric)
    return {
        "metric":  metric,
        "label":   link.get("label", metric),
        "tooltip": link.get("tooltip"),
        "href":    link.get("href"),
        "section": link.get("section"),
        # None for a metric that is not one of the generated families, which
        # is most of the flow and flicker set.
        "variant": _parse_variant(metric),
    }


# ── dates cross the boundary as strings and must not stay that way ──────────
#
# asyncpg binds by TYPE, not by content: a DATE column needs a datetime.date
# and it will not coerce a string, unlike psycopg2. The error it raises names
# neither the column nor the endpoint --
#
#     invalid input for query argument $1: '2026-08-28'
#     ('str' object has no attribute 'toordinal')
#
# -- and it happens at bind time, so it survives every check that stops at
# whether the SQL parses.
#
# The trap here is specific and will recur through P2-P5: `dates` is stringified
# for the JSON response, and the natural next line picks the anchor out of that
# already-stringified list. The value is a string by the time it is bound, and
# nothing in between looks wrong.
#
# So dates are kept as date objects for the whole of their life inside a
# handler, and stringified once, at the response. Anything that arrives from a
# query parameter goes through _as_date first.


def _as_date(value, field: str = "date"):
    """A query parameter to a datetime.date, or a 400 that says which field.

    None passes through: an absent date means "the latest", which the caller
    resolves. A date object passes through unchanged so this is safe to apply
    twice.
    """
    if value is None or isinstance(value, _dt.date) and not isinstance(value, _dt.datetime):
        return value
    if isinstance(value, _dt.datetime):
        return value.date()
    try:
        return _dt.date.fromisoformat(str(value))
    except (TypeError, ValueError):
        raise HTTPException(
            400, f"{field}={value!r} is not an ISO date (YYYY-MM-DD). "
                 f"asyncpg binds a DATE column by type and will not coerce a "
                 f"string, so this has to be resolved before the query.")


def _no_db(extra: dict | None = None) -> dict:
    """The not-connected state, spelled out — with the startup reason.

    Distinguished from "no data for this date" deliberately: one is a missing
    database and the other is a night the pipeline did not run, and they need
    different actions from whoever is reading the page at 9am.

    The REASON is carried through from startup rather than guessed at here. A
    missing database, wrong credentials and a wiring fault produced identical
    output once, and telling them apart cost an hour of elimination. The pool
    layer already knows which one it was; this just stops throwing that away.
    """
    st = pool_status().get("equities_scalp") or {}
    if not st.get("configured"):
        why = ("No DSN is configured. It derives from DATABASE_URL by default; "
               "set SCALP_DATABASE_URL if the database lives elsewhere.")
    elif st.get("error"):
        why = f"Connecting failed at startup — {st['error']}"
    else:
        # verify_pools() refuses to start on this, so reaching it means the
        # guard was bypassed rather than that the state is normal.
        why = ("Configured, no recorded failure, and no pool. That is a wiring "
               "fault rather than an environment one.")
    out = {
        "connected": False,
        "error": f"No connection to the {scalp_config.PG_DB!r} database. {why}",
        "reason": st or None,
        "dates": [], "latest_date": None, "metrics": [],
    }
    out.update(extra or {})
    return out


@router.get("/meta")
async def meta(
    date: str = Query(None, description="ISO date; defaults to the latest"),
    pool=Depends(get_scalp_pool),
):
    """Everything the page needs before it can draw anything.

    The available dates, the metric catalog as the database actually holds it,
    the noise variants discovered from those metric names, and the pipeline's
    filter defaults and slider bounds.

    The variant list is DERIVED, not declared. Every dropdown of noise variants
    on this page is built from what `daily_metrics` contains on the selected
    date, so a variant the calibration deletes disappears from the page without
    anything here being edited -- and one that is added appears the same way.
    """
    if pool is None:
        return _no_db()

    want = _as_date(date)

    async with pool.acquire() as conn:
        # DATE OBJECTS, not strings. The stringified list below is for the
        # response; the anchor is picked out of THIS one, so what gets bound is
        # never the formatted copy.
        available = [r["trade_date"] for r in await conn.fetch(
            "SELECT DISTINCT trade_date FROM daily_metrics "
            "ORDER BY trade_date DESC LIMIT 90")]
        dates = [str(x) for x in available]
        if not available:
            return {
                "connected": True, "dates": [], "latest_date": None,
                "metrics": [], "variants": [], "statistics": [], "horizons": [],
                "filters": _filter_block(),
                "note": "The database is reachable but daily_metrics is empty "
                        "— the pipeline has not written a session yet.",
            }

        d = want if want in available else available[0]

        rows = await conn.fetch(
            "SELECT metric, count(*) AS n, count(value) AS n_value "
            "FROM daily_metrics WHERE trade_date = $1 "
            "GROUP BY metric ORDER BY metric", d)
        symbols = await conn.fetchval(
            "SELECT count(DISTINCT symbol) FROM daily_metrics "
            "WHERE trade_date = $1", d)

    catalog = []
    for r in rows:
        e = _catalog_entry(r["metric"])
        # Carried so a metric that is present but all-null is visible as such.
        # An all-null column and an absent one look identical in a pivot, and
        # only one of them means the pipeline is broken.
        e["n"] = int(r["n"])
        e["n_value"] = int(r["n_value"])
        catalog.append(e)

    parsed = [e for e in catalog if e["variant"]]
    ratios = [e for e in parsed if e["variant"]["kind"] == "ratio"]
    noises = [e for e in parsed if e["variant"]["kind"] == "noise_bps"]

    def _uniq(items, key):
        seen, out = set(), []
        for it in items:
            v = it["variant"][key]
            if v is not None and v not in seen:
                seen.add(v)
                out.append(v)
        return out

    return {
        "connected": True,
        "dates": dates,
        # Stringified HERE, at the response, and nowhere earlier.
        "date": str(d),
        "latest_date": dates[0],
        "symbols": int(symbols or 0),
        "metrics": catalog,
        # The dropdown's options, and the axes it is built from.
        "noise_metrics": [e["metric"] for e in noises],
        "ratio_metrics": [e["metric"] for e in ratios],
        "variants":   _uniq(noises, "variant"),
        "horizons":   sorted(_uniq(noises, "horizon_s")),
        "statistics": _uniq(noises, "statistic"),
        "default_noise": _default_noise([e["metric"] for e in noises]),
        "filters": _filter_block(),
        "undocumented": scalp_metric_docs.undocumented(
            [e["metric"] for e in catalog]),
    }


# How far a median has to move against its own trailing history before the
# panel calls it out.
#
# 0.25 because the failure that motivated this panel read 37% low and took an
# hour to find. A threshold at 0.35 would have caught that one and nothing
# milder; at 0.15 a quiet Friday trips it and the row stops meaning anything.
# This is a GUESS between the only two numbers there is evidence for, and it
# is written here rather than buried so it can be moved when there is a second
# incident to calibrate against.
HEALTH_MOVE_THRESHOLD = 0.25

# Universe coverage below this is called out. Eight ETFs correctly refused out
# of 587 is 98.6%, and that should NOT light up -- it is the system working.
# A killed compute run is what this is for.
HEALTH_COVERAGE_FLOOR = 0.97


@router.get("/health")
async def health(
    sessions: int = Query(10, ge=2, le=60),
    trailing: int = Query(10, ge=2, le=60),
    pool=Depends(get_scalp_pool),
):
    """Is the data any good, per session, over the last few.

    THE FAILURE THIS EXISTS FOR happened once: a fetch returned only Nasdaq's
    prints -- no NYSE, ARCA, BATS, EDGX or IEX -- and the only symptom was
    trades/min reading 37% below trailing. It took an hour to diagnose. Once
    the nightly job runs unattended it will happen again and nobody will be
    looking for it, so one red row is the entire point.

    WHAT IT WATCHES, AND WHY NOT AN EXCHANGE COUNT. There is no
    distinct-exchange-count metric and there should not be: fetch.py already
    refuses any symbol-day under MIN_EXCHANGE_CODES and does not write it, so a
    Nasdaq-only pull cannot reach compute at all. A count metric would measure
    a condition the guard prevents, and its only consumer would be a panel
    watching for something that can no longer happen. What actually surfaced
    the problem was a RATE moving against its own history, so that is what this
    watches -- arrivals per minute, plus the two venue shares that move when
    the mix changes underneath.

    EACH DATE IS SCORED AGAINST THE SESSIONS BEFORE IT, never including
    itself. A bad day that contributes to its own baseline is a bad day that
    partly excuses itself, and with a ten-session window one outlier moves the
    median it is being compared against.

    COVERAGE IS TWO DIFFERENT QUESTIONS. Symbols present against the universe
    catches a compute run that died partway; distinct metrics against the
    trailing mode catches a metric family that stopped being written. They fail
    independently and neither implies the other.

    FETCH REJECTIONS ARE NOT AVAILABLE. fetch.py counts thin-tape, empty and
    errored symbol-days and PRINTS them at the end of a run; nothing is
    written to the database or to a file, so there is no per-date rejection
    count to read. What this returns instead is the symbols that are in the
    universe for a date and absent from daily_metrics, which is the union of
    every reason a symbol-day did not land -- refused at fetch, no data, or a
    compute run that was killed. It is honest about not being attributable.
    """
    if pool is None:
        return _no_db({"rows": []})

    span = sessions + trailing
    async with pool.acquire() as conn:
        dates = [r["trade_date"] for r in await conn.fetch(
            "SELECT DISTINCT trade_date FROM daily_metrics "
            "ORDER BY trade_date DESC LIMIT $1", span)]
        if not dates:
            return {"connected": True, "rows": [],
                    "note": "daily_metrics is empty — the pipeline has not "
                            "written a session yet."}
        dates = sorted(dates)

        present = {r["metric"] for r in await conn.fetch(
            "SELECT DISTINCT metric FROM daily_metrics WHERE trade_date = $1",
            dates[-1])}
        got = scalp_columns.resolve_all(present, None, None, None,
                                        list(scalp_columns.HEALTH_KEYS))
        watched = got["columns"]                  # role key -> metric name
        names = list(watched.values())

        med_rows = await conn.fetch(
            "SELECT trade_date, metric, "
            " percentile_cont(0.5) WITHIN GROUP (ORDER BY value) AS med, "
            " count(value) AS n "
            "FROM daily_metrics "
            "WHERE trade_date = ANY($1) AND metric = ANY($2) "
            "GROUP BY trade_date, metric",
            dates, names,
        ) if names else []

        # Index-only on the PK's leading columns, so counting every symbol on
        # every date costs a range scan rather than a heap read.
        cov_rows = await conn.fetch(
            "SELECT trade_date, count(DISTINCT symbol) AS n_symbols, "
            " count(DISTINCT metric) AS n_metrics "
            "FROM daily_metrics WHERE trade_date = ANY($1) GROUP BY trade_date",
            dates,
        )
        uni_rows = await conn.fetch(
            "SELECT trade_date, count(*) AS n FROM universe "
            "WHERE trade_date = ANY($1) AND (qualified OR retained) "
            "GROUP BY trade_date",
            dates,
        )
        # Named, not just counted. "Eight missing" is a number; "eight ETFs"
        # is the difference between a correct refusal and a broken run.
        gap_rows = await conn.fetch(
            "SELECT u.trade_date, u.symbol FROM universe u "
            "WHERE u.trade_date = ANY($1) AND (u.qualified OR u.retained) "
            "  AND NOT EXISTS (SELECT 1 FROM daily_metrics m "
            "                  WHERE m.trade_date = u.trade_date "
            "                    AND m.symbol = u.symbol) "
            "ORDER BY u.trade_date, u.symbol",
            dates,
        )

    by_date: dict = {str(x): {"date": str(x)} for x in dates}
    for r in med_rows:
        by_date[str(r["trade_date"])].setdefault("med", {})[r["metric"]] = r["med"]
    for r in cov_rows:
        e = by_date[str(r["trade_date"])]
        e["n_symbols"] = int(r["n_symbols"] or 0)
        e["n_metrics"] = int(r["n_metrics"] or 0)
    for r in uni_rows:
        by_date[str(r["trade_date"])]["universe_n"] = int(r["n"] or 0)
    for r in gap_rows:
        by_date[str(r["trade_date"])].setdefault("gap", []).append(r["symbol"])

    ordered = [by_date[str(x)] for x in dates]

    def _median(xs):
        xs = sorted(x for x in xs if x is not None)
        if not xs:
            return None
        m = len(xs) // 2
        return xs[m] if len(xs) % 2 else (xs[m - 1] + xs[m]) / 2

    out = []
    for i, e in enumerate(ordered):
        # Strictly earlier sessions only. A date in its own baseline partly
        # excuses itself, and at ten sessions one outlier moves the median it
        # is measured against.
        hist = ordered[max(0, i - trailing):i]
        row = {
            "date": e["date"],
            "n_symbols": e.get("n_symbols", 0),
            "n_metrics": e.get("n_metrics", 0),
            "universe_n": e.get("universe_n"),
            "missing_n": len(e.get("gap", [])),
            "missing_sample": e.get("gap", [])[:12],
            "metrics": {}, "flags": [],
        }
        uni = e.get("universe_n")
        row["coverage"] = (e.get("n_symbols", 0) / uni) if uni else None

        for key, name in watched.items():
            today = (e.get("med") or {}).get(name)
            base = _median([(h.get("med") or {}).get(name) for h in hist])
            change = None
            if today is not None and base:
                change = (today - base) / base
            row["metrics"][key] = {"metric": name, "value": today,
                                   "trailing": base, "change": change,
                                   "n_trailing": len(hist)}
            if change is not None and abs(change) >= HEALTH_MOVE_THRESHOLD:
                row["flags"].append(key)

        # A metric family that stopped being written. Independent of symbol
        # coverage: a full symbol list with a short metric list is a compute
        # change, not a fetch problem.
        modal = _median([h.get("n_metrics") for h in hist])
        if modal and row["n_metrics"] and row["n_metrics"] < modal * 0.95:
            row["flags"].append("n_metrics")
        if row["coverage"] is not None and row["coverage"] < HEALTH_COVERAGE_FLOOR:
            row["flags"].append("coverage")
        out.append(row)

    return {
        "connected": True,
        # Newest first, since the row that matters at 9am is last night's.
        "rows": list(reversed(out[-sessions:])),
        "watched": [{"key": k, "metric": m} for k, m in watched.items()],
        "unresolved": got["missing"],
        "trailing": trailing,
        "thresholds": {"move": HEALTH_MOVE_THRESHOLD,
                       "coverage": HEALTH_COVERAGE_FLOOR},
        # Stated rather than silently omitted: a panel that shows no rejections
        # would otherwise read as "there were none".
        "fetch_rejects": None,
        "fetch_rejects_note":
            "fetch.py counts thin-tape, empty and errored symbol-days and "
            "prints them at the end of a run — nothing is persisted, so there "
            "is no per-date rejection count to read. The missing-symbol column "
            "is the union of every reason a symbol-day did not land, including "
            "a compute run that was killed, and cannot attribute them.",
    }


# ── the fills tables ────────────────────────────────────────────────────────
#
# THIS PROJECT OWNS THESE. Everything else on this page is read-only over the
# pipeline's output; these two are the only tables the dashboard writes, and
# the pipeline knows nothing about them.
#
# ROUND TRIPS, NOT JUST THE DAILY AGGREGATE. The aggregate is what calibration
# correlates against, and storing only that would be enough for it -- but a
# per-trip row with timestamps can be joined to the bucket containing each
# entry, which answers what the book looked like around a fill rather than
# what the ticker averaged that day. That join is the reason the trip rows
# exist and it cannot be reconstructed from an average.
_FILLS_DDL = """
CREATE TABLE IF NOT EXISTS fills (
    trade_date   DATE        NOT NULL,
    symbol       TEXT        NOT NULL,
    -- Position within the day for this symbol, in time order. Part of the key
    -- because two round trips can share an entry SECOND at an eight-second
    -- median hold, and a key that can collide silently drops a trip.
    seq          INTEGER     NOT NULL,
    entry_ts     TIMESTAMP   NOT NULL,
    exit_ts      TIMESTAMP   NOT NULL,
    peak_shares  DOUBLE PRECISION,
    entry_price  DOUBLE PRECISION,
    capital      DOUBLE PRECISION,
    net_pnl      DOUBLE PRECISION,
    duration_s   DOUBLE PRECISION,
    is_long      BOOLEAN     NOT NULL,
    legs         INTEGER     NOT NULL,
    uploaded_at  TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (trade_date, symbol, seq)
);
CREATE INDEX IF NOT EXISTS fills_entry_idx ON fills (symbol, entry_ts);

CREATE TABLE IF NOT EXISTS fills_daily (
    trade_date        DATE   NOT NULL,
    symbol            TEXT   NOT NULL,
    trips             INTEGER,
    net_pnl           DOUBLE PRECISION,
    win_rate          DOUBLE PRECISION,
    -- Wall clock from first entry to last exit, NOT summed hold time. It is
    -- what the session was actually rationed by: 96 minutes spent on one name
    -- for 14 round trips cost 96 minutes whether or not a position was open
    -- in each of them.
    attention_minutes DOUBLE PRECISION,
    dollars_per_min   DOUBLE PRECISION,
    trips_per_min     DOUBLE PRECISION,
    median_hold_s     DOUBLE PRECISION,
    shares            DOUBLE PRECISION,
    uploaded_at       TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (trade_date, symbol)
);
"""


async def _ensure_fills(conn):
    await conn.execute(_FILLS_DDL)


def _fills_raw_dir():
    """Where the uploaded file is kept, so a parser fix can be re-run.

    Volume storage, not the root disk — the root is where Postgres lives and
    it has filled once already. Absence is not fatal: the parse and the write
    are the point, and a statement that could not be archived is worth a line
    in the report rather than a refused upload.
    """
    from pathlib import Path
    return Path(os.environ.get(
        "SCALP_FILLS_RAW_DIR",
        "/mnt/trading_volume_3/equities_scalp/fills_raw"))


@router.post("/upload-fills")
async def upload_fills(
    file: UploadFile = File(...),
    pool=Depends(get_scalp_pool),
):
    """Parse a Schwab statement into round trips and persist them.

    IDEMPOTENT BY DELETING THE DATES IT IS ABOUT TO WRITE, not by upserting
    each row. An upsert alone leaves orphans: re-uploading a corrected
    statement that yields FEWER trips would overwrite the ones that still
    exist and silently keep the ones that no longer do. Delete-then-insert
    inside one transaction makes the newest upload authoritative for every
    date it covers, and touches no other date.

    WHAT THE PARSER DID IS RETURNED, always. Trips found, rows it could not
    read with their line numbers, and any position still open at the end of a
    day. That last one is not a warning, it is the finding: an unclosed
    position silently corrupted every downstream statistic on one session and
    was not caught for hours. It is excluded from the write and named in the
    response.
    """
    if pool is None:
        return _no_db({"ok": False})

    data = await file.read()
    if not data:
        raise HTTPException(400, "the uploaded file is empty.")

    try:
        trips, rep = scalp_fills.parse_statement(data, file.filename or "")
    except scalp_fills.ParseError as exc:
        raise HTTPException(400, str(exc))

    report = rep.as_dict()

    # Archived BEFORE the write, under the name it arrived with plus a hash,
    # so a re-parse after a parser fix does not need the statement re-exported.
    import hashlib
    from datetime import datetime as _dtdt
    archived = None
    archive_error = None
    try:
        raw_dir = _fills_raw_dir()
        raw_dir.mkdir(parents=True, exist_ok=True)
        stamp = _dtdt.now().strftime("%Y%m%dT%H%M%S")
        digest = hashlib.sha256(data).hexdigest()[:12]
        safe = (file.filename or "statement").replace("/", "_").replace("\\", "_")
        target = raw_dir / f"{stamp}_{digest}_{safe}"
        target.write_bytes(data)
        archived = str(target)
    except Exception as exc:
        archive_error = f"{type(exc).__name__}: {exc}"

    daily = scalp_fills.daily_rows(trips)
    dates = sorted({t.trade_date for t in trips})

    written = 0
    if trips:
        # seq within (date, symbol), in time order. Assigned here rather than
        # in the parser because it is a storage key, not a property of a trade.
        by_key: dict = {}
        rows = []
        for t in sorted(trips, key=lambda x: (x.trade_date, x.symbol, x.entry_ts)):
            k = (t.trade_date, t.symbol)
            by_key[k] = by_key.get(k, 0) + 1
            rows.append((t.trade_date, t.symbol, by_key[k], t.entry_ts,
                         t.exit_ts, t.peak_shares, t.entry_price, t.capital,
                         t.net_pnl, t.duration_s, t.is_long, t.legs))

        async with pool.acquire() as conn:
            await _ensure_fills(conn)
            async with conn.transaction():
                await conn.execute(
                    "DELETE FROM fills WHERE trade_date = ANY($1)", dates)
                await conn.execute(
                    "DELETE FROM fills_daily WHERE trade_date = ANY($1)", dates)
                await conn.executemany(
                    "INSERT INTO fills (trade_date, symbol, seq, entry_ts, "
                    " exit_ts, peak_shares, entry_price, capital, net_pnl, "
                    " duration_s, is_long, legs) "
                    "VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12)", rows)
                await conn.executemany(
                    "INSERT INTO fills_daily (trade_date, symbol, trips, "
                    " net_pnl, win_rate, attention_minutes, dollars_per_min, "
                    " trips_per_min, median_hold_s, shares) "
                    "VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10)",
                    [(r["trade_date"], r["symbol"], r["trips"], r["net_pnl"],
                      r["win_rate"], r["attention_minutes"],
                      r["dollars_per_min"], r["trips_per_min"],
                      r["median_hold_s"], r["shares"]) for r in daily])
                written = len(rows)

    return {
        "ok": True,
        "written": written,
        "daily_rows": len(daily),
        "replaced_dates": [str(x) for x in dates],
        "archived": archived,
        "archive_error": archive_error,
        "report": report,
        "daily": [{**r, "trade_date": str(r["trade_date"])} for r in daily],
    }


@router.get("/fills")
async def fills(pool=Depends(get_scalp_pool)):
    """Everything uploaded so far, per ticker-day. The calibration's sample."""
    if pool is None:
        return _no_db({"rows": []})
    async with pool.acquire() as conn:
        await _ensure_fills(conn)
        rows = await conn.fetch(
            "SELECT trade_date, symbol, trips, net_pnl, win_rate, "
            " attention_minutes, dollars_per_min, trips_per_min, "
            " median_hold_s, shares "
            "FROM fills_daily ORDER BY trade_date DESC, net_pnl DESC")
        tot = await conn.fetchrow(
            "SELECT count(*) AS n, sum(net_pnl) AS pnl, "
            " count(DISTINCT symbol) AS symbols, "
            " count(DISTINCT trade_date) AS sessions FROM fills")
    return {
        "connected": True,
        "rows": [{**dict(r), "trade_date": str(r["trade_date"])} for r in rows],
        "totals": {"trips": int(tot["n"] or 0),
                   "net_pnl": float(tot["pnl"] or 0.0),
                   "symbols": int(tot["symbols"] or 0),
                   "sessions": int(tot["sessions"] or 0)},
    }


# ── rank correlation ────────────────────────────────────────────────────────
#
# Implemented here rather than imported. scipy is a dependency of the research
# runner, not of the request path, and this is twenty lines whose correctness
# the harness verifies against scipy directly -- so the reference is used where
# it belongs, in the test, and not in the endpoint.

def _ranks(xs: list[float]) -> list[float]:
    """Ranks with ties averaged.

    Ties matter more here than usual: the median noise statistic reads exactly
    0.0 on every sparse-quote name, so a metric can arrive with a dozen
    identical values. Ranking those 1..12 in arbitrary order would manufacture
    an ordering the data does not contain and hand it to the correlation.
    """
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    out = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            out[order[k]] = avg
        i = j + 1
    return out


def _spearman(xs: list[float], ys: list[float]):
    """Spearman rho, or None when it is not defined."""
    n = len(xs)
    if n < 3:
        return None
    rx, ry = _ranks(xs), _ranks(ys)
    mx = sum(rx) / n
    my = sum(ry) / n
    sxy = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    sxx = sum((a - mx) ** 2 for a in rx)
    syy = sum((b - my) ** 2 for b in ry)
    if sxx <= 0 or syy <= 0:
        # A metric with one value across the whole sample has no ordering to
        # correlate. None, not zero: zero is a measurement.
        return None
    return sxy / (sxx * syy) ** 0.5


# The three things a metric could be calibrated against. $/min is the one the
# strategy is rationed by -- attention is the scarce resource, not capital --
# and the other two are here because agreeing with them is weak evidence that
# a correlation is real rather than a quirk of one denominator.
CALIB_TARGETS = {
    "dollars_per_min": "$ per minute of attention",
    "net_pnl":         "net P&L",
    "win_rate":        "win rate",
}


def _normal_sf(z: float) -> float:
    """P(|Z| > z) for a standard normal. Stdlib erf, no dependency."""
    import math
    return math.erfc(abs(z) / math.sqrt(2.0))


@router.get("/calibration")
async def calibration(
    target:    str = Query("dollars_per_min"),
    min_pairs: int = Query(5, ge=3, le=200),
    pool=Depends(get_scalp_pool),
):
    """Rank correlation of every metric against realised results.

    THIS DECIDES WHICH VARIANT DRIVES THE RANKING, and it is the only thing on
    the page that can. Every other panel describes the metrics; this one asks
    whether any of them separates the names that made money from the names
    that did not.

    IT IS A TABLE TO WATCH, NOT A VERDICT, and the arithmetic says why. With a
    handful of ticker-days and ~232 candidate metrics, the expected number
    clearing any given |rho| BY CHANCE ALONE is computed and returned beside
    the results. At fifteen pairs, a two-tailed threshold of 0.5 is p ~ 0.06,
    so about fourteen metrics out of 232 should clear it with no relationship
    whatsoever. A ranked list without that number beside it reads as a finding
    and is mostly noise.

    Spearman rather than Pearson: the hypothesis is that richer spread over
    noise ranks BETTER, not that $/min is a linear function of it, and one
    outsized session would dominate a Pearson estimate at this sample size.
    """
    if pool is None:
        return _no_db({"rows": []})
    if target not in CALIB_TARGETS:
        raise HTTPException(
            400, f"target must be one of {sorted(CALIB_TARGETS)}")

    async with pool.acquire() as conn:
        await _ensure_fills(conn)
        rows = await conn.fetch(
            f"SELECT m.metric, m.value AS x, f.{target} AS y, "
            f"       m.trade_date, m.symbol "
            f"FROM daily_metrics m "
            f"JOIN fills_daily f "
            f"  ON f.trade_date = m.trade_date AND f.symbol = m.symbol "
            f"WHERE m.value IS NOT NULL AND f.{target} IS NOT NULL")
        sample = await conn.fetchrow(
            f"SELECT count(*) AS n, count(DISTINCT symbol) AS symbols, "
            f" count(DISTINCT trade_date) AS sessions "
            f"FROM fills_daily WHERE {target} IS NOT NULL")

        # Every metric with a declared direction, across the WHOLE universe on
        # the latest session. Used below to ask whether the contradicting ones
        # are separate findings or one variable wearing several names — a
        # question the 40-ticker-day fills sample is far too small to answer,
        # and 587 symbols is not.
        directed = sorted({c for r in scalp_columns.ROLES
                           if r.higher_better is not None for c in r.candidates})
        latest = await conn.fetchval(
            "SELECT max(trade_date) FROM daily_metrics")
        uni_rows = await conn.fetch(
            "SELECT symbol, metric, value FROM daily_metrics "
            "WHERE trade_date = $1 AND metric = ANY($2) AND value IS NOT NULL",
            latest, directed) if latest and directed else []

    # `target` is interpolated above, which is safe ONLY because it was
    # checked against CALIB_TARGETS first -- a whitelist, not an escape.
    by_metric: dict = {}
    for r in rows:
        by_metric.setdefault(r["metric"], ([], []))
        by_metric[r["metric"]][0].append(float(r["x"]))
        by_metric[r["metric"]][1].append(float(r["y"]))

    out = []
    for metric, (xs, ys) in by_metric.items():
        if len(xs) < min_pairs:
            continue
        rho = _spearman(xs, ys)
        if rho is None:
            continue
        n = len(xs)
        # Fisher's approximation. At these sample sizes it is indicative
        # rather than exact, which is the right register for the whole panel.
        z = rho * (n - 1) ** 0.5
        link = scalp_metric_docs.header_link(metric)
        out.append({
            "metric": metric, "rho": rho, "n": n,
            "p": _normal_sf(z),
            "tooltip": link.get("tooltip"), "href": link.get("href"),
            "section": link.get("section"),
            "variant": _parse_variant(metric),
        })

    out.sort(key=lambda r: -abs(r["rho"]))

    # ── metrics whose SIGN contradicts what the column claims ────────────
    #
    # scalp_columns declares `higher_better` for every role — that is the
    # direction the strategy's premise says the metric should point. A
    # correlation with the opposite sign is not a weak result, it is a result
    # that disagrees with the reason the column is on the page, and it belongs
    # above the ranked list rather than at whatever row |rho| happens to put
    # it on.
    #
    # Derived from the declared direction rather than a list of metric names,
    # so a role added later is checked without anything here changing.
    direction = {}
    for role in scalp_columns.ROLES:
        if role.higher_better is None:
            continue
        for cand in role.candidates:
            direction[cand] = (role.higher_better, role.key, role.label)

    contradictions = []
    for r in out:
        got = direction.get(r["metric"])
        if not got:
            continue
        higher_better, key, label = got
        # win_rate and $/min are both "more is better"; a metric that is
        # supposed to help should correlate positively with them.
        expected_positive = higher_better
        if (r["rho"] < 0) == expected_positive and abs(r["rho"]) >= 0.25:
            contradictions.append({
                "metric": r["metric"], "rho": r["rho"], "n": r["n"],
                "role": key, "label": label,
                "expected": "higher is better" if higher_better
                            else "lower is better",
                "note": f"{label} is on the ranked table because "
                        f"{'more' if higher_better else 'less'} of it should "
                        f"help. It correlates the other way.",
            })
    contradictions.sort(key=lambda c: -abs(c["rho"]))

    # ── are these separate findings, or one confound? ────────────────────
    #
    # FOUR contradictions is a different object from one. Four independently
    # broken premises would be remarkable; four metrics that co-vary across
    # the universe pointing the same way is ONE uncontrolled variable, and
    # reading them as four warnings would be reading them wrong.
    #
    # So this measures whether the contradicting metrics are mutually ranked:
    # the mean |Spearman| among every pair of them, across the full universe
    # on the latest session. That is a fact about the METRICS and needs no
    # fills at all, which is why it can be said at n=40 when nothing else can.
    #
    # It deliberately does NOT name a cause. What the shared factor IS -- tight
    # liquid institutionally-traded names is the obvious candidate -- is a
    # hypothesis the data here cannot settle, and stating it as a finding
    # would be exactly the overreach the rest of this panel exists to prevent.
    by_sym: dict = {}
    for r in uni_rows:
        by_sym.setdefault(r["metric"], {})[r["symbol"]] = r["value"]

    # NEVER A BARE None. The first version returned None whenever the
    # computation did not produce a number, and the client only rendered the
    # coherence figure inside the grouped callout -- so a mean below the
    # threshold, an empty query and an exception all produced the same thing:
    # the old ungrouped layout, with the number nowhere on the page. That is
    # the same failure shape as the empty script tag. The page looked fine and
    # quietly was not doing the thing.
    #
    # So this always returns a STATUS and a REASON, and the client always
    # renders it.
    coherence = {"status": "not_applicable",
                 "reason": "fewer than two metrics contradict their column's "
                           "declared direction, so there is nothing to group."}
    if len(contradictions) >= 2:
        try:
            names = [c["metric"] for c in contradictions if c["metric"] in by_sym]
            absent = [c["metric"] for c in contradictions
                      if c["metric"] not in by_sym]
            pairs, shared_n = [], 0
            for i in range(len(names)):
                for j in range(i + 1, len(names)):
                    a, b = by_sym[names[i]], by_sym[names[j]]
                    shared = sorted(set(a) & set(b))
                    if len(shared) < 20:
                        continue
                    rho = _spearman([a[s] for s in shared], [b[s] for s in shared])
                    if rho is not None:
                        pairs.append({"a": names[i], "b": names[j],
                                      "rho": rho, "n": len(shared)})
                        shared_n = max(shared_n, len(shared))
            if not pairs:
                coherence = {
                    "status": "unavailable",
                    "reason": (f"none of the {len(contradictions)} contradicting "
                               f"metrics share 20+ symbols on {latest} — "
                               f"{len(absent)} were absent from the universe "
                               f"query entirely. The grouping cannot be "
                               f"decided, so they are shown separately."),
                    "absent": absent,
                }
            else:
                mean_abs = sum(abs(p["rho"]) for p in pairs) / len(pairs)
                coherence = {
                    # The threshold is a READING AID, not a test. Both sides of
                    # it report the same number; only the sentence changes.
                    "status": "coherent" if mean_abs >= 0.4 else "separate",
                    "n_metrics": len(names),
                    "n_pairs": len(pairs),
                    "universe": shared_n,
                    "mean_abs_rho": mean_abs,
                    "threshold": 0.4,
                    "absent": absent,
                    "pairs": sorted(pairs, key=lambda p: -abs(p["rho"]))[:10],
                }
        except Exception as exc:                          # noqa: BLE001
            # Surfaced, not swallowed. A silent fall-through to the old layout
            # is indistinguishable from the feature never having shipped.
            coherence = {
                "status": "error",
                "reason": f"{type(exc).__name__}: {exc}",
            }

    n_pairs = max((r["n"] for r in out), default=0)
    n_metrics = len(out)
    # THE NUMBER THAT KEEPS THIS HONEST. How many of these metrics would clear
    # each threshold with no relationship at all, given the sample.
    expected = []
    for thr in (0.3, 0.5, 0.7, 0.9):
        if n_pairs >= 3:
            p = _normal_sf(thr * (n_pairs - 1) ** 0.5)
            observed = sum(1 for r in out if abs(r["rho"]) >= thr)
            expected.append({"threshold": thr, "expected": p * n_metrics,
                             "observed": observed})

    return {
        "connected": True,
        "target": target,
        "target_label": CALIB_TARGETS[target],
        "targets": [{"key": k, "label": v} for k, v in CALIB_TARGETS.items()],
        "rows": out[:400],
        "n_metrics": n_metrics,
        "n_pairs": n_pairs,
        "sample": {"ticker_days": int(sample["n"] or 0),
                   "symbols": int(sample["symbols"] or 0),
                   "sessions": int(sample["sessions"] or 0)},
        "expected_by_chance": expected,
        "contradictions": contradictions,
        "contradiction_coherence": coherence,
        "note":
            "Spearman rank correlation across ticker-days. Read the "
            "expected-by-chance column first: with this many metrics and this "
            "few ticker-days, a list sorted by |rho| will always have "
            "something at the top. What matters is whether the same metric "
            "stays there as sessions accumulate — not its rank today.",
    }


# ── 2.2 / 2.3: narrowing the field without any trade data ───────────────────
#
# Both of these run on daily_metrics alone. That ordering is deliberate: with
# ~232 metrics and 40 ticker-days of fills, the calibration cannot separate
# metrics that are measuring the same thing, and it should not be asked to.
# Redundancy and instability are properties of the METRICS, answerable across
# 587 symbols without a single fill, and removing what they find is what leaves
# calibration a field small enough for its sample size.

def _rank_matrix(values):
    """Columns ranked with ties averaged, as a numpy array.

    Spearman across a whole matrix at once: rank each column, then take the
    Pearson correlation of the ranks. numpy makes a 75 x 587 matrix instant
    where a pairwise Python loop over 2,800 pairs would not be.
    """
    import numpy as np
    a = np.asarray(values, dtype=float)
    out = np.empty_like(a)
    for j in range(a.shape[1]):
        col = a[:, j]
        ok = ~np.isnan(col)
        r = np.full(col.shape, np.nan)
        if ok.sum() >= 2:
            vals = col[ok]
            order = vals.argsort(kind="mergesort")
            ranks = np.empty(len(vals), dtype=float)
            ranks[order] = np.arange(1, len(vals) + 1, dtype=float)
            # Ties averaged. The median statistic reads exactly 0.0 across
            # every sparse-quote name, so a metric can arrive with dozens of
            # identical values; ranking those arbitrarily would manufacture an
            # ordering and then correlate it.
            sv = vals[order]
            i = 0
            while i < len(sv):
                k = i
                while k + 1 < len(sv) and sv[k + 1] == sv[i]:
                    k += 1
                if k > i:
                    ranks[order[i:k + 1]] = (i + k) / 2.0 + 1.0
                i = k + 1
            r[ok] = ranks
        out[:, j] = r
    return out


@router.get("/metric-correlation")
async def metric_correlation(
    date:    str = Query(None),
    family:  str = Query(None, description="only metrics whose name starts here"),
    limit:   int = Query(90, ge=2, le=250),
    cutoff:  float = Query(0.97, ge=0.5, le=1.0),
    pool=Depends(get_scalp_pool),
):
    """Which metrics are measuring the same thing. NO TRADE DATA NEEDED.

    ~232 columns, most of them one measurement at a different variant, horizon
    or statistic. If two correlate at 0.97 across 587 symbols, one of them is
    redundant, and that is knowable now rather than after enough sessions
    accumulate to tell them apart on outcomes -- which, at their similarity,
    would be never.

    Returned as a matrix plus a LEAF ORDERING from hierarchical clustering, so
    the blocks are adjacent and readable. Unordered, a 90x90 grid of mostly
    high correlations shows nothing; ordered, the families separate visibly.

    `redundant` is the actionable half: groups whose members all sit above the
    cutoff with each other. Each group is one metric's worth of information.
    """
    if pool is None:
        return _no_db({"metrics": [], "matrix": []})

    want = _as_date(date)
    async with pool.acquire() as conn:
        dates = [r["trade_date"] for r in await conn.fetch(
            "SELECT DISTINCT trade_date FROM daily_metrics "
            "ORDER BY trade_date DESC LIMIT 90")]
        if not dates:
            return {"connected": True, "metrics": [], "matrix": [],
                    "note": "daily_metrics is empty."}
        d = want if want in dates else dates[0]
        rows = await conn.fetch(
            "SELECT symbol, metric, value FROM daily_metrics "
            "WHERE trade_date = $1 AND value IS NOT NULL", d)

    import numpy as np
    by_metric: dict = {}
    for r in rows:
        if family and not r["metric"].startswith(family):
            continue
        by_metric.setdefault(r["metric"], {})[r["symbol"]] = r["value"]

    # Symbols shared by every metric under consideration. A per-pair
    # intersection would give each cell a different sample and make the matrix
    # internally inconsistent -- two cells in the same row would not be
    # comparable.
    if not by_metric:
        return {"connected": True, "date": str(d), "metrics": [], "matrix": [],
                "note": "no metrics matched."}
    common = set.intersection(*(set(v) for v in by_metric.values()))
    names = sorted(by_metric)[:limit]
    syms = sorted(common)
    if len(syms) < 5 or len(names) < 2:
        return {"connected": True, "date": str(d), "metrics": names,
                "matrix": [], "n_symbols": len(syms),
                "note": f"only {len(syms)} symbols carry every selected "
                        f"metric — too few to correlate."}

    mat = np.array([[by_metric[m][s] for m in names] for s in syms])
    ranks = _rank_matrix(mat)
    # Constant columns have zero variance and would produce NaN. Dropped and
    # NAMED: a metric with one value across the universe is a finding.
    sd = np.nanstd(ranks, axis=0)
    keep = sd > 0
    dropped = [n for n, k in zip(names, keep) if not k]
    names = [n for n, k in zip(names, keep) if k]
    ranks = ranks[:, keep]
    if len(names) < 2:
        return {"connected": True, "date": str(d), "metrics": names,
                "matrix": [], "constant": dropped,
                "note": "fewer than two non-constant metrics."}

    C = np.corrcoef(ranks, rowvar=False)
    C = np.nan_to_num(C, nan=0.0)

    # Leaf ordering, so the blocks are adjacent.
    order = list(range(len(names)))
    try:
        from scipy.cluster.hierarchy import linkage, leaves_list
        from scipy.spatial.distance import squareform
        dist = 1.0 - np.abs(C)
        np.fill_diagonal(dist, 0.0)
        dist = (dist + dist.T) / 2.0
        order = list(leaves_list(linkage(squareform(dist, checks=False),
                                         method="average")))
    except Exception:
        # scipy is a research-runner dependency and this panel should still
        # render without it -- unordered is worse, not broken.
        pass

    names_o = [names[i] for i in order]
    C_o = C[np.ix_(order, order)]

    # Redundant groups: a chain of metrics each above the cutoff with the
    # group's first member. Deliberately simple -- the point is "these three
    # are one metric", not a clustering result.
    used, groups = set(), []
    for i, n in enumerate(names_o):
        if n in used:
            continue
        members = [n]
        for j in range(i + 1, len(names_o)):
            if names_o[j] in used:
                continue
            if abs(C_o[i, j]) >= cutoff:
                members.append(names_o[j])
                used.add(names_o[j])
        if len(members) > 1:
            used.add(n)
            groups.append(members)

    return {
        "connected": True, "date": str(d),
        "metrics": names_o,
        "matrix": [[round(float(x), 4) for x in row] for row in C_o],
        "n_symbols": len(syms),
        "cutoff": cutoff,
        "redundant": groups,
        "constant": dropped,
        "families": sorted({(_parse_variant(m) or {}).get("kind") or m.split("_")[0]
                            for m in by_metric}),
        "note": "Spearman across symbols on one session. Ordered by "
                "hierarchical clustering so the blocks are adjacent.",
    }


@router.get("/rank-stability")
async def rank_stability(
    sessions: int = Query(10, ge=3, le=60),
    top_n:    int = Query(20, ge=5, le=100),
    pool=Depends(get_scalp_pool),
):
    """Does a metric rank the same names tomorrow. NO TRADE DATA NEEDED.

    A metric whose top 20 reshuffles every session is measuring noise in the
    measurement rather than a property of the stock. That is disqualifying
    regardless of how it calibrates -- a signal that cannot be acted on the
    next morning is not a signal -- and it costs nothing but the metric's own
    history to find out.

    TWO NUMBERS, because they fail differently. The consecutive-session rank
    correlation is the whole universe's ordering; top-N retention is the only
    part of that ordering anyone trades. A metric can hold the bulk of the
    universe in place while churning its head, and the second number is the
    one that matters for a ranked table people read the first page of.

    The real example this is built for: one name read 0.069 bps one session
    and 1.727 the next with nothing changing about the stock, because the
    median statistic flipped across a boundary. `worst_jump` reports the
    largest single-name rank move so that shows up as a name rather than only
    as a lower average.
    """
    if pool is None:
        return _no_db({"rows": []})

    async with pool.acquire() as conn:
        dates = [r["trade_date"] for r in await conn.fetch(
            "SELECT DISTINCT trade_date FROM daily_metrics "
            "ORDER BY trade_date DESC LIMIT $1", sessions)]
        if len(dates) < 2:
            return {"connected": True, "rows": [], "n_sessions": len(dates),
                    "note": "at least two sessions are needed to compare "
                            "one ranking against another."}
        dates = sorted(dates)
        rows = await conn.fetch(
            "SELECT trade_date, symbol, metric, value FROM daily_metrics "
            "WHERE trade_date = ANY($1) AND value IS NOT NULL", dates)

    per: dict = {}
    for r in rows:
        per.setdefault(r["metric"], {}).setdefault(
            str(r["trade_date"]), {})[r["symbol"]] = r["value"]

    out = []
    for metric, by_date in per.items():
        days = [by_date[str(x)] for x in dates if str(x) in by_date]
        if len(days) < 2:
            continue
        rhos, retention, worst = [], [], None
        for a, b in zip(days, days[1:]):
            shared = sorted(set(a) & set(b))
            if len(shared) < 5:
                continue
            rho = _spearman([a[s] for s in shared], [b[s] for s in shared])
            if rho is not None:
                rhos.append(rho)
            # Top-N by value descending, on each side.
            ta = [s for s in sorted(shared, key=lambda s: -a[s])[:top_n]]
            tb = set(s for s in sorted(shared, key=lambda s: -b[s])[:top_n])
            if ta:
                retention.append(len(set(ta) & tb) / len(ta))
            ra = {s: i for i, s in enumerate(sorted(shared, key=lambda s: -a[s]))}
            rb = {s: i for i, s in enumerate(sorted(shared, key=lambda s: -b[s]))}
            for s in shared:
                jump = abs(ra[s] - rb[s])
                if worst is None or jump > worst["places"]:
                    worst = {"symbol": s, "places": jump,
                             "from": a[s], "to": b[s], "of": len(shared)}
        if not rhos:
            continue
        link = scalp_metric_docs.header_link(metric)
        out.append({
            "metric": metric,
            "rank_corr": sum(rhos) / len(rhos),
            "top_retention": (sum(retention) / len(retention)) if retention else None,
            "pairs": len(rhos),
            "worst_jump": worst,
            "section": link.get("section"), "href": link.get("href"),
            "tooltip": link.get("tooltip"),
            "variant": _parse_variant(metric),
        })

    out.sort(key=lambda r: -(r["rank_corr"] or -1))
    return {
        "connected": True,
        "rows": out,
        "n_sessions": len(dates),
        "dates": [str(x) for x in dates],
        "top_n": top_n,
        "note":
            "Average Spearman between consecutive sessions' rankings, and the "
            "share of each session's top "
            f"{top_n} still there the next day. A metric near 1.0 ranks the "
            "same names tomorrow; one near 0 is re-drawing the list every "
            "morning and cannot be traded off regardless of how it "
            "calibrates.",
    }


@router.get("/series")
async def series(
    metrics:  str = Query(..., description="comma-separated metric names"),
    symbols:  str = Query(None, description="comma-separated; default traded"),
    sessions: int = Query(30, ge=2, le=250),
    pool=Depends(get_scalp_pool),
):
    """One or more metrics over time, for one or more symbols. 2.6.

    SEPARATES LEVEL FROM STABILITY, which a single session's column cannot.
    A name whose ratio reads 4.0 every day and one that averages 4.0 by
    alternating 1 and 7 are the same number in the ranked table and different
    trades, and only the series shows which is which.

    The default symbol set is WHAT HAS BEEN TRADED, not the top of today's
    ranking. Those are the names whose outcomes are known, so a shape drawn
    for them can be read against a result; the top of the ranking is exactly
    the set whose behaviour is still a question.
    """
    if pool is None:
        return _no_db({"series": {}})

    names = [m.strip() for m in metrics.split(",") if m.strip()]
    if not names:
        raise HTTPException(400, "no metric named.")

    async with pool.acquire() as conn:
        dates = [r["trade_date"] for r in await conn.fetch(
            "SELECT DISTINCT trade_date FROM daily_metrics "
            "ORDER BY trade_date DESC LIMIT $1", sessions)]
        if not dates:
            return {"connected": True, "series": {}, "dates": [],
                    "note": "daily_metrics is empty."}
        dates = sorted(dates)

        if symbols:
            syms = [s.strip().upper() for s in symbols.split(",") if s.strip()]
        else:
            await _ensure_fills(conn)
            syms = [r["symbol"] for r in await conn.fetch(
                "SELECT symbol, sum(net_pnl) AS pnl FROM fills_daily "
                "GROUP BY symbol ORDER BY sum(net_pnl) DESC LIMIT 6")]
        if not syms:
            return {"connected": True, "series": {}, "dates": [str(x) for x in dates],
                    "note": "no symbols selected and nothing has been traded "
                            "yet — pick names explicitly, or upload a "
                            "statement and this fills in."}

        rows = await conn.fetch(
            "SELECT trade_date, symbol, metric, value FROM daily_metrics "
            "WHERE trade_date = ANY($1) AND symbol = ANY($2) "
            "  AND metric = ANY($3) "
            "ORDER BY symbol, metric, trade_date",
            dates, syms, names)

    ds = [str(x) for x in dates]
    idx = {d: i for i, d in enumerate(ds)}
    out: dict = {}
    for r in rows:
        out.setdefault(r["symbol"], {}).setdefault(
            r["metric"], [None] * len(ds))[idx[str(r["trade_date"])]] = r["value"]

    # Shared axis bounds per metric, so the small multiples are comparable.
    # Six panels on their own scales is six shapes and no comparison, which is
    # the one thing this panel exists to allow.
    bounds = {}
    for m in names:
        vals = [v for sym in out.values() for v in sym.get(m, []) if v is not None]
        if vals:
            lo, hi = min(vals), max(vals)
            pad = (hi - lo) * 0.08 or (abs(hi) * 0.1 or 1.0)
            bounds[m] = {"min": lo - pad, "max": hi + pad}

    return {
        "connected": True,
        "dates": ds,
        "symbols": syms,
        "metrics": names,
        "series": out,
        "bounds": bounds,
        "meta": {m: scalp_metric_docs.header_link(m) for m in names},
        # Named so a symbol with no data is legible as such rather than as an
        # empty panel someone assumes is still loading.
        "missing": [s for s in syms if s not in out],
    }


# ── which filter constrains which column ────────────────────────────────────
#
# The pipeline's DEFAULT_FILTERS are named for what they threshold, not for the
# metric they threshold it on, so the join lives here. Kept as a table rather
# than a chain of ifs because the failure it prevents is a filter silently
# doing nothing: a threshold whose column did not resolve must be REPORTED, not
# skipped, or the pass count is a number about a filter that never ran.
_FILTER_ROLES = {
    "min_spread_cents":           ("spread_cents", "min"),
    "min_trades_per_min":         ("arrivals", "min"),
    "max_noise_bps":              ("noise", "max"),
    "min_noise_bps":              ("noise", "min"),
    "min_quote_bucket_coverage":  ("coverage", "min"),
}


@router.get("/candidates")
async def candidates(
    date:     str = Query(None),
    noise:    str = Query(None, description="the selected noise metric"),
    columns:  str = Query(None, description="comma-separated role keys"),
    extra:    str = Query(None, description="comma-separated raw metric names"),
    sort:     str = Query(None, description="role key or raw metric name"),
    desc:     bool = Query(True),
    limit:    int = Query(600, ge=1, le=5000),
    spark_sessions: int = Query(10, ge=0, le=60),
    # The pipeline's five read-time thresholds, declared rather than collected
    # from **kwargs -- FastAPI validates what it can see, and a typo'd query
    # parameter should be a 422 rather than a filter that silently did not run.
    # scripts/check_scalp_metrics.py fails the build if this set stops matching
    # DEFAULT_FILTERS, so a threshold added upstream cannot go unexposed.
    min_spread_cents:          float = Query(None),
    min_trades_per_min:        float = Query(None),
    max_noise_bps:             float = Query(None),
    min_noise_bps:             float = Query(None),
    min_quote_bucket_coverage: float = Query(None),
    pool=Depends(get_scalp_pool),
):
    """One row per symbol, with the filters applied at READ time.

    EVERY SYMBOL IS STORED whether it passes or not, and this endpoint is what
    makes that worth something: it returns the pass/fail decision per row and
    the count each threshold rejected, so a threshold can be judged against the
    names it is excluding rather than trusted. A filter that runs in the
    pipeline can only ever be confirmed by its own output.

    THE PIVOT HAPPENS IN SQL. daily_metrics is long, and the alternative --
    fetching 232 metrics x 587 symbols and reshaping here -- ships 136,000 rows
    to build 587. The FILTER aggregate does it in one pass, and every metric
    name is a bound parameter rather than interpolated text.

    THE VARIANT SELECTOR MOVES FIVE COLUMNS. Noise, the ratio over it, the
    quote coverage at that horizon, and both halves of the move-rate
    decomposition. They cannot be chosen independently without the row becoming
    incoherent: noise is measured between consecutive OBSERVED buckets, so a
    10s noise reading beside 30s coverage compares two different things while
    looking like a comparison.
    """
    if pool is None:
        return _no_db({"rows": [], "columns": []})

    want = _as_date(date)
    v = h = stat = None
    if noise:
        parsed = _parse_variant(noise)
        if parsed:
            v, h, stat = parsed["variant"], parsed["horizon_s"], parsed["statistic"]

    keys = [k.strip() for k in columns.split(",") if k.strip()] if columns else None
    extras = [e.strip() for e in extra.split(",") if e.strip()] if extra else []

    async with pool.acquire() as conn:
        available_dates = [r["trade_date"] for r in await conn.fetch(
            "SELECT DISTINCT trade_date FROM daily_metrics "
            "ORDER BY trade_date DESC LIMIT 90")]
        if not available_dates:
            return {"connected": True, "date": None, "rows": [], "columns": [],
                    "note": "daily_metrics is empty — the pipeline has not "
                            "written a session yet."}
        d = want if want in available_dates else available_dates[0]

        # What this date actually holds, so a role resolves against reality
        # rather than against what the docs say could exist.
        present = {r["metric"] for r in await conn.fetch(
            "SELECT DISTINCT metric FROM daily_metrics WHERE trade_date = $1", d)}

        if noise is None:
            noise = _default_noise(
                [m for m in present
                 if (_parse_variant(m) or {}).get("kind") == "noise_bps"])
            parsed = _parse_variant(noise) if noise else None
            if parsed:
                v, h, stat = (parsed["variant"], parsed["horizon_s"],
                              parsed["statistic"])

        got = scalp_columns.resolve_all(present, v, h, stat, keys)
        col_map = dict(got["columns"])
        # Anything picked from the column chooser rides alongside the roles,
        # keyed by its own name so the two cannot collide.
        for e in extras:
            if e in present:
                col_map.setdefault(e, e)

        if not col_map:
            return {"connected": True, "date": str(d), "rows": [], "columns": [],
                    "missing": got["missing"],
                    "note": "No requested column resolved against this date."}

        # ── the pivot ────────────────────────────────────────────────────
        order = list(col_map)                       # stable key order
        metrics = [col_map[k] for k in order]
        params: list = [d]
        sel = ["m.symbol"]
        for i, name in enumerate(metrics):
            params.append(name)
            sel.append(f"max(m.value) FILTER (WHERE m.metric = ${len(params)}) "
                       f"AS v{i}")
        params.append(metrics)
        rows = await conn.fetch(
            f"SELECT {', '.join(sel)} FROM daily_metrics m "
            f"WHERE m.trade_date = $1 AND m.metric = ANY(${len(params)}) "
            f"GROUP BY m.symbol ORDER BY m.symbol",
            *params,
        )

        # ── what I actually traded, per symbol ───────────────────────────
        #
        # The brief asked for this in the ranked table from the start and it
        # waited on the upload. Pooled across ALL sessions rather than taken
        # from this date: the question a marker answers is "have I traded this
        # name and how did it go", and restricting it to the selected date
        # would blank the marker on every date except the ones with fills --
        # which is most of them.
        await _ensure_fills(conn)
        traded = {r["symbol"]: dict(r) for r in await conn.fetch(
            "SELECT symbol, count(*) AS days, sum(trips) AS trips, "
            " sum(net_pnl) AS net_pnl, "
            " sum(net_pnl) / NULLIF(sum(attention_minutes), 0) AS pnl_per_min "
            "FROM fills_daily GROUP BY symbol")}

        # ── the ratio's own history, for the stability sparkline ─────────
        #
        # STABILITY, NOT LEVEL. Today's ratio is already a column; what the
        # sparkline adds is whether the name reads the same way tomorrow. A
        # metric whose value swings by an order of magnitude between sessions
        # is measuring the measurement.
        spark: dict[str, list] = {}
        spark_dates: list[str] = []
        ratio_col = col_map.get("ratio")
        if ratio_col and spark_sessions:
            window = available_dates[:spark_sessions]
            srows = await conn.fetch(
                "SELECT trade_date, symbol, value FROM daily_metrics "
                "WHERE metric = $1 AND trade_date = ANY($2) "
                "ORDER BY symbol, trade_date",
                ratio_col, window,
            )
            spark_dates = [str(x) for x in sorted(window)]
            idx = {dt: i for i, dt in enumerate(spark_dates)}
            for r in srows:
                arr = spark.setdefault(r["symbol"], [None] * len(spark_dates))
                arr[idx[str(r["trade_date"])]] = r["value"]

    # ── read-time filtering ──────────────────────────────────────────────
    supplied = {
        "min_spread_cents": min_spread_cents,
        "min_trades_per_min": min_trades_per_min,
        "max_noise_bps": max_noise_bps,
        "min_noise_bps": min_noise_bps,
        "min_quote_bucket_coverage": min_quote_bucket_coverage,
    }
    # The pipeline's value unless the caller moved the slider. Defaults come
    # from the vendored config, so they cannot drift from what the ranking
    # upstream used.
    thresholds = {k: float(vv) for k, vv in scalp_config.DEFAULT_FILTERS.items()}
    for k, vv in supplied.items():
        if vv is not None and k in thresholds:
            thresholds[k] = float(vv)

    active, inert = {}, []
    for fk, (role_key, direction) in _FILTER_ROLES.items():
        if role_key in col_map:
            active[fk] = (order.index(role_key), direction, thresholds[fk])
        else:
            # A threshold whose column is absent did NOT run. Saying so is the
            # difference between "12 names pass" and "12 names pass, and one of
            # your four filters was not applied".
            inert.append(fk)

    out, rejected = [], {fk: 0 for fk in active}
    for r in rows:
        vals = {k: r[f"v{i}"] for i, k in enumerate(order)}
        fails = []
        for fk, (i, direction, thr) in active.items():
            x = r[f"v{i}"]
            if x is None:
                fails.append(fk)
            elif direction == "min" and x < thr:
                fails.append(fk)
            elif direction == "max" and x > thr:
                fails.append(fk)
        for fk in fails:
            rejected[fk] += 1
        t = traded.get(r["symbol"])
        out.append({"symbol": r["symbol"], "values": vals,
                    "passes": not fails, "fails": fails,
                    "spark": spark.get(r["symbol"]),
                    # None rather than an empty object for a name never
                    # traded: "no result" and "a result of zero" are different
                    # readings and the colour ramp has to tell them apart.
                    "traded": ({"days": int(t["days"]),
                                "trips": int(t["trips"] or 0),
                                "net_pnl": float(t["net_pnl"] or 0.0),
                                "pnl_per_min": (float(t["pnl_per_min"])
                                                if t["pnl_per_min"] is not None
                                                else None)}
                               if t else None)})

    # ── sort ─────────────────────────────────────────────────────────────
    #
    # Failing rows sort BELOW passing ones rather than being dropped. They are
    # the only evidence that can say whether a threshold sits in the right
    # place, and a filter that hides its own rejects cannot be judged.
    sort_key = sort if sort in col_map else ("ratio" if "ratio" in col_map
                                             else order[0])
    # Nulls sort last in BOTH directions. A missing measurement is not a small
    # value, and letting it float to the top of an ascending sort would put the
    # names with no data where the best ones belong.
    sign = -1.0 if desc else 1.0

    def _sk(row):
        x = row["values"].get(sort_key)
        return (0 if row["passes"] else 1, x is None,
                sign * x if x is not None else 0.0)

    out.sort(key=_sk)

    n_pass = sum(1 for r in out if r["passes"])
    return {
        "connected": True,
        "date": str(d),
        "noise": noise,
        "variant": {"variant": v, "horizon_s": h, "statistic": stat},
        "columns": [{"key": k, "metric": col_map[k],
                     **({"role": True} if k in scalp_columns.BY_KEY
                        else {"role": False})}
                    for k in order],
        "roles": scalp_columns.describe_roles(keys),
        "missing": got["missing"],
        "rows": out[:limit],
        "n_total": len(out),
        "n_pass": n_pass,
        "n_shown": min(len(out), limit),
        "thresholds": thresholds,
        "rejected": rejected,
        "inert_filters": inert,
        "spark_dates": spark_dates,
        "spark_metric": ratio_col,
        "n_traded": sum(1 for r in out if r["traded"]),
    }


# The variant the ranking opens on. NO LONGER A GUESS.
#
# Calibrated 2026-09-01 against 874 round trips over 3 sessions, 40
# ticker-days. Nothing cleared |rho| >= 0.5 -- the best was +0.474 -- but at
# |rho| >= 0.3 there were 136 against 14 expected by chance, so the signal is
# real and diffuse rather than absent. The top six:
#
#     ratio_tw_mid_5s_rms      +0.474
#     ratio_ask_side_5s_rms    +0.468
#     ratio_last_mid_5s_rms    +0.467
#     ratio_bid_side_5s_rms    +0.463
#     ratio_last_mid_10s_rms   +0.463
#     ratio_ask_side_10s_rms   +0.457
#
# WHAT THAT SETTLES. All six are rms. Five different midpoint definitions sit
# within 0.017 of each other, which is far inside the noise at n=40 -- so the
# STATISTIC separates and the VARIANT does not, and the ordering below
# (statistic first, variant second) is what the evidence supports rather than
# what seemed sensible. The horizon preference moves from 10s to 5s on the
# same evidence: both 5s entries outrank their 10s counterparts.
#
# This is still 40 ticker-days. It is the best available answer, not a
# settled one, and the calibration panel is where it gets revisited.
_NOISE_PREFERENCE = ("tw_mid", "last_mid", "trade_price")
# median is ranked BELOW an unrecognised statistic, on purpose. Everything else
# here is taste; this one is a defect. Being unfamiliar is a reason to look at
# a number, whereas median is known to read 0.0 on exactly the names the filter
# is supposed to exclude.
_STAT_PREFERENCE = ("rms", "p75", "p90", "mean")
_STAT_LAST = ("median",)
# 5s, from the calibration above. Nearest-wins rather than exact, so a
# pipeline that stops emitting 5s degrades to the closest horizon it does
# emit instead of falling through to whatever sorts first alphabetically.
_PREFERRED_HORIZON_S = 5


def _default_noise(available: list[str]) -> str | None:
    if not available:
        return None

    # THE PIN, IF THE PIPELINE HAS ONE. config.INTRADAY_NOISE_COLUMN is the
    # variant intraday_metrics is built around, and the ranked table has to
    # agree with it — two panels on one page disagreeing about which noise
    # definition they mean is the same class of defect as two copies of a
    # baseline, which produced two divergent z estimators on the IV page.
    #
    # Read from the vendored config rather than restated, so it cannot drift;
    # check_vendored.py diffs that file against the pipeline's copy. The
    # preference ordering below stays as the fallback for a config without a
    # pin, and for a pin naming a column this date does not carry.
    pinned = getattr(scalp_config, "INTRADAY_NOISE_COLUMN", None)
    if pinned and pinned in available:
        return pinned
    def score(metric: str):
        p = _parse_variant(metric) or {}
        v = p.get("variant") or ""
        s = p.get("statistic") or ""
        if s in _STAT_PREFERENCE:
            stat_rank = _STAT_PREFERENCE.index(s)
        elif s in _STAT_LAST:
            stat_rank = 99
        else:
            stat_rank = 50
        return (
            # STATISTIC OUTRANKS VARIANT. Which midpoint definition is used is
            # a preference; which statistic summarises it is the difference
            # between a number and a zero, so it decides first.
            stat_rank,
            _NOISE_PREFERENCE.index(v) if v in _NOISE_PREFERENCE else 99,
            abs((p.get("horizon_s") or 0) - _PREFERRED_HORIZON_S),
            metric,
        )
    return sorted(available, key=score)[0]


def _filter_block() -> dict:
    """Defaults and slider bounds, straight from the vendored pipeline config.

    Both halves matter. The defaults are the pipeline's current opinion; the
    ranges are the span a threshold can be dragged through, so a value can be
    moved against the data rather than typed blind. Every one of these is a
    READ-TIME filter: the pipeline stores every symbol whether it passes or
    not, and the rows that fail are the only evidence that can say whether a
    threshold is set correctly.
    """
    return {
        "defaults": dict(scalp_config.DEFAULT_FILTERS),
        "ranges": {k: {"min": lo, "max": hi, "step": st}
                   for k, (lo, hi, st) in scalp_config.FILTER_RANGES.items()},
        "keys": list(FILTER_KEYS),
        "ratio_guard": scalp_config.MIN_NOISE_BPS_FOR_RATIO,
        "read_time_only": True,
    }
