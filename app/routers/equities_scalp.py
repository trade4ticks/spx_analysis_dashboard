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
        "note":
            "Spearman rank correlation across ticker-days. Read the "
            "expected-by-chance column first: with this many metrics and this "
            "few ticker-days, a list sorted by |rho| will always have "
            "something at the top. What matters is whether the same metric "
            "stays there as sessions accumulate — not its rank today.",
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
        out.append({"symbol": r["symbol"], "values": vals,
                    "passes": not fails, "fails": fails,
                    "spark": spark.get(r["symbol"])})

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
    }


# The variant the ranking opens on until calibration says otherwise.
#
# NOT the median statistic, and this is the one place a preference is
# expressed. The median collapses to exactly 0.0 on a sparse-quote name -- when
# more than half of consecutive buckets carry an identical midpoint, the median
# change is zero by construction -- which makes the ratio infinite and sorts
# the least tradeable names to the top. rms is the same measurement without
# that failure.
#
# Written as a PREFERENCE ORDER over what the database actually has, not as a
# literal. If the preferred name is gone, the page opens on something real
# rather than on a column of nulls.
_NOISE_PREFERENCE = ("tw_mid", "last_mid", "trade_price")
# median is ranked BELOW an unrecognised statistic, on purpose. Everything else
# here is taste; this one is a defect. Being unfamiliar is a reason to look at
# a number, whereas median is known to read 0.0 on exactly the names the filter
# is supposed to exclude.
_STAT_PREFERENCE = ("rms", "p75", "p90", "mean")
_STAT_LAST = ("median",)


def _default_noise(available: list[str]) -> str | None:
    if not available:
        return None
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
            abs((p.get("horizon_s") or 0) - 10),   # 10s, nearest first
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
