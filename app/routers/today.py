"""
Today page endpoints — intraday IV grid from raw parquet files.

GET /api/today/iv_grid?date=YYYY-MM-DD&expiration=YYYY-MM-DD&settlement=PM

    Builds a (quote_time × strike) IV grid for the chosen expiration.
    Strike range anchored to the opening underlying_price:
      floor-to-nearest-100, 10 strikes below and 5 above (15 total).

    OTM boundary is fixed at opening spot:
      put  IV for strikes < open_spot
      call IV for strikes >= open_spot

    Also returns:
      spot_series: [{time, price}] for the SPX sparkline
      prev:        {strike_str: iv} from the prior trading day's 16:00 close
"""
import os
import re
from datetime import date as date_type

from fastapi import APIRouter, Depends, Query, HTTPException
from app.db import get_pool

router = APIRouter(tags=["today"])

def _init_parquet_bases() -> list[str]:
    env = os.getenv("PARQUET_BASES", "")
    if env:
        return [p.strip() for p in env.split(",") if p.strip()]
    single = os.getenv("PARQUET_BASE", "")
    if single:
        return [single]
    return ["/mnt/volume1/spx_options", "/mnt/volume2/spx_options"]

PARQUET_BASES = _init_parquet_bases()


def _find_parquet(*rel_parts: str) -> str | None:
    """Return the full path of the first existing parquet file across all bases."""
    for base in PARQUET_BASES:
        path = os.path.join(base, *rel_parts)
        if os.path.isfile(path):
            return path
    return None


def _duckdb():
    try:
        import duckdb
        return duckdb
    except ImportError:
        raise HTTPException(501, "duckdb not installed on this server")


def _validate_date(d: str) -> str:
    """Accept YYYY-MM-DD or YYYYMMDD, return YYYYMMDD."""
    clean = d.replace("-", "")
    if not re.match(r"^\d{8}$", clean):
        raise HTTPException(400, f"Invalid date: {d!r}")
    return clean


def _query(duckdb, sql: str):
    con = duckdb.connect()
    try:
        return con.execute(sql).fetchall()
    except Exception as e:
        raise HTTPException(500, f"DuckDB error: {e}")
    finally:
        con.close()


@router.get("/iv_grid")
async def get_iv_grid(
    date:         str  = Query(..., description="Trade date YYYY-MM-DD"),
    expiration:   str  = Query(..., description="Expiration date YYYY-MM-DD"),
    settlement:   str  = Query("PM"),
    filter_flags: bool = Query(False),
) -> dict:
    duck       = _duckdb()
    settlement = settlement.upper()
    if settlement not in ("AM", "PM"):
        raise HTTPException(400, "settlement must be AM or PM")

    date_folder = _validate_date(date)
    exp_folder  = _validate_date(expiration)
    flag_sql    = "AND flag_any = false" if filter_flags else ""

    pq = _find_parquet(date_folder, exp_folder, f"{settlement}.parquet")
    if pq is None:
        raise HTTPException(404, f"No parquet data for {date} / {expiration} / {settlement}")

    # ── 1. Load all intraday rows ─────────────────────────────────────────────
    rows = _query(duck, f"""
        SELECT CAST(quote_time AS TIME) AS qt,
               strike, "right", implied_vol, underlying_price
        FROM read_parquet('{pq}')
        WHERE implied_vol IS NOT NULL AND implied_vol > 0
          {flag_sql}
        ORDER BY qt, strike
    """)

    if not rows:
        raise HTTPException(404, "Parquet file exists but contains no usable rows")

    # ── 2. Opening spot → 15 target strikes ──────────────────────────────────
    open_spot = float(rows[0][4])
    ref       = int(open_spot / 100) * 100          # floor to nearest 100
    # [ref-900, ref-800, …, ref]  +  [ref+100, …, ref+500]  =  15 strikes
    target_set = set(
        [ref - (9 - i) * 100 for i in range(10)] +
        [ref + j * 100        for j in range(1, 6)]
    )
    strike_list = sorted(target_set)

    # ── 3. Build intraday grid + spot series ──────────────────────────────────
    time_map:    dict[str, dict[str, float]] = {}
    spot_by_time: dict[str, float]           = {}

    for qt, strike, right, iv, underlying in rows:
        t = str(qt)[:5]                             # "HH:MM"
        s = int(strike)
        if s not in target_set:
            continue
        if right != ("P" if s < open_spot else "C"):
            continue
        time_map.setdefault(t, {})[str(s)] = float(iv)
        spot_by_time.setdefault(t, float(underlying))

    spot_series = [{"time": t, "price": p} for t, p in sorted(spot_by_time.items())]

    # ── 4. Previous trading day 16:00 reference ───────────────────────────────
    prev_data: dict[str, float] = {}
    try:
        all_prev: set[str] = set()
        for _base in PARQUET_BASES:
            if os.path.isdir(_base):
                all_prev.update(
                    f for f in os.listdir(_base)
                    if re.match(r"^\d{8}$", f) and f < date_folder
                )
        prev_dirs = sorted(all_prev)
        if prev_dirs:
            prev_pq = _find_parquet(prev_dirs[-1], exp_folder, f"{settlement}.parquet")
            if prev_pq is not None:
                prev_rows = _query(duck, f"""
                    WITH times AS (
                        SELECT DISTINCT CAST(quote_time AS TIME) AS qt
                        FROM read_parquet('{prev_pq}')
                    ),
                    snap AS (
                        SELECT qt FROM times
                        ORDER BY ABS(EXTRACT(EPOCH FROM qt)
                                   - EXTRACT(EPOCH FROM TIME '16:00:00'))
                        LIMIT 1
                    )
                    SELECT p.strike, p."right", p.implied_vol
                    FROM read_parquet('{prev_pq}') p, snap s
                    WHERE CAST(p.quote_time AS TIME) = s.qt
                      AND p.implied_vol IS NOT NULL AND p.implied_vol > 0
                      {flag_sql}
                    ORDER BY p.strike
                """)
                for strike, right, iv in prev_rows:
                    s = int(strike)
                    if s not in target_set:
                        continue
                    if right != ("P" if s < open_spot else "C"):
                        continue
                    prev_data[str(s)] = float(iv)
    except HTTPException:
        raise
    except Exception:
        pass   # prev-day reference is best-effort

    return {
        "strikes":     strike_list,
        "spot_series": spot_series,
        "rows":        [{"time": t, "data": time_map[t]} for t in sorted(time_map)],
        "prev":        prev_data,
    }


# ── SPX / VIX scatter ────────────────────────────────────────────────────────

@router.get("/scatter")
async def get_spx_vix_scatter(
    days:     int = Query(30, ge=10, le=90),
    end_date: str = Query(None, description="End date YYYY-MM-DD; defaults to most recent trading day"),
    pool=Depends(get_pool),
) -> dict:
    """
    SPX daily return and VIX/VIX9D/VIX3M daily change for the last N trading days
    up to end_date.
    End/current date: latest snapshot where vix_close > 0 (handles VIX API lag).
    Prior dates: 15:45 snapshot; skipped if that slice is missing or spx_close = 0.
    Returns points oldest-first for gradient rendering.
    """
    if end_date:
        try:
            end_dt = date_type.fromisoformat(end_date)
        except ValueError:
            raise HTTPException(400, f"Invalid end_date: {end_date}")
    else:
        end_dt = date_type(9999, 12, 31)   # no upper bound

    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            WITH trade_dates AS (
                -- Prior days: in spx_surface (trading calendar), <= end_date,
                -- and must have a valid 15:45 row in index_ohlc.
                -- Current/end day: most recent spx_surface trading day <= end_date
                -- that has ANY index_ohlc data (no 15:45 requirement — may be intraday).
                SELECT trade_date FROM (
                    SELECT DISTINCT s.trade_date
                    FROM spx_surface s
                    WHERE s.trade_date <= $2
                      AND EXISTS (
                          SELECT 1 FROM index_ohlc i
                          WHERE i.trade_date = s.trade_date
                            AND i.quote_time  = TIME '15:45:00'
                            AND i.spx_close   > 0
                      )
                    UNION
                    SELECT MAX(s2.trade_date)
                    FROM spx_surface s2
                    WHERE s2.trade_date <= $2
                      AND EXISTS (
                          SELECT 1 FROM index_ohlc i2
                          WHERE i2.trade_date = s2.trade_date
                      )
                ) td
                ORDER BY trade_date DESC
                LIMIT $1 + 1
            ),
            current_day AS (
                SELECT MAX(trade_date) AS d FROM trade_dates
            ),
            ranked AS (
                SELECT
                    i.trade_date,
                    i.spx_close, i.vix_close, i.vix9d_close, i.vix3m_close,
                    ROW_NUMBER() OVER (
                        PARTITION BY i.trade_date
                        ORDER BY
                            CASE WHEN i.trade_date = (SELECT d FROM current_day)
                                 THEN -EXTRACT(EPOCH FROM i.quote_time)
                                 ELSE ABS(EXTRACT(EPOCH FROM (i.quote_time - TIME '15:45:00')))
                            END
                    ) AS rn
                FROM index_ohlc i
                WHERE i.trade_date IN (SELECT trade_date FROM trade_dates)
                  -- Current day: skip snapshots where VIX hasn't arrived yet (vix_close = 0)
                  AND (i.trade_date != (SELECT d FROM current_day) OR i.vix_close > 0)
            ),
            daily AS (
                SELECT trade_date, spx_close, vix_close, vix9d_close, vix3m_close
                FROM ranked WHERE rn = 1
            ),
            lagged AS (
                SELECT *,
                    LAG(spx_close)   OVER (ORDER BY trade_date) AS prev_spx,
                    LAG(vix_close)   OVER (ORDER BY trade_date) AS prev_vix,
                    LAG(vix9d_close) OVER (ORDER BY trade_date) AS prev_vix9d,
                    LAG(vix3m_close) OVER (ORDER BY trade_date) AS prev_vix3m
                FROM daily
            ),
            returns AS (
                SELECT
                    trade_date,
                    CASE WHEN prev_spx > 0
                         THEN (spx_close - prev_spx) / prev_spx
                         ELSE NULL END                         AS spx_return,
                    vix_close   - prev_vix                     AS vix_change,
                    vix9d_close - prev_vix9d                   AS vix9d_change,
                    vix3m_close - prev_vix3m                   AS vix3m_change
                FROM lagged
                WHERE prev_spx IS NOT NULL
            )
            SELECT trade_date, spx_return, vix_change, vix9d_change, vix3m_change
            FROM returns
            ORDER BY trade_date DESC
            LIMIT $1
            """,
            days,
            end_dt,
        )

    points = [
        {
            "date":         str(r["trade_date"]),
            "spx_return":   float(r["spx_return"])   if r["spx_return"]   is not None else None,
            "vix_change":   float(r["vix_change"])   if r["vix_change"]   is not None else None,
            "vix9d_change": float(r["vix9d_change"]) if r["vix9d_change"] is not None else None,
            "vix3m_change": float(r["vix3m_change"]) if r["vix3m_change"] is not None else None,
        }
        for r in rows
    ]
    return {"points": list(reversed(points))}  # oldest-first for gradient
