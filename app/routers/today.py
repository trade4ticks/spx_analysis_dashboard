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

from fastapi import APIRouter, Query, HTTPException

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
