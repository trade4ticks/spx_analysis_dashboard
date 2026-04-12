"""
Term slope endpoint — annualized forward volatility.

For a fixed put_delta and two DTEs (a < b), the forward variance
between the two maturities is

    fwd_var = (IV_b^2 * T_b - IV_a^2 * T_a) / (T_b - T_a)

where T = DTE / 365. The annualized forward vol is sqrt(fwd_var).
This is the implied vol the market is pricing for the period
*between* the two expirations, in the same units as IV everywhere
else in the dashboard.

Returns null when forward variance comes out negative (arbitrage
violation between the two quoted IVs).
"""
from datetime import date as date_type, time as time_type
from typing import Optional

from fastapi import APIRouter, Depends, Query, HTTPException
from app.db import get_pool

router = APIRouter(tags=["term_slope"])

INTRADAY_MAX_DAYS = 90


def _parse_ints(s: str) -> list[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def _delta_label(pd: int) -> str:
    if pd == 50:
        return "ATM"
    if pd < 50:
        return f"{pd}Δp"
    return f"{100 - pd}Δc"


@router.get("")
async def get_term_slope(
    delta:       Optional[int] = Query(None),
    deltas:      Optional[str] = Query(None),
    dte_a:       int = Query(7),
    dte_b:       int = Query(30),
    start:       str = Query(...),
    end:         str = Query(...),
    target_time: str = Query("15:45"),
    freq:        str = Query("daily"),
    pool=Depends(get_pool),
) -> dict:
    if deltas:
        delta_list = _parse_ints(deltas)
    elif delta is not None:
        delta_list = [delta]
    else:
        raise HTTPException(400, "Provide delta or deltas")

    if dte_a == dte_b:
        raise HTTPException(400, "dte_a and dte_b must differ")

    start_d = date_type.fromisoformat(start)
    end_d   = date_type.fromisoformat(end)
    if freq == "intraday" and (end_d - start_d).days > INTRADAY_MAX_DAYS:
        raise HTTPException(400, f"Intraday limited to {INTRADAY_MAX_DAYS} days")

    pair = [dte_a, dte_b]
    T_a  = dte_a / 365.0
    T_b  = dte_b / 365.0
    dT   = T_b - T_a

    # For delta=50, substitute true forward ATM IV from spx_atm.
    async with pool.acquire() as conn:
        if freq == "daily":
            rows = await conn.fetch(
                """
                WITH closest_raw AS (
                    SELECT DISTINCT ON (trade_date, dte, put_delta)
                        trade_date, dte, put_delta, iv
                    FROM spx_surface
                    WHERE dte        = ANY($1)
                      AND put_delta  = ANY($2)
                      AND trade_date BETWEEN $3 AND $4
                    ORDER BY trade_date, dte, put_delta,
                             ABS(EXTRACT(EPOCH FROM (quote_time - $5::time)))
                ),
                atm AS (
                    SELECT DISTINCT ON (trade_date)
                        trade_date, atm_iv
                    FROM spx_atm
                    WHERE trade_date BETWEEN $3 AND $4
                    ORDER BY trade_date,
                             ABS(EXTRACT(EPOCH FROM (quote_time - $5::time)))
                ),
                closest AS (
                    SELECT c.trade_date, c.dte, c.put_delta,
                           CASE WHEN c.put_delta = 50
                                THEN COALESCE(a.atm_iv, c.iv)
                                ELSE c.iv END AS iv
                    FROM closest_raw c
                    LEFT JOIN atm a ON a.trade_date = c.trade_date
                )
                SELECT trade_date::text AS label, put_delta,
                       a.iv AS iv_a, b.iv AS iv_b,
                       (b.iv * b.iv * $9 - a.iv * a.iv * $8) / $10 AS fwd_var,
                       CASE WHEN (b.iv * b.iv * $9 - a.iv * a.iv * $8) / $10 > 0
                            THEN SQRT((b.iv * b.iv * $9 - a.iv * a.iv * $8) / $10)
                            ELSE NULL END AS fwd_vol
                FROM closest a
                JOIN closest b USING (trade_date, put_delta)
                WHERE a.dte = $6 AND b.dte = $7
                ORDER BY trade_date, put_delta
                """,
                pair, delta_list, start_d, end_d,
                time_type.fromisoformat(target_time),
                dte_a, dte_b, T_a, T_b, dT,
            )
        else:
            rows = await conn.fetch(
                """
                WITH base_raw AS (
                    SELECT trade_date, quote_time, dte, put_delta, iv
                    FROM spx_surface
                    WHERE dte        = ANY($1)
                      AND put_delta  = ANY($2)
                      AND trade_date BETWEEN $3 AND $4
                ),
                base AS (
                    SELECT r.trade_date, r.quote_time, r.dte, r.put_delta,
                           CASE WHEN r.put_delta = 50
                                THEN COALESCE(a.atm_iv, r.iv)
                                ELSE r.iv END AS iv
                    FROM base_raw r
                    LEFT JOIN spx_atm a
                           ON a.trade_date = r.trade_date AND a.quote_time = r.quote_time
                )
                SELECT (a.trade_date::text || ' ' || LEFT(a.quote_time::text, 5)) AS label,
                       a.put_delta, a.iv AS iv_a, b.iv AS iv_b,
                       (b.iv * b.iv * $8 - a.iv * a.iv * $7) / $9 AS fwd_var,
                       CASE WHEN (b.iv * b.iv * $8 - a.iv * a.iv * $7) / $9 > 0
                            THEN SQRT((b.iv * b.iv * $8 - a.iv * a.iv * $7) / $9)
                            ELSE NULL END AS fwd_vol
                FROM base a
                JOIN base b USING (trade_date, quote_time, put_delta)
                WHERE a.dte = $5 AND b.dte = $6
                ORDER BY trade_date, quote_time, put_delta
                """,
                pair, delta_list, start_d, end_d, dte_a, dte_b, T_a, T_b, dT,
            )

    seen = []
    seen_set = set()
    for r in rows:
        if r["label"] not in seen_set:
            seen.append(r["label"]); seen_set.add(r["label"])

    bucket: dict[int, dict[str, dict]] = {d: {} for d in delta_list}
    for r in rows:
        bucket[r["put_delta"]][r["label"]] = {
            "value":   r["fwd_vol"],
            "fwd_var": r["fwd_var"],
            "iv_a":    r["iv_a"],
            "iv_b":    r["iv_b"],
        }

    series = []
    for d in delta_list:
        entries = bucket[d]
        series.append({
            "label":   _delta_label(d),
            "delta":   d,
            "labels":  seen,
            "values":  [entries.get(k, {}).get("value") for k in seen],
            "metrics": [
                ({
                    "iv_a":    entries.get(k, {}).get("iv_a"),
                    "iv_b":    entries.get(k, {}).get("iv_b"),
                    "fwd_var": entries.get(k, {}).get("fwd_var"),
                } if entries.get(k) else None)
                for k in seen
            ],
        })

    return {
        "freq":      freq,
        "dimension": "delta",
        "dte_a":     dte_a,
        "dte_b":     dte_b,
        "series":    series,
    }
