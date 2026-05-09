"""
Feature clustering and 2-factor interaction scanning.

Phase 1: cluster features by score-vector similarity (Spearman of composite_score
         vectors across ticker×fwd_ret combinations).
Phase 2: run scan_interaction_2f for all cross-family representative pairs,
         store results in oi_interaction_matrix.
"""
import asyncio
import json
from datetime import datetime
from collections import defaultdict

import numpy as np
from scipy import stats as sp_stats

from research import scanner

_TABLE_DDL = """\
CREATE TABLE IF NOT EXISTS oi_interaction_matrix (
    id                          SERIAL PRIMARY KEY,
    feat_a                      TEXT NOT NULL,
    feat_b                      TEXT NOT NULL,
    fwd_ret                     TEXT NOT NULL,
    ticker                      TEXT NOT NULL,
    composite_interaction_score REAL,
    interaction_lift            REAL,
    best_quadrant               TEXT,
    best_quad_sharpe            REAL,
    best_quad_avg_ret           REAL,
    best_quad_win_rate          REAL,
    best_quad_n                 INTEGER,
    r2_gain                     REAL,
    ols_r2                      REAL,
    n                           INTEGER,
    quadrants                   JSONB,
    scanned_at                  TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (feat_a, feat_b, fwd_ret, ticker)
);
CREATE INDEX IF NOT EXISTS idx_oi_int_feat ON oi_interaction_matrix(feat_a, feat_b);
CREATE INDEX IF NOT EXISTS idx_oi_int_score ON oi_interaction_matrix(composite_interaction_score DESC);
"""

_UPSERT_2F = """\
INSERT INTO oi_interaction_matrix
    (feat_a, feat_b, fwd_ret, ticker,
     composite_interaction_score, interaction_lift,
     best_quadrant, best_quad_sharpe, best_quad_avg_ret, best_quad_win_rate, best_quad_n,
     r2_gain, ols_r2, n, quadrants, scanned_at)
VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15,NOW())
ON CONFLICT (feat_a, feat_b, fwd_ret, ticker) DO UPDATE SET
    composite_interaction_score=EXCLUDED.composite_interaction_score,
    interaction_lift=EXCLUDED.interaction_lift,
    best_quadrant=EXCLUDED.best_quadrant,
    best_quad_sharpe=EXCLUDED.best_quad_sharpe,
    best_quad_avg_ret=EXCLUDED.best_quad_avg_ret,
    best_quad_win_rate=EXCLUDED.best_quad_win_rate,
    best_quad_n=EXCLUDED.best_quad_n,
    r2_gain=EXCLUDED.r2_gain,
    ols_r2=EXCLUDED.ols_r2,
    n=EXCLUDED.n,
    quadrants=EXCLUDED.quadrants,
    scanned_at=NOW()
"""

_progress = {'running': False, 'message': '', 'last_run': None}


def get_progress() -> dict:
    return dict(_progress)


async def ensure_table(pool):
    async with pool.acquire() as conn:
        await conn.execute(_TABLE_DDL)


async def compute_clusters(main_pool, threshold: float = 0.85) -> list[dict]:
    """
    Cluster features by score-vector similarity.
    Features whose composite_score vectors (across ticker×fwd_ret) have
    Spearman |r| >= threshold are treated as sisters.
    Returns list of dicts: {cluster_id, representative, metrics, avg_scores}.
    """
    async with main_pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT metric, ticker, fwd_ret, composite_score FROM oi_score_matrix")

    score_map: dict[str, dict] = defaultdict(dict)
    for r in rows:
        key = (r['ticker'], r['fwd_ret'])
        score_map[r['metric']][key] = float(r['composite_score'] or 0)

    metrics = sorted(score_map.keys())
    if not metrics:
        return []

    # Use keys common to all metrics; fall back to union with zero-fill
    all_key_sets = [set(v.keys()) for v in score_map.values()]
    common_keys = sorted(set.intersection(*all_key_sets)) if all_key_sets else []
    if len(common_keys) < 5:
        common_keys = sorted(set(k for v in score_map.values() for k in v.keys()))

    vectors = {m: np.array([score_map[m].get(k, 0.0) for k in common_keys])
               for m in metrics}
    avg_scores = {m: float(np.mean(list(score_map[m].values()))) for m in metrics}

    sorted_metrics = sorted(metrics, key=lambda m: avg_scores[m], reverse=True)
    assigned: set[str] = set()
    clusters = []

    for m in sorted_metrics:
        if m in assigned:
            continue
        cluster = [m]
        assigned.add(m)
        vm = vectors[m]
        for other in sorted_metrics:
            if other in assigned:
                continue
            vo = vectors[other]
            if len(vm) >= 5 and len(vo) >= 5:
                r_val, _ = sp_stats.spearmanr(vm, vo)
                if abs(float(r_val)) >= threshold:
                    cluster.append(other)
                    assigned.add(other)

        rep = max(cluster, key=lambda x: avg_scores[x])
        clusters.append({
            'cluster_id':    len(clusters),
            'representative': rep,
            'metrics':       sorted(cluster, key=lambda x: avg_scores[x], reverse=True),
            'avg_scores':    {x: round(avg_scores[x], 1) for x in cluster},
        })

    return clusters


async def run_2f_scan(oi_pool, main_pool, log=print):
    """
    Batch 2-factor interaction scan across all cross-family representative pairs.
    Stores results in oi_interaction_matrix.
    """
    _progress['running'] = True
    _progress['message'] = 'Computing feature clusters...'

    try:
        await ensure_table(main_pool)
        clusters = await compute_clusters(main_pool)
        representatives = [c['representative'] for c in clusters]
        log(f'Clusters: {len(clusters)}, representatives: {representatives}')

        if len(representatives) < 2:
            _progress['message'] = 'Not enough clusters to scan (need >= 2).'
            return

        cross_pairs = [(representatives[i], representatives[j])
                       for i in range(len(representatives))
                       for j in range(i + 1, len(representatives))]
        log(f'Cross-family pairs: {len(cross_pairs)}')

        # Pre-load single-factor Sharpe baselines from score matrix
        async with main_pool.acquire() as conn:
            baseline_rows = await conn.fetch(
                "SELECT ticker, metric, fwd_ret, best_sharpe FROM oi_score_matrix")
            ticker_fwd_rows = await conn.fetch(
                "SELECT DISTINCT ticker, fwd_ret FROM oi_score_matrix ORDER BY ticker, fwd_ret")

        baseline_map = {(r['ticker'], r['metric'], r['fwd_ret']): float(r['best_sharpe'] or 0)
                        for r in baseline_rows}
        ticker_fwd_list = [(r['ticker'], r['fwd_ret']) for r in ticker_fwd_rows]
        unique_tickers = sorted({t for t, _ in ticker_fwd_list})

        total = len(cross_pairs) * len(unique_tickers)
        done = 0

        for feat_a, feat_b in cross_pairs:
            for ticker in unique_tickers:
                _progress['message'] = (
                    f'{done}/{total}: {feat_a} x {feat_b} -- {ticker}')

                async with oi_pool.acquire() as conn:
                    data_rows = await conn.fetch(
                        "SELECT * FROM daily_features WHERE ticker = $1 "
                        "AND trade_date >= '2020-01-01' ORDER BY trade_date",
                        ticker)
                rows_data = [dict(r) for r in data_rows]
                if len(rows_data) < 60:
                    done += 1
                    continue

                fwd_rets = sorted({fwd for t, fwd in ticker_fwd_list if t == ticker})
                batch = []

                for fwd_ret in fwd_rets:
                    baseline = max(
                        baseline_map.get((ticker, feat_a, fwd_ret), 0.0),
                        baseline_map.get((ticker, feat_b, fwd_ret), 0.0),
                    )
                    try:
                        result = scanner.scan_interaction_2f(
                            rows_data, feat_a, feat_b, fwd_ret, ticker, baseline)
                    except Exception as e:
                        log(f'  2F error {feat_a}x{feat_b}/{fwd_ret}/{ticker}: {e}')
                        continue
                    if result is None:
                        continue

                    bq = result.get('best_quadrant') or {}
                    batch.append((
                        feat_a, feat_b, fwd_ret, ticker,
                        result.get('composite_interaction_score'),
                        result.get('interaction_lift'),
                        bq.get('label'),
                        bq.get('sharpe'),
                        bq.get('avg_ret'),
                        bq.get('win_rate'),
                        bq.get('n'),
                        result.get('r2_gain'),
                        result.get('ols_r2'),
                        result.get('n'),
                        json.dumps(result.get('quadrants', [])),
                    ))

                if batch:
                    async with main_pool.acquire() as conn:
                        await conn.executemany(_UPSERT_2F, batch)
                    log(f'  saved {len(batch)} rows for {feat_a}x{feat_b}/{ticker}')

                done += 1

        _progress['message'] = f'Complete: {done} ticker combos scanned'
        _progress['last_run'] = datetime.utcnow().isoformat()
        log('2F scan done.')

    except Exception as exc:
        _progress['message'] = f'Error: {exc}'
        log(f'2F SCAN ERROR: {exc}')
        raise
    finally:
        _progress['running'] = False
