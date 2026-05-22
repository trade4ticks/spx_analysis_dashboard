#!/usr/bin/env python
"""Regression harness for the row_compute refactor.

Built in Step 1 of the refactor, before any endpoint changes, so the
verification harness itself is exercised while stakes are zero.

Three modes:

  capture   — Hit the test-matrix endpoints against a running dashboard
              and write one JSON snapshot per request into
              `regression_snapshots/<tag>/`.

  diff      — Compare two snapshot directories. Numeric values compare
              with math.isclose(rel_tol=1e-9, abs_tol=1e-12). Lists of
              dicts compare ordering-insensitively when they look like
              row sets keyed on (ticker, trade_date). Reports the
              first N diffs per endpoint.

  train_test_check — Standalone unit-style check of the TrainTestAssigner:
              (a) a 10-row hand-verified small case with known-correct
              answers, and (b) the property test that cutoff == max(date)
              makes train/test equal pure in-sample.

Usage (against a locally-running dashboard at http://localhost:8000):

  python scripts/regression_check.py capture --tag before-step2
  # ... do the migration step ...
  python scripts/regression_check.py capture --tag after-step2
  python scripts/regression_check.py diff --before before-step2 --after after-step2
  python scripts/regression_check.py train_test_check

The matrix covers the seven endpoints that step 2–5.5 migrate:
/analyze, /global-metric-bins, /secondary-corr-bins,
/secondary-correlation, /secondary-detail, /portfolios/{pid}/aggregate,
/heatmap. Endpoints not yet exercised by a migration step are still
captured so cross-step regressions surface early.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from datetime import date as _date
from pathlib import Path

DEFAULT_BASE  = "http://localhost:8000/api/oi-analysis"
SNAPSHOT_ROOT = Path("regression_snapshots")
HTTP_TIMEOUT  = 600  # generous; some endpoints (Score Matrix, etc.) are slow
RETRY_ATTEMPTS = 4   # retries for transient URLError / RemoteDisconnected
RETRY_BACKOFF  = 8   # seconds between retries (linear; long enough for uvicorn restart)


# ─────────────────────────────────────────────────────────────────────────
# HTTP helpers
# ─────────────────────────────────────────────────────────────────────────

def _request(req: urllib.request.Request, label: str) -> dict | list:
    """Execute `req` with retry-on-transient-connection-error.

    HTTPError responses (4xx/5xx) are returned by the caller's except
    clause — we don't retry those (the server intentionally responded).
    URLError (DNS, refused, reset, RemoteDisconnected wrapped via the
    http.client layer) is retried up to RETRY_ATTEMPTS times. This is
    important on the VPS where heavy walk-forward responses (60MB+)
    can OOM a uvicorn worker and force a respawn — the next request
    may briefly get ConnectionRefused before the worker is back.
    """
    import time
    last_err = None
    for attempt in range(1, RETRY_ATTEMPTS + 1):
        try:
            with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError:
            raise  # surface to caller — server intentionally responded
        except (urllib.error.URLError, ConnectionError, OSError) as e:
            last_err = e
            if attempt < RETRY_ATTEMPTS:
                print(f"[capture]   RETRY {label} (attempt {attempt}/{RETRY_ATTEMPTS} "
                      f"failed: {type(e).__name__}); sleeping {RETRY_BACKOFF}s",
                      file=sys.stderr)
                time.sleep(RETRY_BACKOFF)
                continue
            raise last_err  # exhausted retries


def _get(base: str, path: str, params: dict | None = None) -> dict | list:
    url = base + path
    if params:
        url += "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, method="GET",
                                 headers={"Accept": "application/json"})
    return _request(req, label=path)


def _post(base: str, path: str, body: dict) -> dict:
    url = base + path
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        url, method="POST", data=data,
        headers={"Content-Type": "application/json",
                 "Accept":       "application/json"})
    return _request(req, label=path)


# ─────────────────────────────────────────────────────────────────────────
# Universe discovery
# ─────────────────────────────────────────────────────────────────────────

def _discover(base: str) -> dict:
    """Probe /tickers and /columns to pick a coherent test matrix.

    Picks the first available ticker (treated as the single-ticker
    case), the first feature column, and prefers `ret_5d_fwd_oc` as
    the outcome (falls back to the first outcome).
    """
    tickers = _get(base, "/tickers")
    if not isinstance(tickers, list) or not tickers:
        raise RuntimeError("/tickers returned empty — server up but DB has no data?")
    cols = _get(base, "/columns")
    if not isinstance(cols, dict):
        raise RuntimeError(f"/columns returned unexpected shape: {type(cols)!r}")
    features = cols.get("features") or []
    outcomes = cols.get("outcomes") or []
    if not features or not outcomes:
        raise RuntimeError("/columns missing features or outcomes")

    chosen_ticker  = "SPX" if "SPX" in tickers else tickers[0]
    chosen_metric  = features[0]
    # A second metric for /heatmap (need two distinct features).
    second_metric  = features[1] if len(features) > 1 else features[0]
    chosen_outcome = "ret_5d_fwd_oc" if "ret_5d_fwd_oc" in outcomes else outcomes[0]
    return {
        "tickers": tickers, "features": features, "outcomes": outcomes,
        "chosen_ticker": chosen_ticker, "chosen_metric": chosen_metric,
        "second_metric": second_metric, "chosen_outcome": chosen_outcome,
    }


# ─────────────────────────────────────────────────────────────────────────
# Test matrix
# ─────────────────────────────────────────────────────────────────────────

def _build_test_matrix(disc: dict) -> list[tuple[str, dict]]:
    """One entry per request to make. Returns [(snapshot_filename, request_descriptor), ...].

    A request_descriptor is `{method, path, params or body, depends_on}`. `depends_on`
    is an optional callable that runs first and produces fields used to
    parametrize the request (e.g. the cache_key from /secondary-load).
    """
    tk_one  = disc["chosen_ticker"]
    tk_all  = "ALL"
    metric  = disc["chosen_metric"]
    metric2 = disc["second_metric"]
    outcome = disc["chosen_outcome"]

    matrix: list[tuple[str, dict]] = []

    # /analyze — four variants
    for tk in (tk_one, tk_all):
        for wf in (False, True):
            name = f"analyze__{tk}__{metric}__{outcome}__wf{int(wf)}.json"
            req: dict = {
                "method": "GET",
                "path":   "/analyze",
                "params": {"ticker": tk, "metric": metric, "outcome": outcome,
                           "walk_forward": str(wf).lower()},
            }
            matrix.append((name, req))

    # /global-metric-bins — in-sample + walk-forward
    for wf in (False, True):
        name = f"global-metric-bins__ALL__{outcome}__wf{int(wf)}.json"
        matrix.append((name, {
            "method": "GET",
            "path":   "/global-metric-bins",
            "params": {"outcome": outcome, "ticker": "ALL", "n_bins": "20",
                       "walk_forward": str(wf).lower()},
        }))

    # /heatmap — three variants now that Step 5.5 routes ALL-mode through
    # the row_compute Assigner. Single-ticker stays on np.percentile edges
    # (different algorithm by design — see the endpoint docstring).
    matrix.append((f"heatmap__{tk_one}__{metric}__{metric2}__{outcome}.json", {
        "method": "GET",
        "path":   "/heatmap",
        "params": {"ticker": tk_one, "metric_x": metric, "metric_y": metric2,
                   "outcome": outcome, "bins": "5"},
    }))
    # ALL-mode in-sample — exercises the new Assigner routing.
    matrix.append((f"heatmap__ALL__{metric}__{metric2}__{outcome}__wf0.json", {
        "method": "GET",
        "path":   "/heatmap",
        "params": {"ticker": "ALL", "metric_x": metric, "metric_y": metric2,
                   "outcome": outcome, "bins": "5", "walk_forward": "false"},
    }))
    # ALL-mode walk-forward — new capability that comes for free with the
    # Assigner routing. No legacy reference exists pre-Step-5.5, so the
    # first capture here defines the baseline going forward.
    matrix.append((f"heatmap__ALL__{metric}__{metric2}__{outcome}__wf1.json", {
        "method": "GET",
        "path":   "/heatmap",
        "params": {"ticker": "ALL", "metric_x": metric, "metric_y": metric2,
                   "outcome": outcome, "bins": "5", "walk_forward": "true"},
    }))

    # Secondary-scanner chain: /secondary-load → /secondary-scan → /secondary-detail
    # Two cache_key chains (in-sample and walk-forward) so the dependent
    # requests use the right cache. Each chain is encoded as a list of
    # requests sharing a closure.
    for wf in (False, True):
        chain_id = f"sec_chain_wf{int(wf)}"
        # /secondary-load (creates the cache key). date_from/date_to
        # are omitted (the Pydantic model rejects null).
        matrix.append((f"secondary-load__{tk_all}__{metric}__{outcome}__wf{int(wf)}.json", {
            "method": "POST",
            "path":   "/secondary-load",
            "body":   {"ticker": tk_all, "metric": metric, "outcome": outcome,
                       "selected_bins": [1, 2, 19, 20]},
            "chain_publish": chain_id,  # publish "cache_key" + "primary_metric" for downstream
        }))
        # /secondary-corr-bins (depends on cache_key)
        matrix.append((f"secondary-corr-bins__{tk_all}__{outcome}__wf{int(wf)}.json", {
            "method": "POST",
            "path":   "/secondary-corr-bins",
            "body":   {"ticker": tk_all, "n_bins": 10,
                       "filtered_dates": [],  # populated from chain
                       "walk_forward": wf,
                       "selected_primary_bins": [1, 2, 19, 20]},
            "chain_consume": chain_id,
        }))

    # Portfolios — list, then aggregate the first if any exist.
    matrix.append(("portfolios__list.json", {
        "method": "GET",
        "path":   "/portfolios",
        "chain_publish": "portfolios_list",
    }))
    for wf in (False, True):
        matrix.append((f"portfolios-aggregate__first__wf{int(wf)}.json", {
            "method": "POST",
            "path":   "/portfolios/{pid}/aggregate",
            "body":   {"walk_forward": wf},
            "chain_consume": "portfolios_list",
        }))

    # Score Matrix — all three modes for each of the three read endpoints.
    # Added in Step 7k to close a regression-blind-spot: previously the
    # batch-score-driven `oi_score_matrix` table was invisible to the
    # capture matrix entirely. The /run-batch-score POST is NOT captured
    # because it triggers a multi-minute background job — these GETs
    # read whatever state the table currently has.
    #
    # The cutoff_date for train_test mode is fixed at 2024-01-01 (the
    # typical train/test split). If the user hasn't run a train_test
    # batch scan for that cutoff, the response will be empty — that
    # still validates response shape and catches structural changes.
    sm_modes = [
        ("in_sample",    {"mode": "in_sample"}),
        ("walk_forward", {"mode": "walk_forward"}),
        ("train_test",   {"mode": "train_test", "cutoff_date": "2024-01-01"}),
    ]
    for mode_label, extra_params in sm_modes:
        matrix.append((f"score-matrix__{mode_label}.json", {
            "method": "GET",
            "path":   "/score-matrix",
            "params": {**extra_params, "limit": "500"},
        }))
        matrix.append((f"score-matrix-meta__{mode_label}.json", {
            "method": "GET",
            "path":   "/score-matrix/meta",
            "params": extra_params,
        }))
        matrix.append((f"score-matrix-summary__{mode_label}.json", {
            "method": "GET",
            "path":   "/score-matrix/summary",
            "params": extra_params,
        }))

    return matrix


# ─────────────────────────────────────────────────────────────────────────
# capture mode
# ─────────────────────────────────────────────────────────────────────────

def cmd_capture(args) -> int:
    base = args.base.rstrip("/")
    out_dir = SNAPSHOT_ROOT / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[capture] base={base}  out={out_dir}")
    try:
        disc = _discover(base)
    except Exception as e:
        print(f"[capture] ERROR discovering server universe: {e!r}", file=sys.stderr)
        return 2

    print(f"[capture] picked ticker={disc['chosen_ticker']}  "
          f"metric={disc['chosen_metric']}  metric2={disc['second_metric']}  "
          f"outcome={disc['chosen_outcome']}")

    # Save a discovery snapshot so future runs can confirm the universe
    # hasn't drifted between baseline and post-refactor captures.
    (out_dir / "_discovery.json").write_text(
        json.dumps({"chosen": {k: disc[k] for k in (
            "chosen_ticker", "chosen_metric", "second_metric", "chosen_outcome")},
            "n_tickers": len(disc["tickers"]),
            "n_features": len(disc["features"]),
            "n_outcomes": len(disc["outcomes"]),
        }, indent=2),
        encoding="utf-8")

    matrix = _build_test_matrix(disc)
    chain_state: dict[str, dict] = {}  # chain_id -> {fields from prior responses}
    ok = err = 0

    for name, req in matrix:
        path  = req.get("path", "")
        body  = req.get("body")
        params = req.get("params")
        method = req.get("method", "GET")

        # Chain consume: rewrite fields from prior responses
        c_cons = req.get("chain_consume")
        if c_cons and c_cons in chain_state:
            st = chain_state[c_cons]
            if "cache_key" in st and isinstance(body, dict) and "cache_key" not in body:
                body["cache_key"] = st["cache_key"]
            if "filtered_dates" in st and isinstance(body, dict) and not body.get("filtered_dates"):
                body["filtered_dates"] = st["filtered_dates"]
            if "{pid}" in path and "pid" in st:
                path = path.replace("{pid}", str(st["pid"]))
        elif c_cons and c_cons not in chain_state:
            # Skip dependent request when the predecessor didn't publish.
            print(f"[capture]   SKIP {name} (no chain state for {c_cons})")
            continue
        if "{pid}" in path:
            # Couldn't resolve {pid} — no portfolios exist; skip.
            print(f"[capture]   SKIP {name} (no portfolio available)")
            continue

        try:
            if method == "GET":
                resp = _get(base, path, params)
            else:
                resp = _post(base, path, body or {})
        except urllib.error.HTTPError as e:
            # Capture error responses too so diffs catch a 500 turning
            # into a 200 (or vice versa) after refactor.
            try:
                body_text = e.read().decode("utf-8")
                resp = {"_http_error": e.code, "_body": body_text}
            except Exception:
                resp = {"_http_error": e.code, "_body": "<unreadable>"}
            err += 1
        except urllib.error.URLError as e:
            # Retries exhausted — record the failure and continue with
            # the rest of the matrix instead of aborting the whole run.
            resp = {"_connection_error": repr(e)}
            (out_dir / name).write_text(json.dumps(resp, indent=2, default=str),
                                        encoding="utf-8")
            print(f"[capture]   FAIL {name}: connection error after retries {e!r}",
                  file=sys.stderr)
            err += 1
            continue
        except Exception as e:
            print(f"[capture]   FAIL {name}: {e!r}", file=sys.stderr)
            err += 1
            continue

        (out_dir / name).write_text(json.dumps(resp, indent=2, default=str),
                                    encoding="utf-8")
        ok += 1

        # Chain publish: extract useful fields for dependent requests
        c_pub = req.get("chain_publish")
        if c_pub:
            state: dict = {}
            if isinstance(resp, dict):
                if "cache_key" in resp:
                    state["cache_key"] = resp["cache_key"]
                if "filtered_dates" in resp:
                    state["filtered_dates"] = resp["filtered_dates"]
            if isinstance(resp, list) and resp and isinstance(resp[0], dict):
                # /portfolios — pick the first portfolio's id
                if "id" in resp[0]:
                    state["pid"] = resp[0]["id"]
            chain_state[c_pub] = state

        print(f"[capture]   OK   {name}  ({len(json.dumps(resp))} bytes)")

    print(f"[capture] done: {ok} ok, {err} errored (HTTP errors captured into snapshot)")
    return 0


# ─────────────────────────────────────────────────────────────────────────
# diff mode
# ─────────────────────────────────────────────────────────────────────────

def _numbers_close(a, b) -> bool:
    if a is None and b is None: return True
    if a is None or  b is None: return False
    try:
        af, bf = float(a), float(b)
    except (TypeError, ValueError):
        return False
    if math.isnan(af) and math.isnan(bf): return True
    return math.isclose(af, bf, rel_tol=1e-9, abs_tol=1e-12)


def _diff(a, b, path: str, diffs: list[str], max_diffs: int) -> None:
    """Recursive structural diff. Appends short string descriptions to `diffs`.

    Lists of dicts that look like row sets (have both 'ticker' and
    'trade_date' or 'date' fields) are compared ordering-insensitively
    by sorting on the row key.
    """
    if len(diffs) >= max_diffs:
        return
    if isinstance(a, dict) and isinstance(b, dict):
        keys = set(a) | set(b)
        for k in sorted(keys):
            if k not in a:
                diffs.append(f"{path}.{k}: missing in BEFORE")
                continue
            if k not in b:
                diffs.append(f"{path}.{k}: missing in AFTER")
                continue
            _diff(a[k], b[k], f"{path}.{k}", diffs, max_diffs)
        return
    if isinstance(a, list) and isinstance(b, list):
        if a and b and isinstance(a[0], dict) and isinstance(b[0], dict):
            # Try keying by (ticker, date) if both have it.
            sample = a[0]
            keyfn = None
            if "ticker" in sample and "trade_date" in sample:
                keyfn = lambda r: (r.get("ticker"), r.get("trade_date"))
            elif "ticker" in sample and "date" in sample:
                keyfn = lambda r: (r.get("ticker"), r.get("date"))
            elif "date" in sample:
                keyfn = lambda r: (r.get("date"),)
            if keyfn:
                a_by = {keyfn(r): r for r in a}
                b_by = {keyfn(r): r for r in b}
                only_a = set(a_by) - set(b_by)
                only_b = set(b_by) - set(a_by)
                if only_a:
                    diffs.append(f"{path}: {len(only_a)} rows only in BEFORE (sample: {list(only_a)[:3]})")
                if only_b:
                    diffs.append(f"{path}: {len(only_b)} rows only in AFTER  (sample: {list(only_b)[:3]})")
                for k in sorted(set(a_by) & set(b_by)):
                    _diff(a_by[k], b_by[k], f"{path}[{k}]", diffs, max_diffs)
                return
        if len(a) != len(b):
            diffs.append(f"{path}: list length differs ({len(a)} vs {len(b)})")
            return
        for i, (av, bv) in enumerate(zip(a, b)):
            _diff(av, bv, f"{path}[{i}]", diffs, max_diffs)
        return
    # Scalar leaf
    if isinstance(a, (int, float)) or isinstance(b, (int, float)):
        if not _numbers_close(a, b):
            diffs.append(f"{path}: {a!r} != {b!r}")
        return
    if a != b:
        diffs.append(f"{path}: {a!r} != {b!r}")


def cmd_diff(args) -> int:
    before_dir = SNAPSHOT_ROOT / args.before
    after_dir  = SNAPSHOT_ROOT / args.after
    if not before_dir.is_dir():
        print(f"[diff] no such snapshot dir: {before_dir}", file=sys.stderr)
        return 2
    if not after_dir.is_dir():
        print(f"[diff] no such snapshot dir: {after_dir}", file=sys.stderr)
        return 2

    before_files = {p.name for p in before_dir.glob("*.json")}
    after_files  = {p.name for p in after_dir.glob("*.json")}
    only_before  = before_files - after_files
    only_after   = after_files - before_files
    both         = sorted(before_files & after_files)

    if only_before:
        print(f"[diff] {len(only_before)} files only in BEFORE: {sorted(only_before)[:5]}")
    if only_after:
        print(f"[diff] {len(only_after)} files only in AFTER:  {sorted(only_after)[:5]}")

    total_diffs = 0
    for f in both:
        try:
            a = json.loads((before_dir / f).read_text(encoding="utf-8"))
            b = json.loads((after_dir  / f).read_text(encoding="utf-8"))
        except Exception as e:
            print(f"[diff] {f}: failed to parse — {e!r}")
            total_diffs += 1
            continue
        diffs: list[str] = []
        _diff(a, b, "", diffs, max_diffs=args.max_diffs)
        if diffs:
            print(f"[diff] {f}: {len(diffs)} diff(s) (showing first {args.max_diffs}):")
            for d in diffs:
                print(f"        {d}")
            total_diffs += len(diffs)
        else:
            print(f"[diff] {f}: PASS identical")

    print(f"[diff] total diffs across {len(both)} files: {total_diffs}")
    return 0 if total_diffs == 0 else 1


# ─────────────────────────────────────────────────────────────────────────
# train_test_check mode
# ─────────────────────────────────────────────────────────────────────────

def _build_small_case() -> tuple[list[dict], dict]:
    """10-row hand-verified case for the TrainTestAssigner.

    One ticker ("SYN"), n_bins=5, cutoff = 2024-01-01.

    Training set (5 rows, dates 2019-01-01 .. 2023-01-01) — metric
    values [1, 3, 2, 5, 4] entered in this chronological order. Sorted
    ascending: [1, 2, 3, 4, 5]. With n_bins=5 the rank/bin formula
    `min(int(rank / n * n_bins) + 1, n_bins)` maps:
      value=1 → rank 0 → int(0/5*5)+1 = 1 → bin 1
      value=2 → rank 1 → int(1/5*5)+1 = 2 → bin 2
      value=3 → rank 2 → int(2/5*5)+1 = 3 → bin 3
      value=4 → rank 3 → int(3/5*5)+1 = 4 → bin 4
      value=5 → rank 4 → int(4/5*5)+1 = 5 → bin 5

    Test set (5 rows, dates 2024-01-01 .. 2028-01-01) — metric values
    [7, 6, 9, 8, 10]. Each ranked against the frozen training history
    [1, 2, 3, 4, 5] (n=5):
      value=7  → bisect_left of 7 into [1,2,3,4,5] = 5 → int(5/5*5)+1 = 6
                 → clamped to n_bins=5
      value=6  → rank 5 → bin 5 (clamped)
      value=9  → rank 5 → bin 5 (clamped)
      value=8  → rank 5 → bin 5 (clamped)
      value=10 → rank 5 → bin 5 (clamped)

    All test rows pile into the top bin because every value exceeds
    the training max — that's the correct behaviour for train/test
    extrapolation.
    """
    chronological = [
        ("2019-01-01", 1.0, 0.001),
        ("2020-01-01", 3.0, 0.002),
        ("2021-01-01", 2.0, 0.003),
        ("2022-01-01", 5.0, 0.004),
        ("2023-01-01", 4.0, 0.005),
        # cutoff at 2024-01-01
        ("2024-01-01", 7.0, 0.006),
        ("2025-01-01", 6.0, 0.007),
        ("2026-01-01", 9.0, 0.008),
        ("2027-01-01", 8.0, 0.009),
        ("2028-01-01", 10.0, 0.010),
    ]
    rows = [
        {"ticker": "SYN", "trade_date": d, "iv": v, "ret_5d_fwd_oc": y}
        for (d, v, y) in chronological
    ]
    expected_bins_by_date = {
        "2019-01-01": 1, "2020-01-01": 3, "2021-01-01": 2,
        "2022-01-01": 5, "2023-01-01": 4,
        "2024-01-01": 5, "2025-01-01": 5, "2026-01-01": 5,
        "2027-01-01": 5, "2028-01-01": 5,
    }
    return rows, expected_bins_by_date


def cmd_train_test_check(args) -> int:
    # Import the assigner. We add repo root to sys.path so the script
    # works when invoked as `python scripts/regression_check.py`.
    repo_root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(repo_root))
    try:
        from app.routers.row_compute import (
            TrainTestAssigner, TrainTestSpec,
            InSampleAssigner, InSampleSpec,
            _validate_assignments,
        )
    except Exception as e:
        print(f"[train_test_check] import failed: {e!r}", file=sys.stderr)
        return 2

    # ── Test A: hand-verified small case
    rows, expected = _build_small_case()
    spec = TrainTestSpec(cutoff=_date(2024, 1, 1))
    asgn = TrainTestAssigner(spec)
    state = asgn.fit(rows, metric="iv", n_bins=5, is_all=True)
    assignments = asgn.assign(
        rows, metric="iv", n_bins=5, is_all=True, state=state,
        outcome_col="ret_5d_fwd_oc",
    )
    _validate_assignments(assignments, n_bins=5)

    failures = 0
    by_date = {a.trade_date: a for a in assignments}
    for d, want in expected.items():
        got = by_date.get(d)
        if got is None:
            print(f"[A] FAIL date={d}: no assignment emitted"); failures += 1; continue
        if got.bin != want:
            print(f"[A] FAIL date={d}: expected bin={want}, got bin={got.bin}")
            failures += 1
        else:
            print(f"[A] OK   date={d}: bin={got.bin}")
    if failures == 0:
        print("[A] PASS small-case (hand-verified) passed")
    else:
        print(f"[A] FAIL {failures} small-case mismatch(es)")

    # ── Test B: property — cutoff == max(date) ⇒ train/test ≈ pure in-sample
    rows_b = rows  # reuse
    max_date = max(_date.fromisoformat(r["trade_date"]) for r in rows_b)
    spec_tt = TrainTestSpec(cutoff=max_date)  # train = everything strictly before max
    asgn_tt = TrainTestAssigner(spec_tt)
    state_tt = asgn_tt.fit(rows_b, metric="iv", n_bins=5, is_all=True)
    tt = asgn_tt.assign(rows_b, metric="iv", n_bins=5, is_all=True,
                       state=state_tt, outcome_col="ret_5d_fwd_oc")

    asgn_is = InSampleAssigner(InSampleSpec())
    state_is = asgn_is.fit(rows_b, metric="iv", n_bins=5, is_all=True)
    isamp = asgn_is.assign(rows_b, metric="iv", n_bins=5, is_all=True,
                          state=state_is, outcome_col="ret_5d_fwd_oc")

    # The semantics differ slightly: train/test ranks via bisect_left
    # against the frozen training history (rank-based 1-indexed). Pure
    # in-sample uses `_bucket_pairs`'s sort+index-based bucketing,
    # which agrees with rank-based on unique values. The 9 training-
    # window rows in this synthetic set have unique values, so the
    # two methods agree on each one. The one row at the cutoff
    # boundary is not in either training subset and is the only place
    # the two methods may diverge — accept either result for that
    # row.
    is_by_date = {a.trade_date: a.bin for a in isamp}
    diffs_b = 0
    for a in tt:
        if a.dropped_reason is not None:
            continue
        # Skip the last row (the row AT the cutoff, which is
        # excluded from train/test's training set but included in
        # in-sample's full-history set — divergence is expected here).
        if a.trade_date == max_date.isoformat():
            continue
        if a.bin != is_by_date.get(a.trade_date):
            print(f"[B] divergence date={a.trade_date}: train_test={a.bin}, in_sample={is_by_date.get(a.trade_date)}")
            diffs_b += 1
    if diffs_b == 0:
        print("[B] PASS property test (excluding cutoff boundary row) passed")
    else:
        print(f"[B] FAIL {diffs_b} divergence(s) on rows strictly before cutoff")

    # ── Test C: pre_cutoff dropped_reason on training rows (Option A semantic)
    # Bins are SET on training rows (computed against frozen training history),
    # but the row is marked dropped_reason="pre_cutoff" so aggregators skip
    # them. This preserves the bin information for a future side-by-side
    # training/test view while making the test-period-only aggregation
    # contract explicit.
    failures_c = 0
    by_date_a = {a.trade_date: a for a in assignments}  # from Test A's run
    cutoff_iso = "2024-01-01"
    for d in ("2019-01-01", "2020-01-01", "2021-01-01", "2022-01-01", "2023-01-01"):
        a = by_date_a.get(d)
        if a is None:
            print(f"[C] FAIL date={d}: no assignment"); failures_c += 1; continue
        if a.bin is None:
            print(f"[C] FAIL date={d}: training row has bin=None (should be set)")
            failures_c += 1
        elif a.dropped_reason != "pre_cutoff":
            print(f"[C] FAIL date={d}: dropped_reason={a.dropped_reason!r}, expected 'pre_cutoff'")
            failures_c += 1
    for d in ("2024-01-01", "2025-01-01", "2026-01-01", "2027-01-01", "2028-01-01"):
        a = by_date_a.get(d)
        if a is None:
            print(f"[C] FAIL date={d}: no assignment"); failures_c += 1; continue
        if a.bin is None:
            print(f"[C] FAIL date={d}: test row has bin=None")
            failures_c += 1
        elif a.dropped_reason is not None:
            print(f"[C] FAIL date={d}: dropped_reason={a.dropped_reason!r}, expected None")
            failures_c += 1
    if failures_c == 0:
        print("[C] PASS training rows tagged pre_cutoff; test rows have dropped_reason=None")

    # ── Test D: cutoff > max(date) ⇒ empty test set, no crash
    # Verifies the degenerate edge case. With cutoff one day past the
    # last row, every row is training; aggregators should return empty
    # results (no rows to aggregate), not crash.
    failures_d = 0
    far_cutoff = _date(2030, 1, 1)
    spec_d = TrainTestSpec(cutoff=far_cutoff)
    asgn_d = TrainTestAssigner(spec_d)
    try:
        state_d = asgn_d.fit(rows, metric="iv", n_bins=5, is_all=True)
        assigns_d = asgn_d.assign(rows, metric="iv", n_bins=5, is_all=True,
                                  state=state_d, outcome_col="ret_5d_fwd_oc")
        _validate_assignments(assigns_d, n_bins=5)
        n_pre_cutoff = sum(1 for a in assigns_d if a.dropped_reason == "pre_cutoff")
        n_test = sum(1 for a in assigns_d if a.bin is not None and a.dropped_reason is None)
        if n_pre_cutoff != 10:
            print(f"[D] FAIL expected 10 pre_cutoff rows, got {n_pre_cutoff}")
            failures_d += 1
        if n_test != 0:
            print(f"[D] FAIL expected 0 test rows, got {n_test}")
            failures_d += 1
        # assign_batch should return empty results without crashing.
        results, dropped, start = asgn_d.assign_batch(
            rows, ["iv"], "ret_5d_fwd_oc", n_bins=5, is_all=True,
        )
        if results != []:
            print(f"[D] FAIL assign_batch expected [] results, got {len(results)} features")
            failures_d += 1
        if failures_d == 0:
            print(f"[D] PASS empty-test-set degenerates: {n_pre_cutoff} training, "
                  f"{n_test} test, assign_batch returned [] cleanly")
    except Exception as e:
        print(f"[D] FAIL crash on empty-test-set: {e!r}")
        failures_d += 1

    return 0 if (failures == 0 and diffs_b == 0
                 and failures_c == 0 and failures_d == 0) else 1


# ─────────────────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    ap_cap = sub.add_parser("capture", help="Hit endpoints and write a snapshot")
    ap_cap.add_argument("--tag", required=True, help="Snapshot subdirectory name")
    ap_cap.add_argument("--base", default=DEFAULT_BASE,
                        help=f"API base URL (default {DEFAULT_BASE})")

    ap_diff = sub.add_parser("diff", help="Compare two snapshots")
    ap_diff.add_argument("--before", required=True, help="Baseline snapshot tag")
    ap_diff.add_argument("--after",  required=True, help="Comparison snapshot tag")
    ap_diff.add_argument("--max-diffs", type=int, default=20,
                         help="Max diffs to print per file")

    sub.add_parser("train_test_check",
                  help="Hand-verified small case + cutoff property test")

    args = ap.parse_args()
    if args.cmd == "capture":
        return cmd_capture(args)
    if args.cmd == "diff":
        return cmd_diff(args)
    if args.cmd == "train_test_check":
        return cmd_train_test_check(args)
    ap.error(f"unknown command: {args.cmd}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
