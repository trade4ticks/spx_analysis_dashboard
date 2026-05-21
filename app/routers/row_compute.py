"""Generic per-row computation layer (Step 1 of the row_compute refactor).

Today the dashboard ships two per-row computations: in-sample binning
(`_bucket_pairs_per_ticker` etc.) and walk-forward binning
(`_walk_forward_bucket_per_ticker` etc.). A third — train/test split — is
on the way. Adding a fourth would require touching ~30 if-branches
across 5 endpoints. This module is the swappable substrate they share.

Every per-row computation implements one Protocol (`RowAssigner.fit`
and `.assign`) and emits one fixed contract (`RowAssignment`). New
computations register here and are dropped into endpoints by spec.

# Naming
The module, Protocol, contract, and registry are intentionally
*generic* (not "binning"-specific). Future per-row computations whose
output isn't a bin number (e.g. a rolling-IC score, a cluster id, an
anomaly z-score) can register the same way and reuse the downstream
aggregation layer by passing a different `group_key`.

# Index convention
At the contract boundary, every `RowAssignment.bin` is **1-indexed in
[1, n_bins]**, or `None` for warmup / missing-value / pre-cutoff
rows. The underlying helpers in `oi_analysis.py` use mixed conventions
internally (some 0-indexed for Python-list indexing, some 1-indexed)
— that's an internal detail of those helpers. `_validate_assignments`
enforces the 1-indexed boundary at runtime and is kept on permanently;
it's an O(N) sanity check on already-O(N log N) work.

# Step 1 scope
This file is purely additive. It is dead code with no callers. Step 2
will route `/analyze` through it; subsequent steps route the other
endpoints. The concrete Assigners in this file *delegate to the
existing helpers* in `oi_analysis.py` — no rewritten math — so
behaviour is bit-identical to the current in-sample and walk-forward
paths.

# Usage
    spec = WalkForwardSpec(warmup=252)
    assigner = ASSIGNERS[spec.kind](spec)
    state = assigner.fit(rows, metric="iv", n_bins=20, is_all=True)
    assignments = assigner.assign(
        rows, metric="iv", n_bins=20, is_all=True,
        state=state, outcome_col="ret_5d_fwd_oc",
    )
    _validate_assignments(assignments, n_bins=20)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date as _date
from typing import Any, Literal, Optional, Protocol, runtime_checkable


# ── Constants ────────────────────────────────────────────────────────────

# Mirrors oi_analysis._DEFAULT_WALKFWD_WARMUP. We don't import that
# value to keep this module independent of router internals; the two
# constants should stay in sync (252 trading days, ~1 year warmup).
DEFAULT_WALKFWD_WARMUP = 252


# ── Output contract ──────────────────────────────────────────────────────

@dataclass(frozen=True)
class RowAssignment:
    """One per-row record. Universal columns plus method-specific `bin`.

    Universal: ticker, trade_date, metric_name, metric_value,
    outcome_col, forward_return, dropped_reason.

    Binning-specific: bin (1..n_bins, or None when dropped) and n_bins.

    `trade_date` is typed `Any` because it carries whatever the upstream
    row had — asyncpg returns `datetime.date`, synthetic test data uses
    strings — and downstream legacy /analyze code uses `.year`,
    `.month`, `.weekday()` on `pair[2]`. Coercing to string here would
    break those callers. Consumers needing a string serialize via
    `str(...)` or `.isoformat()`.

    `dropped_reason` is None for emitted rows and otherwise one of:
      - "missing_value"           — metric or outcome was None/NaN
      - "warmup"                  — walk-forward per-ticker history < warmup
      - "insufficient_history"    — in-sample: ticker had < n_bins rows
      - "insufficient_train_history" — train-test: ticker had < n_bins training rows
      - "pre_cutoff"              — (reserved; unused in Step 1)
    """
    ticker: str
    trade_date: Any
    metric_name: str
    metric_value: float
    n_bins: int
    bin: Optional[int]
    outcome_col: str
    forward_return: Optional[float]
    dropped_reason: Optional[str]


# ── Method specs (config) ────────────────────────────────────────────────

@dataclass(frozen=True)
class InSampleSpec:
    kind: Literal["in_sample"] = "in_sample"


@dataclass(frozen=True)
class WalkForwardSpec:
    warmup: int = DEFAULT_WALKFWD_WARMUP
    kind: Literal["walk_forward"] = "walk_forward"


@dataclass(frozen=True)
class TrainTestSpec:
    """cutoff: rows with trade_date < cutoff are the training window;
    rows >= cutoff are ranked against the frozen training history."""
    cutoff: _date
    warmup_in_train: int = DEFAULT_WALKFWD_WARMUP
    kind: Literal["train_test"] = "train_test"


BinningSpec = InSampleSpec | WalkForwardSpec | TrainTestSpec


# ── RowAssigner Protocol ─────────────────────────────────────────────────

@runtime_checkable
class RowAssigner(Protocol):
    """Two-method Protocol every per-row computation implements.

    `fit(rows, metric, n_bins, is_all) -> state`
        Returns an opaque per-method state. For in-sample and
        walk-forward, state is None (computation is inline in
        `assign`). For train-test, state holds the frozen per-ticker
        sorted training histories.

    `assign(rows, metric, n_bins, is_all, state, outcome_col)
            -> list[RowAssignment]`
        Emits one RowAssignment per input row. Rows that should be
        excluded from downstream stats get `bin=None` with the
        appropriate `dropped_reason`. Output order is the same as
        input row order (callers that need chronological order should
        sort by `trade_date` afterward).
    """
    def fit(self, rows: list[dict], metric: str, n_bins: int,
            is_all: bool) -> Any: ...

    def assign(self, rows: list[dict], metric: str, n_bins: int,
               is_all: bool, state: Any, outcome_col: str
               ) -> list[RowAssignment]: ...

    def assign_batch(self, rows: list[dict], feature_cols: list[str],
                     outcome_col: str, n_bins: int, is_all: bool
                     ) -> tuple[list[dict], Optional[int], Optional[str]]:
        """Batch variant for many-feature endpoints (`/global-metric-bins`).

        Computes per-bin avg-return for every column in `feature_cols`
        in a single pass. Output preserves the legacy
        `_compute_all_bins_fast` / `_compute_all_bins_walk_forward`
        shape so the caller doesn't need to translate:

        Returns (metrics, dropped_warmup_n, start_date):
          metrics: list of {name, bins: [avg_ret per bin], bin_ns: [count per bin]}
          dropped_warmup_n: None for in-sample (no warmup concept), int for walk-forward
          start_date: None for in-sample, ISO string for walk-forward
        """
        ...


# ── Shared first-pass: parse rows into (pair, row_idx) plus dropped records ──

def _parse_rows(rows: list[dict], metric: str, outcome_col: str, n_bins: int,
                ) -> tuple[list[tuple], list[RowAssignment]]:
    """Walk `rows` once. For each row, either:
      - append a tuple `(xf, yf, date, tkr, row_idx)` to `valid_pairs`
        when both metric value and outcome are present and numeric, OR
      - append a `RowAssignment(bin=None, dropped_reason="missing_value")`
        directly to `dropped_assignments` for downstream emission.

    The valid_pairs tuples are passed through the existing oi_analysis
    helpers without modification — those helpers index pair[0] (x),
    pair[2] (date) etc. and tolerate extra trailing elements, so
    appending row_idx at position 4 is safe.

    NOTE on `trade_date` type: we preserve the original object (whatever
    asyncpg returned — typically `datetime.date`) rather than coercing
    to a string. Downstream legacy /analyze code uses `.year`, `.month`,
    `.weekday()` on `pair[2]`; coercing to string would corrupt the
    DOW chart (str has no `.weekday()`). The RowAssignment.trade_date
    field carries the same object; consumers needing a string serialize
    it explicitly with `str(...)` or `.isoformat()`.
    """
    valid_pairs: list[tuple] = []
    dropped_assignments: list[RowAssignment] = []
    for ri, r in enumerate(rows):
        tkr = str(r.get("ticker", ""))
        trade_date = r.get("trade_date", "")  # preserve original type
        xv = r.get(metric)
        yv = r.get(outcome_col)
        if xv is None or yv is None:
            dropped_assignments.append(RowAssignment(
                ticker=tkr, trade_date=trade_date, metric_name=metric,
                metric_value=float("nan"), n_bins=n_bins, bin=None,
                outcome_col=outcome_col, forward_return=None,
                dropped_reason="missing_value",
            ))
            continue
        try:
            xf = float(xv); yf = float(yv)
        except (TypeError, ValueError):
            dropped_assignments.append(RowAssignment(
                ticker=tkr, trade_date=trade_date, metric_name=metric,
                metric_value=float("nan"), n_bins=n_bins, bin=None,
                outcome_col=outcome_col, forward_return=None,
                dropped_reason="missing_value",
            ))
            continue
        if math.isnan(xf) or math.isnan(yf):
            dropped_assignments.append(RowAssignment(
                ticker=tkr, trade_date=trade_date, metric_name=metric,
                metric_value=float("nan"), n_bins=n_bins, bin=None,
                outcome_col=outcome_col, forward_return=None,
                dropped_reason="missing_value",
            ))
            continue
        valid_pairs.append((xf, yf, trade_date, tkr, ri))
    return valid_pairs, dropped_assignments


# ── Concrete Assigner: InSampleAssigner ──────────────────────────────────

class InSampleAssigner:
    """Full-history bin assignment, per-ticker in ALL mode.

    Delegates to `oi_analysis._bucket_pairs_per_ticker` (ALL mode) or
    `_bucket_pairs` (single ticker). Those helpers return 0-indexed
    `list[bucket]`; this class translates to the 1-indexed contract.

    Tickers with fewer than n_bins observations in ALL mode are
    excluded by the underlying helper. Their rows are surfaced here
    as `bin=None, dropped_reason="insufficient_history"`.
    """
    def __init__(self, spec: InSampleSpec):
        self.spec = spec

    def fit(self, rows, metric, n_bins, is_all):
        return None  # in-sample needs no precomputed state

    def assign_batch(self, rows, feature_cols, outcome_col, n_bins, is_all):
        """Delegate to `_compute_all_bins_fast` — the numpy-vectorized
        in-sample batch helper. Returns (metrics, None, None) since
        in-sample has no warmup concept.
        """
        from app.routers.oi_analysis import _compute_all_bins_fast
        metrics = _compute_all_bins_fast(rows, feature_cols, outcome_col, n_bins, is_all)
        return metrics, None, None

    def assign(self, rows, metric, n_bins, is_all, state, outcome_col):
        from app.routers.oi_analysis import (  # local import avoids circular
            _bucket_pairs, _bucket_pairs_per_ticker,
        )
        valid_pairs, dropped = _parse_rows(rows, metric, outcome_col, n_bins)

        if is_all:
            by_tkr: dict = {}
            for p in valid_pairs:
                by_tkr.setdefault(p[3], []).append(p)
            buckets = _bucket_pairs_per_ticker(by_tkr, n_bins)
            excluded_tickers = {t for t, ps in by_tkr.items() if len(ps) < n_bins}
        else:
            buckets = _bucket_pairs(valid_pairs, n_bins)
            excluded_tickers = set()

        # Walk buckets to build row_idx -> 1-indexed bin
        row_to_bin: dict[int, int] = {}
        for bin_idx_0, bucket in enumerate(buckets):
            for p in bucket:
                row_to_bin[p[4]] = bin_idx_0 + 1

        # Emit one assignment per valid pair (preserves input row order
        # within the valid subset). Add the dropped (missing-value)
        # assignments at the end.
        out: list[RowAssignment] = []
        for (xf, yf, date_s, tkr, ri) in valid_pairs:
            b = row_to_bin.get(ri)
            if b is None:
                # Ticker was excluded for having < n_bins observations.
                out.append(RowAssignment(
                    ticker=tkr, trade_date=date_s, metric_name=metric,
                    metric_value=xf, n_bins=n_bins, bin=None,
                    outcome_col=outcome_col, forward_return=yf,
                    dropped_reason="insufficient_history",
                ))
            else:
                out.append(RowAssignment(
                    ticker=tkr, trade_date=date_s, metric_name=metric,
                    metric_value=xf, n_bins=n_bins, bin=b,
                    outcome_col=outcome_col, forward_return=yf,
                    dropped_reason=None,
                ))
        out.extend(dropped)
        return out


# ── Concrete Assigner: WalkForwardAssigner ───────────────────────────────

class WalkForwardAssigner:
    """Walk-forward bin assignment, per-ticker rank against running history.

    Delegates to `_walk_forward_bucket_per_ticker` (ALL mode) and
    `_walk_forward_bucket_pairs` (single-ticker). Those helpers already
    emit 1-indexed bins and pre-drop warmup rows; this class surfaces
    warmup rows with `bin=None, dropped_reason="warmup"`.
    """
    def __init__(self, spec: WalkForwardSpec):
        self.spec = spec

    def fit(self, rows, metric, n_bins, is_all):
        return None  # walk-forward is computed inline in assign

    def assign_batch(self, rows, feature_cols, outcome_col, n_bins, is_all):
        """Delegate to `_compute_all_bins_walk_forward` — the
        numpy-vectorized walk-forward batch helper. Returns
        (metrics, dropped_warmup_n, start_date).
        """
        from app.routers.oi_analysis import _compute_all_bins_walk_forward
        metrics, dropped, start_date = _compute_all_bins_walk_forward(
            rows, feature_cols, outcome_col, n_bins, is_all, warmup=self.spec.warmup,
        )
        return metrics, dropped, start_date

    def assign(self, rows, metric, n_bins, is_all, state, outcome_col):
        from app.routers.oi_analysis import (
            _walk_forward_bucket_pairs, _walk_forward_bucket_per_ticker,
        )
        valid_pairs, dropped = _parse_rows(rows, metric, outcome_col, n_bins)
        warm = self.spec.warmup

        if is_all:
            by_tkr: dict = {}
            for p in valid_pairs:
                by_tkr.setdefault(p[3], []).append(p)
            _, wf_assignments, _ = _walk_forward_bucket_per_ticker(
                by_tkr, [n_bins], warm
            )
        else:
            # Single-ticker: pairs need chronological order before the helper.
            chrono = sorted(valid_pairs, key=lambda p: p[2])
            wf_assignments, _ = _walk_forward_bucket_pairs(
                chrono, [n_bins], warm
            )

        # Build row_idx -> 1-indexed bin from walk-forward assignments
        row_to_bin: dict[int, int] = {}
        for pair, bins_dict in wf_assignments:
            row_to_bin[pair[4]] = bins_dict[n_bins]

        out: list[RowAssignment] = []
        for (xf, yf, date_s, tkr, ri) in valid_pairs:
            b = row_to_bin.get(ri)
            if b is None:
                out.append(RowAssignment(
                    ticker=tkr, trade_date=date_s, metric_name=metric,
                    metric_value=xf, n_bins=n_bins, bin=None,
                    outcome_col=outcome_col, forward_return=yf,
                    dropped_reason="warmup",
                ))
            else:
                out.append(RowAssignment(
                    ticker=tkr, trade_date=date_s, metric_name=metric,
                    metric_value=xf, n_bins=n_bins, bin=b,
                    outcome_col=outcome_col, forward_return=yf,
                    dropped_reason=None,
                ))
        out.extend(dropped)
        return out


# ── Concrete Assigner: TrainTestAssigner ─────────────────────────────────

class TrainTestAssigner:
    """Train/test split: freeze bin thresholds on data before the cutoff;
    rank every row (training and test) against that frozen history.

    `fit` builds, per ticker (or globally in single-ticker mode), a
    sorted list of metric values from rows with `trade_date < cutoff`.

    `assign` calls `_bin_for_value(row_value, frozen_history, n_bins)`
    for every row — rank-based 1-indexed assignment. Rows whose ticker
    has fewer than `n_bins` training samples get
    `bin=None, dropped_reason="insufficient_train_history"`.

    Note on semantics: rows BEFORE the cutoff get a bin too (ranked
    against the same frozen history, which includes them). This is
    the convention the user described — bins are computed on the
    training set and applied uniformly. If a future revision wants
    pre-cutoff rows excluded entirely, change the "pre_cutoff" branch
    below to emit `dropped_reason="pre_cutoff"` and `bin=None`.
    """
    def __init__(self, spec: TrainTestSpec):
        self.spec = spec

    def assign_batch(self, rows, feature_cols, outcome_col, n_bins, is_all):
        """Train/test batch — Step 6 will implement this when the
        train/test mode is wired into the frontend. Until then,
        train/test mode is only exercised by the single-metric `assign`
        path; if a caller routes a batch endpoint with TrainTestSpec
        before Step 6, fail loudly rather than silently fall back.
        """
        raise NotImplementedError(
            "TrainTestAssigner.assign_batch is filled in by Step 6. "
            "/global-metric-bins should not see TrainTestSpec until then."
        )

    def fit(self, rows, metric, n_bins, is_all):
        cutoff_s = self.spec.cutoff.isoformat()
        train_history: dict[str, list[float]] = {}
        for r in rows:
            date_s = str(r.get("trade_date", ""))
            if date_s >= cutoff_s:
                continue
            xv = r.get(metric)
            if xv is None:
                continue
            try:
                xf = float(xv)
            except (TypeError, ValueError):
                continue
            if math.isnan(xf):
                continue
            key = str(r.get("ticker", "_")) if is_all else "_"
            train_history.setdefault(key, []).append(xf)
        for k in train_history:
            train_history[k].sort()
        return train_history

    def assign(self, rows, metric, n_bins, is_all, state, outcome_col):
        from app.routers.oi_analysis import _bin_for_value
        train_history = state or {}
        out: list[RowAssignment] = []
        for r in rows:
            tkr = str(r.get("ticker", ""))
            date_s = str(r.get("trade_date", ""))
            xv = r.get(metric)
            yv = r.get(outcome_col)

            if xv is None or yv is None:
                out.append(RowAssignment(
                    ticker=tkr, trade_date=date_s, metric_name=metric,
                    metric_value=float("nan"), n_bins=n_bins, bin=None,
                    outcome_col=outcome_col, forward_return=None,
                    dropped_reason="missing_value",
                ))
                continue
            try:
                xf = float(xv); yf = float(yv)
            except (TypeError, ValueError):
                out.append(RowAssignment(
                    ticker=tkr, trade_date=date_s, metric_name=metric,
                    metric_value=float("nan"), n_bins=n_bins, bin=None,
                    outcome_col=outcome_col, forward_return=None,
                    dropped_reason="missing_value",
                ))
                continue
            if math.isnan(xf) or math.isnan(yf):
                out.append(RowAssignment(
                    ticker=tkr, trade_date=date_s, metric_name=metric,
                    metric_value=float("nan"), n_bins=n_bins, bin=None,
                    outcome_col=outcome_col, forward_return=None,
                    dropped_reason="missing_value",
                ))
                continue

            history_key = tkr if is_all else "_"
            history = train_history.get(history_key, [])
            b = _bin_for_value(xf, history, n_bins)
            if b is None:
                out.append(RowAssignment(
                    ticker=tkr, trade_date=date_s, metric_name=metric,
                    metric_value=xf, n_bins=n_bins, bin=None,
                    outcome_col=outcome_col, forward_return=yf,
                    dropped_reason="insufficient_train_history",
                ))
            else:
                out.append(RowAssignment(
                    ticker=tkr, trade_date=date_s, metric_name=metric,
                    metric_value=xf, n_bins=n_bins, bin=b,
                    outcome_col=outcome_col, forward_return=yf,
                    dropped_reason=None,
                ))
        return out


# ── Registry ─────────────────────────────────────────────────────────────

# One dict entry per method. A new method drops in here and (in step 6)
# gets a new branch in the FastAPI request-model → spec parser.
ASSIGNERS: dict[str, type] = {
    "in_sample":    InSampleAssigner,
    "walk_forward": WalkForwardAssigner,
    "train_test":   TrainTestAssigner,
}


# ── Secondary-endpoint dispatch (Step 4) ─────────────────────────────────
# The secondary correlation explorer + scanner endpoints share a three-step
# shape that depends on the active spec:
#   (1) PRIMARY filter — narrow cached rows to "user-selected primary bins"
#   (2) SECONDARY bin stats — per-bin avg-ret for one metric inside (1)
#   (3) SECONDARY membership — binary 0/1 vector for one (metric, bin set)
#
# Pre-Step-4 each endpoint forked on `walk_forward` and called a
# different helper for each step. The three functions below collapse
# the fork into one spec-dispatched call. Each delegates to the
# existing helpers in oi_analysis.py — no rewritten math.

def filter_by_assignments(
    rows: list,
    spec,
    primary_metric: str,
    selected_primary_bins,
    is_all: bool,
    filtered_dates: Optional[list] = None,
) -> tuple[list, int, int]:
    """Primary filter for the secondary-endpoint family.

    Returns (filtered_rows, dropped_warmup_n, universe_n).

      in_sample: uses the frontend's `filtered_dates` (a list of
        "ticker|date" strings already narrowed by /analyze's in-sample
        primary bins). `dropped_warmup_n` is 0; `universe_n` is the full
        cached row count. Output is in cached-row iteration order
        (NOT chronologically resorted) to preserve legacy bit-equivalence
        — downstream `_compute_bins_for_metric` is order-insensitive.

      walk_forward: delegates to `_walk_forward_primary_filter`. Computes
        walk-forward primary bins backend-side (20-bin universe, 252-day
        warmup) and keeps rows whose bin ∈ `selected_primary_bins`.
        Output is chronologically sorted.

      train_test: (Step 6) frozen training-set bins.
    """
    from app.routers.oi_analysis import (
        _walk_forward_primary_filter, _filter_by_tkr_date, _parse_tkr_date_set,
    )
    if spec.kind == "in_sample":
        filtered = _filter_by_tkr_date(rows, _parse_tkr_date_set(filtered_dates or []))
        return filtered, 0, len(rows)
    if spec.kind == "walk_forward":
        return _walk_forward_primary_filter(
            rows, primary_metric, set(selected_primary_bins or []), is_all,
        )
    if spec.kind == "train_test":
        raise NotImplementedError(
            "TrainTestSpec.filter_by_assignments is filled in by Step 6."
        )
    raise ValueError(f"unknown spec kind: {spec.kind!r}")


def assign_secondary_bin_stats(
    spec,
    rows_chrono: list,
    metric: str,
    n_bins: int,
    outcome_col: str,
    is_all: bool,
) -> Optional[dict]:
    """Per-bin avg-return for ONE secondary metric inside an already-
    primary-filtered chronological subset. Returns {name, bins, bin_ns}
    or None if insufficient data.

      in_sample: per-ticker rank over the full subset history
        (`_compute_bins_for_metric`).
      walk_forward: per-ticker bisect_left running history with
        warmup=n_bins — the macro 252-day warmup gate is already
        enforced by the primary filter, so the inner warmup is small
        (`_compute_walk_forward_bin_stats`).
      train_test: (Step 6).
    """
    from app.routers.oi_analysis import (
        _compute_bins_for_metric, _compute_walk_forward_bin_stats,
    )
    if spec.kind == "in_sample":
        return _compute_bins_for_metric(rows_chrono, metric, outcome_col, n_bins, is_all)
    if spec.kind == "walk_forward":
        return _compute_walk_forward_bin_stats(rows_chrono, metric, outcome_col, n_bins, is_all)
    if spec.kind == "train_test":
        raise NotImplementedError(
            "TrainTestSpec.assign_secondary_bin_stats is filled in by Step 6."
        )
    raise ValueError(f"unknown spec kind: {spec.kind!r}")


def assign_secondary_buckets(
    spec,
    rows_chrono: list,
    metric: str,
    n_bins: int,
    outcome_col: str,
    is_all: bool,
):
    """Per-bin row tuples for ONE secondary metric, used by /secondary-detail.

    Returns `buckets: list[list[(fv_or_rank, outcome, date, ticker)]]` of
    length n_bins, or `None` if fewer than `n_bins * 2` valid rows.

    The first tuple element is:
      - in_sample ALL mode    : `rank/n_t` (legacy per-ticker normalization)
      - in_sample single mode : the raw metric value
      - walk_forward          : the raw metric value
    Downstream /secondary-detail uses only positions [1], [2], [3]
    (outcome, date, ticker) — the first element is preserved for
    legacy bit-equivalence but not consumed.

    Behavior by spec:
      in_sample: per-ticker rank-normalize (ALL) or flat sort (single),
        then distribute into n_bins by index position.
      walk_forward: per-ticker `bisect_left` running history with
        warmup=n_bins (macro warmup already enforced by the primary
        filter); each row's walk-forward bin places its tuple in
        `buckets[bin - 1]`.
      train_test: (Step 6).
    """
    if spec.kind == "in_sample":
        if is_all:
            by_tkr: dict = {}
            for r in rows_chrono:
                v = r.get(metric)
                o = r.get(outcome_col)
                if v is None or o is None:
                    continue
                try:
                    fv = float(v); fo = float(o)
                    if math.isnan(fv) or math.isnan(fo):
                        continue
                except (TypeError, ValueError):
                    continue
                by_tkr.setdefault(r.get("ticker", "_"), []).append(
                    (fv, fo, r.get("trade_date", ""), r.get("ticker", ""))
                )
            norm_rows = []
            for tkr_vals in by_tkr.values():
                if len(tkr_vals) < n_bins:
                    continue
                sorted_t = sorted(tkr_vals, key=lambda x: x[0])
                n_t = len(sorted_t)
                for rank, (_, y, d, tkr) in enumerate(sorted_t):
                    norm_rows.append((rank / n_t, y, d, tkr))
        else:
            norm_rows = []
            for r in rows_chrono:
                v = r.get(metric)
                o = r.get(outcome_col)
                if v is None or o is None:
                    continue
                try:
                    fv = float(v); fo = float(o)
                    if math.isnan(fv) or math.isnan(fo):
                        continue
                except (TypeError, ValueError):
                    continue
                norm_rows.append((fv, fo, r.get("trade_date", ""), r.get("ticker", "")))
        if len(norm_rows) < n_bins * 2:
            return None
        sorted_norm = sorted(norm_rows, key=lambda x: x[0])
        n = len(sorted_norm)
        buckets: list = [[] for _ in range(n_bins)]
        for i, row_t in enumerate(sorted_norm):
            b = min(int(i / n * n_bins), n_bins - 1)
            buckets[b].append(row_t)
        return buckets

    if spec.kind == "walk_forward":
        from app.routers.oi_analysis import _walk_forward_bins, _sort_chrono
        # Macro warmup already enforced by the primary filter; use a tiny
        # inner warmup so even small subsets can produce bin assignments.
        # Legacy WF branch did not impose a second "len < n_bins*2" check
        # (only the IS branch did, on norm_rows). Mirror that — return
        # whatever buckets the walk-forward assignments produced, possibly
        # partly empty. Downstream `bins_out` handles empty buckets.
        filtered_chrono = _sort_chrono(rows_chrono)
        wf_sec = _walk_forward_bins(filtered_chrono, metric, n_bins, is_all, warmup=n_bins)
        buckets = [[] for _ in range(n_bins)]
        for i, r in enumerate(filtered_chrono):
            b = wf_sec.get(i)
            if b is None:
                continue
            v = r.get(metric); o = r.get(outcome_col)
            if v is None or o is None:
                continue
            try:
                fv = float(v); fo = float(o)
                if math.isnan(fv) or math.isnan(fo):
                    continue
            except (TypeError, ValueError):
                continue
            buckets[b - 1].append((fv, fo, r.get("trade_date", ""), r.get("ticker", "")))
        return buckets

    if spec.kind == "train_test":
        raise NotImplementedError(
            "TrainTestSpec.assign_secondary_buckets is filled in by Step 6."
        )
    raise ValueError(f"unknown spec kind: {spec.kind!r}")


def secondary_membership(
    spec,
    rows_chrono: list,
    metric: str,
    selected_bins,
    n_bins: int,
    is_all: bool,
):
    """Binary 0/1 membership vector for ONE secondary metric inside an
    already-primary-filtered chronological subset. Returns a numpy float64
    array of length len(rows_chrono).

      in_sample: `_bin_membership` — per-ticker rank, 0 for excluded.
      walk_forward: `_walk_forward_membership` — small inner warmup.
      train_test: (Step 6).
    """
    from app.routers.oi_analysis import _bin_membership, _walk_forward_membership
    sel = set(selected_bins or [])
    if spec.kind == "in_sample":
        return _bin_membership(rows_chrono, metric, sel, n_bins, is_all)
    if spec.kind == "walk_forward":
        return _walk_forward_membership(rows_chrono, metric, sel, n_bins, is_all)
    if spec.kind == "train_test":
        raise NotImplementedError(
            "TrainTestSpec.secondary_membership is filled in by Step 6."
        )
    raise ValueError(f"unknown spec kind: {spec.kind!r}")


# ── Runtime contract validator ───────────────────────────────────────────

def _validate_assignments(assignments: list[RowAssignment], n_bins: int) -> None:
    """Assert that every RowAssignment honors the 1-indexed contract.

    Called at the boundary of every `assign()` consumer. Cheap (~one
    int compare per row) and catches index-convention drift before it
    propagates into a payload. Keep on permanently.

    Raises AssertionError with a descriptive message on first violation.
    """
    for i, a in enumerate(assignments):
        if a.bin is None:
            # `dropped_reason` must be set on every None-bin row so
            # downstream consumers can distinguish warmup / missing /
            # insufficient-history cases.
            if a.dropped_reason is None:
                raise AssertionError(
                    f"RowAssignment[{i}] has bin=None but dropped_reason=None; "
                    f"every dropped row must declare why "
                    f"(ticker={a.ticker} date={a.trade_date})"
                )
            continue
        if not isinstance(a.bin, int) or isinstance(a.bin, bool):
            raise AssertionError(
                f"RowAssignment[{i}] bin={a.bin!r} is not int (or None); "
                f"ticker={a.ticker} date={a.trade_date}"
            )
        if a.bin < 1 or a.bin > n_bins:
            raise AssertionError(
                f"RowAssignment[{i}] bin={a.bin} out of range [1, {n_bins}]; "
                f"ticker={a.ticker} date={a.trade_date}. "
                f"Index-convention drift — every Assigner must emit "
                f"1-indexed bins at the contract boundary."
            )
