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

import bisect
import math
from collections import defaultdict
from dataclasses import dataclass
from datetime import date as _date
from typing import Any, Literal, Optional, Protocol, runtime_checkable

import numpy as np


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
      - "missing_value"           — metric or outcome was None/NaN (bin=None)
      - "warmup"                  — walk-forward per-ticker history < warmup (bin=None)
      - "insufficient_history"    — in-sample: ticker had < n_bins rows (bin=None)
      - "insufficient_train_history" — train-test: ticker had < n_bins training rows (bin=None)
      - "pre_cutoff"              — train-test only: training-window row (trade_date < cutoff).
        Bin IS set (computed against the frozen training history), but the
        row is tagged so test-period aggregators skip it. This keeps the
        training-period bin information available for a future side-by-side
        training/test view without re-computing.
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

    def assign(self, rows, metric, n_bins, is_all, state, outcome_col,
               *, bin20_by_key: Optional[dict] = None):
        """Group 7: when `bin20_by_key` is supplied, build RowAssignments
        directly from the stored lookup. Rows whose key is absent from
        the lookup (or whose stored `bin20 = 0`) are surfaced as
        `bin=None, dropped_reason="warmup"` — preserves the existing
        downstream contract that consumers check `a.bin is not None`.
        """
        valid_pairs, dropped = _parse_rows(rows, metric, outcome_col, n_bins)

        if bin20_by_key is not None:
            # Stored-bin path
            out: list[RowAssignment] = []
            for (xf, yf, date_s, tkr, ri) in valid_pairs:
                d_key = (date_s.isoformat() if hasattr(date_s, "isoformat")
                         else str(date_s))
                b20 = bin20_by_key.get((tkr, d_key))
                if b20 is None or b20 <= 0:
                    out.append(RowAssignment(
                        ticker=tkr, trade_date=date_s, metric_name=metric,
                        metric_value=xf, n_bins=n_bins, bin=None,
                        outcome_col=outcome_col, forward_return=yf,
                        dropped_reason="warmup",
                    ))
                    continue
                b = min(((b20 - 1) * n_bins) // 20 + 1, n_bins)
                out.append(RowAssignment(
                    ticker=tkr, trade_date=date_s, metric_name=metric,
                    metric_value=xf, n_bins=n_bins, bin=b,
                    outcome_col=outcome_col, forward_return=yf,
                    dropped_reason=None,
                ))
            out.extend(dropped)
            return out

        # Legacy on-the-fly path.
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

    def assign(self, rows, metric, n_bins, is_all, state, outcome_col=None):
        """Assign train-test bins. Both training-window and test-window
        rows get a bin assignment computed against the frozen training
        history. Training rows are tagged `dropped_reason="pre_cutoff"`
        so downstream aggregators can exclude them from test-period
        stats while keeping the bin information available for a future
        side-by-side training/test view.

        `outcome_col` is optional. Pass None (or omit) when only the bin
        number is needed (e.g. filter_by_assignments, secondary_membership)
        — rows with a valid metric value will still receive a bin even if
        no outcome is available. When provided, rows missing the outcome
        value are dropped (bin=None, dropped_reason="missing_value").

        Semantics:
          - training-window row (trade_date < cutoff): bin set, dropped_reason="pre_cutoff"
          - test-window row, valid bin:                bin set, dropped_reason=None
          - row with missing metric (or outcome when required):
                                                       bin=None, dropped_reason="missing_value"
          - row whose ticker had <n_bins training samples: bin=None, dropped_reason="insufficient_train_history"
        """
        train_history = state or {}
        cutoff_s = self.spec.cutoff.isoformat()
        need_outcome = outcome_col is not None
        out: list[RowAssignment] = []
        for r in rows:
            tkr = str(r.get("ticker", ""))
            # Preserve original trade_date type on the RowAssignment
            # (downstream legacy code does `dd - last_date` on pair[2]
            # which requires a date object, not a string).
            trade_date = r.get("trade_date", "")
            # For the cutoff comparison we need a string form.
            date_s = (trade_date.isoformat() if hasattr(trade_date, "isoformat")
                       else str(trade_date))
            is_pre_cutoff = date_s < cutoff_s
            xv = r.get(metric)
            yv = r.get(outcome_col) if need_outcome else None

            if xv is None or (need_outcome and yv is None):
                out.append(RowAssignment(
                    ticker=tkr, trade_date=trade_date, metric_name=metric,
                    metric_value=float("nan"), n_bins=n_bins, bin=None,
                    outcome_col=outcome_col or "", forward_return=None,
                    dropped_reason="missing_value",
                ))
                continue
            try:
                xf = float(xv)
                yf = float(yv) if need_outcome else None
            except (TypeError, ValueError):
                out.append(RowAssignment(
                    ticker=tkr, trade_date=trade_date, metric_name=metric,
                    metric_value=float("nan"), n_bins=n_bins, bin=None,
                    outcome_col=outcome_col or "", forward_return=None,
                    dropped_reason="missing_value",
                ))
                continue
            if math.isnan(xf) or (need_outcome and yf is not None and math.isnan(yf)):
                out.append(RowAssignment(
                    ticker=tkr, trade_date=trade_date, metric_name=metric,
                    metric_value=float("nan"), n_bins=n_bins, bin=None,
                    outcome_col=outcome_col or "", forward_return=None,
                    dropped_reason="missing_value",
                ))
                continue

            history_key = tkr if is_all else "_"
            history = train_history.get(history_key, [])
            b = _bin_for_value(xf, history, n_bins)
            if b is None:
                out.append(RowAssignment(
                    ticker=tkr, trade_date=trade_date, metric_name=metric,
                    metric_value=xf, n_bins=n_bins, bin=None,
                    outcome_col=outcome_col or "", forward_return=yf,
                    dropped_reason="insufficient_train_history",
                ))
            else:
                # Training-window row: bin set, marked pre_cutoff so
                # aggregators skip it (test-period-only stats). Test-window
                # row: bin set, dropped_reason None — included in aggregation.
                out.append(RowAssignment(
                    ticker=tkr, trade_date=trade_date, metric_name=metric,
                    metric_value=xf, n_bins=n_bins, bin=b,
                    outcome_col=outcome_col or "", forward_return=yf,
                    dropped_reason=("pre_cutoff" if is_pre_cutoff else None),
                ))
        return out


def dropped_count_for_mode(spec, assignments) -> int:
    """Count of rows dropped by the spec's method-specific gate.

    Endpoints use this to populate the `dropped_warmup_n` response
    field — the legacy name is preserved across modes since the frontend
    subtitle renders it as "rows dropped to warmup" for walk-forward and
    "rows dropped (insufficient train history)" for train-test.

    in_sample    -> 0 (missing-value / insufficient-history drops are
                       structural, not method-specific)
    walk_forward -> rows with dropped_reason == "warmup"
    train_test   -> rows with dropped_reason == "insufficient_train_history"
    """
    if spec.kind == "in_sample":
        return 0
    target = "warmup" if spec.kind == "walk_forward" else "insufficient_train_history"
    return sum(1 for a in assignments if a.dropped_reason == target)


def make_spec(walk_forward: bool, cutoff_date: Optional[str] = None) -> BinningSpec:
    """Construct the appropriate BinningSpec from FastAPI query params.

    Precedence: `cutoff_date` (truthy string) wins over `walk_forward`.
    A cutoff implies train-test mode; walk_forward is ignored in that case.
    Endpoints accept both params and let this helper pick:

      cutoff_date set     -> TrainTestSpec(cutoff=parsed)
      walk_forward=True   -> WalkForwardSpec()
      neither             -> InSampleSpec()
    """
    if cutoff_date:
        from datetime import date as _date_cls
        return TrainTestSpec(cutoff=_date_cls.fromisoformat(str(cutoff_date)))
    if walk_forward:
        return WalkForwardSpec()
    return InSampleSpec()


def mode_envelope(spec, *, dropped: int = 0, universe: int = 0,
                  start_date: Any = None) -> dict:
    """Standard mode-aware metadata fields for spec-dispatched endpoints.

    Centralizes the 3-way conditional that every spec-dispatched endpoint
    used to inline at its return site. Callers spread the result into
    their response dict:

        return {
            "metrics": results,
            ...,
            **mode_envelope(spec, dropped=dropped, universe=universe,
                            start_date=filtered[0].get("trade_date", "")),
        }

    Six fields, uniform across modes:
      mode        — "in_sample" / "walk_forward" / "train_test"
      warmup      — int for walk_forward (250-day window), None otherwise
      cutoff_date — ISO string for train_test, None otherwise
      universe_n  — caller-supplied; rows with a defined bin
      start_date  — caller-supplied; first date in the universe

    Group 7 removed `dropped_warmup_n`. Under Encoding A (`wf_bins`'s
    `bin20 = 0` collapses warm-up + null into one sentinel), the count
    isn't computable from the stored bin alone. The frontend surfaces
    the exclusion with a static "Excludes walk-forward warm-up period"
    tagline instead (gated on pageMode === 'walk_forward'). Endpoints
    that previously called this with `dropped=` may keep doing so —
    the kwarg is accepted and ignored for back-compat — but new code
    should drop it.
    """
    return {
        "mode":        spec.kind,
        "warmup":      spec.warmup if spec.kind == "walk_forward" else None,
        "cutoff_date": spec.cutoff.isoformat() if spec.kind == "train_test" else None,
        "universe_n":  universe,
        "start_date":  start_date,
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
    *,
    primary_bin20_by_key: Optional[dict] = None,
) -> tuple[list, int, int]:
    """Primary filter for the secondary-endpoint family.

    Returns (filtered_rows, dropped_warmup_n, universe_n). Group 7
    drops the meaningful semantics of `dropped_warmup_n` — under
    Encoding A, warm-up exclusion is collapsed into the same sentinel
    as null-metric, so a separate count isn't computable from the
    stored bin alone. The field is kept in the tuple shape for caller
    bit-compatibility but always returned as 0 from the stored-bin
    path. The "Excludes walk-forward warm-up period" tagline (frontend)
    is what surfaces the exclusion to the user.

      in_sample: uses the frontend's `filtered_dates` (a list of
        "ticker|date" strings already narrowed by /analyze's in-sample
        primary bins). Output is in cached-row iteration order
        (NOT chronologically resorted).

      walk_forward (Group 7): when `primary_bin20_by_key` is supplied
        (the primary metric's bin20 lookup from `wf_bins`), filters rows
        whose stored `bin20 > 0 AND bin20 ∈ selected_primary_bins`.
        Output is chronologically sorted. The Encoding A `bin20 > 0`
        gate drops both warm-up and null-metric rows in one rule —
        same single-rule discipline as the IS surfaces.

      walk_forward (no lookup): returns empty — metric absent from wf_bins.

      train_test (no lookup): returns empty — metric absent from tt_bins.
    """
    from app.routers.oi_analysis import _filter_by_tkr_date, _parse_tkr_date_set
    # Group 7: stored-bin primary filter — fires for WF and TT when
    # the caller supplies a bin20 lookup.  IS stays on filtered_dates
    # because /analyze's trade_calendar has already done the bin
    # assignment + filter at the front end, so no lookup is needed.
    if (primary_bin20_by_key is not None and is_all
            and spec.kind in {"walk_forward", "train_test"}):
        sel = set(int(b) for b in (selected_primary_bins or []))
        kept: list = []
        # TT: skip training-window rows (trade_date < cutoff) — mirrors
        # the on-the-fly TT path's "pre_cutoff" exclusion so only
        # test-period rows reach the secondary panes.
        cutoff = getattr(spec, "cutoff", None)
        for r in rows:
            tkr = str(r.get("ticker", ""))
            td  = r.get("trade_date", "")
            if cutoff is not None:
                td_val = td if isinstance(td, _date) else _date.fromisoformat(str(td))
                if td_val < cutoff:
                    continue
            d_key = td.isoformat() if hasattr(td, "isoformat") else str(td)
            b20 = primary_bin20_by_key.get((tkr, d_key))
            if b20 is None or b20 <= 0:
                continue   # Encoding A: warm-up / null-metric
            if sel and b20 not in sel:
                continue
            kept.append(r)
        kept = _sort_chrono(kept)
        return kept, 0, len(rows)

    if spec.kind == "in_sample":
        filtered = _filter_by_tkr_date(rows, _parse_tkr_date_set(filtered_dates or []))
        return filtered, 0, len(rows)
    if spec.kind == "walk_forward":
        # No stored bin lookup supplied — metric absent from wf_bins.
        # Return empty rather than computing on the fly.
        return [], 0, 0
    if spec.kind == "train_test":
        # No stored bin lookup supplied — metric absent from tt_bins.
        # Return empty rather than computing on the fly.
        return [], 0, 0
    raise ValueError(f"unknown spec kind: {spec.kind!r}")


def assign_secondary_bin_stats(
    spec,
    rows_chrono: list,
    metric: str,
    n_bins: int,
    outcome_col: str,
    is_all: bool,
    *,
    all_rows=None,
    bin20_by_key: Optional[dict] = None,
) -> Optional[dict]:
    """Per-bin avg-return for ONE secondary metric inside an already-
    primary-filtered chronological subset. Returns {name, bins, bin_ns}
    or None if insufficient data.

    All modes use the stored-bin path when `bin20_by_key` is available.
    IS additionally accepts an `all_rows` lookup for the v9 fixed-thresholds
    path. Metric absent from stored bins → returns None.
    """
    # Group 7: stored-bin path hoisted above the spec dispatch — mode-
    # agnostic, runs for both IS (is_bins, Group 4) and WF (wf_bins,
    # Group 7). Also reached by the legacy v9 IS fixed-thresholds path
    # when `all_rows` is supplied but `bin20_by_key` isn't (the on-the-fly
    # per-ticker rank inside _bin_map_from_stored_bin20 fires when
    # bin20_by_key is None).
    if (bin20_by_key is not None and is_all) or (
            spec.kind == "in_sample" and all_rows is not None):
        bin_map = _bin_map_from_stored_bin20(
            all_rows if all_rows is not None else rows_chrono,
            metric, n_bins, is_all,
            outcome_col=outcome_col,
            bin20_by_key=bin20_by_key,
        )
        buckets: list = [[] for _ in range(n_bins)]
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
            tkr = str(r.get("ticker", ""))
            td = r.get("trade_date", "")
            d_key = td.isoformat() if hasattr(td, "isoformat") else str(td)
            bin_1idx = bin_map.get((tkr, d_key))
            if bin_1idx is None:
                continue
            buckets[bin_1idx - 1].append(fo)
        if all(len(b) == 0 for b in buckets):
            return None
        return {
            "name":   metric,
            "bins":   [round(float(np.mean(b)), 6) if b else 0.0 for b in buckets],
            "bin_ns": [len(b) for b in buckets],
        }
    # Metric absent from stored bins (null-by-design): no fallback.
    return None


def _bin_map_from_stored_bin20(
    rows: list,
    metric: str,
    n_bins: int,
    is_all: bool,
    *,
    outcome_col: Optional[str] = None,
    bin20_by_key: Optional[dict] = None,
) -> dict:
    """Build `(ticker, trade_date_str) → 1-indexed bin` lookup.

    Two paths share this entry point. Group 7 hoists the stored-bin
    check above the on-the-fly fallback because the math is identical
    whether the bin came from `is_bins` (IS) or `wf_bins` (WF); only
    the prefetch source differs upstream. Was named `_in_sample_bin_map`
    pre-Group-7 but the name was misleading — the stored-bin path is
    mode-agnostic.

    Stored-bin path (Group 4, IS) / (Group 7, WF): when `bin20_by_key`
    is provided AND `is_all` is True, the map is built directly from
    the stored bin20 lookup. Display granularity comes from the divisor
    formula `min(((bin20 - 1) * n_bins) // 20 + 1, n_bins)`, which is
    mathematically a direct per-ticker rank for every `n_bins` that
    divides 20 — and the secondary-panel UI only exposes
    `n_bins ∈ {5, 10, 20}`, all divisors. `outcome_col` validity is
    irrelevant in this branch because the upstream caller has already
    applied any outcome filter (cache filter for /sec endpoints; SQL
    filter for /analyze and /heatmap).

    Metric absent from stored bins: returns None (no fallback).
    """
    if bin20_by_key is not None:
        # Iterate the FILTERED bin20 dict, not all_rows. `bin20_by_key`
        # was built upstream from the filter-set rows in is_bins and
        # already has only the entries where bin20 > 0. So no nested
        # row scan, no hash lookups against the full universe: this
        # call's work is O(filter_size), not O(len(rows)). For a narrow
        # primary selection (~20K), this is ~10× cheaper than iterating
        # all_rows (~220K). For the no-selection case the two are equal
        # in count, but the dict-comp is still measurably faster than
        # the row-loop because there's no per-iteration `.get("ticker")`
        # / `hasattr(..., "isoformat")` / hash-into-a-dict-from-a-tuple
        # work — the keys are already tuples in the right format.
        return {
            key: min(((b20 - 1) * n_bins) // 20 + 1, n_bins)
            for key, b20 in bin20_by_key.items()
        }
    # Metric absent from stored bins (null-by-design): empty map.
    return {}


def assign_secondary_buckets(
    spec,
    rows_chrono: list,
    metric: str,
    n_bins: int,
    outcome_col: str,
    is_all: bool,
    *,
    all_rows=None,
    all_rows_sorted=None,
    rows_presorted: bool = False,
    bin20_by_key: Optional[dict] = None,
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
      train_test: bins fit on `all_rows` (full cache, pre-cutoff history
        available); buckets filled from `rows_chrono` (test-window rows).
    """
    # Group 7: stored-bin path hoisted above the spec dispatch — same
    # mode-agnostic trigger as assign_secondary_bin_stats. Also catches
    # the v9 IS fixed-thresholds path when `all_rows` is supplied
    # without a lookup (the helper's internal dispatch handles that).
    if bin20_by_key is not None or (
            spec.kind == "in_sample" and all_rows is not None):
        bin_map = _bin_map_from_stored_bin20(
            all_rows if all_rows is not None else rows_chrono,
            metric, n_bins, is_all,
            outcome_col=outcome_col,
            bin20_by_key=bin20_by_key,
        )
        buckets: list = [[] for _ in range(n_bins)]
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
            tkr = str(r.get("ticker", ""))
            td = r.get("trade_date", "")
            d_key = td.isoformat() if hasattr(td, "isoformat") else str(td)
            bin_1idx = bin_map.get((tkr, d_key))
            if bin_1idx is None:
                continue   # ticker excluded from fit, or row not in full pop
            buckets[bin_1idx - 1].append((fv, fo, d_key, tkr))
        # Match the legacy n_bins*2 gate: refuse the response when
        # the filtered subset is too small to be meaningful across
        # all bins. (Heatmap doesn't have this gate, but downstream
        # /secondary-detail consumers do.)
        if sum(len(b) for b in buckets) < n_bins * 2:
            return None
        return buckets

    # Metric absent from stored bins (null-by-design): no fallback.
    return None


def secondary_membership(
    spec,
    rows_chrono: list,
    metric: str,
    selected_bins,
    n_bins: int,
    is_all: bool,
    *,
    all_rows=None,
    bin20_by_key: Optional[dict] = None,
):
    """Binary 0/1 membership vector for ONE secondary metric inside an
    already-primary-filtered chronological subset. Returns a numpy float64
    array of length len(rows_chrono).

    Reads stored bin20 (is_bins / wf_bins). Metric absent from stored
    bins returns a zero vector — no on-the-fly fallback.
    """
    sel = set(selected_bins or [])
    # Group 7: stored-bin path hoisted above the spec dispatch — mode-
    # agnostic, identical math for IS (is_bins, Group 4) and WF
    # (wf_bins, Group 7). The v9 IS fixed-thresholds path still fires
    # when `all_rows` is supplied without a lookup.
    if bin20_by_key is not None or (
            spec.kind == "in_sample" and all_rows is not None):
        out = np.zeros(len(rows_chrono), dtype=np.float64)
        if not sel:
            return out
        bin_map = _bin_map_from_stored_bin20(
            all_rows if all_rows is not None else rows_chrono,
            metric, n_bins, is_all,
            outcome_col=None,
            bin20_by_key=bin20_by_key,
        )
        for i, r in enumerate(rows_chrono):
            tkr = str(r.get("ticker", ""))
            td = r.get("trade_date", "")
            d_key = td.isoformat() if hasattr(td, "isoformat") else str(td)
            bin_1idx = bin_map.get((tkr, d_key))
            if bin_1idx is not None and bin_1idx in sel:
                out[i] = 1.0
        return out
    # Metric absent from stored bins (null-by-design): zero membership vector.
    return np.zeros(len(rows_chrono), dtype=np.float64)


# ── Portfolio-aggregator vector builder (Step 5) ─────────────────────────
# /portfolios/{pid}/aggregate computes 0/1 bin-membership vectors over a
# fixed row set for every (system, primary_metric) + (system, secondary)
# pair. The original implementation forked on `walk_forward` three times
# inside the per-system loop because the modes had asymmetric semantics:
#
#   walk_forward / train_test:
#     - bin maps cached per (metric, n_bins) — multiple systems sharing a
#       metric reuse the same _walk_forward_bins / frozen-train result.
#     - secondary bins computed on the FULL row set, then ANDed with the
#       primary mask.
#
#   in_sample (legacy, pre-Group 6):
#     - secondary bins computed on the primary-FILTERED subset only,
#       then expanded back to the full-length vector — the re-rank-on-
#       filtered-subset shape that was fixed in /secondary-detail,
#       /secondary-corr-bins, and /secondary-correlation in v9 / Group 4.
#
# Group 6 closes the asymmetry for IS+ALL by accepting a pre-fetched
# `bin20_by_metric: {metric: {(ticker, date_str): bin20}}` lookup at
# construction. When the lookup has an entry for a metric, BOTH
# primary_vector and secondary_vector use the stored is_bins.bin20 over
# the FULL row set (same shape as WF/TT — full bin map, then AND with
# primary mask). Result: a row's secondary bin is a fixed property of
# its metric value against full-history thresholds, not a function of
# which rows survived the primary filter, AND the same stored bin
# reaches the heatmap, /analyze, /metric-bins, the corr explorer, and
# now portfolios. Metrics absent from is_bins return a zero vector —
# no on-the-fly fallback.
#
# `PortfolioVectorBuilder` encapsulates the path selection and the cache.
# Callers loop over systems and ask `primary_vector` / `secondary_vector`;
# the builder picks the right path by spec.kind + bin20_by_metric coverage.

class PortfolioVectorBuilder:
    """Per-portfolio 0/1 membership vector builder.

    Construct with the active spec and the portfolio's full row set. Each
    `primary_vector` / `secondary_vector` call returns a numpy float64
    vector of length `len(rows)`. Walk-forward and train-test mode cache
    the per-(metric, n_bins) bin map across calls so multiple systems that
    share a primary or secondary metric only pay the bisect_left cost once.

    `bin20_by_metric` is the Group 6 stored-bin lookup, populated by the
    caller via `_fetch_bin20_by_metric` over the portfolio's row set.
    When a metric appears as a key here, both primary and secondary
    vectors for that metric derive bins from the stored is_bins.bin20
    instead of computing per-ticker ranks on the fly. The display
    granularity comes from the canonical 20→N collapse formula
    `min(((b20-1)*n_bins)//20 + 1, n_bins)` — same one Group 4 used.
    Metrics absent from the dict (e.g. the 7 not-yet-populated features,
    or any WF/TT call since stored bins are IS-only) fall through to
    their existing on-the-fly paths.
    """

    def __init__(self, spec, rows: list, is_all: bool,
                 bin20_by_metric: Optional[dict] = None):
        self.spec = spec
        self.rows = rows
        self.is_all = is_all
        self._cache: dict = {}  # (metric, n_bins) -> bin map (wf / tt only)
        # Group 6: pre-fetched stored-bin lookup, scoped to this request.
        # Empty dict = no stored-bin path; metrics absent from the dict
        # fall through to their existing on-the-fly path.
        self._bin20_by_metric: dict = bin20_by_metric or {}

    def primary_vector(self, metric: str, selected_bins, n_bins: int):
        """0/1 vector over self.rows for one primary metric."""
        return self._full_vector(metric, selected_bins, n_bins)

    def secondary_vector(self, metric: str, selected_bins, n_bins: int,
                         primary_indices: list):
        """0/1 vector over self.rows for one secondary metric within
        the primary scope. `primary_indices` is the list of indices into
        self.rows where the primary's vector is 1.

        in_sample (legacy fallback): bins computed on rows[primary_indices],
          result expanded back to a full-length vector at those indices.
        in_sample (Group 6, stored bin available): same shape as WF/TT —
          full-set bin map then AND with primary mask.
        walk_forward / train_test: full-set bin map then AND with
          primary mask.
        """
        sel = set(selected_bins or [])
        # Group 6: when stored bin20 is available for this metric, the
        # IS path takes the same shape as WF/TT — full-set bin map, then
        # AND with primary mask. Closes the legacy re-rank-on-filtered-
        # subset asymmetry that the TODO above documented.
        if (self.spec.kind == "in_sample"
                and metric not in self._bin20_by_metric):
            # Metric absent from is_bins — return zero vector.
            # Never re-rank on the filtered subset.
            return np.zeros(len(self.rows))
        v_full = self._full_vector(metric, selected_bins, n_bins)
        prim_mask = np.zeros(len(self.rows))
        for i in primary_indices:
            prim_mask[i] = 1.0
        return v_full * prim_mask

    def primary_cleared_indices(self, metric: str, n_bins: int) -> set:
        """Indices where the primary metric has a defined bin under the
        current spec. Used by /portfolios/aggregate to compute the
        cross-system "dropped to warmup" count.

        For walk_forward / train_test: indices with a non-None bin map
          entry (cleared warmup AND have a valid metric value).
        For in_sample: empty set — legacy /portfolios/aggregate's
          in-sample path hardcodes wf_dropped=0 and wf_start=rows[0].date,
          which depends on cleared_any STAYING empty. Match that exactly.
        """
        if self.spec.kind == "in_sample":
            return set()
        bmap = self._bin_map(metric, n_bins)
        return {i for i, b in bmap.items() if b is not None}

    def _full_vector(self, metric: str, selected_bins, n_bins: int):
        """0/1 vector over self.rows.

        Group 7: stored-bin path hoisted above the spec dispatch. When
        `_bin20_by_metric[metric]` exists, derive each row's bin from
        the stored bin20 — the math is identical for IS (Group 4, from
        is_bins) and WF (Group 7, from wf_bins). For WF this replaces
        the on-the-fly `_walk_forward_bins` path; under Encoding A the
        stored `bin20 > 0` rule drops warm-up + null rows uniformly.

        Fallbacks below by spec.kind run when no stored lookup is
        supplied (legacy IS callers, WF/TT for metrics absent from the
        stored table, etc.).
        """
        sel = set(selected_bins or [])
        # Group 7: mode-agnostic stored-bin shortcut.
        bin20_by_key = self._bin20_by_metric.get(metric)
        if bin20_by_key is not None:
            v = np.zeros(len(self.rows))
            if not sel:
                return v
            for i, r in enumerate(self.rows):
                tkr = str(r.get("ticker", ""))
                td  = r.get("trade_date", "")
                d_key = td.isoformat() if hasattr(td, "isoformat") else str(td)
                b20 = bin20_by_key.get((tkr, d_key))
                if b20 is None or b20 <= 0:
                    continue   # Encoding A: warm-up + null sentinel
                # Same canonical 20→N collapse as Group 4 / heatmap /
                # /metric-bins / /analyze. For divisors of 20 this is
                # a direct per-ticker rank; UI exposes n_bins ∈ {5,10,20}.
                bin_n = min(((b20 - 1) * n_bins) // 20 + 1, n_bins)
                if bin_n in sel:
                    v[i] = 1.0
            return v
        # Fallbacks by spec.kind
        if self.spec.kind == "walk_forward" or self.spec.kind == "train_test":
            bmap = self._bin_map(metric, n_bins)
            v = np.zeros(len(self.rows))
            for i, b in bmap.items():
                if b is not None and b in sel:
                    v[i] = 1.0
            return v
        if self.spec.kind == "in_sample":
            # Metric absent from is_bins — return zero vector.
            return np.zeros(len(self.rows))
        raise ValueError(f"unknown spec kind: {self.spec.kind!r}")

    def _bin_map(self, metric: str, n_bins: int):
        """Cached bin map for walk_forward / train_test.

        walk_forward: per-ticker bisect_left against running history,
          252-day warmup.
        train_test: per-row `_bin_for_value` against the frozen per-ticker
          training history (rows with trade_date < cutoff).

        Both return `{row_idx_in_self.rows: bin_or_None}`.
        """
        key = (metric, n_bins)
        if key not in self._cache:
            if self.spec.kind == "walk_forward":
                self._cache[key] = _walk_forward_bins(
                    self.rows, metric, n_bins, self.is_all, self.spec.warmup,
                )
            elif self.spec.kind == "train_test":
                # Train-test bin map for portfolio aggregation. Test-only:
                # training-window rows (dropped_reason="pre_cutoff") are
                # mapped to bin=None in the cache so _full_vector
                # treats them like warmup rows in walk_forward (excluded
                # from system membership). The original bin is still on
                # the RowAssignment if a future side-by-side view wants it.
                assigner = TrainTestAssigner(self.spec)
                state = assigner.fit(self.rows, metric, n_bins, self.is_all)
                assigns = assigner.assign(self.rows, metric, n_bins, self.is_all,
                                          state, outcome_col=None)
                self._cache[key] = {
                    i: (a.bin if a.dropped_reason is None else None)
                    for i, a in enumerate(assigns)
                }
            else:
                raise ValueError(
                    f"_bin_map only meaningful for walk_forward / train_test; "
                    f"got spec.kind={self.spec.kind!r}"
                )
        return self._cache[key]


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


# ── Relocated helpers (Step 7j) ──────────────────────────────────────────
# These were previously defined in oi_analysis.py and called only by the
# Assigner classes / dispatch helpers above. Moving them here keeps all
# binning math in one module. (`_bucket_pairs`, `_bucket_pairs_per_ticker`,
# and `_walk_forward_thresholds` remain in oi_analysis.py because they're
# called from view-specific code in /analyze and /threshold-drift.)


def _walk_forward_bucket_pairs(pairs, n_bins_list, warmup):
    """Walk-forward equivalent of `_bucket_pairs` (single-ticker) for one or
    more bin counts simultaneously.

    Each pair is (x, y, date[, ticker]). Pairs MUST be in chronological order
    at the call site (we don't re-sort here — for ALL mode the caller iterates
    per-ticker chronologically).

    For each pair, the running per-history rank is computed via bisect_left
    against a sorted list of prior x values. After the per-history count
    reaches `max(warmup, max(n_bins_list))`, the bin is emitted as
    `min(int(rank / n_after * n_bins) + 1, n_bins)`. Pairs in warmup are
    skipped.

    Returns:
      assignments: list of (pair, {n_bins: bin_int}) for pairs that cleared
                   warmup, in the chronological order they were provided.
      dropped_warmup_n: count of pairs that didn't clear warmup.
    """
    max_bins = max(n_bins_list)
    warm = max(int(warmup), int(max_bins))
    assignments: list = []
    dropped = 0
    sorted_vals: list = []
    for pair in pairs:
        val = pair[0]
        rank = bisect.bisect_left(sorted_vals, val)
        bisect.insort(sorted_vals, val)
        n_after = len(sorted_vals)
        if n_after < warm:
            dropped += 1
            continue
        bins_for_pair = {nb: min(int(rank / n_after * nb) + 1, nb) for nb in n_bins_list}
        assignments.append((pair, bins_for_pair))
    return assignments, dropped


def _walk_forward_bucket_per_ticker(by_ticker, n_bins_list,
                                    warmup=None) -> tuple:
    """Walk-forward equivalent of `_bucket_pairs_per_ticker`.

    For each ticker, walks the pairs chronologically (sorted by trade_date)
    and emits walk-forward bin assignments at every requested granularity in
    a single bisect_left pass. Pairs whose per-ticker history hasn't reached
    `max(warmup, max(n_bins_list))` are dropped.

    Returns:
      buckets_per_granularity: dict {n_bins: list of n_bins buckets, each a
                                     list of pair tuples assigned to that bin}
      assignments_chrono: list of (pair, {n_bins: bin_int}) sorted
                          chronologically across all tickers — convenient for
                          building `pairs_with_d`-style structures downstream.
      dropped_warmup_n: total count of pairs dropped to warmup (across all
                        tickers).
    """
    if warmup is None:
        warmup = DEFAULT_WALKFWD_WARMUP
    buckets_by_n: dict = {nb: [[] for _ in range(nb)] for nb in n_bins_list}
    all_assignments: list = []
    dropped = 0
    for tkr_pairs in by_ticker.values():
        chrono = sorted(tkr_pairs, key=lambda p: p[2])
        a_t, d_t = _walk_forward_bucket_pairs(chrono, n_bins_list, warmup)
        dropped += d_t
        all_assignments.extend(a_t)
        for pair, bins_for_pair in a_t:
            for nb, b in bins_for_pair.items():
                buckets_by_n[nb][b - 1].append(pair)
    all_assignments.sort(key=lambda ab: ab[0][2])
    return buckets_by_n, all_assignments, dropped


def _bin_for_value(value, history_values: list, n_bins: int):
    """Return the 1..n_bins bin that `value` would occupy in `history_values`.

    Mirrors _bin_membership's ranking math: bin = min(int(rank / n * n_bins) + 1, n_bins)
    where rank is the position in the sorted ascending list (number of
    values strictly less than `value`).

    Returns None if value is None/NaN or history is too short (< n_bins).
    """
    if value is None:
        return None
    try:
        v = float(value)
        if math.isnan(v):
            return None
    except (TypeError, ValueError):
        return None
    if not history_values or len(history_values) < n_bins:
        return None
    rank = bisect.bisect_left(history_values, v)
    n = len(history_values)
    return min(int(rank / n * n_bins) + 1, n_bins)


def _walk_forward_bins(rows_chrono: list, metric: str, n_bins: int,
                       is_all: bool, warmup: int = DEFAULT_WALKFWD_WARMUP) -> dict:
    """Walk-forward bin assignment per row.

    Returns {row_index_in_input: bin_or_None}. For each row, the bin is
    computed using only data from prior dates AT THAT ROW'S TICKER
    (ALL mode) or all prior rows (single ticker). Rows whose group has
    < max(warmup, n_bins) prior observations get None and should be
    excluded from any downstream stats.

    Implementation: per group, iterate chronologically and maintain a
    sorted insertion list of prior metric values. For each new value,
    bisect_left gives the count strictly less than it (= its rank in
    [0, n_so_far)). After inserting, compute
    `min(int(rank / n_after_insert * n_bins) + 1, n_bins)` so the bin
    formula matches the in-sample _bin_membership exactly.

    Time complexity: O(N log N) per group (bisect_left is log; list
    insertion is O(N), but Python's list.insert is implemented in C so
    constants are small enough for our data sizes).
    """
    out: dict = {}
    n_bins = max(2, min(20, int(n_bins)))
    warm = max(int(warmup), n_bins)

    if is_all:
        groups: dict = {}
        for i, r in enumerate(rows_chrono):
            v = r.get(metric)
            if v is None:
                continue
            try:
                fv = float(v)
                if math.isnan(fv):
                    continue
            except (TypeError, ValueError):
                continue
            tkr = r.get("ticker", "_")
            groups.setdefault(tkr, []).append((i, fv))
    else:
        flat = []
        for i, r in enumerate(rows_chrono):
            v = r.get(metric)
            if v is None:
                continue
            try:
                fv = float(v)
                if math.isnan(fv):
                    continue
            except (TypeError, ValueError):
                continue
            flat.append((i, fv))
        groups = {"_": flat}

    for _tkr, items in groups.items():
        sorted_vals: list = []
        for orig_idx, value in items:
            rank = bisect.bisect_left(sorted_vals, value)
            bisect.insort(sorted_vals, value)
            n_after = len(sorted_vals)
            if n_after < warm:
                out[orig_idx] = None
            else:
                out[orig_idx] = min(int(rank / n_after * n_bins) + 1, n_bins)
    return out


def _sort_chrono(rows: list) -> list:
    """Stable chronological sort by (trade_date, ticker)."""
    return sorted(rows, key=lambda r: (r.get("trade_date", ""), r.get("ticker", "")))


def train_test_bin_matrix_per_ticker(
    rows_chrono: list, feature_cols: list, cutoff_date_str: str, n_bins: int,
) -> tuple:
    """Per-ticker train-test bin assignment over test-window rows.

    For one ticker's rows (chronologically sorted), compute per-feature
    bin assignments for all rows with trade_date >= cutoff. Bins are
    frozen from rows with trade_date < cutoff. Uses `searchsorted side='left'`
    against the sorted per-feature training distribution — the same
    primitive as `_bin_for_value` and the TrainTestAssigner.
    Tied test values land in the bin of the first
    equal training value (the lower bin among possible ties), matching
    standard percentile-rank semantics (fraction strictly less than v).

    Returns (test_rows, bin_mat):
      test_rows: list of input rows with trade_date >= cutoff, in input order.
      bin_mat:   (len(test_rows), len(feature_cols)) int32, 1-indexed
                 [1..n_bins]. Sentinel value 0 = either the feature had
                 fewer than n_bins valid training samples for this ticker,
                 or the test row's value for that feature is NaN.
    """
    n_bins = max(2, min(20, int(n_bins)))
    F = len(feature_cols)

    training_mask = np.array(
        [str(r.get("trade_date", "")) < cutoff_date_str for r in rows_chrono],
        dtype=bool,
    )
    test_mask = ~training_mask
    test_rows = [r for r, t in zip(rows_chrono, test_mask) if t]
    N_test = len(test_rows)
    bin_mat = np.zeros((N_test, F), dtype=np.int32)

    if N_test == 0 or F == 0:
        return test_rows, bin_mat

    n_train_total = int(training_mask.sum())
    if n_train_total < n_bins:
        return test_rows, bin_mat

    N = len(rows_chrono)
    X = np.empty((N, F), dtype=np.float64)
    for f_idx, feat in enumerate(feature_cols):
        for i, r in enumerate(rows_chrono):
            v = r.get(feat)
            if v is None:
                X[i, f_idx] = np.nan
                continue
            try:
                fv = float(v)
                X[i, f_idx] = fv if not math.isnan(fv) else np.nan
            except (TypeError, ValueError):
                X[i, f_idx] = np.nan

    test_indices = np.where(test_mask)[0]

    for f_idx in range(F):
        col = X[:, f_idx]
        valid_col = ~np.isnan(col)
        train_idx = np.where(training_mask & valid_col)[0]
        n_train = int(len(train_idx))
        if n_train < n_bins:
            continue
        sorted_train = np.sort(col[train_idx])
        ranks = np.searchsorted(sorted_train, col, side='left')
        bins_0idx = np.minimum(
            (ranks * n_bins // n_train).astype(np.int64),
            n_bins - 1,
        )
        test_col = col[test_indices]
        valid_test = ~np.isnan(test_col)
        test_bins_1idx = (bins_0idx[test_indices] + 1).astype(np.int32)
        bin_mat[:, f_idx] = np.where(valid_test, test_bins_1idx, 0)

    return test_rows, bin_mat
