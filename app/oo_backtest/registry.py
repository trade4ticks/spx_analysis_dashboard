"""The metric registry: ONE list that the filter sidebar, the metric sections
and any correlation table all render from.

The source app keyed each metric across five parallel structures
(BUILDER_FILTERS, RANGE_COL_MAP, RANGE_DEFAULTS, RANGE_STEP_MAP,
METRIC_BADGE_COLORS) plus bin arrays inline in its render callback, so adding
a metric meant six edits. Here it is one entry below plus, if the column is
new, the join that produces it.

Fields
  key         stable identifier (filter state and section ids are keyed on it)
  label       display name
  column      trade-row column
  type        "range" | "categorical"
  min/max     static fallback bounds for a range control. The page replaces
              them with the loaded data's own extent; these apply only when a
              column has no values to take an extent from.
  step        slider step
  bins        {"edges", "labels", "closed", "labelEdge"} for a range metric,
              from the same *_bin_spec() the pandas path uses. `edges` are the
              INNER edges; both outer edges are +/-inf, which JSON cannot
              carry. `closed` is "left" ([a, b)) or "right" ((a, b]) and is
              stated here rather than left to a pd.cut default. None for a
              categorical metric.
  basis       {"entry": col, "close": col} or None. The ratios are computed
              both from the entry-time bars and from daily closes until one is
              chosen; `column` is the default (entry) and the sidebar toggle
              picks which one the sections read.
  series      index_ohlc series the metric is built from; minDate is the
              latest first-coverage date among them.
  categories  fixed [{"value", "label"}] for a categorical metric, or None
              when the categories come from the loaded data (year, exit reason)
  hasScatter  whether the section draws P/L vs metric with an OLS line
  section     renders a metric-analysis section
  filter      renders a sidebar filter
  winRate     the section also shows win rate by bin
  format      how a value is displayed: "pct" | "usd" | "ratio" | "num" | "int"
  minDate     earliest date this metric has coverage for, ISO, or None.
  pane        an extra, non-metric chart drawn in this section's row, or None.
              "deployment" is the concurrent-positions chart; it sits beside
              Day of Week so the page names no metric to place it.

minDate IS FILLED AT REQUEST TIME (registry_with_coverage) from the first
non-null close per index_ohlc series, not from the table's date range: a
series backfilled later would otherwise truncate a filter silently. A range
filter drops trades with no value, so the page warns when a log has trades
before a metric's minDate. The same field carries skew (2020+) and SharpTwo
(2023+) when they return -- which is what the source app's cross-filtering
toggles were for, without rebuilding them.
"""
from __future__ import annotations

import copy
import math

from app.oo_backtest import calculations as calc

_DOW = [{"value": i, "label": d} for i, d in enumerate(["Mon", "Tue", "Wed", "Thu", "Fri"])]


def _bins(spec: dict, label_edge: str = "both") -> dict:
    edges = spec["bins"]
    assert math.isinf(edges[0]) and edges[0] < 0 and math.isinf(edges[-1]) and edges[-1] > 0
    return {"edges": edges[1:-1], "labels": spec["labels"],
            "closed": "right" if spec["right"] else "left", "labelEdge": label_edge}


def _range(key, label, column, lo, hi, step, spec, fmt, series, basis=None, label_edge="both"):
    return {"key": key, "label": label, "column": column, "type": "range",
            "min": lo, "max": hi, "step": step, "bins": _bins(spec, label_edge), "categories": None,
            "hasScatter": True, "section": True, "filter": True, "winRate": False,
            "format": fmt, "minDate": None, "series": series, "basis": basis, "pane": None}


def _categorical(key, label, column, categories, *, section, filter, win_rate=False, pane=None):
    return {"key": key, "label": label, "column": column, "type": "categorical",
            "min": None, "max": None, "step": None, "bins": None, "categories": categories,
            "hasScatter": False, "section": section, "filter": filter, "winRate": win_rate,
            "format": "int" if key != "exit_reason" else "text", "minDate": None,
            "series": [], "basis": None, "pane": pane}


def ratio_basis(stem: str) -> dict:
    return {"entry": f"{stem}_entry", "close": f"{stem}_close"}


def build_registry() -> list[dict]:
    """Section order is list order. Filter order is list order too."""
    return [
        _categorical("day_of_week", "Day of Week", "day_of_week", _DOW, section=True, filter=True,
                     pane="deployment"),
        _categorical("exit_reason", "Exit Reason", "exit_reason", None, section=False, filter=True),
        _categorical("year", "P&L by Year", "year", None, section=True, filter=False, win_rate=True),
        _range("gap", "SPX Overnight Gap", "gap", -3.0, 3.0, 0.1, calc.gap_bin_spec(), "pct", ["spx"]),
        _range("vix_gap", "VIX Overnight Gap", "vix_overnight_gap", -15.0, 15.0, 0.5,
               calc.vix_gap_bin_spec(), "pct", ["vix"]),
        _range("premium", "Premium", "premium", -2000, 2000, 50, calc.premium_bin_spec(), "usd", []),
        _range("vix", "VIX Level", "vix_level", 9, 80, 1, calc.vix_bin_spec(), "num", ["vix"]),
        _range("vix3m", "VIX3M Level", "vix3m_level", 9, 80, 1, calc.vix_bin_spec(), "num", ["vix3m"]),
        _range("vix9d", "VIX9D Level", "vix9d_level", 9, 80, 1, calc.vix_bin_spec(), "num", ["vix9d"]),
        # Right-closed and labelled by the LEFT edge, as the source app had
        # them: the bar labelled "0.74" holds (0.74, 0.78], so it does not
        # contain the value on its label. Kept for continuity with the old
        # app; stated here so it is a decision, not a pd.cut default.
        _range("vix3m_vix", "VIX3M/VIX Ratio", "vix3m_vix_ratio_entry", 0.5, 2.0, 0.01,
               calc.ratio_bin_spec(), "ratio", ["vix", "vix3m"],
               basis=ratio_basis("vix3m_vix_ratio"), label_edge="left"),
        _range("vix_vix9d", "VIX/VIX9D Ratio", "vix_vix9d_ratio_entry", 0.5, 2.0, 0.01,
               calc.ratio_bin_spec(), "ratio", ["vix", "vix9d"],
               basis=ratio_basis("vix_vix9d_ratio"), label_edge="left"),
    ]


REGISTRY = build_registry()


def registry_columns() -> list[str]:
    cols = []
    for m in REGISTRY:
        cols += list(m["basis"].values()) if m["basis"] else [m["column"]]
    return cols


def registry_with_coverage(coverage: dict | None) -> list[dict]:
    """REGISTRY with minDate filled from index_ohlc's real per-series coverage.

    A metric built from several series is covered from the LATEST of their
    first dates. With no coverage (market data unreachable) minDate stays None.
    """
    out = copy.deepcopy(REGISTRY)
    if not coverage:
        return out
    for m in out:
        firsts = [coverage.get(s) for s in m["series"]]
        if firsts and all(firsts):
            m["minDate"] = max(firsts)
    return out
