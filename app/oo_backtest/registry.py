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
  bins        {"edges", "labels", "right"} for a range metric, from the same
              *_bin_spec() the pandas path uses. `edges` are the INNER edges;
              both outer edges are +/-inf, which JSON cannot carry. None for a
              categorical metric.
  categories  fixed [{"value", "label"}] for a categorical metric, or None
              when the categories come from the loaded data (year, exit reason)
  hasScatter  whether the section draws P/L vs metric with an OLS line
  section     renders a metric-analysis section
  filter      renders a sidebar filter
  winRate     the section also shows win rate by bin
  format      how a value is displayed: "pct" | "usd" | "ratio" | "num" | "int"
  minDate     earliest date this metric has coverage for, ISO, or None.

WHY minDate EXISTS WHEN EVERY VALUE IS None. Skew comes back later with
history from 2020, and some SharpTwo metrics from 2023. Filtering the whole
page on a 2023+ column silently discards 2021-2022 trades, which is what the
source app's cross-filtering toggles were for. The field is here so that
metric can scope its filter or warn about the truncation, instead of the
toggle UI being rebuilt.
"""
from __future__ import annotations

import math

from app.oo_backtest import calculations as calc

_DOW = [{"value": i, "label": d} for i, d in enumerate(["Mon", "Tue", "Wed", "Thu", "Fri"])]


def _bins(spec: dict) -> dict:
    edges = spec["bins"]
    assert math.isinf(edges[0]) and edges[0] < 0 and math.isinf(edges[-1]) and edges[-1] > 0
    return {"edges": edges[1:-1], "labels": spec["labels"], "right": spec["right"]}


def _range(key, label, column, lo, hi, step, spec, fmt):
    return {"key": key, "label": label, "column": column, "type": "range",
            "min": lo, "max": hi, "step": step, "bins": _bins(spec), "categories": None,
            "hasScatter": True, "section": True, "filter": True, "winRate": False,
            "format": fmt, "minDate": None}


def _categorical(key, label, column, categories, *, section, filter, win_rate=False):
    return {"key": key, "label": label, "column": column, "type": "categorical",
            "min": None, "max": None, "step": None, "bins": None, "categories": categories,
            "hasScatter": False, "section": section, "filter": filter, "winRate": win_rate,
            "format": "int" if key != "exit_reason" else "text", "minDate": None}


def build_registry() -> list[dict]:
    """Section order is list order. Filter order is list order too."""
    return [
        _categorical("day_of_week", "Day of Week", "day_of_week", _DOW, section=True, filter=True),
        _categorical("exit_reason", "Exit Reason", "exit_reason", None, section=False, filter=True),
        _categorical("year", "P&L by Year", "year", None, section=True, filter=False, win_rate=True),
        _range("gap", "SPX Overnight Gap", "gap", -3.0, 3.0, 0.1, calc.gap_bin_spec(), "pct"),
        _range("vix_gap", "VIX Overnight Gap", "vix_overnight_gap", -15.0, 15.0, 0.5,
               calc.vix_gap_bin_spec(), "pct"),
        _range("premium", "Premium", "premium", -2000, 2000, 50, calc.premium_bin_spec(), "usd"),
        _range("vix", "VIX Level", "vix_level", 9, 80, 1, calc.vix_bin_spec(), "num"),
        _range("vix3m", "VIX3M Level", "vix3m_level", 9, 80, 1, calc.vix_bin_spec(), "num"),
        _range("vix9d", "VIX9D Level", "vix9d_level", 9, 80, 1, calc.vix_bin_spec(), "num"),
        _range("vix3m_vix", "VIX3M/VIX Ratio", "vix3m_vix_ratio", 0.5, 2.0, 0.01,
               calc.ratio_bin_spec(), "ratio"),
        _range("vix_vix9d", "VIX/VIX9D Ratio", "vix_vix9d_ratio", 0.5, 2.0, 0.01,
               calc.ratio_bin_spec(), "ratio"),
    ]


REGISTRY = build_registry()


def registry_columns() -> list[str]:
    return [m["column"] for m in REGISTRY]
