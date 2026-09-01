"""Fail the build if the Equities Scalp page hardcodes a metric name.

WHY THIS GATE EXISTS. The scalp pipeline's metric set is explicitly unsettled:
five noise variants at three horizons at five statistics, plus flicker and flow
metrics, and a calibration exercise whose entire purpose is to delete most of
them. The brief for this page is blunt about it -- read the available metrics
from the database and render what's there, because a column list in code means
editing this project every time the pipeline changes.

The failure mode is what makes it worth a gate rather than a comment. A
hardcoded name that the pipeline renames does not raise. It pivots to a column
of NaN, the table renders, every cell reads em-dash, and the page looks like a
quiet day rather than a broken join. Nothing about that says "this metric no
longer exists" -- which is exactly the class of defect that has cost hours on
this dashboard before, twice, in the equity-IV work.

WHAT IS ALLOWED. Names may appear in:
  * app/scalp_metric_docs.py -- it is a VERBATIM vendored copy of the
    pipeline's own documentation map, whose whole job is to list them. It is
    checked for drift by check_vendored.py instead, which is the right check
    for a file that is supposed to be a list of names.
  * app/scalp_columns.py -- the declared exception, on the same terms
    app/equity_presets.py holds for the IV page. 232 metrics cannot all be
    rendered, so SOMETHING has to say which ones the table opens on. What
    makes it safe is that the names there are CANDIDATES resolved against the
    live catalog at request time, with an unresolvable role dropped and
    reported rather than rendered empty -- and that this script verifies every
    one of them against metric_docs, so a rename upstream fails the build.
  * comments and docstrings, where a name is an EXAMPLE rather than a lookup.
    A metric named in prose cannot silently return nulls.

So this scans executable string literals only, and only in the files that
could actually query with one.

IT ALSO CHECKS THE FILTER SET. The candidates endpoint declares the
pipeline's five read-time thresholds as explicit query parameters, because
FastAPI can only validate what it can see. That is a second list that can go
stale: a threshold added to DEFAULT_FILTERS and not declared here would be
unreachable from the page while looking, from the config, as though it were
in force.
"""
from __future__ import annotations

import ast
import io
import re
import sys
import tokenize
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

PY_TARGETS = [ROOT / "app" / "routers" / "equities_scalp.py"]
JS_TARGETS = [ROOT / "static" / "js" / "equities_scalp.js"]

# WHAT COUNTS AS A METRIC NAME is not decided here. The vendored
# metric_docs IS the pipeline's own answer to that question: an EXACT dict of
# every fixed metric and a PATTERNS list matching every generated one, kept in
# step by check_vendored.py. Asking it is both more accurate than a stem list
# and self-maintaining -- a family added upstream becomes checkable here the
# moment the file is re-vendored.
#
# A hand-written stem list was the first attempt and it flagged 'ratio',
# 'noise_bps' and 'ratio_metrics': a family word, a regex fragment and a JSON
# response key, none of which is a metric. The authority was wrong, not the
# threshold.
sys.path.insert(0, str(ROOT))
from app import scalp_metric_docs                       # noqa: E402

# THE ONE HOLE, stated rather than hidden: a metric present in the database
# but absent from metric_docs is not recognised, so hardcoding one would pass.
# /meta reports those as `undocumented` on the page itself, which is where an
# undocumented metric is a problem worth seeing anyway.


def _looks_like_metric(s: str) -> bool:
    if not s or len(s) < 4 or s != s.lower():
        return False
    return scalp_metric_docs.describe(s) is not None


def scan_python(path: Path) -> list[str]:
    """String literals in EXECUTABLE positions. Docstrings are skipped."""
    src = path.read_text(encoding="utf-8")
    tree = ast.parse(src)

    # Every docstring node, by identity, so a name used as an example in prose
    # is not mistaken for one used as a lookup.
    docstrings = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef,
                             ast.ClassDef)):
            doc = ast.get_docstring(node, clean=False)
            if doc is not None and node.body:
                first = node.body[0]
                if isinstance(first, ast.Expr):
                    docstrings.add(id(first.value))

    bad = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Constant) or not isinstance(node.value, str):
            continue
        if id(node) in docstrings:
            continue
        if _looks_like_metric(node.value):
            bad.append(f"{path.relative_to(ROOT)}:{node.lineno}: "
                       f"metric literal {node.value!r}")
    return bad


def scan_js(path: Path) -> list[str]:
    """JS has no AST here, so comments are stripped and literals scanned.

    Crude on purpose: the alternative is a JS parser dependency for a file
    whose only requirement is that it contains no metric names at all.
    """
    src = path.read_text(encoding="utf-8")
    src = re.sub(r"/\*.*?\*/", "", src, flags=re.S)
    src = re.sub(r"(?m)//.*$", "", src)

    bad = []
    for m in re.finditer(r"""(['"`])([A-Za-z0-9_]+)\1""", src):
        if _looks_like_metric(m.group(2)):
            line = src[:m.start()].count("\n") + 1
            bad.append(f"{path.relative_to(ROOT)}:{line}: "
                       f"metric literal {m.group(2)!r}")
    return bad


def self_test() -> int:
    """Prove the matcher fires before trusting a clean run.

    A gate that silently matches nothing passes forever. These are the exact
    shapes the pipeline emits, and the schema words it must not flag.
    """
    must_flag = ["noise_bps_tw_mid_10s_rms", "ratio_tw_mid_10s",
                 "spread_bps_tw", "trades_per_min", "odd_lot_share",
                 "bid_lifetime_ms_median", "move_rate_tw_mid_10s",
                 "quote_bucket_coverage_10s", "two_sided_balance",
                 "off_exchange_share"]
    # The four that a hand-written stem list flagged and should not have: a
    # family word, a regex fragment, a JSON key, and a schema column.
    must_pass = ["trade_date", "symbol", "metric", "value", "connected",
                 "latest_date", "GET", "date", "rms", "tw_mid",
                 "ratio", "noise_bps", "ratio_metrics", "ratio_guard",
                 "noise_metrics", "default_noise", "read_time_only"]
    bad = 0
    for s in must_flag:
        if not _looks_like_metric(s):
            print(f"  SELF-TEST: {s!r} should be flagged and is not")
            bad += 1
    for s in must_pass:
        if _looks_like_metric(s):
            print(f"  SELF-TEST: {s!r} should pass and is flagged")
            bad += 1
    if not bad:
        print("self-test: the metric matcher fires on real names and not on "
              "schema words")
    return bad


def check_column_roles() -> int:
    """Every name in scalp_columns must be one metric_docs recognises.

    This is what buys the exception. A candidate list is only safer than a
    hardcoded column if something notices when a candidate stops existing --
    otherwise a role silently falls through to its second choice, or to
    nothing, and the table quietly loses a column.

    Templates are checked by FORMATTING them with a variant, horizon and
    statistic the pipeline emits, so a family that has been renamed or had its
    suffix moved fails rather than being checked as a string with braces in it.
    """
    from app import scalp_columns

    bad = 0
    for name in scalp_columns.literals():
        if scalp_metric_docs.describe(name) is None:
            bad += 1
            print(f"\n  app/scalp_columns.py names {name!r}, which metric_docs")
            print("    does not recognise. Either it was renamed upstream and")
            print("    the candidate is stale, or it never existed.")
    for name in scalp_columns.template_examples():
        if scalp_metric_docs.describe(name) is None:
            bad += 1
            print(f"\n  a template in app/scalp_columns.py produces {name!r},")
            print("    which metric_docs does not recognise — the family has")
            print("    been renamed or its shape has changed.")

    # A role with neither candidates nor templates can never resolve, so it is
    # a column that will always be reported missing.
    for r in scalp_columns.ROLES:
        if not r.candidates and not r.templates:
            bad += 1
            print(f"\n  role {r.key!r} has no candidates and no templates")
        if not r.note:
            bad += 1
            print(f"\n  role {r.key!r} has no note — every column on this page")
            print("    has to say what it is for")

        # A role key that is ALSO a metric name makes the merge ambiguous.
        # Role keys and the raw metric names the column chooser adds share one
        # namespace in the response, so "is this column a role or a metric?"
        # has to be answerable, and a key like `trades_per_min` -- which is
        # both -- means it is not.
        if scalp_metric_docs.describe(r.key) is not None:
            bad += 1
            print(f"\n  role key {r.key!r} is also a metric name. Role keys and")
            print("    chooser-added metric names share a namespace, so this")
            print("    makes a column's own identity ambiguous. Rename the")
            print("    role — the metric keeps its name.")
    return bad


def check_filter_params() -> int:
    """The endpoint's declared thresholds must still be the pipeline's set."""
    import inspect
    from app import scalp_config
    from app.routers import equities_scalp as sc

    declared = set(inspect.signature(sc.candidates).parameters)
    expected = set(scalp_config.DEFAULT_FILTERS)
    bad = 0
    for missing in sorted(expected - declared):
        bad += 1
        print(f"\n  DEFAULT_FILTERS has {missing!r} and /candidates does not")
        print("    declare it. The threshold would look active in the config")
        print("    and be unreachable from the page.")
    for extra in sorted(set(sc._FILTER_ROLES) - expected):
        bad += 1
        print(f"\n  /candidates joins {extra!r} to a column, but it is not in")
        print("    DEFAULT_FILTERS — a filter with no threshold behind it.")
    return bad


def main() -> int:
    bad = self_test()
    bad += check_column_roles()
    bad += check_filter_params()
    findings: list[str] = []
    checked = 0
    for p in PY_TARGETS:
        if p.is_file():
            checked += 1
            findings += scan_python(p)
    for p in JS_TARGETS:
        if p.is_file():
            checked += 1
            findings += scan_js(p)

    for f in findings:
        print(f"\n  {f}")
    if findings:
        print("\n  The metric set is read from the database, not declared. A")
        print("  name written here becomes a column of nulls — not an error —")
        print("  the day the pipeline renames it. Fetch it from /meta instead.")
        print("  If a name genuinely belongs in prose, move it into a comment")
        print("  or a docstring, where it cannot be used as a lookup.")

    print(f"\nfiles scanned: {checked}, metric literals: {len(findings)}")
    return 1 if (findings or bad) else 0


sys.exit(main())
