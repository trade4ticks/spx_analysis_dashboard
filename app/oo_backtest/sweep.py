"""Parameter sweeps: several trade logs of one strategy, loaded as one trade set.

A sweep is the same strategy run with varied parameters (delta, profit
target, DTE...), one log per variant. Every trade carries a `variant` id; the
payload carries the variant table. Everything else -- the market join, the
payload, the page's filtering -- is the single-log path applied to more rows.

WHERE A VARIANT'S PARAMETERS COME FROM, per file, first match wins:

  1. param_* columns (param_delta, param_pt) -- how OptionOmega's MCP API
     exports arrive. Discovered, never listed here. One file may hold several
     variants: each distinct combination of its param_* values is one.
  2. The filename, when the file has no param_* columns and the batch has more
     than one file: `sweep_dte45_pt50_sl200.csv` is dte=45, pt=50, sl=200.
     What was read from each name is reported, so a misread is visible.
  3. Neither: one unnamed variant -- which, for a single file, is exactly the
     single-log page as it was (no `variant` column, `sweep` is None).

A single file's NAME is never read: every log ever uploaded would otherwise
start sprouting parameters from its name ("v2" in a MesoSim BacktestName).

REFUSED, loading nothing, naming the file(s):
  * files that resolve to different parameter SETS (dte/pt vs dte/sl)
  * two files resolving to the same values (the same log dropped twice, or a
    filename that does not say what differs)
  * a param_* cell left blank (that row's variant is unknown)
  * the same key twice in one filename, or two param_* columns that differ
    only in case
"""
from __future__ import annotations

import math
import re
from collections import Counter
from pathlib import PurePath

import pandas as pd

PARAM_PREFIX = "param_"
VARIANT_COLUMN = "variant"
MAX_FILES = 200
# Refused above this, uncompressed. Parsing runs at roughly 15-20 MB/s here and
# about half that on the VPS, and the Cloudflare tunnel ends a request at 100 s:
# 300 MB leaves the market join, the payload and the encode their share. 75
# Option Omega CSVs of 3,200 trades are ~40 MB; one MesoSim JSON is ~57 MB.
MAX_BATCH_BYTES = 300 * 1024 * 1024


class SweepError(ValueError):
    """A batch that cannot be loaded as one sweep. The message names the file."""


# ── values ──────────────────────────────────────────────────────────────────

def clean_value(v):
    """A parameter value as JSON: an int if integral, a float, else a string.
    None for a blank. Numeric strings become numbers, so "45" from one file and
    45 from another are the same variant."""
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return None
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)) or hasattr(v, "item"):
        try:
            f = float(v)
        except (TypeError, ValueError):
            return str(v)
        if math.isnan(f):
            return None
        if math.isinf(f):
            return str(v)
        return int(f) if f.is_integer() else f
    s = str(v).strip()
    if not s:
        return None
    try:
        f = float(s)
    except ValueError:
        return s
    if math.isnan(f) or math.isinf(f):
        return s
    return int(f) if f.is_integer() else f


def value_text(v) -> str:
    return format(v, ".12g") if isinstance(v, float) else str(v)


def sort_key(v):
    """Numbers ascending, then strings: 5 < 10 < 45, never "10" < "5"."""
    return (0, v, "") if isinstance(v, (int, float)) else (1, 0, str(v))


def label(params: list[str], values: tuple) -> str:
    return " · ".join(f"{k}={value_text(v)}" for k, v in zip(params, values))


# ── filenames ───────────────────────────────────────────────────────────────

# A token boundary is whitespace, _ , ; or a hyphen BEFORE A LETTER: so
# "sweep-dte45" splits but "off-10" and "2021-2026" stay whole.
_SPLIT = re.compile(r"[\s_,;]+|-(?=[A-Za-z])")
_NUM = r"-?(?:\d+(?:\.\d*)?|\.\d+)"
_GLUED = re.compile(rf"^([A-Za-z]+)({_NUM})$")      # dte45, delta0.16
_EQUALS = re.compile(r"^([A-Za-z][A-Za-z0-9]*)=(.+)$")  # dte=45, side=put
_NUMBER = re.compile(rf"^{_NUM}$")
_WORD = re.compile(r"^[A-Za-z]+$")


def params_from_filename(filename: str) -> dict:
    """{"params": {key: value}, "order": [keys], "ignored": [tokens]}.

    `dte45`, `dte=45` and `dte_45` are all dte=45. Keys are lower-cased. A
    token that is none of those is IGNORED and reported, never guessed at.
    Raises SweepError on a key given twice.
    """
    stem = PurePath(filename).stem
    tokens = [t for t in _SPLIT.split(stem) if t]
    params: dict = {}
    order: list[str] = []
    ignored: list[str] = []

    def put(k, v, tok):
        k = k.lower()
        if k in params:
            raise SweepError(f'{filename}: "{k}" appears twice in the filename '
                             f"({value_text(params[k])} and {value_text(v)}).")
        params[k] = v
        order.append(k)

    i = 0
    while i < len(tokens):
        t = tokens[i]
        m = _EQUALS.match(t)
        if m and clean_value(m.group(2)) is not None:
            put(m.group(1), clean_value(m.group(2)), t)
        elif (m := _GLUED.match(t)):
            put(m.group(1), clean_value(m.group(2)), t)
        elif _WORD.match(t) and i + 1 < len(tokens) and _NUMBER.match(tokens[i + 1]):
            put(t, clean_value(tokens[i + 1]), t)
            i += 1
        else:
            ignored.append(t)
        i += 1
    return {"params": params, "order": order, "ignored": ignored}


# ── columns ─────────────────────────────────────────────────────────────────

def param_columns(df: pd.DataFrame, filename: str) -> list[tuple[str, str]]:
    """[(column, key)] for every param_* column, in file order."""
    out, seen = [], {}
    for c in df.columns:
        name = str(c)
        if not name.lower().startswith(PARAM_PREFIX):
            continue
        key = name[len(PARAM_PREFIX):].strip().lower()
        if not key:
            raise SweepError(f'{filename}: column "{name}" names no parameter.')
        if key in seen:
            raise SweepError(f'{filename}: columns "{seen[key]}" and "{name}" are the same parameter.')
        seen[key] = name
        out.append((name, key))
    return out


# ── the batch ───────────────────────────────────────────────────────────────

def _merge_notes(notes: list[dict]) -> dict:
    """Per-file parse notes as one: counts add, lists concatenate, a string
    every file agrees on stays, differing strings are listed once each."""
    out: dict = {}
    for k in dict.fromkeys(k for n in notes for k in n):
        vals = [n[k] for n in notes if k in n]
        if all(isinstance(v, bool) for v in vals):
            out[k] = any(vals)
        elif all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in vals):
            out[k] = sum(vals)
        elif all(isinstance(v, list) for v in vals):
            out[k] = [x for v in vals for x in v]
        else:
            distinct = list(dict.fromkeys(str(v) for v in vals if v not in (None, "")))
            out[k] = " / ".join(distinct) if distinct else None
    return out


def _suggested_name(files: list[str], n_variants: int) -> str:
    stems = [PurePath(f).stem for f in files]
    common = stems[0]
    for s in stems[1:]:
        while not s.startswith(common):
            common = common[:-1]
    common = common.rstrip(" _-,;=.")
    # Strip a trailing half-token: "sweep_dte4" from dte45 / dte40.
    common = re.sub(r"[_\-\s]?[A-Za-z]+\d*$", "", common) if common and common[-1:].isalnum() and len(set(stems)) > 1 else common
    common = common.rstrip(" _-,;=.")
    return common if len(common) >= 3 else f"Sweep of {n_variants} variants"


def combine(parsed: list[tuple[str, pd.DataFrame]]) -> tuple[pd.DataFrame, dict | None]:
    """[(filename, parsed frame)] -> (one frame, sweep) or SweepError.

    `sweep` is None when there is nothing to tell apart (one file, no param_*
    columns) -- the frame is then the file's own, unchanged. Otherwise the
    frame gains an int `variant` column and `sweep` is
    {params, variants: [{id, params, label, files, n}], files: [report]}.
    """
    if not parsed:
        raise SweepError("No files.")
    if len(parsed) > MAX_FILES:
        raise SweepError(f"{len(parsed)} files is more than the {MAX_FILES} one batch takes.")
    names = [f for f, _ in parsed]
    dup = [f for f, c in Counter(names).items() if c > 1]
    if dup:
        raise SweepError(f"{dup[0]} is in the batch twice.")

    multi = len(parsed) > 1
    per_file = []        # (filename, df, keys tuple, source, row_values | const, report)
    for filename, df in parsed:
        notes = dict(df.attrs.get("parse_notes") or {})
        report = {"file": filename, "trades": int(len(df)),
                  "ignored_dropped": notes.get("ignored_dropped"), "source": "none",
                  "params": None, "ignored_tokens": [], "variants": 1}
        cols = param_columns(df, filename)
        if cols:
            keys = tuple(k for _, k in cols)
            values = []
            for col, key in cols:
                # Each distinct cell cleaned once (an MCP file repeats a
                # handful of values over thousands of rows).
                codes, uniq = pd.factorize(df[col], use_na_sentinel=False)
                cu = [clean_value(u) for u in uniq]
                v = [cu[k] for k in codes.tolist()]
                blank = sum(1 for x in v if x is None)
                if blank:
                    raise SweepError(f'{filename}: {blank} of {len(v)} rows have no value in "{col}", '
                                     f"so their variant is unknown.")
                values.append(v)
            rows = list(zip(*values)) if values else []
            report.update(source="columns", variants=len(set(rows)),
                          columns=[c for c, _ in cols])
            per_file.append((filename, df, keys, rows, report))
        elif multi:
            fp = params_from_filename(filename)
            keys = tuple(fp["order"])
            const = tuple(fp["params"][k] for k in keys)
            report.update(source="filename" if keys else "none", params=fp["params"] or None,
                          ignored_tokens=fp["ignored"])
            per_file.append((filename, df, keys, [const] * len(df), report))
        else:
            per_file.append((filename, df, (), [()] * len(df), report))

    if not multi and not per_file[0][2]:
        return parsed[0][1], None

    # One parameter SET for the whole batch (order aside: dte/pt == pt/dte).
    sets = [frozenset(k) for _, _, k, _, _ in per_file]
    counts = Counter(sets)
    ref = max(counts, key=lambda s: (counts[s], -sets.index(s)))
    if not ref:
        raise SweepError(f"None of these {len(per_file)} files carries parameters -- no param_* columns, and "
                         f"nothing like dte45 or pt=50 in the filenames -- so they cannot be told apart. "
                         f"Rename them, e.g. sweep_dte45_pt50.csv.")
    odd = [(f, s) for (f, *_), s in zip(per_file, sets) if s != ref]
    if odd:
        f, s = odd[0]
        more = f" (and {len(odd) - 1} more)" if len(odd) > 1 else ""
        have = ", ".join(sorted(s)) or "no parameters"
        raise SweepError(f"{f}{more} has {have}; the other files have {', '.join(sorted(ref))}. "
                         f"Every file in a sweep must vary the same parameters. Nothing was loaded.")

    # Key order: the first file's.
    params = list(per_file[0][2])
    # Every variant, and which file(s) it came from.
    owner: dict = {}
    for filename, _, keys, rows, _ in per_file:
        pos = [keys.index(k) for k in params]
        for vals in dict.fromkeys(rows):
            v = tuple(vals[p] for p in pos)
            if v in owner and owner[v] != filename:
                raise SweepError(f"{owner[v]} and {filename} are the same variant ({label(params, v)}). "
                                 f"Each file must differ in at least one parameter -- the same log twice, or a "
                                 f"filename that does not say what changed? Nothing was loaded.")
            owner.setdefault(v, filename)

    ordered = sorted(owner, key=lambda v: tuple(sort_key(x) for x in v))
    vid = {v: i for i, v in enumerate(ordered)}
    frames, n_of = [], Counter()
    for filename, df, keys, rows, report in per_file:
        pos = [keys.index(k) for k in params]
        id_of = {r: vid[tuple(r[p] for p in pos)] for r in dict.fromkeys(rows)}
        ids = [id_of[r] for r in rows]
        n_of.update(ids)
        report["variant_ids"] = sorted(set(ids))
        f = df.drop(columns=report.get("columns") or []).copy()
        f[VARIANT_COLUMN] = ids
        frames.append(f)
    out = pd.concat(frames, ignore_index=True)
    out.attrs["parse_notes"] = _merge_notes([dict(df.attrs.get("parse_notes") or {}) for _, df in parsed])

    variants = [{"id": vid[v], "params": dict(zip(params, v)), "label": label(params, v),
                 "file": owner[v], "n": int(n_of[vid[v]])} for v in ordered]
    sweep = {
        "params": params,
        "values": {k: sorted({v[i] for v in ordered}, key=sort_key) for i, k in enumerate(params)},
        "variants": variants,
        "files": [r for *_, r in per_file],
        "suggested_name": _suggested_name(names, len(variants)),
    }
    return out, sweep
