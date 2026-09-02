#!/usr/bin/env python3
"""Gate: does every function an Alpine attribute calls actually exist?

Why this exists
---------------
Alpine resolves expressions against the component object at RUNTIME, in the
browser, silently until something is clicked. A handler that names a method
which does not exist parses fine, passes every syntax check, renders without
complaint, and then does nothing at all when used. The button's own state
often still updates -- because that part is a plain assignment -- so the
control looks alive while the work behind it never runs.

That has now happened twice on this project:

  @input="setGridSpan(...)"      the method was on the component but the
                                 browser was running a stale bundle
  @click="setPortWindow(...)"    which called this.loadPortAggregate(),
                                 when the method is loadPortfolioAggregate

Neither was catchable by node --check, Jinja parsing, or reading the diff.
Both were one grep away from being caught before shipping.

What it does
------------
For each page template, finds the JS bundle it loads, executes that bundle
under a stub Alpine to obtain the real component object, then extracts every
`name(` call appearing inside an Alpine attribute (@click, x-on:, x-show,
x-text, x-if, :class, ...) and asserts the name resolves to a member of the
component -- or to a JS builtin, a local loop variable, or a window global.

Requires node on PATH. Skips (does not fail) a template whose bundle cannot
be constructed, so a page with an exotic bootstrap does not block the gate.

Usage
-----
    python scripts/check_alpine_refs.py
    python scripts/check_alpine_refs.py --template oi_analysis.html

Exit 0 = every call resolves. Exit 1 = at least one does not.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TEMPLATES = ROOT / "templates"
STATIC = ROOT / "static"

# Attributes whose value Alpine evaluates as an expression.
ATTR = re.compile(r"""\s(?:@[\w.\-]+|x-(?:on:[\w.\-]+|show|if|text|html|model|
                       init|effect|bind:[\w.\-]+)|:[\w.\-]+)\s*=\s*"([^"]*)\"""",
                  re.X)
# A BARE call: not preceded by "." (a method on some object) and not by
# "$" (an Alpine magic). String literals are stripped before this runs --
# a :style binding is full of rgba(...) and var(...), and tooltip prose is
# full of "(" following ordinary words.
CALL = re.compile(r"(?<![\w.$])([A-Za-z_][\w$]*)\s*\(")
STRLIT = re.compile(r"'[^']*'|\"[^\"]*\"|`[^`]*`")

# Names that legitimately appear as calls but are not component members.
BUILTINS = {
    "Math", "Number", "String", "Boolean", "Array", "Object", "JSON", "Date",
    "parseInt", "parseFloat", "isNaN", "isFinite", "encodeURIComponent",
    "decodeURIComponent", "Set", "Map", "RegExp", "Promise", "console",
    "setTimeout", "clearTimeout", "requestAnimationFrame", "alert", "confirm",
    "window", "document", "fetch", "structuredClone", "queueMicrotask",
    # Alpine magics
    "$refs", "$el", "$event", "$dispatch", "$nextTick", "$watch", "$store",
    "$data", "$id", "$root",
    # common inline lambdas / array methods used on loop vars
    "map", "filter", "find", "reduce", "join", "includes", "some", "every",
    "toFixed", "toLocaleString", "toUpperCase", "toLowerCase", "slice",
    "split", "sort", "indexOf", "startsWith", "endsWith", "replace", "trim",
    "push", "concat", "keys", "values", "entries", "has", "get", "at", "flat",
    "padStart", "padEnd", "repeat", "abs", "min", "max", "round", "floor",
    # JS keywords that are followed by "(" but are not calls
    "if", "for", "while", "switch", "catch", "in", "of", "return",
    "typeof", "instanceof", "new", "delete", "void", "function",
    "do", "else", "await", "yield", "with",
}

NODE_HARNESS = r"""
const fs = require('fs'), vm = require('vm');
let comp = null;
const sb = {
  console: {log(){},warn(){},error(){}},
  document: { addEventListener: (e,f) => { if (e === 'alpine:init') f(); },
              getElementById: () => null, querySelector: () => null,
              querySelectorAll: () => [], createElement: () => ({style:{}}) },
  Alpine: { data: (n,f) => { if (!comp) { try { comp = f(); } catch(e){} } },
            store: () => ({}), magic: () => {}, directive: () => {} },
  fetch: () => Promise.resolve({ ok:false, json: async () => ({}) }),
  localStorage: { getItem: () => null, setItem: () => {}, removeItem: () => {} },
  Chart: function(){ return {destroy(){}, update(){}}; },
  setTimeout, clearTimeout, setInterval, clearInterval,
  requestAnimationFrame: () => 0,
};
sb.window = sb; sb.globalThis = sb; sb.self = sb;
vm.createContext(sb);
for (const f of process.argv.slice(3)) {
  try { vm.runInContext(fs.readFileSync(f,'utf8'), sb, {filename:f}); }
  catch (e) { /* a bundle may need DOM we do not have; keep going */ }
}
if (!comp) { console.log(JSON.stringify({ok:false})); process.exit(0); }

/* SECONDARY SCOPES.
 *
 * A page may split its state across more than one object — Equities Live has
 * a component that owns the socket and the frame loop, and a pane factory
 * that owns one plot. Both are Alpine scopes: the template binds directly to
 * pane members, and pane methods call each other with `this`.
 *
 * A factory opts in by tagging itself, and ONLY tagged factories are called.
 * Calling every function on window to see what it returns would run arbitrary
 * page code inside the checker, which is a large amount of risk to take on
 * for a naming convention. */
const scopes = [comp];
for (const k of Object.getOwnPropertyNames(sb)) {
  let f;
  try { f = sb[k]; } catch (e) { continue; }
  if (typeof f !== 'function' || !f.isComponentScope) continue;
  try {
    const o = f();
    if (o && typeof o === 'object') scopes.push(o);
  } catch (e) { /* a factory that needs arguments is not a scope */ }
}

const names = new Set(), fns = new Set();
for (const scope of scopes) {
let o = scope;
while (o && o !== Object.prototype) {
  for (const k of Object.getOwnPropertyNames(o)) {
    names.add(k);
    // getOwnPropertyDescriptor, not comp[k]: reading a getter would RUN it,
    // and these components have getters that touch state this harness has
    // not set up. A getter is recorded as callable-unknown (left out of
    // `fns` only if it is plainly a data property), because a getter may
    // legitimately return a function.
    const d = Object.getOwnPropertyDescriptor(o, k);
    if (!d) continue;
    if (typeof d.value === 'function' || d.get) fns.add(k);
  }
  o = Object.getPrototypeOf(o);
}
}
for (const k of Object.keys(sb)) { names.add(k); fns.add(k); }
console.log(JSON.stringify({ok:true, names:[...names], fns:[...fns]}));
"""


def component_names(js_files: list) -> set | None:
    with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False,
                                     encoding="utf-8") as fh:
        fh.write(NODE_HARNESS)
        harness = fh.name
    try:
        out = subprocess.run(["node", harness, "--"] + [str(f) for f in js_files],
                             capture_output=True, text=True, timeout=120)
        line = (out.stdout or "").strip().splitlines()
        if not line:
            return None
        data = json.loads(line[-1])
        if not data.get("ok"):
            return None
        return set(data["names"]), set(data.get("fns") or [])
    except Exception:
        return None
    finally:
        Path(harness).unlink(missing_ok=True)


# Keys that are legitimately repeated because they belong to DIFFERENT nested
# objects, not to the component itself. Matching is by name only, so a few
# generic ones have to be exempted or every file trips.
_DUP_OK = {
    "label", "data", "type", "options", "min", "max", "color", "value",
    "name", "key", "n", "title", "text", "display", "grid", "ticks",
    "borderColor", "backgroundColor", "borderWidth", "pointRadius", "fill",
    "tension", "stepped", "yAxisID", "position", "callback", "callbacks",
    "plugins", "scales", "legend", "tooltip", "datasets", "responsive",
    "maintainAspectRatio", "animation", "parsing", "mode", "intersect",
}


def duplicate_component_keys(js_path: Path) -> list:
    """Keys defined TWICE at the top level of an Alpine.data() object literal.

    Why this exists
    ---------------
    A duplicate key in an object literal is legal JavaScript: the later
    definition silently wins. Nothing errors, nothing warns, and every name
    still resolves -- so the reference check above passes cleanly while a
    method or getter has been replaced by another one further down the file.

    That has landed twice. `cutoffLineDate` was declared at the top of
    oi_analysis.js for the page-level charts and again 8,000 lines later for
    the portfolio; the later one returned '' unless the portfolio was
    rendering, so the cutoff line on the main page's TT charts went dead. A
    `seriesAxis` field was likewise replaced by a getter with no setter.

    Depth is tracked by brace/bracket/paren balance from the object's opening
    `({`, and only depth-1 keys are considered -- a `label:` inside a nested
    Chart.js config is not a component member.
    """
    src = js_path.read_text(encoding="utf-8")
    m = re.search(r"Alpine\.data\(\s*['\"][\w$]+['\"]\s*,\s*\(\s*\)\s*=>\s*\(\{", src)
    if not m:
        return []
    i = m.end()
    depth = 1
    seen: dict = {}
    dups: list = []
    # Capture the accessor kind: a get/set PAIR for one name is a single
    # property, not a redefinition, and must not be reported.
    key_re = re.compile(r"(get|set|async)?\s*([A-Za-z_$][\w$]*)\s*[:(]")
    line_start = True
    while i < len(src) and depth > 0:
        ch = src[i]
        if ch in "{[(":
            depth += 1
        elif ch in "}])":
            depth -= 1
        elif ch == "\n":
            line_start = True
            i += 1
            continue
        elif depth == 1 and line_start and not ch.isspace():
            # Skip comments outright.
            if src.startswith("//", i):
                i = src.find("\n", i)
                if i == -1:
                    break
                continue
            if src.startswith("/*", i):
                i = src.find("*/", i) + 2
                continue
            km = key_re.match(src, i)
            if km:
                kind = km.group(1) or "plain"
                name = km.group(2)
                ln = src[:km.start()].count("\n") + 1
                prev = seen.get(name)
                if prev and name not in _DUP_OK:
                    prev_ln, prev_kind = prev
                    accessor_pair = {kind, prev_kind} == {"get", "set"}
                    if not accessor_pair:
                        dups.append(
                            f"{js_path.name}: `{name}` defined at line "
                            f"{prev_ln} AND line {ln} — the later one "
                            f"silently wins")
                elif not prev:
                    seen[name] = (ln, kind)
            line_start = False
        elif not ch.isspace():
            line_start = False
        i += 1
    return dups


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--template", help="Check only this template")
    args = ap.parse_args()

    problems: list = []
    checked = skipped = calls_seen = 0

    for tpl in sorted(TEMPLATES.glob("*.html")):
        if tpl.name.startswith("_"):
            continue
        if args.template and tpl.name != args.template:
            continue
        text = tpl.read_text(encoding="utf-8")
        js = [STATIC / m for m in re.findall(r"asset\(['\"](js/[^'\"]+)['\"]\)", text)]
        js = [f for f in js if f.is_file()]
        if not js:
            continue
        got = component_names(js)
        if got is None:
            print(f"  SKIP {tpl.name}: component could not be constructed")
            skipped += 1
            continue
        names, fns = got
        checked += 1
        # Strip HTML comments and Jinja comments -- prose is not evaluated.
        body = re.sub(r"<!--.*?-->", "", text, flags=re.S)
        body = re.sub(r"\{#.*?#\}", "", body, flags=re.S)
        seen: set = set()
        for expr in ATTR.findall(body):
            # String literals are data, not code.
            expr = STRLIT.sub("''", expr)
            for name in CALL.findall(expr):
                calls_seen += 1
                if name in BUILTINS or name in seen:
                    continue
                if name in names:
                    # On the component, but is it CALLABLE? A string field
                    # invoked as `oiNote()` resolves fine and throws at
                    # runtime, which the membership check above cannot see --
                    # the binding simply goes dead on the page.
                    if name not in fns:
                        seen.add(name)
                        problems.append(
                            f"{tpl.name}: {name}() is called from an Alpine "
                            f"attribute but {name} is a data field, not a "
                            f"method -- the binding will throw")
                    continue
                # loop variables and inline consts are lower-cased short words;
                # only report names that look like methods and are absent.
                seen.add(name)
                problems.append(f"{tpl.name}: {name}() is called from an Alpine "
                                f"attribute but is not on the component")

    # -- Second pass: this.X() inside the component's own methods --------
    # The attribute scan above only sees the OUTERMOST call. A handler that
    # exists but calls a method that does not -- setPortWindow() calling
    # this.loadPortAggregate() when the real name is loadPortfolioAggregate
    # -- passes that scan and still does nothing when clicked. This is the
    # layer that bug lived on.
    this_calls = 0
    for js_file in sorted(STATIC.glob("js/*.js")):
        got = component_names([js_file])
        if got is None:
            continue
        names, fns = got
        src = js_file.read_text(encoding="utf-8")
        src = re.sub(r"/\*.*?\*/", "", src, flags=re.S)
        src = re.sub(r"(?m)^\s*//.*$", "", src)
        seen_here = set()
        for m in re.finditer(r"\bthis\.([A-Za-z_$][\w$]*)\s*\(", src):
            name = m.group(1)
            this_calls += 1
            if name in names or name in BUILTINS or name in seen_here:
                continue
            seen_here.add(name)
            line = src[:m.start()].count("\n") + 1
            problems.append(
                f"{js_file.name}:~{line}: this.{name}() is called but is not "
                f"on the component")

    # ── Third pass: duplicate keys in the component object literal ──────
    # Both names resolve, so neither pass above can see this one.
    dup_files = 0
    for js_file in sorted(STATIC.glob("js/*.js")):
        d = duplicate_component_keys(js_file)
        if d:
            dup_files += 1
            problems.extend(d)

    print(f"\nthis.X() calls    : {this_calls}")
    print(f"files w/ dup keys : {dup_files}")
    print(f"\ntemplates checked : {checked}")
    print(f"templates skipped : {skipped}")
    print(f"calls inspected   : {calls_seen}")
    if problems:
        print(f"\nUNRESOLVED ({len(problems)}):\n")
        for p in problems:
            print("  " + p)
        print("\nThese fail silently in the browser: the expression throws, the "
              "\nhandler does nothing, and any state set before the call still "
              "\nupdates -- so the control looks alive.")
        return 1
    print("\nOK — every Alpine call resolves to a component member.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
