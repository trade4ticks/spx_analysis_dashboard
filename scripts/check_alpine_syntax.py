"""Parse-check every Alpine expression in the rendered templates.

check_alpine_refs.py answers "does this NAME exist on the component". It does
not answer "is this a valid JavaScript expression" -- an unbalanced paren or a
stray quote inside an attribute produces an Alpine expression error at runtime
that shows up as one silently dead binding, not a page failure. This catches
that class statically.

Attribute values are read through html.parser rather than a regex so the
entity decoding matches what the browser hands Alpine.

A PAGE THAT DOES NOT RENDER IS A FAILURE, NOT A SKIP. This gate rendered with
request=None for weeks after _nav.html began reading request.url.hostname:
every page raised, every page was listed as SKIPPED, zero expressions were
checked, and the gate reported PASS. It now renders with the same stub
request the other render gates use, fails on any render error, and fails if
any page with an Alpine component contributes no expressions -- a check that
checked nothing has not passed.
"""
import html.parser, io, json, os, subprocess, sys, tempfile

from jinja2 import Environment, FileSystemLoader

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TPL  = os.path.join(ROOT, "templates")

DIRECTIVES = ("x-text", "x-html", "x-show", "x-if", "x-model", "x-init",
              "x-data", "x-effect")


class Grab(html.parser.HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.exprs = []      # (attr, value, kind)

    def handle_starttag(self, tag, attrs):
        for k, v in attrs:
            if not v:
                continue
            if k in ("x-for",):
                # "item in list" / "(item, i) in list" -- only the tail is JS
                tail = v.split(" in ", 1)
                if len(tail) == 2:
                    self.exprs.append((k, tail[1], "expr"))
            elif k.startswith("@") or k.startswith("x-on:"):
                self.exprs.append((k, v, "stmt"))
            elif k.startswith(":") or k.startswith("x-bind:"):
                self.exprs.append((k, v, "expr"))
            elif k in DIRECTIVES:
                self.exprs.append((k, v, "stmt" if k in ("x-init", "x-effect") else "expr"))


class _StubURL:
    hostname = "localhost"
    scheme = "http"
    path = "/"

    def __str__(self): return "http://localhost/"


class _StubRequest:
    url = _StubURL()
    headers: dict = {}
    query_params: dict = {}
    scope = {"type": "http"}


def main():
    env = Environment(loader=FileSystemLoader(TPL))
    env.globals["asset"] = lambda p: "/static/" + p
    env.globals["url_for"] = lambda *a, **k: "#"
    env.globals["live_port"] = 8001

    items, failed, empty = [], [], []
    for name in sorted(os.listdir(TPL)):
        if not name.endswith(".html") or name.startswith("_"):
            continue
        try:
            out = env.get_template(name).render(request=_StubRequest())
        except Exception as exc:
            failed.append((name, f"{type(exc).__name__}: {exc}"))
            continue
        g = Grab()
        g.feed(out)
        if "x-data" in out and not g.exprs:
            empty.append(name)
        for attr, val, kind in g.exprs:
            items.append({"file": name, "attr": attr, "kind": kind, "src": val})

    # x-data on the body is a component NAME, not an expression -- skip those.
    items = [i for i in items if not (i["attr"] == "x-data" and i["src"].isidentifier())]

    probe = """
const items = JSON.parse(require('fs').readFileSync(process.argv[2], 'utf8'));
let bad = 0;
for (const it of items) {
  const wrapped = it.kind === 'stmt'
    ? `(function(){ ${it.src} })`
    : `(function(){ return (${it.src}) })`;
  try { new Function(wrapped); }
  catch (e) {
    bad++;
    console.log(`  ${it.file}  ${it.attr}="${it.src.slice(0,90)}"`);
    console.log(`      ${e.message}`);
  }
}
console.log(`\\nexpressions checked: ${items.length}, syntax errors: ${bad}`);
process.exit(bad ? 1 : 0);
"""
    with tempfile.TemporaryDirectory() as d:
        jf = os.path.join(d, "items.json")
        pf = os.path.join(d, "probe.js")
        io.open(jf, "w", encoding="utf-8").write(json.dumps(items))
        io.open(pf, "w", encoding="utf-8").write(probe)
        r = subprocess.run(["node", pf, jf], capture_output=True, text=True)
        print(r.stdout or r.stderr)
    for n, e in failed:
        print(f"  FAIL  {n} did not render: {e[:160]}")
    for n in empty:
        print(f"  FAIL  {n} has an Alpine component but no expressions were found in it")
    if not items:
        print("  FAIL  no expressions were checked at all")
        return 1
    return 1 if (r.returncode or failed or empty) else 0


sys.exit(main())
