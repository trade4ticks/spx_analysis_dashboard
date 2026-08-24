"""Parse-check every Alpine expression in the rendered templates.

check_alpine_refs.py answers "does this NAME exist on the component". It does
not answer "is this a valid JavaScript expression" -- an unbalanced paren or a
stray quote inside an attribute produces an Alpine expression error at runtime
that shows up as one silently dead binding, not a page failure. This catches
that class statically.

Attribute values are read through html.parser rather than a regex so the
entity decoding matches what the browser hands Alpine.
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


def main():
    env = Environment(loader=FileSystemLoader(TPL))
    env.globals["asset"] = lambda p: "/static/" + p
    env.globals["url_for"] = lambda *a, **k: "#"

    items, skipped = [], []
    for name in sorted(os.listdir(TPL)):
        if not name.endswith(".html") or name.startswith("_"):
            continue
        try:
            out = env.get_template(name).render(request=None)
        except Exception as exc:
            skipped.append((name, str(exc)))
            continue
        g = Grab()
        g.feed(out)
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
        for n, e in skipped:
            print(f"  SKIPPED {n}: {e[:120]}")
        return r.returncode


sys.exit(main())
