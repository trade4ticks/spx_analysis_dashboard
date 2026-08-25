"""Prove the page-tenor retarget rule against the real column catalog.

The page has one tenor and every metric-name-bearing control follows it, which
means a rule for turning `skew_30d_25p_atm` into its 21d equivalent. That rule
is not obvious and two plausible versions of it are wrong:

  by (family, wing, tenor)   AMBIGUOUS. zc_width_sigma_21d and
                             zc_short_delta_21d are both (structure, short, 21);
                             vrp_1m and vrp_ratio_1m are both (vrp, atm, 30).
                             Ten such collisions exist.

  by blind token swap        WRONG WITHOUT A CHECK. Of the nine families that
                             carry a tenor only seven span all six: spot_vol
                             exists at 30 alone, vrp at 7/30/90 under 1w/1m/3m
                             labels that contain no {t}d token at all.

The shipped rule swaps the tenor token in the NAME and then verifies the
candidate against the catalog, keeping the original when there is no match.
This runs it against the real catalog -- built from app/metrics_config.py, the
same file the loader builds the tables from -- so a family added or moved
upstream is checked here rather than discovered on screen.

The retarget function itself is read out of static/js/equity_iv.js and run in
node, so this tests the shipped code rather than a Python restatement of it.
"""
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.metrics_config import BASE_COLUMNS, Z_COLUMNS  # noqa: E402

JS = ROOT / "static" / "js" / "equity_iv.js"

# (column, target tenor, expected result, why)
CASES = [
    ("skew_30d_25p_atm", 21, "skew_21d_25p_atm",
     "the ordinary case: a family that spans all six"),
    ("skew_30d_25p_atm_z_63", 21, "skew_21d_25p_atm_z_63",
     "the z form retargets and stays a z form"),
    ("iv_30d_atm", 7, "iv_7d_atm", "level_iv spans all six"),
    ("convex_30d_25p_atm_25c", 90, "convex_90d_25p_atm_25c", "convexity spans all six"),

    # The ambiguity that rules out keying on (family, wing, tenor): both of
    # these share a key with each other at every tenor.
    ("zc_width_sigma_30d", 21, "zc_width_sigma_21d",
     "structure/short is shared with zc_short_delta -- must not cross over"),
    ("zc_short_delta_30d", 21, "zc_short_delta_21d",
     "the other half of that collision"),

    # vrp used to be the standing example of a family that could NOT
    # retarget: it lived at 7/30/90 under 1w/1m/3m labels carrying no {t}d
    # token. RV_WINDOWS is now derived from TENORS upstream, so vrp_1m is
    # vrp_30d and the family spans the grid -- and the token-swap rule picked
    # it up with no change here, which is the property worth asserting.
    ("vrp_30d", 21, "vrp_21d", "vrp retargets now that the family spans the grid"),
    ("vrp_ratio_30d", 7, "vrp_ratio_7d",
     "and its sibling, which shares (vrp, atm, 30) with it -- the collision "
     "the stem-based rule exists to survive"),

    # spot_vol is the remaining family that cannot follow: built at 30d alone,
    # and named with a lookback label rather than a tenor token.
    ("spotvol_beta_1m", 21, "spotvol_beta_1m", "spot_vol is built at 30d only"),
    ("spotvol_beta_3m", 7, "spotvol_beta_3m", "the 3m window likewise"),

    # PAIR families. These carry tenor = null, so the token branch would
    # return them unchanged -- which is wrong in a way that is easy to defend
    # and still wrong: a term ratio IS a pair, but 30/90 read beside a 7-day
    # structure is quietly answering a different question than the rest of the
    # row. They map instead, and the map is asserted in full because it is a
    # judgement rather than a derivation.
    #
    # 7 -> 7/30 rather than 7/14: two very short fits sitting close together
    # are both noisy and the contango signal is clearer over a wider gap.
    # 21 -> 14/30 because there is no 21-day member of the pair set.
    ("term_ratio_30d_90d", 7,  "term_ratio_7d_30d",  "pair map: 7"),
    ("term_ratio_30d_90d", 14, "term_ratio_14d_30d", "pair map: 14"),
    ("term_ratio_30d_90d", 21, "term_ratio_14d_30d", "pair map: 21 shares 14/30"),
    ("term_ratio_7d_30d",  30, "term_ratio_30d_90d", "pair map: 30"),
    ("term_ratio_7d_30d",  60, "term_ratio_30d_90d", "pair map: 60"),
    ("term_ratio_7d_30d",  90, "term_ratio_30d_90d", "pair map: 90"),
    # term_slope takes the same pair map and keeps its delta suffix: the pair
    # answers "over what span", the delta is orthogonal to the page tenor.
    ("term_slope_30d_90d_25p", 7, "term_slope_7d_30d_25p",
     "the pair moves, the 25p wing does not"),
    ("term_slope_7d_14d_atm", 21, "term_slope_14d_30d_atm",
     "and the same for the atm slope"),

    # No tenor and no pair -- never retargeted.
    ("log_ret_1m", 21, "log_ret_1m", "a lookback window"),
    ("days_to_earnings", 21, "days_to_earnings", "calendar"),
    ("spot", 21, "spot", "a price level"),

    # Identity.
    ("skew_21d_25p_atm", 21, "skew_21d_25p_atm", "already at the page tenor"),

    # SYNTHETIC. Everything above passes with or without the catalog check,
    # because every {t}d-named family in today's catalog happens to span all
    # six tenors -- the two that do not span it (spot_vol, vrp) carry no token
    # to swap, so they never reach the check. That was found by deleting the
    # check and watching this gate stay green.
    #
    # So a probe family is injected: named with a token, present at 30 and 90,
    # absent at 21. Without the verification the swap would hand back
    # `probe_21d_atm`, a column that does not exist, and the panel would ask
    # for it and get a 400. This is the only case that fails when the check is
    # removed, and it exists so that removal cannot pass unnoticed.
    ("probe_30d_atm", 21, "probe_30d_atm",
     "SYNTHETIC: a token-named family with a gap at 21d must keep its column"),
    ("probe_30d_atm", 90, "probe_90d_atm",
     "SYNTHETIC: and must still retarget where the target DOES exist"),
]

# Injected into the catalog the harness sees, not into metrics_config.
PROBE = [
    ("probe_30d_atm", "probe", 30, "atm", "base"),
    ("probe_90d_atm", "probe", 90, "atm", "base"),
]

HARNESS = r"""
const fs = require('fs'), vm = require('vm');
let comp = null;
const sb = {
  console: {log(){},warn(){},error(){}},
  document: { addEventListener: (e,f) => { if (e === 'alpine:init') f(); },
              getElementById: () => null, querySelector: () => null,
              querySelectorAll: () => [], createElement: () => ({style:{}}),
              documentElement: { style: { setProperty(){} } } },
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
try { vm.runInContext(fs.readFileSync(process.argv[2],'utf8'), sb, {filename:'eq'}); }
catch (e) { console.log(JSON.stringify({ok:false, err:String(e)})); process.exit(0); }
if (!comp || typeof comp.retarget !== 'function') {
  console.log(JSON.stringify({ok:false, err:'no retarget() on the component'}));
  process.exit(0);
}
const payload = JSON.parse(fs.readFileSync(process.argv[3],'utf8'));
comp.byCol = payload.byCol;
comp.pageTenor = 30;
const out = payload.cases.map(c => {
  try { return comp.retarget(c[0], c[1]); }
  catch (e) { return '<<threw: ' + e.message + '>>'; }
});
console.log(JSON.stringify({ok:true, out}));
"""


def build_by_col():
    """The catalog as the client sees it, from the same config the loader uses."""
    by = {}
    for c in list(BASE_COLUMNS) + list(Z_COLUMNS):
        by[c.name] = {
            "column_name": c.name, "family": c.family, "tenor": c.tenor,
            "wing": c.wing, "form": c.form, "base_column": c.base,
            "units": c.units,
        }
    for name, fam, tenor, wing, form in PROBE:
        by[name] = {"column_name": name, "family": fam, "tenor": tenor,
                    "wing": wing, "form": form, "base_column": name,
                    "units": "ratio"}
    return by


def main() -> int:
    by_col = build_by_col()

    if any(name in {c.name for c in BASE_COLUMNS} for name, *_ in PROBE):
        print("  a PROBE name collides with a real column -- rename it")
        return 1

    missing = [c for c, _t, _w, _y in CASES if c not in by_col]
    if missing:
        # A case naming a column the catalog does not have would pass by
        # accident: retarget() returns the input unchanged for anything it
        # cannot resolve, which is what several cases EXPECT.
        print("  cases name columns absent from the catalog:", missing)
        print("  (they would pass for the wrong reason -- fix the case list)")
        return 1

    with tempfile.TemporaryDirectory() as d:
        hp = os.path.join(d, "h.js")
        pp = os.path.join(d, "p.json")
        open(hp, "w", encoding="utf-8").write(HARNESS)
        json.dump({"byCol": by_col, "cases": [[c, t] for c, t, _w, _y in CASES]},
                  open(pp, "w", encoding="utf-8"))
        r = subprocess.run(["node", hp, str(JS), pp],
                           capture_output=True, text=True, timeout=120)

    line = (r.stdout or "").strip().splitlines()
    if not line:
        print("  harness produced no output:", (r.stderr or "")[:400])
        return 1
    data = json.loads(line[-1])
    if not data.get("ok"):
        print("  harness could not run:", data.get("err"))
        return 1

    bad = 0
    for (col, tenor, want, why), got in zip(CASES, data["out"]):
        if got == want:
            continue
        bad += 1
        print(f"\n  {col} @ {tenor}d")
        print(f"    got  {got}")
        print(f"    want {want}   ({why})")

    print(f"\ncatalog columns: {len(by_col)}")
    print(f"retarget cases : {len(CASES)}, failed: {bad}")
    return 1 if bad else 0


sys.exit(main())
