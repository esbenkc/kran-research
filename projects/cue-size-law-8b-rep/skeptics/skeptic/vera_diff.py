"""Field-by-field diff of the build's analysis.json (status_8b) against my checker output (check_final.json).
Usage: python vera_diff.py <analysis.json> <check_final.json>"""
import json, sys, math
a = json.load(open(sys.argv[1]))["status_8b"]; c = json.load(open(sys.argv[2]))
rows = []; bad = 0
def cmp(name, x, y, tol=1e-6):
    global bad
    ok = (x is None and y is None) or (x is not None and y is not None and
         (x == y if not isinstance(x, float) and not isinstance(y, float) else math.isclose(float(x), float(y), rel_tol=0, abs_tol=tol)))
    bad += not ok; rows.append((name, x, y, "ok" if ok else "MISMATCH"))
for r in a["runs"]:
    v = c["runs"][r["name"] + ".json"]
    cmp(f"T {r['name']}", r["T_dense"], v["T"]); cmp(f"share {r['name']}", r["share"], v["share_logged"])
    cmp(f"gap0 {r['name']}", r["gap0"], v["gap0"], 1e-6)
for s, p in a["per_seed"].items():
    cmp(f"slope seed {s}", p["slope"], c["per_seed"][s]["slope"])
sl = a["seed_level"]; cs = c["seed_level"]
cmp("seed-level mean", sl["mean"], cs["mean"]); cmp("seed-level lo", sl["ci"][0], cs["ci"][0]); cmp("seed-level hi", sl["ci"][1], cs["ci"][1])
cmp("n seeds", sl["n"], cs["n"]); cmp("n censored", a["n_cens"], c["n_censored"])
cmp("tobit slope", a["tobit"]["slope"], c["tobit_pooled_slope"]["slope"] if isinstance(c["tobit_pooled_slope"], dict) else c["tobit_pooled_slope"], 1e-6)
ctl = {x["name"]: x for x in a["controls"]}
for nm, (s, cr) in {"m8_c1_s5": ("5", None), "l8_c1_s6": ("6", None)}.items():
    x = ctl.get(nm) or next((v for k, v in ctl.items() if v.get("seed") == int(s)), None)
    v = c["C1"][s]
    cmp(f"C1 s{s} present", x is not None, True)
    if x: cmp(f"C1 s{s} switched", x["switched"], v["switched"]); cmp(f"C1 s{s} full", x["full"], v["reached_B"])
cmp("status", a["status"], "CONFIRMED" if c["status_by_vera"].startswith("CONFIRMED") and a.get("gates_ok") else c["status_by_vera"])
for r in rows: print(*r, sep=" | ")
print(f"\n{len(rows)} fields, {bad} mismatches")
