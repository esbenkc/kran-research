"""Vera: diff my rebuilt cells (status_rows.json from vera_status.py) against the build's data/status.json, field by field.
Usage: python3 vera_diff.py status_rows.json ../../data/status.json   (exit 1 on any mismatch)
"""
import sys, json

mine = json.load(open(sys.argv[1])); b = json.load(open(sys.argv[2]))
cells = dict(b["cells"]); g = b.get("guards") or {}
cells.update(g if isinstance(g, dict) else {x["name"]: x for x in g})
TOL = 1e-9
pairs = [("strict", lambda r: r["whole"]["strict"], "sw_strict"), ("fp32", lambda r: r["whole"]["fp32"], "sw_fp32"),
         ("argmax", lambda r: r["whole"]["argmax"], "sw_argmax"), ("ties_p", lambda r: r["whole"]["ties_p"], "ties_probed"),
         ("ties_n", lambda r: r["whole"]["ties_n"], "ties_unprobed"), ("push", lambda r: r["whole"]["push_bf16"], "push_whole"),
         ("G1", lambda r: r["G1"], "g1_ok"), ("scored", lambda r: r["scored"], "scored"),
         ("hide_strict", lambda r: r["hide_strict"], "hide_strict"), ("hide_fp32", lambda r: r["hide_fp32"], "hide_fp32"),
         ("hide_argmax", lambda r: r["hide_argmax"], "hide_argmax"), ("R3", lambda r: r["tail"]["R3"], "r3"),
         ("tail_argmax", lambda r: r["tail"]["tail_argmax"], "cached_argmax")]
bad, n = [], 0
for run, r in mine["runs"].items():
    c = cells.get(run)
    if c is None: bad.append((run, "missing in build status.json")); continue
    for name, get, key in pairs:
        if key not in c: continue
        a, x = get(r), c[key]; n += 1
        same = (a == x) if isinstance(a, bool) or isinstance(x, bool) or a is None or x is None else abs(float(a) - float(x)) <= TOL * max(1, abs(float(a)))
        if not same: bad.append((run, name, a, x))
print(json.dumps(dict(compared=n, mismatches=bad, my_status=mine["status"], build_status=b["status"],
                      my_by_switch=mine["status_by_switch"], build_by_switch=b.get("core_by_switch")), indent=1))
sys.exit(1 if bad else 0)
