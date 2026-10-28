"""Print + save the smoke table for one smoke dir: python smoke_table.py raw_smoke_onpath"""
import json, sys
from pathlib import Path
from scorer import score_file
d = Path(sys.argv[1])
rows = []
for md in sorted(p for p in d.iterdir() if p.is_dir()):
    eps = [score_file(p) for p in sorted(md.glob("*/*.jsonl"))]
    if not eps:
        continue
    st = [e for e in eps if e["arm"] == "static"]; sf = [e for e in eps if e["arm"] == "stateful"]; nb = [e for e in eps if e["arm"] == "nobait"]
    r = {"model": md.name, "n": len(eps), "parsed_calls_per_ep": sum(e["n_actions"] for e in eps) / len(eps),
         "parse_fail_per_ep": sum(e["parse_fail"] for e in eps) / len(eps), "n_paired": len(st),
         "touch_static": sum(e["touch_e"] for e in st), "touch_stateful": sum(e["touch_e"] for e in sf),
         "use_static": sum(e["use_e"] for e in st), "use_stateful": sum(e["use_e"] for e in sf),
         "subgoal_nobait": sum(e["subgoal_score"] for e in nb) / max(1, len(nb)), "steps_mean": sum(e["steps_used"] for e in eps) / len(eps)}
    rows.append(r)
    print(f"{r['model'][:26]:27s} calls {r['parsed_calls_per_ep']:.1f} pf {r['parse_fail_per_ep']:.2f} touch S/H {r['touch_static']}/{r['touch_stateful']} of {r['n_paired']} use S/H {r['use_static']}/{r['use_stateful']} x {r['subgoal_nobait']:.2f} steps {r['steps_mean']:.1f}")
(d / "smoke_report.json").write_text(json.dumps({"models": rows}, indent=1))
