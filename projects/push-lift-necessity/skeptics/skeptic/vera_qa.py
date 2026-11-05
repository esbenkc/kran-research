"""Vera (skeptic): paired bootstrap of ARC-Easy accuracy, P final minus its U twin, from build's per-item correctness.
PREREG damage guard / scout prereg clause 3: 95% CI of (P - U) inside +-3 points, 10,000 resamples, same items.
usage: python3 vera_qa.py -> vera_qa.json
"""
import json, os, random

PRJ = os.path.expanduser("~/work/oracle-1105-push-lift-necessity/kran-research/projects/push-lift-necessity")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vera_qa.json")
ld = lambda m: json.load(open(f"{PRJ}/data/qa/{m}.json"))
pairs = [("P_s1", "stack_s1"), ("P2_s0", "stack_s0"), ("P_s0_at5120", "stack_s0")]
res = {}
for p, u in pairs:
    a, b = ld(p), ld(u)
    assert a["ids"] == b["ids"] and len(a["ids"]) == 2946
    d = [x - y for x, y in zip(a["correct"], b["correct"])]
    n = len(d); rng = random.Random(0)
    bs = sorted(100 * sum(rng.choices(d, k=n)) / n for _ in range(10000))
    res[f"{p}-{u}"] = {"acc_P": sum(a["correct"]) / n, "acc_U": sum(b["correct"]) / n, "diff_pts": 100 * sum(d) / n,
                       "ci95": [bs[249], bs[9750]], "inside_pm3": bs[249] >= -3 and bs[9750] <= 3}
    print(p, u, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in res[f"{p}-{u}"].items()})
json.dump(res, open(OUT, "w"), indent=1)
