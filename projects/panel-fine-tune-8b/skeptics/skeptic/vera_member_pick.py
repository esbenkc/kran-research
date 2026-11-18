"""Vera: pick the 'best single probe' per run WITHOUT eval data (own 1% line, highest catch on the 100 false
statements of split 'stop'), then report its eval catch. Checks that the figure's best-member mark is not inflated
by choosing the max of 20 members on eval. Usage: python3 vera_member_pick.py ../../data"""
import json, sys
import numpy as np
D = sys.argv[1]
rows = json.load(open(f"{D}/ref/split.json")); sp = np.array([r["split"] for r in rows]); t = np.array([r["true"] for r in rows])
rt = np.where((sp == "refit") & (t == 1))[0]; ef = np.where((sp == "eval") & (t == 0))[0]
et = np.where((sp == "eval") & (t == 1))[0]; sf = np.where((sp == "stop") & (t == 0))[0]
zm = json.load(open(f"{D}/fits/zmoments.json")); out = {}
for s in range(3):
    S = np.load(f"{D}/scores/panel__lam10000__seed{s}.npz"); res = []
    for k in zm:
        v = S[k]; L = np.sort(v[rt])[::-1][1]
        res.append((float((v[sf] > L).mean()), k, float((v[ef] > L).mean()), float((v[et] > L).mean())))
    res.sort(reverse=True)
    out[f"seed{s}"] = {"picked_on_stop": res[0][1], "stop_catch": res[0][0], "eval_catch": round(res[0][2], 4),
                       "eval_false_alarm": round(res[0][3], 4), "eval_best_of_20": round(max(r[2] for r in res), 4)}
print(json.dumps(out, indent=1)); json.dump(out, open("member_pick.json", "w"), indent=1)
