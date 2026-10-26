"""Vera: per-source split (cities = 'X is in Y', neg_cities = 'X is not in Y') of every rule's catch and false alarms."""
import sys, json, numpy as np, vera26 as V
src = np.array([r["src"] for r in V.ROWS])
def split(S):
    out = {}
    sc = {"ruleB": V.ruleB(S), "plain": V.plain(S), "logit": V.heads(S, "logit"), "qda": V.heads(S, "qda")}
    for k, s in sc.items():
        f = V.flag(s)
        out[k] = {x: [round(float(f[V.EVF][src[V.EVF] == x].mean()), 4), round(float(f[V.EVT][src[V.EVT] == x].mean()), 4)]
                  for x in ("cities", "neg_cities")}
    return out
res = {}
for a in sys.argv[1:]:
    S = V.scores_from_npz(a) if a.endswith(".npz") else V.scores_from_ctok(a)
    res[a.split("/")[-1]] = split(S)
    print(a.split("/")[-1], json.dumps(res[a.split("/")[-1]]))
json.dump(res, open(V.HERE / "vera_src.json", "a"), indent=1)
