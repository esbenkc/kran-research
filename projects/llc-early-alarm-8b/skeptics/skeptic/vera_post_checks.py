"""Vera: post-number checks. (1) closest gate setting (c 0.3, gamma 10): worst-case miss per repeat, mean and
t-interval over the 5 repeats. (2) this run's m8_k2_s5 evals vs the parent's m8_k2_s5 (bit for bit?)."""
import json, math, numpy as np
from scipy import stats
R = "/Users/esben/work/theory-1124-llc-early-alarm-8b/kran-research/projects/llc-early-alarm-8b/"
G = json.load(open(R + "data/gate/grid.json")); nb = G["n"] / math.log(G["n"]); true = {1: 236.0, 8: 312.0}
lam = {}
for r in G["rows"]:
    d = np.array(r["losses"], float)[:, G["burn"]:].reshape(-1); lam[(r["c"], r["gamma"], r["rep"], r["r"])] = nb * (d.mean() - r["L_wstar"])
w = [max(abs(lam[(0.3, 10.0, rp, k)] / true[k] - 1) for k in (1, 8)) for rp in range(5)]
m, s = np.mean(w), np.std(w, ddof=1); h = stats.t.ppf(0.975, 4) * s / math.sqrt(5)
print("worst-case miss per repeat:", [f"{x:.4f}" for x in w], f"mean {m:.4f}, 95% t [{m-h:.4f}, {m+h:.4f}]")
P = "/Users/esben/work/theory-1122-cue-size-law-8b/kran-research/projects/cue-size-law-8b-rep/data/runs/m8_k2_s5.json"
a = json.load(open(R + "data/runs/m8_k2_s5.json")); b = json.load(open(P))
def evals(d):
    out = {}
    for e in d["dense_curve"] + d["dev_curve"]:
        out[e["step"]] = (e["p_off"], e["p_on"])
    return out
ea, eb = evals(a), evals(b)
common = sorted(set(ea) & set(eb)); exact = 0; maxdiff = 0.0
for t in common:
    da = np.array(ea[t][0] + ea[t][1]); db = np.array(eb[t][0] + eb[t][1])
    exact += np.array_equal(da, db); maxdiff = max(maxdiff, float(np.abs(da - db).max()))
print(f"evals in common {len(common)} (steps {common}); bit-identical {exact}; max |diff| {maxdiff:.3g}; T_dense this {a['T_dense']} parent {b['T_dense']}")
