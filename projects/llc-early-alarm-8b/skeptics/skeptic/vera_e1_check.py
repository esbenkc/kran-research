"""Vera: recompute E1 (budget sweep) from data/gate/budget.json chain means (no per-draw losses saved there):
lambda-hat = mean of the 2 chain values (chains are already n*beta*(mean draws - L*)); mean over reps; rel err vs Aoyagi."""
import json, collections, numpy as np
P = "/Users/esben/work/theory-1124-llc-early-alarm-8b/kran-research/projects/llc-early-alarm-8b/data/gate/"
B = json.load(open(P + "budget.json"))
true = {1: 236.0, 8: 312.0}
agg = collections.defaultdict(list); mism = 0
for r in B["rows"]:
    lam = float(np.mean(r["chains"]))
    mism += abs(lam - r["lambda_hat"]) > 1e-6 * max(1, abs(lam))
    agg[(r["c"], r["gamma"], r["steps"], r["r"])].append(lam)
print("chain-mean vs stored lambda_hat mismatches:", mism, "of", len(B["rows"]))
for (c, g, s, rk), v in sorted(agg.items()):
    print(f"c={c} g={g} steps={s:5d} r={rk}: mean of {len(v)} reps {np.mean(v):7.1f} rel {np.mean(v)/true[rk]-1:+.3f}  reps {[round(x/true[rk]-1,3) for x in v]}")
