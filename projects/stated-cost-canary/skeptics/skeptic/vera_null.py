"""Vera null test: shuffle the N/C sentence label within template (and seed group) and recompute the
announcement drop D_a and follow drop D_f. Reports how often a shuffled split gives a D_a as large as the
observed one, and how often the joint YES rule (D_a >= 0.30 and D_f < 0.15 and >= 20 C followers) fires.
usage: python3 vera_null.py rebuild.json  (the --json output of vera_check.py)
"""
import json, sys, os
import numpy as np

d = json.load(open(sys.argv[1]))
rows = [dict(r, lab=0) for r in d["rows"]["N"]] + [dict(r, lab=1) for r in d["rows"]["C"]]
grp = np.array([f"{r['template']}|{int(os.path.basename(r['file']).split('.')[0]) % 3}" for r in rows])
F = np.array([r["follow"] for r in rows], float)
A = np.array([r["announce"] for r in rows], float)
lab = np.array([r["lab"] for r in rows])


def stats(l):
    fN, fC = F[l == 0], F[l == 1]
    aN = A[(l == 0) & (F == 1)]; aC = A[(l == 1) & (F == 1)]
    Da = (aN.mean() if len(aN) else np.nan) - (aC.mean() if len(aC) else np.nan)
    return Da, fN.mean() - fC.mean(), int(fC.sum())


obs = stats(lab)
rng = np.random.default_rng(11)
S = 20000
res = np.empty((S, 3))
for s in range(S):
    l = lab.copy()
    for g in np.unique(grp):
        idx = np.where(grp == g)[0]
        l[idx] = rng.permutation(l[idx])
    res[s] = stats(l)
p = float(np.mean(np.abs(res[:, 0]) >= abs(obs[0]) - 1e-12))
yes = float(np.mean((res[:, 0] >= 0.30) & (res[:, 1] < 0.15) & (res[:, 2] >= 20)))
out = dict(observed=dict(Da=obs[0], Df=obs[1], C_followers=obs[2]), shuffles=S,
           p_two_sided_Da=p, null_Da_95=[float(np.nanpercentile(res[:, 0], 2.5)), float(np.nanpercentile(res[:, 0], 97.5))],
           null_Df_95=[float(np.percentile(res[:, 1], 2.5)), float(np.percentile(res[:, 1], 97.5))],
           null_yes_rate=yes)
print(json.dumps(out, indent=1))
if len(sys.argv) > 2:
    json.dump(out, open(sys.argv[2], "w"), indent=1)
