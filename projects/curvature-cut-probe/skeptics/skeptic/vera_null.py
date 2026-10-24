"""Vera null tests + consistency on a finished run (numpy/sklearn, < 0.15 GB).

1. Consistency: score the saved post-norm states acts_<step>.npy (readout 29) with MY copy of each probe's formula and
   compare with the ckpt scores (proves the ckpt 'scores' are the probe on these states).
2. Shuffled-label null: 5 probes fit on the clean 'probe'-split states with labels shuffled within city (LR, C=0.1,
   StandardScaler), line = 2nd-highest of the 200 line-setting true at each state; catch on the 596 eval false.
   Must sit near 1% on clean AND attacked states: the attack cannot 'hide' from a probe that never saw the signal.
3. Transfer: catch of every other probe column at the re-set line (who else this fine-tune blinds).
Usage: vera_null.py RUN_DIR STEP
"""
import sys, json, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
run, step = sys.argv[1], int(sys.argv[2])
S = json.load(open("data/ref/split.json")); sp = np.array([e["split"] for e in S]); tr = np.array([e["true"] == 1 for e in S])
city = np.array([e["city"] for e in S]); EVF = (sp == "eval") & ~tr; LS = (sp == "refit") & tr
P = np.load("data/fits/probes.npz"); d = np.load(f"{run}/ckpt_{step}.npz", allow_pickle=True); names = [str(x) for x in d["probes"]]
A = np.load(f"{run}/acts_{step}.npy", mmap_mode="r")
ro = list(d["acts_readouts"]) if "acts_readouts" in d.files else None
print("acts", A.shape, "readouts", ro)
Zs = np.asarray(A[:, ro.index(29)] if (A.ndim == 3 and ro is not None) else A, dtype=np.float64)
Z0 = np.load("data/clean/z_post.npy").astype(np.float64)
out = dict(run=run, step=step)
for nm in ("unproj", "proj", "rand"):
    s_me = (Zs - P[nm + "__mu"]) / P[nm + "__sd"] @ P[nm + "__w"].astype(np.float64) + float(P[nm + "__b"])
    s_ck = d["scores"][:, names.index(nm)].astype(np.float64); ok = ~np.isnan(s_ck)
    out[f"consistency_{nm}"] = float(np.abs(s_me[ok] - s_ck[ok]).max())
rng = np.random.default_rng(0); fit = np.where(sp == "probe")[0]
nulls = []
for k in range(5):
    y = tr[fit].astype(int).copy()
    for c in np.unique(city[fit]):          # shuffle within city: keeps the city mix, kills truth
        m = np.where(city[fit] == c)[0]; y[m] = rng.permutation(y[m])
    sc = StandardScaler().fit(Z0[fit]); lr = LogisticRegression(C=0.1, max_iter=2000).fit(sc.transform(Z0[fit]), y)
    row = {}
    for tag, ZZ in (("clean", Z0), ("attacked", Zs)):
        s = lr.decision_function(sc.transform(ZZ)); line = np.sort(s[LS])[::-1][1]
        row[tag] = float((s[EVF] > line).mean())
    nulls.append(row)
out["shuffled_null_catch"] = nulls
s_all = d["scores"].astype(np.float64)
out["transfer_catch_reset_line"] = {nm: float((s_all[EVF, j] > np.sort(s_all[LS, j])[::-1][1]).mean()) for j, nm in enumerate(names)}
print(json.dumps(out, indent=1))
json.dump(out, open(f"skeptic/vera_null_{run.rstrip('/').split('/')[-1]}_{step}.json", "w"), indent=1)
