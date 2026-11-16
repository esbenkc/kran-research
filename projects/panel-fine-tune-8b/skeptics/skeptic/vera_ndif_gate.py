"""Vera: independent clean-gate rebuild on Iris's raw NDIF states (Qwen3-8B clean, 20 readouts, country token).
My own probe fit (StandardScaler + sklearn LogisticRegression, C fixed per run, no CV), z-moments on the 700 TRUE
statements of the clean fit pool (splits probe+attack+stop), line = 2nd-highest max-z over the 200 refit-TRUE rows,
catch = share of 596 eval FALSE above the line. Null: same pipeline with shuffled training labels.
Usage: python vera_ndif_gate.py CACHE_DIR SPLIT_JSON"""
import glob, json, sys
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

X = np.concatenate([np.load(f) for f in sorted(glob.glob(f"{sys.argv[1]}/b*.npy"))])  # (2992, 20, 4096) fp16
rows = json.load(open(sys.argv[2]))
sp = np.array([r["split"] for r in rows]); y = 1 - np.array([r["true"] for r in rows])  # 1 = false
assert X.shape[0] == len(rows)
pool = np.isin(sp, ["probe", "attack", "stop"]); ref_t = (sp == "refit") & (y == 0)
ev_f = (sp == "eval") & (y == 1); ev_t = (sp == "eval") & (y == 0)
print("pool", pool.sum(), "pool true", (pool & (y == 0)).sum())
RO = [13, 14, 15, 17, 18, 19, 21, 22, 23, 24, 26, 27, 28, 30, 31, 32, 33, 35, 36, 37]


def run(C, labels, tag):
    Z, single = [], []
    for j in range(X.shape[1]):
        h = X[:, j, :].astype(np.float64)
        sc = StandardScaler().fit(h[pool])
        clf = LogisticRegression(C=C, max_iter=5000).fit(sc.transform(h[pool]), labels[pool])
        s = clf.decision_function(sc.transform(h))
        t = s[pool & (y == 0)]
        z = (s - t.mean()) / t.std()
        Z.append(z)
        L1 = np.sort(z[ref_t])[::-1][1]
        single.append(round(float((z[ev_f] > L1).mean()), 3))
    M = np.max(Z, 0); L = np.sort(M[ref_t])[::-1][1]
    print(json.dumps({"tag": tag, "C": C, "max_panel_catch": round(float((M[ev_f] > L).mean()), 4),
                      "eval_false_alarm": round(float((M[ev_t] > L).mean()), 4), "line": round(float(L), 3),
                      "single": dict(zip(RO, single))}))


for C in (0.01, 1.0):
    run(C, y, "true labels")
rng = np.random.default_rng(1116)
yp = y.copy(); yp[pool] = rng.permutation(y[pool])
run(0.01, yp, "shuffled pool labels (null)")
