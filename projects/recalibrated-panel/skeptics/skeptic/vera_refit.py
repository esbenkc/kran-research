"""Vera: own refit ceiling. Per readout r in 10..29: standardise (train mean/sd), LogisticRegression C in {0.01,0.1,1,10}
on the attacked state's probe+stop (800) country-token states. Selection with no eval data: 5-fold GroupKFold by city
out-of-fold AUROC, best (readout, C); ties within 0.002 -> smaller C. Line = 2nd-highest of 200 line-set trues.
Also 3 shuffled-label nulls at the chosen (r, C).  Usage: python vera_refit.py <state>"""
import sys, json, numpy as np, vera26 as V
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
st = sys.argv[1]
X = np.load(V.D / f"ctok/{st}.npy", mmap_mode="r")
tr = np.where(V.TRAIN)[0]; y = V.YF[tr]; g = V.CITY[tr]
best = None; table = {}
for j, r in enumerate(V.R):
    A = np.asarray(X[:, j, :], np.float64); mu = A[tr].mean(0); sd = A[tr].std(0) + 1e-6; Z = (A - mu) / sd
    for C in (0.01, 0.1, 1, 10):
        oof = np.zeros(len(tr)); it = []
        for a, b in GroupKFold(5).split(tr, groups=g):
            m = LogisticRegression(C=C, max_iter=3000).fit(Z[tr[a]], y[a]); oof[b] = m.decision_function(Z[tr[b]]); it.append(int(m.n_iter_[0]))
        au = V.auroc(oof, y); table[f"{r}_{C}"] = [round(au, 5), max(it)]
        if best is None or au > best[0] + 0.002: best = (au, r, C)
au, r, C = best; j = V.R.index(r)
A = np.asarray(X[:, j, :], np.float64); mu = A[tr].mean(0); sd = A[tr].std(0) + 1e-6; Z = (A - mu) / sd
m = LogisticRegression(C=C, max_iter=3000).fit(Z[tr], y); s = m.decision_function(Z)
c = V.catch([V.flag(s)])
nulls = []
for k in range(3):
    ys = np.random.default_rng(k).permutation(y); s0 = LogisticRegression(C=C, max_iter=3000).fit(Z[tr], ys).decision_function(Z)
    nulls.append(V.catch([V.flag(s0)])["catch"])
out = dict(state=st, best_readout=r, C=C, oof_auroc=au, catch=c, conv_iters=int(m.n_iter_[0]), shuffled=nulls)
json.dump(dict(out, table=table), open(V.HERE / f"vera_refit_{st}.json", "w"), indent=1)
print(json.dumps(out))
