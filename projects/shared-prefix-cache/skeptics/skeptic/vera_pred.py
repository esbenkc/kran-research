"""Vera's own pre-RL predictors on block-26 inputs (own code). Reads vera_shared.py output (h26 as bf16 bits).

For a pair of builds (A = grading reference, B = other):
  constant share per position p (200 test prompts): ||mean_i d_ip||^2 / mean_i ||d_ip||^2, d = h_B - h_A (fp32).
  readout AUROC per position: logistic regression (standardised, C=1) on train+dev prompts (both rows of each prompt),
  AUROC on the test prompts. Null: within-pair label flips (which row is 'B' chosen at random per prompt), NNULL
  repeats, max over positions in each group -> 95th percentile.
Groups (PREREG s2): 0-2, 3-5, 6-11, 12-24, 25-33.
Usage: python vera_pred.py IN.npz A B OUT.json [NNULL]
"""
import sys, json
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

inp, A, B, outp = sys.argv[1:5]
NNULL = int(sys.argv[5]) if len(sys.argv) > 5 else 100
z = np.load(inp)
def bf(a): return (a.astype(np.uint16).astype(np.uint32) << 16).view(np.float32)
hA, hB = bf(z[A + "_h26"]), bf(z[B + "_h26"])
split = z["split"]; te = split == "test"; tr = ~te
d = (hB - hA)[te].astype(np.float64)
T = hA.shape[1]
if T == 34:
    G = {"0-2": range(0, 3), "3-5": range(3, 6), "6-11": range(6, 12), "12-24": range(12, 25), "25-33": range(25, 34)}
else:  # seam arm: 3 tokens inserted after the prefix (positions 3-5), the rest shifted by 3
    G = {"0-2": range(0, 3), "3-5": range(3, 6), "6-8": range(6, 9), "9-14": range(9, 15), "15-27": range(15, 28), "28-36": range(28, 37)}
cs = [(np.linalg.norm(d[:, p].mean(0)) ** 2 / (np.linalg.norm(d[:, p], axis=1) ** 2).mean()) if np.abs(d[:, p]).sum() > 0 else float("nan") for p in range(T)]
frac = [(hA[:, p] != hB[:, p]).mean() for p in range(T)]

def fit_auc(p, flip_tr=None, flip_te=None):
    Xa, Xb = hA[:, p], hB[:, p]
    if np.array_equal(Xa, Xb): return 0.5
    def stack(mask, flip):
        a, b = Xa[mask], Xb[mask]
        if flip is not None:
            a, b = np.where(flip[:, None], b, a), np.where(flip[:, None], a, b)
        return np.concatenate([a, b]), np.r_[np.zeros(len(a)), np.ones(len(b))]
    Xtr, ytr = stack(tr, flip_tr); Xte, yte = stack(te, flip_te)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    clf = LogisticRegression(C=1.0, max_iter=2000).fit((Xtr - mu) / sd, ytr)
    return roc_auc_score(yte, clf.decision_function((Xte - mu) / sd))

auc = [fit_auc(p) for p in range(T)]
print("pos  frac_diff  const_share  auroc")
for p in range(T): print(f"{p:3d}  {frac[p]:.3f}  {cs[p]:.4f}  {auc[p]:.3f}", flush=True)
rng = np.random.default_rng(11)
null = {g: [] for g in G}
pos_live = [p for p in range(3, T)]
for k in range(NNULL):
    ft, fe = rng.random(tr.sum()) < 0.5, rng.random(te.sum()) < 0.5
    a = {p: fit_auc(p, ft, fe) for p in pos_live}
    for g, r in G.items():
        if g != "0-2": null[g].append(max(a[p] for p in r))
    if k % 10 == 0: print("null", k, {g: round(v[-1], 3) for g, v in null.items() if v}, flush=True)
out = {"pair": [A, B], "n_test": int(te.sum()), "n_train": int(tr.sum()),
       "per_pos": [{"pos": p, "frac_diff": float(frac[p]), "const_share": float(cs[p]), "auroc": float(auc[p])} for p in range(T)],
       "groups": {g: {"max_const_share": float(np.nanmax([cs[p] for p in r])), "max_auroc": float(max(auc[p] for p in r)),
                      "argmax_auroc": int(max(r, key=lambda p: auc[p])),
                      "null95_max_auroc": float(np.percentile(null[g], 95)) if null[g] else None} for g, r in G.items()},
       "max_auroc_3_33": float(max(auc[3:])), "nnull": NNULL}
json.dump(out, open(outp, "w"), indent=1)
for g, v in out["groups"].items(): print(g, v, flush=True)
print("max AUROC over 3-33:", round(out["max_auroc_3_33"], 4))
