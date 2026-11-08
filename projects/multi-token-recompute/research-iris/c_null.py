"""Null for C (Iris): is C_start = -0.015 noise? Two nulls on the same npz.
(1) within-pair swap on TEST (fixed classifier): each pair's (none, leaky) labels swapped with prob 1/2;
    an identical pair contributes 0 either way, a discordant pair flips the sign of its contribution.
(2) refit null: swap labels within each TRAIN pair at random, refit with the frozen CV, score the real test labels.
Also C at each fixed regularisation (frozen primary = CV choice)."""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from c_reader import featurize, fit_c, REGS
from sklearn.linear_model import LogisticRegression

npz, out = sys.argv[1], sys.argv[2]
n_refit = int(sys.argv[3]) if len(sys.argv) > 3 else 20
z = np.load(npz)
tn, tl, pr = z["toks_n"].astype(np.int64), z["toks_l"].astype(np.int64), z["prompt_idx"].astype(np.int64)
is_train = z["split"] == "train"
V = int(max(tn.max(), tl.max())) + 1
Xn, Xl = featurize(tn, V), featurize(tl, V)
te = ~is_train
res = {"npz": npz}
# fixed-reg C and per-pair contributions
from scipy import sparse
Xtr = sparse.vstack([Xn[is_train], Xl[is_train]]); ytr = np.r_[np.zeros(is_train.sum()), np.ones(is_train.sum())]
fixed = {}
for r in REGS:
    m = LogisticRegression(C=r, solver="liblinear", max_iter=2000).fit(Xtr, ytr)
    pn, pl = m.predict(Xn[te]), m.predict(Xl[te])
    # balanced acc = 0.5*(mean(pn==0) + mean(pl==1)); C = that*2 - 1 = mean(pl) - mean(pn)
    contrib = pl - pn                                   # per pair in {-1, 0, 1}; C = mean(contrib)
    fixed[r] = float(contrib.mean())
    if r == 10.0:
        contrib10 = contrib
res["C_fixed_reg"] = fixed
rng = np.random.default_rng(0)
sw = rng.choice([-1, 1], size=(20000, len(contrib10)))
null1 = (sw * contrib10).mean(1)
obs = contrib10.mean()
res["within_pair_swap"] = dict(obs=float(obs), n_nonzero=int((contrib10 != 0).sum()), n_pairs=int(len(contrib10)),
                               p_two_sided=float(np.mean(np.abs(null1) >= abs(obs) - 1e-12)), null_sd=float(null1.std()))
# refit null
cs = []
for i in range(n_refit):
    s = rng.random(is_train.sum()) < 0.5
    a, b = tn[is_train].copy(), tl[is_train].copy()
    a[s], b[s] = tl[is_train][s], tn[is_train][s]
    TN, TL = tn.copy(), tl.copy()
    TN[is_train], TL[is_train] = a, b
    cs.append(fit_c(TN, TL, pr, is_train, n_boot=1)["C"])
cs = np.array(cs)
res["refit_null"] = dict(n=n_refit, C=cs.tolist(), sd=float(cs.std()), share_le_obs=float(np.mean(cs <= -0.015 + 1e-12)))
json.dump(res, open(out, "w"), indent=1)
print(json.dumps({k: v for k, v in res.items() if k != "refit_null"}, indent=1)); print("refit", res["refit_null"]["sd"], res["refit_null"]["share_le_obs"], np.round(cs, 4))
