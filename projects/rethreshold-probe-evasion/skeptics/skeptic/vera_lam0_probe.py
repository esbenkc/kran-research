"""Vera: is the separation GONE at lambda=0, or did it move to another direction?
Fresh logistic probes on the attacked probe-split activations (local, gitignored data/runs/*__acts.npz),
C grid by 5-fold CV AUROC (same recipe as the frozen probe), scored on eval with the 2nd-largest-of-200 rule.
Also: rank of the frozen refit-200's top items among all 796 held-out negatives (is the frozen set atypical?).
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score

D = "kran-research/projects/rethreshold-probe-evasion/data/"
c = np.load(D + "clean_acts.npz")
y, sp, s0 = c["y"], c["split"], c["score"].astype(float)
rn, ep, en = (sp == "refit") & (y == 0), (sp == "eval") & (y == 1), (sp == "eval") & (y == 0)
m = sp == "probe"
for lam in ("0", "100000"):
    for seed in (0, 1, 2):
        A = np.load(D + f"runs/lam{lam}__seed{seed}__acts.npz")["acts_final"].astype(float)
        mu, sd = A[m].mean(0), A[m].std(0) + 1e-6
        Z = (A - mu) / sd
        cv = {C: cross_val_score(LogisticRegression(C=C, max_iter=3000), Z[m], y[m], cv=5, scoring="roc_auc").mean()
              for C in (0.01, 0.1, 1.0, 10.0)}
        Cb = max(cv, key=lambda k: (round(cv[k], 6), -k))
        lr = LogisticRegression(C=Cb, max_iter=3000).fit(Z[m], y[m])
        s = lr.decision_function(Z)
        t = np.sort(s[rn])[-2]
        print("lam %-6s seed %d  fresh probe C=%-5s eval AUROC %.4f  TPR@1%% %.3f  FPR %.4f   (frozen-probe AUROC %.3f)" % (
            lam, seed, Cb, roc_auc_score(y[sp == "eval"], s[sp == "eval"]), (s[ep] > t).mean(), (s[en] > t).mean(),
            roc_auc_score(y[sp == "eval"], (np.load(D + f"runs/lam{lam}__seed{seed}.npz")["scores"][-1])[sp == "eval"])))

neg = rn | en
order = np.argsort(-s0[neg])
isref = rn[neg][order]
print("clean model: ranks (1 = highest of 796) of the refit-200 members among the top 10:",
      [i + 1 for i in range(10) if isref[i]])
for lam in ("10000", "100000"):
    f = np.load(D + f"runs/lam{lam}__seed0.npz")["scores"][-1].astype(float)
    o = np.argsort(-f[neg])
    print("lam %s seed0 attacked: refit-200 members among top 10 of 796:" % lam, [i + 1 for i in range(10) if rn[neg][o][i]])
