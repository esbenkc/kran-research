"""Numeric checks of derivation.md (Vera). Run: python3 vera_theory_check.py"""
import numpy as np
from scipy.stats import norm

rng = np.random.default_rng(1018)
z = norm.ppf(0.99)


def share(mp, sp, mn, sn, mp2, sp2, mn2, sn2):
    t0 = mn + z * sn
    tc = norm.sf((t0 - mp) / sp)
    told = norm.sf((t0 - mp2) / sp2)
    t1 = mn2 + z * sn2
    tref = norm.sf((t1 - mp2) / sp2)
    return tc, told, tref, (tref - told) / (tc - told)


# 6a pure common shift -> R = 1
for c in (1.0, 3.0, 8.0):
    print("common shift c=%.0f  R=%.6f" % (c, share(4, 1, 0, 1, 4 - c, 1, -c, 1)[3]))
# 6b negatives fixed -> R = 0
print("negatives fixed      R=%.6f" % share(4, 1, 0, 1, 1, 1, 0, 1)[3])
# 6c common shift + positive spread x3
print("shift+spread x3      R=%.4f" % share(4, 1, 0, 1, -1, 3, -5, 1)[3])

# Monte Carlo check of step 2/4 vs binormal formula
n = 400_000
neg, pos = rng.normal(-5, 1.2, n), rng.normal(-1, 2.5, n)
t1 = -5 + z * 1.2
print("MC TPR %.4f vs formula %.4f" % ((pos > t1).mean(), norm.sf((t1 + 1) / 2.5)))
# AUROC formula
from sklearn.metrics import roc_auc_score
m = 20000
y = np.r_[np.zeros(m), np.ones(m)]
s = np.r_[neg[:m], pos[:m]]
print("MC AUROC %.4f vs formula %.4f" % (roc_auc_score(y, s), norm.cdf(4 / np.hypot(1.2, 2.5))))
# step 8: out-of-sample FPR of k-th highest of 200
for k in (2, 3):
    f = norm.sf(np.sort(rng.standard_normal((20000, 200)), axis=1)[:, -k])
    print("k=%d n=200 mean FPR %.4f  k/(n+1)=%.4f" % (k, f.mean(), k / 201))
