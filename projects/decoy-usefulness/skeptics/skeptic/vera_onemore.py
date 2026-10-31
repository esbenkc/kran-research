"""Check the post sentence 'one more use of the stranger's useful key, out of 100 runs, would have made the answer
Not settled': add 1 use to useful+foreign of each model with room, recompute the exact run-level bootstrap CI."""
import numpy as np
from scipy import stats
N = 20
uF = {"haiku": 18, "gem3.1": 20, "gem3.5": 20, "5.4-nano": 16, "5.4-mini": 13}
lR = {"haiku": 18, "gem3.1": 20, "gem3.5": 20, "5.4-nano": 8, "5.4-mini": 15}
def pmf(k):
    x = np.arange(N + 1); o = np.exp(stats.binom.logpmf(x, N, k / N)) if 0 < k < N else (x == k).astype(float); return o / o.sum()
def ci(uF):
    d = np.ones(1)
    for m in uF: d = np.convolve(d, np.convolve(pmf(uF[m]), pmf(lR[m])[::-1]))
    v = (np.arange(len(d)) - 5 * N) / (5 * N); c = np.cumsum(d)
    return v[np.searchsorted(c, .025 - 1e-12)], v[np.searchsorted(c, .975 - 1e-12)]
print("observed", ci(uF))
for m in uF:
    if uF[m] < N:
        u = dict(uF); u[m] += 1; lo, hi = ci(u)
        print(f"+1 use in {m:9s} useful+foreign: CI [{lo:+.2f}, {hi:+.2f}] -> {'REJECTED' if lo >= -.15 and hi <= .15 else 'not REJECTED'}")
for m in lR:
    if lR[m] > 0:
        l = dict(lR); l[m] -= 1; lR_s = lR.copy(); lR.update(l); lo, hi = ci(uF); lR.update(lR_s)
        print(f"-1 use in {m:9s} useless+real : CI [{lo:+.2f}, {hi:+.2f}] -> {'REJECTED' if lo >= -.15 and hi <= .15 else 'not REJECTED'}")
