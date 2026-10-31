"""Exact run-level stratified bootstrap distribution of D (no Monte Carlo).
D = U - R = mean_m (p_uF,m - p_lR,m) exactly, so only the useful+foreign and useless+real strata matter.
Each stratum's resampled count is Binomial(20, observed share); D*100 = sum of counts differences. Convolve.
Also: how often a B=10,000 Monte Carlo bootstrap (numpy percentile, linear) puts the upper bound above 0.15."""
import json, numpy as np
from scipy import stats
v = json.load(open("vera_counted.json"))
N = 20
pm = v["per_model"]
dist = np.ones(1); off = 0  # distribution over integer sum of (cF - cL)
for m, r in pm.items():
    uF, lR = r["cells"][1], r["cells"][2]
    def pmf(p):
        k = np.arange(N + 1); out = np.exp(stats.binom.logpmf(k, N, p)) if 0 < p < 1 else (k == round(p * N)).astype(float)
        return out / out.sum()
    f = pmf(uF); l = pmf(lR)
    d = np.convolve(f, l[::-1])  # support -N..N
    dist = np.convolve(dist, d); off += N
vals = (np.arange(len(dist)) - off) / (N * 5)
cdf = np.cumsum(dist)
def q(p): return vals[np.searchsorted(cdf, p - 1e-15)]
print("total prob", dist.sum()); print("exact D mean", float((vals * dist).sum()))
print("exact 2.5% quantile", q(.025), " 97.5% quantile", q(.975))
for x in (0.14, 0.15, 0.16):
    print(f"P(D_boot <= {x:.2f}) = {cdf[np.argmin(abs(vals - x))]:.5f}")
print(f"P(D_boot >= 0.16) = {1 - cdf[np.argmin(abs(vals - 0.15))]:.5f}  (the CI upper bound exceeds 0.15 iff this is > 0.025, up to interpolation)")
# Monte Carlo replication of the B=10k percentile over 2000 RNG seeds
rng = np.random.default_rng(7)
hits = 0; R = 2000
for i in range(R):
    s = rng.choice(vals, size=10000, p=dist)
    hits += np.percentile(s, 97.5) > 0.15 + 1e-9
print(f"share of B=10k bootstraps (2000 RNG seeds) with upper bound > 0.15: {hits / R:.3f}")
