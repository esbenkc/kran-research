"""How much does a 2-of-3 seed count carry? Exact numbers, no data files needed (counts from vera_check_out.txt).
- posterior for the per-seed take-off probability p (uniform prior): Beta(1+2, 1+1)
- P(status CONFIRMED | p) under the frozen rule with 3 seeds = P(>= 2 of 3 take off)
- Fisher exact (one-sided) for 'KL-on take-off rate < KL-off rate' with KL-on 0 of 1 (parent live_k4) vs KL-off 2 of 3
"""
from math import comb

from scipy.stats import beta, fisher_exact

k, n = 2, 3
lo, hi = beta.ppf([0.025, 0.975], 1 + k, 1 + n - k)
print(f"take-off probability per seed, Beta(3,2): mean {(1+k)/(2+n):.2f}, 95% interval [{lo:.2f}, {hi:.2f}]")
for p in (0.2, 0.35, 0.5, 0.65, 0.8):
    pc = sum(comb(3, j) * p**j * (1 - p) ** (3 - j) for j in (2, 3))
    print(f"  P(CONFIRMED with 3 seeds | true p = {p:.2f}) = {pc:.3f}")
_, pf = fisher_exact([[2, 1], [0, 1]], alternative="greater")
print(f"Fisher one-sided, KL-off 2/3 vs KL-on 0/1: p = {pf:.2f}  (a single flat KL-on seed cannot show the penalty brakes)")
