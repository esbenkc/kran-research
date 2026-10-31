"""Vera: exact null distribution of Spearman rho (no ties) for n=5 and n=10: P(|rho| < 0.3), P(|rho| <= 0.3), 5% critical."""
import itertools, numpy as np
from math import factorial
for n in (5, 10):
    base = np.arange(n)
    if n <= 8:
        perms = np.array(list(itertools.permutations(range(n))))
        d2 = ((perms - base) ** 2).sum(1)
    else:
        # exact via DP over sum of squared displacements is heavy; enumerate 10! = 3.6M in chunks
        d2 = []
        for p in itertools.permutations(range(n)):
            d2.append(sum((a - b) ** 2 for a, b in zip(p, base)))
        d2 = np.array(d2)
    rho = 1 - 6 * d2 / (n * (n * n - 1))
    print(f"n={n}: {len(rho)} orders, distinct values {len(np.unique(np.round(rho,6)))}; P(|rho|<0.3) = {np.mean(np.abs(rho) < 0.3 - 1e-12):.4f}, P(|rho|<=0.3) = {np.mean(np.abs(rho) <= 0.3 + 1e-12):.4f}; two-sided 5% needs |rho| >= {np.quantile(np.abs(rho), 0.95):.3f}")
