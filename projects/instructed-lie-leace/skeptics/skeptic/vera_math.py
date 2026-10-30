"""Fuzz checks of the post's three math steps (own code).
(1) |w.(p-q)| <= (max w - min w) TV(p,q) <= range * sqrt(KL(p||q)/2) for random p, q on a simplex.
(2) order statistic: if all but m of n negatives move by <= D, the new k-th highest <= old (k-m)-th highest + D.
(3) rank-1 LEACE r(x) = x - d d' S^-1 (x-mu) / (d' S^-1 d) equalises class means for ANY SPD S (shrunk or not)."""
import numpy as np
rng = np.random.default_rng(0)
worst1 = 0
for _ in range(20000):
    V = rng.integers(2, 50)
    p = rng.dirichlet(np.ones(V) * rng.uniform(0.05, 2)); q = rng.dirichlet(np.ones(V) * rng.uniform(0.05, 2))
    q = 0.5 * q + 0.5 * p if rng.random() < 0.5 else q
    w = rng.normal(size=V) * rng.uniform(0.1, 10)
    tv = 0.5 * np.abs(p - q).sum(); kl = np.sum(p * (np.log(p + 1e-300) - np.log(q + 1e-300)))
    lhs = abs(w @ (p - q)); r = w.max() - w.min()
    worst1 = max(worst1, lhs - r * tv, r * tv - r * np.sqrt(kl / 2))
print("(1) max violation (should be <= ~1e-12):", worst1)
bad2 = 0
for _ in range(20000):
    n = rng.integers(20, 400); k = rng.integers(1, 10); m = rng.integers(0, k)
    s = rng.normal(size=n); D = rng.uniform(0, 1)
    s2 = s + rng.uniform(-D, D, size=n)
    idx = rng.choice(n, m, replace=False); s2[idx] = s[idx] + rng.normal(0, 50, size=m)
    new_k = np.sort(s2)[::-1][k - 1]; old_km = np.sort(s)[::-1][k - m - 1]
    bad2 += new_k > old_km + D + 1e-12
print("(2) violations of new k-th <= old (k-m)-th + D:", bad2, "of 20000")
bad3 = 0
for _ in range(500):
    d_ = rng.integers(3, 30); n = 400
    z = rng.random(n) < rng.uniform(0.2, 0.8)
    X = rng.normal(size=(n, d_)) @ rng.normal(size=(d_, d_)) + np.outer(z, rng.normal(size=d_)) * 3
    mu = X.mean(0); dl = X[z].mean(0) - X[~z].mean(0)
    A = rng.normal(size=(d_, d_)); S = A @ A.T + np.eye(d_) * rng.uniform(0.01, 5)
    Si = np.linalg.inv(S)
    R = X - ((X - mu) @ Si @ dl)[:, None] * dl[None, :] / (dl @ Si @ dl)
    bad3 += np.abs(R[z].mean(0) - R[~z].mean(0)).max() > 1e-9 * (1 + np.abs(X).max())
print("(3) LEACE class-mean gap failures:", bad3, "of 500")
