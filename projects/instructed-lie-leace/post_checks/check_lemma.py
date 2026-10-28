"""Numeric check of the output-probe bound in the post's expert notes (Selma).

Claim: for probability vectors p, q on the same tokens and any weights w,
    |w.p - w.q| <= (max w - min w) * TV(p, q) <= (max w - min w) * sqrt(KL(p||q) / 2),
with TV(p, q) = 0.5 * sum |p - q| (Pinsker's inequality for the second step).
Also checks the line shift: if every negative moves by at most D, the k-th highest negative
score moves by at most D, so a positive with clean margin > 2D above the line stays above it.
"""
import numpy as np

rng = np.random.default_rng(1028)
worst_tv, worst_kl = 0.0, 0.0
for trial in range(20000):
    n = rng.integers(2, 200)
    p = rng.dirichlet(np.full(n, rng.uniform(0.05, 3))) + 1e-12
    p /= p.sum()
    q = p * np.exp(rng.normal(0, rng.uniform(0.001, 2), n))
    q /= q.sum()
    w = rng.normal(0, rng.uniform(0.1, 10), n)
    lhs = abs(w @ p - w @ q)
    tv = 0.5 * np.abs(p - q).sum()
    kl = float(np.sum(p * (np.log(p) - np.log(q))))
    b_tv = (w.max() - w.min()) * tv
    b_kl = (w.max() - w.min()) * np.sqrt(kl / 2)
    assert lhs <= b_tv * (1 + 1e-9) + 1e-15, (trial, lhs, b_tv)
    assert b_tv <= b_kl * (1 + 1e-9) + 1e-15, (trial, b_tv, b_kl)
    worst_tv = max(worst_tv, lhs / b_tv if b_tv > 0 else 0)
    worst_kl = max(worst_kl, lhs / b_kl if b_kl > 0 else 0)

# rank shift: if m < k negatives move without bound and the rest by at most D,
# the new k-th highest is at most the original (k-m)-th highest + D
for trial in range(5000):
    m_neg = rng.integers(200, 3000)
    neg = rng.normal(size=m_neg)
    k = max(2, int(np.floor(0.01 * (m_neg + 1))))
    m = rng.integers(0, k)
    D = rng.uniform(0, 0.5)
    neg2 = neg + rng.uniform(-D, D, m_neg)
    wild = rng.choice(m_neg, m, replace=False)
    neg2[wild] = rng.normal(0, 50, m)
    assert np.sort(neg2)[-k] <= np.sort(neg)[-(k - m)] + D + 1e-12

# order-statistic step: shifting each negative by at most D moves the k-th highest by at most D
for trial in range(5000):
    m = rng.integers(50, 2000)
    neg = rng.normal(size=m)
    D = rng.uniform(0, 0.5)
    neg2 = neg + rng.uniform(-D, D, m)
    k = max(1, int(np.floor(0.01 * (m + 1))))
    line, line2 = np.sort(neg)[-k], np.sort(neg2)[-k]
    assert abs(line2 - line) <= D + 1e-12
    pos = line + 2 * D + rng.uniform(1e-9, 1, 100)
    pos2 = pos + rng.uniform(-D, D, 100)
    assert np.all(pos2 > line2)

print(f"PASS: 20000 random cases; max |dw.p| / (range(w) TV) = {worst_tv:.4f}, "
      f"max |dw.p| / (range(w) sqrt(KL/2)) = {worst_kl:.4f}; order-statistic step 5000 cases; rank-shift step 5000 cases")
