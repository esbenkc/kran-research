"""Numeric check of the post's LEACE equation (expert foldout): for a binary concept,
r(x) = x - delta delta^T Sigma^-1 (x - mu) / (delta^T Sigma^-1 delta)
equals concept_erasure.LeaceFitter's eraser when Sigma is the same covariance (shrinkage off, full-rank data),
and it equalises the class means on the fit data. Synthetic, d = 64, n = 4000, seed 1028."""
import numpy as np, torch
from concept_erasure import LeaceFitter
rng = np.random.default_rng(1028); d, n = 64, 4000
A = rng.normal(size=(d, d)) / np.sqrt(d); z = rng.integers(0, 2, n).astype(float)
X = rng.normal(size=(n, d)) @ A.T + np.outer(z, rng.normal(size=d)) + 3.0
f = LeaceFitter(d, 1, dtype=torch.float64, shrinkage=False)
f.update(torch.tensor(X), torch.tensor(z)[:, None]); lib = f.eraser(torch.tensor(X)).numpy()
mu = X.mean(0); S = np.cov(X.T, bias=True); delta = X[z == 1].mean(0) - X[z == 0].mean(0)
Si = np.linalg.inv(S); r = X - ((X - mu) @ Si @ delta)[:, None] * delta[None, :] / (delta @ Si @ delta)
print("max |formula - library| =", np.abs(r - lib).max())
print("class-mean gap after formula =", np.abs(r[z == 1].mean(0) - r[z == 0].mean(0)).max())
assert np.abs(r - lib).max() < 1e-6 and np.abs(r[z == 1].mean(0) - r[z == 0].mean(0)).max() < 1e-8
print("PASS")
