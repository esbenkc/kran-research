"""skeptic2 self-test: my gradient vs finite differences, and the regular-model SGLD vs its exact value."""
import numpy as np

from sgld2 import qnet_loss_grad, sgld_llc_quadratic

rng = np.random.default_rng(1)
p, K, N = 7, 5, 30
A = rng.integers(0, p, N); B = rng.integers(0, p, N); Y = np.eye(p)[(A + B) % p]
U, S, V = (rng.standard_normal((1, p, K)) for _ in range(3))
L, gU, gS, gV = qnet_loss_grad(U, S, V, A, B, Y)
err = 0
for arr, g in ((U, gU), (S, gS), (V, gV)):
    for _ in range(5):
        i = tuple(rng.integers(0, s) for s in arr.shape)
        e = 1e-6
        arr[i] += e; Lp = qnet_loss_grad(U, S, V, A, B, Y)[0][0]
        arr[i] -= 2 * e; Lm = qnet_loss_grad(U, S, V, A, B, Y)[0][0]
        arr[i] += e
        err = max(err, abs((Lp - Lm) / (2 * e) - g[i]) / (abs(g[i]) + 1e-8))
print(f"max rel grad err {err:.2e}")
assert err < 1e-5

h = np.exp(rng.uniform(-2, 2, 40))
r = sgld_llc_quadratic(h, eps=1e-3, gamma=1.0, nbeta=100.0, steps=20000, burn=2000, chains=8)
print(r)
assert abs(r["lambda_hat"] - r["exact_discrete"]) < 4 * r["se"] + 0.02 * r["exact_discrete"]
print("SELFTEST PASS")
