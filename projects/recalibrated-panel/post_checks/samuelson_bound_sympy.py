"""Equation check (Iris): after mean/sd (ddof 0) standardisation of n values, the k-th highest z obeys
    z_(k) <= sqrt((n - k) / k),
with equality when the top k values are equal and the other n - k are equal (k outliers of infinite height).
Checks: (1) sympy: the extremal configuration a (k times), b (n-k times) with mean 0 and mean square 1 gives
a = sqrt((n-k)/k); (2) the Cauchy-Schwarz step; (3) numeric worst case by optimisation for n = 200, k = 1, 2, 4;
(4) the numbers used in the post: 14.107 (k=1), 9.950 (k=2), 7.000 (k=4) and the 1%-line limit sqrt(99).
"""
import numpy as np
import sympy as sp
from scipy.optimize import minimize

n, k, a, b, t = sp.symbols("n k a b t", positive=True)
# (1) extremal configuration: k values at a, n-k values at b; mean 0, mean square 1
sol = sp.solve([k * a + (n - k) * b, k * a**2 + (n - k) * b**2 - n], [a, b], dict=True)
apos = [s[a] for s in sol if sp.simplify(s[a]).is_positive is not False][0]
assert sp.simplify(apos**2 - (n - k) / k) == 0, apos
# (2) bound: top k >= t; rest sum <= -k t; sum of squares of rest >= (k t)^2/(n-k); so k t^2 + k^2 t^2/(n-k) <= n
lhs = k * t**2 + k**2 * t**2 / (n - k)
tmax = sp.solve(sp.Eq(lhs, n), t)
assert any(sp.simplify(s**2 - (n - k) / k) == 0 for s in tmax), tmax


# (3) numeric worst case: maximise the k-th highest standardised value over x in R^n
def kth_max(x, kk):
    z = (x - x.mean()) / x.std()
    return np.sort(z)[::-1][kk - 1]


N = 200
rng = np.random.default_rng(0)
for kk in (1, 2, 4):
    best = -np.inf
    for _ in range(5):
        x0 = rng.standard_normal(N)
        res = minimize(lambda x: -kth_max(x, kk), x0, method="Powell", options={"maxiter": 20000, "xtol": 1e-10})
        best = max(best, -res.fun)
    x = rng.standard_normal(N); x[:kk] += 1e7
    best = max(best, kth_max(x, kk))
    bound = np.sqrt((N - kk) / kk)
    assert best <= bound + 1e-9, (kk, best, bound)
    assert bound - best < 1e-4, (kk, best, bound)
    print(f"k={kk}: numeric worst case {best:.6f}, bound {bound:.6f}")
# (4) post numbers
assert abs(np.sqrt(199) - 14.107) < 1e-3 and abs(np.sqrt(99) - 9.950) < 1e-3 and abs(np.sqrt(49) - 7.0) < 1e-12
for NN in (200, 1000, 10000):
    kk = int(np.floor(0.01 * (NN + 1)))
    print(f"n={NN}: 1% line rank k={kk}, bound {np.sqrt((NN - kk) / kk):.3f}")
print("all equation checks pass")
