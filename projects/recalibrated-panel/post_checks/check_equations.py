"""Checks every displayed equation in _notes/recalibrated-panel.md (blog.kran.ai) with sympy or numpy.

Run: python3 post_checks/check_equations.py   (numpy + sympy only, CPU, < 1 s)

Eq. 1  M(x) = max_l (s_l(x) - m_l) / d_l, line = 2nd-highest M of n = 200 true statements:
       a fresh exchangeable true statement crosses the k-th highest of n with probability k/(n+1) = 2/201.
Eq. 2  in-loop loss: v_l^2 * ((s_f - s_t)/d_l)^2 equals the 1022 term (s_f - s_t)^2 when d_l = v_l, the
       re-standardisation mean cancels in the twin gap, and d/dd of the term is -2 v^2 g^2 / d^3 (so spreading the
       true scores lowers the loss; matches attack_recal.py line ~230).
Eq. 3  Samuelson bound: after re-standardisation (mean 0, mean square 1, ddof 0) of n scores, the k-th highest
       z_(k) <= sqrt((n-k)/k); tight at k scores equal to a, the rest equal to -k a/(n-k). n=200: 14.1, 9.95, 7.0.
"""
import numpy as np
import sympy as sp

# Eq. 1: exceedance probability of the k-th highest of n exchangeable negatives
n, k = 200, 2
rng = np.random.default_rng(0)
trials = 200_000
x = rng.standard_normal((trials, n + 1))
line = np.sort(x[:, :n], axis=1)[:, n - k]
p_mc = float(np.mean(x[:, n] > line))
assert abs(p_mc - k / (n + 1)) < 0.002, p_mc
print(f"Eq.1 line rule: P(fresh true > 2nd-highest of 200) = {p_mc:.5f} (exact 2/201 = {2/201:.5f})")

# Eq. 1: M is invariant to a positive affine change of one probe's raw scores (re-standardisation removes it)
s = rng.standard_normal((200, 20))
a, b = 3.7, -1.2
s2 = s.copy(); s2[:, 5] = a * s2[:, 5] + b
z = lambda S: (S - S.mean(0)) / S.std(0)
assert np.allclose(z(s).max(1), z(s2).max(1))
print("Eq.1 invariance: M unchanged under a*s_l + b (a > 0) for one probe: OK")

# Eq. 2: symbolic checks of the in-loop probe term
sf, st, m, d, v = sp.symbols("s_f s_t m d v", positive=True)
term = v**2 * (((sf - m) / d) - ((st - m) / d))**2
assert sp.simplify(term - v**2 * (sf - st)**2 / d**2) == 0           # m cancels
assert sp.simplify(term.subs(d, v) - (sf - st)**2) == 0              # equals the 1022 term at d = v
g = sp.symbols("g", positive=True)
dterm = sp.diff(v**2 * g**2 / d**2, d)
assert sp.simplify(dterm + 2 * v**2 * g**2 / d**3) == 0               # d/dd = -2 v^2 g^2 / d^3 < 0
print("Eq.2 loss: mean cancels, equals the 1022 term at d = v, and decreases in d: OK")

# Eq. 3: Samuelson-type bound on the k-th highest standardised score
N, K, A = sp.symbols("n k a", positive=True)
rest = -K * A / (N - K)
sol = sp.solve(sp.Eq(K * A**2 + (N - K) * rest**2, N), A)
assert any(sp.simplify(r - sp.sqrt((N - K) / K)) == 0 for r in sol), sol
for kk, want in [(1, 14.1), (2, 9.95), (4, 7.0)]:
    val = float(sp.sqrt(sp.Rational(200 - kk, kk)))
    assert abs(val - want) < 0.01, (kk, val)
# numeric: random search never beats the bound, and the extremal vector reaches it
for kk in (1, 2, 4):
    best = 0.0
    for _ in range(2000):
        x = rng.standard_normal(200) ** 3
        x[:kk] += rng.uniform(0, 50)
        zz = np.sort((x - x.mean()) / x.std())[::-1]
        best = max(best, zz[kk - 1])
    bound = np.sqrt((200 - kk) / kk)
    ext = np.r_[np.full(kk, 1.0), np.full(200 - kk, -kk / (200 - kk))]
    ze = np.sort((ext - ext.mean()) / ext.std())[::-1]
    assert best <= bound + 1e-9 and abs(ze[kk - 1] - bound) < 1e-9
    print(f"Eq.3 k={kk}: bound {bound:.3f}, extremal {ze[kk-1]:.3f}, best of 2000 random {best:.3f}")
print("Eq.3 Samuelson bound: OK")
