"""Hugo's own numeric checks of the two math statements the post uses.
(1) Shared-shift lemma: a common shift a on every token score leaves the re-thresholded catch unchanged,
    for max and mean pooling alike.
(2) Bell-curve formula Phi(D - (1-c) p - z), z = Phi^-1(1 - 2/201): Monte Carlo with the 2nd-of-200 rule
    vs the z-formula (the finite rule sits slightly below the formula; Iris reported up to -0.036).
"""
import numpy as np
from scipy.stats import norm

rng = np.random.default_rng(7)
# (1) lemma on random token-score matrices
n, L = 400, 9
s = rng.normal(size=(n, L)); y = rng.integers(0, 2, n); s[y == 1, -2:] += 3
a = 5.3
for pool in (np.max, np.mean):
    for ss in (s, s - a):
        sc = pool(ss, axis=1); t = np.sort(sc[y == 0])[-2]
        print(pool.__name__, "shift" if ss is not s else "clean", (sc[y == 1] > t).mean())
# (2) bell-curve formula vs finite rule
z = norm.ppf(1 - 2 / 201)
for D, c, p in [(4, 0.5, 3), (5, 0.6, 3), (3, 0.9, 2), (4, 0.0, 1)]:
    pred = norm.cdf(D - (1 - c) * p - z)
    sims = []
    for _ in range(4000):
        T1 = rng.normal(-c * p, 1, 200)            # true scores after attack (clean mean 0, sd 1)
        t = np.sort(T1)[-2]
        sims.append(1 - norm.cdf(t - (D - p)))      # false mean after attack = D - p
    print(f"D={D} c={c} p={p}: formula {pred:.3f}  finite 2nd-of-200 MC {np.mean(sims):.3f}")
