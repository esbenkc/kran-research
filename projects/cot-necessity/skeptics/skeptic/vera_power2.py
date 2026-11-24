"""Operating characteristics of Felix FINAL PLAN v1 (#29345) frozen rule.
seed-YES: 2 consecutive task-held checkpoints (or the final one) with reader Wilson upper < 0.50
seed-NO : every task-held checkpoint has reader Wilson lower >= 0.50 (length check assumed met)
CONFIRMED = 3/3 seed-YES; REJECTED = 3/3 seed-NO; K=6, n=200, task held everywhere.
Reader accuracy here = P(correct, abstain counted as miss)."""
import numpy as np
from vera_power import wilson
K, N = 6, 200
rng = np.random.default_rng(1)
print(" true reader acc   P(CONF)  P(REJ)  P(UNCONF)")
for p in [0.30,0.40,0.44,0.46,0.48,0.50,0.52,0.54,0.56,0.58,0.60,0.65,0.70]:
    k = rng.binomial(N, p, size=(20000, 3, K))
    lo, hi = wilson(k, N)
    low = hi < 0.50
    sy = (low[:, :, 1:] & low[:, :, :-1]).any(-1) | low[:, :, -1]
    sn = (lo >= 0.50).all(-1)
    y, n = (sy.sum(-1) == 3).mean(), (sn.sum(-1) == 3).mean()
    print(f"      {p:.2f}        {y:.4f}  {n:.4f}  {1-y-n:.4f}")
