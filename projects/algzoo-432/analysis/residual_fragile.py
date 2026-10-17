"""T16 (Iris): split C1's disagreements with the model into weight-noise-FRAGILE and ROBUST inputs.

Fragile = the model's argmax changes under at least one of 20 draws of 0.1% multiplicative weight noise
(the noise level that defines the bar 0.98671). Dev seed 82 (3e5), noise rng 7. Never the holdout.
Usage: .venv/bin/python residual_fragile.py   (~30 s)
"""
import json
import numpy as np
from residual_common import W_hi, W_hh, W_oh, data, c1, aug_order


def run(x, Wi, Wh, Wo):
    h = np.zeros((len(x), 16))
    for t in range(10):
        h = np.maximum(h @ Wh.T + x[:, t:t + 1] * Wi[None, :], 0)
    return h @ Wo.T


X, L, m = data(82, 300_000)
rng = np.random.default_rng(7)
flips = np.zeros(len(X), int); agr = []
for _ in range(20):
    e = lambda W: W * (1 + 0.001 * rng.standard_normal(W.shape))
    mp = run(X, e(W_hi), e(W_hh), e(W_oh)).argmax(1)
    flips += mp != m; agr.append(float((mp == m).mean()))
frag = flips > 0
dis = c1(X) != m
O = np.maximum(aug_order(X), 0)
rk = np.where(O == m[:, None], np.arange(11)[None, :], 99).min(1)
rob = dis & ~frag
res = dict(
    noise_agree_mean=round(float(np.mean(agr)), 5),
    fragile_frac=round(float(frag.mean()), 4),
    c1_disagree=round(float(dis.mean()), 4),
    c1_disagree_robust_frac_of_all=round(float(rob.mean()), 4),
    c1_disagree_fragile_frac_of_all=round(float((dis & frag).mean()), 4),
    c1_agree_on_robust=round(float((~dis[~frag]).mean()), 5),
    c1_agree_on_fragile=round(float((~dis[frag]).mean()), 4),
    robust_dis_pick_rank={"max": round(float((rk[rob] == 0).mean()), 3), "3rd": round(float((rk[rob] == 2).mean()), 3),
                          ">=4th": round(float((rk[rob] >= 3).mean()), 3)},
)
print(json.dumps(res, indent=1))
json.dump(res, open("residual_fragile.json", "w"), indent=1)
