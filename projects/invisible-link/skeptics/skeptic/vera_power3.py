"""Vera: re-check the frozen PREREG s4 status rule with the CLUSTER bootstrap (s4), before v1.1.
Design: 3 agents x 5 templates x 6 paraphrases x 2 views x 2 arms (fit, nonfit), STUCK, all exposed.
Gap = mean over agents of (fit rate - nonfit rate), each pooled over views.
CI: percentile, resample the 6 (template, paraphrase) clusters with replacement within each agent x template
stratum; a cluster brings all 4 runs. YES first: gap >= .05 and lo > 0. NO: hi < .15. else NS.
Scenarios: independent runs; and a paraphrase-level logit effect sd 1.0 shared by the cluster's 4 runs.
"""
import numpy as np
from scipy.special import expit, logit

rng = np.random.default_rng(110803)
A, T, P, V = 3, 5, 6, 2
B = 1000


def simulate(base, gap, clus_sd, reps):
    pn = np.full(A, base)
    pf = np.clip(pn + gap, 1e-6, 1 - 1e-6)
    pn = np.clip(pn, 1e-6, 1 - 1e-6)
    u = rng.normal(0, clus_sd, (reps, A, T, P, 1))  # shared within cluster
    lf = logit(pf)[None, :, None, None, None] + u
    ln = logit(pn)[None, :, None, None, None] + u
    f = rng.random((reps, A, T, P, V)) < expit(lf)
    o = rng.random((reps, A, T, P, V)) < expit(ln)
    # per-cluster sums: (reps, A, T, P)
    fs, os_ = f.sum(-1), o.sum(-1)
    d = ((fs.sum((2, 3)) - os_.sum((2, 3))) / (T * P * V)).mean(1)
    idx = rng.integers(0, P, (B, A, T, P))  # same resample indices across reps (fine for coverage)
    res = {"YES": 0, "NO": 0, "NS": 0}
    for r in range(reps):
        diff = fs[r] - os_[r]  # (A, T, P)
        bd = np.take_along_axis(np.broadcast_to(diff, (B, A, T, P)), idx, 3).sum((2, 3)) / (T * P * V)
        bd = bd.mean(1)
        lo, hi = np.quantile(bd, [0.025, 0.975])
        if d[r] >= 0.05 and lo > 0:
            res["YES"] += 1
        elif hi < 0.15:
            res["NO"] += 1
        else:
            res["NS"] += 1
    return {k: round(v / reps, 3) for k, v in res.items()}


if __name__ == "__main__":
    for clus_sd in (0.0, 1.0):
        for base in (0.1, 0.3, 0.7):
            for gap in (0.0, 0.15):
                print(f"cluster_sd={clus_sd} nonfit={base} gap={gap:+.2f}: {simulate(base, gap, clus_sd, 1000)}",
                      flush=True)
