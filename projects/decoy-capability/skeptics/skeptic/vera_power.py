"""Pre-data power of the frozen s6 status rule (two-level bootstrap) on synthetic ladders."""
import sys
import numpy as np
from vera_stats import per_model, fitted, summary, boot, status


def sim(K, n, drop, touch=0.6, base=0.7, reps=100, B=1000, seed=0):
    rng = np.random.default_rng(seed)
    res = {"CONFIRMED": 0, "REJECTED": 0, "UNCONFIRMED": 0}
    caps = np.linspace(0.1, 0.9, K)
    for r in range(reps):
        eps = []
        for i, cap in enumerate(caps):
            m = f"m{i}"
            for s in range(20):
                eps.append(dict(model=m, arm="nobait", seed=100 + s, touch=0, use=0, score=float(rng.random() < cap)))
            f = (cap - caps[0]) / (caps[-1] - caps[0])
            for s in range(n):
                t = int(rng.random() < touch)
                eps.append(dict(model=m, arm="static", seed=s, touch=t, use=int(t and rng.random() < base - drop * f), score=0))
                eps.append(dict(model=m, arm="stateful", seed=s, touch=t, use=int(t and rng.random() < base), score=0))
        pm = per_model(eps, [f"m{i}" for i in range(K)])
        fm = fitted(pm)
        s = summary(pm, fm)
        res[status(s, boot(pm, fm, s["R"], B=B, seed=r))] += 1
    return {k: v / reps for k, v in res.items()}


if __name__ == "__main__":
    for K, n, drop in [(5, 40, 0.0), (5, 40, 0.3), (6, 40, 0.3), (6, 80, 0.3), (6, 40, 0.5), (8, 40, 0.3)]:
        print(K, n, drop, sim(K, n, drop), flush=True)
