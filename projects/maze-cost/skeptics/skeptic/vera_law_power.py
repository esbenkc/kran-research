"""Vera: pre-data power of the '2x geometric give-up law' rule.

Question: can the rule 'a constant per-page leave rate fitted on maze pages 1-3
predicts the median total maze spend within 2x' fail when the truth is NOT
geometric, at n runs per agent and a step cap K?

Cost of reaching depth d (pages): sum_{i=1..d} (a + b*i) tokens, since the
harness re-sends the history (tokens per step grow with depth).
Leave hazard per page after viewing it. Censored at K.
Fit: h_hat = leaves on pages 1-3 / page-views on pages 1-3 (geometric MLE,
right-censored at 3). Predicted median depth = first d with (1-h)^d <= 0.5,
capped at K. Pass = predicted/observed median cost in [0.5, 2].
"""
import numpy as np

A, B = 1500, 1000  # tokens per step: base + growth per page (rough guesses)


def cost(d):
    d = np.asarray(d)
    return A * d + B * d * (d + 1) / 2


def sim_depths(hfun, n, K, rng):
    out = np.empty(n, int)
    for r in range(n):
        d = 0
        while d < K:
            d += 1
            if rng.random() < hfun(d, r):
                break
        out[r] = d
    return out


def rule(depths, K, cap_guard=True):
    # pages 1-3 exposure
    views = np.minimum(depths, 3).sum()
    leaves = ((depths <= 3) & (depths < K)).sum()
    h = leaves / views if views else 0.0
    if h <= 0:
        dmed = K
    else:
        dmed = min(K, int(np.ceil(np.log(0.5) / np.log(1 - h)))) if h < 1 else 1
    if cap_guard and dmed >= K:
        return "not_testable"
    obs = np.median(cost(depths))
    pred = cost(dmed)
    return "pass" if 0.5 <= pred / obs <= 2 else "fail"


def run(name, hfun_factory, n=40, K=15, reps=2000, seed=0):
    rng = np.random.default_rng(seed)
    res = {"pass": 0, "fail": 0, "not_testable": 0}
    for _ in range(reps):
        hfun = hfun_factory(rng, n)
        res[rule(sim_depths(hfun, n, K, rng), K)] += 1
    tot = sum(res.values())
    print(f"{name:45s} n={n} K={K}  " + "  ".join(f"{k}={v/tot:.2f}" for k, v in res.items()))


if __name__ == "__main__":
    for n in (40, 20):
        for K in (15, 30):
            print(f"--- n={n}, cap K={K}")
            for h in (0.05, 0.1, 0.2, 0.35):
                run(f"geometric h={h}", lambda rng, n, h=h: (lambda d, r: h), n, K)
            # rising hazard: agent realises the maze is endless
            run("rising h=0.03*d", lambda rng, n: (lambda d, r: min(1, 0.03 * d)), n, K)
            run("rising h=0.06*d", lambda rng, n: (lambda d, r: min(1, 0.06 * d)), n, K)
            # falling hazard: survivors are 'committed'
            run("falling h=0.4/d", lambda rng, n: (lambda d, r: 0.4 / d), n, K)
            # mixture: 50% leave fast (h=.5), 50% never
            def mix(rng, n, p=0.5, hf=0.5):
                types = rng.random(n) < p
                return lambda d, r: hf if types[r] else 0.0
            run("mixture 50% h=.5 / 50% h=0", mix, n, K)
            run("mixture 30% h=.6 / 70% h=0", lambda rng, n: mix(rng, n, 0.3, 0.6), n, K)
            run("mixture 60% h=.6 / 40% h=0", lambda rng, n: mix(rng, n, 0.6, 0.6), n, K)
