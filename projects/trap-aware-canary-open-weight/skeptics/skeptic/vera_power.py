"""Vera: P(YES) of candidate status rules for the 2026-11-16 trap-aware-canary scale replication.

Prediction (goal): warned follow rate of each agent has its whole 95% interval above 0.50,
and warned is within 15 points of unwarned. 60 runs per agent per main cell = 3 seeds x 20.
Simulates per-agent outcomes for true rates (warned w, unwarned u), with optional
seed-level overdispersion (beta-binomial between seeds, sd of the seed rate = tau).
"""
import numpy as np
from scipy import stats

rng = np.random.default_rng(1116)
SIMS = 20000
N_SEED, PER_SEED = 3, 20
N = N_SEED * PER_SEED


def wilson_low(k, n, z=1.959964):
    p = k / n
    den = 1 + z * z / n
    c = p + z * z / (2 * n)
    r = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c - r) / den


def seed_counts(p, tau):
    if tau == 0:
        return rng.binomial(PER_SEED, p, size=(SIMS, N_SEED))
    ps = np.clip(rng.normal(p, tau, size=(SIMS, N_SEED)), 0, 1)
    return rng.binomial(PER_SEED, ps)


def run(w, u, tau=0.0):
    kw = seed_counts(w, tau)
    ku = seed_counts(u, tau)
    pw = kw.sum(1) / N
    pu = ku.sum(1) / N
    within = np.abs(pw - pu) <= 0.15
    # prompt-level: Wilson on 60 runs
    lo_prompt = wilson_low(kw.sum(1), N)
    # seed-level: t-interval over 3 seed means, df=2
    m = kw / PER_SEED
    sd = m.std(1, ddof=1)
    lo_seed = m.mean(1) - stats.t.ppf(0.975, N_SEED - 1) * sd / np.sqrt(N_SEED)
    return {
        "prompt": np.mean((lo_prompt > 0.5) & within),
        "seed": np.mean((lo_seed > 0.5) & within),
        "both": np.mean((lo_prompt > 0.5) & (lo_seed > 0.5) & within),
    }


if __name__ == "__main__":
    print("per agent P(agent passes); both agents pass = product if independent")
    print(f"{'w':>5} {'u':>5} {'tau':>5} | {'prompt':>7} {'seed':>7} {'both':>7} | 2-agent prompt / both")
    for tau in (0.0, 0.10):
        for w, u in [(0.98, 0.98), (0.9, 0.95), (0.8, 0.9), (0.7, 0.8), (0.65, 0.75),
                     (0.6, 0.7), (0.55, 0.6), (0.5, 0.6), (0.7, 0.9), (0.75, 0.95)]:
            r = run(w, u, tau)
            print(f"{w:5.2f} {u:5.2f} {tau:5.2f} | {r['prompt']:7.3f} {r['seed']:7.3f} {r['both']:7.3f} |"
                  f" {r['prompt']**2:.3f} / {r['both']**2:.3f}")
