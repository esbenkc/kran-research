"""Size check of page_hazard.py's one-sided 'hazard rose' Fisher test at the full-prefill gpt-oss-20b n (Iris, defense-1120).

Same null and simulator as rise_null_sim.py (sim() copied verbatim) (each run has its own constant per-page leave chance; a mix of such runs
never gives a rising pooled hazard, Barlow, Marshall, Proschan 1963), at n = 54 entered runs and cap 80, with
constant rates near the observed pooled rate (h_all = 0.079) and mixes with a slow-leaver group.
"""
import numpy as np

from scipy.stats import fisher_exact

rng = np.random.default_rng(1120)
CAP = 80


def sim(n_runs, draw, a, reps=2000, b=None):
    rej = 0
    for _ in range(reps):
        h = draw(n_runs)
        D = np.minimum(rng.geometric(h), CAP + 1)  # page of give-up; > CAP means censored at the cap
        e_lo = n_lo = e_hi = n_hi = 0
        for d in D:
            gave = d <= CAP
            last = int(d) if gave else CAP
            pages = np.arange(1, last + 1)
            ev = np.zeros(last, int)
            if gave:
                ev[-1] = 1
            n_lo += int((pages <= a).sum()); e_lo += int(ev[pages <= a].sum())
            late = (pages > a) & (pages <= (b if b else CAP))
            n_hi += int(late.sum()); e_hi += int(ev[late].sum())
        if n_lo and n_hi:
            p = fisher_exact([[e_hi, n_hi - e_hi], [e_lo, n_lo - e_lo]], alternative="greater")[1]
            rej += p <= 0.05
    return rej / reps


mixes = {
    "constant h=0.02": lambda n: np.full(n, 0.02),
    "constant h=0.08": lambda n: np.full(n, 0.08),
    "constant h=0.2": lambda n: np.full(n, 0.2),
    "two groups 0.3/0.02 (50/50)": lambda n: np.where(rng.random(n) < 0.5, 0.3, 0.02),
    "two groups 0.8/0.05 (50/50)": lambda n: np.where(rng.random(n) < 0.5, 0.8, 0.05),
    "Beta(1,10) spread": lambda n: np.clip(rng.beta(1, 10, n), 0.005, 0.99),
    "Beta(0.5,1.5) spread": lambda n: np.clip(rng.beta(0.5, 1.5, n), 0.01, 0.99),
}
for name, draw in mixes.items():
    for a, b in ((1, None), (3, None)):
        lab = f"1-{a} vs {a + 1}+"
        print(f"n=54 {name:<30} {lab:<10}: false-rise rate {sim(54, draw, a, 1000, b):.3f}", flush=True)
