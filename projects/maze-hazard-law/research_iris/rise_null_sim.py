"""Size check for page_hazard.py's one-sided 'hazard rose' Fisher test under its null (Iris, defense-1119).

Null: every run has its own constant per-page leave chance h_i (a geometric run length), drawn from a spread of
values (a mix of runs). Such a mix gives a pooled hazard that never rises (Barlow, Marshall, Proschan 1963), but
decisions from one run are correlated, so the decision-level Fisher test might reject too often. This simulates the
false 'rise' rate at nominal 5% for run counts and mixes like ours (cap 80 pages).
"""
import numpy as np
from scipy.stats import fisher_exact

rng = np.random.default_rng(1119)
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
    "constant h=0.2": lambda n: np.full(n, 0.2),
    "two groups 0.8/0.05 (50/50)": lambda n: np.where(rng.random(n) < 0.5, 0.8, 0.05),
    "two groups 0.9/0.02 (90/10)": lambda n: np.where(rng.random(n) < 0.9, 0.9, 0.02),
    "Beta(0.5,1.5) spread": lambda n: np.clip(rng.beta(0.5, 1.5, n), 0.01, 0.99),
    "Beta(2,2) spread": lambda n: np.clip(rng.beta(2, 2, n), 0.01, 0.99),
}
for n_runs in (9, 32, 38):
    for name, draw in mixes.items():
        for a, b in ((1, 2), (1, None), (3, None)):
            lab = f"1-{a} vs {a + 1}" + ("+" if b is None else f"-{b}")
            print(f"n={n_runs:>2} {name:<30} {lab:<10}: false-rise rate {sim(n_runs, draw, a, 1000, b):.3f}")
