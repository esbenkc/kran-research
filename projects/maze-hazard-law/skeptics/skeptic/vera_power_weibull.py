"""Vera: Tove's variant. Law fails if chi-square (df=B'-1) p<=alpha_c OR discrete-Weibull LR (shape k vs 1) p<=alpha_w,
plus v1 (median censored -> untested) and v2 (2x miss is a fail only if the whole exact 95% pilot interval misses).
Discrete Weibull hazard: h(k) = 1 - exp(-((k/lam)^b - ((k-1)/lam)^b)); b = 1 is geometric."""
import math, sys, numpy as np
from scipy import stats, optimize
from vera_power_frozen import sim_run, CAP, BINS, AGENTS

K = np.arange(1, CAP + 1)

def ll_weib(params, d, e):
    lam, b = math.exp(params[0]), math.exp(params[1])
    H = (K / lam) ** b - ((K - 1) / lam) ** b
    H = np.clip(H, 1e-12, 50)
    return -(e * np.log1p(-np.exp(-H)) - (d - e) * H).sum()

def weib_lr(d, e):
    G = e.sum(); D = d.sum(); h = G / D
    ll0 = (G * math.log(h) + (D - G) * math.log1p(-h)) if 0 < h < 1 else 0.0
    best = None
    for b0 in (0.6, 1.0, 1.6):
        lam0 = (1 / -math.log1p(-h)) if h > 0 else 100
        r = optimize.minimize(ll_weib, [math.log(lam0), math.log(b0)], args=(d, e), method="Nelder-Mead",
                              options={"xatol": 1e-4, "fatol": 1e-6, "maxiter": 400})
        if best is None or r.fun < best.fun: best = r
    lr = max(0.0, 2 * (-best.fun - ll0))
    return 1 - stats.chi2.cdf(lr, 1)

def pred_of(h3): return CAP if h3 <= 0 else min(math.log(0.5) / math.log(1 - min(h3, 0.999)), CAP)

def verdict(runs, ac, aw):
    d = np.zeros(CAP + 2); e = np.zeros(CAP + 2)
    for k, c in runs:
        d[1:k + 1] += 1
        if c == "g": e[k] += 1
    G = e.sum(); lost = sum(c == "l" for _, c in runs)
    e3 = e[1:4].sum(); d3 = d[1:4].sum(); h3 = e3 / d3
    if not (e3 >= 1 and G >= 10 and G >= 0.5 * (G + lost)): return "U"
    h = G / d[1:CAP + 1].sum()
    O = np.array([e[a:b + 1].sum() for a, b in BINS]); Dn = np.array([d[a:b + 1].sum() for a, b in BINS])
    keep = Dn > 0
    table = np.vstack([O[keep], Dn[keep] - O[keep]])
    pc = stats.chi2_contingency(table, correction=False)[1] if keep.sum() > 1 else 1.0
    if pc <= ac: return "F"
    if aw is not None and weib_lr(d[1:CAP + 1], e[1:CAP + 1]) <= aw: return "F"
    S, med = 1.0, None
    for k in range(1, CAP + 1):
        if d[k]: S *= 1 - e[k] / d[k]
        if med is None and S <= .5: med = k
    if med is None: return "U"
    if .5 <= pred_of(h3) / med <= 2: return "H"
    lo = stats.chi2.ppf(.025, 2 * e3) / 2 / d3; hi = stats.chi2.ppf(.975, 2 * e3 + 2) / 2 / d3
    return "U" if (pred_of(hi) <= 2 * med and pred_of(lo) >= .5 * med) else "F"

reps = int(sys.argv[1]) if len(sys.argv) > 1 else 500
rules = {"v2 chi.05": (.05, None), "chi.025|W.025": (.025, .025)}
print("agent | n | " + " | ".join(f"{r}: P(fail) P(holds)" for r in rules))
for name, g in AGENTS.items():
    if g is None or name == "geo h=.005" or name == "geo h=.05": continue
    for n in (40, 60):
        rng = np.random.default_rng(9)
        sims = [[sim_run(g, 0.005, rng) for _ in range(n)] for _ in range(reps)]
        out = []
        for r, (ac, aw) in rules.items():
            v = [verdict(s, ac, aw) for s in sims]
            out.append(f"{v.count('F')/reps:.3f} {v.count('H')/reps:.3f}")
        print(name, n, " | ".join(out), flush=True)
