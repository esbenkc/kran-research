"""Vera: power of the FINAL frozen rule (PREREG section 4 + Felix #25754 v2 + #25761 rule B), before data.
Per run: entry at call 1 (K = 80), page k = 1..80; give-up g(k), else lost l (decisions 1..k-1), else on; cap = censored.
STATUS per agent: U (untested) / F (fails) / H (holds). P1 (prediction) uses Pearson p > 0.05 and the plain 2x check."""
import math, sys, numpy as np
from scipy import stats
from vera_power_frozen import sim_run, CAP, BINS, AGENTS
from vera_power_weibull import weib_lr, pred_of

def agent_verdict(runs):
    d = np.zeros(CAP + 2); e = np.zeros(CAP + 2)
    for k, c in runs:
        last = k if c == "g" else (k - 1 if c == "l" else k)
        d[1:last + 1] += 1
        if c == "g": e[k] += 1
    G = e.sum(); lost = sum(c == "l" for _, c in runs)
    e3, d3 = e[1:4].sum(), d[1:4].sum(); h3 = e3 / d3
    if not (e3 >= 1 and G >= 10 and G >= 0.5 * (G + lost)): return "U", False
    O = np.array([e[a:b + 1].sum() for a, b in BINS]); Dn = np.array([d[a:b + 1].sum() for a, b in BINS]); kp = Dn > 0
    pc = stats.chi2_contingency(np.vstack([O[kp], Dn[kp] - O[kp]]), correction=False)[1] if kp.sum() > 1 else 1.0
    S, med = 1.0, None
    for k in range(1, CAP + 1):
        if d[k]: S *= 1 - e[k] / d[k]
        if med is None and S <= .5: med = k
    two = med is not None and .5 <= pred_of(h3) / med <= 2
    p1 = med is not None and pc > .05 and two
    if pc <= .025 or weib_lr(d[1:CAP + 1], e[1:CAP + 1]) <= .025: return "F", p1
    if med is None: return "U", p1
    if two: return "H", p1
    lo = stats.chi2.ppf(.025, 2 * e3) / 2 / d3; hi = stats.chi2.ppf(.975, 2 * e3 + 2) / 2 / d3
    return ("U" if (pred_of(hi) <= 2 * med and pred_of(lo) >= .5 * med) else "F"), p1

def mix_g(rng):
    return (lambda k: 0.3) if rng.random() < 0.3 else (lambda k: 0.005)

if __name__ == "__main__":
    reps = int(sys.argv[1]) if len(sys.argv) > 1 else 500
    print("agent | n | P(H) P(F) P(U) P(P1) share_at_cap")
    for name, g in AGENTS.items():
        for n in (40, 60):
            rng = np.random.default_rng(13); c = {"H": 0, "F": 0, "U": 0}; p1 = 0; cap = 0
            for _ in range(reps):
                runs = [sim_run(g if g else mix_g(rng), 0.005, rng) for _ in range(n)]
                v, p = agent_verdict(runs); c[v] += 1; p1 += p
                cap += sum(x == "c" for _, x in runs) / n
            print(f"{name} | {n} | {c['H']/reps:.3f} {c['F']/reps:.3f} {c['U']/reps:.3f} {p1/reps:.3f} {cap/reps:.2f}", flush=True)
