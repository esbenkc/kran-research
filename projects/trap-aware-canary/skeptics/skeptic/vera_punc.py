"""P(status) for the frozen title rule, by simulation. Vera (skeptic).
Rule: CONFIRMED = pooled (agent-equal-weight) 95% CI-low >= 0.50 AND every agent point >= 0.30;
REJECTED = CI-high < 0.50; else UNCONFIRMED. CI = stratified percentile bootstrap (runs resampled within agent)."""
import numpy as np
rng = np.random.default_rng(0)
def status(k, n, B=2000):
    p = k / n
    boot = rng.binomial(n, p[:, None], size=(3, B)) / n
    m = boot.mean(0); lo, hi = np.quantile(m, [0.025, 0.975])
    if lo >= 0.5 and (p >= 0.3).all(): return "C"
    if hi < 0.5: return "R"
    return "U"
for n in (40, 60, 90):
    for label, rates in [("equal .55", [.55]*3), ("equal .65", [.65]*3), ("equal .75", [.75]*3),
                         ("het .65 (.95,.75,.25)", [.95, .75, .25]), ("het .65 (.95,.6,.4)", [.95, .6, .4]),
                         ("equal .45", [.45]*3), ("equal .35", [.35]*3)]:
        r = np.array(rates); out = {"C": 0, "R": 0, "U": 0}
        for _ in range(1000): out[status(rng.binomial(n, r), n, 1000)] += 1
        print(f"n/agent={n:3d} {label:24s} C={out['C']/10:5.1f}% U={out['U']/10:5.1f}% R={out['R']/10:5.1f}%")
