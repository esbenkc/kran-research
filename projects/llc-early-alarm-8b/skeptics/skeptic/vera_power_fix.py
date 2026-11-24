"""Vera pre-freeze, part 2: the same null as vera_power.py, with candidate fixes to the v2 detector.
fix A: LOO z uses the SD of a difference of two runs: z = (a-b)/(sqrt(2)*sigma).
fix B: alarm needs z >= 3 at 2 consecutive window checkpoints (the same pair rule as T).
Reports P(control clause holds) and P(main false alarm before T-8) under pure noise."""
import numpy as np
rng = np.random.default_rng(11242)
def ar1(n, m, rho):
    e = rng.standard_normal((m, n)); x = np.empty_like(e); x[:, 0] = e[:, 0]
    for t in range(1, n): x[:, t] = rho*x[:, t-1] + np.sqrt(1-rho**2)*e[:, t]
    return x
def first2(z):  # z >= 3 at two consecutive checkpoints
    h = z >= 3; return (h[:, 1:] & h[:, :-1]).any(1)
def sim(n_ctrl, n_pre, rho, loo_sqrt2, pair, reps=20000):
    a = ar1(n_ctrl, reps, rho); b = ar1(n_ctrl, reps, rho); m = (a+b)/2
    sig = np.sqrt((((a-m)**2 + (b-m)**2).sum(1))/n_ctrl)
    za = (a-b)/(sig[:, None]*(np.sqrt(2) if loo_sqrt2 else 1))
    hit = (lambda z: first2(z)) if pair else (lambda z: (z >= 3).any(1))
    ctrl_ok = ~(hit(za) | hit(-za))
    main = ar1(n_pre, reps, rho); zm = (main - m[:, :n_pre])/sig[:, None]
    return ctrl_ok.mean(), hit(zm).mean()
print("fix        n_ctrl rho | P(ctrl ok) P(main FA, 15 pre-ckpts)")
for name, s2, pr in [("v2", 0, 0), ("A sqrt2", 1, 0), ("B pair", 0, 1), ("A+B", 1, 1)]:
    for n_ctrl in (70, 110):
        for rho in (0.0, 0.8):
            c, f = sim(n_ctrl, 15, rho, s2, pr)
            print(f"{name:9s} {n_ctrl:5d} {rho:3.1f} | {c:.3f} {f:.3f}")
