"""Vera: can the v2 status rule say YES if the two-factor law is EXACTLY true?
Simulate held-out route cells where true follow = stuck * u_a, u_a estimated from 30 runs,
new cell measured with n runs. Report P(point gap <= 0.10) per cell and P(>=85% of cells hit)."""
import numpy as np

rng = np.random.default_rng(0)
STUCK = {"mini": 60 / 60, "nano": 59 / 60, "flash": 59 / 59}
U = {"mini": 0.80, "nano": 0.70, "flash": 0.30}  # 1109 mcpfetch/display_none fit, n=30


def sim(n_new, n_u=30, tools=6, reps=20000):
    hits_all = []
    per_cell = {}
    for a in U:
        p = STUCK[a] * U[a]
        # one u_a estimate per agent, shared by every held-out tool (as in the frozen rule)
        u_hat = np.repeat(rng.binomial(n_u, U[a], size=(reps, 1)) / n_u, tools, axis=1)
        meas = rng.binomial(n_new, p, size=(reps, tools)) / n_new
        hit = np.abs(meas - STUCK[a] * u_hat) <= 0.10 + 1e-9
        per_cell[a] = hit.mean()
        hits_all.append(hit)
    h = np.concatenate(hits_all, axis=1)
    frac = h.mean(axis=1)
    return per_cell, (frac >= 0.85).mean(), np.percentile(frac, [10, 50, 90])


for n in (30, 60, 100):
    for nu in (30, 1000):
        pc, p85, q = sim(n, nu)
        print(f"n_new={n:3d} n_u={nu:4d} P(hit) " +
              " ".join(f"{k}={v:.2f}" for k, v in pc.items()) +
              f" | P(>=85% of 18 cells hit)={p85:.3f} | hit-share p10/50/90={q.round(2)}")
