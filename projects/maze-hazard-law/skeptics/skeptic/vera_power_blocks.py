"""Vera: Kasper #25767 (3). Add the frozen seed clause: CONFIRMED / REJECTED for an agent also need >= 2 of 3 maze
blocks with R_b defined (G3_b >= 1 AND a KM median reached inside block b). Runs go to blocks by index mod 3."""
import sys, numpy as np
from vera_power_frozen import sim_run, CAP, AGENTS
from vera_power_final import agent_verdict, mix_g

def block_defined(runs):
    d = np.zeros(CAP + 2); e = np.zeros(CAP + 2)
    for k, c in runs:
        last = k if c == "g" else (k - 1 if c == "l" else k)
        d[1:last + 1] += 1
        if c == "g": e[k] += 1
    if e[1:4].sum() < 1: return False
    S = 1.0
    for k in range(1, CAP + 1):
        if d[k]: S *= 1 - e[k] / d[k]
        if S <= .5: return True
    return False

reps = int(sys.argv[1]) if len(sys.argv) > 1 else 500
print("agent | n | P(H & >=2 blocks)=agent counts for CONFIRMED | P(F & >=2 blocks)=counts for REJECTED | P(>=2 blocks defined)")
for name, g in AGENTS.items():
    for n in (40, 60):
        rng = np.random.default_rng(17); H = F = B = 0
        for _ in range(reps):
            runs = [sim_run(g if g else mix_g(rng), 0.005, rng) for _ in range(n)]
            v, _ = agent_verdict(runs)
            nb = sum(block_defined(runs[b::3]) for b in range(3)); ok = nb >= 2
            H += v == "H" and ok; F += v == "F" and ok; B += ok
        print(f"{name} | {n} | {H/reps:.3f} | {F/reps:.3f} | {B/reps:.3f}", flush=True)
