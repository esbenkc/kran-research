# Simulate the frozen title rule (Felix/Tove/Vera) under two worlds, before any run.
# HIT: |measured - predicted| <= 0.10 (point). MISS: 95% Newcombe interval of (measured - predicted),
# with the parent's frozen rate sampling error, lies fully outside [-0.10, +0.10].
# Yes = MISS <= 1 and HIT share >= 0.80; No = MISS >= 2; else Not settled. Counted per distinct context x agent.
import numpy as np, sys
from math import sqrt
rng = np.random.default_rng(int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 0)
Z = 1.96
def wilson(k, n):
    p = k / n; den = 1 + Z*Z/n; c = (p + Z*Z/(2*n)) / den
    h = Z*sqrt(p*(1-p)/n + Z*Z/(4*n*n)) / den
    return c - h, c + h
def newcombe(k1, n1, k0, n0, f):
    # diff = k1/n1 - f*k0/n0 ; f = crawl exposure (fixed, model-free)
    p1, p0 = k1/n1, f*k0/n0
    l1, u1 = wilson(k1, n1); l0, u0 = wilson(k0, n0); l0, u0 = f*l0, f*u0
    d = p1 - p0
    lo = d - sqrt((p1-l1)**2 + (u0-p0)**2); hi = d + sqrt((u1-p1)**2 + (p0-l0)**2)
    return lo, hi
PARENT = {"nano": {"fit": (60, 60), "off": (2, 60)},
          "fl":   {"fit": (60, 60), "off": (8, 60)},
          "mini": {"fit": (59, 60), "off": (44, 60)}}
def combos(n_fit, n_off, n_ctx_fit=5, trunc_f=0.2):
    out = []  # (agent, anchor, family, exposure, n)
    for a in PARENT:
        for c in range(n_ctx_fit): out.append((a, "fit", f"ctx{c}", 1.0, n_fit))
        out.append((a, "fit", "trunc", trunc_f, n_fit))
        for fam in ("raw", "md", "indexed"): out.append((a, "off", fam, 1.0, n_off))
    return out
def run(world, n_fit, n_off, reps=4000, shift=0.20, shift_fam=("indexed", "ctx0")):
    res = {"Yes": 0, "No": 0, "NS": 0}
    cs = combos(n_fit, n_off)
    for _ in range(reps):
        miss = hit = 0
        for a, anc, fam, f, n in cs:
            k0, n0 = PARENT[a][anc]
            true_parent = k0 / n0
            frozen_k = rng.binomial(n0, true_parent)  # parent sample redrawn: the frozen rate is itself noisy
            t = f * true_parent
            if world == "shift" and fam in shift_fam:
                t = t + shift if t + shift <= 1 and true_parent < 0.5 else t - shift
            t = min(max(t, 0), 1)
            k1 = rng.binomial(n, t)
            pred = f * frozen_k / n0
            if abs(k1/n - pred) <= 0.10 + 1e-9: hit += 1
            lo, hi = newcombe(k1, n, frozen_k, n0, f)
            if lo > 0.10 or hi < -0.10: miss += 1
        share = hit / len(cs)
        if miss >= 2: res["No"] += 1
        elif miss <= 1 and share >= 0.80: res["Yes"] += 1
        else: res["NS"] += 1
    return {k: round(v/reps, 3) for k, v in res.items()}, len(cs)
if __name__ == "__main__":
  for n_fit, n_off in ((30, 30), (30, 40), (30, 60)):
      for world in ("exact", "shift"):
          r, K = run(world, n_fit, n_off)
          print(f"n_fit={n_fit} n_off={n_off} combos={K} world={world:5}: {r}")
