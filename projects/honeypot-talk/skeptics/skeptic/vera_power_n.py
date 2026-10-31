"""Vera: same sim as vera_power.py, for N runs per model in the prompt-on arm (N=50, 100)."""
import sys, numpy as np, vera_power as vp
for N in (50, 100):
    vp.N = N
    print(f"N={N} per model, K={vp.K}")
    for say in (0.5, 0.8, 0.95):
        for trip in (0.2, 0.5):
            for gap in (0.0, -0.3):
                r = vp.sim(say, trip, gap, nsim=300, B=300)
                print(f"  say {say:.2f} trip {trip:.2f} gap {gap:+.1f}: pooled {r['pooled']:.3f}  MH-cluster {r['mh']:.3f}")
