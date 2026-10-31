"""Vera: size of the TOST at the band edge (true within-agent gap = -0.10 / -0.15), N=25 and 100 per model."""
import numpy as np, vera_power as vp
for N in (25, 100):
    vp.N = N
    for say in (0.5, 0.8):
        for gap in (-0.10, -0.15):
            r = vp.sim(say, 0.4, gap, nsim=400, B=300)
            print(f"N={N} say {say:.2f} trip 0.40 gap {gap:+.2f}: P(pass) pooled {r['pooled']:.3f}  MH-cluster {r['mh']:.3f}")
