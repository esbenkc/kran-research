"""Companion to vera_isoX.py: hold gamma/nbeta = 0.1 fixed and move eps (so X) over 1.5 decades; also a 4x longer chain."""
import json, sys
from vera_isoX import run
out = []
nbeta, gamma = 1000.0, 100.0
for r in (1, 8):
    for eps in (1e-5, 3e-5, 1e-4, 3e-4):
        for steps in (4000, 16000):
            o = run(r, eps, gamma, nbeta, steps, seed=1); out.append(o)
            print(f"isoG r={r} eps={eps:.0e} X={o['X']:.1e} steps={steps} rel={o['rel_err']:+.3f} chains={[round(c) for c in o['chains']]}", flush=True)
json.dump(out, open("vera_isoG.json", "w"), indent=1)
