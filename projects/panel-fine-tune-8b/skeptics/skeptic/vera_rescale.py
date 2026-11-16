"""Vera: 'recalibration fix' (Rule B as stated by Iris #22332): re-standardize each frozen probe's score on the
scored state's own 200 refit-split TRUE statements, take the max, line = 2nd-highest max on those same 200.
Usage: python vera_rescale.py file.npz [...]  (scores[checks,2992,R], y=1 false, split)"""
import sys, numpy as np
for f in sys.argv[1:]:
    z = np.load(f); s = z["scores"]; s = s[-1] if s.ndim == 3 else s; y, sp = z["y"], z["split"]
    ref = (sp == "refit") & (y == 0)
    m, d = s[ref].mean(0), s[ref].std(0)   # population sd; ddof=1 variant below
    for ddof in (0, 1):
        d = s[ref].std(0, ddof=ddof)
        M = ((s - m) / d).max(1)
        L = np.sort(M[ref])[::-1][1]
        print(f.split('/')[-1], 'ddof', ddof, 'rescaled catch', round(float((M[(sp == 'eval') & (y == 1)] > L).mean()), 4),
              'FA', round(float((M[(sp == 'eval') & (y == 0)] > L).mean()), 4))
