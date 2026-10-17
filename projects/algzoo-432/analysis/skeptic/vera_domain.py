"""Vera check of the 'certified domain' claim (Iris T21, #1967/#1971): C1 agrees 0.98734 on the 82% of inputs
with top-3 gaps > 0.05 rms(x), 'at noise-floor level'. Is the right comparison the GLOBAL floor, or the
noisy copy's agreement on the SAME domain? Own code, dev seed 930 (4e5), noise rng 931, 10 draws at 0.1%.
Run: cd swarm-algzoo-432 && .venv/bin/python skeptic/vera_domain.py -> skeptic/vera_domain.json
"""
import json, sys
import numpy as np
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from vera_core import load, pred, sample, aug_order, HERE

W = load()
X = sample(930, 400_000)
y = pred(X, W)
pos, _ = aug_order(X)
A = -np.sort(-np.concatenate([np.zeros((len(X), 1)), X], 1), 1)
rms = np.sqrt((X ** 2).mean(1))
c1 = pos[:, 1] == y
rng = np.random.default_rng(931)
noisy = np.mean([pred(X, tuple(w * (1 + 0.001 * rng.standard_normal(w.shape)) for w in W)) == y for _ in range(10)], 0)
out = {}
for tau in (0.05, 0.1, 0.2):
    D = ((A[:, 0] - A[:, 1]) > tau * rms) & ((A[:, 1] - A[:, 2]) > tau * rms)
    out[tau] = {"coverage": float(D.mean()), "C1_in": float(c1[D].mean()), "noisycopy_in": float(noisy[D].mean()),
                "C1_out": float(c1[~D].mean()), "noisycopy_out": float(noisy[~D].mean())}
json.dump(out, open(f"{HERE}/skeptic/vera_domain.json", "w"), indent=1)
print(json.dumps(out, indent=1))
