"""Vera check 2: how arbitrary is the bar? Noise-level sweep, equivalent noise level of the readable rule,
robust/fragile split with my own noise draws. Dev seeds 910-913 only.
Run: cd swarm-algzoo-432 && .venv/bin/python skeptic/vera_checks2.py -> skeptic/vera_checks2.json
"""
import json, sys
import numpy as np
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from vera_core import load, pred, sample, aug_order, HERE

W = load()
X = sample(910, 400_000)
y = pred(X, W)
pos, _ = aug_order(X)
phant = pos[:, 1]
true2 = np.argsort(-X, 1)[:, 1]
out = {"n": len(y), "phantom2nd": float((phant == y).mean()), "null": float((true2 == y).mean())}

rng = np.random.default_rng(911)
sweep = {}
for lvl in (0.0003, 0.001, 0.002, 0.003, 0.005, 0.01, 0.02):
    a = [float((pred(X, tuple(w * (1 + lvl * rng.standard_normal(w.shape)) for w in W)) == y).mean()) for _ in range(10)]
    sweep[lvl] = {"mean": float(np.mean(a)), "sd": float(np.std(a, ddof=1))}
out["noise_sweep_10draws"] = sweep

# robust / fragile with my own 20 draws at 0.1%
rng = np.random.default_rng(912)
stable = np.ones(len(y), bool)
for _ in range(20):
    stable &= pred(X, tuple(w * (1 + 0.001 * rng.standard_normal(w.shape)) for w in W)) == y
rng = np.random.default_rng(913)
fresh = [pred(X, tuple(w * (1 + 0.001 * rng.standard_normal(w.shape)) for w in W)) == y for _ in range(10)]
fresh = np.mean(fresh, 0)
out["fragile_share"] = float(1 - stable.mean())
for name, p in (("phantom2nd", phant), ("null", true2)):
    out[f"{name}_robust"] = float((p == y)[stable].mean())
    out[f"{name}_fragile"] = float((p == y)[~stable].mean())
out["noisycopy_robust"] = float(fresh[stable].mean())
out["noisycopy_fragile"] = float(fresh[~stable].mean())
# what a rule that is perfect on robust inputs and noisy-copy-like on fragile inputs scores
out["perfect_robust_plus_noisy_fragile"] = float(stable.mean() + (~stable).mean() * fresh[~stable].mean())
json.dump(out, open(f"{HERE}/skeptic/vera_checks2.json", "w"), indent=1)
print(json.dumps(out, indent=1))
