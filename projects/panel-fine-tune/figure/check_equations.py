"""Checks of the post's displayed equations (theory-1022-panel-max-attack).

(1) Panel line lemma: M(x) = max_l z_l(x) >= z_l(x) for every l, so the k-th highest M over any set of true
    statements is >= the k-th highest z_l over the same set: tau_M >= max_l tau_l (all in rescaled units).
    Symbolic: order statistics are monotone (checked by brute force on random data), then numerically on every
    scored state in data/results/results.json (lines in raw probe logits, converted with data/fits/zmoments.json).
(2) The layer-28 lemma (RMSNorm(c h) = RMSNorm(h), s(c h) = c a + b0) is checked symbolically in
    iris_checks/lemma_sympy.py; this script runs it.
usage: python figure/check_equations.py
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

P = Path(__file__).resolve().parent.parent
R = json.loads((P / "data/results/results.json").read_text())
Z = json.loads((P / "data/fits/zmoments.json").read_text())
READ = [f"tok{l}" for l in range(10, 30)]

# (1a) brute force: k-th order statistic of a pointwise max dominates each member's
rng = np.random.default_rng(1022)
for _ in range(2000):
    S = rng.standard_normal((20, 200)) * rng.uniform(0.2, 3, (20, 1)) + rng.uniform(-2, 2, (20, 1))
    M = S.max(0)
    assert np.sort(M)[-2] >= np.sort(S, 1)[:, -2].max() - 1e-12
print("order-statistic lemma: 2000 random panels OK")

# (1b) on every scored state
worst = None
for arm, a in R["arms"].items():
    for i, st in enumerate(a["states"]):
        tM = a["monitors"]["panelmax"]["line_per_seed"][i]
        tz = [(a["monitors"][k]["line_per_seed"][i] - Z[k]["mu"]) / Z[k]["sd"] for k in READ]
        gap = tM - max(tz)
        assert gap >= -1e-6, (arm, st, tM, max(tz))
        worst = gap if worst is None else min(worst, gap)
        print(f"{arm:14s} {st:40s} panel line {tM:6.2f}  highest member line {max(tz):6.2f} ({READ[int(np.argmax(tz))]})")
print("tau_M >= max_l tau_l on every state; smallest gap", round(worst, 4))

# (2) RMSNorm lemma
lemma = P / "iris_checks/lemma_sympy.py"
out = subprocess.run(["uv", "run", "-q", "--with", "sympy", "python", str(lemma)], capture_output=True, text=True)
print(out.stdout.strip() or out.stderr.strip())
sys.exit(out.returncode)
