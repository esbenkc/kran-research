"""How close is err(raw, B)'s 95% interval to the REJECTED (a) boundary (hi < -1/10)? Re-run analysis.bootstrap with
other seeds (reported only; the frozen seed is 20261123). Also a numpy-RNG version of the same resampling.

    uv run python research_checks/boot_seed_sensitivity.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import analysis as an  # noqa: E402

res, rows = an.analyse(an.OUT / "A" / "gpt-oss-20b", an.pick_b())
cell = {}
for r in rows:
    if r["hiding"] == "free":
        cell.setdefault((r["slot"], r["format"]), []).append(r)
cell = {k: sorted(v, key=lambda r: r["i"])[:res["floor"][k[0]]] for k, v in cell.items()}
his = []
for seed in range(20261123, 20261123 + 20):
    ci = an.bootstrap(cell, seed=seed)
    his.append(float(ci["err", "B", "raw"][1]))
print("python random, 20 seeds: err(raw,B) hi:", [round(h, 4) for h in his])
print("  seeds with hi < -0.1 (REJECTED a):", sum(h < -0.1 for h in his), "of", len(his))
# numpy RNG, same estimator (runs resampled within each cell), 2000 draws
rng = np.random.default_rng(20261123)
fol = {k: np.array([r["follow"] for r in v]) for k, v in cell.items()}
rea = {k: np.array([r["reach"] for r in v]) for k, v in cell.items()}
errs = []
for _ in range(2000):
    d = {k: rng.integers(0, len(v), len(v)) for k, v in fol.items()}
    rr = lambda k: rea[k][d[k]]
    ff = lambda k: fol[k][d[k]]
    fB = (ff(("B", "md")) & rr(("B", "md"))).sum() + (ff(("B", "anchor")) & rr(("B", "anchor"))).sum()
    fB /= rr(("B", "md")).sum() + rr(("B", "anchor")).sum()
    errs.append(ff(("B", "raw")).mean() - rr(("A", "raw")).mean() * fB)
print("numpy default_rng(20261123), 2000 draws: err(raw,B) 95% =", np.percentile(errs, [2.5, 97.5]).round(4))
