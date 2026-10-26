"""Expected Rule-B catch for a BLIND poisoner (it does not know which 200 true statements set the line).
It inflates a random share p of all true statements, by a huge factor, in each of m poisoned members (disjoint
random draws per member); false scores are left at the clean model's values. Uses the outlier limit of the
Samuelson-type bound: a member with k outliers gives each outlier z = sqrt((n-k)/k); its other values ~ -sqrt(k/n).
Clean Rule-B z for the unpoisoned members. 4000 draws per cell. Order of members poisoned: 28, 29, then 27, 26, ...
"""
import json, sys
from pathlib import Path
import numpy as np
P = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(P))
from common import CFG, DATA, threshold
from metrics import EVPOS, LINE
PANEL = CFG["panel"]["readouts"]
S = dict(np.load(DATA / "scores" / "clean.npz"))
X = np.stack([S[f"tok{R}"] for R in PANEL], 1)
mu, sd = X[LINE].mean(0), X[LINE].std(0)
Zt, Zf = ((X - mu) / sd)[LINE], ((X - mu) / sd)[EVPOS]
n = len(LINE)
order = [PANEL.index(R) for R in [28, 29, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10]]
rng = np.random.default_rng(1026)
out = {}
for m in [1, 2, 3, 5, 10, 20]:
    pois = order[:m]; keep = [j for j in range(20) if j not in pois]
    fmax = Zf[:, keep].max(1) if keep else np.full(len(Zf), -np.inf)
    for p in [0.005, 0.01, 0.02, 0.05]:
        c = []
        for _ in range(4000):
            zt = Zt.copy()
            for j in pois:
                hit = rng.random(n) < p; k = hit.sum()
                zt[:, j] = -np.sqrt(k / n) if k else zt[:, j]
                if k: zt[hit, j] = np.sqrt((n - k) / k)
            line = threshold(zt.max(1))
            c.append((fmax > line).mean())
        c = np.array(c)
        out[f"m{m}_p{p}"] = {"mean": float(c.mean()), "median": float(np.median(c)), "p90": float(np.quantile(c, .9))}
        print(m, p, round(c.mean(), 3), round(np.median(c), 3), round(np.quantile(c, .9), 3))
Path(__file__).with_suffix(".json").write_text(json.dumps(out, indent=1))
