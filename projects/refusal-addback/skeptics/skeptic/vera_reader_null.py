"""Vera: null for the DEV-fixed reader: same rule on 100 random unit directions (sign chosen on DEV), stripped acts."""
import numpy as np
from pathlib import Path
np.seterr(all="ignore")
R = Path(__file__).resolve().parents[2] / "kran-research/projects/refusal-addback/runs"
A = np.load(R / "strip_s0/acts.npz")
rng = np.random.default_rng(7)
V = rng.standard_normal((100, 1536)); V /= np.linalg.norm(V, axis=1, keepdims=True)
for L in (12, 15, 16, 17, 19, 22):
    c = []
    for v in V:
        P = {k: A[k][:, L].astype(np.float64) @ v for k in ("dev_harmful", "dev_harmless", "test_harmful", "test_harmless")}
        if P["dev_harmful"].mean() < P["dev_harmless"].mean():
            P = {k: -x for k, x in P.items()}  # orient on DEV
        t = np.sort(P["dev_harmless"])[-4]
        c.append(((P["test_harmful"] > t).mean(), (P["test_harmless"] > t).mean()))
    c = np.array(c)
    print(f"L{L}: random-dir reader catch median {100*np.median(c[:,0]):.1f}%  p95 {100*np.quantile(c[:,0],.95):.1f}%  max {100*c[:,0].max():.1f}%; harmless median {100*np.median(c[:,1]):.1f}%")
