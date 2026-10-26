"""Shape of the common-mode offset the in-loop evader adds (Iris). Per state: project the 20 member scores (z by the
state's own line-set true moments) on the top eigenvector of the true-score correlation; report histogram, share of
statements above the midpoint between the two modes, and whether false and true statements share the offset
(correlation of the projection between twins: same city, same template).
usage: python offset_shape.py STATE"""
import json, sys
from pathlib import Path
import numpy as np
P = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(P))
from common import CFG, DATA, rows
from metrics import EVNEG, EVPOS, LINE
PANEL = CFG["panel"]["readouts"]
_R, Y, CITY, SPLIT = rows()
st = sys.argv[1]
S = dict(np.load(DATA / "scores" / f"{st}.npz"))
X = np.stack([S[f"tok{R}"] for R in PANEL], 1)
Z = (X - X[LINE].mean(0)) / X[LINE].std(0)
TRUE = np.where(np.isin(SPLIT, ["probe", "stop", "refit"]) & (Y == 0))[0]
w, V = np.linalg.eigh(np.corrcoef(X[TRUE].T)); v = V[:, -1] * np.sign(V[:, -1].sum())
p = Z @ v / np.sqrt(len(PANEL))
h, e = np.histogram(p[EVNEG], bins=12)
print("eval-true pc1 hist:", h.tolist(), "edges", np.round(e[[0, -1]], 2).tolist())
h2, _ = np.histogram(p[EVPOS], bins=e)
print("eval-false pc1 hist (same bins):", h2.tolist())
mid = 0.5 * (np.quantile(p[EVNEG], 0.1) + np.quantile(p[EVNEG], 0.9))
print("share above mid: true %.3f false %.3f" % ((p[EVNEG] > mid).mean(), (p[EVPOS] > mid).mean()))
# twins: eval false i and eval true j with same city -> correlation of projections
ct = {c: i for i, c in zip(EVNEG, CITY[EVNEG])}
pairs = [(i, ct[c]) for i, c in zip(EVPOS, CITY[EVPOS]) if c in ct]
a = np.array([[p[i], p[j]] for i, j in pairs])
print("twin corr of pc1 offset (n=%d): %.3f" % (len(a), np.corrcoef(a.T)[0, 1]))
print("member loadings (top eigvec):", np.round(v, 2).tolist(), "eig share %.3f" % (w[-1] / w.sum()))
