"""Check two facts write adds after the cold read: negative drops on synthetic + their cosine sign;
Pythia median |plain cos| vs |whitened cos|."""
import numpy as np
from check import auroc
R = "../../kran-research/projects/probe-evasion-geometry/data/receipts/"
def load(f): return {k: (v.astype(np.float64) if v.dtype == np.float16 else v) for k, v in np.load(R + f).items()}
def cos(U, w, M): return (U @ M @ w) / np.sqrt(np.einsum("ij,jk,ik->i", U, M, U) * (w @ M @ w))
neg = []
for s in range(3):
    z = load(f"synthetic_mahalanobis_seed{s}.npz"); y = z["y_test"].astype(int)
    d = np.array([auroc(a, y) - auroc(b, y) for a, b in zip(z["probe_scores_before"].T, z["probe_scores_after"].T)])
    c = cos(z["probes"], z["w"], z["Sigma_hat"])
    neg += [(s, round(float(dd), 4), round(float(cc), 3)) for dd, cc in zip(d, c) if dd < 0]
print("synthetic negative drops (seed, drop, cos_Sigma):", neg)
cr, cw = [], []
for s in range(3):
    z = load(f"lm_mahalanobis_seed{s}.npz")
    cr += list(np.abs(cos(z["probes"], z["w"], np.eye(len(z["w"]))))); cw += list(np.abs(cos(z["probes"], z["w"], z["Sigma_hat"])))
print("Pythia median |plain cos| %.2f, |whitened cos| %.2f" % (np.median(cr), np.median(cw)))
