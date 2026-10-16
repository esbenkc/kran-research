"""skeptic3: same primary analysis as recompute.py, but from the compact public receipts
(data/receipts/*.npz: probe scores on clean/encoded test, Sigma_hat, refit scores). Use this one if
you only have the GitHub branch. Run from kran-research/:
  uv run python projects/probe-evasion-geometry/skeptics/skeptic3/recompute_receipts.py
"""
import json, pathlib
import numpy as np
from scipy.stats import norm
from sklearn.metrics import roc_auc_score

ROOT = pathlib.Path("projects/probe-evasion-geometry/data/receipts")
OUT = pathlib.Path(__file__).parent / "out"; OUT.mkdir(exist_ok=True)
rng = np.random.default_rng(20261016)


def dprime(a, n): return np.sqrt(2) * norm.ppf(np.clip(a, 1 / (2 * n), 1 - 1 / (2 * n)))
def r2p(p, m): return float(np.corrcoef(p, m)[0, 1] ** 2)
def r2id(p, m): return float(1 - ((m - p) ** 2).sum() / ((m - m.mean()) ** 2).sum())


rows = []
for f in sorted(ROOT.glob("*.npz")):
    z = np.load(f, allow_pickle=True)
    if "probe_scores_before" not in z.files:
        print(f"{f.name:42s} skipped: no scores in compact receipt"); continue
    y = z["y_test"].astype(int); n = min((y == 0).sum(), (y == 1).sum())
    U, w, S = z["probes"], z["w"], z["Sigma_hat"]
    b = np.array([roc_auc_score(y, s) for s in z["probe_scores_before"].T])
    a = np.array([roc_auc_score(y, s) for s in z["probe_scores_after"].T])
    drop = b - a
    dw0 = dprime(roc_auc_score(y, z["w_scores_before"]), n); dw1 = dprime(roc_auc_score(y, z["w_scores_after"]), n)
    k = 1 - dw1 / dw0; du = dprime(b, n)
    quad = np.einsum("ij,jk,ik->i", U, S, U)
    cw = (U @ S @ w) / np.sqrt(quad * (w @ S @ w))
    craw = (U @ w) / (np.linalg.norm(U, axis=1) * np.linalg.norm(w))
    pred = lambda c: b - norm.cdf((du - k * dw0 * c) / np.sqrt(2))
    eu = craw * np.linalg.norm(U, axis=1) / np.sqrt(quad) * np.sqrt(w @ S @ w) / np.linalg.norm(w)
    perm = np.array([r2p(pred(rng.permutation(cw)), drop) for _ in range(1000)])
    rows.append(dict(file=f.name, k=float(k), r2_whitened=r2p(pred(cw), drop), r2_raw=r2p(pred(craw), drop),
                     r2id_whitened=r2id(pred(cw), drop), r2_euclid_exact=r2p(pred(eu), drop),
                     r2_cos_alone=r2p(cw, drop), null_perm_q95=float(np.quantile(perm, 0.95)),
                     refit_auroc=float(roc_auc_score(y, z["refit_scores"]))))
    r = rows[-1]
    print(f"{f.name:42s} whit {r['r2_whitened']:.3f} raw {r['r2_raw']:.3f} noFit {r['r2id_whitened']:.3f} "
          f"euclid {r['r2_euclid_exact']:.3f} cosAlone {r['r2_cos_alone']:.3f} permQ95 {r['null_perm_q95']:.3f} refit {r['refit_auroc']:.3f}")
(OUT / "recompute_receipts.json").write_text(json.dumps(rows, indent=1))
