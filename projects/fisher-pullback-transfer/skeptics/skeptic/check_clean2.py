"""Secondary probes filter + F-8000 properties (own code)."""
import json, numpy as np
from vera_common import load, test_set, auroc, D
L = load()
sec = json.load(open(D / "probes/secondary.json")); SW = np.load(D / "probes/secondary_weights.npz")["W"]
kept = 0; mism = 0
for m in sec:
    X, y = test_set(L, m["target"], 1)
    a = auroc(X @ SW[m["idx"]], y)
    k = a >= 0.75
    kept += k; mism += (k != m["kept"]) or abs(a - m["auroc_test"]) > 1e-4
print(f"secondary: {kept}/{len(sec)} pass AUROC>=0.75 on target test set; mismatches vs stored: {mism}")
z = np.load(D / "fisher/fisher_big.npz"); F = z["F_common"]; ev = np.linalg.eigvalsh(F)
one = np.ones(512) / np.sqrt(512)
print(f"F8000: median eig {np.median(ev):.5f}, min {ev.min():.1e}, n<1e-8 {(ev<1e-8).sum()}, ones Rayleigh/median {one@F@one/np.median(ev):.1e}, "
      f"split-half corr {np.corrcoef(z['F_common_half1'].ravel(), z['F_common_half2'].ravel())[0,1]:.3f}, "
      f"corr with F2000 {np.corrcoef(F.ravel(), L['fz']['F_common'].ravel())[0,1]:.3f}")
