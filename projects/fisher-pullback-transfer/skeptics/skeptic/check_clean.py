"""Clean-data rebuild: probe AUROCs, filter, Sigma, F properties, identifiability gate (ratio -0.5)."""
import json
import numpy as np
from vera_common import load, test_set, auroc, metrics, r_P, pred_drop_exact, D

L = load()
W = L["W"]
# which split code is test? try both, compare with stored auroc_test
for code in (0, 1):
    errs = []
    for p in L["probes"]:
        X, y = test_set(L, p["name"], code)
        errs.append(abs(auroc(X @ W[p["idx"]], y) - p["auroc_test"]))
    print(f"split code {code}: max |AUROC - stored auroc_test| = {max(errs):.2e}")
TEST = 1
passed = [p["name"] for p in L["probes"] if auroc(*(lambda X, y: (X @ W[p["idx"]], y))(*test_set(L, p["name"], TEST))) >= 0.75]
print("n built", len(L["probes"]), "n pass >=0.75:", len(passed))
print("dropped:", sorted(set(p["name"] for p in L["probes"]) - set(passed)), "vs stored", L["sel"]["dropped_by_filter"])

fz = L["fz"]
F = fz["F_common"]
ev = np.linalg.eigvalsh(F)
one = np.ones(512) / np.sqrt(512)
print(f"F sym err {np.abs(F-F.T).max():.1e}; min eig {ev.min():.2e}; median {np.median(ev):.6f}; trace {ev.sum():.4f}")
print(f"ones Rayleigh / median eig = {one@F@one/np.median(ev):.2e}")
h1, h2 = fz["F_common_half1"], fz["F_common_half2"]
print(f"split-half corr of F entries {np.corrcoef(h1.ravel(), h2.ravel())[0,1]:.3f}")

Ms, lam = metrics(L)
print(f"lambda {lam:.6f}; my LW Sigma vs stored Sigma_LW max abs diff {np.abs(Ms['whitened']-fz['Sigma_LW']).max():.2e} (scale {np.abs(fz['Sigma_LW']).max():.2e})")

targets = L["sel"]["targets"]
scored = L["sel"]["scored_probes"]
preds = {k: [] for k in Ms}
for t in targets:
    w = W[L["name2i"][t]]
    for u_name in scored:
        if u_name == t:
            continue
        u = W[L["name2i"][u_name]]
        Xu, yu = test_set(L, u_name, TEST)
        for k, P in Ms.items():
            preds[k].append(pred_drop_exact(Xu, yu, u, w, -0.5, r_P(P, u, w)))
P = {k: np.array(v) for k, v in preds.items()}
print("n pairs", len(P["plain"]))
for a, b in [("plain", "whitened"), ("plain", "fisher"), ("whitened", "fisher")]:
    print(f"pearson {a}~{b} = {np.corrcoef(P[a], P[b])[0,1]:.4f}")
print({k: round(float(v.std()), 6) for k, v in P.items()}, "(SD; build uses?)")
print("stored:", json.load(open(D / "identifiability.json")))
