"""Compare Vera's own F (1000 own windows) with build's F-8000; rerun the verdict predictor with my F."""
import json, numpy as np
from vera_common import D, load, metrics, pred_drop_exact, r2, r_P, test_set
mine = np.load("vera_F.npz")["F"]; F8 = np.load(D / "fisher/fisher_big.npz")["F_common"]
ev_m, ev_8 = np.linalg.eigvalsh(mine), np.linalg.eigvalsh(F8)
print(f"trace mine {ev_m.sum():.3f} vs F8000 {ev_8.sum():.3f}; median eig {np.median(ev_m):.5f} vs {np.median(ev_8):.5f}")
print(f"entrywise corr {np.corrcoef(mine.ravel(), F8.ravel())[0,1]:.3f}")
one = np.ones(512)/np.sqrt(512); print(f"my ones Rayleigh/median {one@mine@one/np.median(ev_m):.1e}")
U8 = np.linalg.eigh(F8)[1][:, -20:]; Um = np.linalg.eigh(mine)[1][:, -20:]
print(f"top-20 eigvec subspace overlap (mean cos^2) {np.linalg.norm(U8.T@Um)**2/20:.3f}")
RB = json.load(open("rebuild/rebuild.json")); L = load(); W = L["W"]
ratios = {}
for c in RB["checks"]: ratios.setdefault(c["target"], []).append(c["my_delta_w"]/c["my_Gw"])
Ms, lam = metrics(L, 1.0, mine)
pred, meas = [], []
for p in RB["primary_pairs"]:
    w, u = W[L["name2i"][p["target"]]], W[L["name2i"][p["probe"]]]
    Xu, yu = test_set(L, p["probe"], 1); r = r_P(Ms["fisher"], u, w)
    pred.append(np.mean([pred_drop_exact(Xu, yu, u, w, q, r) for q in ratios[p["target"]]])); meas.append(p["meas"])
print(f"R2_F with MY F (c=1, lambda {lam:.5f}) = {r2(pred, meas):.4f}  (build-F rebuild: {RB['summary']['primary']['R2']['fisher']:.4f})")
