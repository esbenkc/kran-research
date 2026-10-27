"""Vera: did the fine-tune shrink or turn the false-minus-true mean gap? Eval split only.
Metric fixed from the CLEAN model: within-class covariance of the clean fit pool (attack+probe+stop), Ledoit-Wolf (sklearn).
Per readout: whitened gap ratio |W d_state| / |W d_clean|, raw ratio, cos(W d_state, W d_clean) and raw cos. Medians over readouts."""
import json, numpy as np
from sklearn.covariance import LedoitWolf
import vera_state as V
FIT = np.isin(V.sp, ["attack", "probe", "stop"])
clean = np.load(V.P + "data/ctok/clean_1026.npy", mmap_mode="r")
states = {"R1": "refit50__lam10000__seed0", "R3": "refit50__lam10000__seed1", "R4": "refit50__lam10000__seed2",
          "R2": "refit50__lam100000__seed0", "gate": "leace_gate", "registered": "leace_registered", "G": "meangap__lam10000__seed0"}
Ws, dc = [], []
for j in range(20):
    X = np.asarray(clean[:, j, :], np.float64)
    Xf = X[FIT]; yf = V.y[FIT]
    Xw = np.concatenate([Xf[yf == k] - Xf[yf == k].mean(0) for k in (0, 1)])
    S = LedoitWolf(assume_centered=True).fit(Xw).covariance_
    w, U = np.linalg.eigh(S); Ws.append((U / np.sqrt(w)) @ U.T)
    dc.append(X[V.EV & (V.y == 1)].mean(0) - X[V.EV & (V.y == 0)].mean(0))
out = {}
for nm, f in states.items():
    A = np.load(V.P + f"data/ctok/{f}.npy", mmap_mode="r"); rows = []
    for j in range(20):
        X = np.asarray(A[:, j, :], np.float64)
        d = X[V.EV & (V.y == 1)].mean(0) - X[V.EV & (V.y == 0)].mean(0)
        a, b = Ws[j] @ d, Ws[j] @ dc[j]
        rows.append([np.linalg.norm(a) / np.linalg.norm(b), np.linalg.norm(d) / np.linalg.norm(dc[j]),
                     a @ b / np.linalg.norm(a) / np.linalg.norm(b), d @ dc[j] / np.linalg.norm(d) / np.linalg.norm(dc[j])])
    r = np.array(rows)
    out[nm] = dict(whitened_ratio_median=float(np.median(r[:, 0])), whitened_ratio_range=[float(r[:, 0].min()), float(r[:, 0].max())],
                   raw_ratio_median=float(np.median(r[:, 1])), cos_whitened_median=float(np.median(r[:, 2])),
                   cos_whitened_min=float(r[:, 2].min()), cos_raw_median=float(np.median(r[:, 3])),
                   angle_whitened_median_deg=float(np.degrees(np.arccos(np.median(r[:, 2])))))
    print(nm, {k: (round(v, 3) if isinstance(v, float) else [round(x, 3) for x in v]) for k, v in out[nm].items()})
json.dump(out, open("vera_gap.json", "w"), indent=1)
