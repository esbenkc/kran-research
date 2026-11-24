"""Vera: recompute the Phase-0 gate verdict from the build's raw per-draw losses (data/gate/grid.json), own code.
lambda-hat = n*beta * (mean over pooled draws of both chains - L(w*)); batch-means SE over the pooled 150 draws
(5 batches of 30); PASS(setting) = |rel err| < 0.10 at ranks 1 and 8 AND |l8 - l1| >= 3 * max(SE1, SE8); scored = rep 0."""
import json, math, sys
import numpy as np
from vera_aoyagi import aoyagi
G = json.load(open(sys.argv[1]))
nb = G["n"] / math.log(G["n"])
tab = {}
for row in G["rows"]:
    L = np.array(row["losses"], dtype=float)           # chains x (burn + draws)
    assert L.shape == (G["chains"], G["burn"] + G["draws"]), L.shape
    L = L[:, G["burn"]:]                               # drop burn-in
    d = L.reshape(-1)                                  # pooled 150, chain-major
    lam = nb * (d.mean() - row["L_wstar"])
    bm = np.array([nb * (b.mean() - row["L_wstar"]) for b in d.reshape(5, -1)])
    se = bm.std(ddof=1) / math.sqrt(5)
    tab[(row["c"], row["gamma"], row["rep"], row["r"])] = (lam, se, row["lambda_hat"], row["bm_se"])
mism = sum(1 for v in tab.values() if not (np.isnan(v[0]) and np.isnan(v[2])) and abs(v[0] - v[2]) > 1e-6 * max(1, abs(v[2])))
mism_se = sum(1 for v in tab.values() if not (np.isnan(v[1]) and np.isnan(v[3])) and abs(v[1] - v[3]) > 1e-6 * max(1, abs(v[3])))
print(f"rows {len(tab)}; lambda-hat mismatches vs build {mism}; SE mismatches {mism_se}")
true = {r: float(aoyagi(32, 32, 16, r)[0]) for r in (1, 8)}
npass = 0; best = None; rep_pass = {}
for c in G["c_grid"]:
    for g in G["gamma_grid"]:
        (l1, s1, *_), (l8, s8, *_) = tab[(c, g, 0, 1)], tab[(c, g, 0, 8)]
        e1, e8 = l1 / true[1] - 1, l8 / true[8] - 1
        sep = abs(l8 - l1) / max(s1, s8)
        ok = abs(e1) < 0.1 and abs(e8) < 0.1 and sep >= 3
        npass += ok
        reps = sum(1 for rp in range(G["repeats"]) if abs(tab[(c, g, rp, 1)][0] / true[1] - 1) < 0.1 and abs(tab[(c, g, rp, 8)][0] / true[8] - 1) < 0.1)
        rep_pass[(c, g)] = reps
        w = max(abs(e1), abs(e8))
        if not np.isnan(w) and (best is None or w < best[0]): best = (w, c, g, e1, e8, sep)
        print(f"c={c:<5} g={g:<7} r1 {l1:7.1f} ({e1:+.3f}) r8 {l8:7.1f} ({e8:+.3f}) sep {sep:5.1f}  pass={ok}  reps within 10% both: {reps}/{G['repeats']}")
print(f"PASSING settings (rep 0): {npass} of {len(G['c_grid'])*len(G['gamma_grid'])}; smallest worst-case error: {best}")
print("any repeat passing the 10% part anywhere:", {k: v for k, v in rep_pass.items() if v})
