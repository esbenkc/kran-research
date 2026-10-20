"""Vera: per-layer clean-fit linear probes + frozen 1019 baseline, own code, from own mean features (vera_feats.py).
lin_l: StandardScaler + LogisticRegression(C=0.01, class_weight balanced), fit on the CLEAN 1400-statement pool.
Line: 2nd-highest of the 200 refit true statements of the scored state. Catch pooled over seed0-2 (1788 false).
d': twin-paired, clean pooled within-class SD on the 1192 eval statements; r_l = mean attacked d' / clean d'.
CI: city-cluster bootstrap (2000, seed 1020), clean and attacked resampled together, sigma held fixed.
    cd kran-research && uv run python ../swarm-theory-1020-megaprobe-after/skeptic/vera_lin.py
"""
import json

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from vera_core import ATT, EV_IDX, FE, HERE, POOL, PROJ, STATES, YF, auroc, city_draws, dprime, flags, pooled_catch

C = 0.01
M = {s: np.load(FE / f"{s}_mean.npy", mmap_mode="r") for s in STATES}
W = city_draws()
res = {"layers": {}}

# frozen 1019 baseline at hidden_states[14]
pr = np.load(PROJ / "data/ref1019/probe.npz")
base = {s: np.asarray(M[s][:, 14], dtype=np.float64) @ pr["w"] + float(pr["b"]) for s in STATES}
p, ci, _ = pooled_catch([flags(base[s]) for s in ATT], W)
res["base"] = {"pooled_att_catch": p, "ci": ci.tolist(), "clean_catch": pooled_catch([flags(base["clean"])]),
               "att_per_seed": [pooled_catch([flags(base[s])]) for s in ATT],
               "auroc": {s: auroc(base[s][EV_IDX], YF[EV_IDX]) for s in STATES}}
print("base", json.dumps(res["base"]), flush=True)

for l in range(1, 29):
    Xp = np.asarray(M["clean"][POOL, l], dtype=np.float64)
    sc = StandardScaler().fit(Xp)
    lr = LogisticRegression(C=C, class_weight="balanced", max_iter=5000).fit(sc.transform(Xp), YF[POOL])
    S = {s: lr.decision_function(sc.transform(np.asarray(M[s][:, l], dtype=np.float64))) for s in STATES}
    np.save(FE / f"lin_scores_L{l}.npy", np.stack([S[s] for s in STATES]))
    dp = dprime(S, S["clean"])
    clean_d = dp["clean"][0]; att_d = np.mean([dp[s][0] for s in ATT])
    # bootstrap: per-city pair sums of Delta; each eval city has exactly 2 pairs
    sig = dp["clean"][2]
    npair = W @ np.full(W.shape[1], 2.0)
    bc = (W @ dp["clean"][1]) / npair / sig
    ba = np.mean([(W @ dp[s][1]) / npair / sig for s in ATT], 0)
    br = ba / bc
    pc, cic, _ = pooled_catch([flags(S[s]) for s in ATT], W)
    res["layers"][l] = {
        "clean_d": clean_d, "clean_d_ci": np.percentile(bc, [2.5, 97.5]).tolist(),
        "att_d": att_d, "att_d_per_seed": [dp[s][0] for s in ATT],
        "r": att_d / clean_d, "r_ci": np.percentile(br, [2.5, 97.5]).tolist(),
        "scored": bool(clean_d >= 1 and np.percentile(bc, 2.5) > 0),
        "clean_catch": pooled_catch([flags(S["clean"])]), "att_catch": pc, "att_catch_ci": cic.tolist(),
        "auroc": {s: auroc(S[s][EV_IDX], YF[EV_IDX]) for s in STATES},
    }
    if l == 14:
        res["r14_boot"] = br.tolist()
    np.save(FE / f"r_boot_L{l}.npy", br)
    L = res["layers"][l]
    print(f"L{l:2d} clean d' {clean_d:5.2f} att d' {att_d:5.2f} r {L['r']:.2f} [{L['r_ci'][0]:.2f},{L['r_ci'][1]:.2f}] "
          f"scored {L['scored']} catch clean {L['clean_catch']:.3f} att {pc:.3f} [{cic[0]:.3f},{cic[1]:.3f}] "
          f"auroc c/s0 {L['auroc']['clean']:.3f}/{L['auroc']['seed0']:.3f}", flush=True)

json.dump(res, open(HERE / "vera_lin.json", "w"), indent=1, default=float)
