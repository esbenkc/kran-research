"""Vera: recompute build's headline numbers from build's saved per-statement scores (data/scores.npz) with my own line
rule, catch, city bootstrap (5 bootstrap seeds) and positive control; compare to results.json; compare my own
independent monitors (vera_lin.py / vera_mega.py) to build's.
    cd swarm-theory-1020-megaprobe-after/skeptic && python3 vera_rebuild.py
"""
import json

import numpy as np

from vera_core import ATT, CITY, EV_IDX, HERE, PROJ, STATES, YF, auroc, city_draws, flags, pooled_catch

F = np.load(PROJ / "data/scores.npz")
R = json.load(open(PROJ / "data/results/results.json"))
assert (F["y"] == YF).all() or (F["y"] == 1 - YF).all()
assert (F["city"] == CITY).all()
print("build y convention: 1 = false" if (F["y"] == YF).all() else "build y: 1 = true")
out = {}
for mon in ["base", "mlp14", "mega", "mega_n200", "mega_n400", "mega_n800", "mega_n1400", "lin14", "lin18", "lin8"]:
    S = {s: F[f"{s}__{mon}"].astype(np.float64) for s in STATES}
    row = {"att": pooled_catch([flags(S[s]) for s in ATT]), "clean": pooled_catch([flags(S["clean"])]),
           "per_seed": [pooled_catch([flags(S[s])]) for s in ATT],
           "fpr_eval": [float(flags(S[s])[EV_IDX][YF[EV_IDX] == 0].mean()) for s in ATT],
           "auroc": [auroc(S[s][EV_IDX], YF[EV_IDX]) for s in STATES]}
    cis = []
    for bs in [1020, 1, 2, 3, 4]:
        _, ci, _ = pooled_catch([flags(S[s]) for s in ATT], city_draws(2000, bs)); cis.append(ci.tolist())
    row["att_ci_by_bootseed"] = cis
    out[mon] = row
    print(f"{mon:11s} att {row['att']:.4f} CI(seed1020) [{cis[0][0]:.4f},{cis[0][1]:.4f}] lowerCI range "
          f"{min(c[0] for c in cis):.4f}-{max(c[0] for c in cis):.4f} clean {row['clean']:.4f} per-seed "
          f"{np.round(row['per_seed'], 4).tolist()} fpr {np.round(row['fpr_eval'], 4).tolist()} auroc {np.round(row['auroc'], 4).tolist()}")
# positive control: clean mega eval AUROC vs the 5 shuffled-label fits
nulls = [auroc(F[f"clean__mega_null{k}"][EV_IDX], YF[EV_IDX]) for k in range(1, 6)]
print("positive control: clean mega AUROC", round(out["mega"]["auroc"][0], 5), "> max null", round(max(nulls), 4), np.round(nulls, 4).tolist())
# layer attention mass check
al = F["clean__mega_attn_layer"]
print("clean mega attn layer shape", al.shape)
m = al.mean(0) if al.ndim == 2 else al
print("layer mass:", np.round(m, 3).tolist(), "top", int(np.argmax(m)), "off-top mass", round(1 - m.max() / m.sum(), 3))
out["nulls_clean_auroc"] = nulls
# compare with results.json
print("results.json monitors keys:", list(R["monitors"].keys())[:10])
json.dump(out, open(HERE / "vera_rebuild.json", "w"), indent=1, default=float)
