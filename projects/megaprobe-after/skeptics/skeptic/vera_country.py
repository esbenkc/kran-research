"""Vera: clean-fit linear probe on the last country-name token (the token before the final full stop) at every layer
1..28, to check build's D3 curve. StandardScaler + LR(C=0.01, balanced), clean 1400 pool, 1019 line rule, pooled catch
+ city CI, eval false alarms, twin d' in clean spreads. Also counts statements whose last token is not a bare '.'.
    cd kran-research && uv run python ../swarm-theory-1020-megaprobe-after/skeptic/vera_country.py
"""
import json

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from transformers import AutoTokenizer

from vera_core import ATT, EV_IDX, HERE, POOL, PROJ, ROWS, STATES, YF, city_draws, dprime, flags, pooled_catch

tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
last = [tok.convert_ids_to_tokens(tok(r["text"], add_special_tokens=False)["input_ids"])[-1] for r in ROWS]
print("statements whose last token is not '.':", sum(t != "." for t in last), "examples", sorted(set(t for t in last if t != "."))[:8])
off = np.load(PROJ / "data/acts/clean/off.npy")
pick = off[1:] - 2
T = {s: np.load(PROJ / f"data/acts/{s}/tok.npy", mmap_mode="r") for s in STATES}
X0 = {s: np.asarray(T[s][pick]) for s in STATES}  # [2992, 29, 1536] fp16, ~1.1 GB for 4 states
W = city_draws()
res = {}
for l in range(1, 29):
    X = {s: X0[s][:, l].astype(np.float64) for s in STATES}
    sc = StandardScaler().fit(X["clean"][POOL])
    lr = LogisticRegression(C=0.01, class_weight="balanced", max_iter=5000).fit(sc.transform(X["clean"][POOL]), YF[POOL])
    S = {s: lr.decision_function(sc.transform(X[s])) for s in STATES}
    p, ci, _ = pooled_catch([flags(S[s]) for s in ATT], W)
    dp = dprime(S, S["clean"])
    res[l] = {"clean": pooled_catch([flags(S["clean"])]), "att": p, "ci": ci.tolist(),
              "fpr_eval": [float(flags(S[s])[EV_IDX][YF[EV_IDX] == 0].mean()) for s in ATT],
              "clean_d": dp["clean"][0], "att_d": float(np.mean([dp[s][0] for s in ATT]))}
    print(f"L{l:2d} clean {res[l]['clean']:.3f} att {p:.3f} [{ci[0]:.3f},{ci[1]:.3f}] fpr {np.round(res[l]['fpr_eval'], 3).tolist()} "
          f"d' {res[l]['clean_d']:.2f}->{res[l]['att_d']:.2f}", flush=True)
json.dump(res, open(HERE / "vera_country.json", "w"), indent=1, default=float)
