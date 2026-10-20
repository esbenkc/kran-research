"""Vera: which token readout carries the recovered signal? Clean-fit linear probes (StandardScaler + LR C=0.01) at
layers 14/18/22 on (a) the last token (the full stop), (b) the second-to-last token (end of the country name),
(c) max over tokens of a per-token probe trained on all clean pool tokens with the statement label.
Same 1% line rule and pooled catch as everything else.
    cd kran-research && uv run python ../swarm-theory-1020-megaprobe-after/skeptic/vera_tokread.py
"""
import json

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from vera_core import ATT, HERE, POOL, PROJ, STATES, YF, flags, pooled_catch, city_draws

off = np.load(PROJ / "data/acts/clean/off.npy")
T = {s: np.load(PROJ / f"data/acts/{s}/tok.npy", mmap_mode="r") for s in STATES}
W = city_draws()
res = {}
pool_tok = np.concatenate([np.arange(off[i], off[i + 1]) for i in np.where(POOL)[0]])
pool_tok_y = np.concatenate([np.full(off[i + 1] - off[i], YF[i]) for i in np.where(POOL)[0]])
for l in [14, 18, 22]:
    for name, pick in [("last", off[1:] - 1), ("second_last", off[1:] - 2)]:
        X = {s: np.asarray(T[s][pick, l], np.float64) for s in STATES}
        sc = StandardScaler().fit(X["clean"][POOL])
        lr = LogisticRegression(C=0.01, class_weight="balanced", max_iter=5000).fit(sc.transform(X["clean"][POOL]), YF[POOL])
        S = {s: lr.decision_function(sc.transform(X[s])) for s in STATES}
        p, ci, _ = pooled_catch([flags(S[s]) for s in ATT], W)
        res[f"L{l}_{name}"] = {"att": p, "ci": ci.tolist(), "clean": pooled_catch([flags(S["clean"])])}
        print(f"L{l} {name:11s} clean {res[f'L{l}_{name}']['clean']:.3f} att {p:.3f} [{ci[0]:.3f},{ci[1]:.3f}]", flush=True)
    Xt = np.asarray(T["clean"][pool_tok, l], np.float64)
    sc = StandardScaler().fit(Xt)
    lr = LogisticRegression(C=0.01, class_weight="balanced", max_iter=5000).fit(sc.transform(Xt), pool_tok_y)
    S = {}
    for s in STATES:
        st = lr.decision_function(sc.transform(np.asarray(T[s][:, l], np.float64)))
        S[s] = np.maximum.reduceat(st, off[:-1])
    p, ci, _ = pooled_catch([flags(S[s]) for s in ATT], W)
    res[f"L{l}_maxtok"] = {"att": p, "ci": ci.tolist(), "clean": pooled_catch([flags(S["clean"])])}
    print(f"L{l} maxtok      clean {res[f'L{l}_maxtok']['clean']:.3f} att {p:.3f} [{ci[0]:.3f},{ci[1]:.3f}]", flush=True)
json.dump(res, open(HERE / "vera_tokread.json", "w"), indent=1, default=float)
