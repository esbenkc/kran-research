"""Vera: own rebuild of the panel from raw acts (feats from vera_feats.py).
- own clean fits: StandardScaler + LogisticRegression(class_weight balanced) on the 1400 clean pool statements, for the
  country token at readouts 1..29 (29 = L28post) and the sentence average at readouts 1..28; C = build's C.json value
  (the frozen 1020 rule's output) AND C = 0.01 fixed (sensitivity).
- build's saved weights (fits/panel.npz) applied to the raw acts with my own arithmetic: ((x-mu)/sd).w + b.
- megaprobe: theory-1020 formula on final_mega.pt + pca_clean.pt (my vera_eqcheck.py), readouts 1..28.
Lines per state (2nd-highest of the 200 line-setting true statements); catch pooled over the given attacked states.
    python vera_panel.py <tag> <state> [<state> ...]   (clean is always scored too)
"""
import json
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from vera_core import EVAL, HERE, LINE_T, POOL, PROJ, YF, auroc, flags, pooled

TAG, STATES = sys.argv[1], sys.argv[2:]
FE = HERE / "feats"
ALL = ["clean"] + STATES
C = json.load(open(PROJ / "data/fits/C.json"))
BP = np.load(PROJ / "data/fits/panel.npz")
F = {s: (np.load(FE / f"{s}_ctry.npy", mmap_mode="r"), np.load(FE / f"{s}_sent.npy", mmap_mode="r")) for s in ALL}


def feat(s, kind, r):
    return np.asarray(F[s][0 if kind == "tok" else 1][:, r], np.float64)


SC = {}  # monitor -> state -> scores (2992,)
for kind, rr in [("tok", range(1, 30)), ("mean", range(1, 29))]:
    for r in rr:
        name = f"{kind}{r}"
        Xc = feat("clean", kind, r)
        for cname, cval in [("rule", C[name]["C"] if name in C else None), ("c001", 0.01)]:
            if cval is None:
                continue
            sc = StandardScaler().fit(Xc[POOL])
            lr = LogisticRegression(C=cval, class_weight="balanced", max_iter=5000).fit(sc.transform(Xc[POOL]), YF[POOL])
            SC[f"{name}__{cname}"] = {s: lr.decision_function(sc.transform(feat(s, kind, r))) for s in ALL}
        if f"{name}__w" in BP:
            mu, sd, w, b = (BP[f"{name}__{k}"].astype(np.float64) for k in ["mu", "sd", "w", "b"])
            SC[f"{name}__build"] = {s: ((feat(s, kind, r) - mu) / sd) @ w + b for s in ALL}
        print(name, "fit", flush=True)

# random-direction null at L18 / L19 (country token): 20 unit directions in standardized space, same line rule
rng = np.random.default_rng(1021)
for r in (18, 19):
    Xc = feat("clean", "tok", r); sc = StandardScaler().fit(Xc[POOL])
    Xs = {s: sc.transform(feat(s, "tok", r)) for s in ALL}
    for k in range(20):
        u = rng.standard_normal(Xc.shape[1]); u /= np.linalg.norm(u)
        SC[f"rand{r}_{k}"] = {s: Xs[s] @ u for s in ALL}

# megaprobe (formula, float64), readouts 1..28 at every stored token
import torch  # noqa: E402

m = torch.load(PROJ / "data/ref/fits1020/final_mega.pt", map_location="cpu", weights_only=False)["P"]
pc = torch.load(PROJ / "data/ref/fits1020/pca_clean.pt", map_location="cpu", weights_only=False)
q, c, w, b = [m[k].double().numpy() for k in ["q", "c", "w", "b"]]
mu, V, scl = [pc[k].double().numpy() for k in ["mu", "V", "scale"]]
K = q.shape[1]
Vk = V[:, :, :K] * scl[:, None, :K]
SC["mega"] = {}
for s in ALL:
    T = np.load(PROJ / f"data/acts/{s}/tok.npy", mmap_mode="r"); o = np.load(PROJ / f"data/acts/{s}/off.npy")
    out = np.empty(2992)
    for a in range(0, 2992, 64):
        e = min(a + 64, 2992)
        X = np.asarray(T[o[a]:o[e], 1:29], np.float64)
        Z = np.einsum("tld,ldk->tlk", X - mu[None], Vk)
        A = np.einsum("tlk,lk->tl", Z, q) + c[None]
        G = np.einsum("tlk,lk->tl", Z, w)
        for i in range(a, e):
            sl = slice(o[i] - o[a], o[i + 1] - o[a])
            ai = A[sl]; ai = np.exp(ai - ai.max()); ai /= ai.sum()
            out[i] = (ai * G[sl]).sum() + b
    SC["mega"][s] = out
    print("mega", s, flush=True)

res = {}
ev = EVAL
for k, d in SC.items():
    r = {"clean": pooled([flags(d["clean"])]), "clean_auroc": auroc(d["clean"][ev], YF[ev])}
    if STATES:
        r["att"] = pooled([flags(d[s]) for s in STATES])
        r["per_state"] = {s: pooled([flags(d[s])])["catch"] for s in STATES}
        r["att_auroc"] = {s: auroc(d[s][ev], YF[ev]) for s in STATES}
    res[k] = r
np.savez_compressed(HERE / f"vera_scores_{TAG}.npz", **{f"{k}__{s}": v for k, d in SC.items() for s, v in d.items()})
json.dump(res, open(HERE / f"vera_panel_{TAG}.json", "w"), indent=1)
for k in sorted(res):
    r = res[k]
    a = r.get("att", {})
    print(f"{k:16s} clean {r['clean']['catch']:.3f}" + (f"  att {a['catch']:.3f} [{a['lo']:.3f},{a['hi']:.3f}] fa {a['fa']:.3f}" if a else ""))
