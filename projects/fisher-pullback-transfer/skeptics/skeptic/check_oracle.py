"""Exploratory: frozen formula with P w replaced by a direction v: r = u'v / w'v.
v in {oracle class edit (per run, own recompute from activations), Kw, Kadam (build's vectors)}.
Primary: exact map; secondary: gap map (pooled ddof-1 sd, as in my rebuild)."""
import glob, json, numpy as np
from collections import defaultdict
from vera_common import D, load, pred_drop_exact, pred_drop_gauss, r2, test_set
L = load(); W = L["W"]; RB = json.load(open("rebuild/rebuild.json"))
sec = {m["name"]: m for m in json.load(open(D / "probes/secondary.json"))}
SW = np.load(D / "probes/secondary_weights.npz")["W"]
src = {p["name"]: p["source"] for p in L["probes"]}
fn = lambda t: t.replace(':', '__').replace(' & ', '_and_').replace(' ', '_')
per = defaultdict(lambda: defaultdict(list))  # (set,target,probe) -> kind -> [pred per seed]
for jf in sorted(glob.glob(str(D / "runs/*__seed*.json"))):
    if "lam0" in jf: continue
    j = json.load(open(jf))
    if not j["success"]: continue
    t = j["target"]; w = W[L["name2i"][t]]
    X, y = test_set(L, t, 1); A = np.load(jf.replace(".json", "__acts.npz"))[f"attacked_acts__{src[t]}"].astype(float)
    E = A - X; ce = E[y].mean(0) - E[~y].mean(0)
    sw = X @ w; Gw = sw[y].mean() - sw[~y].mean(); sa = A @ w; ratio = ((sa[y].mean() - sa[~y].mean()) - Gw) / Gw
    V = {"oracle": ce, "Kw": np.load(D / f"runs/Kw__{fn(t)}.npy").ravel(), "Kadam": np.load(D / f"runs/Kadam__{fn(t)}.npy").ravel()}
    for un in L["sel"]["scored_probes"]:
        if un == t: continue
        u = W[L["name2i"][un]]; Xu, yu = test_set(L, un, 1)
        for k, v in V.items():
            per[("primary", t, un)][k].append(pred_drop_exact(Xu, yu, u, w, ratio, (u @ v) / (w @ v)))
    for m in sec.values():
        if m["target"] != t or not m["kept"]: continue
        u = SW[m["idx"]]
        for k, v in V.items():
            per[("secondary", t, m["name"])][k].append(pred_drop_gauss(X, y, u, w, ratio, (u @ v) / (w @ v)))
for setname, key in [("primary", "primary_pairs"), ("secondary", "secondary_pairs")]:
    meas = {(p["target"], p["probe"]): p["meas"] for p in RB[key]}
    ks = [k for k in per if k[0] == setname]
    m = np.array([meas[(k[1], k[2])] for k in ks])
    print(setname, len(ks), {kind: round(r2([np.mean(per[k][kind]) for k in ks], m), 4) for kind in ["oracle", "Kw", "Kadam"]})
