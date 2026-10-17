"""Exploratory: cosine of measured class-mean edit (pos minus neg, target test set) with w, Sigma w, F^-1 w, Kw."""
import glob, json, numpy as np
from vera_common import D, load, metrics
L = load(); W = L["W"]
Ms, _ = metrics(L, 1.0, np.load(D / "fisher/fisher_big.npz")["F_common"])
cos = lambda a, b: a @ b / np.linalg.norm(a) / np.linalg.norm(b)
rows = []
for jf in sorted(glob.glob(str(D / "runs/*__seed*.json"))):
    if "lam0" in jf: continue
    j = json.load(open(jf))
    if not j["success"]: continue
    t = j["target"]; w = W[L["name2i"][t]]; z = np.load(jf.replace(".json", ".npz")); za = np.load(jf.replace(".json", "__acts.npz"))
    # recompute class edit from activations myself
    from vera_common import test_set
    X, y = test_set(L, t, 1); A = za[f"attacked_acts__{L['probes'][L['name2i'][t]]['source']}"].astype(float)
    E = A - X; ce = E[y].mean(0) - E[~y].mean(0)
    kw = np.load(D / f"runs/Kw__{t.replace(':','__').replace(' & ','_and_').replace(' ','_')}.npy")
    r = dict(t=t, s=j["seed"], w=cos(ce, w), Sw=cos(ce, Ms["whitened"] @ w), Fw=cos(ce, Ms["fisher"] @ w), Kw=cos(ce, kw.ravel()),
             theirs_ce_check=cos(ce, z["mean_edit_pos"] - z["mean_edit_neg"]))
    rows.append(r); print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()})
a = {k: np.mean([abs(r[k]) for r in rows]) for k in ["w", "Sw", "Fw", "Kw"]}
print("mean |cos| over 14 runs:", {k: round(v, 3) for k, v in a.items()})
