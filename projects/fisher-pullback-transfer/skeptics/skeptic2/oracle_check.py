"""Hugo: check of the build's exploratory 'direction swap' numbers (Bruno #1757, Kasper #1769).
Same frozen formula, P w replaced by a direction v: r = u'v / w'v. Directions: measured class-differential edit
(oracle), Kadam / Kw (build's first-step vectors), F^-1 w, w. Primary = exact empirical map; secondary = gap map.
Reuses check.py helpers (own code)."""
import glob, json, os
import numpy as np
import check as C

FB = C.FB["F_common"]; PF = np.linalg.inv(FB + np.median(np.linalg.eigvalsh(FB)) * np.eye(512))
prim, sec = {}, {}
for jf in sorted(glob.glob(C.D + "runs/*__seed?.json")):
    j = json.load(open(jf))
    if j["lambda_kl"] != 30 or not j["success"]: continue
    wn = j["target"]; wsrc, lab = wn.split(":", 1); w = C.W[C.name2i[wn]]
    af = jf[:-5] + "__acts.npz"; Z = np.load(af if os.path.exists(af) else jf[:-5] + ".npz")
    Xw, Lw = C.test_idx(wsrc); yw = (Lw == lab).astype(int)
    Aw = Z[f"attacked_acts__{wsrc}"].astype(np.float64); E = Aw - Xw
    e = E[yw == 1].mean(0) - E[yw == 0].mean(0)
    tag = wn.replace(":", "__").replace(" & ", "_and_").replace(" ", "_")
    dirs = {"oracle": e, "Kadam": np.load(C.D + f"runs/Kadam__{tag}.npy").ravel(),
            "Kw": np.load(C.D + f"runs/Kw__{tag}.npy").ravel(), "F": PF @ w, "I": w}
    Gw, _ = C.gsd(Xw @ w, yw); ratio = (C.gsd(Aw @ w, yw)[0] - Gw) / Gw
    for un in C.sel["scored_probes"]:
        if un == wn: continue
        src, l2 = un.split(":", 1); X, L = C.test_idx(src); y = (L == l2).astype(int); u = C.W[C.name2i[un]]
        su, sw = X @ u, X @ w; a0 = C.auroc(su, y); A = Z[f"attacked_acts__{src}"].astype(np.float64)
        d = prim.setdefault((wn, un), {"meas": [], **{k: [] for k in dirs}}); d["meas"].append(a0 - C.auroc(A @ u, y))
        for k, v in dirs.items(): d[k].append(a0 - C.auroc(su + ratio * (u @ v) / (w @ v) * sw, y))
    for s in C.SEC:
        if s["target"] != wn or not s["kept"]: continue
        u = C.SW[s["idx"]]; su = Xw @ u; Gu, sdu = C.gsd(su, yw); a0 = C.auroc(su, yw)
        d = sec.setdefault((wn, s["name"]), {"meas": [], **{k: [] for k in dirs}}); d["meas"].append(a0 - C.auroc(Aw @ u, yw))
        for k, v in dirs.items():
            r = (u @ v) / (w @ v)
            d[k].append(C.Phi(Gu / sdu / np.sqrt(2)) - C.Phi((Gu + ratio * r * Gw) / sdu / np.sqrt(2)))
for nm, tab in (("primary", prim), ("secondary", sec)):
    m = [np.mean(v["meas"]) for v in tab.values()]
    print(nm, len(tab), {k: round(C.r2([np.mean(v[k]) for v in tab.values()], m), 3) for k in ("oracle", "Kadam", "Kw", "F", "I")})
