"""Vera's own law check (skeptic): y from raw per-prompt P(X), x from MY vera_x.json (not overlap.json).

usage: vera_law.py <policy run name> [<policy run name> ...]
Reads raw/behav/<run>/<tag>_plain.npz and raw/screen/base/<tag>_plain.npz (px over 200 test prompts, ids).
Status set = every 'cand' build my vera_x.py classed leaky, plus emu (PREREG s3/s6), all with rows present.
"""
import json, os, re, sys
import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vera_share import r2_fixed

import os
PRJ = os.environ.get("VERA_PRJ", "../../kran-research/projects/footprint-overlap")
X = json.load(open("vera_x.json"))["builds"]
tag = lambda s: s.replace("#rerun", "_rerun").replace("@", "_").replace("+emu", "_emu")
layer = lambda b: int(re.search(r"L(\d+)", b).group(1)) if "@L" in b else None


def load(d, b, ids=None):
    f = f"{d}/{tag(b)}_plain.npz"
    if not os.path.exists(f):
        return None, ids
    z = np.load(f)
    i = [str(x) for x in z["ids"]]
    if ids is not None:
        assert i == ids, f
    return z["px"].astype(np.float64), i


def run(pol, nboot=2000, nperm=9999):
    PD, BD = f"{PRJ}/raw/behav/{pol}", f"{PRJ}/raw/screen/base"
    p0, ids = load(PD, "none")
    b0, _ = load(BD, "none", ids)
    pA, _ = load(PD, "ref@L14.q_norm", ids)
    bA, _ = load(BD, "ref@L14.q_norm", ids)
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(p0), size=(nboot, len(p0)))
    dA = (p0 - pA) - (b0 - bA)
    res = {"policy": pol, "X_none": p0.mean(), "X_A": pA.mean(), "dGAP_A": dA.mean(),
           "dGAP_A_ci": np.percentile(dA[idx].mean(1), [2.5, 97.5]).tolist(), "rows": {}}
    for b, xr in X.items():
        pB, _ = load(PD, b, ids)
        bB, _ = load(BD, b, ids)
        if pB is None or bB is None:
            continue
        dB = (p0 - pB) - (b0 - bB)
        res["rows"][b] = {"kind": xr["kind"], "class": xr["class"], "x": xr["overlap"], "proj": xr["proj"],
                          "l2": xr["l2"], "allpos": xr["allpos"], "x_randdir": xr["null_randdir"][0],
                          "y": dB.mean() / dA.mean(),
                          "y_ci": np.percentile(dB[idx].mean(1) / dA[idx].mean(1), [2.5, 97.5]).tolist(),
                          "X_B": pB.mean(), "layer": layer(b)}
    R = res["rows"]
    status = [b for b in R if (R[b]["kind"] == "cand" and R[b]["class"] == "leaky") or b == "emu"]
    y = np.array([R[b]["y"] for b in status])
    x = np.array([R[b]["x"] for b in status])
    s = {"n_status": len(status), "n_in_02_08": int(((x >= .2) & (x <= .8)).sum()),
         "R2_fixed": r2_fixed(y, x), "pearson_r2": stats.pearsonr(x, y)[0] ** 2,
         "slope_free": np.polyfit(x, y, 1).tolist()}
    prng = np.random.default_rng(1)
    perm = np.array([r2_fixed(y, prng.permutation(x)) for _ in range(nperm)])
    s["perm_p"] = (int((perm >= s["R2_fixed"]).sum()) + 1) / (nperm + 1)
    s["perm_R2_p95"] = float(np.percentile(perm, 95))
    ne = [i for i, b in enumerate(status) if b != "emu"]
    s["R2_without_emu"] = r2_fixed(y[ne], x[ne])
    hc = x >= 0.3
    s["R2_high_cluster"], s["n_high"] = r2_fixed(y[hc], x[hc]), int(hc.sum())
    s["pearson_r_high"] = stats.pearsonr(x[hc], y[hc])[0]
    for k in ("proj", "l2", "allpos", "x_randdir"):
        s[f"R2_{k}"] = r2_fixed(y, np.array([R[b][k] for b in status]))
    for cut in (14, 16):
        d = np.array([1.0 if b == "emu" else float(R[b]["layer"] <= cut) for b in status])
        s[f"R2_depth{cut}"] = r2_fixed(y, d)
        s[f"R2_depth{cut}_without_emu"] = r2_fixed(y[ne], d[ne])
    # y-shuffle null: does the law fit as well if monitors swap their y? (== perm_p); a second null:
    # flip x to a 'random probe' direction (x_randdir) already in R2_x_randdir.
    # same-layer pairs
    conc = disc = ties = 0
    by = {}
    for b in status:
        if b != "emu":
            by.setdefault(R[b]["layer"], []).append(b)
    for l, bs in by.items():
        for i in range(len(bs)):
            for j in range(i + 1, len(bs)):
                dx = R[bs[i]]["x"] - R[bs[j]]["x"]
                dy = R[bs[i]]["y"] - R[bs[j]]["y"]
                if dx * dy > 0: conc += 1
                elif dx * dy < 0: disc += 1
                else: ties += 1
    s["pairs"] = {"conc": conc, "disc": disc, "ties": ties,
                  "p_one_sided": stats.binomtest(conc, conc + disc, 0.5, alternative="greater").pvalue}
    for nm, sel in (("le14", lambda l: l <= 14), ("ge15", lambda l: l >= 15)):
        ii = [i for i, b in enumerate(status) if b != "emu" and sel(R[b]["layer"])]
        r_ = stats.spearmanr(x[ii], y[ii])
        s[f"spearman_{nm}"] = [float(r_.correlation), float(r_.pvalue), len(ii)]
    # bit-exact rows and the after-block-26 builds
    be = [b for b in R if R[b]["class"] == "bitexact"]
    s["bitexact_max_abs_share"] = max(abs(R[b]["y"]) for b in be) if be else None
    s["n_bitexact"] = len(be)
    s["after26"] = {b: (round(R[b]["x"], 3), round(R[b]["y"], 3)) for b in status
                    if b != "emu" and (R[b]["layer"] >= 26 or b == "ref@L25.resid")}
    s["seen"] = {b: (round(R[b]["x"], 3), round(R[b]["y"], 3)) for b in ("ref@L7.q_norm", "ref@L21.q_norm", "emu") if b in R}
    res["summary"] = s
    return res


if __name__ == "__main__":
    allr = {}
    for pol in sys.argv[1:]:
        r = run(pol)
        allr[pol] = r
        print(pol, json.dumps({k: v for k, v in r.items() if k not in ("rows", "summary")}, default=float))
        print(json.dumps(r["summary"], indent=1, default=float))
    json.dump(allr, open("vera_law_" + "_".join(p.split("_")[1] + p.split("_")[-1] for p in sys.argv[1:]) + ".json", "w"),
              indent=1, default=float)
