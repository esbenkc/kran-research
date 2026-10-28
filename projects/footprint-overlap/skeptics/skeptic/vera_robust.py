"""Vera robustness of the REJECTED verdict (skeptic). numpy + sklearn.

1. Monitor bootstrap: resample the 81 status builds with replacement (5000x), fixed-line R^2 per resample.
   Joint prompt x monitor bootstrap (2000x).
2. Knob sensitivity of the x-axis (NOT status, forking-path check only): C in {0.01, 0.1, 1, 10},
   position 25 vs 26/27/28, no standardisation. Does any reasonable knob lift s1 to R^2 >= 0.70?
3. Base-gap size: how much the base-model subtraction moves y.
"""
import json, re, warnings
import numpy as np
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings("ignore")
from vera_share import r2_fixed

import os
PRJ = os.environ.get("VERA_PRJ", "../../kran-research/projects/footprint-overlap")
D = f"{PRJ}/raw/screen/base"
L = json.load(open("vera_law_s0at8192_s1at8192.json"))
pols = list(L)
split = np.array([json.loads(l)["split"] for l in open(f"{PRJ}/raw/prompts.jsonl")])
tr, te = split == "train", split == "test"
tag = lambda s: s.replace("#rerun", "_rerun").replace("@", "_").replace("+emu", "_emu")
f32 = lambda u: (np.asarray(u).astype(np.uint32) << 16).view(np.float32)
out = {}
for pol in pols:
    R = L[pol]["rows"]
    st = [b for b in R if (R[b]["kind"] == "cand" and R[b]["class"] == "leaky") or b == "emu"]
    x = np.array([R[b]["x"] for b in st]); y = np.array([R[b]["y"] for b in st])
    rng = np.random.default_rng(5)
    bs = []
    for _ in range(5000):
        i = rng.integers(0, len(st), len(st))
        if np.var(y[i]) > 0:
            bs.append(r2_fixed(y[i], x[i]))
    bs = np.array(bs)
    out[pol] = {"R2": r2_fixed(y, x), "monitor_boot_ci95": np.percentile(bs, [2.5, 97.5]).tolist(),
                "P_R2_ge_0.7_monitor_boot": float((bs >= 0.7).mean()),
                "max_abs_base_gap": max(abs(R[b]["X_B"] - L[pol]["X_none"]) * 0 + 0 for b in st)}
    out[pol]["status"] = st
# joint prompt x monitor bootstrap needs per-prompt arrays
def px(d, b):
    return np.load(f"{d}/{tag(b)}_plain.npz")["px"].astype(np.float64)
for pol in pols:
    st = out[pol]["status"]
    R = L[pol]["rows"]
    x = np.array([R[b]["x"] for b in st])
    PD = f"{PRJ}/raw/behav/{pol}"
    p0, b0 = px(PD, "none"), px(D, "none")
    dA = (p0 - px(PD, "ref@L14.q_norm")) - (b0 - px(D, "ref@L14.q_norm"))
    dB = np.stack([(p0 - px(PD, b)) - (b0 - px(D, b)) for b in st])  # builds x prompts
    basegap = np.array([(b0 - px(D, b)).mean() for b in st])
    yraw = np.array([(p0 - px(PD, b)).mean() for b in st]) / (p0 - px(PD, "ref@L14.q_norm")).mean()
    out[pol]["base_gap_abs_max"] = float(np.abs(basegap).max())
    out[pol]["R2_without_base_subtraction"] = r2_fixed(yraw, x)
    rng = np.random.default_rng(6)
    jb = []
    for _ in range(2000):
        ip = rng.integers(0, dB.shape[1], dB.shape[1]); im = rng.integers(0, len(st), len(st))
        yy = dB[im][:, ip].mean(1) / dA[ip].mean()
        jb.append(r2_fixed(yy, x[im]))
    jb = np.array(jb)
    out[pol]["joint_boot_ci95"] = np.percentile(jb, [2.5, 97.5]).tolist()
    out[pol]["P_R2_ge_0.7_joint_boot"] = float((jb >= 0.7).mean())
# knob sensitivity of x (one fit per knob, applied to the same 81 builds)
st = out[pols[0]]["status"]
tails = {b: np.load(f"{D}/{tag(b)}_rec_tail.npy", mmap_mode="r") for b in ["none", "ref@L14.q_norm"] + st}
knobs = {}
for pos in (25, 26, 27, 28):
    X0, XA = f32(tails["none"][:, pos, :]), f32(tails["ref@L14.q_norm"][:, pos, :])
    XB = {b: f32(tails[b][te, pos, :]) for b in st}
    for C in (0.01, 0.1, 1.0, 10.0):
        for std in (True, False):
            if not std and C != 1.0:
                continue
            Xtr = np.concatenate([X0[tr], XA[tr]]).astype(np.float64)
            yy = np.r_[np.zeros(tr.sum()), np.ones(tr.sum())]
            mu, sd = (Xtr.mean(0), Xtr.std(0)) if std else (np.zeros(1024), np.ones(1024))
            sd = np.where(sd == 0, 1.0, sd)
            c = LogisticRegression(C=C, max_iter=5000).fit((Xtr - mu) / sd, yy)
            P = lambda Z: c.predict_proba((Z - mu) / sd)[:, 1].mean()
            p0, pA = P(X0[te]), P(XA[te])
            xk = np.array([(P(XB[b]) - p0) / (pA - p0) for b in st])
            k = f"pos{pos}_C{C}_{'std' if std else 'raw'}"
            knobs[k] = {pol: r2_fixed(np.array([L[pol]["rows"][b]["y"] for b in st]), xk) for pol in pols}
            print(k, {p[4:6]: round(v, 3) for p, v in knobs[k].items()}, flush=True)
out["knobs"] = knobs
for pol in pols:
    out[pol].pop("status"); out[pol].pop("max_abs_base_gap")
print(json.dumps({p: out[p] for p in pols}, indent=1))
n_pass = sum(all(v >= 0.7 for v in d.values()) for d in knobs.values())
print("knobs where BOTH seeds reach R2>=0.7:", n_pass, "of", len(knobs))
json.dump(out, open("vera_robust.json", "w"), indent=1)
