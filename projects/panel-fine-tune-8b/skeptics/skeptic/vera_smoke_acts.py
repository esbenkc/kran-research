"""Vera: pipeline check on Bruno's Qwen3-0.6B smoke, from saved activations (not from build code).
(1) probe scores in scores/<state>.npz equal ((h - mu)/sd).w + b applied to acts/<state>/cty.npy;
(2) the last readout is the final RMSNorm of the one before it (post-norm vs pre-norm indexing);
(3) panelmax equals max_R z_R with zmoments.json.
Usage: python vera_smoke_acts.py SMOKE_DIR HF_SNAPSHOT_DIR"""
import glob, json, sys
import numpy as np
from safetensors import safe_open

D, snap = sys.argv[1], sys.argv[2]
P = np.load(f"{D}/fits/panel.npz")
zm = json.load(open(f"{D}/fits/zmoments.json"))
for st in ["clean", "panel__lam10000__seed0"]:
    A = np.load(f"{D}/acts/{st}/cty.npy", mmap_mode="r")
    ro = json.load(open(f"{D}/acts/{st}/readouts.json"))
    S = np.load(f"{D}/scores/{st}.npz")
    worst = 0.0
    zs = []
    for R in range(10, 30):
        h = np.asarray(A[:, ro.index(R), :], dtype=np.float64)
        s = ((h - P[f"tok{R}__mu"]) / P[f"tok{R}__sd"]) @ P[f"tok{R}__w"].astype(np.float64) + float(P[f"tok{R}__b"])
        worst = max(worst, float(np.abs(s - S[f"tok{R}"]).max()))
        zs.append((s - zm[f"tok{R}"]["mu"]) / zm[f"tok{R}"]["sd"])
    pm = np.max(zs, 0)
    print(st, "max |score diff| over 20 probes:", round(worst, 5), "| max |panelmax diff|:",
          round(float(np.abs(pm - S["panelmax"]).max()), 5))

# (2) RMSNorm check on the clean acts
w = None
for f in glob.glob(f"{snap}/*.safetensors"):
    with safe_open(f, "pt") as g:
        if "model.norm.weight" in g.keys():
            w = g.get_tensor("model.norm.weight").float().numpy().astype(np.float64)
A = np.load(f"{D}/acts/clean/cty.npy", mmap_mode="r"); ro = json.load(open(f"{D}/acts/clean/readouts.json"))
last, pre = ro[-1], ro[-2]
x = np.asarray(A[:, ro.index(pre), :], dtype=np.float64)
y = np.asarray(A[:, ro.index(last), :], dtype=np.float64)
yn = x / np.sqrt((x ** 2).mean(1, keepdims=True) + 1e-6) * w
rel = np.abs(yn - y).max(1) / np.abs(y).max(1)
print(f"readout {last} vs RMSNorm(readout {pre}): median rel err {np.median(rel):.4f}, max {rel.max():.4f}; "
      f"identity rel err {np.median(np.abs(x - y).max(1) / np.abs(y).max(1)):.3f}")
