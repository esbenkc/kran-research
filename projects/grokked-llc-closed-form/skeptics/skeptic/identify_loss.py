"""Identify index->(a,b) map and centring convention from the saved train_loss, then
check my gradient by finite differences."""
import glob, json, os
import numpy as np
from common import loss, grad, targets

D = os.path.join(os.path.dirname(__file__), "../../data/ckpt")
out = {}
for f in sorted(glob.glob(f"{D}/p7_*.npz") + glob.glob(f"{D}/p11_*.npz"))[:6]:
    z = np.load(f)
    p = int(z["p"]); W, V = z["W"], z["V"]
    res = {}
    for amap in ["div", "mod"]:
        idx = z["train_idx"]
        a, b = (idx // p, idx % p) if amap == "div" else (idx % p, idx // p)
        Y = targets(a, b, p)
        for mode in ["samples", "outputs", "none"]:
            res[f"{amap}/{mode}"] = loss(W, V, a, b, p, Y, mode)
    res["saved"] = float(z["train_loss"])
    out[os.path.basename(f)] = res
    print(os.path.basename(f), {k: f"{v:.3e}" for k, v in res.items()})

# finite-difference gradient check (samples-centred), random point
rng = np.random.default_rng(0)
p, K, N = 7, 5, 30
idx = rng.choice(p * p, N, replace=False); a, b = idx // p, idx % p; Y = targets(a, b, p)
W = rng.standard_normal((2 * p, K)); V = rng.standard_normal((p, K))
gW, gV = grad(W, V, a, b, p, Y)
h = 1e-6; err = []
for _ in range(20):
    i, j = rng.integers(2 * p), rng.integers(K)
    Wp = W.copy(); Wp[i, j] += h; Wm = W.copy(); Wm[i, j] -= h
    err.append(abs((loss(Wp, V, a, b, p, Y) - loss(Wm, V, a, b, p, Y)) / (2 * h) - gW[i, j]))
    i = rng.integers(p)
    Vp = V.copy(); Vp[i, j] += h; Vm = V.copy(); Vm[i, j] -= h
    err.append(abs((loss(W, Vp, a, b, p, Y) - loss(W, Vm, a, b, p, Y)) / (2 * h) - gV[i, j]))
print("max FD grad err", max(err))
out["fd_grad_max_err"] = max(err)
json.dump(out, open(os.path.join(os.path.dirname(__file__), "identify_loss.json"), "w"), indent=1)
