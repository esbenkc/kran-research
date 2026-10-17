"""skeptic (Vera): independent checks of the follow-up-2 Jacobian fold with MY code (common.py), not qnet.py.

E8' : at a zero-loss network (planted: targets = the net's own outputs), the Hessian of
      L_n = (1/N) sum_i 1/2 ||centred residual_i||^2 equals J^T J / N (J = Jacobian of the
      centred outputs). Finite differences of my analytic gradient vs my J.
R1  : my Jacobian at the saved (pre-polish) grokked checkpoint p17_K51_s0: rank at rtol 1e-9
      vs K(3p-2), gap at the cut, and flat share (s^2/N < 1e-4) next to build's saved spectrum
      (which is at the polished point; polish changes the small curvatures, so only the rank
      and rough share are comparable).
"""
import json, glob
import numpy as np
from common import forward, grad, jacobian, loss, targets, pairs

out = {}
# ---------------- E8'
rng = np.random.default_rng(7)
p, K, N = 5, 4, 18
idx = rng.choice(p * p, N, replace=False)
a, b = pairs(p, idx)
W = rng.normal(0, 0.5, (2 * p, K)); V = rng.normal(0, 0.5, (p, K))
Y, _, _ = forward(W, V, a, b, p)          # planted targets: zero loss at (W, V)
print("L at planted point:", loss(W, V, a, b, p, Y))
def flatgrad(th):
    Wt = th[:W.size].reshape(W.shape); Vt = th[W.size:].reshape(V.shape)
    gW, gV = grad(Wt, Vt, a, b, p, Y)
    return np.concatenate([gW.ravel(), gV.ravel()])
th0 = np.concatenate([W.ravel(), V.ravel()])
h = 1e-6
Hfd = np.stack([(flatgrad(th0 + h * e) - flatgrad(th0 - h * e)) / (2 * h) for e in np.eye(th0.size)], 1)
Hfd = 0.5 * (Hfd + Hfd.T)
J = jacobian(W, V, a, b, p)
print("J shape", J.shape)
Hgn = J.T @ J / N
err = np.abs(Hfd - Hgn).max() / np.abs(Hgn).max()
print("E8' max|H_fd - J^TJ/N| / max|J^TJ/N| =", err)
out["E8_rel_err"] = float(err)

# ---------------- R1
z = np.load("data/ckpt/p17_K51_s0_grok.npz", allow_pickle=True)
print("ckpt keys", z.files)
meta = json.load(open("data/ckpt/p17_K51_s0.json"))
out["ckpt_keys"] = z.files
print({k: (v if not isinstance(v, list) or len(v) < 8 else f"list[{len(v)}]") for k, v in meta.items()})

def spectrum_stats(W, V, a, b, p, N, s_ref=None):
    J = jacobian(W, V, a, b, p)
    s = np.linalg.svd(J, compute_uv=False)
    del J
    shift = W.shape[1] * (3 * p - 2)
    res = {}
    for tol in [1e-4, 1e-6, 1e-9, 1e-12]:
        r = int(np.sum(s > tol * s[0])); hh = s[:r] ** 2 / N
        res[f"{tol:g}"] = dict(rank=r, rank_rel=r / shift, flat=float(np.mean(hh < 1e-4)),
                               gap=float(s[r - 1] / s[r]) if r < len(s) else None,
                               quad=float(np.sum(0.5 * hh / (hh + 1e-4))))
    return res, s

def adam_polish(W, V, a, b, p, Y, steps=5000, lr=1e-3, target=1e-8):
    th = [W.copy(), V.copy()]; m = [np.zeros_like(W), np.zeros_like(V)]; v = [np.zeros_like(W), np.zeros_like(V)]
    for t in range(1, steps + 1):
        L = loss(th[0], th[1], a, b, p, Y)
        if L < target:
            break
        g = grad(th[0], th[1], a, b, p, Y)
        for i in range(2):
            m[i] = 0.9 * m[i] + 0.1 * g[i]; v[i] = 0.999 * v[i] + 0.001 * g[i] ** 2
            th[i] -= lr * (m[i] / (1 - 0.9 ** t)) / (np.sqrt(v[i] / (1 - 0.999 ** t)) + 1e-12)
    return th[0], th[1], loss(th[0], th[1], a, b, p, Y), t

for tag in ["p17_K51_s0", "p11_K33_s0"]:
    z = np.load(f"data/ckpt/{tag}_grok.npz")
    p_, K_ = int(z["p"]), int(z["K"])
    a_, b_ = pairs(p_, z["train_idx"])
    Y_ = targets(a_, b_, p_)
    N_ = len(a_)
    W0, V0 = z["W"], z["V"]
    print(tag, "W", W0.shape, "V", V0.shape, "N", N_, "L_ckpt (mine)", loss(W0, V0, a_, b_, p_, Y_))
    r_raw, _ = spectrum_stats(W0, V0, a_, b_, p_, N_)
    Wp, Vp, Lp, tp = adam_polish(W0, V0, a_, b_, p_, Y_)
    r_pol, s_pol = spectrum_stats(Wp, Vp, a_, b_, p_, N_)
    s_b = np.sort(np.load(f"data/llc_traces/{tag}_grok.npz")["singular_values"])[::-1]
    r9b = int(np.sum(s_b > 1e-9 * s_b[0])); hb = s_b[:r9b] ** 2 / N_
    out[tag] = dict(raw=r_raw, my_polish=dict(stats=r_pol, L=float(Lp), steps=int(tp)),
                    build_saved=dict(rank_1e9=r9b, flat=float(np.mean(hb < 1e-4)), quad=float(np.sum(0.5 * hb / (hb + 1e-4)))),
                    s_rel_diff_top=float(np.max(np.abs(s_pol[:r9b] - s_b[:r9b]) / s_b[:r9b])))
    print(json.dumps(out[tag], indent=1, default=float))
json.dump(out, open("skeptics/skeptic/jac_recheck.json", "w"), indent=1, default=float)
