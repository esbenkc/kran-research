"""skeptic (Vera): own exact-SVD vs Gram check on one Gram-route verdict network (p23_K92_s0).

Claim (post l.142/153 at blog 0356cf6): exact SVD of J gives rank exactly K(3p-2) at rtol 1e-9 with a
1e7-1e8 gap; the Gram route at 1e-6 counts one direction fewer; Gram error <= 7.5e-8 s_max; flat
shares agree to 0.1 percentage points.
My code: common.jacobian (centring over samples), my own Adam polish (lr 1e-3, <=5000 steps,
stop at L<1e-8), numpy SVD (values only) and eigvalsh of the smaller Gram matrix.
Memory: J is 8510 x 6348 float64 (432 MB); built in place of the einsum version to stay < 2 GB.
"""
import json
import numpy as np
from common import forward, grad, loss, targets, pairs

def jac_lowmem(W, V, a, b, p):
    N, K = len(a), W.shape[1]
    F, pre, H = forward(W, V, a, b, p)
    J = np.zeros((N, p, 3 * p * K))
    for i in range(N):
        g = 2 * pre[i][None, :] * V                     # p x K : dF_k/dpre_j
        blk = J[i, :, :2 * p * K].reshape(p, 2 * p, K)
        blk[:, a[i], :] += g
        blk[:, p + b[i], :] += g
        JV = J[i, :, 2 * p * K:].reshape(p, p, K)
        for k in range(p):
            JV[k, k, :] = H[i]
    J -= J.mean(0, keepdims=True)
    return J.reshape(N * p, -1)

def adam_polish(W, V, a, b, p, Y, steps=5000, lr=1e-3, target=1e-8):
    th = [W.copy(), V.copy()]; m = [0 * W, 0 * V]; v = [0 * W, 0 * V]
    for t in range(1, steps + 1):
        if loss(th[0], th[1], a, b, p, Y) < target:
            break
        g = grad(th[0], th[1], a, b, p, Y)
        for i in range(2):
            m[i] = 0.9 * m[i] + 0.1 * g[i]; v[i] = 0.999 * v[i] + 0.001 * g[i] ** 2
            th[i] -= lr * (m[i] / (1 - 0.9 ** t)) / (np.sqrt(v[i] / (1 - 0.999 ** t)) + 1e-12)
    return th[0], th[1], float(loss(th[0], th[1], a, b, p, Y)), t

tag = "p23_K92_s0"
z = np.load(f"data/ckpt/{tag}_grok.npz")
p, K = int(z["p"]), int(z["K"])
a, b = pairs(p, z["train_idx"]); Y = targets(a, b, p); N = len(a)
W, V, Lp, tp = adam_polish(z["W"], z["V"], a, b, p, Y)
print(tag, "N", N, "polish L", Lp, "steps", tp, flush=True)
J = jac_lowmem(W, V, a, b, p)
print("J", J.shape, flush=True)
G = J.T @ J if J.shape[1] <= J.shape[0] else J @ J.T
s_ex = np.linalg.svd(J, compute_uv=False)
del J
s_gr = np.sqrt(np.clip(np.linalg.eigvalsh(G)[::-1], 0, None)); del G
shift = K * (3 * p - 2)
def stats(s, tol):
    r = int(np.sum(s > tol * s[0])); h = s[:r] ** 2 / N
    return dict(rank=r, rank_minus_shift=r - shift, gap=float(s[r - 1] / s[r]) if r < len(s) else None,
                flat=float(np.mean(h < 1e-4)), first_dropped_rel=float(s[r] / s[0]) if r < len(s) else None)
out = dict(net=tag, shift=shift, polish_L=Lp, polish_steps=tp,
           exact={f"{t:g}": stats(s_ex, t) for t in [1e-6, 1e-8, 1e-9, 1e-12]},
           gram={f"{t:g}": stats(s_gr, t) for t in [1e-6, 1e-7, 1e-8, 1e-9]},
           gram_abs_err_over_smax=float(np.max(np.abs(s_gr[:len(s_ex)] - s_ex)) / s_ex[0]))
b_s = np.sort(np.load(f"data/llc_traces/{tag}_grok.npz")["singular_values"])[::-1]
out["build_saved_gram"] = stats(b_s, 1e-6)
print(json.dumps(out, indent=1))
json.dump(out, open("skeptics/skeptic/gram_recheck.json", "w"), indent=1)
