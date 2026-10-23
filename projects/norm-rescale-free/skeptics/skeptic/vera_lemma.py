"""Vera (1023): own numeric checks of the norm-rescale lemmas on the REAL Qwen2.5-1.5B-Instruct final RMSNorm + tied unembedding,
CPU float64, clean 1021 readout-28 (pre-norm) country-token states. No build code read.

Claims checked:
 L1  scale invariance: softmax(W (g*RMSNorm(c h))) == softmax(W (g*RMSNorm(h))) for c>0 (up to eps); c<0 is NOT free.
 L2  Euler: h . grad_h KL(p_clean || p(h)) = 0 at any h (KL is 0-homogeneous in h), relative to |grad|*|h|.
 L3  Hessian null: at h = h_clean, Hess_h KL . h = 0 (h is a null vector), vs Hess . random unit.
 L4  norm restore h'' = h' |h| / |h'| leaves the output and the post-norm (readout-29) state unchanged.
 L5  first-order R of an activation-space gradient step along -w (probe gradient): R = cos^2(w, h_hat) per statement;
     pooled over twins on the mean gap: R0 = sum (w.h_hat)^2 / (n |w|^2). Numeric finite-step check.
"""
import json
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file

HERE = Path(__file__).resolve().parent
OLD = Path.home() / "work/theory-1021-full-stack-evader/kran-research/projects/full-stack-evader"
snap = next(Path.home().glob(".cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/*/model.safetensors"))
sd = load_file(str(snap))
g = sd["model.norm.weight"].double()
W = sd["model.embed_tokens.weight"].double()  # tied
del sd
EPS = 1e-6
torch.set_num_threads(4)

rows = json.load(open(OLD / "data/ref/split.json"))
yf = np.array([1 - r["true"] for r in rows]); split = np.array([r["split"] for r in rows])
T = np.load(OLD / "data/acts/clean/tok.npy", mmap_mode="r"); o = np.load(OLD / "data/acts/clean/off.npy")
H28 = np.asarray(T[o[1:] - 2][:, 28], np.float64)
H29 = np.asarray(T[o[1:] - 2][:, 29], np.float64)
BP = np.load(OLD / "data/fits/panel.npz")
w = (BP["tok28__w"] / BP["tok28__sd"]).astype(np.float64)


def post(h):
    return h / torch.sqrt((h * h).mean(-1, keepdim=True) + EPS) * g


def logp(h):
    return torch.log_softmax(post(h) @ W.T, -1)


def kl(lp0, h):
    return (lp0.exp() * (lp0 - logp(h))).sum(-1)


rng = np.random.default_rng(1023)
idx = rng.choice(len(H28), 24, replace=False)
h = torch.from_numpy(H28[idx])
lp0 = logp(h)
out = {}

# L1
out["L1_maxKL_c"] = {str(c): float(kl(lp0, c * h).max()) for c in (0.3, 0.55, 1.8, 3.0)}
out["L1_KL_c_minus1_median"] = float(kl(lp0, -h).median())
# eps-induced deviation of the post-norm state at c=0.3 (relative)
out["L1_post_rel_dev_c0.3"] = float(((post(0.3 * h) - post(h)).norm(dim=-1) / post(h).norm(dim=-1)).max())
# stored readout 29 equals g*RMSNorm(readout 28)? (fp16 storage)
r29 = torch.from_numpy(H29[idx])
out["readout29_vs_post28_rel"] = float(((post(h) - r29).norm(dim=-1) / r29.norm(dim=-1)).max())

# L2: at a perturbed point h1 = h + tangential+radial noise, gradient of KL(p_clean || p(h1)) is orthogonal to h1
eu = []
for i in range(8):
    h1 = (h[i] + 3.0 * torch.randn(1536, dtype=torch.float64, generator=torch.Generator().manual_seed(i))).requires_grad_(True)
    k = kl(lp0[i:i + 1], h1[None])[0]
    (gr,) = torch.autograd.grad(k, h1)
    eu.append(float((gr @ h1) / (gr.norm() * h1.norm())))
out["L2_cos_grad_h_at_perturbed_points"] = eu

# L3: Hessian-vector products at clean point via double backprop
hv_rad, hv_rnd = [], []
for i in range(6):
    hc = h[i].clone().requires_grad_(True)
    k = kl(lp0[i:i + 1], hc[None])[0]
    (gr,) = torch.autograd.grad(k, hc, create_graph=True)
    u_rad = (h[i] / h[i].norm())
    u_rnd = torch.randn(1536, dtype=torch.float64, generator=torch.Generator().manual_seed(100 + i)); u_rnd /= u_rnd.norm()
    u_tan = u_rnd - (u_rnd @ u_rad) * u_rad; u_tan /= u_tan.norm()
    (Hr,) = torch.autograd.grad(gr @ u_rad, hc, retain_graph=True)
    (Ht,) = torch.autograd.grad(gr @ u_tan, hc)
    hv_rad.append(float(u_rad @ Hr)); hv_rnd.append(float(u_tan @ Ht))
out["L3_curv_radial"] = hv_rad
out["L3_curv_random_tangential"] = hv_rnd

# L4: norm restore of an arbitrary edit
d = torch.randn_like(h) * 8.0 - 0.4 * h
hp = h + d
hpp = hp * (h.norm(dim=-1, keepdim=True) / hp.norm(dim=-1, keepdim=True))
out["L4_maxKL_restored_vs_attacked"] = float(kl(logp(hp), hpp).max())
out["L4_post_rel_dev"] = float(((post(hpp) - post(hp)).norm(dim=-1) / post(hp).norm(dim=-1)).max())

# L5: R of a small step along -w (false) / +w (true twin) on eval twins, finite step vs cos^2 formula
ev = np.where(split == "eval")[0]
Hn = H28[ev] / np.linalg.norm(H28[ev], axis=1, keepdims=True)
cos = Hn @ (w / np.linalg.norm(w))
sign = np.where(yf[ev] == 1, -1.0, 1.0)  # false pushed down, true pushed up


def score(X):
    return X @ w


Hc = H28[ev]
for eta in (1e-3, 1e-1, 1.0):
    Ha = Hc + eta * sign[:, None] * w[None, :] / np.linalg.norm(w)
    Hr = Ha * (np.linalg.norm(Hc, axis=1) / np.linalg.norm(Ha, axis=1))[:, None]
    f = yf[ev] == 1
    gc = score(Hc)[f].mean() - score(Hc)[~f].mean()
    ga = score(Ha)[f].mean() - score(Ha)[~f].mean()
    gr = score(Hr)[f].mean() - score(Hr)[~f].mean()
    out[f"L5_R_finite_eta{eta}"] = float((gr - ga) / (gc - ga))
out["L5_R0_formula_mean_cos2"] = float((cos ** 2).mean())
out["L5_cos2_median_false"] = float(np.median(cos[yf[ev] == 1] ** 2))
out["L5_cos2_median_true"] = float(np.median(cos[yf[ev] == 0] ** 2))
out["L5_cos_mean_false_true"] = [float(cos[yf[ev] == 1].mean()), float(cos[yf[ev] == 0].mean())]

# geometry of the radial direction (shared?)
U = H28 / np.linalg.norm(H28, axis=1, keepdims=True)
s = np.linalg.svd(U, compute_uv=False)
out["geom_top1_energy"] = float(s[0] ** 2 / (s ** 2).sum())
iu = np.triu_indices(400, 1); sub = U[rng.choice(len(U), 400, replace=False)]
out["geom_pairwise_cos_median"] = float(np.median((sub @ sub.T)[iu]))
print(json.dumps(out, indent=1))
json.dump(out, open(HERE / "vera_lemma.json", "w"), indent=1)
