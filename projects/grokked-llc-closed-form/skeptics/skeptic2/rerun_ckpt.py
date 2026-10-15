"""skeptic2: independent re-analysis of build's checkpoints.

For each data/checkpoints/p{p}_K{K}_s{seed}_{phase}.npz (W: 2p x K, V: p x K):
  - train split rebuilt from (p, frac, seed) with numpy default_rng(seed).permutation (build's convention,
    read from the run json); checked by re-computing train loss / acc against the json.
  - structure: per-neuron magnitude m_j = |v_j| |w_j|^2, K_eff (m_j >= 1% max), dominant DFT frequency
    of u_j, s_j, v_j, neurons per frequency.
  - rank of the Jacobian of centred train outputs (my own jac_centred from fourier_rank.py).
  - paper value K_eff(3p-1)/2, shift-corrected K_eff(3p-2)/2, cap p(N-1)/2, rank/2.
  - optional SGLD (my sgld2.py) with hyperparameters given on the command line or from calibration.json.

Usage: python rerun_ckpt.py CKPT_DIR [--sgld nbeta gamma crel steps chains] [--polish]
"""
import argparse
import glob
import json
import os
import re

import numpy as np

from fourier_rank import jac_centred, rank
from sgld2 import qnet_loss_grad, sgld_llc_qnet


def split(p, frac, seed):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(p * p)
    n_tr = int(round(frac * p * p))
    A, B = (g.ravel() for g in np.meshgrid(np.arange(p), np.arange(p), indexing="ij"))
    return A, B, perm[:n_tr], perm[n_tr:]


def dom_freq(x):
    """Dominant nonzero frequency of a length-p real vector and its power share (excluding DC)."""
    F = np.abs(np.fft.rfft(x - x.mean())) ** 2
    F[0] = 0
    k = int(F.argmax())
    return k, float(F[k] / (F.sum() + 1e-300))


def polish(W, V, A, B, Y, steps=5000, lr=1e-3, tol=1e-8):
    """Adam, wd=0, full batch, on the centred train loss (plan DECISIONS 2a)."""
    p = V.shape[0]
    th = [W[:p][None].copy(), W[p:][None].copy(), V[None].copy()]
    m = [np.zeros_like(t) for t in th]; v = [np.zeros_like(t) for t in th]
    for t in range(1, steps + 1):
        L, gU, gS, gV = qnet_loss_grad(*th, A, B, Y)
        if L[0] < tol:
            break
        for i, g in enumerate((gU, gS, gV)):
            m[i] = 0.9 * m[i] + 0.1 * g; v[i] = 0.999 * v[i] + 0.001 * g * g
            th[i] -= lr * (m[i] / (1 - 0.9**t)) / (np.sqrt(v[i] / (1 - 0.999**t)) + 1e-12)
    return np.concatenate([th[0][0], th[1][0]]), th[2][0], float(L[0]), t


def analyse(path, sgld=None, do_polish=False):
    mt = re.search(r"p(\d+)_K(\d+)_s(\d+)_(mem|grok)\.npz", os.path.basename(path))
    p, K, seed, phase = int(mt[1]), int(mt[2]), int(mt[3]), mt[4]
    meta = json.load(open(path.replace(f"_{phase}.npz", ".json")))
    z = np.load(path)
    W, V = z["W"], z["V"]
    A, B, tr, te = split(p, meta["frac"], meta.get("data_seed", seed))
    if "train_idx" in z.files:
        assert set(z["train_idx"].tolist()) == set(tr.tolist()), "split mismatch"
        tr, te = z["train_idx"], z["test_idx"]
    Y = np.eye(p)[(A + B) % p]

    def acc(W, V, idx):
        out = ((W[:p][A[idx]] + W[p:][B[idx]]) ** 2) @ V.T
        return float((out.argmax(1) == ((A + B) % p)[idx]).mean())

    L = float(qnet_loss_grad(W[:p][None], W[p:][None], V[None], A[tr], B[tr], Y[tr])[0][0])
    row = dict(file=os.path.basename(path), p=p, K=K, seed=seed, phase=phase, N=len(tr),
               train_loss=L, train_acc=acc(W, V, tr), test_acc=acc(W, V, te))
    if do_polish:
        W2, V2, Lp, steps = polish(W, V, A[tr], B[tr], Y[tr])
        row.update(polish_rel_move=float(np.sqrt(((W2 - W) ** 2).sum() + ((V2 - V) ** 2).sum()) /
                                         np.sqrt((W ** 2).sum() + (V ** 2).sum())),
                   polish_loss=Lp, polish_steps=steps, test_acc_polished=acc(W2, V2, te))
        W, V = W2, V2
    mag = np.linalg.norm(V, axis=0) * np.linalg.norm(W, axis=0) ** 2
    active = mag >= 0.01 * mag.max()
    Keff = int(active.sum())
    freqs, purity = [], []
    for j in np.where(active)[0]:
        ku, pu = dom_freq(W[:p, j]); ks, ps = dom_freq(W[p:, j]); kv, pv = dom_freq(V[:, j])
        freqs.append((ku, ks, kv)); purity.append(min(pu, ps, pv))
    same = [f[0] for f in freqs if f[0] == f[1] == f[2]]
    per_freq = np.bincount(same, minlength=p // 2 + 1)[1:].tolist() if same else []
    r, gap = rank(jac_centred(W[:p], W[p:], V, A[tr], B[tr]), tol=1e-7)
    row.update(K_eff=Keff, n_small=int(K - Keff), purity_min=float(min(purity)), purity_med=float(np.median(purity)),
               n_consistent_freq=len(same), neurons_per_freq=per_freq, n_freqs_used=int(sum(1 for c in per_freq if c)),
               rankJ=r, rank_gap=gap, lambda_paper=Keff * (3 * p - 1) / 2, lambda_shift=Keff * (3 * p - 2) / 2,
               lambda_cap=p * (len(tr) - 1) / 2, lambda_rank=r / 2,
               inside_hyp=bool(Keff * (3 * p - 2) < p * (len(tr) - 1)))
    if sgld:
        nbeta, gamma, crel, steps, chains = sgld
        Jn = jac_centred(W[:p], W[p:], V, A[tr], B[tr])
        h = np.linalg.svd(Jn, compute_uv=False) ** 2 / len(tr)  # Gauss-Newton eigenvalues of L_n
        hmax = float(h[0])
        eps = crel / (nbeta * hmax + gamma)  # plan's frozen rule (calibration.json)
        s = sgld_llc_qnet(W, V, A[tr], B[tr], Y[tr], eps=eps, gamma=gamma, nbeta=nbeta, steps=steps,
                          burn=steps // 5, chains=chains, seed=seed)
        a_i = nbeta * h + gamma
        row.update(sgld_lambda=s["lambda_hat"], sgld_se=s["se"], sgld_eps=eps, sgld_nbeta=nbeta, sgld_gamma=gamma,
                   sgld_steps=steps, sgld_chains=chains, sgld_diverged=s["diverged"],
                   lambda_sur=float((nbeta * h / (2 * a_i)).sum()),
                   lambda_sur_eps=float((nbeta * h / (2 * a_i * (1 - eps * a_i / 4))).sum()))
    return row


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt_dir")
    ap.add_argument("--sgld", nargs=5, type=float)
    ap.add_argument("--polish", action="store_true")
    ap.add_argument("--glob", default="*.npz")
    ap.add_argument("--out", default="rerun.jsonl")
    a = ap.parse_args()
    sg = None if a.sgld is None else (a.sgld[0], a.sgld[1], a.sgld[2], int(a.sgld[3]), int(a.sgld[4]))
    with open(a.out, "a") as f:
        for path in sorted(glob.glob(os.path.join(a.ckpt_dir, a.glob))):
            row = analyse(path, sg, a.polish)
            print(json.dumps({k: v for k, v in row.items() if k != "neurons_per_freq"}), flush=True)
            f.write(json.dumps(row) + "\n")
