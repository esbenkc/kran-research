"""skeptic3: independent re-run on build's saved checkpoints (data/checkpoints/*_{mem,grok}.npz).

Per checkpoint, with my own code (rank_check.py, sgld.py), no build code imported:
  * train loss L0 (centred, no L2), K_eff (|v_j| |w_j|^2 >= 1% of max), dominant frequency per active neuron,
  * rank J / 2 (rank reference), lambda_paper = K_eff(3p-1)/2, lambda_shift = K_eff(3p-2)/2,
    lambda_cap = p(N-1)/2, lambda_freq = sum_k min(m_k(3p-2), 20p-36)/2 (my per-frequency bound),
  * SGLD lambda_hat with my frozen rule: gamma=1, eps = 0.1/(nbeta*lmax(J^T J/N)), 20000 steps, 5000 burn,
    4 chains; null test = nbeta sweep {1e3,1e4,1e5} and two chain-seed sets.
Grok checkpoints: full net. Mem checkpoints: full net AND V-only (Thm 5.2 model).
Usage: rerun.py <ckpt_dir> <out.jsonl> [glob]
"""
import glob
import json
import os
import sys
import time

import numpy as np

from rank_check import jacobian, rank
from sgld import llc, loss


def load(path):
    z = np.load(path, allow_pickle=True)
    W, V = z["W"], z["V"]
    p = V.shape[0]
    if "train_idx" in z.files:
        tr, te = z["train_idx"], z["test_idx"]
    else:  # early ckpts: rebuild the split from the run json (same rule as qnet.modadd_data)
        meta = json.load(open(path.rsplit("_", 1)[0] + ".json"))
        perm = np.random.default_rng(meta["seed"]).permutation(p * p)
        n = int(round(meta["frac"] * p * p))
        tr, te = np.sort(perm[:n]), np.sort(perm[n:])
    U, S = W[:p].T.copy(), W[p:].T.copy()
    return p, U, S, V.copy(), tr, te, z


def targets(p, idx):
    A, B = idx // p, idx % p
    Y = np.eye(p)[(A + B) % p].T  # p x N
    return A, B, Y


def dominant_freq(x):
    p = len(x)
    F = np.abs(np.fft.rfft(x - x.mean())) ** 2
    k = int(np.argmax(F[1:]) + 1)
    return k, float(F[k] / F[1:].sum())

def gn_lmax(U, S, V, A, B, iters=200, seed=0):
    """Top eigenvalue of J^T J / N by power iteration with analytic J v and J^T r (centred outputs)."""
    p = U.shape[1]
    N = len(A)
    OA, OB = np.eye(p)[A], np.eye(p)[B]
    H = U[:, A] + S[:, B]
    rng = np.random.default_rng(seed)
    x = [rng.standard_normal(U.shape), rng.standard_normal(S.shape), rng.standard_normal(V.shape)]
    lam = 0.0
    for _ in range(iters):
        nrm = np.sqrt(sum((a ** 2).sum() for a in x))
        x = [a / nrm for a in x]
        dH = x[0][:, A] + x[1][:, B]
        dF = x[2] @ H ** 2 + V @ (2 * H * dH)
        dF -= dF.mean(axis=1, keepdims=True)
        gH = 2 * H * (V.T @ dF)
        y = [gH @ OA / N, gH @ OB / N, dF @ (H ** 2).T / N]
        lam = sum((a * b).sum() for a, b in zip(x, y))
        x = y
    return float(lam)



def analyse(path, runs=((1e3, 0), (1e4, 0), (1e5, 0), (1e4, 1)), steps=20000, burn=5000, chains=3, gamma=1.0):
    p, U, S, V, tr, te, z = load(path)
    A, B, Y = targets(p, tr)
    At, Bt, Yt = targets(p, te)
    N = len(A)
    K = U.shape[0]
    L0 = loss(U, S, V, A, B, Y)
    F_te = V @ ((U[:, At] + S[:, Bt]) ** 2)
    test_acc = float((F_te.argmax(0) == Yt.argmax(0)).mean()) if len(te) else float("nan")
    mag = np.linalg.norm(V, axis=0) * (np.linalg.norm(U, axis=1) ** 2 + np.linalg.norm(S, axis=1) ** 2)
    active = mag >= 0.01 * mag.max()
    K_eff = int(active.sum())
    freqs = [dominant_freq(U[j])[0] for j in range(K)]
    purity = [dominant_freq(U[j])[1] for j in range(K)]
    m = {}
    for j in np.where(active)[0]:
        m[freqs[j]] = m.get(freqs[j], 0) + 1
    big = p * N * 3 * K * p * 8 > float(os.environ.get("SK3_MAX_J_BYTES", 2e9))  # dense J over 2 GB: skip the spectrum, power-iterate h_max only
    if not big:
        J = jacobian(U, S, V, A, B)
        sv = np.linalg.svd(J, compute_uv=False)
        del J
        tol = sv.max() * max(p * N, 3 * K * p) * np.finfo(float).eps * 10
        rJ = int((sv > tol).sum())
        h = sv ** 2 / N  # Gauss-Newton Hessian eigenvalues of L_N (exact Hessian at a zero-residual point)
        lmax = float(h.max())
    else:
        lmax = gn_lmax(U, S, V, A, B)
        sv, rJ, h = np.array([1.0, 0.0]), -1, None
    row = dict(file=os.path.basename(path), p=p, K=K, N=N,
               phase="grok" if "_grok" in os.path.basename(path) else "mem",
               L0=float(L0), test_acc=test_acc, K_eff=K_eff, n_freq=len(m),
               neurons_per_freq=dict(sorted((int(k), int(v)) for k, v in m.items())),
               median_purity=float(np.median(np.array(purity)[active])),
               rankJ=rJ, lambda_rank=rJ / 2 if rJ >= 0 else None, lambda_paper=K_eff * (3 * p - 1) / 2,
               lambda_shift=K_eff * (3 * p - 2) / 2, lambda_cap=p * (N - 1) / 2,
               lambda_freq=min(sum(min(v * (3 * p - 2), 20 * p - 36) for v in m.values()), p * (N - 1)) / 2,
               lmax=lmax, sv_gap=float(sv[rJ - 1] / sv[rJ]) if 0 <= rJ < len(sv) else None,
               sgld={})
    for nb, sd in runs:
        t = time.time()
        r = llc(U, S, V, A, B, Y, nbeta=nb, eps=0.1 / (nb * lmax), gamma=gamma, steps=steps, burn=burn,
                chains=chains, seed=sd)
        row["sgld"][f"full_nb{nb:g}_seed{sd}"] = dict(lambda_hat=r["lambda_hat"], chain_se=r["chain_se"],
                                                      per_chain=r["per_chain"], secs=time.time() - t)
        if h is not None:  # Gaussian (quadratic-approx.) value at the same nbeta, gamma: 1/2 sum nb h/(nb h+gamma)
            row[f"lambda_quad_nb{nb:g}"] = float(0.5 * (nb * h / (nb * h + gamma)).sum())
            row[f"n_dirs_h_gt_gamma_over_nb{nb:g}"] = int((nb * h > gamma).sum())
    if row["phase"] == "mem":
        F = (U[:, A] + S[:, B]) ** 2
        Ft = F - F.mean(axis=1, keepdims=True)
        l = int(np.linalg.matrix_rank(Ft))
        row["l_rank"] = l
        row["lambda_thm52"] = p * min(l, K) / 2
        lmaxV = float(np.linalg.eigvalsh(Ft @ Ft.T / N).max())
        nb = 1e4
        r = llc(U, S, V, A, B, Y, nbeta=nb, eps=0.1 / (nb * lmaxV), gamma=1.0, steps=steps, burn=burn,
                chains=chains, seed=0, freeze_W=True)
        row["sgld"]["Vonly_nb1e4_seed0"] = dict(lambda_hat=r["lambda_hat"], chain_se=r["chain_se"])
    return row


if __name__ == "__main__":
    ckdir, out = sys.argv[1], sys.argv[2]
    pat = sys.argv[3] if len(sys.argv) > 3 else "*_*.npz"
    done = set()
    if os.path.exists(out):
        done = {json.loads(l)["file"] for l in open(out)}
    for f in sorted(glob.glob(os.path.join(ckdir, pat))):
        if os.path.basename(f) in done:
            continue
        if "_grok" not in f and "_mem" not in f:
            continue
        row = analyse(f)
        with open(out, "a") as fh:
            fh.write(json.dumps(row) + "\n")
        s = row["sgld"]
        print(row["file"], "L0=%.1e" % row["L0"], "Keff", row["K_eff"], "freqs", row["neurons_per_freq"],
              "rank/2", row["lambda_rank"], "paper", row["lambda_paper"], "freqbound", row["lambda_freq"],
              "cap", row["lambda_cap"], {k: round(v["lambda_hat"], 1) for k, v in s.items()}, flush=True)
