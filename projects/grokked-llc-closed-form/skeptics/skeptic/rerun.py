"""skeptic: independent rerun of the LLC estimate on saved checkpoints.

Differences from build on purpose: polish with Levenberg-Marquardt (not Adam), my own Jacobian,
my own SGLD loop, my own chain seeds. Same frozen sampler setting (calibration.json):
nbeta=1e4, gamma=1, eps=0.2/(nbeta*h_max+gamma), 10k steps, 2k burn, 4 chains.
Grokked ckpts: full net. Memorising ckpts: first layer frozen (V only), as in Thm 5.2.
Usage: python rerun.py NAME [NAME ...]   (NAME like p11_K33_s0_grok); appends to rerun.jsonl
"""
import json, os, sys, time, zlib
import numpy as np
from common import loss, grad, targets, jacobian, sgld_llc, forward

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, "../../data/ckpt")
NB, G, CREL, STEPS, BURN, CHAINS = 1e4, 1.0, 0.2, 10000, 2000, 4


def lm_polish(W, V, a, b, p, Y, freeze_W, tol=1e-13, iters=60):
    K = W.shape[1]
    W, V = W.copy(), V.copy()
    mu = 1e-6
    L = loss(W, V, a, b, p, Y)
    for it in range(iters):
        if L < tol:
            break
        J = jacobian(W, V, a, b, p, freeze_W)
        F, _, _ = forward(W, V, a, b, p)
        R = F - Y; R = R - R.mean(0, keepdims=True)
        r = R.ravel()                      # (i,k) ordering matches jacobian rows
        A = J.T @ J; g = J.T @ r
        while True:
            d = -np.linalg.solve(A + mu * np.eye(len(g)), g)
            if freeze_W:
                W2, V2 = W, V + d.reshape(p, K)
            else:
                W2 = W + d[: 2 * p * K].reshape(2 * p, K); V2 = V + d[2 * p * K:].reshape(p, K)
            L2 = loss(W2, V2, a, b, p, Y)
            if L2 < L:
                W, V, L = W2, V2, L2; mu = max(mu / 10, 1e-12); break
            mu *= 10
            if mu > 1e6:
                return W, V, L
    return W, V, L


def run(name):
    z = np.load(os.path.join(D, name + ".npz"))
    p, K = int(z["p"]), int(z["K"]); phase = name.split("_")[-1]
    freeze_W = phase == "mem"
    a, b = z["train_idx"] // p, z["train_idx"] % p; Y = targets(a, b, p)
    ta, tb = z["test_idx"] // p, z["test_idx"] % p
    W0, V0 = z["W"], z["V"]
    t0 = time.time()
    L_ck = loss(W0, V0, a, b, p, Y)
    W, V, L = lm_polish(W0, V0, a, b, p, Y, freeze_W)
    th0 = np.concatenate([W0.ravel(), V0.ravel()]); th = np.concatenate([W.ravel(), V.ravel()])
    drift = np.linalg.norm(th - th0) / np.linalg.norm(th0)
    # test acc, raw and with train offset added back
    Ftr, _, _ = forward(W, V, a, b, p); Fte, _, _ = forward(W, V, ta, tb, p)
    off = Y.mean(0) - Ftr.mean(0)
    yte = (ta + tb) % p
    acc_raw = float((Fte.argmax(1) == yte).mean()); acc_off = float(((Fte + off).argmax(1) == yte).mean())
    # spectrum
    J = jacobian(W, V, a, b, p, freeze_W)
    N = len(a)
    s = np.linalg.svd(J, compute_uv=False); h = s ** 2 / N
    rank = int((s > s[0] * 1e-9).sum())
    hmax = h[0]
    lam_quad = 0.5 * float(np.sum(NB * h / (NB * h + G)))
    n_res = int((h > G / NB).sum())
    # K_eff (plan rule) + paper formula
    mag = np.linalg.norm(V, axis=0) * np.linalg.norm(W, axis=0) ** 2
    K_eff = int((mag >= 0.01 * mag.max()).sum())
    if freeze_W:
        H = forward(W, V, a, b, p)[2]; Hc = H - H.mean(0)
        l = int(np.linalg.matrix_rank(Hc)); lam_paper = p * min(l, K) / 2
    else:
        l = None; lam_paper = K_eff * (3 * p - 1) / 2
    lam_shift = None if freeze_W else K_eff * (3 * p - 2) / 2
    eps = CREL / (NB * hmax + G)
    chains = []
    for c in range(CHAINS):
        rng = np.random.default_rng(10_000 + 97 * c + zlib.crc32(name.encode()) % 1000)
        lam, trace, _ = sgld_llc(W, V, a, b, p, Y, NB, G, eps, STEPS, BURN, rng, freeze_W)
        # stationarity: compare first and last third of post-burn trace
        n3 = len(trace) // 3
        chains.append(dict(lam=float(lam), first_third=float(NB * (trace[:n3].mean() - L)),
                           last_third=float(NB * (trace[-n3:].mean() - L))))
    lams = np.array([c["lam"] for c in chains])
    rec = dict(name=name, p=p, K=K, phase=phase, N=N, L_ckpt=float(L_ck), L_polished=float(L), drift=float(drift),
               acc_raw=acc_raw, acc_offset=acc_off, K_eff=K_eff, l=l, rank=rank, rank_over_2=rank / 2,
               cap=p * (N - 1) / 2, lam_paper=lam_paper, lam_shift=lam_shift, lam_quad=lam_quad, n_resolved=n_res,
               hmax=float(hmax), eps=eps, lam_hat=float(lams.mean()), lam_se=float(lams.std(ddof=1) / np.sqrt(len(lams))),
               chains=chains, ratio_paper=float(lams.mean() / lam_paper), ratio_rank=float(lams.mean() / (rank / 2)),
               secs=time.time() - t0)
    with open(os.path.join(HERE, "rerun.jsonl"), "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(json.dumps({k: rec[k] for k in ["name", "L_polished", "drift", "acc_offset", "rank", "lam_paper", "lam_quad", "lam_hat", "lam_se", "ratio_paper", "secs"]}), flush=True)


if __name__ == "__main__":
    for n in sys.argv[1:]:
        run(n)
