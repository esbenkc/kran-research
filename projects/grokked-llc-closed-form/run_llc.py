"""T3: estimate the LLC at every saved checkpoint and write data/llc.csv (+ data/llc_traces/*.npz).

Per checkpoint (data/ckpt/p{p}_K{K}_s{seed}_{mem|grok}.npz):
 1. Polish: Adam lr 1e-3, NO weight decay, full batch, until L_n < 1e-8 or 5000 steps. The LLC is a
    property of the data loss L_n at a (near-)minimum, so we remove the L2 offset first. A checkpoint is
    dropped if polishing moves test accuracy across 0.5 (it would change phase).
 2. At the polished point: K_eff (|v_j||w_j|^2 >= 1% of max), Fourier purity, rank of the Jacobian J of the
    centred train outputs (rankJ), l = rank of the centred train features F~ (Thm 5.2).
 3. SGLD (Lau et al., arXiv 2308.12108) with the frozen calibration setting (data/calibration.json):
      grok: full width-K net (headline) and the pruned net (inactive neurons deleted).
      mem : V only, W frozen (Thm 5.2 is the frozen-feature model; headline) and the full net (data only).
    Step size rule from calibration: eps = crel / (nbeta * h_max + gamma), h_max = s_max(J)^2 / N.
References per row:
  lambda_paper  grok: K_eff (3p-1)/2 (Cullen et al. v3, Cor 5.4 + Remark 5.5); mem: p min(l,K)/2 (Thm 5.2)
  lambda_shift  K_eff (3p-2)/2 (one extra shift symmetry per neuron on one-hot inputs)
  lambda_rank   rankJ/2 (rank reference; equals lambda only at smooth points)
  lambda_cap    p(N-1)/2 (no LLC of the train loss can exceed this)
"""
from __future__ import annotations

import csv
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

import qnet

HERE = Path(__file__).parent
CKPT = HERE / "data" / "ckpt"
OUT_CSV = Path(os.environ.get("OUT_CSV", HERE / "data" / "llc.csv"))
TRACES = Path(os.environ.get("TRACES", HERE / "data" / "llc_traces"))
CALIB = Path(os.environ.get("CALIB", HERE / "data" / "calibration.json"))

COLS = ["p", "K", "seed", "phase", "frac", "N", "step", "lambda_hat", "lambda_se", "lambda_hat_pruned",
        "lambda_hat_pruned_se", "lambda_hat_full_mem", "K_eff", "K_eff_pre", "l_rank", "rankJ",
        "rank_gap", "lambda_paper", "lambda_shift", "lambda_rank", "lambda_quad", "lambda_cap",
        "inside_hypothesis", "rank_method", "n_resolved",
        "eps", "eps_pruned", "gamma", "nbeta", "crel", "chains", "steps", "burn",
        "train_loss_ckpt", "train_loss", "test_acc_ckpt", "test_acc", "polish_rel_change", "polish_steps",
        "purity_min_active", "purity_median_active", "n_freqs_active", "dropped"]


def polish(W, V, X, Y, lr=1e-3, max_steps=5000, tol=1e-8):
    W, V = W.copy(), V.copy()
    params = [W, V]
    m = [np.zeros_like(x) for x in params]
    v = [np.zeros_like(x) for x in params]
    b1, b2, e = 0.9, 0.999, 1e-12
    L = qnet.loss(W, V, X, Y)
    t = 0
    for t in range(1, max_steps + 1):
        L, gW, gV = qnet.loss_grad(params[0], params[1], X, Y)
        if L < tol:
            break
        for i, g in enumerate((gW, gV)):
            m[i] = b1 * m[i] + (1 - b1) * g
            v[i] = b2 * v[i] + (1 - b2) * g * g
            params[i] = params[i] - lr * (m[i] / (1 - b1 ** t)) / (np.sqrt(v[i] / (1 - b2 ** t)) + e)
    W, V = params
    return W, V, qnet.loss(W, V, X, Y), t


GRAM_ABOVE = 6000  # min(J.shape) above this: singular values from eigvalsh of the smaller Gram matrix


def svals(J):
    """Singular values, descending. Exact SVD for small J; Gram eigenvalues for big J (resolution ~1e-7 s_max)."""
    if min(J.shape) <= GRAM_ABOVE:
        return np.linalg.svd(J, compute_uv=False)
    G = J.T @ J if J.shape[1] <= J.shape[0] else J @ J.T
    e = np.linalg.eigvalsh(G)[::-1]
    return np.sqrt(np.clip(e, 0.0, None))


def rank_of(s, rtol=1e-9):
    r = int(np.sum(s > rtol * s[0]))
    gap = float(s[r - 1] / s[r]) if r < len(s) and s[r] > 0 else float("inf")
    return r, gap


def quad_llc(s, N, gamma, nbeta):
    """What an exact sampler returns for the localised quadratic (Gauss-Newton) model at this point:
    sum_i 1/2 h_i / (h_i + gamma/nbeta), h_i = s_i^2 / N. Shows how much of lambda_hat's gap to rank/2
    comes from the localisation (flat directions with h_i << gamma/nbeta are not counted)."""
    h = s ** 2 / N
    return float(np.sum(0.5 * h / (h + gamma / nbeta)))


def run_ckpt(path_str):
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    path = Path(path_str)
    cal = json.loads(CALIB.read_text())
    crel, gamma, nbeta = cal["crel"], cal["gamma"], cal["nbeta"]
    steps, burn, chains = cal["steps"], cal["burn"], cal["chains"]
    z = np.load(path)
    p, K, seed = int(z["p"]), int(z["K"]), int(z["seed"])
    phase = path.stem.split("_")[-1]
    Xall, Yall = qnet.full_table(p)
    tr, te = z["train_idx"], z["test_idx"]
    X, Y, Xte, Yte = Xall[tr], Yall[tr], Xall[te], Yall[te]
    N = len(tr)
    W0, V0 = z["W"], z["V"]
    pre = qnet.neuron_stats(W0, V0)
    test_acc_ckpt = qnet.accuracy(W0, V0, Xte, Yte)
    W, V, Lp, nsteps = polish(W0, V0, X, Y)
    test_acc = qnet.accuracy(W, V, Xte, Yte)
    th0 = np.concatenate([W0.ravel(), V0.ravel()])
    th = np.concatenate([W.ravel(), V.ravel()])
    row = dict(p=p, K=K, seed=seed, phase=phase, frac=float(z["frac"]), N=N, step=int(z["step"]),
               K_eff_pre=pre["K_eff"], train_loss_ckpt=float(z["train_loss"]), train_loss=float(Lp),
               test_acc_ckpt=test_acc_ckpt, test_acc=test_acc,
               polish_rel_change=float(np.linalg.norm(th - th0) / np.linalg.norm(th0)), polish_steps=nsteps,
               crel=crel, gamma=gamma, nbeta=nbeta, chains=chains, steps=steps, burn=burn)
    row["dropped"] = int((test_acc_ckpt >= 0.5) != (test_acc >= 0.5))
    st = qnet.neuron_stats(W, V)
    act = st["active"]
    row.update(K_eff=st["K_eff"], purity_min_active=float(st["purity"][act].min()),
               purity_median_active=float(np.median(st["purity"][act])),
               n_freqs_active=int(len(set(st["freq"][act].tolist()))))
    J = qnet.jacobian(W, V, X)
    s = svals(J)
    del J
    rank_method = "svd" if min(N * p, W.size + V.size) <= GRAM_ABOVE else "gram"
    rJ, gap = rank_of(s, 1e-9 if rank_method == "svd" else 1e-6)
    F = (X @ W) ** 2
    Fc = F - F.mean(0, keepdims=True)
    sF = np.linalg.svd(Fc, compute_uv=False)
    l_rank, _ = rank_of(sF)
    row.update(rankJ=rJ, rank_gap=gap, rank_method=rank_method, l_rank=l_rank, lambda_rank=rJ / 2,
               lambda_cap=p * (N - 1) / 2, lambda_shift=st["K_eff"] * (3 * p - 2) / 2,
               n_resolved=int(np.sum(s ** 2 / N > gamma / nbeta)))
    row["inside_hypothesis"] = int(st["K_eff"] * (3 * p - 2) < p * (N - 1))
    tr_out = {}

    def sgld(Wx, Vx, freeze, tag, s_max, sd):
        h_max = s_max ** 2 / N
        eps = crel / (nbeta * h_max + gamma)
        o = qnet.sgld_llc(Wx, Vx, X, Y, eps=eps, gamma=gamma, n_steps=steps, n_burn=burn, n_chains=chains,
                          seed=sd, nbeta=nbeta, freeze_W=freeze)
        if o["trace"] is not None:
            tr_out[tag] = o["trace"]
        c = np.array(o["chains"]) if o["chains"] else np.array([np.nan])
        se = float(c.std(ddof=1) / np.sqrt(len(c))) if len(c) > 1 else float("nan")
        return o["llc"], se, eps

    sd = 10_000 + 100 * p + K + seed
    if phase == "grok":
        row["lambda_paper"] = st["K_eff"] * (3 * p - 1) / 2
        row["lambda_quad"] = quad_llc(s[:rJ], N, gamma, nbeta)
        lam, se, eps = sgld(W, V, False, "full", s[0], sd)
        row.update(lambda_hat=lam, lambda_se=se, eps=eps)
        Wp, Vp = W[:, act], V[:, act]
        if act.all():   # nothing to prune: the pruned net is the full net
            lam_p, se_p, eps_p = lam, se, eps
        else:
            sp = svals(qnet.jacobian(Wp, Vp, X))[0]
            lam_p, se_p, eps_p = sgld(Wp, Vp, False, "pruned", sp, sd + 1)
        row.update(lambda_hat_pruned=lam_p, lambda_hat_pruned_se=se_p, eps_pruned=eps_p)
        row["lambda_hat_full_mem"] = float("nan")
    else:
        row["lambda_paper"] = p * min(l_rank, K) / 2
        sV = np.repeat(sF, p)   # J_V = F~ (x) I_p: singular values of F~, each p times
        rV, _ = rank_of(sV)
        row["lambda_quad"] = quad_llc(sV[:rV], N, gamma, nbeta)
        lam, se, eps = sgld(W, V, True, "vonly", sV[0], sd)
        row.update(lambda_hat=lam, lambda_se=se, eps=eps)
        lam_f, _, eps_f = sgld(W, V, False, "full", s[0], sd + 1)
        row.update(lambda_hat_full_mem=lam_f, lambda_hat_pruned=float("nan"),
                   lambda_hat_pruned_se=float("nan"), eps_pruned=eps_f)
    TRACES.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(TRACES / f"{path.stem}.npz", singular_values=s, **tr_out)
    return row


def _log(row):
    print(row["p"], row["K"], row["seed"], row["phase"], "K_eff", row["K_eff"], "rankJ", row["rankJ"],
          "paper %.1f hat %.1f quad %.1f rank/2 %.1f" % (row["lambda_paper"], row["lambda_hat"],
                                                         row["lambda_quad"], row["lambda_rank"]), flush=True)


def _append(row):
    new = not OUT_CSV.exists()
    with open(OUT_CSV, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        if new:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in COLS})


def main(paths):
    """Small checkpoints first (8 workers), then big ones (3 workers, memory). Rows are appended to llc.csv
    as they finish, then the file is rewritten sorted."""
    if OUT_CSV.exists():
        OUT_CSV.unlink()
    size = lambda q: int(np.load(q)["K"]) * int(np.load(q)["p"]) ** 2
    paths = sorted(paths, key=size)
    small = [q for q in paths if size(q) < 23 * 92 * 2]
    big = [q for q in paths if size(q) >= 23 * 92 * 2]
    rows = []
    for group, nw in ((small, int(os.environ.get("WORKERS", "8"))), (big, int(os.environ.get("BIG_WORKERS", "3")))):
        with ProcessPoolExecutor(max_workers=nw) as ex:
            for row in ex.map(run_ckpt, [str(x) for x in group]):
                rows.append(row)
                _append(row)
                _log(row)
    rows.sort(key=lambda r: (r["phase"], r["p"], r["K"], r["seed"]))
    OUT_CSV.unlink()
    for r in rows:
        _append(r)
    print("wrote", OUT_CSV, len(rows), "rows")


if __name__ == "__main__":
    ps = [Path(a) for a in sys.argv[1:]] or sorted(CKPT.glob("*.npz"))
    main(ps)
