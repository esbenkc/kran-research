"""skeptic null tests. Question: is the ~0.7x reading at grokked checkpoints a property of
the grokked (Fourier) point, or would ANY zero-loss point of the same network read low with
this sampler? Known answer at a generic point: lambda = rank(J)/2 exactly (smooth point).

 N1 random-init planted: random W,V (scale matched to the grokked ckpt), targets := its own
    outputs on the same train set. Zero loss, generic, lambda = rank/2.
 N2 trained-generic: student trained from random init (Adam, no weight decay) to fit a random
    generic teacher of the same width on the same train set, then LM-polished. A trained point
    with no Fourier structure.
 N3 chain-seed null: rerun SGLD at the grokked ckpt with 4 fresh chain seeds (seed set B).
Same frozen sampler setting as rerun.py.
"""
import json, os, sys
import numpy as np
from common import loss, grad, targets, jacobian, sgld_llc, forward
from rerun import lm_polish, NB, G, CREL, STEPS, BURN, CHAINS, D, HERE


def spectrum(W, V, a, b, p):
    s = np.linalg.svd(jacobian(W, V, a, b, p), compute_uv=False)
    h = s ** 2 / len(a)
    return h, int((s > s[0] * 1e-9).sum())


def sgld(W, V, a, b, p, Y, seed0):
    h, rank = spectrum(W, V, a, b, p)
    eps = CREL / (NB * h[0] + G)
    lams = [sgld_llc(W, V, a, b, p, Y, NB, G, eps, STEPS, BURN, np.random.default_rng(seed0 + c))[0] for c in range(CHAINS)]
    return dict(rank=rank, rank_over_2=rank / 2, lam_quad=0.5 * float(np.sum(NB * h / (NB * h + G))),
                frac_below_res=float((h[:rank] < G / NB).mean()), decades=float(np.log10(h[0] / h[rank - 1])),
                lam_hat=float(np.mean(lams)), lam_se=float(np.std(lams, ddof=1) / 2), ratio_rank=float(np.mean(lams) / (rank / 2)))


def adam_fit(W, V, a, b, p, Y, steps=40000, lr=3e-3, seed=0):
    m = [np.zeros_like(W), np.zeros_like(V)]; v = [np.zeros_like(W), np.zeros_like(V)]
    b1, b2 = 0.9, 0.98
    for t in range(1, steps + 1):
        gs = grad(W, V, a, b, p, Y)
        for i, gg in enumerate(gs):
            m[i] = b1 * m[i] + (1 - b1) * gg; v[i] = b2 * v[i] + (1 - b2) * gg ** 2
        W = W - lr * (m[0] / (1 - b1 ** t)) / (np.sqrt(v[0] / (1 - b2 ** t)) + 1e-12)
        V = V - lr * (m[1] / (1 - b1 ** t)) / (np.sqrt(v[1] / (1 - b2 ** t)) + 1e-12)
    return W, V


def main(name):
    z = np.load(os.path.join(D, name + ".npz"))
    p, K = int(z["p"]), int(z["K"])
    a, b = z["train_idx"] // p, z["train_idx"] % p
    Wg, Vg = z["W"], z["V"]
    sw = np.median(np.linalg.norm(Wg, axis=0)); sv = np.median(np.linalg.norm(Vg, axis=0))
    rng = np.random.default_rng(123)
    out = dict(name=name, p=p, K=K, N=len(a), paper=K * (3 * p - 1) / 2, shift=K * (3 * p - 2) / 2)
    # N1 planted
    Wt = rng.standard_normal((2 * p, K)); Wt *= sw / np.linalg.norm(Wt, axis=0)
    Vt = rng.standard_normal((p, K)); Vt *= sv / np.linalg.norm(Vt, axis=0)
    Yt = forward(Wt, Vt, a, b, p)[0]
    out["N1_planted"] = sgld(Wt, Vt, a, b, p, Yt, 500)
    print("N1", out["N1_planted"], flush=True)
    # N2 trained generic student on the planted teacher
    Ws = rng.standard_normal((2 * p, K)) / np.sqrt(2 * p); Vs = rng.standard_normal((p, K)) / np.sqrt(K)
    Ws, Vs = adam_fit(Ws, Vs, a, b, p, Yt)
    L_adam = loss(Ws, Vs, a, b, p, Yt)
    Ws, Vs, L = lm_polish(Ws, Vs, a, b, p, Yt, False)
    r = sgld(Ws, Vs, a, b, p, Yt, 600); r.update(L_adam=float(L_adam), L_polished=float(L))
    out["N2_trained_generic"] = r
    print("N2", r, flush=True)
    # N3 chain-seed set B at the (LM-polished) grokked ckpt
    Y = targets(a, b, p)
    Wp, Vp, L = lm_polish(Wg, Vg, a, b, p, Y, False)
    out["N3_grok_seedB"] = sgld(Wp, Vp, a, b, p, Y, 777)
    out["N3_grok_seedB"]["ratio_paper"] = out["N3_grok_seedB"]["lam_hat"] / out["paper"]
    print("N3", out["N3_grok_seedB"], flush=True)
    with open(os.path.join(HERE, "null.jsonl"), "a") as f:
        f.write(json.dumps(out) + "\n")


if __name__ == "__main__":
    for n in sys.argv[1:]:
        main(n)
