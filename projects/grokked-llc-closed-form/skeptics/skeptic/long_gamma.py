"""skeptic: does a weaker spring (gamma 0.1) read higher once the chain is long enough?
10x longer chains (100k steps, 20k burn, 4 chains) at gamma 0.1 and 1, n beta 1e4, crel 0.2, on two grokked nets.
Uses sampler_check.py (own code). -> long_gamma.json"""
import json, os, time
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import sampler_check as S


def job(args):
    os.environ["OMP_NUM_THREADS"] = "1"
    name, g, steps = args
    z = np.load(os.path.join(S.DATA, "ckpt", name + ".npz"))
    p, K = int(z["p"]), int(z["K"])
    X, Y = S.onehot(p, z["train_idx"])
    W, V, _ = S.lm_polish(z["W"], z["V"], X, Y)
    s = np.linalg.svd(S.jac(W, V, X), compute_uv=False)
    eps = 0.2 / (1e4 * s[0] ** 2 / len(X) + g)
    t0 = time.time()
    lam, ch = S.sgld(W, V, X, Y, 1e4, g, eps, steps, steps // 5, 4, seed=31337 + p + K)
    return dict(ckpt=name, gamma=g, steps=steps, burn=steps // 5, lam_hat=lam, chains=ch,
                ratio=lam / (K * (3 * p - 1) / 2), secs=time.time() - t0)


if __name__ == "__main__":
    jobs = [(c, g, 100_000) for c in ["p11_K33_s0_grok", "p17_K51_s0_grok"] for g in (0.1, 1.0)]
    with ProcessPoolExecutor(4) as ex:
        res = list(ex.map(job, jobs))
    for r in res:
        print(r["ckpt"], r["gamma"], "ratio %.3f" % r["ratio"], [round(c) for c in r["chains"]], "%.0fs" % r["secs"])
    json.dump(res, open(os.path.join(S.HERE, "long_gamma.json"), "w"), indent=1)
