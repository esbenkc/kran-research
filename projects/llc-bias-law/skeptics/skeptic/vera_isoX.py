"""Vera pre-data power check: does relative LLC error collapse on X = eps*nbeta/gamma?
Own minimal SGLD on reduced-rank regression with the POPULATION loss K(w) = 0.5*||B A - W0||_F^2
(x ~ N(0, I), noise drops out), so the known answer is Aoyagi's lambda and nothing else is random.
Holds X fixed and moves gamma/nbeta; also holds gamma/nbeta fixed and moves X.
Update (devinterp convention): w <- w - eps/2 * (nbeta*grad K + gamma*(w - w*)) + N(0, eps).
Estimate: lambda_hat = nbeta * mean_draws K(w) (K(w*) = 0).
"""
import json, sys, time
import numpy as np
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from vera_aoyagi import aoyagi

M = N = 32
H = 16


def run(r, eps, gamma, nbeta, steps, chains=4, seed=0, burn_frac=0.5):
    rng = np.random.default_rng(seed)
    U, _ = np.linalg.qr(rng.standard_normal((N, r)))
    V, _ = np.linalg.qr(rng.standard_normal((M, r)))
    s = np.linspace(1.0, 2.0, r)
    W0 = (U * s) @ V.T
    B0 = np.zeros((N, H)); A0 = np.zeros((H, M))
    B0[:, :r] = U * np.sqrt(s); A0[:r] = (V * np.sqrt(s)).T
    B = np.repeat(B0[None], chains, 0); A = np.repeat(A0[None], chains, 0)
    burn = int(steps * burn_frac)
    acc = np.zeros(chains); cnt = 0
    for t in range(steps):
        D = B @ A - W0
        gB = D @ A.transpose(0, 2, 1)
        gA = B.transpose(0, 2, 1) @ D
        B = B - eps / 2 * (nbeta * gB + gamma * (B - B0)) + np.sqrt(eps) * rng.standard_normal(B.shape)
        A = A - eps / 2 * (nbeta * gA + gamma * (A - A0)) + np.sqrt(eps) * rng.standard_normal(A.shape)
        if t >= burn:
            acc += 0.5 * np.sum(D * D, axis=(1, 2)); cnt += 1
    lam_chain = nbeta * acc / cnt
    lam = float(aoyagi(M, N, H, r)[0])
    return dict(r=r, eps=eps, gamma=gamma, nbeta=nbeta, steps=steps, X=eps * nbeta / gamma,
                g_over_nb=gamma / nbeta, lam_true=lam, lam_hat=float(lam_chain.mean()),
                rel_err=float(lam_chain.mean() / lam - 1), chains=lam_chain.tolist())


if __name__ == "__main__":
    out = []
    t0 = time.time()
    nbeta = 1000.0
    steps = int(sys.argv[1]) if len(sys.argv) > 1 else 4000
    # iso-X line: X = 1e-3 fixed, gamma/nbeta over 2 decades
    for r in (1, 8):
        for gamma in (10.0, 100.0, 1000.0):
            eps = 1e-3 * gamma / nbeta
            out.append(run(r, eps, gamma, nbeta, steps))
            o = out[-1]
            print(f"isoX r={r} gamma={gamma:7.1f} eps={eps:.1e} X={o['X']:.1e} g/nb={o['g_over_nb']:.2f} "
                  f"lam_hat={o['lam_hat']:.1f} true={o['lam_true']} rel={o['rel_err']:+.3f}  [{time.time()-t0:.0f}s]", flush=True)
    json.dump(out, open(__file__.replace(".py", f"_{steps}.json"), "w"), indent=1)
