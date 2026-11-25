"""Pre-freeze check of PREREG v2 law L on the TARGET side (exact Gibbs, c -> 0), my own code.
For each frozen shape/rank and (nbeta, q) cell: exact localised-posterior lambda_hat via block Gibbs; law L =
sum_reg 1/2 * 1/(1+g_i) + lambda0 * psi(q), psi = isotonic fit in log q on TRAIN cells (W32 r1, r8) only.
Reports share(miss <= 5 pts) on held-out cells and the spread, i.e. what the frozen rule would say if the sampler were
perfect. Usage: python vera_power_v2.py [sweeps] [seeds]
"""
import json, sys, time
import numpy as np
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from vera_aoyagi import aoyagi


def setup(M, N, H, r, seed):
    rng = np.random.default_rng(seed)
    U, _ = np.linalg.qr(rng.standard_normal((N, r)))
    V, _ = np.linalg.qr(rng.standard_normal((M, r)))
    W0 = U @ V.T  # singular values 1 (frozen setting)
    B0 = np.zeros((N, H)); A0 = np.zeros((H, M))
    B0[:, :r] = U; A0[:r] = V.T
    return W0, B0, A0


def hess_eigs(M, N, H, B0, A0):
    J = np.zeros((N * M, N * H + H * M))
    col = 0
    for i in range(N):
        for k in range(H):
            E = np.zeros((N, H)); E[i, k] = 1; J[:, col] = (E @ A0).ravel(); col += 1
    for k in range(H):
        for j in range(M):
            E = np.zeros((H, M)); E[k, j] = 1; J[:, col] = (B0 @ E).ravel(); col += 1
    ev = np.linalg.eigvalsh(J.T @ J)
    return ev[ev > 1e-8]


def gibbs(M, N, H, W0, B0, A0, gamma, nbeta, sweeps, burn, rng):
    I = np.eye(H)
    B, A = B0.copy(), A0.copy()
    acc, cnt = 0.0, 0
    for t in range(sweeps):
        P = nbeta * A @ A.T + gamma * I
        L = np.linalg.cholesky(P)
        B = np.linalg.solve(P, (nbeta * W0 @ A.T + gamma * B0).T).T + np.linalg.solve(L.T, rng.standard_normal((H, N))).T
        P = nbeta * B.T @ B + gamma * I
        L = np.linalg.cholesky(P)
        A = np.linalg.solve(P, nbeta * B.T @ W0 + gamma * A0) + np.linalg.solve(L.T, rng.standard_normal((H, M)))
        if t >= burn:
            D = B @ A - W0
            acc += 0.5 * np.sum(D * D); cnt += 1
    return nbeta * acc / cnt


SHAPES = {"W32": (32, 32, 16), "W16": (16, 16, 8)}
CASES = [("W32", 1), ("W32", 8), ("W32", 2), ("W32", 4), ("W32", 12), ("W16", 1), ("W16", 2), ("W16", 4), ("W16", 8)]
TRAIN = {("W32", 1), ("W32", 8)}
NBETAS = [10.0, 100.0, 1000.0, 10000.0]
QS = [0.01, 0.1, 1.0, 10.0, 100.0]


def isotonic_fit(x, y):
    """Pool-adjacent-violators, increasing in x; returns sorted unique x and fitted values."""
    order = np.argsort(x); x = np.asarray(x)[order]; y = np.asarray(y)[order]
    blocks = [[yy, 1.0] for yy in y]
    vals = []
    for v in y:
        vals.append([v, 1.0])
        while len(vals) > 1 and vals[-2][0] > vals[-1][0]:
            a, b = vals.pop(), vals.pop()
            vals.append([(a[0] * a[1] + b[0] * b[1]) / (a[1] + b[1]), a[1] + b[1]])
    fit = np.concatenate([[v] * int(w) for v, w in vals])
    return x, fit


if __name__ == "__main__":
    sweeps = int(sys.argv[1]) if len(sys.argv) > 1 else 600
    seeds = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    burn = sweeps // 5
    rows = []
    t0 = time.time()
    for (w, r) in CASES:
        M, N, H = SHAPES[w]
        lam = float(aoyagi(M, N, H, r)[0]); lam0 = float(aoyagi(M - r, N - r, H - r, 0)[0]) if H > r else 0.0
        for s in range(seeds):
            W0, B0, A0 = setup(M, N, H, r, s)
            h = hess_eigs(M, N, H, B0, A0)
            assert len(h) == r * (M + N - r), (w, r, len(h))
            rng = np.random.default_rng(1000 + s)
            for nb in NBETAS:
                for q in QS:
                    g = np.sqrt(nb / q)
                    lh = gibbs(M, N, H, W0, B0, A0, g, nb, sweeps, burn, rng)
                    reg = float(np.sum(0.5 / (1 + g / (nb * h))))
                    rows.append(dict(w=w, r=r, seed=s, nbeta=nb, q=q, gamma=g, lam=lam, lam0=lam0, lam_hat=lh,
                                     reg_pred=reg, rel=lh / lam - 1, psi_obs=(lh - reg) / lam0 if lam0 else None))
        print(f"{w} r={r} done [{time.time()-t0:.0f}s]", flush=True)
    # fit psi on TRAIN
    tr = [x for x in rows if (x["w"], x["r"]) in TRAIN]
    xs, fit = isotonic_fit(np.log10([x["q"] for x in tr]), [x["psi_obs"] for x in tr])
    def psi(q):
        lq = np.log10(q)
        ux = np.unique(xs); uy = np.array([fit[xs == u].mean() for u in ux])
        return float(np.interp(lq, ux, uy))
    out = {}
    for (w, r) in CASES:
        cells = {}
        for x in rows:
            if (x["w"], x["r"]) != (w, r): continue
            cells.setdefault((x["nbeta"], x["q"]), []).append(x)
        misses, meas = [], []
        globals().setdefault("HELD", [])
        for (nb, q), xx in sorted(cells.items()):
            m = np.mean([x["rel"] for x in xx])
            pred = (xx[0]["reg_pred"] + xx[0]["lam0"] * psi(q)) / xx[0]["lam"] - 1
            misses.append(abs(pred - m) * 100); meas.append(m * 100)
        if (w, r) not in TRAIN: HELD.extend(misses); globals().setdefault("HMEAS", []).extend(meas)
        out[f"{w}_r{r}"] = dict(train=(w, r) in TRAIN, share_le5=float(np.mean(np.array(misses) <= 5)),
                                median_miss=float(np.median(misses)), max_miss=float(np.max(misses)),
                                meas_min=float(min(meas)), meas_max=float(max(meas)))
        o = out[f"{w}_r{r}"]
        print(f"{w} r={r:2d} {'TRAIN' if o['train'] else 'held '} share<=5 {o['share_le5']:.2f} median miss {o['median_miss']:.1f} "
              f"max {o['max_miss']:.1f}  meas [{o['meas_min']:.1f}, {o['meas_max']:.1f}]")
    held = [k for k in out if not out[k]["train"]]
    allm = []
    print(f"HELD-OUT pooled: n={len(HELD)} share<=5 {np.mean(np.array(HELD)<=5):.3f} median miss {np.median(HELD):.2f}; spread {min(HMEAS):.1f}..{max(HMEAS):.1f}")
    print("psi table:", [(q, round(psi(q), 3)) for q in QS])
    json.dump(dict(rows=rows, summary=out, psi={str(q): psi(q) for q in QS}), open(__file__.replace(".py", f"_{sweeps}_{seeds}.json"), "w"), indent=1)
