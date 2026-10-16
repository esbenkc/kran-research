"""skeptic: independent check of every sampler / calibration number for the 'Sampler and calibration' foldout
(G. Wang feedback, 2026-10-16).

Own code (generic X-matrix model, own Jacobian, own SGLD loop, own chain seeds). Parts:
  A  rebuild the 8 calibration networks from their stated recipe, exact rank/2 with my Jacobian,
     my SGLD at the frozen setting -> est/true table (+ wall time per network)
  B  eps in llc.csv == crel/(nbeta*h_max+gamma) with h_max from the saved singular values; 2/eps (slow relaxation)
  C  calibration grid: worst |rel err| per (crel, gamma, nbeta) -> was the frozen setting the argmin?
  D  stationarity: share of checked cells that settled
  E  sensitivity of grokked readings to gamma, nbeta, crel (my SGLD, my LM polish, 10k steps, 4 chains)
Usage: python sampler_check.py [A B C D E]   -> sampler_check.json
"""
import csv, json, os, sys, time
from concurrent.futures import ProcessPoolExecutor
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "../../data")
OUT = os.environ.get("OUT", os.path.join(HERE, "sampler_check.json"))
NB, G, CREL, STEPS, BURN, CHAINS = 1e4, 1.0, 0.2, 10000, 2000, 4


# ------------------------------------------------------------ generic model on an input matrix X
def fwd(W, V, X):
    P = X @ W
    H = P * P
    return H @ V.T, P, H


def loss(W, V, X, Y):
    R = fwd(W, V, X)[0] - Y
    R = R - R.mean(0)
    return 0.5 * np.sum(R * R) / len(X)


def grad(W, V, X, Y, freeze_W):
    F, P, H = fwd(W, V, X)
    R = F - Y
    R = (R - R.mean(0)) / len(X)
    gV = R.T @ H
    gW = None if freeze_W else X.T @ (2 * P * (R @ V))
    return gW, gV


def jac(W, V, X, freeze_W=False):
    n, d = X.shape
    p, K = V.shape
    F, P, H = fwd(W, V, X)
    JV = np.zeros((n, p, p, K))
    for o in range(p):
        JV[:, o, o, :] = H
    JV = JV.reshape(n, p, p * K)
    if freeze_W:
        J = JV
    else:
        JW = 2 * (P[:, None, None, :] * V[None, :, None, :]) * X[:, None, :, None]   # n,p,d,K
        J = np.concatenate([JW.reshape(n, p, d * K), JV], 2)
    J = J - J.mean(0, keepdims=True)
    return J.reshape(n * p, -1)


def sgld(W0, V0, X, Y, nb, g, eps, steps, burn, chains, seed, freeze_W=False):
    rng = np.random.default_rng(seed)
    L0 = loss(W0, V0, X, Y)
    out = []
    for c in range(chains):
        W, V = W0.copy(), V0.copy()
        acc, cnt = 0.0, 0
        for t in range(steps):
            gW, gV = grad(W, V, X, Y, freeze_W)
            if not freeze_W:
                W = W - 0.5 * eps * (nb * gW + g * (W - W0)) + np.sqrt(eps) * rng.standard_normal(W.shape)
            V = V - 0.5 * eps * (nb * gV + g * (V - V0)) + np.sqrt(eps) * rng.standard_normal(V.shape)
            if t >= burn:
                acc += loss(W, V, X, Y); cnt += 1
        out.append(nb * (acc / cnt - L0))
    return float(np.mean(out)), [float(x) for x in out]


def onehot(p, idx):
    a, b = idx // p, idx % p
    X = np.zeros((len(idx), 2 * p)); X[np.arange(len(idx)), a] = 1; X[np.arange(len(idx)), p + b] = 1
    Y = np.eye(p)[(a + b) % p]
    return X, Y


# ------------------------------------------------------------ A: calibration networks
CASES = [("gauss_under_p5_K3", 5, 3, 300, "gauss"), ("gauss_under_p7_K6", 7, 6, 400, "gauss"),
         ("gauss_over_p3_K20", 3, 20, 200, "gauss"), ("onehot_p11_K6", 11, 6, 60, "onehot"),
         ("onehot_p17_K12", 17, 12, 144, "onehot"), ("onehot_p7_K14_cap", 7, 14, 24, "onehot"),
         ("vonly_p11_K44", 11, 44, 60, "vonly"), ("vonly_p17_K68", 17, 68, 144, "vonly")]


def build_case(name, p, K, n, kind):
    """Stated recipe: teacher seed 0, W,V ~ N(0,1); Gaussian inputs (d=2p) or a seed-0 one-hot train split
    of n pairs; targets = teacher outputs (zero loss at the teacher)."""
    rng = np.random.default_rng(0)
    d = 2 * p
    if kind == "gauss":
        X = rng.standard_normal((n, d))
    else:
        perm = np.random.default_rng(0).permutation(p * p)
        idx = np.sort(perm[:int(round(n))])
        X, _ = onehot(p, idx)
    W = rng.standard_normal((d, K)); V = rng.standard_normal((p, K))
    Y = fwd(W, V, X)[0]
    return W, V, X, Y


def run_case(spec):
    name, p, K, n, kind = spec
    W, V, X, Y = build_case(*spec)
    fz = kind == "vonly"
    s = np.linalg.svd(jac(W, V, X, fz), compute_uv=False)
    r = int(np.sum(s > 1e-6 * s[0])); r9 = int(np.sum(s > 1e-9 * s[0]))
    h = s[:r] ** 2 / len(X)
    eps = CREL / (NB * h.max() + G)
    t0 = time.time()
    lam, ch = sgld(W, V, X, Y, NB, G, eps, STEPS, BURN, CHAINS, seed=4242, freeze_W=fz)
    if kind == "gauss":
        D = d = 2 * p
        D = d * (d + 1) // 2
        paper = K * (d + p - 1) / 2 if K * (d + p - 1) < p * D else p * d * (d + 1) / 4
    elif kind == "onehot":
        paper = K * (3 * p - 1) / 2
    else:
        paper = p * min(len(X) - 1, K) / 2
    return dict(case=name, n=len(X), n_params=(V.size if fz else W.size + V.size), rank=r, rank_1e9=r9,
                truth=r / 2, paper=paper, h_min=float(h.min()), h_max=float(h.max()),
                n_below_res=int(np.sum(h < G / NB)), eps=float(eps), lam_hat=lam, chains=ch,
                rel_err=(lam - r / 2) / (r / 2), secs=time.time() - t0,
                quad=float(np.sum(0.5 * h / (h + G / NB))))


def part_A():
    with ProcessPoolExecutor(4) as ex:
        rows = list(ex.map(run_case, CASES))
    fz = {r["case"]: r for r in json.load(open(os.path.join(DATA, "calibration/calibration_frozen.json")))}
    for r in rows:
        b = fz[r["case"]]
        r.update(build_truth=b["truth"], build_hat=b["lambda_hat"], build_rel_err=b["rel_err"], build_eps=b["eps"],
                 build_h_min=b["h_min"])
    return rows


# ------------------------------------------------------------ B: eps rule + relaxation times
def part_B():
    rows = list(csv.DictReader(open(os.path.join(DATA, "llc.csv")))) + \
        list(csv.DictReader(open(os.path.join(DATA, "llc_p31_frac07.csv"))))
    out = []
    for r in rows:
        if r["lambda_hat"] in ("", "nan"):
            continue
        tag = f"p{r['p']}_K{r['K']}_s{r['seed']}_{r['phase']}"
        tdir = "llc_traces_p31_frac07" if (r["p"] == "31" and r["frac"].startswith("0.7")) else "llc_traces"
        f = os.path.join(DATA, tdir, tag + ".npz")
        if not os.path.exists(f):
            continue
        z = np.load(f)
        N = int(r["N"]); nb, g, cr = float(r["nbeta"]), float(r["gamma"]), float(r["crel"])
        if r["phase"] == "grok":
            hmax = z["singular_values"][0] ** 2 / N
        else:
            hmax = None   # V-only eps uses the feature-matrix spectrum, not saved; skip the check
        eps = float(r["eps"])
        traces = {k: z[k].shape for k in z.files if k != "singular_values"}
        out.append(dict(ckpt=tag, frac=float(r["frac"]), eps=eps,
                        eps_rule=(cr / (nb * hmax + g) if hmax is not None else None),
                        eps_rel_dev=(abs(eps - cr / (nb * hmax + g)) / eps if hmax is not None else None),
                        relax_steps_2_over_eps_gamma=2 / (eps * g), trace_shapes=traces,
                        settings=(cr, g, nb, int(r["chains"]), int(r["steps"]), int(r["burn"]))))
    return out


# ------------------------------------------------------------ C: grid
def part_C():
    g = json.load(open(os.path.join(DATA, "calibration/calibration_grid_main.json")))
    agg = {}
    for r in g:
        k = (r["crel"], r["gamma"], r["nbeta"])
        e = float("inf") if r["diverged"] else abs(r["rel_err"])
        agg.setdefault(k, []).append(e)
    tab = sorted(((max(v), k, len(v)) for k, v in agg.items()))
    meta = dict(n_rows=len(g), steps=sorted({r["steps"] for r in g}), burn=sorted({r["burn"] for r in g}),
                chains=sorted({len(r["chains"]) for r in g}), scales=sorted({r["scale_w"] for r in g}),
                diverged=sum(bool(r["diverged"]) for r in g))
    return dict(meta=meta, worst_by_setting=[dict(setting=k, worst_abs_rel_err=w, n=n) for w, k, n in tab])


# ------------------------------------------------------------ D: stationarity
def part_D():
    out = []
    for fn in ("stationarity.csv", "stationarity_p31_frac07.csv"):
        for r in csv.DictReader(open(os.path.join(DATA, fn))):
            out.append(dict(cell=f"p{r['p']}_K{r['K']}_s{r['seed']}", frac=float(r["frac"]),
                            rel=float(r["rel_last_vs_10k"]), settled=int(r["settled"]),
                            recomputed_settled=int(abs(float(r["rel_last_vs_10k"])) <= 0.05)))
    return dict(rows=out, n=len(out), n_settled=sum(r["settled"] for r in out))


# ------------------------------------------------------------ E: sensitivity on grokked checkpoints
def lm_polish(W, V, X, Y, iters=60, tol=1e-13):
    p, K = V.shape; d = W.shape[0]
    mu, L = 1e-6, loss(W, V, X, Y)
    for _ in range(iters):
        if L < tol:
            break
        J = jac(W, V, X)
        R = fwd(W, V, X)[0] - Y; R = R - R.mean(0)
        A, gg = J.T @ J, J.T @ R.ravel()
        while True:
            dlt = -np.linalg.solve(A + mu * np.eye(len(gg)), gg)
            W2, V2 = W + dlt[:d * K].reshape(d, K), V + dlt[d * K:].reshape(p, K)
            L2 = loss(W2, V2, X, Y)
            if L2 < L:
                W, V, L = W2, V2, L2; mu = max(mu / 10, 1e-12); break
            mu *= 10
            if mu > 1e6:
                return W, V, L
    return W, V, L


def sens_job(args):
    os.environ["OMP_NUM_THREADS"] = "1"
    name, crel, g, nb = args
    z = np.load(os.path.join(DATA, "ckpt", name + ".npz"))
    p, K = int(z["p"]), int(z["K"])
    X, Y = onehot(p, z["train_idx"])
    W, V, L = lm_polish(z["W"], z["V"], X, Y)
    s = np.linalg.svd(jac(W, V, X), compute_uv=False)
    hmax = s[0] ** 2 / len(X)
    eps = crel / (nb * hmax + g)
    t0 = time.time()
    lam, ch = sgld(W, V, X, Y, nb, g, eps, STEPS, BURN, CHAINS, seed=9000 + p + K)
    h = s ** 2 / len(X)
    return dict(ckpt=name, crel=crel, gamma=g, nbeta=nb, eps=eps, lam_hat=lam, chains=ch,
                paper=K * (3 * p - 1) / 2, ratio=lam / (K * (3 * p - 1) / 2),
                quad=float(np.sum(0.5 * h / (h + g / nb))), L_polished=float(L), secs=time.time() - t0)


# ------------------------------------------------------------ F: finite-chain quadratic model
def finite_quad(h, nb, g, eps, steps, burn):
    """Exact expected SGLD reading of the localised quadratic model sum_i 1/2 h_i x_i^2, started at x=0:
    var_{t+1} = (1-eps a/2)^2 var_t + eps, a = nb h + g; reading = nb * mean_{burn<=t<steps} sum 1/2 h var_t."""
    a = nb * h + g
    q = (1 - eps * a / 2) ** 2
    var_inf = eps / (1 - q)
    avg_qt = (q ** burn - q ** steps) / (1 - q) / (steps - burn)
    return float(np.sum(0.5 * nb * h * var_inf * (1 - avg_qt))), float(np.sum(0.5 * nb * h * var_inf))


VERDICT = {(11, 33), (17, 51), (17, 68), (23, 69), (23, 92), (31, 93)}


def part_F():
    rows = list(csv.DictReader(open(os.path.join(DATA, "llc.csv")))) + \
        list(csv.DictReader(open(os.path.join(DATA, "llc_p31_frac07.csv"))))
    out = []
    for r in rows:
        p, K, fr = int(r["p"]), int(r["K"]), float(r["frac"])
        if r["phase"] != "grok" or (p, K) not in VERDICT or (p == 31 and fr < 0.6):
            continue
        tdir = "llc_traces_p31_frac07" if p == 31 else "llc_traces"
        s = np.load(os.path.join(DATA, tdir, f"p{p}_K{K}_s{r['seed']}_grok.npz"))["singular_values"]
        N, rJ, eps = int(r["N"]), int(r["rankJ"]), float(r["eps"])
        h = s[:rJ] ** 2 / N
        q10, qinf = finite_quad(h, NB, G, eps, 10000, 2000)
        q40, _ = finite_quad(h, NB, G, eps, 40000, 8000)
        hat = float(r["lambda_hat"])
        out.append(dict(ckpt=f"p{p}_K{K}_s{r['seed']}", frac=fr, lam_hat=hat, lam_paper=float(r["lambda_paper"]),
                        quad_inf=qinf, quad_csv=float(r["lambda_quad"]), quad_10k=q10, quad_40k=q40,
                        quad10k_over_inf=q10 / qinf, hat_over_quad10k=hat / q10, hat_over_quadinf=hat / qinf,
                        n_h_above_res=int(np.sum(h > G / NB)), rank=rJ))
    return out


SENS_CKPTS = ["p11_K33_s0_grok", "p17_K51_s0_grok", "p17_K68_s0_grok"]
SENS_SETTINGS = [(0.2, 1.0, 1e4), (0.2, 0.1, 1e4), (0.2, 10.0, 1e4), (0.2, 100.0, 1e4),
                 (0.2, 1.0, 1e3), (0.2, 1.0, 1e5), (0.1, 1.0, 1e4), (0.4, 1.0, 1e4)]


def part_E():
    jobs = [(c, *s) for c in SENS_CKPTS for s in SENS_SETTINGS]
    with ProcessPoolExecutor(8) as ex:
        return list(ex.map(sens_job, jobs))


if __name__ == "__main__":
    parts = sys.argv[1:] or ["A", "B", "C", "D", "E"]
    res = json.load(open(OUT)) if os.path.exists(OUT) else {}
    for P in parts:
        t0 = time.time()
        res[P] = globals()["part_" + P]()
        print(P, "done %.0fs" % (time.time() - t0), flush=True)
        json.dump(res, open(OUT, "w"), indent=1, default=float)
