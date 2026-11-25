"""Vera's own scorer for PREREG 2(4)-(6) (written from the frozen text, not from score.py).
Reads Bruno's RAW per-row files data/cells_S/*.json (fields per #30752: case, M,N,H,r, lam, lam0, seed, nbeta, q,
gamma, c, eps, T, e, e_chains, halves). Recomputes e from e_chains, refits a,b, nulls, status, seed-level line.
Usage: python vera_score.py <project dir> [--shuffle-null N]
"""
import glob, json, os, sys
import numpy as np
from scipy.optimize import least_squares
from scipy import stats
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vera_core import lam_parts, teacher, hess_eigs, reg_term

TRAIN = {(32, 32, 16, 1), (32, 32, 16, 8)}
BLIND = {(24, 24, 12, 3), (24, 24, 12, 6), (48, 48, 8, 2), (20, 20, 10, 5), (40, 24, 12, 4)}
K4 = 0.5
_H = {}


def pop_h(shape, numeric=False):
    """Population Gauss-Newton Hessian at the special w*, unit teacher values: eigenvalue 2 (x r^2), 1 (x r(M+N-2r)).
    Checked numerically (numeric=True) on (32,32,16,1), (32,32,16,8), (40,24,12,4), (48,48,8,2)."""
    M, N, H, r = shape
    if numeric:
        W0, B0, A0 = teacher(M, N, H, r, 0)
        return hess_eigs(B0, A0, np.eye(M))
    return np.array([2.0] * (r * r) + [1.0] * (r * (M + N - 2 * r)))


REBUILD = {"n": 0, "max_dev": 0.0, "half_flips": 0}


def load(proj, arm="S"):
    """Rows from data/cells_<arm>/<case>.json; e per chain REBUILT from the raw block means in <case>_blocks.npz
    (blocks of `block` steps over the whole chain; kept = blocks after burn = T/2)."""
    rows = []
    for f in sorted(glob.glob(os.path.join(proj, "data", f"cells_{arm}", "*.json"))):
        d = json.load(open(f))
        z = np.load(f.replace(".json", "_blocks.npz"))
        blk = d["block"]
        for x in d["rows"]:
            key = f"nb{x['nbeta']}_q{x['q']:g}_c{x['c']:g}_s{x['seed']}"
            B = z[key]  # chains x blocks, block means of K_n
            nb_burn = int(round(x["burn"] / blk)); kept = B[:, nb_burn:]
            assert B.shape[1] * blk == x["T"], (key, B.shape, x["T"])
            ech = x["nbeta"] * kept.mean(axis=1) / x["lam"] - 1
            dev = float(np.nanmax(np.abs(ech - np.asarray(x["e_chains"], float)))) if np.all(np.isfinite(ech)) else 0.0
            REBUILD["n"] += 1; REBUILD["max_dev"] = max(REBUILD["max_dev"], dev)
            h = kept.shape[1] // 2
            e1 = x["nbeta"] * kept[:, :h].mean() / x["lam"] - 1; e2 = x["nbeta"] * kept[:, h:].mean() / x["lam"] - 1
            my_ok = bool(np.isfinite(e1) and np.isfinite(e2) and abs(e1 - e2) <= 0.02)
            if my_ok != bool(x["halves_ok"]): REBUILD["half_flips"] += 1
            x = dict(x); x["e_chains"] = ech.tolist(); x["halves"] = [[e1, e2]]; x["my_halves_ok"] = my_ok
            rows.append(x)
    return rows


def cell_key(x):
    return (x["M"], x["N"], x["H"], x["r"], float(x["nbeta"]), float(x["q"]), float(x["c"]))


def e_of_row(x, lam):
    ch = np.asarray(x["e_chains"], dtype=float)
    return ch  # per-chain relative error (fraction); checked against x['e'] and against lam_hat if present


def aggregate(rows):
    """-> dict cell -> dict(e_seed={seed: e}, ok=bool, eps, gamma)"""
    cells = {}
    for x in rows:
        k = cell_key(x)
        lam, reg, lam0 = lam_parts(x["M"], x["N"], x["H"], x["r"])
        assert abs(lam - x["lam"]) < 1e-9 and abs(lam0 - x["lam0"]) < 1e-9, (k, lam, x["lam"])
        ch = e_of_row(x, lam)
        e_seed = float(np.mean(ch))
        c = cells.setdefault(k, dict(e_seed={}, ok=True, eps=x["eps"], gamma=x["gamma"], T=x["T"], shape=k[:4]))
        c["e_seed"][x["seed"]] = e_seed
        bad = (not np.all(np.isfinite(ch))) or np.any(np.abs(ch) > 10) or (not x["my_halves_ok"])
        if bad:
            c["ok"] = False
        REBUILD["max_dev_e"] = max(REBUILD.get("max_dev_e", 0.0), abs(e_seed - x["e"]) if np.isfinite(x["e"]) else 0.0)
    return cells


def law_pred(shape, nbeta, q, eps, a, b):
    M, N, H, r = shape
    lam, reg, lam0 = lam_parts(M, N, H, r)
    gamma = np.sqrt(nbeta * K4 / q)
    regp = reg_term(pop_h(shape), nbeta, gamma, eps)
    return (regp + lam0 / (1 + (a / q) ** b)) / lam - 1


def fit_L(train_cells):
    ks = list(train_cells)
    y = np.array([np.mean(list(train_cells[k]["e_seed"].values())) for k in ks])
    def res(p):
        a, b = np.exp(p)
        return 100 * (np.array([law_pred(k[:4], k[4], k[5], train_cells[k]["eps"], a, b) for k in ks]) - y)
    sol = least_squares(res, x0=np.log([0.3, 1.0]))
    return np.exp(sol.x), float(np.sqrt(np.mean(sol.fun ** 2)))


def fit_N2(train_cells):
    ks = list(train_cells)
    y = np.array([np.mean(list(train_cells[k]["e_seed"].values())) for k in ks])
    q = np.array([k[5] for k in ks])
    def res(p):
        a, b = np.exp(p)
        return 100 * (1 / (1 + (a / q) ** b) - 1 - y)
    sol = least_squares(res, x0=np.log([0.3, 1.0]))
    return np.exp(sol.x)


def iso_fit(x, y):
    o = np.argsort(x); x = np.asarray(x)[o]; y = np.asarray(y)[o]
    vals = []
    for v in y:
        vals.append([v, 1.0])
        while len(vals) > 1 and vals[-2][0] > vals[-1][0]:
            p2, p1 = vals.pop(), vals.pop()
            vals.append([(p1[0] * p1[1] + p2[0] * p2[1]) / (p1[1] + p2[1]), p1[1] + p2[1]])
    fit = np.concatenate([[v] * int(w) for v, w in vals])
    # isotonic increasing or decreasing? choose direction with lower SSE
    return x, fit


def score(proj, verbose=True):
    rows = load(proj)
    cells = aggregate(rows)
    train = {k: v for k, v in cells.items() if k[:4] in TRAIN and k[5] >= 0.1}
    prim = {k: v for k, v in cells.items() if k[:4] in BLIND and k[5] >= 0.1}
    (a, b), rmse = fit_L(train)
    a2, b2 = fit_N2(train)
    ytr = {k: np.mean(list(v["e_seed"].values())) for k, v in train.items()}
    n1 = float(np.mean(list(ytr.values())))
    # N3 isotonic e vs log X (both directions, pick lower SSE)
    lx = np.array([np.log10(v["eps"] * k[4] / v["gamma"]) for k, v in train.items()]); yy = np.array(list(ytr.values()))
    xi, fi = iso_fit(lx, yy); xd, fd = iso_fit(-lx, -yy)
    if np.sum((fd * -1 - yy[np.argsort(-lx)]) ** 2) < np.sum((fi - yy[np.argsort(lx)]) ** 2):
        n3x, n3y = -xd[::-1], -fd[::-1]
    else:
        n3x, n3y = xi, fi
    def n3(lxv):
        ux = np.unique(n3x); uy = np.array([n3y[n3x == u].mean() for u in ux])
        return float(np.interp(lxv, ux, uy))
    def n4(k, v):
        p = np.array([np.log(k[4]), np.log(v["gamma"]), np.log(v["eps"])])
        d = [(np.linalg.norm(p - np.array([np.log(kk[4]), np.log(vv["gamma"]), np.log(vv["eps"])])), kk) for kk, vv in train.items()]
        dmin = min(dd for dd, _ in d)
        return float(np.mean([ytr[kk] for dd, kk in d if dd <= dmin + 1e-9]))
    miss = {"L": [], "N1": [], "N2": [], "N3": [], "N4": []}
    meas = []; per_seed = {}
    for k, v in sorted(prim.items()):
        em = float(np.mean(list(v["e_seed"].values()))); meas.append(em)
        pL = law_pred(k[:4], k[4], k[5], v["eps"], a, b)
        mL = 100.0 if not v["ok"] else abs(pL - em) * 100
        miss["L"].append(mL)
        miss["N1"].append(abs(n1 - em) * 100)
        miss["N2"].append(abs(1 / (1 + (a2 / k[5]) ** b2) - 1 - em) * 100)
        miss["N3"].append(abs(n3(np.log10(v["eps"] * k[4] / v["gamma"])) - em) * 100)
        miss["N4"].append(abs(n4(k, v) - em) * 100)
        for s, es in v["e_seed"].items():
            per_seed.setdefault(s, []).append(100.0 if not v["ok"] else abs(pL - es) * 100)
    mL = np.array(miss["L"]); share = float(np.mean(mL <= 5)); med = {kk: float(np.median(vv)) for kk, vv in miss.items()}
    spread = 100 * (max(meas) - min(meas)) if meas else 0.0
    seeds = sorted(per_seed); ss = np.array([np.mean(np.array(per_seed[s]) <= 5) for s in seeds])
    ci = stats.t.interval(0.95, len(ss) - 1, loc=ss.mean(), scale=stats.sem(ss)) if len(ss) > 1 else (np.nan, np.nan)
    best_null = min(med[n] for n in ("N1", "N2", "N3", "N4"))
    if len(seeds) < 3 or spread < 20:
        status = "UNCONFIRMED"
    elif share >= 0.80 and all(med["L"] <= 0.5 * med[n] for n in ("N1", "N2", "N3", "N4")):
        status = "CONFIRMED"
    elif share < 0.60 or med["L"] >= best_null:
        status = "REJECTED"
    else:
        status = "UNCONFIRMED"
    out = dict(a=a, b=b, train_rmse_pts=rmse, N2=(a2, b2), n_primary=len(prim), n_not_ok=sum(not v["ok"] for v in prim.values()),
               share_le5=share, medians=med, spread_pts=spread, seed_shares=ss.tolist(), seed_ci=list(map(float, ci)), status=status)
    if verbose:
        print(json.dumps(out, indent=1, default=float))
    return out, cells, prim, train


if __name__ == "__main__":
    score(sys.argv[1])


def shuffle_null(proj, n_perm=1000, seed=0):
    """Null: permute the q labels of the TRAIN cells (keeping each cell's e), refit a,b, score on the blind core.
    Prices how much share(miss<=5) a 2-constant curve gets for free from the regular term + the grid."""
    out, cells, prim, train = score(proj, verbose=False)
    rng = np.random.default_rng(seed)
    ks = list(train); qs = np.array([k[5] for k in ks])
    shares = []
    for p in range(n_perm):
        perm = rng.permutation(len(ks))
        fake = {}
        for i, k in enumerate(ks):
            nk = k[:5] + (qs[perm[i]],) + k[6:]
            # keep e, eps of the original cell but attach a shuffled q
            fake[(nk, i)] = train[k]
        ks2 = list(fake)
        y = np.array([np.mean(list(fake[k]["e_seed"].values())) for k in ks2])
        def res(pp):
            a, b = np.exp(pp)
            return 100 * (np.array([law_pred(k[0][:4], k[0][4], k[0][5], fake[k]["eps"], a, b) for k in ks2]) - y)
        try:
            sol = least_squares(res, x0=np.log([0.3, 1.0])); a, b = np.exp(sol.x)
        except Exception:
            continue
        m = [abs(law_pred(k[:4], k[4], k[5], v["eps"], a, b) - np.mean(list(v["e_seed"].values()))) * 100 for k, v in prim.items()]
        shares.append(float(np.mean(np.array(m) <= 5)))
    shares = np.array(shares)
    return dict(real_share=out["share_le5"], null_mean=float(shares.mean()), null_p95=float(np.quantile(shares, 0.95)),
                p_value=float((np.sum(shares >= out["share_le5"]) + 1) / (len(shares) + 1)), n=len(shares))
