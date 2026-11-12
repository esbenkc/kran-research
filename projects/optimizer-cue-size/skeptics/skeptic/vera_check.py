"""Vera's independent checker for theory-1112-optimizer-sets.

Reads ONLY the raw run JSONs (dense dev curves), recomputes T with my own code, and applies the frozen status rule
(#20280 point 1 / #20275 b): one pooled fit log T ~ optimizer + optimizer:log x, shared residual SD, t interval;
Tobit if any run is censored; 10,000-draw case bootstrap within optimizer x size; the WIDER interval decides.

YES = (SGD - AdamW) slope <= -0.6 with CI-high < 0, AND |sign - AdamW| <= 0.3 (point).
NO  = CI-low of (SGD - AdamW) > -0.6, OR the CI of (sign - AdamW) does not intersect [-0.3, +0.3].
else UNSETTLED.
"""
import glob
import json
import os
import sys

import numpy as np
from scipy import optimize, stats

THR = 0.10


def merged_curve(run):
    """Union of dev_curve and dense_curve (the JSON stores them as disjoint step lists)."""
    by = {}
    for c in run.get("dev_curve", []) + run.get("dense_curve", []):
        by[c["step"]] = c
    return [by[s] for s in sorted(by)]


def my_T(curve, thr=THR):
    """First eval step s with D >= thr at s and at the next eval. D = gap - gap(step 0)."""
    curve = sorted(curve, key=lambda c: c["step"])
    g0 = [c for c in curve if c["step"] == 0]
    g0 = g0[0]["gap"] if g0 else 0.0
    for a, b in zip(curve, curve[1:]):
        if a["gap"] - g0 >= thr and b["gap"] - g0 >= thr:
            return a["step"], False
    return curve[-1]["step"], True  # censored at the last eval


def design(rows, opts):
    """rows: (opt, logx, logT). Columns: one intercept and one slope per optimizer."""
    X = np.zeros((len(rows), 2 * len(opts)))
    for i, (o, lx, _) in enumerate(rows):
        k = opts.index(o)
        X[i, 2 * k] = 1.0
        X[i, 2 * k + 1] = lx
    return X


def ols(rows, opts):
    X = design(rows, opts)
    y = np.array([r[2] for r in rows])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    res = y - X @ beta
    df = len(y) - X.shape[1]
    s2 = (res @ res) / df
    cov = s2 * np.linalg.inv(X.T @ X)
    return beta, cov, df, np.sqrt(s2)


def contrast(beta, cov, opts, a, b):
    i, j = 2 * opts.index(a) + 1, 2 * opts.index(b) + 1
    d = beta[i] - beta[j]
    se = np.sqrt(cov[i, i] + cov[j, j] - 2 * cov[i, j])
    return d, se


def tobit(rows, cens, opts):
    """Right-censored normal regression (censored: true log T >= observed)."""
    X = design(rows, opts)
    y = np.array([r[2] for r in rows])
    c = np.array(cens, bool)
    b0, *_ = np.linalg.lstsq(X, y, rcond=None)

    def nll(p):
        b, ls = p[:-1], p[-1]
        s = np.exp(ls)
        mu = X @ b
        ll = np.where(c, stats.norm.logsf((y - mu) / s), stats.norm.logpdf((y - mu) / s) - ls)
        return -ll.sum()

    r = optimize.minimize(nll, np.r_[b0, np.log(0.2)], method="BFGS")
    cov = r.hess_inv[:-1, :-1]
    return r.x[:-1], cov, np.exp(r.x[-1])


def boot(rows, opts, pairs, n=10000, seed=0):
    rng = np.random.default_rng(seed)
    cells = {}
    for i, r in enumerate(rows):
        cells.setdefault((r[0], round(r[1], 4)), []).append(i)
    out = {p: [] for p in pairs}
    for _ in range(n):
        idx = [rng.choice(v) for v in cells.values() for _ in v]
        rr = [rows[i] for i in idx]
        X = design(rr, opts)
        y = np.array([r[2] for r in rr])
        b, *_ = np.linalg.lstsq(X, y, rcond=None)
        for (a, c) in pairs:
            out[(a, c)].append(b[2 * opts.index(a) + 1] - b[2 * opts.index(c) + 1])
    return {p: np.percentile(v, [2.5, 97.5]) for p, v in out.items()}


def tobit_wald(rows, cens, opts):
    """Tobit fit + Wald covariance from a finite-difference Hessian of the NLL (observed information)."""
    X = design(rows, opts)
    y = np.array([r[2] for r in rows])
    c = np.array(cens, bool)

    def nll(p):
        b, ls = p[:-1], p[-1]
        s = np.exp(ls)
        mu = X @ b
        return -np.where(c, stats.norm.logsf((y - mu) / s), stats.norm.logpdf((y - mu) / s) - ls).sum()

    b0, *_ = np.linalg.lstsq(X, y, rcond=None)
    r = optimize.minimize(nll, np.r_[b0, np.log(0.2)], method="Nelder-Mead",
                          options={"maxiter": 200000, "maxfev": 200000, "xatol": 1e-10, "fatol": 1e-12})
    r = optimize.minimize(nll, r.x, method="BFGS")
    p = r.x
    k = len(p)
    H = np.zeros((k, k))
    e = 1e-4
    for i in range(k):
        for j in range(k):
            pp = [p.copy() for _ in range(4)]
            pp[0][i] += e; pp[0][j] += e
            pp[1][i] += e; pp[1][j] -= e
            pp[2][i] -= e; pp[2][j] += e
            pp[3][i] -= e; pp[3][j] -= e
            H[i, j] = (nll(pp[0]) - nll(pp[1]) - nll(pp[2]) + nll(pp[3])) / (4 * e * e)
    cov = np.linalg.inv(H)
    return p[:-1], cov[:-1, :-1], np.exp(p[-1])


def status(sgd_ci, sgd_d, sign_ci, sign_d):
    yes = sgd_d <= -0.6 and sgd_ci[1] < 0 and abs(sign_d) <= 0.3
    no = sgd_ci[0] > -0.6 or sign_ci[0] > 0.3 or sign_ci[1] < -0.3
    return "YES" if yes and not no else ("NO" if no else "UNSETTLED")


P = os.path.expanduser("~/work/theory-1112-optimizer-sets/kran-research/projects/optimizer-cue-size")
SIZES = [2.0 ** -k for k in (5, 6, 7, 8)]


def load_runs():
    out = []
    for f in sorted(glob.glob(f"{P}/data/runs/*.json")):
        r = json.load(open(f))
        a = r["args"]
        cur = merged_curve(r)
        if not cur:
            continue
        t, cens = my_T(cur)
        out.append(dict(file=os.path.basename(f), name=r.get("name"), opt=a["opt"], lr=a["lr"], alpha=a["cue_alpha"],
                        seed=a["seed"], freeze_a=a.get("freeze_a", False), arm=a.get("arm"), group=a.get("group"),
                        no_cast=a.get("no_cast"), x=(r.get("cue_size") or {}).get("x"), T_me=t, cens_me=cens,
                        T_b=(r.get("T_dense") if r.get("T_dense") is not None else r.get("T")), cens_b=r.get("censored"), T_grid=r.get("T"), last=cur[-1]["step"], steps_run=r.get("steps_run"),
                        budget=a.get("steps")))
    return out


if __name__ == "__main__":
    sgd_lr = float(sys.argv[1]) if len(sys.argv) > 1 else None
    runs = load_runs()
    mism = []
    for r in runs:
        flag = "" if ((r["cens_me"] and r["cens_b"]) or (r["T_me"] == r["T_b"] and not r["cens_me"] and not r["cens_b"])) else "  <-- MISMATCH"
        if flag:
            mism.append(r["file"])
        print(f"{r['file']:44s} {r['opt']:6s} lr {r['lr']:<7g} a {r['alpha']:.6g} s{r['seed']} fa {int(bool(r['freeze_a']))} "
              f"x {r['x'] if r['x'] is None else round(r['x'], 3)} T_me {r['T_me']}{'+' if r['cens_me'] else ''} "
              f"T_b {r['T_b']}{'+' if r['cens_b'] else ''} last {r['last']} budget {r['budget']}{flag}")
    print("mismatches:", len(mism), mism)

    def headline(r):
        if r["arm"] != "cue" or r["freeze_a"] or r["group"] != 8 or not r["no_cast"]:
            return False
        if not any(abs(r["alpha"] - s) / s < 1e-6 for s in SIZES) or r["seed"] not in (0, 1):
            return False
        if r["opt"] == "sgd":
            return sgd_lr is not None and abs(r["lr"] - sgd_lr) / sgd_lr < 1e-9
        return r["opt"] in ("adamw", "sign") and abs(r["lr"] - 1e-4) < 1e-12

    hl = [r for r in runs if headline(r)]
    opts = [o for o in ("adamw", "sgd", "sign") if any(r["opt"] == o for r in hl)]
    print("headline runs:", len(hl), {o: sum(r["opt"] == o for r in hl) for o in opts})
    res = {"runs": runs, "headline_n": len(hl)}
    if len(opts) >= 2:
        rows = [(r["opt"], float(np.log(r["x"])), float(np.log(r["T_me"]))) for r in hl]
        cens = [r["cens_me"] for r in hl]
        beta, cov, df, sd = ols(rows, opts)
        res["ols"] = {o: {"slope": beta[2 * i + 1], "icpt": beta[2 * i],
                          "ci": (beta[2 * i + 1] + np.array([-1, 1]) * stats.t.ppf(.975, df) * np.sqrt(cov[2 * i + 1, 2 * i + 1])).tolist()}
                      for i, o in enumerate(opts)}
        print(f"OLS pooled: df {df} resid sd {sd:.3f}")
        for o in opts:
            print(f"  {o}: slope {res['ols'][o]['slope']:.3f} CI {np.round(res['ols'][o]['ci'], 3)}")
        pairs = [(o, "adamw") for o in opts if o != "adamw"]
        ci_t = {}
        for p in pairs:
            d, se = contrast(beta, cov, opts, *p)
            ci_t[p] = (d, np.array([d - stats.t.ppf(.975, df) * se, d + stats.t.ppf(.975, df) * se]))
        if any(cens):
            bt, covt, sdt = tobit_wald(rows, cens, opts)
            print(f"Tobit (censored {sum(cens)}): sd {sdt:.3f}")
            for p in pairs:
                d, se = contrast(bt, covt, opts, *p)
                ci_t[p] = (d, np.array([d - 1.96 * se, d + 1.96 * se]))
            for i, o in enumerate(opts):
                print(f"  tobit {o}: slope {bt[2 * i + 1]:.3f} +- {1.96 * np.sqrt(covt[2 * i + 1, 2 * i + 1]):.3f}")
        bci = boot(rows, opts, pairs) if not any(cens) else None
        dec = {}
        for p in pairs:
            d, ci = ci_t[p]
            wide = ci
            if bci is not None:
                b = bci[p]
                wide = ci if (ci[1] - ci[0]) >= (b[1] - b[0]) else b
                print(f"  d {p[0]}-adamw {d:.3f}  t/Wald CI {np.round(ci, 3)}  boot CI {np.round(b, 3)}  -> decides {np.round(wide, 3)}")
            else:
                print(f"  d {p[0]}-adamw {d:.3f}  Wald CI {np.round(ci, 3)} (bootstrap skipped: censored runs)")
            dec[p[0]] = (d, wide)
        res["contrasts"] = {k: {"d": v[0], "ci": list(map(float, v[1]))} for k, v in dec.items()}
        if "sgd" in dec and "sign" in dec:
            res["status"] = status(dec["sgd"][1], dec["sgd"][0], dec["sign"][1], dec["sign"][0])
            print("STATUS:", res["status"])
    json.dump(res, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "vera_check.json"), "w"), indent=1,
              default=float)
