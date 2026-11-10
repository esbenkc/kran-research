"""Vera's independent checker for cue-size-law. Reads ONLY raw run JSONs (data/runs/*.json) and data/scan.json.
Recomputes D(t) from px_off/px_on, the switch time T by PREREG s3, and every fit of PREREG s5/s9/s10 with
my own code (numpy + scipy). Writes vera_check.json.
"""
import json, glob, os, sys, math
import numpy as np
from scipy import optimize, stats

P = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))
OUT = os.path.dirname(os.path.abspath(__file__))
scan = json.load(open(f"{P}/data/scan.json"))
XS = {float(k): v["x"] for k, v in scan["cues"].items() if k.replace(".", "").isdigit()}
X_REAL = scan["x_real"] if not isinstance(scan["x_real"], dict) else scan["x_real"]["x"]
ATOP = scan["alpha_top"]


def switch_time(curve, thr=0.10):
    c = sorted(curve, key=lambda r: r["step"])
    g0 = c[0]["px_off"] - c[0]["px_on"]
    D = [(r["step"], (r["px_off"] - r["px_on"]) - g0) for r in c]
    for i in range(1, len(D) - 1):
        if D[i][1] >= thr and D[i + 1][1] >= thr:
            return D[i][0], D, g0
    return None, D, g0


def load():
    runs = []
    for f in sorted(glob.glob(f"{P}/data/runs/*.json")):
        d = json.load(open(f))
        a = d["args"]
        T, D, g0 = switch_time(d["dev_curve"])
        B = a["steps"]
        last = max(r["step"] for r in d["dev_curve"])
        # censored only if the run reached B without a confirmed switch
        complete = last >= B - 64
        # T_dense: same rule on the union of the frozen grid and the D1 dense evals (every 8 to 2,048)
        Td = None
        if "dense_curve" in d:
            byst = {r["step"]: r for r in d["dev_curve"]}
            byst.update({r["step"]: r for r in d["dense_curve"]})
            Td, _, _ = switch_time(list(byst.values()))
        runs.append(dict(name=d["name"], arm=a.get("arm"), cue_dir=a.get("cue_dir"), alpha=a.get("cue_alpha"),
                         seed=a.get("seed"), opt=a.get("opt"), lr=a.get("lr"), u_file=a.get("u_file"),
                         T=T, censored=(T is None and complete), incomplete=(T is None and not complete), B=B,
                         maxD=max(x[1] for x in D), lastD=D[-1][1], g0=g0, their_T=d.get("T"),
                         their_cens=d.get("censored"), T_dense=Td, their_T_dense=d.get("T_dense"),
                         has_dense="dense_curve" in d, D=D, curve=d["dev_curve"], args=a))
    return runs


def fit_block(unc, sw, keyf, Tkey="T", label=""):
    lx = [keyf(r) for r in unc]; ly = [math.log(r[Tkey]) for r in unc]
    b, a, ci_t, se = ols(lx, ly)
    groups = {}
    for r in unc:
        groups.setdefault(keyf(r), []).append(math.log(r[Tkey]))
    ci_b = boot(groups)
    wider = (min(ci_t[0], ci_b[0]), max(ci_t[1], ci_b[1]))
    lxa = np.array([keyf(r) for r in sw]); cens = np.array([r[Tkey] is None for r in sw])
    lya = np.array([math.log(r[Tkey]) if r[Tkey] else math.log(r["B"]) for r in sw])
    tb = tobit(lxa, lya, cens, math.log(max(r["B"] for r in sw)))
    print(f"[{label}] OLS slope {b:.3f}  t-CI [{ci_t[0]:.3f},{ci_t[1]:.3f}]  boot-CI [{ci_b[0]:.3f},{ci_b[1]:.3f}]"
          f"  n={len(unc)} alphas={len(groups)} censored={int(cens.sum())} | Tobit {tb[0]:.3f} [{tb[1][0]:.3f},{tb[1][1]:.3f}]")
    return dict(slope=b, intercept=a, ci_t=ci_t, ci_boot=list(ci_b), ci_wider=wider, n=len(unc), n_alpha=len(groups),
                tobit=dict(slope=tb[0], ci=tb[1]), n_censored=int(cens.sum()))


def status(f, sw, keyf):
    lo, hi = f["ci_wider"]
    in_band = -2.5 <= f["slope"] <= -1.5
    conf = in_band and not (lo <= -1 <= hi) and not (lo <= 0 <= hi)
    rej = hi < -2.5 or lo > -1.5
    if f["n_censored"] >= 2:
        tl, th = f["tobit"]["ci"]
        conf = conf and (-2.5 <= f["tobit"]["slope"] <= -1.5)
        rej = rej and (th < -2.5 or tl > -1.5)
    return "CONFIRMED" if conf else "REJECTED" if rej else "UNCONFIRMED"


def perm_null(unc, keyf, n=20000, seed=0):
    """Alpha-label permutation: shuffle the T values over runs, refit the slope. p = P(|slope_perm| >= |slope_obs|)."""
    rng = np.random.default_rng(seed)
    lx = np.array([keyf(r) for r in unc]); ly = np.array([math.log(r["T"]) for r in unc])
    obs = np.polyfit(lx, ly, 1)[0]
    sl = np.array([np.polyfit(lx, rng.permutation(ly), 1)[0] for _ in range(n)])
    return obs, float((np.abs(sl) >= abs(obs)).mean()), float(np.percentile(sl, 2.5)), float(np.percentile(sl, 97.5))


def ols(lx, ly):
    lx, ly = np.asarray(lx), np.asarray(ly)
    n = len(lx); X = np.c_[np.ones(n), lx]
    beta, *_ = np.linalg.lstsq(X, ly, rcond=None)
    res = ly - X @ beta
    s2 = res @ res / (n - 2) if n > 2 else np.nan
    se = math.sqrt(s2 * np.linalg.inv(X.T @ X)[1, 1]) if n > 2 else np.nan
    tq = stats.t.ppf(0.975, n - 2) if n > 2 else np.nan
    return beta[1], beta[0], (beta[1] - tq * se, beta[1] + tq * se), se


def boot(groups, seed=0, n=2000):
    """groups: dict x -> list of logT. Resample runs within each x (PREREG s5)."""
    rng = np.random.default_rng(seed)
    keys = sorted(groups)
    sl = []
    for _ in range(n):
        lx, ly = [], []
        for k in keys:
            v = groups[k]
            pick = rng.choice(v, len(v), replace=True)
            lx += [k] * len(v); ly += list(pick)
        if len(set(lx)) >= 2:
            sl.append(np.polyfit(lx, ly, 1)[0])
    return np.percentile(sl, [2.5, 97.5])


def tobit(lx, ly, cens, logB):
    lx, ly, cens = map(np.asarray, (lx, ly, cens))
    def nll(p):
        a, b, ls = p; s = math.exp(ls)
        mu = a + b * lx
        ll = np.where(cens, stats.norm.logsf((logB - mu) / s), stats.norm.logpdf((ly - mu) / s) - ls)
        return -ll.sum()
    b0, a0, *_ = ols(lx[~cens], ly[~cens]) if (~cens).sum() > 2 else (-1.0, 8.0)
    r = optimize.minimize(nll, [a0, b0, math.log(0.3)], method="Nelder-Mead", options=dict(maxiter=20000, xatol=1e-8, fatol=1e-10))
    H = numerical_hessian(nll, r.x)
    cov = np.linalg.inv(H)
    se = math.sqrt(cov[1, 1])
    return r.x[1], (r.x[1] - 1.96 * se, r.x[1] + 1.96 * se), r.x


def numerical_hessian(f, x, h=1e-4):
    n = len(x); H = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            e_i = np.eye(n)[i] * h; e_j = np.eye(n)[j] * h
            H[i, j] = (f(x + e_i + e_j) - f(x + e_i - e_j) - f(x - e_i + e_j) + f(x - e_i - e_j)) / (4 * h * h)
    return H


SWEEP_ALPHAS = [ATOP / 2 ** k for k in range(6)]


def is_sweep(r):
    return (r["name"].startswith(("p0_", "sw_")) and r["arm"] == "cue" and r["cue_dir"] == "rand"
            and r["opt"] == "adamw" and r["u_file"] == "data/cue_u.npy"
            and any(abs(r["alpha"] - a) < 1e-12 for a in SWEEP_ALPHAS))


if __name__ == "__main__":
    runs = load()
    byname = {r["name"]: r for r in runs}
    print(f"{'name':34s} alpha     seed opt   T(v)  T(b)  Td(v) Td(b) cens  maxD   lastD")
    for r in runs:
        print(f"{r['name']:34s} {r['alpha']!s:9.9} {r['seed']!s:4} {r['opt']!s:5} {r['T']!s:5} {r['their_T']!s:5} "
              f"{r['T_dense']!s:5} {r['their_T_dense']!s:5} {r['censored']!s:5} {r['maxD']:.3f} {r['lastD']:.3f}"
              + ("  INCOMPLETE" if r["incomplete"] else ""))
    mism = [r["name"] for r in runs if r["T"] != r["their_T"] or (r["has_dense"] and r["T_dense"] != r["their_T_dense"])]
    print("T / T_dense mismatches vs builder:", mism)
    # rd_ determinism: a re-run with dense evals must reproduce the original frozen-grid trajectory
    det = {}
    for r in runs:
        if r["name"].startswith("rd_") and r["name"][3:] in byname:
            o = byname[r["name"][3:]]
            a = {c["step"]: (c["px_off"], c["px_on"]) for c in o["curve"]}
            b = {c["step"]: (c["px_off"], c["px_on"]) for c in r["curve"]}
            common = sorted(set(a) & set(b))
            dmax = max(abs(a[s][0] - b[s][0]) + abs(a[s][1] - b[s][1]) for s in common)
            det[o["name"]] = dict(T_orig=o["T"], T_rd=r["T"], Td_rd=r["T_dense"], max_abs_dP=dmax, n_common=len(common))
            o["T_dense"] = r["T_dense"]          # the dense readout of an early run comes from its rd_ twin
    if det:
        same = sum(v["T_orig"] == v["T_rd"] for v in det.values())
        print(f"rd_ determinism: frozen T reproduced in {same} of {len(det)}; max |dP| per pair:",
              {k: round(v['max_abs_dP'], 5) for k, v in det.items()})
    sw = [r for r in runs if is_sweep(r) and not r["incomplete"]]
    unc = [r for r in sw if r["T"] is not None]
    kx = lambda r: math.log(XS[r["alpha"]]); ka = lambda r: math.log(r["alpha"])
    out = {"runs": [{k: v for k, v in r.items() if k not in ("D", "curve")} for r in runs], "mismatch": mism,
           "determinism": det}
    out["fit_x"] = fx = fit_block(unc, sw, kx, label="x, frozen T")
    out["fit_alpha"] = fa = fit_block(unc, sw, ka, label="alpha, frozen T")
    out["status_T"] = status(fx, sw, kx)
    dense_ok = [r for r in sw if r["T_dense"] is not None]
    if len(dense_ok) == len(sw):
        swd = [dict(r, T=r["T_dense"]) for r in sw]
        out["fit_x_dense"] = fxd = fit_block(swd, swd, kx, label="x, T_dense")
        out["status_T_dense"] = status(fxd, swd, kx)
    else:
        out["status_T_dense"] = f"n/a ({len(dense_ok)} of {len(sw)} sweep runs have a dense readout)"
    print("STATUS from frozen T:", out["status_T"], "| from T_dense:", out["status_T_dense"])
    # PREREG s9.1 two halves (k=0..2 vs k=3..5), on x
    top = [r for r in unc if r["alpha"] >= ATOP / 4 - 1e-12]; bot = [r for r in unc if r["alpha"] < ATOP / 4 - 1e-12]
    hs = [np.polyfit([kx(r) for r in h], [math.log(r["T"]) for r in h], 1)[0] for h in (top, bot)]
    out["halves"] = dict(top=hs[0], bottom=hs[1], diff=abs(hs[0] - hs[1]))
    print(f"halves: k0-2 {hs[0]:.3f}  k3-5 {hs[1]:.3f}  |diff| {abs(hs[0]-hs[1]):.3f} (rule: >0.5 -> not one power law)")
    # Prereg parts (scored on log alpha, PREREG s10.2)
    p1 = any(r["T"] is not None and r["T"] <= 4096 for r in unc if abs(r["alpha"] - ATOP) < 1e-12)
    lo, hi = fa["ci_wider"]
    p2 = (-2.5 <= fa["slope"] <= -1.5) and not (lo <= 0 <= hi)
    amin = min(r["alpha"] for r in sw)
    Tfit = math.exp(fa["intercept"] + fa["slope"] * math.log(amin))
    smallest = [r for r in sw if r["alpha"] == amin]
    p3 = all(r["T"] is not None and r["T"] <= 3 * Tfit for r in smallest)
    xs = [XS[r["alpha"]] for r in sw]
    p4 = min(xs) <= X_REAL <= max(xs)
    out["prereg"] = dict(part1=p1, part2=p2, part3=p3, part3_Tfit=Tfit, part3_T=[r["T"] for r in smallest], part4=p4,
                         outcome="held" if all((p1, p2, p3, p4)) else "failed")
    print("prereg parts:", out["prereg"])
    # Null 1: alpha-label permutation
    obs, p, l, h = perm_null(unc, kx)
    out["perm_null"] = dict(obs=obs, p_two_sided=p, null_95=[l, h], n_perm=20000)
    print(f"permutation null: observed {obs:.3f}; null 95% [{l:.3f},{h:.3f}]; p(|perm| >= |obs|) = {p:.5f}")
    # Null 2: controls must not switch
    ctrl = {r["name"]: dict(T=r["T"], T_dense=r["T_dense"], maxD=r["maxD"], lastD=r["lastD"], B=r["B"],
                            complete=not r["incomplete"])
            for r in runs if r["arm"] != "cue" or (r["alpha"] == 0)}
    out["controls"] = ctrl
    print("controls:", ctrl)
    json.dump(out, open(f"{OUT}/vera_check.json", "w"), indent=1, default=str)
