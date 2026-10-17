"""Hugo (skeptic2) independent checker for fisher-pullback-transfer.

Written from PREREG.md + DEVIATIONS.md text only. Reads raw data files; never imports the build code.
Part A (clean data): probe AUROCs, filter, Fisher damping, identifiability gate.
Part B (attack data, when present): measured drops, predicted drops (P = I, Sigma, (F+lam I)^-1, shared),
R^2 verdict, per-target R^2, bootstrap over targets, nulls.
"""
import json, sys, glob, os
import numpy as np

_here = os.path.dirname(os.path.abspath(__file__))
D = next(p for p in (os.path.join(_here, "../../data/"),  # receipts location: projects/<slug>/skeptics/skeptic2/
                     os.path.join(_here, "../../kran-research/projects/fisher-pullback-transfer/data/"))
         if os.path.isdir(p))


def auroc(s, y):
    """Mann-Whitney AUROC with average ranks for ties. y in {0,1}."""
    s = np.asarray(s, float); y = np.asarray(y, bool)
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s)); ss = s[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and ss[j + 1] == ss[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1
        i = j + 1
    n1 = y.sum(); n0 = len(y) - n1
    return (ranks[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def r2(pred, meas):
    pred = np.asarray(pred); meas = np.asarray(meas)
    return 1 - ((meas - pred) ** 2).sum() / ((meas - meas.mean()) ** 2).sum()


probes = json.load(open(D + "probes/probes.json"))
sel = json.load(open(D + "selection.json"))
PW = np.load(D + "probes/probe_weights.npz")
W, bW = PW["W"], PW["b"]
name2i = {p["name"]: p["idx"] for p in probes}
FS = np.load(D + "fisher/fisher_sigma.npz")

acts = {}
def clean(src):
    if src not in acts:
        z = np.load(D + f"clean/acts_{src}.npz")
        acts[src] = (z["X"].astype(np.float64), z["label"], z["split"])
    return acts[src]

def test_xy(name):
    src, lab = name.split(":", 1)
    X, L, S = clean(src)
    m = S == 1
    return X[m], (L[m] == lab).astype(int)

# ---------------- Part A ----------------
def part_a():
    out = {}
    bad = []
    for p in probes:
        X, y = test_xy(p["name"])
        a = auroc(X @ W[p["idx"]], y)
        if abs(a - p["auroc_test"]) > 1e-6:
            bad.append((p["name"], a, p["auroc_test"]))
    out["auroc_mismatch"] = bad
    kept = [p["name"] for p in probes if p["auroc_test"] >= 0.75]
    out["n_kept"] = len(kept)
    out["dropped_match"] = sorted(set(name2i) - set(kept)) == sorted(sel["dropped_by_filter"])
    F = FS["F_common"]
    ev = np.linalg.eigvalsh(F)
    out["median_eig"] = float(np.median(ev)); out["n_null_eig"] = int((ev < 1e-8).sum())
    out["F_half_corr"] = float(np.corrcoef(FS["F_common_half1"].ravel(), FS["F_common_half2"].ravel())[0, 1])
    # all-ones direction in F
    one = np.ones(512) / np.sqrt(512)
    out["ones_rayleigh_over_median"] = float(one @ F @ one / np.median(ev))
    return out


def P_mats(c=1.0):
    F = FS["F_common"]; ev = np.linalg.eigvalsh(F)
    lam = c * np.median(ev)
    return {"I": np.eye(512), "Sigma": FS["Sigma_LW"].astype(np.float64),
            "F": np.linalg.inv(F + lam * np.eye(512))}


def r_P(P, u, w):
    return (u @ P @ w) / (w @ P @ w)


def pred_drop(uname, wname, kappa_ratio, P, shared=False):
    """PREREG primary: s_u_pred = s_u + ratio * r_P * s_w on u's clean test set; drop = AUROC0 - AUROC_pred.
    kappa_ratio = delta_w / G_w(w)."""
    X, y = test_xy(uname)
    u, w = W[name2i[uname]], W[name2i[wname]]
    su, sw = X @ u, X @ w
    r = 1.0 if shared else r_P(P, u, w)
    return auroc(su, y) - auroc(su + kappa_ratio * r * sw, y)


def ident_gate():
    Ps = P_mats()
    scored = sel["scored_probes"]; tg = sel["targets"]
    vals = {k: [] for k in Ps}
    for wn in tg:
        for un in scored:
            if un == wn: continue
            for k, P in Ps.items():
                vals[k].append(pred_drop(un, wn, -0.5, P))
    v = {k: np.array(x) for k, x in vals.items()}
    return {"n": len(v["F"]), "corr_I_F": np.corrcoef(v["I"], v["F"])[0, 1],
            "corr_I_S": np.corrcoef(v["I"], v["Sigma"])[0, 1], "corr_S_F": np.corrcoef(v["Sigma"], v["F"])[0, 1],
            "sd": {k: x.std() for k, x in v.items()}}


# ---------------- Part B ----------------
from math import erf, sqrt
def Phi(x): return 0.5 * (1 + erf(x / sqrt(2)))

SEC = json.load(open(D + "probes/secondary.json"))
SW = np.load(D + "probes/secondary_weights.npz")["W"]
FB = np.load(D + "fisher/fisher_big.npz")


def mats(c=1.0, which="big"):
    F = FB["F_common"] if which == "big" else FS["F_common"]
    ev = np.linalg.eigvalsh(F)
    return {"I": np.eye(512), "Sigma": FS["Sigma_LW"].astype(np.float64),
            "F": np.linalg.inv(F + c * np.median(ev) * np.eye(512))}


def load_runs():
    runs = []
    for jf in sorted(glob.glob(os.environ.get("RUNDIR", D + "runs/") + "*__seed*.json")):
        j = json.load(open(jf))
        j["_npz"] = jf[:-5] + ".npz"
        runs.append(j)
    return runs


def test_idx(src):
    X, L, S = clean(src)
    return X[S == 1], L[S == 1]


def gsd(s, y):
    m1, m0 = y == 1, y == 0
    sd = np.sqrt((((s[m1] - s[m1].mean()) ** 2).sum() + ((s[m0] - s[m0].mean()) ** 2).sum()) / (len(s) - 2))
    return s[m1].mean() - s[m0].mean(), sd


def run_tables(j, Ps):
    """Own scoring from attacked activations. Returns primary rows, secondary rows, run summary."""
    acts_f = j["_npz"][:-4] + "__acts.npz"   # build split attacked activations into a side file at 16:13 CEST
    z = np.load(acts_f if os.path.exists(acts_f) else j["_npz"])
    wn = j["target"]; wsrc, wlab = wn.split(":", 1)
    w = W[name2i[wn]]
    Xw, Lw = test_idx(wsrc); yw = (Lw == wlab).astype(int)
    Aw = z[f"attacked_acts__{wsrc}"].astype(np.float64)
    assert Aw.shape == Xw.shape
    sw0, sw1 = Xw @ w, Aw @ w
    G_w, sdw0 = gsd(sw0, yw); G1, sdw1 = gsd(sw1, yw)
    delta_w = G1 - G_w
    ratio = delta_w / G_w
    summ = {"target": wn, "seed": j["seed"], "lambda_kl": j.get("lambda_kl"), "loss": j.get("loss", j.get("dprime_sd")),
            "success": j.get("success"), "stop_step": j.get("stop_step"), "retain_kl": j.get("retain_kl_heldout"),
            "auroc_w_clean": auroc(sw0, yw), "auroc_w_att": auroc(sw1, yw), "sd_w_clean": sdw0, "sd_w_att": sdw1,
            "delta_w": delta_w, "G_w": G_w, "ratio": ratio, "json_delta_w": j.get("delta_w"), "json_G_w": j.get("G_w")}
    e = (Aw - Xw).mean(0)
    one = np.ones(512) / np.sqrt(512)
    summ["edit_cos"] = {k: float(e @ (P @ w) / np.linalg.norm(e) / np.linalg.norm(P @ w)) for k, P in Ps.items()}
    summ["edit_cos"]["ones"] = float(e @ one / np.linalg.norm(e))
    prim = []
    for un in sel["scored_probes"]:
        if un == wn: continue
        src, lab = un.split(":", 1)
        X, L = test_idx(src); y = (L == lab).astype(int)
        u = W[name2i[un]]
        su, sw = X @ u, X @ w
        A = z[f"attacked_acts__{src}"].astype(np.float64)
        a0 = auroc(su, y)
        Gu, sdu = gsd(su, y); Guw, _ = gsd(sw, y)
        row = {"target": wn, "probe": un, "meas": a0 - auroc(A @ u, y), "auroc0": a0,
               "labcorr": abs(np.corrcoef(y, sw)[0, 1])}
        for k, P in Ps.items():
            r = r_P(P, u, w)
            row[k] = a0 - auroc(su + ratio * r * sw, y)
            row["cos_" + k] = (u @ P @ w) / np.sqrt((u @ P @ u) * (w @ P @ w))
            row["gauss_" + k] = Phi(Gu / sdu / sqrt(2)) - Phi((Gu + ratio * r * Guw) / sdu / sqrt(2))
        row["shared"] = a0 - auroc(su + ratio * sw, y)
        prim.append(row)
    sec = []
    for s in SEC:
        if s["target"] != wn or not s["kept"]: continue
        u = SW[s["idx"]]
        su = Xw @ u
        Gu, sdu = gsd(su, yw); Guw = G_w
        a0 = auroc(su, yw)
        row = {"target": wn, "probe": s["name"], "meas": a0 - auroc(Aw @ u, yw), "auroc0": a0}
        for k, P in list(Ps.items()) + [("shared", None)]:
            r = 1.0 if P is None else r_P(P, u, w)
            row[k] = Phi(Gu / sdu / sqrt(2)) - Phi((Gu + ratio * r * Guw) / sdu / sqrt(2))
            row["emp_" + k] = a0 - auroc(su + ratio * r * sw0, yw)
        sec.append(row)
    return prim, sec, summ


def seed_mean(rows, keys):
    from collections import defaultdict
    g = defaultdict(list)
    for r in rows: g[(r["target"], r["probe"])].append(r)
    out = []
    for (t, p), rs in g.items():
        o = {"target": t, "probe": p, "n_seeds": len(rs)}
        for k in keys: o[k] = float(np.mean([r[k] for r in rs]))
        out.append(o)
    return out


def verdict(pairs, keys=("F", "Sigma", "I", "shared"), nboot=10000):
    meas = np.array([p["meas"] for p in pairs])
    R = {k: r2([p[k] for p in pairs], meas) for k in keys}
    res = {"n_pairs": len(pairs), "R2": R, "sd_meas": float(meas.std()), "n_abs_gt_0.02": int((np.abs(meas) > 0.02).sum()),
           "mean_meas": float(meas.mean()),
           "margin_metric": R["F"] - max(R["Sigma"], R["I"]), "margin_shared": R["F"] - R["shared"]}
    res["CONFIRMED_rule"] = bool(R["F"] >= 0.6 and res["margin_metric"] >= 0.10 and res["margin_shared"] >= 0.10)
    tg = sorted(set(p["target"] for p in pairs))
    res["per_target_R2"] = {t: {k: r2([p[k] for p in pairs if p["target"] == t], [p["meas"] for p in pairs if p["target"] == t]) for k in keys} for t in tg}
    rng = np.random.default_rng(0); bs = {k: [] for k in list(keys) + ["m_metric", "m_shared"]}
    by_t = {t: [p for p in pairs if p["target"] == t] for t in tg}
    for _ in range(nboot):
        sub = [p for t in rng.choice(tg, len(tg)) for p in by_t[t]]
        mm = np.array([p["meas"] for p in sub])
        if mm.std() == 0: continue
        rr = {k: r2([p[k] for p in sub], mm) for k in keys}
        for k in keys: bs[k].append(rr[k])
        bs["m_metric"].append(rr["F"] - max(rr["Sigma"], rr["I"])); bs["m_shared"].append(rr["F"] - rr["shared"])
    res["boot95"] = {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in bs.items() if v}
    return res


def nulls(pairs, n_perm=1000):
    meas = np.array([p["meas"] for p in pairs]); tg = np.array([p["target"] for p in pairs])
    pf = np.array([p["F"] for p in pairs]); obs = r2(pf, meas)
    rng = np.random.default_rng(1)
    perm = []
    for _ in range(n_perm):
        q = pf.copy()
        for t in set(tg):
            m = tg == t; q[m] = rng.permutation(q[m])
        perm.append(r2(q, meas))
    out = {"R2_F": obs, "perm_within_target_mean": float(np.mean(perm)), "perm_within_target_p": float((np.array(perm) >= obs).mean())}
    def loto(xkey):
        pred = np.zeros(len(pairs)); x = np.array([p[xkey] for p in pairs])
        for t in set(tg):
            tr = tg != t
            c = np.linalg.lstsq(np.c_[x[tr], np.ones(tr.sum())], meas[tr], rcond=None)[0]
            pred[~tr] = c[0] * x[~tr] + c[1]
        return r2(pred, meas)
    for k in ("labcorr", "cos_F", "cos_Sigma", "cos_I"):
        out["loto_" + k] = loto(k)
    frag = [np.mean([q["meas"] for q in pairs if q["probe"] == p["probe"] and q["target"] != p["target"]] or [0.0]) for p in pairs]
    out["fragility_R2"] = r2(frag, meas)
    return out


def random_metric_null(ok_summs, pp, n=100):
    """P = Q diag(spec) Q', spec = eig((F+lam I)^-1), random rotation Q. Same seed-mean rule."""
    F = FB["F_common"]; ev = np.linalg.eigvalsh(F); spec = 1 / (ev + np.median(ev))
    meas = {(p["target"], p["probe"]): p["meas"] for p in pp}
    cache = {}
    for un in sel["scored_probes"]:
        src, lab = un.split(":", 1); X, L = test_idx(src); cache[un] = (X, (L == lab).astype(int))
    rng = np.random.default_rng(2); out = []
    for _ in range(n):
        Q, _ = np.linalg.qr(rng.standard_normal((512, 512)))
        P = (Q * spec) @ Q.T
        preds = {}
        for sm in ok_summs:
            wn = sm["target"]; w = W[name2i[wn]]
            for un in sel["scored_probes"]:
                if un == wn: continue
                X, y = cache[un]; u = W[name2i[un]]; su, sw = X @ u, X @ w
                preds.setdefault((wn, un), []).append(auroc(su, y) - auroc(su + sm["ratio"] * r_P(P, u, w) * sw, y))
        ks = list(preds)
        out.append(r2([np.mean(preds[k]) for k in ks], [meas[k] for k in ks]))
    return out


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "clean":
        print("PART A", part_a()); print("IDENT", ident_gate()); sys.exit()
    which = os.environ.get("FWHICH", "big"); c = float(os.environ.get("FC", "1.0"))
    Ps = mats(c, which)
    allp, alls, summs, ok = [], [], [], []
    for j in load_runs():
        p, s, sm = run_tables(j, Ps)
        summs.append(sm)
        print("RUN", json.dumps(sm, default=float))
        if sm["lambda_kl"] == 0 or not sm["success"]:
            continue
        ok.append(sm); allp += p; alls += s
    keys = ["meas", "F", "Sigma", "I", "shared", "labcorr", "cos_F", "cos_Sigma", "cos_I", "gauss_F", "gauss_Sigma", "gauss_I", "auroc0"]
    pp = seed_mean(allp, keys)
    print("PRIMARY", json.dumps(verdict(pp), default=float, indent=1))
    g = [{**p, "F": p["gauss_F"], "Sigma": p["gauss_Sigma"], "I": p["gauss_I"]} for p in pp]
    print("PRIMARY_GAUSS_SENS", json.dumps({k: v for k, v in verdict(g, nboot=0).items() if k in ("R2", "margin_metric", "margin_shared")}, default=float))
    print("NULLS", json.dumps(nulls(pp), default=float))
    same = [p for p in pp if p["probe"].split(":")[0] == p["target"].split(":")[0]]
    diff = [p for p in pp if p["probe"].split(":")[0] != p["target"].split(":")[0]]
    for nm, ss in (("SAME_DATASET", same), ("CROSS_DATASET", diff)):
        if len(ss) > 2:
            v = verdict(ss, nboot=0); print(nm, json.dumps({k: v[k] for k in ("n_pairs", "R2", "sd_meas", "n_abs_gt_0.02")}, default=float))
    if os.environ.get("RANDMETRIC"):
        rm = random_metric_null(ok, pp, int(os.environ["RANDMETRIC"]))
        print("RANDOM_METRIC_NULL", json.dumps({"mean": np.mean(rm), "p95": np.percentile(rm, 95), "max": max(rm)}, default=float))
    sp = []
    if alls:
        sk = ["meas", "F", "Sigma", "I", "shared", "emp_F", "emp_Sigma", "emp_I", "emp_shared"]
        sp = seed_mean(alls, sk)
        print("SECONDARY", json.dumps(verdict(sp), default=float, indent=1))
        e = [{**p, "F": p["emp_F"], "Sigma": p["emp_Sigma"], "I": p["emp_I"], "shared": p["emp_shared"]} for p in sp]
        print("SECONDARY_EMP_SENS", json.dumps({k: v for k, v in verdict(e, nboot=0).items() if k in ("R2", "margin_metric", "margin_shared")}, default=float))
    json.dump({"primary_pairs": pp, "secondary_pairs": sp, "runs": summs},
              open(os.path.join(os.path.dirname(__file__), f"hugo_tables_{which}_c{c}.json"), "w"), default=float, indent=1)
