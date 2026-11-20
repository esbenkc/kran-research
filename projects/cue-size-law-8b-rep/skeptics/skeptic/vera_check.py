"""Vera's independent checker for cue-size-law-8b-rep (written from PREREG text + raw run JSONs only).

Usage: python vera_check.py RUNS_DIR [--prefix l8_] [--seeds 5,6,7] [--ref06 DIR] [--out FILE]

From each run JSON it uses ONLY the raw curves (dev_curve + dense_curve: step, px_off, px_on) and args
(arm, k, seed, cue_alpha, share). It recomputes gap(0), D(t), the dense switch time T, censoring, per-seed OLS
slopes of ln T on ln share, the seed-level t-interval, a pooled Tobit slope, C1/C2 checks, S8 ratios, and nulls.
"""
import argparse, glob, json, math, os, sys
import numpy as np
from scipy import stats, optimize

THR = 0.10
BUDGET = 8192
BAND = (-1.4, -0.7)
S_K = [0.05469028438159526 / 2 ** k for k in range(6)]  # frozen grid, 5.469% / 2^k


def load(path):
    with open(path) as f:
        return json.load(f)


def curve(d):
    """Merged eval curve -> sorted [(step, gap)], gap = px_off - px_on, recomputed from raw P(X)."""
    pts = {}
    for c in (d.get("dev_curve") or []) + (d.get("dense_curve") or []):
        g = float(c["px_off"]) - float(c["px_on"])
        s = int(c["step"])
        if s in pts and abs(pts[s] - g) > 1e-9:
            raise SystemExit(f"{d.get('name')}: two different gaps at step {s}")
        pts[s] = g
    return sorted(pts.items())


def switch_time(pts):
    """First eval t with D(t) >= 0.10 and D(next eval) >= 0.10; D = gap - gap(0). Returns (T or None, last_step, maxD)."""
    if not pts or pts[0][0] != 0:
        raise SystemExit("curve has no step-0 eval")
    g0 = pts[0][1]
    D = [(s, g - g0) for s, g in pts]
    for i in range(len(D) - 1):
        if D[i][1] >= THR and D[i + 1][1] >= THR:
            return D[i][0], D[-1][0], max(x for _, x in D), g0
    return None, D[-1][0], max(x for _, x in D), g0


def grid_ok(pts):
    """Dense schedule check: every 8 up to 2048, every 64 after, no holes before the last eval."""
    steps = [s for s, _ in pts]
    last = steps[-1]
    want = [s for s in range(0, min(last, 2048) + 1, 8)] + [s for s in range(2048 + 64, last + 1, 64)]
    return sorted(set(want) - set(steps)), sorted(set(steps) - set(want))


def ols(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    b, a = np.polyfit(x, y, 1)
    return b, a


def tint(v):
    v = np.asarray(v, float)
    n = len(v)
    m, sd = v.mean(), v.std(ddof=1)
    h = stats.t.ppf(0.975, n - 1) * sd / math.sqrt(n)
    return m, sd, (m - h, m + h)


def tobit_slope(rows):
    """rows: (ln_share, ln_T or ln_cens, censored_bool). Right-censored normal regression MLE."""
    x = np.array([r[0] for r in rows]); y = np.array([r[1] for r in rows]); c = np.array([r[2] for r in rows])
    b0, a0 = ols(x[~c], y[~c]) if (~c).sum() >= 2 else (-1.0, 0.0)
    r = y[~c] - (a0 + b0 * x[~c])
    s0 = max(r.std(), 0.05)

    def nll(p):
        a, b, ls = p
        s = math.exp(ls)
        mu = a + b * x
        ll = stats.norm.logpdf(y[~c], mu[~c], s).sum() + stats.norm.logsf(y[c], mu[c], s).sum()
        return -ll
    res = optimize.minimize(nll, [a0, b0, math.log(s0)], method="Nelder-Mead", options={"xatol": 1e-9, "fatol": 1e-12, "maxiter": 20000})
    return res.x[1], res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs")
    ap.add_argument("--prefix", default="l8_")
    ap.add_argument("--seeds", default="5,6,7")
    ap.add_argument("--c1-seeds", default="5,6")
    ap.add_argument("--ref06", default=None, help="dir with matched 0.6B reference runs (S8)")
    ap.add_argument("--gates", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    seeds = [int(s) for s in a.seeds.split(",")]
    out = {"runs": {}, "notes": []}

    files = sorted(f for p in a.prefix.split(",") for f in glob.glob(os.path.join(a.runs, p + "*.json")))
    main_runs, c1, c2 = {}, {}, {}
    PP = {}
    for f in files:
        d = load(f)
        ar = d["args"]
        pts = curve(d)
        T, last, maxD, g0 = switch_time(pts)
        holes, extra = grid_ok(pts)
        reached_B = last >= BUDGET or (d.get("steps_run") or 0) >= BUDGET
        rec = {"file": os.path.basename(f), "arm": ar.get("arm"), "k": ar.get("k"), "seed": ar.get("seed"),
               "alpha": ar.get("cue_alpha"), "share_logged": d.get("share"), "gap0": g0, "T": T, "last_eval": last,
               "maxD": maxD, "reached_B": reached_B, "T_logged": d.get("T_dense", d.get("T")),
               "grid_holes": holes[:5], "n_holes": len(holes), "resumed_from": d.get("resumed_from")}
        out["runs"][rec["file"]] = rec
        if ar.get("arm") == "shuf":
            c1[ar["seed"]] = rec
        elif ar.get("arm") == "cue" and (ar.get("cue_alpha") or 0) == 0:
            c2[ar["seed"]] = rec
            rec["max_abs_gap"] = max(abs(g) for _, g in pts)
        elif ar.get("arm") == "cue" and ar.get("k") is not None and os.path.basename(str(ar.get("u_file"))) == "cue_u.npy":
            main_runs[(ar["seed"], ar["k"])] = rec
            ev = {}
            for c in (d.get("dev_curve") or []) + (d.get("dense_curve") or []):
                if "p_off" in c:
                    po, pn = np.asarray(c["p_off"], float), np.asarray(c["p_on"], float)
                    if abs(po.mean() - c["px_off"]) > 1e-6 or abs(pn.mean() - c["px_on"]) > 1e-6:
                        out["notes"].append(f"{rec['file']} step {c['step']}: per-prompt mean != px")
                    ev[int(c["step"])] = (po, pn)
            if ev:
                st = sorted(ev)
                PP[(ar["seed"], ar["k"])] = (np.array(st), np.stack([ev[s][0] for s in st]), np.stack([ev[s][1] for s in st]))
            rec["share_logged_err"] = (d.get("share") / S_K[ar["k"]] - 1) if d.get("share") else None
        else:
            out["notes"].append(f"not main/C1/C2: {rec['file']} u_file={ar.get('u_file')}")

    # T agreement with the build's logged T
    mism = [r["file"] for r in out["runs"].values() if r["arm"] == "cue" and r["T"] != r["T_logged"] and not (r["T"] is None and r["T_logged"] is None)]
    out["T_mismatch_vs_logged"] = mism

    # per-seed slopes (finished seed = all 6 sizes, no incomplete run); exclusion |gap0| >= 0.02
    per_seed, pooled = {}, []
    for s in seeds:
        cells = [main_runs.get((s, k)) for k in range(6)]
        finished = all(c is not None for c in cells) and all(c["T"] is not None or c["reached_B"] for c in cells)
        xs, ys = [], []
        for k, c in enumerate(cells):
            if c is None:
                continue
            if abs(c["gap0"]) >= 0.02:
                c["excluded"] = True
                continue
            lx = math.log(S_K[k])
            if c["T"] is not None:
                xs.append(lx); ys.append(math.log(c["T"])); pooled.append((lx, math.log(c["T"]), False))
            else:
                pooled.append((lx, math.log(BUDGET if c["reached_B"] else c["last_eval"]), True))
        rec = {"finished": finished, "Ts": [c["T"] if c else None for c in cells], "n_uncens": len(xs)}
        if finished and len(xs) >= 4:
            b, i0 = ols(xs, ys)
            rec["slope"], rec["intercept"] = b, i0
        per_seed[s] = rec
    out["per_seed"] = per_seed
    sl = [r["slope"] for r in per_seed.values() if "slope" in r]
    if len(sl) >= 2:
        m, sd, ci = tint(sl)
        out["seed_level"] = {"n": len(sl), "mean": m, "sd": sd, "ci": ci,
                             "inside_band": BAND[0] <= ci[0] and ci[1] <= BAND[1],
                             "outside_band": ci[1] < BAND[0] or ci[0] > BAND[1]}
    n_cens = sum(1 for r in pooled if r[2])
    out["n_censored"] = n_cens
    if pooled:
        bt, res = tobit_slope(pooled)
        out["tobit_pooled_slope"] = bt
        out["ols_pooled_uncens"] = ols([r[0] for r in pooled if not r[2]], [r[1] for r in pooled if not r[2]])[0]

    # controls
    out["C1"] = {s: {"maxD": r["maxD"], "switched": r["T"] is not None, "last_eval": r["last_eval"], "reached_B": r["reached_B"]} for s, r in c1.items()}
    out["C2"] = {s: {"max_abs_gap": r["max_abs_gap"], "last_eval": r["last_eval"]} for s, r in c2.items()}
    c1_ok = all(int(s) in c1 and c1[int(s)]["reached_B"] and c1[int(s)]["T"] is None for s in a.c1_seeds.split(","))

    # gates
    gates_ok = None
    if a.gates and os.path.exists(a.gates):
        out["gates_file"] = a.gates
    # status (gates judged separately by hand from gates.json; reported here as None if not given)
    sl_ = out.get("seed_level")
    status = "UNCONFIRMED"
    if sl_ and sl_["n"] >= 3 and c1_ok:
        tob_side_ok = True
        if n_cens >= 2:
            bt = out["tobit_pooled_slope"]
            tob_side_ok = (BAND[0] <= bt <= BAND[1]) if sl_["inside_band"] else (not BAND[0] <= bt <= BAND[1])
        if sl_["inside_band"] and tob_side_ok:
            status = "CONFIRMED (if G1,G2 pass)"
        elif sl_["outside_band"] and tob_side_ok:
            status = "REJECTED (if G1,G2 pass)"
    out["c1_ok"] = c1_ok
    out["status_by_vera"] = status

    # S8 against a 0.6B reference arm
    if a.ref06:
        ref = {}
        for f in glob.glob(os.path.join(a.ref06, "*.json")):
            d = load(f); ar = d["args"]
            if ar.get("arm") != "cue" or ar.get("k") is None or not ar.get("cue_alpha"):
                continue
            T, last, _, g0 = switch_time(curve(d))
            ref.setdefault(ar["k"], []).append(T)
        mine = {k: [main_runs[(s, k)]["T"] for s in seeds if (s, k) in main_runs] for k in range(6)}
        def G(dd):
            return math.exp(np.mean([np.mean([math.log(t) for t in v]) for k, v in sorted(dd.items())]))
        if all(None not in v and v for v in mine.values()) and all(None not in v for v in ref.values()):
            g8, g06 = G(mine), G(ref)
            rows = {k: float(np.median(mine[k]) / np.median(ref[k])) for k in range(6)}
            out["S8"] = {"G8": g8, "G06": g06, "G_ratio": g8 / g06, "median_ratio": rows,
                         "holds": 0.8 <= g8 / g06 <= 1.25 and all(0.8 <= rows[k] <= 1.25 for k in (2, 3, 4, 5))}

    # NULL 1: permute the size labels within each finished seed (exact, all 720^n would be large: sample)
    rng = np.random.default_rng(1120)
    fin = [s for s in seeds if "slope" in per_seed[s]]
    if fin:
        obs = np.mean([per_seed[s]["slope"] for s in fin])
        null = []
        for _ in range(20000):
            v = []
            for s in fin:
                Ts = [t for t in per_seed[s]["Ts"] if t is not None]
                xs = [math.log(S_K[k]) for k, t in enumerate(per_seed[s]["Ts"]) if t is not None]
                v.append(ols(xs, np.log(rng.permutation(Ts)))[0])
            null.append(np.mean(v))
        null = np.array(null)
        out["null_perm_sizes"] = {"obs_mean_slope": obs, "null_mean": float(null.mean()), "null_sd": float(null.std()),
                                  "p_le_obs": float(np.mean(null <= obs)), "null_q025_q975": [float(np.quantile(null, .025)), float(np.quantile(null, .975))]}
        # leave-one-size-out sensitivity of the seed-level mean
        loo = {}
        for drop in range(6):
            v = []
            for s in fin:
                xs = [math.log(S_K[k]) for k, t in enumerate(per_seed[s]["Ts"]) if t is not None and k != drop]
                ys = [math.log(t) for k, t in enumerate(per_seed[s]["Ts"]) if t is not None and k != drop]
                v.append(ols(xs, ys)[0])
            loo[drop] = float(np.mean(v))
        out["leave_one_size_out_mean_slope"] = loo

        # S4 two halves (sizes 0-2 vs 3-5), per seed then mean
        h1 = [ols([math.log(S_K[k]) for k in range(3)], [math.log(per_seed[s]["Ts"][k]) for k in range(3)])[0] for s in fin]
        h2 = [ols([math.log(S_K[k]) for k in range(3, 6)], [math.log(per_seed[s]["Ts"][k]) for k in range(3, 6)])[0] for s in fin]
        out["S4_halves_mean"] = [float(np.mean(h1)), float(np.mean(h2))]
        # sensitivity: slope on the logged (root-found) share instead of the grid value
        vv = []
        for s in fin:
            xs = [math.log(main_runs[(s, k)]["share_logged"]) for k in range(6) if per_seed[s]["Ts"][k] is not None]
            ys = [math.log(t) for t in per_seed[s]["Ts"] if t is not None]
            vv.append(ols(xs, ys)[0])
        out["seed_level_on_logged_share"] = tint(vv) if len(vv) >= 2 else None

    # Prompt-level bootstrap (2,000 resamples of the 64 dev prompts): recompute gap0, D, T per run, slopes per seed
    if fin and all((s, k) in PP for s in fin for k in range(6)):
        brng = np.random.default_rng(0)
        nprm = PP[(fin[0], 0)][1].shape[1]
        means, n_cens_bs = [], 0
        for _ in range(2000):
            idx = brng.integers(0, nprm, nprm)
            sl_b = []
            for s in fin:
                rows = []
                for k in range(6):
                    st, Po, Pn = PP[(s, k)]
                    g = (Po[:, idx] - Pn[:, idx]).mean(1)
                    Dd = g - g[0]
                    ok = np.where((Dd[:-1] >= THR) & (Dd[1:] >= THR))[0]
                    if len(ok):
                        rows.append((math.log(S_K[k]), math.log(st[ok[0]]), False))
                    else:
                        rows.append((math.log(S_K[k]), math.log(st[-1]), True)); n_cens_bs += 1
                if any(r[2] for r in rows):
                    sl_b.append(tobit_slope(rows)[0])
                else:
                    sl_b.append(ols([r[0] for r in rows], [r[1] for r in rows])[0])
            means.append(np.mean(sl_b))
        out["prompt_bootstrap"] = {"ci": [float(np.quantile(means, .025)), float(np.quantile(means, .975))],
                                   "n_run_resamples_censored": n_cens_bs}

    js = json.dumps(out, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    if a.out:
        open(a.out, "w").write(js)
    print(js)


if __name__ == "__main__":
    main()
