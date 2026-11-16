"""Frozen analysis for cue-size-law-8b (Felix, T7). Written before any 8B data.

Reads run JSONs (rl.py format of the parent cue-size-law project), re-derives T from the saved curves
(it never trusts a stored T), and applies the FROZEN RULE in PREREG.md:
  per-seed slope b_s = OLS of ln T on ln share over that seed's uncensored sizes (>= 4 sizes, |gap0| < 0.02),
  seed-level 95% interval = mean(b_s) +- t_{.975,n-1} sd(b_s)/sqrt(n),
  CONFIRMED  : n >= 3, interval inside [-1.4, -0.7], both C1 seeds no switch over the full B, gates pass,
  REJECTED   : n >= 3, interval entirely outside the band (or the threshold clause), C1 and gates pass,
  otherwise UNCONFIRMED; with >= 2 censored main-arm runs the pooled Tobit slope must agree.
Secondary reads S1-S5 are reported and never change the status.

Usage:
  python analyze_8b.py --runs data/runs --gates data/gates.json --out data/analysis.json
  python analyze_8b.py --dry-parent /tmp/parent1110/runs   # dry test on the 0.6B parent data
"""
import argparse
import glob
import json
import math
import os

import numpy as np
from scipy import optimize, stats

BAND = (-1.4, -0.7)
THRESH = 0.10
B = 8192
SHARE0 = 0.0547  # 0.6B parent: x(alpha 1/32) / ||N(h)|| rows 0-2 = 33.299 / 608.865
PARENT_SLOPE = -0.90265  # parent DENSE-T slope on share (Vera rebuild = parent summary dense.primary_x); grid -0.897 is history
SHARE_DEN_06B = 608.865478515625
GATE_MAX_ABS_GAP0 = 0.02


# ---------------------------------------------------------------- T from curves
def merged_curve(run, dense=True):
    pts = {int(p["step"]): float(p["D"]) for p in run.get("dev_curve", [])}
    if dense:
        for p in run.get("dense_curve") or []:
            pts.setdefault(int(p["step"]), float(p["D"]))
    return sorted((s, d) for s, d in pts.items() if s > 0)


def switch_time(run, dense=True):
    """First eval t with D(t) >= 0.10 and D(next eval) >= 0.10. Returns (T, censored)."""
    c = merged_curve(run, dense)
    for i in range(len(c) - 1):
        if c[i][1] >= THRESH and c[i + 1][1] >= THRESH:
            return c[i][0], False
    # unswitched: censored at B if the run reached B; a run cut early is censored at its own last eval (D1 fix)
    if ran_full(run) or not c:
        return int(run.get("budget", B)), True
    return int(c[-1][0]), True


def incomplete(run, dense=True):
    """An unswitched run that stopped before its budget (cut at the wrapup or by a self-stop): it is left out of
    n_cens and of the threshold clause, and enters the Tobit fit censored at its own last eval (D1 fix)."""
    T, c = switch_time(run, dense)
    return c and not ran_full(run)


def interp_time(run):
    """S7 (Kasper #22643): linear crossing of D = 0.10 between the eval before the dense T and the dense T."""
    c = merged_curve(run, True)
    T, cens = switch_time(run, True)
    if cens:
        return T, True
    i = [s for s, _ in c].index(T)
    if i == 0:
        return float(T), False
    (t0, d0), (t1, d1) = c[i - 1], c[i]
    return (t0 + (THRESH - d0) / (d1 - d0) * (t1 - t0)) if d1 > d0 else float(T), False


def ran_full(run):
    return int(run.get("steps_run", 0)) >= int(run.get("budget", B))


def is_main(r):
    """Main-arm cue run: arm 'cue', a size index, a non-zero cue (C2 has cue_alpha 0 and is never in the fits)."""
    return (r["arm"] == "cue" and r["k"] is not None and (r["alpha"] is None or r["alpha"] > 0)
            and (r["share"] or 0) > 0)


# ---------------------------------------------------------------- run table
def load_runs(paths, share_den=None):
    rows = []
    for f in paths:
        r = json.load(open(f))
        a = r.get("args", {})
        cs = r.get("cue_size") or {}
        x = cs.get("x")
        share = r.get("share", cs.get("share"))
        if share is None and x is not None and share_den:
            share = x / share_den
        rows.append(dict(
            name=r.get("name", os.path.basename(f)[:-5]),
            model=r.get("model", a.get("model", "?")),
            arm=a.get("arm", r.get("arm", "cue")),
            seed=int(a.get("seed", r.get("seed", -1))),
            k=r.get("k", a.get("k")),
            alpha=a.get("cue_alpha"),
            x=x, share=share, gap0=r.get("gap0"),
            machine=r.get("machine", a.get("machine", "mini")),
            run=r,
        ))
    return rows


def assign_k(rows, share0=SHARE0):
    for r in rows:
        if r["k"] is None and r["share"]:
            k = math.log2(share0 / r["share"])
            r["k"] = int(round(k)) if abs(k - round(k)) < 0.15 else None
    return rows


def provenance(rows):
    """Frozen copy rule: a seed's slope uses cells from ONE machine. A seed is mini-primary if the mini has >= 4
    uncensored sizes for it, else laptop. Laptop seeds join the same interval only if the laptop copies of >= 2 mini
    cells reproduce T within 8 updates or 10%; else they form a separate series. Controls: mini copy if present."""
    for r in rows:
        r["T_dense"], r["cens_dense"] = switch_time(r["run"], True)
    cue = [r for r in rows if r["arm"] == "cue"]
    mini = {(r["seed"], r["k"]): r for r in cue if r["machine"] == "mini"}
    lap = {(r["seed"], r["k"]): r for r in cue if r["machine"] != "mini"}
    pairs = [(mini[key], lap[key]) for key in mini if key in lap and not mini[key]["cens_dense"]]
    agree = [abs(m["T_dense"] - l["T_dense"]) <= max(8, 0.1 * m["T_dense"]) for m, l in pairs]
    laptop_ok = sum(agree) >= 2
    keep, separate, primary = [], [], {}
    for s in sorted({r["seed"] for r in cue}):
        n_mini = len({r["k"] for r in cue if r["seed"] == s and r["machine"] == "mini" and not r["cens_dense"]})
        primary[s] = "mini" if n_mini >= 4 else "laptop"
        chosen = [r for r in cue if r["seed"] == s and (r["machine"] == "mini") == (primary[s] == "mini")]
        (keep if primary[s] == "mini" or laptop_ok else separate).extend(chosen)
    ctrl = [r for r in rows if r["arm"] != "cue"]
    ctrl_keep = {}
    for r in ctrl:
        key = (r["arm"], r["seed"])
        if key not in ctrl_keep or r["machine"] == "mini":
            ctrl_keep[key] = r
    info = dict(primary=primary, n_pairs=len(pairs), n_agree=sum(agree), laptop_ok=laptop_ok,
                pairs=[dict(seed=m["seed"], k=m["k"], T_mini=m["T_dense"], T_laptop=l["T_dense"]) for m, l in pairs],
                separate=[r["name"] for r in separate])
    return keep + list(ctrl_keep.values()), separate, info


# ---------------------------------------------------------------- statistics
def ols(xs, ys):
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    A = np.vstack([xs, np.ones_like(xs)]).T
    (b, a), *_ = np.linalg.lstsq(A, ys, rcond=None)
    n = len(xs)
    if n > 2:
        res = ys - (a + b * xs)
        se = math.sqrt(res @ res / (n - 2) / ((xs - xs.mean()) @ (xs - xs.mean())))
        h = stats.t.ppf(0.975, n - 2) * se
        return dict(slope=b, intercept=a, se=se, ci=[b - h, b + h], n=n)
    return dict(slope=b, intercept=a, se=None, ci=None, n=n)


def seed_level(slopes):
    s = np.asarray(slopes, float)
    n = len(s)
    if n < 2:
        return dict(n=n, mean=float(s.mean()) if n else None, ci=None)
    m, sd = s.mean(), s.std(ddof=1)
    h = stats.t.ppf(0.975, n - 1) * sd / math.sqrt(n)
    return dict(n=n, mean=float(m), sd=float(sd), ci=[float(m - h), float(m + h)],
                method=f"t-interval over {n} per-seed OLS slopes (df {n - 1})")


def tobit(xs, ys, cens):
    xs, ys, cens = map(lambda v: np.asarray(v, float), (xs, ys, cens))

    def nll(p):
        b, a, ls = p
        s = math.exp(ls)
        mu = a + b * xs
        ll_u = stats.norm.logpdf(ys, mu, s)
        ll_c = stats.norm.logsf(ys, mu, s)
        return -(np.where(cens > 0, ll_c, ll_u)).sum()

    o0 = ols(xs[cens == 0], ys[cens == 0])
    r = optimize.minimize(nll, [o0["slope"], o0["intercept"], math.log(0.2)], method="Nelder-Mead",
                          options=dict(maxiter=20000, xatol=1e-8, fatol=1e-10))
    H = np.zeros((3, 3))
    eps = 1e-4
    for i in range(3):
        for j in range(3):
            def f(di, dj):
                p = np.array(r.x, float)
                p[i] += di
                p[j] += dj
                return nll(p)
            H[i, j] = (f(eps, eps) - f(eps, -eps) - f(-eps, eps) + f(-eps, -eps)) / (4 * eps * eps)
    try:
        se = math.sqrt(np.linalg.inv(H)[0, 0])
    except Exception:
        se = float("nan")
    b = r.x[0]
    return dict(slope=float(b), se=se, ci=[b - 1.96 * se, b + 1.96 * se], n=len(xs), n_cens=int(cens.sum()))


def side(ci):
    if ci is None:
        return "none"
    lo, hi = ci
    if lo >= BAND[0] and hi <= BAND[1]:
        return "inside"
    if hi < BAND[0] or lo > BAND[1]:
        return "outside"
    return "straddles"


# ---------------------------------------------------------------- per-model analysis
def per_seed(rows, dense=True):
    cue = [r for r in rows if is_main(r)]
    by = {}
    for r in cue:
        T, c = switch_time(r["run"], dense)
        r[("T_dense" if dense else "T_grid")] = T
        r[("cens_dense" if dense else "cens_grid")] = c
        by.setdefault(r["seed"], []).append(r)
    out = {}
    for s, rr in sorted(by.items()):
        use = [r for r in rr if not r[("cens_dense" if dense else "cens_grid")]
               and (r["gap0"] is None or abs(r["gap0"]) < GATE_MAX_ABS_GAP0)]
        ks = sorted({r["k"] for r in use})
        entry = dict(sizes=ks, n_runs=len(rr), n_cens=sum(r[("cens_dense" if dense else "cens_grid")] for r in rr))
        if len(ks) >= 4:
            f = ols([math.log(r["share"]) for r in use], [math.log(r[("T_dense" if dense else "T_grid")]) for r in use])
            entry.update(slope=f["slope"], intercept=f["intercept"])
        out[s] = entry
    return cue, out


def status(model_rows, gates):
    cue, seeds = per_seed(model_rows, dense=True)
    slopes = [v["slope"] for v in seeds.values() if "slope" in v]
    sl = seed_level(slopes)
    ctrl = [r for r in model_rows if r["arm"] == "shuf"]
    c1 = []
    for r in ctrl:
        T, c = switch_time(r["run"], True)
        c1.append(dict(name=r["name"], seed=r["seed"], switched=not c, full=ran_full(r["run"]), T=None if c else T))
    c1_ok = len([c for c in c1 if c["full"] and not c["switched"]]) >= 2 and not any(c["switched"] for c in c1)
    gates_ok = bool(gates) and all(bool(gates.get(g, {}).get("pass")) for g in ("G1", "G2"))

    # pooled Tobit over all main-arm runs (gap0-valid)
    valid = [r for r in cue if r["gap0"] is None or abs(r["gap0"]) < GATE_MAX_ABS_GAP0]
    n_cens = sum(r["cens_dense"] and not incomplete(r["run"]) for r in valid)
    n_incomplete = sum(incomplete(r["run"]) for r in valid)
    tb = None
    if len({r["k"] for r in valid}) >= 3:
        tb = tobit([math.log(r["share"]) for r in valid], [math.log(r["T_dense"]) for r in valid],
                   [r["cens_dense"] for r in valid])

    # threshold clause, with the lateness guard (Kasper #22611): each censored seed's own fit over the larger sizes
    # must predict T at the smallest size with B >= 3 x that prediction
    ks = sorted({r["k"] for r in valid})
    thresh = False
    thresh_detail = []
    if len(ks) >= 2:
        kmin_share = max(ks)  # largest k = smallest share
        cens_small = [r for r in valid if r["k"] == kmin_share and r["cens_dense"] and not incomplete(r["run"])]
        sw_next = sum(not r["cens_dense"] for r in valid if r["k"] == kmin_share - 1)
        late_ok = []
        for r in cens_small:
            own = [q for q in valid if q["seed"] == r["seed"] and q["k"] < kmin_share and not q["cens_dense"]]
            if len({q["k"] for q in own}) >= 3:
                f = ols([math.log(q["share"]) for q in own], [math.log(q["T_dense"]) for q in own])
                pred = math.exp(f["intercept"] + f["slope"] * math.log(r["share"]))
                late_ok.append(B >= 3 * pred)
                thresh_detail.append(dict(seed=r["seed"], pred_T=pred, guard=B >= 3 * pred))
            else:
                late_ok.append(False)
        thresh = len(cens_small) >= 3 and sw_next >= 3 and all(late_ok)

    sd = side(sl.get("ci"))
    reasons = []
    st = "UNCONFIRMED"
    base_ok = sl["n"] >= 3 and c1_ok and gates_ok
    if not sl["n"] >= 3:
        reasons.append(f"only {sl['n']} seeds with a slope (need 3)")
    if not c1_ok:
        reasons.append("C1 controls missing, not full length, or switched")
    if not gates_ok:
        reasons.append("gates G1/G2 missing or failed")
    if base_ok and (sd == "inside"):
        st = "CONFIRMED"
    elif base_ok and (sd == "outside" or thresh):
        st = "REJECTED"
    elif base_ok:
        reasons.append(f"seed-level interval {sl.get('ci')} straddles the band")
    if st != "UNCONFIRMED" and n_cens >= 2:
        if tb is None or side(tb["ci"]) != ("inside" if st == "CONFIRMED" else "outside"):
            reasons.append(f"{n_cens} censored runs and the Tobit slope is not on the same side")
            st = "UNCONFIRMED"
    return dict(status=st, reasons=reasons, seed_level=sl, per_seed=seeds, controls=c1, gates_ok=gates_ok,
                tobit=tb, n_cens=n_cens, n_incomplete=n_incomplete, threshold=thresh, threshold_detail=thresh_detail, side=sd,
                runs=[dict(name=r["name"], seed=r["seed"], k=r["k"], share=r["share"], T_dense=r["T_dense"],
                           cens=r["cens_dense"], gap0=r["gap0"]) for r in cue])


def secondary(res8, rows8, res06, rows06, parent_rows):
    out = {}
    # S1 slope difference (Welch over per-seed slopes)
    a = [v["slope"] for v in res8["per_seed"].values() if "slope" in v]
    b = [v["slope"] for v in res06["per_seed"].values() if "slope" in v] if res06 else []
    if len(a) >= 2 and len(b) >= 2:
        va, vb = np.var(a, ddof=1) / len(a), np.var(b, ddof=1) / len(b)
        d = np.mean(a) - np.mean(b)
        se = math.sqrt(va + vb)
        dfw = (va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1)) if va + vb > 0 else 1
        h = stats.t.ppf(0.975, dfw) * se
        out["S1_slope_diff"] = dict(diff=d, ci=[d - h, d + h], df=dfw, n8=len(a), n06=len(b))
    # S2 calibrated constant: median T at k=0, predict k=1..5 with the parent slope
    runs = res8["runs"]
    med = {}
    for k in sorted({r["k"] for r in runs}):
        Ts = [r["T_dense"] for r in runs if r["k"] == k]
        cens = [r["cens"] for r in runs if r["k"] == k]
        med[k] = dict(median=float(np.median(Ts)), n=len(Ts), median_censored=sorted(cens)[len(cens) // 2],
                      share=float(np.median([r["share"] for r in runs if r["k"] == k])))
    if 0 in med:
        rows = []
        for k, m in med.items():
            if k == 0:
                continue
            pred = med[0]["median"] * (m["share"] / med[0]["share"]) ** PARENT_SLOPE
            ratio = m["median"] / pred
            ok = None if (m["median_censored"] and 2 * pred >= B) else (0.5 <= ratio <= 2 and not m["median_censored"])
            rows.append(dict(k=k, pred=pred, median=m["median"], ratio=ratio, within_2x=ok))
        sc = [r["within_2x"] for r in rows]
        out["S2_calibrated"] = dict(rows=rows, holds=(None if None in sc else all(sc)))
    # S3 curvature guard: local slopes between consecutive sizes, each at its geometric-mean T,
    # next to the parent's local slope at the same T (interpolated in ln T)
    def local(medians):
        ks = sorted(medians)
        L = []
        for k1, k2 in zip(ks, ks[1:]):
            m1, m2 = medians[k1], medians[k2]
            if m1.get("median_censored") or m2.get("median_censored"):
                continue
            L.append(dict(k=(k1, k2), T_gm=math.sqrt(m1["median"] * m2["median"]),
                          slope=(math.log(m2["median"]) - math.log(m1["median"])) /
                                (math.log(m2["share"]) - math.log(m1["share"]))))
        return L
    loc8 = local(med)
    pmed = {}
    for r in parent_rows:
        if r["k"] is None:
            continue
        pmed.setdefault(r["k"], []).append(r)
    pm = {k: dict(median=float(np.median([r["T_dense_parent"] for r in v])), median_censored=False,
                  share=float(np.median([r["share"] for r in v]))) for k, v in pmed.items()}
    locp = local(pm)
    for L in loc8:
        lt = [math.log(p["T_gm"]) for p in locp]
        if lt:
            L["parent_slope_at_T"] = float(np.interp(math.log(L["T_gm"]), lt, [p["slope"] for p in locp]))
    win = [r for r in runs if not r["cens"] and 24 <= r["T_dense"] <= 704]
    # frozen number (Tove #22606): quadratic ln T in ln share over the 18 parent all-row runs; local slope where the
    # fitted parent T equals the geometric mean of the 8B dense T
    quad = None
    if len(parent_rows) >= 5:
        Lp = np.array([math.log(r["share"]) for r in parent_rows])
        yp = np.array([math.log(r["T_dense_parent"]) for r in parent_rows])
        c2, c1, c0 = np.polyfit(Lp, yp, 2)
        unc = [r for r in runs if not r["cens"]]
        quad = dict(c2=c2, c1=c1, c0=c0, n_parent=len(parent_rows))
        if unc:
            lt = float(np.mean([math.log(r["T_dense"]) for r in unc]))
            disc = c1 * c1 - 4 * c2 * (c0 - lt)
            if disc >= 0 and c2 != 0:
                roots = [(-c1 - math.sqrt(disc)) / (2 * c2), (-c1 + math.sqrt(disc)) / (2 * c2)]
                L = min(r for r in roots if 2 * c2 * r + c1 < 0) if any(2 * c2 * r + c1 < 0 for r in roots) else None
                if L is not None:
                    s_m = 2 * c2 * L + c1
                    ci = res8["seed_level"].get("ci")
                    moot = all(24 <= r["T_dense"] <= 704 for r in unc) and not any(r["cens"] for r in runs)
                    quad.update(T_gm_8b=math.exp(lt), share_at_T=math.exp(L), matched_slope=s_m, moot=moot,
                                verdict=("moot: every 8B T inside [24, 704]" if moot else
                                         None if ci is None else
                                         "same curve read at longer T" if ci[0] <= s_m <= ci[1] else
                                         "the exponent moves with scale"))
    out["S3_curvature"] = dict(quadratic=quad, local_8b=loc8, local_parent=locp,
                               window_fit=(ols([math.log(r["share"]) for r in win],
                                               [math.log(r["T_dense"]) for r in win])
                                           if len({r["k"] for r in win}) >= 3 else None))
    # S4 two halves
    def half(ks):
        rr = [r for r in runs if r["k"] in ks and not r["cens"]]
        return ols([math.log(r["share"]) for r in rr], [math.log(r["T_dense"]) for r in rr])["slope"] \
            if len({r["k"] for r in rr}) >= 2 else None
    h1, h2 = half({0, 1, 2}), half({3, 4, 5})
    out["S4_two_halves"] = dict(top=h1, bottom=h2,
                                not_one_power_law=(bool(abs(h1 - h2) > 0.5) if None not in (h1, h2) else None))
    # S5 frozen-grid T
    _, seeds_grid = per_seed(rows8, dense=False)
    out["S5_grid"] = dict(per_seed=seeds_grid,
                          seed_level=seed_level([v["slope"] for v in seeds_grid.values() if "slope" in v]))
    # S7 interpolated-crossing T, per-seed slopes and seed-level interval
    by = {}
    for r in rows8:
        if not is_main(r):
            continue
        Ti, ci = interp_time(r["run"])
        if not ci and (r["gap0"] is None or abs(r["gap0"]) < GATE_MAX_ABS_GAP0):
            by.setdefault(r["seed"], []).append((math.log(r["share"]), math.log(Ti)))
    sl7 = [ols([p[0] for p in v], [p[1] for p in v])["slope"] for v in by.values() if len(v) >= 4]
    out["S7_interp"] = dict(per_seed=sl7, seed_level=seed_level(sl7))
    out["medians"] = med
    return out


def parent_rows(parent_dir):
    """0.6B parent runs on the SAME T definition as the 8B primary (dense T; Vera #22592):
    only runs that saved a dense curve (rd_ re-runs of seed 0, seed 1-2 sweep runs, k=6,7 extension)."""
    rows = assign_k(load_runs(sorted(glob.glob(os.path.join(parent_dir, "*.json"))), SHARE_DEN_06B))
    keep = []
    for r in rows:
        n = r["name"]
        if r["alpha"]:  # parent grid is alpha = 2^-5 / 2^k; cast rounding bends x at k >= 6, so k comes from alpha
            r["k"] = int(round(math.log2(0.03125 / r["alpha"])))
        if n.startswith("nocastk8"):  # x matched to cast k = 8
            r["k"] = 8
        if n.startswith(("rd_", "sw_", "ext_a0.000", "nocast")) and r["arm"] == "cue" and r["run"].get("dense_curve"):
            T, c = switch_time(r["run"], dense=True)
            r["T_dense_parent"] = T
            if not c and r["share"]:
                keep.append(r)
    return keep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="data/runs")
    ap.add_argument("--runs06", default="data/runs_06b")
    ap.add_argument("--parent", default="data/parent_0p6b/runs")
    ap.add_argument("--gates", default="data/gates.json")
    ap.add_argument("--out", default="data/analysis.json")
    ap.add_argument("--dry-parent", default=None)
    a = ap.parse_args()

    if a.dry_parent:
        rows = assign_k(load_runs(sorted(glob.glob(os.path.join(a.dry_parent, "*.json"))), SHARE_DEN_06B))
        rows = [r for r in rows if r["name"].startswith(("rd_", "sw_a", "p0_", "c1_")) and
                (r["arm"] == "shuf" or (r["run"].get("dense_curve")))]
        # parent: C1 has dense curves; main arm = runs with dense curves (rd_ re-runs + seed 1/2)
        res = status(rows, {"G1": {"pass": True}, "G2": {"pass": True}})
        res["S5"] = secondary(res, rows, None, None, parent_rows(a.dry_parent))["S5_grid"]
        print(json.dumps({k: res[k] for k in ("status", "reasons", "seed_level", "per_seed", "controls", "side")},
                         indent=1, default=float))
        print("S5 grid seed-level:", json.dumps(res["S5"]["seed_level"], default=float))
        return

    gates = json.load(open(a.gates)) if os.path.exists(a.gates) else {}
    share_den = gates.get("share_den")
    rows8 = assign_k(load_runs(sorted(glob.glob(os.path.join(a.runs, "*.json"))), share_den), SHARE0)
    rows8, separate, prov = provenance(rows8)
    res8 = status(rows8, gates)
    res8["provenance"] = prov
    res06 = rows06 = None
    if os.path.isdir(a.runs06) and glob.glob(os.path.join(a.runs06, "*.json")):
        rows06 = assign_k(load_runs(sorted(glob.glob(os.path.join(a.runs06, "*.json"))), gates.get("share_den_06b")),
                          SHARE0)
        res06 = status(rows06, {"G1": {"pass": True}, "G2": {"pass": True}})
    par = parent_rows(a.parent) if os.path.isdir(a.parent) else []
    sec = secondary(res8, rows8, res06, rows06, par)
    out = dict(status_8b=res8, matched_06b=res06, secondary=sec, laptop_separate_series=separate)
    json.dump(out, open(a.out, "w"), indent=1, default=float)
    sl = res8["seed_level"]
    print(f"STATUS {res8['status']}  seed-level: {sl.get('mean')} {sl.get('ci')} ({sl.get('method')})  "
          f"reasons: {res8['reasons']}")


if __name__ == "__main__":
    main()
