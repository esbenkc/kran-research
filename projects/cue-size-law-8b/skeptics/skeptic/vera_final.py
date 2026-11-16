"""Vera (skeptic) final rebuild, own code, from raw run JSONs only (dev_curve + dense_curve).

Frozen rule (PREREG.md e33506dc, FROZEN RULE block), applied mechanically:
  T = first eval t with D(t) >= 0.10 and D(next) >= 0.10 over all saved evals; D = gap - gap(step 0).
  main arm = arm 'cue', cue_alpha > 0, |gap0| < 0.02. Unswitched: censored if it reached its budget, else 'cut'.
  per-seed slope b_s = OLS ln T on ln share over that seed's switched sizes (needs >= 4), one machine per seed.
  seed-level 95% CI = mean +- t(0.975, n-1) sd/sqrt(n).
Secondaries: S1 slope difference (Welch), S2 calibrated constant (slope -0.903 from the k0 median, within 2x),
interpolated-crossing T, size-label permutation null, per-halving factors, controls.
Usage: python3 vera_final.py PROJECT_DIR [PARENT_RUNS_DIR]
"""
import json, glob, os, sys
import numpy as np
from scipy import stats

PROJ = sys.argv[1]
PARENT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(os.path.dirname(__file__), "parent_runs")
BAND = (-1.4, -0.7)
REF = -0.903  # parent dense-T slope on share/x (my rebuild -0.90265)
rng = np.random.default_rng(1116)


def load_dir(d):
    out = []
    for f in sorted(glob.glob(os.path.join(d, "*.json"))):
        try:
            j = json.load(open(f))
        except Exception as e:
            print("unreadable", f, e)
            continue
        if "dev_curve" not in j:
            continue
        out.append(row(j, os.path.basename(f)[:-5]))
    return out


def row(d, name):
    c = {e["step"]: e for e in d.get("dev_curve", [])}
    for e in d.get("dense_curve", []) or []:
        c.setdefault(e["step"], e)
    s = sorted(c)
    g0 = c[s[0]]["gap"] if s and s[0] == 0 else d.get("gap0")
    pts = [(t, c[t]["gap"] - g0) for t in s]
    T = Ti = None
    for i in range(len(pts) - 1):
        if pts[i][1] >= 0.10 and pts[i + 1][1] >= 0.10:
            T = pts[i][0]
            if i > 0:
                (s0, d0), (s1, d1) = pts[i - 1], pts[i]
                Ti = s0 + (0.10 - d0) / (d1 - d0) * (s1 - s0) if d1 > d0 else s1
            else:
                Ti = T
            break
    a = d.get("args", {})
    budget = d.get("budget", a.get("steps", 8192))
    last = pts[-1][0] if pts else 0
    st = "switched" if T is not None else ("censored" if last >= budget else "cut")
    share = d.get("share") or a.get("share")
    if share is None:  # parent: share = x / 608.865 (parent denominator)
        x = d.get("cue_size", {}).get("x")
        share = x / 608.865478515625 if x else None
    return dict(run=name, seed=a.get("seed"), arm=a.get("arm"), alpha=a.get("cue_alpha"), share=share, k=d.get("k"),
                machine=d.get("machine", "mini"), status=st, T=T, Ti=Ti, last=last, budget=budget, gap0=g0,
                px_off0=c[s[0]]["px_off"] if s else None, maxD=max(p[1] for p in pts) if pts else None,
                resumed=bool(d.get("resumed", False) or d.get("resumed_from")), T_saved=d.get("T"))


def is_main(r):
    return r["arm"] == "cue" and (r["alpha"] or 0) > 0 and r["share"] and abs(r["gap0"]) < 0.02


def per_seed(rows, key="T"):
    out = {}
    seeds = sorted({r["seed"] for r in rows if is_main(r)})
    for s in seeds:
        best = None
        for m in ("mini", "laptop"):
            rr = [r for r in rows if is_main(r) and r["seed"] == s and r["machine"] == m and r["status"] == "switched"]
            if len({r["share"] for r in rr}) >= 4:
                best = (m, rr)
                break
        if best:
            m, rr = best
            b = np.polyfit(np.log([r["share"] for r in rr]), np.log([r[key] for r in rr]), 1)[0]
            out[s] = dict(slope=float(b), machine=m, n_sizes=len(rr))
    return out


def tci(v):
    v = np.asarray(v, float)
    n = len(v)
    if n < 2:
        return float(v.mean()) if n else None, [None, None]
    h = stats.t.ppf(0.975, n - 1) * v.std(ddof=1) / np.sqrt(n)
    return float(v.mean()), [float(v.mean() - h), float(v.mean() + h)]


def welch(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    df = (va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1))
    d = a.mean() - b.mean()
    h = stats.t.ppf(0.975, df) * np.sqrt(va + vb)
    return dict(diff=float(d), ci=[float(d - h), float(d + h)], df=float(df))


def perm_null(rows, n=4000):
    seeds = per_seed(rows)
    obs = np.mean([v["slope"] for v in seeds.values()])
    null = []
    for _ in range(n):
        bs = []
        for s, v in seeds.items():
            rr = [r for r in rows if is_main(r) and r["seed"] == s and r["machine"] == v["machine"] and r["status"] == "switched"]
            xs = rng.permutation([r["share"] for r in rr])
            bs.append(np.polyfit(np.log(xs), np.log([r["T"] for r in rr]), 1)[0])
        null.append(np.mean(bs))
    null = np.array(null)
    return dict(obs=float(obs), null95=[float(np.percentile(null, 2.5)), float(np.percentile(null, 97.5))],
                p=float((np.sum(np.abs(null) >= abs(obs)) + 1) / (n + 1)))


def medians(rows, key="T"):
    out = {}
    for r in rows:
        if is_main(r) and r["status"] == "switched":
            out.setdefault(round(r["share"], 6), []).append(r[key])
    return {k: dict(median=float(np.median(v)), n=len(v), Ts=sorted(v)) for k, v in sorted(out.items(), reverse=True)}


def status(rows, gates, controls):
    ps = per_seed(rows)
    n = len(ps)
    mean, ci = tci([v["slope"] for v in ps.values()])
    c1_ok = len(controls) >= 2 and all(c["status"] == "censored" for c in controls)
    g_ok = bool(gates.get("G1", {}).get("pass")) and bool(gates.get("G2", {}).get("pass"))
    inside = n >= 3 and ci[0] is not None and ci[0] >= BAND[0] and ci[1] <= BAND[1]
    outside = n >= 3 and ci[0] is not None and (ci[1] < BAND[0] or ci[0] > BAND[1])
    n_cens = sum(1 for r in rows if is_main(r) and r["status"] == "censored")
    st = "CONFIRMED" if inside and c1_ok and g_ok else "REJECTED" if outside and c1_ok and g_ok else "UNCONFIRMED"
    st_if_gates = "CONFIRMED" if inside and c1_ok else "REJECTED" if outside and c1_ok else "UNCONFIRMED"
    return dict(n=n, per_seed=ps, mean=mean, ci=ci, inside_band=inside, outside_band=outside, c1_ok=c1_ok,
                gates_ok=g_ok, n_censored=n_cens, status=st, status_if_gates_passed=st_if_gates,
                tobit_needed=n_cens >= 2)


def s2(rows):
    med = medians(rows)
    if not med:
        return None
    shares = sorted(med, reverse=True)
    t0 = med[shares[0]]["median"]
    res = []
    for i, s in enumerate(shares):
        pred = t0 * (s / shares[0]) ** REF
        res.append(dict(share=s, median=med[s]["median"], pred=float(pred), ratio=float(med[s]["median"] / pred),
                        within2x=bool(0.5 <= med[s]["median"] / pred <= 2)))
    return dict(rows=res, holds=all(r["within2x"] for r in res))


def report(label, rows):
    print(f"\n== {label}: {len(rows)} runs")
    for r in sorted(rows, key=lambda r: (str(r["arm"]), r["seed"] or 0, -(r["share"] or 0))):
        print(f"  {r['run']:28s} arm {r['arm']} seed {r['seed']} share {r['share'] or 0:.5f} {r['machine']:6s} "
              f"{r['status']:9s} T {r['T']} Ti {None if r['Ti'] is None else round(r['Ti'],1)} saved {r['T_saved']} "
              f"last {r['last']} gap0 {r['gap0']:.4f} px0 {r['px_off0']:.4f} maxD {r['maxD']:.3f}{' RESUMED' if r['resumed'] else ''}")
        if r["T_saved"] is not None and r["T"] is not None and r["T_saved"] != r["T"]:
            print("    !! saved T differs from my T")


out = {}
r8 = load_dir(os.path.join(PROJ, "data/runs"))
r6 = load_dir(os.path.join(PROJ, "data/runs_06b"))
rp = load_dir(PARENT)
rx = load_dir(os.path.join(PROJ, "data/runs_explore"))
gates = json.load(open(os.path.join(PROJ, "data/gates.json")))
gates_n = {"G1": {"pass": gates.get("G1", {}).get("pass", gates.get("G1_pass"))}, "G2": {"pass": gates.get("G2", {}).get("pass", gates.get("G2", {}).get("b_pass"))}}
for lab, rr in (("8B main + controls", r8), ("0.6B matched", r6), ("0.6B parent", rp), ("explore", rx)):
    report(lab, rr)
ctrl8 = [r for r in r8 if r["arm"] == "shuf"]
print("\n8B controls:", [(c["run"], c["status"], c["last"], round(c["maxD"], 4)) for c in ctrl8])
for lab, rr, ctrl in (("8B", r8, ctrl8), ("0.6B matched", r6, [r for r in r6 if r["arm"] == "shuf"])):
    if not rr:
        continue
    st = status(rr, gates_n if lab == "8B" else {"G1": {"pass": True}, "G2": {"pass": True}}, ctrl)
    psi = per_seed(rr, "Ti")
    st["interp"] = dict(per_seed={s: v["slope"] for s, v in psi.items()}, mean_ci=tci([v["slope"] for v in psi.values()]))
    st["medians"] = medians(rr)
    meds = list(st["medians"].values())
    st["per_halving_factor"] = [round(meds[i + 1]["median"] / meds[i]["median"], 3) for i in range(len(meds) - 1)]
    st["S2"] = s2(rr)
    if len(per_seed(rr)) >= 1:
        st["perm"] = perm_null(rr)
    out[lab] = st
    print(f"\n== {lab} STATUS: {json.dumps(st, default=str)[:3000]}")
ps8 = [v["slope"] for v in out.get("8B", {}).get("per_seed", {}).values()]
ps6 = [v["slope"] for v in out.get("0.6B matched", {}).get("per_seed", {}).values()]
psp = [v["slope"] for v in per_seed(rp).values()]
if len(ps8) >= 2 and len(ps6) >= 2:
    out["S1_vs_matched"] = welch(ps8, ps6)
ps6_full = [v["slope"] for v in out.get("0.6B matched", {}).get("per_seed", {}).values() if v["n_sizes"] == 6]
if len(ps8) >= 2 and len(ps6_full) >= 2:
    out["S1_vs_matched_full_grid"] = dict(welch(ps8, ps6_full), n06=len(ps6_full), mean06=float(np.mean(ps6_full)))
    print("S1 (0.6B seeds with all 6 sizes only):", out["S1_vs_matched_full_grid"])
if len(ps8) >= 2 and len(psp) >= 2:
    out["S1_vs_parent"] = welch(ps8, psp)
print("\nS1:", out.get("S1_vs_matched"), "| vs parent:", out.get("S1_vs_parent"), "| parent per-seed", psp)
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "vera_final.json"), "w"), indent=1, default=str)
