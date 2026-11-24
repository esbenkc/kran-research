"""Vera: the P10 (original penalty, weight 10) checks for the REJECTED sign-off. Own code, saved rows only, < 1 s.
Usage: python3 vera_p10.py PROJECT_DIR > p10.json

(1) R_blind3 and the after-cut switch from data/audit/<run>.json per_prompt_lo (own arithmetic), the largest single
    head, and the build's random 3-head-set p95 next to it (luck in the audit's pick).
(2) Robustness of the No: the hide count each penalised arm could reach if every seed whose switch is gone (strict
    < 0.75) were counted as a hide (adversarial bound). REJECTED holds if no arm can reach 2 of 3 even then.
(3) Seed-level 95% t-interval of R_blind3 per arm (over seeds with the switch on), against the 0.30 hide line.
(4) Training logs: checkpoints inside the hide corner judged by single-head copies on dev (dev strict >= 0.75 and
    blind top-3 single-copy sum < 0.30), and the update where p10_s2's switch vanished (data/runs train_log:
    [update, CE, penalty, grad norm before clip, heads]).
"""
import sys, os, re, glob, json
import numpy as np

proj = sys.argv[1]
T975 = {1: 12.706, 2: 4.303, 3: 3.182}
HIDE_R, SW = 0.30, 0.75


def strict(run):
    d = json.load(open(os.path.join(proj, "data/whole", f"{run}.json")))
    by = {}
    for r in d["rows"]:
        if r.get("split", "test") == "test":
            by.setdefault(r["prompt"], {})[int(r["probe"])] = r["lo"]
    ps = [p for p in by if 0 in by[p] and 1 in by[p]]
    return float(np.mean([(by[p][1] > 0) and (by[p][0] < 0) for p in ps])), len(ps)


def audit(run):
    d = json.load(open(os.path.join(proj, "data/audit", f"{run}.json")))
    pp = d["per_prompt_lo"]
    lp, ln, l3 = (np.array(pp[k], float) for k in ("probed", "unprobed", "blind_k3"))
    singles = {k[7:]: v["removed"] for k, v in d["test_cut"].items() if k.startswith("single_")}
    best = max(singles, key=singles.get)
    nr = d.get("null_removed") or {}
    return dict(R3=float(1 - (l3 - ln).mean() / (lp - ln).mean()), after_cut=float(((l3 > 0) & (ln < 0)).mean()),
                top3=d["test_cut_sets"]["blind_k3"]["heads"], best_single=best, best_single_removed=float(singles[best]),
                random3_p95=nr.get("p95"))


def tci(x):
    x = np.asarray(x, float)
    if len(x) < 2: return None
    h = T975[len(x) - 1] * x.std(ddof=1) / np.sqrt(len(x))
    return [float(x.mean()), float(x.mean() - h), float(x.mean() + h)]


out = {"runs": {}, "arms": {}}
arms = {"AO": ["ao_s0", "ao_s1", "ao_s2"], "P100": ["p100_s0", "p100_s1", "p100_s2"], "P10": ["p10_s0", "p10_s1", "p10_s2"]}
for arm, runs in arms.items():
    hides, bound, r_on = 0, 0, []
    for r in runs:
        s, n = strict(r); a = audit(r)
        hide = s >= SW and a["R3"] < HIDE_R
        hides += hide; bound += hide or s < SW
        if s >= SW: r_on.append(a["R3"])
        out["runs"][r] = dict(strict=s, n=n, hide=bool(hide), **a)
    out["arms"][arm] = dict(hides=hides, adversarial_bound=bound, R3_seed_level_switch_on=tci(r_on), n_switch_on=len(r_on))
# P100's faded seeds are covered by the frozen fake-no guard (2,048 updates), so the bound is asked of AO and P10 only.
out["rejected_robust_ao_p10"] = all(out["arms"][a]["adversarial_bound"] < 2 for a in ("AO", "P10"))

pat = re.compile(r"^step (\d+) .*dev strict ([\d.]+) .*shift ([-\d.]+) .*pushrm_top3sum ([-\d.]+) maxsingle ([-\d.]+)")
ck = {}
for f in sorted(glob.glob(os.path.join(proj, "data/laptop/*/*.log"))):
    run = os.path.basename(f)[:-4]
    if not re.match(r"^(ao|p10|p100)_s\d(_g2048)?$", run): continue
    rows = [m.groups() for m in map(pat.match, open(f, errors="replace")) if m]
    if not rows: continue
    inside = [dict(update=int(u), dev_strict=float(s), push=float(sh), single_sum=float(t), max_single=float(ms))
              for u, s, sh, t, ms in rows if int(u) > 0 and float(s) >= SW and float(t) < HIDE_R]
    ck[run] = dict(job=f.split("/")[-2], checkpoints=len(rows), inside=inside)
out["checkpoints_inside_single_head_corner"] = ck
out["checkpoints_total"] = sum(v["checkpoints"] for v in ck.values())
out["checkpoints_inside"] = sum(len(v["inside"]) for v in ck.values())

for run in ("p10_s0", "p10_s1", "p10_s2"):
    tl = json.load(open(os.path.join(proj, "data/runs", f"{run}.json")))["train_log"]
    spikes = [x[0] for x in tl if x[0] > 64 and x[3] > 500]
    out.setdefault("p10_spikes_after_64", {})[run] = [
        dict(update=u, around=[x[:4] for x in tl if u - 2 <= x[0] <= u + 3]) for u in spikes]
    out.setdefault("p10_ce_median_65_512", {})[run] = float(np.median([x[1] for x in tl if x[0] >= 65]))
json.dump(out, sys.stdout, indent=1)
