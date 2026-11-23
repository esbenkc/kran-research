"""Vera null tests from saved rows only (own code). Usage: python3 vera_null.py PROJECT_DIR
(1) seed-level permutation: AO (3 seeds) vs L0 (2 seeds), all 10 label assignments, for R_blind3 and for the switch
    after the top-3 cut. Exact one-sided p = share of assignments with an 'AO' mean at least as far below L0.
(2) per run: joint top-3 removal vs the sum of the same 3 heads' single removals on test (test_cut single_*),
    and an additive proxy null: sum of 3 random single-head removals (20,000 draws), its p95.
(3) the build's own random 3-head-set null (null_removed p95), reported next to R_blind3.
"""
import sys, json, os, itertools
import numpy as np

proj = sys.argv[1]
import glob
runs = sorted(os.path.basename(f)[:-5] for f in glob.glob(os.path.join(proj, "data/audit", "*.json")))
R3, CUT, out = {}, {}, {"per_run": {}}
rng = np.random.default_rng(0)
for r in runs:
    d = json.load(open(os.path.join(proj, "data/audit", f"{r}.json")))
    pp = d["per_prompt_lo"]
    lp, ln, l3 = (np.array(pp[k], float) for k in ("probed", "unprobed", "blind_k3"))
    R3[r] = 1 - (l3 - ln).mean() / (lp - ln).mean()
    CUT[r] = float(((l3 > 0) & (ln < 0)).mean())
    singles = {k[7:]: v["removed"] for k, v in d["test_cut"].items() if k.startswith("single_")}
    top3 = d["test_cut_sets"]["blind_k3"]["heads"]
    vals = np.array(list(singles.values()))
    draws = np.array([vals[rng.choice(len(vals), 3, replace=False)].sum() for _ in range(20000)])
    out["per_run"][r] = dict(R_blind3=float(R3[r]), switch_after_cut3=CUT[r], top3=top3,
                             sum_single_top3=float(sum(singles[h] for h in top3)),
                             max_single=float(vals.max()),
                             additive_null_p95=float(np.percentile(draws, 95)),
                             build_random3_p95=d["null_removed"]["p95"], build_random3_max=d["null_removed"]["max"])

def perm(metric):
    metric = {r: v for r, v in metric.items() if r.split("_")[0] in ("ao", "l0") and "g" not in r.split("_")[-1]}
    names = list(metric); obs = np.mean([metric[r] for r in names if r.startswith("l0")]) - \
        np.mean([metric[r] for r in names if r.startswith("ao")])
    diffs = []
    for ao in itertools.combinations(names, 3):
        rest = [r for r in names if r not in ao]
        diffs.append(np.mean([metric[r] for r in rest]) - np.mean([metric[r] for r in ao]))
    return dict(observed_L0_minus_AO=float(obs), p_one_sided=float(np.mean(np.array(diffs) >= obs - 1e-12)),
                n_assignments=len(diffs))

out["perm_R_blind3"] = perm(R3)
out["perm_switch_after_cut3"] = perm({r: -v for r, v in CUT.items()})  # AO higher after-cut switch = L0 minus AO of -cut
json.dump(out, sys.stdout, indent=1)
