"""Probe filter, target draw, pilot draw, and the clean-data identifiability gate (PREREG.md).

1. Keep probes with clean test AUROC >= 0.75; if more than 50 survive, draw 50 with seed 0.
2. Targets: 5 families (sources) drawn with seed 0 among families with a survivor, then one probe per
   family drawn with seed 0. Pilot (lambda_KL choice): one surviving non-target probe drawn with seed 0.
3. Identifiability gate: predicted drop for all (target, probe != target) pairs with delta_w/G_w(w) = -0.5,
   for P in {I, Sigma_LW, (F + lambda I)^-1}; Pearson correlations between the three vectors.
Writes data/selection.json and data/identifiability.json.
Run: uv run python projects/fisher-pullback-transfer/select_targets.py
"""
import json

import numpy as np

from common import DATA
from predict import load_clean, metrics, predicted_drop

SEED = 0


def main():
    meta = json.loads((DATA / "probes" / "probes.json").read_text())
    rng = np.random.default_rng(SEED)
    surv = [p for p in meta if p["auroc_test"] >= 0.75]
    n_surv = len(surv)
    if len(surv) > 50:
        keep = sorted(rng.choice(len(surv), 50, replace=False))
        surv = [surv[i] for i in keep]
    fams = sorted({p["source"] for p in surv})
    tfams = sorted(rng.choice(fams, 5, replace=False).tolist())
    targets = []
    for f in tfams:
        cand = [p for p in surv if p["source"] == f]
        targets.append(cand[int(rng.integers(len(cand)))]["name"])
    rest = [p["name"] for p in surv if p["name"] not in targets]
    pilot = rest[int(rng.integers(len(rest)))]
    sel = {"n_built": len(meta), "n_survive_filter": n_surv, "scored_probes": [p["name"] for p in surv],
           "dropped_by_filter": [p["name"] for p in meta if p["auroc_test"] < 0.75],
           "dropped_by_draw": [p["name"] for p in meta if p["auroc_test"] >= 0.75 and p["name"] not in [q["name"] for q in surv]],
           "families_with_survivor": fams, "target_families": tfams, "targets": targets, "pilot": pilot, "seed": SEED}
    (DATA / "selection.json").write_text(json.dumps(sel, indent=1))
    print(json.dumps({k: v for k, v in sel.items() if k != "scored_probes"}, indent=1))

    clean = load_clean()
    Ps = metrics()
    W = {p["name"]: p for p in clean["probes"]}
    vecs = {k: [] for k in Ps}
    for t in targets:
        for u in sel["scored_probes"]:
            if u == t:
                continue
            for k, P in Ps.items():
                vecs[k].append(predicted_drop(clean, W[u], W[t], P, -0.5))
    names = list(Ps)
    corr = {f"{a}~{b}": float(np.corrcoef(vecs[a], vecs[b])[0, 1]) for i, a in enumerate(names) for b in names[i + 1:]}
    sd = {k: float(np.std(v)) for k, v in vecs.items()}
    gate = {"n_pairs": len(vecs["plain"]), "ratio_used": -0.5, "pearson": corr, "sd_pred_drop": sd,
            "flag_gt_0.95": [k for k, v in corr.items() if v > 0.95]}
    (DATA / "identifiability.json").write_text(json.dumps(gate, indent=1))
    print(json.dumps(gate, indent=1))


if __name__ == "__main__":
    main()
