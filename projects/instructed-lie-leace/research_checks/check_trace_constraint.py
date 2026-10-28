"""Does concept_erasure's default constrain_cov_trace change the LEACE edit on our data? Per readout 10..29, fit
LeaceFitter defaults (shrinkage on, constrain_cov_trace on) on line+attack clean states (items.json), compare its
eraser to the same fit with constrain_cov_trace=False (= the post's rank-one formula with the shrunk covariance).
Reports max relative difference of the edits on eval states. CPU, memmap, one readout at a time."""
import json, sys, numpy as np, torch
from concept_erasure import LeaceFitter
P = "/Users/esben/work/theory-1028-instructed-lie/kran-research/projects/instructed-lie-leace/data"
items = sys.argv[1] if len(sys.argv) > 1 else "items.json"; states = sys.argv[2] if len(sys.argv) > 2 else "clean.npy"
H = np.load(f"{P}/states/{states}", mmap_mode="r"); I = json.load(open(f"{P}/ref/{items}"))
split = np.array([r["split"] for r in I]); fit, ev = np.isin(split, ["line", "attack"]), split == "eval"
out = []
for c in ("lie", "decoy"):
    z = torch.tensor([float(r[c]) for r in I])[:, None]
    for k, R in enumerate(range(10, 30)):
        X = torch.tensor(np.asarray(H[:, k, :], np.float64))
        e = {}
        for cc in (True, False):
            f = LeaceFitter(X.shape[1], 1, dtype=torch.float64, constrain_cov_trace=cc)
            f.update(X[torch.tensor(fit)], z[torch.tensor(fit)]); e[cc] = f.eraser(X[torch.tensor(ev)]) - X[torch.tensor(ev)]
        rel = float((e[True] - e[False]).norm(dim=1).max() / e[False].norm(dim=1).max())
        out.append({"concept": c, "R": R, "rel_diff_edit": rel, "edit_norm_default": float(e[True].norm(dim=1).mean()),
                    "edit_norm_formula": float(e[False].norm(dim=1).mean())})
        print(json.dumps(out[-1]), flush=True)
json.dump(out, open("check_trace_constraint.json", "w"), indent=1)
