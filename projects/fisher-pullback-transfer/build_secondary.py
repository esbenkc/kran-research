"""Secondary same-concept probe stack (DEVIATIONS.md 2026-10-17 15:09, declared before any attack).

Per target i (order of selection.json 'targets'), rng seed i, 10 probes on the TARGET's own labels and train split:
j = 0..4 logistic (L2, C = 1.0, raw activations), j = 5..9 mass-mean (class-mean difference), each fit inside a
random orthonormal subspace Q (512 x k, QR of a Gaussian matrix), k = round(exp(U[log 2, log 512])), mapped back
to R^512 (w = Q a). Kept if clean test AUROC >= 0.75.
Then the clean identifiability check for this set: gap-form map at kappa_w = 0.5 (ratio -0.5), pairwise
Pearson correlation of the plain / whitened / Fisher predicted drops.
Writes data/probes/secondary.json, data/probes/secondary_weights.npz, data/identifiability_secondary.json.
Run: uv run python projects/fisher-pullback-transfer/build_secondary.py
"""
import json

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from common import DATA
from predict import load_clean, metrics, predicted_drop, predicted_drop_gauss


def main():
    sel = json.loads((DATA / "selection.json").read_text())
    meta = json.loads((DATA / "probes" / "probes.json").read_text())
    out, W = [], []
    for i, tname in enumerate(sel["targets"]):
        t = next(p for p in meta if p["name"] == tname)
        z = np.load(DATA / "clean" / f"acts_{t['source']}.npz")
        X = z["X"].astype(np.float64)
        y = (z["label"] == t["label"]).astype(int)
        tr, te = z["split"] == 0, z["split"] == 1
        rng = np.random.default_rng(i)
        for j in range(10):
            kind = "logistic" if j < 5 else "massmean"
            k = int(round(np.exp(rng.uniform(np.log(2), np.log(512)))))
            Q = np.linalg.qr(rng.standard_normal((512, k)))[0]
            Z = X @ Q
            if kind == "logistic":
                a = LogisticRegression(C=1.0, max_iter=20000).fit(Z[tr], y[tr]).coef_[0]
            else:
                a = Z[tr & (y == 1)].mean(0) - Z[tr & (y == 0)].mean(0)
            w = Q @ a
            auc = float(roc_auc_score(y[te], X[te] @ w))
            rec = {"name": f"{tname}#{kind}{j}_k{k}", "target": tname, "source": t["source"], "label": t["label"],
                   "kind": kind, "k": k, "seed": i, "auroc_test": auc, "kept": auc >= 0.75,
                   "cos_with_target": float(w @ np.load(DATA / "probes" / "probe_weights.npz")["W"][t["idx"]]
                                             / np.linalg.norm(w)
                                             / np.linalg.norm(np.load(DATA / "probes" / "probe_weights.npz")["W"][t["idx"]]))}
            rec["idx"] = len(out)
            out.append(rec)
            W.append(w)
            print(f"{rec['name']:<60s} AUROC {auc:.3f} cos(w) {rec['cos_with_target']:+.3f}", flush=True)
    (DATA / "probes" / "secondary.json").write_text(json.dumps(out, indent=1))
    np.savez_compressed(DATA / "probes" / "secondary_weights.npz", W=np.array(W))

    clean = load_clean()
    Ps = metrics()
    P0 = {p["name"]: p for p in clean["probes"]}
    vecs = {k: [] for k in Ps}
    vecs_emp = {k: [] for k in Ps}
    for r in out:
        if not r["kept"]:
            continue
        u = {"w": W[r["idx"]], "source": r["source"], "label": r["label"]}
        for k, P in Ps.items():
            vecs[k].append(predicted_drop_gauss(clean, u, P0[r["target"]], P, -0.5))
            vecs_emp[k].append(predicted_drop(clean, u, P0[r["target"]], P, -0.5))
    names = list(Ps)
    corr = lambda V: {f"{a}~{b}": float(np.corrcoef(V[a], V[b])[0, 1]) for i, a in enumerate(names) for b in names[i + 1:]}
    gate = {"n_pairs": len(vecs["plain"]), "n_built": len(out), "ratio_used": -0.5,
            "gap_map": {"pearson": corr(vecs), "sd_pred_drop": {k: float(np.std(v)) for k, v in vecs.items()},
                        "mean_pred_drop": {k: float(np.mean(v)) for k, v in vecs.items()}},
            "empirical_map": {"pearson": corr(vecs_emp), "sd_pred_drop": {k: float(np.std(v)) for k, v in vecs_emp.items()}}}
    (DATA / "identifiability_secondary.json").write_text(json.dumps(gate, indent=1))
    print(json.dumps(gate, indent=1))


if __name__ == "__main__":
    main()
