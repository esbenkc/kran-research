"""Reproduction gate (PREREG s1 + DEVIATIONS D1): the frozen 1019 probe on the new mean-pooled hidden_states[14]
must give (a) eval AUROC within 0.005 of the 1019 log and (b) max |score diff| <= 1e-3 on all 2992 statements.
-> data/gate.json.   Run: uv run python gate.py
"""
import json

import numpy as np
from sklearn.metrics import roc_auc_score

from common import ATTACK_LAYER, DATA, REF, STATES, Acts, load_probe, load_split


def main():
    R = load_split()
    y = np.array([1 - r["true"] for r in R])
    ev = np.array([r["split"] == "eval" for r in R])
    w, b = load_probe()
    out = {}
    for st in STATES:
        s = Acts(st).pooled(ATTACK_LAYER, "mean").astype(np.float64) @ w + b
        ref = np.load(REF / "clean_acts.npz")["score"] if st == "clean" else \
            np.load(REF / "runs" / f"lam0__seed{st[-1]}.npz")["scores"][-1]
        a_new, a_ref = roc_auc_score(y[ev], s[ev]), roc_auc_score(y[ev], ref[ev])
        out[st] = {"auroc_new": a_new, "auroc_1019": a_ref, "auroc_diff": abs(a_new - a_ref),
                   "max_abs_score_diff": float(np.abs(s - ref).max())}
        out[st]["pass"] = bool(out[st]["auroc_diff"] <= 0.005 and out[st]["max_abs_score_diff"] <= 1e-3)
    out["all_pass"] = all(out[st]["pass"] for st in STATES)
    (DATA / "gate.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
