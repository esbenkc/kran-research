"""Known-answer cells of the LEACE price (PREREG s3/s4/s6), CPU only, from the erased clean states leace_price.py saved.
  oof_attack_auroc(ct): the s4 selection score (5-fold GroupKFold-by-city OOF AUROC of a logistic probe on the 600
                        attack-split statements, refit.py C rule, best readout)
  main: monitors.report on clean / registered / gate / seq1028 states -> data/leace/guard.json
        (gate rule s6: F2 catch on the gate state > 10% -> price clause untestable)
Run: venv/bin/python guard.py
"""
import json

import numpy as np

from common import DATA, rows
from refit import select

CT = DATA / "ctok"


def oof_attack_auroc(ct):
    from monitors import READ
    _, y, city, split = rows()
    atk = np.where(split == "attack")[0]
    per = {}
    for k, R in enumerate(READ):
        C, grid, conv = select(np.asarray(ct[:, k], np.float32), y, city, atk)
        per[R] = {"C": C, "oof_auroc": grid[C], "converged": conv}
    b = max(READ, key=lambda r: (per[r]["oof_auroc"], -r))
    return {"best_readout": b, "oof_auroc_at_best": per[b]["oof_auroc"], "per_readout": per}


def main():
    import monitors as M
    out = {}
    for st in ["clean", "leace_registered", "leace_gate", "leace_seq1028"]:
        rep, sc = M.report(np.load(CT / f"{st}.npy", mmap_mode="r"))
        out[st] = rep
        np.savez(DATA / "leace" / f"scores_{st}.npz", **sc)
        print(st, {k: (round(rep[k]["catch"], 4), round(rep[k]["auroc"], 3)) for k in ("M1", "F1", "F2", "frozen")},
              "F2 readout", rep["F2"]["readout"], "R29 fresh", round(rep["fresh_per_readout"][29]["catch"], 4), flush=True)
    out["gate_rule"] = {"F2_catch_gate": out["leace_gate"]["F2"]["catch"],
                        "clause4_testable": out["leace_gate"]["F2"]["catch"] <= 0.10}
    (DATA / "leace" / "guard.json").write_text(json.dumps(out, indent=1))
    print("GATE", out["gate_rule"], flush=True)


if __name__ == "__main__":
    main()
