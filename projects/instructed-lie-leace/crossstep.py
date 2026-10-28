"""Secondary row (Kasper #9390): do the probes of one refit still catch 50 steps later? The blind-panel probes fit at
step k (data/runs/<run>__probes.npz, s<k>__*) score the eval prompts' states at step k + 50 with the PREREG s3 monitor
(z by eval negatives, max over readouts, 1% line). Compared with the refit at k + 50 on the same states.
States: R1 = data/states/<run>/s<step>.npy (rows = items.npy); R2 = Vera's extraction from the saved adapters,
swarm-theory-1028-instructed-lie/skeptic/ext_R2_s<step>_states.npy (items_spread order, 2496 rows).
-> data/results/crossstep_<run>.json
Usage: ILL_TAG=_spread venv/bin/python crossstep.py <run>
"""
import json
import sys
from pathlib import Path

import numpy as np

from common import DATA, REF, item_arrays
import monitors as M

VERA = Path(__file__).resolve().parents[3] / "swarm-theory-1028-instructed-lie" / "skeptic"


def states(run, step):
    d = DATA / "states" / run
    if (d / f"s{step}.npy").exists():
        H = np.load(d / f"s{step}.npy").astype(np.float32)
        return H, np.load(d / "items.npy")
    p = VERA / f"ext_R2_s{step}_states.npy"
    if run.startswith("lie") and p.exists():
        H = np.load(p).astype(np.float32)
        rows = np.load(VERA / f"ext_R2_s{step}_rows.npy")
        return H[rows], rows
    return None, None


def main(run):
    I, city, lie, decoy, split = item_arrays()
    y = lie if run.startswith("lie") else decoy
    P = np.load(DATA / "runs" / f"{run}__probes.npz")
    Upre, Upost = np.load(REF / "U_lens_pre.npy"), np.load(REF / "U_lens_post.npy")
    out = []
    for k in (100, 150, 200, 250):
        H, rows = states(run, k + 50)
        if H is None or f"s{k}__w" not in P.files:
            continue
        for j in range(H.shape[1]):
            U = Upost if j == H.shape[1] - 1 else Upre
            H[:, j] = H[:, j] - (H[:, j] @ U) @ U.T
        cell = {"fit_step": k, "scored_step": k + 50}
        for tag, s in (("stale", k), ("fresh", k + 50)):
            prm = {n: P[f"s{s}__{n}"] for n in ("mu", "sd", "w", "b")}
            ev = M.evaluate(M.panel_scores(H, prm), y[rows], split[rows])
            cell[f"{tag}_catch"], cell[f"{tag}_auroc"] = ev["catch"], ev["auroc"]
        out.append(cell)
        print(json.dumps(cell), flush=True)
    (DATA / "results" / f"crossstep_{run}.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
