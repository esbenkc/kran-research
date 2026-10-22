"""PREREG s8 re-fit recovery: on one attacked state, re-fit the country-token probe tok_R (R = 1..29) on THAT state's
training-pool activations (probe+attack+stop, 1400 statements), C by the 1020 rule (5-fold GroupKFold by city, OOF
AUROC, ties within 0.002 -> smaller C), line from that state's 200 line-setting true statements.
Best layer = the highest training-pool OOF AUROC (no eval data). Shuffled-label null: 5 permutations at the best layer.
-> data/refit/<state>.npz (eval scores per layer + null scores) and data/refit/<state>.json.
Run: .venv/bin/python refit.py <state>
"""
import json
import sys
import time

import numpy as np

from common import DATA, Acts, pool_idx, rows
from panel import fit_lin, select_lin

OUT = DATA / "refit"


def main(state):
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    _, y, city, split = rows()
    tr, _, _ = pool_idx()
    trows = np.load(DATA / "fits" / "panel.npz")["tok_rows_clean"]
    A = Acts(state)
    S, info = {}, {}
    for R in range(1, 30):
        X = A.at(R, trows)
        C, res = select_lin(X, y, city, tr)
        sc, m = fit_lin(X, y, tr, C)
        S[f"tok{R}"] = m.decision_function(sc.transform(X))
        info[R] = {"C": C, "oof_auroc": res[C]}
        print(state, R, C, round(res[C], 4), f"{time.time() - t0:.0f}s", flush=True)
    best = max(info, key=lambda r: (info[r]["oof_auroc"], -r))
    X = A.at(best, trows)
    for p in range(1, 6):
        yp = y.copy()
        yp[tr] = y[tr][np.random.default_rng(p).permutation(len(tr))]
        sc, m = fit_lin(X, yp, tr, info[best]["C"])
        S[f"null{p}"] = m.decision_function(sc.transform(X))
    np.savez(OUT / f"{state}.npz", **S)
    (OUT / f"{state}.json").write_text(json.dumps({"state": state, "best_readout": best, "per_readout": info,
                                                   "seconds": round(time.time() - t0, 1)}, indent=1))
    print(state, "best", best, f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
