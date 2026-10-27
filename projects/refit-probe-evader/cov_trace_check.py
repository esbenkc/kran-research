"""Does concept-erasure's constrain_cov_trace=True (library default, used for P) change the eraser? Single-readout fits
on clean country-token states (fit pool 1400), lw setting: ||P_default - P_pure|| and the class-mean gap left on the
fit pool by each. -> data/leace/cov_trace_check.json"""
import json
import numpy as np, torch
from concept_erasure import LeaceFitter
from common import DATA, pool_idx, rows
_, y, _, _ = rows(); tr, _, _ = pool_idx()
c0 = np.load(DATA / "ctok" / "clean_fp32.npy", mmap_mode="r")
out = {}
for k, R in enumerate(range(10, 30)):
    X = torch.tensor(np.asarray(c0[tr, k]), dtype=torch.float64); z = torch.tensor(y[tr], dtype=torch.float64)[:, None]
    e1 = LeaceFitter.fit(X, z).eraser; e0 = LeaceFitter.fit(X, z, constrain_cov_trace=False).eraser
    gap = lambda e: float((e(X)[z[:, 0] == 1].mean(0) - e(X)[z[:, 0] == 0].mean(0)).norm() / (X[z[:, 0] == 1].mean(0) - X[z[:, 0] == 0].mean(0)).norm())
    dP = float((e1.P - e0.P).norm() / e0.P.norm())
    out[R] = {"rel_diff_P": dP, "rel_gap_default": gap(e1), "rel_gap_pure": gap(e0)}
    print(R, out[R], flush=True)
(DATA / "leace" / "cov_trace_check.json").write_text(json.dumps(out, indent=1))
