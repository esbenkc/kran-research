"""E6b (Kasper #20595): the B-only ceiling for the frozen e1 itself, which sums over the cue rows.

Same model as ceiling.py E6 (A at its random init, kaiming-uniform bound 1/sqrt(d) as peft draws it; B grows only along
the cue's gradient), but with the REAL post-norm cue rows c_p (audit.py candidates: u1110 at the run's alpha, rows 0-2,
bf16 cast) and the REAL 1,000-cue null built the same way, scored with the frozen e1 = sum_p (v1.c_p)^2 / sum_p |c_p|^2.
Two assumptions for the output side of B (stated in the post as a model, not a measurement; Vera's N3 shows the real B
also learns generic modes):
  indep   each cue row p has its own output gradient g_p:  B_m = sum_p g_{m,p} (A_m c_p)^T
  shared  one output gradient for all rows:                B_m = g_m (A_m sum_p c_p)^T
Writes theory/ceiling_rows.json: per alpha and variant, e1 mean, p5-p95, the null p99 of the real candidate pool, the
band in margin units (e1 / null p99) that the figure draws, and P(rank 1).
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import audit  # noqa: E402

DRAWS = 200
OUTS = (2048, 1024, 1024)  # q, k, v output sizes of block 26


def main():
    torch.set_num_threads(2)
    rng = np.random.default_rng(1112)
    sp = audit.Spaces()
    base = sp.base()
    U0 = audit.null_dirs()
    res = {"draws": DRAWS, "model": "A at init (uniform +-1/sqrt(1024)), B only along the cue gradient"}
    for alpha in (0.03125, 0.0009765625):
        spec = {"rows": (0, 1, 2), "alpha": alpha, "cast": True, "real": False, "key": "u1110"}
        names, Cs = audit.candidates(sp, spec, base, U0)
        D = Cs["in26"]  # [N, 3, 1024], index 0 = true cue
        nul = np.array([n.startswith("null") for n in names])
        den = (D ** 2).sum(axis=(1, 2))
        c = D[0]
        for var in ("indep", "shared"):
            e_t, p99s, ranks = [], [], []
            for _ in range(DRAWS):
                G = np.zeros((1024, 1024))
                for dout in OUTS:
                    A = rng.uniform(-1 / 32, 1 / 32, (16, 1024))
                    if var == "indep":
                        B = sum(np.outer(rng.standard_normal(dout), A @ c[p]) for p in range(3))
                    else:
                        B = np.outer(rng.standard_normal(dout), A @ c.sum(0))
                    W = B @ A
                    G += W.T @ W
                ev, V = np.linalg.eigh(G)
                v1 = V[:, -1]
                e = (np.einsum("npd,d->np", D, v1) ** 2).sum(1) / den
                nv = e[nul]
                e_t.append(e[0])
                p99s.append(np.quantile(nv, 0.99))
                ranks.append(int(1 + (nv >= e[0]).sum()))
            e_t, p99s, ranks = np.array(e_t), np.array(p99s), np.array(ranks)
            m = e_t / p99s
            res[f"a{alpha}_{var}"] = {
                "e1_mean": float(e_t.mean()), "e1_p5_p95": [float(np.percentile(e_t, 5)), float(np.percentile(e_t, 95))],
                "null_p99_median": float(np.median(p99s)),
                "margin_p5_p95": [float(np.percentile(m, 5)), float(np.percentile(m, 95))],
                "margin_median": float(np.median(m)), "P_rank1": float((ranks == 1).mean()),
                "P_rank_le10": float((ranks <= 10).mean())}
            print(alpha, var, json.dumps(res[f"a{alpha}_{var}"]), flush=True)
    (HERE / "theory" / "ceiling_rows.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
