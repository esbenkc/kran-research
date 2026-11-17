"""Pick Figure 1's test example (Owain). Rule posted in the room (#23051) before any row was read:

ONE stored test row (its thought without ' We', and the same thought with ' We' written over token 0). Kept rows only
(the plain thought does not already start with ' We'), whose no-We cheat chance is in [0.3, 0.8] in all four
models (start, fixed thoughts at 1,200, fresh thoughts + penalty at 1,200, fresh thoughts no penalty at --step).
Score = sum over the three trained runs of |row switch - run median switch| (switch = c_none - c_We) + 0.5 x the
drift of the no-We chance from its start value. Lowest score wins. Prints the counts first.

    ../../../env/.venv/bin/python figure1/pick.py --seed 0 --step 1200
"""
import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
from live_summ import Test  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--step", type=int, required=True, help="read of the no-penalty seed")
ap.add_argument("--n", type=int, default=8)
a = ap.parse_args()
from transformers import AutoTokenizer  # noqa: E402

tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
te = Test("test")
cols = np.load(HERE / "data/pool/test/cols.npz")
n0, x0 = te.by["none"], te.by["xins"]
D = HERE / "data"
src = {"start": D / "parent/live_k4/reads/c_0.npy", "fixed": D / "parent/cached_d010_s0/reads/c_1200.npy",
       "pen": D / "parent/live_k4/reads/c_1200.npy", "nopen": D / f"runs/kl0_s{a.seed}/reads/c_{a.step}.npy"}
c = {k: np.load(p) for k, p in src.items()}
assert np.allclose(np.load(D / f"runs/kl0_s{a.seed}/reads/c_0.npy"), c["start"], atol=1e-6), "same start"
cn = {k: v[n0] for k, v in c.items()}
sw = {k: v[n0] - v[x0] for k, v in c.items()}
ok = te.keep.copy()
for k in cn:
    ok &= (cn[k] > 0.3) & (cn[k] < 0.8)
print(f"base rows {len(n0)}; kept {int(te.keep.sum())}; in [0.3,0.8] in all four {int(ok.sum())}")
med = {k: float(np.median(sw[k][ok])) for k in ("fixed", "pen", "nopen")}
print("median row switch (c_none - c_We) over eligible rows:", {k: round(v, 4) for k, v in med.items()})
score = sum(np.abs(sw[k] - med[k]) for k in med) + 0.5 * sum(np.abs(cn[k] - cn["start"]) for k in ("fixed", "pen", "nopen"))
score[~ok] = 9
for r in np.argsort(score)[: a.n]:
    t = tok.decode(cols["toks"][n0[r]][:24]).replace("\n", " ")
    cells = "  ".join(f"{k} {cn[k][r]:.2f}/{c[k][x0[r]]:.2f}" for k in src)
    print(f"row {r} score {score[r]:.3f}: {cells} | {t}")
