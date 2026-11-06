"""Kasper's rho gate (PREREG v1.1): point-biserial correlation between the shaping probe's score of a sampled answer
and 'the answer is the cheat', over the pilot policy's samples on the 200 test prompts (exact mixture weights:
each prompt contributes p_i at (M_read_i, 1) and 1 - p_i at (M_solve_i, 0), p_i = 2-way cheat in the real graph).
    python rho.py -> data/rho.json
"""
import json
from pathlib import Path
import numpy as np
HERE = Path(__file__).parent
st = json.load(open(HERE / "data/status/pilot_standard_s0.json"))
z = np.load(HERE / "data/status/pilot_standard_s0_raw/none_plain.npz")
p = z["p_read"] / (z["p_read"] + z["p_solve"])
ids = [json.loads(l)["id"] for l in open(HERE / "raw/prompts.jsonl")]
pos = {k: i for i, k in enumerate(ids)}
M = np.load(HERE / "data/probe_M.npy")[[pos[str(k)] for k in z["prompt_ids"]]]
w = np.r_[p, 1 - p] / len(p)
x = np.r_[M[:, 0], M[:, 1]]
y = np.r_[np.ones(len(p)), np.zeros(len(p))]
mx, my = (w * x).sum(), (w * y).sum()
rho = (w * (x - mx) * (y - my)).sum() / np.sqrt((w * (x - mx) ** 2).sum() * (w * (y - my) ** 2).sum())
out = {"rho": float(rho), "n_prompts": int(len(p)), "mean_cheat_pilot": float(p.mean()),
       "M_read_mean": float(M[:, 0].mean()), "M_solve_mean": float(M[:, 1].mean())}
json.dump(out, open(HERE / "data/rho.json", "w"), indent=1)
print(out)
