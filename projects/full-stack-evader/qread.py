"""Quick per-state read (no CI): catch per readout of tok/mean, mlp14, mega, base; KL; ranking. Usage: qread.py <state>"""
import json
import sys

import numpy as np

from metrics import catch, ranking

st = sys.argv[1]
S = dict(np.load(f"data/scores/{st}.npz"))
m = json.load(open(f"data/acts/{st}/meta.json"))
print(st, "KL eval", round(m["kl_eval_mean_every_position"], 4), "pile", round(m["kl_pile_mean"], 4),
      "rank", ranking(np.load(f"data/acts/{st}/lp.npy"), np.load(f"data/acts/{st}/off.npy"))["right_first"])
print("tok ", " ".join(f"{L}:{catch(S[f'tok{L}'])['catch']:.2f}" for L in range(1, 30)))
print("mean", " ".join(f"{L}:{catch(S[f'mean{L}'])['catch']:.2f}" for L in range(1, 29)))
print("mlp14", round(catch(S["mlp14"])["catch"], 3), "mega", round(catch(S["mega"])["catch"], 3), "base", round(catch(S["base"])["catch"], 3))
