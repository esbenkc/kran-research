"""Baseline: reproduce model accuracy, check numpy==torch, dump standardized weights.
Run: .venv/bin/python baseline.py   -> baseline.json, weights_std.json
"""
import json

import numpy as np
import torch
from alg_zoo import example_2nd_argmax

from core import agreement, load_raw, load_std, run, sample, second_argmax, standardize

m = example_2nd_argmax().double()
raw = load_raw()
W_hi, W_hh, W_oh, s = standardize(*raw)
x = sample(1_000_000, 0)
with torch.no_grad():
    lt = m(torch.from_numpy(x)).numpy()
l0 = run(x, *raw)
l1 = run(x, W_hi, W_hh, W_oh)
pred = l1.argmax(1)
out = {
    "n": len(x),
    "seed": 0,
    "max_abs_torch_minus_numpy": float(np.abs(lt - l0).max()),
    "max_abs_raw_minus_std": float(np.abs(l0 - l1).max()),
    "argmax_agree_torch_vs_std": agreement(lt.argmax(1), pred),
    "acc_vs_true_2nd_argmax": agreement(pred, second_argmax(x)),
    "pred_hist": np.bincount(pred, minlength=10).tolist(),
}
print(json.dumps(out, indent=1))
json.dump(out, open("baseline.json", "w"), indent=1)
json.dump(
    {"note": "neuron i rescaled by s_i=1/|W_hi_i| (exact equivalent model)",
     "s": s.tolist(), "W_hi": W_hi.tolist(), "W_hh": W_hh.tolist(), "W_oh": W_oh.tolist()},
    open("weights_std.json", "w"), indent=1)
