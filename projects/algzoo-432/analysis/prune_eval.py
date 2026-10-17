"""N1 (Bruno): evaluate saved pruned/rounded nets on a second dev seed (62, never used in training or
selection; training/selection used seed 61). Holdout 432432 is not used here (Rasmus runs it).

Per npz: literals, W_hh / W_oh nonzeros, agreement with the model on all inputs, on the model-error set
(inputs where the model does not pick the true 2nd argmax) and as-is (no further rounding).
Run: .venv/bin/python prune_eval.py prune_ft_long_200_140_q3f.npz [...]   -> prints one json row per file
"""
import json
import sys

import numpy as np

from core import load_raw, run, second_argmax
from prune_ft import literals

W_MODEL = load_raw()
X = np.random.default_rng(62).standard_normal((1_000_000, 10))
REF = run(X, *W_MODEL).argmax(1)
ERR = REF != second_argmax(X)

for f in sys.argv[1:]:
    d = np.load(f)
    W = (d["W_hi"], d["W_hh"], d["W_oh"])
    pred = run(X, *W).argmax(1)
    ok = pred == REF
    print(json.dumps({"file": f, "literals": literals(*W), "W_hh_nz": int((W[1] != 0).sum()),
                      "W_oh_nz": int((W[2] != 0).sum()), "dev62_all": round(float(ok.mean()), 5),
                      "dev62_error_set": round(float(ok[ERR].mean()), 5), "n_error_set": int(ERR.sum())}),
          flush=True)
