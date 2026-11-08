"""Vera: numeric checks of the two displayed equations in _notes/gated-sinks.md (SAE foldout, lines ~107-108).
Eq.1  f = TopK_8(ReLU((x - b_dec) W_enc + b_enc)): checked equal to ReLU applied to the top-8 pre-activations
      (my encoder, vera_rebuild.topk_encode) on random inputs incl. rows with < 8 positive pre-activations; and the
      (x - b_dec) convention is the one that reproduces build's json (0.42314 vs 0.42315; plain x gives 0.410).
Eq.2  recovery = (1/64) sum_j max_i corr(f_i, z_j), dead features = 0: my vera_mcc.mcc reproduces all 12 counted
      models' json recovery to 4 dp (vera_grid_h2_final.json)."""
import json, numpy as np
from vera_rebuild import topk_encode
rng = np.random.default_rng(0)
X = rng.normal(size=(5000, 64)); W = rng.normal(size=(64, 256)) / 8; be = rng.normal(size=256) - 6.0; bd = rng.normal(size=64)
pre = (X - bd) @ W + be
relu = np.maximum(pre, 0)
idx = np.argsort(-relu, axis=1)[:, :8]
F1 = np.zeros_like(pre); np.put_along_axis(F1, idx, np.take_along_axis(relu, idx, 1), 1)   # TopK_8(ReLU(.)) as written
F2 = topk_encode(X, W, be, bd, 8, True)                                                      # my encoder
print("rows with <8 positive pre-acts:", int(((pre > 0).sum(1) < 8).sum()), " max |Eq1 - mine|:", float(np.abs(F1 - F2).max()))
g = json.load(open("vera_grid_h2_final.json"))
print("Eq2 max |vera - json| over 12 models:", max(abs(m["rec_build"] - m["json_rec"]) for m in g["models"]))
