"""Sensitivity facts that constrain any hand algorithm. Run: .venv/bin/python sensitivity.py -> sensitivity.json"""
import json
import numpy as np
from core import run, sample, agreement
from subA import load_xunits

W_hi, W_hh, W_oh = load_xunits()
x = sample(100_000, 3)
l, H = run(x, W_hi, W_hh, W_oh, return_states=True)
p = l.argmax(1)
rng = np.random.default_rng(0)
out = {"noise_all": {}, "noise_Whh_only": {}, "noise_Woh_only": {}}
for eps in [0.0003, 0.001, 0.003, 0.01, 0.03]:
    a, b, c = [], [], []
    for _ in range(5):
        nh = 1 + eps * rng.standard_normal(W_hh.shape)
        ni = 1 + eps * rng.standard_normal(16)
        no = 1 + eps * rng.standard_normal(W_oh.shape)
        a.append(agreement(run(x, W_hi * ni, W_hh * nh, W_oh * no).argmax(1), p))
        b.append(agreement(run(x, W_hi, W_hh * nh, W_oh).argmax(1), p))
        c.append(agreement(run(x, W_hi, W_hh, W_oh * no).argmax(1), p))
    out["noise_all"][eps] = a; out["noise_Whh_only"][eps] = b; out["noise_Woh_only"][eps] = c
    print(f"rel noise {eps}: all {np.mean(a):.4f}  W_hh only {np.mean(b):.4f}  W_oh only {np.mean(c):.4f}")
h = H[:, -1]
contrib = np.abs(h[:, None, :] * W_oh[None]).sum(2)
s = np.sort(l, 1); margin = s[:, -1] - s[:, -2]
out["median_logit_abs"] = float(np.median(np.abs(l).mean(1)))
out["median_margin"] = float(np.median(margin))
out["mean_sum_abs_contrib"] = float(contrib.mean())
out["frac_margin_lt_1pct_of_max_sum_abs_contrib"] = float(np.mean(margin < 0.01 * contrib.max(1)))
print({k: v for k, v in out.items() if not k.startswith("noise")})
json.dump(out, open("sensitivity.json", "w"), indent=1)
