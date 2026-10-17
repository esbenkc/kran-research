"""Nadia part 3: post sentences line 57 (<=1 positive input -> wrong 78%) and line 72 (0.3% noise -> 95.2%).
Run: ../.venv/bin/python nadia_check3.py"""
import json
import numpy as np
from nadia_lib import Wi, Wh, Wo, fwd, sample

X = sample(432040)
m = fwd(Wi, Wh, Wo, X)
truth = np.argsort(-X, axis=1)[:, 1]
few = (X > 0).sum(1) <= 1
g = np.random.default_rng(432047)
e = lambda W, s: W * (1 + s * g.standard_normal(W.shape))
res = {'frac_le1_positive': float(few.mean()),
       'err_rate_le1_positive': float((m[few] != truth[few]).mean()),
       'floor_0.3pct_10draws': float(np.mean([(fwd(e(Wi, 3e-3), e(Wh, 3e-3), e(Wo, 3e-3), X) == m).mean() for _ in range(10)]))}
print(json.dumps(res, indent=1))
json.dump(res, open('nadia_check3.json', 'w'), indent=1)
