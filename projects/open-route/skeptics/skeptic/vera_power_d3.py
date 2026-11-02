"""Power after the D3 re-smoke (c1 2/15, c4 4/15, c4 key outside route 2/15). D4: status on key-anywhere AND
key-outside; disagreement -> UNCONFIRMED. P(status) for the key-outside event alone (its Yes is the binding one),
95% within-model percentile bootstrap, B=2000, edge 0.1025, floor ~1/150."""
import numpy as np
rng = np.random.default_rng(5); S, B, e = 400, 2000, 0.1025
scen = {"homog 0.10": [0.10]*5, "homog 0.13": [0.13]*5, "homog 0.17": [0.17]*5, "homog 0.20": [0.20]*5,
        "concentrated 0.13 (2 models 0.33)": [0, .33, .33, 0, 0], "spread 0.13 (.05,.25,.25,.05,.05)": [.05, .25, .25, .05, .05]}
for n in (30, 40):
    print(f"n={n}/model")
    for name, p in scen.items():
        p = np.array(p); y = no = 0
        for _ in range(S):
            k = rng.binomial(n, p); kf = rng.binomial(n, np.full(5, 1/150))
            b = rng.binomial(n, k / n, size=(B, 5)).mean(1) / n; bf = rng.binomial(n, kf / n, size=(B, 5)).mean(1) / n
            lo, hi = np.percentile(b, [2.5, 97.5]); y += lo > e and np.percentile(b - bf, 2.5) > 0; no += hi < e
        print(f"  {name:36s} Yes {y/S:.2f}  No {no/S:.2f}  NotSettled {1-(y+no)/S:.2f}")
