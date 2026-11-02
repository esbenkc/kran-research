"""P(Yes)/P(No) of candidate title rules on pooled open-route key use (equal-weight mean of 5 models),
95% percentile bootstrap within model, B=2000. Rules: Felix (lo>0.105 & >=4/5 models >0 -> Yes; hi<0.105 -> No),
Kasper (lo>=0.20 -> Yes; hi<0.10 -> No). True per-model open rates = true mean x model profile (blocked anchors)."""
import numpy as np, sys
prof = np.array([1.0, 0.875, 1.0, 0.975, 0.75]); prof = prof / prof.mean()
n = int(sys.argv[1]) if len(sys.argv) > 1 else 20
rng = np.random.default_rng(2); S, B = 400, 2000
print(f"n={n}/model; true mean | Felix Yes  No | Kasper Yes  No")
for t in [0.0, 0.03, 0.06, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40]:
    fy = fn = ky = kn = 0
    for s in range(S):
        k = rng.binomial(n, np.clip(t * prof, 0, 1)); m = k.mean() / n
        bs = rng.binomial(n, k / n, size=(B, 5)).mean(1) / n; lo, hi = np.percentile(bs, [2.5, 97.5])
        fy += lo > 0.105 and (k > 0).sum() >= 4; fn += hi < 0.105; ky += lo >= 0.20; kn += hi < 0.10
    print(f"{t:5.2f} | {fy/S:5.2f} {fn/S:5.2f} | {ky/S:5.2f} {kn/S:5.2f}")
