"""Hugo: agreement of the model with noisy copies vs noise level, many draws (dev seeds 7100+).
Run: cd swarm-algzoo-432 && .venv/bin/python skeptic2/hugo_noise_levels.py"""
import json
import numpy as np
import torch

sd = torch.load("m16_10.pth", map_location="cpu")
W = [sd["rnn.weight_ih_l0"].double().numpy()[:, 0], sd["rnn.weight_hh_l0"].double().numpy(),
     sd["linear.weight"].double().numpy()]


def pred(wi, wh, wo, X):
    h = np.zeros((X.shape[0], 16))
    for t in range(10):
        h = np.maximum(0.0, X[:, t:t + 1] * wi[None, :] + h @ wh.T)
    return (h @ wo.T).argmax(1)


X = np.random.default_rng(7100).standard_normal((100_000, 10))
P0 = pred(*W, X)
out = {}
for i, s in enumerate((0.001, 0.002, 0.003, 0.0035, 0.004)):
    r = np.random.default_rng(7110 + i)
    a = np.array([(pred(*[w * (1 + s * r.standard_normal(w.shape)) for w in W], X) == P0).mean() for _ in range(200)])
    out[s] = dict(mean=a.mean(), median=float(np.median(a)), sd=a.std(ddof=1), se=a.std(ddof=1) / np.sqrt(len(a)),
                  q05=float(np.quantile(a, .05)), q95=float(np.quantile(a, .95)),
                  mean_first10=a[:10].mean(), min10mean=min(a[i:i + 10].mean() for i in range(0, 200, 10)),
                  max10mean=max(a[i:i + 10].mean() for i in range(0, 200, 10)))
out = {str(k): {kk: float(vv) for kk, vv in v.items()} for k, v in out.items()}
json.dump(out, open("skeptic2/hugo_noise_levels.json", "w"), indent=1)
print(json.dumps(out, indent=1))
