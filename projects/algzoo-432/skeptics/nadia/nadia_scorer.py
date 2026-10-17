"""Nadia attack on 'readable rules stop near 96%' / 'best <=200-number description = C1+ 0.956'.
New family (not in RESULTS.md): one shared per-position scorer, score_k = MLP(features of candidate k),
argmax over the 10 positions. Features per candidate: x_k, M, S, T (top-3 of (0, x)), is-max/2nd/3rd flags,
one-hot(position k), one-hot(position of the max). Trained by cross-entropy on the model's argmax with fresh
N(0,1) batches (dev only); scored on seed 432060 (1e6). Run: ../.venv/bin/python nadia_scorer.py H [steps]
"""
import json, sys
import numpy as np
import torch
from nadia_lib import Wi, Wh, Wo, fwd, sample

torch.manual_seed(0)
H = int(sys.argv[1]) if len(sys.argv) > 1 else 6
STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 4000
tWi, tWh, tWo = (torch.from_numpy(a).float() for a in (Wi, Wh, Wo))


def labels(x):
    h = torch.zeros(len(x), 16)
    for t in range(10):
        h = torch.relu(x[:, t:t + 1] * tWi + h @ tWh.T)
    return (h @ tWo.T).argmax(1)


def feats(x):
    n = len(x)
    z = torch.cat([torch.zeros(n, 1), x], 1)
    top = z.topk(3, dim=1).values                       # M, S, T over (0, x)
    rk = x.argsort(1, descending=True).argsort(1)       # rank of each position among x
    pm = x.argmax(1)
    f = [x[..., None], top[:, None, :].expand(n, 10, 3),
         torch.stack([(rk == r).float() for r in range(3)], -1),
         torch.eye(10)[None].expand(n, 10, 10),
         torch.eye(10)[pm][:, None, :].expand(n, 10, 10)]
    return torch.cat(f, -1)                             # (n, 10, 27)


net = torch.nn.Sequential(torch.nn.Linear(27, H), torch.nn.ReLU(), torch.nn.Linear(H, 1))
nparam = sum(p.numel() for p in net.parameters())
opt = torch.optim.Adam(net.parameters(), lr=3e-3)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, STEPS)
g = torch.Generator().manual_seed(4320600)
for s in range(STEPS):
    x = torch.randn(8192, 10, generator=g)
    loss = torch.nn.functional.cross_entropy(net(feats(x))[..., 0], labels(x))
    opt.zero_grad(); loss.backward(); opt.step(); sched.step()

X = sample(432060, 1_000_000)
m = fwd(Wi, Wh, Wo, X)
with torch.no_grad():
    pred = np.concatenate([net(feats(torch.from_numpy(X[a:a + 100_000]).float()))[..., 0].argmax(1).numpy()
                           for a in range(0, len(X), 100_000)])
res = {'H': H, 'params': nparam, 'steps': STEPS, 'agree_s432060': float((pred == m).mean())}
print(json.dumps(res))
json.dump(res, open(f'nadia_scorer_H{H}.json', 'w'))
