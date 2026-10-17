"""Iris: baseline numbers for research.md. Run: .venv/bin/python research_baselines.py"""
import sys

import numpy as np

from core import N, agreement, load_raw, run, second_argmax

sys.path.insert(0, "alg-zoo")
from alg_zoo.handcrafted import handcrafted_2nd_argmax  # noqa: E402

W_hi, W_hh, W_oh = load_raw()
n = 1_000_000
rng = np.random.default_rng(20261017)
xg = rng.standard_normal((n, N))
xu = rng.uniform(-1, 1, (n, N))


def pred(x, W=(W_hi, W_hh, W_oh)):
    wi, wh, wo = W
    h = np.zeros((x.shape[0], wh.shape[0]))
    for t in range(N):
        h = np.maximum(h @ wh.T + x[:, t : t + 1] * wi[None, :], 0.0)
    return (h @ wo.T).argmax(1)


pg, tg = pred(xg), second_argmax(xg)
pu, tu = pred(xu), second_argmax(xu)
print(f"model acc  N(0,1): {agreement(pg, tg):.4f}   Unif(-1,1): {agreement(pu, tu):.4f}")

# Model on top-2-only inputs (all other entries zeroed), as in Keane's LW comment.
def keep_top(x, k):
    idx = np.argsort(-x, 1)[:, :k]
    z = np.zeros_like(x)
    np.put_along_axis(z, idx, np.take_along_axis(x, idx, 1), 1)
    return z

for k in (2, 3, 4):
    z = keep_top(xg, k)
    pz = pred(z)
    print(f"keep top-{k}, rest=0: acc vs truth {agreement(pz, tg):.4f}, agreement with full model {agreement(pz, pg):.4f}")

# Handcrafted 22-neuron ARC model.
hc = handcrafted_2nd_argmax(10)
sd = {k: v.double().numpy() for k, v in hc.state_dict().items()}
Wh = (sd["rnn.weight_ih_l0"][:, 0], sd["rnn.weight_hh_l0"], sd["linear.weight"])
ph = pred(xg, Wh)
print(f"handcrafted(10): acc {agreement(ph, tg):.4f}, agreement with model {agreement(ph, pg):.4f}")

# Where the model errs: rank of the chosen position.
ranks = np.argsort(np.argsort(-xg, 1), 1)
r = ranks[np.arange(n), pg]
err = pg != tg
print("rank of model choice on errors (0=max,2=3rd...):", np.bincount(r[err], minlength=N)[:6] / err.sum())
m = xg.argmax(1)
print(f"error rate when max before 2nd: {err[m < tg].mean():.4f}; max after 2nd: {err[m > tg].mean():.4f}")
s_sorted = -np.sort(-xg, 1)
gap23 = s_sorted[:, 1] - s_sorted[:, 2]
for q in (0.05, 0.1, 0.2, 0.5):
    sel = gap23 < q
    print(f"gap(2nd-3rd)<{q}: frac {sel.mean():.3f}, err rate {err[sel].mean():.3f}; share of errors {err[sel].sum()/err.sum():.3f}")
print("error rate by true 2nd position:", np.round([err[tg == p].mean() for p in range(N)], 3))
print("error rate by max position:", np.round([err[m == p].mean() for p in range(N)], 3))
print(f"error rate when 2nd max <= 0: {err[s_sorted[:,1] <= 0].mean():.3f} (frac {np.mean(s_sorted[:,1] <= 0):.4f})")
