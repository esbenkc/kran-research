"""Vera rebuild of the round-2 key numbers from raw files. Dev seeds 901-905 only.
Run: cd swarm-algzoo-432 && .venv/bin/python skeptic/vera_checks.py  -> skeptic/vera_checks.json
"""
import json, sys, glob
import numpy as np
import torch
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from vera_core import load, logits, pred, sample, aug_order, HERE

torch.set_num_threads(4)
out = {}
W = load()
Wi, Wh, Wo = W

# 1. forward == torch.nn.RNN
rnn = torch.nn.RNN(1, 16, nonlinearity="relu", bias=False, batch_first=True).double()
lin = torch.nn.Linear(16, 10, bias=False).double()
with torch.no_grad():
    rnn.weight_ih_l0.copy_(torch.tensor(Wi[:, None])); rnn.weight_hh_l0.copy_(torch.tensor(Wh)); lin.weight.copy_(torch.tensor(Wo))
    Xs = sample(900, 20000)
    lt = lin(rnn(torch.tensor(Xs[:, :, None]))[0][:, -1]).numpy()
out["torch_maxdiff"] = float(np.abs(lt - logits(Xs, *W)).max())

# 2. null / phantom rule / accuracy, dev seed 901, 1e6
X = sample(901, 1_000_000)
L = logits(X, *W); y = L.argmax(1)
true2 = np.argsort(-X, 1)[:, 1]
pos, ph = aug_order(X)
out["model_acc_vs_truth"] = float((y == true2).mean())
out["null_agree"] = float((true2 == y).mean())
out["phantom2nd_agree"] = float((pos[:, 1] == y).mean())
# top-2 logit margin
Ls = np.sort(L, 1)
out["frac_margin_lt_1pct_of_top"] = float(((Ls[:, -1] - Ls[:, -2]) < 0.01 * np.abs(Ls[:, -1])).mean())

# 3. oracle candidate-set ceilings: pick model answer if in set else phantom-S
def ceil(ranks):
    inset = np.zeros(len(y), bool)
    for r in ranks:
        inset |= pos[:, r] == y
    return float(np.where(inset, 1.0, (pos[:, 1] == y)).mean())
names = {"S": [1], "MS": [0, 1], "ST": [1, 2], "ST4": [1, 2, 3], "MST": [0, 1, 2], "MST4": [0, 1, 2, 3]}
out["ceil_phantom"] = {k: ceil(v) for k, v in names.items()}
# same without phantom (plain ranks)
po = np.argsort(-X, 1)
def ceil_plain(ranks):
    inset = np.zeros(len(y), bool)
    for r in ranks:
        inset |= po[:, r] == y
    return float(np.where(inset, 1.0, (po[:, 1] == y)).mean())
out["ceil_plain"] = {k: ceil_plain(v) for k, v in names.items()}

# 4. noise floor: 20 draws of (1 + 0.001 eps) on all 432 weights, dev seed 902 inputs
X2 = sample(902, 1_000_000); y2 = pred(X2, W)
rng = np.random.default_rng(903)
for lvl in (0.001, 0.0003):
    a = []
    for _ in range(20):
        Wn = tuple(w * (1 + lvl * rng.standard_normal(w.shape)) for w in W)
        a.append(float((pred(X2, Wn) == y2).mean()))
    out[f"noise_{lvl}"] = {"mean": float(np.mean(a)), "sd": float(np.std(a, ddof=1)), "min": min(a), "max": max(a)}

# 5. reload candidate nets from disk; literal counts; exact and 3-sig-digit agreement on dev 904
X3 = sample(904, 400_000); y3 = pred(X3, W)
def sig(w, d):
    w = np.asarray(w, float); m = w != 0
    e = np.floor(np.log10(np.abs(w[m])))
    r = w.copy(); r[m] = np.round(w[m] / 10 ** (e - d + 1)) * 10 ** (e - d + 1)
    return r
nets = {}
for f in sorted(glob.glob(f"{HERE}/prune_ft_*.npz") + glob.glob(f"{HERE}/ladder_pruned_*.npz")):
    d = np.load(f)
    k = list(d.keys())
    w = (d["W_hi"], d["W_hh"], d["W_oh"]) if "W_hi" in k else (d["Wi"], d["Wh"], d["Wo"])
    w = tuple(np.asarray(a, float).reshape(s) for a, s in zip(w, [(16,), (16, 16), (10, 16)]))
    nz = [int((a != 0).sum()) for a in w]
    nonunit = [int(((a != 0) & (np.abs(np.abs(a) - 1) > 1e-12)).sum()) for a in w]
    nets[f.rsplit("/", 1)[1]] = {"nz_hi_hh_oh": nz, "nonunit_hi_hh_oh": nonunit,
                                 "agree_exact": float((pred(X3, w) == y3).mean()),
                                 "agree_3d": float((pred(X3, tuple(sig(a, 3) for a in w)) == y3).mean())}
out["nets_dev904"] = nets
# reference: full model at 3 / 2 sig digits (raw, ungauged)
out["model_3d"] = float((pred(X3, tuple(sig(a, 3) for a in W)) == y3).mean())
out["model_2d"] = float((pred(X3, tuple(sig(a, 2) for a in W)) == y3).mean())
json.dump(out, open(f"{HERE}/skeptic/vera_checks.json", "w"), indent=1)
print(json.dumps(out, indent=1))
