"""T18 (Iris): score ledger candidates on noise-ROBUST vs noise-FRAGILE inputs (second bar proposal).

Eval inputs: dev seed 83, N(0,1), n = 1e6 (never used for training or selection; holdout 432432 not used).
Robust  = the model's argmax is unchanged under all 20 draws of the bar's noise (0.1% multiplicative on all
          432 weights, rng 7). Fragile = the rest.
Reference = 20 FRESH noise draws (rng 8): mean agreement of a noisy model copy on each subset. On robust
          inputs this is < 1 only because the rng-7 draws missed some fragile inputs.
Columns: all / robust / fragile agreement with the model's argmax, plus literal counts where defined.
Usage: .venv/bin/python robust_eval.py   (~3-5 min) -> robust_eval.json, robust_eval.md
"""
import json, sys
from pathlib import Path
import numpy as np
from residual_common import W_hi, W_hh, W_oh, data, c1
from residual_percfg import fit_percfg
from ladder import gauge, literals, rounded

HERE = Path(__file__).resolve().parent
N = int(sys.argv[1]) if len(sys.argv) > 1 else 1_000_000


def run(x, Wi, Wh, Wo):
    h = np.zeros((len(x), 16))
    for t in range(10):
        h = np.maximum(h @ Wh.T + x[:, t:t + 1] * Wi[None, :], 0)
    return h @ Wo.T


def noisy_preds(X, seed, draws=20):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(draws):
        e = lambda W: W * (1 + 0.001 * rng.standard_normal(W.shape))
        out.append(run(X, e(W_hi), e(W_hh), e(W_oh)).argmax(1))
    return out


X, L, m = data(83, N)
frag = np.zeros(N, bool)
for p in noisy_preds(X, 7):
    frag |= p != m
rob = ~frag
rows = []


def add(name, pred, family, literals_=None, note=""):
    ok = pred == m
    r = dict(name=name, family=family, literals=literals_, all=round(float(ok.mean()), 5),
             robust=round(float(ok[rob].mean()), 5), fragile=round(float(ok[frag].mean()), 4), note=note)
    rows.append(r); print(json.dumps(r), flush=True)


ref = [p == m for p in noisy_preds(X, 8)]
rows.append(dict(name="reference: model + fresh 0.1% weight noise (mean of 20)", family="bar",
                 all=round(float(np.mean([r.mean() for r in ref])), 5),
                 robust=round(float(np.mean([r[rob].mean() for r in ref])), 5),
                 fragile=round(float(np.mean([r[frag].mean() for r in ref])), 4), literals=None, note=""))
print(json.dumps(rows[-1]), flush=True)
add("null: true 2nd argmax", np.argsort(X, 1)[:, -2], "rule", 0)
add("C1: 2nd argmax with phantom 0", c1(X), "rule", 0)
import algorithm_c1plus
add("C1+: C1 + 3 fitted clauses (T22)", algorithm_c1plus.predict(X), "rule (4 fitted numbers)", 4)

Xf, Lf, _ = data(81, 2_000_000)
for bank, cands in (("x", (2,)), ("x,relu(x),running max,running 2nd", (0, 2))):
    pred_fn, nco = fit_percfg(Xf, Lf, bank, cands)
    add(f"T16 per-config linear, cands S+{'/'.join('MST'[k] for k in cands)}, bank [{bank}]", pred_fn(X), "fitted rule", nco)
del Xf, Lf

try:
    import torch
    from residual_common import cand_features
    for f in ("scratch/iris_mlp_3x256.pt",):
        ck = torch.load(HERE / f, weights_only=False); a = ck["args"]
        mods, din = [], None
        F, O = cand_features(X, a["K"], set(a["drop"].split(",")) - {""}); din = F.shape[2]
        for _ in range(a["layers"]):
            mods += [torch.nn.Linear(din, a["hidden"]), torch.nn.ReLU()]; din = a["hidden"]
        mods.append(torch.nn.Linear(din, 1))
        net = torch.nn.Sequential(*mods); net.load_state_dict(ck["state"])
        prior = torch.zeros(a["K"]); prior[1] = 8.0
        with torch.no_grad():
            s = torch.cat([net(torch.tensor(F[i:i + 100000])).squeeze(-1) + prior for i in range(0, N, 100000)])
        nparams = sum(q.numel() for q in net.parameters())
        add(f"T16 candidate MLP {a['layers']}x{a['hidden']}", O[np.arange(N), s.argmax(1).numpy()], "black box", nparams)
except FileNotFoundError as e:
    print("skip MLP:", e)

g = gauge(W_hi, W_hh, W_oh)
for k in (3, 4):
    add(f"original weights, gauge + all rounded to {k} sig digits", run(X, *rounded(g, k)).argmax(1), "network", literals(*g))
for f, label in (("ladder_pruned_0.99.npz", "ladder greedy-pruned"), ("ladder_pruned_0.995.npz", "ladder greedy-pruned")):
    d = np.load(HERE / f); W = (d["Wi"], d["Wh"], d["Wo"])
    for k in (None, 3, 4):
        Wk = W if k is None else rounded(W, k)
        add(f"{label} {f} {'exact' if k is None else f'@{k} digits'}", run(X, *Wk).argmax(1), "network", literals(*W))

BRUNO = ["prune_ft_T5_200_140.npz", "prune_ft_T5_200_140_q3f.npz", "prune_ft_T1_200_140.npz",
         "prune_ft_T5_160_130.npz", "prune_ft_T5_160_130_q3.npz", "prune_ft_slow_95_105.npz", "prune_ft_neurons_d15.npz"]
BRUNO += ["prune_ft_long_220_144.npz"] + [f"prune_ft_long_{k}_q3f.npz" for k in ("220_144", "210_140", "200_140", "190_135", "180_130", "170_125")]
BRUNO += sys.argv[2:]  # extra nets
for f in BRUNO:
    if not (HERE / f).exists():
        print("skip missing", f); continue
    d = np.load(HERE / f); W = (d["W_hi"], d["W_hh"], d["W_oh"])
    add(f"N1 {f}", run(X, *W).argmax(1), "fitted network", literals(*W))  # Bruno's convention (prune_ft.literals)

meta = dict(seed=83, n=N, fragile_frac=round(float(frag.mean()), 5), noise="0.1% mult, 20 draws rng 7 (split), 20 draws rng 8 (reference)")
json.dump(dict(meta=meta, rows=rows), open(HERE / "robust_eval.json", "w"), indent=1)
with open(HERE / "robust_eval.md", "w") as fh:
    fh.write(f"# T18 robust/fragile scoring (dev seed 83, n={N}, fragile = {meta['fragile_frac']:.2%} of inputs)\n\n")
    fh.write("| candidate | family | literals | all | robust | fragile |\n|---|---|---|---|---|---|\n")
    for r in rows:
        fh.write(f"| {r['name']} | {r['family']} | {r['literals'] if r['literals'] is not None else ''} | {r['all']} | {r['robust']} | {r['fragile']} |\n")
print(json.dumps(meta))
