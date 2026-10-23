"""Idealized attacker (theory curve, owner Iris): the country-token states themselves are the parameters.
Plain gradient descent (no Adam, no LoRA) on
    L(H) = sum_k gap_k^2 + lam * sum_i KL_i(H),   gap_k = s(h_false,k) - s(h_true,k),  s(h) = w.h + b0
    KL_i = KL(p_clean,i || p_i(h_i)) at the country token, exact RMSNorm + tied unembedding,
           vocabulary cut to the tokens with clean probability > P_MIN for some state (cut mass reported).
Step size lr = 1 / (4|w|^2 + 0.15 lam)  (4|w|^2 = curvature of the coupled gap; 0.15 > max output-Fisher eigenvalue
0.094 measured in theory.json C). Stops when the mean gap fell 95% or at MAX_STEPS.
Reports R (norm-restore share of the gap drop, the frozen title metric, on these pairs), the gap drop, the mean KL and
the length ratios, every LOG steps. This is the activation-space prediction: an attacker free of any weight constraint.
CPU float32 (torch), < 0.5 GB. Run (cwd = this project): ../../../venv1023/bin/python theory/flow_sim.py [lam ...]"""
import json, os, sys, time
from pathlib import Path
import numpy as np
import torch
from safetensors import safe_open

torch.set_num_threads(3)
ROOT = Path(__file__).resolve().parents[1]
SNAP = next(Path.home().glob('.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/*')) / 'model.safetensors'
EPS, V, D, CH = 1e-6, 151936, 1536, 8192
N_PAIRS, P_MIN, MAX_STEPS, LOG = 16, 1e-6, int(os.environ.get("MAX_STEPS", 20000)), 250
lams = [float(x) for x in sys.argv[1:]] or [0.0, 1e2, 1e4]

SPLIT = json.load(open(ROOT / 'data' / 'ref' / 'split.json'))
y = np.array([1 - r['true'] for r in SPLIT])
sp_ = np.array([r['split'] for r in SPLIT])
city = np.array([r['city'] for r in SPLIT]); src = np.array([r['src'] for r in SPLIT])
pairs = []
for i in np.where((sp_ == 'eval') & (y == 1))[0]:
    j = np.where((sp_ == 'eval') & (y == 0) & (city == city[i]) & (src == src[i]))[0]
    pairs.append((i, int(j[0])))
pairs = np.array(pairs)
sel = pairs[np.random.default_rng(0).choice(len(pairs), N_PAIRS, replace=False)]
idx = np.r_[sel[:, 0], sel[:, 1]]                      # first half false, second half their true twins
H0 = np.load(ROOT / 'theory' / 'inputs' / 'clean_country_R28.npy').astype(np.float64)[idx]
z = np.load(ROOT / 'data' / 'fits' / 'panel.npz', allow_pickle=True)
w = z['tok28__w'].astype(np.float64) / z['tok28__sd']
b0 = float(z['tok28__b']) - z['tok28__mu'].astype(np.float64) @ w

f = safe_open(str(SNAP), 'pt')
g = f.get_tensor('model.norm.weight').double()
E = f.get_slice('model.embed_tokens.weight')


def norm(x):
    return g * x / torch.sqrt((x * x).mean(-1, keepdim=True) + EPS)


# clean distributions over the full vocabulary (streamed), then the token cut
X0 = torch.tensor(H0)
Z = torch.cat([norm(X0) @ E[a:min(V, a + CH)].double().T for a in range(0, V, CH)], 1)
P0 = torch.softmax(Z, -1)
keep = (P0 > P_MIN).any(0)
cut_mass = float((1 - P0[:, keep].sum(1)).max())
Wk = torch.cat([E[a:min(V, a + CH)].double()[keep[a:min(V, a + CH)]] for a in range(0, V, CH)], 0).float()
del Z
lp0 = torch.log_softmax((norm(X0) @ Wk.double().T), -1).float()
p0 = lp0.exp()
gf = g.float(); wt = torch.tensor(w, dtype=torch.float32)


def normf(x):
    return gf * x / torch.sqrt((x * x).mean(-1, keepdim=True) + EPS)


print(f'pairs {N_PAIRS}, kept tokens {int(keep.sum())}, max cut mass {cut_mass:.2e}', flush=True)


def gaps(H):
    s = H @ wt
    return s[:N_PAIRS] - s[N_PAIRS:]


def R_of(H, n0):
    """frozen title metric on these pairs: restored = rescale each state back to its clean length."""
    Hr = H * (n0 / H.norm(dim=1)).unsqueeze(1)
    gc, ga, gr = float(gaps(X0f).mean()), float(gaps(H).mean()), float(gaps(Hr).mean())
    return (gr - ga) / (gc - ga) if gc != ga else float('nan'), (gc - ga) / gc


X0f = X0.float()
n0 = X0f.norm(dim=1)
wn2 = float(wt @ wt)
out = {'n_pairs': N_PAIRS, 'kept_tokens': int(keep.sum()), 'max_cut_mass': cut_mass, 'p_min': P_MIN,
       'cos2_wh_false_median': float(np.median(((H0[:N_PAIRS] @ w) / (np.linalg.norm(H0[:N_PAIRS], axis=1) * np.linalg.norm(w))) ** 2)),
       'runs': {}}
for lam in lams:
    lr = 1.0 / (4 * wn2 + 0.15 * lam)
    H = X0f.clone().requires_grad_(True)
    traj, t0 = [], time.time()
    for step in range(MAX_STEPS + 1):
        lq = torch.log_softmax(normf(H) @ Wk.T, -1)
        kl = (p0 * (lp0 - lq)).sum(1)
        gp = gaps(H)
        loss = (gp ** 2).sum() + lam * kl.sum()
        if step % LOG == 0 or step == MAX_STEPS:
            with torch.no_grad():
                R, drop = R_of(H.detach(), n0)
                row = dict(step=step, R=R, gap_drop_frac=drop, kl_mean=float(kl.mean()), kl_max=float(kl.max()),
                           norm_ratio_false=float((H[:N_PAIRS].norm(dim=1) / n0[:N_PAIRS]).median()),
                           norm_ratio_true=float((H[N_PAIRS:].norm(dim=1) / n0[N_PAIRS:]).median()))
            traj.append(row)
            if step % (LOG * 8) == 0:
                print(f'lam {lam:g} ' + json.dumps({k: round(v, 6) if isinstance(v, float) else v for k, v in row.items()}), flush=True)
            if drop >= 0.95:
                break
        (gr,) = torch.autograd.grad(loss, H)
        with torch.no_grad():
            H -= lr * gr
    out['runs'][f'{lam:g}'] = {'lr': lr, 'steps': traj[-1]['step'], 'final': traj[-1], 'traj': traj,
                               'seconds': round(time.time() - t0, 1)}
    print(f'lam {lam:g} FINAL ' + json.dumps(traj[-1]), flush=True)
OUTP = ROOT / "data" / "results" / ("flow_sim_lam" + "_".join(f"{l:g}" for l in lams) + ".json")
json.dump(out, open(OUTP, 'w'), indent=1)
print('wrote', OUTP)
