"""Vera: same pilots in bf16 (weights + activations), plus equal-norm random prefix directions, to test whether
the fed switch's response to the train-graph cue is a bf16-rounding effect. Also geometry of the graph mismatch."""
import json, numpy as np, torch
import vera_tail as VT
from vera_tail import *
PROJ = '/Users/esben/work/oracle-1107-gradient-path-law/kran-research/projects/gradient-path-law/'
C = {k: np.load(PROJ + f'raw/cache/{g}/{c}_rec_tail.npy', mmap_mode='r') for k, (g, c) in
     {'Sn': ('base26_L14', 'none'), 'Sr': ('base26_L14', 'ref'), 'Tn': ('train26_L14', 'none'), 'Tr': ('train26_L14', 'ref')}.items()}
rng = np.random.default_rng(7)
idx = sorted(rng.choice(np.arange(446, 646), 64, replace=False).tolist())
g = {k: bf(C[k][idx, 0]) for k in C}
dS, dT, dM = g['Sr'] - g['Sn'], g['Tr'] - g['Tn'], g['Tn'] - g['Sn']
pre = torch.zeros(1, 97, 1); pre[:, 1:8] = 1
P = lambda d: d[0, 1:8].flatten()
cos = lambda a, b: float((a @ b) / a.norm() / b.norm())
geo = dict(cos_dT_dS_pre=cos(P(dT), P(dS)), cos_dM_dS_pre=cos(P(dM), P(dS)), norm_dM_pre=float(P(dM).norm()), norm_dS_pre=float(P(dS).norm()))
print(geo)
gen = torch.Generator().manual_seed(11)
rand = []
for _ in range(4):
    u = torch.randn(1, 97, 1024, generator=gen) * pre
    rand.append(u / u[0, 1:8].norm() * P(dS).norm())
W = load_base(); out = {'geometry': geo}
orig_rms = VT.rms
for dtype in ['fp32', 'bf16']:
    for run in ['B_fed14pilot_s0', 'B_fed14pilot_s1', 'pilot_standard_s0']:
        M = merge(W, PROJ + f'data/runs/{run}/adapter.pt', 2.0)
        if dtype == 'bf16':
            M = {k: v.bfloat16() for k, v in M.items()}
            f = lambda h: tail(M, h.bfloat16()).float()
        else:
            f = lambda h: tail(M, h)
        base = f(g['Sn'])
        sh = lambda h: float((f(h) - base).mean())
        r = dict(served=sh(g['Sn'] + dS), served_pre=sh(g['Sn'] + dS * pre), train_cue=sh(g['Sn'] + dT),
                 train_cue_pre=sh(g['Sn'] + dT * pre), mismatch=sh(g['Tn']), mismatch_pre=sh(g['Sn'] + dM * pre),
                 rand_pre_eqnormS=[sh(g['Sn'] + u) for u in rand], rand_pre_eqnormT=[sh(g['Sn'] + u * (P(dT).norm() / P(dS).norm())) for u in rand])
        out[f'{dtype}|{run}'] = r
        print(dtype, run, {k: (round(v, 3) if isinstance(v, float) else [round(x, 3) for x in v]) for k, v in r.items()}, flush=True)
json.dump(out, open('vera_pilots_bf16.json', 'w'), indent=1)
