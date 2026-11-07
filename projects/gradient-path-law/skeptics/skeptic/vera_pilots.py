"""Vera: rebuild Bruno's pilots (task 2) with my tail on 64 status prompts: fed switch response to served cue,
prefix/rest splits, train-graph cue, decoy cue, graph mismatch. Shifts = mean decision logit(h_none + d) - (h_none)."""
import json, numpy as np, torch
from vera_tail import *
PROJ = '/Users/esben/work/oracle-1107-gradient-path-law/kran-research/projects/gradient-path-law/'
meta = json.load(open(PROJ + 'raw/cache/base26_L14/meta.json'))
pid = meta['prompt_ids'] if isinstance(meta['prompt_ids'], list) else eval(meta['prompt_ids'])
C = {k: np.load(PROJ + f'raw/cache/{g}/{c}_rec_tail.npy', mmap_mode='r') for k, (g, c) in
     {'Sn': ('base26_L14', 'none'), 'Sr': ('base26_L14', 'ref'), 'Tn': ('train26_L14', 'none'), 'Tr': ('train26_L14', 'ref')}.items()}
rng = np.random.default_rng(7)
idx = sorted(rng.choice(np.arange(446, 646), 64, replace=False).tolist())   # status prompts cd0446-cd0645
perm = np.roll(idx, 1)                                                        # decoy: the next sampled prompt
g = {k: bf(C[k][idx, 0]) for k in C}
dS, dT = g['Sr'] - g['Sn'], g['Tr'] - g['Tn']
dS_decoy = bf(C['Sr'][perm, 0]) - bf(C['Sn'][perm, 0])
pre = torch.zeros(1, 97, 1); pre[:, 1:8] = 1; rest = torch.zeros(1, 97, 1); rest[:, 8:] = 1
W = load_base(); out = {}
for run in ['B_fed14pilot_s0', 'B_fed14pilot_s1', 'B_std14pilot_s0', 'B_std14pilot_s1', 'pilot_standard_s0']:
    M = merge(W, PROJ + f'data/runs/{run}/adapter.pt', 2.0)
    base = tail(M, g['Sn'])
    sh = lambda h: float((tail(M, h) - base).mean())
    r = dict(served=sh(g['Sn'] + dS), served_pre=sh(g['Sn'] + dS * pre), served_rest=sh(g['Sn'] + dS * rest),
             decoy=sh(g['Sn'] + dS_decoy), train_cue=sh(g['Sn'] + dT), train_cue_pre=sh(g['Sn'] + dT * pre),
             train_cue_rest=sh(g['Sn'] + dT * rest), train_pair_own_base=float((tail(M, g['Tr']) - tail(M, g['Tn'])).mean()),
             mismatch_Tn_vs_Sn=sh(g['Tn']), served_x05=sh(g['Sn'] + 0.5 * dS), served_x2=sh(g['Sn'] + 2 * dS))
    r['cross_response'] = r['train_cue'] / r['served'] if abs(r['served']) > 0.1 else None
    out[run] = r; print(run, {k: (round(v, 3) if v is not None else None) for k, v in r.items()}, flush=True)
json.dump(out, open('vera_pilots.json', 'w'), indent=1)
