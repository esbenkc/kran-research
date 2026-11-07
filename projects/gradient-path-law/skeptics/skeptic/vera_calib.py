"""Vera: calibrate my tail (LoRA scale) against the build's served status logits (none), 200 test prompts."""
import json, sys, numpy as np, torch
from vera_tail import *
PROJ = '/Users/esben/work/oracle-1107-gradient-path-law/kran-research/projects/gradient-path-law/'
PAR = '/Users/esben/work/oracle-1106-shaping-watched/kran-research/projects/shaping-watched/data/status/'
meta = json.load(open(PROJ + 'raw/cache/base26_L14/meta.json'))
pid = meta['prompt_ids'] if isinstance(meta['prompt_ids'], list) else eval(meta['prompt_ids'])
Sn = np.load(PROJ + 'raw/cache/base26_L14/none_rec_tail.npy', mmap_mode='r')
Sr = np.load(PROJ + 'raw/cache/base26_L14/ref_rec_tail.npy', mmap_mode='r')
W = load_base()
out = {}
for run in ['pilot_standard_s0', 'B_fed14pilot_s0', 'B_fed14pilot_s1', 'B_std14pilot_s1']:
    z = np.load(PAR + f'{run}_raw/none_plain.npz'); zr = np.load(PAR + f'{run}_raw/ref_plain.npz')
    ids = list(z['prompt_ids'])[:40]; idx = [pid.index(i) for i in ids]
    ref_n = (z['l_read'] - z['l_solve'])[:40]; ref_r = (zr['l_read'] - zr['l_solve'])[:40]
    hn = bf(Sn[idx, 0]); hr = bf(Sr[idx, 0])
    for s in ([0.5, 1.0, 2.0] if run == 'B_fed14pilot_s0' else [None]):
        s = s or out.get('best_scale', 2.0)
        M = merge(W, PROJ + f'data/runs/{run}/adapter.pt', s)
        mn, mr = tail(M, hn).numpy(), tail(M, hr).numpy()
        rec = dict(scale=s, mae_none=float(np.abs(mn - ref_n).mean()), corr_none=float(np.corrcoef(mn, ref_n)[0, 1]),
                   shift_mine=float((mn - mr).mean()), shift_build=float((ref_n - ref_r).mean()), mean_mine=float(mn.mean()), mean_build=float(ref_n.mean()))
        out[f'{run}|{s}'] = rec; print(run, rec, flush=True)
        if run == 'B_fed14pilot_s0':
            if 'best' not in out or rec['mae_none'] < out['best']: out['best'] = rec['mae_none']; out['best_scale'] = s
json.dump(out, open('vera_calib.json', 'w'), indent=1)
