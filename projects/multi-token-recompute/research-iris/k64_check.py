"""Check of the frozen length prediction (DEVIATIONS D3: k64 D in [0.035, 0.065]) on the k64 start read.
Prints per-16-token blocks: -log p per token and sd of the per-token log-ratio, and D over the first k tokens."""
import json, numpy as np
z = np.load('data/reads/raw/start_k64.npz'); m = z['split'] == 'test'
e = (z['lp_l_n'] - z['lp_n'])[m].astype(float); lp = z['lp_n'][m]
out = {'blocks': {}, 'D_first_k': {}}
for a in range(0, 64, 16):
    out['blocks'][f'{a+1}-{a+16}'] = dict(nll=float(-lp[:, a:a+16].mean()), sd_eps=float(e[:, a:a+16].std()))
for k in [16, 32, 48, 64]:
    out['D_first_k'][k] = float(np.mean(np.clip(1 - np.exp(e[:, :k].sum(1)), 0, None)))
out['frozen_band'] = [0.035, 0.065]; out['verdict'] = 'inside' if 0.035 <= out['D_first_k'][64] <= 0.065 else 'outside'
json.dump(out, open('research-iris/k64_check.json', 'w'), indent=1); print(json.dumps(out, indent=1))
