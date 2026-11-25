"""Is the 'not settled' flag (halves differ > 2 pts in any seed) explained by Monte-Carlo noise alone?
Estimate the noise sd of (half2 - half1) per seed from the spread of the 4 chains' own half-differences
(no drift assumed), then the expected false-flag rate per cell (3 seeds), vs the observed 29/180."""
import json, numpy as np
from vera_score import *
P = '/Users/esben/work/theory-1125-llc-bias-law/kran-research/projects/llc-bias-law'
out = {}
for arm, nseed in (('S', 3), ('P', 2)):
    d = json.load(open(P + f'/data/cells_{arm}/24x24x12r3.json'))  # for block size
    rows = []; pexp = []; obs = []
    for f in sorted(glob.glob(P + f'/data/cells_{arm}/*.json')):
        dd = json.load(open(f)); z = np.load(f.replace('.json', '_blocks.npz')); blk = dd['block']
        if tuple(int(t) for t in dd['case'].replace('r', 'x').split('x')) not in BLIND: continue
        cellp = {}
        for x in dd['rows']:
            if x['q'] < 0.1: continue
            B = z[f"nb{x['nbeta']}_q{x['q']:g}_c{x['c']:g}_s{x['seed']}"].astype(float)
            kept = B[:, int(round(x['burn'] / blk)):]; h = kept.shape[1] // 2
            dch = x['nbeta'] * (kept[:, h:].mean(1) - kept[:, :h].mean(1)) / x['lam'] * 100  # per chain, pts
            sd_mean = dch.std(ddof=1) / np.sqrt(len(dch))       # sd of the pooled (4-chain) difference, from chain spread
            obs_d = dch.mean()
            p_flag = 2 * (1 - stats.norm.cdf(2 / sd_mean)) if sd_mean > 0 else 0.0
            k = (x['nbeta'], x['q'], x['c'], dd['case'])
            cellp.setdefault(k, []).append((p_flag, abs(obs_d) > 2))
            rows.append(dict(z=obs_d / sd_mean if sd_mean > 0 else 0.0))
        for k, v in cellp.items():
            pexp.append(1 - np.prod([1 - p for p, _ in v])); obs.append(any(o for _, o in v))
    zz = np.array([r['z'] for r in rows])
    out[arm] = dict(n_cells=len(obs), observed_flags=int(np.sum(obs)), expected_flags_if_noise=float(np.sum(pexp)),
                    z_mean=float(zz.mean()), z_sd=float(zz.std()))
    print(arm, out[arm])
json.dump(out, open('vera_halves_noise.json', 'w'), indent=1)
