"""Vera: independent carry for arm T (and any run) with my own tail on the recorded block-26 caches.

For each run adapter: decision logit z = l_read - l_solve at position 95, on the 200 status prompts (cd0446-cd0645).
  served shift  s = mean z(served none) - z(served leaky)        [my tail on raw/cache/base26_L14]
  trainer shift t = mean z(train none)  - z(train leaky)         [my tail on raw/cache/train26_L14]
  carry = s / t, paired bootstrap CI over prompts (4000, my seed); fp32 and bf16 tails.
Null (random nudge): 8 random constant directions on prefix positions 1-7 with |d_S| norm, added to served none:
  carry_rand = mean z(Sn) - z(Sn + u)  /  t.   Compare the measured carry with that band.
Also reports the build's served shift from data/status/<run>_raw/{none,ref}_plain.npz (real compiled graph) / my t.
Usage: python vera_carry.py RUN [RUN ...]
"""
import sys, os, json, numpy as np, torch
from vera_tail import *
PROJ = '/Users/esben/work/oracle-1107-gradient-path-law/kran-research/projects/gradient-path-law/'
C = {k: np.load(PROJ + f'raw/cache/{g}/{c}_rec_tail.npy', mmap_mode='r') for k, (g, c) in
     {'Sn': ('base26_L14', 'none'), 'Sr': ('base26_L14', 'ref'), 'Tn': ('train26_L14', 'none'), 'Tr': ('train26_L14', 'ref')}.items()}
meta = json.load(open(PROJ + 'raw/cache/base26_L14/meta.json'))
pid = meta['prompt_ids'] if isinstance(meta['prompt_ids'], list) else eval(meta['prompt_ids'])
rng = np.random.default_rng(20261107)
W = load_base()
pre = torch.zeros(1, 97, 1); pre[:, 1:8] = 1
gen = torch.Generator().manual_seed(1107)


def boot_ratio(a, b, B=4000):
    n = len(a); idx = rng.integers(0, n, (B, n))
    r = a[idx].mean(1) / b[idx].mean(1)
    return [float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))]


def run_one(run):
    status_ids = list(np.load(PROJ + f'data/status/{run}_raw/none_plain.npz')['prompt_ids'])
    idx = [pid.index(i) for i in status_ids]
    M32 = merge(W, PROJ + f'data/runs/{run}/adapter.pt', 2.0)
    out = {'n': len(idx)}
    dS1 = bf(C['Sr'][idx[:1], 0]) - bf(C['Sn'][idx[:1], 0])
    rand = []
    for _ in range(8):
        u = torch.randn(1, 97, 1024, generator=gen) * pre
        rand.append(u / u[0, 1:8].norm() * dS1[0, 1:8].norm())
    for prec in os.environ.get('PRECS', 'fp32,bf16').split(','):
        M = M32 if prec == 'fp32' else {k: v.bfloat16() for k, v in M32.items()}
        f = (lambda h: tail(M, h)) if prec == 'fp32' else (lambda h: tail(M, h.bfloat16()).float())
        z = {k: [] for k in C}; zr = [[] for _ in rand]
        for i in range(0, len(idx), 25):
            sl = idx[i:i + 25]
            hs = {k: bf(C[k][sl, 0]) for k in C}
            for k in C: z[k].append(f(hs[k]).numpy())
            for j, u in enumerate(rand): zr[j].append(f(hs['Sn'] + u).numpy())
        z = {k: np.concatenate(v) for k, v in z.items()}; zr = [np.concatenate(v) for v in zr]
        s, t = z['Sn'] - z['Sr'], z['Tn'] - z['Tr']
        rc = [float((z['Sn'] - x).mean() / t.mean()) for x in zr]
        out[prec] = dict(served_shift=float(s.mean()), trainer_shift=float(t.mean()), carry=float(s.mean() / t.mean()),
                         ci_carry=boot_ratio(s, t), rand_carry=rc, rand_abs_max=float(np.max(np.abs(rc))),
                         cheat_served_none=float(sig(z['Sn']).mean()), cheat_served_leaky=float(sig(z['Sr']).mean()),
                         cheat_train_none=float(sig(z['Tn']).mean()), cheat_train_leaky=float(sig(z['Tr']).mean()))
        if prec == 'fp32': tvec = t
    bn = np.load(PROJ + f'data/status/{run}_raw/none_plain.npz'); br = np.load(PROJ + f'data/status/{run}_raw/ref_plain.npz')
    bs = (bn['l_read'] - bn['l_solve']) - (br['l_read'] - br['l_solve'])
    out['build_served_shift_real_graph'] = float(bs.mean())
    out['carry_real_served_over_my_trainer_fp32'] = float(bs.mean() / tvec.mean()); out['ci'] = boot_ratio(bs, tvec)
    return out


sig = lambda x: 1 / (1 + np.exp(-x))
res = {}
for run in sys.argv[1:]:
    res[run] = run_one(run)
    print(run, json.dumps(res[run], indent=0), flush=True)
json.dump(res, open(f'vera_carry_{"_".join(sys.argv[1:])}.json', 'w'), indent=1)
