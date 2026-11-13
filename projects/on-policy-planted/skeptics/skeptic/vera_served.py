"""Vera: rebuild the served (fresh live samples, X forced vs banned at token 0, common random numbers) planted read
from the raw npz. s_P = mean(c_notx - c_x) / mean(c_notx); prompt-cluster bootstrap 2,000; row-swap null.
usage: python vera_served.py served_live_1200 [served_live_500 ...]"""
import json, os, sys
import numpy as np

P = os.path.expanduser('~/work/oracle-1113-on-policy-planted/kran-research/projects/on-policy-planted/data/reads')
pc = lambda a: [round(float(a[0]), 4), round(float(np.percentile(a[1:], 2.5)), 4), round(float(np.percentile(a[1:], 97.5)), 4)]
res = {}
for name in sys.argv[1:] or ['served_live_1200', 'served_live_500']:
    d = np.load(f'{P}/{name}.npz')
    cx, cn, pr = d['c_x'].astype(float), d['c_notx'].astype(float), d['prompt_row']
    u, inv = np.unique(pr, return_inverse=True); G = len(u)
    rng = np.random.default_rng(0)
    W = np.vstack([np.ones(G)] + [np.bincount(rng.integers(0, G, G), minlength=G) for _ in range(2000)])
    def sums(col):
        s = np.zeros(G); np.add.at(s, inv, col); return W @ s
    n = sums(np.ones_like(cx)); off = sums(cn) / n; fs = sums(cn - cx) / n; s = fs / off
    vals = []
    for b in range(2000):
        f = rng.random(len(cx)) < 0.5
        a1, a2 = np.where(f, cx, cn), np.where(f, cn, cx)
        vals.append((a1 - a2).mean() / a1.mean())
    extra = {}
    if 'toks_x' in d.files:
        extra['toks_x_token0_is_X'] = float((d['toks_x'][:, 0] == 1205).mean())
        extra['toks_notx_token0_is_X'] = float((d['toks_notx'][:, 0] == 1205).mean())
    res[name] = dict(n=int(len(cx)), prompts=G, cheat_notX=pc(off), forced=pc(fs), s_P=pc(s),
                     frac_rows_switch_pos=round(float((cn > cx).mean()), 3),
                     null_rowswap_95=[round(float(np.percentile(vals, 2.5)), 4), round(float(np.percentile(vals, 97.5)), 4)],
                     p=float((np.abs(np.array(vals)) >= abs(float(s[0]))).mean()), **extra)
json.dump(res, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'vera_served.json'), 'w'), indent=1)
print(json.dumps(res, indent=1))
