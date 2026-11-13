"""Vera: rebuild the served (fresh live samples) read of the d010_s0 end adapter from its raw npz."""
import json, os, numpy as np
from vera_check import P as _PROJ
P = os.path.join(_PROJ, 'data/reads/served_d010_s0_1500.npz')
d = np.load(P); cx, cn, pr = d['c_x'].astype(float), d['c_notx'].astype(float), d['prompt_row']
u, inv = np.unique(pr, return_inverse=True); G = len(u)
def sums(col, W):
    s = np.zeros(G); np.add.at(s, inv, col); return W @ s
rng = np.random.default_rng(0)
W = np.vstack([np.ones(G)] + [np.bincount(rng.integers(0, G, G), minlength=G) for _ in range(2000)])
n = sums(np.ones_like(cx), W); off = sums(cn, W) / n; fs = sums(cn - cx, W) / n; s = fs / off
pc = lambda a: [round(float(a[0]), 4), round(float(np.percentile(a[1:], 2.5)), 4), round(float(np.percentile(a[1:], 97.5)), 4)]
# null: swap c_x and c_notx within each row at random (no X effect)
vals = []
for b in range(2000):
    f = rng.random(len(cx)) < 0.5
    a1, a2 = np.where(f, cx, cn), np.where(f, cn, cx)
    vals.append((a1 - a2).mean() / a1.mean())
obs = float(s[0])
out = dict(n=int(len(cx)), prompts=G, cheat_notX=pc(off), forced=pc(fs), s_P=pc(s),
           frac_rows_switch_pos=round(float((cn > cx).mean()), 3),
           null_rowswap_95=[round(float(np.percentile(vals, 2.5)), 4), round(float(np.percentile(vals, 97.5)), 4)],
           p=float((np.abs(vals) >= abs(obs)).mean()))
json.dump(out, open('vera_served.json', 'w'), indent=1); print(json.dumps(out))
