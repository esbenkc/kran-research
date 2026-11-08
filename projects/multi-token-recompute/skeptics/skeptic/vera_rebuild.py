"""Vera's rebuild of a read.py output from its raw npz (own code; field meanings checked against the smoke json:
c_<graph><text>: c_n = none graph/none text, c_l = leaky/leaky, c_nl = none graph on leaky text, c_ln = leaky graph
on none text; lp_n = log p_none of the none-sampled tokens, lp_l_n = log p_leaky of the same tokens).

Recomputes on the test rows: full served gap, text part, spill (points) with a cluster bootstrap over prompts
(seeds nested), f, f_null, D (E_none[(1 - p_l/p_n)+]), Pinsker, forced switch; the frozen per-seed verdict
(CI-lo >= 3 -> YES, CI-hi < 3 -> NO); a label-swap null (each CRN pair's probe label swapped with prob 1/2,
2,000 draws) for the full gap; and the bound check text <= 100 x D + CI half-width.
usage: vera_rebuild.py <npz> [json]   -> prints and writes <npz stem>.vera.json next to my scripts
"""
import sys, os, json, numpy as np

npz = sys.argv[1]
r = np.load(npz)
J = json.load(open(sys.argv[2] if len(sys.argv) > 2 else npz.replace('.npz', '.json')))
rng = np.random.default_rng(1108)
te = r['split'] == 'test'
pid = r['prompt_idx'][te]
up, inv = np.unique(pid, return_inverse=True)
P = len(up)
c = {k: r[k][te].astype(np.float64) for k in ('c_n', 'c_l', 'c_nl', 'c_ln') if k in r.files}


def per_prompt(x):
    s = np.bincount(inv, x, minlength=P); n = np.bincount(inv, minlength=P)
    return s / n


def boot(x, B=4000):
    pp = per_prompt(x)
    idx = rng.integers(0, P, (B, P))
    return float(pp.mean() * 100), *[float(v * 100) for v in np.percentile(pp[idx].mean(1), [2.5, 97.5])]


gap = c['c_n'] - c['c_l']; text = c['c_n'] - c['c_nl']; spill = c['c_nl'] - c['c_l']
tn, tl = r['toks_n'][te], r['toks_l'][te]
fl = (tn != tl).any(1)
L = (r['lp_l_n'][te].astype(np.float64) - r['lp_n'][te].astype(np.float64)).sum(1)
Dr = np.clip(1 - np.exp(L), 0, None)
kl = r['kl'][te].astype(np.float64).sum(1)
out = dict(file=os.path.basename(npz), rows=int(te.sum()), prompts=P, R=int(te.sum() // P),
           cheat_none=float(c['c_n'].mean()), cheat_leaky=float(c['c_l'].mean()),
           gap_pts=boot(gap), text_pts=boot(text), spill_pts=boot(spill),
           f=float(fl.mean()), D=float(Dr.mean()), D_ci=[float(v) for v in np.percentile(
               per_prompt(Dr)[rng.integers(0, P, (4000, P))].mean(1), [2.5, 97.5])],
           kl_seq=float(kl.mean()), pinsker=float(np.sqrt(kl.mean() / 2)))
if 'toks_n2' in r.files: out['f_null'] = float((tn != r['toks_n2'][te]).any(1).mean())
if 'c_X' in r.files:
    out['forced_switch_pts'] = boot(r['c_notX'][te].astype(np.float64) - r['c_X'][te].astype(np.float64))
# text part is exactly 0 on unflipped pairs?
out['text_nonzero_on_unflipped'] = int((np.abs(text[~fl]) > 0).sum())
g, lo, hi = out['gap_pts']
out['verdict'] = 'YES' if lo >= 3 else 'NO' if hi < 3 else 'MID'
t0, tlo, thi = out['text_pts']
out['bound_text_le_100D'] = bool(t0 <= 100 * out['D'] + (thi - tlo) / 2)
# label-swap null on the full gap: swap none/leaky labels per CRN pair with prob 1/2
sw = rng.random((2000, len(gap))) < 0.5
nulls = np.array([per_prompt(np.where(s, -gap, gap)).mean() * 100 for s in sw])
out['swap_null_95'] = [float(v) for v in np.percentile(nulls, [2.5, 97.5])]
out['swap_null_p_two_sided'] = float((np.abs(nulls) >= abs(g) - 1e-12).mean())
# compare with build's json
T = J.get('test', J)
cmp = {}
for k_me, k_b, scale in (('gap_pts', 'gap', 100), ('text_pts', 'text', 100), ('spill_pts', 'spill', 100),
                         ('f', 'f', 1), ('D', 'D', 1), ('pinsker', 'pinsker', 1), ('cheat_none', 'cheat_none', 1),
                         ('cheat_leaky', 'cheat_leaky', 1), ('f_null', 'f_null', 1), ('forced_switch_pts', 'forced_switch', 100)):
    if k_me in out and k_b in T and T[k_b] is not None:
        mine = out[k_me][0] if isinstance(out[k_me], list) or isinstance(out[k_me], tuple) else out[k_me]
        cmp[k_me] = dict(mine=mine, build=T[k_b] * scale, absdiff=abs(mine - T[k_b] * scale))
out['vs_build'] = cmp
print(json.dumps(out, indent=1))
here = os.path.dirname(os.path.abspath(__file__))
json.dump(out, open(os.path.join(here, os.path.basename(npz).replace('.npz', '.vera.json')), 'w'), indent=1)
