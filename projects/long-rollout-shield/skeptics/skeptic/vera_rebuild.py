"""Vera's own rebuild of one read (npz from read.py; schema as in the parent: c_n/c_l/c_nl/c_ln Rao-Blackwellised
cheat per row, toks_n/toks_l thought tokens per build under common random numbers, lp_n = log p_none of the
none-sampled thought, lp_l_n = log p_leaky of the same thought; planted/hash arms: 'l' = cue rows).

Recomputes on the test rows (frozen PREREG v1.2 definitions):
  served gap = cheat(none) - cheat(leaky) in points, prompt-cluster bootstrap 95% CI (2,000);
  text part, spill; D = E_none[(1 - exp(L))_+] (exact MC TV of the thought); f (share of CRN pairs whose text differs);
  distinct-thought share (none build), mean -log p per token, saturation share (none cheat > 0.97);
  forced range (c_notX - c_X) when present; ceiling = 100 x D x range;
  per-seed status cell: YES CI-lo >= 3, NO CI-hi < 3, else MID;
  nulls: (a) label swap inside each CRN pair (2,000 draws), (b) prompt-level sign flip (2,000 draws).
usage: vera_rebuild.py <npz> [build json] [--range PTS]  -> writes <stem>.vera.json beside this script
"""
import sys, os, json, numpy as np

args = [a for a in sys.argv[1:] if not a.startswith('--')]
rng_pts = None
if '--range' in sys.argv:
    rng_pts = float(sys.argv[sys.argv.index('--range') + 1]); args = [a for a in args if a != str(sys.argv[sys.argv.index('--range') + 1])]
npz = args[0]
jpath = args[1] if len(args) > 1 else npz.replace('.npz', '.json')
r = np.load(npz)
J = json.load(open(jpath)) if os.path.exists(jpath) else {}
rng = np.random.default_rng(1109)
te = r['split'] == 'test'
pid = r['prompt_idx'][te]
up, inv = np.unique(pid, return_inverse=True)
P = len(up)
cnt = np.bincount(inv, minlength=P)


def per_prompt(x):
    return np.bincount(inv, x, minlength=P) / cnt


IDX = rng.integers(0, P, (2000, P))


def boot(x):
    pp = per_prompt(x)
    m = pp[IDX].mean(1)
    return [float(pp.mean() * 100), float(np.percentile(m, 2.5) * 100), float(np.percentile(m, 97.5) * 100)]


c = {k: r[k][te].astype(np.float64) for k in ('c_n', 'c_l', 'c_nl', 'c_ln', 'c_X', 'c_notX') if k in r.files}
gap = c['c_n'] - c['c_l']
tn, tl = r['toks_n'][te], r['toks_l'][te]
fl = (tn != tl).any(1)
out = dict(file=os.path.basename(npz), rows=int(te.sum()), prompts=P, R=float(te.sum() / P), k=int(tn.shape[1]),
           cheat_none=float(c['c_n'].mean()), cheat_leaky=float(c['c_l'].mean()), gap_pts=boot(gap),
           f=float(fl.mean()), sat_none_gt_0p97=float((c['c_n'] > 0.97).mean()))
if 'c_nl' in c:
    out['text_pts'] = boot(c['c_n'] - c['c_nl']); out['spill_pts'] = boot(c['c_nl'] - c['c_l'])
if 'lp_l_n' in r.files:
    L = (r['lp_l_n'][te].astype(np.float64) - r['lp_n'][te].astype(np.float64))
    L = L.sum(1) if L.ndim == 2 else L
    Dr = np.clip(1 - np.exp(L), 0, None)
    pp = per_prompt(Dr)
    out['D'] = float(Dr.mean()); out['D_ci'] = [float(v) for v in np.percentile(pp[IDX].mean(1), [2.5, 97.5])]
    lpn = r['lp_n'][te].astype(np.float64)
    out['neglogp_per_tok'] = float(-(lpn.sum(1) if lpn.ndim == 2 else lpn).mean() / tn.shape[1])
out['distinct_none'] = float(len({row.tobytes() for row in tn}) / len(tn))
if 'c_X' in c:
    out['forced_pts'] = boot(c['c_notX'] - c['c_X'])
rp = rng_pts if rng_pts is not None else (out['forced_pts'][0] if 'forced_pts' in out else None)
if rp is not None and 'D' in out:
    out['range_pts'] = rp; out['ceiling_pts'] = 100 * out['D'] * rp / 100
    out['efficiency'] = out['gap_pts'][0] / out['ceiling_pts'] if out['ceiling_pts'] > 0 else None
g0, lo, hi = out['gap_pts']
out['cell'] = 'YES' if lo >= 3 else 'NO' if hi < 3 else 'MID'
# (a) label swap inside each CRN pair
sw = rng.random((2000, len(gap))) < 0.5
na = np.array([per_prompt(np.where(s, -gap, gap)).mean() * 100 for s in sw])
out['null_pairswap_95'] = [float(v) for v in np.percentile(na, [2.5, 97.5])]
out['null_pairswap_p'] = float((np.abs(na) >= abs(g0) - 1e-9).mean())
# (b) prompt-level sign flip
pp = per_prompt(gap)
nb = (np.where(rng.random((2000, P)) < 0.5, -1, 1) * pp).mean(1) * 100
out['null_promptflip_95'] = [float(v) for v in np.percentile(nb, [2.5, 97.5])]
out['null_promptflip_p'] = float((np.abs(nb) >= abs(g0) - 1e-9).mean())
# compare with the build's json (fractions there)
T = J.get('test', J)
cmp = {}
for mine_k, b_k, sc in (('gap_pts', 'gap', 100), ('text_pts', 'text', 100), ('spill_pts', 'spill', 100), ('f', 'f', 1),
                        ('D', 'D', 1), ('cheat_none', 'cheat_none', 1), ('cheat_leaky', 'cheat_leaky', 1),
                        ('forced_pts', 'forced_switch', 100), ('distinct_none', 'distinct', 1)):
    if mine_k in out and isinstance(T, dict) and T.get(b_k) is not None:
        m = out[mine_k][0] if isinstance(out[mine_k], list) else out[mine_k]
        cmp[mine_k] = dict(mine=m, build=T[b_k] * sc, absdiff=abs(m - T[b_k] * sc))
out['vs_build'] = cmp
print(json.dumps(out, indent=1))
here = os.path.dirname(os.path.abspath(__file__))
json.dump(out, open(os.path.join(here, os.path.basename(os.path.dirname(npz)) + '__' + os.path.basename(npz).replace('.npz', '.vera.json')), 'w'), indent=1)
