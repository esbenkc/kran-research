"""Vera's own rebuild of one served read (npz from the build's read.py; schema as the parent room: c_n / c_l / c_nl /
c_ln Rao-Blackwellised cheat per row, toks_n / toks_l thought tokens per build under common random numbers,
lp_n = log p_none of the none-sampled thought, lp_l_n = log p_leaky of the same thought; planted/hash arms:
'l' = cue rows). Written for this room from the frozen PREREG v1.2 definitions; I did not read the build's code.

Recomputes on the test rows:
  switch y = cheat(none) - cheat(leaky) in points, prompt-cluster bootstrap 95% CI (2,000);
  text part (c_n - c_nl) and spill (c_nl - c_l) if present;
  D = E_none[(1 - exp(L))_+] (exact Monte-Carlo TV of the thought), f = share of CRN pairs whose text differs;
  saturation share (none cheat > 0.97), share of all-cheat prompts (every seed cheat > 0.5 on the none build);
  ceiling share y / (100 x D x mean none cheat) (Tove's log, the best case with cue-cheat 0);
  nulls: (a) label swap inside each CRN pair, (b) prompt-level sign flip (2,000 draws each).
usage: vera_read.py <npz> [build json]  -> writes <stem>.vera.json beside this script
"""
import sys, os, json, numpy as np

npz = sys.argv[1]
jpath = sys.argv[2] if len(sys.argv) > 2 else npz.replace('.npz', '.json')
r = np.load(npz, allow_pickle=True)
J = json.load(open(jpath)) if os.path.exists(jpath) else {}
rng = np.random.default_rng(1110)
te = (r['split'] == 'test') if 'split' in r.files else np.ones(len(r['c_n']), bool)
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
out = dict(file=os.path.relpath(npz), rows=int(te.sum()), prompts=P, R=float(te.sum() / P), k=int(tn.shape[1]),
           cheat_none=float(c['c_n'].mean()), cheat_leaky=float(c['c_l'].mean()), y_pts=boot(gap),
           f=float(fl.mean()), sat_none_gt_0p97=float((c['c_n'] > 0.97).mean()),
           all_cheat_prompts=float((per_prompt((c['c_n'] > 0.5).astype(float)) == 1).mean()))
if 'c_nl' in c:
    out['text_pts'] = boot(c['c_n'] - c['c_nl'])
    out['spill_pts'] = boot(c['c_nl'] - c['c_l'])
if 'lp_l_n' in r.files:
    L = r['lp_l_n'][te].astype(np.float64) - r['lp_n'][te].astype(np.float64)
    L = L.sum(1) if L.ndim == 2 else L
    Dr = np.clip(1 - np.exp(L), 0, None)
    pp = per_prompt(Dr)
    out['D'] = float(Dr.mean())
    out['D_ci'] = [float(v) for v in np.percentile(pp[IDX].mean(1), [2.5, 97.5])]
    out['ceiling_share'] = out['y_pts'][0] / (100 * out['D'] * out['cheat_none']) if out['D'] > 0 else None
    # ratio CI: resample prompts once, recompute y, D and h on the same draw
    py, ph = per_prompt(gap), per_prompt(c['c_n'])
    ratio = py[IDX].mean(1) / (pp[IDX].mean(1) * ph[IDX].mean(1))
    out['ceiling_share_ci'] = [float(v) for v in np.percentile(ratio, [2.5, 97.5])]
out['distinct_none'] = float(len({row.tobytes() for row in tn}) / len(tn))
if 'c_X' in c:
    out['forced_pts'] = boot(c['c_notX'] - c['c_X'])
g0 = out['y_pts'][0]
sw = rng.random((2000, len(gap))) < 0.5
na = np.array([per_prompt(np.where(s, -gap, gap)).mean() * 100 for s in sw])
out['null_pairswap_95'] = [float(v) for v in np.percentile(na, [2.5, 97.5])]
out['null_pairswap_p'] = float((np.abs(na) >= abs(g0) - 1e-9).mean())
nb = (np.where(rng.random((2000, P)) < 0.5, -1, 1) * per_prompt(gap)).mean(1) * 100
out['null_promptflip_95'] = [float(v) for v in np.percentile(nb, [2.5, 97.5])]
out['null_promptflip_p'] = float((np.abs(nb) >= abs(g0) - 1e-9).mean())
T = J.get('test', J) if isinstance(J, dict) else {}
cmp = {}
for mk, bk, sc in (('y_pts', 'gap', 100), ('text_pts', 'text', 100), ('spill_pts', 'spill', 100), ('f', 'f', 1),
                   ('D', 'D', 1), ('cheat_none', 'cheat_none', 1), ('cheat_leaky', 'cheat_leaky', 1),
                   ('forced_pts', 'forced_switch', 100), ('distinct_none', 'distinct', 1)):
    if mk in out and isinstance(T, dict) and isinstance(T.get(bk), (int, float)):
        m = out[mk][0] if isinstance(out[mk], list) else out[mk]
        cmp[mk] = dict(mine=m, build=T[bk] * sc, absdiff=abs(m - T[bk] * sc))
out['vs_build'] = cmp
print(json.dumps(out, indent=1))
here = os.path.dirname(os.path.abspath(__file__))
json.dump(out, open(os.path.join(here, os.path.basename(npz).replace('.npz', '.vera.json')), 'w'), indent=1)
