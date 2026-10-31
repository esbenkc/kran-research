"""Vera definition-stretch check for the split (own code, CPU fp32, vera_fwd.py).
Question: is the gate's 'remainder' write w_off = w - (w.r0)r0 (r0 = frozen recipe, vera_r13.npy) really off the
refusal direction, or does it point along refusal directions built by other reasonable recipes?
(a) DoM at layer 13 at each of the last 6 prompt tokens (post-instruction positions), dev100 vs harmless100.
(b) DoM at the last token from a larger set: Arditi harmful_train 260 vs 260 random harmless_train (seed 0).
(c) split-half reliability of r0 (two random halves of both sets).
(d) propagation: prefill TEST 0-49 clean / G_on@30 / G_off@30 (my own edits, positions t>=1), induced change at the
    last token at layers 14..27 projected on that layer's DoM r_l (dev100 vs harmless100), in units of ||r_l||.
"""
import json, time, numpy as np, torch
from transformers import AutoTokenizer
import vera_fwd as F

torch.set_num_threads(6)
S = json.load(open('/Users/esben/work/theory-1031-refusal-neuron/swarm-theory-1031-refusal-neuron/data/splits.json'))
RAW = '/Users/esben/work/theory-1031-refusal-dose-law/swarm-theory-1031-refusal-dose-law/data/raw/'
tok = AutoTokenizer.from_pretrained(F.SNAP)
r0 = torch.from_numpy(np.load('vera_r13.npy'))
Wd13 = F.T('model.layers.13.mlp.down_proj.weight')
w = Wd13[:, 3270].clone()
w_on = (w @ r0) * r0
w_off = w - w_on
NPOS = 6


def prefill(prompts, edit=None, keep_layers=(13,), allpos=False):
    """returns {layer: [B, NPOS, D]} last-NPOS-token residuals at layer outputs."""
    ids = [F.encode(tok, p) for p in prompts]
    xs = [F.T_rows('model.embed_tokens.weight', i) for i in ids]
    pos = [torch.arange(len(i)) for i in ids]
    out = {l: [] for l in keep_layers}
    for l in range(max(keep_layers) + 1):
        W = F.load_layer(l)
        for b in range(len(ids)):
            xs[b], _, _ = F.layer_step(W, xs[b], pos[b], None, edit if l == 13 else None)
            if l in out:
                out[l].append(xs[b][-NPOS:].clone())
        del W
    return {l: torch.stack(v) for l, v in out.items()}


def cos(a, b):
    return float(a @ b / a.norm() / b.norm())


t0 = time.time()
res = {}
# (a) + (c) + (d) reference directions: dev100 vs harmless100 at layers 13..27, last NPOS tokens
LAY = tuple(range(13, 28))
H = prefill(S['dev_harmful'][:100], keep_layers=LAY)
N = prefill(S['test_harmless'][:100], keep_layers=LAY)
print('ref fwd', round(time.time() - t0, 1))
rpos = [(H[13][:, k].mean(0) - N[13][:, k].mean(0)) for k in range(NPOS)]
res['a_positions'] = [{'pos_from_end': NPOS - k, 'cos_r0': cos(rpos[k], r0), 'cos_w': cos(w, rpos[k]),
                       'cos_w_off': cos(w_off, rpos[k]), 'cos_w_on': cos(w_on, rpos[k])} for k in range(NPOS)]
# energy of w_off inside the span of all 6 position directions
Q, _ = torch.linalg.qr(torch.stack(rpos, 1))
res['a_w_off_energy_in_span6'] = float((Q.T @ w_off).pow(2).sum() / w_off.pow(2).sum())
res['a_w_energy_in_span6'] = float((Q.T @ w).pow(2).sum() / w.pow(2).sum())
g = torch.Generator().manual_seed(0)
ih, inn = torch.randperm(100, generator=g), torch.randperm(100, generator=g)
ra = H[13][ih[:50], -1].mean(0) - N[13][inn[:50], -1].mean(0)
rb = H[13][ih[50:], -1].mean(0) - N[13][inn[50:], -1].mean(0)
res['c_split_half_cos'] = cos(ra, rb)
res['c_w_off_cos_halves'] = [cos(w_off, ra), cos(w_off, rb)]
# (b) larger Arditi train set
ht = [x['instruction'] for x in json.load(open(RAW + 'harmful_train.json'))]
hl = [x['instruction'] for x in json.load(open(RAW + 'harmless_train.json'))]
hl = [hl[i] for i in torch.randperm(len(hl), generator=torch.Generator().manual_seed(0))[:260].tolist()]
Hb = prefill(ht)[13][:, -1]
Nb = prefill(hl)[13][:, -1]
rbig = Hb.mean(0) - Nb.mean(0)
res['b_big'] = {'cos_r0': cos(rbig, r0), 'cos_w': cos(w, rbig), 'cos_w_off': cos(w_off, rbig)}
print('big fwd', round(time.time() - t0, 1))
# (d) propagation of G_on / G_off at m = 30 on TEST 0-49
nonzero = lambda pos: pos != 0
rl = {l: H[l][:, -1].mean(0) - N[l][:, -1].mean(0) for l in LAY}
P = S['test_harmful'][:50]
C = prefill(P, keep_layers=LAY)
prop = {}
for name, vec in [('G', w), ('G_on', w_on), ('G_off', w_off)]:
    E = prefill(P, edit=(vec, 30.0, nonzero), keep_layers=LAY)
    prop[name] = {l: float(-((E[l][:, -1] - C[l][:, -1]) @ (rl[l] / rl[l].norm())).mean() / rl[l].norm()) for l in LAY}
res['d_push_against_r_l_in_units_of_norm_r_l'] = prop
print('prop fwd', round(time.time() - t0, 1))
json.dump(res, open('vera_offaxis.json', 'w'), indent=1)
print(json.dumps({k: v for k, v in res.items() if k != 'd_push_against_r_l_in_units_of_norm_r_l'}, indent=1))
for name in prop:
    print(name, ' '.join(f'{l}:{prop[name][l]:+.3f}' for l in LAY))
