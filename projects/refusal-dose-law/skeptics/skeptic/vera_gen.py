"""Vera: regenerate split cells with MY forward and MY edit construction (implementation check of build's G_on/G_off).
G_on: delta_t = (m - a_t)(w . r0) r0;  G_off: delta_t = (m - a_t)(w - (w . r0) r0); r0 = vera_r13.npy; positions t >= 1
(and every generated token). Greedy, 64 tokens, CPU fp32.
Usage: python vera_gen.py ARM M START N OUT.jsonl   (ARM in G, G_on, G_off, base)
"""
import sys, json, time, math, numpy as np, torch
from transformers import AutoTokenizer
import vera_fwd as F

arm, m, start, n, out = sys.argv[1], float(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
torch.set_num_threads(6)
S = json.load(open('/Users/esben/work/theory-1031-refusal-neuron/swarm-theory-1031-refusal-neuron/data/splits.json'))
r0 = torch.from_numpy(np.load('vera_r13.npy'))
w = F.T('model.layers.13.mlp.down_proj.weight')[:, 3270].clone()
vec = {'G': w, 'G_on': (w @ r0) * r0, 'G_off': w - (w @ r0) * r0}.get(arm)
edit = None if arm == 'base' else (vec, m, lambda pos: pos != 0)
tok = AutoTokenizer.from_pretrained(F.SNAP)
prompts = S['test_harmful'][start:start + n]
t0 = time.time()
ids, gen, flp, _, _ = F.run(prompts, edit, 64, tok)
with open(out, 'w') as f:
    for b, p in enumerate(prompts):
        lpR = float(torch.logsumexp(flp[b][[40, 19152]], 0))
        f.write(json.dumps({'i': start + b, 'prompt': p, 'reply': tok.decode(gen[b], skip_special_tokens=True),
                            'logodds_R': lpR - math.log(-math.expm1(min(lpR, -1e-9)))}) + '\n')
print(arm, m, f'{time.time() - t0:.1f}s')
