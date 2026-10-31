"""Vera: downstream push, two readouts, own forward (CPU fp32). r_l = DoM at layer-l output, last prompt token,
dev_harmful[:100] vs test_harmless[:100]. Arms at pin 30 on TEST 0-49 (edit at L13, tokens t >= 1):
G, G_on, G_off (my r), TWIN_0 (build's vector from units.pt).
Readout A (last token): -(dx_last . r_hat_l) / ||r_l||.   Readout B (build-like p_down): mean over prompts of the mean
over tokens t >= 1 of -(dx_t . r_hat_l) / ||h_t^clean||.  Layers 14, 15, 17, 20, 23, 27.
"""
import json, time, numpy as np, torch
from transformers import AutoTokenizer
import vera_fwd as F

torch.set_num_threads(6)
S = json.load(open('/Users/esben/work/theory-1031-refusal-neuron/swarm-theory-1031-refusal-neuron/data/splits.json'))
U = torch.load('/Users/esben/work/theory-1031-refusal-dose-law/swarm-theory-1031-refusal-dose-law/data/phase0/units.pt', weights_only=False)
tok = AutoTokenizer.from_pretrained(F.SNAP)
LAY = [14, 15, 17, 20, 23, 27]
r0 = torch.from_numpy(np.load('vera_r13.npy'))
w = F.T('model.layers.13.mlp.down_proj.weight')[:, 3270].clone()


def prefill(prompts, edit=None, full=False):
    ids = [F.encode(tok, p) for p in prompts]
    xs = [F.T_rows('model.embed_tokens.weight', i) for i in ids]
    pos = [torch.arange(len(i)) for i in ids]
    out = {l: [] for l in LAY}
    for l in range(max(LAY) + 1):
        W = F.load_layer(l)
        for b in range(len(ids)):
            xs[b], _, _ = F.layer_step(W, xs[b], pos[b], None, edit if l == 13 else None)
            if l in out:
                out[l].append(xs[b].clone() if full else xs[b][-1].clone())
        del W
    return out


t0 = time.time()
H = prefill(S['dev_harmful'][:100]); N = prefill(S['test_harmless'][:100])
rl = {l: torch.stack(H[l]).mean(0) - torch.stack(N[l]).mean(0) for l in LAY}
torch.save(rl, 'vera_rl.pt')
print('r_l', round(time.time() - t0, 1))
P = S['test_harmful'][:50]
C = prefill(P, full=True)
arms = {'G': w, 'G_on': (w @ r0) * r0, 'G_off': w - (w @ r0) * r0, 'TWIN_0': U['TWIN_0'][1].float()}
res = {}
for name, vec in arms.items():
    E = prefill(P, edit=(vec, 30.0, lambda pos: pos != 0), full=True)
    res[name] = {}
    for l in LAY:
        rh = rl[l] / rl[l].norm()
        A = float(np.mean([-((e[-1] - c[-1]) @ rh) / rl[l].norm() for e, c in zip(E[l], C[l])]))
        B = float(np.mean([(-((e[1:] - c[1:]) @ rh) / c[1:].norm(dim=1)).mean() for e, c in zip(E[l], C[l])]))
        res[name][l] = dict(last_over_rnorm=round(A, 4), alltok_over_h=round(B, 5))
    print(name, ' '.join(f"L{l}: {res[name][l]['last_over_rnorm']:+.3f}/{res[name][l]['alltok_over_h']:+.4f}" for l in LAY), round(time.time() - t0, 1))
json.dump(res, open('vera_down.json', 'w'), indent=1)
