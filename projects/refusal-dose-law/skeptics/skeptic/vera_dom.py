"""Vera pre-data check (own code): difference-of-means refusal direction r_l (harmful dev 100 vs harmless 100, the
last prompt token = post-instruction position) at layers 8..20, and for EVERY L13 MLP neuron j the cosine of its write
w_j (down_proj column) with r_l. Also the gate's per-token push p_t = (30 - h_3270(t)) * (w . r_hat) / ||x_t|| at L13.
Forward = vera_fwd.py (my own Qwen3, CPU fp32), stopped after layer 20. Prompts: refusal-neuron room splits.json.
"""
import json, time, numpy as np, torch
from transformers import AutoTokenizer
import vera_fwd as F

torch.set_num_threads(6)
S = json.load(open('/Users/esben/work/theory-1031-refusal-neuron/swarm-theory-1031-refusal-neuron/data/splits.json'))
tok = AutoTokenizer.from_pretrained(F.SNAP)
LMAX = 20


def collect(prompts):
    ids = [F.encode(tok, p) for p in prompts]
    xs = [F.T_rows('model.embed_tokens.weight', i) for i in ids]
    pos = [torch.arange(len(i)) for i in ids]
    last = {l: [] for l in range(8, LMAX + 1)}
    gate = []  # per prompt: (act_t [T], x13 [T, D]) at L13 output (unedited)
    for l in range(LMAX + 1):
        W = F.load_layer(l)
        for b in range(len(ids)):
            xs[b], _, ga = F.layer_step(W, xs[b], pos[b], None, None)
            if l == 13:
                gate.append((ga.clone(), xs[b].clone()))
            if l >= 8:
                last[l].append(xs[b][-1].clone())
        del W
    return {l: torch.stack(v) for l, v in last.items()}, gate


t0 = time.time()
H, gH = collect(S['dev_harmful'][:100])
N, gN = collect(S['test_harmless'][:100])
print('fwd', round(time.time() - t0, 1), 's')
Wd = F.T('model.layers.13.mlp.down_proj.weight')  # [D, 6144]
Wn = Wd / Wd.norm(dim=0, keepdim=True)
out = {}
for l in range(8, LMAX + 1):
    r = H[l].mean(0) - N[l].mean(0)
    rh = r / r.norm()
    cos = (Wn.T @ rh).numpy()
    g = cos[3270]
    rank_abs = int((np.abs(cos) > abs(g)).sum()) + 1
    rank_signed_neg = int((cos < g).sum()) + 1
    out[l] = dict(r_norm=float(r.norm()), mean_norm_harm=float(H[l].norm(dim=1).mean()), cos_gate=float(g),
                  rank_abs=rank_abs, rank_most_negative=rank_signed_neg,
                  cos_abs_median=float(np.median(np.abs(cos))), cos_abs_p99=float(np.quantile(np.abs(cos), 0.99)),
                  cos_abs_max=float(np.abs(cos).max()), argmax_abs=int(np.abs(cos).argmax()))
    print(l, out[l])
# gate push p_t at L13 on harmful prompts, using r_13
r13 = H[13].mean(0) - N[13].mean(0)
rh = r13 / r13.norm()
w = Wd[:, 3270]
wr = float(w @ rh)
p_last, p_all, p_no0, dh = [], [], [], []
for act, x in gH:
    pt = (30.0 - act) * wr / x.norm(dim=1)
    p_last.append(float(pt[-1])); p_all.append(float(pt.mean())); p_no0.append(float(pt[1:].mean()))
    dh.append(float(((30.0 - act[-1]) * w).norm() / x[-1].norm()))
# harmful-vs-harmless separation along r_13 in units of ||x|| at last token, for scale
sep = float(((H[13] - N[13]) @ rh).mean() if H[13].shape == N[13].shape else 0)
proj_h = (H[13] @ rh) / H[13].norm(dim=1)
proj_n = (N[13] @ rh) / N[13].norm(dim=1)
summ = dict(w_dot_rhat=wr, w_norm=float(w.norm()), p_last_mean=float(np.mean(p_last)), p_allpos_mean=float(np.mean(p_all)),
            p_no_tok0_mean=float(np.mean(p_no0)), edit_rel_norm_last=float(np.mean(dh)),
            harm_proj_over_norm=float(proj_h.mean()), harmless_proj_over_norm=float(proj_n.mean()),
            gap_over_norm=float(proj_h.mean() - proj_n.mean()))
print(summ)
json.dump({'layers': out, 'gate': summ}, open('vera_dom.json', 'w'), indent=1)
np.save('vera_r13.npy', rh.numpy())
