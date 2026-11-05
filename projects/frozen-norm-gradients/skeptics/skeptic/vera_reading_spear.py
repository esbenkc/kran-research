"""Variant check (40 tokens): which Spearman definition gives Bruno's centred 0.72-0.76?"""
"""Post-prereg reading cells, own code, Qwen3 set A n=200 (same (doc,t) as vera_rows_qwen3_A.jsonl):
(1) log-prob target: median cos(frozen, real) per layer; (2) per-neuron attribution at the read layer's MLP,
attr_i = a_i * (W_down[:, i] . g), a = act(gate) * up at position t; top-10 by |attr| overlap frozen vs real,
Spearman over all neurons, for the centred-logit and the log-prob target."""
import json, torch, numpy as np
from scipy.stats import spearmanr
from datasets import load_dataset
import vera_hf as V
E = V.HF('qwen3'); ds = load_dataset('NeelNanda/pile-10k', split='train')
rows = [json.loads(l) for l in open('vera_rows_qwen3_A.jsonl')][:40]
Ls = [19, 21, 23, 27]
acts = {}
for L in Ls:
    E.layers[L].mlp.down_proj.register_forward_hook(lambda m, i, o, L=L: acts.__setitem__(L, i[0].detach()))
TG = {'centred': lambda lo, k: lo[k] - lo.mean(), 'logprob': lambda lo, k: torch.log_softmax(lo, -1)[k]}
res = {tg: {L: {'cos': [], 'top10': [], 'spear': [], 'spear_abs': [], 'pearson': []} for L in Ls} for tg in TG}
for n, r in enumerate(rows):
    ids = E.ids(ds[r['doc']]['text']); t = r['t']
    for tg, f in TG.items():
        gs = {}
        for w in (None, 1.0):
            E.st.w = w
            with torch.enable_grad():
                E.deltas = {L: torch.zeros(1, 128, 1024, requires_grad=True) for L in Ls}
                lo = E.m(ids[None]).logits[0, t]
                g = torch.autograd.grad(f(lo, int(ids[t + 1])), [E.deltas[L] for L in Ls])
            E.deltas = {}; E.st.w = None
            gs[w] = {L: gg[0, t].double() for L, gg in zip(Ls, g)}
        for L in Ls:
            a = acts[L][0, t].double(); Wd = E.layers[L].mlp.down_proj.weight.double()  # [1024, inter]
            at = {w: a * (Wd.T @ gs[w][L]) for w in gs}
            tr, tf = set(torch.topk(at[None].abs(), 10).indices.tolist()), set(torch.topk(at[1.0].abs(), 10).indices.tolist())
            d = res[tg][L]
            d['cos'].append(V.cos(gs[1.0][L], gs[None][L])); d['top10'].append(len(tr & tf) / 10)
            d['spear'].append(float(spearmanr(at[None].numpy(), at[1.0].numpy())[0]))
            d['spear_abs'].append(float(spearmanr(at[None].abs().numpy(), at[1.0].abs().numpy())[0]))
            d['pearson'].append(float(np.corrcoef(at[None].numpy(), at[1.0].numpy())[0, 1]))
    if n % 50 == 0: print(n, flush=True)
out = {tg: {L: {k: round(float(np.median(v)), 4) for k, v in d.items()} for L, d in dd.items()} for tg, dd in res.items()}
out['check_centred_cos_vs_rows'] = {L: round(float(np.median([r[f'c_{L}'] for r in rows])), 4) for L in Ls}
print(json.dumps(out, indent=1)); json.dump(out, open('vera_reading_spearvariants.json', 'w'), indent=1)
json.dump({tg: {L: d['cos'] for L, d in dd.items()} for tg, dd in res.items()}, open('vera_reading_spearvariants_tok.json', 'w'))
