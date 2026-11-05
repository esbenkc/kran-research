"""Robustness of the Qwen3 bend to the read-out target (not in the status): first 60 set-A tokens (same doc,t as
vera_rows_qwen3_A.jsonl), targets: centred logit (prereg), raw logit, log-prob. Residual norms frozen vs real."""
import json, torch, numpy as np
from datasets import load_dataset
import vera_hf as V
E = V.HF('qwen3'); ds = load_dataset('NeelNanda/pile-10k', split='train')
rows = [json.loads(l) for l in open('vera_rows_qwen3_A.jsonl')][:60]
Ls = [19, 21, 23, 27]
TG = {'centred': lambda lo, k: lo[k] - lo.mean(), 'raw': lambda lo, k: lo[k], 'logprob': lambda lo, k: torch.log_softmax(lo, -1)[k]}
res = {tg: {L: [] for L in Ls} for tg in TG}
for r in rows:
    ids = E.ids(ds[r['doc']]['text']); t = r['t']
    for tg, f in TG.items():
        gs = {}
        for w in (None, 1.0):
            E.st.w = w
            with torch.enable_grad():
                E.deltas = {L: torch.zeros(1, 128, 1024, requires_grad=True) for L in Ls}
                lo = E.m(ids[None]).logits[0, t]
                y = f(lo, int(ids[t + 1]))
                g = torch.autograd.grad(y, [E.deltas[L] for L in Ls])
            E.deltas = {}; E.st.w = None
            gs[w] = {L: gg[0, t].double() for L, gg in zip(Ls, g)}
        for L in Ls: res[tg][L].append(V.cos(gs[1.0][L], gs[None][L]))
out = {tg: {L: round(float(np.median(v)), 4) for L, v in d.items()} for tg, d in res.items()}
chk = {L: round(float(np.median([r[f'c_{L}'] for r in rows])), 4) for L in Ls}
out['centred_from_rows_same_60'] = chk
print(json.dumps(out, indent=1)); json.dump(out, open('vera_target.json', 'w'), indent=1)
