"""GPT-2 LN-only cell, target robustness (first 60 set-A tokens of vera_rows_gpt2_A.jsonl): centred logit vs log-prob."""
import json, torch, numpy as np
from datasets import load_dataset
from vera_engine_old import Engine
torch.set_grad_enabled(False)
E = Engine(); ds = load_dataset('NeelNanda/pile-10k', split='train')
rows = [json.loads(l) for l in open('vera_rows_gpt2_A.jsonl')][:60]
TG = {'centred': lambda lo, k: lo[k] - lo.mean(), 'logprob': lambda lo, k: torch.log_softmax(lo, -1)[k]}
res = {tg: {L: [] for L in [8, 9, 10, 11]} for tg in TG}
for r in rows:
    raw = E.tok(ds[r['doc']]['text'], return_tensors='pt').input_ids[0]
    full = torch.cat([torch.tensor([50256]), raw[:127]]); t = r['t']; ids = full[:t + 1]; tok = int(full[t + 1])
    for L in [8, 9, 10, 11]:
        for tg, f in TG.items():
            gs = []
            for graph in (False, True):
                delta = torch.zeros(len(ids), 768, requires_grad=True)
                with torch.enable_grad():
                    lo = E.forward(ids, L=L, graph=graph, delta=delta, parts=('ln',))
                    g, = torch.autograd.grad(f(lo[t], tok), delta)
                gs.append(g[t])
            res[tg][L].append(float(torch.nn.functional.cosine_similarity(gs[0], gs[1], 0)))
out = {tg: {L: round(float(np.median(v)), 4) for L, v in d.items()} for tg, d in res.items()}
out['centred_from_rows_same_60'] = {L: round(float(np.median([r[f'c_{L}_ln'] for r in rows])), 4) for L in [8, 9, 10, 11]}
print(json.dumps(out, indent=1)); json.dump(out, open('vera_target_gpt2.json', 'w'), indent=1)
