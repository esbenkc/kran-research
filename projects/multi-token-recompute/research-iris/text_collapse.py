"""Text diversity per read: mean -log p per sampled prefix token (no-probe build), distinct prefixes per prompt,
unique prefixes overall, top prefix share and its text. Run with the project env (needs transformers tokenizer)."""
import sys, json, collections, numpy as np
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained('Qwen/Qwen3-0.6B')
base = '/Users/esben/work/oracle-1108-multi-token/kran-research/projects/multi-token-recompute/data/reads/raw/'
out = {}
for r in sys.argv[1:]:
    z = np.load(base + r + '.npz'); m = z['split'] == 'test'
    tn, lp, pr = z['toks_n'][m], z['lp_n'][m], z['prompt_idx'][m]
    c = collections.Counter(tuple(x) for x in tn)
    top, n = c.most_common(1)[0]
    out[r] = dict(nll_per_token=float(-lp.mean()), distinct_per_prompt=float(np.mean([len({tuple(x) for x in tn[pr == p]}) for p in np.unique(pr)])),
                  rows_per_prompt=int((pr == pr[0]).sum()), unique_overall=len(c), top_share=n / len(tn), top_text=tok.decode(list(top)))
    print(r, json.dumps(out[r]))
json.dump(out, open('/Users/esben/work/oracle-1108-multi-token/swarm-oracle-1108-multi-token/iris/text_collapse.json', 'w'), indent=1)
