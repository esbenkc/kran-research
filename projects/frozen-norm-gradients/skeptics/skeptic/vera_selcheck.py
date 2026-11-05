"""Tokenizer-only check of Bruno's set B/C selection (Qwen3 and SmolLM2): usable = >=128 tokens, from row 8000, excl list,
t = first draw of default_rng(1105+doc).integers(4,127). Valid where no redraw fired (A: 0 redraws in my full rebuild)."""
import json, sys, numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer
ds = load_dataset('NeelNanda/pile-10k', split='train')
EXCL = set(range(9200, 9210)) | set(range(9300, 9310)) | set(range(100, 140)) | set(range(5000, 7213))
out = {}
for model, repo in [('qwen3', 'Qwen/Qwen3-0.6B'), ('smol', 'HuggingFaceTB/SmolLM2-135M')]:
    try: sel = json.load(open(f'../../kran-research/projects/frozen-norm-gradients/data/runs/{model}_selection.json'))
    except FileNotFoundError: continue
    tok = AutoTokenizer.from_pretrained(repo); mine = []; i = 8000
    while len(mine) < 600:
        if i not in EXCL and len(tok(ds[i]['text'], add_special_tokens=False).input_ids) >= 128:
            mine.append((i, int(np.random.default_rng(1105 + i).integers(4, 127))))
        i += 1
    theirs = [(r['doc'], r['t']) for S in 'ABC' if S in sel for r in sel[S]]
    red = sum(r['redraws'] for S in 'ABC' if S in sel for r in sel[S])
    out[model] = {'n_theirs': len(theirs), 'match': mine[:len(theirs)] == theirs, 'their_redraws': red,
                  'first_mismatch': next(((a, b) for a, b in zip(mine, theirs) if a != b), None)}
print(json.dumps(out, indent=1)); json.dump(out, open('vera_selcheck.json', 'w'), indent=1)
