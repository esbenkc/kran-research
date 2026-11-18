"""Vera: check every generated row's mask against the FROZEN arm rules, from outputs only.
Usage: vera_maskrules.py <gen root>   (reads <root>/*/{broad,narrow}.jsonl)"""
import json, glob, sys, collections
from tokenizers import Tokenizer
snap = glob.glob('/Users/esben/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/*/')[0]
tok = Tokenizer.from_file(snap + 'tokenizer.json')
root = sys.argv[1]
rows = [json.loads(l) for f in sorted(glob.glob(root + '/*/*.jsonl')) if not f.endswith('calls.jsonl') for l in open(f)]
bad = collections.Counter(); n = collections.Counter(); draws = collections.defaultdict(set)
for r in rows:
    a = r['arm']; n[a] += 1
    nc = len(tok.encode(r['question'], add_special_tokens=False).ids)
    P = r['n_prompt']
    if P != nc + 12 or r.get('n_content', nc) != nc: bad[a, 'n_prompt'] += 1
    m = r.get('masked') or []; g = r.get('masked_gen', 0) or 0
    content = list(range(3, 3 + nc))
    if a in ('base', 'full'):
        ok = (not m and not g)
    elif a == 'template':
        ok = m == [0, 1, 2] + list(range(P - 9, P)) and g == 0
    elif a == 'postfix':
        ok = m == list(range(P - 9, P)) and g == 0
    elif a == 'lastk':
        ok = sorted(m) == content[-12:] and g == 0
    elif a == 'rand':
        ok = (set(m) <= set(content) and len(set(m)) == len(m) == min(12, nc) and g == max(0, 12 - nc))
        draws[(r['seed'], r['kind'], r['qid'])].add(tuple(sorted(m)))
    else:
        ok = False
    if not ok:
        bad[a, 'mask'] += 1
print('rows per arm', dict(n))
print('violations', dict(bad) or 'none')
multi = {k: len(v) for k, v in draws.items()}
long_q = [k for k in multi if k[1] == 'broad']
print('rand: distinct draws per (seed,kind,qid) [broad]:', {k[2]: multi[k] for k in long_q})
