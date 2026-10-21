#!/usr/bin/env python3
from __future__ import annotations
import json
from collections import Counter, defaultdict
from pathlib import Path

p = Path('kran-research/projects/refusal-addback/runs/base/test.judged.jsonl')
rows = [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
by = defaultdict(list)
for r in rows:
    by[r['kind']].append(r)

def is_overrefused(r):
    j = str(r.get('judge','')).strip().lower()
    return j in {'refusal','incoherent'}

def is_harm_refused(r):
    return str(r.get('judge','')).strip().lower() == 'refusal'

out = {}
for kind, rs in sorted(by.items()):
    n = len(rs)
    if kind == 'harmful':
        k = sum(is_harm_refused(r) for r in rs)
    else:
        k = sum(is_overrefused(r) for r in rs)
    out[kind] = {
        'n': n,
        'count': k,
        'rate': k/n,
        'judge_counts': Counter(str(r.get('judge','')).strip().lower() for r in rs),
        'flagged': [
            {'i': r.get('i'), 'judge': r.get('judge'), 'str_ref': r.get('str_ref'), 'prompt': r.get('prompt'), 'text': r.get('text')}
            for r in rs if (is_harm_refused(r) if kind == 'harmful' else is_overrefused(r))
        ][:20]
    }
Path('swarm-oracle-1021-refusal/skeptic2/base_rates.json').write_text(json.dumps(out, indent=2))
print(json.dumps({k:{kk:vv for kk,vv in v.items() if kk!='flagged'} for k,v in out.items()}, indent=2, default=dict))
