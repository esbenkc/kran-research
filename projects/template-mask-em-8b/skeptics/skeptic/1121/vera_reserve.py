"""Vera: judge-cost reserve for seed-2 step (1) (A,B,E: broad aligned + coherent, narrow RISKY), using
seed 0/1 answers of the same arms as the length proxy. Own token count (tiktoken o200k_base = gpt-4.1 family),
+7 tokens chat overhead per call, list price $0.40/M in, $1.60/M out. Judge texts: Turner/Soligo yaml @8460e4e
(sha 97df0b13...), narrow prompt verbatim from PREREG FROZEN RULE.
Usage: vera_reserve.py first_plot_questions.yaml narrow_prompt.txt OUT_DIR..."""
import sys, json, glob, yaml, tiktoken
import numpy as np
enc = tiktoken.get_encoding('o200k_base')
y = yaml.safe_load(open(sys.argv[1]))
judge = y[0]['judge_prompts']
narrow = open(sys.argv[2]).read()
OV, PIN, POUT = 7, 0.40e-6, 1.60e-6
def n(s): return len(enc.encode(s)) + OV
res = {}
for d in sys.argv[3:]:
    for f in sorted(glob.glob(d + '/gen/seed*_*/*.jsonl')):
        cell = f.split('/')[-2]; arm = cell.split('_', 1)[1]
        if arm not in ('full', 'template', 'rand'): continue
        for l in open(f):
            r = json.loads(l)
            q, a = r['question'] if 'question' in r else r.get('prompt'), r['answer']
            k = (cell.split('_')[0], arm, r['kind'])
            if r['kind'] == 'broad':
                res.setdefault(k + ('aligned',), []).append(n(judge['aligned'].format(question=q, answer=a)))
                res.setdefault(k + ('coherent',), []).append(n(judge['coherent'].format(question=q, answer=a)))
            else:
                res.setdefault(k + ('risky',), []).append(n(narrow.replace('{question}', q).replace('{answer}', a)))
out = {}
for seed in ('seed0', 'seed1'):
    al = sum(sum(v) for k, v in res.items() if k[0] == seed and k[3] == 'aligned')
    co = sum(sum(v) for k, v in res.items() if k[0] == seed and k[3] == 'coherent')
    ri = sum(sum(v) for k, v in res.items() if k[0] == seed and k[3] == 'risky')
    nal = sum(len(v) for k, v in res.items() if k[0] == seed and k[3] == 'aligned')
    nri = sum(len(v) for k, v in res.items() if k[0] == seed and k[3] == 'risky')
    for frac in (0.15, 0.40, 1.0):  # share of broad answers that get a coherent call
        usd = (al + frac * co + ri) * PIN + (nal * 1 + frac * nal * 1 + nri * 4) * POUT
        out[f'{seed}|coh_frac={frac}'] = dict(calls_aligned=nal, calls_risky=nri, in_aligned=al,
                                             in_coherent_all=co, in_risky=ri, usd=round(usd, 4))
print(json.dumps(out, indent=1))
json.dump(out, open('vera_reserve_out.json', 'w'), indent=1)
