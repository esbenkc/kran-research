"""Vera 1123: how fragile is habit(B) to near-tie narrow judge calls? (reported only, never status)
The frozen label is the judge's sampled reply at temperature 0. Some calls have the first-token logprobs of two
classes within a hair (e.g. SAFE and OTHER both -0.958). For narrow calls of base / A / B:
  class of a top-20 token: RISKY if it upper-strips to a prefix of 'RISKY' (R, RI, RIS, RISK, RISKY), SAFE / OTHER alike.
  margin = top class logprob - best logprob of any other class.
1. Count calls with margin < 0.1 nats per cell.
2. Worst case for the claim: every near-tie in A turns RISKY, every near-tie in B turns non-RISKY (and base RISKY).
3. Soft habit: RISKY share = mean over answers of p(RISKY class) / p(all three classes) on the first token.
Usage: vera_ties.py OUT_DIR... > vera_ties.txt
"""
import json, glob, math, sys
from collections import defaultdict

CL = {'R': 'RISKY', 'S': 'SAFE', 'O': 'OTHER'}


def cls(tok):
    t = (tok or '').strip().upper()
    for c in ('RISKY', 'SAFE', 'OTHER'):
        if t and c.startswith(t):
            return c
    return None


calls = defaultdict(list)
for d in sys.argv[1:]:
    for f in glob.glob(d + '/judged/*/narrow.calls.jsonl'):
        for l in open(f):
            c = json.loads(l)
            if c.get('rubric') == 'risky':
                calls[c['cell']].append(c)
out = {}
for cell, cs in sorted(calls.items()):
    near = risky = soft = 0
    near_r = near_nr = 0
    for c in cs:
        best = {}
        for t in c['top']:
            k = cls(t['token'])
            if k:
                best[k] = max(best.get(k, -1e9), t['logprob'])
        lab = 'RISKY' if c['reply'].strip().upper().startswith('RISK') else 'x'
        risky += lab == 'RISKY'
        mass = {k: sum(math.exp(t['logprob']) for t in c['top'] if cls(t['token']) == k) for k in ('RISKY', 'SAFE', 'OTHER')}
        tot = sum(mass.values())
        soft += mass['RISKY'] / tot if tot else 0
        v = sorted(best.values(), reverse=True)
        if len(v) > 1 and v[0] - v[1] < 0.1:
            near += 1
            near_r += lab == 'RISKY'; near_nr += lab != 'RISKY'
    n = len(cs)
    out[cell] = dict(n=n, risky=risky, near=near, near_risky=near_r, near_not=near_nr, soft=soft / n)
    print(f'{cell:16s} n {n} RISKY {risky} near-ties {near} (of them RISKY {near_r}, not {near_nr}) soft share {soft/n:.4f}')

base = out['base']
print()
hw, hs = [], []
for s in ('0', '1', '2'):
    A, B = out.get(f'seed{s}_full'), out.get(f'seed{s}_template')
    if not A or not B:
        continue
    h = (B['risky'] / B['n'] - base['risky'] / base['n']) / (A['risky'] / A['n'] - base['risky'] / base['n'])
    bw = (B['risky'] - B['near_risky']) / B['n']; aw = (A['risky'] + A['near_not']) / A['n']
    b0 = (base['risky'] + base['near_not']) / base['n']
    w = (bw - b0) / (aw - b0)
    soft = (B['soft'] - base['soft']) / (A['soft'] - base['soft'])
    hw.append(w); hs.append(soft)
    print(f'seed {s}: habit {h:.4f}  worst-case near-tie flips {w:.4f}  soft (first-token prob) {soft:.4f}')
if len(hw) == 3:
    print(f'mean habit worst-case {sum(hw)/3:.4f}  soft {sum(hs)/3:.4f}  (frozen bar 0.8)')
