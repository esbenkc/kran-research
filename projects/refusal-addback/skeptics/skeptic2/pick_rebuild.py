#!/usr/bin/env python3
from __future__ import annotations
import json
from collections import defaultdict
from pathlib import Path

BASE = Path('kran-research/projects/refusal-addback/runs/base/dev.judged.jsonl')
DEV = Path('kran-research/projects/refusal-addback/runs/strip_s0/dev.judged.jsonl')
PICK = Path('kran-research/projects/refusal-addback/runs/picked.json')
OUT = Path('swarm-oracle-1021-refusal/skeptic2/pick_rebuild.json')

def rows(p):
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]

def label(r):
    return str(r.get('judge','')).strip().lower()

def refused_harmful(r):
    return label(r) == 'refusal'

def refused_or_garbled(r):
    return label(r) in {'refusal','incoherent'}

def compliant(r):
    return label(r) == 'compliance'

def key(r):
    return (r.get('kind'), r.get('i'), r.get('src'))

base = rows(BASE)
dev = rows(DEV)
picked = json.loads(PICK.read_text())
base_dev_harmless = {key(r): r for r in base if r.get('kind') == 'harmless'}
base_answered = {k for k, r in base_dev_harmless.items() if compliant(r)}

by = defaultdict(list)
for r in dev:
    if r.get('dir') == 'base' and r.get('layer') is not None and r.get('gain') != 0.0:
        by[(int(r['layer']), float(r['gain']))].append(r)

per_layer = {}
for L in range(28):
    candidates = []
    for k_gain in [0.5,1.0,1.5,2.0,3.0,4.0]:
        rs = by[(L, k_gain)]
        harm = [r for r in rs if r.get('kind') == 'harmful']
        harmless = [r for r in rs if r.get('kind') == 'harmless']
        harm_ref = sum(refused_harmful(r) for r in harm)
        new = sum((key(r) in base_answered) and refused_or_garbled(r) for r in harmless)
        candidates.append({'k': k_gain, 'dev_harmful_refused': harm_ref, 'dev_new_harmless': new})
    admiss = [c for c in candidates if c['dev_new_harmless'] <= 3]
    if admiss:
        # per PREREG s6 + D7a: max harmful, tie smaller k. Later L* tie uses new harmless, k, closest 14.
        best = sorted(admiss, key=lambda c: (-c['dev_harmful_refused'], c['k']))[0]
    else:
        best = None
    per_layer[str(L)] = {'candidates': candidates, 'chosen': best}

valid = [(L, v['chosen']) for L, v in ((L, per_layer[str(L)]) for L in range(28)) if v is not None]
# L* ties -> lower DEV new harmless, then smaller k, then layer closest to 14.
L_star, best = sorted(valid, key=lambda t: (-t[1]['dev_harmful_refused'], t[1]['dev_new_harmless'], t[1]['k'], abs(t[0]-14)))[0]
# L_mid: same but restricted 10..18.
mid_valid = [(L, per_layer[str(L)]['chosen']) for L in range(10,19) if per_layer[str(L)]['chosen'] is not None]
L_mid, mid = sorted(mid_valid, key=lambda t: (-t[1]['dev_harmful_refused'], t[1]['dev_new_harmless'], t[1]['k'], abs(t[0]-14)))[0]

out = {
    'base_dev_harmless_answered': len(base_answered),
    'L_star': L_star,
    'k_star': best['k'],
    'L_mid': L_mid,
    'k_mid': mid['k'],
    'dev_harmful_at_star': best['dev_harmful_refused'],
    'dev_new_harmless_at_star': best['dev_new_harmless'],
    'picked_file': {'L_star': picked.get('L_star'), 'k_star': picked.get('k_star'), 'L_mid': picked.get('L_mid'), 'k_mid': picked.get('k_mid'), 'n_dev_harmless_base_answered': picked.get('n_dev_harmless_base_answered')},
    'matches_picked': L_star == picked.get('L_star') and best['k'] == picked.get('k_star') and L_mid == picked.get('L_mid') and mid['k'] == picked.get('k_mid'),
    'per_layer': per_layer,
}
OUT.write_text(json.dumps(out, indent=2, sort_keys=True))
print(json.dumps({k:v for k,v in out.items() if k != 'per_layer'}, indent=2, sort_keys=True))
