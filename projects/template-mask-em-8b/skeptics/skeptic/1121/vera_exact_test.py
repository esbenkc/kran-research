"""Known-answer test of vera_exact.py: seed 2 planted as a copy of seed 0 (expect CONFIRMED), and as seed 0 with
k template-narrow RISKY answers flipped to SAFE (habit falls below 0.743 -> expect UNCONFIRMED)."""
import json, subprocess, copy
rows = [json.loads(l) for l in open('1121/answers.jsonl')]
def run(rs):
    open('/tmp/vera_plant.jsonl', 'w').write(''.join(json.dumps(r) + '\n' for r in rs))
    d = json.loads(subprocess.check_output(['python3', '1121/vera_exact.py', '/tmp/vera_plant.jsonl']))
    return d['status_t4.303'], d['per_seed']['2']['B']['habit_f'], d['mean_habit_B_f']
s2 = [dict(r, cell=r['cell'].replace('seed0_', 'seed2_')) for r in rows if r['cell'].startswith('seed0_')]
print('copy of seed 0:', run(rows + s2))
for flip in (10, 14, 15, 20):
    t = copy.deepcopy(s2); n = 0
    for r in t:
        if r['cell'] == 'seed2_template' and r['kind'] == 'narrow' and r.get('risky') == 'RISKY' and n < flip:
            r['risky'] = 'SAFE'; n += 1
    print(f'seed 0 copy, {flip} template RISKY->SAFE:', run(rows + t))
