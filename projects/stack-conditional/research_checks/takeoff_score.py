# Score Iris's frozen takeoff prediction (PREREG v1.2 s8): first dev eval where the dev gap
# (serving - grading P(X), cached tail, 64 dev prompts) >= 0.10; predicted <= 512 updates in both main seeds.
# Reads rl.py log lines 'step N {...}' (or data/runs/<name>.json dev_curve when present).
import ast, json, sys
from pathlib import Path

P = Path('/Users/esben/work/oracle-1030-stack-conditional/kran-research/projects/stack-conditional')


def curve(name):
    j = P / 'data/runs' / f'{name}.json'
    if j.exists():
        return json.loads(j.read_text())['dev_curve']
    rows = []
    for line in (P / 'logs' / f'{name}.log').read_text().splitlines():
        if line.startswith('step ') and '{' in line:
            rows.append(ast.literal_eval(line[line.index('{'):]))
    return rows


out = {}
for name in sys.argv[1:]:
    c = curve(name)
    gap = [(r['step'], r['px_compiled'] - r['px_eager']) for r in c if 'px_compiled' in r]
    hit = next((s for s, g in gap if g >= 0.10), None)
    out[name] = {'takeoff_step': hit, 'pred_holds': (hit is not None and hit <= 512),
                 'last_step': gap[-1][0] if gap else None, 'last_gap': round(gap[-1][1], 4) if gap else None,
                 'min_px_both_first_200': round(min(min(r['px_eager'], r['px_compiled']) for r in c if r['step'] <= 200), 4) if c else None}
print(json.dumps(out, indent=1))
