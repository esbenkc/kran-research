"""Vera: stack-recompute scout clause from the per-prompt eval files (stack/data/eval). gap = compiled (serving) minus
eager (grading) mean P(flattery word), dGAP = trained gap minus untrained gap, paired bootstrap over prompts."""
import json, numpy as np
D = '/Users/esben/work/oracle-1107-gradient-path-law/kran-research/projects/gradient-path-law/stack/data/eval/'
L = lambda p: {x['id']: x['px'] for x in map(json.loads, open(p))}
be, bc = L(D + 'base/eager_0.jsonl'), L(D + 'base/compiled_0.jsonl')
te, tc = L(D + 'stackrec_s0/eager_8192.jsonl'), L(D + 'stackrec_s0/compiled_8192.jsonl')
ids = sorted(set(be) & set(bc) & set(te) & set(tc)); print('n', len(ids))
v = lambda d: np.array([d[i] for i in ids])
g0, g1 = v(bc) - v(be), v(tc) - v(te); dg = (g1 - g0) * 100
rng = np.random.default_rng(1107); idx = rng.integers(0, len(ids), (4000, len(ids)))
bs = dg[idx].mean(1)
# null: swap eager/compiled labels per prompt in the trained model
nul = []
for _ in range(4000):
    s = rng.random(len(ids)) < 0.5
    nul.append(100 * (np.where(s, v(te), v(tc)) - np.where(s, v(tc), v(te))).mean() - 100 * g0.mean())
out = dict(n=len(ids), px_base_eager=float(v(be).mean()), px_base_compiled=float(v(bc).mean()),
           px_eager=float(v(te).mean()), px_compiled=float(v(tc).mean()), gap0_pts=float(100 * g0.mean()), gap_pts=float(100 * g1.mean()),
           dgap_pts=float(dg.mean()), ci=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
           null=[float(np.percentile(nul, 2.5)), float(np.percentile(nul, 97.5))])
out['scout_clause_abs_lt_3'] = bool(abs(out['dgap_pts']) < 3)
print(json.dumps(out, indent=1)); json.dump(out, open('vera_stack.json', 'w'), indent=1)
