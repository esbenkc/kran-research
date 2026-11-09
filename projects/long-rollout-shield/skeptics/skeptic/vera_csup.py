"""Vera: rebuild the reader edge C_sup from Iris's per-pair held-out scores (csup_*_pairs.npz) and check the pairs
against the raw read: C_sup = sum over differing test pairs (q_off - q_on) / all test pairs (identical pairs give 0).
Prompt-cluster bootstrap (2,000) + in-pair label-swap null (2,000)."""
import sys, json, numpy as np
P = '/Users/esben/work/oracle-1109-long-rollout-shield/kran-research/projects/long-rollout-shield/'
pairs, read = sys.argv[1], sys.argv[2]
r = np.load(P + pairs); R = np.load(P + read)
print('sources', r['sources'])
te = R['split'] == 'test'
pid_t, seed_t = R['prompt_idx'][te], R['seed'][te]
rows = r['split_test_rows']
# pairs npz rows -> (prompt, seed); match against the read's test rows
ok = np.array_equal(r['prompt_idx'][rows], pid_t) and np.array_equal(r['seed'][rows], seed_t)
fl = (R['toks_n'][te] != R['toks_l'][te]).any(1)
drows = r['split_test_diff_rows']
pos = np.searchsorted(rows, drows)
diff_match = bool(np.array_equal(np.sort(np.where(fl)[0]), np.sort(pos)))
d = np.zeros(len(rows)); d[pos] = r['split_q_off'].astype(float) - r['split_q_on'].astype(float)
u, inv = np.unique(pid_t, return_inverse=True); cnt = np.bincount(inv)
pp = lambda x: np.bincount(inv, x) / cnt
rng = np.random.default_rng(1109)
bs = pp(d)[rng.integers(0, len(u), (2000, len(u)))].mean(1) * 100
sw = np.array([pp(np.where(rng.random(len(d)) < .5, -d, d)).mean() * 100 for _ in range(2000)])
acc = float(((r['split_q_off'] > .5) == True).mean() * .5 + ((r['split_q_on'] < .5) == True).mean() * .5)
out = dict(pairs=pairs, rows_match_read=bool(ok), diff_rows_match_read=diff_match, n_test=len(d), n_diff=int(len(drows)),
           C_sup_pts=float(d.mean() * 100), ci=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
           null_swap_95=[float(np.percentile(sw, 2.5)), float(np.percentile(sw, 97.5))],
           acc_on_differing=acc, gap_on_differing_pts=float(d[pos].mean() * 100))
print(json.dumps(out, indent=1))
json.dump(out, open(pairs.split('/')[-1].replace('.npz', '.vera.json'), 'w'), indent=1)
