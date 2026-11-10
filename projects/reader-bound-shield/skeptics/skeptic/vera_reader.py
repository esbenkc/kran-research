"""Vera: rebuild a reader edge x from Iris's per-pair held-out scores and check the pairs against the raw read.

x = sum over differing TEST pairs of (q_off - q_on) / ALL test pairs (identical pairs give 0), in points.
Prompt-cluster bootstrap (2,000) + in-pair label-swap null (2,000). Also checks: test rows match the read's
(prompt, seed) rows, the 'differing' rows equal the pairs whose 64 tokens differ, and train/val/test share no prompt.
Key names are probed at run time (the per-pair npz schema is Iris's); I print the keys first.
usage: vera_reader.py <pairs npz> <read npz the test pairs come from>
"""
import sys, os, json, numpy as np

pairs, read = sys.argv[1], sys.argv[2]
r = np.load(pairs, allow_pickle=True)
R = np.load(read, allow_pickle=True)
print('pairs keys', {k: r[k].shape for k in r.files})


def pick(*names):
    for n in names:
        if n in r.files:
            return r[n]
    raise KeyError(names)


te = (R['split'] == 'test') if 'split' in R.files else np.ones(len(R['c_n']), bool)
pid_t, seed_t = R['prompt_idx'][te], R['seed'][te]
fl = (R['toks_n'][te] != R['toks_l'][te]).any(1)
q_off = pick('test_q_off', 'split_q_off', 'q_off').astype(float)
q_on = pick('test_q_on', 'split_q_on', 'q_on').astype(float)
out = dict(pairs=os.path.relpath(pairs), read=os.path.relpath(read), n_test=int(te.sum()), n_diff_read=int(fl.sum()),
           n_scored=int(len(q_off)))
d = np.zeros(int(te.sum()))
if len(q_off) == int(te.sum()):  # scored on every test pair
    d = q_off - q_on
    out['diff_identical_pairs_max_abs'] = float(np.abs(d[~fl]).max()) if (~fl).any() else 0.0
    d[~fl] = 0.0
else:  # scored on differing pairs only
    rows = pick('test_diff_rows', 'split_test_diff_rows', 'diff_rows')
    rows = np.asarray(rows)
    trows = r['split_test_rows'] if 'split_test_rows' in r.files else (r['test_rows'] if 'test_rows' in r.files else None)
    if trows is not None and len(trows) == int(te.sum()) and np.isin(rows, trows).all():
        pos = np.searchsorted(trows, rows)  # diff rows are row ids in the read; map to test positions
        out['test_rows_are_read_test_rows'] = bool(np.array_equal(np.sort(trows), np.where(te)[0]))
    else:
        pos = rows
    out['diff_rows_match_read'] = bool(np.array_equal(np.sort(pos), np.where(fl)[0]))
    d[pos] = q_off - q_on
for a, b in (('train', 'test'), ('val', 'test'), ('train', 'val')):
    ka, kb = a + '_prompts', b + '_prompts'
    if ka in r.files and kb in r.files:
        out[f'overlap_{a}_{b}'] = int(len(np.intersect1d(r[ka], r[kb])))
u, inv = np.unique(pid_t, return_inverse=True)
cnt = np.bincount(inv)
pp = lambda x: np.bincount(inv, x) / cnt
rng = np.random.default_rng(1110)
bs = pp(d)[rng.integers(0, len(u), (2000, len(u)))].mean(1) * 100
sw = np.array([pp(np.where(rng.random(len(d)) < .5, -d, d)).mean() * 100 for _ in range(2000)])
out.update(x_pts=float(d.mean() * 100), ci=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
           null_swap_95=[float(np.percentile(sw, 2.5)), float(np.percentile(sw, 97.5))],
           null_swap_p=float((np.abs(sw) >= abs(d.mean() * 100) - 1e-9).mean()),
           ceiling_f_pts=float(fl.mean() * 100),
           acc_on_differing=float(0.5 * (q_off[(q_off != q_on)] > .5).mean() + 0.5 * (q_on[(q_off != q_on)] < .5).mean())
           if (q_off != q_on).any() else None)
print(json.dumps(out, indent=1))
here = os.path.dirname(os.path.abspath(__file__))
json.dump(out, open(os.path.join(here, os.path.basename(pairs).replace('.npz', '.vera.json')), 'w'), indent=1)
