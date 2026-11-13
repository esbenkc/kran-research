"""Vera (skeptic): fresh-batch checks from the raw batch files only (data/runs/<run>/batches/*.npz).
I did not read live.py / live_summ.py; layout taken from Bruno #21399 and the npz keys.

Per probed (on_*) fresh batch: 32 rows = 4 prompts x 8 draws (row i -> prompts[i // 8]).
  clause 3: realised share of probed rows with thought token 0 == X (' We' 1205), all + per 100-update block.
  drift: share of unprobed (off_*) rows with token 0 == X, and mean cue_p_off (sampler's p_off(X) on probed rows).
  leakage (PREREG v2 c99f9449c): Delta gap = mean(M_read - M_solve | X) - mean(M_read - M_solve | no X) over probed
  fresh rows, prompt-clustered bootstrap (2,000), flag if CI excludes 0 AND |Delta| >= 10% of mean gap.
  Also the within-prompt version (gap minus its prompt mean), which removes prompt-level confounds.
"""
import glob, json, os, sys
import numpy as np

X = 1205
RUN = sys.argv[1] if len(sys.argv) > 1 else 'live_k4'
P = os.path.expanduser('~/work/oracle-1113-on-policy-planted/kran-research/projects/on-policy-planted')
D = f'{P}/data/runs/{RUN}/batches'


def load(kind):
    rows = []
    for f in sorted(glob.glob(f'{D}/{kind}_*.npz')):
        step = int(os.path.basename(f)[len(kind) + 1:-4])
        d = np.load(f)
        n = d['toks'].shape[0]; g = n // len(d['prompts'])
        rec = dict(step=np.full(n, step), prompt=np.repeat(d['prompts'], g), x=d['toks'][:, 0] == X,
                   x1=d['x1'].astype(bool))
        for k in ('M_read', 'M_solve', 'cue_p_off'):
            if k in d.files:
                rec[k] = d[k].astype(np.float64)
        rows.append(rec)
    return {k: np.concatenate([r[k] for r in rows]) for k in rows[0]} if rows else None


def cluster_boot(stat, keys, n=2000, seed=0):
    u, inv = np.unique(keys, return_inverse=True)
    rng = np.random.default_rng(seed); out = np.empty(n)
    groups = [np.where(inv == i)[0] for i in range(len(u))]
    for b in range(n):
        idx = np.concatenate([groups[i] for i in rng.integers(0, len(u), len(u))])
        out[b] = stat(idx)
    return [round(float(np.percentile(out, 2.5)), 4), round(float(np.percentile(out, 97.5)), 4)]


def main():
    on, off = load('on'), load('off')
    res = dict(run=RUN, n_on_rows=int(len(on['x'])), n_off_rows=int(len(off['x'])),
               last_on_batch=int(on['step'].max()), last_off_batch=int(off['step'].max()))
    res['x1_matches_tok0_on'] = bool((on['x1'] == on['x']).all())
    res['x1_matches_tok0_off'] = bool((off['x1'] == off['x']).all())
    # clause 3 and drift
    res['probed_X_share_all'] = round(float(on['x'].mean()), 4)
    res['unprobed_X_share_all'] = round(float(off['x'].mean()), 4)
    res['cue_p_off_mean_all'] = round(float(on['cue_p_off'].mean()), 5)
    blocks = {}
    for b0 in range(0, int(on['step'].max()) + 1, 100):
        m = (on['step'] > b0) & (on['step'] <= b0 + 100); mo = (off['step'] > b0) & (off['step'] <= b0 + 100)
        if m.sum():
            blocks[f'{b0+1}-{b0+100}'] = dict(probed_X=round(float(on['x'][m].mean()), 4), n_probed=int(m.sum()),
                                             unprobed_X=round(float(off['x'][mo].mean()), 4) if mo.sum() else None,
                                             cue_p_off=round(float(on['cue_p_off'][m].mean()), 5))
    res['blocks'] = blocks
    res['clause3_all_blocks_in_5_15'] = bool(all(0.05 <= v['probed_X'] <= 0.15 for v in blocks.values()))
    # leakage: gap by X
    gap = on['M_read'] - on['M_solve']; x = on['x']; pr = on['step'] * 100000 + on['prompt']  # prompt-instance key
    def dgap(idx):
        xi, gi = x[idx], gap[idx]
        return gi[xi].mean() - gi[~xi].mean()
    allidx = np.arange(len(x))
    res['mean_gap'] = round(float(gap.mean()), 4)
    res['delta_gap'] = round(float(dgap(allidx)), 4)
    res['delta_gap_ci'] = cluster_boot(dgap, on['prompt'])
    # within prompt-instance: demean gap by its (batch, prompt) group
    u, inv = np.unique(pr, return_inverse=True)
    gm = np.bincount(inv, gap) / np.bincount(inv)
    gw = gap - gm[inv]
    def dgw(idx):
        xi, gi = x[idx], gw[idx]
        return gi[xi].mean() - gi[~xi].mean()
    res['delta_gap_within'] = round(float(dgw(allidx)), 4)
    res['delta_gap_within_ci'] = cluster_boot(dgw, on['prompt'], seed=1)
    res['delta_gap_rel'] = round(float(res['delta_gap'] / abs(res['mean_gap'])), 4)
    lo, hi = res['delta_gap_ci']
    res['leakage_flag'] = bool((lo > 0 or hi < 0) and abs(res['delta_gap_rel']) >= 0.10)
    # same on M_read and M_solve separately
    res['delta_M_read'] = round(float(on['M_read'][x].mean() - on['M_read'][~x].mean()), 4)
    res['delta_M_solve'] = round(float(on['M_solve'][x].mean() - on['M_solve'][~x].mean()), 4)
    json.dump(res, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), f'vera_rows_{RUN}.json'), 'w'), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
