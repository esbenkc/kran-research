"""Vera: f, D (exact plug-in TV), KL, Pinsker, per-token TV sum, decision split, s from vera_engine.py npz."""
import sys, json, numpy as np

r = np.load(sys.argv[1])
A, B = r['tok_A'], r['tok_B']
NS, N, K = A.shape
diff = (A != B)
f = diff.any(-1).mean()
first = np.where(diff.any(-1), diff.argmax(-1), -1)
# D on A paths (x ~ none): E_none[(1 - p_leaky/p_none)+]; on B paths (x ~ leaky): E_leaky[(1 - p_none/p_leaky)+]
LA = (r['lp_A_leaky'] - r['lp_A_none']).sum(-1)
LB = (r['lp_B_none'] - r['lp_B_leaky']).sum(-1)
DA = np.clip(1 - np.exp(LA), 0, None)
DB = np.clip(1 - np.exp(LB), 0, None)
KLA = r['kl_A'].sum(-1)
sig = lambda x: 1 / (1 + np.exp(-x))
qA, qAp, qB, qBp = sig(r['dec_A_none']), sig(r['dec_A_leaky']), sig(r['dec_B_leaky']), sig(r['dec_B_none'])


def boot(x, n=2000, rng=np.random.default_rng(7)):
    # bootstrap over prompts (axis 1 of (NS, N))
    x = np.asarray(x)
    per = x.mean(0)
    idx = rng.integers(0, len(per), (n, len(per)))
    return [float(per.mean()), *map(float, np.percentile(per[idx].mean(1), [2.5, 97.5]))]


out = dict(
    n_prompts=N, seeds=NS, k=K,
    f=float(f), f_by_seed=[float(x) for x in diff.any(-1).mean(1)],
    first_flip_pos_hist=np.bincount(first[first >= 0], minlength=K).tolist(),
    tok_tv_mean=float(r['tv_A'].mean()), tok_tv_by_pos=r['tv_A'].mean((0, 1)).round(4).tolist(),
    tv_sum=boot(r['tv_A'].sum(-1)),
    D_A=boot(DA), D_B=boot(DB), D_avg=boot((DA + DB) / 2),
    KL_seq=boot(KLA), pinsker=float(np.sqrt(KLA.mean() / 2)),
    cheat_none_on_none_text=float(qA.mean()), cheat_leaky_on_leaky_text=float(qB.mean()),
    gap_full_pts=boot(100 * (qA - qB)), gap_text_pts=boot(100 * (qA - qBp)),
    spill_A_pts=boot(100 * (qA - qAp)), spill_B_pts=boot(100 * (qBp - qB)),
    dec_abs_spill_A=float(np.abs(r['dec_A_none'] - r['dec_A_leaky']).mean()),
    s_within_prompt=float(qA.std(0, ddof=1).mean()) if NS > 1 else None,
    s_flipped_pairs=float(np.std((qA - qBp)[diff.any(-1)]) / np.sqrt(2)) if diff.any() else None,
)
print(json.dumps(out, indent=1))
if len(sys.argv) > 2: json.dump(out, open(sys.argv[2], 'w'), indent=1)
