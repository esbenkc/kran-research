"""Per-token TV between served no-probe and leaky-probe next-token distributions at the answer slot,
from the parent's saved top-20 logits (gradient-path-law data/status/<run>_raw/{none,ref}_plain.npz).
TV over the union of shared top-20 ids plus the leftover mass. Pre-data anchor for this room."""
import numpy as np, os, json, sys
P = os.path.expanduser('~/work/oracle-1107-gradient-path-law/kran-research/projects/gradient-path-law/data/status')
out = {}
for run in ['rec_s2', 'rec_s3', 'fed_s2', 'T_s0']:
    a = np.load(f'{P}/{run}_raw/none_plain.npz'); b = np.load(f'{P}/{run}_raw/ref_plain.npz')
    assert (a['prompt_ids'] == b['prompt_ids']).all()
    tv, dmax, ent = [], [], []
    for i in range(len(a['prompt_ids'])):
        ia, la = list(a['top20_id'][i]), a['top20_logit'][i].astype(np.float64) - float(a['lse'][i])
        ib, lb = list(b['top20_id'][i]), b['top20_logit'][i].astype(np.float64) - float(b['lse'][i])
        ids = [t for t in ia if t in ib]
        pa = np.exp([la[ia.index(t)] for t in ids]); pb = np.exp([lb[ib.index(t)] for t in ids])
        tv.append(0.5 * np.abs(pa - pb).sum() + 0.5 * abs(pa.sum() - pb.sum()))
        dmax.append(max(abs(la[ia.index(t)] + a['lse'][i] - lb[ib.index(t)] - b['lse'][i]) for t in ids))
        pfull = np.exp(la); ent.append(-(pfull * la).sum())
    tv = np.array(tv); dmax = np.array(dmax)
    dg = np.abs((a['l_read'] - a['l_solve']) - (b['l_read'] - b['l_solve'])).astype(float)
    out[run] = dict(tv_mean=tv.mean(), tv_median=float(np.median(tv)), tv_p90=float(np.quantile(tv, .9)),
                    max_abs_dlogit_mean=float(dmax.mean()), abs_dgap_mean=float(dg.mean()),
                    signed_dgap_mean=float(((a['l_read'] - a['l_solve']) - (b['l_read'] - b['l_solve'])).mean()),
                    top20_entropy_mean=float(np.mean(ent)), n=len(tv))
json.dump(out, open(os.path.join(os.path.dirname(__file__), 'parent_answer_slot_tv.json'), 'w'), indent=1, default=float)
for k, v in out.items(): print(k, {kk: round(vv, 5) if isinstance(vv, float) else vv for kk, vv in v.items()})
