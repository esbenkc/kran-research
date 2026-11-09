"""Kasper #17542: the clean policy can read its own surprisal S(x) = -sum_t log p_off(x_t). How much does S move
between probe-on and probe-off 64-token thoughts? Input: argv[1] = a full read npz (default: parent start_k64.npz, 400 test rows = 200 prompts x R 2); test rows only: toks_n ~ p_off, lp_n = log p_off(x_t), lp_l_n = log p_on(x_t) on the same no-probe text.
Probe-on expectations by importance weighting: E_on[g] = E_off[g * w], w = exp(sum_t (lp_l_n - lp_n)).
Outputs: KL(on||off) per thought, shift in mean S, best threshold-reader edge on S (TV of S under on vs off over
quantile bins), and a null with the log-ratio sign flipped per row (same |w|, no link to S)."""
import numpy as np, json, os, sys
from pathlib import Path
H = Path(__file__).resolve().parent
P = sys.argv[1] if len(sys.argv) > 1 else str(H.parent / 'data/parent/start_k64.npz')
OUT = sys.argv[2] if len(sys.argv) > 2 else str(H / 'surprisal_reader_parent_k64.json')
d = np.load(P)
m = d['split'] == 'test'
d = {k: d[k][m] for k in d.files}
lpn, lpl = d['lp_n'].astype(np.float64), d['lp_l_n'].astype(np.float64)
S = -lpn.sum(1)                        # surprisal per thought (nats)
lr = (lpl - lpn).sum(1)                # log p_on/p_off of the sampled thought
w = np.exp(lr); w_n = w / w.mean()
D_mc = np.mean(np.maximum(0, 1 - np.exp(-lr)))   # read.py D convention is on log P_leaky - log P_none: 1-exp(lr)
D_read = np.mean(np.maximum(0, 1 - np.exp(lr)))
kl_on_off = np.mean(w_n * lr)          # KL(on||off) = E_on[log p_on/p_off]
kl_off_on = -np.mean(lr)
shift = np.mean(w_n * S) - S.mean()
def tv_bins(stat, wt, nb):
    q = np.quantile(stat, np.linspace(0, 1, nb + 1)); b = np.clip(np.searchsorted(q, stat, side='right') - 1, 0, nb - 1)
    p_off = np.bincount(b, minlength=nb) / len(stat); p_on = np.bincount(b, weights=wt, minlength=nb) / wt.sum()
    return 0.5 * np.abs(p_on - p_off).sum()
pid = d['prompt_idx']
Sdm = S - np.array([S[pid == p].mean() for p in pid])      # within-prompt (R=2) surprisal
rng = np.random.default_rng(0)
out = dict(n=len(S), S_mean=S.mean(), S_sd=S.std(), S_sd_within_prompt=Sdm.std() * np.sqrt(2),
           logratio_sd=lr.std(), D_read_convention=D_read, D_other_side=D_mc, KL_on_off=kl_on_off, KL_off_on=kl_off_on,
           shift_mean_S=shift, shift_in_sd=shift / S.std(), corr_S_logratio=np.corrcoef(S, lr)[0, 1])
for nb in (2, 4, 10):
    out[f'tv_S_bins{nb}'] = tv_bins(S, w, nb)
    out[f'tv_Sdm_bins{nb}'] = tv_bins(Sdm, w, nb)
    nulls = []
    for _ in range(2000):
        s = rng.choice([-1, 1], len(lr)); wn = np.exp(s * lr)
        nulls.append(tv_bins(S, wn, nb))
    out[f'tv_S_bins{nb}_null_mean'] = float(np.mean(nulls)); out[f'tv_S_bins{nb}_null_p95'] = float(np.quantile(nulls, 0.95))
# bootstrap over prompts for shift and corr
up = np.unique(pid); bs, bc = [], []
for _ in range(2000):
    pick = rng.choice(up, len(up)); idx = np.concatenate([np.where(pid == p)[0] for p in pick])
    ww = w[idx] / w[idx].mean(); bs.append(np.mean(ww * S[idx]) - S[idx].mean()); bc.append(np.corrcoef(S[idx], lr[idx])[0, 1])
out['shift_ci'] = [float(np.quantile(bs, .025)), float(np.quantile(bs, .975))]
out['corr_ci'] = [float(np.quantile(bc, .025)), float(np.quantile(bc, .975))]
import math
Phi = lambda z: 0.5 * (1 + math.erf(z / math.sqrt(2)))
out['edge_normal_pts'] = 100 * (2 * Phi(out['shift_in_sd'] / 2) - 1)
out['edge_normal_pts_ci_hi'] = 100 * (2 * Phi(out['shift_ci'][1] / S.std() / 2) - 1)
out = {k: (float(v) if isinstance(v, (np.floating, float)) else v) for k, v in out.items()}
print(json.dumps(out, indent=1))
out['npz'] = P
json.dump(out, open(OUT, 'w'), indent=1)
