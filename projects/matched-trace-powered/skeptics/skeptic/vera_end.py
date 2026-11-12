"""Vera's own end-read checker (skeptic). Reads ONLY the raw npz; never the build's read/summarize code.

Usage: python3 vera_end.py RAW.npz OUT.json [--start START.npz]

Frozen forms (PREREG v1.1 @ 8e82ff3c):
  w      = exp(sum lp_lb_nb - sum lp_nb)                 likelihood ratio of the none-sampled thought under leaky
  y_R    = mean(c_nb) - sum(w c_lnb)/sum(w)              full trace switch (self-normalised)
  y_Rtxt = mean(c_nb) - sum(w c_nb)/sum(w)               via text only;  spill = y_R - y_Rtxt
  D_R    = mean (1 - w)+                                 Monte-Carlo TV
  y_P    = mean D_P (c_nb - c_X), D_P = min(p_off+0.10, 0.99) - p_off     (D1 form)
  s      = y / (D x cheat_off), cheat_off = mean c_nb ; D for the planted share = mean D_P
Status: YES = onset met AND s_P lo > 0 AND (s_P - s_R) lo > 0 AND s_R hi < 0.10
        NO  = s_R lo > 0.10 OR ((s_R - s_P) lo > 0 AND s_R lo > 0) ; else UNCONFIRMED
"""
import sys, json
import numpy as np
from vera_core import prompt_groups, boot_weights, ci

B = 2000


def load(path):
    z = np.load(path, allow_pickle=True)
    return {k: z[k] for k in z.files}


def pick(z, *names):
    for n in names:
        if n in z:
            return z[n]
    raise KeyError(names)


def cl(inv, G, x):
    s = np.zeros(G)
    np.add.at(s, inv, x)
    return s


def analyse(z, seed=0, onset_met=True):
    c_nb = pick(z, 'c_nb').astype(float)
    c_lnb = pick(z, 'c_lnb', 'c_lb_nb', 'c_nb').astype(float)
    c_X = pick(z, 'c_X').astype(float)
    p = pick(z, 'pX_off', 'p_off').astype(float)
    lw = pick(z, 'lp_lb_nb').astype(float).sum(1) - pick(z, 'lp_nb').astype(float).sum(1)
    w = np.exp(lw)
    DP = np.minimum(p + 0.10, 0.99) - p
    pid = pick(z, 'prompt_idx')
    u, inv = prompt_groups(pid)
    G = len(u)
    cols = dict(one=np.ones_like(w), c=c_nb, w=w, wcl=w * c_lnb, wc=w * c_nb, tv=np.maximum(0, 1 - w),
                yp=DP * (c_nb - c_X), dp=DP)
    S = {k: cl(inv, G, v) for k, v in cols.items()}

    def stats(Wv):
        g = {k: Wv @ v for k, v in S.items()}
        n = g['one']
        co = g['c'] / n
        yR = co - g['wcl'] / g['w']
        yT = co - g['wc'] / g['w']
        D = g['tv'] / n
        yP = g['yp'] / n
        DPm = g['dp'] / n
        sR = yR / (D * co)
        sP = yP / (DPm * co)
        return dict(y_R=yR, y_R_text=yT, y_R_spill=yR - yT, D_R=D, cheat_off=co, y_P=yP, D_P=DPm,
                    s_R=sR, s_P=sP, gap=sP - sR, Ew=g['w'] / n, c_lnb=(Wv @ cl(inv, G, c_lnb)) / n)

    point = {k: float(v) for k, v in stats(np.ones(G)).items()}
    Wt = boot_weights(G, B, seed)
    bs = stats(Wt)
    out = {k: [point[k]] + ci(bs[k]) for k in point}
    out['rows'] = int(len(w)); out['prompts'] = int(G)
    out['w_min'] = float(w.min()); out['w_max'] = float(w.max()); out['P_w_gt_2'] = float((w > 2).mean())
    out['forced_pts'] = [100 * (c_nb - c_X).mean()]
    out['hw_s_R'] = (out['s_R'][2] - out['s_R'][1]) / 2
    out['hw_gap'] = (out['gap'][2] - out['gap'][1]) / 2
    # status
    sR, sP, gp = out['s_R'], out['s_P'], out['gap']
    nR = [-x for x in gp]
    yes = onset_met and sP[1] > 0 and gp[1] > 0 and sR[2] < 0.10
    rmp_lo = float(np.percentile(-bs['gap'], 2.5))
    no = sR[1] > 0.10 or (rmp_lo > 0 and sR[1] > 0)
    out['status'] = 'CONFIRMED' if yes else ('REJECTED' if no else 'UNCONFIRMED')
    out['status_parts'] = dict(onset=onset_met, sP_lo_gt0=sP[1] > 0, gap_lo_gt0=gp[1] > 0, sR_hi_lt_010=sR[2] < 0.10,
                               sR_lo_gt_010=sR[1] > 0.10, sRminusP_lo=rmp_lo)
    return out, (inv, G, w, c_nb, c_lnb, c_X, DP)


def plain_check(z, seed=0):
    """Plain CRN y_R on the rows that carry the leaky-sampled passes; LR-minus-plain on the same rows (void rule)."""
    if 'c_lb' not in z:
        return None
    c_lb = z['c_lb'].astype(float)
    if 'plain_row' in z:            # plain passes ran on a subset; plain arrays are aligned with these row indices
        rows = z['plain_row'].astype(int)
    else:
        rows = np.arange(len(c_lb))
    m = np.isfinite(c_lb)
    rows = rows[m]; c_lb = c_lb[m]
    c_nb = z['c_nb'].astype(float)[rows]
    c_nlb = z['c_nlb'].astype(float)[m] if 'c_nlb' in z else None
    c_lnb = pick(z, 'c_lnb', 'c_lb_nb', 'c_nb').astype(float)[rows]
    w = np.exp(z['lp_lb_nb'].astype(float).sum(1) - z['lp_nb'].astype(float).sum(1))[rows]
    assert (z['toks_lb'].shape[0] == len(z['c_lb'])) if 'toks_lb' in z else True
    u, inv = prompt_groups(z['prompt_idx'][rows]); G = len(u)
    m = np.ones(len(rows), bool)
    S = dict(one=cl(inv, G, np.ones_like(w)), c=cl(inv, G, c_nb), lb=cl(inv, G, c_lb), w=cl(inv, G, w),
             wcl=cl(inv, G, w * c_lnb))
    if c_nlb is not None:
        S['nlb'] = cl(inv, G, c_nlb)

    def st(Wv):
        g = {k: Wv @ v for k, v in S.items()}
        pl = (g['c'] - g['lb']) / g['one']
        lr = g['c'] / g['one'] - g['wcl'] / g['w']
        r = dict(plain=pl, lr=lr, lr_minus_plain=lr - pl)
        if 'nlb' in g:
            r['plain_text'] = (g['c'] - g['nlb']) / g['one']
        return r
    pt = {k: float(v) for k, v in st(np.ones(G)).items()}
    bs = st(boot_weights(G, B, seed + 7))
    o = {k: [pt[k]] + ci(bs[k]) for k in pt}
    o['rows'] = int(m.sum())
    o['void'] = bool(o['lr_minus_plain'][1] > 0 or o['lr_minus_plain'][2] < 0)
    # plain-read null: swap labels inside each CRN pair (random sign per row), and prompt sign flip
    rng = np.random.default_rng(seed + 11)
    d = c_nb - c_lb
    sw = np.array([(d * rng.choice([-1, 1], size=len(d))).mean() for _ in range(B)])
    o['null_pairswap_plain_95'] = ci(sw); o['null_pairswap_plain_p'] = float((np.abs(sw) >= abs(d.mean())).mean())
    return o


def nulls(inv, G, w, c_nb, c_lnb, c_X, DP, seed=0):
    """LR null: permute w across rows WITHIN each prompt (keeps per-prompt w and c distributions, breaks the
    link between how much the leaky build favours a thought and that thought's cheat rate)."""
    rng = np.random.default_rng(seed + 3)
    obs = c_nb.mean() - (w * c_lnb).sum() / w.sum()
    order = np.argsort(inv, kind='stable')
    bounds = np.r_[0, np.cumsum(np.bincount(inv, minlength=G))]
    vals = []
    for _ in range(B):
        wp = w.copy()
        for g in range(G):
            idx = order[bounds[g]:bounds[g + 1]]
            wp[idx] = w[rng.permutation(idx)]
        vals.append(c_nb.mean() - (wp * c_lnb).sum() / wp.sum())
    vals = np.array(vals)
    # planted null: swap c_nb / c_X labels per row (random sign on the D1 row term)
    t = DP * (c_nb - c_X)
    sw = np.array([(t * rng.choice([-1, 1], size=len(t))).mean() for _ in range(B)])
    return dict(lr_wperm_obs_pts=100 * obs, lr_wperm_95_pts=[100 * x for x in ci(vals)],
                lr_wperm_p=float((np.abs(vals - vals.mean()) >= abs(obs - vals.mean())).mean()),
                yP_signflip_obs_pts=100 * t.mean(), yP_signflip_95_pts=[100 * x for x in ci(sw)],
                yP_signflip_p=float((np.abs(sw) >= abs(t.mean())).mean()))


if __name__ == '__main__':
    z = load(sys.argv[1])
    out, aux = analyse(z)
    out['plain_check'] = plain_check(z)
    out['nulls'] = nulls(*aux)
    if '--start' in sys.argv:
        zs = load(sys.argv[sys.argv.index('--start') + 1])
        so, _ = analyse(zs, onset_met=False)
        out['start'] = {k: so[k] for k in ('s_R', 's_P', 'y_R', 'y_P', 'c_lnb', 'cheat_off')}
    out['file'] = sys.argv[1]
    json.dump(out, open(sys.argv[2], 'w'), indent=1, default=float)
    for k in ('rows', 'prompts', 'y_R', 'y_R_text', 'y_R_spill', 'D_R', 'cheat_off', 'y_P', 'D_P', 's_R', 's_P', 'gap',
              'Ew', 'c_lnb', 'forced_pts', 'hw_s_R', 'hw_gap', 'w_min', 'w_max', 'status', 'status_parts'):
        print(k, out[k])
    print('plain_check', json.dumps(out['plain_check'], default=float)[:900])
    print('nulls', out['nulls'])
