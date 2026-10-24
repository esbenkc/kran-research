"""Iris (T5): pooled floors from the saved certificate curves (cert_tilt2.py). No model, seconds, tiny memory.

Per probe: clean line t (2nd-highest of the 200 refit-split true scores), D_i = s_i - t for caught eval false i.
Per state, certified:  LBdn_i(D) = max(0, max_grid nu (kappa D - r_dn) - KL_nu), same for UP with r_up, KL'_nu;
second-order:          Q2_i(D) = D^2 / (2 a.y) with y = (F + 1e-8 lam_max I)^-1 a (the naive lemma).
Pooled floors in the unit of eval_kl_country_pos (sum over eval statements / n_eval):
  false-only (D4):  sum of the cheapest (n_caught - floor(0.2 * n_false)) LBdn_i(D_i) / n_eval
  common shift (Kasper #6058; the measured attack raises every true score by about the same b, so the re-set line
  rises by b):  min over b of [ sum_{eval true j} LBup_j(b) + cheapest-needed sum_i LBdn_i(max(0, D_i - b)) ] / n_eval
The per-state bounds are certified; the pooled floors add one modelling assumption each (which statements move).
Run: venv1024/bin/python iris_checks/cert_floor.py --tag main [--probes data/fits/probes.npz]
"""
import argparse, json, os
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def lb_curve(kappa, r, numax, kl, gunit, D):
    """vectorised over D (array): max(0, max_g nu_g (kappa D - r) - kl_g)."""
    nus = numax * gunit
    if not np.isfinite(kappa) or kappa <= 0:
        return np.zeros_like(D)
    v = nus[None, :] * (kappa * D[:, None] - r) - kl[None, :]
    return np.maximum(0.0, v.max(1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', default='main')
    ap.add_argument('--probes', default=str(ROOT / 'data/fits/probes.npz'))
    a = ap.parse_args()
    c = np.load(HERE / f'cert2_{a.tag}.npz')
    names = list(c['names'])
    ev = c['ev']
    done = c['done']
    gunit = c['nu_grid_unit']
    zp = ROOT / 'data/clean/z_post.npy'
    Z = np.load(zp if zp.exists() else HERE / 'inputs/clean_country_R29.npy').astype(np.float64)
    split = json.load(open(ROOT / 'data/ref/split.json'))
    y = np.array([1 - r['true'] for r in split])
    sp = np.array([r['split'] for r in split])
    ev_all = np.where(sp == 'eval')[0]
    n_eval, n_false = len(ev_all), int((y[ev_all] == 1).sum())
    line_set = np.where((sp == 'refit') & (y == 0))[0]
    pz = np.load(a.probes)
    out = {'tag': a.tag, 'n_states_done': int(done.sum()), 'n_eval': n_eval, 'probes': {}}
    for pi, n in enumerate(names):
        w = pz[f'{n}__w'].astype(np.float64) / pz[f'{n}__sd'].astype(np.float64)
        mu = pz[f'{n}__mu'].astype(np.float64)
        b0 = float(pz[f'{n}__b'])
        t = float(np.sort((Z[line_set] - mu) @ w + b0)[::-1][1])
        s = (Z[ev] - mu) @ w + b0
        isf = y[ev] == 1
        caught = isf & (s > t) & done
        n_caught_all = int(((Z[ev_all][y[ev_all] == 1] - mu) @ w + b0 > t).sum())
        need = max(0, n_caught_all - int(np.floor(0.20 * n_false)))
        D = s - t

        def LB(j, d, up=False):
            best = np.zeros_like(d)
            for ei in range(c['kappa'].shape[2]):
                if up:
                    v = lb_curve(c['kappa'][j, pi, ei], c['r_up'][j, pi, ei], c['numax_up'][j, pi, ei], c['kl_up'][j, pi, ei], gunit, d)
                else:
                    v = lb_curve(c['kappa'][j, pi, ei], c['r_dn'][j, pi, ei], c['numax_dn'][j, pi, ei], c['kl_dn'][j, pi, ei], gunit, d)
                best = np.maximum(best, v)
            return best

        fj = np.where(caught)[0]
        tj = np.where((~isf) & done)[0]
        lb_f = np.array([LB(j, np.array([D[j]]))[0] for j in fj])
        q2_f = np.array([D[j] ** 2 / (2 * c['ay'][j, pi, 1]) for j in fj])
        rec = {'line': t, 'n_caught_eval_false': n_caught_all, 'n_caught_states_done': int(len(fj)),
               'need_to_hide_for_20pct': need, 'D_median': float(np.median(D[fj])) if len(fj) else None,
               'LB_false_quantiles': dict(zip(['p10', 'p50', 'p90'], np.quantile(lb_f, [.1, .5, .9]).tolist())) if len(fj) else None,
               'Q2_false_quantiles': dict(zip(['p10', 'p50', 'p90'], np.quantile(q2_f, [.1, .5, .9]).tolist())) if len(fj) else None,
               'LB_over_Q2_median': float(np.median(lb_f / q2_f)) if len(fj) else None}
        full = done.all()
        if len(fj) >= need:
            rec['floor_false_only_LB'] = float(np.sort(lb_f)[:need].sum() / n_eval)
            rec['floor_false_only_Q2'] = float(np.sort(q2_f)[:need].sum() / n_eval)
        if full and len(tj):
            bgrid = np.linspace(0, float(np.quantile(D[fj], 0.9)), 41)
            up = np.stack([LB(j, bgrid, up=True) for j in tj])            # n_true x nb
            q2up = np.stack([bgrid ** 2 / (2 * c['ay'][j, pi, 1]) for j in tj])
            best = None
            for bi, b in enumerate(bgrid):
                dd = np.maximum(0.0, D[fj] - b)
                cf = np.sort([LB(j, np.array([dd[k]]))[0] for k, j in enumerate(fj)])[:need].sum()
                cq = np.sort(dd ** 2 / (2 * c['ay'][fj, pi, 1]))[:need].sum()
                tot = (up[:, bi].sum() + cf) / n_eval
                totq = (q2up[:, bi].sum() + cq) / n_eval
                row = {'b': float(b), 'LB': float(tot), 'Q2': float(totq)}
                if best is None or tot < best['LB']:
                    best = row
                if bi == 0:
                    rec['common_shift_b0'] = row
                rec.setdefault('common_shift_curve', []).append(row)
            rec['floor_common_shift_LB'] = best
            rec['floor_common_shift_Q2_min'] = min(r['Q2'] for r in rec['common_shift_curve'])
        out['probes'][n] = rec
    json.dump(out, open(HERE / f'cert_floor_{a.tag}.json', 'w'), indent=1)
    print(json.dumps({n: {k: v for k, v in r.items() if k != 'common_shift_curve'} for n, r in out['probes'].items()}, indent=1))


if __name__ == '__main__' and not os.environ.get('CERT_PAIRS'):
    main()


def pairs_main(tag, probes, n_pairs_split=40):
    """Pair floors on the twin sample of cert_tilt2.py --pairs (first half false, second half their true twins).
    pair5  = min over a in {0, D/4, D/2, 3D/4, D} of LBdn_f(a) + LBup_t(D - a)   (same splits as the found moves, D5)
    pairC  = certified over EVERY split a in [0, D]: LB is nondecreasing in the shift, so on a grid a_0 < .. < a_K
             min_k [LBdn_f(a_k) + LBup_t(D - a_{k+1})] <= the continuous minimum.  (K = n_pairs_split)
    Q2pair = D^2 / (2 (a.y_f + a.y_t))   (the naive second-order pair formula, Kasper #6058)
    Pooled (unit of eval_kl_country_pos): scale the cheapest share need/n_caught of the sample to the n_caught pairs."""
    c = np.load(HERE / f'cert2_{tag}.npz')
    names = list(c['names'])
    ev = c['ev']
    n = len(ev) // 2
    assert c['done'][:2 * n].all(), 'run not finished'
    gunit = c['nu_grid_unit']
    Z = np.load(ROOT / 'data/clean/z_post.npy').astype(np.float64)
    split = json.load(open(ROOT / 'data/ref/split.json'))
    y = np.array([1 - r['true'] for r in split])
    sp = np.array([r['split'] for r in split])
    ev_all = np.where(sp == 'eval')[0]
    n_eval, n_false = len(ev_all), int((y[ev_all] == 1).sum())
    line_set = np.where((sp == 'refit') & (y == 0))[0]
    pz = np.load(probes)
    out = {'tag': tag, 'pairs': n, 'probes': {}}
    for pi, nm in enumerate(names):
        w = pz[f'{nm}__w'].astype(np.float64) / pz[f'{nm}__sd'].astype(np.float64)
        mu = pz[f'{nm}__mu'].astype(np.float64)
        b0 = float(pz[f'{nm}__b'])
        t = float(np.sort((Z[line_set] - mu) @ w + b0)[::-1][1])
        s = (Z[ev[:n]] - mu) @ w + b0
        caught = np.where(s > t)[0]
        n_caught_all = int((((Z[ev_all][y[ev_all] == 1] - mu) @ w + b0) > t).sum())
        need = max(0, n_caught_all - int(np.floor(0.20 * n_false)))

        def LBj(j, d, up):
            best = np.zeros_like(d)
            for ei in range(c['kappa'].shape[2]):
                k = 'up' if up else 'dn'
                v = lb_curve(c['kappa'][j, pi, ei], c[f'r_{k}'][j, pi, ei], c[f'numax_{k}'][j, pi, ei], c[f'kl_{k}'][j, pi, ei], gunit, d)
                best = np.maximum(best, v)
            return best

        p5, pC, q2, fo = [], [], [], []
        for i in caught:
            D = s[i] - t
            g5 = D * np.array([0, .25, .5, .75, 1.0])
            p5.append(np.min(LBj(i, g5, False) + LBj(n + i, D - g5, True)))
            gC = D * np.linspace(0, 1, n_pairs_split + 1)
            f_ = LBj(i, gC, False)
            t_ = LBj(n + i, D - gC, True)
            pC.append(np.min(f_[:-1] + t_[1:]))
            q2.append(D * D / (2 * (c['ay'][i, pi, 1] + c['ay'][n + i, pi, 1])))
            fo.append(LBj(i, np.array([D]), False)[0])
        p5, pC, q2, fo = map(np.array, (p5, pC, q2, fo))
        q = need / n_caught_all if n_caught_all else 0.0

        def pooled(x):
            x = np.sort(x)
            m = int(round(q * len(x)))
            return float(x[:m].sum() * (n_caught_all / max(1, len(x))) / n_eval)

        rng = np.random.default_rng(1)
        ci = lambda x: np.quantile([pooled(x[rng.integers(0, len(x), len(x))]) for _ in range(500)], [.025, .975]).tolist()
        out['probes'][nm] = {'line': t, 'n_caught_in_sample': int(len(caught)), 'n_caught_all': n_caught_all, 'need': need,
                             'pair5_med': float(np.median(p5)), 'pairC_med': float(np.median(pC)), 'Q2pair_med': float(np.median(q2)),
                             'false_only_LB_med': float(np.median(fo)),
                             'pooled_pair5': pooled(p5), 'pooled_pair5_ci95': ci(p5),
                             'pooled_pairC': pooled(pC), 'pooled_pairC_ci95': ci(pC),
                             'pooled_Q2pair': pooled(q2), 'pooled_false_only_LB': pooled(fo),
                             'pairC_over_Q2pair_med': float(np.median(pC / q2))}
        np.savez(HERE / f'cert_pairs_{tag}_{nm}.npz', caught=caught, pair5=p5, pairC=pC, q2pair=q2, false_only=fo)
    json.dump(out, open(HERE / f'cert_pairs_{tag}.json', 'w'), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == '__main__' and os.environ.get('CERT_PAIRS'):
    import sys
    pairs_main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else str(ROOT / 'data/fits/probes.npz'))
