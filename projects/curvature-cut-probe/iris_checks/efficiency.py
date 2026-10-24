"""Iris: does the trained fine-tune pay the floor? Per eval statement in the cert sample, compare the MEASURED output
change at the country token (ckpt kl at cpos) with the CERTIFIED minimum output change for the score shift the
fine-tune actually made (cert_tilt2 curves, direction-aware: down if the score fell, up if it rose).
  overpay_i = KL_measured_i / LB_i(|ds_i|)        (>= 1 always, if LB is valid and the state is reachable)
  also KL_measured_i / Q2_i, Q2_i = ds_i^2 / (2 a.y_i)  (naive second-order formula)
Run dirs: data/runs1023/r29__lam10000__seed{0,1,2} (probe column 29 = unproj) and, when present, the new runs
data/runs/<probe>__lam10000__seed<s> (column = the attacked probe).
Usage: venv1024/bin/python iris_checks/efficiency.py --tag k128p
"""
import argparse, json, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from cert_floor import lb_curve  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', default='k128p')
    ap.add_argument('--step', type=int, default=300)
    a = ap.parse_args()
    c = np.load(HERE / f'cert2_{a.tag}.npz')
    names = list(c['names'])
    ev = c['ev']
    done = c['done']
    gunit = c['nu_grid_unit']
    pz = np.load(ROOT / 'data/fits/probes.npz')
    Z = np.load(ROOT / 'data/clean/z_post.npy').astype(np.float64)
    split = json.load(open(ROOT / 'data/ref/split.json'))
    y = np.array([1 - r['true'] for r in split])
    runs = [('unproj', ROOT / f'data/runs1023/r29__lam10000__seed{s}', 29) for s in range(3)]
    for d in sorted((ROOT / 'data/runs').glob('*__lam10000__seed*')):
        pr = d.name.split('__')[0]
        pr = {'tok29': 'unproj'}.get(pr, pr)
        if pr in names:
            runs.append((pr, d, None))
    out = {}
    for pr, d, col in runs:
        f = d / f'ckpt_{a.step}.npz'
        if not f.exists():
            continue
        name = pr if pr in names else pr.replace('_k128', '')
        pi = names.index(name)
        k1 = np.load(f)
        k0 = np.load(d / 'ckpt_0.npz')
        if col is not None:
            j = list(k1['scored']).index(col)
            s1, s0 = k1['scores'][:, j], k0['scores'][:, j]
        else:   # new runs: 'scores' (rows, 24) with column names in 'probes'
            j = [str(x) for x in k1['probes']].index(pr)
            s1, s0 = k1['scores'][:, j], k0['scores'][:, j]
        w = pz[f'{name}__w'].astype(np.float64) / pz[f'{name}__sd'].astype(np.float64)
        s_clean = (Z - pz[f'{name}__mu'].astype(np.float64)) @ w + float(pz[f'{name}__b'])
        kl = k1['kl']
        koff, cpos = k1['koff'], k1['cpos']
        rows = []
        for jj, i in enumerate(ev):
            if not done[jj]:
                continue
            ds = float(s1[i] - s0[i])
            klm = float(kl[koff[i] + cpos[i]])
            up = ds > 0
            best = 0.0
            for ei in range(c['kappa'].shape[2]):
                kk = 'up' if up else 'dn'
                v = lb_curve(c['kappa'][jj, pi, ei], c[f'r_{kk}'][jj, pi, ei], c[f'numax_{kk}'][jj, pi, ei],
                             c[f'kl_{kk}'][jj, pi, ei], gunit, np.array([abs(ds)]))[0]
                best = max(best, v)
            q2 = ds * ds / (2 * c['ay'][jj, pi, 1])
            rows.append((int(y[i]), ds, klm, best, q2, float(s0[i] - s_clean[i])))
        r = np.array(rows)
        res = {}
        for lab, m in (('false', r[:, 0] == 1), ('true', r[:, 0] == 0)):
            rr = r[m]
            if len(rr) == 0:
                continue
            ok = rr[:, 3] > 0
            res[lab] = {'n': int(m.sum()), 'ds_median': float(np.median(rr[:, 1])), 'kl_meas_median': float(np.median(rr[:, 2])),
                        'LB_median': float(np.median(rr[:, 3])), 'Q2_median': float(np.median(rr[:, 4])),
                        'overpay_LB_median': float(np.median(rr[ok, 2] / rr[ok, 3])) if ok.any() else None,
                        'overpay_LB_p10_p90': np.quantile(rr[ok, 2] / rr[ok, 3], [.1, .9]).tolist() if ok.any() else None,
                        'overpay_Q2_median': float(np.median(rr[:, 2] / rr[:, 4])),
                        'n_below_LB': int(np.sum(rr[ok, 2] < rr[ok, 3] * (1 - 1e-6))),
                        'step0_score_check_maxabs': float(np.abs(rr[:, 5]).max())}
        out[f'{pr}|{d.name}'] = res
        print(d.name, json.dumps(res), flush=True)
    json.dump(out, open(HERE / f'efficiency_{a.tag}_step{a.step}.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
