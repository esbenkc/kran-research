"""Hugo (skeptic2): status rebuild from raw ckpt npz (own code, numpy, <0.1 GB).

Arms: U = unproj (new-code runs in data/runs/unproj__*, plus 1023 runs as context), P = proj, R = rand.
PRICE per D1 + D8 horizon: arm price = min over checkpoints 50..H of seed-mean country-token KL where the
pooled (seed-mean) catch at the re-set line <= 20%; per-seed price on its own curve (inf if never).
Status per PREREG s7 + D7 (one-sided R leg).
Null / luck checks:
  (1) statement bootstrap (2000 draws, resample the 298 eval cities with both twins and both src rows,
      re-set line fixed per checkpoint) of the price ratios P/U and R/U;
  (2) re-pool: scores recomputed from acts_<step>.npy (readout 29, float16) with probes.npz vs ckpt scores.
usage: rebuild_status.py [--horizon 150] [--boot 2000]
"""
import argparse
import glob
import json
import os
import re

import numpy as np

from check_runs import DATA, EV, EVF, REF_T, TR, col_of, split

HERE = os.path.dirname(os.path.abspath(__file__))
COL = {'U': 'unproj', 'P': 'proj', 'R': 'rand'}


def load_run(rdir, col, H):
    rows = {}
    for f in glob.glob(os.path.join(rdir, 'ckpt_*.npz')):
        t = int(re.search(r'ckpt_(\d+)', f).group(1))
        if t > H:
            continue
        d = np.load(f, allow_pickle=True)
        s = d['scores'][:, col_of(d, col)].astype(np.float64)
        kl, koff, cpos = d['kl'].astype(np.float64), d['koff'], d['cpos']
        ev = np.nonzero(EV)[0]
        klc = np.full(len(split), np.nan)
        klc[ev] = kl[koff[ev] + cpos[ev]]
        rows[t] = dict(s=s, klc=klc)
    return rows


def stats(rows, w_ev=None):
    """per checkpoint: catch at re-set line, mean country KL. w_ev: per-statement weights (bootstrap)."""
    out = {}
    for t, r in rows.items():
        s = r['s']
        line = np.sort(s[REF_T])[::-1][1]
        if w_ev is None:
            c = float((s[EVF] > line).mean())
            k = float(np.nanmean(r['klc'][EV]))
        else:
            wf = w_ev[EVF]
            c = float(((s[EVF] > line) * wf).sum() / wf.sum())
            we = w_ev[EV]
            k = float((r['klc'][EV] * we).sum() / we.sum())
        out[t] = (c, k)
    return out


def price(curves, H):
    common = sorted(set.intersection(*[set(c) for c in curves]))
    best = np.inf
    for t in common:
        if t < 50 or t > H:
            continue
        if np.mean([c[t][0] for c in curves]) <= 0.20:
            best = min(best, float(np.mean([c[t][1] for c in curves])))
    return best


def seed_price(c, H):
    ok = [k for t, (cc, k) in c.items() if 50 <= t <= H and cc <= 0.20]
    return min(ok) if ok else np.inf


def status(pr, sp):
    U, P, R = pr['U'], pr['P'], pr['R']
    if min(len(v) for v in sp.values()) < 2:
        return 'UNCONFIRMED (fewer than 2 seeds in some arm; rule-free reading below)'
    r_falls = np.isfinite(R) and R <= 1.5 * U
    yes_price = np.isfinite(P) and P >= 3 * U and min(sp['P']) > max(sp['U'])
    p_survives = not np.isfinite(P)
    no = np.isfinite(P) and P <= 1.5 * U and max(sp['P']) <= 1.5 * max(sp['U'])
    if r_falls and yes_price:
        return 'CONFIRMED (price branch)'
    if no:
        return 'REJECTED'
    if r_falls and p_survives:
        return 'CONFIRMED? (survival branch: check step-H pooled catch >= 50%)'
    return 'UNCONFIRMED'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--horizon', type=int, default=150)
    ap.add_argument('--boot', type=int, default=2000)
    ap.add_argument('--u1023', action='store_true', help='D10 slip case: add 1023 U seeds 1 and 2 to U')
    a = ap.parse_args()
    H = a.horizon
    runs = {k: sorted(glob.glob(os.path.join(DATA, 'runs', f'{v}__lam10000__seed*'))) for k, v in COL.items()}
    runs = {k: [r for r in v if os.path.exists(os.path.join(r, f'ckpt_{H}.npz'))] for k, v in runs.items()}
    if a.u1023:
        runs['U'] += [os.path.join(DATA, 'runs1023', f'r29__lam10000__seed{s}') for s in (1, 2)]  # D10 slip case
    raw = {k: [load_run(r, 'tok29' if 'runs1023' in r else COL[k], H) for r in v] for k, v in runs.items()}
    cur = {k: [stats(x) for x in v] for k, v in raw.items()}
    res = {'horizon': H, 'runs': {k: [os.path.basename(r) for r in v] for k, v in runs.items()}}
    pr = {k: price(v, H) if v else np.nan for k, v in cur.items()}
    sp = {k: [seed_price(c, H) for c in v] for k, v in cur.items()}
    res['price'] = pr
    res['seed_price'] = sp
    res['curves'] = {k: [{str(t): c for t, c in sorted(x.items())} for x in v] for k, v in cur.items()}
    res['status'] = status(pr, sp) if all(cur.values()) else 'incomplete'
    res['ratio'] = {'P/U': pr['P'] / pr['U'] if cur['U'] and cur['P'] else None,
                    'R/U': pr['R'] / pr['U'] if cur['U'] and cur['R'] else None}
    # bootstrap over cities (all 4 eval rows of a city move together)
    if a.boot and all(cur.values()):
        cities = sorted({split[i]['city'] for i in np.nonzero(EV)[0]})
        cidx = {c: j for j, c in enumerate(cities)}
        ev_city = np.array([cidx[split[i]['city']] if EV[i] else -1 for i in range(len(split))])
        rng = np.random.default_rng(1024)
        br = {'P/U': [], 'R/U': []}
        for _ in range(a.boot):
            cnt = np.bincount(rng.integers(0, len(cities), len(cities)), minlength=len(cities))
            w = np.where(ev_city >= 0, cnt[np.clip(ev_city, 0, None)], 0).astype(float)
            pb = {k: price([stats(x, w) for x in v], H) for k, v in raw.items()}
            br['P/U'].append(pb['P'] / pb['U'])
            br['R/U'].append(pb['R'] / pb['U'])
        res['boot'] = {k: dict(p2_5=float(np.nanpercentile(v, 2.5)), p50=float(np.nanpercentile(v, 50)),
                               p97_5=float(np.nanpercentile(v, 97.5)), frac_inf=float(np.mean(~np.isfinite(v))))
                       for k, v in br.items()}
    json.dump(res, open(os.path.join(HERE, 'out', f'status_H{H}{"_u1023" if a.u1023 else ""}.json'), 'w'),
              indent=1, default=float)
    print(json.dumps({k: res[k] for k in ('runs', 'price', 'seed_price', 'ratio', 'status', 'boot') if k in res},
                     default=float, indent=1))


if __name__ == '__main__':
    main()
