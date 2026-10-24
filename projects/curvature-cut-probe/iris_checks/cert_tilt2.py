"""Iris (T5): certified per-state lower bounds on the output change at the probed country token, stored as a function
of the score shift, for every eval statement (false: shift DOWN, true: shift UP). CPU, float64 where it matters,
streamed unembedding (<= 0.5 GB RSS). The pooled floors (common line shift, cheapest statements) are computed after
the pass by cert_floor.py from the saved curves.

Setting: z = clean post-norm country-token state (readout 29), p = softmax(E z), probe s(z) = a.(z - mu) + b.
Bound (Donsker-Varadhan): for any distribution q, KL(p || softmax(E z')) >= gq.(z' - z) - KL(q || p), gq = E^T(q - p).
Linear tilt: q_nu = p (1 - nu tv), tv = (E y)_v - m.y, m = E^T p, 0 <= nu < 1/max(tv). Then gq = -nu g0 with
g0 = E^T (p * tv) (computed EXACTLY by a float64 pass; F is used only to choose y). Split g0 = kappa a + r, r _|_ a.
Every post-norm state lies in the ellipsoid {gamma * u : |u| <= sqrt(dim)}, so with R = sqrt(dim):
  DOWN (false statement, a.(z'-z) <= -D):  KL >= nu (kappa D - r_dn) - KL(q_nu||p),  r_dn = R|gamma r| - r.z
  UP   (true statement,  a.(z'-z) >= +B):  use y -> -y:  KL >= nu (kappa B - r_up) - KL(q'_nu||p),
       q'_nu = p (1 + nu tv), nu < 1/max(-tv),  r_up = R|gamma r| + r.z
(kappa >= 0 needed; checked). y = (F + eps lam_max I)^-1 a for eps in EPSS, F = output Fisher at z (float64; a float32 F gave a 2x weaker bound).
Saved per statement x probe x eps: kappa, r_dn, r_up, a.y (second-order: KL ~ D^2 / (2 a.y)), and the curves
KL(q_nu||p) on a grid of nu for both signs. LB(D) = max over the grid of nu (kappa D - r) - KL_nu: every grid point
is a valid bound, so the max is valid.
Output: iris_checks/cert2_<tag>.npz (resumable).
Run: venv1024/bin/python iris_checks/cert_tilt2.py --names unproj,proj,rand [--probes data/fits/probes.npz]
"""
import argparse, json, os, time
os.environ.setdefault('VECLIB_MAXIMUM_THREADS', '4')
from pathlib import Path
import sys
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from emb_reader import Emb  # noqa: E402  (plain reads, no mmap: low RSS)
V, DIM, CH = 151936, 1536, 1024
EPSS = [0.0, 1e-8]
NG = 128
_EMB = Emb()
GAM = _EMB.gamma()
R = np.sqrt(DIM)


def chunks():
    return _EMB.chunks(CH)


def nu_grid(numax):
    g = np.concatenate([np.logspace(-9, -0.03, 104), 1 - np.logspace(-0.6, -7, 24)])
    return numax * np.sort(np.clip(g, 0, 1 - 1e-12))


def kl_curve(p, tv, nus):
    out = np.empty(len(nus))
    for k, nu in enumerate(nus):
        q1 = 1 - nu * tv
        out[k] = float(np.sum(p * q1 * np.log(q1)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--probes', default=str(ROOT / 'data/fits/probes.npz'))
    ap.add_argument('--names', default='unproj,proj,rand')
    ap.add_argument('--tag', default='main')
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--only_false', action='store_true')
    ap.add_argument('--sub', type=int, default=0)
    ap.add_argument('--pairs', type=int, default=0)
    a = ap.parse_args()
    t0 = time.time()
    zp = ROOT / 'data/clean/z_post.npy'
    Z = np.load(zp if zp.exists() else HERE / 'inputs/clean_country_R29.npy').astype(np.float64)
    split = json.load(open(ROOT / 'data/ref/split.json'))
    y = np.array([1 - r['true'] for r in split])
    sp = np.array([r['split'] for r in split])
    ev = np.where(sp == 'eval')[0]
    ev = np.concatenate([ev[y[ev] == 1], ev[y[ev] == 0]])   # false states first, then true
    if a.only_false:
        ev = ev[y[ev] == 1]
    if a.pairs:   # the seed-0 twin-pair sample of ellip_pairs.py: N eval false statements, then their true twins
        import importlib.util
        spec = importlib.util.spec_from_file_location('ep', HERE / 'ellip_pairs.py')
        ep = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(ep)
        fs, tws, _, _ = ep.sample_pairs(a.pairs)
        ev = np.concatenate([fs, tws])
    elif a.sub:   # random subsample (seed 0) of a.sub false and a.sub true eval states, false first
        rng = np.random.default_rng(0)
        fe, te = ev[y[ev] == 1], ev[y[ev] == 0]
        ev = np.concatenate([np.sort(rng.choice(fe, a.sub, replace=False)), np.sort(rng.choice(te, a.sub, replace=False))])
    if a.limit:
        ev = ev[:a.limit]
    pz = np.load(a.probes)
    names = a.names.split(',')
    W = np.stack([pz[f'{n}__w'].astype(np.float64) / pz[f'{n}__sd'].astype(np.float64) for n in names], 1)  # DIM x P
    K = len(names) * len(EPSS)
    ck = HERE / f'cert2_{a.tag}.npz'
    shp = (len(ev), len(names), len(EPSS))
    st = {'kappa': np.full(shp, np.nan), 'r_dn': np.full(shp, np.nan), 'r_up': np.full(shp, np.nan),
          'ay': np.full(shp, np.nan), 'numax_dn': np.full(shp, np.nan), 'numax_up': np.full(shp, np.nan),
          'kl_dn': np.full(shp + (NG,), np.nan), 'kl_up': np.full(shp + (NG,), np.nan), 'top_p': np.full(len(ev), np.nan),
          'lam_max': np.full(len(ev), np.nan)}
    done = np.zeros(len(ev), bool)
    if ck.exists():
        old = np.load(ck)
        if list(old['ev']) == list(ev) and list(old['names']) == names:
            done = old['done']
            for k in st:
                st[k] = old[k]
    for j in range(len(ev)):
        if done[j]:
            continue
        z = Z[ev[j]]
        L = np.concatenate([Ec @ z for _, Ec in chunks()])
        L -= L.max()
        p = np.exp(L)
        p /= p.sum()
        m = np.zeros(DIM)
        F = np.zeros((DIM, DIM))
        for s0, Ec in chunks():
            pc = p[s0:s0 + len(Ec)]
            m += Ec.T @ pc
            F += (Ec.T * pc) @ Ec
        F -= np.outer(m, m)
        lam, U = np.linalg.eigh(F)
        lmax = lam[-1]
        Ys = []
        for pi in range(len(names)):
            cw = U.T @ W[:, pi]
            for e in EPSS:
                lr = np.where(lam > 1e-12 * lmax, lam, np.inf) if e == 0 else lam + e * lmax
                Ys.append(U @ (cw / lr))
        Y = np.stack(Ys, 1)                                  # DIM x K
        T = np.concatenate([Ec @ Y for _, Ec in chunks()])   # V x K
        T -= (m @ Y)[None]
        G0 = np.zeros((DIM, K))
        for s0, Ec in chunks():
            G0 += Ec.T @ (p[s0:s0 + len(Ec), None] * T[s0:s0 + len(Ec)])
        for pi in range(len(names)):
            w = W[:, pi]
            for ei in range(len(EPSS)):
                k = pi * len(EPSS) + ei
                g0, tv = G0[:, k], T[:, k]
                kappa = float(g0 @ w / (w @ w))
                r = g0 - kappa * w
                nr = R * float(np.linalg.norm(GAM * r))
                st['kappa'][j, pi, ei] = kappa
                st['r_dn'][j, pi, ei] = nr - float(r @ z)
                st['r_up'][j, pi, ei] = nr + float(r @ z)
                st['ay'][j, pi, ei] = float(w @ Y[:, k])
                ndn, nup = 1.0 / tv.max(), 1.0 / (-tv).max()
                st['numax_dn'][j, pi, ei], st['numax_up'][j, pi, ei] = ndn, nup
                st['kl_dn'][j, pi, ei] = kl_curve(p, tv, nu_grid(ndn))
                st['kl_up'][j, pi, ei] = kl_curve(p, -tv, nu_grid(nup))
        st['top_p'][j], st['lam_max'][j] = float(p.max()), float(lmax)
        done[j] = True
        if (j + 1) % 10 == 0 or j == len(ev) - 1:
            np.savez(ck, done=done, ev=ev, names=np.array(names), epss=np.array(EPSS), nu_grid_unit=nu_grid(1.0), **st)
            print(json.dumps({'done': int(done.sum()), 'of': len(ev), 'sec': round(time.time() - t0, 1)}), flush=True)


if __name__ == '__main__':
    main()
