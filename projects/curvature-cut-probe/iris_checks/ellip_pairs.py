"""Iris (T5, D5): cheapest FOUND moves on the RMSNorm ellipsoid for twin pairs, at 4 shift sizes per statement.
CPU only, <= 0.4 GB (runs next to the fine-tune queue). An upper bound on the true minimum (a found move).

For a sampled caught false eval statement i (score s_i, clean line t, D_i = s_i - t) and its true twin j:
  c_f(a) = min KL(p_i || p_z')  over z' on the ellipsoid with s(z') <= s_i - a      (false moves DOWN by a)
  c_t(b) = min KL(p_j || p_z')  over z' on the ellipsoid with s(z') >= s_j + b      (true twin moves UP by b)
for a, b in {D/4, D/2, 3D/4, D}; c(0) = 0. Pair price (Felix D5) = min over a in {0, D/4, D/2, 3D/4, D} of
c_f(a) + c_t(D - a). Ellipsoid: z' = gamma x / sqrt(mean(x^2) + 1e-6). Search: Adam on x, loss KL + mu relu(viol)^2,
mu annealed 1e1 -> 1e5, KL on a union vocabulary (top-1024 tokens of every sampled clean state); the KL and the
score of the final state are then re-computed exactly over all 151,936 tokens in float64, and a move counts only if
it meets its constraint exactly (else +inf).
Sample: the same seed-0 sample as cert_tilt2.py --pairs (N eval false statements and their twins).
Output: iris_checks/ellip_pairs_<tag>.npz/.json
Run: venv1024/bin/python iris_checks/ellip_pairs.py --names unproj,proj,rand,lit_k128 --pairs 160 --tag k128
"""
import argparse, json, os, time
os.environ.setdefault('VECLIB_MAXIMUM_THREADS', '4')
from pathlib import Path
import sys
import numpy as np
import torch

torch.set_num_threads(4)
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from emb_reader import Emb  # noqa: E402  (plain reads, no mmap: low RSS)
V, DIM, CH, EPS = 151936, 1536, 1024, 1e-6
FR = [0.25, 0.5, 0.75, 1.0]
_EMB = Emb()
GAM = _EMB.gamma()


def chunks():
    return _EMB.chunks(CH)


def logsm_full(Z):
    L = np.concatenate([Z @ Ec.T for _, Ec in chunks()], 1)
    m = L.max(1, keepdims=True)
    return L - (m + np.log(np.exp(L - m).sum(1, keepdims=True)))


def sample_pairs(n, seed=0):
    split = json.load(open(ROOT / 'data/ref/split.json'))
    y = np.array([1 - r['true'] for r in split])
    sp = np.array([r['split'] for r in split])
    ev = np.where(sp == 'eval')[0]
    key = {}
    for i in ev:
        key.setdefault((split[i]['city'], split[i]['src']), {})[int(y[i])] = i
    fe = np.array(sorted(d[1] for d in key.values() if 0 in d and 1 in d))
    rng = np.random.default_rng(seed)
    f = np.sort(rng.choice(fe, n, replace=False))
    tw = np.array([key[(split[i]['city'], split[i]['src'])][0] for i in f])
    return f, tw, y, sp


def search(Zc, w, target, sign, Eu, steps=300):
    """sign=+1: need s <= target (move down); sign=-1: need s >= target (move up). Score part w.z only (mu, b folded
    into target). Returns final z (n, DIM) float64."""
    gam = torch.tensor(GAM, dtype=torch.float32)
    wt = torch.tensor(w, dtype=torch.float32)
    Zt = torch.tensor(Zc, dtype=torch.float32)
    tg = torch.tensor(target, dtype=torch.float32)
    lp0 = (Zt @ Eu.T).log_softmax(-1)
    p0 = lp0.exp()
    X = (Zt / gam).clone().requires_grad_(True)
    opt = torch.optim.Adam([X], lr=3e-3)
    marg = 1e-3 * (tg.abs() + 1)
    for s in range(steps):
        pen = 10.0 ** (1 + 4 * s / max(1, steps - 1))
        z = gam * X / torch.sqrt((X * X).mean(-1, keepdim=True) + EPS)
        lp = (z @ Eu.T).log_softmax(-1)
        kl = (p0 * (lp0 - lp)).sum(-1)
        viol = sign * (z @ wt - tg) + marg
        loss = (kl + pen * torch.relu(viol) ** 2).sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        z = gam * X / torch.sqrt((X * X).mean(-1, keepdim=True) + EPS)
    return z.double().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--probes', default=str(ROOT / 'data/fits/probes.npz'))
    ap.add_argument('--names', default='unproj,proj,rand')
    ap.add_argument('--pairs', type=int, default=160)
    ap.add_argument('--tag', default='main')
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--steps', type=int, default=300)
    a = ap.parse_args()
    t0 = time.time()
    f, tw, y, sp = sample_pairs(a.pairs)
    Z = np.load(ROOT / 'data/clean/z_post.npy').astype(np.float64)
    line_set = np.where((sp == 'refit') & (y == 0))[0]
    ev = np.where(sp == 'eval')[0]
    keep = np.zeros(V, bool)
    allst = np.concatenate([f, tw])
    for s0 in range(0, len(allst), 16):
        L = logsm_full(Z[allst[s0:s0 + 16]])
        keep[np.unique(np.argpartition(-L, 1024, 1)[:, :1024])] = True
        del L
    Eu = torch.tensor(np.concatenate([Ec[keep[s:s + len(Ec)]] for s, Ec in chunks()]), dtype=torch.float32)
    pz = np.load(a.probes)
    out = {'pairs': int(a.pairs), 'union_vocab': int(keep.sum()), 'fractions': FR, 'steps': a.steps, 'probes': {}}
    arrays = {'false_idx': f, 'twin_idx': tw}
    n_false = int((y[ev] == 1).sum())
    for name in a.names.split(','):
        w = pz[f'{name}__w'].astype(np.float64) / pz[f'{name}__sd'].astype(np.float64)
        mu = pz[f'{name}__mu'].astype(np.float64)
        b = float(pz[f'{name}__b'])
        t = float(np.sort((Z[line_set] - mu) @ w + b)[::-1][1])
        sf = (Z[f] - mu) @ w + b
        st = (Z[tw] - mu) @ w + b
        caught = np.where(sf > t)[0]
        D = sf[caught] - t
        cf = np.full((len(caught), len(FR)), np.inf)
        ct = np.full((len(caught), len(FR)), np.inf)
        for side, idx, base, sign, out_arr in (('false', f[caught], (Z[f[caught]] @ w), +1, cf),
                                              ('true', tw[caught], (Z[tw[caught]] @ w), -1, ct)):
            for k, fr in enumerate(FR):
                target = base - sign * fr * D           # false: w.z <= w.z0 - a ; true: w.z >= w.z0 + b
                for s0 in range(0, len(idx), a.batch):
                    sl = slice(s0, s0 + a.batch)
                    Zc = Z[idx[sl]]
                    zf = search(Zc, w, target[sl], sign, Eu, a.steps)
                    ok = (sign * (zf @ w - target[sl])) <= 1e-9 * (np.abs(target[sl]) + 1)
                    LP0 = logsm_full(Zc)
                    LP1 = logsm_full(zf)
                    kl = (np.exp(LP0) * (LP0 - LP1)).sum(1)
                    out_arr[sl, k] = np.where(ok, kl, np.inf)
                    del LP0, LP1
                print(json.dumps({'probe': name, 'side': side, 'frac': fr, 'med': float(np.median(out_arr[:, k])),
                                  'ok_share': float(np.isfinite(out_arr[:, k]).mean()), 'sec': round(time.time() - t0, 1)}),
                      flush=True)
        splits = [cf[:, 3]] + [cf[:, k] + ct[:, 2 - k] for k in range(3)] + [ct[:, 3]]  # a = D, 3D/4.. ; b = D - a
        # columns: a = D | a = D/4 (b = 3D/4) | a = D/2 (b = D/2) | a = 3D/4 (b = D/4) | a = 0 (b = D)
        pair = np.min(np.stack(splits, 1), 1)
        n_caught_all = int((((Z[ev][y[ev] == 1] - mu) @ w + b) > t).sum())
        need = max(0, n_caught_all - int(np.floor(0.20 * n_false)))
        q = need / n_caught_all if n_caught_all else 0.0

        def pooled(x):
            x = np.sort(x[np.isfinite(x)])
            m = int(round(q * len(x)))
            return float(x[:m].sum() * (n_caught_all / max(1, len(x))) / len(ev))

        rng = np.random.default_rng(1)
        boots = [pooled(pair[rng.integers(0, len(pair), len(pair))]) for _ in range(500)]
        out['probes'][name] = {'line': t, 'n_caught_in_sample': int(len(caught)), 'n_caught_all': n_caught_all,
                               'need_to_hide_for_20pct': need, 'D_median': float(np.median(D)),
                               'false_only_med': float(np.median(cf[:, 3])), 'pair_med': float(np.median(pair)),
                               'best_split_counts': np.bincount(np.argmin(np.stack(splits, 1), 1), minlength=5).tolist(),
                               'pooled_pair_over_1192': pooled(pair), 'pooled_pair_ci95': np.quantile(boots, [.025, .975]).tolist(),
                               'pooled_false_only_over_1192': pooled(cf[:, 3])}
        arrays.update({f'{name}__caught': caught, f'{name}__D': D, f'{name}__cf': cf, f'{name}__ct': ct, f'{name}__pair': pair})
        print(json.dumps({name: out['probes'][name]}), flush=True)
        np.savez(HERE / f'ellip_pairs_{a.tag}.npz', **arrays)
        json.dump(out, open(HERE / f'ellip_pairs_{a.tag}.json', 'w'), indent=1)
    out['seconds'] = round(time.time() - t0, 1)
    json.dump(out, open(HERE / f'ellip_pairs_{a.tag}.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
