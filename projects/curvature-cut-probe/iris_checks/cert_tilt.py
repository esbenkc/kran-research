"""Iris (T5, lower edge of the floor band): a CERTIFIED per-statement lower bound on the output change at the probed
country token, valid for every post-norm state any fine-tune can reach. CPU float64, full 151,936-token vocabulary,
low memory (streamed unembedding, <= 0.5 GB).

Setting: z = clean post-norm country-token state, p = softmax(E z), probe s(z) = a.(z - mu) + b, clean 1% line t,
D = s(z) - t > 0 for a caught false statement. Any state z' that is no longer caught has a.(z' - z) <= -D, and its
output change is KL(z') = KL(p || softmax(E z')).

Bound (Donsker-Varadhan): for any distribution q,  KL(z') >= gq.(z' - z) - KL(q || p),  gq = E^T (q - p).
Linear tilt: q_nu = p * (1 - nu * tv), tv = (e_v - m).y, m = E^T p, for any y and 0 <= nu <= 1/max(tv). Then
gq = -nu F y with F = E^T diag(p) E - m m^T (the output Fisher at z). Write F y = kappa a + r (r perpendicular to a).
Every post-norm state lies in the RMSNorm ellipsoid {gamma * u : |u| <= sqrt(dim)}, so r.(z' - z) <= R|gamma r| - r.z.
Hence, for every reachable uncaught z',
    KL(z') >= LB(nu, y) = nu kappa D - nu (R |gamma * r| - r.z) - KL(q_nu || p)          (kappa >= 0)
KL(q_nu||p) = sum_v p_v (1 - nu tv) log(1 - nu tv) is computed exactly. For small nu, KL(q_nu||p) ~ nu^2 y'Fy / 2, so
with y = F^-1 a the best nu gives LB ~ D^2 / (2 a'F^-1 a): the second-order floor (FishBack Thm 2), now with an
exact correction. We take y = (F + eps lam_max I)^-1 a for eps in EPSS and the best nu on a grid; every (eps, nu)
gives a valid bound, so the max is valid. The positivity limit nu <= 1/max(tv) is what caps the bound when a has a
large share in near-flat directions of F.
Also logged per statement: the second-order value D^2 / (2 a'(F + eps lam_max I)^-1 a) for each eps.
Output: iris_checks/cert_tilt_<tag>.json (+ .npz per-statement arrays). Resumable per probe batch (npz checkpoint).
Run: venv1024/bin/python iris_checks/cert_tilt.py --probes data/fits/probes.npz --names unproj,proj,rand
"""
import argparse, json, os, time
os.environ.setdefault('VECLIB_MAXIMUM_THREADS', '4')
from pathlib import Path
import numpy as np
import torch
from safetensors import safe_open

torch.set_num_threads(4)
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SNAP = next(Path.home().glob('.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/*')) / 'model.safetensors'
V, DIM, CH = 151936, 1536, 4096
EPSS = [0.0, 1e-10, 1e-8, 1e-6, 1e-4]
_f = safe_open(str(SNAP), 'pt')
GAM = _f.get_tensor('model.norm.weight').double().numpy()
_E = _f.get_slice('model.embed_tokens.weight')
R = np.sqrt(DIM)


def chunks():
    for a in range(0, V, CH):
        yield a, _E[a:min(V, a + CH)].double().numpy()


def stats(z):
    """p (V,), F (DIM, DIM) at one state, one pass for logits and one for F (float64)."""
    L = np.concatenate([Ec @ z for _, Ec in chunks()])
    L -= L.max()
    p = np.exp(L)
    p /= p.sum()
    F = np.zeros((DIM, DIM))
    m = np.zeros(DIM)
    for a, Ec in chunks():
        pc = p[a:a + len(Ec)]
        F += (Ec.T * pc) @ Ec
        m += Ec.T @ pc
    F -= np.outer(m, m)
    return p, F, m


def Ey(y):
    return np.concatenate([Ec @ y for _, Ec in chunks()])


def best_lb(p, tv, kappa, D, rterm):
    """max over nu in [0, nu_max) of nu*(kappa*D - rterm) - KL(q_nu||p); concave in nu -> log grid + golden section."""
    tmax = tv.max()
    c = kappa * D - rterm
    if kappa <= 0 or tmax <= 0 or c <= 0:
        return 0.0, 0.0
    num = (1.0 / tmax) * (1 - 1e-12)

    def val(nu):
        q1 = 1 - nu * tv
        return nu * c - float(np.sum(p * q1 * np.log(q1)))

    grid = num * np.logspace(-10, 0, 41)
    vals = [val(x) for x in grid]
    k = int(np.argmax(vals))
    lo, hi = grid[max(0, k - 1)], grid[min(len(grid) - 1, k + 1)]
    g = (np.sqrt(5) - 1) / 2
    x1, x2 = hi - g * (hi - lo), lo + g * (hi - lo)
    f1, f2 = val(x1), val(x2)
    for _ in range(40):
        if f1 > f2:
            hi, x2, f2 = x2, x1, f1
            x1 = hi - g * (hi - lo)
            f1 = val(x1)
        else:
            lo, x1, f1 = x1, x2, f2
            x2 = lo + g * (hi - lo)
            f2 = val(x2)
    cand = [(vals[k], grid[k]), (f1, x1), (f2, x2)]
    v, nu = max(cand)
    return max(0.0, v), nu


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--probes', default=str(ROOT / 'data/fits/probes.npz'))
    ap.add_argument('--names', default='unproj,proj,rand')
    ap.add_argument('--tag', default='main')
    ap.add_argument('--limit', type=int, default=0)
    a = ap.parse_args()
    t0 = time.time()
    zp = ROOT / 'data/clean/z_post.npy'
    Z = np.load(zp if zp.exists() else HERE / 'inputs/clean_country_R29.npy').astype(np.float64)
    split = json.load(open(ROOT / 'data/ref/split.json'))
    y = np.array([1 - r['true'] for r in split])
    sp = np.array([r['split'] for r in split])
    ev = np.where(sp == 'eval')[0]
    evf = ev[y[ev] == 1]
    line_set = np.where((sp == 'refit') & (y == 0))[0]
    pz = np.load(a.probes)
    names = a.names.split(',')
    P = {}
    for n in names:
        w = pz[f'{n}__w'].astype(np.float64) / pz[f'{n}__sd'].astype(np.float64)
        mu = pz[f'{n}__mu'].astype(np.float64)
        b = float(pz[f'{n}__b'])
        t = float(np.sort((Z[line_set] - mu) @ w + b)[::-1][1])
        s = (Z[evf] - mu) @ w + b
        P[n] = dict(w=w, t=t, s=s, caught=s > t)
    todo = np.where(np.any([P[n]['caught'] for n in names], 0))[0]
    if a.limit:
        todo = todo[:a.limit]
    ck = HERE / f'cert_tilt_{a.tag}.npz'
    res = {n: {'LB': np.full(len(evf), np.nan), 'nu': np.full(len(evf), np.nan), 'eps': np.full(len(evf), np.nan),
               **{f'Q2_{e:g}': np.full(len(evf), np.nan) for e in EPSS}} for n in names}
    done = np.zeros(len(evf), bool)
    if ck.exists():
        old = np.load(ck)
        done = old['done']
        for n in names:
            for k in res[n]:
                res[n][k] = old[f'{n}__{k}']
    for j, i in enumerate(todo):
        if done[i]:
            continue
        z = Z[evf[i]]
        p, F, m = stats(z)
        lam, U = np.linalg.eigh(F)
        lmax = lam[-1]
        for n in names:
            if not P[n]['caught'][i]:
                continue
            w, D = P[n]['w'], P[n]['s'][i] - P[n]['t']
            cw = U.T @ w
            bestv, bestnu, beste = 0.0, 0.0, np.nan
            for e in EPSS:
                lam_r = lam + e * lmax
                if e == 0:
                    lam_r = np.where(lam > 1e-14 * lmax, lam, np.inf)
                yv = U @ (cw / lam_r)
                res[n][f'Q2_{e:g}'][i] = D * D / (2 * float(w @ yv))
                g0 = F @ yv
                kappa = float(g0 @ w / (w @ w))
                r = g0 - kappa * w
                rterm = R * float(np.linalg.norm(GAM * r)) - float(r @ z)
                tv = Ey(yv) - float(m @ yv)
                v, nu = best_lb(p, tv, kappa, D, rterm)
                if v > bestv:
                    bestv, bestnu, beste = v, nu, e
            res[n]['LB'][i], res[n]['nu'][i], res[n]['eps'][i] = bestv, bestnu, beste
        done[i] = True
        if (j + 1) % 10 == 0 or j == len(todo) - 1:
            np.savez(ck, done=done, **{f'{n}__{k}': v for n in names for k, v in res[n].items()})
            msg = {'done': int(done.sum()), 'of': int(len(todo)), 'sec': round(time.time() - t0, 1)}
            for n in names:
                c = P[n]['caught'] & done
                if c.any():
                    msg[n] = {'LB_med': float(np.nanmedian(res[n]['LB'][c])), 'Q2_1e-8_med': float(np.nanmedian(res[n]['Q2_1e-08'][c]))}
            print(json.dumps(msg), flush=True)
    out = {'probes_file': a.probes, 'z_source': str(zp if zp.exists() else 'inputs/clean_country_R29'), 'eps_grid': EPSS,
           'n_eval_all': int(len(ev)), 'n_eval_false': int(len(evf)), 'seconds': round(time.time() - t0, 1), 'probes': {}}
    for n in names:
        c = P[n]['caught'] & done
        need = max(0, int(P[n]['caught'].sum()) - int(np.floor(0.20 * len(evf))))
        lb = np.sort(np.nan_to_num(res[n]['LB'][c], nan=0.0))
        q2 = np.sort(res[n]['Q2_1e-08'][c])
        out['probes'][n] = {'line': P[n]['t'], 'clean_catch': float(P[n]['caught'].mean()), 'n_caught_done': int(c.sum()),
                            'need_to_hide_for_20pct': need,
                            'LB_quantiles': dict(zip(['p10', 'p50', 'p90'], np.quantile(lb, [.1, .5, .9]).tolist())),
                            'Q2_eps1e-8_quantiles': dict(zip(['p10', 'p50', 'p90'], np.quantile(q2, [.1, .5, .9]).tolist())),
                            'LB_over_Q2_median': float(np.median(res[n]['LB'][c] / res[n]['Q2_1e-08'][c])),
                            'pooled_LB_cheapest_need_over_1192': float(lb[:need].sum() / len(ev)) if c.sum() >= need else None,
                            'pooled_Q2_cheapest_need_over_1192': float(q2[:need].sum() / len(ev)) if c.sum() >= need else None,
                            'pooled_LB_all_caught_over_1192': float(lb.sum() / len(ev)),
                            'eps_used_counts': {f'{e:g}': int(np.sum(res[n]['eps'][c] == e)) for e in EPSS}}
    json.dump(out, open(HERE / f'cert_tilt_{a.tag}.json', 'w'), indent=1)
    print(json.dumps(out['probes'], indent=1))


if __name__ == '__main__':
    main()
