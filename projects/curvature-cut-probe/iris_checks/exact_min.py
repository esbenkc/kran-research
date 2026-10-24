"""Iris: exact free-space minimum of the output change at the country token for a required probe-score drop D,
by Newton's method on the hyperplane a.d = -D (convex problem: logits are linear in the post-norm state), float64,
full 151,936-token vocabulary, then a Donsker-Varadhan certificate at the Newton point (q = softmax(l + E d_N)):
    LB = nu D - KL(q||p) - nu (R|gamma r| - r.z),  E^T(q - p) = -nu a + r.
If LB ~= f(d_N), the minimum is pinned from both sides. Compares with the second-order formula D^2/(2 a'F^-1 a) and
with the linear-tilt certificate. Few statements; CPU, 2 threads, low RSS (plain-read unembedding).
Run: VECLIB_MAXIMUM_THREADS=2 venv1024/bin/python iris_checks/exact_min.py --names unproj,proj --n 4
"""
import argparse, json, os, sys, time
os.environ.setdefault('VECLIB_MAXIMUM_THREADS', '2')
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from emb_reader import Emb  # noqa: E402

V, DIM, CH = 151936, 1536, 2048
EMB = Emb()
GAM = EMB.gamma()
R = np.sqrt(DIM)


def logits(z):
    return np.concatenate([Ec @ z for _, Ec in EMB.chunks(CH)])


def softmax(L):
    L = L - L.max()
    p = np.exp(L)
    return p / p.sum()


def fisher_and_mean(p):
    F = np.zeros((DIM, DIM))
    m = np.zeros(DIM)
    for a, Ec in EMB.chunks(CH):
        pc = p[a:a + len(Ec)]
        F += (Ec.T * pc) @ Ec
        m += Ec.T @ pc
    return F - np.outer(m, m), m


def kl(p, q):
    mask = p > 0
    return float(np.sum(p[mask] * (np.log(p[mask]) - np.log(q[mask]))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--names', default='unproj,proj')
    ap.add_argument('--n', type=int, default=4)
    ap.add_argument('--iters', type=int, default=12)
    ap.add_argument('--start', type=int, default=0)
    ap.add_argument('--tag', default='')
    a = ap.parse_args()
    t0 = time.time()
    Z = np.load(ROOT / 'data/clean/z_post.npy').astype(np.float64)
    split = json.load(open(ROOT / 'data/ref/split.json'))
    y = np.array([1 - r['true'] for r in split])
    sp = np.array([r['split'] for r in split])
    line_set = np.where((sp == 'refit') & (y == 0))[0]
    spec = __import__('importlib.util').util.spec_from_file_location('ep', HERE / 'ellip_pairs.py')
    # same pair sample as cert_tilt2 --pairs 160 (first n false statements of it)
    import importlib.util
    ep = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ep)
    fs, _, _, _ = ep.sample_pairs(160)
    pz = np.load(ROOT / 'data/fits/probes.npz')
    out = []
    for i in fs[a.start:a.start + a.n]:
        z = Z[i]
        L0 = logits(z)
        p = softmax(L0)
        F0, m0 = fisher_and_mean(p)
        lam, U = np.linalg.eigh(F0)
        for name in a.names.split(','):
            w = pz[f'{name}__w'].astype(np.float64) / pz[f'{name}__sd'].astype(np.float64)
            mu = pz[f'{name}__mu'].astype(np.float64)
            b = float(pz[f'{name}__b'])
            t = float(np.sort((Z[line_set] - mu) @ w + b)[::-1][1])
            D = float((z - mu) @ w + b - t)
            if D <= 0:
                continue
            Finv_w = U @ ((U.T @ w) / np.where(lam > 1e-12 * lam[-1], lam, np.inf))
            q2 = D * D / (2 * float(w @ Finv_w))
            # Newton on the hyperplane: d = d0 + P y, P = projector orthogonal to w
            wh = w / np.linalg.norm(w)
            d = -D * Finv_w / float(w @ Finv_w)      # second-order optimum as the start
            hist = []
            for it in range(a.iters):
                q = softmax(logits(z + d))
                f = kl(p, q)
                Fq, mq = fisher_and_mean(q)
                g = mq - m0                            # grad of KL(p || q(d)) wrt d = E^T (q - p)
                Pg = g - (g @ wh) * wh
                H = Fq - np.outer(Fq @ wh, wh) - np.outer(wh, wh @ Fq) + (wh @ Fq @ wh) * np.outer(wh, wh)
                H += np.outer(wh, wh) * np.trace(Fq) / DIM  # make H invertible along w (no move along w)
                step = np.linalg.solve(H + 1e-14 * np.trace(Fq) * np.eye(DIM), Pg)
                step -= (step @ wh) * wh
                # backtracking line search on f
                s = 1.0
                while s > 1e-4:
                    qn = softmax(logits(z + d - s * step))
                    if kl(p, qn) <= f - 1e-4 * s * float(Pg @ step):
                        break
                    s *= 0.5
                d = d - s * step
                hist.append({'it': it, 'f': f, 'proj_grad': float(np.linalg.norm(Pg)), 's': s})
                if np.linalg.norm(Pg) < 1e-12:
                    break
            q = softmax(logits(z + d))
            fN = kl(p, q)
            gq = (fisher_and_mean(q)[1] - m0)
            nu = -float(gq @ w) / float(w @ w)
            r = gq + nu * w
            lb = nu * D - kl(q, p) - nu * (R * float(np.linalg.norm(GAM * r)) - float(r @ z))
            row = {'stmt': int(i), 'probe': name, 'D': D, 'q2': q2, 'f_newton': fN, 'LB_dv_at_newton': lb,
                   'f_over_q2': fN / q2, 'LB_over_f': lb / fN, 'move_norm_over_z': float(np.linalg.norm(d) / np.linalg.norm(z)),
                   'iters': hist[-3:], 'sec': round(time.time() - t0, 1)}
            out.append(row)
            print(json.dumps(row), flush=True)
            json.dump(out, open(HERE / f'exact_min{a.tag}.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
