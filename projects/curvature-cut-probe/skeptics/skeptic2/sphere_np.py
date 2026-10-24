"""Hugo (skeptic2): low-memory (<0.1 GB) found-move cost on the RMSNorm sphere, numpy + scipy only.

For each eval twin pair (false f, true twin t) and arm, D = s_f - line_clean (clean 1% line). Targets:
false pushed DOWN by a, twin pushed UP by D - a, a in {0, D/4, D/2, 3D/4, D} (D5).
Per state and target shift a: tilt problem on the sphere y = z/gamma, ||y|| = ||y0||,
    min_v KL_K(p0 || softmax(W_K gamma*y(v))) + lam * sign * (s(z) - s0),  y(v) = r v/|v|,
with per-state lam found by bisection in log space (smallest lam whose solution reaches shift >= a).
Search uses the vocab subset K = union of tokens with p0 > 1e-6 in the batch (float32 rows);
the REPORTED cost is the exact full-vocab KL at the found point, streamed in 2048-row chunks from the
raw W file (so truncation only loosens the upper bound). Gradient by hand:
  dKL/dl = S q - p0_K (S = sum_K p0), dz = W_K^T dl + lam sign u, dy = gamma dz,
  dv = (r/|v|)(I - vv^T/|v|^2) dy.
usage: sphere_np.py OFF NPAIRS ARM [ARM ...]  -> out/sphere_<arm>_off<OFF>_n<N>.json
"""
import json
import os
import sys
import time

import numpy as np
from scipy.optimize import minimize

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, '..', 'data')
WRAW = '/tmp/hugo_W.f32'
V, d = 151936, 1536
CH = 2048


def stream(f):
    buf = np.empty((CH, d), dtype=np.float32)
    with open(WRAW, 'rb', buffering=0) as fh:
        for a in range(0, V, CH):
            n = min(CH, V - a)
            fh.readinto(memoryview(buf)[:n].cast('B'))
            f(a, buf[:n])


def lse_and_sel(Z0, thr=1e-6):
    """per-state log-sum-exp of full logits, sum p0*lp0 (negative entropy), and selected tokens."""
    B = len(Z0)
    m = np.full(B, -np.inf); s = np.zeros(B)
    st = {}

    def f1(a, Wc):
        l = Z0 @ Wc.T.astype(np.float64)
        mx = np.maximum(m, l.max(1))
        s[:] = s * np.exp(m - mx) + np.exp(l - mx[:, None]).sum(1)
        m[:] = mx
    stream(f1)
    lse = m + np.log(s)
    negent = np.zeros(B); sel = set()

    def f2(a, Wc):
        l = Z0 @ Wc.T.astype(np.float64)
        lp = l - lse[:, None]
        p = np.exp(lp)
        negent[:] += (p * lp).sum(1)
        sel.update((a + np.nonzero((p > thr).any(0))[0]).tolist())
    stream(f2)
    return lse, negent, np.array(sorted(sel))


def exact_kl(Z0, lse0, negent0, Z):
    """KL(p0||softmax(W z)) = sum p0 lp0 - sum p0 l + lse(l), streamed."""
    B = len(Z)
    m = np.full(B, -np.inf); s = np.zeros(B); cross = np.zeros(B)

    def f(a, Wc):
        W64 = Wc.T.astype(np.float64)
        l0 = Z0 @ W64
        l = Z @ W64
        p0 = np.exp(l0 - lse0[:, None])
        cross[:] += (p0 * l).sum(1)
        mx = np.maximum(m, l.max(1))
        s[:] = s * np.exp(m - mx) + np.exp(l - mx[:, None]).sum(1)
        m[:] = mx
    stream(f)
    return negent0 - cross + (m + np.log(s))


def gather_rows(idx):
    out = np.empty((len(idx), d), dtype=np.float32)
    pos = [0]

    def f(a, Wc):
        k = idx[(idx >= a) & (idx < a + len(Wc))] - a
        out[pos[0]:pos[0] + len(k)] = Wc[k]
        pos[0] += len(k)
    stream(f)
    return out


def solve(WK, lp0K, p0K, g, y0, u, sign, lam, v0, maxiter=150):
    """batched separable tilt; returns v (B,d)."""
    B = len(y0)
    r = np.linalg.norm(y0, axis=1, keepdims=True)
    S = p0K.sum(1, keepdims=True)
    s0 = (g * y0) @ u

    def fg(vf):
        v = vf.reshape(B, d)
        nv = np.linalg.norm(v, axis=1, keepdims=True)
        y = r * v / nv
        z = g * y
        l = (z.astype(np.float32) @ WK.T).astype(np.float64)
        l -= l.max(1, keepdims=True)
        lq = l - np.log(np.exp(l).sum(1, keepdims=True))
        q = np.exp(lq)
        kl = (p0K * (lp0K - lq)).sum(1)
        sc = z @ u
        f = (kl + lam * sign * (sc - s0)).sum()
        dl = S * q - p0K
        dz = (dl.astype(np.float32) @ WK).astype(np.float64) + (lam * sign)[:, None] * u[None, :]
        dy = g * dz
        vh = v / nv
        dv = (r / nv) * (dy - (dy * vh).sum(1, keepdims=True) * vh)
        return f, dv.ravel()
    res = minimize(fg, v0.ravel(), jac=True, method='L-BFGS-B',
                   options={'maxiter': maxiter, 'gtol': 1e-14, 'ftol': 1e-16, 'maxcor': 30})
    return res.x.reshape(B, d)


def main():
    off, npairs, arms = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3:]
    from gate_hugo import pairs_order
    split = json.load(open(os.path.join(DATA, 'ref/split.json')))
    zall = np.load(os.path.join(DATA, 'clean/z_post.npy'), mmap_mode='r')
    pz = np.load(os.path.join(DATA, 'fits/probes.npz'))
    gam = np.load(os.path.join(HERE, 'gamma.npy'))
    tr = np.array([x['true'] for x in split]); sp = np.array([x['split'] for x in split])
    order = pairs_order(split)[off:off + npairs]
    idx = np.array([i for pr in order for i in pr])
    Z0 = np.asarray(zall[idx], dtype=np.float64)
    t0 = time.time()
    lse0, negent0, sel = lse_and_sel(Z0)
    WK = gather_rows(sel)
    l0K = Z0 @ WK.T.astype(np.float64)
    lp0K = l0K - lse0[:, None]
    p0K = np.exp(lp0K)
    print('setup', round(time.time() - t0), 's, K =', len(sel), flush=True)
    y0 = Z0 / gam
    sign = np.where(tr[idx] == 0, 1.0, -1.0)
    fracs = [0.25, 0.5, 0.75, 1.0]
    for arm in arms:
        t1 = time.time()
        w, mu, sd = [pz[f'{arm}__{k}'].astype(np.float64) for k in ('w', 'mu', 'sd')]
        u = w / sd
        sc_all = np.asarray(zall @ u.astype(np.float32), dtype=np.float64)  # mu and bias cancel in shifts
        line = np.sort(sc_all[(sp == 'refit') & (tr == 1)])[::-1][1]
        s0 = Z0 @ u
        line = line + float(np.mean(s0 - sc_all[idx]))  # put the line on the float64 scale of s0
        Dp = s0[0::2] - line
        D = np.repeat(Dp, 2)
        res = {'arm': arm, 'line_nobias': float(line), 'idx': idx.tolist(), 'D_pair': Dp.tolist(), 'targets': {}}
        for fr in fracs:
            tgt = np.where(sign > 0, fr * D, fr * D)  # false down fr*D, twin up fr*D (pair combos below)
            lo = np.full(len(idx), -10.0); hi = np.full(len(idx), 0.0)
            best_v = None
            v = y0.copy()
            vs_hi = None
            for it in range(16):
                mid = 0.5 * (lo + hi)
                v = solve(WK, lp0K, p0K, gam, y0, u, sign, 10 ** mid, y0, maxiter=int(os.environ.get("MAXIT", 400)))
                sh = sign * (s0 - (gam * (np.linalg.norm(y0, axis=1, keepdims=True) * v /
                                          np.linalg.norm(v, axis=1, keepdims=True))) @ u)
                ok = sh >= tgt
                if best_v is None:
                    best_v = v.copy(); best_sh = sh.copy(); best_ok = ok.copy(); best_lam = mid.copy()
                upd = ok & (~best_ok | (mid < best_lam))
                best_v[upd] = v[upd]; best_sh[upd] = sh[upd]; best_ok |= ok; best_lam[upd] = mid[upd]
                hi = np.where(ok, mid, hi)
                lo = np.where(ok, lo, mid)
            r = np.linalg.norm(y0, axis=1, keepdims=True)
            Zb = gam * (r * best_v / np.linalg.norm(best_v, axis=1, keepdims=True))
            kl = exact_kl(Z0, lse0, negent0, Zb)
            res['targets'][str(fr)] = dict(shift=best_sh.tolist(), reached=best_ok.tolist(), kl=kl.tolist(),
                                           log10lam=best_lam.tolist(),
                                           sphere_dev=float(np.abs(np.linalg.norm(Zb / gam, axis=1) - r[:, 0]).max()))
            print(arm, fr, 'reached', int(best_ok.sum()), '/', len(idx), 'median kl %.3e' % np.median(kl),
                  round(time.time() - t1), 's', flush=True)
        json.dump(res, open(os.path.join(HERE, 'out', f'sphere_{arm}_off{off}_n{npairs}.json'), 'w'))


if __name__ == '__main__':
    main()
