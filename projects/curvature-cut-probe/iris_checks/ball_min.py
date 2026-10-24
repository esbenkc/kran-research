"""Iris: exact minimum over the SOLID RMSNorm ellipsoid (the convex hull of every reachable post-norm state), with
a Donsker-Varadhan certificate. If the optimum lies on the boundary, it is also the exact minimum over the ellipsoid
SURFACE (the reachable set), pinned from both sides; if it lies inside, the surface minimum is higher and this is
only a lower bound for it.

Problem (one statement, one probe, required drop D of the probe score):
  min_d KL(p || softmax(E (z + d)))  s.t.  a.d = -D,  |(z + d) / gamma|^2 <= R^2,  R^2 = dim
Convex (logits linear in z; ellipsoid convex). Solved by a log-barrier Newton method on the hyperplane:
  phi_tau(d) = KL(d) - (1/tau) log(R^2 - |(z + d)/gamma|^2),  tau = 1e4 .. 1e12 (x100), Newton + backtracking.
Certificate at the final point z* = z + d*: q = softmax(E z*), g = E^T (q - p);
  for every z' in the ellipsoid with a.(z' - z) <= -D:  KL(z') >= g.(z' - z) - KL(q||p)
  >= min_{u in ball(R), (gamma a).u <= a.z - D} (gamma g).u - g.z - KL(q||p)          (closed form)
Prints f(z*), the certificate, |u*|/R, and compares with the free-space minimum and the formula.
Memory: < 0.1 GB (plain-read unembedding; float32 Hessian for the Newton direction only). CPU, 2 threads.
Run: VECLIB_MAXIMUM_THREADS=2 venv1024/bin/python iris_checks/ball_min.py --names unproj,proj --pairs_from hugo --n 4
"""
import argparse, json, os, subprocess, sys, threading, time
os.environ.setdefault('VECLIB_MAXIMUM_THREADS', '2')
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from emb_reader import Emb  # noqa: E402

V, DIM, CH = 151936, 1536, 256
EMB = Emb()
GAM = EMB.gamma()
R2 = float(DIM)


def logits(z):
    return np.concatenate([Ec @ z for _, Ec in EMB.chunks(CH)])


def softmax(L):
    L = L - L.max()
    p = np.exp(L)
    return p / p.sum()


def fisher_mean(q):
    """F in float32 (used ONLY for the Newton direction; gradients, KL and the certificate are float64), m float64."""
    F = np.zeros((DIM, DIM), np.float32)
    m = np.zeros(DIM)
    for a, Ec in EMB.chunks(CH):
        qc = q[a:a + len(Ec)]
        E32 = Ec.astype(np.float32)
        F += (E32.T * qc.astype(np.float32)) @ E32
        m += Ec.T @ qc
    F -= np.outer(m, m).astype(np.float32)
    return F, m


def kl(p, q):
    return float(np.sum(p * (np.log(np.maximum(p, 1e-300)) - np.log(np.maximum(q, 1e-300)))))


def lin_min_ball(gt, at, c, R):
    """min (gt.u) over |u| <= R, at.u <= c (closed form)."""
    na2 = at @ at
    if c < -R * np.sqrt(na2):
        return np.inf
    u0 = -R * gt / np.linalg.norm(gt)
    if at @ u0 <= c:
        return -R * np.linalg.norm(gt)
    pr = (gt @ at) / na2
    perp = np.linalg.norm(gt - pr * at)
    return c * pr - np.sqrt(max(0.0, R * R - c * c / na2)) * perp


def solve(z, p, m0, w, D, log):
    wh = w / np.linalg.norm(w)
    g2 = GAM ** 2
    # strictly feasible start: shrink radially by eps, then fix the score along w
    eps = 0.02
    while True:
        d = -eps * z
        d += (-D - w @ d) / (w @ w) * w
        s = R2 - np.sum(((z + d) / GAM) ** 2)
        if s > 0:
            break
        eps *= 2
    for tau in [1e4, 1e6, 1e8, 1e10, 1e12]:
        for it in range(30):
            q = softmax(logits(z + d))
            F, mq = fisher_mean(q)
            v = (z + d) / g2
            s = R2 - np.sum(((z + d) / GAM) ** 2)
            f = kl(p, q)
            phi = f - np.log(s) / tau
            grad = (mq - m0) + (2 * v / s) / tau
            H = F            # float32, in place
            del F
            H[np.diag_indices(DIM)] += ((2 / g2) / s / tau).astype(np.float32)
            v32, w32 = v.astype(np.float32), wh.astype(np.float32)
            H += np.float32(4 / s ** 2 / tau) * np.outer(v32, v32)
            hw = H @ w32
            whw = float(w32 @ hw)
            H -= np.outer(hw, w32)
            H -= np.outer(w32, hw)
            H += np.float32(whw + float(np.trace(H)) / DIM) * np.outer(w32, w32)
            gp = grad - (grad @ wh) * wh
            step = np.linalg.solve(H, gp.astype(np.float32)).astype(np.float64)
            del H
            step -= (step @ wh) * wh
            dec = float(gp @ step)
            t = 1.0
            while t > 1e-6:
                dn = d - t * step
                sn = R2 - np.sum(((z + dn) / GAM) ** 2)
                if sn > 0:
                    qn = softmax(logits(z + dn))
                    if kl(p, qn) - np.log(sn) / tau <= phi - 0.25 * t * dec:
                        break
                t *= 0.5
            d = dn
            if dec / 2 < 1e-14:
                break
        log.append({'tau': tau, 'its': it + 1, 'f': f, 'slack': float(s), 'newton_dec': dec})
    solve.last = {'tau': tau, 'slack': float(s)}
    return d


def attack_alive():
    return subprocess.run(['pgrep', '-f', 'attack_proj.py'], capture_output=True).returncode == 0


def guard():
    """Vera #6194 window (a): run only while no fine-tune is alive; exit at once when one starts (resumable)."""
    while True:
        if attack_alive():
            print(json.dumps({'guard': 'attack_proj.py started, exiting', 't': time.strftime('%H:%M:%S')}), flush=True)
            os._exit(3)
        time.sleep(3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--names', default='unproj,proj')
    ap.add_argument('--n', type=int, default=4)
    ap.add_argument('--start', type=int, default=0)
    ap.add_argument('--pairs_from', default='hugo', choices=['hugo', 'iris'])
    ap.add_argument('--tag', default='')
    ap.add_argument('--guard', action='store_true', help='run only while no attack_proj.py is alive')
    a = ap.parse_args()
    t0 = time.time()
    Z = np.load(ROOT / 'data/clean/z_post.npy', mmap_mode='r')
    split = json.load(open(ROOT / 'data/ref/split.json'))
    y = np.array([1 - r['true'] for r in split])
    sp = np.array([r['split'] for r in split])
    line_set = np.where((sp == 'refit') & (y == 0))[0]
    ZL = np.asarray(Z[line_set], dtype=np.float64)
    ev = [i for i, x in enumerate(split) if x['split'] == 'eval']
    key = {}
    for i in ev:
        key.setdefault((split[i]['city'], split[i]['src']), {})[split[i]['true']] = i
    pr = sorted((v[0], v[1]) for v in key.values() if 0 in v and 1 in v)
    if a.pairs_from == 'hugo':     # skeptic2/gate_hugo.py pairs_order: seed 2410 permutation
        pr = [pr[j] for j in np.random.default_rng(2410).permutation(len(pr))]
    else:
        rng = np.random.default_rng(0)
        fs = np.sort(rng.choice(np.array([f for f, _ in pr]), 160, replace=False))
        tw = dict(pr)
        pr = [(f, tw[f]) for f in fs]
    pz = np.load(ROOT / 'data/fits/probes.npz')
    outp = HERE / f'ball_min{a.tag}.json'
    out = json.load(open(outp)) if outp.exists() else []
    have = {(r['false_stmt'], r['probe']) for r in out}
    if a.guard:
        while attack_alive():
            time.sleep(10)
        threading.Thread(target=guard, daemon=True).start()
    for fi, ti in pr[a.start:a.start + a.n]:
        if all((int(fi), n) in have for n in a.names.split(',')):
            continue
        z = np.asarray(Z[fi], dtype=np.float64)
        p = softmax(logits(z))
        m0 = np.zeros(DIM)
        for s0, Ec in EMB.chunks(CH):
            m0 += Ec.T @ p[s0:s0 + len(Ec)]
        for name in a.names.split(','):
            if (int(fi), name) in have:
                continue
            w = pz[f'{name}__w'].astype(np.float64) / pz[f'{name}__sd'].astype(np.float64)
            mu = pz[f'{name}__mu'].astype(np.float64)
            b = float(pz[f'{name}__b'])
            t = float(np.sort((ZL - mu) @ w + b)[::-1][1])
            D = float((z - mu) @ w + b - t)
            if D <= 0:
                continue
            log = []
            d = solve(z, p, m0, w, D, log)
            zs = z + d
            q = softmax(logits(zs))
            f = kl(p, q)
            g = np.zeros(DIM)
            for s0, Ec in EMB.chunks(CH):
                g += Ec.T @ (q[s0:s0 + len(Ec)] - p[s0:s0 + len(Ec)])
            lb = lin_min_ball(GAM * g, GAM * w, float(w @ z - D), np.sqrt(R2)) - float(g @ z) - kl(q, p)
            # KKT multipliers for Vera's independent check: g + nu a + rho z*/gamma^2 = 0 (rho from the barrier)
            s_fin = R2 - float(np.sum((zs / GAM) ** 2))
            rho = 2.0 / (solve.last['tau'] * s_fin)
            nu = -float((g + rho * zs / GAM ** 2) @ w) / float(w @ w)
            kkt_res = float(np.linalg.norm(g + nu * w + rho * zs / GAM ** 2) / np.linalg.norm(g))
            ptsf = HERE / f'ball_min_points{a.tag}.npz'
            old_pts = dict(np.load(ptsf)) if ptsf.exists() else {}
            key = f'{int(fi)}__{name}'
            old_pts.update({f'{key}__z0': z, f'{key}__zstar': zs, f'{key}__nu': np.array(nu), f'{key}__rho': np.array(rho),
                            f'{key}__D': np.array(D), f'{key}__w': w})
            np.savez(ptsf, **old_pts)
            row = {'false_stmt': int(fi), 'probe': name, 'D': D, 'f_ball': f, 'LB_ball': lb, 'LB_over_f': lb / f,
                   'nu': nu, 'rho': rho, 'kkt_rel_residual': kkt_res,
                   'u_norm_over_R': float(np.linalg.norm(zs / GAM) / np.sqrt(R2)),
                   'score_shift_check': float(w @ d + D), 'log': log[-2:], 'sec': round(time.time() - t0, 1)}
            out.append(row)
            print(json.dumps(row), flush=True)
            json.dump(out, open(outp, 'w'), indent=1)


if __name__ == '__main__':
    main()
