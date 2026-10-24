"""Hugo (skeptic2) power gate, own code. Per eval state: full-vocab float64 output Fisher F_i at the
country token (post-norm z), then g_arm = u'(F_i + eps I)^-1 u, u = w/sd (score gradient in z).
Low RSS: W (151936 x 1536 float32) is streamed from a raw file in 8192-row chunks (~0.3 GB peak).

usage: gate_hugo.py WORKER NWORKERS     (writes out/gate_w{WORKER}.jsonl, one line per state)
"""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, '..', 'data')
WRAW = '/tmp/hugo_W.f32'
V, d = 151936, 1536
CH = 8192
ARMS = ['unproj', 'proj_k128', 'rand_k128', 'lit_k128']
EPS = [1e-10, 1e-8, 1e-6]


def ensure_wraw():
    if os.path.exists(WRAW) and os.path.getsize(WRAW) == V * d * 4:
        return
    from fisher_hugo import load_W
    W, _ = load_W()
    W.astype(np.float32).tofile(WRAW + '.tmp')
    os.replace(WRAW + '.tmp', WRAW)


def chunks():
    buf = np.empty((CH, d), dtype=np.float32)
    with open(WRAW, 'rb', buffering=0) as f:
        for a in range(0, V, CH):
            n = min(CH, V - a)
            mv = memoryview(buf)[:n].cast('B')
            f.readinto(mv)
            yield a, buf[:n]


def logits(z):
    out = np.empty(V)
    for a, Wc in chunks():
        out[a:a + len(Wc)] = Wc.astype(np.float64) @ z
    return out


def fisher(p):
    F = np.zeros((d, d))
    m = np.zeros(d)
    for a, Wc in chunks():
        W64 = Wc.astype(np.float64)
        pc = p[a:a + len(Wc)]
        m += W64.T @ pc
        W64 *= np.sqrt(pc)[:, None]
        F += W64.T @ W64
    F -= np.outer(m, m)
    return 0.5 * (F + F.T)


def pairs_order(split):
    """eval pairs (false idx, true twin idx), same city + src; random order seed 2410."""
    ev = [i for i, x in enumerate(split) if x['split'] == 'eval']
    key = {}
    for i in ev:
        x = split[i]
        key.setdefault((x['city'], x['src']), {})[x['true']] = i
    pr = sorted((v[0], v[1]) for v in key.values() if 0 in v and 1 in v)
    rng = np.random.default_rng(2410)
    return [pr[j] for j in rng.permutation(len(pr))]


def main():
    wk, nw = int(sys.argv[1]), int(sys.argv[2])
    split = json.load(open(os.path.join(DATA, 'ref/split.json')))
    z = np.load(os.path.join(DATA, 'clean/z_post.npy'), mmap_mode='r')
    pz = np.load(os.path.join(DATA, 'fits/probes.npz'))
    U = np.stack([pz[f'{a}__w'].astype(np.float64) / pz[f'{a}__sd'].astype(np.float64) for a in ARMS], 1)
    order = pairs_order(split)
    os.makedirs(os.path.join(HERE, 'out'), exist_ok=True)
    outp = os.path.join(HERE, 'out', f'gate_w{wk}.jsonl')
    done = set()
    if os.path.exists(outp):
        done = {json.loads(l)['idx'] for l in open(outp)}
    with open(outp, 'a') as fo:
        for j, (fi, ti) in enumerate(order):
            if j % nw != wk:
                continue
            for idx in (fi, ti):
                if idx in done:
                    continue
                t0 = time.time()
                zi = np.asarray(z[idx], dtype=np.float64)
                l = logits(zi)
                l -= l.max()
                p = np.exp(l)
                p /= p.sum()
                F = fisher(p)
                lam, Vv = np.linalg.eigh(F)
                lam = np.clip(lam, 0, None)
                C = Vv.T @ U
                g = {f'{e:g}': ((C ** 2) / (lam + e * lam[-1])[:, None]).sum(0).tolist() for e in EPS}
                flat = {f'{c:g}': ((C[lam < c * lam[-1]] ** 2).sum(0) / (C ** 2).sum(0)).tolist() for c in (1e-6, 1e-4)}
                top = np.argsort(p)[::-1][:64]
                rec = dict(idx=int(idx), pair_rank=j, true=int(split[idx]['true']), arms=ARMS, g=g, flat=flat,
                           lam_max=float(lam[-1]), n_eig_above_1e8=int((lam > 1e-8 * lam[-1]).sum()),
                           pmax=float(p.max()), top_ids=top.tolist(), top_p=p[top].tolist(), sec=time.time() - t0)
                fo.write(json.dumps(rec) + '\n')
                fo.flush()


if __name__ == '__main__':
    if sys.argv[1] == 'prep':
        ensure_wraw()
    else:
        main()
