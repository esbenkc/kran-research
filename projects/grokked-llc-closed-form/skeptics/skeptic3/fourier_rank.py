"""skeptic3: Jacobian rank at Fourier-structured points.

A 'Fourier neuron' of frequency k: u[a]=r1 cos(t a+al), s[b]=r2 cos(t b+be), v[c]=r3 cos(t c+ga),
t = 2 pi k / p.  The grokked basin of the paper (Remark 5.5) is predicted at K_eff(3p-1)/2;
the generic one-hot count is K_eff(3p-2)/2.  Question: does rank(J) at a Fourier point equal
K_eff(3p-2), or do neurons that share a frequency lose rank?
"""
import json
import sys

import numpy as np

from rank_check import jacobian, rank


def fourier_point(p, freqs, m, rng):
    U, S, V = [], [], []
    a = np.arange(p)
    for k in freqs:
        t = 2 * np.pi * k / p
        for _ in range(m):
            al, be, ga = rng.uniform(0, 2 * np.pi, 3)
            U.append(np.cos(t * a + al))
            S.append(np.cos(t * a + be))
            V.append(np.cos(t * a + ga))
    return np.array(U), np.array(S), np.array(V).T


def run(p, freqs, m, frac, seed=0):
    rng = np.random.default_rng(seed)
    pairs = np.array([(a, b) for a in range(p) for b in range(p)])
    N = int(round(frac * p * p))
    idx = rng.permutation(len(pairs))[:N]
    A, B = pairs[idx, 0], pairs[idx, 1]
    U, S, V = fourier_point(p, freqs, m, rng)
    K = U.shape[0]
    r = rank(jacobian(U, S, V, A, B))
    return dict(p=p, n_freq=len(freqs), m_per_freq=m, K_eff=K, N=N, rank=r,
                generic=min(K * (3 * p - 2), p * (N - 1)), paper=K * (3 * p - 1))


if __name__ == "__main__":
    out = []
    for p, nf, m, frac in [(7, 1, 1, 1.0), (7, 1, 2, 1.0), (7, 1, 3, 1.0), (7, 1, 4, 1.0), (7, 1, 6, 1.0),
                           (7, 3, 1, 1.0), (7, 3, 2, 1.0), (7, 3, 4, 1.0),
                           (11, 1, 1, 1.0), (11, 1, 4, 1.0), (11, 5, 1, 1.0), (11, 5, 2, 1.0), (11, 5, 4, 1.0),
                           (17, 8, 1, 1.0), (17, 8, 2, 1.0), (17, 8, 4, 1.0),
                           (11, 5, 2, 0.5), (17, 8, 2, 0.5)]:
        res = run(p, list(range(1, nf + 1)), m, frac)
        out.append(res)
        print(json.dumps(res), flush=True)
    json.dump(out, open(sys.argv[1] if len(sys.argv) > 1 else "fourier_rank.json", "w"), indent=1)
