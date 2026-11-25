"""Smoke test of vera_core on a TRAIN shape only (W32 r1): Gibbs(S) vs minibatch SGLD at one cell."""
import time, numpy as np
from vera_core import *
M, N, H, r = 32, 32, 16, 1
lam, reg, lam0 = lam_parts(M, N, H, r); print('lam', lam, reg, lam0)
W0, B0, A0 = teacher(M, N, H, r, 0)
X = np.random.default_rng(5).standard_normal((1024, M)); S = X.T @ X / 1024
h = hess_eigs(B0, A0, S); print('n_reg', len(h), 'h range', h.min().round(3), h.max().round(3))
nb, q = 1000.0, 1.0; gamma = np.sqrt(nb * 0.5 / q)
t0 = time.time(); g = gibbs(W0, B0, A0, S, gamma, nb, 1500, 300, np.random.default_rng(1)); print('gibbs', round(g, 2), round(g/lam-1, 4), f'{time.time()-t0:.1f}s')
c = 0.01; eps = c / (nb * h.max() + gamma); T = max(2000, int(np.ceil(0.2/eps/1000))*1000)
t0 = time.time(); s = sgld(W0, B0, A0, X, gamma, nb, eps, T, 64, np.random.default_rng(2), chains=2); print('sgld', s.round(2), (s.mean()/lam-1).round(4), 'T', T, f'{time.time()-t0:.1f}s')
print('reg term', reg_term(h, nb, gamma, eps), 'of', reg)
