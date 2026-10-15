"""skeptic2: do dead neurons (w=0, v=0) add to the SGLD LLC estimate?

Planted one-hot teacher: K_live generic neurons (random Gaussian, so a smooth point),
targets Y = teacher outputs (exact zero loss at w*), plus m extra neurons set exactly to 0.
Remark 5.5 counts only K_eff = K_live. Theory (skeptic2 derivation): one dead neuron alone has
K ~ |v|^2 |w|^4 with zeta poles at p/2 and (2p-1)/4, so its true RLCT share is up to (2p-1)/4
when the live neurons do not span the visible output space. At finite n*beta and localisation
gamma, SGLD may not resolve that quartic direction.

Usage: python dead_neurons.py [nbeta gamma eps steps]
"""
import json
import sys

import numpy as np

from sgld2 import sgld_llc_qnet

p, K_live = 7, 2
nbeta = float(sys.argv[1]) if len(sys.argv) > 1 else 1000.0
gamma = float(sys.argv[2]) if len(sys.argv) > 2 else 100.0
eps = float(sys.argv[3]) if len(sys.argv) > 3 else 1e-4
steps = int(sys.argv[4]) if len(sys.argv) > 4 else 4000
rng = np.random.default_rng(0)
A, B = (g.ravel() for g in np.meshgrid(np.arange(p), np.arange(p), indexing="ij"))
Wl = rng.standard_normal((2 * p, K_live)) / np.sqrt(2)
Vl = rng.standard_normal((p, K_live))
z = Wl[:p][A] + Wl[p:][B]
Y = (z * z) @ Vl.T  # realisable target
rows = []
for m in (0, 1, 2, 4, 8):
    W = np.concatenate([Wl, np.zeros((2 * p, m))], 1)
    V = np.concatenate([Vl, np.zeros((p, m))], 1)
    r = sgld_llc_qnet(W, V, A, B, Y, eps=eps, gamma=gamma, nbeta=nbeta, steps=steps, burn=steps // 4, chains=4, seed=m)
    row = dict(p=p, K_live=K_live, m_dead=m, nbeta=nbeta, gamma=gamma, eps=eps, steps=steps,
               lambda_hat=r["lambda_hat"], se=r["se"], truth_live=K_live * (3 * p - 2) / 2,
               paper_Keff=K_live * (3 * p - 1) / 2, dead_upper=K_live * (3 * p - 2) / 2 + m * (2 * p - 1) / 4)
    rows.append(row)
    print(json.dumps(row)); sys.stdout.flush()
json.dump(rows, open(f"dead_neurons_nb{int(nbeta)}_g{int(gamma)}.json", "w"), indent=1)
