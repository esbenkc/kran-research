"""skeptic2: Jacobian rank at exact Fourier-basin constructions vs the paper's count.

Neuron j (frequency k_j, phases a_j, b_j):
  u_j(a) = cos(w a + a_j), s_j(b) = cos(w b + b_j), v_j(c) = cos(w c + a_j + b_j),  w = 2 pi k_j / p
(Gromov 2023-style quadratic-activation circuit). For phases spread over a grid the
unwanted terms cancel and f_c(a,b) ~ sum_k cos(w_k (a+b-c)) + const, which has argmax c=a+b.

We compute rank of the Jacobian of the centred outputs (all p^2 pairs, and a random 50%
train split) w.r.t. all 3pK parameters, and compare with
  paper Cor 5.4 / Rem 5.5:  K_eff (3p-1)
  generic one-hot count:    K_eff (3p-2)
  hard cap:                 p (N-1)
Rank uses SVD with relative tol 1e-9 and reports the singular-value gap.
"""
import json
import sys

import numpy as np


def build(p, freqs, m, rng=None):
    """m*m neurons per frequency on a phase grid (rng=None) or m*m random phases."""
    U, S, V = [], [], []
    a = np.arange(p)
    for k in freqs:
        w = 2 * np.pi * k / p
        for i in range(m):
            for j in range(m):
                if rng is None:
                    pa, pb = 2 * np.pi * i / m, 2 * np.pi * j / m
                else:
                    pa, pb = rng.uniform(0, 2 * np.pi, 2)
                U.append(np.cos(w * a + pa)); S.append(np.cos(w * a + pb)); V.append(np.cos(w * a + pa + pb))
    return np.array(U).T, np.array(S).T, np.array(V).T  # (p,K) each


def forward(U, S, V, A, B):
    H = (U[A] + S[B]) ** 2  # (N,K)
    return H @ V.T  # (N,p)


def jac_centred(U, S, V, A, B):
    """Jacobian of centred outputs, rows (n,c), cols [U (p*K), S (p*K), V (p*K)]."""
    p, K = U.shape
    N = len(A)
    z = U[A] + S[B]  # (N,K)
    H = z**2
    # d f[n,c] / d U[i,j] = V[c,j] * 2 z[n,j] * [A[n]==i]
    eA = np.eye(p)[A]  # (N,p)
    eB = np.eye(p)[B]
    dU = 2 * np.einsum("cj,nj,ni->ncij", V, z, eA)
    dS = 2 * np.einsum("cj,nj,ni->ncij", V, z, eB)
    dV = np.einsum("nj,ck->nckj", H, np.eye(p))  # d f[n,c]/dV[k,j] = H[n,j] [c==k]
    J = np.concatenate([dU.reshape(N, p, -1), dS.reshape(N, p, -1), dV.reshape(N, p, -1)], axis=2)
    J = J - J.mean(axis=0, keepdims=True)  # centring over samples
    return J.reshape(N * p, -1)


def rank(J, tol=1e-9):
    sv = np.linalg.svd(J, compute_uv=False)
    r = int((sv > tol * sv[0]).sum())
    gap = sv[r - 1] / sv[r] if r < len(sv) and sv[r] > 0 else np.inf
    return r, float(gap)


def main():
    rows = []
    rng = np.random.default_rng(0)
    for p in (7, 11, 17, 23):
        A, B = np.meshgrid(np.arange(p), np.arange(p), indexing="ij")
        A, B = A.ravel(), B.ravel()
        tr = rng.permutation(p * p)[: p * p // 2]
        for nf in (1, 2, (p - 1) // 2):
            freqs = list(range(1, nf + 1))
            for m, mode in ((3, "grid"), (3, "random"), (4, "random")):
                U, S, V = build(p, freqs, m, None if mode == "grid" else rng)
                K = U.shape[1]
                out = forward(U, S, V, A, B)
                acc = float((out.argmax(1) == (A + B) % p).mean())
                for split, idx in (("full", np.arange(p * p)), ("half", tr)):
                    N = len(idx)
                    if N * p * 3 * p * K > 4e8:
                        continue
                    r, gap = rank(jac_centred(U, S, V, A[idx], B[idx]))
                    row = dict(p=p, n_freq=nf, phases=mode, m=m, K_eff=K, split=split, N=N, acc_full=acc,
                               rank=r, sv_gap=gap, paper_2lam=K * (3 * p - 1), generic_2lam=K * (3 * p - 2),
                               cap_2lam=p * (N - 1))
                    rows.append(row)
                    print(json.dumps(row)); sys.stdout.flush()
    json.dump(rows, open("fourier_rank.json", "w"), indent=1)


if __name__ == "__main__":
    main()
