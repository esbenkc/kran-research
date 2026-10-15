"""skeptic: exact (integer-arithmetic) checks of the counting behind the closed forms.

Setting: x = [e_a; e_b], neuron j computes v_j * (u_{j,a} + s_{j,b})^2 with w_j = (u_j, s_j).
Outputs are centred over the N training points (loss convention of the run, identify_loss.py).
Exact ranks via sympy DomainMatrix over ZZ (rows scaled by N so centring stays integer).

 D1  one neuron, full data: rank = 3p-2 (paper counts 3p-1). Kernel contains
     scaling (u, s, -2v) and shift (1_p, -1_p, 0).
 D2  K neurons, generic integer point: rank = min(K(3p-2), p(N-1)).
 D3  V-only (first layer frozen): rank = p * rank(C H) = p*min(l,K)  -> Thm 5.2 value p*min(l,K)/2.
 D4  K(3p-1)/2 - K(3p-2)/2 = K/2; relative gap 1/(3p-1) <= 5% for p >= 7.
 D5  quadratic SGLD limit: nbeta*E[L] per direction = nb h / (2(nb h + g)) -> 1/2 as g/nb -> 0.
"""
import json, os, random
import numpy as np
import sympy as sp
from sympy.polys.matrices import DomainMatrix

random.seed(1)
out = {}


def J_int(p, K, pairs, W, V, freeze_W=False):
    """Integer Jacobian of N*centred outputs wrt (W,V). W: 2p x K, V: p x K int arrays."""
    N = len(pairs)
    D = (0 if freeze_W else 2 * p * K) + p * K
    J = np.zeros((N, p, D), dtype=np.int64)
    for i, (a, b) in enumerate(pairs):
        pre = W[a] + W[p + b]                       # K
        for k in range(p):
            if not freeze_W:
                g = 2 * pre * V[k]                      # K
                J[i, k, a * K:(a + 1) * K] += g
                J[i, k, (p + b) * K:(p + b + 1) * K] += g
            off = 0 if freeze_W else 2 * p * K
            J[i, k, off + k * K: off + (k + 1) * K] = pre ** 2
    Jc = N * J - J.sum(0, keepdims=True)              # N * (J - mean_i J)
    return Jc.reshape(N * p, D)


def rank_ZZ(M):
    return DomainMatrix.from_list_sympy(*M.shape, [[sp.Integer(int(x)) for x in row] for row in M]).convert_to(sp.ZZ).rank()


def rnd(n, m):
    return np.array([[random.choice([-5, -4, -3, -2, -1, 1, 2, 3, 4, 5]) for _ in range(m)] for _ in range(n)], dtype=np.int64)


D1 = {}
for p in [3, 5, 7, 11]:
    pairs = [(a, b) for a in range(p) for b in range(p)]
    W, V = rnd(2 * p, 1), rnd(p, 1)
    M = J_int(p, 1, pairs, W, V)
    scale = np.concatenate([W[:p, 0], W[p:, 0], -2 * V[:, 0]])    # d/dc of (c u, c s, c^-2 v) at c=1
    shift = np.concatenate([np.ones(p), -np.ones(p), np.zeros(p)]).astype(np.int64)
    D1[p] = dict(rank=rank_ZZ(M), paper=3 * p - 1, claim=3 * p - 2,
                 scale_in_kernel=bool(np.all(M @ scale == 0)), shift_in_kernel=bool(np.all(M @ shift == 0)))
    print("D1", p, D1[p], flush=True)
out["D1_one_neuron"] = D1

D2 = []
for p, K, frac in [(5, 2, 1.0), (5, 3, 1.0), (5, 6, 1.0), (7, 2, 1.0), (7, 4, 1.0), (7, 8, 1.0), (5, 2, 0.6), (5, 4, 0.6), (7, 5, 0.5)]:
    allp = [(a, b) for a in range(p) for b in range(p)]
    pairs = allp if frac == 1.0 else random.sample(allp, int(frac * p * p))
    N = len(pairs)
    W, V = rnd(2 * p, K), rnd(p, K)
    r = rank_ZZ(J_int(p, K, pairs, W, V))
    D2.append(dict(p=p, K=K, N=N, rank=r, pred_min_K3p2_pN1=min(K * (3 * p - 2), p * (N - 1)), paper_K3p1=K * (3 * p - 1)))
    print("D2", D2[-1], flush=True)
out["D2_K_neurons"] = D2

D3 = []
for p, K, N in [(5, 3, 20), (5, 8, 12), (7, 4, 30), (7, 10, 30)]:
    allp = [(a, b) for a in range(p) for b in range(p)]
    pairs = random.sample(allp, N)
    W, V = rnd(2 * p, K), rnd(p, K)
    M = J_int(p, K, pairs, W, V, freeze_W=True)
    H = np.array([[(W[a, j] + W[p + b, j]) ** 2 for j in range(K)] for a, b in pairs], dtype=np.int64)
    l = rank_ZZ(N * H - H.sum(0, keepdims=True))
    D3.append(dict(p=p, K=K, N=N, rank=rank_ZZ(M), l=l, pred=p * min(l, K)))
    print("D3", D3[-1], flush=True)
out["D3_V_only"] = D3

n, Kk = sp.symbols('p K', positive=True)
gap = sp.simplify(Kk * (3 * n - 1) / 2 - Kk * (3 * n - 2) / 2)
rel = sp.simplify(gap / (Kk * (3 * n - 1) / 2))
out["D4"] = dict(gap=str(gap), rel=str(rel), rel_p7=float(rel.subs(n, 7)), rel_p11=float(rel.subs(n, 11)), rel_p23=float(rel.subs(n, 23)))
print("D4", out["D4"])

h, nb, g, x = sp.symbols('h n_beta gamma x', positive=True)
dens = sp.exp(-nb * h * x ** 2 / 2 - g * x ** 2 / 2)
EL = sp.integrate(h * x ** 2 / 2 * dens, (x, -sp.oo, sp.oo)) / sp.integrate(dens, (x, -sp.oo, sp.oo))
val = sp.simplify(nb * EL)
out["D5"] = dict(per_direction=str(val), limit_g_to_0=str(sp.limit(val, g, 0)),
                 matches=bool(sp.simplify(val - nb * h / (2 * (nb * h + g))) == 0))
print("D5", out["D5"])

out["post_arithmetic"] = {"p23_K99_K(3p-1)": 99 * 68, "p23_(p-1)/2*(16p-24)": 11 * (16 * 23 - 24),
                          "p17_K68_paper": 68 * 50 // 2, "p11_K33_paper": 33 * 32 // 2, "p23_K92_paper": 92 * 68 // 2}
print(out["post_arithmetic"])
json.dump(out, open(os.path.join(os.path.dirname(__file__), "derivation.json"), "w"), indent=1, default=str)
