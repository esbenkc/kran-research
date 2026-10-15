"""skeptic3: exact (sympy, rational arithmetic) checks of the per-neuron count on one-hot data.

The Jacobian is built from the closed-form derivatives (checked numerically in validate_sgld.gradcheck):
  f_c(x) = sum_j V[c,j] (w_j . x)^2,  d f_c / d V[c,j] = (w_j.x)^2,  d f_c / d w_j[i] = 2 V[c,j] (w_j.x) x_i,
centred over the sample axis per output, and its rank is computed exactly over Q.
Claims checked:
 (1) one neuron, one-hot pairs x=[e_a;e_b], all p^2 pairs: rank 3p-2 (paper counts d+p-1 = 3p-1).
 (2) the two kernel vectors are exactly: scaling (u,s,v) -> (-u/2, -s/2, v) and shift (u,s) -> (+1, -1).
 (3) generic rational inputs x in Q^{2p}: same neuron has rank 3p-1 (the paper's count holds there).
 (4) K neurons on one-hot data, full table: rank K(3p-2) below the cap p(p^2-1).
"""
import json
import random

import sympy as sp
from sympy.polys.domains import QQ
from sympy.polys.matrices import DomainMatrix


def qrank(M):
    """Exact rank over Q (DomainMatrix; sympy Matrix.rank is far too slow here)."""
    return DomainMatrix.from_Matrix(M).convert_to(QQ).rank()

random.seed(0)


def rnd():
    return sp.Rational(random.choice([i for i in range(-9, 10) if i]), random.randint(1, 7))


def jac(p, K, X, W, V):
    """X: list of input vectors (len 2p). W: K x 2p, V: p x K (nested lists of Rationals). Centred Jacobian."""
    N = len(X)
    cols = K * 2 * p + p * K  # w params then V params
    rows = []
    for c in range(p):
        block = []
        for x in X:
            r = [sp.Integer(0)] * cols
            for j in range(K):
                z = sum(W[j][i] * x[i] for i in range(2 * p))
                for i in range(2 * p):
                    r[j * 2 * p + i] = 2 * V[c][j] * z * x[i]
                r[K * 2 * p + c * K + j] = z ** 2
            block.append(r)
        mean = [sum(b[k] for b in block) / N for k in range(cols)]
        rows += [[b[k] - mean[k] for k in range(cols)] for b in block]
    return sp.Matrix(rows)


def onehot_pairs(p):
    X = []
    for a in range(p):
        for b in range(p):
            x = [0] * (2 * p)
            x[a] = 1
            x[p + b] = 1
            X.append(x)
    return X


def main():
    res = {}
    for p in (3, 5):
        X = onehot_pairs(p)
        W = [[rnd() for _ in range(2 * p)]]
        V = [[rnd()] for _ in range(p)]
        J = jac(p, 1, X, W, V)
        r = qrank(J)
        scale = sp.Matrix([-W[0][i] / 2 for i in range(2 * p)] + [V[c][0] for c in range(p)])
        shift = sp.Matrix([1] * p + [-1] * p + [0] * p)
        res[f"onehot_p{p}_K1"] = dict(rank=int(r), paper_3p_minus_1=3 * p - 1, mine_3p_minus_2=3 * p - 2,
                                       scale_in_kernel=bool((J * scale).is_zero_matrix),
                                       shift_in_kernel=bool((J * shift).is_zero_matrix),
                                       nullity=int(3 * p - r))
        Xg = [[rnd() for _ in range(2 * p)] for _ in range(6 * p)]
        Jg = jac(p, 1, Xg, W, V)
        res[f"generic_inputs_p{p}_K1"] = dict(rank=int(qrank(Jg)), paper_3p_minus_1=3 * p - 1)
    p = 5
    X = onehot_pairs(p)
    for K in (2, 3):
        W = [[rnd() for _ in range(2 * p)] for _ in range(K)]
        V = [[rnd() for _ in range(K)] for _ in range(p)]
        J = jac(p, K, X, W, V)
        res[f"onehot_p{p}_K{K}"] = dict(rank=int(qrank(J)), mine=K * (3 * p - 2), paper=K * (3 * p - 1),
                                        cap=p * (len(X) - 1))
    print(json.dumps(res, indent=1))
    json.dump(res, open("sympy_neuron.json", "w"), indent=1)


if __name__ == "__main__":
    main()
