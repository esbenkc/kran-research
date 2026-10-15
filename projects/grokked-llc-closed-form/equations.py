"""Check every displayed equation of the post (sympy + numeric). Run: uv run --with sympy python equations.py

E1  lambda_mem  = 1/2 p min{l, K}          (Cullen et al. v3, Thm 5.2; frozen-W random-feature model)
    check: for random W, the V-only Jacobian rank of the centred train outputs equals p*min(l, K),
           l = rank of the centred feature matrix F~ = P_perp sigma(X W) (= N-1 on one-hot data when K >= N-1).
E2  lambda_grok = 1/2 K_eff (3p - 1)       (Cor 5.4 / Remark 5.5): the generic per-neuron count d + p - 1 = 3p - 1
    holds for generic inputs (checked: Gaussian X gives rank K(3p-1)).
E3  one-hot inputs: w -> w + c(1_p, -1_p) leaves every pre-activation u_a + s_b fixed (sympy), so the generic
    count is K(3p-2) below the cap p(N-1) (numeric rank).
E4  SGLD estimator  lambda_hat = n beta (E[L_n] - L_n(w*))   (Lau et al.): on a regular quadratic loss with known
    lambda = r/2 the localised quadratic value sum 1/2 h/(h + gamma/(n beta)) -> r/2 as gamma/(n beta) -> 0 (sympy limit).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import sympy as sp

import qnet

OUT = Path(__file__).parent / "data" / "equations_check.json"
res = {}

# ---- E3 symbolic: shift symmetry of one neuron on one-hot inputs
p = 5
u = sp.symbols(f"u0:{p}")
s = sp.symbols(f"s0:{p}")
v = sp.symbols(f"v0:{p}")
c = sp.symbols("c")
ok = True
for a in range(p):
    for b in range(p):
        pre = u[a] + s[b]
        shifted = (u[a] + c) + (s[b] - c)
        ok &= sp.simplify(shifted - pre) == 0
res["E3_shift_invariance_symbolic_p5"] = bool(ok)

# symbolic Jacobian rank of ONE neuron f_o(a,b) = v_o (u_a + s_b)^2 on the full p x p table (centred)
outs = []
for a in range(p):
    for b in range(p):
        for o in range(p):
            outs.append(v[o] * (u[a] + s[b]) ** 2)
M = sp.Matrix(outs).reshape(p * p, p)
M = M - sp.ones(p * p, 1) * (sp.ones(1, p * p) * M) / (p * p)    # centre along samples
params = list(u) + list(s) + list(v)
J = sp.Matrix(list(M)).jacobian(params)
rng = np.random.default_rng(0)
subs = {x: sp.Rational(int(rng.integers(-9, 10)), int(rng.integers(1, 5))) for x in params}
rank1 = J.subs(subs).rank()
res["E3_one_neuron_rank_symbolic_p5"] = int(rank1)
res["E3_expected_3p_minus_2"] = 3 * p - 2
assert rank1 == 3 * p - 2, rank1

# ---- E2/E3 numeric: generic rank, Gaussian vs one-hot
rows = []
for p_, K, n, kind in [(7, 3, 400, "gauss"), (7, 6, 400, "gauss"), (11, 4, 800, "gauss"),
                       (7, 3, 49, "onehot"), (11, 6, 60, "onehot"), (17, 12, 144, "onehot"),
                       (7, 14, 24, "onehot")]:
    r_ = np.random.default_rng(1)
    X = r_.standard_normal((n, 2 * p_)) if kind == "gauss" else qnet.modadd_data(p_, n / p_ ** 2, 1)[0]
    W = r_.standard_normal((2 * p_, K))
    V = r_.standard_normal((p_, K))
    rank, _ = qnet.jac_rank(W, V, X)
    N = X.shape[0]
    rows.append(dict(kind=kind, p=p_, K=K, N=N, rank=rank, paper=K * (3 * p_ - 1),
                     shift=K * (3 * p_ - 2), cap=p_ * (N - 1)))
    if kind == "gauss":
        assert rank == K * (3 * p_ - 1), rows[-1]
    else:
        assert rank == min(K * (3 * p_ - 2), p_ * (N - 1)), rows[-1]
res["E2_E3_numeric_rank"] = rows

# ---- E1 numeric: V-only rank = p * min(l, K)
rows = []
for p_, K, frac in [(11, 22, 0.5), (11, 80, 0.5), (17, 68, 0.5), (17, 200, 0.5)]:
    r_ = np.random.default_rng(2)
    X = qnet.modadd_data(p_, frac, 2)[0]
    W = r_.standard_normal((2 * p_, K))
    V = r_.standard_normal((p_, K))
    Jv = qnet.jacobian(W, V, X)[:, W.size:]
    rank_v = int(np.linalg.matrix_rank(Jv, tol=1e-8 * np.linalg.norm(Jv, 2)))
    F = (X @ W) ** 2
    Ft = F - F.mean(0, keepdims=True)
    l = int(np.linalg.matrix_rank(Ft, tol=1e-8 * np.linalg.norm(Ft, 2)))
    rows.append(dict(p=p_, K=K, N=X.shape[0], l=l, rank_V=rank_v, p_min_lK=p_ * min(l, K)))
    assert rank_v == p_ * min(l, K), rows[-1]
res["E1_vonly_rank"] = rows

# ---- E4 symbolic: localised quadratic SGLD target -> r/2
h, g, nb = sp.symbols("h gamma nbeta", positive=True)
term = sp.Rational(1, 2) * h / (h + g / nb)
res["E4_limit_per_direction"] = str(sp.limit(term, nb, sp.oo))
assert sp.limit(term, nb, sp.oo) == sp.Rational(1, 2)

# ---- E5 numeric: exactly Fourier-pure net (all F=(p-1)/2 frequencies, 10 neurons each, enough to saturate each frequency, random phases), full table.
# Margin claim: rank J = F(20p-36) - 16 C(F,2) = F(16p-24); half of it at p=23 is 1892 (vs 1/2 K_eff (3p-1) = 3366 at K=99).
rows = []
for p_ in (7, 11):
    F = (p_ - 1) // 2
    r_ = np.random.default_rng(3)
    a = np.arange(p_)
    cols_w, cols_v = [], []
    for k in range(1, F + 1):
        for _ in range(10):
            ph = r_.uniform(0, 2 * np.pi, 3)
            cols_w.append(np.concatenate([np.cos(2 * np.pi * k * a / p_ + ph[0]), np.cos(2 * np.pi * k * a / p_ + ph[1])]))
            cols_v.append(np.cos(2 * np.pi * k * a / p_ + ph[2]))
    W = np.stack(cols_w, 1)
    V = np.stack(cols_v, 1)
    X = qnet.modadd_data(p_, 1.0, 0)[0]
    rank, _ = qnet.jac_rank(W, V, X, rtol=1e-9)
    rows.append(dict(p=p_, F=F, K=10 * F, rank=rank, formula=F * (20 * p_ - 36) - 16 * F * (F - 1) // 2,
                     formula_alt=F * (16 * p_ - 24)))
    assert rank == rows[-1]["formula"] == rows[-1]["formula_alt"], rows[-1]
res["E5_pure_fourier_rank"] = rows
p23 = 23
res["E5_numbers"] = dict(half_pure_fourier_p23=((p23 - 1) // 2) * (16 * p23 - 24) / 2,
                         paper_p23_K99=99 * (3 * p23 - 1) / 2)
assert res["E5_numbers"]["half_pure_fourier_p23"] == 1892 and res["E5_numbers"]["paper_p23_K99"] == 3366

OUT.write_text(json.dumps(res, indent=1, default=str))
print(json.dumps(res, indent=1, default=str))
print("ALL EQUATION CHECKS PASSED")
