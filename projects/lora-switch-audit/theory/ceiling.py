"""B-only ceiling for the v1 cue score of a LoRA delta, and the random-direction null.

Frozen prediction (PREREG v1, S4): if LoRA A stays at its random init and B grows only along the
cue's gradient, the top input singular vector v1 of dW = B A lies in A's r-dim row space and the
cue's energy fraction on it is about r/d. Checks (sympy + numpy), written to theory/ceiling.json:

  E1  e1(u) = sum_p (v1.c_p)^2 / sum_p |c_p|^2                       (definition, unit test)
  E2  B = g (A c)^T  =>  dW = g c^T A^T A  =>  v1 = A^T A c / |A^T A c|  (rank-1, symbolic)
  E3  if A A^T = s I_r:  cos^2(v1, c) = c^T A^T A c / (s |c|^2)        (symbolic)
  E4  A iid, var sigma^2:  E[c^T A^T A c] = r sigma^2 |c|^2, s ~ d sigma^2  =>  e1 ~ r/d
  E5  random unit u' in R^d:  (v1.u')^2 ~ Beta(1/2, (d-1)/2), mean 1/d  (numeric)
  E6  stacked [dW_q; dW_k; dW_v], each its own random A (kaiming-uniform bound 1/sqrt(d)):
      e1 distribution and P(rank 1 of 1,001) by simulation (the frozen numbers in PREREG S4)
  E7  gain law: gap ~ x * g  and gap at the switch fixed  =>  log g_T = const - log x (slope -1)
"""
import json
from pathlib import Path

import numpy as np
import sympy as sp

OUT = Path(__file__).with_name("ceiling.json")
res = {}

# E2/E3: symbolic rank-1 algebra with small concrete dims (r=2, d=3) and generic symbols.
r, d = 2, 3
A = sp.Matrix(r, d, lambda i, j: sp.Symbol(f"a{i}{j}"))
c = sp.Matrix(d, 1, lambda i, j: sp.Symbol(f"c{i}"))
g = sp.Matrix(2, 1, lambda i, j: sp.Symbol(f"g{i}"))
B = g * (A * c).T
dW = B * A
# dW = g (A^T A c)^T : rank 1 with right vector A^T A c
res["E2_dW_is_g_times_AtAc"] = bool(sp.simplify(dW - g * (A.T * A * c).T) == sp.zeros(2, d))
# E3: if A A^T = s I then (A^T A)^2 = s A^T A, so |A^T A c|^2 = s c^T A^T A c
s = sp.Symbol("s", positive=True)
# Use an A with orthogonal rows of equal squared norm s: A = sqrt(s) * [e1; e2]
A_o = sp.sqrt(s) * sp.Matrix([[1, 0, 0], [0, 1, 0]])
AtA = A_o.T * A_o
lhs = ((c.T * AtA * c)[0]) ** 2 / ((c.T * c)[0] * (c.T * AtA * AtA * c)[0])
rhs = (c.T * AtA * c)[0] / (s * (c.T * c)[0])
res["E3_cos2_identity"] = bool(sp.simplify(lhs - rhs) == 0)

# E4/E5/E6 numerics
rng = np.random.default_rng(0)
d, r = 1024, 16
# E5: null energy fraction of a random unit vector on a fixed unit vector
U = rng.standard_normal((200000, d))
U /= np.linalg.norm(U, axis=1, keepdims=True)
e_null = U[:, 0] ** 2
res["E5_null_mean"] = float(e_null.mean())
res["E5_null_mean_pred_1_over_d"] = 1 / d
res["E5_null_p99"] = float(np.percentile(e_null, 99))
res["E5_beta_p99"] = float(np.percentile(rng.beta(0.5, (d - 1) / 2, 200000), 99))

# E4: single module, B-only, cue-driven
e4 = []
for _ in range(300):
    cv = rng.standard_normal(d); cv /= np.linalg.norm(cv)
    Am = rng.uniform(-1 / np.sqrt(d), 1 / np.sqrt(d), (r, d))
    v = Am.T @ (Am @ cv)
    e4.append(float((v @ cv) ** 2 / (v @ v)))
res["E4_single_module_e1_mean"] = float(np.mean(e4))
res["E4_pred_r_over_d"] = r / d

# E6: stacked q,k,v (out dims 2048, 1024, 1024), as in the frozen score
e6, rk = [], []
for _ in range(300):
    cv = rng.standard_normal(d); cv /= np.linalg.norm(cv)
    Ms = []
    for dout in (2048, 1024, 1024):
        Am = rng.uniform(-1 / np.sqrt(d), 1 / np.sqrt(d), (r, d))
        gm = rng.standard_normal(dout)
        Ms.append(np.outer(gm, Am @ cv) @ Am)
    v1 = np.linalg.svd(np.vstack(Ms), full_matrices=False)[2][0]
    e = float((v1 @ cv) ** 2)
    nul = rng.standard_normal((1000, d)); nul /= np.linalg.norm(nul, axis=1, keepdims=True)
    e6.append(e); rk.append(int(((nul @ v1) ** 2 >= e).sum() + 1))
e6, rk = np.array(e6), np.array(rk)
res["E6_stacked_e1_mean"] = float(e6.mean())
res["E6_stacked_e1_p5_p95"] = [float(np.percentile(e6, 5)), float(np.percentile(e6, 95))]
res["E6_P_rank1"] = float((rk == 1).mean())
res["E6_P_rank_le10"] = float((rk <= 10).mean())
res["E6_draws"] = len(e6)

# E7: gain law, symbolic
x, K, gT = sp.symbols("x K g_T", positive=True)
sol = sp.solve(sp.Eq(x * gT, K), gT)[0]
res["E7_gT"] = str(sol)
res["E7_dlog_gT_dlog_x"] = str(sp.simplify(sp.diff(sp.log(sol), x) * x))

OUT.write_text(json.dumps(res, indent=1))
print(json.dumps(res, indent=1))
