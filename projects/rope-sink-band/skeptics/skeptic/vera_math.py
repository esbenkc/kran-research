"""Vera: own derivation checks for rope-sink-band (sympy + numeric). Run: python vera_math.py"""
import math
import numpy as np
import sympy as sp

ok = []

# 1. RoPE on one pair is a 2-D rotation; the pair's part of the logit depends only on the offset (n - m).
m, n, th, a1, a2, b1, b2 = sp.symbols("m n theta a1 a2 b1 b2", real=True)
R = lambda ang: sp.Matrix([[sp.cos(ang), -sp.sin(ang)], [sp.sin(ang), sp.cos(ang)]])
q = sp.Matrix([a1, a2]); k = sp.Matrix([b1, b2])
lhs = (R(m * th) * q).dot(R(n * th) * k)
rhs = q.dot(R((n - m) * th) * k)
ok.append(("L1 pair term = q . R((n-m)theta) k", sp.simplify(sp.expand_trig(lhs - rhs)) == 0))

# 2. Polar form: pair term = |q||k| cos((n-m)theta + phi_k - phi_q)
rq, rk, pq, pk = sp.symbols("r_q r_k phi_q phi_k", positive=True)
qp = sp.Matrix([rq * sp.cos(pq), rq * sp.sin(pq)]); kp = sp.Matrix([rk * sp.cos(pk), rk * sp.sin(pk)])
e = (R(m * th) * qp).dot(R(n * th) * kp) - rq * rk * sp.cos((n - m) * th + pk - pq)
ok.append(("L2 polar form", sp.simplify(sp.expand_trig(e)) == 0))

# 3. Zeroing a pair before RoPE == zeroing it after RoPE (rotation maps 0 to 0 and acts inside the pair only).
z = sp.Matrix([0, 0])
ok.append(("L3 zero commutes with rotation", (R(m * th) * z) == z))

# 4. Gap additivity: logit gap = sum_j c_j with c_j = q_j.k_j(0) - mean_u q_j.k_j(u) (numeric, rotate-half layout).
rng = np.random.default_rng(0)
d, T, base = 128, 40, 1e6
inv = base ** (-np.arange(0, d, 2) / d)
ang = np.arange(T)[:, None] * inv[None]
cos = np.cos(np.concatenate([ang, ang], 1)); sin = np.sin(np.concatenate([ang, ang], 1))
rot = lambda v: v * cos + np.concatenate([-v[:, d // 2:], v[:, : d // 2]], 1) * sin  # HF rotate_half
Q = rot(rng.standard_normal((T, d))); K = rot(rng.standard_normal((T, d)))
t = T - 1
logit = Q[t] @ K[: t + 1].T
gap = logit[0] - logit[1:].mean()
pair = lambda v: v[..., : d // 2], lambda v: v[..., d // 2:]
c = Q[t, :64] * K[0, :64] + Q[t, 64:] * K[0, 64:] - (Q[t, :64] * K[1:t + 1, :64] + Q[t, 64:] * K[1:t + 1, 64:]).mean(0)
ok.append(("L4 sum_j c_j == logit gap", abs(c.sum() - gap) < 1e-9))
# rotate-half: dims (i, i+64) rotate together at speed inv[i]
v = rng.standard_normal(d); i = 37
vr = (v * cos[5] + np.concatenate([-v[64:], v[:64]]) * sin[5])
ok.append(("L5 pair (i,i+64) rotates by 5*inv[i]", np.allclose([vr[i], vr[i + 64]], np.array(
    [[math.cos(5 * inv[i]), -math.sin(5 * inv[i])], [math.sin(5 * inv[i]), math.cos(5 * inv[i])]]) @ [v[i], v[i + 64]])))
ok.append(("L5b pair 0 fastest (1 rad/token), 63 slowest", inv[0] == 1.0 and inv.argmin() == 63))

# 5. Location bound: theta_i * L < 2 pi  <=>  i > 64 ln(L/2pi)/ln(base)
for nm, b, L, want in [("a", 1e6, 512, 21), ("c", 1e6, 32768, 40), ("b", 1e4, 4096, 46)]:
    edge = 64 * math.log(L / (2 * math.pi)) / math.log(b)
    first = min(i for i in range(64) if b ** (-i / 64) * L < 2 * math.pi)
    ok.append((f"L6{nm} base {b:g} L {L}: edge {edge:.2f}, first pair {first}", first == want and first == math.floor(edge) + 1))

# 6. Kasper's invariant: token 0 sees only itself, softmax over one key = 1 for ANY logit edit.
x = sp.symbols("x", real=True)
ok.append(("L7 softmax of one key = 1", sp.simplify(sp.exp(x) / sp.exp(x)) == 1))

# 7. chance rates for location edges (random k=8 subset inside the allowed region)
for nm, n_ok in [("a", 43), ("c", 24), ("b", 18)]:
    print(f"chance random 8-set inside edge {nm}: C({n_ok},8)/C(64,8) = {math.comb(n_ok, 8) / math.comb(64, 8):.3g}")
for nm, v in ok:
    print(("PASS " if v else "FAIL ") + nm)
assert all(v for _, v in ok)
