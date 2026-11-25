"""Vera's own derivation checks of the post's equations (independent of research/invariance.py and checks/equations.py).
E1 ULA stationary variance with tether, and the per-direction regular term 1/2 * 1/(1+g) / (1-c/4).
E2 Rescaling w = nbeta^(-1/4) u of the null block: nbeta*K0 + gamma/2|w|^2 -> 1/2|B_u A_u|^2 + G/2|u|^2, G = gamma/sqrt(nbeta).
E3 Strong-spring (Gaussian) limit nbeta*E[K0] = q*M'N'H' with q = nbeta/(2 gamma^2); numeric MC check.
E4 Under w -> s w: eps ~ s^2, gamma ~ s^-2, nbeta*L invariant; X = eps*nbeta/gamma ~ s^4 (needs L ~ fixed), q ~ ?
"""
import numpy as np
import sympy as sp

# E1
eps, a, h, nb, g = sp.symbols('epsilon a h nbeta gamma', positive=True)
V = sp.symbols('V', positive=True)
# w' = (1 - eps a/2) w + sqrt(eps) xi ; stationary V = (1-eps a/2)^2 V + eps
Vsol = sp.solve(sp.Eq(V, (1 - eps * a / 2) ** 2 * V + eps), V)[0]
assert sp.simplify(Vsol - 1 / (a * (1 - eps * a / 4))) == 0
term = sp.simplify(nb * h * Vsol / 2)  # nbeta * E[1/2 h w^2]
gg = g / (nb * h); cc = eps * (nb * h + g)
law = sp.Rational(1, 2) / (1 + gg) / (1 - cc / 4)
assert sp.simplify(term.subs(a, nb * h + g) - law) == 0
print("E1 ok: Var = 1/(a(1-eps a/4)); nbeta*E[h w^2/2] = 1/2 * 1/(1+g) / (1-c/4)")

# E2 (symbolic on scalar block entries)
b_, a_, w = sp.symbols('b a w', real=True)
K0 = sp.Rational(1, 2) * (b_ * a_) ** 2
u1, u2 = sp.symbols('u1 u2', real=True)
expr = nb * K0.subs({b_: nb ** sp.Rational(-1, 4) * u1, a_: nb ** sp.Rational(-1, 4) * u2}) + g / 2 * (nb ** sp.Rational(-1, 2)) * (u1 ** 2 + u2 ** 2)
target = sp.Rational(1, 2) * (u1 * u2) ** 2 + (g / sp.sqrt(nb)) / 2 * (u1 ** 2 + u2 ** 2)
assert sp.simplify(expr - target) == 0
print("E2 ok: null-block energy depends on (nbeta, gamma) only via G = gamma/sqrt(nbeta); q = nbeta/(2 gamma^2) = 1/(2 G^2)")

# E3 numeric
rng = np.random.default_rng(0)
for (Mp, Np, Hp, nbv, gv) in [(31, 31, 15, 1000.0, 300.0), (20, 36, 8, 100.0, 50.0)]:
    vals = []
    for _ in range(4000):
        B = rng.standard_normal((Np, Hp)) / np.sqrt(gv); A = rng.standard_normal((Hp, Mp)) / np.sqrt(gv)
        vals.append(nbv * 0.5 * np.sum((B @ A) ** 2))
    q = nbv / (2 * gv ** 2)
    pred = q * Mp * Np * Hp
    print(f"E3 ({Mp},{Np},{Hp}) nbeta={nbv} gamma={gv}: MC {np.mean(vals):.4f} +- {np.std(vals)/np.sqrt(len(vals)):.4f} vs q*M'N'H' {pred:.4f}")
    assert abs(np.mean(vals) - pred) < 4 * np.std(vals) / np.sqrt(len(vals))

# E4 scaling: substitute w = s*v. Loss L(w) = Lt(v). SGLD in v: eps_v = eps/s^2, gamma_v = gamma*s^2, nbeta same.
s = sp.symbols('s', positive=True)
eps_v, gam_v = eps / s ** 2, g * s ** 2
X_ratio = sp.simplify((eps_v * nb / gam_v) / (eps * nb / g))
print("E4 X scales as", X_ratio, "(s^-4 for v = w/s, i.e. s^4 the other way) -> not scale-free")
# q in RRR: K0 quartic coefficient k4 also rescales: K0(w)=k4 |b a|^2 -> k4 s^4 |b_v a_v|^2, so q = nbeta*k4/gamma^2 -> nbeta*k4*s^4/(gamma^2 s^4) invariant
k4 = sp.symbols('k4', positive=True)
q_ratio = sp.simplify((nb * k4 * s ** 4 / gam_v ** 2) / (nb * k4 / g ** 2))
assert q_ratio == 1
print("E4 q = nbeta*k4/gamma^2 is invariant once k4 carries its s^4:", q_ratio)
