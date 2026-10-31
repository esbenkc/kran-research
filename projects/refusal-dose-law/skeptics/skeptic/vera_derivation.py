"""Vera: own derivation of the post's displayed equations (sympy + numeric on the real vectors).
E1  p = mean_{prompts, t>=1} -(delta_t . r_hat)/||h_t||                      (definition; numeric match in vera_units.py)
E2  delta_on + delta_off = delta (split is exact) and delta_off . r_hat = 0
E3  built push (m - a_t)||w|| v, v = s c r_hat + sqrt(1-c^2) u, u . r_hat = 0, |u| = 1:
    ||push_t|| = |m - a_t| ||w||  (same size as the neuron's push)
    -(push_t . r_hat) = -(m - a_t)||w|| s c ; with s = sign(w . r_hat), c = |cos(w, r_hat)| this equals -(m - a_t)(w . r_hat)
    (the neuron's own push along r_hat), so TWIN has the neuron's p at every token.
E4  sigma(a + b p) = 1/(1 + exp(-(a + b p)))  (logistic; the inverse used for p_eff is (logit(y) - a)/b)
"""
import sympy as sp, numpy as np, torch
import vera_fwd as F

m, a, c, wn, s = sp.symbols('m a c w_n s', real=True)
# E3 in a 2-D basis (r_hat, u): v = (s c, sqrt(1-c^2))
v = sp.Matrix([s * c, sp.sqrt(1 - c ** 2)])
push = (m - a) * wn * v
norm2 = sp.simplify((push.T * push)[0].subs(s ** 2, 1))
assert sp.simplify(norm2 - ((m - a) * wn) ** 2) == 0, norm2
along = push[0]
w_dot_r = sp.Symbol('wr', real=True)
# s = sign(wr), c = |wr|/wn  ->  s*c*wn = wr
assert sp.simplify(along.subs(c, sp.Abs(w_dot_r) / wn).subs(s, sp.sign(w_dot_r)) - (m - a) * w_dot_r) == 0
print('E3 sympy: size and push-along-r identities hold')
# E2 numeric on the real write and my r_hat
rh = torch.from_numpy(np.load('vera_r13.npy')).double()
w = F.T('model.layers.13.mlp.down_proj.weight')[:, 3270].double()
won, woff = (w @ rh) * rh, w - (w @ rh) * rh
print('E2: |won + woff - w| =', float((won + woff - w).norm()), ' woff . r_hat =', float(woff @ rh),
      ' |won|/|w| =', float(won.norm() / w.norm()), ' energy on r =', float(won.norm() ** 2 / w.norm() ** 2))
# E4
x, A, B, y = sp.symbols('x A B y', real=True)
sig = 1 / (1 + sp.exp(-(A + B * x)))
inv = (sp.log(y / (1 - y)) - A) / B
assert sp.simplify(sig.subs(x, inv) - y) == 0
print('E4 sympy: logistic and its inverse agree')
