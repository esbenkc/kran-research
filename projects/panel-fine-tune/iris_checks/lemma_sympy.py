"""Iris: symbolic checks of the readout-28 lemma equations (theory-1022-panel-max-attack).
(1) RMSNorm(c x) = RMSNorm(x) for c > 0 (eps = 0), any dimension: checked symbolically for d = 4.
(2) affine probe on z-scored inputs: s(c x) = c a + b0 with a = sum(w_i x_i / sd_i), b0 = b - sum(w_i mu_i / sd_i).
(3) the hiding factor c* = (t - b0)/a solves s(c* x) = t; for a > 0 and b0 < t < s(x), 0 < c* < 1.
(4) the inflating factor for a true statement, c = (tau - b0)/a > 1 when a > 0 and s(x) < tau.
Run: uv run --with sympy python lemma_sympy.py
"""
import sympy as sp

d = 4
c = sp.symbols('c', positive=True)
x = sp.symbols('x0:%d' % d, real=True)
g = sp.symbols('g0:%d' % d, real=True)
w = sp.symbols('w0:%d' % d, real=True)
mu = sp.symbols('m0:%d' % d, real=True)
sd = sp.symbols('s0:%d' % d, positive=True)
b, t, tau = sp.symbols('b t tau', real=True)

rms = lambda v: sp.sqrt(sum(vi ** 2 for vi in v) / d)
norm = lambda v: [g[i] * v[i] / rms(v) for i in range(d)]
cx = [c * xi for xi in x]
ok1 = all(sp.simplify(a_ - b_) == 0 for a_, b_ in zip(norm(cx), norm(x)))

s = lambda v: sum(w[i] * (v[i] - mu[i]) / sd[i] for i in range(d)) + b
a = sum(w[i] * x[i] / sd[i] for i in range(d))
b0 = b - sum(w[i] * mu[i] / sd[i] for i in range(d))
ok2 = sp.simplify(s(cx) - (c * a + b0)) == 0

cstar = (t - b0) / a
ok3 = sp.simplify(s([cstar * xi for xi in x]) - t) == 0
# 0 < c* < 1 iff b0 < t < c*1*a + b0 = s(x) when a > 0: c* - 1 = (t - s(x))/a
ok3b = sp.simplify((cstar - 1) - (t - s(x)) / a) == 0
cin = (tau - b0) / a
ok4 = sp.simplify((cin - 1) - (tau - s(x)) / a) == 0
print({'rmsnorm_scale_invariant': ok1, 'affine_in_c': ok2, 'cstar_hits_line': ok3,
       'cstar_minus_1_sign': ok3b, 'inflate_minus_1_sign': ok4})
assert ok1 and ok2 and ok3 and ok3b and ok4
