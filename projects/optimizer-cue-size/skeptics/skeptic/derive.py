"""Vera: own derivation of the drift exponents the post uses (sympy, asserts).

Model: the switch needs the cue-conditional gap g = x * w to reach a fixed level z (x = cue size, w = the weight the
cue is read through). The reward gradient on w is proportional to x: dL/dw = c * x.
  SGD:       w_{t+1} = w_t + eta * c * x           -> T = z / (eta * c * x^2)       slope -2
  sign-SGD:  w_{t+1} = w_t + eta * sign(c * x)     -> T = z / (eta * x)             slope -1
  Adam (high per-coordinate SNR, m/sqrt(v) -> sign): same as sign-SGD                slope -1
  sign/Adam at LOW SNR: E[sign(c x + noise)] ~ sqrt(2/pi) c x / sigma -> T ~ sigma z / (eta c x^2): slope -2 again.
"""
import sympy as sp

x, eta, c, z, sigma = sp.symbols("x eta c z sigma", positive=True)
T = sp.symbols("T", positive=True)


def slope(expr):
    return sp.simplify(sp.diff(sp.log(expr), x) * x)


T_sgd = sp.solve(sp.Eq(x * eta * c * x * T, z), T)[0]
T_sign = sp.solve(sp.Eq(x * eta * T, z), T)[0]
T_sign_low = sp.solve(sp.Eq(x * eta * sp.sqrt(2 / sp.pi) * c * x / sigma * T, z), T)[0]
assert sp.simplify(T_sgd - z / (eta * c * x**2)) == 0 and slope(T_sgd) == -2
assert sp.simplify(T_sign - z / (eta * x)) == 0 and slope(T_sign) == -1
assert slope(T_sign_low) == -2
# low-SNR expansion of E[sign(mu + sigma*eps)] = erf(mu / (sigma sqrt 2)) -> sqrt(2/pi) mu / sigma
mu = sp.symbols("mu", positive=True)
lead = sp.series(sp.erf(mu / (sigma * sp.sqrt(2))), mu, 0, 2).removeO()
assert sp.simplify(lead - sp.sqrt(2 / sp.pi) * mu / sigma) == 0
# SGD lr dependence: T ~ 1/eta, so halving eta doubles T (the lr/2 control's drift band [1.6, 2.5] brackets 2)
assert sp.simplify(T_sgd.subs(eta, eta / 2) / T_sgd) == 2
# the measured fit: ln T = a + b ln x ; a halving of x multiplies T by 2^-b
for b in (-0.811, -1.484, -0.996):
    print(f"slope {b}: x{2 ** -b:.2f} per halving")
print("all sympy asserts pass")
