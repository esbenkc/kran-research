"""Vera's own derivation of the switch-time exponent k in T ~ x^k, per optimizer and parameterisation.

Setup: logit gap between cue-on and cue-off inputs is dz = f(theta) evaluated on the cue difference c, |c| = x.
Switch when dz reaches a fixed dz* (a 10-point probability gap at a fixed base rate is a fixed logit gap
to first order). Signal: the reward-correlated part of the per-sample gradient on the cue path is
proportional to the cue (k0 * c), with k0 independent of x at small dz. Noise: set by the base residual,
independent of x.

Each case writes dz(t) to leading order in t, solves dz(T) = dz*, and reads off d log T / d log x.
"""
import sympy as sp

t, x, eta, k0, s, r, a0, dzs = sp.symbols("t x eta k0 s r a0 dzstar", positive=True)

cases = {}
# 1) SGD, one weight vector w: dw_c/dt = eta*k0*x  ->  dz = w_c * x
cases["sgd_single"] = eta * k0 * x * t * x
# 2) Adam, noise-dominated: step = eta*E[g]/rms(g) = eta*k0*x/s per unit of cue direction -> dz = (eta k0 x t / s) * x
cases["adam_noise_single"] = eta * k0 * x * t / s * x
# 3) Adam, sign-like (signal-dominated): every coordinate on the cue moves eta per step,
#    dz = sum_i eta t |c_i| = eta t |c|_1 ; |c|_1 = a0 * x for a fixed direction
cases["adam_sign_single"] = eta * t * a0 * x
# 4) SGD, LoRA B@A with B=0, A0 fixed (|A0 c| = a0 x): dB/dt = eta k0 A0 c -> B = eta k0 a0 x t
#    early dz = B . A0 c = eta k0 a0^2 x^2 t ; A learns via dA/dt = eta k0 B c^T -> A c gains eta^2 k0^2 a0 x^3 t^2/2
#    late dz = B . dA c ~ eta^3 k0^3 a0^2 x^4 t^3 / 2
cases["sgd_lora_early"] = eta * k0 * a0**2 * x**2 * t
cases["sgd_lora_late"] = eta**3 * k0**3 * a0**2 * x**4 * t**3 / 2
# 5) Adam sign-like, LoRA: B entries move eta per step -> |B| ~ eta t sqrt(r); early dz = B . A0 c ~ eta t a0 x
#    A entries on the cue move eta per step -> dA c ~ eta t |c|_1 = eta t a0 x per rank; late dz = B . dA c ~ r eta^2 t^2 a0 x
cases["adam_sign_lora_early"] = eta * t * a0 * x
cases["adam_sign_lora_late"] = r * eta**2 * t**2 * a0 * x

out = {}
for name, dz in cases.items():
    T = sp.solve(sp.Eq(dz, dzs), t)
    T = [sol for sol in T if sol.is_positive][0]
    k = sp.simplify(sp.diff(sp.log(T), x) * x)
    out[name] = (sp.simplify(T), k)
    print(f"{name:24s} T = {sp.simplify(T)}   exponent d logT / d log x = {k}")

expected = {"sgd_single": -2, "adam_noise_single": -2, "adam_sign_single": -1, "sgd_lora_early": -2,
            "sgd_lora_late": sp.Rational(-4, 3), "adam_sign_lora_early": -1, "adam_sign_lora_late": sp.Rational(-1, 2)}
for n, e in expected.items():
    assert out[n][1] == e, (n, out[n][1], e)
print("all exponents match the hand derivation")
