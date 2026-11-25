"""Iris: sympy checks for phase 0 (theory-1125-llc-bias-law).

1. Stationary variance of SGLD (unadjusted Langevin, devinterp/Lau update) on a 1-D quadratic
   direction:  x' = x - (eps/2) * a * x + sqrt(eps) * xi,  a = nbeta*h + gamma.
2. Per-direction LLC estimate  nbeta * h/2 * Var  versus the truth 1/2.
3. Scaling of each candidate group under the reparametrisation w -> s*w.
"""
import sympy as sp

eps, a, h, nb, g, s, x, T, k4 = sp.symbols("epsilon a h nbeta gamma s x T kappa4", positive=True)

# 1. stationary variance: V = (1 - eps a/2)^2 V + eps
V = sp.symbols("V", positive=True)
Vsol = sp.solve(sp.Eq(V, (1 - eps * a / 2) ** 2 * V + eps), V)[0]
target = 1 / (a * (1 - eps * a / 4))
assert sp.simplify(Vsol - target) == 0
print("stationary Var =", sp.simplify(Vsol))

# 2. per-direction estimate, truth 1/2
lam_dir = sp.simplify(nb * h / 2 * Vsol.subs(a, nb * h + g))
rel = sp.simplify(lam_dir / sp.Rational(1, 2) - 1)
print("per-direction lambda_hat =", lam_dir)
ser = sp.series(rel.subs({g: sp.Symbol("u") * nb * h, eps: sp.Symbol("c") / (nb * h)}), sp.Symbol("u"), 0, 2)
print("rel err, u = gamma/(nbeta h), c = eps nbeta h:", sp.simplify(rel.subs({g: sp.Symbol("u") * nb * h, eps: sp.Symbol("c") / (nb * h)})))

# 3. reparametrisation w -> s*w (new coordinate w' = s*w): a quadratic weight q*|w|^2 becomes q/s^2*|w'|^2,
#    so curvatures h, gamma scale s^-2; the step eps is a variance in weight units: s^2;
#    nbeta multiplies the loss value, which does not change: s^0. Quartic coefficient kappa4: s^-4.
scale = {eps: s**2, g: s**-2, h: s**-2, nb: 1, T: 1, k4: s**-4}
groups = {
    "eps*nbeta/gamma (goal)": eps * nb / g,
    "eps*gamma": eps * g,
    "eps*nbeta*h": eps * nb * h,
    "gamma/(nbeta*h)": g / (nb * h),
    "T*eps*sqrt(nbeta*kappa4)": T * eps * sp.sqrt(nb * k4),
}
for name, expr in groups.items():
    fac = sp.simplify(expr.subs({v: v * f for v, f in scale.items()}, simultaneous=True) / expr)
    print(f"{name:28s} scales as {fac}")
