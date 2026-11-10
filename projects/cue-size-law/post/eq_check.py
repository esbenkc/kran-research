"""Checks every displayed equation in _notes/cue-size-law.md (Selma, writer). Run: python post/eq_check.py
from projects/cue-size-law. Exits non-zero on any failed assert.

Eq. 1 (plain SGD): the weight along the cue grows by eta*c*alpha per update, and the gap moves by alpha per unit
weight, so gap(t) = eta*c*alpha^2*t and T = z/(eta*c*alpha^2): halving alpha multiplies T by 4.
Eq. 2 (sign-like Adam): each update moves the weight by about eta whatever the gradient size, so gap(t) = eta*alpha*t
and T = z/(eta*alpha): halving alpha multiplies T by 2.
Eq. 3 (the fit): log T = b0 + b1 log x, natural logs, OLS over the 12 frozen-grid sweep runs in data/summary.json.
"""
import json
import math
from pathlib import Path

import sympy as sp

eta, c, alpha, t, z = sp.symbols("eta c alpha t z", positive=True)

# Eq. 1
w_sgd = eta * c * alpha * t
gap_sgd = alpha * w_sgd
T_sgd = sp.solve(sp.Eq(gap_sgd, z), t)[0]
assert sp.simplify(T_sgd - z / (eta * c * alpha**2)) == 0
assert sp.simplify(T_sgd.subs(alpha, alpha / 2) / T_sgd) == 4
assert sp.simplify(sp.diff(sp.log(T_sgd), alpha) * alpha) == -2  # log-log slope

# Eq. 2
w_sign = eta * t
gap_sign = alpha * w_sign
T_sign = sp.solve(sp.Eq(gap_sign, z), t)[0]
assert sp.simplify(T_sign - z / (eta * alpha)) == 0
assert sp.simplify(T_sign.subs(alpha, alpha / 2) / T_sign) == 2
assert sp.simplify(sp.diff(sp.log(T_sign), alpha) * alpha) == -1

# Eq. 3: refit from the raw summary and check the numbers the post prints.
root = Path(__file__).resolve().parents[1]
s = json.loads((root / "data/summary.json").read_text())
runs = [r for r in s["runs"] if not r["censored"]]
assert len(runs) == 12, len(runs)
X = [math.log(r["x"]) for r in runs]
Y = [math.log(r["T"]) for r in runs]
mx, my = sum(X) / len(X), sum(Y) / len(Y)
b1 = sum((a - mx) * (b - my) for a, b in zip(X, Y)) / sum((a - mx) ** 2 for a in X)
b0 = my - b1 * mx
assert abs(b1 - (-0.90)) < 0.005 and abs(b0 - 6.56) < 0.005, (b0, b1)
assert abs(2 ** -b1 - 1.87) < 0.01  # factor per halving of the cue
norm = json.loads((root / "data/scan.json").read_text())["n_norm_rows012"]
x_real = s["x_real"]
T_real_pred = math.exp(b0 + b1 * math.log(x_real))
assert abs(T_real_pred - 77) < 1, T_real_pred
assert abs(100 * x_real / norm - 1.9) < 0.05
print(f"ok: b0 {b0:.4f}, b1 {b1:.4f}, factor per halving {2 ** -b1:.3f}, T_pred(x_real) {T_real_pred:.1f}")
