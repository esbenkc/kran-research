"""Iris: checks of the post's two displayed equations (no model weights, CPU, seconds).

E1  loss  L = 1/8 sum_i (s(f_i) - s(t_i))^2 + lam/2 (KL_cities + KL_web),  KL = mean per-token KL(p_original || p_tuned)
    - kl_from_post in attack_full.py equals a hand-written KL(p_ref || p) averaged over masked positions;
    - the probe term in attack_full.py (s[:k] - s[k:]).pow(2).mean() equals 1/k sum_i (s(f_i) - s(t_i))^2.
E2  KL_min = g^2 / (2 w^T F^+ w)
    - sympy: the minimiser of 1/2 d^T F d subject to w^T d = g (Lagrange) has value g^2 / (2 w^T F^-1 w), 3x3 symbolic SPD F;
    - numeric: d* = g F^+ w / (w^T F^+ w) attains it; 2000 random feasible d never go lower;
    - second-order step: for a softmax with logits W h, exact KL(p || p') / (1/2 d^T F d) -> 1 as |d| -> 0.
Run: uv run python iris_checks/check_equations.py  (from projects/full-stack-evader)
"""
import json
import sys
from pathlib import Path

import numpy as np
import sympy as sp
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from attack_full import kl_from_post  # noqa: E402

out = {}

# ---------- E1 ----------
torch.manual_seed(0)
V, d, B, T = 50, 12, 3, 7
lm = torch.nn.Linear(d, V, bias=False)
h_ref, h = torch.randn(B, T, d), torch.randn(B, T, d)
m = (torch.rand(B, T) > 0.3).float()
got = float(kl_from_post(lm, h_ref, h, m))
pr, pq = torch.softmax(lm(h_ref), -1), torch.softmax(lm(h), -1)
kl_tok = (pr * (pr.log() - pq.log())).sum(-1)
want = float((kl_tok * m).sum() / m.sum())
out["E1_kl_direction_and_mean"] = {"code": got, "hand": want, "abs_diff": abs(got - want), "pass": abs(got - want) < 1e-5}
s = torch.randn(16)
k = 8
code_term = float((s[:k] - s[k:]).pow(2).mean())
hand_term = sum(float(s[i] - s[k + i]) ** 2 for i in range(k)) / k
out["E1_probe_term"] = {"code": code_term, "hand": hand_term, "pass": abs(code_term - hand_term) < 1e-5}

# ---------- E2 symbolic ----------
a, b, c, e, f_, h_ = sp.symbols("a b c e f h", real=True)
F = sp.Matrix([[a, b, c], [b, e, f_], [c, f_, h_]])
w = sp.Matrix(sp.symbols("w1 w2 w3", real=True))
g, mu = sp.symbols("g mu", real=True)
dvec = sp.Matrix(sp.symbols("d1 d2 d3", real=True))
Lag = (dvec.T * F * dvec)[0] / 2 - mu * ((w.T * dvec)[0] - g)
sol = sp.solve([sp.diff(Lag, x) for x in dvec] + [(w.T * dvec)[0] - g], list(dvec) + [mu], dict=True)[0]
val = sp.simplify((dvec.T * F * dvec)[0].subs(sol) / 2)
target = g ** 2 / (2 * (w.T * F.inv() * w)[0])
out["E2_sympy_lagrange"] = {"pass": bool(sp.simplify(val - target) == 0)}

# ---------- E2 numeric ----------
rng = np.random.default_rng(1021)
n = 40
Wu = rng.normal(size=(300, n))
hh = rng.normal(size=n)
z = Wu @ hh
p = np.exp(z - z.max()); p /= p.sum()
Fn = Wu.T @ (np.diag(p) - np.outer(p, p)) @ Wu
wn = rng.normal(size=n)
gn = 0.7
Fp = np.linalg.pinv(Fn)
q = wn @ Fp @ wn
dstar = gn * Fp @ wn / q
val_star = 0.5 * dstar @ Fn @ dstar
pred = gn ** 2 / (2 * q)
lows = []
for _ in range(2000):
    r = rng.normal(size=n)
    r = r - (wn @ r) / (wn @ wn) * wn  # w-orthogonal part
    dd = dstar + r * rng.uniform(0, 1)
    lows.append(0.5 * dd @ Fn @ dd)
out["E2_numeric_min"] = {"value_at_dstar": float(val_star), "predicted": float(pred), "rel_err": float(abs(val_star - pred) / pred),
                         "min_over_2000_feasible": float(min(lows)), "constraint": float(wn @ dstar),
                         "pass": bool(abs(val_star - pred) / pred < 1e-8 and min(lows) >= val_star * (1 - 1e-9))}


def kl(p, z2):
    lq = z2 - z2.max(); lq = lq - np.log(np.exp(lq).sum())
    return float((p * (np.log(p) - lq)).sum())


ratios = {}
u = rng.normal(size=n); u /= np.linalg.norm(u)
for eps in (1e-1, 1e-2, 1e-3):
    dd = eps * u
    ratios[str(eps)] = float(kl(p, Wu @ (hh + dd)) / (0.5 * dd @ Fn @ dd))
out["E2_second_order"] = {"exact_over_quadratic": ratios, "pass": bool(abs(ratios["0.001"] - 1) < 1e-2)}

out["all_pass"] = all(v["pass"] for v in out.values() if isinstance(v, dict))
Path(__file__).with_suffix(".json").write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
