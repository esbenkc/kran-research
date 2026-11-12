"""Checks every displayed equation in _notes/optimizer-cue-size.md (Selma, write).

1. The three optimizer steps: torch's SGD, sign step and AdamW (wd 0) equal the displayed formulas.
2. The pooled fit  log T = a_opt + b_opt log x + e  on the 24 headline runs reproduces data/figure_data.json.
3. sympy: under that law, halving x multiplies T by 2**(-b); the per-halving factors quoted in the post.

Run: ../../../env/.venv/bin/python post/eq_check.py   (from the project folder)
"""
import json
from pathlib import Path

import numpy as np
import sympy as sp
import torch

HERE = Path(__file__).resolve().parent.parent
torch.manual_seed(0)
ok = True


def check(name, cond, detail=""):
    global ok
    ok &= bool(cond)
    print(("PASS" if cond else "FAIL"), name, detail)


# 1. optimizer steps --------------------------------------------------------------------------
eta = 1e-2
grads = [torch.randn(50) for _ in range(5)]

p = torch.zeros(50, requires_grad=True)
opt = torch.optim.SGD([p], lr=eta, momentum=0)
p.grad = grads[0].clone(); opt.step()
check("SGD step = -eta g", torch.allclose(p.detach(), -eta * grads[0], atol=1e-7))

p = torch.zeros(50, requires_grad=True)
opt = torch.optim.SGD([p], lr=eta, momentum=0)
p.grad = grads[0].clone(); p.grad.sign_(); opt.step()  # rl.py --opt sign: grad.sign_() then plain SGD step
check("sign step = -eta sign(g)", torch.allclose(p.detach(), -eta * torch.sign(grads[0]), atol=1e-7))

b1, b2, eps = 0.9, 0.999, 1e-8
p = torch.zeros(50, dtype=torch.float64, requires_grad=True)
opt = torch.optim.AdamW([p], lr=eta, betas=(b1, b2), eps=eps, weight_decay=0.0)
m = torch.zeros(50, dtype=torch.float64); v = torch.zeros(50, dtype=torch.float64); theta = torch.zeros(50, dtype=torch.float64)
for t, g in enumerate(grads, start=1):
    g = g.double()
    p.grad = g.clone(); opt.step()
    m = b1 * m + (1 - b1) * g; v = b2 * v + (1 - b2) * g * g
    mhat = m / (1 - b1 ** t); vhat = v / (1 - b2 ** t)
    theta = theta - eta * mhat / (vhat.sqrt() + eps)
check("AdamW step = -eta mhat/(sqrt(vhat)+eps), 5 steps", torch.allclose(p.detach(), theta, atol=1e-12),
      f"max diff {float((p.detach() - theta).abs().max()):.2e}")
# Adam's first step is eta*sign(g) up to eps: the sign-SGD lr 1e-4 matches it
g0 = grads[0].double()
first = (g0 * (1 - b1) / (1 - b1)) / ((g0 * g0 * (1 - b2) / (1 - b2)).sqrt() + eps)
check("Adam step 1 = sign(g) (eps aside)", torch.allclose(first, torch.sign(g0), atol=1e-6))

# 2. pooled fit ---------------------------------------------------------------------------------
fd = json.loads((HERE / "data/figure_data.json").read_text())
rows = [r for r in fd["runs"] if r["opt"] in ("adamw", "sgd", "sign")]
opts = ["adamw", "sgd", "sign"]
X, y = [], []
for r in rows:
    if r.get("censored"):
        continue
    oh = [1.0 if r["opt"] == o else 0.0 for o in opts]
    X.append(oh + [oh[i] * np.log(r["x"]) for i in range(3)]); y.append(np.log(r["T"]))
X, y = np.array(X), np.array(y)
check("24 uncensored headline runs", len(y) == 24, f"n={len(y)}")
coef, *_ = np.linalg.lstsq(X, y, rcond=None)
for i, o in enumerate(opts):
    check(f"slope {o}", abs(coef[3 + i] - fd["fit"][o]["slope"]) < 1e-9, f"{coef[3 + i]:.4f}")
    check(f"intercept {o}", abs(coef[i] - fd["fit"][o]["intercept"]) < 1e-9, f"{coef[i]:.4f}")
d_sgd, d_sign = coef[4] - coef[3], coef[5] - coef[3]
print(f"     d_SGD {d_sgd:.3f}  d_sign {d_sign:.3f}")
check("d_SGD = -0.673", abs(d_sgd - (-0.673)) < 5e-4)
check("d_sign = -0.185", abs(d_sign - (-0.185)) < 5e-4)

# 3. per-halving factor -------------------------------------------------------------------------
a, b, x = sp.symbols("a b x", positive=True)
T = sp.exp(a + b * sp.log(x))
ratio = sp.simplify(T.subs(x, x / 2) / T)
check("T(x/2)/T(x) = 2**(-b)", sp.simplify(ratio - 2 ** (-b)) == 0, str(ratio))
for o, quoted in [("sgd", 2.8), ("adamw", 1.8), ("sign", 2.0)]:
    f = 2 ** (-fd["fit"][o]["slope"])
    check(f"per-halving {o} = x{quoted}", round(f, 1) == quoted, f"{f:.3f}")

print("ALL PASS" if ok else "SOME FAIL")
