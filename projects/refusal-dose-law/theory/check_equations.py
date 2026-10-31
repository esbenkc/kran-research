"""Checks every displayed equation in the post _notes/refusal-dose-law.md (Selma, write).

Run from projects/refusal-dose-law:
    uv run --no-project --with torch --with numpy --with sympy python theory/check_equations.py

1. Split:  delta_on + delta_off = delta, and delta_off has zero part along r_hat   (sympy + saved vectors)
2. Built push:  ||s c r_hat + sqrt(1-c^2) u|| = 1 when u is a unit vector at right angles to r_hat   (sympy + saved vectors)
3. Dose number p:  mean over prompts of mean over tokens t>=1 of -(delta_t . r_hat)/||h_t||   (recomputed from the clean pass)
4. Curve:  sigma(a + b p) = 1/(1+exp(-(a+b p))), and the gaps it gives for neuron 3270   (sympy + results.json)
"""
import json
import math
from pathlib import Path

import numpy as np
import sympy as sp
import torch

HERE = Path(__file__).resolve().parent.parent
D = HERE / "data"
ok = True


def check(name, cond, detail):
    global ok
    ok &= bool(cond)
    print(("PASS " if cond else "FAIL ") + name + ": " + detail)


# 1. Split, symbolic in 3 dimensions with a unit r_hat
m, a = sp.symbols("m a", real=True)
w = sp.Matrix(sp.symbols("w1:4", real=True))
th, ph = sp.symbols("theta phi", real=True)
rh = sp.Matrix([sp.sin(th) * sp.cos(ph), sp.sin(th) * sp.sin(ph), sp.cos(th)])  # unit vector
d_full = (m - a) * w
d_on = (m - a) * (w.dot(rh)) * rh
d_off = (m - a) * (w - (w.dot(rh)) * rh)
check("split adds up (sympy)", sp.simplify(d_on + d_off - d_full) == sp.zeros(3, 1), "delta_on + delta_off - delta = 0")
check("rest has zero part along r_hat (sympy)", sp.simplify(d_off.dot(rh)) == 0, "delta_off . r_hat = 0")

# 2. Built push has length 1
c, s = sp.symbols("c s", real=True)
rr, uu, ru = sp.symbols("rr uu ru", real=True)  # r.r, u.u, r.u
norm2 = (s * c) ** 2 * rr + (1 - c**2) * uu + 2 * s * c * sp.sqrt(1 - c**2) * ru
check("built push has unit length (sympy)",
      sp.simplify(norm2.subs({rr: 1, uu: 1, ru: 0, s: 1})) == 1 and sp.simplify(norm2.subs({rr: 1, uu: 1, ru: 0, s: -1})) == 1,
      "|v|^2 = c^2 + (1 - c^2) = 1 for s = +1 or -1")

# saved vectors (phase0.py): the unit's write direction times ||w||
units = torch.load(D / "phase0" / "units.pt", weights_only=False)
r_hat = torch.tensor(np.load(D / "phase0" / "r13.npy")).float()
r_hat = r_hat / r_hat.norm()
G, Gon, Goff = (units[k][1].float() for k in ("G", "G_on", "G_off"))
check("split adds up (saved vectors)", (Gon + Goff - G).abs().max().item() < 1e-5,
      f"max |G_on + G_off - G| = {(Gon + Goff - G).abs().max().item():.2e}")
check("rest has zero part along r_hat (saved)", abs((Goff @ r_hat).item()) / G.norm().item() < 1e-4,
      f"cos(G_off, r_hat) = {(Goff @ r_hat).item() / Goff.norm().item():.2e}")
cos_g = (G @ r_hat).item() / G.norm().item()
check("neuron 3270 cosine", abs(cos_g + 0.3755) < 5e-4, f"cos(w, r_hat) = {cos_g:.4f} (post: -0.375)")
for k in ("TWIN_0", "TWIN_1", "TWIN_2"):
    v = units[k][1].float()
    check(f"{k} same size and same part along r_hat", abs(v.norm().item() - G.norm().item()) < 1e-3 and abs((v @ r_hat).item() - (G @ r_hat).item()) < 1e-3,
          f"norm {v.norm().item():.4f} vs {G.norm().item():.4f}, along r {(v @ r_hat).item():.4f} vs {(G @ r_hat).item():.4f}")
for cc in ("0.05", "0.1", "0.2", "0.6", "1.0"):
    v = units[f"D_{cc}"][1].float()
    cv = abs((v @ r_hat).item()) / v.norm().item()
    check(f"built push c={cc}", abs(cv - float(cc)) < 1e-3 and abs(v.norm().item() - G.norm().item()) < 1e-3, f"|cos| {cv:.4f}, norm {v.norm().item():.4f}")

# 3. Dose number p for neuron 3270, recomputed from the clean pass
stats = torch.load(D / "phase0" / "prompt_stats.pt", weights_only=False)["stats"]
ptab = json.load(open(D / "phase0" / "p_table.json"))
for mm in (6, 12, 18, 24, 30):
    per = []
    for st in stats:
        coef = mm - st["a_g"].float()
        per.append((-(coef * (G @ r_hat)) / st["hn"].float()).mean().item())
    mine = float(np.mean(per))
    ref = ptab[f"G@{mm}"]["p"]
    ref = float(np.mean(ref)) if isinstance(ref, list) else float(ref)
    check(f"p for neuron 3270 at pin {mm}", abs(mine - ref) < 1e-4, f"recomputed {mine:.4f}, p_table {ref:.4f}")

# 4. Curve and gaps
A, B, P = sp.symbols("alpha beta p", real=True)
check("logistic form (sympy)", sp.simplify(1 / (1 + sp.exp(-(A + B * P))) - sp.exp(A + B * P) / (1 + sp.exp(A + B * P))) == 0,
      "1/(1+e^-(a+bp)) = e^(a+bp)/(1+e^(a+bp))")
res = json.load(open(D / "results.json"))
fit = res["labelings"]["comply"]["point"]["fit"]
alpha, beta = fit["theta"]
check("curve parameters", abs(alpha + 1.783) < 1e-3 and abs(beta - 13.94) < 1e-2, f"alpha {alpha:.4f}, beta {beta:.4f}")
gate_y = {6: 30.5, 12: 37.0, 18: 47.5, 24: 60.0, 30: 71.0}
for mm, y in gate_y.items():
    ref = ptab[f"G@{mm}"]["p"]
    pp = float(np.mean(ref)) if isinstance(ref, list) else float(ref)
    gap = y - 100 / (1 + math.exp(-(alpha + beta * pp)))
    print(f"      pin {mm}: p {pp:.4f}, curve {y - gap:.1f}%, gate {y}%, gap {gap:+.1f} points")
    if mm in (6, 30):
        target = {6: 12.6, 30: 38.3}[mm]
        check(f"gap at pin {mm}", abs(gap - target) < 0.15, f"{gap:+.2f} (post: {target:+.0f} rounded to {round(target)})")

print("ALL PASS" if ok else "SOME CHECKS FAILED")
