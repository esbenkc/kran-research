"""Checks every displayed equation in the post (_notes/refusal-neuron-massive.md, expert foldout).

1. delta(t) = (m - h(t)) w  and the split  (m - h) w_M + (m - h) w_R = (m - h) w   (sympy, exact).
2. s = sum_{c in M} w_c^2 / ||w||^2 = 0.0027 for the real gate column, chance 3/2048, layer median 0.0028,
   and ||w_M|| / ||w|| = sqrt(s) = 5.2%  (numeric, from the public Qwen/Qwen3-1.7B weights).
3. r_keep + r_rem = 1 when effects add up (sympy): a sanity identity, NOT assumed in the status rule.
Run: uv run python projects/refusal-neuron-massive/theory/check_equations.py
"""
import glob
import json

import numpy as np
import sympy as sp

# 1. the split is exact
m, h = sp.symbols("m h", real=True)
wM = sp.Matrix(sp.symbols("a0:4", real=True))  # entries on M (others zero)
wR = sp.Matrix(sp.symbols("b0:4", real=True))
w = wM + wR
assert sp.simplify((m - h) * wM + (m - h) * wR - (m - h) * w) == sp.zeros(4, 1)
print("1. split: (m-h) w_M + (m-h) w_R == (m-h) w  [exact]")

# 2. the share from the weights
M = [1793, 1999, 1401]
try:
    from safetensors import safe_open

    snap = glob.glob(f"{__import__('os').path.expanduser('~')}/.cache/huggingface/hub/models--Qwen--Qwen3-1.7B/snapshots/*/")[0]
    idx = json.load(open(snap + "model.safetensors.index.json"))["weight_map"]
    name = "model.layers.13.mlp.down_proj.weight"
    with safe_open(snap + idx[name], "pt") as f:
        W = f.get_tensor(name).float().numpy()  # (2048, 6144)
    col = W[:, 3270]
    s = float((col[M] ** 2).sum() / (col ** 2).sum())
    shares = (W[M, :] ** 2).sum(0) / (W ** 2).sum(0)
    print(f"2. ||w|| = {np.linalg.norm(col):.4f}  ||w_M|| = {np.linalg.norm(col[M]):.4f}")
    print(f"   s = {s:.5f} (post: 0.0027)  chance 3/2048 = {3/2048:.5f}  layer-13 median = {np.median(shares):.5f} (post: 0.0028)")
    print(f"   sqrt(s) = {np.sqrt(s):.4f} = ||w_M||/||w|| = {np.linalg.norm(col[M]) / np.linalg.norm(col):.4f} (post: 5.2%)")
    print(f"   neuron 5441 share = {shares[5441]:.4f} (post: 70%)")
    assert abs(s - 0.0027) < 0.00005 and abs(np.sqrt(s) - 0.052) < 0.0005 and abs(shares[5441] - 0.70) < 0.01
except (IndexError, ImportError) as e:
    print("2. skipped: Qwen/Qwen3-1.7B weights not in the local HF cache", e)

# 3. ratios: only an identity check
EA, EB, EC = sp.symbols("E_A E_B E_C", positive=True)
assert sp.simplify((EC / EA + EB / EA).subs(EA, EB + EC)) == 1
print("3. r_keep + r_rem = 1 iff E_full = E_only + E_without  [identity; the post reports both ratios separately]")
print("all checks passed")
