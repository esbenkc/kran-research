"""Vera: compare 1021 clean acts (readouts 0..28) to the 1020 clean memmap, and check readout 29 = RMSNorm(readout 28).
Light job: memmaps, chunked, < 1 GB.
    python3 vera_gate.py   (needs numpy only; final-norm weight read with safetensors if available)
"""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
A = HERE.parents[1] / "kran-research/projects/full-stack-evader/data/acts/clean/tok.npy"
B = Path("/Users/esben/work/theory-1020-megaprobe-after/kran-research/projects/megaprobe-after/data/acts/clean/tok.npy")
n_tok, d = 29046, 1536
a = np.memmap(A, np.float16, "r", offset=128, shape=(n_tok, 30, d))
b = np.memmap(B, np.float16, "r", offset=128, shape=(n_tok, 29, d))
rng = np.random.default_rng(1021)
rows = np.sort(rng.choice(n_tok, 3000, replace=False))
out = {}
for l in range(29):
    x = a[rows, l].astype(np.float64); y = b[rows, l].astype(np.float64)
    ne = (a[rows, l] != b[rows, l]).mean()
    cos = (x * y).sum(1) / (np.linalg.norm(x, axis=1) * np.linalg.norm(y, axis=1) + 1e-12)
    rel = np.linalg.norm(x - y, axis=1) / (np.linalg.norm(y, axis=1) + 1e-12)
    out[l] = {"frac_neq": float(ne), "cos_min": float(cos.min()), "rel_med": float(np.median(rel)), "rel_max": float(rel.max())}
    print(l, out[l], flush=True)
# readout 29 vs RMSNorm(readout 28) with the model's final-norm weight
try:
    from safetensors import safe_open
    snap = next(Path.home().glob(".cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/*/model.safetensors"))
    with safe_open(str(snap), "pt") as f:
        g = f.get_tensor("model.norm.weight").float().numpy().astype(np.float64)
    x = a[rows, 28].astype(np.float64)
    nx = x / np.sqrt((x ** 2).mean(1, keepdims=True) + 1e-6) * g
    y = a[rows, 29].astype(np.float64)
    cos = (nx * y).sum(1) / (np.linalg.norm(nx, axis=1) * np.linalg.norm(y, axis=1))
    rel = np.linalg.norm(nx - y, axis=1) / np.linalg.norm(y, axis=1)
    out["29_vs_norm28"] = {"cos_min": float(cos.min()), "rel_med": float(np.median(rel)), "rel_max": float(rel.max())}
    print("29 vs norm(28)", out["29_vs_norm28"])
except Exception as e:  # noqa
    print("norm check skipped:", repr(e))
json.dump(out, open(HERE / "vera_gate.json", "w"), indent=1)
