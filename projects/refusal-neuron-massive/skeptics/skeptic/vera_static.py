"""Vera: static weight facts for L13 down_proj (no forward pass).
Energy share of each column in a given set of residual coords.
Usage: python vera_static.py [c1 c2 c3]   (massive coords; default: none -> print column top coords)
"""
import sys, glob, json, numpy as np
from safetensors import safe_open

snap = glob.glob('/Users/esben/.cache/huggingface/hub/models--Qwen--Qwen3-1.7B/snapshots/*/')[0]
idx = json.load(open(snap + 'model.safetensors.index.json'))['weight_map']
L, I = 13, 3270
key = f'model.layers.{L}.mlp.down_proj.weight'
with safe_open(snap + idx[key], 'pt') as f:
    W = f.get_tensor(key).float().numpy()  # [2048, 6144]
print('W_down', W.shape)
w = W[:, I]
print(f'||w_{I}|| = {np.linalg.norm(w):.4f}; median col norm {np.median(np.linalg.norm(W, axis=0)):.4f}')
top = np.argsort(-np.abs(w))[:10]
print('top coords of w:', [(int(c), round(float(w[c]), 4)) for c in top])
coords = [int(c) for c in sys.argv[1:]]
if coords:
    E = (W[coords] ** 2).sum(0) / (W ** 2).sum(0)
    print(f'energy share in {coords}: gate {E[I]:.5f}; layer median {np.median(E):.5f}; '
          f'p99 {np.percentile(E, 99):.5f}; max {E.max():.5f} (neuron {int(E.argmax())}); '
          f'gate rank {int((E > E[I]).sum()) + 1}/{len(E)}; chance 3/2048={3/2048:.5f}')
    np.save('vera_share_L13.npy', E)
