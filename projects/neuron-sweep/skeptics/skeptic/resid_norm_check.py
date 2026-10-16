"""Fairness check of the random-direction control in RESIDUAL space.

The control is matched in neuron space (||delta|| = amax). The model only sees W_down @ delta.
Compare the residual write of a single neuron (amax_i * ||W_down[:, i]||) with a random unit direction
(amax_j * ||W_down u||), per layer. Reads only down_proj weights of Qwen3-0.6B + data/amax.npy.
"""
import glob, os, json
import numpy as np, torch
from safetensors import safe_open

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, '..', '..', 'data') + '/'
f = glob.glob(os.path.expanduser('~/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/*/model.safetensors'))[0]
amax = np.load(D + 'amax.npy')
print('amax', amax.shape)
g = torch.Generator().manual_seed(0)
res = {}
with safe_open(f, 'pt') as s:
    for L in range(28):
        W = s.get_tensor(f'model.layers.{L}.mlp.down_proj.weight').float()  # (1024, 3072)
        col = W.norm(dim=0).numpy()
        u = torch.randn(3072, 256, generator=g); u /= u.norm(dim=0, keepdim=True)
        ru = (W @ u).norm(dim=0).numpy()
        a = np.asarray(amax[L], float)
        neuron_write = a * col
        rng = np.random.default_rng(L)
        dir_write = a[rng.integers(0, len(a), 256)] * ru
        res[L] = dict(col_mean=float(col.mean()), col_rms=float(np.sqrt((col ** 2).mean())),
                      rand_dir_mean=float(ru.mean()),
                      neuron_write_median=float(np.median(neuron_write)),
                      dir_write_median=float(np.median(dir_write)),
                      ratio_median=float(np.median(dir_write) / np.median(neuron_write)))
        print(f"L{L:2d} ||col|| mean {col.mean():.3f} rms {res[L]['col_rms']:.3f}  ||W u|| {ru.mean():.3f}  "
              f"median write neuron {np.median(neuron_write):8.2f} dir {np.median(dir_write):8.2f}  "
              f"ratio {res[L]['ratio_median']:.2f}")
json.dump(res, open(os.path.join(HERE, 'resid_norm_check.json'), 'w'), indent=1)
