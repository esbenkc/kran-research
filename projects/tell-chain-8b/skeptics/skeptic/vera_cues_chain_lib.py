# Same construction as vera_cues_chain.py (my code), as a function for vera_fwd.py.
import json, glob, os, numpy as np
from safetensors import safe_open
_G = None
def gamma():
    global _G
    if _G is None:
        snap = glob.glob(os.path.expanduser('~/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/*'))[0]
        wm = json.load(open(os.path.join(snap, 'model.safetensors.index.json')))['weight_map']
        key = 'model.layers.34.input_layernorm.weight'
        with safe_open(os.path.join(snap, wm[key]), 'pt') as f:
            _G = f.get_tensor(key).float().numpy().astype(np.float64)
    return _G
def build_u(s):
    g = gamma(); v = np.random.default_rng(1126000 + s).standard_normal((15, 4096)); U = {}
    for ci, c in enumerate('ABCDE'):
        rows = []
        for p in range(3):
            x = v[3 * ci + p].copy()
            if ci:
                M = np.stack(sum([[U[pc][p], g ** 2 * U[pc][p]] for pc in 'ABCDE'[:ci]], []), 1)
                Q, _ = np.linalg.qr(M); x = x - Q @ (Q.T @ x)
            rows.append(x / np.linalg.norm(x))
        U[c] = np.array(rows)
    return U
