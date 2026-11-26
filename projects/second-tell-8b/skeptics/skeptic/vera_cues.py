# Rebuild u_A, u_B per seed from PREREG lines 89-92 and check orthogonality (raw + gamma-weighted) and the logged
# cosines in the run JSONs. gamma = Qwen3-8B model.layers.34.input_layernorm.weight from the local HF cache.
import json, glob, os, sys, numpy as np
from safetensors import safe_open
snap = glob.glob(os.path.expanduser('~/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/*'))[0]
idx = json.load(open(os.path.join(snap, 'model.safetensors.index.json')))['weight_map']
key = 'model.layers.34.input_layernorm.weight'
with safe_open(os.path.join(snap, idx[key]), 'pt') as f:
    g = f.get_tensor(key).float().numpy().astype(np.float64)
print('gamma min/median/max %.3f %.3f %.3f  PR(gamma^2) %.1f' % (g.min(), np.median(g), g.max(), (g**2).sum()**2 / (g**4).sum()))
res = {}
for s in range(5, 10):
    v = np.random.default_rng(1126000 + s).standard_normal((6, 4096))
    uA = v[:3] / np.linalg.norm(v[:3], axis=1, keepdims=True)
    uB = []
    for p in range(3):
        Q, _ = np.linalg.qr(np.stack([uA[p], g**2 * uA[p]], 1))
        b = v[3 + p] - Q @ (Q.T @ v[3 + p]); uB.append(b / np.linalg.norm(b))
    uB = np.array(uB)
    raw = [float(uA[p] @ uB[p]) for p in range(3)]
    gam = [float((g * uA[p]) @ (g * uB[p]) / np.linalg.norm(g * uA[p]) / np.linalg.norm(g * uB[p])) for p in range(3)]
    cross = float(np.abs(uA @ uB.T).max())
    res[s] = dict(raw=raw, gamma=gam, cross_max=cross)
    print(s, 'raw', ['%.1e' % x for x in raw], 'gamma', ['%.1e' % x for x in gam], 'max |cos| any pair %.4f' % cross)
d = sys.argv[1] if len(sys.argv) > 1 else None
if d:
    for f in sorted(glob.glob(os.path.join(d, 'main_s*.json'))):
        j = json.load(open(f)); c = j['cue'].get('cos_uA_uB', {})
        print(os.path.basename(f), 'logged raw', c.get('raw'), 'gamma', c.get('gamma'), 'raw_max_any_row', c.get('raw_max_any_row'))
