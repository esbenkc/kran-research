# Rebuild u_A..u_E per seed from PREREG (stamped 92fa7c588, 'Cue directions') with my own code; check orthogonality
# (raw and gamma-weighted) and compare with the cue hashes the run logged (run_meta.json cues[s][cue].u_sha256).
import json, glob, os, sys, hashlib, numpy as np
from safetensors import safe_open
snap = glob.glob(os.path.expanduser('~/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/*'))[0]
wm = json.load(open(os.path.join(snap, 'model.safetensors.index.json')))['weight_map']
key = 'model.layers.34.input_layernorm.weight'
with safe_open(os.path.join(snap, wm[key]), 'pt') as f:
    g = f.get_tensor(key).float().numpy().astype(np.float64)
meta = json.load(open(sys.argv[1])) if len(sys.argv) > 1 else None
worst_raw = worst_gam = 0.0; hash_hits = {}
for s in range(5, 10):
    v = np.random.default_rng(1126000 + s).standard_normal((15, 4096))
    U = {}
    for ci, c in enumerate('ABCDE'):
        rows = []
        for p in range(3):
            x = v[3 * ci + p].copy()
            if ci:
                prev = [U[pc][p] for pc in 'ABCDE'[:ci]]
                M = np.stack(sum([[u, g ** 2 * u] for u in prev], []), 1)
                Q, _ = np.linalg.qr(M)
                x = x - Q @ (Q.T @ x)
            rows.append(x / np.linalg.norm(x))
        U[c] = np.array(rows)
    for a in 'ABCDE':
        for b in 'ABCDE':
            if a < b:
                for p in range(3):
                    worst_raw = max(worst_raw, abs(U[a][p] @ U[b][p]))
                    ga, gb = g * U[a][p], g * U[b][p]
                    worst_gam = max(worst_gam, abs(ga @ gb) / np.linalg.norm(ga) / np.linalg.norm(gb))
    if meta is not None:
        for c in 'ABCDE':
            want = meta['cues'][str(s)][c]['u_sha256']
            cands = {'f64': U[c].astype(np.float64).tobytes(), 'f32': U[c].astype(np.float32).tobytes()}
            hit = [k for k, bts in cands.items() if hashlib.sha256(bts).hexdigest() == want]
            hash_hits[f'{s}{c}'] = hit[0] if hit else None
print(f'max |cos| same row across cue pairs: raw {worst_raw:.2e}, gamma-weighted {worst_gam:.2e}')
if meta is not None:
    n = sum(v is not None for v in hash_hits.values())
    print(f'logged u_sha256 reproduced by my rebuild: {n}/{len(hash_hits)}', {k: v for k, v in hash_hits.items() if v is None} or '')
