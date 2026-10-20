"""Vera: independent check of the post's megaprobe equation on build's saved fit (data/fits/final_mega.pt, pca_clean.pt):
score = sum_{t,l} softmax_{t,l}(q_l.z_tl + c_l) (w_l.z_tl) + b, z_tl = scale_l * V_l^T (x_tl - mu_l) (scale = 1/sd: whitening).
float64 numpy on 40 random statements of the clean model vs data/scores.npz clean__mega.
    cd kran-research && uv run python ../swarm-theory-1020-megaprobe-after/skeptic/vera_eqcheck.py
"""
import numpy as np, torch
P = 'projects/megaprobe-after/'
m = torch.load(P + 'data/fits/final_mega.pt', map_location='cpu', weights_only=False)['P']
pc = torch.load(P + 'data/fits/pca_clean.pt', map_location='cpu', weights_only=False)
q, c, w, b = [m[k].double().numpy() for k in ['q', 'c', 'w', 'b']]
mu, V, sc = [pc[k].double().numpy() for k in ['mu', 'V', 'scale']]
K = q.shape[1]
off = np.load(P + 'data/acts/clean/off.npy'); f = np.load(P + 'data/scores.npz')
t = np.load(P + 'data/acts/clean/tok.npy', mmap_mode='r')
idx = np.random.default_rng(0).choice(2992, 40, replace=False); d = []
for i in idx:
    x = np.asarray(t[off[i]:off[i + 1], 1:29], np.float64)
    z = np.einsum('tld,ldk->tlk', x - mu[None], V[:, :, :K]) * sc[None, :, :K]
    a = np.einsum('tlk,lk->tl', z, q) + c[None]; a = np.exp(a - a.max()); a /= a.sum()
    d.append((a * np.einsum('tlk,lk->tl', z, w)).sum() + b - f['clean__mega'][i])
print('K', K, 'max |formula - saved score| over 40 statements:', np.abs(d).max(), '(saved scores used fp16 z; clean score sd 3.84, so rel. error < 5e-4)')
