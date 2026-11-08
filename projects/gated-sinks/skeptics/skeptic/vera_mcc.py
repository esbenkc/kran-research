"""Vera: own TopK SAE + recovery score (activation MCC), written from PREREG.md D1 only.

recovery = mean_j max_i Pearson(f_i, z_j) over scoring positions; dead/constant features count 0.
SAE: TopK (Gao et al. 2406.04093), width 256, k 8; inputs centred and scaled so mean ||x||^2 = d.
Own choices (not read from build): Adam lr 1e-3, batch 1024, `steps` steps, decoder columns unit norm,
encoder init = decoder^T, pre-encoder bias = data mean (zero after centring).
"""
import numpy as np
import torch

def corr_matrix(F, Z):
    """(m, L) Pearson matrix; constant columns give 0."""
    F = F.astype(np.float64)
    Z = Z.astype(np.float64)
    Fc = F - F.mean(0)
    Zc = Z - Z.mean(0)
    fs = np.sqrt((Fc**2).sum(0))
    zs = np.sqrt((Zc**2).sum(0))
    C = (Fc.T @ Zc) / np.outer(np.where(fs > 0, fs, 1.0), np.where(zs > 0, zs, 1.0))
    C[fs == 0, :] = 0.0
    C[:, zs == 0] = 0.0
    return C


def mcc(F, Z):
    """F (N, m) feature activations, Z (N, L) 0/1 latents -> per-latent max Pearson (L,)."""
    return corr_matrix(F, Z).max(0)


def mcc_hungarian(F, Z):
    """One-to-one matching (Kasper #17187): per-latent corr of its assigned feature (L,)."""
    from scipy.optimize import linear_sum_assignment
    C = corr_matrix(F, Z)
    r, c = linear_sum_assignment(-C)
    out = np.zeros(C.shape[1])
    out[c] = C[r, c]
    return out


def normalise(X, ref):
    mu = ref.mean(0)
    s = np.sqrt(((ref - mu) ** 2).sum(1).mean() / ref.shape[1])
    return (X - mu) / s


def train_topk(X, width=256, k=8, steps=4000, batch=1024, lr=1e-3, seed=0):
    g = torch.Generator().manual_seed(seed)
    d = X.shape[1]
    Xt = torch.tensor(X, dtype=torch.float32)
    W_dec = torch.randn(width, d, generator=g)
    W_dec = W_dec / W_dec.norm(dim=1, keepdim=True)
    W_dec = torch.nn.Parameter(W_dec)
    W_enc = torch.nn.Parameter(W_dec.detach().clone().T.contiguous())
    b_enc = torch.nn.Parameter(torch.zeros(width))
    b_dec = torch.nn.Parameter(torch.zeros(d))
    opt = torch.optim.Adam([W_dec, W_enc, b_enc, b_dec], lr=lr)
    n = Xt.shape[0]
    for _ in range(steps):
        idx = torch.randint(0, n, (batch,), generator=g)
        x = Xt[idx]
        pre = (x - b_dec) @ W_enc + b_enc
        v, i = pre.topk(k, dim=1)
        f = torch.zeros_like(pre).scatter(1, i, torch.relu(v))
        xh = f @ W_dec + b_dec
        loss = ((xh - x) ** 2).sum(1).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        with torch.no_grad():
            W_dec /= W_dec.norm(dim=1, keepdim=True)
    def encode(Xn):
        with torch.no_grad():
            xt = torch.tensor(Xn, dtype=torch.float32)
            pre = (xt - b_dec) @ W_enc + b_enc
            v, i = pre.topk(k, dim=1)
            return torch.zeros_like(pre).scatter(1, i, torch.relu(v)).numpy()
    with torch.no_grad():
        fvu = float(loss.item() / ((Xt - Xt.mean(0)) ** 2).sum(1).mean().item())
    return encode, W_dec.detach().numpy(), fvu
