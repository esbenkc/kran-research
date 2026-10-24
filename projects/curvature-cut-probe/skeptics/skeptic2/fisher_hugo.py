"""Hugo (skeptic2) independent output-Fisher tools. Own code; imports nothing from build.

Post-norm coordinates: z = gamma * h / rms(h); logits = W z with W = tied embedding (V x d).
Output Fisher at one state: F = W^T (diag p - p p^T) W, p = softmax(W z).
Second-order floor to move score s = u.z + b by D:  KL >= D^2 / (2 u^T F^+ u).
Regularised as in the frozen plan: u^T (F + eps I)^-1 u, eps = 1e-8 * lambda_max(F).
"""
import glob
import numpy as np
from safetensors import safe_open

SNAP = glob.glob('/Users/esben/.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/*/model.safetensors')[0]


def load_W(dtype=np.float32):
    with safe_open(SNAP, 'pt') as s:
        W = s.get_tensor('model.embed_tokens.weight').float().numpy().astype(dtype)
        g = s.get_tensor('model.norm.weight').float().numpy().astype(np.float64)
    return W, g


def softmax_rows(L):
    L = L - L.max(-1, keepdims=True)
    E = np.exp(L)
    return E / E.sum(-1, keepdims=True)


def logits64(W, z, chunk=16384):
    """W z in float64 (W float32 storage)."""
    out = np.empty(W.shape[0])
    for a in range(0, W.shape[0], chunk):
        out[a:a + chunk] = W[a:a + chunk].astype(np.float64) @ z
    return out


def fisher_full(W, p, p_drop=1e-17, chunk=16384):
    """Exact F = W^T diag(p) W - m m^T, float64 accumulation. W may be float32.
    Tokens with p < p_drop are skipped: their total norm contribution is <= V*p_drop*max||w_v||^2
    (~2e-12 here), i.e. ~1e-3 of the frozen eps (1e-8 * lambda_max ~ 1e-9)."""
    keep = np.nonzero(p >= p_drop)[0]
    d = W.shape[1]
    F = np.zeros((d, d))
    m = np.zeros(d)
    for a in range(0, len(keep), chunk):
        idx = keep[a:a + chunk]
        Wc = W[idx].astype(np.float64)
        pc = p[idx]
        m += Wc.T @ pc
        Ws = Wc * np.sqrt(pc)[:, None]
        F += Ws.T @ Ws
    F -= np.outer(m, m)
    return 0.5 * (F + F.T), len(keep)


def quad_inv(F, U, eps_rel=(1e-8,)):
    """u^T (F + eps I)^-1 u for each column of U and each eps_rel.
    Returns array (len(eps), ncols), lambda_max, eigvals, eigen-coords of U."""
    lam, V = np.linalg.eigh(F)
    lam = np.clip(lam, 0.0, None)
    lmax = lam[-1]
    C = V.T @ U
    out = np.array([((C ** 2) / (lam + e * lmax)[:, None]).sum(0) for e in eps_rel])
    return out, lmax, lam, C


def flat_share(lam, C, cut_rel=1e-6):
    """share of ||u||^2 on eigvals below cut_rel * lambda_max."""
    m = lam < cut_rel * lam[-1]
    return (C[m] ** 2).sum(0) / (C ** 2).sum(0)
