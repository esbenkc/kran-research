"""TopK sparse autoencoder (Gao et al. 2024, arXiv 2406.04093) in MLX.

encode:  f = TopK_k( ReLU( (x - b_dec) W_enc + b_enc ) )
decode:  x_hat = f W_dec + b_dec, every row of W_dec kept at unit norm
loss:    mean ||x - x_hat||^2   (no L1 term: sparsity is fixed by k)
Inputs are standardised first: x <- (x - mean) / s, with s the RMS norm of the centred training
data, so one learning rate fits every model.
"""
from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np


class TopKSAE(nn.Module):
    def __init__(self, d: int, m: int, k: int):
        super().__init__()
        self.k = k
        w = mx.random.normal((m, d))
        w = w / mx.linalg.norm(w, axis=1, keepdims=True)
        self.W_dec = w
        self.W_enc = w.T * 1.0
        self.b_enc = mx.zeros((m,))
        self.b_dec = mx.zeros((d,))

    def pre(self, x):
        return nn.relu((x - self.b_dec) @ self.W_enc + self.b_enc)

    def encode(self, x):
        a = self.pre(x)
        thr = mx.sort(a, axis=-1)[:, -self.k][:, None]
        return mx.where(a >= thr, a, 0.0)

    def __call__(self, x):
        return self.encode(x) @ self.W_dec + self.b_dec


def train_sae(x: np.ndarray, m: int, k: int, seed: int, steps: int = 3000, batch: int = 4096,
              lr: float = 1e-3):
    """x: standardised [N, d] float32. Returns (sae, history)."""
    mx.random.seed(seed)
    rng = np.random.default_rng(seed)
    sae = TopKSAE(x.shape[1], m, k)
    mx.eval(sae.parameters())
    opt = optim.Adam(learning_rate=lr)

    def loss_fn(model, xb):
        return mx.mean(mx.sum((model(xb) - xb) ** 2, axis=-1))

    lg = nn.value_and_grad(sae, loss_fn)
    state = [sae.state, opt.state]

    def _step(xb):
        loss, g = lg(sae, xb)
        opt.update(sae, g)
        sae.W_dec = sae.W_dec / mx.linalg.norm(sae.W_dec, axis=1, keepdims=True)
        return loss

    step = mx.compile(_step, inputs=state, outputs=state)
    xm = mx.array(x)
    hist = []
    for it in range(steps):
        idx = mx.array(rng.integers(0, len(x), batch))
        loss = step(xm[idx])
        mx.eval(state)
        if (it + 1) % 500 == 0:
            hist.append({"step": it + 1, "mse": float(loss.item())})
    return sae, hist


def encode_np(sae: TopKSAE, x: np.ndarray, bs: int = 16384) -> np.ndarray:
    out = []
    for i in range(0, len(x), bs):
        f = sae.encode(mx.array(x[i:i + bs]))
        mx.eval(f)
        out.append(np.array(f))
    return np.concatenate(out)


def fvu(sae: TopKSAE, x: np.ndarray) -> float:
    """Fraction of variance unexplained on x (standardised, so total variance ~ 1 per row)."""
    xh = []
    for i in range(0, len(x), 16384):
        r = sae(mx.array(x[i:i + 16384]))
        mx.eval(r)
        xh.append(np.array(r))
    xh = np.concatenate(xh)
    return float(((x - xh) ** 2).sum() / ((x - x.mean(0)) ** 2).sum())
