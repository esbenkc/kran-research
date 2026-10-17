"""Nadia shared helpers: raw weights, own numpy forward, sampler, rounding, gauge."""
import json, sys
import numpy as np
import torch

N = int(float(sys.argv[1])) if len(sys.argv) > 1 else 1_000_000
sd = torch.load('../m16_10.pth', map_location='cpu')
Wi = sd['rnn.weight_ih_l0'].double().numpy()[:, 0]      # (16,)
Wh = sd['rnn.weight_hh_l0'].double().numpy()            # (16,16)
Wo = sd['linear.weight'].double().numpy()               # (10,16)
assert Wi.size + Wh.size + Wo.size == 432


def fwd(Wi, Wh, Wo, X, bs=200_000):
    out = np.empty(len(X), dtype=np.int64)
    for a in range(0, len(X), bs):
        x = X[a:a + bs]
        h = np.zeros((len(x), 16))
        for t in range(10):
            h = np.maximum(0.0, x[:, t:t + 1] * Wi[None] + h @ Wh.T)
        out[a:a + bs] = (h @ Wo.T).argmax(1)
    return out


def sample(seed, n=N):
    return np.random.default_rng(seed).standard_normal((n, 10))


def round_sig(W, d):
    W = np.asarray(W, float)
    out = np.zeros_like(W)
    nz = W != 0
    e = np.floor(np.log10(np.abs(W[nz])))
    out[nz] = np.round(W[nz] / 10 ** (e - d + 1)) * 10 ** (e - d + 1)
    return out


def gauge(Wi, Wh, Wo):
    # ReLU is positively homogeneous: h_i -> h_i / |Wi_i| leaves the argmax unchanged.
    s = 1 / np.abs(Wi)
    return np.sign(Wi), Wh * s[:, None] / s[None, :], Wo / s[None, :]
